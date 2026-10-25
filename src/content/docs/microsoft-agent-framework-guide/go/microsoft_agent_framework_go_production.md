---
title: "Microsoft Agent Framework (Go) — Production"
description: "Running Go agents and workflows in production: timeouts at every layer, retries, cancellation, concurrency limits and per-session locking, persisting sessions, credentials and secrets, limits on the tool loop, workflow durability, a deployable service with health checks and graceful shutdown, a container image and a checklist."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Production"
  order: 15
---

This page collects what changes when an agent leaves your laptop. Most of it is ordinary Go service engineering, because the SDK leans on the standard library: every call takes a `context.Context`, configuration is plain structs, and hosting is an `http.Handler`. Where a topic has its own page, this one summarises it and links there.

The programs were compiled with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%% and run against a local stand-in for the OpenAI API (`OPENAI_BASE_URL`), so their model replies are not real output.

## Timeouts at every layer

An agent run is a loop: model call, tool calls, model call, and so on. Each layer has its own limit, and you want all of them:

| Layer | Set with | Covers |
| --- | --- | --- |
| HTTP request to the model | `option.WithRequestTimeout(d)` on the OpenAI or Anthropic client | One attempt; the client's own retries each get a fresh timeout |
| The whole run | `context.WithTimeout` on the context passed to `Run` / `RunText` | Every model and tool call in the loop |
| One tool call | A `FunctionInvocationMiddleware` that derives a shorter context | A slow tool, without failing the run |
| The tool loop | `toolautocall.Config.MaximumIterationsPerRequest` (40 by default) | Runaway tool use |
| A workflow run | The context passed to `Run` / `RunStreaming`, or `StreamingRun.CancelRun` | Every superstep; there is no superstep limit |
| Your HTTP server | `http.Server{ReadHeaderTimeout, ReadTimeout, IdleTimeout}` and the request context | Slow clients, and disconnects (which cancel the run) |

`WriteTimeout` on an `http.Server` cuts off long streaming responses, so leave it at zero for SSE endpoints and rely on the run's own deadline instead.

A per-tool timeout is a few lines of function middleware. When it fires, the tool loop reports the failed call to the model, which carries on without the result:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// toolTimeout gives every tool call its own deadline.
func toolTimeout(d time.Duration) agent.FunctionInvocationMiddleware {
	return func(next func(context.Context, *agent.FunctionInvocationContext) (any, error),
		ctx context.Context, inv *agent.FunctionInvocationContext) (any, error) {
		ctx, cancel := context.WithTimeout(ctx, d)
		defer cancel()
		result, err := next(ctx, inv)
		if errors.Is(err, context.DeadlineExceeded) {
			log.Printf("tool %s timed out after %v", inv.Function.Name(), d)
		}
		return result, err
	}
}

type lookupIn struct {
	Account string `json:"account" jsonschema:"the account number"`
}

func main() {
	slowLookup := functool.MustNew(functool.Config{Name: "lookup_account", Description: "Look up an account."},
		func(ctx context.Context, in lookupIn) (string, error) {
			select {
			case <-time.After(5 * time.Second): // a backend having a bad day
				return "account " + in.Account + " is active", nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		})

	client := openai.NewClient(
		option.WithRequestTimeout(30*time.Second), // per HTTP attempt
		option.WithMaxRetries(2),
	)
	support := openaiprovider.NewAgent(client, openaiprovider.AgentConfig{
		Model: "%%model=openai-small%%",
		Config: agent.Config{
			Name:                "support",
			Tools:               []tool.Tool{slowLookup},
			FunctionMiddlewares: []agent.FunctionInvocationMiddleware{toolTimeout(200 * time.Millisecond)},
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute) // the whole run
	defer cancel()
	start := time.Now()
	resp, err := support.RunText(ctx, "Is account 1234 active?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("answered in under a second: %t\n", time.Since(start) < time.Second)
	fmt.Println(resp)
}
```

Output against the stand-in (plus the timeout line logged to standard error):

```text
answered in under a second: true
Here is what the tool returned: Error: context deadline exceeded
```

The stand-in quotes the tool result back, which shows what the model was told: the error's text. That is true of every tool error in v%%version=github.com/microsoft/agent-framework-go%%, as the next section explains.

A tool that ignores its context cannot be stopped this way: the middleware returns, but the goroutine running the tool keeps going. Pass the context into every network and database call your tools make.

## Retries

The vendor clients already retry transient HTTP failures (connection errors, 408, 409, 429 and 5xx; two retries by default for OpenAI and Anthropic), which covers most cases. Tune them with `option.WithMaxRetries` and `option.WithMaxRetryDelay`. A retry at the agent level re-runs the whole tool loop, so only do it for failures that happen before any output, and keep tools idempotent. The [middleware page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/#retries-and-timeouts) has a run-level retry middleware that does exactly that.

Two kinds of failure should **not** be retried blindly: a context that was cancelled or timed out (`errors.Is(err, context.Canceled)` or `context.DeadlineExceeded`), and 4xx errors other than 408, 409 and 429, which will fail the same way again. Match provider errors with the vendor's error type (`*openai.Error`, `*anthropic.Error`); the framework passes them through unwrapped.

Workflows have no retry at all: a failing executor ends the run. Retry inside the executor, around the call that can fail, or resume from the last checkpoint.

## Cancellation

Cancellation follows the context everywhere: provider HTTP calls, tool calls (through the context the tool receives), MCP calls, workflow executors (through `*workflow.Context`), and A2A and AG-UI handlers (through the request context, so a client that disconnects cancels its run). There is no separate cancel API on an agent. For a workflow, `StreamingRun.CancelRun()` stops a run you are watching.

Stopping early when *you* stop reading is also cancellation: breaking out of `for update, err := range a.Run(...)` ends the stream, and the provider call is abandoned. A stream is a Go iterator, so it does no work unless you range over it, and `Collect` drains it for you.

## Concurrency

- **Agents** (`*agent.Agent`) are safe to share between goroutines. Build them once at start-up.
- **Sessions** are safe for concurrent access to their state, but a conversation must not run twice at once: two concurrent runs on one session interleave their history writes. Serialise runs per session, as below.
- **Tool calls** from one model response run one at a time unless `toolautocall.Config.AllowConcurrentInvocations` is set; then your tools must be safe for concurrent use.
- **Provider rate limits** are per deployment or key. Cap concurrent runs with a semaphore or worker pool rather than relying on 429 retries; the [middleware page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/#rate-limits-and-concurrency) shows a semaphore middleware.
- **Workflows** are owned by one runner at a time. Build one per run, or use `inproc.Concurrent` with shareable executors ([details](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/#running-one-workflow-concurrently)).

This program processes a batch with a bounded pool, sharing one agent, and serialises turns per conversation with a keyed lock:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"golang.org/x/sync/errgroup"
)

// sessionLocks hands out one mutex per conversation ID.
type sessionLocks struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

func (l *sessionLocks) lock(id string) func() {
	l.mu.Lock()
	m, ok := l.locks[id]
	if !ok {
		m = &sync.Mutex{}
		l.locks[id] = m
	}
	l.mu.Unlock()
	m.Lock()
	return m.Unlock
}

func main() {
	ctx := context.Background()
	summariser := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Summarise the ticket in one line.",
		Config:       agent.Config{Name: "summariser"},
	})

	tickets := []string{"printer jammed", "VPN drops hourly", "invoice missing", "laptop fan noisy",
		"cannot log in", "screen flickers", "email bounced", "disk almost full"}
	results := make([]string, len(tickets))

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(3) // at most three model calls in flight
	for i, ticket := range tickets {
		g.Go(func() error {
			resp, err := summariser.RunText(gctx, ticket).Collect()
			if err != nil {
				return fmt.Errorf("ticket %d: %w", i, err)
			}
			results[i] = resp.String()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(results), "tickets summarised; first:", results[0])

	// Two goroutines on one conversation take turns instead of interleaving.
	locks := &sessionLocks{locks: map[string]*sync.Mutex{}}
	session, err := summariser.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	var wg sync.WaitGroup
	for _, msg := range []string{"first message", "second message"} {
		wg.Go(func() {
			unlock := locks.lock("conversation-7")
			defer unlock()
			if _, err := summariser.RunText(ctx, msg, agent.WithSession(session)).Collect(); err != nil {
				log.Print(err)
			}
		})
	}
	wg.Wait()
	fmt.Println("both turns completed on one session")
}
```

Output against the stand-in:

```text
8 tickets summarised; first: Mock reply to: printer jammed
both turns completed on one session
```

A keyed lock only works inside one process. With several replicas, route a conversation to one replica (sticky sessions), or use a lease or row lock in the store that holds the session.

## Persisting sessions

An `*agent.Session` marshals to JSON. For a service, load it at the start of a request, run, and save it afterwards. Two things decide what the JSON holds:

- **Where the history lives.** With the default in-memory history provider and a provider that does not store conversations (Chat Completions, Anthropic, Gemini, or OpenAI Responses with `DisableStoreOutput: true`), the transcript is in the session itself. With the OpenAI Responses API and stored output (the `openaiprovider.NewAgent` default), the session mostly holds the service's response ID, and the history lives with OpenAI; it expires according to OpenAI's retention rules.
- **What else you put in it.** `session.Set` values and context-provider state are saved too.

The [sessions page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/#saving-and-restoring-sessions) covers the format and the history providers. In a service, add optimistic concurrency so that a lost update shows up as an error instead of a silently forgotten turn:

```go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sync"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

var ErrConflict = errors.New("session was changed by another request")

// sessionRepo stands in for a database table: id -> (version, JSON).
type sessionRepo struct {
	mu   sync.Mutex
	rows map[string]struct {
		version int
		data    []byte
	}
}

func (r *sessionRepo) Load(ctx context.Context, a *agent.Agent, id string) (*agent.Session, int, error) {
	r.mu.Lock()
	row, ok := r.rows[id]
	r.mu.Unlock()
	if !ok {
		s, err := a.CreateSession(ctx)
		return s, 0, err
	}
	var s agent.Session
	if err := json.Unmarshal(row.data, &s); err != nil {
		return nil, 0, err
	}
	return &s, row.version, nil
}

// Save writes only if nobody saved since we loaded (UPDATE ... WHERE version = ?).
func (r *sessionRepo) Save(id string, s *agent.Session, loadedVersion int) error {
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.rows[id].version != loadedVersion {
		return ErrConflict
	}
	r.rows[id] = struct {
		version int
		data    []byte
	}{loadedVersion + 1, data}
	return nil
}

func handleTurn(ctx context.Context, a *agent.Agent, repo *sessionRepo, id, text string) (string, error) {
	session, version, err := repo.Load(ctx, a, id)
	if err != nil {
		return "", err
	}
	resp, err := a.RunText(ctx, text, agent.WithSession(session)).Collect()
	if err != nil {
		return "", err
	}
	return resp.String(), repo.Save(id, session, version)
}

func main() {
	ctx := context.Background()
	assistant := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai-small%%",
		Config: agent.Config{Name: "assistant"},
	})
	repo := &sessionRepo{rows: map[string]struct {
		version int
		data    []byte
	}{}}

	for _, text := range []string{"My order number is 5521.", "What was my order number?"} {
		reply, err := handleTurn(ctx, assistant, repo, "user-42", text)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(reply)
	}
	s, v, _ := repo.Load(ctx, assistant, "user-42")
	history, _ := s.InMemoryHistory("")
	fmt.Printf("version %d, %d messages stored\n", v, len(history))

	// A stale writer is refused.
	if err := repo.Save("user-42", s, v-1); errors.Is(err, ErrConflict) {
		fmt.Println("stale save rejected")
	}
}
```

Output against the stand-in:

```text
Mock reply to: My order number is 5521.
Mock reply to: What was my order number?
version 2, 4 messages stored
stale save rejected
```

A session holds conversation content, so store it with the same care as any user data: encryption at rest, retention limits, and deletion on request. The A2A hosting layer has its own `SessionStore` interface for the same job; see [session stores and tenants](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_hosting_and_a2a/#session-stores-and-tenants).

## Credentials and secrets

- **Keys from the environment or a secret store**, never from source. `openai.NewClient()` reads `OPENAI_API_KEY` (and `OPENAI_BASE_URL`), and the Anthropic client reads `ANTHROPIC_API_KEY`. In Kubernetes mount them from a Secret; in Azure, read them from Key Vault at start-up.
- **Prefer managed identity on Azure.** The Foundry provider takes an `azcore.TokenCredential`, so `azidentity.NewDefaultAzureCredential(nil)` works on a laptop (Azure CLI login) and in production (managed or workload identity) with no keys at all. In production you can pin it with `azidentity.NewManagedIdentityCredential` so that it does not try developer credentials. Assign the narrowest role that works (for Foundry projects, "Azure AI User").
- **Never log secrets or payloads.** Keep `LogSensitiveData` off on agents and the tool loop, and keep workflow `EnableSensitiveData` off unless the tracing backend is approved for user data. Agent spans never carry content.
- **Tools run with your service's permissions.** Treat tool arguments as untrusted input, since they come from model output that a user can influence. Validate them (the `functool` schema helps), use allow-lists, and put destructive tools behind `tool.ApprovalRequiredFunc`.

```go
package main

import (
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/foundryprovider"
)

func main() {
	// Azure CLI on a laptop, managed or workload identity in production.
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		log.Fatal(err)
	}
	a := foundryprovider.NewAgent(
		os.Getenv("FOUNDRY_PROJECT_ENDPOINT"), // https://<resource>.services.ai.azure.com/api/projects/<project>
		cred,
		foundryprovider.ModelDeployment(os.Getenv("FOUNDRY_MODEL")),
		foundryprovider.AgentConfig{Config: agent.Config{Name: "assistant"}},
	)
	_ = a
}
```

This one was compiled but not run, as it needs an Azure subscription.

## Limits on the tool loop

| Setting (`toolautocall.Config`) | Default | Production advice |
| --- | --- | --- |
| `MaximumIterationsPerRequest` | 40 | Lower it (5 to 10 is plenty for most agents); when reached, the model must answer without tools |
| `MaximumConsecutiveErrorsPerRequest` | 3 | Keep it low so a broken tool fails the run instead of burning tokens |
| `IncludeDetailedErrors` | false | Keep false, but do not rely on it with the OpenAI adapters (below) |
| `TerminateOnUnknownCalls` | false | Set true if an unknown tool name should stop the run rather than be reported to the model |
| `AllowConcurrentInvocations` | false | Turn on only for tools that are safe in parallel |

**Tool error text reaches the model.** The tool loop replaces a failed call's result with `Error: Function failed.` unless `IncludeDetailedErrors` is set, but it also keeps the original error on the `FunctionResultContent`, and in v%%version=github.com/microsoft/agent-framework-go%% both OpenAI adapters (Responses and Chat Completions) send that error's text to the model instead of the generic result. A tool returning `pq: password authentication failed for user admin` puts exactly that in the conversation. Until this is fixed, sanitise tool errors yourself with function middleware, logging the real error and returning a bland one:

```go
func sanitiseToolErrors(logger *slog.Logger) agent.FunctionInvocationMiddleware {
	return func(next func(context.Context, *agent.FunctionInvocationContext) (any, error),
		ctx context.Context, inv *agent.FunctionInvocationContext) (any, error) {
		result, err := next(ctx, inv)
		if err != nil {
			logger.ErrorContext(ctx, "tool failed", "tool", inv.Function.Name(), "error", err)
			return nil, fmt.Errorf("%s is unavailable right now", inv.Function.Name())
		}
		return result, nil
	}
}
```

Set the loop limits through the provider config's `ToolAutoCall` field; the [tools page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#the-tool-loop) describes each. Limit the user's input too: cap request bodies (`http.MaxBytesReader`) and message counts before they reach the model, and set `max_tokens`-style limits through the provider's request options where the bill matters.

## Workflows in production

- **Checkpoint anything that waits for people.** Use `checkpoint.NewJSONManager` with a durable store. The file-system store locks its directory to one process, so for several replicas implement `checkpoint.Store` over your database ([how](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/#time-travel-and-custom-stores)).
- **Version the graph.** A checkpoint resumes only into the same topology and executor IDs. Treat a graph change like a schema migration: finish or drain old runs, or keep the old builder around under a version number.
- **Give hosted agents fixed IDs** (`agent.Config.ID`), because the executor ID of an `agentworkflow.New` node is derived from it.
- **Make steps idempotent.** A resumed run repeats the superstep that was in progress, including its model and tool calls.
- **Bound every run** with a context deadline, since loops have no superstep limit.
- **Watch for both failure events.** `ExecutorFailedEvent` and `ErrorEvent` are how failures arrive; `Run` returns a nil error for them.

## A deployable service

Putting it together: a small HTTP service with configuration from the environment, OpenTelemetry, liveness and readiness endpoints, request limits, and a graceful shutdown that lets in-flight runs finish:

```go
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolautocall"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/provider/otelprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type config struct {
	Addr     string
	Model    string
	RunLimit time.Duration
}

func loadConfig() config {
	limit, err := time.ParseDuration(cmp.Or(os.Getenv("RUN_TIMEOUT"), "90s"))
	if err != nil {
		limit = 90 * time.Second
	}
	return config{
		Addr:     cmp.Or(os.Getenv("ADDR"), ":8080"),
		Model:    cmp.Or(os.Getenv("MODEL"), "%%model=openai-small%%"),
		RunLimit: limit,
	}
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	cfg := loadConfig()

	assistant := openaiprovider.NewAgent(
		openai.NewClient(option.WithRequestTimeout(60*time.Second), option.WithMaxRetries(2)),
		openaiprovider.AgentConfig{
			Model:        cfg.Model,
			Instructions: "You are a concise internal help-desk assistant.",
			ToolAutoCall: &toolautocall.Config{MaximumIterationsPerRequest: new(8), Logger: logger},
			Config: agent.Config{
				Name:        "helpdesk",
				Logger:      logger,
				Middlewares: []agent.Middleware{otelprovider.NewMiddleware(otelprovider.MiddlewareConfig{SourceName: "helpdesk"})},
			},
		})

	var ready atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		if !ready.Load() {
			http.Error(w, "shutting down", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("POST /ask", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Question string `json:"question"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil || req.Question == "" {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), cfg.RunLimit)
		defer cancel()
		resp, err := assistant.RunText(ctx, req.Question).Collect()
		if err != nil {
			logger.ErrorContext(ctx, "run failed", "error", err)
			status := http.StatusBadGateway
			if errors.Is(err, context.DeadlineExceeded) {
				status = http.StatusGatewayTimeout
			}
			http.Error(w, "the assistant is unavailable", status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"answer": resp.String()})
	})

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}
	go func() {
		logger.Info("listening", "addr", cfg.Addr, "model", cfg.Model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	}()
	ready.Store(true)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()

	ready.Store(false)          // fail readiness so the load balancer stops sending traffic
	time.Sleep(2 * time.Second) // give it a moment to notice
	shutdown, cancel := context.WithTimeout(context.Background(), cfg.RunLimit)
	defer cancel()
	if err := srv.Shutdown(shutdown); err != nil { // waits for in-flight requests
		logger.Error("shutdown", "error", err)
	}
	logger.Info("stopped")
}
```

Run against the stand-in, `curl -s -X POST localhost:8080/ask -d '{"question":"How do I reset my VPN?"}'` returned `{"answer":"Mock reply to: How do I reset my VPN?"}`, `/healthz` and `/readyz` returned 204, and `SIGTERM` stopped it cleanly. Add the OTLP set-up from the [observability page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/#exporting-with-otlp) to export the spans the middleware creates.

### A container image

The SDK is pure Go with no cgo, so a static binary in a minimal image works:

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/helpdesk ./cmd/helpdesk

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/helpdesk /helpdesk
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/helpdesk"]
```

The distroless image includes CA certificates, which the HTTPS calls to model providers need. If you use the `shelltool` package, the image needs the shell and tools it runs, so use a fuller base image for that service (and think hard about whether a model should run shell commands in production).

In Kubernetes, point the liveness probe at `/healthz` and the readiness probe at `/readyz`, set `terminationGracePeriodSeconds` longer than your run timeout, and give the pod a workload identity instead of a key where the provider supports it.

## Checklist

- [ ] Exact module version pinned in `go.mod`; release notes read before upgrading (the SDK is 0.x and breaks between minor versions).
- [ ] A deadline on every run; per-attempt HTTP timeouts on the provider client; per-tool timeouts for slow tools.
- [ ] Tool loop limits lowered; `IncludeDetailedErrors` off; destructive tools behind approval.
- [ ] Concurrency capped below the provider's rate limit; one run at a time per session.
- [ ] Sessions persisted with versioning; retention and deletion handled; workflow checkpoints in a durable, shared store.
- [ ] Hosted agents given fixed IDs; workflow graphs versioned.
- [ ] Secrets from the environment or a vault; managed identity on Azure; `LogSensitiveData` and `EnableSensitiveData` off.
- [ ] OpenTelemetry exported, with traces joined to incoming requests; alerts on error rate and latency.
- [ ] Liveness and readiness endpoints; graceful shutdown longer than the run timeout.
- [ ] A2A and AG-UI endpoints behind authentication, with stores scoped to the caller's identity.
- [ ] Integration tests against a real deployment, in addition to the [fake-provider unit tests](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_testing/).

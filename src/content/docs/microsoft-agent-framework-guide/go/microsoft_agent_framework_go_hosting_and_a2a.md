---
title: "Microsoft Agent Framework (Go) — Hosting and A2A"
description: "Serving Go agents and workflows over HTTP: A2A servers and clients with a2aprovider (agent cards, sessions, tasks, polling, authentication, multi-tenant stores), AG-UI servers and clients, exposing an agent over MCP streamable HTTP, and a plain SSE endpoint; plus what the Go SDK does not host."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Hosting and A2A"
  order: 12
---

The Go SDK has no hosting framework of its own. Instead, it gives you `http.Handler`s for two agent protocols, and you mount them on whatever server you already run: `net/http`, a router such as chi, or a gateway in front.

| Surface | Serve an agent | Call a remote agent | Package |
| --- | --- | --- | --- |
| **A2A** (Agent2Agent) | `a2aprovider.NewHandler` with `a2asrv.NewJSONRPCHandler` / `NewRESTHandler` | `a2aprovider.NewAgentFromResolver`, `NewAgentFromCard`, `NewAgent` | `provider/a2aprovider`, built on `github.com/a2aproject/a2a-go/v2` |
| **AG-UI** (agent-to-frontend events over SSE) | `aguiprovider.NewJSONHTTPHandler` | `aguiprovider.NewAgent` | `provider/aguiprovider`, built on the AG-UI community Go SDK |
| **MCP** | `mcptool.AddTool` on an `mcp.Server`, served over stdio or streamable HTTP | `mcptool.Connect` and `ListTools` (tools, not agents) | `tool/mcptool`, built on `github.com/modelcontextprotocol/go-sdk` |
| Your own HTTP API | Any handler that calls `agent.Run` and streams | | `net/http` |

**Not available in Go:** an OpenAI-compatible Responses or Chat Completions server for your agents, DevUI, deployment to Foundry hosted agents, and a dependency-injection hosting layer like .NET's `Microsoft.Agents.AI.Hosting`. The [plain SSE endpoint](#your-own-http-endpoint) below is the usual substitute for the first.

Because a remote A2A or AG-UI agent is an ordinary `*agent.Agent`, it composes with everything else: middleware, sessions, `agenttool.New` to make it a tool, or `agentworkflow.New` to make it a workflow participant. And because a workflow can be wrapped as an agent with `agentworkflow.NewAgent`, every surface here can serve a whole workflow too.

The programs on this page were compiled with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%% and run. The hosted agents use `openaiprovider` pointed at a local stand-in for the OpenAI API (`OPENAI_BASE_URL`), so their replies are not real model output. The examples that need both ends start the server inside the same process with `net/http/httptest` or a listener on a random port, so they run as one program.

## A2A server

An A2A server publishes an **agent card** at `/.well-known/agent-card.json` and answers JSON-RPC and HTTP+JSON (REST) requests. `a2aprovider.NewHandler` adapts an `*agent.Agent` to a2a-go's `RequestHandler`; the a2a-go transport handlers turn that into `http.Handler`s:

```go
package main

import (
	"cmp"
	"context"
	"crypto/subtle"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

// requireBearer protects the RPC endpoints when A2A_TOKEN is set. The agent
// card stays public so that clients can discover the agent.
func requireBearer(token string, next http.Handler) http.Handler {
	if token == "" {
		return next
	}
	want := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), want) != 1 {
			http.Error(w, "unauthorised", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	addr := cmp.Or(os.Getenv("ADDR"), "127.0.0.1:5000")
	publicURL := cmp.Or(os.Getenv("PUBLIC_URL"), "http://"+addr) // what clients should call

	policy := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You answer questions about the returns policy in two sentences.",
		Config:       agent.Config{Name: "PolicyAgent", Description: "Answers returns-policy questions."},
	})

	card := &a2a.AgentCard{
		Name:               "PolicyAgent",
		Description:        "Answers questions about the returns policy.",
		Version:            "1.0.0",
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		Capabilities:       a2a.AgentCapabilities{Streaming: true},
		Skills: []a2a.AgentSkill{{
			ID:          "returns",
			Name:        "Returns policy",
			Description: "Explains what can be returned and how.",
			Tags:        []string{"policy", "returns"},
			Examples:    []string{"Can I return opened items?"},
		}},
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(publicURL, a2a.TransportProtocolJSONRPC),
			a2a.NewAgentInterface(publicURL, a2a.TransportProtocolHTTPJSON),
		},
	}

	handler := a2aprovider.NewHandler(policy,
		a2aprovider.ExecutorConfig{SessionStore: a2aprovider.NewInMemorySessionStore()},
		a2asrv.WithExtendedAgentCard(card),
	)

	token := os.Getenv("A2A_TOKEN")
	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle("POST /{$}", requireBearer(token, a2asrv.NewJSONRPCHandler(handler)))
	mux.Handle("/", requireBearer(token, a2asrv.NewRESTHandler(handler)))

	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()
	log.Printf("A2A agent at %s (auth: %t)", publicURL, strings.TrimSpace(token) != "")

	// Shut down cleanly on Ctrl+C or SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdown)
}
```

Points to note:

- **The card must name reachable URLs.** `SupportedInterfaces` is what clients use to choose a transport and an address, so behind a proxy set it to the public URL, not the listen address. List both transports unless you have a reason not to; the Go client prefers HTTP+JSON.
- **Routing.** JSON-RPC requests are `POST /`, the REST endpoints live under `/` too, and the card has its own path. `"POST /{$}"` matches only the root, so the REST handler gets everything else.
- **Sessions.** The executor keeps one `*agent.Session` per A2A **context ID** in its `SessionStore`, so a client that reuses a context gets a multi-turn conversation. With a nil store it creates a private in-memory one, which is lost on restart and not shared between replicas. See [session stores](#session-stores-and-tenants).
- **Authentication** is yours to add. The card can advertise security schemes (`SecuritySchemes`, `SecurityRequirements`), but enforcing them is ordinary HTTP middleware, as `requireBearer` shows.
- **Other options** are a2a-go's: `a2asrv.WithTaskStore`, `WithLogger`, `WithConcurrencyConfig`, `WithAgentInactivityTimeout`, `WithPushNotifications`, `WithCallInterceptors` and more, passed after the `ExecutorConfig`.

`ExecutorConfig` has two fields:

| Field | Default | Effect |
| --- | --- | --- |
| `SessionStore` | A private in-memory store | Where sessions live, keyed by context ID |
| `AgentRunMode` | `ReturnMessage()` | The response shape for new requests: `ReturnMessage()` collects the agent's output into one message; `ReturnTask()` creates a task (submitted, working, artifacts, completed); `ReturnTaskWhen(fn)` decides per request |

## A2A client

`NewAgentFromResolver` fetches the card from a base URL and builds an agent from it. The agent's name and description default to the card's. Use `NewAgentFromCard` when you already hold a trusted card, or `NewAgent` with your own `*a2aclient.Client` when you need to manage its lifetime. This client talks to the server above, sending a bearer token when `A2A_TOKEN` is set:

```go
package main

import (
	"cmp"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2aclient/agentcard"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
)

// bearer adds an Authorization header to every request.
type bearer struct {
	token string
	next  http.RoundTripper
}

func (b bearer) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("Authorization", "Bearer "+b.token)
	return b.next.RoundTrip(r)
}

func main() {
	ctx := context.Background()
	baseURL := cmp.Or(os.Getenv("A2A_URL"), "http://127.0.0.1:5000")

	var opts []a2aclient.FactoryOption
	if token := os.Getenv("A2A_TOKEN"); token != "" {
		hc := &http.Client{Transport: bearer{token: token, next: http.DefaultTransport}}
		opts = append(opts, a2aclient.WithJSONRPCTransport(hc), a2aclient.WithRESTTransport(hc))
	}

	// The resolver fetches the public card with its own HTTP client; the
	// options configure how the agent itself is called.
	policy, err := a2aprovider.NewAgentFromResolver(ctx, agentcard.DefaultResolver, baseURL,
		a2aprovider.AgentConfig{}, opts...)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("connected to %s: %s\n", policy.Name(), policy.Description())

	// One session = one A2A context = one conversation on the server.
	session, err := policy.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := policy.RunText(ctx, "Can I return opened items?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("reply 1:", resp)

	fmt.Print("reply 2 (streamed): ")
	for update, err := range policy.RunText(ctx, "And sale items?", agent.WithSession(session), agent.Stream(true)) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(update.String())
	}
	fmt.Println()
	fmt.Println("context ID kept in the session:", session.ServiceID() != "")
}
```

Output, with the server running and the same `A2A_TOKEN` on both sides:

```text
connected to PolicyAgent: Answers questions about the returns policy.
reply 1: Mock reply to: Can I return opened items?
reply 2 (streamed): Mock reply to: And sale items?
context ID kept in the session: true
```

Without the token, discovery still succeeds because the card is public, but the first run fails. The a2a-go client reports the 401 only as `server error`, so log the status in your middleware if you need to tell authentication failures apart.

Client-side options are a2a-go's `FactoryOption`s:

| Option | Use |
| --- | --- |
| `a2aclient.WithJSONRPCTransport(hc)`, `WithRESTTransport(hc)` | Supply the `*http.Client` (authentication, timeouts, proxies, mTLS) |
| `a2aclient.WithConfig(a2aclient.Config{PreferredTransports: ...})` | Choose the transport when the card offers several. The Go client prefers HTTP+JSON by default. |
| `a2aclient.WithCallInterceptors(...)` | Per-call hooks, for example to add headers or log |

A2A-specific run options:

| Option or helper | Effect |
| --- | --- |
| `a2aprovider.WithMetadata(map[string]any)` | Request-level metadata for one run; a hosted agent receives incoming metadata the same way |
| `a2aprovider.WithTaskID(id)` | At `CreateSession`, together with `agent.WithServiceID(contextID)`, resumes an existing task |
| `a2aprovider.TaskIDFromSession(s)`, `TaskStateFromSession(s)` | The current task ID and its last state |
| `agent.AllowBackgroundResponses(true)`, `agent.WithContinuationToken(t)` | Start a task without waiting, then poll or reconnect (see below) |

`NewAgentFromCard` treats the card as trusted. Use `NewAgentFromResolver` for discovery, and only connect to hosts you trust: the remote agent's replies are untrusted model output, like any other.

## Long-running tasks

With `AgentRunMode: a2aprovider.ReturnTask()` the server creates an A2A task for every request. A client that passes `agent.AllowBackgroundResponses(true)` gets control back at once, with a `ContinuationToken` on the response; it polls by running with `agent.WithContinuationToken` and no messages until the token comes back empty:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient/agentcard"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
)

// slowReporter stands in for an agent that takes a while.
func slowReporter() *agent.Agent {
	run := func(ctx context.Context, msgs []*message.Message, _ ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			select {
			case <-time.After(1500 * time.Millisecond):
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			}
			yield(&agent.ResponseUpdate{Role: message.RoleAssistant, MessageID: "report",
				Contents: message.Contents{&message.TextContent{Text: "report ready: " + msgs[len(msgs)-1].String()}}}, nil)
		}
	}
	return agent.New(agent.ProviderConfig{ProviderName: "offline", Run: run}, agent.Config{Name: "Reporter"})
}

func serve() string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}
	url := "http://" + ln.Addr().String()
	card := &a2a.AgentCard{
		Name: "Reporter", Description: "Builds sales reports.", Version: "1.0.0",
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		Capabilities: a2a.AgentCapabilities{Streaming: true},
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(url, a2a.TransportProtocolHTTPJSON),
			a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC),
		},
	}
	h := a2aprovider.NewHandler(slowReporter(), a2aprovider.ExecutorConfig{AgentRunMode: a2aprovider.ReturnTask()})
	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle("POST /{$}", a2asrv.NewJSONRPCHandler(h))
	mux.Handle("/", a2asrv.NewRESTHandler(h))
	go func() { _ = http.Serve(ln, mux) }()
	return url
}

func main() {
	ctx := context.Background()
	reporter, err := a2aprovider.NewAgentFromResolver(ctx, agentcard.DefaultResolver, serve(), a2aprovider.AgentConfig{})
	if err != nil {
		log.Fatal(err)
	}
	session, err := reporter.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := reporter.RunText(ctx, "Q3 sales by region", agent.WithSession(session),
		agent.AllowBackgroundResponses(true)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("returned at once, state:", a2aprovider.TaskStateFromSession(session))
	for resp.ContinuationToken != "" {
		time.Sleep(500 * time.Millisecond)
		resp, err = reporter.Run(ctx, nil, agent.WithSession(session),
			agent.WithContinuationToken(resp.ContinuationToken)).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("polled, state:", a2aprovider.TaskStateFromSession(session))
	}
	fmt.Println("result:", resp.String())
}
```

Output:

```text
returned at once, state: TASK_STATE_SUBMITTED
polled, state: TASK_STATE_WORKING
polled, state: TASK_STATE_WORKING
polled, state: TASK_STATE_COMPLETED
result: report ready: Q3 sales by region
```

A streamed run in task mode puts a continuation token on its first update. If the connection drops, run again with `agent.WithContinuationToken(token)`, `agent.Stream(true)` and `agent.WithServiceID(contextID)` (the update's `ConversationID`) to resubscribe to the task; the upstream sample `examples/02-agents/a2a/stream_reconnection` shows the whole loop.

**Gotcha: task mode and follow-up turns.** After a task completes, the next run on the same session sends the old task ID as a reference, and the Go server rejects it with `referenceTaskIds is not supported, an agent cannot resume from arbitrary prior task context`. With `ReturnTask()`, use a new session for each task. If you want both, use `ReturnTaskWhen` so that only long jobs become tasks, and keep conversational turns as messages:

```go
cfg := a2aprovider.ExecutorConfig{
	AgentRunMode: a2aprovider.ReturnTaskWhen(func(_ context.Context, ec *a2asrv.ExecutorContext) (bool, error) {
		m := ec.Message
		return m != nil && len(m.Parts) > 0 && strings.HasPrefix(m.Parts[0].Text(), "/report"), nil
	}),
}
```

A task that is waiting for input (`TASK_STATE_INPUT_REQUIRED`) is different: the client links the next message to that task directly, which is how a hosted agent's tool approval or follow-up question is answered over A2A.

## Session stores and tenants

`a2aprovider.SessionStore` has two methods, `Get(ctx, contextID)` and `Save(ctx, contextID, session)`. `Get` returns `nil, nil` for an unknown context, and both must be safe for concurrent calls. Sessions are JSON-serialisable, so any key-value store will do. This one keeps one file per context, which survives a restart:

```go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"sync"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

// fileSessionStore implements a2aprovider.SessionStore with one JSON file per
// A2A context ID.
type fileSessionStore struct {
	dir string
	mu  sync.Mutex
}

var safeID = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

func (s *fileSessionStore) path(contextID string) (string, error) {
	if !safeID.MatchString(contextID) { // context IDs come from clients
		return "", fmt.Errorf("invalid context ID %q", contextID)
	}
	return filepath.Join(s.dir, contextID+".json"), nil
}

func (s *fileSessionStore) Get(_ context.Context, contextID string) (*agent.Session, error) {
	p, err := s.path(contextID)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	data, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var session agent.Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, err
	}
	return &session, nil
}

func (s *fileSessionStore) Save(_ context.Context, contextID string, session *agent.Session) error {
	p, err := s.path(contextID)
	if err != nil {
		return err
	}
	data, err := json.Marshal(session)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, p) // atomic replace
}

func main() {
	ctx := context.Background()
	dir, err := os.MkdirTemp("", "a2a-sessions")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	// Chat Completions keeps the history in the session itself (the Responses
	// API would keep a service-side conversation ID instead).
	host := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai-small%%",
		Config: agent.Config{Name: "Notes"},
	})
	srv := httptest.NewUnstartedServer(nil)
	url := "http://" + srv.Listener.Addr().String()
	card := &a2a.AgentCard{Name: "Notes", Description: "Remembers things.", Version: "1.0.0",
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(url, a2a.TransportProtocolHTTPJSON)}}
	h := a2aprovider.NewHandler(host, a2aprovider.ExecutorConfig{SessionStore: &fileSessionStore{dir: dir}})
	srv.Config.Handler = a2asrv.NewRESTHandler(h)
	srv.Start()
	defer srv.Close()

	notes, err := a2aprovider.NewAgentFromCard(ctx, card, a2aprovider.AgentConfig{})
	if err != nil {
		log.Fatal(err)
	}
	session, err := notes.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, msg := range []string{"Remember: the door code is 4471", "What is the door code?"} {
		if _, err := notes.RunText(ctx, msg, agent.WithSession(session)).Collect(); err != nil {
			log.Fatal(err)
		}
	}

	files, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	fmt.Println("session files:", len(files))
	stored, err := (&fileSessionStore{dir: dir}).Get(ctx, session.ServiceID())
	if err != nil {
		log.Fatal(err)
	}
	history, err := stored.InMemoryHistory("")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("messages stored on the server:", len(history))
}
```

Output:

```text
session files: 1
messages stored on the server: 4
```

**Context IDs are not an authorisation boundary.** Any client that learns another client's context ID can continue that conversation. A server shared by several users or tenants must namespace both stores with a **trusted** identity (from your authentication middleware, not from the request body):

- Wrap your `SessionStore` so that it prefixes keys with the caller's identity taken from `ctx`.
- Wrap the task store with `a2aprovider.NewIsolationKeyScopedTaskStore(inner, a2aprovider.IsolationKeyScopedTaskStoreConfig{ResolveKey: ...})` and pass it with `a2asrv.WithTaskStore`. It rewrites task and context IDs per partition. With no key it fails closed unless `AllowUnscoped` is set.

```go
type tenantKey struct{}

store := a2aprovider.NewIsolationKeyScopedTaskStore(
	taskstore.NewInMemory(nil),
	a2aprovider.IsolationKeyScopedTaskStoreConfig{
		ResolveKey: func(ctx context.Context) (string, error) {
			tenant, _ := ctx.Value(tenantKey{}).(string) // set by your auth middleware
			return tenant, nil
		},
	},
)
handler := a2aprovider.NewHandler(hostAgent, cfg, a2asrv.WithTaskStore(store))
```

## Serving a workflow over A2A

Wrap the workflow with `agentworkflow.NewAgent` and host the result like any agent. The remote caller sees one agent and gets every participant's reply:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"net/http/httptest"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
)

func roleAgent(name string, reply func([]*message.Message) string) *agent.Agent {
	run := func(_ context.Context, msgs []*message.Message, _ ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			yield(&agent.ResponseUpdate{Role: message.RoleAssistant, MessageID: fmt.Sprint(name, len(msgs)),
				Contents: message.Contents{&message.TextContent{Text: reply(msgs)}}}, nil)
		}
	}
	return agent.New(agent.ProviderConfig{ProviderName: "offline", Run: run}, agent.Config{Name: name})
}

func main() {
	ctx := context.Background()
	translate := roleAgent("translator", func(m []*message.Message) string { return "FR: " + m[len(m)-1].String() })
	review := roleAgent("reviewer", func([]*message.Message) string { return "reviewed, no changes" })
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(translate, review).Build()
	if err != nil {
		log.Fatal(err)
	}
	pipeline, err := agentworkflow.NewAgent(wf, agentworkflow.AgentConfig{
		Config: agent.Config{Name: "TranslationPipeline"},
	})
	if err != nil {
		log.Fatal(err)
	}

	srv := httptest.NewUnstartedServer(nil)
	url := "http://" + srv.Listener.Addr().String()
	card := &a2a.AgentCard{Name: "TranslationPipeline", Description: "Translates and reviews.", Version: "1.0.0",
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(url, a2a.TransportProtocolHTTPJSON)}}
	srv.Config.Handler = a2asrv.NewRESTHandler(a2aprovider.NewHandler(pipeline, a2aprovider.ExecutorConfig{}))
	srv.Start()
	defer srv.Close()

	remote, err := a2aprovider.NewAgentFromCard(ctx, card, a2aprovider.AgentConfig{})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := remote.RunText(ctx, "Hello, world").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.String())
}
```

Output:

```text
FR: Hello, worldreviewed, no changes
```

In the default message mode the reply is one A2A message, so the participants' replies arrive joined, with no separator. Use task mode if the caller should see progress as the workflow runs.

## AG-UI

AG-UI is an event protocol between an agent backend and a user interface (CopilotKit and others speak it). The client POSTs a run input with the **whole** conversation and its own tool declarations; the server streams Server-Sent Events: run started, text message deltas, tool calls, state snapshots, run finished.

`aguiprovider.NewJSONHTTPHandler(agent, cfg)` is the server; it handles only `POST` and holds no conversation state between requests. `aguiprovider.NewAgent(sseClient, cfg)` is a Go client that turns a remote AG-UI endpoint back into an `*agent.Agent`, keeping the history in its session and sending it each time. In a browser you would use an AG-UI JavaScript client instead; the Go client is handy for testing and for agent-to-agent use.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"net/http/httptest"

	aguiSSEClient "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/client/sse"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/aguiprovider"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
	"github.com/sirupsen/logrus"
)

type stockIn struct {
	SKU string `json:"sku" jsonschema:"the product code"`
}

func main() {
	logrus.SetLevel(logrus.WarnLevel) // the AG-UI SSE client logs each connection at info level
	ctx := context.Background()

	// Server side: an agent with a backend tool, mounted at /agui.
	stock := functool.MustNew(functool.Config{Name: "check_stock", Description: "Units in stock for a SKU."},
		func(_ context.Context, in stockIn) (string, error) { return "12 units of " + in.SKU, nil })
	shop := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You are a shop assistant.",
		Config:       agent.Config{Name: "ShopAssistant", Tools: []tool.Tool{stock}},
	})
	mux := http.NewServeMux()
	mux.Handle("/agui", aguiprovider.NewJSONHTTPHandler(shop, aguiprovider.HandlerConfig{Logger: slog.Default()}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// Client side.
	remote := aguiprovider.NewAgent(
		aguiSSEClient.NewClient(aguiSSEClient.Config{Endpoint: srv.URL + "/agui"}),
		aguiprovider.AgentConfig{Instructions: "Answer in one sentence."}, // sent as a system message
	)
	session, err := remote.CreateSession(ctx, aguiprovider.WithThreadID("thread-42"))
	if err != nil {
		log.Fatal(err)
	}
	for update, err := range remote.RunText(ctx, "Is KB-01 in stock?", agent.WithSession(session), agent.Stream(true)) {
		if err != nil {
			log.Fatal(err)
		}
		for _, c := range update.Contents {
			switch c := c.(type) {
			case *message.FunctionCallContent:
				fmt.Printf("server called %s(%s)\n", c.Name, c.Arguments)
			case *message.FunctionResultContent:
				fmt.Printf("tool result: %v\n", c.Result)
			case *message.TextContent:
				fmt.Printf("text: %s\n", c.Text)
			}
		}
	}
	history, _ := session.InMemoryHistory("")
	fmt.Println("messages kept by the client:", len(history))
}
```

Output:

```text
server called check_stock({"sku": "example sku"})
tool result: 12 units of example sku
text: Here is what the tool returned: 12 units of example sku
messages kept by the client: 4
```

The server's tool calls and results stream to the client as events, so a UI can show them. `aguiprovider.WithThreadID` sets the AG-UI thread ID; it identifies the run, but the server does not store history under it.

### Frontend tools

Tools declared by the **client** (a "frontend tool", such as reading the browser's location) are sent with the run input. The model may call them, and the call is streamed to the client, which runs the tool and sends the result back in the next request. In v%%version=github.com/microsoft/agent-framework-go%% this only works if the **server's** automatic tool calling is switched off; otherwise the server tries to run the client's tool itself and the call fails with `client tool "…" cannot be invoked on server`. The upstream sample does this with `ToolAutoCall`:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http/httptest"

	aguiSSEClient "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/client/sse"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolautocall"
	"github.com/microsoft/agent-framework-go/provider/aguiprovider"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
	"github.com/sirupsen/logrus"
)

func main() {
	logrus.SetLevel(logrus.WarnLevel)
	ctx := context.Background()

	// Server: auto-calling off, so client tool calls are passed to the client.
	travel := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Suggest a day trip near the user. Use get_user_location.",
		ToolAutoCall: &toolautocall.Config{MaximumIterationsPerRequest: new(0)},
		Config:       agent.Config{Name: "TravelAgent"},
	})
	srv := httptest.NewServer(aguiprovider.NewJSONHTTPHandler(travel, aguiprovider.HandlerConfig{}))
	defer srv.Close()

	// Client: declares and runs the frontend tool.
	calls := 0
	location := functool.MustNew(functool.Config{Name: "get_user_location", Description: "The user's current city."},
		func(context.Context, struct{}) (string, error) { calls++; return "Utrecht", nil })
	ui := aguiprovider.NewAgent(aguiSSEClient.NewClient(aguiSSEClient.Config{Endpoint: srv.URL}),
		aguiprovider.AgentConfig{Config: agent.Config{Tools: []tool.Tool{location}}})

	session, err := ui.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := ui.RunText(ctx, "Where should I go today?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("frontend tool calls:", calls)
	fmt.Println(resp.String())
}
```

Output:

```text
frontend tool calls: 1
Here is what the tool returned: Utrecht
```

With auto-calling off, the server's own tools are not run automatically either. Keep backend tools and frontend tools on different agents, or handle backend calls in middleware.

**Approvals over AG-UI are not translated in v%%version=github.com/microsoft/agent-framework-go%%.** The AG-UI server has no mapping for `ToolApprovalRequestContent`, so an agent with a `tool.ApprovalRequiredFunc` tool returns an empty response to an AG-UI client when it asks for approval. Until that changes, implement confirmation as a frontend tool (the upstream client sample recognises a `request_approval` tool call for this), or host approval-gated agents over A2A, where approvals work.

**State.** AG-UI state snapshots are produced from `DataContent` with media type `application/json` in the agent's updates. The upstream sample `examples/02-agents/agui/step05_state_management` uses a middleware that detects a JSON object in the model's text and emits it as a `DataContent`.

## MCP over HTTP

An agent exposed as an MCP tool can serve any MCP client: IDEs, desktop assistants and other agents. `mcptool.AddTool` registers a `tool.FuncTool` on an `mcp.Server`, and `agenttool.New` makes an agent into one. The [comprehensive guide](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_comprehensive_guide/#mcp) covers MCP in general, including stdio. Over HTTP, use the MCP SDK's streamable HTTP handler:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/agenttool"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	reviewer := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You review Go code and list problems.",
		Config:       agent.Config{Name: "go_reviewer", Description: "Reviews Go code."},
	})

	server := mcp.NewServer(&mcp.Implementation{Name: "review-service", Version: "1.0.0"}, nil)
	mcptool.AddTool(server, agenttool.New(reviewer, agenttool.Config{}))
	handler := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil)
	srv := httptest.NewServer(handler)
	defer srv.Close()

	// Any MCP client can now call the agent; here, the Go SDK's own client.
	session, err := mcptool.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: srv.URL})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	tools, err := mcptool.ListTools(ctx, session)
	if err != nil {
		log.Fatal(err)
	}
	for _, t := range tools {
		fmt.Printf("tool %s: %s\n", t.Name(), t.Description())
	}
	result, err := tools[0].(tool.FuncTool).Call(ctx, `{"query":"func f() { for {} }"}`)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("result:", result)
}
```

Output against the stand-in:

```text
tool go_reviewer: Reviews Go code.
result: [Mock reply to: func f() { for {} }]
```

The tool's result arrives as the MCP content list, which prints as a slice. The `getServer` function runs for each new MCP session, so it is also where you would pick a per-tenant server. Put authentication middleware in front of the handler as for A2A.

## Your own HTTP endpoint

When the caller is your own frontend or service and neither protocol fits, a handler that runs the agent and streams the text is a few lines. Keep sessions in a store keyed by a conversation ID that you issue, and pass the request's context so that a client disconnect cancels the model call:

```go
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

type chatRequest struct {
	ConversationID string `json:"conversation_id"`
	Message        string `json:"message"`
}

type sessions struct {
	mu   sync.Mutex
	byID map[string]*agent.Session
}

func (s *sessions) get(r *http.Request, a *agent.Agent, id string) (*agent.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sess, ok := s.byID[id]; ok {
		return sess, nil
	}
	sess, err := a.CreateSession(r.Context())
	if err != nil {
		return nil, err
	}
	s.byID[id] = sess
	return sess, nil
}

func chatHandler(a *agent.Agent, store *sessions) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil || req.Message == "" {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		session, err := store.get(r, a, req.ConversationID)
		if err != nil {
			http.Error(w, "session error", http.StatusInternalServerError)
			return
		}
		flusher, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		for update, err := range a.RunText(r.Context(), req.Message, agent.WithSession(session), agent.Stream(true)) {
			if err != nil {
				fmt.Fprintf(w, "event: error\ndata: %s\n\n", "the agent failed") // log err; do not leak it
				log.Printf("run failed: %v", err)
				return
			}
			if text := update.String(); text != "" {
				data, _ := json.Marshal(map[string]string{"text": text})
				fmt.Fprintf(w, "data: %s\n\n", data)
				if flusher != nil {
					flusher.Flush()
				}
			}
		}
		fmt.Fprint(w, "event: done\ndata: {}\n\n")
	}
}

func main() {
	assistant := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai-small%%",
		Config: agent.Config{Name: "assistant"},
	})
	mux := http.NewServeMux()
	mux.Handle("POST /chat", chatHandler(assistant, &sessions{byID: map[string]*agent.Session{}}))
	srv := httptest.NewServer(mux)
	defer srv.Close()

	for _, msg := range []string{"Hello", "What did I just say?"} {
		body := strings.NewReader(fmt.Sprintf(`{"conversation_id":"c-1","message":%q}`, msg))
		resp, err := http.Post(srv.URL+"/chat", "application/json", body)
		if err != nil {
			log.Fatal(err)
		}
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			if line := sc.Text(); line != "" {
				fmt.Println(line)
			}
		}
		_ = resp.Body.Close()
	}
}
```

Output against the stand-in:

```text
data: {"text":"Mock reply to: Hello"}
event: done
data: {}
data: {"text":"Mock reply to: What did I just say?"}
event: done
data: {}
```

An in-memory map is fine for one replica. For more, store `json.Marshal(session)` in a database or cache and load it per request, as the [production page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_production/#persisting-sessions) shows. Do not run two requests on one session at the same time.

## Compared with Python and .NET

| Python / .NET | Go |
| --- | --- |
| A2A hosting (`A2AServer`, `MapA2A`) | `a2aprovider.NewHandler` plus a2a-go transport handlers |
| A2A client agent | `a2aprovider.NewAgentFromResolver` / `NewAgentFromCard` / `NewAgent` |
| AG-UI hosting (`add_agent_framework_fastapi_endpoint`, `MapAGUI`) | `aguiprovider.NewJSONHTTPHandler` |
| AG-UI client | `aguiprovider.NewAgent` |
| OpenAI-compatible hosting (.NET `MapOpenAIResponses`) | Not available |
| DevUI | Not available |
| Foundry hosted agents | Calling existing ones only; deployment not available |
| Durable agents (Azure Functions, Durable Task) | Not available |

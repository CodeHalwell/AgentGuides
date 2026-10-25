---
title: "Microsoft Agent Framework (Go) — Middleware"
description: "Agent middleware and function middleware in the Go SDK: the run pipeline and its order, streaming and short-circuiting, changing messages and options, intercepting every model call, guardrails, redaction, caching, usage and rate limits, retries and timeouts in the vendor clients, error types, message injection and the loop middleware."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Middleware"
  order: 6
---

Go has two kinds of middleware, both set on `agent.Config`:

- **Agent middleware** (`Config.Middlewares`, type `agent.Middleware`) wraps a whole run. It receives the input messages and options and a `next` function, and returns the stream of updates, so it can change the input, filter or replace the output, time, log, retry or refuse the run.
- **Function middleware** (`Config.FunctionMiddlewares`, type `agent.FunctionInvocationMiddleware`) wraps each tool call. It sees the tool, the call ID and the raw JSON arguments, and can change the arguments, replace the result or skip the call.

There is no separate "chat middleware" type. Code that must see every individual model call (inside the tool loop) can be placed there by building the agent with `agent.New`, shown in [Intercepting every model call](#intercepting-every-model-call); HTTP-level concerns such as retries belong in the vendor client.

Every program on this page compiles with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%%, and all were run against a local stand-in for the OpenAI API that calls the first tool it is offered and then answers with the tool's result.

## The run pipeline

A run passes through these layers, outermost first:

1. **The run logger**, if `Config.Logger` is set (and `DisableRunLogs` is not). It logs `run invoked`, `run completed` with the duration, and `run failed` at error level; messages, options and updates only with `LogSensitiveData`.
2. **`Config.Middlewares`**, in order: the first is outermost.
3. **The agent lifecycle**: the history provider loads history, then each context provider adds context. After the provider finishes, history is stored and context providers are told the outcome.
4. **Provider middleware**, inside the lifecycle: for the OpenAI, Foundry, Anthropic and Gemini providers this is the `toolautocall` tool loop, followed by message injection, per-call history persistence and structured-output decoding when configured.
5. **The provider's `Run`**, which calls the model.

Function middleware is applied to every tool the loop executes, in registration order, first outermost.

So agent middleware sees your messages *before* history and context are added, and sees the final response *after* the tool loop has finished. If it calls `next` twice (a retry), the history and context providers run twice too, which is what you want: a failed run stores nothing, because the default history and context providers skip storing when the run fails.

## Agent middleware

`agent.Middleware` is an interface with one method; `agent.MiddlewareFunc` adapts a plain function:

```text
type MiddlewareFunc func(next agent.RunFunc, ctx context.Context, messages []*message.Message, options ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error]
```

A middleware returns an iterator. Nothing runs until the caller ranges over it (or calls `Collect`), and code placed before `return func(yield ...)` runs when the run is *created*, not when it starts. Inside the iterator, range over `next(...)` and pass each update on with `yield`; when `yield` returns false the caller has stopped reading, and you must return. The same middleware serves streaming and non-streaming runs: without `agent.Stream(true)` the provider yields fewer, larger updates, but the shape is the same.

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"log/slog"
	"os"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

// timing logs how long each run took, how many updates it produced and its token usage.
var timing agent.MiddlewareFunc = func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	return func(yield func(*agent.ResponseUpdate, error) bool) {
		start := time.Now()
		var updates int
		var usage message.UsageDetails
		defer func() {
			name := ""
			if a, ok := agent.AgentFromContext(ctx); ok {
				name = a.Name()
			}
			slog.Info("run", "agent", name, "took", time.Since(start), "updates", updates,
				"in", usage.InputTokenCount, "out", usage.OutputTokenCount)
		}()
		for update, err := range next(ctx, msgs, opts...) {
			if update != nil {
				updates++
				usage.Add(update.Usage())
			}
			if !yield(update, err) {
				return
			}
		}
	}
}

// house adds an instruction to every run and leaves the messages alone.
var house agent.MiddlewareFunc = func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	opts = append(opts[:len(opts):len(opts)], agent.WithInstructions("Use British spelling."))
	return next(ctx, msgs, opts...)
}

func main() {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai%%",
		Config: agent.Config{
			Name:        "Helper",
			Middlewares: []agent.Middleware{timing, house}, // timing is outermost
		},
	})

	ctx := context.Background()
	resp, err := a.RunText(ctx, "What colour is the sky?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	for update, err := range a.RunText(ctx, "And grass?", agent.Stream(true)) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(update)
	}
	fmt.Println()
}
```

`opts[:len(opts):len(opts)]` forces `append` to copy, so the caller's slice is not modified. The framework requires middleware to treat the message and option slices, and the messages in them, as read-only: clone a slice before changing it, and `msg.Clone()` a message before editing its contents. Options are a slice of values looked up by type; adding a second `WithInstructions` adds an instruction, while adding a second single-valued option (`WithToolMode`, a provider's native parameters) overrides the earlier one, because lookups start from the end.

`agent.AgentFromContext(ctx)` returns the agent running the middleware. The session is an option: `session, _ := agent.GetOption(opts, agent.WithSession)`. When the caller passed no session, the agent creates a throwaway one for the run, so the lookup still returns a session.

### Short-circuiting and replacing the output

A middleware does not have to call `next`. To refuse a run, yield an error; to answer it yourself, yield updates. `(*agent.Response).ToUpdates()` turns a whole response into updates, which suits caches:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"regexp"
	"strings"
	"sync"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

var ErrBlocked = errors.New("request blocked by policy")

var blocked = regexp.MustCompile(`(?i)\b(password|credit card)\b`)

// guardrail refuses runs whose input mentions blocked topics, without calling the model.
var guardrail agent.MiddlewareFunc = func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	for _, m := range msgs {
		if blocked.MatchString(m.String()) {
			return func(yield func(*agent.ResponseUpdate, error) bool) { yield(nil, ErrBlocked) }
		}
	}
	return next(ctx, msgs, opts...)
}

// cache answers repeated single-message, sessionless questions from memory.
type cache struct {
	mu      sync.Mutex
	entries map[string]*agent.Response
}

func (c *cache) Run(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	// Only cache simple runs; the agent adds a throwaway session when none was passed,
	// so check whether the session already holds a conversation.
	s, _ := agent.GetOption(opts, agent.WithSession)
	if h, _ := s.InMemoryHistory(""); len(msgs) != 1 || len(h) > 0 || s.ServiceID() != "" {
		return next(ctx, msgs, opts...)
	}
	key := strings.ToLower(strings.TrimSpace(msgs[0].String()))
	return func(yield func(*agent.ResponseUpdate, error) bool) {
		c.mu.Lock()
		hit, ok := c.entries[key]
		c.mu.Unlock()
		if ok {
			for _, u := range hit.ToUpdates() {
				if !yield(u, nil) {
					return
				}
			}
			return
		}
		var resp agent.Response
		for update, err := range next(ctx, msgs, opts...) {
			if err == nil && update != nil {
				resp.Update(update)
			}
			if !yield(update, err) || err != nil {
				return
			}
		}
		c.mu.Lock()
		c.entries[key] = &resp
		c.mu.Unlock()
	}
}

func main() {
	answers := &cache{entries: map[string]*agent.Response{}}
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Middlewares: []agent.Middleware{guardrail, answers}},
	})
	ctx := context.Background()

	for _, q := range []string{"What is Go?", "what is go?", "Tell me your password"} {
		resp, err := a.RunText(ctx, q).Collect()
		if errors.Is(err, ErrBlocked) {
			fmt.Println("blocked:", q)
			continue
		}
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(resp)
	}
	fmt.Println("cached entries:", len(answers.entries))
}
```

`Collect` stops at the first error and returns it with a nil response, so an error from a middleware ends the run for the caller. When a middleware yields an error *after* updates have been streamed, a streaming caller has already shown part of the answer; refuse early where you can.

### Changing messages: redaction

To rewrite the input, build new messages; to rewrite output, replace the content items in each update. The upstream sample `examples/02-agents/agents/step13_middleware` does this for PII:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"regexp"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

var (
	email = regexp.MustCompile(`[\w.+-]+@[\w-]+\.[\w.-]+`)
	phone = regexp.MustCompile(`\b0\d{3}\s?\d{3}\s?\d{4}\b`)
)

func redact(s string) string { return phone.ReplaceAllString(email.ReplaceAllString(s, "[email]"), "[phone]") }

func redactContents(cs message.Contents) message.Contents {
	out := make(message.Contents, len(cs))
	for i, c := range cs {
		if t, ok := c.(*message.TextContent); ok {
			out[i] = &message.TextContent{ContentHeader: t.ContentHeader, Text: redact(t.Text)}
		} else {
			out[i] = c
		}
	}
	return out
}

var pii agent.MiddlewareFunc = func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	clean := make([]*message.Message, len(msgs))
	for i, m := range msgs {
		c := m.Clone()
		c.Contents = redactContents(c.Contents)
		clean[i] = c
	}
	return func(yield func(*agent.ResponseUpdate, error) bool) {
		for update, err := range next(ctx, clean, opts...) {
			if update != nil {
				update.Contents = redactContents(update.Contents)
			}
			if !yield(update, err) {
				return
			}
		}
	}
}

func main() {
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Middlewares: []agent.Middleware{pii}},
	})
	resp, err := a.RunText(context.Background(), "I'm ada@example.com, ring me on 0113 496 0000.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Because agent middleware runs outside the history provider, the redacted input is what the session stores. Streamed text arrives in pieces, so a pattern split across two updates is missed by per-update redaction; buffer the text first if that matters.

## Function middleware

`agent.FunctionInvocationMiddleware` is a function type:

```text
type FunctionInvocationMiddleware func(
	next func(context.Context, *agent.FunctionInvocationContext) (any, error),
	ctx context.Context,
	inv *agent.FunctionInvocationContext,
) (any, error)
```

| `FunctionInvocationContext` field | Meaning |
| --- | --- |
| `Function tool.FuncTool` | the tool about to run. Replace it to redirect the call; the replacement runs directly, skipping later middleware |
| `CallID string` | the model's call ID; read-only, empty if the provider gave none |
| `Arguments string` | the raw JSON arguments, before the tool validates them. Replace it to change them |

Return without calling `next` to skip the tool: whatever you return is the result (or error) the model sees, and the loop carries on. Errors are treated like tool errors: the error text reaches the model (see [tools](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#what-the-model-sees-when-a-tool-fails)) and the consecutive-error count rises. Function middleware is also where to sanitise those errors; the [production page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_production/#limits-on-the-tool-loop) has one. Function middleware runs for every executed tool, including tools from context providers and MCP servers, and runs only when a tool actually executes; for an approval-required tool, that is after approval.

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type PathIn struct {
	Path string `json:"path"`
}

var readFile = functool.MustNew(functool.Config{Name: "read_file", Description: "Read a report."},
	func(ctx context.Context, in PathIn) (string, error) { return "contents of " + in.Path, nil })

// audit logs every call with its duration and outcome.
func audit(next func(context.Context, *agent.FunctionInvocationContext) (any, error), ctx context.Context, inv *agent.FunctionInvocationContext) (any, error) {
	start := time.Now()
	result, err := next(ctx, inv)
	slog.Info("tool", "name", inv.Function.Name(), "call", inv.CallID, "took", time.Since(start), "err", err)
	return result, err
}

// confine rewrites relative paths into reports/ and refuses anything outside it.
func confine(next func(context.Context, *agent.FunctionInvocationContext) (any, error), ctx context.Context, inv *agent.FunctionInvocationContext) (any, error) {
	var args map[string]any
	if err := json.Unmarshal([]byte(inv.Arguments), &args); err != nil {
		return next(ctx, inv) // let the tool's own validation report it
	}
	p, ok := args["path"].(string)
	if !ok {
		return next(ctx, inv)
	}
	if strings.Contains(p, "..") || strings.HasPrefix(p, "/") && !strings.HasPrefix(p, "/reports/") {
		return "refused: only files under /reports/ may be read", nil // the tool never runs
	}
	if !strings.HasPrefix(p, "/") {
		args["path"] = "/reports/" + p
		b, _ := json.Marshal(args)
		inv.Arguments = string(b)
	}
	return next(ctx, inv)
}

// memo caches results per tool and arguments for the life of the process.
type memo struct {
	mu   sync.Mutex
	seen map[string]any
}

func (m *memo) invoke(next func(context.Context, *agent.FunctionInvocationContext) (any, error), ctx context.Context, inv *agent.FunctionInvocationContext) (any, error) {
	key := inv.Function.Name() + inv.Arguments
	m.mu.Lock()
	r, ok := m.seen[key]
	m.mu.Unlock()
	if ok {
		return r, nil
	}
	r, err := next(ctx, inv)
	if err == nil {
		m.mu.Lock()
		m.seen[key] = r
		m.mu.Unlock()
	}
	return r, err
}

func main() {
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))
	cache := &memo{seen: map[string]any{}}

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai%%",
		Config: agent.Config{
			Tools:               []tool.Tool{readFile},
			FunctionMiddlewares: []agent.FunctionInvocationMiddleware{audit, confine, cache.invoke},
		},
	})
	for _, q := range []string{"Read the summary", "Read it again"} {
		resp, err := a.RunText(context.Background(), q).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(resp)
	}
	fmt.Println("cached results:", len(cache.seen))
}
```

The loop runs calls one at a time unless `AllowConcurrentInvocations` is set, in which case several invocations of the same middleware run at once; each has its own `FunctionInvocationContext`, but shared state (the cache map above) needs a lock either way.

## Agent-level, not per-run

Middleware is fixed when the agent is built; there is no per-run middleware option. To vary behaviour per run, pass an option the middleware looks for. Run options are an open set: any type with a `MAFValue() any` method is an option, and `agent.GetOption` finds it by type.

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

// tenantOpt is a custom run option. Each option needs its own distinct type.
type tenantOpt string

func (t tenantOpt) MAFValue() any { return string(t) }

// WithTenant returns the option; its signature lets agent.GetOption look it up.
func WithTenant(id string) agent.Option { return tenantOpt(id) }

var tenantInstructions agent.MiddlewareFunc = func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	if tenant, ok := agent.GetOption(opts, WithTenant); ok {
		opts = append(opts[:len(opts):len(opts)], agent.WithInstructions("You are answering for tenant "+tenant+"."))
	}
	return next(ctx, msgs, opts...)
}

func main() {
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Middlewares: []agent.Middleware{tenantInstructions}},
	})
	resp, err := a.RunText(context.Background(), "Who am I talking to?", WithTenant("contoso")).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Providers ignore option types they do not know, so custom options are safe to pass through.

## Intercepting every model call

Agent middleware sees one run; the tool loop inside it may call the model many times. To see each call (to log the exact messages sent, count calls, or enforce a per-call budget) put the middleware where the provider's tool loop is, in `agent.ProviderConfig.Middlewares`. The provider constructors do not expose that field, but `agent.New` does, and an existing agent can serve as the provider's `Run`:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolautocall"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type SKUIn struct {
	SKU string `json:"sku"`
}

var stock = functool.MustNew(functool.Config{Name: "check_stock", Description: "Units in stock."},
	func(ctx context.Context, in SKUIn) (int, error) { return 12, nil })

// perCall runs once per model call, inside the tool loop.
func perCall(calls *int) agent.MiddlewareFunc {
	return func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		*calls++
		fmt.Printf("model call %d: %d messages\n", *calls, len(msgs))
		return next(ctx, msgs, opts...)
	}
}

func main() {
	// The inner agent only talks to the model: no tool loop, no history of its own.
	inner := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:              "%%model=openai%%",
		DisableStoreOutput: true,
		ToolAutoCall:       &toolautocall.Config{MaximumIterationsPerRequest: new(0)},
		Config: agent.Config{
			HistoryProvider: agent.NewHistoryProvider(agent.HistoryProviderConfig{SourceID: "none"}),
		},
	})

	var calls int
	a := agent.New(agent.ProviderConfig{
		ProviderName: inner.ProviderName(),
		Run: func(ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
			return iter.Seq2[*agent.ResponseUpdate, error](inner.Run(ctx, msgs, opts...))
		},
		// The tool loop first (outermost), then the per-call middleware inside it.
		Middlewares: []agent.Middleware{toolautocall.New(toolautocall.Config{}), perCall(&calls)},
	}, agent.Config{
		Name:  "Counted",
		Tools: []tool.Tool{stock},
	})

	ctx := context.Background()
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "Is T-100 in stock?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
	fmt.Println("model calls in one run:", calls)
}
```

The run makes two model calls (one that asks for the tool, one that answers), and `perCall` sees both. The inner agent has its tool loop switched off and a history provider that does nothing, and output storage disabled, so the outer agent alone runs tools and keeps the conversation. Structured output still works because the outer run's options reach the inner agent unchanged.

For HTTP-level interception (headers, raw request bodies, status codes) use the vendor client's own middleware instead, such as `option.WithMiddleware` on the OpenAI and Anthropic clients. That also sees every model call, and needs no restructuring.

## Common patterns

### Rate limits and concurrency

An `*agent.Agent` can be shared between goroutines. To cap concurrent runs, hold a semaphore for the life of the stream:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"sync"
	"sync/atomic"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func limit(n int) agent.MiddlewareFunc {
	slots := make(chan struct{}, n)
	return func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			select {
			case slots <- struct{}{}:
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			}
			defer func() { <-slots }()
			for update, err := range next(ctx, msgs, opts...) {
				if !yield(update, err) {
					return
				}
			}
		}
	}
}

func main() {
	var inFlight, peak atomic.Int32
	watch := agent.MiddlewareFunc(func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			n := inFlight.Add(1)
			defer inFlight.Add(-1)
			for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
			}
			for update, err := range next(ctx, msgs, opts...) {
				if !yield(update, err) {
					return
				}
			}
		}
	})

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Middlewares: []agent.Middleware{limit(2), watch}},
	})

	var wg sync.WaitGroup
	for i := range 6 {
		wg.Go(func() {
			if _, err := a.RunText(context.Background(), fmt.Sprintf("Question %d", i)).Collect(); err != nil {
				log.Print(err)
			}
		})
	}
	wg.Wait()
	fmt.Println("peak concurrent runs:", peak.Load())
}
```

For a rate in requests per second rather than a concurrency cap, use `golang.org/x/time/rate` in the same position, calling `limiter.Wait(ctx)` before `next`.

### Logging and tracing

Set `agent.Config.Logger` for the built-in run logs, and `toolautocall.Config.Logger` (or leave `ToolAutoCall` nil so it inherits the agent's) for tool-invocation logs. `LogSensitiveData` adds messages, arguments and results, so leave it off in production. For spans and metrics following the OpenTelemetry GenAI conventions, add `otelprovider.NewMiddleware(...)` to `Middlewares`; see [observability](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/#agent-tracing-and-metrics).

## Retries and timeouts

Transient HTTP failures are retried by the vendor clients, below the framework, so there is usually nothing to add:

| Client | Default | Configure with |
| --- | --- | --- |
| `openai.Client` (OpenAI, Azure OpenAI, Foundry) | 2 retries with backoff, on connection errors, 408, 409, 429 and 5xx | `option.WithMaxRetries(n)`, `option.WithRequestTimeout(d)` (per attempt) |
| `anthropic.Client` | 2 retries | `option.WithMaxRetries(n)`, `option.WithRequestTimeout(d)` from `github.com/anthropics/anthropic-sdk-go/option` |
| `*genai.Client` | per `HTTPOptions` | `genai.ClientConfig{HTTPOptions: genai.HTTPOptions{Timeout: ..., RetryOptions: &genai.HTTPRetryOptions{Attempts: ...}}}` |

Options passed to the client constructor apply to every request; for Foundry pass them in `AgentConfig.OpenAIOptions`. The deadline that covers a whole run, including every model call and tool call in the loop, is the `context.Context` you pass to `RunText`:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// retryRun retries a whole run when it fails before producing any output.
// The vendor client already retries individual HTTP requests; this covers
// failures it does not, such as a rate limit that outlasts its backoff.
func retryRun(attempts int, wait time.Duration) agent.MiddlewareFunc {
	return func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			for attempt := 1; ; attempt++ {
				started := false
				var failure error
				for update, err := range next(ctx, msgs, opts...) {
					if err != nil && !started && retryable(err) && attempt < attempts {
						failure = err
						break
					}
					started = true
					if !yield(update, err) || err != nil {
						return
					}
				}
				if failure == nil {
					return
				}
				select {
				case <-time.After(wait * time.Duration(attempt)):
				case <-ctx.Done():
					yield(nil, ctx.Err())
					return
				}
			}
		}
	}
}

func retryable(err error) bool {
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 429 || apiErr.StatusCode >= 500
	}
	return false
}

func main() {
	client := openai.NewClient(
		option.WithMaxRetries(3),
		option.WithRequestTimeout(45*time.Second),
	)
	a := openaiprovider.NewAgent(client, openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{Middlewares: []agent.Middleware{retryRun(3, 2*time.Second)}},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	resp, err := a.RunText(ctx, "Summarise the Go memory model in one sentence.").Collect()
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		log.Fatal("the run took too long")
	case err != nil:
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

A retry at this level re-runs the tool loop from the start, so tools that ran before the failure run again; keep tools idempotent, or retry only failures that happen before any tool call (as above, where a retry is attempted only before the first update).

## Errors

Errors surface as the second value of the stream. `Collect` returns the first one, with a nil response. The framework does not wrap provider errors in its own types, so match them with the vendor SDK's types:

| Source | Match with | Useful fields |
| --- | --- | --- |
| OpenAI, Azure OpenAI, Foundry | `var e *openai.Error; errors.As(err, &e)` | `StatusCode`, `Code`, `Type`, `Message`, `Request`, `Response`, `DumpResponse(true)` |
| Anthropic | `var e *anthropic.Error; errors.As(err, &e)` | `StatusCode`, `Request`, `Response` |
| Gemini | `var e genai.APIError; errors.As(err, &e)` | `Code`, `Status`, `Message`, `Details` |
| cancellation and deadlines | `errors.Is(err, context.Canceled)`, `context.DeadlineExceeded` | |
| tools, after the consecutive-error limit | the tool's own error (`errors.Join` of a round's errors when several failed) | |
| MCP transport failures | errors wrapping `MCP tool call failed` | |

Errors the framework itself returns are plain `errors.New` values with descriptive text, for example `message cannot be blank` from `RunText("")`, `a session must be provided when AllowBackgroundResponses is enabled`, or the history-conflict error described on the [sessions page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/#service-held-and-local-history). Configuration mistakes (a nil client, a blank `SourceID`, an invalid shell timeout) mostly panic at construction instead.

A tool error does not fail the run; the model is told and the loop continues. Approval requests are not errors either: the run ends normally with the request in the response.

## Injecting messages during a run

`agent.Config.MessageInjector` lets code outside the model add messages to a run that is already going, between model calls. Typically a tool learns something the model should know, or another goroutine has news. `EnqueueMessages(session, msgs...)` queues messages for that session; the injector, which runs inside the tool loop, adds them to the next model call. Its zero value is ready to use.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type OrderIn struct {
	OrderID string `json:"order_id"`
}

func main() {
	ctx := context.Background()
	injector := &agent.MessageInjector{}
	var session *agent.Session

	lookup := functool.MustNew(functool.Config{Name: "lookup_order", Description: "Current status of an order."},
		func(ctx context.Context, in OrderIn) (string, error) {
			// Something else has happened that the model should hear about on its next call.
			err := injector.EnqueueMessages(session, message.NewText("Update: order "+in.OrderID+" has just shipped."))
			return "order " + in.OrderID + " is being packed", err
		})

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:              "%%model=openai%%",
		DisableStoreOutput: true,
		Config: agent.Config{
			Tools:           []tool.Tool{lookup},
			MessageInjector: injector,
			// Keep injected messages in local history between model calls.
			RequirePerServiceCallHistoryPersistence: true,
		},
	})

	var err error
	session, err = a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "Where is order A-1234?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	pending, _ := injector.PendingMessages(session)
	fmt.Println("still queued:", len(pending))
}
```

Messages queued while no run is active wait for the next run on that session. With a stateless service (Chat Completions, Anthropic, Gemini, or Responses with output storage disabled) set `RequirePerServiceCallHistoryPersistence` so the injected messages, and the responses between calls, are kept in the session.

## The loop middleware

`agent/harness/loop` re-runs an agent until it decides the work is done, which suits draft-and-critique and "keep going until the tests pass" tasks. Each **evaluator** looks at the latest response and returns `loop.Continue(feedback)`, `loop.ContinueWithMessages(msgs)` or `loop.Stop()`; the loop continues while any evaluator asks it to, up to `MaxIterations` (default 10).

| `loop.Config` field | Effect |
| --- | --- |
| `Evaluators []loop.Evaluator` | checked in order after each iteration; the first that asks to continue wins |
| `MaxIterations *int` | safety cap per run; default 10 |
| `FreshContextPerIteration` | restart each iteration from the original input plus a log of the feedback, and reset the session to its state before the first iteration |
| `NonStreamingReturnsLastResponseOnly` | a non-streaming run returns only the last iteration's response |
| `OnBehalfOfAuthorName`, `ExcludeOnBehalfOfMessages` | attribute the feedback messages, or keep them out of the output (they are still sent to the agent) |
| `SessionCreatedCallback` | called when the loop creates a fresh session |

`loop.NewCompletionMarkerEvaluator(CompletionMarkerConfig{Marker: "DONE"})` stops when the marker appears in the response text, and otherwise asks the agent to continue with a templated feedback message. `loop.EvaluatorFunc` adapts a function; it receives a `*loop.Context` with `InitialMessages`, `LastResponse`, `Iteration`, the feedback so far, and a scratch map.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/loop"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	tooLong := loop.EvaluatorFunc(func(ctx context.Context, lc *loop.Context) (loop.Evaluation, error) {
		if words := len(strings.Fields(lc.LastResponse.String())); words > 12 {
			return loop.Continue(fmt.Sprintf("That was %d words. Cut it to 12 or fewer.", words)), nil
		}
		return loop.Stop(), nil
	})

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Write product slogans.",
		Config: agent.Config{
			Middlewares: []agent.Middleware{loop.New(loop.Config{
				Evaluators:                          []loop.Evaluator{tooLong},
				MaxIterations:                       new(4),
				NonStreamingReturnsLastResponseOnly: true,
			})},
		},
	})
	resp, err := a.RunText(context.Background(), "A slogan for a tent that pitches itself.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

The loop is agent middleware, so it wraps the history and context providers: each iteration is a complete run with its own tool loop, and with a session each iteration's exchange is stored.

## Differences from Python

- **Functions, not classes or decorators.** Python has `AgentMiddleware`, `ChatMiddleware` and `FunctionMiddleware` classes and `@agent_middleware`-style decorators with a context object and `call_next`. Go middleware is a function that receives `next` and returns an iterator.
- **No chat middleware type.** Per-model-call code goes in `agent.ProviderConfig.Middlewares` via `agent.New`, or in the vendor client's HTTP middleware.
- **No `terminate` flag.** To stop, return without calling `next`, yield an error, or (in function middleware) return a result.
- **Per-run middleware** (Python's `middleware=` on `run`) does not exist; use a custom run option as shown above.
- **Retries** live in the vendor clients, not in a framework retry wrapper.

The [Python middleware page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_middleware/) covers its three middleware levels.

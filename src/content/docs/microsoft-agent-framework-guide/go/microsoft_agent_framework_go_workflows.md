---
title: "Microsoft Agent Framework (Go) — Workflows"
description: "The graph workflow engine in the Go SDK: function, struct and raw executors, bindings, edges, switch, fan-out and fan-in, loops, execution environments, outputs, shared state, events and streaming, human in the loop, checkpointing, sub-workflows, cancellation, agents in workflows and drawing a graph."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Workflows"
  order: 10
---

The `workflow` package is a graph-execution engine. You build a graph of **executors** joined by **edges**, then run it in an execution environment from `workflow/inproc`. Execution proceeds in *supersteps* (the Pregel, or bulk-synchronous, model): every message sent during a superstep is delivered at the start of the next one, the executors that receive messages in a superstep run concurrently, and the run halts when no messages are left or an executor is waiting for outside input.

This page covers the engine itself and how agents fit into it. The prebuilt multi-agent patterns (sequential, concurrent and group chat) and wrapping a workflow as an agent are on the [orchestrations page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_orchestrations/). Tracing a workflow with OpenTelemetry is on the [observability page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/), and testing one on the [testing page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_testing/).

Every program on this page was compiled with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%% and run. The ones built only from executors run offline. The two that call a model were run against a local stand-in for the OpenAI API (set `OPENAI_BASE_URL` and `OPENAI_API_KEY`), so their printed replies are not real model output.

| Package | What it holds |
| --- | --- |
| `workflow` | `Builder`, `Executor`, `ExecutorBinding`, `Context`, edges and edge options, `RequestPort`, `ExternalRequest`/`ExternalResponse`, `PortableValue`, events, `TurnToken`, `StatefulExecutorCache` |
| `workflow/inproc` | The in-process runtime: the `Default`, `OffThread`, `Concurrent` and `Lockstep` environments, `Run`, `StreamingRun`, `BindSubworkflowAsExecutor` |
| `workflow/checkpoint` | `Manager`, `NewInMemoryManager`, `NewJSONManager`, the `Store` interface and `FileSystemJSONStore` |
| `workflow/agentworkflow` | Agents as executors (`New`), the sequential, concurrent and group-chat builders, and `NewAgent` (a workflow as an agent) |
| `message/messageworkflow` | Building blocks for chat-style executors that collect messages until a `TurnToken` arrives |
| `workflow/observability`, `workflow/observability/opentelemetry` | The tracing abstraction and its OpenTelemetry implementation |

## Executors

An executor is a node in the graph. It declares which message types it accepts, which it sends to connected executors, and which it yields as workflow output. The builder uses those declarations to check every edge when you call `Build`.

There are three ways to write one, from least to most code:

| Style | How | Use it when |
| --- | --- | --- |
| Function | `workflow.NewExecutor(id, func(in T) U)` | A pure transformation. The return value is sent along every outgoing edge and, if the executor is an output source, yielded as output. |
| Struct | A struct with a `Handle(ctx *workflow.Context, msg T) error` method, plus `workflow.AttrSendsMessage[T]` and `workflow.AttrYieldsOutput[T]` marker fields | You need the `Context` to send selectively, yield output, read state or raise requests. |
| Raw | A `*workflow.Executor` literal with `ConfigureProtocol` registering handlers on a `RouteBuilder` | One executor must handle several message types, have a catch-all, or switch off the automatic send and yield. |

`NewExecutor` accepts a function with exactly one non-context input, optionally preceded by `*workflow.Context`, and at most one ordinary output plus an optional final `error`. Variadic functions are rejected. It panics, rather than returning an error, for an unsupported signature, a nil value or a mismatched ID, so a mistake shows up the first time the program builds its graph.

The program below uses all three styles. `intake` is a raw executor that accepts either a plain string or an `Order` value, `price` is a struct executor, and `format` is a function:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"reflect"
	"strconv"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type Order struct {
	SKU      string
	Quantity int
}

type Quote struct {
	Order Order
	Total float64
}

// newIntake accepts either "SKU xN" strings or Order values. A raw executor
// registers one handler per message type on the RouteBuilder.
func newIntake() workflow.ExecutorBinding {
	return (&workflow.Executor{
		ID:               "intake",
		ImplementationID: "intake", // Bind requires one on a raw executor
		ConfigureProtocol: func(pb *workflow.ProtocolBuilder) (*workflow.ProtocolBuilder, error) {
			pb.RouteBuilder.
				AddHandlerRaw(reflect.TypeFor[string](), reflect.TypeFor[Order](),
					func(_ *workflow.Context, msg any) (any, error) {
						sku, qty, ok := strings.Cut(msg.(string), " x")
						if !ok {
							return nil, fmt.Errorf("cannot parse %q", msg)
						}
						n, err := strconv.Atoi(qty)
						if err != nil {
							return nil, err
						}
						return Order{SKU: sku, Quantity: n}, nil
					}).
				AddHandlerRaw(reflect.TypeFor[Order](), reflect.TypeFor[Order](),
					func(_ *workflow.Context, msg any) (any, error) { return msg, nil })
			return pb, nil
		},
	}).Bind()
}

// price is a struct executor. The marker field declares what it sends.
type price struct {
	_ workflow.AttrSendsMessage[Quote]
}

func (price) Handle(ctx *workflow.Context, o Order) error {
	unit := map[string]float64{"KB-01": 49.5, "MS-02": 19}[o.SKU]
	return ctx.SendMessage("", Quote{Order: o, Total: unit * float64(o.Quantity)})
}

func main() {
	intake := newIntake()
	pricing := workflow.NewExecutor("price", price{}).Bind()
	format := workflow.NewExecutor("format", func(q Quote) string {
		return fmt.Sprintf("%d x %s = £%.2f", q.Order.Quantity, q.Order.SKU, q.Total)
	}).Bind()

	wf, err := workflow.NewBuilder(intake).
		AddChain(intake, []workflow.ExecutorBinding{pricing, format}, false).
		WithOutputFrom(format).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	for _, id := range []string{"intake", "price", "format"} {
		b, _ := wf.ExecutorBinding(id)
		ex, _ := b.CreateInstance("describe")
		p := ex.DescribeProtocol()
		fmt.Printf("%-6s accepts %v, sends %v\n", id, p.Accepts, p.Sends)
	}

	for _, in := range []any{"KB-01 x2", Order{SKU: "MS-02", Quantity: 3}} {
		run, err := inproc.Default.Run(context.Background(), wf, in)
		if err != nil {
			log.Fatal(err)
		}
		for evt := range run.NewEvents() {
			if o, ok := evt.(workflow.OutputEvent); ok {
				fmt.Println("output:", o.Output)
			}
		}
		_ = run.Close(context.Background())
	}
}
```

Output (the order of the accepted types can vary between runs):

```text
intake accepts [string main.Order], sends [main.Order]
price  accepts [main.Order], sends [main.Quote]
format accepts [main.Quote], sends [string]
output: 2 x KB-01 = £99.00
output: 3 x MS-02 = £57.00
```

Some details of how handlers behave:

- **Automatic send and yield.** A handler's return value is sent along every outgoing edge, and yielded as output if the executor is an output source. Set `AutoSendMessageHandlerResultObject` or `AutoYieldOutputHandlerResultObject` to `new(false)` on a raw executor to turn either off. Passing `nil` as the output type to `AddHandlerRaw` registers an action-style handler whose return value is ignored.
- **Declared types.** A struct or raw executor must declare everything it sends or yields through the `Context`: with the marker fields, or with `ProtocolBuilder.SendsMessageType` and `YieldsOutputType`. `ctx.YieldOutput` with an undeclared type fails the executor.
- **Catch-all.** `RouteBuilder.AddCatchAll` receives any message without a typed handler as a `workflow.PortableValue`. A catch-all also makes the executor accept every type for edge validation. Registering a second handler for the same type, or a second catch-all, is an error unless you pass `workflow.WithHandlerOverwrite(true)`.
- **Errors.** A handler that returns an error produces an `ExecutorFailedEvent` followed by an `ErrorEvent`, and the run ends (`RunStatusEnded`). `Run` itself still returns a nil error, so check the events. There is no retry at the engine level.
- **No handler.** A message of a type the executor does not handle is an error, `no handler found for message type T`. Edge validation catches most such cases at build time.

### The workflow context

`*workflow.Context` embeds `context.Context`, so you can pass it straight to an agent run or an HTTP call and it carries the run's cancellation. Its other members are function fields:

| Member | What it does |
| --- | --- |
| `SendMessage(targetID, msg)` | Queues a message for the next superstep. An empty `targetID` sends along every outgoing edge; a non-empty one sends only to that executor, which must still be connected by an edge. |
| `YieldOutput(v)` | Emits `v` as workflow output (an `OutputEvent`) if this executor is an output source. |
| `AddEvent(e)` | Raises any value that implements `workflow.Event` (`Data() any`) in the event stream at the end of the superstep. |
| `RequestHalt()` | Halts the run at the end of the current superstep. |
| `PostRequest(req)` | Raises an `ExternalRequest` for the caller to answer (see [human in the loop](#human-in-the-loop)). |
| `ReadState`, `ReadOrInitState`, `ReadStateKeys`, `QueueStateUpdate`, `QueueClearScope` | Shared and private state (see [shared state](#shared-state)). |
| `TraceContext()` | The trace context carried by the message being processed. |
| `ConcurrentRunsEnabled` | Whether the environment allows several runs of this workflow at once. |

### Lifecycle hooks

Fields on `workflow.Executor` hook into the executor's life. Combine your own hooks with a function or struct executor through `Extend`, which runs the existing hooks first and then the new ones:

| Field | Called |
| --- | --- |
| `InitializeFunc` | When the executor instance is created for a run |
| `OnMessageDeliveryStartingFunc` / `OnMessageDeliveryFinishedFunc` | Once per superstep, before the first and after the last message delivered to this executor |
| `OnCheckpointFunc` / `OnCheckpointRestoredFunc` | Before a checkpoint is taken and after one is restored |
| `ResetFunc` | Before a shared instance is reused by another run |
| `CloseFunc` | When the run ends |

`OnMessageDeliveryFinishedFunc` is the natural place to act once on everything an executor received in a superstep, for example to emit one summary after a fan-in.

## Bindings and per-run state

The builder does not store executors. It stores `ExecutorBinding`s, which say how a runner gets an executor instance for a run. The two ways to get one behave very differently when the executor has fields or captured variables:

| Binding | Instance | State between runs |
| --- | --- | --- |
| `executor.Bind()` | One shared instance for every run of the workflow | Kept, unless you supply `ResetFunc` |
| `workflow.BindNewExecutorFunc(id, factory)` | A new instance per run, from `factory(sessionID, executorID)` | Fresh every run |

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

func counterWorkflow(perRun bool) *workflow.Workflow {
	var counter workflow.ExecutorBinding
	if perRun {
		counter = workflow.BindNewExecutorFunc("count", func(_, id string) (*workflow.Executor, error) {
			n := 0
			return workflow.NewExecutor(id, func(s string) string { n++; return fmt.Sprintf("%s #%d", s, n) }), nil
		})
	} else {
		n := 0
		counter = workflow.NewExecutor("count", func(s string) string { n++; return fmt.Sprintf("%s #%d", s, n) }).Bind()
	}
	wf, err := workflow.NewBuilder(counter).WithOutputFrom(counter).Build()
	if err != nil {
		log.Fatal(err)
	}
	return wf
}

func runOnce(wf *workflow.Workflow, in string) {
	ctx := context.Background()
	run, err := inproc.Default.Run(ctx, wf, in)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	for evt := range run.NewEvents() {
		if o, ok := evt.(workflow.OutputEvent); ok {
			fmt.Println(o.Output)
		}
	}
}

func main() {
	shared := counterWorkflow(false)
	runOnce(shared, "shared")
	runOnce(shared, "shared")

	perRun := counterWorkflow(true)
	runOnce(perRun, "per-run")
	runOnce(perRun, "per-run")

	// A workflow can be owned by one runner at a time.
	ctx := context.Background()
	first, err := inproc.Default.RunStreaming(ctx, shared, "a")
	if err != nil {
		log.Fatal(err)
	}
	_, err = inproc.Default.RunStreaming(ctx, shared, "b")
	fmt.Println("second run:", err)
	_ = first.Close(ctx)
}
```

Output:

```text
shared #1
shared #2
per-run #1
per-run #1
second run: cannot use a Workflow that is already owned by another runner or parent workflow
```

Use `BindNewExecutorFunc` for anything with per-run state, or give the shared executor a `ResetFunc` that restores its fields.

There is a catch when you mix the two. As soon as **any** executor in a workflow has a reset hook (`NewAggregatingExecutor` and `StatefulExecutorCache` users usually do), the workflow tries to reset every shared executor when a run ends, and a shared executor without a `ResetFunc` makes that fail. The next run then fails to start with `cannot reuse Workflow with shared Executor instances that do not implement IResettableExecutor`. A workflow whose shared executors have no hooks at all can be reused, but keeps their state, as `shared #2` shows. A `*workflow.Workflow` is owned by one runner at a time, so a server that runs the same graph for many requests needs either a fresh workflow per request (building one is cheap) or the `inproc.Concurrent` environment, described [below](#running-one-workflow-concurrently).

## Edges and routing

| Builder method | Shape |
| --- | --- |
| `AddEdge(src, dst, opts...)` | Direct edge. With `WithEdgeCondition`, only messages that satisfy the predicate pass. |
| `AddChain(src, []ExecutorBinding, allowRepetition)` | `src → a → b → …` in one call |
| `AddFanOutEdge(src, targets, opts...)` | One message to many targets. With `WithEdgeAssigner`, a function picks the targets per message. |
| `AddFanInBarrierEdge(sources, dst)` | Waits until every source has sent, then delivers all their messages to `dst` in the same superstep |
| `AddSwitch(src).AddCase(pred, targets...).WithDefault(targets...).AddToBuilder(b)` | First matching case wins; the default catches the rest |
| `BindExecutor(b)` | Registers an executor with no edges (for targets reached only through `SendMessage` with a target ID, or ports) |

| Edge option | Effect |
| --- | --- |
| `WithEdgeCondition[T](func(T) bool)` | Condition on a direct edge. Messages are decoded as `T`; a message that is not a `T` arrives as the zero value, so check for that if an edge can carry several types. |
| `WithEdgeCondition0(func() bool)` | Condition that ignores the message |
| `WithEdgeAssigner[T](func(targetCount int, msg T) iter.Seq[int])` | Fan-out target selection by index |
| `WithEdgeLabel(label)` | A label for visualisation (`ReflectEdges` returns it) |
| `IdempotentEdge()` | Adding the same unconditional edge twice is a no-op instead of an error |

### Conditional edges

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type Ticket struct {
	Text   string
	Urgent bool
}

func main() {
	classify := workflow.NewExecutor("classify", func(text string) Ticket {
		return Ticket{Text: text, Urgent: strings.Contains(strings.ToLower(text), "outage")}
	}).Bind()
	page := workflow.NewExecutor("page_on_call", func(t Ticket) string {
		return "Paged on-call: " + t.Text
	}).Bind()
	queue := workflow.NewExecutor("queue", func(t Ticket) string {
		return "Queued for triage: " + t.Text
	}).Bind()

	wf, err := workflow.NewBuilder(classify).
		AddEdge(classify, page, workflow.WithEdgeCondition(func(t Ticket) bool { return t.Urgent })).
		AddEdge(classify, queue, workflow.WithEdgeCondition(func(t Ticket) bool { return !t.Urgent })).
		WithOutputFrom(page, queue).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	run, err := inproc.Default.Run(context.Background(), wf, "Total outage in eu-west")
	if err != nil {
		log.Fatal(err)
	}
	for evt := range run.NewEvents() {
		switch e := evt.(type) {
		case workflow.ExecutorCompletedEvent:
			fmt.Printf("[%s] done\n", e.ExecutorID)
		case workflow.OutputEvent:
			fmt.Println("output:", e.Output)
		}
	}
}
```

Output:

```text
[classify] done
[page_on_call] done
output: Paged on-call: Total outage in eu-west
```

### Switch, fan-out and fan-in

A switch is a fan-out edge with a generated assigner. Its predicates receive the message as `any`, so assert the type yourself. The next program routes a purchase request by value with a switch, fans an approved request out to three suppliers, and fans their quotes back in to an aggregating executor:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type Purchase struct {
	Item   string
	Budget float64
}

type Quote struct {
	Supplier string
	Price    float64
}

type Best struct {
	Seen  int
	Quote Quote
}

func supplier(name string, price float64) workflow.ExecutorBinding {
	return workflow.NewExecutor(name, func(p Purchase) Quote { return Quote{Supplier: name, Price: price} }).Bind()
}

func buildWorkflow() *workflow.Workflow {
	start := workflow.NewExecutor("start", func(p Purchase) Purchase { return p }).Bind()
	reject := workflow.NewExecutor("reject", func(p Purchase) string {
		return fmt.Sprintf("rejected %s: budget £%.2f is over the limit", p.Item, p.Budget)
	}).Bind()
	tender := workflow.NewExecutor("tender", func(p Purchase) Purchase { return p }).Bind()
	acme, globex, initech := supplier("acme", 12.5), supplier("globex", 9.75), supplier("initech", 11)

	// NewAggregatingExecutor folds each incoming message into its state and
	// sends the aggregate on after every message.
	cheapest := workflow.NewAggregatingExecutor("cheapest", func(best *Best, q Quote) *Best {
		if best == nil {
			best = &Best{Quote: q}
		} else if q.Price < best.Quote.Price {
			best.Quote = q
		}
		best.Seen++
		return best
	}).Bind()
	announce := workflow.NewExecutor("announce", func(b Best) string {
		return fmt.Sprintf("%s wins at £%.2f", b.Quote.Supplier, b.Quote.Price)
	}).Bind()

	suppliers := []workflow.ExecutorBinding{acme, globex, initech}
	b := workflow.NewBuilder(start)
	b.AddSwitch(start).
		AddCase(func(msg any) bool { return msg.(Purchase).Budget > 10_000 }, reject).
		WithDefault(tender).
		AddToBuilder(b).
		AddFanOutEdge(tender, suppliers).
		AddFanInBarrierEdge(suppliers, cheapest).
		AddEdge(cheapest, announce, workflow.WithEdgeCondition(func(b Best) bool { return b.Seen == len(suppliers) })).
		WithOutputFrom(reject, announce)
	wf, err := b.Build()
	if err != nil {
		log.Fatal(err)
	}
	return wf
}

func main() {
	for _, p := range []Purchase{{"widgets", 500}, {"forklift", 25_000}} {
		// A fresh workflow per run: see "Bindings and per-run state".
		run, err := inproc.Default.Run(context.Background(), buildWorkflow(), p)
		if err != nil {
			log.Fatal(err)
		}
		for evt := range run.NewEvents() {
			if o, ok := evt.(workflow.OutputEvent); ok {
				fmt.Println(o.Output)
			}
		}
		_ = run.Close(context.Background())
	}
}
```

Output:

```text
globex wins at £9.75
rejected forklift: budget £25000.00 is over the limit
```

Fan-in in Go differs from Python. The barrier waits for every source, then invokes the target **once per message** in the same superstep, in no fixed order. The target does not receive a list. Reduce the messages with `NewAggregatingExecutor` (whose state is saved in checkpoints), with a struct executor that collects them and acts in `OnMessageDeliveryFinishedFunc`, or with a condition on the next edge, as here. Note that `NewAggregatingExecutor` takes a pointer but sends the aggregate as a value, so the next executor accepts `Best`, not `*Best`.

For a multi-selection fan-out, where one message may go to several targets, pass an assigner. It yields target indexes in the order of the targets slice:

```go
b.AddFanOutEdge(analyse, []workflow.ExecutorBinding{spam, reply, summarise},
	workflow.WithEdgeAssigner(func(_ int, r Analysis) iter.Seq[int] {
		return func(yield func(int) bool) {
			if r.Spam {
				yield(0)
				return
			}
			if yield(1) && len(r.Email) > 100 {
				yield(2)
			}
		}
	}))
```

### Validation

`Build` checks the graph and returns the first problem it found. These are the errors from v%%version=github.com/microsoft/agent-framework-go%%:

| Problem | Error |
| --- | --- |
| An edge whose source sends no type the target accepts | `type incompatibility between executors "a" -> "c": source sends [string] but target accepts [int]` |
| An executor that is bound but unreachable from the start | `workflow cannot be built because there are orphaned executors: [d]` |
| The same unconditional edge twice | `an edge from 'a' to 'b' already exists without a condition` |
| An output executor that is not in the graph | `output executor "x" is not present in the workflow graph` |
| Two different executors with one ID | `cannot bind executor with ID "x" because an executor with the same ID but a different implementation ID … is already bound` |

`BuildWithOptions(workflow.BuildOptions{DisableOrphanValidation: true})` skips the orphan check only. The builder also logs through `slog.Default()`: an `INFO` line listing dead-end executors (no outgoing edges) and a `WARN` line for a self-loop. Neither is an error; route `slog.Default()` elsewhere if they are noise.

## Loops

Edges may form cycles, and an executor may have an edge to itself. **There is no superstep limit in Go** (Python stops at `max_iterations`, 100 by default), so a loop runs until it stops sending messages, calls `RequestHalt`, or its context is cancelled. Give long or model-driven loops an explicit counter and a deadline.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

// Page asks for one page of a paginated API.
type Page struct {
	Number int
	Seen   int
}

// fetcher pretends to call an API that has 23 records, 10 per page.
type fetcher struct {
	_ workflow.AttrSendsMessage[Page]
	_ workflow.AttrYieldsOutput[string]
}

func (fetcher) Handle(ctx *workflow.Context, p Page) error {
	const total, perPage = 23, 10
	got := min(perPage, total-p.Seen)
	if err := ctx.YieldOutput(fmt.Sprintf("page %d: %d records", p.Number, got)); err != nil {
		return err
	}
	if p.Seen+got >= total || p.Number >= 10 { // a hard cap as well as the natural end
		return ctx.YieldOutput(fmt.Sprintf("done: %d records", p.Seen+got))
	}
	return ctx.SendMessage("", Page{Number: p.Number + 1, Seen: p.Seen + got})
}

func main() {
	fetch := workflow.NewExecutor("fetch", fetcher{}).Bind()
	wf, err := workflow.NewBuilder(fetch).AddEdge(fetch, fetch).WithOutputFrom(fetch).Build()
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	run, err := inproc.Default.Run(ctx, wf, Page{Number: 1})
	if err != nil {
		log.Fatal(err)
	}
	for evt := range run.NewEvents() {
		if o, ok := evt.(workflow.OutputEvent); ok {
			fmt.Println(o.Output)
		}
	}
}
```

Output (plus the builder's self-loop warning on standard error):

```text
page 1: 10 records
page 2: 10 records
page 3: 3 records
done: 23 records
```

## Running a workflow

An `inproc.ExecutionEnvironment` runs workflows. The package provides four:

| Environment | Behaviour |
| --- | --- |
| `inproc.Default` | The same as `OffThread` |
| `inproc.OffThread` | Runs supersteps on a background goroutine and streams events as they are raised |
| `inproc.Concurrent` | Like `OffThread`, but several runs of one workflow may execute at once. Every executor must be shareable across runs (see below). |
| `inproc.Lockstep` | Runs supersteps on the goroutine that reads events and releases each superstep's events when it completes |

`env.WithCheckpointing(manager)` returns a copy of any of them that checkpoints after every superstep.

Each environment starts runs in one of two shapes:

| Method | Returns | Use it for |
| --- | --- | --- |
| `Run(ctx, wf, msg, opts...)` | `*inproc.Run` | Batch work. It runs to the first halt and accumulates the events. |
| `RunStreaming(ctx, wf, msg, opts...)` | `*inproc.StreamingRun` | Reacting to events as they happen, sending responses, cancelling |
| `OpenStreaming(ctx, wf, opts...)` | `*inproc.StreamingRun` | A run with no first message; send inputs later with `TrySendMessage` |
| `Resume` / `ResumeStreaming(ctx, wf, checkpointInfo, opts...)` | as above | Continuing from a checkpoint |

`inproc.WithSessionID(id)` names the run's session; otherwise a UUID is generated. `inproc.WithPendingRequestRepublish(false)` stops a resumed run re-emitting the requests that were outstanding when the checkpoint was taken (they are re-emitted by default).

On an `*inproc.Run`, `NewEvents()` iterates the events raised since the last call and `OutgoingEvents()` all events so far. `Resume(ctx, msgs...)` sends more input (for example an `*workflow.ExternalResponse`) and runs to the next halt, and `RunToNextHalt` continues without new input.

On an `*inproc.StreamingRun`:

| Method | Notes |
| --- | --- |
| `WatchStream(ctx)` | Iterates events and errors. It stays open while requests are pending, waiting for you to answer them, so it suits interactive loops. |
| `WatchUntilHalt(ctx)` | Iterates until the run halts, without waiting on pending requests. Use it when you will answer later, from another process. |
| `TrySendMessage(ctx, msg)` | Sends input. It returns `false` with no error when the start executor does not accept the type. |
| `SendResponse(ctx, resp)` | Answers a pending `ExternalRequest` |
| `CancelRun()`, `Close(ctx)` | Stop the run; always `defer run.Close(ctx)` |
| `GetStatus`, `SessionID`, `Checkpoints`, `LastCheckpoint`, `RestoreCheckpoint` | Inspection and time travel |

Only one goroutine may watch a stream at a time. A run that has halted is `RunStatusIdle` and accepts more input, so one streaming run can serve a sequence of requests:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

func main() {
	ctx := context.Background()
	shout := workflow.NewExecutor("shout", func(s string) string { return strings.ToUpper(s) + "!" }).Bind()
	wf, err := workflow.NewBuilder(shout).WithOutputFrom(shout).Build()
	if err != nil {
		log.Fatal(err)
	}

	run, err := inproc.Default.OpenStreaming(ctx, wf, inproc.WithSessionID("demo-session"))
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()

	ok, err := run.TrySendMessage(ctx, 42)
	fmt.Println("accepted an int:", ok, err)

	for _, in := range []string{"hello", "again"} {
		if _, err := run.TrySendMessage(ctx, in); err != nil {
			log.Fatal(err)
		}
		for evt, err := range run.WatchUntilHalt(ctx) {
			if err != nil {
				log.Fatal(err)
			}
			if o, ok := evt.(workflow.OutputEvent); ok {
				fmt.Println("output:", o.Output)
			}
		}
		status, _ := run.GetStatus(ctx)
		fmt.Println("status idle:", status == inproc.RunStatusIdle, "session:", run.SessionID())
	}
}
```

Output:

```text
accepted an int: false <nil>
output: HELLO!
status idle: true session: demo-session
output: AGAIN!
status idle: true session: demo-session
```

The run statuses are `RunStatusNotStarted`, `RunStatusIdle` (halted, ready for input), `RunStatusPendingRequests` (halted, waiting for responses), `RunStatusRunning` and `RunStatusEnded`. `RunStatus` has no `String` method, so it prints as a number.

### Running one workflow concurrently

`inproc.Concurrent` lets several runs share one `*workflow.Workflow`, but only if every executor is safe for that. Mark a stateless shared executor with `SetCrossRunShareable(true)`. For a factory binding, set `SupportsConcurrentSharedExecution` on the binding; the factory then gives each run its own instance. Otherwise the run fails with `workflow must only consist of cross-run share-capable or factory-created executors`.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

func main() {
	upper := workflow.BindNewExecutorFunc("upper", func(_, id string) (*workflow.Executor, error) {
		return workflow.NewExecutor(id, strings.ToUpper), nil
	})
	upper.SupportsConcurrentSharedExecution = true
	suffix := workflow.NewExecutor("suffix", func(s string) string { return s + "!" }).
		SetCrossRunShareable(true).
		Bind()

	wf, err := workflow.NewBuilder(upper).AddEdge(upper, suffix).WithOutputFrom(suffix).Build()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("AllowConcurrent:", wf.AllowConcurrent())

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		results []string
	)
	for _, in := range []string{"a", "b", "c"} {
		wg.Go(func() {
			run, err := inproc.Concurrent.Run(context.Background(), wf, in)
			if err != nil {
				log.Fatal(err)
			}
			for evt := range run.NewEvents() {
				if o, ok := evt.(workflow.OutputEvent); ok {
					mu.Lock()
					results = append(results, o.Output.(string))
					mu.Unlock()
				}
			}
		})
	}
	wg.Wait()
	fmt.Println(len(results), "runs finished")
}
```

Output:

```text
AllowConcurrent: true
3 runs finished
```

## Outputs

Only executors registered with `WithOutputFrom` (terminal outputs) or `WithIntermediateOutputFrom` (progress) produce `OutputEvent`s. An executor that is not registered can still return values and send messages, but nothing it yields reaches the caller. `OutputEvent.IsIntermediate()` (or `HasTag(workflow.OutputTagIntermediate)`) tells the two apart, and `OutputEvent.ExecutorID` names the source.

A run can produce any number of outputs, and producing an output does not end the run. The run ends when there is no more work. To stop early, call `ctx.RequestHalt()`.

## Shared state

State lives in **scopes**. Each executor has a private default scope (pass `""` as the scope name), and any named scope is shared by every executor that uses the name.

Writes are queued and applied at the end of the superstep. The executor that wrote a value reads it back immediately. Other executors running in the same superstep do not see it until the next superstep. Two executors that write the same key in one superstep make the run fail with `expected exactly one update for key 'k'`, so give parallel branches their own keys. Reading a key that does not exist returns `nil, nil`.

State is saved in checkpoints, so it must survive JSON. Structs and the other named types you write are restored as the same Go type, because the engine records a type ID with each value.

`workflow.StatefulExecutorCache[T]` wraps the read-modify-write cycle with a typed value and an initial-value factory:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

const docScope = "documents"

type Stats struct {
	Words, Lines int
}

func main() {
	// load stores the document once in a shared scope and passes on a key.
	load := workflow.NewExecutor("load", func(ctx *workflow.Context, name string) (string, error) {
		text := "Agents call tools.\nWorkflows route messages.\nCheckpoints save state."
		return name, ctx.QueueStateUpdate(name, docScope, text)
	}).Bind()

	read := func(ctx *workflow.Context, key string) (string, error) {
		v, err := ctx.ReadState(key, docScope)
		if err != nil {
			return "", err
		}
		s, _ := v.(string)
		return s, nil
	}
	words := workflow.NewExecutor("words", func(ctx *workflow.Context, key string) (Stats, error) {
		text, err := read(ctx, key)
		return Stats{Words: len(strings.Fields(text))}, err
	}).Bind()
	lines := workflow.NewExecutor("lines", func(ctx *workflow.Context, key string) (Stats, error) {
		text, err := read(ctx, key)
		return Stats{Lines: strings.Count(text, "\n") + 1}, err
	}).Bind()

	// total keeps a running sum in its private scope through a typed cache.
	cache := &workflow.StatefulExecutorCache[Stats]{
		StateKey:            "running_total",
		InitialStateFactory: func() Stats { return Stats{} },
	}
	total := workflow.NewExecutor("total", func(ctx *workflow.Context, s Stats) (string, error) {
		var out Stats
		err := cache.InvokeWithState(ctx, false, func(_ *workflow.Context, acc Stats) (Stats, error) {
			acc.Words += s.Words
			acc.Lines += s.Lines
			out = acc
			return acc, nil
		})
		return fmt.Sprintf("so far: %d words, %d lines", out.Words, out.Lines), err
	}).Extend(&workflow.Executor{
		ResetFunc:                cache.Reset,
		OnCheckpointRestoredFunc: cache.OnCheckpointRestored,
	}).Bind()

	branches := []workflow.ExecutorBinding{words, lines}
	wf, err := workflow.NewBuilder(load).
		AddFanOutEdge(load, branches).
		AddFanInBarrierEdge(branches, total).
		WithOutputFrom(total).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	run, err := inproc.Default.Run(context.Background(), wf, "notes.txt")
	if err != nil {
		log.Fatal(err)
	}
	var last string
	for evt := range run.NewEvents() {
		switch e := evt.(type) {
		case workflow.OutputEvent:
			last = e.Output.(string)
		case workflow.ErrorEvent:
			log.Fatal(e.Error)
		}
	}
	fmt.Println(last)
}
```

Output:

```text
so far: 9 words, 3 lines
```

`ReadOrInitState(key, scope, init)` initialises a missing value, `ReadStateKeys(scope)` iterates the keys in a scope, and `QueueClearScope(scope)` empties one. The cache keeps a copy in the executor between reads; it skips that copy when concurrent runs are enabled, or when you pass `skipCache` as `true`. Wire `OnCheckpointRestored` as shown, so a restore does not leave a stale cached value.

## Events

Every run produces a stream of events. They all implement `workflow.Event`, so a type switch reads them:

| Event | When | Useful fields |
| --- | --- | --- |
| `StartedEvent` | Before the first superstep of each input-to-halt cycle | `Message` |
| `SuperStepStartedEvent` | A superstep begins | `StepNumber`, `StartInfo` |
| `ExecutorInvokedEvent` | A handler is about to run | `ExecutorID`, `Message` |
| `ExecutorCompletedEvent` | A handler returned | `ExecutorID`, `Result` |
| `ExecutorFailedEvent` | A handler returned an error | `ExecutorID`, `Error` |
| `OutputEvent` | An output source yielded | `ExecutorID`, `Output`, `Tags` |
| `RequestInfoEvent` | An external request is waiting | `Request` |
| `SuperStepCompletedEvent` | A superstep ended | `StepNumber`, `CompletionInfo` (activated executors, pending messages and requests, `CheckpointInfo`) |
| `RequestHaltEvent` | A halt was requested | `Result` |
| `WorkflowWarningEvent` | A non-fatal problem | `Message`, `SubWorkflowID` |
| `ErrorEvent` | A workflow-level failure | `Error`, `SubWorkflowID` |

Always handle `ExecutorFailedEvent` and `ErrorEvent`. A failure does not appear as an error from `Run`, and without those cases a failing executor just ends the stream with no output.

`ctx.AddEvent` raises your own events, which is the usual way to report progress from inside a long executor:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

// ProgressEvent is a custom event; anything with Data() any is an Event.
type ProgressEvent struct {
	Stage string
	Done  int
	Total int
}

func (e ProgressEvent) Data() any { return e }

func main() {
	clean := workflow.NewExecutor("clean", func(ctx *workflow.Context, rows []string) ([]string, error) {
		out := make([]string, 0, len(rows))
		for i, r := range rows {
			out = append(out, strings.TrimSpace(r))
			if err := ctx.AddEvent(ProgressEvent{Stage: "clean", Done: i + 1, Total: len(rows)}); err != nil {
				return nil, err
			}
		}
		return out, nil
	}).Bind()
	count := workflow.NewExecutor("count", func(rows []string) string {
		return fmt.Sprintf("%d rows loaded", len(rows))
	}).Bind()

	wf, err := workflow.NewBuilder(clean).AddEdge(clean, count).
		WithIntermediateOutputFrom(clean).
		WithOutputFrom(count).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	run, err := inproc.Default.RunStreaming(ctx, wf, []string{" a ", "b ", " c"})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.SuperStepStartedEvent:
			fmt.Println("superstep", e.StepNumber)
		case ProgressEvent:
			fmt.Printf("  %s %d/%d\n", e.Stage, e.Done, e.Total)
		case workflow.OutputEvent:
			if e.IsIntermediate() {
				fmt.Printf("  intermediate from %s: %q\n", e.ExecutorID, e.Output)
				continue
			}
			fmt.Println("final:", e.Output)
			return
		case workflow.ExecutorFailedEvent:
			log.Fatalf("%s failed: %v", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			log.Fatal(e.Error)
		}
	}
}
```

Output:

```text
superstep 0
  clean 1/3
  clean 2/3
  clean 3/3
  intermediate from clean: ["a" "b" "c"]
superstep 1
final: 3 rows loaded
```

Events raised during a superstep are released together when the superstep ends, in the order they were raised.

## Human in the loop

A workflow asks the outside world for input through an `ExternalRequest`, which surfaces as a `RequestInfoEvent`. You answer with `request.CreateResponse(value)`, which checks the value against the port's response type, and deliver the response with `StreamingRun.SendResponse` (or `Run.Resume` on a non-streaming run). The response reaches the executor that asked. A run with unanswered requests halts with `RunStatusPendingRequests`, and stays resumable for as long as you keep it, or its checkpoint.

There are two ways to raise a request.

**A request port in the graph.** A `workflow.RequestPort` with an ID, a request type and a response type becomes a node with `port.Bind()`. Messages of the request type that reach it become requests, and the responses flow on along its outgoing edges:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"reflect"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/checkpoint"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type Refund struct {
	Order  string
	Amount float64
}

// review is a struct executor: Handle receives the typed message plus a
// *workflow.Context for sending messages and yielding output.
type review struct {
	_ workflow.AttrSendsMessage[Refund]
	_ workflow.AttrYieldsOutput[string]
}

func (review) Handle(ctx *workflow.Context, r Refund) error {
	if r.Amount < 100 {
		return ctx.YieldOutput(fmt.Sprintf("auto-approved %s (£%.2f)", r.Order, r.Amount))
	}
	return ctx.SendMessage("", r) // large refunds go to a human
}

func main() {
	approvalPort := workflow.RequestPort{
		ID:       "manager_approval",
		Request:  reflect.TypeFor[Refund](),
		Response: reflect.TypeFor[bool](),
	}
	ask := approvalPort.Bind()
	rev := workflow.NewExecutor("review", review{}).Bind()
	decide := workflow.NewExecutor("decide", func(ok bool) string {
		if ok {
			return "refund issued"
		}
		return "refund declined"
	}).Bind()

	wf, err := workflow.NewBuilder(rev).
		AddEdge(rev, ask).
		AddEdge(ask, decide).
		WithOutputFrom(rev, decide).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	env := inproc.Default.WithCheckpointing(checkpoint.NewInMemoryManager())
	run, err := env.RunStreaming(ctx, wf, Refund{Order: "A-1009", Amount: 249})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()

	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.RequestInfoEvent:
			r, _ := workflow.PortableValueAs[Refund](e.Request.Data)
			fmt.Printf("approval needed for %s (£%.2f)\n", r.Order, r.Amount)
			resp, err := e.Request.CreateResponse(true) // a real app would ask a person
			if err != nil {
				log.Fatal(err)
			}
			if err := run.SendResponse(ctx, resp); err != nil {
				log.Fatal(err)
			}
		case workflow.SuperStepCompletedEvent:
			if e.CompletionInfo != nil && e.CompletionInfo.CheckpointInfo != nil {
				fmt.Println("checkpoint saved")
			}
		case workflow.OutputEvent:
			fmt.Println("output:", e.Output)
			return
		case workflow.ExecutorFailedEvent:
			log.Fatalf("%s failed: %v", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			log.Fatal(e.Error)
		}
	}
}
```

Output:

```text
checkpoint saved
approval needed for A-1009 (£249.00)
checkpoint saved
checkpoint saved
output: refund issued
```

`workflow.PortableValueAs[T]` reads a request's payload. `e.Request.PortInfo.PortID` says which port asked, which matters once a workflow has several.

**A request raised from code.** `ctx.PostRequest` raises a request from inside an executor, and the response comes back to that same executor as a `*workflow.ExternalResponse` message. This keeps the question and the follow-up in one place. It needs a raw executor, because the executor must handle `*workflow.ExternalResponse`. This version uses the non-streaming `Run`, which suits a request that is answered later by a separate call:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"reflect"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type Deploy struct {
	Service string
	Version string
}

var confirmPort = workflow.RequestPort{
	ID:       "confirm_deploy",
	Request:  reflect.TypeFor[Deploy](),
	Response: reflect.TypeFor[string](),
}

func newGate() workflow.ExecutorBinding {
	b := workflow.BindNewExecutorFunc("gate", func(_, id string) (*workflow.Executor, error) {
		return &workflow.Executor{
			ID:                                 id,
			AutoSendMessageHandlerResultObject: new(false),
			AutoYieldOutputHandlerResultObject: new(false),
			ConfigureProtocol: func(pb *workflow.ProtocolBuilder) (*workflow.ProtocolBuilder, error) {
				pb.YieldsOutputType(reflect.TypeFor[string]())
				pb.RouteBuilder.
					AddHandlerRaw(reflect.TypeFor[Deploy](), nil, func(ctx *workflow.Context, msg any) (any, error) {
						req, err := workflow.NewExternalRequest("", confirmPort, msg.(Deploy))
						if err != nil {
							return nil, err
						}
						return nil, ctx.PostRequest(req)
					}).
					AddHandlerRaw(reflect.TypeFor[*workflow.ExternalResponse](), nil, func(ctx *workflow.Context, msg any) (any, error) {
						who, _ := workflow.PortableValueAs[string](msg.(*workflow.ExternalResponse).Data)
						return nil, ctx.YieldOutput("deploy confirmed by " + who)
					})
				return pb, nil
			},
		}, nil
	})
	b.Ports = []workflow.RequestPort{confirmPort} // lists the port in ReflectPorts
	return b
}

func main() {
	ctx := context.Background()
	gate := newGate()
	wf, err := workflow.NewBuilder(gate).WithOutputFrom(gate).Build()
	if err != nil {
		log.Fatal(err)
	}

	run, err := inproc.Default.Run(ctx, wf, Deploy{Service: "billing", Version: "2.4.0"})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()

	var pending *workflow.ExternalRequest
	for evt := range run.NewEvents() {
		if e, ok := evt.(workflow.RequestInfoEvent); ok {
			pending = e.Request
		}
	}
	status, _ := run.GetStatus(ctx)
	fmt.Println("waiting for a response:", status == inproc.RunStatusPendingRequests)
	d, _ := workflow.PortableValueAs[Deploy](pending.Data)
	fmt.Printf("asked to confirm %s %s on port %s\n", d.Service, d.Version, pending.PortInfo.PortID)

	resp, err := pending.CreateResponse("ana@example.com")
	if err != nil {
		log.Fatal(err)
	}
	if _, err := run.Resume(ctx, resp); err != nil {
		log.Fatal(err)
	}
	for evt := range run.NewEvents() {
		if o, ok := evt.(workflow.OutputEvent); ok {
			fmt.Println(o.Output)
		}
	}
	fmt.Println("ports:", len(wf.ReflectPorts()))
}
```

Output:

```text
waiting for a response: true
asked to confirm billing 2.4.0 on port confirm_deploy
deploy confirmed by ana@example.com
ports: 1
```

`CreateResponse` returns an error if the value is not of the port's response type, and `NewExternalRequest` does the same for the request type. Agents hosted in a workflow use the same mechanism for tool approvals and for function calls that the workflow cannot run itself; see [agents in workflows](#agents-in-workflows).

## Checkpointing

An environment made with `WithCheckpointing(manager)` saves a checkpoint at the end of every superstep. A checkpoint holds the queued messages, shared and private state, pending requests and output bookkeeping, and is identified by a `workflow.CheckpointInfo` (`SessionID` plus `CheckpointID`). You get them from `SuperStepCompletedEvent.CompletionInfo.CheckpointInfo`, `run.Checkpoints()`, `run.LastCheckpoint()`, or `manager.LatestCheckpoint(ctx, sessionID)`.

| Manager | Storage |
| --- | --- |
| `checkpoint.NewInMemoryManager()` | Process memory; for tests and in-process time travel |
| `checkpoint.NewJSONManager(store)` | Serialises to JSON and hands the bytes to a `checkpoint.Store[json.RawMessage]` |
| `checkpoint.NewFileSystemJSONStore(dir)` | A store that writes one file per checkpoint plus an `index.jsonl`, and holds an exclusive lock on the directory until `Close` |

### Resuming in a new process

The usual reason to checkpoint is to stop a run that is waiting for a person and finish it later, possibly in a different process. The program below does both halves. Phase one runs until the approval request and closes everything. Phase two reopens the store, finds the latest checkpoint for the session, and resumes a **freshly built** workflow from it; the pending request is re-emitted, answered, and the run completes:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"reflect"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/checkpoint"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type Refund struct {
	Order  string
	Amount float64
}

type Audit struct {
	ReceivedBy string
}

// buildWorkflow must produce the same topology and executor IDs every time:
// a checkpoint can only be resumed into an identical graph.
func buildWorkflow() *workflow.Workflow {
	port := workflow.RequestPort{ID: "approval", Request: reflect.TypeFor[Refund](), Response: reflect.TypeFor[bool]()}
	ask := port.Bind()
	intake := workflow.NewExecutor("intake", func(ctx *workflow.Context, r Refund) (Refund, error) {
		return r, ctx.QueueStateUpdate("audit", "case", Audit{ReceivedBy: "intake-v1"})
	}).Bind()
	decide := workflow.NewExecutor("decide", func(ctx *workflow.Context, approved bool) (string, error) {
		v, err := ctx.ReadState("audit", "case")
		if err != nil {
			return "", err
		}
		a := v.(Audit) // restored as the same Go type
		return fmt.Sprintf("approved=%t (received by %s)", approved, a.ReceivedBy), nil
	}).Bind()
	wf, err := workflow.NewBuilder(intake).AddEdge(intake, ask).AddEdge(ask, decide).WithOutputFrom(decide).Build()
	if err != nil {
		log.Fatal(err)
	}
	return wf
}

// phaseOne runs until the workflow needs a human, then shuts down.
func phaseOne(ctx context.Context, dir string) string {
	store, err := checkpoint.NewFileSystemJSONStore(dir)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = store.Close() }() // releases the directory lock

	env := inproc.Default.WithCheckpointing(checkpoint.NewJSONManager(store))
	run, err := env.RunStreaming(ctx, buildWorkflow(), Refund{Order: "A-1", Amount: 250})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	for evt, err := range run.WatchUntilHalt(ctx) { // does not block on the request
		if err != nil {
			log.Fatal(err)
		}
		if e, ok := evt.(workflow.RequestInfoEvent); ok {
			fmt.Println("phase 1: waiting on", e.Request.PortInfo.PortID)
		}
	}
	fmt.Println("phase 1: checkpoints written:", len(run.Checkpoints()))
	return run.SessionID()
}

// phaseTwo could run in another process, hours later.
func phaseTwo(ctx context.Context, dir, sessionID string) {
	store, err := checkpoint.NewFileSystemJSONStore(dir)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = store.Close() }()
	mgr := checkpoint.NewJSONManager(store)

	latest, err := mgr.LatestCheckpoint(ctx, sessionID)
	if err != nil || latest == nil {
		log.Fatalf("no checkpoint for %s: %v", sessionID, err)
	}
	run, err := inproc.Default.WithCheckpointing(mgr).ResumeStreaming(ctx, buildWorkflow(), *latest)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.RequestInfoEvent: // re-emitted on resume
			r, _ := workflow.PortableValueAs[Refund](e.Request.Data)
			fmt.Printf("phase 2: approving %s (£%.2f)\n", r.Order, r.Amount)
			resp, err := e.Request.CreateResponse(true)
			if err != nil {
				log.Fatal(err)
			}
			if err := run.SendResponse(ctx, resp); err != nil {
				log.Fatal(err)
			}
		case workflow.OutputEvent:
			fmt.Println("phase 2:", e.Output)
			return
		case workflow.ErrorEvent:
			log.Fatal(e.Error)
		}
	}
}

func main() {
	ctx := context.Background()
	dir, err := os.MkdirTemp("", "refund-checkpoints")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	sessionID := phaseOne(ctx, dir)
	phaseTwo(ctx, dir, sessionID)
}
```

Output:

```text
phase 1: waiting on approval
phase 1: checkpoints written: 2
phase 2: approving A-1 (£250.00)
phase 2: approved=true (received by intake-v1)
```

Rules for resuming:

- **Rebuild the same graph.** Executor IDs, edges and ports must match. For agents hosted with `agentworkflow.New`, the executor ID is derived from the agent's name and ID, and the ID is a random UUID unless you set `agent.Config.ID`. Set it, or a rebuilt workflow will not match its checkpoints.
- **Keep executor fields in state.** Checkpoints contain workflow state, not your struct fields or captured variables. An executor that keeps counters in fields must copy them into state in `OnCheckpointFunc` and back in `OnCheckpointRestoredFunc`, or keep them in state from the start (`StatefulExecutorCache`, `NewAggregatingExecutor`).
- **One process per directory.** `NewFileSystemJSONStore` fails with `store at "…" is already in use` while another store, even in the same process, holds the directory. Use your own `Store` for anything shared between replicas.
- **A superstep is the unit of replay.** A resumed run repeats the superstep that was in progress when the checkpoint was taken, including any agent calls and tool calls in it. Make side effects idempotent.

Executor-local fields are saved with the hooks:

```go
type judge struct {
	_     workflow.AttrSendsMessage[Signal]
	_     workflow.AttrYieldsOutput[string]
	tries int
}

func (j *judge) OnCheckpoint(ctx *workflow.Context) error {
	return ctx.QueueStateUpdate("tries", "", j.tries) // "" = this executor's private scope
}

func (j *judge) OnCheckpointRestored(ctx *workflow.Context) error {
	v, err := ctx.ReadState("tries", "")
	if err != nil {
		return err
	}
	j.tries, _ = v.(int)
	return nil
}

func (j *judge) Reset() error { j.tries = 0; return nil }

func newJudge() *workflow.Executor {
	j := &judge{}
	return workflow.NewExecutor("judge", j).Extend(&workflow.Executor{
		ResetFunc:                j.Reset,
		OnCheckpointFunc:         j.OnCheckpoint,
		OnCheckpointRestoredFunc: j.OnCheckpointRestored,
	})
}
```

### Time travel and custom stores

`run.RestoreCheckpoint(ctx, info)` rewinds a live run to any earlier checkpoint of the same session; watch the stream again to replay from there. A custom store implements three methods. This one keeps checkpoints in a map, standing in for a database table keyed by session, and is used to rewind a run to its first checkpoint:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	"github.com/google/uuid"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/checkpoint"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

// mapStore implements checkpoint.Store[json.RawMessage].
type mapStore struct {
	mu    sync.Mutex
	data  map[string]json.RawMessage
	index map[string][]workflow.CheckpointInfo
	links map[string]string // checkpoint ID -> parent ID
}

func newMapStore() *mapStore {
	return &mapStore{data: map[string]json.RawMessage{}, index: map[string][]workflow.CheckpointInfo{}, links: map[string]string{}}
}

func (s *mapStore) CreateCheckpoint(_ context.Context, sessionID string, data json.RawMessage, parent *workflow.CheckpointInfo) (workflow.CheckpointInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	info := workflow.CheckpointInfo{SessionID: sessionID, CheckpointID: uuid.NewString()}
	s.data[info.CheckpointID] = append(json.RawMessage(nil), data...)
	s.index[sessionID] = append(s.index[sessionID], info)
	if parent != nil {
		s.links[info.CheckpointID] = parent.CheckpointID
	}
	return info, nil
}

func (s *mapStore) RetrieveCheckpoint(_ context.Context, _ string, info workflow.CheckpointInfo) (json.RawMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	d, ok := s.data[info.CheckpointID]
	if !ok {
		return nil, fmt.Errorf("checkpoint %s not found", info.CheckpointID)
	}
	return d, nil
}

func (s *mapStore) RetrieveIndex(_ context.Context, sessionID string, withParent *workflow.CheckpointInfo) ([]workflow.CheckpointInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []workflow.CheckpointInfo
	for _, info := range s.index[sessionID] {
		if withParent == nil || s.links[info.CheckpointID] == withParent.CheckpointID {
			out = append(out, info)
		}
	}
	return out, nil
}

func main() {
	ctx := context.Background()
	step := workflow.NewExecutor("step", func(n int) int { return n + 1 }).Bind()
	report := workflow.NewExecutor("report", func(n int) string { return fmt.Sprint("result ", n) }).Bind()
	wf, err := workflow.NewBuilder(step).AddEdge(step, report).WithOutputFrom(report).Build()
	if err != nil {
		log.Fatal(err)
	}

	store := newMapStore()
	run, err := inproc.Default.WithCheckpointing(checkpoint.NewJSONManager(store)).RunStreaming(ctx, wf, 41)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	watch := func(label string) {
		for evt, err := range run.WatchUntilHalt(ctx) {
			if err != nil {
				log.Fatal(err)
			}
			if o, ok := evt.(workflow.OutputEvent); ok {
				fmt.Println(label, o.Output)
			}
		}
	}
	watch("first pass:")
	cps := run.Checkpoints()
	fmt.Println("checkpoints:", len(cps), "stored:", len(store.data))

	// Rewind to the end of superstep 0 and replay the rest.
	if err := run.RestoreCheckpoint(ctx, cps[0]); err != nil {
		log.Fatal(err)
	}
	watch("after rewind:")
}
```

Output:

```text
first pass: result 42
checkpoints: 2 stored: 2
after rewind: result 42
```

The framework serialises its own checkpoint structure before calling the store, so a store only moves bytes. Make it durable and safe for concurrent use if several runs share it.

## Sub-workflows

`inproc.BindSubworkflowAsExecutor(child, id)` turns a whole workflow into one executor of another. The child's start executor decides what the node accepts, and its output executors decide what it sends on:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

func buildNormaliser() *workflow.Workflow {
	trim := workflow.NewExecutor("trim", strings.TrimSpace).Bind()
	lower := workflow.NewExecutor("lower", strings.ToLower).Bind()
	wf, err := workflow.NewBuilder(trim).AddEdge(trim, lower).WithOutputFrom(lower).Build()
	if err != nil {
		log.Fatal(err)
	}
	return wf
}

func main() {
	normalise := inproc.BindSubworkflowAsExecutor(buildNormaliser(), "normalise")
	tag := workflow.NewExecutor("tag", func(s string) string { return "#" + strings.ReplaceAll(s, " ", "-") }).Bind()

	parent, err := workflow.NewBuilder(normalise).AddEdge(normalise, tag).WithOutputFrom(tag).Build()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	run, err := inproc.Default.RunStreaming(ctx, parent, "  Release Notes  ")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.ExecutorCompletedEvent:
			fmt.Printf("[%s] %v\n", e.ExecutorID, e.Result)
		case workflow.OutputEvent:
			fmt.Println("output:", e.Output)
			return
		case workflow.ErrorEvent:
			log.Fatal(e.Error)
		}
	}
}
```

Output:

```text
[normalise] <nil>
[trim] Release Notes
[lower] release notes
[tag] #release-notes
output: #release-notes
```

The parent's event stream includes the child's executor events as well as the sub-workflow node itself, so a trace of a nested run is complete. Only the parent's output executors produce the parent's `OutputEvent`s.

Things to know:

- **Binding takes ownership.** A child workflow can be bound into one parent at a time, so build a fresh child for every parent (and every concurrent parent run), as `buildNormaliser` does.
- **Requests bubble up with a qualified ID.** A request port inside a child appears to the parent as `<subworkflowID>.<portID>` (for example `approvals.manager`). An unanswered request becomes a top-level `RequestInfoEvent` with that ID, and the response is routed back down.
- **The parent can answer for the child.** Because the child's requests leave the sub-workflow node as `*workflow.ExternalRequest` messages, an edge from the node to an interceptor executor lets the parent answer some of them itself (with `req.CreateResponse` sent back to the node) and forward the rest to a top-level port whose ID matches the qualified one. The upstream sample `examples/03-workflows/subworkflows/request_interception` does exactly this for expense approvals, with conditions such as `func(r *workflow.ExternalRequest) bool { return r != nil }` on the edges.

## Cancellation and deadlines

A run stops when its context is cancelled, when you call `StreamingRun.CancelRun()`, or when `Close` is called. The executors' `*workflow.Context` carries the cancellation, so a slow executor should select on `ctx.Done()` (or pass `ctx` to the call it is making). After cancelling, the stream ends with `context.Canceled`, either as the iterator's error or as an `ErrorEvent`; treat that as a clean end:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type tick int

type ticker struct {
	_ workflow.AttrSendsMessage[tick]
	_ workflow.AttrYieldsOutput[string]
}

func (ticker) Handle(ctx *workflow.Context, n tick) error {
	select {
	case <-time.After(20 * time.Millisecond): // simulated slow work
	case <-ctx.Done():
		return ctx.Err()
	}
	if err := ctx.YieldOutput(fmt.Sprintf("tick %d", n)); err != nil {
		return err
	}
	return ctx.SendMessage("", n+1)
}

func main() {
	t := workflow.NewExecutor("ticker", ticker{}).Bind()
	wf, err := workflow.NewBuilder(t).AddEdge(t, t).WithOutputFrom(t).Build()
	if err != nil {
		log.Fatal(err)
	}

	// A deadline on the context bounds the whole run.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	run, err := inproc.Default.RunStreaming(ctx, wf, tick(1))
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(context.Background()) }()

	seen, cancelled := 0, false
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			if cancelled && errors.Is(err, context.Canceled) {
				break
			}
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			seen++
			fmt.Println(e.Output)
			if seen == 3 && !cancelled {
				cancelled = true
				if err := run.CancelRun(); err != nil {
					log.Fatal(err)
				}
			}
		case workflow.ErrorEvent:
			if cancelled && errors.Is(e.Error, context.Canceled) {
				fmt.Println("cancelled cleanly")
				return
			}
			log.Fatal(e.Error)
		}
	}
	fmt.Println("cancelled cleanly")
}
```

Output:

```text
tick 1
tick 2
tick 3
cancelled cleanly
```

## Agents in workflows

There are two ways to use an agent inside a graph.

**Call it from an executor.** An executor can run any `*agent.Agent` with its `*workflow.Context` as the context. You choose the prompt, parse the result (structured output works well) and send whatever type the next executor wants. This is the simplest option and the one to reach for when an agent is one step in a data pipeline:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	"github.com/openai/openai-go/v3"
)

type Triage struct {
	Category string `json:"category" jsonschema:"one of billing, technical or other"`
	Summary  string `json:"summary" jsonschema:"one sentence"`
}

func main() {
	// openai.NewClient reads OPENAI_API_KEY (and OPENAI_BASE_URL, if set).
	classifier := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Classify the support ticket and summarise it in one sentence.",
		Config:       agent.Config{Name: "classifier"},
	})

	classify := workflow.NewExecutor("classify", func(ctx *workflow.Context, ticket string) (Triage, error) {
		var t Triage
		_, err := classifier.RunText(ctx, ticket, agent.WithStructuredOutput(&t)).Collect()
		return t, err
	}).Bind()
	route := workflow.NewExecutor("route", func(t Triage) string {
		return fmt.Sprintf("queue=%s summary=%q", t.Category, t.Summary)
	}).Bind()

	wf, err := workflow.NewBuilder(classify).AddEdge(classify, route).WithOutputFrom(route).Build()
	if err != nil {
		log.Fatal(err)
	}
	run, err := inproc.Default.Run(context.Background(), wf, "I was charged twice for March.")
	if err != nil {
		log.Fatal(err)
	}
	for evt := range run.NewEvents() {
		switch e := evt.(type) {
		case workflow.OutputEvent:
			fmt.Println(e.Output)
		case workflow.ExecutorFailedEvent:
			log.Fatalf("%s failed: %v", e.ExecutorID, e.Error)
		}
	}
}
```

Output against the stand-in, which fills the schema with sample values:

```text
queue=example category summary="example summary"
```

**Host it as an executor.** `agentworkflow.New(agent, cfg)` returns a binding that speaks the chat protocol the prebuilt orchestrations use. It accepts `string`, `*message.Message`, `[]*message.Message` and `iter.Seq[*message.Message]`, and **collects** them until a `workflow.TurnToken` arrives. Then it runs the agent on everything collected and sends the resulting conversation (`[]*message.Message`) and the turn token downstream. Send a token after your messages, or the agent never runs:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	"github.com/openai/openai-go/v3"
)

// toChat turns a plain string into a chat message plus a turn token.
type toChat struct {
	_ workflow.AttrSendsMessage[*message.Message]
	_ workflow.AttrSendsMessage[workflow.TurnToken]
}

func (toChat) Handle(ctx *workflow.Context, s string) error {
	if err := ctx.SendMessage("", message.NewText(s)); err != nil {
		return err
	}
	return ctx.SendMessage("", workflow.TurnToken{EmitEvents: new(true)})
}

func main() {
	writer := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Write a two-line release note.",
		// A fixed ID keeps the executor ID stable across rebuilds (checkpoints).
		Config: agent.Config{ID: "release-writer", Name: "writer"},
	})
	host := agentworkflow.New(writer, agentworkflow.Config{
		ForwardIncomingMessages: new(false), // send only the agent's reply onward
		EmitResponseEvents:      true,       // also yield the aggregated *agent.Response
	})
	fmt.Println("executor ID:", host.ID)

	in := workflow.NewExecutor("to_chat", toChat{}).Bind()
	publish := workflow.NewExecutor("publish", func(msgs []*message.Message) string {
		return fmt.Sprintf("publishing %d message(s): %s", len(msgs), msgs[len(msgs)-1].String())
	}).Bind()

	wf, err := workflow.NewBuilder(in).AddEdge(in, host).AddEdge(host, publish).
		WithIntermediateOutputFrom(host). // the agent's updates become intermediate outputs
		WithOutputFrom(publish).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	run, err := inproc.Default.Run(context.Background(), wf, "v2.4.0 adds SSO")
	if err != nil {
		log.Fatal(err)
	}
	for evt := range run.NewEvents() {
		switch e := evt.(type) {
		case workflow.OutputEvent:
			switch out := e.Output.(type) {
			case *agent.ResponseUpdate:
				if s := out.String(); s != "" {
					fmt.Println("update:", s)
				}
			case *agent.Response:
				fmt.Println("response:", out.String())
			case string:
				fmt.Println(out)
			}
		case workflow.ExecutorFailedEvent:
			log.Fatalf("%s failed: %v", e.ExecutorID, e.Error)
		}
	}
}
```

Output against the stand-in:

```text
executor ID: writer_release_writer
update: Mock reply to: v2.4.0 adds SSO
response: Mock reply to: v2.4.0 adds SSO
publishing 1 message(s): Mock reply to: v2.4.0 adds SSO
```

`agentworkflow.Config`:

| Field | Default | Effect |
| --- | --- | --- |
| `EmitUpdateEvents` | `nil`: the turn token's `EmitEvents` decides | Yield each `*agent.ResponseUpdate` as an output while the agent streams |
| `EmitResponseEvents` | `false` | Yield the aggregated `*agent.Response` at the end of each turn |
| `ForwardIncomingMessages` | `true` | Send the incoming messages downstream before the agent's reply, so the next agent sees the whole conversation |
| `ReassignOtherAgentsAsUsers` | `true` | Rewrite other agents' assistant messages as user messages before the agent sees them |
| `InterceptUserInputRequests` | `false` | Tool approval requests become `ExternalRequest`s (false) or ordinary messages to other executors (true) |
| `InterceptUnterminatedFunctionCalls` | `false` | Function calls the agent cannot run itself become `ExternalRequest`s (false) or messages (true) |

The agent's outputs reach the caller only if the binding is listed in `WithOutputFrom` or `WithIntermediateOutputFrom`. Each turn emits a few empty updates (usage and finish markers), so skip updates whose `String()` is empty. Sending `agentworkflow.ResetSignal{}` to the executor clears its conversation.

When the hosted agent has a tool wrapped in `tool.ApprovalRequiredFunc`, the turn pauses and the workflow raises a `RequestInfoEvent` whose payload is a `*message.ToolApprovalRequestContent`. Answer it with `e.Request.CreateResponse(approvalRequest.CreateResponse(true, "reason"))`. The [orchestrations page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_orchestrations/#tool-approval-inside-an-orchestration) has a complete example.

`message/messageworkflow` holds the pieces `agentworkflow` is built from. `messageworkflow.Configure(executor, &Options{StateKey, TakeTurnHandler})` makes any raw executor collect chat messages and call your handler when a turn token arrives. `ConfigureForwarding` makes a pass-through node that relays messages and tokens. Use them to write a non-agent participant, such as a rules engine or a human proxy, that sits in an agent conversation.

## Inspecting and drawing a workflow

There is no built-in visualiser in Go (Python and .NET can export Mermaid and DOT), but a workflow describes itself. `ReflectEdges()` returns the edges by source, with labels and whether each has a condition or assigner. `ReflectExecutors()`, `ReflectPorts()`, `OutputExecutors()`, `StartExecutorID()` and `DescribeProtocol()` cover the rest. That is enough to print a Mermaid diagram:

```go
package main

import (
	"fmt"
	"log"
	"maps"
	"slices"
	"strings"

	"github.com/microsoft/agent-framework-go/workflow"
)

// mermaid renders a workflow as a Mermaid flowchart.
func mermaid(wf *workflow.Workflow) string {
	var b strings.Builder
	b.WriteString("flowchart TD\n")
	fmt.Fprintf(&b, "  %s([%s])\n", wf.StartExecutorID(), wf.StartExecutorID())
	edges := wf.ReflectEdges()
	for _, src := range slices.Sorted(maps.Keys(edges)) {
		for _, e := range edges[src] {
			arrow := "-->"
			if e.HasCondition || e.HasAssigner {
				arrow = "-.->"
			}
			label := e.Label
			if label != "" {
				label = "|" + label + "|"
			}
			for _, s := range e.Connection.SourceIDs {
				for _, t := range e.Connection.SinkIDs {
					fmt.Fprintf(&b, "  %s %s%s %s\n", s, arrow, label, t)
				}
			}
		}
	}
	for _, id := range slices.Sorted(maps.Keys(wf.OutputExecutors())) {
		fmt.Fprintf(&b, "  %s:::output\n", id)
	}
	b.WriteString("  classDef output stroke-width:3px\n")
	return b.String()
}

func main() {
	classify := workflow.NewExecutor("classify", func(s string) string { return s }).Bind()
	urgent := workflow.NewExecutor("urgent", func(s string) string { return s }).Bind()
	normal := workflow.NewExecutor("normal", func(s string) string { return s }).Bind()
	wf, err := workflow.NewBuilder(classify).
		AddEdge(classify, urgent, workflow.WithEdgeLabel("outage"),
			workflow.WithEdgeCondition(func(s string) bool { return strings.Contains(s, "outage") })).
		AddEdge(classify, normal, workflow.WithEdgeLabel("other"),
			workflow.WithEdgeCondition(func(s string) bool { return !strings.Contains(s, "outage") })).
		WithOutputFrom(urgent, normal).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(mermaid(wf))
}
```

Output:

```text
flowchart TD
  classify([classify])
  classify -.->|outage| urgent
  classify -.->|other| normal
  normal:::output
  urgent:::output
  classDef output stroke-width:3px
```

With telemetry enabled, the `workflow.build` span also records the whole definition as JSON in its `workflow.definition` attribute.

## Compared with Python

| Topic | Python | Go |
| --- | --- | --- |
| Executors | `@executor` functions and `Executor` subclasses with `@handler` | Functions, structs with `Handle`, or raw `*workflow.Executor` with typed routes |
| Type checking | Handler annotations | `reflect.Type` declarations, checked on every edge at `Build` |
| Superstep limit | `max_iterations`, 100 by default | None; use a counter, `RequestHalt` or a context deadline |
| Fan-in | Target receives a list | Target is invoked once per message; aggregate yourself |
| Running | `workflow.run(...)` / `run(stream=True)` | An `ExecutionEnvironment` (`Default`, `Concurrent`, `Lockstep`) returning `Run` or `StreamingRun` |
| Concurrent runs of one workflow | Separate workflow instances | `inproc.Concurrent` with shareable or factory executors, or one workflow per run |
| Checkpoint storage | In-memory, file and others | In-memory, file system, or your own `Store` |
| Visualisation | Mermaid and DOT export | None built in; `ReflectEdges` and friends |

**Not available in Go yet:** declarative (YAML) workflows, functional workflows (the Python `@workflow` and `@step` style), handoff and Magentic orchestration, and a workflow visualiser or DevUI.

---
title: "Microsoft Agent Framework (Go) — Workflows"
description: "Graph workflows in the Go SDK: executors and edges, conditional routing, sequential / concurrent / group-chat agent orchestration, request ports for human in the loop, checkpointing, and hosting a workflow as an agent."
framework: "microsoft-agent-framework"
language: "go"
---

The `workflow` package is a graph-execution engine. You build a graph of **executors** joined by **edges**, then run it with an execution environment from `workflow/inproc`. Execution proceeds in *supersteps*: each message delivered along an edge triggers its target executor, and the run streams events (`ExecutorCompletedEvent`, `OutputEvent`, `RequestInfoEvent` and others) as it goes.

`workflow/agentworkflow` adds agents to that graph and provides the prebuilt multi-agent patterns.

Every program on this page uses offline stand-ins instead of model-backed agents, and was compiled and run against v%%version=github.com/microsoft/agent-framework-go%%.

**Not available in Go yet:** handoff orchestration, declarative workflows and functional workflows. The patterns that do exist are sequential, concurrent, round-robin group chat, and custom graphs.

## Executors, edges and conditional routing

The simplest executor is a plain function. `workflow.NewExecutor(id, fn)` infers the input type from the parameter and forwards the return value along the outgoing edges. `.Bind()` produces the `ExecutorBinding` that the builder works with. `WithEdgeCondition` takes a typed predicate over the message, and `WithOutputFrom` marks which executors' results become workflow outputs.

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

`inproc.Default.Run` runs to completion and lets you read the events afterwards with `run.NewEvents()`. Use `RunStreaming` (below) to react to events while the run is in progress, which human in the loop requires.

Other routing options on the builder:

- `workflow.WithEdgeAssigner` — choose target indices per message, for multi-selection.
- Switch/case routing through the builder's `SwitchBuilder`.
- Fan-out and fan-in, with `workflow.NewAggregatingExecutor` for the reducing step.
- `inproc.BindSubworkflowAsExecutor(wf, id)` — nest one workflow inside another.

The upstream `examples/03-workflows` directory has a sample for each of these.

## Struct executors and request ports (human in the loop)

When an executor needs `*workflow.Context` (to send messages selectively, yield output, or read and write state), give it a `Handle(ctx *workflow.Context, msg T) error` method and declare what it emits with the zero-size marker fields `workflow.AttrSendsMessage[T]` and `workflow.AttrYieldsOutput[T]`.

A `workflow.RequestPort` is a node that pauses the workflow and asks the caller for a value. When the run reaches it, the stream emits a `RequestInfoEvent`. Answer it with `e.Request.CreateResponse(value)` and `run.SendResponse`, and the response flows on to the next executor.

This example also enables checkpointing, which writes a checkpoint at the end of every superstep:

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

## Checkpointing and resume

`inproc.Default.WithCheckpointing(manager)` returns an environment that checkpoints after each superstep. `checkpoint.NewInMemoryManager()` keeps checkpoints in memory. For durable storage, use `checkpoint.NewJSONManager(store)` with `checkpoint.NewFileSystemJSONStore(dir)` or your own `checkpoint.Store`.

Collect the `CheckpointInfo` from `SuperStepCompletedEvent.CompletionInfo.CheckpointInfo`. Then call `run.RestoreCheckpoint(ctx, info)` and watch the stream again to rewind ("time travel"). Executors that keep fields in memory must save and restore them. Extend the executor with `OnCheckpointFunc` (write with `ctx.QueueStateUpdate`) and `OnCheckpointRestoredFunc` (read with `ctx.ReadState`):

```go
return workflow.NewExecutor("Judge", j).Extend(&workflow.Executor{
	ResetFunc:                j.Reset,
	OnCheckpointFunc:         j.OnCheckpoint,
	OnCheckpointRestoredFunc: j.OnCheckpointRestored,
})
```

The complete version is `examples/03-workflows/checkpoint/checkpoint_and_resume` upstream. Checkpointed state is serialised, so keep it to plain, JSON-friendly types.

## Agents in workflows

`agentworkflow` provides three prebuilt multi-agent patterns, each with a builder that returns a `*workflow.Workflow`:

| Pattern | Builder |
| --- | --- |
| Sequential: each agent sees the conversation so far | `agentworkflow.NewSequentialWorkflowBuilder(agents...)` |
| Concurrent: all agents get the same input, and the results are aggregated | `agentworkflow.NewConcurrentWorkflowBuilder(agents...)` |
| Group chat: a manager picks the next speaker | `agentworkflow.NewGroupChatWorkflowBuilder(managerFactory, agents...)` with `NewRoundRobinGroupChatManager` |

Agent workflows take `[]*message.Message` as input. Agents only start work when they receive a `workflow.TurnToken`. Its `EmitEvents` field makes them stream their `*agent.ResponseUpdate`s as `OutputEvent`s, and the final `OutputEvent` carries the whole conversation as `[]*message.Message`.

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

// echoAgent stands in for a model-backed agent so the example runs offline.
func echoAgent(name, prefix string) *agent.Agent {
	run := func(ctx context.Context, msgs []*message.Message, _ ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			last := msgs[len(msgs)-1].String()
			yield(&agent.ResponseUpdate{
				Role:      message.RoleAssistant,
				MessageID: name,
				Contents:  message.Contents{&message.TextContent{Text: prefix + last}},
			}, nil)
		}
	}
	return agent.New(agent.ProviderConfig{ProviderName: "echo", Run: run}, agent.Config{Name: name})
}

func main() {
	ctx := context.Background()
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(
		echoAgent("drafter", "draft: "),
		echoAgent("editor", "edited: "),
	).Build()
	if err != nil {
		log.Fatal(err)
	}

	run, err := inproc.Default.RunStreaming(ctx, wf, []*message.Message{message.NewText("release notes")})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()

	emit := true
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{EmitEvents: &emit}); err != nil {
		log.Fatal(err)
	}
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			if msgs, ok := e.Output.([]*message.Message); ok {
				for _, m := range msgs {
					fmt.Printf("%s (%s): %s\n", m.AuthorName, m.Role, m.String())
				}
				return
			}
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
 (user): release notes
drafter (assistant): draft: release notes
editor (assistant): edited: draft: release notes
```

Replace `echoAgent` with any provider-built agent (`openaiprovider.NewAgent`, `foundryprovider.NewAgent` and so on) and nothing else changes.

A round-robin group chat with an iteration cap:

```go
wf, err := agentworkflow.NewGroupChatWorkflowBuilder(
	func(as []*agent.Agent) *agentworkflow.GroupChatManager {
		return agentworkflow.NewRoundRobinGroupChatManager(as,
			agentworkflow.RoundRobinGroupChatOptions{MaximumIterationCount: 4})
	},
	writer, critic,
).WithName("review loop").Build()
```

To put an agent into a custom graph next to ordinary executors, bind it with `agentworkflow.New(agent, agentworkflow.Config{})`.

## A workflow as an agent

`agentworkflow.NewAgent(wf, cfg)` wraps a workflow so it looks like any other `*agent.Agent`. You can then run it with `RunText`, expose it over A2A or MCP, or nest it as a tool. Pending `RequestInfoEvent`s surface as request content in the response, and you resume by sending the matching response content in the next run.

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
)

func echoAgent(name, prefix string) *agent.Agent {
	run := func(ctx context.Context, msgs []*message.Message, _ ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			yield(&agent.ResponseUpdate{Role: message.RoleAssistant, MessageID: name,
				Contents: message.Contents{&message.TextContent{Text: prefix + msgs[len(msgs)-1].String()}}}, nil)
		}
	}
	return agent.New(agent.ProviderConfig{ProviderName: "echo", Run: run}, agent.Config{Name: name})
}

func main() {
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(
		echoAgent("drafter", "draft: "), echoAgent("editor", "edited: "),
	).Build()
	if err != nil {
		log.Fatal(err)
	}
	pipeline, err := agentworkflow.NewAgent(wf, agentworkflow.AgentConfig{
		Config: agent.Config{Name: "writing_pipeline"},
	})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := pipeline.RunText(context.Background(), "release notes").Collect()
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range resp.Messages {
		fmt.Printf("%s: %s\n", m.AuthorName, m.String())
	}
}
```

Output:

```text
drafter: draft: release notes
editor: edited: draft: release notes
```

## Event reference

| Event | When |
| --- | --- |
| `StartedEvent` | The run began |
| `SuperStepStartedEvent` / `SuperStepCompletedEvent` | Superstep boundaries; the completion info carries the checkpoint, if any |
| `ExecutorInvokedEvent` / `ExecutorCompletedEvent` / `ExecutorFailedEvent` | Per executor |
| `OutputEvent` | An executor listed in `WithOutputFrom` produced output (`OutputTagIntermediate` marks non-final outputs) |
| `RequestInfoEvent` | A request port is waiting for a response |
| `ErrorEvent`, `WorkflowWarningEvent` | Workflow-level failures and warnings |

Always handle `ExecutorFailedEvent` and `ErrorEvent` in your event loop. Otherwise a failing executor just ends the stream with no output.

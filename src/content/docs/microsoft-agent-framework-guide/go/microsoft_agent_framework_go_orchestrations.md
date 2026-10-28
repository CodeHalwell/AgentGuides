---
title: "Microsoft Agent Framework (Go) — Orchestrations"
description: "Prebuilt multi-agent patterns in the Go SDK's agentworkflow package: sequential, concurrent with custom aggregators, round-robin and custom group chat, tool approval inside an orchestration, and wrapping any workflow as an agent; plus what is missing compared with Python and .NET."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Orchestrations"
  order: 11
---

`workflow/agentworkflow` builds three common multi-agent patterns on top of the [workflow engine](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/): **sequential**, **concurrent** and **group chat**. Each builder returns an ordinary `*workflow.Workflow`, so everything on the workflows page (streaming, checkpointing, request handling, telemetry) applies. The package also wraps any workflow as an `*agent.Agent` with `agentworkflow.NewAgent`.

**Not available in Go yet:** handoff orchestration and Magentic orchestration (both exist in Python and .NET), and the orchestrator-agent group-chat manager. You can route between specialists with a [custom group-chat manager](#a-custom-manager-routing-between-specialists), which covers many handoff use cases, but the agents do not hand off by calling tools.

The examples that need no tool calls use small offline agents built with `agent.New`, so they run without credentials and print deterministic output; replace them with any provider agent (`openaiprovider.NewAgent`, `foundryprovider.NewAgent` and so on) and nothing else changes. The tool-approval and agent-as-tool examples use `openaiprovider` against a local stand-in for the OpenAI API, so their replies are not real model output. All of them were compiled and run against v%%version=github.com/microsoft/agent-framework-go%%.

## Choosing a pattern

| Pattern | Builder | Each agent sees | Final output |
| --- | --- | --- | --- |
| Sequential | `NewSequentialWorkflowBuilder(agents...)` | The whole conversation so far (or only the previous reply, with `WithChainOnlyAgentResponses(true)`) | The whole conversation, `[]*message.Message` |
| Concurrent | `NewConcurrentWorkflowBuilder(agents...)` | The same input, independently | Each agent's last message, or whatever `WithAggregator` returns |
| Group chat | `NewGroupChatWorkflowBuilder(managerFactory, agents...)` | The shared conversation | The whole conversation when the manager stops |
| Custom | `workflow.NewBuilder` with `agentworkflow.New(agent, cfg)` nodes | Whatever you route to it | Whatever you choose |

All three builders also take `WithName`, `WithDescription`, `WithOutputFrom(agents...)` and `WithIntermediateOutputFrom(agents...)`.

## How an orchestration is driven

Orchestrations speak a small chat protocol:

1. The input is `[]*message.Message` (`message.NewText` makes a user message).
2. Agents only run when a `workflow.TurnToken` arrives. Start the run with the messages, then send the token with `run.TrySendMessage`.
3. `TurnToken.EmitEvents` set to `true` makes each agent stream its `*agent.ResponseUpdate`s as intermediate `OutputEvent`s while it works. The terminal `OutputEvent` carries the result as `[]*message.Message`.

The helper below drives any of the orchestrations on this page. Each example repeats it so that it compiles on its own.

## Sequential

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

// roleAgent is an offline stand-in for a model-backed agent: it replies with
// reply(messages it was given).
func roleAgent(name string, reply func([]*message.Message) string) *agent.Agent {
	run := func(_ context.Context, msgs []*message.Message, _ ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			yield(&agent.ResponseUpdate{
				Role:      message.RoleAssistant,
				MessageID: name,
				Contents:  message.Contents{&message.TextContent{Text: reply(msgs)}},
			}, nil)
		}
	}
	return agent.New(agent.ProviderConfig{ProviderName: "offline", Run: run}, agent.Config{Name: name})
}

// runOrchestration sends the input and a turn token, prints streamed updates
// and returns the final conversation.
func runOrchestration(ctx context.Context, wf *workflow.Workflow, prompt string) ([]*message.Message, error) {
	run, err := inproc.Default.RunStreaming(ctx, wf, []*message.Message{message.NewText(prompt)})
	if err != nil {
		return nil, err
	}
	defer func() { _ = run.Close(ctx) }()
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{EmitEvents: new(true)}); err != nil {
		return nil, err
	}
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			return nil, err
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			switch out := e.Output.(type) {
			case *agent.ResponseUpdate:
				if s := out.String(); s != "" {
					fmt.Println("  streaming:", s)
				}
			case []*message.Message:
				if !e.IsIntermediate() {
					return out, nil
				}
			}
		case workflow.ExecutorFailedEvent:
			return nil, fmt.Errorf("%s failed: %w", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			return nil, e.Error
		}
	}
	return nil, fmt.Errorf("workflow ended without output")
}

func last(msgs []*message.Message) string { return msgs[len(msgs)-1].String() }

func team() []*agent.Agent {
	return []*agent.Agent{
		roleAgent("drafter", func(m []*message.Message) string { return "Draft: " + last(m) }),
		roleAgent("editor", func(m []*message.Message) string { return fmt.Sprintf("Edited (%d messages seen)", len(m)) }),
		roleAgent("checker", func(m []*message.Message) string { return fmt.Sprintf("Checked (%d messages seen)", len(m)) }),
	}
}

func main() {
	ctx := context.Background()
	for _, chainOnly := range []bool{false, true} {
		wf, err := agentworkflow.NewSequentialWorkflowBuilder(team()...).
			WithName("release-notes").
			WithChainOnlyAgentResponses(chainOnly).
			Build()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("chain only agent responses:", chainOnly)
		conv, err := runOrchestration(ctx, wf, "v2.4 adds single sign-on")
		if err != nil {
			log.Fatal(err)
		}
		for _, m := range conv {
			fmt.Printf("  %-8s %-9s %s\n", m.AuthorName, m.Role, m.String())
		}
	}
}
```

Output:

```text
chain only agent responses: false
  streaming: Draft: v2.4 adds single sign-on
  streaming: Edited (2 messages seen)
  streaming: Checked (3 messages seen)
           user      v2.4 adds single sign-on
  drafter  assistant Draft: v2.4 adds single sign-on
  editor   assistant Edited (2 messages seen)
  checker  assistant Checked (3 messages seen)
chain only agent responses: true
  streaming: Draft: v2.4 adds single sign-on
  streaming: Edited (1 messages seen)
  streaming: Checked (1 messages seen)
  checker  assistant Checked (1 messages seen)
```

By default each agent receives the whole conversation, with earlier agents' replies rewritten as user messages (the hosting executor's `ReassignOtherAgentsAsUsers` behaviour), and the output is the full transcript including the original input. With `WithChainOnlyAgentResponses(true)` each agent receives only the previous agent's reply, and the output is the last reply alone. Use that for a pipeline of transformations where earlier context would only distract the next agent and cost tokens.

## Concurrent

The concurrent pattern sends the same input to every agent at once and fans their answers in. The default aggregator keeps each agent's last message, in the order the agents finished. `WithAggregator` replaces it with a `MessageAggregator`, a function from one message batch per agent to the output messages:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

func roleAgent(name string, reply func([]*message.Message) string) *agent.Agent {
	run := func(_ context.Context, msgs []*message.Message, _ ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			yield(&agent.ResponseUpdate{Role: message.RoleAssistant, MessageID: name,
				Contents: message.Contents{&message.TextContent{Text: reply(msgs)}}}, nil)
		}
	}
	return agent.New(agent.ProviderConfig{ProviderName: "offline", Run: run}, agent.Config{Name: name})
}

func runOrchestration(ctx context.Context, wf *workflow.Workflow, prompt string) ([]*message.Message, error) {
	run, err := inproc.Default.RunStreaming(ctx, wf, []*message.Message{message.NewText(prompt)})
	if err != nil {
		return nil, err
	}
	defer func() { _ = run.Close(ctx) }()
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{}); err != nil {
		return nil, err
	}
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			return nil, err
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			if out, ok := e.Output.([]*message.Message); ok && !e.IsIntermediate() {
				return out, nil
			}
		case workflow.ExecutorFailedEvent:
			return nil, fmt.Errorf("%s failed: %w", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			return nil, e.Error
		}
	}
	return nil, fmt.Errorf("workflow ended without output")
}

func reviewers() []*agent.Agent {
	review := func(area, verdict string) func([]*message.Message) string {
		return func([]*message.Message) string { return area + ": " + verdict }
	}
	return []*agent.Agent{
		roleAgent("security", review("security", "no secrets in the diff")),
		roleAgent("performance", review("performance", "one N+1 query in ListOrders")),
		roleAgent("style", review("style", "fine")),
	}
}

// summarise folds the reviewers' answers into one assistant message.
func summarise(_ context.Context, batches [][]*message.Message) []*message.Message {
	var b strings.Builder
	b.WriteString("Review summary:")
	for _, batch := range batches {
		if len(batch) > 0 {
			m := batch[len(batch)-1]
			fmt.Fprintf(&b, "\n- [%s] %s", m.AuthorName, m.String())
		}
	}
	out := message.NewText(b.String())
	out.Role = message.RoleAssistant
	return []*message.Message{out}
}

func main() {
	ctx := context.Background()

	wf, err := agentworkflow.NewConcurrentWorkflowBuilder(reviewers()...).Build()
	if err != nil {
		log.Fatal(err)
	}
	msgs, err := runOrchestration(ctx, wf, "Review PR #812")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("default aggregator:")
	for _, m := range msgs {
		fmt.Printf("  %s: %s\n", m.AuthorName, m.String())
	}

	wf, err = agentworkflow.NewConcurrentWorkflowBuilder(reviewers()...).WithAggregator(summarise).Build()
	if err != nil {
		log.Fatal(err)
	}
	msgs, err = runOrchestration(ctx, wf, "Review PR #812")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("custom aggregator:")
	fmt.Println(msgs[0].String())
}
```

Output (the order of the lines changes from run to run):

```text
default aggregator:
  style: style: fine
  performance: performance: one N+1 query in ListOrders
  security: security: no secrets in the diff
custom aggregator:
Review summary:
- [performance] performance: one N+1 query in ListOrders
- [style] style: fine
- [security] security: no secrets in the diff
```

The agents really do run in parallel, and both their streamed updates and the batches the aggregator receives arrive in completion order, not in the order you listed the agents. Sort by `AuthorName` in the aggregator if the order matters. An aggregator is a plain function, so it can also call another agent to write a combined answer (pass it the context it receives).

## Group chat

In a group chat, a **manager** decides who speaks next and when to stop. Every participant sees the shared conversation. `NewGroupChatWorkflowBuilder` takes a factory, called once per run with the participants, that returns the manager.

### Round robin

`NewRoundRobinGroupChatManager` cycles through the agents in order. It stops after `MaximumIterationCount` participant turns (40 if zero), or earlier when `ShouldTerminate` returns true:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
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

func runOrchestration(ctx context.Context, wf *workflow.Workflow, prompt string) ([]*message.Message, error) {
	run, err := inproc.Default.RunStreaming(ctx, wf, []*message.Message{message.NewText(prompt)})
	if err != nil {
		return nil, err
	}
	defer func() { _ = run.Close(ctx) }()
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{}); err != nil {
		return nil, err
	}
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			return nil, err
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			if out, ok := e.Output.([]*message.Message); ok && !e.IsIntermediate() {
				return out, nil
			}
		case workflow.ExecutorFailedEvent:
			return nil, fmt.Errorf("%s failed: %w", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			return nil, e.Error
		}
	}
	return nil, fmt.Errorf("workflow ended without output")
}

func main() {
	drafts := 0
	writer := roleAgent("writer", func([]*message.Message) string {
		drafts++
		return fmt.Sprintf("slogan draft %d", drafts)
	})
	critic := roleAgent("critic", func([]*message.Message) string {
		if drafts >= 2 {
			return "APPROVED"
		}
		return "too long, try again"
	})

	wf, err := agentworkflow.NewGroupChatWorkflowBuilder(
		func(agents []*agent.Agent) *agentworkflow.GroupChatManager {
			return agentworkflow.NewRoundRobinGroupChatManager(agents, agentworkflow.RoundRobinGroupChatOptions{
				MaximumIterationCount: 6, // a hard cap on participant turns
				ShouldTerminate: func(_ context.Context, history []*message.Message, _ int) (bool, error) {
					return strings.Contains(history[len(history)-1].String(), "APPROVED"), nil
				},
			})
		},
		writer, critic,
	).WithName("slogan-review").Build()
	if err != nil {
		log.Fatal(err)
	}

	conv, err := runOrchestration(context.Background(), wf, "A slogan for an electric bike")
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range conv {
		fmt.Printf("%-6s %s\n", m.AuthorName, m.String())
	}
}
```

Output:

```text
       A slogan for an electric bike
writer slogan draft 1
critic too long, try again
writer slogan draft 2
critic APPROVED
```

### A custom manager: routing between specialists

A `GroupChatManager` is a struct of functions, so a custom one is a few lines:

| Field | Required | Called |
| --- | --- | --- |
| `SelectNextAgent(ctx, history)` | Yes | Before each turn. Return the next speaker, or `nil` to end the chat. |
| `ShouldTerminate(ctx, history, iterationCount)` | No | Before selection. If nil, the chat stops after 40 participant turns. |
| `UpdateHistory(ctx, messages)` | No | Filters the messages broadcast to participants for the turn; return `nil` to keep them all |
| `Reset()` | No | When the hosting executor is reset |
| `OnCheckpoint(ctx)` / `OnCheckpointRestored(ctx)` | No | To save and restore the manager's own state; keys are prefixed with `GroupChatManager_` |

The manager below behaves like a simple handoff: a triage agent speaks first, then the manager routes to the billing or technical specialist according to the triage agent's verdict, and ends the chat once a specialist has answered. Selection is plain Go here; it could equally ask a small model for a structured verdict.

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
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

func runOrchestration(ctx context.Context, wf *workflow.Workflow, prompt string) ([]*message.Message, error) {
	run, err := inproc.Default.RunStreaming(ctx, wf, []*message.Message{message.NewText(prompt)})
	if err != nil {
		return nil, err
	}
	defer func() { _ = run.Close(ctx) }()
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{}); err != nil {
		return nil, err
	}
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			return nil, err
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			if out, ok := e.Output.([]*message.Message); ok && !e.IsIntermediate() {
				return out, nil
			}
		case workflow.ExecutorFailedEvent:
			return nil, fmt.Errorf("%s failed: %w", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			return nil, e.Error
		}
	}
	return nil, fmt.Errorf("workflow ended without output")
}

func byName(agents []*agent.Agent, name string) *agent.Agent {
	for _, a := range agents {
		if a.Name() == name {
			return a
		}
	}
	return nil
}

// routingManager: triage first, then the specialist triage names, then stop.
func routingManager(agents []*agent.Agent) *agentworkflow.GroupChatManager {
	return &agentworkflow.GroupChatManager{
		SelectNextAgent: func(_ context.Context, history []*message.Message) (*agent.Agent, error) {
			lastMsg := history[len(history)-1]
			switch lastMsg.AuthorName {
			case "": // only the user has spoken
				return byName(agents, "triage"), nil
			case "triage":
				if strings.Contains(lastMsg.String(), "billing") {
					return byName(agents, "billing"), nil
				}
				return byName(agents, "technical"), nil
			default:
				return nil, nil // a specialist answered: end the chat
			}
		},
		ShouldTerminate: func(_ context.Context, _ []*message.Message, turns int) (bool, error) {
			return turns >= 4, nil // a safety cap
		},
	}
}

func main() {
	triage := roleAgent("triage", func(m []*message.Message) string {
		if strings.Contains(strings.ToLower(m[0].String()), "invoice") {
			return "route: billing"
		}
		return "route: technical"
	})
	billing := roleAgent("billing", func([]*message.Message) string { return "I have re-sent invoice INV-2207." })
	technical := roleAgent("technical", func([]*message.Message) string { return "Please clear the app cache." })

	for _, prompt := range []string{"I never got my March invoice", "The app crashes on start"} {
		wf, err := agentworkflow.NewGroupChatWorkflowBuilder(routingManager, triage, billing, technical).Build()
		if err != nil {
			log.Fatal(err)
		}
		conv, err := runOrchestration(context.Background(), wf, prompt)
		if err != nil {
			log.Fatal(err)
		}
		for _, m := range conv[1:] {
			fmt.Printf("%-9s %s\n", m.AuthorName, m.String())
		}
		fmt.Println()
	}
}
```

Output:

```text
triage    route: billing
billing   I have re-sent invoice INV-2207.

triage    route: technical
technical Please clear the app cache.

```

`SelectNextAgent` must return one of the participants it was given (or `nil`). Keep manager state in the closure that the factory creates, not in package variables, because the factory runs once per workflow run and a manager is not shared between runs. A manager that keeps state across turns should implement `OnCheckpoint` and `OnCheckpointRestored` if the workflow is checkpointed.

## Tool approval inside an orchestration

A participant's tools work as they do outside a workflow. A tool wrapped in `tool.ApprovalRequiredFunc` pauses the orchestration: the hosting executor raises a `RequestInfoEvent` whose payload is a `*message.ToolApprovalRequestContent`. Answer it by wrapping the approval content in an external response. The agent is then re-invoked with the decision and the turn continues:

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
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	"github.com/openai/openai-go/v3"
)

type deployIn struct {
	Version string `json:"version" jsonschema:"the version to deploy"`
}

func main() {
	deploy := functool.MustNew(functool.Config{
		Name:        "deploy_to_production",
		Description: "Deploy a version to production.",
	}, func(_ context.Context, in deployIn) (string, error) {
		return "deployed " + in.Version, nil
	})

	devops := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You deploy releases. Call deploy_to_production, then report the result.",
		Config: agent.Config{
			Name:  "devops",
			Tools: []tool.Tool{tool.ApprovalRequiredFunc(deploy)},
		},
	})

	wf, err := agentworkflow.NewSequentialWorkflowBuilder(devops).Build()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	run, err := inproc.Default.RunStreaming(ctx, wf, []*message.Message{message.NewText("Ship v2.4.0")})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{}); err != nil {
		log.Fatal(err)
	}

	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.RequestInfoEvent:
			req, ok := workflow.PortableValueAs[*message.ToolApprovalRequestContent](e.Request.Data)
			if !ok {
				log.Fatalf("unexpected request on %s", e.Request.PortInfo.PortID)
			}
			call := req.ToolCall.(*message.FunctionCallContent)
			fmt.Printf("approve %s(%s)? yes\n", call.Name, call.Arguments)
			resp, err := e.Request.CreateResponse(req.CreateResponse(true, "approved by on-call"))
			if err != nil {
				log.Fatal(err)
			}
			if err := run.SendResponse(ctx, resp); err != nil {
				log.Fatal(err)
			}
		case workflow.OutputEvent:
			if conv, ok := e.Output.([]*message.Message); ok && !e.IsIntermediate() {
				fmt.Println("final:", conv[len(conv)-1].String())
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

Output against the stand-in, which always calls the first tool it is offered:

```text
approve deploy_to_production({"version": "example version"})? yes
final: Here is what the tool returned: deployed example version
```

The same works in concurrent and group-chat orchestrations. `e.Request.PortInfo.PortID` tells you which participant asked: it is the participant's executor ID followed by `_UserInput` (here `devops_<agent ID>_UserInput`). Set `InterceptUserInputRequests` in `agentworkflow.Config` (custom graphs only) to receive approval requests as ordinary workflow messages instead, so another executor, such as a policy engine, can answer them.

## A workflow as an agent

`agentworkflow.NewAgent(wf, cfg)` wraps a workflow in an `*agent.Agent`. Anything that takes an agent then takes the workflow: `RunText`, streaming, sessions, [A2A or AG-UI hosting](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_hosting_and_a2a/), or `agenttool.New` to make it a tool of another agent.

The workflow's start executor must accept `[]*message.Message`, which every `agentworkflow` builder satisfies. For a custom graph, start it with an `agentworkflow.New` node or with an executor configured by `messageworkflow.Configure`.

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
	drafter := roleAgent("drafter", func(m []*message.Message) string { return "draft of: " + m[len(m)-1].String() })
	editor := roleAgent("editor", func(m []*message.Message) string { return fmt.Sprintf("edited (%d messages in context)", len(m)) })

	wf, err := agentworkflow.NewSequentialWorkflowBuilder(drafter, editor).Build()
	if err != nil {
		log.Fatal(err)
	}
	pipeline, err := agentworkflow.NewAgent(wf, agentworkflow.AgentConfig{
		Config: agent.Config{Name: "writing_pipeline", Description: "Drafts and edits short texts."},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Non-streaming: the response holds each participant's reply.
	resp, err := pipeline.RunText(ctx, "a welcome email").Collect()
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range resp.Messages {
		fmt.Printf("%s: %s\n", m.AuthorName, m.String())
	}

	// A session keeps one workflow run open across turns.
	session, err := pipeline.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, prompt := range []string{"a release note", "now shorter"} {
		fmt.Printf("> %s\n", prompt)
		for update, err := range pipeline.RunText(ctx, prompt, agent.WithSession(session), agent.Stream(true)) {
			if err != nil {
				log.Fatal(err)
			}
			if s := update.String(); s != "" {
				fmt.Printf("  [%s] %s\n", update.AuthorName, s)
			}
		}
	}
}
```

Output:

```text
drafter: draft of: a welcome email
editor: edited (2 messages in context)
> a release note
  [drafter] draft of: a release note
  [editor] edited (2 messages in context)
> now shorter
  [drafter] draft of: now shorter
  [editor] edited (5 messages in context)
```

The second turn on the session shows the workflow remembering the first: the editor now sees five messages. The first run on a session opens a streaming workflow run, and later runs on the same session send their messages into that open run. That makes the session tied to the process that holds the run.

`agentworkflow.AgentConfig`:

| Field | Default | Effect |
| --- | --- | --- |
| `Config` | | The usual `agent.Config`: name, description, middleware (including [OpenTelemetry](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/)) |
| `Environment` | `inproc.Concurrent` if every executor allows it, otherwise `inproc.OffThread` | The execution environment for each run; pass one with checkpointing to make the workflow durable |
| `IncludeOutputsInResponse` | `false` | Also surface `OutputEvent`s carrying `*message.Message`, `[]*message.Message` or `*agent.Response`. By default only the hosted agents' streamed updates become response content. |
| `IncludeErrorDetails` | `false` | Put the full `ErrorEvent` text in the response instead of a generic message. Leave it off for anything user-facing. |

### Approvals across the agent boundary

When a participant needs approval, a workflow agent surfaces the request as ordinary response content, a `*message.ToolApprovalRequestContent`, exactly as a single agent does. Resume by sending the approval in the next run on the **same session**:

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
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/openai/openai-go/v3"
)

type refundIn struct {
	Order string `json:"order" jsonschema:"the order ID"`
}

func main() {
	ctx := context.Background()
	refund := functool.MustNew(functool.Config{Name: "issue_refund", Description: "Refund an order."},
		func(_ context.Context, in refundIn) (string, error) { return "refunded " + in.Order, nil })
	support := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You handle refunds.",
		Config:       agent.Config{Name: "support", Tools: []tool.Tool{tool.ApprovalRequiredFunc(refund)}},
	})
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(support).Build()
	if err != nil {
		log.Fatal(err)
	}
	desk, err := agentworkflow.NewAgent(wf, agentworkflow.AgentConfig{Config: agent.Config{Name: "refund_desk"}})
	if err != nil {
		log.Fatal(err)
	}

	session, err := desk.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := desk.RunText(ctx, "Refund order A-17", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	// As of v0.2.1 the response carries each request twice (once from the workflow,
	// once from the participant), so answer each tool call once.
	var answers []message.Content
	seen := map[string]bool{}
	for c := range resp.Contents() {
		req, ok := c.(*message.ToolApprovalRequestContent)
		if !ok {
			continue
		}
		call := req.ToolCall.(*message.FunctionCallContent)
		if seen[call.CallID] {
			continue
		}
		seen[call.CallID] = true
		fmt.Printf("needs approval: %s(%s)\n", call.Name, call.Arguments)
		answers = append(answers, req.CreateResponse(true, "within policy"))
	}
	resp, err = desk.RunMessage(ctx, message.New(answers...), agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("final:", resp.String())
}
```

Output against the stand-in:

```text
needs approval: issue_refund({"order": "example order"})
final: Here is what the tool returned: refunded example order
```

In v%%version=github.com/microsoft/agent-framework-go%% the response contains two `ToolApprovalRequestContent` items for one pending call: one authored by the workflow agent, whose `RequestID` is the workflow's external request ID, and the participant's own, whose `RequestID` is its tool call ID. Answering either one resumes the turn, so the example answers each `CallID` once. This is what lets a workflow sit behind an A2A or AG-UI endpoint and still ask the caller for approval.

### A workflow as a tool

Because a workflow agent is an agent, `agenttool.New` exposes it as a function tool with a `{"query": string}` argument, named after the agent:

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
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/agenttool"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/openai/openai-go/v3"
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
	research := roleAgent("researcher", func([]*message.Message) string { return "3 sources found" })
	summary := roleAgent("summariser", func([]*message.Message) string { return "summary: adoption doubled in 2026" })
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(research, summary).WithChainOnlyAgentResponses(true).Build()
	if err != nil {
		log.Fatal(err)
	}
	pipeline, err := agentworkflow.NewAgent(wf, agentworkflow.AgentConfig{
		Config: agent.Config{Name: "research_pipeline", Description: "Researches a topic and summarises it."},
	})
	if err != nil {
		log.Fatal(err)
	}

	assistant := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Use research_pipeline for research questions.",
		Config:       agent.Config{Name: "assistant", Tools: []tool.Tool{agenttool.New(pipeline, agenttool.Config{})}},
	})
	resp, err := assistant.RunText(context.Background(), "How fast is agent adoption growing?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.String())
}
```

Output against the stand-in:

```text
Here is what the tool returned: 3 sources found
summary: adoption doubled in 2026
```

The tool result is the text of the workflow agent's whole response, which includes every participant's reply even with `WithChainOnlyAgentResponses`. If the calling model should see only the final answer, wrap the pipeline in a small `functool` tool that runs it and returns the last message.

## Compared with Python and .NET

| Python / .NET | Go |
| --- | --- |
| Sequential | `NewSequentialWorkflowBuilder`; `WithChainOnlyAgentResponses` for reply-only chaining |
| Concurrent with custom aggregator | `NewConcurrentWorkflowBuilder(...).WithAggregator(fn)` |
| Group chat, code-driven selection | `NewGroupChatWorkflowBuilder` with a `GroupChatManager` (round robin built in) |
| Group chat, orchestrator-agent selection | Not built in; call an agent from `SelectNextAgent` |
| Handoff | Not available; route with a custom manager or conditional edges |
| Magentic | Not available |
| `workflow.as_agent()` / `AsAIAgent()` | `agentworkflow.NewAgent(wf, cfg)` |
| Declarative orchestrations | Not available |

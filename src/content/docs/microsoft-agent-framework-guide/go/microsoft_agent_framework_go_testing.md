---
title: "Microsoft Agent Framework (Go) — Testing with a fake provider"
description: "Unit-test Go agents without a model: script provider turns through agent.ProviderConfig, then assert on tool calls, approvals, sessions, agent-as-tool and workflows with go test."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Testing"
  order: 30
---

The Go SDK ships no public mock model. Its own test helper, `internal/agenttest`, is internal, so other modules cannot import it. You do not need one, though. Every provider is just an `agent.ProviderConfig` passed to `agent.New`, and its only required field is a `Run` function that returns a stream of `*agent.ResponseUpdate`. Write your own `Run` that replays scripted turns, and you get a real `*agent.Agent`, with real middleware, sessions, history and tool calling, that never touches the network.

All the code on this page is one test package. It passes `go vet` and `go test` against v%%version=github.com/microsoft/agent-framework-go%%.

## The fake model

Two details matter:

1. **Install `toolautocall` yourself.** Real providers add the `agent/harness/toolautocall` middleware in `ProviderConfig.Middlewares`, and that middleware is what turns a `FunctionCallContent` into a tool invocation and a second model call. Leave it out and the fake returns the function call to your test unexecuted. Sometimes that is what you want, but usually it is not.
2. **One scripted turn per provider call.** A tool round-trip is two provider calls: the model asks for the tool, then the model answers using the result. Recording the messages each call receives lets you assert on what the model would have seen.

```go
package weather

import (
	"context"
	"fmt"
	"iter"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolautocall"
	"github.com/microsoft/agent-framework-go/message"
)

// scriptedModel replays one canned turn per provider call and records
// the messages it was given.
type scriptedModel struct {
	turns [][]message.Content
	calls [][]*message.Message
}

func (m *scriptedModel) run(ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	return func(yield func(*agent.ResponseUpdate, error) bool) {
		m.calls = append(m.calls, msgs)
		i := len(m.calls) - 1
		if i >= len(m.turns) {
			yield(nil, fmt.Errorf("scriptedModel: no turn %d scripted", i))
			return
		}
		yield(&agent.ResponseUpdate{
			Role:      message.RoleAssistant,
			MessageID: fmt.Sprintf("msg-%d", i),
			Contents:  m.turns[i],
		}, nil)
	}
}

// newFakeAgent wires the scripted model into a real agent.Agent, including
// the same automatic tool-calling middleware the real providers install.
func newFakeAgent(m *scriptedModel, cfg agent.Config) *agent.Agent {
	return agent.New(agent.ProviderConfig{
		ProviderName: "fake",
		Run:          m.run,
		Middlewares:  []agent.Middleware{toolautocall.New(toolautocall.Config{})},
	}, cfg)
}
```

`agent.New` takes the same `agent.Config` as the real providers (the provider configs embed it), so the tools, middleware, context providers and history provider under test are configured exactly as in production. The only difference is the `Run` function.

## Testing tools, approvals and sessions

```go
package weather

import (
	"context"
	"strings"
	"testing"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
)

type weatherIn struct {
	City string `json:"city"`
}

func TestAgentCallsWeatherTool(t *testing.T) {
	var gotCity string
	weather := functool.MustNew(functool.Config{
		Name:        "get_weather",
		Description: "Get the weather for a city",
	}, func(_ context.Context, in weatherIn) (string, error) {
		gotCity = in.City
		return "14°C and drizzle", nil
	})

	m := &scriptedModel{turns: [][]message.Content{
		{&message.FunctionCallContent{CallID: "call-1", Name: "get_weather", Arguments: `{"city":"Leeds"}`}},
		{&message.TextContent{Text: "It is 14°C and drizzly in Leeds."}},
	}}
	a := newFakeAgent(m, agent.Config{Name: "weather", Tools: []tool.Tool{weather}})

	resp, err := a.RunText(context.Background(), "Weather in Leeds?").Collect()
	if err != nil {
		t.Fatal(err)
	}
	if gotCity != "Leeds" {
		t.Errorf("tool got city %q, want Leeds", gotCity)
	}
	if !strings.Contains(resp.String(), "Leeds") {
		t.Errorf("unexpected reply %q", resp.String())
	}
	if len(m.calls) != 2 {
		t.Fatalf("model called %d times, want 2", len(m.calls))
	}
	// The second model call must carry the tool result back.
	var sawResult bool
	for _, msg := range m.calls[1] {
		for _, c := range msg.Contents {
			if r, ok := c.(*message.FunctionResultContent); ok && r.CallID == "call-1" {
				sawResult = true
			}
		}
	}
	if !sawResult {
		t.Error("second model call did not include the function result")
	}
}

func TestSessionKeepsHistory(t *testing.T) {
	m := &scriptedModel{turns: [][]message.Content{
		{&message.TextContent{Text: "Hello Ada."}},
		{&message.TextContent{Text: "Your name is Ada."}},
	}}
	a := newFakeAgent(m, agent.Config{})
	ctx := context.Background()
	s, err := a.CreateSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.RunText(ctx, "I am Ada", agent.WithSession(s)).Collect(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.RunText(ctx, "Who am I?", agent.WithSession(s)).Collect(); err != nil {
		t.Fatal(err)
	}
	if n := len(m.calls[1]); n != 3 {
		t.Errorf("second call saw %d messages, want 3 (user, assistant, user)", n)
	}
}

func TestApproval(t *testing.T) {
	calls := 0
	del := functool.MustNew(functool.Config{Name: "delete_file", Description: "Delete a file"},
		func(_ context.Context, in struct {
			Path string `json:"path"`
		}) (string, error) {
			calls++
			return "deleted " + in.Path, nil
		})
	m := &scriptedModel{turns: [][]message.Content{
		{&message.FunctionCallContent{CallID: "c1", Name: "delete_file", Arguments: `{"path":"/tmp/x"}`}},
		{&message.TextContent{Text: "Done."}},
	}}
	a := newFakeAgent(m, agent.Config{Tools: []tool.Tool{tool.ApprovalRequiredFunc(del)}})
	ctx := context.Background()
	s, _ := a.CreateSession(ctx)
	resp, err := a.RunText(ctx, "delete /tmp/x", agent.WithSession(s)).Collect()
	if err != nil {
		t.Fatal(err)
	}
	var replies []message.Content
	for c := range resp.Contents() {
		if req, ok := c.(*message.ToolApprovalRequestContent); ok {
			replies = append(replies, req.CreateResponse(true, ""))
		}
	}
	if len(replies) != 1 || calls != 0 {
		t.Fatalf("replies=%d calls=%d", len(replies), calls)
	}
	resp, err = a.RunMessage(ctx, message.New(replies...), agent.WithSession(s)).Collect()
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || resp.String() != "Done." {
		t.Fatalf("calls=%d resp=%q", calls, resp.String())
	}
}
```

What these tests demonstrate:

- **Tool round-trip.** The tool receives typed, schema-validated arguments (`weatherIn`). The second provider call includes a `FunctionResultContent` with the matching `CallID`.
- **Session history.** With the default in-memory history provider, the second run on a session replays the previous user and assistant messages.
- **Approvals.** A tool wrapped in `tool.ApprovalRequiredFunc` is not called on the first run. Instead, the response carries a `ToolApprovalRequestContent`. Sending `req.CreateResponse(true, "")` back with `RunMessage` on the same session executes the tool and makes the next scripted call.

## Sessions, agent-as-tool, errors and workflows

```go
package weather

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/agenttool"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

func TestSessionSurvivesJSON(t *testing.T) {
	m := &scriptedModel{turns: [][]message.Content{
		{&message.TextContent{Text: "Noted."}},
		{&message.TextContent{Text: "You are Ada."}},
	}}
	a := newFakeAgent(m, agent.Config{})
	ctx := context.Background()

	s, _ := a.CreateSession(ctx)
	if _, err := a.RunText(ctx, "I am Ada", agent.WithSession(s)).Collect(); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}

	var restored agent.Session
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if _, err := a.RunText(ctx, "Who am I?", agent.WithSession(&restored)).Collect(); err != nil {
		t.Fatal(err)
	}
	if got := len(m.calls[1]); got != 3 {
		t.Fatalf("restored session replayed %d messages, want 3", got)
	}
}

func TestAgentAsTool(t *testing.T) {
	specialist := newFakeAgent(&scriptedModel{turns: [][]message.Content{
		{&message.TextContent{Text: "VAT is 20%."}},
	}}, agent.Config{Name: "tax_expert", Description: "Answers UK tax questions"})

	manager := &scriptedModel{turns: [][]message.Content{
		{&message.FunctionCallContent{CallID: "c1", Name: "tax_expert", Arguments: `{"query":"VAT rate?"}`}},
		{&message.TextContent{Text: "The expert says VAT is 20%."}},
	}}
	a := newFakeAgent(manager, agent.Config{
		Tools: []tool.Tool{agenttool.New(specialist, agenttool.Config{})},
	})
	resp, err := a.RunText(context.Background(), "What's the VAT rate?").Collect()
	if err != nil {
		t.Fatal(err)
	}
	if resp.String() != "The expert says VAT is 20%." {
		t.Fatalf("unexpected reply %q", resp.String())
	}
}

func TestProviderErrorSurfaces(t *testing.T) {
	m := &scriptedModel{} // no turns scripted: every call fails
	a := newFakeAgent(m, agent.Config{})
	_, err := a.RunText(context.Background(), "hi").Collect()
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "no turn 0 scripted") {
		t.Fatalf("unexpected error %v", err)
	}
}

func TestWorkflowRouting(t *testing.T) {
	double := workflow.NewExecutor("double", func(n int) int { return n * 2 }).Bind()
	label := workflow.NewExecutor("label", func(n int) string {
		if n > 10 {
			return "big"
		}
		return "small"
	}).Bind()
	wf, err := workflow.NewBuilder(double).AddEdge(double, label).WithOutputFrom(label).Build()
	if err != nil {
		t.Fatal(err)
	}
	run, err := inproc.Default.Run(context.Background(), wf, 7)
	if err != nil {
		t.Fatal(err)
	}
	var outputs []any
	for evt := range run.NewEvents() {
		if o, ok := evt.(workflow.OutputEvent); ok {
			outputs = append(outputs, o.Output)
		}
	}
	if len(outputs) != 1 || outputs[0] != "big" {
		t.Fatalf("outputs = %v", outputs)
	}
}
```

Notes:

- **Session round-trip.** A session that goes through `json.Marshal` and `json.Unmarshal` keeps its in-memory history. This is the same mechanism you would use to persist conversations in a database.
- **Agent as a tool.** `agenttool.New` exposes the specialist with a `{"query": string}` schema and names the tool after the agent's `Name`. Scripting the manager's function call with that name exercises the full delegation path.
- **Errors.** An error the provider yields comes back from `Collect()` unchanged, so `errors.Is` and `errors.As` work on your own error types.
- **Workflows.** Plain-function executors are deterministic, so a workflow test is just `inproc.Default.Run` followed by collecting the `OutputEvent`s. For agent workflows, build the agents with `newFakeAgent`. The offline examples on the [workflows page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/) use the same approach.

## Testing tools and middleware directly

You often do not need an agent at all:

- A `functool` tool is a `tool.FuncTool`. Call ``t.Call(ctx, `{"city":"York"}`)`` and assert on the result. Invalid arguments return a validation error before your function runs, which is worth testing as well.
- `t.(tool.SchemaTool).Schema()` returns the derived JSON schema. A golden-file test on it catches accidental breaking changes to a tool's contract.
- An `agent.MiddlewareFunc` is an ordinary function. Call it with a stub `next` that yields canned updates, and assert on what it passes on.

## What a fake cannot tell you

A scripted model checks your wiring: which tools exist, how arguments decode, how approvals and history flow, and what the graph routes where. It cannot tell you whether a real model will choose the tool, follow the instructions, or produce valid structured output. Keep a small set of integration tests against a real deployment for that, behind a build tag or an environment check:

```go
func TestLiveWeather(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set OPENAI_API_KEY to run live tests")
	}
	// ... build the agent with openaiprovider.NewAgent and assert loosely on the reply
}
```

The Go SDK has no evaluation framework comparable to Python's `evaluate_agent`. For quality evaluation, score recorded responses in your own harness.

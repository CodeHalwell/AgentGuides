---
title: "Microsoft Agent Framework (Go) — Testing with a fake provider"
description: "Unit-test Go agents and workflows without a model: a scripted fake provider, tool calls, approvals, sessions, agent-as-tool, tools and middleware in isolation, human-in-the-loop and checkpoint resume, orchestrations, A2A and AG-UI endpoints, and telemetry assertions with go test."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Testing"
  order: 14
---

The Go SDK ships no public mock model. Its own test helper, `internal/agenttest`, is internal, so other modules cannot import it. You do not need one, though. Every provider is just an `agent.ProviderConfig` passed to `agent.New`, and its only required field is a `Run` function that returns a stream of `*agent.ResponseUpdate`. Write your own `Run` that replays scripted turns, and you get a real `*agent.Agent`, with real middleware, sessions, history and tool calling, that never touches the network.

All the code on this page is one test package (`package weather`), split into several files. It passes `go vet` and `go test -race` against v%%version=github.com/microsoft/agent-framework-go%%.

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
	"sync"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolautocall"
	"github.com/microsoft/agent-framework-go/message"
)

// scriptedModel replays one canned turn per provider call and records
// the messages it was given. The mutex makes it safe to call concurrently.
type scriptedModel struct {
	mu    sync.Mutex
	turns [][]message.Content
	calls [][]*message.Message
}

func (m *scriptedModel) run(ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	return func(yield func(*agent.ResponseUpdate, error) bool) {
		m.mu.Lock()
		m.calls = append(m.calls, msgs)
		i := len(m.calls) - 1
		m.mu.Unlock()
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

You often do not need an agent at all. A `functool` tool is a `tool.FuncTool`, so call it with JSON and assert on the result; invalid arguments fail validation before your function runs. `Schema()` (from `tool.SchemaTool`) returns the derived JSON schema, and comparing it with a checked-in copy catches accidental changes to a tool's contract. An `agent.MiddlewareFunc` is an ordinary function: call it with a stub `next` that yields canned updates.

```go
package weather

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
)

type forecastIn struct {
	City string `json:"city" jsonschema:"the city name"`
	Days int    `json:"days" jsonschema:"number of days"`
}

func newForecastTool() tool.FuncTool {
	return functool.MustNew(functool.Config{Name: "forecast", Description: "Weather forecast"},
		func(_ context.Context, in forecastIn) (string, error) {
			if in.Days > 7 {
				return "", errors.New("at most 7 days")
			}
			return strings.Repeat("sun ", in.Days), nil
		})
}

func TestForecastTool(t *testing.T) {
	ft := newForecastTool()
	ctx := context.Background()

	got, err := ft.Call(ctx, `{"city":"York","days":2}`)
	if err != nil || got != "sun sun " {
		t.Fatalf("got %q, %v", got, err)
	}
	if _, err := ft.Call(ctx, `{"city":"York","days":9}`); err == nil {
		t.Error("expected the 7-day limit to be enforced")
	}
	if _, err := ft.Call(ctx, `{"city":42}`); err == nil {
		t.Error("expected a validation error for a non-string city")
	}
}

func TestForecastSchemaIsStable(t *testing.T) {
	schema, err := json.Marshal(newForecastTool().(tool.SchemaTool).Schema())
	if err != nil {
		t.Fatal(err)
	}
	// In a real suite, compare with testdata/forecast.schema.json.
	for _, want := range []string{`"city"`, `"days"`, `"the city name"`} {
		if !strings.Contains(string(schema), want) {
			t.Errorf("schema %s lacks %s", schema, want)
		}
	}
}

// redact is the middleware under test: it masks card numbers in replies.
var redact = agent.MiddlewareFunc(func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	return func(yield func(*agent.ResponseUpdate, error) bool) {
		for u, err := range next(ctx, msgs, opts...) {
			if u != nil {
				for _, c := range u.Contents {
					if tc, ok := c.(*message.TextContent); ok {
						tc.Text = strings.ReplaceAll(tc.Text, "4111111111111111", "[card]")
					}
				}
			}
			if !yield(u, err) {
				return
			}
		}
	}
})

func TestRedactMiddleware(t *testing.T) {
	stub := func(context.Context, []*message.Message, ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			yield(&agent.ResponseUpdate{Contents: message.Contents{&message.TextContent{Text: "card 4111111111111111 saved"}}}, nil)
		}
	}
	for u, err := range redact.Run(stub, context.Background(), nil) {
		if err != nil {
			t.Fatal(err)
		}
		if got := u.String(); got != "card [card] saved" {
			t.Errorf("got %q", got)
		}
	}
}
```

## Testing workflows

Workflows built from executors are deterministic, so most workflow tests need no fake at all: run the graph and assert on the events. Three things deserve their own tests: the human-in-the-loop path, resuming from a checkpoint, and the orchestrations, where the fake model stands in for each participant.

```go
package weather

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/checkpoint"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type refund struct {
	Order  string
	Amount float64
}

type refundReview struct {
	_ workflow.AttrSendsMessage[refund]
	_ workflow.AttrYieldsOutput[string]
}

func (refundReview) Handle(ctx *workflow.Context, r refund) error {
	if r.Amount < 100 {
		return ctx.YieldOutput("auto-approved " + r.Order)
	}
	return ctx.SendMessage("", r)
}

// buildRefundWorkflow returns a fresh, identical graph on every call.
func buildRefundWorkflow() *workflow.Workflow {
	port := workflow.RequestPort{ID: "approval", Request: reflect.TypeFor[refund](), Response: reflect.TypeFor[bool]()}
	review := workflow.NewExecutor("review", refundReview{}).Bind()
	ask := port.Bind()
	decide := workflow.NewExecutor("decide", func(ok bool) string { return fmt.Sprint("approved=", ok) }).Bind()
	wf, err := workflow.NewBuilder(review).AddEdge(review, ask).AddEdge(ask, decide).
		WithOutputFrom(review, decide).Build()
	if err != nil {
		panic(err)
	}
	return wf
}

func outputs(events func(func(workflow.Event) bool)) (outs []any, reqs []*workflow.ExternalRequest, failure error) {
	for evt := range events {
		switch e := evt.(type) {
		case workflow.OutputEvent:
			outs = append(outs, e.Output)
		case workflow.RequestInfoEvent:
			reqs = append(reqs, e.Request)
		case workflow.ExecutorFailedEvent:
			failure = e.Error
		case workflow.ErrorEvent:
			failure = e.Error
		}
	}
	return
}

func TestSmallRefundNeedsNoHuman(t *testing.T) {
	run, err := inproc.Default.Run(context.Background(), buildRefundWorkflow(), refund{"A-1", 20})
	if err != nil {
		t.Fatal(err)
	}
	outs, reqs, failure := outputs(run.NewEvents())
	if failure != nil || len(reqs) != 0 || len(outs) != 1 || outs[0] != "auto-approved A-1" {
		t.Fatalf("outs=%v reqs=%d err=%v", outs, len(reqs), failure)
	}
}

func TestLargeRefundWaitsAndResumes(t *testing.T) {
	ctx := context.Background()
	run, err := inproc.Default.Run(ctx, buildRefundWorkflow(), refund{"A-2", 500})
	if err != nil {
		t.Fatal(err)
	}
	_, reqs, _ := outputs(run.NewEvents())
	if len(reqs) != 1 || reqs[0].PortInfo.PortID != "approval" {
		t.Fatalf("expected one approval request, got %d", len(reqs))
	}
	if status, _ := run.GetStatus(ctx); status != inproc.RunStatusPendingRequests {
		t.Fatalf("status = %v", status)
	}
	resp, err := reqs[0].CreateResponse(false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := run.Resume(ctx, resp); err != nil {
		t.Fatal(err)
	}
	outs, _, _ := outputs(run.NewEvents())
	if len(outs) != 1 || outs[0] != "approved=false" {
		t.Fatalf("outs = %v", outs)
	}
}

func TestResumeFromCheckpointInFreshGraph(t *testing.T) {
	ctx := context.Background()
	mgr := checkpoint.NewInMemoryManager()
	env := inproc.Default.WithCheckpointing(mgr)

	first, err := env.RunStreaming(ctx, buildRefundWorkflow(), refund{"A-3", 900}, inproc.WithSessionID("case-3"))
	if err != nil {
		t.Fatal(err)
	}
	for range first.WatchUntilHalt(ctx) {
	}
	_ = first.Close(ctx)

	latest, err := mgr.LatestCheckpoint(ctx, "case-3")
	if err != nil || latest == nil {
		t.Fatalf("no checkpoint: %v", err)
	}
	second, err := env.ResumeStreaming(ctx, buildRefundWorkflow(), *latest)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = second.Close(ctx) }()
	for evt, err := range second.WatchStream(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.RequestInfoEvent:
			resp, _ := e.Request.CreateResponse(true)
			if err := second.SendResponse(ctx, resp); err != nil {
				t.Fatal(err)
			}
		case workflow.OutputEvent:
			if e.Output != "approved=true" {
				t.Fatalf("output = %v", e.Output)
			}
			return
		}
	}
	t.Fatal("resumed run produced no output")
}

// runOrchestration drives an agentworkflow orchestration to its final output.
func runOrchestration(t *testing.T, wf *workflow.Workflow, prompt string) []*message.Message {
	t.Helper()
	ctx := context.Background()
	run, err := inproc.Default.RunStreaming(ctx, wf, []*message.Message{message.NewText(prompt)})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{}); err != nil {
		t.Fatal(err)
	}
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			if msgs, ok := e.Output.([]*message.Message); ok && !e.IsIntermediate() {
				return msgs
			}
		case workflow.ExecutorFailedEvent:
			t.Fatalf("%s failed: %v", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			t.Fatal(e.Error)
		}
	}
	t.Fatal("no output")
	return nil
}

func text(s string) []message.Content { return []message.Content{&message.TextContent{Text: s}} }

func TestSequentialPassesTheConversationOn(t *testing.T) {
	drafter := &scriptedModel{turns: [][]message.Content{text("draft")}}
	editor := &scriptedModel{turns: [][]message.Content{text("edited")}}
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(
		newFakeAgent(drafter, agent.Config{Name: "drafter"}),
		newFakeAgent(editor, agent.Config{Name: "editor"}),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	conv := runOrchestration(t, wf, "release notes")
	if len(conv) != 3 || conv[2].AuthorName != "editor" || conv[2].String() != "edited" {
		t.Fatalf("conversation = %v", conv)
	}
	if n := len(editor.calls[0]); n != 2 { // the user's message and the draft
		t.Errorf("editor saw %d messages, want 2", n)
	}
}

func TestGroupChatRespectsTheTurnCap(t *testing.T) {
	endless := func() *scriptedModel {
		return &scriptedModel{turns: [][]message.Content{text("again"), text("again"), text("again")}}
	}
	wf, err := agentworkflow.NewGroupChatWorkflowBuilder(
		func(as []*agent.Agent) *agentworkflow.GroupChatManager {
			return agentworkflow.NewRoundRobinGroupChatManager(as, agentworkflow.RoundRobinGroupChatOptions{MaximumIterationCount: 3})
		},
		newFakeAgent(endless(), agent.Config{Name: "a"}),
		newFakeAgent(endless(), agent.Config{Name: "b"}),
	).Build()
	if err != nil {
		t.Fatal(err)
	}
	conv := runOrchestration(t, wf, "start")
	if turns := len(conv) - 1; turns != 3 {
		t.Fatalf("got %d participant turns, want 3", turns)
	}
}

func TestWorkflowAsAgent(t *testing.T) {
	m := &scriptedModel{turns: [][]message.Content{text("bonjour")}}
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(newFakeAgent(m, agent.Config{Name: "translator"})).Build()
	if err != nil {
		t.Fatal(err)
	}
	a, err := agentworkflow.NewAgent(wf, agentworkflow.AgentConfig{Config: agent.Config{Name: "pipeline"}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := a.RunText(context.Background(), "hello").Collect()
	if err != nil {
		t.Fatal(err)
	}
	if resp.String() != "bonjour" {
		t.Fatalf("got %q", resp.String())
	}
}
```

Each participant gets its own `scriptedModel`, so you can assert on exactly what each one was sent. Concurrent orchestrations call several fakes at once; `scriptedModel` guards its slices with a mutex for that reason.

## Testing hosted endpoints

The A2A and AG-UI handlers are `http.Handler`s, so `httptest.NewServer` plus the matching client agent gives an end-to-end test of the protocol layer with a fake model behind it:

```go
package weather

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	aguiSSEClient "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/client/sse"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
	"github.com/microsoft/agent-framework-go/provider/aguiprovider"
)

func TestA2AEndpoint(t *testing.T) {
	m := &scriptedModel{turns: [][]message.Content{text("opened items: 30 days"), text("sale items: 14 days")}}
	hosted := newFakeAgent(m, agent.Config{Name: "policy"})

	srv := httptest.NewUnstartedServer(nil)
	url := "http://" + srv.Listener.Addr().String()
	srv.Config.Handler = a2asrv.NewRESTHandler(a2aprovider.NewHandler(hosted, a2aprovider.ExecutorConfig{}))
	srv.Start()
	defer srv.Close()

	card := &a2a.AgentCard{Name: "policy", Description: "Returns policy", Version: "1",
		DefaultInputModes: []string{"text/plain"}, DefaultOutputModes: []string{"text/plain"},
		SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(url, a2a.TransportProtocolHTTPJSON)}}
	ctx := context.Background()
	remote, err := a2aprovider.NewAgentFromCard(ctx, card, a2aprovider.AgentConfig{})
	if err != nil {
		t.Fatal(err)
	}
	s, err := remote.CreateSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []string{"opened items: 30 days", "sale items: 14 days"} {
		resp, err := remote.RunText(ctx, "question", agent.WithSession(s)).Collect()
		if err != nil {
			t.Fatal(err)
		}
		if resp.String() != want {
			t.Errorf("turn %d: got %q", i, resp.String())
		}
	}
	// The server kept the conversation: the second model call saw three messages.
	if n := len(m.calls[1]); n != 3 {
		t.Errorf("second server-side call saw %d messages, want 3", n)
	}
}

func TestAGUIEndpoint(t *testing.T) {
	m := &scriptedModel{turns: [][]message.Content{text("hello from the server")}}
	srv := httptest.NewServer(aguiprovider.NewJSONHTTPHandler(newFakeAgent(m, agent.Config{Name: "ui"}), aguiprovider.HandlerConfig{}))
	defer srv.Close()

	client := aguiprovider.NewAgent(aguiSSEClient.NewClient(aguiSSEClient.Config{Endpoint: srv.URL}), aguiprovider.AgentConfig{})
	resp, err := client.RunText(context.Background(), "hi").Collect()
	if err != nil {
		t.Fatal(err)
	}
	if resp.String() != "hello from the server" {
		t.Fatalf("got %q", resp.String())
	}
}
```

## Asserting on telemetry

Install an in-memory exporter as the global tracer provider for the test and inspect the spans. With the fake provider you get the agent and tool spans (model-call spans come only from the real adapters). This is also the place to check that no user text leaks into attributes. Tests that replace the global provider must not run in parallel with each other.

```go
package weather

import (
	"context"
	"strings"
	"testing"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/otelprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestAgentEmitsSpansWithoutContent(t *testing.T) {
	spans := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(spans))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() {
		otel.SetTracerProvider(previous)
		_ = tp.Shutdown(context.Background())
	})

	m := &scriptedModel{turns: [][]message.Content{
		{&message.FunctionCallContent{CallID: "c1", Name: "forecast", Arguments: `{"city":"Leeds","days":1}`}},
		text("Sunny in Leeds."),
	}}
	a := newFakeAgent(m, agent.Config{
		Name:        "forecaster",
		Tools:       []tool.Tool{newForecastTool()},
		Middlewares: []agent.Middleware{otelprovider.NewMiddleware(otelprovider.MiddlewareConfig{SourceName: "test"})},
	})
	if _, err := a.RunText(context.Background(), "Weather in Leeds?").Collect(); err != nil {
		t.Fatal(err)
	}

	names := map[string]bool{}
	for _, s := range spans.GetSpans() {
		names[s.Name] = true
		for _, kv := range s.Attributes {
			if strings.Contains(kv.Value.Emit(), "Leeds") {
				t.Errorf("span %q attribute %s leaks user text", s.Name, kv.Key)
			}
		}
	}
	for _, want := range []string{"invoke_agent forecaster", "execute_tool forecast"} {
		if !names[want] {
			t.Errorf("missing span %q; got %v", want, names)
		}
	}
}
```

## Running the suite

Run the tests with the race detector, which catches the most common agent bug in Go services, sharing a session or a tool's state between goroutines:

```bash
go test -race ./...
```

Everything on this page passed `go vet` and `go test -race` against v%%version=github.com/microsoft/agent-framework-go%%, in well under a second, with no network access.

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

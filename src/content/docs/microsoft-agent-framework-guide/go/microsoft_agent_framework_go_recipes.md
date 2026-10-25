---
title: "Microsoft Agent Framework (Go) — Recipes"
description: "Complete Go programs built with Microsoft Agent Framework: ticket triage, a research brief from parallel experts, a writer-critic loop, an expense-approval HTTP service with checkpoints, A2A delegation, a support agent with approvals and saved sessions, a traced data-quality pipeline, validated extraction with a retry, and a workflow served over AG-UI."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Recipes"
  order: 16
---

Each recipe is one complete `main` package that you can copy into a directory, `go mod init`, `go get github.com/microsoft/agent-framework-go@v%%version=github.com/microsoft/agent-framework-go%%` and run. They combine the pieces documented on the other pages, which are linked where a recipe relies on them.

All of them were compiled with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%% and run against a local stand-in for the OpenAI API. Set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if you use a compatible endpoint) to run them for real. The outputs shown are from the stand-in, which echoes the last user message, calls the first tool it is offered, and fills structured-output schemas with sample values (strings become `example <field>`, integers 3), so they show the control flow, not real model quality.

| Recipe | Shows |
| --- | --- |
| [Ticket triage](#ticket-triage) | Structured output in an executor, switch routing, intermediate outputs |
| [Research brief from parallel experts](#research-brief-from-parallel-experts) | Concurrent orchestration with an aggregator that calls another agent |
| [Writer and critic loop](#writer-and-critic-loop) | A bounded loop between two agents with structured feedback |
| [Expense approvals over HTTP](#expense-approvals-over-http) | Human in the loop across requests, checkpoints, resuming by session ID |
| [Delegating to a remote agent over A2A](#delegating-to-a-remote-agent-over-a2a) | Hosting an agent with A2A and using it as a tool |
| [Support agent with approvals and saved sessions](#support-agent-with-approvals-and-saved-sessions) | Approval-gated tools, approve and deny, sessions saved to disk |
| [Traced data-quality pipeline](#traced-data-quality-pipeline) | Fan-out and fan-in over data, an agent summary, workflow and agent telemetry |
| [Validated extraction with a retry](#validated-extraction-with-a-retry) | Structured output, validation, feedback to the model, a fallback |
| [A workflow behind an AG-UI endpoint](#a-workflow-behind-an-ag-ui-endpoint) | Workflow as an agent, AG-UI hosting, a streaming client |

## Ticket triage

A classifier agent turns a free-text ticket into a typed `Triage` value through structured output. A switch sends it to the right specialist, and urgent tickets also produce an intermediate "page on-call" output. Each specialist is an agent called from an ordinary executor, so the routing stays deterministic Go code.

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

type Ticket struct {
	ID   string
	Text string
}

type Triage struct {
	Category string `json:"category" jsonschema:"billing, technical or other"`
	Priority int    `json:"priority" jsonschema:"1 (low) to 5 (critical)"`
	Summary  string `json:"summary" jsonschema:"one-sentence summary"`
}

type Triaged struct {
	Ticket Ticket
	Triage Triage
}

type Reply struct {
	TicketID string
	Team     string
	Text     string
}

func newAgent(name, instructions string) *agent.Agent {
	return openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: instructions,
		Config:       agent.Config{Name: name},
	})
}

// specialist wraps an agent in an executor that drafts a reply.
func specialist(team string, a *agent.Agent) workflow.ExecutorBinding {
	return workflow.NewExecutor(team, func(ctx *workflow.Context, t Triaged) (Reply, error) {
		prompt := fmt.Sprintf("Ticket %s (priority %d): %s", t.Ticket.ID, t.Triage.Priority, t.Ticket.Text)
		resp, err := a.RunText(ctx, prompt).Collect()
		if err != nil {
			return Reply{}, err
		}
		return Reply{TicketID: t.Ticket.ID, Team: team, Text: resp.String()}, nil
	}).Bind()
}

func buildWorkflow() *workflow.Workflow {
	classifier := newAgent("classifier", "Classify support tickets. Use priority 5 only for outages.")

	classify := workflow.NewExecutor("classify", func(ctx *workflow.Context, t Ticket) (Triaged, error) {
		var tr Triage
		if _, err := classifier.RunText(ctx, t.Text, agent.WithStructuredOutput(&tr)).Collect(); err != nil {
			return Triaged{}, err
		}
		return Triaged{Ticket: t, Triage: tr}, nil
	}).Bind()

	billing := specialist("billing", newAgent("billing", "You resolve billing questions. Be brief."))
	technical := specialist("technical", newAgent("technical", "You resolve technical problems. Be brief."))
	general := specialist("general", newAgent("general", "You answer general questions. Be brief."))

	page := workflow.NewExecutor("page_on_call", func(t Triaged) string {
		return fmt.Sprintf("PAGE: %s is priority %d", t.Ticket.ID, t.Triage.Priority)
	}).Bind()
	send := workflow.NewExecutor("send", func(r Reply) string {
		return fmt.Sprintf("[%s -> %s] %s", r.TicketID, r.Team, r.Text)
	}).Bind()

	b := workflow.NewBuilder(classify)
	b.AddSwitch(classify).
		AddCase(func(m any) bool { return m.(Triaged).Triage.Category == "billing" }, billing).
		AddCase(func(m any) bool { return m.(Triaged).Triage.Category == "technical" }, technical).
		WithDefault(general).
		AddToBuilder(b).
		AddEdge(classify, page, workflow.WithEdgeCondition(func(t Triaged) bool { return t.Triage.Priority >= 3 })).
		AddEdge(billing, send).
		AddEdge(technical, send).
		AddEdge(general, send).
		WithIntermediateOutputFrom(page).
		WithOutputFrom(send)
	wf, err := b.Build()
	if err != nil {
		log.Fatal(err)
	}
	return wf
}

func main() {
	tickets := []Ticket{
		{ID: "T-101", Text: "I was charged twice for my March invoice."},
		{ID: "T-102", Text: "The mobile app crashes when I open settings."},
	}
	for _, t := range tickets {
		run, err := inproc.Default.Run(context.Background(), buildWorkflow(), t)
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
		_ = run.Close(context.Background())
	}
}
```

Output against the stand-in. Its classifier always answers with the sample category `example category` and priority 3, so both tickets go to the default team and page on-call:

```text
PAGE: T-101 is priority 3
[T-101 -> general] Mock reply to: Ticket T-101 (priority 3): I was charged twice for my March invoice.
PAGE: T-102 is priority 3
[T-102 -> general] Mock reply to: Ticket T-102 (priority 3): The mobile app crashes when I open settings.
```

With a real model the categories vary, and the same graph sends billing tickets to the billing agent. A fresh workflow is built per ticket because the specialists are shared executors without reset hooks ([why](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/#bindings-and-per-run-state)); building one costs microseconds.

## Research brief from parallel experts

Three expert agents answer the same question at the same time. The concurrent orchestration's aggregator then hands their answers to an editor agent, which writes the brief. The aggregator is a plain function, so calling another agent from it is ordinary code.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	"github.com/openai/openai-go/v3"
)

func newAgent(name, instructions string) *agent.Agent {
	return openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: instructions,
		Config:       agent.Config{Name: name},
	})
}

func main() {
	experts := []*agent.Agent{
		newAgent("economist", "Answer from an economic point of view in two sentences."),
		newAgent("engineer", "Answer from an engineering point of view in two sentences."),
		newAgent("regulator", "Answer from a regulatory point of view in two sentences."),
	}
	editor := newAgent("editor", "Combine the expert notes into a five-line brief with a recommendation.")

	// The aggregator receives one batch of messages per expert.
	writeBrief := func(ctx context.Context, batches [][]*message.Message) []*message.Message {
		var notes strings.Builder
		for _, batch := range batches {
			if len(batch) > 0 {
				m := batch[len(batch)-1]
				fmt.Fprintf(&notes, "%s: %s\n", m.AuthorName, m.String())
			}
		}
		resp, err := editor.RunText(ctx, "Expert notes:\n"+notes.String()).Collect()
		if err != nil {
			out := message.NewText("brief unavailable: " + err.Error())
			out.Role = message.RoleAssistant
			return []*message.Message{out}
		}
		return resp.Messages
	}

	wf, err := agentworkflow.NewConcurrentWorkflowBuilder(experts...).
		WithName("research-brief").
		WithAggregator(writeBrief).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	run, err := inproc.Default.RunStreaming(ctx, wf,
		[]*message.Message{message.NewText("Should the city replace diesel buses with battery buses by 2030?")})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = run.Close(ctx) }()
	if _, err := run.TrySendMessage(ctx, workflow.TurnToken{EmitEvents: new(true)}); err != nil {
		log.Fatal(err)
	}

	answered := map[string]bool{}
	for evt, err := range run.WatchStream(ctx) {
		if err != nil {
			log.Fatal(err)
		}
		switch e := evt.(type) {
		case workflow.OutputEvent:
			switch out := e.Output.(type) {
			case *agent.ResponseUpdate:
				if out.String() != "" && !answered[out.AuthorName] {
					answered[out.AuthorName] = true
					fmt.Printf("%s has answered\n", out.AuthorName)
				}
			case []*message.Message:
				if !e.IsIntermediate() {
					fmt.Println("--- brief ---")
					fmt.Println(out[len(out)-1].String())
					return
				}
			}
		case workflow.ExecutorFailedEvent:
			log.Fatalf("%s failed: %v", e.ExecutorID, e.Error)
		case workflow.ErrorEvent:
			log.Fatal(e.Error)
		}
	}
}
```

Output against the stand-in. The experts finish in any order, and the stand-in quotes only the first 80 characters of the editor's prompt:

```text
economist has answered
engineer has answered
regulator has answered
--- brief ---
Mock reply to: Expert notes:
engineer: Mock reply to: Should the city replace diesel buses with
```

The aggregator runs inside the workflow, so its context carries the run's deadline and cancellation. The expert batches arrive in completion order; the editor sees them in whatever order they finished.

## Writer and critic loop

A writer drafts, a critic scores the draft with structured output, and the loop continues until the score is good enough or three rounds have passed. The round number travels in the message, so the executors stay stateless and the workflow can be checkpointed safely.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	"github.com/openai/openai-go/v3"
)

const (
	passMark  = 8
	maxRounds = 3
)

type Brief struct{ Topic string }

type Draft struct {
	Topic    string
	Text     string
	Round    int
	Feedback string
}

type Review struct {
	Score    int    `json:"score" jsonschema:"1 to 10"`
	Feedback string `json:"feedback" jsonschema:"what to change"`
}

type Reviewed struct {
	Draft  Draft
	Review Review
}

func newAgent(name, instructions string) *agent.Agent {
	return openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: instructions,
		Config:       agent.Config{Name: name},
	})
}

func main() {
	writer := newAgent("writer", "You write short, concrete taglines.")
	critic := newAgent("critic", "Score taglines strictly from 1 to 10.")

	start := workflow.NewExecutor("brief", func(b Brief) Draft { return Draft{Topic: b.Topic} }).Bind()

	// write handles the first brief and every revision request.
	write := workflow.NewExecutor("writer", func(ctx *workflow.Context, d Draft) (Draft, error) {
		prompt := "Write a product tagline about " + d.Topic
		if d.Feedback != "" {
			prompt = fmt.Sprintf("Revise %q. Feedback: %s", d.Text, d.Feedback)
		}
		resp, err := writer.RunText(ctx, prompt).Collect()
		if err != nil {
			return Draft{}, err
		}
		d.Text, d.Round = resp.String(), d.Round+1
		return d, nil
	}).Bind()

	review := workflow.NewExecutor("critic", func(ctx *workflow.Context, d Draft) (Reviewed, error) {
		var r Review
		if _, err := critic.RunText(ctx, d.Text, agent.WithStructuredOutput(&r)).Collect(); err != nil {
			return Reviewed{}, err
		}
		return Reviewed{Draft: d, Review: r}, nil
	}).Bind()
	revise := workflow.NewExecutor("revise", func(rv Reviewed) Draft {
		d := rv.Draft
		d.Feedback = rv.Review.Feedback
		return d
	}).Bind()
	publish := workflow.NewExecutor("publish", func(rv Reviewed) string {
		verdict := "accepted"
		if rv.Review.Score < passMark {
			verdict = "best effort after the round limit"
		}
		return fmt.Sprintf("round %d, score %d, %s: %s", rv.Draft.Round, rv.Review.Score, verdict, rv.Draft.Text)
	}).Bind()

	done := func(rv Reviewed) bool { return rv.Review.Score >= passMark || rv.Draft.Round >= maxRounds }
	wf, err := workflow.NewBuilder(start).
		AddEdge(start, write).
		AddEdge(write, review).
		AddEdge(review, publish, workflow.WithEdgeCondition(done)).
		AddEdge(review, revise, workflow.WithEdgeCondition(func(rv Reviewed) bool { return !done(rv) })).
		AddEdge(revise, write).
		WithOutputFrom(publish).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	run, err := inproc.Default.Run(ctx, wf, Brief{Topic: "a solar-powered bike light"})
	if err != nil {
		log.Fatal(err)
	}
	for evt := range run.NewEvents() {
		switch e := evt.(type) {
		case workflow.ExecutorCompletedEvent:
			if e.ExecutorID == "critic" {
				rv := e.Result.(Reviewed)
				fmt.Printf("round %d scored %d\n", rv.Draft.Round, rv.Review.Score)
			}
		case workflow.OutputEvent:
			fmt.Println(e.Output)
		case workflow.ExecutorFailedEvent:
			log.Fatalf("%s failed: %v", e.ExecutorID, e.Error)
		}
	}
}
```

Output against the stand-in, whose critic always scores 3 (and which truncates the quoted prompt):

```text
round 1 scored 3
round 2 scored 3
round 3 scored 3
round 3, score 3, best effort after the round limit: Mock reply to: Revise "Mock reply to: Revise \"Mock reply to: Write a product tagline about a s
```

Two conditional edges leave the critic with complementary conditions, and the round limit lives in the condition. Because the Go engine has no superstep limit, that condition, plus the context deadline, is what guarantees the loop ends.

## Expense approvals over HTTP

A small service accepts expense claims. Claims up to £100 are approved automatically; larger ones wait for a manager. Waiting may take days, so the run is checkpointed and the HTTP request returns at once with `202 Accepted`. A later `POST /expenses/{id}/decision` resumes the run from its checkpoint, in whichever process handles the request. The expense ID doubles as the workflow session ID, which is how the decision finds its run.

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync"

	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/checkpoint"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
)

type Expense struct {
	ID       string  `json:"id"`
	Employee string  `json:"employee"`
	Amount   float64 `json:"amount"`
}

type Decision struct {
	Approved bool   `json:"approved"`
	By       string `json:"by"`
}

type Outcome struct {
	ExpenseID string `json:"expense_id"`
	Status    string `json:"status"`
	By        string `json:"by,omitempty"`
}

var managerPort = workflow.RequestPort{
	ID:       "manager",
	Request:  reflect.TypeFor[Expense](),
	Response: reflect.TypeFor[Decision](),
}

// policy auto-approves small claims and sends the rest to the manager port.
type policy struct {
	_ workflow.AttrSendsMessage[Expense]
	_ workflow.AttrYieldsOutput[Outcome]
}

func (policy) Handle(ctx *workflow.Context, e Expense) error {
	if err := ctx.QueueStateUpdate("expense", "claim", e); err != nil {
		return err
	}
	if e.Amount <= 100 {
		return ctx.YieldOutput(Outcome{ExpenseID: e.ID, Status: "approved", By: "policy"})
	}
	return ctx.SendMessage("", e)
}

func buildWorkflow() *workflow.Workflow {
	check := workflow.NewExecutor("policy", policy{}).Bind()
	ask := managerPort.Bind()
	record := workflow.NewExecutor("record", func(ctx *workflow.Context, d Decision) (Outcome, error) {
		v, err := ctx.ReadState("expense", "claim")
		if err != nil {
			return Outcome{}, err
		}
		status := "rejected"
		if d.Approved {
			status = "approved"
		}
		return Outcome{ExpenseID: v.(Expense).ID, Status: status, By: d.By}, nil
	}).Bind()
	wf, err := workflow.NewBuilder(check).AddEdge(check, ask).AddEdge(ask, record).
		WithOutputFrom(check, record).Build()
	if err != nil {
		log.Fatal(err)
	}
	return wf
}

type service struct {
	mgr     checkpoint.Manager
	mu      sync.Mutex
	pending map[string]Expense
}

// drive watches a run and reports an outcome, if any. Without an answer it
// stops at the first halt and leaves requests pending; with one it keeps
// watching until the answered run produces its outcome.
func drive(ctx context.Context, run *inproc.StreamingRun, answer *Decision) (*Outcome, error) {
	events := run.WatchUntilHalt(ctx)
	if answer != nil {
		events = run.WatchStream(ctx)
	}
	for evt, err := range events {
		if err != nil {
			return nil, err
		}
		switch e := evt.(type) {
		case workflow.RequestInfoEvent:
			if answer == nil {
				continue // leave it pending
			}
			resp, err := e.Request.CreateResponse(*answer)
			if err != nil {
				return nil, err
			}
			if err := run.SendResponse(ctx, resp); err != nil {
				return nil, err
			}
			answer = nil
		case workflow.OutputEvent:
			o := e.Output.(Outcome)
			return &o, nil
		case workflow.ExecutorFailedEvent:
			return nil, e.Error
		case workflow.ErrorEvent:
			return nil, e.Error
		}
	}
	return nil, nil
}

func (s *service) submit(w http.ResponseWriter, r *http.Request) {
	var e Expense
	if err := json.NewDecoder(r.Body).Decode(&e); err != nil || e.ID == "" {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	env := inproc.Default.WithCheckpointing(s.mgr)
	run, err := env.RunStreaming(r.Context(), buildWorkflow(), e, inproc.WithSessionID(e.ID))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer func() { _ = run.Close(r.Context()) }()
	outcome, err := drive(r.Context(), run, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if outcome == nil {
		s.mu.Lock()
		s.pending[e.ID] = e
		s.mu.Unlock()
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(Outcome{ExpenseID: e.ID, Status: "awaiting manager"})
		return
	}
	_ = json.NewEncoder(w).Encode(outcome)
}

func (s *service) decide(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var d Decision
	if err := json.NewDecoder(r.Body).Decode(&d); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	latest, err := s.mgr.LatestCheckpoint(r.Context(), id)
	if err != nil || latest == nil {
		http.Error(w, "no pending expense "+id, http.StatusNotFound)
		return
	}
	run, err := inproc.Default.WithCheckpointing(s.mgr).ResumeStreaming(r.Context(), buildWorkflow(), *latest)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer func() { _ = run.Close(r.Context()) }()
	outcome, err := drive(r.Context(), run, &d)
	if err != nil || outcome == nil {
		http.Error(w, fmt.Sprint("resume failed: ", err), http.StatusInternalServerError)
		return
	}
	s.mu.Lock()
	delete(s.pending, id)
	s.mu.Unlock()
	_ = json.NewEncoder(w).Encode(outcome)
}

func (s *service) list(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Expense, 0, len(s.pending))
	for _, e := range s.pending {
		out = append(out, e)
	}
	_ = json.NewEncoder(w).Encode(out)
}

func call(method, url, body string) string {
	req, _ := http.NewRequest(method, url, bytes.NewBufferString(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	b, _ := io.ReadAll(resp.Body)
	return fmt.Sprintf("%d %s", resp.StatusCode, bytes.TrimSpace(b))
}

func main() {
	dir, err := os.MkdirTemp("", "expenses")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()
	store, err := checkpoint.NewFileSystemJSONStore(dir) // one process; use a database store for replicas
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	svc := &service{mgr: checkpoint.NewJSONManager(store), pending: map[string]Expense{}}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /expenses", svc.submit)
	mux.HandleFunc("GET /expenses/pending", svc.list)
	mux.HandleFunc("POST /expenses/{id}/decision", svc.decide)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	fmt.Println(call("POST", srv.URL+"/expenses", `{"id":"EX-1","employee":"sam","amount":42.50}`))
	fmt.Println(call("POST", srv.URL+"/expenses", `{"id":"EX-2","employee":"kim","amount":1250}`))
	fmt.Println(call("GET", srv.URL+"/expenses/pending", ""))
	fmt.Println(call("POST", srv.URL+"/expenses/EX-2/decision", `{"approved":true,"by":"lee"}`))
	fmt.Println(call("GET", srv.URL+"/expenses/pending", ""))
}
```

Output:

```text
200 {"expense_id":"EX-1","status":"approved","by":"policy"}
202 {"expense_id":"EX-2","status":"awaiting manager"}
200 [{"id":"EX-2","employee":"kim","amount":1250}]
200 {"expense_id":"EX-2","status":"approved","by":"lee"}
200 []
```

The pending list here is an in-memory map for brevity. In a real service, keep it in your database next to the checkpoints, and add authentication and an authorisation check that the caller may approve the claim. The resumed run re-emits the pending request (`WithPendingRequestRepublish` defaults to true), which is what lets `drive` answer it.

## Delegating to a remote agent over A2A

An inventory team runs its agent as an A2A service. A concierge agent in another service discovers it from its agent card and uses it as a tool. Both halves run in one process here; in practice they are separate deployments.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient/agentcard"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/agenttool"
	"github.com/openai/openai-go/v3"
)

// startInventoryService is the other team's service.
func startInventoryService() string {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}
	url := "http://" + ln.Addr().String()

	inventory := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You know stock levels for every warehouse.",
		Config:       agent.Config{Name: "inventory"},
	})
	card := &a2a.AgentCard{
		Name:               "inventory",
		Description:        "Answers stock-level questions for all warehouses.",
		Version:            "2.1.0",
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		Skills: []a2a.AgentSkill{{ID: "stock", Name: "Stock levels",
			Description: "How many units of a product are in stock.", Tags: []string{"inventory"}}},
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(url, a2a.TransportProtocolHTTPJSON),
			a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC),
		},
	}
	h := a2aprovider.NewHandler(inventory, a2aprovider.ExecutorConfig{})
	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle("POST /{$}", a2asrv.NewJSONRPCHandler(h))
	mux.Handle("/", a2asrv.NewRESTHandler(h))
	go func() { _ = http.Serve(ln, mux) }()
	return url
}

func main() {
	ctx := context.Background()
	inventoryURL := startInventoryService()

	// The concierge's service: discover the remote agent and wrap it as a tool.
	remote, err := a2aprovider.NewAgentFromResolver(ctx, agentcard.DefaultResolver, inventoryURL, a2aprovider.AgentConfig{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("discovered %q: %s\n", remote.Name(), remote.Description())

	concierge := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You help shop staff. Ask the inventory agent about stock.",
		Config: agent.Config{
			Name:  "concierge",
			Tools: []tool.Tool{agenttool.New(remote, agenttool.Config{})}, // tool name: "inventory"
		},
	})
	resp, err := concierge.RunText(ctx, "Do we have the blue kettle in Leeds?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Output against the stand-in, whose concierge calls the only tool with a sample query:

```text
discovered "inventory": Answers stock-level questions for all warehouses.
Here is what the tool returned: Mock reply to: returns policy
```

The tool's description is the remote card's description, which is what the calling model reads when deciding whether to use it, so write cards for a model audience. Each tool call is a separate A2A request with no shared session; give the agent tool a session-aware wrapper if the remote agent needs conversation history across calls.

## Support agent with approvals and saved sessions

A support agent can look up orders freely but needs a person to approve refunds. The conversation is saved to disk after each turn and loaded by a freshly built agent, as a new process would. The first refund is approved; the second is denied, and the tool does not run.

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type refundIn struct {
	Order  string  `json:"order" jsonschema:"the order ID"`
	Amount float64 `json:"amount" jsonschema:"refund amount in pounds"`
}

type lookupIn struct {
	Order string `json:"order" jsonschema:"the order ID"`
}

var refundsIssued int

func newSupportAgent() *agent.Agent {
	refund := functool.MustNew(functool.Config{Name: "issue_refund", Description: "Refund an order."},
		func(_ context.Context, in refundIn) (string, error) {
			refundsIssued++
			return fmt.Sprintf("refunded £%.2f on %s", in.Amount, in.Order), nil
		})
	lookup := functool.MustNew(functool.Config{Name: "lookup_order", Description: "Order status."},
		func(_ context.Context, in lookupIn) (string, error) { return in.Order + " was delivered on 3 May", nil })

	return openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You are a support agent. Refunds need approval.",
		// The Responses API keeps the conversation service-side; the session
		// file holds the conversation's response ID. See the note below.
		Config: agent.Config{
			Name:  "support",
			Tools: []tool.Tool{tool.ApprovalRequiredFunc(refund), lookup},
		},
	})
}

// turn runs one user message, asking the "human" about each approval.
func turn(ctx context.Context, a *agent.Agent, s *agent.Session, text string, approve bool) {
	fmt.Println("user:", text)
	resp, err := a.RunText(ctx, text, agent.WithSession(s)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	for {
		var answers []message.Content
		for c := range resp.Contents() {
			if req, ok := c.(*message.ToolApprovalRequestContent); ok {
				call := req.ToolCall.(*message.FunctionCallContent)
				fmt.Printf("  approval needed for %s(%s): %t\n", call.Name, call.Arguments, approve)
				answers = append(answers, req.CreateResponse(approve, "checked by duty manager"))
			}
		}
		if len(answers) == 0 {
			break
		}
		if resp, err = a.RunMessage(ctx, message.New(answers...), agent.WithSession(s)).Collect(); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println("agent:", resp.String())
}

func save(path string, s *agent.Session) {
	data, err := json.Marshal(s)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		log.Fatal(err)
	}
}

func load(path string) *agent.Session {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	var s agent.Session
	if err := json.Unmarshal(data, &s); err != nil {
		log.Fatal(err)
	}
	return &s
}

func main() {
	ctx := context.Background()
	path := filepath.Join(os.TempDir(), "support-session.json")
	defer func() { _ = os.Remove(path) }()

	a := newSupportAgent()
	s, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	turn(ctx, a, s, "Order A-77 arrived broken, please refund it.", true)
	save(path, s)

	// Later: a new process builds the agent again and loads the conversation.
	b := newSupportAgent()
	restored := load(path)
	turn(ctx, b, restored, "Refund my other order too.", false)
	save(path, restored)

	fmt.Printf("refunds issued: %d, service conversation ID saved: %t\n", refundsIssued, restored.ServiceID() != "")
}
```

Output against the stand-in, which calls the first tool (the refund) each time:

```text
user: Order A-77 arrived broken, please refund it.
  approval needed for issue_refund({"amount": 21.5, "order": "example order"}): true
agent: Here is what the tool returned: refunded £21.50 on example order
user: Refund my other order too.
  approval needed for issue_refund({"amount": 21.5, "order": "example order"}): false
agent: Here is what the tool returned: Tool call invocation rejected. checked by duty manager
refunds issued: 1, service conversation ID saved: true
```

A denied call is reported to the model as a rejection, with your reason, and the model carries on.

**Keep the service-side history here.** With the history kept in the session instead (Chat Completions, or Responses with `DisableStoreOutput: true`), a *second* approval-gated call on the same session failed in v%%version=github.com/microsoft/agent-framework-go%% with `ToolApprovalRequestContent found with ToolCall.CallID(s) '…' that have no matching ToolApprovalResponseContent`, whether or not the session had been saved and reloaded. The first approval always worked. We reproduced this against the stand-in only, so check it against your provider before relying on locally held history with repeated approvals. Approval requests are bound to the session that raised them, so answer them with the same session, after loading it if necessary. The [tools page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#approvals) covers standing approvals ("always allow this tool").

## Traced data-quality pipeline

A nightly job checks several tables in parallel, merges the findings, and asks an agent for a short summary for the on-call channel. Both the workflow and the agent are traced; the spans go to an in-memory exporter here so that the program can print a summary of them, and to OTLP in production.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"maps"
	"slices"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/provider/otelprovider"
	"github.com/microsoft/agent-framework-go/workflow"
	"github.com/microsoft/agent-framework-go/workflow/inproc"
	workflowotel "github.com/microsoft/agent-framework-go/workflow/observability/opentelemetry"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type Row map[string]any

var tables = map[string][]Row{
	"customers": {{"id": 1, "email": "a@example.com"}, {"id": 2, "email": nil}, {"id": 3, "email": "c@example.com"}},
	"orders":    {{"id": 10, "total": 25.0}, {"id": 11, "total": -4.0}},
	"payments":  {{"id": 90, "order_id": 10}},
}

type Job struct{ Night string }

type Finding struct {
	Table  string
	Issues []string
}

type Report struct {
	Seen     int
	Findings []Finding
}

func checker(table string) workflow.ExecutorBinding {
	return workflow.NewExecutor("check_"+table, func(Job) Finding {
		f := Finding{Table: table}
		for _, r := range tables[table] {
			for col, v := range r {
				switch x := v.(type) {
				case nil:
					f.Issues = append(f.Issues, fmt.Sprintf("row %v: %s is null", r["id"], col))
				case float64:
					if x < 0 {
						f.Issues = append(f.Issues, fmt.Sprintf("row %v: %s is negative", r["id"], col))
					}
				}
			}
		}
		return f
	}).Bind()
}

func main() {
	ctx := context.Background()
	spans := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(spans))
	defer func() { _ = tp.Shutdown(ctx) }()
	otel.SetTracerProvider(tp)

	summariser := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Summarise data-quality findings for an on-call engineer in two lines.",
		Config: agent.Config{
			Name:        "dq-summariser",
			Middlewares: []agent.Middleware{otelprovider.NewMiddleware(otelprovider.MiddlewareConfig{SourceName: "dq-nightly"})},
		},
	})

	start := workflow.NewExecutor("start", func(j Job) Job { return j }).Bind()
	var checks []workflow.ExecutorBinding
	for _, t := range slices.Sorted(maps.Keys(tables)) {
		checks = append(checks, checker(t))
	}
	merge := workflow.NewAggregatingExecutor("merge", func(r *Report, f Finding) *Report {
		if r == nil {
			r = &Report{}
		}
		r.Seen++
		if len(f.Issues) > 0 {
			r.Findings = append(r.Findings, f)
		}
		return r
	}).Bind()
	summarise := workflow.NewExecutor("summarise", func(wctx *workflow.Context, r Report) (string, error) {
		if len(r.Findings) == 0 {
			return "all tables clean", nil
		}
		slices.SortFunc(r.Findings, func(a, b Finding) int { return strings.Compare(a.Table, b.Table) })
		var b strings.Builder
		for _, f := range r.Findings {
			fmt.Fprintf(&b, "%s: %s\n", f.Table, strings.Join(f.Issues, "; "))
		}
		resp, err := summariser.RunText(wctx, b.String()).Collect()
		if err != nil {
			return "", err
		}
		return resp.String(), nil
	}).Bind()

	wf, err := workflow.NewBuilder(start).
		AddFanOutEdge(start, checks).
		AddFanInBarrierEdge(checks, merge).
		AddEdge(merge, summarise, workflow.WithEdgeCondition(func(r Report) bool { return r.Seen == len(checks) })).
		WithOutputFrom(summarise).
		WithName("dq-nightly").
		WithTelemetry(workflowotel.New(workflowotel.Config{SourceName: "dq-nightly"}), workflow.TelemetryOptions{}).
		Build()
	if err != nil {
		log.Fatal(err)
	}

	run, err := inproc.Default.Run(ctx, wf, Job{Night: "2026-10-24"})
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
	_ = run.Close(ctx)

	counts := map[string]int{}
	for _, s := range spans.GetSpans() {
		name, _, _ := strings.Cut(s.Name, " ")
		counts[name]++
	}
	for _, name := range slices.Sorted(maps.Keys(counts)) {
		fmt.Printf("%-20s %d\n", name, counts[name])
	}
}
```

Output against the stand-in:

```text
Mock reply to: customers: row 2: email is null
orders: row 11: total is negative

chat                 1
edge_group.process   8
executor.process     8
invoke_agent         1
message.send         8
workflow.build       1
workflow.session     1
workflow_invoke      1
```

The checks are pure Go and run in parallel within one superstep; only the summary costs a model call, and it is skipped when everything is clean. The agent's `invoke_agent` and `chat` spans sit under the `summarise` executor's span in the same trace.

## Validated extraction with a retry

Structured output guarantees the JSON shape, not that the values make sense. This recipe extracts action items from meeting notes, validates them in Go (owners must be attendees, dates must be real dates), and if validation fails, asks the model once more with the errors. If the second attempt also fails, the items go to a human review queue rather than into the tracker.

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

type ActionItem struct {
	Task  string `json:"task" jsonschema:"what needs doing"`
	Owner string `json:"owner" jsonschema:"one of the attendees"`
	Due   string `json:"due" jsonschema:"due date as YYYY-MM-DD"`
}

type Extraction struct {
	Items []ActionItem `json:"items"`
}

func validate(x Extraction, attendees []string) error {
	var errs []error
	if len(x.Items) == 0 {
		errs = append(errs, errors.New("no action items found"))
	}
	for i, it := range x.Items {
		if !slices.Contains(attendees, it.Owner) {
			errs = append(errs, fmt.Errorf("item %d: owner %q is not an attendee", i+1, it.Owner))
		}
		if _, err := time.Parse(time.DateOnly, it.Due); err != nil {
			errs = append(errs, fmt.Errorf("item %d: due date %q is not YYYY-MM-DD", i+1, it.Due))
		}
	}
	return errors.Join(errs...)
}

// extract asks once, validates, and asks again with the errors if needed.
func extract(ctx context.Context, a *agent.Agent, notes string, attendees []string) (Extraction, error) {
	prompt := fmt.Sprintf("Attendees: %s\n\nNotes:\n%s", strings.Join(attendees, ", "), notes)
	var lastErr error
	for attempt := 1; attempt <= 2; attempt++ {
		var x Extraction
		if _, err := a.RunText(ctx, prompt, agent.WithStructuredOutput(&x)).Collect(); err != nil {
			return Extraction{}, err
		}
		if lastErr = validate(x, attendees); lastErr == nil {
			return x, nil
		}
		fmt.Printf("attempt %d rejected:\n  %s\n", attempt, strings.ReplaceAll(lastErr.Error(), "\n", "\n  "))
		prompt += "\n\nYour previous answer had these problems; fix them:\n" + lastErr.Error()
	}
	return Extraction{}, fmt.Errorf("needs human review: %w", lastErr)
}

func main() {
	extractor := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Extract action items from meeting notes. Use only attendees as owners.",
		Config:       agent.Config{Name: "extractor"},
	})
	notes := `Priya will draft the Q4 roadmap by next Friday.
Tom to fix the flaky login test before the release on 2026-11-02.`

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	items, err := extract(ctx, extractor, notes, []string{"Priya", "Tom", "Ana"})
	if err != nil {
		fmt.Println("queued for review:", errors.Unwrap(err) != nil)
		return
	}
	for _, it := range items.Items {
		fmt.Printf("%s -> %s by %s\n", it.Task, it.Owner, it.Due)
	}
}
```

Output against the stand-in, whose sample values fail both checks:

```text
attempt 1 rejected:
  item 1: owner "example owner" is not an attendee
  item 1: due date "example due" is not YYYY-MM-DD
attempt 2 rejected:
  item 1: owner "example owner" is not an attendee
  item 1: due date "example due" is not YYYY-MM-DD
queued for review: true
```

A real model usually passes on the first or second attempt. Keep the retry count small: a model that fails twice on the same constraint rarely succeeds on the fifth try, and each attempt costs a full call.

## A workflow behind an AG-UI endpoint

A translation pipeline (translate, then review) is wrapped as an agent and served over AG-UI, so a web front end can stream it like a single assistant. The Go AG-UI client stands in for the front end. The participants here are offline stand-ins, so this recipe needs no model at all.

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"net/http/httptest"
	"strings"

	aguiSSEClient "github.com/ag-ui-protocol/ag-ui/sdks/community/go/pkg/client/sse"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/aguiprovider"
	"github.com/microsoft/agent-framework-go/workflow/agentworkflow"
	"github.com/sirupsen/logrus"
)

// roleAgent is an offline participant; swap in openaiprovider.NewAgent for real use.
func roleAgent(name string, reply func(last string) string) *agent.Agent {
	run := func(_ context.Context, msgs []*message.Message, _ ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			yield(&agent.ResponseUpdate{Role: message.RoleAssistant, MessageID: fmt.Sprint(name, len(msgs)),
				Contents: message.Contents{&message.TextContent{Text: reply(msgs[len(msgs)-1].String())}}}, nil)
		}
	}
	return agent.New(agent.ProviderConfig{ProviderName: "offline", Run: run}, agent.Config{Name: name})
}

func main() {
	logrus.SetLevel(logrus.WarnLevel)
	ctx := context.Background()

	translator := roleAgent("translator", func(last string) string {
		return "NL: " + strings.ReplaceAll(last, "Good morning", "Goedemorgen")
	})
	reviewer := roleAgent("reviewer", func(last string) string { return "Reviewed: " + last })
	wf, err := agentworkflow.NewSequentialWorkflowBuilder(translator, reviewer).
		WithChainOnlyAgentResponses(true).
		Build()
	if err != nil {
		log.Fatal(err)
	}
	pipeline, err := agentworkflow.NewAgent(wf, agentworkflow.AgentConfig{
		Config: agent.Config{Name: "translation-pipeline"},
	})
	if err != nil {
		log.Fatal(err)
	}

	srv := httptest.NewServer(aguiprovider.NewJSONHTTPHandler(pipeline, aguiprovider.HandlerConfig{}))
	defer srv.Close()

	ui := aguiprovider.NewAgent(aguiSSEClient.NewClient(aguiSSEClient.Config{Endpoint: srv.URL}), aguiprovider.AgentConfig{})
	session, err := ui.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for update, err := range ui.RunText(ctx, "Good morning, team", agent.WithSession(session), agent.Stream(true)) {
		if err != nil {
			log.Fatal(err)
		}
		if text := update.String(); text != "" {
			fmt.Printf("[%s] %s\n", update.AuthorName, text)
		}
	}
}
```

Output:

```text
[] NL: Goedemorgen, team
[] Reviewed: NL: Goedemorgen, team
```

The updates arrive without an author name: the AG-UI event stream carries text and message IDs but not which participant wrote them, so put the participant into the text if the front end should show it. The AG-UI handler is stateless, and so is a workflow agent run without a session: each request starts a fresh workflow run with the history the client sends. That suits AG-UI, whose clients send the whole conversation every time. Approvals inside the workflow do not cross AG-UI in v%%version=github.com/microsoft/agent-framework-go%% ([details](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_hosting_and_a2a/#frontend-tools)); serve approval-gated workflows over A2A instead.

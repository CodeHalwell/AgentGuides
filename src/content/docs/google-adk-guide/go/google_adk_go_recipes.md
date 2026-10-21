---
title: "Google ADK for Go - Recipes"
description: "Task-shaped, compile-checked ADK for Go programs: tools, structured output, multi-agent patterns, persistent sessions, streaming, HTTP serving, workers and Cloud Storage artifacts."
framework: google-adk
language: go
---

# Google ADK for Go - Recipes

**Verified against:** `google.golang.org/adk` v%%version=google.golang.org/adk%% (checked %%versions-checked%%). Each recipe is a complete program that compiles with `go vet` against that release. Recipes that only build an agent take the model as a `model.LLM` parameter; create one with `gemini.NewModel` as in [Hello World](#hello-world).

Concepts are explained in the [comprehensive guide](/google-adk-guide/go/google_adk_go_comprehensive_guide/); deployment is in the [production guide](/google-adk-guide/go/google_adk_go_production_guide/).

---

## Table of Contents

1. [Hello World](#hello-world)
2. [Basic Agents](#basic-agents)
3. [Tools](#tools)
4. [Multi-Agent Systems](#multi-agent-systems)
5. [Sessions and State](#sessions-and-state)
6. [Streaming](#streaming)
7. [MCP Integration](#mcp-integration)
8. [A2A Protocol](#a2a-protocol)
9. [Production Patterns](#production-patterns)
10. [Cloud Integration](#cloud-integration)

---

## Hello World

### Simplest Agent

```go
// Hello World: an agent you can chat with in the terminal (`go run . console`)
// or in the ADK dev UI (`go run . web api webui`).
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model/gemini"
)

func main() {
	ctx := context.Background()

	m, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "hello_agent",
		Model:       m,
		Instruction: "You are a friendly assistant. Keep answers short.",
	})
	if err != nil {
		log.Fatal(err)
	}

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{AgentLoader: agent.NewSingleLoader(a)}, os.Args[1:]); err != nil {
		log.Fatalf("%v\n\n%s", err, l.CommandLineSyntax())
	}
}
```

```bash
export GOOGLE_API_KEY=...
go run . console
```

---

## Basic Agents

### Q&A Bot with Tools

```go
// Q&A bot with tools: a terminal loop over one session, so the agent
// remembers earlier turns.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type timeArgs struct {
	Timezone string `json:"timezone" jsonschema:"IANA time zone, for example Europe/London"`
}

type timeResult struct {
	Time string `json:"time"`
}

func currentTime(_ agent.ToolContext, a timeArgs) (timeResult, error) {
	loc, err := time.LoadLocation(a.Timezone)
	if err != nil {
		return timeResult{}, fmt.Errorf("unknown time zone %q", a.Timezone)
	}
	return timeResult{Time: time.Now().In(loc).Format(time.RFC1123)}, nil
}

func main() {
	ctx := context.Background()

	m, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
	if err != nil {
		log.Fatal(err)
	}

	clock, err := functiontool.New(functiontool.Config{
		Name:        "current_time",
		Description: "Returns the current time in a time zone.",
	}, currentTime)
	if err != nil {
		log.Fatal(err)
	}

	bot, err := llmagent.New(llmagent.Config{
		Name:        "qa_bot",
		Model:       m,
		Instruction: "Answer questions. Use current_time whenever the user asks about the time.",
		Tools:       []tool.Tool{clock},
	})
	if err != nil {
		log.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "qa_bot",
		Agent:             bot,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	in := bufio.NewScanner(os.Stdin)
	for fmt.Print("> "); in.Scan(); fmt.Print("> ") {
		msg := genai.NewContentFromText(in.Text(), genai.RoleUser)
		for event, err := range r.Run(ctx, "local-user", "chat", msg, agent.RunConfig{}) {
			if err != nil {
				log.Printf("error: %v", err)
				break
			}
			if event.IsFinalResponse() && event.Content != nil {
				for _, p := range event.Content.Parts {
					fmt.Print(p.Text)
				}
				fmt.Println()
			}
		}
	}
}
```

### Structured Output Agent

```go
// Structured output: the agent replies with JSON matching a schema, and Go
// code decodes it into a struct.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

type Recipe struct {
	Title       string   `json:"title"`
	Ingredients []string `json:"ingredients"`
	Minutes     int      `json:"minutes"`
}

func main() {
	ctx := context.Background()

	m, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
	if err != nil {
		log.Fatal(err)
	}

	chef, err := llmagent.New(llmagent.Config{
		Name:        "chef",
		Model:       m,
		Instruction: "Suggest one recipe for the user's request.",
		OutputSchema: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"title":       {Type: genai.TypeString},
				"ingredients": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
				"minutes":     {Type: genai.TypeInteger},
			},
			Required: []string{"title", "ingredients", "minutes"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		AppName: "chef", Agent: chef, SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	var raw strings.Builder
	msg := genai.NewContentFromText("Something vegetarian and quick", genai.RoleUser)
	for event, err := range r.Run(ctx, "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		if event.IsFinalResponse() && event.Content != nil {
			for _, p := range event.Content.Parts {
				raw.WriteString(p.Text)
			}
		}
	}

	var recipe Recipe
	if err := json.Unmarshal([]byte(raw.String()), &recipe); err != nil {
		log.Fatalf("decode %q: %v", raw.String(), err)
	}
	fmt.Printf("%s (%d min): %s\n", recipe.Title, recipe.Minutes, strings.Join(recipe.Ingredients, ", "))
}
```

---

## Tools

### Function Tool - Weather API

```go
// A function tool that calls an external HTTP API (Open-Meteo, no key needed).
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

var httpClient = &http.Client{Timeout: 10 * time.Second}

type forecastArgs struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

type forecastResult struct {
	TemperatureC float64 `json:"temperature_c"`
	WindKmh      float64 `json:"wind_kmh"`
}

func forecast(ctx agent.ToolContext, a forecastArgs) (forecastResult, error) {
	q := url.Values{}
	q.Set("latitude", fmt.Sprint(a.Latitude))
	q.Set("longitude", fmt.Sprint(a.Longitude))
	q.Set("current", "temperature_2m,wind_speed_10m")

	// ToolContext is a context.Context: the request is cancelled with the run.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		"https://api.open-meteo.com/v1/forecast?"+q.Encode(), nil)
	if err != nil {
		return forecastResult{}, err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return forecastResult{}, fmt.Errorf("weather service unavailable: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return forecastResult{}, fmt.Errorf("weather service returned %s", resp.Status)
	}

	var body struct {
		Current struct {
			Temperature float64 `json:"temperature_2m"`
			Wind        float64 `json:"wind_speed_10m"`
		} `json:"current"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return forecastResult{}, err
	}
	return forecastResult{TemperatureC: body.Current.Temperature, WindKmh: body.Current.Wind}, nil
}

func newWeatherAgent(m model.LLM) (agent.Agent, error) {
	t, err := functiontool.New(functiontool.Config{
		Name:        "forecast",
		Description: "Current temperature and wind for a latitude/longitude.",
	}, forecast)
	if err != nil {
		return nil, err
	}
	return llmagent.New(llmagent.Config{
		Name:        "weather_agent",
		Model:       m,
		Instruction: "Answer weather questions. Work out the coordinates of the place yourself, then call forecast.",
		Tools:       []tool.Tool{t},
	})
}

func main() {
	var m model.LLM
	if _, err := newWeatherAgent(m); err != nil {
		log.Fatal(err)
	}
}
```

Return an `error` for upstream failures: the model sees the message and can tell the user, retry, or try something else.

### Database Tool

```go
// A read-only database tool over database/sql. Bring your own driver
// (for example github.com/jackc/pgx/v5/stdlib) and pass in the *sql.DB.
package main

import (
	"database/sql"
	"fmt"
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type orderArgs struct {
	CustomerEmail string `json:"customer_email"`
	Limit         int    `json:"limit,omitempty" jsonschema:"maximum rows, default 5"`
}

type order struct {
	ID     int64   `json:"id"`
	Status string  `json:"status"`
	Total  float64 `json:"total"`
}

type orderResult struct {
	Orders []order `json:"orders"`
}

// newOrdersTool exposes one parameterised query rather than free-form SQL,
// so the model cannot read or change anything else.
func newOrdersTool(db *sql.DB) (tool.Tool, error) {
	recentOrders := func(ctx agent.ToolContext, a orderArgs) (orderResult, error) {
		if a.Limit <= 0 || a.Limit > 20 {
			a.Limit = 5
		}
		rows, err := db.QueryContext(ctx,
			`SELECT id, status, total FROM orders WHERE customer_email = $1 ORDER BY created_at DESC LIMIT $2`,
			a.CustomerEmail, a.Limit)
		if err != nil {
			return orderResult{}, fmt.Errorf("query orders: %w", err)
		}
		defer rows.Close()

		var res orderResult
		for rows.Next() {
			var o order
			if err := rows.Scan(&o.ID, &o.Status, &o.Total); err != nil {
				return orderResult{}, err
			}
			res.Orders = append(res.Orders, o)
		}
		return res, rows.Err()
	}

	return functiontool.New(functiontool.Config{
		Name:        "recent_orders",
		Description: "Lists a customer's most recent orders.",
	}, recentOrders)
}

func newOrdersAgent(m model.LLM, db *sql.DB) (agent.Agent, error) {
	t, err := newOrdersTool(db)
	if err != nil {
		return nil, err
	}
	return llmagent.New(llmagent.Config{
		Name:        "orders_agent",
		Model:       m,
		Instruction: "Answer questions about a customer's orders using recent_orders.",
		Tools:       []tool.Tool{t},
	})
}

func main() {
	var m model.LLM
	var db *sql.DB // sql.Open("pgx", os.Getenv("DATABASE_URL"))
	if _, err := newOrdersAgent(m, db); err != nil {
		log.Fatal(err)
	}
}
```

Expose narrow, parameterised queries rather than a "run any SQL" tool, and use a database role with read-only grants.

---

## Multi-Agent Systems

### Sequential Workflow and Parallel Analysis

Sequential pipelines and parallel fan-out are in the comprehensive guide under [Agent Orchestration](/google-adk-guide/go/google_adk_go_comprehensive_guide/#agent-orchestration).

### Supervisor-Worker Pattern

```go
// Supervisor-worker: a supervisor calls specialist agents as tools and
// writes the final answer itself.
package main

import (
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
)

func newSupervisor(m model.LLM) (agent.Agent, error) {
	worker := func(name, description, instruction string) (tool.Tool, error) {
		a, err := llmagent.New(llmagent.Config{
			Name:        name,
			Description: description,
			Model:       m,
			Instruction: instruction,
		})
		if err != nil {
			return nil, err
		}
		return agenttool.New(a, nil), nil
	}

	researcher, err := worker("researcher", "Finds facts about a topic.", "List key facts about the request.")
	if err != nil {
		return nil, err
	}
	analyst, err := worker("analyst", "Analyses facts and draws conclusions.", "Analyse the facts you are given.")
	if err != nil {
		return nil, err
	}
	writer, err := worker("writer", "Turns analysis into a short report.", "Write a concise report from the analysis.")
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:  "supervisor",
		Model: m,
		Instruction: "Answer the user's request by calling researcher, then analyst, then writer, " +
			"passing each one the previous result. Return the writer's report.",
		Tools: []tool.Tool{researcher, analyst, writer},
	})
}

func main() {
	var m model.LLM
	if _, err := newSupervisor(m); err != nil {
		log.Fatal(err)
	}
}
```

With `agenttool` the supervisor stays in charge and sees every worker's output. Use `SubAgents` instead when a specialist should take over the conversation with the user.

---

## Sessions and State

### Multi-Turn Conversation

```go
// Multi-turn conversation with sessions that survive restarts (SQLite via GORM).
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/glebarez/sqlite"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/session/database"
)

const appName = "notes_assistant"

// ensureSession returns the existing session or creates it with initial state.
func ensureSession(ctx context.Context, svc session.Service, userID, sessionID string) error {
	_, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err == nil {
		return nil
	}
	if !errors.Is(err, session.ErrNotFound) {
		return err
	}
	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		State:     map[string]any{"user:display_name": "Sam"},
	})
	return err
}

func main() {
	ctx := context.Background()

	sessions, err := database.NewSessionService(sqlite.Open("sessions.db"))
	if err != nil {
		log.Fatal(err)
	}
	if err := database.AutoMigrate(sessions); err != nil {
		log.Fatal(err)
	}

	m, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "notes",
		Model:       m,
		Instruction: "You are {user:display_name}'s note-taking assistant. Remember what they tell you.",
	})
	if err != nil {
		log.Fatal(err)
	}

	r, err := runner.New(runner.Config{AppName: appName, Agent: a, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}

	if err := ensureSession(ctx, sessions, "sam", "notes-1"); err != nil {
		log.Fatal(err)
	}
	for _, text := range []string{"Remember: the dentist is on Tuesday.", "When is the dentist?"} {
		msg := genai.NewContentFromText(text, genai.RoleUser)
		for event, err := range r.Run(ctx, "sam", "notes-1", msg, agent.RunConfig{}) {
			if err != nil {
				log.Fatal(err)
			}
			if event.IsFinalResponse() && event.Content != nil {
				fmt.Println(event.Content.Parts[0].Text)
			}
		}
	}
}
```

### Persistent Sessions

The recipe above already persists to SQLite. For PostgreSQL, swap the dialector for `gorm.io/driver/postgres` (`postgres.Open(dsn)`); for managed storage on Google Cloud use `session/vertexai`. ADK for Go has no Firestore session service.

---

## Streaming

### Stream Response Chunks

```go
// Streaming: print text as it arrives instead of waiting for the full reply.
package main

import (
	"context"
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
)

func streamReply(ctx context.Context, r *runner.Runner, userID, sessionID, text string) error {
	msg := genai.NewContentFromText(text, genai.RoleUser)
	cfg := agent.RunConfig{StreamingMode: agent.StreamingModeSSE}

	for event, err := range r.Run(ctx, userID, sessionID, msg, cfg) {
		if err != nil {
			return err
		}
		if event.Content == nil {
			continue
		}
		// Partial events carry incremental text; the final, non-partial event
		// repeats the complete message, so skip it to avoid printing twice.
		if event.Partial {
			for _, p := range event.Content.Parts {
				fmt.Print(p.Text)
			}
		}
	}
	fmt.Println()
	return nil
}

func main() {
	var r *runner.Runner
	_ = streamReply(context.Background(), r, "u1", "s1", "Tell me a short story")
}
```

### HTTP Streaming Endpoint

```go
// An HTTP endpoint that streams the agent's reply to the browser as
// Server-Sent Events.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
)

func chatHandler(r *runner.Runner) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		var body struct {
			UserID    string `json:"user_id"`
			SessionID string `json:"session_id"`
			Message   string `json:"message"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")

		msg := genai.NewContentFromText(body.Message, genai.RoleUser)
		cfg := agent.RunConfig{StreamingMode: agent.StreamingModeSSE}
		// req.Context() is cancelled if the client disconnects, which stops the run.
		for event, err := range r.Run(req.Context(), body.UserID, body.SessionID, msg, cfg) {
			if err != nil {
				fmt.Fprintf(w, "event: error\ndata: %q\n\n", err.Error())
				flusher.Flush()
				return
			}
			if event.Partial && event.Content != nil {
				for _, p := range event.Content.Parts {
					data, _ := json.Marshal(p.Text)
					fmt.Fprintf(w, "data: %s\n\n", data)
				}
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "event: done\ndata: {}\n\n")
		flusher.Flush()
	}
}

func main() {
	var r *runner.Runner // runner.New(runner.Config{..., AutoCreateSession: true})
	http.Handle("POST /chat", chatHandler(r))
	log.Fatal(http.ListenAndServe(":8080", nil))
}
```

If you do not need a custom protocol, the ADK REST API (next section) already offers `/run_sse`.

---

## MCP Integration

### Using MCP Tools

See [Model Context Protocol](/google-adk-guide/go/google_adk_go_comprehensive_guide/#model-context-protocol-mcp) for a stdio and a Streamable HTTP `mcptoolset` example. The ADK repository's `examples/mcp` also shows an in-process MCP server built with `mcp.NewInMemoryTransports()`, which is handy in tests.

---

## A2A Protocol

### Expose Agent via A2A

Either run `go run . web a2a` with the `full` launcher, or mount the A2A handlers yourself as shown in [Exposing Agent via A2A](/google-adk-guide/go/google_adk_go_comprehensive_guide/#exposing-agent-via-a2a).

### Consume Remote A2A Agent

See [Consuming Remote A2A Agent](/google-adk-guide/go/google_adk_go_comprehensive_guide/#consuming-remote-a2a-agent): `remoteagent.NewA2A` from `agent/remoteagent/v2` returns an ordinary `agent.Agent`.

---

## Production Patterns

### Web Server with Agent

```go
// Web server: the ADK REST API (including /run and /run_sse) mounted under
// /api on your own mux, next to a health check.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/server/adkrest"
	"google.golang.org/adk/session"
)

func main() {
	ctx := context.Background()

	m, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: m, Instruction: "Be helpful and brief."})
	if err != nil {
		log.Fatal(err)
	}

	api, err := adkrest.NewServer(adkrest.ServerConfig{
		AgentLoader:     agent.NewSingleLoader(a),
		SessionService:  session.InMemoryService(), // use session/database in production
		SSEWriteTimeout: 2 * time.Minute,
	})
	if err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.Handle("/api/", http.StripPrefix("/api", api))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	srv := &http.Server{Addr: ":" + port, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	log.Printf("listening on :%s", port)
	log.Fatal(srv.ListenAndServe())
}
```

The REST API exposes session CRUD plus `POST /run` and `POST /run_sse` (the same API the ADK dev UI uses), so this one handler covers most front ends.

### Background Worker

```go
// Background worker: a fixed pool of goroutines drains a job queue, one
// session per job.
package main

import (
	"context"
	"log"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
)

type Job struct {
	ID     string
	Prompt string
}

type Result struct {
	JobID string
	Text  string
	Err   error
}

func runJob(ctx context.Context, r *runner.Runner, job Job) Result {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	var out strings.Builder
	msg := genai.NewContentFromText(job.Prompt, genai.RoleUser)
	for event, err := range r.Run(ctx, "worker", "job-"+job.ID, msg, agent.RunConfig{}) {
		if err != nil {
			return Result{JobID: job.ID, Err: err}
		}
		if event.IsFinalResponse() && event.Content != nil {
			for _, p := range event.Content.Parts {
				out.WriteString(p.Text)
			}
		}
	}
	return Result{JobID: job.ID, Text: out.String()}
}

// process runs jobs on n workers until jobs is closed or ctx is cancelled.
func process(ctx context.Context, r *runner.Runner, jobs <-chan Job, n int) <-chan Result {
	results := make(chan Result)
	var wg sync.WaitGroup
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				select {
				case <-ctx.Done():
					return
				case results <- runJob(ctx, r, job):
				}
			}
		}()
	}
	go func() { wg.Wait(); close(results) }()
	return results
}

func main() {
	var r *runner.Runner // runner.New(runner.Config{..., AutoCreateSession: true})
	jobs := make(chan Job)
	go func() {
		defer close(jobs)
		jobs <- Job{ID: "1", Prompt: "Summarise ticket 1"}
	}()
	for res := range process(context.Background(), r, jobs, 4) {
		if res.Err != nil {
			log.Printf("job %s failed: %v", res.JobID, res.Err)
			continue
		}
		log.Printf("job %s: %s", res.JobID, res.Text)
	}
}
```

---

## Cloud Integration

### Cloud Storage Agent

```go
// Cloud Storage artifacts: a tool saves a generated report as a versioned
// artifact in GCS, and load_artifacts lets the model read it back later.
package main

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact/gcsartifact"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/loadartifactstool"
)

type saveArgs struct {
	Filename string `json:"filename" jsonschema:"for example report.md"`
	Content  string `json:"content"`
}

type saveResult struct {
	Filename string `json:"filename"`
	Version  int64  `json:"version"`
}

func saveReport(ctx agent.ToolContext, a saveArgs) (saveResult, error) {
	resp, err := ctx.Artifacts().Save(ctx, a.Filename, genai.NewPartFromBytes([]byte(a.Content), "text/markdown"))
	if err != nil {
		return saveResult{}, fmt.Errorf("save artifact: %w", err)
	}
	return saveResult{Filename: a.Filename, Version: resp.Version}, nil
}

func newRunner(ctx context.Context, m model.LLM, bucket string) (*runner.Runner, error) {
	artifacts, err := gcsartifact.NewService(ctx, bucket) // uses Application Default Credentials
	if err != nil {
		return nil, err
	}

	save, err := functiontool.New(functiontool.Config{
		Name:        "save_report",
		Description: "Saves a report so the user can download it later.",
	}, saveReport)
	if err != nil {
		return nil, err
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "report_writer",
		Model:       m,
		Instruction: "Write reports on request and save them with save_report.",
		Tools:       []tool.Tool{save, loadartifactstool.New()},
	})
	if err != nil {
		return nil, err
	}

	return runner.New(runner.Config{
		AppName:         "reports",
		Agent:           a,
		SessionService:  session.InMemoryService(),
		ArtifactService: artifacts,
	})
}

func main() {
	var m model.LLM
	if _, err := newRunner(context.Background(), m, "my-artifacts-bucket"); err != nil {
		log.Fatal(err)
	}
}
```

### BigQuery and other Google APIs

ADK for Go does not ship BigQuery or Google Workspace toolsets (the Python kit does). Wrap the official Go client (`cloud.google.com/go/bigquery` and friends) in a `functiontool` as in the [Database Tool](#database-tool) recipe, exposing specific queries rather than arbitrary SQL.

---

## Revision History

| Date | Version | Changes |
|---|---|---|
| 2026-10-21 | current | Rewritten against the current release. Every recipe is now a program compiled with `go vet`; the earlier recipes used constructors and packages that do not exist in ADK for Go (`gemini.New`, `session.Store`, `tool.Schema`, a Firestore session service). |
| April 2026 | 1.0.0 | Initial Go recipes. |

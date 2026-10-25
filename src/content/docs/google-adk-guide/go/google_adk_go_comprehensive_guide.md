---
title: "Google ADK for Go - Comprehensive Technical Guide"
description: "Building agents with ADK for Go: LLM agents, workflow agents, function tools, MCP, A2A, sessions, plugins, telemetry and testing."
framework: "google-adk"
language: "go"
sidebar:
  label: "Comprehensive guide"
  order: 10
---

**Verified against:** `google.golang.org/adk` v%%version=google.golang.org/adk%% (checked %%versions-checked%%). Every complete program on this page was compiled with `go vet` against that release, and the testing example was run with `go test`. ADK for Go reached v1.0.0 in March 2026 and has kept its public API source-compatible since; see [What's new since v1.0.0](#whats-new-since-v100).

ADK for Go versions independently of the Python and TypeScript kits, so feature names match but release numbers do not.

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Core Architecture](#core-architecture)
3. [Package Overview](#package-overview)
4. [Types and Interfaces](#types-and-interfaces)
5. [LLMAgent](#llmagent)
6. [Agent Orchestration](#agent-orchestration)
7. [Tools System](#tools-system)
8. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
9. [Agent2Agent (A2A) Protocol](#agent2agent-a2a-protocol)
10. [Go Idioms and Patterns](#go-idioms-and-patterns)
11. [Session Management](#session-management)
12. [Memory and Artifacts](#memory-and-artifacts)
13. [Testing](#testing)
14. [Best Practices](#best-practices)
15. [What's new since v1.0.0](#whats-new-since-v100)

For deployment, telemetry and serving, continue in the [production guide](/google-adk-guide/go/google_adk_go_production_guide/); for task-shaped examples see the [recipes](/google-adk-guide/go/google_adk_go_recipes/).

---

## Installation and Setup

### Prerequisites

ADK for Go needs **Go 1.25 or later** (its `go.mod` declares `go 1.25.0`).

```bash
go version
```

### Install ADK Package

```bash
mkdir my-agent && cd my-agent
go mod init example.com/my-agent
go get google.golang.org/adk@v%%version=google.golang.org/adk%%
```

The Gemini client comes from `google.golang.org/genai`, which `go get` pulls in as a dependency.

### Environment Setup

For the Gemini API, set an API key from Google AI Studio and pass it in `genai.ClientConfig`:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

For Vertex AI, authenticate with Application Default Credentials (`gcloud auth application-default login`, or a service account on Cloud Run/GKE) and select the Vertex backend:

```go
model, err := gemini.NewModel(ctx, "%%model=gemini%%", &genai.ClientConfig{
	Backend:  genai.BackendVertexAI,
	Project:  "my-gcp-project",
	Location: "us-central1",
})
```

The `genai` client also reads `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` when the corresponding fields are empty.

### Verify Installation

A complete program: one agent, one turn, printed to the terminal.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "%%model=gemini%%", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("create model: %v", err)
	}

	assistant, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       model,
		Description: "A helpful assistant.",
		Instruction: "You are a helpful assistant. Answer concisely.",
	})
	if err != nil {
		log.Fatalf("create agent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "quickstart",
		Agent:             assistant,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true, // create "s1" on first use
	})
	if err != nil {
		log.Fatalf("create runner: %v", err)
	}

	msg := genai.NewContentFromText("What is 15 + 27?", genai.RoleUser)
	for event, err := range r.Run(ctx, "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run: %v", err)
		}
		if event.IsFinalResponse() && event.Content != nil {
			for _, p := range event.Content.Parts {
				fmt.Print(p.Text)
			}
			fmt.Println()
		}
	}
}
```

```bash
go run .
```

Note the two habits ADK for Go expects everywhere: constructors take a `Config` **value** and return `(value, error)`, and agents never run themselves — a `runner.Runner` drives them and yields events as an `iter.Seq2[*session.Event, error]`.

---

## Core Architecture

### Design Philosophy

- **Code first.** Agents, tools and orchestration are ordinary Go values built from `Config` structs.
- **Composable agents.** Every agent implements `agent.Agent`; LLM agents, workflow agents, remote A2A agents and custom agents nest freely as sub-agents.
- **Events as the unit of work.** Each model response, tool call, tool result and state change is a `session.Event`. Runners stream them; session services persist them.
- **Pluggable services.** Sessions, artifacts and memory are interfaces with in-memory, database and Vertex AI implementations.

### Architecture Layers

```
┌───────────────────────────────────────────────────────────────┐
│ Serving: cmd/launcher (console, web), server/adkrest (REST),   │
│          server/adka2a/v2 (A2A), server/agentengine           │
├───────────────────────────────────────────────────────────────┤
│ runner.Runner — invocation loop, plugins, session bookkeeping  │
├───────────────────────────────────────────────────────────────┤
│ Agents: llmagent · workflowagents (sequential/parallel/loop) · │
│         remoteagent/v2 · agent.New (custom)                    │
├──────────────────────┬────────────────────────────────────────┤
│ model.LLM            │ tool.Tool / tool.Toolset               │
│ (gemini, apigee)     │ (functiontool, mcptoolset, geminitool, │
│                      │  agenttool, skilltoolset, ...)         │
├──────────────────────┴────────────────────────────────────────┤
│ Services: session · artifact · memory  (in-memory, database,   │
│           GCS, Vertex AI)                                     │
└───────────────────────────────────────────────────────────────┘
```

### Request Flow

1. Your code calls `runner.Run(ctx, userID, sessionID, msg, runConfig)`.
2. The runner loads (or, with `AutoCreateSession`, creates) the session, appends the user message, and picks the agent that should answer — the root, or the sub-agent the conversation was last transferred to.
3. An LLM agent builds an `LLMRequest` (instruction with `{state}` placeholders filled, history, tool declarations), calls the model, runs any requested tools, and loops until it produces a final response or transfers control.
4. Each step is yielded as a `*session.Event` and appended to the session, applying its `Actions.StateDelta`.
5. You range over the iterator; `event.IsFinalResponse()` marks the answer to show the user.

---

## Package Overview

All packages live under `google.golang.org/adk`.

| Package | What it provides |
|---|---|
| `agent` | `Agent`, `Config`, `New` (custom agents), contexts (`InvocationContext`, `CallbackContext`, `ToolContext`, `ReadonlyContext`), `RunConfig`, `Loader` |
| `agent/llmagent` | The LLM-backed agent and its callback types |
| `agent/workflowagents/sequentialagent`, `parallelagent`, `loopagent` | Deterministic orchestration of sub-agents |
| `agent/remoteagent/v2` | An agent that delegates to a remote A2A agent (the v1 package is deprecated) |
| `model` | `LLM` interface, `LLMRequest`, `LLMResponse` |
| `model/gemini`, `model/apigee` | Gemini (API or Vertex AI) and Apigee-proxied model implementations |
| `tool` | `Tool`, `Toolset`, predicates, `FilterToolset`, `WithConfirmation` |
| `tool/functiontool` | Typed Go functions as tools (schemas inferred from structs) |
| `tool/geminitool` | Gemini built-in tools (`GoogleSearch`, or any `genai.Tool`) |
| `tool/agenttool` | Another agent wrapped as a tool |
| `tool/mcptoolset` | Tools from an MCP server |
| `tool/exitlooptool`, `loadartifactstool`, `loadmemorytool`, `preloadmemorytool`, `skilltoolset`, `toolconfirmation` | Built-in tools and HITL support |
| `runner` | `Runner`, `Config`, `PluginConfig` |
| `session`, `session/database`, `session/vertexai` | Session service interface, GORM-backed and Vertex AI implementations |
| `artifact`, `artifact/gcsartifact` | Versioned file storage per session |
| `memory`, `memory/vertexai` | Long-term memory search |
| `plugin`, `plugin/loggingplugin`, `plugin/retryandreflect`, `plugin/functioncallmodifier` | Runner-wide callbacks |
| `telemetry` | OpenTelemetry providers, with export to Google Cloud |
| `server/adkrest`, `server/adka2a/v2`, `server/agentengine` | HTTP serving: ADK REST API, A2A, Vertex AI Agent Engine |
| `cmd/launcher`, `cmd/launcher/full` | Ready-made `console` and `web` entry points (dev UI, REST, A2A) |

---

## Types and Interfaces

### Core Agent Interface

```go
type Agent interface {
	Name() string
	Description() string
	Run(InvocationContext) iter.Seq2[*session.Event, error]
	SubAgents() []Agent
	FindAgent(name string) Agent
	FindSubAgent(name string) Agent
	// Has unexported methods.
}
```

The unexported methods mean you cannot implement `Agent` directly yet; build agents with the constructors (`llmagent.New`, the workflow agents, `remoteagent.NewA2A`) or, for custom logic, `agent.New` with a `Run` function (see [Custom agents](#custom-agents)).

### Tool Interface

```go
type Tool interface {
	Name() string
	Description() string
	IsLongRunning() bool
}

type Toolset interface {
	Name() string
	Tools(ctx agent.ReadonlyContext) ([]Tool, error)
}
```

Tools are normally created with `functiontool.New`, `geminitool`, `agenttool.New` or `mcptoolset.New` rather than implemented by hand. A `Toolset` is the extension point for dynamic tool lists.

### Model Interface

```go
type LLM interface {
	Name() string
	GenerateContent(ctx context.Context, req *LLMRequest, stream bool) iter.Seq2[*LLMResponse, error]
}
```

`gemini.NewModel(ctx, modelName, *genai.ClientConfig)` returns one. Implementing `LLM` yourself is how you plug in another provider — or a fake for tests (see [Testing](#testing)).

### Session Interface

```go
type Session interface {
	ID() string
	AppName() string
	UserID() string
	State() State
	Events() Events
	LastUpdateTime() time.Time
}

type State interface {
	Get(string) (any, error)
	Set(string, any) error
	All() iter.Seq2[string, any]
}
```

A `session.Event` embeds `model.LLMResponse` (so `event.Content`, `event.ErrorMessage`, `event.Partial` are direct fields) and adds `Author`, `InvocationID`, `Branch`, `LongRunningToolIDs` and `Actions` (`StateDelta`, `ArtifactDelta`, `TransferToAgent`, `Escalate`, `SkipSummarization`, `RequestedToolConfirmations`).

---

## LLMAgent

### Basic Configuration

The minimum is a name and a model; in practice you add an instruction and tools:

```go
assistant, err := llmagent.New(llmagent.Config{
	Name:        "assistant",
	Model:       model,
	Instruction: "You are a helpful assistant.",
})
```

### Full Configuration Options

| Field | Purpose |
|---|---|
| `Name`, `Description` | Identity; the description is what a parent agent reads when deciding whether to transfer |
| `Model` | A `model.LLM` |
| `Instruction` | System instruction. `{key}` is filled from session state, `{key?}` is optional, `{artifact.name}` inserts an artifact's text |
| `InstructionProvider` | Builds the instruction per call; no placeholder filling (use `instructionutil.InjectSessionState` if you need it) |
| `GlobalInstruction`, `GlobalInstructionProvider` | Instruction applied to every agent in the tree |
| `GenerateContentConfig` | `*genai.GenerateContentConfig`: temperature, max tokens, safety settings, thinking config |
| `Tools`, `Toolsets` | Tools offered to the model |
| `SubAgents` | Agents the model may transfer the conversation to |
| `DisallowTransferToParent`, `DisallowTransferToPeers` | Restrict transfers |
| `IncludeContents` | `IncludeContentsNone` sends no history (stateless helper agents) |
| `InputSchema`, `OutputSchema` | `*genai.Schema` for structured input/output |
| `OutputKey` | Save the final text (or JSON) to this state key |
| `BeforeAgentCallbacks`, `AfterAgentCallbacks` | Run around the whole agent turn |
| `BeforeModelCallbacks`, `AfterModelCallbacks`, `OnModelErrorCallbacks` | Inspect, replace or recover model calls |
| `BeforeToolCallbacks`, `AfterToolCallbacks`, `OnToolErrorCallbacks` | Inspect, replace or recover tool calls |

### Advanced LLMAgent Example

Callbacks, state templating, generation settings and `OutputKey` together:

```go
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
)

// logRequest runs before every model call. Returning a non-nil response
// skips the model and uses that response instead.
func logRequest(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
	log.Printf("[%s] calling model with %d content(s)", ctx.AgentName(), len(req.Contents))
	return nil, nil
}

// redactResult runs after every tool call and may replace the result.
func redactResult(ctx agent.ToolContext, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	if err != nil {
		return nil, err
	}
	delete(result, "internal_id")
	return result, nil
}

func newSupportAgent(ctx context.Context) (agent.Agent, error) {
	m, err := gemini.NewModel(ctx, "%%model=gemini%%", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Description: "Answers customer questions about orders.",
		Model:       m,
		// {customer_name} is filled from session state at call time.
		Instruction: "You help {customer_name} with questions about their orders. Be brief.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			Temperature:     genai.Ptr[float32](0.2),
			MaxOutputTokens: 1024,
		},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{logRequest},
		AfterToolCallbacks:   []llmagent.AfterToolCallback{redactResult},
		OutputKey:            "last_answer", // final text is also saved to state
	})
}

func main() {
	if _, err := newSupportAgent(context.Background()); err != nil {
		log.Fatal(err)
	}
}
```

A before-callback that returns a non-nil response (or content, for agent callbacks) short-circuits the step: the model or agent is skipped and your value is used. That is the hook for caching, guardrails and canned answers.

### Structured Output

```go
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
)

func newClassifier(ctx context.Context) (agent.Agent, error) {
	m, err := gemini.NewModel(ctx, "%%model=gemini%%", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		return nil, err
	}

	ticketSchema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"category": {Type: genai.TypeString, Enum: []string{"billing", "technical", "other"}},
			"priority": {Type: genai.TypeInteger, Description: "1 (low) to 5 (urgent)"},
			"summary":  {Type: genai.TypeString},
		},
		Required: []string{"category", "priority", "summary"},
	}

	return llmagent.New(llmagent.Config{
		Name:         "ticket_classifier",
		Model:        m,
		Instruction:  "Classify the support ticket in the user's message.",
		OutputSchema: ticketSchema, // the reply is JSON matching this schema
		OutputKey:    "ticket",     // and is stored in session state under "ticket"
	})
}

func main() {
	if _, err := newClassifier(context.Background()); err != nil {
		log.Fatal(err)
	}
}
```

With `OutputSchema` set the model replies with JSON only; downstream agents can read it from state (`{ticket}`) and Go code can `json.Unmarshal` the final event's text.

---

## Agent Orchestration

Workflow agents run sub-agents in a fixed pattern without asking a model what to do next. They share session state, so the usual hand-off is `OutputKey` on one agent and a `{placeholder}` in the next agent's instruction.

### SequentialAgent

```go
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
)

func newModel(ctx context.Context) (model.LLM, error) {
	return gemini.NewModel(ctx, "%%model=gemini%%", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
}

func newArticlePipeline(ctx context.Context) (agent.Agent, error) {
	m, err := newModel(ctx)
	if err != nil {
		return nil, err
	}

	researcher, err := llmagent.New(llmagent.Config{
		Name:        "researcher",
		Model:       m,
		Instruction: "List the five most important facts about the user's topic.",
		OutputKey:   "facts",
	})
	if err != nil {
		return nil, err
	}

	writer, err := llmagent.New(llmagent.Config{
		Name:        "writer",
		Model:       m,
		Instruction: "Write a 150-word article using only these facts:\n{facts}",
		OutputKey:   "draft",
	})
	if err != nil {
		return nil, err
	}

	editor, err := llmagent.New(llmagent.Config{
		Name:        "editor",
		Model:       m,
		Instruction: "Tighten this draft and fix any errors. Return only the final text:\n{draft}",
	})
	if err != nil {
		return nil, err
	}

	return sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:        "article_pipeline",
			Description: "Researches, drafts and edits a short article.",
			SubAgents:   []agent.Agent{researcher, writer, editor},
		},
	})
}

func main() {
	if _, err := newArticlePipeline(context.Background()); err != nil {
		log.Fatal(err)
	}
}
```

### ParallelAgent

Parallel sub-agents run concurrently on separate branches of the same session. Give each its own `OutputKey`, then merge in a following step:

```go
package main

import (
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model"
)

// newReviewPipeline fans out to three reviewers, then merges their notes.
func newReviewPipeline(m model.LLM) (agent.Agent, error) {
	reviewer := func(name, focus, key string) (agent.Agent, error) {
		return llmagent.New(llmagent.Config{
			Name:        name,
			Model:       m,
			Instruction: "Review the user's code for " + focus + " only. List concrete issues.",
			OutputKey:   key,
		})
	}

	security, err := reviewer("security_reviewer", "security problems", "security_notes")
	if err != nil {
		return nil, err
	}
	performance, err := reviewer("performance_reviewer", "performance problems", "performance_notes")
	if err != nil {
		return nil, err
	}
	style, err := reviewer("style_reviewer", "readability and style", "style_notes")
	if err != nil {
		return nil, err
	}

	fanOut, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "reviews",
			SubAgents: []agent.Agent{security, performance, style},
		},
	})
	if err != nil {
		return nil, err
	}

	merger, err := llmagent.New(llmagent.Config{
		Name:  "merger",
		Model: m,
		Instruction: "Combine these reviews into one prioritised list.\n" +
			"Security: {security_notes}\nPerformance: {performance_notes}\nStyle: {style_notes}",
	})
	if err != nil {
		return nil, err
	}

	return sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "code_review",
			SubAgents: []agent.Agent{fanOut, merger},
		},
	})
}

func main() {
	var m model.LLM // e.g. from gemini.NewModel
	if _, err := newReviewPipeline(m); err != nil {
		log.Fatal(err)
	}
}
```

### LoopAgent

A loop runs its sub-agents repeatedly until one of them escalates — the `exit_loop` tool does exactly that — or `MaxIterations` is reached:

```go
package main

import (
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/exitlooptool"
)

func newRefinementLoop(m model.LLM) (agent.Agent, error) {
	exitLoop, err := exitlooptool.New()
	if err != nil {
		return nil, err
	}

	drafter, err := llmagent.New(llmagent.Config{
		Name:        "drafter",
		Model:       m,
		Instruction: "Improve the slogan in {slogan?} (or write one for the user's product). Reply with the slogan only.",
		OutputKey:   "slogan",
	})
	if err != nil {
		return nil, err
	}

	critic, err := llmagent.New(llmagent.Config{
		Name:  "critic",
		Model: m,
		Instruction: "Judge this slogan: {slogan}. If it is under eight words and memorable, " +
			"call exit_loop. Otherwise explain what to fix.",
		Tools: []tool.Tool{exitLoop},
	})
	if err != nil {
		return nil, err
	}

	return loopagent.New(loopagent.Config{
		MaxIterations: 5, // hard stop if the critic never calls exit_loop
		AgentConfig: agent.Config{
			Name:      "slogan_loop",
			SubAgents: []agent.Agent{drafter, critic},
		},
	})
}

func main() {
	var m model.LLM
	if _, err := newRefinementLoop(m); err != nil {
		log.Fatal(err)
	}
}
```

### Hierarchical Multi-Agent System

Two ways to combine LLM agents:

- **Sub-agents (transfer).** The parent's model may hand the conversation to a sub-agent, which then answers the user directly and keeps the conversation until it transfers back.
- **Agent as a tool (`agenttool`).** The parent calls the other agent like a function and gets its answer back, staying in control.

```go
package main

import (
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
)

func newHelpDesk(m model.LLM) (agent.Agent, error) {
	billing, err := llmagent.New(llmagent.Config{
		Name:        "billing",
		Description: "Handles invoices, refunds and payment questions.",
		Model:       m,
		Instruction: "You resolve billing questions.",
	})
	if err != nil {
		return nil, err
	}

	technical, err := llmagent.New(llmagent.Config{
		Name:        "technical",
		Description: "Handles errors, outages and configuration problems.",
		Model:       m,
		Instruction: "You resolve technical problems step by step.",
	})
	if err != nil {
		return nil, err
	}

	// A translator used as a tool: the coordinator calls it and keeps control.
	translator, err := llmagent.New(llmagent.Config{
		Name:        "translator",
		Description: "Translates text into English.",
		Model:       m,
		Instruction: "Translate the input into plain English. Reply with the translation only.",
	})
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:  "help_desk",
		Model: m,
		Instruction: "Route the customer to the right specialist. " +
			"If the message is not in English, translate it first.",
		// Sub-agents: the model can transfer the conversation to them.
		SubAgents: []agent.Agent{billing, technical},
		// Agent as tool: called like a function, result returned to help_desk.
		Tools: []tool.Tool{agenttool.New(translator, nil)},
	})
}

func main() {
	var m model.LLM
	if _, err := newHelpDesk(m); err != nil {
		log.Fatal(err)
	}
}
```

### Custom agents

For deterministic steps that need no model, `agent.New` takes a `Run` function that yields events:

```go
package main

import (
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// greet is a non-LLM agent: it yields one event and finishes.
func greet(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		name := "there"
		if v, err := ctx.Session().State().Get("customer_name"); err == nil {
			if s, ok := v.(string); ok {
				name = s
			}
		}
		event := session.NewEvent(ctx.InvocationID())
		event.Author = ctx.Agent().Name()
		event.LLMResponse = model.LLMResponse{
			Content: genai.NewContentFromText("Hello, "+name+"!", genai.RoleModel),
		}
		event.Actions.StateDelta = map[string]any{"greeted": true}
		yield(event, nil)
	}
}

func main() {
	greeter, err := agent.New(agent.Config{
		Name:        "greeter",
		Description: "Greets the customer by name.",
		Run:         greet,
	})
	if err != nil {
		log.Fatal(err)
	}
	_ = greeter // use as a root agent or as a sub-agent of a workflow agent
}
```

---

## Tools System

### Function Tool

`functiontool.New` turns a typed Go function into a tool. The argument and result types are reflected into JSON schemas (use `json` and `jsonschema` struct tags); returning an error reports it to the model rather than crashing the run.

```go
package main

import (
	"fmt"
	"log"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// The argument and result types define the tool's JSON schema.
// `jsonschema` tags become parameter descriptions for the model.
type weatherArgs struct {
	City  string `json:"city" jsonschema:"City name, for example Leeds"`
	Units string `json:"units,omitempty" jsonschema:"celsius or fahrenheit (default celsius)"`
}

type weatherResult struct {
	City     string  `json:"city"`
	Forecast string  `json:"forecast"`
	Temp     float64 `json:"temp"`
}

func getWeather(ctx agent.ToolContext, args weatherArgs) (weatherResult, error) {
	if strings.TrimSpace(args.City) == "" {
		return weatherResult{}, fmt.Errorf("city is required")
	}
	// Tools can read and write session state through the context.
	if err := ctx.State().Set("last_city", args.City); err != nil {
		return weatherResult{}, err
	}
	return weatherResult{City: args.City, Forecast: "light rain", Temp: 12}, nil
}

type refundArgs struct {
	OrderID string  `json:"order_id"`
	Amount  float64 `json:"amount"`
}

func issueRefund(ctx agent.ToolContext, args refundArgs) (map[string]any, error) {
	return map[string]any{"status": "refunded", "order_id": args.OrderID}, nil
}

func newTools() ([]tool.Tool, error) {
	weather, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Returns the current weather for a city.",
	}, getWeather)
	if err != nil {
		return nil, err
	}

	refund, err := functiontool.New(functiontool.Config{
		Name:        "issue_refund",
		Description: "Refunds an order.",
		// Ask a human before refunds over 100.
		RequireConfirmationProvider: func(args refundArgs) bool { return args.Amount > 100 },
	}, issueRefund)
	if err != nil {
		return nil, err
	}
	return []tool.Tool{weather, refund}, nil
}

func newAgent(m model.LLM) (agent.Agent, error) {
	tools, err := newTools()
	if err != nil {
		return nil, err
	}
	return llmagent.New(llmagent.Config{
		Name:        "concierge",
		Model:       m,
		Instruction: "Use get_weather for weather and issue_refund for refunds.",
		Tools:       tools,
	})
}

func main() {
	var m model.LLM
	if _, err := newAgent(m); err != nil {
		log.Fatal(err)
	}
}
```

`RequireConfirmation: true` (always) or `RequireConfirmationProvider` (per call, typed on your argument struct) pauses the run and emits an `adk_request_confirmation` function call; the client answers with a function response and the tool runs or is rejected. Inside a tool you can also call `ctx.RequestConfirmation(hint, payload)` and read `ctx.ToolConfirmation()` yourself. Set `IsLongRunning: true` for tools that start work and finish later.

### Human approval for tool calls

With `RequireConfirmation` or `RequireConfirmationProvider` set, ADK does not run the tool when the model first asks for it. The turn ends with two things: a function response to the model saying the tool `requires confirmation, please approve or reject`, and an `adk_request_confirmation` function call (`toolconfirmation.FunctionCallName`) whose arguments carry the original call. Your application shows that call to a person and answers it on the next turn with a `FunctionResponse` that has the same ID, the same name and `{"confirmed": true}` (or `false`). The program below runs the whole round trip; a small scripted model stands in for Gemini so it runs without credentials:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/toolconfirmation"
)

type refundArgs struct {
	OrderID string  `json:"order_id"`
	Amount  float64 `json:"amount"`
}

func issueRefund(ctx agent.ToolContext, args refundArgs) (map[string]any, error) {
	return map[string]any{"status": "refunded", "order_id": args.OrderID}, nil
}

// approve answers a pending adk_request_confirmation call.
func approve(callID string, ok bool) *genai.Content {
	return &genai.Content{
		Role: string(genai.RoleUser),
		Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID:       callID,
			Name:     toolconfirmation.FunctionCallName,
			Response: map[string]any{"confirmed": ok},
		}}},
	}
}

// turn runs one turn and returns the ID of a confirmation request, if any.
func turn(ctx context.Context, r *runner.Runner, sessionID string, msg *genai.Content) (string, error) {
	var pending string
	for event, err := range r.Run(ctx, "u1", sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return "", err
		}
		if event.Content == nil {
			continue
		}
		for _, p := range event.Content.Parts {
			if fc := p.FunctionCall; fc != nil && fc.Name == toolconfirmation.FunctionCallName {
				original, err := toolconfirmation.OriginalCallFrom(fc)
				if err != nil {
					return "", err
				}
				fmt.Printf("approval needed: %s(%v)\n", original.Name, original.Args)
				pending = fc.ID
			}
			if fr := p.FunctionResponse; fr != nil && fr.Name == "issue_refund" {
				fmt.Println("tool result:", fr.Response)
			}
		}
	}
	return pending, nil
}

func run(ctx context.Context, m model.LLM) error {
	refund, err := functiontool.New(functiontool.Config{
		Name:                        "issue_refund",
		Description:                 "Refunds an order.",
		RequireConfirmationProvider: func(a refundArgs) bool { return a.Amount > 100 },
	}, issueRefund)
	if err != nil {
		return err
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "support",
		Model:       m,
		Instruction: "Issue refunds when asked.",
		Tools:       []tool.Tool{refund},
	})
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{
		AppName:           "shop",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return err
	}

	// Turn 1: the model calls issue_refund; ADK pauses and asks for approval.
	callID, err := turn(ctx, r, "s1", genai.NewContentFromText("Refund order A1, 250 pounds.", genai.RoleUser))
	if err != nil || callID == "" {
		return err
	}
	// Turn 2: a person approves (false would reject the call).
	_, err = turn(ctx, r, "s1", approve(callID, true))
	return err
}

func main() {
	if err := run(context.Background(), &scripted{}); err != nil {
		log.Fatal(err)
	}
}

// scripted stands in for Gemini: it asks for a refund, then answers.
type scripted struct{ n int }

func (s *scripted) Name() string { return "scripted" }

func (s *scripted) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	s.n++
	var c *genai.Content
	if s.n == 1 {
		c = &genai.Content{Role: string(genai.RoleModel), Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "call-1", Name: "issue_refund", Args: map[string]any{"order_id": "A1", "amount": 250},
		}}}}
	} else {
		c = genai.NewContentFromText("Done: order A1 has been refunded.", genai.RoleModel)
	}
	return func(yield func(*model.LLMResponse, error) bool) { yield(&model.LLMResponse{Content: c}, nil) }
}
```

- `toolconfirmation.OriginalCallFrom(fc)` extracts the tool call being approved, so you can show its name and arguments.
- A `"payload"` field in the response reaches the tool through `ctx.ToolConfirmation().Payload`, which lets the approver adjust the call (an approved amount, say) rather than only accept or reject it.
- The session must survive between the two turns, so use a persistent session service when approval can take longer than a process lifetime.
- `mcptoolset.Config` takes the same two fields, and `tool.WithConfirmation` wraps any toolset.

### Custom Tool Implementation

You rarely implement `tool.Tool` directly. To control *which* tools are offered on each turn, implement `tool.Toolset`:

```go
package main

import (
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type sumArgs struct {
	Numbers []float64 `json:"numbers"`
}

func sum(_ agent.ToolContext, a sumArgs) (float64, error) {
	total := 0.0
	for _, n := range a.Numbers {
		total += n
	}
	return total, nil
}

// adminToolset offers its tools only to users whose session state marks them as admins.
type adminToolset struct {
	tools []tool.Tool
}

func (t adminToolset) Name() string { return "admin_tools" }

func (t adminToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	if v, err := ctx.ReadonlyState().Get("user:is_admin"); err == nil && v == true {
		return t.tools, nil
	}
	return nil, nil
}

func newAgent(m model.LLM) (agent.Agent, error) {
	sumTool, err := functiontool.New(functiontool.Config{Name: "sum", Description: "Adds numbers."}, sum)
	if err != nil {
		return nil, err
	}

	// Built-in wrappers: filter a toolset by name, or require confirmation for all of it.
	admin := tool.FilterToolset(adminToolset{tools: []tool.Tool{sumTool}}, tool.StringPredicate([]string{"sum"}))

	return llmagent.New(llmagent.Config{
		Name:     "ops_agent",
		Model:    m,
		Toolsets: []tool.Toolset{admin},
	})
}

func main() {
	var m model.LLM
	if _, err := newAgent(m); err != nil {
		log.Fatal(err)
	}
}
```

`tool.WithConfirmation(ts, true, nil)` wraps any toolset so every tool in it needs approval (marked experimental; model-side built-ins such as `geminitool` cannot be confirmed and pass through unchanged).

### Gemini Native Tools

Gemini's server-side tools run inside the model call:

```go
package main

import (
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
)

func newResearchAgent(m model.LLM) (agent.Agent, error) {
	codeExecution := geminitool.New("code_execution", "Runs Python in Gemini's sandbox.",
		&genai.Tool{CodeExecution: &genai.ToolCodeExecution{}})

	return llmagent.New(llmagent.Config{
		Name:        "researcher",
		Model:       m,
		Instruction: "Search the web for current facts and run code for calculations.",
		Tools:       []tool.Tool{geminitool.GoogleSearch{}, codeExecution},
	})
}

func main() {
	var m model.LLM
	if _, err := newResearchAgent(m); err != nil {
		log.Fatal(err)
	}
}
```

Gemini does not allow every built-in tool to be combined with function tools in a single request. If you hit that limit, put the built-in tool on its own agent and expose that agent with `agenttool.New`.

---

## Model Context Protocol (MCP)

### Using MCP Tools

`mcptoolset.New` connects to an MCP server through any transport from the official Go SDK (`github.com/modelcontextprotocol/go-sdk/mcp`) and exposes its tools as a `tool.Toolset`:

```go
package main

import (
	"log"
	"net/http"
	"os/exec"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
)

func newFileAgent(m model.LLM) (agent.Agent, error) {
	// Local MCP server over stdio.
	files, err := mcptoolset.New(mcptoolset.Config{
		Transport: &mcp.CommandTransport{
			Command: exec.Command("npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"),
		},
		// Offer only read-only tools to the model.
		ToolFilter: tool.StringPredicate([]string{"read_file", "list_directory"}),
	})
	if err != nil {
		return nil, err
	}

	// Remote MCP server over Streamable HTTP.
	remote, err := mcptoolset.New(mcptoolset.Config{
		Transport: &mcp.StreamableClientTransport{
			Endpoint:   "https://mcp.example.com/mcp",
			HTTPClient: http.DefaultClient, // add auth with an oauth2 client or custom RoundTripper
		},
		RequireConfirmation: true, // ask a human before any remote tool runs
	})
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:        "file_assistant",
		Model:       m,
		Instruction: "Help the user inspect files in the workspace.",
		Toolsets:    []tool.Toolset{files, remote},
	})
}

func main() {
	var m model.LLM
	if _, err := newFileAgent(m); err != nil {
		log.Fatal(err)
	}
}
```

You can also pass a pre-configured `*mcp.Client` in `Config.Client`. For tests, `mcp.NewInMemoryTransports()` gives a client/server pair in one process.

### Exposing ADK Tools via MCP

ADK for Go has no MCP *server* helper. To publish tools to MCP hosts, write an MCP server with the Go SDK (`mcp.NewServer`, `mcp.AddTool`) and call your Go functions from the handlers; to expose a whole agent to other agents, use A2A (next section).

---

## Agent2Agent (A2A) Protocol

The A2A packages moved to `/v2` (built on `github.com/a2aproject/a2a-go/v2`); the original `server/adka2a` and `agent/remoteagent` packages still build but are deprecated.

### Exposing Agent via A2A

```go
package main

import (
	"log"
	"net/http"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adka2a/v2"
	"google.golang.org/adk/session"
)

// serveA2A exposes an ADK agent at baseURL+"/invoke" with an agent card at
// the well-known path.
func serveA2A(a agent.Agent, baseURL, addr string) error {
	card := &a2a.AgentCard{
		Name:        a.Name(),
		Description: a.Description(),
		SupportedInterfaces: []*a2a.AgentInterface{{
			URL:             baseURL + "/invoke",
			ProtocolBinding: a2a.TransportProtocolJSONRPC,
			ProtocolVersion: a2a.Version,
		}},
		Version:            "1.0.0",
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		Skills:             adka2a.BuildAgentSkills(a),
		Capabilities:       a2a.AgentCapabilities{Streaming: true},
	}

	executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        a.Name(),
			Agent:          a,
			SessionService: session.InMemoryService(),
		},
	})

	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))
	return http.ListenAndServe(addr, mux)
}

func main() {
	var weatherAgent agent.Agent // build with llmagent.New
	log.Fatal(serveA2A(weatherAgent, "http://localhost:8001", ":8001"))
}
```

The `full` launcher can do this for you: `go run . web a2a` serves the root agent at `/a2a/v1/invoke`.

### Consuming Remote A2A Agent

A remote agent is an ordinary `agent.Agent`, so it can be a sub-agent, a workflow step or wrapped with `agenttool`:

```go
package main

import (
	"log"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/remoteagent/v2"
	"google.golang.org/adk/model"
)

func newTravelPlanner(m model.LLM) (agent.Agent, error) {
	// The agent card is fetched from http://localhost:8001/.well-known/agent-card.json.
	weather, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:              "weather",
		Description:       "Remote agent that answers weather questions.",
		AgentCardProvider: remoteagent.NewAgentCardProvider("http://localhost:8001"),
	})
	if err != nil {
		return nil, err
	}

	return llmagent.New(llmagent.Config{
		Name:        "travel_planner",
		Model:       m,
		Instruction: "Plan trips. Transfer weather questions to the weather agent.",
		SubAgents:   []agent.Agent{weather},
	})
}

func main() {
	var m model.LLM
	if _, err := newTravelPlanner(m); err != nil {
		log.Fatal(err)
	}
}
```

### Multi-Agent A2A System

Run each specialist as its own service (its own deployment, scaling and credentials) and compose them in a coordinator with `remoteagent.NewA2A`. The agent card is resolved when the remote agent is invoked rather than at construction, so the coordinator starts even if a peer is briefly unavailable; use `BeforeRequestCallbacks` to add authentication headers or trace IDs to outgoing requests.

---

## Go Idioms and Patterns

### Idiomatic Error Handling

- Constructors return `(value, error)`: check every one at start-up and fail fast.
- `runner.Run` yields `(event, err)` pairs; an error ends that invocation. Wrap it with the user and session IDs for logs.
- Tool functions return `error` for expected failures (bad input, upstream 404). The model sees the message and can recover; reserve panics for programmer errors.
- `session.ErrNotFound` is returned by session services for unknown sessions — check it with `errors.Is`.

### Context Handling

Every context in ADK embeds `context.Context`. Cancelling the context passed to `runner.Run` stops the invocation, including in-flight model calls; tools receive the same context through `agent.ToolContext`, so pass it to HTTP clients and database calls.

### Goroutine Patterns

A `Runner` can be shared across goroutines; run concurrent turns on *different* sessions:

```go
package main

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/runner"
)

// ask runs one turn and returns the final text. The context bounds the whole
// invocation: cancelling it stops the model call and any running tools.
func ask(ctx context.Context, r *runner.Runner, userID, sessionID, text string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	var out strings.Builder
	msg := genai.NewContentFromText(text, genai.RoleUser)
	for event, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return "", fmt.Errorf("run %s/%s: %w", userID, sessionID, err)
		}
		if event.IsFinalResponse() && event.Content != nil {
			for _, p := range event.Content.Parts {
				out.WriteString(p.Text)
			}
		}
	}
	return out.String(), nil
}

// askAll fans questions out across goroutines, one session per question.
// A Runner is safe to share; keep concurrent turns on *different* sessions.
func askAll(ctx context.Context, r *runner.Runner, questions []string) ([]string, error) {
	answers := make([]string, len(questions))
	errs := make([]error, len(questions))
	sem := make(chan struct{}, 4) // at most four model calls in flight

	var wg sync.WaitGroup
	for i, q := range questions {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			answers[i], errs[i] = ask(ctx, r, "batch", fmt.Sprintf("q-%d", i), q)
		}()
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return answers, err
		}
	}
	return answers, nil
}

func main() {
	var r *runner.Runner // runner.New(runner.Config{..., AutoCreateSession: true})
	_, _ = askAll(context.Background(), r, []string{"a", "b"})
}
```

### Interface-Based Design

Accept `model.LLM`, `session.Service` and `agent.Agent` as parameters instead of constructing them inside your functions. Production code passes `gemini.NewModel(...)` and `database.NewSessionService(...)`; tests pass fakes. Every snippet on this page that takes `m model.LLM` follows that pattern.

---

## Session Management

### Basic Session Usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/adk/session"
)

func main() {
	ctx := context.Background()
	svc := session.InMemoryService()

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "shop",
		UserID:  "u42",
		State: map[string]any{
			"customer_name":  "Priya",          // session scope
			"user:tier":      "gold",           // shared by all of u42's sessions
			"app:promo_code": "AUTUMN",         // shared by every user of "shop"
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	id := created.Session.ID()

	// ... run turns with runner.Run(ctx, "u42", id, msg, agent.RunConfig{}) ...

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "shop", UserID: "u42", SessionID: id})
	if err != nil {
		log.Fatal(err)
	}
	for key, value := range got.Session.State().All() {
		fmt.Printf("%s = %v\n", key, value)
	}
	fmt.Println("events so far:", got.Session.Events().Len())
}
```

State keys are scoped by prefix: no prefix is per session, `user:` is shared by all of a user's sessions, `app:` by every user of the app, and `temp:` lasts only for the current invocation.

### Persistent Sessions

`session/database` stores sessions through GORM, so any GORM dialector works (PostgreSQL, SQLite, Spanner, MySQL):

```go
package main

import (
	"log"

	"github.com/glebarez/sqlite" // pure-Go SQLite driver for GORM
	"gorm.io/driver/postgres"

	"google.golang.org/adk/session"
	"google.golang.org/adk/session/database"
)

func newSessionService(postgresDSN string) (session.Service, error) {
	var svc session.Service
	var err error
	if postgresDSN != "" {
		svc, err = database.NewSessionService(postgres.Open(postgresDSN))
	} else {
		svc, err = database.NewSessionService(sqlite.Open("sessions.db"))
	}
	if err != nil {
		return nil, err
	}
	// The service never creates or alters tables itself: migrate on every start.
	if err := database.AutoMigrate(svc); err != nil {
		return nil, err
	}
	return svc, nil
}

func main() {
	if _, err := newSessionService(""); err != nil {
		log.Fatal(err)
	}
}
```

For managed storage on Google Cloud, `session/vertexai` uses Vertex AI Agent Engine sessions.

### Session State Management

Write state through the context you are given rather than through the service:

- In tools: `ctx.State().Set(key, value)` on `agent.ToolContext`.
- In callbacks: `ctx.State()` on `agent.CallbackContext`.
- In custom agents: put changes in `event.Actions.StateDelta` on the events you yield.
- From the caller: `runner.Run(..., runner.WithStateDelta(map[string]any{...}))` applies a delta with the user message.

The runner applies these deltas when it appends the event, so state and history stay consistent.

---

## Memory and Artifacts

**Artifacts** are versioned files stored per session: reports a tool generates, uploads, images. Give the runner an `ArtifactService` and tools reach it through `ctx.Artifacts()` (`Save`, `Load`, `LoadVersion`, `List`). Each save of a name creates a new version, and a name starting with `user:` is shared across all of the user's sessions. `loadartifactstool.New()` lets the model list and open artifacts itself.

**Memory** is searchable knowledge across a user's sessions. Add a finished session to a `memory.Service` with `AddSessionToMemory`; agents search it with `loadmemorytool.New()` (the model decides when) or `preloadmemorytool.New()` (searched on every turn and added to the request), and tools can call `ctx.SearchMemory(ctx, query)`.

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/loadartifactstool"
	"google.golang.org/adk/tool/loadmemorytool"
)

type reportArgs struct {
	Title string `json:"title"`
	Body  string `json:"body"`
}

// saveReport stores a Markdown file as a session artifact. Each save of the
// same name creates a new version.
func saveReport(ctx agent.ToolContext, a reportArgs) (map[string]any, error) {
	part := genai.NewPartFromBytes([]byte("# "+a.Title+"\n\n"+a.Body), "text/markdown")
	resp, err := ctx.Artifacts().Save(ctx, "report.md", part)
	if err != nil {
		return nil, err
	}
	return map[string]any{"file": "report.md", "version": resp.Version}, nil
}

func newRunner(m model.LLM, sessions session.Service, artifacts artifact.Service, mem memory.Service) (*runner.Runner, error) {
	save, err := functiontool.New(functiontool.Config{
		Name:        "save_report",
		Description: "Saves a report for the user to download.",
	}, saveReport)
	if err != nil {
		return nil, err
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       m,
		Instruction: "Write reports with save_report. Use load_memory when the user refers to earlier conversations.",
		Tools:       []tool.Tool{save, loadartifactstool.New(), loadmemorytool.New()},
	})
	if err != nil {
		return nil, err
	}
	return runner.New(runner.Config{
		AppName:           "reports",
		Agent:             a,
		SessionService:    sessions,
		ArtifactService:   artifacts, // needed by ctx.Artifacts() and load_artifacts
		MemoryService:     mem,
		AutoCreateSession: true,
	})
}

func main() {
	ctx := context.Background()
	sessions := session.InMemoryService()
	artifacts := artifact.InMemoryService() // gcsartifact.NewService(ctx, bucket) in production
	mem := memory.InMemoryService()         // memory/vertexai for Memory Bank

	r, err := newRunner(&scripted{}, sessions, artifacts, mem)
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("Write a status report. Our launch date is 3 March.", genai.RoleUser)
	for _, err := range r.Run(ctx, "u1", "monday", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
	}

	// Read artifacts through the service, outside the agent.
	list, err := artifacts.List(ctx, &artifact.ListRequest{AppName: "reports", UserID: "u1", SessionID: "monday"})
	if err != nil {
		log.Fatal(err)
	}
	for _, name := range list.FileNames {
		got, err := artifacts.Load(ctx, &artifact.LoadRequest{AppName: "reports", UserID: "u1", SessionID: "monday", FileName: name})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s: %q\n", name, got.Part.InlineData.Data)
	}

	// When a conversation ends, add it to memory so later sessions can search it.
	done, err := sessions.Get(ctx, &session.GetRequest{AppName: "reports", UserID: "u1", SessionID: "monday"})
	if err != nil {
		log.Fatal(err)
	}
	if err := mem.AddSessionToMemory(ctx, done.Session); err != nil {
		log.Fatal(err)
	}
	found, err := mem.SearchMemory(ctx, &memory.SearchRequest{AppName: "reports", UserID: "u1", Query: "launch date"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(found.Memories), "memories match")
}

// scripted stands in for Gemini: it calls save_report once, then answers.
type scripted struct{ n int }

func (s *scripted) Name() string { return "scripted" }

func (s *scripted) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	s.n++
	c := genai.NewContentFromText("Saved report.md. Launch is on 3 March.", genai.RoleModel)
	if s.n == 1 {
		c = &genai.Content{Role: string(genai.RoleModel), Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "c1", Name: "save_report", Args: map[string]any{"title": "Status", "body": "On track."},
		}}}}
	}
	return func(yield func(*model.LLMResponse, error) bool) { yield(&model.LLMResponse{Content: c}, nil) }
}
```

| Service | In memory | Google Cloud |
|---|---|---|
| Artifacts | `artifact.InMemoryService()` | `gcsartifact.NewService(ctx, bucket)` |
| Memory | `memory.InMemoryService()` (word matching) | `memory/vertexai` (Vertex AI Memory Bank) |

The in-memory services lose everything when the process exits; use them for tests and local runs.

---

## Testing

### Unit Testing Agents

Keep agent construction in a function that takes the model as a parameter, then drive it through a real `Runner` with an in-memory session service:

### Mock Model for Testing

```go
package support

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
)

// NewAgent takes the model as a parameter so tests can pass a fake.
func NewAgent(m model.LLM) (agent.Agent, error) {
	return llmagent.New(llmagent.Config{
		Name:        "support",
		Model:       m,
		Instruction: "Answer support questions.",
	})
}
```

```go
package support

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// fakeLLM implements model.LLM with a canned reply and records requests.
type fakeLLM struct {
	reply    string
	requests []*model.LLMRequest
}

func (f *fakeLLM) Name() string { return "fake" }

func (f *fakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	f.requests = append(f.requests, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content: genai.NewContentFromText(f.reply, genai.RoleModel),
		}, nil)
	}
}

func TestAgentReplies(t *testing.T) {
	fake := &fakeLLM{reply: "Have you tried turning it off and on again?"}
	a, err := NewAgent(fake)
	if err != nil {
		t.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "test",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	var got string
	msg := genai.NewContentFromText("My router is broken", genai.RoleUser)
	for event, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.IsFinalResponse() && event.Content != nil {
			got += event.Content.Parts[0].Text
		}
	}

	if got != fake.reply {
		t.Errorf("reply = %q, want %q", got, fake.reply)
	}
	if len(fake.requests) != 1 {
		t.Errorf("model called %d times, want 1", len(fake.requests))
	}
}
```

```bash
go test ./...
```

### Scripting tool calls

A fake that only returns text never exercises your tools. Script the model's responses instead, including function calls, and assert on what the model was sent next: that is where tool results, state templating and instructions show up.

```go
package orders

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

type orderArgs struct {
	OrderID string `json:"order_id"`
}

type orderStatus struct {
	OrderID string `json:"order_id"`
	Status  string `json:"status"`
}

func getOrder(_ agent.ToolContext, a orderArgs) (orderStatus, error) {
	return orderStatus{OrderID: a.OrderID, Status: "shipped"}, nil
}

// NewAgent takes the model as a parameter so tests can pass a fake.
func NewAgent(m model.LLM) (agent.Agent, error) {
	lookup, err := functiontool.New(functiontool.Config{
		Name:        "get_order",
		Description: "Looks up an order.",
	}, getOrder)
	if err != nil {
		return nil, err
	}
	return llmagent.New(llmagent.Config{
		Name:        "orders",
		Model:       m,
		Instruction: "Answer order questions with get_order.",
		Tools:       []tool.Tool{lookup},
	})
}
```

```go
package orders

import (
	"context"
	"fmt"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

// scriptedLLM returns its responses in order and records every request.
type scriptedLLM struct {
	responses []*genai.Content
	requests  []*model.LLMRequest
}

func (s *scriptedLLM) Name() string { return "scripted" }

func (s *scriptedLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	s.requests = append(s.requests, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(s.requests) > len(s.responses) {
			yield(nil, fmt.Errorf("unexpected model call %d", len(s.requests)))
			return
		}
		yield(&model.LLMResponse{Content: s.responses[len(s.requests)-1]}, nil)
	}
}

func TestLooksUpOrder(t *testing.T) {
	fake := &scriptedLLM{responses: []*genai.Content{
		{Role: string(genai.RoleModel), Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID: "call-1", Name: "get_order", Args: map[string]any{"order_id": "A123"},
		}}}},
		genai.NewContentFromText("Order A123 has shipped.", genai.RoleModel),
	}}
	a, err := NewAgent(fake)
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName: "test", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	var answer string
	msg := genai.NewContentFromText("Where is A123?", genai.RoleUser)
	for event, err := range r.Run(t.Context(), "u1", "s1", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		if event.IsFinalResponse() && event.Content != nil {
			answer = event.Content.Parts[0].Text
		}
	}

	if answer != "Order A123 has shipped." {
		t.Errorf("answer = %q", answer)
	}
	if len(fake.requests) != 2 {
		t.Fatalf("model called %d times, want 2", len(fake.requests))
	}
	// The second request carries the tool's result back to the model.
	last := fake.requests[1].Contents[len(fake.requests[1].Contents)-1]
	fr := last.Parts[0].FunctionResponse
	if fr == nil || fr.Name != "get_order" || fr.Response["status"] != "shipped" {
		t.Errorf("tool result not sent to the model: %+v", last.Parts[0])
	}
}
```

`session/sessiontestsuite` provides a shared conformance suite if you write your own `session.Service`.

---

## Best Practices

### 1. Context Propagation

Pass the request context into `runner.Run` and on into tools; set a timeout per invocation so a stuck model call cannot hold a goroutine forever.

### 2. Error Handling

Check every constructor error at start-up. In tools, return errors the model can act on ("order 123 not found") rather than opaque ones.

### 3. Resource Cleanup

Build models, toolsets (especially MCP connections) and runners once and reuse them. Shut down telemetry providers with `Shutdown(ctx)` and give plugins a `CloseTimeout`.

### 4. Configuration

Read model names, project IDs and endpoints from the environment or flags. Pin the ADK module version in `go.mod` and upgrade deliberately.

### 5. Concurrent Execution

One runner, many sessions. Bound concurrency with a semaphore or worker pool to stay inside model quotas, and never run two turns on the same session at once.

### Observability

```go
package main

import (
	"context"
	"log"

	"google.golang.org/adk/telemetry"
)

func main() {
	ctx := context.Background()

	// Exports traces and logs to Google Cloud (Cloud Trace / Cloud Logging).
	providers, err := telemetry.New(ctx,
		telemetry.WithOtelToCloud(true),
		telemetry.WithGcpResourceProject("my-gcp-project"),
		telemetry.WithGenAICaptureMessageContent(false), // keep prompts out of traces
	)
	if err != nil {
		log.Fatal(err)
	}
	providers.SetGlobalOtelProviders()
	defer func() {
		if err := providers.Shutdown(ctx); err != nil {
			log.Printf("telemetry shutdown: %v", err)
		}
	}()

	// ... build agents and runners as usual; ADK emits spans for invocations,
	// agent runs, model calls and tool calls.
}
```

### Plugins

```go
package main

import (
	"log"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/plugin/loggingplugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func newRunner(root agent.Agent) (*runner.Runner, error) {
	// A plugin applies to every agent, model call and tool call in the runner.
	audit, err := plugin.New(plugin.Config{
		Name: "audit",
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			log.Printf("invocation %s for user %s", ic.InvocationID(), ic.Session().UserID())
			return nil, nil // non-nil content would end the run with that reply
		},
		OnEventCallback: func(ic agent.InvocationContext, e *session.Event) (*session.Event, error) {
			if e.ErrorMessage != "" {
				log.Printf("model error: %s", e.ErrorMessage)
			}
			return nil, nil // nil keeps the event unchanged
		},
	})
	if err != nil {
		return nil, err
	}

	return runner.New(runner.Config{
		AppName:        "support",
		Agent:          root,
		SessionService: session.InMemoryService(),
		PluginConfig: runner.PluginConfig{
			Plugins:      []*plugin.Plugin{loggingplugin.MustNew("logging"), audit},
			CloseTimeout: 5 * time.Second,
		},
	})
}

func main() {
	var root agent.Agent
	if _, err := newRunner(root); err != nil {
		log.Fatal(err)
	}
}
```

### Launcher

The `full` launcher gives any agent a terminal chat, the ADK dev UI, a REST API and an A2A endpoint without writing server code:

```go
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
)

func main() {
	ctx := context.Background()
	var root agent.Agent // build with llmagent.New

	l := full.NewLauncher()
	cfg := &launcher.Config{AgentLoader: agent.NewSingleLoader(root)}
	// go run . console         -> chat in the terminal
	// go run . web api webui   -> REST API plus the ADK dev UI
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("%v\n\n%s", err, l.CommandLineSyntax())
	}
}
```

---

## What's new since v1.0.0

Releases since v1.0.0 (March 2026) have added packages rather than removed them: superseded packages such as the v1 A2A ones are deprecated but still build. The changes worth knowing about:

- **A2A v2.** `server/adka2a/v2` and `agent/remoteagent/v2` target `a2a-go/v2`; the v1 packages are deprecated. Remote agents take an `AgentCardProvider` (for example `remoteagent.NewAgentCardProvider(url)`).
- **Skills.** `tool/skilltoolset` loads skill folders (a `SKILL.md` plus resources) for an agent.
- **Vertex AI services.** `memory/vertexai` (Memory Bank) alongside `session/vertexai`, and `server/agentengine` plus `cmd/launcher/agentengine` for deploying to Vertex AI Agent Engine.
- **Triggers.** The web launcher gained `pubsub` and `eventarc` sub-launchers to run agents from Pub/Sub messages and Eventarc events.
- **Context types.** `tool.Context` is now an alias of `agent.ToolContext`, so tool callbacks and function tools take `agent.ToolContext` (existing code using `tool.Context` still compiles).
- **Testing.** `session/sessiontestsuite` for custom session services.

---

## Revision History

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | current | Added human approval for tool calls (a runnable confirmation round trip), a Memory and Artifacts section, and a scripted-model test that exercises a tool call. Every new program run with `go run` or `go test`. |
| 2026-10-21 | current | Rewritten against the current release. Earlier revisions used an API that ADK for Go never had (`gemini.New`, pointer `Config` arguments, single-value constructors, `agent.Response`, `tool.Schema`, `server/mcp`, `server/restapi`, `adk.NewAgent`); every example is now a program compiled with `go vet`, and the testing example runs under `go test`. Corrected the Go requirement to 1.25 and the workflow-agent import paths (`agent/workflowagents/...`). |
| April 2026 | 1.0.0 | GA guide. |

---
title: "Microsoft Agent Framework (Go) — Comprehensive guide"
description: "An overview of the Go SDK: the package map, agents, runs and streaming, structured output, and short summaries of sessions, tools, middleware, context providers, providers, MCP, A2A and OpenTelemetry with links to their own pages."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Comprehensive guide"
  order: 2
---

This page is an overview of the single-agent surface of `github.com/microsoft/agent-framework-go`: building and running agents and structured output in full, and short summaries of the topics that have their own pages: [providers](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_providers/), [tools](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/), [MCP](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_mcp/), [middleware](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/), [sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/), [hosting and A2A](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_hosting_and_a2a/) and [observability](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/). Workflows have [their own page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/), as does [testing](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_testing/).

The SDK is a public preview. Every program below compiles with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%%; the ones that call a model need credentials to run.

## Package map

| Package | Purpose |
| --- | --- |
| `agent` | `Agent`, `Config`, `Session`, run options, middleware and context-provider interfaces |
| `message` | `Message`, `Role`, and the typed content parts |
| `tool`, `tool/functool` | Tool interfaces; typed Go functions as tools |
| `tool/agenttool`, `tool/mcptool`, `tool/hostedtool`, `tool/shelltool` | An agent as a tool; MCP client and server bridging; provider-hosted tools; a shell tool |
| `provider/openaiprovider` | OpenAI and Azure OpenAI (Responses or Chat Completions), plus embeddings |
| `provider/foundryprovider` | Microsoft Foundry project models and existing server-side agents, Foundry memory |
| `provider/anthropicprovider`, `provider/geminiprovider`, `provider/copilotprovider` | Anthropic, Google Gemini, GitHub Copilot SDK |
| `provider/a2aprovider`, `provider/aguiprovider` | A2A client and server; AG-UI |
| `provider/otelprovider` | OpenTelemetry middleware (GenAI semantic conventions) |
| `agent/harness/...` | Tool auto-call, tool approval, agent modes, Todo planning, file memory, loops |
| `agent/skills`, `agent/memory`, `agent/textsearch`, `agent/compaction` | Skills, vector-backed chat memory, RAG, history compaction |
| `workflow`, `workflow/agentworkflow`, `workflow/inproc`, `workflow/checkpoint` | Graph workflows |

## Agents and runs

Every provider constructor returns the same `*agent.Agent`. You run it with one of three methods:

- `RunText(ctx, "text", opts...)` — a single user text message
- `RunMessage(ctx, msg, opts...)` — a single `*message.Message` (for images, files, or tool-approval responses)
- `Run(ctx, msgs, opts...)` — a slice of messages

All three return an `agent.ResponseStream`, which is an `iter.Seq2[*agent.ResponseUpdate, error]`. Nothing is sent until you consume it. Either range over it to stream, or call `.Collect()` to get an `*agent.Response`, whose `String()` method returns the concatenated text.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	// openai.NewClient reads OPENAI_API_KEY from the environment.
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You are a concise assistant. Answer in one sentence.",
		Config:       agent.Config{Name: "Helper"},
	})

	ctx := context.Background()

	// Non-streaming: Collect gathers every update into one *agent.Response.
	resp, err := a.RunText(ctx, "What is a goroutine?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.String())

	// Streaming: range over the ResponseStream.
	for update, err := range a.RunText(ctx, "Name three Go proverbs.", agent.Stream(true)) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(update.String())
	}
	fmt.Println()
}
```

`agent.Config` is embedded in every provider's `AgentConfig`, so the common fields (`Name`, `Description`, `Tools`, `Middlewares`, `FunctionMiddlewares`, `ContextProviders`, `HistoryProvider`, `Logger`, `RunOptions`) are set the same way regardless of the provider.

## Sessions and history

Without a session each run is independent. Create one with `a.CreateSession(ctx)` and pass it with `agent.WithSession(session)` to carry a conversation across runs. Depending on the provider, the history is kept on the service (the OpenAI Responses API and Foundry by default, with the session holding the response or conversation ID) or in the session itself (Chat Completions, Anthropic, Gemini, or Responses with `DisableStoreOutput: true`). Sessions implement `encoding/json`, so saving a conversation is `json.Marshal(session)`.

[Sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/) covers the session state bag, service-held and local history, saving and restoring, custom history stores, message filters, context providers and compaction.

## Structured output

`agent.WithStructuredOutput(&v)` derives a JSON schema from `v`'s type, asks the model for that shape, and decodes the result into `v`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

type Person struct {
	Name       string `json:"name"`
	Age        int    `json:"age"`
	Occupation string `json:"occupation"`
}

func main() {
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Extract the person described.",
	})

	var p Person
	_, err := a.RunText(context.Background(),
		"Ada Lovelace, 36, mathematician and writer.",
		agent.WithStructuredOutput(&p),
	).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v\n", p)
}
```

To fix a format for every run instead, put `agent.WithResponseFormat(jsonformat.MustFor[Person]())` (from `agent/format/jsonformat`) in `Config.RunOptions`.

## Function tools

`functool.New` and `functool.MustNew` turn a typed function `func(context.Context, In) (Out, error)` into a tool. The input schema is derived from `In` (property names from `json` tags, descriptions from `jsonschema` tags, required unless `omitempty`), and arguments are validated before your function runs. The OpenAI, Foundry, Anthropic and Gemini providers run tools automatically in a loop of up to 40 rounds, tuned with `AgentConfig.ToolAutoCall`. `tool.ApprovalRequiredFunc` makes a tool wait for a person's decision, `agenttool.New` turns an agent into a tool, and `hostedtool` has the provider-hosted web search, file search, code interpreter and MCP markers.

[Tools](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/) covers schemas, hand-written tools, the tool loop and its limits, tool modes, approvals and standing approval rules, agents as tools, hosted tools and the shell tool.

## Middleware

Agent middleware (`Config.Middlewares`, an `agent.Middleware` or `agent.MiddlewareFunc`) wraps a whole run and can change the input, filter the output, retry or refuse the run. Function middleware (`Config.FunctionMiddlewares`) wraps each tool call and can change the arguments, replace the result or skip the call. Both run in registration order, the first outermost.

[Middleware](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/) covers the run pipeline, streaming, short-circuiting, per-model-call interception, retries and timeouts, error types, message injection and the loop middleware.

## Context providers

A context provider runs before each run to add instructions, messages or tools, and after it to store state. `agent.NewContextProvider` builds one from two hooks, `Provide` and `Store`. Ready-made providers include `agent/textsearch` (retrieval), `agent/memory` (vector-backed chat memory), `agent/compaction`, `agent/skills`, the harness's todo, mode and file-memory providers, and `foundryprovider.NewMemoryProvider`. Context providers are trusted completely, so sanitise anything they fetch from external stores.

[Sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/#context-providers) covers all of them.

## Providers

| Provider | Constructor | Client |
| --- | --- | --- |
| OpenAI (Responses API) | `openaiprovider.NewAgent` / `NewResponsesAgent` | `openai.NewClient()` from `github.com/openai/openai-go/v3` |
| OpenAI (Chat Completions), OpenAI-compatible servers | `openaiprovider.NewChatCompletionsAgent` | same, with `option.WithBaseURL` for other servers |
| Azure OpenAI | the two above | `openai.NewClient(azure.WithEndpoint(endpoint, apiVersion), azure.WithTokenCredential(cred))` |
| Microsoft Foundry | `foundryprovider.NewAgent(endpoint, cred, target, cfg)` | built for you from the project endpoint |
| Anthropic, including Claude on Bedrock, Vertex AI and Foundry | `anthropicprovider.NewAgent` | `anthropic.NewClient()` from `github.com/anthropics/anthropic-sdk-go` |
| Google Gemini and Vertex AI | `geminiprovider.NewAgent` | `*genai.Client` from `google.golang.org/genai` |
| GitHub Copilot | `copilotprovider.NewAgent` | `*copilot.Client` from `github.com/github/copilot-sdk/go` |
| Remote A2A agent | `a2aprovider.NewAgentFromResolver` / `NewAgentFromCard` / `NewAgent` | `a2a-go` client |

Every constructor returns the same `*agent.Agent`, and provider-native request parameters are passed per run with `openaiprovider.ResponsesNewParams`, `openaiprovider.ChatCompletionNewParams`, `anthropicprovider.MessageNewParams` or `geminiprovider.GenerateContentConfig`.

[Providers](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_providers/) covers each provider's configuration, credentials, wire behaviour and limits, embeddings, and writing your own provider with `agent.New`.

## MCP

`tool/mcptool` builds on the official MCP Go SDK. `mcptool.Connect` opens a client session over any `mcp.Transport` (stdio, streamable HTTP, SSE, in memory) and `mcptool.ListTools` turns the server's tools into `[]tool.Tool`. In the other direction, `mcptool.AddTool` registers any `tool.FuncTool`, including an agent wrapped with `agenttool.New`, on an `*mcp.Server`. `hostedtool.MCPServer` asks the OpenAI Responses service to call a remote MCP server itself.

[MCP](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_mcp/) covers transports, authentication, tool naming and filtering, result mapping, prompts and resources, reconnecting, serving, and hosted MCP.

## A2A

`provider/a2aprovider` works in both directions on top of `github.com/a2aproject/a2a-go/v2`. `a2aprovider.NewHandler` adapts an `*agent.Agent` into an A2A request handler that you mount with `a2asrv`'s JSON-RPC and REST handlers next to the agent card, and `a2aprovider.NewAgentFromResolver` (or `NewAgentFromCard`, `NewAgent`) turns a remote A2A agent into an ordinary `*agent.Agent`, usable as a tool or a workflow participant. `ExecutorConfig.AgentRunMode` chooses between message replies and long-running tasks.

[Hosting and A2A](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_hosting_and_a2a/) covers servers, clients, long-running tasks, session stores, AG-UI and serving agents over HTTP.

## Observability

`otelprovider.NewMiddleware` emits agent, tool and model-call spans and metrics following the OpenTelemetry GenAI semantic conventions, using the global OpenTelemetry providers. Prompt content, tool arguments and tool results are not recorded. Setting `agent.Config.Logger` to an `*slog.Logger` logs run diagnostics, and `LogSensitiveData` adds payloads to those logs, so leave it off in production.

[Observability](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/) covers the spans and metrics, OTLP export, workflow telemetry and logging.

## Production notes

Every run takes a `context.Context`, which carries deadlines and cancellation to the provider and to each tool call; an `*agent.Agent` can be shared between goroutines, but one session should not run twice at once; and the SDK is 0.x, so pin the exact version. [Production](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_production/) covers timeouts at each layer, retries, concurrency limits, persisting sessions, credentials, tool-loop limits, workflow durability, a deployable service and a checklist; the `User-Agent` usage telemetry and its opt-out variables are on the [observability page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_observability/#usage-telemetry-in-the-user-agent-header).

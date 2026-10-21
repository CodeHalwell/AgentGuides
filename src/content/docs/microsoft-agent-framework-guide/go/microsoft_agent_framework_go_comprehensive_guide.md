---
title: "Microsoft Agent Framework (Go) — Comprehensive guide"
description: "Agents, runs and streaming, sessions, structured output, function tools and approvals, middleware, context providers, OpenAI / Azure OpenAI / Foundry / Anthropic / Gemini providers, MCP, A2A and OpenTelemetry in the Go SDK."
framework: "microsoft-agent-framework"
language: "go"
---

This page covers the single-agent surface of `github.com/microsoft/agent-framework-go`: building and running agents, sessions, structured output, tools, middleware, providers, MCP, A2A and tracing. Workflows have [their own page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_workflows/), as does [testing](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_testing/).

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

Without a session, each run is independent. Create one with `a.CreateSession(ctx)` and pass it with `agent.WithSession(session)` to carry a conversation across runs. If the provider stores history server-side (the OpenAI Responses API does by default), the session holds the service's conversation ID; otherwise the default in-memory history provider stores the messages in the session itself.

`agent.Session` implements `json.Marshaler` and `json.Unmarshaler`, so persisting a conversation is just encoding it:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:              "%%model=openai-small%%",
		DisableStoreOutput: true, // keep history locally in the session, not in the service
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	if _, err := a.RunText(ctx, "My name is Ada.", agent.WithSession(session)).Collect(); err != nil {
		log.Fatal(err)
	}

	// Persist: sessions marshal to JSON.
	data, err := json.Marshal(session)
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile("session.json", data, 0o600); err != nil {
		log.Fatal(err)
	}

	// Later, possibly in another process: restore and continue.
	var restored agent.Session
	if err := json.Unmarshal(data, &restored); err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "What is my name?", agent.WithSession(&restored)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

`DisableStoreOutput: true` tells the Responses agent not to store output in the service, which keeps the whole history local and portable. To store history somewhere else (a database, Cosmos DB), implement `agent.HistoryProvider` or build one with `agent.NewHistoryProvider` and set it as `Config.HistoryProvider`.

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

`functool.New` and `functool.MustNew` turn a typed function `func(context.Context, In) (Out, error)` into a `tool.FuncTool`. The input schema is derived from `In` using its `json` tags, and the `jsonschema` tag supplies a field description. Arguments are validated against the schema before your function is called, so a missing required field never reaches it. Fields without `omitempty` are required.

Wrap a tool in `tool.ApprovalRequiredFunc` to require a human decision. The run then ends with a `*message.ToolApprovalRequestContent` instead of calling the tool. Reply with `req.CreateResponse(approved, reason)` in the next run on the same session:

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

type WeatherIn struct {
	City  string `json:"city" jsonschema:"City name, e.g. Leeds"`
	Units string `json:"units,omitempty" jsonschema:"celsius or fahrenheit"`
}

type WeatherOut struct {
	TempC   float64 `json:"temp_c"`
	Summary string  `json:"summary"`
}

var weather = functool.MustNew(functool.Config{
	Name:        "get_weather",
	Description: "Get the current weather for a city",
}, func(ctx context.Context, in WeatherIn) (WeatherOut, error) {
	return WeatherOut{TempC: 14, Summary: "drizzle in " + in.City}, nil
})

type DeleteIn struct {
	Path string `json:"path"`
}

var deleteFile = functool.MustNew(functool.Config{
	Name:        "delete_file",
	Description: "Delete a file from the shared drive",
}, func(ctx context.Context, in DeleteIn) (string, error) {
	return "deleted " + in.Path, nil
})

func main() {
	ctx := context.Background()
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You are an operations assistant.",
		Config: agent.Config{
			Tools: []tool.Tool{
				weather,
				tool.ApprovalRequiredFunc(deleteFile), // needs a human yes/no
			},
		},
	})

	// Approval round-trips need a session so the agent can match responses to requests.
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "Delete /reports/old.csv", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}

	var answers []message.Content
	for c := range resp.Contents() {
		if req, ok := c.(*message.ToolApprovalRequestContent); ok {
			if call, ok := req.ToolCall.(*message.FunctionCallContent); ok {
				fmt.Printf("approve %s(%s)? (auto-yes in this sample)\n", call.Name, call.Arguments)
			}
			answers = append(answers, req.CreateResponse(true, ""))
		}
	}
	if len(answers) == 0 {
		fmt.Println(resp)
		return
	}
	resp, err = a.RunMessage(ctx, message.New(answers...), agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Tools can also be supplied per run with `agent.WithTool(t)`, and `agent.WithToolMode(tool.ToolModeRequired)` or `tool.RequireTool("name")` force a tool call.

For "approve once, then always allow" behaviour, the `agent/harness/toolapproval` middleware keeps standing approval rules in the session (`req.AlwaysApproveToolResponse()`).

### How tool calling works

The OpenAI, Foundry, Anthropic and Gemini providers install the `agent/harness/toolautocall` middleware. When the model returns `FunctionCallContent`, it invokes the matching tool, appends a `FunctionResultContent`, and calls the model again, up to 40 rounds by default. Tune it through the provider's `AgentConfig.ToolAutoCall`:

```go
maxRounds := 5
cfg := openaiprovider.AgentConfig{
	Model: "%%model=openai-small%%",
	ToolAutoCall: &toolautocall.Config{
		MaximumIterationsPerRequest: &maxRounds,
		AllowConcurrentInvocations:  true,
		IncludeDetailedErrors:       false, // the model sees a generic error message
	},
}
```

### An agent as a tool

`agenttool.New(a, agenttool.Config{})` wraps an agent as a `tool.FuncTool` with a single `query` string argument. The tool name is the agent's `Name`, sanitised. This gives you the "manager calls specialists" pattern without a workflow.

## Middleware

There are two kinds:

- **Agent middleware** (`agent.Middleware`, or the `agent.MiddlewareFunc` adapter) wraps the whole run. It receives `next`, the messages and the options, and returns the update stream, so it can rewrite input, filter output, time or log runs. Treat the input slices as read-only and clone messages before changing them (`msg.Clone()`).
- **Function middleware** (`agent.FunctionInvocationMiddleware`) wraps each tool call. It sees the tool, the call ID and the raw JSON arguments, and can change the arguments, replace the result, or skip the call entirely.

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"
	"log/slog"
	"os"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

// timing wraps each agent run and logs how long the stream took.
var timing agent.MiddlewareFunc = func(next agent.RunFunc, ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
	return func(yield func(*agent.ResponseUpdate, error) bool) {
		start := time.Now()
		defer func() { slog.Info("agent run", "took", time.Since(start)) }()
		for update, err := range next(ctx, msgs, opts...) {
			if !yield(update, err) {
				return
			}
		}
	}
}

// guardPaths rejects tool calls whose "path" argument leaves /reports.
func guardPaths(next func(context.Context, *agent.FunctionInvocationContext) (any, error), ctx context.Context, inv *agent.FunctionInvocationContext) (any, error) {
	var args struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(inv.Arguments), &args); err == nil && args.Path != "" {
		if len(args.Path) < 9 || args.Path[:9] != "/reports/" {
			return "refused: only /reports/ may be touched", nil
		}
	}
	return next(ctx, inv)
}

type ReadIn struct {
	Path string `json:"path"`
}

var readFile = functool.MustNew(functool.Config{Name: "read_file", Description: "Read a report"},
	func(ctx context.Context, in ReadIn) (string, error) { return "contents of " + in.Path, nil })

func main() {
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai-small%%",
		Config: agent.Config{
			Logger:              slog.New(slog.NewTextHandler(os.Stderr, nil)),
			Middlewares:         []agent.Middleware{timing},
			FunctionMiddlewares: []agent.FunctionInvocationMiddleware{guardPaths},
			Tools:               []tool.Tool{readFile},
		},
	})
	resp, err := a.RunText(context.Background(), "Read /etc/passwd").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Middlewares run in registration order, with the first outermost.

## Context providers

A context provider runs before each run to add instructions, messages or tools, and after it to store state. `agent.NewContextProvider` builds an additive one from two optional hooks, `Provide` and `Store`:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

// clock adds the current time to every run's instructions.
var clock = agent.NewContextProvider(agent.ContextProviderConfig{
	SourceID: "clock",
	Provide: func(ctx context.Context, inv agent.InvokingContext) ([]*message.Message, []agent.Option, error) {
		now := time.Now().Format(time.RFC1123)
		return nil, []agent.Option{agent.WithInstructions("The current time is " + now + ".")}, nil
	},
})

func main() {
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai-small%%",
		Config: agent.Config{ContextProviders: []agent.ContextProvider{clock}},
	})
	resp, err := a.RunText(context.Background(), "What day is it?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Provider state belongs in the session (`session.Set(key, v)` and `session.Get(key, &v)`; retrieve the session in a hook with `agent.GetOption(inv.Options, agent.WithSession)`). The upstream `examples/01-get-started/04_memory` sample shows a complete memory provider. Ready-made providers include `agent/memory` (vector-store chat memory), `agent/textsearch` (RAG), `agent/skills` and `foundryprovider.NewMemoryProvider` (Foundry memory stores).

Context providers may inject system messages and are trusted completely. Sanitise anything they fetch from external stores, because it is a route for indirect prompt injection.

## Providers

| Provider | Constructor | Client |
| --- | --- | --- |
| OpenAI (Responses API) | `openaiprovider.NewAgent` / `NewResponsesAgent` | `openai.NewClient()` from `github.com/openai/openai-go/v3` |
| OpenAI (Chat Completions) | `openaiprovider.NewChatCompletionsAgent` | same |
| Azure OpenAI | the two above | `openai.NewClient(option.WithBaseURL(...), azure.WithTokenCredential(cred))` |
| Microsoft Foundry | `foundryprovider.NewAgent(endpoint, cred, target, cfg)` | built for you from the project endpoint |
| Anthropic | `anthropicprovider.NewAgent` | `anthropic.NewClient()` from `github.com/anthropics/anthropic-sdk-go` |
| Google Gemini | `geminiprovider.NewAgent` | `*genai.Client` from `google.golang.org/genai` |
| GitHub Copilot | `copilotprovider.NewAgent` | `*copilot.Client` from `github.com/github/copilot-sdk/go` |
| Remote A2A agent | `a2aprovider.NewAgentFromResolver` / `NewAgentFromCard` / `NewAgent` | `a2a-go` client |

### OpenAI

See [Agents and runs](#agents-and-runs). `openai.NewClient()` reads `OPENAI_API_KEY`. `NewAgent` uses the Responses API. Choose `NewChatCompletionsAgent` for servers that only implement Chat Completions.

### Azure OpenAI

Azure OpenAI uses the same `openaiprovider` constructors, with the OpenAI client pointed at your resource. `Model` is the **deployment name**.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		log.Fatal(err)
	}

	client := openai.NewClient(
		// e.g. https://my-resource.openai.azure.com/openai/v1/
		option.WithBaseURL(os.Getenv("AZURE_OPENAI_ENDPOINT")),
		azure.WithTokenCredential(cred), // or azure.WithAPIKey(key)
	)

	// Responses API (the default). Use NewChatCompletionsAgent for Chat Completions.
	a := openaiprovider.NewResponsesAgent(client, openaiprovider.AgentConfig{
		Model:        os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), // the deployment name
		Instructions: "You are a helpful assistant.",
		Config:       agent.Config{Name: "AzureHelper"},
	})

	resp, err := a.RunText(context.Background(), "Summarise RBAC in two lines.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

`DefaultAzureCredential` is convenient for development. In production, prefer a specific credential such as managed identity.

### Microsoft Foundry

`foundryprovider.NewAgent` takes the **project endpoint** (`FOUNDRY_PROJECT_ENDPOINT`), an `azcore.TokenCredential`, and a target. There are two targets:

- `foundryprovider.ModelDeployment("gpt-4o-mini")` — the agent is defined in your Go code and calls a model deployed in the project through its OpenAI-compatible Responses endpoint.
- `foundryprovider.ServerAgent("name")` — invokes an agent that already exists in the Foundry project. Its instructions are owned by the service, so `AgentConfig.Instructions` is ignored.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/foundryprovider"
)

func main() {
	endpoint := os.Getenv("FOUNDRY_PROJECT_ENDPOINT")
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	// 1. A local agent definition that calls a model deployment in the project.
	local := foundryprovider.NewAgent(endpoint, cred,
		foundryprovider.ModelDeployment(os.Getenv("FOUNDRY_MODEL")),
		foundryprovider.AgentConfig{
			Instructions: "You are a helpful assistant.",
			Config:       agent.Config{Name: "Local"},
		},
	)
	resp, err := local.RunText(ctx, "Hello!").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	// 2. An existing server-side agent, addressed by name. Its instructions
	//    live in Foundry, so AgentConfig.Instructions is ignored.
	server := foundryprovider.NewAgent(endpoint, cred,
		foundryprovider.ServerAgent("support-triage"),
		foundryprovider.AgentConfig{},
	)
	resp, err = server.RunText(ctx, "My invoice is wrong.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

The package also provides `NewOpenAIClient` (a raw OpenAI client for a Foundry endpoint), `NewEmbeddingGenerator`, `NewMemoryProvider`, and per-run `WithClientHeader(s)` for `x-client-*` headers. Foundry-hosted agent *deployment* is not implemented in Go.

### Anthropic and Gemini

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/anthropicprovider"
)

func main() {
	// anthropic.NewClient reads ANTHROPIC_API_KEY from the environment.
	a := anthropicprovider.NewAgent(anthropic.NewClient(), anthropicprovider.AgentConfig{
		Model:        "%%model=anthropic%%",
		Instructions: "You are a helpful assistant.",
		Config:       agent.Config{Name: "Claude"},
	})
	resp, err := a.RunText(context.Background(), "Explain context.Context briefly.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/geminiprovider"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  os.Getenv("GEMINI_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		log.Fatal(err)
	}
	a := geminiprovider.NewAgent(client, geminiprovider.AgentConfig{
		Model:        "%%model=gemini%%",
		Instructions: "You are a helpful assistant.",
		Config:       agent.Config{Name: "Gemini"},
	})
	resp, err := a.RunText(ctx, "Hello").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Provider-native request parameters can be passed per run: `openaiprovider.ResponsesNewParams(...)`, `openaiprovider.ChatCompletionNewParams(...)`, `anthropicprovider.MessageNewParams(...)` and `geminiprovider.GenerateContentConfig(...)` all return an `agent.Option`.

## MCP

`tool/mcptool` builds on the official MCP Go SDK (`github.com/modelcontextprotocol/go-sdk/mcp`).

**Consuming MCP servers.** `mcptool.Connect` opens a client session over any `mcp.Transport`, and `mcptool.ListTools` turns the server's tools into `[]tool.Tool`:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os/exec"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()

	// Remote server over streamable HTTP.
	docs, err := mcptool.Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint: "https://learn.microsoft.com/api/mcp",
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = docs.Close() }()

	// Local server as a subprocess over stdio.
	fs, err := mcptool.Connect(ctx, &mcp.CommandTransport{
		Command: exec.Command("npx", "-y", "@modelcontextprotocol/server-filesystem", "."),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = fs.Close() }()

	var tools []tool.Tool
	for _, s := range []*mcp.ClientSession{docs, fs} {
		ts, err := mcptool.ListTools(ctx, s)
		if err != nil {
			log.Fatal(err)
		}
		tools = append(tools, ts...)
	}

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Use the tools to answer questions about Azure and local files.",
		Config:       agent.Config{Tools: tools},
	})
	resp, err := a.RunText(ctx, "How do I create a storage account with the az CLI?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Close the sessions when you are done; for `CommandTransport` that also stops the child process.

**Exposing an agent as an MCP server.** Combine `agenttool.New` with `mcptool.AddTool`:

```go
package main

import (
	"context"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool/agenttool"
	"github.com/microsoft/agent-framework-go/tool/mcptool"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3"
)

func main() {
	reviewer := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You review Go code for correctness and idiom.",
		Config: agent.Config{
			Name:        "go_reviewer", // becomes the tool name
			Description: "Reviews a Go snippet and lists problems.",
		},
	})

	srv := mcp.NewServer(&mcp.Implementation{Name: "go-reviewer", Version: "1.0.0"}, nil)
	mcptool.AddTool(srv, agenttool.New(reviewer, agenttool.Config{}))

	// Serve over stdio; any MCP client (Claude Desktop, VS Code, another agent) can call it.
	if err := srv.Run(context.Background(), &mcp.StdioTransport{}); err != nil {
		log.Fatal(err)
	}
}
```

`mcptool.AddTool` accepts any `tool.FuncTool`, so plain `functool` tools can be served the same way.

## A2A

`provider/a2aprovider` works in both directions on top of `github.com/a2aproject/a2a-go/v2`.

**Hosting an agent.** `a2aprovider.NewHandler` adapts an `*agent.Agent` into an A2A request handler, which you mount with `a2asrv`'s JSON-RPC and REST handlers alongside the agent card:

```go
package main

import (
	"log"
	"net/http"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	const url = "http://localhost:5000"

	policy := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You answer questions about the returns policy.",
		Config:       agent.Config{Name: "PolicyAgent"},
	})

	card := &a2a.AgentCard{
		Name:               "PolicyAgent",
		Description:        "Answers questions about the returns policy.",
		Version:            "1.0.0",
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		Skills: []a2a.AgentSkill{{
			ID: "returns", Name: "Returns policy",
			Description: "Explains the returns policy.",
			Tags:        []string{"policy"},
		}},
		SupportedInterfaces: []*a2a.AgentInterface{
			a2a.NewAgentInterface(url, a2a.TransportProtocolJSONRPC),
			a2a.NewAgentInterface(url, a2a.TransportProtocolHTTPJSON),
		},
	}

	handler := a2aprovider.NewHandler(policy,
		a2aprovider.ExecutorConfig{SessionStore: a2aprovider.NewInMemorySessionStore()},
		a2asrv.WithExtendedAgentCard(card),
	)

	mux := http.NewServeMux()
	mux.Handle("POST /{$}", a2asrv.NewJSONRPCHandler(handler))
	mux.Handle("/", a2asrv.NewRESTHandler(handler))
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	log.Fatal(http.ListenAndServe("localhost:5000", mux))
}
```

**Calling a remote agent.** The remote agent becomes an ordinary `*agent.Agent`, so it can be used as a tool or a workflow participant:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/a2aproject/a2a-go/v2/a2aclient/agentcard"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/a2aprovider"
)

func main() {
	ctx := context.Background()

	// Fetches /.well-known/agent-card.json and builds an *agent.Agent from it.
	remote, err := a2aprovider.NewAgentFromResolver(ctx, agentcard.DefaultResolver,
		"http://localhost:5000/", a2aprovider.AgentConfig{})
	if err != nil {
		log.Fatal(err)
	}

	session, err := remote.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := remote.RunText(ctx, "Can I return opened items?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

`ExecutorConfig.AgentRunMode` selects whether the server replies with a message (the default) or a long-running task (`a2aprovider.ReturnTask()`, `ReturnTaskWhen(...)`). On the client side, `a2aprovider.WithTaskID`, `TaskIDFromSession` and `TaskStateFromSession` handle task continuation. See `examples/02-agents/a2a` upstream for polling and stream reconnection.

## Observability

`otelprovider.NewMiddleware` emits agent, tool and model-call spans and metrics following the OpenTelemetry GenAI semantic conventions. It uses the global OpenTelemetry providers, so configure those first:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/provider/otelprovider"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

func main() {
	ctx := context.Background()

	exp, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		log.Fatal(err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exp))
	defer func() { _ = tp.Shutdown(ctx) }()
	otel.SetTracerProvider(tp) // otelprovider uses the global providers

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai-small%%",
		Config: agent.Config{
			Name:        "Traced",
			Middlewares: []agent.Middleware{otelprovider.NewMiddleware(otelprovider.MiddlewareConfig{SourceName: "my-service"})},
		},
	})
	resp, err := a.RunText(ctx, "Hello").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Prompt content, tool arguments and tool results are not recorded. `MiddlewareConfig.DisableModelTracing` drops the model-call spans and keeps the agent and tool spans. Workflows have a separate package, `workflow/observability/opentelemetry`.

Setting `agent.Config.Logger` to an `*slog.Logger` logs run diagnostics. `LogSensitiveData` adds payloads to those logs, so leave it off in production.

## Production notes

- **Telemetry.** The OpenAI and Foundry adapters add the SDK identity and a feature-usage bitmask to the `User-Agent` header, but only on approved Azure/Foundry HTTPS hosts. Set `AGENT_FRAMEWORK_FEATURE_MASK_DISABLED=1` to drop the bitmask, or `AGENT_FRAMEWORK_USER_AGENT_DISABLED=1` to drop the framework's whole `User-Agent` contribution.
- **Cancellation.** Every run takes a `context.Context`, which is passed through to the provider client and to each tool call. Set deadlines on it.
- **Concurrency.** An `*agent.Agent` can be shared. A `*agent.Session` is safe for concurrent use after initialisation, but one conversation should not run twice at once.
- **Background responses.** `agent.AllowBackgroundResponses(true)` and `agent.WithContinuationToken(token)` support long-running Responses-API work that you poll for or resume.
- **Version pinning.** Pin the exact v0.x version in `go.mod` and read the release notes before upgrading.

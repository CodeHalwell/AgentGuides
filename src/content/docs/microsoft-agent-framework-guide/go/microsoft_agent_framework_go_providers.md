---
title: "Microsoft Agent Framework (Go) — Providers"
description: "Every model provider in the Go SDK: OpenAI Responses and Chat Completions, OpenAI-compatible servers, Azure OpenAI, Microsoft Foundry model deployments and server-side agents, Anthropic (including Claude on Bedrock, Vertex AI and Foundry), Google Gemini and Vertex AI, the GitHub Copilot SDK, embeddings, credentials, per-run native parameters and writing your own provider."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Providers"
  order: 3
---

In the Go SDK a provider is a constructor that returns an ordinary `*agent.Agent`. There is no separate chat-client type to wrap: `openaiprovider.NewAgent`, `anthropicprovider.NewAgent` and the rest each take a client from the vendor's own Go SDK plus an `AgentConfig`, and hand back the same agent type. Everything else in the framework (sessions, tools, middleware, context providers, workflows) works with any of them, so changing provider changes one constructor.

This page goes through each provider: how to construct it, which credentials and environment variables it reads, what it sends on the wire, and what it does not support. The [comprehensive guide](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_comprehensive_guide/) covers runs, streaming and structured output; [tools](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/) covers function tools and hosted tools in detail.

Every program below compiles with `go vet` and `go build` against `github.com/microsoft/agent-framework-go` v%%version=github.com/microsoft/agent-framework-go%% (Go 1.26). The OpenAI, OpenAI-compatible, Anthropic, Gemini, embedding and custom-provider programs were also run against a local stand-in for each wire API, reached through the vendor SDKs' base-URL environment variables. The Foundry behaviour described here (URLs, token audience, headers, session pinning) was checked with a stub credential against an HTTPS stand-in, but the Foundry program itself, and the Azure OpenAI, Bedrock, Vertex AI and Copilot programs, were only compiled.

## Choosing a provider

| Provider | Constructor | Client it takes | Wire API | Conversation history | Hosted tools | Structured output |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI Responses | `openaiprovider.NewAgent`, `NewResponsesAgent` | `openai.Client` (`github.com/openai/openai-go/v3`) | `POST /responses` | service-side by default, local with `DisableStoreOutput` | web search, file search, code interpreter, hosted MCP | `text.format` JSON schema |
| OpenAI Chat Completions | `openaiprovider.NewChatCompletionsAgent` | `openai.Client` | `POST /chat/completions` | local (in the session) | web search (`web_search_options`) | `response_format` JSON schema |
| OpenAI-compatible server | `NewChatCompletionsAgent` + `option.WithBaseURL` | `openai.Client` | `/chat/completions` | local | depends on the server | depends on the server |
| Azure OpenAI | either OpenAI constructor | `openai.Client` + `azure.WithTokenCredential` or `azure.WithAPIKey` | `/openai/v1/responses` or `/chat/completions` | as OpenAI | as OpenAI | as OpenAI |
| Microsoft Foundry | `foundryprovider.NewAgent` | built for you from the project endpoint and an `azcore.TokenCredential` | `{project}/openai/v1/responses`, or the agent's own endpoint | service-side by default | as OpenAI Responses | as OpenAI Responses |
| Anthropic | `anthropicprovider.NewAgent` | `anthropic.Client` (`github.com/anthropics/anthropic-sdk-go`) | `POST /v1/messages` | local | web search, code execution | `output_config.format` |
| Google Gemini | `geminiprovider.NewAgent` | `*genai.Client` (`google.golang.org/genai`) | `generateContent` / `streamGenerateContent` | local | Google Search, code execution, file search | `responseJsonSchema` |
| GitHub Copilot | `copilotprovider.NewAgent` | `*copilot.Client` (`github.com/github/copilot-sdk/go`) | the Copilot CLI over JSON-RPC | in the Copilot session | Copilot's own tools | not supported |
| Remote A2A agent | `a2aprovider.NewAgent...` | an A2A client | A2A JSON-RPC or HTTP+JSON | on the remote agent | n/a | n/a |

A rough guide to choosing:

- **OpenAI or Azure OpenAI, new code:** the Responses constructors. They support every hosted tool, return reasoning summaries, can run in the background, and by default keep the conversation on the service so a long session does not resend its history.
- **Any server that claims OpenAI compatibility** (Ollama, vLLM, LM Studio, LiteLLM, OpenRouter): `NewChatCompletionsAgent` with a base URL. Few of them implement the Responses API.
- **Microsoft Foundry projects:** `foundryprovider.NewAgent`. It is the Responses agent with Foundry's URL, audience and extra headers already set, and it can also call an agent that already exists in the project.
- **Claude:** `anthropicprovider.NewAgent`. The same constructor serves Claude on Amazon Bedrock, Google Vertex AI and Microsoft Foundry, because those are options on the Anthropic client.
- **Gemini:** `geminiprovider.NewAgent`, for both the Gemini API and Vertex AI.

There is no dedicated Ollama, Amazon Bedrock (Converse), Mistral or Foundry Local provider in v%%version=github.com/microsoft/agent-framework-go%%. Use the OpenAI-compatible route where the server offers one. Calling a remote A2A agent is covered under [A2A](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_comprehensive_guide/#a2a) in the comprehensive guide.

## What every provider has in common

Each provider package has its own `AgentConfig`, and each one embeds `agent.Config`. The embedded fields (`Name`, `Description`, `Tools`, `Middlewares`, `FunctionMiddlewares`, `ContextProviders`, `HistoryProvider`, `Logger`, `RunOptions` and the rest) are set the same way whatever the provider. The provider-specific fields are few:

| Field | OpenAI | Foundry | Anthropic | Gemini | Copilot |
| --- | --- | --- | --- | --- | --- |
| `Model` | model ID, or Azure deployment name | the target argument instead | model ID | model ID | set in `SessionConfig` |
| `Instructions` | yes | yes (ignored for `ServerAgent`) | yes | yes | appended to the Copilot system message |
| `ToolAutoCall *toolautocall.Config` | yes | yes | yes | yes | no (Copilot runs tools itself) |
| `DisableStoreOutput` | Responses only | yes | no | no | no |
| `ProviderName` | yes, default `"openai"` | fixed to `"microsoft.foundry"` | fixed to `"anthropic"` | fixed to `"gcp.gemini"` | fixed to `"copilot"` |
| `OpenAIOptions` | no | per-agent OpenAI client options | no | no | no |
| `SessionConfig *copilot.SessionConfig` | no | no | no | no | yes |

`Instructions` is turned into an `agent.WithInstructions` run option and prepended to `RunOptions`, so instructions passed per run are added to it rather than replacing it. All providers join multiple instruction values with a newline.

The OpenAI, Foundry, Anthropic and Gemini providers install the `toolautocall` middleware, which runs function tools and loops until the model stops asking for them; `ToolAutoCall` tunes it. When the field is nil the defaults apply (40 rounds, three consecutive failing rounds, serial calls), and the agent's `Logger` and `LogSensitiveData` are passed through. A non-nil `ToolAutoCall` replaces that whole configuration, so set `Logger` on it yourself if you want tool logs. The [tools page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#the-tool-loop) has the details.

Each provider also has one escape hatch, a run option that carries the vendor SDK's own request type. Anything the framework does not map can be set there:

| Provider | Run option | Type |
| --- | --- | --- |
| OpenAI Responses, Foundry | `openaiprovider.ResponsesNewParams(p)` | `responses.ResponseNewParams` |
| OpenAI Chat Completions | `openaiprovider.ChatCompletionNewParams(p)` | `openai.ChatCompletionNewParams` |
| Anthropic | `anthropicprovider.MessageNewParams(p)` | `anthropic.MessageNewParams` |
| Gemini | `geminiprovider.GenerateContentConfig(c)` | `genai.GenerateContentConfig` |

The framework starts from the struct you pass and fills in what it owns: messages, tools, tool choice, instructions, the response format, and the model when yours is empty. A `Model` set in the native parameters wins over `AgentConfig.Model` (Gemini has no model field in `GenerateContentConfig`, so there it is always `AgentConfig.Model`). Put a native option in `AgentConfig.RunOptions` to apply it to every run, or pass it to one run. Single-valued options are looked up from the end of the option list, and agent-level `RunOptions` are prepended to each run's options, so a native-parameters option passed to a run replaces the agent-level one entirely. The two structs are not merged.

## OpenAI

### The Responses API

`openaiprovider.NewAgent` currently calls `NewResponsesAgent`; the package documents that the API behind `NewAgent` may change, so call `NewResponsesAgent` when you depend on Responses behaviour. `openai.NewClient()` reads `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` from the environment; pass `option.WithAPIKey` and friends to set them in code.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()

	// openai.NewClient reads OPENAI_API_KEY (and OPENAI_BASE_URL, if set).
	a := openaiprovider.NewResponsesAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "You are a concise assistant.",
		Config:       agent.Config{Name: "Helper"},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "Name a prime number.", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	// The service stored the response; the session now holds its ID.
	fmt.Println("service ID:", session.ServiceID())
	fmt.Println("response ID:", resp.ID, "finish reason:", resp.FinishReason)

	usage := resp.Usage()
	fmt.Printf("tokens: in=%d out=%d\n", usage.InputTokenCount, usage.OutputTokenCount)

	// Streaming: each update carries part of the text.
	for update, err := range a.RunText(ctx, "And another?", agent.WithSession(session), agent.Stream(true)) {
		if err != nil {
			log.Fatal(err)
		}
		for _, c := range update.Contents {
			if t, ok := c.(*message.TextContent); ok {
				fmt.Print(t.Text)
			}
		}
	}
	fmt.Println()
}
```

What the agent sends, per run:

| Framework input | Responses request field |
| --- | --- |
| `AgentConfig.Model`, unless the native params set one | `model` |
| `Instructions` and every `agent.WithInstructions` | `instructions`, joined with newlines |
| messages (history, context, your input) | `input` items |
| function tools | `tools[]` of type `function`, with the tool's JSON schema as `parameters` |
| `agent.WithToolMode` | `tool_choice` (`auto`, `none`, `required`, or a named function) |
| `agent.WithStructuredOutput` / `WithResponseFormat` | `text.format` of type `json_schema` |
| session service ID starting `conv_` | `conversation` |
| any other session service ID | `previous_response_id` |
| `agent.AllowBackgroundResponses(true)` | `background: true` |
| `DisableStoreOutput: true` | `store: false`, plus `include: ["reasoning.encrypted_content"]` |
| `agent.Stream(true)` | `stream: true` |

By default the Responses API stores each response, and the agent records the response ID in the session as its service ID. The next run on that session sends only the new message with `previous_response_id`, not the whole history. With `DisableStoreOutput: true` the request says `store: false`, the session gets no service ID, and the agent keeps the transcript itself in the session through its history provider. That keeps the whole conversation local and portable, at the cost of resending it on every turn. [Sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/#service-held-and-local-history) explains both modes and how to use an OpenAI Conversations object instead.

When output storage is off, the agent also asks for `reasoning.encrypted_content`, so reasoning items from a reasoning model can be sent back on the next turn without the service holding them. Pass `openaiprovider.ResponsesIncludeReasoningEncryptedContent(false)` to stop that, for example for a model that rejects the include. Setting `Store` in `ResponsesNewParams` overrides `DisableStoreOutput` in either direction.

### Reasoning

Reasoning effort and summaries are set through the native parameters. Summaries come back as `*message.TextReasoningContent`, separate from the answer text, and reasoning tokens are counted in `UsageDetails.ReasoningTokenCount`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

func main() {
	a := openaiprovider.NewResponsesAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Think carefully, then answer in one sentence.",
	})

	reasoning := openaiprovider.ResponsesNewParams(responses.ResponseNewParams{
		Reasoning: shared.ReasoningParam{
			Effort:  shared.ReasoningEffortMedium,
			Summary: shared.ReasoningSummaryAuto,
		},
		MaxOutputTokens: openai.Int(2000),
	})

	resp, err := a.RunText(context.Background(),
		"A train covers 180 miles at 60 mph. How long does it take?", reasoning).Collect()
	if err != nil {
		log.Fatal(err)
	}
	for c := range resp.Contents() {
		if r, ok := c.(*message.TextReasoningContent); ok && r.Text != "" {
			fmt.Println("[reasoning]", r.Text)
		}
	}
	fmt.Println(resp)
	fmt.Println("reasoning tokens:", resp.Usage().ReasoningTokenCount)
}
```

### Chat Completions

`NewChatCompletionsAgent` takes the same client and the same `AgentConfig`, but talks to `/chat/completions`. Instructions become a leading `system` message. History is always local: Chat Completions has no stored-response concept, `DisableStoreOutput` is ignored, and the session carries the transcript. Native parameters go through `ChatCompletionNewParams`:

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
	ctx := context.Background()
	a := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Answer in British English.",
		Config: agent.Config{
			Name: "Chat",
			// Applied to every run of this agent.
			RunOptions: []agent.Option{
				openaiprovider.ChatCompletionNewParams(openai.ChatCompletionNewParams{
					Temperature: openai.Float(0.2),
					Seed:        openai.Int(7),
				}),
			},
		},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, q := range []string{"My name is Ada.", "What is my name?"} {
		resp, err := a.RunText(ctx, q, agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(resp)
	}
	// No service ID: the history lives in the session.
	fmt.Printf("service ID: %q\n", session.ServiceID())
}
```

### Chat Completions or Responses?

Use Responses for OpenAI and Azure OpenAI unless you have a reason not to. It is the only OpenAI path with file search, code interpreter, hosted MCP, background responses and server-held conversations, and reasoning summaries are only returned there. Use Chat Completions for servers that implement nothing else, and when you want the agent, not the service, to hold the transcript without having to remember `DisableStoreOutput`.

## OpenAI-compatible servers

Point the OpenAI client at another server with `option.WithBaseURL` and use `NewChatCompletionsAgent`. `ProviderName` changes the name the agent reports (`a.ProviderName()`) and the `gen_ai.provider.name` recorded by the OpenTelemetry middleware; it has no effect on the request. Ollama, for example, serves the OpenAI API under `/v1/` and ignores the key:

```go
package main

import (
	"cmp"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	baseURL := cmp.Or(os.Getenv("LOCAL_LLM_BASE_URL"), "http://localhost:11434/v1/")
	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey("unused"), // Ollama ignores the key, but the client needs one
	)

	a := openaiprovider.NewChatCompletionsAgent(client, openaiprovider.AgentConfig{
		Model:        cmp.Or(os.Getenv("LOCAL_LLM_MODEL"), "llama3.2"),
		Instructions: "You are a helpful assistant.",
		ProviderName: "ollama",
	})
	fmt.Println("provider:", a.ProviderName())

	resp, err := a.RunText(context.Background(), "Say hello in French.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Whether tools, streaming and structured output work depends on the server and the model. If a server rejects `response_format` with a JSON schema, leave out `WithStructuredOutput` and parse the reply yourself.

## Azure OpenAI

Azure OpenAI uses the same two constructors. Build the client with the resource's v1 endpoint (`https://<resource>.openai.azure.com/openai/v1/`) and an Azure credential option from `github.com/openai/openai-go/v3/azure`; `Model` is then your **deployment name**, not the model ID.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/azure"
	"github.com/openai/openai-go/v3/option"
)

func main() {
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")             // https://my-resource.openai.azure.com/openai/v1/
	deployment := os.Getenv("AZURE_OPENAI_DEPLOYMENT_NAME")    // the deployment, not the model

	var auth option.RequestOption
	if key := os.Getenv("AZURE_OPENAI_API_KEY"); key != "" {
		auth = azure.WithAPIKey(key)
	} else {
		cred, err := azidentity.NewDefaultAzureCredential(nil)
		if err != nil {
			log.Fatal(err)
		}
		auth = azure.WithTokenCredential(cred) // scope https://cognitiveservices.azure.com/.default
	}
	client := openai.NewClient(option.WithBaseURL(endpoint), auth)

	responsesAgent := openaiprovider.NewResponsesAgent(client, openaiprovider.AgentConfig{
		Model:        deployment,
		Instructions: "You are a helpful assistant.",
	})
	chatAgent := openaiprovider.NewChatCompletionsAgent(client, openaiprovider.AgentConfig{
		Model:        deployment,
		Instructions: "You are a helpful assistant.",
	})

	ctx := context.Background()
	r1, err := responsesAgent.RunText(ctx, "Summarise RBAC in one line.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	r2, err := chatAgent.RunText(ctx, "Summarise ABAC in one line.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(r1, "\n", r2)
}
```

`azure.WithTokenCredential` takes any `azcore.TokenCredential`. `DefaultAzureCredential` is convenient on a developer machine; in production use a specific credential such as `azidentity.NewManagedIdentityCredential`, which avoids probing a chain of sources on every cold start. To use a non-default scope, pass `azure.WithTokenCredentialScopes(...)` as an option to `WithTokenCredential`.

For an older, deployment-routed endpoint with an `api-version` query, use `azure.WithEndpoint(endpoint, apiVersion)` instead of `option.WithBaseURL`. The v1 endpoint needs no API version and is what the upstream samples use.

## Microsoft Foundry

`foundryprovider.NewAgent(endpoint, credential, target, config)` takes the **project endpoint** (`https://<resource>.services.ai.azure.com/api/projects/<project>`), an `azcore.TokenCredential`, and a target. The provider builds the OpenAI client itself: it appends `/openai/v1/` to the project endpoint, requests tokens for the `https://ai.azure.com/.default` audience, and removes the `api-version` query that the Azure option would otherwise add. The constructor panics if the endpoint is not an absolute URL, the credential is nil, or the target name is blank, and the Azure SDK refuses to send a bearer token over plain HTTP.

There are two targets:

| Target | What runs | Who owns the instructions and tools |
| --- | --- | --- |
| `foundryprovider.ModelDeployment("name")` | your Go agent definition, calling a model deployment in the project through its Responses endpoint | your code: `Instructions`, `Tools`, middleware |
| `foundryprovider.ServerAgent("name")` | an agent that already exists in the project, called through `{project}/agents/{name}/endpoint/protocols/openai/responses?api-version=v1` | the service; `AgentConfig.Instructions` is ignored, and `ID` and `Name` default to the agent's name |

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
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/hostedtool"
)

func main() {
	endpoint := os.Getenv("FOUNDRY_PROJECT_ENDPOINT")
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	// 1. A Go-defined agent on a model deployment in the project.
	local := foundryprovider.NewAgent(endpoint, cred,
		foundryprovider.ModelDeployment(os.Getenv("FOUNDRY_MODEL")),
		foundryprovider.AgentConfig{
			Instructions: "You answer with current information and cite sources.",
			Config: agent.Config{
				Name:  "Researcher",
				Tools: []tool.Tool{&hostedtool.WebSearch{}},
			},
		},
	)
	resp, err := local.RunText(ctx, "What changed in Go 1.26?",
		foundryprovider.WithClientHeader("x-client-request-source", "docs-sample"),
	).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
	if model, ok := resp.AdditionalProperties["ServedModel"]; ok {
		fmt.Println("served by:", model) // from the x-ms-served-model response header
	}

	// 2. An agent defined in the Foundry project, addressed by name.
	triage := foundryprovider.NewAgent(endpoint, cred,
		foundryprovider.ServerAgent("support-triage"),
		foundryprovider.AgentConfig{},
	)
	session, err := triage.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err = triage.RunText(ctx, "My invoice is wrong.", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(triage.Name(), "says:", resp)
}
```

Foundry-specific options and helpers:

| API | Purpose |
| --- | --- |
| `AgentConfig.DisableStoreOutput` | as for OpenAI: `store: false`, history kept locally |
| `AgentConfig.OpenAIOptions` | extra `option.RequestOption`s for the per-agent OpenAI client (an HTTP client, retries, middleware). Foundry's own endpoint, authentication and protocol options are applied after them and win |
| `WithClientHeader(name, value)`, `WithClientHeaders(map)` | per-run `x-client-*` headers. Names that do not start with `x-client-` panic |
| `WithHostedAgentUserIdentity(id)` | sends `x-ms-user-identity` for a hosted agent, per run |
| `WithHostedAgentSessionID(id)`, `SetHostedAgentSessionID(session, id)`, `HostedAgentSessionID(session)` | pins a hosted agent's sandbox session (`agent_session_id`). The ID the service returns is stored in the agent session and sent again on later runs; a run option that disagrees with the stored ID fails the run |
| `resp.AdditionalProperties["ServedModel"]` | the model that actually served the request, when the service reports one |
| `NewOpenAIClient(endpoint, cred, opts...)` | a plain `openai.Client` for a Foundry resource or project, with the right base URL and audience |
| `NewEmbeddingGenerator(endpoint, cred, cfg)` | embeddings from a Foundry resource (see [Embeddings](#embeddings)) |
| `NewMemoryProvider(...)` | a context provider over Foundry memory stores (see [sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/#foundry-memory)) |

Creating or deploying agents in a Foundry project is not implemented in Go. Define the agent in the portal or with another SDK, then call it with `ServerAgent`.

## Anthropic

`anthropicprovider.NewAgent` takes an `anthropic.Client`. `anthropic.NewClient()` reads `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) and `ANTHROPIC_BASE_URL`. Instructions and any system-role messages are sent as `system` text blocks; history is always local.

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

type Verdict struct {
	Answer     string  `json:"answer"`
	Confidence float64 `json:"confidence" jsonschema:"between 0 and 1"`
}

func main() {
	ctx := context.Background()
	a := anthropicprovider.NewAgent(anthropic.NewClient(), anthropicprovider.AgentConfig{
		Model:        "%%model=anthropic%%",
		Instructions: "You are a careful assistant.",
		Config:       agent.Config{Name: "Claude"},
	})

	// Plain text, with a larger output budget than the 4096-token default.
	resp, err := a.RunText(ctx, "Explain context.Context in two sentences.",
		anthropicprovider.MessageNewParams(anthropic.MessageNewParams{MaxTokens: 1024}),
	).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	// Structured output uses Anthropic's native JSON output format.
	var v Verdict
	if _, err := a.RunText(ctx, "Is 97 prime?", agent.WithStructuredOutput(&v)).Collect(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v\n", v)
}
```

Behaviour to know about:

- **`max_tokens` defaults to 4096.** The Messages API requires a value, so the provider fills one in when `MessageNewParams` does not. Raise it for long answers, and keep it above any thinking budget.
- **Structured output** goes in `output_config.format` as a JSON schema, which current Claude models honour natively. A non-object result type is wrapped in a `data` property on the wire and unwrapped for you, as on the other providers.
- **Extended thinking** is enabled with `MessageNewParams{Thinking: anthropic.ThinkingConfigParamOfEnabled(budget)}`. Thinking comes back as `*message.TextReasoningContent`; redacted thinking arrives with empty `Text` and the signature in `ProtectedData`, which must be kept in history for the next turn (the session does this).
- **Hosted tools:** `&hostedtool.WebSearch{}` becomes the `web_search_20250305` tool, and `&hostedtool.CodeInterpreter{}` becomes `code_execution_20250825`. File search and hosted MCP are not mapped.
- **Files:** `DataContent` and `URIContent` images and PDFs are sent inline or by URL. A `HostedFileContent` (a provider file ID) is rejected with an error.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/anthropicprovider"
)

func main() {
	a := anthropicprovider.NewAgent(anthropic.NewClient(), anthropicprovider.AgentConfig{
		Model:        "%%model=anthropic%%",
		Instructions: "Reason it through, then reply with only the final answer.",
	})

	thinking := anthropicprovider.MessageNewParams(anthropic.MessageNewParams{
		MaxTokens: 8000,
		Thinking:  anthropic.ThinkingConfigParamOfEnabled(4000), // must be below MaxTokens
	})
	resp, err := a.RunText(context.Background(),
		"A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball?",
		thinking,
	).Collect()
	if err != nil {
		log.Fatal(err)
	}
	for c := range resp.Contents() {
		if r, ok := c.(*message.TextReasoningContent); ok {
			switch {
			case r.Text != "":
				fmt.Println("[thinking]", r.Text)
			case r.ProtectedData != "":
				fmt.Println("[thinking redacted]")
			}
		}
	}
	fmt.Println(resp)
}
```

### Claude on Bedrock, Vertex AI and Foundry

The Anthropic Go SDK reaches Claude on other clouds through client options, and `anthropicprovider.NewAgent` accepts the resulting client unchanged. The model name follows each platform's convention.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/foundry"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/anthropicprovider"
)

func main() {
	ctx := context.Background()
	var a *agent.Agent

	switch os.Getenv("CLAUDE_PLATFORM") {
	case "bedrock":
		// AWS credentials and region from the default chain (env, profile, role).
		a = anthropicprovider.NewAgent(
			anthropic.NewClient(bedrock.WithLoadDefaultConfig(ctx)),
			anthropicprovider.AgentConfig{Model: "us.anthropic.%%model=anthropic%%-v1:0"},
		)
	case "vertex":
		// Application Default Credentials; region and project are required.
		a = anthropicprovider.NewAgent(
			anthropic.NewClient(vertex.WithGoogleAuth(ctx, "us-east5", os.Getenv("GOOGLE_CLOUD_PROJECT"))),
			anthropicprovider.AgentConfig{Model: "%%model=anthropic%%"},
		)
	default:
		// Claude deployed in a Microsoft Foundry resource, with Entra ID.
		cred, err := azidentity.NewDefaultAzureCredential(nil)
		if err != nil {
			log.Fatal(err)
		}
		fc, err := foundry.NewClient(foundry.ClientConfig{
			Resource: os.Getenv("ANTHROPIC_FOUNDRY_RESOURCE"),
			AzureADTokenProvider: func(ctx context.Context) (string, error) {
				tok, err := cred.GetToken(ctx, policy.TokenRequestOptions{Scopes: []string{foundry.EntraIDScope}})
				return tok.Token, err
			},
		})
		if err != nil {
			log.Fatal(err)
		}
		// foundry.Client carries the options; anthropic.NewClient turns them into a full client.
		a = anthropicprovider.NewAgent(anthropic.NewClient(fc.Options...),
			anthropicprovider.AgentConfig{Model: "%%model=anthropic%%"}) // the deployment name
	}

	resp, err := a.RunText(ctx, "Hello from Go.").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

All three options switch off the SDK's environment defaults, so a first-party `ANTHROPIC_API_KEY` is never sent to another cloud.

## Google Gemini

`geminiprovider.NewAgent` takes a `*genai.Client` and panics if it is nil. `genai.NewClient` reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` for the Gemini API; for Vertex AI set `Backend: genai.BackendVertexAI` with a project and location, or the `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` variables. History is always local.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/geminiprovider"
	"google.golang.org/genai"
)

type City struct {
	Name    string `json:"name"`
	Country string `json:"country"`
}

func main() {
	ctx := context.Background()

	cfg := &genai.ClientConfig{APIKey: os.Getenv("GEMINI_API_KEY"), Backend: genai.BackendGeminiAPI}
	if project := os.Getenv("GOOGLE_CLOUD_PROJECT"); project != "" && os.Getenv("GEMINI_API_KEY") == "" {
		cfg = &genai.ClientConfig{Backend: genai.BackendVertexAI, Project: project, Location: "europe-west2"}
	}
	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		log.Fatal(err)
	}

	a := geminiprovider.NewAgent(client, geminiprovider.AgentConfig{
		Model:        "%%model=gemini%%",
		Instructions: "You are a geography tutor.",
		Config:       agent.Config{Name: "Gemini"},
	})

	// Native generation settings, including thought summaries.
	settings := geminiprovider.GenerateContentConfig(genai.GenerateContentConfig{
		Temperature:    genai.Ptr[float32](0.3),
		ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true},
	})
	resp, err := a.RunText(ctx, "Why is London foggy?", settings).Collect()
	if err != nil {
		log.Fatal(err)
	}
	for c := range resp.Contents() {
		if r, ok := c.(*message.TextReasoningContent); ok && r.Text != "" {
			fmt.Println("[thought]", r.Text)
		}
	}
	fmt.Println(resp)

	var city City
	if _, err := a.RunText(ctx, "Which city is the capital of Scotland?", agent.WithStructuredOutput(&city)).Collect(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%+v\n", city)
}
```

Instructions are appended to `systemInstruction`. Function tools are sent as one `genai.Tool` holding every function declaration, with each tool's JSON schema in `parametersJsonSchema` and its return schema in `responseJsonSchema`. Hosted tools each become their own `genai.Tool`: `WebSearch` as Google Search, `CodeInterpreter` as code execution, and `FileSearch` with its `HostedVectorStoreContent` inputs as file-search store names. Structured output sets `responseMimeType: application/json` and `responseJsonSchema`. `agent.WithToolMode(tool.ToolModeRequired)` becomes function-calling mode `ANY`, and `tool.RequireTool(name)` also sets `allowedFunctionNames`.

## GitHub Copilot

`copilotprovider.NewAgent` drives the GitHub Copilot CLI through the Copilot Go SDK. Unlike the other providers, Copilot runs its own agent loop: it has built-in tools (shell, file edits), it executes your function tools itself, and it asks for permission through `SessionConfig.OnPermissionRequest` rather than returning `ToolApprovalRequestContent`. The agent's `Name` and `Description` default to "GitHub Copilot Agent" and "An AI agent powered by GitHub Copilot".

```go
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	copilot "github.com/github/copilot-sdk/go"
	"github.com/github/copilot-sdk/go/rpc"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/copilotprovider"
)

func askUser(req copilot.PermissionRequest, _ copilot.PermissionInvocation) (rpc.PermissionDecision, error) {
	fmt.Printf("\nCopilot wants to: %s. Allow? (y/n) ", req.Kind())
	line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	if strings.EqualFold(strings.TrimSpace(line), "y") {
		return &rpc.PermissionDecisionApproveOnce{}, nil
	}
	return &rpc.PermissionDecisionReject{}, nil
}

func main() {
	ctx := context.Background()

	// Uses the bundled Copilot CLI, or COPILOT_CLI_PATH. Sign in with the CLI first.
	client := copilot.NewClient(nil)
	if err := client.Start(ctx); err != nil {
		log.Fatal(err)
	}
	defer func() { _ = client.Stop() }()

	a := copilotprovider.NewAgent(client, copilotprovider.AgentConfig{
		Instructions:  "Keep answers short.",
		SessionConfig: &copilot.SessionConfig{OnPermissionRequest: askUser},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for update, err := range a.RunText(ctx, "List the Go files in this directory.",
		agent.WithSession(session), agent.Stream(true)) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(update)
	}
	fmt.Println()
}
```

The Copilot session ID becomes the agent session's service ID, so a later run with the same `agent.Session` resumes the Copilot session. Function tools are registered as Copilot tools; tools wrapped with `tool.ApprovalRequiredFunc` are gated with a pre-tool-use hook that answers "ask", unless you supply your own `SessionConfig.Hooks.OnPreToolUse`, in which case the framework logs a warning and leaves gating to you. Structured output, tool modes and the `ToolAutoCall` settings do not apply.

## Embeddings

Two providers also produce text embeddings, for the vector-backed memory and retrieval providers or your own use. Both return an `embedding.Generator[string, *embedding.Of[float32]]`:

| Constructor | Notes |
| --- | --- |
| `openaiprovider.NewEmbeddingGenerator(client, EmbeddingGeneratorConfig{Model, Dimensions})` | any OpenAI or Azure OpenAI client; requests base64 encoding by default |
| `foundryprovider.NewEmbeddingGenerator(endpoint, cred, EmbeddingGeneratorConfig{Model, Dimensions, OpenAIOptions})` | a Foundry project or resource endpoint; `Model` (the deployment) must not be blank, and invalid arguments panic |

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/embedding"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	gen := openaiprovider.NewEmbeddingGenerator(openai.NewClient(), openaiprovider.EmbeddingGeneratorConfig{
		Model:      "text-embedding-3-small",
		Dimensions: new(256),
	})

	// A batch: one embedding per input, in input order.
	out, err := gen.Generate(ctx, []string{"tents", "sleeping bags"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	for i, e := range out.Embeddings {
		fmt.Printf("%d: %d dimensions\n", i, e.Dimensions())
	}
	if out.Usage != nil {
		fmt.Println("tokens:", out.Usage.InputTokenCount)
	}

	// One value, as a plain []float32, with a per-request override.
	vec, err := embedding.GenerateVector(ctx, gen, "waterproof jacket",
		&embedding.Options{Dimensions: new(64)})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("single vector:", len(vec))
}
```

`Dimensions: new(256)` uses Go 1.26's `new(expr)`. An empty batch returns an error without making a request. `embedding.Bind(gen)` turns a typed generator into the `*embedding.BoundGenerator` that the vector stores expect; [sessions and memory](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_sessions_and_memory/#vector-stores-and-embeddings) shows it in use.

## Credentials and environment variables

The framework itself reads no credentials. Each vendor client does, and the upstream samples use a few more variables for their own settings:

| Provider | Read by the client | Used by the samples |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `OPENAI_CUSTOM_HEADERS` | `OPENAI_CHAT_MODEL_NAME` |
| Azure OpenAI | none (pass `azure.WithAPIKey` or `azure.WithTokenCredential`); `azidentity` reads the usual `AZURE_*` variables | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| Foundry | `azidentity` credentials | `FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_MODEL`, `FOUNDRY_EMBEDDING_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`; `ANTHROPIC_FOUNDRY_*` for Claude on Foundry | |
| Gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GEMINI_BASE_URL`; Vertex AI: `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | `GEMINI_MODEL` |
| GitHub Copilot | `COPILOT_CLI_PATH` | |

The OpenAI and Foundry adapters add the SDK's identity and a feature-usage bitmask to the `User-Agent` header, but only on approved Azure and Foundry HTTPS hosts. `AGENT_FRAMEWORK_FEATURE_MASK_DISABLED=1` drops the bitmask and `AGENT_FRAMEWORK_USER_AGENT_DISABLED=1` drops the framework's whole contribution.

Retries and timeouts are the vendor clients' business too: see [middleware](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/#retries-and-timeouts).

## Writing your own provider

A provider is an `agent.ProviderConfig` passed to `agent.New`. Only `Run` is required; it receives the prepared messages (history and context already added) and the run options, and returns a stream of `*agent.ResponseUpdate`.

| `ProviderConfig` field | Purpose |
| --- | --- |
| `ProviderName` | returned by `a.ProviderName()` and used in telemetry |
| `Run` | `func(ctx, messages, ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error]`; read options with `agent.GetOption` and `agent.AllOptions` |
| `Middlewares` | run after history and context providers, around every call to `Run`. Add `toolautocall.New(...)` here if your `Run` returns function calls and you want the framework to execute them |
| `ManagesToolExecution` | set when `Run` executes tools itself (as Copilot does), so the agent's function middleware is applied to the tools in the options |
| `Format`, `Unmarshal` | support `agent.WithStructuredOutput`: build a `ResponseFormat` for a Go value, and decode the reply into it. `jsonformat.ForType` and `jsonformat.FromResponseFormat` do the work |
| `CreateSession` | set up provider-specific session state when `CreateSession` is called, or when a run creates a throwaway session |

Treat the message and option slices as read-only; clone before changing them. Set `ConversationID` on an update only if the service now holds the history, because the agent then stops storing it locally.

The [testing page](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_testing/) uses this to build a scripted fake model. A more practical use is composition. The provider below wraps two existing agents and falls back to the second when the first fails before producing any output, which is a common way to ride out a regional outage:

```go
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"os"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolautocall"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// stateless strips the session so an inner agent neither reads nor writes history:
// the outer agent has already added the history to the messages.
func stateless(opts []agent.Option) []agent.Option {
	out := make([]agent.Option, 0, len(opts))
	for _, o := range opts {
		if _, isSession := agent.GetOption([]agent.Option{o}, agent.WithSession); !isSession {
			out = append(out, o)
		}
	}
	return out
}

// fallback tries each agent in turn. Once one has yielded an update, its errors are final.
func fallback(agents ...*agent.Agent) agent.RunFunc {
	return func(ctx context.Context, msgs []*message.Message, opts ...agent.Option) iter.Seq2[*agent.ResponseUpdate, error] {
		return func(yield func(*agent.ResponseUpdate, error) bool) {
			var lastErr error
			for _, a := range agents {
				started := false
				failed := false
				for update, err := range a.Run(ctx, msgs, stateless(opts)...) {
					if err != nil && !started {
						lastErr, failed = fmt.Errorf("%s: %w", a.Name(), err), true
						break
					}
					started = true
					if !yield(update, err) || err != nil {
						return
					}
				}
				if !failed {
					return
				}
			}
			yield(nil, lastErr)
		}
	}
}

func main() {
	// Function tools must not be run twice, so the inner agents only return tool calls
	// and the outer agent's toolautocall middleware executes them.
	noLoop := &toolautocall.Config{MaximumIterationsPerRequest: new(0)}

	primary := openaiprovider.NewChatCompletionsAgent(
		openai.NewClient(option.WithBaseURL("http://127.0.0.1:1/v1/"), option.WithMaxRetries(0)),
		openaiprovider.AgentConfig{Model: "%%model=openai%%", ToolAutoCall: noLoop, Config: agent.Config{Name: "primary"}},
	)
	secondary := openaiprovider.NewChatCompletionsAgent(openai.NewClient(),
		openaiprovider.AgentConfig{Model: "%%model=openai-small%%", ToolAutoCall: noLoop, Config: agent.Config{Name: "secondary"}},
	)

	a := agent.New(agent.ProviderConfig{
		ProviderName: "fallback",
		Run:          fallback(primary, secondary),
		Middlewares:  []agent.Middleware{toolautocall.New(toolautocall.Config{})},
	}, agent.Config{
		Name:       "Resilient",
		RunOptions: []agent.Option{agent.WithInstructions("You are a helpful assistant.")},
	})

	ctx := context.Background()
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "Hello!", agent.WithSession(session)).Collect()
	if err != nil {
		fmt.Fprintln(os.Stderr, "both providers failed:", err)
		os.Exit(1)
	}
	fmt.Println(resp.Messages[0].AuthorName, "->", resp)
}
```

Two details make this work. The inner agents run without a session, so their default in-memory history provider stays out of the way and the outer agent alone stores the conversation. And tool execution happens once, in the outer agent: the inner agents have their tool loop switched off with `MaximumIterationsPerRequest: new(0)`, so they pass function calls straight back. With the primary unreachable, the program prints `secondary -> ...`: an agent stamps its name on updates only when they carry no author yet, so the inner agent's name survives and tells you which provider answered.

## Differences from Python and .NET

- **One agent type.** Python and .NET wrap a chat client (`OpenAIChatClient`, `IChatClient`) in an agent. Go has no public chat-client abstraction: each provider constructor returns an `*agent.Agent`, and `agent.ProviderConfig` is the extension point.
- **Vendor SDKs, not framework clients.** You construct `openai.Client`, `anthropic.Client` and `*genai.Client` yourself, so their options (base URL, retries, HTTP client, cloud platform) are available directly.
- **Native options instead of a common options type.** Temperature, max tokens, seed and similar settings are set through each provider's native-parameters run option, not a shared `ChatOptions`.
- **Not in Go:** Ollama, Bedrock Converse, Mistral, Foundry Local and Copilot Studio providers, the Claude Agent SDK agent, and creating Foundry agents. Python's [providers page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/) lists what it has.

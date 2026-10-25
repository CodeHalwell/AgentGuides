---
title: "Microsoft Agent Framework (Rust) — Providers"
description: "Every chat client and embedding client in agent-framework-rs: OpenAI Chat Completions and Responses, OpenAI-compatible servers, Azure OpenAI and the Entra ID credential chain, Microsoft Foundry, Foundry Local, Anthropic on its own API and on Bedrock, Vertex AI and Foundry, Amazon Bedrock, Gemini, Mistral, Ollama, GitHub Copilot, Copilot Studio, and writing your own ChatClient."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Providers"
  order: 3
---

A provider in `agent-framework` is a type that implements the `ChatClient` trait. `Agent::builder` takes any of them, and so do the workflow engine, the orchestrations and the hosting routers, so changing provider changes one constructor and nothing else. This page goes through every client the crate ships: how to construct it, which environment variables it reads, what it sends on the wire for each `ChatOptions` field, and what it cannot do. The [comprehensive guide](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_comprehensive_guide/) covers the agent side (runs, sessions, structured output, tools), and the [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/) covers function tools and hosted tools in general.

Every program on this page was compiled against `agent-framework` %%version=agent-framework-rs%% with the `full` feature. The OpenAI, Azure OpenAI, Foundry, Anthropic, Gemini, Mistral, Ollama, Foundry Local and Copilot Studio examples were run against local stand-ins that speak each wire format, and the request bodies quoted below are what those stand-ins received. The Bedrock, Claude-on-Bedrock, Claude-on-Vertex and GitHub Copilot examples were only compiled, because their endpoints cannot be redirected to a local server.

## Choosing a provider

| Provider | Type | Feature | Wire API | Auth | Streaming | Structured output | Hosted tools |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI Chat Completions | `OpenAIChatCompletionClient` | `openai` (default) | `POST /chat/completions` | bearer key | SSE | native `response_format` | web search only |
| OpenAI Responses | `OpenAIChatClient` | `openai` | `POST /responses` | bearer key | SSE | native `text.format` | web search, file search, code interpreter, image generation, MCP |
| OpenAI-compatible servers | `OpenAIChatCompletionClient` + `with_base_url` | `openai` | `/chat/completions` | bearer key, often ignored | SSE | depends on server | depends on server |
| Azure OpenAI Chat Completions | `azure::AzureOpenAIClient` | `azure` | `/openai/deployments/{d}/chat/completions` | `api-key` or Entra ID | SSE | native | web search only |
| Azure OpenAI Responses | `azure::AzureOpenAIResponsesClient` | `azure` | `/openai/v1/responses` | `api-key` or Entra ID | SSE | native | as OpenAI Responses |
| Microsoft Foundry project | `FoundryChatClient`, `FoundryAgent` | `foundry` | `{project}/openai/v1/responses` | Entra ID or `api-key` | SSE | native | as OpenAI Responses |
| Foundry Local | `FoundryLocalChatClient` | `foundry-local` | `/chat/completions` | none, or bearer | SSE | depends on model | none |
| Ollama | `OllamaChatClient` | `ollama` | OpenAI-compatible `/v1/chat/completions` | none, or bearer | SSE | `response_format` | none |
| Anthropic | `AnthropicClient` | `anthropic` | `POST /v1/messages` | `x-api-key` | SSE | schema folded into the system prompt | web search, code execution, MCP connector |
| Claude on Bedrock | `AnthropicBedrockClient` | `anthropic` | Bedrock `InvokeModel` | SigV4 | one buffered update | system prompt | as Anthropic, but no beta header |
| Claude on Vertex AI | `AnthropicVertexClient` | `anthropic` | Vertex `rawPredict` | Google bearer token | one buffered update | system prompt | as Anthropic, but no beta header |
| Claude on Foundry | `AnthropicFoundryClient` | `anthropic` (+ `azure` for credentials) | Messages-shaped, path configurable | Entra ID | one buffered update | system prompt | as Anthropic, but no beta header |
| Amazon Bedrock | `BedrockChatClient` | `bedrock` | Converse | SigV4 | one buffered update | not supported | none |
| Google Gemini | `GeminiChatClient` | `gemini` | `generateContent` | `x-goog-api-key` | SSE | native `responseSchema` | Google Search, code execution |
| Mistral | `MistralChatClient` | `mistral` | `/v1/chat/completions` | bearer key | SSE | native `response_format` | none |
| GitHub Copilot | `GitHubCopilotChatClient` | `github-copilot` | `/chat/completions` | GitHub token exchanged for a Copilot token | SSE | `response_format` | none |
| Copilot Studio agent | `CopilotStudioAgent` | `copilotstudio` | Direct-to-Engine | Power Platform bearer token | buffered | n/a (an agent, not a chat client) | n/a |

"One buffered update" means that `get_streaming_response` makes an ordinary request and yields the whole answer as a single update. Code that streams still works, but nothing appears until the model has finished. "Hosted tools" are the `hosted_*()` markers from the core crate. A marker that a provider does not support is skipped with a `tracing` warning, never sent and never an error, so turn on `warn` logging for `agent_framework` while you choose a provider.

A rough guide to choosing:

- **OpenAI or Azure OpenAI, new code:** the Responses clients. They support every hosted tool, return reasoning summaries, and keep the conversation on the service, so a long session does not resend its history.
- **Any server that claims OpenAI compatibility** (vLLM, llama.cpp, LM Studio, LiteLLM, Groq, Together, OpenRouter): `OpenAIChatCompletionClient` with `with_base_url`. Almost none of them implement the Responses API.
- **Microsoft Foundry projects:** `FoundryChatClient`. It is the Azure Responses client with Foundry's URL and audience.
- **Claude:** `AnthropicClient` for Anthropic's own API. Use the Bedrock, Vertex or Foundry transports only when the model must be billed or hosted there, and accept that they do not stream.
- **Local models:** `OllamaChatClient` or `FoundryLocalChatClient`, which need no key.

## Feature flags and imports

Only `openai` is on by default. Each provider crate appears as a module of `agent_framework` when its feature is on, and its main types are added to `agent_framework::prelude`:

| Feature | Module | In the prelude | Notes |
| --- | --- | --- | --- |
| `openai` | `agent_framework::openai` | `OpenAIChatCompletionClient`, `OpenAIChatClient`, `OpenAIEmbeddingClient` | default |
| `azure` | `agent_framework::azure` | `AzureOpenAIClient`, `AzureOpenAIEmbeddingClient`, `StaticTokenCredential`, `TokenCredential` | `AzureOpenAIResponsesClient` and the credential chain need `agent_framework::azure::...` |
| `foundry` | `agent_framework::foundry` | `FoundryAgent`, `FoundryChatClient`, `FoundryEmbeddingClient` | compiles the Azure crate, but its credentials are importable only with `azure` on as well |
| `foundry-local` | `agent_framework::foundry_local` | `FoundryLocalChatClient` | |
| `anthropic` | `agent_framework::anthropic` | `AnthropicClient`, `AnthropicBedrockClient`, `AnthropicVertexClient`, `AnthropicFoundryClient` | the Foundry transport needs `azure` for a credential type |
| `bedrock` | `agent_framework::bedrock` | `BedrockChatClient`, `BedrockEmbeddingClient` | |
| `gemini` | `agent_framework::gemini` | `GeminiChatClient` | `GeminiEmbeddingClient` is in `agent_framework::gemini` |
| `mistral` | `agent_framework::mistral` | `MistralChatClient`, `MistralEmbeddingClient` | |
| `ollama` | `agent_framework::ollama` | `OllamaChatClient`, `OllamaEmbeddingClient` | |
| `github-copilot` | `agent_framework::github_copilot` | `GitHubCopilotChatClient` | |
| `copilotstudio` | `agent_framework::copilotstudio` | `CopilotStudioAgent` | |

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["azure", "foundry", "anthropic"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
futures = "0.3"
async-trait = "0.1"
```

The `agent-framework-azure` crate has one feature the umbrella crate does not forward, `entra-sdk`, described under [Entra ID credentials](#microsoft-entra-id-credentials). To use it, depend on `agent-framework-azure` directly at the same version.

## What every client has in common

All the clients follow the same conventions, so it is worth knowing them once.

**Construction.** Each client has a `new(...)` that takes its settings as arguments and a `from_env(...)` that reads them from environment variables. `from_env` returns `Error::Configuration` naming the missing variable; it never defers the failure to the first request. The model passed to `new` or `from_env` is the default; `ChatOptions::model`, set per run or by the agent builder's `.model(...)`, overrides it on every client. Builder methods (`with_base_url`, `with_max_tokens` and so on) consume and return the client, which is cheap to clone because its state sits behind an `Arc`.

**Options.** Each client maps the typed `ChatOptions` fields it understands onto the wire and silently drops the rest. The table below is what each one sends. `additional_properties` is the escape hatch for everything else: its entries are copied into the top level of the request body, but only for keys the typed fields have not already written, so an `additional_properties` entry can add a field and never replace one.

| `ChatOptions` field | Chat Completions (OpenAI, Azure, Ollama, Foundry Local, Copilot) | Responses (OpenAI, Azure, Foundry) | Anthropic | Gemini | Bedrock | Mistral |
| --- | --- | --- | --- | --- | --- | --- |
| `temperature`, `top_p` | sent | sent | sent | `generationConfig` | `inferenceConfig` | sent |
| `max_tokens` | `max_tokens` | `max_output_tokens` | `max_tokens` (default 1024) | `maxOutputTokens` | `maxTokens` | sent |
| `stop` | sent | dropped | `stop_sequences` | `stopSequences` | `stopSequences` | sent |
| `seed` | sent | dropped | dropped | dropped | dropped | `random_seed` |
| `frequency_penalty`, `presence_penalty` | sent | dropped | dropped | dropped | dropped | sent |
| `logit_bias` | sent | dropped | dropped | dropped | dropped | dropped |
| `user`, `metadata`, `store` | sent | sent | dropped | dropped | dropped | dropped |
| `response_format` | `response_format` | `text.format` | system prompt | `responseMimeType`, `responseSchema` | dropped | sent |
| `tool_choice` | sent | sent | sent | `toolConfig` | `toolChoice` (`None` is ignored) | sent |
| `allow_multiple_tool_calls` | `parallel_tool_calls` when function tools exist | `parallel_tool_calls` when tools exist | `disable_parallel_tool_use`, only with a `tool_choice` | dropped | dropped | `parallel_tool_calls` |
| `conversation_id` | dropped | `previous_response_id` | dropped | dropped | dropped | dropped |
| `instructions` | system message | `instructions` | `system` | `systemInstruction` | `system` | system message |

**Errors.** Every HTTP client classifies a failed response the same way, so `RetryingChatClient` and your own `match` work across providers. 401 and 403 become `Error::ServiceInvalidAuth`; 400 (and on the OpenAI-shaped clients 404 and 422) becomes `Error::ServiceInvalidRequest`, or `Error::ServiceContentFilter` when the body says `content_filter`; everything else, including 408, 429 and 5xx, is `Error::ServiceStatus` carrying the status and, where the server sent one, the `Retry-After` delay in seconds. A connection failure is `Error::Service`. Anthropic reports a content refusal as a successful response with `finish_reason` `content_filter`, not as an error; Azure OpenAI's content-filter errors carry a `ContentFilterDetail` with the policy code and per-category verdicts. The [error handling section](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_comprehensive_guide/#retries-and-errors) of the comprehensive guide shows the retry policy.

**Metadata.** `ChatClient::model()` returns the default model and `service_url()` the endpoint. `ObservableChatClient` records both on its `chat` span (see [observability](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/)), and `Agent::builder` copies `model()` into the agent's default options.

**HTTP.** The clients build their own `reqwest` client with no request timeout. The clients in the `azure` and `foundry` crates send a `User-Agent` of the form `agent-framework-rust/<version> (<crate>)`; the others send `reqwest`'s default. Only `AzureOpenAIResponsesClient::with_http_client` accepts a `reqwest::Client` you configured yourself (proxy, timeouts); for the rest, bound a call with `tokio::time::timeout` or a retry policy.

## OpenAI

The `openai` crate has two chat clients, one per OpenAI API, and an embeddings client. They read the same environment variables:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | all three `from_env` | required; sent as `Authorization: Bearer` |
| `OPENAI_BASE_URL` | all three `from_env` | optional; default `https://api.openai.com/v1` |
| `OPENAI_EMBEDDING_MODEL` | `OpenAIEmbeddingClient::from_env("")` | used when the model argument is empty |

`with_organization(org)` adds the `OpenAI-Organization` header. There is no project header and no Azure mode; Azure has its own clients below.

### Chat Completions

`OpenAIChatCompletionClient` speaks `POST {base_url}/chat/completions`. It converts text, images (`Content::Data` and `Content::Uri` with an `image/*` type become `image_url` parts), audio (`input_audio`, WAV and MP3) and `application/*` data (`file` parts) in user messages, function tools, tool results and assistant tool calls. Streaming always sends `stream_options.include_usage`, so the final update carries token usage just as the buffered call does.

```rust
use agent_framework::prelude::*;
use futures::StreamExt;
use serde_json::json;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    // OPENAI_API_KEY is required; OPENAI_BASE_URL is optional.
    let client = OpenAIChatCompletionClient::from_env("%%model=openai%%")?;

    // Client-level defaults go on the agent; everything is a ChatOptions field.
    let agent = Agent::builder(client)
        .instructions("You are a concise assistant.")
        .temperature(0.2)
        .max_tokens(300)
        .build();

    // Per-run overrides, including fields the builder has no method for.
    let mut chat = ChatOptions::new();
    chat.seed = Some(7);
    chat.stop = Some(vec!["\n\n".into()]);
    chat.user = Some("user-1234".into());
    // Anything the typed fields do not cover goes in additional_properties
    // and is copied into the request body unless a typed field set it already.
    chat.additional_properties
        .insert("reasoning_effort".into(), json!("low"));
    chat.additional_properties.insert("logprobs".into(), json!(true));

    let response = agent
        .run_with_options(
            vec![Message::user("What does the borrow checker do?")],
            None,
            AgentRunOptions::new().with_chat_options(chat),
        )
        .await?;
    println!("{}", response.text());
    println!("finish: {:?}", response.finish_reason);
    if let Some(usage) = &response.usage_details {
        println!(
            "tokens: in={:?} out={:?}",
            usage.input_token_count, usage.output_token_count
        );
    }

    // Streaming sends stream_options.include_usage, so the last update
    // carries token usage just as the buffered call does.
    let mut stream = agent.run_stream_once("Name three Rust editions.").await?;
    while let Some(update) = stream.next().await {
        print!("{}", update?.text());
        std::io::stdout().flush().ok();
    }
    println!();
    Ok(())
}
```

The first request went out as:

```json
{
  "model": "%%model=openai%%",
  "temperature": 0.2,
  "max_tokens": 300,
  "seed": 7,
  "stop": ["\n\n"],
  "user": "user-1234",
  "reasoning_effort": "low",
  "logprobs": true,
  "messages": [ ... ]
}
```

A few behaviours are easy to miss:

- **Reasoning models.** There is no typed reasoning option; pass `reasoning_effort` through `additional_properties` as above. Reasoning text itself is not returned by OpenAI's Chat Completions, but providers on this API that do return a `reasoning_details` array (DeepSeek, OpenRouter, vLLM) get it back as a `Content::TextReasoning` with the payload in `protected_data`, and the client sends it back on the next request of the turn, which those providers require during a tool loop.
- **Refusals.** A model refusal (`message.refusal`, or `delta.refusal` when streaming) becomes a text content marked as a refusal. `Message::text()` leaves it out, so check `finish_reason` and the contents rather than assuming empty text means an empty answer.
- **Log probabilities.** With `logprobs` set, they land in `ChatResponse::additional_properties["logprobs"]`. When streaming, each update carries its own chunk's entries and the final update carries all of them, so the aggregated response matches the buffered one. Upstream Python keeps only the last chunk's.
- **Hosted web search.** `hosted_web_search()` is the only hosted marker Chat Completions supports, and it becomes the top-level `web_search_options` field (with `user_location` if you set one), not an entry in `tools`. It needs a search-capable model. `ToolMode::None` disables it like any other tool. Every other hosted marker is skipped with a warning.
- **`parallel_tool_calls`** is sent only when at least one function tool is present, as upstream does, because the API rejects it otherwise.

### Responses API

`OpenAIChatClient` speaks `POST {base_url}/responses`. Its name follows upstream, where the Responses client became the default `OpenAIChatClient`; the constructors and environment variables are the same as the Chat Completions client. The request shape differs in ways that matter:

- **Instructions** (the agent's instructions plus a leading system message) go in the top-level `instructions` field, and messages become `input` items.
- **Conversation state lives on the service by default.** Each response has an id, which the client returns as `ChatResponse::conversation_id`. The agent stores it on the session as the service session id, and the next turn sends `previous_response_id` plus only the new message. The agent then leaves its history provider out of the run so that nothing is sent twice. Set `ChatOptions::store` to `Some(false)` to opt out: the service keeps nothing, the client returns no conversation id, and the agent goes back to replaying history.
- **Encrypted reasoning.** A reasoning item has to be replayed verbatim in a stateless tool loop, and the service only accepts it with its `encrypted_content`. So whenever a request does not continue a stored response, the client adds `reasoning.encrypted_content` to `include` for you. Your own `include` entries in `additional_properties` are kept alongside it.
- **Reasoning settings** have no typed field: pass `{"effort": ..., "summary": ...}` under the `reasoning` key of `additional_properties`. Reasoning summaries come back as `Content::TextReasoning`.
- **Structured output** becomes `text.format`, the Responses API's flat form of a JSON schema.
- **Dropped options.** `stop`, `seed`, `frequency_penalty`, `presence_penalty` and `logit_bias` have no Responses equivalent and are not sent. Pass a field through `additional_properties` if a model accepts it.
- **Media.** Images become `input_image`, audio `input_audio`, other data `input_file`, and `Content::HostedFile` an `input_file` by id.

```rust
use agent_framework::prelude::*;
use serde::Deserialize;
use serde_json::json;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Answer {
    city: String,
    population_millions: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // The Responses API client: same constructor and variables as Chat Completions.
    let client = OpenAIChatClient::from_env("%%model=openai%%")?;

    // Reasoning settings have no typed field; they ride in additional_properties.
    let mut defaults = ChatOptions::new();
    defaults.additional_properties.insert(
        "reasoning".into(),
        json!({ "effort": "medium", "summary": "auto" }),
    );

    let agent = Agent::builder(client)
        .instructions("You answer geography questions.")
        .chat_options(defaults)
        .build();

    // Turn 1 and 2 share a session. The first reply carries a response id,
    // which the agent stores as the session's service id; the second turn
    // sends it as previous_response_id instead of replaying the history.
    let mut session = agent.create_session();
    let first = agent
        .run(vec![Message::user("Which city is the capital of Japan?")], Some(&mut session))
        .await?;
    println!("1: {}", first.text());
    println!("service id: {:?}", session.service_session_id());

    let second = agent
        .run(vec![Message::user("And roughly how many people live there?")], Some(&mut session))
        .await?;
    println!("2: {}", second.text());

    // Structured output becomes text.format on the Responses wire.
    let schema = json!({
        "type": "object",
        "properties": {
            "city": { "type": "string" },
            "population_millions": { "type": "number" }
        },
        "required": ["city", "population_millions"],
        "additionalProperties": false
    });
    let typed = agent
        .run_with_options(
            vec![Message::user("Give the city and its population.")],
            None,
            AgentRunOptions::new().with_chat_options(
                ChatOptions::new()
                    .with_response_format(ResponseFormat::json_schema("Answer", schema)),
            ),
        )
        .await?;
    let answer: Answer = typed.parse_json()?;
    println!("{answer:?}");

    // store: Some(false) asks the service not to keep the response. The
    // client then returns no conversation id.
    let mut stateless = ChatOptions::new();
    stateless.store = Some(false);
    let reply = agent
        .run_with_options(
            vec![Message::user("One more fact about the city, please.")],
            None,
            AgentRunOptions::new().with_chat_options(stateless),
        )
        .await?;
    println!("stateless: {}", reply.text());
    Ok(())
}
```

The first request carried `"include": ["reasoning.encrypted_content"]`, the `reasoning` object and the instructions; the second carried `"previous_response_id": "resp_1"` and a one-item `input` holding only the new question. The same chaining works without an agent: call `client.get_response(...)`, then set `ChatOptions::conversation_id` to the response's `conversation_id` on the next call, as the library's `openai_responses` example does.

A Responses reply can contain more than text. Hosted-tool citations arrive as `annotations` on text content (URL, title and span); hosted MCP calls as `McpServerToolCall` and `McpServerToolResult` contents; a hosted MCP approval request as a `FunctionApprovalRequest`; generated images as `ImageGenerationToolResult`. A response whose body says `"status": "failed"` despite a 2xx HTTP status is turned into an error (`ServiceContentFilter` when its code is `content_filter`). Code-interpreter output is not parsed into its own content type yet.

### Hosted tools on the Responses API

The hosted markers are plain `ToolDefinition`s with builder methods for their settings. This program prints exactly what the Responses clients (OpenAI, Azure and Foundry share the conversion) send for each:

```rust
use std::collections::HashMap;

use agent_framework::openai::responses::tool_to_responses_spec;
use agent_framework::prelude::*;
use serde_json::json;

fn main() {
    let tools = vec![
        hosted_web_search().user_location(json!({ "city": "London", "country": "GB" })),
        hosted_code_interpreter().file_ids(vec!["file-abc123".into()]),
        hosted_file_search(Some(5)).vector_store_ids(vec!["vs_123".into()]),
        hosted_mcp(
            "docs server",
            "https://learn.microsoft.com/api/mcp",
            Some(vec!["microsoft_docs_search".into()]),
        )
        .description("Microsoft Learn documentation")
        .headers(HashMap::from([("authorization".to_string(), "Bearer t".to_string())]))
        .mcp_approval_mode(McpApprovalMode::Never),
        hosted_image_generation(),
    ];
    // What OpenAIChatClient (and the Azure and Foundry Responses clients,
    // which share this conversion) sends for each marker.
    for tool in &tools {
        println!("{}", tool_to_responses_spec(tool));
    }
}
```

```text
{"type":"web_search","user_location":{"city":"London","country":"GB","type":"approximate"}}
{"container":{"file_ids":["file-abc123"],"type":"auto"},"type":"code_interpreter"}
{"max_num_results":5,"type":"file_search","vector_store_ids":["vs_123"]}
{"allowed_tools":["microsoft_docs_search"],"headers":{"authorization":"Bearer t"},"require_approval":"never","server_description":"Microsoft Learn documentation","server_label":"docs_server","server_url":"https://learn.microsoft.com/api/mcp","type":"mcp"}
{"type":"image_generation"}
```

The details behind that output:

| Marker | Builder methods read by Responses | Notes |
| --- | --- | --- |
| `hosted_web_search()` | `user_location` (keeps `city`, `country`, `region`, `timezone`) | `max_uses` is Anthropic-only and ignored here |
| `hosted_code_interpreter()` | `file_ids`, `container` | a `container` object replaces the default `{"type": "auto"}` and any `file_ids` |
| `hosted_file_search(max_results)` | `vector_store_ids`, `max_results` | the argument wins over `.max_results(n)`; the API requires vector store ids |
| `hosted_mcp(name, url, allowed_tools)` | `description`, `headers`, `mcp_approval_mode` | spaces in the name become underscores in `server_label`; without `mcp_approval_mode`, `.require_approval()` on the definition maps to `"always"` |
| `hosted_image_generation()` | none | any extra keys you put in its `parameters` (size, quality) are passed through |

`McpApprovalMode::PerTool { always, never }` produces the per-tool `require_approval` object. How approval requests come back to you and how to answer them is on the [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/).

### Chat Completions or Responses?

Use the Responses client when you want hosted tools beyond web search, reasoning summaries, or service-side state. Use Chat Completions when you need `seed`, `stop`, penalties or `logit_bias`, when you target an OpenAI-compatible server, or when you want every request to be self-contained (although `store: Some(false)` gives you that on Responses too). The agent code is the same either way, so switching later is a one-line change.

## OpenAI-compatible endpoints, Ollama and Foundry Local

Three clients cover servers that imitate OpenAI's Chat Completions API. All of them reuse the OpenAI crate's conversion, so they support the same content types and options as `OpenAIChatCompletionClient`; whether the server honours an option is up to the server.

| Client | Default base URL | `from_env` reads | Key |
| --- | --- | --- | --- |
| `OpenAIChatCompletionClient::new(key, model).with_base_url(url)` | none (OpenAI's) | `OPENAI_API_KEY` (required), `OPENAI_BASE_URL` | always sent as bearer |
| `OllamaChatClient::new(model)` | `http://localhost:11434/v1` | `OLLAMA_HOST` (optional), never fails | `with_api_key`, for a proxy |
| `FoundryLocalChatClient::new(model)` | `http://localhost:5273/v1` | `FOUNDRY_LOCAL_ENDPOINT`, `FOUNDRY_LOCAL_API_KEY` (both optional) | `with_api_key` |

`OLLAMA_HOST` takes the same form Ollama's own CLI does, `127.0.0.1:11434` or a full URL, and the client appends `/v1`. `with_base_url` on either client takes the OpenAI-compatible root, so include `/v1` yourself. Both expose `base_url()`. Foundry Local picks its port when the service starts and the crate does not discover it; read it from `foundry service status` or the Foundry Local SDK and pass it in.

```rust
use agent_framework::foundry_local::FoundryLocalChatClient;
use agent_framework::ollama::OllamaChatClient;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Any server that speaks /v1/chat/completions: vLLM, llama.cpp,
    // LM Studio, LiteLLM, Groq, Together, OpenRouter and so on.
    let base = std::env::var("COMPAT_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:8000/v1".into());
    let key = std::env::var("COMPAT_API_KEY").unwrap_or_else(|_| "not-needed".into());

    let compat = OpenAIChatCompletionClient::new(key, "my-served-model").with_base_url(&base);

    // Ollama: no key. from_env reads OLLAMA_HOST ("127.0.0.1:11434" or a URL)
    // and appends /v1; with_base_url takes the /v1 root directly.
    let ollama = OllamaChatClient::new("llama3.2").with_base_url(&base);

    // Foundry Local: no key by default; the port is whatever the service reports.
    let local = FoundryLocalChatClient::new("phi-4-mini").with_base_url(&base);

    for (label, client) in [
        ("compatible", Box::new(compat) as Box<dyn ChatClient>),
        ("ollama", Box::new(ollama)),
        ("foundry-local", Box::new(local)),
    ] {
        let agent = Agent::builder(std::sync::Arc::<dyn ChatClient>::from(client))
            .instructions("Answer in one sentence.")
            .build();
        println!("[{label}] {}", agent.run_once("What is Rust?").await?.text());
    }
    Ok(())
}
```

The OpenAI-compatible client sends the key as a bearer token even when it is a placeholder, which local servers ignore. Ollama and Foundry Local send no `Authorization` header unless you call `with_api_key`.

Things that differ from OpenAI itself: tool calling and `response_format` depend on the model the server runs (an Ollama model without tool support answers in text and never calls a tool); usage in streaming needs the server to honour `stream_options.include_usage`; and hosted web search becomes `web_search_options`, which these servers ignore. Ollama is reached through its OpenAI-compatible layer rather than its native `/api/chat`, so native fields such as `keep_alive` or `options.num_ctx` are not available as typed options; pass them in `additional_properties` only if the compatibility layer accepts them. Python's Ollama client uses the native API, so some behaviour differs.

## Azure OpenAI

The `azure` crate has a Chat Completions client, a Responses client, an embeddings client and its own Microsoft Entra ID credential chain. The two chat clients reuse the OpenAI crate's conversion and error classification, so everything in the OpenAI sections above applies; only the URL, the API version and the authentication differ.

| Variable | Read by | Meaning |
| --- | --- | --- |
| `AZURE_OPENAI_ENDPOINT` | all three `from_env` | required; `https://<resource>.openai.azure.com` |
| `AZURE_OPENAI_API_KEY` | all three `from_env` | required by `from_env`, which builds only key-authenticated clients |
| `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` | `AzureOpenAIClient::from_env` | required |
| `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME` | `AzureOpenAIResponsesClient::from_env` | required |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | `AzureOpenAIEmbeddingClient::from_env` | required |
| `AZURE_OPENAI_API_VERSION` | all three | optional; see the defaults below |
| `AZURE_OPENAI_BASE_URL` | Responses only | optional; replaces the derived `/openai/v1` root |

`from_env` covers key authentication only. For Entra ID, construct the client with `with_token_credential(endpoint, deployment, credential)`; the credential is asked for a token before every request and caches it itself. All three clients send the key as an `api-key` header and a token as `Authorization: Bearer`.

### Chat Completions

`AzureOpenAIClient` posts to `{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...`. The deployment in the URL selects the model, and `model()` returns the deployment name. The default API version is `DEFAULT_CHAT_API_VERSION`, `2024-12-01-preview`, the same preview version upstream uses: the last GA version rejects `store`, which the agent can send. `with_api_version` or `AZURE_OPENAI_API_VERSION` pins another; a variable that is set but blank keeps the default, because this route needs the parameter.

```rust
use std::sync::Arc;

use agent_framework::azure::{AzureOpenAIClient, ManagedIdentityCredential};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and
    // AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, plus an optional AZURE_OPENAI_API_VERSION.
    let keyed = AzureOpenAIClient::from_env()?;
    println!("deployment={} api-version={}", keyed.deployment(), keyed.api_version());

    let agent = Agent::builder(keyed)
        .instructions("You are concise.")
        .build();
    println!("{}", agent.run_once("Say hello.").await?.text());

    // Entra ID instead of a key. The credential is asked for a token on every
    // request (it caches and refreshes it itself).
    let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").expect("set above");
    // IMDS by default; IDENTITY_ENDPOINT / IDENTITY_HEADER on App Service.
    let credential = Arc::new(ManagedIdentityCredential::new(
        "https://cognitiveservices.azure.com/.default",
    ));
    let entra = AzureOpenAIClient::with_token_credential(&endpoint, "chat-prod", credential)
        .with_api_version("2025-04-01-preview");
    let agent = Agent::builder(entra).build();
    println!("{}", agent.run_once("Say hello again.").await?.text());
    Ok(())
}
```

Run against a stand-in, the second call first fetched a managed-identity token (`GET ...?api-version=2018-02-01&resource=https://cognitiveservices.azure.com` with `Metadata: true`) and then posted to `/openai/deployments/chat-prod/chat/completions?api-version=2025-04-01-preview` with `Authorization: Bearer <token>`.

Content filtering is where Azure differs most from OpenAI. A filtered prompt fails with `Error::ServiceContentFilter`, whose detail names the policy (`ResponsibleAIPolicyViolation` and so on) and carries each category's `filtered`, `detected` and `severity` values, copied through unchanged so that a category Azure adds later still arrives. A filtered completion is not an error: it ends with `finish_reason` `content_filter`.

### Responses API

`AzureOpenAIResponsesClient` posts to `{endpoint}/openai/v1/responses`. There is no deployment segment in that URL, so the deployment goes in the body's `model` field. The API version defaults to `preview`, as upstream's does. GA v1 resources and many gateways reject any `api-version` parameter, so `without_api_version()` drops it (or set `AZURE_OPENAI_API_VERSION` to an empty string); `with_base_url` replaces the derived `{endpoint}/openai/v1` root for a gateway. Conversation chaining, `store`, encrypted reasoning, hosted tools and output parsing behave exactly as on OpenAI's Responses API, because the same functions do the work.

```rust
use agent_framework::azure::AzureOpenAIResponsesClient;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").expect("AZURE_OPENAI_ENDPOINT");
    let key = std::env::var("AZURE_OPENAI_API_KEY").expect("AZURE_OPENAI_API_KEY");

    // POST {endpoint}/openai/v1/responses?api-version=preview by default.
    // The deployment travels in the body's `model` field.
    let preview = AzureOpenAIResponsesClient::new(&endpoint, "chat-prod", &key);
    println!("api-version: {:?}", preview.api_version());

    // GA v1 resources and many gateways want no api-version at all.
    let ga = AzureOpenAIResponsesClient::new(&endpoint, "chat-prod", &key).without_api_version();
    println!("api-version: {:?}", ga.api_version());

    let agent = Agent::builder(ga)
        .instructions("You are concise.")
        .build();
    let mut session = agent.create_session();
    agent
        .run(vec![Message::user("My name is Ada.")], Some(&mut session))
        .await?;
    // The second turn goes out with previous_response_id, not the history.
    let reply = agent
        .run(vec![Message::user("What is my name?")], Some(&mut session))
        .await?;
    println!("{} (service id {:?})", reply.text(), session.service_session_id());
    Ok(())
}
```

Two methods exist mainly for the Foundry crate, which builds on this client: `without_implicit_encrypted_reasoning()` stops the client adding `reasoning.encrypted_content` to `include`, and `with_http_client(reqwest::Client)` replaces the HTTP client (useful for a proxy or timeouts). The client also forwards per-call `x-client-*` headers from `ChatOptions`, described under [Foundry](#per-run-client-headers).

### Microsoft Entra ID credentials

Every credential implements the `TokenCredential` trait: `get_token()` returns a token for the scope the credential was built with, and `get_token_for_scope(scope)` returns one for another audience. The Azure OpenAI clients call `get_token()`, so build the credential with the scope of the service it is for:

| Service | Scope |
| --- | --- |
| Azure OpenAI, Claude on Foundry, Foundry embeddings | `https://cognitiveservices.azure.com/.default` |
| Foundry project Responses API (`FoundryChatClient`) | `https://ai.azure.com/.default` (`foundry::FOUNDRY_SCOPE`) |

The crate implements these credentials itself, over `reqwest`, with no Azure SDK dependency:

| Credential | Constructor | How it gets a token | Configuration |
| --- | --- | --- | --- |
| `StaticTokenCredential` | `new(token)` | returns the same token | none; for tests, scripts, or tokens refreshed elsewhere |
| `AzureCliCredential` | `new(scope)` | runs `az account get-access-token --scope <scope> --output json` | `with_command("az.cmd")` or an absolute path; a missing binary is `Error::Configuration` |
| `ClientSecretCredential` | `new(tenant, client_id, secret, scope)` | OAuth client-credentials grant at `{authority}/{tenant}/oauth2/v2.0/token` | `with_authority(url)` for sovereign clouds; default `DEFAULT_AUTHORITY` (`https://login.microsoftonline.com`) |
| `EnvironmentCredential` | `new(scope)` returns `Result` | as `ClientSecretCredential` | `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`; fails at construction naming every missing variable |
| `ManagedIdentityCredential` | `new(scope)` | `GET` on the managed-identity endpoint with `Metadata: true` | endpoint from `IDENTITY_ENDPOINT`, then `MSI_ENDPOINT`, then `DEFAULT_IMDS_ENDPOINT` (`http://169.254.169.254/metadata/identity/oauth2/token`); `IDENTITY_HEADER` sent as `X-IDENTITY-HEADER`; `with_client_id` for a user-assigned identity; `with_endpoint`, `with_identity_header` |
| `WorkloadIdentityCredential` | `new(scope)` returns `Result` | exchanges the Kubernetes service-account token as a client assertion | `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_FEDERATED_TOKEN_FILE`; `with_overrides(scope, tenant, client, file)`, `with_client_id(scope, id)`, `with_authority` |
| `ChainedTokenCredential` | `new(vec![...])` | tries each in order | none |
| `DefaultAzureCredential` | `new(scope)` | a prebuilt chain | see below |

Each credential caches tokens per scope and fetches a new one `REFRESH_SKEW` (two minutes) before the token expires. A token response without an expiry is assumed to last an hour, except an Azure CLI token without a machine-readable `expires_on` (older `az` versions), which is trusted for only five minutes. The managed-identity request has a two-second connect timeout and a ten-second total timeout, so on a laptop it fails fast and a chain moves on; it always sends `api-version=2018-02-01` and derives the `resource` parameter by removing `/.default` from the scope. The workload-identity credential re-reads the token file on every token request, because the kubelet rotates it.

`ChainedTokenCredential` returns the first token it gets and remembers which credential produced it, so later calls try that one first. If every credential fails, the error lists each one's failure in order. `DefaultAzureCredential` is a chain of `EnvironmentCredential`, `WorkloadIdentityCredential`, `ManagedIdentityCredential` and `AzureCliCredential`, in that order, which is the order Azure Identity's own default chain uses for those four. The first two are included only when their environment variables are all set; the last two are always present. The Visual Studio Code, Azure PowerShell, Azure Developer CLI, shared token cache and interactive browser credentials of Azure Identity are not implemented, and neither are certificate or client-assertion credentials.

```rust
use std::sync::Arc;

use agent_framework::azure::{
    AzureCliCredential, ChainedTokenCredential, ClientSecretCredential, DefaultAzureCredential,
    ManagedIdentityCredential, TokenCredential, DEFAULT_AUTHORITY,
};
use agent_framework::prelude::*;
use async_trait::async_trait;

const OPENAI_SCOPE: &str = "https://cognitiveservices.azure.com/.default";
const FOUNDRY_SCOPE: &str = "https://ai.azure.com/.default";

/// A credential of your own: here, a token another process keeps fresh in a file.
struct TokenFile(std::path::PathBuf);

#[async_trait]
impl TokenCredential for TokenFile {
    async fn get_token(&self) -> Result<String> {
        std::fs::read_to_string(&self.0)
            .map(|t| t.trim().to_string())
            .map_err(|e| Error::Configuration(format!("token file: {e}")))
    }
    // get_token_for_scope defaults to get_token, which is right for a fixed token.
}

#[tokio::main]
async fn main() -> Result<()> {
    // A service principal. with_authority is for sovereign clouds
    // (https://login.microsoftonline.us, https://login.chinacloudapi.cn).
    let authority =
        std::env::var("ENTRA_AUTHORITY").unwrap_or_else(|_| DEFAULT_AUTHORITY.to_string());
    let service_principal = ClientSecretCredential::new(
        std::env::var("SP_TENANT_ID").unwrap_or_default(),
        std::env::var("SP_CLIENT_ID").unwrap_or_default(),
        std::env::var("SP_CLIENT_SECRET").unwrap_or_default(),
        OPENAI_SCOPE,
    )
    .with_authority(authority);

    // Your own order: file first, then the service principal, then a
    // user-assigned managed identity, then the developer's `az login`.
    let chain = ChainedTokenCredential::new(vec![
        Arc::new(TokenFile("/run/secrets/aoai-token".into())),
        Arc::new(service_principal),
        Arc::new(ManagedIdentityCredential::new(OPENAI_SCOPE).with_client_id("00000000-uami")),
        Arc::new(AzureCliCredential::new(OPENAI_SCOPE)),
    ]);
    // The first credential that succeeds is remembered and tried first next time.
    println!("chain: {}", chain.get_token().await?);
    println!("chain again: {}", chain.get_token().await?);

    // The prebuilt chain: environment, workload identity, managed identity, CLI.
    let default = DefaultAzureCredential::new(OPENAI_SCOPE);
    // One credential, two audiences: each scope is fetched and cached separately.
    println!("openai scope: {}", default.get_token().await?);
    println!("foundry scope: {}", default.get_token_for_scope(FOUNDRY_SCOPE).await?);
    Ok(())
}
```

Against a stand-in token endpoint, the file credential failed (no such file), the service principal succeeded, and the second `get_token` was answered from that credential's cache without a request. `DefaultAzureCredential` then made one managed-identity request per scope.

Three gotchas. A `StaticTokenCredential` ignores the scope, so the token you give it must already be for the right audience; a Foundry token sent to Azure OpenAI is rejected with 401. `DefaultAzureCredential` on a developer machine without `az login` fails only after the managed-identity attempt times out, a couple of seconds. And `EnvironmentCredential` with two of its three variables set is silently left out of the default chain, which then falls through to the CLI; construct it directly if you want the misconfiguration reported.

**`entra-sdk`.** For credentials this crate does not implement (client certificates, client assertions, Azure Pipelines, the Azure Developer CLI), enable the `entra-sdk` feature of `agent-framework-azure`. It adds `SdkTokenCredential`, which wraps any credential from the official `azure_identity` crate: `SdkTokenCredential::new(inner, scope)`, or the shortcuts `SdkTokenCredential::azure_openai(inner)` and `SdkTokenCredential::foundry(inner)`, with the scope constants `AZURE_OPENAI_SCOPE` and `FOUNDRY_SCOPE`. The Azure SDK uses `reqwest` 0.13 with `aws-lc-rs`, so a build with this feature needs CMake and a C compiler. The umbrella crate does not forward the feature, and this page's examples do not use it.

## Microsoft Foundry

The `foundry` crate targets a Microsoft Foundry (formerly Azure AI Foundry) project through its Responses API, `{project endpoint}/openai/v1/responses`. It does not use the old Azure AI Agents threads-and-runs API, which upstream has removed.

### FoundryChatClient

`FoundryChatClient` is a thin wrapper around `AzureOpenAIResponsesClient` with three Foundry settings applied: no `api-version` parameter (the route is path-versioned), no implicit `reasoning.encrypted_content` (Foundry wants it only when asked for), and Foundry's own `User-Agent`. Everything else, including streaming, hosted tools, conversation chaining and errors, is the Azure Responses client's behaviour.

| Constructor | Auth |
| --- | --- |
| `FoundryChatClient::with_token_credential(endpoint, model, credential)` | Entra ID; build the credential with `FOUNDRY_SCOPE` |
| `FoundryChatClient::new(endpoint, model, api_key)` | `api-key` header |
| `FoundryChatClient::from_env()` | `FOUNDRY_ENDPOINT` or `FOUNDRY_PROJECT_ENDPOINT` (required), `FOUNDRY_MODEL` (required), `FOUNDRY_API_KEY` (if set, key auth; otherwise `DefaultAzureCredential::new(FOUNDRY_SCOPE)`) |

The endpoint is the project endpoint, `https://<resource>.services.ai.azure.com/api/projects/<project>`, and the model is the deployment name. `with_base_url` overrides the derived route; `endpoint()` and `model_name()` return the configured values. These `FOUNDRY_*` names replace the older `AZURE_AI_PROJECT_ENDPOINT` and `AZURE_AI_MODEL_DEPLOYMENT_NAME`, as upstream's do.

### FoundryAgent and PromptAgentDefinition

`FoundryAgent` describes a Foundry prompt agent (a name, a model, instructions, a description and tools) and runs it through a core `Agent` over a `FoundryChatClient`. The definition is a serialisable `PromptAgentDefinition`; `to_prompt_agent()` returns it and `FoundryAgent::from_definition(client, definition)` builds an agent from one, so a definition can be stored as JSON and loaded later. Tools in a deserialised definition have no executor, because a definition describes tools rather than carrying their code.

The builder takes `name` (default: the model name, which also becomes the agent id), `model` (default: the client's), `instructions`, `description`, `tool` and `tools`. `FoundryAgent` implements `SupportsAgentRun`, including streaming and `run_with_options`; `inner()` returns the core `Agent` for anything else, such as `as_tool`.

The agent runs entirely on the client side. It does not create, fetch or bind to an agent stored in the Foundry project (`AIProjectClient.agents` upstream, or .NET's `AsAIAgent`), and hosted Foundry agents are not supported. If your agent already exists in the portal, copy its instructions and tools into a definition.

### Per-run client headers

The Foundry platform forwards request headers whose names start with `x-client-` into the agent container, which is how a run says on whose behalf it is made (`x-client-end-user-id`). The `FoundryClientHeaders` trait adds `with_client_header(name, value)`, `with_client_headers(pairs)` and `client_headers()` to `ChatOptions`. Names are checked when you add them: a name outside the `x-client-` prefix, an empty name or value, or a NUL, CR or LF is an `Error::Configuration`, and `with_client_headers` applies all pairs or none. A second header with the same name, compared case-insensitively, replaces the first.

The headers travel on `ChatOptions` rather than the client because one client usually serves many users. That has a consequence: per-run options replace the agent's header set rather than adding to it, because `ChatOptions::merge` takes the per-run value of each `additional_properties` key whole. Put every header a run needs on that run's options. In .NET the same feature needs an agent decorator and a transport policy and is a silent no-op without them; here a stamped header is always sent and an invalid one is an error.

```rust
use std::sync::Arc;

use agent_framework::azure::StaticTokenCredential;
use agent_framework::foundry::FoundryClientHeaders;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // https://<resource>.services.ai.azure.com/api/projects/<project>
    let endpoint = std::env::var("FOUNDRY_PROJECT_ENDPOINT").expect("FOUNDRY_PROJECT_ENDPOINT");
    // In production: AzureCliCredential::new(FOUNDRY_SCOPE) or
    // DefaultAzureCredential::new(FOUNDRY_SCOPE): the scope must be the Foundry one.
    let credential = Arc::new(StaticTokenCredential::new(
        std::env::var("FOUNDRY_TOKEN").unwrap_or_default(),
    ));
    let client = FoundryChatClient::with_token_credential(&endpoint, "chat-prod", credential);

    // A Prompt Agent defined client-side and run over the Responses API.
    let agent = FoundryAgent::builder(client)
        .name("support-agent")
        .description("Answers product questions")
        .instructions("You answer questions about our products. Be brief.")
        .build();

    // The definition is serialisable, so it can be stored or diffed.
    println!("{}", serde_json::to_string_pretty(&agent.to_prompt_agent())?);

    // Per-run x-client-* headers, forwarded by the Foundry platform into the
    // agent container. Here: who the run is on behalf of.
    let options = ChatOptions::new()
        .with_client_header("x-client-end-user-id", "user-42")?
        .with_client_header("x-client-request-id", "req-7f3c")?;
    let response = agent
        .run_with_options(
            vec![Message::user("What is your returns policy?")],
            None,
            AgentRunOptions::new().with_chat_options(options),
        )
        .await?;
    println!("{}", response.text());

    // A header outside the x-client- namespace is refused before any request.
    let err = ChatOptions::new().with_client_header("authorization", "nope").unwrap_err();
    println!("refused: {err}");
    Ok(())
}
```

```text
{
  "name": "support-agent",
  "model": "chat-prod",
  "instructions": "You answer questions about our products. Be brief.",
  "description": "Answers product questions"
}
Mock reply to: What is your returns policy?
refused: configuration error: client header name 'authorization' must start with 'x-client-' (case-insensitive): ...
```

The request went to `/api/projects/<project>/openai/v1/responses` with no query string, `Authorization: Bearer <token>`, both `x-client-*` headers and no `include` field.

### Web search and Bing grounding

`hosted_web_search()` on a Foundry client becomes the Responses API's keyless `{"type": "web_search"}` tool, the same as on OpenAI, and needs nothing configured in the project. The `ToolDefinition::connection_id(...)` and `custom_connection(id, instance)` builders exist for the old Agents API's `bing_grounding` and `bing_custom_search` tools, but in %%version=agent-framework-rs%% nothing reads them: they are stored on the definition and not sent. Bing grounding through a project connection is therefore not available. The library's `azure_foundry_bing_grounding` example shows the keyless search.

Foundry's other surfaces are covered elsewhere: `FoundryMemoryProvider` (Foundry memory stores as a context provider) on the [context and memory page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/), and `FoundryEmbeddingClient` under [embeddings](#embeddings) below.

## Anthropic

### AnthropicClient

`AnthropicClient` speaks the Messages API, `POST {base_url}/v1/messages`, with its own conversion and SSE parser rather than Anthropic's SDK.

| Item | Value |
| --- | --- |
| `new(api_key, model)`, `from_env(model)` | `ANTHROPIC_API_KEY` (required), `ANTHROPIC_BASE_URL` (optional, default `https://api.anthropic.com`, without `/v1`) |
| Headers | `x-api-key`, `anthropic-version: 2023-06-01`, `anthropic-beta` |
| `with_max_tokens(n)` | the `max_tokens` sent when `ChatOptions::max_tokens` is unset; default 1024, because the API requires the field |
| `with_default_options(options)` | client-level `ChatOptions` under each request's (request values win) |
| `with_additional_beta_flags(flags)` | added to the default beta flags on every request |

Anthropic has no system or tool role, so the instructions and a leading system message become the top-level `system` field and everything that is not an assistant message is sent as a user turn. Images (`Content::Data` as base64, `Content::Uri` as a URL source) are converted; other media types, including PDFs, are not. Extended thinking comes back as `Content::TextReasoning` with its signature, and the client replays thinking blocks verbatim on the next request, as the API requires during a tool loop. It drops unsigned reasoning content, such as reasoning carried over from another provider, because the API rejects it; if you move one conversation between providers, drop reasoning content at the boundary. A `stop_reason` of `refusal` becomes `finish_reason` `content_filter`.

The `anthropic-beta` header always carries `mcp-client-2025-04-04` and `code-execution-2025-08-25`, as upstream's does, plus the client's extra flags and any listed under `additional_properties["additional_beta_flags"]` for one request (that key is removed from the body). The Messages API has no structured-output field, so `ResponseFormat::JsonSchema` and `JsonObject` are turned into an instruction appended to the system prompt, with the schema pretty-printed for `JsonSchema`. That is weaker than native structured output: the model usually complies but nothing enforces the schema, so handle a `parse_json` failure. Upstream Python and .NET drop `response_format` for Anthropic altogether.

```rust
use agent_framework::anthropic::AnthropicClient;
use agent_framework::prelude::*;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Verdict {
    language: String,
    memory_safe: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // ANTHROPIC_API_KEY is required; ANTHROPIC_BASE_URL is optional.
    let client = AnthropicClient::from_env("%%model=anthropic%%")?
        // Sent whenever ChatOptions::max_tokens is unset (the API requires it).
        .with_max_tokens(4096)
        // Added to the default anthropic-beta flags on every request.
        .with_additional_beta_flags(["context-1m-2025-08-07"]);

    let agent = Agent::builder(client.clone())
        .instructions("You are a terse code reviewer.")
        .build();
    println!("{}", agent.run_once("Is unsafe Rust ever justified?").await?.text());

    // Streaming is real SSE, with usage folded into the updates.
    let mut stream = agent.run_stream_once("List two Rust lints.").await?;
    while let Some(update) = stream.next().await {
        print!("{}", update?.text());
    }
    println!();

    // Structured output: the schema is appended to the system prompt,
    // because the Messages API has no response_format field.
    let schema = json!({
        "type": "object",
        "properties": {
            "language": { "type": "string" },
            "memory_safe": { "type": "boolean" }
        },
        "required": ["language", "memory_safe"]
    });
    let typed = Agent::builder(client.clone())
        .response_format(ResponseFormat::json_schema("Verdict", schema))
        .build()
        .run_once("Classify the Rust language.")
        .await?;
    let verdict: Verdict = typed.parse_json()?;
    println!("{verdict:?}");

    // Extended thinking has no typed option; pass the API's own field.
    let mut thinking = ChatOptions::new();
    thinking.max_tokens = Some(8000);
    thinking.additional_properties.insert(
        "thinking".into(),
        json!({ "type": "enabled", "budget_tokens": 4000 }),
    );
    let reply = agent
        .run_with_options(
            vec![Message::user("Why does Rust have lifetimes?")],
            None,
            AgentRunOptions::new().with_chat_options(thinking),
        )
        .await?;
    println!("{}", reply.text());
    Ok(())
}
```

The requests carried `anthropic-beta: code-execution-2025-08-25,context-1m-2025-08-07,mcp-client-2025-04-04`. The structured-output request had no schema field; its `system` was the agent's instructions (none here) followed by "Respond only with a single valid JSON object that conforms exactly to the following JSON Schema (named "Verdict")..." and the schema. The thinking request carried `"max_tokens": 8000` and the `thinking` object. The API requires `budget_tokens` to be less than `max_tokens` and rejects `temperature` alongside thinking, which the client does not check.

`seed`, the penalties, `logit_bias`, `user`, `metadata` and `store` are not sent. Tool choice maps to Anthropic's `auto`, `any`, `tool` and `none`; `allow_multiple_tool_calls: Some(false)` becomes `disable_parallel_tool_use: true`, but only when a `tool_choice` is also set, because that flag lives inside the `tool_choice` object. Function tools are sent as `{"type": "custom", ...}` with an `input_schema`.

### Hosted tools on Anthropic

| Marker | Sent as | Builder methods read |
| --- | --- | --- |
| `hosted_web_search()` | `{"type": "web_search_20250305", "name": "web_search"}` | `max_uses`, `user_location` |
| `hosted_code_interpreter()` | `{"type": "code_execution_20250825", "name": "code_execution"}` | none |
| `hosted_mcp(name, url, allowed_tools)` | an entry in the top-level `mcp_servers` array (Anthropic's MCP connector) | `headers`, of which only `authorization` is used, as `authorization_token` |
| `hosted_file_search`, `hosted_image_generation` | skipped with a warning | |

`mcp_approval_mode` is not read: Anthropic's connector has no per-tool approval. Search results come back as citation annotations on the text, which the library's `anthropic_hosted_tools` example prints; server tool calls appear as function-call contents with no local executor, which the function-invocation loop leaves alone.

### Claude on Bedrock, Vertex AI and Foundry

Three more clients send the same Messages-shaped body through other clouds. Each puts the model in the URL and an `anthropic_version` in the body instead of a top-level `model`, and each is a transport over the same conversion, so everything above about content, options, structured output and hosted tools applies, with three differences: none sends the `anthropic-beta` header (so MCP connector and code-execution tools may be refused), none has `with_default_options` or `with_additional_beta_flags`, and none truly streams. Their `get_streaming_response` makes one ordinary request and returns it as a single update; the AWS event-stream framing and Vertex's `streamRawPredict` are not implemented.

| Client | Constructor | `from_env` | Endpoint and auth | Other builders |
| --- | --- | --- | --- | --- |
| `AnthropicBedrockClient` | `new(access_key_id, secret_access_key, region, model)` | `from_env(model)`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (required), `AWS_SESSION_TOKEN`, region from `AWS_REGION`, then `AWS_DEFAULT_REGION`, then `us-east-1` | `https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke`, SigV4; body version `bedrock-2023-05-31` | `with_session_token`, `with_region`, `with_max_tokens`; `region()` |
| `AnthropicVertexClient` | `new(project_id, region, model, token_provider)` | `from_env(model, token_provider)`: `GOOGLE_CLOUD_PROJECT` or `ANTHROPIC_VERTEX_PROJECT_ID` (required), `CLOUD_ML_REGION` or `ANTHROPIC_VERTEX_REGION` (default `us-east5`) | `https://{region}-aiplatform.googleapis.com/.../publishers/anthropic/models/{model}:rawPredict`, `Authorization: Bearer`; body version `vertex-2023-10-16` | `with_max_tokens`; `project_id()`, `region()` |
| `AnthropicFoundryClient` | `with_token_credential(base_url, model, credential)` | none | `{base_url}{path}`, default path `/v1/messages`, Entra ID bearer for `DEFAULT_SCOPE` (`https://cognitiveservices.azure.com/.default`); body version `foundry-2025-01-01` | `with_scope`, `with_path`, `with_anthropic_version`, `with_max_tokens` |

The Bedrock transport only reads static keys: there is no AWS profile, SSO, instance-metadata or web-identity support, so on EC2, ECS or Lambda you fetch credentials yourself and pass them with `new(...)` and `with_session_token`. Vertex has no Google credential discovery either; `VertexTokenProvider` is a synchronous trait with one method, `access_token()`, called once per request, and `StaticVertexToken::new(token)` wraps a token from `gcloud auth print-access-token`. Implement the trait yourself to refresh tokens (keep the refresh outside the call, since it cannot await). The Foundry transport's default path and version are not fixed by any published Azure contract, so confirm both against your deployment's documentation and change them with `with_path` and `with_anthropic_version` if needed.

```rust
use std::collections::HashMap;
use std::sync::Arc;

use agent_framework::anthropic::convert::tools_to_anthropic;
use agent_framework::anthropic::{
    AnthropicBedrockClient, AnthropicFoundryClient, AnthropicVertexClient, StaticVertexToken,
};
use agent_framework::azure::StaticTokenCredential;
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // How Anthropic's converter maps each hosted marker.
    let tools = vec![
        hosted_web_search()
            .max_uses(3)
            .user_location(json!({ "type": "approximate", "city": "London", "country": "GB" })),
        hosted_code_interpreter(),
        hosted_mcp("docs", "https://example.com/mcp", Some(vec!["search".into()]))
            .headers(HashMap::from([("authorization".to_string(), "Bearer t".to_string())])),
        hosted_file_search(None), // not supported: skipped with a warning
    ];
    let (tool_specs, mcp_servers) = tools_to_anthropic(&tools);
    println!("tools: {}", json!(tool_specs));
    println!("mcp_servers: {}", json!(mcp_servers));

    // Claude on Azure AI Foundry: Entra ID bearer token, Messages-shaped body.
    let base = std::env::var("FOUNDRY_ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://my-res.services.ai.azure.com/anthropic".into());
    let foundry = AnthropicFoundryClient::with_token_credential(
        &base,
        "%%model=anthropic%%",
        Arc::new(StaticTokenCredential::new("<entra token>")),
    )
    .with_max_tokens(2048);
    let agent = Agent::builder(foundry).instructions("Be brief.").build();
    println!("[foundry] {}", agent.run_once("Say hello.").await?.text());

    // Claude on Bedrock (InvokeModel, SigV4) and on Vertex AI (rawPredict).
    // Constructed only: they need real AWS and Google credentials.
    let _bedrock = AnthropicBedrockClient::new(
        "AKIA...",
        "secret",
        "eu-west-2",
        "us.anthropic.%%model=anthropic%%",
    )
    .with_session_token("sts-token")
    .with_max_tokens(2048);
    let _vertex = AnthropicVertexClient::new(
        "my-gcp-project",
        "us-east5",
        "%%model=anthropic%%",
        Arc::new(StaticVertexToken::new("ya29....")),
    );
    Ok(())
}
```

```text
tools: [{"max_uses":3,"name":"web_search","type":"web_search_20250305","user_location":{"city":"London","country":"GB","type":"approximate"}},{"name":"code_execution","type":"code_execution_20250825"}]
mcp_servers: [{"authorization_token":"Bearer t","name":"docs","tool_configuration":{"allowed_tools":["search"]},"type":"url","url":"https://example.com/mcp"}]
[foundry] Mock reply to: Say hello.
```

The Foundry request carried `"anthropic_version": "foundry-2025-01-01"`, `"max_tokens": 2048` and a bearer token, and no `model`, `anthropic-version` or `anthropic-beta` header. The library's `anthropic_multicloud` example runs the Bedrock and Vertex transports when their credentials are present.

## Amazon Bedrock

`BedrockChatClient` speaks Bedrock's model-agnostic Converse API, `POST https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse`, signed with SigV4 by the crate's own implementation (checked against AWS's published test vector), so it needs no AWS SDK. It works with any model Converse supports, Claude included; `AnthropicBedrockClient` above is the alternative for Claude-specific features.

| Item | Value |
| --- | --- |
| `new(access_key_id, secret_access_key, region, model)` | static credentials |
| `from_env(model)` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (required), `AWS_SESSION_TOKEN` (if set and not blank), region from `AWS_REGION`, then `AWS_DEFAULT_REGION`, then `us-east-1` |
| Builders | `with_region` (re-derives the host), `with_session_token`; `model()`, `region()` |
| Errors | as the other clients, with the `x-amzn-errortype` header in the message |

Like the Claude transports, it reads no AWS profile, SSO, instance-metadata or web-identity credentials, and it cannot be pointed at a different endpoint (a VPC endpoint or a test server); `BedrockEmbeddingClient` has `with_endpoint` for that, the chat client does not.

What it sends: system messages from any position, plus the instructions, as `system` blocks; turns merged so that user and assistant alternate, as Converse requires; `temperature`, `max_tokens`, `top_p` and `stop` in `inferenceConfig`; function tools as `toolSpec`s in `toolConfig`. Images are sent as inline bytes in user turns only; `Content::Uri` images are dropped, because Converse takes bytes or S3 locations, not URLs. Reasoning (`reasoningContent`) is parsed and replayed with its signature. A `guardrail_intervened` or `content_filtered` stop reason becomes `finish_reason` `content_filter`.

What it does not do: there is no structured output (`response_format` is dropped; Converse has no such field), no hosted tools, and no true streaming (one buffered update, because `ConverseStream` uses AWS's binary event-stream framing). `ToolMode::None` leaves the tools in the request with Converse's default `auto`, because Converse has no "none" choice; to stop tool use, remove the tools. With a Prompt Management ARN as the model id (one containing `:prompt/`), the client drops `inferenceConfig`, `system`, `toolConfig` and `additionalModelRequestFields` with a warning, because Converse rejects them alongside a managed prompt. It also drops `guardrailConfig.streamProcessingMode`, which only `ConverseStream` accepts.

```rust
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required; AWS_SESSION_TOKEN
    // is read when set; the region is AWS_REGION, then AWS_DEFAULT_REGION, then us-east-1.
    let client = BedrockChatClient::from_env("us.anthropic.%%model=anthropic%%")?;
    println!("region {}, model {}", client.region(), client.model());

    // Or explicit credentials, e.g. from your own STS AssumeRole call.
    let _explicit = BedrockChatClient::new("AKIA...", "secret", "eu-west-1", "amazon.nova-pro-v1:0")
        .with_session_token("sts-session-token");

    // Converse fields with no ChatOptions equivalent pass through as body keys.
    let mut options = ChatOptions::new().with_max_tokens(1024);
    options.additional_properties.insert(
        "guardrailConfig".into(),
        json!({ "guardrailIdentifier": "gr-abc123", "guardrailVersion": "1" }),
    );
    options.additional_properties.insert(
        "additionalModelRequestFields".into(),
        json!({ "thinking": { "type": "enabled", "budget_tokens": 2048 } }),
    );

    let agent = Agent::builder(client)
        .instructions("You are concise.")
        .build();
    let response = agent
        .run_with_options(
            vec![Message::user("What is the capital of Japan?")],
            None,
            AgentRunOptions::new().with_chat_options(options),
        )
        .await?;
    println!("{}", response.text());
    Ok(())
}
```

This example was compiled but not run. Without credentials, `from_env` returns `Error::Configuration("AWS_ACCESS_KEY_ID is not set")`. Python's Bedrock client goes through boto3 and so inherits its whole credential chain; that is the main practical difference.

## Google Gemini

`GeminiChatClient` speaks the Gemini API's `generateContent` (and `streamGenerateContent?alt=sse` for streaming) under `v1beta`, with the key in the `x-goog-api-key` header rather than the URL.

| Item | Value |
| --- | --- |
| `new(api_key, model)`, `from_env(model)` | `GEMINI_API_KEY`, else `GOOGLE_API_KEY` (one required); `GEMINI_BASE_URL` optional, default `https://generativelanguage.googleapis.com`, without `/v1beta` |
| `with_base_url`, `with_default_options` | as on Anthropic |

It is the Gemini Developer API only: there is no Vertex AI mode for Gemini models and no Google credential support, so a Vertex-hosted Gemini needs an API-key-compatible proxy or a custom client.

The mapping: instructions and every system message, wherever it is, become `systemInstruction`; assistant becomes `model` and tool results go in `function` turns; `temperature`, `max_tokens`, `top_p`, `stop` and the response format go in `generationConfig`; function tools are gathered into one `functionDeclarations` entry; `ToolMode` maps to `toolConfig.functionCallingConfig` (`AUTO`, `ANY` with `allowedFunctionNames`, `NONE`). Structured output is native: `JsonObject` sets `responseMimeType: application/json` and `JsonSchema` adds `responseSchema`. Gemini accepts a subset of JSON Schema there, so keep schemas simple. `hosted_web_search()` becomes `{"googleSearch": {}}` and `hosted_code_interpreter()` `{"codeExecution": {}}`; the other markers are skipped. Images and other media go as `inlineData` (base64 data URIs) or `fileData` (URIs). Thought summaries come back as reasoning content, and their `thoughtSignature` is kept and replayed, as Gemini requires for multi-turn tool use. `seed`, the penalties, `user` and the rest are dropped.

```rust
use agent_framework::gemini::GeminiChatClient;
use agent_framework::prelude::*;
use serde::Deserialize;
use serde_json::json;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct City {
    name: String,
    country: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // GEMINI_API_KEY, else GOOGLE_API_KEY; optional GEMINI_BASE_URL (no /v1beta).
    let mut defaults = ChatOptions::new();
    // Gemini-specific fields pass through additional_properties into the body.
    defaults.additional_properties.insert(
        "safetySettings".into(),
        json!([{ "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH" }]),
    );
    let client = GeminiChatClient::from_env("%%model=gemini%%")?.with_default_options(defaults);

    let agent = Agent::builder(client.clone())
        .instructions("You are a concise travel guide.")
        .build();
    println!("{}", agent.run_once("Suggest a city for a weekend in May.").await?.text());

    // Structured output is native: responseMimeType + responseSchema.
    let schema = json!({
        "type": "object",
        "properties": { "name": { "type": "string" }, "country": { "type": "string" } },
        "required": ["name", "country"]
    });
    let typed = Agent::builder(client.clone())
        .response_format(ResponseFormat::json_schema("City", schema))
        .build()
        .run_once("Pick one European capital.")
        .await?;
    let city: City = typed.parse_json()?;
    println!("{city:?}");

    // Thinking settings live inside generationConfig. Put them there whole:
    // a typed option such as temperature makes the client write its own
    // generationConfig, and an additional_properties entry under the same
    // key would then be ignored.
    let mut thinking = ChatOptions::new();
    thinking.additional_properties.insert(
        "generationConfig".into(),
        json!({ "temperature": 0.3, "thinkingConfig": { "thinkingBudget": 1024 } }),
    );
    let reply = agent
        .run_with_options(
            vec![Message::user("Why is Lisbon hilly?")],
            None,
            AgentRunOptions::new().with_chat_options(thinking),
        )
        .await?;
    println!("{}", reply.text());

    // Hosted tools: web search becomes googleSearch, code interpreter codeExecution.
    let _grounded = Agent::builder(client)
        .tool(hosted_web_search())
        .tool(hosted_code_interpreter())
        .build();
    Ok(())
}
```

The last request carried `"generationConfig": {"temperature": 0.3, "thinkingConfig": {"thinkingBudget": 1024}}` alongside the client's `safetySettings`. The gotcha in the comment is real and general: `additional_properties` never replaces a field a typed option wrote, and on Gemini every typed sampling option writes `generationConfig`. There is no typed thinking option, so thinking and sampling settings must travel together in one `generationConfig` value.

## Mistral

`MistralChatClient` speaks Mistral's Chat Completions API, which has OpenAI's message, tool and usage shapes, so it reuses the OpenAI conversion with its own option mapping. Upstream Python and .NET have no Mistral chat client, only embeddings.

| Item | Value |
| --- | --- |
| `new(api_key, model)`, `from_env(model)` | `MISTRAL_API_KEY` (required), `MISTRAL_BASE_URL` (optional, default `https://api.mistral.ai/v1`) |
| Options sent | `temperature`, `top_p`, `max_tokens`, `stop`, `frequency_penalty`, `presence_penalty`, `response_format`, `tool_choice`, `parallel_tool_calls` (with function tools), and `seed` as `random_seed` |
| Not sent | `store`, `metadata`, `logit_bias`, `user`; all hosted tools |

```rust
use agent_framework::mistral::MistralChatClient;
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // MISTRAL_API_KEY; optional MISTRAL_BASE_URL (default https://api.mistral.ai/v1).
    let client = MistralChatClient::from_env("%%model=mistral%%")?;

    let weather = FunctionTool::new(
        "get_weather",
        "Current weather for a city.",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
        |args| async move { Ok(json!({ "city": args["city"], "forecast": "sunny, 21C" })) },
    )
    .into_definition();

    let agent = Agent::builder(client)
        .instructions("Use tools when they help.")
        .tool(weather)
        .build();

    // seed goes out as Mistral's random_seed; store, metadata, logit_bias
    // and user are not sent at all.
    let mut options = ChatOptions::new();
    options.seed = Some(42);
    options.user = Some("ignored-by-mistral".into());
    // Mistral-only fields pass through.
    options.additional_properties.insert("safe_prompt".into(), json!(true));

    let response = agent
        .run_with_options(
            vec![Message::user("What's the weather in Paris?")],
            None,
            AgentRunOptions::new().with_chat_options(options),
        )
        .await?;
    println!("{}", response.text());
    Ok(())
}
```

Both requests of the tool loop carried `"random_seed": 42`, `"safe_prompt": true` and the tool, and no `user`. Mistral's error bodies are not OpenAI-shaped, so its errors are classified by status alone: there is no content-filter detection.

## GitHub Copilot

`GitHubCopilotChatClient` calls GitHub Copilot's OpenAI-compatible chat endpoint, `POST https://api.githubcopilot.com/chat/completions`, with the OpenAI conversion. You give it a GitHub OAuth token or personal access token for an account with Copilot; before the first request it exchanges that for a short-lived Copilot token at `https://api.github.com/copilot_internal/v2/token`, caches it, and exchanges again within about 60 seconds of its expiry. Every request carries the `Editor-Version` and `Copilot-Integration-Id` headers the service requires.

| Item | Value |
| --- | --- |
| `new(github_token, model)`, `from_env(model)` | `GITHUB_COPILOT_TOKEN`, else `GH_COPILOT_TOKEN` (one required); `GITHUB_COPILOT_BASE_URL` optional |
| `with_base_url`, `base_url()`, `model()` | for an enterprise or proxied endpoint |
| Failed exchange | `Error::ServiceInvalidAuth` |

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // GITHUB_COPILOT_TOKEN (or GH_COPILOT_TOKEN): a GitHub OAuth token or PAT
    // for an account with Copilot. GITHUB_COPILOT_BASE_URL overrides the endpoint.
    let client = GitHubCopilotChatClient::from_env("%%model=openai%%")?;
    println!("{}", client.base_url());

    // The GitHub token is exchanged for a short-lived Copilot token on the
    // first request and refreshed about a minute before it expires.
    let agent = Agent::builder(client)
        .instructions("You are a pair programmer.")
        .build();
    println!("{}", agent.run_once("Explain Rust's ? operator.").await?.text());
    Ok(())
}
```

This was compiled but not run: the token-exchange URL is fixed, so the client cannot be tested against a local server. The exchange endpoint is an internal GitHub API, not a documented one, and the model names it accepts are whatever Copilot offers your account. This is a different thing from upstream's `GitHubCopilotAgent`, which drives the Copilot CLI through the GitHub Copilot SDK (with its own MCP servers, skills and permission callbacks); that agent has no Rust counterpart.

## Copilot Studio agents

`CopilotStudioAgent` is not a chat client. It talks to an agent published in Microsoft Copilot Studio over the Direct-to-Engine API and implements `SupportsAgentRun`, so it can stand wherever an agent can: in a workflow, as a tool through [hosting and A2A](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/), or on its own. The model, instructions and tools are the Copilot Studio agent's, configured in Copilot Studio, so `ChatOptions` do not apply.

Configuration has two layers. `CopilotStudioSettings::from_env()` reads the four variables upstream uses, mapping an absent or empty one to `None`; `CopilotStudioConnectionSettings::from_settings(&settings)` then requires the first two and builds the connection.

| Variable | Field | Use |
| --- | --- | --- |
| `COPILOTSTUDIOAGENT__ENVIRONMENTID` | `environment_id` | required; the Power Platform environment |
| `COPILOTSTUDIOAGENT__SCHEMANAME` | `schema_name` | required; the agent's schema name |
| `COPILOTSTUDIOAGENT__AGENTAPPID` | `agent_app_id` | read but unused; there is no token acquisition |
| `COPILOTSTUDIOAGENT__TENANTID` | `tenant_id` | read but unused |

`CopilotStudioConnectionSettings::new(environment_id, agent_identifier)` builds one directly. Its builders are `with_cloud(PowerPlatformCloud::...)` (default `Prod`; `Gov`, `High`, `DoD`, `Mooncake` and the rest are there, and `Other` needs `with_custom_power_platform_cloud(host)`), `with_agent_type(AgentType::Prebuilt)` (default `Published`), and `with_direct_connect_url(url)`, which bypasses host derivation entirely. `conversation_url(None)` shows the URL it will call. The API version is fixed at `API_VERSION`, `2022-03-01-preview`.

Authentication is entirely yours. Upstream Python acquires a Power Platform token through MSAL, falling back to an interactive browser login; the Rust crate has no MSAL and will not open a browser. Implement the `TokenProvider` trait (`async fn get_token(&self) -> Result<String>`) or wrap a token in `StaticTokenProvider::new(token)`. The token is for `https://api.powerplatform.com/.default`, or your cloud's equivalent. The `TokenCredential` types from the Azure crate are a different trait, so a small adapter is needed to use one here.

```rust
use agent_framework::copilotstudio::{
    CopilotStudioConnectionSettings, CopilotStudioSettings, PowerPlatformCloud,
    StaticTokenProvider,
};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // COPILOTSTUDIOAGENT__ENVIRONMENTID and __SCHEMANAME are required here;
    // __AGENTAPPID and __TENANTID are read but not used for auth.
    let settings = CopilotStudioSettings::from_env();
    let mut connection = CopilotStudioConnectionSettings::from_settings(&settings)?
        .with_cloud(PowerPlatformCloud::Prod);
    // A full base URL bypasses host derivation (a test double, a pre-resolved endpoint).
    if let Ok(url) = std::env::var("COPILOTSTUDIO_DIRECT_URL") {
        connection = connection.with_direct_connect_url(url);
    }
    println!("{}", connection.conversation_url(None)?);

    // Bring a Power Platform API token (scope https://api.powerplatform.com/.default).
    let token = std::env::var("COPILOTSTUDIO_TOKEN").unwrap_or_default();
    let agent = CopilotStudioAgent::new(connection, StaticTokenProvider::new(token))
        .with_name("helpdesk");

    // The session keeps the Direct-to-Engine conversation id across turns.
    let mut session = agent.create_session();
    for question in ["Hello!", "How do I reset my password?"] {
        let response = agent
            .run(vec![Message::user(question)], Some(&mut session))
            .await?;
        println!("> {question}\n{}", response.text());
    }
    println!("conversation: {:?}", session.service_session_id());
    Ok(())
}
```

Against a stand-in Direct-to-Engine server, the first `run` posted `{"emitStartConversationEvent": true}` to `.../conversations`, took the conversation id from the returned activities and stored it on the session, then posted the question as a `message` activity to `.../conversations/conv-42`; the second `run` reused the id and posted only the new question. Without a direct URL, an environment id such as `52d1e846-e080-4341-a63f-58f2ab72fb28` in the public cloud becomes the host `52d1e846e0804341a63f58f2ab72fb.28.environment.api.powerplatform.com`.

Behaviour to know: only the last message of each `run` is sent, since the service holds the conversation (Python joins every message passed to the call); Python starts a new conversation on every call, while this crate reuses the session's; without a session, each `run` starts a fresh conversation. Only `message` activities with text become response messages, with the sender's name as `author_name`; typing and trace activities are dropped. `run_with_options` ignores its options with a warning, and `run_stream` is the default buffered implementation. A non-2xx status is an `Error::ServiceStatus` carrying the body. Upstream's `subscribe` operation, documented as internal to Microsoft, is not implemented. `with_id`, `with_description` and `connection()` round out the type.

## Embeddings

Embedding clients implement the `EmbeddingClient` trait: `get_embeddings(values, options)` returns one vector per input, in input order, as a `GeneratedEmbeddings`, which derefs to a slice of `Embedding` (`vector: Vec<f32>`, `dimensions()`, `model`) and carries `usage`. An input is an `EmbeddingInput`, a list of contents embedded together as one vector; strings convert with `.into()`, and `EmbeddingInput::from(Content::Data(...))` makes an image input for the clients that take one. A text-only client given an image returns an error naming the input's index rather than skipping it, so results always line up. `EmbeddingGenerationOptions` has `model`, `dimensions` (`with_dimensions`) and `additional_properties`, from which each client forwards only the keys its API defines.

| Client | Feature | Constructors | Environment | Notes |
| --- | --- | --- | --- | --- |
| `OpenAIEmbeddingClient` | `openai` | `new(key, model)`, `from_env(model)` | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_EMBEDDING_MODEL` | one request per batch; forwards `encoding_format` (base64 is decoded) and `user` |
| `AzureOpenAIEmbeddingClient` | `azure` | `new(endpoint, deployment, key)`, `with_token_credential(...)`, `from_env()` | `AZURE_OPENAI_*`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | default API version `2024-10-21` |
| `FoundryEmbeddingClient` | `foundry` | `new(models_endpoint, model, key)`, `with_credential(...)`, `with_project_endpoint(project, model, credential)`, `from_env(model)` | see below | text and images (Models endpoint only) |
| `OllamaEmbeddingClient` | `ollama` | `new(model)`, `from_env(model)` | `OLLAMA_HOST` | OpenAI-compatible `/v1/embeddings` |
| `MistralEmbeddingClient` | `mistral` | `new(key, model)`, `from_env(model)` (`None` means `mistral-embed`) | `MISTRAL_API_KEY`, `MISTRAL_BASE_URL` | |
| `GeminiEmbeddingClient` | `gemini` | `new(key)`, `from_env()`, `try_with_model(model)` | `GEMINI_API_KEY` or `GOOGLE_API_KEY`, `GOOGLE_EMBEDDING_MODEL` | default `gemini-embedding-2`; task types through `GeminiEmbeddingOptions`; multimodal inputs |
| `BedrockEmbeddingClient` | `bedrock` | `new(...)`, `from_env(model)` | the `AWS_*` variables | Titan models through `InvokeModel`, one signed request per input, run concurrently; `with_endpoint` for PrivateLink |

`FoundryEmbeddingClient` reaches two different surfaces. The Foundry Models inference endpoint (`POST {models_endpoint}/embeddings?api-version=2024-05-01-preview`) takes a key or Entra ID, accepts text and images (images go to `/images/embeddings`, using `with_image_model` or the `image_model` option, else the text model), and forwards `encoding_format`, `input_type` and the entries of an `extra_parameters` map. A project endpoint is turned by `openai_model_base_url` into the resource's OpenAI v1 route (`https://<resource>.openai.azure.com/openai/v1`), which takes Entra ID only, text only, and forwards `encoding_format` and `user`. Both use the Azure OpenAI audience, `DEFAULT_SCOPE` (`https://cognitiveservices.azure.com/.default`), not `FOUNDRY_SCOPE`; `with_scope` changes it. `from_env(model)` reads `FOUNDRY_MODELS_ENDPOINT`, then `FOUNDRY_PROJECT_ENDPOINT` or `FOUNDRY_ENDPOINT`; the Models endpoint wins when both are set. The model comes from the argument or `FOUNDRY_EMBEDDING_MODEL`, the key from `FOUNDRY_MODELS_API_KEY` (otherwise `DefaultAzureCredential`), and the image model from `FOUNDRY_IMAGE_EMBEDDING_MODEL`.

```rust
use std::sync::Arc;

use agent_framework::azure::StaticTokenCredential;
use agent_framework::foundry::embeddings::{openai_model_base_url, FoundryEmbeddingClient};
use agent_framework::prelude::*;
use serde_json::json;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::main]
async fn main() -> Result<()> {
    // OpenAI: OPENAI_API_KEY, optional OPENAI_BASE_URL. An empty model
    // argument falls back to OPENAI_EMBEDDING_MODEL.
    let openai = OpenAIEmbeddingClient::from_env("text-embedding-3-small")?;
    let texts = ["The cat sat on the mat.", "A feline rested on the rug.", "Revenue grew 12%."];
    let mut options = EmbeddingGenerationOptions::new().with_dimensions(256);
    // encoding_format and user are the only extras the OpenAI client forwards.
    options.additional_properties.insert("encoding_format".into(), json!("float"));
    let batch = openai
        .get_embeddings(texts.into_iter().map(Into::into).collect(), Some(options))
        .await?;
    for (text, e) in texts.iter().zip(&batch) {
        println!("{:>4} dims  {text}", e.dimensions());
    }
    println!("usage: {:?}", batch.usage.as_ref().and_then(|u| u.input_token_count));
    println!("cat~feline {:.3}", cosine(&batch[0].vector, &batch[1].vector));

    // Foundry: the project endpoint maps to the resource's OpenAI v1 route.
    let project = "https://my-res.services.ai.azure.com/api/projects/my-proj";
    println!("{}", openai_model_base_url(project)?);

    // Explicit Models inference endpoint with an API key...
    let models_endpoint = std::env::var("FOUNDRY_MODELS_ENDPOINT")
        .unwrap_or_else(|_| "https://my-res.services.ai.azure.com/models".into());
    let foundry = FoundryEmbeddingClient::new(
        &models_endpoint,
        "text-embedding-3-small",
        std::env::var("FOUNDRY_MODELS_API_KEY").unwrap_or_default(),
    );
    let one = foundry.get_embeddings(vec!["hello".into()], None).await?;
    println!("foundry: {} dims", one[0].dimensions());

    // ...or the project route, which takes Entra ID only.
    let _project_client = FoundryEmbeddingClient::with_project_endpoint(
        project,
        "text-embedding-3-small",
        Arc::new(StaticTokenCredential::new("<token for cognitiveservices scope>")),
    )?;
    Ok(())
}
```

Gemini embeddings are a Rust addition: upstream's Gemini package has no embedding client. The vector stores that consume these clients are on the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/).

## Writing your own ChatClient

Implement `ChatClient` to reach an in-house gateway, a provider the crate does not cover, or a test double (the [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/) has scripted fakes). The trait, from `agent_framework::client`, has two required async methods and two optional ones:

| Method | Required | Purpose |
| --- | --- | --- |
| `get_response(messages, options) -> Result<ChatResponse>` | yes | one complete reply |
| `get_streaming_response(messages, options) -> Result<ChatStream>` | yes | a `Pin<Box<dyn Stream<Item = Result<ChatResponseUpdate>> + Send>>` |
| `model() -> Option<&str>` | no (default `None`) | the default model; `Agent::builder` copies it into the agent's options |
| `service_url() -> Option<&str>` | no (default `None`) | the endpoint, recorded on observability spans |

What an implementation owes its callers, all of which the library's `custom_chat_client` example demonstrates offline:

1. **Honour `ChatOptions`**, at least `model` (which wins over your default), `temperature`, `max_tokens`, `tools`, `tool_choice` and `response_format`, and ignore `options.session`, which is an internal side channel and never belongs on the wire.
2. **Return tool requests as `Content::FunctionCall`** with `finish_reason` `tool_calls`. The agent's function-invocation loop runs the tools and calls you again with `Content::FunctionResult` messages; you implement no tool logic.
3. **Fill in `usage_details`**, which observability and cost tracking read. When streaming, put it in a `Content::Usage` on the last update.
4. **Classify errors** with `Error::service_invalid_auth`, `service_invalid_request`, `service_content_filter` and `service_status(status, message, retry_after)` rather than a flat string, so that `RetryingChatClient` retries only what is transient.
5. **Stream for real** if the backend streams, yielding updates as chunks arrive; `ChatResponseUpdate::text(...)` builds a text update.

Because `Arc<dyn ChatClient>` and `Arc<T>` are themselves clients, a custom client can also wrap others. This one fails over between providers, which a retry policy cannot do because it repeats the same call:

```rust
use std::sync::Arc;

use agent_framework::anthropic::AnthropicClient;
use agent_framework::prelude::*;
use async_trait::async_trait;

/// Tries each client in turn, moving on only for failures another provider
/// might not share: rate limits, 5xx, timeouts, connection errors.
struct FallbackChatClient {
    clients: Vec<Arc<dyn ChatClient>>,
}

fn worth_falling_back(err: &Error) -> bool {
    match err {
        Error::ServiceStatus { status, .. } => *status == 408 || *status == 429 || *status >= 500,
        Error::Service(_) => true, // transport failure
        _ => false,                // bad request, bad auth, content filter: same everywhere
    }
}

#[async_trait]
impl ChatClient for FallbackChatClient {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let mut last = Error::Configuration("no clients configured".into());
        for client in &self.clients {
            // Each provider has its own model ids, so drop a model override
            // the agent filled in from the first client.
            let mut options = options.clone();
            options.model = None;
            match client.get_response(messages.clone(), options).await {
                Ok(response) => return Ok(response),
                Err(e) if worth_falling_back(&e) => {
                    eprintln!("falling back after: {e}");
                    last = e;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last)
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        // Fall back only while opening the stream; a failure mid-stream is
        // returned to the caller, since part of the answer has been shown.
        let mut last = Error::Configuration("no clients configured".into());
        for client in &self.clients {
            let mut options = options.clone();
            options.model = None;
            match client.get_streaming_response(messages.clone(), options).await {
                Ok(stream) => return Ok(stream),
                Err(e) if worth_falling_back(&e) => last = e,
                Err(e) => return Err(e),
            }
        }
        Err(last)
    }

    // Left as None so the agent does not stamp one provider's model id on
    // every request.
    fn model(&self) -> Option<&str> {
        None
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let primary = OpenAIChatCompletionClient::new("sk-unused", "%%model=openai%%")
        .with_base_url("http://127.0.0.1:9/v1"); // nothing listens here
    let secondary = AnthropicClient::from_env("%%model=anthropic%%")?;

    let client = FallbackChatClient {
        clients: vec![Arc::new(primary), Arc::new(secondary)],
    };
    let agent = Agent::builder(client).instructions("Be brief.").build();
    println!("{}", agent.run_once("Say hello.").await?.text());
    Ok(())
}
```

```text
falling back after: service error: request failed: error sending request for url (http://127.0.0.1:9/v1/chat/completions)
Mock reply to: Say hello.
```

Two details matter in a wrapper like this. `options.model` must be cleared, or the second provider receives the first provider's model id. And conversation state does not transfer: a Responses `conversation_id` means nothing to Anthropic, so a failover client should sit under stateless agents, or under ones whose history lives in a history provider. The agent wraps whatever you give it in a `FunctionInvokingChatClient`, so the tool loop runs above your wrapper and each round trip can fail over independently.

## Switching providers at runtime

Since `Arc<dyn ChatClient>` is a `ChatClient`, a factory can choose the provider from configuration and the rest of the program never names a concrete type:

```rust
use std::sync::Arc;

use agent_framework::anthropic::AnthropicClient;
use agent_framework::azure::{AzureOpenAIClient, DefaultAzureCredential};
use agent_framework::gemini::GeminiChatClient;
use agent_framework::ollama::OllamaChatClient;
use agent_framework::prelude::*;

/// Pick a provider from configuration. Arc<dyn ChatClient> is itself a
/// ChatClient, so the rest of the program does not care which one it got.
fn chat_client(provider: &str) -> Result<Arc<dyn ChatClient>> {
    Ok(match provider {
        "anthropic" => Arc::new(AnthropicClient::from_env("%%model=anthropic%%")?),
        "gemini" => Arc::new(GeminiChatClient::from_env("%%model=gemini%%")?),
        "ollama" => Arc::new(OllamaChatClient::from_env("llama3.2")?),
        "azure" => Arc::new(AzureOpenAIClient::with_token_credential(
            std::env::var("AZURE_OPENAI_ENDPOINT")
                .map_err(|_| Error::Configuration("AZURE_OPENAI_ENDPOINT is not set".into()))?,
            std::env::var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
                .map_err(|_| Error::Configuration("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME is not set".into()))?,
            Arc::new(DefaultAzureCredential::new("https://cognitiveservices.azure.com/.default")),
        )),
        "openai-responses" => Arc::new(OpenAIChatClient::from_env("%%model=openai%%")?),
        _ => Arc::new(OpenAIChatCompletionClient::from_env("%%model=openai%%")?),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    for provider in ["openai", "openai-responses", "anthropic"] {
        let client = chat_client(provider)?;
        println!("{provider}: model={:?} url={:?}", client.model(), client.service_url());
        let agent = Agent::builder(client).instructions("Answer in five words.").build();
        println!("  {}", agent.run_once("What is Rust?").await?.text());
    }
    Ok(())
}
```

What stays the same across providers is the message and tool model; what changes is everything in the option table above. Code that relies on `seed`, on `stop`, on structured output being enforced, or on hosted tools should check which provider it was given, or keep to the common subset: instructions, `temperature`, `max_tokens`, function tools and `tool_choice`.

## Differences from Python and .NET

- **Names.** `OpenAIChatClient` is the Responses client and `OpenAIChatCompletionClient` the Chat Completions one, as in current upstream Python. Azure has separate types (`AzureOpenAIClient`, `AzureOpenAIResponsesClient`) where Python passes Azure settings to the OpenAI classes.
- **Credentials.** Python and .NET take any `azure-identity` or `Azure.Identity` credential. Rust ships its own seven credential types implementing `TokenCredential`, plus `SdkTokenCredential` behind `entra-sdk` for the official Rust SDK's credentials.
- **Typed options.** Python uses a per-provider `TypedDict` of options (`OpenAIChatOptions`, `AnthropicChatOptions`). Rust has one `ChatOptions` struct for all providers, and anything provider-specific goes in `additional_properties` under the API's own field name.
- **Not ported:** the Claude Agent SDK agent (`ClaudeAgent`), upstream's `GitHubCopilotAgent` (the Rust GitHub Copilot client is a plain chat client), binding `FoundryAgent` to an agent stored in a Foundry project, Bing grounding connections, Ollama's native API, Gemini on Vertex AI, real streaming for Bedrock and for Claude on Bedrock, Vertex AI and Foundry, and AWS and Google credential discovery.
- **Rust additions:** a Mistral chat client, Gemini embeddings, structured output on Anthropic through the system prompt, accumulated streaming log probabilities, and content-filter details on Azure OpenAI errors.

The [Python providers page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/) covers the same ground for `agent-framework` on PyPI.

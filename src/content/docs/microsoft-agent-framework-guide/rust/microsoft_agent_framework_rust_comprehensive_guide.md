---
title: "Microsoft Agent Framework (Rust) — Comprehensive guide"
description: "A tour of the community Rust port agent-framework-rs from first agent to workflows: runs and streaming, messages, providers, sessions, structured output, tools and approvals, per-run options, middleware, context providers, retries and errors, workflows and orchestrations, with links to the topic pages."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Comprehensive guide"
  order: 2
---

This page is a tour of the `agent-framework` crate, in the order you meet its parts when building something: an agent and its runs, the provider behind it, sessions, structured output, tools and approvals, middleware, context providers, error handling, and then workflows and the prebuilt multi-agent orchestrations. Each section explains how the piece works and gives one runnable example, then points to the topic page that covers it in full:

| Topic | Page |
| --- | --- |
| Every chat and embedding client, their settings and wire behaviour | [Providers](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_providers/) |
| Function tools, the tool loop and its limits, approvals, hosted tools, shell tools | [Tools](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/) |
| The three middleware levels, client decorators, retries, the error enum, agent hooks | [Middleware](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/) |
| Sessions, history providers, persistence, the session store | [Sessions](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/) |
| Context providers, retrieval, compaction, skills, the agent harness, vector stores | [Context and memory](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/) |
| The graph engine | [Workflows](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/) |
| Sequential, concurrent, group chat, handoff and Magentic | [Orchestrations](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/) |
| MCP servers as tools | [MCP](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/) |
| Serving agents over HTTP (DevUI-style, OpenAI-compatible, AG-UI, A2A), and calling remote A2A agents | [Hosting and A2A](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/) |
| Declarative agents, memory and storage backends, vector stores, Purview, Copilot Studio, shell tools | [Integrations](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/) |
| Tracing spans, metrics and export | [Observability](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/) |
| Test doubles, `cargo test` patterns, production settings | [Testing and production](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/) |
| Complete programs | [Recipes](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/) |

`agent-framework` is a community port maintained in [codehalwell/agent-framework-rs](https://github.com/codehalwell/agent-framework-rs), not a Microsoft package (see the [overview](/microsoft-agent-framework-guide/rust/)). Every program below was compiled against version %%version=agent-framework-rs%% from crates.io and run: the ones that call OpenAI against a local stand-in for the OpenAI API, so their replies are placeholders, and the workflow example with no network at all.

## Dependencies

Most examples need only the default `openai` feature plus a few common crates:

```toml
[dependencies]
agent-framework = "%%version=agent-framework-rs%%"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
async-trait = "0.1"  # implementing ChatClient, Middleware, ContextProvider
futures = "0.3"      # StreamExt for streaming
schemars = "0.8"     # FunctionTool::typed
```

`schemars` must be the 0.8 series, which is the version the crate derives tool schemas with. The crate's minimum supported Rust version is 1.88. Other providers and integrations are cargo features of the same crate; the [overview](/microsoft-agent-framework-guide/rust/#cargo-features) lists them.

## Crate map

The umbrella crate re-exports `agent-framework-core` at its root, so `agent_framework::types`, `agent_framework::workflow` and so on are the core modules. Each companion crate appears as a module when its feature is on (`agent_framework::anthropic`, `agent_framework::mcp`, and so on; the `tools` feature's shell tools are `agent_framework::builtin_tools`, since `tools` is taken by the core module), and `agent_framework::prelude` gathers the common names from all of them.

| Module | Purpose |
| --- | --- |
| `agent` | `Agent`, `AgentBuilder`, `AgentRunOptions`, the `SupportsAgentRun` trait, `AsToolOptions` |
| `client` | The `ChatClient` and `EmbeddingClient` traits, `FunctionInvokingChatClient`, `RetryingChatClient`, `RetryPolicy` |
| `types` | `Message`, `Role`, `Content` and its variants, `ChatOptions`, `ChatResponse`, `AgentResponse`, `ResponseFormat`, `ToolMode` |
| `tools` | `FunctionTool`, `ToolDefinition`, `ApprovalMode`, `FunctionInvocationConfig`, the `Tool` and `ToolSource` traits, the hosted-tool constructors |
| `session`, `history` | `AgentSession`, `SessionState`, `InMemoryHistoryProvider`, `HistoryProvider` |
| `memory` | The `ContextProvider` trait and `SessionContext` |
| `middleware` | `Middleware<C>`, `Next`, `FnMiddleware`, and the three contexts `AgentContext`, `ChatContext`, `FunctionInvocationContext` |
| `skills`, `text_search`, `compaction` | Skills, retrieval over any search function, history compaction strategies |
| `observability` | `ObservableChatClient` and OpenTelemetry GenAI spans |
| `settings` | `SecretString` and `load_setting` |
| `workflow` | The graph engine and the prebuilt orchestrations |
| `error` | `Error` and `Result` |
| `harness`, `agent_hooks`, `session_store`, `vectors` | Experimental, each behind its own feature ([below](#experimental-agent-apis)) |

Python users will recognise the layout: the port follows `agent_framework`'s module structure, so `_types` became `types`, `_middleware` became `middleware`, and so on.

## Agents and runs

`Agent::builder(client)` takes any `ChatClient`. The builder sets a name, instructions, default options, tools, context providers and middleware, and `build()` returns an `Agent`. The builder wraps the client in a `FunctionInvokingChatClient`, so local tools run automatically; you never wrap it yourself. An `Agent` is cheap to clone, because the client, providers and middleware sit behind `Arc`s.

| `AgentBuilder` method | Sets |
| --- | --- |
| `id`, `name`, `description` | Identity. The id is a random UUID unless set; orchestrations and hosting use it |
| `instructions` | The system prompt |
| `model`, `temperature`, `max_tokens`, `response_format`, `chat_options` | Default `ChatOptions` for every run |
| `tool`, `tools`, `tool_source` | [Tools](#function-tools) |
| `function_invocation_config` | [Limits on the tool loop](#bounding-the-tool-loop) |
| `context_provider`, `context_providers`, `with_compaction` | [Context providers](#context-providers) |
| `middleware`, `chat_middleware`, `function_middleware` | [Middleware](#middleware) |
| `observability_config` | Content capture and semantic-convention version for the tool loop's spans |

There are four ways to run an agent:

- `agent.run_once(input)` runs a single turn in a fresh session that is then discarded. `input` is anything that converts into messages: a `&str`, a `String`, a `Message` or a `Vec<Message>`.
- `agent.run(messages, Some(&mut session))` runs with a `Vec<Message>` and an optional session. It is a method of the `SupportsAgentRun` trait, which the prelude imports, and is what workflows, orchestrations and hosting call.
- `agent.run_with_options(messages, session, options)` applies [per-run options](#per-run-options).
- `agent.run_stream(input, session, options)` and `agent.run_stream_once(input)` return a stream of `AgentResponseUpdate`s. `run_stream` takes the session by value: the stream owns it and records the turn in its history provider when the stream has been read to the end.

All return `Result<AgentResponse>` (or a stream). `AgentResponse::text()` concatenates the assistant text; `messages`, `usage_details`, `finish_reason`, `response_id`, `conversation_id` and `value` hold the rest.

```rust
use agent_framework::prelude::*;
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    // Reads OPENAI_API_KEY, and OPENAI_BASE_URL if set.
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a concise assistant. Answer in one sentence.")
        .temperature(0.2)
        .build();

    // One turn, no session.
    let response = agent.run_once("What is a borrow checker?").await?;
    println!("{}", response.text());
    println!("finish reason: {:?}", response.finish_reason);
    println!("usage: {:?}", response.usage_details);

    // Streaming: each update carries a text fragment.
    let mut stream = agent.run_stream_once("Name three Rust editions.").await?;
    while let Some(update) = stream.next().await {
        print!("{}", update?.text());
        std::io::stdout().flush().ok();
    }
    println!();
    Ok(())
}
```

`finish_reason` deserves a check in production code. A response cut off by the token limit (`length`) or a content filter (`content_filter`) still has text that reads like a finished answer; the finish reason is the only reliable signal.

Two things limit streaming. When the request has a local tool that can run, the tool loop runs to completion and then replays its messages as updates, and an agent with agent-level middleware buffers the whole run the same way. Text then arrives in whole messages rather than token by token. The [tools](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#concurrency-usage-and-streaming) and [middleware](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/#streaming-runs) pages give the details.

### Messages and content

A `Message` has a `role`, a list of `contents` and an optional `author_name`. `Message::user`, `Message::assistant` and `Message::system` build text messages; `Message::with_contents` takes any mix of `Content` variants. Images and files go in as `Content::Data` (inline bytes, base64-encoded as a data URI) or `Content::Uri` (a URL plus media type):

```rust
use agent_framework::prelude::*;
use agent_framework::types::{DataContent, UriContent};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai%%")?;
    let agent = Agent::builder(client).build();

    let png = std::fs::read("diagram.png")?;
    let message = Message::with_contents(
        Role::user(),
        vec![
            Content::text("Compare these two images."),
            Content::Data(DataContent::from_bytes(&png, "image/png")),
            Content::Uri(UriContent {
                uri: "https://example.com/photo.jpg".into(),
                media_type: "image/jpeg".into(),
            }),
        ],
    );

    let response = agent.run(vec![message], None).await?;
    println!("{}", response.text());
    Ok(())
}
```

Other variants include `FunctionCall`, `FunctionResult`, `FunctionApprovalRequest` and `FunctionApprovalResponse`, `TextReasoning`, `Usage`, `HostedFile`, and the hosted-tool call and result types. A tool's result is JSON sent to the model as text, so a tool cannot hand the model an image; send images in message content instead. Which modalities a provider converts is listed on the [providers page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_providers/).

## Providers

Every provider is a `ChatClient`, so the agent code is the same whichever you pick, and so are workflows, orchestrations and the hosting routers. Each lives behind a cargo feature; only `openai` is on by default.

| Provider | Type | Feature |
| --- | --- | --- |
| OpenAI Chat Completions, or any compatible endpoint | `OpenAIChatCompletionClient` | `openai` |
| OpenAI Responses API | `OpenAIChatClient` | `openai` |
| Azure OpenAI Chat Completions and Responses | `azure::AzureOpenAIClient`, `azure::AzureOpenAIResponsesClient` | `azure` |
| Microsoft Foundry project | `FoundryChatClient`, `FoundryAgent` | `foundry` |
| Anthropic, and Claude on Bedrock, Vertex AI or Foundry | `AnthropicClient`, `AnthropicBedrockClient`, `AnthropicVertexClient`, `AnthropicFoundryClient` | `anthropic` |
| Google Gemini | `GeminiChatClient` | `gemini` |
| Mistral | `MistralChatClient` | `mistral` |
| Ollama, Foundry Local | `OllamaChatClient`, `FoundryLocalChatClient` | `ollama`, `foundry-local` |
| Amazon Bedrock Converse API | `BedrockChatClient` | `bedrock` |
| GitHub Copilot | `GitHubCopilotChatClient` | `github-copilot` |
| A published Copilot Studio agent | `CopilotStudioAgent` (an agent, not a chat client) | `copilotstudio` |

Each client has a `from_env(...)` constructor that reads its provider's environment variables and fails with `Error::Configuration` naming a missing one, and a `new(...)` that takes the settings directly. The model given to the constructor is a default that `ChatOptions::model` overrides.

Two OpenAI clients exist because the APIs differ. `OpenAIChatCompletionClient` speaks `/v1/chat/completions`, which every OpenAI-compatible server (vLLM, LM Studio, gateways) also speaks. `OpenAIChatClient` speaks the Responses API, which supports every hosted tool, reasoning items and service-side conversation state: it returns a response id that the agent stores on the session, so later turns send `previous_response_id` instead of replaying the history.

Choosing a provider at run time needs nothing more than a `match`:

```toml
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["anthropic", "azure", "gemini", "ollama"] }
```

```rust
use std::sync::Arc;

use agent_framework::anthropic::AnthropicClient;
use agent_framework::azure::{AzureOpenAIClient, DefaultAzureCredential};
use agent_framework::gemini::GeminiChatClient;
use agent_framework::ollama::OllamaChatClient;
use agent_framework::prelude::*;

fn build(provider: &str) -> Result<Agent> {
    let builder = match provider {
        "anthropic" => Agent::builder(AnthropicClient::from_env("%%model=anthropic%%")?),
        "gemini" => Agent::builder(GeminiChatClient::from_env("%%model=gemini%%")?),
        "ollama" => Agent::builder(OllamaChatClient::from_env("llama3.2")?),
        "azure-key" => Agent::builder(AzureOpenAIClient::from_env()?),
        "azure-entra" => {
            // Environment, workload identity, managed identity, then the Azure CLI.
            let credential = Arc::new(DefaultAzureCredential::new(
                "https://cognitiveservices.azure.com/.default",
            ));
            Agent::builder(AzureOpenAIClient::with_token_credential(
                "https://my-resource.openai.azure.com",
                "my-deployment",
                credential,
            ))
        }
        _ => Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai%%")?),
    };
    Ok(builder.instructions("You are a helpful assistant.").build())
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider = std::env::var("PROVIDER").unwrap_or_default();
    let agent = build(&provider)?;
    println!("{}", agent.run_once("Say hello.").await?.text());
    Ok(())
}
```

The differences between providers are mostly in what they drop. Each client maps the `ChatOptions` fields it understands and silently ignores the rest, and a hosted tool a provider does not support is skipped with a `tracing` warning. Anthropic has no `response_format` field, so the client folds the schema into the system prompt, and it always sends `max_tokens` because the Messages API requires it. Claude on Bedrock, Vertex AI and Foundry, and `BedrockChatClient`, do not stream: their streaming methods make one ordinary request and return it as a single update. The [providers page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_providers/) has the full table of options per client, the environment variables, the Entra ID credential chain, embeddings, and fallback between providers.

### Writing your own client

`ChatClient` has two required methods, `get_response` and `get_streaming_response`, plus optional `model()` and `service_url()`. Implementing it is how you reach an in-house gateway, wrap another client (a fallback, a cache, a per-request hook), or build a test double that lets `cargo test` exercise the tool loop, approvals and workflows without a network. Return tool requests as `Content::FunctionCall` and the function-invocation loop does the rest, and return classified errors (`Error::service_status`, `Error::service_invalid_auth` and so on, see [errors](#retries-and-errors)) so that retries behave. The [providers page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_providers/#writing-your-own-chatclient) lists what an implementation owes its callers, and the [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/#a-scripted-chat-client) builds a scripted client step by step.

## Sessions and history

An `AgentSession` is one conversation: a `session_id`, an optional service-side conversation id, a free-form `state` bag and a list of context providers. History is not stored on the session itself. It lives in a history provider, which is an ordinary [context provider](#context-providers) that loads stored turns before a run and records the new ones after it. `agent.create_session()` returns a session with an `InMemoryHistoryProvider` already attached, and a session passed without one gets one on first use, so multi-turn conversations need no setup:

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are a friendly assistant.")
        .build();

    let mut session = agent.create_session();
    agent
        .run(vec![Message::user("My name is Ada.")], Some(&mut session))
        .await?;
    let reply = agent
        .run(vec![Message::user("What is my name?")], Some(&mut session))
        .await?;
    println!("{}", reply.text());
    Ok(())
}
```

When the service keeps the conversation (the Responses API by default, or a session created with a service id), the agent stores the service's conversation id on the session and leaves history providers out of later runs, so earlier turns are not sent twice.

Persisting a conversation means saving two things, because `AgentSession::to_dict()` serialises the ids and the state bag but not the messages: the session, and its history provider (`InMemoryHistoryProvider::to_dict`). Restore both with `agent.session_from_dict` and `InMemoryHistoryProvider::from_dict`, and reattach the provider. For durable storage, swap the in-memory provider for one that writes through: `RedisChatMessageStore` and `CosmosChatMessageStore` (see [memory and storage backends](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#memory-and-storage-backends)), `FileHistoryProvider` (feature `experimental-file-history`), or your own. The `experimental-session-store` feature adds `AgentSessionStore`, which saves and restores whole sessions, history included, under a key. The [sessions page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/) covers all of these, with streaming, writing a message store, and the options that switch between local and service history.

`SessionState` is shared by reference: clones of a session see the same bag, which is how a context provider, or a sub-agent called through `as_tool` with `propagate_session(true)`, shares state with its caller.

## Structured output

Set a `ResponseFormat` on the builder (or per run in `ChatOptions`) and parse the reply with `parse_json::<T>()`. The function-invocation loop also parses JSON responses into `AgentResponse::value` when a JSON format was requested.

```rust
use agent_framework::prelude::*;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Recipe {
    name: String,
    minutes: u32,
    ingredients: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "minutes": { "type": "integer" },
            "ingredients": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["name", "minutes", "ingredients"],
        "additionalProperties": false
    });

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You suggest simple recipes.")
        .response_format(ResponseFormat::json_schema("Recipe", schema))
        .build();

    let response = agent.run_once("A quick weeknight pasta, please.").await?;
    let recipe: Recipe = response.parse_json()?;
    println!("{} ({} min): {}", recipe.name, recipe.minutes, recipe.ingredients.join(", "));
    assert!(response.value.is_some());
    Ok(())
}
```

`parse_json` reads the text of the last assistant message that has any, ignoring reasoning content, so a reasoning model's chain of thought does not break the parse. There is no derive-based helper that turns a Rust type into a response format; generate the schema with `schemars` (`serde_json::to_value(schemars::schema_for!(Recipe))`) if you prefer that to writing it by hand. `ResponseFormat::JsonObject` asks for any valid JSON object without a schema, and `ResponseFormat::Text` is the default. How each provider carries the format (natively, or folded into the prompt on Anthropic) is on the [providers page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_providers/); Bedrock's Converse client drops it.

## Function tools

A `FunctionTool` pairs a name, a description, a JSON Schema for the arguments and an async closure. Two constructors exist:

- `FunctionTool::new(name, description, schema, |args: Value| async move { ... })` takes a hand-written schema and gives the closure a `serde_json::Value`.
- `FunctionTool::typed(name, description, |args: T| async move { ... })` derives the schema from a type that implements `Deserialize` and `schemars::JsonSchema`, and deserialises the arguments into it. Doc comments on fields become schema descriptions. Arguments that do not deserialise produce a tool error, which the model sees as a failed call.

Either way, `.into_definition()` turns the tool into the `ToolDefinition` that `AgentBuilder::tool` accepts. Everything an agent can call, including MCP tools and other agents, ends up as a `ToolDefinition`.

```rust
use agent_framework::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// Arguments for get_weather.
#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// The city to look up.
    city: String,
    /// "celsius" or "fahrenheit"; defaults to celsius.
    #[serde(default)]
    units: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let weather = FunctionTool::typed(
        "get_weather",
        "Get the current weather for a city.",
        |args: WeatherArgs| async move {
            let units = args.units.unwrap_or_else(|| "celsius".into());
            Ok(json!({ "city": args.city, "temperature": 21, "units": units }))
        },
    );
    // The schema the model will see.
    println!("{}", serde_json::to_string_pretty(&weather.parameters_schema())?);

    let time = FunctionTool::new(
        "get_time",
        "Get the current UTC time.",
        json!({ "type": "object", "properties": {} }),
        |_args| async move { Ok(json!("12:00 UTC")) },
    );

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Use the tools when they help.")
        .tool(weather.into_definition())
        .tool(time.into_definition())
        .build();

    println!("{}", agent.run_once("What's the weather in Paris?").await?.text());
    Ok(())
}
```

When the model asks for a tool, the loop runs it, appends the result, and calls the model again until it answers in text. The calls of one model response run concurrently, and their results go back in the order the model asked for them. A tool error does not fail the run: the model receives `"tool execution failed"` (or the error's text, with `include_detailed_errors`) and can try again or explain. Two cases stop the run instead: `Error::MiddlewareFailure` from a tool or from [function middleware](#middleware), and a call to an unknown tool when `terminate_on_unknown_calls` is set.

A `FunctionTool` can also cap its own use with `max_invocations(n)` and `max_invocation_exceptions(n)`. The counters live in the tool and are shared between its clones, so they count across runs, not per run.

The [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/) covers schemas in depth, implementing the `Tool` trait yourself, exactly what the model sees for each kind of failure, tools that change at run time (per-run tools, declaration-only tools, `ToolSource`) and the shell tools.

### Bounding the tool loop

`FunctionInvocationConfig`, set with the builder's `function_invocation_config`, bounds the loop:

| Field | Default | Meaning |
| --- | --- | --- |
| `max_iterations` | 40 | Model round trips that may request tools |
| `max_function_calls` | `None` | Tool executions per request, across iterations and approval round trips |
| `max_duration_seconds` | `None` | Wall-clock budget for the loop, including time waiting for approvals |
| `max_consecutive_errors_per_request` | 3 | Consecutive failing rounds tolerated before tools are switched off |
| `terminate_on_unknown_calls` | `false` | Fail the run on a call to a tool that does not exist |
| `include_detailed_errors` | `false` | Send a failed tool's error text to the model |
| `allow_concurrent_invocation` | `true` | Run the calls of one model response concurrently |
| `enabled` | `true` | `false` returns function calls to the caller unexecuted |

None of the three limits fails the run. When one is reached, the loop makes one last request with tools switched off and returns whatever the model answers. The budgets are checked after each batch of calls, so one batch can overshoot, and they carry over across approval round trips through the session's state. For an unattended agent, set all three.

`ToolMode` controls whether the model must call a tool. Set it per run through `ChatOptions::with_tool_choice`: `ToolMode::auto()`, `ToolMode::none()`, `ToolMode::required_any()` or `ToolMode::required_function("name")`. The [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#the-three-limits) shows each limit taking effect.

### Hosted tools

`hosted_web_search()`, `hosted_code_interpreter()`, `hosted_file_search(max_results)`, `hosted_image_generation()` and `hosted_mcp(name, url, allowed_tools)` return marker `ToolDefinition`s for tools the provider runs on its own servers, so the tool loop, function middleware and `ApprovalMode` do not apply to them. Support depends on the client:

| Client | Hosted tools it sends |
| --- | --- |
| `OpenAIChatClient`, `AzureOpenAIResponsesClient`, `FoundryChatClient` (Responses API) | All five |
| `AnthropicClient` and its Bedrock, Vertex and Foundry variants | Web search, code execution and MCP servers |
| `GeminiChatClient` | Google Search (`hosted_web_search`) and code execution (`hosted_code_interpreter`) |
| Chat Completions clients (`OpenAIChatCompletionClient`, `AzureOpenAIClient`, and the Chat Completions-shaped Ollama, Foundry Local and GitHub Copilot clients) | Web search only, as the top-level `web_search_options` field; useful only with OpenAI's search-enabled models |
| `MistralChatClient`, `BedrockChatClient` | None |

A tool a client does not support is skipped with a `tracing` warning rather than failing the request, so declaring one is safe but not a guarantee that the model has it. The [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#hosted-tools) shows the settings each tool takes and the exact wire form per provider.

### An agent as a tool

`agent.as_tool(AsToolOptions::new().name(...).description(...))` exposes one agent to another as a function tool that takes a single `task` string and returns the sub-agent's text. `AsToolOptions` can also rename the argument, require approval for the call, stream the sub-agent's updates to a callback, and, with `propagate_session(true)`, give the sub-agent a child session that shares the caller's state bag but not its service conversation id. The [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#an-agent-as-a-tool) has a worked example; for more than a couple of agents, the [orchestrations](#orchestrations) are usually a better fit.

## Tool approvals

Mark a tool `ApprovalMode::AlwaysRequire` and the loop stops before running it. The response then carries `FunctionApprovalRequestContent` items, returned by `user_input_requests()`. Answer each with `req.create_response(true)` (or `false` to refuse) and send the answers back as a user message on the same session:

```rust
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let delete_file = FunctionTool::new(
        "delete_file",
        "Permanently delete a file.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
        |args| async move { Ok(json!({ "deleted": args["path"] })) },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition();

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You manage files. Use the tools for file actions.")
        .tool(delete_file)
        .build();

    let mut session = agent.create_session();
    let mut input = vec![Message::user("Delete scratch.txt")];

    // Bounded, so a model that keeps asking cannot loop forever.
    for _ in 0..5 {
        let response = agent.run(input, Some(&mut session)).await?;
        let requests = response.user_input_requests();
        if requests.is_empty() {
            println!("{}", response.text());
            break;
        }
        let mut answers = Vec::new();
        for req in requests {
            println!("approve {}({:?})? yes", req.function_call.name, req.function_call.arguments);
            answers.push(Content::FunctionApprovalResponse(req.create_response(true)));
        }
        input = vec![Message::with_contents(Role::user(), answers)];
    }
    Ok(())
}
```

Approval is decided per batch: if any call in a model response needs approval, every call in that response comes back as a request. A rejected call is not run, and the model is told it was rejected. Each answer carries the function call it approves, so an approval can be answered in another process as long as you persist the session (state bag and history) in between, and answering the same request twice does not run the call twice. Approval also exists for sub-agents (`AsToolOptions::approval_mode`), in the sequential and concurrent orchestrations (`with_request_info()`), for hosted MCP servers (`McpApprovalMode`, enforced by the service), and as standing rules (`ToolApprovalAgent`, experimental). The [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#approvals) covers all of them.

## Per-run options

`AgentRunOptions` overrides the agent's defaults for a single call. Its `ChatOptions` are merged over the build-time options: per-run values win for scalar fields, map fields (`metadata`, `logit_bias`) merge per key, instructions are appended to the agent's on a new line rather than replacing them, and tools are combined by name, so a per-run tool whose name the agent already uses is dropped. `with_tool` and `with_tools` add tools for that run only. Middleware cannot be changed per run.

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are terse.")
        .temperature(0.0)
        .build();

    let options = AgentRunOptions::new().with_chat_options(
        ChatOptions::new()
            .with_temperature(0.9)
            .with_max_tokens(200)
            .with_instructions("This time, be enthusiastic."),
    );
    let response = agent
        .run_with_options(vec![Message::user("Describe Rust.")], None, options)
        .await?;
    println!("{}", response.text());
    Ok(())
}
```

`ChatOptions` covers the usual fields (`model`, `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `logit_bias`, `tool_choice`, `allow_multiple_tool_calls`, `response_format`, `conversation_id`, `store`, `metadata`, `user`) plus `additional_properties`, whose entries are copied into the top level of the request body for provider-specific values. `SupportsAgentRun::run_with_options` on an agent that does not support options (the trait's default) logs a warning and ignores them.

## Middleware

Middleware runs at three levels, and each level has its own context type:

| Level | Builder method | Context | Runs |
| --- | --- | --- | --- |
| Agent | `.middleware(...)` | `AgentContext` (`messages`, `result`, `terminate`, `metadata`, `is_streaming`) | Once per run, around everything after context providers have loaded |
| Chat | `.chat_middleware(...)` | `ChatContext` (`messages`, `chat_options`, `result`, `terminate`, ...) | Once per run, around the call into the chat client, which includes the whole tool loop |
| Function | `.function_middleware(...)` | `FunctionInvocationContext` (`function_name`, `arguments`, `session`, `result`, `terminate`, ...) | Once per local tool call |

**Chat middleware runs once per run, not once per model request.** This is the main difference from Python, whose chat middleware wraps each model request and so sees every round trip of the tool loop. In the Rust port the agent applies chat middleware outside the function-invocation loop: it sees the messages before the first model request and receives the final response after the last, so it never sees tool results going back to the model. To act on every model request, wrap the client in your own `ChatClient` instead, as `RetryingChatClient` and `ObservableChatClient` do. On a streaming run without agent middleware, chat middleware can change the request but not the response.

A middleware implements `Middleware<C>`: it receives the context and a `Next`, and can change the context before calling `next.run(ctx)`, inspect the result after it, or set `result` and return without calling `next` to short-circuit. Middleware added first runs outermost. `FnMiddleware::new(closure)` (in `agent_framework::middleware`) adapts an async closure if you prefer not to write a struct.

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;
use serde_json::json;

/// Agent level: refuse some requests outright and count tokens on the rest.
#[derive(Default)]
struct Guardrail {
    output_tokens: AtomicU64,
}

#[async_trait]
impl Middleware<AgentContext> for Guardrail {
    async fn process(&self, mut ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        let last = ctx.messages.last().map(Message::text).unwrap_or_default();
        if last.to_lowercase().contains("password") {
            ctx.result = Some(AgentResponse {
                messages: vec![Message::assistant("I can't help with credentials.")],
                ..Default::default()
            });
            return Ok(ctx); // the model is never called
        }
        let ctx = next.run(ctx).await?;
        if let Some(usage) = ctx.result.as_ref().and_then(|r| r.usage_details.as_ref()) {
            self.output_tokens
                .fetch_add(usage.output_token_count.unwrap_or(0), Ordering::Relaxed);
        }
        Ok(ctx)
    }
}

/// Chat level: runs once per run, before the first model request.
struct LogRuns;

#[async_trait]
impl Middleware<ChatContext> for LogRuns {
    async fn process(&self, ctx: ChatContext, next: Next<ChatContext>) -> Result<ChatContext> {
        println!("[chat] run starts with {} message(s)", ctx.messages.len());
        next.run(ctx).await
    }
}

/// Function level: block a tool for some arguments.
struct NoProduction;

#[async_trait]
impl Middleware<FunctionInvocationContext> for NoProduction {
    async fn process(
        &self,
        ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        if ctx.arguments["env"] == "production" {
            // The model sees this message as the tool's result and can recover.
            // Error::middleware_failure would stop the whole run instead.
            return Err(Error::tool_rejected("Deploying to production is not allowed."));
        }
        next.run(ctx).await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let deploy = FunctionTool::new(
        "deploy",
        "Deploy the service to an environment.",
        json!({
            "type": "object",
            "properties": { "env": { "type": "string", "enum": ["staging", "production"] } },
            "required": ["env"]
        }),
        |args| async move { Ok(json!({ "deployed_to": args["env"] })) },
    )
    .into_definition();

    let guardrail = Arc::new(Guardrail::default());
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .tool(deploy)
        .middleware(guardrail.clone())
        .chat_middleware(Arc::new(LogRuns))
        .function_middleware(Arc::new(NoProduction))
        .build();

    println!("{}", agent.run_once("What is the admin password?").await?.text());
    // Two model requests (the tool call, then the answer), one chat middleware call.
    println!("{}", agent.run_once("Deploy to staging.").await?.text());
    println!("output tokens so far: {}", guardrail.output_tokens.load(Ordering::Relaxed));
    Ok(())
}
```

For a refused tool call, return `Error::tool_rejected(msg)` when the model should see the reason and carry on, or `Error::middleware_failure(msg)` when the run must stop; any other error becomes an ordinary failed call. An agent with any agent-level middleware does not stream token by token: `run_stream` runs the chain to completion and replays the result. The [middleware page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/) covers ordering, `terminate`, what each context can change, common patterns (logging, rate limits, guardrails, redaction, caching), client decorators for per-request work, and the experimental agent hooks. The [Purview integration](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#purview-compliance) is built on the same traits.

## Context providers

A `ContextProvider` runs around every agent run. `before_run` receives a `SessionContext` and can add instructions (`add_instructions`), messages or tools for this run; `after_run` sees the request and response messages, and the error when the run failed. Providers attached with the builder's `context_provider` run for every session; providers attached to a session (`with_context_providers`) run only for that session, and run before the agent's.

```rust
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Remembers the user's name and reminds the model of it on later turns.
#[derive(Default)]
struct UserFacts {
    name: Mutex<Option<String>>,
}

#[async_trait]
impl ContextProvider for UserFacts {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        if let Some(name) = self.name.lock().unwrap().clone() {
            ctx.add_instructions(format!("The user's name is {name}."));
        }
        Ok(())
    }

    async fn after_run(
        &self,
        request: &[Message],
        _response: &[Message],
        _error: Option<&Error>,
    ) -> Result<()> {
        for message in request {
            if let Some(rest) = message.text().strip_prefix("My name is ") {
                *self.name.lock().unwrap() = Some(rest.trim_end_matches('.').to_string());
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let facts = Arc::new(UserFacts::default());
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .context_provider(facts.clone())
        .build();

    agent.run_once("My name is Ada.").await?; // no session: history is not kept
    println!("{}", agent.run_once("Greet me by name.").await?.text());
    println!("remembered: {:?}", facts.name.lock().unwrap());
    Ok(())
}
```

A provider instance attached to the builder is shared by every run of the agent, so anything it keeps in its own fields, as `UserFacts` does, is global. When one agent serves several users, keep per-user data in the session's `state` bag (reachable as `ctx.session`) instead. The crate ships these providers:

- **History**: `InMemoryHistoryProvider`, `FileHistoryProvider` (experimental), and the Redis and Cosmos DB stores ([sessions](#sessions-and-history)).
- **Retrieval**: `TextSearchProvider` wraps any async search function and adds its results to the run, either before every run (the default) or as a `Search` tool the model calls on demand (`TextSearchBehavior::OnDemandFunctionCalling`). A failed search is logged and the run continues.
- **Memory services**: Redis, Mem0, Azure AI Search and Foundry memory (see [memory and storage backends](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#memory-and-storage-backends)).
- **Skills** and **compaction**, below.
- With `experimental-harness`, a todo list, operating modes and file memory for long-running agents.

The [context and memory page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/) covers the exact order in which providers run and how the request is assembled, failures, per-session memory, retrieval, compaction, skills, the experimental agent harness and vector stores.

### Skills

A `Skill` is a named package of instructions and resources (`Skill::new(name, description).with_instructions(...).with_resource(name, text)`). `SkillsProvider::new(skills)` lists each skill's name and description in the instructions and gives the model `load_skill` and `read_skill_resource` tools, so a skill's full text is only sent once the model decides it needs it. Loaded skills belong to the provider instance, not to a session. Only inline skills built in code are ported; upstream's file-based skills, MCP-backed skills and skill scripts are not.

### Compaction

`AgentBuilder::with_compaction(strategy)` registers a `CompactionProvider` that trims the history sent to the model on each run. It runs after the session's history provider, so it sees the full history; the stored history itself is not changed, and the run's own input is never trimmed. The strategies are:

- `Truncation::new(n)`: leading system messages plus the newest messages, `n` in total.
- `SlidingWindow::new(n)`: leading system messages plus the newest `n` others.
- `TokenBudget::new(tokens)`: leading system messages plus the newest messages that fit a token budget, counted by `ApproxTokenizer` (four characters to a token) unless you register `CompactionProvider::with_tokenizer` yourself.
- `SelectiveToolResult::new(n)`: every message, with all but the newest `n` tool results replaced by a placeholder.

The first three accept `.preserve_first_user()` to keep the opening request however long the conversation grows, so `.with_compaction(SlidingWindow::new(20).preserve_first_user())` is a reasonable default for a chat assistant. Compaction never leaves a tool call without its result: an exchange that a strategy would cut in half is dropped whole. There is no summarising (LLM-backed) strategy, and no compaction between model calls inside one tool loop.

## Retries and errors

`RetryingChatClient` wraps any client with a `RetryPolicy`: exponential backoff with jitter, with a server's `Retry-After` honoured over the computed delay (capped at `max_delay`). The defaults are three retries, a 500 ms initial delay, a 30 s cap, a multiplier of 2.0 and jitter of 0.3. Wrap the client before handing it to the builder, so each model request of the tool loop is retried on its own:

```rust
use std::time::Duration;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = RetryingChatClient::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .with_policy(
            RetryPolicy::with_max_retries(5)
                .initial_delay(Duration::from_secs(1))
                .max_delay(Duration::from_secs(20)),
        );
    let agent = Agent::builder(client).build();

    match agent.run_once("Hello").await {
        Ok(response) => println!("{}", response.text()),
        Err(Error::ServiceStatus { status, retry_after, .. }) => {
            eprintln!("gave up: HTTP {status}, retry after {retry_after:?}")
        }
        Err(Error::ServiceInvalidAuth { message }) => eprintln!("check your key: {message}"),
        Err(Error::ServiceContentFilter { message, .. }) => eprintln!("blocked: {message}"),
        Err(other) => eprintln!("failed: {other}"),
    }
    Ok(())
}
```

The default rule (`RetryOn::Default`) retries `Error::ServiceStatus` with status 408, 429 or 5xx, and transport failures reported as `Error::Service`. It never retries `ServiceInvalidAuth` (401/403), `ServiceInvalidRequest` (400/404/422) or `ServiceContentFilter`, because repeating them gives the same answer. `RetryOn::predicate(|err| ...)` replaces the rule. The clients set no request timeout of their own, so bound a call with `tokio::time::timeout` where that matters.

Every fallible call returns `agent_framework::Result<T>`. The `Error` enum is `#[non_exhaustive]`, so a `match` needs a catch-all arm; besides the service variants above it has `Service`, `Tool`, `ToolRejected`, `MiddlewareFailure`, `AgentExecution`, `Configuration` (for example a missing environment variable in `from_env`), `Workflow`, `Serialization`, `Json` and `Other`, among others. `Error::status()` and `Error::retry_after()` read the HTTP details without a match. The [middleware page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/#retries) has the full policy options, where the retry layer belongs relative to other decorators, retrying tools, and the complete error table.

## Workflows

The `workflow` module is a graph engine in the style of Python's `WorkflowBuilder`. A workflow is a set of executors joined by edges, run in supersteps: every message sent during a superstep is delivered at the start of the next one, and the executors scheduled in one superstep run concurrently. A run ends when no messages remain, pauses when an executor waits for outside input, and fails when an executor returns an error or the run exceeds its superstep limit (100 by default).

The main difference from Python and .NET is typing. Every message, output and request payload is a `serde_json::Value`: there are no typed handlers, no routing by message type, and no type checks between connected executors, so a shape mismatch shows up at run time. In return, checkpoints are plain JSON and nothing needs registering for serialisation.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let upper = FunctionExecutor::new("upper", |message, ctx| async move {
        let text = message.as_str().unwrap_or_default().to_uppercase();
        ctx.send_message(json!(text)).await?;
        Ok(())
    });
    let count = FunctionExecutor::new("count", |message, ctx| async move {
        let text = message.as_str().unwrap_or_default();
        ctx.yield_output(json!({ "text": text, "words": text.split_whitespace().count() }))
            .await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(upper))
        .add_executor(Arc::new(count))
        .set_start("upper")
        .add_edge("upper", "count")
        .build()?;

    let run = workflow.run("graphs all the way down").await?;
    println!("state:  {:?}", run.state());
    println!("output: {}", run.last_output().unwrap());
    Ok(())
}
```

```text
state:  Idle
output: {"text":"GRAPHS ALL THE WAY DOWN","words":5}
```

`FunctionExecutor` is the counterpart of Python's `@executor`; implement the `Executor` trait for an executor with its own state. Beyond plain edges, the builder has conditional edges, switch-case, fan-out, multi-selection and fan-in. An executor can read and write shared state, raise a request for human input (the run pauses as `IdleWithPendingRequests` until `send_response`), and a workflow can save a checkpoint after every superstep and resume from it, in the same process or after a restart. `AgentExecutor` puts an agent in the graph, a whole workflow can be a node of another, and `workflow.as_agent(name)` wraps a workflow as an agent. The [workflows page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/) covers each of these, along with events and streaming, validation and visualisation. Runs cannot be cancelled in this release, and functional (`@workflow`) workflows are not ported.

## Orchestrations

The orchestration builders assemble a complete workflow from a list of agents. Each takes `Arc<dyn SupportsAgentRun>` participants (an `Agent`, a workflow agent, an A2A or Copilot Studio agent, or your own type) and returns an ordinary `Workflow`, whose output is the conversation as a JSON array of `Message`.

| Builder | Pattern |
| --- | --- |
| `SequentialBuilder` | Each participant sees the conversation so far and appends its reply |
| `ConcurrentBuilder` | Every participant answers the same input at once |
| `GroupChatBuilder` | A manager (round-robin by default, a closure, a `GroupChatManager` or an agent) picks the next speaker each round |
| `HandoffBuilder` | Agents pass control by calling `handoff_to_<name>` tools |
| `MagenticBuilder` | A manager plans, tracks progress in a ledger and assigns work |

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let writer = Agent::builder(client.clone())
        .name("writer")
        .instructions("Write a first draft paragraph on the topic.")
        .build();
    let editor = Agent::builder(client)
        .name("editor")
        .instructions("Improve the previous draft for clarity.")
        .build();

    let pipeline = SequentialBuilder::new()
        .add(Arc::new(writer))
        .add(Arc::new(editor))
        .build()?;

    let run = pipeline.run("Rust for embedded systems").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    for m in &conversation {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
    Ok(())
}
```

Two differences from Python matter when writing orchestration code. The handoff builder does not add handoff tools to your agents, because a built agent's tool list is fixed: attach `handoff_tool_spec(target, description)` to each participant yourself. And group chat, handoff and Magentic each run as a single executor that calls its participants in a loop, so their events carry one executor id and participants are told apart by `author_name`. The [orchestrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/) covers each builder, approval gates, plan review, stall intervention and streaming.

## MCP, hosting and integrations

The remaining pieces connect an agent to other systems, each behind its own cargo feature:

- **MCP** (`mcp`): `McpStdioTool`, `McpStreamableHttpTool` and `McpWebsocketTool` connect to an MCP server and turn its tools into `ToolDefinition`s, so they go through the same tool loop, middleware and approvals as function tools. The crate has no MCP server; the MCP page shows [a minimal hand-written one](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/#exposing-an-agent-as-an-mcp-server). See the [MCP page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/) for transports, filtering, prompts, sampling and the low-level client.
- **Hosting** (`hosting`, `a2a`): `AgentHost` and the routers in `agent_framework::hosting` serve agents and workflows over axum, as DevUI-style, OpenAI-compatible (Chat Completions and Responses), AG-UI and A2A endpoints; `A2AAgent` calls a remote A2A agent as a participant. Azure Functions, Durable Task and Foundry hosted-agent hosting are not ported. See the [hosting page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/), including [securing and composing](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/#securing-and-composing) the routers.
- **Integrations**: declarative YAML and JSON agents and workflows (`experimental-declarative-agents`), Redis, Mem0, Cosmos DB, Azure AI Search and Foundry memory and storage, Purview data-loss prevention, Copilot Studio agents, and the local and Docker shell tools (`tools`, as `agent_framework::builtin_tools`). See the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/).

## Observability

The agent emits an `invoke_agent` span per run and the tool loop an `execute_tool` span per call. Wrap the client in `ObservableChatClient::new(client, "openai")` to add a `chat` span per model request, with model, token-usage and finish-reason attributes. All follow the OpenTelemetry GenAI semantic conventions and are ordinary `tracing` spans, so any `tracing` subscriber sees them. Message content is recorded only when `ENABLE_SENSITIVE_DATA` is set or `with_content_capture(true)` is called. The `otel-metrics` and `otel-export` features add GenAI metrics and a ready-made OTLP exporter; the [observability page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/) lists every span attribute and covers [exporting with `otel-export`](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/#exporting-with-otel-export) and wiring to a collector.

## Testing

The crate ships no mock chat client, and you do not need one: a `ChatClient` that replays scripted replies takes about a hundred lines, and everything above it (the tool loop, approvals, sessions, middleware, retries, workflows and orchestrations) then runs for real in `cargo test` with no network. The [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/) builds that client and tests each layer with it, then covers production settings: timeouts, retries, cancellation, concurrency, secrets and shutdown.

## Settings and secrets

`SecretString` wraps a key so that `Debug` and `Display` print a mask rather than the value (`expose_secret()` returns it). `load_setting(key, explicit, default)` resolves a setting from an explicit value, then a `.env` file in the working directory, then the environment variable, then the default. Provider clients read their own environment variables in `from_env`, so these helpers matter mostly for your own configuration code and custom providers.

## Experimental agent APIs

Some agent features carry no compatibility promise and compile only with their `experimental-*` feature (the `experimental` feature turns them all on, and `full` includes none of them):

- `experimental-harness`: `LoopAgent` (re-runs an agent until a `Judge` or a todo list says it is done), `TodoProvider`, `ToolApprovalAgent` (standing approval rules), `AgentModeProvider`, and file memory over an `AgentFileStore`. See [context and memory](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/#the-agent-harness) and, for `ToolApprovalAgent`, [tools](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#standing-approvals-with-toolapprovalagent).
- `experimental-agent-hooks`: interception at the agent, model-call and tool-call seams, with allow, deny and transform verdicts ([middleware](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/#agent-hooks-experimental)).
- `experimental-progressive-tools`: add or remove tools mid-run from function middleware through `FunctionInvocationContext` ([tools](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/#progressive-tools-experimental)).
- `experimental-session-store` and `experimental-file-history`: whole-session persistence and a file-backed history provider ([sessions](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/)).
- `experimental-vector-stores`: the vector store abstraction and vector collection tools ([context and memory](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/#vector-stores)), and the Cosmos DB and Azure AI Search stores ([integrations](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#vector-stores)).
- `experimental-declarative-agents`: the declarative crate ([integrations](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#declarative-agents-and-workflows)).

Each can change in any minor release. The crate documentation on docs.rs is built with all features and labels each gated item with the feature it needs.

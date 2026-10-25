---
title: "Microsoft Agent Framework (Rust) — Comprehensive guide"
description: "Agents, runs and streaming, chat-client providers, sessions and persistence, structured output, function tools and approvals, middleware, context providers, skills, compaction, retries and errors in the community Rust port agent-framework-rs."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Comprehensive guide"
  order: 2
---

This page covers the single-agent surface of the `agent-framework` crate: building and running agents, choosing a provider, sessions, structured output, tools, middleware, context providers and error handling. Workflows and orchestrations have [their own page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/), as do [MCP, A2A, hosting and memory backends](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/) and [testing](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/).

`agent-framework` is a community port maintained in [codehalwell/agent-framework-rs](https://github.com/codehalwell/agent-framework-rs), not a Microsoft package (see the [overview](/microsoft-agent-framework-guide/rust/)). Every program below was compiled against version %%version=agent-framework-rs%% from crates.io. The ones that call OpenAI were run against a local stand-in for the OpenAI API; the rest need no network.

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

`schemars` must be the 0.8 series, which is the version the crate derives tool schemas with. The crate's minimum supported Rust version is 1.88.

## Crate map

The umbrella crate re-exports `agent-framework-core` at its root, so `agent_framework::types`, `agent_framework::workflow` and so on are the core modules. Each companion crate appears as a module when its feature is on (`agent_framework::anthropic`, `agent_framework::mcp`, ...), and `agent_framework::prelude` gathers the common names from all of them.

| Module | Purpose |
| --- | --- |
| `agent` | `Agent`, `AgentBuilder`, `AgentRunOptions`, the `SupportsAgentRun` trait, `AsToolOptions` |
| `client` | The `ChatClient` and `EmbeddingClient` traits, `FunctionInvokingChatClient`, `RetryingChatClient`, `RetryPolicy` |
| `types` | `Message`, `Role`, `Content` and its variants, `ChatOptions`, `ChatResponse`, `AgentResponse`, `ResponseFormat`, `ToolMode` |
| `tools` | `FunctionTool`, `ToolDefinition`, `ApprovalMode`, `FunctionInvocationConfig`, the `Tool` and `ToolSource` traits, hosted-tool markers |
| `session`, `history` | `AgentSession`, `SessionState`, `InMemoryHistoryProvider`, `HistoryProvider` |
| `memory` | The `ContextProvider` trait and `SessionContext` |
| `middleware` | `Middleware<C>`, `Next`, and the three contexts `AgentContext`, `ChatContext`, `FunctionInvocationContext` |
| `skills`, `text_search`, `compaction` | Skills, retrieval over any search function, history compaction strategies |
| `observability` | `ObservableChatClient` and OpenTelemetry GenAI spans |
| `workflow` | The graph engine and the prebuilt orchestrations |
| `error` | `Error` and `Result` |

Python users will recognise the layout: the port follows `agent_framework`'s module structure, so `_types` became `types`, `_middleware` became `middleware`, and so on.

## Agents and runs

`Agent::builder(client)` takes any `ChatClient`. The builder sets a name, instructions, default options and tools, and `build()` returns an `Agent`. The builder wraps the client in a `FunctionInvokingChatClient`, so local tools run automatically; you never wrap it yourself.

There are four ways to run an agent:

- `agent.run_once("text")` runs a single turn with no session.
- `agent.run(messages, Some(&mut session))` runs with a `Vec<Message>` and an optional session. It is a method of the `SupportsAgentRun` trait, which the prelude imports.
- `agent.run_with_options(messages, session, options)` applies per-run overrides (see [per-run options](#per-run-options)).
- `agent.run_stream(...)` and `agent.run_stream_once(...)` return a stream of updates.

All return `Result<AgentResponse>` (or a stream). `AgentResponse::text()` concatenates the assistant text; `messages`, `usage_details`, `finish_reason` and `value` hold the rest.

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

`finish_reason` deserves a check in production code. A response cut off by the token limit (`length`) or an Azure content filter (`content_filter`) still has text that reads like a finished answer; the finish reason is the only reliable signal.

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

Other variants include `FunctionCall`, `FunctionResult`, `FunctionApprovalRequest`/`FunctionApprovalResponse`, `TextReasoning`, `Usage`, `HostedFile` and the MCP server-tool call and result types. OpenAI and Anthropic convert images, audio and files; check a provider's crate documentation before relying on a modality.

## Providers

Every provider is a `ChatClient`, so the agent code is the same whichever you pick. Each lives behind a cargo feature; only `openai` is on by default.

| Provider | Type (module) | Feature | `from_env` reads |
| --- | --- | --- | --- |
| OpenAI Chat Completions, or any compatible endpoint | `OpenAIChatCompletionClient` (`openai`) | `openai` | `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` |
| OpenAI Responses API | `OpenAIChatClient` (`openai`) | `openai` | the same |
| Azure OpenAI Chat Completions | `AzureOpenAIClient` (`azure`) | `azure` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, optional `AZURE_OPENAI_API_VERSION` |
| Azure OpenAI Responses API | `AzureOpenAIResponsesClient` (`azure`) | `azure` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME`, optional `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_BASE_URL` |
| Microsoft Foundry project (Responses API) | `FoundryChatClient`, `FoundryAgent` (`foundry`) | `foundry` | `FOUNDRY_ENDPOINT` or `FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_MODEL`, optional `FOUNDRY_API_KEY` (otherwise `DefaultAzureCredential`) |
| Anthropic Messages API | `AnthropicClient` (`anthropic`) | `anthropic` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL` |
| Claude on Bedrock, Vertex AI or Foundry | `AnthropicBedrockClient`, `AnthropicVertexClient`, `AnthropicFoundryClient` (`anthropic`) | `anthropic` | per client |
| Google Gemini | `GeminiChatClient` (`gemini`) | `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY`, optional `GEMINI_BASE_URL` |
| Mistral | `MistralChatClient` (`mistral`) | `mistral` | `MISTRAL_API_KEY`, optional `MISTRAL_BASE_URL` |
| Ollama | `OllamaChatClient` (`ollama`) | `ollama` | optional `OLLAMA_HOST` (default `http://localhost:11434/v1`) |
| Foundry Local | `FoundryLocalChatClient` (`foundry_local`) | `foundry-local` | optional `FOUNDRY_LOCAL_ENDPOINT`, `FOUNDRY_LOCAL_API_KEY` |
| Amazon Bedrock Converse API | `BedrockChatClient` (`bedrock`) | `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` or `AWS_DEFAULT_REGION` (default `us-east-1`), optional `AWS_SESSION_TOKEN` |
| GitHub Copilot | `GitHubCopilotChatClient` (`github_copilot`) | `github-copilot` | `GITHUB_COPILOT_TOKEN` or `GH_COPILOT_TOKEN` |
| Copilot Studio agent | `CopilotStudioAgent` (`copilotstudio`) | `copilotstudio` | an agent, not a chat client |

Every client also has a `new(...)` constructor that takes its settings directly, and most have `with_base_url`. Embedding clients exist for OpenAI, Azure OpenAI, Foundry, Ollama, Gemini, Mistral and Bedrock.

Two OpenAI clients exist because the APIs differ. `OpenAIChatCompletionClient` speaks `/v1/chat/completions`, which every OpenAI-compatible server (vLLM, LM Studio, gateways) also speaks. `OpenAIChatClient` speaks the Responses API, which supports hosted tools, reasoning items and service-side conversation state: by default it returns a response id that the agent stores on the session, so later turns send `previous_response_id` instead of replaying the history.

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

Notes on individual providers:

- **Azure and Entra ID.** The `azure` crate implements its own credential chain: `AzureCliCredential`, `ClientSecretCredential`, `EnvironmentCredential`, `ManagedIdentityCredential`, `WorkloadIdentityCredential`, `ChainedTokenCredential` and `DefaultAzureCredential`. `DefaultAzureCredential` tries environment, workload identity, managed identity and Azure CLI credentials in that order; the IDE, PowerShell, Azure Developer CLI and interactive credentials of the Azure SDKs are not implemented.
- **Anthropic.** The Messages API requires `max_tokens` on every request, so the client sends a default when `ChatOptions::max_tokens` is unset (`with_max_tokens` changes it). Structured output is supported by folding the schema into the system prompt, since the API has no `response_format` field.
- **Ollama** uses Ollama's OpenAI-compatible endpoint rather than its native API.
- **Foundry.** `FoundryAgent` builds a Foundry prompt agent on the client side and runs it over the Responses API. It cannot attach to an agent already hosted in a Foundry project.
- **Streaming on Bedrock and Vertex.** For Claude on Bedrock or Vertex AI, the streaming methods currently make one non-streaming request and return it as a single update.

### Writing your own client

`ChatClient` has two required methods, `get_response` and `get_streaming_response`, plus optional `model()` and `service_url()`. Implementing it is how you reach an in-house gateway or build a test double (the [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/) has a complete one). For tool calling to work, return tool requests as `Content::FunctionCall`; the function-invocation loop does the rest. Return classified errors (`Error::service_status`, `Error::service_invalid_auth` and so on, see [errors](#retries-and-errors)) so that retries behave.

## Sessions and history

An `AgentSession` carries a `session_id`, an optional service-side conversation id, a free-form `state` bag and a list of context providers. History is not stored on the session itself: it lives in a history provider, which is an ordinary [context provider](#context-providers). When you pass a session that has no history provider, the agent attaches an `InMemoryHistoryProvider` to it on first use, so multi-turn conversations need no setup:

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

When the service keeps the conversation (a session with a service id, or a Responses API reply that returned one), the agent leaves history providers out of the run, so earlier turns are not sent twice (releases before 0.14.0 sent them twice).

### Persisting a session

`AgentSession::to_dict()` serialises the id, service id and state bag, but not the history. To save a conversation, attach the history provider yourself, keep a handle to it, and serialise both:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?).build();

    // Turn 1, with a history provider we hold on to.
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session = AgentSession::new()
        .with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    session.state.insert("tier", json!("pro"));
    agent
        .run(vec![Message::user("Remember: my project is called skylark.")], Some(&mut session))
        .await?;

    // Save: two independent JSON values.
    let saved = json!({ "session": session.to_dict(), "history": history.to_dict() });
    let text = serde_json::to_string(&saved)?;

    // Later, possibly in another process: restore both and reattach.
    let saved: serde_json::Value = serde_json::from_str(&text)?;
    let history = Arc::new(InMemoryHistoryProvider::from_dict(&saved["history"])?);
    let mut session = agent
        .session_from_dict(&saved["session"])?
        .with_context_providers(vec![history as Arc<dyn ContextProvider>]);
    assert_eq!(session.state.get("tier"), Some(json!("pro")));

    let reply = agent
        .run(vec![Message::user("What is my project called?")], Some(&mut session))
        .await?;
    println!("{}", reply.text());
    Ok(())
}
```

For durable storage, swap the in-memory provider for one that writes through: `RedisChatMessageStore` and `CosmosChatMessageStore` are history providers (see [memory backends](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#memory-and-storage-backends)), and `FileHistoryProvider` (feature `experimental-file-history`) keeps the history in a local JSON file, rewritten atomically after each run. The `experimental-session-store` feature adds an `AgentSessionStore` trait with in-memory and file implementations that save and restore whole sessions, including in-memory history, under a key.

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

`parse_json` reads the text of the last assistant message that has any, ignoring reasoning content, so a reasoning model's chain of thought does not break the parse. There is no derive-based helper that turns a Rust type into a response format; generate the schema with `schemars` (`serde_json::to_value(schemars::schema_for!(Recipe))`) if you prefer that to writing it by hand. `ResponseFormat::JsonObject` asks for any valid JSON object without a schema.

## Function tools

A `FunctionTool` pairs a name, a description, a JSON Schema for the arguments and an async closure. Two constructors exist:

- `FunctionTool::new(name, description, schema, |args: Value| async move { ... })` takes a hand-written schema and gives the closure a `serde_json::Value`.
- `FunctionTool::typed(name, description, |args: T| async move { ... })` derives the schema from a type that implements `Deserialize` and `schemars::JsonSchema`, and deserialises the arguments into it. Doc comments on fields become schema descriptions. Arguments that do not deserialise produce a tool error, which the model sees as a failed call.

Either way, `.into_definition()` turns the tool into the `ToolDefinition` that `AgentBuilder::tool` accepts.

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

When the model asks for a tool, the loop runs it, appends the result, and calls the model again until it answers in text. Independent calls in one model turn run concurrently. Tool errors do not fail the run: the model receives `"tool execution failed"` (or the error text, with `include_detailed_errors`) and can try again or explain. Two cases stop the run instead: `Error::MiddlewareFailure` from [function middleware](#middleware), and a call to an unknown tool when `terminate_on_unknown_calls` is set.

A `FunctionTool` can also cap its own use with `max_invocations(n)` and `max_invocation_exceptions(n)`; the counters are shared between clones of the tool, so they apply across runs.

### Bounding the tool loop

`FunctionInvocationConfig` bounds the loop three ways, and the builder's `function_invocation_config` is the only place to set it:

| Field | Default | Meaning |
| --- | --- | --- |
| `max_iterations` | 40 | Model round trips per request |
| `max_function_calls` | `None` | Total tool executions per request, across iterations and approval round trips |
| `max_duration_seconds` | `None` | Wall-clock budget for the loop, including time spent waiting for approvals |
| `max_consecutive_errors_per_request` | 3 | After more consecutive rounds with a failed call than this, tools are switched off for the request |
| `terminate_on_unknown_calls` | `false` | Fail the run when the model calls a tool that does not exist |
| `include_detailed_errors` | `false` | Send the tool's error text to the model instead of a generic message |
| `allow_concurrent_invocation` | `true` | Run the calls of one model turn concurrently |

When `max_function_calls` or `max_duration_seconds` runs out, the run does not fail: tools are switched off and the model is asked to answer with what it has. Both budgets are checked after each batch of calls, so one batch can overshoot.

`ToolMode` controls whether the model must call a tool. Set it per run through `ChatOptions::with_tool_choice`: `ToolMode::auto()`, `ToolMode::none()`, `ToolMode::required_any()` or `ToolMode::required_function("name")`.

### Hosted tools

`hosted_web_search()`, `hosted_code_interpreter()`, `hosted_file_search(max_results)`, `hosted_image_generation()` and `hosted_mcp(...)` return marker `ToolDefinition`s that the provider runs on its side. They only work with providers that support them (the OpenAI Responses API, Azure OpenAI Responses, Foundry, and Anthropic for its own server tools); a Chat Completions endpoint does not.

### An agent as a tool

`agent.as_tool(AsToolOptions::new().name(...).description(...))` exposes one agent to another as a function tool that takes a single `task` string. With `propagate_session(true)` the sub-agent receives a child session that shares the caller's state bag but not its service conversation id.

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;

    let researcher = Agent::builder(client.clone())
        .name("researcher")
        .instructions("You answer factual questions in one sentence.")
        .build();

    let coordinator = Agent::builder(client)
        .name("coordinator")
        .instructions("Delegate factual questions to the researcher tool.")
        .tool(researcher.as_tool(
            AsToolOptions::new()
                .name("ask_researcher")
                .description("Ask the research specialist a factual question."),
        ))
        .build();

    println!("{}", coordinator.run_once("When was Rust 1.0 released?").await?.text());
    Ok(())
}
```

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

Each approval response carries the function call it approves, and the loop's call and time budgets are parked in the session's state while the run waits, so an approval can be answered in another process as long as you persist the session (state bag and history) in between. Answering a request twice does not run the call twice. Approval also exists at the workflow level (`with_request_info()` on the sequential and concurrent builders) and for MCP servers (`McpApprovalMode`).

## Per-run options

`AgentRunOptions` overrides the agent's defaults for a single call. Its `ChatOptions` are merged over the build-time options: per-run values win for scalar fields, map fields (`metadata`, `logit_bias`) merge per key, tools are combined by name, and instructions are appended to the agent's on a new line rather than replacing them. `with_tool` adds tools for that run only. A tool with no executor (a declaration only) is returned to the caller as a function call instead of being run, which is how hosting surfaces such as AG-UI forward client-side tools.

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

`ChatOptions` covers the usual fields (`model`, `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `tool_choice`, `response_format`, `conversation_id`, `store`, `metadata`, `user`) plus `additional_properties` for provider-specific values.

## Middleware

Middleware runs at three levels, and each level has its own context type:

| Level | Builder method | Context | Wraps |
| --- | --- | --- | --- |
| Agent | `.middleware(...)` | `AgentContext` (`messages`, `result`, `terminate`, `metadata`, `is_streaming`) | The whole run |
| Chat | `.chat_middleware(...)` | `ChatContext` (`messages`, `chat_options`, `result`, `terminate`, ...) | The call to the chat client, once per run, around the whole tool loop |
| Function | `.function_middleware(...)` | `FunctionInvocationContext` (`function_name`, `arguments`, `session`, `result`, `terminate`, ...) | Each tool call |

Chat middleware differs from Python here. Python's chat middleware wraps each model request, so it sees every round trip of the tool loop; in the Rust port the agent applies chat middleware outside the function-invocation loop, so it runs once per run, sees the messages before the first model call, and receives the final response after the last. On a streaming run it can only change the request: `result` is empty after `next.run` and short-circuiting is ignored, unless the agent also has agent-level middleware (that path runs to completion and replays the result as updates). To act on every model request, wrap the client in your own `ChatClient` instead, as `RetryingChatClient` and `ObservableChatClient` do.

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

/// Chat level: log each model call.
struct LogCalls;

#[async_trait]
impl Middleware<ChatContext> for LogCalls {
    async fn process(&self, ctx: ChatContext, next: Next<ChatContext>) -> Result<ChatContext> {
        println!("[chat] sending {} message(s)", ctx.messages.len());
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
        .chat_middleware(Arc::new(LogCalls))
        .function_middleware(Arc::new(NoProduction))
        .build();

    println!("{}", agent.run_once("What is the admin password?").await?.text());
    println!("{}", agent.run_once("Deploy to staging.").await?.text());
    println!("output tokens so far: {}", guardrail.output_tokens.load(Ordering::Relaxed));
    Ok(())
}
```

For a refused tool call, return `Error::tool_rejected(msg)` when the model should see the reason and carry on, or `Error::middleware_failure(msg)` when the run must stop; any other error becomes an ordinary failed call. The same three middleware traits are what the [Purview integration](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#purview-compliance) uses.

## Context providers

A `ContextProvider` runs around every agent run. `before_run` receives a `SessionContext` and can add instructions (`add_instructions`), messages or tools for this run; `after_run` sees the request and response messages, on failure as well as success. Providers attached with the builder's `context_provider` run on every session; providers attached to a session (`with_context_providers`) run only for that session, and run first.

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

A provider instance attached to the builder is shared by every run of the agent; keep per-user data in the session's `state` (reachable as `ctx.session`) when one agent serves several users. The crate ships these providers:

- **History**: `InMemoryHistoryProvider`, `FileHistoryProvider` (experimental), and the Redis and Cosmos stores.
- **Retrieval**: `TextSearchProvider` wraps any async search function and adds its results to the run, either before every run (the default) or as a `Search` tool the model calls on demand (`TextSearchBehavior::OnDemandFunctionCalling`). A failed search is logged and the run continues.
- **Memory services**: Redis, Mem0, Azure AI Search and Foundry memory (see [memory backends](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#memory-and-storage-backends)).
- **Skills** and **compaction**, below.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::text_search::{TextSearchProvider, TextSearchResult};

#[tokio::main]
async fn main() -> Result<()> {
    // Any async function from a query to results: a vector store, a web search, SQL.
    let search = TextSearchProvider::new(|query: String| async move {
        let mut hits = Vec::new();
        if query.to_lowercase().contains("return") {
            hits.push(
                TextSearchResult::new("Items can be returned within 30 days with a receipt.")
                    .with_source("Returns policy", "https://example.com/returns"),
            );
        }
        Ok(hits)
    });

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Answer from the provided context and cite the source.")
        .context_provider(Arc::new(search))
        .build();

    println!("{}", agent.run_once("What is your returns policy?").await?.text());
    Ok(())
}
```

### Skills

A `Skill` is a named package of instructions and resources. `SkillsProvider` lists each skill's name and description in the instructions and gives the model `load_skill` and `read_skill_resource` tools, so the full text is only loaded when the model decides it needs it. Only inline skills built in code are supported; upstream's file-based skills and skill scripts are not ported.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let skills = vec![
        Skill::new("code-review", "Review code changes for defects and style")
            .with_instructions("Check error handling first, then naming, then tests.")
            .with_resource("checklist", "1. unwrap() in library code  2. missing context"),
        Skill::new("release-notes", "Draft release notes from a changelog")
            .with_instructions("Group by area and lead with breaking changes."),
    ];

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are an engineering assistant.")
        .context_provider(Arc::new(SkillsProvider::new(skills)))
        .build();

    println!("{}", agent.run_once("Review this: fn f() { x.unwrap() }").await?.text());
    Ok(())
}
```

### Compaction

`AgentBuilder::with_compaction(strategy)` trims the message list sent to the model on each run. It runs after the history provider, so it sees the full history; the stored history itself is not changed. The strategies are:

- `Truncation::new(n)`: keep the most recent `n` messages plus any leading system messages.
- `SlidingWindow::new(n)`: keep leading system messages plus the last `n` other messages.
- `TokenBudget::new(tokens)`: keep the newest messages that fit a token budget, counted by `ApproxTokenizer` unless you supply a `Tokenizer` through `CompactionProvider::with_tokenizer`.
- `SelectiveToolResult::new(n)`: replace the payload of all but the last `n` tool results with a placeholder, keeping the call and result pairing that providers require.

The first three accept `.preserve_first_user()` to keep the opening request however long the conversation grows. Compaction never leaves a tool call without its result: an exchange that a strategy would cut in half is dropped whole. There is no summarising (LLM-backed) strategy.

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .with_compaction(SlidingWindow::new(20).preserve_first_user())
        .build();

    let mut session = agent.create_session();
    for i in 0..3 {
        let reply = agent
            .run(vec![Message::user(format!("Message {i}"))], Some(&mut session))
            .await?;
        println!("{}", reply.text());
    }
    Ok(())
}
```

## Retries and errors

`RetryingChatClient` wraps any client with a `RetryPolicy`: exponential backoff with jitter, a server's `Retry-After` honoured over the computed delay (capped at `max_delay`). The defaults are three retries, 500 ms initial delay, a 30 s cap, a multiplier of 2.0 and jitter of 0.3. Wrap the client before handing it to the builder:

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

The default rule (`RetryOn::Default`) retries `Error::ServiceStatus` with status 408, 429 or 5xx, and transport failures reported as `Error::Service`. It never retries `ServiceInvalidAuth` (401/403), `ServiceInvalidRequest` (400/404/422) or `ServiceContentFilter`, because repeating them gives the same answer. `RetryOn::predicate(|err| ...)` replaces the rule.

The `Error` enum also has `AgentExecution`, `Tool`, `ToolRejected`, `MiddlewareFailure`, `Workflow`, `Configuration` (for example a missing environment variable in `from_env`), `Serialization`, `Json` and `Other`. `Error::status()` and `Error::retry_after()` read the HTTP details without a match.

## Settings and secrets

`SecretString` wraps a key so that `Debug` and `Display` print a mask rather than the value (`expose_secret()` returns it). `load_setting(key, override, default)` resolves a setting from an explicit value, then a `.env` file in the working directory, then the environment variable, then the default. Provider clients read their own environment variables in `from_env`, so these helpers matter mostly for your own configuration code.

## Observability

The agent emits an `invoke_agent` span per run and the tool loop an `execute_tool` span per call. Wrap the client in `ObservableChatClient::new(client, "openai")` to add a `chat` span per model call, with model, token-usage and finish-reason attributes. All follow the OpenTelemetry GenAI semantic conventions and are ordinary `tracing` spans. Message content is recorded only when `ENABLE_SENSITIVE_DATA` is set or `with_content_capture(true)` is called. Exporting the spans and the optional metrics is covered on the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#observability-and-opentelemetry).

## Experimental agent APIs

Some agent features carry no compatibility promise and compile only with their `experimental-*` feature:

- `experimental-harness`: `LoopAgent` (re-runs an agent until a `Judge` or a todo list says it is done), `TodoProvider`, `ToolApprovalAgent` (standing approval rules), `AgentModeProvider`, and file memory over an `AgentFileStore`.
- `experimental-agent-hooks`: interception at the agent, model-call and tool-call seams, with allow, deny and transform verdicts.
- `experimental-progressive-tools`: add or remove tools mid-run from function middleware through `FunctionInvocationContext`.
- `experimental-session-store` and `experimental-file-history`, described under [sessions](#persisting-a-session).

Each can change in any minor release. Read the crate documentation on docs.rs (built with all features) before depending on one.

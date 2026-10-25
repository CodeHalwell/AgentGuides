---
title: "Microsoft Agent Framework (Rust) — MCP, A2A, hosting and memory"
description: "agent-framework-rs integrations: MCP tools over stdio, HTTP and WebSocket; calling and serving A2A agents; DevUI-style, OpenAI-compatible and AG-UI hosting with axum; declarative YAML agents and workflows; Redis, Mem0, Cosmos DB, Azure AI Search and Foundry memory; Purview; shell tools; OpenTelemetry."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "MCP, A2A, hosting and memory"
  order: 4
---

The integrations live in companion crates, each behind a cargo feature of the `agent-framework` crate. The examples were compiled against version %%version=agent-framework-rs%%. The MCP stdio, A2A, hosting, declarative, Redis, shell-tool and OpenTelemetry examples were run locally, with a stand-in for the OpenAI API and an OTLP endpoint, a local Redis and the reference MCP server; the remote MCP, Azure and Purview ones need real services and were compiled only.

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["mcp", "a2a", "hosting", "redis"] }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
axum = "0.8"  # only to serve the hosting routers
```

## MCP tools

The `mcp` feature adds three MCP clients with the same builder methods: `McpStdioTool` (spawns a server process), `McpStreamableHttpTool` (an HTTP endpoint, JSON or SSE replies) and `McpWebsocketTool` (`ws://` or `wss://`). They speak protocol version 2025-06-18 and accept a server that negotiates down to an older revision. Newer revisions are not negotiated yet.

There are two ways to give a server's tools to an agent:

- `.tool_source(Arc::new(mcp))` registers the server as a `ToolSource`. The agent connects on its first run and resolves the tool list on every run, from a cache that is refreshed when the server sends `notifications/tools/list_changed`. A connection failure fails the run.
- `.tools(mcp.tool_definitions().await?)` lists the tools once and freezes them into the agent.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mcp = Arc::new(
        McpStdioTool::new("everything", "npx")
            .args(["-y", "@modelcontextprotocol/server-everything"])
            // Only these tools are offered to the model.
            .allowed_tools(["echo", "get-sum"]),
    );

    // Listing is optional; the agent resolves tools itself on each run.
    for tool in mcp.tool_definitions().await? {
        println!("{}: {}", tool.name, tool.description);
    }

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Use the tools when they help.")
        .tool_source(mcp.clone())
        .build();
    println!("{}", agent.run_once("Echo back: hello from MCP").await?.text());

    mcp.close().await?;
    Ok(())
}
```

A child process started by `McpStdioTool` inherits only a small set of environment variables (`PATH`, `HOME`, the temporary-directory and locale variables, and their Windows equivalents) unless you say otherwise: pass secrets with `.env([(key, value)])` or `inherit_env_var(name)`, or turn on `inherit_parent_environment(true)`. Proxy settings are not in the default set, so an `npx` or `uvx` server that downloads its package behind a proxy needs `inherit_env_vars(["HTTPS_PROXY", "NO_PROXY"])`. Other builder methods:

| Method | Purpose |
| --- | --- |
| `approval_mode(McpApprovalMode::...)` | `always_require_all()`, `never_require_all()` (default) or `per_tool(["name", ...])`; approval then works as for [local tools](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_comprehensive_guide/#tool-approvals) |
| `headers([...])` (HTTP, WebSocket) | Authentication headers such as `Authorization` |
| `request_timeout(...)` (stdio, WebSocket), `timeout(...)` (HTTP) | Per-request timeout |
| `sampling_handler(chat_client_sampling_handler(client))` | Answer the server's `sampling/createMessage` requests with your own model |
| `roots([Root::new("file:///workspace").with_name("Workspace")])` | Answer `roots/list` |
| `load_tools(false)`, `load_prompts(false)` | Skip a category entirely |

Prompts are available with `mcp.prompts().await?` and `mcp.get_prompt(name, args).await?`, which returns the prompt as `Vec<Message>` ready to pass to `agent.run`. For lower-level access, `McpClient` takes a transport directly and exposes `list_tools`, `call_tool`, `list_prompts` and `ping`.

```rust
use std::sync::Arc;

use agent_framework::mcp::{chat_client_sampling_handler, McpApprovalMode, McpStreamableHttpTool};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let token = std::env::var("MCP_TOKEN").unwrap_or_default();

    let github = McpStreamableHttpTool::new("github", "https://mcp.example.com/mcp")
        .headers([("Authorization", format!("Bearer {token}"))])
        .approval_mode(McpApprovalMode::per_tool(["create_issue", "merge_pull_request"]))
        .sampling_handler(chat_client_sampling_handler(Arc::new(client.clone())));

    let agent = Agent::builder(client)
        .tool_source(Arc::new(github))
        .build();
    let response = agent.run_once("List my open pull requests.").await?;
    for request in response.user_input_requests() {
        println!("needs approval: {}", request.function_call.name);
    }
    println!("{}", response.text());
    Ok(())
}
```

Separately, `hosted_mcp(...)` in the core crate describes an MCP server that the model provider calls on its side (OpenAI Responses, Anthropic). Its approval setting is the core `McpApprovalMode` (`Always`, `Never`, `PerTool`), not the client one above.

Not implemented: serving an agent as an MCP server (`as_mcp_server`), elicitation, automatic reconnection, and listening on the standalone SSE stream of the HTTP transport.

## A2A: calling and serving remote agents

The `a2a` feature adds a client. `A2AAgent::from_url(name, url)` implements `SupportsAgentRun`, so a remote agent can be run directly, wrapped as a tool, or used as a workflow participant. `initialize()` fetches the agent card (from `/.well-known/agent-card.json` at the URL's origin, falling back to `/.well-known/agent.json`) and switches to the endpoint the card names. A session keeps the A2A context id between turns. Underneath, `A2AClient` exposes the whole protocol: `send_message`, `send_message_stream`, `get_task`, `cancel_task`, `resubscribe`, push-notification configuration and the authenticated extended card, with `with_bearer_token` and `with_header` for authentication.

Serving is in the `hosting` crate: `A2ARouter::for_agent(name, agent, public_url)` serves the card and a JSON-RPC endpoint at `/`. This program serves an agent and then calls it through A2A:

```rust
use agent_framework::hosting::a2a::A2ARouter;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("helper")
        .description("Answers general questions.")
        .build();

    // Serve it.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8083").await.unwrap();
    let app = A2ARouter::for_agent("helper", agent, "http://127.0.0.1:8083/").into_router();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Call it as a remote agent.
    let remote = A2AAgent::from_url("helper", "http://127.0.0.1:8083/");
    let card = remote.initialize().await?;
    println!("connected to {}: {}", card.name, card.description);
    println!("{}", remote.run_once("What is A2A?").await?.text());
    Ok(())
}
```

The A2A server is simpler than the client. Every `message/send` runs to completion and returns a `completed` task, so there is no `working` or `input-required` state and `tasks/cancel` always reports the task as not cancelable. `message/stream`, push notifications and the extended card are not served, and the card lists one skill derived from the agent's name and description (add more with `.skill(...)`). Finished tasks are kept in memory for `tasks/get`, bounded by `max_tasks`, `task_ttl` and `max_task_bytes`.

## Hosting agents over HTTP

The `hosting` feature turns agents and workflows into [axum](https://docs.rs/axum) routers. Each surface is a separate router that you can serve alone or merge into one application:

| Router | Routes | Clients |
| --- | --- | --- |
| `AgentHost` (DevUI-style API) | `GET /health`, `GET /v1/entities`, `GET /v1/entities/{id}/info`, `POST /v1/responses` (JSON or SSE), a built-in debug page at `/` and `/ui` | DevUI-style tools, any OpenAI Responses client |
| `openai_compat::OpenAiRouter` | `POST /v1/chat/completions` (JSON or SSE) | Any OpenAI Chat Completions client |
| `agui::AgUiRouter` | `POST /` (or a path you choose), streaming AG-UI events | CopilotKit and other AG-UI frontends |
| `a2a::A2ARouter` | Agent card and JSON-RPC | A2A clients |

`AgentHost` registers several agents and workflows by name; `POST /v1/responses` picks one by `metadata.entity_id`, then `extra_body.entity_id`, then `model`. The other routers serve one agent each. Runs are stateless: the caller sends the conversation with each request. The exception is AG-UI with a snapshot store, below.

```rust
use std::time::Duration;

use agent_framework::hosting::agui::AgUiRouter;
use agent_framework::hosting::openai_compat::OpenAiRouter;
use agent_framework::hosting::HostingSecurity;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let assistant = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("assistant")
        .instructions("You are concise.")
        .build();

    let app = AgentHost::new()
        .agent("assistant", assistant.clone())
        .into_router()
        .merge(OpenAiRouter::for_agent("assistant", assistant.clone()).into_router())
        .nest("/agui", AgUiRouter::for_agent("assistant", assistant).into_router());

    // The guards on AgentHost cover only its own routes; wrap the composed app instead.
    let app = HostingSecurity::new()
        .with_bearer_token("dev-token")
        .with_default_localhost_hosts()
        .apply(app);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await.unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Any OpenAI client can now use the agent; the "model" is the agent's name.
    let client = OpenAIChatCompletionClient::new("dev-token", "assistant")
        .with_base_url("http://127.0.0.1:8080/v1");
    let reply = client
        .get_response(vec![Message::user("Hello over HTTP")], ChatOptions::new())
        .await?;
    println!("{}", reply.text());
    Ok(())
}
```

`AgentHost::serve(addr)` binds and serves the DevUI routes alone; `serve_secure` and `into_secure_router` add the localhost host check by default. Security is opt-in everywhere else: `with_bearer_token` requires `Authorization: Bearer <token>`, and `with_allowed_hosts` (or `with_default_localhost_hosts`) rejects requests whose `Host` header is not listed, which protects a local server against DNS rebinding. The debug page is a small embedded HTML page, not the DevUI frontend, which is not ported.

### AG-UI

`AgUiRouter` streams AG-UI events (`RUN_STARTED`, `TEXT_MESSAGE_*`, `TOOL_CALL_*`, `RUN_FINISHED`) and passes the frontend's tools to the agent as declaration-only tools for that run, so their calls go back to the browser. `.agent(name, agent, path)` adds further agents at other paths.

By default the server keeps no state between requests. With `snapshot_store(Arc::new(InMemoryAgUiThreadSnapshotStore::new()))` it stores each thread (messages, shared state, pending approvals and the agent session) and can replay a thread to a reconnecting client and resume a paused approval. A snapshot store must be paired with an authorisation mode, either `authorization_scope(|request_parts| ...)` resolving a scope per request or `single_user()`; `try_into_router()` reports a missing one as an error, and `into_router()` panics on it. Not yet ported: server-side predictive state updates (`STATE_DELTA`, `confirm_changes`).

With the `agui-client` feature, `AgUiChatClient::new(endpoint)` is the reverse: a `ChatClient` backed by a remote AG-UI server.

## Declarative agents and workflows

The `experimental-declarative-agents` feature (not part of `full`) loads agents from YAML or JSON in the official declarative-agent vocabulary: `kind: Prompt`, `name`, `description`, `instructions`, `model` (`id`, `provider`, `apiType`, `connection`, `options`), `tools` and `outputSchema`. String fields accept `${VAR}` and `${VAR:-default}` environment references.

The crate depends on no provider crate, so you supply the clients: a `ChatClientFactory` maps a `provider.apiType` key such as `OpenAI.Chat` to a closure that builds a client (`with_default` catches the rest). `kind: function` tools are bound by name to Rust tools in a `ToolRegistry`; `web_search`, `file_search`, `code_interpreter` and `mcp` tools become provider-hosted tools.

```rust
use std::sync::Arc;

use agent_framework::declarative::{AgentRegistry, ChatClientFactory, DeclarativeLoader, ToolRegistry};
use agent_framework::prelude::*;
use serde_json::json;

const WRITER: &str = r#"
kind: Prompt
name: Writer
instructions: Draft one sentence on the topic. Use get_fact if it helps.
model:
  id: ${WRITER_MODEL:-%%model=openai-small%%}
  provider: OpenAI
  apiType: Chat
  options:
    temperature: 0.7
tools:
  - kind: function
    name: get_fact
    description: Look up a fact about the topic.
"#;

const EDITOR: &str = r#"
kind: Prompt
name: Editor
instructions: Shorten the previous draft to twelve words or fewer.
model:
  id: %%model=openai-small%%
  provider: OpenAI
  apiType: Chat
"#;

const PIPELINE: &str = r#"
kind: Workflow
name: write-then-edit
type: sequential
participants: [writer, editor]
"#;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let factory = ChatClientFactory::new().with("OpenAI.Chat", |model| {
        let id = model.id.clone().unwrap_or_default();
        Ok(Arc::new(OpenAIChatCompletionClient::from_env(id)?) as Arc<dyn ChatClient>)
    });
    let get_fact = FunctionTool::new(
        "get_fact",
        "Look up a fact.",
        json!({ "type": "object", "properties": {} }),
        |_| async move { Ok(json!("Rust 1.0 shipped in May 2015.")) },
    );
    let loader = DeclarativeLoader::new()
        .with_client_factory(factory)
        .with_tool_registry(ToolRegistry::new().with(get_fact.into_definition()));

    let mut agents = AgentRegistry::new();
    agents.register_chat_agent("writer", loader.load_agent(WRITER)?);
    agents.register_chat_agent("editor", loader.load_agent(EDITOR)?);

    let workflow = loader.load_workflow(PIPELINE, &agents)?;
    let run = workflow.run("The history of Rust").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("{}", conversation.last().map(Message::text).unwrap_or_default());
    Ok(())
}
```

Declarative workflows are where the port departs furthest from upstream. Microsoft's declarative workflow format is a Power Fx action language from Copilot Studio, which does not fit this graph engine and is out of scope. The crate defines its own `kind: Workflow` format instead, with two forms: orchestration shorthand (`type: sequential | concurrent | group_chat | handoff` plus `participants`), or an explicit graph of `nodes` (each wrapping a registered agent), `edges`, `fanOut`, `fanIn` and `switch`. An edge or case can carry a `condition` in a small expression language (`PATH OP LITERAL`, such as `status == done`) or name a Rust `predicate` registered in a `PredicateRegistry`. These files will not load in the Python or .NET SDKs, nor theirs here.

## Memory and storage backends

Each backend is a context provider or a history provider, attached as on the [comprehensive guide](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_comprehensive_guide/#context-providers):

| Type | Feature | Kind | Notes |
| --- | --- | --- | --- |
| `RedisChatMessageStore` | `redis` | History | One Redis list per session; `with_max_messages` trims old entries |
| `RedisContextProvider` | `redis` | Long-term memory | Scoped by application, agent, user and thread ids. Full-text (BM25) search with RediSearch (Redis Stack), otherwise a slower scan over plain Redis. No vector search |
| `Mem0Provider` | `mem0` | Long-term memory | The hosted Mem0 API; `from_env` reads `MEM0_API_KEY` and optional `MEM0_API_BASE` |
| `CosmosChatMessageStore` | `cosmos` | History | Cosmos DB NoSQL over REST, master key or Entra ID; partitioned by thread id, or by tenant, user and thread (`CosmosPartitionStrategy::Hierarchical`) |
| `CosmosCheckpointStorage` | `cosmos` | Workflow checkpoints | A `CheckpointStorage` for [checkpointing](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#checkpointing) |
| `AzureAISearchProvider` | `azure-ai-search` | Retrieval | Semantic search with an optional vector query (`with_embedding_function`); the agentic retrieval mode is not ported |
| `FoundryMemoryProvider` | `foundry` | Long-term memory | A Foundry project's managed memory store |

Redis connects lazily, so constructing the providers does no I/O; `RedisChatMessageStore::ping()` checks the connection. This example ran against a local `redis-server`:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into());

    // Conversation history for one session, kept in Redis.
    let history = Arc::new(
        RedisChatMessageStore::new(&url, Some("session-42".into()))?.with_max_messages(200),
    );
    if !history.ping().await {
        eprintln!("Redis is not reachable at {url}");
        return Ok(());
    }

    // Long-term memories for one user, searched before each run.
    let memories = RedisContextProvider::new(&url)?
        .with_user_id("user-42")
        .with_limit(5);

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .context_provider(Arc::new(memories))
        .build();
    let mut session =
        AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);

    for text in ["I love hiking in the Alps.", "What do you know about me?"] {
        let reply = agent.run(vec![Message::user(text)], Some(&mut session)).await?;
        println!("{text} -> {}", reply.text());
    }
    println!("{} messages stored under {}", history.list_messages().await?.len(), history.redis_key());
    Ok(())
}
```

The Azure-backed providers take an endpoint and either a key or a `TokenCredential` from the `azure` crate:

```rust
use std::sync::Arc;

use agent_framework::azure::{DefaultAzureCredential, TokenCredential};
use agent_framework::foundry::{FoundryMemoryProvider, FOUNDRY_SCOPE};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let env = |k: &str| std::env::var(k).unwrap_or_default();

    let search = AzureAISearchProvider::with_api_key(
        env("AZURE_SEARCH_ENDPOINT"),
        env("AZURE_SEARCH_INDEX"),
        env("AZURE_SEARCH_API_KEY"),
    )
    .with_top(5);

    let credential: Arc<dyn TokenCredential> = Arc::new(DefaultAzureCredential::new(FOUNDRY_SCOPE));
    let foundry_memory =
        FoundryMemoryProvider::new(env("FOUNDRY_PROJECT_ENDPOINT"), "user-memories", credential)
            .with_scope("user-42");

    let history = CosmosChatMessageStore::new(
        env("COSMOS_ENDPOINT"),
        env("COSMOS_KEY"),
        "agents",
        "chat-messages",
        Some("thread-42".into()),
    )?
    .with_user_id("user-42")?;
    history.ensure_created().await?;

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai%%")?)
        .context_provider(Arc::new(search))
        .context_provider(Arc::new(foundry_memory))
        .build();
    let mut session =
        AgentSession::new().with_context_providers(vec![Arc::new(history) as Arc<dyn ContextProvider>]);
    let reply = agent
        .run(vec![Message::user("What does our travel policy say?")], Some(&mut session))
        .await?;
    println!("{}", reply.text());
    Ok(())
}
```

### Vector stores

The `experimental-vector-stores` feature adds a provider-neutral `VectorStore` and `VectorCollection` API in `agent_framework::vectors`, with an `InMemoryVectorStore` and connectors for Cosmos DB and Azure AI Search. Filters are written once (`Filter::gte("year", 2020)` and so on) and each connector translates them into its own query language, so retrieval code tested against the in-memory store runs unchanged against the cloud one. `VectorCollectionContextProvider` turns a collection into search and upsert tools for an agent, with writes requiring approval by default. Upstream's PostgreSQL, Qdrant, Redis, SQL Server and other connectors are not ported.

## Purview compliance

The `purview` feature provides `PurviewAgentMiddleware` and `PurviewChatMiddleware`, which send each prompt and response, including attachments, tool calls and tool results, to Microsoft Graph's `processContent` API and block content that a data loss prevention policy flags. `PurviewSettings` sets the application name, the tenant id and application location (both required: Python can infer them from the token's claims, the port cannot, and a run without them fails with a configuration error), the messages shown when a prompt or response is blocked, and whether service errors or a payment-required answer let content through (`with_ignore_exceptions`, `with_ignore_payment_required`).

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::purview::{
    PurviewAppLocation, PurviewLocationType, PurviewSettings, StaticTokenProvider,
};

#[tokio::main]
async fn main() -> Result<()> {
    let token = std::env::var("PURVIEW_TOKEN").unwrap_or_default(); // a Microsoft Graph token
    let settings = PurviewSettings::new("my-agent-app")
        // Both are required: unlike Python, the port does not read them from the token.
        .with_tenant_id(std::env::var("AZURE_TENANT_ID").unwrap_or_default())
        .with_purview_app_location(PurviewAppLocation::new(
            PurviewLocationType::Application,
            std::env::var("PURVIEW_APP_ID").unwrap_or_default(),
        ))
        .with_blocked_prompt_message("That request was blocked by policy.");

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .middleware(Arc::new(PurviewAgentMiddleware::new(StaticTokenProvider::new(token), settings)))
        .build();
    println!("{}", agent.run_once("Summarise the attached contract.").await?.text());
    Ok(())
}
```

Not yet ported: the protection-scopes pre-check and its cache, background content-activity logging, and deriving the user identity from a JWT.

## Shell tools

The `tools` feature (re-exported as `agent_framework::builtin_tools`) contains two shell tools. `LocalShellTool` runs commands on the host and requires approval for every call by default; `DockerShellTool` runs them in a container started with restrictive defaults (no network, a read-only root file system, memory and process limits). Both accept a `ShellPolicy` of allow and deny patterns, an output size limit and a timeout, and can keep one shell session open between calls or start a fresh one each time. `ShellEnvironmentProvider` tells the model which shell and working directory it is using. Read the `shell` module's security notes on docs.rs before enabling either.

```rust
use std::sync::Arc;

use agent_framework::builtin_tools::shell::{LocalShellTool, ShellEnvironmentProvider, ShellPolicy};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let shell = LocalShellTool::builder()
        .workdir(std::env::temp_dir())
        .policy(ShellPolicy::new().with_denylist([r"\brm\b", r"\bsudo\b"])?)
        .build()?;

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are a careful operator.")
        .tool(shell.as_function())
        .context_provider(Arc::new(ShellEnvironmentProvider::new(Arc::new(shell.clone()), None)))
        .build();

    let response = agent.run_once("How much disk space is free?").await?;
    for request in response.user_input_requests() {
        println!("approve? {:?}", request.function_call.arguments);
    }
    Ok(())
}
```

## Observability and OpenTelemetry

The agent, the tool loop and `ObservableChatClient` emit `tracing` spans named and attributed according to the OpenTelemetry GenAI semantic conventions. They cost almost nothing until a subscriber is installed, so there is no switch to turn telemetry on. Message content is left out unless `ENABLE_SENSITIVE_DATA` is set or you configure it with `ObservabilityConfig`; `OTEL_SEMCONV_STABILITY_OPT_IN` selects the newer experimental convention version. The `otel-metrics` feature records the GenAI histograms (token usage, operation duration, tool duration) through the `opentelemetry` API crate.

To see the spans locally, any `tracing` subscriber will do:

```rust
use agent_framework::prelude::*;
use tracing_subscriber::fmt::format::FmtSpan;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_span_events(FmtSpan::CLOSE).init();

    let client = ObservableChatClient::new(
        OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?,
        "openai",
    );
    let agent = Agent::builder(client).name("assistant").build();
    println!("{}", agent.run_once("Hello").await?.text());
    Ok(())
}
```

To export, either wire `tracing-opentelemetry` and an `opentelemetry_sdk` meter provider yourself, or enable the `otel-export` feature for a ready-made OTLP/HTTP pipeline. It reads `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`). Point it at an OpenTelemetry Collector to reach Azure Monitor or another backend; there is no Rust exporter for Application Insights.

```rust
use agent_framework::observability::export::OtelExport;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let pipeline = OtelExport::new("my-agent-service").build()?;
    pipeline.install()?; // claims the global tracing subscriber

    let client = ObservableChatClient::new(
        OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?,
        "openai",
    );
    let agent = Agent::builder(client).name("assistant").build();
    println!("{}", agent.run_once("Hello").await?.text());

    pipeline.shutdown()?; // flush before exit
    Ok(())
}
```

Use `OtelExport::build()` and `OtelPipeline::tracing_layer()` instead of `install()` when the application already has a subscriber stack.

## Other integrations

- **Copilot Studio** (`copilotstudio`): `CopilotStudioAgent` calls a published Copilot Studio agent through the Direct-to-Engine API and implements `SupportsAgentRun`.
- **Foundry embeddings** (`foundry`): `FoundryEmbeddingClient`, for text only.
- **Not ported**: Durable Task and Azure Functions hosting, Foundry hosted-agent deployment, ChatKit, sandboxed code execution, and the Claude Agent SDK and Telegram adapters. The repository's `SCOPE.md` explains each decision.

---
title: "Microsoft Agent Framework (Rust) — Integrations"
description: "agent-framework-rs integrations: declarative YAML agents and workflows with the full spec schema; Redis, Mem0, Cosmos DB, Azure AI Search and Foundry memory backends; vector stores with portable filters; Purview compliance middleware; Copilot Studio agents; local and Docker shell tools."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Integrations"
  order: 12
---

This page covers the companion crates that connect agents to other systems: declarative specs, memory and storage services, vector stores, Microsoft Purview, Copilot Studio and the shell tools. MCP, hosting and OpenTelemetry have their own pages ([MCP](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/), [hosting and A2A](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/), [observability](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/)).

| Feature | Module | Contents |
| --- | --- | --- |
| `experimental-declarative-agents` | `agent_framework::declarative` | YAML/JSON agent and workflow specs (not in `full`) |
| `redis` | `agent_framework::redis` | `RedisChatMessageStore`, `RedisContextProvider` |
| `mem0` | `agent_framework::mem0` | `Mem0Provider` |
| `cosmos` | `agent_framework::cosmos` | `CosmosChatMessageStore`, `CosmosCheckpointStorage`, `CosmosVectorStore` (experimental) |
| `azure-ai-search` | `agent_framework::azure_ai_search` | `AzureAISearchProvider`, `AzureAISearchStore` (experimental) |
| `foundry` | `agent_framework::foundry` | `FoundryMemoryProvider` (with the Foundry chat and embedding clients) |
| `experimental-vector-stores` | `agent_framework::vectors` | Vector store traits, `InMemoryVectorStore`, portable filters, `VectorCollectionContextProvider` |
| `purview` | `agent_framework::purview` | `PurviewAgentMiddleware`, `PurviewChatMiddleware` |
| `copilotstudio` | `agent_framework::copilotstudio` | `CopilotStudioAgent` |
| `tools` | `agent_framework::builtin_tools` | `LocalShellTool`, `DockerShellTool` |

The `experimental-*` features carry no compatibility promise and are turned on together by `experimental`. Every example was compiled against %%version=agent-framework-rs%%. The declarative, Redis (against a local `redis-server`), vector-store, shell and Copilot Studio URL examples were run, as were the Mem0, Azure AI Search and Purview ones against small in-process stand-ins for those services; the Cosmos DB, Foundry memory, Docker and live Copilot Studio code was compiled only.

## Declarative agents and workflows

The `experimental-declarative-agents` feature loads agents from YAML or JSON in the vocabulary of the official agent schema (`kind: Prompt`), and workflows in a format of the port's own. The crate depends on no provider crate: you register a `ChatClientFactory` to build clients, a `ToolRegistry` of Rust tools, and for workflows an `AgentRegistry` of agents and a `PredicateRegistry` of routing functions.

```toml
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["experimental-declarative-agents"] }
```

### The agent spec

Every field of `AgentSpec`, with its nested types. Unknown keys are rejected at every level except inside schemas, so a typo is an error rather than silently ignored.

| Key | Type | Effect |
| --- | --- | --- |
| `kind` | `Prompt` or `Agent` (any case) | Required; anything else is `UnsupportedKind` |
| `name`, `displayName` | string | The agent's name; `displayName` is used when `name` is absent |
| `description` | string | The agent's description |
| `instructions`, `additionalInstructions` | string | Joined with a newline into the instructions |
| `metadata` | map | Carried on the spec; not applied to the agent |
| `model.id` | string | `ChatOptions::model` |
| `model.provider`, `model.apiType` | string | Pick the client factory: `provider.apiType`, then `provider`, then the default |
| `model.connection` | object | Handed to the factory: `kind` (`reference`, `remote`, `key`, `apikey`, `anonymous`), `endpoint`, `key` or `apiKey` (`resolved_key()` prefers `key`), `name`, `target`, `authenticationMode`, `usageDescription` |
| `model.options` | object | `temperature`, `topP`, `topK` (sent as `additional_properties.top_k`), `frequencyPenalty`, `presencePenalty`, `maxOutputTokens`, `seed`, `stopSequences`, `allowMultipleToolCalls`, `chatToolMode` (`auto`, `required`, `none`), `additionalProperties` (passed through) |
| `tools` | list | See below |
| `outputSchema` | schema | Structured output: a strict JSON-schema `response_format` named after the agent |
| `inputSchema` | schema | Parsed but not enforced |

A schema (`PropertySchema`) has `properties`, `strict` (adds `additionalProperties: false`) and `examples`. Each property (`PropertySpec`) has `kind` (or `type`), `description`, `required`, `default`, `example`, `enum`, `items` and nested `properties`; `required: true` flags are collected into the standard top-level `required` array.

Tools are a flat union keyed by `kind`:

| `kind` | Becomes | Notes |
| --- | --- | --- |
| `function` | A function tool | Bound to the `ToolRegistry` entry named in `bindings` (the first key that matches) or by `name`; the spec's description and parameters override the Rust tool's. Unbound, it is declaration-only, so a call is returned to the caller |
| `custom` | The registered Rust tool | Must be bound, or loading fails |
| `web_search` | `hosted_web_search()` | |
| `file_search` | `hosted_file_search(maximumResultCount)` | `vectorStoreIds` are passed; `ranker`, `scoreThreshold` and `filters` are parsed but not sent |
| `code_interpreter` | `hosted_code_interpreter()` | `fileIds` parsed but not sent |
| `mcp` | `hosted_mcp(name, url, allowedTools)` | Provider-side MCP; `approvalMode` (`always`, `never`, `specify`) collapses to one setting: `specify` with any `alwaysRequireApprovalTools` means every tool requires approval |

`openapi` is accepted by the parser but rejected by the loader, as is any other kind. String values anywhere in a spec may use `${VAR}` (an error if unset), `${VAR:-default}` (the default also when the variable is empty) and `$${` for a literal `${`. Upstream's PowerFx `=Env.VAR` expressions are not evaluated; they pass through as text. `with_env(...)` replaces the process environment with any `EnvSource`, including a closure, which keeps tests hermetic.

```rust
use std::sync::Arc;

use agent_framework::declarative::{ChatClientFactory, DeclarativeLoader, ToolRegistry};
use agent_framework::prelude::*;
use serde_json::json;

const SPEC: &str = r#"
kind: Prompt
name: TripPlanner
displayName: Trip planner
description: Plans short trips.
instructions: Plan a trip. Call get_weather for the destination first.
additionalInstructions: Keep the plan to three bullet points.
metadata:
  owner: travel-team
model:
  id: ${PLANNER_MODEL:-%%model=openai-small%%}
  provider: OpenAI
  apiType: Chat
  connection:
    kind: key
    endpoint: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
    key: ${OPENAI_API_KEY}
  options:
    temperature: 0.3
    maxOutputTokens: 800
    seed: 7
    chatToolMode: auto
    additionalProperties:
      user: planner-service
tools:
  - kind: function
    name: get_weather
    description: Current weather for a city.
    parameters:
      properties:
        city: { kind: string, description: City name, required: true }
  - kind: function
    name: book_hotel
    description: Declared but not bound, so the call is returned to the caller.
outputSchema:
  strict: true
  properties:
    destination: { kind: string, required: true }
    bullets: { kind: array, items: { kind: string }, required: true }
"#;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // The factory sees the parsed (and interpolated) model block.
    let factory = ChatClientFactory::new().with("OpenAI.Chat", |model| {
        let connection = model.connection.as_ref();
        let key = connection.and_then(|c| c.resolved_key()).unwrap_or_default();
        let mut client = OpenAIChatCompletionClient::new(key, model.id.clone().unwrap_or_default());
        if let Some(endpoint) = connection.and_then(|c| c.endpoint.as_deref()) {
            client = client.with_base_url(endpoint);
        }
        Ok(Arc::new(client) as Arc<dyn ChatClient>)
    });
    let weather = FunctionTool::new(
        "get_weather",
        "Look up the weather.",
        json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        |args| async move { Ok(json!({ "city": args["city"], "forecast": "sunny, 21C" })) },
    );
    let loader = DeclarativeLoader::new()
        .with_client_factory(factory)
        .with_tool_registry(ToolRegistry::new().with(weather.into_definition()));

    // Parse without building, to inspect or validate a spec.
    let spec = loader.load_agent_spec(SPEC)?;
    println!("{} -> provider key {:?}", spec.effective_name().unwrap_or_default(), spec.model.as_ref().and_then(|m| m.provider_key()));

    let agent = loader.load_agent(SPEC)?;
    let reply = agent.run_once("A weekend in Lisbon").await?;
    println!("{}", reply.text());
    Ok(())
}
```

`load_agent_spec` and `load_workflow_spec` parse and interpolate without building; `AgentSpec::from_yaml` parses without interpolation, and `to_yaml` writes a spec back. Errors are `DeclarativeError` variants (`Parse`, `UnsupportedKind`, `MissingField`, `MissingEnvVar`, `MalformedPlaceholder`, `UnknownReference`, `NoClientFactory`, `Invalid`, `InvalidCondition`, `Core`), each naming what was wrong and where.

### Workflow specs

Microsoft's declarative workflow format is a Power Fx action language from Copilot Studio, which does not map onto this graph engine and is out of scope. The crate defines its own `kind: Workflow` format instead, so these files do not load in the Python or .NET SDKs, nor theirs here. A spec takes one of two forms.

**Orchestration shorthand** sets `type` and `participants` (agent ids from the `AgentRegistry`):

| `type` | Built with | Extra keys |
| --- | --- | --- |
| `sequential` | `SequentialBuilder` | |
| `concurrent` | `ConcurrentBuilder` | |
| `group_chat` | `GroupChatBuilder` with the round-robin manager | `roundRobin` (must be `true`; an LLM manager cannot be declared), `maxRounds` |
| `handoff` | `HandoffBuilder` | `start` (initial agent, default the first participant), `handoffs: [{from, to: [...]}]`, `autonomous` (default `true`), `maxIterations` |

**An explicit graph** declares `nodes` (`id`, `agent`, `output`), `start`, and any of `edges` (`from`, `to`, optional `condition` or `predicate`), `fanOut` (`from`, `to: [...]`), `fanIn` (`from: [...]`, `to`), `switch` (`from`, ordered `cases` of `condition` or `predicate` with `to` and an optional `label`, and a `default`) and `maxIterations`. Each node wraps its agent in an `AgentExecutor`, which receives the conversation as a JSON array of messages and sends it on with the reply appended; `output: true` makes the node yield the conversation as a workflow output instead. `name` and `description` apply to either form.

A `condition` is `PATH OP LITERAL`: a dotted path into the message (numeric segments index arrays, so `1.role` is the second message's role), one of `==`, `!=`, `<`, `<=`, `>`, `>=`, and a JSON literal or bare string. Equality compares JSON values, numbers by value; ordering applies to numbers only; a missing path is `false`. Anything more needs a named Rust predicate:

```rust
use std::sync::Arc;

use agent_framework::declarative::{
    AgentRegistry, ChatClientFactory, DeclarativeLoader, PredicateRegistry,
};
use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

/// A deterministic model: the classifier answers "billing" or "technical";
/// the specialists say who they are.
#[derive(Clone)]
struct Scripted;

#[async_trait]
impl ChatClient for Scripted {
    async fn get_response(&self, messages: Vec<Message>, _options: ChatOptions) -> Result<ChatResponse> {
        // The agent sends its instructions as a leading system message.
        let instructions = messages.iter().find(|m| m.role == Role::system()).map(Message::text).unwrap_or_default();
        let question = messages.iter().find(|m| m.role == Role::user()).map(Message::text).unwrap_or_default();
        let reply = if instructions.starts_with("Classify") {
            if question.contains("invoice") { "billing" } else { "technical" }.to_string()
        } else {
            format!("{} answered: {question}", instructions.split('.').next().unwrap_or("?"))
        };
        Ok(ChatResponse::from_text(reply))
    }
    async fn get_streaming_response(&self, m: Vec<Message>, o: ChatOptions) -> Result<ChatStream> {
        let response = self.get_response(m, o).await?;
        let update = ChatResponseUpdate { contents: response.messages[0].contents.clone(), role: Some(Role::assistant()), ..Default::default() };
        Ok(futures::stream::iter(vec![Ok(update)]).boxed())
    }
}


const AGENTS: [(&str, &str); 3] = [
    ("classifier", "kind: Prompt\nname: classifier\ninstructions: Classify the request as billing or technical.\nmodel: { provider: Scripted }\n"),
    ("billing", "kind: Prompt\nname: billing\ninstructions: Billing desk. Explain charges.\nmodel: { provider: Scripted }\n"),
    ("technical", "kind: Prompt\nname: technical\ninstructions: Technical desk. Fix problems.\nmodel: { provider: Scripted }\n"),
];

const WORKFLOW: &str = r#"
kind: Workflow
name: support-triage
start: classify
maxIterations: 10
nodes:
  - { id: classify, agent: classifier }
  - { id: billing, agent: billing, output: true }
  - { id: technical, agent: technical, output: true }
switch:
  - from: classify
    cases:
      - { predicate: says_billing, to: billing, label: billing }
      # `PATH OP LITERAL` over the forwarded conversation (a JSON array).
      - { condition: "1.role == assistant", to: technical, label: anything else }
    default: technical
"#;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let factory = ChatClientFactory::new().with("Scripted", |_| Ok(Arc::new(Scripted) as Arc<dyn ChatClient>));
    // A named predicate: the last message of the conversation says "billing".
    let predicates = PredicateRegistry::new().with("says_billing", |conversation: &Value| {
        conversation
            .as_array()
            .and_then(|messages| messages.last())
            .map(|last| last.to_string().contains("billing"))
            .unwrap_or(false)
    });
    let loader = DeclarativeLoader::new().with_client_factory(factory).with_predicates(predicates);

    let mut agents = AgentRegistry::new();
    for (id, yaml) in AGENTS {
        agents.register_chat_agent(id, loader.load_agent(yaml)?);
    }
    let workflow = loader.load_workflow(WORKFLOW, &agents)?;
    println!("{}", workflow.viz().to_mermaid());

    for question in ["Why is my invoice higher this month?", "The app crashes on start."] {
        let run = workflow.run(question).await?;
        let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
        println!("{question} -> {}", conversation.last().map(Message::text).unwrap_or_default());
    }
    Ok(())
}
```

Setting both `condition` and `predicate` on one edge is an error, as is a graph without `start` or a spec with neither `type` nor `nodes`.

## Memory and storage backends

Each backend plugs into an agent as a [context provider](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/), either a history provider attached to a [session](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/) or a memory or retrieval provider attached to the agent:

| Type | Kind | What it does before and after a run |
| --- | --- | --- |
| `RedisChatMessageStore` | History | Loads the session's list; appends the new messages |
| `RedisContextProvider` | Long-term memory | Searches stored memories for the input; stores the exchange |
| `Mem0Provider` | Long-term memory | Searches the Mem0 API; adds the exchange to it |
| `CosmosChatMessageStore` | History | As Redis, in a Cosmos DB container |
| `CosmosCheckpointStorage` | Workflow checkpoints | A `CheckpointStorage` for [checkpointing](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#checkpointing) |
| `AzureAISearchProvider` | Retrieval | Searches an index for the latest user message; adds the hits to the instructions |
| `FoundryMemoryProvider` | Long-term memory | Searches a Foundry memory store; writes the turn back |

A provider attached with the builder's `context_provider` is shared by every run of the agent. Scope memory providers to a user or tenant, and keep per-user stores on the session.

### Redis

`RedisChatMessageStore::new(url, session_id)` keeps one Redis list per conversation at `{prefix}:{session id}` (default prefix `chat_messages`; a `None` id becomes `thread_{uuid}`), each element a message serialised as JSON. Both halves of the key are escaped, so a session id containing `:` cannot address another prefix's list. `with_max_messages(n)` trims the list to the newest `n` after each write (`0` stores nothing at all); `list_messages`, `add_messages`, `clear` and `ping` work on it directly, and `to_dict`/`from_dict` save the configuration (URL, prefix, id, limit), not the messages. A caller that resends its whole transcript is recognised and only the new turns are appended, though two runs finishing at once on one session can still both append.

`RedisContextProvider::new(url)` stores memories scoped by `with_application_id`, `with_agent_id`, `with_user_id` and `with_thread_id`; at least one is required, or the run fails with a configuration error. Before a run it searches with the input text and adds up to `with_limit(n)` (default 10) hits as one user message headed by `with_context_prompt` (default `## Memories\nConsider the following memories when answering user questions:`); after a run it stores the user, assistant and system messages of the exchange that have text. `with_scope_to_per_operation_thread_id(true)` takes the thread id from the session id instead, and errors if a different session later uses the same provider.

Retrieval depends on the server. With RediSearch loaded (Redis Stack) the provider creates a JSON index and runs a BM25 full-text `FT.SEARCH` filtered by scope. On plain Redis it falls back to `SCAN` over every memory under the prefix, keeping those that contain any non-stopword of the query and taking the most recent: fine for modest volumes, linear in the number of memories. `with_force_scan_fallback(true)` forces the fallback. The two paths store memories differently (`JSON.SET` against plain strings), so memories written under one are invisible to the other. There is no vector or hybrid search, unlike Python's `RedisProvider`. Both types connect lazily, so constructing them does no I/O. This example ran against a plain local `redis-server`:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into());

    // History for one conversation: a Redis list at "{prefix}:{session id}".
    let history = Arc::new(
        RedisChatMessageStore::new(&url, Some("session-42".into()))?
            .with_key_prefix("support")
            .with_max_messages(200),
    );
    if !history.ping().await {
        eprintln!("Redis is not reachable at {url}");
        return Ok(());
    }
    history.clear().await?;

    // Long-term memory for one user, searched before every run.
    let memories = RedisContextProvider::new(&url)?
        .with_key_prefix("support-memory")
        .with_application_id("support-bot")
        .with_user_id("user-42")
        .with_limit(5);

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .context_provider(Arc::new(memories))
        .build();

    let mut session = AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    agent.run(vec![Message::user("I love hiking in the Alps.")], Some(&mut session)).await?;
    println!("{} messages stored under {}", history.list_messages().await?.len(), history.redis_key());

    // A brand-new session: no history, but the memory provider finds the
    // earlier turn because the question shares a word with it ("hiking").
    let reply = agent.run_once("Any hiking tips for me?").await?;
    println!("{}", reply.text());

    // The store serialises its configuration, not the messages.
    let saved = history.to_dict();
    let restored = RedisChatMessageStore::from_dict(&saved)?;
    println!("restored {} -> {} messages", restored.session_id(), restored.list_messages().await?.len());
    Ok(())
}
```

The assistant's reply was stored as a memory too, so the second run saw both lines.

### Mem0

`Mem0Provider::new(api_key)` (or `from_env()`, reading `MEM0_API_KEY` and optional `MEM0_API_BASE`) talks to the Mem0 REST API: `POST /v1/memories/` after a run with the exchange, and `POST /v2/memories/search/` before one with the latest input. It authenticates with `Authorization: Token <key>`; `with_api_base` points it elsewhere.

Its scopes come in two sets, and the difference matters. The storage scope (`with_user_id`, `with_agent_id`, `with_application_id`, plus `with_thread_id` sent as `run_id`) stamps what is written. The retrieval scope (`with_search_user_id`, `with_search_agent_id`, `with_search_application_id`) selects what is read, never inherits from the storage scope, and when unset means nothing is retrieved (with one warning in the log). This stops an agent-wide `agent_id` from reading every user's memories into every conversation. Each retrieval scope is queried separately, because Mem0 ANDs the fields of one filter; `search_application_id` is used only when neither of the others is set. The crate's own README example and `mem0_memory` example set only `with_user_id`, and so retrieve nothing. Here a stand-in Mem0 server shows the difference:

```rust
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

type Memories = Arc<Mutex<Vec<(String, String)>>>; // (user_id, memory)

/// A stand-in for the two Mem0 endpoints the provider calls.
fn fake_mem0(store: Memories) -> Router {
    Router::new()
        .route("/v1/memories/", post(|State(s): State<Memories>, Json(body): Json<Value>| async move {
            let user = body["user_id"].as_str().unwrap_or("-").to_string();
            for m in body["messages"].as_array().into_iter().flatten() {
                s.lock().unwrap().push((user.clone(), m["content"].as_str().unwrap_or_default().to_string()));
            }
            Json(json!({ "results": [] }))
        }))
        .route("/v2/memories/search/", post(|State(s): State<Memories>, Json(body): Json<Value>| async move {
            let user = body["filters"]["user_id"].as_str().unwrap_or_default().to_string();
            let hits: Vec<Value> = s.lock().unwrap().iter()
                .filter(|(u, _)| *u == user)
                .map(|(_, m)| json!({ "memory": m }))
                .collect();
            Json(Value::Array(hits))
        }))
        .with_state(store)
}

#[tokio::main]
async fn main() -> Result<()> {
    let store: Memories = Arc::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = fake_mem0(store.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let key = std::env::var("MEM0_API_KEY").unwrap_or_else(|_| "test-key".into());
    let make_agent = |memory: Mem0Provider| -> Result<Agent> {
        Ok(Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
            .context_provider(Arc::new(memory))
            .build())
    };

    // Storage scope only: memories are written, but nothing is ever read back.
    let write_only = Mem0Provider::new(&key).with_api_base(&base).with_user_id("user-42");
    make_agent(write_only)?.run_once("My favourite tea is Earl Grey.").await?;
    println!("stored: {}", store.lock().unwrap().len());

    // Retrieval needs its own scope, which never inherits from storage.
    let read_write = Mem0Provider::new(&key)
        .with_api_base(&base)
        .with_user_id("user-42")
        .with_search_user_id("user-42");
    let agent = make_agent(read_write)?;
    // The memories arrive as a user message headed "## Memories".
    println!("{}", agent.run_once("What tea do I like?").await?.text());
    Ok(())
}
```

The crate targets the `/v1` and `/v2` routes, matching the Python package at the time of the port; Mem0's newer `/v3` API is not used. No result limit or threshold is configurable.

### Cosmos DB

`CosmosChatMessageStore::new(endpoint, key, database, container, thread_id)` stores each message as a document (`id`, `threadId`, `seq`, `message` as a JSON string) and lists them `ORDER BY c.seq`, where `seq` is the write time in milliseconds times 1,000 plus the index within the batch. `with_token_credential(endpoint, credential, database, container, thread_id, scope)` authenticates with Microsoft Entra ID instead of the master key; the default scope is `https://cosmos.azure.com/.default` (or `AZURE_COSMOS_AAD_SCOPE_OVERRIDE`), and the principal needs a data-plane role such as Cosmos DB Built-in Data Contributor. That is the only mode that works on an account with local authentication disabled.

`with_tenant_id` and `with_user_id` write `tenantId` and `userId` on every document; `with_partition_strategy(CosmosPartitionStrategy::Hierarchical)` partitions by `/tenantId`, `/userId`, `/threadId` instead of `/threadId` alone, and needs both ids. The strategy must match the container. `ensure_created()` creates the database and container with the matching partition key, but only with the master key: Entra data-plane roles cannot create containers, so with a credential it reports that the container must be provisioned out of band (Bicep, the CLI or the portal). The repository's `SCOPE.md` still lists tenant and user fields as not ported; the 0.14.0 source has them.

Other differences from .NET's store: messages are written one `POST` each rather than as a transactional batch, so a multi-message write is not atomic, and no TTL is set, so messages stay until `clear()`. `serialize()` and `from_state` save and restore the store's configuration. The thread id selects history but does not protect it: anyone with the container's credentials can read any thread, so give mutually distrusting parties separate containers or accounts.

`CosmosCheckpointStorage::new(endpoint, key, database, container)` (or `with_token_credential`) stores workflow checkpoints, partitioned by `/id`.

### Azure AI Search

`AzureAISearchProvider::with_api_key(endpoint, index, key)` or `with_token_credential(endpoint, index, credential)` (scope `https://search.azure.com/.default`, changeable with `with_scope`) runs one search per run against `POST {endpoint}/indexes('{index}')/docs/search?api-version=2024-07-01`, using the latest non-empty user message as the query, and adds the hits to the instructions under `with_context_prompt` (default `Use the following context to answer the question:`).

| Method | Default | Effect |
| --- | --- | --- |
| `with_top(n)` | 5 | Documents retrieved |
| `with_select_fields([...])` | all | `select` |
| `with_semantic_configuration(name)` | none | Semantic ranking with extractive captions; the vector query's `k` is raised to at least 50 for the reranker |
| `with_vector_field(field)` | none | Adds a vector query; without an embedding function the service vectorises the text (`kind: text`) |
| `with_embedding_function(f)` | none | Embeds the query client-side (`kind: vector`) |
| `with_filter(odata)` | none | An OData `$filter`, applied by the service before ranking |
| `with_api_version(v)` | `2024-07-01` | |

Each hit becomes `[Source: <id>] <text>`, the text taken from the first of `content`, `text`, `description`, `body` or `chunk`, or else every string field as `name: value`. On a shared index the filter is an isolation control, not only a relevance setting: without one, every tenant's documents are candidates for every run. The filter is sent verbatim, so escape any literal you interpolate (single quotes are doubled in OData). The example runs against a stand-in that prints the request it receives:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use axum::{Json, Router};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<()> {
    // A stand-in for `POST {endpoint}/indexes('{index}')/docs/search`: it
    // prints the request body and returns two documents.
    let app = Router::new().fallback(|Json(body): Json<Value>| async move {
        println!("search body: {body}");
        Json(json!({ "value": [
            { "id": "policy-7", "content": "Economy class for flights under six hours.", "@search.score": 3.1 },
            { "id": "policy-9", "title": "Hotels", "region": "EU", "@search.score": 2.4 }
        ] }))
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = std::env::var("AZURE_SEARCH_ENDPOINT")
        .unwrap_or_else(|_| format!("http://{}", listener.local_addr().unwrap()));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let search = AzureAISearchProvider::with_api_key(&endpoint, "travel-policies", "query-key")
        .with_top(3)
        .with_select_fields(["id", "content", "title", "region"])
        .with_semantic_configuration("default")
        .with_filter("tenant_id eq 'acme'")
        // A client-side embedding for the vector query (otherwise the service
        // vectorises the text itself).
        .with_vector_field("content_vector")
        .with_embedding_function(|query: String| async move { Ok(vec![query.len() as f32, 0.0, 1.0]) });

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .context_provider(Arc::new(search))
        .build();
    println!("{}", agent.run_once("Can I fly business class?").await?.text());
    Ok(())
}
```

A failed search fails the run with `Error::ServiceStatus`. Python's agentic (knowledge base) retrieval mode is not ported.

### Foundry memory

`FoundryMemoryProvider::new(project_endpoint, store_name, credential)` uses a Foundry project's managed memory store: `memory_stores/{name}:search_memories` before a run and `:update_memories` after it, with a bearer token for `FOUNDRY_SCOPE` (`https://ai.azure.com/.default`; `with_token_scope` changes it). The first run for a scope also fetches the scope's static memories (the user profile) once, and keeps them. Memories are injected ahead of the conversation under `with_context_prompt`.

Set `with_scope` on a provider shared across sessions. Without it the scope is the session id, and because `after_run` is not told which session it belongs to, a provider that has seen more than one session stops writing (with a warning) rather than guess. `with_update_delay(seconds)` controls how long the service waits before processing an update (it defaults to 300 when unset; `0` writes at once). Search and update cursors are cached per scope, at most 512 scopes (`with_max_cached_scopes`). The update is a single POST; as in Python, the long-running operation is not polled.

The three Azure-backed providers together, with Entra ID throughout (compiled, not run):

```rust
use std::sync::Arc;

use agent_framework::azure::{DefaultAzureCredential, TokenCredential};
use agent_framework::cosmos::{CosmosCheckpointStorage, CosmosPartitionStrategy};
use agent_framework::foundry::{FoundryMemoryProvider, FOUNDRY_SCOPE};
use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;

#[tokio::main]
async fn main() -> Result<()> {
    let env = |k: &str| std::env::var(k).unwrap_or_default();

    // Entra ID throughout: one credential per audience.
    let cosmos_credential: Arc<dyn TokenCredential> =
        Arc::new(DefaultAzureCredential::new("https://cosmos.azure.com/.default"));
    let search_credential: Arc<dyn TokenCredential> =
        Arc::new(DefaultAzureCredential::new("https://search.azure.com/.default"));
    let foundry_credential: Arc<dyn TokenCredential> = Arc::new(DefaultAzureCredential::new(FOUNDRY_SCOPE));

    // History in Cosmos DB, partitioned by tenant, user and thread. With a
    // token credential the container must already exist (ensure_created
    // needs control-plane rights that Entra data-plane roles do not grant).
    let history = CosmosChatMessageStore::with_token_credential(
        env("COSMOS_ENDPOINT"),
        cosmos_credential.clone(),
        "agents",
        "chat-messages",
        Some("thread-42".into()),
        None, // default scope
    )?
    .with_tenant_id("contoso")?
    .with_user_id("user-42")?
    .with_partition_strategy(CosmosPartitionStrategy::Hierarchical);

    let search = AzureAISearchProvider::with_token_credential(env("AZURE_SEARCH_ENDPOINT"), "policies", search_credential)
        .with_top(5)
        .with_filter("tenant_id eq 'contoso'");

    // Pin the scope on a provider shared across sessions.
    let memory = FoundryMemoryProvider::new(env("FOUNDRY_PROJECT_ENDPOINT"), "user-memories", foundry_credential)
        .with_scope("contoso:user-42")
        .with_update_delay(0);

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai%%")?)
        .context_provider(Arc::new(search))
        .context_provider(Arc::new(memory))
        .build();
    let mut session = AgentSession::new().with_context_providers(vec![Arc::new(history) as Arc<dyn ContextProvider>]);
    let reply = agent
        .run(vec![Message::user("What does our travel policy say about hotels?")], Some(&mut session))
        .await?;
    println!("{}", reply.text());

    // Workflow checkpoints in Cosmos DB (partitioned by /id).
    let checkpoints = CosmosCheckpointStorage::with_token_credential(
        env("COSMOS_ENDPOINT"), cosmos_credential, "agents", "checkpoints", None,
    )?;
    let step = FunctionExecutor::new("step", |message, ctx| async move { ctx.yield_output(message).await });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(step))
        .set_start("step")
        .with_checkpointing(Arc::new(checkpoints))
        .build()?;
    workflow.run("hello").await?;
    Ok(())
}
```

## Vector stores

The `experimental-vector-stores` feature adds a provider-neutral vector store API in `agent_framework::vectors`. `VectorStoreField::key`, `data` and `vector(name, dimensions)` describe fields (`with_type`, `indexed`, `full_text_indexed`, `with_index_kind`, `with_distance_function`, `with_storage_name`); `VectorStoreCollectionDefinition::new` validates them (one key, at least one vector, no duplicates). A `VectorStore` hands out `VectorCollection`s, which `ensure_collection_exists`, `upsert`, `get`, `delete` and `search(vector, &VectorSearchOptions)`. Records are `serde_json::Value` objects keyed by field name, so a typed struct converts with `serde_json::to_value`, and both traits are object-safe, so `Box<dyn VectorCollection>` swaps between providers. The implementations are `InMemoryVectorStore` (brute force, for tests), `CosmosVectorStore` and `AzureAISearchStore`; upstream's PostgreSQL, Qdrant, Redis, SQL Server, MongoDB and other connectors are not ported.

`VectorSearchOptions::new(top)` takes `with_skip`, `with_include_vectors`, `with_vector_field_name` (needed when there are several vector fields), `with_filter` and `with_provider_filter` (a raw dialect string, ANDed with the portable filter). A result has `record`, `score` and `score_kind`. Score direction depends on the service, not only on the distance function: Azure AI Search always returns a higher-is-better relevance score, and hybrid search an RRF score, so read `result.higher_is_closer(...)` rather than assuming.

### Portable filters

`Filter` leaves (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `between`, `any_of`, `none_of`, `contains`, `contains_any`, `contains_all`, `starts_with`, `ends_with`, `contains_text`, `exists`, `is_null`, `is_not_null`) combine with `FilterGroup::and`, `or` and `not`, validated at construction (at most depth 8 and 64 nodes). Each connector translates the same expression into its own language, so retrieval code tested against the in-memory store runs unchanged against a cloud one, and an operator a service cannot express is refused with an error naming the alternative rather than dropped. A missing field is a non-match for every operator except `exists`, so `ne` and `not(eq)` differ; a boolean never equals a number; numbers compare by value. Upstream's `Param` placeholders are not ported, since Rust code builds the filter with the values it already has. The concrete collections expose their translation and schema, which this example prints without any network calls:

```rust
use agent_framework::azure_ai_search::AzureAISearchStore;
use agent_framework::cosmos::CosmosVectorStore;
use agent_framework::prelude::*;
use agent_framework::vectors::{
    DistanceFunction, Filter, FilterGroup, InMemoryVectorStore, VectorSearchOptions, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("title").with_type("str").indexed(),
        VectorStoreField::data("year").with_type("int").indexed(),
        VectorStoreField::data("tags").with_type("list[str]").indexed(),
        VectorStoreField::vector("embedding", 3)
            .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
    ])?;

    // One portable filter: written once, translated by each connector.
    let filter = FilterGroup::and(vec![
        Filter::gte("year", 2020)?.into(),
        Filter::contains("tags", "rust")?.into(),
        FilterGroup::not(Filter::eq("title", "Draft")?.into())?,
    ])?;
    let options = VectorSearchOptions::new(2).with_filter(filter);

    // The in-memory store evaluates it locally: right for tests.
    let memory = InMemoryVectorStore::new();
    let docs = memory.get_collection("docs", definition.clone())?;
    docs.ensure_collection_exists().await?;
    docs.upsert(vec![
        json!({ "id": "a", "title": "Async Rust", "year": 2024, "tags": ["rust"], "embedding": [1.0, 0.0, 0.0] }),
        json!({ "id": "b", "title": "Draft", "year": 2025, "tags": ["rust"], "embedding": [1.0, 0.1, 0.0] }),
        json!({ "id": "c", "title": "Old Rust", "year": 2015, "tags": ["rust"], "embedding": [0.9, 0.0, 0.1] }),
    ])
    .await?;
    for hit in docs.search(vec![1.0, 0.0, 0.0], &options).await? {
        println!("in-memory hit: {} score={:?}", hit.record["id"], hit.score);
    }

    // The cloud connectors expose their translation, so you can see what
    // they would send. Nothing here touches the network.
    let search = AzureAISearchStore::with_api_key("https://example.search.windows.net", "admin-key")
        .collection("docs", definition.clone())?;
    println!("OData:  {:?}", search.prepare_filter(&options)?);
    println!("index:  {}", search.build_index()?["fields"].as_array().map_or(0, Vec::len));

    let cosmos = CosmosVectorStore::new("https://example.documents.azure.com:443/", "a2V5", "agents")?
        .collection("docs", definition)?;
    let mut parameters = Vec::new();
    println!("Cosmos: {:?} {parameters:?}", cosmos.prepare_filter(&options, &mut parameters)?);
    println!("policy: {}", serde_json::Value::Object(cosmos.build_container_policies())["vectorEmbeddingPolicy"]);

    // An operator a connector cannot express is refused, not dropped.
    let unsupported = VectorSearchOptions::new(2).with_filter(Filter::ends_with("title", "Rust")?);
    println!("ends_with on Azure: {:?}", search.prepare_filter(&unsupported).err().map(|e| e.to_string()));
    Ok(())
}
```

Each connector has its own rules: Azure AI Search filters only on `indexed()` fields and runs vector search with the filter applied before ranking (with `k` covering `skip + top`); Cosmos DB has no full-text search, so it rejects `full_text_indexed` fields, and binds filter values as query parameters. `AzureAISearchStore::collection` returns the concrete `AzureAISearchCollection`, which adds `search_hybrid(text, vector, options)` (keyword and vector, fused with RRF), `build_index` and `prepare_filter`; the store adds `create_or_update_alias` and `delete_alias`. Uploads are batched within the service's limits, and a partial success (HTTP 207) is an error rather than read as success. `CosmosVectorStore::new(endpoint, key, database)` (or `with_token_credential`) has `ensure_database_exists` and `collection`; containers are partitioned by `/id` and created with service-default vector policies, and quantiser and search-list tuning are not configurable: create the container yourself with them and point the collection at it.

### Vector tools for an agent

`VectorCollectionContextProvider::builder(collection, embedder)` turns a collection into tools the model calls: `{prefix}_search`, `_get`, `_upsert` and `_delete`. The writes require approval by default (`approval_mode(VectorToolKind::..., mode)` changes it). `embed_from_field("text")` names the field the vector is computed from, so the model never writes an embedding and the vector field is not in the tool schema. `scope_filter(...)` is ANDed into every search and checked on every record a tool touches. It groups records between cooperating callers but is not a security boundary (a delete is read, check, delete, which is not atomic); give mutually distrusting parties separate collections. Other builder methods: `tool_prefix`, `source_id`, `instructions`, `vector_field`, `include_search`, `include_get`, `include_delete`, `search_top`, `max_tool_batch_size`.

```rust
use std::sync::Arc;

use agent_framework::client::EmbeddingClient;
use agent_framework::prelude::*;
use agent_framework::types::{Embedding, EmbeddingGenerationOptions, EmbeddingInput, GeneratedEmbeddings};
use agent_framework::vectors::{
    Filter, InMemoryVectorStore, VectorCollection, VectorCollectionContextProvider, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use async_trait::async_trait;
use serde_json::json;

/// A toy embedder for tests: counts a few keywords.
struct KeywordEmbedder;

#[async_trait]
impl EmbeddingClient for KeywordEmbedder {
    async fn get_embeddings(&self, values: Vec<EmbeddingInput>, _o: Option<EmbeddingGenerationOptions>) -> Result<GeneratedEmbeddings> {
        let embed = |text: String| {
            let t = text.to_lowercase();
            Embedding::new(["coffee", "expense", "launch"].iter().map(|w| t.matches(w).count() as f32 + 0.01).collect())
        };
        let text_of = |input: &EmbeddingInput| -> String {
            input.contents.iter().filter_map(|c| match c { Content::Text(t) => Some(t.text.clone()), _ => None }).collect()
        };
        Ok(GeneratedEmbeddings::new(values.iter().map(|v| embed(text_of(v))).collect()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("text").with_type("str"),
        VectorStoreField::data("tenant").with_type("str").indexed(),
        VectorStoreField::vector("embedding", 3),
    ])?;
    let store = InMemoryVectorStore::new();
    let notes: Arc<dyn VectorCollection> = Arc::from(store.get_collection("notes", definition)?);
    notes.ensure_collection_exists().await?;
    notes.upsert(vec![
        json!({ "id": "n1", "text": "The coffee machine is on the third floor.", "tenant": "acme", "embedding": [1.0, 0.0, 0.0] }),
        json!({ "id": "n2", "text": "Launch date is March 14th.", "tenant": "other", "embedding": [0.0, 0.0, 1.0] }),
    ]).await?;

    let provider = VectorCollectionContextProvider::builder(notes, Arc::new(KeywordEmbedder))
        .tool_prefix("notes")
        .scope_filter(Filter::eq("tenant", "acme")?)
        .embed_from_field("text")
        .search_top(3)
        .build()?;
    for tool in provider.tools() {
        println!("{} (approval: {})", tool.name, tool.requires_approval());
    }

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Answer from the notes collection.")
        .context_provider(Arc::new(provider))
        .build();
    println!("{}", agent.run_once("Where is the coffee machine?").await?.text());
    Ok(())
}
```

## Purview compliance

The `purview` feature sends each prompt and response to Microsoft Graph's `dataSecurityAndGovernance/processContent` API and blocks content that a data loss prevention policy flags. `PurviewAgentMiddleware` attaches with the agent builder's `middleware`; `PurviewChatMiddleware` with `chat_middleware`, which runs once per agent run around the whole tool loop, not around each model request (see [Middleware](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/)). Both check the request before the call (a block short-circuits with one system message, `blocked_prompt_message`, and the model is never called) and the response after it (a block replaces it with `blocked_response_message`, keeping only control fields such as the response id and usage, so the blocked text cannot be recovered from the raw payload). Every content item is evaluated, not only text: attachments as binary content, tool calls and results serialised whole; only usage content is skipped. Both directions are evaluated as `uploadText`, matching a quirk of the Python package.

Purview evaluates per user. The user id is taken from a message's `additional_properties["user_id"]`, then from a GUID-shaped `author_name`, and must be a GUID. These are trusted inputs: set them on the server from a validated token, never from anything a client controls, or a caller can pick a weaker user's policy. With no user id the evaluation fails rather than allowing the content.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::purview::{PurviewAppLocation, PurviewLocationType, PurviewSettings, StaticTokenProvider};
use axum::{Json, Router};
use serde_json::{json, Value};

const USER: &str = "6f1c2a3b-0000-4000-8000-0000000000aa";

#[tokio::main]
async fn main() -> Result<()> {
    // A stand-in for Graph's processContent: block anything mentioning "SECRET".
    let graph = Router::new().fallback(|uri: axum::http::Uri, Json(body): Json<Value>| async move {
        let blocked = body.to_string().contains("SECRET");
        println!("  processContent {} -> {}", uri.path(), if blocked { "block" } else { "allow" });
        let actions = if blocked { json!([{ "action": "blockAccess" }]) } else { json!([]) };
        Json(json!({ "policyActions": actions }))
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let graph_base = format!("http://{}/v1.0/", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, graph).await.unwrap() });

    let settings = PurviewSettings::new("travel-assistant")
        .with_app_version("1.4.0")
        // Both are required: unlike Python, the port does not read them from the token.
        .with_tenant_id("00000000-0000-4000-8000-000000000001")
        .with_purview_app_location(PurviewAppLocation::new(
            PurviewLocationType::Application,
            "00000000-0000-4000-8000-000000000002",
        ))
        .with_graph_base_uri(graph_base) // omit in production
        .with_blocked_prompt_message("That request was blocked by policy.");
    let token = std::env::var("GRAPH_TOKEN").unwrap_or_else(|_| "test-token".into());

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .middleware(Arc::new(PurviewAgentMiddleware::new(StaticTokenProvider::new(token), settings)))
        .build();

    // Purview evaluates for a user. The id comes from the message, set by your
    // server from a validated token, never from client input.
    let ask = |text: &str| {
        let mut message = Message::user(text);
        message.additional_properties.insert("user_id".into(), json!(USER));
        message
    };
    for text in ["Summarise our travel policy.", "Email me the SECRET merger plan."] {
        println!("{text}");
        println!("  -> {}", agent.run(vec![ask(text)], None).await?.text());
    }

    // No user id: evaluation fails closed.
    match agent.run_once("Hello").await {
        Err(e) => println!("no user id -> {e}"),
        Ok(r) => println!("no user id -> {}", r.text()),
    }
    Ok(())
}
```

| `PurviewSettings` | Default | Effect |
| --- | --- | --- |
| `new(app_name)` | | Required; sent as the integrated and protected app name |
| `with_app_version` | none | |
| `with_tenant_id` | none | Required: unlike Python, not read from the token |
| `with_purview_app_location(PurviewAppLocation::new(type, value))` | none | Required: `Application`, `Uri` or `Domain` |
| `with_graph_base_uri` | `https://graph.microsoft.com/v1.0/` | |
| `with_blocked_prompt_message`, `with_blocked_response_message` | `Prompt blocked by policy`, `Response blocked by policy` | |
| `with_ignore_exceptions(true)` | `false` | Let content through when evaluation fails (fail open) |
| `with_ignore_payment_required(true)` | `false` | Let content through on HTTP 402, checked separately |
| `with_cache_ttl_seconds`, `with_max_cache_size_bytes` | 14,400; 200 MiB | Carried for parity; unused, since nothing is cached |

A response is a block when any policy action is `blockAccess` or has `restrictionAction: block`; `restrictAccess` alone is not. An action the crate does not know fails the parse, and so, by default, the request. Tokens come from a `TokenProvider` (`StaticTokenProvider` for a fixed one) with the Graph scope; there is no built-in credential chain. A streamed agent run with `PurviewAgentMiddleware` is buffered so the response can be checked; `PurviewChatMiddleware` cannot see a streamed model call's response, so use the agent middleware when streaming. Not ported: the protection-scopes pre-check and its cache, background content-activity logging, and deriving the tenant, user or app location from the token. Every request sends `Prefer: evaluateInline`.

## Copilot Studio

`CopilotStudioAgent` calls a published Copilot Studio agent through the Direct-to-Engine API and implements `SupportsAgentRun`. `CopilotStudioConnectionSettings::new(environment_id, schema_name)` (or `from_settings(&CopilotStudioSettings::from_env())`, which reads `COPILOTSTUDIOAGENT__ENVIRONMENTID`, `__SCHEMANAME`, `__AGENTAPPID` and `__TENANTID`) builds the conversation URLs from the environment id and cloud. `with_cloud(PowerPlatformCloud::...)` (default `Prod`; `Gov`, `High`, `DoD`, `Mooncake` and test clouds; `Other` with `with_custom_power_platform_cloud(host)`), `with_agent_type(AgentType::Prebuilt)` and `with_direct_connect_url(url)` (bypasses host construction) adjust it.

```rust
use agent_framework::copilotstudio::settings::{AgentType, PowerPlatformCloud};
use agent_framework::copilotstudio::{
    CopilotStudioAgent, CopilotStudioConnectionSettings, CopilotStudioSettings, StaticTokenProvider,
};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // From COPILOTSTUDIOAGENT__ENVIRONMENTID and __SCHEMANAME when set.
    let settings = CopilotStudioSettings::from_env();
    let connection = match CopilotStudioConnectionSettings::from_settings(&settings) {
        Ok(connection) => connection,
        Err(_) => CopilotStudioConnectionSettings::new("8f1d2c3b-4a5e-6f70-8192-a3b4c5d6e7f8", "cr123_travelAgent"),
    }
    .with_cloud(PowerPlatformCloud::Prod)
    .with_agent_type(AgentType::Published);

    // The Direct-to-Engine URLs, derived from the environment id and cloud.
    println!("start: {}", connection.conversation_url(None)?);
    println!("turn:  {}", connection.conversation_url(Some("conv-1"))?);

    let Ok(token) = std::env::var("POWER_PLATFORM_TOKEN") else {
        println!("set POWER_PLATFORM_TOKEN (scope https://api.powerplatform.com/.default) to call the agent");
        return Ok(());
    };
    let agent = CopilotStudioAgent::new(connection, StaticTokenProvider::new(token)).with_name("travel");
    let mut session = agent.create_session();
    agent.run(vec![Message::user("Book me a train to York.")], Some(&mut session)).await?;
    // Same session, same Copilot Studio conversation.
    let reply = agent.run(vec![Message::user("Make it first class.")], Some(&mut session)).await?;
    println!("{}", reply.text());
    Ok(())
}
```

Authentication is bring-your-own token through `TokenProvider`, with the scope `https://api.powerplatform.com/.default`; the MSAL interactive login the Python package performs is not reproduced. A session keeps the Direct-to-Engine conversation id, so reusing it continues the conversation, whereas Python starts a new conversation on every call. Only the newest message is sent each time, and the internal `subscribe` endpoint is not implemented.

## Shell tools

The `tools` feature (`agent_framework::builtin_tools::shell`) gives an agent a shell. Both tools implement `ShellExecutor` (`start`, `run(command, timeout)`, `close`) and become a function tool with `as_function()` (named `run_shell`) or `as_function_with(name, description)`. Avoid a name another, auto-approved tool uses, since approval can be matched by name.

`LocalShellTool::builder()` runs commands on the host:

| Method | Default | Effect |
| --- | --- | --- |
| `mode(ShellMode)` | `Persistent` | One long-lived shell (`cd` and exports carry over) or a fresh process per command (`Stateless`) |
| `shell(spec)` | `AGENT_FRAMEWORK_SHELL`, then `bash`/`sh` (`pwsh` on Windows) | A command line or argv |
| `workdir(dir)` | the process's | Starting directory |
| `confine_workdir(bool)` | `true` | Re-enter the workdir before each command; not confinement, since one command can still `cd` elsewhere |
| `env`, `envs`, `clean_env(bool)` | inherit | `clean_env(true)` gives commands only the listed variables, keeping host secrets out |
| `policy(ShellPolicy)` | allow all | Allow and deny patterns, checked before approval |
| `timeout(Option<Duration>)` | 30 s | Kills the command's whole process group |
| `max_output_bytes(n)` | 64 KiB | Per stream, truncated head and tail |
| `approval_mode(mode)` | `AlwaysRequire` | `NeverRequire` fails to build without `acknowledge_unsafe(true)` |
| `on_command(f)` | none | Audit hook for every command the policy admits |

`ShellPolicy::new()` has no patterns; `with_denylist`, `with_allowlist`, `with_deny_regex`, `with_allow_regex` and `with_custom` add rules (case-insensitive; empty commands are always denied; deny, then allowlist, then the custom rule). It is a convenience for clear errors, not a security boundary: `r\m`, variable expansion or an interpreter defeat any pattern. A rejected command is returned to the model as text rather than failing the run. `ShellEnvironmentProvider` probes the shell, OS, working directory and CLI versions (`git`, `node`, `python`, `docker` by default) and tells the model about them. A persistent shell holds one conversation's state; build one per session and do not share it between users.

```rust
use std::sync::Arc;
use std::time::Duration;

use agent_framework::builtin_tools::shell::{
    LocalShellTool, ShellEnvironmentProvider, ShellEnvironmentProviderOptions, ShellExecutor, ShellMode, ShellPolicy,
};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let workdir = std::env::temp_dir();
    let shell = LocalShellTool::builder()
        .mode(ShellMode::Persistent) // the default: one long-lived shell
        .shell("bash --norc")
        .workdir(&workdir)
        .clean_env(true) // commands see only the variables below
        .env("PATH", "/usr/bin:/bin")
        .policy(ShellPolicy::new().with_denylist([r"\bsudo\b", r"\bssh\b"])?)
        .timeout(Some(Duration::from_secs(10)))
        .max_output_bytes(16 * 1024)
        .on_command(|command| println!("  audit: {command}"))
        .build()?;

    // Driving the executor directly: policy and audit apply, approval does not.
    shell.run("export GREETING=hello; cd /", None).await?;
    let r = shell.run("echo $GREETING from $(pwd)", None).await?;
    println!("persistent: {} (exit {})", r.stdout.trim(), r.exit_code);
    let slow = shell.run("sleep 5", Some(Duration::from_millis(300))).await?;
    println!("timed out: {}", slow.timed_out);
    match shell.run("sudo reboot", None).await {
        Err(e) => println!("policy: {e}"),
        Ok(_) => println!("policy let it through"),
    }

    // What the model is told about its shell.
    let options = ShellEnvironmentProviderOptions { probe_tools: vec!["git".into()], ..Default::default() };
    let environment = ShellEnvironmentProvider::new(Arc::new(shell.clone()), Some(options));
    let snapshot = environment.refresh().await?;
    println!("env: {:?} {} tools={:?}", snapshot.family, snapshot.os_description, snapshot.tool_versions);

    // As a tool: every call needs approval unless you acknowledge otherwise.
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are a careful operator.")
        .tool(shell.as_function())
        .context_provider(Arc::new(environment))
        .build();
    let response = agent.run_once("How much disk space is free?").await?;
    for request in response.user_input_requests() {
        println!("needs approval: {} {:?}", request.function_call.name, request.function_call.arguments);
    }
    shell.close().await?;
    Ok(())
}
```

The model's command waited for approval, as the default intends. `ShellResult` has `stdout`, `stderr`, `exit_code`, `duration`, `truncated` and `timed_out`, with `format_for_model`, and `to_command_output`/`to_tool_result` for the hosted-shell content types: the function is an ordinary function tool rather than upstream's `kind="shell"` marker, so the OpenAI Responses client does not declare it as a native shell.

`DockerShellTool::builder()` runs commands in a container with restrictive defaults: no network, a read-only root file system, user `65534:65534`, all capabilities dropped, `no-new-privileges`, 512 MB of memory and 256 processes, with `host_workdir` mounted read-only at `/workspace`. With those defaults kept, `approval_mode(ApprovalMode::NeverRequire)` needs no acknowledgement, because the container is the boundary; weakening any of them (`user`, `network`, `memory`, `pids_limit`, `read_only_root(false)`, `mount_readonly(false)`) makes `build` require `acknowledge_unsafe(true)`. `extra_run_args` rejects flags that would undo the defaults; `docker_binary("podman")` uses Podman. The image defaults to `mcr.microsoft.com/azurelinux/base/core:3.0` and the shell to `bash`. Compiled, not run (no Docker daemon was available):

```rust
use agent_framework::builtin_tools::shell::{ShellExecutor, ShellMode};
use agent_framework::prelude::*;
use agent_framework::tools::ApprovalMode;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let sandbox = DockerShellTool::builder()
        .image("python:3.12-slim")
        .shell("sh") // the image has no bash
        .mode(ShellMode::Persistent)
        .host_workdir(std::env::current_dir()?) // mounted read-only at /workspace
        // The isolation defaults are kept (no network, read-only root,
        // unprivileged user, memory and process caps), so the container is
        // the boundary and approval may be switched off without acknowledgement.
        .approval_mode(ApprovalMode::NeverRequire)
        .build()?;
    println!("{}", sandbox.run_argv().join(" "));

    let result = sandbox.run("python -c 'print(6 * 7)'", None).await?;
    println!("{}", result.format_for_model());

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .tool(sandbox.as_function_with("run_in_sandbox", None))
        .build();
    println!("{}", agent.run_once("Count the Rust files in /workspace.").await?.text());
    sandbox.close().await?;
    Ok(())
}
```

## Other integrations

- **Foundry embeddings** (`foundry`): `FoundryEmbeddingClient`, for text only; see [providers](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_providers/).
- **Not ported**: the Cosmos DB memory provider, Azure AI Content Understanding, ChatKit, sandboxed code execution, the Claude Agent SDK and Telegram adapters, and the `lab` packages. The repository's `SCOPE.md` explains each decision.

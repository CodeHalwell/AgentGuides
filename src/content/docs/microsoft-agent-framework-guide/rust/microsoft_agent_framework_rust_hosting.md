---
title: "Microsoft Agent Framework (Rust) — Hosting and A2A"
description: "Serving agents and workflows over HTTP with agent-framework-rs: the DevUI-style Responses API and debug page, an OpenAI Chat Completions endpoint, A2A server and client (cards, tasks, streaming, push notifications), AG-UI server with thread snapshots and the AG-UI chat client, SSE behaviour, bearer and host guards, CORS and limits with axum."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Hosting and A2A"
  order: 11
---

The `hosting` feature turns agents and workflows into [axum](https://docs.rs/axum) routers, one per wire protocol, and the `a2a` feature adds a client for remote agents that speak Agent2Agent. Each router is an ordinary `axum::Router`, so you serve one alone, merge several into one application, or nest them under paths beside your own routes.

| Router | Routes | Serves | Clients |
| --- | --- | --- | --- |
| `AgentHost` | `GET /health`, `GET /v1/entities`, `GET /v1/entities/{id}/info`, `POST /v1/responses`, debug page at `/` and `/ui` | Several agents and workflows, by name | OpenAI Responses clients, DevUI-style tools |
| `openai_compat::OpenAiRouter` | `POST /v1/chat/completions` | One agent | OpenAI Chat Completions clients |
| `a2a::A2ARouter` | `GET /.well-known/agent-card.json`, `POST /` (JSON-RPC) | One agent | A2A clients, including `A2AAgent` |
| `agui::AgUiRouter` | `POST {path}` per agent (default `/`) | One or more agents | CopilotKit and other AG-UI frontends, `AgUiChatClient` |

Every router accepts an `Agent`, a `WorkflowAgent` (a [workflow wrapped as an agent](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#a-workflow-as-an-agent)) or an `Arc<dyn SupportsAgentRun>`, through the `IntoAgentRegistration` trait. Every example here was compiled against %%version=agent-framework-rs%% and run, with the server and its client in one process and a stand-in for the OpenAI API behind the hosted agents.

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["hosting", "a2a"] }
tokio = { version = "1", features = ["full"] }
axum = "0.8"
serde_json = "1"
futures = "0.3"
```

The `agui-client` feature adds `AgUiChatClient` and implies `hosting`; it is separate so that a server does not pull in an HTTP client. `full` includes all three.

## The DevUI-style API

`AgentHost` is a registry. `agent(name, agent)` and `workflow(name, workflow)` register entities under a name that becomes the entity id in URLs and requests; registering the same name again replaces the earlier entity. `into_router()` builds the router, and `serve(addr)` binds and serves it alone.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use futures::StreamExt;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A tiny HTTP/1.1 GET, to show the raw discovery JSON without an HTTP crate.
async fn get(addr: &str, path: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let request = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut raw = String::new();
    stream.read_to_string(&mut raw).await.unwrap();
    raw.split("\r\n\r\n").nth(1).unwrap_or_default().to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    let assistant = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("assistant")
        .description("Answers general questions.")
        .instructions("You are concise.")
        .build();
    let shout = FunctionExecutor::new("shout", |message, ctx| async move {
        ctx.yield_output(json!(message.as_str().unwrap_or_default().to_uppercase())).await
    });
    let workflow = WorkflowBuilder::new()
        .name("shout")
        .description("Upper-cases its input.")
        .add_executor(Arc::new(shout))
        .set_start("shout")
        .build()?;

    // The registration name is the entity id used in URLs and requests.
    let app = AgentHost::new()
        .agent("assistant", assistant)
        .workflow("shout", workflow)
        .into_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    println!("{}", get(&addr, "/health").await);
    println!("{}", get(&addr, "/v1/entities").await);

    // POST /v1/responses speaks the OpenAI Responses API, so the crate's own
    // Responses client works against it. The `model` names the entity.
    let base = format!("http://{addr}/v1");
    let to_agent = OpenAIChatClient::new("unused", "assistant").with_base_url(&base);
    let reply = to_agent.get_response(vec![Message::user("Hello")], ChatOptions::new()).await?;
    println!("agent: {}", reply.text());

    let to_workflow = OpenAIChatClient::new("unused", "shout").with_base_url(&base);
    let reply = to_workflow.get_response(vec![Message::user("quiet please")], ChatOptions::new()).await?;
    println!("workflow: {}", reply.text());

    // "stream": true returns OpenAI Responses SSE events.
    let mut stream = to_agent
        .get_streaming_response(vec![Message::user("Stream this")], ChatOptions::new())
        .await?;
    while let Some(update) = stream.next().await {
        print!("{}", update?.text_content());
    }
    println!();
    Ok(())
}
```

`POST /v1/responses` follows the OpenAI Responses shape, so any Responses client, including the crate's own `OpenAIChatClient`, can call it. The entity is chosen by `metadata.entity_id`, then `extra_body.entity_id`, then `model`; a missing or unknown id is an OpenAI-style error (`missing_entity_id`, 400; `entity_not_found`, 404). `input` may be a string, an array of input items (`{"type": "message", ...}` or `{role, content}`, plus `function_call` and `function_call_output` items) or an object.

- **An agent** receives the input as messages and runs with no session: runs are stateless, so a client sends the conversation it wants the agent to see. The response carries the text in `output_text`, one assistant message item, and a `function_call` item for each call the agent left for the client (a declaration-only tool). `status` is `incomplete`, with `incomplete_details`, when the model stopped on a content filter or token limit. When the agent reported no token usage, `usage` is estimated at four characters per token.
- **A workflow** receives a string: a string input as is, an array of items flattened to their text joined by newlines, an object passed through as JSON. Each workflow output becomes an assistant message, and pending `request_info` requests are listed under `pending_requests`. There is no endpoint to answer them, so a workflow that pauses for a human cannot be resumed over this API.

`GET /v1/entities` and `/v1/entities/{id}/info` return `EntityInfo` (`id`, `type`, `name`, `description`, `instructions` for agents, `executors`, `start_executor_id` and `input_schema` for workflows). An `Arc<dyn SupportsAgentRun>` exposes no description or instructions; wrap it as `AgentRegistration::new(agent).with_description(...).with_instructions(...)` to provide them. Tool lists and model names are never filled in, because the agent trait does not expose them.

The page at `/` and `/ui` is a small embedded HTML page that lists the entities and runs them against `/v1/responses`, showing the stream as it arrives. It is a debugging aid, not Microsoft's DevUI frontend, which is a separate TypeScript application and is not ported.

### Streaming over SSE

With `"stream": true` an agent run is framed as Responses events while it runs: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, one `response.output_text.delta` per text update, then the function-call items, `response.completed` with the aggregated response, and `data: [DONE]`. A failure mid-stream ends with an error event instead of `response.completed`. Function calls are held back until the stream ends, because a call the agent answers itself arrives with its result later, and announcing it on sight would ask the client to run work already done.

The producer writes into a bounded channel of 32 events. A slow client therefore applies backpressure to the agent rather than letting events pile up in memory, and a client that disconnects cancels the run, so it stops calling the model and tools for output nobody will read. A workflow is different: it runs to completion first, and its events (`ExecutorInvoked` as `response.output_item.added`, executor completion or failure as `response.output_item.done`, outputs, `response.request_info.requested`, and `response.workflow_event.completed` for the rest) are then sent in one go. The OpenAI-compatible router streams agents the same way as `AgentHost`; AG-UI streams with its own event names and no `[DONE]`.

An agent streams token by token only if its chat client does, and only when no agent middleware is attached: agent middleware needs the whole response, so a run with middleware is buffered and replayed as updates.

## The OpenAI-compatible endpoint

`OpenAiRouter::for_agent(name, agent)` serves `POST /v1/chat/completions` for one agent. It reads `messages`, `model` and `stream` from the request and nothing else: sampling options and client-supplied `tools` are ignored, and the agent's own configuration applies. The `model` field does not route (the router serves one agent) but is echoed in the response, defaulting to `name`. An assistant message's `tool_calls` and a `role: "tool"` message's `tool_call_id` are read back into function calls and results, so a client can run the calls of a declaration-only tool and continue the conversation. Usage is the agent's reported counts, or the four-characters-per-token estimate.

The next example composes this router with the others and secures the result.

## Securing and composing

Every guard is opt-in: a plain `into_router()` is unauthenticated. `HostingSecurity` provides two:

| Method | Effect |
| --- | --- |
| `with_bearer_token(token)` | Requires `Authorization: Bearer <token>`; anything else gets 401 `unauthorized` in the OpenAI error shape |
| `with_allowed_hosts(vec![...])` | Rejects (403 `host_not_allowed`) a request whose `Host` header, port removed, is not listed. A request with no `Host` header is let through |
| `with_default_localhost_hosts()` | The allowlist `localhost`, `127.0.0.1`, `[::1]`, `::1` |
| `apply(router)` | Wraps the router; the host check runs before the token check |

The host allowlist defends a server bound to localhost against DNS rebinding, where a web page points a name it controls at `127.0.0.1` and has the user's browser call your agent. `AgentHost` has the same two settings as builder methods, plus `into_secure_router()` and `serve_secure()`, which apply the localhost allowlist by default. Those guard only the DevUI routes `AgentHost` builds, not routers merged or nested afterwards, which would sit outside the guard. When you serve more than one surface, compose the whole application first and wrap it once with `HostingSecurity`:

```rust
use std::time::Duration;

use agent_framework::hosting::agui::AgUiRouter;
use agent_framework::hosting::openai_compat::OpenAiRouter;
use agent_framework::hosting::HostingSecurity;
use agent_framework::prelude::*;
use axum::extract::{DefaultBodyLimit, Request};
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const ORIGIN: &str = "https://app.example.com";

/// CORS for one known browser origin. Preflights are answered here, before
/// the bearer check, because browsers do not send credentials on them.
async fn cors(request: Request, next: Next) -> Response {
    let allowed = request.headers().get(header::ORIGIN).and_then(|o| o.to_str().ok()) == Some(ORIGIN);
    let mut response = if request.method() == Method::OPTIONS {
        StatusCode::NO_CONTENT.into_response()
    } else {
        next.run(request).await
    };
    if allowed {
        let h = response.headers_mut();
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static(ORIGIN));
        h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("authorization, content-type"));
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST"));
    }
    response
}

/// Bound the time to the response head. A streamed body continues after this.
async fn deadline(request: Request, next: Next) -> Response {
    match tokio::time::timeout(Duration::from_secs(60), next.run(request)).await {
        Ok(response) => response,
        Err(_) => StatusCode::GATEWAY_TIMEOUT.into_response(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let token = std::env::var("HOST_TOKEN").unwrap_or_else(|_| "dev-token".into());
    let assistant = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("assistant")
        .instructions("You are concise.")
        .build();

    // Compose every surface first...
    let app = AgentHost::new()
        .agent("assistant", assistant.clone())
        .into_router()
        .merge(OpenAiRouter::for_agent("assistant", assistant.clone()).into_router())
        .nest("/agui", AgUiRouter::for_agent("assistant", assistant).into_router());

    // ...then guard the whole app once. Layers added later wrap the earlier ones.
    let app = HostingSecurity::new()
        .with_bearer_token(&token)
        .with_default_localhost_hosts()
        .apply(app)
        .layer(DefaultBodyLimit::max(256 * 1024))
        .layer(middleware::from_fn(deadline))
        .layer(middleware::from_fn(cors));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let base = format!("http://{addr}/v1");

    // Any OpenAI Chat Completions client works; the model name is ignored for
    // routing, since the router serves one agent.
    let client = OpenAIChatCompletionClient::new(&token, "assistant").with_base_url(&base);
    let reply = client.get_response(vec![Message::user("Hello over HTTP")], ChatOptions::new()).await?;
    println!("buffered: {}", reply.text());

    let mut stream = client
        .get_streaming_response(vec![Message::user("And streamed")], ChatOptions::new())
        .await?;
    print!("streamed: ");
    while let Some(update) = stream.next().await {
        print!("{}", update?.text_content());
    }
    println!();

    let intruder = OpenAIChatCompletionClient::new("wrong-token", "assistant").with_base_url(&base);
    match intruder.get_response(vec![Message::user("Hi")], ChatOptions::new()).await {
        Err(e) => println!("wrong token: {e}"),
        Ok(_) => println!("wrong token accepted?"),
    }

    // A browser preflight succeeds without the token.
    let mut tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let preflight = format!(
        "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: {ORIGIN}\r\n\
         Access-Control-Request-Method: POST\r\nConnection: close\r\n\r\n"
    );
    tcp.write_all(preflight.as_bytes()).await.unwrap();
    let mut raw = String::new();
    tcp.read_to_string(&mut raw).await.unwrap();
    println!("preflight: {}", raw.lines().next().unwrap_or_default());
    Ok(())
}
```

The crate has no CORS, rate-limiting or body-size settings of its own; use axum and tower for them, as above. Layer order matters: the CORS layer is added last so that it wraps the bearer check, because a browser's preflight `OPTIONS` request carries no credentials. axum's `Json` extractor already refuses bodies over 2 MB; `DefaultBodyLimit::max` changes that. The timeout here bounds the time to the response head only, since a streamed body continues after the handler returns; bound a whole run with a timeout around the agent call, or with the agent's own [tool-loop limits](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_comprehensive_guide/#bounding-the-tool-loop). If you add `tower-http`, its `CorsLayer`, `TimeoutLayer` and `ConcurrencyLimitLayer` do the same jobs. TLS is not handled here either: terminate it in front of the server, because a bearer token over plain HTTP is only as private as the network.

The token is compared as a plain string, there is one per router, and it carries no identity. For per-user authorisation, put your own authentication middleware in front and pass the principal on through request extensions, which is what AG-UI's [authorisation scope](#thread-snapshots) reads.

## Serving A2A

`A2ARouter::for_agent(name, agent, public_url)` serves the agent card at `GET /.well-known/agent-card.json` and JSON-RPC 2.0 at `POST /`. `public_url` is the endpoint the card advertises, so it must be the URL clients can reach, including any path prefix you nest the router under.

| Method | Default | Effect |
| --- | --- | --- |
| `version(v)` | `1.0.0` | The card's `version` |
| `skill(AgentSkill)` | One skill from the agent's name and description | Replace the skill list |
| `add_skill(AgentSkill)` | | Add a skill |
| `max_tasks(n)` | 1,024 | Finished tasks kept for `tasks/get`, oldest evicted first |
| `task_ttl(d)`, `no_task_ttl()` | one hour | Expiry of kept tasks |
| `max_task_bytes(n)` | 16 MiB | Cap on the total serialised size of kept tasks |
| `card()` | | Inspect the card before serving |

The server is simple next to the client below. Every `message/send` runs the agent to completion and returns a `completed` task whose artifact (named `response`) and history hold the reply. Only the text parts of the incoming message are used, joined, and the agent runs with no session, so a reused `contextId` is echoed back but earlier turns are not remembered. There is no `working` or `input-required` state; `tasks/cancel` on a known task answers `TaskNotCancelable` (-32002). `message/stream` answers `UnsupportedOperation` (-32004), and push notifications, `tasks/resubscribe` and the authenticated extended card are not served. The card says so (`streaming: false`, `pushNotifications: false`).

```rust
use std::time::Duration;

use agent_framework::a2a::{Message as A2AMessage, MessageSendParams, Part, SendMessageResult, TextPart};
use agent_framework::hosting::a2a::{A2ARouter, AgentSkill};
use agent_framework::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/", listener.local_addr().unwrap());

    let analyst = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("analyst")
        .description("Weighs a question and gives a short verdict.")
        .build();
    let app = A2ARouter::for_agent("analyst", analyst, &base)
        .version("2.1.0")
        .add_skill(AgentSkill {
            id: "verdict".into(),
            name: "Verdict".into(),
            description: "Gives a yes/no verdict with one reason.".into(),
            tags: vec!["analysis".into()],
            examples: vec!["Is Rust a good fit for agents?".into()],
        })
        // Finished tasks are kept for tasks/get, within these bounds.
        .max_tasks(500)
        .task_ttl(Duration::from_secs(15 * 60))
        .max_task_bytes(8 * 1024 * 1024)
        .into_router();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // The protocol client.
    let client = A2AClient::from_url(&base).with_timeout(Duration::from_secs(30))?;
    let card = client.get_agent_card().await?;
    let skills: Vec<_> = card.skills.iter().map(|s| s.id.as_str()).collect();
    println!("{} v{} ({:?}) skills={skills:?} streaming={}", card.name, card.version, card.protocol_version, card.capabilities.streaming);

    let message = A2AMessage::user(vec![Part::Text(TextPart { text: "Is tea better than coffee?".into(), metadata: None })]);
    let task_id = match client.send_message(MessageSendParams::new(message)).await? {
        SendMessageResult::Message(m) => {
            println!("inline message with {} part(s)", m.parts.len());
            None
        }
        SendMessageResult::Task(task) => {
            println!("task {} is {:?} (terminal: {})", task.id, task.status.state, task.status.state.is_terminal());
            for artifact in task.artifacts.iter().flatten() {
                for part in &artifact.parts {
                    if let Part::Text(t) = part {
                        println!("artifact {:?}: {}", artifact.name, t.text);
                    }
                }
            }
            Some(task.id)
        }
    };

    if let Some(id) = &task_id {
        println!("tasks/get -> {:?}", client.get_task(id).await?.status.state);
        if let Err(e) = client.cancel_task(id).await {
            println!("tasks/cancel -> {e}");
        }
    }
    // This server does not stream. Its JSON-RPC error is not an SSE frame,
    // so the client sees an empty stream rather than an error: check the card.
    let probe = A2AMessage::user(vec![Part::Text(TextPart { text: "stream?".into(), metadata: None })]);
    let events: Vec<_> = client.send_message_stream(MessageSendParams::new(probe)).await?.collect().await;
    println!("message/stream events: {}", events.len());

    // The agent wrapper: a remote agent behind SupportsAgentRun.
    let remote = A2AAgent::from_url("analyst", &base);
    let reply = remote.run_once("Should I learn Rust?").await?;
    println!("A2AAgent: {} (response id {:?})", reply.text(), reply.response_id);
    Ok(())
}
```

## Calling A2A agents

The client side is in `agent_framework::a2a`. `A2AClient` is the protocol; `A2AAgent` wraps it as a `SupportsAgentRun`, so a remote agent can be run directly, wrapped as a tool, or placed in a workflow or orchestration like a local one.

| `A2AClient` method | JSON-RPC |
| --- | --- |
| `get_agent_card()` | `GET {origin}/.well-known/agent-card.json`, falling back to `/.well-known/agent.json` |
| `get_extended_card()` | `agent/getAuthenticatedExtendedCard` |
| `send_message(params)` | `message/send`, returning `SendMessageResult::Message` or `::Task` |
| `send_message_stream(params)` | `message/stream` (SSE), returning `A2AEventStream` |
| `get_task(id)`, `cancel_task(id)` | `tasks/get`, `tasks/cancel` |
| `set_push_notification_config(id, config)`, `get_push_notification_config(id)` | `tasks/pushNotificationConfig/set` and `/get` |
| `resubscribe(id)` | `tasks/resubscribe` (SSE) |

`A2AClient::from_url(url)` discovers the card lazily; `from_card(card)` never makes a discovery call. `with_bearer_token`, `with_header` and `with_timeout` return `Result`, because a bad header value fails at build time rather than on every request. Discovery uses the URL's origin, not its path, and on success switches the client to the endpoint the card names (`rpc_url()`); when the card sets `supportsAuthenticatedExtendedCard`, it fetches the extended card and uses that, falling back to the public one on failure. The card is cached (`cached_agent_card()`).

That origin rule has a consequence for nested routers. An `A2ARouter` nested at `/a2a` serves its card at `/a2a/.well-known/agent-card.json`, where discovery does not look, so `initialize()` fails. Runs still work, because `A2AAgent` treats discovery as best-effort and falls back to posting to the URL it was given; serve the router at the root of its own host or port if clients need the card.

Whether a server answers with a message or a task is its choice, so handle both. A task's `status.state` is a `TaskState`; `is_terminal()` is true for `Completed`, `Canceled`, `Failed` and `Rejected`, while `Submitted`, `Working`, `InputRequired` and `AuthRequired` are not. An unrecognised state parses as `Unknown`. To wait for a long task, poll `get_task` until it is terminal, or register a webhook with `PushNotificationConfig::new(url).with_token(secret)` (or `with_authentication`). Note that `tasks/pushNotificationConfig/get` sends the task id as `id`, not `taskId`, which is what the A2A 0.3 specification and SDK do.

### Streaming and the agent wrapper

`send_message_stream` returns `MessageStreamEvent`s: `Message`, `Task`, `StatusUpdate` (with `is_final`, the wire's `final`) and `ArtifactUpdate` (with `append` and `last_chunk`). Against a stand-in server that streams a task through `working`, an artifact and `completed`:

```rust
use std::convert::Infallible;

use agent_framework::a2a::{
    Message as A2AMessage, MessageSendParams, MessageStreamEvent, Part, TaskState, TextPart,
};
use agent_framework::prelude::*;
use axum::response::sse::{Event, Sse};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Value};

/// A stand-in for a streaming A2A server: working -> artifact -> completed.
async fn rpc(Json(request): Json<Value>) -> Sse<impl futures::Stream<Item = std::result::Result<Event, Infallible>>> {
    let id = request["id"].clone();
    let context = request["params"]["message"]["contextId"].as_str().unwrap_or("ctx-1").to_string();
    let frames = vec![
        json!({ "kind": "status-update", "taskId": "task-1", "contextId": context,
                "status": { "state": "working" }, "final": false }),
        json!({ "kind": "artifact-update", "taskId": "task-1", "contextId": context,
                "artifact": { "artifactId": "a1", "name": "report",
                              "parts": [{ "kind": "text", "text": "Draft findings." }] } }),
        json!({ "kind": "status-update", "taskId": "task-1", "contextId": context,
                "status": { "state": "completed" }, "final": true }),
    ];
    let events = frames.into_iter().map(move |result| {
        Ok(Event::default().data(json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string()))
    });
    Sse::new(futures::stream::iter(events))
}

#[tokio::main]
async fn main() -> Result<()> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, Router::new().route("/", post(rpc))).await.unwrap() });

    // Raw events from the protocol client.
    let client = A2AClient::from_url(&url);
    let message = A2AMessage::user(vec![Part::Text(TextPart { text: "Research Rust".into(), metadata: None })]);
    let mut events = client.send_message_stream(MessageSendParams::new(message)).await?;
    while let Some(event) = events.next().await {
        match event? {
            MessageStreamEvent::StatusUpdate(s) => {
                println!("status {:?} final={}", s.status.state, s.is_final);
                if s.status.state == TaskState::InputRequired {
                    // reply on the same task with a message carrying s.task_id
                }
            }
            MessageStreamEvent::ArtifactUpdate(a) => println!("artifact {:?} (append={})", a.artifact.name, a.append),
            MessageStreamEvent::Task(t) => println!("task {}", t.id),
            MessageStreamEvent::Message(m) => println!("message {}", m.message_id),
        }
    }

    // The agent wrapper maps the same events to AgentResponseUpdates.
    let remote = A2AAgent::from_url("researcher", &url);
    let session = remote.create_session();
    let mut stream = remote.run_stream(vec![Message::user("Research Rust")], Some(session), None).await?;
    while let Some(update) = stream.next().await {
        let update = update?;
        println!("update text={:?} response_id={:?}", update.text(), update.response_id);
    }
    Ok(())
}
```

`A2AAgent` sends only the newest message of each run, because the remote agent holds the earlier turns under its `contextId`. It keeps the context and task ids in the session's `service_session_id` (as JSON), so reusing an `AgentSession` continues the same remote conversation, including answering a task paused in `input-required`. Python's `A2AAgent` accepts a thread but never reads or writes it. Outgoing text, error, URI, data and hosted-file content map to A2A parts; function calls and results cannot be sent and fail with `Error::Content`. Incoming file parts become `Content::Uri` or `Content::Data`, and data parts become JSON text.

`run` makes one `message/send` call and maps what comes back: a message as one response message; a task as one message per artifact, followed by the task's status message when it is waiting for input or authorisation or has failed, been cancelled or rejected (so you see the question or the reason, as in .NET). A task still `working` gives no messages and the task id as `response_id`; poll it through `agent.client()`. `run_stream` uses `message/stream` and maps each event to an `AgentResponseUpdate` whose `response_id` is the task id. Per-run options have no A2A equivalent and are ignored with a warning.

Two cautions. A server that does not stream answers `message/stream` with a JSON-RPC error, not an SSE frame, and the client then yields an empty stream rather than that error, so check `card.capabilities.streaming` before relying on `run_stream`. And the `run_stream` path writes the ids back to its own copy of the session, which a clone taken before streaming does not see; use `run` for multi-turn conversations. Only JSON-RPC over HTTP is spoken: there is no gRPC or REST transport negotiation, and the card's `securitySchemes` are kept as raw JSON rather than acted on.

## AG-UI

`AgUiRouter::for_agent(name, agent)` serves the AG-UI protocol at `POST /`; `path(p)` moves it and `agent(name, agent, path)` adds more agents at other paths. A request is a `RunAgentInput` (`threadId`, `runId`, `messages`, `tools`, `state`, `context`, `forwardedProps`, `resume`); the response is a stream of SSE events with no `[DONE]`:

| Event | When |
| --- | --- |
| `RUN_STARTED` | First, with `threadId` and `runId` |
| `STATE_SNAPSHOT` | With a snapshot store, at the start |
| `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` | Assistant text, one content event per update |
| `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `TOOL_CALL_RESULT` | Tool calls; a call with no result (a frontend tool) is closed without `TOOL_CALL_RESULT` |
| `CUSTOM` (`function_approval_request`) | A tool call waiting for approval |
| `MESSAGES_SNAPSHOT` | With a snapshot store, at the end |
| `RUN_FINISHED` or `RUN_ERROR` | Last |

The frontend's `tools` are passed to the agent for that run as declaration-only tools, so when the model calls one the call goes back to the browser to execute. Only an `Agent` accepts per-run tools; another `SupportsAgentRun` logs a warning and ignores them. `context` and `forwardedProps` are accepted and ignored. The constants are in `agui::event_type`.

### The AG-UI client

With `agui-client`, `AgUiChatClient::new(endpoint)` is a `ChatClient` backed by a remote AG-UI server, so an AG-UI server can be the model of a local agent. The local agent's function tools are declared to the server; when the server's model calls one, the call returns as a `FunctionCallContent` with no result and the local tool loop runs it and sends the result on the next request. Here the server agent has no tools and the client agent supplies one:

```rust
use agent_framework::hosting::agui::{AgUiChatClient, AgUiRouter};
use agent_framework::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // Server: one agent at POST /agent. It has no tools of its own.
    let server_agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("assistant")
        .instructions("Use the client's tools when the user asks about their screen.")
        .build();
    let app = AgUiRouter::for_agent("assistant", server_agent).path("/agent").into_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/agent", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Client: an agent whose model is the remote AG-UI server. Its tool is
    // declared to the server, called by the server's model, and run here.
    let selection = FunctionTool::new(
        "get_selection",
        "Return the text the user has selected on screen.",
        json!({ "type": "object", "properties": {} }),
        |_| async move { Ok(json!("fn main() {}")) },
    );
    let frontend = Agent::builder(AgUiChatClient::new(&endpoint))
        .tool(selection.into_definition())
        .build();
    let mut session = frontend.create_session();
    let reply = frontend
        .run(vec![Message::user("What have I selected?")], Some(&mut session))
        .await?;
    println!("{}", reply.text());
    Ok(())
}
```

The client sends exactly the messages it is given and keeps no history, so use a session on the wrapping agent, or drive a snapshot server with a fixed thread id. It takes the thread id from `ChatOptions::metadata["thread_id"]` (a fresh one otherwise), sends `instructions` as a leading system message, turns a message built with `state_carrier(map)` into the run's `state`, and reads `available_interrupts`, `resume`, `forwarded_props` and `context` from `additional_properties`. Server-side tool results come back as call and result, which the tool loop treats as already answered. State events arrive as JSON `DataContent`, `MESSAGES_SNAPSHOT` and `CUSTOM` events in `additional_properties`, and `RUN_ERROR` as an `ErrorContent` in an otherwise successful response. `with_timeout` sets the connect and per-read timeout (60 seconds by default) and `with_http_client(endpoint, reqwest::Client)` takes your own client for headers or a proxy, which needs a direct `reqwest` 0.12 dependency.

### Thread snapshots

By default the router keeps nothing between requests and ignores inbound `state`. With `snapshot_store(Arc::new(InMemoryAgUiThreadSnapshotStore::new()))` it keeps, per thread, the transcript, the shared state, pending approvals and the agent session's private state:

- A request with messages runs the agent against the thread's restored session. Only what is new is added to the stored transcript, matched by message id or by role and content, and only user turns and results for calls the server issued are kept, so a client cannot plant `system` or `assistant` turns. Request `state` is merged over stored state, except values shaped like tool-approval state and keys the server owns or the framework reserves (`__af_`, `__ag_ui`).
- A request with no messages and no `resume` replays the thread without running the agent: `RUN_STARTED`, `STATE_SNAPSHOT`, `MESSAGES_SNAPSHOT`, `RUN_FINISHED`.
- A `resume` must answer exactly the interrupts stored for that thread; the approved call is rebuilt from the stored record, not from the request, and marked spent before it runs, so it runs at most once. A thread with pending interrupts refuses other turns (`APPROVAL_RESUME_REQUIRED`).

A thread id identifies a thread but does not authorise access to it, so a store must be paired with an authorisation mode, or `try_into_router()` returns a configuration error (and `into_router()` panics). `authorization_scope(|parts| ...)` resolves a scope from the request head, typically from claims your authentication layer put in the extensions; `Err(status)` rejects the request before any state is read. `single_user()` declares one principal per router, with a scope that does not survive a restart. Snapshots are keyed by scope and thread id:

```rust
use std::sync::Arc;

use agent_framework::hosting::agui::{AgUiRouter, InMemoryAgUiThreadSnapshotStore};
use agent_framework::prelude::*;
use axum::http::StatusCode;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// POST a RunAgentInput and return the AG-UI event types in order.
async fn run(addr: &str, user: &str, body: Value) -> Vec<String> {
    let body = body.to_string();
    let mut tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let request = format!(
        "POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nX-User: {user}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    tcp.write_all(request.as_bytes()).await.unwrap();
    let mut raw = String::new();
    tcp.read_to_string(&mut raw).await.unwrap();
    if !raw.starts_with("HTTP/1.1 200") {
        return vec![raw.lines().next().unwrap_or_default().to_string()];
    }
    raw.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .map(|event| event["type"].as_str().unwrap_or_default().to_string())
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .name("assistant")
        .build();
    let app = AgUiRouter::for_agent("assistant", agent)
        .snapshot_store(Arc::new(InMemoryAgUiThreadSnapshotStore::new()))
        // In production, read the principal an authentication layer stored in
        // the request extensions; a bare header is for illustration only.
        .authorization_scope(|parts| {
            parts
                .headers
                .get("x-user")
                .and_then(|v| v.to_str().ok())
                .map(|user| format!("user:{user}"))
                .ok_or(StatusCode::UNAUTHORIZED)
        })
        .try_into_router()?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let turn = |text: &str, id: &str| json!({
        "threadId": "t-1", "runId": id,
        "messages": [{ "id": format!("m-{id}"), "role": "user", "content": text }],
        "state": { "theme": "dark" }
    });
    println!("turn 1: {:?}", run(&addr, "alice", turn("Hello", "r1")).await);
    // Only the new message: the server holds the transcript.
    println!("turn 2: {:?}", run(&addr, "alice", turn("Still there?", "r2")).await);
    // No messages: the stored thread is replayed without running the agent.
    let hydrate = json!({ "threadId": "t-1", "runId": "r3", "messages": [] });
    println!("hydrate: {:?}", run(&addr, "alice", hydrate.clone()).await);
    // Another principal with the same thread id sees a different (empty) thread.
    println!("bob: {:?}", run(&addr, "bob", hydrate).await);
    Ok(())
}
```

Turns on one thread run one at a time within the process, so a double-click cannot resume one approval twice; replicas sharing a store need the store to serialise writes. If saving fails, the run ends with `RUN_ERROR` (`SNAPSHOT_SAVE_FAILED`) and the previous snapshot stands; a failed or abandoned run is not saved. `InMemoryAgUiThreadSnapshotStore` holds at most 1,000 snapshots (`with_max_snapshots`) and evicts the least recently saved; implement `AgUiThreadSnapshotStore` (`save`, `get`, `delete`, `clear`) for a durable one. All routes of one router share the store, so two agents on one router that reuse a thread id share the thread. Do not serve an agent that has a default `conversation_id` in its chat options: the stored transcript is sent in full on every run, so it would receive every turn twice, and on any hosting route it would share one service conversation across callers.

Not ported: `STATE_DELTA`, predictive state updates (`predict_state_config`, `confirm_changes`) and the state schema, because the agent trait has nothing that produces them.

## Reusing the Responses conversion

The OpenAI Responses types behind `AgentHost` are a standalone module, `hosting::responses` (re-exported at `hosting::`): `ResponsesRequest`, `responses_to_run(&request) -> Vec<Message>`, `responses_from_run(&response, id, model) -> ResponseObject`, and `openai_error(...)` for error bodies. Use them to write an endpoint with your own routing, authentication or session handling while keeping the wire format. This one picks the agent from a tenant header:

```rust
use std::sync::Arc;

use agent_framework::hosting::responses::openai_error;
use agent_framework::hosting::{responses_from_run, responses_to_run, ResponsesRequest};
use agent_framework::prelude::*;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};

/// A Responses-compatible endpoint with your own routing: here the agent is
/// chosen by a tenant header rather than by `model`.
async fn responses(
    State(agents): State<Arc<Vec<(String, Agent)>>>,
    headers: HeaderMap,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    let tenant = headers.get("x-tenant").and_then(|v| v.to_str().ok()).unwrap_or_default();
    let Some((name, agent)) = agents.iter().find(|(t, _)| t == tenant) else {
        let body = openai_error("unknown tenant", "invalid_request_error", Some("unknown_tenant"));
        return (StatusCode::NOT_FOUND, Json(body)).into_response();
    };
    match agent.run(responses_to_run(&request), None).await {
        Ok(run) => Json(responses_from_run(&run, &format!("resp_{tenant}"), name)).into_response(),
        Err(e) => {
            let body = openai_error(e.to_string(), "server_error", None);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let make = |tone: &str| -> Result<Agent> {
        Ok(Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
            .instructions(format!("Answer in a {tone} tone."))
            .build())
    };
    let agents = Arc::new(vec![("acme".to_string(), make("formal")?), ("globex".to_string(), make("casual")?)]);
    let app = Router::new().route("/v1/responses", post(responses)).with_state(agents);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Plain JSON over a raw socket, to keep the example dependency-free.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let body = r#"{"model":"any","input":"Hello"}"#;
    let mut tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Tenant: acme\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    tcp.write_all(request.as_bytes()).await.unwrap();
    let mut raw = String::new();
    tcp.read_to_string(&mut raw).await.unwrap();
    let json: serde_json::Value = serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    println!("{} {} -> {}", json["id"], json["status"], json["output_text"]);
    Ok(())
}
```

## What is not here

There is no Azure Functions, Durable Task or Foundry hosted-agent hosting, no conversation store or run-resume endpoint behind `/v1/responses`, no `/v1/models` route, and no MCP server (see [exposing an agent over MCP](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/#exposing-an-agent-as-an-mcp-server)). The .NET ASP.NET Core hosting extensions map onto these routers rather than being ported. Telemetry for hosted runs is the agents' own; see [observability](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/).

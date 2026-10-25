---
title: "Microsoft Agent Framework (Rust) — Testing and production"
description: "Testing agent-framework-rs agents with a scripted ChatClient and cargo test: tools, approvals, streaming, structured output, middleware, retries, workflows, checkpoints, orchestrations, MCP and HTTP hosting, golden and property-style tests, gated live tests and CI. Then running in production: timeouts, retries, cancellation, concurrency, secrets, telemetry, deployment and graceful shutdown."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Testing and production"
  order: 14
---

The crate ships no public fake or mock chat client, and you do not need one. `ChatClient` is a trait with two required methods, so a test double that replays scripted replies takes about a hundred lines. Everything above the client then runs for real in `cargo test`, with no network and no API key: the tool loop, approvals, sessions, middleware, retries, workflows, orchestrations, MCP clients and the HTTP routers.

Every test on this page passes against `agent-framework` %%version=agent-framework-rs%%. Where a test pins down behaviour that is easy to get wrong (when budgets are checked, what a cancelled run stores, what a group-chat manager appends), the assertion is the documentation.

## Setting up

Integration tests live in `tests/`, one binary per file, with shared helpers in `tests/common/mod.rs` (a subdirectory, so Cargo does not treat it as a test of its own). Each test file starts with `mod common;`.

```toml
[dev-dependencies]
# Turn on what the tests exercise; dev-dependency features apply to test builds only.
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["mcp", "hosting"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread", "time", "sync"] }
async-trait = "0.1"
futures = "0.3"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
schemars = "0.8"   # FunctionTool::typed and schema_for!
axum = "0.8"       # only for the hosting tests
```

| Test file | Needs feature | Covers |
| --- | --- | --- |
| `test_agent.rs` | default (`openai`, for the prelude) | Tool loop, parallel calls, tool errors, approvals, budgets, per-run options, sessions |
| `test_streaming.rs` | default | Streaming, aggregation, history write-back, structured output |
| `test_tools.rs` | default | Tools without an agent, golden schema files, property-style checks |
| `test_middleware.rs` | default | Agent, chat and function middleware, ordering, short-circuits |
| `test_resilience.rs` | default | Retries, `Retry-After`, timeouts, cancellation, concurrency limits |
| `test_workflows.rs` | default | Checkpoints and restarts, events, sequential, concurrent, handoff, group chat |
| `test_mcp.rs` | `mcp` | An in-process MCP server |
| `test_hosting.rs` | `hosting` | The OpenAI-compatible router over real HTTP |
| `test_live.rs` | default | A real model, ignored unless asked for |

## A scripted chat client

`ScriptedClient` replays a queue of replies in order and records every request, so a test can check both what the agent did and what it sent. `FnClient` computes each reply from the request instead, which is what you want when the order of calls is not fixed: several agents sharing one client, or a concurrent orchestration.

```rust
// tests/common/mod.rs
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework::prelude::*;
use agent_framework::types::{FunctionArguments, UsageContent};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

/// One request the agent sent to the model.
#[derive(Clone, Debug)]
pub struct Request {
    pub messages: Vec<Message>,
    pub options: ChatOptions,
}

impl Request {
    /// The text of every message, in order (the system message included).
    pub fn texts(&self) -> Vec<String> {
        self.messages.iter().map(Message::text).collect()
    }

    /// The names of the tools offered on this request.
    pub fn tool_names(&self) -> Vec<String> {
        self.options.tools.iter().map(|t| t.name.clone()).collect()
    }

    /// Every tool result carried by this request.
    pub fn tool_results(&self) -> Vec<FunctionResultContent> {
        self.messages.iter().flat_map(|m| m.function_results()).cloned().collect()
    }
}

/// A ChatClient that replays scripted replies in order and records requests.
#[derive(Clone, Default)]
pub struct ScriptedClient {
    replies: Arc<Mutex<VecDeque<Result<ChatResponse>>>>,
    requests: Arc<Mutex<Vec<Request>>>,
    delay: Option<Duration>,
}

impl ScriptedClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// The model answers with text.
    pub fn text(self, text: &str) -> Self {
        self.push(Ok(ChatResponse {
            finish_reason: Some(FinishReason::stop()),
            usage_details: Some(UsageDetails {
                input_token_count: Some(10),
                output_token_count: Some(5),
                total_token_count: Some(15),
                ..Default::default()
            }),
            ..ChatResponse::from_text(text)
        }))
    }

    /// The model asks for one tool call.
    pub fn tool_call(self, name: &str, args: Value) -> Self {
        self.tool_calls(&[(name, args)])
    }

    /// The model asks for several tool calls in one turn (parallel calls).
    pub fn tool_calls(self, calls: &[(&str, Value)]) -> Self {
        let turn = self.replies.lock().unwrap().len();
        let contents = calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                let call_id = format!("call_{turn}_{i}");
                let arguments = Some(FunctionArguments::Raw(args.to_string()));
                Content::FunctionCall(FunctionCallContent::new(call_id, *name, arguments))
            })
            .collect();
        self.push(Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), contents)],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        }))
    }

    /// The service fails.
    pub fn error(self, error: Error) -> Self {
        self.push(Err(error))
    }

    /// Any hand-built response.
    pub fn reply(self, response: ChatResponse) -> Self {
        self.push(Ok(response))
    }

    /// Wait this long before every reply (for timeout and cancellation tests).
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    fn push(self, reply: Result<ChatResponse>) -> Self {
        self.replies.lock().unwrap().push_back(reply);
        self
    }

    /// Every request so far.
    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }

    /// Replies not yet used. A finished test usually expects zero.
    pub fn remaining(&self) -> usize {
        self.replies.lock().unwrap().len()
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        self.requests.lock().unwrap().push(Request { messages, options });
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        let next = self.replies.lock().unwrap().pop_front();
        next.unwrap_or_else(|| Err(Error::other("the script has no more replies")))
    }

    async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatStream> {
        // Replay the next reply as one update per word, so streaming code is exercised.
        let response = self.get_response(messages, options).await?;
        Ok(futures::stream::iter(to_updates(response)).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some("scripted-model")
    }
}

/// Split a response into the updates a streaming provider would send.
fn to_updates(response: ChatResponse) -> Vec<Result<ChatResponseUpdate>> {
    let mut updates = Vec::new();
    for message in response.messages {
        let text = message.text();
        if text.is_empty() {
            updates.push(Ok(ChatResponseUpdate {
                contents: message.contents,
                role: Some(message.role),
                ..Default::default()
            }));
        } else {
            for (i, word) in text.split(' ').enumerate() {
                let chunk = if i == 0 { word.to_string() } else { format!(" {word}") };
                updates.push(Ok(ChatResponseUpdate::text(chunk)));
            }
        }
    }
    let mut last = ChatResponseUpdate {
        finish_reason: response.finish_reason,
        ..Default::default()
    };
    if let Some(usage) = response.usage_details {
        last.contents.push(Content::Usage(UsageContent { details: usage }));
    }
    updates.push(Ok(last));
    updates
}

type Responder = dyn Fn(&[Message], &ChatOptions) -> Result<ChatResponse> + Send + Sync;

/// A ChatClient that computes each reply from the request. Use it where the
/// order of calls is not fixed: concurrent orchestrations, shared clients.
#[derive(Clone)]
pub struct FnClient {
    respond: Arc<Responder>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl FnClient {
    pub fn new(respond: impl Fn(&[Message], &ChatOptions) -> Result<ChatResponse> + Send + Sync + 'static) -> Self {
        Self { respond: Arc::new(respond), requests: Arc::default() }
    }

    /// Answer every request with the same text.
    pub fn always(text: &'static str) -> Self {
        Self::new(move |_, _| Ok(ChatResponse::from_text(text)))
    }

    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatClient for FnClient {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let reply = (self.respond)(&messages, &options);
        self.requests.lock().unwrap().push(Request { messages, options });
        reply
    }

    async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        Ok(futures::stream::iter(to_updates(response)).boxed())
    }
}

/// The text of the last user message in a request.
pub fn last_user_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::user())
        .map(Message::text)
        .unwrap_or_default()
}
```

Three details matter. Tool calls must come back as `Content::FunctionCall` for the function-invocation loop to run them, exactly as a provider client returns them. The agent's instructions arrive as a leading `system` message, so assertions on `messages` must allow for it. And the streaming replay sends usage as a `Content::Usage` item on the last update, which is how the provider clients report it and what `AgentResponse::from_updates` folds back into `usage_details`.

A script that runs out returns an error rather than panicking, so a test that makes one more model call than expected fails with "the script has no more replies". Checking `remaining() == 0` at the end catches the opposite mistake: a reply the agent never asked for.

## Agents, tools and approvals

```rust
// tests/test_agent.rs
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use common::ScriptedClient;
use serde_json::json;

/// A weather tool that counts its calls.
fn weather(calls: Arc<AtomicUsize>) -> FunctionTool {
    FunctionTool::new(
        "get_weather",
        "Get the weather for a city.",
        json!({ "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] }),
        move |args| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!({ "city": args["city"], "temperature": 21 }))
            }
        },
    )
}

#[tokio::test]
async fn the_tool_result_goes_back_to_the_model() {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = ScriptedClient::new()
        .tool_call("get_weather", json!({ "city": "York" }))
        .text("It is 21 degrees in York.");
    let agent = Agent::builder(client.clone())
        .instructions("Use tools.")
        .tool(weather(calls.clone()).into_definition())
        .build();

    let response = agent.run_once("Weather in York?").await.unwrap();

    assert_eq!(response.text(), "It is 21 degrees in York.");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let requests = client.requests();
    assert_eq!(requests.len(), 2);
    // Instructions arrive as a leading system message, and the tool is offered.
    assert_eq!(requests[0].messages[0].role, Role::system());
    assert_eq!(requests[0].texts(), ["Use tools.", "Weather in York?"]);
    assert_eq!(requests[0].tool_names(), ["get_weather"]);
    // The second request carries the call and its result.
    let result = &requests[1].tool_results()[0];
    assert_eq!(result.result.as_ref().unwrap()["city"], "York");
    assert_eq!(client.remaining(), 0);
    // The response holds the whole exchange: call, result, answer.
    assert_eq!(response.messages.len(), 3);
}

#[tokio::test]
async fn parallel_calls_all_run_before_the_next_model_turn() {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = ScriptedClient::new()
        .tool_calls(&[
            ("get_weather", json!({ "city": "York" })),
            ("get_weather", json!({ "city": "Leeds" })),
        ])
        .text("Both are mild.");
    let agent = Agent::builder(client.clone())
        .tool(weather(calls.clone()).into_definition())
        .build();

    agent.run_once("York and Leeds?").await.unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let results = client.requests()[1].tool_results();
    // Results come back in the order the model asked, whatever order they finished in.
    let cities: Vec<_> = results.iter().map(|r| r.result.as_ref().unwrap()["city"].clone()).collect();
    assert_eq!(cities, [json!("York"), json!("Leeds")]);
}

#[tokio::test]
async fn a_failing_tool_is_reported_to_the_model_not_the_caller() {
    let failing = FunctionTool::new("lookup", "Look up.", json!({ "type": "object" }), |_| async {
        Err(Error::tool("database unavailable"))
    })
    .into_definition();
    let client = ScriptedClient::new().tool_call("lookup", json!({})).text("Sorry, try later.");
    let agent = Agent::builder(client.clone()).tool(failing).build();

    let response = agent.run_once("Look it up").await.unwrap();

    assert_eq!(response.text(), "Sorry, try later.");
    let result = &client.requests()[1].tool_results()[0];
    // Without include_detailed_errors the model sees a generic message.
    assert!(result.is_error());
    assert!(!result.exception.as_deref().unwrap().contains("database"));
}

#[tokio::test]
async fn a_rejected_approval_never_runs_the_tool() {
    let calls = Arc::new(AtomicUsize::new(0));
    let tool = weather(calls.clone())
        .with_approval_mode(ApprovalMode::AlwaysRequire)
        .into_definition();
    let client = ScriptedClient::new()
        .tool_call("get_weather", json!({ "city": "Hull" }))
        .text("Understood, I will not check.");
    let agent = Agent::builder(client.clone()).tool(tool).build();
    let mut session = agent.create_session();

    let first = agent.run(vec![Message::user("Weather in Hull?")], Some(&mut session)).await.unwrap();
    let request = first.user_input_requests()[0].clone();
    assert_eq!(request.function_call.name, "get_weather");

    let answer = Content::FunctionApprovalResponse(request.create_response(false));
    let second = agent
        .run(vec![Message::with_contents(Role::user(), vec![answer])], Some(&mut session))
        .await
        .unwrap();

    assert_eq!(second.text(), "Understood, I will not check.");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    // The model is told the call was rejected.
    let result = &client.requests()[1].tool_results()[0];
    let told = format!("{:?} {:?}", result.result, result.exception);
    assert!(told.contains("rejected"), "{told}");
}

#[tokio::test]
async fn the_call_budget_switches_tools_off_instead_of_failing() {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = ScriptedClient::new()
        .tool_calls(&[
            ("get_weather", json!({ "city": "York" })),
            ("get_weather", json!({ "city": "Leeds" })),
        ])
        .text("Here is what I found.");
    let agent = Agent::builder(client.clone())
        .tool(weather(calls.clone()).into_definition())
        .function_invocation_config(FunctionInvocationConfig {
            max_function_calls: Some(1),
            ..Default::default()
        })
        .build();

    let response = agent.run_once("Check everywhere").await.unwrap();

    assert_eq!(response.text(), "Here is what I found.");
    // The budget is checked after a batch, so both calls in the batch ran.
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    // On the next request tools were switched off, so the model had to answer.
    let last = client.requests().last().cloned().unwrap();
    assert_eq!(last.options.tool_choice.map(|m| m.as_str()), Some("none"));
}

#[tokio::test]
async fn per_run_options_reach_the_client() {
    let client = ScriptedClient::new().text("ok");
    let agent = Agent::builder(client.clone()).instructions("Base.").temperature(0.0).build();

    let options = AgentRunOptions::new().with_chat_options(
        ChatOptions::new().with_temperature(0.9).with_instructions("Extra."),
    );
    agent.run_with_options(vec![Message::user("hi")], None, options).await.unwrap();

    let request = &client.requests()[0];
    assert_eq!(request.options.temperature, Some(0.9));
    // Per-run instructions are appended to the agent's, in the system message.
    assert_eq!(request.texts(), ["Base.\nExtra.", "hi"]);
}

#[tokio::test]
async fn a_session_carries_history_and_state() {
    let client = ScriptedClient::new().text("Hello, Ada.").text("Your name is Ada.");
    let agent = Agent::builder(client.clone()).build();
    let mut session = agent.create_session();
    session.state.insert("tenant", json!("acme"));

    agent.run(vec![Message::user("I am Ada.")], Some(&mut session)).await.unwrap();
    agent.run(vec![Message::user("Who am I?")], Some(&mut session)).await.unwrap();

    assert_eq!(client.requests()[1].texts(), ["I am Ada.", "Hello, Ada.", "Who am I?"]);
    // The session round-trips through JSON with its state.
    let restored = AgentSession::from_dict(&session.to_dict()).unwrap();
    assert_eq!(restored.session_id(), session.session_id());
    assert_eq!(restored.state.get("tenant"), Some(json!("acme")));
}

#[tokio::test]
async fn a_service_error_fails_the_run() {
    let client = ScriptedClient::new().error(Error::service_status(500, "boom", None));
    let agent = Agent::builder(client).build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(err.status(), Some(500));
}
```

Several of these tests record behaviour rather than just exercise it:

- **The response holds the whole exchange.** `response.messages` contains the assistant's call, the tool result and the final answer, not only the last message. `text()` concatenates assistant text, so it ignores the plumbing.
- **Tool failures stay inside the run.** A tool that returns `Err` produces a failed result the model sees ("tool execution failed" unless `include_detailed_errors` is set), and the run carries on. Only `Error::middleware_failure` from function middleware, or an unknown tool with `terminate_on_unknown_calls`, fails the run.
- **A rejected approval is a result, not an error.** The model receives "Error: Tool call invocation was rejected by user." as the call's result (the same text Python sends) and can reply to it.
- **Budgets are checked after a batch.** With `max_function_calls: Some(1)` and a model that asks for two calls in one turn, both run; the next request goes out with `tool_choice` set to `none`, so the model must answer. A run never fails because a budget ran out.
- **Per-run instructions are appended.** `ChatOptions::with_instructions` on a single run adds a line to the agent's instructions in the same system message; it does not replace them.

## Streaming and structured output

```rust
// tests/test_streaming.rs
mod common;

use std::sync::Arc;

use agent_framework::prelude::*;
use common::ScriptedClient;
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// Drain a stream, keeping every update.
async fn collect(mut stream: AgentRunStream) -> Vec<AgentResponseUpdate> {
    let mut updates = Vec::new();
    while let Some(update) = stream.next().await {
        updates.push(update.expect("stream error"));
    }
    updates
}

#[tokio::test]
async fn updates_add_up_to_the_same_response() {
    let agent = Agent::builder(ScriptedClient::new().text("one two three")).build();
    let updates = collect(agent.run_stream_once("Count").await.unwrap()).await;

    assert!(updates.len() >= 3, "expected one update per word, got {}", updates.len());
    let text: String = updates.iter().map(AgentResponseUpdate::text).collect();
    assert_eq!(text, "one two three");

    // Aggregating the updates gives what a non-streaming run would return.
    let response = AgentResponse::from_updates(updates);
    assert_eq!(response.text(), "one two three");
    assert_eq!(response.usage_details.unwrap().output_token_count, Some(5));
}

#[tokio::test]
async fn tools_run_inside_a_streamed_run() {
    let lookup = FunctionTool::new("lookup", "Look up.", json!({ "type": "object" }), |_| async {
        Ok(json!({ "answer": 42 }))
    })
    .into_definition();
    let client = ScriptedClient::new().tool_call("lookup", json!({})).text("The answer is 42.");
    let agent = Agent::builder(client.clone()).tool(lookup).build();

    let updates = collect(agent.run_stream_once("What is the answer?").await.unwrap()).await;

    let text: String = updates.iter().map(AgentResponseUpdate::text).collect();
    assert_eq!(text, "The answer is 42.");
    // The call and its result were streamed too, as content updates.
    let has_result = updates
        .iter()
        .flat_map(|u| &u.contents)
        .any(|c| matches!(c, Content::FunctionResult(_)));
    assert!(has_result);
    assert_eq!(client.requests().len(), 2);
}

#[tokio::test]
async fn a_streamed_session_writes_history_when_the_stream_ends() {
    let client = ScriptedClient::new().text("Noted.").text("You said blue.");
    let agent = Agent::builder(client.clone()).build();

    // run_stream takes the session by value; keep a handle on the history.
    let history = Arc::new(InMemoryHistoryProvider::new());
    let session = AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    let stream = agent
        .run_stream(vec![Message::user("My colour is blue.")], Some(session.clone()), None)
        .await
        .unwrap();
    collect(stream).await;
    assert_eq!(history.list_messages().len(), 2, "user message and reply stored");

    // The clone shares the provider, so the next turn sees the history.
    let mut session = session;
    agent.run(vec![Message::user("What did I say?")], Some(&mut session)).await.unwrap();
    assert_eq!(client.requests()[1].texts(), ["My colour is blue.", "Noted.", "What did I say?"]);
}

#[tokio::test]
async fn dropping_a_stream_early_stores_nothing() {
    let client = ScriptedClient::new().text("a long answer that nobody reads");
    let agent = Agent::builder(client).build();
    let history = Arc::new(InMemoryHistoryProvider::new());
    let session = AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);

    let mut stream = agent.run_stream(vec![Message::user("Tell me")], Some(session), None).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.text(), "a");
    drop(stream);

    assert!(history.list_messages().is_empty());
}

/// The structured reply.
#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Verdict {
    approved: bool,
    /// Why, in one sentence.
    reason: String,
}

fn verdict_format() -> ResponseFormat {
    let schema = serde_json::to_value(schemars::schema_for!(Verdict)).unwrap();
    ResponseFormat::json_schema("Verdict", schema)
}

#[tokio::test]
async fn structured_output_parses_and_the_schema_is_sent() {
    let client = ScriptedClient::new().text(r#"{"approved": true, "reason": "Clear."}"#);
    let agent = Agent::builder(client.clone()).response_format(verdict_format()).build();

    let response = agent.run_once("Approve?").await.unwrap();

    let verdict: Verdict = response.parse_json().unwrap();
    assert_eq!(verdict, Verdict { approved: true, reason: "Clear.".into() });
    assert_eq!(response.value.as_ref().unwrap()["approved"], true);
    // The format reached the client, with the derived schema.
    let sent = client.requests()[0].options.response_format.clone().unwrap();
    let sent = serde_json::to_value(&sent).unwrap();
    assert!(sent.to_string().contains("reason"));
}

#[tokio::test]
async fn invalid_structured_output_is_an_error_you_can_handle() {
    let client = ScriptedClient::new().text("Sure! Here you go: approved.");
    let agent = Agent::builder(client).response_format(verdict_format()).build();

    let response = agent.run_once("Approve?").await.unwrap();

    assert!(response.value.is_none());
    assert!(response.parse_json::<Verdict>().is_err());
}

#[tokio::test]
async fn a_streamed_structured_reply_parses_after_aggregation() {
    let client = ScriptedClient::new().text(r#"{"approved": false, "reason": "Too vague."}"#);
    let agent = Agent::builder(client).response_format(verdict_format()).build();

    let updates = collect(agent.run_stream_once("Approve?").await.unwrap()).await;
    let verdict: Verdict = AgentResponse::from_updates(updates).parse_json().unwrap();
    assert!(!verdict.approved);
}
```

`run_stream` takes the session by value, because the returned stream owns it until it ends. History is written when the stream has been read to the end, through the history provider, so keep an `Arc` to the provider (or a clone of the session, which shares it) to see the result. A stream dropped part-way stores nothing: neither the user's message nor the partial reply reaches history. That is usually what you want for a user who cancels, but a chat UI that shows partial text should not assume the next turn will remember it.

For structured output, `response.value` is filled only when the reply parses as JSON, and `parse_json::<T>()` returns an error you can handle when it does not. Test both paths: a model that ignores the format, or a reply cut off by `max_tokens`, produces the second.

## Testing tools on their own

Most tool logic needs no agent at all. `FunctionTool` implements the `Tool` trait, so `tool.invoke(json!({...}))` runs it with raw JSON exactly as the loop would, and `into_definition().to_openai_spec()` gives the full contract the model sees. Two further techniques are worth having: a golden file for that contract, so a renamed field or a lost doc comment shows up in review, and property-style checks over many generated inputs, because a model will eventually send arguments you did not think of.

```rust
// tests/test_tools.rs
use std::path::PathBuf;

use agent_framework::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize, JsonSchema)]
struct ConvertArgs {
    /// The temperature to convert.
    value: f64,
    /// "c_to_f" or "f_to_c".
    direction: String,
}

fn convert_tool() -> FunctionTool {
    FunctionTool::typed(
        "convert_temperature",
        "Convert between Celsius and Fahrenheit.",
        |args: ConvertArgs| async move {
            let converted = match args.direction.as_str() {
                "c_to_f" => args.value * 9.0 / 5.0 + 32.0,
                "f_to_c" => (args.value - 32.0) * 5.0 / 9.0,
                other => return Err(Error::tool(format!("unknown direction {other:?}"))),
            };
            Ok(json!({ "value": converted }))
        },
    )
}

/// Compare `actual` with a JSON file under tests/golden/. A missing file is
/// written; set UPDATE_GOLDEN=1 to rewrite a changed one on purpose.
fn assert_golden(name: &str, actual: &Value) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(format!("{name}.json"));
    let pretty = serde_json::to_string_pretty(actual).unwrap() + "\n";
    if std::env::var_os("UPDATE_GOLDEN").is_some() || !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, pretty).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap();
    assert_eq!(pretty, expected, "{name} changed; rerun with UPDATE_GOLDEN=1 if intended");
}

#[tokio::test]
async fn a_tool_runs_without_an_agent() {
    let tool = convert_tool();
    let out = tool.invoke(json!({ "value": 100.0, "direction": "c_to_f" })).await.unwrap();
    assert_eq!(out, json!({ "value": 212.0 }));
}

#[tokio::test]
async fn bad_arguments_are_a_tool_error_not_a_panic() {
    let tool = convert_tool();
    let err = tool.invoke(json!({ "value": "hot" })).await.unwrap_err();
    assert!(matches!(err, Error::Tool(_)), "{err:?}");
    assert!(err.to_string().contains("invalid arguments"));
}

#[test]
fn the_schema_the_model_sees_is_pinned() {
    let tool = convert_tool();
    // The full contract: name, description and parameters, as sent to OpenAI.
    assert_golden("convert_temperature", &tool.into_definition().to_openai_spec());
}

#[tokio::test]
async fn invocation_limits_are_counted_across_clones() {
    let tool = convert_tool().max_invocations(2);
    let clone = tool.clone();
    let args = json!({ "value": 0.0, "direction": "c_to_f" });
    tool.invoke(args.clone()).await.unwrap();
    clone.invoke(args.clone()).await.unwrap();
    assert!(tool.invoke(args).await.is_err(), "the third call is refused");
    assert_eq!(tool.invocation_count(), 2);
}

/// A small deterministic generator, so failures reproduce from the seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn float(&mut self, range: f64) -> f64 {
        (self.next() % 2_000_000) as f64 / 1_000_000.0 * range - range
    }
    /// Arbitrary JSON, the way a confused model might send it.
    fn json(&mut self, depth: u32) -> Value {
        match self.next() % if depth == 0 { 4 } else { 6 } {
            0 => Value::Null,
            1 => json!(self.next() % 2 == 0),
            2 => json!(self.float(1e6)),
            3 => json!(["c_to_f", "f_to_c", "", "sideways"][(self.next() % 4) as usize]),
            4 => Value::Array((0..self.next() % 3).map(|_| self.json(depth - 1)).collect()),
            _ => {
                let keys = ["value", "direction", "units"];
                let mut map = serde_json::Map::new();
                for _ in 0..self.next() % 4 {
                    map.insert(keys[(self.next() % 3) as usize].into(), self.json(depth - 1));
                }
                Value::Object(map)
            }
        }
    }
}

#[tokio::test]
async fn round_trips_hold_for_many_inputs() {
    let tool = convert_tool();
    let mut rng = Rng(0x5eed);
    for _ in 0..500 {
        let celsius = rng.float(500.0);
        let f = tool.invoke(json!({ "value": celsius, "direction": "c_to_f" })).await.unwrap();
        let back = tool.invoke(json!({ "value": f["value"], "direction": "f_to_c" })).await.unwrap();
        let back = back["value"].as_f64().unwrap();
        assert!((back - celsius).abs() < 1e-9, "{celsius} came back as {back}");
    }
}

#[tokio::test]
async fn arbitrary_arguments_never_panic() {
    let tool = convert_tool();
    let mut rng = Rng(42);
    for _ in 0..1000 {
        let args = rng.json(3);
        // Ok or a tool error are both fine; a panic fails the test.
        if let Err(err) = tool.invoke(args.clone()).await {
            assert!(matches!(err, Error::Tool(_)), "{args} gave {err:?}");
        }
    }
}
```

The golden helper writes a missing file and fails on a changed one; set `UPDATE_GOLDEN=1` to accept a deliberate change, and commit the files in `tests/golden/`. The same helper suits request bodies and workflow outputs. The [insta](https://docs.rs/insta) crate does the same with review tooling, and [proptest](https://docs.rs/proptest) adds shrinking to the property checks; neither is needed for the tests here, which use only the crates listed above.

The schema in the golden file is the typed tool's derived schema with `$schema` and `title` removed, as the crate sends it. A `FunctionTool::typed` tool rejects arguments that do not deserialise with `Error::Tool("invalid arguments for tool ...")` before your closure runs, and the `arbitrary_arguments_never_panic` test holds the closure to the same standard. `max_invocations` counters are shared by every clone of a tool, so they count across runs and agents, not per conversation.

## Middleware

```rust
// tests/test_middleware.rs
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::middleware::FnMiddleware;
use agent_framework::prelude::*;
use agent_framework::tools::BoxFuture;
use async_trait::async_trait;
use common::ScriptedClient;
use serde_json::json;

/// Agent level: answers "ping" without calling the model.
struct PingShortCircuit;

#[async_trait]
impl Middleware<AgentContext> for PingShortCircuit {
    async fn process(&self, mut ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        if ctx.messages.last().map(Message::text).as_deref() == Some("ping") {
            ctx.result = Some(AgentResponse {
                messages: vec![Message::assistant("pong")],
                ..Default::default()
            });
            return Ok(ctx);
        }
        next.run(ctx).await
    }
}

#[tokio::test]
async fn agent_middleware_can_answer_without_the_model() {
    let client = ScriptedClient::new().text("from the model");
    let agent = Agent::builder(client.clone()).middleware(Arc::new(PingShortCircuit)).build();

    assert_eq!(agent.run_once("ping").await.unwrap().text(), "pong");
    assert!(client.requests().is_empty());
    assert_eq!(agent.run_once("hello").await.unwrap().text(), "from the model");
}

#[tokio::test]
async fn chat_middleware_runs_once_per_run_around_the_tool_loop() {
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = seen.clone();
    let count_calls = FnMiddleware::new(move |ctx: ChatContext, next: Next<ChatContext>| {
        let counter = counter.clone();
        async move {
            counter.fetch_add(1, Ordering::SeqCst);
            next.run(ctx).await
        }
    });
    let tool = FunctionTool::new("noop", "Does nothing.", json!({ "type": "object" }), |_| async {
        Ok(json!("done"))
    })
    .into_definition();
    let client = ScriptedClient::new().tool_call("noop", json!({})).text("finished");
    let agent = Agent::builder(client.clone())
        .tool(tool)
        .chat_middleware(Arc::new(count_calls))
        .build();

    agent.run_once("go").await.unwrap();

    // Two model requests, one pass through chat middleware (Python would make two).
    assert_eq!(client.requests().len(), 2);
    assert_eq!(seen.load(Ordering::SeqCst), 1);
}

/// Function level: redacts an argument before the tool sees it.
struct RedactEmail;

#[async_trait]
impl Middleware<FunctionInvocationContext> for RedactEmail {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        if ctx.arguments.get("email").is_some() {
            ctx.arguments["email"] = json!("[redacted]");
        }
        next.run(ctx).await
    }
}

#[tokio::test]
async fn function_middleware_changes_arguments() {
    let pipeline = MiddlewarePipeline::new(vec![Arc::new(RedactEmail) as Arc<dyn Middleware<_>>]);
    // The terminal closure stands in for the tool: it echoes what it received.
    let terminal = Box::new(|mut ctx: FunctionInvocationContext| {
        Box::pin(async move {
            ctx.result = Some(ctx.arguments.clone());
            Ok(ctx)
        }) as BoxFuture<Result<FunctionInvocationContext>>
    });
    let ctx = FunctionInvocationContext::new("send", json!({ "email": "ada@example.com", "body": "hi" }));

    let ctx = pipeline.execute(ctx, terminal).await.unwrap();

    assert_eq!(ctx.result, Some(json!({ "email": "[redacted]", "body": "hi" })));
}

#[tokio::test]
async fn middleware_failure_stops_the_whole_run() {
    let stop = FnMiddleware::new(|_ctx: FunctionInvocationContext, _next: Next<FunctionInvocationContext>| async {
        Err(Error::middleware_failure("policy violation"))
    });
    let tool = FunctionTool::new("pay", "Pay.", json!({ "type": "object" }), |_| async { Ok(json!("paid")) })
        .into_definition();
    let client = ScriptedClient::new().tool_call("pay", json!({})).text("never reached");
    let agent = Agent::builder(client.clone()).tool(tool).function_middleware(Arc::new(stop)).build();

    let err = agent.run_once("Pay the invoice").await.unwrap_err();

    assert!(err.is_middleware_failure(), "{err:?}");
    assert_eq!(client.requests().len(), 1, "the model was not called again");
}

#[tokio::test]
async fn middleware_order_is_first_added_outermost() {
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    let tag = |name: &'static str| {
        let order = order.clone();
        Arc::new(FnMiddleware::new(move |ctx: AgentContext, next: Next<AgentContext>| {
            let order = order.clone();
            async move {
                order.lock().unwrap().push(format!("{name} before"));
                let ctx = next.run(ctx).await?;
                order.lock().unwrap().push(format!("{name} after"));
                Ok(ctx)
            }
        }))
    };
    let agent = Agent::builder(ScriptedClient::new().text("ok"))
        .middleware(tag("outer"))
        .middleware(tag("inner"))
        .build();

    agent.run_once("hi").await.unwrap();

    assert_eq!(*order.lock().unwrap(), ["outer before", "inner before", "inner after", "outer after"]);
}
```

Middleware added first runs outermost, at every level. Chat middleware wraps the whole tool loop, so it runs once per run however many model requests the loop makes; Python's chat middleware runs once per request. To observe every request, wrap the client instead (the next section's wrappers are examples). Function middleware can be tested with no agent through `MiddlewarePipeline::execute` and a terminal closure standing in for the tool.

## Retries, timeouts, cancellation and concurrency

The provider clients have no timeout setting (see [timeouts](#timeouts) below), and the crate has no client-level concurrency limit, so both are client wrappers you write once. They are defined in this test file so the tests can exercise them; in an application, move them into your crate.

```rust
// tests/test_resilience.rs
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_framework::prelude::*;
use async_trait::async_trait;
use common::ScriptedClient;
use tokio::sync::Semaphore;

/// Fails a model call that takes longer than `limit`. The error text contains
/// "timed out", which the default retry rule treats as transient.
#[derive(Clone)]
pub struct TimeoutChatClient<C> {
    inner: C,
    limit: Duration,
}

impl<C> TimeoutChatClient<C> {
    pub fn new(inner: C, limit: Duration) -> Self {
        Self { inner, limit }
    }
}

#[async_trait]
impl<C: ChatClient> ChatClient for TimeoutChatClient<C> {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        tokio::time::timeout(self.limit, self.inner.get_response(messages, options))
            .await
            .map_err(|_| Error::service(format!("model call timed out after {:?}", self.limit)))?
    }

    async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatStream> {
        // Bounds the time to open the stream, not the whole stream.
        tokio::time::timeout(self.limit, self.inner.get_streaming_response(messages, options))
            .await
            .map_err(|_| Error::service(format!("stream did not start within {:?}", self.limit)))?
    }

    fn model(&self) -> Option<&str> {
        self.inner.model()
    }
}

/// Lets at most `permits` model calls be in flight at once, across every
/// agent that shares this client.
#[derive(Clone)]
pub struct ConcurrencyLimitedClient<C> {
    inner: C,
    permits: Arc<Semaphore>,
}

impl<C> ConcurrencyLimitedClient<C> {
    pub fn new(inner: C, permits: usize) -> Self {
        Self { inner, permits: Arc::new(Semaphore::new(permits)) }
    }
}

#[async_trait]
impl<C: ChatClient> ChatClient for ConcurrencyLimitedClient<C> {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let _permit = self.permits.acquire().await.map_err(Error::other)?;
        self.inner.get_response(messages, options).await
    }

    async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatStream> {
        // The permit covers opening the stream only; hold it in the stream to cover more.
        let _permit = self.permits.acquire().await.map_err(Error::other)?;
        self.inner.get_streaming_response(messages, options).await
    }
}

fn fast_retries() -> RetryPolicy {
    RetryPolicy::with_max_retries(2).initial_delay(Duration::from_millis(1)).jitter(0.0)
}

#[tokio::test]
async fn retry_after_is_honoured_up_to_max_delay() {
    let flaky = ScriptedClient::new()
        .error(Error::service_status(429, "slow down", Some(0.2)))
        .text("ok");
    let client = RetryingChatClient::new(flaky.clone())
        .with_policy(fast_retries().max_delay(Duration::from_millis(50)));

    let started = Instant::now();
    let reply = client.get_response(vec![Message::user("hi")], ChatOptions::new()).await.unwrap();

    assert_eq!(reply.text(), "ok");
    let waited = started.elapsed();
    // The server asked for 200 ms; max_delay capped it at 50 ms.
    assert!(waited >= Duration::from_millis(50) && waited < Duration::from_millis(200), "{waited:?}");
}

#[tokio::test]
async fn the_default_rule_retries_a_timeout() {
    // Every reply takes 30 ms and each attempt is given 10 ms.
    let slow = ScriptedClient::new().text("late").text("late again").with_delay(Duration::from_millis(30));
    let client = RetryingChatClient::new(TimeoutChatClient::new(slow.clone(), Duration::from_millis(10)))
        .with_policy(RetryPolicy::with_max_retries(1).initial_delay(Duration::from_millis(1)).jitter(0.0));

    let err = client.get_response(vec![Message::user("hi")], ChatOptions::new()).await.unwrap_err();

    assert!(err.to_string().contains("timed out"), "{err}");
    assert_eq!(slow.requests().len(), 2, "the first attempt and one retry");
}

#[tokio::test]
async fn cancelling_a_run_leaves_the_session_unchanged() {
    let slow = ScriptedClient::new().text("never seen").with_delay(Duration::from_secs(5));
    let agent = Agent::builder(slow).build();
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session = AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);

    let outcome = tokio::time::timeout(
        Duration::from_millis(20),
        agent.run(vec![Message::user("hello")], Some(&mut session)),
    )
    .await;

    assert!(outcome.is_err(), "the run was cancelled");
    // after_run never ran, so neither the input nor a reply was stored.
    assert!(history.list_messages().is_empty());
}

#[tokio::test]
async fn shared_agents_respect_a_global_limit() {
    let in_flight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));

    /// Records how many calls overlap.
    #[derive(Clone)]
    struct Gauge {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl ChatClient for Gauge {
        async fn get_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(ChatResponse::from_text("done"))
        }
        async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    let client = ConcurrencyLimitedClient::new(Gauge { in_flight, peak: peak.clone() }, 3);
    let agent = Agent::builder(client).build();
    let runs = (0..12).map(|i| {
        let agent = agent.clone();
        async move { agent.run_once(format!("job {i}")).await }
    });
    let results = futures::future::join_all(runs).await;

    assert!(results.iter().all(|r| r.is_ok()));
    assert_eq!(peak.load(Ordering::SeqCst), 3);
}
```

What these tests establish:

- **`Retry-After` wins over the computed backoff, but is capped by `max_delay`.** A server asking for 200 ms with `max_delay` at 50 ms waits 50 ms. Server-advised delays are not jittered.
- **A timeout is retryable by default** if its error says so. The default rule retries `Error::Service` messages containing "timed out", "timeout", "connection", "request failed" or "stream error", and statuses 408, 429 and 5xx. Put the timeout inside the retry (`RetryingChatClient::new(TimeoutChatClient::new(...))`) so each attempt gets its own limit.
- **Cancelling a run is dropping its future**, and a dropped run stores nothing in history: context providers' `after_run` never runs. Tool calls already in flight are dropped with it; a side effect the tool had already made stays made.
- **An `Agent` is cheap to clone and safe to share.** Clones share the client, so a limit in the client is a limit across every clone. The semaphore above holds a permit only while a request is being sent or a stream opened; a streamed reply that is still arriving does not hold one.

`RetryPolicy` has no test clock in this setup, so keep delays to a millisecond and `jitter(0.0)` in tests. The crate's own tests use tokio's `test-util` feature with `start_paused` for exact timing; add that feature to the dev-dependency if you want the same.

## Workflows, checkpoints and orchestrations

Function executors need no model, agent executors use the scripted clients, and the orchestrations are ordinary workflows, so all of it runs in-process.

```rust
// tests/test_workflows.rs
mod common;

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{
    handoff_tool_spec, ApprovalRequest, FunctionExecutor, HandoffUserInputRequest, RequestResponse,
};
use common::{last_user_text, FnClient, ScriptedClient};
use futures::StreamExt;
use serde_json::json;

/// draft -> review (asks a person) -> publish.
fn review_pipeline(storage: Arc<dyn CheckpointStorage>) -> Workflow {
    let draft = FunctionExecutor::new("draft", |m, ctx| async move {
        ctx.send_message(json!({ "draft": format!("Draft about {}", m.as_str().unwrap_or("?")) })).await?;
        Ok(())
    });
    let review = FunctionExecutor::new("review", |m, ctx| async move {
        match RequestResponse::from_message(&m) {
            Some(answer) => ctx.send_message(json!({ "draft": answer.original_request["draft"], "verdict": answer.data })).await?,
            None => ctx.request_info(json!({ "draft": m["draft"], "question": "Publish?" })).await?,
        }
        Ok(())
    });
    let publish = FunctionExecutor::new("publish", |m, ctx| async move {
        ctx.yield_output(m).await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .name("review")
        .add_executor(Arc::new(draft))
        .add_executor(Arc::new(review))
        .add_executor(Arc::new(publish))
        .set_start("draft")
        .add_chain(["draft", "review", "publish"].map(String::from))
        .with_checkpointing(storage)
        .build()
        .unwrap()
}

#[tokio::test]
async fn a_pending_request_survives_a_restart() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());

    let run = review_pipeline(storage.clone()).run("lifetimes").await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    drop(run); // the process "exits" here

    // A new process builds the same graph. Its workflow id is new, so look
    // checkpoints up by name rather than by workflow id.
    let restarted = review_pipeline(storage.clone());
    let mut checkpoints: Vec<_> = storage
        .list(None)
        .await
        .unwrap()
        .into_iter()
        .filter(|cp| cp.workflow_name.as_deref() == Some("review") && !cp.pending_requests.is_empty())
        .collect();
    checkpoints.sort_by_key(|cp| cp.timestamp_millis);
    let latest = checkpoints.last().expect("a checkpoint with the pending request");

    let mut resumed = restarted.run_from_checkpoint(&latest.checkpoint_id, storage).await.unwrap();
    assert_eq!(resumed.state(), WorkflowRunState::IdleWithPendingRequests);
    let request = resumed.pending_requests().remove(0);
    assert_eq!(request.request_data["question"], "Publish?");

    resumed.send_response(request.request_id, json!("approve")).await.unwrap();
    assert_eq!(
        resumed.last_output(),
        Some(json!({ "draft": "Draft about lifetimes", "verdict": "approve" }))
    );
}

#[tokio::test]
async fn an_answer_to_an_unknown_request_is_an_error() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let mut run = review_pipeline(storage).run("x").await.unwrap();
    assert!(run.send_response("no-such-id", json!("approve")).await.is_err());
}

#[tokio::test]
async fn the_event_stream_shows_each_step() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let workflow = review_pipeline(storage);

    let mut stream = workflow.run_stream("events");
    let mut invoked = Vec::new();
    let mut requests = 0;
    while let Some(event) = stream.next().await {
        match event {
            WorkflowEvent::ExecutorInvoked { executor_id } => invoked.push(executor_id),
            WorkflowEvent::RequestInfo { .. } => requests += 1,
            _ => {}
        }
    }
    assert_eq!(invoked, ["draft", "review"]);
    assert_eq!(requests, 1);
    let run = stream.into_run().await.unwrap();
    assert_eq!(run.pending_requests().len(), 1);
}

fn agent(name: &str, client: impl ChatClient + 'static) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(client).name(name).instructions(format!("You are {name}.")).build())
}

fn conversation(run: &WorkflowRun) -> Vec<Message> {
    serde_json::from_value(run.last_output().unwrap()).unwrap()
}

fn speakers(messages: &[Message]) -> Vec<String> {
    messages.iter().filter_map(|m| m.author_name.clone()).collect()
}

#[tokio::test]
async fn concurrent_replies_are_gathered_in_participant_order() {
    // One client for both agents: reply with the agent's own instructions,
    // so the order of calls does not matter.
    let client = FnClient::new(|messages, _| Ok(ChatResponse::from_text(messages[0].text())));
    let workflow = ConcurrentBuilder::new()
        .participants(vec![agent("optimist", client.clone()), agent("sceptic", client.clone())])
        .build()
        .unwrap();

    let run = workflow.run("Rewrite it in Rust?").await.unwrap();

    let texts: Vec<String> = conversation(&run).iter().map(Message::text).collect();
    assert_eq!(texts, ["Rewrite it in Rust?", "You are optimist.", "You are sceptic."]);
    assert_eq!(client.requests().len(), 2);
}

#[tokio::test]
async fn sequential_with_request_info_pauses_after_each_agent() {
    let workflow = SequentialBuilder::new()
        .participants(vec![
            agent("writer", FnClient::always("A first draft.")),
            agent("editor", FnClient::always("A better draft.")),
        ])
        .with_request_info()
        .build()
        .unwrap();

    let mut run = workflow.run("Write about traits").await.unwrap();
    let mut reviewed = Vec::new();
    while let Some(request) = run.pending_requests().into_iter().next() {
        let pending: ApprovalRequest = serde_json::from_value(request.request_data).unwrap();
        reviewed.push(pending.reply.last().unwrap().text());
        // An empty answer approves; any other text is sent back as feedback.
        run.send_response(request.request_id, json!("")).await.unwrap();
    }

    assert_eq!(reviewed, ["A first draft.", "A better draft."]);
    assert_eq!(conversation(&run).last().unwrap().text(), "A better draft.");
}

#[tokio::test]
async fn handoff_routes_to_the_specialist() {
    let triage_client = ScriptedClient::new().tool_call("handoff_to_billing", json!({})).text("unused");
    let triage: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(triage_client.clone())
            .name("triage")
            .tool(handoff_tool_spec("billing", None))
            .build(),
    );
    let workflow = HandoffBuilder::new()
        .participant("triage", triage)
        .participant("billing", agent("billing", ScriptedClient::new().text("Refund issued.")))
        .initial_agent("triage")
        .autonomous()
        .build()
        .unwrap();

    let run = workflow.run("I was charged twice.").await.unwrap();

    let messages = conversation(&run);
    assert_eq!(messages.last().unwrap().text(), "Refund issued.");
    assert_eq!(speakers(&messages).last().map(String::as_str), Some("billing"));
    assert_eq!(triage_client.requests()[0].tool_names(), ["handoff_to_billing"]);
}

#[tokio::test]
async fn human_in_loop_handoff_asks_the_user_and_returns_to_the_initial_agent() {
    let triage: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(
            ScriptedClient::new()
                .tool_call("handoff_to_billing", json!({}))
                .text("Glad that is sorted."),
        )
        .name("triage")
        .tool(handoff_tool_spec("billing", None))
        .build(),
    );
    let workflow = HandoffBuilder::new()
        .participant("triage", triage)
        .participant("billing", agent("billing", ScriptedClient::new().text("Which invoice?")))
        .initial_agent("triage")
        .build()
        .unwrap();

    let mut run = workflow.run("I was charged twice.").await.unwrap();
    let request = run.pending_requests().remove(0);
    let ask: HandoffUserInputRequest = serde_json::from_value(request.request_data).unwrap();
    assert_eq!(ask.awaiting_agent, "billing");
    assert_eq!(ask.conversation.last().unwrap().text(), "Which invoice?");

    run.send_response(request.request_id, json!("INV-7, thanks")).await.unwrap();

    // The reply went to triage (the initial agent), which answered.
    let last = run.pending_requests().remove(0);
    let ask: HandoffUserInputRequest = serde_json::from_value(last.request_data).unwrap();
    assert_eq!(ask.awaiting_agent, "triage");
    assert_eq!(ask.conversation.last().unwrap().text(), "Glad that is sorted.");
}

#[tokio::test]
async fn a_scripted_group_chat_manager_sets_the_speaking_order() {
    let echo = FnClient::new(|messages, _| Ok(ChatResponse::from_text(format!("re: {}", last_user_text(messages)))));
    let workflow = GroupChatBuilder::new()
        .participant("writer", agent("writer", echo.clone()))
        .participant("critic", agent("critic", echo.clone()))
        .manager_fn(|state| match state.round_index {
            0 | 2 => GroupChatDirective::speak("writer"),
            1 => GroupChatDirective::speak("critic"),
            _ => GroupChatDirective::finish(),
        })
        .build()
        .unwrap();

    let run = workflow.run("A tagline").await.unwrap();

    let messages = conversation(&run);
    // finish() appends a closing message from the manager ("custom_manager"
    // for manager_fn unless manager_name sets another).
    assert_eq!(speakers(&messages), ["writer", "critic", "writer", "custom_manager"]);
    assert_eq!(messages.last().unwrap().text(), "Conversation completed.");
}
```

Things these tests pin down:

- **Workflow ids are new on every build.** A restarted process builds a workflow with a different id, so `storage.list(Some(workflow.id()))` finds nothing from before the restart. Find checkpoints by `workflow_name`, by keeping a separate storage location per case (as the [review recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#a-human-approved-review-workflow-with-checkpoints) does), or by recording the checkpoint id yourself.
- **Pending requests survive a checkpoint.** `run_from_checkpoint` comes back in `IdleWithPendingRequests` with the same request ids, so `send_response` works as if the process had never stopped.
- **A failed checkpoint save is only a warning.** `with_checkpointing` logs a storage error and the run continues, so a test of durability should assert that the checkpoint exists, as the first test does.
- **Concurrent output keeps participant order**, whatever order the agents finished in, after the original user message.
- **`with_request_info()` pauses after every participant** with an `ApprovalRequest` (`conversation`, `reply`). An empty string, `null` or empty array approves; anything else is sent back to that agent as feedback.
- **Handoff in the default human-in-the-loop mode returns the user's reply to the initial agent**, not to the specialist who asked. `HandoffUserInputRequest::awaiting_agent` names who was waiting.
- **A group chat's `finish()` appends a closing message** from the manager ("Conversation completed.", authored `custom_manager` for `manager_fn` unless `manager_name` sets another), so the conversation is one longer than the participants' turns.

Workflow-level detail is on the [workflows](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/) and [orchestrations](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/) pages.

## MCP with an in-process server

`McpClient` is written against the public `McpTransport` trait, so a transport that answers JSON-RPC methods itself is a complete MCP server for testing: no child process, no port. The crate's tool wrappers (`McpStdioTool` and the others) build their transports internally and cannot take a custom one, so the test adapts the client's tools for an agent with a small `Tool` implementation, the same thing the wrappers do inside.

```rust
// tests/test_mcp.rs
mod common;

use std::sync::{Arc, Mutex};

use agent_framework::mcp::{McpClient, McpTransport};
use agent_framework::prelude::*;
use async_trait::async_trait;
use common::ScriptedClient;
use serde_json::{json, Value};

/// An MCP server that lives in the test process: the transport answers each
/// JSON-RPC method directly instead of sending it anywhere.
#[derive(Default)]
struct InProcessServer {
    calls: Mutex<Vec<(String, Value)>>,
}

#[async_trait]
impl McpTransport for InProcessServer {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.calls.lock().unwrap().push((method.to_string(), params.clone()));
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": "2025-06-18",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "in-process", "version": "1.0.0" }
            })),
            "tools/list" => Ok(json!({ "tools": [{
                "name": "stock_level",
                "description": "Units in stock for a SKU.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "sku": { "type": "string" } },
                    "required": ["sku"]
                }
            }]})),
            "tools/call" => match params["arguments"]["sku"].as_str() {
                Some("A-1") => Ok(json!({ "content": [{ "type": "text", "text": "{\"sku\":\"A-1\",\"units\":7}" }] })),
                _ => Ok(json!({ "content": [{ "type": "text", "text": "unknown SKU" }], "isError": true })),
            },
            other => Err(Error::service(format!("method not found: {other}"))),
        }
    }

    async fn notify(&self, _method: &str, _params: Value) -> Result<()> {
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// Exposes one MCP tool to an agent, the way the crate's own wrappers do.
struct McpBackedTool {
    client: Arc<McpClient>,
    name: String,
    description: String,
    schema: Value,
}

#[async_trait]
impl Tool for McpBackedTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }
    async fn invoke(&self, arguments: Value) -> Result<Value> {
        self.client.call_tool_value(&self.name, arguments).await
    }
}

async fn connect() -> (Arc<InProcessServer>, Arc<McpClient>) {
    let server = Arc::new(InProcessServer::default());
    let client = Arc::new(McpClient::new(server.clone()));
    client.initialize("tests", "0.1.0").await.unwrap();
    (server, client)
}

#[tokio::test]
async fn the_client_lists_and_calls_tools() {
    let (server, client) = connect().await;

    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools[0].name, "stock_level");

    // A text block holding JSON comes back parsed.
    let value = client.call_tool_value("stock_level", json!({ "sku": "A-1" })).await.unwrap();
    assert_eq!(value["units"], 7);

    // isError: true becomes Error::Tool with the server's text.
    let err = client.call_tool_value("stock_level", json!({ "sku": "Z-9" })).await.unwrap_err();
    assert!(err.to_string().contains("unknown SKU"), "{err}");

    let methods: Vec<String> = server.calls.lock().unwrap().iter().map(|(m, _)| m.clone()).collect();
    assert_eq!(methods, ["initialize", "tools/list", "tools/call", "tools/call"]);
}

#[tokio::test]
async fn an_agent_uses_the_mcp_tool() {
    let (_server, client) = connect().await;
    let definitions: Vec<ToolDefinition> = client
        .list_tools()
        .await
        .unwrap()
        .into_iter()
        .map(|d| {
            ToolDefinition::from_tool(Arc::new(McpBackedTool {
                client: client.clone(),
                name: d.name.clone(),
                description: d.description.unwrap_or_default(),
                schema: d.input_schema,
            }))
        })
        .collect();

    let model = ScriptedClient::new()
        .tool_call("stock_level", json!({ "sku": "A-1" }))
        .text("There are 7 in stock.");
    let agent = Agent::builder(model.clone()).tools(definitions).build();

    let answer = agent.run_once("How many A-1?").await.unwrap();

    assert_eq!(answer.text(), "There are 7 in stock.");
    assert_eq!(model.requests()[1].tool_results()[0].result.as_ref().unwrap()["units"], 7);
}
```

`call_tool_value` maps the result the way the agent integration does: a single text block holding JSON is parsed, other text stays a string, and `isError: true` becomes `Error::Tool` with the server's text. To test against a real server binary over stdio, see the [MCP recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#an-agent-backed-by-an-mcp-server), which is its own server; the [MCP page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/) covers the transports.

## HTTP surfaces

The hosting routers are plain axum routers, so a test can serve one on port 0 and talk to it with the crate's own OpenAI client, which exercises the JSON and SSE paths end to end.

```rust
// tests/test_hosting.rs
mod common;

use std::net::SocketAddr;

use agent_framework::hosting::openai_compat::OpenAiRouter;
use agent_framework::hosting::HostingSecurity;
use agent_framework::prelude::*;
use common::FnClient;
use futures::StreamExt;

/// Serve a router on a free local port for the length of a test.
async fn serve(app: axum::Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

fn app() -> axum::Router {
    let model = FnClient::new(|messages, _| Ok(ChatResponse::from_text(format!("echo: {}", common::last_user_text(messages)))));
    let agent = Agent::builder(model).name("echo").build();
    HostingSecurity::new()
        .with_bearer_token("test-token")
        .apply(OpenAiRouter::for_agent("echo", agent).into_router())
}

#[tokio::test]
async fn the_agent_answers_over_http() {
    let addr = serve(app()).await;
    let client = OpenAIChatCompletionClient::new("test-token", "echo").with_base_url(format!("http://{addr}/v1"));

    let reply = client.get_response(vec![Message::user("hello")], ChatOptions::new()).await.unwrap();
    assert_eq!(reply.text(), "echo: hello");

    // The same route streams as server-sent events.
    let mut stream = client.get_streaming_response(vec![Message::user("streamed")], ChatOptions::new()).await.unwrap();
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text_content());
    }
    assert_eq!(text, "echo: streamed");
}

#[tokio::test]
async fn a_wrong_token_is_rejected() {
    let addr = serve(app()).await;
    let client = OpenAIChatCompletionClient::new("wrong", "echo").with_base_url(format!("http://{addr}/v1"));

    let err = client.get_response(vec![Message::user("hello")], ChatOptions::new()).await.unwrap_err();
    assert!(matches!(err, Error::ServiceInvalidAuth { .. }), "{err:?}");
}
```

A rejected bearer token arrives at the OpenAI client as `Error::ServiceInvalidAuth` (HTTP 401), which `RetryingChatClient` never retries. If your test environment sets `HTTP_PROXY` for plain HTTP, add `127.0.0.1` to `NO_PROXY`: reqwest honours both.

## Live tests

A script checks wiring: which tools exist, how arguments decode, how approvals, history and errors flow, and where a workflow routes. It cannot tell you whether a real model chooses the tool, follows the instructions or fills the schema. Keep a few tests against a real deployment for that, marked `#[ignore]` so `cargo test` skips them, and run them on purpose with `cargo test --test test_live -- --ignored`.

```rust
// tests/test_live.rs
//! Tests against a real model. Ignored by default: run them with
//! `cargo test --test test_live -- --ignored` and OPENAI_API_KEY set.
use agent_framework::prelude::*;
use serde::Deserialize;
use serde_json::json;

/// A real client, or None when no key is configured.
fn live_client() -> Option<OpenAIChatCompletionClient> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%").ok();
    if client.is_none() {
        eprintln!("OPENAI_API_KEY not set; skipping");
    }
    client
}

#[tokio::test]
#[ignore = "calls a real model"]
async fn live_the_model_chooses_the_tool() {
    let Some(client) = live_client() else { return };
    let tool = FunctionTool::new(
        "stock_level",
        "Units in stock for a SKU.",
        json!({ "type": "object", "properties": { "sku": { "type": "string" } }, "required": ["sku"] }),
        |args| async move { Ok(json!({ "sku": args["sku"], "units": 7 })) },
    )
    .into_definition();
    let agent = Agent::builder(client)
        .instructions("Use the tools for anything about stock.")
        .tool(tool)
        .build();

    let response = agent.run_once("How many units of A-1 are in stock?").await.unwrap();

    let called = response
        .messages
        .iter()
        .flat_map(|m| m.function_calls())
        .any(|c| c.name == "stock_level");
    assert!(called, "the model answered without the tool: {}", response.text());
    assert!(response.text().contains('7'));
}

#[tokio::test]
#[ignore = "calls a real model"]
async fn live_structured_output_matches_the_schema() {
    #[derive(Deserialize)]
    struct Capital {
        country: String,
        capital: String,
    }
    let Some(client) = live_client() else { return };
    let agent = Agent::builder(client).instructions("Answer with JSON only.").build();
    let schema = json!({
        "type": "object",
        "properties": { "country": { "type": "string" }, "capital": { "type": "string" } },
        "required": ["country", "capital"],
        "additionalProperties": false
    });
    let options = AgentRunOptions::new()
        .with_chat_options(ChatOptions::new().with_response_format(ResponseFormat::json_schema("Capital", schema)));

    let response = agent
        .run_with_options(vec![Message::user("The capital of France?")], None, options)
        .await
        .unwrap();

    let parsed: Capital = response.parse_json().unwrap();
    assert_eq!(parsed.country, "France");
    assert_eq!(parsed.capital, "Paris");
    assert_eq!(response.finish_reason.as_ref().map(|f| f.as_str()), Some("stop"));
}
```

Assert on behaviour that should hold for any competent model (the tool was called, the JSON parses, the finish reason is `stop`) rather than exact wording. There is no evaluation framework in the Rust port comparable to Python's `agent_framework` evaluation helpers; the [evaluation recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#a-regression-suite-with-a-judge-model) is a small one built from the pieces above. The [evaluation page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_evaluation/) covers rule checks, the `Judge`, datasets and CI gates in full.

## Running the suite in CI

The scripted tests need no secrets and run on every push; the live tests run on a schedule, or by hand, with the key in the repository's secrets. Building the crate with many features is the slow part, so cache `target/` and the Cargo registry between runs.

```yaml
# .github/workflows/test.yml
name: test
on:
  push:
  pull_request:
  schedule:
    - cron: "17 3 * * 1"   # live tests weekly
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      # Build and test exactly the features you ship.
      - run: cargo test --locked --features mcp,hosting
      - run: cargo clippy --locked --features mcp,hosting --all-targets -- -D warnings

  msrv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@1.88   # the crate's minimum supported Rust
      - uses: Swatinem/rust-cache@v2
      - run: cargo check --locked --features mcp,hosting

  live:
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --locked --test test_live -- --ignored
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

This workflow was not run as part of checking this page; the commands in it were. Two points carry over to any CI system. Test with the same feature set you deploy, since features change which code compiles (an `experimental-*` feature especially). And commit `Cargo.lock` for an application and build with `--locked`, so CI and production resolve the same versions of the crate's many dependencies.

## Production

The rest of this page is about running agents unattended. The [recipes](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/) put most of it together in complete programs: the [HTTP service](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#an-http-service-openai-compatible-with-sse) for a hosted server and the [batch job](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#a-scheduled-batch-job-with-retries-and-telemetry) for scheduled work.

### Versions and features

The crate is pre-1.0, and its changelog follows the convention that a minor release may break the API (0.13.0 had two small breaks, 0.11.0 four). Cargo's default requirement, `"%%version=agent-framework-rs%%"`, already accepts only patch releases of the current minor; read the changelog before moving to the next. Anything behind an `experimental-*` feature can change in any release, and `experimental` is never part of `full`. The minimum supported Rust version is 1.88.

Enable only the features you use. Each provider and integration brings its own dependencies, and `full` builds all of them. `openai` is the only default; set `default-features = false` if you do not use it.

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", default-features = false, features = ["anthropic", "hosting"] }
```

### Timeouts

The provider clients build a plain `reqwest::Client` with no timeout and offer no way to set one, so a request to an endpoint that accepts the connection and never answers waits indefinitely. The function-invocation loop's `max_duration_seconds` does not help here: it is checked between tool batches, not during a model call. Set limits at each layer yourself:

| Layer | Setting | Default |
| --- | --- | --- |
| One model call | A wrapper such as `TimeoutChatClient` [above](#retries-timeouts-cancellation-and-concurrency) | none |
| One HTTP client (Azure OpenAI Responses only) | `AzureOpenAIResponsesClient::with_http_client(reqwest_client)` | none |
| The tool loop | `FunctionInvocationConfig::max_duration_seconds`, checked after each batch | none |
| One tool | Your own `tokio::time::timeout` inside the tool body | none |
| A whole run | `tokio::time::timeout(limit, agent.run(...))` | none |
| An MCP request | `McpStdioTool::request_timeout`, `McpWebsocketTool::request_timeout`, `McpStreamableHttpTool::timeout` | none |
| A shell command | `LocalShellTool` timeout (kills the process group) | see the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/) |

A timeout wrapper around a streaming call bounds the time to open the stream, not the stream's length; to bound a whole streamed reply, put the timeout around the loop that reads it. The official OpenAI SDKs that the Python and .NET clients build on apply default timeouts, so code ported from them loses that protection without any sign.

### Retries

Wrap each provider client in `RetryingChatClient`. It retries throttling, server errors and transport failures with exponential backoff and jitter (defaults: three retries, 500 ms, doubling, capped at 30 s, jitter 0.3), honours `Retry-After`, and never retries authentication failures, bad requests or content-filter refusals. For streaming it retries only while opening the stream and before the first update; after that, errors reach the caller, because the caller may already have shown partial text.

Order the wrappers deliberately. `RetryingChatClient::new(ObservableChatClient::new(TimeoutChatClient::new(provider, limit), "openai"))` gives each attempt its own time limit and its own `chat` span, so failed attempts are visible in traces; with the retry inside the observable client you get one span per logical call instead. A custom `ChatClient` should return the classified errors (`Error::service_status`, `Error::service_invalid_auth` and so on) so the default rule can tell transient failures from permanent ones. For failing over between providers, see the [fallback recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#a-multi-provider-fallback-client).

### Cancellation

There is no cancellation token. A run, a stream or a workflow run is cancelled by dropping its future: `tokio::time::timeout`, `tokio::select!` against a shutdown signal, or aborting the task that owns it. The consequences, as the tests above show:

- In-flight model requests and tool futures are dropped. A tool that spawned its own task, started a process or wrote to a database has done so; make such tools idempotent or record their effects.
- Nothing from the cancelled run reaches history, so the conversation is as it was before the run. The user's message is lost too; resend it if the turn should be retried.
- A workflow run has no cancellation of its own (Python's has). Dropping it loses work since the last checkpoint; with checkpointing on, resume from the latest.
- With `allow_concurrent_invocation` on (the default), a middleware failure in one call of a batch cancels the batch's other calls that are still running.

### Concurrency

`Agent` is `Clone`, `Send` and `Sync`, and a clone shares the client, tools, middleware and context providers. Serve many users from one agent, with one session per conversation; do not share one `AgentSession` between simultaneous runs, because two runs would interleave their history writes. The [HTTP service recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#an-http-service-openai-compatible-with-sse) keeps a lock per session for this.

Bound the work in two places. A semaphore in the client (`ConcurrencyLimitedClient` above) caps requests to a provider across all agents, which is what a provider's rate limit counts. `buffer_unordered(n)` or a `JoinSet` with a semaphore caps how many runs are in progress, which bounds memory and open sessions; the [extraction recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#structured-extraction-over-many-documents) does this. Within one run, the tool loop runs a turn's calls concurrently; set `allow_concurrent_invocation: false` when tools share a resource that cannot take overlapping calls.

### Budgets

For unattended agents set all three loop bounds in `FunctionInvocationConfig`: `max_iterations` (model round trips, 40 by default), `max_function_calls` and `max_duration_seconds` (both unbounded by default). The last two are cumulative across approval round trips, so a paused run cannot reset its budget, and running out ends tool use rather than failing the run. Add `max_invocations` to individual expensive tools, and check `finish_reason`: a `length` or `content_filter` finish produces text that reads like a complete answer.

### Secrets

The provider clients read their keys from the environment in `from_env`. For your own settings, `load_setting(key, explicit, default)` resolves an explicit value, then a `.env` file in the working directory, then the environment variable, then the default; `SecretString` keeps a value out of `Debug` and `Display` output, with `expose_secret()` to read it.

```rust
use agent_framework::prelude::*;

fn main() {
    let token = load_setting("SERVICE_TOKEN", None, None).map(SecretString::new);
    match &token {
        Some(token) => println!("token configured: {token:?}"), // prints a mask, not the value
        None => println!("SERVICE_TOKEN is not set"),
    }
    let model = load_setting("CHAT_MODEL", None, Some("%%model=openai-small%%".into())).unwrap();
    println!("model: {model}");
}
```

Do not ship a `.env` file in a container image; inject secrets as environment variables or mounted files from the platform's secret store. MCP stdio servers start with a minimal environment (`PATH`, the temporary directory and locale), so an API key in the parent's environment does not reach them unless you pass it with `inherit_env_var`. Keep `ENABLE_SENSITIVE_DATA` unset in production: with it, spans carry prompts, replies and tool arguments.

### Logging and telemetry

The agent, the tool loop and `ObservableChatClient` emit `tracing` spans following the OpenTelemetry GenAI conventions, and the `otel-metrics` and `otel-export` features add token and duration histograms and an OTLP pipeline. The [observability page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/) covers configuration; the [batch job recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#a-scheduled-batch-job-with-retries-and-telemetry) composes a console layer with OTLP export. Two production points: the span exporter batches, so call `OtelPipeline::shutdown()` before the process exits or the last spans are lost; and agents without a `name` are reported under their generated id, so name every agent you want to find in a trace.

### Building and deploying

A release build of an agent service is a single native binary. The provider clients use reqwest with rustls and bundled Mozilla root certificates (webpki-roots), so the build needs no OpenSSL and the image needs no system CA bundle for public endpoints; a TLS-inspecting corporate proxy's root certificate is not trusted by those clients. Size and build time depend mostly on the features you enable. These release settings trade build time for a smaller, faster binary:

```toml
[profile.release]
lto = "thin"
codegen-units = 1
strip = "symbols"
```

Leave `panic` at its default of `unwind` for servers: tokio catches a panic in a spawned task and the server keeps running, while `panic = "abort"` would take the whole process down. A multi-stage container build:

```dockerfile
FROM rust:1.88-bookworm AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked && cp target/release/agent-service /agent-service

FROM gcr.io/distroless/cc-debian12
COPY --from=build /agent-service /agent-service
USER nonroot
ENV BIND=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/agent-service"]
```

The Dockerfile and the release profile were not built as part of checking this page. `distroless/cc` provides the C runtime that a default (glibc) Rust binary links against; for a `scratch` image, build for `x86_64-unknown-linux-musl` instead.

### Graceful shutdown

Container platforms stop a service with SIGTERM and kill it after a grace period. `axum::serve(listener, app).with_graceful_shutdown(signal)` stops accepting connections when `signal` resolves and waits for in-flight requests, including open SSE streams, to finish. Agent runs can be long, so make the platform's grace period (Kubernetes' `terminationGracePeriodSeconds` defaults to 30 s) longer than your per-request timeout, or a streamed reply is cut off at shutdown. Flush telemetry after `serve` returns. The [HTTP service recipe](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_recipes/#an-http-service-openai-compatible-with-sse) has a `shutdown_signal` that handles both Ctrl-C and SIGTERM.

### Persistence and HTTP security

Sessions and paused workflows live in memory until you persist them. Serialise sessions with `to_dict` plus their history provider, or use a durable history store (Redis, Cosmos DB, or the experimental file history and session store); see the [sessions page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/). Workflows survive a restart only with checkpointing to file or Cosmos DB storage, and checkpoints accumulate: prune old ones.

The hosting routers have no authentication by default. Apply `HostingSecurity` with a bearer token and an allowed-hosts list to the final composed router, set a body-size limit, and treat an AG-UI snapshot store as user data that needs an authorisation scope; the [hosting page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/) has the details.

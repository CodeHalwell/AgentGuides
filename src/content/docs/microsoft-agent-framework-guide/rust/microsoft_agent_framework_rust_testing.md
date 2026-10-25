---
title: "Microsoft Agent Framework (Rust) — Testing and production"
description: "Testing agent-framework-rs agents, tools, approvals, sessions, middleware, retries and workflows with a scripted ChatClient and cargo test, plus production notes: version pinning, timeouts, budgets, secrets and hosting security."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Testing and production"
  order: 5
---

The crate has no public fake or mock chat client. You do not need one: `ChatClient` is a two-method trait, and a test double that replays scripted replies takes about fifty lines. Everything above the client (the tool loop, approvals, sessions, middleware, retries, workflows, hosting) then runs for real in `cargo test`, without network access or an API key. All the tests on this page pass against `agent-framework` %%version=agent-framework-rs%%.

## A scripted chat client

Put the double in `tests/common/mod.rs` so every integration test file can use it. It replays a queue of replies and records each request, so a test can check both what the agent did and what it sent to the model.

```rust
// tests/common/mod.rs
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

/// One request the agent sent to the model.
#[derive(Clone, Debug)]
pub struct Request {
    pub messages: Vec<Message>,
    pub options: ChatOptions,
}

/// A ChatClient that replays scripted replies in order and records requests.
#[derive(Clone, Default)]
pub struct ScriptedClient {
    replies: Arc<Mutex<VecDeque<Result<ChatResponse>>>>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl ScriptedClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// The model answers with text.
    pub fn text(self, text: &str) -> Self {
        self.push(Ok(ChatResponse {
            finish_reason: Some(FinishReason::stop()),
            ..ChatResponse::from_text(text)
        }))
    }

    /// The model asks for a tool call.
    pub fn tool_call(self, name: &str, args: Value) -> Self {
        let call_id = format!("call_{}", self.replies.lock().unwrap().len());
        let call = FunctionCallContent::new(call_id, name, Some(FunctionArguments::Raw(args.to_string())));
        self.push(Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        }))
    }

    /// The service fails.
    pub fn error(self, error: Error) -> Self {
        self.push(Err(error))
    }

    fn push(self, reply: Result<ChatResponse>) -> Self {
        self.replies.lock().unwrap().push_back(reply);
        self
    }

    /// Every request so far.
    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        self.requests.lock().unwrap().push(Request { messages, options });
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(Error::other("the script has no more replies")))
    }

    async fn get_streaming_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatStream> {
        // Replay the next reply as one update per word, so streaming code is exercised.
        let response = self.get_response(messages, options).await?;
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
        updates.push(Ok(ChatResponseUpdate {
            finish_reason: response.finish_reason,
            ..Default::default()
        }));
        Ok(futures::stream::iter(updates).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some("scripted-model")
    }
}
```

Two details matter. Tool calls must be returned as `Content::FunctionCall` for the function-invocation loop to act on them, exactly as a real provider client does. And the agent's instructions arrive as a leading `system` message, so assertions on `requests()[n].messages` should allow for it.

The dev-dependencies are the same crates the examples use:

```toml
[dev-dependencies]
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
async-trait = "0.1"
futures = "0.3"
serde_json = "1"
```

## Tools, approvals and sessions

```rust
// tests/agent.rs
mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use common::ScriptedClient;
use futures::StreamExt;
use serde_json::json;

fn counting_tool(name: &'static str, calls: Arc<AtomicUsize>) -> FunctionTool {
    FunctionTool::new(
        name,
        "A test tool.",
        json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
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
async fn runs_the_tool_and_returns_the_result_to_the_model() {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = ScriptedClient::new()
        .tool_call("get_weather", json!({ "city": "York" }))
        .text("It is 21 degrees in York.");
    let agent = Agent::builder(client.clone())
        .instructions("Use tools.")
        .tool(counting_tool("get_weather", calls.clone()).into_definition())
        .build();

    let response = agent.run_once("Weather in York?").await.unwrap();

    assert_eq!(response.text(), "It is 21 degrees in York.");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // The second request carries the tool's result back to the model.
    let second = &client.requests()[1];
    let result = second
        .messages
        .iter()
        .flat_map(|m| m.function_results())
        .next()
        .expect("a tool result");
    assert_eq!(result.result.as_ref().unwrap()["city"], "York");
    // The tool was offered on the first request.
    assert_eq!(client.requests()[0].options.tools[0].name, "get_weather");
}

#[tokio::test]
async fn an_approval_pauses_until_answered() {
    let calls = Arc::new(AtomicUsize::new(0));
    let client = ScriptedClient::new()
        .tool_call("get_weather", json!({ "city": "Leeds" }))
        .text("Done.");
    let tool = counting_tool("get_weather", calls.clone())
        .with_approval_mode(ApprovalMode::AlwaysRequire)
        .into_definition();
    let agent = Agent::builder(client).tool(tool).build();
    let mut session = agent.create_session();

    let first = agent
        .run(vec![Message::user("Weather in Leeds?")], Some(&mut session))
        .await
        .unwrap();
    let requests = first.user_input_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "nothing runs before approval");

    let approval = Content::FunctionApprovalResponse(requests[0].create_response(true));
    let second = agent
        .run(vec![Message::with_contents(Role::user(), vec![approval])], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(second.text(), "Done.");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_session_carries_history_between_runs() {
    let client = ScriptedClient::new().text("Hello, Ada.").text("Your name is Ada.");
    let agent = Agent::builder(client.clone()).build();
    let mut session = agent.create_session();

    agent.run(vec![Message::user("I am Ada.")], Some(&mut session)).await.unwrap();
    agent.run(vec![Message::user("Who am I?")], Some(&mut session)).await.unwrap();

    let texts: Vec<String> = client.requests()[1].messages.iter().map(Message::text).collect();
    assert_eq!(texts, ["I am Ada.", "Hello, Ada.", "Who am I?"]);
}

#[tokio::test]
async fn streaming_yields_the_scripted_words() {
    let agent = Agent::builder(ScriptedClient::new().text("one two three")).build();
    let mut stream = agent.run_stream_once("Count").await.unwrap();
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
    }
    assert_eq!(text, "one two three");
}

#[tokio::test]
async fn structured_output_parses_into_a_struct() {
    #[derive(serde::Deserialize)]
    struct Verdict {
        approved: bool,
    }
    let agent = Agent::builder(ScriptedClient::new().text(r#"{"approved": true}"#))
        .response_format(ResponseFormat::json_schema(
            "Verdict",
            json!({ "type": "object", "properties": { "approved": { "type": "boolean" } } }),
        ))
        .build();
    let response = agent.run_once("Approve?").await.unwrap();
    assert!(response.parse_json::<Verdict>().unwrap().approved);
    assert_eq!(response.value, Some(json!({ "approved": true })));
}
```

The session test shows a useful property of the double: because it records requests, a test can pin down exactly what history, instructions and tools reached the model, which is where most agent bugs are.

## Errors, retries and middleware

A scripted error checks the retry policy without waiting: set the delays to a millisecond and jitter to zero. Middleware can be tested inside an agent, or on its own by running a `MiddlewarePipeline` with a terminal closure that stands in for the rest of the run.

```rust
// tests/policy.rs
mod common;

use std::sync::Arc;
use std::time::Duration;

use agent_framework::prelude::*;
use agent_framework::tools::BoxFuture;
use async_trait::async_trait;
use common::ScriptedClient;
use serde_json::json;

#[tokio::test]
async fn a_503_is_retried_and_a_401_is_not() {
    let fast = || RetryPolicy::with_max_retries(2).initial_delay(Duration::from_millis(1)).jitter(0.0);

    let flaky = ScriptedClient::new()
        .error(Error::service_status(503, "unavailable", None))
        .text("recovered");
    let agent = Agent::builder(RetryingChatClient::new(flaky.clone()).with_policy(fast())).build();
    assert_eq!(agent.run_once("hi").await.unwrap().text(), "recovered");
    assert_eq!(flaky.requests().len(), 2);

    let denied = ScriptedClient::new().error(Error::service_invalid_auth("bad key"));
    let agent = Agent::builder(RetryingChatClient::new(denied.clone()).with_policy(fast())).build();
    let err = agent.run_once("hi").await.unwrap_err();
    assert!(matches!(err, Error::ServiceInvalidAuth { .. }));
    assert_eq!(denied.requests().len(), 1);
}

/// Refuses calls to the `deploy` tool.
struct BlockDeploy;

#[async_trait]
impl Middleware<FunctionInvocationContext> for BlockDeploy {
    async fn process(
        &self,
        ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        if ctx.function_name == "deploy" {
            return Err(Error::tool_rejected("deploy is disabled in tests"));
        }
        next.run(ctx).await
    }
}

#[tokio::test]
async fn function_middleware_blocks_a_tool_and_the_model_is_told() {
    let client = ScriptedClient::new().tool_call("deploy", json!({})).text("I could not deploy.");
    let deploy = FunctionTool::new("deploy", "Deploy.", json!({ "type": "object" }), |_| async {
        panic!("must not run")
    })
    .into_definition();
    let agent = Agent::builder(client.clone())
        .tool(deploy)
        .function_middleware(Arc::new(BlockDeploy))
        .build();

    assert_eq!(agent.run_once("Deploy").await.unwrap().text(), "I could not deploy.");
    let result = client.requests()[1]
        .messages
        .iter()
        .flat_map(|m| m.function_results())
        .next()
        .cloned()
        .unwrap();
    assert_eq!(result.exception.as_deref(), Some("deploy is disabled in tests"));
}

#[tokio::test]
async fn middleware_can_be_tested_without_an_agent() {
    let pipeline = MiddlewarePipeline::new(vec![Arc::new(BlockDeploy) as Arc<dyn Middleware<_>>]);
    let terminal = Box::new(|mut ctx: FunctionInvocationContext| {
        Box::pin(async move {
            ctx.result = Some(json!("ran"));
            Ok(ctx)
        }) as BoxFuture<Result<FunctionInvocationContext>>
    });
    let ctx = FunctionInvocationContext::new("status", json!({}));
    let ctx = pipeline.execute(ctx, terminal).await.unwrap();
    assert_eq!(ctx.result, Some(json!("ran")));
}
```

Tools can be tested with no agent at all. A `FunctionTool` implements the `Tool` trait, so `tool.invoke(json!({...})).await` runs it with raw JSON arguments exactly as the loop would, and `tool.parameters_schema()` returns the schema the model will see. A snapshot test on that schema catches accidental changes to a tool's contract; for `FunctionTool::typed`, an invalid argument object returns `Error::Tool` rather than panicking.

## Workflows

Graph workflows built from function executors need no model at all, and agent executors can use the scripted client. Assert on outputs, run state, pending requests and events:

```rust
// tests/workflow.rs
mod common;

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{AgentExecutor, FunctionExecutor};
use common::ScriptedClient;
use serde_json::json;

#[tokio::test]
async fn the_reviewer_sees_the_writers_draft() {
    let writer_client = ScriptedClient::new().text("Draft: Rust is fast.");
    let reviewer_client = ScriptedClient::new().text("APPROVED");
    let writer = Agent::builder(writer_client).name("writer").build();
    let reviewer = Agent::builder(reviewer_client.clone()).name("reviewer").build();

    let workflow = SequentialBuilder::new()
        .participants(vec![
            Arc::new(writer) as Arc<dyn SupportsAgentRun>,
            Arc::new(reviewer) as Arc<dyn SupportsAgentRun>,
        ])
        .build()
        .unwrap();
    let run = workflow.run("Write about Rust").await.unwrap();

    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap()).unwrap();
    assert_eq!(conversation.last().unwrap().text(), "APPROVED");
    let seen: Vec<String> = reviewer_client.requests()[0].messages.iter().map(Message::text).collect();
    assert!(seen.contains(&"Draft: Rust is fast.".to_string()));
}

#[tokio::test]
async fn a_request_pauses_the_run_and_the_answer_resumes_it() {
    let gate = FunctionExecutor::new("gate", |message, ctx| async move {
        match agent_framework::workflow::RequestResponse::from_message(&message) {
            Some(answer) => ctx.yield_output(answer.data).await?,
            None => ctx.request_info(json!({ "question": "Proceed?" })).await?,
        }
        Ok(())
    });
    let summarise = AgentExecutor::new(
        "summarise",
        Arc::new(Agent::builder(ScriptedClient::new().text("summary")).build()),
    );
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(summarise))
        .add_executor(Arc::new(gate))
        .set_start("summarise")
        .add_edge("summarise", "gate")
        .build()
        .unwrap();

    let mut run = workflow.run("A long text").await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let request = run.pending_requests().remove(0);
    assert_eq!(request.request_data["question"], "Proceed?");

    run.send_response(request.request_id, json!("yes")).await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!("yes")));
}
```

## What a scripted client cannot tell you

A script checks your wiring: which tools exist, how arguments decode, how approvals, history and errors flow, and where a workflow routes. It cannot tell you whether a real model will choose the tool, follow the instructions or produce valid structured output. Keep a few tests against a real deployment for that, skipped when no key is configured:

```rust
// tests/live.rs
use agent_framework::prelude::*;

#[tokio::test]
async fn live_answer_mentions_paris() {
    let Ok(client) = OpenAIChatCompletionClient::from_env("%%model=openai-small%%") else {
        eprintln!("OPENAI_API_KEY not set; skipping");
        return;
    };
    let agent = Agent::builder(client).instructions("Answer in one word.").build();
    let answer = agent.run_once("What is the capital of France?").await.unwrap();
    assert!(answer.text().to_lowercase().contains("paris"));
}
```

There is no evaluation framework in the Rust port comparable to Python's `evaluate_agent`; score recorded responses in your own harness. The repository's `examples/providers/custom_chat_client.rs` is a longer reference for writing a client, covering option handling, usage reporting and error classification.

## Production notes

- **Pin the version.** The crate is pre-1.0 and its changelog follows the convention that a minor release may break the API (0.13.0 had two small breaks, 0.11.0 four). Cargo's default `"0.14"` requirement already stops at 0.15; read the changelog before moving. Anything behind an `experimental-*` feature can change in any release. The minimum supported Rust version is 1.88.
- **Enable only the features you use.** Each provider and integration pulls its own dependencies; `full` is convenient in examples but builds everything. Set `default-features = false` if you do not use OpenAI.
- **Set your own timeouts.** The provider clients build plain `reqwest` clients, and the OpenAI, Anthropic, Gemini and Mistral clients have no timeout setting, so a request to an unresponsive endpoint waits indefinitely. Wrap runs in `tokio::time::timeout`. `max_duration_seconds` does not help here: it is checked between tool batches, not during a model call.
- **Bound the tool loop.** Set `max_function_calls` and `max_duration_seconds` in `FunctionInvocationConfig` for unattended agents; both survive approval round trips, so a paused run cannot reset its own budget. Use `max_invocations` on individual expensive tools.
- **Retry deliberately.** Wrap clients in `RetryingChatClient`. It honours `Retry-After`, and it only retries errors that are classified as transient, so a custom `ChatClient` should return `Error::service_status` and the other classified variants rather than `Error::other`.
- **Keep secrets out of telemetry.** Content capture in spans is off unless `ENABLE_SENSITIVE_DATA` is set; leave it off in production. `SecretString` keeps keys out of `Debug` output.
- **Guard every HTTP surface.** The hosting routers have no authentication by default. Apply `HostingSecurity` with a bearer token and an allowed-hosts list to the final composed router, and pair an AG-UI snapshot store with an authorisation scope.
- **Least privilege for tools.** MCP stdio servers start with a minimal environment; pass only the variables they need. Keep `LocalShellTool` approval on, or use `DockerShellTool` with its defaults.
- **Persist what must survive a restart.** Sessions are in memory until you serialise them (or use a Redis or Cosmos history store); paused workflows survive only with checkpointing to file or Cosmos storage.
- **Watch `finish_reason`.** A `length` or `content_filter` finish produces a response that looks complete. Check it before acting on the text.

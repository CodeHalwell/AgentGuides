---
title: "Microsoft Agent Framework (Rust) — Tools"
description: "Function tools, typed schemas, the tool loop and its limits, approvals, agents as tools, tools that change at run time, provider-hosted tools and the shell tools in the community Rust port agent-framework-rs."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Tools"
  order: 4
---

This page covers everything an agent in `agent-framework` can call: local function tools, the loop that runs them, human approval, other agents wrapped as tools, tool lists that change at run time, tools the provider runs on its own servers, and the shell tools in the `agent-framework-tools` crate. Interception of tool calls (function middleware) is on the [middleware page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_middleware/), and MCP servers have [their own page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/).

Every program here was compiled against `agent-framework` %%version=agent-framework-rs%%. The ones that use `OpenAIChatCompletionClient` were run against a local stand-in for the OpenAI API; most of the others use a scripted `ChatClient` so that the behaviour being described is visible in the output, and need no network at all. The [testing page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_testing/) explains the scripted-client technique.

## How tools are represented

An agent never holds a `FunctionTool` or an MCP client directly. Everything it can call is a `ToolDefinition`, a plain cloneable struct with public fields:

| Field | Meaning |
| --- | --- |
| `name`, `description` | What the model sees |
| `parameters` | The JSON Schema of the arguments, sent to the provider unmodified |
| `kind` | `ToolKind::Function`, or one of the hosted kinds (`HostedWebSearch`, `HostedCodeInterpreter`, `HostedFileSearch { max_results }`, `HostedImageGeneration`, `HostedMcp { url, allowed_tools }`) |
| `approval_mode` | `ApprovalMode::NeverRequire` (the default) or `ApprovalMode::AlwaysRequire` |
| `executor` | `Option<Arc<dyn Tool>>`: the local implementation, or `None` |

The combination of `kind` and `executor` decides what happens to a call. A `Function` tool with an executor runs in your process; `is_executable()` is true only for these. A `Function` tool without an executor is *declaration-only*: its calls are handed back to the caller (see [per-run tools](#per-run-tools-and-declaration-only-tools)). A hosted kind never has an executor and is run by the provider.

`AgentBuilder::tool` and `tools` add definitions to the agent's default options, `AgentRunOptions::with_tool` adds them for one run, and `AgentBuilder::tool_source` adds a source that is asked for its tools on every run. Names must be unique. A tool from a context provider, a per-run option or a tool source is dropped when its name is already taken, but two tools with the same name given to the builder are both sent, which most providers reject. `ToolDefinition::to_openai_spec()` shows the OpenAI `{"type": "function", ...}` form of a function tool, which is handy when debugging a schema.

## Function tools

### Hand-written schemas: `FunctionTool::new`

`FunctionTool::new(name, description, schema, closure)` takes the schema as a `serde_json::Value` and gives the closure the arguments as a `Value`. The closure must be `Fn` (it may run many times, and concurrently), `Send + Sync + 'static`, and return a future of `Result<Value>`. State is shared the usual way: clone an `Arc` into the closure, and clone it again inside for each call's future.

```rust
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<()> {
    // State the tool writes to. The closure owns one clone of the Arc and
    // hands a fresh clone to each call's future.
    let orders: Arc<Mutex<Vec<String>>> = Arc::default();
    let book = orders.clone();

    let place_order = FunctionTool::new(
        "place_order",
        "Place an order for a product and return its order number.",
        json!({
            "type": "object",
            "properties": {
                "product": { "type": "string", "description": "The product name" },
                "quantity": { "type": "integer", "minimum": 1 }
            },
            "required": ["product", "quantity"],
            "additionalProperties": false
        }),
        move |args: Value| {
            let book = book.clone();
            async move {
                let product = args["product"]
                    .as_str()
                    .ok_or_else(|| Error::tool("product must be a string"))?;
                let quantity = args["quantity"].as_u64().unwrap_or(1);
                let mut orders = book.lock().unwrap();
                orders.push(format!("{quantity} x {product}"));
                Ok(json!({ "order_number": orders.len(), "status": "placed" }))
            }
        },
    )
    .into_definition();

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You take orders. Use place_order for every order.")
        .tool(place_order)
        .build();

    let response = agent.run_once("Two boxes of tea, please.").await?;

    // The response holds the whole exchange: the call, its result, the answer.
    for message in &response.messages {
        for content in &message.contents {
            match content {
                Content::FunctionCall(call) => println!("call   {} {:?}", call.name, call.arguments),
                Content::FunctionResult(result) => println!("result {:?}", result.result),
                Content::Text(text) => println!("text   {}", text.text),
                _ => {}
            }
        }
    }
    println!("orders: {:?}", orders.lock().unwrap());
    Ok(())
}
```

Run against the stand-in, this prints:

```text
call   place_order Some(Raw("{\"product\": \"example product\", \"quantity\": 3}"))
result Some(Object {"order_number": Number(1), "status": String("placed")})
text   Here is what the tool returned: {"order_number":1,"status":"placed"}
orders: ["3 x example product"]
```

Three details are worth knowing. `response.messages` contains the whole exchange, not just the final answer: the assistant message with the call, the tool message with its result, then the answer, so the history a session records includes the tool traffic. The arguments arrive as `FunctionArguments::Raw` (the provider's JSON string); the loop parses them into an object before calling the tool, and a string that is not JSON never reaches the tool at all. Nothing checks the arguments against your schema: if the model sends `"quantity": "three"`, the closure receives exactly that, so validate what you read.

### Derived schemas: `FunctionTool::typed`

`FunctionTool::typed(name, description, closure)` derives the schema from the closure's argument type, which must implement `serde::Deserialize` and `schemars::JsonSchema` (schemars **0.8**, the version the crate uses). The closure receives the deserialised value and may return any `Serialize` type. This is the closest equivalent of Python's `@tool` decorator, with the schema coming from a struct instead of a function signature.

```rust
use agent_framework::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Priority {
    Low,
    Normal,
    High,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Site {
    building: String,
    floor: Option<u8>,
}

/// Arguments for create_ticket.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct TicketArgs {
    /// A one-line summary of the problem.
    title: String,
    /// How urgent the ticket is.
    priority: Priority,
    /// Estimated effort in hours.
    #[schemars(range(min = 1, max = 40))]
    hours: u32,
    /// Free-form labels.
    #[serde(default)]
    labels: Vec<String>,
    /// Where the problem is, if it is somewhere physical.
    site: Option<Site>,
}

/// Any `Serialize` type can be the result.
#[derive(Serialize)]
struct Ticket {
    id: u32,
    title: String,
    priority: String,
    hours: u32,
    labels: Vec<String>,
    location: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let create_ticket = FunctionTool::typed(
        "create_ticket",
        "Open a support ticket.",
        |args: TicketArgs| async move {
            Ok(Ticket {
                id: 101,
                title: args.title,
                priority: format!("{:?}", args.priority),
                hours: args.hours,
                labels: args.labels,
                location: args.site.map(|s| format!("{}, floor {:?}", s.building, s.floor)),
            })
        },
    );

    // The schema the model is given.
    println!("{}", serde_json::to_string_pretty(&create_ticket.parameters_schema())?);

    // Tool::invoke is what the tool loop calls: JSON in, JSON out.
    let ok = create_ticket
        .invoke(json!({ "title": "Printer jam", "priority": "high", "hours": 2 }))
        .await?;
    println!("ok:  {ok}");

    // Arguments that do not deserialise are an Error::Tool, not a panic.
    let bad = create_ticket
        .invoke(json!({ "title": "Printer jam", "priority": "urgent", "hours": 2 }))
        .await
        .unwrap_err();
    println!("bad: {bad}");

    // The range is advice to the model, not validation: 0 deserialises fine.
    let out_of_range = create_ticket
        .invoke(json!({ "title": "Printer jam", "priority": "low", "hours": 0 }))
        .await?;
    println!("0 hours accepted: {out_of_range}");
    Ok(())
}
```

The schema it prints shows the rules:

- The struct's doc comment becomes the top-level `description`, and each field's doc comment becomes that property's `description`. These are what the model reads, so write them for the model.
- A field is listed in `required` unless it is an `Option<_>` or has `#[serde(default)]`; a defaulted field also gets a `default` in the schema.
- `serde` attributes are honoured: `rename_all = "lowercase"` gives the enum values `low`, `normal`, `high`, and `deny_unknown_fields` becomes `"additionalProperties": false`.
- Nested structs and enums are not inlined. They go into a top-level `definitions` map and are referenced with `$ref` (wrapped in `allOf` when the field has a description, `anyOf` with `null` when it is optional). Every provider converter forwards the schema unmodified, so this works wherever the provider accepts `$ref`; flatten the type if one does not.
- The top-level `$schema` and `title` keys are removed.

The schema is generated once, when the tool is built. When the model's arguments do not deserialise into the type, the tool returns `Error::Tool("invalid arguments for tool 'create_ticket': unknown variant ...")` instead of running the closure, and the model sees a failed call (see [what the model sees](#what-the-model-sees-when-a-tool-fails)). A result that fails to serialise is reported the same way.

### Controlling the schema

`schemars` attributes change the schema without changing the type. `#[schemars(range(min = 1, max = 40))]` adds `minimum` and `maximum`, `#[schemars(length(max = 80))]` adds string or array bounds, and `#[schemars(description = "...")]` overrides a doc comment. These are advice to the model, not validation: the last call in the example passes `"hours": 0` and the tool accepts it, because `serde` only checks types. If a bound matters, check it in the closure and return an error the model can act on.

When the derived schema is not what you want (a provider that rejects `$ref`, a strict-mode schema with every property required, a `oneOf` that schemars cannot express), use `FunctionTool::new` with a hand-written schema and deserialise inside the closure with `serde_json::from_value`. `agent_framework::tools::empty_schema()` returns the `{"type": "object", "properties": {}}` schema for a tool with no arguments.

### Implementing `Tool` yourself

A struct that implements the `Tool` trait is a tool too, and `Arc::new(tool).into()` (or `ToolDefinition::from_tool`) turns it into a definition. The trait has four required methods (`name`, `description`, `parameters_schema`, `invoke`) and one provided one, `invoke_in_context`, which the tool loop calls with the `FunctionInvocationContext` of the call. Its default forwards to `invoke`; override it to read the run's session, the middleware metadata (the loop puts the model's `call_id` there) or, with the progressive-tools feature, the live tool list. This is the Rust counterpart of a Python tool function that declares a `ctx: FunctionInvocationContext` parameter.

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::{json, Value};

/// A tool written as a struct: it has its own configuration and reads the
/// session of the run that called it.
struct SaveNote {
    max_len: usize,
}

#[async_trait]
impl Tool for SaveNote {
    fn name(&self) -> &str {
        "save_note"
    }
    fn description(&self) -> &str {
        "Save a short note for the current user."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
    }

    // Called when the tool is invoked outside an agent run.
    async fn invoke(&self, _arguments: Value) -> Result<Value> {
        Err(Error::tool("save_note needs an agent session"))
    }

    // The tool loop calls this one; the default forwards to `invoke`.
    async fn invoke_in_context(
        &self,
        arguments: Value,
        ctx: &FunctionInvocationContext,
    ) -> Result<Value> {
        let Some(session) = &ctx.session else {
            return self.invoke(arguments).await;
        };
        let text: String = arguments["text"].as_str().unwrap_or_default().chars().take(self.max_len).collect();
        let user = session
            .state
            .get("user")
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| "anonymous".into());
        let mut notes = session.state.get("notes").unwrap_or_else(|| json!([]));
        notes.as_array_mut().unwrap().push(json!(text));
        session.state.insert("notes", notes);
        Ok(json!({ "saved_for": user, "call_id": ctx.metadata["call_id"] }))
    }
}

/// A canned model: asks for one tool call, then answers.
#[derive(Clone)]
struct Scripted(Arc<Mutex<VecDeque<ChatResponse>>>);

#[async_trait]
impl ChatClient for Scripted {
    async fn get_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        Ok(self.0.lock().unwrap().pop_front().unwrap_or_else(|| ChatResponse::from_text("done")))
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

fn tool_call(id: &str, name: &str, args: Value) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                id,
                name,
                Some(FunctionArguments::Raw(args.to_string())),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = Scripted(Arc::new(Mutex::new(VecDeque::from([
        tool_call("call_1", "save_note", json!({ "text": "Buy oat milk on the way home" })),
        ChatResponse::from_text("Saved."),
    ]))));

    // From<Arc<T: Tool>> builds the ToolDefinition.
    let agent = Agent::builder(model)
        .tool(Arc::new(SaveNote { max_len: 12 }).into())
        .build();

    let mut session = agent.create_session();
    session.state.insert("user", json!("ada"));
    let response = agent.run(vec![Message::user("Note: buy oat milk")], Some(&mut session)).await?;

    for content in response.messages.iter().flat_map(|m| &m.contents) {
        if let Content::FunctionResult(result) = content {
            println!("tool result: {:?}", result.result);
        }
    }
    println!("notes in session: {:?}", session.state.get("notes"));

    // Outside a run there is no context, so `invoke` is what runs.
    println!("direct call: {:?}", SaveNote { max_len: 12 }.invoke(json!({ "text": "x" })).await);
    Ok(())
}
```

```text
tool result: Some(Object {"call_id": String("call_1"), "saved_for": String("ada")})
notes in session: Some(Array [String("Buy oat milk")])
direct call: Err(Tool("save_note needs an agent session"))
```

`ctx.session` is a clone of the run's `AgentSession` whose state bag shares storage with the original, so writes made by the tool are visible to the caller after the run. It is `None` when the agent runs without a session: `run_once` and `run(.., None)` create a throwaway one, so in practice it is present for every agent run, but not when you call `invoke` yourself.

## What the model sees when a tool fails

A tool error does not fail the run. The loop turns it into a `FunctionResultContent` with `result: None` and an `exception` string, sends that to the model, and lets the model try again, try something else or explain. What the string says depends on the error and on `FunctionInvocationConfig`:

```rust
use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

/// A model that asks for every call in `batch` at once, then reports what
/// came back for each call.
#[derive(Clone)]
struct Reporter {
    batch: Vec<(&'static str, Value)>,
}

#[async_trait]
impl ChatClient for Reporter {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let results: Vec<String> = messages
            .iter()
            .flat_map(|m| &m.contents)
            .filter_map(|c| match c {
                Content::FunctionResult(r) => Some(format!(
                    "  {} -> result={:?} exception={:?}",
                    r.call_id, r.result, r.exception
                )),
                _ => None,
            })
            .collect();
        if !results.is_empty() {
            return Ok(ChatResponse::from_text(results.join("\n")));
        }
        let calls = self
            .batch
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                Content::FunctionCall(FunctionCallContent::new(
                    format!("{name}#{i}"),
                    *name,
                    Some(FunctionArguments::Raw(args.to_string())),
                ))
            })
            .collect();
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), calls)],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct LookupArgs {
    id: u32,
}

fn tools() -> Vec<ToolDefinition> {
    let empty = json!({ "type": "object", "properties": {} });
    vec![
        FunctionTool::new("flaky", "Fails with internal detail.", empty.clone(), |_| async {
            Err::<Value, _>(Error::tool("db timeout on host pg-07 (10.0.3.7)"))
        })
        .into_definition(),
        FunctionTool::new("refuse", "Refuses with a message for the model.", empty.clone(), |_| async {
            Err::<Value, _>(Error::tool_rejected("Refunds over 100 EUR need a manager."))
        })
        .into_definition(),
        FunctionTool::typed("lookup", "Look a record up by id.", |args: LookupArgs| async move {
            Ok(json!({ "id": args.id }))
        })
        .into_definition(),
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    let batch = vec![
        ("flaky", json!({})),
        ("refuse", json!({})),
        ("lookup", json!({ "id": "seven" })), // wrong type
        ("missing", json!({})),               // no such tool
    ];

    for detailed in [false, true] {
        let agent = Agent::builder(Reporter { batch: batch.clone() })
            .tools(tools())
            .function_invocation_config(FunctionInvocationConfig {
                include_detailed_errors: detailed,
                ..Default::default()
            })
            .build();
        println!("include_detailed_errors = {detailed}");
        println!("{}", agent.run_once("go").await?.text());
    }

    // An unknown tool can also end the run instead.
    let strict = Agent::builder(Reporter { batch: vec![("missing", json!({}))] })
        .tools(tools())
        .function_invocation_config(FunctionInvocationConfig {
            terminate_on_unknown_calls: true,
            ..Default::default()
        })
        .build();
    println!("terminate_on_unknown_calls: {:?}", strict.run_once("go").await.map(|r| r.text()));

    // So does Error::MiddlewareFailure, from a tool or from function middleware.
    let fail_closed = FunctionTool::new("halt", "Stops the run.", json!({ "type": "object" }), |_| async {
        Err::<Value, _>(Error::middleware_failure("policy: tool disabled for this tenant"))
    })
    .into_definition();
    let agent = Agent::builder(Reporter { batch: vec![("halt", json!({}))] })
        .tool(fail_closed)
        .build();
    println!("MiddlewareFailure: {:?}", agent.run_once("go").await.map(|r| r.text()));

    // Per-tool caps: the third call is refused before the closure runs.
    let once = FunctionTool::new("once", "Can run twice.", json!({ "type": "object" }), |_| async {
        Ok(json!("ran"))
    })
    .max_invocations(2);
    let counter = once.clone(); // clones share the counters
    let agent = Agent::builder(Reporter { batch: vec![("once", json!({})), ("once", json!({})), ("once", json!({}))] })
        .function_invocation_config(FunctionInvocationConfig {
            include_detailed_errors: true,
            ..Default::default()
        })
        .tool(once.into_definition())
        .build();
    println!("max_invocations(2):\n{}", agent.run_once("go").await?.text());
    println!("invocation_count = {}", counter.invocation_count());
    Ok(())
}
```

```text
include_detailed_errors = false
  flaky#0 -> result=None exception=Some("tool execution failed")
  refuse#1 -> result=None exception=Some("Refunds over 100 EUR need a manager.")
  lookup#2 -> result=None exception=Some("tool execution failed")
  missing#3 -> result=None exception=Some("tool 'missing' not found")
include_detailed_errors = true
  flaky#0 -> result=None exception=Some("tool error: db timeout on host pg-07 (10.0.3.7)")
  refuse#1 -> result=None exception=Some("Refunds over 100 EUR need a manager.")
  lookup#2 -> result=None exception=Some("tool error: invalid arguments for tool 'lookup': invalid type: string \"seven\", expected u32")
  missing#3 -> result=None exception=Some("tool 'missing' not found")
terminate_on_unknown_calls: Err(Tool("unknown tool: missing"))
MiddlewareFailure: Err(MiddlewareFailure("policy: tool disabled for this tenant"))
max_invocations(2):
  once#0 -> result=Some(String("ran")) exception=None
  once#1 -> result=Some(String("ran")) exception=None
  once#2 -> result=None exception=Some("tool error: Function 'once' has reached its maximum invocation limit, you can no longer use this tool.")
invocation_count = 2
```

| Outcome | What the model receives | Run continues? |
| --- | --- | --- |
| `Ok(value)` | `value` as the result | yes |
| `Err(Error::ToolRejected(msg))` | `msg`, verbatim, whatever the detail setting | yes |
| Any other error (`Error::Tool`, `Error::Service`, ...) | `"tool execution failed"`, or the error's `Display` text with `include_detailed_errors: true` | yes |
| A tool name that does not exist | `"tool '<name>' not found"` | yes, unless `terminate_on_unknown_calls` is set, which fails the run with `Error::Tool("unknown tool: <name>")` |
| Arguments that are not a JSON object | `"invalid tool arguments"` (with the parse error when detailed) | yes; the tool and its middleware never run |
| `Err(Error::MiddlewareFailure(msg))` | nothing: the run fails with that error | no |

`include_detailed_errors` is off by default because a tool's error text can carry internal detail (hosts, SQL, file paths) that should not reach the model or, through it, the user. `Error::ToolRejected` exists for the opposite case: a message deliberately written for the model, such as a policy refusal. `Error::MiddlewareFailure` is the fail-closed signal; it is carried by the error type, so a tool that returns it stops the run just as function middleware would. With concurrent calls, sibling calls still in flight are cancelled; with `allow_concurrent_invocation: false`, the calls after it never start.

Two more rules keep a confused model from looping. After more than `max_consecutive_errors_per_request` (default 3) consecutive rounds that contained a failed call, the loop sets the tool choice to `none` and the model has to answer. And an error result is stored redacted: `FunctionResultContent::exception` serialises as a fixed marker, so history written to Redis, Cosmos, a file or a workflow checkpoint does not keep whatever the failing tool said. The model sees the real text during the run.

### Per-tool limits

`FunctionTool::max_invocations(n)` caps how many times the closure may run, and `max_invocation_exceptions(n)` how many failures it tolerates; past either limit the tool returns an `Error::Tool` with a fixed message instead of running. The slot is reserved atomically, so parallel calls cannot slip past the cap. `invocation_count()` and `invocation_exception_count()` report the counters. The counters live in the `FunctionTool` and are shared by its clones, so they are not per run: an agent built once and used for many requests stops calling a `max_invocations(1)` tool after the first request. Python validates that the limits are at least 1; here `0` is accepted and means the tool never runs. For a per-run limit, use [`max_function_calls`](#the-three-limits) or a [one-shot tool](#progressive-tools-experimental).

## The tool loop

`Agent::builder` wraps the client in a `FunctionInvokingChatClient`, which runs the loop: send the request, execute any function calls in the response, append the results, and call the model again until it answers without calling a tool. The builder's `function_invocation_config` is the only way to configure it for an agent (wrapping a client in your own `FunctionInvokingChatClient` first would just add a second loop).

| `FunctionInvocationConfig` field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | `false` turns the loop off: function calls come back in the response unexecuted |
| `max_iterations` | 40 | Model round trips that may request tools |
| `max_function_calls` | `None` | Tool executions per request, across iterations and approval round trips |
| `max_duration_seconds` | `None` | Wall-clock budget for the loop, including time waiting for approvals |
| `max_consecutive_errors_per_request` | 3 | Consecutive failing rounds tolerated before tools are switched off |
| `terminate_on_unknown_calls` | `false` | Fail the run on a call to a tool that does not exist |
| `include_detailed_errors` | `false` | Send a failed tool's error text to the model |
| `allow_concurrent_invocation` | `true` | Run the calls of one model response concurrently |

The values are checked when a run starts, not when the agent is built: `max_iterations: 0`, `max_function_calls: Some(0)`, or a duration that is not positive and finite makes every run fail with `Error::Configuration`.

### Tool choice

`ToolMode` is the wire-level `tool_choice`. Set it per run with `ChatOptions::with_tool_choice`, or as an agent default through the builder's `chat_options`:

```rust
use agent_framework::prelude::*;
use async_trait::async_trait;
use serde_json::json;

/// Reports the tool choice and tool list each request carries.
#[derive(Clone)]
struct Echo;

#[async_trait]
impl ChatClient for Echo {
    async fn get_response(&self, _m: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let tools: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
        Ok(ChatResponse::from_text(format!(
            "tool_choice={:?} tools={tools:?}",
            options.tool_choice
        )))
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

fn tool(name: &str) -> ToolDefinition {
    FunctionTool::new(name, "A tool.", json!({ "type": "object", "properties": {} }), |_| async {
        Ok(json!("ok"))
    })
    .into_definition()
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(Echo).tool(tool("search")).tool(tool("extract")).build();

    // Unset: the loop fills in Auto whenever tools are present.
    println!("{}", agent.run_once("hi").await?.text());

    for mode in [
        ToolMode::none(),
        ToolMode::required_any(),
        ToolMode::required_function("extract"),
    ] {
        let options = AgentRunOptions::new()
            .with_chat_options(ChatOptions::new().with_tool_choice(mode));
        let response = agent.run_with_options(vec![Message::user("hi")], None, options).await?;
        println!("{}", response.text());
    }

    // The choice can also be an agent default, through the builder's chat options.
    let forced = Agent::builder(Echo)
        .tool(tool("extract"))
        .chat_options(ChatOptions::new().with_tool_choice(ToolMode::required_function("extract")))
        .build();
    println!("{}", forced.run_once("hi").await?.text());
    Ok(())
}
```

```text
tool_choice=Some(Auto) tools=["search", "extract"]
tool_choice=Some(None) tools=["search", "extract"]
tool_choice=Some(Required(None)) tools=["search", "extract"]
tool_choice=Some(Required(Some("extract"))) tools=["search", "extract"]
tool_choice=Some(Required(Some("extract"))) tools=["extract"]
```

`ToolMode::auto()` lets the model decide and is filled in whenever tools are present and nothing was set. `ToolMode::none()` keeps the tools in the request but forbids calls. `ToolMode::required_any()` forces some call, and `ToolMode::required_function("name")` a specific one, which is how extraction pipelines make the tool schema the output schema. `ToolMode::as_str()` gives the wire value (`"auto"`, `"none"`, `"required"`).

The loop does not relax a required choice after the first call. With `required_any()` and an obliging model, every round trip is forced to call a tool, so the run ends only when `max_iterations` runs out and the final tools-off call is made. In a test with `max_iterations: 3` the model was called with `required` three times and then once with `none`, running six tool calls on the way. Use a required choice with a low `max_iterations`, or, for single-shot extraction, read the call from the response with the loop disabled (`enabled: false`).

### The three limits

The limits bound different shapes of runaway run: `max_iterations` a model that calls one tool per turn for ever, `max_function_calls` one that asks for many parallel calls per turn, and `max_duration_seconds` slow tools or approvals nobody answers. None of them fails the run. When one is reached, the loop makes one last request with the tool choice set to `none` and returns whatever the model answers.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// A model that asks for three `work` calls on every turn it is allowed to,
/// and records the tool choice of each request.
#[derive(Clone, Default)]
struct Greedy {
    choices: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ChatClient for Greedy {
    async fn get_response(&self, _m: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let choice = options.tool_choice.as_ref().map(ToolMode::as_str).unwrap_or("unset");
        self.choices.lock().unwrap().push(choice.to_string());
        if choice == "none" {
            return Ok(ChatResponse::from_text("final answer without tools"));
        }
        let n = self.choices.lock().unwrap().len();
        let calls = (0..3)
            .map(|i| {
                Content::FunctionCall(FunctionCallContent::new(
                    format!("c{n}-{i}"),
                    "work",
                    Some(FunctionArguments::Raw("{}".into())),
                ))
            })
            .collect();
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), calls)],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

async fn run(label: &str, config: FunctionInvocationConfig, delay: Duration) -> Result<()> {
    let executed = Arc::new(AtomicUsize::new(0));
    let count = executed.clone();
    let work = FunctionTool::new("work", "Do a unit of work.", json!({ "type": "object" }), move |_| {
        let count = count.clone();
        async move {
            tokio::time::sleep(delay).await;
            count.fetch_add(1, Ordering::SeqCst);
            Ok(json!("done"))
        }
    })
    .into_definition();

    let model = Greedy::default();
    let choices = model.choices.clone();
    let agent = Agent::builder(model)
        .tool(work)
        .function_invocation_config(config)
        .build();
    let response = agent.run_once("start").await?;
    let choices = choices.lock().unwrap().clone();
    println!(
        "{label}: {} model calls {choices:?}, {} tool runs, answer: {:?}",
        choices.len(),
        executed.load(Ordering::SeqCst),
        response.text()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let fast = Duration::from_millis(0);
    run(
        "max_iterations=2",
        FunctionInvocationConfig { max_iterations: 2, ..Default::default() },
        fast,
    )
    .await?;
    run(
        "max_function_calls=4",
        FunctionInvocationConfig { max_function_calls: Some(4), ..Default::default() },
        fast,
    )
    .await?;
    run(
        "max_duration_seconds=0.25",
        FunctionInvocationConfig { max_duration_seconds: Some(0.25), ..Default::default() },
        Duration::from_millis(100),
    )
    .await?;

    // Invalid values are rejected when the run starts, not at build time.
    let agent = Agent::builder(Greedy::default())
        .function_invocation_config(FunctionInvocationConfig {
            max_function_calls: Some(0),
            ..Default::default()
        })
        .tool(FunctionTool::new("work", "", json!({}), |_| async { Ok(json!(1)) }).into_definition())
        .build();
    println!("max_function_calls=0: {:?}", agent.run_once("start").await.map(|r| r.text()));
    Ok(())
}
```

```text
max_iterations=2: 3 model calls ["auto", "auto", "none"], 6 tool runs, answer: "final answer without tools"
max_function_calls=4: 3 model calls ["auto", "auto", "none"], 6 tool runs, answer: "final answer without tools"
max_duration_seconds=0.25: 4 model calls ["auto", "auto", "auto", "none"], 9 tool runs, answer: "final answer without tools"
max_function_calls=0: Err(Configuration("max_function_calls must be >= 1, or None for unbounded"))
```

The output shows how each is counted:

- **`max_iterations`** counts model requests that may call tools, so a run makes at most `max_iterations + 1` model calls.
- **`max_function_calls`** is checked after each batch, never inside one, because a half-executed batch would leave calls without results, which providers reject. With four calls allowed and three per batch, the second batch runs in full and the total is six. Only executions count: a call to an unknown tool or with unparseable arguments does not, while a call that function middleware stopped does.
- **`max_duration_seconds`** is measured from the start of the loop and checked after each batch and again after each model response. A response that arrives after the budget is spent has its unexecuted calls removed (results the provider produced itself, such as an Anthropic server-side search, are kept) and the final tools-off call is made.

Both budgets are cumulative across approval round trips. When a run pauses for approval, the loop parks the elapsed time and call count in the session's state (under the reserved key `__af_function_invocation_budget__`), and the request that carries the approval answers picks them up again, so a run cannot reset its budget by pausing. A request that does not carry approval answers discards a parked budget, and the limits themselves are always read from the current configuration. Without a session, each request has its own budget, which is also how upstream behaves. An approved call that arrives after the budget is spent is not executed; it gets the result `"Error: Tool call was not executed: the request's function-invocation budget is spent."`.

### Concurrency, usage and streaming

The calls in one model response run concurrently by default. Their results are returned in the order the model emitted the calls either way; `allow_concurrent_invocation: false` matters for side effects, such as a tool that writes to a shared resource, or a model that emits "create the record, then update it" in one batch.

`usage_details` on the final response is the sum over every model call the loop made, not just the last one, so token accounting in middleware and in the OpenTelemetry metrics covers the whole run.

Streaming and local tools do not combine as you might expect. When the request has any executable tool, `get_streaming_response` runs the entire loop without streaming and then replays each resulting message as one update. You get the same messages, in order, but not token by token. Hosted tools and declaration-only tools do not trigger this, because the loop only takes over when there is something for it to execute.

### Using the loop without an agent

`FunctionInvokingChatClient::new(client)` is an ordinary `ChatClient`, so the loop can be used without an `Agent`: pass tools in `ChatOptions::with_tool` and call `get_response`. `with_config` sets the configuration and `with_function_middleware` the function middleware, which is what the agent builder does internally. This is useful in workflow executors that call a model directly.

## Approvals

### The approval round trip

A tool marked `ApprovalMode::AlwaysRequire` (with `with_approval_mode`, or `require_approval()` on a `ToolDefinition`) is never run without a human's answer. When the model calls it, the loop stops and returns a response whose last assistant message carries the original calls plus one `FunctionApprovalRequestContent` per call. `user_input_requests()` (on `AgentResponse`, `ChatResponse` and stream updates) collects them. Each request has an `id` and the `function_call` it is about; `create_response(true)` approves it and `create_response(false)` rejects it. Send the answers back as a user message on the same session and the loop carries on:

```rust
use agent_framework::prelude::*;
use serde_json::json;

fn delete_file() -> ToolDefinition {
    FunctionTool::new(
        "delete_file",
        "Permanently delete a file.",
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        }),
        |args| async move { Ok(json!({ "deleted": args["path"] })) },
    )
    .into_definition()
    .require_approval() // same as .with_approval_mode(ApprovalMode::AlwaysRequire)
}

/// Run until the agent stops asking, answering every request with `decide`.
async fn drive(
    agent: &Agent,
    prompt: &str,
    decide: impl Fn(&FunctionApprovalRequestContent) -> bool,
) -> Result<String> {
    let mut session = agent.create_session();
    let mut input = vec![Message::user(prompt)];
    for _ in 0..5 {
        let response = agent.run(input, Some(&mut session)).await?;
        let requests = response.user_input_requests();
        if requests.is_empty() {
            return Ok(response.text());
        }
        let mut answers = Vec::new();
        for request in requests {
            let approved = decide(request);
            println!(
                "  {} {}({:?}) [request id {}]",
                if approved { "approve" } else { "reject " },
                request.function_call.name,
                request.function_call.arguments,
                request.id
            );
            answers.push(Content::FunctionApprovalResponse(request.create_response(approved)));
        }
        input = vec![Message::with_contents(Role::user(), answers)];
    }
    Err(Error::other("still asking for approval after 5 rounds"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You manage files. Use the tools for file actions.")
        .tool(delete_file())
        .build();

    println!("approved:");
    println!("  -> {}", drive(&agent, "Delete scratch.txt", |_| true).await?);

    println!("rejected:");
    println!("  -> {}", drive(&agent, "Delete scratch.txt", |_| false).await?);
    Ok(())
}
```

```text
approved:
  approve delete_file(Some(Raw("{\"path\": \"scratch.txt\"}"))) [request id af-call-1c97ac64-734c-432f-bd64-0c75a509e339]
  -> Here is what the tool returned: {"deleted":"scratch.txt"}
rejected:
  reject  delete_file(Some(Raw("{\"path\": \"scratch.txt\"}"))) [request id af-call-eddfcce6-b3df-4e87-ad2e-3ecb755160be]
  -> Here is what the tool returned: Error: Tool call invocation was rejected by user.
```

An approved call runs through function middleware like any other, and its result is spliced into the conversation before the model is called again. The session matters: it holds the history that contains the pending calls, and the parked loop budget. Because the answers carry the function call they approve, the round trip can span processes, provided you persist the session in between (see the [sessions page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/)).

### Rejections, batches and repeats

A rejected call is not executed; the model receives the result string `"Error: Tool call invocation was rejected by user."`, the same text Python uses, and usually explains that it could not do the task. A rejection is an answer, not an error, so it does not count towards `max_consecutive_errors_per_request`.

Approval is decided per batch, not per call. If any call in a model response needs approval, every call in that response is deferred and returned as a request, including calls to tools that need none. In a test where the model asked for `free` (no approval) and `gated` (approval) together, both came back as requests; approving `free` and rejecting `gated` ran only `free`. This matches Python, and avoids running half a batch whose other half the human might refuse.

Each deferred call is stamped with an occurrence id (`af-call-<uuid>`), which is also the request's `id`. It ties an answer to exactly one pending call even when a provider reuses `call_id` values, so two approvals pending under the same `call_id` cannot be confused. Answers that have already been acted on are recognised as settled and never execute their call again, so replaying the same approval message, or a session history that still contains it, is harmless.

Approval also exists at other levels: `AsToolOptions::approval_mode` gates calls to a sub-agent, `with_request_info()` adds a review step to the sequential and concurrent [orchestrations](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/), and a hosted MCP connector has the service-side `McpApprovalMode` described under [hosted tools](#results-and-approval). The [shell tools](#shell-tools) require approval by default.

### Standing approvals with `ToolApprovalAgent`

Asking a human about every call gets tedious. `ToolApprovalAgent`, in `agent_framework::harness` behind the `experimental-harness` feature, wraps any agent and keeps approval state in the session. It approves calls that match an auto-approval rule without asking, records "always approve" answers as standing rules, and queues the remaining requests so that the caller is asked about one call at a time.

```rust
// Needs the `experimental-harness` feature.
use std::sync::Arc;

use agent_framework::harness::ToolApprovalAgent;
use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// A model that calls `deploy` for each user message and answers once it
/// sees the result.
#[derive(Clone)]
struct Deployer;

#[async_trait]
impl ChatClient for Deployer {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let last = messages.last().unwrap();
        if let Some(Content::FunctionResult(r)) = last.contents.first() {
            return Ok(ChatResponse::from_text(format!("deploy result: {:?}", r.result)));
        }
        let env = if last.text().contains("prod") { "production" } else { "staging" };
        let call = FunctionCallContent::new(
            format!("call-{}", messages.len()),
            "deploy",
            Some(FunctionArguments::Raw(json!({ "env": env }).to_string())),
        );
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let deploy = FunctionTool::new(
        "deploy",
        "Deploy the service.",
        json!({ "type": "object", "properties": { "env": { "type": "string" } } }),
        |args| async move { Ok(json!({ "deployed": args["env"] })) },
    )
    .into_definition()
    .require_approval();

    let inner = Agent::builder(Deployer).tool(deploy).build();
    let agent = ToolApprovalAgent::new(Arc::new(inner))
        // Heuristic rules approve without asking; this one trusts staging.
        .with_auto_approval_rule(|call: &FunctionCallContent| {
            call.parse_arguments().is_ok_and(|a| a.get("env") == Some(&json!("staging")))
        });

    let mut session = agent.create_session(); // required: the state lives here

    // 1. Staging: auto-approved, the caller never sees a request.
    let r = agent.run(vec![Message::user("deploy to staging")], Some(&mut session)).await?;
    println!("staging: {} (requests: {})", r.text(), r.user_input_requests().len());

    // 2. Production: the caller is asked. Answer "always approve this tool".
    let r = agent.run(vec![Message::user("deploy to prod")], Some(&mut session)).await?;
    let request = r.user_input_requests()[0].clone();
    println!("prod asks: {}({:?})", request.function_call.name, request.function_call.arguments);
    let answer = agent.always_approve_tool_response(&session, &request, Some("on-call said yes"))?;
    let r = agent
        .run(
            vec![Message::with_contents(Role::user(), vec![Content::FunctionApprovalResponse(answer)])],
            Some(&mut session),
        )
        .await?;
    println!("prod: {}", r.text());

    // 3. Production again: the standing rule approves it.
    let r = agent.run(vec![Message::user("deploy to prod again")], Some(&mut session)).await?;
    println!("prod again: {} (requests: {})", r.text(), r.user_input_requests().len());
    println!("rules: {:?}", agent.state(&session)?.rules);
    Ok(())
}
```

```text
staging: deploy result: Some(Object {"deployed": String("staging")}) (requests: 0)
prod asks: deploy(Some(Raw("{\"env\":\"production\"}")))
prod: deploy result: Some(Object {"deployed": String("production")})
prod again: deploy result: Some(Object {"deployed": String("production")}) (requests: 0)
rules: [ToolApprovalRule { tool_name: "deploy", arguments: None }]
```

`with_auto_approval_rule` accepts any `Fn(&FunctionCallContent) -> bool`, or a type implementing the async `ToolAutoApprovalRule` trait. `always_approve_tool_response` records a rule for every later call to the tool, and `always_approve_tool_with_arguments_response` one for calls with exactly the same arguments (compared as canonical JSON). `state(&session)` returns the `ToolApprovalState`: the `rules`, the requests still queued and the answers collected so far. State is kept under the key `"tool_approval"` (`with_source_id` changes it), and a run without a session is an error. When every request in a turn is auto-approved, the wrapper re-runs the inner agent itself, at most `with_max_auto_approval_iterations` times (default 40).

A rule that matches by name approves any tool with that name, including one added later for another purpose, so keep tool names unique or match on arguments. If an inner run fails after some approved calls ran, sending the last answer again re-sends the whole collected batch, and the calls that already ran run again. The harness is experimental and can change in any minor release.

## An agent as a tool

`agent.as_tool(options)` returns a `ToolDefinition` that runs the agent as a function tool taking a single string. The coordinating agent calls it like any other tool; the sub-agent's text is the result.

```rust
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// The coordinator's model: delegates the user's question to the first
/// tool it is offered, then relays the answer.
#[derive(Clone)]
struct Coordinator;

#[async_trait]
impl ChatClient for Coordinator {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        if let Some(Content::FunctionResult(r)) = messages.last().and_then(|m| m.contents.first()) {
            return Ok(ChatResponse::from_text(format!(
                "relayed: {}",
                r.result.clone().or(r.exception.clone().map(Into::into)).unwrap_or_default()
            )));
        }
        let tool = &options.tools[0];
        println!("  coordinator sees tool {} with schema {}", tool.name, tool.parameters);
        let arg = tool.parameters["required"][0].as_str().unwrap().to_string();
        let call = FunctionCallContent::new(
            "call-1",
            tool.name.clone(),
            Some(FunctionArguments::Raw(json!({ arg: messages.last().unwrap().text() }).to_string())),
        );
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// The specialist's model: repeats its system prompt and task, in two chunks
/// when streamed.
#[derive(Clone)]
struct Specialist;

impl Specialist {
    fn answer(messages: &[Message]) -> String {
        let system = messages.iter().find(|m| m.role == Role::system()).map(Message::text);
        format!("[system: {system:?}] answer to {:?}", messages.last().unwrap().text())
    }
}

#[async_trait]
impl ChatClient for Specialist {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(Self::answer(&messages)))
    }
    async fn get_streaming_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        let text = Self::answer(&messages);
        let (a, b) = text.split_at(text.len() / 2);
        let updates = [a, b].map(|part| {
            Ok(ChatResponseUpdate {
                contents: vec![Content::text(part)],
                role: Some(Role::assistant()),
                ..Default::default()
            })
        });
        Ok(Box::pin(futures::stream::iter(updates)))
    }
}

/// Adds the session's user to the instructions, so we can see which
/// session the specialist ran on.
struct WhoIsAsking;

#[async_trait]
impl ContextProvider for WhoIsAsking {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let user = ctx.session.as_ref().and_then(|s| s.state.get("user"));
        ctx.add_instructions(format!("user={user:?}"));
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let specialist = Agent::builder(Specialist)
        .name("Tax Expert") // derived tool names are sanitised
        .description("Answers tax questions.")
        .context_provider(Arc::new(WhoIsAsking))
        .build();

    // Defaults: name from the agent, argument "task", a fresh session per call.
    let chunks = Arc::new(Mutex::new(0usize));
    let seen = chunks.clone();
    let plain = specialist.as_tool(
        AsToolOptions::new().stream_callback(Arc::new(move |_update: &AgentResponseUpdate| {
            *seen.lock().unwrap() += 1;
        })),
    );
    // Everything set: own name and argument, the caller's session.
    let shared = specialist.as_tool(
        AsToolOptions::new()
            .name("ask_tax")
            .description("Ask the tax specialist.")
            .arg_name("question")
            .arg_description("The question, in full.")
            .propagate_session(true),
    );

    for (label, tool) in [("defaults", plain), ("propagate_session", shared)] {
        let coordinator = Agent::builder(Coordinator).tool(tool).build();
        let mut session = coordinator.create_session();
        session.state.insert("user", json!("ada"));
        let response = coordinator
            .run(vec![Message::user("Is a bicycle deductible?")], Some(&mut session))
            .await?;
        println!("{label}: {}", response.text());
    }
    println!("streamed updates seen by the callback: {}", chunks.lock().unwrap());
    Ok(())
}
```

```text
  coordinator sees tool Tax_Expert with schema {"additionalProperties":false,"properties":{"task":{"description":"Task for Tax_Expert","type":"string"}},"required":["task"],"type":"object"}
defaults: relayed: "[system: Some(\"user=None\")] answer to \"Is a bicycle deductible?\""
  coordinator sees tool ask_tax with schema {"additionalProperties":false,"properties":{"question":{"description":"The question, in full.","type":"string"}},"required":["question"],"type":"object"}
propagate_session: relayed: "[system: Some(\"user=Some(String(\\\"ada\\\"))\")] answer to \"Is a bicycle deductible?\""
streamed updates seen by the callback: 2
```

| `AsToolOptions` method | Default | Effect |
| --- | --- | --- |
| `name` | the agent's name, made into a valid identifier (`Tax Expert` becomes `Tax_Expert`), else its id | The tool name |
| `description` | the agent's description, else empty | The tool description |
| `arg_name` | `"task"` | The single string argument |
| `arg_description` | `"Task for <tool name>"` | Its description |
| `approval_mode` | `NeverRequire` | Gate calls to the sub-agent behind approval |
| `stream_callback` | none | Run the sub-agent with `run_stream` and call this with every update |
| `propagate_session` | `false` | Run the sub-agent on a child of the caller's session |

By default each call runs the sub-agent on a fresh session, so it remembers nothing between calls and sees nothing of the caller. With `propagate_session(true)` it runs on `AgentSession::child` of the caller's session: the same session id and a shared state bag (the example's context provider reads the caller's `user`), but its own service-side conversation id, so a parent's server-held conversation with a pending tool call does not leak into the sub-agent's requests. History is not shared: the child has no context providers of its own, so it gets a fresh in-memory history on each call and sees neither the caller's conversation nor its own earlier calls. Pass what the specialist needs in the task string.

Only the sub-agent's final text is returned; its own tool calls stay inside it. If the sub-agent ends its run with approval requests of its own, they cannot be answered from inside a tool call, and the tool fails with `"agent tool '<name>' ended its run with pending user-input requests, which cannot be satisfied from within a tool call"` (upstream raises `UserInputRequiredException`). Give a delegated agent tools that need no approval, or put the approval on the delegation itself with `approval_mode`. An error from the sub-agent is an ordinary tool error, so the coordinator's model sees `"tool execution failed"` and can carry on; for multi-agent designs where control should move rather than return, see the handoff orchestration on the [orchestrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/).

## Changing tools at run time

Three mechanisms change what an agent can call without rebuilding it: tools passed for one run, tool sources resolved at the start of each run, and (experimental) changes to the tool list in the middle of a run.

### Per-run tools and declaration-only tools

`AgentRunOptions::with_tool` and `with_tools` add tools for one call of `run_with_options` or `run_stream`. They are appended after the agent's own tools and any tools a context provider added; a per-run tool whose name is already taken is dropped.

A function tool with no executor is declaration-only. When the model calls it, the loop does not treat the call as unknown: it runs any local calls in the same batch, then ends the run and returns the declaration-only call to the caller with its `call_id` intact. This is how hosting front ends such as AG-UI expose tools that run in the browser (see the [hosting page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/)). The caller runs the tool and sends a `FunctionResultContent` with that `call_id` on the next request. There is no dedicated constructor; set `executor` to `None` on a definition, or build a `ToolDefinition` literal with `kind: ToolKind::Function`.

### Tool sources

A `ToolSource` is asked for its tools at the start of every run (`run`, `run_with_options` and `run_stream`), so the agent sees the current catalogue rather than the one it was built with. The MCP tool wrappers implement it, which is how an agent follows a server's `tools/list_changed` notifications (see the [MCP page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_mcp/)). The trait has two methods: `resolve_tools()` and `source_name()`, used in log messages.

```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// Calls the tool named in the user's message, then answers.
#[derive(Clone)]
struct CallsNamedTool;

#[async_trait]
impl ChatClient for CallsNamedTool {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let names: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
        let last = messages.last().unwrap();
        if last.role == Role::tool() {
            return Ok(ChatResponse::from_text(format!("tools {names:?}; got {:?}", last.contents)));
        }
        let wanted = last.text();
        let call = FunctionCallContent::new("call-1", wanted, Some(FunctionArguments::Raw("{}".into())));
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

fn local(name: &str) -> ToolDefinition {
    let label = name.to_string();
    FunctionTool::new(name, "A local tool.", json!({ "type": "object" }), move |_| {
        let label = label.clone();
        async move { Ok(json!(format!("{label} ran"))) }
    })
    .into_definition()
}

/// A catalogue that changes between runs, as an MCP server's might.
struct Catalogue {
    version: AtomicUsize,
}

#[async_trait]
impl ToolSource for Catalogue {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        let v = self.version.fetch_add(1, Ordering::SeqCst);
        // `clock` collides with the agent's own tool and is skipped (with a warning).
        Ok(vec![local(&format!("report_v{v}")), local("clock")])
    }
    fn source_name(&self) -> &str {
        "catalogue"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(CallsNamedTool)
        .tool(local("clock"))
        .tool_source(Arc::new(Catalogue { version: AtomicUsize::new(1) }))
        .build();

    // A ToolSource is resolved again on every run.
    println!("{}", agent.run_once("report_v1").await?.text());
    println!("{}", agent.run_once("report_v2").await?.text());

    // A tool for one run only.
    let options = AgentRunOptions::new().with_tool(local("export"));
    let r = agent.run_with_options(vec![Message::user("export")], None, options).await?;
    println!("{}", r.text());

    // A declaration-only tool: no executor, so its call comes back to the caller.
    let mut confirm = local("confirm_in_ui");
    confirm.executor = None;
    let options = AgentRunOptions::new().with_tool(confirm);
    let r = agent.run_with_options(vec![Message::user("confirm_in_ui")], None, options).await?;
    for content in r.messages.iter().flat_map(|m| &m.contents) {
        if let Content::FunctionCall(call) = content {
            println!("returned to caller: {} (call id {})", call.name, call.call_id);
        }
    }
    Ok(())
}
```

```text
tools ["clock", "report_v1"]; got [FunctionResult(FunctionResultContent { call_id: "call-1", result: Some(String("report_v1 ran")), exception: None })]
tools ["clock", "report_v2"]; got [FunctionResult(FunctionResultContent { call_id: "call-1", result: Some(String("report_v2 ran")), exception: None })]
tools ["clock", "export", "report_v3"]; got [FunctionResult(FunctionResultContent { call_id: "call-1", result: Some(String("export ran")), exception: None })]
returned to caller: confirm_in_ui (call id call-1)
```

Source tools come last, after the agent's, the context providers' and the per-run tools. A source tool whose name is taken is skipped with a `tracing` warning (the source's `clock` above), and the first source registered wins between sources. An error from `resolve_tools` fails the run, as upstream does when an MCP server cannot be reached. Because the method runs on every run, a source that talks to a remote server should connect lazily and cache its list until something invalidates it.

### Progressive tools (experimental)

With the `experimental-progressive-tools` feature, a run's tool list can change while the run is in progress. `FunctionInvocationContext::tools` is the run's `LiveToolList`, and `add_tools` and `remove_tools` on the context (or on the list) change it. The loop takes a snapshot of the list at the start of each model iteration, so a change applies from the next request on and never affects calls already requested.

```rust
// Needs the `experimental-progressive-tools` feature.
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// Prints the tools it can see, then calls each name in its script once,
/// skipping any it cannot see.
#[derive(Clone)]
struct Model {
    script: Vec<&'static str>,
}

#[async_trait]
impl ChatClient for Model {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        let visible: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
        let turn = messages.iter().filter(|m| m.role == Role::assistant()).count();
        println!("  turn {turn}: sees {visible:?}");
        match self.script.get(turn) {
            Some(name) if visible.contains(name) => {
                let call = FunctionCallContent::new(format!("c{turn}"), *name, Some(FunctionArguments::Raw("{}".into())));
                Ok(ChatResponse {
                    messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
                    finish_reason: Some(FinishReason::tool_calls()),
                    ..Default::default()
                })
            }
            _ => Ok(ChatResponse::from_text("done")),
        }
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

fn tool(name: &str) -> ToolDefinition {
    FunctionTool::new(name, "", json!({ "type": "object" }), |_| async { Ok(json!("ok")) }).into_definition()
}

/// After a successful `login`, add the account tools and drop `login`.
struct UnlockAfterLogin;

#[async_trait]
impl Middleware<FunctionInvocationContext> for UnlockAfterLogin {
    async fn process(
        &self,
        ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        let is_login = ctx.function_name == "login";
        let ctx = next.run(ctx).await?;
        if is_login && ctx.result.is_some() {
            ctx.add_tools([tool("balance"), tool("transfer")])?;
            ctx.remove_tools(["login"])?;
        }
        Ok(ctx)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(Model { script: vec!["login", "balance", "login"] })
        .tool(tool("login"))
        .function_middleware(Arc::new(UnlockAfterLogin))
        .build();
    agent.run_once("check my balance").await?;

    // Outside a run there is no live list to change.
    let detached = FunctionInvocationContext::new("login", json!({}));
    println!("{:?}", detached.add_tools([tool("x")]));

    // A duplicate name is refused and nothing is added.
    let live = LiveToolList::new(vec![tool("a")]);
    println!("{:?}", live.add_tools([tool("b"), tool("a")]));
    println!("{:?}", live.snapshot().iter().map(|t| t.name.clone()).collect::<Vec<_>>());
    Ok(())
}
```

```text
  turn 0: sees ["login"]
  turn 1: sees ["balance", "transfer"]
  turn 2: sees ["balance", "transfer"]
Err(Configuration("cannot add tools: this FunctionInvocationContext is not bound to a live agent run"))
Err(Configuration("cannot add tool 'a': a tool with that name already exists in this run"))
["a"]
```

This keeps the schema the model sees small (progressive disclosure), and supports one-shot tools that remove themselves after a successful call. `add_tools` validates the whole batch first and rejects a name already in the list, or repeated within the batch, leaving the list unchanged. On a context that is not part of a run (one you built yourself, or a direct `Tool::invoke`), both methods return `Error::Configuration`, as upstream raises `RuntimeError`. A tool can make the same changes from `invoke_in_context`. The changes last for the run only; the next run starts again from the agent's tools.

Because the in-flight batch is unaffected, removing a tool does not stop a model that asks for it three times in parallel: all three calls run. Use approval when a call must happen at most once. Without the feature, the loop still uses a live list internally, but the handle is not public.

## Hosted tools

### Constructors and settings

Hosted tools run on the provider's servers: you declare them, the service executes them, and the results come back folded into the response. They never pass through your process, so the tool loop, function middleware and `ApprovalMode` do not apply to them. Five constructors return marker definitions:

| Constructor | Tool name | `ToolKind` |
| --- | --- | --- |
| `hosted_web_search()` | `web_search` | `HostedWebSearch` |
| `hosted_code_interpreter()` | `code_interpreter` | `HostedCodeInterpreter` |
| `hosted_file_search(max_results)` | `file_search` | `HostedFileSearch { max_results }` |
| `hosted_image_generation()` | `image_generation` | `HostedImageGeneration` |
| `hosted_mcp(name, url, allowed_tools)` | `name` | `HostedMcp { url, allowed_tools }` |

Provider-specific settings are builder methods on `ToolDefinition` that write keys into its `parameters` object. Each provider's converter reads the keys it understands and ignores the rest:

| Method | Writes | Read by |
| --- | --- | --- |
| `description(text)` | the description | OpenAI Responses, as a hosted MCP server's `server_description` |
| `user_location(json)` | `user_location` | OpenAI Responses and Chat Completions (as an approximate location), Anthropic (verbatim) |
| `max_uses(n)` | `max_uses` | Anthropic web search only |
| `vector_store_ids(ids)` | `vector_store_ids` | OpenAI Responses and Foundry file search |
| `max_results(n)` | `max_results` | OpenAI Responses only, as a fallback; the argument to `hosted_file_search` wins |
| `file_ids(ids)` | `file_ids` | OpenAI Responses code interpreter, inside an `auto` container |
| `container(json)` | `container` | OpenAI Responses code interpreter; replaces the default container and `file_ids` |
| `headers(map)` | `headers` | OpenAI Responses and Foundry (verbatim); Anthropic reads only a lower-case `authorization` entry, as `authorization_token` |
| `mcp_approval_mode(mode)` | `approval_mode` | OpenAI Responses and Foundry, as `require_approval`; not Anthropic |
| `connection_id(id)`, `custom_connection(id, instance)` | `connection_id`, `custom_connection_id`, `instance_name` | Documented as Foundry Bing grounding settings, but no converter in %%version=agent-framework-rs%% reads them: Foundry uses the Responses API's plain `web_search` tool |

`hosted_image_generation()` passes every key in `parameters` through to the Responses API, so options such as `size` or `quality` can be set by writing to the public `parameters` field.

### What each provider sends

The converters are public, so the wire form can be checked without a network call:

```rust
use std::collections::HashMap;

use agent_framework::anthropic::convert::tools_to_anthropic;
use agent_framework::openai::convert::apply_options;
use agent_framework::openai::responses::tool_to_responses_spec;
use agent_framework::prelude::*;
use serde_json::{json, Map};

fn main() {
    let tools = vec![
        hosted_web_search()
            .user_location(json!({ "city": "Leeds", "country": "GB" }))
            .max_uses(3),
        hosted_code_interpreter().file_ids(vec!["file-abc".into()]),
        hosted_file_search(Some(5)).vector_store_ids(vec!["vs_123".into()]),
        hosted_image_generation(),
        hosted_mcp("docs", "https://mcp.example.com/mcp", Some(vec!["search_docs".into()]))
            .description("Product documentation")
            .headers(HashMap::from([("authorization".into(), "Bearer t0ken".into())]))
            .mcp_approval_mode(McpApprovalMode::PerTool {
                always: vec![],
                never: vec!["search_docs".into()],
            }),
    ];

    for tool in &tools {
        println!("{:<17} kind={:?} executable={}", tool.name, tool.kind, tool.is_executable());
    }

    println!("\nOpenAI Responses (also Azure OpenAI Responses and Foundry):");
    for tool in &tools {
        println!("  {}", tool_to_responses_spec(tool));
    }

    let (anthropic_tools, mcp_servers) = tools_to_anthropic(&tools);
    println!("\nAnthropic tools:       {}", json!(anthropic_tools));
    println!("Anthropic mcp_servers: {}", json!(mcp_servers));

    // Chat Completions: only web search survives, as a top-level field.
    let mut options = ChatOptions::new();
    options.tools = tools.clone();
    let mut body = Map::new();
    apply_options(&mut body, &options);
    println!("\nChat Completions body: {}", json!(body));
}
```

```text
web_search        kind=HostedWebSearch executable=false
code_interpreter  kind=HostedCodeInterpreter executable=false
file_search       kind=HostedFileSearch { max_results: Some(5) } executable=false
image_generation  kind=HostedImageGeneration executable=false
docs              kind=HostedMcp { url: "https://mcp.example.com/mcp", allowed_tools: Some(["search_docs"]) } executable=false

OpenAI Responses (also Azure OpenAI Responses and Foundry):
  {"type":"web_search","user_location":{"city":"Leeds","country":"GB","type":"approximate"}}
  {"container":{"file_ids":["file-abc"],"type":"auto"},"type":"code_interpreter"}
  {"max_num_results":5,"type":"file_search","vector_store_ids":["vs_123"]}
  {"type":"image_generation"}
  {"allowed_tools":["search_docs"],"headers":{"authorization":"Bearer t0ken"},"require_approval":{"never":{"tool_names":["search_docs"]}},"server_description":"Product documentation","server_label":"docs","server_url":"https://mcp.example.com/mcp","type":"mcp"}

Anthropic tools:       [{"max_uses":3,"name":"web_search","type":"web_search_20250305","user_location":{"city":"Leeds","country":"GB"}},{"name":"code_execution","type":"code_execution_20250825"}]
Anthropic mcp_servers: [{"authorization_token":"Bearer t0ken","name":"docs","tool_configuration":{"allowed_tools":["search_docs"]},"type":"url","url":"https://mcp.example.com/mcp"}]

Chat Completions body: {"web_search_options":{"user_location":{"approximate":{"city":"Leeds","country":"GB"},"type":"approximate"}}}
```

| Client | Web search | Code interpreter | File search | Image generation | MCP |
| --- | --- | --- | --- | --- | --- |
| `OpenAIChatClient` (Responses), `AzureOpenAIResponsesClient`, `FoundryChatClient` | `web_search` | `code_interpreter` | `file_search` | `image_generation` | `mcp` |
| `AnthropicClient` and its Bedrock, Vertex and Foundry variants | `web_search_20250305` | `code_execution_20250825` | dropped | dropped | `mcp_servers` entry |
| `GeminiChatClient` | `googleSearch` | `codeExecution` | dropped | dropped | dropped |
| `OpenAIChatCompletionClient`, `AzureOpenAIClient`, and the Chat Completions-shaped Ollama, Foundry Local and GitHub Copilot clients | top-level `web_search_options` | dropped | dropped | dropped | dropped |
| `MistralChatClient`, `BedrockChatClient` | dropped | dropped | dropped | dropped | dropped |

A dropped tool is skipped with a `tracing` warning rather than failing the request, so declaring one is safe but not a guarantee that the model has it. On Chat Completions, `web_search_options` only does something with the search-enabled OpenAI models, and an explicit `ToolMode::none()` removes it like any other tool. Whether a given deployment or model supports a hosted tool is the provider's business; the [providers page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_providers/) has the details per client.

### Results and approval

Responses-based clients return hosted activity as dedicated content variants: `SearchToolCall` and `SearchToolResult`, `CodeInterpreterToolCall` and `CodeInterpreterToolResult`, `ImageGenerationToolCall` and `ImageGenerationToolResult`, and `McpServerToolCall` and `McpServerToolResult`. Anthropic's server-side tools arrive as a `FunctionCall` and `FunctionResult` pair in the same response (with `HostedFile` items for files that code execution produced). The loop recognises a call whose result is already in the response as provider-executed and does not try to run it.

Because a hosted MCP call happens on the service, its approval gate is the service's too: `mcp_approval_mode(McpApprovalMode::Always)`, `Never`, or `PerTool { always, never }` (each list sent only when non-empty) goes out on the wire as `require_approval`. Restrict a hosted MCP server with `allowed_tools` rather than handing the model its whole surface; for Anthropic, an explicitly empty list exposes none of the server's tools rather than all of them.

## Images and tools

A tool's result is a `serde_json::Value`, which providers send to the model as text. A tool cannot return an image, audio or a file for the model to look at; Python's richer tool results have no counterpart here. Images go in through message content instead: `Content::Data` with `DataContent::from_bytes(bytes, media_type)` for inline bytes (sent as a base64 `data:` URI) or `Content::Uri` for a URL, as shown in the [comprehensive guide](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_comprehensive_guide/#messages-and-content). When a tool produces an image, return a reference and have the application send the bytes on the next turn:

```rust
use agent_framework::prelude::*;
use agent_framework::types::DataContent;
use serde_json::json;

/// The smallest valid PNG (1x1 grey), standing in for a rendered chart.
const PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x3a, 0x7e, 0x9b,
    0x55, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8, 0x0f, 0x00, 0x01,
    0x01, 0x01, 0x00, 0x1c, 0xb0, 0x8c, 0x99, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
    0x42, 0x60, 0x82,
];

#[tokio::main]
async fn main() -> Result<()> {
    // A tool's result is JSON. It can name an image, not carry one.
    let render_chart = FunctionTool::new(
        "render_chart",
        "Render the sales chart and return its file name.",
        json!({ "type": "object", "properties": { "quarter": { "type": "string" } } }),
        |_args| async move { Ok(json!({ "file": "chart-q3.png", "bytes": PNG.len() })) },
    )
    .into_definition();

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai%%")?)
        .instructions("Use render_chart for charts.")
        .tool(render_chart)
        .build();
    let mut session = agent.create_session();

    let first = agent.run(vec![Message::user("Chart Q3 sales.")], Some(&mut session)).await?;
    println!("{}", first.text());

    // The application loads the file and sends the pixels as message content.
    let look = Message::with_contents(
        Role::user(),
        vec![
            Content::text("Here is the chart. What does it show?"),
            Content::Data(DataContent::from_bytes(PNG, "image/png")),
        ],
    );
    let second = agent.run(vec![look], Some(&mut session)).await?;
    println!("{}", second.text());
    Ok(())
}
```

The second request reaches the provider as a user message with a text part and an `image_url` part holding the PNG as a data URI. A vision-capable model is needed for it to mean anything, and the session keeps the image in its history, so it is sent again on every later turn of that session.

## Shell tools

The `agent-framework-tools` crate, enabled with the `tools` feature and re-exported as `agent_framework::builtin_tools`, ports upstream's shell tool: `LocalShellTool` runs model-written commands on the host, and `DockerShellTool` runs them in a locked-down container. Both produce an ordinary function tool named `run_shell` with one string argument, `command`. Read the security notes below before enabling either: a shell tool gives the model arbitrary code execution.

The defences, from strongest to weakest, are the sandbox (`DockerShellTool`; the local tool has none), approval in the loop (on by default for both), and `ShellPolicy`, which is a convenience filter and not a defence. `agent_framework::prelude` exports `LocalShellTool` and `DockerShellTool`; everything else is in `agent_framework::builtin_tools::shell`.

### `LocalShellTool`

```rust
// Needs the `tools` feature.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework::builtin_tools::shell::{LocalShellTool, ShellExecutor, ShellPolicy};
use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use serde_json::json;

/// A model that runs each command in `commands` through `run_shell`, one per
/// turn, then summarises the last result.
#[derive(Clone)]
struct Operator {
    commands: Vec<&'static str>,
}

#[async_trait]
impl ChatClient for Operator {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let turn = messages.iter().filter(|m| m.role == Role::assistant()).count();
        let Some(command) = self.commands.get(turn) else {
            let last = messages.last().map(|m| format!("{:?}", m.contents)).unwrap_or_default();
            return Ok(ChatResponse::from_text(format!("last tool message: {last}")));
        };
        let call = FunctionCallContent::new(
            format!("sh-{turn}"),
            "run_shell",
            Some(FunctionArguments::Raw(json!({ "command": command }).to_string())),
        );
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        })
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let audit = Arc::new(Mutex::new(Vec::<String>::new()));
    let log = audit.clone();
    let workdir = std::env::temp_dir();

    let shell = LocalShellTool::builder()
        .workdir(&workdir)
        .clean_env(true) // commands see only the variables given here
        .env("PATH", "/usr/bin:/bin")
        .env("GREETING", "hello")
        .policy(ShellPolicy::new().with_denylist([r"\bsudo\b", r"\bcurl\b"])?)
        .timeout(Some(Duration::from_secs(5)))
        .max_output_bytes(4 * 1024)
        .on_command(move |command| log.lock().unwrap().push(command.to_string()))
        .build()?;
    println!("argv {:?}, approval {:?}", shell.argv(), shell.approval_mode());

    // ShellExecutor::run is the tool without the agent: no approval step.
    let r = shell.run("cd /; export SEEN=yes; echo $GREETING from $(pwd)", None).await?;
    println!("1. {:?}", r.format_for_model());
    let r = shell.run("echo SEEN=$SEEN in $(pwd)", None).await?;
    println!("2. {:?} (persistent: the variable survives, the cd does not)", r.format_for_model());
    let r = shell.run("sleep 3", Some(Duration::from_millis(200))).await?;
    println!("3. timed_out={} exit_code={}", r.timed_out, r.exit_code);
    println!("4. {:?}", shell.run("sudo ls", None).await.unwrap_err().to_string());
    shell.close().await?;

    // Through an agent: every call is an approval request first.
    let agent = Agent::builder(Operator { commands: vec!["sudo reboot", "uname -s"] })
        .tool(shell.as_function())
        .build();
    let mut session = agent.create_session();
    let mut input = vec![Message::user("check the OS")];
    loop {
        let response = agent.run(input, Some(&mut session)).await?;
        let requests = response.user_input_requests();
        if requests.is_empty() {
            println!("answer: {}", response.text());
            break;
        }
        let answers = requests
            .iter()
            .map(|r| {
                println!("approve? {:?}", r.function_call.arguments);
                Content::FunctionApprovalResponse(r.create_response(true))
            })
            .collect();
        input = vec![Message::with_contents(Role::user(), answers)];
    }
    println!("audit log: {:?}", audit.lock().unwrap());

    // Turning approval off needs an explicit acknowledgement.
    let refused = LocalShellTool::builder().approval_mode(ApprovalMode::NeverRequire).build();
    println!("NeverRequire alone: {}", refused.unwrap_err());
    Ok(())
}
```

```text
argv ["/bin/bash", "--noprofile", "--norc"], approval AlwaysRequire
1. "hello from /\nexit_code: 0"
2. "SEEN=yes in /tmp\nexit_code: 0" (persistent: the variable survives, the cd does not)
3. timed_out=true exit_code=124
4. "Command rejected by policy: matches denylist pattern: \\bsudo\\b"
approve? Some(Raw("{\"command\":\"sudo reboot\"}"))
approve? Some(Raw("{\"command\":\"uname -s\"}"))
answer: last tool message: [FunctionResult(FunctionResultContent { call_id: "sh-1", result: Some(String("Linux\nexit_code: 0")), exception: None })]
audit log: ["cd /; export SEEN=yes; echo $GREETING from $(pwd)", "echo SEEN=$SEEN in $(pwd)", "sleep 3", "uname -s"]
NeverRequire alone: invalid shell tool configuration: Setting approval mode to NeverRequire disables the only built-in security boundary of LocalShellTool. If you understand the risk (arbitrary commands run on the host with the agent's privileges; ShellPolicy is a UX pre-filter, not a defense), call acknowledge_unsafe(true). For untrusted input prefer a sandboxed executor such as DockerShellTool.
```

| `LocalShellToolBuilder` method | Default | Notes |
| --- | --- | --- |
| `mode(ShellMode)` | `Persistent` | One long-lived shell; `Stateless` starts a fresh process per command |
| `shell(spec)` | `AGENT_FRAMEWORK_SHELL`, then `pwsh` or `powershell` on Windows, `bash` or `sh` elsewhere | A command line (split with POSIX word rules) or an argv |
| `workdir(path)` | the process's directory | Made absolute when the tool is built |
| `confine_workdir(bool)` | `true` | Prefix each persistent-mode command with a `cd` back to the working directory |
| `env(k, v)`, `envs(iter)` | none | Added to the inherited environment |
| `clean_env(bool)` | `false` | Commands see only the variables you set, which keeps the host's secrets out of reach |
| `policy(ShellPolicy)` | empty | See below |
| `timeout(Option<Duration>)` | 30 s (`DEFAULT_TIMEOUT`) | `None` disables it |
| `max_output_bytes(n)` | 64 KiB (`DEFAULT_MAX_OUTPUT_BYTES`) | Per stream, head and tail kept; must be positive |
| `approval_mode(mode)` | `AlwaysRequire` | `NeverRequire` fails `build()` unless `acknowledge_unsafe(true)` is also set |
| `on_command(hook)` | none | Called with every command that passes the policy; a panic in it is caught and logged |

`build()` returns `Result<LocalShellTool, ShellError>`. The tool implements `ShellExecutor`, whose `run(command, timeout)` runs a command directly, without the agent and without approval; `as_function()` (or `as_function_with(name, description)`) gives the definition for the agent, with the configured approval mode. In persistent mode a `cd` or `export` in one call carries into the next, except that `confine_workdir` re-anchors the directory before each command. That is a convenience, not confinement: `cd /tmp && rm -rf .` in a single call still runs in `/tmp`. A command that exceeds its timeout has its whole process group killed and comes back as a result with `timed_out: true` (exit code 124 when the shell could not report one); in persistent mode the shell is restarted afterwards, so its state is lost. `ShellResult::format_for_model()` is the text the model receives: stdout, then `stderr: ...`, the truncation and timeout markers, and the exit code.

A persistent tool belongs to one conversation. Its shell keeps state between calls and serialises them, and clones share the shell, so build one tool per session and call `close()` when the session ends. Use `ShellMode::Stateless` if one instance must be shared between users. Choose a name for `as_function_with` that no other tool uses: approval rules match by name, and a shell tool named after an auto-approved tool would be approved with it.

### `ShellPolicy`

`ShellPolicy::new()` allows every non-empty command. `with_denylist(patterns)` and `with_allowlist(patterns)` add regular expressions (the `regex` crate, compiled case-insensitively, linear-time, so there is no match timeout as there is upstream); `with_deny_regex` and `with_allow_regex` take precompiled ones; `with_custom(|request| ...)` adds a final rule that may return `Some(ShellDecision)` to override the outcome. Evaluation stops at the first hit: an empty command is denied, then any deny-list match denies, then a command matching no allow-list pattern is denied, then the custom rule, then allow. `evaluate(&ShellRequest)` and `evaluate_command(&str)` run it directly.

A denied command is not an error: the model receives `"Command rejected by policy: <reason>"` as the result and can try something else. It ships with no patterns, deliberately, because a regular expression over a command's spelling cannot see what the shell will run after expansion (`r\m`, `${RM:=rm}`, `python -c`, base64 and absolute paths all slip past). Use it to give clear refusals for site-specific rules, not for safety.

The policy is checked when the command is about to run, which for the agent's function is after the human has approved it. In the example the operator was asked to approve `sudo reboot`, and only then did the policy refuse it. The crate's documentation describes the policy as applied before approval; in %%version=agent-framework-rs%% that holds only for direct `ShellExecutor::run` calls. If refused commands should never reach the approver, check them in function middleware as well.

### Telling the model about the shell

`ShellEnvironmentProvider` is a context provider that probes the shell once (family and version, working directory, which command-line tools answer `--version`) and adds a block of instructions to every run, so the model writes PowerShell in a PowerShell session and bash in a bash one. It registers no tools and works with any `ShellExecutor`:

```rust
// Needs the `tools` feature.
use std::sync::Arc;
use std::time::Duration;

use agent_framework::builtin_tools::shell::{
    default_instructions_formatter, truncate_text_head_tail, ShellEnvironmentProvider,
    ShellEnvironmentProviderOptions, ShellError, ShellExecutor, ShellFamily, ShellResult,
};
use agent_framework::prelude::*;
use async_trait::async_trait;

/// A stand-in for a remote or sandboxed backend: answers a few commands.
struct FakeSandbox;

#[async_trait]
impl ShellExecutor for FakeSandbox {
    async fn start(&self) -> std::result::Result<(), ShellError> {
        Ok(())
    }
    async fn close(&self) -> std::result::Result<(), ShellError> {
        Ok(())
    }
    async fn run(
        &self,
        command: &str,
        _timeout: Option<Duration>,
    ) -> std::result::Result<ShellResult, ShellError> {
        let (stdout, exit_code) = match command {
            c if c.starts_with("echo \"VERSION=") => ("VERSION=5.2.21\nCWD=/sandbox".to_string(), 0),
            "git --version" => ("git version 2.47.0".to_string(), 0),
            _ => (String::new(), 127),
        };
        Ok(ShellResult { stdout, exit_code, ..Default::default() })
    }
    fn shell_family(&self) -> Option<ShellFamily> {
        Some(ShellFamily::Posix)
    }
    fn os_description(&self) -> Option<String> {
        Some("a Linux microVM".into())
    }
}

/// Repeats the system prompt it receives.
#[derive(Clone)]
struct ShowSystem;

#[async_trait]
impl ChatClient for ShowSystem {
    async fn get_response(&self, messages: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
        let system = messages.iter().find(|m| m.role == Role::system()).map(Message::text);
        Ok(ChatResponse::from_text(system.unwrap_or_default()))
    }
    async fn get_streaming_response(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = ShellEnvironmentProviderOptions {
        probe_tools: vec!["git".into(), "cargo".into()],
        ..Default::default()
    }
    .with_instructions_formatter(|snapshot| {
        format!("{}\nNever run destructive commands.", default_instructions_formatter(snapshot))
    });
    let provider = Arc::new(ShellEnvironmentProvider::new(Arc::new(FakeSandbox), Some(options)));

    let agent = Agent::builder(ShowSystem)
        .instructions("You are an operator.")
        .context_provider(provider.clone())
        .build();
    println!("{}", agent.run_once("hi").await?.text());
    println!("\nsnapshot: {:?}", provider.current_snapshot().map(|s| s.tool_versions));

    // Results convert to the hosted-shell content types.
    let result = ShellResult { stdout: "ok".into(), ..Default::default() };
    println!("\n{:?}", result.to_tool_result("call-1", Some(65536)));

    // The head/tail truncation every executor applies.
    let (text, cut) = truncate_text_head_tail(&"x".repeat(100), 20);
    println!("{text:?} truncated={cut}");
    Ok(())
}
```

```text
You are an operator.
## Shell environment
You are operating a POSIX shell 5.2.21 session on a Linux microVM.
Use POSIX shell idioms (bash/sh).
- Set environment variables for the next command with `export NAME=value`.
- Reference environment variables as `$NAME` or `${NAME}`.
- Paths use `/` separators.
Working directory: /sandbox
Available CLIs: git (git version 2.47.0)
Not installed: cargo
Never run destructive commands.
```

| `ShellEnvironmentProviderOptions` field | Default |
| --- | --- |
| `probe_tools` | `git`, `node`, `python`, `docker`; names that are not plain identifiers are skipped, since they are spliced into a command |
| `override_family` | none: the executor's `shell_family()`, then the platform default |
| `probe_timeout` | 5 seconds per probe command |
| `instructions_formatter` | `default_instructions_formatter`; set with `with_instructions_formatter` |

The probe runs on the first agent run and is cached; `refresh()` probes again and `current_snapshot()` returns the cached `ShellEnvironmentSnapshot`. Probe commands go straight to the executor's `run`, so they bypass approval but not the policy: with a strict allow-list, some probes are refused and recorded as missing. A probe that fails with an expected error is recorded as a missing value; `ShellError::Other` from a custom executor propagates and fails the run, so a bug is not hidden.

### `DockerShellTool`

`DockerShellTool` drives the `docker` CLI (or `podman`, through `docker_binary`). In persistent mode it starts one container (`sleep infinity`) and runs a shell in it with `docker exec`; in stateless mode each command gets a fresh `docker run --rm`. Building the tool starts nothing; the container starts on the first command or `start()`, and `close()` removes it.

```rust
// Needs the `tools` feature. Building the tool does not need Docker.
use agent_framework::builtin_tools::shell::{is_docker_available, DockerShellTool, ShellMode};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let docker = DockerShellTool::builder()
        .image("alpine:3.20")
        .shell("sh") // alpine has no bash
        .host_workdir(std::env::current_dir()?) // mounted read-only at /workspace
        .build()?;
    println!("persistent container: {}", docker.run_argv().join(" "));
    println!("exec: {}", docker.exec_argv(true).join(" "));

    let stateless = DockerShellTool::builder().mode(ShellMode::Stateless).build()?;
    println!("stateless: {}", stateless.stateless_argv("af-shell-demo", "ls").join(" "));

    // With the isolation defaults kept, approval can be switched off...
    let unattended = DockerShellTool::builder().approval_mode(ApprovalMode::NeverRequire).build();
    println!("NeverRequire, defaults kept: ok={}", unattended.is_ok());

    // ...but not once they are weakened, unless acknowledged.
    let weakened = DockerShellTool::builder()
        .approval_mode(ApprovalMode::NeverRequire)
        .network("bridge")
        .user("0:0")
        .build();
    println!("NeverRequire, weakened: {}", weakened.unwrap_err());

    // Flags that would undo the defaults are refused outright.
    let sneaky = DockerShellTool::builder().extra_run_args(["--label=team", "-itv/:/host"]).build();
    println!("extra_run_args: {}", sneaky.unwrap_err());

    println!("docker available: {}", is_docker_available("docker").await);
    if is_docker_available("docker").await {
        use agent_framework::builtin_tools::shell::ShellExecutor;
        println!("{}", docker.run("uname -a", None).await?.format_for_model());
        docker.close().await?; // removes the container
    }
    Ok(())
}
```

```text
persistent container: docker run -d --rm --name af-shell-03386c8aabcc --user 65534:65534 --network none --memory 512m --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,size=64m --workdir /workspace --read-only -v /home/me/project:/workspace:ro alpine:3.20 sleep infinity
exec: docker exec -i af-shell-03386c8aabcc sh
stateless: docker run --rm -i --name af-shell-demo --user 65534:65534 --network none --memory 512m --pids-limit 256 --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,size=64m --workdir /workspace --read-only mcr.microsoft.com/azurelinux/base/core:3.0 bash -c ls
NeverRequire, defaults kept: ok=true
NeverRequire, weakened: invalid shell tool configuration: Setting approval mode to NeverRequire relies on the container as the security boundary, but this configuration weakens it (the container runs as root; the container has network access). Restore the isolation defaults, keep approval on, or call acknowledge_unsafe(true) if unapproved model commands with these privileges are intended.
extra_run_args: invalid shell tool configuration: extra_run_args contains flags that would dismantle DockerShellTool's isolation defaults: ["-itv/:/host"]. Use the dedicated builder methods (network, host_workdir, mount_readonly, read_only_root, memory, pids_limit, user) instead. Every token is checked, so pass a value that starts with '-' attached: --flag=value.
docker available: false
```

| `DockerShellToolBuilder` method | Default |
| --- | --- |
| `image` | `mcr.microsoft.com/azurelinux/base/core:3.0` (`DEFAULT_IMAGE`); it must contain the shell, and `sleep` for persistent mode |
| `shell` | `bash`; use `sh` for images without bash |
| `mode` | `Persistent` |
| `container_name` | generated (`af-shell-...`) |
| `host_workdir`, `mount_readonly` | no mount; read-only when set |
| `workdir` | `/workspace` |
| `network` | `none` |
| `memory` | `512m`; `0` or a non-size removes the limit |
| `pids_limit` | 256; `0` is unlimited |
| `user` | `65534:65534` (nobody) |
| `read_only_root` | `true` |
| `extra_run_args` | none; flags that would undo a default (`--privileged`, `-v`, `--network`, `--cap-add`, `--user`, ...) are rejected, including clustered short forms |
| `env` | none; values appear on the `docker` command line, so not for secrets |
| `policy`, `timeout`, `max_output_bytes`, `on_command` | as for the local tool |
| `approval_mode`, `acknowledge_unsafe` | `AlwaysRequire`; `NeverRequire` is allowed without acknowledgement only while the isolation defaults are kept |
| `docker_binary` | `docker` |

Every container also gets `--cap-drop ALL`, `--security-opt no-new-privileges` and a 64 MB `/tmp` tmpfs. `run_argv()`, `exec_argv(interactive)` and `stateless_argv(name, command)` return the exact command lines, which is useful in tests, and `is_docker_available(binary)` checks that the CLI exists and its daemon answers. After a timeout or a cancelled command in persistent mode, the tool kills every process in the container except its init. The same single-session ownership applies as for the local tool.

### Custom executors and the lower-level pieces

`ShellExecutor` is the backend trait both tools implement: `start`, `close` and `run(command, timeout)`, plus optional `shell_family()` and `os_description()` for the environment provider. Implement it for a microVM, a remote host or a WASI runtime; the `FakeSandbox` above is a minimal example. Implementations must enforce the timeout themselves and must not leak processes when it fires or when the future is dropped. Note that the `run_shell` function is built privately inside the two tools, so a custom executor gets the provider but not `as_function`; wrap it in a `FunctionTool` yourself.

`ShellError` has five variants: `Rejected` (policy), `Execution` (the shell could not start or the session broke), `Timeout` (only from custom executors; the built-in ones report a timed-out result instead), `Config` (from `build()`) and `Other`. It converts into `Error::Tool`. `ShellResult` has `stdout`, `stderr`, `exit_code` (negative for a signal), `duration`, `truncated` and `timed_out`; `to_command_output()` and `to_tool_result(call_id, max_output_length)` convert it to the hosted-shell content types `ShellCommandOutputContent` and `ShellToolResultContent`. The rest of the module is plumbing that is public for reuse: `ShellSession` (the persistent-shell protocol, with sentinel-delimited output), `resolve_shell` and `ShellSpec`, `is_powershell`, and `truncate_head_tail` and `truncate_text_head_tail`, which keep the first and last halves of an over-long output around a `[... truncated N bytes ...]` marker.

Upstream marks its shell function with `kind="shell"` so that the OpenAI Responses client declares it as a native local shell; in this port it is an ordinary function tool, which works with every client. Process-tree termination signals the process group rather than walking descendants, and on Windows a persistent session restarts its shell on timeout instead of sending `CTRL_BREAK`.

## Differences from Python and .NET

- **Tool definitions.** Python's `@tool` decorator reads a function signature; here `FunctionTool::typed` reads a `Deserialize + JsonSchema` struct, or `FunctionTool::new` takes a hand-written schema. There is no result parser hook (`result_parser`, `SKIP_PARSING`): return the `Value` you want the model to see.
- **Context access.** A tool reads the session and metadata by implementing `Tool::invoke_in_context`, not by declaring a context parameter.
- **Configuration.** The loop's configuration lives on the agent builder (`function_invocation_config`) rather than on the chat client; there is no per-run override.
- **Rich results.** Tool results are JSON only. Images and files reach the model through message content.
- **Invocation limits.** `max_invocations(0)` is accepted and blocks the tool, where Python rejects it at construction.
- **Hosted tools.** These are markers only, as in Python; `connection_id` and `custom_connection` are written but not read by any converter.
- **Shell tool.** The shell function is a plain function tool, not a `kind="shell"` tool.
- **Not ported.** Agents exposed as MCP servers (`as_mcp_server()`) are not ported; the hosting crate serves agents over A2A and OpenAI-compatible HTTP instead (see the [hosting page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/)).

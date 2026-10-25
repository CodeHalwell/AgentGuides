---
title: "Microsoft Agent Framework (Rust) — Workflows"
description: "The graph workflow engine in agent-framework-rs: function and custom executors, edges, conditions, switch-case, fan-out, fan-in and multi-selection, loops and the superstep limit, validation, shared state, outputs, events and streaming, human-in-the-loop requests, checkpointing, sub-workflows, visualisation, agents as nodes and workflows as agents."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Workflows"
  order: 8
---

The `workflow` module is a graph engine in the style of the Python `WorkflowBuilder`. A workflow is a set of executors joined by edges. It runs in supersteps (the Pregel or bulk-synchronous model): every message sent during a superstep is delivered at the start of the next one, and the executors scheduled in one superstep run concurrently. A run ends when no messages remain, pauses when an executor is waiting for outside input, and fails when an executor returns an error or the run exceeds its superstep limit.

This page covers the engine itself. The prebuilt multi-agent patterns built on it (sequential, concurrent, group chat, handoff and Magentic) are on the [orchestrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/).

Everything here is in the stable surface of `agent-framework` %%version=agent-framework-rs%%; no `experimental-*` feature is needed. The examples were compiled against that release. Those built from function executors run offline; the ones with agents were run against a local stand-in for the OpenAI API, so their printed replies are not real model output.

Most types are in the prelude (`use agent_framework::prelude::*`). The rest are imported from `agent_framework::workflow`: `FunctionExecutor`, `AgentExecutor`, `AgentApprovalExecutor`, `Case`, `Default`, `EdgeGroup`, the `wrap_*` helpers, `RequestResponse`, `PendingRequest`, `get_checkpoint_summary`, `validate_workflow_graph`, `ValidationType` and `WorkflowViz`.

## Messages are JSON values

The biggest difference from the Python and .NET engines is typing. There, an executor declares typed handlers, a message is routed to the handler whose input type matches, and the builder checks that connected executors agree on types. Here there are no typed handlers: every message, output and request payload is a `serde_json::Value`. An executor receives a `Value`, decides what it means, and sends `Value`s on. Validation therefore checks the shape of the graph but not the types flowing along it, and a mismatch shows up at run time.

In return, nothing needs registering for serialisation: checkpoints are plain JSON, and any `Serialize` type can be sent with `serde_json::to_value`. The usual pattern is to keep your own structs at the edges of each executor, deserialise on the way in and serialise on the way out. A short generic helper gives you executors with typed inputs, and a wrong-shaped payload fails the run with an error that names the executor:

```rust
use std::future::Future;
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Order {
    sku: String,
    quantity: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Priced {
    sku: String,
    total_pence: u64,
}

/// A function executor whose handler takes a typed payload. The message is
/// deserialised before the handler runs; a payload of the wrong shape fails
/// the run with an error naming the executor.
fn typed<T, F, Fut>(id: &'static str, handler: F) -> Arc<dyn Executor>
where
    T: DeserializeOwned + Send + 'static,
    F: Fn(T, WorkflowContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let handler = Arc::new(handler);
    Arc::new(FunctionExecutor::new(id, move |message, ctx| {
        let handler = handler.clone();
        async move {
            let input: T = serde_json::from_value(message)
                .map_err(|e| Error::Workflow(format!("{id}: unexpected payload: {e}")))?;
            handler(input, ctx).await
        }
    }))
}

fn pipeline() -> Result<Workflow> {
    let price = typed("price", |order: Order, ctx| async move {
        let priced = Priced { sku: order.sku, total_pence: 250 * order.quantity as u64 };
        ctx.send_message(serde_json::to_value(priced)?).await?;
        Ok(())
    });
    let invoice = typed("invoice", |priced: Priced, ctx| async move {
        ctx.yield_output(json!(format!("{}: £{:.2}", priced.sku, priced.total_pence as f64 / 100.0)))
            .await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(price)
        .add_executor(invoice)
        .set_start("price")
        .add_edge("price", "invoice")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let workflow = pipeline()?;
    let run = workflow.run(json!({ "sku": "TEA-01", "quantity": 3 })).await?;
    println!("{}", run.last_output().unwrap());

    match workflow.run(json!({ "sku": "TEA-01" })).await {
        Ok(_) => unreachable!(),
        Err(e) => println!("rejected: {e}"),
    }
    Ok(())
}
```

```text
"TEA-01: £7.50"
rejected: workflow error: price: unexpected payload: missing field `quantity`
```

The helper is ordinary user code, not a library API. Routing by message type, which Python does with several `@handler` methods on one executor, becomes a `match` on the payload inside one `execute`.

Agent-based executors and the orchestrations exchange conversations, serialised as a JSON array of `Message`. They accept a plain string as input (turned into one user message), a single message object, or an array of messages.

## Executors and edges

An executor is anything that implements the `Executor` trait. There are two ways to write one: a closure wrapped in `FunctionExecutor`, or your own type.

### Function executors

`FunctionExecutor::new(id, |message, ctx| async move { ... })` is the Rust counterpart of Python's `@executor` decorator. The closure receives the message and an owned `WorkflowContext`, which is a cheap handle you may hold across `.await` points, and returns `Result<()>`. Returning `Err` fails the run.

`WorkflowBuilder` registers executors with `add_executor`, picks the entry point with `set_start`, adds edges, and validates the graph in `build()`:

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
        .name("shout-and-count")
        .description("Upper-cases a sentence and counts its words.")
        .add_executor(Arc::new(upper))
        .add_executor(Arc::new(count))
        .set_start("upper")
        .add_edge("upper", "count")
        .build()?;

    println!("{:?} starts at {}", workflow.name(), workflow.start_executor_id());

    let run = workflow.run("graphs all the way down").await?;
    println!("state:  {:?}", run.state());
    println!("output: {}", run.last_output().unwrap());
    println!("events: {}", run.events().len());

    // The same workflow can be run again; each run is independent.
    let again = workflow.run("twice").await?;
    println!("second: {}", again.last_output().unwrap());
    Ok(())
}
```

```text
Some("shout-and-count") starts at upper
state:  Idle
output: {"text":"GRAPHS ALL THE WAY DOWN","words":5}
events: 12
second: {"text":"TWICE","words":1}
```

A built `Workflow` is cheap to clone (it is an `Arc` around the graph) and can be run any number of times, including concurrently. Each run gets fresh shared state and its own message queue; the executors themselves are shared between runs, so anything an executor keeps in its own fields carries over from one run to the next.

### Custom executor types

Implement `Executor` yourself when an executor needs configuration, owns a client or a cache, or keeps state that should survive a checkpoint. The trait has two required methods and two optional ones:

| Method | Purpose |
| --- | --- |
| `fn id(&self) -> &str` | The executor's id, unique within the workflow; edges refer to it |
| `async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()>` | Handle one message |
| `async fn snapshot_state(&self) -> Option<Value>` | Return state to store in each checkpoint; the default `None` means stateless |
| `async fn restore_state(&self, state: Value) -> Result<()>` | Put back what `snapshot_state` returned, when resuming from a checkpoint |

The trait is `async_trait`-based, so add the `async-trait` crate. `execute` takes `&self`: keep mutable state behind a `Mutex` or atomics. The engine never runs two deliveries to the same executor at once (see [concurrency](#concurrency-within-a-superstep)), but separate runs of the same workflow share the executor and can overlap.

```rust
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Forwards each event id the first time it is seen and drops repeats. The set
/// of seen ids is executor state, so it is snapshotted into checkpoints.
struct Dedupe {
    id: String,
    seen: Mutex<BTreeSet<String>>,
}

#[async_trait]
impl Executor for Dedupe {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        let event_id = message["id"].as_str().unwrap_or_default().to_string();
        let fresh = self.seen.lock().unwrap().insert(event_id);
        if fresh {
            ctx.yield_output(message).await?;
        }
        Ok(())
    }

    async fn snapshot_state(&self) -> Option<Value> {
        Some(json!(self.seen.lock().unwrap().iter().collect::<Vec<_>>()))
    }

    async fn restore_state(&self, state: Value) -> Result<()> {
        let ids: Vec<String> = serde_json::from_value(state)?;
        *self.seen.lock().unwrap() = ids.into_iter().collect();
        Ok(())
    }
}

fn build(storage: Arc<dyn CheckpointStorage>) -> Result<Workflow> {
    WorkflowBuilder::new()
        .add_executor(Arc::new(Dedupe { id: "dedupe".into(), seen: Mutex::new(BTreeSet::new()) }))
        .set_start("dedupe")
        .with_checkpointing(storage)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let workflow = build(storage.clone())?;

    // Executor state lives in the executor, so it carries across runs.
    let mut latest = None;
    for id in ["a", "b", "a"] {
        let before = storage.list(None).await?.len();
        let run = workflow.run(json!({ "id": id })).await?;
        println!("{id}: {} output(s)", run.outputs().len());
        // Each run writes one checkpoint; keep the newest by content, since
        // several runs can share a millisecond timestamp.
        assert_eq!(storage.list(None).await?.len(), before + 1);
        latest = storage.list(None).await?.into_iter().max_by_key(|cp| cp.executor_states["dedupe"].as_array().map_or(0, |a| a.len()));
    }
    let latest = latest.unwrap();

    // A fresh executor starts empty, until a checkpoint restores it.
    println!("snapshot: {}", latest.executor_states["dedupe"]);
    let restarted = build(storage.clone())?;
    restarted.run_from_checkpoint(&latest.checkpoint_id, storage).await?;
    let run = restarted.run(json!({ "id": "b" })).await?;
    println!("b after restore: {} output(s)", run.outputs().len());
    Ok(())
}
```

```text
a: 1 output(s)
b: 1 output(s)
a: 0 output(s)
snapshot: ["a","b"]
b after restore: 0 output(s)
```

A `FunctionExecutor` can also capture state (an `Arc<Mutex<_>>` moved into the closure), but it always returns `None` from `snapshot_state`, so that state is lost on a restore. Keep checkpoint-worthy state either in a custom executor or in [shared state](#shared-state).

### The workflow context

Inside an executor, the `WorkflowContext` is how it acts on the run. None of its effects take place immediately: messages, outputs, events and requests are collected while the executor runs and applied when it returns successfully. If `execute` returns `Err`, none of them happen.

| Method | Effect |
| --- | --- |
| `send_message(value)` | Send along every outgoing edge of this executor; edge conditions decide which targets receive it |
| `send_to(id, value)` | Send to one executor by id, ignoring edges and conditions (see the note below) |
| `yield_output(value)` | Emit a workflow output (see [outputs](#outputs)) |
| `request_info(value)` | Ask the caller for input; the answer comes back to this executor (see [human in the loop](#human-in-the-loop)) |
| `add_event(event)` | Put an event on the stream, normally `WorkflowEvent::Custom(value)` |
| `shared_state()` | The run's [shared state](#shared-state) |
| `executor_id()` | This executor's id |
| `source_executor_ids()` | Who sent the current message: one id, all the sources for a fan-in delivery, or `"__start__"` for the workflow input |

`send_to` needs no edge: any registered executor can be addressed, and a target that does not exist fails the run with `executor 'a' sent a message to unknown target executor 'bb'`. Because it skips routing, it also skips fan-in barriers, which is a trap described under [fan-in](#fan-in). Edges still matter for validation, which requires every executor to be reachable from the start along declared edges.

### Builder reference

| `WorkflowBuilder` method | Effect |
| --- | --- |
| `new()` | An empty builder (also `Default`) |
| `add_executor(Arc<dyn Executor>)` | Register an executor under its `id()`. Registering a second executor with the same id replaces the first without warning |
| `set_start(id)` | The executor that receives the workflow input; required |
| `add_edge(a, b)` | Always deliver from `a` to `b` |
| `add_conditional_edge(a, b, predicate)` / `add_conditional_edge_async` | Deliver when the predicate holds ([conditions](#conditional-edges)) |
| `add_switch(a, cases, default)` | First matching case wins ([switch-case](#switch-case)) |
| `add_fan_out(a, targets)` | Deliver to every target ([fan-out](#fan-out-and-multi-selection)) |
| `add_multi_selection(a, targets, selection)` | A function picks a subset of targets per message |
| `add_fan_in(sources, b)` | Barrier: deliver once every source has sent ([fan-in](#fan-in)) |
| `add_chain(ids)` | Consecutive `add_edge` calls; takes an iterator of `String` |
| `set_max_iterations(n)` | Superstep limit, default 100; values below 1 become 1 ([loops](#loops-and-the-superstep-limit)) |
| `name(s)`, `description(s)` | Metadata, returned by `Workflow::name()` and `description()` |
| `with_checkpointing(storage)` | Save a checkpoint after every superstep ([checkpointing](#checkpointing)) |
| `output_from(ids)`, `intermediate_output_from(ids)` | Which executors' yields count as outputs ([outputs](#outputs)) |
| `build()` | [Validate](#validation) and return a `Workflow` |

Methods that take several ids accept any iterator: `vec!["a".to_string()]`, `["a", "b"].map(String::from)`, or (for `output_from`) plain `["a"]`.

### Running a workflow

`workflow.run(input).await` drives the run until it finishes or pauses and returns a `WorkflowRun`. The input is anything that converts into a `Value`: a `&str`, a `json!` value, a number. `run_stream(input)` does the same while streaming events ([below](#streaming-events)), and `run_from_checkpoint` resumes a saved run.

| `Workflow` method | Returns |
| --- | --- |
| `run(input)` | `Result<WorkflowRun>`: `Err` if any executor or routing step failed |
| `run_stream(input)` | `WorkflowRunStream`: a `Stream` of events, then `into_run()` |
| `run_from_checkpoint(id, storage)` / `run_from_checkpoint_unchecked` | Resume from a checkpoint ([checkpointing](#checkpointing)) |
| `id()` | A random UUID assigned at `build()`; checkpoints are filed under it |
| `name()`, `description()`, `start_executor_id()`, `graph_signature()` | Metadata |
| `viz()` | Mermaid and DOT rendering ([visualisation](#visualisation)) |
| `as_agent(name)` | Wrap as an agent (from `WorkflowAgentExt`, see [below](#a-workflow-as-an-agent)) |

| `WorkflowRun` method | Returns |
| --- | --- |
| `state()` | The `WorkflowRunState` |
| `outputs()`, `last_output()` | Every terminal output in order, or the last one |
| `events()` | Every event so far, in order |
| `pending_requests()` | Open requests, ordered by request id |
| `send_response(id, value)`, `send_responses(map)` | Answer requests and continue the run |
| `shared_state()` | The run's shared state, still readable after it finishes |

`WorkflowRunState` has the same values as Python's enum (and serialises to the same `SCREAMING_SNAKE_CASE` strings): `Started`, `InProgress`, `InProgressPendingRequests` (still working, with requests already raised), `Idle` (finished), `IdleWithPendingRequests` (paused, resumable), `Failed` and `Cancelled`. A finished run is `Idle` or `IdleWithPendingRequests`; a failed one returns `Err` from `run`, so you see `Failed` only through events or a run that failed while resuming. `Cancelled` exists for parity but nothing in this release produces it: runs cannot be cancelled.

## Routing: conditions, switch-case and fan-out

An executor's `send_message` is offered to every edge group whose source is that executor. Each group decides which of its targets receive the message, and the results are combined, so an executor can have plain edges, conditional edges and a switch all at once.

| Edge group | Who receives a message |
| --- | --- |
| `add_edge` | The target, always |
| `add_conditional_edge` | The target, if the predicate is true |
| `add_switch` | The first case whose predicate is true, else the default |
| `add_fan_out` | Every target |
| `add_multi_selection` | Whichever targets the selection function returns |
| `add_fan_in` | The target, once every source has delivered |

### Conditional edges

A predicate receives `&Value` and returns either `bool` or `Result<bool>` (both satisfy the `IntoConditionResult` trait). Every conditional edge whose predicate holds fires, so several targets can receive the same message; use a switch when you want exactly one. Returning `Err` fails the run rather than counting as "no": a predicate that cannot read its input should not quietly send the message nowhere. `add_conditional_edge_async` takes a closure returning a future, for a check that has to await something; the predicate is evaluated when the message is routed, at the start of the next superstep.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Ticket {
    text: String,
    priority: u8,
}

fn team(name: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(name, move |message, ctx| async move {
        ctx.yield_output(json!({ "team": name, "ticket": message["text"] })).await?;
        Ok(())
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let triage = FunctionExecutor::new("triage", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(triage))
        .add_executor(team("billing"))
        .add_executor(team("on_call"))
        .add_executor(team("audit"))
        .set_start("triage")
        // Plain bool predicate.
        .add_conditional_edge("triage", "billing", |msg: &Value| {
            msg["text"].as_str().is_some_and(|t| t.contains("invoice"))
        })
        // Fallible predicate: a payload that does not parse fails the run
        // instead of quietly counting as "no".
        .add_conditional_edge("triage", "on_call", |msg: &Value| -> Result<bool> {
            let ticket: Ticket = serde_json::from_value(msg.clone())?;
            Ok(ticket.priority >= 8 || ticket.text.contains("outage"))
        })
        // Async predicate, for a check that has to await something.
        .add_conditional_edge_async("triage", "audit", |msg: &Value| {
            let priority = msg["priority"].as_u64().unwrap_or(0);
            async move {
                tokio::task::yield_now().await;
                priority >= 5
            }
        })
        .build()?;

    let run = workflow.run(json!({ "text": "invoice outage", "priority": 9 })).await?;
    println!("every matching edge fired:\n{}", run.outputs().iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n"));

    match workflow.run(json!({ "text": "no priority field" })).await {
        Ok(_) => unreachable!(),
        Err(e) => println!("predicate error: {e}"),
    }
    Ok(())
}
```

```text
every matching edge fired:
{"team":"audit","ticket":"invoice outage"}
{"team":"billing","ticket":"invoice outage"}
{"team":"on_call","ticket":"invoice outage"}
predicate error: json error: missing field `priority`
```

The outputs come out in executor-id order, not edge order, because executors in one superstep are applied in sorted id order (see [concurrency](#concurrency-within-a-superstep)).

### Switch-case

`add_switch(source, cases, default)` evaluates the cases in order and routes to the first that matches, or to the default. Build cases with `Case::new(predicate, target)`, `Case::new_async`, `Case::labeled(predicate, target, label)` or `Case::labeled_async`; the label only affects [visualisation](#visualisation). The default type is `agent_framework::workflow::Default`, which clashes with the standard `Default` trait, so import it under another name. A case predicate that returns `Err` fails the run instead of falling through to the default.

Several cases may route to the same executor, and the default may also be a case target. The builder merges them into one edge (the executor receives a message once, not once per matching case) and joins their labels:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{Case, Default as SwitchDefault, FunctionExecutor};
use serde_json::{json, Value};

fn handler(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        ctx.yield_output(json!({ "handled_by": id, "request": message })).await?;
        Ok(())
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let classify = FunctionExecutor::new("classify", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });
    let kind = |k: &'static str| move |msg: &Value| msg["kind"] == k;

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(classify))
        .add_executor(handler("billing"))
        .add_executor(handler("security"))
        .add_executor(handler("general"))
        .set_start("classify")
        .add_switch(
            "classify",
            vec![
                // Two cases may share a target; it is still one edge.
                Case::labeled(kind("refund"), "billing", "refund"),
                Case::labeled(kind("chargeback"), "billing", "chargeback"),
                Case::labeled_async(
                    |msg: &Value| {
                        let suspicious = msg["ip_reputation"].as_f64().unwrap_or(1.0) < 0.2;
                        async move { suspicious }
                    },
                    "security",
                    "bad ip",
                ),
            ],
            SwitchDefault::new("general"),
        )
        .build()?;

    for request in [
        json!({ "kind": "refund" }),
        json!({ "kind": "chargeback", "ip_reputation": 0.1 }),
        json!({ "kind": "question", "ip_reputation": 0.05 }),
        json!({ "kind": "question" }),
    ] {
        let run = workflow.run(request.clone()).await?;
        println!("{request} -> {}", run.last_output().unwrap()["handled_by"]);
    }
    println!("{}", workflow.viz().to_mermaid());
    Ok(())
}
```

```text
{"kind":"refund"} -> "billing"
{"ip_reputation":0.1,"kind":"chargeback"} -> "billing"
{"ip_reputation":0.05,"kind":"question"} -> "security"
{"kind":"question"} -> "general"
flowchart TD
  classify["classify (Start)"]
  billing["billing"]
  general["general"]
  security["security"]
  classify -- "refund, chargeback" --> billing
  classify -- "bad ip" --> security
  classify -- "default" --> general
```

The second request matches both `chargeback` and `bad ip`; the earlier case wins.

### Fan-out and multi-selection

`add_fan_out(source, targets)` delivers every message to every target. A target named twice receives it once.

`add_multi_selection(source, targets, selection)` hands each message and the group's target ids to a selection function, which returns the ids that should receive it: none, one or several. Build the function with `wrap_selection` (a closure returning `Vec<String>` or `Result<Vec<String>>`) or `wrap_async_selection`. Two rules are enforced per message: an id that is not one of the group's targets fails the run, and an id returned twice is delivered once. Python's `add_multi_selection_edge_group` is the equivalent.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{wrap_selection, FunctionExecutor};
use serde_json::{json, Value};

fn reviewer(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        ctx.send_message(json!({ "reviewer": id, "doc": message["title"] })).await?;
        Ok(())
    }))
}

fn targets(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| s.to_string()).collect()
}

fn build(selection_bug: bool) -> Result<Workflow> {
    let intake = FunctionExecutor::new("intake", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });
    let collect = FunctionExecutor::new("collect", |message, ctx| async move {
        ctx.yield_output(message).await?;
        Ok(())
    });

    // Pick reviewers from the message's tags.
    let by_tag = wrap_selection(move |msg: &Value, candidates: &[String]| -> Result<Vec<String>> {
        if selection_bug {
            return Ok(vec!["finance".to_string()]); // not one of the candidates
        }
        let tags: Vec<&str> = msg["tags"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        Ok(candidates.iter().filter(|c| tags.contains(&c.as_str())).cloned().collect())
    });

    WorkflowBuilder::new()
        .add_executor(Arc::new(intake))
        .add_executor(reviewer("legal"))
        .add_executor(reviewer("security"))
        .add_executor(reviewer("privacy"))
        .add_executor(Arc::new(collect))
        .set_start("intake")
        .add_multi_selection("intake", targets(&["legal", "security", "privacy"]), by_tag)
        .add_edge("legal", "collect")
        .add_edge("security", "collect")
        .add_edge("privacy", "collect")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let workflow = build(false)?;
    let run = workflow
        .run(json!({ "title": "New data retention policy", "tags": ["legal", "privacy", "legal"] }))
        .await?;
    println!("selected reviewers:\n{}", run.outputs().iter().map(|v| v.to_string()).collect::<Vec<_>>().join("\n"));

    let run = workflow.run(json!({ "title": "Typo fix", "tags": [] })).await?;
    println!("no tags -> {} outputs, state {:?}", run.outputs().len(), run.state());

    match build(true)?.run(json!({ "title": "x" })).await {
        Ok(_) => unreachable!(),
        Err(e) => println!("selection error: {e}"),
    }
    Ok(())
}
```

```text
selected reviewers:
{"doc":"New data retention policy","reviewer":"legal"}
{"doc":"New data retention policy","reviewer":"privacy"}
no tags -> 0 outputs, state Idle
selection error: workflow error: executor 'intake' selected target 'finance', which is not one of its fan-out group's targets: legal, security, privacy
```

A selection that picks nothing is not an error: the message goes nowhere and, if nothing else is queued, the run finishes `Idle` with no output.

### Fan-in

`add_fan_in(sources, target)` is a barrier. Messages from the sources are buffered until every source has delivered at least once; the target then runs once and receives a JSON array of the messages, in the order the sources were declared (not the order they arrived). The buffer persists across supersteps, so the sources do not have to finish in the same superstep, and it is saved in checkpoints. A source that sends twice before the barrier completes replaces its earlier message.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::{json, Value};

/// A worker that loops on itself `hops` times before reporting, so the two
/// workers reach the fan-in in different supersteps.
fn worker(id: &'static str, hops: u64, report_direct: bool) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        let n = message["hop"].as_u64().unwrap_or(0);
        if n < hops {
            ctx.send_to(id, json!({ "hop": n + 1 })).await?; // loop to itself
        } else if report_direct {
            ctx.send_to("merge", json!({ "worker": id, "hops": n })).await?;
        } else {
            ctx.send_message(json!({ "worker": id, "hops": n })).await?;
        }
        Ok(())
    }))
}

fn build(report_direct: bool) -> Result<Workflow> {
    let dispatch = FunctionExecutor::new("dispatch", |_message, ctx| async move {
        ctx.send_message(json!({ "hop": 0 })).await?;
        Ok(())
    });
    let merge = FunctionExecutor::new("merge", |message: Value, ctx| async move {
        ctx.yield_output(message).await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(dispatch))
        .add_executor(worker("fast", 0, report_direct))
        .add_executor(worker("slow", 2, false))
        .add_executor(Arc::new(merge))
        .set_start("dispatch")
        .add_fan_out("dispatch", vec!["fast".to_string(), "slow".to_string()])
        .add_fan_in(vec!["slow".to_string(), "fast".to_string()], "merge")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let run = build(false)?.run("go").await?;
    // One array, in declared source order (slow, fast), not arrival order.
    println!("barrier: {}", run.last_output().unwrap());
    let steps = run.events().iter().filter(|e| matches!(e, WorkflowEvent::SuperStepCompleted(_))).count();
    println!("supersteps: {steps}");

    // send_to aimed at a fan-in target skips the barrier.
    let run = build(true)?.run("go").await?;
    for out in run.outputs() {
        println!("send_to: {out}");
    }
    Ok(())
}
```

```text
barrier: [{"hops":2,"worker":"slow"},{"hops":0,"worker":"fast"}]
supersteps: 5
send_to: {"hops":0,"worker":"fast"}
```

The second run shows the trap. `send_to` addresses an executor directly, and direct messages bypass the barrier: `merge` runs at once with `fast`'s bare message (not an array), and `slow`'s later message then sits in the barrier waiting for a partner that never comes. The run still ends `Idle`. Report into a fan-in with `send_message`.

Give each target at most one fan-in group: the barrier is looked up by target, so a second group into the same target shares the first group's buffer and fires at the wrong time. A fan-in also never fires for a source that did not run, so a fan-in after a switch (where only one branch runs) waits forever; join switch branches with plain edges instead.

### Building edge groups by hand

The builder methods create `EdgeGroup` values, and the enum is public: `EdgeGroup::Single { source, target, condition }`, `EdgeGroup::FanOut { source, targets, selection, case_labels }` and `EdgeGroup::FanIn { sources, target }`. Each has `sources()`, `targets()` and `has_condition()`. `Condition` and `Selection` are the boxed async-closure types the variants hold; `wrap_sync_condition`, `wrap_async_condition`, `wrap_selection` and `wrap_async_selection` build them. You need these only to call [`validate_workflow_graph`](#validation) on a graph assembled outside the builder; `WorkflowBuilder` has no method that accepts an `EdgeGroup`.

## Loops and the superstep limit

Cycles are allowed: an edge from an executor back to itself, or around a loop of several, is how a workflow iterates. The engine has no notion of "done" beyond an empty queue, so a loop ends when its executor stops sending. `set_max_iterations(n)` caps the number of supersteps (100 by default). Exceeding it fails the run with `max_iterations (100) exceeded`: the check happens before the next superstep starts, so the events end with `Failed` and `Status(Failed)` and no `ExecutorFailed`.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

fn build(max_iterations: usize) -> Result<Workflow> {
    // Halve the number until it reaches 1, looping back to itself.
    let collatz = FunctionExecutor::new("collatz", |message, ctx| async move {
        let n = message.as_u64().unwrap_or(1);
        if n == 1 {
            ctx.yield_output(json!("reached 1")).await?;
        } else {
            let next = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
            ctx.send_message(json!(next)).await?;
        }
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(collatz))
        .set_start("collatz")
        .add_edge("collatz", "collatz")
        .set_max_iterations(max_iterations)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let run = build(100)?.run(json!(6)).await?;
    let steps = run.events().iter().filter(|e| matches!(e, WorkflowEvent::SuperStepCompleted(_))).count();
    println!("6: {} after {steps} supersteps", run.last_output().unwrap());

    match build(100)?.run(json!(27)).await {
        Ok(_) => unreachable!(),
        Err(e) => println!("27: {e}"),
    }
    let run = build(200)?.run(json!(27)).await?;
    println!("27 with 200: {}", run.last_output().unwrap());
    Ok(())
}
```

```text
6: "reached 1" after 9 supersteps
27: workflow error: max_iterations (100) exceeded
27 with 200: "reached 1"
```

The count is of supersteps, not executor invocations: a superstep that runs ten executors counts once. It is also cumulative over the life of a run. A run that pauses for a request and resumes continues counting from where it was, and a run restored from a checkpoint starts from the checkpoint's iteration count, so a long human-in-the-loop conversation inside a loop needs a limit that covers all its turns. Python's default is also 100 supersteps.

## Concurrency within a superstep

At the start of each superstep the engine takes every queued message, resolves its targets (running edge predicates and selections), and resolves fan-in barriers. It then runs the resulting invocations concurrently with `futures::future::join_all`, on the current task rather than spawned tasks. Different executors overlap; several messages to the same executor in one superstep run one after another, in the order they were sent, so a stateful executor never sees two of its own `execute` calls interleaved (upstream's per-executor lock does the same).

Once every invocation has finished, the effects are applied in a fixed order: by executor id, sorted, and within one executor in delivery order. Events, outputs and the next superstep's queue are therefore deterministic, whatever order the executors actually finished in.

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

fn slow(id: &'static str, ms: u64) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        tokio::time::sleep(Duration::from_millis(ms)).await;
        ctx.yield_output(json!({ "by": id, "got": message })).await?;
        Ok(())
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    // "start" sends two messages to "b_serial" and one each to the others.
    let start = FunctionExecutor::new("start", |_m, ctx| async move {
        ctx.send_to("c_slow", json!(1)).await?;
        ctx.send_to("a_fast", json!(2)).await?;
        ctx.send_to("b_serial", json!(3)).await?;
        ctx.send_to("b_serial", json!(4)).await?;
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(start))
        .add_executor(slow("a_fast", 10))
        .add_executor(slow("b_serial", 200))
        .add_executor(slow("c_slow", 300))
        .set_start("start")
        .add_fan_out("start", ["a_fast", "b_serial", "c_slow"].map(String::from))
        .build()?;

    let began = Instant::now();
    let run = workflow.run("go").await?;
    // a_fast, c_slow and b_serial overlap; b_serial's two deliveries run one
    // after the other, so the step takes about 400 ms, not 710 ms.
    println!("elapsed ~{} ms", began.elapsed().as_millis() / 100 * 100);
    // Events are applied in sorted executor-id order, not completion order.
    for e in run.events() {
        if let WorkflowEvent::Output { source_executor_id, data } = e {
            println!("{source_executor_id}: {data}");
        }
    }
    Ok(())
}
```

```text
elapsed ~400 ms
a_fast: {"by":"a_fast","got":2}
b_serial: {"by":"b_serial","got":3}
b_serial: {"by":"b_serial","got":4}
c_slow: {"by":"c_slow","got":1}
```

Three consequences are worth knowing:

- An executor's events, including `ExecutorInvoked`, are emitted only after the whole superstep has finished, so a streaming consumer sees nothing from a superstep until its slowest executor returns. This includes agent token updates (see [events](#events)).
- If one invocation fails, the superstep fails. Executors that ran beside it may already have done external work (sent an email, called an API), and there is no rollback for that. Effects are applied in id order up to the failing executor, so the outputs and events of executors sorted before it are recorded and those after it are not; no messages from the superstep are passed on, and its shared-state writes are discarded.
- Because the futures run on the calling task, a CPU-heavy executor holds up its siblings. Move blocking or CPU-bound work into `tokio::task::spawn_blocking`.

## Validation

`build()` checks the graph before returning a workflow. The checks run in this order and stop at the first failure:

| Check | `ValidationType` | Error message begins |
| --- | --- | --- |
| `set_start` was called | (none) | `no start executor set` |
| The start executor is registered | `StartNotRegistered` | `[START_NOT_REGISTERED] start executor 'x' is not registered` |
| Every edge names registered executors | `UnknownExecutor` | `[UNKNOWN_EXECUTOR] edge references unknown executor 'ghost'` |
| No directed edge is declared twice, across all edge groups | `EdgeDuplication` | `[EDGE_DUPLICATION] duplicate edge detected: 'a' -> 'b'` |
| Every executor is reachable from the start | `GraphConnectivity` | `[GRAPH_CONNECTIVITY] the following executors are unreachable ...` |
| `output_from` and `intermediate_output_from` do not overlap, and name registered executors | `OutputValidation` | `[OUTPUT_VALIDATION] ...` |

The duplicate check treats a plain edge and a conditional edge between the same pair as duplicates, and ignores a target repeated inside one fan-out or switch (which the builder merges anyway). Cycles pass. `build()` returns these as `Error::Workflow` with the message above. To get the category as a value, call `validate_workflow_graph(executors, edge_groups, start, output_from, intermediate_output_from)` yourself; it returns `Result<(), WorkflowValidationError>`, whose `validation_type` and `message` fields are public and whose `Display` adds the bracketed name.

```rust
use std::collections::HashMap;
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{validate_workflow_graph, EdgeGroup, FunctionExecutor, ValidationType};
use serde_json::json;

fn node(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, |m, ctx| async move {
        ctx.send_message(m).await?;
        Ok(())
    }))
}

fn report(label: &str, result: Result<Workflow>) {
    match result {
        Ok(_) => println!("{label:<14} ok"),
        Err(e) => println!("{label:<14} {e}"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let base = || WorkflowBuilder::new().add_executor(node("a")).add_executor(node("b"));

    report("no start", base().add_edge("a", "b").build());
    report("bad start", base().set_start("x").add_edge("a", "b").build());
    report("unknown", base().set_start("a").add_edge("a", "b").add_edge("b", "ghost").build());
    report("duplicate", base().set_start("a").add_edge("a", "b").add_conditional_edge("a", "b", |_| true).build());
    report("unreachable", base().set_start("a").build());
    report("output clash", base().set_start("a").add_edge("a", "b").output_from(["b"]).intermediate_output_from(["b"]).build());
    report("output typo", base().set_start("a").add_edge("a", "b").output_from(["c"]).build());
    report("cycle", base().set_start("a").add_edge("a", "b").add_edge("b", "a").build());

    // The same checks with the category available.
    let mut executors: HashMap<String, Arc<dyn Executor>> = HashMap::new();
    executors.insert("a".into(), node("a"));
    executors.insert("b".into(), node("b"));
    let edges = vec![EdgeGroup::Single { source: "a".into(), target: "b".into(), condition: None }];
    let err = validate_workflow_graph(&executors, &edges, "b", &[], &[]).unwrap_err();
    assert_eq!(err.validation_type, ValidationType::GraphConnectivity);
    println!("{:?} / {} / {}", err.validation_type, err.validation_type, err.message);

    // Routing mistakes the builder cannot see fail at run time.
    let typo = FunctionExecutor::new("a", |m, ctx| async move {
        ctx.send_to("bb", m).await?;
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(typo))
        .add_executor(node("b"))
        .set_start("a")
        .add_edge("a", "b")
        .build()?;
    match workflow.run(json!("x")).await {
        Ok(_) => unreachable!(),
        Err(e) => println!("run time:      {e}"),
    }
    Ok(())
}
```

```text
no start       workflow error: no start executor set
bad start      workflow error: [START_NOT_REGISTERED] start executor 'x' is not registered
unknown        workflow error: [UNKNOWN_EXECUTOR] edge references unknown executor 'ghost'
duplicate      workflow error: [EDGE_DUPLICATION] duplicate edge detected: 'a' -> 'b'. Each edge must be unique.
unreachable    workflow error: [GRAPH_CONNECTIVITY] the following executors are unreachable from the start executor 'a': ["b"]. This may indicate a disconnected workflow graph.
output clash   workflow error: [OUTPUT_VALIDATION] executor id(s) ["b"] are designated as both an output source (`output_from`) and an intermediate-output source (`intermediate_output_from`); an executor's yields cannot be both terminal and non-terminal.
output typo    workflow error: [OUTPUT_VALIDATION] workflow-output designation references unknown executor 'c'
cycle          ok
GraphConnectivity / GRAPH_CONNECTIVITY / the following executors are unreachable from the start executor 'b': ["a"]. This may indicate a disconnected workflow graph.
run time:      workflow error: executor 'a' sent a message to unknown target executor 'bb'
```

Python's validator also checks that connected executors' handler types are compatible; that check does not exist here, because there are no handler types. What validation cannot see fails at run time instead: a `send_to` to an unknown id, a selection naming an id outside its group, a predicate that returns `Err`. Each of those fails the run with `Error::Workflow`.

## Outputs

`ctx.yield_output(value)` is how a workflow produces results. With no configuration, every yield from every executor becomes a `WorkflowEvent::Output`, is returned by `run.outputs()`, and the last one is `run.last_output()`. Outputs are recorded in event order, which within a superstep is executor-id order.

### Which yields count as output

`output_from(ids)` and `intermediate_output_from(ids)` split yields into results and progress. A yield from an executor in `intermediate_output_from` becomes `WorkflowEvent::Intermediate`, which is on the event stream but never in `outputs()` or `last_output()`. Once `output_from` names at least one executor, yields from any executor not named there are also demoted to `Intermediate` rather than dropped. Naming the same executor in both, or an unknown id, fails validation. Python's equivalent is `WorkflowBuilder(output_from=[...])`.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use futures::StreamExt;
use serde_json::json;

fn echo(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |m, ctx| async move {
        ctx.yield_output(json!({ "from": id, "data": m })).await?;
        ctx.send_message(m).await?;
        Ok(())
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let workflow = WorkflowBuilder::new()
        .add_executor(echo("draft"))
        .add_executor(echo("review"))
        .add_executor(echo("publish"))
        .set_start("draft")
        .add_chain(["draft", "review", "publish"].map(String::from))
        .intermediate_output_from(["draft"])
        .output_from(["publish"])
        .build()?;

    let mut stream = workflow.run_stream("post");
    while let Some(event) = stream.next().await {
        match event {
            WorkflowEvent::Output { source_executor_id, .. } => println!("Output       from {source_executor_id}"),
            WorkflowEvent::Intermediate { source_executor_id, .. } => println!("Intermediate from {source_executor_id}"),
            _ => {}
        }
    }
    let run = stream.into_run().await?;
    println!("outputs(): {} -> {}", run.outputs().len(), run.last_output().unwrap());
    Ok(())
}
```

```text
Intermediate from draft
Intermediate from review
Output       from publish
outputs(): 1 -> {"data":"post","from":"publish"}
```

`review` was named in neither list but is demoted because `output_from` is set. The orchestration builders accept the same two methods, naming participants instead of executor ids; see [output designation](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#choosing-which-participants-produce-output) on the orchestrations page.

## Shared state

`ctx.shared_state()` returns the run's `SharedState`: a key-to-JSON store shared by every executor in the run, the Rust counterpart of Python's workflow `State`. It is cheap to clone, and all its methods are async because it sits behind a Tokio `RwLock`.

| Method | Behaviour |
| --- | --- |
| `get(key)` | `Option<Value>`: the latest value, pending or committed |
| `set(key, value)` | Stage a write |
| `has(key)` | Whether the key currently has a value |
| `delete(key)` | Stage a deletion; returns whether the key existed |
| `update(key, \|old\| new)` | Read and write one key under the write lock, so concurrent updates do not lose each other's changes |
| `clear()` | Remove everything, staged and committed, immediately |

Writes are staged in a pending buffer and committed at the end of each successful superstep. The buffer is shared, not per executor, so a write is visible straight away to any executor that reads after it, including one running concurrently in the same superstep. What staging guarantees is about failure and checkpoints: a checkpoint contains only committed values, and a superstep that fails has its staged writes discarded. Do not rely on two executors in one superstep seeing or not seeing each other's writes; if one depends on another's write, put an edge between them so the reader runs in a later superstep, or use `update` for read-modify-write.

```rust
use std::sync::Arc;
use std::time::Duration;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<()> {
    let fan = FunctionExecutor::new("fan", |m, ctx| async move {
        ctx.send_message(m).await?;
        Ok(())
    });
    // Writes a key, then lingers.
    let writer = FunctionExecutor::new("writer", |_m, ctx| async move {
        ctx.shared_state().set("draft", json!("v1")).await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        ctx.send_message(json!("done")).await?;
        Ok(())
    });
    // Runs in the same superstep, and reads after the writer has written.
    let peeker = FunctionExecutor::new("peeker", |_m, ctx| async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let seen = ctx.shared_state().get("draft").await;
        ctx.yield_output(json!({ "peeker_saw": seen })).await?;
        Ok(())
    });
    let counter = FunctionExecutor::new("counter", |_m, ctx| async move {
        let state = ctx.shared_state();
        state
            .update("count", |old| json!(old.and_then(|v| v.as_i64()).unwrap_or(0) + 1))
            .await;
        let removed = state.delete("draft").await;
        ctx.yield_output(json!({
            "count": state.get("count").await,
            "deleted_draft": removed,
            "has_draft": state.has("draft").await,
        }))
        .await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(fan))
        .add_executor(Arc::new(writer))
        .add_executor(Arc::new(peeker))
        .add_executor(Arc::new(counter))
        .set_start("fan")
        .add_fan_out("fan", ["writer", "peeker"].map(String::from))
        .add_edge("writer", "counter")
        .build()?;

    let run = workflow.run(Value::Null).await?;
    for out in run.outputs() {
        println!("{out}");
    }
    // The run's state stays readable after the run.
    println!("after run: count = {:?}", run.shared_state().get("count").await);

    // Each run starts with empty shared state.
    let second = workflow.run(Value::Null).await?;
    println!("second run count: {:?}", second.shared_state().get("count").await);
    Ok(())
}
```

```text
{"peeker_saw":"v1"}
{"count":1,"deleted_draft":true,"has_draft":false}
after run: count = Some(Number(1))
second run count: Some(Number(1))
```

Each run starts with an empty store, and `run.shared_state()` stays readable after the run. Keys beginning with `_` are reserved: the handoff and Magentic orchestrations keep their paused state under `_handoff_state`, `_magentic_plan_review_state` and `_magentic_stall_state`. A sub-workflow has its own store, separate from its parent's.

Shared state is the right place for large payloads that several executors need (pass the key along the edge, not the document) and for state that must survive a checkpoint when the executor is a `FunctionExecutor`.

## Events

Every run records a sequence of `WorkflowEvent`s, available as `run.events()` after the run or live from `run_stream`. The variants map onto Python's event classes:

| Variant | When | Python |
| --- | --- | --- |
| `Started` | A run starts, or resumes from a checkpoint | `WorkflowStartedEvent` |
| `Status(WorkflowRunState)` | The run state changes | `WorkflowStatusEvent` |
| `SuperStepStarted(n)`, `SuperStepCompleted(n)` | Around each superstep; `n` counts from 1 | `SuperStepStartedEvent` / `SuperStepCompletedEvent` |
| `ExecutorInvoked { executor_id }`, `ExecutorCompleted { executor_id }` | Around each successful invocation | `ExecutorInvokedEvent` / `ExecutorCompletedEvent` |
| `ExecutorFailed { executor_id, error }` | An executor returned `Err` | `ExecutorFailedEvent` |
| `AgentRunUpdate { executor_id, update }` | A streamed chunk from an agent node, as a serialised `AgentResponseUpdate` | `AgentRunUpdateEvent` |
| `AgentRun { executor_id, response }` | An agent node finished, as a serialised `AgentResponse` | `AgentRunEvent` |
| `Output { data, source_executor_id }` | A terminal yield | `WorkflowOutputEvent` |
| `Intermediate { data, source_executor_id }` | A non-terminal yield | |
| `RequestInfo { request_id, source_executor_id, request_data }` | A request for outside input | `RequestInfoEvent` |
| `Custom(Value)` | Added by an executor with `ctx.add_event` | custom `WorkflowEvent` subclasses |
| `Failed { error }` | The run failed | `WorkflowFailedEvent` |

`as_output()`, `as_intermediate()`, `as_request_info()` and `as_status()` pick the data out of the matching variant. The enum derives `Serialize` and `Deserialize`, so events can be logged or sent to a browser as JSON.

Within an executor's invocation the order is fixed: `ExecutorInvoked`, its outputs, its added events (including agent updates), its requests (each `RequestInfo`, plus `Status(InProgressPendingRequests)` the first time in a run), then `ExecutorCompleted`. So an agent executor's `Output` appears before the `AgentRunUpdate` events that led to it. `ctx.add_event` accepts any variant, not only `Custom`; nothing stops an executor emitting a misleading `Output` event that is not a real yield, so stick to `Custom` for your own events.

### Streaming events

`workflow.run_stream(input)` spawns the run on the Tokio runtime and returns a `WorkflowRunStream`, which implements `futures::Stream<Item = WorkflowEvent>`. When the stream ends, `into_run().await` returns the finished `WorkflowRun`, or the run's error. A failing run still streams its `ExecutorFailed`, `Failed` and `Status(Failed)` events before ending:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use futures::StreamExt;
use serde_json::json;

fn trace(event: &WorkflowEvent) -> String {
    match event {
        WorkflowEvent::Started => "Started".into(),
        WorkflowEvent::Status(s) => format!("Status({s:?})"),
        WorkflowEvent::SuperStepStarted(n) => format!("SuperStepStarted({n})"),
        WorkflowEvent::SuperStepCompleted(n) => format!("SuperStepCompleted({n})"),
        WorkflowEvent::ExecutorInvoked { executor_id } => format!("  ExecutorInvoked({executor_id})"),
        WorkflowEvent::ExecutorCompleted { executor_id } => format!("  ExecutorCompleted({executor_id})"),
        WorkflowEvent::ExecutorFailed { executor_id, error } => format!("  ExecutorFailed({executor_id}): {error}"),
        WorkflowEvent::AgentRunUpdate { executor_id, .. } => format!("  AgentRunUpdate({executor_id})"),
        WorkflowEvent::AgentRun { executor_id, .. } => format!("  AgentRun({executor_id})"),
        WorkflowEvent::Output { data, .. } => format!("  Output({data})"),
        WorkflowEvent::Intermediate { data, .. } => format!("  Intermediate({data})"),
        WorkflowEvent::Custom(data) => format!("  Custom({data})"),
        WorkflowEvent::RequestInfo { request_id, .. } => format!("  RequestInfo({request_id})"),
        WorkflowEvent::Failed { error } => format!("Failed({error})"),
    }
}

fn build() -> Result<Workflow> {
    let parse = FunctionExecutor::new("parse", |m, ctx| async move {
        let n: i64 = m.as_str().unwrap_or_default().trim().parse().map_err(|_| Error::Workflow(format!("not a number: {m}")))?;
        ctx.add_event(WorkflowEvent::Custom(json!({ "parsed": n })));
        ctx.send_message(json!(n)).await?;
        Ok(())
    });
    let square = FunctionExecutor::new("square", |m, ctx| async move {
        let n = m.as_i64().unwrap_or(0);
        ctx.yield_output(json!(n * n)).await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(parse))
        .add_executor(Arc::new(square))
        .set_start("parse")
        .add_edge("parse", "square")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let workflow = build()?;
    let mut stream = workflow.run_stream("12");
    while let Some(event) = stream.next().await {
        println!("{}", trace(&event));
    }
    let run = stream.into_run().await?;
    println!("result: {:?}\n", run.last_output());

    // A failing executor: the stream ends with Failed, and into_run returns Err.
    let mut stream = workflow.run_stream("twelve");
    while let Some(event) = stream.next().await {
        println!("{}", trace(&event));
    }
    match stream.into_run().await {
        Ok(_) => unreachable!(),
        Err(e) => println!("into_run: {e}"),
    }
    Ok(())
}
```

```text
Started
Status(InProgress)
SuperStepStarted(1)
  ExecutorInvoked(parse)
  Custom({"parsed":12})
  ExecutorCompleted(parse)
SuperStepCompleted(1)
SuperStepStarted(2)
  ExecutorInvoked(square)
  Output(144)
  ExecutorCompleted(square)
SuperStepCompleted(2)
Status(Idle)
result: Some(Number(144))

Started
Status(InProgress)
SuperStepStarted(1)
  ExecutorInvoked(parse)
  ExecutorFailed(parse): workflow error: not a number: "twelve"
Failed(workflow error: not a number: "twelve")
Status(Failed)
into_run: workflow error: not a number: "twelve"
```

Streaming is granular to the superstep, not the token: events from a superstep arrive together once all its executors have finished (see [concurrency](#concurrency-within-a-superstep)). An agent's `AgentRunUpdate` chunks arrive as a burst after that agent has finished, not as the model produces them. For live token streaming from a single agent, stream the agent directly; the [orchestrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#streaming-updates) shows what a workflow stream looks like in practice.

`run_stream` must be called inside a Tokio runtime. Dropping the stream does not stop the run: the spawned task carries on to the end (or to a pause), and its events are discarded. Only the streamed run is affected; to resume a paused streamed run, get it with `into_run()` and call `send_response` on it, which runs without streaming.

## Human in the loop

A workflow pauses for outside input by raising a request. There are two ways to raise one:

- `ctx.request_info(payload)` inside any executor. The answer comes back to the same executor.
- A `RequestInfoExecutor::new(id)` node. Every message it receives becomes a request, and the answer goes back to the executor that sent the message, not along the node's outgoing edges. (Python uses `ctx.request_info` with `@response_handler`; .NET routes the answer along the node's edges. Here it returns to the requester.)

Either way, the request is recorded as a `PendingRequest` (`request_id`, `source_executor_id`, `reply_to_executor_id`, `request_data`), a `RequestInfo` event is emitted, and the run carries on until no messages remain. It then stops in `IdleWithPendingRequests` instead of `Idle`. Other branches keep running while a request is open; only the requesting branch waits.

`run.send_response(id, value)` answers one request and `run.send_responses(map)` answers several (delivered in request-id order); both continue the same run until it finishes or pauses again. You do not have to answer every open request at once. The answer reaches the executor as an ordinary message shaped like `RequestResponse { request_id, data, original_request }`, so the executor can tell an answer from fresh input with `RequestResponse::from_message(&message)`, and gets the original payload back without keeping it anywhere. `send_responses` checks first: an unknown id fails with `response provided for unknown request id '...'`, and a run with nothing pending fails with `no pending requests to respond to`; neither changes the run.

```rust
use std::collections::HashMap;
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{FunctionExecutor, RequestResponse};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // Asks two questions at once with ctx.request_info; answers come back here.
    let planner = FunctionExecutor::new("planner", |message, ctx| async move {
        match RequestResponse::from_message(&message) {
            Some(answer) => {
                let field = answer.original_request["field"].clone();
                ctx.send_message(json!({ "field": field, "value": answer.data })).await?;
            }
            None => {
                ctx.request_info(json!({ "field": "date", "question": "Which day?" })).await?;
                ctx.request_info(json!({ "field": "guests", "question": "How many guests?" })).await?;
            }
        }
        Ok(())
    });
    // Collects both answers in shared state, then hands a summary to a
    // RequestInfoExecutor for final sign-off.
    let booking = FunctionExecutor::new("booking", |message, ctx| async move {
        if let Some(answer) = RequestResponse::from_message(&message) {
            let verdict = if answer.data == json!("yes") { "booked" } else { "cancelled" };
            ctx.yield_output(json!({ verdict: answer.original_request })).await?;
            return Ok(());
        }
        let field = message["field"].as_str().unwrap_or_default().to_string();
        let state = ctx.shared_state();
        state.set(field, message["value"].clone()).await;
        if state.has("date").await && state.has("guests").await {
            let summary = json!({ "date": state.get("date").await, "guests": state.get("guests").await });
            ctx.send_message(summary).await?;
        }
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(planner))
        .add_executor(Arc::new(booking))
        .add_executor(Arc::new(RequestInfoExecutor::new("confirm")))
        .set_start("planner")
        .add_edge("planner", "booking")
        .add_edge("booking", "confirm")
        .build()?;

    let mut run = workflow.run("Book a table").await?;
    println!("{:?}, {} pending", run.state(), run.pending_requests().len());

    // Answer both at once.
    let answers: HashMap<String, serde_json::Value> = run
        .pending_requests()
        .into_iter()
        .map(|r| {
            let value = if r.request_data["field"] == "date" { json!("Friday") } else { json!(4) };
            (r.request_id, value)
        })
        .collect();
    run.send_responses(answers).await?;

    let confirm = run.pending_requests().remove(0);
    println!(
        "confirm request from {} (reply to {}): {}",
        confirm.source_executor_id, confirm.reply_to_executor_id, confirm.request_data
    );

    // Mistakes are errors, and leave the run as it was.
    println!("{}", run.send_response("not-an-id", json!("yes")).await.unwrap_err());

    run.send_response(confirm.request_id, json!("yes")).await?;
    println!("{:?}: {}", run.state(), run.last_output().unwrap());
    println!("{}", run.send_response("late", json!("yes")).await.unwrap_err());
    Ok(())
}
```

```text
IdleWithPendingRequests, 2 pending
confirm request from confirm (reply to booking): {"date":"Friday","guests":4}
workflow error: response provided for unknown request id 'not-an-id'
Idle: {"booked":{"date":"Friday","guests":4}}
workflow error: no pending requests to respond to
```

`RequestResponse::from_message` tries to deserialise any object with `request_id`, `data` and `original_request` fields, so an ordinary input that happens to have those three fields would be mistaken for an answer. Do not use those field names in your own payloads.

A paused `WorkflowRun` lives in memory, and there is no way to rebuild one from a request id alone. To wait for a person across a restart, enable [checkpointing](#checkpointing): pending requests are saved in every checkpoint, and restoring brings them back under the same ids.

## Checkpointing

`with_checkpointing(storage)` saves a `WorkflowCheckpoint` after every successful superstep. A checkpoint holds everything needed to continue:

| Field | Contents |
| --- | --- |
| `checkpoint_id`, `workflow_id`, `workflow_name` | A fresh UUID; the id and name of the workflow that wrote it |
| `timestamp_millis`, `iteration_count` | When it was written, and how many supersteps had run |
| `messages` | The queue for the next superstep |
| `fanin_state` | Partially filled fan-in barriers, by target and source |
| `shared_state` | Committed shared state |
| `executor_states` | Whatever each executor returned from `snapshot_state` |
| `pending_requests` | Open requests, with their payloads |
| `metadata` | `superstep` and `checkpoint_type` (`"superstep_N"`) |
| `graph_signature`, `version` | A fingerprint of the graph topology, and the format version |

Storage implements the `CheckpointStorage` trait (`save`, `load`, `list(Option<workflow_id>)`, `delete`). Three implementations exist:

| Storage | Where | Notes |
| --- | --- | --- |
| `InMemoryCheckpointStorage::new()` | A map in the process | Cloning shares the map; for tests and short-lived processes |
| `FileCheckpointStorage::new(dir)` | One pretty-printed `<checkpoint_id>.json` per checkpoint | Creates the directory; writes to a temporary file and renames it, so a crash never leaves a half-written checkpoint; rejects ids other than ASCII letters, digits, `-` and `_`, so an id cannot escape the directory |
| `CosmosCheckpointStorage` | Azure Cosmos DB | `agent_framework::cosmos`, behind the `cosmos` feature; see the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/) |

`list` returns checkpoints oldest first by `timestamp_millis`. Several supersteps can finish within one millisecond, so sort by `iteration_count` when order matters. `get_checkpoint_summary(&cp)` gives a short view: `iteration_count`, `status` (`"idle"`, `"awaiting next superstep"` or `"awaiting request response"`), `pending_request_ids`, the ids of executors with saved state, and `targets`, which lists only explicitly addressed messages (`send_to` and request answers), not ordinary edge traffic.

A failed save is logged with `tracing::warn!` and does not fail the run, so a full disk shows up only in your logs. Nothing is ever pruned: every superstep of every run adds a checkpoint until you call `delete`.

### Resuming

`workflow.run_from_checkpoint(checkpoint_id, storage)` loads the checkpoint, restores the queue, fan-in buffers, shared state, iteration count, pending requests and executor states, re-emits a `RequestInfo` event for each open request, and continues. The storage you resume from can differ from the workflow's own: new checkpoints from the resumed run go to the storage passed to `with_checkpointing` (if any) and are filed under the resuming workflow's id, which is a new UUID each time a workflow is built.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{get_checkpoint_summary, FunctionExecutor};
use serde_json::json;

fn pipeline(storage: Option<Arc<dyn CheckpointStorage>>, extra_step: bool) -> Result<Workflow> {
    let step = |id: &'static str, add: i64| {
        Arc::new(FunctionExecutor::new(id, move |m, ctx| async move {
            ctx.shared_state().update("trail", |t| {
                let mut t = t.and_then(|v| v.as_array().cloned()).unwrap_or_default();
                t.push(json!(id));
                json!(t)
            }).await;
            ctx.send_message(json!(m.as_i64().unwrap_or(0) + add)).await?;
            Ok(())
        })) as Arc<dyn Executor>
    };
    let report = FunctionExecutor::new("report", |m, ctx| async move {
        let trail = ctx.shared_state().get("trail").await;
        ctx.yield_output(json!({ "result": m, "trail": trail })).await?;
        Ok(())
    });
    let mut ids = vec!["add_one".to_string(), "add_ten".to_string()];
    let mut builder = WorkflowBuilder::new()
        .name("adder")
        .add_executor(step("add_one", 1))
        .add_executor(step("add_ten", 10))
        .add_executor(Arc::new(report))
        .set_start("add_one");
    if extra_step {
        builder = builder.add_executor(step("add_hundred", 100));
        ids.push("add_hundred".to_string());
    }
    ids.push("report".to_string());
    builder = builder.add_chain(ids);
    if let Some(storage) = storage {
        builder = builder.with_checkpointing(storage);
    }
    builder.build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("wf-checkpoints-demo");
    let _ = std::fs::remove_dir_all(&dir);
    let storage: Arc<dyn CheckpointStorage> = Arc::new(FileCheckpointStorage::new(&dir)?);

    let workflow = pipeline(Some(storage.clone()), false)?;
    let run = workflow.run(json!(1)).await?;
    println!("full run: {}", run.last_output().unwrap());
    println!("signature: {}", workflow.graph_signature());

    let mut checkpoints = storage.list(Some(workflow.id())).await?;
    checkpoints.sort_by_key(|cp| cp.iteration_count);
    for cp in &checkpoints {
        let s = get_checkpoint_summary(cp);
        println!("  superstep {}: {} ({} queued)", s.iteration_count, s.status, cp.messages.len());
    }
    let after_first = checkpoints.iter().find(|cp| cp.iteration_count == 1).unwrap();
    println!("saved state: {:?}, queued: {}", after_first.shared_state, after_first.messages.len());

    // Same graph, new process: resumes where it stopped.
    let resumed = pipeline(None, false)?
        .run_from_checkpoint(&after_first.checkpoint_id, storage.clone())
        .await?;
    println!("resumed: {}", resumed.last_output().unwrap());

    // A changed graph is refused...
    let changed = pipeline(None, true)?;
    let err = changed.run_from_checkpoint(&after_first.checkpoint_id, storage.clone()).await;
    println!("changed graph: {}", err.err().unwrap().to_string().chars().take(52).collect::<String>());

    // ...unless you say you know what you are doing.
    let forced = changed.run_from_checkpoint_unchecked(&after_first.checkpoint_id, storage.clone()).await?;
    println!("unchecked: {}", forced.last_output().unwrap());

    println!("bad id: {}", storage.load("../etc/passwd").await.unwrap_err());
    println!("deleted: {}", storage.delete(&after_first.checkpoint_id).await?);
    Ok(())
}
```

```text
full run: {"result":12,"trail":["add_one","add_ten"]}
signature: v2-f36ef3d4b4b94e05
  superstep 1: awaiting next superstep (1 queued)
  superstep 2: awaiting next superstep (1 queued)
  superstep 3: idle (0 queued)
saved state: {"trail": Array [String("add_one")]}, queued: 1
resumed: {"result":12,"trail":["add_one","add_ten"]}
changed graph: workflow error: checkpoint graph signature mismatch:
unchecked: {"result":112,"trail":["add_one","add_ten","add_hundred"]}
bad id: workflow error: invalid checkpoint id "../etc/passwd": only ASCII alphanumerics, '-', and '_' are allowed
deleted: true
```

Each checkpoint records the graph's signature (`workflow.graph_signature()`): a hash of the start executor, the executor ids and the edge groups, independent of the order you declared them in. Adding, removing or retargeting an executor or edge changes it, and so does adding or removing a condition or changing a switch label. Changing what a predicate or an executor does, without changing the shape, does not. Resuming into a workflow whose signature differs is an error that names both signatures; `run_from_checkpoint_unchecked` resumes anyway (logging a warning), which is what you want after a deliberate change you know to be compatible. The unchecked resume above routed the queued message through the new `add_hundred` step. A checkpoint written by an older signature scheme gets a different error that says so; a checkpoint with no signature at all is resumed with a warning.

Restoring calls `restore_state` on every executor that has saved state. The executors are the workflow's shared instances, so the restored state is also what the next fresh `run` of that workflow starts with, as in the [custom executor example](#custom-executor-types).

### Resuming across a restart

A pending request survives in the checkpoint, so a process can stop while waiting for a person and a new process can pick the run up and answer the same request id:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{FunctionExecutor, RequestResponse};
use serde_json::json;

/// A review step that waits for a person. The workflow is built twice, as if
/// the process restarted while the request was open.
fn build(storage: Arc<dyn CheckpointStorage>) -> Result<Workflow> {
    let review = FunctionExecutor::new("review", |message, ctx| async move {
        match RequestResponse::from_message(&message) {
            Some(answer) => {
                ctx.yield_output(json!({ "doc": answer.original_request["doc"], "approved": answer.data })).await?;
            }
            None => {
                ctx.shared_state().set("submitted", message.clone()).await;
                ctx.request_info(json!({ "doc": message, "question": "Approve?" })).await?;
            }
        }
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(review))
        .set_start("review")
        .with_checkpointing(storage)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("wf-hitl-checkpoints");
    let _ = std::fs::remove_dir_all(&dir);
    let storage: Arc<dyn CheckpointStorage> = Arc::new(FileCheckpointStorage::new(&dir)?);

    // First process: start the run and stop once it is waiting.
    let first = build(storage.clone())?;
    let run = first.run("Q3 report").await?;
    let request_id = run.pending_requests()[0].request_id.clone();
    let checkpoint = storage.list(Some(first.id())).await?.pop().unwrap();
    println!("{:?}; pending in checkpoint: {}", run.state(), checkpoint.pending_requests.len());
    drop(run);

    // Second process: restore, then answer the same request id.
    let second = build(storage.clone())?;
    let mut resumed = second.run_from_checkpoint(&checkpoint.checkpoint_id, storage.clone()).await?;
    let same_id = resumed.pending_requests()[0].request_id == request_id;
    println!("restored: {:?}, same request id: {same_id}", resumed.state());
    let re_raised = resumed.events().iter().filter(|e| matches!(e, WorkflowEvent::RequestInfo { .. })).count();
    println!("RequestInfo events re-emitted on restore: {re_raised}");
    resumed.send_response(request_id, json!(true)).await?;
    println!("{:?}: {}", resumed.state(), resumed.last_output().unwrap());
    println!("shared state survived: {:?}", resumed.shared_state().get("submitted").await);

    // The resumed run checkpoints under the new workflow's id.
    println!("new checkpoints: {}", storage.list(Some(second.id())).await?.len());
    Ok(())
}
```

```text
IdleWithPendingRequests; pending in checkpoint: 1
restored: IdleWithPendingRequests, same request id: true
RequestInfo events re-emitted on restore: 1
Idle: {"approved":true,"doc":"Q3 report"}
shared state survived: Some(String("Q3 report"))
new checkpoints: 1
```

When restored from a paused checkpoint with nothing queued, `run_from_checkpoint` returns at once in `IdleWithPendingRequests`. The fan-in buffer is restored in the same way, so a checkpoint taken while a barrier is half full (one source delivered, the other still a superstep away) resumes and fires the barrier when the second source arrives; the library's `checkpoint_resume_fanin` example demonstrates exactly that.

Things that are not checkpointed:

- State in a `FunctionExecutor`'s closure, or in a custom executor that does not implement `snapshot_state`.
- The child runs of a [sub-workflow](#sub-workflows). `WorkflowExecutor` keeps a paused child run in memory and does not snapshot it, so a request raised inside a sub-workflow cannot be answered after a restart.
- The prebuilt orchestrations, whose builders have no `with_checkpointing`. See [checkpointing orchestrations](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#checkpointing-an-approval-gate) for the part you can rebuild by hand.

The library has no checkpoint-backed resume of a streamed run either: `run_from_checkpoint` does not stream.

## Sub-workflows

`WorkflowExecutor::new(id, child_workflow)` runs a whole workflow as one node of another. Each message it receives starts a new run of the child, with that message as the child's input. When the child finishes (or pauses), its outputs are sent on from the node as ordinary messages, one per output; `with_direct_output(true)` yields them as outputs of the parent instead. `workflow()` returns the wrapped child.

Requests raised inside the child are re-raised on the parent under the child's request ids, with the wrapper as their source, so the parent's caller sees them in `pending_requests()` and answers with the usual `send_response`. The wrapper routes the answer into the paused child run, continues it, and forwards any new outputs.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{FunctionExecutor, RequestResponse};
use serde_json::json;

/// Child workflow: asks a person for a translation of each phrase.
fn translator() -> Result<Workflow> {
    let ask = FunctionExecutor::new("ask", |message, ctx| async move {
        match RequestResponse::from_message(&message) {
            Some(answer) => ctx.yield_output(answer.data).await?,
            None => ctx.request_info(json!({ "translate": message })).await?,
        }
        Ok(())
    });
    WorkflowBuilder::new().name("translator").add_executor(Arc::new(ask)).set_start("ask").build()
}

fn word_count() -> Result<Workflow> {
    let count = FunctionExecutor::new("count", |m, ctx| async move {
        ctx.yield_output(json!(m.as_str().unwrap_or_default().split_whitespace().count())).await?;
        Ok(())
    });
    WorkflowBuilder::new().add_executor(Arc::new(count)).set_start("count").build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let intro = FunctionExecutor::new("intro", |m, ctx| async move {
        ctx.send_message(m).await?;
        Ok(())
    });
    let report = FunctionExecutor::new("report", |m, ctx| async move {
        ctx.yield_output(json!({ "parent_saw": m })).await?;
        Ok(())
    });

    let parent = WorkflowBuilder::new()
        .add_executor(Arc::new(intro))
        .add_executor(Arc::new(WorkflowExecutor::new("translate", translator()?)))
        // This child's outputs become parent outputs directly.
        .add_executor(Arc::new(WorkflowExecutor::new("words", word_count()?).with_direct_output(true)))
        .add_executor(Arc::new(report))
        .set_start("intro")
        .add_fan_out("intro", ["translate", "words"].map(String::from))
        .add_edge("translate", "report")
        .build()?;

    let mut run = parent.run("good morning").await?;
    println!("outputs so far: {}", serde_json::Value::from(run.outputs()));
    // The child's request surfaces on the parent run, under the child's id.
    let request = run.pending_requests().remove(0);
    println!("from {}: {}", request.source_executor_id, request.request_data);
    run.send_response(request.request_id, json!("bonjour")).await?;
    println!("{:?}: {}", run.state(), serde_json::Value::from(run.outputs()));
    Ok(())
}
```

```text
outputs so far: [2]
from translate: {"translate":"good morning"}
Idle: [2,{"parent_saw":"bonjour"}]
```

Points to know:

- The child runs in its own shared state, cannot see the parent's, and gets its own superstep limit. It runs to completion inside one parent superstep, so to the parent the whole child is one step.
- The child's events are not forwarded. The parent's stream shows the wrapper's `ExecutorInvoked` and `ExecutorCompleted` but nothing from inside the child, including agent updates. If you need them, have the child yield intermediate progress, which becomes messages or outputs of the wrapper.
- A child failure fails the wrapper executor, and so the parent run.
- Paused child runs are held in memory in the wrapper and are not checkpointed (see [above](#resuming-across-a-restart)).

Python wraps child requests in `SubWorkflowRequestMessage` and lets the parent intercept and answer them itself with a handler. That interception is not ported: here every child request goes to the parent's caller.

## Visualisation

`workflow.viz()` returns a `WorkflowViz` with `to_mermaid()` (a `flowchart TD`) and `to_dot()` (Graphviz). The start executor is marked, conditional edges are dashed and labelled `conditional`, switch branches carry their case labels (`case 0`, `case 1`, ... when unlabelled, and `default`), and each fan-in is drawn as a separate join node. Mermaid node ids are sanitised (`audit-log` becomes `audit_log`) with collisions resolved, while labels keep the real id. Output is deterministic, so it can be checked into a repository and diffed.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{Case, Default as SwitchDefault, FunctionExecutor};

fn noop(id: &str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id.to_string(), |_m, _ctx| async move { Ok(()) }))
}

fn main() -> Result<()> {
    let workflow = WorkflowBuilder::new()
        .add_executor(noop("intake"))
        .add_executor(noop("audit-log"))
        .add_executor(noop("fast_path"))
        .add_executor(noop("review"))
        .add_executor(noop("merge"))
        .add_executor(noop("archive"))
        .set_start("intake")
        .add_conditional_edge("intake", "audit-log", |m| m["audit"] == true)
        .add_switch(
            "intake",
            vec![Case::labeled(|m| m["amount"].as_f64().unwrap_or(0.0) < 100.0, "fast_path", "small")],
            SwitchDefault::new("review"),
        )
        .add_edge("fast_path", "merge")
        .add_edge("review", "merge")
        .add_fan_in(["audit-log", "merge"].map(String::from), "archive")
        .build()?;

    println!("{}", workflow.viz().to_mermaid());
    println!("{}", workflow.viz().to_dot());
    Ok(())
}
```

```text
flowchart TD
  intake["intake (Start)"]
  archive["archive"]
  audit_log["audit-log"]
  fast_path["fast_path"]
  merge["merge"]
  review["review"]
  fan_in_archive_0((fan-in))
  audit_log --> fan_in_archive_0
  merge --> fan_in_archive_0
  fan_in_archive_0 --> archive
  intake -. conditional .-> audit_log
  intake -- "small" --> fast_path
  intake -- "default" --> review
  fast_path --> merge
  review --> merge
digraph Workflow {
  rankdir=TD;
  node [shape=box, style=filled, fillcolor=lightblue];
  edge [color=black, arrowhead=vee];

  "intake" [fillcolor=lightgreen, label="intake\n(Start)"];
  "archive" [label="archive"];
  "audit-log" [label="audit-log"];
  "fast_path" [label="fast_path"];
  "merge" [label="merge"];
  "review" [label="review"];
  "fan_in_archive_0" [shape=ellipse, fillcolor=lightgoldenrod, label="fan-in"];
  "audit-log" -> "fan_in_archive_0";
  "merge" -> "fan_in_archive_0";
  "fan_in_archive_0" -> "archive";
  "intake" -> "audit-log" [style=dashed, label="conditional"];
  "intake" -> "fast_path" [label="small"];
  "intake" -> "review" [label="default"];
  "fast_path" -> "merge";
  "review" -> "merge";
}
```

There is no image export (Python's `WorkflowViz` can write SVG or PNG through Graphviz); pipe the DOT text to `dot -Tsvg` yourself. A sub-workflow appears as a single node.

## Agents in workflows

`AgentExecutor::new(id, agent)` wraps any `Arc<dyn SupportsAgentRun>` as a node, so agents and function executors can share a graph. It accepts a string, a message or a conversation, runs the agent over it, appends the agent's reply messages, and sends the whole conversation on as a JSON array of `Message`. Every message in the reply gets the executor id as its `author_name` if it has none.

| Option | Effect |
| --- | --- |
| `with_output(true)` | Yield the conversation as a workflow output instead of sending it on |
| `with_also_send(true)` | With `with_output(true)`, also send it on; needed for an agent in the middle of a graph whose conversation you want as an output |

The executor always drives the agent through `run_stream`, so the chat client's streaming path is used even for `workflow.run`, and a test double must implement `get_streaming_response`. Each chunk becomes an `AgentRunUpdate` event and the finished response an `AgentRun` event, both carrying the executor id.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{AgentExecutor, FunctionExecutor};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let writer: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone())
            .name("writer")
            .instructions("Write one sentence on the topic.")
            .build(),
    );
    let critic: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client)
            .name("critic")
            .instructions("Reply APPROVED if the sentence is clear, otherwise say why not.")
            .build(),
    );

    // A plain function node between agents: trims the conversation to the
    // latest draft so the critic sees only that.
    let latest_only = FunctionExecutor::new("latest_only", |message, ctx| async move {
        let conversation: Vec<Message> = serde_json::from_value(message)?;
        let draft = conversation.last().map(Message::text).unwrap_or_default();
        ctx.send_message(json!(format!("Review this sentence: {draft}"))).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        // The writer's conversation is yielded as an intermediate output and
        // still passed on.
        .add_executor(Arc::new(AgentExecutor::new("writer", writer).with_output(true).with_also_send(true)))
        .add_executor(Arc::new(latest_only))
        .add_executor(Arc::new(AgentExecutor::new("critic", critic).with_output(true)))
        .set_start("writer")
        .add_chain(["writer", "latest_only", "critic"].map(String::from))
        .intermediate_output_from(["writer"])
        .output_from(["critic"])
        .build()?;

    let run = workflow.run("Why Rust has no garbage collector").await?;
    for event in run.events() {
        if let WorkflowEvent::AgentRun { executor_id, response } = event {
            let response: AgentResponse = serde_json::from_value(response.clone())?;
            println!("{executor_id}: {}", response.text());
        }
    }
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("critic saw {} messages", conversation.len());
    Ok(())
}
```

```text
writer: Mock reply to: Why Rust has no garbage collector
critic: Mock reply to: Review this sentence: Mock reply to: Why Rust has no garbage collector
critic saw 2 messages
```

The executor passes no session to the agent, so each invocation starts from the conversation it is given, with no memory of earlier runs; history between agents is whatever travels along the edges. Python's `AgentExecutor` keeps a session per executor and offers `context_mode`, `AgentExecutorRequest`/`AgentExecutorResponse` envelopes and `should_respond`; none of those is ported, and a `FunctionExecutor` between agents (as above) is how you shape what the next agent sees.

`AgentApprovalExecutor::new(id, agent)` is the same node with a review step after each agent turn. It has the same two options. The [orchestrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_orchestrations/#agent-approval) covers it, since the sequential and concurrent builders use it for `with_request_info()`.

## A workflow as an agent

`workflow.as_agent(name)` (from the `WorkflowAgentExt` trait in the prelude) or `WorkflowAgent::new(workflow, name)` wraps any workflow as a `WorkflowAgent`, which implements `SupportsAgentRun`. It can then be a participant in an orchestration, an `AgentExecutor` node in another workflow, a tool for another agent, or an agent served over HTTP by the [hosting crate](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/).

| Method | Effect |
| --- | --- |
| `with_id(id)` | Replace the generated id (`workflow_agent_<uuid>`) |
| `with_description(text)` | Set the description, used by `as_tool` |
| `workflow()`, `description()` | Accessors |
| `run(messages, session)` | Run the workflow with the messages as a JSON array; the response holds its outputs |
| `run_stream(messages, session, options)` | Stream the workflow's agent updates as `AgentResponseUpdate`s |
| `run_stream_with_thread(messages, session)` | The same, without the trait's `Result` wrapper |
| `as_tool()` | A `ToolDefinition` with one `task` string parameter; the tool returns the response text |

`run` maps each workflow output to messages: an array of messages is used as is, a single message or string becomes one message, and anything else is turned into JSON text. Open requests are added to the response as an assistant message with a `request_info` function call and approval request, mirroring Python. Streaming forwards each `AgentRunUpdate` (with the agent's name if the update has none) and each `RequestInfo`; other events are dropped. Per-run `AgentRunOptions` are ignored with a warning.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let agent = |name: &str, instructions: &str| -> Arc<dyn SupportsAgentRun> {
        Arc::new(Agent::builder(client.clone()).name(name).instructions(instructions).build())
    };
    let pipeline = SequentialBuilder::new()
        .participants(vec![
            agent("drafter", "Draft a short answer."),
            agent("polisher", "Tighten the previous draft."),
        ])
        .build()?;

    let writer = pipeline
        .as_agent("draft_and_polish")
        .with_id("draft-and-polish")
        .with_description("Drafts and polishes a short answer.");

    // 1. Run it like any agent. The sequential output is the whole
    //    conversation, so the response includes the user's message too.
    let mut session = writer.create_session();
    let response = writer.run(vec![Message::user("A haiku about lifetimes")], Some(&mut session)).await?;
    for m in &response.messages {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }

    // 2. Stream it: one update per participant chunk. The final chunks of
    //    each agent carry no text (finish reason, usage), so skip those.
    let mut stream = writer.run_stream(vec![Message::user("Explain borrowing")], None, None).await?;
    while let Some(update) = stream.next().await {
        let update = update?;
        if !update.text().is_empty() {
            println!("update from {:?}: {:?}", update.author_name, update.text());
        }
    }

    // 3. Hand it to another agent as a tool.
    let tool = writer.as_tool();
    println!("tool name: {}, description: {}", tool.name, tool.description);
    let editor = Agent::builder(client.clone())
        .name("editor")
        .instructions("Use the draft_and_polish tool, then report the result.")
        .tool(tool)
        .build();
    let reply = editor.run(vec![Message::user("Write about ownership")], None).await?;
    println!("editor: {}", reply.text());
    Ok(())
}
```

```text
user: A haiku about lifetimes
drafter: Mock reply to: A haiku about lifetimes
polisher: Mock reply to: A haiku about lifetimes
update from Some("drafter"): "Mock"
update from Some("drafter"): " reply"
update from Some("drafter"): " to:"
update from Some("drafter"): " Explain"
update from Some("drafter"): " borrowing"
update from Some("polisher"): "Mock"
update from Some("polisher"): " reply"
update from Some("polisher"): " to:"
update from Some("polisher"): " Explain"
update from Some("polisher"): " borrowing"
tool name: draft_and_polish, description: Drafts and polishes a short answer.
editor: Here is what the tool returned: example taskMock reply to: example taskMock reply to: example task
```

Gotchas:

- A sequential or concurrent workflow outputs the whole conversation, so the response includes the input user message, and `as_tool` returns the text of every message run together, task included (visible in the last line above). Put a function executor at the end that yields only the final message if you want just the answer.
- Sessions are written to, not read from. After a run, the session's context providers' `after_run` is called with the input and response messages, so a history provider accumulates the conversation; but `before_run` is never called, so earlier history is not fed into the workflow. Python behaves the same way.
- A request raised inside the workflow is reported in the response, but each `run` is a new workflow run, so there is no way to answer it through the agent interface. Workflows that pause are better driven directly.
- Streaming has the [superstep granularity](#streaming-events) of the underlying workflow.

## What is not here yet

Compared with the Python engine, this release does not have:

- typed executor handlers, routing by message type, or type checking between connected executors (see [above](#messages-are-json-values));
- upstream's agent envelopes (`AgentExecutorRequest`, `AgentExecutorResponse`), `context_mode` and per-executor agent sessions;
- interception of sub-workflow requests by the parent, and checkpointing of paused sub-workflow runs;
- run cancellation, or any way to stop a run between supersteps;
- workflow-level OpenTelemetry spans (agents inside a workflow are traced as usual; see the [observability page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_observability/));
- functional (`@workflow` / `@step`) workflows, listed as `FUNCTIONAL_WORKFLOWS` in the crate's feature-stage table as not ported;
- the Power Fx declarative workflow language. The declarative crate loads its own Rust-native workflow format onto this engine instead (see the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/)).

The crate's `PARITY.md` and `SCOPE.md` track the current list.

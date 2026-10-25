---
title: "Microsoft Agent Framework (Rust) — Workflows and orchestrations"
description: "Graph workflows in agent-framework-rs: executors, edges, conditions, switch-case, fan-out and fan-in, shared state, human-in-the-loop requests, checkpointing, sub-workflows, streaming events and visualisation, plus the sequential, concurrent, group chat, handoff and Magentic orchestrations."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Workflows"
  order: 3
---

The `workflow` module is a graph engine in the style of the Python `WorkflowBuilder`. A workflow is a set of executors joined by edges; it runs in supersteps (Pregel style), where every message sent during a superstep is delivered at the start of the next one, and executors scheduled in the same superstep run concurrently. A run ends when no messages remain, or fails when it exceeds the superstep limit (100 by default, `set_max_iterations` to change). The prebuilt orchestrations at the end of the page are built on the same engine.

All examples were compiled against `agent-framework` %%version=agent-framework-rs%%. The graph examples use plain function executors and were run offline; the orchestration examples call OpenAI and were run against a local stand-in for its API. Everything here is in the stable surface: no `experimental-*` feature is needed.

## Messages are JSON values

The biggest difference from the Python and .NET engines is typing. There, executors declare handler types and the builder checks that edges connect compatible types. Here every message, output and request payload is a `serde_json::Value`. An executor receives a `Value`, decides what it means, and sends `Value`s on. Use `serde_json::from_value` and `to_value` with your own structs where a payload has structure, and treat a type mismatch as a run-time error. In return, checkpoints need no type registry: everything is already JSON.

Agent-based executors and the orchestrations exchange conversations, serialised as a JSON array of `Message`. A plain string as workflow input is turned into one user message.

## Executors and edges

An executor implements the `Executor` trait (`id`, `execute`, and optionally `snapshot_state` / `restore_state` for checkpointing), or is built from an async closure with `FunctionExecutor::new(id, |message, ctx| async move { ... })`. Inside, the `WorkflowContext` is how an executor acts:

| Method | Effect |
| --- | --- |
| `ctx.send_message(value)` | Send to every executor connected by an outgoing edge |
| `ctx.send_to(id, value)` | Send to one specific downstream executor |
| `ctx.yield_output(value)` | Emit a workflow output |
| `ctx.request_info(value)` | Ask the caller for input and pause (see [human in the loop](#human-in-the-loop)) |
| `ctx.shared_state()` | The run's [shared state](#shared-state) |
| `ctx.add_event(event)` | Add a custom event to the stream |
| `ctx.executor_id()`, `ctx.source_executor_ids()` | Who is running, and who sent the message |

`WorkflowBuilder` registers executors with `add_executor`, picks the entry point with `set_start`, and adds edges. `add_chain(ids)` is shorthand for consecutive edges. `build()` validates the graph: the start executor must exist, every edge must name registered executors, no edge may be declared twice, and every executor must be reachable from the start. Cycles are allowed.

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
        .add_executor(Arc::new(upper))
        .add_executor(Arc::new(count))
        .set_start("upper")
        .add_edge("upper", "count")
        .build()?;

    let run = workflow.run("graphs all the way down").await?;
    println!("state: {:?}", run.state());
    println!("output: {}", run.last_output().unwrap());
    Ok(())
}
```

`workflow.run(input)` returns a `WorkflowRun` with `state()`, `outputs()`, `last_output()`, `events()`, `pending_requests()` and `shared_state()`. A failing executor fails the run: `run` returns `Err`, after recording `ExecutorFailed` and `Failed` events. A built `Workflow` is cheap to clone and can be run many times; each run gets fresh shared state, but executor structs keep whatever state they hold in their own fields.

## Routing: conditions, switch-case and fan-out

Edges can carry conditions, and edge groups route one message to several targets:

| Builder method | Routing |
| --- | --- |
| `add_edge(a, b)` | Always deliver |
| `add_conditional_edge(a, b, \|msg\| bool)` | Deliver when the predicate holds; `add_conditional_edge_async` takes an async predicate |
| `add_switch(a, cases, Default::new(b))` | The first matching `Case` wins, otherwise the default target |
| `add_fan_out(a, targets)` | Deliver to every target |
| `add_multi_selection(a, targets, selection)` | A function picks any subset of the targets per message (`wrap_selection`) |
| `add_fan_in(sources, b)` | Wait until every source has sent, then deliver a JSON array of their messages in the order the sources were declared |

A predicate may return `bool` or `Result<bool>`; an `Err` fails the run rather than counting as "no match". The switch default type is called `Default`, so import it under another name to avoid clashing with the standard `Default` trait.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{Case, Default as SwitchDefault, FunctionExecutor};
use serde_json::{json, Value};

fn worker(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        ctx.send_message(json!({ "by": id, "ticket": message })).await?;
        Ok(())
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let triage = FunctionExecutor::new("triage", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });
    let report = FunctionExecutor::new("report", |message, ctx| async move {
        ctx.yield_output(message).await?;
        Ok(())
    });
    // Fan-in sink: receives an array once both reviewers have replied.
    let merge = FunctionExecutor::new("merge", |message, ctx| async move {
        let reviews = message.as_array().cloned().unwrap_or_default();
        ctx.yield_output(json!({ "reviews": reviews.len() })).await?;
        Ok(())
    });

    let mentions = |word: &'static str| {
        move |msg: &Value| msg.as_str().is_some_and(|s| s.to_lowercase().contains(word))
    };

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(triage))
        .add_executor(worker("billing"))
        .add_executor(worker("security_review"))
        .add_executor(worker("legal_review"))
        .add_executor(worker("general"))
        .add_executor(Arc::new(report))
        .add_executor(Arc::new(merge))
        .set_start("triage")
        .add_switch(
            "triage",
            vec![
                Case::labeled(mentions("invoice"), "billing", "invoice"),
                Case::labeled(mentions("breach"), "security_review", "breach"),
            ],
            SwitchDefault::new("general"),
        )
        // A breach goes to security, which fans out to legal as well.
        .add_fan_out("security_review", vec!["legal_review".to_string()])
        .add_fan_in(
            vec!["security_review".to_string(), "legal_review".to_string()],
            "merge",
        )
        .add_edge("billing", "report")
        .add_edge("general", "report")
        .build()?;

    for ticket in ["My invoice is wrong", "We had a data breach", "Opening hours?"] {
        let run = workflow.run(ticket).await?;
        println!("{ticket:>22} -> {}", run.last_output().unwrap());
    }
    Ok(())
}
```

In a fan-in, a source that sends twice before the barrier completes replaces its earlier message rather than adding a second one.

### Which yields count as output

By default every `yield_output` is a terminal `WorkflowEvent::Output` and appears in `outputs()`. Once you call `output_from(ids)` on the builder, only those executors produce outputs; yields from the others become `WorkflowEvent::Intermediate` events, which appear in the event stream but not in `outputs()`. `intermediate_output_from(ids)` marks executors as intermediate explicitly. The orchestration builders accept the same two methods, naming participants.

## Shared state

`ctx.shared_state()` returns a key-to-JSON store shared by every executor in a run, with async `get`, `set`, `has`, `delete` and `update(key, |old| new)`. Writes are staged: an executor sees its own writes at once, but other executors see them from the next superstep, and a checkpoint only ever contains committed values. A superstep that fails has its writes discarded. Keys starting with `_` are reserved for the framework. A sub-workflow gets its own store.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::{json, Value};

async fn log(ctx: &WorkflowContext, step: &'static str) {
    ctx.shared_state()
        .update("trail", move |old| {
            let mut trail = old.and_then(|v| v.as_array().cloned()).unwrap_or_default();
            trail.push(json!(step));
            Value::Array(trail)
        })
        .await;
}

#[tokio::main]
async fn main() -> Result<()> {
    let ingest = FunctionExecutor::new("ingest", |message, ctx| async move {
        ctx.shared_state().set("doc", message).await;
        log(&ctx, "ingest").await;
        ctx.send_message(json!("doc")).await?; // pass a key, not the payload
        Ok(())
    });
    let finish = FunctionExecutor::new("finish", |key, ctx| async move {
        let state = ctx.shared_state();
        let doc = state.get(key.as_str().unwrap()).await.unwrap_or_default();
        log(&ctx, "finish").await;
        let trail = state.get("trail").await.unwrap_or_default();
        ctx.yield_output(json!({ "doc": doc, "trail": trail })).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(ingest))
        .add_executor(Arc::new(finish))
        .set_start("ingest")
        .add_edge("ingest", "finish")
        .build()?;

    let run = workflow.run("A long document").await?;
    println!("{}", run.last_output().unwrap());
    Ok(())
}
```

## Human in the loop

There are two ways to pause for outside input. An executor can call `ctx.request_info(payload)`, and the answer comes back to that same executor. Or a `RequestInfoExecutor` node turns every message it receives into a request, and the answer goes back to the executor that sent it. Either way, the run finishes its current work and stops in `WorkflowRunState::IdleWithPendingRequests`. `run.pending_requests()` lists the requests (`request_id`, `source_executor_id`, `request_data`), and `run.send_response(id, value)` or `run.send_responses(map)` continues the same run.

The answer arrives as an ordinary message shaped like `RequestResponse` (`request_id`, `data`, `original_request`); `RequestResponse::from_message(&message)` tells an answer apart from a fresh input.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{FunctionExecutor, RequestResponse};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let drafter = FunctionExecutor::new("drafter", |message, ctx| async move {
        match RequestResponse::from_message(&message) {
            // The reviewer answered.
            Some(answer) if answer.data == json!("approve") => {
                ctx.yield_output(json!({ "published": answer.original_request["draft"] }))
                    .await?;
            }
            Some(answer) => {
                ctx.yield_output(json!({ "rejected": answer.data })).await?;
            }
            // First visit: draft something and ask for review.
            None => {
                let draft = format!("Draft about {}", message.as_str().unwrap_or("?"));
                ctx.request_info(json!({ "draft": draft, "question": "Publish?" }))
                    .await?;
            }
        }
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(drafter))
        .set_start("drafter")
        .build()?;

    let mut run = workflow.run("supersteps").await?;
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);

    let request = run.pending_requests().remove(0);
    println!("asked: {}", request.request_data);
    run.send_response(request.request_id, json!("approve")).await?;

    println!("{:?}: {}", run.state(), run.last_output().unwrap());
    Ok(())
}
```

A paused run lives in memory. To wait for a person across a restart, enable checkpointing: pending requests are saved in each checkpoint, and resuming from one restores them, after which `send_response` works as before.

## Checkpointing

`with_checkpointing(storage)` saves a `WorkflowCheckpoint` at every superstep boundary: queued messages, fan-in buffers, shared state, executor snapshots, pending requests and the iteration count. `InMemoryCheckpointStorage` and `FileCheckpointStorage` (one JSON file per checkpoint) are built in, and the `cosmos` feature adds a Cosmos DB store. `storage.list(Some(workflow_id))` lists saved checkpoints, and `get_checkpoint_summary` gives a short view of one.

`workflow.run_from_checkpoint(id, storage)` restores a run and carries on. Each checkpoint records a signature of the graph's topology; resuming into a workflow whose graph has changed is an error, and `run_from_checkpoint_unchecked` skips the check when you have changed the graph on purpose. Only executors that implement `snapshot_state` and `restore_state` keep their own state across a restore; a `FunctionExecutor` keeps nothing, so keep state that must survive in shared state.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{get_checkpoint_summary, FunctionExecutor};
use serde_json::json;

fn pipeline(storage: Arc<dyn CheckpointStorage>) -> Result<Workflow> {
    let double = FunctionExecutor::new("double", |m, ctx| async move {
        ctx.send_message(json!(m.as_i64().unwrap_or(0) * 2)).await?;
        Ok(())
    });
    let add_ten = FunctionExecutor::new("add_ten", |m, ctx| async move {
        ctx.send_message(json!(m.as_i64().unwrap_or(0) + 10)).await?;
        Ok(())
    });
    let report = FunctionExecutor::new("report", |m, ctx| async move {
        ctx.yield_output(m).await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(double))
        .add_executor(Arc::new(add_ten))
        .add_executor(Arc::new(report))
        .set_start("double")
        .add_chain(["double", "add_ten", "report"].map(String::from))
        .with_checkpointing(storage)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("maf-rust-checkpoints");
    let _ = std::fs::remove_dir_all(&dir);
    let storage: Arc<dyn CheckpointStorage> = Arc::new(FileCheckpointStorage::new(&dir)?);

    let workflow = pipeline(storage.clone())?;
    let run = workflow.run(json!(3)).await?;
    println!("full run: {:?}", run.last_output());

    let mut checkpoints = storage.list(Some(workflow.id())).await?;
    checkpoints.sort_by_key(|cp| cp.iteration_count);
    for cp in &checkpoints {
        let summary = get_checkpoint_summary(cp);
        println!("checkpoint {} after superstep {}", summary.checkpoint_id, summary.iteration_count);
    }

    // A fresh workflow with the same graph, as after a restart.
    let restarted = pipeline(storage.clone())?;
    let resumed = restarted
        .run_from_checkpoint(&checkpoints[0].checkpoint_id, storage)
        .await?;
    println!("resumed: {:?}", resumed.last_output());
    assert_eq!(resumed.last_output(), run.last_output());
    Ok(())
}
```

## Streaming events

`workflow.run_stream(input)` returns a `WorkflowRunStream` of `WorkflowEvent`s as they happen. Call `into_run()` after the stream ends to get the final `WorkflowRun`. The events are `Started`, `Status`, `SuperStepStarted`/`SuperStepCompleted`, `ExecutorInvoked`/`ExecutorCompleted`/`ExecutorFailed`, `AgentRunUpdate` (a streamed agent token, as JSON), `AgentRun` (a finished agent response), `Output`, `Intermediate`, `RequestInfo`, `Custom` and `Failed`.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use futures::StreamExt;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let step = FunctionExecutor::new("step", |m, ctx| async move {
        ctx.add_event(WorkflowEvent::Custom(json!({ "progress": "halfway" })));
        ctx.yield_output(m).await?;
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(step))
        .set_start("step")
        .build()?;

    let mut stream = workflow.run_stream("hello");
    while let Some(event) = stream.next().await {
        match event {
            WorkflowEvent::ExecutorInvoked { executor_id } => println!("running {executor_id}"),
            WorkflowEvent::Custom(data) => println!("custom: {data}"),
            WorkflowEvent::Output { data, .. } => println!("output: {data}"),
            other => println!("{other:?}"),
        }
    }
    let run = stream.into_run().await?;
    println!("final state: {:?}", run.state());
    Ok(())
}
```

## Sub-workflows and visualisation

`WorkflowExecutor::new(id, child_workflow)` runs a whole workflow as one node. The child's outputs are sent on as messages from that node (or yielded directly as parent outputs with `with_direct_output(true)`), and the child's requests are re-raised in the parent under their original ids, so the parent's caller answers them with the usual `send_response`.

`workflow.viz()` renders the graph as Mermaid (`to_mermaid()`) or Graphviz DOT (`to_dot()`), with switch-case labels on the edges. There is no image export; pipe the DOT text to Graphviz yourself.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

fn word_count() -> Result<Workflow> {
    let count = FunctionExecutor::new("count", |m, ctx| async move {
        let words = m.as_str().unwrap_or_default().split_whitespace().count();
        ctx.yield_output(json!(words)).await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .name("word-count")
        .add_executor(Arc::new(count))
        .set_start("count")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let intro = FunctionExecutor::new("intro", |m, ctx| async move {
        ctx.send_message(m).await?;
        Ok(())
    });
    let report = FunctionExecutor::new("report", |m, ctx| async move {
        ctx.yield_output(json!({ "words": m })).await?;
        Ok(())
    });

    let parent = WorkflowBuilder::new()
        .add_executor(Arc::new(intro))
        .add_executor(Arc::new(WorkflowExecutor::new("counter", word_count()?)))
        .add_executor(Arc::new(report))
        .set_start("intro")
        .add_chain(["intro", "counter", "report"].map(String::from))
        .build()?;

    println!("{}", parent.run("sub-workflows nest cleanly").await?.last_output().unwrap());
    println!("{}", parent.viz().to_mermaid());
    Ok(())
}
```

## Agents in workflows

`AgentExecutor::new(id, agent)` wraps any `Arc<dyn SupportsAgentRun>` as a node. It accepts a string, a message or a conversation, runs the agent over it, appends the reply and sends the whole conversation on (or yields it, with `with_output(true)`). Each agent run is also reported as an `AgentRun` event, with `AgentRunUpdate` events when the workflow is streamed.

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

    let verdict = FunctionExecutor::new("verdict", |message, ctx| async move {
        let conversation: Vec<Message> = serde_json::from_value(message)?;
        let last = conversation.last().map(Message::text).unwrap_or_default();
        ctx.yield_output(json!({ "turns": conversation.len(), "verdict": last })).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(AgentExecutor::new("writer", writer)))
        .add_executor(Arc::new(AgentExecutor::new("critic", critic)))
        .add_executor(Arc::new(verdict))
        .set_start("writer")
        .add_chain(["writer", "critic", "verdict"].map(String::from))
        .build()?;

    let run = workflow.run("Why Rust has no garbage collector").await?;
    println!("{}", run.last_output().unwrap());
    Ok(())
}
```

`AgentApprovalExecutor` is the same with a review step: after each agent turn it raises a request, and an empty answer approves the turn while any other answer is fed back to the agent as revision feedback.

## Orchestrations

The orchestration builders assemble a complete workflow from a list of agents. Each takes `Arc<dyn SupportsAgentRun>` participants, and each produces an ordinary `Workflow` whose output is the final conversation as a JSON array of messages. Unlike Python, where each orchestration is a graph of separate executors exchanging envelope messages, the Rust group chat, handoff and Magentic orchestrations are a single executor that loops over its participants inside one superstep. Their semantics match, but you will see one executor id in the events rather than one per agent.

| Builder | Pattern |
| --- | --- |
| `SequentialBuilder` | Each participant sees the conversation so far and appends its reply |
| `ConcurrentBuilder` | Every participant answers the same input; replies are gathered in participant order |
| `GroupChatBuilder` | A manager picks the next speaker each round: round-robin, a closure, a `GroupChatManager` or an agent |
| `HandoffBuilder` | Agents pass control by calling `handoff_to_<name>` tools |
| `MagenticBuilder` | A manager plans, tracks progress in a ledger, and assigns work, with optional plan review |

### Sequential and concurrent

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

fn agent(client: &OpenAIChatCompletionClient, name: &str, instructions: &str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(client.clone()).name(name).instructions(instructions).build())
}

fn print(output: Option<serde_json::Value>) {
    let conversation: Vec<Message> = serde_json::from_value(output.unwrap_or_default()).unwrap_or_default();
    for m in conversation {
        println!("  {}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;

    let pipeline = SequentialBuilder::new()
        .participants(vec![
            agent(&client, "writer", "Write a first draft paragraph on the topic."),
            agent(&client, "editor", "Improve the previous draft for clarity."),
        ])
        .build()?;
    println!("sequential:");
    print(pipeline.run("Rust for embedded systems").await?.last_output());

    let panel = ConcurrentBuilder::new()
        .participants(vec![
            agent(&client, "optimist", "Give the strongest argument for the idea."),
            agent(&client, "sceptic", "Give the strongest argument against the idea."),
        ])
        .build()?;
    println!("concurrent:");
    print(panel.run("Rewrite our Python service in Rust").await?.last_output());
    Ok(())
}
```

Both builders accept `with_request_info()`, which pauses after every participant's turn for approval in the same way as `AgentApprovalExecutor`.

### Group chat

`GroupChatBuilder` registers participants by name (`participant`, or `participant_described` to give a manager agent a description to choose by) and needs a manager: `round_robin()`, `manager_fn(|state| directive)`, `manager(Arc<dyn GroupChatManager>)` or `manager_agent(agent)`. A manager returns `GroupChatDirective::speak(name)`, `speak_with(name, instruction)`, `finish()` or `finish_text(text)`. `max_rounds` caps the conversation (40 rounds if unset) and `termination_condition(|conversation| bool)` ends it early.

A manager agent is prompted with the participant list and asked to answer with a JSON object of the form `{"selected_participant", "instruction", "finish", "final_message"}`; a reply wrapped in a Markdown code fence is accepted too.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::GroupChatDirective;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let writer: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("writer").instructions("Draft a tagline.").build(),
    );
    let critic: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("critic").instructions("Critique the tagline briefly.").build(),
    );

    // A deterministic manager: writer, critic, writer, then stop.
    let scripted = GroupChatBuilder::new()
        .participant("writer", writer.clone())
        .participant("critic", critic.clone())
        .manager_fn(|state| match state.round_index {
            0 | 2 => GroupChatDirective::speak("writer"),
            1 => GroupChatDirective::speak_with("critic", "Be blunt."),
            _ => GroupChatDirective::finish(),
        })
        .build()?;
    let run = scripted.run("A tagline for a coffee shop called Terra").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("scripted: {} messages", conversation.len());

    // An LLM manager choosing between described participants.
    let manager: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(client).name("manager").build());
    let managed = GroupChatBuilder::new()
        .participant_described("writer", "Drafts marketing copy.", writer)
        .participant_described("critic", "Critiques copy for clarity.", critic)
        .manager_agent(manager)
        .max_rounds(6)
        .build()?;
    let run = managed.run("A tagline for a coffee shop called Terra").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("managed: {}", conversation.last().map(Message::text).unwrap_or_default());
    Ok(())
}
```

### Handoff

`HandoffBuilder` runs one agent at a time. When the current agent calls a `handoff_to_<name>` tool, the coordinator records a tool result acknowledging the transfer and switches to that agent. Here the Rust port differs from Python in a way that matters: the builder does not add the handoff tools to your agents for you, because a built `Agent`'s tool list is fixed. Attach them yourself with `handoff_tool_spec(target, description)`, which produces a tool with no executor that the coordinator intercepts.

`add_handoff(source).to([targets])` restricts who may hand off to whom; with no such edges every participant may hand off to every other. In the default `HumanInLoop` mode, an agent that answers without handing off pauses the workflow with a `HandoffUserInputRequest` (the cleaned conversation, the waiting agent and a prompt), and the user's reply (a string, or messages) is sent with `send_response`. The reply goes to the initial agent, not to the agent that was waiting, matching upstream's default of returning to the coordinator. `.autonomous()` finishes on the first answer instead. `max_iterations` caps the number of agent turns (50 by default) and `termination_condition` ends the conversation early.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{handoff_tool_spec, HandoffBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;

    let triage: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone())
            .name("triage")
            .instructions("Route billing questions to billing and technical ones to tech_support.")
            .tool(handoff_tool_spec("billing", Some("Transfer to the billing specialist.")))
            .tool(handoff_tool_spec("tech_support", Some("Transfer to technical support.")))
            .build(),
    );
    let billing: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("billing").instructions("Resolve billing questions.").build(),
    );
    let tech: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client).name("tech_support").instructions("Fix technical problems.").build(),
    );

    let workflow = HandoffBuilder::new()
        .participant("triage", triage)
        .participant("billing", billing)
        .participant("tech_support", tech)
        .initial_agent("triage")
        .add_handoff("triage")
        .to(["billing", "tech_support"])
        .autonomous()
        .build()?;

    let run = workflow.run("I was charged twice this month.").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    for m in &conversation {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
    Ok(())
}
```

The autonomous mode also differs from Python, which re-prompts the same agent until a turn limit: here the workflow completes on the first answer that is not a handoff.

### Magentic

`MagenticBuilder` follows the Magentic-One pattern. A manager first builds a task ledger (facts and a plan); then, each round, it fills in a progress ledger (is the request satisfied, are we looping, is progress being made, who speaks next and with what instruction) and dispatches the next participant. Stalls trigger a re-plan, and the run ends with a final answer from the manager. `StandardMagenticManager::new(agent)` drives all of this with an LLM, using prompts ported from upstream; implement the `MagenticManager` trait to control it yourself. `max_round_count`, `max_stall_count` and `max_reset_count` bound the run.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai%%")?;
    let researcher: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("researcher").instructions("Find and summarise facts.").build(),
    );
    let writer: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("writer").instructions("Write a short final answer.").build(),
    );
    let manager_agent: Arc<dyn SupportsAgentRun> =
        Arc::new(Agent::builder(client).name("magentic_manager").build());

    let workflow = MagenticBuilder::new()
        .participant_described("researcher", "Finds and summarises facts.", researcher)
        .participant_described("writer", "Writes polished answers.", writer)
        .standard_manager(StandardMagenticManager::new(manager_agent).max_round_count(10))
        .with_plan_review()
        .build()?;

    let mut run = workflow.run("When was the Eiffel Tower completed, and why?").await?;

    // with_plan_review(): the run pauses with the plan before any work starts.
    while let Some(request) = run.pending_requests().into_iter().next() {
        let review: MagenticPlanReviewRequest = serde_json::from_value(request.request_data)?;
        println!("plan (round {}): {}", review.round, review.plan);
        let decision = MagenticPlanReviewDecision::approve();
        run.send_response(request.request_id, serde_json::to_value(decision)?).await?;
    }

    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap_or_default())?;
    println!("{}", conversation.last().map(Message::text).unwrap_or_default());
    Ok(())
}
```

A plan review can also be answered with `approve_with_comments`, `approve_with_edited_plan`, `revise_with_comments` or `revise_with_edited_plan`; `max_plan_review_rounds` caps the back and forth. `with_stall_intervention()` pauses with a `MagenticStallInterventionRequest` when the manager detects a stall, to be answered with a `MagenticStallInterventionDecision` (`continue_as_is`, `replan`, `replan_with_guidance` or `abort`).

## A workflow as an agent

`workflow.as_agent(name)` (from the `WorkflowAgentExt` trait in the prelude) wraps any workflow as a `WorkflowAgent`, which implements `SupportsAgentRun`. It can then be a participant in another orchestration, a tool (`as_tool()`), or an agent served by the [hosting crate](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#hosting-agents-over-http). The workflow's outputs become the agent's response messages, and streaming a `WorkflowAgent` streams the participants' updates.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let pipeline = SequentialBuilder::new()
        .participants(vec![
            Arc::new(Agent::builder(client.clone()).name("drafter").build()) as Arc<dyn SupportsAgentRun>,
            Arc::new(Agent::builder(client).name("polisher").build()) as Arc<dyn SupportsAgentRun>,
        ])
        .build()?;

    let agent = pipeline.as_agent("draft-and-polish");
    let response = agent.run(vec![Message::user("A haiku about lifetimes")], None).await?;
    println!("{} message(s); last: {}", response.messages.len(), response.messages.last().map(Message::text).unwrap_or_default());
    Ok(())
}
```

## What is not here yet

Compared with the Python engine, the port does not have typed executor handlers with routing by message type, upstream's envelope types such as `AgentExecutorRequest`, interception of sub-workflow requests by the parent, workflow run cancellation, functional (`@workflow` / `@step`) workflows, or the Power Fx declarative workflow language (the [declarative crate](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/#declarative-agents-and-workflows) has its own workflow format instead). The repository's README and `SCOPE.md` keep the current list.

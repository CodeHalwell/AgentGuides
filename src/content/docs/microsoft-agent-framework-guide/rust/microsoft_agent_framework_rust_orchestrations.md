---
title: "Microsoft Agent Framework (Rust) — Orchestrations"
description: "The prebuilt multi-agent patterns in agent-framework-rs: sequential, concurrent, group chat, handoff and Magentic orchestrations, with approval gates, plan review, stall intervention, streaming updates and what is not ported."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Orchestrations"
  order: 9
---

The orchestration builders assemble a complete [workflow](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/) from a list of agents. Each takes `Arc<dyn SupportsAgentRun>` participants (an `Agent`, a `WorkflowAgent`, an A2A or Copilot Studio agent, or your own implementation) and returns an ordinary `Workflow`, which you run, stream, answer requests on and wrap as an agent exactly as on the workflows page.

All examples were compiled against `agent-framework` %%version=agent-framework-rs%%. Those with OpenAI agents were run against a local stand-in for the OpenAI API, so the printed replies (`Mock reply to: ...`) are not real model output; the Magentic stall example uses a scripted manager and runs offline. The builders and their request and decision types are in the prelude, apart from `HandoffBuilder`'s helpers, `ApprovalRequest`, `AgentApprovalExecutor` and the Magentic ledger types, which are in `agent_framework::workflow`. Nothing here needs an `experimental-*` feature.

## How the Rust orchestrations differ

In Python, each orchestration is a graph of separate executors (an orchestrator plus one executor per participant) exchanging envelope messages. In Rust, sequential and concurrent are graphs of agent executors, but group chat, handoff and Magentic are each a **single executor** that calls its participants in a loop inside one superstep. The behaviour matches upstream; what you see in events differs:

| Builder | Executor ids in the built workflow |
| --- | --- |
| `SequentialBuilder` | `seq_0`, `seq_1`, ... one per participant, chained |
| `ConcurrentBuilder` | `dispatch`, `agent_0`, `agent_1`, ..., `aggregate` |
| `GroupChatBuilder` | `group_chat_orchestrator` |
| `HandoffBuilder` | `handoff_coordinator` |
| `MagenticBuilder` | `magentic_orchestrator` |

So in a group chat every `AgentRunUpdate` and `AgentRun` event carries the executor id `group_chat_orchestrator`, and you tell participants apart by the `author_name` on the update. A whole group chat, handoff or Magentic run is also one superstep (or one per human answer), so it does not use up the workflow's superstep limit, and its events reach a stream only when the loop pauses or finishes (see [streaming updates](#streaming-updates)).

Every orchestration drives its participants through `run_stream`, so a test double must implement the chat client's streaming method (the group chat and Magentic manager agents are called with `run`). No session is passed to the participants: the conversation the orchestration builds is all the context they get.

## Choosing a pattern

| Builder | Pattern | Output | Human input |
| --- | --- | --- | --- |
| [`SequentialBuilder`](#sequential) | Each participant sees the conversation so far and appends its reply | The full conversation | Optional approval after each turn |
| [`ConcurrentBuilder`](#concurrent) | Every participant answers the same input at once | The prompt plus each participant's last message | Optional approval after each turn |
| [`GroupChatBuilder`](#group-chat) | A manager picks the next speaker each round | The full conversation, plus the manager's closing message | None built in |
| [`HandoffBuilder`](#handoff) | Agents pass control by calling `handoff_to_<name>` tools | The conversation without tool plumbing | By default, after every answer |
| [`MagenticBuilder`](#magentic) | A manager plans, tracks progress in a ledger and assigns work | The manager's final answer only | Optional plan review and stall intervention |

Every output is a JSON array of `Message`; deserialise it with `serde_json::from_value::<Vec<Message>>`.

## Sequential

`SequentialBuilder` chains its participants: the first receives the input, each appends its reply and passes the whole conversation on, and the last yields the conversation as the workflow output. Add participants with `participants(iter)` (which replaces any already added) or `add(agent)` one at a time; `name` sets the workflow name; `build()` fails if there are no participants.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

fn agent(client: &OpenAIChatCompletionClient, id: &str, instructions: &str) -> Arc<dyn SupportsAgentRun> {
    // output_from names participants by agent id, so set one.
    Arc::new(Agent::builder(client.clone()).id(id).name(id).instructions(instructions).build())
}

fn print(conversation: &[Message]) {
    for m in conversation {
        println!("  {}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;

    let pipeline = SequentialBuilder::new()
        .name("write-edit-check")
        .add(agent(&client, "writer", "Write a first draft paragraph on the topic."))
        .add(agent(&client, "editor", "Improve the previous draft for clarity."))
        .add(agent(&client, "checker", "List any factual claims that need a source."))
        .build()?;
    let run = pipeline.run("Rust for embedded systems").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("default: one output, {} messages", conversation.len());
    print(&conversation);

    // Make the editor's conversation the result and treat the writer's as progress.
    let redesignated = SequentialBuilder::new()
        .participants(vec![
            agent(&client, "writer", "Write a first draft paragraph on the topic."),
            agent(&client, "editor", "Improve the previous draft for clarity."),
            agent(&client, "checker", "List any factual claims that need a source."),
        ])
        .output_from(["editor"])
        .intermediate_output_from(["writer"])
        .build()?;
    let run = redesignated.run("Rust for embedded systems").await?;
    for event in run.events() {
        match event {
            WorkflowEvent::Output { source_executor_id, .. } => println!("Output from {source_executor_id}"),
            WorkflowEvent::Intermediate { source_executor_id, .. } => println!("Intermediate from {source_executor_id}"),
            _ => {}
        }
    }
    Ok(())
}
```

```text
default: one output, 4 messages
  user: Rust for embedded systems
  writer: Mock reply to: Rust for embedded systems
  editor: Mock reply to: Rust for embedded systems
  checker: Mock reply to: Rust for embedded systems
Intermediate from seq_0
Output from seq_1
```

Each participant receives every earlier message, including the other agents' replies as assistant messages. Some providers answer an assistant-ending prompt with empty text; upstream added `ensure_trailing_user_turn` for that, which is not ported, so give later participants instructions that make sense after an assistant turn, or put a function executor in between in a hand-built graph.

### Choosing which participants produce output

`output_from(ids)` and `intermediate_output_from(ids)` work as on the [workflows page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#outputs), with a difference in what you name:

| Builder | Names | Effect |
| --- | --- | --- |
| `SequentialBuilder`, `ConcurrentBuilder` | Agent ids (`agent.id()`) | Each named participant yields its conversation; the rest are silent or demoted |
| `GroupChatBuilder`, `MagenticBuilder` | Participant names | Naming any participant applies to the one orchestrator executor: `output_from` keeps its result terminal (the default anyway), `intermediate_output_from` demotes it |
| `HandoffBuilder` | (not available) | |

An `Agent`'s id is a random UUID unless you set one with `Agent::builder(...).id("writer")`, so for sequential and concurrent either set ids or read `agent.id()` before handing the agent over. An unknown id fails `build()` with `output_from references unknown participant agent id '...'`. When any designation is set on a sequential workflow, the last participant stops yielding unless it is named. On a concurrent workflow, naming participants in `output_from` also demotes the aggregator's merged result to `Intermediate`, because the aggregator is not one of the named executors; `intermediate_output_from` alone leaves it terminal. The group chat and Magentic forms exist mainly to demote the result when the orchestration feeds a later stage.

## Concurrent

`ConcurrentBuilder` fans the input out to every participant and gathers the replies through a fan-in. The participants run in the same superstep, so their model calls overlap. It has the same `participants`, `add`, `name`, `output_from`, `intermediate_output_from` and `with_request_info` methods as `SequentialBuilder`.

The built-in aggregator merges the participants' conversations into one: the conversation from the first participant minus its last message (normally just the prompt), followed by the last message of each participant, in participant order. If an agent's reply spans several messages (a tool call, its result and the answer, say), only the final message survives. There is no option to plug in a different aggregator (Python's `with_aggregator` is not ported), but the same shape is a few lines of `WorkflowBuilder`: a dispatch node, an `AgentExecutor` per participant, and a fan-in into your own function executor, which receives one conversation per participant in declared order.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{AgentExecutor, FunctionExecutor};
use serde_json::{json, Value};

fn agent(client: &OpenAIChatCompletionClient, name: &str, instructions: &str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(client.clone()).name(name).instructions(instructions).build())
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let panel = || {
        vec![
            agent(&client, "optimist", "Give the strongest argument for the idea."),
            agent(&client, "sceptic", "Give the strongest argument against the idea."),
            agent(&client, "accountant", "Estimate the cost in one sentence."),
        ]
    };

    // Built-in aggregation: the prompt, then each participant's last message
    // in participant order.
    let workflow = ConcurrentBuilder::new().participants(panel()).build()?;
    let run = workflow.run("Rewrite our Python service in Rust").await?;
    let merged: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    for m in &merged {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }

    // A custom aggregator is a fan-in node of your own.
    let dispatch = FunctionExecutor::new("dispatch", |m, ctx| async move {
        ctx.send_message(m).await?;
        Ok(())
    });
    let summarise = FunctionExecutor::new("summarise", |message, ctx| async move {
        // One conversation per participant, in the order the sources were declared.
        let conversations: Vec<Vec<Message>> = serde_json::from_value(message)?;
        let verdicts: Vec<Value> = conversations
            .iter()
            .filter_map(|c| c.last())
            .map(|m| json!({ "who": m.author_name, "said": m.text() }))
            .collect();
        ctx.yield_output(json!({ "votes": verdicts.len(), "verdicts": verdicts })).await?;
        Ok(())
    });
    let ids: Vec<String> = ["optimist", "sceptic", "accountant"].map(String::from).to_vec();
    let mut builder = WorkflowBuilder::new()
        .add_executor(Arc::new(dispatch))
        .add_executor(Arc::new(summarise))
        .set_start("dispatch");
    for (id, a) in ids.iter().zip(panel()) {
        builder = builder.add_executor(Arc::new(AgentExecutor::new(id.clone(), a)));
    }
    let custom = builder
        .add_fan_out("dispatch", ids.clone())
        .add_fan_in(ids, "summarise")
        .build()?;
    let run = custom.run("Rewrite our Python service in Rust").await?;
    println!("{}", run.last_output().unwrap());
    Ok(())
}
```

```text
user: Rewrite our Python service in Rust
optimist: Mock reply to: Rewrite our Python service in Rust
sceptic: Mock reply to: Rewrite our Python service in Rust
accountant: Mock reply to: Rewrite our Python service in Rust
{"verdicts":[{"said":"Mock reply to: Rewrite our Python service in Rust","who":"optimist"},{"said":"Mock reply to: Rewrite our Python service in Rust","who":"sceptic"},{"said":"Mock reply to: Rewrite our Python service in Rust","who":"accountant"}],"votes":3}
```

If one participant fails, the whole run fails; there is no partial result. The fan-in waits for every participant, so a single slow agent sets the pace.

## Group chat

`GroupChatBuilder` keeps one running conversation. Before each turn a manager looks at the state and either names the next speaker (optionally with an instruction, appended to the conversation as an assistant message authored by the manager) or finishes.

| Method | Effect |
| --- | --- |
| `participant(name, agent)` | Register a participant; its description defaults to its name |
| `participant_described(name, description, agent)` | Register with a description, which an LLM manager chooses by |
| `participants(iter of (String, agent))` | Register several; adds to any already registered |
| `round_robin()` | Take turns in registration order (the default; the same as `manager(Arc::new(RoundRobinManager::new()))`) |
| `manager_fn(\|state\| directive)` | A synchronous closure |
| `manager(Arc<dyn GroupChatManager>)` | Your own manager type, which may be async |
| `manager_agent(agent)` | An LLM agent chooses, with prompts ported from Python |
| `max_rounds(n)` | Cap on participant turns; `DEFAULT_GROUP_CHAT_MAX_ITERATIONS` (40) if unset |
| `termination_condition(\|conversation\| bool)` | Checked before every manager decision |
| `manager_name(name)` | `author_name` for the manager's messages |
| `name`, `output_from`, `intermediate_output_from` | As for the other builders |

A manager returns a `GroupChatDirective`: `speak(name)`, `speak_with(name, instruction)`, `finish()` (closes with "Conversation completed."), `finish_with(message)` or `finish_text(text)`. It sees a `GroupChatState` with `task` (the last input message), `participants` as `(name, description)` pairs in registration order, the whole `conversation`, and `round_index` (the number of participant turns so far). Naming a participant that does not exist fails the run with `manager selected unknown participant '...'`.

When `max_rounds` is reached the conversation ends with "Conversation halted after reaching manager round limit."; when the termination condition holds it ends with "Conversation halted after termination condition was met.". Both are appended as assistant messages from the manager, and neither is an error. The manager's display name is `round_robin_manager`, `custom_manager` (for `manager_fn`), `manager` (for `manager_agent`), or whatever `GroupChatManager::name` returns, unless `manager_name` overrides it.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

fn agent(client: &OpenAIChatCompletionClient, name: &str, instructions: &str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(client.clone()).name(name).instructions(instructions).build())
}

fn transcript(run: &WorkflowRun) -> Vec<Message> {
    serde_json::from_value(run.last_output().unwrap()).unwrap()
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let writer = agent(&client, "writer", "Draft or revise a tagline.");
    let critic = agent(&client, "critic", "Critique the tagline briefly. Say LGTM when it is good.");

    // Round-robin is the default manager; max_rounds counts participant turns.
    let round_robin = GroupChatBuilder::new()
        .participant("writer", writer.clone())
        .participant("critic", critic.clone())
        .max_rounds(3)
        .build()?;
    let run = round_robin.run("A tagline for a coffee shop called Terra").await?;
    for m in transcript(&run) {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }

    // A closure manager, plus a termination condition checked before every turn.
    let scripted = GroupChatBuilder::new()
        .participants([("writer".to_string(), writer), ("critic".to_string(), critic)])
        .manager_fn(|state| match state.round_index {
            0 | 2 => GroupChatDirective::speak("writer"),
            1 => GroupChatDirective::speak_with("critic", "Be blunt."),
            _ => GroupChatDirective::finish_text("Done after three turns."),
        })
        .manager_name("editor_in_chief")
        .termination_condition(|conversation| conversation.iter().any(|m| m.text().contains("LGTM")))
        .build()?;
    let run = scripted.run("A tagline for a coffee shop called Terra").await?;
    let conversation = transcript(&run);
    println!("\nscripted: {} messages", conversation.len());
    for m in &conversation {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
    Ok(())
}
```

```text
user: A tagline for a coffee shop called Terra
writer: Mock reply to: A tagline for a coffee shop called Terra
critic: Mock reply to: A tagline for a coffee shop called Terra
writer: Mock reply to: A tagline for a coffee shop called Terra
round_robin_manager: Conversation halted after reaching manager round limit.

scripted: 6 messages
user: A tagline for a coffee shop called Terra
writer: Mock reply to: A tagline for a coffee shop called Terra
editor_in_chief: Be blunt.
critic: Mock reply to: A tagline for a coffee shop called Terra
writer: Mock reply to: A tagline for a coffee shop called Terra
editor_in_chief: Done after three turns.
```

Python's group chat has no built-in round-robin manager and no default round cap (only the workflow's superstep limit); both are Rust conveniences. Round-robin never finishes by itself, so it always ends at `max_rounds` or the termination condition.

### Custom and LLM managers

Implement `GroupChatManager` (`async fn select(&self, state) -> Result<GroupChatDirective>`, plus an optional `name`) when the choice needs async work or state of its own. Returning `Err` fails the run.

`manager_agent(agent)` uses an agent as the manager. Each round it is sent a system message containing `DEFAULT_MANAGER_INSTRUCTIONS` (ported verbatim from Python), the participant list as `- name: description` lines, and a request for a JSON object of the form `{"selected_participant", "instruction", "finish", "final_message"}` (`ManagerSelectionResponse`), followed by the conversation. The reply is read from the response's structured `value` if there is one, else from the text, else from the last Markdown code fence in the text, since many models wrap JSON in one. An unparseable reply, or `finish: false` with no participant, fails the run. Names are case-sensitive. The manager agent's own instructions are added by the agent as usual, so keep them short; the routing prompt is supplied for you. `DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT` is also exported, for reuse in a custom manager.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// A manager that picks whoever has spoken least (ties go to the earlier
/// participant) and finishes once everyone has had `turns` turns.
struct Quota {
    turns: usize,
}

#[async_trait]
impl GroupChatManager for Quota {
    async fn select(&self, state: &GroupChatState) -> Result<GroupChatDirective> {
        let spoken = |name: &str| {
            state.conversation.iter().filter(|m| m.author_name.as_deref() == Some(name)).count()
        };
        let next = state
            .participants
            .iter()
            .enumerate()
            .map(|(position, (name, _))| (spoken(name), position, name))
            .filter(|(n, _, _)| *n < self.turns)
            .min();
        Ok(match next {
            Some((_, _, name)) => GroupChatDirective::speak(name.clone()),
            None => GroupChatDirective::finish_text(format!(
                "Finished after {} turns on: {}",
                state.round_index,
                state.task.as_ref().map(Message::text).unwrap_or_default()
            )),
        })
    }

    fn name(&self) -> &str {
        "quota_manager"
    }
}

fn agent(client: &OpenAIChatCompletionClient, name: &str, instructions: &str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(client.clone()).name(name).instructions(instructions).build())
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let writer = agent(&client, "writer", "Draft marketing copy.");
    let critic = agent(&client, "critic", "Critique copy for clarity.");

    let quota = GroupChatBuilder::new()
        .participant("writer", writer.clone())
        .participant("critic", critic.clone())
        .manager(Arc::new(Quota { turns: 1 }))
        .build()?;
    let run = quota.run("A tagline for Terra coffee").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    for m in &conversation {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }

    // An LLM manager picks speakers from the participant descriptions.
    let manager: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(client).name("manager").build());
    let managed = GroupChatBuilder::new()
        .participant_described("writer", "Drafts marketing copy.", writer)
        .participant_described("critic", "Critiques copy for clarity.", critic)
        .manager_agent(manager)
        .max_rounds(6)
        .build()?;
    let run = managed.run("A tagline for Terra coffee").await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("\nLLM-managed:");
    for m in &conversation {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
    Ok(())
}
```

```text
user: A tagline for Terra coffee
writer: Mock reply to: A tagline for Terra coffee
critic: Mock reply to: A tagline for Terra coffee
quota_manager: Finished after 2 turns on: A tagline for Terra coffee

LLM-managed:
user: A tagline for Terra coffee
writer: Mock reply to: A tagline for Terra coffee
manager: Agreed final answer.
```

When an LLM manager finishes with a `final_message`, that text is the last message of the output, which makes it a convenient place to ask for a summary.

## Handoff

`HandoffBuilder` runs one agent at a time. When the current agent calls a `handoff_to_<name>` tool, the coordinator appends a tool result (`{"handoff_to": "<name>"}`) and the named agent takes the next turn over the same conversation. When an agent answers without handing off, the turn is over: in the default `HumanInLoop` mode the workflow pauses for the user's next message, and in autonomous mode it finishes.

| Method | Effect |
| --- | --- |
| `participant(name, agent)`, `participants(iter)` | Register participants |
| `initial_agent(name)` / `coordinator(name)` | Who receives user input; the first participant if unset |
| `add_handoff(source).to([targets])` | Restrict who may hand off to whom (see below); `add_handoff` returns a `HandoffEdgeBuilder` whose `to` hands the builder back |
| `with_user_input_request()` / `interaction_mode(HandoffInteractionMode::HumanInLoop)` | Pause for the user after each answer (the default) |
| `autonomous()` / `interaction_mode(HandoffInteractionMode::Autonomous)` | Finish on the first answer that is not a handoff |
| `max_iterations(n)` | Cap on agent turns per user message, default 50; `0` becomes `1` |
| `termination_condition(\|conversation\| bool)` | Finish when it holds; by default after 10 user messages |
| `request_prompt(text)` | The prompt in each user-input request; default "Provide your next input for the conversation." |
| `name(text)` | The workflow name |

### Handoff tools

Here the port differs from Python in a way that matters: the builder does not add handoff tools to your agents, because a built `Agent`'s tool list is fixed. Attach them yourself with `handoff_tool_spec(target, description)`, which returns a declaration-only `ToolDefinition`: it has a JSON schema (an optional `context` string) but no executor. When the model calls it, the agent's function-calling loop returns the call to its caller instead of running it, and the coordinator reads it from the response. `handoff_tool_name(target)` gives the name it uses: `handoff_to_` plus the target lowercased, with runs of other characters turned into `_`, cut to the 64 characters providers accept. `build()` rejects two participants whose tool names collide after that.

The coordinator accepts more than its own tool names. Any function call named `handoff_to_x`, `handoff to x` or `transfer_to_x` (any case, `_`, `-` or space between the words), or any call with a `handoff_to` string argument, is treated as a handoff to `x`, matched against participant names and their sanitised forms. If a response contains several handoff calls, the first wins and the rest are logged. A handoff to a name that does not resolve, or that the source may not reach, is not an error: the coordinator appends a tool result with `Error: unknown handoff target '...'` and the same agent runs again, so it can correct itself.

Routing rules come from `add_handoff`. With no `add_handoff` calls, every participant may hand off to every other. Once any is declared, the declarations are authoritative: a source may hand off only to the targets listed for it, and a participant with no declaration cannot hand off at all. Remember to attach the matching tools as well; the routing rules do not add or remove tools, they only decide which calls are honoured.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{handoff_tool_name, handoff_tool_spec, HandoffBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;

    // The handoff tools are declaration-only: the agent's tool loop returns
    // the call instead of running it, and the coordinator acts on it.
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
    println!("tool names: {}, {}", handoff_tool_name("billing"), handoff_tool_name("Tech Support (EU)"));

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
    // Who actually answered?
    for event in run.events() {
        if let WorkflowEvent::AgentRun { response, .. } = event {
            let r: AgentResponse = serde_json::from_value(response.clone())?;
            let calls: Vec<String> = r.messages.iter().flat_map(|m| m.contents.iter()).filter_map(|c| match c {
                Content::FunctionCall(fc) => Some(fc.name.clone()),
                _ => None,
            }).collect();
            println!("turn by {:?}: calls {calls:?}", r.messages.first().and_then(|m| m.author_name.clone()));
        }
    }
    Ok(())
}
```

```text
tool names: handoff_to_billing, handoff_to_tech_support_eu
user: I was charged twice this month.
billing: Here is what the tool returned: {"handoff_to":"billing"}
turn by Some("triage"): calls ["handoff_to_billing"]
turn by Some("billing"): calls []
```

The output conversation is cleaned: tool messages, function calls and results are removed, and so is any assistant message left empty (triage's turn above, which was only a tool call). User messages keep their images and files; non-text content on other messages is dropped. The `AgentRun` events still have every turn in full. The stand-in server echoes the tool result, which is why billing's reply quotes it.

### Human in the loop

In the default mode, an agent that answers without handing off pauses the run with a request whose payload is a `HandoffUserInputRequest`: the cleaned `conversation`, the `awaiting_agent` that answered, and the `prompt`. The answer to `send_response` can be a string, a message, an array of messages (any role, converted to user messages), or an object with a `text` or `content` string. It is appended as user input and goes to the initial agent, not to the agent that was waiting, which matches upstream's default of returning to the coordinator. The termination condition is checked when the answer arrives, before any agent runs, so a "bye" ends the conversation without another model call.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{handoff_tool_spec, HandoffBuilder, HandoffUserInputRequest};
use serde_json::json;

fn build(client: &OpenAIChatCompletionClient) -> Result<Workflow> {
    let triage: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone())
            .name("triage")
            .instructions("Hand billing questions to billing.")
            .tool(handoff_tool_spec("billing", None))
            .build(),
    );
    let billing: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("billing").instructions("Resolve billing questions.").build(),
    );
    HandoffBuilder::new()
        .participant("triage", triage)
        .participant("billing", billing)
        .initial_agent("triage")
        // HumanInLoop is the default; this says so explicitly.
        .with_user_input_request()
        .request_prompt("Anything else for the support team?")
        .max_iterations(6)
        .termination_condition(|conversation| {
            conversation.last().is_some_and(|m| m.role == Role::user() && m.text().contains("bye"))
        })
        .name("support-desk")
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let workflow = build(&client)?;

    let mut run = workflow.run("I was charged twice.").await?;
    for reply in [json!("It was on the 3rd of May."), json!({ "text": "Thanks, bye" })] {
        let pending = run.pending_requests().remove(0);
        let request: HandoffUserInputRequest = serde_json::from_value(pending.request_data)?;
        println!(
            "[{}] {} (last: {})",
            request.awaiting_agent,
            request.prompt,
            request.conversation.last().map(Message::text).unwrap_or_default()
        );
        run.send_response(pending.request_id, reply).await?;
    }
    println!("{:?}", run.state());
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    for m in &conversation {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
    Ok(())
}
```

```text
[billing] Anything else for the support team? (last: Here is what the tool returned: {"handoff_to":"billing"})
[billing] Anything else for the support team? (last: Here is what the tool returned: {"handoff_to":"billing"})
Idle
user: I was charged twice.
billing: Here is what the tool returned: {"handoff_to":"billing"}
user: It was on the 3rd of May.
billing: Here is what the tool returned: {"handoff_to":"billing"}
user: Thanks, bye
```

When `max_iterations` agent turns pass without a final answer (agents handing off back and forth, say), the workflow finishes with the conversation so far; it is not an error. The count restarts with each user message. The paused conversation is kept in the run's shared state under `_handoff_state`.

Two differences from Python: autonomous mode completes on the first answer that is not a handoff, where Python re-prompts the same agent until a turn limit; and turn limits are global, where Python's `with_autonomous_mode(agents=..., turn_limits=...)` sets them per agent.

## Magentic

`MagenticBuilder` follows the Magentic-One pattern. The manager first builds a task ledger: a fact sheet and a plan, from two model calls. Then, each round, it fills in a progress ledger (`MagenticProgressLedger`: is the request satisfied, is the team in a loop, is progress being made, who speaks next, and with what instruction), appends the instruction to the conversation, and runs the chosen participant over the whole history. When the ledger says the request is satisfied, the manager writes a final answer and the run ends.

Stalls are counted: a round with no progress, or in a loop, adds one; a productive round takes one off. When the count exceeds `max_stall_count`, the manager resets (clears the history, keeping the task and the previous ledger) and replans. The run also ends early when `max_round_count` rounds have run or `max_reset_count` replans have happened; it then yields the most recent assistant message as a partial result, or "Stopped due to round limit. No partial result available.".

| Setting | `StandardMagenticManager` | `MagenticBuilder` | Default |
| --- | --- | --- | --- |
| Rounds | `max_round_count(n)` | `max_round_count(n)` | Unlimited |
| Stall threshold | `max_stall_count(n)` | `max_stall_count(n)` | 3 |
| Replans | `max_reset_count(n)` | `max_reset_count(n)` | Unlimited |
| Ledger parse retries | `progress_ledger_retry_count(n)` | | 3 (minimum 1) |
| Plan review | | `with_plan_review()`, `max_plan_review_rounds(n)` | Off; 10 rounds |
| Stall intervention | | `with_stall_intervention()` | Off |

A builder setting overrides the manager's. With no round or reset limit, a run that never satisfies the ledger is bounded only by the stall and replan cycle, which can go on for a long time and many model calls; set `max_round_count`.

`StandardMagenticManager::new(agent)` drives all of this with an LLM, using the Magentic-One prompts ported from Python (exported as the `ORCHESTRATOR_*` constants). It needs no instructions of its own. The progress ledger is parsed from the reply's first fenced JSON object, else the first balanced `{...}` (Python-style `True`, `False` and `None` are accepted); a ledger that does not parse, or that names an unknown speaker, is retried up to the retry count. If every attempt fails to parse, the orchestrator resets and replans; if the ledger names an unknown speaker, the manager writes the final answer. Register participants with `participant_described` so the manager knows what each one does.

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
    // The manager needs no instructions: StandardMagenticManager supplies the
    // Magentic-One prompts.
    let manager_agent: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(client).name("magentic_manager").build());

    let manager = StandardMagenticManager::new(manager_agent)
        .max_round_count(10)
        .max_stall_count(2)
        .max_reset_count(1)
        .progress_ledger_retry_count(2);

    let workflow = MagenticBuilder::new()
        .participant_described("researcher", "Finds and summarises facts.", researcher)
        .participant_described("writer", "Writes polished answers.", writer)
        .standard_manager(manager)
        .build()?;

    let run = workflow.run("When was the Eiffel Tower completed, and why?").await?;
    // The output is the final answer only, as a one-message conversation.
    let answer: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("{} message from {:?}", answer.len(), answer[0].author_name);

    // The ledger and instructions appear as AgentRunUpdate events carrying a
    // Message, from the orchestrator itself.
    for event in run.events() {
        if let WorkflowEvent::AgentRunUpdate { executor_id, update } = event {
            if executor_id == "magentic_orchestrator" {
                if let Ok(m) = serde_json::from_value::<Message>(update.clone()) {
                    let first_line = m.text().trim().lines().next().unwrap_or_default().to_string();
                    println!("[{} {}] {first_line}", m.role.as_str(), m.author_name.as_deref().unwrap_or("-"));
                }
            }
        }
    }
    Ok(())
}
```

```text
1 message from Some("magentic_manager")
[user -] When was the Eiffel Tower completed, and why?
[assistant magentic_manager] We are working to address the following user request:
```

Unlike the other orchestrations, the output is not the conversation: it is a one-element array holding the manager's final answer, authored `magentic_manager` (`MAGENTIC_MANAGER_NAME`). The working history is visible only through events. The orchestrator emits its own messages (the task, each task ledger, each instruction) as `AgentRunUpdate` events whose `update` is a serialised `Message`, not an `AgentResponseUpdate`; the participants' updates are `AgentResponseUpdate`s as usual. Here the stand-in reported the request satisfied at once, so no participant ran.

Take the manager by value with `standard_manager(manager)`, or share one with `manager(Arc<dyn MagenticManager>)`. `StandardMagenticManager` keeps no run state (the task ledger lives in the run's `MagenticContext`), so one instance can serve concurrent runs.

### Custom managers

Implement `MagenticManager` to drive the loop yourself:

| Method | Called |
| --- | --- |
| `plan(&mut MagenticContext) -> Result<Message>` | Once at the start; return the combined task ledger |
| `replan(&mut MagenticContext) -> Result<Message>` | After a reset, and for plan-review feedback |
| `create_progress_ledger(&MagenticContext) -> Result<MagenticProgressLedger>` | Every round; an `Err` triggers a reset and replan |
| `prepare_final_answer(&MagenticContext) -> Result<Message>` | When the request is satisfied, the next speaker is unknown, or a human aborts |
| `max_stall_count()`, `max_reset_count()`, `max_round_count()` | Defaults: 3, `None`, `None` |

`MagenticContext` carries the `task`, `chat_history`, `participant_descriptions`, `round_count`, `stall_count`, `reset_count` and an optional `task_ledger` (`MagenticTaskLedger { facts, plan }`). Set `task_ledger` in `plan` and `replan` if you want plan review and stall intervention to show the facts and plan separately; without it, plan review shows the combined message as the plan and stall intervention shows neither. Build a `MagenticProgressLedger` with `serde_json::from_value` (each field is a `MagenticProgressLedgerItem { reason, answer }`, with `answer_bool()` and `answer_str()` helpers). The [stall intervention example](#stall-intervention) uses a scripted manager.

### Plan review

`with_plan_review()` pauses after the initial plan, before any participant runs, with a `MagenticPlanReviewRequest` (`task`, `facts`, `plan`, `round`). Answer with a serialised `MagenticPlanReviewDecision`:

| Decision | Effect |
| --- | --- |
| `approve()` | Proceed with the plan as it is |
| `approve_with_edited_plan(text)` | Adopt `text` as the plan, with no model call, and proceed |
| `approve_with_comments(text)` | Add the comments as "Human plan feedback: ..." and replan once, then proceed without further review |
| `revise_with_edited_plan(text)` | Adopt `text` and ask again |
| `revise_with_comments(text)` | Add the comments, replan and ask again |

The decision serialises with a `decision` tag: `{"decision":"approve"}`, `{"decision":"revise","comments":"..."}`. If both an edited plan and comments are present, the edit wins. After `max_plan_review_rounds` revisions the orchestrator stops asking: it appends "Plan review closed after max rounds. ..." and proceeds with the current plan, ignoring that final revision. Review happens only once, for the initial plan; replans after a stall are not reviewed, matching Python.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai%%")?;
    let researcher: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("researcher").instructions("Find and summarise facts.").build(),
    );
    let manager_agent: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(client).name("magentic_manager").build());

    let workflow = MagenticBuilder::new()
        .participant_described("researcher", "Finds and summarises facts.", researcher)
        .standard_manager(StandardMagenticManager::new(manager_agent))
        .with_plan_review()
        .max_plan_review_rounds(3)
        .build()?;

    let mut run = workflow.run("Summarise the history of the Eiffel Tower.").await?;
    let mut decisions = vec![
        MagenticPlanReviewDecision::revise_with_comments("Cover the 1889 World's Fair."),
        MagenticPlanReviewDecision::revise_with_edited_plan("1. researcher: dates and purpose. 2. Answer."),
        MagenticPlanReviewDecision::approve(),
    ]
    .into_iter();

    while run.state() == WorkflowRunState::IdleWithPendingRequests {
        let pending = run.pending_requests().remove(0);
        let review: MagenticPlanReviewRequest = serde_json::from_value(pending.request_data)?;
        println!("round {}: plan = {}", review.round, review.plan.chars().take(50).collect::<String>());
        let decision = decisions.next().expect("one decision per round");
        println!("  answering {}", serde_json::to_value(&decision)?);
        run.send_response(pending.request_id, serde_json::to_value(decision)?).await?;
    }
    let answer: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("{:?}: {}", run.state(), answer[0].text().chars().take(60).collect::<String>());
    Ok(())
}
```

```text
round 0: plan = Mock reply to: Fantastic. To address this request 
  answering {"comments":"Cover the 1889 World's Fair.","decision":"revise"}
round 1: plan = Mock reply to: Please briefly explain what went wr
  answering {"decision":"revise","edited_plan":"1. researcher: dates and purpose. 2. Answer."}
round 2: plan = 1. researcher: dates and purpose. 2. Answer.
  answering {"decision":"approve"}
Idle: Mock reply to: 
We are working on the following task:
Summar
```

A response that does not deserialise as a decision fails the run with `invalid plan-review decision`.

### Stall intervention

`with_stall_intervention()` asks a person instead of replanning automatically when the stall count passes the threshold. The run pauses with a `MagenticStallInterventionRequest` (`task`, `reason`, `stall_count`, `max_stall_count`, `round`, `resets_so_far`, `facts`, `plan`, `last_agent`), and the answer is a `MagenticStallInterventionDecision`:

| Decision | Effect |
| --- | --- |
| `continue_as_is()` | Reset the stall count and carry on |
| `replan()` | Reset and replan, as without intervention |
| `replan_with_guidance(text)` | Reset and replan, then add "Human guidance to help with stall: ..." to the history for the following rounds |
| `abort()` | Write the final answer from the history so far |

`reason` is "No progress being made", "Agents appear to be in a loop", or both. Python's stall decisions are `CONTINUE`, `REPLAN` and `GUIDANCE`; here guidance is folded into `Replan`, and `Abort` is an addition.

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::workflow::{MagenticProgressLedger, MagenticTaskLedger};
use async_trait::async_trait;
use serde_json::json;

fn ledger(satisfied: bool, progress: bool, next: &str) -> MagenticProgressLedger {
    serde_json::from_value(json!({
        "is_request_satisfied":    { "reason": "scripted", "answer": satisfied },
        "is_in_loop":              { "reason": "scripted", "answer": false },
        "is_progress_being_made":  { "reason": "scripted", "answer": progress },
        "next_speaker":            { "reason": "scripted", "answer": next },
        "instruction_or_question": { "reason": "scripted", "answer": "Load the data." },
    }))
    .unwrap()
}

/// A deterministic manager: the ledgers it returns are scripted.
struct Scripted {
    ledgers: Mutex<VecDeque<MagenticProgressLedger>>,
}

#[async_trait]
impl MagenticManager for Scripted {
    async fn plan(&self, context: &mut MagenticContext) -> Result<Message> {
        context.task_ledger = Some(MagenticTaskLedger {
            facts: Message::assistant("The data is in data/."),
            plan: Message::assistant("1. analyst loads the data. 2. analyst reports."),
        });
        Ok(Message::assistant("Plan: analyst loads, then reports."))
    }
    async fn replan(&self, context: &mut MagenticContext) -> Result<Message> {
        let guidance = context.chat_history.iter().any(|m| m.text().starts_with("Human guidance"));
        println!("  replan (reset {}; guidance already in history: {guidance})", context.reset_count);
        Ok(Message::assistant("Revised plan: sample the data first."))
    }
    async fn create_progress_ledger(&self, context: &MagenticContext) -> Result<MagenticProgressLedger> {
        println!("  round {} (stalls so far: {})", context.round_count, context.stall_count);
        Ok(self.ledgers.lock().unwrap().pop_front().unwrap_or_else(|| ledger(true, true, "analyst")))
    }
    async fn prepare_final_answer(&self, context: &MagenticContext) -> Result<Message> {
        Ok(Message::assistant(format!("Done after {} rounds.", context.round_count)))
    }
    fn max_stall_count(&self) -> usize {
        1
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let analyst: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(client).name("analyst").build());

    let manager = Scripted {
        ledgers: Mutex::new(VecDeque::from([
            ledger(false, false, "analyst"), // stall 1: allowed
            ledger(false, false, "analyst"), // stall 2: over the limit -> pause
            ledger(false, true, "analyst"),
        ])),
    };
    let workflow = MagenticBuilder::new()
        .participant("analyst", analyst)
        .manager(Arc::new(manager))
        .with_stall_intervention()
        .build()?;

    let mut run = workflow.run("Analyse the quarterly dataset").await?;
    let pending = run.pending_requests().remove(0);
    let stall: MagenticStallInterventionRequest = serde_json::from_value(pending.request_data)?;
    println!(
        "paused: {} (stall {}/{}, round {}, last agent {}, plan: {})",
        stall.reason, stall.stall_count, stall.max_stall_count, stall.round, stall.last_agent, stall.plan
    );
    let decision = MagenticStallInterventionDecision::replan_with_guidance("Try a 10% sample first.");
    run.send_response(pending.request_id, serde_json::to_value(decision)?).await?;

    let answer: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("{:?}: {}", run.state(), answer[0].text());
    Ok(())
}
```

```text
  round 1 (stalls so far: 0)
  round 2 (stalls so far: 1)
paused: No progress being made (stall 2/1, round 2, last agent analyst, plan: 1. analyst loads the data. 2. analyst reports.)
  replan (reset 1; guidance already in history: false)
  round 3 (stalls so far: 0)
  round 4 (stalls so far: 0)
Idle: Done after 4 rounds.
```

The guidance is added after the replan, so `replan` itself does not see it, but every later progress ledger and participant turn does. The paused state is kept in shared state under `_magentic_stall_state` (and `_magentic_plan_review_state` for plan review).

## Agent approval

`with_request_info()` on `SequentialBuilder` or `ConcurrentBuilder` wraps each participant in an `AgentApprovalExecutor`, which pauses **after** the agent replies so a person can review the actual reply. The request payload is an `ApprovalRequest` with the `conversation` the agent was given and its pending `reply`. An empty answer approves: `null`, a blank string, `[]` or `{}`. Anything else is feedback: it is added to the conversation as a user message (strings as they are, other values as JSON text), the same agent runs again, and a new request is raised, until an answer approves. This matches upstream's current `AgentApprovalExecutor`, which replaced an older interceptor that paused before the agent ran. Python's `with_request_info(agents=[...])` can gate a subset of participants; the Rust method gates all of them, so build the graph by hand with `AgentApprovalExecutor` and `AgentExecutor` to gate only some.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::ApprovalRequest;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let agent = |name: &str| -> Arc<dyn SupportsAgentRun> {
        Arc::new(Agent::builder(client.clone()).name(name).instructions("Draft a one-line release note.").build())
    };

    let workflow = SequentialBuilder::new()
        .participants(vec![agent("writer"), agent("editor")])
        .with_request_info()
        .build()?;

    let mut run = workflow.run("Release notes for v2.0").await?;
    // writer: revise once, then approve; editor: approve straight away.
    let mut answers = vec![json!("Mention the breaking changes."), json!(""), json!(null)].into_iter();
    while run.state() == WorkflowRunState::IdleWithPendingRequests {
        let pending = run.pending_requests().remove(0);
        let request: ApprovalRequest = serde_json::from_value(pending.request_data)?;
        let answer = answers.next().unwrap();
        println!(
            "{} replied {:?} after {} messages -> {}",
            pending.source_executor_id,
            request.reply.last().map(Message::text).unwrap_or_default(),
            request.conversation.len(),
            if answer.as_str().is_some_and(|s| !s.is_empty()) { "revise" } else { "approve" },
        );
        run.send_response(pending.request_id, answer).await?;
    }
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    for m in &conversation {
        println!("{}: {}", m.author_name.as_deref().unwrap_or(m.role.as_str()), m.text());
    }
    Ok(())
}
```

```text
seq_0 replied "Mock reply to: Release notes for v2.0" after 1 messages -> revise
seq_0 replied "Mock reply to: Mention the breaking changes." after 3 messages -> approve
seq_1 replied "Mock reply to: Mention the breaking changes." after 4 messages -> approve
user: Release notes for v2.0
writer: Mock reply to: Release notes for v2.0
user: Mention the breaking changes.
writer: Mock reply to: Mention the breaking changes.
editor: Mock reply to: Mention the breaking changes.
```

The feedback stays in the conversation, so later participants see the reviewer's comments as well as the revised reply. In a concurrent workflow each participant pauses separately, and all the requests are open at once; answer them together with `send_responses`. `pending_request.source_executor_id` (`seq_0`, `agent_1`, ...) says which participant is waiting; the agent's name is on the reply messages.

The group chat, handoff and Magentic builders have no `with_request_info`. Handoff already pauses for the user, and Magentic has plan review and stall intervention. Tool-call approval (an agent asking before running a tool marked `ApprovalMode::AlwaysRequire`) is a separate mechanism at the agent level, described on the [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/). No orchestration turns a participant's tool-approval request into a workflow request; it stays in that agent's response.

### Checkpointing an approval gate

The orchestration builders have no `with_checkpointing`, so a paused orchestration lives only in memory. `AgentApprovalExecutor` is public, though, and keeps everything it needs in the request payload, so a hand-built graph of approval gates can be checkpointed and answered after a restart without running the agent again:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{AgentApprovalExecutor, ApprovalRequest};
use serde_json::json;

/// The approval gate built by hand, so the graph can be checkpointed: the
/// orchestration builders have no with_checkpointing.
fn build(client: &OpenAIChatCompletionClient, storage: Arc<dyn CheckpointStorage>) -> Result<Workflow> {
    let writer: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(client.clone()).name("writer").instructions("Draft a one-line announcement.").build(),
    );
    WorkflowBuilder::new()
        .add_executor(Arc::new(AgentApprovalExecutor::new("writer", writer).with_output(true)))
        .set_start("writer")
        .with_checkpointing(storage)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let dir = std::env::temp_dir().join("orch-approval-checkpoints");
    let _ = std::fs::remove_dir_all(&dir);
    let storage: Arc<dyn CheckpointStorage> = Arc::new(FileCheckpointStorage::new(&dir)?);

    // Process one: the agent drafts, and the run waits for a reviewer.
    let first = build(&client, storage.clone())?;
    let run = first.run("Announce the new office").await?;
    let request_id = run.pending_requests()[0].request_id.clone();
    let checkpoint = storage.list(Some(first.id())).await?.pop().unwrap();
    drop(run);

    // Process two: restore and approve. The pending draft is in the request
    // payload, so the agent does not run again.
    let second = build(&client, storage.clone())?;
    let mut run = second.run_from_checkpoint(&checkpoint.checkpoint_id, storage).await?;
    let request: ApprovalRequest = serde_json::from_value(run.pending_requests()[0].request_data.clone())?;
    println!("restored draft: {}", request.reply[0].text());
    run.send_response(request_id, json!("")).await?;
    let conversation: Vec<Message> = serde_json::from_value(run.last_output().unwrap())?;
    println!("{:?}, {} messages", run.state(), conversation.len());
    let agent_runs = run.events().iter().filter(|e| matches!(e, WorkflowEvent::AgentRun { .. })).count();
    println!("agent runs after restore: {agent_runs}");
    Ok(())
}
```

```text
restored draft: Mock reply to: Announce the new office
Idle, 2 messages
agent runs after restore: 0
```

The group chat, handoff and Magentic executors are private, so there is no equivalent for them.

## Streaming updates

`workflow.run_stream(input)` works on any orchestration. Each participant's streamed chunks arrive as `AgentRunUpdate { executor_id, update }` events, where `update` deserialises to an `AgentResponseUpdate` whose `author_name` is the participant's name, followed by an `AgentRun` event with the whole response.

The workflow engine applies an executor's events only when its superstep ends, so chunks are not live. For sequential and concurrent workflows they arrive as each superstep finishes: all of the first agent's chunks after it is done, and so on. For group chat, handoff and Magentic, which run their whole loop as one executor, every chunk of every participant arrives together when the loop finishes or pauses. Streaming a workflow is useful for progress and for attributing output to participants; for token-by-token display, stream a single agent instead.

```rust
use std::io::Write;
use std::sync::Arc;

use agent_framework::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?;
    let agent = |name: &str, instructions: &str| -> Arc<dyn SupportsAgentRun> {
        Arc::new(Agent::builder(client.clone()).name(name).instructions(instructions).build())
    };
    let workflow = ConcurrentBuilder::new()
        .participants(vec![
            agent("optimist", "Argue for the idea in one sentence."),
            agent("sceptic", "Argue against the idea in one sentence."),
        ])
        .build()?;

    let mut stream = workflow.run_stream("Adopt Rust for the payments service");
    let mut current = String::new();
    while let Some(event) = stream.next().await {
        match event {
            WorkflowEvent::AgentRunUpdate { executor_id, update } => {
                let update: AgentResponseUpdate = serde_json::from_value(update)?;
                if executor_id != current {
                    print!("\n[{executor_id} / {}] ", update.author_name.as_deref().unwrap_or("?"));
                    current = executor_id;
                }
                print!("{}", update.text());
                std::io::stdout().flush().ok();
            }
            WorkflowEvent::AgentRun { executor_id, .. } => println!("\n({executor_id} finished)"),
            WorkflowEvent::SuperStepCompleted(n) => println!("-- superstep {n} done"),
            _ => {}
        }
    }
    let run = stream.into_run().await?;
    println!("final state: {:?}", run.state());
    Ok(())
}
```

```text
-- superstep 1 done

[agent_0 / optimist] Mock reply to: Adopt Rust for the payments service
(agent_0 finished)

[agent_1 / sceptic] Mock reply to: Adopt Rust for the payments service
(agent_1 finished)
-- superstep 2 done
-- superstep 3 done
final state: Idle
```

An executor's `Output` event comes before its `AgentRunUpdate` events, so the last participant of a sequential workflow appears to produce the result before its chunks. Wrapped as an agent with `as_agent`, a workflow's stream is these same updates converted back to `AgentResponseUpdate`s; see [a workflow as an agent](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#a-workflow-as-an-agent).

## What is not ported

Compared with the Python orchestrations, this release does not have:

- the multi-executor graph shape for group chat, handoff and Magentic (see [above](#how-the-rust-orchestrations-differ)), and so no per-participant executor ids or checkpoints for them;
- automatic injection of handoff tools into participants; attach `handoff_tool_spec` yourself;
- per-agent autonomous turn limits for handoff, and autonomous re-prompting of the same agent;
- a pluggable aggregator for `ConcurrentBuilder` (build the graph by hand, as [above](#concurrent));
- `SequentialBuilder::ensure_trailing_user_turn`;
- `with_checkpointing` on any orchestration builder;
- a progress-ledger retry back-off (retries are immediate);
- orchestration telemetry beyond the agents' own spans.

Additions with no Python counterpart: the round-robin group chat manager, the 40-round default cap on group chats, and the `Abort` stall decision.

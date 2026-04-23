---
title: "Microsoft Agent Framework (Python) — Multi-Agent Orchestration"
description: "Sequential, Concurrent, Handoff, GroupChat and Magentic-One builders — all from agent-framework-orchestrations 1.0.0b260421. Real signatures and runnable examples."
framework: microsoft-agent-framework
language: python
---

# Multi-Agent Orchestration — Python

Five built-in orchestration patterns ship in `agent-framework-orchestrations`. Each is a fluent builder that produces a `Workflow` — the same object type returned by `WorkflowBuilder`. Once you have a workflow, run it with `workflow.run(...)` or stream events with `workflow.run(..., stream=True)`.

All signatures below are verified against `agent-framework-orchestrations==1.0.0b260421`.

| Pattern | Builder | Topology | Use case |
|---|---|---|---|
| Sequential | `SequentialBuilder` | A → B → C | Document pipeline (research → analyse → summarise) |
| Concurrent | `ConcurrentBuilder` | Fan-out / fan-in | Independent opinions aggregated once |
| Handoff | `HandoffBuilder` | Mesh or directed | Support triage routed to specialists |
| GroupChat | `GroupChatBuilder` | Star, orchestrator picks speaker | Panel discussion, code review |
| Magentic | `MagenticBuilder` | Manager + workers + task ledger | Open-ended research with replanning |

## Imports

```python
from agent_framework_orchestrations import (
    SequentialBuilder,
    ConcurrentBuilder,
    HandoffBuilder,
    GroupChatBuilder,
    MagenticBuilder,
    GroupChatState,
    StandardMagenticManager,
)
```

The `agent_framework_orchestrations` package is distinct from `agent_framework`. The meta-install (`pip install agent-framework`) pulls it in; if you pin sub-packages, add `agent-framework-orchestrations` explicitly.

## Building the participant agents

The examples below reuse three agents. Note that `name=` is required for most builders (especially `Handoff` and `Magentic`) because routing is keyed by name.

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

researcher = Agent(
    client=client,
    name="researcher",
    instructions="You are a researcher. Produce bullet-point facts.",
)
analyst = Agent(
    client=client,
    name="analyst",
    instructions="You analyse facts into a coherent narrative.",
)
writer = Agent(
    client=client,
    name="writer",
    instructions="You write a final one-paragraph summary.",
)
```

## Sequential

The output of each participant becomes part of the conversation passed to the next. The final output is the conversation emitted by the last participant.

```python
from agent_framework_orchestrations import SequentialBuilder

workflow = SequentialBuilder(participants=[researcher, analyst, writer]).build()

result = await workflow.run("Quantum computing in 2026")
print(result.get_outputs()[-1])  # final conversation
```

Optional knobs:

- `chain_only_agent_responses=True` — pass only assistant messages between agents (skip user history). Useful when the conversation would otherwise balloon.
- `intermediate_outputs=True` — yield per-participant responses as events, not just the final one.
- `checkpoint_storage=...` — persist state between runs; see the [HITL page](./microsoft_agent_framework_python_hitl/) for resuming from a checkpoint.
- `.with_request_info(agents=[...])` — pause after each (or a subset of) participants for human review.

```python
workflow = (
    SequentialBuilder(participants=[researcher, analyst, writer], intermediate_outputs=True)
    .with_request_info(agents=[analyst])  # pause only after the analyst
    .build()
)
```

Non-agent participants are `Executor` subclasses — useful for deterministic transforms (e.g. deduplicate citations, canonicalise JSON) inserted into the chain.

## Concurrent

Dispatches the same input to every participant in parallel and aggregates the results. Useful for ensembling opinions or running independent analyses that you then reduce.

```python
from agent_framework_orchestrations import ConcurrentBuilder

workflow = ConcurrentBuilder(participants=[researcher, analyst, writer]).build()
result = await workflow.run("Summarise agent-framework 1.1.0")
```

Custom aggregator (sync or async, returning a value or `None`):

```python
from agent_framework import AgentExecutorResponse

def join_as_bullets(responses: list[AgentExecutorResponse]) -> str:
    return "\n".join(
        f"- ({r.executor_id}) {r.agent_run_response.messages[-1].text}" for r in responses
    )

workflow = (
    ConcurrentBuilder(participants=[researcher, analyst, writer])
    .with_aggregator(join_as_bullets)
    .build()
)
```

The aggregator callback may also accept `(responses, ctx: WorkflowContext)` — use `ctx.yield_output(...)` to emit structured events when you don't want to return a scalar.

## Handoff

A decentralised, mesh-or-directed routing pattern. Each agent receives tools that let it hand the conversation off to another agent. Great for support triage and "route to the right expert" workflows.

```python
from agent_framework_orchestrations import HandoffBuilder

triage = Agent(client=client, name="triage",
               instructions="Classify the request and hand off to billing or technical.")
billing = Agent(client=client, name="billing",
                instructions="You resolve billing questions.",
                description="Handles invoices, refunds, plan changes.")
technical = Agent(client=client, name="technical",
                  instructions="You resolve technical questions.",
                  description="Handles bugs, API errors, outages.")

workflow = (
    HandoffBuilder(participants=[triage, billing, technical])
    .add_handoff(triage, [billing, technical])
    .with_start_agent(triage)
    .build()
)

result = await workflow.run("My card was charged twice for last month.")
```

Notes:

- Participants **must** be `Agent` instances — the builder clones them and injects handoff tools, which isn't possible for the bare `SupportsAgentRun` protocol.
- If you omit `add_handoff(...)`, every agent can hand off to every other (mesh topology).
- `agent.description` is used as the handoff tool's description — fill this in for each specialist so the triage agent picks correctly.
- `.with_autonomous_mode(enabled_agents=[...], turn_limits={...})` lets certain specialists answer autonomously for N turns before re-querying the user.
- `.with_termination_condition(lambda conv: len(conv) > 20)` — stop after a size or content check.

## GroupChat

A central orchestrator picks the next speaker on every turn. Two ways to drive it: a **selection function** (pure code, no LLM) or an **orchestrator agent** (LLM-driven).

### Code-driven selection

```python
from agent_framework_orchestrations import GroupChatBuilder, GroupChatState

def pick_next_speaker(state: GroupChatState) -> str | None:
    if state.current_round >= 3:
        return None                        # None terminates the chat
    last = state.conversation[-1].author_name if state.conversation else None
    return "writer" if last == "researcher" else "researcher"

workflow = (
    GroupChatBuilder(
        participants=[researcher, writer],
        selection_func=pick_next_speaker,
        max_rounds=10,
    )
    .build()
)

result = await workflow.run("Write an abstract on RAG evaluation.")
```

### Orchestrator-agent selection

Let an LLM pick the next speaker. The orchestrator is just another `Agent`.

```python
orchestrator = Agent(
    client=client,
    name="moderator",
    instructions=(
        "You moderate a panel of a researcher and a writer. "
        "Given the conversation, choose who speaks next by replying with just their name. "
        "Reply 'DONE' to end."
    ),
)

workflow = (
    GroupChatBuilder(
        participants=[researcher, writer],
        orchestrator_agent=orchestrator,
        max_rounds=8,
    )
    .build()
)
```

Optional:

- `termination_condition=lambda conv: any("DONE" in m.text for m in conv[-1:])`
- `checkpoint_storage=...` for mid-session persistence.
- `intermediate_outputs=True` to observe each speaker's message as it happens.

## Magentic (Magentic-One)

Magentic adds a **task ledger** and **progress ledger**. A manager plans the task, dispatches subtasks to participants, re-plans on stalls, and synthesises a final answer. This is the pattern used in Microsoft's Magentic-One research system.

```python
from agent_framework_orchestrations import MagenticBuilder

# The manager is itself an Agent — give it a capable model.
manager_agent = Agent(
    client=OpenAIChatClient(model="gpt-5"),
    name="magentic-manager",
    instructions="You coordinate specialists. Be concise.",
)

workflow = (
    MagenticBuilder(
        participants=[researcher, analyst, writer],
        manager_agent=manager_agent,
        max_stall_count=3,      # replan after 3 rounds without progress
        max_round_count=20,
        enable_plan_review=True,  # HITL — approve the initial plan
    )
    .build()
)

result = await workflow.run("Write a research memo on post-training alignment.")
```

Alternative: bring your own manager by subclassing `MagenticManagerBase` and passing `manager=`. Use this when the default LLM-driven manager doesn't match your domain (e.g. you want deterministic planning).

### Observability

Magentic emits structured events you can hook into:

```python
from agent_framework_orchestrations import MagenticOrchestratorEventType

async for event in workflow.run("…", stream=True):
    if event.type == "orchestrator" and event.data.kind == MagenticOrchestratorEventType.TASK_LEDGER:
        print("Plan:", event.data.payload)
    elif event.type == "output":
        print("Final:", event.data)
```

### HITL specialist hooks

- `enable_plan_review=True` — human approves the plan before execution begins.
- `.with_human_input_on_stall()` — human intervenes when the workflow stalls instead of auto-replanning.

See the [Human-in-the-loop page](./microsoft_agent_framework_python_hitl/) for how to respond to these events from your caller.

## Picking a pattern

- **Linear pipeline with 2–5 agents** → `SequentialBuilder`.
- **Independent opinions / ensembling** → `ConcurrentBuilder` with a custom aggregator.
- **Triage + specialists, user in the loop** → `HandoffBuilder` with per-agent `description`.
- **Deterministic speaker rotation / moderated debate** → `GroupChatBuilder` with a selection function.
- **Open-ended research with replanning** → `MagenticBuilder` with plan review.

All five produce an identical `Workflow` object, so checkpointing, streaming, and HITL patterns work the same across them.

## Building arbitrary graphs with `WorkflowBuilder` and `AgentExecutor`

The five builders above cover the common topologies. When you need something shaped differently — a custom router, a deterministic transform between two agents, a sub-workflow nested inside another — drop to `WorkflowBuilder` and wire `AgentExecutor` nodes manually.

```python
from agent_framework import AgentExecutor, WorkflowBuilder

research_node = AgentExecutor(researcher)
analyse_node = AgentExecutor(analyst)
write_node = AgentExecutor(writer)

workflow = (
    WorkflowBuilder(start_executor=research_node)
    .add_edge(research_node, analyse_node)
    .add_edge(analyse_node, write_node)
    .build()
)

result = await workflow.run("Quantum sensors in 2026")
```

`WorkflowBuilder(start_executor=..., output_executors=[...])` both accept an executor instance or a bare agent — the builder wraps agents in `AgentExecutor` automatically. Use `add_chain([a, b, c])` as a shorthand when the edges are strictly linear.

`AgentExecutor` is the executor that `SequentialBuilder` / `HandoffBuilder` / etc. wrap your agents in behind the scenes. Constructing it directly lets you:

- Reuse the same executor instance at multiple positions in a graph (shared message cache).
- Control context inheritance via `context_mode` (see below).
- Mix with non-agent `Executor` subclasses for deterministic transforms.

### Controlling context flow — `context_mode`

When one `AgentExecutor` sends its response downstream to another `AgentExecutor`, the downstream executor has three options for how much of the upstream conversation to inherit:

| `context_mode` | Behaviour |
|---|---|
| `"full"` (default) | Append the entire upstream conversation (prior messages + agent response). Preserves full context chains across many hops. |
| `"last_agent"` | Append only the upstream agent's own response messages. Keeps the prompt small at the cost of losing earlier turns. |
| `"custom"` | Pass a `context_filter` callable that picks the messages to inherit. |

```python
from agent_framework import AgentExecutor, Message, WorkflowBuilder


def drop_tool_calls(history: list[Message]) -> list[Message]:
    """Strip function_call / function_result content before passing to the summariser."""
    return [m for m in history if m.role in {"user", "assistant", "system"}]


summariser = AgentExecutor(
    writer,
    context_mode="custom",
    context_filter=drop_tool_calls,
)
```

### Transforming agent output — `AgentExecutorResponse.with_text`

A custom executor inserted between two `AgentExecutor` nodes can transform the text without breaking the context chain. The catch: if you just emit a plain `str`, the next `AgentExecutor.from_str` handler wipes the cache because only the string lands. Use `AgentExecutorResponse.with_text(...)` instead — the framework keeps the full prior conversation and only substitutes the final assistant message:

```python
from agent_framework import AgentExecutorResponse, WorkflowContext, executor


@executor(
    id="translate_to_english",
    input=AgentExecutorResponse,
    output=AgentExecutorResponse,
)
async def translate(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse, AgentExecutorResponse],
) -> None:
    english = await translate_text(response.agent_response.text, target="en")
    await ctx.send_message(response.with_text(english))
```

The downstream `AgentExecutor` now sees the translation as the assistant turn, with every prior message (researcher findings, tool calls, system prompt) still in the cache. Without `with_text`, the translation would arrive as a bare `str` and the writer would start from zero context.

### Manual routing with edges

`WorkflowBuilder` exposes helper methods for the common edge shapes — the packaged builders use them internally:

```python
from agent_framework import Case, Default, WorkflowBuilder

# Fan out to three reviewers, fan in to a merger.
builder = WorkflowBuilder(start_executor=router)
builder.add_fan_out_edges(router, [reviewer_a, reviewer_b, reviewer_c])
builder.add_fan_in_edges([reviewer_a, reviewer_b, reviewer_c], merger)

# Switch on the router's output — each Case is a condition + target pair.
builder.add_switch_case_edge_group(
    source=triage,
    cases=[
        Case(condition=lambda msg: msg.category == "billing", target=billing_agent),
        Case(condition=lambda msg: msg.category == "technical", target=technical_agent),
        Default(target=general_agent),
    ],
)

# Also available:
#   .add_chain([a, b, c])               — A → B → C shorthand
#   .add_edge(a, b, condition=lambda m: ...)   — conditional single edge
#   .add_multi_selection_edge_group(...)       — fan-out with a picker
```

## Nested workflows — `WorkflowExecutor` and sub-workflows

`WorkflowExecutor` wraps a whole `Workflow` as a single node in a parent workflow. Great for reuse (a "document pipeline" sub-workflow called from many higher-level flows) and for isolating state between phases.

```python
from agent_framework import AgentExecutor, WorkflowBuilder, WorkflowExecutor

# Inner workflow — turn a research prompt into a structured dossier.
inner = SequentialBuilder(participants=[researcher, analyst]).build()

# Outer workflow — call the dossier producer, then the writer.
dossier_node = WorkflowExecutor(inner, id="dossier")
writer_node = AgentExecutor(writer, id="writer")

outer = (
    WorkflowBuilder(start_executor=dossier_node)
    .add_edge(dossier_node, writer_node)
    .build()
)

result = await outer.run("Mercury fuel cells")
```

### Passing requests from child to parent

A sub-workflow can "ask" its parent for information — the parent sees a `SubWorkflowRequestMessage`, resolves it, and sends back a `SubWorkflowResponseMessage`. This is how you plug a sub-workflow into a parent that owns a database connection, an auth token, or human-in-the-loop approval.

```python
from agent_framework import (
    Executor,
    SubWorkflowRequestMessage,
    SubWorkflowResponseMessage,
    WorkflowContext,
    handler,
)


class UserLookupExecutor(Executor):
    """Parent-side handler that resolves a sub-workflow's user lookup request."""

    def __init__(self, user_db):
        super().__init__("user_lookup")
        self._users = user_db

    @handler
    async def on_request(
        self,
        request: SubWorkflowRequestMessage,
        ctx: WorkflowContext[SubWorkflowResponseMessage],
    ) -> None:
        event = request.source_event
        # event.data carries whatever the inner executor sent via ctx.request_info(...)
        user_id = event.data
        profile = self._users.get(user_id, {"name": "<unknown>"})

        # create_response validates the return type against the original request.
        response = request.create_response(data=profile)
        await ctx.send_message(response, target_id=request.executor_id)


# Wire the lookup handler into the parent workflow.
outer = (
    WorkflowBuilder(start_executor=dossier_node)
    .add_edge(dossier_node, UserLookupExecutor(user_db={"u-1": {"name": "Ada"}}))
    .add_edge(dossier_node, writer_node)
    .build()
)
```

Inside the inner workflow, an executor triggers a request by emitting a `WorkflowEvent` via `ctx.request_info(event, response_type=dict)` — the framework captures it, wraps it in `SubWorkflowRequestMessage`, and routes it to the parent. The inner executor pauses until the matching `SubWorkflowResponseMessage` arrives and resumes with the response data.

Set `propagate_request=True` on the `WorkflowExecutor` to forward requests further up (to the grandparent or the workflow caller) instead of handling them in the parent:

```python
propagating = WorkflowExecutor(inner, id="dossier", propagate_request=True)
```

And `allow_direct_output=True` makes the sub-workflow's `ctx.yield_output(...)` calls surface directly in the parent workflow's event stream rather than being re-emitted as messages — useful when you want the sub-workflow's output to be the parent's output verbatim.

### Type checking across the boundary

`WorkflowExecutor.input_types` derives from the wrapped workflow's start executor — the parent workflow validates messages at graph build time, so mismatches fail early:

```python
assert SubWorkflowResponseMessage in dossier_node.input_types
assert dossier_node.input_types == inner.input_types + [SubWorkflowResponseMessage]
```

## Streaming events

Every workflow — built from one of the five builders or from `WorkflowBuilder` directly — can stream incremental events:

```python
async for event in workflow.run("Topic", stream=True):
    if event.type == "output":
        print("output:", event.data)
    elif event.type == "executor_completed":
        print(f"[{event.executor_id}] done")
    elif event.type == "request_info":
        print(f"waiting on {event.request_type.__name__}")
    elif event.type == "failed":
        print(f"workflow failed: {event.details}")
```

Event types you'll see (the `type` attribute is a `Literal[...]` string):

- `started` / `failed` — workflow-level lifecycle.
- `superstep_started` / `superstep_completed` — one super-step of the graph just advanced.
- `executor_invoked` / `executor_completed` / `executor_failed` — per-executor transitions.
- `output` — a consumable output yielded by an executor.
- `request_info` — a sub-workflow (or any executor) is asking for external data; see the HITL / sub-workflow sections above.
- `group_chat` / `handoff_sent` / `magentic_orchestrator` — pattern-specific events from the built-in orchestrators.
- `warning` / `error` — diagnostic events; inspect `event.details` for the error payload.

`event.origin` tells you whether the event came from the framework itself (`WorkflowEventSource.FRAMEWORK`) or from an executor (`WorkflowEventSource.EXECUTOR`) — useful when you want to skip framework-emitted super-step events and only see outputs from your own code.

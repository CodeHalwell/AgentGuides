---
title: "Microsoft Agent Framework (Python) — Multi-Agent Orchestration"
description: "Sequential, Concurrent, Handoff, GroupChat and Magentic-One builders — verified against agent-framework 1.11.0. Real signatures and runnable examples."
framework: microsoft-agent-framework
language: python
---

# Multi-Agent Orchestration — Python

Five built-in orchestration patterns ship in `agent-framework-orchestrations`. Each is a fluent builder that produces a `Workflow` — the same object type returned by `WorkflowBuilder`. Once you have a workflow, run it with `workflow.run(...)` or stream events with `workflow.run(..., stream=True)`.

All signatures below are verified against `agent-framework==1.11.0`.

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
result = await workflow.run("Summarise agent-framework 1.5.0")
```

Custom aggregator (sync or async, returning a value or `None`):

```python
from agent_framework import AgentExecutorResponse

def join_as_bullets(responses: list[AgentExecutorResponse]) -> str:
    return "\n".join(
        f"- ({r.executor_id}) {r.agent_response.text}" for r in responses
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

The builder takes a `StandardMagenticManager` (wrapping your manager agent) as its sole positional argument; participants are added fluently with `.add_participant()`.

```python
from agent_framework_orchestrations import MagenticBuilder, StandardMagenticManager

# The manager is itself an Agent — give it a capable model.
manager_agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="magentic-manager",
    instructions="You coordinate specialists. Produce a task ledger, assign subtasks, and synthesise results.",
)

workflow = (
    MagenticBuilder(StandardMagenticManager(manager_agent))
    .add_participant(researcher)
    .add_participant(analyst)
    .add_participant(writer)
    .build()
)

result = await workflow.run("Write a research memo on post-training alignment.")
print(result.output)
```

Alternative: bring your own manager by subclassing `MagenticManagerBase` and wrapping it instead of `StandardMagenticManager`. Use this when you want deterministic planning logic rather than an LLM-driven coordinator.

### HITL — plan sign-off

Call `.require_plan_signoff()` on the builder before `.build()` to pause after the manager produces the initial task ledger. Register a handler for `MagenticPlanReviewRequest` on the resulting workflow:

```python
from agent_framework_orchestrations import (
    MagenticBuilder,
    StandardMagenticManager,
    MagenticPlanReviewRequest,
    MagenticPlanReviewResponse,
)

workflow = (
    MagenticBuilder(StandardMagenticManager(manager_agent))
    .add_participant(researcher)
    .add_participant(analyst)
    .add_participant(writer)
    .require_plan_signoff()   # pause for human approval
    .build()
)

async def human_plan_review(request: MagenticPlanReviewRequest) -> MagenticPlanReviewResponse:
    print("Proposed plan:")
    print(request.plan.text)           # plan is a Message, not an iterable
    approved = input("Approve? [y/n] ").strip().lower() == "y"
    if approved:
        return request.approve()
    else:
        return request.revise("Please break the plan into smaller steps.")

workflow.add_request_info_handler(MagenticPlanReviewRequest, human_plan_review)
result = await workflow.run("Analyse LLM safety benchmarks for 2025.")
```

### Observability

Stream events from a running workflow to observe plan updates, participant turns, and the final output:

```python
async for event in workflow.run("…", stream=True):
    match event.type:
        case "orchestrator":
            print("[manager]", event.data)
        case "agent":
            print(f"[{event.agent_name}]", event.data)
        case "output":
            print("Final answer:", event.data)
```

See the [Human-in-the-loop page](./microsoft_agent_framework_python_hitl/) for how to respond to request-info events from your caller.

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

### Sharing an `AgentSession` across executors

Every `AgentExecutor` owns an `AgentSession` internally. By default each executor creates its own session — useful when you want each node to have isolated conversation state, but wasteful when several executors wrap the same agent (e.g. a multi-turn loop) or you want a shared scratchpad.

Pass `session=` to reuse a single session across every executor that should share state:

```python
from agent_framework import Agent, AgentExecutor, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

planner = Agent(client=OpenAIChatClient(), name="planner", instructions="Break tasks down.")
actor   = Agent(client=OpenAIChatClient(), name="actor",   instructions="Execute steps.")

# Both executors write into the same session — the actor sees the planner's
# notes in session.state and can update a shared scratchpad.
shared = planner.create_session(session_id="job-42")

plan_node  = AgentExecutor(planner, session=shared, id="plan")
act_node   = AgentExecutor(actor,   session=shared, id="act")

workflow = (
    WorkflowBuilder(start_executor=plan_node)
    .add_edge(plan_node, act_node)
    .build()
)
```

`shared.state` is a plain mutable dict — tools, middleware, and context providers on either agent read and write it. That's how you pass "what's already been tried" between planner and actor without stuffing it into the conversation history.

When to keep sessions separate:

- Each executor answers a different part of the same question (review → summariser) and neither needs the other's scratchpad.
- You want to parallelise multiple runs with distinct `session_id`s so context providers (mem0, Redis) don't collide.

When to share:

- Multi-turn loops where one agent plans, another executes, and a third verifies — all against the same scratchpad.
- A supervisor agent that needs to see what a worker remembered from an earlier turn.

### Cloning executors for graph reuse — `AgentExecutor.clone`

When the same agent appears at multiple positions in a graph, wrap it in two separate `AgentExecutor`s with distinct ids — the constructor gives each one its own cache and session:

```python
first_pass  = AgentExecutor(reviewer, id="review_a")
second_pass = AgentExecutor(reviewer, id="review_b")   # same agent, independent executor state
```

`AgentExecutor.clone(deep=True)` is for snapshotting an executor that has already accumulated in-memory state (cached messages, open sessions, handler configuration) so the copy picks up exactly where the original left off:

```python
snapshot = first_pass.clone()            # deepcopy — preserves cache and session contents
# snapshot.id == first_pass.id — ids are copied as-is; use clone() inside rollback / fan-out
# tooling where the replica runs in a different workflow scope than the original.
```

`deep=False` produces a shallow copy that shares the underlying agent, cache, and session with the original — useful for observability wrappers that want read-through access to the live state, but unsafe for graphs where both copies will receive messages.

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

### Cache-only sends — `AgentExecutorRequest(should_respond=False)`

When you want an upstream node to *prime* a downstream `AgentExecutor` with extra context but *not* trigger a model call yet, send an `AgentExecutorRequest` with `should_respond=False`:

```python
from agent_framework import AgentExecutorRequest, Message, WorkflowContext, executor


@executor(id="prime")
async def prime(_: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    # Push a system note into the next executor's cache without invoking the model.
    ctx_msg = Message(role="user", contents=["Today's tenant is acme. Use the acme tone."])
    await ctx.send_message(AgentExecutorRequest(messages=[ctx_msg], should_respond=False))


@executor(id="ask")
async def ask(_: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
    # On the next hop, the cached priming message is already in scope.
    ctx_msg = Message(role="user", contents=["Draft the welcome email."])
    await ctx.send_message(AgentExecutorRequest(messages=[ctx_msg], should_respond=True))
```

This is the formal way to interleave deterministic state injection with model calls in the same `AgentExecutor`. The cache survives across messages until the next request with `should_respond=True` arrives — at which point the agent sees the full priming history followed by the actual prompt.

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

`WorkflowBuilder` exposes helper methods for every edge shape — the packaged builders use these internally. All edge methods accept bare `SupportsAgentRun` instances (agents) as well as `Executor` instances; the builder wraps agents in `AgentExecutor` automatically.

#### Fan-out / fan-in (parallel execution)

```python
from agent_framework import WorkflowBuilder

# Fan out to three reviewers, fan in to a merger.
workflow = (
    WorkflowBuilder(start_executor=router)
    .add_fan_out_edges(router, [reviewer_a, reviewer_b, reviewer_c])
    .add_fan_in_edges([reviewer_a, reviewer_b, reviewer_c], merger)
    .build()
)
```

Each reviewer receives the router's output concurrently. `merger` receives `list[AgentExecutorResponse]` — one entry per reviewer — only after all three complete.

#### Switch-case routing

Route a message to exactly one branch based on conditions evaluated in order. Use `Case` for conditions and `Default` as the fallback:

```python
from dataclasses import dataclass
from agent_framework import (
    Agent,
    Case,
    Default,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.openai import OpenAIChatClient


@dataclass
class SupportTicket:
    category: str
    text: str


class TriageExecutor(Executor):
    """Classify incoming text into a SupportTicket."""

    @handler
    async def classify(self, text: str, ctx: WorkflowContext[SupportTicket]) -> None:
        category = "billing" if "invoice" in text.lower() else "technical"
        await ctx.send_message(SupportTicket(category=category, text=text))


client = OpenAIChatClient()
triage = TriageExecutor(id="triage")
billing_agent = Agent(client=client, name="billing", instructions="You resolve billing questions.")
technical_agent = Agent(client=client, name="technical", instructions="You resolve technical questions.")
general_agent = Agent(client=client, name="general", instructions="You handle general inquiries.")

workflow = (
    WorkflowBuilder(start_executor=triage)
    .add_switch_case_edge_group(
        source=triage,
        cases=[
            Case(condition=lambda msg: msg.category == "billing",   target=billing_agent),
            Case(condition=lambda msg: msg.category == "technical", target=technical_agent),
            Default(target=general_agent),
        ],
    )
    .build()
)

result = await workflow.run("I was charged twice on my last invoice.")
print(result.get_outputs()[-1].text)
```

Conditions are evaluated in declaration order; the first `Case` whose condition returns `True` wins. The `Default` case matches everything that falls through. Exactly one branch executes.

#### Multi-selection edge group

A `selection_func` receives the outgoing message and the list of candidate executor IDs, and returns the subset that should receive it. Unlike switch-case (exactly one branch), multi-selection can send to any number of targets — including all or none.

```python
from dataclasses import dataclass
from agent_framework import (
    Agent,
    AgentExecutor,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)


@dataclass
class AnalysisTask:
    priority: str   # "high" | "low"
    text: str


class Dispatcher(Executor):
    @handler
    async def dispatch(self, text: str, ctx: WorkflowContext[AnalysisTask]) -> None:
        priority = "high" if len(text) > 50 else "low"
        await ctx.send_message(AnalysisTask(priority=priority, text=text))


def route_by_priority(task: AnalysisTask, candidate_ids: list[str]) -> list[str]:
    """High-priority tasks go to all workers; low-priority only to the fast worker."""
    if task.priority == "high":
        return candidate_ids                # all workers
    return [cid for cid in candidate_ids if "fast" in cid]


fast_agent = Agent(id="fast_agent", model="gpt-4o-mini", instructions="Answer concisely.")
thorough_agent = Agent(id="thorough_agent", model="gpt-4o", instructions="Answer thoroughly.")

dispatcher = Dispatcher(id="dispatcher")
fast_worker = AgentExecutor(id="fast_worker", agent=fast_agent)
thorough_worker = AgentExecutor(id="thorough_worker", agent=thorough_agent)

workflow = (
    WorkflowBuilder(start_executor=dispatcher)
    .add_multi_selection_edge_group(
        source=dispatcher,
        targets=[fast_worker, thorough_worker],
        selection_func=route_by_priority,
    )
    .build()
)
```

The selection function is synchronous; the framework calls it once per outgoing message, before dispatching to the chosen subset. Return an empty list to drop the message entirely.

#### Chain shorthand

```python
# A → B → C with a single call instead of three .add_edge calls
workflow = WorkflowBuilder(start_executor=step_a).add_chain([step_a, step_b, step_c]).build()
```

#### Conditional single edge

```python
# Only route to reviewer if the planner flagged the result as draft
workflow = (
    WorkflowBuilder(start_executor=planner)
    .add_edge(planner, reviewer, condition=lambda msg: getattr(msg, "is_draft", False))
    .build()
)
```

## Wrapping a workflow as an agent — `Workflow.as_agent`

Any `Workflow` can be exposed as an `Agent` — same `.run(...)` surface, same streaming events, same compatibility with orchestration builders. This lets you plug a whole multi-step pipeline into a higher-level orchestration as if it were a single agent, without manually instantiating `WorkflowAgent`:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import SequentialBuilder, ConcurrentBuilder

client = OpenAIChatClient()

# A peer agent that runs alongside the inner pipeline.
fact_checker = Agent(
    client=client,
    name="fact-checker",
    instructions="Flag any claim that cannot be verified against a primary source.",
)

# Inner pipeline — takes a topic, returns a dossier.
dossier_pipeline = SequentialBuilder(participants=[researcher, analyst]).build()

# Wrap it as an agent so it composes with other builders.
dossier_agent = dossier_pipeline.as_agent(
    name="dossier-agent",
    description="Research a topic and produce a structured dossier.",
)

# Outer pipeline — dossier agent runs alongside the fact-checker in parallel.
merged = (
    ConcurrentBuilder(participants=[dossier_agent, fact_checker])
    .build()
)

result = await merged.run("Post-quantum cryptography migration")
```

Caveats worth knowing:

- The workflow's **start executor must accept `list[Message]`** as one of its input types. `Workflow.as_agent()` raises `ValueError` at wrap time if the start executor can't accept the converted messages — check early, not at run time.
- You can pass `context_providers=` to the wrapped agent, same as a normal `Agent`. The providers run around every invocation of the inner workflow.
- The returned `WorkflowAgent` is itself a `SupportsAgentRun`, so it drops into `SequentialBuilder`, `HandoffBuilder`, `AgentExecutor`, or any other place that takes an agent.

### Converting string inputs to messages

The wrapping layer normalises the caller's input (string / `Message` / list of either) into `list[Message]` before handing it to the workflow's start executor. Inside the workflow, the start executor decorates its handler with `input=list[Message]`:

```python
from agent_framework import Executor, Message, WorkflowContext, handler


class IntakeExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="intake")

    @handler
    async def on_messages(
        self,
        messages: list[Message],            # must be list[Message] for as_agent() compat
        ctx: WorkflowContext[dict],
    ) -> None:
        topic = messages[-1].text           # last user message as the topic
        await ctx.send_message({"topic": topic})
```

If your start executor was written to accept a `dict` directly, either rewrite it to accept `list[Message]` or sandwich a small intake executor in front of it when you want the `as_agent()` surface.

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

### `FanOutEdgeGroup` rules and serialization

`.add_fan_out_edges(...)` and `.add_multi_selection_edge_group(...)` both instantiate a `FanOutEdgeGroup` under the hood. A few constraints the class enforces that are easy to miss:

- **At least two targets.** The constructor raises `ValueError("FanOutEdgeGroup must contain at least two targets.")` when you pass a single-target list. Drop to `.add_edge(source, target)` for 1:1 flows.
- **Stable IDs for selection functions.** When the graph is serialised (checkpointing, `.to_dict()` / `.from_dict()` round trips), the selector callable itself cannot cross the wire. Pass `selection_func_name=` with a stable identifier so deserialisation can re-resolve the callable from your registry.

```python
from agent_framework import FanOutEdgeGroup

# Not using the builder helper — useful for code that introspects / serialises
# the edge group itself. The builder's .add_multi_selection_edge_group(...) is
# still the normal way to add one to a workflow.
broadcast = FanOutEdgeGroup(
    source_id="dispatcher",
    target_ids=["worker_a", "worker_b", "worker_c"],
    selection_func=lambda msg, available: (
        available if msg.get("broadcast") else [available[0]]
    ),
    selection_func_name="broadcast_or_primary",  # stable — used during deserialisation
    id="primary-dispatch",                         # stable edge-group id
)

assert broadcast.target_ids == ["worker_a", "worker_b", "worker_c"]
# target_ids returns a shallow copy so callers can't mutate the group in place.
```

If you don't pass `selection_func_name=` the framework tries to derive it from the callable's `__qualname__`; lambdas and closures don't have a useful one, which is why you'll want the explicit name for anything you plan to persist.

### `FanOutEdgeGroup` selectors — runtime routing

`selection_func` is invoked **per message** with `(payload, available_target_ids)` and must return the subset of ids that should receive the payload. Returning all of them is equivalent to omitting the selector. Returning `[]` causes the message to be dropped — use this when the right answer to "where should this go?" is sometimes "nowhere".

A real-world example: shard a stream of orders by region, with a fallback for unknown regions:

```python
from typing import Any
from agent_framework import (
    FanOutEdgeGroup,
    FunctionExecutor,
    WorkflowBuilder,
)


def shard_by_region(order: dict[str, Any], available: list[str]) -> list[str]:
    region = (order.get("region") or "").lower()
    if region == "eu":
        return ["worker_eu"]
    if region in ("us", "ca"):
        return ["worker_na"]
    if region in ("jp", "kr", "sg", "in"):
        return ["worker_apac"]
    return ["dead_letter"]                       # everything else


# Stand-in executors — in a real workflow these would be AgentExecutors
# wrapping region-specific agents.
ingest = FunctionExecutor(lambda raw: raw, id="ingest")
worker_eu   = FunctionExecutor(lambda o: o, id="worker_eu")
worker_na   = FunctionExecutor(lambda o: o, id="worker_na")
worker_apac = FunctionExecutor(lambda o: o, id="worker_apac")
dead_letter = FunctionExecutor(lambda o: o, id="dead_letter")

shard = FanOutEdgeGroup(
    source_id=ingest.id,
    target_ids=[worker_eu.id, worker_na.id, worker_apac.id, dead_letter.id],
    selection_func=shard_by_region,
    selection_func_name="shard_by_region",      # stable id for serialisation
    id="region-shard",
)

workflow = (
    WorkflowBuilder(start_executor=ingest)
    .add_executors([worker_eu, worker_na, worker_apac, dead_letter])
    .add_edge_group(shard)
    .build()
)
```

Three properties of the selector that surface in serialisation and debugging:

- **`available` is a defensive copy.** Mutating the list inside the selector is harmless to the framework. Return a new list — don't try to "filter in place."
- **The callable does NOT cross checkpoints.** When the workflow is checkpointed (`checkpoint_storage=...`) only `selection_func_name` is persisted. On resume, the runner looks up the same name in the running process — keep the selector module-level so the import path stays stable.
- **No selector means broadcast.** A `FanOutEdgeGroup` constructed with `selection_func=None` (or no kwarg at all) sends every message to every target — the same semantics as `.add_fan_out_edges(...)` without a selector.

Inspect a configured group at runtime — the API exposes a snapshot of the configuration without leaking the live callable identity:

```python
print(shard.target_ids)          # ['worker_eu', 'worker_na', 'worker_apac', 'dead_letter']
print(shard.selection_func is shard_by_region)  # True — same callable, no copy
print(shard.to_dict()["selection_func_name"])   # 'shard_by_region'
```

### Switch-case with `Case` and `Default`

`add_switch_case_edge_group` accepts a list of `Case` predicates plus a terminal `Default`. The first matching `Case` wins — evaluation is top-to-bottom — so order your conditions from most specific to least specific:

```python
from dataclasses import dataclass
from agent_framework import (
    Case,
    Default,
    Executor,
    FunctionExecutor,
    WorkflowBuilder,
)


@dataclass
class Ticket:
    category: str
    priority: str


class DeadLetter(Executor):
    def __init__(self) -> None:
        super().__init__(id="dead_letter", defer_discovery=True)


# Minimal stand-ins — in a real workflow these would be AgentExecutors wrapping
# named agents, or FunctionExecutors running your handler code.
classifier   = FunctionExecutor(lambda raw: Ticket(**raw), id="classify")
vip_billing  = FunctionExecutor(lambda t: "vip billing handled",   id="vip_billing")
billing      = FunctionExecutor(lambda t: "billing handled",       id="billing")
technical    = FunctionExecutor(lambda t: "technical handled",     id="technical")


workflow = (
    WorkflowBuilder(start_executor=classifier)
    .add_switch_case_edge_group(
        source=classifier,
        cases=[
            # Most specific first — high-priority billing jumps the queue.
            Case(
                condition=lambda t: t.priority == "P0" and t.category == "billing",
                target=vip_billing,
            ),
            Case(condition=lambda t: t.category == "billing",   target=billing),
            Case(condition=lambda t: t.category == "technical", target=technical),
            # Default is mandatory — guarantees routing never produces an empty target.
            Default(target=DeadLetter()),
        ],
    )
    .build()
)
```

`Default` is not optional — the framework guarantees every message lands somewhere, even if no `Case` predicate matches. Point its target at a dead-letter executor if that's the correct behaviour for "I have no idea what to do with this."

### Constructing `FunctionExecutor` directly

`@executor` is the everyday entry point — it wraps a standalone module-level function in a `FunctionExecutor`. Construct the class directly when you need to:

- Pass an explicit `input` / `output` / `workflow_output` type that overrides what introspection finds (or when the function lacks annotations).
- Build executors dynamically from configuration (loop over a registry of handler callables).
- Construct from string forward references (`input="MyType | int"`) without `from __future__ import annotations` boilerplate.

```python
from agent_framework import FunctionExecutor, WorkflowContext


# Sync function — runs in the asyncio thread pool, no event loop blocked.
def normalise(text: str) -> str:
    return text.strip().lower()


normalise_node = FunctionExecutor(
    normalise,
    id="normalise",
    input=str,
    output=str,
)

# Async function with explicit, narrower output types — handy when introspection
# would pick up a wider union from the WorkflowContext type parameter.
async def classify(text: str, ctx: WorkflowContext[str, str]) -> None:
    label = "billing" if "invoice" in text else "general"
    await ctx.send_message(label)


classify_node = FunctionExecutor(classify, id="classify", output=str)
```

A few rules the constructor enforces — each raises `ValueError` at build time so wiring bugs surface before the workflow runs:

- **Standalone functions only.** Passing a `staticmethod` or `classmethod` raises with a hint to use the `Executor` base class + `@handler` instead.
- **Either 1 or 2 parameters.** `(message)` or `(message, ctx)`. Three or more is rejected.
- **Message annotation required** unless `input=` is supplied explicitly. Generic `TypeVar`s aren't allowed — give a concrete type.

`@executor` is just sugar over this constructor — the equivalent decorator form is `@executor(id="normalise", input=str, output=str)`. Use the class form when you want to keep the executor reference local rather than turning the function name into a module-level executor instance.

### Async predicates on edges

`Edge.condition` and the predicates inside `Case(...)` may be **either sync or async** — `EdgeCondition` is `Callable[[Any], bool | Awaitable[bool]]`. Use this when routing depends on an out-of-band check (auth lookup, feature flag, vector lookup) that you don't want to block the event loop on:

```python
from agent_framework import Case, Default, WorkflowBuilder


async def is_premium_tenant(msg: dict) -> bool:
    return await tenant_lookup.is_premium(msg["tenant_id"])    # async I/O


builder = (
    WorkflowBuilder(start_executor=triage)
    .add_switch_case_edge_group(
        source=triage,
        cases=[
            Case(condition=is_premium_tenant, target=premium_route),
            Default(target=standard_route),
        ],
    )
)
```

The framework `await`s the coroutine and treats the boolean result as the routing decision. Wrap the lookup in a cache or short timeout — predicates run on the hot path between supersteps, so a slow predicate stalls the entire workflow.

### Building edge groups directly — `SwitchCaseEdgeGroup`, `FanOutEdgeGroup`, `FanInEdgeGroup`

The builder helpers (`.add_switch_case_edge_group`, `.add_fan_out_edges`, `.add_fan_in_edges`) wrap these classes. Constructing them yourself is useful when:

- You're persisting a workflow definition and need stable, hand-written ids on each edge group.
- You're introspecting the topology programmatically (debug tooling, viz layers) and want to round-trip through `to_dict()` / `from_dict()`.
- You need to register the same `FanOutEdgeGroup` against multiple parent workflows.

```python
from agent_framework import (
    FanInEdgeGroup,
    FanOutEdgeGroup,
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
    WorkflowBuilder,
)

# Fan-out: one upstream → many. Constructor enforces ≥ 2 targets.
broadcast = FanOutEdgeGroup(
    source_id="dispatcher",
    target_ids=["worker_a", "worker_b", "worker_c"],
    id="primary-broadcast",                   # stable id (optional)
)
assert broadcast.target_ids == ["worker_a", "worker_b", "worker_c"]
assert broadcast.selection_func is None      # no selector → all targets receive

# Fan-in: many sources → one target. Also enforces ≥ 2 sources.
collector = FanInEdgeGroup(
    source_ids=["worker_a", "worker_b", "worker_c"],
    target_id="merger",
    id="merger-fanin",
)

# Switch/case: each Case carries its predicate; one Default is mandatory.
switch = SwitchCaseEdgeGroup(
    source_id="router",
    cases=[
        SwitchCaseEdgeGroupCase(
            condition=lambda payload: payload["kind"] == "csv",
            target_id="csv_handler",
        ),
        SwitchCaseEdgeGroupCase(
            condition=lambda payload: payload["kind"] == "json",
            target_id="json_handler",
        ),
        SwitchCaseEdgeGroupDefault(target_id="dead_letter"),
    ],
)

# Each edge group has a stable to_dict() shape — useful for diffing or logging.
snapshot = switch.to_dict()
assert snapshot["cases"][0]["type"] == "Case"
assert snapshot["cases"][-1]["type"] == "Default"
```

Key constraints surfaced by the constructors (each raises `ValueError` early so wiring bugs fail at workflow build time, not at run time):

- `FanOutEdgeGroup` — minimum 2 targets. For 1:1, use `.add_edge(source, target)`.
- `FanInEdgeGroup` — minimum 2 sources. For 1:1, use `.add_edge(source, target)`.
- `SwitchCaseEdgeGroup` — minimum 2 cases including the default; **exactly one** `SwitchCaseEdgeGroupDefault`. The framework warns (rather than errors) if `Default` isn't last, but cases evaluate top-to-bottom so an early `Default` short-circuits everything after it.
- `SwitchCaseEdgeGroupCase` — `target_id` is required and non-empty.

Once built, register the group with `WorkflowBuilder.add_edge_group(...)` (the low-level entry point that all the helpers funnel into):

```python
builder = WorkflowBuilder(start_executor=dispatcher)
builder.add_edge_group(broadcast)
builder.add_edge_group(collector)
```

#### Round-tripping through serialised form

When a callable can't be persisted (lambdas, closures, instance methods on objects unavailable at load time), supply `selection_func_name=` (fan-out) or `condition_name=` (switch case) so the deserialised group can re-resolve the callable from your registry. The clean pattern is to read the persisted name and reconstruct the case fresh with the live callable — never patch the placeholder in place:

```python
persisted = {"target_id": "csv_handler", "condition_name": "is_csv_payload"}

# A peek at the deserialised case tells you which name to look up.
stub = SwitchCaseEdgeGroupCase.from_dict(persisted)
assert stub.condition_name == "is_csv_payload"     # placeholder raises if invoked

# Rebuild via the public constructor with the resolved callable.
restored = SwitchCaseEdgeGroupCase(
    condition=condition_registry["is_csv_payload"],
    target_id=stub.target_id,
    condition_name=stub.condition_name,
)
```

The `from_dict`-produced placeholder fails loudly (`RuntimeError: Callable 'is_csv_payload' is unavailable after serialization`) so a forgotten registration crashes the run instead of silently routing to the wrong branch. Reconstructing rather than mutating the placeholder keeps you on the supported public surface — `_condition` is an internal field and may be renamed without notice.

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

## Further `WorkflowBuilder` patterns

### Dynamic fan-out with a selection function

`add_multi_selection_edge_group` evaluates a selector per payload and sends the message to the subset of targets it returns. Good for priority routing or feature-flag gated workers:

```python
from dataclasses import dataclass
from agent_framework import WorkflowBuilder


@dataclass
class Task:
    priority: str
    data: str


def pick_workers(task: Task, available: list[str]) -> list[str]:
    if task.priority == "high":
        return available                    # broadcast to all workers
    return [available[0]]                   # single worker for low-priority


workflow = (
    WorkflowBuilder(start_executor=dispatcher)
    .add_multi_selection_edge_group(
        dispatcher,
        [worker_a, worker_b, worker_c],
        selection_func=pick_workers,
    )
    .build()
)
```

### Collecting outputs from a subset of executors

By default the workflow yields outputs from any executor that calls `ctx.yield_output(...)`. Restrict to specific ones with `output_executors=` — useful when upstream nodes emit debug traces you don't want in the final `get_outputs()`:

```python
workflow = (
    WorkflowBuilder(
        start_executor=classifier,
        output_executors=[billing, refund, fallback],
    )
    .add_switch_case_edge_group(classifier, cases=[...])
    .build()
)
```

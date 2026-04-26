---
title: "Microsoft Agent Framework (Python) — Functional workflows"
description: "@workflow / @step decorators, RunContext, FunctionalWorkflow, and FunctionalWorkflowAgent — write workflows as plain async Python with checkpointing, HITL, and streaming."
framework: microsoft-agent-framework
language: python
---

# Functional workflows — Python

Graph-based workflows (`WorkflowBuilder`, `add_edge`, `add_fan_out_edges` …) work great when the topology is fixed. When branching and parallelism are *dynamic*, write the workflow as plain async Python and decorate it with `@workflow`. You get the same primitives — checkpointing, HITL via `request_info`, streaming, an agent adapter — but with native control flow.

Imports are stable in `agent-framework-core==1.2.0` (the feature is gated under `ExperimentalFeature.FUNCTIONAL_WORKFLOWS` — using it emits one `ExperimentalWarning` per process).

## Hello, workflow

```python
import asyncio
from agent_framework import step, workflow


@step
async def to_upper(text: str) -> str:
    return text.upper()


@step
async def add_exclaim(text: str) -> str:
    return f"{text}!"


@workflow
async def shout(message: str) -> str:
    loud = await to_upper(message)
    return await add_exclaim(loud)


async def main():
    result = await shout.run("hello")
    print(result.get_outputs())   # ['HELLO!']


asyncio.run(main())
```

A few things to notice:

- The workflow function is just an `async def`. Native `if/else`, `for`, `asyncio.gather` all work.
- Each `@step` is cached by `(step_name, call_index)` — see [Caching](#caching-and-replay) below.
- The decorator returns a `FunctionalWorkflow` (`shout` here) that exposes `.run()` and `.run_streaming()` — not the original function. Call it via `shout.run(...)` rather than `shout(...)`.
- `run()` returns a `WorkflowRunResult` with `.get_outputs()`, `.get_request_info_events()`, etc.

## `@workflow` and `@step` — what the decorators do

`@step` wraps an async function in a `StepWrapper`. Inside a running workflow, the wrapper intercepts the call to:

- emit `executor_invoked` / `executor_completed` / `executor_failed` events,
- cache the result by `(step_name, call_index)` so HITL replay and checkpoint restore don't re-run completed work,
- inject `RunContext` automatically when the step declares one,
- snapshot a checkpoint after each step when checkpoint storage is configured.

Outside a workflow, the wrapper is transparent — `await my_step(...)` just calls the underlying coroutine. That makes step functions trivially unit-testable.

`@workflow` returns a `FunctionalWorkflow`. The decorator validates the signature: at most one non-`RunContext` parameter is allowed (it receives whatever you pass to `.run(...)`).

```python
from agent_framework import RunContext, step, workflow


# Both decorator forms work:
@workflow
async def basic(message: str) -> str: ...


@workflow(name="my_pipeline", description="Documented pipeline")
async def parameterised(message: str, ctx: RunContext) -> str: ...
```

## `RunContext` — the workflow superpowers

`RunContext` is opt-in. Add it as a typed parameter (`ctx: RunContext`) or by name (`ctx`) in either the workflow function or any `@step`:

| Method | Use when |
|---|---|
| `await ctx.request_info(payload, response_type)` | Pause for human input (HITL) |
| `await ctx.add_event(event)` | Emit a custom event into the workflow stream |
| `ctx.get_state(key, default)` / `ctx.set_state(key, value)` | Workflow-scoped key/value store, included in checkpoints |
| `ctx.is_streaming()` | Branch behaviour when `stream=True` was passed |

```python
from agent_framework import RunContext, step, workflow


@step
async def review(draft: str, ctx: RunContext) -> str:
    """Pause the workflow until a human supplies edits."""
    edits = await ctx.request_info(
        request_data={"draft": draft, "instructions": "Tighten the prose."},
        response_type=str,
    )
    return edits or draft


@workflow
async def edit_pipeline(topic: str, ctx: RunContext) -> str:
    ctx.set_state("started_at", asyncio.get_event_loop().time())

    draft = f"Initial draft about {topic}."
    final = await review(draft)

    await ctx.add_event(
        # Custom event lands in the workflow event stream.
        # WorkflowEvent constructors live on the WorkflowEvent class itself.
        __import__("agent_framework").WorkflowEvent.executor_completed(
            executor_id="custom_marker",
            output={"length": len(final)},
        )
    )
    return final
```

State keys must not start with `_` — that's reserved for framework bookkeeping (`_step_cache`, `_original_message`, …) and the constructor raises `ValueError` if you try.

## Caching and replay

Inside a workflow, every call to a `@step` is keyed `(step_name, call_index)`:

- `call_index` increments for each *call* to the same step name in the run, so a `for` loop over the same step produces a deterministic key per iteration.
- The cache survives HITL replay and checkpoint restore. On replay, the workflow function executes from the top *again*, but cached steps return their stored result and emit a single `executor_bypassed` event instead of `executor_invoked` / `executor_completed`.
- `request_info` IDs auto-generated inside a step (when you didn't pass `request_id=`) follow the same determinism contract — they're indexed by call order so replay maps responses back to the right call.

Practical implication: keep workflow logic pure between cached steps. If your workflow function reads `random.random()` outside of a step, replay will see a different value.

## HITL — the resume cycle

`ctx.request_info` is the entry point. The first call raises an internal interrupt that the framework catches; the `WorkflowRunResult` contains pending request events:

```python
from agent_framework import RunContext, workflow


@workflow
async def approve_then_ship(payload: dict, ctx: RunContext) -> dict:
    decision = await ctx.request_info(
        {"item": payload["item"], "amount": payload["amount"]},
        response_type=str,
        request_id="approval",      # explicit id makes resume keys easy
    )
    if decision != "approved":
        return {"status": "rejected", "reason": decision}
    return {"status": "shipped", "item": payload["item"]}


# 1. First run — pauses on the request_info call.
result = await approve_then_ship.run({"item": "widget-42", "amount": 99})
events = result.get_request_info_events()
print(events[0].request_id)        # 'approval'
print(events[0].request_data)      # {'item': 'widget-42', 'amount': 99}

# 2. Resume with the response.
result2 = await approve_then_ship.run(responses={"approval": "approved"})
print(result2.get_outputs())       # [{'status': 'shipped', 'item': 'widget-42'}]
```

`responses=` and `message=` are mutually exclusive — supply either a fresh input or HITL responses, not both. (You *can* combine `responses=` with `checkpoint_id=` to restore a saved run and inject responses in a single call.)

If a workflow has no pending requests, calling it with `responses=` raises a `ValueError` so you can't accidentally replay against stale state.

## Checkpointing

Pass a `CheckpointStorage` to the decorator (or per-run) to snapshot after every step:

```python
from agent_framework import FileCheckpointStorage, RunContext, step, workflow


storage = FileCheckpointStorage("./checkpoints")


@step
async def fetch(url: str) -> str:
    return f"<contents of {url}>"


@step
async def summarise(text: str) -> str:
    return text[:200]


@workflow(checkpoint_storage=storage)
async def fetch_and_summarise(url: str) -> str:
    raw = await fetch(url)
    return await summarise(raw)


# Run normally — each step result lands in ./checkpoints/.
result = await fetch_and_summarise.run("https://example.com")

# … process crashes, restart …

# Resume from the latest checkpoint:
result2 = await fetch_and_summarise.run(checkpoint_id="<id>")
```

The checkpoint encodes the step cache, the workflow's user state (`set_state` values), pending request_info events, and the original input. JSON-serialisable values only — if you need to persist domain objects, register their `to_dict`/`from_dict` (see the [Sessions](./microsoft_agent_framework_python_sessions/) guide for the same pattern).

Override storage per run with `checkpoint_storage=`:

```python
from agent_framework import InMemoryCheckpointStorage

result = await fetch_and_summarise.run(
    "https://example.com",
    checkpoint_storage=InMemoryCheckpointStorage(),   # tests
)
```

## Streaming

`stream=True` returns a `ResponseStream[WorkflowEvent[Any], WorkflowRunResult]`:

```python
from agent_framework import workflow


@workflow
async def write(topic: str) -> str:
    return f"Article about {topic}."


stream = write.run("dolphins", stream=True)
async for event in stream:
    print(event)         # WorkflowEvent(...) — executor_invoked / completed / status / output / …

final = await stream.get_final_response()    # WorkflowRunResult
print(final.get_outputs())
```

`ctx.is_streaming()` lets the workflow itself branch on streaming mode — useful when you want to emit progress events only when a UI is listening.

## `FunctionalWorkflowAgent` — exposing a workflow as an agent

A functional workflow is callable on its own, but if you want to drop it into anywhere an `Agent` is expected — orchestration builders, A2A, Foundry — wrap it in `FunctionalWorkflowAgent`:

```python
from agent_framework import FunctionalWorkflowAgent, workflow


@workflow
async def research(query: str) -> dict:
    return {"summary": f"Research on {query}", "sources": []}


# Adapt the workflow to the agent protocol.
research_agent = FunctionalWorkflowAgent(research, name="researcher")
print(research_agent.name)        # 'researcher'
print(research_agent.id)          # 'FunctionalWorkflowAgent_researcher'

# Use it like any other agent — same .run() signature.
response = await research_agent.run("dolphins")
print(response.text)              # the workflow output, serialised as text
```

Behind the scenes the adapter:

- Exposes the same overloaded `.run(messages, *, stream=...)` signature as `BaseAgent`, returning `AgentResponse` (non-streaming) or `ResponseStream[AgentResponseUpdate, AgentResponse]` (streaming).
- Translates `request_info` events into `FunctionApprovalRequestContent` items on the response, so HITL flows are callable through the agent surface.
- Keeps the original workflow's `name` and `description` unless you override them.

### HITL through the agent surface

```python
from agent_framework import FunctionalWorkflowAgent, RunContext, workflow


@workflow
async def confirm_then_send(message: str, ctx: RunContext) -> dict:
    answer = await ctx.request_info(
        {"prompt": f"Send this message? {message!r}"},
        response_type=str,
        request_id="confirm",
    )
    if answer.lower() != "yes":
        return {"sent": False, "reason": answer}
    return {"sent": True}


agent = FunctionalWorkflowAgent(confirm_then_send, name="sender")

# 1. First call — pauses on request_info.
response = await agent.run("Hello world")
for req in response.user_input_requests:
    print(req.user_input_request)  # surfaces the {prompt: ...} payload

# 2. Resume — same request_id key as the workflow.
final = await agent.run(responses={"confirm": "yes"})
print(final.text)                  # '{"sent": true}'
```

The `pending_requests` property on the agent is a dict of any unresolved `WorkflowEvent` from the last run — useful when you need to inspect the pending IDs programmatically.

### Composing into orchestration

Because `FunctionalWorkflowAgent` satisfies `SupportsAgentRun`, you can drop it into any orchestration builder alongside vanilla `Agent`s:

```python
from agent_framework import (
    Agent, FunctionalWorkflowAgent, SequentialBuilder, workflow,
)
from agent_framework.openai import OpenAIChatClient


@workflow
async def fetch_and_clean(url: str) -> str:
    # ... fetch + clean ...
    return "cleaned text"


fetcher = FunctionalWorkflowAgent(fetch_and_clean, name="fetcher")
summariser = Agent(client=OpenAIChatClient(), name="summariser")

pipeline = (
    SequentialBuilder()
    .add(fetcher)
    .add(summariser)
    .build()
)
```

The fetcher runs as deterministic Python; the summariser does the LLM work. Mixing functional workflows with chat agents this way keeps the deterministic parts of your pipeline cheap, debuggable, and unit-testable.

## When to choose functional vs. graph

| Use functional (`@workflow`) when | Use graph (`WorkflowBuilder`) when |
|---|---|
| Branching depends on **runtime data** (user input, retrieval scores) | Topology is **fixed and visualised** ahead of time |
| You want native `try`/`except`, `for`, `asyncio.gather` | You need declarative YAML / JSON workflows |
| Steps are mostly Python, sprinkled with one or two LLM calls | The graph is the artefact reviewers see |
| You're moving an existing async script behind a workflow boundary | Multi-team approval flows where the topology is the contract |

Both run on the same checkpoint / event / HITL plumbing, so you can mix them — call a graph workflow from inside a `@workflow`, or wrap a `FunctionalWorkflowAgent` as a node inside a graph.

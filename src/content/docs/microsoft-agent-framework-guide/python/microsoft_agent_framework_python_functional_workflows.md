---
title: "Microsoft Agent Framework (Python) — Functional Workflows"
description: "Build workflows with plain Python functions using the @workflow and @step decorators, RunContext for HITL and state, FunctionalWorkflow checkpointing, and as_agent() for multi-agent composition. Verified against agent-framework-core 1.4.0."
framework: microsoft-agent-framework
language: python
---

# Functional Workflows — Python

> **Experimental.** `FunctionalWorkflow`, `@workflow`, `@step`, and `RunContext` are marked `ExperimentalFeature` in `agent-framework-core==1.4.0`. The API is stable enough to build on but may change between minor releases.

Functional workflows let you write a workflow as a plain `async` Python function — no executor classes, no graph wiring, no edge objects. Control flow is ordinary Python: `if`/`else`, `for`, `asyncio.gather`. The framework tracks step results, emits events, handles HITL pauses, and persists checkpoints automatically.

Verified against `agent-framework-core==1.4.0` (`agent_framework._workflows._functional`).

## When to choose functional vs graph workflows

| Situation | Prefer |
|---|---|
| Variable branching, dynamic parallelism, complex Python logic | `@workflow` (functional) |
| Fixed topology, reuse executor classes, time-travel debugging | `WorkflowBuilder` (graph) |
| Expose a workflow as a node inside a larger graph | Use `.as_agent()` on either |
| Team already uses LangGraph / other graph-style orchestration | `WorkflowBuilder` — familiar mental model |

## The three building blocks

| API | Purpose |
|---|---|
| `@workflow` | Decorator that converts an `async` function into a `FunctionalWorkflow` |
| `@step` | Optional decorator that marks a function as a **tracked step** — enables caching, events, per-step checkpointing |
| `RunContext` | Injected handle for HITL (`request_info`), workflow-scoped state, and custom events |

## Minimal example

```python
import asyncio
from agent_framework import Agent, workflow
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", instructions="Return concise bullet-point facts.")
writer = Agent(client=client, name="writer", instructions="Expand bullets into a short paragraph.")


@workflow
async def research_pipeline(topic: str) -> str:
    facts = await researcher.run(f"Research: {topic}")
    article = await writer.run(f"Expand:\n{facts.text}")
    return article.text


result = asyncio.run(research_pipeline.run("solar-sail propulsion"))
print(result.get_outputs()[-1])
```

`@workflow` wraps the function in a `FunctionalWorkflow`. Call `.run(message)` exactly as you would a graph `Workflow`.

## `@workflow` decorator — both forms

```python
from agent_framework import workflow

# Bare form — workflow name defaults to the function name
@workflow
async def my_pipeline(data: str) -> str: ...

# Parameterised form — override name/description
@workflow(name="Content Pipeline", description="Research → write → review")
async def content_pipeline(topic: str) -> str: ...
```

Constructor signature (accessed indirectly via the decorator):

```python
FunctionalWorkflow(
    func,
    *,
    name: str | None = None,              # defaults to func.__name__
    description: str | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
)
```

## `@step` — tracked execution units

Plain `async` functions work inside `@workflow` without `@step`. Add `@step` only when you want:

- **Result caching** — if the workflow is resumed from a checkpoint, completed steps aren't re-executed.
- **Event emission** — the framework emits a `step_started` / `step_completed` event pair in the stream.
- **Per-step granularity in checkpoints** — the step name and index appear in checkpoint metadata.

```python
import asyncio
from agent_framework import Agent, step, workflow
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
researcher = Agent(client=client, name="researcher", instructions="Return facts as bullets.")
writer = Agent(client=client, name="writer", instructions="Turn bullets into a paragraph.")
reviewer = Agent(client=client, name="reviewer", instructions="Rate quality 1–10 and give one-line feedback.")


@step
async def research(topic: str) -> str:
    result = await researcher.run(f"Research: {topic}")
    return result.text


@step
async def write(facts: str) -> str:
    result = await writer.run(f"Expand:\n{facts}")
    return result.text


@step(name="quality-review")
async def review(article: str) -> str:
    result = await reviewer.run(f"Review:\n{article}")
    return result.text


@workflow(name="editorial-pipeline")
async def editorial_pipeline(topic: str) -> str:
    facts = await research(topic)
    draft = await write(facts)
    verdict = await review(draft)
    return f"{draft}\n\n[Review: {verdict}]"


result = asyncio.run(editorial_pipeline.run("fusion energy breakthroughs"))
print(result.get_outputs()[-1])
```

`@step` supports both bare (`@step`) and parameterised (`@step(name="...")`) forms.

### Steps are independently testable

`@step` returns the decorated function unchanged — you can call it directly from tests without a running workflow:

```python
import asyncio

# No workflow context needed — behaves exactly like the original async function
facts = asyncio.run(research("quantum computing"))
assert "qubit" in facts.lower()
```

## `RunContext` — HITL, state, and events

Declare a `RunContext` parameter (by type annotation or by the name `ctx`) anywhere in the `@workflow` function or inside `@step` functions to access workflow-only features:

```python
from agent_framework import RunContext, workflow
```

| Method | Purpose |
|---|---|
| `await ctx.request_info(data, response_type)` | Pause the workflow and ask for external input (human-in-the-loop) |
| `await ctx.add_event(WorkflowEvent(...))` | Emit a custom event into the run stream |
| `ctx.get_state(key, default)` | Read workflow-scoped key/value state |
| `ctx.set_state(key, value)` | Write workflow-scoped key/value state |
| `ctx.is_streaming` | `True` when the caller used `run(..., stream=True)` |

### Human-in-the-loop with `request_info`

```python
import asyncio
from dataclasses import dataclass
from agent_framework import Agent, RunContext, workflow
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
drafter = Agent(client=client, name="drafter", instructions="Draft a marketing headline.")
polisher = Agent(client=client, name="polisher", instructions="Polish the approved headline.")


@dataclass
class HeadlineReview:
    draft: str
    options: list[str] = None

    def __post_init__(self):
        self.options = ["approve", "reject"]


@workflow
async def headline_pipeline(product: str, ctx: RunContext) -> str:
    # Step 1 — draft
    draft_result = await drafter.run(f"Write a headline for: {product}")
    draft = draft_result.text

    # Step 2 — pause and ask a human to approve or reject
    decision = await ctx.request_info(
        HeadlineReview(draft=draft),
        response_type=str,           # "approve" or "reject"
    )

    if decision == "reject":
        return f"Headline rejected: {draft}"

    # Step 3 — polish the approved headline
    polished = await polisher.run(f"Polish: {draft}")
    return polished.text


async def run_with_review() -> None:
    # First run — pauses at request_info and returns the pending event
    result = await headline_pipeline.run("NebulaCloud storage service")

    pending = result.get_request_info_events()
    if not pending:
        print(result.get_outputs()[-1])
        return

    # Inspect the request and collect a human decision
    event = pending[0]
    review: HeadlineReview = event.data
    print(f"Draft headline: {review.draft}")
    print(f"Options: {review.options}")
    decision = input("approve or reject? ").strip().lower()

    # Resume with the human's answer keyed by the request ID
    resumed = await headline_pipeline.run(responses={event.request_id: decision})
    print(resumed.get_outputs()[-1])


asyncio.run(run_with_review())
```

Key points:
- `await ctx.request_info(data, response_type)` suspends the workflow on first call — the framework never exposes the internal `WorkflowInterrupted` signal to your code.
- `WorkflowRunResult.get_request_info_events()` returns a list of pending `WorkflowEvent` objects; each has a `request_id` and `data`.
- Resume by calling `run(responses={request_id: value})` — the same `@workflow` function re-executes and `request_info` returns `value` directly on the second pass.
- Pass `request_id=` explicitly to `request_info(...)` when you want a stable ID (e.g. one tied to a database row) rather than a generated UUID.

### Streaming with `request_info`

```python
async def stream_with_review(topic: str) -> None:
    pending: dict[str, str] = {}
    stream = headline_pipeline.run(topic, stream=True)

    while True:
        async for event in stream:
            if event.type == "request_info":
                review: HeadlineReview = event.data
                print(f"\nApprove '{review.draft}'? [approve/reject]")
                answer = await asyncio.to_thread(input, "> ")
                pending[event.request_id] = answer.strip()
            elif event.type == "output":
                print("Final:", event.data)
                return

        if not pending:
            return
        stream = headline_pipeline.run(responses=pending, stream=True)
        pending = {}
```

### Workflow-scoped state

```python
from agent_framework import RunContext, workflow


@workflow
async def stateful_pipeline(items: list[str], ctx: RunContext) -> str:
    ctx.set_state("total", len(items))
    ctx.set_state("processed", 0)

    results = []
    for item in items:
        processed = await some_expensive_step(item)
        results.append(processed)
        ctx.set_state("processed", ctx.get_state("processed") + 1)
        # Emit progress so a streaming caller can show a progress bar
        from agent_framework import WorkflowEvent
        await ctx.add_event(WorkflowEvent(
            type="progress",
            data={"done": ctx.get_state("processed"), "total": ctx.get_state("total")},
        ))

    return "\n".join(results)
```

State survives checkpoints — `get_state` / `set_state` values are persisted when a checkpoint is taken.

### `get_run_context()` — accessing RunContext from nested helpers

When a utility function deep in the call stack needs to emit events or read state, pass `ctx` explicitly *or* call `get_run_context()`. The `RunContext` is stored in a `ContextVar` for the duration of the `@workflow` call, so `get_run_context()` retrieves it from any depth without threading the parameter through every signature:

```python
import asyncio
from agent_framework import (
    Agent,
    RunContext,
    WorkflowEvent,
    get_run_context,
    step,
    workflow,
)
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
extractor = Agent(client=client, name="extractor", instructions="Extract five key facts as a numbered list.")


async def _emit_progress(message: str) -> None:
    """Utility helper — no RunContext parameter needed at the call site."""
    ctx = get_run_context()
    if ctx is not None:
        await ctx.add_event(WorkflowEvent(type="progress", data=message))


@step
async def extract_facts(topic: str) -> list[str]:
    await _emit_progress(f"Extracting facts about '{topic}' …")
    result = await extractor.run(f"Give me five facts about: {topic}")
    await _emit_progress("Extraction complete.")
    return [line for line in result.text.split("\n") if line.strip()]


@workflow
async def analysis_pipeline(topic: str) -> str:
    facts = await extract_facts(topic)
    return "\n".join(f"• {f}" for f in facts)


result = asyncio.run(analysis_pipeline.run("quantum entanglement"))
print(result.get_outputs()[-1])
```

`get_run_context()` returns `None` when called outside a running workflow, which makes helpers reusable from both workflow and non-workflow callsites. The guard `if ctx is not None` is the idiomatic pattern.

> **Thread safety note.** `ContextVar` propagation follows Python's standard rules — the value is inherited by tasks created with `asyncio.create_task()` and `asyncio.gather()`, but **not** by threads spawned with `asyncio.to_thread()` or `threading.Thread`. If you dispatch to a thread inside a `@step`, pass `ctx` explicitly rather than relying on `get_run_context()`.

## Parallel execution

Use native `asyncio` for parallelism — no special API needed:

```python
import asyncio
from agent_framework import Agent, step, workflow
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
angle_agent = Agent(client=client, name="angle", instructions="Give a one-line angle on this topic.")


@step
async def get_angle(topic: str, perspective: str) -> str:
    result = await angle_agent.run(f"Topic: {topic}. Perspective: {perspective}")
    return f"[{perspective}] {result.text}"


@workflow
async def multi_angle_research(topic: str) -> str:
    perspectives = ["technical", "business", "regulatory", "public opinion"]

    # Run all four angles in parallel
    angles = await asyncio.gather(*[get_angle(topic, p) for p in perspectives])

    return "\n".join(angles)


result = asyncio.run(multi_angle_research.run("AI Act compliance"))
print(result.get_outputs()[-1])
```

`@step` result caching is keyed by `(step_name, invocation_index)` — each call to `get_angle` in the gather gets its own cache slot, so resume-from-checkpoint correctly re-populates only the steps that didn't complete.

## Checkpointing

Pass a `CheckpointStorage` to persist step results across process restarts:

```python
import asyncio
from agent_framework import FileCheckpointStorage, workflow, step

storage = FileCheckpointStorage(base_path="./checkpoints")


@step
async def slow_research(topic: str) -> str:
    import time; time.sleep(2)          # simulated slow call
    return f"research on {topic}"


@step
async def slow_write(facts: str) -> str:
    import time; time.sleep(2)
    return f"article from: {facts}"


@workflow(checkpoint_storage=storage)
async def long_pipeline(topic: str) -> str:
    facts = await slow_research(topic)
    article = await slow_write(facts)
    return article


# First run — checkpoints every completed step
result = asyncio.run(long_pipeline.run("climate modelling"))

# If the process dies after slow_research and before slow_write, restart and
# resume — slow_research is not re-executed (cached in the checkpoint)
checkpoints = asyncio.run(storage.list_checkpoints(workflow_name="long_pipeline"))
latest = checkpoints[-1]
result = asyncio.run(long_pipeline.run(checkpoint_id=latest.checkpoint_id))
print(result.get_outputs()[-1])
```

Pass `checkpoint_storage=` either in `@workflow(checkpoint_storage=...)` (per-workflow default) or as a `run()` override — the `run()` argument takes precedence.

### `run()` parameters

```python
FunctionalWorkflow.run(
    message,                            # workflow input (first arg of your function)
    *,
    stream: bool = False,               # True → returns ResponseStream
    responses: dict[str, Any] | None = None,   # HITL reply dict
    checkpoint_id: str | None = None,   # resume from a specific checkpoint
    checkpoint_storage: CheckpointStorage | None = None,  # override
    include_status_events: bool = False,  # include internal status events in stream
    function_invocation_kwargs: ... = None,
    client_kwargs: ... = None,
)
```

## `FunctionalWorkflow.as_agent()` — composition

Wrap a functional workflow as an agent so it can participate in orchestrations:

```python
from agent_framework import Agent, FunctionalWorkflowAgent, workflow, step
from agent_framework_orchestrations import SequentialBuilder
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
extractor = Agent(client=client, name="extractor", instructions="Extract key claims from text.")
scorer = Agent(client=client, name="scorer", instructions="Score each claim 1–5 for credibility.")


@step
async def extract(text: str) -> str:
    r = await extractor.run(text)
    return r.text


@step
async def score(claims: str) -> str:
    r = await scorer.run(claims)
    return r.text


@workflow(name="fact-checker")
async def fact_checker(article: str) -> str:
    claims = await extract(article)
    scores = await score(claims)
    return scores


# Expose the functional workflow as a named agent
fact_checker_agent: FunctionalWorkflowAgent = fact_checker.as_agent(
    name="fact-checker",
    description="Extract and score factual claims from an article.",
)

# Use it anywhere an agent is expected — including orchestration builders
summariser = Agent(client=client, name="summariser", instructions="Summarise the scored claims.")

pipeline = SequentialBuilder(participants=[fact_checker_agent, summariser]).build()
result = asyncio.run(pipeline.run("... article text ..."))
print(result.get_outputs()[-1])
```

`as_agent()` returns a `FunctionalWorkflowAgent` — it has the same `run()` signature as `Agent` and plugs into `WorkflowBuilder`, `SequentialBuilder`, and the `as_tool()` helper:

```python
# Wrap the functional workflow agent as a tool for a supervisor agent
tool = fact_checker_agent.as_tool(
    description="Check factual claims in an article and return credibility scores.",
)

supervisor = Agent(client=client, instructions="You coordinate research and fact-checking.", tools=[tool])
```

## `WorkflowRunResult` helpers

```python
result = await my_workflow.run("input")

# All outputs yielded by the workflow (usually just the return value)
outputs = result.get_outputs()          # list[Any]
final = outputs[-1]                     # the last (or only) output

# Any pending HITL requests (non-empty when workflow paused at request_info)
events = result.get_request_info_events()

# Run state: "completed", "paused", "failed"
print(result.state)

# Workflow name and any custom events emitted during the run
print(result.workflow_name)
events_all = result.events
```

## Patterns

**Research pipeline with reviewer.** Use `@step` for the main stages (research, draft, review) and `request_info` in the final step so a human can approve before the document is published.

**Dynamic fan-out.** Use `asyncio.gather` over a list built at runtime — the step cache handles variable-length parallelism gracefully, where a static graph with hard-coded fan-out wouldn't.

**Multi-stage HITL.** Call `request_info` multiple times in the same workflow function; each call suspends independently. Respond to all pending events in a single `run(responses={...})` call to resume them together.

**Cost guardrail.** Track spending in `ctx.set_state("spend", ...)` across steps and call `request_info` when a budget is exceeded — the workflow pauses until a manager approves continuation.

**Incremental streaming UI.** `ctx.add_event(WorkflowEvent(type="progress", data=...))` inside a step gives a streaming caller real-time progress updates without polling.

**Testing without a model.** Since `@step` functions are callable as regular `async` functions, mock the underlying agents and call steps directly in pytest — the full `@workflow` integration test runs against the real model.

## See also

- [HITL — `request_info` in graph workflows](./microsoft_agent_framework_python_hitl/)
- [Checkpointing — `FileCheckpointStorage`](./microsoft_agent_framework_python_checkpointing/)
- [Workflows & Declarative Agents](../microsoft_agent_framework_graphs_declarative/) — graph-based `WorkflowBuilder` and YAML workflows
- [Orchestration](./microsoft_agent_framework_python_orchestration/) — `SequentialBuilder`, `MagenticBuilder`, etc.

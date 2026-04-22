---
title: "Workflows (graph orchestration)"
description: "Compose agents and functions into a DAG with conditional routing, retries, timeouts, HITL, and state schemas."
framework: google-adk
language: python
sidebar:
  order: 25
---

Verified against google-adk==2.0.0b1 (`google/adk/workflow/`).

`Workflow` is the graph-based orchestrator that replaces `SequentialAgent`, `ParallelAgent`, and `LoopAgent` in ADK 2.x. It is a `BaseNode` (not a `BaseAgent`) — wire it to a `Runner` via `App(root_agent=workflow)`.

## Public surface

`google.adk.workflow` exports (`workflow/__init__.py`):

| Name | Purpose |
|---|---|
| `Workflow` | Graph orchestrator. Takes `edges=[...]` and runs the DAG. |
| `BaseNode` | Pydantic base for every node. |
| `Node` | Subclass-friendly base — implement `run_node_impl`. |
| `node` | Decorator / function to wrap a callable, agent, or tool as a `BaseNode`. |
| `FunctionNode` | What `@node` produces for a function. |
| `JoinNode` | Fan-in — waits for all predecessors before emitting. |
| `Edge` | Explicit `from_node → to_node`, optional `route`. |
| `RetryConfig` | Per-node retry policy. |
| `NodeTimeoutError` | Raised when a node exceeds its `timeout`. |
| `START` | Sentinel for the graph entry point. |
| `DEFAULT_ROUTE` | Matches routes with no explicit mapping. |

## Minimal example

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START

drafter = LlmAgent(
    name="drafter",
    model="gemini-2.5-flash",
    instruction="Write a tight 3-sentence summary of the input.",
    mode="single_turn",
)
polisher = LlmAgent(
    name="polisher",
    model="gemini-2.5-flash",
    instruction="Shorten and sharpen the input. Return only the final text.",
    mode="single_turn",
)

@node
def trim(node_input: str) -> str:
    return node_input.strip()

pipeline = Workflow(
    name="summarize_pipeline",
    edges=[(START, trim, drafter, polisher)],
)

async def main():
    app = App(name="demo", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Electric cars sold well in Q1.", user_id="u1", session_id="s1"
    )

asyncio.run(main())
```

A tuple in `edges=` is a **chain** — each adjacent pair becomes an `Edge`. `START` is the sentinel that receives `new_message` on invocation.

## Edge syntax

Four ways to declare edges, all verified in `workflow/_workflow_graph.py`:

### 1. Chain (tuple)

```python
edges = [(START, a, b, c)]   # START→a, a→b, b→c
```

### 2. Fan-out (nested tuple)

```python
edges = [(START, (a, b, c), join)]   # START fans out to a/b/c, all feed join
```

### 3. Routing map (dict)

```python
edges = [
    (START, classifier, {"billing": billing_agent, "support": support_agent}),
]
```
Edges carry a `route` value. The source node picks one by setting `ctx.route = "billing"` (in a `FunctionNode`) or by the router's output matching a key. Use `DEFAULT_ROUTE` for the fallback.

### 4. Explicit `Edge`

```python
from google.adk.workflow import Edge
edges = [Edge(from_node=a, to_node=b, route="yes")]
```

Mix them freely. `BaseAgent`, `BaseTool`, and plain callables are auto-wrapped via `build_node()` when they appear in an edge.

## `@node` decorator

```python
from google.adk.workflow import node, RetryConfig

@node
async def fetch(node_input: str, ctx) -> dict:
    ctx.state["last_query"] = node_input
    return {"query": node_input, "results": [...]}

@node(
    name="safe_fetch",
    retry_config=RetryConfig(max_attempts=3),
    timeout=10.0,
    rerun_on_resume=True,
)
async def safe_fetch(node_input: str, ctx): ...
```

Signatures recognised by `FunctionNode`:
- `node_input` — the incoming value (the predecessor's output, or the user's `new_message` for START successors).
- `ctx` — the `Context` (state, `run_node`, `route`, `interrupt`, artifact helpers).
- Any other parameter must be declared in the enclosing `Workflow.state_schema` — the framework injects its current value.

Return types are honoured: returning a value sets `ctx.output`; yielding values from an async generator lets you stream partials.

## `Node` subclass

Use when you need class-level state or `parallel_worker=True`:

```python
from google.adk.workflow import Node
from collections.abc import AsyncGenerator
from typing import Any

class DedupeNode(Node):
    name: str = "dedupe"
    seen: set[str] = set()

    async def run_node_impl(self, *, ctx, node_input: list[str]) -> AsyncGenerator[Any, None]:
        fresh = [x for x in node_input if x not in self.seen]
        self.seen.update(fresh)
        yield fresh
```

Setting `parallel_worker=True` lets the node be invoked concurrently per trigger without sharing state with other invocations.

## `Workflow` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Must be a Python identifier |
| `edges` | `list[EdgeItem]` | `[]` | Tuples or `Edge` objects |
| `max_concurrency` | `int \| None` | `None` | Caps concurrent graph-scheduled nodes |
| `state_schema` | `type[BaseModel]` | `None` | Validates `ctx.state` mutations |
| `rerun_on_resume` | `bool` | `True` | Workflow-level resume behaviour |
| `input_schema` / `output_schema` | `SchemaType` | `None` | Validates workflow-level in/out |

All fields from `BaseNode` (`retry_config`, `timeout`, `wait_for_output`, ...) apply too.

## Routing and conditions

A node can steer the graph by setting `ctx.route`:

```python
@node
async def classify(node_input: str, ctx):
    intent = "billing" if "invoice" in node_input.lower() else "support"
    ctx.route = intent
    return node_input

workflow = Workflow(
    name="triage",
    edges=[
        (START, classify, {
            "billing": billing_agent,
            "support": support_agent,
            DEFAULT_ROUTE: fallback_agent,
        }),
    ],
)
```

## Loops

Replace `LoopAgent` with a self-referential edge and an escalate condition:

```python
from google.adk.workflow import Workflow, node, START, DEFAULT_ROUTE

@node(rerun_on_resume=True)
async def critic(draft: str, ctx) -> str:
    if len(draft) < 500:
        ctx.route = "done"
        return draft
    ctx.route = "continue"
    return draft[:500]  # trimmed draft fed back in

loop = Workflow(
    name="refine",
    edges=[
        (START, critic, {"continue": critic, "done": END_NODE}),
    ],
)
```

`max_concurrency=1` + a routing map back to the same node gives you a bounded loop. Persist iteration count in `ctx.state` if you need `max_iterations` semantics.

## JoinNode

Fan-in that waits for **all** declared predecessors. Useful after a `(START, (a, b, c), join)` fan-out.

```python
from google.adk.workflow import JoinNode, Workflow, START

join = JoinNode(name="merge")
@node
def finalize(node_input: dict) -> str:
    # node_input is a dict keyed by predecessor name
    return f"A={node_input['a']} B={node_input['b']}"

wf = Workflow(
    name="fanin",
    edges=[(START, (a, b), join, finalize)],
)
```

`JoinNode` receives a dict of `{predecessor_name: output}` as its `node_input`.

## Dynamic nodes (`ctx.run_node`)

Call another node from inside a node — the result is awaited in-place. **The caller must have `rerun_on_resume=True`** or `run_node` raises `ValueError` (`agents/context.py:399-405`).

```python
@node(rerun_on_resume=True)
async def supervisor(q: str, ctx):
    research = await ctx.run_node(research_agent, q)
    answer = await ctx.run_node(writer_agent, research)
    return answer
```

## Human-in-the-loop

Yield a `RequestInput` from a node to pause the workflow until `Runner.run_async` is called again with a `new_message` carrying a matching function-response. Pair with `auth_config=` on the `@node` to gate on an OAuth flow.

```python
from google.adk.events.request_input import RequestInput

@node(rerun_on_resume=True)
async def approve(draft: str, ctx):
    decision = yield RequestInput(
        id="approval",
        hint="Approve the draft? yes/no",
    )
    if decision == "yes":
        return draft
    ctx.route = "rewrite"
```

Combine with `ResumabilityConfig(is_resumable=True)` on the `App` so state is persisted across the pause.

## Retries and timeouts

```python
from google.adk.workflow import node, RetryConfig

@node(retry_config=RetryConfig(max_attempts=3, backoff_base=2.0), timeout=30.0)
async def flaky(q: str, ctx): ...
```

A node that exceeds `timeout` is cancelled and raises `NodeTimeoutError`. If retries are configured, the node is restarted.

## Patterns

### 1 — Linear pipeline
`edges=[(START, step1, step2, step3)]` — direct replacement for `SequentialAgent`.

### 2 — Map-reduce
`edges=[(START, split, (worker1, worker2, worker3), join, summarize)]` — fan-out with `JoinNode`.

### 3 — Router → specialist fleet
`classify` node sets `ctx.route` to a string; routing map fans to the matching agent; optional `DEFAULT_ROUTE` catches unknowns.

### 4 — Retry-aware scraper
Wrap the scraper with `retry_config=RetryConfig(max_attempts=5)` and `timeout=20`. Log retries via `LoggingPlugin` on the `App`.

### 5 — HITL review gate
Insert a `@node(rerun_on_resume=True, auth_config=...)` that yields `RequestInput` between producer and publisher. The workflow pauses, the event is persisted, `Runner.run_async` resumes on the next user turn.

## Gotchas

- Nodes are **Pydantic models** — if you subclass `Node`, annotate fields or they won't serialise.
- `Workflow` is not a `BaseAgent`. `Runner(agent=workflow)` fails. Use `App(root_agent=workflow)` and `Runner(app=app, session_service=...)`.
- `ctx.run_node(callable)` requires `rerun_on_resume=True` on the calling node.
- `wait_for_output=True` means a node *must* yield output/route before it's marked complete. Forget that and the workflow deadlocks.
- When a tuple contains only one element (e.g. `(START, single_node)`), you still get a single edge — not sugar for fan-out. Fan-out needs a **nested** tuple: `(START, (a, b))`.
- Setting `nodes=` explicitly on `Workflow` raises — nodes are inferred from `edges`.

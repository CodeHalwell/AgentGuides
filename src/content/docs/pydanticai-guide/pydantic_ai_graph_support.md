---
title: "Pydantic AI: Graph Support Guide"
description: "Typed, multi-step workflows with pydantic_graph: GraphBuilder steps, decisions, parallel map/join, BaseNode classes, iteration and Mermaid rendering — combined with Pydantic AI agents."
framework: pydanticai
---

# Pydantic AI: Graph Support Guide

Verified against **pydantic-ai %%version=pydantic-ai%%**, which pins the matching `pydantic-graph` release (checked %%versions-checked%%). Every example on this page was run against that release.

> **This page was rewritten for V2.** Earlier editions described a `pydantic_ai.graph` module with `@graph`/`@node` decorators, `StateMachine`, `conditional_edge` and automatic parallel scheduling. None of that existed. Graphs live in the separate **`pydantic_graph`** package, installed with `pydantic-ai`.

---

## When to use a graph

Most applications do not need one. An `Agent` already runs a loop of model calls and tool calls, and ordinary Python (`if`, `await`, `asyncio.gather`) composes several agents perfectly well. Reach for `pydantic_graph` when you want:

- an explicit, typed state machine whose edges are checked when the graph is **built**;
- fan-out / fan-in over a list with a defined join;
- a diagram of the workflow generated from the code (`graph.render()`);
- step-by-step control of execution (`graph.iter()`).

Pydantic AI's own agent loop is implemented on `pydantic_graph`; that is what `agent.iter()` exposes as `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` → `End`.

## What changed in V2

| V1 | V2 |
|---|---|
| `from pydantic_graph.beta import GraphBuilder` | `from pydantic_graph import GraphBuilder` |
| `Graph(nodes=[...])` built from `BaseNode` classes | Build with `GraphBuilder`; add `BaseNode` classes with `builder.node(NodeClass)` |
| `pydantic_graph.persistence` (`FullStatePersistence`, `FileStatePersistence`, `iter_from_persistence`) | Removed — no replacement in `pydantic_graph`. For agent-run persistence see [Durable execution](/pydanticai-guide/pydantic_ai_durable_execution/) |
| `pydantic_graph.mermaid` | Removed — use `graph.render()` |
| `StateT`/`DepsT`/`RunEndT` default to `None` | Default to `object` (type-checking only) |

---

## Core vocabulary

| Concept | API | Notes |
|---|---|---|
| Builder | `GraphBuilder(state_type=, deps_type=, input_type=, output_type=)` | Collects steps and edges; `build()` validates and returns a `Graph`. |
| Step | `@builder.step` on `async def f(ctx: StepContext[State, Deps, Input]) -> Output` | `ctx.state` (mutable, shared), `ctx.deps`, `ctx.inputs` (the previous step's output). |
| Edge | `builder.edge_from(a).to(b)` / `builder.add_edge(a, b)` | Output type of `a` must fit the input type of `b`. |
| Start / end | `builder.start_node`, `builder.end_node` | The graph's `inputs=` flows out of the start node; whatever reaches the end node is the result. |
| Decision | `builder.decision().branch(builder.match(T).to(step))` | Routes on the **type** of the value (with an optional `matches=` predicate). |
| Fan-out / fan-in | `.map()` on an edge, `builder.join(reducer, ...)` | One parallel branch per item, collected by a reducer such as `reduce_list_append`. |
| Legacy nodes | `BaseNode` subclasses + `builder.node(Cls)` | Edges come from the `run()` return annotation. |

---

## 1. Steps, state and a decision — routing a support ticket

Each step is an async function. A decision node routes the triage result to one of two desks; the agents use `TestModel` here so the example runs offline — swap in a real model string.

```python
import asyncio
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphBuilder, StepContext, reduce_list_append


@dataclass
class TicketState:
    log: list[str] = field(default_factory=list)


class Triage(BaseModel):
    category: Literal['billing', 'technical']
    summary: str


triage_agent = Agent(TestModel(), output_type=Triage, instructions='Classify the support ticket.')
reply_agent = Agent(TestModel(), instructions='Draft a short, polite reply.')

g = GraphBuilder(state_type=TicketState, input_type=str, output_type=str)


@g.step
async def triage(ctx: StepContext[TicketState, None, str]) -> Triage:
    result = await triage_agent.run(ctx.inputs)
    ctx.state.log.append(f'triaged as {result.output.category}')
    return result.output


@g.step
async def billing(ctx: StepContext[TicketState, None, Triage]) -> str:
    ctx.state.log.append('billing desk')
    return (await reply_agent.run(f'Billing issue: {ctx.inputs.summary}')).output


@g.step
async def technical(ctx: StepContext[TicketState, None, Triage]) -> str:
    ctx.state.log.append('tech desk')
    return (await reply_agent.run(f'Technical issue: {ctx.inputs.summary}')).output


g.add(
    g.edge_from(g.start_node).to(triage),
    g.edge_from(triage).to(
        g.decision()
        .branch(g.match(Triage, matches=lambda t: t.category == 'billing').to(billing))
        .branch(g.match(Triage).to(technical))
    ),
    g.edge_from(billing, technical).to(g.end_node),
)
graph = g.build()


async def main():
    state = TicketState()
    reply = await graph.run(state=state, inputs='I was charged twice this month.')
    print(reply)
    print(state.log)
    print(graph.render())

asyncio.run(main())
```

Output (with `TestModel`):

```
success (no tool calls)
['triaged as billing', 'billing desk']
stateDiagram-v2
  triage
  state decision <<choice>>
  billing
  technical

  [*] --> triage
  triage --> decision
  decision --> billing
  decision --> technical
  billing --> [*]
  technical --> [*]
```

Notes:

- `builder.match()` takes a **type** (a class, or `Literal[...]` for string values) — not a raw value. Use `matches=` for anything finer, as above. Branches are tried in order.
- `graph.render()` returns a Mermaid `stateDiagram-v2`; paste it into any Mermaid renderer.
- Steps share `ctx.state`; the value passed along an edge is the step's return value.

## 2. Parallel fan-out and join — summarising several documents

`.map()` on an edge sends each element of an iterable down its own branch; a `join` collects the branches back together with a reducer.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphBuilder, StepContext, reduce_list_append

summariser = Agent(TestModel(custom_output_text='summary'), instructions='Summarise in one sentence.')

g = GraphBuilder(input_type=list[str], output_type=list[str])


@g.step
async def split(ctx: StepContext[None, None, list[str]]) -> list[str]:
    return [doc.strip() for doc in ctx.inputs if doc.strip()]


@g.step
async def summarise(ctx: StepContext[None, None, str]) -> str:
    result = await summariser.run(ctx.inputs)
    return f'{ctx.inputs[:12]}… → {result.output}'


collect = g.join(reduce_list_append, initial_factory=list[str])

g.add(
    g.edge_from(g.start_node).to(split),
    g.edge_from(split).map().to(summarise),   # one parallel branch per document
    g.edge_from(summarise).to(collect),       # fan back in
    g.edge_from(collect).to(g.end_node),
)
graph = g.build()

print(asyncio.run(graph.run(inputs=['First report text', 'Second report text', '  '])))
```

```
['First report… → summary', 'Second repor… → summary']
```

Built-in reducers: `reduce_list_append`, `reduce_list_extend`, `reduce_dict_update`, `reduce_sum`, `reduce_null`, and `ReduceFirstValue` (first result wins; the reducer context can cancel the sibling branches). A custom reducer is any `(current, item) -> new_current` function, optionally taking a `ReducerContext` first.

## 3. `BaseNode` classes — the explicit state-machine style

If you prefer each node to be a class whose `run()` return annotation declares where it can go next, subclass `BaseNode` and add the classes with `builder.node(...)`. The start edge needs a node *instance*, passed as the graph input.

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext


@dataclass
class CountState:
    count: int = 0


@dataclass
class Increment(BaseNode[CountState, None, int]):
    async def run(self, ctx: GraphRunContext[CountState, None]) -> Check:
        ctx.state.count += 1
        return Check()


@dataclass
class Check(BaseNode[CountState, None, int]):
    async def run(self, ctx: GraphRunContext[CountState, None]) -> Increment | End[int]:
        if ctx.state.count >= 3:
            return End(ctx.state.count)
        return Increment()


g = GraphBuilder(state_type=CountState, input_type=Increment, output_type=int)
g.add(
    g.node(Increment),
    g.node(Check),
    g.edge_from(g.start_node).to(Increment),
)
graph = g.build()

state = CountState()
print(asyncio.run(graph.run(state=state, inputs=Increment())), state)

async def show_steps():
    async with graph.iter(state=CountState(), inputs=Increment()) as run:
        async for tasks in run:          # each step yields the tasks about to run (or the end marker)
            if isinstance(tasks, list):
                print([str(t.node_id) for t in tasks])
        print('output:', run.output)
asyncio.run(show_steps())
```

```
3 CountState(count=3)
['Increment']
['Check']
['Increment']
['Check']
['Increment']
['Check']
output: 3
```

`from __future__ import annotations` lets nodes reference classes defined later in the file. The return annotation is what the builder reads to create edges, so it must be accurate: `Increment | End[int]` means "go to `Increment` or finish with an `int`".

## 4. Driving a graph step by step

`graph.iter(...)` is an async context manager yielding a `GraphRun`. Iterating it yields, at each step, the list of `GraphTask`s about to run (each has a `node_id` and `inputs`), and finally an end marker; `run.output` holds the result. Use it for progress reporting, or to stop early by breaking out of the loop.

## 5. Agents inside steps

A step is plain async code, so an agent call is just `await agent.run(...)` (example 1). A few practical rules:

- **Pass `deps` through**: give the graph a `deps_type`, then call `agent.run(..., deps=ctx.deps)` inside steps.
- **Keep usage together**: create one `RunUsage()` and pass `usage=` to each `agent.run` call if you want a single total for the whole graph, and `usage_limits=` to cap it.
- **Structured hand-offs**: give each agent an `output_type` that matches the next step's input type, so the builder's type checking covers the hand-off.
- **Instrumentation**: `GraphBuilder(auto_instrument=True)` (the default) instruments the graph run when Logfire / OpenTelemetry is configured, so graph and agent spans appear in one trace.

---

## Gotchas

- **Types are the routing table.** Edges and decisions are checked from annotations at `build()` time. A step annotated `-> dict` cannot feed a step expecting `Triage`.
- **`match()` needs a type**: `builder.match('billing')` fails; use `builder.match(Literal['billing'])` or `matches=`.
- **No built-in persistence.** If the process dies mid-graph, the run is gone. Put the graph inside a durable workflow (Temporal / DBOS / Prefect) if it must survive restarts.
- **State is shared across parallel branches.** Branches created by `.map()` run concurrently and see the same `ctx.state`; prefer returning values to a join over mutating shared state.

## Reference

- `GraphBuilder`, `Graph`, `GraphRun`, `GraphTask`, `StepContext`, `BaseNode`, `End`, `GraphRunContext`, reducers — `pydantic_graph` (top-level exports)
- Edge-path builders and markers — `pydantic_graph.paths`
- Decisions — `pydantic_graph.decision`; joins and reducers — `pydantic_graph.join`
- The comprehensive guide's [graph reference section](/pydanticai-guide/pydantic_ai_comprehensive_guide/#persistence--graph-support) lists the lower-level types (`Path`, `Fork`, `Join`, ID types)

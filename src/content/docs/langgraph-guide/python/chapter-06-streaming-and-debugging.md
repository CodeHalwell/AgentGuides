---
title: "Chapter 6 — Streaming & Debugging"
description: "All 7 stream modes, typed StreamPart v2 API, GraphOutput, token-level message streaming, custom StreamWriter, UI streaming with push_ui_message/UIMessage, experimental v3 stream_events API, graph visualization, and checkpoint inspection."
framework: langgraph
language: python
sidebar:
  label: "6 · Streaming & debugging"
  order: 6
---

# Chapter 6 — Streaming & Debugging

**What you'll learn:** every streaming mode in langgraph 1.2.x, how to get typed output from the v2 API, streaming tokens from LLMs token-by-token, writing custom events from inside nodes, combining multiple stream modes, the new experimental v3 `stream_events` API with `GraphRunStream` / `SubgraphRunStream` / `LifecyclePayload` / `StreamChannel`, **UI streaming** with `push_ui_message` / `UIMessage` / `delete_ui_message` for live front-end component updates, visualizing your graph, and inspecting / modifying checkpoints for time-travel debugging.

Verified against **`langgraph==1.2.11`** (modules: `langgraph.types`, `langgraph.stream`, `langgraph.graph.ui`).

**Time:** ~30 minutes.

> Prereqs: [Chapter 2 — Your first agent](/langgraph-guide/python/chapter-02-simple-agents/).

---

## The 7 Stream Modes

`graph.stream()` and `graph.astream()` accept a `stream_mode` parameter (string or list of strings). There are seven modes:

| Mode | What it emits | Best for |
|---|---|---|
| `"values"` | Full state snapshot after every step | State inspection, simple UIs |
| `"updates"` | Only the delta each node wrote | Lightweight monitoring |
| `"messages"` | LLM tokens one-by-one + metadata | Token streaming to frontends |
| `"custom"` | Anything you write via `StreamWriter` | Progress bars, structured events |
| `"checkpoints"` | Same payload as `get_state()` per step | Audit trails, replay |
| `"tasks"` | Task start + result events | Dependency graph, task timing |
| `"debug"` | Combined checkpoints + tasks (legacy) | Step-by-step debugging |

---

## Mode 1: `"values"` — full state after each step

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int

def node_a(state: State) -> dict:
    return {"step_count": state["step_count"] + 1}

def node_b(state: State) -> dict:
    return {"step_count": state["step_count"] + 1}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
graph = builder.compile()

for snapshot in graph.stream(
    {"messages": [], "step_count": 0},
    stream_mode="values",
):
    # snapshot is the full state dict after that step
    print(f"step_count={snapshot['step_count']}")

# Output:
# step_count=1   (after node_a)
# step_count=2   (after node_b)
```

---

## Mode 2: `"updates"` — only what changed

```python
for event in graph.stream(
    {"messages": [], "step_count": 0},
    stream_mode="updates",
):
    # event maps node_name -> partial dict of what that node returned
    for node_name, updates in event.items():
        print(f"{node_name}: {updates}")

# Output:
# a: {'step_count': 1}
# b: {'step_count': 2}
```

`"updates"` transfers far less data than `"values"` — prefer it for high-throughput production usage.

---

## Mode 3: `"messages"` — LLM token streaming

`"messages"` emits `(chunk, metadata)` tuples as the LLM generates each token. `chunk` is an `AIMessageChunk`; `metadata` carries graph coordinates.

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

def call_model(state: ChatState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(ChatState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile()

async def stream_tokens():
    async for chunk, metadata in graph.astream(
        {"messages": [{"role": "user", "content": "Write a haiku about graphs"}]},
        stream_mode="messages",
    ):
        # metadata keys: langgraph_node, langgraph_step, langgraph_triggers, ls_model_name
        if metadata.get("langgraph_node") == "model":
            print(chunk.content, end="", flush=True)
    print()  # newline after streaming finishes

asyncio.run(stream_tokens())
```

To stream only content chunks and skip metadata noise:

```python
async for chunk, metadata in graph.astream(
    {"messages": [{"role": "user", "content": "Write a haiku about graphs"}]},
    stream_mode="messages",
):
    # AIMessageChunk has .content — skip empty tool-call chunks
    if hasattr(chunk, "content") and chunk.content:
        print(chunk.content, end="", flush=True)
```

---

## Mode 4: `"custom"` — write your own events

Declare `writer: StreamWriter` as a keyword argument in any node — LangGraph injects the writer automatically. Call `writer(data)` to push any JSON-serializable value downstream.

```python
from langgraph.types import StreamWriter

def research_node(state: State, writer: StreamWriter) -> dict:
    writer({"status": "starting", "task": "web search"})

    # Simulate sub-steps
    for i, source in enumerate(["Wikipedia", "arxiv", "GitHub"]):
        writer({"status": "fetching", "source": source, "progress": (i + 1) / 3})
        # ... real fetch here ...

    writer({"status": "done", "sources_checked": 3})
    return {"step_count": state["step_count"] + 1}

builder2 = StateGraph(State)
builder2.add_node("research", research_node)
builder2.add_edge(START, "research")
builder2.add_edge("research", END)
graph2 = builder2.compile()

for event in graph2.stream(
    {"messages": [], "step_count": 0},
    stream_mode="custom",
):
    # event is whatever you passed to writer(...)
    print(event)

# Output:
# {'status': 'starting', 'task': 'web search'}
# {'status': 'fetching', 'source': 'Wikipedia', 'progress': 0.333}
# {'status': 'fetching', 'source': 'arxiv', 'progress': 0.667}
# {'status': 'fetching', 'source': 'GitHub', 'progress': 1.0}
# {'status': 'done', 'sources_checked': 3}
```

`StreamWriter` is a no-op when you use `invoke()` or run without `stream_mode="custom"` — safe to leave in production code.

---

## Mode 5: `"checkpoints"` — checkpoint events per step

Each step emits a full `CheckpointPayload` — the same shape as `get_state()` but pushed in real time.

```python
from langgraph.checkpoint.memory import InMemorySaver

graph_cp = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "audit-run-1"}}

for event in graph_cp.stream(
    {"messages": [], "step_count": 0},
    cfg,
    stream_mode="checkpoints",
):
    # event is a CheckpointPayload TypedDict
    print(f"step={event['metadata']['step']}  next={event['next']}")
    print(f"  checkpoint_id={event['config']['configurable']['checkpoint_id']}")
```

---

## Mode 6: `"tasks"` — task lifecycle events

`"tasks"` emits two events per node: a `TaskPayload` (task start) and a `TaskResultPayload` (task end). Use it to build dependency graphs or measure per-node timing.

```python
import time

for event in graph.stream(
    {"messages": [], "step_count": 0},
    stream_mode="tasks",
):
    # Discriminate by presence of "error"/"result" vs "triggers"
    if "result" in event or "error" in event:
        # TaskResultPayload — task finished
        print(f"DONE  id={event['id']} name={event['name']} result={event.get('result')}")
    else:
        # TaskPayload — task started
        print(f"START id={event['id']} name={event['name']} triggers={event['triggers']}")
```

---

## Mode 7: `"debug"` — combined checkpoint + task events

`"debug"` is a legacy combined mode that wraps checkpoint and task events under a common envelope:

```python
for event in graph_cp.stream(
    {"messages": [], "step_count": 0},
    cfg,
    stream_mode="debug",
):
    # event["type"] is "checkpoint", "task", or "task_result"
    print(f"type={event['type']}  step={event['step']}")
```

---

## Multi-Mode Streaming

Pass a **list** to receive all modes simultaneously. Each yielded item is a `(mode, data)` tuple:

```python
for mode, data in graph_cp.stream(
    {"messages": [], "step_count": 0},
    cfg,
    stream_mode=["values", "updates", "custom"],
):
    if mode == "values":
        print(f"[values] step_count={data['step_count']}")
    elif mode == "updates":
        print(f"[updates] {data}")
    elif mode == "custom":
        print(f"[custom] {data}")
```

This is useful for frontends that need both token streaming (`"messages"`) and state snapshots (`"values"`) from a single request.

---

## Type-Safe v2 Streaming API

Opt in to the v2 typed API by passing `version="v2"` to `astream()`. Each item is a typed `StreamPart` TypedDict — discriminate on `part["type"]`:

```python
from langgraph.types import (
    ValuesStreamPart,
    UpdatesStreamPart,
    MessagesStreamPart,
    CustomStreamPart,
    CheckpointStreamPart,
    TasksStreamPart,
)

async for part in graph_cp.astream(
    {"messages": [], "step_count": 0},
    cfg,
    stream_mode=["values", "updates", "messages", "custom"],
    version="v2",
):
    match part["type"]:
        case "values":
            # part: ValuesStreamPart — full state + any pending interrupts
            state = part["data"]
            interrupts = part["interrupts"]   # tuple[Interrupt, ...]
            print(f"[values] step_count={state['step_count']}")

        case "updates":
            # part: UpdatesStreamPart — delta dict
            print(f"[updates] {part['data']}")

        case "messages":
            # part: MessagesStreamPart — (AIMessageChunk, metadata)
            chunk, meta = part["data"]
            print(chunk.content, end="", flush=True)

        case "custom":
            # part: CustomStreamPart — your StreamWriter data
            print(f"[custom] {part['data']}")
```

Each `StreamPart` also has a `ns` field (`tuple[str, ...]`) that identifies the subgraph namespace — useful in nested subgraph scenarios.

---

## `GraphOutput` — Typed v2 Invoke

`ainvoke()` / `invoke()` with `version="v2"` returns a `GraphOutput` dataclass instead of a plain dict:

```python
from langgraph.types import GraphOutput, Interrupt

result: GraphOutput = await graph_cp.ainvoke(
    {"messages": [{"role": "user", "content": "Hello"}], "step_count": 0},
    cfg,
    version="v2",
)

# .value — final state (typed as your OutputT if you annotate it)
print(result.value["step_count"])      # 2

# .interrupts — tuple of Interrupt objects (empty if none occurred)
interrupts: tuple[Interrupt, ...] = result.interrupts
if interrupts:
    for interrupt in interrupts:
        print(f"Interrupt id={interrupt.id}  value={interrupt.value}")
```

`GraphOutput` also supports legacy dict-style access for backwards compatibility, but the property accessors are preferred.

---

## Graph Visualization

```python
from IPython.display import Image, display

# Mermaid diagram source (paste into mermaid.live)
print(graph.get_graph().draw_mermaid())

# Render as PNG in Jupyter / Colab
display(Image(graph.get_graph().draw_mermaid_png()))

# ASCII art for terminal debugging
print(graph.get_graph().draw_ascii())
```

Example ASCII output for the three-node graph above:

```
    ┌─────────────────────┐
    │        START        │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │          a          │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │          b          │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │         END         │
    └─────────────────────┘
```

For subgraphs, use `get_graph(xray=True)` to expand all nested nodes:

```python
print(graph.get_graph(xray=True).draw_mermaid())
```

---

## State Inspection and Time-Travel

### Get current state

```python
cfg = {"configurable": {"thread_id": "debug-session"}}
list(graph_cp.stream({"messages": [], "step_count": 0}, cfg))

state = graph_cp.get_state(cfg)
print(f"next={state.next}")           # ('',) when done
print(f"values={state.values}")
print(f"interrupts={state.interrupts}")
```

### Walk the full history

```python
history = list(graph_cp.get_state_history(cfg))
for i, snap in enumerate(history):
    cp_id = snap.config["configurable"]["checkpoint_id"]
    print(f"Step {i}: checkpoint={cp_id}  next={snap.next}")
```

### Time-travel: resume from a past checkpoint

```python
# Take the second-most-recent snapshot and re-run from there
old_snap = history[1]
result = graph_cp.invoke(None, config=old_snap.config)  # None = resume from checkpoint as-is
```

### Inject state between runs (`update_state`)

```python
graph_cp.update_state(
    cfg,
    {"step_count": 99},   # override the value
    as_node="a",          # attribute the update to node "a"
)

# Continue from the patched state (pass None so the patched value isn't overwritten)
result = graph_cp.invoke(None, cfg)
print(result["step_count"])   # 100 (99 + 1 from node_b)
```

---

## Batch Invocation and Error Collection

```python
inputs = [{"messages": [], "step_count": i} for i in range(5)]
configs = [{"configurable": {"thread_id": f"batch-{i}"}} for i in range(5)]

results, errors = [], []
for inp, cfg_i in zip(inputs, configs):
    try:
        results.append(graph_cp.invoke(inp, config=cfg_i))
    except Exception as exc:
        errors.append((cfg_i["configurable"]["thread_id"], str(exc)))

print(f"OK={len(results)}  FAIL={len(errors)}")
for tid, err in errors:
    print(f"  {tid}: {err}")
```

For true concurrent batch execution use `graph.abatch()`:

```python
import asyncio

async def run_batch():
    results = await graph_cp.abatch(inputs, configs)
    return results

asyncio.run(run_batch())
```

---

## Pydantic State and Auto-Coercion

Since v1.1.x, `invoke()` automatically coerces a plain dict input into your Pydantic or dataclass state type:

```python
from pydantic import BaseModel

class TypedState(BaseModel):
    query: str
    result: str = ""

builder3 = StateGraph(TypedState)
# ... nodes ...
graph3 = builder3.compile()

# Pass a dict — auto-coerced to TypedState on entry
result = graph3.invoke({"query": "What is LangGraph?"})
# v1 invoke returns a dict; use version="v2" for a typed GraphOutput
print(type(result))   # dict
```

---

## Experimental v3 Streaming — `stream_events(version="v3")`

LangGraph 1.2.11 introduces an experimental v3 streaming protocol built on typed `StreamChannel` projections. Instead of iterating a flat event stream, you drive the graph by consuming named projections on a `GraphRunStream` context object. Each projection is a **single-consumer drainable queue** — there is no background thread; your `for` loop is the pump.

> **Warning:** `version="v3"` is experimental and may change in future releases. Gate it behind a feature flag in production.

### `GraphRunStream` — the v3 run handle

The v3 protocol has separate sync and async entry points:

- `graph.stream_events(..., version="v3")` returns a `GraphRunStream` you enter with `with ... as run:`.
- `graph.astream_events(..., version="v3")` is a coroutine — `await` it, then enter with `async with ... as run:` to get an `AsyncGraphRunStream`.

Both handles expose the same four native projections:

| Attribute | Type | What it emits |
|---|---|---|
| `run.values` | `StreamChannel[dict]` | Full state snapshot after every step |
| `run.messages` | `StreamChannel[ChatModelStream]` | LLM token stream |
| `run.lifecycle` | `StreamChannel[LifecyclePayload]` | Subgraph start/end/error lifecycle events |
| `run.subgraphs` | `StreamChannel[SubgraphRunStream]` | Nested subgraph run handles |

Opt-in projections (`updates`, `custom`, `checkpoints`, `tasks`) are also exposed as **direct attributes** (`run.updates`, `run.custom`, etc.) when their built-in transformers are registered via `transformers=`. Only user-authored non-native transformers use `run.extensions["channel_name"]`.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

def call_model(state: ChatState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}

graph = (
    StateGraph(ChatState)
    .add_node("model", call_model)
    .add_edge(START, "model")
    .add_edge("model", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "v3-demo"}}

# stream_events(version="v3") returns a GraphRunStream — not an iterator.
# Use it as a context manager so cleanup runs on early exit.
with graph.stream_events(
    {"messages": [HumanMessage(content="Write a haiku")]},
    cfg,
    version="v3",
) as run:
    # run.messages yields ChatModelStream objects — one per LLM response.
    # Use the .text projection to get string deltas token-by-token.
    for msg_chunk in run.messages:
        for delta in msg_chunk.text:
            print(delta, end="", flush=True)
print()

# OR: consume final state snapshots via run.values
with graph.stream_events(
    {"messages": [HumanMessage(content="Hello")]},
    cfg,
    version="v3",
) as run2:
    for state_snapshot in run2.values:
        print(f"step_count messages: {len(state_snapshot['messages'])}")

# OR: get final output without iteration
with graph.stream_events(
    {"messages": [HumanMessage(content="Hi")]},
    cfg,
    version="v3",
) as run3:
    final_state = run3.output   # drives graph to completion, returns final state
```

Async callers use `astream_events()` instead, awaiting it before entering:

```python
import asyncio

async def main():
    # astream_events() is a coroutine — await it to get the AsyncGraphRunStream
    run = await graph.astream_events(
        {"messages": [HumanMessage(content="Async hi")]},
        cfg,
        version="v3",
    )
    async with run:
        async for msg_chunk in run.messages:
            async for delta in msg_chunk.text:
                print(delta, end="", flush=True)

asyncio.run(main())
```

> **Multi-projection consumption.** `StreamChannel` uses **lazy-subscribe** — items are only buffered on a channel after something has subscribed to it. If you drain `run.values` fully before touching `run.messages`, the graph pumps to completion while `run.messages` has no subscriber, and every token pushed to it is silently discarded.
>
> Use `run.interleave("values", "messages", ...)` on `GraphRunStream` for the sync case; it yields `(name, item)` tuples in arrival order across the named projections and locks each channel for the duration. For async, subscribe every projection **before** starting the pump (e.g. `asyncio.gather` over per-projection consumer tasks) — `AsyncGraphRunStream` does not expose an `ainterleave` helper in 1.2.11.

```python
# Sync multi-projection consumption — arrival-ordered, no dropped events.
with graph.stream_events(
    {"messages": [HumanMessage(content="Interleave demo")]},
    cfg,
    version="v3",
) as run:
    for name, item in run.interleave("values", "messages"):
        if name == "values":
            print(f"[state] messages={len(item['messages'])}")
        else:  # "messages"
            for delta in item.text:
                print(delta, end="", flush=True)
```

### `LifecyclePayload` — subgraph lifecycle events

`run.lifecycle` emits a `LifecyclePayload` when a subgraph starts, completes, or errors. This replaces the need to watch `stream_mode="debug"` just to know when nested graphs begin/end.

```python
from langgraph.stream.transformers import LifecyclePayload  # TypedDict

# LifecyclePayload fields:
#   event: "started" | "completed" | "failed" | "interrupted" | "drained"
#          (SubgraphStatus Literal — "drained" = cooperative shutdown via
#           RunControl.request_drain() (e.g., SIGTERM). The checkpoint is
#           saved and the run can be resumed later with the same thread_id;
#           it is NOT a clean end-of-run.)
#   namespace: list[str]       — path to the subgraph (e.g. ['inner_team:abc123'])
#   graph_name: str | None     — compiled graph name if known (defaults to
#                                "LangGraph" unless `name=...` is passed to
#                                `StateGraph.compile()`)
#   trigger_call_id: str | None — parent-graph task_id in the child's namespace
#                                segment (`node:task_id`, split by
#                                `_parse_ns_segment`). It is the same id that
#                                the parent's `TaskResultPayload` carries, so
#                                consumers can pair a child `started` with the
#                                parent-task result that closes it. Populated
#                                whenever the child segment carries a
#                                `:task_id` suffix (e.g. functional-API
#                                `@task`/`call()` and subagent-style tool
#                                dispatch); the LifecycleTransformer skips the
#                                `started` payload when it is missing.
#   cause: LifecycleCause | None — how the subgraph was triggered. The
#                                LifecycleTransformer only attaches `cause`
#                                for a subagent boundary — a nested run whose
#                                parent task carries `lc_agent_name` metadata
#                                (set by `create_agent`) and a harvested
#                                tool_call_id — and emits it as:
#     {"type": "toolCall", "tool_call_id": "..."}
#                                Other `LifecycleCause` variants
#                                (`{"type": "send", ...}`,
#                                `{"type": "edge", ...}`) exist in the type
#                                but are attached by different transformers,
#                                not by `LifecycleTransformer` itself.
#   error: str | None          — error message if event == "failed"

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class OuterState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class InnerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def inner_node(state: InnerState) -> dict:
    return {"messages": [("assistant", "inner response")]}

inner_graph = (
    StateGraph(InnerState)
    .add_node("inner", inner_node)
    .add_edge(START, "inner")
    .add_edge("inner", END)
    # Pass name= so `LifecyclePayload.graph_name` matches the value shown
    # in the expected output below; without it the default is "LangGraph".
    .compile(name="inner", checkpointer=True)   # inherit parent checkpointer
)

def outer_node(state: OuterState) -> dict:
    return {"messages": [("assistant", "outer response")]}

outer_graph = (
    StateGraph(OuterState)
    .add_node("outer", outer_node)
    .add_node("inner_team", inner_graph)   # subgraph as node
    .add_edge(START, "outer")
    .add_edge("outer", "inner_team")
    .add_edge("inner_team", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "lifecycle-demo"}}

with outer_graph.stream_events(
    {"messages": [HumanMessage(content="Go")]},
    cfg,
    version="v3",
) as run:
    # Lifecycle events tell you exactly when each subgraph starts and ends
    for event in run.lifecycle:
        print(f"[lifecycle] event={event.get('event')} ns={event.get('namespace')} graph={event.get('graph_name')}")
# [lifecycle] event=started    ns=['inner_team:abc123'] graph=inner
# [lifecycle] event=completed  ns=['inner_team:abc123'] graph=inner
```

### `SubgraphRunStream` — per-subgraph run handles

When `run.subgraphs` is consumed, each item is a `SubgraphRunStream` — a `GraphRunStream` subclass that also carries `.path` (the subgraph's namespace tuple), `.graph_name`, and `.status` / `.error`. This lets you consume a nested subgraph's own `values`, `messages`, and `lifecycle` projections independently.

The snippet below reuses `outer_graph` and `cfg` from the `LifecyclePayload` example above:

```python
with outer_graph.stream_events(
    {"messages": [HumanMessage(content="Go")]},
    cfg,
    version="v3",
) as run:
    for subgraph_run in run.subgraphs:
        print(f"Subgraph discovered: path={subgraph_run.path} name={subgraph_run.graph_name}")
        # Each SubgraphRunStream has the same projections as GraphRunStream
        for sub_snapshot in subgraph_run.values:
            print(f"  inner state messages: {len(sub_snapshot['messages'])}")
```

### `StreamChannel` — custom named projections

`StreamChannel` is a typed single-consumer drainable queue. You can expose a custom projection by passing `transformers=` to `stream_events()`. Custom projections are reached via `run.extensions["my_channel"]`.

`StreamChannel` objects are created by transformers and wired into the `StreamMux` before a run starts. They are single-consumer, pull-driven queues — you can't instantiate one standalone and iterate it; the mux must bind it first.

There are two kinds of channels on `GraphRunStream`:

- **Native projections** — always present as direct attributes: `run.values`, `run.messages`, `run.lifecycle`, `run.subgraphs`. Opt-in native transformers (when registered via `transformers=`) also appear as direct attributes: `run.updates`, `run.custom`, `run.checkpoints`, `run.tasks`.
- **Non-native custom channels** — registered by user-authored transformers and accessed via `run.extensions["channel_name"]`.

```python
from langgraph.stream.stream_channel import StreamChannel

# StreamChannel[T] is the type of every projection on GraphRunStream.
# Always-present native projections (direct attributes):
#   run.values:    StreamChannel[dict[str, Any]]
#   run.messages:  StreamChannel[ChatModelStream]
#   run.lifecycle: StreamChannel[LifecyclePayload]
#   run.subgraphs: StreamChannel[SubgraphRunStream]
#
# Opt-in native projections (direct attributes when the transformer is registered):
#   run.updates / run.custom / run.checkpoints / run.tasks
#
# Non-native custom transformer channels (accessed via extensions dict):
#   channel: StreamChannel[Any] = run.extensions["my_transformer_name"]

# Iterating any channel drives the graph pump forward one event at a time.
# `graph` and `cfg` here are the ones set up in the earlier v3 examples:
with graph.stream_events({"messages": [HumanMessage(content="Hi")]}, cfg, version="v3") as run:
    for snapshot in run.values:   # StreamChannel[dict] — one snapshot per step
        print(snapshot)
```

For most use cases the built-in `stream_mode="custom"` (v1/v2 API) is simpler than registering a custom transformer. Reach for `StreamChannel` types when you need fine-grained control over event buffering or fan-out within the v3 protocol.

### v3 vs v1/v2 — when to use which

| Concern | v1/v2 (`stream(stream_mode=...)`) | v3 (`stream_events(version="v3")`) |
|---|---|---|
| Stability | Stable / production-ready | Experimental — may change |
| Token streaming | `stream_mode="messages"` | `run.messages` projection |
| State snapshots | `stream_mode="values"` | `run.values` projection |
| Subgraph visibility | `subgraphs=True` flag | `run.subgraphs` with typed handles |
| Lifecycle events | `stream_mode="debug"` | `run.lifecycle` (typed `LifecyclePayload`) |
| Multiple modes at once | `stream_mode=[...]` → `(mode, data)` tuples | `run.interleave("a", "b", ...)` → arrival-ordered `(name, item)` tuples (sync); concurrent consumer tasks (async) |
| Type safety | `version="v2"` `StreamPart` TypedDict | Generically typed `StreamChannel[T]` |

Use **v1/v2** for production today. Experiment with **v3** when you need per-subgraph handles or typed lifecycle events, and be ready to adapt when the API stabilises.

---

## UI Streaming — `push_ui_message`, `UIMessage`, and `delete_ui_message`

LangGraph 1.2.11 ships a first-class UI streaming layer under `langgraph.graph.ui`. Nodes call `push_ui_message()` from inside their body and the event is forwarded to every consumer watching `stream_mode="custom"`. This lets a React / Vue / Svelte front-end render named components from live graph state without polling.

Verified against **`langgraph==1.2.11`** (module: `langgraph.graph.ui`).

### Primitives at a glance

| Symbol | Module | Purpose |
|---|---|---|
| `UIMessage` | `langgraph.graph.ui` | TypedDict describing a UI component to render |
| `RemoveUIMessage` | `langgraph.graph.ui` | TypedDict describing a UI component to remove |
| `push_ui_message(name, props)` | `langgraph.graph.ui` | Create and stream a `UIMessage` from inside a node |
| `delete_ui_message(id)` | `langgraph.graph.ui` | Create and stream a `RemoveUIMessage` from inside a node |
| `ui_message_reducer` | `langgraph.graph.ui` | Reducer for a `ui` state key — handles add/merge/remove |

### `UIMessage` TypedDict (source-verified)

```python
# langgraph.graph.ui (source-verified, langgraph 1.2.11)
class UIMessage(TypedDict):
    type: Literal["ui"]         # always "ui"
    id: str                     # unique identifier (UUID if not provided)
    name: str                   # component name the client maps to a React element
    props: dict[str, Any]       # arbitrary props passed to the component
    metadata: dict[str, Any]    # run_id, tags, message_id, merge flag, …
```

`RemoveUIMessage` is simpler:

```python
class RemoveUIMessage(TypedDict):
    type: Literal["remove-ui"]  # always "remove-ui"
    id: str                     # id of the UIMessage to remove
```

### Minimal working example

Wire a `ui` key into your state using `ui_message_reducer`, then call `push_ui_message()` from any node:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage, RemoveUIMessage, push_ui_message, ui_message_reducer

# AnyUIMessage is the union type used by the reducer
AnyUIMessage = UIMessage | RemoveUIMessage


class State(TypedDict):
    query: str
    result: str
    # Annotate with the reducer — it handles add / merge / remove semantics
    ui: Annotated[list[AnyUIMessage], ui_message_reducer]


def search_node(state: State) -> dict:
    # Push a loading spinner to the UI before doing work
    spinner = push_ui_message(
        name="Spinner",
        props={"label": "Searching…"},
        id="spinner-1",      # stable id so we can remove it later
    )

    # … do the actual work …
    result = f"results for: {state['query']}"

    # Replace spinner with a result card (merge=False replaces props)
    push_ui_message(
        name="ResultCard",
        props={"content": result, "query": state["query"]},
    )

    return {"result": result}


graph = (
    StateGraph(State)
    .add_node("search", search_node)
    .add_edge(START, "search")
    .add_edge("search", END)
    .compile()
)

# Consumers receive UIMessage objects in the "custom" stream
for mode, data in graph.stream(
    {"query": "langgraph streaming", "ui": []},
    stream_mode=["values", "custom"],
):
    if mode == "custom":
        print(f"UI event: {data}")      # {'type': 'ui', 'name': 'Spinner', ...}
    elif mode == "values":
        print(f"State ui list: {data.get('ui', [])}")
```

### Removing a UI component with `delete_ui_message`

Call `delete_ui_message(id)` to emit a `RemoveUIMessage`. The `ui_message_reducer` splices it out of the `ui` list so the state stays consistent:

```python
from langgraph.graph.ui import push_ui_message, delete_ui_message


def multi_step_node(state: State) -> dict:
    # Show a progress bar
    push_ui_message("ProgressBar", {"value": 0, "max": 3}, id="progress")

    # … step 1 …
    push_ui_message("ProgressBar", {"value": 1, "max": 3}, id="progress", merge=True)

    # … step 2 …
    push_ui_message("ProgressBar", {"value": 2, "max": 3}, id="progress", merge=True)

    # Done — remove the progress bar entirely
    delete_ui_message("progress")

    return {}
```

> **`merge=True`** tells `ui_message_reducer` to *merge* the new `props` dict into the existing `UIMessage` with the same `id` rather than replacing the whole entry. Use it for incremental updates (e.g. progress values, streaming text chunks) and `merge=False` (the default) for full replacements.

### Async node example

`push_ui_message` and `delete_ui_message` are synchronous helpers that call the stream writer internally — they work inside both sync and async nodes without any change:

```python
import asyncio
from langgraph.graph.ui import push_ui_message, delete_ui_message


async def async_search_node(state: State) -> dict:
    push_ui_message("Spinner", {"label": "Loading…"}, id="async-spinner")
    await asyncio.sleep(0.1)   # simulate async I/O
    result = f"async results for {state['query']}"
    delete_ui_message("async-spinner")
    push_ui_message("ResultCard", {"content": result})
    return {"result": result}
```

### Connecting to a React front-end (sketch)

`UIMessage.name` is the key the client uses to look up the component to render:

```typescript
// Pseudo-code: client receives UIMessage events from the stream
const COMPONENTS = {
  Spinner: SpinnerComponent,
  ResultCard: ResultCardComponent,
  ProgressBar: ProgressBarComponent,
};

for await (const event of graphStream) {
  if (event.type === "ui") {
    const Component = COMPONENTS[event.name];
    if (Component) render(<Component key={event.id} {...event.props} />);
  } else if (event.type === "remove-ui") {
    removeRenderedComponent(event.id);
  }
}
```

### Quick Reference — UI streaming

| Task | Code |
|---|---|
| Push a UI update | `push_ui_message("MyComponent", {"key": "val"})` |
| Push with stable ID | `push_ui_message("MyComponent", {...}, id="my-id")` |
| Merge props into existing | `push_ui_message("MyComponent", {...}, id="my-id", merge=True)` |
| Remove a component | `delete_ui_message("my-id")` |
| Declare the state key | `ui: Annotated[list[AnyUIMessage], ui_message_reducer]` |
| Receive in stream | `stream_mode="custom"` (events appear alongside other modes) |

---

## Quick Reference

| Task | Code |
|---|---|
| Full state after each step | `stream_mode="values"` |
| Only changed keys | `stream_mode="updates"` |
| LLM tokens | `stream_mode="messages"` |
| Custom progress events | `stream_mode="custom"` + `StreamWriter` param |
| Checkpoint per step | `stream_mode="checkpoints"` |
| Task timing | `stream_mode="tasks"` |
| Multiple at once | `stream_mode=["values", "messages"]` |
| Typed stream parts | `astream(..., version="v2")` |
| Typed final output | `ainvoke(..., version="v2")` → `GraphOutput` |
| v3 run handle (sync) | `graph.stream_events(input, cfg, version="v3")` → `GraphRunStream` |
| v3 run handle (async) | `await graph.astream_events(input, cfg, version="v3")` → `AsyncGraphRunStream` |
| v3 multi-projection (sync) | `for name, item in run.interleave("values", "messages"): ...` |
| v3 token stream | `for chunk in run.messages: ...` |
| v3 state snapshots | `for snap in run.values: ...` |
| v3 lifecycle events | `for ev in run.lifecycle: ...` → `LifecyclePayload` |
| v3 subgraph handles | `for sub in run.subgraphs: ...` → `SubgraphRunStream` |
| v3 final output (drives graph) | `run.output` |
| Visualise graph | `graph.get_graph().draw_mermaid()` |
| Inspect state | `graph.get_state(cfg)` |
| History / time-travel | `graph.get_state_history(cfg)` |
| Patch state | `graph.update_state(cfg, {...})` |
| Push UI component | `push_ui_message("Name", props)` → `UIMessage` |
| Remove UI component | `delete_ui_message(id)` → `RemoveUIMessage` |
| Merge UI props | `push_ui_message("Name", delta, id=..., merge=True)` |
| Declare UI state key | `ui: Annotated[list[AnyUIMessage], ui_message_reducer]` |

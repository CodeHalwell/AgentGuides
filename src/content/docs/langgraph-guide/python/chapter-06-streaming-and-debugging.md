---
title: "Chapter 6 — Streaming & Debugging"
description: "All 7 stream modes, typed StreamPart v2 API, GraphOutput, token-level message streaming, custom StreamWriter, multi-mode streaming, graph visualization, and checkpoint inspection."
framework: langgraph
language: python
sidebar:
  label: "6 · Streaming & debugging"
  order: 6
---

# Chapter 6 — Streaming & Debugging

**What you'll learn:** every streaming mode in langgraph 1.2.x, how to get typed output from the v2 API, streaming tokens from LLMs token-by-token, writing custom events from inside nodes, combining multiple stream modes, visualizing your graph, and inspecting / modifying checkpoints for time-travel debugging.

Verified against **`langgraph==1.2.6`** (modules: `langgraph.types`, `langgraph.stream`).

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
| Visualise graph | `graph.get_graph().draw_mermaid()` |
| Inspect state | `graph.get_state(cfg)` |
| History / time-travel | `graph.get_state_history(cfg)` |
| Patch state | `graph.update_state(cfg, {...})` |

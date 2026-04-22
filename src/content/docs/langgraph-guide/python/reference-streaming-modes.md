---
title: "Streaming modes — API reference"
description: "All seven stream_mode values (values, updates, messages, custom, checkpoints, tasks, debug), the v1 vs v2 API, GraphOutput, StreamPart — what each one yields and when to pick it."
framework: langgraph
language: python
sidebar:
  label: "Ref · Streaming modes"
  order: 35
---

# Streaming modes — API reference

Verified against **`langgraph==1.1.9`** (modules: `langgraph.types`, `langgraph.pregel.main`).

Every compiled graph (both `StateGraph` and `@entrypoint` workflows) exposes:

```python
graph.stream(input, config=None, *, stream_mode=..., version="v1" | "v2", ...)
graph.astream(input, config=None, *, stream_mode=..., version="v1" | "v2", ...)
graph.invoke(input, config=None, *, version="v1" | "v2", ...)
graph.ainvoke(input, config=None, *, version="v1" | "v2", ...)
```

`stream_mode` controls **what** is yielded. `version` controls **how it is typed**. The v2 API yields structured `StreamPart` dicts from `langgraph.types`; v1 yields raw values.

## Minimal runnable example

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class S(TypedDict):
    x: int


def step(state: S) -> dict:
    return {"x": state["x"] + 1}


graph = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "t"}}

for chunk in graph.stream({"x": 0}, cfg, stream_mode="updates"):
    print(chunk)
# {'step': {'x': 1}}

for part in graph.stream({"x": 0}, cfg, stream_mode="updates", version="v2"):
    # part is a typed StreamPart dict
    print(part["type"], part["ns"], part["data"])
# updates () {'step': {'x': 1}}
```

## The seven stream modes

| Mode | Yields | Typical use |
|---|---|---|
| `"values"` | Full state after each step. | Show the current state as the graph runs. |
| `"updates"` | `{node_name: state_update}` per node per step. Interrupts come through as `{"__interrupt__": (Interrupt,...)}`. | Activity feed; detecting `__interrupt__`. |
| `"messages"` | `(message, metadata)` tuples for every LLM token emitted inside any node. | Token-by-token chat UIs. |
| `"custom"` | Whatever you passed to `StreamWriter` / `runtime.stream_writer`. | Domain-specific progress events. |
| `"checkpoints"` | Checkpoint payloads (`config`, `values`, `metadata`, `next`, `parent_config`, `tasks`). | Audit logs, progress DBs. |
| `"tasks"` | Task start / result events (`id`, `name`, `input`, `triggers` / `id`, `name`, `error`, `interrupts`, `result`). | Observability dashboards. |
| `"debug"` | All of the above as a single wrapped `DebugPayload`. | Replacing prints while developing. |

You can also pass a **list** of modes. The iterator then yields `(mode, data)` tuples:

```python
for mode, data in graph.stream(inp, cfg, stream_mode=["updates", "messages"]):
    if mode == "updates":
        ...
    elif mode == "messages":
        token, meta = data
```

## `stream(..., subgraphs=True)`

Set `subgraphs=True` to see events from inside child graphs. The leading element of each yielded tuple becomes the namespace path:

```python
for ns, data in graph.stream(inp, cfg, stream_mode="updates", subgraphs=True):
    # ns = ('parent_node:<task_id>', 'child_node:<task_id>')
    ...
for ns, mode, data in graph.stream(inp, cfg, stream_mode=["updates", "messages"], subgraphs=True):
    ...
```

With `version="v2"`, `ns` is already a tuple on every `StreamPart` regardless of `subgraphs=`.

## v1 vs v2 API

```python
graph.stream(input, cfg, stream_mode="updates")                  # v1 (default)
graph.stream(input, cfg, stream_mode="updates", version="v2")    # v2
```

- **v1**: yields raw values. Simple to consume, but you often have to sniff types (`isinstance(chunk, tuple)`, `"__interrupt__" in chunk`, etc.).
- **v2**: yields `StreamPart` TypedDicts with `type`, `ns`, `data` fields. Interrupts are pulled out into `ValuesStreamPart.interrupts` for `stream_mode="values"`.

The `StreamPart` union (from `langgraph.types`):

```python
StreamPart = (
    ValuesStreamPart
    | UpdatesStreamPart
    | MessagesStreamPart
    | CustomStreamPart
    | CheckpointStreamPart
    | TasksStreamPart
    | DebugStreamPart
)
```

Narrow by `part["type"]`:

```python
for part in graph.astream(inp, cfg, version="v2"):
    match part["type"]:
        case "values":
            # part["data"]: final-ish full state; part["interrupts"]: tuple[Interrupt, ...]
            ...
        case "messages":
            msg, meta = part["data"]           # (BaseMessage, metadata dict)
        case "custom":
            payload = part["data"]             # whatever StreamWriter wrote
        case "checkpoints":
            cp = part["data"]                  # CheckpointPayload
        case "tasks":
            # task start has input/triggers; task result has error/result/interrupts
            ...
        case "updates" | "debug":
            ...
```

## `invoke(..., version="v2")` → `GraphOutput`

With v2, `invoke` returns a typed container instead of a dict:

```python
from langgraph.types import GraphOutput

result: GraphOutput = graph.invoke({"x": 0}, cfg, version="v2")
print(result.value)        # final state — dict / Pydantic / dataclass per state_schema
print(result.interrupts)   # tuple[Interrupt, ...]
```

For back-compat, `result["key"]` still works on a `GraphOutput` but emits `DeprecationWarning`; prefer `result.value["key"]`.

## Stream mode details

### `"values"`

Emits the **entire** state after each step. For the functional API, emits exactly once at the end.

```python
for s in graph.stream(inp, cfg, stream_mode="values"):
    # v1: s is the state dict (or your state_schema instance)
    print(s)
```

v2 shape (`ValuesStreamPart`):

```python
{"type": "values", "ns": (), "data": <state>, "interrupts": (Interrupt(...),)}
```

### `"updates"`

Emits one event per node per step, keyed by node name:

```python
{"planner": {"messages": [...], "next": "writer"}}
```

Interrupts show up as a sibling key `"__interrupt__"` whose value is a tuple of `Interrupt` dataclasses. Parallel nodes in the same super-step produce separate events.

### `"messages"`

Yields tuples of `(message, metadata)` for every LLM invocation inside any node:

- `message` — usually an `AIMessageChunk`; see `langchain_core.messages`.
- `metadata` — dict with `langgraph_step`, `langgraph_node`, `langgraph_triggers`, `langgraph_path`, `ls_model_name`, `ls_provider`, etc.

Wire an LLM normally and let LangGraph's callbacks do the work:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def draft(state: dict) -> dict:
    return {"text": llm.invoke(state["prompt"]).content}

for msg, meta in graph.stream({"prompt": "hi"}, cfg, stream_mode="messages"):
    if meta["langgraph_node"] == "draft":
        print(msg.content, end="", flush=True)
```

### `"custom"`

Write arbitrary values from inside a node using `StreamWriter` or `runtime.stream_writer`:

```python
from langgraph.runtime import Runtime

def work(state: dict, runtime: Runtime) -> dict:
    runtime.stream_writer({"phase": "start", "pct": 0})
    ...
    runtime.stream_writer({"phase": "halfway", "pct": 50})
    ...
    runtime.stream_writer({"phase": "done", "pct": 100})
    return {"done": True}

for ev in graph.stream(inp, cfg, stream_mode="custom"):
    print(ev)
```

Outside `stream_mode="custom"`, calls to `stream_writer` are no-ops — it's safe to leave them in.

### `"checkpoints"`

Emits a `CheckpointPayload` each time a checkpoint is created:

```python
{
    "config": {...},           # pointer to this checkpoint
    "metadata": {"source": "loop", "step": 1, "parents": {}, "run_id": "..."},
    "values": {<state>},
    "next": ["writer"],
    "parent_config": {...},
    "tasks": [{"id": "...", "name": "planner", "result": {...}, "state": None, "interrupts": []}],
}
```

Requires a checkpointer; otherwise the mode yields nothing.

### `"tasks"`

Two payload shapes interleaved on one stream:

```python
# task start
{"id": "...", "name": "planner", "input": {...}, "triggers": ["branch:to:planner"]}
# task result
{"id": "...", "name": "planner", "error": None, "interrupts": [], "result": {"x": 1}}
```

Pair these with `"messages"` to annotate token events with the owning task.

### `"debug"`

Emits `DebugPayload` wrappers: a discriminated union of `{"type": "checkpoint" | "task" | "task_result", "step", "timestamp", "payload"}`. Useful to replace `print()` during development and turn off with a flag.

## Durability interacts with streaming

On `invoke` / `stream`, set `durability="sync" | "async" | "exit"` to trade checkpoint-write timing against speed:

```python
graph.stream(inp, cfg, stream_mode="updates", durability="sync")
```

With `durability="exit"` you will not see `"checkpoints"` events per step — only at the very end.

## `ainvoke` / `astream`

Same signatures, awaitable. v2 typing works the same:

```python
async for part in graph.astream({"x": 0}, cfg, version="v2"):
    if part["type"] == "messages":
        msg, meta = part["data"]
```

## Patterns

### 1. Token streaming to stdout

```python
async for msg, meta in graph.astream(inp, cfg, stream_mode="messages"):
    if msg.content and meta["langgraph_node"] == "writer":
        print(msg.content, end="", flush=True)
```

### 2. Server-Sent Events with multiple modes

```python
import json

async for mode, data in graph.astream(inp, cfg, stream_mode=["updates", "messages"]):
    if mode == "updates" and "__interrupt__" in data:
        yield f"event: interrupt\ndata: {json.dumps([i.value for i in data['__interrupt__']])}\n\n"
    elif mode == "messages":
        tok, _ = data
        if tok.content:
            yield f"event: token\ndata: {tok.content}\n\n"
```

### 3. Progress bar using `"custom"`

```python
def download(state, runtime):
    for i, url in enumerate(state["urls"], start=1):
        fetch(url)
        runtime.stream_writer({"pct": int(100 * i / len(state["urls"]))})
    return {"done": True}
```

### 4. v2 `invoke` with typed return

```python
from langgraph.types import GraphOutput
out: GraphOutput = await graph.ainvoke(inp, cfg, version="v2")
if out.interrupts:
    return {"status": "awaiting_input", "prompts": [i.value for i in out.interrupts]}
return {"status": "done", "state": out.value}
```

### 5. Checkpoint-driven audit log

```python
async for part in graph.astream(inp, cfg, stream_mode="checkpoints", version="v2"):
    cp = part["data"]
    audit.write({
        "run_id": cp["metadata"]["run_id"],
        "step": cp["metadata"]["step"],
        "next": cp["next"],
        "updated": cp["metadata"].get("writes"),
    })
```

## Gotchas

- **Default stream mode is `"updates"`.** Passing `stream_mode=None` inherits from the graph's own default (which is `"updates"` for root graphs and `"values"` when invoked as a subgraph step).
- **`"checkpoints"` needs a checkpointer.** Without one you get no events, not an error.
- **`stream_mode="messages"` requires callbacks.** If you construct LLMs outside LangGraph and hand back messages manually, you won't see tokens. Use the LangChain `ChatModel` interface inside a node so callbacks fire.
- **v2 is opt-in per call.** There is no global switch. Always pass `version="v2"` if you want typed output; otherwise you get the legacy shape.
- **`print_mode=` is separate from `stream_mode=`.** `print_mode` prints to stdout for debugging and does not change what `stream()` yields.
- **`subgraphs=True` changes the tuple shape.** With a single mode you get `(ns, data)`; with a list of modes you get `(ns, mode, data)`. With `version="v2"` this collapses because `ns` is always part of the `StreamPart`.
- **The `"__interrupt__"` key only appears in `"updates"` mode.** For `"values"` v2, interrupts live in `part["interrupts"]`.
- **`stream_writer` calls are lost outside `"custom"` mode.** That's intentional — but if you expected to see them, add `"custom"` to `stream_mode`.
- **`print_mode` is additive.** Passing `print_mode="updates"` both prints updates **and** keeps whatever your `stream_mode` emits to the iterator.

## Breaking changes

| Version | Change |
|---|---|
| 1.1 | `version="v2"` on `stream`/`astream` yields typed `StreamPart` dicts; `invoke`/`ainvoke` with `version="v2"` return `GraphOutput`. `GraphOutput[key]` indexing raises `DeprecationWarning`. |
| 1.0 | `stream_mode="tasks"` split from `"debug"`; `"checkpoints"` added as its own mode. |
| 0.6 | `interrupt_before` / `interrupt_after` on `invoke`/`stream` accept `"*"` for all nodes. |
| 0.5 | `checkpoint_during=False` deprecated in favor of `durability="exit"`. |

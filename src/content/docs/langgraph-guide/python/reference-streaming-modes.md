---
title: "Streaming modes — API reference"
description: "All seven stream_mode values (values, updates, messages, custom, checkpoints, tasks, debug), the v1 vs v2 API, GraphOutput, StreamPart typed union, StreamWriter injection — what each one yields and when to pick it."
framework: langgraph
language: python
sidebar:
  label: "Ref · Streaming modes"
  order: 35
---

# Streaming modes — API reference

Verified against **`langgraph==1.2.1`** (modules: `langgraph.types`, `langgraph.pregel.main`, `langgraph.config`).

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

`StreamMode` is a `Literal` union exported from `langgraph.types`:

```python
StreamMode = Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]
```

| Mode | Yields | Typical use |
|---|---|---|
| `"values"` | Full state after each step. | Show the current state as the graph runs. |
| `"updates"` | `{node_name: state_update}` per node per step. Interrupts come through as `{"__interrupt__": (Interrupt,...)}`. | Activity feed; detecting `__interrupt__`. |
| `"messages"` | `(message, metadata)` tuples for every LLM token emitted inside any node. | Token-by-token chat UIs. |
| `"custom"` | Whatever you passed to `StreamWriter` / `get_stream_writer()`. | Domain-specific progress events. |
| `"checkpoints"` | Checkpoint payloads (`config`, `values`, `metadata`, `next`, `parent_config`, `tasks`). | Audit logs, progress DBs. |
| `"tasks"` | Task start / result events (`id`, `name`, `input`, `triggers` / `id`, `name`, `error`, `interrupts`, `result`). | Observability dashboards. |
| `"debug"` | All checkpoint + task events wrapped in a `DebugPayload` with step number and timestamp. | Replacing prints while developing. |

You can also pass a **list** of modes. With `version="v1"` the iterator yields `(mode, data)` tuples; with `version="v2"` every chunk is a `StreamPart` and you discriminate by `chunk["type"]`:

```python
# v1 list: yields (mode, data) tuples
for mode, data in graph.stream(inp, cfg, stream_mode=["updates", "messages"]):
    if mode == "updates":
        ...
    elif mode == "messages":
        token, meta = data

# v2 list: yields StreamPart dicts — use chunk["type"] to discriminate
for chunk in graph.stream(inp, cfg, stream_mode=["updates", "custom"], version="v2"):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Custom event: {chunk['data']}")
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

### `StreamPart` typed union

The `StreamPart` union (from `langgraph.types`) is the sum of all per-mode TypedDicts:

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

Each `StreamPart` has three guaranteed fields:

| Field | Type | Meaning |
|---|---|---|
| `type` | `str` (mode name) | Discriminator — narrow to the concrete TypedDict. |
| `ns` | `tuple[str, ...]` | Namespace path (empty tuple for root graph events). |
| `data` | varies per type | The actual payload — see per-mode sections below. |

#### Complete per-mode TypedDict shapes

```python
class ValuesStreamPart(TypedDict):
    type: Literal["values"]
    ns: tuple[str, ...]
    data: OutputT                   # full state (dict / Pydantic / dataclass)
    interrupts: tuple[Interrupt, ...]

class UpdatesStreamPart(TypedDict):
    type: Literal["updates"]
    ns: tuple[str, ...]
    data: dict[str, Any]            # {node_name: node_output}

class MessagesStreamPart(TypedDict):
    type: Literal["messages"]
    ns: tuple[str, ...]
    data: tuple[AnyMessage, dict[str, Any]]  # (message_chunk, metadata)

class CustomStreamPart(TypedDict):
    type: Literal["custom"]
    ns: tuple[str, ...]
    data: Any                       # whatever StreamWriter emitted

class CheckpointStreamPart(TypedDict):
    type: Literal["checkpoints"]
    ns: tuple[str, ...]
    data: CheckpointPayload         # see "checkpoints" section

class TasksStreamPart(TypedDict):
    type: Literal["tasks"]
    ns: tuple[str, ...]
    data: TaskPayload | TaskResultPayload

class DebugStreamPart(TypedDict):
    type: Literal["debug"]
    ns: tuple[str, ...]
    data: DebugPayload              # _DebugCheckpointPayload | _DebugTaskPayload | _DebugTaskResultPayload
```

#### Narrowing with `match` / `if`

```python
async for part in graph.astream(inp, cfg, version="v2"):
    match part["type"]:
        case "values":
            state = part["data"]
            pending = part["interrupts"]   # tuple[Interrupt, ...]
        case "updates":
            updates: dict[str, Any] = part["data"]  # {node_name: output}
        case "messages":
            msg, meta = part["data"]       # (BaseMessage, metadata dict)
        case "custom":
            payload = part["data"]         # whatever StreamWriter wrote
        case "checkpoints":
            cp = part["data"]              # CheckpointPayload
        case "tasks":
            ev = part["data"]              # TaskPayload or TaskResultPayload
        case "debug":
            dbg = part["data"]             # DebugPayload — discriminate on dbg["type"]
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

Full v2 example:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    count: int
    message: str


def increment(state: State) -> dict:
    return {"count": state["count"] + 1, "message": f"step {state['count'] + 1}"}


graph = (
    StateGraph(State)
    .add_node("increment", increment)
    .add_edge(START, "increment")
    .add_edge("increment", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "demo"}}

for part in graph.stream({"count": 0, "message": ""}, cfg, stream_mode="values", version="v2"):
    print(part["type"], part["ns"])
    print("state:", part["data"])
    if part["interrupts"]:
        print("pending interrupts:", part["interrupts"])
# values ()
# state: {'count': 1, 'message': 'step 1'}
```

### `"updates"`

Emits one event per node per step, keyed by node name:

```python
{"planner": {"messages": [...], "next": "writer"}}
```

Interrupts show up as a sibling key `"__interrupt__"` whose value is a tuple of `Interrupt` dataclasses. Parallel nodes in the same super-step produce separate events.

Full v2 example:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    x: int
    y: int


def node_a(state: State) -> dict:
    return {"x": state["x"] * 2}


def node_b(state: State) -> dict:
    return {"y": state["y"] + 10}


graph = (
    StateGraph(State)
    .add_node("node_a", node_a)
    .add_node("node_b", node_b)
    .add_edge(START, "node_a")
    .add_edge("node_a", "node_b")
    .add_edge("node_b", END)
    .compile()
)

for part in graph.stream({"x": 3, "y": 0}, stream_mode="updates", version="v2"):
    # part["type"] == "updates"
    # part["data"] == {"node_a": {"x": 6}} then {"node_b": {"y": 10}}
    for node_name, node_output in part["data"].items():
        print(f"Node `{node_name}` returned: {node_output}")

# Detect interrupts in v1 updates mode
for chunk in graph.stream(inp, cfg, stream_mode="updates"):
    if "__interrupt__" in chunk:
        for interrupt in chunk["__interrupt__"]:
            print("Interrupt:", interrupt.value)
```

### `"messages"`

Yields tuples of `(message, metadata)` for every LLM invocation inside any node.

- `message` — usually an `AIMessageChunk`; see `langchain_core.messages`. Concatenate `.content` for the full text.
- `metadata` — dict with the following keys:

| Metadata key | Type | Description |
|---|---|---|
| `langgraph_step` | `int` | Execution step number within the current run. |
| `langgraph_node` | `str` | Name of the node that produced this token. |
| `langgraph_triggers` | `list[str]` | Channel writes that caused this node to execute. |
| `langgraph_path` | `tuple[str, ...]` | Full namespace path, including subgraph nesting. |
| `langgraph_checkpoint_ns` | `str` | Checkpoint namespace string for this execution context. |
| `ls_model_name` | `str` | LangSmith model name tag (if set on the LLM). |
| `ls_provider` | `str` | LangSmith provider tag (if set on the LLM). |

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

Full v2 example filtering by node name:

```python
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    topic: str
    joke: str
    poem: str


def write_joke(state: State):
    joke_response = model.invoke(
        [{"role": "user", "content": f"Write a joke about {state['topic']}"}]
    )
    return {"joke": joke_response.content}


def write_poem(state: State):
    poem_response = model.invoke(
        [{"role": "user", "content": f"Write a short poem about {state['topic']}"}]
    )
    return {"poem": poem_response.content}


graph = (
    StateGraph(State)
    .add_node(write_joke)
    .add_node(write_poem)
    # run both concurrently from START
    .add_edge(START, "write_joke")
    .add_edge(START, "write_poem")
    .compile()
)

# v2: use chunk["type"] to identify messages chunks, then filter by node
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # Only print tokens from the poem node
        if msg.content and metadata["langgraph_node"] == "write_poem":
            print(msg.content, end="|", flush=True)
```

Accessing all available metadata fields:

```python
for chunk in graph.stream(inputs, stream_mode="messages", version="v2"):
    if chunk["type"] == "messages":
        msg, meta = chunk["data"]
        print(f"step={meta['langgraph_step']}")
        print(f"node={meta['langgraph_node']}")
        print(f"triggers={meta['langgraph_triggers']}")
        print(f"path={meta['langgraph_path']}")
        print(f"checkpoint_ns={meta['langgraph_checkpoint_ns']}")
```

### `"custom"`

Write arbitrary values from inside a node using `get_stream_writer()` (preferred, context-var based) or by declaring `stream_writer: StreamWriter` as a node parameter (injection-based).

#### `StreamWriter` type

```python
# from langgraph.types
StreamWriter = Callable[[Any], None]
```

LangGraph always injects a `StreamWriter` into nodes that declare it as a parameter, but the callable is a no-op unless `stream_mode="custom"` (or a list containing `"custom"`) is active. This means you can safely leave `stream_writer` calls in production code without performance impact when not streaming.

#### Method 1: `get_stream_writer()` (recommended)

`get_stream_writer()` retrieves the `StreamWriter` via a context variable — no parameter declaration needed. Works in both `StateGraph` nodes and `@task` decorators. Requires Python 3.11+ in async contexts.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


class State(TypedDict):
    topic: str
    joke: str


def generate_joke(state: State):
    writer = get_stream_writer()
    writer({"status": "thinking of a joke..."})
    result = f"Why did the {state['topic']} go to school? To get a sundae education!"
    writer({"status": "done"})
    return {"joke": result}


graph = (
    StateGraph(State)
    .add_node(generate_joke)
    .add_edge(START, "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

Output:

```
Status: thinking of a joke...
Status: done
Node generate_joke updated: {'joke': 'Why did the ice cream go to school? To get a sundae education!'}
```

#### Method 2: `StreamWriter` as a node parameter (injection)

```python
from langgraph.types import StreamWriter


def my_node(state: State, stream_writer: StreamWriter) -> dict:
    stream_writer({"progress": "Step 1 complete"})
    # ... do work ...
    stream_writer({"progress": "Step 2 complete"})
    return {"result": "done"}
```

#### Using `get_stream_writer()` inside a tool

```python
from langchain.tools import tool
from langgraph.config import get_stream_writer


@tool
def query_database(query: str) -> str:
    """Query the database."""
    writer = get_stream_writer()
    writer({"data": "Retrieved 0/100 records", "type": "progress"})
    # ... perform query ...
    writer({"data": "Retrieved 100/100 records", "type": "progress"})
    return "some-answer"


# Consume the custom events
for chunk in graph.stream(inputs, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        print(f"{chunk['data']['type']}: {chunk['data']['data']}")
```

#### `get_stream_writer()` in the functional API

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

checkpointer = InMemorySaver()


@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> int:
    writer = get_stream_writer()
    writer("Started processing")
    result = inputs["x"] * 2
    writer(f"Result is {result}")
    return result


config = {"configurable": {"thread_id": "abc"}}

# v1 list form: yields (mode, chunk) tuples
for mode, chunk in main.stream({"x": 5}, stream_mode=["custom", "updates"], config=config):
    print(f"{mode}: {chunk}")
```

Output:

```
custom: Started processing
custom: Result is 10
updates: {'main': 10}
```

Outside `stream_mode="custom"`, calls to the `StreamWriter` are no-ops — it's safe to leave them in production code.

### `"checkpoints"`

Emits a `CheckpointPayload` each time a checkpoint is created:

```python
# CheckpointPayload shape
{
    "config": {...},           # RunnableConfig pointer to this checkpoint
    "metadata": {
        "source": "loop",      # "input" | "loop" | "update"
        "step": 1,
        "parents": {},
        "run_id": "...",
        "writes": {...},       # channel writes at this step
    },
    "values": {<state>},       # full state at this checkpoint
    "next": ["writer"],        # nodes scheduled to run next
    "parent_config": {...},    # pointer to parent checkpoint (or None)
    "tasks": [
        {
            "id": "...",
            "name": "planner",
            "result": {...},
            "state": None,
            "interrupts": [],
        }
    ],
}
```

Requires a checkpointer; otherwise the mode yields nothing.

Full v2 example:

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = (
    StateGraph(State)
    # ... nodes / edges ...
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "audit-run"}}

async for part in graph.astream(inp, cfg, stream_mode="checkpoints", version="v2"):
    # part["type"] == "checkpoints"
    cp = part["data"]           # CheckpointPayload
    print(f"step={cp['metadata']['step']}")
    print(f"next={cp['next']}")
    print(f"state={cp['values']}")
```

### `"tasks"`

Two payload shapes interleaved on one stream — a `TaskPayload` when a task starts and a `TaskResultPayload` when it finishes:

```python
# TaskPayload — emitted when a node begins executing
{
    "id": "abc123",             # unique task identifier
    "name": "planner",          # node name
    "input": {...},             # task input data
    "triggers": ["branch:to:planner"],  # channel writes that triggered this task
}

# TaskResultPayload — emitted when a node finishes
{
    "id": "abc123",             # same id as the corresponding TaskPayload
    "name": "planner",          # node name
    "error": None,              # error message string, or None on success
    "interrupts": [],           # list of interrupt dicts, if any
    "result": {"x": 1},         # channel writes returned by this node
}
```

Full v2 example:

```python
for part in graph.stream(inp, cfg, stream_mode="tasks", version="v2"):
    ev = part["data"]           # TaskPayload | TaskResultPayload
    if "triggers" in ev:
        # it's a TaskPayload (task start)
        print(f"[START] {ev['name']} (id={ev['id']}) triggered by {ev['triggers']}")
    else:
        # it's a TaskResultPayload (task end)
        if ev["error"]:
            print(f"[ERROR] {ev['name']}: {ev['error']}")
        else:
            print(f"[DONE ] {ev['name']}: result={ev['result']}")
```

Pair with `"messages"` to annotate token events with the owning task:

```python
for chunk in graph.stream(inp, cfg, stream_mode=["tasks", "messages"], version="v2"):
    if chunk["type"] == "tasks" and "triggers" not in chunk["data"]:
        # TaskResultPayload
        print(f"task finished: {chunk['data']['name']}")
    elif chunk["type"] == "messages":
        msg, meta = chunk["data"]
        print(f"  token from {meta['langgraph_node']}: {msg.content!r}")
```

### `"debug"`

Emits `DebugPayload` wrappers — a discriminated union of three event types, all sharing a `step` (int) and `timestamp` (ISO 8601 str):

```python
# _DebugCheckpointPayload
{"type": "checkpoint", "step": 1, "timestamp": "2026-05-24T12:00:00Z", "payload": <CheckpointPayload>}

# _DebugTaskPayload
{"type": "task", "step": 1, "timestamp": "2026-05-24T12:00:01Z", "payload": <TaskPayload>}

# _DebugTaskResultPayload
{"type": "task_result", "step": 1, "timestamp": "2026-05-24T12:00:02Z", "payload": <TaskResultPayload>}
```

Full v2 example:

```python
for part in graph.stream(inp, cfg, stream_mode="debug", version="v2"):
    dbg = part["data"]          # DebugPayload
    match dbg["type"]:
        case "checkpoint":
            print(f"[step {dbg['step']}] checkpoint — next: {dbg['payload']['next']}")
        case "task":
            print(f"[step {dbg['step']}] task start — {dbg['payload']['name']}")
        case "task_result":
            r = dbg["payload"]
            status = "ERROR" if r["error"] else "OK"
            print(f"[step {dbg['step']}] task result — {r['name']} [{status}]")
```

Useful for replacing `print()` during development — controlled by a flag:

```python
DEBUG = True

for part in graph.stream(
    inp, cfg,
    stream_mode="debug" if DEBUG else "updates",
    version="v2",
):
    ...
```

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
        print(msg.content, end="", flush=True)
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

With v2 for cleaner type discrimination:

```python
import json

async for chunk in graph.astream(inp, cfg, stream_mode=["updates", "messages"], version="v2"):
    if chunk["type"] == "updates" and "__interrupt__" in chunk["data"]:
        interrupts = [i.value for i in chunk["data"]["__interrupt__"]]
        yield f"event: interrupt\ndata: {json.dumps(interrupts)}\n\n"
    elif chunk["type"] == "messages":
        msg, _ = chunk["data"]
        if msg.content:
            yield f"event: token\ndata: {msg.content}\n\n"
```

### 3. Progress bar using `"custom"`

```python
from langgraph.config import get_stream_writer


def download(state: State) -> dict:
    writer = get_stream_writer()
    urls = state["urls"]
    for i, url in enumerate(urls, start=1):
        fetch(url)
        writer({"pct": int(100 * i / len(urls)), "url": url})
    return {"done": True}


# Consume progress on the caller side
for chunk in graph.stream(inp, cfg, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        pct = chunk["data"]["pct"]
        print(f"\r[{'#' * (pct // 10):<10}] {pct}%", end="", flush=True)
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

### 6. Full StreamPart dispatch with all seven modes

```python
from langgraph.types import (
    ValuesStreamPart,
    UpdatesStreamPart,
    MessagesStreamPart,
    CustomStreamPart,
    CheckpointStreamPart,
    TasksStreamPart,
    DebugStreamPart,
)

ALL_MODES = ["values", "updates", "messages", "custom", "checkpoints", "tasks", "debug"]

async for part in graph.astream(inp, cfg, stream_mode=ALL_MODES, version="v2"):
    match part["type"]:
        case "values":
            print("STATE:", part["data"], "INTERRUPTS:", part["interrupts"])
        case "updates":
            print("UPDATES:", part["data"])
        case "messages":
            msg, meta = part["data"]
            print(f"TOKEN [{meta['langgraph_node']}]:", repr(msg.content))
        case "custom":
            print("CUSTOM:", part["data"])
        case "checkpoints":
            cp = part["data"]
            print(f"CHECKPOINT step={cp['metadata']['step']} next={cp['next']}")
        case "tasks":
            ev = part["data"]
            if "triggers" in ev:
                print(f"TASK START: {ev['name']}")
            else:
                print(f"TASK END: {ev['name']} error={ev['error']}")
        case "debug":
            dbg = part["data"]
            print(f"DEBUG [{dbg['type']}] step={dbg['step']}")
```

### 7. `StreamWriter` injection for library code

When building reusable graph components that should not depend on `get_stream_writer()`'s context-var mechanism, declare the writer as a parameter instead:

```python
from langgraph.types import StreamWriter


def reusable_node(state: State, stream_writer: StreamWriter) -> dict:
    """Works whether called inside a graph stream or directly in tests."""
    stream_writer({"event": "started"})
    result = do_work(state)
    stream_writer({"event": "finished", "output": result})
    return {"output": result}
```

In unit tests, pass a no-op or a list-appending collector:

```python
events: list = []
reusable_node({"x": 1}, stream_writer=events.append)
assert events == [{"event": "started"}, {"event": "finished", "output": ...}]
```

## Gotchas

- **Default stream mode is `"updates"`.** Passing `stream_mode=None` inherits from the graph's own default (which is `"updates"` for root graphs and `"values"` when invoked as a subgraph step).
- **`"checkpoints"` needs a checkpointer.** Without one you get no events, not an error.
- **`stream_mode="messages"` requires callbacks.** If you construct LLMs outside LangGraph and hand back messages manually, you won't see tokens. Use the LangChain `ChatModel` interface inside a node so callbacks fire.
- **v2 is opt-in per call.** There is no global switch. Always pass `version="v2"` if you want typed output; otherwise you get the legacy shape.
- **`print_mode=` is separate from `stream_mode=`.** `print_mode` prints to stdout for debugging and does not change what `stream()` yields.
- **`subgraphs=True` changes the tuple shape.** With a single mode you get `(ns, data)`; with a list of modes you get `(ns, mode, data)`. With `version="v2"` this collapses because `ns` is always part of the `StreamPart`.
- **The `"__interrupt__"` key only appears in `"updates"` v1 mode.** For `"values"` v2, interrupts live in `part["interrupts"]`.
- **`stream_writer` calls are lost outside `"custom"` mode.** That's intentional — but if you expected to see them, add `"custom"` to `stream_mode`.
- **`print_mode` is additive.** Passing `print_mode="updates"` both prints updates **and** keeps whatever your `stream_mode` emits to the iterator.
- **`get_stream_writer()` requires Python 3.11+ in async contexts.** In async nodes on Python 3.10 and below, declare `stream_writer: StreamWriter` as a parameter instead.
- **v2 multi-mode no longer yields `(mode, data)` tuples.** When using a list of modes with `version="v2"`, every chunk is a `StreamPart` dict — use `chunk["type"]` instead of unpacking a tuple.
- **`"debug"` is a superset of `"checkpoints"` + `"tasks"`.** Using `"debug"` alone in development is equivalent to receiving both, wrapped with step/timestamp context.

## Breaking changes

| Version | Change |
|---|---|
| 1.2.1 | `get_stream_writer()` added to `langgraph.config` as the preferred context-var-based alternative to parameter injection. `StreamWriter` injection as a node keyword argument remains supported. |
| 1.1 | `version="v2"` on `stream`/`astream` yields typed `StreamPart` dicts; `invoke`/`ainvoke` with `version="v2"` return `GraphOutput`. `GraphOutput[key]` indexing raises `DeprecationWarning`. With v2, multi-mode lists yield `StreamPart` dicts (not `(mode, data)` tuples). |
| 1.0 | `stream_mode="tasks"` split from `"debug"`; `"checkpoints"` added as its own mode. `StreamMode` literal type (`"values" \| "updates" \| "checkpoints" \| "tasks" \| "debug" \| "messages" \| "custom"`) stabilized in `langgraph.types`. |
| 0.6 | `interrupt_before` / `interrupt_after` on `invoke`/`stream` accept `"*"` for all nodes. |
| 0.5 | `checkpoint_during=False` deprecated in favor of `durability="exit"`. |

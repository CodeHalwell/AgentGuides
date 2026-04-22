---
title: "StateGraph — API reference"
description: "Exhaustive reference for the StateGraph builder and CompiledStateGraph runtime — constructor, add_node, add_edge, add_conditional_edges, add_sequence, compile, plus retry/cache/defer/destinations options."
framework: langgraph
language: python
sidebar:
  label: "Ref · StateGraph"
  order: 30
---

# StateGraph — API reference

Verified against **`langgraph==1.1.9`** (`/tmp/langgraph-install/langgraph/graph/state.py`, `/tmp/langgraph-install/langgraph/types.py`).

`StateGraph` is the primary graph builder. You declare a state schema, add nodes and edges, then call `.compile()` to get a `CompiledStateGraph` that implements the LangChain `Runnable` protocol (`invoke` / `stream` / `ainvoke` / `astream`).

## Minimal runnable example

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    counter: int


def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}


builder: StateGraph[State, None, State, State] = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"counter": 0}, config))  # {'counter': 1}
```

## Imports at a glance

All symbols below come from the exact module path in the installed package.

| Symbol | Import path |
|---|---|
| `StateGraph`, `CompiledStateGraph` | `langgraph.graph.state` (also re-exported from `langgraph.graph`) |
| `START`, `END` | `langgraph.graph` (re-exported from `langgraph.constants`) |
| `add_messages`, `MessagesState`, `REMOVE_ALL_MESSAGES` | `langgraph.graph.message` |
| `Command`, `Send`, `interrupt`, `StateSnapshot`, `Interrupt`, `Overwrite`, `RetryPolicy`, `CachePolicy`, `Durability`, `GraphOutput` | `langgraph.types` |
| `Runtime`, `ExecutionInfo`, `ServerInfo`, `get_runtime` | `langgraph.runtime` |
| `InMemorySaver` | `langgraph.checkpoint.memory` |
| `BaseStore`, `InMemoryStore` | `langgraph.store.base`, `langgraph.store.memory` |

The top-level `langgraph.graph.__init__` only re-exports `START`, `END`, `StateGraph`, `add_messages`, `MessagesState`, `MessageGraph` — everything else must be imported from its real module.

## Constructor

```python
StateGraph(
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
)
```

- `state_schema` — a `TypedDict`, dataclass, or Pydantic `BaseModel`. Each field defines a **channel**; annotating with `Annotated[T, reducer]` turns it into a reducing channel.
- `context_schema` — run-scoped read-only context (e.g. `user_id`, `db_conn`). Injected via `Runtime[ContextT]` (see below).
- `input_schema` / `output_schema` — optional narrower schemas that differ from the main state.

Deprecated kwargs that still work but warn:
- `config_schema` → use `context_schema` (deprecated since v0.6).
- `input`, `output` → use `input_schema`, `output_schema` (deprecated since v0.5).

## Reducers and `add_messages`

A reducer is a function `(current, update) -> new_value` attached to a state key with `Annotated[...]`.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    visited: Annotated[list[str], operator.add]
```

- `add_messages` merges two message lists **by `id`**: same-id messages overwrite, new-id messages append. Pass `format="langchain-openai"` to coerce to OpenAI-format blocks (requires `langchain-core>=0.3.11`).
- `REMOVE_ALL_MESSAGES` (from `langgraph.graph.message`) is a sentinel id on a `RemoveMessage(id=REMOVE_ALL_MESSAGES)` that wipes the history.
- Without a reducer, a channel uses `LastValue` semantics: the latest write wins, and two concurrent writes in one super-step raise `InvalidUpdateError`.

Bypass a reducer for a single write with `Overwrite`:

```python
from langgraph.types import Overwrite

def replace_messages(state: ChatState) -> dict:
    return {"messages": Overwrite(value=[])}
```

## `add_node`

Four overloads, all returning `Self` for chaining:

```python
builder.add_node(fn)                              # name = fn.__name__
builder.add_node("my_node", fn)                   # explicit name
builder.add_node(fn, input_schema=NodeInput)      # per-node input schema
builder.add_node("my_node", fn, input_schema=NodeInput)
```

All overloads accept the same keyword options:

| Option | Type | Effect |
|---|---|---|
| `defer` | `bool` | Run this node **only when the graph is about to finish** (after all other tasks drain). Useful for summarization/finalization. |
| `metadata` | `dict` | Attached to the node; surfaces in tracing/streaming metadata. |
| `input_schema` | `type` | Node receives a narrower shape. Channels outside this schema are not visible. |
| `retry_policy` | `RetryPolicy \| Sequence[RetryPolicy]` | Controls retries on exceptions. First matching policy in a sequence wins. |
| `cache_policy` | `CachePolicy` | Cache the node's output by input hash. Requires a `cache=` backend on `.compile()`. |
| `destinations` | `dict[str, str] \| tuple[str, ...]` | Visualization hint for edgeless nodes that return `Command(goto=...)`. Does **not** affect execution. |

A node's callable signature can be any of:

```python
def node(state: State): ...
def node(state: State, config: RunnableConfig): ...
def node(state: State, runtime: Runtime[Context]): ...
def node(state: State, *, writer: StreamWriter): ...   # opt-in custom stream
async def node(state: State, runtime: Runtime[Context]): ...
```

Return types: `dict`, the state schema instance, `None`, or a `Command`. Returning `None` is a no-op on all channels.

### `RetryPolicy`

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "risky",
    risky_fn,
    retry_policy=RetryPolicy(
        initial_interval=0.5,   # seconds before first retry
        backoff_factor=2.0,     # exponential multiplier
        max_interval=128.0,
        max_attempts=3,
        jitter=True,
        retry_on=ConnectionError,   # type, tuple, or Callable[[Exception], bool]
    ),
)
```

`retry_on` accepts an exception type, a tuple of types, or a predicate. Default is `langgraph._internal._retry.default_retry_on` (retries on `httpx.HTTPStatusError` 5xx, `httpx.TransportError`, `ConnectionError`, and request timeouts).

### `CachePolicy`

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

builder.add_node("lookup", lookup, cache_policy=CachePolicy(ttl=300))
graph = builder.compile(cache=InMemoryCache())
```

`key_func` defaults to pickle-hashing the input. Pass a custom `(input) -> str | bytes` for deterministic cache keys.

## `add_edge`

```python
builder.add_edge("a", "b")                # single
builder.add_edge(["a", "b"], "c")         # waits for ALL of a, b (barrier edge)
builder.add_edge(START, "a")              # entry point
builder.add_edge("last", END)             # finish point
```

Raises `ValueError` if the start is `END`, the end is `START`, or a named node is missing.

## `add_conditional_edges`

```python
builder.add_conditional_edges(
    source: str,
    path: Callable[..., Hashable | Sequence[Hashable]],
    path_map: dict[Hashable, str] | list[str] | None = None,
)
```

`path` is called with the state (and optionally config/runtime) and returns:

- a single node name → routes there,
- a list of node names → fan-out to all,
- one or more `Send(node, arg)` instances → map-reduce with custom per-destination state,
- the string `"END"` or `END` → stop.

If your `path` returns arbitrary labels, map them to node names with `path_map`. Adding a `Literal[...]` return annotation or passing `path_map` keeps the Mermaid diagram accurate — without either, the visualizer assumes every node is reachable.

## `add_sequence`

```python
builder.add_sequence([
    load_docs,
    ("retrieve", retrieve_fn),     # tuple = (name, callable)
    rerank,
])
```

Wires the nodes in order with auto-generated edges and uses each callable's `__name__` if no explicit name is given. Raises on empty input or duplicate names.

## Entry / exit helpers

```python
builder.set_entry_point("planner")            # == add_edge(START, "planner")
builder.set_finish_point("writer")            # == add_edge("writer", END)

builder.set_conditional_entry_point(
    router, path_map={"yes": "a", "no": "b"}
)
```

## `compile(...)`

```python
graph = builder.compile(
    checkpointer=None,       # BaseCheckpointSaver | True | False | None
    *,
    cache=None,              # BaseCache, needed for CachePolicy
    store=None,              # BaseStore for long-term memory
    interrupt_before=None,   # list[str] | "*" | None
    interrupt_after=None,    # list[str] | "*" | None
    debug=False,
    name=None,
)
```

- `checkpointer=True` is only valid when the graph is used as a **subgraph** — it inherits the parent's checkpointer. On a root graph, `True` raises `RuntimeError`.
- `checkpointer=False` explicitly disables checkpointing even when the parent has one.
- `interrupt_before` / `interrupt_after` accept `"*"` (all nodes) or a list of node names.
- `store` is required whenever any tool/node uses `InjectedStore` or reads `runtime.store`.

Returns a `CompiledStateGraph`, which exposes (all inherited from `Pregel`):

| Method | Purpose |
|---|---|
| `invoke(input, config=None, *, context=None, stream_mode=None, interrupt_before=None, interrupt_after=None, durability=None, version="v1")` | Run to completion, return final state. |
| `stream(...)` | Yield per-step events (see the [Streaming modes reference](./reference-streaming-modes/)). |
| `ainvoke` / `astream` | Async variants. |
| `get_state(config, *, subgraphs=False)` | Return the current `StateSnapshot` for a thread. Requires a checkpointer. |
| `get_state_history(config, *, filter=None, before=None, limit=None)` | Iterate historical snapshots (newest first). |
| `update_state(config, values, as_node=None, task_id=None)` | Write an update as if it came from `as_node`. |
| `bulk_update_state(config, supersteps)` | Apply multiple `StateUpdate` groups as distinct super-steps. |
| `get_subgraphs(namespace=None, recurse=False)` | Iterate nested compiled graphs. |
| `get_graph(...)` / `draw_mermaid()` / `draw_png()` | Visualization helpers. |

### Durability modes

Pass `durability="sync" | "async" | "exit"` on `invoke`/`stream`. Semantics:

| Mode | When checkpoints are persisted |
|---|---|
| `"sync"` | Before the next step begins. Strongest guarantee, slowest. |
| `"async"` | Written asynchronously while the next step runs. **Default.** |
| `"exit"` | Only at graph exit. Cheapest, no mid-run time-travel. |

`checkpoint_during=False` is deprecated and maps to `durability="exit"`.

## Runtime context (`Runtime[Context]`)

`Runtime` bundles per-run data separate from state. Added in v0.6.

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Ctx:
    user_id: str

def node(state: State, runtime: Runtime[Ctx]) -> dict:
    uid = runtime.context.user_id
    if runtime.store:
        memory = runtime.store.get(("users",), uid)
    return {...}

graph = StateGraph(State, context_schema=Ctx).add_node(node).compile()
graph.invoke({...}, context=Ctx(user_id="alice"))
```

Runtime fields:

- `context: ContextT` — what you passed in `context=`.
- `store: BaseStore | None` — what you passed to `compile(store=...)`.
- `stream_writer: (Any) -> None` — writes to `stream_mode="custom"`.
- `previous: Any` — functional API only, the last return value for this thread.
- `execution_info: ExecutionInfo | None` — `checkpoint_id`, `thread_id`, `run_id`, `node_attempt`, `node_first_attempt_time`.
- `server_info: ServerInfo | None` — set by LangGraph Platform only.

To get the config instead, add `config: RunnableConfig` as a parameter or call `get_config()` from `langgraph.config`.

## State schema: TypedDict vs Pydantic vs dataclass

All three work as `state_schema`. Since v1.1, `invoke()` **coerces** input dicts into the declared type before calling nodes.

```python
from pydantic import BaseModel

class State(BaseModel):
    counter: int = 0

graph.invoke({"counter": 0})
# Nodes receive State(counter=0) — a real Pydantic instance.
```

`Annotated[..., reducer]` works the same across all three schema styles. For Pydantic, use `Field(default_factory=...)` if the default depends on call time.

## Patterns

### 1. Fan-out / map-reduce with `Send`

```python
from langgraph.types import Send

def dispatch(state: State) -> list[Send]:
    return [Send("worker", {"item": i}) for i in state["items"]]

builder.add_conditional_edges("planner", dispatch)
builder.add_node("worker", worker_fn)
builder.add_edge("worker", "aggregate")
builder.add_edge(["planner", "worker"], "aggregate")   # barrier: wait for all workers
```

### 2. Deferred finalization

```python
builder.add_node("summarize", summarize_fn, defer=True)
builder.add_edge(START, "research")
builder.add_edge("research", "write")
builder.add_edge("write", "summarize")
# `summarize` runs last, after every other task in the run has drained.
```

### 3. Per-node retry on transient HTTP errors

```python
import httpx
from langgraph.types import RetryPolicy

builder.add_node(
    "fetch",
    fetch_fn,
    retry_policy=RetryPolicy(
        max_attempts=5,
        retry_on=(httpx.TransportError, httpx.HTTPStatusError),
    ),
)
```

### 4. Cached expensive step

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

builder.add_node("embed", embed_fn, cache_policy=CachePolicy(ttl=3600))
graph = builder.compile(cache=InMemoryCache(), checkpointer=InMemorySaver())
```

### 5. Narrow node input with `input_schema`

```python
class QueryOnly(TypedDict):
    query: str

def classify(state: QueryOnly) -> dict:
    return {"category": "billing" if "bill" in state["query"] else "other"}

builder.add_node("classify", classify, input_schema=QueryOnly)
```

The node cannot read unrelated channels and stays cheap to trace.

## Gotchas

- **Two writes, no reducer, one super-step → `InvalidUpdateError`.** Either add a reducer or stagger the writes with edges.
- **`checkpointer=None`** disables every feature that depends on persistence: `interrupt()`, `get_state`, `update_state`, `get_state_history`, time travel, thread-scoped memory. Use `InMemorySaver()` while developing.
- **`config_schema=` is deprecated**, but still accepted. Rename to `context_schema=` before v2.0.
- **`AgentState` / `AgentStatePydantic`** in `langgraph.prebuilt` are deprecated in v1.0 — they now live in `langchain.agents`.
- **`create_react_agent`** in `langgraph.prebuilt` is deprecated in v1.0 — migrate to `langchain.agents.create_agent`. The signature here still works; the deprecation is runtime-warning level.
- **Root graphs cannot have `checkpointer=True`.** That value is only for subgraphs inheriting from the parent.
- **`destinations=` does not route** — it only labels edges in the rendered diagram for nodes that return `Command(goto=...)`.

## Breaking changes

| Version | Change |
|---|---|
| 1.1 | `invoke()`/`stream()` coerce input dicts into the declared state schema for Pydantic/dataclass. V2 stream mode emits typed `StreamPart` dicts. Python 3.9 dropped. |
| 1.0 | `AgentState`, `AgentStatePydantic`, `create_react_agent` deprecated in favor of `langchain.agents.create_agent`. `ns`, `when`, `resumable`, `interrupt_id` removed from `Interrupt` (in v0.6). |
| 0.6 | `config_schema` on `StateGraph` deprecated; use `context_schema`. `Runtime[Ctx]` replaces ad-hoc `config["configurable"]` usage for run context. |
| 0.5 | `input` / `output` kwargs on `StateGraph.__init__` deprecated; use `input_schema` / `output_schema`. |

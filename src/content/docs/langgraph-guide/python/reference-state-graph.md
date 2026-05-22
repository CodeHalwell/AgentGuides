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

Verified against **`langgraph==1.2.1`** (modules: `langgraph.graph.state`, `langgraph.types`).

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
| `timeout` | `float \| timedelta \| TimeoutPolicy \| None` | Per-attempt timeout. A plain `float`/`timedelta` is the wall-clock limit; `TimeoutPolicy` adds idle-timeout and heartbeat support. |
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

### `TimeoutPolicy`

`TimeoutPolicy` (from `langgraph.types`) controls per-attempt cancellation for **async** nodes. A plain `float` or `timedelta` on the `timeout=` kwarg is a shorthand for `TimeoutPolicy(run_timeout=...)`.

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

@dataclass(**_DC_KWARGS)
class TimeoutPolicy:
    run_timeout:  float | timedelta | None = None   # hard wall-clock cap
    idle_timeout: float | timedelta | None = None   # max time between progress signals
    refresh_on:   Literal["auto", "heartbeat"] = "auto"
```

**`run_timeout`** — hard cap for a single attempt. Never refreshed, even by heartbeats.

**`idle_timeout`** — cap on how long the attempt may sit without a progress signal. Progress signals under `"auto"` mode include:
- Any LangChain callback event inside the node or its descendants.
- Explicit `runtime.heartbeat()` calls.
- Stream writer writes.

Under `"heartbeat"` mode, **only** `runtime.heartbeat()` resets the clock.

When the timeout fires, `NodeTimeoutError` is raised. The node's `retry_policy` (if set) then decides whether to retry.

```python
from langgraph.types import TimeoutPolicy, RetryPolicy

# Hard cap: abort after 30 seconds regardless of progress
builder.add_node(
    "llm_call",
    llm_node,
    timeout=30.0,   # same as TimeoutPolicy(run_timeout=30.0)
)

# Idle cap: reset on every LLM callback token, abort if silent for 10 s
builder.add_node(
    "streaming_llm",
    streaming_node,
    timeout=TimeoutPolicy(idle_timeout=10.0),
)

# Explicit heartbeat mode: node must call runtime.heartbeat() every 60 s
async def long_download(state: State, runtime: Runtime) -> dict:
    for chunk in download_chunks(state["url"]):
        runtime.heartbeat()          # prevents idle-timeout eviction
        process(chunk)
    return {"done": True}

builder.add_node(
    "download",
    long_download,
    timeout=TimeoutPolicy(idle_timeout=60.0, refresh_on="heartbeat"),
    retry_policy=RetryPolicy(max_attempts=3),
)
```

> **Sync nodes are not supported.** `timeout=` on a sync node raises `ValueError` at compile time. Use `asyncio.to_thread` or wrap the node in an async wrapper if you need timeouts on CPU-bound code.

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

| Field | Type | Description |
|---|---|---|
| `context` | `ContextT` | What you passed in `context=`. |
| `store` | `BaseStore \| None` | What you passed to `compile(store=...)`. |
| `stream_writer` | `(Any) -> None` | Writes a value to `stream_mode="custom"`. |
| `heartbeat` | `() -> None` | Signals progress to reset an idle timeout (see `TimeoutPolicy`). No-op outside an idle-timed attempt. |
| `previous` | `Any` | Functional API only — the last saved return value for this thread. |
| `execution_info` | `ExecutionInfo \| None` | Read-only metadata for the current node run (see below). |
| `server_info` | `ServerInfo \| None` | Set by LangGraph Platform only; `None` when running open-source. |
| `control` | `RunControl \| None` | Run-scoped cooperative draining handle (see below). |

To get the config instead, add `config: RunnableConfig` as a parameter or call `get_config()` from `langgraph.config`.

### `ExecutionInfo`

```python
from langgraph.runtime import ExecutionInfo
```

`runtime.execution_info` is a frozen dataclass with read-only per-run metadata. It is populated after task preparation, so it is `None` only in very early lifecycle hooks.

| Field | Type | Description |
|---|---|---|
| `checkpoint_id` | `str` | The ULID-style checkpoint ID written by this step. |
| `checkpoint_ns` | `str` | The checkpoint namespace (empty string for root; `"parent:task_id"` for subgraphs). |
| `task_id` | `str` | The Pregel task ID for the current node invocation. |
| `thread_id` | `str \| None` | The thread ID from `config["configurable"]["thread_id"]`. `None` when no checkpointer. |
| `run_id` | `str \| None` | The LangSmith run ID, if `run_id` was set in `RunnableConfig`. |
| `node_attempt` | `int` | Current attempt number (1-indexed). Increments on each retry. |
| `node_first_attempt_time` | `float \| None` | Unix timestamp for when the first attempt of this node started. |

```python
from langgraph.runtime import Runtime

def audit_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info
    if info:
        print(f"thread={info.thread_id} ns={info.checkpoint_ns} attempt={info.node_attempt}")
    return {}
```

`execution_info` also has a `patch(**overrides)` helper that returns a new `ExecutionInfo` with selected fields replaced. You will not normally need this outside testing.

### `Runtime.heartbeat()` — resetting idle timeouts

When a node is registered with `timeout=TimeoutPolicy(idle_timeout=..., refresh_on="heartbeat")`, the idle clock only resets on explicit `runtime.heartbeat()` calls. This is useful for long-running loops that do not naturally emit LangChain callback events.

```python
import asyncio
from langgraph.runtime import Runtime
from langgraph.types import TimeoutPolicy

async def batch_processor(state: State, runtime: Runtime) -> dict:
    results = []
    for i, item in enumerate(state["items"]):
        result = await process_item(item)
        results.append(result)
        # Signal that we're still alive — prevents idle-timeout eviction
        runtime.heartbeat()
    return {"results": results}

builder.add_node(
    "batch",
    batch_processor,
    timeout=TimeoutPolicy(idle_timeout=30.0, refresh_on="heartbeat"),
)
```

Outside an idle-timed attempt (e.g., when `timeout=None` or the node is sync), `runtime.heartbeat()` is a no-op.

### `RunControl` — cooperative draining

`runtime.control` is a `RunControl` instance that lets external code signal a graceful shutdown to a running node. The node cooperates by checking `runtime.drain_requested` and returning early when set.

```python
from langgraph.runtime import Runtime, RunControl

async def interruptible_worker(state: State, runtime: Runtime) -> dict:
    results = []
    for item in state["items"]:
        if runtime.drain_requested:
            # Graceful shutdown: save progress and exit early
            return {"results": results, "partial": True, "reason": runtime.drain_reason}
        result = await process(item)
        results.append(result)
    return {"results": results, "partial": False}
```

`RunControl` is populated automatically by the Pregel executor — you never create one yourself. Key properties:

| Property / Method | Description |
|---|---|
| `drain_requested: bool` | `True` after `request_drain()` has been called. |
| `drain_reason: str \| None` | The string passed to `request_drain()` (e.g. `"shutdown"`). |
| `request_drain(reason="shutdown")` | Called externally to signal the node to exit. |

The `runtime.drain_requested` and `runtime.drain_reason` convenience properties forward to `runtime.control` (or return `False` / `None` if `control` is `None`).

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

### 6. Async node with idle timeout + heartbeat

For nodes that stream from an LLM or process long lists, use `idle_timeout` so stalled attempts are cancelled before the hard wall-clock cap fires:

```python
import asyncio
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class BatchState(TypedDict):
    items: list[str]
    results: list[str]


async def process_batch(state: BatchState, runtime: Runtime) -> dict:
    results = []
    for item in state["items"]:
        # Expensive per-item work
        result = await asyncio.to_thread(expensive_cpu_work, item)
        results.append(result)
        # Reset the idle clock after each item
        runtime.heartbeat()
    return {"results": results}


builder.add_node(
    "process",
    process_batch,
    # Abort if no progress for 20 s; retry up to 3 times
    timeout=TimeoutPolicy(idle_timeout=20.0, refresh_on="heartbeat"),
    retry_policy=RetryPolicy(max_attempts=3, retry_on=asyncio.TimeoutError),
)
```

### 7. Reading `ExecutionInfo` for per-node tracing

Correlate a node's LangSmith run with your own observability system using `execution_info`:

```python
import logging
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class State(TypedDict):
    query: str
    answer: str


def traced_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info
    span_attrs = {
        "thread_id": info.thread_id if info else None,
        "checkpoint_id": info.checkpoint_id if info else None,
        "attempt": info.node_attempt if info else 1,
    }
    logger.info("node start", extra=span_attrs)
    answer = call_llm(state["query"])
    logger.info("node done", extra={**span_attrs, "tokens": len(answer)})
    return {"answer": answer}
```

## Gotchas

- **Two writes, no reducer, one super-step → `InvalidUpdateError`.** Either add a reducer or stagger the writes with edges.
- **`checkpointer=None`** disables every feature that depends on persistence: `interrupt()`, `get_state`, `update_state`, `get_state_history`, time travel, thread-scoped memory. Use `InMemorySaver()` while developing.
- **`config_schema=` is deprecated**, but still accepted. Rename to `context_schema=` before v2.0.
- **`AgentState` / `AgentStatePydantic`** in `langgraph.prebuilt` are deprecated in v1.0 — they now live in `langchain.agents`.
- **`create_react_agent`** in `langgraph.prebuilt` is deprecated in v1.0 — migrate to `langchain.agents.create_agent`. The signature here still works; the deprecation is runtime-warning level.
- **Root graphs cannot have `checkpointer=True`.** That value is only for subgraphs inheriting from the parent.
- **`destinations=` does not route** — it only labels edges in the rendered diagram for nodes that return `Command(goto=...)`.
- **`TimeoutPolicy` only works on async nodes.** Setting `timeout=` on a synchronous node raises `ValueError` at node registration. Convert the node to `async` or wrap it with `asyncio.to_thread`.
- **`runtime.heartbeat()` is a no-op without `refresh_on="heartbeat"`.** Under `refresh_on="auto"` (default), progress is detected automatically from callbacks and stream writes; calling `heartbeat()` is still valid but redundant.
- **`runtime.execution_info` is `None` briefly during startup.** Don't access it in lifecycle hooks that run before task preparation.
- **`RunControl` is populated automatically.** You cannot construct or inject your own `RunControl` — it is owned by the executor and forwarded through `Runtime.control`.

## Breaking changes

| Version | Change |
|---|---|
| 1.2 | `TimeoutPolicy` dataclass introduced with `run_timeout`, `idle_timeout`, `refresh_on`. `Runtime.heartbeat()` added. `RunControl` and cooperative draining added (`runtime.control`, `runtime.drain_requested`, `runtime.drain_reason`). `ExecutionInfo` extended with `checkpoint_ns` and `task_id` fields. |
| 1.1 | `invoke()`/`stream()` coerce input dicts into the declared state schema for Pydantic/dataclass. V2 stream mode emits typed `StreamPart` dicts. Python 3.9 dropped. |
| 1.0 | `AgentState`, `AgentStatePydantic`, `create_react_agent` deprecated in favor of `langchain.agents.create_agent`. `ns`, `when`, `resumable`, `interrupt_id` removed from `Interrupt` (in v0.6). |
| 0.6 | `config_schema` on `StateGraph` deprecated; use `context_schema`. `Runtime[Ctx]` replaces ad-hoc `config["configurable"]` usage for run context. |
| 0.5 | `input` / `output` kwargs on `StateGraph.__init__` deprecated; use `input_schema` / `output_schema`. |

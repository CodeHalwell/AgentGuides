---
title: "StateGraph — API reference"
description: "Exhaustive reference for the StateGraph builder and CompiledStateGraph runtime — constructor, add_node, add_edge, add_conditional_edges, add_sequence, compile, plus retry/cache/defer/destinations options."
framework: "langgraph"
language: "python"
sidebar:
  label: "Ref · StateGraph"
  order: 30
---

Verified against **`langgraph==%%version=langgraph%%`** (modules: `langgraph.graph.state`, `langgraph.types`).

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

```text
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
| `error_handler` | callable | Runs when the node fails after retries; receives a `NodeError`. See [Error handlers](#error-handlers). |
| `destinations` | `dict[str, str] \| tuple[str, ...]` | Declares where a node that returns `Command(goto=...)` can go, so the diagram shows those edges. It does not route, but the targets are checked at compile time: each must be a node in the **same** graph (use `Command(graph=Command.PARENT)` for parent targets and leave them out). |
| `trace_policy` | `TracePolicy \| None` | Control how this node's run is traced in LangSmith (omit inputs, outputs, or both). See [TracePolicy reference](/langgraph-guide/python/reference-trace-policy/). |

A node's callable signature can be any of:

```python
def node(state: State): ...
def node(state: State, config: RunnableConfig): ...
def node(state: State, runtime: Runtime[Context]): ...
def node(state: State, *, writer: StreamWriter): ...   # opt-in custom stream
async def node(state: State, runtime: Runtime[Context]): ...
```

Return types: `dict`, the state schema instance, `None`, or a `Command`. Returning `None` is a no-op on all channels.

## Node policies

This is the guide's canonical write-up of the per-node policies. Other pages link here rather than repeating it.

| Policy | `add_node` keyword | What it does |
|---|---|---|
| `RetryPolicy` | `retry_policy=` | Re-runs a failed attempt with exponential backoff |
| `TimeoutPolicy` (or a number) | `timeout=` | Cancels an attempt that runs too long or goes quiet. **Async nodes only.** |
| `CachePolicy` | `cache_policy=` | Skips the node when it has seen the same input before. Needs `compile(cache=...)`. |
| error handler | `error_handler=` | A function that runs instead of failing the graph once retries are exhausted |

`set_node_defaults(...)` applies any of these to every node; per-node values override the defaults. The order on failure is: timeout fires (raising `NodeTimeoutError`) → `retry_policy` decides whether to retry → when retries are exhausted, the error handler runs, if there is one; otherwise the exception propagates out of `invoke`.

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
        max_attempts=3,         # counts the first attempt
        jitter=True,
        retry_on=ConnectionError,   # type, tuple of types, or Callable[[Exception], bool]
    ),
)
```

The default `retry_on` (`langgraph.types.default_retry_on`) is a block-list, not an allow-list. It does **not** retry `ValueError`, `TypeError`, `ArithmeticError`, `ImportError`, `LookupError`, `NameError`, `SyntaxError`, `RuntimeError`, `ReferenceError`, `StopIteration`/`StopAsyncIteration` or any `OSError` (which includes the built-in `TimeoutError`). For `httpx.HTTPStatusError` and `requests.HTTPError` it retries only 5xx responses (so not HTTP 429). Everything else is retried, including `ConnectionError` and LangGraph's `NodeTimeoutError`. Pass your own predicate when that doesn't fit, for example `retry_on=lambda e: isinstance(e, (ConnectionError, NodeTimeoutError))`.

`retry_policy=` also takes a **list**. LangGraph uses the **first policy whose `retry_on` matches** the exception; the policies don't combine:

```python
builder.add_node(
    "unreliable",
    unreliable_fn,
    retry_policy=[
        RetryPolicy(initial_interval=0.01, max_attempts=5, retry_on=ConnectionError),  # quick retries
        RetryPolicy(initial_interval=0.5, max_attempts=2, retry_on=ValueError),        # fewer, slower
    ],
)
```

### `CachePolicy`

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

builder.add_node("lookup", lookup, cache_policy=CachePolicy(ttl=300))   # seconds; None = no expiry
graph = builder.compile(cache=InMemoryCache())
```

`key_func` defaults to hashing the pickled node input. Pass a `(input) -> str | bytes` function for deterministic keys. Clear entries with `graph.clear_cache()` or `graph.clear_cache(nodes=["lookup"])`. Backends are covered in the [cache backends reference](/langgraph-guide/python/reference-cache-backends/).

### `TimeoutPolicy`

A plain `float` or `timedelta` on `timeout=` is shorthand for `TimeoutPolicy(run_timeout=...)`. The dataclass fields:

```text
TimeoutPolicy(
    run_timeout:  float | timedelta | None = None,   # hard cap per attempt, never refreshed
    idle_timeout: float | timedelta | None = None,   # cap on time without a progress signal
    refresh_on:   "auto" | "heartbeat" = "auto",
)
```

Under `refresh_on="auto"`, any LangChain callback event inside the node (for example streamed model tokens), a stream-writer write or a `runtime.heartbeat()` call resets the idle clock. Under `"heartbeat"`, only `runtime.heartbeat()` does. When a timeout fires, `NodeTimeoutError` (from `langgraph.errors`) is raised; it is not a subclass of the built-in `TimeoutError`, so the default `retry_on` retries it.

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy


class State(TypedDict, total=False):
    items: list[str]
    done: int


async def process(state: State, runtime: Runtime) -> dict:
    for _ in state["items"]:
        await asyncio.sleep(0.01)     # real work here
        runtime.heartbeat()           # progress signal
    return {"done": len(state["items"])}


graph = (
    StateGraph(State)
    .add_node(
        "process",
        process,
        timeout=TimeoutPolicy(run_timeout=60, idle_timeout=5, refresh_on="heartbeat"),
        retry_policy=RetryPolicy(max_attempts=2),
    )
    .add_edge(START, "process")
    .compile()
)
print(asyncio.run(graph.ainvoke({"items": ["a", "b", "c"]})))   # {'items': [...], 'done': 3}
```

**Sync nodes cannot have timeouts.** `add_node(..., timeout=...)` accepts a sync function, but `compile()` raises `ValueError: Node timeouts are only supported for async nodes`. This includes sync error handlers when `set_node_defaults(timeout=...)` is set, because the default timeout applies to handler nodes too. Make the node `async def`; for blocking code, call it with `await asyncio.to_thread(...)`.

### Error handlers

An error handler is a function registered with `add_node(..., error_handler=fn)` (or as a default with `set_node_defaults`). It runs when the node has failed and retries are exhausted. LangGraph injects the failure into the parameter annotated `NodeError`; the annotation is what matters, so `error: Exception` does not work (the call fails with `TypeError: missing 1 required positional argument`).

`NodeError` has two fields: `node` (the failed node's name) and `error` (the exception). The handler returns a state update (or a `Command`), and the graph continues along the failed node's outgoing edges.

### `set_node_defaults`

```text
builder.set_node_defaults(
    *,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    error_handler: StateNode | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
) -> Self
```

| Parameter | Applies to |
|---|---|
| `retry_policy` | All nodes, including error-handler nodes |
| `cache_policy` | Regular nodes only (handler results are never cached) |
| `error_handler` | Regular nodes that have no handler of their own |
| `timeout` | All nodes, including error-handler nodes |

Defaults are resolved at compile time and are not inherited by subgraphs. Here are all four together:

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import RetryPolicy, CachePolicy
from langgraph.cache.memory import InMemoryCache


class State(TypedDict, total=False):
    query: str
    answer: str
    last_error: str


calls = {"fetch": 0}


async def fetch(state: State) -> dict:
    calls["fetch"] += 1
    if calls["fetch"] < 3:
        raise ConnectionError("upstream reset")      # retried by the policy
    return {"answer": f"result for {state['query']}"}


async def summarise(state: State) -> dict:
    raise ValueError("bad payload")                  # not retried; goes to the handler


async def on_error(state: State, error: NodeError) -> dict:   # async: the default timeout applies to it
    return {"last_error": f"{error.node}: {type(error.error).__name__}: {error.error}"}


graph = (
    StateGraph(State)
    .set_node_defaults(timeout=30.0, error_handler=on_error)       # every node
    .add_node("fetch", fetch,
              retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01),
              cache_policy=CachePolicy(ttl=300))
    .add_node("summarise", summarise)
    .add_edge(START, "fetch")
    .add_edge("fetch", "summarise")
    .add_edge("summarise", END)
    .compile(cache=InMemoryCache())
)

print(asyncio.run(graph.ainvoke({"query": "weather"})))
# {'query': 'weather', 'answer': 'result for weather', 'last_error': 'summarise: ValueError: bad payload'}
print("fetch attempts:", calls["fetch"])   # 3
```

The same policies apply to Functional API tasks: `@task(retry_policy=..., cache_policy=..., timeout=...)`. See the [Functional API reference](/langgraph-guide/python/reference-functional-api/).

## `add_edge`

```python
builder.add_edge("a", "b")                # single
builder.add_edge(["a", "b"], "c")         # waits for ALL of a, b (barrier edge)
builder.add_edge(START, "a")              # entry point
builder.add_edge("last", END)             # finish point
```

Raises `ValueError` if the start is `END`, the end is `START`, or a named node is missing.

## `add_conditional_edges`

```text
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
- `END` (the string `"__end__"`) → stop. The literal string `"END"` is treated as a node name and fails.

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

```text
graph = builder.compile(
    checkpointer=None,       # BaseCheckpointSaver | True | False | None
    *,
    cache=None,              # BaseCache, needed for CachePolicy
    store=None,              # BaseStore for long-term memory
    interrupt_before=None,   # list[str] | "*" | None
    interrupt_after=None,    # list[str] | "*" | None
    debug=False,
    name=None,
    transformers=None,       # stream transformers for the v3 streaming API
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
| `stream(...)` | Yield per-step events (see the [Streaming modes reference](/langgraph-guide/python/reference-streaming-modes/)). |
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
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Ctx:
    user_id: str


class State(TypedDict, total=False):
    greeting: str


def node(state: State, runtime: Runtime[Ctx]) -> dict:
    uid = runtime.context.user_id
    item = runtime.store.get(("users",), uid) if runtime.store else None
    name = item.value["name"] if item else uid
    return {"greeting": f"Hello, {name}"}


store = InMemoryStore()
store.put(("users",), "alice", {"name": "Alice"})
graph = (
    StateGraph(State, context_schema=Ctx)
    .add_node(node)
    .add_edge(START, "node")
    .compile(store=store)
)
print(graph.invoke({}, context=Ctx(user_id="alice")))   # {'greeting': 'Hello, Alice'}
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

`RunControl` can be pre-created and passed to `graph.invoke(..., control=control)` so that external code (e.g. a signal handler) can call `control.request_drain()` before the graph starts. If you do not pass one, the Pregel executor creates it internally. Key properties:

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
builder.add_edge("worker", "aggregate")   # aggregate runs once, after every worker in the step has finished
```

### 2. Deferred finalization

Retry, cache and timeout patterns are in [Node policies](#node-policies).

```python
builder.add_node("summarize", summarize_fn, defer=True)
builder.add_edge(START, "research")
builder.add_edge("research", "write")
builder.add_edge("write", "summarize")
# `summarize` runs last, after every other task in the run has drained.
```

### 3. Narrow node input with `input_schema`

```python
class QueryOnly(TypedDict):
    query: str

def classify(state: QueryOnly) -> dict:
    return {"category": "billing" if "bill" in state["query"] else "other"}

builder.add_node("classify", classify, input_schema=QueryOnly)
```

The node cannot read unrelated channels and stays cheap to trace.

### 4. Reading `ExecutionInfo` for per-node tracing

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

### 5. Pydantic state schema

`StateGraph` accepts a Pydantic `BaseModel` as its state schema. Since v1.1, `invoke()` coerces dict inputs to the model before nodes run, so nodes receive a typed instance.

```python
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END


class DocState(BaseModel):
    text: str = ""
    word_count: int = 0
    tags: Annotated[list[str], operator.add] = Field(default_factory=list)


def count_words(state: DocState) -> dict:
    # `state` is a real DocState instance — full type safety + validation
    count = len(state.text.split())
    return {"word_count": count}


def tag_document(state: DocState) -> dict:
    tag = "long" if state.word_count > 50 else "short"
    return {"tags": [tag]}   # operator.add appends to existing list


builder = StateGraph(DocState)
builder.add_node("count", count_words)
builder.add_node("tag", tag_document)
builder.add_edge(START, "count")
builder.add_edge("count", "tag")
builder.add_edge("tag", END)

graph = builder.compile()

# Dict input is coerced to DocState before nodes run
result = graph.invoke({"text": "LangGraph builds stateful multi-actor apps with LLMs."})
print(result)  # {'text': '...', 'word_count': 8, 'tags': ['short']}
```

### 6. Narrowing I/O with `input_schema` and `output_schema`

`input_schema` restricts what callers must pass; `output_schema` restricts what the graph returns. Internal channels that are not in `output_schema` are stripped from the final result.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class FullState(TypedDict):
    raw_text: str
    cleaned_text: str
    word_count: int
    _internal_flag: bool   # should never be visible externally


class UserInput(TypedDict):
    raw_text: str          # callers only need to provide raw text


class UserOutput(TypedDict):
    cleaned_text: str      # callers only see the cleaned result
    word_count: int


def clean(state: FullState) -> dict:
    return {
        "cleaned_text": state["raw_text"].strip().lower(),
        "_internal_flag": True,
    }


def count(state: FullState) -> dict:
    return {"word_count": len(state["cleaned_text"].split())}


builder = StateGraph(FullState, input_schema=UserInput, output_schema=UserOutput)
builder.add_node("clean", clean)
builder.add_node("count", count)
builder.add_edge(START, "clean")
builder.add_edge("clean", "count")
builder.add_edge("count", END)

graph = builder.compile()

# Caller provides only UserInput fields
result = graph.invoke({"raw_text": "  Hello World  "})
print(result)   # {'cleaned_text': 'hello world', 'word_count': 2}
# '_internal_flag' is NOT present — stripped by output_schema
```

### 7. Chaining nodes with `add_sequence`

`add_sequence` wires a list of nodes sequentially: node[0] → node[1] → … → node[n]. It uses each callable's `__name__` as the node name; pass `(name, fn)` tuples for explicit names.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class Pipeline(TypedDict):
    text: str
    tokens: list[str]
    normalized: list[str]
    result: str


def tokenize(state: Pipeline) -> dict:
    return {"tokens": state["text"].split()}


def normalize(state: Pipeline) -> dict:
    return {"normalized": [t.lower() for t in state["tokens"]]}


def join_result(state: Pipeline) -> dict:
    return {"result": " ".join(state["normalized"])}


builder = StateGraph(Pipeline)
builder.add_edge(START, "tokenize")
# add_sequence registers the three nodes and connects them in order
builder.add_sequence([tokenize, normalize, join_result])
builder.add_edge("join_result", END)

graph = builder.compile()
print(graph.invoke({"text": "Hello World from LangGraph", "tokens": [], "normalized": [], "result": ""}))
# {'text': 'Hello World from LangGraph', 'tokens': [...], 'normalized': [...], 'result': 'hello world from langgraph'}
```

You can also mix plain callables and `(name, fn)` tuples:

```python
builder.add_sequence([
    ("load", load_fn),          # explicit name
    transform_fn,               # uses transform_fn.__name__
    ("save", save_fn),          # explicit name
])
```

### 8. Streaming custom events with `StreamWriter`

Nodes can push arbitrary values to `stream_mode="custom"` by declaring a `writer: StreamWriter` parameter:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    items: list[str]
    processed: list[str]


def batch_process(state: State, writer: StreamWriter) -> dict:
    """Process items one at a time, streaming progress to the caller."""
    results = []
    for i, item in enumerate(state["items"]):
        # Emit a progress event — only visible with stream_mode="custom"
        writer({"progress": i + 1, "total": len(state["items"]), "item": item})
        results.append(item.upper())
    return {"processed": results}


builder = StateGraph(State)
builder.add_node("process", batch_process)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "stream-1"}}

# Receive both state updates and custom events
for mode, data in graph.stream(
    {"items": ["a", "b", "c"], "processed": []},
    cfg,
    stream_mode=["updates", "custom"],
):
    print(f"{mode}: {data}")
# custom: {'progress': 1, 'total': 3, 'item': 'a'}
# custom: {'progress': 2, 'total': 3, 'item': 'b'}
# custom: {'progress': 3, 'total': 3, 'item': 'c'}
# updates: {'process': {'processed': ['A', 'B', 'C']}}
```

## Gotchas

- **Two writes, no reducer, one super-step → `InvalidUpdateError`.** Either add a reducer or stagger the writes with edges.
- **`checkpointer=None`** disables every feature that depends on persistence: `interrupt()`, `get_state`, `update_state`, `get_state_history`, time travel, thread-scoped memory. Use `InMemorySaver()` while developing.
- **`config_schema=` is deprecated**, but still accepted. Rename to `context_schema=` before v2.0.
- **`AgentState` / `AgentStatePydantic`** in `langgraph.prebuilt` are **deprecated** as of v1.0 — their top-level re-export was removed from `langgraph.prebuilt.__init__`, but both are still importable from `langgraph.prebuilt.chat_agent_executor`; scheduled for removal in v2.0.0. Use `MessagesState` (built-in) or a custom `TypedDict` with an `add_messages`-annotated field for new code.
- **Root graphs cannot have `checkpointer=True`.** That value is only for subgraphs inheriting from the parent.
- **`destinations=` does not route**, but its targets must exist in the same graph: `compile()` raises `Found edge ending at unknown node` otherwise.
- **`TimeoutPolicy` only works on async nodes.** `add_node` accepts a timeout on a sync node; `compile()` then raises `ValueError`. Convert the node to `async` and use `asyncio.to_thread` for blocking work.
- **`runtime.heartbeat()` is a no-op without `refresh_on="heartbeat"`.** Under `refresh_on="auto"` (default), progress is detected automatically from callbacks and stream writes; calling `heartbeat()` is still valid but redundant.
- **`runtime.execution_info` is `None` briefly during startup.** Don't access it in lifecycle hooks that run before task preparation.
- **`RunControl` can be pre-created and injected.** Pass `control=RunControl()` to `graph.invoke()` so external code (e.g. a SIGTERM handler) can call `control.request_drain()` before the run begins. If omitted, the executor creates one internally.

## Breaking changes

| Version | Change |
|---|---|
| 1.2 | `TimeoutPolicy` dataclass introduced with `run_timeout`, `idle_timeout`, `refresh_on`. `Runtime.heartbeat()` added. `RunControl` and cooperative draining added (`runtime.control`, `runtime.drain_requested`, `runtime.drain_reason`). `ExecutionInfo` extended with `checkpoint_ns` and `task_id` fields. |
| 1.1 | `invoke()`/`stream()` coerce input dicts into the declared state schema for Pydantic/dataclass. V2 stream mode emits typed `StreamPart` dicts. Python 3.9 dropped. |
| 1.0 | `AgentState`, `AgentStatePydantic` **deprecated** in `langgraph.prebuilt` (top-level re-export removed from `__init__`; still importable from `langgraph.prebuilt.chat_agent_executor`; scheduled removal in v2.0.0) — use `MessagesState` or a custom TypedDict for new code. `create_react_agent` is **deprecated** per official LangGraph v1 migration docs in favour of `langchain.agents.create_agent` (separate `langchain` package; `pip install langchain`). It remains in `langgraph.prebuilt` for backwards compatibility. `ns`, `when`, `resumable`, `interrupt_id` removed from `Interrupt` (in v0.6). |
| 0.6 | `config_schema` on `StateGraph` deprecated; use `context_schema`. `Runtime[Ctx]` replaces ad-hoc `config["configurable"]` usage for run context. |
| 0.5 | `input` / `output` kwargs on `StateGraph.__init__` deprecated; use `input_schema` / `output_schema`. |

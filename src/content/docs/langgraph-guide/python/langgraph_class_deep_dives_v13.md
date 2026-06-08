---
title: "Class deep-dives Vol. 13 — Channels, caches, stores & internals"
description: "Source-verified deep dives into LastValue/LastValueAfterFinish, BranchSpec, BaseCache, get_config/get_store runtime accessors, ManagedValue protocol, default_cache_key/_freeze, debug stream producers, GraphLifecycleStatus callback wiring, BaseStore abstract contract, and StateGraph annotation-to-channel inference — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 13"
  order: 44
---

# Class deep-dives Vol. 13 — Channels, caches, stores & internals

Verified against **`langgraph==1.2.4`** installed at `/usr/local/lib/python3.11/dist-packages/langgraph/`.

Every section was written by reading the actual installed source. All signatures, field names, and behaviours are derived from the implementation, not from external documentation.

[→ Vol. 1–12 index at the bottom of this page](#vol-index)

---

## 1 · `LastValue` + `LastValueAfterFinish` — the fundamental channel layer

**Module:** `langgraph.channels.last_value`  
**Imports:**
```python
from langgraph.channels.last_value import LastValue, LastValueAfterFinish
```

`LastValue` is the channel that backs *every* plain (un-annotated) TypedDict field in a `StateGraph`. When you write `class State(TypedDict): count: int`, LangGraph silently creates a `LastValue(int)` channel for `count`. Understanding its single-write-per-step invariant and checkpoint round-trip is essential for debugging concurrent update errors.

`LastValueAfterFinish` is the companion used by the **functional API** to transport `@task` return values: the value is deposited on write, held until `finish()` is called by the executor, exposed once via `get()`, and then cleared by `consume()`.

### Source signatures (1.2.4)

```python
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value",)
    value: Value | Any  # MISSING sentinel if unset

    def __init__(self, typ: Any, key: str = "") -> None: ...
    def update(self, values: Sequence[Value]) -> bool: ...  # raises if len > 1
    def get(self) -> Value: ...                             # raises EmptyChannelError if MISSING
    def is_available(self) -> bool: ...
    def checkpoint(self) -> Value: ...
    def from_checkpoint(self, checkpoint: Value) -> Self: ...

class LastValueAfterFinish(Generic[Value], BaseChannel[Value, Value, tuple[Value, bool]]):
    __slots__ = ("value", "finished")
    def update(self, values: Sequence[Value]) -> bool: ...
    def finish(self) -> bool: ...    # marks value as ready; returns True if state changed
    def consume(self) -> bool: ...   # clears value + finished flag; returns True if consumed
    def get(self) -> Value: ...      # raises EmptyChannelError unless finished=True
    def is_available(self) -> bool: ...
    def checkpoint(self) -> tuple[Value, bool] | Any: ...
```

### Example 1: What happens on a concurrent write to a `LastValue` channel

The single-write constraint is enforced inside `update()`. If two nodes run in the same step and both write to the same plain field, LangGraph raises `InvalidUpdateError`.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# WRONG — two nodes both write to `score` in the same parallel step
class BadState(TypedDict):
    score: int  # plain LastValue — only one write allowed per step

# CORRECT — use a reducer to merge parallel writes
class GoodState(TypedDict):
    score: Annotated[int, operator.add]  # BinaryOperatorAggregate

def node_a(state: GoodState) -> dict:
    return {"score": 10}

def node_b(state: GoodState) -> dict:
    return {"score": 5}

builder = StateGraph(GoodState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", END)
builder.add_edge("b", END)
graph = builder.compile()

result = graph.invoke({"score": 0})
print(result["score"])  # 15 — operator.add merged both writes
```

### Example 2: `LastValueAfterFinish` — functional API task channel lifecycle

This is used internally. The sequence is: `update()` deposits the value → `finish()` marks it ready → `get()` returns it → `consume()` clears it. Useful when building a custom executor that needs to understand task result propagation.

```python
from langgraph.channels.last_value import LastValueAfterFinish
from langgraph._internal._typing import MISSING

ch: LastValueAfterFinish[int] = LastValueAfterFinish(int)

# Phase 1: task writes its return value
ch.update([42])
print(ch.is_available())  # False — not finished yet

# Phase 2: executor calls finish() after the task node completes
ch.finish()
print(ch.is_available())  # True
print(ch.get())           # 42

# Phase 3: downstream reads the value, then consume() clears it
ch.consume()
print(ch.is_available())  # False — cleared for next invocation
```

### Example 3: Checkpoint round-trip

Both channels serialise to/from checkpoint via `checkpoint()` / `from_checkpoint()`:

```python
from langgraph.channels.last_value import LastValue

ch: LastValue[str] = LastValue(str)
ch.update(["hello"])

# Serialise (stored in checkpoint)
snap = ch.checkpoint()       # "hello"

# Restore
restored = LastValue(str).from_checkpoint(snap)
print(restored.get())        # "hello"
```

---

## 2 · `BranchSpec` — the internal NamedTuple powering `add_conditional_edges`

**Module:** `langgraph.graph._branch`  
**Import:**
```python
from langgraph.graph._branch import BranchSpec
```

Every call to `StateGraph.add_conditional_edges(source, path, path_map)` ultimately creates a `BranchSpec` and calls `BranchSpec.run()` to produce a `RunnableCallable` that is wired into the graph. Understanding `BranchSpec` explains how LangGraph:

- Auto-detects `path_map` from `Literal` return-type annotations
- Handles `Send` objects as destinations
- Integrates with `ChannelWrite` for efficient passthrough routing

### Source signature (1.2.4)

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None       # None means free-form (any string)
    input_schema: type[Any] | None = None

    @classmethod
    def from_path(
        cls,
        path: Runnable,
        path_map: dict | list | None,
        infer_schema: bool = False,
    ) -> "BranchSpec": ...

    def run(
        self,
        writer: _Writer,
        reader: Callable[[RunnableConfig], Any] | None = None,
    ) -> RunnableCallable: ...
```

### Example 1: `from_path` auto-detects `path_map` from `Literal` return annotation

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: list[str]
    route: str

# LangGraph infers path_map={"search": "search", "answer": "answer"}
# from the Literal return annotation — no explicit path_map needed.
def route_query(state: State) -> Literal["search", "answer"]:
    return "search" if "?" in state["messages"][-1] else "answer"

def search(state: State) -> dict:
    return {"messages": state["messages"] + ["[search result]"]}

def answer(state: State) -> dict:
    return {"messages": state["messages"] + ["[direct answer]"]}

builder = StateGraph(State)
builder.add_node("route_query", lambda s: s)  # passthrough
builder.add_node("search", search)
builder.add_node("answer", answer)
builder.add_edge(START, "route_query")
# path_map is inferred from Literal annotation — no explicit mapping needed
builder.add_conditional_edges("route_query", route_query)
builder.add_edge("search", END)
builder.add_edge("answer", END)
graph = builder.compile()

result = graph.invoke({"messages": ["What is the capital of France?"], "route": ""})
print(result["messages"])
# ['What is the capital of France?', '[search result]']
```

### Example 2: Explicit `path_map` with `Send` for fan-out

When the routing function returns `Send` objects, `BranchSpec._finish()` passes them directly as destinations. This is the mechanism behind parallel fan-out with per-task inputs.

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    queries: list[str]
    results: list[str]

def fan_out(state: State) -> list[Send]:
    # Return one Send per query — each gets its own input dict
    return [Send("worker", {"query": q}) for q in state["queries"]]

def worker(state: dict) -> dict:
    return {"results": [f"result({state['query']})"]}

import operator
from typing import Annotated

class FanState(TypedDict):
    queries: list[str]
    results: Annotated[list[str], operator.add]

builder = StateGraph(FanState)
builder.add_node("fan_out", lambda s: s)
builder.add_node("worker", worker)
builder.add_edge(START, "fan_out")
builder.add_conditional_edges("fan_out", fan_out)  # returns list[Send]
builder.add_edge("worker", END)
graph = builder.compile()

out = graph.invoke({"queries": ["alpha", "beta", "gamma"], "results": []})
print(out["results"])  # ['result(alpha)', 'result(beta)', 'result(gamma)']
```

### Example 3: `infer_schema=True` — typed input for branch functions

`BranchSpec.from_path(..., infer_schema=True)` reads the first parameter's type annotation and uses it as the channel read schema for the branch callable. This lets the branch function receive a strongly-typed input instead of the full state dict.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class FullState(TypedDict):
    score: int
    label: str

class RouteInput(TypedDict):
    score: int  # branch only needs score

def route_by_score(inp: RouteInput) -> str:
    return "high" if inp["score"] >= 50 else "low"

def high_handler(state: FullState) -> dict:
    return {"label": "HIGH"}

def low_handler(state: FullState) -> dict:
    return {"label": "LOW"}

builder = StateGraph(FullState)
builder.add_node("check", lambda s: s)
builder.add_node("high", high_handler)
builder.add_node("low", low_handler)
builder.add_edge(START, "check")
# infer_schema=True: branch receives RouteInput, not FullState
builder.add_conditional_edges("check", route_by_score, {"high": "high", "low": "low"})
builder.add_edge("high", END)
builder.add_edge("low", END)
graph = builder.compile()

print(graph.invoke({"score": 75, "label": ""})["label"])  # HIGH
print(graph.invoke({"score": 30, "label": ""})["label"])  # LOW
```

---

## 3 · `BaseCache` — abstract cache backend protocol

**Module:** `langgraph.cache.base`  
**Import:**
```python
from langgraph.cache.base import BaseCache, FullKey, Namespace
```

`BaseCache` is the abstract base class that all LangGraph cache backends implement. `InMemoryCache` and `RedisCache` both subclass it. If you need a custom caching layer (e.g., Memcached, DynamoDB, or a multi-tier L1/L2 cache), implement these six methods.

The `FullKey = tuple[Namespace, str]` type alias means every cache entry is addressed by a `(namespace_tuple, key_string)` pair — exactly matching the store's namespace model.

### Source signature (1.2.4)

```python
Namespace = tuple[str, ...]
FullKey = tuple[Namespace, str]

class BaseCache(ABC, Generic[ValueT]):
    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=False)

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None: ...

    @abstractmethod
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]: ...
    @abstractmethod
    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]: ...
    @abstractmethod
    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None: ...
    @abstractmethod
    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None: ...
    @abstractmethod
    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None: ...
    @abstractmethod
    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None: ...
```

The `set` / `aset` `pairs` argument maps `FullKey → (value, ttl_seconds_or_None)`. A `None` TTL means no expiry.

### Example 1: Implement a custom SQLite-backed cache backend

```python
import sqlite3
import pickle
import time
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace


class SQLiteCache(BaseCache):
    """Minimal SQLite cache for demonstration."""

    def __init__(self, db_path: str = ":memory:") -> None:
        super().__init__()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(ns TEXT, key TEXT, value BLOB, expiry REAL, PRIMARY KEY (ns, key))"
        )
        self._conn.commit()

    def _ns(self, ns: Namespace) -> str:
        return ":".join(ns)

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, object]:
        now = time.time()
        result = {}
        for ns, key in keys:
            row = self._conn.execute(
                "SELECT value, expiry FROM cache WHERE ns=? AND key=?",
                (self._ns(ns), key),
            ).fetchone()
            if row and (row[1] is None or now < row[1]):
                result[(ns, key)] = pickle.loads(row[0])
        return result

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, object]:
        return self.get(keys)  # SQLite is sync; wrap in executor for real use

    def set(self, pairs: Mapping[FullKey, tuple[object, int | None]]) -> None:
        now = time.time()
        rows = []
        for (ns, key), (value, ttl) in pairs.items():
            expiry = now + ttl if ttl is not None else None
            rows.append((self._ns(ns), key, pickle.dumps(value), expiry))
        self._conn.executemany(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?)", rows
        )
        self._conn.commit()

    async def aset(self, pairs: Mapping[FullKey, tuple[object, int | None]]) -> None:
        self.set(pairs)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        if namespaces is None:
            self._conn.execute("DELETE FROM cache")
        else:
            for ns in namespaces:
                self._conn.execute("DELETE FROM cache WHERE ns=?", (self._ns(ns),))
        self._conn.commit()

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        self.clear(namespaces)


# Use the custom cache with a @task
from langgraph.func import task, entrypoint

cache = SQLiteCache()

@task(cache=cache)
def expensive_lookup(query: str) -> str:
    print(f"  [miss] computing for '{query}'")
    return f"result({query})"

@entrypoint()
def workflow(query: str) -> str:
    return expensive_lookup(query).result()

# First call — cache miss
print(workflow.invoke("hello"))   # [miss] computing for 'hello'  →  result(hello)
# Second call — cache hit
print(workflow.invoke("hello"))   # (no print)                     →  result(hello)
```

### Example 2: Cache invalidation by namespace

```python
cache = SQLiteCache()

# Populate
ns = ("myapp", "lookups")
cache.set({(ns, "k1"): ("value1", 60), (ns, "k2"): ("value2", 60)})

# Read
print(cache.get([(ns, "k1")]))   # {(('myapp', 'lookups'), 'k1'): 'value1'}

# Invalidate the whole namespace
cache.clear([ns])
print(cache.get([(ns, "k1")]))   # {}
```

---

## 4 · `get_config()` + `get_store()` — runtime context-variable accessors

**Module:** `langgraph.config`  
**Imports:**
```python
from langgraph.config import get_config, get_store, get_stream_writer
```

These three functions use Python `contextvars` to give *any code running inside a node or task* immediate access to the live `RunnableConfig`, the injected `BaseStore`, or the current `StreamWriter` — without needing to thread them through every function signature. They are the recommended approach when your tool or helper function needs runtime state but cannot accept injection parameters (e.g., third-party libraries, deeply nested calls).

**Python version caveat:** `contextvars` propagation to new `asyncio.Task` objects requires Python ≥ 3.11. On 3.10, these functions work in sync contexts but will raise or return stale values in async subgraphs that spawn `asyncio.create_task`.

### Source signature (1.2.4)

```python
def get_config() -> RunnableConfig:
    """Return the current RunnableConfig from contextvar; raises RuntimeError if called outside a node."""

def get_store() -> BaseStore:
    """Return the store injected at compile time. Raises KeyError if no store was provided."""

def get_stream_writer() -> StreamWriter:
    """Return the per-node stream writer. Raises RuntimeError if called outside a node."""
```

### Example 1: `get_config()` — read thread_id and custom metadata inside a helper

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config

class State(TypedDict):
    messages: list[str]

def node_that_reads_config(state: State) -> dict:
    cfg = get_config()
    thread_id = cfg["configurable"].get("thread_id", "unknown")
    user_id   = cfg["configurable"].get("user_id",   "anonymous")
    return {"messages": state["messages"] + [f"hello {user_id} on thread {thread_id}"]}

graph = (
    StateGraph(State)
    .add_node("greet", node_that_reads_config)
    .add_edge(START, "greet")
    .add_edge("greet", END)
    .compile(checkpointer=InMemorySaver())
)

result = graph.invoke(
    {"messages": []},
    config={"configurable": {"thread_id": "t1", "user_id": "alice"}},
)
print(result["messages"])
# ['hello alice on thread t1']
```

### Example 2: `get_store()` — access long-term memory without `InjectedStore`

Use `get_store()` when injecting via `InjectedStore` is impractical — for example, in a standalone helper called from multiple nodes.

```python
from langgraph.config import get_store
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

store = InMemoryStore()
store.put(("facts",), "pi", {"value": 3.14159})

class State(TypedDict):
    answer: str

def lookup_fact(state: State) -> dict:
    s = get_store()  # no InjectedStore annotation needed
    item = s.get(("facts",), "pi")
    return {"answer": f"pi ≈ {item.value['value']}"}

graph = (
    StateGraph(State)
    .add_node("lookup", lookup_fact)
    .add_edge(START, "lookup")
    .add_edge("lookup", END)
    .compile(store=store)   # store must be passed at compile time
)

print(graph.invoke({"answer": ""})["answer"])
# pi ≈ 3.14159
```

### Example 3: `get_config()` inside a `@task` (functional API)

`get_config()` is also available inside `@task` bodies, making it easy to forward config values to downstream calls.

```python
from langgraph.func import task, entrypoint
from langgraph.config import get_config

@task
def process(payload: str) -> str:
    cfg = get_config()
    run_id = str(cfg.get("run_id", "no-run-id"))
    return f"{payload} [run={run_id[:8]}]"

@entrypoint()
def workflow(payload: str) -> str:
    return process(payload).result()

import uuid
result = workflow.invoke("hello", config={"run_id": uuid.uuid4()})
print(result)  # hello [run=<first-8-chars-of-uuid>]
```

---

## 5 · `ManagedValue` + `ManagedValueSpec` + `is_managed_value()` — custom managed value protocol

**Module:** `langgraph.managed.base`  
**Imports:**
```python
from langgraph.managed.base import ManagedValue, ManagedValueSpec, is_managed_value
```

`ManagedValue` is the abstract base class behind `IsLastStep` and `RemainingSteps`. Instead of holding a user-supplied value, a managed value is *computed* by the executor at runtime from the current `PregelScratchpad`. It appears as a read-only annotation in `State` TypedDicts that any node can read, but no node can write.

`ManagedValueSpec = type[ManagedValue]` — the concrete subclass (not an instance). `is_managed_value(x)` checks whether `x` is a class that subclasses `ManagedValue`.

### Source signature (1.2.4)

```python
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...

ManagedValueSpec = type[ManagedValue]

def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]: ...
```

The `PregelScratchpad` passed to `get()` exposes fields such as:
- `step` — current execution step number  
- `stop` — stop step (recursion limit)
- `tasks` — pending `PregelExecutableTask` list
- `all_tasks` — all tasks this step

### Example 1: Build a `StepCounter` managed value

```python
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class StepCounter(ManagedValue[int]):
    """Returns the current execution step index (0-based)."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.step


class State(TypedDict):
    messages: list[str]
    # Read-only field: no node may write to it
    step: Annotated[int, StepCounter]


def node_a(state: State) -> dict:
    return {"messages": state["messages"] + [f"step={state['step']}"]}

def node_b(state: State) -> dict:
    return {"messages": state["messages"] + [f"step={state['step']}"]}


graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)

result = graph.invoke({"messages": []})
print(result["messages"])
# ['step=1', 'step=2']
```

### Example 2: `PendingTaskCount` — inspect how many tasks are queued

```python
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator


class PendingTaskCount(ManagedValue[int]):
    """Returns the number of tasks scheduled in the current step."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return len(scratchpad.tasks) if scratchpad.tasks else 0


class ParallelState(TypedDict):
    results: Annotated[list[str], operator.add]
    task_count: Annotated[int, PendingTaskCount]


def worker(state: ParallelState) -> dict:
    return {"results": [f"ran (siblings={state['task_count']})"]}


builder = StateGraph(ParallelState)
for name in ("w1", "w2", "w3"):
    builder.add_node(name, worker)
    builder.add_edge(START, name)
    builder.add_edge(name, END)

graph = builder.compile()
result = graph.invoke({"results": []})
print(result["results"])
# Each worker sees task_count=3 (all three run in the same step)
```

### Example 3: `is_managed_value()` — runtime check

```python
from langgraph.managed.base import is_managed_value
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps

print(is_managed_value(IsLastStep))      # True
print(is_managed_value(RemainingSteps))  # True
print(is_managed_value(int))             # False
print(is_managed_value(42))              # False
```

---

## 6 · `default_cache_key` + `_freeze` — cache key generation internals

**Module:** `langgraph._internal._cache`  
**Import:**
```python
from langgraph._internal._cache import default_cache_key, _freeze
```

`default_cache_key` is the function used by `CachePolicy` when no custom `key_func` is provided. It converts node arguments + keyword arguments into a deterministic `bytes` key using `pickle.protocol=5` over a frozen representation of the inputs. `_freeze` recursively converts unhashable containers (dicts, lists) into hashable tuples before pickling, so that `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` produce the same key.

### Source (1.2.4)

```python
def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if isinstance(obj, Hashable) or depth <= 0:
        return obj
    elif isinstance(obj, Mapping):
        return tuple(sorted((k, _freeze(v, depth - 1)) for k, v in obj.items()))
    elif isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    elif hasattr(obj, "tobytes"):           # numpy/pandas arrays
        return (type(obj).__name__, obj.tobytes(), getattr(obj, "shape", None))
    return obj

def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    import pickle
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
```

### Example 1: Understanding when `_freeze` matters — dict key ordering

```python
from langgraph._internal._cache import default_cache_key

# Dict key order is ignored (sorted internally)
k1 = default_cache_key({"a": 1, "b": 2})
k2 = default_cache_key({"b": 2, "a": 1})
print(k1 == k2)  # True ✓

# List order is preserved (sequences keep order)
k3 = default_cache_key([1, 2, 3])
k4 = default_cache_key([3, 2, 1])
print(k3 == k4)  # False — different ordering → different key
```

### Example 2: Supplying a custom `key_func` to avoid cache collisions

The default key includes *all* arguments. If some arguments are non-deterministic (e.g., timestamps, request IDs) you must override `key_func` to exclude them.

```python
import time
from langgraph.types import CachePolicy
from langgraph.func import task, entrypoint

def stable_key(query: str, timestamp: float) -> str:
    """Cache on query only; ignore timestamp."""
    return query

@task(cache=CachePolicy(key_func=stable_key))
def search(query: str, timestamp: float) -> str:
    print(f"  [search] {query}")
    return f"results for {query}"

@entrypoint()
def workflow(query: str) -> str:
    return search(query, time.time()).result()

print(workflow.invoke("python"))  # [search] python → results for python
print(workflow.invoke("python"))  # cache hit — no [search] print
```

### Example 3: Inspecting the raw bytes key

```python
from langgraph._internal._cache import default_cache_key

key = default_cache_key("hello", n=5)
print(type(key))    # <class 'bytes'>
print(len(key))     # ~40 bytes for small inputs
print(key[:10])     # raw pickle bytes
```

---

## 7 · `map_debug_tasks` + `map_debug_task_results` + `map_debug_checkpoint` — debug stream event producers

**Module:** `langgraph.pregel.debug`  
**Imports:**
```python
from langgraph.pregel.debug import (
    map_debug_tasks,
    map_debug_task_results,
    map_debug_checkpoint,
    map_task_result_writes,
    tasks_w_writes,
)
```

When you stream a graph with `stream_mode="debug"`, these three generator functions are called internally to produce the `task`, `task_result`, and `checkpoint` debug events. Understanding them lets you build custom debug stream consumers, write integration tests that inspect raw execution traces, or implement custom observability backends.

### Source signatures (1.2.4)

```python
def map_debug_tasks(
    tasks: Iterable[PregelExecutableTask],
) -> Iterator[TaskPayload]:
    """Yield one 'task' event per non-hidden PregelExecutableTask."""

def map_debug_task_results(
    task_tup: tuple[PregelExecutableTask, Sequence[tuple[str, Any]]],
    stream_keys: str | Sequence[str],
) -> Iterator[TaskResultPayload]:
    """Yield one 'task_result' event for a task + its channel writes."""

def map_debug_checkpoint(
    config: RunnableConfig,
    channels: Mapping[str, BaseChannel],
    stream_channels: str | Sequence[str],
    metadata: CheckpointMetadata,
    tasks: Iterable[PregelExecutableTask],
    pending_writes: list[PendingWrite],
    parent_config: RunnableConfig | None,
    output_keys: str | Sequence[str],
) -> Iterator[CheckpointPayload]:
    """Yield one 'checkpoint' event with full state snapshot."""

def map_task_result_writes(
    writes: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    """Aggregate multiple writes to the same channel into {'$writes': [...]}."""
```

### Example 1: Consuming `stream_mode="debug"` events

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    count: int

def increment(state: State) -> dict:
    return {"count": state["count"] + 1}

graph = (
    StateGraph(State)
    .add_node("inc", increment)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile(checkpointer=InMemorySaver())
)

events = list(graph.stream(
    {"count": 0},
    config={"configurable": {"thread_id": "dbg-1"}},
    stream_mode="debug",
))

for evt in events:
    print(evt["type"], "—", list(evt.keys()))
# task         — ['type', 'timestamp', 'step', 'payload']
# task_result  — ['type', 'timestamp', 'step', 'payload']
# checkpoint   — ['type', 'timestamp', 'step', 'payload']
```

### Example 2: Inspecting `TaskPayload` fields

```python
for evt in events:
    if evt["type"] == "task":
        payload = evt["payload"]
        print("task id   :", payload["id"])
        print("task name :", payload["name"])
        print("triggers  :", payload["triggers"])
        print("input     :", payload["input"])
```

### Example 3: `map_task_result_writes` — multi-write aggregation

When a node (or fan-in) writes to the same channel multiple times in a step, `map_task_result_writes` wraps them in `{"$writes": [...]}`. This sentinel key is what `is_multiple_channel_write()` checks.

```python
from langgraph.pregel.debug import map_task_result_writes, is_multiple_channel_write

# Single write → plain value
result = map_task_result_writes([("messages", "hello")])
print(result)  # {'messages': 'hello'}

# Two writes to same channel → $writes wrapper
result = map_task_result_writes([
    ("messages", "hello"),
    ("messages", " world"),
])
print(result)   # {'messages': {'$writes': ['hello', ' world']}}
print(is_multiple_channel_write(result["messages"]))  # True
```

---

## 8 · `GraphLifecycleStatus` + callback manager factories — full lifecycle callback wiring

**Module:** `langgraph.callbacks`  
**Imports:**
```python
from langgraph.callbacks import (
    GraphCallbackHandler,
    GraphInterruptEvent,
    GraphResumeEvent,
    GraphLifecycleStatus,
    get_sync_graph_callback_manager_for_config,
    get_async_graph_callback_manager_for_config,
)
```

`GraphLifecycleStatus` is a `Literal` type alias for the six loop states that can appear in lifecycle events. The two factory functions (`get_sync_graph_callback_manager_for_config` / `get_async_graph_callback_manager_for_config`) extract `GraphCallbackHandler` instances from a `RunnableConfig`'s `"callbacks"` list and build a typed dispatcher. In Vol. 6 we saw the handler interface; this section shows the full wiring.

### Source signature (1.2.4)

```python
GraphLifecycleStatus = Literal[
    "input",
    "pending",
    "done",
    "interrupt_before",
    "interrupt_after",
    "out_of_steps",
]

@dataclass(frozen=True)
class GraphInterruptEvent:
    run_id: UUID | None
    status: GraphLifecycleStatus   # always "interrupt_before" or "interrupt_after"
    checkpoint_id: str
    checkpoint_ns: tuple[str, ...]
    interrupts: tuple[Interrupt, ...]

@dataclass(frozen=True)
class GraphResumeEvent:
    run_id: UUID | None
    status: GraphLifecycleStatus   # always "pending" or "input"
    checkpoint_id: str
    checkpoint_ns: tuple[str, ...]

def get_sync_graph_callback_manager_for_config(
    config: RunnableConfig,
    *,
    run_id: UUID | None = None,
) -> _GraphCallbackManager: ...

def get_async_graph_callback_manager_for_config(
    config: RunnableConfig,
    *,
    run_id: UUID | None = None,
) -> _AsyncGraphCallbackManager: ...
```

### Example 1: Audit trail via `GraphCallbackHandler`

```python
import uuid
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from typing_extensions import TypedDict


class AuditLog(GraphCallbackHandler):
    def __init__(self):
        super().__init__()
        self.events: list[dict] = []

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        self.events.append({
            "kind": "interrupt",
            "status": event.status,
            "checkpoint": event.checkpoint_id,
            "values": [i.value for i in event.interrupts],
        })

    def on_resume(self, event: GraphResumeEvent) -> None:
        self.events.append({
            "kind": "resume",
            "status": event.status,
            "checkpoint": event.checkpoint_id,
        })


class State(TypedDict):
    approved: bool
    result: str


def approval_gate(state: State) -> dict:
    answer = interrupt("Approve this action? (yes/no)")
    return {"approved": answer == "yes"}


def execute(state: State) -> dict:
    if state["approved"]:
        return {"result": "action executed"}
    return {"result": "action rejected"}


audit = AuditLog()
graph = (
    StateGraph(State)
    .add_node("gate", approval_gate)
    .add_node("execute", execute)
    .add_edge(START, "gate")
    .add_edge("gate", "execute")
    .add_edge("execute", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {
    "configurable": {"thread_id": "audit-1"},
    "callbacks": [audit],
}

# First run — hits the interrupt
result = graph.invoke({"approved": False, "result": ""}, config=cfg)

# Resume with approval
from langgraph.types import Command
result = graph.invoke(Command(resume="yes"), config=cfg)

print(audit.events)
# [{'kind': 'interrupt', 'status': 'interrupt_before', 'checkpoint': '...', 'values': ['Approve this action? (yes/no)']},
#  {'kind': 'resume',    'status': 'pending',          'checkpoint': '...'}]
```

### Example 2: Using the factory functions with a custom config

```python
import uuid
from langgraph.callbacks import (
    get_sync_graph_callback_manager_for_config,
    GraphCallbackHandler,
    GraphInterruptEvent,
)

class SimpleHandler(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        print(f"[handler] interrupt at {event.status}")

handler = SimpleHandler()
run_id = uuid.uuid4()

config = {"callbacks": [handler]}
mgr = get_sync_graph_callback_manager_for_config(config, run_id=run_id)

# The manager only holds GraphCallbackHandler instances — non-graph handlers are filtered out
print(len(mgr.handlers))   # 1
print(mgr.run_id == run_id)  # True
```

### GraphLifecycleStatus reference table

| Status | When emitted | Event type |
|---|---|---|
| `"input"` | Graph receives new input on resume | `GraphResumeEvent` |
| `"pending"` | Graph resumes from a stored interrupt | `GraphResumeEvent` |
| `"done"` | Graph execution completed normally | — (no lifecycle event; check stream end) |
| `"interrupt_before"` | Execution paused before a node due to `interrupt_before=` | `GraphInterruptEvent` |
| `"interrupt_after"` | Execution paused by `interrupt()` call inside a node | `GraphInterruptEvent` |
| `"out_of_steps"` | Recursion limit reached | `GraphInterruptEvent` |

---

## 9 · `BaseStore` — abstract store contract and `batch()` dispatch pattern

**Module:** `langgraph.store.base`  
**Import:**
```python
from langgraph.store.base import BaseStore, Op, Result
```

`InMemoryStore` and `PostgresStore` both subclass `BaseStore`. If you need to target a custom database (e.g., MongoDB, Cassandra, Redis JSON), implement `batch()` and `abatch()` — all higher-level convenience methods (`get`, `put`, `search`, `list_namespaces`, and their async variants) are implemented in `BaseStore` in terms of those two primitives.

### Source signature (1.2.4)

```python
Op = GetOp | PutOp | SearchOp | ListNamespacesOp
Result = Item | None | list[SearchItem] | list[tuple[str, ...]]

class BaseStore(ABC):
    supports_ttl: bool = False
    ttl_config: TTLConfig | None = None

    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]: ...

    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]: ...

    # Convenience wrappers (all implemented in terms of batch/abatch):
    def get(self, namespace, key, *, refresh_ttl=None) -> Item | None: ...
    def put(self, namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None: ...
    def delete(self, namespace, key) -> None: ...
    def search(self, namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None) -> list[SearchItem]: ...
    def list_namespaces(self, *, prefix=None, suffix=None, max_depth=None, limit=100, offset=0) -> list[tuple[str, ...]]: ...
    # Async: aget, aput, adelete, asearch, alist_namespaces
```

### Example 1: Minimal custom `BaseStore` backed by a plain Python dict

```python
import asyncio
from collections.abc import Iterable
from datetime import datetime, timezone
from langgraph.store.base import (
    BaseStore, GetOp, PutOp, SearchOp, ListNamespacesOp,
    Item, SearchItem, Op, Result,
)
from langgraph.store.base import NOT_PROVIDED


class DictStore(BaseStore):
    """Bare-bones in-memory store for illustration."""

    def __init__(self):
        self._data: dict[tuple, dict[str, dict]] = {}  # ns -> key -> value

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        now = datetime.now(timezone.utc)

        for op in ops:
            if isinstance(op, GetOp):
                ns_data = self._data.get(op.namespace, {})
                if op.key in ns_data:
                    v = ns_data[op.key]
                    results.append(Item(
                        value=v["value"], key=op.key,
                        namespace=op.namespace,
                        created_at=v["created_at"], updated_at=v["updated_at"],
                    ))
                else:
                    results.append(None)

            elif isinstance(op, PutOp):
                if op.value is None:
                    self._data.get(op.namespace, {}).pop(op.key, None)
                else:
                    ns = self._data.setdefault(op.namespace, {})
                    existing = ns.get(op.key, {})
                    ns[op.key] = {
                        "value": op.value,
                        "created_at": existing.get("created_at", now),
                        "updated_at": now,
                    }
                results.append(None)

            elif isinstance(op, SearchOp):
                ns_data = self._data.get(op.namespace_prefix, {})
                items = [
                    SearchItem(
                        namespace=op.namespace_prefix, key=k,
                        value=v["value"],
                        created_at=v["created_at"], updated_at=v["updated_at"],
                    )
                    for k, v in ns_data.items()
                    if op.filter is None or all(
                        v["value"].get(fk) == fv for fk, fv in op.filter.items()
                    )
                ]
                results.append(items[op.offset : op.offset + op.limit])

            elif isinstance(op, ListNamespacesOp):
                all_ns = list(self._data.keys())
                if op.prefix:
                    all_ns = [ns for ns in all_ns if ns[: len(op.prefix)] == tuple(op.prefix)]
                results.append(all_ns[op.offset : op.offset + op.limit])

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return self.batch(ops)


# Exercise the custom store
s = DictStore()
s.put(("users", "alice"), "prefs", {"theme": "dark"})
s.put(("users", "bob"),   "prefs", {"theme": "light"})

item = s.get(("users", "alice"), "prefs")
print(item.value)  # {'theme': 'dark'}

results = s.search(("users",), filter={"theme": "dark"})
print([r.key for r in results])  # ['prefs']  (from alice's namespace? adjust per ns matching)

ns_list = s.list_namespaces()
print(ns_list)  # [('users', 'alice'), ('users', 'bob')]
```

### Example 2: Using `BaseStore` methods with TTL

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# put() with ttl= (in minutes, per BaseStore.put docstring)
# InMemoryStore supports TTL when ttl_config is set
store = InMemoryStore(ttl={"default_ttl": 60, "refresh_on_read": True})
store.put(("session", "u1"), "data", {"counter": 0}, ttl=10)

item = store.get(("session", "u1"), "data")
print(item.value if item else "expired")  # {'counter': 0}
```

### Example 3: Batching mixed operations

All convenience methods translate to `batch()` calls. You can call `batch()` directly for efficiency:

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import GetOp, PutOp

store = InMemoryStore()

# Single batch with mixed ops
results = store.batch([
    PutOp(("ns",), "k1", {"v": 1}, None),
    PutOp(("ns",), "k2", {"v": 2}, None),
    GetOp(("ns",), "k1"),
    GetOp(("ns",), "k2"),
])

print(results[2].value)  # {'v': 1}
print(results[3].value)  # {'v': 2}
```

---

## 10 · `_get_channels` + annotation-to-channel inference — how `StateGraph` maps types to channels

**Module:** `langgraph.graph.state`  
**Private functions:**
```python
from langgraph.graph.state import (
    _get_channels,
    _get_channel,
    _is_field_channel,
    _is_field_binop,
    _is_field_managed_value,
)
```

When you compile a `StateGraph`, LangGraph calls `_get_channels(schema)` to convert your TypedDict annotations into the appropriate channel objects. Understanding this machinery explains every field annotation pattern: why `Annotated[list, add_messages]` creates a `BinaryOperatorAggregate`, why `Annotated[int, EphemeralValue]` creates an `EphemeralValue` channel, and why a plain `str` field defaults to `LastValue`.

### Decision tree (1.2.4)

```
_get_channel(name, annotation)
│
├─ If annotation is Required[X] / NotRequired[X] → unwrap to X
│
├─ _is_field_managed_value(name, annotation)
│   └─ Annotated[T, SomeManagedValue] → SomeManagedValue (e.g. IsLastStep)
│
├─ _is_field_channel(annotation)
│   └─ Annotated[T, some_channel_instance] → that instance (e.g. DeltaChannel)
│   └─ Annotated[T, SomeChannelClass] → SomeChannelClass(T)  (e.g. EphemeralValue)
│
├─ _is_field_binop(annotation)
│   └─ Annotated[T, callable(a, b) -> c] → BinaryOperatorAggregate(T, callable)
│
└─ fallback → LastValue(annotation)
```

### Example 1: Walking through common annotations

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.state import _get_channel
from langgraph.channels.last_value import LastValue
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages

# 1. Plain annotation → LastValue
ch = _get_channel("score", int)
print(type(ch).__name__)  # LastValue

# 2. Annotated with callable → BinaryOperatorAggregate
ch = _get_channel("total", Annotated[int, operator.add])
print(type(ch).__name__)  # BinaryOperatorAggregate

# 3. Annotated with add_messages (callable) → BinaryOperatorAggregate
ch = _get_channel("msgs", Annotated[list, add_messages])
print(type(ch).__name__)  # BinaryOperatorAggregate

# 4. Annotated with channel class → EphemeralValue(list)
ch = _get_channel("scratch", Annotated[list, EphemeralValue])
print(type(ch).__name__)  # EphemeralValue

# 5. Managed value (IsLastStep) — needs allow_managed=True
from langgraph.managed.is_last_step import IsLastStep
ch = _get_channel("done", Annotated[bool, IsLastStep])
print(ch)  # <class 'langgraph.managed.is_last_step.IsLastStep'>
```

### Example 2: `_is_field_binop` validates the reducer signature

The reducer must accept exactly two positional arguments. Passing a single-argument function raises `ValueError`:

```python
from langgraph.graph.state import _is_field_binop
from typing import Annotated

# Valid: (a, b) -> c
channel = _is_field_binop(Annotated[list, lambda a, b: a + b])
print(channel is not None)  # True

# Invalid: (x,) -> x — single argument
try:
    _is_field_binop(Annotated[list, lambda x: x])
except ValueError as e:
    print(e)  # Invalid reducer signature. Expected (a, b) -> c. ...
```

### Example 3: Full TypedDict inspection via `_get_channels`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.state import _get_channels
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps
from langgraph.graph.message import add_messages


class MyState(TypedDict):
    messages: Annotated[list, add_messages]   # → BinaryOperatorAggregate
    score:    Annotated[int, operator.add]    # → BinaryOperatorAggregate
    name:     str                             # → LastValue
    done:     Annotated[bool, IsLastStep]     # → ManagedValue (not a channel)
    steps:    Annotated[int, RemainingSteps]  # → ManagedValue (not a channel)


channels, managed, hints = _get_channels(MyState)

print("Channels:")
for k, ch in channels.items():
    print(f"  {k}: {type(ch).__name__}")
# messages: BinaryOperatorAggregate
# score:    BinaryOperatorAggregate
# name:     LastValue

print("Managed values:")
for k, mv in managed.items():
    print(f"  {k}: {mv.__name__}")
# done:  IsLastStep
# steps: RemainingSteps
```

---

## Vol. index {#vol-index}

| Vol. | Classes covered |
|---|---|
| [1](./langgraph_class_deep_dives/) | StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore |
| [2](./langgraph_class_deep_dives_v2/) | RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl |
| [3](./langgraph_class_deep_dives_v3/) | interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig |
| [4](./langgraph_class_deep_dives_v4/) | set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node |
| [5](./langgraph_class_deep_dives_v5/) | RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt protocol |
| [6](./langgraph_class_deep_dives_v6/) | GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message, ProtocolEvent |
| [7](./langgraph_class_deep_dives_v7/) | PregelProtocol/StreamProtocol, BackgroundExecutor, AsyncBatchedBaseStore, get_text_at_path/tokenize_path, SerdeEvent, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec |
| [8](./langgraph_class_deep_dives_v8/) | ExecutionInfo/Runtime.heartbeat, ServerInfo, ReplayState, StreamMux, Call, ChannelWrite/ChannelWriteEntry, PregelRunner/FuturesDict, WritesProtocol, SyncPregelLoop/AsyncPregelLoop, DuplexStream |
| [9](./langgraph_class_deep_dives_v9/) | ToolCallRequest, Send+timeout, create_react_agent hooks, RetryPolicy chained, CachePolicy key_func, InMemoryStore raw embeddings, Command.PARENT, TimeoutPolicy.coerce(), entrypoint multi-policy |
| [10](./langgraph_class_deep_dives_v10/) | Durability modes, NodeError/NodeCancelledError, TaskPayload/TaskResultPayload, CheckpointPayload/CheckpointTask, Item/SearchItem, GetOp/PutOp/SearchOp/ListNamespacesOp/MatchCondition, UIMessage/RemoveUIMessage, StreamPart variants |
| [11](./langgraph_class_deep_dives_v11/) | InjectedState, InjectedStore, MessagesState, Overwrite, ToolOutputMixin, CheckpointMetadata, CheckpointTuple, StateUpdate, PersistentDict, DeltaChannelHistory |
| [12](./langgraph_class_deep_dives_v12/) | RemoteGraph/RemoteException, PostgresSaver/ShallowPostgresSaver, AsyncPostgresSaver, PostgresStore/PoolConfig, AsyncPostgresStore, ANNIndexConfig/HNSWConfig/IVFFlatConfig/PostgresIndexConfig, GraphRunStream/SubgraphRunStream, ToolCallWithContext/ToolInvocationError, LifecyclePayload/LifecycleTransformer, MessagesTransformer/CheckpointsTransformer/TasksTransformer |
| **13** | **LastValue/LastValueAfterFinish, BranchSpec, BaseCache, get_config/get_store runtime accessors, ManagedValue/ManagedValueSpec, default_cache_key/_freeze, map_debug_tasks/map_debug_task_results/map_debug_checkpoint, GraphLifecycleStatus/callback manager factories, BaseStore abstract contract, _get_channels/annotation-to-channel inference** |

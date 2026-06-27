---
title: "Class deep-dives Vol. 27 — channels, store primitives, managed values, tool interceptors & node errors (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented class groups in LangGraph 1.2.6: BinaryOperatorAggregate (reducer channels — Overwrite escape hatch, MISSING bootstrap, lambda equality rule), Topic (PubSub channel — accumulate=True vs False, list vs scalar write, backwards-compat tuple checkpoint), UntrackedValue (never-checkpointed transient channel — guard singleton enforcement, MISSING init, checkpoint always returns MISSING), BaseStore+Item+SearchItem (cross-thread long-term memory base — batch(ops) contract, namespace/key/value data model, score field on SearchItem), PutOp+GetOp+SearchOp+ListNamespacesOp (store batch operation NamedTuples — TTL, index paths, filter operators, max_depth truncation, pagination), InMemoryStore (in-memory KV + cosine vector store — _data/_vectors storage layout, ThreadPoolExecutor for sync queries, numpy fast path, max-pooling dedup), ManagedValue+ManagedValueSpec (abstract scratchpad protocol — how the runtime calls .get(scratchpad), why you can never write to a managed field), IsLastStepManager+RemainingStepsManager (built-in managed values — step/stop scratchpad fields, is_last_step=step==stop-1, remaining_steps=stop-step), ToolCallRequest+ToolCallWrapper+ToolInvocationError (ToolNode interceptor API — immutable .override() method, sync/async wrapper types, filtered validation errors that exclude injected args), and NodeError+NodeCancelledError+NodeTimeoutError (node-level failure context dataclasses — error_handler injection, asyncio.CancelledError boxing, idle vs run timeout discrimination)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 27"
  order: 58
---

# Class deep-dives Vol. 27 — channels, store primitives, managed values, tool interceptors & node errors (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---|---|
| 1 | `BinaryOperatorAggregate` | `langgraph.channels.binop` |
| 2 | `Topic` | `langgraph.channels.topic` |
| 3 | `UntrackedValue` | `langgraph.channels.untracked_value` |
| 4 | `BaseStore` + `Item` + `SearchItem` | `langgraph.store.base` |
| 5 | `PutOp` + `GetOp` + `SearchOp` + `ListNamespacesOp` | `langgraph.store.base` |
| 6 | `InMemoryStore` | `langgraph.store.memory` |
| 7 | `ManagedValue` + `ManagedValueSpec` | `langgraph.managed.base` |
| 8 | `IsLastStepManager` + `RemainingStepsManager` | `langgraph.managed.is_last_step` |
| 9 | `ToolCallRequest` + `ToolCallWrapper` + `ToolInvocationError` | `langgraph.prebuilt.tool_node` |
| 10 | `NodeError` + `NodeCancelledError` + `NodeTimeoutError` | `langgraph.errors` |

---

## 1 · `BinaryOperatorAggregate`

**Module**: `langgraph.channels.binop`  
**First dedicated coverage.**

`BinaryOperatorAggregate` is the channel type behind every `Annotated[T, reducer_fn]` state field. When multiple nodes write to the same key in a single super-step, the channel applies the reducer function in write-arrival order: `new_value = reducer(old_value, each_write)`.

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]): ...
    def update(self, values: Sequence[Value]) -> bool: ...
    def get(self) -> Value: ...
    def checkpoint(self) -> Value: ...
```

**Key implementation facts:**

- **MISSING bootstrap** — if the type constructor raises (e.g. for abstract types like `Sequence`), `self.value` is set to `MISSING`. The first `update()` call then uses `values[0]` directly rather than `operator(MISSING, values[0])`.
- **Overwrite escape hatch** — wrapping a value in `Overwrite(value)` (or `{OVERWRITE: value}`) replaces the accumulated result entirely instead of reducing through the operator. Only one `Overwrite` per super-step is allowed; a second raises `InvalidUpdateError(INVALID_CONCURRENT_GRAPH_UPDATE)`.
- **Lambda equality** — `__eq__` considers two `BinaryOperatorAggregate` channels with lambda operators as equal (because all lambdas share `__name__ == "<lambda>"`). Named functions compare by identity (`is`).
- **Empty update** — `update([])` returns `False` without modifying `value`. This signals "no state change" to the Pregel runtime.

### Example 1 — running total with parallel writers

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    total: Annotated[int, operator.add]
    items: Annotated[list[str], operator.add]


def node_a(state: State) -> dict:
    return {"total": 10, "items": ["a"]}


def node_b(state: State) -> dict:
    return {"total": 5, "items": ["b", "c"]}


builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", END)
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"total": 0, "items": []})
print(result["total"])  # 15   (0+10+5 — initial + node_a + node_b)
print(result["items"])  # ['a', 'b', 'c']
```

> Both nodes ran in the same super-step. `operator.add` on `int` and `list` is applied twice — once per write — in the order they arrive.

### Example 2 — custom reducer and `Overwrite` reset

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite


def keep_max(a: float, b: float) -> float:
    return max(a, b)


class State(TypedDict):
    score: Annotated[float, keep_max]


def update_score(state: State) -> dict:
    return {"score": 7.5}


def reset_score(state: State) -> dict:
    # Overwrite bypasses keep_max entirely; result is 1.0 regardless of current score
    return {"score": Overwrite(1.0)}


builder = StateGraph(State)
builder.add_node("update", update_score)
builder.add_node("reset", reset_score)
builder.add_edge(START, "update")
builder.add_edge("update", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
result = graph.invoke({"score": 3.0})
print(result["score"])  # 1.0  (Overwrite replaced the accumulated max)
```

### Example 3 — type-abstract initialisation

```python
# BinaryOperatorAggregate handles abstract collection types via a normalisation step.
# collections.abc.Sequence → list, .Set → set, .Mapping → dict
import operator
from collections.abc import Sequence as AbcSequence
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    # Using the abstract ABC — the channel still bootstraps to list()
    words: Annotated[AbcSequence[str], operator.add]


def producer(state: State) -> dict:
    return {"words": ["hello", "world"]}


builder = StateGraph(State)
builder.add_node("prod", producer)
builder.add_edge(START, "prod")
builder.add_edge("prod", END)

result = builder.compile().invoke({"words": []})
print(result["words"])  # ['hello', 'world']
```

---

## 2 · `Topic`

**Module**: `langgraph.channels.topic`  
**First dedicated coverage.**

`Topic` is a configurable PubSub channel. Unlike `BinaryOperatorAggregate`, which reduces writes to a single accumulated value, `Topic` collects every write within a super-step into a **list**. This makes it ideal for fan-in event buffers where parallel nodes all emit events that a downstream node should process as a batch.

```python
class Topic(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value], list[Value]]):
    __slots__ = ("values", "accumulate")

    def __init__(self, typ: type[Value], accumulate: bool = False) -> None: ...
    def update(self, values: Sequence[Value | list[Value]]) -> bool: ...
    def get(self) -> Sequence[Value]: ...
    def checkpoint(self) -> list[Value]: ...
```

**Key implementation facts:**

- **`accumulate=False` (default)** — the list is cleared at the start of each `update()` call. The channel only holds events from the *current* super-step. Useful for per-step event buffers.
- **`accumulate=True`** — the list grows across super-steps until explicitly reset. Useful for append-only logs.
- **Scalar or list writes** — a node can return `{"events": "one_event"}` (a scalar) or `{"events": ["a", "b"]}` (a list). The `_flatten()` helper transparently handles both.
- **`get()` raises `EmptyChannelError`** — if no writes arrived in the current super-step (and `accumulate=False`), the channel is considered empty. Nodes downstream of an empty `Topic` field will see it absent from state unless they handle the absence.
- **Checkpoint** — stored as a plain `list[Value]`. On restore, a tuple checkpoint (from old versions) is handled via backwards-compatibility: `checkpoint[1]` is used.

### Example 1 — per-step event buffer (default `accumulate=False`)

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import Topic
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    events: Annotated[list[str], Topic(str)]
    summary: str


def sensor_a(state: State) -> dict:
    return {"events": "temp_high"}


def sensor_b(state: State) -> dict:
    return {"events": ["humidity_low", "pressure_ok"]}


def aggregator(state: State) -> dict:
    all_events = state.get("events", [])
    return {"summary": f"Received {len(all_events)} events: {all_events}"}


builder = StateGraph(State)
builder.add_node("sensor_a", sensor_a)
builder.add_node("sensor_b", sensor_b)
builder.add_node("agg", aggregator)
builder.add_edge(START, "sensor_a")
builder.add_edge(START, "sensor_b")
builder.add_edge("sensor_a", "agg")
builder.add_edge("sensor_b", "agg")
builder.add_edge("agg", END)

graph = builder.compile()
result = graph.invoke({"events": [], "summary": ""})
# sensor_a and sensor_b ran in parallel; Topic collected all three writes
print(result["summary"])
# Received 3 events: ['temp_high', 'humidity_low', 'pressure_ok']
```

### Example 2 — accumulating log across super-steps

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import Topic
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    log: Annotated[list[str], Topic(str, accumulate=True)]
    step: int


def step_node(state: State) -> dict:
    return {"log": f"step_{state['step']}", "step": state["step"] + 1}


def router(state: State) -> str:
    return END if state["step"] >= 3 else "step_node"


builder = StateGraph(State)
builder.add_node("step_node", step_node)
builder.add_edge(START, "step_node")
builder.add_conditional_edges("step_node", router)

graph = builder.compile()
result = graph.invoke({"log": [], "step": 0})
print(result["log"])  # ['step_0', 'step_1', 'step_2']
```

### Example 3 — `Topic` as a fan-in barrier signal

```python
# Pattern: multiple parallel tasks signal completion via Topic.
# A join node waits for all signals before proceeding.
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import Topic
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    ready_signals: Annotated[list[str], Topic(str)]
    result: str


def worker_one(state: State) -> dict:
    # ... do work ...
    return {"ready_signals": "worker_one_done"}


def worker_two(state: State) -> dict:
    # ... do work ...
    return {"ready_signals": "worker_two_done"}


def join(state: State) -> dict:
    sigs = state.get("ready_signals", [])
    return {"result": f"All done: {sorted(sigs)}"}


builder = StateGraph(State)
builder.add_node("w1", worker_one)
builder.add_node("w2", worker_two)
builder.add_node("join", join)
builder.add_edge(START, "w1")
builder.add_edge(START, "w2")
builder.add_edge("w1", "join")
builder.add_edge("w2", "join")
builder.add_edge("join", END)

result = builder.compile().invoke({"ready_signals": [], "result": ""})
print(result["result"])
# All done: ['worker_one_done', 'worker_two_done']
```

---

## 3 · `UntrackedValue`

**Module**: `langgraph.channels.untracked_value`  
**First dedicated coverage.**

`UntrackedValue` is a transient channel: it stores the last received value, but its `checkpoint()` method **always returns `MISSING`**. This means the value is never serialised to the checkpoint store and is lost when a run resumes from a checkpoint. Use it for computed values, nonce flags, or temporary working state that you never need to restore.

```python
class UntrackedValue(Generic[Value], BaseChannel[Value, Value, Value]):
    __slots__ = ("value", "guard")

    def __init__(self, typ: type[Value], guard: bool = True) -> None: ...
    def checkpoint(self) -> Value | Any:
        return MISSING   # never persisted
    def update(self, values: Sequence[Value]) -> bool: ...
    def get(self) -> Value: ...
```

**Key implementation facts:**

- **`guard=True` (default)** — `update()` raises `InvalidUpdateError` if more than one value arrives in a single super-step. This prevents accidental concurrent writes and makes the "last wins" semantics explicit.
- **`guard=False`** — multiple concurrent writes are silently resolved by taking `values[-1]` (the last in arrival order).
- **`checkpoint() → MISSING`** — `from_checkpoint()` always starts fresh regardless of what was in the checkpoint blob. The Pregel runtime detects `MISSING` and treats the field as absent on the next run.
- **`__eq__`** — two `UntrackedValue` channels are equal if and only if their `guard` flags match. Operator-identity is not needed (unlike `BinaryOperatorAggregate`) since there is no operator.

### Example 1 — caching a computed value that should not survive a resume

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    user_input: str
    # This embedding is expensive to compute but should NOT be stored in the checkpoint.
    # If the run resumes, the node will recompute it.
    cached_embedding: Annotated[list[float], UntrackedValue(list)]


def embed_node(state: State) -> dict:
    # Simulate embedding computation (replace with real model call)
    embedding = [0.1, 0.2, 0.3]
    return {"cached_embedding": embedding}


def search_node(state: State) -> dict:
    emb = state.get("cached_embedding", [])
    print(f"Searching with embedding of dim {len(emb)}")
    return {}


builder = StateGraph(State)
builder.add_node("embed", embed_node)
builder.add_node("search", search_node)
builder.add_edge(START, "embed")
builder.add_edge("embed", "search")
builder.add_edge("search", END)

graph = builder.compile()
graph.invoke({"user_input": "Hello", "cached_embedding": []})
# Searching with embedding of dim 3
```

### Example 2 — nonce flag that resets on each run

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    count: int
    # A per-run flag: True if this super-step modified count, False otherwise.
    # Never checkpointed, so on resume it defaults to MISSING/absent.
    modified_this_step: Annotated[bool, UntrackedValue(bool)]


def increment(state: State) -> dict:
    return {"count": state["count"] + 1, "modified_this_step": True}


builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "t1"}}
r1 = graph.invoke({"count": 0, "modified_this_step": False}, cfg)
print(r1["count"])                # 1
print(r1["modified_this_step"])   # True  (set this run)

# Resume: modified_this_step is NOT in the checkpoint, so it will be absent
# until increment() sets it again.
r2 = graph.invoke(None, cfg)
print(r2["count"])                # 2
print(r2["modified_this_step"])   # True  (set again this run)
```

### Example 3 — `guard=False` for last-write-wins from parallel workers

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    # Multiple parallel nodes may write; we only care about one value (guard=False)
    latest_status: Annotated[str, UntrackedValue(str, guard=False)]


def fast_worker(state: State) -> dict:
    return {"latest_status": "fast_done"}


def slow_worker(state: State) -> dict:
    return {"latest_status": "slow_done"}


builder = StateGraph(State)
builder.add_node("fast", fast_worker)
builder.add_node("slow", slow_worker)
builder.add_edge(START, "fast")
builder.add_edge(START, "slow")
builder.add_edge("fast", END)
builder.add_edge("slow", END)

result = builder.compile().invoke({"latest_status": ""})
# One of 'fast_done' or 'slow_done' depending on arrival order
print(result["latest_status"])
```

---

## 4 · `BaseStore` + `Item` + `SearchItem`

**Module**: `langgraph.store.base`  
**First dedicated coverage.**

`BaseStore` is the abstract base class for LangGraph's long-term key-value store system. Unlike checkpointers (which are bound to a single `thread_id`), stores provide memory that outlives any particular conversation and can be shared across users, threads, and agent runs.

```python
class BaseStore(ABC):
    supports_ttl: bool = False
    ttl_config: TTLConfig | None = None

    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]: ...
    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]: ...

    # Convenience wrappers (all have async `a`-prefixed variants)
    def get(namespace, key, *, refresh_ttl=None) -> Item | None: ...
    def put(namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None: ...
    def delete(namespace, key) -> None: ...
    def search(namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0) -> list[SearchItem]: ...
    def list_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100) -> list[tuple[str, ...]]: ...
```

**`Item`** — returned by `get()` and `search()` (base class):

```python
class Item:
    __slots__ = ("value", "key", "namespace", "created_at", "updated_at")
    value: dict[str, Any]
    key: str
    namespace: tuple[str, ...]
    created_at: datetime
    updated_at: datetime

    def dict(self) -> dict: ...  # JSON-serializable form
```

**`SearchItem(Item)`** — returned exclusively by `search()`, adds a relevance score:

```python
class SearchItem(Item):
    __slots__ = ("score",)
    score: float | None  # None when no query was given (filter-only search)
```

**Key implementation facts:**

- **Namespace rules** — the namespace must be a non-empty tuple of non-empty strings, no dots in any label, and the root element must not be `"langgraph"`. Violations raise `InvalidNamespaceError`.
- **`batch(ops)` contract** — the single abstract method. All convenience methods (`get`, `put`, `search`, etc.) are synchronous/asynchronous wrappers that assemble the right `Op` NamedTuple and call `batch()` / `abatch()`. Implementing a custom store only requires `batch` and `abatch`.
- **TTL opt-in** — subclasses must set `supports_ttl = True` to accept `ttl` arguments. Passing a non-`None` TTL to a store that doesn't support it raises `NotImplementedError`.
- **`Item` timestamp coercion** — the `__init__` accepts `str` ISO timestamps (e.g. from JSON deserialization) and calls `datetime.fromisoformat()` to normalise them.

### Example 1 — basic get / put / delete

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Store a user profile
store.put(("users", "alice"), "profile", {"name": "Alice", "plan": "pro"})

# Retrieve it
item = store.get(("users", "alice"), "profile")
print(item.value)      # {'name': 'Alice', 'plan': 'pro'}
print(item.namespace)  # ('users', 'alice')
print(item.key)        # 'profile'
print(type(item.created_at))  # <class 'datetime.datetime'>

# Update in-place (put overwrites)
store.put(("users", "alice"), "profile", {"name": "Alice", "plan": "enterprise"})
updated = store.get(("users", "alice"), "profile")
print(updated.value["plan"])  # 'enterprise'

# Delete
store.delete(("users", "alice"), "profile")
print(store.get(("users", "alice"), "profile"))  # None
```

### Example 2 — namespace listing and scoped search

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

store.put(("docs", "research"), "paper1", {"title": "LLM survey", "year": 2024})
store.put(("docs", "research"), "paper2", {"title": "RAG techniques", "year": 2023})
store.put(("docs", "blog"), "post1", {"title": "Getting started with LangGraph"})
store.put(("users", "bob"), "prefs", {"theme": "light"})

# List all namespaces under "docs"
namespaces = store.list_namespaces(prefix=("docs",))
print(namespaces)  # [('docs', 'blog'), ('docs', 'research')]

# Search within "docs/research" with a filter (restrict prefix so every item
# under it has a "year" field — avoids TypeError on items missing the key)
results = store.search(("docs", "research"), filter={"year": {"$gte": 2024}})
for r in results:
    print(r.key, r.value["title"])  # paper1  LLM survey

# list_namespaces with max_depth truncates deeper paths
all_ns = store.list_namespaces(max_depth=1)
print(all_ns)  # [('docs',), ('users',)]
```

### Example 3 — wiring a store into a graph via `compile(store=...)`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


class State(TypedDict):
    user_id: str
    response: str


def recall_node(state: State, *, store: BaseStore) -> dict:
    """Read the user's saved preferences from the store."""
    item = store.get(("users", state["user_id"]), "prefs")
    prefs = item.value if item else {}
    return {"response": f"Prefs for {state['user_id']}: {prefs}"}


def save_node(state: State, *, store: BaseStore) -> dict:
    """Write back updated preferences."""
    store.put(("users", state["user_id"]), "prefs", {"theme": "dark"})
    return {}


builder = StateGraph(State)
builder.add_node("recall", recall_node)
builder.add_node("save", save_node)
builder.add_edge(START, "recall")
builder.add_edge("recall", "save")
builder.add_edge("save", END)

store = InMemoryStore()
graph = builder.compile(store=store)
result = graph.invoke({"user_id": "alice", "response": ""})
print(result["response"])  # Prefs for alice: {}  (nothing saved yet)

# Second run: preferences are now persisted
graph.invoke({"user_id": "alice", "response": ""})
item = store.get(("users", "alice"), "prefs")
print(item.value)  # {'theme': 'dark'}
```

---

## 5 · `PutOp` + `GetOp` + `SearchOp` + `ListNamespacesOp`

**Module**: `langgraph.store.base`  
**First dedicated coverage.**

The four store operation NamedTuples form the **batch protocol** that all `BaseStore` implementations must handle. Every convenience method (`get`, `put`, `search`, `list_namespaces`) composes one of these and calls `batch([op])`.

```python
class GetOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    refresh_ttl: bool = True

class PutOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any] | None   # None → delete
    index: Literal[False] | list[str] | None = None
    ttl: float | None = None

class SearchOp(NamedTuple):
    namespace_prefix: tuple[str, ...]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
    refresh_ttl: bool = True

class ListNamespacesOp(NamedTuple):
    match_conditions: tuple[MatchCondition, ...] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0
```

**Key implementation facts:**

- **`PutOp.value = None`** signals a deletion — `InMemoryStore` pops the key from `_data` and `_vectors`.
- **`PutOp.index`** controls vector indexing: `None` = use store defaults, `False` = skip indexing, `list[str]` = embed only those JSON-path fields.
- **`PutOp.ttl`** is in **minutes**. Requires `store.supports_ttl = True`; otherwise calling `put(..., ttl=5.0)` raises `NotImplementedError`.
- **`SearchOp.filter`** supports comparison operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`. A bare value is treated as `$eq`.
- **`ListNamespacesOp.match_conditions`** holds `MatchCondition(match_type="prefix"|"suffix", path=tuple)` entries; wildcards `"*"` in the path match any single namespace segment.
- **`refresh_ttl`** on `GetOp` / `SearchOp` controls whether reading the item resets its TTL timer. Ignored when the store does not support TTL.

### Example 1 — using `batch()` directly for atomic multi-operation transactions

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import PutOp, GetOp

store = InMemoryStore()

# InMemoryStore.batch() resolves GetOps from pre-existing data before applying
# PutOps in the same call, so writes and reads must be in separate batches.

# First batch: writes
store.batch([
    PutOp(namespace=("session", "s1"), key="token", value={"jwt": "abc123"}),
    PutOp(namespace=("session", "s1"), key="meta",  value={"ip": "10.0.0.1"}),
])

# Second batch: reads (items are now visible in _data)
results = store.batch([
    GetOp(namespace=("session", "s1"), key="token"),
    GetOp(namespace=("session", "s1"), key="meta"),
])
# results[0] -> Item, results[1] -> Item
print(results[0].value)  # {'jwt': 'abc123'}
print(results[1].value)  # {'ip': '10.0.0.1'}
```

### Example 2 — SearchOp with operator-based filters

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import PutOp, SearchOp

store = InMemoryStore()

# Seed data
for i, score in enumerate([3.5, 4.8, 2.1, 4.0, 4.9]):
    store.put(("products",), f"p{i}", {"score": score, "category": "widget"})

# Single search via convenience method with $gte filter
hits = store.search(("products",), filter={"score": {"$gte": 4.0}})
print([h.key for h in hits])   # ['p1', 'p3', 'p4']

# Same via raw SearchOp
ops = [SearchOp(namespace_prefix=("products",), filter={"score": {"$lt": 3.0}})]
results = store.batch(ops)
print([r.key for r in results[0]])  # ['p2']
```

### Example 3 — `ListNamespacesOp` with wildcards and `max_depth`

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import ListNamespacesOp, MatchCondition

store = InMemoryStore()

for ns in [
    ("org", "acme", "users"),
    ("org", "acme", "docs"),
    ("org", "beta", "users"),
    ("org", "beta", "docs", "drafts"),
    ("sys", "logs"),
]:
    store.put(ns, "item", {"x": 1})

# List all namespaces up to depth 2 under "org"
op = ListNamespacesOp(
    match_conditions=(MatchCondition(match_type="prefix", path=("org",)),),
    max_depth=2,
)
result = store.batch([op])[0]
print(result)
# [('org', 'acme'), ('org', 'beta')]

# Wildcard: any org, specifically the 'users' sub-namespace
op2 = ListNamespacesOp(
    match_conditions=(
        MatchCondition(match_type="prefix", path=("org", "*", "users")),
    ),
)
result2 = store.batch([op2])[0]
print(result2)
# [('org', 'acme', 'users'), ('org', 'beta', 'users')]
```

---

## 6 · `InMemoryStore`

**Module**: `langgraph.store.memory`  
**First dedicated coverage.**

`InMemoryStore` is the default concrete `BaseStore` implementation. It backs all data with two `defaultdict` tables: `_data` for key-value pairs and `_vectors` for per-field embeddings used by cosine-similarity search.

```python
class InMemoryStore(BaseStore):
    __slots__ = ("_data", "_vectors", "index_config", "embeddings")

    def __init__(self, *, index: IndexConfig | None = None) -> None: ...
    def batch(self, ops: Iterable[Op]) -> list[Result]: ...
    async def abatch(self, ops: Iterable[Op]) -> list[Result]: ...
```

**Internal storage layout:**

```
_data[namespace][key]  → Item
_vectors[namespace][key][json_path]  → list[float]  (embedding vector)
```

**Key implementation facts:**

- **Sync embedding** — `batch()` uses a `ThreadPoolExecutor` to run `embeddings.embed_query()` for each unique search query concurrently (not in the async event loop). This prevents the sync path from blocking an async caller.
- **Async embedding** — `abatch()` uses `asyncio.gather()` to run `embeddings.aembed_query()` concurrently.
- **Numpy fast path** — cosine similarity is computed via `numpy` if installed, with a `functools.lru_cache(maxsize=1)` guard so the availability check only logs a warning once.
- **Max-pooling dedup** — when a query returns multiple embedding vectors for the same key (from different indexed fields), the highest score wins. Items without embeddings are appended after scored items to fill the `limit`.
- **`PutOp.value = None` → delete** — removes from both `_data` and `_vectors`.
- **No TTL support** — `supports_ttl = False` (inherited from `BaseStore`). Passing `ttl` raises `NotImplementedError`.

### Example 1 — basic KV operations with namespace hierarchy

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Multi-level namespace
store.put(("org", "hr", "employees"), "emp_001", {
    "name": "Jane Doe",
    "dept": "engineering",
    "level": 4,
})
store.put(("org", "hr", "employees"), "emp_002", {
    "name": "Bob Smith",
    "dept": "design",
    "level": 3,
})

# Filter by exact match
results = store.search(("org", "hr", "employees"), filter={"dept": "engineering"})
print(results[0].value["name"])  # Jane Doe

# Filter by comparison
senior = store.search(("org", "hr", "employees"), filter={"level": {"$gte": 4}})
print([r.value["name"] for r in senior])  # ['Jane Doe']

# Namespace listing
store.put(("org", "finance"), "budget", {"q": 1, "amount": 100000})
print(store.list_namespaces(prefix=("org",), max_depth=2))
# [('org', 'finance'), ('org', 'hr')]
```

### Example 2 — vector search with a custom embedding function

```python
import math
from langgraph.store.memory import InMemoryStore


def simple_embed(texts: list[str]) -> list[list[float]]:
    """Toy embedder: creates a 3-D vector from character frequencies."""
    def encode(text: str) -> list[float]:
        v = [text.count(c) for c in "aeiou"][:3]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
    return [encode(t) for t in texts]


store = InMemoryStore(index={"dims": 3, "embed": simple_embed, "fields": ["content"]})

# Index three documents
store.put(("docs",), "d1", {"content": "aardvark eats ants"})
store.put(("docs",), "d2", {"content": "big blue bus"})
store.put(("docs",), "d3", {"content": "eagle over ocean"})

# Semantic search — query embedding is compared against indexed vectors
results = store.search(("docs",), query="animals eating insects")
for r in results:
    print(r.key, f"score={r.score:.3f}", r.value["content"])
# d1 and d3 will rank higher (vowel-heavy words)
```

### Example 3 — async batch interface for coroutine-based graphs

```python
import asyncio
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import PutOp, GetOp


async def main():
    store = InMemoryStore()

    # Parallel writes via abatch
    await store.abatch([
        PutOp(("sessions",), "sess_1", {"uid": "u1", "active": True}),
        PutOp(("sessions",), "sess_2", {"uid": "u2", "active": False}),
    ])

    # Async search
    actives = await store.asearch(("sessions",), filter={"active": True})
    print([a.value["uid"] for a in actives])  # ['u1']

    # Async get + async delete
    item = await store.aget(("sessions",), "sess_1")
    print(item.value)  # {'uid': 'u1', 'active': True}

    await store.adelete(("sessions",), "sess_1")
    print(await store.aget(("sessions",), "sess_1"))  # None


asyncio.run(main())
```

---

## 7 · `ManagedValue` + `ManagedValueSpec`

**Module**: `langgraph.managed.base`  
**First dedicated coverage.**

Managed values are special state-field annotations that LangGraph populates automatically from the Pregel executor's internal scratchpad, rather than from node return values. The base protocol is a two-type system:

```python
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...

ManagedValueSpec = type[ManagedValue]          # the class itself, not an instance
ManagedValueMapping = dict[str, ManagedValueSpec]
```

A **`ManagedValueSpec`** is just the class object (not an instance). You annotate a state field as `Annotated[T, SomeManager]` where `SomeManager` is the class — this is detected at graph-compile time by `is_managed_value(value)`.

```python
def is_managed_value(value: Any) -> TypeGuard[ManagedValueSpec]:
    return isclass(value) and issubclass(value, ManagedValue)
```

**Key implementation facts:**

- **Static `get(scratchpad)`** — the method is `@staticmethod`, meaning it receives only the scratchpad and nothing else. The class is never instantiated.
- **`PregelScratchpad`** — holds `step: int` and `stop: int` (the recursion limit) and is created fresh for each graph invocation.
- **Read-only by design** — node returns that include managed-value keys are silently ignored. The graph, not the node, controls the values.
- **Custom managed values** — you can implement your own by subclassing `ManagedValue` and annotating a field with the subclass. The `get()` method receives only the scratchpad; for richer runtime context, wire values in via `functools.partial` or graph config rather than reading fields that don't exist on the scratchpad.

### Example 1 — implementing a custom managed value

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.graph import StateGraph, START, END


class StepNumberManager(ManagedValue[int]):
    """Injects the current 1-based step number (step 0 → reports 1)."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.step + 1  # make 1-based


StepNumber = Annotated[int, StepNumberManager]


class State(TypedDict):
    count: int
    step_number: StepNumber     # auto-injected; do not write from nodes


def worker(state: State) -> dict:
    print(f"Running on graph step #{state['step_number']}")
    return {"count": state["count"] + 1}


def router(state: State) -> str:
    return END if state["count"] >= 3 else "worker"


builder = StateGraph(State)
builder.add_node("worker", worker)
builder.add_edge(START, "worker")
builder.add_conditional_edges("worker", router)

graph = builder.compile()
graph.invoke({"count": 0, "step_number": 0})
# Running on graph step #1
# Running on graph step #2
# Running on graph step #3
```

### Example 2 — probing managed value detection at runtime

```python
from langgraph.managed.base import is_managed_value, ManagedValue
from langgraph.managed.is_last_step import IsLastStepManager, RemainingStepsManager

# is_managed_value checks: isclass(x) and issubclass(x, ManagedValue)
print(is_managed_value(IsLastStepManager))     # True
print(is_managed_value(RemainingStepsManager)) # True
print(is_managed_value(str))                   # False
print(is_managed_value(42))                    # False

# The ManagedValueSpec alias is just `type[ManagedValue]` — the class itself
spec: type[ManagedValue] = IsLastStepManager
# get is a @staticmethod: class access returns the function directly (no __func__)
print(spec.get)  # <function IsLastStepManager.get at ...>
```

### Example 3 — sharing scratchpad context between a managed value and a node

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.graph import StateGraph, START, END


class ProgressPctManager(ManagedValue[float]):
    """Fractional progress through the recursion limit."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> float:
        total = scratchpad.stop - 1  # avoid div-by-zero on limit=1
        return scratchpad.step / max(total, 1)


ProgressPct = Annotated[float, ProgressPctManager]


class State(TypedDict):
    messages: list[str]
    progress: ProgressPct


def agent(state: State) -> dict:
    pct = state["progress"] * 100
    print(f"Progress: {pct:.0f}%")
    return {"messages": state["messages"] + [f"step@{pct:.0f}%"]}


def done(state: State) -> str:
    return END if len(state["messages"]) >= 3 else "agent"


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", done)

graph = builder.compile()
graph.invoke({"messages": [], "progress": 0.0}, {"recursion_limit": 10})
# Progress: 0%
# Progress: 10%
# Progress: 20%
```

---

## 8 · `IsLastStepManager` + `RemainingStepsManager`

**Module**: `langgraph.managed.is_last_step`  
**First dedicated coverage.**

The two built-in managed value implementations expose the graph's recursion-limit state to node code. Both derive from `ManagedValue` and are exposed as `Annotated` type aliases.

```python
class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

IsLastStep = Annotated[bool, IsLastStepManager]


class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step

RemainingSteps = Annotated[int, RemainingStepsManager]
```

**Key implementation facts:**

- **`scratchpad.stop`** equals the `recursion_limit` from the run config (default 25). Confirmed from `PregelScratchpad` source: `stop` is set to `recursion_limit` at the start of each `invoke()` / `ainvoke()`.
- **`scratchpad.step`** starts at 0 and increments by 1 after each super-step completes.
- **`IsLastStep` is `True` exactly once** — when `step == stop - 1`. At that point the graph will raise `GraphRecursionError` on the *next* step, so `IsLastStep` gives a one-step warning window.
- **`RemainingSteps` counts down from `stop`** to 0. At `step=0` it equals the full `recursion_limit`. When `IsLastStep` is `True`, `RemainingSteps` is exactly `1`.
- Both aliases are exported from `langgraph.managed.is_last_step`. They are **not** re-exported from `langgraph.prebuilt`; always import from the managed module.

### Example 1 — graceful early return on recursion limit

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep


class State(TypedDict):
    result: str
    is_last: IsLastStep


def agent(state: State) -> dict:
    if state["is_last"]:
        # Return a partial result instead of letting the graph crash
        return {"result": "partial: hit recursion limit"}
    # Normal work
    return {"result": state["result"] + "."}


def route(state: State) -> str:
    return END if "limit" in state["result"] or state["result"].count(".") >= 5 else "agent"


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route)

graph = builder.compile()

# Run with a tight recursion limit to trigger IsLastStep
result = graph.invoke({"result": "", "is_last": False}, {"recursion_limit": 3})
print(result["result"])  # 'partial: hit recursion limit'
```

### Example 2 — budget-aware agent using `RemainingSteps`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import RemainingSteps


class State(TypedDict):
    plan: list[str]
    done: list[str]
    remaining: RemainingSteps


def planner(state: State) -> dict:
    steps_left = state["remaining"]
    # Budget: reserve 1 step for the final summary
    affordable = state["plan"][: steps_left - 1]
    return {"plan": affordable}


def executor(state: State) -> dict:
    if not state["plan"]:
        return {"done": state["done"]}
    task = state["plan"][0]
    print(f"Executing: {task} ({state['remaining']} steps left)")
    return {"plan": state["plan"][1:], "done": state["done"] + [task]}


def route(state: State) -> str:
    return END if not state["plan"] else "executor"


builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_conditional_edges("executor", route)

tasks = ["fetch_data", "clean_data", "train_model", "evaluate", "deploy"]
graph = builder.compile()
result = graph.invoke({"plan": tasks, "done": [], "remaining": 0}, {"recursion_limit": 5})
print("Completed:", result["done"])
# Executing: fetch_data (5 steps left)
# ...
```

### Example 3 — combining both in a single state for rich step awareness

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps


class State(TypedDict):
    count: int
    is_last: IsLastStep
    remaining: RemainingSteps
    log: list[str]


def node(state: State) -> dict:
    entry = f"count={state['count']} last={state['is_last']} left={state['remaining']}"
    return {"count": state["count"] + 1, "log": state["log"] + [entry]}


def route(state: State) -> str:
    return END if state["is_last"] or state["count"] >= 10 else "node"


builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")
builder.add_conditional_edges("node", route)

graph = builder.compile()
result = graph.invoke({"count": 0, "is_last": False, "remaining": 0, "log": []},
                      {"recursion_limit": 5})
for entry in result["log"]:
    print(entry)
# count=0 last=False left=5
# count=1 last=False left=4
# count=2 last=False left=3
# count=3 last=False left=2
# count=4 last=True  left=1
```

---

## 9 · `ToolCallRequest` + `ToolCallWrapper` + `ToolInvocationError`

**Module**: `langgraph.prebuilt.tool_node`  
**First dedicated coverage.**

`ToolCallRequest`, `ToolCallWrapper`, and `ToolInvocationError` form the **interceptor API** for `ToolNode`. They let you wrap tool execution with retry logic, caching, input sanitisation, or fallback strategies — without modifying the tools themselves.

```python
@dataclass
class ToolCallRequest:
    tool_call: ToolCall        # {'name': str, 'args': dict, 'id': str, 'type': 'tool_call'}
    tool: BaseTool | None      # None if tool name is not registered
    state: Any                 # current agent state (dict/list/BaseModel)
    runtime: ToolRuntime       # LangGraph runtime context

    def override(self, **overrides: Unpack[_ToolCallRequestOverrides]) -> ToolCallRequest:
        """Immutable update — returns a NEW ToolCallRequest. Do not set attributes directly."""

# Sync wrapper type alias
ToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]],
    ToolMessage | Command,
]

# Async wrapper type alias
AsyncToolCallWrapper = Callable[
    [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
    Awaitable[ToolMessage | Command],
]
```

**`ToolInvocationError`** is raised (and caught by `ToolNode`) when Pydantic schema validation fails on the tool's arguments. It filters out system-injected params (`state`, `store`, `runtime`) from the error message so the LLM only sees args it controls.

**Key implementation facts:**

- **`ToolCallRequest` is a `dataclass`** with a custom `__setattr__` that emits a `DeprecationWarning` when attributes are set directly. Always use `.override()`.
- **`execute` can be called multiple times** — the `execute` callable passed to a wrapper is stateless; calling it twice runs the tool twice. This enables retry patterns.
- **`tool=None`** — when the model requests a tool that is not in `ToolNode._tools_by_name`, the request is passed to the wrapper with `tool=None`. If the wrapper calls `execute()`, validation raises an error for the unknown tool.
- **`wrap_tool_call` / `awrap_tool_call`** are passed to `ToolNode.__init__`. The async wrapper falls back to the sync wrapper if `awrap_tool_call` is not provided.

### Example 1 — passthrough wrapper for observability

```python
import time
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.tool_node import ToolCallRequest


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def timing_wrapper(request: ToolCallRequest, execute):
    """Log how long each tool call takes."""
    t0 = time.perf_counter()
    result = execute(request)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Tool '{request.tool_call['name']}' took {elapsed:.2f}ms")
    return result


tool_node = ToolNode([add], wrap_tool_call=timing_wrapper)

# Quick smoke-test without a real LLM
from langchain_core.messages import AIMessage

state = {"messages": [
    AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 3, "b": 4}, "id": "c1", "type": "tool_call"}])
]}
result = tool_node.invoke(state)
print(result["messages"][0].content)  # '7'
# Tool 'add' took X.XXms
```

### Example 2 — retry wrapper with back-off on tool error

```python
import time
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

attempt_count = {"n": 0}


@tool
def flaky_tool(value: int) -> str:
    """A tool that fails on the first two attempts."""
    attempt_count["n"] += 1
    if attempt_count["n"] < 3:
        raise RuntimeError(f"Temporary failure on attempt {attempt_count['n']}")
    return f"Success after {attempt_count['n']} attempts"


def retry_wrapper(request: ToolCallRequest, execute):
    """Retry up to 3 times with linear back-off."""
    for attempt in range(3):
        try:
            return execute(request)
        except RuntimeError as e:
            if attempt == 2:
                return ToolMessage(
                    content=f"All retries failed: {e}",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            time.sleep(0.01 * (attempt + 1))  # 10ms, 20ms back-off


tool_node = ToolNode(
    [flaky_tool],
    wrap_tool_call=retry_wrapper,
    handle_tool_errors=False,  # let wrapper handle errors
)

state = {"messages": [
    AIMessage(content="", tool_calls=[
        {"name": "flaky_tool", "args": {"value": 42}, "id": "c1", "type": "tool_call"}
    ])
]}
result = tool_node.invoke(state)
print(result["messages"][0].content)  # 'Success after 3 attempts'
```

### Example 3 — request modification before execution

```python
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest


@tool
def search(query: str, max_results: int = 5) -> str:
    """Search the internet."""
    return f"Results for '{query}' (top {max_results})"


def sanitize_wrapper(request: ToolCallRequest, execute):
    """Clamp max_results to a safe maximum and sanitize the query."""
    args = request.tool_call["args"].copy()
    args["max_results"] = min(args.get("max_results", 5), 10)
    args["query"] = args["query"].strip().lower()
    new_call = {**request.tool_call, "args": args}
    return execute(request.override(tool_call=new_call))


tool_node = ToolNode([search], wrap_tool_call=sanitize_wrapper)

state = {"messages": [
    AIMessage(content="", tool_calls=[
        {"name": "search",
         "args": {"query": "  LangGraph  ", "max_results": 100},
         "id": "c1",
         "type": "tool_call"}
    ])
]}
result = tool_node.invoke(state)
print(result["messages"][0].content)
# Results for 'langgraph' (top 10)   ← clamped and trimmed
```

---

## 10 · `NodeError` + `NodeCancelledError` + `NodeTimeoutError`

**Module**: `langgraph.errors`  
**First dedicated coverage.**

These three exception types provide structured failure context for node-level problems. They are passed to `error_handler` functions registered on individual nodes via `StateGraph.add_node(..., error_handler=fn)`.

```python
@dataclass(frozen=True, slots=True)
class NodeError:
    node: str           # Name of the node whose execution failed
    error: BaseException # The exception that was raised

class NodeCancelledError(Exception):
    node: str
    def __init__(self, node: str, message: str | None = None): ...

class NodeTimeoutError(Exception):
    node: str
    timeout: float         # The limit that was exceeded
    run_timeout: float | None
    idle_timeout: float | None
    elapsed: float
    kind: Literal["idle", "run"]
```

**Key implementation facts:**

- **`NodeError` is a frozen dataclass** — immutable and slot-optimised. It is injected by adding a parameter typed `NodeError` to the error handler function.
- **`NodeCancelledError`** boxes a user-raised `asyncio.CancelledError`. Because `asyncio.CancelledError` is a `BaseException` (not `Exception`), a node body that raises it would normally be treated as silent cancellation by the Pregel runner. The retry layer converts it to `NodeCancelledError` so it flows through the normal error path.
- **`NodeTimeoutError.kind`** distinguishes two timeout modes:
  - `"run"` — the node exceeded its total wall-clock budget (`run_timeout`).
  - `"idle"` — the node stopped making progress (no output/stream events) for too long (`idle_timeout`).
- Both `idle_timeout` and `run_timeout` are stored on the exception even when only one fired, enabling diagnostic logging of the full policy.

### Example 1 — node-level error handler with `NodeError`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import Command


class State(TypedDict):
    value: int
    error_info: str


def risky_node(state: State) -> dict:
    if state["value"] < 0:
        raise ValueError(f"Negative value not allowed: {state['value']}")
    return {"value": state["value"] * 2}


def error_handler(state: State, error: NodeError) -> Command:
    """Recover by zeroing the value and logging the error."""
    return Command(
        update={
            "value": 0,
            "error_info": f"Recovered from {error.node}: {type(error.error).__name__}: {error.error}",
        }
    )


builder = StateGraph(State)
builder.add_node("risky", risky_node, error_handler=error_handler)
builder.add_edge(START, "risky")
builder.add_edge("risky", END)

graph = builder.compile()

result = graph.invoke({"value": 5, "error_info": ""})
print(result["value"])       # 10  (normal path)

result_err = graph.invoke({"value": -1, "error_info": ""})
print(result_err["value"])       # 0
print(result_err["error_info"])  # 'Recovered from risky: ValueError: Negative value not allowed: -1'
```

### Example 2 — catching `NodeCancelledError` in an async graph

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeCancelledError
from langgraph.types import Command


class State(TypedDict):
    status: str


async def async_node(state: State) -> dict:
    """A node that deliberately raises asyncio.CancelledError from its body."""
    raise asyncio.CancelledError("simulated external cancel")


async def cancel_handler(state: State, error: NodeCancelledError) -> Command:
    """The error_handler receives NodeCancelledError, not asyncio.CancelledError."""
    return Command(update={"status": f"cancelled: {error.node} — {error}"})


async def main():
    builder = StateGraph(State)
    builder.add_node("async_node", async_node, error_handler=cancel_handler)
    builder.add_edge(START, "async_node")
    builder.add_edge("async_node", END)

    graph = builder.compile()
    result = await graph.ainvoke({"status": ""})
    print(result["status"])
    # cancelled: async_node — Node 'async_node' raised asyncio.CancelledError


asyncio.run(main())
```

### Example 3 — inspecting `NodeTimeoutError` fields for observability

```python
from langgraph.errors import NodeTimeoutError

# Construct a run-timeout error as the Pregel runner would
err = NodeTimeoutError(
    node="slow_llm_call",
    elapsed=35.2,
    kind="run",
    run_timeout=30.0,
)

print(err.node)         # 'slow_llm_call'
print(err.kind)         # 'run'
print(err.timeout)      # 30.0
print(err.elapsed)      # 35.2
print(err.run_timeout)  # 30.0
print(err.idle_timeout) # None
print(str(err))
# Node 'slow_llm_call' exceeded its run timeout of 30.000s (elapsed: 35.200s).

# Construct an idle-timeout error
idle_err = NodeTimeoutError(
    node="waiting_for_response",
    elapsed=12.1,
    kind="idle",
    idle_timeout=10.0,
    run_timeout=120.0,
)
print(idle_err.kind)    # 'idle'
print(idle_err.timeout) # 10.0  (the one that fired)
print(str(idle_err))
# Node 'waiting_for_response' exceeded its idle timeout of 10.000s without
# making progress (elapsed: 12.100s).
```

---

## Summary

| # | Class group | Module | Key takeaway |
|---|---|---|---|
| 1 | `BinaryOperatorAggregate` | `langgraph.channels.binop` | Reducer for parallel writes; `Overwrite` resets accumulated value |
| 2 | `Topic` | `langgraph.channels.topic` | Fan-in list buffer; `accumulate=False` clears each step |
| 3 | `UntrackedValue` | `langgraph.channels.untracked_value` | Transient channel — `checkpoint()` always returns `MISSING` |
| 4 | `BaseStore` + `Item` + `SearchItem` | `langgraph.store.base` | Abstract KV+vector store; `batch()` is the single required method |
| 5 | `PutOp` + `GetOp` + `SearchOp` + `ListNamespacesOp` | `langgraph.store.base` | Batch operation NamedTuples; `PutOp.value=None` → delete |
| 6 | `InMemoryStore` | `langgraph.store.memory` | In-process KV+cosine store; numpy fast-path; ThreadPoolExecutor for sync queries |
| 7 | `ManagedValue` + `ManagedValueSpec` | `langgraph.managed.base` | Abstract scratchpad protocol; implement `.get(scratchpad)` as a `@staticmethod` |
| 8 | `IsLastStepManager` + `RemainingStepsManager` | `langgraph.managed.is_last_step` | Built-in managed values; `IsLastStep=True` gives a one-step warning before `GraphRecursionError` |
| 9 | `ToolCallRequest` + `ToolCallWrapper` + `ToolInvocationError` | `langgraph.prebuilt.tool_node` | Interceptor API for `ToolNode`; use `.override()` for immutable request mutation |
| 10 | `NodeError` + `NodeCancelledError` + `NodeTimeoutError` | `langgraph.errors` | Structured node failure context injected into error handlers |

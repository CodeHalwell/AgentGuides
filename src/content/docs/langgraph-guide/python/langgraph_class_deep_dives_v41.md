---
title: "LangGraph Class Deep-Dives Vol. 41"
description: "Source-verified deep dives (langgraph==1.2.9) into 10 class groups: BinaryOperatorAggregate/Overwrite (operator.add reducer channel with per-step overwrite bypass, _get_overwrite/seen_overwrite guard, _operators_equal lambda detection), Topic (PubSub accumulate channel — accumulate flag, _flatten list-of-lists support, EmptyChannelError when empty, per-step buffer reset when accumulate=False), EphemeralValue (single-step value — guard=True single-write enforcement, MISSING sentinel on clear), NamedBarrierValue/NamedBarrierValueAfterFinish (fan-in synchronization — names set/seen set semantics, finish() deferred unlock, InvalidUpdateError on unknown names), AnyValue (multi-source convergence channel — last-write-wins, MISSING clear on zero updates), BaseCache/InMemoryCache (task result cache abstraction — FullKey namespace tuple, TTL-aware expiry, thread-safe RLock, serde layer), StreamChannel (drainable single-consumer queue — push/close/fail lifecycle, tee(n)/atee(n) fan-out, _bind sync/async mode, caller-driven pump wiring), ValuesTransformer/UpdatesTransformer (native stream projections for values/updates — scope filtering, interrupted/interrupts tracking, StreamChannel log push), IsLastStepManager/RemainingStepsManager/PregelScratchpad (managed step-count values — ManagedValue.get() scratchpad read, Annotated injection pattern, step/stop/counters dataclass), and build_serde_allowlist/curated_core_allowlist/apply_checkpointer_allowlist (_internal/_serde.py strict-msgpack security layer — Pydantic/dataclass/Enum/TypedDict traversal, BaseMessage curated set, with_allowlist checkpointer wrapping)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 41"
  order: 72
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.9` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `BinaryOperatorAggregate` · `Overwrite`

**Modules:** `langgraph.channels.binop`, `langgraph.types`

`BinaryOperatorAggregate` is the channel that powers every `Annotated[T, operator.add]` field in your state. When multiple nodes write to the same channel in a single super-step the channel calls `operator(current, new)` for each write in arrival order. It is the most widely used channel in LangGraph; `StateGraph` transparently creates one whenever it sees a reducer annotation.

**Key source facts** (from `langgraph/channels/binop.py`, `langgraph/types.py`):

- `__init__(typ, operator)` initializes `self.value` by calling `typ()`. Special abstract types (`Sequence`, `MutableSequence`, `Set`, `Mapping`, …) are mapped to their concrete counterparts (`list`, `set`, `dict`) before instantiation.
- `_operators_equal(a, b)` treats any lambda (`__name__ == "<lambda>"`) as equal to any other operator. This prevents false inequality when the same anonymous reducer is defined twice.
- `update(values)` iterates writes in order. Before the first write, if `self.value is MISSING`, the first value *becomes* the current value (no `operator` call). Subsequent writes fold in with `self.operator(self.value, value)`.
- `Overwrite` is a `dataclass` that wraps a replacement value. Detected by `_get_overwrite` in three forms: an `Overwrite` instance, a `{"__overwrite__": v}` dict, or a JSON-serialised `{"value": v, "type": "__overwrite__"}` dict. The last form preserves semantics across JSON API boundaries.
- Only **one** `Overwrite` per super-step is allowed; a second raises `InvalidUpdateError` with `ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE`.
- `checkpoint()` returns `self.value`; `from_checkpoint()` restores it — enabling time-travel across accumulation steps.

### Example 1 — custom numeric accumulator

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class Stats(TypedDict):
    total: Annotated[int, operator.add]
    items: Annotated[list[str], operator.add]

def node_a(state: Stats) -> Stats:
    return {"total": 10, "items": ["a", "b"]}

def node_b(state: Stats) -> Stats:
    return {"total": 5, "items": ["c"]}

g = StateGraph(Stats)
g.add_node("a", node_a)
g.add_node("b", node_b)
g.add_edge(START, "a")
g.add_edge(START, "b")
g.add_edge("a", END)
g.add_edge("b", END)

result = g.compile().invoke({"total": 0, "items": []})
print(result)
# {'total': 15, 'items': ['a', 'b', 'c']}
# Both nodes run in the same super-step; their writes are folded with operator.add
```

### Example 2 — `Overwrite` bypasses accumulation mid-run

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class Log(TypedDict):
    entries: Annotated[list[str], operator.add]

def add_entry(state: Log) -> Log:
    return {"entries": ["appended"]}

def reset_log(state: Log) -> Log:
    # Replace the entire list instead of appending
    return {"entries": Overwrite(["fresh start"])}

g = StateGraph(Log)
g.add_node("add", add_entry)
g.add_node("reset", reset_log)
g.add_edge(START, "add")
g.add_edge("add", "reset")
g.add_edge("reset", END)

result = g.compile().invoke({"entries": ["initial"]})
print(result)
# {'entries': ['fresh start']}
# Overwrite replaces rather than appending to the accumulated list
```

### Example 3 — inspect `BinaryOperatorAggregate` directly

```python
import operator
from langgraph.channels.binop import BinaryOperatorAggregate

# Build a channel that concatenates lists
ch = BinaryOperatorAggregate(list, operator.add)
ch.key = "items"

# Write two batches of values
ch.update([["a", "b"]])
print(ch.get())    # ['a', 'b']

ch.update([["c"], ["d", "e"]])
print(ch.get())    # ['a', 'b', 'c', 'd', 'e']

# Checkpoint / restore round-trip
saved = ch.checkpoint()
restored = ch.from_checkpoint(saved)
print(restored.get())    # ['a', 'b', 'c', 'd', 'e']
```

---

## 2 · `Topic`

**Module:** `langgraph.channels.topic`

`Topic` is a configurable PubSub channel. Multiple nodes can publish values to it in a single super-step and a downstream node receives the full collection as `Sequence[T]`. With `accumulate=False` (the default) the buffer resets after every step; with `accumulate=True` values build up across the entire run.

**Key source facts** (from `langgraph/channels/topic.py`):

- `update(values)` calls `_flatten(values)` which unwraps nested `list[Value]` items. This means a node can send either a single item or a list of items and both are appended flat.
- When `accumulate=False`, `update()` resets `self.values = list()` before appending — giving per-step semantics.
- `get()` raises `EmptyChannelError` when `self.values` is empty. `is_available()` returns `bool(self.values)`.
- `Topic` does not implement `consume()`. With `accumulate=False`, the buffer is cleared at the start of each super-step: `update()` resets `self.values` before appending, so items from the previous step are never visible alongside items written in the current step.
- `__eq__` compares only `accumulate` so two `Topic(int)` and `Topic(str)` channels with the same `accumulate` setting are considered equal by the graph's type checker.

### Example 1 — fan-in with Topic (per-step reset)

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END

# Using Annotated with Topic as the reducer
class State(TypedDict):
    events: Annotated[Sequence[str], Topic(str, accumulate=False)]

def publisher_a(state: State) -> dict:
    return {"events": "event-from-A"}

def publisher_b(state: State) -> dict:
    return {"events": "event-from-B"}

def consumer(state: State) -> dict:
    print("received:", list(state["events"]))
    return {}

g = StateGraph(State)
g.add_node("a", publisher_a)
g.add_node("b", publisher_b)
g.add_node("consumer", consumer)
g.add_edge(START, "a")
g.add_edge(START, "b")
g.add_edge("a", "consumer")
g.add_edge("b", "consumer")
g.add_edge("consumer", END)

g.compile().invoke({"events": []})
# received: ['event-from-A', 'event-from-B']
```

### Example 2 — accumulate=True builds history across steps

```python
from langgraph.channels.topic import Topic

ch = Topic(str, accumulate=True)
ch.key = "history"

ch.update(["step-1-nodeA", "step-1-nodeB"])
print(ch.get())    # ['step-1-nodeA', 'step-1-nodeB']

# Simulate next step — values are NOT reset because accumulate=True
ch.update(["step-2-nodeA"])
print(ch.get())    # ['step-1-nodeA', 'step-1-nodeB', 'step-2-nodeA']
```

### Example 3 — list batch publishing

```python
from langgraph.channels.topic import Topic

ch = Topic(int, accumulate=False)
ch.key = "numbers"

# A single update call can publish a flat int or a list of ints
ch.update([1, [2, 3], 4])   # _flatten unwraps the nested list
print(ch.get())     # [1, 2, 3, 4]

# Next step resets because accumulate=False
ch.update([99])
print(ch.get())     # [99]
```

---

## 3 · `EphemeralValue`

**Module:** `langgraph.channels.ephemeral_value`

`EphemeralValue` stores a value written in one step and **automatically clears** it at the start of the next step. This is ideal for passing transient signals between adjacent nodes without polluting the persistent state, and is the channel used internally by LangGraph for the `__start__` channel.

**Key source facts** (from `langgraph/channels/ephemeral_value.py`):

- `update(values)` with an **empty** `values` sequence clears the channel: `self.value = MISSING`. This is how LangGraph resets the channel at the next step — it calls `update([])` for channels that received no writes.
- When `guard=True` (the default), receiving more than one value in a single step raises `InvalidUpdateError`. Set `guard=False` when multiple nodes may race and you only care about one arbitrary winner.
- `checkpoint()` / `from_checkpoint()` persist the current value so an interrupted graph can resume without losing the ephemeral payload from the interrupted step.
- `get()` raises `EmptyChannelError` when `self.value is MISSING`.

### Example 1 — pass a transient signal between two sequential nodes

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    data: str
    signal: Annotated[str | None, EphemeralValue(str)]

def producer(state: State) -> dict:
    return {"signal": "go"}

def consumer(state: State) -> dict:
    print("signal received:", state.get("signal"))
    return {"data": "done"}

g = StateGraph(State)
g.add_node("producer", producer)
g.add_node("consumer", consumer)
g.add_edge(START, "producer")
g.add_edge("producer", "consumer")
g.add_edge("consumer", END)

g.compile().invoke({"data": "", "signal": None})
# signal received: go
```

### Example 2 — guard=False allows multiple concurrent writers

```python
from langgraph.channels.ephemeral_value import EphemeralValue

# guard=False: last writer wins, no error on multiple updates
ch = EphemeralValue(str, guard=False)
ch.key = "token"

ch.update(["from-A", "from-B"])    # no InvalidUpdateError
print(ch.get())    # 'from-B'   (last wins)

# Clear on empty update (simulates step boundary)
ch.update([])
print(ch.is_available())    # False
```

### Example 3 — checkpoint / resume with an ephemeral value

```python
from langgraph.channels.ephemeral_value import EphemeralValue

ch = EphemeralValue(dict)
ch.key = "payload"

ch.update([{"id": 42, "action": "process"}])
saved = ch.checkpoint()         # {'id': 42, 'action': 'process'}

# Simulate graph resume
restored = ch.from_checkpoint(saved)
print(restored.get())           # {'id': 42, 'action': 'process'}

# Next step produces no writes → channel clears
restored.update([])
print(restored.is_available())  # False
```

---

## 4 · `NamedBarrierValue` · `NamedBarrierValueAfterFinish`

**Module:** `langgraph.channels.named_barrier_value`

These channels implement **fan-in synchronization**. A `NamedBarrierValue` becomes available only after every name in the `names` set has been received at least once. `NamedBarrierValueAfterFinish` adds a deferred unlock: the barrier only fires after `finish()` is also called, enabling a pattern where a controller node signals "all upstream work is done" before the downstream node reads the result.

**Key source facts** (from `langgraph/channels/named_barrier_value.py`):

- `update(values)` adds each received value to `self.seen`. Receiving a value **not** in `self.names` raises `InvalidUpdateError` immediately.
- `get()` raises `EmptyChannelError` unless `self.seen == self.names`. It returns `None` — the channel's value type is `None` because it is a synchronization primitive, not a data channel.
- `consume()` resets `self.seen = set()` after firing, so the barrier can be re-armed for the next step.
- `NamedBarrierValueAfterFinish.finish()` sets `self.finished = True` only when `self.seen == self.names`. It returns `True` if it caused the barrier to become available, `False` otherwise.
- `from_checkpoint()` restores both `seen` (and `finished` for the After-Finish variant) so interrupted runs resume correctly.

### Example 1 — wait for two parallel nodes before proceeding

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue

barrier = NamedBarrierValue(str, names={"worker-A", "worker-B"})
barrier.key = "sync"

# Worker A finishes
barrier.update(["worker-A"])
print(barrier.is_available())   # False — still waiting for B

# Worker B finishes
barrier.update(["worker-B"])
print(barrier.is_available())   # True
print(barrier.get())            # None (sentinel: barrier fired)

# Reset for next round
barrier.consume()
print(barrier.is_available())   # False
```

### Example 2 — `NamedBarrierValueAfterFinish` deferred unlock

```python
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish

barrier = NamedBarrierValueAfterFinish(str, names={"task-1", "task-2"})
barrier.key = "done"

barrier.update(["task-1"])
barrier.update(["task-2"])
print(barrier.is_available())   # False — all names seen but finish() not called yet

fired = barrier.finish()
print(fired)                    # True — finish() caused unlock
print(barrier.is_available())   # True

barrier.consume()
print(barrier.is_available())   # False — reset for next cycle
```

### Example 3 — checkpoint/restore preserves barrier state

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue

barrier = NamedBarrierValue(str, names={"a", "b", "c"})
barrier.key = "three-way"

# Partially complete
barrier.update(["a", "b"])

# Save before third signal arrives (e.g. graph interrupted)
saved = barrier.checkpoint()    # {'a', 'b'}

# Later — restore and complete
restored = barrier.from_checkpoint(saved)
print(restored.is_available())  # False
restored.update(["c"])
print(restored.is_available())  # True
```

---

## 5 · `AnyValue`

**Module:** `langgraph.channels.any_value`

`AnyValue` is a relaxed channel that accepts **multiple concurrent writes** without complaint, assuming they are all equivalent. It stores the last received value. This is used internally for channels that are only ever written once per step (such as `__start__`) or for cases where multiple nodes may write the same computed value and you want to avoid an `InvalidUpdateError`.

**Key source facts** (from `langgraph/channels/any_value.py`):

- `update(values)` with a **non-empty** sequence stores `values[-1]` — the last received value. With an empty sequence it clears `self.value = MISSING`.
- There is no guard: `update(["x", "y", "z"])` quietly stores `"z"`.
- `__eq__` always returns `isinstance(value, AnyValue)` — two `AnyValue` channels are equal regardless of their type parameter. This is intentional: the channel makes no structural distinction.
- Unlike `LastValue`, `AnyValue` does **not** enforce the one-write-per-step rule. Use it only when you are confident concurrent writes are idempotent.

### Example 1 — multiple nodes write the same computed value

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.any_value import AnyValue
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    config: Annotated[str, AnyValue(str)]

def node_a(state: State) -> dict:
    return {"config": "shared-config-v1"}

def node_b(state: State) -> dict:
    return {"config": "shared-config-v1"}   # same value — no conflict

def sink(state: State) -> dict:
    print("config:", state["config"])
    return {}

g = StateGraph(State)
g.add_node("a", node_a)
g.add_node("b", node_b)
g.add_node("sink", sink)
g.add_edge(START, "a")
g.add_edge(START, "b")
g.add_edge("a", "sink")
g.add_edge("b", "sink")
g.add_edge("sink", END)

g.compile().invoke({"config": ""})
# config: shared-config-v1
```

### Example 2 — direct channel inspection

```python
from langgraph.channels.any_value import AnyValue
from langgraph.errors import EmptyChannelError

ch = AnyValue(str)
ch.key = "token"

print(ch.is_available())   # False

ch.update(["first", "second", "third"])
print(ch.get())            # 'third'  (last writer wins)

# Empty update clears the channel
ch.update([])
print(ch.is_available())   # False

try:
    ch.get()
except EmptyChannelError:
    print("channel is empty")
```

### Example 3 — checkpoint / restore

```python
from langgraph.channels.any_value import AnyValue

ch = AnyValue(dict)
ch.key = "context"

ch.update([{"user": "alice", "role": "admin"}])
saved = ch.checkpoint()          # {'user': 'alice', 'role': 'admin'}

restored = ch.from_checkpoint(saved)
print(restored.get())            # {'user': 'alice', 'role': 'admin'}
```

---

## 6 · `BaseCache` · `InMemoryCache`

**Modules:** `langgraph.cache.base`, `langgraph.cache.memory`

`BaseCache` is the abstract key-value cache consumed by `@task(cache_policy=...)`. When a task is called with a `CachePolicy`, LangGraph computes a `FullKey = (Namespace, str)` from the task's arguments (via the policy's `key_func`) and looks it up before executing the function. `InMemoryCache` is the built-in thread-safe in-process implementation.

**Key source facts** (from `langgraph/cache/base/__init__.py`, `langgraph/cache/memory/__init__.py`):

- `FullKey = tuple[Namespace, str]` where `Namespace = tuple[str, ...]`. The namespace is built from the task's identifier and the cache namespace constant `CACHE_NS_WRITES`.
- `BaseCache.serde` is a `SerializerProtocol`; default is `JsonPlusSerializer(pickle_fallback=False)`. Custom caches can swap the serialiser by passing `serde=` to `__init__`.
- `InMemoryCache._cache` is a `dict[Namespace, dict[str, tuple[enc, bytes, expiry]]]`. The inner tuple is `(encoding_str, serialised_bytes, unix_timestamp_or_None)`.
- TTL expiry is checked on `get()` — stale entries are deleted lazily on read, not proactively via a background thread.
- All write operations acquire `self._lock` (a `threading.RLock`), making reads/writes safe from multiple OS threads.
- `aget()` / `aset()` / `aclear()` delegate to their sync counterparts — `InMemoryCache` is a sync cache wrapped with trivially async methods. For a true async cache, subclass `BaseCache` and provide native coroutines.
- `clear(namespaces=None)` with no argument purges the entire cache; passing a list of `Namespace` tuples removes only those namespaces.

### Example 1 — cache task results with `@task` and `CachePolicy`

```python
import asyncio
from langgraph.func import task, entrypoint
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()

@task(cache_policy=CachePolicy(ttl=60))
def expensive_lookup(query: str) -> str:
    print(f"  [MISS] computing for: {query}")
    return f"result-for-{query}"

@entrypoint(cache=cache)
def pipeline(query: str) -> str:
    return expensive_lookup(query).result()

# First call hits the function
r1 = pipeline.invoke("hello")
print(r1)   # [MISS] computing for: hello / result-for-hello

# Second call is served from cache (no [MISS] printed)
r2 = pipeline.invoke("hello")
print(r2)   # result-for-hello
```

### Example 2 — use `InMemoryCache` directly

```python
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()

ns = ("my_app", "embeddings")
key = "doc-42"

# Store with a 30-second TTL
cache.set({(ns, key): ({"vec": [0.1, 0.2]}, 30)})

result = cache.get([(ns, key)])
print(result)
# {(('my_app', 'embeddings'), 'doc-42'): {'vec': [0.1, 0.2]}}

# Clear just this namespace
cache.clear([ns])
print(cache.get([(ns, key)]))   # {}
```

### Example 3 — implement a custom `BaseCache` with a size cap

```python
import threading
from collections import OrderedDict
from langgraph.cache.base import BaseCache, FullKey, Namespace

class LRUCache(BaseCache):
    """Thread-safe LRU cache with a maximum number of entries."""

    def __init__(self, maxsize: int = 128):
        super().__init__()
        self._maxsize = maxsize
        self._store: OrderedDict[FullKey, tuple] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, keys):
        import datetime
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            out = {}
            for k in keys:
                if k in self._store:
                    enc, val, expiry = self._store[k]
                    if expiry is None or now < expiry:
                        self._store.move_to_end(k)   # mark as recently used
                        out[k] = self.serde.loads_typed((enc, val))
                    else:
                        del self._store[k]           # lazy TTL eviction
            return out

    async def aget(self, keys):
        return self.get(keys)

    def set(self, pairs):
        import datetime
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            for k, (value, ttl) in pairs.items():
                expiry = (now + ttl) if ttl is not None else None
                self._store[k] = (*self.serde.dumps_typed(value), expiry)
                self._store.move_to_end(k)
                if len(self._store) > self._maxsize:
                    self._store.popitem(last=False)   # evict oldest

    async def aset(self, pairs):
        self.set(pairs)

    def clear(self, namespaces=None):
        with self._lock:
            if namespaces is None:
                self._store.clear()
            else:
                ns_set = set(namespaces)
                to_del = [k for k in self._store if k[0] in ns_set]
                for k in to_del:
                    del self._store[k]

    async def aclear(self, namespaces=None):
        self.clear(namespaces)

lru = LRUCache(maxsize=2)
lru.set({(("ns",), "a"): ("value-a", None)})
lru.set({(("ns",), "b"): ("value-b", None)})
lru.set({(("ns",), "c"): ("value-c", None)})   # evicts 'a' (LRU)
print(lru.get([(("ns",), "a")]))   # {}  (evicted)
print(lru.get([(("ns",), "c")]))   # {(('ns',), 'c'): 'value-c'}
```

---

## 7 · `StreamChannel`

**Module:** `langgraph.stream.stream_channel`

`StreamChannel[T]` is the single-consumer drainable queue that underpins every stream projection in the v3 streaming API. Each native transformer (`ValuesTransformer`, `UpdatesTransformer`, etc.) owns one or more `StreamChannel` instances — `run.values`, `run.updates`, `run.custom` — that the caller iterates as `for item in run.values` or `async for item in run.values`.

**Key source facts** (from `langgraph/stream/stream_channel.py`):

- Items are stored as `(stamp, item)` tuples. The stamp is a monotonic counter from the owning `StreamMux`. Iterators strip stamps and yield only the item.
- `push(item)` appends to `_items` **only when subscribed**. Auto-forwarding via `_wire_fn` always fires regardless of subscription, so wired channels inject into the main event log unconditionally.
- `_bind(is_async=True/False)` locks the channel into sync or async iteration mode. A second `_bind` call raises `RuntimeError`. Attempting to `async for` a sync-bound channel (or vice versa) raises `TypeError`.
- `close()` marks the channel exhausted; `fail(err)` marks it exhausted *and* stores an error raised on the next iteration.
- `tee(n)` / `atee(n)` subscribe exactly once and fan out to `n` independent iterators via per-branch `deque` buffers. A shared `asyncio.Lock` in `atee` serialises fetch-from-source so exactly one coroutine pulls the next item.
- Memory is caller-driven: iterating one item at a time calls `_request_more()` / `await _arequest_more()` before blocking, keeping the buffer at most one item deep for a single-step-ahead consumer.

### Example 1 — iterate a `StreamChannel` via `tee` fan-out

```python
# Note: StreamChannel is normally bound and driven by StreamMux.
# This example calls semi-private internals (_bind, _items, _closed) to
# demonstrate tee() behaviour in isolation — production code never
# manipulates these directly.

from langgraph.stream.stream_channel import StreamChannel

ch: StreamChannel[str] = StreamChannel()
ch._bind(is_async=False)      # bind to sync mode (done by StreamMux normally)

# tee() subscribes the channel and returns n independent iterators
it1, it2 = ch.tee(2)

# Directly populate the buffer (simulating StreamMux-driven push calls)
ch._items.extend([(0, "alpha"), (1, "beta"), (2, "gamma")])
ch._closed = True             # signal end-of-stream

# Both branches drain independently
print(list(it1))   # ['alpha', 'beta', 'gamma']
print(list(it2))   # ['alpha', 'beta', 'gamma']
```

### Example 2 — `fail()` propagates errors to the consumer

```python
from langgraph.stream.stream_channel import StreamChannel

ch: StreamChannel[int] = StreamChannel()
ch._bind(is_async=False)
# _subscribed must be False before __iter__ — the iterator sets it to True.
# Pre-populating _items directly bypasses push() so we can seed the buffer
# without a running mux.
ch._closed = False

ch._items.extend([(0, 1), (1, 2)])
ch.fail(ValueError("upstream error"))

results = []
try:
    for item in ch:   # __iter__ sets _subscribed=True, then drains _items
        results.append(item)
except ValueError as e:
    print(f"caught: {e}")
print(f"got before error: {results}")
# got before error: [1, 2]
# caught: upstream error
```

### Example 3 — async fan-out with `atee`

```python
import asyncio
from langgraph.stream.stream_channel import StreamChannel

async def main():
    ch: StreamChannel[str] = StreamChannel()
    ch._bind(is_async=True)
    ch._subscribed = False
    ch._closed = False

    a1, a2 = ch.atee(2)

    # Simulate async item production
    async def produce():
        for i, word in enumerate(["one", "two", "three"]):
            ch._items.append((i, word))
        ch._closed = True

    await produce()

    async def drain(it, label):
        return [item async for item in it]

    results = await asyncio.gather(drain(a1, "A"), drain(a2, "B"))
    print("branch A:", results[0])
    print("branch B:", results[1])

asyncio.run(main())
# branch A: ['one', 'two', 'three']
# branch B: ['one', 'two', 'three']
```

---

## 8 · `ValuesTransformer` · `UpdatesTransformer`

**Module:** `langgraph.stream.transformers`

`ValuesTransformer` and `UpdatesTransformer` are two of the five **native** transformers that expose `stream_mode="values"` and `stream_mode="updates"` data as typed `StreamChannel` projections on the `GraphRunStream` / `AsyncGraphRunStream` handle.

**Key source facts** (from `langgraph/stream/transformers.py`):

- Both inherit `StreamTransformer` and set `_native = True` — LangGraph binds their channels as direct attributes on the run stream (`run.values`, `run.updates`).
- `required_stream_modes` tells the `StreamMux` which underlying stream modes to activate. `ValuesTransformer` requires `("values",)`; `UpdatesTransformer` requires `("updates",)`.
- **Pass classes, not instances.** `_normalize_stream_transformer_factories` (called internally by `stream_events`) **rejects pre-built `StreamTransformer` instances** with `TypeError`. Always pass the class (e.g. `ValuesTransformer`) so the mux can call `factory(scope)` for each subgraph namespace independently.
- **Scope filtering**: both transformers compare `event["params"]["namespace"]` against `self._scope_list` (a `list[str]` copy of the tuple scope). Events from deeper subgraphs are left in the main event log but not pushed to the projection.
- `ValuesTransformer` also tracks `_interrupted` and `_interrupts`: whenever a `values` event carries `params["interrupts"]`, the transformer caches them so `run.interrupted` / `run.interrupts` stay in sync without a second event.
- `UpdatesTransformer` is simpler — it pushes `params["data"]` straight to `self._log` (a `StreamChannel[dict[str, Any]]`).
- `init()` returns the projection dict (`{"values": self._log}` or `{"updates": self._log}`) that the mux attaches as attributes on the run stream.

### Example 1 — consume `run.values` with `stream_events(version="v3")`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream.transformers import ValuesTransformer

class State(TypedDict):
    count: int

def increment(state: State) -> State:
    return {"count": state["count"] + 1}

g = StateGraph(State)
g.add_node("inc", increment)
g.add_edge(START, "inc")
g.add_edge("inc", END)
compiled = g.compile()

# Pass the class, not an instance — the mux calls ValuesTransformer(scope)
with compiled.stream_events(
    {"count": 0},
    version="v3",
    transformers=[ValuesTransformer],
) as run:
    for snapshot in run.values:
        print("values snapshot:", snapshot)
# values snapshot: {'count': 1}
```

### Example 2 — consume `run.updates` to see per-node diffs

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream.transformers import UpdatesTransformer

class State(TypedDict):
    x: int
    y: int

def step_a(state: State) -> dict:
    return {"x": state["x"] + 10}

def step_b(state: State) -> dict:
    return {"y": state["y"] + 20}

g = StateGraph(State)
g.add_node("a", step_a)
g.add_node("b", step_b)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)
compiled = g.compile()

with compiled.stream_events(
    {"x": 0, "y": 0},
    version="v3",
    transformers=[UpdatesTransformer],   # pass class, not instance
) as run:
    for update in run.updates:
        print("node update:", update)
# node update: {'a': {'x': 10}}
# node update: {'b': {'y': 20}}
```

### Example 3 — consume updates and values concurrently

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream.transformers import ValuesTransformer, UpdatesTransformer

class State(TypedDict):
    messages: list[str]

def add_greeting(state: State) -> dict:
    return {"messages": state["messages"] + ["hello"]}

def add_farewell(state: State) -> dict:
    return {"messages": state["messages"] + ["goodbye"]}

g = StateGraph(State)
g.add_node("greet", add_greeting)
g.add_node("farewell", add_farewell)
g.add_edge(START, "greet")
g.add_edge("greet", "farewell")
g.add_edge("farewell", END)
compiled = g.compile()

# Two channels must be consumed concurrently — if you drain one first,
# the second channel's push() calls are no-ops (not yet subscribed) and
# all events are lost. astream_events + asyncio.gather solves this.
async def main():
    async with await compiled.astream_events(
        {"messages": []},
        version="v3",
        transformers=[ValuesTransformer, UpdatesTransformer],   # classes, not instances
    ) as run:
        async def drain_updates():
            async for upd in run.updates:
                print("update:", upd)

        async def drain_values():
            async for val in run.values:
                print("final state:", val)

        await asyncio.gather(drain_updates(), drain_values())

asyncio.run(main())
# update: {'greet': {'messages': ['hello']}}
# update: {'farewell': {'messages': ['hello', 'goodbye']}}
# final state: {'messages': ['hello', 'goodbye']}
```

---

## 9 · `IsLastStepManager` · `RemainingStepsManager` · `PregelScratchpad`

**Modules:** `langgraph.managed.is_last_step`, `langgraph._internal._scratchpad`

Managed values are singletons injected by the Pregel runtime into a node's state dict without being stored in the checkpoint. `IsLastStep` and `RemainingSteps` are the two built-in managed values; they read from `PregelScratchpad` — a `dataclass` attached to each execution context that carries the step counter and control callables.

**Key source facts** (from `langgraph/managed/is_last_step.py`, `langgraph/_internal/_scratchpad.py`):

- `ManagedValue.get(scratchpad)` is the single `@staticmethod` every managed value must implement. The runtime calls it once per step and injects the result into the state dict under the annotated key.
- `IsLastStepManager.get(scratchpad)` returns `scratchpad.step == scratchpad.stop - 1`. `step` is zero-indexed; `stop` is `recursion_limit + 1`.
- `RemainingStepsManager.get(scratchpad)` returns `scratchpad.stop - scratchpad.step` — the number of steps remaining including the current one.
- `PregelScratchpad` fields: `step` (int, current step), `stop` (int, exclusive upper bound), `call_counter` / `interrupt_counter` / `subgraph_counter` (callables that return monotonically increasing IDs for calls, interrupts, and subgraphs), `get_null_resume` (callable for generating null resume placeholders), `resume` (list of resume values from `Command(resume=...)`).
- The `Annotated[bool, IsLastStepManager]` pattern is the canonical way to declare a managed value field. `StateGraph` detects the second `Annotated` argument is a `ManagedValueSpec` and excludes the field from the regular channel map.

### Example 1 — guard against infinite loops with `IsLastStep`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep

class State(TypedDict):
    count: int
    is_last: IsLastStep    # injected by Pregel, never in checkpoint

def loop_node(state: State) -> dict | None:
    if state["is_last"]:
        print(f"Reached last step at count={state['count']} — stopping")
        return None
    print(f"step count={state['count']}")
    return {"count": state["count"] + 1}

g = StateGraph(State)
g.add_node("loop", loop_node)
g.add_edge(START, "loop")
g.add_conditional_edges(
    "loop",
    lambda s: "loop" if not s["is_last"] else END,
)

g.compile().invoke({"count": 0}, {"recursion_limit": 5})
# step count=0 / step count=1 / ... / Reached last step at count=4 — stopping
```

### Example 2 — countdown with `RemainingSteps`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import RemainingSteps

class State(TypedDict):
    remaining: RemainingSteps   # managed — counts down automatically

def status_node(state: State) -> dict:
    print(f"  remaining steps: {state['remaining']}")
    return {}

g = StateGraph(State)
g.add_node("status", status_node)
g.add_edge(START, "status")
g.add_edge("status", END)

g.compile().invoke({}, {"recursion_limit": 10})
# remaining steps: 10
```

### Example 3 — custom `ManagedValue` using `PregelScratchpad`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.graph import StateGraph, START, END

class StepFractionManager(ManagedValue[float]):
    """Injects how far through the recursion limit we are (0.0 → 1.0)."""
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> float:
        return scratchpad.step / max(scratchpad.stop - 1, 1)

StepFraction = Annotated[float, StepFractionManager]

class State(TypedDict):
    fraction: StepFraction

def show_fraction(state: State) -> dict:
    print(f"  progress: {state['fraction']:.0%}")
    return {}

g = StateGraph(State)
g.add_node("show", show_fraction)
g.add_edge(START, "show")
g.add_edge("show", END)

g.compile().invoke({}, {"recursion_limit": 4})
# progress: 0%
```

---

## 10 · `build_serde_allowlist` · `curated_core_allowlist` · `apply_checkpointer_allowlist`

**Module:** `langgraph._internal._serde`

When a `StateGraph` is compiled with a checkpointer that supports strict msgpack serialisation (`with_allowlist`), LangGraph introspects the graph's schema and channel types to build a **type allowlist** — a set of `(module, class_name)` tuples. The checkpointer then refuses to deserialise any type not on this list, preventing arbitrary-class deserialisation attacks.

**Key source facts** (from `langgraph/_internal/_serde.py`):

- `curated_core_allowlist()` returns a hardcoded set of `(module, name)` tuples for all `langchain_core.messages` classes (`HumanMessage`, `AIMessage`, `ToolMessage`, etc.). These are always safe.
- `build_serde_allowlist(schemas, channels)` merges `curated_core_allowlist()` with the result of `collect_allowlist_from_schemas()`. `schemas` is the list of graph input/output types; `channels` is the compiled channel dict.
- `collect_allowlist_from_schemas` calls `_collect_from_type(typ, ...)` recursively, traversing: Pydantic models (`model_fields` or `__fields__`), dataclasses (`dataclasses.fields`), `TypedDict` (`get_type_hints`), `Enum` subclasses, `Union`/`Annotated`/`Required`/`NotRequired` generics, and standard containers (`list`, `set`, `dict`, `deque`, `frozenset`).
- `apply_checkpointer_allowlist(checkpointer, allowlist)` calls `checkpointer.with_allowlist(allowlist)` if `BaseCheckpointSaver` supports it. If the checkpointer is `True`, `False`, `None`, or an unsupported version, it is returned unchanged (with a one-time warning for the unsupported case).
- `STRICT_MSGPACK_ENABLED` is `True` when `langgraph.checkpoint.serde._msgpack` is importable — indicating the checkpoint backend supports the strict path.

### Example 1 — inspect the allowlist for a simple graph

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph._internal._serde import build_serde_allowlist

class UserMessage(TypedDict):
    role: str
    content: str

class AppState(TypedDict):
    messages: list[UserMessage]
    step: int

def noop(state: AppState) -> dict:
    return {}

g = StateGraph(AppState)
g.add_node("noop", noop)
g.add_edge(START, "noop")
g.add_edge("noop", END)
compiled = g.compile()

# Build what the allowlist would contain
allowlist = build_serde_allowlist(
    schemas=[AppState],
    channels=compiled.channels,
)

# Filter to app-specific entries (exclude langchain_core built-ins)
app_entries = {entry for entry in allowlist if "langchain" not in entry[0]}
print(sorted(app_entries))
# Note: TypedDicts don't produce allowlist entries (they use standard dicts at runtime)
# Pydantic models and dataclasses would appear here
```

### Example 2 — Pydantic model traversal populates the allowlist

```python
from pydantic import BaseModel
from langgraph._internal._serde import collect_allowlist_from_schemas

class Address(BaseModel):
    street: str
    city: str

class User(BaseModel):
    name: str
    address: Address
    tags: list[str]

result = collect_allowlist_from_schemas(schemas=[User])

# Both User and Address are collected (nested model traversal)
names = {name for module, name in result if "langchain" not in module}
print(sorted(names))
# ['Address', 'User']
```

### Example 3 — `curated_core_allowlist` always includes LangChain message types

```python
from langgraph._internal._serde import curated_core_allowlist

allowlist = curated_core_allowlist()
names = {name for _, name in allowlist}
print(sorted(names))
# ['AIMessage', 'AIMessageChunk', 'BaseMessage', 'BaseMessageChunk',
#  'ChatMessage', 'ChatMessageChunk', 'FunctionMessage', 'FunctionMessageChunk',
#  'HumanMessage', 'HumanMessageChunk', 'RemoveMessage', 'SystemMessage',
#  'SystemMessageChunk', 'ToolMessage', 'ToolMessageChunk']
```

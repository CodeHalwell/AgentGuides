---
title: "Class deep-dives Vol. 25 — cache architecture, node protocols, stream bridges & internal constants (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented class groups in LangGraph 1.2.6: BaseCache+FullKey+Namespace (pluggable cache ABC — 6-method sync/async contract, tuple-addressed key space, custom serde injection, implementing a TTL-aware Redis backend), StateNode union type + full Protocol family (all 9 add_node() Protocol variants — _Node through _NodeWithRuntime, StateNodeSpec dataclass defer/ends/error_handler_node fields), _freeze+default_cache_key (cache key canonicalization — depth-limited deep-freeze, dict key-sorting, numpy escape via .tobytes(), protocol-5 pickle determinism), UUID v6+uuid6 (checkpoint ID generation — monotonic timestamp encoding for DB index locality, _last_v6_timestamp monotonic guard, subsec fractional field, why v6 beats v4 for checkpoints), convert_to_protocol_event (v2→v3 stream bridge — StreamPart→ProtocolEvent conversion, interrupts passthrough, wall-clock timestamp vs monotonic seq), default_retry_on (default RetryPolicy retry predicate — allowlist of non-retryable builtins, httpx/requests 5xx detection, retry-everything-not-in-blocklist semantics), internal Pregel constants (sys.intern() reserved strings — INPUT/INTERRUPT/RESUME/ERROR/TASKS/RETURN, CONFIG_KEY_SEND/READ/CALL/CHECKPOINTER/STREAM, their roles in the write protocol and config.configurable routing), map_debug_tasks+is_multiple_channel_write (debug stream helpers — metadata key filtering, TAG_HIDDEN suppression, multi-write payload detection), ensure_message_ids+_is_message_dict+_state_values (pre-checkpoint message coercion — ID stamping before background thread, OpenAI-role/LangChain-type detection, in-place list mutation, dict/BaseModel/dataclass value extraction), and coerce_timeout_policy+sync_timeout_unsupported (TimeoutPolicy normalization — float/timedelta/policy coercion, sync-cancellation hard limit, async node pattern for safe timeout support)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 25"
  order: 56
---

# Class deep-dives Vol. 25 — cache architecture, node protocols, stream bridges & internal constants (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---|---|
| 1 | `BaseCache` + `FullKey` + `Namespace` | `langgraph.cache.base` |
| 2 | `StateNode` union + Protocol family (`_Node` … `_NodeWithRuntime`) | `langgraph.graph._node` |
| 3 | `_freeze` + `default_cache_key` | `langgraph._internal._cache` |
| 4 | `UUID` v6 + `uuid6` | `langgraph.checkpoint.base.id` |
| 5 | `convert_to_protocol_event` | `langgraph.stream._convert` |
| 6 | `default_retry_on` | `langgraph._internal._retry` |
| 7 | Internal Pregel constants | `langgraph._internal._constants` |
| 8 | `map_debug_tasks` + `is_multiple_channel_write` | `langgraph.pregel.debug` |
| 9 | `ensure_message_ids` + `_is_message_dict` + `_state_values` | `langgraph.pregel._messages` |
| 10 | `coerce_timeout_policy` + `sync_timeout_unsupported` | `langgraph._internal._timeout` |

---

## 1 · `BaseCache` + `FullKey` + `Namespace`

**Module**: `langgraph.cache.base`  
**First dedicated coverage.**

`BaseCache` is the abstract base class for every cache backend in LangGraph. It defines a 6-method sync/async contract with tuple-addressed keys. The two built-in concrete implementations (`InMemoryCache` and `RedisCache`) follow this contract; if neither fits your deployment you can drop in a replacement by subclassing `BaseCache`.

```python
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ValueT = TypeVar("ValueT")
Namespace = tuple[str, ...]          # e.g. ("my_task", "user_123")
FullKey = tuple[Namespace, str]      # (namespace, cache_key_str)


class BaseCache(ABC, Generic[ValueT]):
    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=False)

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        self.serde = serde or self.serde

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

Key design decisions:

| Aspect | Detail |
|--------|--------|
| `FullKey` | A `(Namespace, str)` tuple. `Namespace` is itself a tuple of strings that acts like a path prefix, enabling per-task and per-user isolation without separate cache instances. |
| `ValueT` (bytes) | The concrete type parameter for built-ins is `bytes`. The cache stores serialized blobs; `BaseCache.serde` handles serialization so the storage layer stays format-agnostic. |
| `set` value tuple | `(ValueT, int | None)` — the second element is a TTL in seconds, or `None` for no expiry. |
| Sync/async symmetry | Both lanes are required. In-memory backends implement async as trivial wrappers; production backends (Redis, Memcached) implement the async path natively. |

### Example 1 — Implementing an in-process dict cache

```python
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.cache.base import BaseCache, FullKey, Namespace


class DictCache(BaseCache[bytes]):
    """Minimal in-process cache for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[FullKey, bytes] = {}

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, bytes]:
        return {k: self._store[k] for k in keys if k in self._store}

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, bytes]:
        return self.get(keys)

    def set(self, pairs: Mapping[FullKey, tuple[bytes, int | None]]) -> None:
        for key, (value, _ttl) in pairs.items():
            self._store[key] = value  # TTL ignored for demo

    async def aset(self, pairs: Mapping[FullKey, tuple[bytes, int | None]]) -> None:
        self.set(pairs)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        if namespaces is None:
            self._store.clear()
        else:
            ns_set = set(namespaces)
            self._store = {k: v for k, v in self._store.items() if k[0] not in ns_set}

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        self.clear(namespaces)


# Wire into CachePolicy via compile()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy

class State(TypedDict):
    value: int

cache = DictCache()

def expensive_node(state: State) -> dict:
    print("  [computed]")
    return {"value": state["value"] * 2}

builder = StateGraph(State)
builder.add_node("expensive", expensive_node, cache_policy=CachePolicy())
builder.add_edge(START, "expensive")
builder.add_edge("expensive", END)
graph = builder.compile(cache=cache)

result1 = graph.invoke({"value": 5})   # prints [computed]
result2 = graph.invoke({"value": 5})   # cache hit — no print
print(result1, result2)                # both {value: 10}
```

### Example 2 — Namespace isolation for per-user caches

```python
from langgraph.cache.base import FullKey, Namespace

ns_alice: Namespace = ("summarize_task", "user_alice")
ns_bob: Namespace   = ("summarize_task", "user_bob")

key_alice: FullKey = (ns_alice, "abc123")
key_bob:   FullKey = (ns_bob,   "abc123")

# Same cache_key_str ("abc123"), different namespaces → independent entries.
# Clearing ns_alice leaves ns_bob untouched.
cache = DictCache()
cache.set({
    key_alice: (b"alice result", None),
    key_bob:   (b"bob result",   None),
})
cache.clear(namespaces=[ns_alice])
print(cache.get([key_alice, key_bob]))
# {(('summarize_task', 'user_bob'), 'abc123'): b'bob result'}
```

### Example 3 — Custom serde injection

```python
import pickle
from langgraph.checkpoint.serde.base import SerializerProtocol


class PickleSerializer(SerializerProtocol):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "pickle", pickle.dumps(obj, protocol=5)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        _, raw = data
        return pickle.loads(raw)


# Inject via constructor — swaps JsonPlusSerializer for pickle on all operations
cache = DictCache()
cache.serde = PickleSerializer()

key: FullKey = (("my_ns",), "demo")
serialized, _ = cache.serde.dumps_typed({"complex": {"nested": [1, 2, 3]}})
print(serialized)  # "pickle"
```

---

## 2 · `StateNode` union type + Protocol family

**Module**: `langgraph.graph._node`  
**First dedicated coverage (Protocol family).**

Every callable you pass to `add_node()` must match the `StateNode` type alias — a union of 9 Protocol variants. Understanding this union tells you precisely which keyword arguments your node function may declare and which combinations the type checker will accept.

```python
from typing import TypeAlias
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter

StateNode: TypeAlias = (
    _Node[NodeInputT]                   # (state) -> Any
    | _NodeWithConfig[NodeInputT]       # (state, config) -> Any
    | _NodeWithWriter[NodeInputT]       # (state, *, writer) -> Any
    | _NodeWithStore[NodeInputT]        # (state, *, store) -> Any
    | _NodeWithWriterStore[NodeInputT]  # (state, *, writer, store) -> Any
    | _NodeWithConfigWriter[NodeInputT] # (state, *, config, writer) -> Any
    | _NodeWithConfigStore[NodeInputT]  # (state, *, config, store) -> Any
    | _NodeWithConfigWriterStore[NodeInputT]  # (state, *, config, writer, store) -> Any
    | _NodeWithRuntime[NodeInputT, ContextT]  # (state, *, runtime) -> Any
    | Runnable[NodeInputT, Any]
)
```

`StateNodeSpec` is the compiled representation of a node after `compile()`:

```python
from dataclasses import dataclass
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

@dataclass(slots=True)
class StateNodeSpec:
    runnable: StateNode         # the compiled RunnableCallable wrapper
    metadata: dict | None
    input_schema: type          # narrowed input type for this node
    retry_policy: RetryPolicy | list[RetryPolicy] | None
    cache_policy: CachePolicy | None
    is_error_handler: bool = False
    error_handler_node: str | None = None
    ends: tuple[str, ...] | dict[str, str] | None = ()  # declared outgoing edges
    defer: bool = False         # if True, node runs after all non-deferred nodes
    timeout: TimeoutPolicy | None = None
```

### Example 1 — All 4 common Protocol variants

```python
from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter
from langgraph.store.base import BaseStore

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    counter: int

# Variant 1: plain node
def plain(state: State) -> dict:
    return {"counter": state["counter"] + 1}

# Variant 2: with RunnableConfig (access thread_id, tags, etc.)
def with_config(state: State, config: RunnableConfig) -> dict:
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    print(f"thread={thread_id}")
    return {"counter": state["counter"] + 1}

# Variant 3: with StreamWriter (emit custom events mid-node)
def with_writer(state: State, *, writer: StreamWriter) -> dict:
    writer({"progress": "halfway"})
    return {"counter": state["counter"] + 1}

# Variant 4: with BaseStore (long-term memory lookup)
def with_store(state: State, *, store: BaseStore) -> dict:
    item = store.get(("memory",), "user_pref")
    theme = item.value.get("theme", "default") if item else "default"
    print(f"theme={theme}")
    return {"counter": state["counter"] + 1}

builder = StateGraph(State)
builder.add_node("plain", plain)
builder.add_node("config", with_config)
builder.add_node("writer", with_writer)
builder.add_node("store", with_store)
builder.add_edge(START, "plain")
builder.add_edge("plain", "config")
builder.add_edge("config", "writer")
builder.add_edge("writer", "store")
builder.add_edge("store", END)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
graph = builder.compile(
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)
result = graph.invoke(
    {"messages": [], "counter": 0},
    config={"configurable": {"thread_id": "t1"}},
)
print(result["counter"])  # 4
```

### Example 2 — `_NodeWithRuntime` for typed context injection

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

class AppContext(TypedDict):
    user_name: str
    role: str

class State(TypedDict):
    greeting: str

# _NodeWithRuntime: receives the full Runtime object, including typed context
def greet(state: State, *, runtime: Runtime[AppContext]) -> dict:
    ctx = runtime.context  # typed as AppContext
    return {"greeting": f"Hello, {ctx['user_name']} ({ctx['role']})!"}

builder = StateGraph(State, context_schema=AppContext)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)
graph = builder.compile()

result = graph.invoke(
    {"greeting": ""},
    context={"user_name": "Alice", "role": "admin"},
)
print(result["greeting"])  # Hello, Alice (admin)!
```

### Example 3 — Inspecting `StateNodeSpec` after compile

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy

class State(TypedDict):
    val: int

def my_node(state: State) -> dict:
    return {"val": state["val"] + 1}

builder = StateGraph(State)
builder.add_node(
    "my_node",
    my_node,
    retry_policy=RetryPolicy(max_attempts=3),
    cache_policy=CachePolicy(),
)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()

# Access the StateNodeSpec from the builder (before compilation)
spec = builder.nodes["my_node"]
print(type(spec))                 # <class 'StateNodeSpec'>
print(spec.retry_policy)          # RetryPolicy(max_attempts=3)
print(spec.cache_policy)          # CachePolicy(...)
print(spec.is_error_handler)      # False
print(spec.defer)                 # False
print(spec.ends)                  # ('__end__',)
```

---

## 3 · `_freeze` + `default_cache_key`

**Module**: `langgraph._internal._cache`  
**First dedicated coverage.**

When you use `CachePolicy()` without specifying a `key_func`, LangGraph calls `default_cache_key(*args, **kwargs)` to produce a deterministic cache key from the node's inputs. The key ingredient is `_freeze()` — a depth-limited recursive function that converts any Python object into a hashable, order-stable representation.

```python
from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if isinstance(obj, Hashable) or depth <= 0:
        return obj
    elif isinstance(obj, Mapping):
        # sort keys so {"a":1,"b":2} == {"b":2,"a":1}
        return tuple(sorted((k, _freeze(v, depth - 1)) for k, v in obj.items()))
    elif isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    elif hasattr(obj, "tobytes"):
        # numpy / pandas arrays
        return (type(obj).__name__, obj.tobytes(),
                obj.shape if hasattr(obj, "shape") else None)
    return obj


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    import pickle
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
```

`_freeze` design decisions:

| Input type | Freeze strategy |
|-----------|-----------------|
| Already `Hashable` (int, str, tuple, frozenset…) | Pass through untouched |
| `Mapping` (dict, OrderedDict…) | Sort by key → tuple of `(k, freeze(v))` pairs |
| Non-str `Sequence` (list, deque…) | Convert to tuple of frozen items |
| Array-like with `.tobytes()` (numpy, torch) | `(class_name, bytes, shape)` 3-tuple |
| Anything else at max depth | Pass through as-is |

`protocol=5` in `default_cache_key` was introduced in Python 3.8 and offers better performance than older protocols, especially for large buffers. `fix_imports=False` prevents mapping Python 2 module names, which would make the pickle slightly smaller and faster to compute.

### Example 1 — Key stability for dicts with different insertion orders

```python
from langgraph._internal._cache import _freeze, default_cache_key

d1 = {"b": 2, "a": 1}
d2 = {"a": 1, "b": 2}

assert _freeze(d1) == _freeze(d2)   # True — sorted keys
assert default_cache_key(d1) == default_cache_key(d2)  # True

# Nested dicts are also sorted at every level
nested1 = {"outer": {"z": 9, "a": 1}}
nested2 = {"outer": {"a": 1, "z": 9}}
assert default_cache_key(nested1) == default_cache_key(nested2)
```

### Example 2 — Numpy arrays produce stable byte fingerprints

```python
import numpy as np
from langgraph._internal._cache import _freeze

arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
frozen = _freeze(arr)
print(frozen)
# ('ndarray', b'\x00\x00\x80?\x00\x00\x00@...', (2, 2))

# Two identically valued arrays produce the same frozen key
arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
assert _freeze(arr) == _freeze(arr2)
```

### Example 3 — Custom `key_func` that delegates to `default_cache_key`

```python
from langgraph._internal._cache import default_cache_key
from langgraph.types import CachePolicy
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    user_id: str

def user_scoped_key(state: State) -> bytes:
    # Include user_id in the namespace so each user has isolated cache entries.
    # Delegates normalization to default_cache_key.
    return default_cache_key(state["query"], user_id=state["user_id"])

def expensive_search(state: State) -> dict:
    print(f"  [search] {state['query']}")
    return {}

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache

builder = StateGraph(State)
builder.add_node(
    "search",
    expensive_search,
    cache_policy=CachePolicy(key_func=user_scoped_key),
)
builder.add_edge(START, "search")
builder.add_edge("search", END)
graph = builder.compile(
    checkpointer=InMemorySaver(),
    cache=InMemoryCache(),
)

graph.invoke({"query": "AI news", "user_id": "alice"}, config={"configurable": {"thread_id": "1"}})  # [search]
graph.invoke({"query": "AI news", "user_id": "alice"}, config={"configurable": {"thread_id": "2"}})  # cache hit
graph.invoke({"query": "AI news", "user_id": "bob"},   config={"configurable": {"thread_id": "3"}})  # [search] — different user
```

---

## 4 · `UUID` v6 + `uuid6`

**Module**: `langgraph.checkpoint.base.id`  
**First dedicated coverage.**

LangGraph bundles a UUIDv6 implementation (adapted from [`uuid6-python`](https://github.com/oittaa/uuid6-python)) to generate checkpoint IDs with better database index locality than the standard `uuid.uuid4()`. The bundled copy avoids an optional dependency on the `uuid6` package.

```python
import random, time, uuid

_last_v6_timestamp: int | None = None


class UUID(uuid.UUID):
    """Extension of stdlib UUID that adds v6/v7/v8 support."""

    @property
    def subsec(self) -> int:
        return ((self.int >> 64) & 0x0FFF) << 8 | ((self.int >> 54) & 0xFF)

    @property
    def time(self) -> int:
        if self.version == 6:
            return (
                (self.time_low << 28)
                | (self.time_mid << 12)
                | (self.time_hi_version & 0x0FFF)
            )
        # v7/v8 use millisecond timestamps …
        return super().time


def uuid6(node: int | None = None, clock_seq: int | None = None) -> UUID:
    global _last_v6_timestamp
    nanoseconds = time.time_ns()
    # 0x01b21dd213814000 = 100-ns intervals between UUID epoch (1582-10-15) and
    # Unix epoch (1970-01-01). UUIDv6 time is in 100-ns intervals from UUID epoch.
    timestamp = nanoseconds // 100 + 0x01B21DD213814000
    if _last_v6_timestamp is not None and timestamp <= _last_v6_timestamp:
        timestamp = _last_v6_timestamp + 1   # monotonic guard
    _last_v6_timestamp = timestamp
    clock_seq = clock_seq if clock_seq is not None else random.getrandbits(14)
    node      = node      if node      is not None else random.getrandbits(48)
    # Layout: high 48 time bits | low 12 time bits (v6-reordered) | clock_seq | node
    time_high_and_time_mid = (timestamp >> 12) & 0xFFFFFFFFFFFF
    time_low_and_version   = timestamp & 0x0FFF
    uuid_int  = time_high_and_time_mid << 80
    uuid_int |= time_low_and_version   << 64
    uuid_int |= (clock_seq & 0x3FFF)   << 48
    uuid_int |= node & 0xFFFFFFFFFFFF
    return UUID(int=uuid_int, version=6)
```

**Why UUIDv6 for checkpoints?**  
UUIDv4 is random — inserting many checkpoints produces random primary-key values, causing B-tree index splits on every insert (a "write amplification" hotspot for time-series data). UUIDv6 reorders the timestamp bits so that chronologically adjacent checkpoints produce lexicographically adjacent UUIDs, letting the database append checkpoint rows near each other on disk.

### Example 1 — Generate monotonically increasing checkpoint IDs

```python
from langgraph.checkpoint.base.id import uuid6

ids = [uuid6() for _ in range(5)]
hex_ids = [str(u) for u in ids]

# Each ID is strictly greater than the previous (monotonic ordering)
for a, b in zip(hex_ids, hex_ids[1:]):
    assert a < b, f"{a} >= {b}"

print(hex_ids[0])  # e.g. '1ef9e....' — time-ordered string
```

### Example 2 — Extracting the embedded timestamp

```python
import time
from langgraph.checkpoint.base.id import uuid6, UUID

before = time.time_ns() // 100 + 0x01B21DD213814000
uid = uuid6()
after  = time.time_ns() // 100 + 0x01B21DD213814000

# uid.time returns the 60-bit 100-ns timestamp (UUID epoch)
assert before <= uid.time <= after
print(f"UUID timestamp (100-ns ticks from 1582-10-15): {uid.time}")
print(f"Version: {uid.version}")  # 6
print(f"Subsec: {uid.subsec}")    # fractional sub-second component
```

### Example 3 — UUIDv6 vs UUIDv4 lexicographic ordering

```python
import uuid as stdlib_uuid
from langgraph.checkpoint.base.id import uuid6

# Generate 3 pairs in sequence, each ~1ms apart
import time

v4_ids = []
v6_ids = []
for _ in range(3):
    v4_ids.append(str(stdlib_uuid.uuid4()))
    v6_ids.append(str(uuid6()))
    time.sleep(0.001)

# v4: random — no guaranteed lexicographic order
v4_sorted = sorted(v4_ids) == v4_ids
# v6: monotonic — always sorted in generation order
v6_sorted = sorted(v6_ids) == v6_ids

print(f"v4 generated-order == lexicographic order: {v4_sorted}")  # likely False
print(f"v6 generated-order == lexicographic order: {v6_sorted}")  # True
```

---

## 5 · `convert_to_protocol_event`

**Module**: `langgraph.stream._convert`  
**First dedicated coverage.**

`convert_to_protocol_event` is the bridge between LangGraph's v2 streaming API and the v3 protocol layer. When a v3 consumer (a `StreamMux` or `GraphRunStream`) receives events from a graph that was started with the v2 `stream()` API, each `StreamPart` tuple is converted to the uniform `ProtocolEvent` envelope via this function.

```python
import time
from typing import Any, cast
from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams
from langgraph.types import StreamPart


def convert_to_protocol_event(part: StreamPart) -> ProtocolEvent:
    part_dict = cast(dict[str, Any], part)
    params: _ProtocolEventParams = {
        "namespace": list(part_dict["ns"]),      # tuple → list
        "timestamp": int(time.time() * 1000),    # wall-clock ms (non-monotonic!)
        "data":      part_dict["data"],
    }
    if "interrupts" in part_dict:
        params["interrupts"] = part_dict["interrupts"]
    return {
        "type": "event",
        "method": part_dict["type"],   # "values", "messages", "custom", etc.
        "params": params,
    }
```

**`timestamp` vs `seq`:**  
The `timestamp` field is wall-clock milliseconds (from `time.time()`), which can go backwards across NTP adjustments. Consumers that need a total order across root events must use `ProtocolEvent["seq"]` — assigned by the root `StreamMux` as a monotonically increasing integer.

**`interrupts` passthrough:**  
Only `values` stream parts carry `interrupts` (the set of pending `Interrupt` objects). The function forwards this field unchanged when present, ensuring v3 consumers still see interrupt metadata without re-computing it.

### Example 1 — Manual conversion of a values StreamPart

```python
from langgraph.stream._convert import convert_to_protocol_event
from langgraph.types import Interrupt

# Simulate a v2 values StreamPart with interrupts
part = {
    "type": "values",
    "ns": ("subgraph_a",),
    "data": {"messages": [], "counter": 3},
    "interrupts": (Interrupt(value="Please review"),),
}

event = convert_to_protocol_event(part)
print(event["type"])            # "event"
print(event["method"])          # "values"
print(event["params"]["namespace"])  # ["subgraph_a"]
print(event["params"]["data"])  # {"messages": [], "counter": 3}
print(event["params"]["interrupts"])  # (Interrupt(value='Please review'),)
```

### Example 2 — Converting a custom StreamPart (no interrupts)

```python
from langgraph.stream._convert import convert_to_protocol_event

custom_part = {
    "type": "custom",
    "ns": (),
    "data": {"progress": 0.75, "message": "Loading model..."},
}

event = convert_to_protocol_event(custom_part)
assert "interrupts" not in event["params"]
assert event["method"] == "custom"
assert event["params"]["namespace"] == []
print(event["params"]["data"])  # {"progress": 0.75, "message": "Loading model..."}
```

### Example 3 — Integrating with `StreamMux` via conversion

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.stream._convert import convert_to_protocol_event

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "demo"}}
# v2 stream parts
for part in graph.stream(
    {"messages": [{"role": "user", "content": "Hi"}]},
    config=config,
    stream_mode="values",
    version="v2",
):
    # Each part is a StreamPart dict; convert to v3 protocol envelope
    event = convert_to_protocol_event(part)
    print(f"seq={event.get('seq', '(unset)')} method={event['method']}")
```

---

## 6 · `default_retry_on`

**Module**: `langgraph._internal._retry`  
**First dedicated coverage.**

When you create a `RetryPolicy(retry_on=None)`, LangGraph uses `default_retry_on` as the predicate. It embodies the principle of "retry everything that isn't a known-deterministic failure":

```python
def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.HTTPError):
        return 500 <= exc.response.status_code < 600 if exc.response else True
    if isinstance(exc, (
        ValueError, TypeError, ArithmeticError, ImportError,
        LookupError, NameError, SyntaxError, RuntimeError,
        ReferenceError, StopIteration, StopAsyncIteration, OSError,
    )):
        return False
    return True   # anything else → retry
```

**Logic:**

| Condition | Decision | Rationale |
|-----------|----------|-----------|
| `ConnectionError` | Retry | Network blip — transient |
| `httpx.HTTPStatusError` 5xx | Retry | Server-side failure — often transient |
| `requests.HTTPError` 5xx | Retry | Same rationale |
| Known deterministic exceptions | Do NOT retry | `ValueError`, `TypeError`, etc. are programmer errors — retrying won't help |
| Everything else | Retry | Unknown exceptions are assumed transient |

The "blocklist of deterministic exceptions" approach means new transient exception types (from new LLM SDK versions, new tools, etc.) are automatically retried without updating the predicate.

### Example 1 — Understanding the blocklist

```python
from langgraph._internal._retry import default_retry_on

# Deterministic exceptions — NOT retried
assert default_retry_on(ValueError("bad input"))        is False
assert default_retry_on(TypeError("wrong type"))        is False
assert default_retry_on(RuntimeError("logic error"))    is False
assert default_retry_on(ImportError("missing module"))  is False

# Transient exceptions — retried
assert default_retry_on(ConnectionError("timeout"))     is True

# Unknown custom exception — retried by default
class MyApiThrottleError(Exception): pass
assert default_retry_on(MyApiThrottleError("rate limited"))  is True
```

### Example 2 — Composing with a custom `retry_on` predicate

```python
from langgraph._internal._retry import default_retry_on
from langgraph.types import RetryPolicy
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    result: str

attempt_count = 0

def flaky_node(state: State) -> dict:
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        raise ConnectionError("Simulated network failure")
    return {"result": "success"}

# Custom predicate: use default logic but also retry our custom error
class CustomError(Exception): pass

def my_retry_on(exc: Exception) -> bool:
    if isinstance(exc, CustomError):
        return True
    return default_retry_on(exc)

builder = StateGraph(State)
builder.add_node("flaky", flaky_node, retry_policy=RetryPolicy(
    max_attempts=5,
    retry_on=my_retry_on,
    initial_interval=0.01,  # fast for demo
))
builder.add_edge(START, "flaky")
builder.add_edge("flaky", END)
graph = builder.compile()

result = graph.invoke({"result": ""})
print(result["result"])  # "success"
print(f"Took {attempt_count} attempts")
```

### Example 3 — Verifying httpx 5xx behavior

```python
from unittest.mock import MagicMock
from langgraph._internal._retry import default_retry_on
import httpx

# Build a mock httpx.HTTPStatusError for a 503
mock_response_503 = MagicMock()
mock_response_503.status_code = 503
exc_503 = httpx.HTTPStatusError("Service Unavailable", request=MagicMock(), response=mock_response_503)
assert default_retry_on(exc_503) is True   # 5xx → retry

# 429 Too Many Requests is NOT a 5xx — won't be retried by default
mock_response_429 = MagicMock()
mock_response_429.status_code = 429
exc_429 = httpx.HTTPStatusError("Rate Limited", request=MagicMock(), response=mock_response_429)
assert default_retry_on(exc_429) is False  # 4xx → NOT retried

# To also retry 429, add it to a custom predicate:
def retry_with_429(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    return default_retry_on(exc)
```

---

## 7 · Internal Pregel constants

**Module**: `langgraph._internal._constants`  
**First dedicated coverage.**

Every string used as a reserved write key, cache namespace prefix, or `config["configurable"]` routing key in the Pregel engine is defined here as a `sys.intern()`d constant. Interning means all uses of the same string share one object in memory, making `is` comparisons (and dict lookups by identity) fast.

```python
import sys

# Reserved write keys — written by nodes/runtime, read by Pregel loop
INPUT       = sys.intern("__input__")      # initial graph input
INTERRUPT   = sys.intern("__interrupt__")  # pending interrupt values
RESUME      = sys.intern("__resume__")     # values that resume an interrupt
ERROR       = sys.intern("__error__")      # node-raised exceptions
NO_WRITES   = sys.intern("__no_writes__")  # node produced no state updates
TASKS       = sys.intern("__pregel_tasks") # Send() objects → new tasks
RETURN      = sys.intern("__return__")     # raw return value recording
PREVIOUS    = sys.intern("__previous__")   # implicit branch for Command values

# Reserved cache namespace
CACHE_NS_WRITES = sys.intern("__pregel_ns_writes")

# Reserved config.configurable keys
CONFIG_KEY_SEND        = sys.intern("__pregel_send")        # write() fn
CONFIG_KEY_READ        = sys.intern("__pregel_read")        # read() fn
CONFIG_KEY_CALL        = sys.intern("__pregel_call")        # call() fn
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")
CONFIG_KEY_STREAM      = sys.intern("__pregel_stream")      # StreamProtocol
CONFIG_KEY_CACHE       = sys.intern("__pregel_cache")       # BaseCache
CONFIG_KEY_RESUMING    = sys.intern("__pregel_resuming")    # bool: resuming?
CONFIG_KEY_TASK_ID     = sys.intern("__pregel_task_id")     # current task ID
CONFIG_KEY_THREAD_ID   = sys.intern("thread_id")
CONFIG_KEY_CHECKPOINT_NS = sys.intern("checkpoint_ns")      # subgraph namespace
```

**When you encounter these in practice:**

| Constant | When you see it |
|----------|-----------------|
| `TASKS` | Any `Send(node, value)` you return ends up as a `(TASKS, Send(...))` write |
| `INTERRUPT` | `interrupt(value)` writes `(INTERRUPT, interrupt_value)` |
| `RESUME` | On graph resume, the resume payload is injected via this key |
| `ERROR` | Unhandled node exceptions are captured as `(ERROR, exc)` |
| `CONFIG_KEY_SEND` | `config[CONF][CONFIG_KEY_SEND]` is the raw channel write function available inside nodes via `ChannelWrite.do_write()` |
| `CONFIG_KEY_READ` | `config[CONF][CONFIG_KEY_READ]` is the state-read function used by `ChannelRead` |
| `checkpoint_ns` | Appears in LangSmith traces and in every node's `config["configurable"]["checkpoint_ns"]` |

### Example 1 — Reading the checkpoint namespace from inside a node

```python
from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph._internal._constants import CONFIG_KEY_CHECKPOINT_NS
from langgraph.constants import CONF
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    ns: str

def show_ns(state: State, config: RunnableConfig) -> dict:
    # The checkpoint_ns tells you which subgraph (if any) you're in.
    # In the root graph it's an empty string "".
    ns = config.get("configurable", {}).get(CONFIG_KEY_CHECKPOINT_NS, "")
    print(f"checkpoint_ns={ns!r}")
    return {"ns": ns}

builder = StateGraph(State)
builder.add_node("show", show_ns)
builder.add_edge(START, "show")
builder.add_edge("show", END)

from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())
graph.invoke({"ns": ""}, config={"configurable": {"thread_id": "x"}})
# checkpoint_ns=''
```

### Example 2 — Detecting `INTERRUPT` writes in a custom checkpoint observer

```python
from langgraph._internal._constants import INTERRUPT, ERROR, TASKS

def audit_pending_writes(pending_writes: list[tuple[str, object]]) -> None:
    """Log what a node wrote without modifying it."""
    for channel, value in pending_writes:
        if channel == INTERRUPT:
            print(f"  INTERRUPT raised: {value!r}")
        elif channel == ERROR:
            print(f"  ERROR: {value!r}")
        elif channel == TASKS:
            print(f"  Send() to node: {getattr(value, 'node', value)}")
        else:
            print(f"  channel={channel!r} value={value!r}")


# Simulated pending writes from a node that called interrupt() and returned a Send
from langgraph.types import Send, Interrupt

writes = [
    (INTERRUPT, Interrupt(value="Please approve")),
    (TASKS, Send("next_node", {"data": "x"})),
    ("messages", [{"role": "assistant", "content": "Done"}]),
]
audit_pending_writes(writes)
```

### Example 3 — Using `CONFIG_KEY_SEND` for imperative channel writes

```python
from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph._internal._constants import CONFIG_KEY_SEND
from langgraph.constants import CONF
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    results: list[str]

def fan_out_node(state: State, config: RunnableConfig) -> dict:
    write = config[CONF][CONFIG_KEY_SEND]
    # Imperatively dispatch two parallel tasks
    write([
        (CONF.__class__.__mro__[0].__name__,  # just demonstrating access pattern
         None),   # placeholder
    ])
    # Practical: use Send for fan-out instead of direct write access
    return {"results": []}

# NOTE: In practice, return Send() objects from nodes or edges — that's the
# idiomatic pattern. Direct CONFIG_KEY_SEND access is an internal mechanism.
# The example above shows what the internals look like.
print("CONFIG_KEY_SEND =", repr(CONFIG_KEY_SEND))   # '__pregel_send'
print("TASKS =",           repr(TASKS))              # '__pregel_tasks'
print("INTERRUPT =",       repr(INTERRUPT))          # '__interrupt__'
```

---

## 8 · `map_debug_tasks` + `is_multiple_channel_write`

**Module**: `langgraph.pregel.debug`  
**First dedicated coverage.**

These two utilities sit between the raw Pregel execution state and the `stream_mode="debug"` consumer. They appear whenever `debug` events are produced (either in combined-mode streaming or via `CheckpointPayload.tasks`).

```python
from collections.abc import Iterable, Iterator
from typing import Any
from langgraph.types import PregelExecutableTask, TaskPayload
from langgraph.constants import TAG_HIDDEN
from langgraph._internal._constants import EXCLUDED_METADATA_KEYS


def map_debug_tasks(tasks: Iterable[PregelExecutableTask]) -> Iterator[TaskPayload]:
    for task in tasks:
        # Skip internal/hidden nodes (TAG_HIDDEN suppresses a task from debug output)
        if task.config is not None and TAG_HIDDEN in task.config.get("tags", []):
            continue

        payload: TaskPayload = {
            "id":       task.id,
            "name":     task.name,
            "input":    task.input,
            "triggers": task.triggers,
        }
        # Filter metadata: drop framework internal keys (langgraph_node/step/path/…)
        # but forward user-level metadata (lc_agent_name, ls_integration, tags, …)
        if task.config is not None:
            md = {
                k: v for k, v in (task.config.get("metadata") or {}).items()
                if k not in EXCLUDED_METADATA_KEYS
            }
            # Fold user-visible tags into metadata (internal tags are stripped)
            if (filtered_tags := filter_to_user_tags(task.config.get("tags"))):
                md["tags"] = filtered_tags
            if md:
                payload["metadata"] = md
        yield payload


def is_multiple_channel_write(value: Any) -> bool:
    """True if the payload wraps multiple writes from the same channel.

    Used to detect DeltaChannel multi-write payloads in debug streams.
    Such payloads carry {"$writes": [...]} rather than a raw value.
    """
    return (
        isinstance(value, dict)
        and "$writes" in value
        and isinstance(value["$writes"], list)
    )
```

### Example 1 — Consuming `map_debug_tasks` in a debug stream

```python
import asyncio
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import _DebugTaskPayload

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

def llm_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "debug-demo"}}
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "hello"}]},
    config=config,
    stream_mode="debug",
):
    if isinstance(chunk, dict) and chunk.get("type") == "task":
        payload = chunk["payload"]
        print(f"task  id={payload['id'][:8]}… name={payload['name']}")
        print(f"       triggers={payload['triggers']}")
    elif isinstance(chunk, dict) and chunk.get("type") == "task_result":
        payload = chunk["payload"]
        print(f"done  id={payload['id'][:8]}… error={payload.get('error')}")
```

### Example 2 — `TAG_HIDDEN` suppresses internal nodes

```python
from langgraph.pregel.debug import map_debug_tasks
from langgraph.constants import TAG_HIDDEN
from langchain_core.runnables import RunnableConfig
from unittest.mock import MagicMock

def make_task(name: str, hidden: bool = False) -> MagicMock:
    task = MagicMock()
    task.id = f"id-{name}"
    task.name = name
    task.input = {}
    task.triggers = ["__start__"]
    task.config = {"tags": [TAG_HIDDEN] if hidden else [], "metadata": {}}
    return task

visible_task = make_task("my_agent_node", hidden=False)
hidden_task  = make_task("__checkpoint__", hidden=True)

payloads = list(map_debug_tasks([visible_task, hidden_task]))
print([p["name"] for p in payloads])  # ['my_agent_node']  — hidden omitted
```

### Example 3 — `is_multiple_channel_write` in a custom debug observer

```python
from langgraph.pregel.debug import is_multiple_channel_write

# Single write — normal scalar update
single = {"messages": [{"role": "assistant", "content": "Hi"}]}
print(is_multiple_channel_write(single))  # False

# Multi-write payload — produced by DeltaChannel when multiple updates to
# the same channel arrive in one superstep
multi = {"$writes": [
    {"messages": [{"role": "assistant", "content": "Chunk 1"}]},
    {"messages": [{"role": "assistant", "content": "Chunk 2"}]},
]}
print(is_multiple_channel_write(multi))  # True

# Consumer pattern: expand multi-writes before processing
def expand_write(value: object) -> list[object]:
    if is_multiple_channel_write(value):
        return value["$writes"]  # type: ignore[index]
    return [value]

for write in expand_write(multi):
    print(write)
```

---

## 9 · `ensure_message_ids` + `_is_message_dict` + `_state_values`

**Module**: `langgraph.pregel._messages`  
**First dedicated coverage.**

These three utilities solve a subtle but important correctness issue: **message IDs must be stable across checkpoint reads**. Without stable IDs, every `get_state()` replay would produce a different UUID for the same message, causing LangSmith traces to show duplicate or mismatched messages and breaking deduplication in `add_messages`.

```python
from uuid import uuid4
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages

# Known role values (OpenAI-style) and LangChain type values that
# identify a dict as a message. Checked BEFORE coercing to BaseMessage.
_MESSAGE_ROLES = frozenset({"user","human","assistant","ai","tool","system","function"})
_MESSAGE_TYPES = frozenset({"human","ai","tool","system","function","remove"})

def _is_message_dict(item: dict) -> bool:
    return item.get("role") in _MESSAGE_ROLES or item.get("type") in _MESSAGE_TYPES

def _state_values(obj: Any) -> Sequence[Any]:
    """Extract top-level field values from dict/BaseModel/dataclass state."""
    if isinstance(obj, dict):
        return list(obj.values())
    elif isinstance(obj, BaseModel):
        return [getattr(obj, k) for k in type(obj).model_fields]
    elif is_dataclass(obj) and not isinstance(obj, type):
        return [getattr(obj, f.name) for f in fields(obj)]
    return ()

def ensure_message_ids(value: Any) -> None:
    """Coerce message-like write values to typed BaseMessages with stable IDs.
    Called in put_writes() before DeltaChannel writes are submitted to the checkpointer.
    """
    if isinstance(value, BaseMessage):
        if value.id is None:
            value.id = str(uuid4())
    elif isinstance(value, dict) and _is_message_dict(value):
        if not value.get("id"):
            value["id"] = str(uuid4())
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, BaseMessage):
                if item.id is None:
                    item.id = str(uuid4())
            elif isinstance(item, dict) and _is_message_dict(item):
                msg = convert_to_messages([item])[0]
                if msg.id is None:
                    msg.id = str(uuid4())
                value[i] = msg  # in-place replacement with typed BaseMessage
```

**Why in-place mutation?**  
`put_writes()` is called synchronously before the background checkpoint thread is submitted. Since both `checkpoint_pending_writes` and the background thread share the same list object, stamping IDs here means the serialized bytes always reflect the post-coercion state — no race condition.

### Example 1 — ID stamping for raw dict messages

```python
from langgraph.pregel._messages import ensure_message_ids, _is_message_dict

# OpenAI-style dict (no id)
msg_dict = {"role": "assistant", "content": "Hello"}
print(_is_message_dict(msg_dict))  # True
ensure_message_ids(msg_dict)
print(msg_dict["id"])              # e.g. "4b2a..."

# Already has an id — not overwritten
msg_with_id = {"role": "user", "content": "Hi", "id": "existing-id"}
ensure_message_ids(msg_with_id)
assert msg_with_id["id"] == "existing-id"  # unchanged

# Non-message dict — ignored
data_dict = {"result": 42, "score": 0.9}
ensure_message_ids(data_dict)
assert "id" not in data_dict
```

### Example 2 — `_state_values` across state formats

```python
from dataclasses import dataclass
from typing import TypedDict
from pydantic import BaseModel
from langgraph.pregel._messages import _state_values

# TypedDict → plain dict
class TypedState(TypedDict):
    messages: list
    score: float

assert _state_values({"messages": [], "score": 0.5}) == [[], 0.5]

# Pydantic model
class PydanticState(BaseModel):
    messages: list = []
    score: float = 0.0

state = PydanticState(messages=["hi"], score=0.9)
values = _state_values(state)
assert "hi" in values[0]

# Dataclass
@dataclass
class DCState:
    messages: list
    score: float

dc = DCState(messages=["bye"], score=0.1)
assert _state_values(dc) == [["bye"], 0.1]

# Unknown type — returns empty sequence
assert list(_state_values(42)) == []
```

### Example 3 — In-place list mutation for stable checkpoint IDs

```python
from langchain_core.messages import HumanMessage
from langgraph.pregel._messages import ensure_message_ids

# A list with a mix of typed and dict messages
messages = [
    HumanMessage(content="Hello"),           # id=None initially
    {"role": "assistant", "content": "Hi"},   # dict, no id
    HumanMessage(content="World", id="stable-id"),  # already has id
]

ensure_message_ids(messages)

# HumanMessage now has an id
assert messages[0].id is not None

# dict was upgraded to a typed BaseMessage in-place
from langchain_core.messages import BaseMessage
assert isinstance(messages[1], BaseMessage)
assert messages[1].id is not None

# Pre-existing id is preserved
assert messages[2].id == "stable-id"

# IDs are stable across repeated calls
ids_first_pass = [m.id for m in messages]
ensure_message_ids(messages)
ids_second_pass = [m.id for m in messages]
assert ids_first_pass == ids_second_pass
```

---

## 10 · `coerce_timeout_policy` + `sync_timeout_unsupported`

**Module**: `langgraph._internal._timeout`  
**First dedicated coverage.**

These two functions are the internal normalizers for timeout configuration. They sit between the public API (`add_node(timeout=…)`) and the `TimeoutPolicy` dataclass.

```python
from datetime import timedelta
from typing import Literal
from langgraph.types import TimeoutPolicy

_SYNC_TIMEOUT_PREFIX = (
    "Node timeouts are only supported for async nodes because sync Python "
    "execution cannot be safely cancelled in-process."
)


def coerce_timeout_policy(
    value: float | timedelta | TimeoutPolicy | None,
) -> TimeoutPolicy | None:
    """Normalize a timeout value to a TimeoutPolicy, or None if no timeout."""
    return TimeoutPolicy.coerce(value)


def sync_timeout_unsupported(
    name: str, *, kind: Literal["Node", "Task"] = "Node"
) -> ValueError:
    """Build the canonical error for using `timeout` with a sync target."""
    return ValueError(f"{_SYNC_TIMEOUT_PREFIX} {kind} {name!r} is sync.")
```

**Why can't sync nodes have timeouts?**  
Python threads cannot be safely interrupted from the outside — there is no equivalent of `asyncio.wait_for()` for synchronous code. The only safe pre-emptive cancellation mechanism available in CPython is `asyncio.Task.cancel()`, which injects a `CancelledError` at the next `await` point. For this reason, `TimeoutPolicy` is silently ignored (or raises, depending on context) for sync node functions.

**`TimeoutPolicy.coerce()` normalizations:**

| Input type | Result |
|-----------|--------|
| `None` | `None` |
| `float` (seconds) | `TimeoutPolicy(run_timeout=float)` |
| `timedelta` | `TimeoutPolicy(run_timeout=td.total_seconds())` |
| `TimeoutPolicy` | Passed through unchanged |

### Example 1 — Understanding `coerce_timeout_policy`

```python
from datetime import timedelta
from langgraph._internal._timeout import coerce_timeout_policy
from langgraph.types import TimeoutPolicy

# Float → TimeoutPolicy with run_timeout
p1 = coerce_timeout_policy(30.0)
assert isinstance(p1, TimeoutPolicy)
print(p1.run_timeout)   # 30.0

# timedelta → TimeoutPolicy
p2 = coerce_timeout_policy(timedelta(minutes=2))
assert isinstance(p2, TimeoutPolicy)
print(p2.run_timeout)   # 120.0

# Already a TimeoutPolicy → passed through
p3 = TimeoutPolicy(run_timeout=10.0, idle_timeout=5.0)
assert coerce_timeout_policy(p3) is p3   # same object

# None → None
assert coerce_timeout_policy(None) is None
```

### Example 2 — Async node with timeout (the supported pattern)

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy

class State(TypedDict):
    result: str

async def slow_async_node(state: State) -> dict:
    await asyncio.sleep(0.1)  # simulate work
    return {"result": "done"}

builder = StateGraph(State)
builder.add_node(
    "slow",
    slow_async_node,
    # Both forms are equivalent after coerce_timeout_policy():
    # timeout=5.0  OR  timeout=TimeoutPolicy(run_timeout=5.0)
    timeout=TimeoutPolicy(run_timeout=5.0),
)
builder.add_edge(START, "slow")
builder.add_edge("slow", END)

graph = builder.compile()
result = asyncio.run(graph.ainvoke({"result": ""}))
print(result["result"])  # "done"
```

### Example 3 — Handling `sync_timeout_unsupported` in a custom wrapper

```python
import inspect
from langgraph._internal._timeout import coerce_timeout_policy, sync_timeout_unsupported
from langgraph.types import TimeoutPolicy
from typing import Callable, Any


def safe_add_node_timeout(
    func: Callable[..., Any],
    timeout: Any,
    name: str,
) -> TimeoutPolicy | None:
    """Validate timeout compatibility before passing to StateGraph.add_node()."""
    policy = coerce_timeout_policy(timeout)
    if policy is None:
        return None
    if not inspect.iscoroutinefunction(func):
        # Raise early with the canonical error message
        raise sync_timeout_unsupported(name, kind="Node")
    return policy


# Sync function → error
def my_sync_node(state): return {}

try:
    safe_add_node_timeout(my_sync_node, 30.0, "my_sync_node")
except ValueError as e:
    print(e)
# Node timeouts are only supported for async nodes ... Node 'my_sync_node' is sync.

# Async function → ok
async def my_async_node(state): return {}
policy = safe_add_node_timeout(my_async_node, 30.0, "my_async_node")
print(policy)  # TimeoutPolicy(run_timeout=30.0, ...)
```

---

## Summary

| # | Class / symbol | Key takeaway |
|---|----------------|--------------|
| 1 | `BaseCache` + `FullKey` + `Namespace` | 6-method sync+async ABC; `FullKey=(Namespace, str)` gives per-task/per-user isolation without separate cache instances |
| 2 | `StateNode` + Protocol family | 9 Protocol variants for `add_node()` static typing; `StateNodeSpec` exposes `defer`, `ends`, `is_error_handler`, `error_handler_node` after `compile()` |
| 3 | `_freeze` + `default_cache_key` | Dict-key-sorting + Sequence→tuple + numpy escape produces deterministic, protocol-5 pickle cache keys from any Python object |
| 4 | `UUID` v6 + `uuid6` | Monotonic timestamp encoding for checkpoint IDs; lexicographic sort = insertion-time order → better DB index locality than UUIDv4 |
| 5 | `convert_to_protocol_event` | One-function v2→v3 bridge: `StreamPart` dict → `ProtocolEvent` envelope; `interrupts` passthrough; use `seq`, not `timestamp`, for ordering |
| 6 | `default_retry_on` | Blocklist-of-deterministic-exceptions strategy; httpx/requests 5xx detected; anything not in the blocklist is retried |
| 7 | Internal Pregel constants | `sys.intern()`d reserved strings for write keys (`TASKS`, `INTERRUPT`, `RESUME`) and config routing (`CONFIG_KEY_SEND`, `CONFIG_KEY_READ`, `checkpoint_ns`) |
| 8 | `map_debug_tasks` + `is_multiple_channel_write` | `TAG_HIDDEN` suppresses internal nodes from debug streams; multi-write `$writes` payloads must be expanded before processing |
| 9 | `ensure_message_ids` + helpers | Pre-checkpoint in-place ID stamping prevents replay ID drift; `_is_message_dict` detects both OpenAI-role and LangChain-type message dicts; `_state_values` extracts field values from dict/Pydantic/dataclass |
| 10 | `coerce_timeout_policy` + `sync_timeout_unsupported` | `float/timedelta/TimeoutPolicy/None` normalized via `TimeoutPolicy.coerce()`; sync nodes cannot be safely timed out — only async nodes support `TimeoutPolicy` |

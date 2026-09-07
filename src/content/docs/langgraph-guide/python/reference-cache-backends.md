---
title: "BaseCache & InMemoryCache — cache backend API reference"
description: "Build and use LangGraph node cache backends — BaseCache abstract interface, InMemoryCache thread-safe implementation, custom backend wiring, and CachePolicy integration for LangGraph 1.2.11."
framework: langgraph
language: python
sidebar:
  label: "Ref · Cache backends"
  order: 42
---

# BaseCache & InMemoryCache — cache backends

Verified against **`langgraph==1.2.11`** (modules: `langgraph.cache.base`, `langgraph.cache.memory`).

LangGraph node caching is a two-part system:

| Part | What it is |
|---|---|
| `CachePolicy` | Per-node config: TTL and optional key function. Attached via `add_node(..., cache_policy=...)`. |
| `BaseCache` / `InMemoryCache` | Backend: stores and retrieves cached node outputs. Passed to `compile(cache=...)`. |

This page covers the backend side. See [chapter 9 — Advanced Patterns](/langgraph-guide/python/chapter-09-advanced-patterns) for `CachePolicy` examples.

---

## Class definitions

### `BaseCache`

```python
# langgraph.cache.base
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")
Namespace = tuple[str, ...]      # e.g. ("my_graph", "embed_node")
FullKey = tuple[Namespace, str]  # namespace + hashed string key


class BaseCache(ABC, Generic[ValueT]):
    """Abstract base class for a LangGraph node cache backend."""

    serde: SerializerProtocol  # default: JsonPlusSerializer

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

### `InMemoryCache`

```python
# langgraph.cache.memory
import threading
from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT


class InMemoryCache(BaseCache[ValueT]):
    """Thread-safe in-memory cache with optional TTL eviction."""

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, float | None]]] = {}
        self._lock = threading.RLock()
```

---

## Imports

```python
from langgraph.cache.base import BaseCache
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy
```

---

## Minimal runnable example

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


class State(TypedDict):
    text: str
    embedding: list[float]


def embed(state: State) -> dict:
    """Expensive embedding call — only runs once per unique input."""
    print(f"  [embed] computing for: {state['text']!r}")
    return {"embedding": [hash(state["text"]) % 100 / 100]}


cache = InMemoryCache()

graph = (
    StateGraph(State)
    .add_node("embed", embed, cache_policy=CachePolicy(ttl=3600))
    .add_edge(START, "embed")
    .add_edge("embed", END)
    .compile(cache=cache)
)

# First call: runs embed
r1 = graph.invoke({"text": "hello", "embedding": []})
print(r1["embedding"])  # e.g. [0.28]

# Second call with same input: returns cached result, embed() NOT called
r2 = graph.invoke({"text": "hello", "embedding": []})
print(r2["embedding"])  # same as r1

# Third call with different input: runs embed again
r3 = graph.invoke({"text": "world", "embedding": []})
print(r3["embedding"])  # different value
```

---

## `BaseCache` method reference

### `get(keys)` / `aget(keys)`

Read multiple cached values in one batch. Returns only keys that exist and have not expired.

```python
from langgraph.cache.base import FullKey, Namespace

# FullKey = (Namespace, key_hash)
ns: Namespace = ("my_graph", "embed_node")
keys: list[FullKey] = [(ns, "abc123"), (ns, "def456")]

results: dict[FullKey, Any] = cache.get(keys)
# keys not in cache (or expired) are simply absent from the result
```

### `set(pairs)` / `aset(pairs)`

Write values with optional TTL (in seconds). `None` TTL means no expiry.

```python
from collections.abc import Mapping

pairs: Mapping[FullKey, tuple[Any, int | None]] = {
    (ns, "abc123"): ({"embedding": [0.1]}, 3600),  # expires in 1h
    (ns, "def456"): ({"embedding": [0.2]}, None),   # never expires
}
cache.set(pairs)
```

### `clear(namespaces)` / `aclear(namespaces)`

Clear one namespace, several, or the entire cache.

```python
# Clear a specific node's cache
cache.clear([("my_graph", "embed_node")])

# Clear the entire cache
cache.clear()
```

---

## `InMemoryCache` internals

`InMemoryCache` stores entries as `(encoding, bytes, expiry_timestamp)` tuples, serialized with `JsonPlusSerializer`. The `RLock` makes it safe to call from multiple threads (e.g. when `ToolNode` runs tool calls in parallel).

TTL expiry is lazy: expired entries are detected and removed on the next `get()` rather than on a background timer.

```python
# Inspect the internal cache
cache = InMemoryCache()
cache.set({(("ns", "n"), "k"): ({"v": 1}, 10)})

# Internal structure (not public API — illustrative only)
# cache._cache = {
#   ("ns", "n"): {
#     "k": ("json", b'{"v": 1}', <expiry_ts>)
#   }
# }
```

---

## Building a custom cache backend

Subclass `BaseCache` to use Redis, Memcached, or any other store. Implement all six abstract methods:

```python
import json
from collections.abc import Mapping, Sequence
from typing import Any

import redis

from langgraph.cache.base import BaseCache, FullKey, Namespace


class RedisCache(BaseCache[Any]):
    """Simple Redis-backed cache for LangGraph nodes."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__()
        self._r = redis.Redis.from_url(redis_url, decode_responses=False)

    def _make_redis_key(self, full_key: FullKey) -> str:
        ns, k = full_key
        return f"langgraph:{':'.join(ns)}:{k}"

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        if not keys:
            return {}
        redis_keys = [self._make_redis_key(k) for k in keys]
        values = self._r.mget(redis_keys)
        result: dict[FullKey, Any] = {}
        for full_key, raw in zip(keys, values):
            if raw is not None:
                # Payload is stored as b"<enc>|<serialized-bytes>"
                enc_b, data = raw.split(b"|", 1)
                result[full_key] = self.serde.loads_typed((enc_b.decode(), data))
        return result

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        return self.get(keys)

    def set(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        pipe = self._r.pipeline()
        for full_key, (value, ttl) in pairs.items():
            rk = self._make_redis_key(full_key)
            enc, data = self.serde.dumps_typed(value)
            # Prefix encoding name so get() can reconstruct the typed pair
            payload = enc.encode() + b"|" + data
            if ttl is not None:
                pipe.setex(rk, ttl, payload)
            else:
                pipe.set(rk, payload)
        pipe.execute()

    async def aset(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        self.set(pairs)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        if namespaces is None:
            # Flush all langgraph cache keys
            for key in self._r.scan_iter("langgraph:*"):
                self._r.delete(key)
        else:
            for ns in namespaces:
                prefix = f"langgraph:{':'.join(ns)}:*"
                for key in self._r.scan_iter(prefix):
                    self._r.delete(key)

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        self.clear(namespaces)
```

Use it the same way:

```python
cache = RedisCache("redis://localhost:6379")
graph = builder.compile(cache=cache)
```

---

## Custom key function

By default `CachePolicy` hashes the node's full input with pickle. Supply `key_func` to hash only the fields that matter, reducing false cache misses:

```python
from typing import Any
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache


def text_only_key(state: Any) -> str:
    """Hash only the 'text' field — ignore ephemeral fields like timestamps."""
    if isinstance(state, dict):
        return state.get("text", "")
    return str(state)


cache = InMemoryCache()
graph = (
    builder
    .add_node("embed", embed_fn, cache_policy=CachePolicy(key_func=text_only_key, ttl=600))
    .compile(cache=cache)
)
```

---

## Cache with multiple nodes

Each node gets its own namespace in the cache. Different nodes with the same input hash do not collide:

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

cache = InMemoryCache()

builder = StateGraph(State)
builder.add_node("embed", embed_fn, cache_policy=CachePolicy(ttl=3600))
builder.add_node("classify", classify_fn, cache_policy=CachePolicy(ttl=1800))
builder.add_node("summarize", summarize_fn)  # no cache

graph = builder.compile(cache=cache)
# embed and classify caches are stored in separate namespaces
# summarize is always executed
```

---

## Async graphs

All `BaseCache` methods have both sync and async variants. When compiling an async graph, LangGraph calls `aget` and `aset`. `InMemoryCache`'s async methods delegate to their sync counterparts (the lock is still acquired):

```python
import asyncio
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()

async def main():
    result = await graph.ainvoke({"text": "hello", "embedding": []})
    print(result)

asyncio.run(main())
```

For true async caches (e.g. aioredis), implement `aget` / `aset` / `aclear` with async I/O:

```python
import aioredis
from langgraph.cache.base import BaseCache, FullKey, Namespace
from typing import Any
from collections.abc import Mapping, Sequence


class AsyncRedisCache(BaseCache[Any]):
    def __init__(self, url: str):
        super().__init__()
        self._url = url
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = await aioredis.from_url(self._url)
        return self._client

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        client = await self._get_client()
        result = {}
        for k in keys:
            ns, key_hash = k
            rk = f"lg:{':'.join(ns)}:{key_hash}"
            raw = await client.get(rk)
            if raw:
                enc_b, data = raw.split(b"|", 1)
                result[k] = self.serde.loads_typed((enc_b.decode(), data))
        return result

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, Any]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.aget(keys))

    async def aset(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        client = await self._get_client()
        for full_key, (value, ttl) in pairs.items():
            ns, key_hash = full_key
            rk = f"lg:{':'.join(ns)}:{key_hash}"
            enc, data = self.serde.dumps_typed(value)
            payload = enc.encode() + b"|" + data
            if ttl:
                await client.setex(rk, ttl, payload)
            else:
                await client.set(rk, payload)

    def set(self, pairs: Mapping[FullKey, tuple[Any, int | None]]) -> None:
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aset(pairs))

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        client = await self._get_client()
        async for key in client.scan_iter("lg:*"):
            await client.delete(key)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aclear(namespaces))
```

---

## Clearing caches between test runs

```python
import pytest
from langgraph.cache.memory import InMemoryCache

@pytest.fixture
def fresh_cache():
    cache = InMemoryCache()
    yield cache
    cache.clear()  # wipe all entries after each test


def test_embed_caches_result(fresh_cache):
    graph = builder.compile(cache=fresh_cache)
    r1 = graph.invoke({"text": "hello", "embedding": []})
    r2 = graph.invoke({"text": "hello", "embedding": []})
    assert r1["embedding"] == r2["embedding"]
```

---

## `BaseCache` — abstract method signatures

| Method | Sync | Async | Description |
|---|---|---|---|
| `get(keys)` | ✅ | `aget` | Batch read; returns only found, non-expired keys |
| `set(pairs)` | ✅ | `aset` | Batch write with per-entry optional TTL (seconds) |
| `clear(namespaces)` | ✅ | `aclear` | Clear specific namespaces or the whole cache |

---

## `FullKey` and `Namespace` types

```python
Namespace = tuple[str, ...]  # e.g. ("graph_name", "node_name")
FullKey = tuple[Namespace, str]  # namespace + hashed key string
```

LangGraph constructs these automatically. You only need them when building a custom backend.

---

## Gotchas

- **TTL is wall-clock, not access-time.** `InMemoryCache` uses absolute expiry timestamps, not LRU/LFU. An entry set with `ttl=60` expires 60 seconds after insertion regardless of access pattern.
- **Expired entries are evicted lazily.** They are removed on the next `get()` for their key, not on a background timer.
- **`InMemoryCache` is process-local.** It does not survive process restarts and is not shared between workers. Use a networked backend (Redis, Memcached) for multi-worker deployments.
- **Pickle fallback is disabled** in the default `JsonPlusSerializer`. If your node returns a non-JSON-serializable value, you'll get a serialization error. Pass a custom `serde` to enable pickle: `InMemoryCache(serde=JsonPlusSerializer(pickle_fallback=True))`.
- **`cache_policy` without `compile(cache=...)` is silently ignored.** The node will execute on every call.

---

## Version history

| Version | Change |
|---|---|
| 1.2.11 | `BaseCache`, `InMemoryCache` production-stable |
| 1.2.0 | `CachePolicy.key_func` customization added |
| 1.1.0 | Node caching first introduced with `CachePolicy` + `compile(cache=...)` |

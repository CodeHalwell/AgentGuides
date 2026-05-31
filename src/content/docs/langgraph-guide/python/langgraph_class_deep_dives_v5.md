---
title: "Class deep-dives Vol. 5 — 10 more LangGraph types"
description: "Source-verified deep dives into RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, and the structured HITL protocol — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 5"
  order: 29
---

# Class deep-dives Vol. 5 — 10 more LangGraph types

Verified against **`langgraph==1.2.2`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

[→ Vol. 4 covers set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, error taxonomy](./langgraph_class_deep_dives_v4/)

---

## 1 · `RedisCache`

**Module:** `langgraph.cache.redis`  
**Install:** `pip install redis`

`RedisCache` is a drop-in replacement for `InMemoryCache` that persists node output caches to Redis. It uses `MGET` / pipeline `SET`/`SETEX` for efficient batch reads and writes. When Redis is unavailable, all operations silently degrade to no-ops — the graph still runs correctly, just without caching.

### Constructor

```python
RedisCache(
    redis: Any,              # sync or async Redis client
    *,
    serde: SerializerProtocol | None = None,
    prefix: str = "langgraph:cache:",
)
```

| Parameter | What it does |
|---|---|
| `redis` | Any redis-py (sync or async) client; also works with redis-compatible shims (ioredis, etc.) |
| `serde` | Override the value serializer. Defaults to `JsonPlusSerializer()`. |
| `prefix` | Namespace prefix for all keys, e.g. `"myapp:graph:cache:"`. |

Keys are encoded as `{prefix}{ns1}:{ns2}:…:{cache_key}`. Values are stored as `{encoding}:{data}` bytes, where `encoding` comes from `serde.dumps_typed()`.

### Wiring a `RedisCache` into a graph

```python
import redis
from langgraph.cache.redis import RedisCache
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from typing_extensions import TypedDict

r = redis.Redis(host="localhost", port=6379, decode_responses=False)
cache = RedisCache(r, prefix="myapp:graph:")

class State(TypedDict):
    query: str
    result: str

call_count = {"embed": 0}

def embed_node(state: State) -> dict:
    call_count["embed"] += 1
    # Simulates a slow embedding call
    return {"result": f"embedding({state['query']})"}

graph = (
    StateGraph(State)
    .add_node(
        "embed",
        embed_node,
        cache_policy=CachePolicy(ttl=3600),   # 1-hour TTL in Redis
    )
    .add_edge(START, "embed")
    .add_edge("embed", END)
    .compile(cache=cache)
)

config = {"configurable": {"thread_id": "t1"}}

# First call — hits embed_node
result1 = graph.invoke({"query": "hello", "result": ""}, config)
print(call_count["embed"])  # 1

# Second call with same input — served from Redis
result2 = graph.invoke({"query": "hello", "result": ""}, config)
print(call_count["embed"])  # still 1
```

### Clearing namespaces

```python
# Clear the entire cache for this prefix
cache.clear()

# Clear only one sub-namespace
cache.clear(namespaces=[("langgraph", "cache", "writes")])
```

### Production tips

- Set `prefix` per environment (`"prod:graph:"`, `"staging:graph:"`) to avoid cross-contamination.
- Use Redis Cluster or Sentinel for HA; `RedisCache` works with both since it delegates to the client object you pass in.
- The `serde` default (`JsonPlusSerializer`) serialises most Python built-ins and LangChain types. Pass `EncryptedSerializer` (see §2) to encrypt cached values at rest.

---

## 2 · `EncryptedSerializer`

**Module:** `langgraph.checkpoint.serde.encrypted`  
**Install:** `pip install pycryptodome` (for the AES factory)

`EncryptedSerializer` wraps any `SerializerProtocol` and encrypts the raw bytes before storing them in a checkpoint (or cache). The type string emitted by `dumps_typed` is extended with `+{ciphername}` so the loader knows which cipher to use when decrypting.

### Constructor and protocol

```python
class EncryptedSerializer(SerializerProtocol):
    def __init__(
        self,
        cipher: CipherProtocol,
        serde: SerializerProtocol = JsonPlusSerializer(),
    ) -> None: ...

    # Convenience factory — creates AES-EAX cipher automatically
    @classmethod
    def from_pycryptodome_aes(
        cls,
        serde: SerializerProtocol = JsonPlusSerializer(),
        **kwargs: Any,
    ) -> "EncryptedSerializer": ...
```

`CipherProtocol` is a two-method interface you can implement with any library:

```python
class CipherProtocol:
    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        """Return (cipher_name, ciphertext)."""
        ...
    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        """Return plaintext bytes."""
        ...
```

### Example: AES-EAX encrypted checkpoints

```python
import os
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 32-byte key → AES-256. Can also read from LANGGRAPH_AES_KEY env var.
key = os.urandom(32)
enc_serde = EncryptedSerializer.from_pycryptodome_aes(key=key)

# Pass the encrypted serializer to any saver
saver = InMemorySaver(serde=enc_serde)

class State(TypedDict):
    secret: str

def set_secret(state: State) -> dict:
    return {"secret": "classified-value-42"}

graph = (
    StateGraph(State)
    .add_node("set", set_secret)
    .add_edge(START, "set")
    .add_edge("set", END)
    .compile(checkpointer=saver)
)

config = {"configurable": {"thread_id": "s1"}}
graph.invoke({"secret": ""}, config)

# Raw bytes in the saver are encrypted — only readable with the same key
checkpoint_tuple = saver.get_tuple(config)
raw_channel = checkpoint_tuple.checkpoint["channel_values"].get("__end__")
print(type(raw_channel))   # <class 'bytes'>  (opaque ciphertext)
```

### Example: Custom cipher (Fernet)

```python
from cryptography.fernet import Fernet
from langgraph.checkpoint.serde.base import CipherProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

class FernetCipher(CipherProtocol):
    def __init__(self, key: bytes) -> None:
        self._f = Fernet(key)

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "fernet", self._f.encrypt(plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        assert ciphername == "fernet"
        return self._f.decrypt(ciphertext)

key = Fernet.generate_key()
enc_serde = EncryptedSerializer(FernetCipher(key))
```

### Mixing encrypted and unencrypted checkpoints

`loads_typed` detects whether the type string contains `+` (encrypted) or not (plain). This means you can migrate incrementally: old checkpoints without the `+` suffix are decoded with the plain `serde` fallback automatically.

---

## 3 · `JsonPlusSerializer`

**Module:** `langgraph.checkpoint.serde.jsonplus`

`JsonPlusSerializer` is the default serializer for every `BaseCheckpointSaver`. It uses **ormsgpack** (binary MessagePack) as its primary encoding and falls back to LangChain's JSON-plus format for types that msgpack cannot handle natively. Understanding it matters for three production concerns: performance, security, and custom type support.

### Constructor

```python
JsonPlusSerializer(
    *,
    pickle_fallback: bool = False,
    allowed_json_modules: Iterable[tuple[str, ...]] | Literal[True] | None = None,
    allowed_msgpack_modules: AllowedMsgpackModules | Literal[True] | None = ...,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `pickle_fallback` | `False` | **Never enable in production** — allows arbitrary pickle deserialization. |
| `allowed_json_modules` | `None` | Modules whose classes may be decoded from the legacy JSON format. `True` = allow all (insecure). |
| `allowed_msgpack_modules` | Sentinel | `True` = allow all (default, with warning). `None` = only SAFE_MSGPACK_TYPES. Overridden by `LANGGRAPH_STRICT_MSGPACK=true`. |

### Strict mode

Set `LANGGRAPH_STRICT_MSGPACK=true` in your environment (or `export LANGGRAPH_STRICT_MSGPACK=true`) to lock deserialization to the built-in allowlist. This is the **recommended production setting** — it prevents a compromised checkpoint database from triggering code execution on load.

```bash
# docker / k8s env
LANGGRAPH_STRICT_MSGPACK=true

# Or in Python before importing langgraph
import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
import langgraph  # strict mode is now active
```

### Extending the allowlist for custom types

When strict mode is enabled, you can register additional safe types:

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Allow your own module's types through msgpack
serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("myapp.models", "UserProfile"),
        ("myapp.models", "ConversationState"),
    ]
)
```

### `dumps_typed` / `loads_typed` protocol

All savers call `serde.dumps_typed(obj)` → `(encoding, bytes)` and `serde.loads_typed((encoding, bytes))` → `obj`. The `encoding` string distinguishes the format:

| Encoding | Format |
|---|---|
| `"msgpack"` | ormsgpack binary |
| `"json"` | LangChain JSON-plus (legacy) |
| `"pickle"` | Python pickle (only with `pickle_fallback=True`) |
| `"json+aes"` | JSON-plus then AES-encrypted (when wrapped in `EncryptedSerializer`) |

### Inspecting a live checkpoint's encoding

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    n: int

graph = (
    StateGraph(S)
    .add_node("inc", lambda s: {"n": s["n"] + 1})
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile(checkpointer=InMemorySaver())
)
config = {"configurable": {"thread_id": "x"}}
graph.invoke({"n": 0}, config)

saver = graph.checkpointer
ct = saver.get_tuple(config)
# channel_values are already deserialized objects; raw storage is in saver._storage
for ns_config, (cp, meta, writes) in saver._storage.items():
    print(cp["channel_versions"])
    break
```

---

## 4 · `UntrackedValue` channel

**Module:** `langgraph.channels.untracked_value`

`UntrackedValue` stores a value exactly like `LastValue` but **never writes it to a checkpoint**. Use it for:

- Large in-memory objects (loaded models, connection pools) that are too expensive to serialize
- Values that are meaningless to restore across runs (random seeds, timestamps)
- Computed scratch values that are only needed within a single run

### Channel semantics (from source)

| Property | Behaviour |
|---|---|
| `checkpoint()` | Always returns `MISSING` — nothing is saved |
| `from_checkpoint()` | Constructs empty channel — the value is gone after a run ends |
| `update(values)` | `guard=True` (default): raises `InvalidUpdateError` if more than one node writes per super-step. `guard=False`: last write wins silently. |

### Declaring an `UntrackedValue` channel in a state schema

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import UntrackedValue

class State(TypedDict):
    query: str
    result: str
    # This heavy object is never checkpointed
    model: Annotated[object, UntrackedValue(object)]
```

### Full runnable example

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import UntrackedValue
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# Simulate a large in-memory resource
class FakeModel:
    def __init__(self) -> None:
        self.calls = 0
    def run(self, text: str) -> str:
        self.calls += 1
        return f"model_output({text})"

_global_model = FakeModel()

class State(TypedDict):
    text: str
    output: str
    model: Annotated[object, UntrackedValue(object)]

def load_model(state: State) -> dict:
    """Inject the model handle — never persisted."""
    return {"model": _global_model}

def run_model(state: State) -> dict:
    m = state["model"]
    return {"output": m.run(state["text"])}

graph = (
    StateGraph(State)
    .add_node("load", load_model)
    .add_node("run", run_model)
    .add_edge(START, "load")
    .add_edge("load", "run")
    .add_edge("run", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "u1"}}
result = graph.invoke({"text": "hello", "output": "", "model": None}, config)
print(result["output"])  # model_output(hello)

# The checkpointed state has no "model" key
ct = graph.checkpointer.get_tuple(config)
print("model" in ct.checkpoint["channel_values"])  # False
```

### `guard=False` — allow parallel writes

```python
from langgraph.channels import UntrackedValue

# Any parallel node may overwrite; no InvalidUpdateError
class PState(TypedDict):
    scratch: Annotated[str, UntrackedValue(str, guard=False)]
```

---

## 5 · `AnyValue` channel

**Module:** `langgraph.channels.any_value`

`AnyValue` stores the last value written (like `LastValue`) but **tolerates concurrent writes from multiple parallel nodes**, assuming all writers produce the same value. No reducer is called — if writes disagree, the last one silently wins.

Use `AnyValue` when:
- Multiple parallel nodes all write an identical value (e.g., a shared config reference)
- You want `LastValue` semantics but need parallel graph topology without a reducer

### Source semantics

| Property | Behaviour |
|---|---|
| `update(values=[])` | Clears the value (sets to `MISSING`), returns `True` if previously set |
| `update(values=[v])` | Sets to `v` |
| `update(values=[v1,v2])` | Sets to `v2` — **last write wins, no error raised** |
| `checkpoint()` | Returns current value (unlike `UntrackedValue`, this IS checkpointed) |

### Declaring an `AnyValue` channel

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import AnyValue

class State(TypedDict):
    config_version: Annotated[str, AnyValue(str)]
```

### Full example: parallel nodes, same write

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import AnyValue
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    result_a: str
    result_b: str
    run_id: Annotated[str, AnyValue(str)]  # Both nodes write the same run_id

def node_a(state: State) -> dict:
    return {"result_a": "done_a", "run_id": state["run_id"]}

def node_b(state: State) -> dict:
    return {"result_b": "done_b", "run_id": state["run_id"]}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")
    .add_edge(["a", "b"], END)
    .compile()
)

result = graph.invoke({"result_a": "", "result_b": "", "run_id": "run-42"})
print(result["run_id"])    # run-42
print(result["result_a"])  # done_a
print(result["result_b"])  # done_b
```

### Choosing between `AnyValue`, `LastValue`, and `BinaryOperatorAggregate`

| Scenario | Channel to use |
|---|---|
| Only one node writes per step | `LastValue` (plain field, no annotation) |
| Parallel nodes write identical values | `AnyValue` |
| Parallel nodes write different values that should merge | `BinaryOperatorAggregate` (with a reducer like `operator.add`) |
| Parallel nodes all contribute items to a list | `Topic` |
| Value needed in-step only, not checkpointed | `UntrackedValue` |

---

## 6 · `EmbeddingsLambda` / `ensure_embeddings`

**Module:** `langgraph.store.base.embed`

`EmbeddingsLambda` wraps any Python function (sync or async) that converts `list[str]` → `list[list[float]]` into LangChain's `Embeddings` interface. `ensure_embeddings()` is the factory that builds the right wrapper from several input types (existing `Embeddings`, a raw callable, or a `"provider:model"` string).

This is the primary hook for enabling **semantic search** in `InMemoryStore` without importing a full LangChain embeddings package.

### `ensure_embeddings` signature

```python
def ensure_embeddings(
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str | None,
) -> Embeddings:
```

| Input type | Result |
|---|---|
| `Embeddings` instance | Passed through unchanged |
| Sync `Callable[[list[str]], list[list[float]]]` | Wrapped in `EmbeddingsLambda` |
| Async `Callable[[list[str]], Awaitable[list[list[float]]]]` | Wrapped in `EmbeddingsLambda` (async only) |
| `"provider:model"` string | Calls `langchain.embeddings.init_embeddings(string)` (requires `langchain>=0.3.9`) |

### Example: custom sync embedding function

```python
import numpy as np
from langgraph.store.memory import InMemoryStore

# A trivial bag-of-chars embedding (replace with a real model)
def my_embed(texts: list[str]) -> list[list[float]]:
    vocab = "abcdefghijklmnopqrstuvwxyz "
    result = []
    for t in texts:
        vec = [t.lower().count(c) / max(len(t), 1) for c in vocab]
        result.append(vec)
    return result

store = InMemoryStore(
    index={
        "dims": 27,          # must match the vector dimension
        "embed": my_embed,   # ensure_embeddings() wraps this automatically
        "fields": ["text"],  # which document fields to embed
    }
)

store.put(("docs",), "a", {"text": "python programming"})
store.put(("docs",), "b", {"text": "java enterprise"})
store.put(("docs",), "c", {"text": "python data science"})

results = store.search(("docs",), query="python", limit=2)
for item in results:
    print(item.key, item.score)
# a  <similarity score>
# c  <similarity score>
```

### Example: async embedding function

```python
import asyncio
from langgraph.store.memory import InMemoryStore

async def async_embed(texts: list[str]) -> list[list[float]]:
    # Simulate async model call
    await asyncio.sleep(0.01)
    return [[float(len(t))] * 4 for t in texts]

store = InMemoryStore(index={"dims": 4, "embed": async_embed})

async def main():
    await store.aput(("docs",), "short", {"text": "hi"})
    await store.aput(("docs",), "long", {"text": "a much longer document"})
    results = await store.asearch(("docs",), query="hello world")
    for r in results:
        print(r.key, r.score)

asyncio.run(main())
```

### Path expressions for indexed fields

The `fields` key in `IndexConfig` supports JSONPath-like selectors:

```python
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": my_embed,
        "fields": [
            "content",              # top-level key
            "metadata.summary",     # nested key
            "[0]",                  # first element of an array
            "{title,body}",         # embed both title and body concatenated
            "[*]",                  # all elements of an array
            "$",                    # entire document JSON
        ],
    }
)
```

Per-item field override (index only `abstract`, not the whole document):

```python
store.put(
    ("papers",), "paper-001",
    {"abstract": "...", "body": "... (very long) ..."},
    index=["abstract"],  # overrides the store-level fields for this item
)
```

Disable indexing for a single item:

```python
store.put(("cache",), "temp-key", {"data": "..."}, index=False)
```

---

## 7 · `BaseCheckpointSaver` — building a custom backend

**Module:** `langgraph.checkpoint.base`

`BaseCheckpointSaver[V]` is the abstract base for all checkpoint backends. `V` is the version type (`int`, `float`, or `str`) used to order channel writes. You only need to implement four methods to get a working backend: `get_tuple`, `list`, `put`, and `put_writes`.

### Minimal contract

```python
from collections.abc import Iterator
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)

class MyCustomSaver(BaseCheckpointSaver[int]):

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return the latest checkpoint for the given thread_id, or None."""
        ...

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoints matching criteria (newest first)."""
        ...

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint; return an updated config with checkpoint_id set."""
        ...

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes (partial updates) for a checkpoint."""
        ...
```

### Full example: in-process dict-backed saver

This is a working custom saver that stores checkpoints in a plain Python `dict`. It mirrors the structure used by `InMemorySaver` without the threading locks, to keep the example readable.

```python
from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    get_checkpoint_id,
)


class DictSaver(BaseCheckpointSaver[int]):
    """Minimal dict-backed saver to illustrate the interface."""

    def __init__(self) -> None:
        super().__init__()
        # {(thread_id, ns, checkpoint_id): (checkpoint, metadata)}
        self._store: dict[tuple, tuple[Checkpoint, CheckpointMetadata]] = {}
        # {(thread_id, ns, checkpoint_id): list[PendingWrite]}
        self._writes: dict[tuple, list] = {}

    def _key(self, config: RunnableConfig) -> tuple:
        c = config.get("configurable", {})
        return (
            c.get("thread_id", ""),
            c.get("checkpoint_ns", ""),
            c.get("checkpoint_id", ""),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        tid = config.get("configurable", {}).get("thread_id", "")
        ns = config.get("configurable", {}).get("checkpoint_ns", "")
        cid = get_checkpoint_id(config)

        if cid:
            key = (tid, ns, cid)
            entry = self._store.get(key)
            if entry is None:
                return None
            cp, meta = entry
            return CheckpointTuple(
                config=config,
                checkpoint=cp,
                metadata=meta,
                pending_writes=self._writes.get(key, []),
            )

        # Return the latest checkpoint for this thread
        candidates = [
            (k, v) for k, v in self._store.items() if k[0] == tid and k[1] == ns
        ]
        if not candidates:
            return None
        (key, (cp, meta)) = max(candidates, key=lambda kv: kv[1][0]["ts"])
        thread_id, ns_, cp_id = key
        result_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns_,
                "checkpoint_id": cp_id,
            }
        }
        return CheckpointTuple(
            config=result_config,
            checkpoint=cp,
            metadata=meta,
            pending_writes=self._writes.get(key, []),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        tid = (config or {}).get("configurable", {}).get("thread_id")
        ns = (config or {}).get("configurable", {}).get("checkpoint_ns", "")

        candidates = sorted(
            [(k, v) for k, v in self._store.items() if k[0] == tid and k[1] == ns],
            key=lambda kv: kv[1][0]["ts"],
            reverse=True,
        )

        count = 0
        for key, (cp, meta) in candidates:
            if limit is not None and count >= limit:
                break
            thread_id, ns_, cp_id = key
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns_,
                        "checkpoint_id": cp_id,
                    }
                },
                checkpoint=cp,
                metadata=meta,
                pending_writes=self._writes.get(key, []),
            )
            count += 1

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = config.get("configurable", {})
        tid = c.get("thread_id", "")
        ns = c.get("checkpoint_ns", "")
        cp_id = checkpoint["id"]
        self._store[(tid, ns, cp_id)] = (checkpoint, metadata)
        return {
            "configurable": {
                "thread_id": tid,
                "checkpoint_ns": ns,
                "checkpoint_id": cp_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        key = self._key(config)
        existing = self._writes.setdefault(key, [])
        existing.extend((task_id, ch, val) for ch, val in writes)


# ── Use it ────────────────────────────────────────────────────────────────────

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: Annotated[int, operator.add]

graph = (
    StateGraph(State)
    .add_node("bump", lambda s: {"count": 1})
    .add_edge(START, "bump")
    .add_edge("bump", END)
    .compile(checkpointer=DictSaver())
)

config = {"configurable": {"thread_id": "custom-1"}}
for _ in range(3):
    graph.invoke({"count": 0}, config)

ct = graph.checkpointer.get_tuple(config)
print(ct.checkpoint["channel_values"])  # {"__end__": {"count": 3}}
```

### What `get_next_version` does

`BaseCheckpointSaver` provides a default `get_next_version(current, channel)` that increments an integer. Override it if your backend uses strings or floats:

```python
def get_next_version(self, current: int | None, channel: Any) -> int:
    return (current or 0) + 1
```

---

## 8 · Typed stream parts — `ValuesStreamPart`, `UpdatesStreamPart`, `MessagesStreamPart`, `TasksStreamPart`, `CheckpointStreamPart`

**Module:** `langgraph.types`

The v2 streaming API (added in `langgraph==1.1.x`) wraps every event in a `TypedDict` so type checkers can narrow the `data` field by `type`. The full discriminated union is:

```python
StreamPart = (
    ValuesStreamPart[OutputT]
    | UpdatesStreamPart
    | MessagesStreamPart
    | CustomStreamPart
    | CheckpointStreamPart[StateT]
    | TasksStreamPart
    | DebugStreamPart[StateT]
)
```

### Part schemas (from source)

```python
class ValuesStreamPart(TypedDict, Generic[OutputT]):
    type: Literal["values"]
    ns: tuple[str, ...]           # namespace (empty = root graph)
    data: OutputT                 # full state after the step
    interrupts: tuple[Interrupt, ...]

class UpdatesStreamPart(TypedDict):
    type: Literal["updates"]
    ns: tuple[str, ...]
    data: dict[str, Any]          # {node_name: node_output, …}

class MessagesStreamPart(TypedDict):
    type: Literal["messages"]
    ns: tuple[str, ...]
    data: tuple[AnyMessage, dict[str, Any]]  # (message_chunk, metadata)

class CustomStreamPart(TypedDict):
    type: Literal["custom"]
    ns: tuple[str, ...]
    data: Any                     # whatever StreamWriter wrote

class CheckpointStreamPart(TypedDict, Generic[StateT]):
    type: Literal["checkpoints"]
    ns: tuple[str, ...]
    data: CheckpointPayload[StateT]

class TasksStreamPart(TypedDict):
    type: Literal["tasks"]
    ns: tuple[str, ...]
    data: TaskPayload | TaskResultPayload
```

### Type-safe multi-mode streaming

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import (
    ValuesStreamPart,
    UpdatesStreamPart,
    CustomStreamPart,
    StreamWriter,
)

class State(TypedDict):
    count: Annotated[int, operator.add]

def tick(state: State, writer: StreamWriter) -> dict:
    writer(f"tick: count is now {state['count'] + 1}")
    return {"count": 1}

graph = (
    StateGraph(State)
    .add_node("tick", tick)
    .add_edge(START, "tick")
    .add_edge("tick", END)
    .compile()
)

# Request multiple stream modes at once
for part in graph.stream(
    {"count": 0},
    stream_mode=["updates", "custom"],
    version="v2",
):
    match part["type"]:
        case "updates":
            up: UpdatesStreamPart = part
            print("update →", up["data"])
        case "custom":
            cp: CustomStreamPart = part
            print("custom →", cp["data"])
```

### Narrowing `MessagesStreamPart` for token-by-token display

```python
from langgraph.types import MessagesStreamPart
from langchain_core.messages import AIMessageChunk

for part in graph.stream(input_, stream_mode="messages", version="v2"):
    if part["type"] == "messages":
        mp: MessagesStreamPart = part
        message, meta = mp["data"]
        if isinstance(message, AIMessageChunk) and message.content:
            print(message.content, end="", flush=True)
        ns = mp["ns"]   # empty tuple = root; ("subgraph:abc",) = nested
```

### Iterating `CheckpointStreamPart` for audit trails

```python
from langgraph.types import CheckpointStreamPart, CheckpointPayload

for part in graph.stream(input_, stream_mode="checkpoints", version="v2"):
    if part["type"] == "checkpoints":
        payload: CheckpointPayload = part["data"]
        print(f"step {payload['metadata'].get('step')} → {payload['next']}")
        # payload['values']  — full state at this point
        # payload['tasks']   — list of CheckpointTask (with errors, results)
```

### Checking for interrupts in `ValuesStreamPart`

```python
from langgraph.types import ValuesStreamPart, Interrupt

for part in graph.stream(input_, stream_mode="values", version="v2"):
    if part["type"] == "values":
        vp: ValuesStreamPart = part
        if vp["interrupts"]:
            iv: Interrupt = vp["interrupts"][0]
            print(f"Interrupted — value: {iv.value!r}, id: {iv.id}")
```

---

## 9 · `task.clear_cache()` and `task.aclear_cache()`

**Module:** `langgraph.func` (on the `_TaskFunction` wrapper returned by `@task`)

Every `@task`-decorated function is wrapped in `_TaskFunction`, which exposes two cache-invalidation helpers: `clear_cache(cache)` and `aclear_cache(cache)`. They delete only the specific namespace used by that task — not the entire cache.

### Source implementation

```python
# Inside _TaskFunction
def clear_cache(self, cache: BaseCache) -> None:
    if self.cache_policy is not None:
        cache.clear(((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),))

async def aclear_cache(self, cache: BaseCache) -> None:
    if self.cache_policy is not None:
        await cache.aclear(
            ((CACHE_NS_WRITES, identifier(self.func) or "__dynamic__"),)
        )
```

The namespace is `("langgraph", "cache", "writes", "<function_identifier>")`. If the task is a lambda or a `functools.partial` without a stable `__qualname__`, the identifier falls back to `"__dynamic__"`.

### Full example: selective cache invalidation

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy

cache = InMemoryCache()
call_counts = {"fetch": 0, "process": 0}

@task(cache_policy=CachePolicy(ttl=3600))
def fetch_data(key: str) -> str:
    call_counts["fetch"] += 1
    return f"data_for_{key}"

@task(cache_policy=CachePolicy(ttl=3600))
def process_data(data: str) -> str:
    call_counts["process"] += 1
    return data.upper()

class State(TypedDict):
    key: str
    result: str

@entrypoint(checkpointer=InMemorySaver(), cache=cache)
def pipeline(key: str) -> str:
    raw = fetch_data(key).result()
    return process_data(raw).result()

config = {"configurable": {"thread_id": "cache-test"}}

# First run — both tasks execute
pipeline.invoke("mykey", config)
print(call_counts)  # {'fetch': 1, 'process': 1}

# Second run — both served from cache
pipeline.invoke("mykey", config)
print(call_counts)  # {'fetch': 1, 'process': 1}

# Invalidate only fetch_data's cache
fetch_data.clear_cache(cache)

# Third run — only fetch re-executes; process still cached
pipeline.invoke("mykey", config)
print(call_counts)  # {'fetch': 2, 'process': 1}

# Invalidate process_data's cache
await process_data.aclear_cache(cache)  # async version
```

### Cache key composition

The internal cache key is a hash of the task's arguments (via `xxh3_128_hexdigest` from `xxhash`). The namespace groups all key-value pairs for one task function. Calling `clear_cache` deletes the entire namespace, not a specific argument combination.

To invalidate a single argument set, you would need to use the `BaseCache` API directly and supply the exact `FullKey` — not exposed as public API.

---

## 10 · `HumanInterruptConfig` / `HumanInterrupt` / `HumanResponse`

**Module:** `langgraph.prebuilt.interrupt` (deprecated — see migration note below)

These three `TypedDict`s define a **structured human-in-the-loop protocol** for communication between the graph and an external UI or approval layer. They standardise what options a human can take (ignore, respond, edit, accept) and what the response looks like.

> **Deprecation:** `HumanInterruptConfig`, `ActionRequest`, `HumanInterrupt`, and `HumanResponse` have been moved to `langchain.agents.interrupt` as of LangGraph 1.0. The `langgraph.prebuilt.interrupt` versions emit `LangGraphDeprecatedSinceV10` warnings. Migrate your imports.

### Schema (source-verified)

```python
class HumanInterruptConfig(TypedDict):
    allow_ignore:  bool   # Human can skip this step
    allow_respond: bool   # Human can send text feedback
    allow_edit:    bool   # Human can edit the content
    allow_accept:  bool   # Human can approve as-is

class ActionRequest(TypedDict):
    action: str           # e.g. "run_shell_command"
    args:   dict          # e.g. {"cmd": "rm -rf /tmp/old"}

class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config:         HumanInterruptConfig
    description:    str | None

class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
```

### New import path (migration)

```python
# Old (deprecated) — still works but emits a warning
from langgraph.prebuilt.interrupt import (
    HumanInterruptConfig,
    ActionRequest,
    HumanInterrupt,
    HumanResponse,
)

# New (correct)
from langchain.agents.interrupt import (
    HumanInterruptConfig,
    ActionRequest,
    HumanInterrupt,
    HumanResponse,
)
```

### Full pattern: tool-call approval with structured HITL

```python
from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, ToolCall
from langchain.agents.interrupt import (
    ActionRequest,
    HumanInterruptConfig,
    HumanInterrupt,
    HumanResponse,
)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

# ── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    approved_calls: Annotated[list[str], operator.add]

# ── Simulated LLM output ─────────────────────────────────────────────────────

def call_llm(state: State) -> dict:
    """Pretend the LLM wants to run a shell command."""
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            ToolCall(
                name="run_shell",
                args={"cmd": "ls /etc"},
                id="tc-001",
                type="tool_call",
            )
        ],
    )
    return {"messages": [ai_msg]}

# ── Approval gate ─────────────────────────────────────────────────────────────

def human_approval(state: State) -> dict | Command:
    last: AIMessage = state["messages"][-1]
    if not last.tool_calls:
        return {}

    for call in last.tool_calls:
        request = HumanInterrupt(
            action_request=ActionRequest(
                action=call["name"],
                args=call["args"],
            ),
            config=HumanInterruptConfig(
                allow_ignore=True,
                allow_respond=True,
                allow_edit=True,
                allow_accept=True,
            ),
            description=f"Approve running `{call['name']}` with {call['args']}?",
        )

        # Pause and hand control to external UI
        response: HumanResponse = interrupt(request)

        if response["type"] == "accept":
            return {"approved_calls": [call["id"]]}
        elif response["type"] == "ignore":
            # Skip this call — route to end
            return Command(goto=END)
        elif response["type"] == "edit":
            edited: ActionRequest = response["args"]
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            ToolCall(
                                name=edited["action"],
                                args=edited["args"],
                                id=call["id"],
                                type="tool_call",
                            )
                        ],
                    )
                ],
                "approved_calls": [call["id"]],
            }
        elif response["type"] == "response":
            # Human provided text feedback — re-route to LLM
            return Command(goto="call_llm")

    return {}

# ── Tool executor ─────────────────────────────────────────────────────────────

def run_tools(state: State) -> dict:
    # In production, execute only approved calls
    results = [f"ran_{cid}" for cid in state.get("approved_calls", [])]
    return {}

# ── Graph ─────────────────────────────────────────────────────────────────────

graph = (
    StateGraph(State)
    .add_node("call_llm", call_llm)
    .add_node("human_approval", human_approval)
    .add_node("run_tools", run_tools)
    .add_edge(START, "call_llm")
    .add_edge("call_llm", "human_approval")
    .add_edge("human_approval", "run_tools")
    .add_edge("run_tools", END)
    .compile(checkpointer=InMemorySaver())
)

# ── Invocation ────────────────────────────────────────────────────────────────

config = {"configurable": {"thread_id": "hitl-demo"}}

# First run — graph pauses at human_approval
events = list(graph.stream({"messages": [], "approved_calls": []}, config))
print("Paused — last event:", events[-1])

# Resume with acceptance
for event in graph.stream(
    Command(resume=HumanResponse(type="accept", args=None)),
    config,
):
    print(event)
```

### Type narrowing in your approval handler

```python
def handle_response(r: HumanResponse) -> None:
    match r["type"]:
        case "accept":
            # r["args"] is None
            approve_all()
        case "ignore":
            # r["args"] is None
            skip()
        case "response":
            feedback: str = r["args"]
            send_feedback_to_llm(feedback)
        case "edit":
            edited: ActionRequest = r["args"]
            run_with_edits(edited["action"], edited["args"])
```

---

## Quick-reference: which module exports what

| Symbol | Import path |
|---|---|
| `RedisCache` | `langgraph.cache.redis` |
| `InMemoryCache` | `langgraph.cache.memory` |
| `EncryptedSerializer` | `langgraph.checkpoint.serde.encrypted` |
| `JsonPlusSerializer` | `langgraph.checkpoint.serde.jsonplus` |
| `UntrackedValue` | `langgraph.channels` |
| `AnyValue` | `langgraph.channels` |
| `EmbeddingsLambda`, `ensure_embeddings` | `langgraph.store.base.embed` |
| `BaseCheckpointSaver`, `CheckpointTuple`, `Checkpoint`, `CheckpointMetadata` | `langgraph.checkpoint.base` |
| `ValuesStreamPart`, `UpdatesStreamPart`, `MessagesStreamPart`, `CustomStreamPart`, `CheckpointStreamPart`, `TasksStreamPart` | `langgraph.types` |
| `HumanInterruptConfig`, `ActionRequest`, `HumanInterrupt`, `HumanResponse` | `langchain.agents.interrupt` *(new)* |

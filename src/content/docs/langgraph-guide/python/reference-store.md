---
title: "Store (long-term memory) — API reference"
description: "BaseStore, InMemoryStore, IndexConfig, TTLConfig — the cross-thread key-value-plus-vector-search layer LangGraph uses for durable memory across conversations."
framework: langgraph
language: python
sidebar:
  label: "Ref · Store"
  order: 32
---

# Store (long-term memory) — API reference

Verified against **`langgraph==1.1.9`** (`/tmp/langgraph-install/langgraph/store/base/__init__.py`, `/tmp/langgraph-install/langgraph/store/memory/__init__.py`).

Checkpointers give you **short-term** memory tied to a single `thread_id`. A `Store` gives you **long-term** memory that lives outside any thread — shared across conversations, users, and graph runs. Same backend pattern as checkpointers: one abstract base, multiple implementations.

## Minimal runnable example

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

store.put(("users", "alice"), "prefs", {"theme": "dark", "lang": "en"})
item = store.get(("users", "alice"), "prefs")
print(item.value)     # {'theme': 'dark', 'lang': 'en'}
print(item.namespace) # ('users', 'alice')
print(item.key)       # 'prefs'

for hit in store.search(("users",), filter={"theme": "dark"}):
    print(hit.key, hit.value)
```

Wire a store into a graph so nodes and tools can read/write to it:

```python
from langgraph.graph import StateGraph, START
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = (
    StateGraph(State)
    .add_node("recall", recall_fn)
    .add_edge(START, "recall")
    .compile(store=store)
)
```

## Available backends

| Backend | Import | Persists? | Vector search | TTL |
|---|---|---|---|---|
| `InMemoryStore` | `langgraph.store.memory` | No | Yes (numpy if installed) | Optional |
| `PostgresStore` | `langgraph.store.postgres`<sup>1</sup> | Yes | Yes (pgvector) | Yes |
| `AsyncPostgresStore` | `langgraph.store.postgres.aio`<sup>1</sup> | Yes | Yes (pgvector) | Yes |
| `AsyncBatchedBaseStore` | `langgraph.store.base.batch` | Adapter | Same as wrapped | Same as wrapped |

<sup>1</sup> Ships in the separate `langgraph-checkpoint-postgres` package — the same package as `PostgresSaver`.

## Data model

- **`namespace: tuple[str, ...]`** — hierarchical path (e.g., `("users", "123", "memories")`). The prefix is used for listing and scoped searches.
- **`key: str`** — unique within a namespace.
- **`value: dict[str, Any]`** — JSON-serializable payload. Keys are filterable.
- **`Item`** — returned by `get` / `list_namespaces`. Fields: `value`, `key`, `namespace`, `created_at`, `updated_at`.
- **`SearchItem(Item)`** — returned by `search`. Adds `score: float | None`.

Any of these operations can raise `InvalidNamespaceError` (e.g., empty tuple, empty string label, or `"."` in a label).

## `BaseStore` surface

All methods have sync and `a`-prefixed async variants.

```python
store.get(namespace, key, *, refresh_ttl=None) -> Item | None
store.put(namespace, key, value, index=None, *, ttl=NOT_PROVIDED) -> None
store.delete(namespace, key) -> None
store.search(
    namespace_prefix,
    /, *,
    query=None, filter=None, limit=10, offset=0, refresh_ttl=None,
) -> list[SearchItem]
store.list_namespaces(
    *, prefix=None, suffix=None, max_depth=None, limit=100, offset=0,
) -> list[tuple[str, ...]]
store.batch(ops: Iterable[Op]) -> list[Result]
```

Under the hood, every single-item method funnels through `batch`/`abatch`. Submit mixed `GetOp`, `PutOp`, `SearchOp`, `ListNamespacesOp` for a single round-trip.

## `put()` — details

```python
store.put(
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
    index: Literal[False] | list[str] | None = None,
    *,
    ttl: float | None | NotProvided = NOT_PROVIDED,
) -> None
```

- `index=None` — use the fields you configured on the store (or none if it is not indexed).
- `index=False` — skip embedding for this item even if the store is indexed.
- `index=["metadata.title", "chapters[*].content"]` — path selectors. Supports:
  - dot-separated nesting (`"a.b.c"`),
  - `[*]` for every array element (each embedded separately),
  - `[0]` for a specific index.
- `ttl` — minutes until expiry. Raises `NotImplementedError` if you pass a value and the backend has `supports_ttl = False`.

## `search()` — filter + semantic

Filtering on `value` keys is exact-match across every backend:

```python
store.search(("docs",), filter={"type": "article", "status": "published"}, limit=20)
```

Semantic search requires `IndexConfig`:

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        # Optional: which fields within `value` to embed. Default: ["$"] (whole value).
        "fields": ["text", "summary"],
    }
)

store.put(("docs",), "d1", {"text": "Rust is a systems language", "type": "lang"})
results = store.search(
    ("docs",),
    query="memory-safe low-level languages",
    filter={"type": "lang"},
    limit=5,
)
for r in results:
    print(r.score, r.value["text"])
```

If the store was not created with `index=`, the `query=` argument is silently ignored and `search` returns plain filtered results.

## `IndexConfig`

```python
class IndexConfig(TypedDict, total=False):
    dims: int
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str
    fields: list[str]     # default ["$"] — embed the entire value
    ann_index_config: ...  # backend-specific (e.g., pgvector tuning)
    distance_type: Literal["l2", "inner_product", "cosine"]
```

`embed` can be:

- a LangChain `Embeddings` instance,
- a sync `(list[str]) -> list[list[float]]`,
- an async callable with the same shape,
- a provider string like `"openai:text-embedding-3-small"` (LangChain resolves it).

## `TTLConfig`

```python
class TTLConfig(TypedDict, total=False):
    refresh_on_read: bool          # default True
    default_ttl: float | None      # minutes for new items
    sweep_interval_minutes: int | None
```

Only set `ttl=...` on `put()` if the backend supports TTL. `InMemoryStore` supports TTL by accepting the kwarg but does not run a background sweeper — items are evicted lazily.

## `list_namespaces`

Explore the tree:

```python
store.list_namespaces(prefix=("users",), max_depth=2)
# [('users', 'alice'), ('users', 'bob'), ...]
```

`prefix` / `suffix` accept `NamespacePath` tuples; use `"*"` as a wildcard segment. `max_depth` caps the tuple length returned.

## `InMemoryStore`

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(*, index: IndexConfig | None = None)
```

- Pure-Python, process-local. Data is lost on exit.
- Vector search uses numpy if installed, falls back to a pure-Python dot product otherwise. For any non-trivial corpus, `pip install numpy`.
- Exposes sync and async methods (batched through `AsyncBatchedBaseStore` under the hood for async).

## `PostgresStore` / `AsyncPostgresStore`

```python
from langgraph.store.postgres import PostgresStore

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()       # creates tables + pgvector extension if index is set
    graph = builder.compile(store=store)
    graph.invoke(..., cfg)
```

- `from_conn_string` is a context manager (same pattern as `PostgresSaver`).
- `setup()` is **required** on first use.
- Pass `index=IndexConfig(...)` to enable pgvector semantic search. Requires the `vector` extension in your database.

Async counterpart lives at `langgraph.store.postgres.aio.AsyncPostgresStore` with an async context manager and `await store.setup()`.

## Using a Store from a node

The `Runtime.store` attribute exposes whatever you passed to `compile(store=...)`:

```python
from langgraph.runtime import Runtime

def recall(state: State, runtime: Runtime) -> dict:
    if runtime.store is None:
        return {"memories": []}
    hits = runtime.store.search(
        ("memories", state["user_id"]),
        query=state["query"],
        limit=3,
    )
    return {"memories": [h.value for h in hits]}
```

## Using a Store from a tool (`InjectedStore`)

Tools get the store injected automatically when wrapped by `ToolNode` (from `langgraph.prebuilt`):

```python
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore, ToolNode
from langgraph.store.base import BaseStore

@tool
def save_fact(
    user_id: str,
    fact: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    store.put(("facts", user_id), fact, {"text": fact})
    return f"Saved for {user_id}"

tool_node = ToolNode([save_fact])
```

The `store` argument is stripped from the schema the model sees, so the LLM cannot pass it. `InjectedState` works the same way for whole-state injection; `ToolRuntime` bundles `state + context + config + store + stream_writer + tool_call_id` into one object.

## Patterns

### 1. Per-user preferences

```python
ns = ("users", user_id, "prefs")
store.put(ns, "theme", {"mode": "dark"})
store.put(ns, "lang", {"code": "en"})
for pref in store.list_namespaces(prefix=("users", user_id)):
    for item in store.search(pref):
        print(pref, item.key, item.value)
```

### 2. Semantic memory with filtered recall

```python
store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
store.put(("mem", "alice"), "m1", {"text": "Likes espresso", "kind": "food"})
store.put(("mem", "alice"), "m2", {"text": "Works at Acme", "kind": "work"})

hits = store.search(
    ("mem", "alice"),
    query="favorite drink",
    filter={"kind": "food"},
    limit=3,
)
```

### 3. Batched write on session end

```python
from langgraph.store.base import PutOp

store.batch([
    PutOp(("mem", user_id), f"m{i}", v, index=None, ttl=None)
    for i, v in enumerate(new_memories)
])
```

### 4. Tools that read *and* write memory

```python
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

@tool
def remember(
    user_id: str,
    text: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    store.put(("mem", user_id), f"note-{text[:16]}", {"text": text})
    return "ok"

@tool
def recall(
    user_id: str,
    topic: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> list[str]:
    return [i.value["text"] for i in store.search(("mem", user_id), query=topic, limit=5)]
```

### 5. TTL-bounded cache

```python
store.put(("cache", "bing"), query, {"json": result}, ttl=30)   # minutes
hit = store.get(("cache", "bing"), query, refresh_ttl=True)
```

## Gotchas

- **Namespace rules.** Each segment must be a non-empty string and must not contain `"."`. `("", "x")` raises `InvalidNamespaceError`.
- **`query=` is ignored without an index.** You will get filter-only results without any warning — always assert `store` was built with `index=IndexConfig(...)` when you rely on semantic search.
- **`fields=["$"]`** means the entire value is stringified and embedded. Pick explicit fields for better recall and smaller embedding costs.
- **`InMemoryStore` is not Platform-safe.** LangGraph Platform provides a managed store — don't pass one when deploying there.
- **TTL is in minutes, not seconds.** A `ttl=30` means half an hour, not 30 seconds.
- **`store.search` returns `list[SearchItem]`, not an iterator.** Always bounded by `limit` (default 10). Paginate with `offset`.
- **`delete` uses `PutOp(...value=None)` internally.** If you subclass `BaseStore`, `PutOp.value is None` is the delete signal.

## Breaking changes

| Version | Change |
|---|---|
| 1.1 | Semantic-search result `SearchItem.score` is consistently `float | None` (previously backend-dependent). |
| 1.0 | `Store` moved out of `experimental`; `InjectedStore` is the stable way to pull the store into tools. |
| 0.6 | `runtime.store` replaces `config["configurable"]["store"]` for node injection. |

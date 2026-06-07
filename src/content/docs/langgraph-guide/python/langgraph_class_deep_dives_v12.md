---
title: "Class deep-dives Vol. 12 — Production infrastructure & advanced streaming"
description: "Source-verified deep dives into RemoteGraph/RemoteException, PostgresSaver/ShallowPostgresSaver, AsyncPostgresSaver, PostgresStore/PoolConfig, AsyncPostgresStore, ANNIndexConfig/HNSWConfig/IVFFlatConfig/PostgresIndexConfig, GraphRunStream/SubgraphRunStream, ToolCallWithContext/ToolInvocationError, LifecyclePayload/LifecycleTransformer, and MessagesTransformer/CheckpointsTransformer/TasksTransformer — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 12"
  order: 43
---

# Class deep-dives Vol. 12 — Production infrastructure & advanced streaming

Verified against **`langgraph==1.2.4`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-sdk==0.1.x`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1–11 index at the bottom of this page](#vol-index)

---

## 1 · `RemoteGraph` + `RemoteException` — call a deployed graph as a node

**Modules:** `langgraph.pregel.remote`  
**Imports:**
```python
from langgraph.pregel.remote import RemoteGraph, RemoteException
```

`RemoteGraph` is a `PregelProtocol` implementation that wraps any LangGraph Server–compatible HTTP API — a LangSmith deployment, a self-hosted `langgraph-cli` server, or any service implementing the LangGraph Server spec — and makes it behave exactly like a local compiled graph. You can pass it to `add_node` just as you would a local subgraph.

### Source signature (1.2.4)

```python
class RemoteGraph(PregelProtocol):
    def __init__(
        self,
        assistant_id: str,   # graph_id or assistant_id on the server
        /,
        *,
        url: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        client: LangGraphClient | None = None,
        sync_client: SyncLangGraphClient | None = None,
        config: RunnableConfig | None = None,
        name: str | None = None,
        distributed_tracing: bool = False,
    ): ...
```

`api_key` falls back to `LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY` in that order. At least one of `url`, `client`, or `sync_client` must be provided.

### Example 1: Standalone remote invoke

```python
from langgraph.pregel.remote import RemoteGraph

# Point at a self-hosted LangGraph server running locally
remote = RemoteGraph(
    "my_agent",
    url="http://localhost:2024",
    api_key="local-key",
)

# invoke() / ainvoke() / stream() / astream() all work
result = remote.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={"configurable": {"thread_id": "thread-1"}},
)
print(result["messages"][-1].content)
```

### Example 2: Compose a remote graph into a local graph

The real power is using a remote deployment as a node inside a local orchestration graph. LangGraph transparently handles checkpoint namespacing across the boundary.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.pregel.remote import RemoteGraph
from typing import Annotated

class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]

# The remote graph behaves as a plain node: receives state, returns state update
research_agent = RemoteGraph(
    "research_agent",
    url="https://your-deployment.langsmith.com",
    api_key="lsv2_...",
)

def route(state: OrchestratorState) -> str:
    last = state["messages"][-1]
    if "research" in last.content.lower():
        return "research"
    return END

builder = StateGraph(OrchestratorState)
builder.add_node("research", research_agent)
builder.add_conditional_edges(START, route, {"research": "research", END: END})
builder.add_edge("research", END)

graph = builder.compile()
```

### Example 3: Streaming from a remote graph

```python
import asyncio
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph("my_agent", url="http://localhost:2024")

async def stream_remote():
    async for chunk in remote.astream(
        {"messages": [{"role": "user", "content": "What is quantum computing?"}]},
        config={"configurable": {"thread_id": "t1"}},
        stream_mode=["values", "messages"],
    ):
        print(chunk)

asyncio.run(stream_remote())
```

### Example 4: State management on remote threads

`RemoteGraph` proxies all `StateSnapshot` operations through the LangGraph Server API:

```python
config = {"configurable": {"thread_id": "my-thread"}}

# Get current state
snapshot = remote.get_state(config)
print(snapshot.values)
print(snapshot.next)

# Update state directly (e.g. to inject context or correct a mistake)
remote.update_state(config, {"messages": [{"role": "user", "content": "extra context"}]})

# Replay history
for state in remote.get_state_history(config):
    print(state.config["configurable"]["checkpoint_id"], state.values.get("messages", [])[-1])
```

### `RemoteException`

`RemoteException` is a plain `Exception` subclass raised when the remote server returns an error response. Catch it to distinguish remote failures from local ones:

```python
from langgraph.pregel.remote import RemoteException

try:
    result = remote.invoke({"messages": [...]}, config=config)
except RemoteException as e:
    print(f"Remote graph failed: {e}")
```

### `with_config` and `copy`

Both methods return a new `RemoteGraph` with merged or replaced config — useful when you need thread-local overrides:

```python
scoped = remote.with_config({"configurable": {"thread_id": "t-99", "user_id": "u-1"}})
result = scoped.invoke({"messages": [...]})
```

---

## 2 · `PostgresSaver` + `ShallowPostgresSaver` — production checkpoint backends

**Module:** `langgraph.checkpoint.postgres`  
**Imports:**
```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.shallow import ShallowPostgresSaver
```

Both classes persist graph checkpoints in PostgreSQL (via `psycopg3` + optional `psycopg_pool`). The difference is storage strategy:

| | `PostgresSaver` | `ShallowPostgresSaver` |
|---|---|---|
| History depth | Full — every step stored | Latest only — single row per `thread_id + ns` |
| `get_state_history()` | ✅ Full time-travel | ❌ Only current state |
| Storage cost | Grows with steps | Constant per thread |
| Best for | Debugging, auditing, HITL | Production stateful agents |

### `PostgresSaver` — setup and first use

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int

def inc(state: State) -> dict:
    return {"counter": state["counter"] + 1}

DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable"

# from_conn_string is a context manager — handles connect and close
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # creates tables + runs migrations; call ONCE

    builder = StateGraph(State)
    builder.add_node("inc", inc)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "t1"}}
    print(graph.invoke({"counter": 0}, config))  # {'counter': 1}
    print(graph.invoke({"counter": 0}, config))  # {'counter': 2}  (resumes from checkpoint)
```

### Pipeline mode for throughput

```python
with PostgresSaver.from_conn_string(DB_URI, pipeline=True) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    # All writes are batched in a single pipeline round-trip per step
```

### Connection pool (high concurrency)

Pass an existing `psycopg.Connection` or `psycopg_pool.ConnectionPool` directly:

```python
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

pool = ConnectionPool(
    DB_URI,
    min_size=2,
    max_size=10,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)

checkpointer = PostgresSaver(conn=pool)
checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)
```

### `ShallowPostgresSaver` — constant-space persistence

```python
from langgraph.checkpoint.postgres.shallow import ShallowPostgresSaver

with ShallowPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "shallow-t1"}}
    graph.invoke({"counter": 0}, config)  # saves one row
    graph.invoke({"counter": 0}, config)  # overwrites that row
    # get_state_history returns only the current checkpoint
```

### Time-travel with `PostgresSaver`

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "history-t1"}}

    for i in range(5):
        graph.invoke({"counter": 0}, config)

    # Replay every checkpoint
    for snapshot in graph.get_state_history(config):
        print(snapshot.config["configurable"]["checkpoint_id"], snapshot.values)

    # Fork from step 2 (use its checkpoint_id as the new thread's parent)
    snapshots = list(graph.get_state_history(config))
    step2_config = snapshots[-3].config  # oldest first, so -3 ≈ step 2
    graph.invoke({"counter": 0}, {**config, "configurable": {**config["configurable"], **step2_config["configurable"]}})
```

---

## 3 · `AsyncPostgresSaver` — async checkpoint backend

**Module:** `langgraph.checkpoint.postgres.aio`  
**Import:** `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver`

`AsyncPostgresSaver` is the async twin of `PostgresSaver`. It uses `asyncio.Lock` instead of `threading.Lock` and accepts `AsyncConnection` / `AsyncConnectionPool`.

### Example 1: Async from_conn_string

```python
import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int

def inc(state: State) -> dict:
    return {"counter": state["counter"] + 1}

DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable"

async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()   # async migration

        builder = StateGraph(State)
        builder.add_node("inc", inc)
        builder.add_edge(START, "inc")
        builder.add_edge("inc", END)
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "async-t1"}}
        result = await graph.ainvoke({"counter": 0}, config)
        print(result)  # {'counter': 1}

asyncio.run(main())
```

### Example 2: Async connection pool

```python
import asyncio
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def build_with_pool():
    async with AsyncConnectionPool(
        DB_URI,
        min_size=2,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    ) as pool:
        checkpointer = AsyncPostgresSaver(conn=pool)
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        return graph
```

### Example 3: Pipeline mode for bulk writes

```python
async with AsyncPostgresSaver.from_conn_string(DB_URI, pipeline=True) as checkpointer:
    await checkpointer.setup()
    # checkpoint writes are pipeline-batched — better throughput under load
    graph = builder.compile(checkpointer=checkpointer)
```

### Key differences vs sync `PostgresSaver`

| | `PostgresSaver` | `AsyncPostgresSaver` |
|---|---|---|
| Lock | `threading.Lock` | `asyncio.Lock` |
| Connection type | `Connection` / `ConnectionPool` | `AsyncConnection` / `AsyncConnectionPool` |
| `setup()` | sync | `async def setup()` |
| graph methods | `invoke`, `stream` | `ainvoke`, `astream` |

---

## 4 · `PostgresStore` + `PoolConfig` — persistent long-term memory

**Module:** `langgraph.store.postgres`  
**Imports:**
```python
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.base import PoolConfig
```

`PostgresStore` provides durable key-value storage with optional pgvector semantic search. Unlike checkpointers (which are scoped per thread), stores are shared across all threads — perfect for user preferences, long-term facts, and cross-session memory.

### Source signatures

```python
class PostgresStore(BaseStore, BasePostgresStore[Conn]):
    def __init__(
        self,
        conn: Conn,       # Connection | ConnectionPool
        *,
        pipe: Pipeline | None = None,
        deserializer: Callable | None = None,
        index: PostgresIndexConfig | None = None,   # enables vector search
        ttl: TTLConfig | None = None,               # optional expiry
    ): ...

class PoolConfig(TypedDict, total=False):
    min_size: int          # default 1
    max_size: int | None   # None = unlimited
    kwargs: dict           # extra psycopg connection args
```

### Example 1: Basic key-value store

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph"

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()   # creates the store table; call once

    # Namespaces are tuples of strings — think of them as directory paths
    store.put(("users", "alice"), "preferences", {"theme": "dark", "lang": "en"})
    store.put(("users", "alice"), "session_count", {"count": 42})

    item = store.get(("users", "alice"), "preferences")
    print(item.value)         # {'theme': 'dark', 'lang': 'en'}
    print(item.created_at)    # datetime

    # List all items in a namespace
    items = store.list_namespaces(match=("users",))
    print(items)  # [('users', 'alice')]

    # Delete
    store.delete(("users", "alice"), "session_count")
```

### Example 2: Connection pool for production

```python
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.base import PoolConfig

pool_cfg: PoolConfig = {
    "min_size": 2,
    "max_size": 20,
    "kwargs": {"application_name": "my_agent"},
}

with PostgresStore.from_conn_string(DB_URI, pool_config=pool_cfg) as store:
    store.setup()
    graph = builder.compile(store=store, checkpointer=checkpointer)
```

### Example 3: TTL (time-to-live) expiry

```python
from langgraph.store.base import TTLConfig

ttl_cfg: TTLConfig = {
    "default_ttl": 60 * 24 * 7,      # 7 days in minutes
    "sweep_interval_minutes": 30,     # background cleanup interval
}

with PostgresStore.from_conn_string(DB_URI, ttl=ttl_cfg) as store:
    store.setup()
    # Must explicitly start the background sweeper
    store.start_ttl_sweeper()

    store.put(("sessions",), "user-xyz", {"token": "abc123"})
    # Item expires 7 days after last write

    # Override TTL per item (in minutes)
    store.put(("sessions",), "short-lived", {"token": "xyz"}, ttl=60)  # 1 hour

    # Items are also lazily expired on read
    item = store.get(("sessions",), "short-lived")  # None if expired

    store.stop_ttl_sweeper()   # clean shutdown
```

### Example 4: Using `PostgresStore` in a graph node via `InjectedStore`

```python
from typing import Annotated
from langgraph.prebuilt import InjectedStore, ToolNode
from langgraph.store.base import BaseStore
from langchain_core.tools import tool

@tool
def remember_fact(
    fact: str,
    user_id: str,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Store a fact about the user."""
    existing = store.get(("facts", user_id), "all") or {"items": []}
    existing["items"].append(fact)
    store.put(("facts", user_id), "all", existing)
    return f"Remembered: {fact}"

tool_node = ToolNode([remember_fact])
```

---

## 5 · `AsyncPostgresStore` — async long-term memory

**Module:** `langgraph.store.postgres.aio`  
**Import:** `from langgraph.store.postgres import AsyncPostgresStore`

`AsyncPostgresStore` extends `AsyncBatchedBaseStore`, which batches concurrent coroutine-level store operations into single SQL round-trips — critical in async multi-agent settings where many coroutines hit the store simultaneously.

### Example 1: Async setup and basic operations

```python
import asyncio
from langgraph.store.postgres import AsyncPostgresStore

async def main():
    async with AsyncPostgresStore.from_conn_string(DB_URI) as store:
        await store.setup()

        await store.aput(("users", "bob"), "prefs", {"theme": "light"})
        item = await store.aget(("users", "bob"), "prefs")
        print(item.value)  # {'theme': 'light'}

        await store.adelete(("users", "bob"), "prefs")

asyncio.run(main())
```

### Example 2: Async connection pool

```python
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.store.postgres.base import PoolConfig

async def build_store():
    pool_cfg: PoolConfig = {"min_size": 2, "max_size": 30}
    async with AsyncPostgresStore.from_conn_string(
        DB_URI, pool_config=pool_cfg
    ) as store:
        await store.setup()
        graph = builder.compile(store=store, checkpointer=checkpointer)
        return graph
```

### Example 3: Batch operations (the async advantage)

`AsyncBatchedBaseStore` coalesces overlapping ops. When many concurrent tasks call `aget` / `aput`, they are bundled into one SQL query per type:

```python
import asyncio
from langgraph.store.postgres import AsyncPostgresStore

async def populate(store: AsyncPostgresStore):
    # These three aput calls fire concurrently and are batched into one round-trip
    await asyncio.gather(
        store.aput(("docs",), "d1", {"text": "alpha"}),
        store.aput(("docs",), "d2", {"text": "beta"}),
        store.aput(("docs",), "d3", {"text": "gamma"}),
    )

async def main():
    async with AsyncPostgresStore.from_conn_string(DB_URI) as store:
        await store.setup()
        await populate(store)

        # Batch read
        results = await asyncio.gather(
            store.aget(("docs",), "d1"),
            store.aget(("docs",), "d2"),
            store.aget(("docs",), "d3"),
        )
        for r in results:
            print(r.value)
```

---

## 6 · `ANNIndexConfig` + `HNSWConfig` + `IVFFlatConfig` + `PostgresIndexConfig` — vector search configuration

**Module:** `langgraph.store.postgres.base`  
**Imports:**
```python
from langgraph.store.postgres.base import (
    ANNIndexConfig,
    HNSWConfig,
    IVFFlatConfig,
    PostgresIndexConfig,
    PoolConfig,
)
```

These TypedDicts configure pgvector ANN (Approximate Nearest Neighbour) indexes on the `PostgresStore`. Without an index config only exact cosine/L2 search is available (slow at scale).

### Class hierarchy

```
IndexConfig          ← from langgraph.store.base (sets dims, embed, fields)
  └── PostgresIndexConfig  ← adds ann_index_config, distance_type
        └── ann_index_config: ANNIndexConfig
              ├── HNSWConfig   (kind="hnsw", m, ef_construction)
              └── IVFFlatConfig (kind="ivfflat", nlist)
```

### `HNSWConfig` — default, best for most cases

```python
class HNSWConfig(ANNIndexConfig, total=False):
    kind: Literal["hnsw"]
    m: int              # max connections per layer (default 16)
    ef_construction: int  # candidate list size during build (default 64)
```

HNSW builds a multi-layer proximity graph. Higher `m` and `ef_construction` improve recall at the cost of build time and memory. **Use HNSW when you need consistent high recall and your dataset changes frequently.**

### `IVFFlatConfig` — faster builds, lower memory

```python
class IVFFlatConfig(ANNIndexConfig, total=False):
    kind: Literal["ivfflat"]
    nlist: int   # number of clusters; rule of thumb: sqrt(rows) for >1M rows
```

IVFFlat divides vectors into `nlist` clusters and searches a subset at query time. **Use IVFFlat when build time matters more than query speed, or for very large static datasets.**

### Example 1: Default HNSW index

```python
from langchain.embeddings import init_embeddings
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig, HNSWConfig

index: PostgresIndexConfig = {
    "dims": 1536,
    "embed": init_embeddings("openai:text-embedding-3-small"),
    "fields": ["text", "summary"],   # embed only these JSON fields
    "distance_type": "cosine",
    "ann_index_config": HNSWConfig(
        kind="hnsw",
        m=16,
        ef_construction=64,
    ),
}

with PostgresStore.from_conn_string(DB_URI, index=index) as store:
    store.setup()   # creates pgvector extension + index

    store.put(("knowledge",), "k1", {"text": "Python is a dynamic language"})
    store.put(("knowledge",), "k2", {"text": "Rust ensures memory safety"})
    store.put(("knowledge",), "k3", {"text": "TypeScript adds static types to JS"})

    results = store.search(("knowledge",), query="type-safe programming", limit=2)
    for r in results:
        print(f"{r.score:.3f}  {r.value['text']}")
```

### Example 2: IVFFlat for large corpus

```python
from langgraph.store.postgres.base import IVFFlatConfig, PostgresIndexConfig

# For 1M documents, nlist ≈ sqrt(1_000_000) = 1000
index: PostgresIndexConfig = {
    "dims": 1536,
    "embed": init_embeddings("openai:text-embedding-3-small"),
    "ann_index_config": IVFFlatConfig(
        kind="ivfflat",
        nlist=1000,
    ),
    "distance_type": "l2",
}

# IMPORTANT: create index AFTER inserting data for best performance
with PostgresStore.from_conn_string(DB_URI, index=index) as store:
    store.setup()
    # bulk insert first
    for i, doc in enumerate(corpus):
        store.put(("docs",), f"doc-{i}", {"text": doc})
    # then run setup again (or a separate migration) to build the index
```

### Example 3: Half-precision vectors (reduced memory)

```python
index: PostgresIndexConfig = {
    "dims": 3072,
    "embed": init_embeddings("openai:text-embedding-3-large"),
    "ann_index_config": HNSWConfig(
        kind="hnsw",
        vector_type="halfvec",   # stores 16-bit floats — halves memory
        m=32,
        ef_construction=100,
    ),
    "distance_type": "cosine",
}
```

### Example 4: Async store with vector search

```python
from langgraph.store.postgres import AsyncPostgresStore

async def semantic_memory():
    async with AsyncPostgresStore.from_conn_string(DB_URI, index=index) as store:
        await store.setup()

        await store.aput(("facts",), "f1", {"text": "The user prefers dark mode"})
        await store.aput(("facts",), "f2", {"text": "The user is a Python developer"})

        results = await store.asearch(("facts",), query="UI preferences", limit=1)
        print(results[0].value["text"])  # 'The user prefers dark mode'
```

---

## 7 · `GraphRunStream` + `SubgraphRunStream` — experimental v3 streaming protocol

**Module:** `langgraph.stream.run_stream`  
**Imports:**
```python
from langgraph.stream.run_stream import GraphRunStream, SubgraphRunStream
```

`GraphRunStream` is the object returned by `graph.stream_events(version="v3")`. Unlike the older `stream()` API, `stream_events(version="v3")` gives you **typed projections** — separate iterable handles for values, messages, lifecycle events, and raw protocol events — with **caller-driven pumping** (no background thread).

> **Stability note:** `stream_events(version="v3")` and `GraphRunStream` are marked `@beta` in 1.2.4. The API may change in future releases.

### `GraphRunStream` — core attributes

```python
class GraphRunStream:
    # Native projections (set as direct attributes):
    values: StreamChannel[dict]          # state snapshots at each step
    messages: StreamChannel[ChatModelStream]  # per-LLM-call streaming handles
    lifecycle: StreamChannel[LifecyclePayload]  # subgraph start/stop events

    # Access raw protocol events:
    def __iter__(self) -> Iterator[ProtocolEvent]: ...

    # Terminal state helpers (drive to completion then return):
    @property
    def output(self) -> dict | None: ...
    @property
    def interrupted(self) -> bool: ...
    @property
    def interrupts(self) -> list[Any]: ...

    def abort(self) -> None: ...   # stop early
```

### Example 1: Iterate values (state snapshots)

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

model = ChatAnthropic(model="claude-haiku-4-5")
agent = create_react_agent(model, [add])

with agent.stream_events(
    {"messages": [{"role": "user", "content": "What is 17 + 25?"}]},
    version="v3",
) as run:
    for snapshot in run.values:
        # snapshot is the full state dict after each node
        msgs = snapshot.get("messages", [])
        if msgs:
            print(f"[{type(msgs[-1]).__name__}] {getattr(msgs[-1], 'content', '')[:80]}")
    print("Final output:", run.output)
```

### Example 2: Stream LLM tokens via `run.messages`

Each item from `run.messages` is a `ChatModelStream` handle — iterate its `.text` projection to get individual token chunks:

```python
with agent.stream_events(
    {"messages": [{"role": "user", "content": "Explain async/await briefly"}]},
    version="v3",
) as run:
    for chat_stream in run.messages:
        print(f"\n--- new LLM call (node={chat_stream.node}) ---")
        for chunk in chat_stream.text:
            print(chunk, end="", flush=True)
    print()
```

### Example 3: Async streaming

```python
import asyncio

async def run_async():
    async with agent.astream_events(
        {"messages": [{"role": "user", "content": "Add 7 + 8"}]},
        version="v3",
    ) as run:
        async for chat_stream in run.messages:
            async for chunk in chat_stream.text:
                print(chunk, end="", flush=True)
        print()
        print("Interrupted:", run.interrupted)

asyncio.run(run_async())
```

### Example 4: `SubgraphRunStream` — navigate nested subgraph streams

When a graph contains subgraphs, each subgraph gets its own `SubgraphRunStream` accessible through `run.extensions["subgraphs"]`:

```python
# Assuming a graph with a subgraph node named "planner"
with outer_graph.stream_events(
    {"messages": [{"role": "user", "content": "Plan a trip to Paris"}]},
    version="v3",
) as run:
    # Top-level lifecycle events
    for event in run.lifecycle:
        print(f"Subgraph lifecycle: {event['event']} @ {event['namespace']}")

    # Drill into subgraphs
    if "subgraphs" in run.extensions:
        for subgraph_handle in run.extensions["subgraphs"]:
            print(f"Subgraph: {subgraph_handle.name}")
            for snapshot in subgraph_handle.values:
                print("  state:", list(snapshot.keys()))
```

### Example 5: `abort()` and early termination

```python
with agent.stream_events({"messages": [...]}, version="v3") as run:
    count = 0
    for snapshot in run.values:
        count += 1
        if count >= 3:
            run.abort()   # stop after 3 steps
            break
    print("Stopped early. Last state:", run.output)
```

### Example 6: `interleave()` — merge projections by arrival order

```python
with agent.stream_events({"messages": [...]}, version="v3") as run:
    for name, item in run.interleave("values", "lifecycle"):
        if name == "values":
            print("snapshot keys:", list(item.keys()))
        else:
            print("lifecycle:", item["event"])
```

---

## 8 · `ToolCallWithContext` + `ToolInvocationError` — tool execution internals

**Module:** `langgraph.prebuilt.tool_node`  
**Imports:**
```python
from langgraph.prebuilt.tool_node import ToolCallWithContext, ToolInvocationError
```

These two types surface details of `ToolNode`'s internal execution model that become important when writing custom error handlers or dispatching tool calls via `Send`.

### `ToolCallWithContext`

```python
class ToolCallWithContext(TypedDict):
    tool_call: ToolCall         # the LangChain ToolCall object
    __type: Literal["tool_call_with_context"]
    state: Any                  # full graph state at dispatch time
```

`ToolCallWithContext` is the payload format used when `create_react_agent` distributes parallel tool calls via `Send`. Normally you never construct this yourself — but you need to recognise it when reading raw task inputs in custom graph nodes.

```python
from langgraph.prebuilt.tool_node import ToolCallWithContext
from langgraph.types import Send

# create_react_agent internally dispatches tool calls like this:
def dispatch_tool_calls(state):
    ai_msg = state["messages"][-1]
    return [
        Send(
            "tools",
            ToolCallWithContext(
                tool_call=tc,
                __type="tool_call_with_context",
                state=state,
            ),
        )
        for tc in ai_msg.tool_calls
    ]
```

### `ToolInvocationError`

```python
class ToolInvocationError(ToolException):
    message: str          # human-readable error (from TOOL_INVOCATION_ERROR_TEMPLATE)
    tool_name: str
    tool_kwargs: dict     # the kwargs that caused the error
    source: ValidationError  # original pydantic error
    filtered_errors: list[ErrorDetails] | None
```

`ToolInvocationError` is raised (internally by `ToolNode`) when a tool call fails pydantic argument validation. By default `ToolNode` catches it and returns a `ToolMessage` with the error text — so the LLM can self-correct. Override `handle_tool_errors` to change this behaviour.

### Example 1: Custom error handler that logs and returns structured JSON

```python
import json
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langchain_core.tools import tool

@tool
def divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator."""
    return numerator / denominator

def my_error_handler(e: Exception) -> str:
    if isinstance(e, ToolInvocationError):
        return json.dumps({
            "error": "validation_failed",
            "tool": e.tool_name,
            "details": e.message,
        })
    raise e   # re-raise anything else

tool_node = ToolNode(
    [divide],
    handle_tool_errors=my_error_handler,
)
```

### Example 2: Differentiating `ToolInvocationError` from runtime exceptions

```python
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langchain_core.tools import ToolException

def categorised_handler(e: Exception) -> str:
    if isinstance(e, ToolInvocationError):
        # Pydantic schema mismatch — LLM sent wrong argument types
        return f"SCHEMA_ERROR: {e.message}"
    elif isinstance(e, ZeroDivisionError):
        return "RUNTIME_ERROR: Division by zero."
    elif isinstance(e, ToolException):
        return f"TOOL_ERROR: {e}"
    raise e   # unexpected — let it bubble up

tool_node = ToolNode([divide], handle_tool_errors=categorised_handler)
```

### Example 3: Inspecting `ToolCallWithContext` in a custom node

```python
from typing import Any
from langgraph.prebuilt.tool_node import ToolCallWithContext

def custom_tool_dispatcher(input_payload: Any) -> dict:
    """A node that accepts ToolCallWithContext and logs extra context."""
    if isinstance(input_payload, dict) and input_payload.get("__type") == "tool_call_with_context":
        payload = input_payload  # cast to ToolCallWithContext
        tool_call = payload["tool_call"]
        state = payload["state"]
        user_id = state.get("user_id", "anonymous")
        print(f"User {user_id} called tool {tool_call['name']}")
    # ... proceed with execution
```

---

## 9 · `LifecyclePayload` + `LifecycleTransformer` — subgraph lifecycle events

**Module:** `langgraph.stream.transformers`  
**Imports:**
```python
from langgraph.stream.transformers import LifecyclePayload, LifecycleTransformer
```

`LifecycleTransformer` is a native stream transformer that watches the `tasks` event stream and synthesises higher-level lifecycle protocol events — one per subgraph invocation. `LifecyclePayload` is its output type.

### `LifecyclePayload` shape

```python
class LifecyclePayload(TypedDict, total=False):
    event: SubgraphStatus      # "started" | "completed" | "failed" | "interrupted" | "drained"
    namespace: list[str]       # checkpoint namespace path of the subgraph
    graph_name: str            # e.g. "research_agent" (when set by create_agent)
    trigger_call_id: str       # Pregel task_id that spawned the subgraph
    cause: LifecycleCause      # {"type": "toolCall", "tool_call_id": "..."} if tool-driven
    error: str                 # only present when event == "failed"
```

### Example 1: Monitor subgraph lifecycle in a multi-agent graph

```python
from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent


class State(TypedDict):
    messages: list


model = ChatAnthropic(model="claude-haiku-4-5")

# A top-level agent that calls a subagent for research
research_sub = create_react_agent(model, [search_tool], name="research_agent")

builder = StateGraph(State)
builder.add_node("orchestrator", create_react_agent(model, [research_sub]))
graph = builder.compile()

with graph.stream_events({"messages": [{"role": "user", "content": "Research LangGraph"}]}, version="v3") as run:
    for event in run.lifecycle:
        status = event["event"]
        name = event.get("graph_name", "<unnamed>")
        ns = " > ".join(event["namespace"])
        cause = event.get("cause", {})
        if cause.get("type") == "toolCall":
            print(f"[{status.upper()}] {name} (triggered by tool call {cause['tool_call_id'][:8]}…)")
        else:
            print(f"[{status.upper()}] {name} @ {ns}")
```

### Example 2: Detect failed subgraphs and alert

```python
async def monitor_lifecycle(graph, input_state):
    async with graph.astream_events(input_state, version="v3") as run:
        async for event in run.lifecycle:
            if event["event"] == "failed":
                print(f"ALERT: subgraph {event.get('graph_name')} failed — {event.get('error')}")
            elif event["event"] == "completed":
                print(f"OK: {event.get('graph_name')} completed")
        return run.output
```

### Example 3: Reconstruct a timing trace from lifecycle events

```python
import time

async def time_subgraphs(graph, input_state):
    timings = {}
    async with graph.astream_events(input_state, version="v3") as run:
        async for event in run.lifecycle:
            ns_key = tuple(event["namespace"])
            name = event.get("graph_name", "?")
            if event["event"] == "started":
                timings[ns_key] = {"name": name, "start": time.monotonic()}
            elif ns_key in timings:
                elapsed = time.monotonic() - timings[ns_key]["start"]
                print(f"{name}: {event['event']} after {elapsed:.2f}s")
```

---

## 10 · `MessagesTransformer` + `CheckpointsTransformer` + `TasksTransformer`

**Module:** `langgraph.stream.transformers`  
**Imports:**
```python
from langgraph.stream.transformers import (
    MessagesTransformer,
    CheckpointsTransformer,
    TasksTransformer,
)
```

These are the three remaining native transformers in the v3 streaming protocol. They all implement the `StreamTransformer` protocol, meaning they are wired into a `StreamMux` automatically when you use `stream_events(version="v3")`.

---

### `MessagesTransformer` — typed per-LLM-call streaming

`MessagesTransformer` captures `messages` protocol events and wraps each LLM call lifecycle in a `ChatModelStream` / `AsyncChatModelStream` object. It filters by namespace scope so nested subgraph LLM calls are not mixed into the outer `run.messages`.

**Key behaviours:**
- One `ChatModelStream` is pushed to `run.messages` per LLM call start event.
- Finalized `AIMessage` objects (from `on_chain_end`) are replayed as a synthetic protocol lifecycle so non-streaming models also populate `run.messages`.
- `ToolMessage` objects are excluded — they appear in `run.values` as state updates.
- V1 `AIMessageChunk` (from `on_llm_new_token`) is NOT processed — models must use the `stream_events(version="v3")` protocol.

### Example 1: Get per-node token counts

```python
with agent.stream_events({"messages": [{"role": "user", "content": "Summarise LangGraph"}]}, version="v3") as run:
    for chat_stream in run.messages:
        # Drain the text projection
        tokens = list(chat_stream.text)
        print(f"Node {chat_stream.node}: {len(tokens)} tokens streamed")
        # Get the final output message
        final_msg = chat_stream.output
        if final_msg:
            usage = getattr(final_msg, "usage_metadata", {})
            print(f"  total tokens: {usage}")
```

### Example 2: Async fan-out — process values and messages simultaneously

```python
import asyncio

async def fan_out():
    async with agent.astream_events(
        {"messages": [{"role": "user", "content": "Explain neural networks"}]},
        version="v3",
    ) as run:
        # interleave() merges projections in arrival order
        async for name, item in run.ainterleave("values", "messages"):
            if name == "values":
                msgs = item.get("messages", [])
                print(f"[snapshot] {len(msgs)} messages in state")
            else:
                # item is an AsyncChatModelStream
                async for chunk in item.text:
                    print(chunk, end="", flush=True)

asyncio.run(fan_out())
```

---

### `CheckpointsTransformer` — checkpoint flush events

`CheckpointsTransformer` captures `checkpoint` protocol events and exposes them as `run.extensions["checkpoints"]`. Each event corresponds to one checkpoint write (one Pregel super-step completing).

```python
with graph.stream_events({"counter": 0}, version="v3") as run:
    # checkpoints is in run.extensions (not a native direct attribute)
    for checkpoint_event in run.extensions["checkpoints"]:
        # Each item is a CheckpointStreamPart with the saved checkpoint ID
        print(f"Checkpoint saved: {checkpoint_event}")
    print("Final:", run.output)
```

### Useful pattern: wait for a specific checkpoint

```python
TARGET_STEP = 3

async def wait_for_checkpoint(graph, input_state, config):
    async with graph.astream_events(input_state, config=config, version="v3") as run:
        step = 0
        async for cp in run.extensions["checkpoints"]:
            step += 1
            if step >= TARGET_STEP:
                print(f"Reached step {TARGET_STEP}, checkpoint id: {cp}")
                run.abort()
                break
```

---

### `TasksTransformer` — per-node task execution events

`TasksTransformer` exposes raw task-start and task-result events on `run.extensions["tasks"]`. These are the same events that drive `LifecycleTransformer` and `SubgraphTransformer` internally. Useful for fine-grained tracing.

```python
with graph.stream_events({"counter": 0}, version="v3") as run:
    for task_event in run.extensions["tasks"]:
        # task_event is a ProtocolEvent with method="tasks"
        data = task_event["params"]["data"]
        name = data.get("name", "?")
        if "result" in data:
            print(f"Task '{name}' completed: {data.get('result', [])[:1]}")
        else:
            print(f"Task '{name}' started with input keys: {list((data.get('input') or {}).keys())}")
    print("Done:", run.output)
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
| **12** | **RemoteGraph/RemoteException, PostgresSaver/ShallowPostgresSaver, AsyncPostgresSaver, PostgresStore/PoolConfig, AsyncPostgresStore, ANNIndexConfig/HNSWConfig/IVFFlatConfig/PostgresIndexConfig, GraphRunStream/SubgraphRunStream, ToolCallWithContext/ToolInvocationError, LifecyclePayload/LifecycleTransformer, MessagesTransformer/CheckpointsTransformer/TasksTransformer** |

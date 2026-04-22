---
title: "Checkpointers — API reference"
description: "Every checkpointer backend LangGraph ships — InMemorySaver, SqliteSaver, AsyncSqliteSaver, PostgresSaver, AsyncPostgresSaver, ShallowPostgresSaver — with a feature matrix, setup calls, connection strings, and gotchas."
framework: langgraph
language: python
sidebar:
  label: "Ref · Checkpointers"
  order: 31
---

# Checkpointers — API reference

Verified against **`langgraph-checkpoint==4.0.2`**, **`langgraph-checkpoint-sqlite==3.0.3`**, **`langgraph-checkpoint-postgres==3.0.5`** (source: `/tmp/langgraph-install/langgraph/checkpoint/`, `/tmp/langgraph-cpt/langgraph/checkpoint/{sqlite,postgres}/`).

A checkpointer is a `BaseCheckpointSaver` subclass. It persists the per-thread history of `Checkpoint`/`CheckpointTuple` objects so the graph can pause (`interrupt`), resume (`Command(resume=...)`), replay (`get_state_history`), time-travel, and keep short-term memory across invocations.

Pick the right backend:

| Backend | Import | Best for | Persists? | History? | Async | TTL | Pipeline |
|---|---|---|---|---|---|---|---|
| `InMemorySaver` | `langgraph.checkpoint.memory` | Unit tests, demos, single-process dev | No | Yes | Yes (same class) | No | — |
| `SqliteSaver` | `langgraph.checkpoint.sqlite` | Small single-process apps, CLI tools, on-disk scratch | Yes (file) | Yes | No | No | — |
| `AsyncSqliteSaver` | `langgraph.checkpoint.sqlite.aio` | Async single-process apps (uses `aiosqlite`) | Yes (file) | Yes | Yes | No | — |
| `PostgresSaver` | `langgraph.checkpoint.postgres` | Sync production deployments | Yes | Yes | No | No | Yes |
| `AsyncPostgresSaver` | `langgraph.checkpoint.postgres.aio` | Async production deployments | Yes | Yes | Yes | No | Yes |
| `ShallowPostgresSaver` / `AsyncShallowPostgresSaver` | `langgraph.checkpoint.postgres.shallow` | Latest-only row, no time travel | Yes | **No** | (both) | No | Yes |

> `ShallowPostgresSaver` is **deprecated since 2.0.20** and slated for removal in 3.0.0. Use `PostgresSaver` with `durability="exit"` on `invoke` / `stream` instead.

The SQLite and Postgres packages install separately:

```bash
pip install langgraph-checkpoint-sqlite
pip install langgraph-checkpoint-postgres
```

## Minimal runnable example

```python
import sqlite3
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


class S(TypedDict):
    count: int


def bump(state: S) -> dict:
    return {"count": state["count"] + 1}


builder = StateGraph(S).add_node("bump", bump)
builder.add_edge(START, "bump").add_edge("bump", END)

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "t-1"}}
print(graph.invoke({"count": 0}, cfg))   # {'count': 1}
print(graph.invoke({"count": 0}, cfg))   # {'count': 2} — state resumed from checkpoint
```

## `InMemorySaver`

```python
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver(*, serde=None, factory=defaultdict)
```

Stores checkpoints in a nested `defaultdict`. Lost at process exit. Implements both sync and async methods (`aget`, `aput`, etc.) — it's fine to use under `asyncio`.

No `from_conn_string`, no `setup()`. It is a context manager if you want explicit lifetime (`with InMemorySaver() as saver: ...`).

## `SqliteSaver` / `AsyncSqliteSaver`

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Direct: own the connection yourself.
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
saver = SqliteSaver(conn)                            # __init__(conn, *, serde=None)

# Managed: connection opened and closed for you.
with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    graph = builder.compile(checkpointer=saver)
    graph.invoke(...)
```

- `from_conn_string(conn_string)` is a **context manager** (it uses `@contextmanager`). You **must** use `with`; assigning the result to a variable and indexing into it will not work.
- `setup()` is called **automatically** on first use; you don't need to invoke it.
- The connection is opened with `check_same_thread=False`; internal locking makes it safe across threads.
- Use `":memory:"` as the conn string for an ephemeral in-process DB.

Async variant:

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
    graph = builder.compile(checkpointer=saver)
    await graph.ainvoke(..., cfg)
```

Backed by `aiosqlite`. Same `from_conn_string`-as-context-manager rule applies (async context manager in this case).

## `PostgresSaver` / `AsyncPostgresSaver`

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgres://user:pass@localhost:5432/db?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as saver:
    saver.setup()              # REQUIRED on first use — runs schema migrations
    graph = builder.compile(checkpointer=saver)
    graph.invoke(inputs, {"configurable": {"thread_id": "t-1"}})
```

- **You must call `setup()` explicitly the first time**, unlike `SqliteSaver`. It runs the embedded `MIGRATIONS` and creates tables `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`.
- `from_conn_string(conn_string, *, pipeline=False)` opens a single `psycopg.Connection` with `autocommit=True`, `prepare_threshold=0`, `row_factory=dict_row`. `pipeline=True` wraps it in a `psycopg` pipeline for fewer round-trips (single-connection only).
- Direct constructor: `PostgresSaver(conn, pipe=None, serde=None)`. `conn` may be a `psycopg.Connection` **or** a `psycopg_pool.ConnectionPool` (in which case `pipe` must be `None`).

Async:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI, pipeline=False) as saver:
    await saver.setup()
    graph = builder.compile(checkpointer=saver)
    await graph.ainvoke(..., cfg)
```

Uses `psycopg.AsyncConnection` and optionally `AsyncConnectionPool`.

### Connection pooling

For long-lived pools (web servers), construct directly with a pool:

```python
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

pool = ConnectionPool(
    DB_URI,
    max_size=20,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)
saver = PostgresSaver(pool)
saver.setup()
```

The `autocommit=True, prepare_threshold=0, row_factory=dict_row` combination is required — the saver depends on all three.

## What gets stored per thread

A `Checkpoint` (TypedDict, `langgraph.checkpoint.base`):

```python
{
    "v": 1,                       # format version
    "id": "01HY...",              # monotonically increasing ULID-ish
    "ts": "2026-04-22T12:34:56Z",
    "channel_values": {...},      # current value of every channel
    "channel_versions": {...},    # per-channel monotonic version
    "versions_seen": {...},       # per-node last seen versions
    "updated_channels": [...],    # channels changed in this step
}
```

`CheckpointMetadata` tags each checkpoint with:

- `source`: `"input" | "loop" | "update" | "fork"` — how the checkpoint was created.
- `step`: `-1` for the initial input, `0` for the first loop step, then `1, 2, ...`.
- `parents`: mapping of checkpoint namespace → parent checkpoint id (subgraphs).
- `run_id`: opaque run identifier, matches the LangSmith run.

## Required config keys

Every call that touches a checkpointer needs at least:

```python
{"configurable": {"thread_id": "some-unique-id"}}
```

Optionally also:

- `checkpoint_ns` — subgraph namespace (set automatically by parents).
- `checkpoint_id` — fetch/resume from a specific checkpoint (time travel).

Calling `graph.invoke(input, {"configurable": {}})` on a graph with a checkpointer raises `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id`.

## `BaseCheckpointSaver` methods you'll actually use

```python
saver.get_tuple(config) -> CheckpointTuple | None
saver.list(config, *, filter=None, before=None, limit=None) -> Iterator[CheckpointTuple]
saver.put(config, checkpoint, metadata, new_versions) -> RunnableConfig
saver.put_writes(config, writes, task_id, task_path="") -> None
saver.delete_thread(thread_id) -> None
# All have async twins: aget_tuple, alist, aput, aput_writes, adelete_thread
```

Typical app code uses the graph-level helpers instead:

```python
graph.get_state(cfg)                              # uses saver.get_tuple
list(graph.get_state_history(cfg))                # uses saver.list
graph.update_state(cfg, {"count": 42})            # uses saver.put + saver.put_writes
await graph.adelete_thread(thread_id)             # routes to the checkpointer
```

## Serializers

Default: `JsonPlusSerializer` from `langgraph.checkpoint.serde.jsonplus` — handles Pydantic models, dataclasses, `datetime`, `uuid`, `Decimal`, LangChain `BaseMessage`, and plain JSON.

For confidentiality at rest, wrap it with `EncryptedSerializer`:

```python
from cryptography.fernet import Fernet
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

encrypted = EncryptedSerializer.from_pycryptodome_aes(key=Fernet.generate_key())
saver = PostgresSaver(conn, serde=encrypted)
```

All savers accept `serde=...` in their constructor. `InMemorySaver` accepts it too, via kwarg only.

## Durability vs checkpointing

Checkpointing frequency is controlled per-call by `durability=` on `invoke` / `stream`:

| Value | Behavior |
|---|---|
| `"sync"` | Flush each step's checkpoint before the next runs. |
| `"async"` (default) | Write in the background while the next step runs. |
| `"exit"` | Only persist on graph exit. No time travel mid-run. |

The legacy `checkpoint_during=False` kwarg still works and maps to `durability="exit"`, with a `DeprecationWarning`.

## Patterns

### 1. Conversation memory (short-term)

```python
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "alice"}}
graph.invoke({"messages": [HumanMessage("Hi")]}, cfg)
graph.invoke({"messages": [HumanMessage("What did I say?")]}, cfg)
# Second call sees the full message history for thread 'alice'.
```

### 2. Time-travel / replay

```python
history = list(graph.get_state_history(cfg))
# history[0] is latest; history[-1] is the initial input.
earlier = history[3].config
graph.invoke(None, earlier)            # replays from that checkpoint
```

Passing `None` as input means "continue from the saved state" — the initial state is already there.

### 3. Fork a branch

```python
# Edit state at a past checkpoint, creating a new branch.
new_cfg = graph.update_state(earlier, {"plan": "take-a-different-route"})
graph.invoke(None, new_cfg)
```

`update_state` returns a config pointing at the new checkpoint; passing it to `invoke` continues from there.

### 4. Production Postgres with pooling

```python
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

pool = AsyncConnectionPool(
    DB_URI,
    max_size=32,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    open=False,
)
saver = AsyncPostgresSaver(pool)

@asynccontextmanager
async def lifespan(app):
    await pool.open()
    await saver.setup()
    yield
    await pool.close()
```

### 5. Per-user thread IDs

Namespace thread ids by user so a leak cannot cross accounts:

```python
cfg = {"configurable": {"thread_id": f"user:{user_id}:conv:{conv_id}"}}
graph.invoke({"messages": msgs}, cfg)
```

For cross-thread (long-term) memory, pair with a `Store` — see the [Store reference](./reference-store/).

## Gotchas

- **`from_conn_string` is a context manager**, not a factory. `saver = SqliteSaver.from_conn_string("x.db")` yields a context manager object, not a saver. Always use `with`.
- **Postgres needs `setup()` once.** Don't skip it on first deploy; the migration table is bootstrapped from this call.
- **`ShallowPostgresSaver`** only keeps the latest checkpoint. No `get_state_history`, no forking, no time travel. Deprecated — prefer `PostgresSaver` with `durability="exit"`.
- **`thread_id` is required.** A checkpointed graph called with an empty configurable dict raises `ValueError`.
- **`InMemorySaver` is not persistent.** Restarting the process loses all threads. Not suitable for Platform-hosted agents (the managed checkpointer is injected automatically there — don't pass one at all).
- **Don't share a raw `psycopg.Connection` across threads without the saver.** The saver holds a `threading.Lock`; bypassing it breaks `autocommit` contract.
- Deleting a thread is `delete_thread(thread_id)` — this is a checkpointer method, not a graph method on older versions. In v1.x, `graph.delete_thread` / `graph.adelete_thread` forward to the checkpointer.

## Breaking changes

| Version | Change |
|---|---|
| checkpoint 4.0 | `Checkpoint.v == 1` is the supported format; checkpoints with `v < 4` from the old pending-sends schema are auto-migrated on read. |
| checkpoint 3.x | `checkpoint_during` kwarg deprecated; migrate to `durability="sync" \| "async" \| "exit"`. |
| postgres 3.0 / shallow 2.0.20 | `ShallowPostgresSaver` / `AsyncShallowPostgresSaver` deprecated; prefer `PostgresSaver` with `durability="exit"`. |
| langgraph 0.4 | `Interrupt.id` introduced; `interrupt_id` deprecated. |
| langgraph 0.6 | `Interrupt.ns`, `when`, `resumable`, `interrupt_id` removed. |

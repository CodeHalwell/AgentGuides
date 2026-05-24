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

Verified against **`langgraph==1.2.1`**, **`langgraph-checkpoint==4.1.0`**, **`langgraph-checkpoint-sqlite==3.0.3`**, **`langgraph-checkpoint-postgres==3.0.5`** (modules: `langgraph.checkpoint.{base,memory,sqlite,postgres}`).

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

> `ShallowPostgresSaver` is **deprecated since 2.0.20** and will be removed in a future release (its own `DeprecationWarning` still names 3.0.0, but as of 3.0.5 the class is retained for compatibility). Use `PostgresSaver` with `durability="exit"` on `invoke` / `stream` instead.

The SQLite and Postgres packages install separately:

```bash
pip install langgraph-checkpoint-sqlite
pip install langgraph-checkpoint-postgres
```

## Minimal runnable example

```python
import operator
import sqlite3
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


class S(TypedDict):
    count: Annotated[int, operator.add]   # reducer: add new values to existing


def bump(state: S) -> dict:
    return {"count": 1}   # adds 1 to the persisted value via the reducer


builder = StateGraph(S).add_node("bump", bump)
builder.add_edge(START, "bump").add_edge("bump", END)

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "t-1"}}
print(graph.invoke({"count": 0}, cfg))   # {'count': 1}
print(graph.invoke({"count": 0}, cfg))   # {'count': 2} — accumulated from checkpoint
print(graph.invoke({"count": 0}, cfg))   # {'count': 3}
```

> Without the `operator.add` reducer, `"count"` uses default `LastValue`
> semantics and every call would reset it back to `0`. A reducer (or a
> `MessagesState`-style append-only channel) is what makes state *grow*
> across runs — the checkpointer only persists it.

## `InMemorySaver`

```python
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()
# Or, with a custom serializer:
# saver = InMemorySaver(serde=my_serde)
```

Full constructor: `InMemorySaver(*, serde=None, factory=defaultdict)`. `factory` swaps the underlying mapping type (e.g., a `PersistentDict` for on-disk simulation in tests); most callers leave it at the default.

Stores checkpoints in a nested `defaultdict`. Lost at process exit. Implements both sync and async methods (`aget`, `aput`, etc.) — it's fine to use under `asyncio`.

No `from_conn_string`, no `setup()`. It is a context manager if you want explicit lifetime (`with InMemorySaver() as saver: ...`).

> **Import note:** `InMemorySaver` must be imported from `langgraph.checkpoint.memory`. The name `MemorySaver` exists in the same module as a backward-compatible alias, but `InMemorySaver` is the primary name. Do not import from the top-level `langgraph.checkpoint` package.

## `Checkpointer` type alias — subgraph usage

When composing graphs, the `Checkpointer` type alias controls whether a subgraph participates in checkpointing:

```python
from langgraph.types import Checkpointer
# Checkpointer = None | bool | BaseCheckpointSaver
```

| Value | Effect on the subgraph |
|---|---|
| `None` (default) | Inherit the parent graph's checkpointer. |
| `True` | Enable persistent checkpointing for this subgraph, using the parent's backend. |
| `False` | Disable checkpointing for this subgraph even when the parent has one. |
| A `BaseCheckpointSaver` instance | Use this specific saver for the subgraph. |

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

# Subgraph that opts out of checkpointing
sub_builder = StateGraph(SubState)
sub_builder.add_node("step", my_step)
sub_builder.add_edge(START, "step").add_edge("step", END)
subgraph = sub_builder.compile(checkpointer=False)   # no checkpointing

# Parent graph uses a checkpointer; subgraph does not
parent_builder = StateGraph(ParentState)
parent_builder.add_node("sub", subgraph)
graph = parent_builder.compile(checkpointer=InMemorySaver())
```

The `ensure_valid_checkpointer()` utility validates a value before it reaches the graph compiler:

```python
from langgraph.types import ensure_valid_checkpointer

checkpointer = ensure_valid_checkpointer(my_checkpointer)  # raises if invalid type
```

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
- `writes`: mapping of node name → output written in this step.
- `parents`: mapping of checkpoint namespace → parent checkpoint id (subgraphs).

## `StateSnapshot` — fields

`graph.get_state(config)` returns a `StateSnapshot` namedtuple. All fields:

```python
from langgraph.checkpoint.base import StateSnapshot

snapshot: StateSnapshot = graph.get_state(config)

snapshot.values        # dict[str, Any]            — current channel values
snapshot.next          # tuple[str, ...]            — names of nodes queued to run next
snapshot.config        # RunnableConfig             — config that identifies this checkpoint
snapshot.metadata      # CheckpointMetadata | None  — source, step, writes, parents
snapshot.created_at    # str | None                 — ISO-8601 timestamp of this checkpoint
snapshot.parent_config # RunnableConfig | None      — config of the preceding checkpoint
snapshot.tasks         # tuple[PregelTask, ...]     — pending task descriptors
snapshot.interrupts    # tuple[Interrupt, ...]      — interrupts raised in the current step (new in 1.2+)
```

### `interrupts` field (new in 1.2+)

`snapshot.interrupts` exposes the `Interrupt` objects raised during the most recent step — useful for inspecting why a graph paused without re-running it:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

def approval_node(state):
    answer = interrupt("Approve this action?")   # pauses execution
    return {"approved": answer}

# After the graph pauses, inspect without re-running:
snapshot = graph.get_state(cfg)
for intr in snapshot.interrupts:
    print(intr.value)   # "Approve this action?"
```

### Iterating state history

`get_state_history()` returns an iterator of `StateSnapshot` objects, newest first:

```python
for snapshot in graph.get_state_history(config, limit=10):
    print(snapshot.config["configurable"]["checkpoint_id"])
    print(snapshot.created_at)
    print(snapshot.values)
    print(snapshot.metadata)
```

### Patching state manually

`update_state()` writes a new checkpoint as if the named node had produced the given output:

```python
graph.update_state(
    config,
    {"counter": 42},      # channel values to patch
    as_node="my_node",    # treat this update as if emitted by "my_node"
)
# Returns a RunnableConfig pointing at the newly created checkpoint.
```

## `Durability` setting

`Durability` controls *when* each checkpoint is flushed to the backend. Import the type or pass the literal string directly:

```python
from langgraph.types import Durability
# Durability = Literal["sync", "async", "exit"]
```

| Value | Behavior |
|---|---|
| `"sync"` | Checkpoint is persisted **synchronously** before the next step begins. Safest, slowest. |
| `"async"` (default) | Checkpoint is persisted **in the background** while the next step executes. Best throughput. |
| `"exit"` | Checkpoint is persisted **only when the graph exits**. No mid-run time-travel. Minimal I/O. |

Set it at compile time so every invocation of that graph uses the same policy:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

graph = builder.compile(
    checkpointer=InMemorySaver(),
    durability="async",       # background writes for better throughput
)
```

Or override per-call via `invoke` / `stream` (not yet supported in all backends — check release notes):

```python
graph.invoke(inputs, cfg, durability="sync")
```

The legacy `checkpoint_during=False` kwarg is still accepted and maps to `durability="exit"`, but emits a `DeprecationWarning`. Migrate to the explicit `durability=` spelling.

> Use `durability="exit"` as a lightweight replacement for the deprecated `ShallowPostgresSaver`: you keep a single-row footprint per thread while preserving `PostgresSaver`'s full API.

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

### Thread lifecycle management: `delete_thread` / `adelete_thread`

Every checkpoint, write, and blob for a thread is permanently removed when you call `delete_thread`. Use this for GDPR right-to-erasure flows, session cleanup, or bounded-memory test teardown.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class S(TypedDict):
    count: int


builder = StateGraph(S).add_node("bump", lambda s: {"count": s["count"] + 1})
builder.add_edge(START, "bump").add_edge("bump", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)

cfg = {"configurable": {"thread_id": "user-42"}}
graph.invoke({"count": 0}, cfg)
graph.invoke({"count": 0}, cfg)    # count == 2 after two calls

# Full GDPR erasure for user-42
saver.delete_thread("user-42")
# or via the graph:  graph.delete_thread("user-42")

# The thread no longer exists — next invoke starts fresh
graph.invoke({"count": 0}, cfg)    # count == 1 again
```

Async variant:

```python
await saver.adelete_thread("user-42")
# or:
await graph.adelete_thread("user-42")
```

`InMemorySaver.delete_thread` removes entries from all three internal dicts (`storage`, `writes`, `blobs`) in a single call. `PostgresSaver` / `AsyncPostgresSaver` issue `DELETE` statements targeting all three backing tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) for the given thread ID.

> **Deleting a thread is permanent.** There is no soft-delete or recycle bin — the data is gone immediately. Call `get_state_history` to archive thread content before deletion if you need an audit trail.

## Serializers

Default: `JsonPlusSerializer` from `langgraph.checkpoint.serde.jsonplus` — handles Pydantic models, dataclasses, `datetime`, `uuid`, `Decimal`, LangChain `BaseMessage`, and plain JSON.

For confidentiality at rest, wrap it with `EncryptedSerializer`:

```python
# pip install pycryptodome
import os
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.memory import InMemorySaver

# AES-128 key: must be exactly 16, 24, or 32 bytes
key = os.urandom(16)

encrypted_serde = EncryptedSerializer.from_pycryptodome_aes(key=key)
saver = InMemorySaver(serde=encrypted_serde)

# All data stored in the checkpointer is AES-EAX encrypted.
# PostgresSaver and SqliteSaver also accept serde=:
# saver = PostgresSaver(conn, serde=encrypted_serde)
```

Alternatively, set `LANGGRAPH_AES_KEY` in your environment (16, 24, or 32 character string) and omit the `key=` argument:

```bash
export LANGGRAPH_AES_KEY="your-16-char-key"
```

```python
# key is read from LANGGRAPH_AES_KEY automatically
encrypted_serde = EncryptedSerializer.from_pycryptodome_aes()
```

All savers accept `serde=...` in their constructor. `InMemorySaver` accepts it too, via kwarg only.

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

### 4. Time-travel with full snapshot inspection

Use `StateSnapshot` fields to pick a fork point programmatically:

```python
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)

cfg = {"configurable": {"thread_id": "replay-demo"}}

# Run the graph several times to build up history
for _ in range(5):
    graph.invoke({"count": 0}, cfg)

# Retrieve full history (newest first)
history = list(graph.get_state_history(cfg))

# Inspect each snapshot
for snap in history:
    print(snap.created_at, snap.values, snap.metadata["step"])

# Pick a specific past snapshot by index (index 2 = 3rd-most-recent)
old_snapshot = history[2]
print("Rewinding to step:", old_snapshot.metadata["step"])
print("State at that point:", old_snapshot.values)
print("Interrupts at that point:", old_snapshot.interrupts)  # new in 1.2+

# Fork: replay from that checkpoint with no new input
forked_config = old_snapshot.config
result = graph.invoke(None, forked_config)   # resumes from the saved state

# Or patch state before replaying (creates a new branch)
patched_config = graph.update_state(
    forked_config,
    {"count": 99},        # override a channel value
    as_node="bump",       # attribute the patch to the "bump" node
)
result = graph.invoke(None, patched_config)
```

### 5. Production Postgres with pooling

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

### 6. Per-user thread IDs

Namespace thread ids by user so a leak cannot cross accounts:

```python
cfg = {"configurable": {"thread_id": f"user:{user_id}:conv:{conv_id}"}}
graph.invoke({"messages": msgs}, cfg)
```

For cross-thread (long-term) memory, pair with a `Store` — see the [Store reference](./reference-store/).

### 7. Listing all threads for a user (audit / GDPR)

`saver.list(config=None)` iterates **all** checkpoints across all threads. Filter by metadata or walk the `storage` dict (for `InMemorySaver`) to enumerate threads for a given user prefix:

```python
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()

def list_user_threads(saver: InMemorySaver, user_prefix: str) -> list[str]:
    return [
        tid for tid in saver.storage
        if tid.startswith(user_prefix)
    ]

def purge_user(saver: InMemorySaver, user_id: str) -> int:
    threads = list_user_threads(saver, f"user:{user_id}:")
    for tid in threads:
        saver.delete_thread(tid)
    return len(threads)

# Example:
cfg = {"configurable": {"thread_id": "user:alice:conv:1"}}
graph.invoke({"messages": [HumanMessage("hi")]}, cfg)

purge_user(saver, "alice")   # removes all threads for alice
```

For `PostgresSaver`, query `SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'user:alice:%'` and then call `saver.delete_thread(tid)` for each row.

### 8. `filter=` on `list()` — metadata-scoped history

Use `filter` to restrict `list()` to checkpoints from a specific source or step range:

```python
# Only checkpoints written after a graph invocation (source='loop' or source='update')
checkpoints = list(
    saver.list(cfg, filter={"source": "loop"}, limit=5)
)

# Checkpoints at a specific step:
checkpoints = list(
    saver.list(cfg, filter={"step": 2})
)
```

`filter` is a dict of `CheckpointMetadata` key/value pairs. All pairs must match (AND semantics).

## Gotchas

- **`from_conn_string` is a context manager**, not a factory. `saver = SqliteSaver.from_conn_string("x.db")` yields a context manager object, not a saver. Always use `with`.
- **Postgres needs `setup()` once.** Don't skip it on first deploy; the migration table is bootstrapped from this call.
- **`ShallowPostgresSaver`** only keeps the latest checkpoint. No `get_state_history`, no forking, no time travel. Deprecated — prefer `PostgresSaver` with `durability="exit"`.
- **`thread_id` is required.** A checkpointed graph called with an empty configurable dict raises `ValueError`.
- **`InMemorySaver` is not persistent.** Restarting the process loses all threads. Not suitable for Platform-hosted agents (the managed checkpointer is injected automatically there — don't pass one at all).
- **Don't share a raw `psycopg.Connection` across threads without the saver.** The saver holds a `threading.Lock`; bypassing it breaks `autocommit` contract.
- Deleting a thread is `delete_thread(thread_id)` — this is a checkpointer method, not a graph method on older versions. In v1.x, `graph.delete_thread` / `graph.adelete_thread` forward to the checkpointer.
- **`InMemorySaver` import path.** Always import from `langgraph.checkpoint.memory`. `MemorySaver` is an alias in the same module but `InMemorySaver` is the canonical name.
- **`Checkpointer=False` on subgraphs.** Passing `checkpointer=False` when compiling a subgraph disables checkpointing for that subgraph even when the parent has one. This is intentional; use `None` (the default) to inherit the parent's backend.

## Breaking changes

| Version | Change |
|---|---|
| langgraph 1.2.1 | `StateSnapshot.interrupts` field added; `Checkpointer` type alias (`None \| bool \| BaseCheckpointSaver`) formalised in `langgraph.types`; `ensure_valid_checkpointer()` utility added. |
| checkpoint 4.0 | `Checkpoint.v == 1` is the supported format; checkpoints with `v < 4` from the old pending-sends schema are auto-migrated on read. |
| checkpoint 3.x | `checkpoint_during` kwarg deprecated; migrate to `durability="sync" \| "async" \| "exit"`. |
| postgres 3.0 / shallow 2.0.20 | `ShallowPostgresSaver` / `AsyncShallowPostgresSaver` deprecated; prefer `PostgresSaver` with `durability="exit"`. |
| langgraph 0.4 | `Interrupt.id` introduced; `interrupt_id` deprecated. |
| langgraph 0.6 | `Interrupt.ns`, `when`, `resumable`, `interrupt_id` removed. |

---
title: "Checkpointers — API reference"
description: "Every checkpointer backend LangGraph ships — InMemorySaver, SqliteSaver, AsyncSqliteSaver, PostgresSaver, AsyncPostgresSaver, ShallowPostgresSaver — with a feature matrix, setup calls, connection strings, and gotchas."
framework: "langgraph"
language: "python"
sidebar:
  label: "Ref · Checkpointers"
  order: 31
---

Verified against **`langgraph==%%version=langgraph%%`**, **`langgraph-checkpoint==%%version=langgraph-checkpoint%%`**, **`langgraph-checkpoint-sqlite==%%version=langgraph-checkpoint-sqlite%%`**, **`langgraph-checkpoint-postgres==%%version=langgraph-checkpoint-postgres%%`** (modules: `langgraph.checkpoint.{base,memory,sqlite,postgres}`).

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

> `ShallowPostgresSaver` is **deprecated since 2.0.20**. Its `DeprecationWarning` says it will be removed in 3.0.0, but it is still shipped in `langgraph-checkpoint-postgres` %%version=langgraph-checkpoint-postgres%%. Use `PostgresSaver` instead, with `durability="exit"` to cut writes and `prune(...)` to drop old checkpoints (see [Keeping history small](#keeping-history-small-prune)).

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
    "v": 4,                       # format version written by langgraph %%version=langgraph%%
    "id": "1f1d001b-dc60-...",    # time-ordered UUID string; sorts by creation
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
- `run_id`: the run that wrote the checkpoint, when the config carried one.

There is no `writes` key in %%version=langgraph-checkpoint%%. A node's output is in `tasks[i].result` of the snapshot that scheduled it (the one whose `next` names the node), or stream it with `stream_mode="updates"`.

## `StateSnapshot` — fields

`graph.get_state(config)` returns a `StateSnapshot` namedtuple. All fields:

```python
from langgraph.types import StateSnapshot

snapshot: StateSnapshot = graph.get_state(config)

snapshot.values        # dict[str, Any]            — current channel values
snapshot.next          # tuple[str, ...]            — names of nodes queued to run next
snapshot.config        # RunnableConfig             — config that identifies this checkpoint
snapshot.metadata      # CheckpointMetadata | None  — source, step, writes, parents
snapshot.created_at    # str | None                 — ISO-8601 timestamp of this checkpoint
snapshot.parent_config # RunnableConfig | None      — config of the preceding checkpoint
snapshot.tasks         # tuple[PregelTask, ...]     — pending task descriptors
snapshot.interrupts    # tuple[Interrupt, ...]      — pending interrupts from the current step
```

### `interrupts` field

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

---

## `Interrupt` — struct reference

Checked against the langgraph %%version=langgraph%% source. `Interrupt` is the dataclass emitted by calls to `langgraph.types.interrupt()`. It is surfaced on both `StateSnapshot.interrupts` and `GraphOutput.interrupts` (when using `version="v2"`).

```python
# langgraph.types
@final
@dataclass(init=False, slots=True)
class Interrupt(Generic[ResponseT]):
    value: Any
    """The value passed to interrupt() — the question or payload for the human reviewer."""

    id: str
    """Unique ID for this interrupt. Used to resume a specific interrupt:
    Command(resume={intr.id: answer})"""

    response_schema: type[ResponseT] | dict[str, Any] | None = None
    """JSON Schema for the expected resume value, if interrupt() was given one."""
```

See [typed resume values](/langgraph-guide/python/reference-command-and-send/#typed-resume-values-response_schema) for how `response_schema` validates resume values.

**Key usage patterns:**

```python
from typing import Annotated
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    approved: bool | None


def approval_gate(state: State) -> dict:
    """Pause until a human approves the action."""
    answer = interrupt({
        "question": "Should I proceed with this action?",
        "context": state["messages"][-1].content,
    })
    return {"approved": answer}


graph = (
    StateGraph(State)
    .add_node("gate", approval_gate)
    .add_edge(START, "gate")
    .add_edge("gate", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "review-1"}}

# --- First call: graph pauses at the interrupt ---
graph.invoke({"messages": [HumanMessage("Do the thing")], "approved": None}, cfg)

# --- Inspect the pending interrupt ---
snapshot = graph.get_state(cfg)
assert snapshot.interrupts, "Expected at least one interrupt"
intr = snapshot.interrupts[0]  # select the first (and only) interrupt explicitly
print(f"Interrupt id: {intr.id}")
print(f"Interrupt value: {intr.value}")
# {'question': 'Should I proceed...', 'context': 'Do the thing'}

# --- Resume by addressing the interrupt by ID ---
result = graph.invoke(
    Command(resume={intr.id: True}),  # True = approved
    cfg,
)
print(result["approved"])  # True
```

**`Interrupt` field reference:**

| Field | Type | Description |
|---|---|---|
| `value` | `Any` | The payload passed to `interrupt(value)` — question, instructions, or data for the human |
| `id` | `str` | Identifies the task that raised it. Use in `Command(resume={id: answer})` to address it; several `interrupt()` calls in one node share one id |
| `response_schema` | `dict \| None` | JSON Schema for the expected resume value, if the node passed `response_schema=` |

> **Note on removed fields.** Before v0.6.0, `Interrupt` had `ns`, `when` and `resumable` attributes and an `interrupt_id` property (added in 0.4). v0.6.0 removed the attributes and replaced `interrupt_id` with `id`; `interrupt_id` still works as a deprecated property. If you see code referencing the others, it targets an older LangGraph version.

---

## `PregelTask` — task introspection

Checked against the langgraph %%version=langgraph%% source. `PregelTask` is a `NamedTuple` that appears inside `StateSnapshot.tasks`. Each entry represents one pending or completed task for the current superstep.

```python
# langgraph.types
from typing import Any, NamedTuple
from langchain_core.runnables import RunnableConfig
from langgraph.types import Interrupt, StateSnapshot

class PregelTask(NamedTuple):
    id: str                               # unique task identifier
    name: str                             # node name (e.g. "call_model", "tools")
    path: tuple[str | int | tuple, ...]   # routing path within the graph
    error: Exception | None               # if the task failed, the exception; else None
    interrupts: tuple[Interrupt, ...]     # interrupts raised by this specific task
    state: None | RunnableConfig | StateSnapshot  # subgraph state (if applicable)
    result: Any | None                    # task result after completion
```

**Usage — inspecting pending tasks before resuming:**

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from typing import TypedDict

class State(TypedDict):
    value: int


def node_a(state: State) -> dict:
    interrupt("step A paused")
    return {"value": state["value"] + 1}


def node_b(state: State) -> dict:
    return {"value": state["value"] * 2}


graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "task-demo"}}
graph.invoke({"value": 1}, cfg)   # pauses at interrupt in node_a

snapshot = graph.get_state(cfg)

# snapshot.next tells you which nodes are queued
print(snapshot.next)   # ('a',) — node_a is waiting to resume

# snapshot.tasks gives richer task-level info
for task in snapshot.tasks:
    print(f"Task id:         {task.id}")
    print(f"Node name:       {task.name}")
    print(f"Path:            {task.path}")
    print(f"Error:           {task.error}")       # None — paused, not failed
    print(f"Interrupts:      {task.interrupts}")  # (Interrupt(value='step A paused', id=...),)
    print(f"Result so far:   {task.result}")      # None — not yet complete
```

**Usage — inspecting failed tasks:**

```python
from langgraph.types import Command

def flaky_node(state: State) -> dict:
    if state["value"] < 0:
        raise ValueError("negative value not allowed")
    return {"value": state["value"]}


graph = (
    StateGraph(State)
    .add_node("flaky", flaky_node)
    .add_edge(START, "flaky")
    .add_edge("flaky", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "error-demo"}}
try:
    graph.invoke({"value": -1}, cfg)
except Exception:
    pass

snapshot = graph.get_state(cfg)
for task in snapshot.tasks:
    if task.error:
        print(f"Task '{task.name}' failed: {type(task.error).__name__}: {task.error}")
        # Task 'flaky' failed: ValueError: negative value not allowed
```

**`PregelTask` field reference:**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique task ID within the current superstep |
| `name` | `str` | Name of the node this task belongs to |
| `path` | `tuple` | Internal routing path — useful for subgraph introspection |
| `error` | `Exception \| None` | Exception from the task if it failed; `None` if pending or succeeded |
| `interrupts` | `tuple[Interrupt, ...]` | Interrupts raised by this task specifically |
| `state` | `None \| RunnableConfig \| StateSnapshot` | Subgraph checkpoint state when `subgraphs=True` |
| `result` | `Any \| None` | The task's output once it completes |

---

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

Writes go through the channel reducers, so on an `Annotated[int, operator.add]` field `update_state(cfg, {"total": 100})` adds 100 rather than replacing the value (wrap it in `Overwrite(...)` to replace).

`update_state()` is a one-step wrapper around `bulk_update_state(config, supersteps)`, which applies several supersteps in one call. Each superstep is a sequence of `StateUpdate(values, as_node=None, task_id=None)` named tuples (from `langgraph.types`), applied in order, each one creating a checkpoint:

```python
from langgraph.types import StateUpdate

graph.bulk_update_state(
    config,
    [
        [StateUpdate({"items": ["corrected-item"], "phase": "revised"}, as_node="process")],
        [StateUpdate({"phase": "final"}, as_node="process")],
    ],
)
graph.get_state(config).values["phase"]   # 'final'
```

Use it to replay a corrected sequence of writes after manual intervention without re-running nodes. The async twins are `aupdate_state` / `abulk_update_state`.

## `Durability` setting

`Durability` controls *when* each checkpoint is flushed to the backend. Import the type or pass the literal string directly:

```python
from langgraph.types import Durability
# Durability = Literal["sync", "async", "exit"]
```

| Value | Behaviour |
|---|---|
| `"sync"` | Checkpoint is persisted **synchronously** before the next step begins. Safest, slowest. |
| `"async"` (default) | Checkpoint is persisted **in the background** while the next step executes. Best throughput. |
| `"exit"` | Checkpoint is persisted **only when the graph exits**. No mid-run time-travel. Minimal I/O. |

`durability` is a **per-call** option on `invoke` / `stream` / `ainvoke` / `astream` — it is **not** a `compile()` argument:

```python
# Set per-run when calling invoke or stream:
graph.invoke(inputs, cfg, durability="async")   # background writes, better throughput
graph.stream(inputs, cfg, durability="sync")    # wait for write before next step
graph.invoke(inputs, cfg, durability="exit")    # write only when graph exits
```

The legacy `checkpoint_during=False` kwarg is still accepted and maps to `durability="exit"`, but emits a `DeprecationWarning`. Migrate to the explicit `durability=` spelling.

> `durability="exit"` writes one checkpoint per run instead of one per step, so a thread's history grows by one entry per call rather than staying at a single row as with `ShallowPostgresSaver`. To keep only the latest checkpoint, call `PostgresSaver.prune(thread_ids, strategy="keep_latest")` periodically.

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

```text
saver.get_tuple(config) -> CheckpointTuple | None
saver.list(config, *, filter=None, before=None, limit=None) -> Iterator[CheckpointTuple]
saver.put(config, checkpoint, metadata, new_versions) -> RunnableConfig
saver.put_writes(config, writes, task_id, task_path="") -> None
saver.delete_thread(thread_id) -> None
saver.prune(thread_ids, *, strategy="keep_latest") -> None    # PostgresSaver only
saver.copy_thread(source_thread_id, target_thread_id) -> None  # NotImplementedError in the OSS savers
saver.delete_for_runs(run_ids) -> None                          # NotImplementedError in the OSS savers
# All have async twins: aget_tuple, alist, aput, aput_writes, adelete_thread, aprune, ...
```

Typical app code uses the graph-level helpers instead:

```python
graph.get_state(cfg)                              # uses saver.get_tuple
list(graph.get_state_history(cfg))                # uses saver.list
graph.update_state(cfg, {"count": 42})            # uses saver.put + saver.put_writes
saver.delete_thread(thread_id)                    # no graph-level equivalent: call the saver
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

# Full GDPR erasure for user-42 (compiled graphs have no delete_thread method)
saver.delete_thread("user-42")

# The thread no longer exists — next invoke starts fresh
graph.invoke({"count": 0}, cfg)    # count == 1 again
```

Async variant:

```python
await saver.adelete_thread("user-42")
```

`InMemorySaver.delete_thread` removes entries from all three internal dicts (`storage`, `writes`, `blobs`) in a single call. `PostgresSaver` / `AsyncPostgresSaver` issue `DELETE` statements targeting all three backing tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) for the given thread ID.

### Keeping history small: `prune`

`PostgresSaver` and `AsyncPostgresSaver` can trim old checkpoints. `prune(thread_ids, strategy="keep_latest")` keeps the latest checkpoint of each namespace (the one `get_state` loads) and deletes the rest; `strategy="delete"` removes the threads entirely. `InMemorySaver` and the SQLite savers raise `NotImplementedError`. Combined with `durability="exit"`, this replaces `ShallowPostgresSaver`:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/app"


class S(TypedDict):
    count: Annotated[int, operator.add]


builder = StateGraph(S).add_node("bump", lambda s: {"count": 1})
builder.add_edge(START, "bump").add_edge("bump", END)

with PostgresSaver.from_conn_string(DB_URI) as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "prune-demo"}}

    for _ in range(3):
        graph.invoke({"count": 0}, cfg)
    print(len(list(saver.list(cfg))))      # 9: three checkpoints per run (input, start, bump)

    for _ in range(3):
        graph.invoke({"count": 0}, cfg, durability="exit")
    print(len(list(saver.list(cfg))))      # 12: one more per run

    saver.prune(["prune-demo"], strategy="keep_latest")
    print(len(list(saver.list(cfg))))      # 1
    print(graph.get_state(cfg).values)     # {'count': 6}: the state is intact
```

`keep_latest` refuses threads that use a `DeltaChannel` (it would need the older checkpoints to rebuild the value), and forking from a checkpoint while it is being pruned is not safe. Run it from a maintenance job rather than inside a request.

> **Deleting a thread is permanent.** There is no soft-delete or recycle bin — the data is gone immediately. Call `get_state_history` to archive thread content before deletion if you need an audit trail.

## Serialisers

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

`EncryptedSerializer` appends the cipher name to the inner serialiser's type tag (`"msgpack+aes"`), and blobs whose type has no `+` suffix are passed straight to the inner `serde`. So you can turn encryption on for an existing deployment and still read the unencrypted checkpoints written before. AES-EAX is authenticated: tampered ciphertext fails to decrypt with a `ValueError`.

For KMS-backed keys or another cipher, implement `CipherProtocol` (from `langgraph.checkpoint.serde.base`) and pass it to the constructor:

```python
from langgraph.checkpoint.serde.base import CipherProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer


class XorCipher(CipherProtocol):
    """Toy cipher that shows the interface. Not for production."""

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "xor", bytes(b ^ 0x42 for b in plaintext)   # (cipher name, ciphertext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        assert ciphername == "xor"
        return bytes(b ^ 0x42 for b in ciphertext)


serde = EncryptedSerializer(cipher=XorCipher())   # serde= defaults to JsonPlusSerializer()
typ, data = serde.dumps_typed({"hello": "world"})
print(typ)                              # 'msgpack+xor'
print(serde.loads_typed((typ, data)))   # {'hello': 'world'}
```

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

Use `StateSnapshot` fields to pick a fork point programmatically. This example is self-contained:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


class Counter(TypedDict):
    count: Annotated[int, operator.add]


counter_builder = StateGraph(Counter).add_node("bump", lambda s: {"count": 1})
counter_builder.add_edge(START, "bump").add_edge("bump", END)
graph = counter_builder.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "replay-demo"}}
for _ in range(3):
    graph.invoke({"count": 0}, cfg)          # count: 1, 2, 3

history = list(graph.get_state_history(cfg))   # newest first
for snap in history:
    print(snap.metadata["step"], snap.metadata["source"], snap.values, snap.next)

# Pick the checkpoint just before the second run's "bump" executed.
before_bump = [s for s in history if s.next == ("bump",)][1]
print("State at that point:", before_bump.values)    # {'count': 1}

# Replay from there with no new input: bump runs again on that saved state.
print(graph.invoke(None, before_bump.config))         # {'count': 2}

# Or patch state first; this creates a new branch from the same point.
patched = graph.update_state(before_bump.config, {"count": 99}, as_node="__start__")
print(graph.invoke(None, patched))                    # {'count': 101}
```

The replay and the patched run both become new branches of the thread; the latest checkpoint is now the last one written, so `get_state(cfg)` returns `{'count': 101}`.

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

For cross-thread (long-term) memory, pair with a `Store` — see the [Store reference](/langgraph-guide/python/reference-store/).

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
- Deleting a thread is `saver.delete_thread(thread_id)` (or `adelete_thread`). It is a checkpointer method only; compiled graphs have no `delete_thread` in %%version=langgraph%%, so keep a reference to the saver.
- **`InMemorySaver` import path.** Always import from `langgraph.checkpoint.memory`. `MemorySaver` is an alias in the same module but `InMemorySaver` is the canonical name.
- **`Checkpointer=False` on subgraphs.** Passing `checkpointer=False` when compiling a subgraph disables checkpointing for that subgraph even when the parent has one. This is intentional; use `None` (the default) to inherit the parent's backend.

## Breaking changes

| Version | Change |
|---|---|
| langgraph 1.0 | `interrupt_id` warns with `LangGraphDeprecatedSinceV10`. |
| langgraph 0.6 | `durability=` replaces `checkpoint_during=` (still accepted, with a deprecation warning). `Interrupt` reduced to `value` and `id`: `ns`, `when` and `resumable` removed, `interrupt_id` deprecated in favour of `id`. |
| postgres 2.0.20 | `ShallowPostgresSaver` / `AsyncShallowPostgresSaver` deprecated; prefer `PostgresSaver`. |
| langgraph 0.4 | `Interrupt.interrupt_id` property added. |

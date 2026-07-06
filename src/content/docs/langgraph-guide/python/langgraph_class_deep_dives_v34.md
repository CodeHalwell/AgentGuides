---
title: "LangGraph Class Deep-Dives Vol. 34"
description: "Source-verified deep dives into 10 class groups in langgraph==1.2.7 — channels_from_checkpoint/achannels_from_checkpoint/_needs_replay, exit_delta_task_id/delta_channels_to_snapshot, prepare_push_task_functional/prepare_push_task_send/sanitize_untracked_values_in_send, get_new_channel_versions/find_subgraph_pregel, ProtocolEvent/_ProtocolEventParams anatomy, StreamTransformer class-variable flags/transformer_requires_async, patch_checkpoint_map/_merge_metadata, _TasksLifecycleBase template-method hooks, PregelRunner.commit()/_should_route_to_error_handler/_should_stop_others, and draw_graph static topology inference."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 34"
  order: 65
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.7` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `channels_from_checkpoint` + `achannels_from_checkpoint` + `_needs_replay`

**Module:** `langgraph.pregel._checkpoint`

These functions hydrate live `BaseChannel` instances from a persisted checkpoint dict. They are called at the start of every Pregel superstep to reconstruct in-memory channel state.

**Key source facts** (from `langgraph/pregel/_checkpoint.py`):

- `channels_from_checkpoint(specs, checkpoint, *, saver, config)` iterates over every key in `specs`:
  - Regular channels call `spec.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))`.
  - `DeltaChannel` entries that are absent from `channel_values` (i.e., `_needs_replay` returns `True`) are batched and passed to `saver.get_delta_channel_history(config=config, channels=delta_channels)`.
  - History results arrive as `{"seed": <value or MISSING>, "writes": [...]}`; the channel is seeded with `from_checkpoint(seed)` then `replay_writes(writes)` is called to replay the accumulated writes.
- `_needs_replay(spec, stored)` returns `True` only when `spec` is a `DeltaChannel` **and** `stored is MISSING`. `_DeltaSnapshot` blobs and plain values both resolve directly — only absence triggers the ancestor walk.
- `achannels_from_checkpoint` is the async twin; it uses `await saver.aget_delta_channel_history(...)` instead.
- `ManagedValueSpec` entries are separated into a `managed_specs` dict and returned as-is (they don't live in checkpoint blobs).

### Example 1 — inspect which channels would replay on load

```python
from langgraph.channels.last_value import LastValue
from langgraph.channels.delta import DeltaChannel
from langgraph.pregel._checkpoint import _needs_replay, LATEST_VERSION
from langgraph._internal._typing import MISSING

# Simulate a checkpoint that is missing a delta channel's value
checkpoint_values = {"x": 42}          # only x is stored; delta_log is absent

last_val_spec = LastValue(int)
delta_spec = DeltaChannel(list)

# _needs_replay: False for regular channels regardless of stored value
print(_needs_replay(last_val_spec, checkpoint_values.get("x", MISSING)))  # False
print(_needs_replay(last_val_spec, MISSING))                               # False

# _needs_replay: False when _DeltaSnapshot blob is present (not MISSING)
from langgraph.checkpoint.serde.types import _DeltaSnapshot
snap = _DeltaSnapshot([1, 2, 3])
print(_needs_replay(delta_spec, snap))   # False — seed blob resolves directly

# _needs_replay: True only when DeltaChannel value is absent
print(_needs_replay(delta_spec, MISSING))  # True — ancestor walk required
```

### Example 2 — channels_from_checkpoint without a saver — delta channels seed to typ()

```python
from langgraph.pregel._checkpoint import channels_from_checkpoint, empty_checkpoint
from langgraph.channels.last_value import LastValue
from langgraph.channels.delta import DeltaChannel

# Build a simple two-channel spec
specs = {
    "counter": LastValue(int),
    "log":     DeltaChannel(list),
}

# A checkpoint that has counter but not log (simulates a delta channel
# that snapshots only every N supersteps)
chk = empty_checkpoint()
chk["channel_values"]["counter"] = 7

# No saver → _needs_replay returns True for log but no history is fetched.
# DeltaChannel.from_checkpoint(MISSING) seeds to typ() = list(), making the
# channel available with an empty accumulator.
channels, managed = channels_from_checkpoint(specs, chk)
print(channels["counter"].get())   # 7
# log channel seeds to list() — it is available but empty until writes replay
print(channels["log"].is_available())  # True — from_checkpoint(MISSING) → typ() = []
```

### Example 3 — understand the DeltaChannel ancestor-walk batch path

```python
# The batch optimization: ALL delta channels needing replay are gathered
# into a single saver.get_delta_channel_history() call, not one call each.
# This avoids N round-trips to Postgres when multiple delta channels lag.

# Pseudocode showing the batching invariant:
#
#   delta_channels = [k for k, spec in channel_specs.items()
#                     if _needs_replay(spec, checkpoint["channel_values"].get(k, MISSING))]
#   histories = {}
#   if delta_channels and saver and config:
#       histories = saver.get_delta_channel_history(config=config,
#                                                   channels=delta_channels)
#   # All deltas resolved in one call; regular channels use from_checkpoint directly.

# In practice: if you see high checkpoint load, check whether many
# DeltaChannels are missing their snapshots. Reduce
# LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT (default 5000) to force
# more frequent snapshot blobs and shorten ancestor walks.
import os
os.environ["LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT"] = "100"
# Or lower snapshot_frequency on the DeltaChannel definition.
print("Ancestor-walk batch path active for any absent DeltaChannel values")
```

---

## 2 · `exit_delta_task_id` + advanced `delta_channels_to_snapshot`

**Module:** `langgraph.pregel._checkpoint`

These two functions handle the most subtle edge in DeltaChannel snapshotting: deciding *when* to take a snapshot and generating a synthetic UUID that survives `ORDER BY` in SQL checkpointers.

**Key source facts:**

- `exit_delta_task_id(step, task_id)` creates a synthetic RFC-4122 UUID by embedding the superstep number (`step`) zero-padded to **8 decimal digits** in the first UUID group (`{step:08d}-{parts[1]}-{parts[2]}-{parts[3]}-{parts[4]}`). This ensures `ORDER BY task_id, idx` preserves **chronological order** across supersteps while the string remains a valid UUID (required by Postgres `checkpoint_writes.task_id uuid` column).
- `delta_channels_to_snapshot(channels, counters_since_delta_snapshot)` returns a set of DeltaChannel names that should snapshot now. A channel snapshots when **either**:
  - Its accumulated update count since last snapshot ≥ `snapshot_frequency` (per-channel threshold), **or**
  - Total supersteps since its last snapshot ≥ `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` (env-var global cap, default 5000).
- Both conditions are **pure predicates** — no mutation happens here. The actual snapshot blob (`_DeltaSnapshot`) is written in `create_checkpoint`.
- `counters_since_delta_snapshot` maps channel name → `(update_count, superstep_count)` tuple; this is tracked by the Pregel loop.

### Example 1 — inspect the synthetic UUID structure

```python
from langgraph.pregel._checkpoint import exit_delta_task_id
import uuid

# A real task_id (UUID v6)
real_task_id = str(uuid.uuid4())   # e.g. "550e8400-e29b-41d4-a716-446655440000"

for step in [0, 1, 99, 255]:
    synthetic = exit_delta_task_id(step, real_task_id)
    print(f"step={step:3d} → {synthetic}")

# step=  0 → 00000000-e29b-41d4-a716-446655440000
# step=  1 → 00000001-e29b-41d4-a716-446655440000
# step= 99 → 00000099-e29b-41d4-a716-446655440000
# step=255 → 00000255-e29b-41d4-a716-446655440000
#
# The first group (8 zero-padded decimal digits) encodes the step number.
# ORDER BY task_id in SQL now gives chronological order across supersteps.
```

### Example 2 — understand the two snapshot triggers

```python
import operator
from langgraph.pregel._checkpoint import delta_channels_to_snapshot
from langgraph.channels.delta import DeltaChannel
from langgraph._internal._config import DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT

# Default global cap
print(f"DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT = {DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT}")

# DeltaChannel(reducer, typ=...) — first arg is the reducer function, not the type.
# operator.add concatenates list values; typ=list gives an empty-list seed on MISSING.
ch = DeltaChannel(operator.add, typ=list, snapshot_frequency=10)
ch.update([1, 2, 3])  # seeds value to [] + [1,2,3] → [1,2,3]; is_available() → True

channels = {"events": ch}

# (update_count, superstep_count) tuples
scenarios = [
    (5, 10),    # below both thresholds — no snapshot
    (10, 5),    # update_count == snapshot_frequency — SNAPSHOT
    (3, 5001),  # superstep_count > DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT — SNAPSHOT
]
for updates, supersteps in scenarios:
    result = delta_channels_to_snapshot(channels, {"events": (updates, supersteps)})
    triggered = "events" in result
    print(f"updates={updates}, supersteps={supersteps} → snapshot={triggered}")
```

### Example 3 — tuning with the environment variable

```python
# LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT controls the time-based
# snapshot trigger independently of per-channel snapshot_frequency.
#
# Lower the cap when your graphs run many short supersteps and you want
# compact DeltaChannel history (shorter ancestor walks = faster resume).
#
# Raise it (or set to a very high value) when your graphs use large
# DeltaChannels and you want to minimize checkpoint blob size.

import os

# Example: force a snapshot every 50 supersteps max (for high-frequency graphs)
os.environ["LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT"] = "50"

# Reload the module constant after the env change
import importlib
import langgraph._internal._config as _cfg
importlib.reload(_cfg)

print(f"After env change: {_cfg.DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT}")  # 50
```

---

## 3 · `prepare_push_task_functional` + `prepare_push_task_send` + `sanitize_untracked_values_in_send`

**Module:** `langgraph.pregel._algo`

These three functions handle scheduling of *out-of-band* tasks: `@task` call-graph tasks spawned by the functional API, `Send()` fan-out packets, and a safety filter for `UntrackedValue` channels in `Send.arg`.

**Key source facts:**

- `prepare_push_task_functional(task_path, ...)` handles a `PUSH` write whose last element is a `Call` object (from `@task`). It:
  - Derives `checkpoint_ns` as `"{parent_ns}{NS_SEP}{name}"` and a deterministic `task_id` using `task_id_func`.
  - Appends `True` to the task path (`in_progress_task_path = (*task_path[:3], True)`) to signal that a call is in progress and **interrupts from this task belong to its parent**.
  - Inherits `cache_policy` and `retry_policy` from the `Call` object, falling back to the graph defaults.
  - Returns a `PregelExecutableTask` (for execution) or a lightweight `PregelTask` stub (for graph traversal / static analysis).
- `prepare_push_task_send(task_path, ...)` handles `PUSH` writes from `Send()`. The `task_path` has shape `(PUSH, idx_of_send)`. It looks up `channels[TASKS].get()` to find the `Send` packet at the given index, finds the target `proc` in `processes`, and builds the `PregelExecutableTask`.
  - Appends `False` to the task path: `(*task_path[:3], False)` — `Send`-spawned tasks **can** surface their own interrupts.
- `sanitize_untracked_values_in_send(packet, channels)` strips top-level keys whose corresponding channels are `UntrackedValue` instances from `Send.arg` before checkpointing. This prevents checkpoint bloat from ephemeral values that should never persist.

### Example 1 — Send() fan-out and task path anatomy

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    items: Annotated[list, lambda a, b: a + b]  # reducer merges parallel writes

def router(state):
    # Fan out to multiple parallel worker tasks via Send
    return [Send("worker", {"item": x}) for x in state["items"]]

def worker(state):
    return {"items": [state["item"] * 2]}

builder = StateGraph(State)  # use typed state so the Annotated reducer is honoured
builder.add_node("worker", worker)
builder.add_conditional_edges(START, router)
builder.add_edge("worker", END)

graph = builder.compile()
result = graph.invoke({"items": [1, 2, 3]})
print(result)  # {"items": [2, 4, 6]} — workers run in parallel; reducer merges lists
# Each Send() becomes a PUSH task with task_path (PUSH, idx_in_TASKS_channel)
# prepare_push_task_send resolves the idx to the Send packet.
```

### Example 2 — @task call path with inherited retry policy

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

checkpointer = InMemorySaver()

# The RetryPolicy on the @task is passed through prepare_push_task_functional
# as call.retry_policy, overriding any graph-level retry_policy.
@task(retry_policy=RetryPolicy(max_attempts=3))
def fetch_data(url: str) -> str:
    # Simulating a retriable network call
    return f"data from {url}"

@entrypoint(checkpointer=checkpointer)
def pipeline(urls: list[str]) -> list[str]:
    futures = [fetch_data(u) for u in urls]
    return [f.result() for f in futures]

config = {"configurable": {"thread_id": "t1"}}
result = pipeline.invoke(["http://a.com", "http://b.com"], config=config)
print(result)  # ["data from http://a.com", "data from http://b.com"]
# Each @task() call becomes a prepare_push_task_functional entry with
# in_progress_task_path = (..., True) — interrupts bubble to the entrypoint.
```

### Example 3 — sanitize_untracked_values_in_send in action

```python
# sanitize_untracked_values_in_send is an internal helper in langgraph.pregel._algo
# that strips UntrackedValue channel keys from Send.arg before checkpointing.
# The equivalent logic using the public channel types:

from langgraph.types import Send
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.channels.last_value import LastValue

# Channel map where "scratch" is ephemeral — must NOT survive in a checkpoint
channels = {
    "topic": LastValue(str),
    "scratch": UntrackedValue(dict),
}

raw_arg = {"topic": "hello", "scratch": {"temp": 42}}

# Equivalent of the internal sanitizer: strip any key whose channel is UntrackedValue
safe_arg = {k: v for k, v in raw_arg.items()
            if not isinstance(channels.get(k), UntrackedValue)}

safe_send = Send(node="writer", arg=safe_arg)
print(safe_send.arg)   # {"topic": "hello"} — "scratch" stripped before checkpointing

# UntrackedValue channels are never written to checkpoint blobs; their presence
# in Send.arg would create stale values that cannot be replayed safely.
print(isinstance(channels["scratch"], UntrackedValue))  # True
```

---

## 4 · `get_new_channel_versions` + `find_subgraph_pregel`

**Module:** `langgraph.pregel._utils`

Two utility functions used throughout the Pregel loop: one for computing version diffs on checkpoints, the other for deep-searching a `Runnable` chain for an embedded subgraph.

**Key source facts:**

- `get_new_channel_versions(previous_versions, current_versions)` returns the subset of `current_versions` entries whose values are **strictly greater** than the corresponding entry in `previous_versions`. If `previous_versions` is empty, returns all of `current_versions`. The version type is determined dynamically (`type(next(iter(current_versions.values())))`) so it works for both `int` versions (default) and any custom version type. This is used by the checkpoint writer to determine which channels actually changed in this superstep and only persist those.
- `find_subgraph_pregel(candidate)` performs a BFS over a `Runnable` graph, looking for the first embedded `PregelProtocol` (subgraph). It can traverse:
  - `RunnableSequence` / `RunnableSeq` — extends candidates with `steps`.
  - `RunnableLambda` — extends with `deps`.
  - `RunnableCallable` — uses `get_function_nonlocals(func)` (AST-based closure inspection) to find closure captures.
  - **Exception**: `Pregel` instances with `checkpointer=False` (checkpointing disabled) are **not** considered subgraphs.
- These utilities are `import`-safe internal helpers — not part of the public API.

### Example 1 — channel version diff for incremental checkpoint writes

```python
from langgraph.pregel._utils import get_new_channel_versions

# Typical scenario: only "messages" and "counter" changed in this superstep
previous = {"messages": 3, "counter": 1, "config": 0}
current  = {"messages": 4, "counter": 2, "config": 0}

new = get_new_channel_versions(previous, current)
print(new)   # {"messages": 4, "counter": 2}
# "config" is absent — version unchanged, saver skips it.
```

### Example 2 — empty previous_versions (first superstep)

```python
from langgraph.pregel._utils import get_new_channel_versions

# On the very first superstep there are no previous versions
new = get_new_channel_versions({}, {"messages": 1, "counter": 1})
print(new)   # {"messages": 1, "counter": 1} — all versions returned as-is

# Custom version type (floats, strings, etc.) also works because the
# comparison is via the > operator on whatever type the values are.
float_prev = {"a": 1.0, "b": 2.0}
float_curr = {"a": 1.0, "b": 2.5}
print(get_new_channel_versions(float_prev, float_curr))  # {"b": 2.5}
```

### Example 3 — find_subgraph_pregel for subgraph wiring

```python
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._utils import find_subgraph_pregel
from langchain_core.runnables import RunnableLambda

# A subgraph that will be embedded in a parent
sub_builder = StateGraph(dict)
sub_builder.add_node("step", lambda s: s)
sub_builder.add_edge(START, "step")
sub_builder.add_edge("step", END)
subgraph = sub_builder.compile()

# find_subgraph_pregel discovers it inside a RunnableLambda closure
def make_wrapper(sg):
    def wrapper(state):
        return sg.invoke(state)
    return wrapper

runnable = RunnableLambda(make_wrapper(subgraph))
found = find_subgraph_pregel(runnable)
print(found is subgraph)  # True — closure walk found it

# A Pregel with checkpointer=False is excluded (it has no independent state)
sub_no_ckpt = sub_builder.compile(checkpointer=False)
found2 = find_subgraph_pregel(sub_no_ckpt)
print(found2 is None)  # True — checkpointer=False subgraphs don't count
```

---

## 5 · `ProtocolEvent` + `_ProtocolEventParams` anatomy

**Module:** `langgraph.stream._types`

`ProtocolEvent` is the canonical envelope for every event in the v3 streaming protocol. Understanding its exact field layout is essential for building custom `StreamTransformer` subclasses and reading raw `astream(version="v3")` output.

**Key source facts** (from `langgraph/stream/_types.py`):

- `ProtocolEvent` is a `TypedDict` with fields:
  - `"type"` — always `"event"` (literal, distinguishes from control messages).
  - `"event_id"` — `NotRequired[str]` — snake_case wire field matching the LangChain protocol; present on some events, absent on others.
  - `"seq"` — `NotRequired[int]` — monotonically increasing counter **assigned by the root `StreamMux`**. Use this for total ordering; `params.timestamp` is wall-clock and can go backwards (NTP adjustments).
  - `"method"` — the `StreamMode` value: `"values"`, `"messages"`, `"custom"`, `"tasks"`, etc.
  - `"params"` — a `_ProtocolEventParams` TypedDict:
    - `"namespace"` — `list[str]` subgraph path at the time of emission.
    - `"timestamp"` — `int` wall-clock milliseconds since epoch.
    - `"data"` — the payload, type depends on `method`.
    - `"interrupts"` — `NotRequired[tuple[Any, ...]]` — present on `"values"` events that carry pending interrupts.
- `convert_to_protocol_event(part)` bridges the v2 `StreamPart` TypedDict to a `ProtocolEvent`. It sets `"timestamp"` from `time.time() * 1000` and copies `"interrupts"` when present. The `"seq"` field is **not** set by this bridge — the root mux stamps it.

### Example 1 — reading the raw ProtocolEvent fields

```python
from langgraph.stream._types import ProtocolEvent, _ProtocolEventParams

# Construct a minimal valid ProtocolEvent for illustration
event: ProtocolEvent = {
    "type": "event",
    "seq": 42,
    "method": "values",
    "params": {
        "namespace": ["subgraph_a"],
        "timestamp": 1_700_000_000_000,
        "data": {"counter": 7},
        "interrupts": (),
    },
}

# Consumers that need a total order should use seq, not timestamp
print(event["seq"])                          # 42
print(event["method"])                       # "values"
print(event["params"]["namespace"])          # ["subgraph_a"]
print(event["params"]["data"])               # {"counter": 7}
print("interrupts" in event["params"])       # True — present on values events
```

### Example 2 — writing a StreamTransformer that reads seq for ordering

```python
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from typing import Any

class OrderTracker(StreamTransformer):
    """Tracks the highest seq seen to detect out-of-order delivery."""

    def init(self) -> dict[str, Any]:
        self._max_seq = -1
        return {}   # no projection channels

    def process(self, event: ProtocolEvent) -> bool:
        seq = event.get("seq")
        if seq is not None:
            if seq <= self._max_seq:
                # This would indicate a re-ordering bug — log it
                print(f"Warning: seq {seq} <= max seen {self._max_seq}")
            else:
                self._max_seq = seq
        return True   # pass event through to main log

# The mux assigns seq in push order, so this invariant holds for
# single-consumer graph runs.
print("OrderTracker defined — register with compile(transformers=[OrderTracker])")
```

### Example 3 — using convert_to_protocol_event for v2→v3 bridge

```python
from langgraph.stream._convert import convert_to_protocol_event
from langgraph.types import StreamPart

# v2 StreamPart — the format produced by stream_mode iteration before v3
v2_part: StreamPart = {           # type: ignore[assignment]
    "type": "values",
    "ns": ("agent",),
    "data": {"messages": ["hello"]},
    "interrupts": (),
}

event = convert_to_protocol_event(v2_part)
print(event["type"])                  # "event"
print(event["method"])                # "values"
print(event["params"]["namespace"])   # ["agent"]
print("seq" in event)                 # False — seq not set by bridge, stamped by mux
print("interrupts" in event["params"])# True — carried through when present
```

---

## 6 · `StreamTransformer` class-variable flags + `transformer_requires_async`

**Module:** `langgraph.stream._types`

Beyond the abstract `init` / `process` methods covered in Vol. 22, `StreamTransformer` exposes five class-level flags that control registration, ordering, and async detection.

**Key source facts:**

- `requires_async: ClassVar[bool] = False` — Set `True` on transformers that call `self.schedule()` from a sync `process` (e.g. they start async work but don't override `aprocess`). The mux reads this at registration time and raises a clear error when such a transformer is used under sync `stream()`.
- `supports_sync: ClassVar[bool] = False` — Override to `True` when a transformer overrides async-lane methods (`aprocess`, `afinalize`, `afail`) but **also** fully supports the sync lane. Without this, overriding any async-lane method forces async-only.
- `before_builtins: ClassVar[bool] = False` — Transformers with `True` are registered **before** built-ins (`MessagesTransformer`, `ToolCallTransformer`). Use this for **content-mutating** transformers (PII redaction, profanity filtering) whose changes must be visible to built-ins. Foot-gun: these transformers see `tasks` events before `LifecycleTransformer` and `SubgraphTransformer` — mutating `namespace` or `id`/`result`/`error`/`interrupts` fields in `data` will desync lifecycle bookkeeping.
- `required_stream_modes: ClassVar[tuple[str, ...]] = ()` — Stream modes this transformer consumes. The mux unions all registered transformers' `required_stream_modes` to decide which modes to request from the graph when none are explicitly specified in `stream_events(version="v3")`.
- `transformer_requires_async(transformer)` — convenience predicate that returns `True` if `transformer.requires_async` is `True` **or** if any of `aprocess`, `afinalize`, `afail` are overridden on the class **without** `supports_sync = True`. This is how the mux decides which lane to use.

### Example 1 — required_stream_modes auto-request

```python
from langgraph.stream._types import StreamTransformer
from typing import Any

class TaskCountTransformer(StreamTransformer):
    """Count how many task-start events fire per node."""

    # Tell the mux this transformer needs tasks mode
    required_stream_modes = ("tasks",)

    def init(self) -> dict[str, Any]:
        self._counts: dict[str, int] = {}
        return {"node_task_counts": self._counts}

    def process(self, event) -> bool:
        if event["method"] == "tasks":
            data = event["params"]["data"]
            if "result" not in data:   # task-start event
                name = data.get("name", "unknown")
                self._counts[name] = self._counts.get(name, 0) + 1
        return True

# When registered with compile(transformers=[TaskCountTransformer]),
# the mux automatically includes "tasks" in the stream modes it requests
# from the graph — the caller doesn't need to specify it.
print(TaskCountTransformer.required_stream_modes)  # ("tasks",)
```

### Example 2 — before_builtins for PII redaction

```python
from langgraph.stream._types import StreamTransformer
from typing import Any
import re

class PIIRedactTransformer(StreamTransformer):
    """Redact email addresses from messages events BEFORE MessagesTransformer snapshots them."""

    before_builtins = True   # run before MessagesTransformer / ToolCallTransformer
    required_stream_modes = ("messages",)

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event) -> bool:
        if event["method"] == "messages":
            data = event["params"]["data"]
            # Mutate message content in-place (built-ins read the same dict later)
            if isinstance(data, dict) and "content" in data:
                if isinstance(data["content"], str):
                    data["content"] = re.sub(
                        r"\b[\w.+-]+@[\w-]+\.\w+\b",
                        "[REDACTED_EMAIL]",
                        data["content"],
                    )
        return True

# Safe to mutate data["content"] here; avoid touching data["id"] or
# event["params"]["namespace"] — those are read by LifecycleTransformer.
print(f"before_builtins={PIIRedactTransformer.before_builtins}")  # True
```

### Example 3 — transformer_requires_async predicate

```python
from langgraph.stream._types import StreamTransformer, transformer_requires_async
from typing import Any

class SyncTransformer(StreamTransformer):
    def init(self): return {}
    def process(self, event): return True

class AsyncTransformer(StreamTransformer):
    def init(self): return {}
    async def aprocess(self, event): return True   # overrides async lane

class AsyncSyncTransformer(StreamTransformer):
    supports_sync = True   # explicitly declares sync is also fine
    def init(self): return {}
    async def aprocess(self, event): return True
    def process(self, event): return True

print(transformer_requires_async(SyncTransformer()))        # False
print(transformer_requires_async(AsyncTransformer()))       # True
print(transformer_requires_async(AsyncSyncTransformer()))   # False — supports_sync wins
```

---

## 7 · `patch_checkpoint_map` + `_merge_metadata`

**Module:** `langgraph._internal._config`

Two config-merge utilities that power multi-subgraph state coordination and LangChain's `lc_versions` metadata merging.

**Key source facts** (from `langgraph/_internal/_config.py`):

- `patch_checkpoint_map(config, metadata)` is used when a node transitions from a parent to a child subgraph. It adds an entry to `configurable["checkpoint_map"]` that records the **parent** namespace's current checkpoint ID. This lets the child refer back to the parent's checkpoint for cross-namespace reads (e.g. `Command.PARENT`). If `metadata` is `None` or has no `"parents"` key, the config is returned **unchanged**. The new map entry is `{parent_ns: parent_checkpoint_id}` merged with any existing `checkpoint_map`.
- `_merge_metadata(base, new)` merges two `Mapping[str, Any]` metadata dicts **without mutating either input**. Rules:
  - Top-level keys merge with `new` values winning.
  - **`lc_versions`** is the only mapping-valued key that merges one level deeper — individual package version strings are unioned so different LangChain packages can each contribute their version without clobbering the others.
  - All other mapping-valued keys are **shallow-copied one level** (so deeper nested objects remain shared).
  - `None` inputs are treated as empty metadata.
  - This mirrors `langchain_core.runnables.config._merge_metadata_dicts` and the two are kept in sync.
- The constant `DEFAULT_RECURSION_LIMIT` (default 10007) is read from `LANGGRAPH_DEFAULT_RECURSION_LIMIT` env var and controls the recursion guard before `GraphRecursionError`.

### Example 1 — patch_checkpoint_map adds parent namespace entry

```python
from langgraph._internal._config import patch_checkpoint_map

# Simulate a parent subgraph's config
parent_config = {
    "configurable": {
        "thread_id": "t1",
        "checkpoint_ns": "parent_graph",
        "checkpoint_id": "abc123",
        "checkpoint_map": {},
    }
}

# Metadata carries the parent namespace mapping from the checkpoint
metadata = {"parents": {"parent_graph": "abc123"}}

new_config = patch_checkpoint_map(parent_config, metadata)
cmap = new_config["configurable"]["checkpoint_map"]
print(cmap)   # {"parent_graph": "abc123"}
# Now the child subgraph can look up parent_graph's checkpoint ID via checkpoint_map.
```

### Example 2 — _merge_metadata deep-merges lc_versions

```python
from langgraph._internal._config import _merge_metadata

# Two metadata dicts from different LangChain packages
base = {
    "lc_versions": {"langchain-core": "0.3.0", "langchain": "0.3.0"},
    "run_name": "agent",
}
new = {
    "lc_versions": {"langchain-core": "0.3.1", "langchain-anthropic": "0.2.0"},
    "run_name": "agent_v2",
}

merged = _merge_metadata(base, new)
print(merged["run_name"])            # "agent_v2" — new wins at top level
print(merged["lc_versions"])
# {"langchain-core": "0.3.1",     ← new wins (version upgraded)
#  "langchain": "0.3.0",          ← preserved from base
#  "langchain-anthropic": "0.2.0" ← added from new}
```

### Example 3 — DEFAULT_RECURSION_LIMIT env override

```python
import os
from langgraph._internal._config import DEFAULT_RECURSION_LIMIT

# Default
print(f"Default recursion limit: {DEFAULT_RECURSION_LIMIT}")   # 10007

# Customise for a graph that legitimately has many supersteps
# (set BEFORE import — the constant is read at module import time)
os.environ["LANGGRAPH_DEFAULT_RECURSION_LIMIT"] = "50000"

import importlib
import langgraph._internal._config as _cfg
importlib.reload(_cfg)
print(f"After env change: {_cfg.DEFAULT_RECURSION_LIMIT}")      # 50000

# Pass it as the recursion_limit in your compile config:
# graph.invoke(state, config={"recursion_limit": 50000})
```

---

## 8 · `_TasksLifecycleBase` template-method hooks

**Module:** `langgraph.stream.transformers`

`_TasksLifecycleBase` is the shared bookkeeping engine behind both `LifecycleTransformer` (wire-serializable channel for production) and `SubgraphTransformer` (in-process navigation handles). Understanding its hooks enables building custom lifecycle observers.

**Key source facts** (from `langgraph/stream/transformers.py`):

- `required_stream_modes = ("tasks",)` — declares that this transformer needs the graph to emit `tasks` events.
- Internal state:
  - `_seen: set[tuple[str, ...]]` — namespaces already dispatched to `_on_started`.
  - `_open: dict[tuple[str, ...], str]` — maps tracked namespace → parent task_id whose `TaskResultPayload` will close it.
  - `_lc_by_ns: dict[tuple[str, ...], str | None]` — `lc_agent_name` per namespace (first task event wins).
  - `_pending_tool_calls: dict[str, str]` — `task_id → tool_call_id`, harvested from `tool_call_with_context` input shape or legacy list shape.
  - `_pending_cause` — set immediately before `_on_started` dispatch so overrides can read the cause without a signature change.
- **Template-method hooks** (override in subclasses):
  - `_should_track(ns)` — scope filter; return `True` iff this namespace belongs to this transformer's region.
  - `_on_started(ns, graph_name, trigger_call_id)` — called once per discovered namespace (first observed task event).
  - `_on_terminal(ns, status, error)` — called once per tracked namespace when its parent `TaskResultPayload` arrives, or during `finalize`/`fail` safety-net sweeps.
- `process()` always returns `False` — tasks events are suppressed from the main event log and folded into the subclass's projection.
- `_record_pending_tool_calls` handles **two input shapes**: current Pregel push model (`input` is `tool_call_with_context` dict) and legacy batched model (`input` is a list of tool-call dicts).

### Example 1 — minimal custom lifecycle observer

```python
from langgraph.stream.transformers import _TasksLifecycleBase
from typing import Any

class SubgraphCounter(_TasksLifecycleBase):
    """Count subgraph start/end events from the root scope."""

    def init(self) -> dict[str, Any]:
        self.started = 0
        self.finished = 0
        return {"subgraph_counts": self}

    def _should_track(self, ns):
        # Track direct children of the root scope only (depth == 1)
        return len(ns) == len(self.scope) + 1 and ns[: len(self.scope)] == self.scope

    def _on_started(self, ns, graph_name, trigger_call_id):
        self.started += 1
        print(f"Subgraph started: {'/'.join(ns)} (agent={graph_name})")

    def _on_terminal(self, ns, status, error):
        self.finished += 1
        print(f"Subgraph {status}: {'/'.join(ns)} error={error}")

# Register: graph = builder.compile(transformers=[SubgraphCounter])
print("SubgraphCounter defined — register via compile(transformers=[SubgraphCounter])")
```

### Example 2 — _record_pending_tool_calls cross-payload lookup

```python
# Demonstrates why _pending_tool_calls exists:
# A subagent's namespace segment is "node_name:{task_id}".
# The tool_call_id that triggered that task lives in the PARENT's TaskPayload.
# _record_pending_tool_calls harvests it into a dict so _handle_task_start
# can recover the causal tool_call_id when the child's first task event arrives.

# Two input shapes handled:
# 1. Current shape (single tool call per push task):
single = {
    "id": "task_abc",
    "input": {
        "tool_call": {"id": "tc_123", "name": "search", "args": {}},
        "__type": "tool_call_with_context",
        "state": {},
    },
}

# 2. Legacy shape (batched list):
batched = {
    "id": "task_def",
    "input": [
        {"id": "tc_456", "name": "lookup", "args": {}},
        {"id": "tc_789", "name": "calc",   "args": {}},
    ],
}

# The harvesting logic (from _record_pending_tool_calls):
for task_data in [single, batched]:
    task_id = task_data.get("id")
    payload = task_data.get("input")
    tool_call_id = None
    if isinstance(payload, dict) and isinstance(payload.get("tool_call"), dict):
        tool_call_id = payload["tool_call"].get("id")
    elif isinstance(payload, list):
        for tc in payload:
            if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                tool_call_id = tc["id"]
                break
    print(f"task_id={task_id} → trigger_tool_call_id={tool_call_id}")
# task_id=task_abc → trigger_tool_call_id=tc_123
# task_id=task_def → trigger_tool_call_id=tc_456
```

### Example 3 — finalize / fail safety-net sweeps

```python
# _TasksLifecycleBase.finalize() (and fail()) sweep _open to close any
# tracked namespaces whose parent TaskResultPayload never arrived
# (e.g. graph aborted mid-run, subgraph timed out).
#
# Without this sweep, a SubgraphRunStream would remain in "started" status
# indefinitely, blocking consumers iterating run.subgraphs.
#
# The sweep fires _on_terminal(ns, "drained", None) for every open entry.

# Subclass skeleton showing the safety-net hook:
from langgraph.stream.transformers import _TasksLifecycleBase
from typing import Any

class SafeNetDemo(_TasksLifecycleBase):
    def init(self) -> dict[str, Any]:
        self._statuses: dict[str, str] = {}
        return {}

    def _should_track(self, ns): return len(ns) == 1

    def _on_started(self, ns, graph_name, trigger_call_id):
        self._statuses["/".join(ns)] = "started"

    def _on_terminal(self, ns, status, error):
        self._statuses["/".join(ns)] = status

    def finalize(self):
        # _TasksLifecycleBase.finalize() calls _pop_terminal_transitions()
        # for all _open entries with status="drained".
        super().finalize()

    def fail(self, err):
        # _TasksLifecycleBase.fail() calls _pop_terminal_transitions()
        # for all _open entries with status="failed".
        super().fail(err)

print("SafeNetDemo: finalize/fail close all open subgraph handles")
```

---

## 9 · `PregelRunner.commit()` + `_should_route_to_error_handler` + `_should_stop_others`

**Module:** `langgraph.pregel._runner`

`PregelRunner.commit()` is called by every completed `concurrent.futures.Future` as a done-callback. It is the synchronisation point where writes reach the checkpoint and errors are classified and routed.

**Key source facts** (from `langgraph/pregel/_runner.py`):

- `commit(task, exception)` branches on the exception type:
  - `asyncio.CancelledError` — append `(ERROR, exc)` to writes **and** call `put_writes(task.id, task.writes)`. The loop can then finish the super-step.
  - `GraphInterrupt` — if `exception.args[0]` is non-empty, persist `[(INTERRUPT, args), *resumes]` writes. `resumes` is the list of `RESUME` writes already in `task.writes` (accumulated from prior `interrupt()` calls in the same task).
  - `GraphBubbleUp` — **no writes persisted**; the exception propagates via `_panic_or_proceed`.
  - Any other exception — append `(ERROR, exc)` to writes; if `_should_route_to_error_handler(task)`, also append `(ERROR_SOURCE_NODE, task.name)` and add `id(exc)` to `_handled_exception_ids` (so `_should_stop_others` skips it). Then `put_writes`.
  - No exception — call `node_finished(task.name)` (unless `TAG_HIDDEN` is in config tags), append `(NO_WRITES, None)` if there are no writes, then `put_writes`.
- `_should_route_to_error_handler(task)` returns `True` when:
  1. The task's name has a mapped error handler (`task.name in node_error_handler_map`), **and**
  2. The task is not itself an error handler (`task.name not in error_handler_nodes`).
- `_should_stop_others(done, *, handled_exception_ids)` returns `True` if any **non-cancelled**, **non-GraphBubbleUp**, **non-already-handled** future raised an exception. This drives the early-cancel of sibling tasks in the current super-step.

### Example 1 — error routing with add_node(error_handler=...)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

def risky_node(state):
    if state.get("trigger_error"):
        raise RuntimeError("Something went wrong")
    return {"value": 1}

def error_handler(state):
    # Receives NodeError in state["error"] when registered via add_node(error_handler=)
    return {"value": -1, "error_handled": True}

builder = StateGraph(dict)
builder.add_node("risky", risky_node, error_handler=error_handler)
builder.add_edge(START, "risky")
builder.add_edge("risky", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}  # required by InMemorySaver

# Normal path
r1 = graph.invoke({"trigger_error": False}, config=config)
print(r1)   # {"trigger_error": False, "value": 1}

# Error path — PregelRunner.commit() routes to error_handler via
# _should_route_to_error_handler and appends ERROR_SOURCE_NODE write
config2 = {"configurable": {"thread_id": "t2"}}
r2 = graph.invoke({"trigger_error": True}, config=config2)
print(r2)   # {"trigger_error": True, "value": -1, "error_handled": True}
```

### Example 2 — _should_stop_others early-cancel behaviour

```python
# When one parallel task raises (non-GraphBubbleUp, non-handled),
# _should_stop_others returns True and the runner cancels sibling tasks.
# The _handled_exception_ids set prevents double-reporting when an error
# was already routed to an error_handler node.
from langgraph.graph import StateGraph, START, END
import time

def slow_task(state):
    # Note: Python cannot interrupt a blocking sleep in a thread-pool worker;
    # use a very short delay here so the example completes quickly.
    time.sleep(0.05)
    return {"slow": True}

def fast_fail(state):
    raise ValueError("fast failure")

builder = StateGraph(dict)
builder.add_node("slow", slow_task)
builder.add_node("fail", fast_fail)
builder.add_edge(START, "slow")
builder.add_edge(START, "fail")
builder.add_edge("slow", END)
builder.add_edge("fail", END)

graph = builder.compile()
try:
    # "fail" node errors → _should_stop_others returns True → "slow" future is cancelled
    graph.invoke({})
except Exception as e:
    print(f"Error raised: {type(e).__name__}: {e}")
# The future is marked cancelled; the short sleep means the run finishes promptly.
```

### Example 3 — _handled_exception_ids prevents double-cancel

```python
# When an error is routed to an error_handler, id(exc) is added to
# _handled_exception_ids. _should_stop_others skips futures whose
# exception id is in that set, so the error_handler can complete
# without being aborted by its own "failed" sibling.

# Pseudocode from _should_stop_others:
#
#   for fut in done:
#       if fut.cancelled():
#           continue
#       elif exc := fut.exception():
#           if (id(exc) not in handled_exception_ids
#               and not isinstance(exc, GraphBubbleUp)
#               and fut not in SKIP_RERAISE_SET):
#               return True  # cancel others
#   return False

# In practice: if you use add_node(error_handler=...) and see unexpected
# cancellations, verify that error_handler is listed in node_error_handler_map
# and NOT in error_handler_nodes (which would skip routing).

from langgraph.pregel._runner import _should_stop_others
import concurrent.futures

# Simulate a handled exception
class FakeException(Exception): pass
exc = FakeException("handled")
handled_ids = {id(exc)}

fut = concurrent.futures.Future()
fut.set_exception(exc)

result = _should_stop_others({fut}, handled_exception_ids=handled_ids)
print(result)  # False — handled exception does not trigger cancel-others
```

---

## 10 · `draw_graph` static topology inference

**Module:** `langgraph.pregel._draw`

`draw_graph` infers the graph's edge topology **without executing any node functions**. It is called by `graph.get_graph()` to produce the `Graph` object that powers `draw_mermaid()` and `draw_ascii()`.

**Key source facts** (from `langgraph/pregel/_draw.py`):

- It runs a **simulated Pregel loop** with an `empty_checkpoint()` and an empty input `{}`. No real node callables are invoked — only `ChannelWrite` writers are probed for their **statically declared writes** via `ChannelWrite.get_static_writes(w)`.
- The loop runs for up to `limit` (default 250) supersteps. At each step it calls `prepare_next_tasks(...)` then iterates each task's writers, collecting `Edge` and `TriggerEdge` NamedTuples.
- **Conditional edges** (`add_conditional_edges`) declare their possible targets via `ChannelWrite.get_static_writes`, which returns a list of `(target, is_conditional, label)` triples. Writes to `END` become `Edge(src, END, True, label)` directly instead of being channeled.
- `get_next_version` is borrowed from the attached `checkpointer` if it is a `BaseCheckpointSaver`; otherwise it falls back to `increment` (simple integer counter). This ensures version bumps are compatible with the configured backend.
- Node mappers (`v.mapper`) are stripped before the simulation: `v.copy(update={"mapper": None})`. This prevents mapper side effects from running during topology inference.

### Example 1 — inspect a graph's topology via get_graph()

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(dict)
builder.add_node("A", lambda s: s)
builder.add_node("B", lambda s: s)
builder.add_node("C", lambda s: s)
builder.add_edge(START, "A")
builder.add_edge("A", "B")
builder.add_conditional_edges("B", lambda s: "C" if s.get("go_c") else END,
                              {"C": "C", END: END})
builder.add_edge("C", END)
graph = builder.compile()

g = graph.get_graph()
for node in g.nodes:
    print(f"Node: {node}")
for edge in g.edges:
    print(f"Edge: {edge.source} → {edge.target} (conditional={edge.conditional})")
# The topology is computed by draw_graph without running A, B, or C.
```

### Example 2 — xray=True reveals subgraph internals

```python
from langgraph.graph import StateGraph, START, END

# Build a subgraph
sub = StateGraph(dict)
sub.add_node("inner", lambda s: s)
sub.add_edge(START, "inner")
sub.add_edge("inner", END)
subgraph = sub.compile()

# Build the parent graph
parent = StateGraph(dict)
parent.add_node("sub", subgraph)
parent.add_edge(START, "sub")
parent.add_edge("sub", END)
parent_graph = parent.compile()

# xray=True expands subgraph internals in the topology diagram
g_shallow = parent_graph.get_graph()
g_deep    = parent_graph.get_graph(xray=True)

print(f"Shallow nodes: {list(g_shallow.nodes)}")   # ["__start__", "sub", "__end__"]
print(f"Deep nodes:    {list(g_deep.nodes)}")       # also includes inner nodes of subgraph
```

### Example 3 — Mermaid diagram from static analysis

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(dict)
builder.add_node("planner",  lambda s: s)
builder.add_node("executor", lambda s: s)
builder.add_node("reviewer", lambda s: s)
builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_conditional_edges(
    "executor",
    lambda s: "reviewer" if s.get("needs_review") else END,
    {"reviewer": "reviewer", END: END},
)
builder.add_edge("reviewer", "executor")  # retry loop
graph = builder.compile()

mermaid = graph.get_graph().draw_mermaid()
print(mermaid)
# %%{init: ...}%%
# graph TD;
#     __start__ --> planner;
#     planner --> executor;
#     executor -. conditional .-> reviewer;
#     executor -. conditional .-> __end__;
#     reviewer --> executor;
# The diagram is generated entirely from static write analysis —
# no node functions are called.
```

---
title: "Class deep-dives Vol. 26 — channels, callbacks, cache & prebuilt streaming (1.2.6)"
description: "Source-verified deep dives into 10 previously under-documented class groups in LangGraph 1.2.6: GraphCallbackHandler+GraphInterruptEvent+GraphResumeEvent (public lifecycle callback surface — on_interrupt/on_resume hooks, frozen event dataclasses, config-based injection via get_sync/async_graph_callback_manager_for_config), DeltaChannel (beta reducer channel — replay_writes() batching-invariant contract, snapshot_frequency cadence, three-path from_checkpoint() — MISSING+_DeltaSnapshot+plain-value migration, Overwrite reset semantics), InMemoryCache (thread-safe in-process TTL cache — RLock mechanics, UTC timestamp expiry, serde round-trip on get/set, namespace-scoped clear(), async-as-sync symmetry), LastValue+LastValueAfterFinish (per-step channels — LastValue INVALID_CONCURRENT_GRAPH_UPDATE guard, LastValueAfterFinish finish()+consume() two-stage gate, (value, bool) checkpoint tuple), BranchSpec (add_conditional_edges NamedTuple — path/ends/input_schema fields, from_path() factory handling dict/list path_map, _get_branch_path_input_schema() type-hint introspection), JsonPlusSerializer (ormsgpack checkpoint serde — pickle_fallback, LANGGRAPH_STRICT_MSGPACK allowlist, _is_safe_json_type legacy-resume gate, _warn_once circuit breaker), ToolCallStream+ToolCallTransformer (per-call streaming handles — output_deltas StreamChannel, _bind_pump wiring, process() event routing, finalize()/fail() teardown), AnyValue (last-write-wins channel — update([]) clears to MISSING, __eq__ matches all AnyValue instances, contrast with LastValue strict guard), NamedBarrierValue+NamedBarrierValueAfterFinish (barrier fan-in channels — names/seen set tracking, consume() reset, NamedBarrierValueAfterFinish finish() two-stage gate, (seen, bool) checkpoint tuple), and EphemeralValue (single-step ephemeral channel — guard=True one-write-per-step enforcement, update([]) clears to MISSING, trigger-channel pattern for pass-through activation)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 26"
  order: 57
---

# Class deep-dives Vol. 26 — channels, callbacks, cache & prebuilt streaming (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---|---|
| 1 | `GraphCallbackHandler` + `GraphInterruptEvent` + `GraphResumeEvent` | `langgraph.callbacks` |
| 2 | `DeltaChannel` | `langgraph.channels.delta` |
| 3 | `InMemoryCache` | `langgraph.cache.memory` |
| 4 | `LastValue` + `LastValueAfterFinish` | `langgraph.channels.last_value` |
| 5 | `BranchSpec` | `langgraph.graph._branch` |
| 6 | `JsonPlusSerializer` | `langgraph.checkpoint.serde.jsonplus` |
| 7 | `ToolCallStream` + `ToolCallTransformer` | `langgraph.prebuilt._tool_call_stream` + `._tool_call_transformer` |
| 8 | `AnyValue` | `langgraph.channels.any_value` |
| 9 | `NamedBarrierValue` + `NamedBarrierValueAfterFinish` | `langgraph.channels.named_barrier_value` |
| 10 | `EphemeralValue` | `langgraph.channels.ephemeral_value` |

---

## 1 · `GraphCallbackHandler` + `GraphInterruptEvent` + `GraphResumeEvent`

**Module**: `langgraph.callbacks`  
**First source-verified coverage with full event anatomy and factory patterns.**

`GraphCallbackHandler` is LangGraph's extension of LangChain's `BaseCallbackHandler`, adding two graph-specific lifecycle hooks: `on_interrupt` (fired when execution pauses at an `interrupt()` call) and `on_resume` (fired when a paused graph resumes from a checkpoint). Unlike generic LangChain callback events, these two fire exclusively through LangGraph's Pregel loop — you cannot observe them via a standard `BaseCallbackHandler`.

```python
# Class signatures (from langgraph/callbacks.py)
from dataclasses import dataclass
from typing import Literal, TypeAlias
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.types import Interrupt

GraphLifecycleStatus: TypeAlias = Literal[
    "input", "pending", "done",
    "interrupt_before", "interrupt_after", "out_of_steps",
]

@dataclass(frozen=True)
class GraphInterruptEvent:
    run_id: UUID | None
    status: GraphLifecycleStatus
    checkpoint_id: str
    checkpoint_ns: tuple[str, ...]
    interrupts: tuple[Interrupt, ...]

@dataclass(frozen=True)
class GraphResumeEvent:
    run_id: UUID | None
    status: GraphLifecycleStatus
    checkpoint_id: str
    checkpoint_ns: tuple[str, ...]

class GraphCallbackHandler(BaseCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None: ...
    def on_resume(self, event: GraphResumeEvent) -> None: ...
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| `frozen=True` | Both event dataclasses are immutable — safe to store or forward across threads |
| `checkpoint_ns` | Tuple of namespace segments, e.g. `("subgraph@run1",)` for a subgraph |
| `interrupts` | All `Interrupt` payloads collected in the current pause (may be multiple if parallel nodes each called `interrupt()`) |
| `status` | The Pregel loop status at the moment the event fires — `"interrupt_before"` or `"interrupt_after"` for user-triggered pauses |
| Factory functions | `get_sync_graph_callback_manager_for_config(config, *, run_id)` / `get_async_graph_callback_manager_for_config(config, *, run_id)` filter `config["callbacks"]` to `GraphCallbackHandler` instances and return a manager |

### Example 1 — Audit logger that records every interrupt and resume

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.callbacks import (
    GraphCallbackHandler,
    GraphInterruptEvent,
    GraphResumeEvent,
)


class AuditLogger(GraphCallbackHandler):
    """Records interrupt and resume events for compliance / debugging."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        self.events.append(
            f"INTERRUPTED  checkpoint={event.checkpoint_id[:8]}  "
            f"status={event.status}  "
            f"payloads={[i.value for i in event.interrupts]}"
        )

    def on_resume(self, event: GraphResumeEvent) -> None:
        self.events.append(
            f"RESUMED      checkpoint={event.checkpoint_id[:8]}  "
            f"status={event.status}"
        )


class State(TypedDict):
    draft: str
    approved: bool


def write_draft(state: State) -> dict:
    return {"draft": "My important document"}


def review(state: State) -> dict:
    decision = interrupt({"question": "Approve draft?", "draft": state["draft"]})
    return {"approved": decision}


builder = StateGraph(State)
builder.add_node("write", write_draft)
builder.add_node("review", review)
builder.add_edge(START, "write")
builder.add_edge("write", "review")
builder.add_edge("review", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

logger = AuditLogger()
config = {"configurable": {"thread_id": "doc-42"}, "callbacks": [logger]}

# First run — pauses at interrupt
try:
    graph.invoke({"draft": "", "approved": False}, config)
except Exception:
    pass

print("After pause:", logger.events)
# After pause: ['INTERRUPTED  checkpoint=<id>  status=interrupt_before  payloads=[{...}]']

# Resume — human approves; Command(resume=value) supplies the value back to interrupt()
graph.invoke(Command(resume=True), config)
print("After resume:", logger.events[-1])
# After resume: 'RESUMED      checkpoint=<id>  status=interrupt_before'
```

### Example 2 — Config-level injection using the factory function

The `get_sync_graph_callback_manager_for_config` factory filters any existing `config["callbacks"]` down to `GraphCallbackHandler` instances. This is how the Pregel loop itself constructs the manager on each run — you can call the same factory to compose or inspect the managers.

```python
from uuid import uuid4
from langgraph.callbacks import (
    GraphCallbackHandler,
    GraphInterruptEvent,
    GraphResumeEvent,
    get_sync_graph_callback_manager_for_config,
)
from langchain_core.callbacks import BaseCallbackHandler


class SimpleLogger(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        print(f"[INTERRUPT] ns={event.checkpoint_ns} interrupts={len(event.interrupts)}")

    def on_resume(self, event: GraphResumeEvent) -> None:
        print(f"[RESUME] checkpoint={event.checkpoint_id[:8]}")


class NonGraphHandler(BaseCallbackHandler):
    """Ordinary LangChain callback — will be filtered out."""
    pass


run_id = uuid4()
config = {
    "callbacks": [SimpleLogger(), NonGraphHandler()],
    "configurable": {"thread_id": "t1"},
}

# Factory filters to GraphCallbackHandler instances and binds run_id
manager = get_sync_graph_callback_manager_for_config(config, run_id=run_id)
print(f"Active graph handlers: {len(manager.handlers)}")  # 1 (NonGraphHandler excluded)
print(f"Manager run_id: {manager.run_id}")  # bound UUID

# The manager exposes on_interrupt / on_resume
# (called internally by the Pregel loop — shown here for illustration)
from langgraph.types import Interrupt
from langgraph.callbacks import GraphInterruptEvent
evt = GraphInterruptEvent(
    run_id=run_id,
    status="interrupt_before",
    checkpoint_id="abc123",
    checkpoint_ns=(),
    interrupts=(Interrupt(value="Review me"),),
)
manager.on_interrupt(evt)
# [INTERRUPT] ns=() interrupts=1
```

### Example 3 — Async subgraph event with namespace tracking

When a subgraph fires an interrupt, `checkpoint_ns` captures the full namespace path. This lets a monitoring callback distinguish root-graph interrupts from subgraph interrupts.

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent


class NamespaceTracker(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        ns_depth = len(event.checkpoint_ns)
        prefix = "  " * ns_depth
        ctx = "subgraph" if ns_depth > 0 else "root"
        print(f"{prefix}[{ctx}] interrupt  ns={event.checkpoint_ns}  status={event.status}")


class SubState(TypedDict):
    sub_done: bool


def sub_node(state: SubState) -> dict:
    result = interrupt("sub-graph approval needed")
    return {"sub_done": result}


sub_builder = StateGraph(SubState)
sub_builder.add_node("sub_node", sub_node)
sub_builder.add_edge(START, "sub_node")
sub_builder.add_edge("sub_node", END)
subgraph = sub_builder.compile()

# Add the compiled subgraph directly as a node — LangGraph propagates
# config (including callbacks and checkpointer) automatically.
root_builder = StateGraph(SubState)
root_builder.add_node("sub", subgraph)
root_builder.add_edge(START, "sub")
root_builder.add_edge("sub", END)

checkpointer = InMemorySaver()
graph = root_builder.compile(checkpointer=checkpointer)

tracker = NamespaceTracker()
config = {"configurable": {"thread_id": "ns-test"}, "callbacks": [tracker]}
try:
    graph.invoke({"sub_done": False}, config)
except Exception:
    pass
# [subgraph] interrupt  ns=('sub:...',)  status=interrupt_before
```

---

## 2 · `DeltaChannel`

**Module**: `langgraph.channels.delta`  
**Beta. First source-verified coverage of `replay_writes()`, `snapshot_frequency`, and the three-path `from_checkpoint()` contract.**

`DeltaChannel` is a reducer channel that stores **only a sentinel** (`MISSING`) in each checkpoint blob, then reconstructs its current value by replaying ancestor writes through the reducer on demand. This keeps individual checkpoint blobs tiny while still supporting large accumulated values — at the cost of a replay walk on restore.

```python
# Class signature (langgraph/channels/delta.py)
from collections.abc import Callable, Sequence
from langgraph.channels.base import BaseChannel, Value
from langgraph.checkpoint.serde.types import _DeltaSnapshot

class DeltaChannel(BaseChannel[Any, Any, Any]):
    """Reducer channel that stores only a sentinel in checkpoint blobs and
    reconstructs state by replaying ancestor writes through the reducer."""

    __slots__ = ("value", "reducer", "snapshot_frequency")

    def __init__(
        self,
        reducer: Callable[[Any, Sequence[Any]], Any],
        typ: type[Value] | None = None,
        *,
        snapshot_frequency: int = 1000,   # write snapshot every N updates
    ) -> None: ...

    def replay_writes(self, writes: Sequence[PendingWrite]) -> None:
        """Apply ancestor writes oldest-to-newest via a single reducer call."""

    def checkpoint(self) -> Any:
        """Always returns MISSING — the saver writes _DeltaSnapshot blobs separately."""
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| `snapshot_frequency` | Every Nth update OR when supersteps since last snapshot exceeds `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` (default 5000) — whichever triggers first — a full `_DeltaSnapshot(value)` blob is written so replay depth is bounded |
| `replay_writes()` | Receives writes oldest-to-newest; if any write is an `Overwrite`, the last `Overwrite` value becomes the new base and only subsequent writes go to the reducer |
| `from_checkpoint()` | Three paths: `MISSING` → start empty, caller replays writes; `_DeltaSnapshot` → restore value directly; plain value → backwards-compatibility migration from old `BinaryOperatorAggregate` blobs |
| `checkpoint()` | Always returns `MISSING` — snapshot decisions live in `create_checkpoint` which writes `_DeltaSnapshot(ch.get())` directly into `channel_values` |
| Reducer contract | Must be deterministic AND batching-invariant: `reducer(reducer(s, xs), ys) == reducer(s, xs + ys)` |

### Example 1 — Basic DeltaChannel as a list accumulator

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.delta import DeltaChannel


def list_reducer(state: list, writes: list) -> list:
    """Batching-invariant: appending is associative and order-preserving."""
    result = list(state)
    result.extend(writes)
    return result


class State(TypedDict):
    # DeltaChannel via Annotated — stores sentinel, replays writes on restore
    log: Annotated[list[str], DeltaChannel(list_reducer)]
    step: int


def step_a(state: State) -> dict:
    # Write a scalar string — the reducer receives ["step-a done"] as the writes batch
    return {"log": "step-a done", "step": state["step"] + 1}


def step_b(state: State) -> dict:
    return {"log": "step-b done", "step": state["step"] + 1}


builder = StateGraph(State)
builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
graph = builder.compile()

result = graph.invoke({"log": [], "step": 0})
print(result["log"])   # ['step-a done', 'step-b done']
print(result["step"])  # 2
```

### Example 2 — Custom snapshot_frequency and inspecting checkpoint behaviour

```python
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph._internal._typing import MISSING


def sum_reducer(state: int, writes: list[int]) -> int:
    return state + sum(writes)


# Low snapshot_frequency for demonstration — snapshot every 3 updates
ch = DeltaChannel(sum_reducer, int, snapshot_frequency=3)
ch.key = "running_total"

# Start from empty checkpoint — from_checkpoint(MISSING) sets value to self.typ() = int() = 0
ch = ch.from_checkpoint(MISSING)
print(f"Initial value: {ch.value}")  # 0 — typ() initializes to zero, not MISSING

# Simulate 3 updates — would trigger snapshot on 3rd in real Pregel loop
ch.update([10])
ch.update([20])
ch.update([30])
print(f"After 3 updates: {ch.get()}")  # 60

# checkpoint() always returns MISSING for DeltaChannel
blob = ch.checkpoint()
print(f"Checkpoint blob is MISSING: {blob is MISSING}")  # True

# Simulate what create_checkpoint writes: a _DeltaSnapshot
snapshot = _DeltaSnapshot(value=ch.get())
# Restoring from snapshot — direct value restore, no replay needed
ch2 = DeltaChannel(sum_reducer, int, snapshot_frequency=3)
ch2 = ch2.from_checkpoint(snapshot)
print(f"Restored from snapshot: {ch2.get()}")  # 60

# Restoring from MISSING — caller must call replay_writes()
ch3 = DeltaChannel(sum_reducer, int)
ch3 = ch3.from_checkpoint(MISSING)
# Replay ancestor writes (simulated)
from langgraph.checkpoint.base import PendingWrite
pending: list[PendingWrite] = [
    ("task1", "running_total", 10),
    ("task2", "running_total", 20),
    ("task3", "running_total", 30),
]
ch3.replay_writes(pending)
print(f"Restored via replay: {ch3.get()}")  # 60
```

### Example 3 — Overwrite reset inside DeltaChannel

`DeltaChannel` respects the `Overwrite` sentinel: the last `Overwrite` in a write batch becomes the new base, and only subsequent writes are passed to the reducer.

```python
from langgraph.channels.delta import DeltaChannel
from langgraph.types import Overwrite
from langgraph._internal._typing import MISSING


def concat_reducer(state: str, writes: list[str]) -> str:
    return state + "".join(writes)


ch = DeltaChannel(concat_reducer, str)
ch.key = "text"
ch = ch.from_checkpoint(MISSING)

# Normal accumulation
ch.update(["Hello"])
ch.update([" World"])
print(ch.get())   # Hello World

# Overwrite resets base, then appends trailing writes in same batch
ch.update([Overwrite("RESET"), " suffix"])
print(ch.get())   # RESET suffix

# replay_writes also handles Overwrite:
# last Overwrite becomes base, subsequent values pass to reducer
from langgraph.checkpoint.base import PendingWrite
ch4 = DeltaChannel(concat_reducer, str)
ch4.key = "text"
ch4 = ch4.from_checkpoint(MISSING)
pending: list[PendingWrite] = [
    ("t1", "text", "prefix"),
    ("t2", "text", Overwrite("FRESH")),   # reset point
    ("t3", "text", "-appended"),           # applied after reset
]
ch4.replay_writes(pending)
print(ch4.get())  # FRESH-appended
```

---

## 3 · `InMemoryCache`

**Module**: `langgraph.cache.memory`  
**First source-verified coverage of thread-safety, TTL, and serde round-trip mechanics.**

`InMemoryCache` is the built-in in-process cache backend. It subclasses `BaseCache[ValueT]` and stores serialized `(type_tag: str, data: bytes, expiry: float | None)` triples in a nested `dict[Namespace, dict[str, tuple]]`. The `RLock` (reentrant lock) allows the same thread to acquire the lock multiple times without deadlocking — important because `aset` calls `set`, and `aclear` calls `clear`, while both hold the lock.

```python
# Class signature (langgraph/cache/memory/__init__.py)
import threading
from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol

class InMemoryCache(BaseCache[ValueT]):
    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, float | None]]] = {}
        self._lock = threading.RLock()
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| Storage format | `(enc: str, val: bytes, expiry: float \| None)` — the `enc` type tag and `val` bytes come from `serde.dumps_typed()`; `expiry` is a UTC POSIX timestamp or `None` for no expiry |
| TTL expiry check | `now = datetime.now(timezone.utc).timestamp()` → `if expiry is None or now < expiry` — expired entries are lazily deleted on read, not on a background sweep |
| `RLock` | Reentrant lock protects all reads and writes; async methods are trivial wrappers over sync, so the same lock serves both |
| Serde round-trip | `set()` calls `self.serde.dumps_typed(value)` → `(enc, bytes)`; `get()` calls `self.serde.loads_typed((enc, bytes))` → reconstructed Python object |
| Namespace clearing | `clear(namespaces=[ns])` removes only the specified namespace dicts; `clear()` with no args removes everything |

### Example 1 — Basic put / get with TTL expiry

```python
import time
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.base import Namespace, FullKey


cache: InMemoryCache = InMemoryCache()

ns: Namespace = ("summarize",)
key: FullKey = (ns, "doc-abc123")

# set() takes {FullKey: (value, ttl_seconds)}
cache.set({key: ("This is the summary text.", 2)})  # expires in 2 seconds

# get() returns dict[FullKey, value] — only non-expired entries
result = cache.get([key])
print(result[key])   # 'This is the summary text.'

# Wait for expiry
time.sleep(2.5)
result_after = cache.get([key])
print(result_after)  # {} — expired entry lazily removed
```

### Example 2 — Wiring InMemoryCache into a StateGraph via CachePolicy

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


call_count = 0


class State(TypedDict):
    question: str
    answer: str


def expensive_llm(state: State) -> dict:
    global call_count
    call_count += 1
    print(f"  [LLM call #{call_count}] q={state['question']!r}")
    return {"answer": f"Answer to: {state['question']}"}


builder = StateGraph(State)
# cache_policy= enables per-node caching; compile(cache=...) provides the backend
builder.add_node("llm", expensive_llm, cache_policy=CachePolicy())
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

cache = InMemoryCache()
graph = builder.compile(cache=cache)

# First call — hits the LLM
r1 = graph.invoke({"question": "What is LangGraph?", "answer": ""})
print(r1["answer"])  # Answer to: What is LangGraph?

# Identical input — cache hit, no LLM call
r2 = graph.invoke({"question": "What is LangGraph?", "answer": ""})
print(r2["answer"])  # Answer to: What is LangGraph? (from cache)
print(f"Total LLM calls: {call_count}")  # 1 — only one real call
```

### Example 3 — Namespace isolation and thread-safe concurrent access

```python
import threading
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.base import Namespace, FullKey


cache: InMemoryCache = InMemoryCache()

ns_alice: Namespace = ("summarize", "user_alice")
ns_bob: Namespace   = ("summarize", "user_bob")

key_alice: FullKey = (ns_alice, "doc-1")
key_bob:   FullKey = (ns_bob,   "doc-1")

# Same document key, different user namespaces — independent entries
cache.set({
    key_alice: ("Alice's summary", None),
    key_bob:   ("Bob's summary",   None),
})

errors = []

def worker_alice():
    for _ in range(50):
        result = cache.get([key_alice])
        if result.get(key_alice) != "Alice's summary":
            errors.append("alice mismatch")

def worker_bob():
    for _ in range(50):
        result = cache.get([key_bob])
        if result.get(key_bob) != "Bob's summary":
            errors.append("bob mismatch")

# Run concurrent readers — RLock ensures no corruption
threads = [threading.Thread(target=worker_alice), threading.Thread(target=worker_bob)]
for t in threads: t.start()
for t in threads: t.join()

print(f"Errors: {errors}")  # []

# Namespace-scoped clear: removing Alice leaves Bob untouched
cache.clear(namespaces=[ns_alice])
print(cache.get([key_alice, key_bob]))
# {(('summarize', 'user_bob'), 'doc-1'): "Bob's summary"}
```

---

## 4 · `LastValue` + `LastValueAfterFinish`

**Module**: `langgraph.channels.last_value`  
**First source-verified coverage of the `LastValueAfterFinish` finish-gate lifecycle.**

`LastValue` is the **default channel type** for every plain (non-annotated) field in a `StateGraph`. It stores exactly one value and raises `InvalidUpdateError` if more than one write arrives in the same superstep — enforcing the invariant that non-reducer fields can only be written by one node per step.

`LastValueAfterFinish` wraps a value behind a two-stage gate: `finish()` enables reading; `consume()` resets both the value and the gate. This is the mechanism behind `add_sequence()` — the output of node A must be explicitly "finished" before node B can read it.

```python
# Signatures (langgraph/channels/last_value.py)
class LastValue(BaseChannel[Value, Value, Value]):
    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False                   # no write this superstep — no-op
        if len(values) != 1:
            raise InvalidUpdateError(...)  # INVALID_CONCURRENT_GRAPH_UPDATE
        self.value = values[-1]
        return True

class LastValueAfterFinish(BaseChannel[Value, Value, tuple[Value, bool]]):
    def finish(self) -> bool: ...   # gates is_available() → True
    def consume(self) -> bool: ...  # resets value + finished flag
    def checkpoint(self) -> tuple[Value | Any, bool] | Any: ...
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| `LastValue.update()` | Single-write-per-step guard: `len(values) != 1` raises `InvalidUpdateError` with `ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE` — use `Annotated[T, reducer_fn]` to allow multiple writers |
| `LastValueAfterFinish.finish()` | Sets `self.finished = True` and returns `True` only if `value is not MISSING`; no-op if already finished |
| `LastValueAfterFinish.consume()` | Resets `finished = False` and `value = MISSING`; returns `True` if it actually consumed — LangGraph calls this after a node reads the value |
| Checkpoint tuple | `LastValueAfterFinish.checkpoint()` returns `(value, bool)` or `MISSING` if no value; `from_checkpoint()` unpacks with tuple destructuring |
| `__eq__` | Both return `isinstance(other, SameClass)` — channel topology equality ignores the current value |

### Example 1 — LastValue single-write guard in action

```python
from langgraph.channels.last_value import LastValue
from langgraph.errors import InvalidUpdateError
from langgraph._internal._typing import MISSING


ch: LastValue[int] = LastValue(int, key="counter")
ch = ch.from_checkpoint(MISSING)

# Normal: one write per step
changed = ch.update([42])
print(f"Updated: {changed}, value: {ch.get()}")  # Updated: True, value: 42

# Conflict: two concurrent nodes both try to write — raises
try:
    ch.update([1, 2])  # simulates two parallel nodes writing the same field
except InvalidUpdateError as e:
    print(f"Caught: {e}")
    # Caught: At key 'counter': Can receive only one value per step.
    # Use an Annotated key to handle multiple values.

# No writes: update([]) returns False without modifying value
changed = ch.update([])
print(f"No-op: {changed}, value: {ch.get()}")  # No-op: False, value: 42
```

### Example 2 — LastValueAfterFinish two-stage lifecycle

```python
from langgraph.channels.last_value import LastValueAfterFinish
from langgraph._internal._typing import MISSING
from langgraph.errors import EmptyChannelError


ch: LastValueAfterFinish[str] = LastValueAfterFinish(str, key="pipeline_result")
ch = ch.from_checkpoint(MISSING)

# Stage 1: value written but not yet finished — not available for reading
ch.update(["processed output"])
print(f"Has value: {ch.value is not MISSING}")  # True
try:
    ch.get()   # raises — finish() not yet called
except EmptyChannelError:
    print("Not available yet — finish() not called")

# Stage 2: finish() gates the value as available
finished = ch.finish()
print(f"finish() returned: {finished}")  # True
print(f"Value now readable: {ch.get()}")  # processed output

# Stage 3: consume() resets — value no longer available
consumed = ch.consume()
print(f"consume() returned: {consumed}")  # True
try:
    ch.get()
except EmptyChannelError:
    print("Consumed — back to empty")

# Checkpoint round-trip: (value, bool) tuple preserves gate state
ch.update(["new output"])
ch.finish()
blob = ch.checkpoint()
print(f"Checkpoint: {blob}")  # ('new output', True)

ch2 = LastValueAfterFinish(str, key="pipeline_result")
ch2 = ch2.from_checkpoint(blob)
print(f"Restored value: {ch2.get()}")  # new output
```

### Example 3 — StateGraph using Annotated to allow concurrent writes

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    # LastValue (default) — single writer only
    title: str
    # Annotated with reducer — multiple concurrent writers OK
    messages: Annotated[list, add_messages]


def node_a(state: State) -> dict:
    return {"messages": [AIMessage(content="Hello from A")]}


def node_b(state: State) -> dict:
    # Both A and B can write to messages (reducer merges)
    return {"messages": [AIMessage(content="Hello from B")]}


def set_title(state: State) -> dict:
    return {"title": "My conversation"}


builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("title", set_title)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge(START, "title")
builder.add_edge("a", END)
builder.add_edge("b", END)
builder.add_edge("title", END)

graph = builder.compile()
result = graph.invoke({"title": "", "messages": [HumanMessage(content="Hi")]})
print(result["title"])   # My conversation
print([m.content for m in result["messages"]])
# ['Hi', 'Hello from A', 'Hello from B'] — reducer merged both AI messages
```

---

## 5 · `BranchSpec`

**Module**: `langgraph.graph._branch`  
**First source-verified coverage of the NamedTuple anatomy powering `add_conditional_edges()`.**

`BranchSpec` is a `NamedTuple` that captures everything LangGraph needs to execute a conditional edge: the routing callable (wrapped in a `Runnable`), the optional fixed endpoint map, and an inferred input schema for the branch function. `StateGraph.add_conditional_edges()` creates a `BranchSpec` via `BranchSpec.from_path()` and stores it in the graph topology.

```python
# Signature (langgraph/graph/_branch.py)
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]    # routing callable
    ends: dict[Hashable, str] | None                  # None means open-ended routing
    input_schema: type[Any] | None = None             # inferred from path's first param hint

    @classmethod
    def from_path(
        cls,
        path: Runnable[Any, Hashable | list[Hashable]],
        path_map: dict[Hashable, str] | list[str] | None,
        infer_schema: bool = False,
    ) -> "BranchSpec": ...
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| `ends` | `None` = open routing (return value is the node name); `dict` = maps return values to node names |
| Schema inference | `_get_branch_path_input_schema()` inspects `path`'s first parameter type hint; only types that themselves have type hints (i.e. TypedDict / dataclass) are used — bare `str`, `int`, etc. are ignored |
| `from_path` list coercion | When `path_map` is a `list[str]`, it becomes `{name: name for name in path_map}` — each name maps to itself, so the routing function must return the node name string directly (same as open routing) |
| `path` wrapping | The raw callable is wrapped in `RunnableCallable` so LangGraph can call it with or without a `RunnableConfig` argument |

### Example 1 — Reading BranchSpec from a compiled graph

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph._branch import BranchSpec


class State(TypedDict):
    value: int
    path: str


def router(state: State) -> str:
    return "high" if state["value"] > 10 else "low"


def high_node(state: State) -> dict:
    return {"path": "high"}


def low_node(state: State) -> dict:
    return {"path": "low"}


builder = StateGraph(State)
builder.add_node("high", high_node)
builder.add_node("low", low_node)
builder.add_edge(START, "__router__")
builder.add_node("__router__", lambda s: s)  # placeholder
builder.add_conditional_edges("__router__", router, {"high": "high", "low": "low"})
builder.add_edge("high", END)
builder.add_edge("low", END)

# Inspect the BranchSpec stored on the compiled graph
# (BranchSpec lives in the Pregel branches dict, accessible via the source node)
graph = builder.compile()
print(type(graph))  # <class 'langgraph.pregel.Pregel'> / CompiledStateGraph
```

### Example 2 — Schema inference in BranchSpec.from_path()

```python
from typing_extensions import TypedDict
from langgraph.graph._branch import BranchSpec, _get_branch_path_input_schema
from langchain_core.runnables import RunnableLambda


class MyState(TypedDict):
    score: float


def typed_router(state: MyState) -> str:
    """Router with a typed first parameter — schema will be inferred."""
    return "a" if state["score"] > 0.5 else "b"


# Schema inference: first parameter is MyState (a TypedDict with hints)
inferred = _get_branch_path_input_schema(typed_router)
print(f"Inferred schema: {inferred}")  # <class '__main__.MyState'>

# Untyped router — no schema inferred
def untyped_router(state) -> str:
    return "a"

inferred2 = _get_branch_path_input_schema(untyped_router)
print(f"No schema: {inferred2}")  # None

# BranchSpec.from_path with dict path_map
spec = BranchSpec.from_path(
    RunnableLambda(typed_router),
    path_map={"a": "node_a", "b": "node_b"},
    infer_schema=True,
)
print(f"ends: {spec.ends}")           # {'a': 'node_a', 'b': 'node_b'}
print(f"input_schema: {spec.input_schema}")  # <class '__main__.MyState'>
```

### Example 3 — Open-ended routing vs. fixed-map routing

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    next: str
    result: str


def dynamic_router(state: State) -> str:
    # Return value IS the node name — open-ended routing (ends=None)
    return state["next"]


def node_alpha(state: State) -> dict:
    return {"result": "alpha ran"}


def node_beta(state: State) -> dict:
    return {"result": "beta ran"}


builder = StateGraph(State)
builder.add_node("alpha", node_alpha)
builder.add_node("beta", node_beta)
# No path_map → open routing, return value must match an existing node name
builder.add_conditional_edges(START, dynamic_router)
builder.add_edge("alpha", END)
builder.add_edge("beta", END)

graph = builder.compile()

r1 = graph.invoke({"next": "alpha", "result": ""})
print(r1["result"])   # alpha ran

r2 = graph.invoke({"next": "beta", "result": ""})
print(r2["result"])   # beta ran
```

---

## 6 · `JsonPlusSerializer`

**Module**: `langgraph.checkpoint.serde.jsonplus`  
**Source-verified coverage of ormsgpack format, strict mode, and allowlist management.**

`JsonPlusSerializer` is LangGraph's default checkpoint serializer. Despite its name it actually uses **ormsgpack** (MessagePack) as the primary format, with JSON as a legacy fallback for old checkpoints. The `pickle_fallback` flag extends coverage to arbitrary Python objects at the cost of the pickle security surface.

```python
# Signature (langgraph/checkpoint/serde/jsonplus.py)
class JsonPlusSerializer(SerializerProtocol):
    def __init__(
        self,
        *,
        pickle_fallback: bool = False,
        allowed_msgpack_modules: AllowedMsgpackModules | Literal[True] | None = _SENTINEL,
    ) -> None: ...

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...
    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...

    def with_msgpack_allowlist(
        self, extra_allowlist: Iterable[tuple[str, ...] | type]
    ) -> "JsonPlusSerializer": ...
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| Primary format | ormsgpack with a custom `default` hook that handles LangChain objects, `_DeltaSnapshot`, datetimes, UUIDs, enums, dataclasses, and more |
| `pickle_fallback` | `False` by default — enabling it serializes any Python object via `pickle` as a last resort; security risk if checkpoint storage is untrusted |
| `LANGGRAPH_STRICT_MSGPACK=true` | Sets `allowed_msgpack_modules=None` — deserialization is restricted to a built-in allowlist of safe types; unknown types raise `InvalidModuleError` |
| `allowed_msgpack_modules` sentinel | Default `_SENTINEL` → reads `LANGGRAPH_STRICT_MSGPACK`; `True` → unrestricted; `None` → strict; explicit set/tuple → named-module allowlist |
| `_is_safe_json_type()` | Checks if an `lc=2` id refers to a type in `SAFE_MSGPACK_TYPES` — safe types bypass the `allowed_json_modules` gate, enabling resume of old JSON-format checkpoints without allowlist configuration |
| `_warn_once()` | Circuit breaker: caps at `_MAX_WARNED_TYPES=1000` logged types; racing threads may each emit one warning for the same key (best-effort dedup) |

### Example 1 — Basic serialization round-trip and type tagging

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from datetime import datetime, timezone
import uuid

serde = JsonPlusSerializer()

# Python primitives
enc, data = serde.dumps_typed({"answer": 42, "flag": True})
print(f"type tag: {enc!r}")   # 'msgpack' (ormsgpack format)
value = serde.loads_typed((enc, data))
print(value)  # {'answer': 42, 'flag': True}

# datetime and UUID — handled natively by the ext hook
now = datetime.now(timezone.utc)
uid = uuid.uuid4()
_, dt_bytes = serde.dumps_typed(now)
_, uid_bytes = serde.dumps_typed(uid)
print(serde.loads_typed(("msgpack", dt_bytes)) == now)   # True
print(serde.loads_typed(("msgpack", uid_bytes)) == uid)  # True

# LangChain message objects
from langchain_core.messages import HumanMessage
msg = HumanMessage(content="Hello")
enc2, msg_bytes = serde.dumps_typed(msg)
restored = serde.loads_typed((enc2, msg_bytes))
print(type(restored).__name__, restored.content)  # HumanMessage Hello
```

### Example 2 — LANGGRAPH_STRICT_MSGPACK allowlist mode

In strict mode, only types in the built-in allowlist can be deserialized. Unknown types raise `InvalidModuleError`. Use `with_msgpack_allowlist([MyClass])` to extend the allowlist — accepts type objects or `(module, qualname)` tuples — without modifying the environment.

```python
import os
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.base import SerializerProtocol


# Strict serializer: allowed_msgpack_modules=None
strict_serde = JsonPlusSerializer(allowed_msgpack_modules=None)

# Safe types (LangChain builtins, primitives) pass through
from langchain_core.messages import AIMessage
enc, data = strict_serde.dumps_typed(AIMessage(content="OK"))
restored = strict_serde.loads_typed((enc, data))
print(restored.content)  # OK

# Custom class: not in allowlist — will be blocked on deserialization
import dataclasses

@dataclasses.dataclass
class MyResult:
    score: float

result = MyResult(score=0.92)

# Serialization succeeds (extension hook handles dataclasses)
permissive = JsonPlusSerializer()
enc3, data3 = permissive.dumps_typed(result)

# Deserialization with strict serde raises InvalidModuleError
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer as JPS
strict2 = JPS(allowed_msgpack_modules=None)
try:
    strict2.loads_typed((enc3, data3))
except Exception as e:
    print(f"Blocked: {type(e).__name__}: {e}")
    # Blocked: InvalidModuleError: ...

# Extend allowlist to include MyResult; with_msgpack_allowlist accepts type objects
# or (module, qualname) tuples — passing the class directly is simplest.
extended = strict_serde.with_msgpack_allowlist([MyResult])
restored2 = extended.loads_typed((enc3, data3))
print(f"Allowed: score={restored2.score}")  # Allowed: score=0.92
```

### Example 3 — Injecting JsonPlusSerializer into a custom checkpoint saver

```python
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# JsonPlusSerializer is the default serde used by all built-in savers.
# You can verify or replace it by inspecting .serde on any saver.
saver = MemorySaver()
print(f"Saver serde: {type(saver.serde).__name__}")  # JsonPlusSerializer

# Swap in a strict version (allowlist=None) for the saver
strict_serde = JsonPlusSerializer(allowed_msgpack_modules=None)
saver.serde = strict_serde  # hot-swap the serializer

class State(TypedDict):
    value: str

builder = StateGraph(State)
builder.add_node("n", lambda s: {"value": s["value"].upper()})
builder.add_edge(START, "n")
builder.add_edge("n", END)
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "strict-1"}}
result = graph.invoke({"value": "hello"}, config)
print(result["value"])  # HELLO

# History is checkpointed with the strict serde — only safe types survive
history = list(graph.get_state_history(config))
print(f"Checkpoints: {len(history)}")  # > 0
```

---

## 7 · `ToolCallStream` + `ToolCallTransformer`

**Module**: `langgraph.prebuilt._tool_call_stream` + `langgraph.prebuilt._tool_call_transformer`  
**First source-verified coverage of the streaming handle lifecycle and pump-delegation internals.**

`ToolCallStream` is a scoped handle for a single tool call's streaming execution, produced by `ToolCallTransformer` as `tool-started` / `tool-output-delta` / `tool-finished` / `tool-error` events flow through the `tools` channel. `ToolCallTransformer` is a native `StreamTransformer` that projects those raw protocol events into per-call `ToolCallStream` handles available on `run.tool_calls`.

```python
# Signatures (langgraph/prebuilt/_tool_call_stream.py + _tool_call_transformer.py)
class ToolCallStream:
    tool_call_id: str
    tool_name: str
    input: dict[str, Any] | None
    output: Any          # set by _finish() on tool-finished
    error: str | None    # set by _fail() on tool-error
    completed: bool      # True after terminal event

    @property
    def output_deltas(self) -> StreamChannel[Any]: ...   # iterable delta chunks

class ToolCallTransformer(StreamTransformer):
    _native = True
    required_stream_modes = ("tools",)

    def process(self, event: ProtocolEvent) -> bool: ...   # routes events; always returns True
    def finalize(self) -> None: ...   # closes all active streams at run end
    def fail(self, err: BaseException) -> None: ...   # fails all active streams on error
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| `_native = True` | The `tool_calls` projection is exposed as a direct attribute on the run stream object, not injected into the main event log |
| `required_stream_modes` | Forces `stream_mode="tools"` to be active — the transformer raises if this mode is not included |
| `process()` always returns `True` | Events pass through to the main event log untouched; consumers can subscribe to the `tools` channel directly to receive raw events alongside the handle projection |
| `_bind_pump` / `_bind_apump` | The `StreamMux` calls these to wire sync / async pull callbacks onto each `ToolCallStream`, enabling non-consuming `wait()`-style iteration on `output_deltas` |
| `finalize()` | Called at clean run end; closes any `ToolCallStream` that never received a terminal event (e.g. if a tool call was dropped) |
| `scope` | Namespace tuple — only events whose `params.namespace` matches `self.scope` are projected; subgraph events are routed to the child mini-mux |

### Example 1 — Streaming tool output deltas

```python
import asyncio
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langchain_core.messages import HumanMessage


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


class State(TypedDict):
    messages: list


def call_model(state: State) -> dict:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools([search])
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode([search]))
builder.add_conditional_edges("model", lambda s: "tools" if s["messages"][-1].tool_calls else END)
builder.add_edge("tools", END)
builder.add_edge(START, "model")

# Register ToolCallTransformer — enables run.tool_calls projection
graph = builder.compile(transformers=[ToolCallTransformer])

async def main():
    # astream_events(version="v3") returns AsyncGraphRunStream — await it, then use as context manager
    run = await graph.astream_events(
        {"messages": [HumanMessage(content="Search for LangGraph")]},
        version="v3",
    )
    async with run:
        # run.tool_calls is exposed as a direct attribute because ToolCallTransformer._native = True
        async for tool_stream in run.tool_calls:
            print(f"Tool started: {tool_stream.tool_name!r} id={tool_stream.tool_call_id[:8]}")
            async for delta in tool_stream.output_deltas:
                print(f"  delta: {delta!r}")
            print(f"  final output: {tool_stream.output!r}")
            print(f"  completed: {tool_stream.completed}")

asyncio.run(main())
```

### Example 2 — Inspecting ToolCallStream fields after completion

```python
from langgraph.prebuilt._tool_call_stream import ToolCallStream
from langgraph.stream.stream_channel import StreamChannel

# Construct a ToolCallStream directly for unit testing
stream = ToolCallStream(
    tool_call_id="call-abc123",
    tool_name="weather_lookup",
    input={"city": "London"},
)

print(f"id: {stream.tool_call_id}")   # call-abc123
print(f"name: {stream.tool_name}")    # weather_lookup
print(f"input: {stream.input}")       # {'city': 'London'}
print(f"completed: {stream.completed}")  # False
print(f"output: {stream.output}")     # None (not yet finished)

# Simulate terminal event (normally done by ToolCallTransformer)
stream._finish({"temp": 18, "conditions": "Cloudy"})
print(f"completed: {stream.completed}")   # True
print(f"output: {stream.output}")         # {'temp': 18, 'conditions': 'Cloudy'}

# Simulate error path
stream2 = ToolCallStream("call-xyz", "flaky_tool", input={})
stream2._fail("Connection timeout after 30s")
print(f"error: {stream2.error}")        # Connection timeout after 30s
print(f"completed: {stream2.completed}")  # True
print(f"output: {stream2.output}")      # None
```

### Example 3 — ToolCallTransformer scope and subgraph isolation

Each `ToolCallTransformer` instance receives a `scope` tuple that matches its enclosing graph's namespace. Events from subgraphs have a longer namespace and are routed to child transformers — the parent sees only its own tool calls.

```python
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer

# Root-level transformer: scope=() matches only root-level tool events
root_transformer = ToolCallTransformer(scope=())

# Subgraph transformer: scope=("subgraph@run1",) matches only that subgraph
subgraph_transformer = ToolCallTransformer(scope=("subgraph@run1",))

# Simulate a root tool event
root_event = {
    "method": "tools",
    "params": {
        "namespace": [],   # root-level: empty list → () tuple
        "data": {
            "event": "tool-started",
            "tool_call_id": "call-root-1",
            "tool_name": "root_search",
            "input": {"q": "test"},
        },
    },
    "id": "evt-1",
    "timestamp": 0,
    "seq": 0,
}

# Root transformer processes it; subgraph transformer ignores it
pass_through = root_transformer.process(root_event)
print(f"Root processes root event: {pass_through}")  # True (pass-through)
print(f"Root active streams: {list(root_transformer._active.keys())}")   # ['call-root-1']

pass_through2 = subgraph_transformer.process(root_event)
print(f"Subgraph ignores root event: {pass_through2}")  # True (namespace mismatch)
print(f"Subgraph active streams: {list(subgraph_transformer._active.keys())}")  # []
```

---

## 8 · `AnyValue`

**Module**: `langgraph.channels.any_value`  
**Source-verified coverage of last-write-wins semantics and MISSING clear-on-empty.**

`AnyValue` is the "last value received, assuming all concurrent writes are equal" channel. Unlike `LastValue`, it does **not** raise if multiple values arrive in the same superstep — it silently takes the last one. This makes it suitable for fan-in scenarios where all parallel paths are guaranteed to produce the same value (e.g. a routing signal derived independently from shared input).

```python
# Signature (langgraph/channels/any_value.py)
class AnyValue(BaseChannel[Value, Value, Value]):
    """Stores the last value received, assumes that if multiple values are
    received, they are all equal."""

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            if self.value is not MISSING:
                self.value = MISSING    # explicit clear on empty update
                return True
            return False               # already empty — no state change
        self.value = values[-1]        # always takes the last write
        return True

    def __eq__(self, value: object) -> bool:
        return isinstance(value, AnyValue)  # any AnyValue equals any other
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| No guard | Multiple writers in the same superstep are silently accepted — use only when you can guarantee convergence |
| `update([])` clears | An empty write list actively clears to `MISSING`, triggering `return True` (state changed) — contrast with `LastValue` which returns `False` on empty |
| `__eq__` | Matches all `AnyValue` instances regardless of type parameter — channel topology equality is by class, not by type |
| Clearing semantics | Useful for ephemeral signals: write a value, read it once, then a no-write superstep clears it automatically |

### Example 1 — AnyValue in a parallel fan-in graph

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.any_value import AnyValue


class State(TypedDict):
    input: str
    # All parallel paths compute the same category — AnyValue accepts any writer
    category: Annotated[str, AnyValue(str)]
    # operator.add reducer merges parallel list writes (required for parallel nodes)
    results: Annotated[list[str], operator.add]


def classifier_a(state: State) -> dict:
    # Both classifiers independently derive "technical" from the same input
    cat = "technical" if "code" in state["input"] else "general"
    return {"category": cat, "results": ["A: " + cat]}


def classifier_b(state: State) -> dict:
    cat = "technical" if "code" in state["input"] else "general"
    return {"category": cat, "results": ["B: " + cat]}


def merge(state: State) -> dict:
    return {"results": state["results"]}


builder = StateGraph(State)
builder.add_node("a", classifier_a)
builder.add_node("b", classifier_b)
builder.add_node("merge", merge)
# Both A and B run in parallel and both write to `category`
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", "merge")
builder.add_edge("b", "merge")
builder.add_edge("merge", END)
graph = builder.compile()

result = graph.invoke({"input": "write code for me", "category": "", "results": []})
print(f"category: {result['category']}")   # technical
print(f"results: {result['results']}")     # ['A: technical', 'B: technical']
```

### Example 2 — Comparing AnyValue and LastValue update semantics

```python
from langgraph.channels.any_value import AnyValue
from langgraph.channels.last_value import LastValue
from langgraph.errors import InvalidUpdateError
from langgraph._internal._typing import MISSING


# AnyValue: accepts 0, 1, or N writes — always takes last
av: AnyValue[int] = AnyValue(int, key="signal")
av = av.from_checkpoint(MISSING)

av.update([100])         # one write — fine
print(av.get())          # 100

av.update([200, 300])    # two concurrent writes — takes 300 (last)
print(av.get())          # 300

av.update([])            # empty update — clears to MISSING
print(av.value is MISSING)  # True

# LastValue: raises on >1 write in same superstep
lv: LastValue[int] = LastValue(int, key="counter")
lv = lv.from_checkpoint(MISSING)

lv.update([42])  # fine
try:
    lv.update([1, 2])   # two concurrent writes — raises
except InvalidUpdateError as e:
    print(f"LastValue guard: {e}")

lv.update([])  # empty update — LastValue does NOT clear; returns False
print(lv.value)  # 42 (unchanged)
print(lv.value is MISSING)  # False
```

### Example 3 — Custom AnyValue for convergent routing signals

```python
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.any_value import AnyValue


Destination = Literal["db", "cache", "skip"]


class State(TypedDict):
    payload: dict
    # Convergent routing signal: all analyzers should agree
    destination: Annotated[Destination, AnyValue(str)]
    processed: bool


def security_check(state: State) -> dict:
    # Blocked payloads → skip; safe payloads → cache (matches size_check for small inputs)
    dest: Destination = "skip" if "malicious" in str(state["payload"]) else "cache"
    return {"destination": dest}


def size_check(state: State) -> dict:
    dest: Destination = "cache" if len(str(state["payload"])) < 50 else "db"
    return {"destination": dest}


def route(state: State) -> Destination:
    return state["destination"]


def write_db(state: State) -> dict:
    print(f"Writing to DB: {state['payload']}")
    return {"processed": True}


def write_cache(state: State) -> dict:
    print(f"Writing to cache: {state['payload']}")
    return {"processed": True}


def skip_node(state: State) -> dict:
    print("Skipping malicious payload")
    return {"processed": False}


builder = StateGraph(State)
builder.add_node("security", security_check)
builder.add_node("size", size_check)
builder.add_node("db", write_db)
builder.add_node("cache", write_cache)
builder.add_node("skip", skip_node)

builder.add_edge(START, "security")
builder.add_edge(START, "size")
builder.add_conditional_edges("security", route)
builder.add_conditional_edges("size", route)
builder.add_edge("db", END)
builder.add_edge("cache", END)
builder.add_edge("skip", END)
graph = builder.compile()

# Both analyzers agree: small non-malicious payload → "cache"
result = graph.invoke({"payload": {"key": "small"}, "destination": "cache", "processed": False})
print(f"Destination: {result['destination']}")   # cache — both writers agreed
print(f"Processed: {result['processed']}")        # True
```

---

## 9 · `NamedBarrierValue` + `NamedBarrierValueAfterFinish`

**Module**: `langgraph.channels.named_barrier_value`  
**Source-verified coverage of the barrier fan-in mechanism and two-stage finish gate.**

`NamedBarrierValue` is a fan-in synchronization primitive: it tracks which of a fixed set of named signals have arrived, and only becomes available when the full set has been seen. It is consumed (reset) after each read. `NamedBarrierValueAfterFinish` extends this with a `finish()`-gate: the value only becomes readable after an explicit `finish()` call, even if all names have been seen — used for output channels in sequences and pipelines.

```python
# Signatures (langgraph/channels/named_barrier_value.py)
class NamedBarrierValue(BaseChannel[Value, Value, set[Value]]):
    """Waits until all named values are received before making the value available."""
    names: set[Value]    # expected signals
    seen: set[Value]     # received so far

    def update(self, values: Sequence[Value]) -> bool:
        """Add each value to `seen`; raises if value not in `names`."""
    def get(self) -> Value:
        """Available only when seen == names; returns None."""
    def consume(self) -> bool:
        """Resets `seen` back to empty set; returns True if it was full."""

class NamedBarrierValueAfterFinish(BaseChannel):
    """Additional finish() gate: only made available after finish() is called."""
    def finish(self) -> bool:
        """Gate the barrier; returns True only when seen == names and not yet finished."""
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| `get()` return value | Returns `None` — the value itself is irrelevant; the signal is availability (all names seen) |
| `update()` validation | Any value not in `names` raises `InvalidUpdateError` — prevents mis-named signals |
| `consume()` reset | After the graph reads the available value, `consume()` resets `seen = set()` so the barrier can be triggered again in a later step |
| Checkpoint | `NamedBarrierValue.checkpoint()` serializes `seen` (a `set`); `NamedBarrierValueAfterFinish.checkpoint()` serializes `(seen, bool)` |
| `NamedBarrierValueAfterFinish.finish()` | Must be called explicitly — see `add_sequence()` internals which call this on the trigger channel after all nodes in the sequence write |

### Example 1 — NamedBarrierValue as a fan-in gate

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.errors import EmptyChannelError, InvalidUpdateError
from langgraph._internal._typing import MISSING


# All three workers must signal before the barrier opens
barrier: NamedBarrierValue[str] = NamedBarrierValue(str, names={"worker_a", "worker_b", "worker_c"})
barrier.key = "all_done"
barrier = barrier.from_checkpoint(MISSING)

print(f"Available: {barrier.is_available()}")  # False

# Workers signal one by one
barrier.update(["worker_a"])
print(f"Seen: {barrier.seen}")  # {'worker_a'}

barrier.update(["worker_b"])
print(f"Seen: {barrier.seen}")  # {'worker_a', 'worker_b'}

try:
    barrier.get()   # not yet — worker_c hasn't signalled
except EmptyChannelError:
    print("Still waiting for worker_c")

# Wrong name raises
try:
    barrier.update(["worker_d"])
except InvalidUpdateError as e:
    print(f"Invalid: {e}")

# Final signal
barrier.update(["worker_c"])
print(f"Available: {barrier.is_available()}")  # True
val = barrier.get()
print(f"Gate value: {val}")  # None — availability is the signal, not the value

# consume() resets for the next cycle
reset = barrier.consume()
print(f"Reset: {reset}, seen: {barrier.seen}")  # Reset: True, seen: set()
```

### Example 2 — NamedBarrierValue in a StateGraph fan-in

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.named_barrier_value import NamedBarrierValue


class State(TypedDict):
    data: str
    # Barrier: both "fetch" and "validate" must run before "process" can start
    ready: Annotated[None, NamedBarrierValue(str, names={"fetch", "validate"})]
    result: str


def fetch(state: State) -> dict:
    print("Fetching data...")
    # Signal the barrier with this node's name
    return {"ready": "fetch"}


def validate(state: State) -> dict:
    print("Validating data...")
    return {"ready": "validate"}


def process(state: State) -> dict:
    print("Processing — all prerequisites done!")
    return {"result": f"Processed: {state['data']}"}


builder = StateGraph(State)
builder.add_node("fetch", fetch)
builder.add_node("validate", validate)
builder.add_node("process", process)
# fetch and validate run in parallel; list edge waits for BOTH before scheduling process
builder.add_edge(START, "fetch")
builder.add_edge(START, "validate")
builder.add_edge(["fetch", "validate"], "process")
builder.add_edge("process", END)
graph = builder.compile()

# Do not seed "ready" — NamedBarrierValue validates writes against names={"fetch","validate"}
# and raises InvalidUpdateError on any other value (including None).
result = graph.invoke({"data": "my-data", "result": ""})
print(result["result"])   # Processed: my-data
```

### Example 3 — NamedBarrierValueAfterFinish checkpoint round-trip

```python
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish
from langgraph.errors import EmptyChannelError
from langgraph._internal._typing import MISSING


ch = NamedBarrierValueAfterFinish(str, names={"a", "b"})
ch.key = "gate"
ch = ch.from_checkpoint(MISSING)

# Both signals arrive
ch.update(["a"])
ch.update(["b"])

# Still not available — finish() not called
print(f"is_available: {ch.is_available()}")  # False

# finish() gates availability
did_finish = ch.finish()
print(f"finish(): {did_finish}")  # True
print(f"is_available: {ch.is_available()}")  # True

# Checkpoint serializes (seen, finished_bool)
blob = ch.checkpoint()
print(f"checkpoint blob: {blob}")  # ({'a', 'b'}, True)

# Restore from checkpoint
ch2 = NamedBarrierValueAfterFinish(str, names={"a", "b"})
ch2 = ch2.from_checkpoint(blob)
print(f"Restored is_available: {ch2.is_available()}")  # True

# consume() resets both seen and finished
ch2.consume()
print(f"After consume: {ch2.seen}, finished={ch2.finished}")  # set(), False
```

---

## 10 · `EphemeralValue`

**Module**: `langgraph.channels.ephemeral_value`  
**Source-verified coverage of guard mode, single-step clearing, and trigger-channel patterns.**

`EphemeralValue` stores the value received in the **immediately preceding** step and then clears itself when no write arrives in the current step. This makes it ideal for "trigger" signals — pass a value once, the next node reads it, and it's gone. The `guard` parameter (default `True`) raises `InvalidUpdateError` if more than one value arrives in the same step — useful when only one node should trigger the channel at a time.

```python
# Signature (langgraph/channels/ephemeral_value.py)
class EphemeralValue(BaseChannel[Value, Value, Value]):
    """Stores the value received in the step immediately preceding, clears after."""

    __slots__ = ("value", "guard")

    def __init__(self, typ: Any, guard: bool = True) -> None: ...

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            if self.value is not MISSING:
                self.value = MISSING   # clear when no write this step
                return True            # state changed
            return False

        if len(values) != 1 and self.guard:
            raise InvalidUpdateError(
                f"EphemeralValue(guard=True) can receive only one value per step. "
                "Use guard=False if you want to store any one of multiple values."
            )
        self.value = values[-1]        # always takes the last write
        return True
```

Key design decisions:

| Aspect | Detail |
|--------|--------|
| Auto-clear | When `update([])` is called (no write this step), the value is cleared to `MISSING` — contrasts with `LastValue` which does not clear on empty update |
| `guard=True` | Raises if >1 value arrives per step — safe default for trigger channels where exactly one node should write |
| `guard=False` | Silently accepts multiple writes and takes the last — use when multiple parallel branches may trigger the channel |
| Checkpoint | The value IS checkpointed (unlike `UntrackedValue`) — allows resume after interrupts mid-step |
| `is_available()` | Returns `True` only while the value is not `MISSING` — becomes `False` again in the next step if no write arrives |

### Example 1 — Single-step trigger channel

```python
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph._internal._typing import MISSING
from langgraph.errors import EmptyChannelError


ch: EphemeralValue[str] = EphemeralValue(str, guard=True)
ch.key = "trigger"
ch = ch.from_checkpoint(MISSING)

# Step 1: write arrives → value stored
ch.update(["start_pipeline"])
print(f"Step 1 value: {ch.get()}")  # start_pipeline

# Step 2: no write → value clears
changed = ch.update([])
print(f"Changed: {changed}")  # True (cleared)
try:
    ch.get()
except EmptyChannelError:
    print("Step 2: no value (cleared)")

# Step 3: no write again → already empty → no change
changed2 = ch.update([])
print(f"Changed again: {changed2}")  # False (already MISSING)
```

### Example 2 — guard=True vs guard=False in parallel fan-in

```python
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.errors import InvalidUpdateError
from langgraph._internal._typing import MISSING


# guard=True: only one writer allowed per step
strict = EphemeralValue(str, guard=True)
strict.key = "strict_trigger"
strict = strict.from_checkpoint(MISSING)

strict.update(["node_a"])  # fine
try:
    # Simulate two parallel nodes writing in the same superstep
    strict.update(["node_a", "node_b"])
except InvalidUpdateError as e:
    print(f"guard=True raised: {e}")

# guard=False: last writer wins — no exception
lenient = EphemeralValue(str, guard=False)
lenient.key = "lenient_trigger"
lenient = lenient.from_checkpoint(MISSING)
lenient.update(["node_a", "node_b"])   # no error
print(f"guard=False last value: {lenient.get()}")  # node_b
```

### Example 3 — EphemeralValue as a one-shot activation signal in a loop

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.ephemeral_value import EphemeralValue


class State(TypedDict):
    messages: list[str]
    # Ephemeral signal: True when a new user message arrives; auto-clears after
    new_message: Annotated[bool | None, EphemeralValue(bool)]
    summary: str


def user_input(state: State) -> dict:
    return {
        "messages": state["messages"] + ["User: hello"],
        "new_message": True,   # signal fires once
    }


def check_new_message(state: State) -> str:
    # Route based on whether new_message signal is present
    return "summarize" if state["new_message"] else END


def summarize(state: State) -> dict:
    summary = f"Summary of {len(state['messages'])} messages"
    print(f"Summarizing: {summary}")
    return {"summary": summary}
    # No write to new_message → it auto-clears next step


builder = StateGraph(State)
builder.add_node("input", user_input)
builder.add_node("summarize", summarize)
builder.add_conditional_edges("input", check_new_message)
builder.add_edge(START, "input")
builder.add_edge("summarize", END)
graph = builder.compile()

result = graph.invoke({"messages": [], "new_message": None, "summary": ""})
print(f"Summary: {result['summary']}")
# Summarizing: Summary of 1 messages
# Summary: Summary of 1 messages
```

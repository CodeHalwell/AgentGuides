---
title: "LangGraph Class Deep-Dives Vol. 33"
description: "Source-verified deep dives into 10 class groups in langgraph==1.2.7 — InMemoryCache/BaseCache namespace-TTL caching, EphemeralValue one-step channel, DeltaChannel (beta) replay-based accumulation, NamedBarrierValue/NamedBarrierValueAfterFinish fan-in barriers, RunControl/GraphDrained cooperative drain, entrypoint.final decoupled return/save, ValuesTransformer/UpdatesTransformer/CustomTransformer v3 native stream transformers, MessagesTransformer ChatModelStream lifecycle, AnyValue relaxed last-write channel, and ExecutionInfo/ServerInfo/get_runtime execution metadata — expanded and updated coverage verified against langgraph==1.2.7."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 33"
  order: 64
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, updated and expanded for `langgraph==1.2.7` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `InMemoryCache` + `BaseCache`

**Modules:** `langgraph.cache.memory` / `langgraph.cache.base`

`BaseCache` is the abstract base for all node-result caches. `InMemoryCache` is the thread-safe in-process implementation that ships with LangGraph. Both are parameterised by value type `ValueT`.

**Key source facts** (from `langgraph/cache/memory/__init__.py` and `langgraph/cache/base/__init__.py`):

- Keys are typed as `FullKey = tuple[Namespace, str]` where `Namespace = tuple[str, ...]`. Every cache call is namespaced, so different tasks never collide.
- `get(keys)` / `aget(keys)` accept a sequence of `FullKey`s and return only the hits as `dict[FullKey, ValueT]`.
- `set(pairs)` / `aset(pairs)` accept `Mapping[FullKey, tuple[ValueT, int | None]]`. The `int` is TTL in seconds; `None` means no expiry.
- `clear(namespaces)` / `aclear(namespaces)` delete one or more namespaces. Passing `None` clears all data.
- Serialisation goes through `serde: SerializerProtocol` (default `JsonPlusSerializer`). You can swap to a custom serde at construction time.
- `InMemoryCache` protects its internal `dict` with a `threading.RLock`; the async methods are synchronous wrappers — safe for async code but not non-blocking.
- Wire a cache into a `StateGraph` via `compile(cache=cache)` or into `@entrypoint(cache=cache)`.
- Per-node opt-in uses `add_node(..., cache_policy=CachePolicy(ttl=60))`.

### Example 1 — standalone cache read/write/expiry

```python
import time
from langgraph.cache.memory import InMemoryCache

cache: InMemoryCache[dict] = InMemoryCache()

ns = ("pipeline", "step1")        # namespace tuple
key = "run_42"

# store with 2-second TTL
cache.set({((ns, key)): ({"result": 99}, 2)})

# immediate read — hits
hits = cache.get([(ns, key)])
print(hits)  # {(('pipeline', 'step1'), 'run_42'): {'result': 99}}

# after TTL expires — misses
time.sleep(3)
hits_expired = cache.get([(ns, key)])
print(hits_expired)  # {}
```

### Example 2 — wiring a cache into a `StateGraph`

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

call_count = 0

class State(TypedDict):
    x: int
    doubled: int

def expensive_double(state: State) -> State:
    global call_count
    call_count += 1
    return {"doubled": state["x"] * 2}

cache = InMemoryCache()

graph = (
    StateGraph(State)
    .add_node(
        "double",
        expensive_double,
        cache_policy=CachePolicy(ttl=60),     # cache results for 60 s
    )
    .add_edge(START, "double")
    .add_edge("double", END)
    .compile(cache=cache)
)

r1 = graph.invoke({"x": 5, "doubled": 0})
r2 = graph.invoke({"x": 5, "doubled": 0})   # cache hit — expensive_double not called again
r3 = graph.invoke({"x": 9, "doubled": 0})   # different input — cache miss

print(r1["doubled"], r2["doubled"], r3["doubled"])  # 10 10 18
print("actual calls:", call_count)                  # 2  (x=5 and x=9)
```

### Example 3 — namespace-scoped `clear()` and custom serde

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

cache: InMemoryCache[list] = InMemoryCache(serde=JsonPlusSerializer())

ns_a = ("agent", "tool_a")
ns_b = ("agent", "tool_b")

cache.set({
    (ns_a, "k1"): ([1, 2, 3], None),
    (ns_b, "k2"): ([4, 5, 6], None),
})
print(len(cache.get([(ns_a, "k1"), (ns_b, "k2")])))  # 2

# clear only namespace A
cache.clear([ns_a])
print(len(cache.get([(ns_a, "k1"), (ns_b, "k2")])))  # 1 (ns_b still present)

# clear everything
cache.clear()
print(len(cache.get([(ns_b, "k2")])))  # 0
```

---

## 2 · `EphemeralValue`

**Module:** `langgraph.channels.ephemeral_value`

`EphemeralValue` stores a value written during the current superstep and clears it when the superstep ends. It is the channel type used by the `START` channel in the functional API and in the internal `Pregel` machinery for one-shot inputs.

**Key source facts** (from `langgraph/channels/ephemeral_value.py`):

- Unlike `LastValue`, which retains its value across supersteps, `EphemeralValue.update([])` (empty write) sets `value = MISSING`, making the channel unavailable in the next step.
- `guard=True` (default) raises `InvalidUpdateError` if two values are written in the same superstep. Set `guard=False` to silently take the last of concurrent writes.
- `checkpoint()` returns the raw value (including `MISSING`); the channel is reconstituted from checkpoint via `from_checkpoint()`.
- `is_available()` returns `False` when `value is MISSING`, keeping downstream nodes from triggering prematurely.
- The channel is useful for "one-shot triggers" — you write once, read once, and the channel auto-clears before the next superstep.

### Example 1 — build a custom one-shot trigger channel

```python
from langgraph.channels.ephemeral_value import EphemeralValue

ch: EphemeralValue[str] = EphemeralValue(str)

# write a value
changed = ch.update(["hello"])
print(changed, ch.get())   # True hello

# simulate end of superstep: write nothing → clears
changed = ch.update([])
print(changed)             # True (value changed from "hello" to MISSING)
print(ch.is_available())   # False
```

### Example 2 — concurrent write guard

```python
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.errors import InvalidUpdateError

ch_guarded = EphemeralValue(int, guard=True)
ch_relaxed = EphemeralValue(int, guard=False)

# guard=True: two concurrent writes raises
try:
    ch_guarded.update([1, 2])
except InvalidUpdateError as e:
    print("guarded raised:", e)

# guard=False: silently takes the last write
ch_relaxed.update([1, 2])
print("relaxed last-write:", ch_relaxed.get())  # 2
```

### Example 3 — `EphemeralValue` directly inside a `StateGraph`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.ephemeral_value import EphemeralValue

# StateGraph's channel resolver checks for BaseChannel instances in
# Annotated metadata before falling back to LastValue, so you can
# use EphemeralValue directly — no manual Pregel construction needed.

class State(TypedDict):
    trigger: Annotated[str | None, EphemeralValue(str)]   # clears each superstep
    result: str

def handle_trigger(state: State) -> State:
    trigger = state.get("trigger")
    if trigger:
        return {"result": f"handled: {trigger}"}   # trigger auto-clears; no need to set None
    return {"result": "no trigger"}

graph = (
    StateGraph(State)
    .add_node("handle_trigger", handle_trigger)
    .add_edge(START, "handle_trigger")
    .add_edge("handle_trigger", END)
    .compile()
)

print(graph.invoke({"trigger": "fire", "result": ""}))
# {'trigger': None, 'result': 'handled: fire'}
print(graph.invoke({"trigger": None, "result": ""}))
# {'trigger': None, 'result': 'no trigger'}
```

---

## 3 · `DeltaChannel` (beta)

**Module:** `langgraph.channels.delta`

`DeltaChannel` is a beta channel that reconstructs state by **replaying ancestor writes** through a reducer instead of storing full snapshots at every step. This cuts checkpoint storage dramatically for long-running, append-heavy channels (e.g., message lists on threads with thousands of turns).

**Key source facts** (from `langgraph/channels/delta.py`):

- Constructor: `DeltaChannel(reducer, typ=None, *, snapshot_frequency=1000)`. `reducer(state, list[writes]) -> new_state` must be deterministic and **batching-invariant**: `reducer(reducer(s, xs), ys) == reducer(s, xs + ys)`.
- `snapshot_frequency` controls how often a full `_DeltaSnapshot(value)` blob is written (every N updates, or every `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` supersteps, whichever comes first).
- `checkpoint()` always returns `MISSING`; snapshot blobs are written directly into `channel_values` by `create_checkpoint`. Older checkpointers that do not call `get_delta_channel_history` will not reconstruct correctly.
- `replay_writes(writes)` takes `PendingWrite` triples and applies them oldest-to-newest. An `Overwrite` inside the sequence resets the base and only later writes are reduced.
- The reducer has the signature `(state, writes: list[write]) -> new_state` where `writes` is a **batch** of everything written to this channel in one superstep. Do not use a pairwise function like `operator.add` — it would produce nested lists. Define a bulk reducer that iterates and extends instead.
- Use `Annotated[list[T], DeltaChannel(your_bulk_reducer)]` in your state schema.

### Example 1 — basic usage as state field

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.delta import DeltaChannel

def append_reducer(state: list[str], writes: Sequence[list[str]]) -> list[str]:
    """Bulk reducer: each write is a list[str]; flatten all writes onto state."""
    result = list(state)
    for write in writes:
        result.extend(write)
    return result

class State(TypedDict):
    items: Annotated[list[str], DeltaChannel(append_reducer)]

def add_item(state: State) -> State:
    return {"items": [f"item_{len(state['items']) + 1}"]}

graph = (
    StateGraph(State)
    .add_node("add", add_item)
    .add_edge(START, "add")
    .add_edge("add", END)
    .compile()
)

result = graph.invoke({"items": []})
print(result["items"])  # ['item_1']
```

### Example 2 — low-level channel inspection

```python
from typing import Sequence
from langgraph.channels.delta import DeltaChannel
from langgraph._internal._typing import MISSING

def append_reducer(state: list[str], writes: Sequence[list[str]]) -> list[str]:
    result = list(state)
    for write in writes:
        result.extend(write)
    return result

ch = DeltaChannel(append_reducer, list, snapshot_frequency=5)

# start from checkpoint (MISSING → empty list)
live = ch.from_checkpoint(MISSING)
print(live.get())  # []

# each call to update() passes a list of writes for that superstep
live.update([["a", "b"]])   # one write: the list ["a", "b"]
live.update([["c"]])        # one write: the list ["c"]
print(live.get())           # ['a', 'b', 'c']

# checkpoint() returns MISSING — full snapshot is written separately
print(live.checkpoint() is MISSING)  # True
```

### Example 3 — `replay_writes` for checkpoint reconstruction

```python
from typing import Sequence
from langgraph.channels.delta import DeltaChannel
from langgraph._internal._typing import MISSING

def append_reducer(state: list[str], writes: Sequence[list[str]]) -> list[str]:
    """Bulk reducer — batching-invariant: append_reducer(append_reducer(s, xs), ys) == append_reducer(s, xs+ys)."""
    result = list(state)
    for write in writes:
        result.extend(write)
    return result

ch = DeltaChannel(append_reducer, list)
live = ch.from_checkpoint(MISSING)   # start empty

# PendingWrite = tuple[str, str, Any] — fields are (task_id, channel_name, value).
# replay_writes() only uses the third element; the first two are discarded.
# Each value is a list[str] — the same shape a StateGraph node produces for
# a list[str] state field.
pending = [
    ("task-1", "items", ["alpha"]),
    ("task-2", "items", ["beta"]),
    ("task-3", "items", ["gamma"]),
]
live.replay_writes(pending)
print(live.get())  # ['alpha', 'beta', 'gamma']

# A second replay batch builds on top
more = [("task-4", "items", ["delta"])]
live.replay_writes(more)
print(live.get())  # ['alpha', 'beta', 'gamma', 'delta']
```

---

## 4 · `NamedBarrierValue` + `NamedBarrierValueAfterFinish`

**Module:** `langgraph.channels.named_barrier_value`

These two channels implement synchronization barriers: a downstream node can only fire after **every named producer** has written to the channel. They model fan-out → fan-in without manually tracking which branches have completed.

**Key source facts** (from `langgraph/channels/named_barrier_value.py`):

- Constructor: `NamedBarrierValue(typ, names: set[Value])` — `names` is the complete set of expected writers.
- `update(values)` adds each value to `seen`. Raises `InvalidUpdateError` if a value is not in `names`.
- `is_available()` returns `True` only when `seen == names` (all named writers have contributed).
- `consume()` resets `seen` to `set()`, preparing the barrier for reuse in the next step.
- `NamedBarrierValueAfterFinish` adds a `finished` flag. The barrier arms itself (all names seen → `is_available()`) only after `finish()` is called; the channel is consumed and reset by `consume()`.
- The `get()` method returns `None` — the barrier's purpose is triggering, not passing data through it. Pair it with `read_from` to pull actual data from other channels.

### Example 1 — basic barrier: fire after two branches complete

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue

barrier: NamedBarrierValue[str] = NamedBarrierValue(str, {"branch_a", "branch_b"})

print(barrier.is_available())    # False — no writes yet

barrier.update(["branch_a"])
print(barrier.is_available())    # False — waiting for branch_b

barrier.update(["branch_b"])
print(barrier.is_available())    # True  — all seen!

# consume resets for reuse
barrier.consume()
print(barrier.is_available())    # False
```

### Example 2 — invalid writer raises `InvalidUpdateError`

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.errors import InvalidUpdateError

barrier: NamedBarrierValue[str] = NamedBarrierValue(str, {"a", "b"})

try:
    barrier.update(["c"])    # "c" is not in names
except InvalidUpdateError as exc:
    print("blocked:", exc)
```

### Example 3 — `NamedBarrierValueAfterFinish` two-phase gate

```python
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish

gate: NamedBarrierValueAfterFinish[str] = NamedBarrierValueAfterFinish(
    str, {"worker_1", "worker_2"}
)

gate.update(["worker_1"])
gate.update(["worker_2"])
print(gate.is_available())   # False — finish() not called yet

# finish() arms the gate only when all names are present
gate.finish()
print(gate.is_available())   # True

gate.consume()
print(gate.is_available())   # False — reset; ready for next round
```

---

## 5 · `RunControl` + `GraphDrained`

**Module:** `langgraph.runtime` / `langgraph.errors`

`RunControl` is a lightweight per-run signal object that lets external code (e.g., a SIGTERM handler or a health monitor) request the graph to stop cooperatively at the next superstep boundary. `GraphDrained` is the exception raised when the graph honours that request.

**Key source facts** (from `langgraph/runtime.py` and `langgraph/errors.py`):

- `RunControl` uses a single slot `_drain_reason: str | None`. Writing to it from any thread is safe because a single attribute assignment is atomic in CPython.
- `request_drain(reason="shutdown")` sets `_drain_reason`. The graph checks `drain_requested` at each superstep boundary and raises `GraphDrained(reason)` when `True`.
- `drain_requested: bool` and `drain_reason: str | None` are read-only properties.
- `Runtime.drain_requested` and `Runtime.drain_reason` delegate directly to `self.control`.
- `GraphDrained` inherits from `GraphBubbleUp`, which is caught by the Pregel engine before propagating, allowing the checkpoint to be saved. The run can be resumed from that checkpoint later.
- The control object is available inside nodes via `runtime.control` or via the `Runtime.drain_requested` convenience shortcut.
- `RunControl` is **executor-owned** — you never instantiate it yourself in production code. The executor creates it and forwards it through `Runtime.control`. Access it only via `runtime.control`, `runtime.drain_requested`, or `runtime.drain_reason` inside a node.

### Example 1 — inspect `drain_requested` inside a node

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime, RunControl

class State(TypedDict):
    counter: int

def counting_node(state: State, runtime: Runtime) -> State:
    if runtime.drain_requested:
        print(f"draining: {runtime.drain_reason}")
        # In production you'd raise GraphDrained or finish cleanly.
        return {"counter": state["counter"]}
    return {"counter": state["counter"] + 1}

graph = (
    StateGraph(State)
    .add_node("count", counting_node)
    .add_edge(START, "count")
    .add_edge("count", END)
    .compile()
)

result = graph.invoke({"counter": 0})
print(result)  # {'counter': 1}
```

### Example 2 — `RunControl.request_drain()` from a background thread

```python
# NOTE: In production, RunControl is created by the executor — you never
# construct one yourself. This standalone example demonstrates the API
# surface only; it does not wire into compile()/invoke().
import threading, time
from langgraph.runtime import RunControl

control = RunControl()

def shutdown_later():
    time.sleep(0.05)
    control.request_drain("SIGTERM received")

thread = threading.Thread(target=shutdown_later, daemon=True)
thread.start()
thread.join()

print(control.drain_requested)   # True
print(control.drain_reason)      # 'SIGTERM received'
```

### Example 3 — catching `GraphDrained` at the call site

```python
from langgraph.errors import GraphDrained

def run_with_drain_handling(graph, inputs, config=None):
    """Run a graph; handle cooperative shutdown gracefully."""
    try:
        return graph.invoke(inputs, config or {})
    except GraphDrained as exc:
        print(f"Graph stopped early: {exc.reason}. Checkpoint saved. Resume later.")
        return None

# In a real scenario the drain is signalled externally.
# This shows the exception shape.
exc = GraphDrained("SIGTERM received")
print(exc.reason)        # SIGTERM received
print(str(exc))          # Graph drained: SIGTERM received
print(isinstance(exc, RecursionError))  # False — it's GraphBubbleUp
```

---

## 6 · `entrypoint.final`

**Module:** `langgraph.func`

`entrypoint.final` is a dataclass returned from a `@entrypoint` function to **decouple** what the caller receives (`value`) from what is persisted to the checkpoint (`save`). The next invocation on the same thread sees the `save` value via the `previous` parameter.

**Key source facts** (from `langgraph/func/__init__.py`):

- `entrypoint.final` is a generic dataclass `final(Generic[R, S])` with two fields: `value: R` and `save: S`.
- The `__call__` of `entrypoint` detects `isinstance(value, entrypoint.final)` and routes `value.value` to the output channel and `value.save` to the `PREVIOUS` channel.
- If the return annotation is `-> entrypoint.final[int, str]` then LangGraph infers `output_type=int` and `save_type=str`. An un-parameterised `-> entrypoint.final` maps both to `Any`.
- Without `entrypoint.final`, both the caller-visible output and the checkpoint save are the same object.
- `previous` is injected as a keyword-only parameter; it is `None` on the first invocation (no prior checkpoint).

### Example 1 — counter that exposes `previous` to the caller

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()

@entrypoint(checkpointer=saver)
def counter(
    increment: int,
    *,
    previous: int | None = None,
) -> entrypoint.final[int, int]:
    accumulated = (previous or 0) + increment
    # Return the new total to the caller; save doubled for next turn
    return entrypoint.final(value=accumulated, save=accumulated * 2)

config = {"configurable": {"thread_id": "t1"}}

r1 = counter.invoke(1, config)   # previous=None → accumulated=1; save=2
r2 = counter.invoke(3, config)   # previous=2    → accumulated=5; save=10
r3 = counter.invoke(1, config)   # previous=10   → accumulated=11; save=22

print(r1, r2, r3)  # 1 5 11
```

### Example 2 — returning a summary while checkpointing the raw data

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()

@entrypoint(checkpointer=saver)
def chat_session(
    message: str,
    *,
    previous: list[dict] | None = None,
) -> entrypoint.final[str, list[dict]]:
    history = previous or []
    history.append({"role": "user", "content": message})
    reply = f"Echo: {message}"           # placeholder for LLM call
    history.append({"role": "assistant", "content": reply})
    # Caller gets just the reply string; checkpoint stores full history
    return entrypoint.final(value=reply, save=history)

config = {"configurable": {"thread_id": "session1"}}
print(chat_session.invoke("hello", config))  # Echo: hello
print(chat_session.invoke("world", config))  # Echo: world
```

### Example 3 — `entrypoint.final` fields are always saved (even `None`)

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()

@entrypoint(checkpointer=saver)
def nullify(
    x: int,
    *,
    previous: int | None = None,
) -> entrypoint.final[None, int]:
    """Always returns None to the caller, but saves x."""
    print(f"  previous={previous}")
    return entrypoint.final(value=None, save=x)

config = {"configurable": {"thread_id": "t1"}}

r1 = nullify.invoke(42, config)
print(r1)    # None  ← caller sees None; 42 is saved to checkpoint

r2 = nullify.invoke(99, config)
# previous=42  ← the value saved by entrypoint.final in the prior call
print(r2)    # None  ← still None to the caller; 99 is now saved

# Verify the checkpoint exists for this thread:
snapshot = list(saver.list({"configurable": {"thread_id": "t1"}}))
print(bool(snapshot))  # True — checkpoint persisted even though value=None
```

---

## 7 · `ValuesTransformer` + `UpdatesTransformer` + `CustomTransformer`

**Module:** `langgraph.stream.transformers`

These three classes are **native transformers** for the v3 streaming protocol. `ValuesTransformer` is wired automatically when you call `graph.stream_events(version="v3")`; `UpdatesTransformer` and `CustomTransformer` are **not** wired by default and must be passed explicitly via `transformers=[UpdatesTransformer]` / `transformers=[CustomTransformer]`. Each processes a specific stream mode's raw events and exposes a typed `StreamChannel` projection.

**Key source facts** (from `langgraph/stream/transformers.py`):

- All three inherit `StreamTransformer` and set `_native = True`, which means their projection keys appear as **direct attributes** on `GraphRunStream` once the transformer is registered (e.g. `run.values` when `ValuesTransformer` is active, `run.updates` when `UpdatesTransformer` is passed explicitly).
- `required_stream_modes` declares which `stream_mode` strings must be enabled for the transformer to receive events.
- `ValuesTransformer` captures `"values"` events and exposes `run.values` as `StreamChannel[dict[str, Any]]`. It also maintains `_latest`, `_interrupted`, and `_interrupts` used by `run.output`, `run.interrupted`, and `run.interrupts`.
- `UpdatesTransformer` captures `"updates"` events and exposes `run.updates` as `StreamChannel[dict[str, Any]]`. Each item maps node/task name → update dict.
- `CustomTransformer` captures `"custom"` events (written by `get_stream_writer()` inside nodes) and exposes `run.custom` as `StreamChannel[Any]`.
- All three filter by `namespace` so that subgraph events are kept separate.
- `init()` returns a `dict` that populates `StreamMux.extensions`; `process(event)` returns `True` to continue event delivery.

### Example 1 — consuming `run.values` in the v3 API

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def step(state: State) -> State:
    return {"count": state["count"] + 1}

graph = (
    StateGraph(State)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile()
)

# stream_events(version="v3") wires ValuesTransformer automatically.
# NOTE: stream_mode is NOT accepted under version="v3" — the mux derives
# modes from the transformer set. Remove it to avoid TypeError.
with graph.stream_events(
    {"count": 0},
    version="v3",
) as run:
    for snapshot in run.values:
        print("snapshot:", snapshot)
    print("final:", run.output)
```

### Example 2 — `run.updates` via `UpdatesTransformer`

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    total: int

def add_ten(state: State) -> State:
    return {"total": state["total"] + 10}

def add_five(state: State) -> State:
    return {"total": state["total"] + 5}

graph = (
    StateGraph(State)
    .add_node("add_ten", add_ten)
    .add_node("add_five", add_five)
    .add_edge(START, "add_ten")
    .add_edge("add_ten", "add_five")
    .add_edge("add_five", END)
    .compile()
)

# UpdatesTransformer is NOT wired by default — pass it via transformers=
from langgraph.stream.transformers import UpdatesTransformer

with graph.stream_events(
    {"total": 0},
    version="v3",
    transformers=[UpdatesTransformer],   # opts in run.updates projection
) as run:
    for node_update in run.updates:
        # Each item: {"node_name": {"field": value, ...}}
        print("update:", node_update)
```

### Example 3 — `run.custom` via `CustomTransformer` + `get_stream_writer`

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    items: list[str]

def collecting_node(state: State) -> State:
    writer = get_stream_writer()
    for i, item in enumerate(state["items"]):
        writer({"progress": i + 1, "item": item})
    return {}

graph = (
    StateGraph(State)
    .add_node("collect", collecting_node)
    .add_edge(START, "collect")
    .add_edge("collect", END)
    .compile()
)

# CustomTransformer is NOT wired by default — pass it via transformers=
from langgraph.stream.transformers import CustomTransformer

with graph.stream_events(
    {"items": ["a", "b", "c"]},
    version="v3",
    transformers=[CustomTransformer],    # opts in run.custom projection
) as run:
    for payload in run.custom:
        print("custom:", payload)   # {'progress': 1, 'item': 'a'} …
```

---

## 8 · `MessagesTransformer`

**Module:** `langgraph.stream.transformers`

`MessagesTransformer` is the native v3 transformer for the `"messages"` stream mode. It yields one `ChatModelStream` (sync) or `AsyncChatModelStream` (async) per LLM call, and each stream exposes typed sub-projections (`.text`, `.reasoning`, `.tool_calls`, `.usage`, `.output`).

**Key source facts** (from `langgraph/stream/transformers.py`):

- Each new `message-start` protocol event creates a fresh `ChatModelStream` keyed by `run_id`.
- `message-finish` closes that stream and pushes it to `_log: StreamChannel[ChatModelStream]`.
- Whole `AIMessage` objects arriving via `on_chain_end` are replayed as synthetic protocol events via `message_to_events()` so the same projection API works for non-streaming models.
- V1 `AIMessageChunk` tuples (legacy `on_llm_new_token`) are **not** projected; models must use `stream_events(version="v3")` to populate `run.messages` with streaming content.
- `run.messages` is a `StreamChannel[ChatModelStream]`. Iterating it yields stream handles you then iterate for token-by-token content.
- `required_stream_modes = ("messages",)`.

### Example 1 — iterating `run.messages` to collect token streams

```python
# NOTE: requires a real ChatModel. This example shows the API structure.
# Replace `FakeChatModel` with langchain_anthropic.ChatAnthropic or similar.
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Simplified skeleton — wire a real model to see actual streaming
class State(TypedDict):
    response: str

def llm_node(state: State) -> State:
    # In a real graph this calls model.invoke() or model.stream()
    return {"response": "Hello, world!"}

graph = (
    StateGraph(State)
    .add_node("llm", llm_node)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile()
)

# With a real streaming model, run.messages yields ChatModelStream objects.
# stream_mode= is not accepted under version="v3" — omit it:
#
# with graph.stream_events({"response": ""}, version="v3") as run:
#     for msg_stream in run.messages:
#         for text_chunk in msg_stream.text:
#             print(text_chunk, end="", flush=True)
#         print()   # newline after each message
#     print("done:", run.output)

result = graph.invoke({"response": ""})
print(result["response"])   # Hello, world!
```

### Example 2 — `MessagesTransformer` scope filtering

```python
# MessagesTransformer only captures events whose namespace matches
# the run's own scope. Subgraph LLM calls appear on the subgraph's
# SubgraphRunStream.messages, not on the parent run.messages.
#
# The scope is stored as a list for O(1) equality checks against the
# namespace field in protocol events, which also arrives as a list.

from langgraph.stream.transformers import MessagesTransformer

# Root graph scope (empty tuple)
root_transformer = MessagesTransformer(scope=())
# Subgraph scope named "sub"
sub_transformer = MessagesTransformer(scope=("sub",))

# NOTE: _scope_list is a private implementation detail (leading underscore).
# It is used here only to verify scope assignment; do not depend on it in
# production code as it may change without notice.
print("root scope_list:", root_transformer._scope_list)   # []
print("sub  scope_list:", sub_transformer._scope_list)    # ['sub']

# Scoping rule: process() returns True without acting when
# params["namespace"] != self._scope_list.
# Root transformer ignores ["sub"] events; sub transformer ignores [] events.
assert root_transformer._scope_list != ["sub"]
assert sub_transformer._scope_list == ["sub"]
print("Scope filtering verified (via private _scope_list — for inspection only)")
```

### Example 3 — full async messages streaming pattern

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str

def echo_node(state: State) -> State:
    return {"text": state["text"].upper()}

graph = (
    StateGraph(State)
    .add_node("echo", echo_node)
    .add_edge(START, "echo")
    .add_edge("echo", END)
    .compile()
)

async def stream_messages():
    # astream_events(version="v3") returns a coroutine; await it first to
    # obtain the AsyncGraphRunStream, then enter it as an async context manager.
    # stream_mode= is rejected under version="v3" — omit it.
    run_stream = await graph.astream_events({"text": "hello"}, version="v3")
    async with run_stream as run:
        # With a real streaming model, iterating run.messages yields
        # AsyncChatModelStream objects; each exposes .text, .tool_calls etc.
        # AsyncGraphRunStream.output is an async method — use await + call:
        final = await run.output()
        print("state:", final)

asyncio.run(stream_messages())
```

---

## 9 · `AnyValue`

**Module:** `langgraph.channels.any_value`

`AnyValue` is a last-write-wins channel with **no concurrent-write guard**. When multiple nodes write to it in the same superstep, it silently takes the last value — no `InvalidUpdateError`. Use it when you need to share a channel across parallel branches that are guaranteed to converge to the same value, or when you explicitly want last-write semantics without a reducer.

**Key source facts** (from `langgraph/channels/any_value.py`):

- `update(values)` stores `values[-1]`. If `values` is empty, it clears the channel (sets `value = MISSING`).
- Unlike `LastValue`, there is no guard preventing multiple concurrent writes — it is the caller's responsibility to ensure convergence.
- `is_available()` returns `False` when `value is MISSING`.
- `checkpoint()` / `from_checkpoint()` persist and restore the raw value.
- Not to be confused with `LastValue` (which raises on concurrent writes) or `BinaryOperatorAggregate` (which merges via a reducer).

### Example 1 — concurrent writes take the last value silently

```python
from langgraph.channels.any_value import AnyValue

ch: AnyValue[int] = AnyValue(int)

# Two concurrent writes — no exception raised
ch.update([10, 99])
print(ch.get())   # 99 (last write wins)
```

### Example 2 — empty update clears the channel

```python
from langgraph.channels.any_value import AnyValue

ch: AnyValue[str] = AnyValue(str)
ch.update(["hello"])
print(ch.is_available())   # True

ch.update([])              # empty → clears
print(ch.is_available())   # False
```

### Example 3 — `AnyValue` in `StateGraph` via `Annotated` metadata

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.any_value import AnyValue

# StateGraph's channel resolver checks for BaseChannel instances in
# Annotated metadata before falling back to LastValue, so you can wire
# AnyValue directly — no custom Pregel construction needed.
# This is the correct fix for INVALID_CONCURRENT_GRAPH_UPDATE when you
# genuinely want last-write-wins behaviour across parallel branches.

class State(TypedDict):
    x: int
    result: Annotated[str, AnyValue(str)]   # last concurrent write wins; no exception

def branch_left(state: State) -> State:
    return {"result": "from_left"}

def branch_right(state: State) -> State:
    return {"result": "from_right"}

# Both branches run in the same superstep — AnyValue accepts both writes
# and keeps whichever arrived last (non-deterministic across branches).
# Connect branches directly to END to keep the pattern unambiguous.
graph = (
    StateGraph(State)
    .add_node("left", branch_left)
    .add_node("right", branch_right)
    .add_edge(START, "left")
    .add_edge(START, "right")
    .add_edge("left", END)
    .add_edge("right", END)
    .compile()
)

print(graph.invoke({"x": 1, "result": ""}))
# {'x': 1, 'result': 'from_left'} or 'from_right' — last write wins
```

---

## 10 · `ExecutionInfo` + `ServerInfo` + `get_runtime()`

**Module:** `langgraph.runtime`

`ExecutionInfo` and `ServerInfo` are frozen dataclasses injected into every node via `Runtime.execution_info` and `Runtime.server_info`. `get_runtime()` is the free function that retrieves the current `Runtime` from the config context variable.

**Key source facts** (from `langgraph/runtime.py`):

- `ExecutionInfo` fields: `checkpoint_id`, `checkpoint_ns`, `task_id`, `thread_id` (None without checkpointer), `run_id` (None if not passed in config), `node_attempt` (1-indexed retry count), `node_first_attempt_time` (Unix timestamp).
- `ExecutionInfo.patch(**overrides)` returns a new `ExecutionInfo` with selected fields replaced (uses `dataclasses.replace`).
- `ServerInfo` fields: `assistant_id`, `graph_id`, `user: BaseUser | None`. Populated only when running on LangGraph Platform / LangSmith deployments; `None` on open-source LangGraph.
- `get_runtime(context_schema=None)` reads the runtime from `get_config()[CONF][CONFIG_KEY_RUNTIME]`. It is equivalent to injecting `runtime: Runtime` as a parameter — prefer parameter injection.
- Both classes have `frozen=True, slots=True`, so they are immutable and memory-efficient.

### Example 1 — reading `execution_info` inside a node

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    msg: str

def info_node(state: State, runtime: Runtime) -> State:
    info = runtime.execution_info
    if info:
        print(f"checkpoint_id : {info.checkpoint_id}")
        print(f"task_id       : {info.task_id}")
        print(f"thread_id     : {info.thread_id}")
        print(f"attempt       : {info.node_attempt}")
    return {"msg": "done"}

saver = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("info", info_node)
    .add_edge(START, "info")
    .add_edge("info", END)
    .compile(checkpointer=saver)
)

graph.invoke({"msg": ""}, {"configurable": {"thread_id": "t1"}})
```

### Example 2 — `ExecutionInfo.patch()` for test stubs

```python
from langgraph.runtime import ExecutionInfo

real = ExecutionInfo(
    checkpoint_id="ckpt-001",
    checkpoint_ns="",
    task_id="task-abc",
    thread_id="thread-1",
    run_id="run-xyz",
    node_attempt=1,
    node_first_attempt_time=1_700_000_000.0,
)

# Simulate a retry without reconstructing the whole object
retried = real.patch(node_attempt=2)
print(retried.node_attempt)    # 2
print(retried.checkpoint_id)   # ckpt-001  (unchanged)
print(real.node_attempt)       # 1         (original immutable)
```

### Example 3 — `get_runtime()` as an alternative to parameter injection

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime, get_runtime

class State(TypedDict):
    value: int

def utility_fn() -> str:
    """Called from deep inside a node without direct access to `runtime`."""
    rt = get_runtime()
    info = rt.execution_info
    return f"attempt={info.node_attempt if info else '?'}"

def deep_node(state: State) -> State:
    label = utility_fn()
    return {"value": state["value"] + 1}

graph = (
    StateGraph(State)
    .add_node("deep", deep_node)
    .add_edge(START, "deep")
    .add_edge("deep", END)
    .compile()
)

# get_runtime() works only inside an active graph run context
result = graph.invoke({"value": 0})
print(result)   # {'value': 1}
```

---

## Summary

| # | Class / function | Module | Key use-case |
|---|-----------------|--------|-------------|
| 1 | `InMemoryCache` · `BaseCache` | `langgraph.cache.memory` | Namespace-scoped TTL node-result caching |
| 2 | `EphemeralValue` | `langgraph.channels.ephemeral_value` | One-superstep trigger channel; clears automatically |
| 3 | `DeltaChannel` (β) | `langgraph.channels.delta` | Replay-based accumulation; avoids full snapshot bloat |
| 4 | `NamedBarrierValue` · `NamedBarrierValueAfterFinish` | `langgraph.channels.named_barrier_value` | Fan-in gate: wait until all named branches complete |
| 5 | `RunControl` · `GraphDrained` | `langgraph.runtime` · `langgraph.errors` | Cooperative graceful drain / SIGTERM handling |
| 6 | `entrypoint.final` | `langgraph.func` | Decouple caller return value from checkpoint save value |
| 7 | `ValuesTransformer` · `UpdatesTransformer` · `CustomTransformer` | `langgraph.stream.transformers` | Native v3 stream projections for values/updates/custom |
| 8 | `MessagesTransformer` | `langgraph.stream.transformers` | Per-LLM-call `ChatModelStream` lifecycle in v3 streaming |
| 9 | `AnyValue` | `langgraph.channels.any_value` | Last-write-wins channel; no guard on concurrent writes |
| 10 | `ExecutionInfo` · `ServerInfo` · `get_runtime()` | `langgraph.runtime` | Read-only execution metadata injected into every node |

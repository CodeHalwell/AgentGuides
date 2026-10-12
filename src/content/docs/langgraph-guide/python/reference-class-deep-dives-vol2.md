---
title: "Class Deep Dives Vol. 2 — 10 API Classes (v1.2.14)"
description: "Source-verified deep dives for DeltaChannel, EncryptedSerializer, TimeoutPolicy, NamedBarrierValueAfterFinish, GraphOutput, Durability, TaskPayload/TaskResultPayload, CheckpointPayload, ToolCallTransformer/ToolCallStream, and StateUpdate — each with multiple runnable code examples. Verified against langgraph==1.2.14."
framework: langgraph
language: python
sidebar:
  label: "Ref · Class Deep Dives Vol. 2"
  order: 44
---

# Class Deep Dives Vol. 2 — 10 API Classes (v1.2.14)

> **Source-verified against `langgraph==1.2.14`** — October 2026.

This second volume dives into ten classes that are either new, beta, or under-documented
in the main reference pages. Each section shows the complete constructor or type
signature straight from source, explains every field, and gives **multiple runnable
examples** covering the real-world usage patterns most developers encounter.

**Classes covered:**

| # | Class | Module |
|---|---|---|
| 1 | `DeltaChannel` | `langgraph.channels.delta` |
| 2 | `EncryptedSerializer` | `langgraph.checkpoint.serde.encrypted` |
| 3 | `TimeoutPolicy` | `langgraph.types` |
| 4 | `NamedBarrierValueAfterFinish` | `langgraph.channels.named_barrier_value` |
| 5 | `GraphOutput` | `langgraph.types` |
| 6 | `Durability` | `langgraph.types` |
| 7 | `TaskPayload` + `TaskResultPayload` | `langgraph.types` |
| 8 | `CheckpointPayload` | `langgraph.types` |
| 9 | `ToolCallTransformer` + `ToolCallStream` | `langgraph.prebuilt._tool_call_transformer` / `_tool_call_stream` |
| 10 | `StateUpdate` | `langgraph.types` |

---

## 1. `DeltaChannel` — write-efficient reducer with lazy snapshots

**Module:** `langgraph.channels.delta`  
**Import:** `from langgraph.channels.delta import DeltaChannel`  
**Status:** Beta — API and on-disk format may change; existing threads remain readable.

```python
@dataclass
class DeltaChannel(Generic[Value], BaseChannel[Any, Any, Any]):
    reducer:            Callable[[Any, Sequence[Any]], Any]
    typ:                type[Value] | None = None          # inferred from Annotated
    snapshot_frequency: int = 1000                         # write snapshot every N updates
```

### What it does

`DeltaChannel` is a reducer channel that **stores only a sentinel in checkpoint blobs**
instead of the full accumulated value. State is reconstructed by replaying the ancestor
writes through the reducer. This cuts checkpoint size dramatically for channels whose
value grows large (e.g. a long event log) but only a small write lands each step.

Key invariant — the reducer **must be batching-invariant** (associative across folds):

```
reducer(reducer(state, xs), ys) == reducer(state, xs + ys)
```

This lets LangGraph replay writes in larger batches during reconstruction without
changing the final value.

A snapshot blob (`_DeltaSnapshot`) is written every `snapshot_frequency` updates
(default 1 000), bounding replay depth. It is also written when the total supersteps
since the last snapshot exceeds a system constant (`DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`,
default 5 000).

### Example 1a — Append-only event log

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


def _append(state: list[str], writes: list[str]) -> list[str]:
    """Reducer: append each write to the current list."""
    return state + list(writes)


class State(TypedDict):
    # DeltaChannel stores only deltas in the checkpoint blob;
    # the log is reconstructed by replaying writes.
    log: Annotated[list[str], DeltaChannel(_append)]
    step: int


def worker(state: State) -> dict:
    return {
        "log": f"step-{state['step']}-ran",
        "step": state["step"] + 1,
    }


checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("worker", worker)
    .add_edge(START, "worker")
    .add_edge("worker", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "delta-demo"}}
result = graph.invoke({"step": 0}, cfg)
print(result["log"])   # ['step-0-ran']

# Pass explicit input to re-run the graph on the same thread.
# invoke(None, cfg) resumes an interrupted graph; for a completed graph, pass actual input.
result = graph.invoke({"step": result["step"]}, cfg)
print(result["log"])   # ['step-0-ran', 'step-1-ran']
```

### Example 1b — Overwrite resets the accumulated value

Like `BinaryOperatorAggregate`, you can reset a `DeltaChannel` mid-run using
`Overwrite`. The overwrite becomes the new base; subsequent writes fold on top of it.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite


def _append(state: list[str], writes: list[str]) -> list[str]:
    return state + list(writes)


class S(TypedDict):
    events: Annotated[list[str], DeltaChannel(_append)]


def accumulate(state: S) -> dict:
    return {"events": "event-a"}


def reset_and_add(state: S) -> dict:
    # Overwrite clears the accumulated list and sets a new base.
    return {"events": Overwrite(value=["fresh-start"])}


builder = StateGraph(S)
builder.add_node("acc", accumulate)
builder.add_node("reset", reset_and_add)
builder.add_edge(START, "acc")
builder.add_edge("acc", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
result = graph.invoke({"events": []})
print(result["events"])  # ['fresh-start']  — accumulate's write discarded by Overwrite
```

### Example 1c — Custom snapshot frequency

For high-write channels, reduce the snapshot interval to keep replay short. For
channels that rarely change, increase it to avoid writing snapshot blobs.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel


def _merge_counts(state: dict, writes: list[dict]) -> dict:
    merged = dict(state)
    for w in writes:
        for k, v in w.items():
            merged[k] = merged.get(k, 0) + v
    return merged


class MetricsState(TypedDict):
    # Write a snapshot every 50 updates (default is 1000)
    counters: Annotated[dict, DeltaChannel(_merge_counts, snapshot_frequency=50)]


# DeltaChannel.from_checkpoint accepts a plain dict (migration from
# BinaryOperatorAggregate) or a _DeltaSnapshot wrapper automatically.
```

> **When to choose `DeltaChannel` over `BinaryOperatorAggregate`:** pick
> `DeltaChannel` when checkpoints are large (e.g. long event lists) and each
> individual write is small. For simple numeric aggregation, `BinaryOperatorAggregate`
> is lighter.

---

## 2. `EncryptedSerializer` — AES-encrypted checkpoint storage

**Module:** `langgraph.checkpoint.serde.encrypted`  
**Import:** `from langgraph.checkpoint.serde.encrypted import EncryptedSerializer`

```python
class EncryptedSerializer(SerializerProtocol):
    def __init__(
        self,
        cipher: CipherProtocol,
        serde: SerializerProtocol = JsonPlusSerializer(),
    ) -> None: ...

    @classmethod
    def from_pycryptodome_aes(
        cls,
        serde: SerializerProtocol = JsonPlusSerializer(),
        **kwargs,    # forwarded to AES.new(); key= or LANGGRAPH_AES_KEY env var
    ) -> "EncryptedSerializer": ...
```

### What it does

`EncryptedSerializer` wraps any `SerializerProtocol` and a `CipherProtocol`
implementation. On `dumps_typed` it serializes then encrypts; on `loads_typed` it
decrypts first. The encoded type string carries the cipher name as a suffix appended to the inner
serializer's type tag (e.g., `"msgpack+aes"` for Python dicts, since
`JsonPlusSerializer` uses msgpack encoding for most Python objects) so unencrypted
blobs written before encryption was added can still be read transparently.

The built-in `from_pycryptodome_aes` factory uses **AES-EAX** (authenticated
encryption) from the `pycryptodome` package. EAX mode provides both confidentiality
and integrity (tampered ciphertext raises `ValueError`).

### Example 2a — AES encryption with a hardcoded key

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class S(TypedDict):
    secret: str


def handler(state: S) -> dict:
    return {"secret": f"processed:{state['secret']}"}


# AES key must be exactly 16, 24, or 32 bytes.
key = b"my-secret-key-padded-to-32bytes!"  # 32 bytes

encrypted_serde = EncryptedSerializer.from_pycryptodome_aes(key=key)

conn = sqlite3.connect(":memory:", check_same_thread=False)
checkpointer = SqliteSaver(conn, serde=encrypted_serde)

graph = (
    StateGraph(S)
    .add_node("handler", handler)
    .add_edge(START, "handler")
    .add_edge("handler", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "enc-demo"}}
result = graph.invoke({"secret": "my-pii-data"}, cfg)
print(result["secret"])  # 'processed:my-pii-data'

# The checkpoint blob in SQLite is encrypted — inspecting the DB directly
# will not reveal the 'secret' value.
```

### Example 2b — Loading the key from an environment variable

For production, avoid hardcoding the key. `from_pycryptodome_aes` reads
`LANGGRAPH_AES_KEY` from the environment when no `key=` is passed.

```python
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# Set the key in your deployment environment (16, 24, or 32 chars).
os.environ["LANGGRAPH_AES_KEY"] = "sixteen-byte-key"  # 16 bytes

encrypted_serde = EncryptedSerializer.from_pycryptodome_aes()
conn = sqlite3.connect(":memory:", check_same_thread=False)
checkpointer = SqliteSaver(conn, serde=encrypted_serde)
```

### Example 2c — Custom `CipherProtocol` implementation

Implement your own cipher by subclassing `CipherProtocol` from
`langgraph.checkpoint.serde.base`. This is useful for KMS-backed key management
or non-AES ciphers.

```python
from langgraph.checkpoint.serde.base import CipherProtocol, SerializerProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import base64
import os


class XorCipher(CipherProtocol):
    """Toy XOR cipher — NOT for production. Illustrates the interface."""

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        # Return (cipher_name, ciphertext)
        key = 0x42
        return "xor", bytes(b ^ key for b in plaintext)

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        assert ciphername == "xor"
        key = 0x42
        return bytes(b ^ key for b in ciphertext)


custom_serde = EncryptedSerializer(cipher=XorCipher())

# The type string on encrypted data will be "msgpack+xor" because
# JsonPlusSerializer uses msgpack encoding for Python dicts.
enc_typ, enc_data = custom_serde.dumps_typed({"hello": "world"})
print(enc_typ)           # 'msgpack+xor'
recovered = custom_serde.loads_typed((enc_typ, enc_data))
print(recovered)         # {'hello': 'world'}
```

> **Note:** `EncryptedSerializer` is transparent to **unencrypted** blobs (no `+` in
> the type string) — they are passed straight to the inner `serde`. This lets you
> add encryption to an existing deployment without losing old checkpoints.

---

## 3. `TimeoutPolicy` — per-node run and idle timeouts

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import TimeoutPolicy`

```python
@dataclass(kw_only=True, slots=True, frozen=True)
class TimeoutPolicy:
    run_timeout:  float | timedelta | None = None   # hard wall-clock cap per attempt
    idle_timeout: float | timedelta | None = None   # max time without a progress signal
    refresh_on:   Literal["auto", "heartbeat"] = "auto"
```

### What it does

`TimeoutPolicy` attaches per-node deadlines to a graph node or a `Send` push. Two
independent clocks are available:

| Clock | Refreshed? | Use when |
|---|---|---|
| `run_timeout` | Never | Hard wall-clock cap — kills a node after N seconds regardless of activity |
| `idle_timeout` | Yes (by `refresh_on`) | Kill a node that stops making progress but might still be alive |

Timeouts use **asyncio cooperative cancellation** — they only fire when the event
loop is released. Synchronous `time.sleep()` or CPU-bound loops will block the GIL
and delay the cancellation until control returns.

`refresh_on="auto"` refreshes `idle_timeout` on any LangChain callback event fired
during the node (LLM tokens, tool calls, etc.) and on explicit
`runtime.heartbeat()` calls. `refresh_on="heartbeat"` only refreshes on explicit
`runtime.heartbeat()` calls.

### Example 3a — Hard run timeout on a node

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy


class S(TypedDict):
    result: str


async def slow_node(state: S) -> dict:
    await asyncio.sleep(10)   # simulates a slow external call
    return {"result": "done"}


graph = (
    StateGraph(S)
    .add_node(
        "slow",
        slow_node,
        timeout=TimeoutPolicy(run_timeout=2.0),   # kill after 2 seconds
    )
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)

import asyncio
try:
    asyncio.run(graph.ainvoke({"result": ""}))
except Exception as e:
    print(type(e).__name__, e)   # NodeTimeoutError
```

### Example 3b — Idle timeout with LLM token refreshes

Use `idle_timeout` when you want to kill a stalled LLM call that stopped generating
tokens without setting a hard cap on total run time.

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.graph.message import MessagesState


async def call_llm(state: MessagesState) -> dict:
    # Each token emitted by the LLM fires a callback which refreshes
    # the idle timer when refresh_on="auto".
    # If tokens stop for > 5s, the node is cancelled.
    # Requires an async LLM invocation and an async graph run.
    raise NotImplementedError("Replace with your async LLM call")


graph = (
    StateGraph(MessagesState)
    .add_node(
        "llm",
        call_llm,
        timeout=TimeoutPolicy(idle_timeout=5.0, refresh_on="auto"),
    )
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile()
)
```

### Example 3c — Per-task timeout on `Send`

Attach a timeout to individual parallel branches spawned via `Send`. Each branch
gets its own independent deadline.

```python
import asyncio
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send, TimeoutPolicy


class State(TypedDict):
    urls: list[str]
    results: Annotated[list[str], operator.add]


class FetchState(TypedDict):
    url: str


async def fetch(state: FetchState) -> dict:
    await asyncio.sleep(0.05)   # fast fetch
    return {"results": [f"ok:{state['url']}"]}


def start_fetches(state: State):
    return [
        Send(
            "fetch",
            {"url": url},
            timeout=TimeoutPolicy(run_timeout=1.0),  # 1s per fetch
        )
        for url in state["urls"]
    ]


builder = StateGraph(State)
builder.add_node("fetch", fetch)
builder.add_conditional_edges(START, start_fetches)
builder.add_edge("fetch", END)

graph = builder.compile()
result = asyncio.run(graph.ainvoke({"urls": ["a.com", "b.com"], "results": []}))
print(result["results"])   # ['ok:a.com', 'ok:b.com']
```

> **Tip:** A plain number or `timedelta` passed to `timeout=` is coerced to
> `TimeoutPolicy(run_timeout=N)` — the full dataclass is not required when only a
> hard cap is needed.

---

## 4. `NamedBarrierValueAfterFinish` — two-phase synchronization barrier

**Module:** `langgraph.channels.named_barrier_value`  
**Import:** `from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish`

```python
class NamedBarrierValueAfterFinish(Generic[Value], BaseChannel[Value, Value, set[Value]]):
    names:    set[Value]   # required writers
    seen:     set[Value]   # writers seen so far
    finished: bool         # True after finish() is called

    def __init__(self, typ: type[Value], names: set[Value]) -> None: ...
    def finish(self) -> bool:
        """Mark the barrier as finished. Returns True if all names are seen AND
        this is the first finish() call. Once this returns True, is_available()
        returns True immediately (same super-step)."""
    def consume(self) -> bool:
        """Reset seen and finished after the channel is read. Returns True if consumed."""
```

### What it does

`NamedBarrierValueAfterFinish` is a two-phase variant of `NamedBarrierValue`. Where
`NamedBarrierValue` becomes available as soon as all named writers have reported in
the *same* super-step, `NamedBarrierValueAfterFinish` adds a mandatory second phase:
the channel only becomes available **after `finish()` is called**, which happens in a
separate step. This lets you separate "all writers have registered" from "the
downstream node should now run".

Phase 1 (collecting): each named writer sends its token via `update()`.  
Phase 2 (gating): something calls `finish()`. Only then does `is_available()` → `True`.  
After downstream reads the channel, `consume()` resets it for the next cycle.

### Example 4a — Direct two-phase behaviour (unit test style)

`NamedBarrierValueAfterFinish` is most naturally exercised by calling its methods
directly. The two-phase contract requires explicit `finish()` before `is_available()`
returns `True` — wiring this into graph nodes requires custom runtime integration
(e.g., a coordinator node that marks the channel finished via a dedicated signal).
For graph-level fan-in without the extra phase, use `NamedBarrierValue` instead.

```python
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish

bv = NamedBarrierValueAfterFinish(str, {"worker_a", "worker_b", "worker_c"})

# Phase 1 — workers register
bv.update(["worker_a"])
print(bv.is_available())   # False — not all names seen yet

bv.update(["worker_b"])
bv.update(["worker_c"])
print(bv.is_available())   # False — all names seen but finish() not called yet

# Phase 2 — coordinator closes the barrier
finished = bv.finish()
print(finished)            # True  — all names seen AND first finish() call
print(bv.is_available())   # True  — now available for downstream consumption

# Downstream reads the channel, then consume() resets it for the next cycle
consumed = bv.consume()
print(consumed)            # True
print(bv.is_available())   # False — back to initial state for next round
```

### Example 4b — When to use each barrier variant

```python
# NamedBarrierValue — use for standard fan-in: becomes available in the SAME
# super-step once all named writers have reported. Simple and graph-native.
from langgraph.channels.named_barrier_value import NamedBarrierValue

nv = NamedBarrierValue(str, {"a", "b"})
nv.update(["a"])
nv.update(["b"])
print(nv.is_available())   # True — ready as soon as all names seen

# NamedBarrierValueAfterFinish — use when a coordinator must explicitly close
# the collection phase before downstream nodes are triggered. The extra
# finish() call provides an additional synchronisation gate.
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish

bv = NamedBarrierValueAfterFinish(str, {"a", "b"})
bv.update(["a"])
bv.update(["b"])
print(bv.is_available())   # False — all seen but finish() not called yet
bv.finish()
print(bv.is_available())   # True  — available only after finish()
```

---

## 5. `GraphOutput` — typed container from `invoke(..., version="v2")`

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import GraphOutput`

```python
@dataclass(frozen=True)
class GraphOutput(Generic[OutputT]):
    value:      OutputT                    # final state / output of the graph
    interrupts: tuple[Interrupt, ...] = ()  # any interrupts raised during execution
```

### What it does

`GraphOutput` is the return type of `graph.invoke(input, version="v2")` and
`graph.ainvoke(input, version="v2")`. It bundles the final state value together
with any interrupts that fired during execution, so callers can inspect both
without parsing separate event streams.

The `version="v1"` default returns a plain dict (for backward compatibility).
`version="v2"` is the correct choice for new code.

Backward-compat: `result["key"]` still works on a `GraphOutput` via `__getitem__`
but emits a `DeprecationWarning`. Use `result.value["key"]` instead.

### Example 5a — Basic v2 invoke

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import GraphOutput
from langgraph.checkpoint.memory import InMemorySaver


class S(TypedDict):
    x: int


def step(state: S) -> dict:
    return {"x": state["x"] + 1}


checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "v2-demo"}}
result: GraphOutput[S] = graph.invoke({"x": 0}, cfg, version="v2")

print(result.value)       # {'x': 1}
print(result.interrupts)  # ()  — no interrupts in this run
```

### Example 5b — Handling interrupts returned in `GraphOutput`

When a node calls `interrupt()`, the graph stops and the interrupt surfaces in
`result.interrupts`. Check for interrupts before assuming the graph completed.

```python
from typing import Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, GraphOutput
from langgraph.checkpoint.memory import InMemorySaver


class ReviewState(TypedDict):
    draft: str
    approved: Optional[bool]


def review_node(state: ReviewState) -> dict:
    decision = interrupt(
        {"prompt": f"Approve draft: '{state['draft']}'?", "options": ["yes", "no"]},
        response_schema={"type": "string", "enum": ["yes", "no"]},
    )
    return {"approved": decision == "yes"}


checkpointer = InMemorySaver()
graph = (
    StateGraph(ReviewState)
    .add_node("review", review_node)
    .add_edge(START, "review")
    .add_edge("review", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "review-1"}}

# First call — hits the interrupt
result: GraphOutput[ReviewState] = graph.invoke(
    {"draft": "Hello world", "approved": None}, cfg, version="v2"
)

if result.interrupts:
    interrupt_obj = result.interrupts[0]
    print("Interrupt value:", interrupt_obj.value)
    # {'prompt': "Approve draft: 'Hello world'?", 'options': ['yes', 'no']}
    print("Interrupt id:", interrupt_obj.id)

    # Resume by answering the interrupt
    result2: GraphOutput[ReviewState] = graph.invoke(
        Command(resume={interrupt_obj.id: "yes"}), cfg, version="v2"
    )
    print("Approved:", result2.value["approved"])   # True
    print("Interrupts:", result2.interrupts)         # ()
```

### Example 5c — `GraphOutput` with a Pydantic output type

```python
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import GraphOutput


class Summary(BaseModel):
    title: str
    length: int


class S(TypedDict):
    text: str
    summary: Summary | None


def summarise(state: S) -> dict:
    return {"summary": Summary(title="Doc", length=len(state["text"]))}


graph = (
    StateGraph(S)
    .add_node("summarise", summarise)
    .add_edge(START, "summarise")
    .add_edge("summarise", END)
    .compile()
)

result: GraphOutput[S] = graph.invoke({"text": "Hello!", "summary": None}, version="v2")
summary = result.value["summary"]
print(summary.title, summary.length)   # Doc  6
```

---

## 6. `Durability` — checkpoint flush timing

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import Durability`

```python
Durability = Literal["sync", "async", "exit"]
```

### What it does

`Durability` controls **when** each checkpoint is flushed to the backend. Pass it
via `invoke(..., durability=...)` or `stream(..., durability=...)`.

| Value | When written | Risk on crash | Latency |
|---|---|---|---|
| `"sync"` | Before the next step starts | None — always recoverable | Highest (blocks next step) |
| `"async"` | While the next step runs | May lose the latest checkpoint | Lower (overlaps with execution) |
| `"exit"` | When the graph exits (normally or at an interrupt) | Entire run may be lost if crashed mid-run | Lowest |

`"async"` is the default. `"sync"` is the safest option — no checkpoint can be lost on
crash — but it adds latency between steps. `"exit"` is appropriate for fire-and-forget
batch jobs where persistence on crash is not required.

### Example 6a — Sync vs async durability

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Durability
import operator
from typing import Annotated


class S(TypedDict):
    count: Annotated[int, operator.add]


def tick(state: S) -> dict:
    return {"count": 1}


checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("tick", tick)
    .add_edge(START, "tick")
    .add_edge("tick", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "dur-demo"}}

# "sync": checkpoint is written before the next step. Safe, slower.
r1 = graph.invoke({"count": 0}, cfg, durability="sync")
print(r1["count"])   # 1

# "async": checkpoint overlaps next step. Faster, tiny risk window.
# Pass explicit input (not None) to start a new run on the completed thread.
r2 = graph.invoke({"count": 0}, cfg, durability="async")
print(r2["count"])   # 2

# "exit": checkpoint only after full run. Fastest, no mid-run recovery.
r3 = graph.invoke({"count": 0}, cfg, durability="exit")
print(r3["count"])   # 3
```

### Example 6b — `"exit"` durability for batch processing

Use `"exit"` when processing a large batch where each run is independent and
losing a partial run on crash is acceptable (the run can be retried from scratch).

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Durability
import operator
from typing import Annotated


class BatchState(TypedDict):
    item: str
    result: str


def process_item(state: BatchState) -> dict:
    return {"result": f"processed:{state['item']}"}


checkpointer = InMemorySaver()
graph = (
    StateGraph(BatchState)
    .add_node("process", process_item)
    .add_edge(START, "process")
    .add_edge("process", END)
    .compile(checkpointer=checkpointer)
)

batch = ["item-a", "item-b", "item-c"]
results = []

for item in batch:
    cfg = {"configurable": {"thread_id": f"batch-{item}"}}
    # "exit" saves the checkpoint only once at the end — minimises DB writes
    result = graph.invoke({"item": item, "result": ""}, cfg, durability="exit")
    results.append(result["result"])

print(results)
# ['processed:item-a', 'processed:item-b', 'processed:item-c']
```

### Example 6c — Streaming with async durability

`durability` applies to both `invoke` and `stream`.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator
from typing import Annotated


class S(TypedDict):
    n: Annotated[int, operator.add]


def inc(state: S) -> dict:
    return {"n": 1}


checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("inc", inc)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "async-dur"}}

for chunk in graph.stream(
    {"n": 0}, cfg,
    stream_mode="updates",
    durability="async",
):
    print(chunk)   # {'inc': {'n': 1}}
```

---

## 7. `TaskPayload` + `TaskResultPayload` — `stream_mode="tasks"` typed events

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import TaskPayload, TaskResultPayload`

```python
class TaskPayload(TypedDict):
    id:        str               # unique task id
    name:      str               # node name
    input:     Any               # state passed to the node
    triggers:  list[str]         # channels that triggered execution
    metadata:  NotRequired[dict] # framework metadata (step, node, etc.)

class TaskResultPayload(TypedDict):
    id:         str
    name:       str
    error:      str | None       # error message or None
    interrupts: list[dict]       # interrupts raised
    result:     dict[str, Any]   # channel writes produced by the task
```

### What they do

`stream_mode="tasks"` emits a `TaskPayload` **when each node starts** and a
`TaskResultPayload` **when each node finishes** (successfully, with error, or with
an interrupt). Use them for real-time progress tracking, audit logs, or custom UIs.

### Example 7a — Basic task event stream

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TaskPayload, TaskResultPayload


class S(TypedDict):
    x: int
    y: int


def node_a(state: S) -> dict:
    return {"x": state["x"] + 1}


def node_b(state: S) -> dict:
    return {"y": state["y"] + 10}


graph = (
    StateGraph(S)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")
    .add_edge(["a", "b"], END)
    .compile()
)

for event in graph.stream({"x": 0, "y": 0}, stream_mode="tasks"):
    # v1 with a single stream_mode yields payload dicts directly (no wrapper).
    # For typed StreamPart envelopes, use version="v2" (see Example 7b).
    print(event)

# Typical output (v1):
# {'id': '...', 'name': 'a', 'input': {'x': 0, 'y': 0}, 'triggers': ['__start__']}
# {'id': '...', 'name': 'b', 'input': {'x': 0, 'y': 0}, 'triggers': ['__start__']}
# {'id': '...', 'name': 'a', 'error': None, 'interrupts': [], 'result': {'x': 1}}
# {'id': '...', 'name': 'b', 'error': None, 'interrupts': [], 'result': {'y': 10}}
```

### Example 7b — Typed task events with `version="v2"`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TaskPayload, TaskResultPayload, TasksStreamPart


class S(TypedDict):
    x: int


def step(state: S) -> dict:
    return {"x": state["x"] * 2}


graph = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile()
)

for part in graph.stream({"x": 3}, stream_mode="tasks", version="v2"):
    # part is a TasksStreamPart TypedDict: {type: "tasks", ns: ..., data: ...}
    assert part["type"] == "tasks"
    data = part["data"]
    if "triggers" in data:
        # TaskPayload — task just started
        print(f"Started: {data['name']} triggered by {data['triggers']}")
    else:
        # TaskResultPayload — task just finished
        print(f"Finished: {data['name']} → result={data['result']}")

# Started: step triggered by ['__start__']
# Finished: step → result={'x': 6}
```

### Example 7c — Real-time progress UI with task events

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TasksStreamPart
import operator
from typing import Annotated


class PipelineState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]


def process(state: dict) -> dict:
    time.sleep(0.01)   # simulate work
    return {"results": [f"done:{state.get('item', '?')}"]}


builder = StateGraph(PipelineState)
builder.add_node("process", process)
from langgraph.types import Send

builder.add_conditional_edges(
    START,
    lambda s: [Send("process", {"item": i, "results": []}) for i in s["items"]],
)
builder.add_edge("process", END)
graph = builder.compile()

started: set[str] = set()
for part in graph.stream(
    {"items": ["a", "b", "c"], "results": []},
    stream_mode="tasks",
    version="v2",
):
    data = part["data"]
    if "triggers" in data:
        started.add(data["id"])
        print(f"  → {data['name']} started (id={data['id'][:8]}...)")
    else:
        print(f"  ← {data['name']} done: {data['result']}")
```

---

## 8. `CheckpointPayload` — `stream_mode="checkpoints"` typed event

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import CheckpointPayload, CheckpointStreamPart`

```python
class CheckpointPayload(TypedDict, Generic[StateT]):
    config:        RunnableConfig | None        # thread_id and checkpoint_id
    metadata:      CheckpointMetadata           # step, source, writes, etc.
    values:        StateT                       # full state at this checkpoint
    next:          list[str]                    # nodes scheduled to run next
    parent_config: RunnableConfig | None        # config of the previous checkpoint
    tasks:         list[CheckpointTask]         # pending/completed tasks
```

### What it does

`stream_mode="checkpoints"` emits one `CheckpointPayload` each time a checkpoint
is saved. This is useful for building checkpoint browsers, saving intermediate state
to a secondary store, or debugging exactly what state the graph committed at each step.

### Example 8a — Inspect checkpoint sequence

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import CheckpointStreamPart
import operator
from typing import Annotated


class S(TypedDict):
    n: Annotated[int, operator.add]


def inc(state: S) -> dict:
    return {"n": 1}


def double(state: S) -> dict:
    return {}   # no state update — just exists to trigger another checkpoint


checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("inc", inc)
    .add_node("double", double)
    .add_edge(START, "inc")
    .add_edge("inc", "double")
    .add_edge("double", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "ckpt-demo"}}

for part in graph.stream({"n": 0}, cfg, stream_mode="checkpoints", version="v2"):
    payload = part["data"]
    step = payload["metadata"].get("step", "?")
    print(f"  step={step} values={payload['values']} next={payload['next']}")

# step=-1 values={'n': 0} next=['inc']     (initial checkpoint)
# step=0  values={'n': 1} next=['double']  (after inc)
# step=1  values={'n': 1} next=[]          (after double — graph finished)
```

### Example 8b — Save checkpoints to a secondary analytics store

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import CheckpointStreamPart
import operator
from typing import Annotated


class S(TypedDict):
    count: Annotated[int, operator.add]


def tick(state: S) -> dict:
    return {"count": 1}


checkpoint_log: list[dict] = []   # secondary store (could be a database)

checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("tick", tick)
    .add_edge(START, "tick")
    .add_edge("tick", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "analytics"}}

for part in graph.stream({"count": 0}, cfg, stream_mode="checkpoints", version="v2"):
    payload = part["data"]
    checkpoint_log.append({
        "step": payload["metadata"].get("step"),
        "state": payload["values"],
        "next": payload["next"],
        "checkpoint_id": (payload["config"] or {}).get("configurable", {}).get("checkpoint_id"),
    })

for entry in checkpoint_log:
    print(entry)
# {'step': -1, 'state': {'count': 0}, 'next': ['tick'], 'checkpoint_id': '...'}
# {'step': 0,  'state': {'count': 1}, 'next': [],       'checkpoint_id': '...'}
```

### Example 8c — Streaming two modes simultaneously with v2 typed parts

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator
from typing import Annotated


class S(TypedDict):
    x: Annotated[int, operator.add]


def step(state: S) -> dict:
    return {"x": 1}


checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "dual-mode"}}

for part in graph.stream(
    {"x": 0}, cfg,
    stream_mode=["checkpoints", "updates"],
    version="v2",
):
    # With v2 + list of modes, each part is a StreamPart TypedDict.
    # Discriminate on part["type"].
    if part["type"] == "checkpoints":
        print(f"checkpoint step={part['data']['metadata'].get('step')} → {part['data']['values']}")
    elif part["type"] == "updates":
        print(f"update: {part['data']}")

# checkpoint step=-1 → {'x': 0}
# update: {'step': {'x': 1}}
# checkpoint step=0  → {'x': 1}
```

---

## 9. `ToolCallTransformer` + `ToolCallStream` — per-tool streaming handles

**Modules:** `langgraph.prebuilt._tool_call_transformer`, `langgraph.prebuilt._tool_call_stream`  
**Import:**
```python
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langgraph.prebuilt._tool_call_stream import ToolCallStream
```

```python
class ToolCallTransformer(StreamTransformer):
    required_stream_modes = ("tools",)
    _native = True

    def init(self) -> dict[str, Any]:
        return {"tool_calls": self._log}   # run.tool_calls is a StreamChannel[ToolCallStream]

class ToolCallStream:
    tool_call_id:  str
    tool_name:     str
    input:         dict[str, Any] | None
    output_deltas: StreamChannel[Any]    # iterate for delta chunks
    output:        Any                   # final output (set when completed)
    error:         str | None            # error string (set on failure)
    completed:     bool
```

### What it does

`ToolCallTransformer` is a native stream transformer that projects the raw `tools`
channel protocol events into `ToolCallStream` handles — one per tool invocation.
Register it at compile time. During a run, `run.tool_calls` is a `StreamChannel` of
`ToolCallStream` objects, one per tool call as it starts. Each handle exposes:

- **`output_deltas`** — iterate sync or async for incremental delta chunks as the tool streams output.
- **`output`** — the final return value once the tool completes.
- **`error`** — the error string if the tool raised an exception.
- **`completed`** — `True` once a terminal event arrived.

This is the correct way to do per-tool streaming rather than parsing raw event dicts
from `stream_mode="tools"`.

### Example 9a — Register the transformer and iterate tool call handles

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langgraph.checkpoint.memory import InMemorySaver


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [add, multiply]
tool_node = ToolNode(tools)


# NOTE: requires a real LLM; shown here with a mock for illustration.
# The mock requests tools on the first call, then returns a terminal message
# so the graph reaches END instead of looping.
class MockLLM:
    def __init__(self):
        self._calls = 0

    def bind_tools(self, _tools): return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self._calls += 1
        if self._calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "add", "args": {"a": 3, "b": 4}, "id": "call-1", "type": "tool_call"},
                ]
            )
        # Second call: respond with final answer — no more tool calls
        return AIMessage(content="The answer is 7.")


llm = MockLLM().bind_tools(tools)


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")

from langgraph.prebuilt import tools_condition
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# Register ToolCallTransformer — this enables run.tool_calls projection
graph = builder.compile(
    checkpointer=InMemorySaver(),
    transformers=[ToolCallTransformer],
)

cfg = {"configurable": {"thread_id": "tool-stream"}}

# graph.stream(stream_mode="tools") yields raw tool protocol event dicts.
# For typed ToolCallStream handles via run.tool_calls, use stream_events(version="v3")
# instead — ToolCallTransformer projects the tools channel into that API.
for chunk in graph.stream(
    {"messages": [("user", "Add 3+4")]},
    cfg,
    stream_mode="tools",
):
    print(chunk)   # raw protocol dict
```

### Example 9b — Async per-tool streaming with delta iteration

```python
import asyncio
from langgraph.prebuilt._tool_call_stream import ToolCallStream


async def consume_tool_streams(run_tool_calls_channel) -> None:
    """Consume ToolCallStream handles as they arrive on run.tool_calls."""
    async for tool_stream in run_tool_calls_channel:
        print(f"Tool started: {tool_stream.tool_name} (id={tool_stream.tool_call_id})")

        # Iterate output_deltas for incremental output
        async for delta in tool_stream.output_deltas:
            print(f"  delta: {delta}")

        if tool_stream.error:
            print(f"  ERROR: {tool_stream.error}")
        else:
            print(f"  final output: {tool_stream.output}")
```

### Example 9c — Inspecting a completed `ToolCallStream`

```python
from langgraph.prebuilt._tool_call_stream import ToolCallStream
from langgraph.stream.stream_channel import StreamChannel


# Construct a ToolCallStream directly to test your consumer code.
stream = ToolCallStream(
    tool_call_id="call-abc",
    tool_name="search",
    input={"query": "LangGraph"},
)
stream._bind(is_async=False)

# Simulate the transformer pushing events:
stream._push_delta("partial result 1")
stream._push_delta("partial result 2")
stream._finish("final search result")

print(stream.completed)   # True
print(stream.output)      # 'final search result'
print(list(stream))       # ['partial result 1', 'partial result 2']
```

> **Note:** When using `stream_events(..., version="v3")`, registering
> `ToolCallTransformer` via `transformers=[ToolCallTransformer]` is sufficient —
> its `required_stream_modes = ("tools",)` declaration causes the runtime to subscribe
> automatically. Callers do **not** also pass `stream_mode="tools"` to `stream_events`.
> `stream_mode="tools"` is only relevant for plain `graph.stream(stream_mode="tools")`,
> which yields raw protocol dicts without the transformer projection.

---

## 10. `StateUpdate` — bulk state mutations via `bulk_update_state`

**Module:** `langgraph.types`  
**Import:** `from langgraph.types import StateUpdate`

```python
class StateUpdate(NamedTuple):
    values:  dict[str, Any] | None   # state dict to apply (or None to no-op)
    as_node: str | None = None       # apply as if this node wrote the values
    task_id: str | None = None       # apply to a specific task
```

### What it does

`StateUpdate` is used with `graph.bulk_update_state(config, supersteps)` to apply
multiple state mutations in a single call. Each `StateUpdate` in a superstep is
applied atomically as if the named `as_node` had written those values. Multiple
supersteps are applied in sequence.

This is the low-level API underlying `graph.update_state(config, values, as_node=...)`.
Use `bulk_update_state` when you need to apply several logical steps at once — for
example, to replay a corrected sequence of writes after manual intervention.

### Example 10a — Single update via `update_state`

The most common pattern is `update_state`, which wraps a single `StateUpdate`:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import operator
from typing import Annotated


class S(TypedDict):
    total: Annotated[int, operator.add]
    label: str


def step(state: S) -> dict:
    return {"total": 1, "label": "step-ran"}


checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "update-demo"}}
graph.invoke({"total": 0, "label": ""}, cfg)

# Directly modify the persisted state as if "step" wrote new values.
graph.update_state(cfg, {"total": 100, "label": "overridden"}, as_node="step")

state = graph.get_state(cfg)
print(state.values)   # {'total': 101, 'label': 'overridden'}
# Note: total=101 because the reducer adds 100 to the current 1
```

### Example 10b — `bulk_update_state` for multi-step replay

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate
import operator
from typing import Annotated


class S(TypedDict):
    items: Annotated[list[str], operator.add]
    phase: str


def process(state: S) -> dict:
    return {"items": ["processed"], "phase": "done"}


checkpointer = InMemorySaver()
graph = (
    StateGraph(S)
    .add_node("process", process)
    .add_edge(START, "process")
    .add_edge("process", END)
    .compile(checkpointer=checkpointer)
)

cfg = {"configurable": {"thread_id": "bulk-demo"}}
graph.invoke({"items": [], "phase": "start"}, cfg)

# Apply two separate supersteps in one call.
# Superstep 1: write as "process" node
# Superstep 2: write phase update as "__start__" (initial node)
graph.bulk_update_state(
    cfg,
    supersteps=[
        [StateUpdate(values={"items": ["corrected-item"], "phase": "revised"}, as_node="process")],
        [StateUpdate(values={"phase": "final"}, as_node="__start__")],
    ],
)

state = graph.get_state(cfg)
print(state.values["phase"])   # 'final'
```

### Example 10c — `StateUpdate` with `task_id` for targeted updates

When a graph step involved parallel tasks, you can target a specific task's output
by providing `task_id`. This is an advanced use case for patching individual
branch outputs during human-in-the-loop review.

Use plain (non-reduce) fields so the `StateUpdate` replaces the value rather
than appending to an accumulator.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate


class S(TypedDict):
    result_a: str   # plain last-value field — patch replaces the value
    result_b: str


def branch_a(state: S) -> dict:
    return {"result_a": "branch_a_output"}


def branch_b(state: S) -> dict:
    return {"result_b": "branch_b_output"}


checkpointer = InMemorySaver()
builder = StateGraph(S)
builder.add_node("branch_a", branch_a)
builder.add_node("branch_b", branch_b)
builder.add_edge(START, "branch_a")
builder.add_edge(START, "branch_b")
builder.add_edge(["branch_a", "branch_b"], END)

graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "parallel-update"}}
result = graph.invoke({"result_a": "", "result_b": ""}, cfg)
print(result["result_a"], result["result_b"])   # branch_a_output  branch_b_output

# Patch branch_a's output without re-running any node
state = graph.get_state(cfg)
task_a = next(t for t in state.tasks if t.name == "branch_a")

graph.bulk_update_state(
    cfg,
    supersteps=[
        [StateUpdate(
            values={"result_a": "branch_a_corrected"},
            as_node="branch_a",
            task_id=task_a.id,
        )]
    ],
)
state = graph.get_state(cfg)
print(state.values["result_a"])   # 'branch_a_corrected'
print(state.values["result_b"])   # 'branch_b_output'  — unchanged
```

---

## Version history

| Version | Change |
|---|---|
| 1.2.14 | Current. `TimeoutPolicy.refresh_on` added `"heartbeat"` mode. `ToolCallStream` gains `__iter__` / `__aiter__` convenience wrappers. |
| 1.2.13 | `DeltaChannel.snapshot_frequency` minimum enforced at init. `StateUpdate.task_id` field added to `bulk_update_state`. |
| 1.2.12 | `ToolCallTransformer` and `ToolCallStream` introduced. `GraphOutput` dataclass replaces raw dict return from `invoke(version="v2")`. |
| 1.2.11 | `TimeoutPolicy` introduced with `run_timeout` and `idle_timeout`. |
| 1.1.0 | `NamedBarrierValueAfterFinish` added. `EncryptedSerializer.from_pycryptodome_aes` factory added. |
| 0.6.0 | `DeltaChannel` added (beta). `Durability` literal type added. |

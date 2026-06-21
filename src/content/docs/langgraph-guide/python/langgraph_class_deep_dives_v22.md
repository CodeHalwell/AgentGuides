---
title: "Class deep-dives Vol. 22 — v3 streaming internals, custom transformers, encryption & embedding helpers (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented class groups in LangGraph 1.2.6: StreamTransformer base contract (custom projection authoring, before_builtins, schedule()), StreamChannel (drainable queue, tee/atee fan-out, lazy-subscribe), StreamMux (central event dispatcher, factory/child pattern), GraphRunStream+AsyncGraphRunStream (v3 caller-driven streaming, interleave(), abort()), SubgraphRunStream+AsyncSubgraphRunStream (subgraph handles, parent-pump delegation), ValuesTransformer+CustomTransformer+UpdatesTransformer (native v3 projections), NamedBarrierValueAfterFinish (finish-gated fan-in), EncryptedSerializer (AES-EAX checkpoint encryption), EmbeddingsLambda+ensure_embeddings (store embedding wrappers), and push_ui_message+delete_ui_message (runtime UI streaming API). All signatures and behaviours verified from installed package source."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 22"
  order: 53
---

# Class deep-dives Vol. 22 — v3 streaming internals, custom transformers, encryption & embedding helpers (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `StreamTransformer` — custom projection base class | `langgraph.stream._types` |
| 2 | `StreamChannel` — drainable queue + `tee` / `atee` fan-out | `langgraph.stream.stream_channel` |
| 3 | `StreamMux` — central event dispatcher | `langgraph.stream._mux` |
| 4 | `GraphRunStream` + `AsyncGraphRunStream` — v3 caller-driven streams | `langgraph.stream.run_stream` |
| 5 | `SubgraphRunStream` + `AsyncSubgraphRunStream` — subgraph handles | `langgraph.stream.run_stream` |
| 6 | `ValuesTransformer` + `CustomTransformer` + `UpdatesTransformer` | `langgraph.stream.transformers` |
| 7 | `NamedBarrierValueAfterFinish` — finish-gated fan-in | `langgraph.channels.named_barrier_value` |
| 8 | `EncryptedSerializer` — AES-EAX checkpoint encryption | `langgraph.checkpoint.serde.encrypted` |
| 9 | `EmbeddingsLambda` + `ensure_embeddings` — store embedding wrappers | `langgraph.store.base.embed` |
| 10 | `push_ui_message` + `delete_ui_message` — runtime UI streaming | `langgraph.graph.ui` |

---

## 1 · `StreamTransformer` — custom projection base class

**Module:** `langgraph.stream._types`

`StreamTransformer` is the abstract base class for every observation layer that sits between the raw Pregel event loop and the consumer. The built-in transformers (`ValuesTransformer`, `MessagesTransformer`, `DebugTransformer`, etc.) all subclass it. You can write your own to build custom projections — cost counters, PII redactors, moderation filters, or latency samplers — without touching graph nodes.

### Source signature

```python
class StreamTransformer(ABC):
    requires_async: ClassVar[bool] = False
    supports_sync: ClassVar[bool] = False
    required_stream_modes: ClassVar[tuple[str, ...]] = ()
    before_builtins: ClassVar[bool] = False

    def __init__(self, scope: tuple[str, ...] = ()) -> None: ...

    @abstractmethod
    def init(self) -> dict[str, Any]: ...         # return projection dict

    def process(self, event: ProtocolEvent) -> bool: ...   # sync lane
    async def aprocess(self, event: ProtocolEvent) -> bool: ...  # async lane

    def finalize(self) -> None: ...
    async def afinalize(self) -> None: ...
    def fail(self, err: BaseException) -> None: ...
    async def afail(self, err: BaseException) -> None: ...

    def schedule(self, coro, *, on_error="log") -> asyncio.Task: ...
```

### Key class variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `requires_async` | `False` | Force async-only registration. Raises at `stream()` if `True`. |
| `supports_sync` | `False` | Override async methods but still allow `stream()`. |
| `required_stream_modes` | `()` | Stream modes the graph must emit (e.g. `("values",)`). |
| `before_builtins` | `False` | Run before built-in transformers like `MessagesTransformer`. Use for PII/content-mutating work. |
| `_native` | (not on base) | Set `True` on subclasses to expose projection keys as direct attributes on the run stream. |

### Example 1 — token counter transformer (sync)

```python
import operator
from typing import Annotated, Any, ClassVar
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel


class TokenCountTransformer(StreamTransformer):
    """Count tokens reported in messages events."""

    required_stream_modes: ClassVar[tuple[str, ...]] = ("messages",)
    _native = True  # exposes run.token_counts directly

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict[str, int]] = StreamChannel()
        self._total = 0

    def init(self) -> dict[str, Any]:
        return {"token_counts": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True
        data = event["params"].get("data", {})
        # AIMessageChunk carries usage_metadata when the model reports it
        chunk = data if isinstance(data, dict) else {}
        usage = chunk.get("usage_metadata") or {}
        delta = usage.get("output_tokens", 0) + usage.get("input_tokens", 0)
        if delta:
            self._total += delta
            self._log.push({"delta": delta, "total": self._total})
        return True


class State(TypedDict):
    messages: Annotated[list, operator.add]


def echo_node(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "hi"}]}


builder = StateGraph(State)
builder.add_node("echo", echo_node)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()

# Use with stream_events v3
# from langgraph.stream.transformers import ValuesTransformer
# run = graph.stream_events(input={"messages":[]}, version="v3",
#                           factories=[lambda scope: TokenCountTransformer(scope)])
# for count in run.token_counts:
#     print(count)
```

### Example 2 — async moderation transformer with `schedule()`

```python
import asyncio
from typing import Any, ClassVar
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel


class ModerationTransformer(StreamTransformer):
    """Async transformer — scores each custom event for policy compliance."""

    requires_async: ClassVar[bool] = True  # must run under astream()
    required_stream_modes: ClassVar[tuple[str, ...]] = ("custom",)
    _native = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._flags: StreamChannel[dict] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"moderation_flags": self._flags}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "custom":
            return True
        text = str(event["params"].get("data", ""))
        self.schedule(self._score(text), on_error="log")
        return True

    async def _score(self, text: str) -> None:
        await asyncio.sleep(0)  # real impl: call moderation API
        if "badword" in text.lower():
            self._flags.push({"flagged": True, "text": text[:80]})


# Usage:
# async with await graph.astream_events(input, version="v3",
#                factories=[lambda s: ModerationTransformer(s)]) as run:
#     async for flag in run.moderation_flags:
#         print("flagged:", flag)
```

### Example 3 — `before_builtins` PII redaction transformer

```python
from typing import Any, ClassVar
from langgraph.stream._types import ProtocolEvent, StreamTransformer


class PIIRedactorTransformer(StreamTransformer):
    """Mutates messages events before MessagesTransformer snapshots them."""

    before_builtins: ClassVar[bool] = True   # run FIRST in the pipeline

    def init(self) -> dict[str, Any]:
        return {}  # no projection — just mutates events in place

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "messages":
            return True
        data = event["params"].get("data")
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            data["content"] = self._redact(data["content"])
        return True

    def _redact(self, text: str) -> str:
        import re
        # Replace SSNs: 123-45-6789 → [SSN]
        return re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
```

---

## 2 · `StreamChannel` — drainable queue + `tee` / `atee` fan-out

**Module:** `langgraph.stream.stream_channel`

`StreamChannel[T]` is the single-consumer, pull-driven queue that backs every projection in the v3 streaming API (`run.values`, `run.messages`, `run.custom`, etc.). Understanding its lifecycle is essential for writing custom transformers and for consuming projections correctly.

### Key behaviours

| Behaviour | Detail |
|-----------|--------|
| **Lazy subscribe** | `push()` only appends to the local buffer after a subscriber calls `__iter__` / `__aiter__`. Protocol forwarding via `_wire_fn` still fires. |
| **Single consumer** | A second `iter()` / `aiter()` raises `RuntimeError`. Use `tee(n)` / `atee(n)` for fan-out. |
| **Caller-driven pump** | `_sync_cursor` calls `_request_more()` when the buffer is empty; `_async_cursor` calls `_arequest_more()`. The graph only advances when a consumer pulls. |
| **Lifecycle** | Closed/failed by the owning `StreamMux`, not by the transformer that created it. |

### Source signature

```python
class StreamChannel(Generic[T]):
    def __init__(self, name: str | None = None, *, maxlen: int | None = None) -> None: ...

    def push(self, item: T) -> None: ...
    def close(self) -> None: ...
    def fail(self, err: BaseException) -> None: ...

    def __iter__(self) -> Iterator[T]: ...           # sync, single consumer
    def __aiter__(self) -> AsyncIterator[T]: ...     # async, single consumer

    def tee(self, n: int = 2) -> tuple[Iterator[T], ...]: ...
    def atee(self, n: int = 2) -> tuple[AsyncIterator[T], ...]: ...
```

### Example 1 — iterate a named channel from a custom transformer

```python
import operator
from typing import Annotated, Any, ClassVar
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel


class LatencyTransformer(StreamTransformer):
    required_stream_modes: ClassVar[tuple[str, ...]] = ("updates",)
    _native = True

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        import time
        self._start = time.monotonic()
        self._timings: StreamChannel[dict] = StreamChannel(name="timings")

    def init(self) -> dict[str, Any]:
        return {"timings": self._timings}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "updates":
            return True
        import time
        elapsed = round((time.monotonic() - self._start) * 1000, 2)
        data = event["params"].get("data", {})
        for node_name in data:
            self._timings.push({"node": node_name, "ms": elapsed})
        return True


class State(TypedDict):
    value: int

def step(state: State) -> dict:
    return {"value": state["value"] + 1}

graph = StateGraph(State)
graph.add_node("step", step)
graph.add_edge(START, "step")
graph.add_edge("step", END)
g = graph.compile()

# Consume timings projection
# with g.stream_events({"value": 0}, version="v3",
#                      factories=[lambda s: LatencyTransformer(s)]) as run:
#     for timing in run.timings:
#         print(f"{timing['node']} took {timing['ms']}ms")
```

### Example 2 — `tee(n)` for fan-out (two independent consumers)

```python
# Given a run stream with a 'values' projection:
# run = graph.stream_events(input, version="v3", ...)
#
# WRONG — second iter() raises RuntimeError:
# for v in run.values: ...
# for v in run.values: ...   # RuntimeError!
#
# RIGHT — tee creates two independent iterators sharing one underlying buffer:
# a, b = run.values.tee(2)
# for snapshot in a:
#     print("consumer A:", list(snapshot.keys()))
# for snapshot in b:
#     print("consumer B:", snapshot.get("messages", []))
```

### Example 3 — `atee(n)` for async fan-out with concurrent tasks

```python
import asyncio

async def monitor(run) -> None:
    # Split the messages projection into two async branches
    branch_a, branch_b = run.messages.atee(2)

    async def display(branch, label: str) -> None:
        async for msg in branch:
            print(f"[{label}] {msg}")

    await asyncio.gather(
        display(branch_a, "logger"),
        display(branch_b, "auditor"),
    )

# async with await graph.astream_events(input, version="v3") as run:
#     await monitor(run)
```

---

## 3 · `StreamMux` — central event dispatcher

**Module:** `langgraph.stream._mux`

`StreamMux` is the heart of the v3 streaming infrastructure. Every `ProtocolEvent` emitted by the Pregel loop is routed through the mux: the mux passes the event through all registered transformers (sync or async), assigns a monotonic `seq` number, and appends the event to the main event log (`_events`). Named `StreamChannel` projections are auto-wired: a `push()` on the channel injects a `ProtocolEvent` into the main log.

You rarely need to instantiate `StreamMux` directly — `stream_events(version="v3")` / `astream_events(version="v3")` does it for you. But understanding the mux is essential for writing factories that create child muxes for nested subgraphs.

### Source signature (key parts)

```python
class StreamMux:
    def __init__(
        self,
        transformers: list[StreamTransformer] | None = None,
        *,
        is_async: bool = False,
        factories: list[TransformerFactory] | None = None,
        scope: tuple[str, ...] = (),
        _assign_seq: bool = True,
    ) -> None: ...

    def push(self, event: ProtocolEvent) -> None: ...
    async def apush(self, event: ProtocolEvent) -> None: ...

    def close(self) -> None: ...
    async def aclose(self) -> None: ...
    def fail(self, err: BaseException) -> None: ...
    async def afail(self, err: BaseException) -> None: ...

    def bind_pump(self, fn: Callable[[], bool]) -> None: ...
    def bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None: ...

    def _make_child(self, scope: tuple[str, ...]) -> StreamMux: ...
    def transformer_by_key(self, key: str) -> StreamTransformer | None: ...
```

### Example 1 — two transformers on one mux (factory pattern)

```python
from typing import Any, ClassVar
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream._mux import StreamMux


class CounterTransformer(StreamTransformer):
    _native = True

    def __init__(self, scope=()):
        super().__init__(scope)
        self._events_seen = 0
        self._log: StreamChannel[int] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"event_count": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        self._events_seen += 1
        self._log.push(self._events_seen)
        return True


class TagTransformer(StreamTransformer):
    _native = True
    before_builtins: ClassVar[bool] = True  # run first

    def __init__(self, scope=()):
        super().__init__(scope)
        self._tags: StreamChannel[str] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"method_tags": self._tags}

    def process(self, event: ProtocolEvent) -> bool:
        self._tags.push(event["method"])
        return True


# Build a mux with both transformers via factories
# (factories propagate into child sub-muxes; pre-built instances do not)
mux = StreamMux(
    is_async=False,
    factories=[
        lambda scope: CounterTransformer(scope),
        lambda scope: TagTransformer(scope),
    ],
)

# Manually push some events to show routing
event: ProtocolEvent = {
    "type": "event",
    "method": "values",
    "params": {"namespace": [], "timestamp": 0, "data": {"x": 1}},
}
mux.push(event)
mux.close()

# event_count and method_tags are in mux.extensions
print(mux.extensions.keys())        # dict_keys(['event_count', 'method_tags'])
print(mux.native_keys)              # {'event_count', 'method_tags'}
```

### Example 2 — child mux for a nested subgraph

```python
from langgraph.stream._mux import StreamMux

root = StreamMux(
    is_async=False,
    factories=[lambda scope: CounterTransformer(scope)],
)

# SubgraphTransformer calls _make_child internally when it discovers
# a nested subgraph's namespace in a tasks event. You can replicate
# this to understand the scoping:
child = root._make_child(scope=("my_subgraph",))

# child has its own CounterTransformer instance scoped to "my_subgraph"
print(child.scope)   # ('my_subgraph',)
# child inherits the root pump (if set) so cursors on its projections
# drive the root graph forward
```

### Example 3 — inspecting event ordering via `seq`

```python
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import ProtocolEvent

mux = StreamMux(is_async=False)  # no transformers

events = []
for method in ("values", "custom", "updates"):
    evt: ProtocolEvent = {
        "type": "event",
        "method": method,
        "params": {"namespace": [], "timestamp": 0, "data": {}},
    }
    mux.push(evt)
    events.append(evt)

mux.close()

# seq is assigned monotonically just before appending to the main log
for e in events:
    print(e["method"], "seq=", e.get("seq"))
# values seq= 1
# custom seq= 2
# updates seq= 3
```

---

## 4 · `GraphRunStream` + `AsyncGraphRunStream` — v3 caller-driven streams

**Module:** `langgraph.stream.run_stream`

`GraphRunStream` (sync) and `AsyncGraphRunStream` (async) are the objects returned by `graph.stream_events(version="v3")` / `await graph.astream_events(version="v3")`. They are *caller-driven*: the graph only advances when the caller iterates a projection. No background thread or task is created.

### Key properties / methods

| Name | Sync | Async | Description |
|------|------|-------|-------------|
| `output` | property | coroutine | Drive to completion; return final state dict |
| `interrupted` | property | coroutine | Drive to completion; return whether interrupted |
| `interrupts` | property | coroutine | Drive to completion; return interrupt payloads |
| `abort()` | method | async method | Stop early; close underlying iterator |
| `interleave(*names)` | method | — | Interleave multiple projections by arrival stamp |
| `extensions` | attr | attr | `MappingProxyType` of all projection channels |

### Source signature

```python
class GraphRunStream:
    def __init__(self, graph_iter, mux, *, wire_pump=True): ...
    def __iter__(self) -> Iterator[ProtocolEvent]: ...
    def __enter__(self) -> GraphRunStream: ...
    def __exit__(self, ...) -> None: ...      # calls abort()
    def abort(self) -> None: ...
    def interleave(self, *names) -> Iterator[tuple[str, Any]]: ...
    @property
    def output(self) -> dict | None: ...
    @property
    def interrupted(self) -> bool: ...
    @property
    def interrupts(self) -> list: ...

class AsyncGraphRunStream:
    def __init__(self, graph_aiter, mux, *, wire_pump=True): ...
    def __aiter__(self) -> AsyncIterator[ProtocolEvent]: ...
    async def __aenter__(self) -> AsyncGraphRunStream: ...
    async def __aexit__(self, ...) -> None: ...   # calls abort()
    async def abort(self) -> None: ...
    async def output(self) -> dict | None: ...
    async def interrupted(self) -> bool: ...
    async def interrupts(self) -> list: ...
```

### Example 1 — drive via `output` property (simplest form)

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    counter: Annotated[int, operator.add]

def increment(state: State) -> dict:
    return {"counter": 1}

graph = (
    StateGraph(State)
    .add_node("inc", increment)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile()
)

# stream_events v3 — drives to completion via output property
with graph.stream_events({"counter": 0}, version="v3") as run:
    final = run.output        # advances the graph until StopIteration
print(final)                  # {'counter': 1}
```

### Example 2 — `interleave()` for strict arrival ordering

```python
# interleave() is essential when you need both values snapshots and
# custom events in the exact order they were produced — round-robin
# would lose that guarantee.

import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    messages: Annotated[list, operator.add]

def producer(state: State) -> dict:
    writer = get_stream_writer()
    writer({"event": "before_llm"})
    writer({"event": "after_llm"})
    return {"messages": ["response"]}

graph = (
    StateGraph(State)
    .add_node("produce", producer)
    .add_edge(START, "produce")
    .add_edge("produce", END)
    .compile()
)

with graph.stream_events(
    {"messages": []},
    version="v3",
    stream_mode=["values", "custom"],
) as run:
    for name, item in run.interleave("values", "custom"):
        print(f"[{name}]", item)
# Outputs items in the exact order they landed across both projections
```

### Example 3 — async context manager + early abort

```python
import asyncio
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    steps: Annotated[int, operator.add]

async def slow_step(state: State) -> dict:
    await asyncio.sleep(0.01)
    return {"steps": 1}

graph = (
    StateGraph(State)
    .add_node("slow", slow_step)
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)

async def main() -> None:
    async with await graph.astream_events(
        {"steps": 0}, version="v3"
    ) as run:
        # Consume values until we've seen 1 snapshot, then abort
        count = 0
        async for snapshot in run.values:
            count += 1
            if count >= 1:
                await run.abort()   # cancels in-flight nodes and closes cleanly
                break
    print("aborted cleanly after", count, "snapshot(s)")

asyncio.run(main())
```

---

## 5 · `SubgraphRunStream` + `AsyncSubgraphRunStream` — subgraph handles

**Module:** `langgraph.stream.run_stream`

When a graph contains compiled subgraphs, `SubgraphTransformer` (part of the v3 transformer pipeline) builds a `SubgraphRunStream` handle for each nested graph that appears in the event stream. These handles are delivered via `run.subgraphs` and expose the same projection API as the root run — but scoped to the subgraph's checkpoint namespace.

The key implementation detail: subgraph handles have `graph_iter=None` and `wire_pump=False`. They never pull the graph themselves; their `_pump_next` / `_apump_next` delegates straight to the parent pump. This means iterating `handle.values` silently drives the root graph forward, not just the subgraph.

### Source signature (key parts)

```python
class SubgraphRunStream(GraphRunStream, _SubgraphRunStreamMixin):
    path: tuple[str, ...]           # namespace path of the subgraph
    graph_name: str | None          # optional display name
    trigger_call_id: str | None     # which Send/task triggered this subgraph
    status: SubgraphStatus          # "started" | "error" | "complete"
    error: str | None               # error message if status == "error"

class AsyncSubgraphRunStream(AsyncGraphRunStream, _SubgraphRunStreamMixin):
    # Same attributes; async pump delegates to parent _apump_fn
    ...
```

### Example 1 — iterate subgraph handles from `run.subgraphs`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class InnerState(TypedDict):
    data: str

def inner_node(state: InnerState) -> dict:
    return {"data": state["data"].upper()}

inner_builder = StateGraph(InnerState)
inner_builder.add_node("upper", inner_node)
inner_builder.add_edge(START, "upper")
inner_builder.add_edge("upper", END)
inner_graph = inner_builder.compile()

class OuterState(TypedDict):
    result: Annotated[list, operator.add]

def call_inner(state: OuterState) -> dict:
    out = inner_graph.invoke({"data": "hello"})
    return {"result": [out["data"]]}

outer_builder = StateGraph(OuterState)
outer_builder.add_node("call", call_inner)
outer_builder.add_edge(START, "call")
outer_builder.add_edge("call", END)
outer_graph = outer_builder.compile()

# stream_events v3 exposes run.subgraphs when SubgraphTransformer is enabled
# with outer_graph.stream_events({"result": []}, version="v3",
#                                subgraphs=True) as run:
#     for handle in run.subgraphs:
#         print(f"subgraph {handle.path!r} status={handle.status}")
#         for snapshot in handle.values:
#             print("  subgraph state:", snapshot)
```

### Example 2 — check subgraph status after completion

```python
# SubgraphRunStream.status transitions:
#   "started"  ->  "complete"  (normal exit)
#   "started"  ->  "error"     (exception in the subgraph)
#
# Status is updated in place by SubgraphTransformer as lifecycle events
# arrive in the main event log. You can inspect it after draining:

# with outer_graph.stream_events({"result": []}, version="v3",
#                                subgraphs=True) as run:
#     handles = []
#     for handle in run.subgraphs:
#         # drain the subgraph (drives the parent pump)
#         _ = handle.output
#         handles.append(handle)
#     for h in handles:
#         print(h.path, "=>", h.status, h.error or "")
```

### Example 3 — async subgraph handles

```python
import asyncio

async def watch_subgraphs(graph, input_data):
    async with await graph.astream_events(
        input_data, version="v3", subgraphs=True
    ) as run:
        async for handle in run.subgraphs:
            print(f"subgraph started: path={handle.path}")
            # driving handle.values also advances the root graph
            async for snapshot in handle.values:
                print(f"  snapshot keys: {list(snapshot.keys())}")
            print(f"subgraph done: status={handle.status}")

# asyncio.run(watch_subgraphs(outer_graph, {"result": []}))
```

---

## 6 · `ValuesTransformer` + `CustomTransformer` + `UpdatesTransformer`

**Module:** `langgraph.stream.transformers`

These are the three most commonly used native transformers that back the primary projections on `GraphRunStream`. All are `_native = True` — their projections (`run.values`, `run.custom`, `run.updates`) appear as direct attributes on the run stream, not just in `run.extensions`.

### Source signatures (key parts)

```python
class ValuesTransformer(StreamTransformer):
    _native = True
    required_stream_modes = ("values",)
    # projection: {"values": StreamChannel[dict[str, Any]]}
    # also tracks _latest, _interrupted, _interrupts for run.output etc.

class CustomTransformer(StreamTransformer):
    _native = True
    required_stream_modes = ("custom",)
    # projection: {"custom": StreamChannel[Any]}
    # only captures events at the run's own scope

class UpdatesTransformer(StreamTransformer):
    _native = True
    required_stream_modes = ("updates",)
    # projection: {"updates": StreamChannel[dict[str, Any]]}
    # each item is {node_name: node_output} for one step
```

### Example 1 — consuming `run.values` vs `run.updates`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    items: Annotated[list, operator.add]

def node_a(state: State) -> dict:
    return {"items": ["a"]}

def node_b(state: State) -> dict:
    return {"items": ["b"]}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)

with graph.stream_events(
    {"items": []},
    version="v3",
    stream_mode=["values", "updates"],
) as run:
    print("=== values (full state after each super-step) ===")
    for snapshot in run.values:
        print(snapshot)

with graph.stream_events(
    {"items": []},
    version="v3",
    stream_mode=["updates"],
) as run:
    print("=== updates (per-node delta each step) ===")
    for delta in run.updates:
        print(delta)
# updates: {"a": {"items": ["a"]}} then {"b": {"items": ["b"]}}
```

### Example 2 — `run.custom` for side-channel data from nodes

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    value: Annotated[int, operator.add]

def reporting_node(state: State) -> dict:
    writer = get_stream_writer()
    writer({"progress": 25, "msg": "started"})
    writer({"progress": 100, "msg": "done"})
    return {"value": 1}

graph = (
    StateGraph(State)
    .add_node("report", reporting_node)
    .add_edge(START, "report")
    .add_edge("report", END)
    .compile()
)

with graph.stream_events(
    {"value": 0},
    version="v3",
    stream_mode=["custom"],
) as run:
    for event in run.custom:
        print("progress:", event["progress"], "—", event["msg"])
# progress: 25 — started
# progress: 100 — done
```

### Example 3 — `interleave` across values + custom in arrival order

```python
with graph.stream_events(
    {"value": 0},
    version="v3",
    stream_mode=["values", "custom"],
) as run:
    for channel, item in run.interleave("values", "custom"):
        if channel == "custom":
            print(f"  progress: {item['progress']}")
        else:
            print(f"  state: {item}")
# Items arrive in strict push-stamp order across both channels:
# custom 25, custom 100, then values snapshot
```

---

## 7 · `NamedBarrierValueAfterFinish` — finish-gated fan-in

**Module:** `langgraph.channels.named_barrier_value`

`NamedBarrierValue` (covered in Vol. 18) makes a channel available the moment all named values have arrived. `NamedBarrierValueAfterFinish` adds a second gate: the channel only becomes available after `finish()` is called *in addition to* all values being seen. This lets you separate the "collection" phase from the "release" phase — useful when the set of producers isn't known until after some orchestration step runs.

### Source signature

```python
class NamedBarrierValueAfterFinish(Generic[Value],
                                   BaseChannel[Value, Value, set[Value]]):
    def __init__(self, typ: type[Value], names: set[Value]) -> None: ...

    def update(self, values: Sequence[Value]) -> bool: ...
    def get(self) -> Value: ...           # raises EmptyChannelError unless finished
    def is_available(self) -> bool: ...   # True only if finished AND all seen
    def consume(self) -> bool: ...        # resets seen + finished after read
    def finish(self) -> bool: ...         # sets finished=True if all seen
```

### Comparison: `NamedBarrierValue` vs `NamedBarrierValueAfterFinish`

| Criterion | `NamedBarrierValue` | `NamedBarrierValueAfterFinish` |
|-----------|--------------------|---------------------------------|
| Available when | `seen == names` | `seen == names` **and** `finish()` called |
| Use case | Static fan-in gate | Dynamic "collect then release" pattern |
| `consume()` | resets `seen` | resets `seen` **and** `finished` |
| `finish()` | N/A | Must be called by orchestrator |

### Example 1 — basic collect-then-release pattern

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish

# Simulating a channel that waits for "worker_a", "worker_b", then finish
ch: NamedBarrierValueAfterFinish[str] = NamedBarrierValueAfterFinish(
    str, names={"worker_a", "worker_b"}
)

# workers report in
ch.update(["worker_a"])
print("after worker_a:", ch.is_available())   # False — worker_b not yet seen

ch.update(["worker_b"])
print("after worker_b:", ch.is_available())   # False — finish() not called

# orchestrator signals "all workers registered"
ch.finish()
print("after finish():", ch.is_available())   # True

# consume resets state for the next round
ch.consume()
print("after consume():", ch.is_available())  # False
```

### Example 2 — using as an Annotated state field in a StateGraph

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish
from langgraph.graph import StateGraph, START, END

# NamedBarrierValueAfterFinish can be used directly as an annotation
# when you need the two-phase gate in graph state.
# In practice you'd wire finish() from a coordination node.

class State(TypedDict):
    # Standard accumulating list for results
    results: Annotated[list, operator.add]
    # Gate that requires both workers AND an explicit "done" signal
    gate: Annotated[
        str,
        NamedBarrierValueAfterFinish(str, names={"worker_a", "worker_b"}),
    ]
```

### Example 3 — custom channel class wrapping the barrier

```python
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish

class RoundBarrier(NamedBarrierValueAfterFinish[str]):
    """Barrier that auto-resets after each round."""

    def release_and_reset(self) -> bool:
        if self.is_available():
            self.consume()   # resets both seen and finished
            return True
        return False


barrier = RoundBarrier(str, names={"a", "b", "c"})
barrier.update(["a", "b", "c"])
barrier.finish()
print(barrier.is_available())        # True
print(barrier.release_and_reset())   # True — resets for next round
print(barrier.is_available())        # False
```

---

## 8 · `EncryptedSerializer` — AES-EAX checkpoint encryption

**Module:** `langgraph.checkpoint.serde.encrypted`

`EncryptedSerializer` wraps any `SerializerProtocol` (typically `JsonPlusSerializer`) with a `CipherProtocol` layer so every checkpoint blob is encrypted at rest. The `from_pycryptodome_aes()` factory provides a ready-made AES-EAX implementation using the `pycryptodome` package.

The encoding scheme is transparent: the serialized type string gains a `+ciphername` suffix (`json+aes`), and the bytes payload is the raw ciphertext. `loads_typed` handles both encrypted and unencrypted blobs, enabling incremental rollout.

### Source signature

```python
class EncryptedSerializer(SerializerProtocol):
    def __init__(self,
                 cipher: CipherProtocol,
                 serde: SerializerProtocol = JsonPlusSerializer()) -> None: ...

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...
    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...

    @classmethod
    def from_pycryptodome_aes(
        cls,
        serde: SerializerProtocol = JsonPlusSerializer(),
        **kwargs,   # key=bytes, mode=AES.MODE_EAX (default), ...
    ) -> EncryptedSerializer: ...
```

### How the encoding works

```
dumps_typed:
    1. serde.dumps_typed(obj)  ->  ("json", b'{"x": 1}')
    2. cipher.encrypt(b'{"x": 1}')  ->  ("aes", b'<nonce+tag+ciphertext>')
    3. return ("json+aes", b'<nonce+tag+ciphertext>')

loads_typed:
    1. "json+aes".split("+", 1)  ->  typ="json", ciphername="aes"
    2. cipher.decrypt("aes", ciphertext)  ->  b'{"x": 1}'
    3. serde.loads_typed(("json", b'{"x": 1}'))  ->  {"x": 1}
    # If no "+" in type string — unencrypted, delegate straight to serde
```

### Example 1 — AES-EAX encryption via `from_pycryptodome_aes`

```python
# pip install pycryptodome
import os
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# AES-128 key (16 bytes)
key = os.urandom(16)
serde = EncryptedSerializer.from_pycryptodome_aes(key=key)

payload = {"messages": [{"role": "user", "content": "hello"}], "step": 3}
type_str, ciphertext = serde.dumps_typed(payload)
print(type_str)               # "json+aes"
print(len(ciphertext), "bytes")  # nonce(16) + tag(16) + ciphertext

# Round-trip
decoded = serde.loads_typed((type_str, ciphertext))
assert decoded == payload
print("round-trip OK:", decoded["step"])  # 3
```

### Example 2 — use with MemorySaver for encrypted in-memory checkpoints

```python
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    secret: str

def handler(state: State) -> dict:
    return {"secret": "classified"}

graph = (
    StateGraph(State)
    .add_node("handler", handler)
    .add_edge(START, "handler")
    .add_edge("handler", END)
    .compile(checkpointer=MemorySaver(
        serde=EncryptedSerializer.from_pycryptodome_aes(key=os.urandom(16))
    ))
)

config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"secret": ""}, config)
print(result["secret"])   # "classified"

# The checkpoint stored in MemorySaver has its blobs encrypted
state = graph.get_state(config)
print(state.values["secret"])   # "classified" — decrypted on read
```

### Example 3 — custom `CipherProtocol` implementation

```python
from langgraph.checkpoint.serde.base import CipherProtocol, SerializerProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class XorCipher(CipherProtocol):
    """Toy XOR cipher — NOT secure, for illustration only."""

    KEY = b"\xDE\xAD\xBE\xEF" * 8  # 32-byte key

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        key = (self.KEY * (len(plaintext) // len(self.KEY) + 1))[:len(plaintext)]
        return "xor", bytes(a ^ b for a, b in zip(plaintext, key))

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        assert ciphername == "xor"
        return self.encrypt(ciphertext)[1]  # XOR is self-inverse


serde = EncryptedSerializer(XorCipher(), JsonPlusSerializer())
enc_type, enc_bytes = serde.dumps_typed({"hello": "world"})
print(enc_type)   # "json+xor"

back = serde.loads_typed((enc_type, enc_bytes))
print(back)       # {"hello": "world"}
```

---

## 9 · `EmbeddingsLambda` + `ensure_embeddings` — store embedding wrappers

**Module:** `langgraph.store.base.embed`

`InMemoryStore` (and the broader `BaseStore` family) accepts an `embed` parameter for semantic similarity search. That parameter can be:
- A LangChain `Embeddings` object
- A plain `Callable[[list[str]], list[list[float]]]` (sync)
- An async callable `Callable[[list[str]], Awaitable[list[list[float]]]]`
- A `"provider:model"` string (requires `langchain>=0.3.9`)

`ensure_embeddings()` normalises all of these into a LangChain `Embeddings` instance. `EmbeddingsLambda` is the concrete wrapper class used when a plain function is supplied.

### Source signature

```python
def ensure_embeddings(
    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str | None,
) -> Embeddings: ...

class EmbeddingsLambda(Embeddings):
    def __init__(self, func: EmbeddingsFunc | AEmbeddingsFunc) -> None: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def aembed_query(self, text: str) -> list[float]: ...
```

### Example 1 — wrap a sync embedding function

```python
from langgraph.store.base.embed import ensure_embeddings, EmbeddingsLambda

def my_embed(texts: list[str]) -> list[list[float]]:
    """Trivial hash-based embeddings for testing."""
    return [[hash(t) % 1000 / 1000.0, len(t) / 100.0] for t in texts]

embeddings = ensure_embeddings(my_embed)
assert isinstance(embeddings, EmbeddingsLambda)

print(embeddings.embed_query("hello"))           # [float, float]
print(len(embeddings.embed_documents(["a","b"])))  # 2
```

### Example 2 — wrap an async embedding function

```python
import asyncio
from langgraph.store.base.embed import ensure_embeddings

async def async_embed(texts: list[str]) -> list[list[float]]:
    await asyncio.sleep(0)   # simulate async API call
    return [[float(len(t))] for t in texts]

embeddings = ensure_embeddings(async_embed)

# sync embed_documents raises ValueError (no sync func available):
try:
    embeddings.embed_documents(["test"])
except ValueError as e:
    print("expected:", e)

# async works fine:
async def main():
    vecs = await embeddings.aembed_documents(["hello", "world"])
    print(vecs)   # [[5.0], [5.0]]

asyncio.run(main())
```

### Example 3 — use with `InMemoryStore` for semantic search

```python
from langgraph.store.memory import InMemoryStore

def my_embed(texts):
    # Real usage: call OpenAI / HuggingFace / SentenceTransformers here
    dim = 4
    return [[float(ord(c)) / 1000 for c in t[:dim].ljust(dim)] for t in texts]

store = InMemoryStore(index={"embed": my_embed, "dims": 4})

# Store some items
store.put(("user", "alice"), "fact_1", {"text": "Alice likes hiking"})
store.put(("user", "alice"), "fact_2", {"text": "Alice works in ML"})
store.put(("user", "alice"), "fact_3", {"text": "Alice owns a dog"})

# Semantic search
results = store.search(("user", "alice"), query="outdoor activities", limit=2)
for item in results:
    print(item.key, "->", item.value["text"])
```

### Example 4 — path-based text extraction with `get_text_at_path`

```python
from langgraph.store.base.embed import get_text_at_path

doc = {
    "title": "My Article",
    "body": {"intro": "Hello world", "sections": ["sec1", "sec2"]},
    "tags": ["ml", "python"],
}

# Simple field path
print(get_text_at_path(doc, "title"))              # ["My Article"]

# Nested path
print(get_text_at_path(doc, "body.intro"))         # ["Hello world"]

# Array wildcard
print(get_text_at_path(doc, "body.sections[*]"))   # ["sec1", "sec2"]

# Multi-field selection
print(get_text_at_path(doc, "{title,body.intro}")) # ["My Article", "Hello world"]

# This is used internally by InMemoryStore.index["fields"] to extract
# text fields from stored values before passing them to the embed function.
```

---

## 10 · `push_ui_message` + `delete_ui_message` — runtime UI streaming

**Module:** `langgraph.graph.ui`

`push_ui_message` lets a node emit a structured UI event that is simultaneously:
1. Streamed to the caller via `stream_mode="custom"` (appears in `run.custom` / the `"custom"` stream)
2. Written to the graph's state under `state_key` (default: `"ui"`) using `ui_message_reducer`

`delete_ui_message` removes a previously emitted component by ID, both from the stream and from state. `ui_message_reducer` merges these updates — including `merge=True` prop patching — and handles the delete-by-ID logic.

### Source signatures

```python
def push_ui_message(
    name: str,
    props: dict[str, Any],
    *,
    id: str | None = None,         # auto-generates UUID if None
    metadata: dict[str, Any] | None = None,
    message: AnyMessage | None = None,   # links UI to a chat message
    state_key: str | None = "ui",        # None to skip state write
    merge: bool = False,           # True → patch existing props instead of replace
) -> UIMessage: ...

def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage: ...

def ui_message_reducer(
    left: list[AnyUIMessage] | AnyUIMessage,
    right: list[AnyUIMessage] | AnyUIMessage,
) -> list[AnyUIMessage]: ...
```

### `UIMessage` TypedDict

```python
class UIMessage(TypedDict):
    type: Literal["ui"]
    id: str
    name: str                # component name (e.g. "ChatBubble", "ProgressBar")
    props: dict[str, Any]    # component props passed to the frontend renderer
    metadata: dict[str, Any] # run_id, tags, name, merge flag, message_id, ...
```

### Example 1 — emit a progress bar UI component from a node

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import push_ui_message, AnyUIMessage, ui_message_reducer

class State(TypedDict):
    result: str
    ui: Annotated[list[AnyUIMessage], ui_message_reducer]

def processing_node(state: State) -> dict:
    # Emit a progress bar at 0%
    bar = push_ui_message(
        name="ProgressBar",
        props={"label": "Processing...", "pct": 0},
    )

    # ... do work ...

    # Update the same component to 100% (merge=True patches props in place)
    push_ui_message(
        name="ProgressBar",
        props={"pct": 100, "label": "Done!"},
        id=bar["id"],
        merge=True,
    )

    return {"result": "finished"}

graph = (
    StateGraph(State)
    .add_node("process", processing_node)
    .add_edge(START, "process")
    .add_edge("process", END)
    .compile()
)

# Stream and observe UI events
for chunk in graph.stream(
    {"result": "", "ui": []},
    stream_mode="custom",
):
    print("UI event:", chunk)
```

### Example 2 — link a UI component to a chat message

```python
from langchain_core.messages import AIMessage
from langgraph.graph.ui import push_ui_message

# Inside a node that also emits an AI message:
def agent_node(state):
    ai_msg = AIMessage(content="Let me calculate that...")

    # The UI component is linked to this specific AI message
    push_ui_message(
        name="Calculator",
        props={"expression": "2 + 2", "result": None},
        message=ai_msg,   # sets metadata["message_id"] = ai_msg.id
    )

    # Later, update with the result
    # Frontend can correlate Calculator component ↔ ai_msg by message_id
    push_ui_message(
        name="Calculator",
        props={"result": 4},
        id="...",    # same id as the first push
        merge=True,
    )

    return {"messages": [ai_msg]}
```

### Example 3 — delete a UI component + `ui_message_reducer` mechanics

```python
from langgraph.graph.ui import ui_message_reducer, UIMessage, RemoveUIMessage

# ui_message_reducer handles adds, in-place prop merging, and removals
existing: list[UIMessage] = [
    {"type": "ui", "id": "cmp-1", "name": "Spinner",
     "props": {"label": "Loading"}, "metadata": {}},
    {"type": "ui", "id": "cmp-2", "name": "Banner",
     "props": {"text": "Hello"}, "metadata": {}},
]

# Merge patch props on cmp-2
after_merge = ui_message_reducer(
    existing,
    {"type": "ui", "id": "cmp-2", "name": "Banner",
     "props": {"text": "Updated!"}, "metadata": {"merge": True}},
)
print(after_merge[1]["props"])  # {"text": "Updated!"}

# Remove cmp-1
after_delete = ui_message_reducer(
    after_merge,
    {"type": "remove-ui", "id": "cmp-1"},
)
print(len(after_delete))        # 1 — cmp-1 is gone
print(after_delete[0]["name"])  # "Banner"
```

### Example 4 — `state_key=None` for stream-only emission (no state write)

```python
from langgraph.graph.ui import push_ui_message

def ephemeral_ui_node(state):
    # Emit to the stream but don't accumulate in state
    # Useful for one-shot loading indicators that shouldn't persist
    push_ui_message(
        name="ToastNotification",
        props={"message": "Saved!", "duration": 3000},
        state_key=None,  # skip the state[ui] write
    )
    return {}
```

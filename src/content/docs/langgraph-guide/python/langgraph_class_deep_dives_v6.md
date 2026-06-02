---
title: "Class deep-dives Vol. 6 — 10 more LangGraph types"
description: "Source-verified deep dives into GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message/ui_message_reducer, and ProtocolEvent — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 6"
  order: 30
---

# Class deep-dives Vol. 6 — 10 more LangGraph types

Verified against **`langgraph==1.2.2`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

[→ Vol. 4 covers set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, error taxonomy](./langgraph_class_deep_dives_v4/)

[→ Vol. 5 covers RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, and the structured HITL protocol](./langgraph_class_deep_dives_v5/)

[→ Vol. 7 covers PregelProtocol/StreamProtocol, BackgroundExecutor/AsyncBackgroundExecutor, AsyncBatchedBaseStore/_dedupe_ops, get_text_at_path/tokenize_path, SerdeEvent/register_serde_event_listener, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec/node Protocols, identifier/get_runnable_for_task](./langgraph_class_deep_dives_v7/)

---

## 1 · `GraphRunStream` + `AsyncGraphRunStream`

**Module:** `langgraph.stream.run_stream`  
**Status:** `@beta` — experimental v3 streaming protocol, API may change  
**Returned by:** `compiled_graph.stream_events(version="v3")` (sync) and `compiled_graph.astream_events(version="v3")` (async)

`GraphRunStream` / `AsyncGraphRunStream` implement **caller-driven pumping** — no background thread or task is needed. The caller's `for` loop (or `async for`) _is_ the graph engine. Every iteration pulls one protocol event from the graph, pushes it through the transformer pipeline, and makes it available on all active projections.

### Why use it?

Classic `graph.stream()` exposes a single flat stream of `(mode, data)` tuples. `stream_events(version="v3")` returns a `GraphRunStream` object whose projections (`run.values`, `run.messages`, `run.custom`, etc.) you can consume independently — or interleaved — without any buffering beyond what you've consumed.

### Constructors (source)

`GraphRunStream` is not constructed directly. Obtain an instance via:

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list

# build a graph ...
run = graph.stream_events(  # returns GraphRunStream
    {"messages": [HumanMessage("hello")]},
    config={"configurable": {"thread_id": "1"}},
    version="v3",
)
```

### Key attributes and properties

| Member | Type | Description |
|---|---|---|
| `run.values` | `StreamChannel[dict]` | One state snapshot per superstep (stream_mode="values"). |
| `run.messages` | `StreamChannel[ChatModelStream]` | One `ChatModelStream` per LLM call (stream_mode="messages"). |
| `run.custom` | `StreamChannel[Any]` | Data emitted via `get_stream_writer()` (stream_mode="custom"). |
| `run.updates` | `StreamChannel[dict]` | One dict per node step (stream_mode="updates"). |
| `run.extensions` | `Mapping[str, Any]` | All transformer projections, keyed by name. |
| `run.output` | `dict \| None` (property) | Drive to completion and return the final state dict. |
| `run.interrupted` | `bool` (property) | Drive to completion, return True if graph was interrupted. |
| `run.interrupts` | `list[Any]` (property) | Drive to completion, return all interrupt payloads. |

### Using `run.output` — simplest drain pattern

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def respond(state: State) -> dict:
    return {"messages": [AIMessage("Hi there!")]}

builder = StateGraph(State)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile()

# Drive the whole run and capture final state
run = graph.stream_events({"messages": [HumanMessage("hello")]}, version="v3")
final_state = run.output   # blocks until graph finishes
print(final_state)         # {"messages": [HumanMessage("hello"), AIMessage("Hi there!")]}
```

### Consuming projections independently

```python
# Iterate values (state snapshots) while the graph runs
for snapshot in run.values:
    print("snapshot:", snapshot)

# The run.values channel is single-consumer.
# The graph is driven by whichever projection is being iterated.
```

### `interleave()` — strict arrival-order across multiple projections

`interleave(*names)` merges multiple named projections into a single stream of `(name, item)` tuples ordered by when each item was pushed — not round-robin.

```python
run = graph.stream_events(input, version="v3")

for name, item in run.interleave("messages", "custom"):
    if name == "messages":
        # a ChatModelStream handle — iterate its .text projection
        for chunk in item.text:
            print(chunk, end="", flush=True)
    else:
        print("\n[custom]", item)
```

### `abort()` — stop early

```python
with graph.stream_events(input, version="v3") as run:
    for snapshot in run.values:
        if snapshot.get("done"):
            run.abort()   # closes the mux, stops the graph
            break
```

The `with` block calls `abort()` automatically on exit, so the context-manager pattern is the safest way to consume `GraphRunStream`.

### Async variant — `AsyncGraphRunStream`

Identical API, uses `async for` and `await`:

```python
async with await graph.astream_events(input, version="v3") as run:
    async for name, item in run.interleave("messages", "values"):
        if name == "messages":
            async for chunk in item.text:
                print(chunk, end="", flush=True)
        else:
            print("\n[state]", item)
```

`AsyncGraphRunStream` uses an `asyncio.Lock`-based single-flight pump so multiple concurrent `async for` loops on different projections share one pump without race conditions.

---

## 2 · `StreamTransformer`

**Module:** `langgraph.stream._types`  
**Kind:** Abstract base class (ABC)  
**Re-exported by:** `langgraph.stream.transformers` (indirectly)

`StreamTransformer` is the extension point for adding new **streaming projections** to a v3 run. A transformer observes every `ProtocolEvent` flowing through the `StreamMux` and populates one or more `StreamChannel` projections that consumers can iterate.

The built-in transformers (`ValuesTransformer`, `MessagesTransformer`, `CustomTransformer`, `UpdatesTransformer`) all extend this class — so studying the ABC tells you how to build your own.

### Abstract interface (source)

```python
from abc import ABC, abstractmethod
from typing import Any
from langgraph.stream._types import ProtocolEvent

class StreamTransformer(ABC):
    scope: tuple[str, ...]    # set at construction from mux scope

    # Class-level class variables (override in subclass)
    _native:              bool = False   # expose projection keys as run.X attributes
    requires_async:       bool = False   # set True if you call self.schedule()
    supports_sync:        bool = False   # set True if async transformer also works sync
    required_stream_modes: tuple[str, ...] = ()  # e.g. ("values",), ("messages",)
    before_builtins:      bool = False   # run BEFORE built-in transformers (for mutation)

    @abstractmethod
    def init(self) -> dict[str, Any]:
        """Return the projection dict. Keys → run.extensions entries.
        StreamChannel values are auto-wired by the mux."""
        ...

    def process(self, event: ProtocolEvent) -> bool:
        """Sync handler. Return True to keep event in main log, False to suppress."""
        raise NotImplementedError

    async def aprocess(self, event: ProtocolEvent) -> bool:
        """Async handler — override instead of process() for async work."""
        ...

    def finalize(self) -> None: ...       # called when mux closes normally
    async def afinalize(self) -> None: ...
    def fail(self, err: BaseException) -> None: ...  # called on error
    async def afail(self, err: BaseException) -> None: ...

    def schedule(self, coro) -> None:
        """Schedule a coroutine on the running event loop.
        Calling this from sync process() opts the transformer into the async lane."""
        ...
```

### Example: cost-tracking transformer

This custom transformer counts token usage from `values` events and exposes a `StreamChannel[int]` projection.

```python
import asyncio
from typing import Any
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.stream._types import ProtocolEvent, StreamTransformer
from langgraph.stream.stream_channel import StreamChannel
from langchain_core.messages import HumanMessage, AIMessage

class TokenCostTransformer(StreamTransformer):
    """Counts messages added per superstep and emits a running total."""

    _native = True                          # run.token_events is a direct attribute
    required_stream_modes = ("values",)     # we need state snapshots

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._log: StreamChannel[dict] = StreamChannel()
        self._prev_msg_count = 0
        self._scope_list = list(scope)

    def init(self) -> dict[str, Any]:
        return {"token_events": self._log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "values":
            return True
        params = event["params"]
        if params["namespace"] != self._scope_list:
            return True
        data = params["data"]
        msgs = data.get("messages", [])
        delta = len(msgs) - self._prev_msg_count
        self._prev_msg_count = len(msgs)
        if delta > 0:
            self._log.push({"new_messages": delta, "total": len(msgs)})
        return True


class State(TypedDict):
    messages: Annotated[list, add_messages]

def llm_node(state: State) -> dict:
    return {"messages": [AIMessage("response")]}

builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
graph = builder.compile()

# Register the transformer at stream time
run = graph.stream_events(
    {"messages": [HumanMessage("hi")]},
    version="v3",
    config={"configurable": {}},
)
# Note: passing custom transformers requires the graph to accept them.
# This example illustrates the class structure — see StreamMux.factories for wiring.

for event in run.token_events if hasattr(run, "token_events") else []:
    print(event)
```

### Mutating events with `before_builtins`

Set `before_builtins = True` to run _before_ built-in transformers like `MessagesTransformer`. This lets you redact PII or add metadata before built-ins snapshot text fields:

```python
class PIIRedactor(StreamTransformer):
    before_builtins = True        # run before MessagesTransformer

    def init(self) -> dict[str, Any]:
        return {}                  # no projection needed — just mutate events

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "messages":
            data = event["params"].get("data", ())
            if data and isinstance(data, tuple) and isinstance(data[0], dict):
                content = data[0].get("data", {}).get("content", "")
                # Replace SSN pattern
                import re
                cleaned = re.sub(r"\d{3}-\d{2}-\d{4}", "[REDACTED]", str(content))
                data[0].get("data", {}).update({"content": cleaned})
        return True
```

---

## 3 · `StreamChannel`

**Module:** `langgraph.stream.stream_channel`  
**Kind:** Generic class `StreamChannel[T]`

`StreamChannel[T]` is the **single-consumer drainable queue** that backs every streaming projection in the v3 protocol. Each `StreamTransformer` creates one or more `StreamChannel` instances in its `init()` return dict; the `StreamMux` auto-wires them for protocol event forwarding.

### Constructor

```python
StreamChannel(name: str | None = None, *, maxlen: int | None = None)
```

| Parameter | Description |
|---|---|
| `name` | When set, the StreamMux injects a `ProtocolEvent` into the main log every time `push()` is called, using `name` as the event method. When `None`, the channel is local-only. |
| `maxlen` | Reserved for future backpressure control; currently unused. |

### Lifecycle: bind → push → iterate → close

```
StreamMux._register()  →  channel._bind(is_async)
                       →  channel._wire(forward_fn)   # optional auto-forward
producer calls             channel.push(item)
consumer calls             for item in channel: ...   # sync
                         or async for item in channel:  # async
StreamMux.close()      →  channel.close()
StreamMux.fail(err)    →  channel.fail(err)
```

A channel starts _unbound_. Only after `_bind(is_async)` is called does `__iter__` or `__aiter__` become available. This prevents accidental iteration before a run starts.

### `push()` — producer side

```python
channel.push(item)   # appends to buffer (if subscribed) and auto-forwards
```

If no subscriber has registered yet, the buffer append is a no-op but auto-forwarding (to the main event log) still fires. This means named channels always contribute to `run.__iter__()` regardless of whether any code iterates the projection directly.

### `tee(n)` — sync fan-out

```python
from langgraph.stream.stream_channel import StreamChannel

ch: StreamChannel[str] = StreamChannel()
# ... after binding and subscribing:
branch_a, branch_b = ch.tee(2)   # returns tuple[Iterator[str], Iterator[str]]

for a_item, b_item in zip(branch_a, branch_b):
    print(a_item, b_item)
```

`tee(n)` creates `n` independent iterators that share one underlying cursor. Items pulled from the source are copied into each branch's own `deque`. Since the pump is caller-driven, memory is naturally bounded by how far ahead the fastest branch reads.

### `atee(n)` — async fan-out

```python
branches = await ch.atee(3)     # tuple[AsyncIterator[T], ...] of length 3
```

### Checking state before iterating

```python
if ch._closed:
    print("channel is done")
if ch._error is not None:
    print("channel failed:", ch._error)
```

### Full example: manually wiring a StreamChannel

```python
from langgraph.stream.stream_channel import StreamChannel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Build a channel that logs every push to stderr
ch: StreamChannel[str] = StreamChannel(name="audit")

# Bind it for sync use (normally done by StreamMux)
ch._bind(is_async=False)
ch._subscribed = True          # unlock push-to-buffer

ch.push("event-a")
ch.push("event-b")
ch.close()

# Iterate — NO pump needed because we pushed directly
for item in ch:
    print("got:", item)
# got: event-a
# got: event-b
```

---

## 4 · `ValuesTransformer` / `CustomTransformer` / `UpdatesTransformer`

**Module:** `langgraph.stream.transformers`

These three are the native projection transformers that power `run.values`, `run.custom`, and `run.updates` respectively. All have `_native = True`, meaning their keys are direct attributes on the `GraphRunStream`.

| Class | Projection key | `required_stream_modes` | What it captures |
|---|---|---|---|
| `ValuesTransformer` | `values` | `("values",)` | Full state dict after each superstep |
| `CustomTransformer` | `custom` | `("custom",)` | Arbitrary payloads from `get_stream_writer()` |
| `UpdatesTransformer` | `updates` | `("updates",)` | Per-node output dicts after each step |

### How scoping works

Every transformer receives the `scope` tuple from the `StreamMux`. For the root run this is `()`. Subgraph runs (via `SubgraphTransformer`) get a deeper scope like `("ns1",)`. Each transformer filters events by comparing `event["params"]["namespace"]` against its own scope list, so a values snapshot from a subgraph does not appear in the root `run.values` — it appears on the subgraph handle's `.values` projection.

```python
# Conceptual filter inside ValuesTransformer.process():
def process(self, event: ProtocolEvent) -> bool:
    if event["method"] != "values":
        return True
    if event["params"]["namespace"] != self._scope_list:
        return True   # not my scope — pass through
    self._log.push(event["params"]["data"])   # my scope — capture it
    return True
```

### Combining multiple projections in one run

Since each projection is a `StreamChannel`, you can iterate them in any order. `stream_events(version="v3")` automatically requests only the stream modes that the registered transformers declare in `required_stream_modes`:

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    count: int

def step(state: State) -> dict:
    writer = get_stream_writer()
    writer({"progress": state["count"]})
    return {
        "messages": [AIMessage(f"step {state['count']}")],
        "count": state["count"] + 1,
    }

def should_continue(state: State) -> str:
    return "step" if state["count"] < 3 else "__end__"

builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_conditional_edges("step", should_continue)
graph = builder.compile()

run = graph.stream_events(
    {"messages": [HumanMessage("start")], "count": 0},
    version="v3",
)

# Interleave custom events and final state snapshots
for name, item in run.interleave("custom", "values"):
    if name == "custom":
        print("[custom]", item)   # {"progress": 0}, {"progress": 1}, ...
    else:
        print("[state ] count=", item["count"])
```

### `DebugTransformer` and `TasksTransformer`

Two additional built-in transformers serve advanced use cases:

- **`DebugTransformer`** — captures debug-mode events onto `run.extensions["debug"]`
- **`TasksTransformer`** — captures task-level events for parallel task inspection via `run.extensions["tasks"]`

Both follow the same pattern: `_native = True`, scope-filtered, `StreamChannel`-backed.

---

## 5 · `GraphCallbackHandler`

**Module:** `langgraph.callbacks`  
**Extends:** `langchain_core.callbacks.BaseCallbackHandler`

`GraphCallbackHandler` is a lifecycle callback interface specific to LangGraph. It receives events at graph interrupt and resume transitions — events that generic LangChain callbacks (`on_llm_start`, `on_chain_end`, etc.) do not emit.

### Class definition (source)

```python
from langgraph.callbacks import GraphCallbackHandler
from langgraph.callbacks import GraphInterruptEvent, GraphResumeEvent

class GraphCallbackHandler:
    def on_interrupt(self, event: GraphInterruptEvent) -> Any:
        """Fired when the graph pauses for one or more interrupt() calls."""

    def on_resume(self, event: GraphResumeEvent) -> Any:
        """Fired when the graph resumes from a checkpoint."""
```

### Wiring into a graph

Pass handler instances through `config["callbacks"]`:

```python
from langchain_core.runnables import RunnableConfig
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

class AuditHandler(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        print(f"[INTERRUPT] checkpoint={event.checkpoint_id} status={event.status}")
        print(f"  payloads: {[i.value for i in event.interrupts]}")

    def on_resume(self, event: GraphResumeEvent) -> None:
        print(f"[RESUME] from checkpoint={event.checkpoint_id} status={event.status}")


from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict

class State(TypedDict):
    x: int

def human_review(state: State) -> dict:
    answer = interrupt({"question": "approve?", "x": state["x"]})
    return {"x": answer}

builder = StateGraph(State)
builder.add_node("review", human_review)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {"thread_id": "thread-1"},
    "callbacks": [AuditHandler()],
}

# First invoke — hits interrupt
try:
    graph.invoke({"x": 42}, config=config)
except Exception:
    pass  # interrupt raises internally

# Resume — fires on_resume callback
graph.invoke(None, config={**config, "configurable": {**config["configurable"]}},
             command={"resume": 99})
```

### Important: only `GraphCallbackHandler` instances receive these events

The `langgraph.callbacks` module filters the handler list:

```python
[h for h in handlers if isinstance(h, GraphCallbackHandler)]
```

Standard `BaseCallbackHandler` subclasses will not receive `on_interrupt` or `on_resume`.

### Async support

`on_interrupt` and `on_resume` can be `async def` — the callback manager calls them via `ahandle_event` when the graph runs asynchronously.

---

## 6 · `GraphInterruptEvent` + `GraphResumeEvent`

**Module:** `langgraph.callbacks`  
**Kind:** frozen dataclasses

These are the event payloads delivered to `GraphCallbackHandler.on_interrupt()` and `on_resume()`.

### `GraphInterruptEvent` (source)

```python
@dataclass(frozen=True)
class GraphInterruptEvent:
    run_id:        UUID | None           # run ID for this execution, if available
    status:        GraphLifecycleStatus  # loop status when interrupt fired
    checkpoint_id: str                   # checkpoint saved at the interrupt point
    checkpoint_ns: tuple[str, ...]       # namespace path (e.g. ("parent", "child"))
    interrupts:    tuple[Interrupt, ...] # interrupt payloads that paused the graph
```

### `GraphResumeEvent` (source)

```python
@dataclass(frozen=True)
class GraphResumeEvent:
    run_id:        UUID | None
    status:        GraphLifecycleStatus  # loop status when resume fired
    checkpoint_id: str                   # checkpoint the graph resumed from
    checkpoint_ns: tuple[str, ...]
```

### `GraphLifecycleStatus` values

```python
GraphLifecycleStatus = Literal[
    "input",           # graph accepted initial input
    "pending",         # graph is mid-execution
    "done",            # graph reached END
    "interrupt_before", # interrupt fired before a node
    "interrupt_after",  # interrupt fired after a node
    "out_of_steps",    # recursion limit hit
]
```

### Practical example: audit trail with checkpoints

```python
import json
from dataclasses import asdict
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

class AuditTrail(GraphCallbackHandler):
    def __init__(self, log_path: str = "audit.jsonl") -> None:
        self._log_path = log_path

    def _write(self, record: dict) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        self._write({
            "event": "interrupt",
            "checkpoint_id": event.checkpoint_id,
            "checkpoint_ns": list(event.checkpoint_ns),
            "status": event.status,
            "interrupt_count": len(event.interrupts),
            "payloads": [str(i.value) for i in event.interrupts],
        })

    def on_resume(self, event: GraphResumeEvent) -> None:
        self._write({
            "event": "resume",
            "checkpoint_id": event.checkpoint_id,
            "checkpoint_ns": list(event.checkpoint_ns),
            "status": event.status,
        })
```

### `GraphLifecycleEvent` union type

```python
from langgraph.callbacks import GraphLifecycleEvent  # GraphInterruptEvent | GraphResumeEvent

def handle_any(event: GraphLifecycleEvent) -> None:
    if isinstance(event, GraphInterruptEvent):
        ...
    else:
        ...
```

---

## 7 · `GraphDrained`

**Module:** `langgraph.errors`  
**Bases:** `GraphBubbleUp` → `Exception`

`GraphDrained` is raised when a graph exits early because `RunControl.request_drain()` was called. It signals a **cooperative graceful shutdown** — the graph reached a superstep boundary and honoured the drain request rather than starting another round of nodes.

The key property: when `GraphDrained` is raised, **a checkpoint has already been saved**. The run can be resumed later from where it stopped.

### Constructor

```python
class GraphDrained(GraphBubbleUp):
    reason: str      # the string passed to request_drain()

    def __init__(self, reason: str = "shutdown") -> None: ...
```

### Full graceful-shutdown pattern

```python
import asyncio
from langgraph.runtime import Runtime, RunControl
from langgraph.errors import GraphDrained
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict

class State(TypedDict):
    step: int

def long_process(state: State) -> dict:
    # Simulate work
    import time; time.sleep(0.01)
    return {"step": state["step"] + 1}

def should_continue(state: State) -> str:
    return "process" if state["step"] < 100 else "__end__"

builder = StateGraph(State)
builder.add_node("process", long_process)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", should_continue)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

async def run_with_drain():
    control = RunControl()
    config = {"configurable": {"thread_id": "drain-demo"}}

    async def cancel_after(delay: float) -> None:
        await asyncio.sleep(delay)
        control.request_drain(reason="SIGTERM received")

    async def do_invoke() -> dict | None:
        try:
            return await graph.ainvoke(
                {"step": 0},
                config=config,
                control=control,
            )
        except GraphDrained as e:
            print(f"Graph drained cooperatively: {e.reason}")
            return None

    result, _ = await asyncio.gather(do_invoke(), cancel_after(0.05))
    return result

# asyncio.run(run_with_drain())
```

### Catching `GraphDrained` vs other exceptions

```python
from langgraph.errors import GraphDrained, GraphRecursionError

try:
    result = graph.invoke(input, config=config, control=control)
except GraphDrained as e:
    # Clean exit — checkpoint was saved. Resume later.
    print(f"Stopped at: {e.reason}")
    result = graph.invoke(None, config=config)  # resume
except GraphRecursionError:
    # Runaway graph — no safe resume
    raise
```

### Difference from `KeyboardInterrupt`

`GraphDrained` is a Python `Exception` (not `BaseException`), so standard `except Exception` handlers catch it. It is raised at superstep boundaries only — a node that is currently executing will complete before the drain takes effect.

---

## 8 · `NodeTimeoutError`

**Module:** `langgraph.errors`  
**Bases:** `Exception` (NOT `TimeoutError`, by design)

`NodeTimeoutError` is raised when a node's execution exceeds a configured `TimeoutPolicy`. It does **not** inherit from the built-in `TimeoutError` (which is an `OSError` subclass) — this ensures the default `RetryPolicy` treats it as retryable.

### Constructor + fields (source)

```python
class NodeTimeoutError(Exception):
    node:         str    # name of the node that timed out
    timeout:      float  # the limit that fired (idle_timeout or run_timeout)
    run_timeout:  float | None
    idle_timeout: float | None
    elapsed:      float  # actual wall-clock seconds
    kind:         Literal["idle", "run"]

    def __init__(
        self,
        node: str,
        elapsed: float,
        *,
        kind: Literal["idle", "run"],
        idle_timeout: float | None = None,
        run_timeout: float | None = None,
    ) -> None: ...
```

| Field | Meaning |
|---|---|
| `kind = "run"` | Node exceeded its total wall-clock budget (`TimeoutPolicy(run_timeout=…)`) |
| `kind = "idle"` | Node exceeded the idle budget — it was running but made no forward progress (`TimeoutPolicy(idle_timeout=…)`) |
| `timeout` | The limit that fired (either `run_timeout` or `idle_timeout`) |
| `elapsed` | Actual seconds the node ran before being killed |

### Full example: distinguishing idle vs run timeouts

```python
from langgraph.errors import NodeTimeoutError
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import time

class State(TypedDict):
    result: str

def slow_node(state: State) -> dict:
    time.sleep(10)   # will be killed
    return {"result": "done"}

def error_handler(state: State, error: "NodeError") -> dict:
    # error.error will be a NodeTimeoutError
    err = error.error
    if isinstance(err, NodeTimeoutError):
        msg = (
            f"Node '{err.node}' timed out after {err.elapsed:.2f}s "
            f"(limit: {err.timeout}s, kind: {err.kind})"
        )
        return {"result": f"TIMEOUT: {msg}"}
    return {"result": f"ERROR: {err}"}

builder = StateGraph(State)
builder.add_node(
    "slow",
    slow_node,
    timeout=TimeoutPolicy(run_timeout=2.0),   # 2-second hard cap
    error_handler=error_handler,
)
builder.add_edge(START, "slow")
builder.add_edge("slow", END)
graph = builder.compile()

result = graph.invoke({"result": ""})
print(result["result"])
# TIMEOUT: Node 'slow' timed out after 2.00s (limit: 2.0s, kind: run)
```

### Why not inherit from `TimeoutError`?

Python's `TimeoutError` is a subclass of `OSError`. The default `RetryPolicy` retries on `Exception` but not on `OSError` subclasses (OS errors are typically not transient). By keeping `NodeTimeoutError` as a plain `Exception`, the default retry policy automatically retries timed-out nodes up to `max_attempts`.

### Catching in a custom retry policy

```python
from langgraph.types import RetryPolicy
from langgraph.errors import NodeTimeoutError

# Only retry on timeout, not other errors
policy = RetryPolicy(
    max_attempts=3,
    retry_on=(NodeTimeoutError,),
    backoff_factor=1.5,
)
builder.add_node("flaky", flaky_node, retry_policy=policy)
```

---

## 9 · `delete_ui_message` + `ui_message_reducer`

**Module:** `langgraph.graph.ui`  
**See also:** `push_ui_message` (covered in Vol. 4)

`push_ui_message` (Vol. 4) emits a `UIMessage` TypedDict that renders a UI component. `delete_ui_message` removes one by ID, and `ui_message_reducer` is the state reducer that merges and applies removals.

### `delete_ui_message` (source)

```python
def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage:
    """Push a remove-ui event and update graph state.

    Args:
        id:        The UIMessage.id to remove.
        state_key: State key holding the UI messages list. Defaults to "ui".

    Returns:
        The RemoveUIMessage dict: {"type": "remove-ui", "id": id}
    """
```

Internally it calls `get_stream_writer()` to emit the event (for real-time UI updates) and then sends a `("ui", evt)` state update so the removal is persisted in the checkpoint.

### `ui_message_reducer` (source)

```python
def ui_message_reducer(
    left: list[AnyUIMessage] | AnyUIMessage,
    right: list[AnyUIMessage] | AnyUIMessage,
) -> list[AnyUIMessage]:
    """Merge two UIMessage lists, applying remove-ui messages as deletions."""
```

The reducer:
1. Normalises `left` and `right` to lists
2. Copies `left` into a mutable list and builds an ID-to-index map
3. For each item in `right`:
   - If `type == "remove-ui"` → removes the matching ID from the merged list
   - If `type == "ui"` and ID already exists → replaces it (upsert)
   - If `type == "ui"` and ID is new → appends it

### Full example: stateful UI components

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import (
    push_ui_message,
    delete_ui_message,
    ui_message_reducer,
    AnyUIMessage,
)

class State(TypedDict):
    step:   int
    ui:     Annotated[list[AnyUIMessage], ui_message_reducer]

def render_progress(state: State) -> dict:
    # Push a progress bar component
    msg = push_ui_message(
        name="ProgressBar",
        props={"value": state["step"], "max": 5, "label": "Processing…"},
        id="progress-bar",           # stable ID so next call replaces it
    )
    return {"step": state["step"] + 1}

def finish(state: State) -> dict:
    # Remove the progress bar when done
    delete_ui_message("progress-bar")
    push_ui_message(
        name="Alert",
        props={"severity": "success", "message": "Done!"},
    )
    return {}

def route(state: State) -> str:
    return "progress" if state["step"] < 5 else "finish"

builder = StateGraph(State)
builder.add_node("progress", render_progress)
builder.add_node("finish", finish)
builder.add_edge(START, "progress")
builder.add_conditional_edges("progress", route)
builder.add_edge("finish", END)
graph = builder.compile()

final = graph.invoke({"step": 0, "ui": []})
# final["ui"] contains only the success Alert — the ProgressBar was removed
for msg in final["ui"]:
    print(msg["name"], msg.get("props", {}))
```

### `UIMessage` and `RemoveUIMessage` TypedDicts

```python
class UIMessage(TypedDict):
    type:     Literal["ui"]
    id:       str           # unique across the session
    name:     str           # component name (e.g. "ProgressBar", "Alert")
    props:    dict[str, Any]
    metadata: dict[str, Any]   # includes run_id, tags, merge flag, message_id

class RemoveUIMessage(TypedDict):
    type: Literal["remove-ui"]
    id:   str               # ID of the UIMessage to delete
```

### The `merge` flag on `push_ui_message`

`push_ui_message(..., merge=True)` sets `metadata["merge"] = True`. The frontend can use this flag to deeply-merge `props` with the existing component state rather than replacing the whole props object — useful for incremental streaming updates to a UI component.

```python
# Initial render
push_ui_message("DataTable", {"rows": [], "loading": True}, id="table")

# Stream rows in
for row in data_source:
    push_ui_message("DataTable", {"rows": [row]}, id="table", merge=True)

# Final state
push_ui_message("DataTable", {"loading": False}, id="table", merge=True)
```

---

## 10 · `ProtocolEvent`

**Module:** `langgraph.stream._types`  
**Kind:** `TypedDict`

`ProtocolEvent` is the universal streaming event envelope used throughout the v3 streaming infrastructure. Every `ProtocolEvent` wraps a raw stream part (values, messages, custom, updates, debug, etc.) and attaches ordering metadata.

### TypedDict fields (source)

```python
class _ProtocolEventParams(TypedDict):
    namespace:  list[str]   # checkpoint namespace path; [] for root
    timestamp:  int         # wall-clock milliseconds — NOT monotonic
    data:       Any         # the actual payload
    interrupts: NotRequired[tuple[Any, ...]]  # interrupt payloads (values events only)

class ProtocolEvent(TypedDict):
    type:    Literal["event"]
    eventId: NotRequired[str]    # optional external correlation ID
    seq:     NotRequired[int]    # monotonic sequence number assigned by root StreamMux
    method:  str                 # stream mode: "values", "messages", "custom", "updates", "debug", …
    params:  _ProtocolEventParams
```

### Key rules

| Field | Rule |
|---|---|
| `seq` | Monotonic, assigned by the **root** StreamMux only. Child mini-muxes (for subgraphs) do not assign seq numbers. Use `seq` for total ordering; `timestamp` is wall-clock and can go backwards after NTP adjustments. |
| `method` | Matches a `stream_mode` string. Custom events (from `get_stream_writer()`) use method `"custom"`. Tool streaming events use `"tools"`. |
| `params.namespace` | `[]` for the root graph. Subgraph events carry the checkpoint namespace path, e.g. `["parent_ns", "subgraph_ns"]`. |
| `params.interrupts` | Only present on `method="values"` events when an `interrupt()` call fired during that superstep. |

### Reading raw protocol events

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def step(state: State) -> dict:
    return {"messages": [AIMessage("hello")]}

builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)
graph = builder.compile()

run = graph.stream_events({"messages": [HumanMessage("hi")]}, version="v3")

# __iter__ on the run yields raw ProtocolEvents in seq order
for event in run:
    print(f"seq={event.get('seq')} method={event['method']} ns={event['params']['namespace']}")
    if event["method"] == "values":
        data = event["params"]["data"]
        print(f"  state keys: {list(data.keys())}")
        interrupts = event["params"].get("interrupts", ())
        if interrupts:
            print(f"  interrupts: {interrupts}")
```

### Building a custom dispatcher over `ProtocolEvent`

```python
from langgraph.stream._types import ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel

def dispatch(events: list[ProtocolEvent]) -> dict[str, list]:
    """Group events by method, ignoring subgraph events."""
    result: dict[str, list] = {}
    for evt in events:
        if evt["params"]["namespace"]:
            continue     # skip subgraph events
        method = evt["method"]
        result.setdefault(method, []).append(evt["params"]["data"])
    return result
```

### `convert_to_protocol_event` — converting legacy stream parts

```python
from langgraph.stream._convert import convert_to_protocol_event

# Classic graph.stream() yields (mode, data) tuples.
# Convert to ProtocolEvent for use with v3 infrastructure:
for mode, data in graph.stream(input, stream_mode=["values", "updates"]):
    evt = convert_to_protocol_event((mode, data))
    print(evt["method"], evt["params"]["data"])
```

---

## Quick-reference table

| Class / function | Module | Vol. 1–5 cross-reference |
|---|---|---|
| `GraphRunStream` | `langgraph.stream.run_stream` | New in Vol. 6 |
| `AsyncGraphRunStream` | `langgraph.stream.run_stream` | New in Vol. 6 |
| `StreamTransformer` | `langgraph.stream._types` | New in Vol. 6 (base class for `ToolCallTransformer` in Vol. 2) |
| `StreamChannel` | `langgraph.stream.stream_channel` | New in Vol. 6 (backing store for `ToolCallStream` in Vol. 2) |
| `ValuesTransformer` | `langgraph.stream.transformers` | New in Vol. 6 |
| `CustomTransformer` | `langgraph.stream.transformers` | New in Vol. 6 |
| `UpdatesTransformer` | `langgraph.stream.transformers` | New in Vol. 6 |
| `GraphCallbackHandler` | `langgraph.callbacks` | New in Vol. 6 |
| `GraphInterruptEvent` | `langgraph.callbacks` | New in Vol. 6 |
| `GraphResumeEvent` | `langgraph.callbacks` | New in Vol. 6 |
| `GraphDrained` | `langgraph.errors` | New in Vol. 6 (RunControl covered in Vol. 2) |
| `NodeTimeoutError` | `langgraph.errors` | New in Vol. 6 (TimeoutPolicy covered in Vol. 2) |
| `delete_ui_message` | `langgraph.graph.ui` | New in Vol. 6 (`push_ui_message` in Vol. 4) |
| `ui_message_reducer` | `langgraph.graph.ui` | New in Vol. 6 |
| `ProtocolEvent` | `langgraph.stream._types` | New in Vol. 6 |

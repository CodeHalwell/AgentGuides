---
title: "LangGraph Class Deep-Dives Vol. 36"
description: "Source-verified deep dives into 10 class groups from langgraph==1.2.8 — BinaryOperatorAggregate/Overwrite (custom reducer channels with force-overwrite semantics), Topic (multi-value PubSub channel with optional accumulation), RetryPolicy/TimeoutPolicy/CachePolicy (node execution policy trio), ChannelWrite/ChannelWriteEntry/ChannelWriteTupleEntry (node output routing internals), PregelProtocol/StreamProtocol (graph ABC and stream callback contract), add_messages/MessagesState/push_message/REMOVE_ALL_MESSAGES (canonical message reducer and imperative write API), EmbeddingsLambda/ensure_embeddings/get_text_at_path/tokenize_path (store embedding bridge and path extraction), InMemoryStore (built-in KV+vector store), AsyncBatchedBaseStore (async background-queue store pattern), and call()/SyncAsyncFuture/PregelScratchpad (task scheduling infrastructure)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 36"
  order: 67
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.8` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `BinaryOperatorAggregate` · `Overwrite`

**Module:** `langgraph.channels.binop`

`BinaryOperatorAggregate` is the channel type behind `Annotated[int, operator.add]`-style state fields. Each superstep it applies a binary operator to accumulate incoming writes on top of the existing value. The `Overwrite` dataclass lets any single write skip accumulation and force-set the channel to an exact value instead — useful for resets or authoritative updates that must not be merged.

**Key source facts** (from `langgraph/channels/binop.py`):

- `BinaryOperatorAggregate(typ, operator)` — `typ` is the value type; special-cased for `Sequence`/`Set`/`Mapping` ABC aliases to their concrete counterparts.
- `update(values)` — applies `operator(current, v)` for each `v` in `values`; but if any value is an `Overwrite` (or the sentinel dict forms `{"__overwrite__": v}` / `{"value": …, "type": "__overwrite__"}` for JSON round-trips), it force-sets `self.value` and sets `seen_overwrite = True`. A second `Overwrite` in the same step raises `InvalidUpdateError`.
- `checkpoint() / from_checkpoint()` — standard channel serde; restores to `MISSING` when absent, triggering `EmptyChannelError` on `get()`.
- `_operators_equal(a, b)` — treats any lambda as equal to anything (lambda `__name__` is always `"<lambda>"`), so channel equality is stable under `==` checks.
- `Overwrite(value)` dataclass from `langgraph.types` — `@dataclass(slots=True)` with a single `value` field.

### Example 1 — running total with `operator.add`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    total: Annotated[int, operator.add]  # BinaryOperatorAggregate under the hood

def add_five(state: State) -> dict:
    return {"total": 5}

def add_ten(state: State) -> dict:
    return {"total": 10}

builder = StateGraph(State)
builder.add_node("a", add_five)
builder.add_node("b", add_ten)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"total": 0})
print(result)  # {'total': 15}
```

### Example 2 — force-reset with `Overwrite`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class State(TypedDict):
    score: Annotated[int, operator.add]

def increment(state: State) -> dict:
    return {"score": 1}

def reset_score(state: State) -> dict:
    # Overwrite skips accumulation and force-sets score to 0
    return {"score": Overwrite(0)}

builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_node("reset", reset_score)
builder.add_edge(START, "inc")
builder.add_edge("inc", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
result = graph.invoke({"score": 100})
print(result)  # {'score': 0}  — reset wins over existing 100
```

### Example 3 — JSON boundary round-trip via sentinel dict form

```python
import operator
import json
from langgraph.channels.binop import BinaryOperatorAggregate

# Simulate what happens when an Overwrite is JSON-serialised and deserialised
channel = BinaryOperatorAggregate(int, operator.add)
channel.value = 42

# The JSON-serialised form that arrives from an API boundary
json_overwrite = json.loads('{"value": 99, "type": "__overwrite__"}')

# BinaryOperatorAggregate.update() detects the sentinel dict form
channel.update([json_overwrite])
print(channel.get())  # 99 — overwrite applied correctly across JSON boundary

# A plain dict without the sentinel is treated as a regular value
# (would fail here since int+dict raises TypeError — sentinel detection is critical)
```

---

## 2 · `Topic`

**Module:** `langgraph.channels.topic`

`Topic` is a configurable PubSub-style channel that **collects all values written to it in a superstep** into a list. Unlike `LastValue` (which keeps only the latest) or `BinaryOperatorAggregate` (which folds), `Topic` exposes all received writes as a sequence. The `accumulate` flag controls whether collected values persist across supersteps or are cleared after each one.

**Key source facts** (from `langgraph/channels/topic.py`):

- `Topic(typ, accumulate=False)` — with `accumulate=False` (the default), `update()` clears `self.values` before appending new writes — making it a "this superstep's events" bag. With `accumulate=True`, values grow across all supersteps until reset.
- `update(values)` — flattens each write: if a write is a `list`, its elements are yielded individually; otherwise the value itself is yielded. So nodes can write a single item **or** a batch list.
- `UpdateType = typ | list[typ]` — a node can write `"event"` or `["e1", "e2"]` interchangeably.
- `get()` raises `EmptyChannelError` when `self.values` is empty, so edges that depend on this channel are skipped in empty supersteps.
- `checkpoint()` returns `list[Value]`; `from_checkpoint()` handles backward-compat tuple form from older serialisers.

### Example 1 — collecting events from parallel nodes

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    events: Annotated[Sequence[str], Topic(str)]

def emit_a(state: State) -> dict:
    return {"events": "node_a_ran"}

def emit_b(state: State) -> dict:
    return {"events": "node_b_ran"}

def summarize(state: State) -> dict:
    print("Events this step:", state["events"])
    return {}

builder = StateGraph(State)
builder.add_node("a", emit_a)
builder.add_node("b", emit_b)
builder.add_node("summarize", summarize)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", "summarize")
builder.add_edge("b", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
graph.invoke({"events": []})
# Events this step: ['node_a_ran', 'node_b_ran']
```

### Example 2 — writing a batch list from a single node

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    log: Annotated[Sequence[str], Topic(str)]

def multi_emit(state: State) -> dict:
    # Writing a list — Topic flattens it into individual entries
    return {"log": ["step_start", "validation_ok", "step_end"]}

def read_log(state: State) -> dict:
    print("Log entries:", list(state["log"]))
    return {}

builder = StateGraph(State)
builder.add_node("emit", multi_emit)
builder.add_node("read", read_log)
builder.add_edge(START, "emit")
builder.add_edge("emit", "read")
builder.add_edge("read", END)

graph = builder.compile()
graph.invoke({"log": []})
# Log entries: ['step_start', 'validation_ok', 'step_end']
```

### Example 3 — `accumulate=True` for a growing audit trail

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# accumulate=True keeps values across supersteps
class State(TypedDict):
    audit: Annotated[Sequence[str], Topic(str, accumulate=True)]

def step1(state: State) -> dict:
    return {"audit": "step1"}

def step2(state: State) -> dict:
    return {"audit": "step2"}

builder = StateGraph(State)
builder.add_node("s1", step1)
builder.add_node("s2", step2)
builder.add_edge(START, "s1")
builder.add_edge("s1", "s2")
builder.add_edge("s2", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "audit-1"}}
result = graph.invoke({"audit": []}, config=config)
print(result["audit"])  # ['step1', 'step2'] — both entries preserved
```

---

## 3 · `RetryPolicy` · `TimeoutPolicy` · `CachePolicy`

**Module:** `langgraph.types`

These three policy dataclasses control the **execution contract** for individual nodes and tasks. They attach to nodes at compile time or to `Send`/`call()` at runtime.

**Key source facts** (from `langgraph/types.py`):

- `RetryPolicy` is a `NamedTuple` with fields: `initial_interval=0.5`, `backoff_factor=2.0`, `max_interval=128.0`, `max_attempts=3`, `jitter=True`, `retry_on=default_retry_on`. `retry_on` can be a single exception class, a list, or a `Callable[[Exception], bool]`.
- `TimeoutPolicy` is a frozen dataclass with `run_timeout` (hard wall-clock cap, never refreshed), `idle_timeout` (max silence before cancellation), and `refresh_on: Literal["auto", "heartbeat"]`. `TimeoutPolicy.coerce(float)` converts a bare number to `TimeoutPolicy(run_timeout=float)`. Only works with async nodes (sync nodes raise `sync_timeout_unsupported`).
- `CachePolicy` is a frozen dataclass with `key_func` (defaults to pickle-hash of input) and `ttl: int | None` (seconds; `None` = never expires). Requires a cache backend attached to the graph.

### Example 1 — per-node `RetryPolicy` with custom exception filter

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
import httpx

def fetch_data(state: dict) -> dict:
    # Simulated flaky HTTP call
    response = httpx.get("https://api.example.com/data", timeout=5.0)
    return {"data": response.json()}

# Retry only on network errors, up to 5 attempts, cap at 30s between retries
retry = RetryPolicy(
    max_attempts=5,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=30.0,
    jitter=True,
    retry_on=httpx.NetworkError,
)

builder = StateGraph(dict)
builder.add_node("fetch", fetch_data, retry=retry)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", END)
graph = builder.compile()
```

### Example 2 — `TimeoutPolicy` with idle timeout and `refresh_on="heartbeat"`

```python
import asyncio
from datetime import timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.runtime import get_runtime

async def slow_llm_node(state: dict) -> dict:
    runtime = get_runtime()
    for chunk in ["thinking", "...", "done"]:
        await asyncio.sleep(2)
        # Manual heartbeat so idle_timeout doesn't fire between chunks
        runtime.heartbeat()
    return {"result": "completed"}

# Hard cap 60s; idle cap 10s refreshed only by explicit heartbeat()
timeout = TimeoutPolicy(
    run_timeout=timedelta(seconds=60),
    idle_timeout=10.0,
    refresh_on="heartbeat",
)

builder = StateGraph(dict)
builder.add_node("slow_llm", slow_llm_node, timeout=timeout)
builder.add_edge(START, "slow_llm")
builder.add_edge("slow_llm", END)
graph = builder.compile()
```

### Example 3 — `CachePolicy` with custom key function and TTL

```python
import hashlib
import json
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

def query_key(state: dict) -> str:
    """Cache key: hash of the query string only, ignoring session metadata."""
    payload = json.dumps({"query": state.get("query", "")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def expensive_search(state: dict) -> dict:
    print(f"  → running expensive search for: {state['query']}")
    return {"results": [f"result for {state['query']}"]}

cache_policy = CachePolicy(key_func=query_key, ttl=300)  # 5-minute TTL

builder = StateGraph(dict)
builder.add_node("search", expensive_search, cache_policy=cache_policy)
builder.add_edge(START, "search")
builder.add_edge("search", END)

cache = InMemoryCache()
graph = builder.compile(cache=cache)

# First call hits the node; second call with the same query hits the cache
graph.invoke({"query": "langgraph channels"})
graph.invoke({"query": "langgraph channels"})  # cache hit — no print
```

---

## 4 · `ChannelWrite` · `ChannelWriteEntry` · `ChannelWriteTupleEntry`

**Module:** `langgraph.pregel._write`

`ChannelWrite` is how **node outputs reach channel state**. Every compiled node's bound pipeline ends with a `ChannelWrite` that routes return values to the correct channel keys. You can also use `ChannelWrite.do_write()` imperatively or mark a custom object as a writer with `ChannelWrite.register_writer()`.

**Key source facts** (from `langgraph/pregel/_write.py`):

- `ChannelWriteEntry(channel, value=PASSTHROUGH, skip_none=False, mapper=None)` — `PASSTHROUGH` sentinel means "use the node's input as the value"; `skip_none=True` suppresses writes when the value evaluates to `None`; `mapper` is a `Callable` that transforms the value before it lands in the channel.
- `ChannelWriteTupleEntry(mapper, value=PASSTHROUGH, static=None)` — for conditional writes: `mapper` returns `Sequence[tuple[str, Any]] | None`; `static` declares writes for graph static-analysis (used by `draw_graph`).
- `ChannelWrite._write / _awrite` — replaces `PASSTHROUGH` with the actual node input before calling `do_write`.
- `ChannelWrite.do_write(config, writes)` — assembles writes into `(channel, value)` tuples and calls `config[CONF][CONFIG_KEY_SEND]`. Writing to `TASKS` channel raises `InvalidUpdateError`.
- `ChannelWrite.is_writer(runnable)` — checks if a runnable is a `ChannelWrite` or has `_is_channel_writer` attribute (set by `register_writer`).
- `ChannelWrite.get_static_writes(runnable)` — extracts declared conditional-edge targets for graph topology analysis.
- `SKIP_WRITE` sentinel — a mapper can return `SKIP_WRITE` to completely suppress a write.

### Example 1 — `ChannelWriteEntry` with `skip_none` and `mapper`

```python
from langgraph.pregel._write import ChannelWriteEntry, SKIP_WRITE, _assemble_writes

def upper_or_skip(v: str) -> str | object:
    """Return the upper-cased string, or SKIP_WRITE to suppress the channel write."""
    return v.upper() if v else SKIP_WRITE

# _assemble_writes is the internal helper that applies mappers and respects SKIP_WRITE
entries_hello = [ChannelWriteEntry("result", "hello", skip_none=False, mapper=upper_or_skip)]
entries_empty = [ChannelWriteEntry("result", "", skip_none=False, mapper=upper_or_skip)]
entries_none  = [ChannelWriteEntry("result", None,  skip_none=True,  mapper=None)]

print(_assemble_writes(entries_hello))  # [('result', 'HELLO')]  — mapper applied
print(_assemble_writes(entries_empty))  # []                     — SKIP_WRITE suppresses
print(_assemble_writes(entries_none))   # []                     — skip_none=True suppresses
```

### Example 2 — marking a custom callable as a channel writer

```python
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langchain_core.runnables import RunnableConfig

class MyCustomWriter:
    """A Runnable-like object that writes to channels imperatively."""

    _is_channel_writer = None  # Will be set by register_writer

    def invoke(self, input: dict, config: RunnableConfig) -> dict:
        # In real use, this would call config[CONF][CONFIG_KEY_SEND]
        return input

writer = MyCustomWriter()

# Mark it as a writer so PregelNode.flat_writers() includes it
ChannelWrite.register_writer(
    writer,
    static=[
        (ChannelWriteEntry("output", "static_value"), "output_label"),
    ],
)

print(ChannelWrite.is_writer(writer))    # True
static = ChannelWrite.get_static_writes(writer)
print(static)  # [('output', 'static_value', 'output_label')]
```

### Example 3 — `ChannelWriteTupleEntry` for conditional multi-channel writes

```python
from typing import Any, Sequence
from langgraph.pregel._write import ChannelWrite, ChannelWriteTupleEntry

def route_output(value: dict) -> Sequence[tuple[str, Any]] | None:
    """Route output to different channels based on content."""
    if value.get("error"):
        return [("errors", value["error"]), ("status", "failed")]
    elif value.get("result"):
        return [("results", value["result"]), ("status", "success")]
    return None  # No write at all

# ChannelWriteTupleEntry uses the mapper to decide which channels get written
entry = ChannelWriteTupleEntry(
    mapper=route_output,
    value={"result": "42"},  # Explicit value (not PASSTHROUGH)
    static=[("results", None, "success_path"), ("errors", None, "error_path")],
)

# Simulate the assembled output
assembled = entry.mapper(entry.value)
print(assembled)  # [('results', '42'), ('status', 'success')]
```

---

## 5 · `PregelProtocol` · `StreamProtocol`

**Module:** `langgraph.pregel.protocol`

`PregelProtocol` is the **Abstract Base Class** that all LangGraph graph implementations (`CompiledStateGraph`, `RemoteGraph`, etc.) must satisfy. It defines the full public API surface for running, inspecting, and updating graph state. `StreamProtocol` is the lightweight callback contract used internally to route stream chunks to the right subscriber.

**Key source facts** (from `langgraph/pregel/protocol.py`):

- `PregelProtocol[StateT, ContextT, InputT, OutputT]` extends `Runnable[InputT, Any]` with four abstract generic slots.
- `invoke(input, config, *, context, interrupt_before, interrupt_after, version)` — `version="v2"` returns a typed `GraphOutput[OutputT]`; `version="v1"` (default) returns `dict | Any`. The `version` overloads are separate `@overload` signatures resolved at call time.
- `stream(...)` / `astream(...)` — same dual `version` overloads; `version="v2"` yields `StreamPart[StateT, OutputT]` TypedDicts; `version="v1"` yields `dict[str, Any] | Any`.
- `get_state(config, *, subgraphs=False)` → `StateSnapshot`; `update_state(config, values, as_node=None)` → `RunnableConfig` pointing to the new checkpoint.
- `bulk_update_state(config, updates: Sequence[Sequence[StateUpdate]])` → `RunnableConfig` — applies multiple ordered state updates in one checkpoint write, useful for replay and test seeding.
- `get_graph(config, *, xray=False)` → `DrawableGraph` — for Mermaid/PNG rendering; `xray=True` expands subgraphs inline.
- `StreamProtocol(callable, modes)` — `__slots__ = ("modes", "__call__")`; `modes: set[StreamMode]` gates which stream modes this subscriber handles; stored as an instance attribute via `__call__` — not a method — for zero-overhead dispatch.

### Example 1 — type-safe v2 streaming with `StreamPart` TypedDicts

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def echo(state: State) -> dict:
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"Echo: {last.content}")]}

builder = StateGraph(State)
builder.add_node("echo", echo)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()

# version="v2" yields typed StreamPart TypedDicts
for part in graph.stream(
    {"messages": [HumanMessage(content="Hello")]},
    stream_mode=["values", "updates"],
    version="v2",
):
    print(part["type"], "->", list(part.get("data", {}).keys()) if isinstance(part.get("data"), dict) else "...")
```

### Example 2 — `bulk_update_state` for multi-step replay seeding

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate

class State(TypedDict):
    step: int
    notes: list[str]

def process(state: State) -> dict:
    return {"step": state["step"] + 1}

builder = StateGraph(State)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "bulk-1"}}

# Seed the graph with an initial state before running
graph.invoke({"step": 0, "notes": []}, config=config)

# Apply two ordered state mutations in one checkpoint write
new_config = graph.bulk_update_state(
    config,
    updates=[
        [StateUpdate(values={"step": 10, "notes": ["seeded"]}, as_node="process")],
        [StateUpdate(values={"notes": ["seeded", "corrected"]}, as_node="process")],
    ],
)
snapshot = graph.get_state(new_config)
print(snapshot.values)  # {'step': 10, 'notes': ['seeded', 'corrected']}
```

### Example 3 — implementing a custom `PregelProtocol` wrapper

```python
from collections.abc import Iterator, AsyncIterator
from typing import Any
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import Graph as DrawableGraph
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import StateSnapshot, StateUpdate

class LoggingGraphWrapper(PregelProtocol):
    """Wraps any PregelProtocol and logs every invoke call."""

    def __init__(self, inner: PregelProtocol) -> None:
        self._inner = inner

    def with_config(self, config=None, **kwargs):
        return LoggingGraphWrapper(self._inner.with_config(config, **kwargs))

    def get_graph(self, config=None, *, xray=False) -> DrawableGraph:
        return self._inner.get_graph(config, xray=xray)

    async def aget_graph(self, config=None, *, xray=False) -> DrawableGraph:
        return await self._inner.aget_graph(config, xray=xray)

    def get_state(self, config, *, subgraphs=False) -> StateSnapshot:
        return self._inner.get_state(config, subgraphs=subgraphs)

    async def aget_state(self, config, *, subgraphs=False) -> StateSnapshot:
        return await self._inner.aget_state(config, subgraphs=subgraphs)

    def get_state_history(self, config, *, filter=None, before=None, limit=None):
        return self._inner.get_state_history(config, filter=filter, before=before, limit=limit)

    def aget_state_history(self, config, *, filter=None, before=None, limit=None):
        return self._inner.aget_state_history(config, filter=filter, before=before, limit=limit)

    def bulk_update_state(self, config, updates):
        return self._inner.bulk_update_state(config, updates)

    async def abulk_update_state(self, config, updates):
        return await self._inner.abulk_update_state(config, updates)

    def update_state(self, config, values, as_node=None):
        return self._inner.update_state(config, values, as_node=as_node)

    async def aupdate_state(self, config, values, as_node=None):
        return await self._inner.aupdate_state(config, values, as_node=as_node)

    def stream(self, input, config=None, **kwargs):
        return self._inner.stream(input, config, **kwargs)

    def astream(self, input, config=None, **kwargs):
        return self._inner.astream(input, config, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        print(f"[LOG] invoke called with input keys: {list(input.keys()) if isinstance(input, dict) else type(input).__name__}")
        return self._inner.invoke(input, config, **kwargs)

    async def ainvoke(self, input, config=None, **kwargs):
        print(f"[LOG] ainvoke called")
        return await self._inner.ainvoke(input, config, **kwargs)
```

---

## 6 · `add_messages` · `MessagesState` · `push_message` · `REMOVE_ALL_MESSAGES`

**Module:** `langgraph.graph.message`

`add_messages` is the canonical **message list reducer** used in virtually every chat-based LangGraph application. It merges two lists of messages by ID — new messages with matching IDs overwrite existing ones — making the state naturally append-only unless explicitly edited. `push_message()` lets nodes emit messages imperatively to the stream and state without going through the normal return dict.

**Key source facts** (from `langgraph/graph/message.py`):

- `add_messages(left, right, *, format=None)` — coerces both sides via `convert_to_messages`; assigns `uuid4` IDs to any message that lacks one; checks for `RemoveMessage` tombstones; if a `RemoveMessage` has `id == REMOVE_ALL_MESSAGES`, discards everything before it in `right` and returns only what follows.
- `REMOVE_ALL_MESSAGES = "__remove_all__"` — a special sentinel ID that wipes the conversation history. Write `RemoveMessage(id=REMOVE_ALL_MESSAGES)` to clear the list.
- `format="langchain-openai"` — converts the merged list through `convert_to_openai_messages` for providers that expect the OpenAI content-block schema (text/image_url blocks, ToolMessage as separate object). Requires `langchain-core>=0.3.11`.
- `MessagesState` — a ready-to-use `TypedDict` with a single `messages: Annotated[list[AnyMessage], add_messages]` field.
- `push_message(message, *, state_key="messages")` — must be called from inside a running node; finds the `StreamMessagesHandler` in the active callbacks and emits directly to the stream, then writes to the channel via `CONFIG_KEY_SEND`. `state_key=None` emits stream-only with no state write.
- `_messages_delta_reducer(state, writes)` — experimental batch reducer for `DeltaChannel`; processes all writes in one pass without calling `add_messages`; batching-invariant.

### Example 1 — `add_messages` dedup and `RemoveMessage` tombstoning

```python
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages

# Dedup by ID: same ID overwrites
msgs = [HumanMessage(content="Hello", id="1"), AIMessage(content="Hi", id="2")]
updated = add_messages(msgs, [AIMessage(content="Hi there!", id="2")])
print([m.content for m in updated])  # ['Hello', 'Hi there!']

# RemoveMessage tombstoning
cleaned = add_messages(msgs, [RemoveMessage(id="1")])
print([m.content for m in cleaned])  # ['Hi']

# REMOVE_ALL_MESSAGES wipes history before a fresh start
from langgraph.graph.message import REMOVE_ALL_MESSAGES
wiped = add_messages(msgs, [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(content="Fresh start", id="3")])
print([m.content for m in wiped])  # ['Fresh start']
```

### Example 2 — `MessagesState` with streaming

```python
from typing import Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage

def chatbot(state: MessagesState) -> dict:
    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    return {"messages": [AIMessage(content=f"You said: {last_human.content}")]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "chat-1"}}

# First turn
result = graph.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)
print(result["messages"][-1].content)  # 'You said: Hello'

# Second turn — history is preserved via checkpointer
result = graph.invoke({"messages": [HumanMessage(content="How are you?")]}, config=config)
print(len(result["messages"]))  # 4 — two exchanges
```

### Example 3 — `push_message` for streaming without waiting for node return

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, push_message
from langchain_core.messages import AIMessage, HumanMessage

async def streaming_node(state: MessagesState) -> dict:
    # push_message emits directly to stream; user sees chunks as they arrive
    words = ["The", " answer", " is", " 42"]
    full_content = ""
    for word in words:
        full_content += word
        # Each push_message call creates and emits a new AIMessage to the stream
        push_message(AIMessage(content=full_content, id="stream-msg-1"), state_key="messages")
        await asyncio.sleep(0.05)  # Simulate token latency

    # Final return is deduplicated by ID with the last push_message call
    return {"messages": [AIMessage(content=full_content, id="stream-msg-1")]}

builder = StateGraph(MessagesState)
builder.add_node("streamer", streaming_node)
builder.add_edge(START, "streamer")
builder.add_edge("streamer", END)

graph = builder.compile()

async def main():
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content="What is the answer?")]},
        stream_mode="messages",
    ):
        print(chunk)

asyncio.run(main())
```

---

## 7 · `EmbeddingsLambda` · `ensure_embeddings` · `get_text_at_path` · `tokenize_path`

**Module:** `langgraph.store.base.embed`

This module provides the embedding bridge between LangGraph's vector store layer and arbitrary embedding functions. `ensure_embeddings` is the entry point — it accepts anything from a plain Python function to a provider string — and returns a LangChain `Embeddings` object. `get_text_at_path` and `tokenize_path` power the `fields` index configuration that controls which parts of a stored document get vectorised.

**Key source facts** (from `langgraph/store/base/embed.py`):

- `ensure_embeddings(embed)` — handles four input types: an existing `Embeddings` instance (returned as-is), a `str` like `"openai:text-embedding-3-small"` (dispatched to `langchain.embeddings.init_embeddings` — requires `langchain>=0.3.9`), a sync `Callable[[Sequence[str]], list[list[float]]]` (wrapped in `EmbeddingsLambda`), or an async callable (also wrapped in `EmbeddingsLambda` with `afunc` set).
- `EmbeddingsLambda(func)` — detects async via `asyncio.iscoroutinefunction(func)` or `__call__` introspection; stores as `self.afunc` if async or `self.func` if sync. `embed_documents()` raises `ValueError` if only `afunc` is set; `aembed_documents()` falls back to `super().aembed_documents()` (thread-pool) if only `func` is set.
- `get_text_at_path(obj, path)` — JMESPath-inspired extractor supporting: `"field.nested"` dot paths, `"[0]"` / `"[*]"` / `"[-1]"` array indexing, `"*"` wildcard, `"{field1,nested.field2}"` multi-field selection. Returns `list[str]` of extracted text snippets. `"$"` or empty path serialises the whole object as JSON.
- `tokenize_path(path)` — pre-tokenises a path string into a `list[str]` for `get_text_at_path`; handles nested brackets and braces.

### Example 1 — wrapping a sync embedding function with `ensure_embeddings`

```python
from langgraph.store.base.embed import ensure_embeddings

def my_embed(texts: list[str]) -> list[list[float]]:
    """Toy embedding: ASCII sum of each character, normalised."""
    return [[sum(ord(c) for c in t) / 1000.0] for t in texts]

embeddings = ensure_embeddings(my_embed)
print(type(embeddings).__name__)          # EmbeddingsLambda
print(embeddings.embed_query("hello"))   # [0.532]
print(embeddings.embed_documents(["hi", "world"]))  # [[0.209], [0.5530...]]
```

### Example 2 — async embedding function via `EmbeddingsLambda`

```python
import asyncio
from langgraph.store.base.embed import ensure_embeddings

async def async_embed(texts: list[str]) -> list[list[float]]:
    """Simulate an async model API call."""
    await asyncio.sleep(0.01)
    return [[len(t) / 100.0] for t in texts]

embeddings = ensure_embeddings(async_embed)

async def main():
    result = await embeddings.aembed_documents(["short", "a longer string"])
    print(result)  # [[0.05], [0.14]]

asyncio.run(main())
```

### Example 3 — `get_text_at_path` for selective field vectorisation

```python
from langgraph.store.base.embed import get_text_at_path, tokenize_path

doc = {
    "title": "LangGraph internals",
    "body": "Deep dive into channels and checkpointers.",
    "meta": {"author": "Alice", "tags": ["graphs", "python"]},
    "scores": [9, 8, 10],
}

# Dot-path: extract a single nested field
print(get_text_at_path(doc, "title"))               # ['LangGraph internals']
print(get_text_at_path(doc, "meta.author"))         # ['Alice']

# Array wildcard: all elements of a list
print(get_text_at_path(doc, "meta.tags[*]"))        # ['graphs', 'python']

# Multi-field selection: combine title and body in one call
print(get_text_at_path(doc, "{title,body}"))        # ['LangGraph internals', 'Deep dive...']

# Whole-document JSON: use "$"
whole = get_text_at_path(doc, "$")
print(len(whole))  # 1 — the entire doc serialised as a single JSON string

# Pre-tokenise for repeated extraction (avoids re-parsing on every call)
tokens = tokenize_path("meta.tags[*]")
print(tokens)  # ['meta', 'tags', '[*]']
print(get_text_at_path(doc, tokens))  # ['graphs', 'python']
```

---

## 8 · `InMemoryStore`

**Module:** `langgraph.store.memory`

`InMemoryStore` is LangGraph's built-in key-value store for **cross-thread state** — data that lives beyond a single conversation thread and is visible to multiple agents. It supports optional vector similarity search when configured with an embedding function and `dims`.

**Key source facts** (from `langgraph/store/memory/__init__.py`):

- `InMemoryStore(*, index: IndexConfig | None = None)` — `IndexConfig` is a `TypedDict` with `dims: int`, `embed: Embeddings | Callable | str`, and optional `fields: list[str]` (defaults to `["$"]` = whole document). Without `index`, all `put(…, index=…)` arguments are silently ignored.
- `_data: dict[tuple[str,...], dict[str, Item]]` — namespace-keyed dict of key→`Item` mappings. `_vectors: dict[namespace][key][path] = list[float]`.
- `put(namespace, key, value, index=None)` — stores `Item(namespace, key, value, …)`; if index config is present and `index` arg is not `False`, computes embeddings for the fields specified by `index` (or the store's `index_config["fields"]`) and stores in `_vectors`.
- `search(namespace_prefix, *, query=None, filter=None, limit=10, offset=0)` — with `query`, computes cosine similarity against `_vectors`; without, returns items sorted by `updated_at` descending. `filter` is a dict of exact-match conditions on the `value` dict.
- `list_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0)` — returns all distinct namespace tuples matching the given prefix/suffix constraints.
- `abatch(ops)` is a native async implementation: it calls `await self._aembed_search_queries()` (which uses `asyncio.gather` over `aembed_query` coroutines) and `await self.embeddings.aembed_documents()`. The sync `batch()` path uses a `ThreadPoolExecutor` to run query embeddings concurrently. Public async helpers (`aget`, `aput`, `asearch`, etc.) are inherited from `BaseStore` and route through `abatch()`.

### Example 1 — basic namespace-keyed key-value storage

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Store user preferences under a namespaced key
store.put(("users", "alice"), "preferences", {"theme": "dark", "lang": "en"})
store.put(("users", "bob"), "preferences", {"theme": "light", "lang": "fr"})

# Retrieve a specific item
item = store.get(("users", "alice"), "preferences")
print(item.value)  # {'theme': 'dark', 'lang': 'en'}
print(item.namespace)  # ('users', 'alice')

# List all user namespaces
namespaces = store.list_namespaces(prefix=("users",))
print(namespaces)  # [('users', 'alice'), ('users', 'bob')]

# Delete an item
store.delete(("users", "bob"), "preferences")
print(store.get(("users", "bob"), "preferences"))  # None
```

### Example 2 — vector similarity search with a custom embedding function

```python
from langgraph.store.memory import InMemoryStore

def simple_embed(texts: list[str]) -> list[list[float]]:
    """Toy bag-of-chars embedding for demo purposes."""
    chars = "abcdefghijklmnopqrstuvwxyz "
    return [
        [text.lower().count(c) / max(len(text), 1) for c in chars]
        for text in texts
    ]

store = InMemoryStore(
    index={"dims": len("abcdefghijklmnopqrstuvwxyz "), "embed": simple_embed, "fields": ["text"]}
)

store.put(("kb",), "doc1", {"text": "python programming tutorial"})
store.put(("kb",), "doc2", {"text": "javascript web development"})
store.put(("kb",), "doc3", {"text": "machine learning with python"})

results = store.search(("kb",), query="python guide")
for r in results:
    print(f"{r.key}: score={r.score:.3f} text={r.value['text']}")
# doc1 and doc3 should rank above doc2
```

### Example 3 — using `InMemoryStore` inside a multi-agent graph

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store
from typing_extensions import TypedDict

class State(TypedDict):
    user_id: str
    message: str
    response: str

async def remember_node(state: State) -> dict:
    store = get_store()
    # Load prior context for this user
    item = await store.aget(("memory", state["user_id"]), "context")
    prior = item.value if item else {}
    # Save updated context
    await store.aput(
        ("memory", state["user_id"]),
        "context",
        {**prior, "last_message": state["message"]},
    )
    return {"response": f"Remembered: {state['message']}"}

builder = StateGraph(State)
builder.add_node("remember", remember_node)
builder.add_edge(START, "remember")
builder.add_edge("remember", END)

store = InMemoryStore()
graph = builder.compile(store=store)

async def main():
    await graph.ainvoke({"user_id": "alice", "message": "Hi!", "response": ""})
    await graph.ainvoke({"user_id": "alice", "message": "Remember me?", "response": ""})
    item = await store.aget(("memory", "alice"), "context")
    print(item.value)  # {'last_message': 'Remember me?'}

asyncio.run(main())
```

---

## 9 · `AsyncBatchedBaseStore`

**Module:** `langgraph.store.base.batch`

`AsyncBatchedBaseStore` is the **async-safe wrapper base class** for stores that need to be accessed from inside an asyncio event loop. It serialises concurrent `aget/aput/asearch` calls through a single `asyncio.Queue` processed by a background task — avoiding the thundering-herd problem that arises when many concurrent nodes each make direct store calls.

**Key source facts** (from `langgraph/store/base/batch.py`):

- `AsyncBatchedBaseStore.__init__()` — captures `self._loop = asyncio.get_running_loop()`, creates `self._aqueue: asyncio.Queue[tuple[asyncio.Future, Op]]`, and starts `self._task` via `_ensure_task()`. Must be constructed from within a running event loop.
- Every `async` method (`aget`, `aput`, `asearch`, `adelete`, `alist_namespaces`) creates an `asyncio.Future`, enqueues `(future, op)` onto `_aqueue`, and `await`s the future. The background task calls `self.abatch(ops)` and resolves all futures.
- `_ensure_task()` — restarts the background task if it has died (e.g. exception, cancellation). Called at the top of every async method.
- `@_check_loop` decorator on sync methods (`get`, `batch`, etc.) — detects when the caller is on `self._loop`  and raises `asyncio.InvalidStateError` to prevent deadlock. Only applies when the sync method is called from the same event loop that `AsyncBatchedBaseStore` owns.
- `batch(ops)` — `asyncio.run_coroutine_threadsafe(self.abatch(ops), self._loop).result()` — allows synchronous callers from a **different** thread to submit a batch.
- The `_run(queue, store_ref)` background coroutine drains batches from `_aqueue`, calls `abatch`, and resolves all futures; uses `weakref` to avoid keeping the store alive after all other references drop.

### Example 1 — constructing `AsyncBatchedBaseStore` subclass inside an event loop

```python
import asyncio
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.base import GetOp, PutOp, Op, Result, Item
from datetime import datetime, timezone

class SimpleAsyncStore(AsyncBatchedBaseStore):
    """Minimal in-memory store that extends AsyncBatchedBaseStore."""

    def __init__(self) -> None:
        super().__init__()         # Must be called from within an event loop
        self._data: dict[tuple, dict[str, Item]] = {}

    async def abatch(self, ops: list[Op]) -> list[Result]:
        results = []
        for op in ops:
            if isinstance(op, PutOp):
                if op.value is None:
                    self._data.get(op.namespace, {}).pop(op.key, None)
                else:
                    ns = self._data.setdefault(op.namespace, {})
                    ns[op.key] = Item(
                        value=op.value, key=op.key, namespace=op.namespace,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self._data.get(op.namespace, {}).get(op.key))
            else:
                results.append(None)
        return results

async def main():
    store = SimpleAsyncStore()
    await store.aput(("ns",), "key1", {"data": "hello"})
    item = await store.aget(("ns",), "key1")
    print(item.value)  # {'data': 'hello'}

asyncio.run(main())
```

### Example 2 — concurrent async store operations via the background queue

`InMemoryStore` inherits from `BaseStore` (not `AsyncBatchedBaseStore`), so it doesn't
use the queue pattern. Use `SimpleAsyncStore` from Example 1 to observe the batching.

```python
import asyncio
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.base import GetOp, PutOp, Op, Result, Item
from datetime import datetime, timezone

class SimpleAsyncStore(AsyncBatchedBaseStore):
    def __init__(self) -> None:
        super().__init__()
        self._data: dict[tuple, dict[str, Item]] = {}

    async def abatch(self, ops: list[Op]) -> list[Result]:
        results = []
        for op in ops:
            if isinstance(op, PutOp):
                ns = self._data.setdefault(op.namespace, {})
                if op.value is None:
                    ns.pop(op.key, None)
                else:
                    ns[op.key] = Item(
                        value=op.value, key=op.key, namespace=op.namespace,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self._data.get(op.namespace, {}).get(op.key))
            else:
                results.append(None)
        return results

async def main():
    store = SimpleAsyncStore()

    # Fire many concurrent puts — all serialised through the background asyncio.Queue
    await asyncio.gather(*[
        store.aput(("docs",), f"doc{i}", {"content": f"Document {i}"})
        for i in range(10)
    ])

    # Concurrent gets — each enqueues a GetOp; abatch() resolves them together
    items = await asyncio.gather(*[
        store.aget(("docs",), f"doc{i}")
        for i in range(10)
    ])
    print(f"Retrieved {len([x for x in items if x])} documents")  # 10

asyncio.run(main())
```

### Example 3 — `_check_loop` deadlock prevention

`_check_loop` only fires on `AsyncBatchedBaseStore` subclasses — `InMemoryStore` inherits
from `BaseStore` directly and has no deadlock guard. Use the `SimpleAsyncStore` subclass
from Example 1 to observe the guard in action.

```python
import asyncio
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.base import GetOp, PutOp, Op, Result, Item
from datetime import datetime, timezone

class SimpleAsyncStore(AsyncBatchedBaseStore):
    def __init__(self) -> None:
        super().__init__()
        self._data: dict[tuple, dict[str, Item]] = {}

    async def abatch(self, ops: list[Op]) -> list[Result]:
        results = []
        for op in ops:
            if isinstance(op, PutOp):
                ns = self._data.setdefault(op.namespace, {})
                if op.value is None:
                    ns.pop(op.key, None)
                else:
                    ns[op.key] = Item(
                        value=op.value, key=op.key, namespace=op.namespace,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self._data.get(op.namespace, {}).get(op.key))
            else:
                results.append(None)
        return results

async def main():
    store = SimpleAsyncStore()  # must be constructed inside running event loop
    await store.aput(("test",), "k", {"v": 1})

    # Calling the SYNC get() from inside the SAME event loop raises InvalidStateError
    # (this is _check_loop in action — prevents deadlock)
    try:
        store.get(("test",), "k")   # Would deadlock without _check_loop
    except asyncio.InvalidStateError as e:
        print(f"Blocked correctly: {e}")
        # Use async variant instead:
        item = await store.aget(("test",), "k")
        print(f"Async get succeeded: {item.value}")

asyncio.run(main())
```

---

## 10 · `call()` · `SyncAsyncFuture` · `PregelScratchpad`

**Module:** `langgraph.pregel._call` · `langgraph._internal._scratchpad`

`call()` is the imperative API for **spawning sub-tasks from within a functional `@entrypoint` node**. It schedules a function to run as a tracked Pregel task — with its own retry policy, cache policy, and timeout — and returns a `SyncAsyncFuture` that can be awaited in async contexts or resolved synchronously. `PregelScratchpad` is the per-run mutable bookkeeping object that tracks step counts, interrupt state, and task counters for the Pregel coordinator.

**Key source facts** (from `langgraph/pregel/_call.py` and `_internal/_scratchpad.py`):

- `call(func, *args, retry_policy=None, cache_policy=None, timeout=None, **kwargs) -> SyncAsyncFuture[T]` — wraps `func` in a `get_runnable_for_task(func)` pipeline (which appends a `ChannelWrite([RETURN])`) and schedules it via `config[CONF][CONFIG_KEY_CALL]`. Sync functions with a `timeout` raise `sync_timeout_unsupported` immediately.
- `SyncAsyncFuture[T]` — a `concurrent.futures.Future[T]` subclass with `__await__` implemented as a generator yielding `cast(T, ...)` once. This makes it awaitable in `async` contexts while remaining a regular `Future` for sync resolution.
- `get_runnable_for_task(func)` — wraps `func` in a `RunnableCallable(func, afunc, explode_args=True, …)` followed by `ChannelWrite([ChannelWriteEntry(RETURN)])`, so the task's return value lands in the `RETURN` channel. Caches the wrapper by `(func, True)` key — the same function object always gets the same `Runnable`.
- `get_runnable_for_entrypoint(func)` — similar but no `ChannelWrite`; used for `@entrypoint`-decorated functions.
- `identifier(obj, name=None)` — resolves a function's `module.qualname` for stable cross-process identification (used by the cache key and remote execution). Returns `None` for locally-defined / closure functions.
- `PregelScratchpad` — a frozen-ish `@dataclass(**_DC_KWARGS)` with: `step: int` (current superstep), `stop: int` (total supersteps limit), `call_counter: Callable[[], int]` (monotonic per-thread counter), `interrupt_counter: Callable[[], int]`, `get_null_resume: Callable[[bool], Any]`, `resume: list[Any]` (stack of resume values), `subgraph_counter: Callable[[], int]`.

### Example 1 — `@task` with `RetryPolicy` inside a functional `@entrypoint`

`@task` uses `call()` under the hood to schedule each invocation as an independent
tracked Pregel sub-task. You normally interact with `call()` indirectly through `@task`.

```python
import asyncio
import httpx
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

retry = RetryPolicy(max_attempts=3, initial_interval=0.5, retry_on=httpx.NetworkError)

@task(retry=retry)
async def fetch(url: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10.0)
    return r.text[:100]

@entrypoint(checkpointer=InMemorySaver())
async def pipeline(urls: list[str]) -> list[str]:
    # Each fetch() call dispatches via call() — a separate tracked Pregel sub-task
    futures = [fetch(url) for url in urls]
    results = await asyncio.gather(*futures)
    return list(results)

# pipeline.invoke(["https://example.com", "https://example.org"])
```

### Example 2 — `SyncAsyncFuture` as a `concurrent.futures.Future`

`SyncAsyncFuture.__await__` yields a sentinel value meant only for LangGraph's internal
Pregel scheduler — **do not `await` a bare `SyncAsyncFuture` in plain asyncio** (it raises
`RuntimeError: Task got bad yield`). Use `asyncio.wrap_future()` to bridge it into asyncio,
or call `.result()` to block synchronously from a non-loop thread.

```python
import asyncio
import concurrent.futures
from langgraph.pregel._call import SyncAsyncFuture

fut: SyncAsyncFuture[str] = SyncAsyncFuture()

# Resolve it from a thread (as the Pregel executor does)
def resolve_from_thread():
    import time; time.sleep(0.05)
    fut.set_result("hello from task")

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(resolve_from_thread)

    async def main():
        # asyncio.wrap_future() bridges a concurrent.futures.Future into asyncio —
        # do NOT write `await fut` directly here; __await__ is for LangGraph's scheduler.
        result = await asyncio.wrap_future(fut)
        print(result)  # 'hello from task'

    asyncio.run(main())
```

### Example 3 — inspecting `PregelScratchpad` from a managed value

```python
from langgraph.managed.base import ManagedValue
from langgraph._internal._scratchpad import PregelScratchpad

class StepInfoValue(ManagedValue[dict]):
    """Injects current step number and total steps into every node."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> dict:
        return {
            "current_step": scratchpad.step,
            "total_steps": scratchpad.stop,
            "remaining": scratchpad.stop - scratchpad.step,
            "is_last": scratchpad.step == scratchpad.stop - 1,
        }

# Usage in a graph:
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    step_info: Annotated[dict, StepInfoValue]
    result: str

def my_node(state: State) -> dict:
    info = state["step_info"]
    print(f"Step {info['current_step']} of {info['total_steps']} (remaining: {info['remaining']})")
    return {"result": "done"}

builder = StateGraph(State)
builder.add_node("node", my_node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()
graph.invoke({"result": ""})
# Example output (step/stop values depend on graph depth and recursion limit):
# Step 0 of 25 (remaining: 25)
```

---
title: "Class deep-dives Vol. 7 — 10 more LangGraph types"
description: "Source-verified deep dives into PregelProtocol/StreamProtocol, BackgroundExecutor/AsyncBackgroundExecutor, AsyncBatchedBaseStore/_dedupe_ops, get_text_at_path/tokenize_path, SerdeEvent/register_serde_event_listener, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec/node Protocols, and identifier/get_runnable_for_task — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 7"
  order: 31
---

# Class deep-dives Vol. 7 — 10 more LangGraph types

Verified against **`langgraph==1.2.2`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

[→ Vol. 4 covers set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, error taxonomy](./langgraph_class_deep_dives_v4/)

[→ Vol. 5 covers RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt protocol](./langgraph_class_deep_dives_v5/)

[→ Vol. 6 covers GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message/ui_message_reducer, ProtocolEvent](./langgraph_class_deep_dives_v6/)

---

## 1 · `PregelProtocol` + `StreamProtocol`

**Module:** `langgraph.pregel.protocol`  
**Exported as:** `from langgraph.pregel.protocol import PregelProtocol, StreamProtocol`

`PregelProtocol` is the abstract contract that every compiled LangGraph graph — `CompiledStateGraph`, `Pregel`, `RemoteGraph` — must implement. It inherits from LangChain's `Runnable` and adds the full state-management API: `get_state`, `update_state`, `bulk_update_state`, typed `stream`/`astream` and `invoke`/`ainvoke` with `version` overloads, and graph visualisation via `get_graph`.

`StreamProtocol` is a tiny companion struct: a slot-based callable that carries the set of `StreamMode` values it is interested in. The internal streaming pipeline only calls a `StreamProtocol` instance if the emitted chunk's mode is in `protocol.modes`.

### `PregelProtocol` source (condensed)

```python
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, Generic, Literal, overload

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.graph import Graph as DrawableGraph

from langgraph.types import All, Command, GraphOutput, StateSnapshot, StateUpdate, StreamMode, StreamPart
from langgraph.typing import ContextT, InputT, OutputT, StateT

class PregelProtocol(Runnable[InputT, Any], Generic[StateT, ContextT, InputT, OutputT]):
    @abstractmethod
    def get_state(self, config: RunnableConfig, *, subgraphs: bool = False) -> StateSnapshot: ...

    @abstractmethod
    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
    ) -> RunnableConfig: ...

    @abstractmethod
    def bulk_update_state(
        self,
        config: RunnableConfig,
        updates: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig: ...

    # stream has overloads for version="v1" (raw dict) and version="v2" (typed StreamPart)
    @overload
    @abstractmethod
    def stream(
        self, input: InputT | Command | None, config: RunnableConfig | None = None,
        *, version: Literal["v2"],
    ) -> Iterator[StreamPart[StateT, OutputT]]: ...

    @overload
    @abstractmethod
    def stream(
        self, input: InputT | Command | None, config: RunnableConfig | None = None,
        *, version: Literal["v1"] = ...,
    ) -> Iterator[dict[str, Any] | Any]: ...
    # ... same overloads for astream, invoke, ainvoke
```

### `StreamProtocol` source

```python
StreamChunk = tuple[tuple[str, ...], str, Any]
#              ^node path             ^mode  ^data

class StreamProtocol:
    __slots__ = ("modes", "__call__")
    modes: set[StreamMode]
    __call__: Callable[[Self, StreamChunk], None]

    def __init__(
        self,
        __call__: Callable[[StreamChunk], None],
        modes: set[StreamMode],
    ) -> None:
        self.__call__ = cast(Callable[[Self, StreamChunk], None], __call__)
        self.modes = modes
```

### Why `PregelProtocol` matters

When you write code that accepts *any* compiled graph — useful for building generic wrappers, test harnesses, or multi-graph orchestrators — type-annotate it as `PregelProtocol` instead of `CompiledStateGraph` or `Pregel`. This keeps your code generic across implementations including `RemoteGraph` (the cloud execution wrapper).

```python
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import StateSnapshot

def get_thread_snapshot(
    graph: PregelProtocol,
    thread_id: str,
) -> StateSnapshot:
    """Works with any LangGraph compiled graph."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return graph.get_state(config)


def replay_thread(
    graph: PregelProtocol,
    thread_id: str,
    limit: int = 5,
) -> None:
    """Iterate through checkpoint history using the protocol API."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    for snapshot in graph.get_state_history(config, limit=limit):
        step = snapshot.metadata.get("step", "?")
        print(f"Step {step}: {list(snapshot.values.keys())}")
```

### `bulk_update_state` — applying multiple updates atomically

`bulk_update_state` lets you apply a sequence of `(as_node, values)` update groups in a single checkpoint write. Each inner sequence is treated as updates from the same node; outer groups are applied in order.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int
    messages: list[str]

builder = StateGraph(State)
builder.add_node("worker", lambda s: {"counter": s["counter"] + 1})
builder.add_edge(START, "worker")
builder.add_edge("worker", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "bulk-demo"}}
graph.invoke({"counter": 0, "messages": []}, config)

# Apply two logical update groups: first set counter, then append a message
from langgraph.types import StateUpdate
new_config = graph.bulk_update_state(
    config,
    updates=[
        [StateUpdate(values={"counter": 99}, as_node="worker")],
        [StateUpdate(values={"messages": ["reset"]}, as_node="worker")],
    ],
)
snapshot = graph.get_state(new_config)
print(snapshot.values)   # {'counter': 99, 'messages': ['reset']}
```

### Building a custom `StreamProtocol`

```python
from langgraph.pregel.protocol import StreamProtocol, StreamChunk
from langgraph.types import StreamMode

collected: list[StreamChunk] = []

def my_handler(chunk: StreamChunk) -> None:
    node_path, mode, data = chunk
    collected.append(chunk)
    print(f"[{'.'.join(node_path) or 'root'}] {mode}: {data!r}")

# Only subscribe to "values" and "updates" modes
proto = StreamProtocol(my_handler, modes={"values", "updates"})
print(proto.modes)   # {'values', 'updates'}
# The internal dispatcher checks `proto.modes` before calling proto(chunk)
```

---

## 2 · `BackgroundExecutor` + `AsyncBackgroundExecutor`

**Module:** `langgraph.pregel._executor`

These two context managers are the engine behind LangGraph's parallel node execution. When a step has multiple ready nodes, Pregel submits each node's runnable to the appropriate executor — sync nodes go to `BackgroundExecutor` (which uses LangChain's `get_executor_for_config` thread pool), async nodes to `AsyncBackgroundExecutor` (asyncio tasks with optional semaphore-gated concurrency).

Understanding them is key for:
- Tuning `max_concurrency` per invocation
- Knowing when tasks are cancelled vs. awaited on exit
- Debugging `GraphBubbleUp` (interrupt / error propagation) in parallel steps

### `BackgroundExecutor` source

```python
import concurrent.futures, time
from contextlib import AbstractContextManager, ExitStack
from contextvars import copy_context
from langgraph.errors import GraphBubbleUp

class BackgroundExecutor(AbstractContextManager):
    def __init__(self, config: RunnableConfig) -> None:
        self.stack = ExitStack()
        self.executor = self.stack.enter_context(get_executor_for_config(config))
        self.tasks: dict[concurrent.futures.Future, tuple[bool, bool]] = {}
        # task → (__cancel_on_exit__, __reraise_on_exit__)

    def submit(
        self,
        fn,
        *args,
        __cancel_on_exit__: bool = False,
        __reraise_on_exit__: bool = True,
        __next_tick__: bool = False,
        **kwargs,
    ) -> concurrent.futures.Future:
        ctx = copy_context()           # capture contextvars
        if __next_tick__:
            task = self.executor.submit(next_tick, ctx.run, fn, *args, **kwargs)
        else:
            task = self.executor.submit(ctx.run, fn, *args, **kwargs)
        self.tasks[task] = (__cancel_on_exit__, __reraise_on_exit__)
        task.add_done_callback(self.done)
        return task

    def done(self, task):
        try:
            task.result()
        except GraphBubbleUp:
            # Interrupt signal — not an error; remove from tracking silently
            self.tasks.pop(task)
        except BaseException:
            pass  # kept in dict so __exit__ can re-raise
        else:
            self.tasks.pop(task)
```

### `AsyncBackgroundExecutor` source

```python
import asyncio
from contextvars import copy_context
from langgraph.errors import GraphBubbleUp

class AsyncBackgroundExecutor(AbstractAsyncContextManager):
    def __init__(self, config: RunnableConfig) -> None:
        self.tasks: dict[asyncio.Future, tuple[bool, bool]] = {}
        self.loop = asyncio.get_running_loop()
        if max_concurrency := config.get("max_concurrency"):
            self.semaphore: asyncio.Semaphore | None = asyncio.Semaphore(max_concurrency)
        else:
            self.semaphore = None

    def submit(self, fn, *args, __cancel_on_exit__=False,
               __reraise_on_exit__=True, **kwargs) -> asyncio.Future:
        coro = fn(*args, **kwargs)
        if self.semaphore:
            coro = gated(self.semaphore, coro)   # semaphore-gate the coro
        task = run_coroutine_threadsafe(coro, self.loop, context=copy_context())
        self.tasks[task] = (__cancel_on_exit__, __reraise_on_exit__)
        task.add_done_callback(self.done)
        return task
```

### Using `max_concurrency` to cap parallel node execution

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import asyncio, time

class State(TypedDict):
    results: list[str]

async def slow_node_a(state: State) -> dict:
    await asyncio.sleep(0.1)
    return {"results": ["a"]}

async def slow_node_b(state: State) -> dict:
    await asyncio.sleep(0.1)
    return {"results": ["b"]}

async def slow_node_c(state: State) -> dict:
    await asyncio.sleep(0.1)
    return {"results": ["c"]}

builder = StateGraph(State)
for name, fn in [("a", slow_node_a), ("b", slow_node_b), ("c", slow_node_c)]:
    builder.add_node(name, fn)
    builder.add_edge(START, name)
    builder.add_edge(name, END)

graph = builder.compile()

# Without cap — all three run in parallel (~0.1s)
start = time.perf_counter()
await graph.ainvoke({"results": []})
print(f"Uncapped: {time.perf_counter()-start:.2f}s")  # ~0.1s

# Cap at 1 concurrent task — effectively serial (~0.3s)
start = time.perf_counter()
await graph.ainvoke({"results": []}, config={"max_concurrency": 1})
print(f"Capped(1): {time.perf_counter()-start:.2f}s")  # ~0.3s
```

### `__cancel_on_exit__` vs `__reraise_on_exit__`

These flags control the lifecycle policy for each submitted task:

| Flag | `True` (default for `reraise`) | `False` |
|---|---|---|
| `__cancel_on_exit__` | Cancel the task when the executor exits | Let it finish even after exit |
| `__reraise_on_exit__` | Surface task exceptions to the caller | Swallow exceptions silently |

```python
# Pattern: fire-and-forget background side-effect that must not crash the graph
# The executor silently absorbs any exception from this task
submit(
    log_to_external_service,
    payload,
    __cancel_on_exit__=True,    # cancel if graph already crashed
    __reraise_on_exit__=False,  # don't surface logging errors
)
```

### `next_tick` — cooperative yielding in sync tasks

When `__next_tick__=True`, the task is wrapped with `time.sleep(0)` before execution. This yields control to other threads, allowing them to start before this one. Useful for ensuring the graph's main loop thread stays responsive.

```python
def next_tick(fn, *args, **kwargs):
    time.sleep(0)   # yield to OS scheduler
    return fn(*args, **kwargs)
```

---

## 3 · `AsyncBatchedBaseStore` + `_dedupe_ops`

**Module:** `langgraph.store.base.batch`  
**Inherits:** `BaseStore`

`AsyncBatchedBaseStore` is the base class for production-ready async stores (Redis, Postgres, etc.). Instead of issuing one round-trip per `aget`/`aput`/`asearch` call, it accumulates all operations queued in the *same asyncio tick* into a single `abatch` call, deduplicates reads and collapses consecutive puts to the same key.

### Architecture

```
caller code              AsyncBatchedBaseStore           background _run task
──────────                ──────────────────────          ───────────────────
await store.aget(...)  ─→  create Future, enqueue Op  ─→  drain queue each tick
await store.aput(...)  ─→  create Future, enqueue Op  │   _dedupe_ops(ops)
await store.asearch()  ─→  create Future, enqueue Op  │   await s.abatch(deduped)
                          [caller awaits future]        ─→  set_result on each Future
```

### Constructor + `_ensure_task`

```python
class AsyncBatchedBaseStore(BaseStore):
    __slots__ = ("_loop", "_aqueue", "_task")

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self._aqueue: asyncio.Queue[tuple[asyncio.Future, Op]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ensure_task()

    def _ensure_task(self) -> None:
        # Restart the background drainer if it died (e.g. exception)
        if self._task is None or self._task.done():
            self._task = self._loop.create_task(
                _run(self._aqueue, weakref.ref(self))  # weak ref prevents circular GC
            )
```

### Deadlock guard: `@_check_loop`

Calling the sync interface (`.get`, `.put`, etc.) from within the running event loop would deadlock — the sync method calls `run_coroutine_threadsafe(...).result()` which blocks the event loop thread that would process the future. The `@_check_loop` decorator detects this and raises `asyncio.InvalidStateError` with a helpful message:

```python
@_check_loop
def get(self, namespace, key, *, refresh_ttl=None):
    return asyncio.run_coroutine_threadsafe(
        self.aget(namespace, key=key, refresh_ttl=refresh_ttl), self._loop
    ).result()
```

```python
# Correct — async context inside graph nodes
async def my_node(state, *, store):
    item = await store.aget(("user", "prefs"), "theme")
    return {"result": item}

# Wrong — will raise asyncio.InvalidStateError
def my_node(state, *, store):
    item = store.get(("user", "prefs"), "theme")  # deadlock risk!
    return {"result": item}
```

### `_dedupe_ops` — how batching deduplicates work

```python
def _dedupe_ops(values: list[Op]) -> tuple[list[int] | None, list[Op]]:
    """Returns (listen_indices, deduped_ops).
    listen_indices maps each original op to its position in deduped_ops."""
    dedupped: list[Op] = []
    listen: list[int] = []
    puts: dict[tuple[tuple[str, ...], str], int] = {}  # (namespace, key) → deduped index

    for op in values:
        if isinstance(op, (GetOp, SearchOp, ListNamespacesOp)):
            try:
                listen.append(dedupped.index(op))  # exact duplicate → reuse result
            except ValueError:
                listen.append(len(dedupped))
                dedupped.append(op)
        elif isinstance(op, PutOp):
            putkey = (op.namespace, op.key)
            if putkey in puts:
                dedupped[puts[putkey]] = op   # overwrite earlier put to same key
                listen.append(puts[putkey])
            else:
                puts[putkey] = len(dedupped)
                listen.append(len(dedupped))
                dedupped.append(op)
    return listen, dedupped
```

### Implementing a custom `AsyncBatchedBaseStore`

```python
import asyncio
from langgraph.store.base import BaseStore, GetOp, PutOp, SearchOp, ListNamespacesOp, Op, Result, Item, SearchItem
from langgraph.store.base.batch import AsyncBatchedBaseStore
from typing import Any, Iterable
import json, time

class InMemoryAsyncStore(AsyncBatchedBaseStore):
    """Minimal in-memory store built on AsyncBatchedBaseStore for demonstration."""

    def __init__(self) -> None:
        self._data: dict[tuple[tuple[str, ...], str], dict] = {}
        super().__init__()   # starts the background drainer task

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                entry = self._data.get((op.namespace, op.key))
                results.append(
                    Item(
                        value=entry or {},
                        key=op.key,
                        namespace=op.namespace,
                        created_at=time.time(),
                        updated_at=time.time(),
                    ) if entry is not None else None
                )
            elif isinstance(op, PutOp):
                if op.value is None:
                    self._data.pop((op.namespace, op.key), None)
                else:
                    self._data[(op.namespace, op.key)] = op.value
                results.append(None)
            elif isinstance(op, SearchOp):
                ns = op.namespace_prefix
                matches = [
                    Item(
                        value=v,
                        key=k,
                        namespace=ns_key,
                        created_at=time.time(),
                        updated_at=time.time(),
                    )
                    for (ns_key, k), v in self._data.items()
                    if ns_key[:len(ns)] == ns
                ]
                results.append(matches[op.offset : op.offset + op.limit])
            elif isinstance(op, ListNamespacesOp):
                namespaces = {ns for (ns, _) in self._data}
                results.append(sorted(namespaces))
        return results


async def demo():
    store = InMemoryAsyncStore()
    # Multiple ops in same tick → batched into one abatch call
    await asyncio.gather(
        store.aput(("users",), "alice", {"name": "Alice"}),
        store.aput(("users",), "bob", {"name": "Bob"}),
    )
    alice = await store.aget(("users",), "alice")
    print(alice.value)   # {'name': 'Alice'}

asyncio.run(demo())
```

---

## 4 · `get_text_at_path` + `tokenize_path`

**Module:** `langgraph.store.base.embed`  
**Exported as:** `from langgraph.store.base.embed import get_text_at_path, tokenize_path`

These two functions power LangGraph Store's vector index: when you call `store.aput(..., index=["field.path"])`, the store uses `get_text_at_path` to extract the text that gets embedded. Understanding the path syntax lets you control precisely which parts of complex nested documents end up in your vector index.

### `tokenize_path` — parsing a path string

```python
def tokenize_path(path: str) -> list[str]:
    """Turn a dotted/bracketed path string into a list of tokens."""
```

| Path string | Tokens |
|---|---|
| `"title"` | `["title"]` |
| `"metadata.author"` | `["metadata", "author"]` |
| `"items[0].name"` | `["items", "[0]", "name"]` |
| `"items[*].text"` | `["items", "[*]", "text"]` |
| `"items[-1]"` | `["items", "[-1]"]` |
| `"*"` | `["*"]` |
| `"{title,summary}"` | `["{title,summary}"]` |
| `"{meta.author,title}"` | `["{meta.author,title}"]` |
| `"$"` | `[]` (returns full JSON) |

### `get_text_at_path` — extraction rules

```python
def get_text_at_path(obj: Any, path: str | list[str]) -> list[str]:
    """Extract zero or more text strings from obj at path."""
```

```python
from langgraph.store.base.embed import get_text_at_path, tokenize_path

doc = {
    "title": "LangGraph Guide",
    "metadata": {"author": "Alice", "tags": ["ai", "agents"]},
    "sections": [
        {"heading": "Intro", "body": "LangGraph is a framework."},
        {"heading": "Usage", "body": "Build graphs with nodes."},
    ],
}

# Simple field
print(get_text_at_path(doc, "title"))
# → ['LangGraph Guide']

# Nested field
print(get_text_at_path(doc, "metadata.author"))
# → ['Alice']

# Array wildcard
print(get_text_at_path(doc, "sections[*].heading"))
# → ['Intro', 'Usage']

# Negative index
print(get_text_at_path(doc, "sections[-1].body"))
# → ['Build graphs with nodes.']

# Multi-field selection (both fields extracted, one text per field)
print(get_text_at_path(doc, "{title,metadata.author}"))
# → ['LangGraph Guide', 'Alice']

# Wildcard over dict values
print(get_text_at_path({"a": "hello", "b": "world"}, "*"))
# → ['hello', 'world']

# Full-document JSON ($)
print(get_text_at_path(doc, "$"))
# → ['{"metadata": {...}, "sections": [...], "title": "LangGraph Guide"}']

# Empty path — same as $
print(get_text_at_path(doc, ""))
# → ['{"metadata": {...}, ...}']
```

### Controlling what gets embedded in the store

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base.embed import EmbeddingsLambda

# Mock embedder — returns a fixed-dim vector based on text length
def mock_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t))] * 4 for t in texts]

store = InMemoryStore(index={"embed": mock_embed, "dims": 4})

# Index only the 'body' fields of each section
await store.aput(
    ("docs",), "guide",
    {
        "title": "LangGraph Guide",
        "sections": [
            {"heading": "Intro", "body": "LangGraph is a framework."},
            {"heading": "Usage", "body": "Build graphs with nodes."},
        ],
    },
    index=["sections[*].body"],
)

# Index multiple fields together
await store.aput(
    ("articles",), "article-1",
    {"title": "State Management", "abstract": "How state works in LangGraph."},
    index=["{title,abstract}"],
)

# Disable indexing entirely for a private document
await store.aput(
    ("private",), "secret",
    {"content": "internal only"},
    index=False,
)
```

### Pre-tokenising paths for performance

When embedding thousands of documents with the same path, call `tokenize_path` once and pass the token list directly:

```python
path_tokens = tokenize_path("sections[*].body")
texts = get_text_at_path(large_doc, path_tokens)  # skips re-tokenising
```

---

## 5 · `SerdeEvent` + `register_serde_event_listener`

**Module:** `langgraph.checkpoint.serde.event_hooks`  
**Exported as:** `from langgraph.checkpoint.serde.event_hooks import SerdeEvent, register_serde_event_listener, emit_serde_event`

When LangGraph's `JsonPlusSerializer` serialises or deserialises a custom Python type for checkpointing, it emits a `SerdeEvent`. Registering a listener lets you build runtime allowlists, audit which types transit the checkpoint layer, or warn when unexpected types appear.

### Types

```python
from typing_extensions import TypedDict, NotRequired

class SerdeEvent(TypedDict):
    kind: str             # "serialise" | "deserialise"
    module: str           # e.g. "myapp.models"
    name: str             # e.g. "OrderPayload"
    method: NotRequired[str]   # e.g. "__reduce__" — only present for some paths

SerdeEventListener = Callable[[SerdeEvent], None]
```

### `register_serde_event_listener`

```python
def register_serde_event_listener(listener: SerdeEventListener) -> Callable[[], None]:
    """Returns an unregister callable."""
    with _listeners_lock:
        _listeners.append(listener)

    def unregister() -> None:
        with _listeners_lock:
            try:
                _listeners.remove(listener)
            except ValueError:
                pass

    return unregister
```

### `emit_serde_event`

```python
def emit_serde_event(event: SerdeEvent) -> None:
    """Isolated dispatch — listener exceptions are logged and swallowed."""
    with _listeners_lock:
        listeners = tuple(_listeners)
    for listener in listeners:
        try:
            listener(event)
        except Exception:
            logger.warning("Serde listener failed", exc_info=True)
```

### Pattern 1: auditing which types pass through the checkpoint

```python
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener, SerdeEvent
from collections import Counter

type_counter: Counter[str] = Counter()

def audit_listener(event: SerdeEvent) -> None:
    type_counter[f"{event['module']}.{event['name']}"] += 1

unregister = register_serde_event_listener(audit_listener)

# ... run your graph ...

print(type_counter.most_common(10))
unregister()   # clean up when done
```

### Pattern 2: building a dynamic allowlist

```python
ALLOWED_TYPES: set[str] = set()
BLOCKED_EVENTS: list[SerdeEvent] = []

def allowlist_listener(event: SerdeEvent) -> None:
    fqn = f"{event['module']}.{event['name']}"
    if fqn not in ALLOWED_TYPES:
        BLOCKED_EVENTS.append(event)
        # Could raise here to block serialisation, but that breaks checkpointing
        # Better: log and alert, then fix ALLOWED_TYPES
        print(f"WARN: unexpected type in checkpoint: {fqn}")

unregister = register_serde_event_listener(allowlist_listener)

ALLOWED_TYPES.update({
    "myapp.models.OrderPayload",
    "myapp.models.UserContext",
    "builtins.dict",
    "builtins.list",
})
```

### Pattern 3: using as a context manager

```python
from contextlib import contextmanager

@contextmanager
def capture_serde_events():
    events: list[SerdeEvent] = []
    unregister = register_serde_event_listener(events.append)
    try:
        yield events
    finally:
        unregister()

# Use in tests
with capture_serde_events() as events:
    result = graph.invoke({"messages": [...]}, config)

serde_types = {f"{e['module']}.{e['name']}" for e in events}
assert "myapp.Order" in serde_types
```

---

## 6 · `BaseChannel`

**Module:** `langgraph.channels.base`  
**Exported as:** `from langgraph.channels.base import BaseChannel`

`BaseChannel[Value, Update, Checkpoint]` is the abstract base class every state channel in LangGraph inherits from. Understanding its three type parameters and six abstract/overridable methods is essential for writing custom channels — e.g., capped lists, sliding windows, priority queues.

### Source (full)

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar
from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.errors import EmptyChannelError

Value = TypeVar("Value")
Update = TypeVar("Update")
Checkpoint = TypeVar("Checkpoint")

class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ    # Python type annotation for this channel
        self.key = key    # channel name (set by StateGraph)

    @property
    @abstractmethod
    def ValueType(self) -> Any: ...   # the type of get()

    @property
    @abstractmethod
    def UpdateType(self) -> Any: ...  # the type accepted by update()

    def copy(self) -> Self:
        """Efficient copy via checkpoint round-trip by default."""
        return self.from_checkpoint(self.checkpoint())

    def checkpoint(self) -> Checkpoint | Any:
        """Serialisable snapshot — defaults to get(), MISSING if empty."""
        try:
            return self.get()
        except EmptyChannelError:
            return MISSING

    @abstractmethod
    def from_checkpoint(self, checkpoint: Checkpoint | Any) -> Self: ...

    @abstractmethod
    def get(self) -> Value: ...          # raises EmptyChannelError if empty

    def is_available(self) -> bool:      # override for efficiency
        try:
            self.get()
            return True
        except EmptyChannelError:
            return False

    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool:
        """Called by Pregel with all updates from this step. Returns True if changed."""

    def consume(self) -> bool:
        """Called after a subscribed task ran. Override for ephemeral channels."""
        return False

    def finish(self) -> bool:
        """Called when the Pregel run is finishing. Override to do cleanup."""
        return False
```

### Type parameters

| Parameter | Meaning | Example |
|---|---|---|
| `Value` | What `get()` returns | `list[str]` |
| `Update` | What individual writes pass to `update()` | `str` |
| `Checkpoint` | What is persisted in the checkpointer | `list[str]` |

### Custom channel example: capped ring buffer

```python
from collections import deque
from typing import Any
from typing_extensions import Self
from langgraph.channels.base import BaseChannel
from langgraph.errors import EmptyChannelError

class RingBufferChannel(BaseChannel[list[str], str, list[str]]):
    """Keeps only the last `maxlen` messages."""

    def __init__(self, typ: Any, key: str = "", maxlen: int = 5) -> None:
        super().__init__(typ, key)
        self._maxlen = maxlen
        self._buf: deque[str] = deque(maxlen=maxlen)

    @property
    def ValueType(self) -> Any:
        return list[str]

    @property
    def UpdateType(self) -> Any:
        return str

    def from_checkpoint(self, checkpoint: list[str] | Any) -> Self:
        new = self.__class__(self.typ, self.key, self._maxlen)
        if checkpoint and checkpoint is not ...:  # MISSING sentinel is Ellipsis
            new._buf = deque(checkpoint, maxlen=self._maxlen)
        return new

    def get(self) -> list[str]:
        if not self._buf:
            raise EmptyChannelError()
        return list(self._buf)

    def is_available(self) -> bool:
        return bool(self._buf)

    def update(self, values: list[str]) -> bool:
        if not values:
            return False
        for v in values:
            self._buf.append(v)
        return True


# Use in a graph via Annotated
from typing import Annotated
from langgraph.graph import StateGraph, START, END

def make_ring(typ: Any) -> RingBufferChannel:
    return RingBufferChannel(typ, maxlen=3)

class State(TypedDict):
    log: Annotated[list[str], make_ring]

def node_a(s: State) -> dict:
    return {"log": "step-a"}

def node_b(s: State) -> dict:
    return {"log": "step-b"}

def node_c(s: State) -> dict:
    return {"log": "step-c"}

def node_d(s: State) -> dict:
    return {"log": "step-d"}

builder = StateGraph(State)
for n, fn in [("a", node_a), ("b", node_b), ("c", node_c), ("d", node_d)]:
    builder.add_node(n, fn)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", "c")
builder.add_edge("c", "d")
builder.add_edge("d", END)

graph = builder.compile()
result = graph.invoke({"log": []})
print(result["log"])  # ['step-b', 'step-c', 'step-d']  (ring of 3, 'step-a' dropped)
```

### The `consume` / `finish` hooks

`consume()` is called after every task that subscribes to this channel has executed. The built-in `EphemeralValue` channel overrides it to reset its value — that is how `EphemeralValue` is "ephemeral".

`finish()` is called when the Pregel run loop is about to exit. `NamedBarrierValue` uses `finish()` to release any tasks still waiting on the barrier.

```python
class MyEphemeralChannel(BaseChannel[str, str, None]):
    """Value disappears after it has been consumed."""
    def __init__(self, typ, key=""):
        super().__init__(typ, key)
        self._value: str | None = None

    @property
    def ValueType(self): return str
    @property
    def UpdateType(self): return str

    def from_checkpoint(self, checkpoint) -> Self:
        new = self.__class__(self.typ, self.key)
        new._value = checkpoint
        return new

    def checkpoint(self):
        return self._value  # always serialise even if None

    def get(self) -> str:
        if self._value is None:
            raise EmptyChannelError()
        return self._value

    def update(self, values: list[str]) -> bool:
        if values:
            self._value = values[-1]
            return True
        return False

    def consume(self) -> bool:
        """Clear after being read."""
        if self._value is not None:
            self._value = None
            return True
        return False
```

---

## 7 · `call()` + `SyncAsyncFuture`

**Module:** `langgraph.pregel._call`  
**Public re-export:** `from langgraph.types import call`

`call()` is the low-level primitive that powers the `@task` decorator. Inside a `@entrypoint` or `@task` function, `call(fn, *args, ...)` dispatches `fn` as a sub-task with its own optional `RetryPolicy`, `CachePolicy`, and `TimeoutPolicy`. It returns a `SyncAsyncFuture` — a `concurrent.futures.Future` subclass that is also awaitable, so it works identically in both sync and async entrypoints.

### `SyncAsyncFuture` source

```python
import concurrent.futures
from typing import Generator, Generic, TypeVar, cast

T = TypeVar("T")

class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T]):
    def __await__(self) -> Generator[T, None, T]:
        yield cast(T, ...)
```

The single-yield `__await__` suspends the async caller until the future is resolved by the Pregel engine. In a sync context the `Future` is simply `.result()`-ed when collected.

### `call()` signature

```python
from langgraph.types import call
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy
from datetime import timedelta

def call(
    func: Callable[P, Awaitable[T]] | Callable[P, T],
    *args: Any,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
    **kwargs: Any,
) -> SyncAsyncFuture[T]:
    ...
```

> **Note:** `timeout` is only supported for async `func`. Passing a `timeout` with a sync function raises `NotImplementedError` at call time.

### Example 1: parallel fan-out with retry

```python
from langgraph.func import entrypoint, task
from langgraph.types import call, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver
import asyncio

@task
async def fetch_url(url: str) -> str:
    # Simulated network call
    await asyncio.sleep(0.05)
    return f"content of {url}"

@entrypoint(checkpointer=InMemorySaver())
async def research_agent(urls: list[str]) -> list[str]:
    # Fan out: call() dispatches each fetch as an independent sub-task
    futures = [call(fetch_url, url) for url in urls]
    # Fan in: await all at once
    results = await asyncio.gather(*futures)
    return list(results)

output = await research_agent.ainvoke(
    ["https://a.example", "https://b.example", "https://c.example"],
    config={"configurable": {"thread_id": "research-1"}},
)
print(output)  # ['content of https://a.example', ...]
```

### Example 2: per-call retry + cache

```python
from langgraph.types import CachePolicy, RetryPolicy

@task
async def call_llm(prompt: str) -> str:
    # ... expensive LLM call
    return f"response to: {prompt}"

@entrypoint(checkpointer=InMemorySaver())
async def pipeline(prompts: list[str]) -> list[str]:
    futures = [
        call(
            call_llm,
            prompt,
            retry_policy=[RetryPolicy(max_attempts=3)],
            cache_policy=CachePolicy(),   # cache successful results
            timeout=30.0,                 # 30-second per-call timeout
        )
        for prompt in prompts
    ]
    return await asyncio.gather(*futures)
```

### Example 3: sync entrypoint with `call()`

`call()` also works in synchronous `@entrypoint` functions. Collect the futures, then call `.result()` on each:

```python
@task
def compute_score(item: str) -> float:
    return float(len(item)) / 100.0

@entrypoint(checkpointer=InMemorySaver())
def scoring_pipeline(items: list[str]) -> list[float]:
    futures = [call(compute_score, item) for item in items]
    return [f.result() for f in futures]

result = scoring_pipeline.invoke(
    ["apple", "banana", "cherry"],
    config={"configurable": {"thread_id": "score-1"}},
)
print(result)  # [0.05, 0.06, 0.06]
```

### How `call()` routes through the engine

`call()` reads `CONFIG_KEY_CALL` from the current LangGraph config context and delegates to the engine's internal `impl` function. This means it **only works inside active Pregel execution** — calling it outside an `@entrypoint` or `@task` raises a `KeyError`.

```python
from langgraph._internal._constants import CONF, CONFIG_KEY_CALL
from langgraph.config import get_config

def call(func, *args, **options) -> SyncAsyncFuture:
    config = get_config()          # must be inside Pregel execution
    impl = config[CONF][CONFIG_KEY_CALL]
    return impl(func, (args, kwargs), **options)
```

---

## 8 · `PregelScratchpad`

**Module:** `langgraph._internal._scratchpad`

`PregelScratchpad` is a dataclass that Pregel creates once per entrypoint invocation and injects into the execution context. It carries all the mutable counters and callbacks needed for a single run: step tracking, interrupt/resume state, and subgraph dispatch.

### Source

```python
import dataclasses
from collections.abc import Callable
from typing import Any
from langgraph.types import _DC_KWARGS

@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int                              # current step index (0-based)
    stop: int                              # step at which to stop (-1 = unlimited)
    # Task API counters
    call_counter: Callable[[], int]        # returns next call index, used for cache keys
    # Interrupt / resume API
    interrupt_counter: Callable[[], int]   # returns next interrupt index
    get_null_resume: Callable[[bool], Any] # fetches a resume value or None
    resume: list[Any]                      # queued resume values from interrupt()
    # Subgraph dispatch
    subgraph_counter: Callable[[], int]    # returns next subgraph index
```

`_DC_KWARGS` sets `slots=True, frozen=False` on Python 3.10+ for efficient memory layout.

### What each field does

| Field | Set by | Used by |
|---|---|---|
| `step` | Pregel loop iteration | Debug output, `IsLastStep` check |
| `stop` | `interrupt_before`/`interrupt_after` config | Loop termination guard |
| `call_counter` | Pregel on each invocation | `call()` for stable cache key generation |
| `interrupt_counter` | Pregel on each invocation | `interrupt()` for stable resume index |
| `get_null_resume` | Pregel, reads checkpoint | `interrupt()` to replay non-null resumes |
| `resume` | Populated from checkpoint on resume | `interrupt()` to deliver queued values |
| `subgraph_counter` | Pregel on each invocation | Stable subgraph thread-id suffix |

### Reading `PregelScratchpad` for debugging

The scratchpad is stored under a well-known config key. Inside a running node you can access it:

```python
from langgraph.config import get_config
from langgraph._internal._constants import CONF, CONFIG_KEY_SCRATCHPAD

def debug_node(state):
    config = get_config()
    scratchpad = config[CONF].get(CONFIG_KEY_SCRATCHPAD)
    if scratchpad:
        print(f"  step={scratchpad.step}, stop={scratchpad.stop}")
        print(f"  queued resumes={len(scratchpad.resume)}")
    return state

# Useful for verifying HITL resume payloads during development
```

### Why `call_counter` / `interrupt_counter` matter

These counters exist so that repeated invocations of `call()` or `interrupt()` inside the same node always produce the *same* index, making cache lookups and checkpoint replay deterministic:

```python
@entrypoint(checkpointer=InMemorySaver())
async def agent(msgs):
    # First run: counters produce index 0, 1, 2 ...
    # Resume run: same indices → same cache hits
    a = call(llm_call, msgs[0])
    b = call(llm_call, msgs[1])
    return await asyncio.gather(a, b)
```

---

## 9 · `StateNodeSpec` + node Protocols

**Module:** `langgraph.graph._node`  
**Exported as:** `from langgraph.graph._node import StateNodeSpec, StateNode`

`StateNodeSpec` is the internal record that `StateGraph` stores for every `add_node` call. Each entry captures the wrapped runnable, metadata, inferred input schema, retry/cache/timeout policies, and flags for error-handler nodes and deferred execution. The companion `StateNode` type alias and the seven `_Node*` Protocol variants define the full set of valid node signatures LangGraph understands.

### Node Protocols

```python
class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...

class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...

class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...

class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...

class _NodeWithRuntime(Protocol[NodeInputT_contra, ContextT]):
    def __call__(self, state: NodeInputT_contra, *, runtime: Runtime[ContextT]) -> Any: ...

# ... plus _NodeWithWriterStore, _NodeWithConfigWriter, _NodeWithConfigStore, _NodeWithConfigWriterStore
```

LangGraph inspects the node's call signature at `add_node` time and automatically injects the requested keyword arguments. You never need to call these Protocols directly — just type your function and LangGraph figures it out.

### `StateNodeSpec` source

```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]       # the wrapped Runnable
    metadata: dict[str, Any] | None                 # user-supplied metadata
    input_schema: type[NodeInputT]                  # inferred or explicit input type
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    is_error_handler: bool = False                  # True when error_handler=True
    error_handler_node: str | None = None           # node this handler is attached to
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False                             # deferred node flag
    timeout: TimeoutPolicy | None = None
```

### Introspecting node specs at runtime

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

builder = StateGraph(State)
builder.add_node("compute", lambda s: {"value": s["value"] * 2})
builder.add_node(
    "summarise",
    lambda s: {"value": s["value"]},
    retry=RetryPolicy(max_attempts=3),
    metadata={"description": "Summarises the result"},
)
builder.add_edge(START, "compute")
builder.add_edge("compute", "summarise")
builder.add_edge("summarise", END)

# Access the underlying node specs before compilation
for name, spec in builder.nodes.items():
    if isinstance(spec, StateNodeSpec):
        print(f"{name}: retry={spec.retry_policy}, cache={spec.cache_policy}")
        print(f"       metadata={spec.metadata}, defer={spec.defer}")
```

### Using all supported node signatures

```python
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime

# 1. Minimal — just state
def simple_node(state: State) -> dict:
    return {"value": state["value"] + 1}

# 2. With config — access thread_id, tags, metadata
def config_node(state: State, config: RunnableConfig) -> dict:
    thread = config["configurable"].get("thread_id", "?")
    return {"value": state["value"]}

# 3. With writer — push custom stream events mid-node
def streaming_node(state: State, *, writer: StreamWriter) -> dict:
    writer({"progress": "halfway"})
    return {"value": state["value"] * 2}

# 4. With store — read/write long-term memory
async def memory_node(state: State, *, store: BaseStore) -> dict:
    item = await store.aget(("user",), "profile")
    return {"value": state["value"]}

# 5. With runtime — access typed context (requires context_schema)
def runtime_node(state: State, *, runtime: Runtime) -> dict:
    ctx = runtime.context   # typed context object
    return {"value": state["value"]}

builder = StateGraph(State)
builder.add_node("simple", simple_node)
builder.add_node("cfg", config_node)
builder.add_node("stream", streaming_node)
builder.add_node("mem", memory_node)
```

### `defer=True` — scheduling a node for the next step

Setting `defer=True` on a node makes it run at the *end* of the current step rather than alongside other ready nodes. This is useful for aggregation nodes that should run only after all parallel branches have completed.

```python
builder.add_node("aggregate", aggregate_fn, defer=True)
# aggregate_fn will not run until all other nodes in the current step have finished
```

---

## 10 · `identifier` + `get_runnable_for_task` + `get_runnable_for_entrypoint`

**Module:** `langgraph.pregel._call`  
**Internal utilities — not part of the public API**

These three functions reveal how LangGraph transforms ordinary Python callables into the `Runnable` instances that Pregel executes. Understanding them explains:

- Why task/entrypoint names appear the way they do in traces
- How the `CACHE` avoids wrapping the same function twice
- Why changing a function's `__qualname__` breaks cache hits
- How `@task` adds the automatic `RETURN` channel write

### `identifier` — canonical name for a callable

```python
def identifier(obj: Any, name: str | None = None) -> str | None:
    """Return 'module.qualname' — the stable import path for obj."""
    # Unwraps PregelNode → RunnableSeq → RunnableCallable → raw function
    from langgraph.pregel._read import PregelNode
    from langgraph._internal._runnable import RunnableCallable, RunnableSeq

    if isinstance(obj, PregelNode):   obj = obj.bound
    if isinstance(obj, RunnableSeq):  obj = obj.steps[0]
    if isinstance(obj, RunnableCallable): obj = obj.func

    name = name or getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    module_name = getattr(obj, "__module__", None)
    if name is None or module_name is None:
        return None
    return f"{module_name}.{name}"
```

**Why it matters:** The cache key for `get_runnable_for_task` / `get_runnable_for_entrypoint` is `(func, is_task)`. If `identifier(func)` returns `None` (local/lambda/dynamic function), the result is **not cached**, so a new `RunnableCallable` is built on every call. This is a subtle performance gotcha with lambda nodes.

```python
from langgraph.pregel._call import identifier

def my_tool(x: int) -> int:
    return x * 2

print(identifier(my_tool))      # '__main__.my_tool'

lam = lambda x: x * 2
print(identifier(lam))          # None — lambdas have no stable qualname for caching
```

### `get_runnable_for_entrypoint`

```python
CACHE: dict[tuple[Callable, bool], Runnable] = {}

def get_runnable_for_entrypoint(func: Callable) -> Runnable:
    """Wrap an entrypoint function in a RunnableCallable (with async fallback)."""
    key = (func, False)
    if key in CACHE:
        return CACHE[key]
    if is_async_callable(func):
        run = RunnableCallable(None, func, name=func.__name__, trace=False, recurse=False)
    else:
        # Create an async wrapper that runs the sync function in an executor
        afunc = functools.update_wrapper(
            functools.partial(run_in_executor, None, func), func
        )
        run = RunnableCallable(func, afunc, name=func.__name__, trace=False, recurse=False)
    # Only cache if the function has a stable import path
    if not _lookup_module_and_qualname(func):
        return run
    return CACHE.setdefault(key, run)
```

### `get_runnable_for_task`

```python
def get_runnable_for_task(func: Callable) -> Runnable:
    """Wrap a task function — adds an automatic ChannelWrite([RETURN]) at the end."""
    key = (func, True)
    if key in CACHE:
        return CACHE[key]
    name = getattr(func, "__name__", None) or func.__class__.__name__
    if is_async_callable(func):
        run = RunnableCallable(None, func, explode_args=True, name=name, ...)
    else:
        run = RunnableCallable(func, ..., explode_args=True, name=name, ...)

    seq = RunnableSeq(
        run,
        ChannelWrite([ChannelWriteEntry(RETURN)]),   # ← this is the key difference
        name=name,
        trace_inputs=functools.partial(_explode_args_trace_inputs, inspect.signature(func)),
    )
    if not _lookup_module_and_qualname(func):
        return seq
    return CACHE.setdefault(key, seq)
```

The `ChannelWrite([ChannelWriteEntry(RETURN)])` appended to every task's runnable is what makes `await task_future` work — the task writes its return value to the `RETURN` channel, which the `call()` implementation then reads.

### Practical implications

```python
# These two are cached — stable module.qualname
@task
def process_item(item: str) -> str:
    return item.upper()

@task
async def async_process(item: str) -> str:
    return item.upper()

# This is NOT cached — lambda has no qualname
builder.add_node("transform", lambda s: {"value": s["value"] * 2})
# A new RunnableCallable is created on every compile() call

# Fix: give it a name
def transform(s):
    return {"value": s["value"] * 2}
builder.add_node("transform", transform)  # now cached
```

### Using `identifier` in custom observability

```python
from langgraph.pregel._call import identifier

class NodeTracer:
    """Logs which function handles each node invocation."""

    def __init__(self, graph):
        self._map = {}
        for name, spec in graph.nodes.items():
            fid = identifier(spec.runnable)
            self._map[name] = fid or f"<anonymous:{name}>"

    def report(self):
        for node, fid in self._map.items():
            print(f"  {node} → {fid}")

graph = builder.compile()
tracer = NodeTracer(graph)
tracer.report()
# compute → __main__.compute_fn
# transform → <anonymous:transform>
```

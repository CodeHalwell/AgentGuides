---
title: "LangGraph Class Deep-Dives Vol. 37"
description: "Source-verified deep dives into 10 class groups from langgraph==1.2.8 — StreamMessagesHandler/StreamMessagesHandlerV2 (stream_mode=messages callback pair), ChannelRead (imperatively read channel state inside a node), set_config_context/create_task_in_config_context (thread-safe config propagation), chain_future/run_coroutine_threadsafe (cross-thread async bridge), map_debug_checkpoint/tasks_w_writes/rm_pregel_keys (debug checkpoint formatters), AsyncSubgraphRunStream (async discovered-subgraph handle), _triggers/_scratchpad (PULL task activation and resume state assembly), _proc_input (channel input collection and input cache), prepare_single_task/_TaskIDFn (top-level PUSH/PULL task dispatcher), and _get_channels/_get_channel/_is_field_channel/_is_field_binop (StateGraph annotation-to-channel inference)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 37"
  order: 68
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.8` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `StreamMessagesHandler` + `StreamMessagesHandlerV2`

**Module:** `langgraph.pregel._messages`

`StreamMessagesHandler` is the `langchain_core.callbacks.BaseCallbackHandler` subclass that powers `stream_mode="messages"`. It is instantiated by the Pregel loop and installed into the LangChain callback stack before each step. Two sources of messages are collected:

1. **Chat model stream events** — `on_chat_model_start` records namespace metadata; `on_llm_new_token` / `on_llm_end` emit `AIMessageChunk` / `AIMessage` objects through the stream.
2. **Node outputs** — `on_chain_end` walks the returned value through `_find_and_emit_messages`, which handles `BaseMessage` directly, lists of messages, and state dicts / Pydantic models / dataclasses (via `_state_values`).

**Key source facts** (from `langgraph/pregel/_messages.py`):

- `run_inline = True` — forces the handler to execute synchronously in the main thread, avoiding ordering and locking races with the streaming channel.
- `_emit(meta, message, *, dedupe=False)` — deduplicates by `message.id`; assigns a random UUID if `id is None`. The tuple `(meta[0], "messages", (message, meta[1]))` is the `StreamChunk` written to the graph stream.
- `parent_ns` — the namespace where the handler was installed. Subgraph events are filtered to `subgraphs=True` unless their namespace has an explicit `stream_mode="messages"` subscription.
- `StreamMessagesHandlerV2` extends `StreamMessagesHandler` and also mixes in `_V2StreamingCallbackHandler`; the `on_stream_event` hook collects `on_chat_model_stream` events for the v2 streaming surface, giving richer chunk metadata.
- `_find_and_emit_messages` walks `BaseMessage`, `Sequence[BaseMessage]`, state dict values, Pydantic model field values, and dataclass field values recursively — a node returning `{"messages": [AIMessage(...)]}` is handled automatically.

### Example 1 — observe how `stream_mode="messages"` surfaces chunks from an LLM inside a node

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from typing_extensions import TypedDict

try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    HAS_LLM = True
except Exception:
    HAS_LLM = False


class State(TypedDict):
    messages: list


def call_model(state: State, *, writer: StreamWriter) -> dict:
    if HAS_LLM:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    # Fallback: write a synthetic AIMessage so the example runs offline
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content="hello from node")]}


builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile()

# stream_mode="messages" (single string, no subgraphs) yields the payload
# directly — a (message_chunk, metadata) pair per emission.
# Use a list ["messages"] + subgraphs=True to get (ns, kind, payload) triples.
chunks = []
for msg, meta in graph.stream(
    {"messages": [HumanMessage(content="hi")]},
    stream_mode="messages",
):
    chunks.append(msg)

print(f"collected {len(chunks)} message chunk(s)")
# The last chunk is always the complete AIMessage / final chunk
if chunks:
    print(type(chunks[-1]).__name__)
```

### Example 2 — inspect `_state_values` to understand which node outputs trigger message emission

```python
from dataclasses import dataclass
from langchain_core.messages import AIMessage
from pydantic import BaseModel

# _state_values extracts top-level field values from any state shape
from langgraph.pregel._messages import _state_values

# dict — most common state type
state_dict = {"messages": [AIMessage(content="hi")], "count": 3}
print("dict values:", [type(v).__name__ for v in _state_values(state_dict)])
# ['list', 'int']

# Pydantic model
class PydanticState(BaseModel):
    messages: list
    count: int

ps = PydanticState(messages=[AIMessage(content="hi")], count=3)
print("pydantic values:", [type(v).__name__ for v in _state_values(ps)])
# ['list', 'int']

# dataclass
@dataclass
class DCState:
    messages: list
    count: int

ds = DCState(messages=[AIMessage(content="hi")], count=3)
print("dataclass values:", [type(v).__name__ for v in _state_values(ds)])
# ['list', 'int']

# Non-standard object — returns empty tuple, no messages emitted
print("unknown:", list(_state_values(42)))
# []
```

### Example 3 — subgraph message filtering with `subgraphs=True`

```python
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class InnerState(TypedDict):
    messages: list


def inner_node(state: InnerState) -> dict:
    return {"messages": state["messages"] + [AIMessage(content="inner reply")]}


inner = StateGraph(InnerState)
inner.add_node("inner_node", inner_node)
inner.add_edge(START, "inner_node")
inner.add_edge("inner_node", END)
inner_graph = inner.compile()


class OuterState(TypedDict):
    messages: list


def outer_node(state: OuterState) -> dict:
    # Call the inner graph as a subgraph
    result = inner_graph.invoke(state)
    return {"messages": result["messages"]}


outer = StateGraph(OuterState)
outer.add_node("outer_node", outer_node)
outer.add_edge(START, "outer_node")
outer.add_edge("outer_node", END)
outer_graph = outer.compile()

# subgraphs=False (default): only messages from the outer graph appear
outer_chunks = list(outer_graph.stream(
    {"messages": [HumanMessage(content="hello")]},
    stream_mode="messages",
    subgraphs=False,
))
print(f"outer-only chunks: {len(outer_chunks)}")

# subgraphs=True: messages from every depth appear
all_chunks = list(outer_graph.stream(
    {"messages": [HumanMessage(content="hello")]},
    stream_mode="messages",
    subgraphs=True,
))
print(f"all-depth chunks: {len(all_chunks)}")
```

---

## 2 · `ChannelRead`

**Module:** `langgraph.pregel._read`

`ChannelRead` is a `RunnableCallable` that reads one or more channels from the live graph state inside an executing node. It works by pulling the `CONFIG_KEY_READ` function out of the task's `RunnableConfig` — a lambda over the channel dict that is installed by `prepare_single_task`. Its static `do_read` method is the low-level primitive; the class provides a chainable `Runnable` surface around it.

**Key source facts** (from `langgraph/pregel/_read.py`):

- `channel: str | list[str]` — a single key returns the raw value; a list returns a `dict[str, Any]`.
- `fresh: bool = False` — when `True`, applies the current task's own pending writes to a local copy of the channels before reading (post-write view). When `False` (default), reads the channel snapshot before any writes in this step (pre-write view).
- `mapper: Callable[[Any], Any] | None` — applied to the result before returning; lets `ChannelRead` be used as a transform in a `RunnableSeq`.
- `do_read(config, *, select, fresh, mapper)` — the static method; raises `RuntimeError` if called outside a Pregel execution context.
- `get_name()` — returns `"ChannelRead<channel>"` for display in traces.

### Example 1 — read a single channel inside a node

```python
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import ChannelRead


class State(TypedDict):
    value: int
    doubled: int


def read_and_double(state: State, config: RunnableConfig) -> dict:
    # ChannelRead.do_read lets us re-read the channel mid-execution
    current = ChannelRead.do_read(config, select="value")
    return {"doubled": current * 2}


builder = StateGraph(State)
builder.add_node("node", read_and_double)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()
result = graph.invoke({"value": 7, "doubled": 0})
print(result)  # {'value': 7, 'doubled': 14}
```

### Example 2 — read multiple channels at once

```python
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import ChannelRead


class State(TypedDict):
    x: int
    y: int
    summary: str


def summarize(state: State, config: RunnableConfig) -> dict:
    # Passing a list returns a dict of channel values
    vals = ChannelRead.do_read(config, select=["x", "y"])
    return {"summary": f"x={vals['x']}, y={vals['y']}, sum={vals['x'] + vals['y']}"}


builder = StateGraph(State)
builder.add_node("summarize", summarize)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
result = graph.invoke({"x": 3, "y": 4, "summary": ""})
print(result["summary"])  # x=3, y=4, sum=7
```

### Example 3 — `ChannelRead` as a `Runnable` with a `mapper`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import ChannelRead
from langchain_core.runnables import RunnableLambda


class State(TypedDict):
    items: list
    count: int


# Build a Runnable that reads "items" and maps it to its length
reader = ChannelRead("items", mapper=len)
print(reader.get_name())  # ChannelRead<items>


builder = StateGraph(State)
# Wire reader directly as the node — it reads the "items" channel via CONFIG_KEY_READ
# and pipes the length into a dict update via the lambda
builder.add_node("update", reader | RunnableLambda(lambda count: {"count": count}))
builder.add_edge(START, "update")
builder.add_edge("update", END)

graph = builder.compile()
result = graph.invoke({"items": [1, 2, 3, 4], "count": 0})
print(result["count"])  # 4
```

---

## 3 · `set_config_context` + `create_task_in_config_context`

**Module:** `langgraph._internal._runnable`

LangChain uses a `contextvars.ContextVar` (`var_child_runnable_config`) to propagate the current `RunnableConfig` through the call stack without passing it explicitly. `set_config_context` is a context manager that sets this variable safely within an isolated `contextvars.copy_context()` scope. `create_task_in_config_context` wraps `asyncio.create_task` so that newly spawned tasks inherit the config from the calling coroutine.

**Key source facts** (from `langgraph/_internal/_runnable.py`):

- `_set_config_context(config, run)` — sets `var_child_runnable_config` and, when `run` is provided, also sets the LangSmith tracing context variable (`_context_var`) so traces are correctly parented.
- `_unset_config_context(token, run)` — restores the previous context variable values via the saved token.
- `set_config_context(config, run=None)` — acquires a fresh `copy_context()` so mutations inside the `with` block don't leak to the caller; yields the `Context` object so callers can `ctx.run(...)` inside the same context.
- `create_task_in_config_context(coro_factory, config)` — calls `set_config_context` to enter the correct context, then calls `context.run(lambda: asyncio.create_task(coro_factory()))`. `asyncio.create_task` snapshots the current `contextvars.Context` onto the new task, so the task permanently inherits `config` without the caller needing to keep the `with` block open.
- This function is what Pregel uses internally when forking node tasks in the async loop; user code can use it to spawn background async work that correctly inherits the run's tracing context.

### Example 1 — propagate `RunnableConfig` into a background `asyncio.Task`

```python
import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph._internal._runnable import (
    create_task_in_config_context,
    set_config_context,
)

# Simulate the context-var that LangChain uses internally
try:
    from langchain_core.runnables.config import var_child_runnable_config
    HAS_VAR = True
except ImportError:
    HAS_VAR = False


async def worker() -> str:
    """Read the propagated config from the ContextVar to prove inheritance."""
    if HAS_VAR:
        cfg = var_child_runnable_config.get(None)
        tags = cfg.get("tags", []) if cfg else []
        return f"tags={tags}"
    return "no var_child_runnable_config in this langchain version"


async def main():
    config: RunnableConfig = {"tags": ["example"], "metadata": {"step": 1}}

    # The task inherits the config ContextVar snapshot set by set_config_context
    task = create_task_in_config_context(worker, config)
    result = await task
    print(result)  # tags=['example']


asyncio.run(main())
```

### Example 2 — `set_config_context` for thread-local config isolation

```python
import contextvars
from langchain_core.runnables import RunnableConfig
from langgraph._internal._runnable import set_config_context


config_a: RunnableConfig = {"tags": ["branch-a"]}
config_b: RunnableConfig = {"tags": ["branch-b"]}


def run_in_context(cfg: RunnableConfig) -> None:
    with set_config_context(cfg) as ctx:
        # ctx.run executes a callable inside this isolated context
        def read_tags() -> list:
            # In real LangChain code, var_child_runnable_config.get()
            # would return cfg here
            return cfg.get("tags", [])
        tags = ctx.run(read_tags)
        print(f"tags inside context: {tags}")
    # After the `with` block, the original context is restored


run_in_context(config_a)  # tags inside context: ['branch-a']
run_in_context(config_b)  # tags inside context: ['branch-b']
```

### Example 3 — inherit config in parallel async node tasks

```python
import asyncio
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph._internal._runnable import create_task_in_config_context


class State(TypedDict):
    results: list


async def parallel_node(state: State, config: RunnableConfig) -> dict:
    """Spawn two async sub-tasks that both inherit the current config."""

    async def sub_task(label: str) -> str:
        # In production each sub-task would call an LLM; here we just echo
        return f"{label} complete"

    task_a = create_task_in_config_context(lambda: sub_task("A"), config)
    task_b = create_task_in_config_context(lambda: sub_task("B"), config)
    results = await asyncio.gather(task_a, task_b)
    return {"results": list(results)}


builder = StateGraph(State)
builder.add_node("parallel", parallel_node)
builder.add_edge(START, "parallel")
builder.add_edge("parallel", END)

graph = builder.compile()
out = asyncio.run(graph.ainvoke({"results": []}))
print(out["results"])  # ['A complete', 'B complete']
```

---

## 4 · `chain_future` + `run_coroutine_threadsafe`

**Module:** `langgraph._internal._future`

These two functions bridge the gap between a running `asyncio` event loop and Python threads. `run_coroutine_threadsafe` submits a coroutine to an already-running event loop from any thread (including the main thread when the loop is on a background thread). `chain_future` links two futures — `asyncio.Future` or `concurrent.futures.Future` — so that when the source completes its result (or exception) is forwarded to the destination, with optional cross-loop safety.

**Key source facts** (from `langgraph/_internal/_future.py`):

- `run_coroutine_threadsafe(coro, loop, *, lazy, name, context)` — if called from the same thread as `loop` (detected via `asyncio._get_running_loop() is loop`), it calls `_ensure_future` directly without the threadsafe dance. Otherwise it allocates a bare `asyncio.Future` on `loop` and uses `loop.call_soon_threadsafe` to schedule `chain_future(task, future)`.
- `_ensure_future(coro_or_future, *, loop, lazy)` — on Python ≥ 3.12 with `lazy=False`, uses `asyncio.eager_task_factory` for zero-scheduling-delay start; on older Python falls back to `loop.create_task`. Wraps non-coroutine awaitables via `_wrap_awaitable`.
- `chain_future(source, destination)` — calls `_chain_future`, then returns `destination`. `_chain_future` attaches a done-callback on `destination` to cancel `source` if `destination` is cancelled, and a done-callback on `source` to copy state to `destination` (handling cross-loop scheduling via `dest_loop.call_soon_threadsafe`).
- `_convert_future_exc` — maps `concurrent.futures` exception classes to their `asyncio` equivalents so that cross-future exception propagation preserves the correct type.

### Example 1 — submit a coroutine to a background event loop from the main thread

```python
import asyncio
import threading
from langgraph._internal._future import run_coroutine_threadsafe


async def async_worker(n: int) -> int:
    await asyncio.sleep(0)  # yield control
    return n * n


# Start a background event loop on a daemon thread
loop = asyncio.new_event_loop()
t = threading.Thread(target=loop.run_forever, daemon=True)
t.start()

# Submit work from the main (non-async) thread; returns asyncio.Future (not
# concurrent.futures.Future), so bridge back via threading.Event
async_fut = run_coroutine_threadsafe(async_worker(7), loop)
_done = threading.Event()
_box: list = []
loop.call_soon_threadsafe(
    async_fut.add_done_callback, lambda f: (_box.append(f.result()), _done.set())
)
completed = _done.wait(timeout=5)
if not completed:
    raise TimeoutError("async_worker did not complete in time")
result = _box[0]
print("result:", result)  # result: 49

loop.call_soon_threadsafe(loop.stop)
t.join()
```

### Example 2 — chain two futures to propagate results across contexts

```python
import asyncio
from langgraph._internal._future import chain_future


async def demo():
    loop = asyncio.get_running_loop()

    source: asyncio.Future[str] = loop.create_future()
    destination: asyncio.Future[str] = loop.create_future()

    # Link: when source resolves, destination gets the same result
    chain_future(source, destination)

    # Resolve the source
    source.set_result("hello from source")

    # Destination is now resolved too
    result = await destination
    print(result)  # hello from source


asyncio.run(demo())
```

### Example 3 — exception propagation through `chain_future`

```python
import asyncio
from langgraph._internal._future import chain_future


async def demo():
    loop = asyncio.get_running_loop()

    source: asyncio.Future[str] = loop.create_future()
    destination: asyncio.Future[str] = loop.create_future()
    chain_future(source, destination)

    # Fail the source
    source.set_exception(ValueError("something went wrong"))

    # Destination receives the same exception
    try:
        await destination
    except ValueError as e:
        print(f"caught: {e}")  # caught: something went wrong

    # Cancellation is also forwarded in both directions
    src2: asyncio.Future[str] = loop.create_future()
    dst2: asyncio.Future[str] = loop.create_future()
    chain_future(src2, dst2)
    dst2.cancel()  # cancelling destination also cancels source
    await asyncio.sleep(0)  # let callbacks fire
    print("src2 cancelled:", src2.cancelled())  # src2 cancelled: True


asyncio.run(demo())
```

---

## 5 · `map_debug_checkpoint` + `tasks_w_writes` + `rm_pregel_keys`

**Module:** `langgraph.pregel.debug`

These three functions assemble the objects that appear in `stream_mode="debug"` checkpoint events and in `StateSnapshot.tasks`. They are called at the end of each Pregel superstep by the Pregel loop.

**Key source facts** (from `langgraph/pregel/debug.py`):

- `map_debug_checkpoint(config, channels, stream_channels, metadata, tasks, pending_writes, parent_config, output_keys)` — yields `CheckpointPayload` dicts. For each task with `subgraphs`, it computes the child `checkpoint_ns` and stores the config (a signal that subgraph checkpoint data exists) or, if the subgraph has already been resolved, the full `StateSnapshot`. It reads the output channels via `read_channels`, builds a `CheckpointTask` per task (with interrupts and errors extracted from `pending_writes`), and yields the payload.
- `tasks_w_writes(tasks, pending_writes, states, output_keys)` — applies the writes from `pending_writes` to each `PregelTask` to produce the `PregelTask` tuple returned in `StateSnapshot.tasks`. It extracts `RETURN` channel writes (the final output), `ERROR` exceptions, `INTERRUPT` values (from `wait_for_interrupt()`), and all other per-channel writes from the pending list.
- `rm_pregel_keys(config)` — strips Pregel-internal `configurable` keys (all those starting with `"__pregel_"`) from a `RunnableConfig` so that the slimmed config can be safely returned to the user (e.g. in `StateSnapshot.config`).

### Example 1 — consume `stream_mode="debug"` checkpoint events

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    value: int


def increment(state: State) -> dict:
    return {"value": state["value"] + 1}


builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "debug-demo"}}
checkpoints = []
# stream_mode="debug" (single string) yields each event as a dict directly —
# keys are "type", "step", "timestamp", "payload". No tuple unpacking.
for event in graph.stream(
    {"value": 0},
    config,
    stream_mode="debug",
):
    if event["type"] == "checkpoint":
        checkpoints.append(event)

print(f"received {len(checkpoints)} checkpoint event(s)")
# Event dict shape: {"type", "step", "timestamp", "payload"}
# payload is a nested dict with "values", "next", "config", "metadata", "tasks"
if checkpoints:
    cp = checkpoints[0]
    print("step:", cp.get("step"))  # top-level field, not inside payload
```

### Example 2 — `tasks_w_writes` — apply pending writes to see task outputs in a snapshot

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel.debug import tasks_w_writes


class State(TypedDict):
    counter: int


def bump(state: State) -> dict:
    return {"counter": state["counter"] + 10}


builder = StateGraph(State)
builder.add_node("bump", bump)
builder.add_edge(START, "bump")
builder.add_edge("bump", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "tasks-demo"}}

graph.invoke({"counter": 0}, config)

# Retrieve a snapshot; snapshot.tasks uses tasks_w_writes internally
snapshot = graph.get_state(config)
print("tasks:", snapshot.tasks)
# PregelTask tuples with id, name, path, error, interrupts, state, result
```

### Example 3 — `rm_pregel_keys` — strip internal config keys before returning to a caller

```python
from langgraph.pregel.debug import rm_pregel_keys
from langchain_core.runnables import RunnableConfig

# Simulate a config dict that Pregel would have decorated with internal keys
raw_config: RunnableConfig = {
    "tags": ["my-run"],
    "configurable": {
        "thread_id": "t1",
        "__pregel_send": lambda x: None,        # internal — will be stripped
        "__pregel_read": lambda x, y: None,     # internal — will be stripped
        "__pregel_checkpointer": object(),      # internal — will be stripped
        "my_custom_key": "keep-me",             # user key — preserved
    },
}

cleaned = rm_pregel_keys(raw_config)
print("remaining configurable keys:", list(cleaned["configurable"].keys()))
# ['thread_id', 'my_custom_key']
# All '__pregel_*' keys are gone
```

---

## 6 · `AsyncSubgraphRunStream`

**Module:** `langgraph.stream.run_stream`

`AsyncSubgraphRunStream` is the async counterpart to `SubgraphRunStream`. It is the handle object you receive when iterating `run.subgraphs` during `async for run in graph.astream_events(...)` with `version="v3"`. It extends `AsyncGraphRunStream` and mixes in `_SubgraphRunStreamMixin` to expose subgraph-specific attributes.

**Key source facts** (from `langgraph/stream/run_stream.py`):

- `path: tuple[str, ...]` — the namespace path of this subgraph (e.g. `("my_node", "<task-uuid>")` for a subgraph called from node `"my_node"`).
- `graph_name: str | None` — the name of the subgraph class or graph object if available.
- `trigger_call_id: str | None` — the `call()` ID that triggered this subgraph, when spawned via the functional API.
- `status: Literal["started", "completed", "failed", "interrupted", "drained"]` — updated by the parent mux as terminal events arrive.
- `_apump_next()` — delegates to `self._parent_apump_fn`, the parent mux's pump function, rather than owning its own event loop. This is the key design: subgraph streams share the parent's event source.
- `abort()` / async context manager — inherited from `AsyncGraphRunStream`; calling `abort()` signals the parent mux to stop forwarding events for this scope.

### Example 1 — iterate subgraph events asynchronously in v3 streaming

```python
import asyncio
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class InnerState(TypedDict):
    value: int


def inner_worker(state: InnerState) -> dict:
    return {"value": state["value"] * 2}


inner = StateGraph(InnerState)
inner.add_node("worker", inner_worker)
inner.add_edge(START, "worker")
inner.add_edge("worker", END)
inner_graph = inner.compile()


class OuterState(TypedDict):
    value: int


def outer_node(state: OuterState) -> dict:
    result = inner_graph.invoke({"value": state["value"]})
    return {"value": result["value"]}


outer = StateGraph(OuterState)
outer.add_node("outer", outer_node)
outer.add_edge(START, "outer")
outer.add_edge("outer", END)
outer_graph = outer.compile()


async def main():
    # v3 streaming exposes AsyncSubgraphRunStream objects
    async with await outer_graph.astream_events(
        {"value": 5}, version="v3"
    ) as run:
        print("run type:", type(run).__name__)  # AsyncGraphRunStream
        # Subgraph discovery happens as events are pumped
        async for event in run:
            pass  # consume events to drive the pump
        # Subgraphs discovered during the run
        for sg in run.subgraphs:
            print(f"subgraph path={sg.path}, status={sg.status}")


asyncio.run(main())
```

### Example 2 — read `path`, `graph_name`, and `status` from a discovered subgraph

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class Inner(TypedDict):
    n: int


def double(state: Inner) -> dict:
    return {"n": state["n"] * 2}


inner = StateGraph(Inner)
inner.add_node("double", double)
inner.add_edge(START, "double")
inner.add_edge("double", END)
inner_compiled = inner.compile()


class Outer(TypedDict):
    n: int


def call_inner(state: Outer) -> dict:
    return inner_compiled.invoke({"n": state["n"]})


outer = StateGraph(Outer)
outer.add_node("call_inner", call_inner)
outer.add_edge(START, "call_inner")
outer.add_edge("call_inner", END)
outer_compiled = outer.compile()


async def main():
    async with await outer_compiled.astream_events({"n": 3}, version="v3") as run:
        async for _ in run:
            pass

    for sg in run.subgraphs:
        print("path      :", sg.path)
        print("graph_name:", sg.graph_name)
        print("status    :", sg.status)  # "completed", "failed", "interrupted", or "drained"


asyncio.run(main())
```

### Example 3 — abort a subgraph stream early

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    v: int


def node_a(state: S) -> dict:
    return {"v": state["v"] + 1}


def node_b(state: S) -> dict:
    return {"v": state["v"] + 10}


g = StateGraph(S)
g.add_node("a", node_a)
g.add_node("b", node_b)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)
graph = g.compile()


async def main():
    events_seen = 0
    async with await graph.astream_events({"v": 0}, version="v3") as run:
        async for event in run:
            events_seen += 1
            if events_seen >= 2:
                await run.abort()  # stop the stream early
                break

    print(f"stopped after {events_seen} event(s)")


asyncio.run(main())
```

---

## 7 · `_triggers` + `_scratchpad`

**Module:** `langgraph.pregel._algo`

`_triggers` and `_scratchpad` are two of the three internal helpers that `prepare_single_task` calls for every PULL (ordinary graph node) task. Together they decide **whether** a node should run this step, and **what resume / interrupt state** it gets.

**Key source facts** (from `langgraph/pregel/_algo.py`):

**`_triggers(channels, versions, seen, null_version, proc)`**
- Returns `True` if at least one of `proc.triggers` is available **and** has a version newer than what the node last saw. The `seen` arg is `checkpoint["versions_seen"].get(name)` — `None` on the very first invocation (cold start), which causes the simpler check `channels[chan].is_available()`.
- The version comparison `versions.get(chan) > seen.get(chan, null_version)` is what prevents re-running a node when its trigger channel was written in an earlier step but not in the current one.

**`_scratchpad(parent_scratchpad, pending_writes, task_id, namespace_hash, resume_map, step, stop)`**
- Builds the `PregelScratchpad` for a task. The scratchpad holds the resume queue, interrupt counter, call counter, and subgraph counter — all as `LazyAtomicCounter` objects to be thread-safe.
- It scans `pending_writes` for three kinds of resume entries: a `NULL_TASK_ID + RESUME` global resume (used by `Command(resume=…)`), a `task_id + RESUME` per-task resume (used by `interrupt()` → `Command(resume=…)` roundtrip), and a `namespace_hash`-keyed entry from `resume_map` (used for multi-interrupt flows).
- `get_null_resume(consume=False)` — callable that optionally removes the global resume write so it isn't re-processed in subsequent steps.

### Example 1 — understand trigger detection with channel versions

```python
from langgraph.channels.last_value import LastValue
from langgraph.pregel._read import PregelNode

# Simulate two channels: "a" was updated this step, "b" was not
chan_a = LastValue(int, "a")
chan_a.update([42])    # written this step

chan_b = LastValue(int, "b")
chan_b.update([0])     # written in a prior step, not this one

channels = {"a": chan_a, "b": chan_b}

# Simulate channel versions: current=2 for "a", 1 for "b"
current_versions = {"a": 2, "b": 1}

# What the node saw at its last checkpoint
last_seen = {"a": 1, "b": 1}   # "a" is newer, "b" is same

null_version = 0  # sentinel used for channels never written before

from langgraph.pregel._algo import _triggers

# Node that triggers on "a"
node_a = PregelNode(channels=["a"], triggers=["a"])
# Node that triggers on "b"
node_b = PregelNode(channels=["b"], triggers=["b"])

print("node_a triggered:", _triggers(channels, current_versions, last_seen, null_version, node_a))
# True — "a" version 2 > last_seen 1
print("node_b triggered:", _triggers(channels, current_versions, last_seen, null_version, node_b))
# False — "b" version 1 == last_seen 1
```

### Example 2 — cold-start: `seen=None` triggers on any non-empty channel

```python
from langgraph.channels.last_value import LastValue
from langgraph.pregel._read import PregelNode
from langgraph.pregel._algo import _triggers

empty_chan = LastValue(int, "x")  # never written

filled_chan = LastValue(int, "y")
filled_chan.update([99])

channels = {"x": empty_chan, "y": filled_chan}
versions = {"x": 0, "y": 1}
null_version = 0

node_x = PregelNode(channels=["x"], triggers=["x"])
node_y = PregelNode(channels=["y"], triggers=["y"])

# seen=None means first superstep: only availability matters, not version delta
print("x triggered (cold):", _triggers(channels, versions, None, null_version, node_x))
# False — channel is empty
print("y triggered (cold):", _triggers(channels, versions, None, null_version, node_y))
# True — channel has a value
```

### Example 3 — inspect a `PregelScratchpad` created by `_scratchpad`

```python
from langgraph.pregel._algo import _scratchpad

# A pending_writes list as the Pregel loop would produce it
# (NULL_TASK_ID, RESUME, value) carries a global Command(resume=…)
from langgraph._internal._constants import NULL_TASK_ID, RESUME

pending_writes = [
    (NULL_TASK_ID, RESUME, "user_answer"),  # global resume
]

scratchpad = _scratchpad(
    parent_scratchpad=None,
    pending_writes=pending_writes,
    task_id="task-123",
    namespace_hash="aabbcc",
    resume_map=None,
    step=1,
    stop=25,
)

print("step              :", scratchpad.step)           # 1
print("stop              :", scratchpad.stop)           # 25
print("global resume     :", scratchpad.get_null_resume())  # user_answer
print("task resume queue :", scratchpad.resume)         # []
print("interrupt counter :", scratchpad.interrupt_counter.value)  # 0
```

---

## 8 · `_proc_input`

**Module:** `langgraph.pregel._algo`

`_proc_input` collects the input values for a PULL node from the live channel dict and optional managed values. It is the function that decides what the node callable actually receives as its `state` argument.

**Key source facts** (from `langgraph/pregel/_algo.py`):

- `channels: str` path — if `proc.channels` is a single string, the node receives the raw channel value directly (useful for nodes that accept a flat value rather than a dict).
- `channels: list[str]` path — iterates `proc.channels`, reads each via `channels[chan].get()`, and builds a `dict[str, Any]`. Channels in `managed` (like `IsLastStep`) are read via `managed[chan].get(scratchpad)` instead.
- `mapper` application — if `proc.mapper` is set and `for_execution=True`, the collected dict is passed through `mapper` before being returned. Setting `for_execution=False` skips the mapper so that dry-run task inspection (for `StateGraph.get_graph()`) sees the raw input shape.
- `input_cache: dict[INPUT_CACHE_KEY_TYPE, Any] | None` — the cache key is `(proc.mapper, tuple(proc.channels))`. On a cache hit, a shallow `copy()` of the cached input is returned to prevent one node's mutations from leaking to another that shares the same input shape.

### Example 1 — trace which channels a node reads from and what input it receives

```python
from typing import Annotated
import operator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    x: int
    y: int
    total: Annotated[int, operator.add]


def add_xy(state: State) -> dict:
    print(f"node received: x={state['x']}, y={state['y']}, total={state['total']}")
    return {"total": state["x"] + state["y"]}


builder = StateGraph(State)
builder.add_node("add", add_xy)
builder.add_edge(START, "add")
builder.add_edge("add", END)

graph = builder.compile()
result = graph.invoke({"x": 3, "y": 4, "total": 0})
print("final total:", result["total"])  # 7
# _proc_input built {'x': 3, 'y': 4, 'total': 0} from the channel dict
# and handed it to add_xy as its state argument
```

### Example 2 — standard multi-channel state: `_proc_input` builds the state dict

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str
    upper: str


def uppercase_node(state: State) -> dict:
    """_proc_input assembled this dict from channel values and passed it here."""
    return {"upper": state["text"].upper()}


builder = StateGraph(State)
builder.add_node("upper", uppercase_node)
builder.add_edge(START, "upper")
builder.add_edge("upper", END)

graph = builder.compile()
result = graph.invoke({"text": "hello world", "upper": ""})
print(result["upper"])  # HELLO WORLD
```

### Example 3 — input caching: two nodes share the same channel subscription

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

call_count = {"n": 0}


class State(TypedDict):
    data: list
    result_a: str
    result_b: str


def node_a(state: State) -> dict:
    call_count["n"] += 1
    return {"result_a": f"a:{len(state['data'])}"}


def node_b(state: State) -> dict:
    call_count["n"] += 1
    return {"result_b": f"b:{len(state['data'])}"}


builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
# Both nodes run in the same superstep — they share the same data input
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", END)
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"data": [1, 2, 3], "result_a": "", "result_b": ""})
print("result_a:", result["result_a"])  # a:3
print("result_b:", result["result_b"])  # b:3
# _proc_input's input_cache ensures the channel dict is read once
# and shallow-copied for the second node rather than re-reading channels
```

---

## 9 · `prepare_single_task` + `_TaskIDFn`

**Module:** `langgraph.pregel._algo`

`prepare_single_task` is the top-level dispatcher that turns a `task_path` tuple into either a `PregelTask` (for dry-run inspection) or a `PregelExecutableTask` (for live execution). It routes by the first element of the path — `PUSH` paths go to `prepare_push_task_functional` or `prepare_push_task_send`; `PULL` paths are handled inline.

`_TaskIDFn` is a `Protocol` describing the signature of the task-ID hashing function (`_uuid5_str` for v1 checkpoints, `_xxhash_str` for v2+).

**Key source facts** (from `langgraph/pregel/_algo.py`):

- `task_path` format: `(PULL, node_name)` for regular nodes; `(PUSH, namespace_tuple, index, node_name, Send(...))` for `Send()`-dispatched tasks; `(PUSH, namespace_tuple, index, node_name, Call(...))` for `@task`-dispatched tasks.
- Checkpoint version determines the ID hash function: `checkpoint["v"] > 1` uses `_xxhash_str` (fast, 128-bit XXH3), otherwise `_uuid5_str` (SHA-1-based, for backwards compatibility).
- The assembled `PregelExecutableTask` carries: node callable, input value, a `deque` for writes, the full patched config (with `CONFIG_KEY_READ`, `CONFIG_KEY_SEND`, `CONFIG_KEY_RUNTIME` etc.), retry policies, cache key, task ID, task path, flat writers, subgraph list, and timeout policy.
- `task_id_checksum` — optional assertion parameter used in tests to verify that the computed ID matches a pre-computed value.

### Example 1 — observe `task_path` in a custom error handler

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    value: int
    error: str


def risky_node(state: State) -> dict:
    if state["value"] < 0:
        raise ValueError("value must be non-negative")
    return {"value": state["value"] + 1}


def error_handler(state: State) -> dict:
    return {"error": f"caught error for value={state['value']}"}


builder = StateGraph(State)
builder.add_node("risky", risky_node, error_handler=error_handler)
builder.add_edge(START, "risky")
builder.add_edge("risky", END)
graph = builder.compile()

# When risky_node raises, prepare_single_task builds an error-handler task
result = graph.invoke({"value": -1, "error": ""})
print(result["error"])  # caught error for value=-1
```

### Example 2 — `Send()` creates PUSH task paths, inspect via `stream_mode="debug"`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class State(TypedDict):
    items: list
    # Annotated reducer required: multiple Send tasks write processed in same superstep
    processed: Annotated[list, operator.add]


def fan_out(state: State):
    return [Send("process_item", {"item": x}) for x in state["items"]]


def process_item(state: dict) -> dict:
    return {"processed": [state["item"] * 2]}


builder = StateGraph(State)
builder.add_node("process_item", process_item)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("process_item", END)

graph = builder.compile()

# Each Send becomes a PUSH task path; prepare_single_task routes them
# to prepare_push_task_send. stream_mode="debug" yields event dicts directly.
task_events = []
for event in graph.stream(
    {"items": [1, 2, 3], "processed": []}, stream_mode="debug"
):
    if event["type"] == "task":
        task_events.append(event)

print(f"PUSH task events: {len(task_events)}")
for t in task_events:
    print("  name:", t.get("payload", {}).get("name"))
```

### Example 3 — checkpoint v1 vs v2 task IDs (uuid5 vs xxhash)

```python
from langgraph.pregel._algo import _uuid5_str, _xxhash_str

# Both functions produce stable IDs from the same inputs
ns = b"langgraph_namespace"

id_v1 = _uuid5_str(ns, "my_node", "step_1", "pull")
id_v2 = _xxhash_str(ns, "my_node", "step_1", "pull")

print("uuid5  :", id_v1)   # xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
print("xxhash :", id_v2)   # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (hex)

# IDs are deterministic — same inputs always give same ID
assert _uuid5_str(ns, "my_node", "step_1", "pull") == id_v1
assert _xxhash_str(ns, "my_node", "step_1", "pull") == id_v2

# They are different hash algorithms, so the values differ
assert id_v1 != id_v2
print("IDs differ between v1 and v2 checkpoint format — expected")
```

---

## 10 · `_get_channels` + `_get_channel` + `_is_field_channel` + `_is_field_binop`

**Module:** `langgraph.graph.state`

These four functions are the backbone of `StateGraph`'s annotation-to-channel inference. When you call `StateGraph(MyTypedDict)`, they translate every annotated field into a `BaseChannel` instance (or a `ManagedValueSpec`).

**Key source facts** (from `langgraph/graph/state.py`):

- **`_get_channels(schema)`** — the entry point. Calls `get_type_hints(schema, include_extras=True)` to see `Annotated[…]` metadata, then calls `_get_channel(name, typ)` for each field. Returns a 3-tuple: `(channel_dict, managed_dict, type_hints)`.
- **`_get_channel(name, annotation, *, allow_managed=True)`** — strips `Required`/`NotRequired` wrappers, delegates to `_is_field_managed_value`, `_is_field_channel`, or `_is_field_binop` in order. Falls back to `LastValue(annotation)` for plain unannotated types.
- **`_is_field_channel(typ)`** — inspects `typ.__metadata__` for items that are `BaseChannel` instances or `BaseChannel` subclasses. If a `DeltaChannel` is found and the outer origin is a parameterized type, it re-instantiates the `DeltaChannel` with the correct item type.
- **`_is_field_binop(typ)`** — inspects `typ.__metadata__` for a trailing callable with exactly two positional parameters; wraps it in `BinaryOperatorAggregate(typ, reducer)`. Raises `ValueError` for reducers with the wrong arity.

### Example 1 — trace which channel class each field annotation produces

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.state import _get_channels
from langgraph.channels.last_value import LastValue
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.any_value import AnyValue


class MyState(TypedDict):
    # Plain field → LastValue
    name: str
    # Reducer annotation → BinaryOperatorAggregate
    total: Annotated[int, operator.add]
    # Explicit channel annotation → AnyValue
    scratch: Annotated[str, AnyValue(str)]


channels, managed, hints = _get_channels(MyState)

for field, chan in channels.items():
    print(f"{field:10} → {type(chan).__name__}")
# name       → LastValue
# total      → BinaryOperatorAggregate
# scratch    → AnyValue
```

### Example 2 — `_is_field_binop` validates reducer arity

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.state import _is_field_binop

# Valid two-argument reducer
valid_typ = Annotated[int, operator.add]
chan = _is_field_binop(valid_typ)
print("valid reducer →", type(chan).__name__)  # BinaryOperatorAggregate

# Zero-argument callable → ValueError
try:
    bad_typ = Annotated[int, lambda: 0]
    _is_field_binop(bad_typ)
except ValueError as e:
    print("caught:", e)

# Single-argument callable → ValueError
try:
    bad2_typ = Annotated[int, lambda x: x]
    _is_field_binop(bad2_typ)
except ValueError as e:
    print("caught single-arg:", e)

# Three-argument callable → ValueError (not exactly 2)
try:
    bad3_typ = Annotated[int, lambda a, b, c: a]
    _is_field_binop(bad3_typ)
except ValueError as e:
    print("caught 3-arg:", e)
```

### Example 3 — `_is_field_channel` recognises both class and instance annotations

```python
from typing import Annotated
from langgraph.graph.state import _is_field_channel
from langgraph.channels.any_value import AnyValue
from langgraph.channels.ephemeral_value import EphemeralValue


# Instance annotation — the channel object itself becomes the channel
any_value_inst = _is_field_channel(Annotated[str, AnyValue(str)])
print("instance →", type(any_value_inst).__name__)  # AnyValue

# Class annotation — the class is instantiated with the origin type
ephem_class = _is_field_channel(Annotated[int, EphemeralValue])
print("class    →", type(ephem_class).__name__)    # EphemeralValue

# No metadata → returns None (fallback to LastValue)
plain = _is_field_channel(int)
print("plain    →", plain)                         # None
```

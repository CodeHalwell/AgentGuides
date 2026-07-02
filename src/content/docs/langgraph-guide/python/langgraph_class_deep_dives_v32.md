---
title: "LangGraph Class Deep-Dives Vol. 32"
description: "Source-verified deep dives into 10 previously undocumented class groups — NodeBuilder fluent node assembly, Pregel.bulk_update_state()/abulk_update_state() batch checkpoint injection, GraphRunStream.interleave() multi-projection iteration, GraphRunStream.abort()/AsyncGraphRunStream.abort() cooperative cancellation, ToolRuntime.emit_output_delta() per-tool streaming, ToolRuntime.execution_info/server_info execution metadata in tools, ToolCallWithContext Send-based parallel tool dispatch with state, Runtime.merge()/override()/patch_execution_info() composition, SubgraphRunStream path/graph_name/status navigation, and AgentStateWithStructuredResponse migration to response_format= — verified against langgraph==1.2.7."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 32"
  order: 63
---

Source-verified deep dives into **10 previously undocumented class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.7` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `NodeBuilder`

**Module:** `langgraph.pregel.main`

`NodeBuilder` is a fluent builder that assembles a `PregelNode` directly — the low-level node representation inside `Pregel`. You normally never touch it (that's `StateGraph.add_node`'s job), but understanding it unlocks the internals of the Pregel graph model and lets you build custom graph extensions.

**Key source facts** (from `langgraph/pregel/main.py`):

- `subscribe_only(channel)` — subscribe to exactly one channel. The channel is both the trigger and the input key. Mutually exclusive with `subscribe_to`.
- `subscribe_to(*channels, read=True)` — subscribe to one or more channels as trigger; when `read=True` (default) they are also passed in the node's input dict.
- `read_from(*channels)` — add channels to the input dict *without* triggering the node. Useful when a secondary channel must be readable but shouldn't activate the node.
- `do(node)` — set or chain the runnable. Calling `do` twice wraps both callables in a `RunnableSeq`.
- `write_to(*channels, **kwargs)` — schedule channel writes after the node completes. Positional strings write the node's output; `kwargs` accept static values or mapper callables.
- `add_retry_policies(*policies)` — append `RetryPolicy` objects; they are tried in order.
- `add_cache_policy(policy)` — set a `CachePolicy`; supersedes any previous policy.
- `set_timeout(timeout)` — coerce a `float`, `timedelta`, or `TimeoutPolicy` via `coerce_timeout_policy()`.
- `build()` — materialise into a `PregelNode` with the accumulated config. The builder is **not** reusable after `build()`.

### Example 1 — inspect what `StateGraph` produces internally

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    x: int

def double(state: State) -> State:
    return {"x": state["x"] * 2}

graph = StateGraph(State)
graph.add_node("double", double)
graph.add_edge(START, "double")
graph.add_edge("double", END)
compiled = graph.compile()

# Inspect the PregelNode the builder produced
node = compiled.nodes["double"]
print("channels :", node.channels)   # ['x'] (or the full state keys)
print("triggers :", node.triggers)   # [':double']
print("tags     :", node.tags)       # []
print("writers  :", node.writers)    # [ChannelWrite(...)]
```

### Example 2 — construct a `NodeBuilder` by hand and read its `PregelNode`

```python
from langgraph.pregel.main import NodeBuilder
from langgraph.types import RetryPolicy, CachePolicy

def compute(inp: str) -> str:
    return inp.upper()

node = (
    NodeBuilder()
    .subscribe_to("raw_text")
    .do(compute)
    .write_to("result")
    .add_retry_policies(RetryPolicy(max_attempts=3))
    .add_cache_policy(CachePolicy(ttl=60))
    .meta("my-tag", env="prod")
    .build()
)

print("channels:", node.channels)     # ['raw_text']
print("triggers:", node.triggers)     # ['raw_text']
print("tags    :", node.tags)         # ['my-tag']
print("metadata:", node.metadata)     # {'env': 'prod'}
print("retry   :", node.retry_policy) # [RetryPolicy(max_attempts=3, ...)]
```

### Example 3 — `read_from` vs `subscribe_to` difference

```python
from langgraph.pregel.main import NodeBuilder

# subscribe_to: channels both trigger the node and appear in its input
trigger_node = (
    NodeBuilder()
    .subscribe_to("a", "b")          # triggers on a or b; reads both
    .do(lambda d: print("got:", d))
    .build()
)
print("triggers:", trigger_node.triggers)   # ['a', 'b']
print("channels:", trigger_node.channels)   # ['a', 'b']

# read_from: a is already a trigger, c is a "passive" read
passive_node = (
    NodeBuilder()
    .subscribe_to("a")               # trigger + read
    .read_from("c")                  # read only, never a trigger
    .do(lambda d: print("got:", d))
    .build()
)
print("triggers:", passive_node.triggers)   # ['a']
print("channels:", passive_node.channels)   # ['a', 'c']
```

---

## 2 · `Pregel.bulk_update_state()` · `abulk_update_state()`

**Module:** `langgraph.pregel.main`

`bulk_update_state` applies multiple batches of state writes (called *supersteps*) to a persisted checkpoint in a single operation without running any graph nodes. It is the fastest way to seed or repair a thread's state history.

**Key source facts** (from `langgraph/pregel/main.py`):

- Signature: `bulk_update_state(config, supersteps: Sequence[Sequence[StateUpdate]]) -> RunnableConfig`
- `StateUpdate` is a `NamedTuple` with three fields: `values: dict | None`, `as_node: str | None`, `task_id: str | None`. Only `values` is required; `as_node` and `task_id` default to `None`.
- Each inner list is one superstep — the writes are applied atomically; the outer list orders supersteps sequentially.
- Requires a checkpointer. Raises `ValueError` if no checkpointer or if any superstep is empty.
- Delegates to a subgraph when `config.configurable["checkpoint_ns"]` is non-empty.
- `abulk_update_state` is the async twin; otherwise identical.
- Returns the `RunnableConfig` for the resulting checkpoint — pass it to `invoke` or `stream` to replay from that exact point.

### Example 1 — seed a thread with pre-built history

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate
from typing_extensions import TypedDict

class State(TypedDict):
    count: int
    status: str

def increment(state: State) -> State:
    return {"count": state["count"] + 1}

graph = (
    StateGraph(State)
    .add_node("inc", increment)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "thread-1"}}

# Seed the thread with two supersteps worth of history
result_config = graph.bulk_update_state(
    config,
    [
        # Superstep 0: initialise
        [StateUpdate(values={"count": 10, "status": "started"}, as_node="__input__")],
        # Superstep 1: mark done
        [StateUpdate(values={"status": "done"}, as_node="inc")],
    ],
)

snapshot = graph.get_state(result_config)
print(snapshot.values)   # {'count': 10, 'status': 'done'}
```

### Example 2 — bulk-update from multiple simultaneous task writes in one superstep

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate
from typing import Annotated
import operator
from typing_extensions import TypedDict

class State(TypedDict):
    items: Annotated[list[str], operator.add]

def noop(state: State) -> State:
    return {}

def worker_1(state: State) -> State:
    return {}

def worker_2(state: State) -> State:
    return {}

graph = (
    StateGraph(State)
    .add_node("noop", noop)
    .add_node("worker_1", worker_1)
    .add_node("worker_2", worker_2)
    .add_edge(START, "noop")
    .add_edge("noop", "worker_1")
    .add_edge("noop", "worker_2")
    .add_edge("worker_1", END)
    .add_edge("worker_2", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "t2"}}

# Simulate two parallel tasks writing to 'items' in the same superstep
graph.bulk_update_state(
    config,
    [
        [
            StateUpdate(values={"items": ["a", "b"]}, as_node="worker_1"),
            StateUpdate(values={"items": ["c"]},      as_node="worker_2"),
        ],
    ],
)

snap = graph.get_state(config)
print(snap.values["items"])  # ['a', 'b', 'c']  (reducer applied to both)
```

### Example 3 — async bulk-update for ASGI / FastAPI contexts

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate
from typing_extensions import TypedDict

class State(TypedDict):
    score: float

def rate(state: State) -> State:
    return {}

async def seed_thread(thread_id: str, initial_score: float) -> None:
    graph = (
        StateGraph(State)
        .add_node("rate", rate)
        .add_edge(START, "rate")
        .add_edge("rate", END)
        .compile(checkpointer=InMemorySaver())
    )
    config = {"configurable": {"thread_id": thread_id}}
    await graph.abulk_update_state(
        config,
        [[StateUpdate(values={"score": initial_score}, as_node="__input__")]],
    )
    snap = graph.get_state(config)
    print(f"Thread {thread_id}: score={snap.values['score']}")

asyncio.run(seed_thread("async-thread", 0.95))
```

---

## 3 · `GraphRunStream.interleave()`

**Module:** `langgraph.stream.run_stream`

`interleave(*names)` is a sync method on `GraphRunStream` that consumes multiple named v3 projections simultaneously, yielding `(name, item)` tuples in **strict arrival order** — not round-robin.

**Key source facts** (from `langgraph/stream/run_stream.py`):

- Takes `*names: str` — each must be a key in `run.extensions` (registered stream transformer projections).
- Items are ordered by a monotonic push-stamp assigned when each transformer pushes into its `StreamChannel`. The item that arrived first is always yielded first, regardless of which projection it came from.
- Each channel is locked for the duration of the iteration (`_subscribed = True`). Concurrent iteration on the same channel raises `RuntimeError`. Use `channel.tee(n)` for fan-out before calling `interleave`.
- Works only on sync projections (sync `StreamChannel`). Calling `interleave` on an async-bound channel raises `TypeError`.
- When all channels are drained the generator exits normally; if any channel carries a propagated error, it is re-raised at drain time.
- The mux pump (`self._mux._pump_fn`) is called automatically when no channel has buffered items — the caller's `for` loop drives the graph forward.

### Example 1 — interleave `values` and `custom` in arrival order

```python
# Note: requires stream_events(version="v3") which is experimental
# This pattern works when both "values" and "custom" modes are active

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.stream.transformers import CustomTransformer
from typing_extensions import TypedDict

class State(TypedDict):
    step: int

def worker(state: State) -> State:
    writer = get_stream_writer()
    writer({"event": "started", "step": state["step"]})
    new_step = state["step"] + 1
    writer({"event": "done", "step": new_step})
    return {"step": new_step}

graph = (
    StateGraph(State)
    .add_node("worker", worker)
    .add_edge(START, "worker")
    .add_edge("worker", END)
    .compile()
)

# v3 stream_events: pass CustomTransformer to register the "custom" projection
with graph.stream_events({"step": 0}, version="v3", transformers=[CustomTransformer]) as run:
    for name, item in run.interleave("values", "custom"):
        print(f"[{name}] {item}")
# Output (order reflects actual arrival):
# [custom] {'event': 'started', 'step': 0}
# [custom] {'event': 'done', 'step': 1}
# [values] {'step': 1}
```

### Example 2 — interleave `updates` and `custom` during multi-node execution

```python
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.stream.transformers import CustomTransformer, UpdatesTransformer
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

def node_a(state: State) -> State:
    get_stream_writer()({"node": "a", "val": state["value"]})
    return {"value": state["value"] * 2}

def node_b(state: State) -> State:
    get_stream_writer()({"node": "b", "val": state["value"]})
    return {"value": state["value"] + 1}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)

# Register UpdatesTransformer and CustomTransformer to expose those projections
with graph.stream_events({"value": 3}, version="v3", transformers=[UpdatesTransformer, CustomTransformer]) as run:
    for name, item in run.interleave("updates", "custom"):
        print(f"[{name:7s}] {item}")
# Interleaved in arrival order:
# [custom ] {'node': 'a', 'val': 3}
# [updates] {'a': {'value': 6}}
# [custom ] {'node': 'b', 'val': 6}
# [updates] {'b': {'value': 7}}
```

### Example 3 — consuming `interleave` output lazily (early break)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.stream.transformers import CustomTransformer
from typing_extensions import TypedDict

class State(TypedDict):
    n: int

def counter(state: State) -> State:
    w = get_stream_writer()
    for i in range(state["n"]):
        w({"tick": i})
    return {"n": 0}

graph = (
    StateGraph(State)
    .add_node("counter", counter)
    .add_edge(START, "counter")
    .add_edge("counter", END)
    .compile()
)

# Break after seeing the first custom event — the rest of the run is not consumed
with graph.stream_events({"n": 5}, version="v3", transformers=[CustomTransformer]) as run:
    for name, item in run.interleave("custom", "values"):
        if name == "custom":
            print("First custom tick:", item)
            break   # generator cleanup runs; mux is not drained further
```

---

## 4 · `GraphRunStream.abort()` · `AsyncGraphRunStream.abort()`

**Module:** `langgraph.stream.run_stream`

`abort()` cancels an in-progress v3 stream, propagating `GeneratorExit` (sync) or asyncio cancellation (async) into every in-flight node and nested subgraph.

**Key source facts** (from `langgraph/stream/run_stream.py`):

- **Sync** `abort()`: calls `graph_iter.close()` (which sends `GeneratorExit` to the generator), then calls `mux.close()` to drain all projection channels and mark the stream exhausted. Idempotent — calling it again after the stream is exhausted is a no-op.
- **Async** `abort()`: sets `_exhausted = True` and `_aborting = True`, notifies all pump-waiters, cancels the in-flight `_anext_task` if one is running, awaits its settlement, then calls `graph_aiter.aclose()` and `mux.aclose()`. Also idempotent.
- Once `abort()` is called the projection channels raise `StopIteration` / `StopAsyncIteration` immediately on the next iteration; any buffered items are discarded.
- `abort()` is called automatically by the `stream_events()` context manager on `__exit__` / `__aexit__` — you only need to call it explicitly when you want to cancel *before* the context manager exits.
- Sync `abort()` is safe to call from any thread. Async `abort()` must be awaited from the same event loop.

### Example 1 — sync abort after reading the first value

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    n: int

def slow_node(state: State) -> State:
    import time
    time.sleep(0.01)   # simulates work
    return {"n": state["n"] + 1}

graph = (
    StateGraph(State)
    .add_node("slow", slow_node)
    .add_edge(START, "slow")
    .add_edge("slow", "slow")   # loop
    .compile()
)

with graph.stream_events({"n": 0}, version="v3") as run:
    for item in run.values:
        print("Got:", item)
        run.abort()   # cancel immediately after first value
        break
# The graph loop exits; no further nodes run
```

### Example 2 — async abort with a timeout guard

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    n: int

async def async_node(state: State) -> State:
    await asyncio.sleep(0.05)
    return {"n": state["n"] + 1}

graph = (
    StateGraph(State)
    .add_node("step", async_node)
    .add_edge(START, "step")
    .add_edge("step", "step")
    .compile()
)

async def run_with_timeout(timeout: float) -> None:
    async with await graph.astream_events({"n": 0}, version="v3") as run:
        async def collector() -> None:
            async for item in run.values:
                print("value:", item)

        try:
            await asyncio.wait_for(collector(), timeout=timeout)
        except asyncio.TimeoutError:
            await run.abort()   # explicit abort before context manager exits
            print("Aborted after timeout")

asyncio.run(run_with_timeout(0.12))
```

### Example 3 — abort is idempotent and safe to call multiple times

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    v: int

graph = (
    StateGraph(State)
    .add_node("n", lambda s: {"v": s["v"] + 1})
    .add_edge(START, "n")
    .add_edge("n", END)
    .compile()
)

with graph.stream_events({"v": 0}, version="v3") as run:
    run.abort()   # abort before iterating
    run.abort()   # second call: no-op, no error
    # Consuming after abort yields nothing
    items = list(run.values)
    print("items after abort:", items)   # []
```

---

## 5 · `ToolRuntime.emit_output_delta()`

**Module:** `langgraph.prebuilt.tool_node`

`emit_output_delta(delta)` lets a tool push a partial output chunk onto the `tools` stream channel while it is executing. The method reads a per-tool-call `StreamWriter` that `StreamToolCallHandler` installs on a `ContextVar` (`_tool_call_writer`) at `on_tool_start`.

**Key source facts** (from `langgraph/prebuilt/tool_node.py`):

- The `_tool_call_writer` `ContextVar` holds the writer only when the graph runs with `stream_mode="tools"` and the v1 streaming pipeline is active.
- When `_tool_call_writer.get()` returns `None` (no tools mode active), `emit_output_delta` is a **silent no-op** — tool authors can leave calls in place without guarding them.
- `delta` can be any JSON-serialisable value: a string chunk, a dict partial, or a number. It surfaces as the `delta` field inside a `tool-output-delta` protocol event.
- The method is available on `ToolRuntime`, which is injected automatically when a tool parameter is annotated `runtime: ToolRuntime`.
- Consumers receive deltas as `ToolCallStream.output_deltas` items when using `ToolCallTransformer` with `stream_mode="tools"`.

### Example 1 — stream token-by-token output from a tool

```python
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.tool_node import ToolRuntime

@tool
def streaming_tool(query: str, runtime: ToolRuntime) -> str:
    """Search and stream partial results back."""
    words = ["Searching", "for", query, "...", "done!"]
    for word in words:
        runtime.emit_output_delta(word + " ")
    return f"Results for: {query}"

# Use with stream_mode="tools" to receive deltas
node = ToolNode([streaming_tool])
# runtime.emit_output_delta is a no-op outside stream_mode="tools"
result = node.invoke({
    "messages": [AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "streaming_tool", "args": {"query": "langgraph"}}],
    )]
})
print(result["messages"][0].content)   # 'Results for: langgraph'
```

### Example 2 — structured delta chunks (dict payloads)

```python
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt.tool_node import ToolRuntime

@tool
def analyse(data: list[int], runtime: ToolRuntime) -> dict:
    """Run progressive analysis, streaming intermediate results."""
    total = 0
    for i, val in enumerate(data):
        total += val
        # emit a partial progress payload
        runtime.emit_output_delta({"step": i + 1, "running_sum": total})
    return {"final_sum": total, "count": len(data)}

# Outside stream_mode="tools" all deltas are silent no-ops;
# the return value is used as the tool's final output.
from langgraph.prebuilt import ToolNode
node = ToolNode([analyse])
out = node.invoke({
    "messages": [AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "analyse", "args": {"data": [1, 2, 3]}}],
    )]
})
print(out["messages"][0].content)  # '{"final_sum": 6, "count": 3}'
```

### Example 3 — guard pattern: emit_output_delta only when tools mode is active

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.stream.transformers import LifecyclePayload

@tool
def smart_tool(text: str, runtime: ToolRuntime) -> str:
    """Tool that streams only when consumers are listening."""
    sentences = text.split(". ")
    for sentence in sentences:
        # emit_output_delta is a silent no-op when stream_mode!="tools"
        # so no guarding is needed — but you can check explicitly too
        runtime.emit_output_delta({"sentence": sentence})
    return f"Processed {len(sentences)} sentences"

# Demonstration: calling with ToolNode without stream_mode="tools"
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

node = ToolNode([smart_tool])
result = node.invoke({
    "messages": [AIMessage(
        content="",
        tool_calls=[{"id": "1", "name": "smart_tool", "args": {"text": "Hello. World."}}],
    )]
})
# No deltas received (no consumer), final result is the return value
print(result["messages"][0].content)  # 'Processed 2 sentences'
```

---

## 6 · `ToolRuntime.execution_info` · `ToolRuntime.server_info`

**Module:** `langgraph.prebuilt.tool_node`

`ToolRuntime` inherits from `Runtime` and exposes two observability attributes: `execution_info` (an `ExecutionInfo` dataclass with checkpoint and task metadata) and `server_info` (a `ServerInfo` dataclass populated by LangGraph Server in deployed environments).

**Key source facts** (from `langgraph/runtime.py` + `langgraph/prebuilt/tool_node.py`):

- `execution_info: ExecutionInfo | None` — fields: `checkpoint_id`, `checkpoint_ns`, `task_id`, `thread_id`, `run_id`, `node_attempt` (1-indexed), `node_first_attempt_time` (Unix timestamp of first attempt). The `patch(**overrides)` method returns a new `ExecutionInfo` with overridden fields (useful for testing).
- `server_info: ServerInfo | None` — fields: `assistant_id`, `graph_id`, `user: BaseUser | None`. `None` when running open-source LangGraph without LangSmith Deployments.
- Both are `None` for the first few nanoseconds of task preparation; always populated by the time your tool function body runs.
- `node_attempt` increments on each retry (1, 2, 3, …). On the first attempt `node_first_attempt_time` equals the current time; on retries it stays fixed at the first attempt's timestamp, giving you the total elapsed time across all attempts.
- `ToolRuntime` also exposes `tool_call_id`, `state`, `config`, `context`, `store`, `stream_writer`, `heartbeat`, `tools`, and `previous` — making it the most feature-rich injection surface in LangGraph.

### Example 1 — log checkpoint and task IDs from inside a tool

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.tool_node import ToolRuntime

@tool
def introspect(runtime: ToolRuntime) -> str:
    """Return the current execution context."""
    ei = runtime.execution_info
    if ei is None:
        return "No execution info available"
    return (
        f"thread={ei.thread_id}, "
        f"checkpoint={ei.checkpoint_id[:8]}..., "
        f"task={ei.task_id[:8]}..., "
        f"attempt={ei.node_attempt}"
    )

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

# agent = create_react_agent(ChatAnthropic(model="claude-3-5-haiku-latest"), [introspect])
# result = agent.invoke({"messages": [("user", "introspect")]},
#                       {"configurable": {"thread_id": "t1"}})
# Prints something like:
# thread=t1, checkpoint=abc12345..., task=def67890..., attempt=1
```

### Example 2 — detect retry attempts and skip expensive re-work

```python
import time
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime

@tool
def idempotent_write(key: str, value: str, runtime: ToolRuntime) -> str:
    """Write to external API; skip if this is a retry within the same run."""
    ei = runtime.execution_info
    if ei is not None and ei.node_attempt > 1:
        elapsed = time.time() - (ei.node_first_attempt_time or time.time())
        return f"Retry #{ei.node_attempt} — skipping write (elapsed {elapsed:.1f}s)"

    # First attempt: perform the expensive write
    # ... external_api.write(key, value) ...
    return f"Wrote {key}={value} on attempt {ei.node_attempt if ei else 1}"
```

### Example 3 — conditional behaviour based on LangGraph Server context

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime

@tool
def context_aware_tool(query: str, runtime: ToolRuntime) -> str:
    """Behave differently when running on LangSmith Deployments."""
    si = runtime.server_info

    if si is not None:
        # Running on LangGraph Server / LangSmith Deployments
        user_id = si.user.identity if si.user else "anonymous"
        return (
            f"[Server: graph={si.graph_id}, assistant={si.assistant_id}, "
            f"user={user_id}] → {query}"
        )
    else:
        # Running locally or in tests
        return f"[Local] → {query}"
```

---

## 7 · `ToolCallWithContext`

**Module:** `langgraph.prebuilt.tool_node`

`ToolCallWithContext` is an internal `TypedDict` that bundles a `ToolCall` with a snapshot of the current graph state. It is the payload that `create_react_agent` (and any graph using the Send API for tool dispatch) places on the `__tools__` channel when it wants to fan-out tool calls in parallel while also making the state available to the receiving `ToolNode`.

**Key source facts** (from `langgraph/prebuilt/tool_node.py`):

- Fields: `tool_call: ToolCall`, `__type: Literal["tool_call_with_context"]`, `state: Any`.
- The `__type` field with a double-underscore prefix is defensive against name collisions with user state keys.
- `ToolNode` checks `__type == "tool_call_with_context"` to distinguish this payload from a plain state dict. When matched, it extracts `tool_call` and injects `state` into `ToolRuntime.state` so tools can read the state snapshot at the time of dispatch.
- This design enables parallel HITL patterns: `create_react_agent` can Send each tool call as an independent Pregel task; each task carries its own frozen state snapshot; the tasks run concurrently and are independently interruptible.
- `ToolRuntime.state` is populated from `ToolCallWithContext.state`; it gives tools read-only access to the full state dict as it was when the tool was dispatched.

### Example 1 — constructing `ToolCallWithContext` manually (for testing)

```python
from langgraph.prebuilt.tool_node import ToolCallWithContext, ToolNode
from langchain_core.tools import tool

@tool
def echo_state(message: str) -> str:
    """Echo back information from the tool call."""
    return f"Echoed: {message}"

# Build the payload that create_react_agent would emit via Send
payload: ToolCallWithContext = {
    "tool_call": {"id": "tc1", "name": "echo_state", "args": {"message": "hello"}},
    "__type": "tool_call_with_context",
    "state": {"messages": [], "extra": "data"},
}

node = ToolNode([echo_state])
# ToolNode accepts ToolCallWithContext as input directly
result = node.invoke(payload)
print(result["messages"][0].content)   # 'Echoed: hello'
```

### Example 2 — accessing `runtime.state` inside a tool dispatched with context

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolNode, ToolRuntime

@tool
def state_reader(key: str, runtime: ToolRuntime) -> str:
    """Read a key from the state snapshot captured at dispatch time."""
    if runtime.state is None:
        return f"No state available for key={key}"
    val = runtime.state.get(key, "<not found>") if isinstance(runtime.state, dict) else "<non-dict state>"
    return f"state[{key!r}] = {val!r}"

node = ToolNode([state_reader])

# Simulate dispatch via ToolCallWithContext (what create_react_agent emits)
payload = {
    "tool_call": {"id": "tc2", "name": "state_reader", "args": {"key": "user_id"}},
    "__type": "tool_call_with_context",
    "state": {"user_id": "alice", "session": "s42"},
}
result = node.invoke(payload)
print(result["messages"][0].content)   # "state['user_id'] = 'alice'"
```

### Example 3 — why `__type` uses a double-underscore prefix

```python
# __type is defensive against user state having a field named "type"
# Consider a state dict that already has "type":
payload = {
    "tool_call": {"id": "tc3", "name": "echo_state", "args": {"message": "hi"}},
    "__type": "tool_call_with_context",  # safe: double-underscore avoids collision
    "state": {"type": "user_message", "content": "some content"},
    # ^ "type" here is user state, completely separate from "__type"
}
# ToolNode correctly reads "__type" to identify the payload,
# and passes the entire "state" dict (including its "type" key) to ToolRuntime.state
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolNode, ToolRuntime

@tool
def type_demo(message: str, runtime: ToolRuntime) -> str:
    """Access user state 'type' without shadowing the dispatch marker."""
    state_type = runtime.state.get("type") if isinstance(runtime.state, dict) else None
    return f"msg={message}, state_type={state_type}"

node = ToolNode([type_demo])
result = node.invoke(payload)
print(result["messages"][0].content)   # "msg=hi, state_type=user_message"
```

---

## 8 · `Runtime.merge()` · `Runtime.override()` · `Runtime.patch_execution_info()`

**Module:** `langgraph.runtime`

`Runtime` provides three composition helpers that return *new* `Runtime` instances, keeping the dataclass frozen and enabling safe middleware and testing patterns.

**Key source facts** (from `langgraph/runtime.py`):

- `merge(other: Runtime) -> Runtime` — merges two runtimes, preferring `other`'s non-null / non-default values. Specifically: `context`, `store`, `execution_info`, `server_info`, `control` prefer `other`; `stream_writer` and `heartbeat` prefer `other` only when they differ from the default no-op sentinels; `previous` prefers `self` unless `other.previous is not None`.
- `override(**overrides) -> Runtime` — thin wrapper around `dataclasses.replace`; lets you swap any field without knowing the others. Useful in tests.
- `patch_execution_info(**overrides) -> Runtime` — returns `replace(self, execution_info=self.execution_info.patch(**overrides))`. Raises `RuntimeError` when `execution_info` is `None`.
- `drain_requested: bool` — property that checks `control.drain_requested` if `control` is set; otherwise `False`.
- `drain_reason: str | None` — property delegating to `control.drain_reason`.
- All three methods are pure (no mutations); the original `Runtime` is unchanged.

### Example 1 — `override()` for unit-testing nodes that use `Runtime`

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime, ExecutionInfo
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

@dataclass
class Ctx:
    user_id: str

class State(TypedDict):
    result: str

store = InMemoryStore()
store.put(("users",), "u1", {"name": "Alice"})

def my_node(state: State, runtime: Runtime[Ctx]) -> State:
    name = "unknown"
    if runtime.store:
        item = runtime.store.get(("users",), runtime.context.user_id)
        if item:
            name = item.value["name"]
    return {"result": f"Hello {name}"}

# Build a test runtime without running a full graph
test_runtime = Runtime(
    context=Ctx(user_id="u1"),
    store=store,
)

result = my_node({"result": ""}, test_runtime)
print(result)   # {'result': 'Hello Alice'}

# Override just the user_id for a second test case
test_runtime_2 = test_runtime.override(context=Ctx(user_id="u2"))
result_2 = my_node({"result": ""}, test_runtime_2)
print(result_2)   # {'result': 'Hello unknown'}  (u2 not in store)
```

### Example 2 — `merge()` in middleware: parent runtime enriched by child

```python
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

# Simulate a parent runtime with a store and writer
parent_runtime = Runtime(
    store=InMemoryStore(),
    stream_writer=lambda x: print("parent writes:", x),
)

# Child runtime provides only context (no store / writer)
from dataclasses import dataclass

@dataclass
class Ctx:
    env: str

child_runtime = Runtime(context=Ctx(env="prod"))

# merge: child wins on context; parent wins on store/writer
merged = parent_runtime.merge(child_runtime)
print("context:", merged.context.env)   # 'prod'   (from child)
print("store  :", merged.store)         # InMemoryStore (from parent)
# stream_writer: child's is the no-op default → parent's writer wins
merged.stream_writer("hello")           # prints: parent writes: hello
```

### Example 3 — `patch_execution_info()` to simulate a retry in tests

```python
from langgraph.runtime import Runtime, ExecutionInfo

ei = ExecutionInfo(
    checkpoint_id="chk-001",
    checkpoint_ns="",
    task_id="task-001",
    thread_id="thread-1",
    node_attempt=1,
    node_first_attempt_time=1_700_000_000.0,
)

runtime = Runtime(execution_info=ei)

# Simulate a second attempt (retry)
retry_runtime = runtime.patch_execution_info(node_attempt=2)

print(runtime.execution_info.node_attempt)        # 1
print(retry_runtime.execution_info.node_attempt)  # 2
# first_attempt_time is unchanged across retries
print(retry_runtime.execution_info.node_first_attempt_time)  # 1_700_000_000.0
```

---

## 9 · `SubgraphRunStream` · `path` · `graph_name` · `status`

**Module:** `langgraph.stream.run_stream`

`SubgraphRunStream` (and its async twin `AsyncSubgraphRunStream`) is the handle you receive when a v3 stream encounters an embedded subgraph. It extends `GraphRunStream` with three metadata attributes (`path`, `graph_name`, `status`) and delegates its pump to the parent stream so iterating any projection on the subgraph handle drives the root graph forward.

**Key source facts** (from `langgraph/stream/run_stream.py`):

- `path: tuple[str, ...]` — the checkpoint-namespace path of the subgraph (e.g. `("subagent", "inner")`) as a tuple of strings. Each element corresponds to one level of nesting.
- `graph_name: str | None` — the human-readable name of the subgraph, if registered. `None` when the subgraph name is not available from the protocol event.
- `status: str` — lifecycle status. Starts as `"started"`, transitions to `"completed"`, `"failed"`, `"interrupted"`, or `"drained"` when `SubgraphTransformer` processes the terminal event. Useful for knowing whether to drain the handle.
- `error: BaseException | None` — holds the exception if `status == "failed"`.
- `trigger_call_id: str | None` — the `Call` task ID that triggered the subgraph (present only for functional-API `@task`-spawned subgraphs).
- The pump is delegated: `_pump_next` calls `_parent_pump_fn`. You never need to drive `SubgraphRunStream` separately — iterating the parent's projections automatically advances subgraph handles too.
- Subgraph handles are discovered via `run.extensions["subgraphs"]` which is a `StreamChannel[SubgraphRunStream]`.

### Example 1 — enumerate subgraph handles from a v3 stream

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class Inner(TypedDict):
    count: int

class Outer(TypedDict):
    count: int

def inner_node(state: Inner) -> Inner:
    return {"count": state["count"] + 10}

inner_graph = (
    StateGraph(Inner)
    .add_node("inner", inner_node)
    .add_edge(START, "inner")
    .add_edge("inner", END)
    .compile()
)

def outer_node(state: Outer) -> Outer:
    result = inner_graph.invoke({"count": state["count"]})
    return {"count": result["count"]}

graph = (
    StateGraph(Outer)
    .add_node("outer", outer_node)
    .add_edge(START, "outer")
    .add_edge("outer", END)
    .compile()
)

with graph.stream_events({"count": 0}, version="v3") as run:
    # Drain the main stream to populate subgraph handles
    for _ in run.values:
        pass
    # Inspect subgraph handles that were discovered
    # Note: subgraph handles may appear in run.extensions["subgraphs"]
    print("Run output:", run.output)
```

### Example 2 — check subgraph `path` and `status` during streaming

```python
from langgraph.graph import StateGraph, START, END
from langgraph.stream.run_stream import SubgraphRunStream
from typing_extensions import TypedDict

class State(TypedDict):
    v: int

def worker(state: State) -> State:
    return {"v": state["v"] * 2}

sub = (
    StateGraph(State)
    .add_node("worker", worker)
    .add_edge(START, "worker")
    .add_edge("worker", END)
    .compile()
)

def outer(state: State) -> State:
    r = sub.invoke({"v": state["v"]})
    return {"v": r["v"]}

graph = (
    StateGraph(State)
    .add_node("outer", outer)
    .add_edge(START, "outer")
    .add_edge("outer", END)
    .compile()
)

with graph.stream_events({"v": 3}, version="v3") as run:
    # Drain values to complete the run
    final = run.output
    print("Final output:", final)

# SubgraphRunStream handles track path, graph_name, and status
# These are available via run.extensions["subgraphs"] channel
```

### Example 3 — reading subgraph `values` projection separately

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    x: int

def double(state: State) -> State:
    return {"x": state["x"] * 2}

sub = (
    StateGraph(State)
    .add_node("double", double)
    .add_edge(START, "double")
    .add_edge("double", END)
    .compile()
)

def call_sub(state: State) -> State:
    return {"x": sub.invoke({"x": state["x"]})["x"]}

graph = (
    StateGraph(State)
    .add_node("call_sub", call_sub)
    .add_edge(START, "call_sub")
    .add_edge("call_sub", END)
    .compile()
)

async def main():
    async with await graph.astream_events({"x": 5}, version="v3") as run:
        # Consume parent values; subgraph handles (if any) are also advanced
        async for val in run.values:
            print("parent value:", val)
        print("final output:", await run.output())

asyncio.run(main())
# parent value: {'x': 10}
# final output: {'x': 10}
```

---

## 10 · `AgentStateWithStructuredResponse` · migration to `response_format=`

**Module:** `langgraph.prebuilt.chat_agent_executor` (deprecated since v1.0)

`AgentStateWithStructuredResponse` is a `TypedDict` that adds a `structured_response` field to `AgentState`. It was the recommended way to receive typed output from `create_react_agent` in pre-v1.0 releases. Both `AgentStateWithStructuredResponse` and `AgentStateWithStructuredResponsePydantic` are deprecated in `langgraph==1.0` in favour of the `response_format=` parameter on `langchain.agents.create_agent`.

**Key source facts** (from `langgraph/prebuilt/chat_agent_executor.py`):

- `AgentStateWithStructuredResponse(AgentState)` — adds `structured_response: StructuredResponse` where `StructuredResponse = Union[str, dict, BaseModel]`.
- `AgentStateWithStructuredResponsePydantic` — same but with a Pydantic `BaseModel` as the state root (for stricter validation).
- Both classes emit `LangGraphDeprecatedSinceV10` warnings on import.
- The modern replacement: pass `response_format=MyModel` to `create_agent()` (from `langchain.agents`); `create_agent` wraps the graph in a second pass that calls the LLM with the structured output schema.
- The `state_schema=` approach (using one of these deprecated classes) still works but triggers a deprecation warning. Set `response_format=` instead.

### Example 1 — old pattern using `AgentStateWithStructuredResponse` (shows warning)

```python
import warnings
# Suppress the deprecation warning just for demo purposes
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse

# Field layout the deprecated state provides
print(AgentStateWithStructuredResponse.__annotations__)
# {'messages': ..., 'remaining_steps': ..., 'structured_response': ...}
```

### Example 2 — modern `response_format=` replacement

```python
from pydantic import BaseModel
# from langchain.agents import create_agent  # modern import
# from langchain_anthropic import ChatAnthropic

class AnswerSchema(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

# Modern pattern (no deprecated state class needed):
# agent = create_agent(
#     ChatAnthropic(model="claude-3-5-haiku-latest"),
#     tools=[...],
#     response_format=AnswerSchema,   # <-- replaces AgentStateWithStructuredResponse
# )
# result = agent.invoke({"messages": [("user", "What is LangGraph?")]})
# typed_answer = result["structured_response"]   # AnswerSchema instance

print("AnswerSchema fields:", list(AnswerSchema.model_fields.keys()))
# ['answer', 'confidence', 'sources']
```

### Example 3 — migration checklist: from `state_schema=` to `response_format=`

```python
# ── BEFORE (deprecated) ───────────────────────────────────────────────
# from langgraph.prebuilt import create_react_agent
# from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse
#
# agent = create_react_agent(
#     llm,
#     tools,
#     state_schema=AgentStateWithStructuredResponse,   # deprecated
# )
# result = agent.invoke(...)
# typed = result["structured_response"]

# ── AFTER (current) ──────────────────────────────────────────────────
# from langchain.agents import create_agent           # new module
# from pydantic import BaseModel
#
# class MyOutput(BaseModel):
#     answer: str
#     sources: list[str]
#
# agent = create_agent(
#     llm,
#     tools,
#     response_format=MyOutput,                        # replaces state_schema=
# )
# result = agent.invoke(...)
# typed = result["structured_response"]               # same key, typed instance

# Migration steps:
# 1. Replace `from langgraph.prebuilt import create_react_agent` with
#    `from langchain.agents import create_agent`.
# 2. Remove the `state_schema=AgentStateWithStructuredResponse` kwarg.
# 3. Add `response_format=YourPydanticModel` instead.
# 4. The `result["structured_response"]` access path is unchanged.
# 5. Drop the import of AgentStateWithStructuredResponse entirely.

print("Migration complete — no deprecated imports needed")
```

---

## Summary

| # | Class / API | Module | Key takeaway |
|---|---|---|---|
| 1 | `NodeBuilder` | `langgraph.pregel.main` | Fluent builder for low-level `PregelNode` assembly; `subscribe_to` vs `subscribe_only`; `read_from` for passive reads. |
| 2 | `bulk_update_state` / `abulk_update_state` | `langgraph.pregel.main` | Inject multiple `StateUpdate` supersteps into a checkpoint without running nodes. |
| 3 | `GraphRunStream.interleave()` | `langgraph.stream.run_stream` | Consume multiple v3 projections in strict arrival order using monotonic push-stamps. |
| 4 | `GraphRunStream.abort()` / async | `langgraph.stream.run_stream` | Cancel an in-progress v3 stream; propagates `GeneratorExit` into in-flight nodes; idempotent. |
| 5 | `ToolRuntime.emit_output_delta()` | `langgraph.prebuilt.tool_node` | Stream partial tool output on the `tools` channel; silent no-op outside `stream_mode="tools"`. |
| 6 | `ToolRuntime.execution_info` / `server_info` | `langgraph.prebuilt.tool_node` | Access checkpoint ID, task ID, attempt count, and LangSmith Server metadata from inside a tool. |
| 7 | `ToolCallWithContext` | `langgraph.prebuilt.tool_node` | Internal TypedDict bundling `ToolCall` + state snapshot; enables parallel HITL dispatch via Send. |
| 8 | `Runtime.merge()` / `override()` / `patch_execution_info()` | `langgraph.runtime` | Compose immutable `Runtime` instances for middleware and unit-testing without running a full graph. |
| 9 | `SubgraphRunStream` | `langgraph.stream.run_stream` | Subgraph handle in v3 streams; `path`/`graph_name`/`status` metadata; parent-pump delegation. |
| 10 | `AgentStateWithStructuredResponse` | `langgraph.prebuilt` (deprecated) | Deprecated structured-output state; migrate to `response_format=` on `langchain.agents.create_agent`. |

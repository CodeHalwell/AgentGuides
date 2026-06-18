---
title: "Class deep-dives Vol. 19 — Streaming internals, error taxonomy, HITL protocol & execution model"
description: "Source-verified deep dives into CheckpointsTransformer+TasksTransformer, StreamMessagesHandler internals, _TimedAttemptScope+_AttemptContext+_AttemptEvent retry lifecycle, ChannelRead imperative state access, ErrorCode enum+extended exception hierarchy, ValidationNode schema validation, MessageGraph legacy migration, BackgroundExecutor+AsyncBackgroundExecutor, HumanInterrupt full protocol, and PregelNode+StateNodeSpec node representation — verified against langgraph==1.2.5."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 19"
  order: 50
---

# Class deep-dives Vol. 19 — Streaming internals, error taxonomy, HITL protocol & execution model

Verified against **`langgraph==1.2.5`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `CheckpointsTransformer` + `TasksTransformer` | `langgraph.stream.transformers` |
| 2 | `StreamMessagesHandler` + `StreamMessagesHandlerV2` | `langgraph.pregel._messages` |
| 3 | `_TimedAttemptScope` + `_AttemptContext` + `_AttemptEvent` | `langgraph.pregel._retry` |
| 4 | `ChannelRead` — imperative state access | `langgraph.pregel._read` |
| 5 | `ErrorCode` enum + extended exception hierarchy | `langgraph.errors` |
| 6 | `ValidationNode` — schema validation node | `langgraph.prebuilt.tool_validator` |
| 7 | `MessageGraph` — legacy API + migration | `langgraph.graph.message` |
| 8 | `BackgroundExecutor` + `AsyncBackgroundExecutor` + `Submit` | `langgraph.pregel._executor` |
| 9 | `HumanInterrupt` full protocol | `langgraph.prebuilt.interrupt` → `langchain.agents.interrupt` |
| 10 | `PregelNode` + `StateNodeSpec` — node representation | `langgraph.pregel._read` · `langgraph.graph._node` |

---

## 1 · `CheckpointsTransformer` + `TasksTransformer`

**Module:** `langgraph.stream.transformers`

These are two native v3 streaming transformers that surface `stream_mode="checkpoints"` and `stream_mode="tasks"` data as drainable `StreamChannel` projections on the run handle. They complement `DebugTransformer` (Vol. 18) and `LifecycleTransformer` (Vol. 16) — each transformer captures a different slice of the execution trace.

Both are `_native = True`, meaning their attributes (`run.checkpoints`, `run.tasks`) are direct object properties, not entries in `run.custom`.

### Source signatures

```python
class CheckpointsTransformer(StreamTransformer):
    """Capture checkpoint events as a drainable stream.

    Surfaces stream_mode="checkpoints" data on run.checkpoints as
    a StreamChannel[dict[str, Any]]. Each item is in the same format
    as returned by get_state().

    Only events at the run's own scope are captured; checkpoint data from
    deeper subgraphs is available on the respective subgraph handle's
    .checkpoints projection.
    """
    _native = True
    required_stream_modes = ("checkpoints",)

class TasksTransformer(StreamTransformer):
    """Capture raw task events as a drainable stream.

    Surfaces stream_mode="tasks" data on run.tasks as a
    StreamChannel[dict[str, Any]]. Each item is a task payload
    (start or result).

    LifecycleTransformer and SubgraphTransformer also consume tasks
    events for subgraph discovery and lifecycle tracking. This transformer
    captures the raw payloads independently.
    """
    _native = True
    required_stream_modes = ("tasks",)
```

### Key behaviours

| Property | CheckpointsTransformer | TasksTransformer |
|----------|----------------------|-----------------|
| `run` attribute | `run.checkpoints` | `run.tasks` |
| Stream mode required | `"checkpoints"` | `"tasks"` |
| Scope | Own scope only | Own scope only |
| Event format | `get_state()`-compatible dict | Task start/result dicts |
| Requires checkpointer | Yes (no events without one) | No |

### Example 1 — audit trail with CheckpointsTransformer

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.stream.transformers import CheckpointsTransformer
from typing_extensions import TypedDict

class State(TypedDict):
    count: int
    messages: list[str]

def increment(state: State) -> dict:
    return {"count": state["count"] + 1, "messages": state["messages"] + ["incremented"]}

def double(state: State) -> dict:
    return {"count": state["count"] * 2}

builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_node("double", double)
builder.add_edge(START, "increment")
builder.add_edge("increment", "double")
builder.add_edge("double", END)

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "audit-1"}}

# v3 streaming API: astream_events(version="v3") returns an AsyncGraphRunStream.
# Pass transformers= to register projection handlers; stream_mode is derived
# automatically from the transformer mux — do NOT pass it explicitly.
async def run_with_checkpoint_audit():
    run = await graph.astream_events(
        {"count": 1, "messages": []},
        config=config,
        version="v3",
        transformers=[CheckpointsTransformer],
    )
    async with run:
        # CheckpointsTransformer exposes run.checkpoints
        async for checkpoint in run.checkpoints:
            print(f"Checkpoint at step {checkpoint.get('metadata', {}).get('step', '?')}:")
            print(f"  values: {checkpoint.get('values')}")
            print(f"  next: {checkpoint.get('next')}")

asyncio.run(run_with_checkpoint_audit())
```

### Example 2 — task-level execution tracing with TasksTransformer

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.stream.transformers import TasksTransformer
from langgraph.types import Send
from typing_extensions import TypedDict
from typing import Annotated
import operator

class MapState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]

def process_item(state: dict) -> dict:
    item = state["item"]
    return {"results": [f"processed:{item}"]}

def fan_out(state: MapState) -> list[Send]:
    return [Send("worker", {"item": x}) for x in state["items"]]

builder = StateGraph(MapState)
builder.add_node("worker", process_item)
builder.add_conditional_edges(START, fan_out, ["worker"])
builder.add_edge("worker", END)

graph = builder.compile()

async def trace_tasks():
    run = await graph.astream_events(
        {"items": ["a", "b", "c"], "results": []},
        version="v3",
        transformers=[TasksTransformer],
    )
    async with run:
        async for task_event in run.tasks:
            task_type = task_event.get("type")
            if task_type == "task":
                print(f"Task started: {task_event['payload']['name']}")
            elif task_type == "task_result":
                name = task_event['payload']['name']
                print(f"Task finished: {name}")

asyncio.run(trace_tasks())
```

### Example 3 — combining CheckpointsTransformer + TasksTransformer for full execution audit

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.stream.transformers import CheckpointsTransformer, TasksTransformer
from typing_extensions import TypedDict

class AuditState(TypedDict):
    step: int
    log: list[str]

def node_a(state: AuditState) -> dict:
    return {"step": state["step"] + 1, "log": state["log"] + ["node_a"]}

def node_b(state: AuditState) -> dict:
    return {"step": state["step"] + 1, "log": state["log"] + ["node_b"]}

builder = StateGraph(AuditState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "full-audit"}}

async def full_audit():
    checkpoints = []
    tasks = []

    # astream_events(version="v3") returns an AsyncGraphRunStream.
    # Both transformers are registered before the pump starts, so
    # asyncio.gather correctly drives both projections concurrently —
    # the pump's asyncio.Condition serialises pump steps while letting
    # multiple consumers observe each event.
    run = await graph.astream_events(
        {"step": 0, "log": []},
        config=config,
        version="v3",
        transformers=[CheckpointsTransformer, TasksTransformer],
    )
    async with run:
        async def collect_checkpoints():
            async for cp in run.checkpoints:
                checkpoints.append(cp)

        async def collect_tasks():
            async for t in run.tasks:
                tasks.append(t)

        # Subscribe both channels before driving the pump so each
        # channel buffers its events as the graph executes.
        await asyncio.gather(collect_checkpoints(), collect_tasks())

    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Tasks: {len(tasks)}")
    for t in tasks:
        if t.get("type") == "task":
            print(f"  Task: {t['payload']['name']}")

asyncio.run(full_audit())
```

---

## 2 · `StreamMessagesHandler` + `StreamMessagesHandlerV2`

**Module:** `langgraph.pregel._messages`

`StreamMessagesHandler` is the LangChain callback handler that powers `stream_mode="messages"`. It hooks into `on_chat_model_start`, `on_llm_new_token`, `on_llm_end`, and `on_chain_end` to collect token-by-token LLM output and deduplicated node output messages. `StreamMessagesHandlerV2` is its v2 counterpart used when `version="v2"` streaming is requested.

Understanding these classes is essential for debugging why certain messages appear (or don't appear) in your stream, and for writing custom callback integrations.

### Key implementation details

- `run_inline = True` — runs in the main thread to preserve ordering; no locks needed for most callbacks
- Messages are deduplicated by `message.id` — if the same ID has already been seen in `self.seen`, it's silently dropped
- `subgraphs=False` (default): messages from subgraphs with a non-empty namespace that wasn't the parent's namespace are suppressed
- The `parent_ns` parameter enables a subgraph to forward its streamed messages when it's explicitly invoked via `astream(stream_mode="messages")`
- `TAG_NOSTREAM` on a chat model call suppresses streaming for that specific LLM call

### Source structure

```python
class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    run_inline = True  # run in main thread — key for ordering correctness

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        subgraphs: bool,
        *,
        parent_ns: tuple[str, ...] | None = None,
    ) -> None:
        self.stream = stream          # emitter
        self.subgraphs = subgraphs    # include subgraph messages?
        self.metadata: dict[UUID, Meta] = {}  # keyed by run_id
        self.seen: set[int | str] = set()     # deduplicated message IDs
        self.parent_ns = parent_ns    # for subgraph-initiated streams
```

### Example 1 — understanding which messages stream

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated

class MsgState(TypedDict):
    messages: Annotated[list, add_messages]

def echo_node(state: MsgState) -> dict:
    last = state["messages"][-1]
    # Non-LLM message: emitted once in on_chain_end, deduped by ID
    return {"messages": [AIMessage(content=f"Echo: {last.content}", id="fixed-id-1")]}

builder = StateGraph(MsgState)
builder.add_node("echo", echo_node)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()

async def stream_messages_demo():
    print("Messages stream (stream_mode='messages'):")
    async for chunk, metadata in graph.astream(
        {"messages": [HumanMessage(content="Hello")]},
        stream_mode="messages",
    ):
        # chunk is a BaseMessage (partial token or full message)
        # metadata has langgraph_node, langgraph_checkpoint_ns, etc.
        print(f"  [{metadata['langgraph_node']}] {chunk.content!r}")

asyncio.run(stream_messages_demo())
```

### Example 2 — suppressing LLM streaming with TAG_NOSTREAM

```python
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

# The TAG_NOSTREAM tag tells StreamMessagesHandler to skip token emission
# for that specific LLM run. Useful for classification/routing LLMs you
# don't want to surface to end users.
TAG_NOSTREAM = "nostream"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str

async def classify(state: State) -> dict:
    # In a real app you'd call an LLM here with tags=[TAG_NOSTREAM]
    # to suppress its tokens from the messages stream:
    # result = await llm.ainvoke(state["messages"], config={"tags": [TAG_NOSTREAM]})
    return {"intent": "question"}

async def answer(state: State) -> dict:
    # This LLM call IS surfaced (no TAG_NOSTREAM)
    return {"messages": [AIMessage(content="I'll answer that!", id="ans-1")]}

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("answer", answer)
builder.add_edge(START, "classify")
builder.add_edge("classify", "answer")
builder.add_edge("answer", END)
graph = builder.compile()

async def run():
    async for msg, meta in graph.astream(
        {"messages": [HumanMessage("What is LangGraph?")], "intent": ""},
        stream_mode="messages",
    ):
        print(f"Node: {meta['langgraph_node']}, Content: {msg.content!r}")

asyncio.run(run())
```

### Example 3 — subgraph message forwarding with parent_ns

```python
# StreamMessagesHandler tracks parent_ns to allow subgraphs that are
# explicitly streamed with stream_mode="messages" to forward their output.
# This happens automatically when you call a subgraph inside a node with:
#
#   async for event in subgraph.astream(input, stream_mode="messages"):
#       ...
#
# The parent_ns is injected by the Pregel executor; you never set it directly.
# Here we illustrate the observable effect:

import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated

class SubState(TypedDict):
    messages: Annotated[list, add_messages]

def sub_node(state: SubState) -> dict:
    return {"messages": [AIMessage(content="from subgraph", id="sub-1")]}

sub_builder = StateGraph(SubState)
sub_builder.add_node("sub", sub_node)
sub_builder.add_edge(START, "sub")
sub_builder.add_edge("sub", END)
subgraph = sub_builder.compile()

class MainState(TypedDict):
    messages: Annotated[list, add_messages]

async def main_node(state: MainState) -> dict:
    # Stream subgraph messages explicitly — parent_ns enables forwarding
    results = []
    async for msg, meta in subgraph.astream(
        {"messages": state["messages"]},
        stream_mode="messages",
    ):
        results.append(msg)
    return {"messages": results}

main_builder = StateGraph(MainState)
main_builder.add_node("main", main_node)
main_builder.add_edge(START, "main")
main_builder.add_edge("main", END)
main_graph = main_builder.compile()

async def run():
    # With subgraphs=True the subgraph messages are visible at the top level
    async for msg, meta in main_graph.astream(
        {"messages": [HumanMessage("Hi")]},
        stream_mode="messages",
        subgraphs=True,
    ):
        print(f"Message: {msg.content!r} from {meta['langgraph_node']}")

asyncio.run(run())
```

---

## 3 · `_TimedAttemptScope` + `_AttemptContext` + `_AttemptEvent`

**Module:** `langgraph.pregel._retry`

These three classes form the internal retry/timeout lifecycle observer contract. They are underscore-prefixed because they are an internal API consumed by `langgraph-server`; do not import them by any other path. Understanding them explains exactly what "idle timeout" means, how `runtime.heartbeat()` works, and how `_TimedAttemptScope` serializes writes after a timeout fires.

### `_AttemptContext` — immutable per-attempt metadata

```python
class _AttemptContext(NamedTuple):
    task_id: str
    task_name: str
    attempt: int          # 1-based attempt number
    run_id: str | None
    thread_id: str | None
    checkpoint_ns: str | None
    started_at: datetime
    run_timeout_secs: float | None    # hard wall-clock cap
    idle_timeout_secs: float | None   # progress-based cap
    refresh_on: Literal["auto", "heartbeat"] | None
```

Built once at attempt start and referenced by every `_AttemptEvent` for that attempt — no per-event allocation beyond the event wrapper itself.

### `_AttemptEvent` — lifecycle notification

```python
@dataclass(frozen=True, slots=True)
class _AttemptEvent:
    context: _AttemptContext
    event: Literal["start", "progress", "finish"]
    progress_at: datetime | None = None   # set on "progress" events
    finished_at: datetime | None = None   # set on "finish"
    status: Literal["success", "error"] | None = None
    error_type: str | None = None
    error_message: str | None = None
```

### `_TimedAttemptScope` — the enforcement boundary

`_TimedAttemptScope` wraps the node's `RunnableConfig` to intercept every write, stream event, runtime stream writer call, and child task schedule. For `refresh_on="auto"`, any of these automatically update `_last_progress`, resetting the idle countdown. For `refresh_on="heartbeat"`, only `runtime.heartbeat()` → `scope.touch()` resets the clock.

On timeout, `asyncio.TimeoutError` is raised inside `wait_for_idle_timeout()`. The `close()` method is called under a threading lock so background tasks cannot persist writes past the boundary.

### Example 1 — observing the retry lifecycle via run control metadata

```python
import asyncio
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, TimeoutPolicy
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

class State(TypedDict):
    attempt: int
    result: str

attempt_count = 0

async def flaky_node(state: State, runtime: Runtime) -> dict:
    global attempt_count
    attempt_count += 1

    if attempt_count < 3:
        raise ValueError(f"Simulated failure on attempt {attempt_count}")

    # execution_info.task_id is the _AttemptContext.task_id
    info = runtime.execution_info
    return {
        "attempt": attempt_count,
        "result": f"success on attempt {attempt_count}, task_id={info.task_id}",
    }

builder = StateGraph(State)
builder.add_node(
    "flaky",
    flaky_node,
    retry_policy=RetryPolicy(max_attempts=5, retry_on=ValueError),
    timeout=TimeoutPolicy(run_timeout=10.0),
)
builder.add_edge(START, "flaky")
builder.add_edge("flaky", END)
graph = builder.compile()

async def run():
    global attempt_count
    attempt_count = 0
    result = await graph.ainvoke({"attempt": 0, "result": ""})
    print(f"Result: {result['result']}")

asyncio.run(run())
# Output: success on attempt 3, task_id=<uuid>
```

### Example 2 — idle timeout with explicit heartbeats

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

class State(TypedDict):
    processed: int

async def slow_node(state: State, runtime: Runtime) -> dict:
    """A node that does slow work but uses heartbeat to avoid idle timeout."""
    for i in range(5):
        # Simulate slow external I/O
        await asyncio.sleep(0.1)
        # Tell the executor we're still alive
        runtime.heartbeat()

    return {"processed": state["processed"] + 1}

builder = StateGraph(State)
builder.add_node(
    "slow",
    slow_node,
    timeout=TimeoutPolicy(
        idle_timeout=1.0,       # 1 second idle timeout
        refresh_on="heartbeat", # ONLY heartbeat() resets the clock
    ),
)
builder.add_edge(START, "slow")
builder.add_edge("slow", END)
graph = builder.compile()

async def run():
    result = await graph.ainvoke({"processed": 0})
    print(f"Processed: {result['processed']}")

asyncio.run(run())
# Output: Processed: 1  (heartbeats kept the idle timeout from firing)
```

### Example 3 — understanding the difference between run_timeout and idle_timeout

```python
# run_timeout: wall-clock maximum for the entire node execution
# idle_timeout: maximum time since last observable progress
# refresh_on="auto": writes/streams/child-task-schedules count as progress
# refresh_on="heartbeat": only runtime.heartbeat() counts

import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.errors import NodeTimeoutError
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

async def bounded_node(state: State) -> dict:
    # This node does fast work — will complete within any reasonable timeout
    await asyncio.sleep(0.01)
    return {"result": "done"}

builder = StateGraph(State)
builder.add_node(
    "bounded",
    bounded_node,
    # Hard 5s cap + 2s idle cap; any stream/write resets idle
    timeout=TimeoutPolicy(run_timeout=5.0, idle_timeout=2.0, refresh_on="auto"),
    # Retry up to 2 times on timeout
    retry_policy=RetryPolicy(max_attempts=2, retry_on=NodeTimeoutError),
)
builder.add_edge(START, "bounded")
builder.add_edge("bounded", END)
graph = builder.compile()

async def run():
    result = await graph.ainvoke({"result": ""})
    print(result["result"])

asyncio.run(run())
```

---

## 4 · `ChannelRead` — imperative state access

**Module:** `langgraph.pregel._read`

`ChannelRead` is a `RunnableCallable` that reads one or more channels from the current graph state. It's used internally by LangGraph to wire node inputs and can be used directly to build custom Runnables that inspect graph state without being a node themselves.

The static method `ChannelRead.do_read()` is the lower-level primitive — callable from any code that has access to the LangGraph `RunnableConfig`.

### Source signature

```python
class ChannelRead(RunnableCallable):
    channel: str | list[str]
    fresh: bool = False          # True = bypass in-step cache; read raw channel
    mapper: Callable | None = None  # transform the result before returning

    def __init__(
        self,
        channel: str | list[str],
        *,
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
        tags: list[str] | None = None,
    ) -> None: ...

    @staticmethod
    def do_read(
        config: RunnableConfig,
        *,
        select: str | list[str],
        fresh: bool = False,
        mapper: Callable[[Any], Any] | None = None,
    ) -> Any: ...
```

### Key behaviours

| Parameter | Effect |
|-----------|--------|
| `channel: str` | Returns the raw channel value |
| `channel: list[str]` | Returns `dict[channel_name, value]` |
| `fresh=True` | Bypasses the in-step cache; reads the committed channel value |
| `mapper` | Transform applied to the raw value before returning |
| Outside Pregel context | Raises `RuntimeError("Not configured with a read function")` |

### Example 1 — reading a single channel inside a node

```python
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import ChannelRead
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

class State(TypedDict):
    count: int
    label: str

def annotate_node(state: State, config: RunnableConfig) -> dict:
    # Read the current 'count' channel value imperatively
    current_count = ChannelRead.do_read(config, select="count")
    return {"label": f"count_is_{current_count}"}

builder = StateGraph(State)
builder.add_node("annotate", annotate_node)
builder.add_edge(START, "annotate")
builder.add_edge("annotate", END)
graph = builder.compile()

result = graph.invoke({"count": 42, "label": ""})
print(result["label"])  # count_is_42
```

### Example 2 — reading multiple channels at once

```python
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import ChannelRead
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

class State(TypedDict):
    x: int
    y: int
    summary: str

def summarise(state: State, config: RunnableConfig) -> dict:
    # Reading a list of channels returns a dict
    vals = ChannelRead.do_read(config, select=["x", "y"])
    return {"summary": f"x={vals['x']}, y={vals['y']}, sum={vals['x'] + vals['y']}"}

builder = StateGraph(State)
builder.add_node("summarise", summarise)
builder.add_edge(START, "summarise")
builder.add_edge("summarise", END)
graph = builder.compile()

result = graph.invoke({"x": 3, "y": 7, "summary": ""})
print(result["summary"])  # x=3, y=7, sum=10
```

### Example 3 — ChannelRead as a composable Runnable

```python
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import ChannelRead
from typing_extensions import TypedDict

class State(TypedDict):
    items: list[str]
    count: int

# ChannelRead is a Runnable — pipe it directly into other runnables.
# The channel name ("items") is read from Pregel state and the mapper
# transforms the value before passing it downstream.
count_reader = ChannelRead("items", mapper=len)

builder = StateGraph(State)
# Use count_reader as the node: reads items channel → maps to int →
# RunnableLambda wraps it in the expected state-update dict.
builder.add_node("record", count_reader | RunnableLambda(lambda count: {"count": count}))
builder.add_edge(START, "record")
builder.add_edge("record", END)
graph = builder.compile()

result = graph.invoke({"items": ["a", "b", "c"], "count": 0})
print(result["count"])  # 3
```

---

## 5 · `ErrorCode` enum + extended exception hierarchy

**Module:** `langgraph.errors`

LangGraph defines a rich exception hierarchy. Most developers know `GraphRecursionError` and `NodeTimeoutError`, but the less-documented exceptions (`EmptyInputError`, `TaskNotFound`, `GraphBubbleUp`, `ParentCommand`, `ErrorCode`) are equally important for robust error handling.

### Full exception hierarchy

```
Exception
├── GraphBubbleUp          — internal signal; never catch in user code
│   ├── GraphInterrupt     — raised when interrupt() is called
│   │   └── NodeInterrupt  — deprecated alias (v0.x)
│   └── ParentCommand      — Command.PARENT bubbled up through subgraph
├── GraphRecursionError    — recursion_limit exceeded
├── InvalidUpdateError     — concurrent conflicting channel writes
├── NodeError              — node raised an unexpected exception
├── NodeCancelledError     — node was cancelled
├── NodeTimeoutError       — TimeoutPolicy limit exceeded
├── EmptyInputError        — graph invoked with empty input
└── TaskNotFound           — executor cannot locate a task (distributed mode)
```

### `ErrorCode` enum

```python
class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"
```

`ErrorCode` values are attached to certain exceptions as `error.error_code`. The server uses them to classify errors for structured error responses.

### Example 1 — comprehensive error handling by type

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.errors import (
    GraphRecursionError,
    NodeTimeoutError,
    NodeCancelledError,
    InvalidUpdateError,
    EmptyInputError,
)
from langgraph.types import RetryPolicy, TimeoutPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    value: int
    mode: str  # "normal" | "timeout" | "loop"

async def risky_node(state: State) -> dict:
    if state["mode"] == "timeout":
        await asyncio.sleep(10)   # will hit TimeoutPolicy
    return {"value": state["value"] + 1}

def loop_router(state: State) -> str:
    if state["mode"] == "loop" and state["value"] < 100:
        return "risky"   # infinite loop unless recursion_limit fires
    return "__end__"

builder = StateGraph(State)
builder.add_node("risky", risky_node, timeout=TimeoutPolicy(run_timeout=0.1))
builder.add_conditional_edges("risky", loop_router, ["risky", "__end__"])
builder.add_edge(START, "risky")
graph = builder.compile()

async def run(mode: str):
    try:
        result = await graph.ainvoke(
            {"value": 0, "mode": mode},
            config={"recursion_limit": 5},
        )
        print(f"[{mode}] Success: {result['value']}")
    except GraphRecursionError as e:
        print(f"[{mode}] Recursion limit hit: {e}")
    except NodeTimeoutError as e:
        print(f"[{mode}] Timeout: {e}")
    except EmptyInputError as e:
        print(f"[{mode}] Empty input: {e}")

asyncio.run(run("normal"))   # Success: 1
asyncio.run(run("timeout"))  # Timeout: ...
asyncio.run(run("loop"))     # Recursion limit hit: ...
```

### Example 2 — catching GraphBubbleUp in subgraphs

```python
# GraphBubbleUp and its subclasses (GraphInterrupt, ParentCommand) are
# internal signals. They must NOT be caught in node code. If you need to
# detect an interrupt from outside the graph, catch it at the top level:

import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.errors import GraphBubbleUp
from typing_extensions import TypedDict

class State(TypedDict):
    approved: bool
    result: str

def approval_node(state: State) -> dict:
    response = interrupt({"question": "Approve this action?"})
    return {"approved": response == "yes"}

def action_node(state: State) -> dict:
    if not state["approved"]:
        return {"result": "denied"}
    return {"result": "executed"}

builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_node("action", action_node)
builder.add_edge(START, "approval")
builder.add_edge("approval", "action")
builder.add_edge("action", END)

from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["approval"])

async def run():
    config = {"configurable": {"thread_id": "hitl-1"}}
    # interrupt_before=["approval"] — first invocation pauses before the node
    # and returns normally (no exception raised).
    await graph.ainvoke({"approved": False, "result": ""}, config=config)

    # Inspect pending interrupts before resuming
    snap = await graph.aget_state(config)
    print(f"Next node(s): {snap.next}")   # ('approval',)

    # Resume with approval
    result = await graph.ainvoke(Command(resume="yes"), config=config)
    print(f"Result: {result['result']}")

asyncio.run(run())
```

### Example 3 — EmptyInputError and TaskNotFound in distributed mode

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import EmptyInputError, TaskNotFound
from typing_extensions import TypedDict

class State(TypedDict):
    message: str

def echo(state: State) -> dict:
    return {"message": state["message"]}

builder = StateGraph(State)
builder.add_node("echo", echo)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()

# EmptyInputError fires when the graph receives empty input
# (e.g., {} for a state that requires values)
try:
    graph.invoke({})
except EmptyInputError as e:
    print(f"EmptyInputError: {e}")
except Exception as e:
    print(f"Other: {type(e).__name__}: {e}")

# TaskNotFound is raised by the distributed executor when a task ID
# is referenced but cannot be found in the task registry. It indicates
# a race condition or a missed heartbeat in a distributed LangGraph Platform
# deployment. In local mode it should never occur.
```

---

## 6 · `ValidationNode` — schema validation node

**Module:** `langgraph.prebuilt.tool_validator`

`ValidationNode` validates tool call arguments from the last `AIMessage` against Pydantic schemas without executing the tools. It's useful for extraction workflows where you need to ensure structured output conforms to a schema before proceeding. Note: it is deprecated in v1.0 in favour of using `create_agent` from `langchain.agents` with custom tool error handling, but the validation pattern itself remains important.

**Migration note:** `ValidationNode` emits `LangGraphDeprecatedSinceV10`. Use `ToolNode` with a custom `handle_tool_errors` or a Pydantic-validated `ToolNode` wrapper for new code.

### Source signature

```python
class ValidationNode(RunnableCallable):
    def __init__(
        self,
        schemas: Sequence[BaseTool | type[BaseModel] | Callable],
        *,
        format_error: Callable[[BaseException, ToolCall, type[BaseModel]], str] | None = None,
        name: str = "validation",
        tags: list[str] | None = None,
    ) -> None: ...
```

Accepted schema types:
- `type[BaseModel]` — Pydantic v1 or v2 model class; uses `model.model_name` as tool name
- `BaseTool` — extracts `tool.args_schema`; the tool name is used for routing
- `Callable` — a schema is auto-created from the function signature

### Example 1 — extraction loop with schema re-prompting

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
from typing import Literal, Annotated
from pydantic import BaseModel, field_validator
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ValidationNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from typing_extensions import TypedDict

class ExtractNumber(BaseModel):
    """Extract a lucky number from user input."""
    value: int

    @field_validator("value")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("value must be positive")
        return v

class State(TypedDict):
    messages: Annotated[list, add_messages]

def fake_llm(state: State) -> dict:
    """Simulate an LLM that generates a tool call."""
    # In a real app this would be: llm.bind_tools([ExtractNumber]).invoke(...)
    # Use dict literals — TypedDict keyword-arg syntax fails on Python 3.10.
    # First attempt: send -5 (invalid) to trigger re-prompting
    if len(state["messages"]) == 1:
        tool_call = {"name": "ExtractNumber", "args": {"value": -5}, "id": "tc-1", "type": "tool_call"}
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
    # Second attempt: send valid value
    tool_call = {"name": "ExtractNumber", "args": {"value": 42}, "id": "tc-2", "type": "tool_call"}
    return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

def should_validate(state: State) -> Literal["validation", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "validation"
    return "__end__"

def should_reprompt(state: State) -> Literal["llm", "__end__"]:
    for msg in reversed(state["messages"]):
        if msg.type == "ai":
            return "__end__"
        if getattr(msg, "additional_kwargs", {}).get("is_error"):
            return "llm"
    return "__end__"

builder = StateGraph(State)
builder.add_node("llm", fake_llm)
builder.add_node("validation", ValidationNode([ExtractNumber]))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_validate)
builder.add_conditional_edges("validation", should_reprompt)

graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage("Give me a number")]})
print([m.content for m in result["messages"] if m.type == "tool"])
```

### Example 2 — custom error formatting

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pydantic import BaseModel, field_validator
from langchain_core.messages import ToolCall
from langgraph.prebuilt import ValidationNode

class Coordinates(BaseModel):
    lat: float
    lon: float

    @field_validator("lat")
    @classmethod
    def valid_lat(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError(f"latitude {v} out of range [-90, 90]")
        return v

def my_format_error(exc: BaseException, call: ToolCall, schema: type) -> str:
    return (
        f"Tool '{call['name']}' called with invalid arguments: {exc}. "
        f"Please fix and retry. Schema: {schema.model_fields}"
    )

node = ValidationNode([Coordinates], format_error=my_format_error)
# The node would be wired into a StateGraph the same way as Example 1
print("ValidationNode created with custom error formatter")
```

### Example 3 — modern equivalent using ToolNode

```python
# This is the recommended modern pattern replacing ValidationNode.
# Use ToolNode with handle_tool_errors for schema validation + error surfacing.

from pydantic import BaseModel, field_validator
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

class SearchParams(BaseModel):
    query: str
    max_results: int = 10

    @field_validator("max_results")
    @classmethod
    def reasonable_limit(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"max_results must be between 1 and 100, got {v}")
        return v

@tool(args_schema=SearchParams)
def search(query: str, max_results: int = 10) -> str:
    """Search the web."""
    return f"Found {max_results} results for '{query}'"

# handle_tool_errors=True catches ValidationError and other exceptions,
# surfaces them as ToolMessages, and lets the LLM retry.
tool_node = ToolNode([search], handle_tool_errors=True)
print(f"ToolNode tools: {[t.name for t in tool_node.tools_by_name.values()]}")
```

---

## 7 · `MessageGraph` — legacy API + migration

**Module:** `langgraph.graph.message`

`MessageGraph` was the original LangGraph graph type from v0.x. It hardcoded the state as `Annotated[list[AnyMessage], add_messages]` — the entire state was a single message list. It is **fully deprecated** in v1.0 and will be removed in v2.0.

The migration path is to use `StateGraph` with a `messages` key.

### Source signature

```python
class MessageGraph(StateGraph):
    """Deprecated. Use StateGraph with a messages key instead."""
    def __init__(self) -> None:
        super().__init__(Annotated[list[AnyMessage], add_messages])
```

### Example 1 — original MessageGraph pattern (for reference only)

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import MessageGraph

# DEPRECATED: do not use in new code
builder = MessageGraph()

def chatbot(messages: list) -> list:
    last = messages[-1]
    return [AIMessage(content=f"Echo: {last.content}")]

builder.add_node("chatbot", chatbot)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")
graph = builder.compile()

result = graph.invoke([HumanMessage(content="Hello")])
print([m.content for m in result])
# ['Hello', 'Echo: Hello']
```

### Example 2 — direct migration to StateGraph

```python
# Modern equivalent of the MessageGraph pattern above
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: State) -> dict:
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"Echo: {last.content}")]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="Hello")]})
print([m.content for m in result["messages"]])
# ['Hello', 'Echo: Hello']
```

### Example 3 — using MessagesState shorthand

```python
# MessagesState is the cleanest replacement for MessageGraph state
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import MessagesState

# MessagesState is equivalent to TypedDict with
# messages: Annotated[list[BaseMessage], add_messages]
# plus optional remaining_steps: RemainingSteps

def chatbot_node(state: MessagesState) -> dict:
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"Reply: {last.content}")]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chatbot_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("hi")]})
print(result["messages"][-1].content)  # Reply: hi
```

---

## 8 · `BackgroundExecutor` + `AsyncBackgroundExecutor` + `Submit`

**Module:** `langgraph.pregel._executor`

`BackgroundExecutor` (sync) and `AsyncBackgroundExecutor` (async) are context managers that run tasks in the background — thread pool for sync, asyncio tasks for async. They expose the `Submit` protocol as their context manager return value.

Both manage task lifecycle flags: `__cancel_on_exit__` (cancel if not started when the context exits) and `__reraise_on_exit__` (propagate the first task exception on exit). The `__next_tick__` flag defers execution to the next event loop tick.

`GraphBubbleUp` exceptions from tasks are never re-raised — they're internal interrupt signals consumed by the Pregel loop.

### `Submit` protocol

```python
class Submit(Protocol[P, T]):
    def __call__(
        self,
        fn: Callable[P, T],
        *args: P.args,
        __name__: str | None = None,
        __cancel_on_exit__: bool = False,
        __reraise_on_exit__: bool = True,
        __next_tick__: bool = False,
        **kwargs: P.kwargs,
    ) -> concurrent.futures.Future[T] | asyncio.Future[T]: ...
```

### Example 1 — sync BackgroundExecutor for parallel work

```python
import concurrent.futures
import time
from langgraph.pregel._executor import BackgroundExecutor

def slow_work(name: str, delay: float) -> str:
    time.sleep(delay)
    return f"done:{name}"

# BackgroundExecutor requires a RunnableConfig for max_concurrency support
config = {}

with BackgroundExecutor(config) as submit:
    f1 = submit(slow_work, "task-1", 0.05)
    f2 = submit(slow_work, "task-2", 0.05, __cancel_on_exit__=False)
    f3 = submit(slow_work, "task-3", 0.05, __reraise_on_exit__=False)
    # Context exit waits for all tasks and re-raises errors from reraise=True tasks
    print(f"Submitted 3 tasks; waiting...")

# All done by here
print(f1.result())  # done:task-1
print(f2.result())  # done:task-2
print(f3.result())  # done:task-3
```

### Example 2 — async AsyncBackgroundExecutor with max_concurrency

```python
import asyncio
from langgraph.pregel._executor import AsyncBackgroundExecutor

async def async_work(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"async_done:{name}"

async def run():
    # max_concurrency=2 limits concurrent tasks via asyncio.Semaphore
    config = {"max_concurrency": 2}

    async with AsyncBackgroundExecutor(config) as submit:
        futures = []
        for i in range(5):
            f = submit(async_work, f"task-{i}", 0.02)
            futures.append(f)
        # On __aexit__, waits for all and re-raises errors

    results = [f.result() for f in futures]
    print(results)

asyncio.run(run())
# ['async_done:task-0', 'async_done:task-1', ..., 'async_done:task-4']
```

### Example 3 — cancel_on_exit pattern for background monitoring

```python
import asyncio
from langgraph.pregel._executor import AsyncBackgroundExecutor

async def monitor(name: str, stop_event: asyncio.Event) -> None:
    """Background monitor — should be cancelled when main work finishes."""
    while not stop_event.is_set():
        await asyncio.sleep(0.01)
    print(f"{name} monitor stopped cleanly")

async def main_work() -> str:
    await asyncio.sleep(0.05)
    return "main_done"

async def run():
    stop = asyncio.Event()
    config = {}

    async with AsyncBackgroundExecutor(config) as submit:
        # Monitor is cancelled on context exit (not re-raised)
        submit(
            monitor, "perf", stop,
            __cancel_on_exit__=True,
            __reraise_on_exit__=False,
        )
        # Main work must finish before context exits
        result_future = submit(main_work)

    # monitor was cancelled; main_work result is available
    print(result_future.result())   # main_done

asyncio.run(run())
```

---

## 9 · `HumanInterrupt` full protocol

**Module:** `langgraph.prebuilt.interrupt` (v0.x) → `langchain.agents.interrupt` (v1.0+)

The `HumanInterrupt` protocol (`ActionRequest`, `HumanInterruptConfig`, `HumanInterrupt`, `HumanResponse`) is a structured HITL contract. In v1.0, these classes moved to `langchain.agents.interrupt`; the `langgraph.prebuilt.interrupt` re-exports emit `LangGraphDeprecatedSinceV10`.

Always import from `langchain.agents.interrupt` in new code.

### Class definitions (from `langchain.agents.interrupt`)

```python
class ActionRequest(TypedDict):
    action: str   # e.g. "run_command" or "approve_transfer"
    args: dict    # arbitrary key-value pairs for the action

class HumanInterruptConfig(TypedDict):
    allow_ignore: bool    # human can skip this step
    allow_respond: bool   # human can send text feedback
    allow_edit: bool      # human can edit the ActionRequest args
    allow_accept: bool    # human can approve without changes

class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str | None

class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest  # None for ignore/accept, str for respond, ActionRequest for edit
```

### Example 1 — full HITL approval flow

```python
import asyncio
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

# Import from the new location (v1.0+)
try:
    from langchain.agents.interrupt import (
        ActionRequest, HumanInterruptConfig, HumanInterrupt, HumanResponse
    )
except ImportError:
    # Fallback for environments without langchain.agents
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langgraph.prebuilt.interrupt import (
        ActionRequest, HumanInterruptConfig, HumanInterrupt, HumanResponse
    )

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    command: str | None
    approved: bool

def propose_command(state: State) -> dict:
    cmd = "ls -la /tmp"
    return {"command": cmd}

def request_approval(state: State) -> dict:
    # Use dict literals — TypedDict keyword-arg syntax fails on Python 3.10.
    request: HumanInterrupt = {
        "action_request": {
            "action": "execute_command",
            "args": {"command": state["command"]},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": f"About to run: `{state['command']}`. Approve?",
    }
    # Suspend execution; resume with HumanResponse
    response: HumanResponse = interrupt(request)

    if response["type"] == "accept":
        return {"approved": True}
    elif response["type"] == "edit":
        new_cmd = response["args"]["args"]["command"]
        return {"command": new_cmd, "approved": True}
    elif response["type"] == "ignore":
        return {"approved": False}
    else:  # "response"
        # Feedback as text — treat as rejection
        return {
            "approved": False,
            "messages": [AIMessage(content=f"Feedback: {response['args']}")],
        }

def execute_command(state: State) -> dict:
    if state["approved"]:
        return {"messages": [AIMessage(content=f"Executed: {state['command']}")]}
    return {"messages": [AIMessage(content="Skipped.")]}

builder = StateGraph(State)
builder.add_node("propose", propose_command)
builder.add_node("approve", request_approval)
builder.add_node("execute", execute_command)
builder.add_edge(START, "propose")
builder.add_edge("propose", "approve")
builder.add_edge("approve", "execute")
builder.add_edge("execute", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)

async def run():
    config = {"configurable": {"thread_id": "hitl-approve-1"}}

    # First run: propose + hit interrupt at 'approve'
    await graph.ainvoke(
        {"messages": [], "command": None, "approved": False},
        config=config,
    )

    # Check the state to see the pending interrupt
    state = graph.get_state(config)
    print(f"Pending interrupts: {state.interrupts}")

    # Resume with acceptance
    result = await graph.ainvoke(
        Command(resume={"type": "accept", "args": None}),
        config=config,
    )
    print(result["messages"][-1].content)  # Executed: rm -rf /tmp/old_data

asyncio.run(run())
```

### Example 2 — multi-step HITL with edit capability

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

try:
    from langchain.agents.interrupt import ActionRequest, HumanInterruptConfig, HumanInterrupt, HumanResponse
except ImportError:
    import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langgraph.prebuilt.interrupt import ActionRequest, HumanInterruptConfig, HumanInterrupt, HumanResponse

class State(TypedDict):
    draft: str
    final: str

def draft_node(state: State) -> dict:
    return {"draft": "Subject: Meeting\n\nHi team, please join tomorrow's standup."}

def edit_and_approve(state: State) -> dict:
    # Use dict literals — TypedDict keyword-arg syntax fails on Python 3.10.
    request: HumanInterrupt = {
        "action_request": {
            "action": "review_email",
            "args": {"draft": state["draft"]},
        },
        "config": {
            "allow_ignore": False,
            "allow_respond": False,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": "Review the draft email before sending.",
    }
    response: HumanResponse = interrupt(request)

    if response["type"] == "accept":
        return {"final": state["draft"]}
    elif response["type"] == "edit":
        return {"final": response["args"]["args"]["draft"]}
    return {"final": ""}

def send_node(state: State) -> dict:
    print(f"Sending email:\n{state['final']}")
    return {}

builder = StateGraph(State)
builder.add_node("draft", draft_node)
builder.add_node("review", edit_and_approve)
builder.add_node("send", send_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", "send")
builder.add_edge("send", END)

graph = builder.compile(checkpointer=InMemorySaver())

async def run():
    config = {"configurable": {"thread_id": "email-1"}}

    # Phase 1: draft + interrupt at review
    await graph.ainvoke({"draft": "", "final": ""}, config=config)

    # Phase 2: human edits the email
    edited = "Subject: Meeting Tomorrow\n\nHi team, standup at 10am. Please be on time!"
    result = await graph.ainvoke(
        Command(resume={"type": "edit", "args": {"action": "review_email", "args": {"draft": edited}}}),
        config=config,
    )
    print(f"Final email sent: {result['final'][:40]}...")

asyncio.run(run())
```

### Example 3 — choosing interrupt response type based on HumanInterruptConfig

```python
from typing_extensions import TypedDict

try:
    from langchain.agents.interrupt import HumanInterruptConfig, HumanResponse
except ImportError:
    import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanResponse

def build_response(
    config: HumanInterruptConfig,
    choice: str,
    text: str | None = None,
    action_name: str = "unknown_action",
    edited_args: dict | None = None,
) -> HumanResponse:
    """Build a HumanResponse respecting the config constraints.

    For "edit" responses, action_name should match the original ActionRequest.action
    that the interrupt requested — it identifies which action is being edited.
    """
    allowed = {
        "accept": config["allow_accept"],
        "ignore": config["allow_ignore"],
        "response": config["allow_respond"],
        "edit": config["allow_edit"],
    }
    if not allowed.get(choice, False):
        raise ValueError(f"Action '{choice}' is not permitted by this interrupt config")

    # Use dict literals — TypedDict keyword-arg syntax fails on Python 3.10.
    if choice in ("accept", "ignore"):
        return {"type": choice, "args": None}
    elif choice == "response":
        return {"type": "response", "args": text or ""}
    else:  # edit — args must be an ActionRequest carrying the (edited) action + args
        return {"type": "edit", "args": {"action": action_name, "args": edited_args or {}}}

# Example usage — dict literal for Python 3.10 compatibility
cfg: HumanInterruptConfig = {
    "allow_ignore": True, "allow_respond": True, "allow_edit": True, "allow_accept": True
}

accept = build_response(cfg, "accept")
print(accept)  # {'type': 'accept', 'args': None}

respond = build_response(cfg, "response", text="Looks good, but change the subject")
print(respond)  # {'type': 'response', 'args': 'Looks good...'}

edit = build_response(cfg, "edit", action_name="execute_command", edited_args={"command": "ls -la"})
print(edit)  # {'type': 'edit', 'args': {'action': 'execute_command', 'args': {'command': 'ls -la'}}}
```

---

## 10 · `PregelNode` + `StateNodeSpec` — node representation internals

**Modules:** `langgraph.pregel._read`, `langgraph.graph._node`

`StateNodeSpec` is the declarative spec that `StateGraph.add_node()` builds and stores. `PregelNode` is the compiled representation that the Pregel executor uses at runtime. Understanding these two classes explains how policies (retry, cache, timeout, error handler) attach to nodes and how the input/output wiring works.

### `StateNodeSpec` — the declarative spec

```python
@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode         # the actual callable/runnable
    metadata: dict | None
    input_schema: type          # narrowed input type (from input= kwarg on add_node)
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    is_error_handler: bool = False
    error_handler_node: str | None = None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
    timeout: TimeoutPolicy | None = None
```

### `PregelNode` — the compiled execution container

```python
class PregelNode:
    channels: str | list[str]   # input channels → node bound
    triggers: list[str]         # write-to → trigger this node next step
    mapper: Callable | None     # transform input before bound
    writers: list[Runnable]     # post-bound output → channel writes
    bound: Runnable             # the actual node logic
    retry_policy: Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    timeout: TimeoutPolicy | None
    is_error_handler: bool
    error_handler_node: str | None
    subgraphs: Sequence[PregelProtocol]
```

### Example 1 — inspecting compiled node metadata

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

def my_node(state: State) -> dict:
    return {"value": state["value"] + 1}

builder = StateGraph(State)
builder.add_node(
    "my_node",
    my_node,
    retry_policy=RetryPolicy(max_attempts=3),
    cache_policy=CachePolicy(ttl=60.0),
    timeout=TimeoutPolicy(run_timeout=5.0),
    metadata={"owner": "team-infra", "version": "2"},
)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)

# Inspect StateNodeSpec BEFORE compilation
spec = builder.nodes["my_node"]
print(f"StateNodeSpec:")
print(f"  retry_policy: {spec.retry_policy}")
print(f"  cache_policy: {spec.cache_policy}")
print(f"  timeout: {spec.timeout}")
print(f"  metadata: {spec.metadata}")
print(f"  is_error_handler: {spec.is_error_handler}")
```

### Example 2 — inspecting PregelNode after compilation

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, TimeoutPolicy
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int

def counter_node(state: State) -> dict:
    return {"counter": state["counter"] + 1}

builder = StateGraph(State)
builder.add_node(
    "counter",
    counter_node,
    retry_policy=RetryPolicy(max_attempts=2),
    timeout=TimeoutPolicy(run_timeout=3.0),
)
builder.add_edge(START, "counter")
builder.add_edge("counter", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Access the compiled PregelNode
pregel_node = graph.nodes["counter"]
print(f"PregelNode:")
print(f"  channels: {pregel_node.channels}")
print(f"  triggers: {pregel_node.triggers}")
print(f"  retry_policy: {pregel_node.retry_policy}")
print(f"  timeout: {pregel_node.timeout}")
print(f"  subgraphs: {pregel_node.subgraphs}")
print(f"  writers count: {len(pregel_node.writers)}")
```

### Example 3 — error handler node wiring

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    result: str
    error: str | None

def risky_operation(state: State) -> dict:
    raise ValueError("Something went wrong")

def error_handler(state: State) -> dict:
    # error handlers receive the same state; access the exception via
    # the langgraph error handling mechanism
    return {"error": "caught", "result": "fallback"}

builder = StateGraph(State)
builder.add_node("risky", risky_operation)
builder.add_node(
    "error_handler",
    error_handler,
    is_error_handler=True,    # marks as error handler
)
builder.add_edge(START, "risky")
builder.add_edge("risky", END)
# Wire error handler: if risky raises, route to error_handler
# This is done via add_node's error_handler kwarg:
# builder.add_node("risky", risky_operation, error_handler="error_handler")

# Check the StateNodeSpec that records the error_handler_node link
spec = builder.nodes.get("risky")
if spec:
    print(f"error_handler_node: {spec.error_handler_node}")
    print(f"is_error_handler: {spec.is_error_handler}")

# Build graph with proper error routing: pass the callable directly,
# not a string. add_node creates/marks the error-handler node internally.
builder2 = StateGraph(State)
builder2.add_node("risky", risky_operation, error_handler=error_handler)
builder2.add_node("error_handler", error_handler)
builder2.add_edge(START, "risky")
builder2.add_edge("risky", END)
builder2.add_edge("error_handler", END)
graph2 = builder2.compile()

result = graph2.invoke({"result": "", "error": None})
print(f"result={result['result']}, error={result['error']}")
# result=fallback, error=caught
```

---

## Summary

| Class | Module | Key use case |
|-------|--------|-------------|
| `CheckpointsTransformer` | `langgraph.stream.transformers` | Checkpoint audit stream in v3 API |
| `TasksTransformer` | `langgraph.stream.transformers` | Task lifecycle stream in v3 API |
| `StreamMessagesHandler` | `langgraph.pregel._messages` | Debug stream_mode="messages" filtering |
| `StreamMessagesHandlerV2` | `langgraph.pregel._messages` | v2 typed messages callback |
| `_TimedAttemptScope` | `langgraph.pregel._retry` | Timeout boundary + progress observation |
| `_AttemptContext` | `langgraph.pregel._retry` | Per-attempt immutable metadata |
| `_AttemptEvent` | `langgraph.pregel._retry` | Retry lifecycle event observer |
| `ChannelRead` | `langgraph.pregel._read` | Imperative state access inside nodes |
| `ErrorCode` | `langgraph.errors` | Structured error classification |
| `ValidationNode` | `langgraph.prebuilt.tool_validator` | Schema validation (deprecated; use ToolNode) |
| `MessageGraph` | `langgraph.graph.message` | Legacy API (deprecated; use StateGraph) |
| `BackgroundExecutor` | `langgraph.pregel._executor` | Sync background task pool |
| `AsyncBackgroundExecutor` | `langgraph.pregel._executor` | Async background task pool |
| `Submit` | `langgraph.pregel._executor` | Background task submission protocol |
| `HumanInterrupt` | `langchain.agents.interrupt` | Structured HITL interrupt payload |
| `ActionRequest` | `langchain.agents.interrupt` | Action + args for HITL requests |
| `HumanInterruptConfig` | `langchain.agents.interrupt` | Allowed response types for HITL |
| `HumanResponse` | `langchain.agents.interrupt` | Human response to HITL interrupt |
| `PregelNode` | `langgraph.pregel._read` | Compiled node execution container |
| `StateNodeSpec` | `langgraph.graph._node` | Declarative node specification |

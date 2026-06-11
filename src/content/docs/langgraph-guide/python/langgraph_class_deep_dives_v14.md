---
title: "Class deep-dives Vol. 14 — Execution engine internals"
description: "Source-verified deep dives into StreamMessagesHandler/V2, PregelNode, ChannelRead, create_checkpoint/channels_from_checkpoint, map_input/map_output_values/map_output_updates, _TimedAttemptScope/_AttemptContext, apply_writes/LazyAtomicCounter, validate_graph, draw_graph/Edge/TriggerEdge, and ensure_message_ids/_is_message_dict — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 14"
  order: 45
---

# Class deep-dives Vol. 14 — Execution engine internals

Verified against **`langgraph==1.2.4`** installed at `/usr/local/lib/python3.11/dist-packages/langgraph/`.

Every section was written by reading the actual installed source. All signatures, field names, and behaviours are derived from the implementation, not from external documentation.

[→ Vol. 1–13 index at the bottom of this page](#vol-index)

---

## 1 · `StreamMessagesHandler` + `StreamMessagesHandlerV2` — the messages stream callback

**Module:** `langgraph.pregel._messages`  
**Not exported publicly** — used internally by Pregel when `stream_mode="messages"`.

When you call `graph.stream(..., stream_mode="messages")`, Pregel attaches one of these callback handlers to every node invocation. The handler intercepts LangChain callback events (`on_chat_model_start`, `on_llm_new_token`, `on_llm_end`, `on_chain_end`) and routes message chunks/objects onto the messages projection channel.

`StreamMessagesHandlerV2` is attached instead of v1 when the internal `CONFIG_KEY_STREAM_MESSAGES_V2` flag is set — it opts into the `stream_events(version="v3")` protocol by inheriting from `_V2StreamingCallbackHandler`, which reroutes `BaseChatModel.invoke` through `_stream_chat_model_events` (firing `on_stream_event`) rather than `_stream` (firing `on_llm_new_token`).

### Source signatures (1.2.4)

```python
class StreamMessagesHandler(BaseCallbackHandler, _StreamingCallbackHandler):
    run_inline = True  # runs in main thread to avoid ordering issues

    def __init__(
        self,
        stream: Callable[[StreamChunk], None],
        subgraphs: bool,
        *,
        parent_ns: tuple[str, ...] | None = None,
    ) -> None: ...

    # Key internal methods:
    def _emit(self, meta: Meta, message: BaseMessage, *, dedupe: bool = False) -> None: ...
    def on_chat_model_start(self, serialized, messages, *, run_id, tags, metadata, **kw): ...
    def on_llm_new_token(self, token, *, chunk, run_id, **kw): ...
    def on_llm_end(self, response, *, run_id, **kw): ...
    def on_chain_end(self, response, *, run_id, **kw): ...


class StreamMessagesHandlerV2(StreamMessagesHandler, _V2StreamingCallbackHandler):
    """V2 variant — drives via on_stream_event instead of on_llm_new_token.
    Used when stream_mode="messages" is paired with stream_events() v3."""

    def on_llm_new_token(self, *args, **kwargs) -> Any:
        """Intentional no-op — v2 uses on_stream_event."""

    def on_stream_event(self, event, *, run_id, tags, **kwargs) -> Any:
        """Forward content-block events from stream_events(v3) to messages channel."""
        ...
```

### Key differences between v1 and v2

| Behaviour | v1 `StreamMessagesHandler` | v2 `StreamMessagesHandlerV2` |
|---|---|---|
| Streaming driver | `on_llm_new_token` (fires per token) | `on_stream_event` (fires `message-start`, `content-block-*`, `message-finish`) |
| `ToolMessage` on `on_chain_end` | Emitted | Skipped (belongs on tools channel in v3) |
| Message deduplication | Via `self.seen` set of message IDs | Same, plus `_streamed_run_ids` guards against double-emit on `on_llm_end` |
| Activation | Default for `stream_mode="messages"` | Set when `CONFIG_KEY_STREAM_MESSAGES_V2` is in config |

### Example 1: Consuming messages stream with metadata

The tuple emitted on the stream is `(namespace_tuple, "messages", (message_or_chunk, metadata))`.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State) -> dict:
    # Simulate a model response (real usage: call an LLM here)
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content="Hello from node!", id="ai-1")]}

graph = (
    StateGraph(State)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)

for event in graph.stream(
    {"messages": [HumanMessage(content="Hi")]},
    stream_mode="messages",
):
    # event is (message_or_chunk, metadata_dict)
    msg, meta = event
    print(f"node={meta.get('langgraph_node')} | {type(msg).__name__}: {msg.content!r}")
```

### Example 2: Understanding `parent_ns` — subgraph message isolation

```python
# When subgraphs=False (default), the handler filters out messages
# from nested subgraphs unless they were explicitly invoked with
# stream_mode="messages". parent_ns tracks where the handler was
# originally created so it can pass-through intentional subgraph streams.
#
# Internally, the namespace check in on_chat_model_start:
#   ns = tuple(metadata["langgraph_checkpoint_ns"].split(NS_SEP))[:-1]
#   if not self.subgraphs and len(ns) > 0 and ns != self.parent_ns:
#       return  # skip this model call — it's in a nested subgraph

# To receive messages from ALL subgraph levels, use subgraphs=True:
for event in graph.stream(
    {"messages": [HumanMessage("hi")]},
    stream_mode="messages",
    subgraphs=True,   # propagate across subgraph boundaries
):
    msg, meta = event
    ns = meta.get("langgraph_checkpoint_ns", "")
    print(f"ns={ns!r} | {type(msg).__name__}")
```

### Example 3: Tag-based filtering — `TAG_NOSTREAM`

The handler respects the `TAG_NOSTREAM` tag. A model call with this tag will not be forwarded to the messages stream, which is useful for "silent" reasoning steps.

```python
from langgraph.constants import TAG_NOSTREAM
from langchain_core.language_models.fake import FakeListChatModel

# Any model invocation tagged with TAG_NOSTREAM is filtered
silent_llm = FakeListChatModel(responses=["internal reasoning"]).with_config(
    tags=[TAG_NOSTREAM]
)
# When called inside a node, its tokens/messages will not appear on stream_mode="messages"
```

---

## 2 · `PregelNode` — the internal compiled-node descriptor

**Module:** `langgraph.pregel._read`  
**Not exported publicly** — constructed by `StateGraph.compile()` for each node.

`PregelNode` is the internal descriptor that backs each node in a compiled `StateGraph`. It is NOT a runnable itself — it acts as a container for the configuration needed to build a `PregelExecutableTask` at runtime: which channels to read, which channels trigger the node, how the input should be mapped, what writers flush the output, and what retry/cache/timeout policies apply.

### Source signature (1.2.4)

```python
class PregelNode:
    channels: str | list[str]
    """Channel(s) whose values are passed as input to `bound`."""

    triggers: list[str]
    """If any of these channels is written to, this node is scheduled next step."""

    mapper: Callable[[Any], Any] | None
    """Optional transform applied to the raw channel read before calling `bound`."""

    writers: list[Runnable]
    """Runnables that flush `bound`'s output to channels (includes ChannelWrite)."""

    bound: Runnable[Any, Any]
    """The node's main logic (your function wrapped in RunnableCallable)."""

    retry_policy: Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    timeout: TimeoutPolicy | None

    tags: Sequence[str] | None
    metadata: Mapping[str, Any] | None

    is_error_handler: bool
    error_handler_node: str | None

    subgraphs: Sequence[PregelProtocol]
    """Nested Pregel graphs discovered by `find_subgraph_pregel`."""

    def __init__(
        self,
        *,
        channels: str | list[str],
        triggers: Sequence[str],
        mapper: Callable[[Any], Any] | None = None,
        writers: list[Runnable] | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        bound: Runnable[Any, Any] | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        is_error_handler: bool = False,
        error_handler_node: str | None = None,
        subgraphs: Sequence[PregelProtocol] | None = None,
        timeout: float | timedelta | TimeoutPolicy | None = None,
    ) -> None: ...

    def copy(self, update: dict[str, Any]) -> PregelNode: ...
    def flat_writers(self) -> list[Runnable]: ...  # cached_property
    def node(self) -> Runnable: ...  # cached_property — bound | writers pipeline
```

### Example 1: Inspecting compiled node descriptors

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    count: Annotated[int, operator.add]

def node_a(state: State) -> dict:
    return {"count": 1}

def node_b(state: State) -> dict:
    return {"count": 2}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)

# Access internal nodes dict on the compiled graph
pregel_nodes = graph.nodes

for name, pnode in pregel_nodes.items():
    print(f"--- Node: {name!r}")
    print(f"  channels : {pnode.channels}")
    print(f"  triggers : {pnode.triggers}")
    print(f"  writers  : {[type(w).__name__ for w in pnode.writers]}")
    print(f"  retry    : {pnode.retry_policy}")
    print(f"  timeout  : {pnode.timeout}")
```

### Example 2: How `mapper` transforms node input

When a node is compiled with a `schema` or type coercion, a `mapper` callable is inserted. The raw channel dict is first passed through `mapper` before reaching your function.

```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class NodeInput(BaseModel):
    """Narrow input type for a node — only the fields it needs."""
    count: int

class FullState(BaseModel):
    count: int
    name: str

def typed_node(state: NodeInput) -> dict:
    print(f"  node sees: count={state.count}")
    return {"count": state.count + 1}

graph = (
    StateGraph(FullState)
    .add_node("typed", typed_node)
    .add_edge(START, "typed")
    .add_edge("typed", END)
    .compile()
)

# The compiled PregelNode has a mapper that coerces the dict to NodeInput
pnode = graph.nodes["typed"]
print(f"mapper present: {pnode.mapper is not None}")  # True when input_schema differs

result = graph.invoke({"count": 5, "name": "Alice"})
print(result)  # count=6, name='Alice'
```

### Example 3: `error_handler_node` — finding the error-handler wiring

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    x: int

def risky(state: S) -> dict:
    raise ValueError("oops")

def handle_error(state: S, exception: Exception) -> dict:
    print(f"Caught: {exception}")
    return {"x": -1}

graph = (
    StateGraph(S)
    .add_node("risky", risky, error_handler=handle_error)
    .add_edge(START, "risky")
    .add_edge("risky", END)
    .compile()
)

# PregelNode has is_error_handler=True on the injected handler node
for name, pnode in graph.nodes.items():
    print(f"{name}: is_error_handler={pnode.is_error_handler}, "
          f"error_handler_node={pnode.error_handler_node!r}")
```

---

## 3 · `ChannelRead` — the state-reading runnable

**Module:** `langgraph.pregel._read`  
**Not exported publicly** — created internally by `StateGraph.compile()`.

`ChannelRead` is a `RunnableCallable` that reads one or more channels from `CONFIG_KEY_READ` at runtime. Every `PregelNode` has at least one `ChannelRead` embedded in its execution pipeline — it is the mechanism by which your node function receives the current graph state.

The key insight is that `ChannelRead` offers a static method `do_read()` you can call imperatively from inside a node body to read the **current step's channel values** without waiting for the next node boundary.

### Source signature (1.2.4)

```python
class ChannelRead(RunnableCallable):
    channel: str | list[str]
    fresh: bool = False
    mapper: Callable[[Any], Any] | None = None

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

### Key parameters

| Parameter | Meaning |
|---|---|
| `channel: str` | Read a single channel; returns the value directly |
| `channel: list[str]` | Read multiple channels; returns a `dict` |
| `fresh=False` | Read the channel value as it was at the START of this step |
| `fresh=True` | Read the channel value **after** the current task's writes have been applied — sees writes from earlier sibling tasks in parallel steps |
| `mapper` | Optional transform applied to the read result before returning |

### Example 1: Imperatively reading state inside a node

```python
from langgraph.pregel._read import ChannelRead
from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    counter: int
    name: str

def inspect_node(state: State, config: RunnableConfig) -> dict:
    # Read a single channel imperatively
    current_counter = ChannelRead.do_read(config, select="counter")
    print(f"counter via do_read: {current_counter}")

    # Read multiple channels
    snapshot = ChannelRead.do_read(config, select=["counter", "name"])
    print(f"snapshot: {snapshot}")

    return {"counter": current_counter + 1}

graph = (
    StateGraph(State)
    .add_node("inspect", inspect_node)
    .add_edge(START, "inspect")
    .add_edge("inspect", END)
    .compile()
)

graph.invoke({"counter": 10, "name": "Alice"})
# counter via do_read: 10
# snapshot: {'counter': 10, 'name': 'Alice'}
```

### Example 2: `fresh=True` — reading sibling-task writes in a parallel step

When multiple nodes run in parallel (fan-out), `fresh=True` lets a later-scheduled task read the writes from a previously-completed parallel task in the same step.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._read import ChannelRead
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    total: Annotated[int, operator.add]

def writer(state: State) -> dict:
    return {"total": 5}

def reader(state: State, config: RunnableConfig) -> dict:
    # fresh=False → total as it was when this step started (before writer ran)
    stale = ChannelRead.do_read(config, select="total", fresh=False)
    # fresh=True  → total after any parallel sibling writes are applied
    current = ChannelRead.do_read(config, select="total", fresh=True)
    print(f"stale={stale}, current={current}")
    return {}

graph = (
    StateGraph(State)
    .add_node("writer", writer)
    .add_node("reader", reader)
    .add_edge(START, "writer")
    .add_edge(START, "reader")  # parallel with writer
    .add_edge("writer", END)
    .add_edge("reader", END)
    .compile()
)
graph.invoke({"total": 0})
# stale=0, current=5  (if writer ran first)
```

---

## 4 · `create_checkpoint` + `channels_from_checkpoint` + `copy_checkpoint` — checkpoint lifecycle

**Module:** `langgraph.pregel._checkpoint`  
**Not exported publicly** — called by the Pregel loop at the end of every step.

These three functions manage the full checkpoint lifecycle: building a new checkpoint from live channel state, hydrating a fresh set of channel objects from a stored checkpoint, and making a safe mutable copy of an existing checkpoint.

### Source signatures (1.2.4)

```python
def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    channels_to_snapshot: set[str] | None = None,
) -> Checkpoint:
    """Build a new Checkpoint from the previous one and live channel state.
    DeltaChannel entries are included only if they appear in channels_to_snapshot."""

def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
    *,
    saver: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """Hydrate channels from a stored Checkpoint.
    Walks ancestor checkpoints for DeltaChannels that need replay."""

def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    """Shallow-copy a Checkpoint, deep-copying nested dicts (versions_seen)."""
```

### `create_checkpoint` key behaviours

| Scenario | What happens |
|---|---|
| `channels=None` | Values are copied from the previous checkpoint unchanged (quick step with no channel writes) |
| `channels_to_snapshot` contains channel `k` | A `_DeltaSnapshot(ch.get())` blob is written to `channel_values[k]` |
| Channel not in `channels_to_snapshot` (DeltaChannel) | Channel is omitted from `channel_values` — ancestor walk reconstructs it |
| `get_next_version is not None` | A forced version bump is applied to snapshot channels that weren't written this step |

### Example 1: Building a checkpoint from channel state (illustrative)

```python
from langgraph.pregel._checkpoint import (
    create_checkpoint,
    channels_from_checkpoint,
    copy_checkpoint,
    empty_checkpoint,
)
from langgraph.channels.last_value import LastValue
from langgraph.channels.binop import BinaryOperatorAggregate
import operator

# 1. Start from an empty checkpoint
checkpoint = empty_checkpoint()
print(f"empty id: {checkpoint['id']!r}")
print(f"channel_values: {checkpoint['channel_values']}")

# 2. Build channel specs
specs = {
    "score": LastValue(int),
    "total": BinaryOperatorAggregate(int, operator.add),
}

# 3. Hydrate channels from the empty checkpoint
channels, managed = channels_from_checkpoint(specs, checkpoint)
print(f"channels: {list(channels.keys())}")  # ['score', 'total']

# 4. Simulate a write to 'score'
channels["score"].update([42])

# 5. Create a new checkpoint capturing the write
new_cp = create_checkpoint(
    checkpoint,
    channels,
    step=1,
    updated_channels={"score"},
)
print(f"channel_values: {new_cp['channel_values']}")
# {'score': 42}  (total omitted — no writes)

# 6. Safe mutation: copy before patching
safe = copy_checkpoint(new_cp)
safe["channel_values"]["score"] = 99
print(f"original: {new_cp['channel_values']['score']}")  # 42 (unchanged)
print(f"copy:     {safe['channel_values']['score']}")     # 99
```

### Example 2: `channels_from_checkpoint` round-trip

```python
from langgraph.pregel._checkpoint import (
    create_checkpoint,
    channels_from_checkpoint,
    empty_checkpoint,
)
from langgraph.channels.last_value import LastValue

specs = {"name": LastValue(str), "age": LastValue(int)}
checkpoint = empty_checkpoint()

# Write values into channels then checkpoint
channels, _ = channels_from_checkpoint(specs, checkpoint)
channels["name"].update(["Alice"])
channels["age"].update([30])

saved = create_checkpoint(checkpoint, channels, step=1)

# Now hydrate fresh channels from the saved checkpoint
fresh_channels, _ = channels_from_checkpoint(specs, saved)
print(fresh_channels["name"].get())  # Alice
print(fresh_channels["age"].get())   # 30
```

---

## 5 · `map_input` + `map_output_values` + `map_output_updates` — I/O mappers

**Module:** `langgraph.pregel._io`  
**Not exported publicly** — called by the Pregel loop to convert user inputs and task outputs to/from channel writes.

These three generator functions sit at the boundary between "user data" and "channel writes". They determine which channels receive which values and how node outputs are packaged into the `stream_mode="values"` and `stream_mode="updates"` projections.

### Source signatures (1.2.4)

```python
def map_input(
    input_channels: str | Sequence[str],
    chunk: dict[str, Any] | Any | None,
) -> Iterator[tuple[str, Any]]:
    """Yield (channel, value) write tuples from a user-supplied input chunk."""

def map_output_values(
    output_channels: str | Sequence[str],
    pending_writes: Literal[True] | Sequence[tuple[str, Any]],
    channels: Mapping[str, BaseChannel],
) -> Iterator[dict[str, Any] | Any]:
    """Yield one output dict per step where any output channel was written."""

def map_output_updates(
    output_channels: str | Sequence[str],
    tasks: list[tuple[PregelExecutableTask, Sequence[tuple[str, Any]]]],
    cached: bool = False,
) -> Iterator[dict[str, Any | dict[str, Any]]]:
    """Yield one dict of {node_name: writes} per step for stream_mode='updates'."""
```

### `map_input` rules

| `input_channels` type | Input chunk type | What happens |
|---|---|---|
| `str` | Any value | Yields `(input_channels, chunk)` directly |
| `Sequence[str]` | `dict` | Yields `(k, chunk[k])` for each `k` in both `chunk` and `input_channels` |
| `Sequence[str]` | Non-dict | Raises `TypeError` |
| Any | `None` | Yields nothing (empty input) |

### Example 1: Tracing `map_input` to understand single vs multi-channel inputs

```python
from langgraph.pregel._io import map_input

# Single-channel graph: input_channels = "__root__"
writes = list(map_input("__root__", "hello"))
print(writes)  # [('__root__', 'hello')]

# Multi-channel graph: input_channels = ["messages", "context"]
writes = list(map_input(["messages", "context"], {"messages": ["hi"], "context": "doc"}))
print(writes)  # [('messages', ['hi']), ('context', 'doc')]

# Extra keys in dict are silently dropped (with a logger.warning)
writes = list(map_input(["messages"], {"messages": ["hi"], "ignored_key": "x"}))
print(writes)  # [('messages', ['hi'])]
```

### Example 2: `map_output_values` — understanding `stream_mode="values"` projection

```python
from langgraph.pregel._io import map_output_values
from langgraph.channels.last_value import LastValue

# Build two channels with known values
ch_a = LastValue(str)
ch_a.update(["hello"])
ch_b = LastValue(int)
ch_b.update([42])
channels = {"a": ch_a, "b": ch_b}

# pending_writes=True means "emit regardless" (used at end of graph run)
outputs = list(map_output_values(["a", "b"], True, channels))
print(outputs)  # [{'a': 'hello', 'b': 42}]

# Only emit when a relevant channel was written this step
pending = [("a", "world")]  # only "a" was written
outputs = list(map_output_values(["a", "b"], pending, channels))
print(outputs)  # [{'a': 'hello', 'b': 42}]  — emits because "a" is in pending

pending = [("unrelated", "x")]
outputs = list(map_output_values(["a", "b"], pending, channels))
print(outputs)  # []  — no output channels written → silent step
```

### Example 3: `map_output_updates` — understanding `stream_mode="updates"` projection

```python
from langgraph.pregel._io import map_output_updates
from langgraph.types import PregelExecutableTask
from langchain_core.runnables import RunnableConfig

# Simulate the internal structure: a list of (task, writes) tuples
# In production Pregel creates PregelExecutableTask; here we use a minimal stub
from unittest.mock import MagicMock

def make_task(name: str):
    t = MagicMock()
    t.name = name
    t.config = {"tags": []}
    t.writes = []
    return t

task_a = make_task("node_a")
task_b = make_task("node_b")

output_channels = "messages"  # single output channel

# writes: each write is (channel_name, value)
writes_a = [("messages", ["response from A"])]
writes_b = [("messages", ["response from B"])]

updates = list(map_output_updates(output_channels, [
    (task_a, writes_a),
    (task_b, writes_b),
]))
# One update dict per step: {node_name: write_value}
print(updates)
# [{'node_a': ['response from A'], 'node_b': ['response from B']}]
```

---

## 6 · `_TimedAttemptScope` + `_AttemptContext` + `_AttemptEvent` — retry and timeout execution engine

**Module:** `langgraph.pregel._retry`  
**Not exported publicly** — used by the async Pregel loop when a node has a `TimeoutPolicy`.

When a node runs with `timeout=TimeoutPolicy(idle_timeout=30.0, refresh_on="auto")`, `_TimedAttemptScope` wraps the node's config. Every write to a channel, stream event, or LangChain callback event emitted by the node becomes a "progress heartbeat" that resets the idle timer. If the timer expires, the node is cancelled.

`_AttemptContext` and `_AttemptEvent` are the immutable metadata objects shared with the internal observer contract (`langgraph-server` reads these via the same module path).

### Source signatures (1.2.4)

```python
@dataclass(frozen=True, slots=True)
class _ResolvedTimeout:
    run_timeout_secs: float | None
    idle_timeout_secs: float | None
    refresh_on: Literal["auto", "heartbeat"] | None


class _AttemptContext(NamedTuple):
    """Immutable per-attempt metadata shared across start/progress/finish events."""
    task_id: str
    task_name: str
    attempt: int
    run_id: str | None
    thread_id: str | None
    checkpoint_ns: str | None
    started_at: datetime
    run_timeout_secs: float | None
    idle_timeout_secs: float | None
    refresh_on: Literal["auto", "heartbeat"] | None


@dataclass(frozen=True, slots=True)
class _AttemptEvent:
    """One lifecycle event for a timed attempt."""
    context: _AttemptContext
    event: Literal["start", "progress", "finish"]
    progress_at: datetime | None = None
    finished_at: datetime | None = None
    status: Literal["success", "error"] | None = None
    error_type: str | None = None
    error_message: str | None = None


class _TimedAttemptScope:
    """Guarded-config window for timed attempts.

    Wraps every channel write, stream send, call dispatch, and stream writer
    callback so they count as progress. Close() serializes guarded writes
    so cancelled background tasks cannot persist writes past the timeout boundary.
    """

    def wrap_config(self, config: RunnableConfig) -> RunnableConfig: ...
    def touch(self) -> None: ...  # manual heartbeat signal
    def close(self) -> None: ...  # called after the task finishes or is cancelled
```

### `refresh_on` behaviour

| `refresh_on` | What resets the idle timer |
|---|---|
| `"auto"` (default) | Any channel write, `StreamWriter` call, LangChain callback event, or `@task` scheduling |
| `"heartbeat"` | Only explicit `runtime.heartbeat()` calls — use for work that emits no events |
| `None` | Idle timer is disabled; only `run_timeout` applies |

### Example 1: Setting up per-node idle timeout

```python
from datetime import timedelta
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def slow_node(state: State) -> dict:
    import asyncio
    # This node emits no events for 2s — it will be cancelled
    # if idle_timeout < 2s
    await asyncio.sleep(2)
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content="done")]}

graph = (
    StateGraph(State)
    .add_node(
        "slow",
        slow_node,
        # Timeout if idle for more than 1 second
        timeout=TimeoutPolicy(idle_timeout=timedelta(seconds=1), refresh_on="auto"),
    )
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)

# Running this raises NodeTimeoutError after ~1 second idle
import asyncio
from langgraph.errors import NodeTimeoutError
from langchain_core.messages import HumanMessage

async def run():
    try:
        await graph.ainvoke({"messages": [HumanMessage("go")]})
    except NodeTimeoutError as e:
        print(f"Timed out: {e}")

asyncio.run(run())
```

### Example 2: `refresh_on="heartbeat"` — manual progress signal

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy
from langgraph.runtime import get_runtime

async def batch_processor(state: dict) -> dict:
    """Processes 1000 items in batches — each batch is a manual heartbeat."""
    runtime = get_runtime()
    items = state.get("items", [])
    results = []
    for i in range(0, len(items), 50):
        batch = items[i:i + 50]
        results.extend([f"processed:{x}" for x in batch])
        # Signal liveness so idle timeout doesn't fire between batches
        runtime.heartbeat()
    return {"results": results}

# attach the node with heartbeat-mode timeout:
# timeout=TimeoutPolicy(idle_timeout=timedelta(seconds=30), refresh_on="heartbeat")
```

---

## 7 · `apply_writes` + `LazyAtomicCounter` — core execution mechanics

**Module:** `langgraph.pregel._algo`  
**Not exported publicly** — the engine behind every Pregel step.

`apply_writes` is the function that closes a step. It reads the pending writes from all tasks that ran in the current step, applies them to the live channel objects, bumps channel versions, and returns the set of updated channels that will be used to schedule the next step's tasks.

`LazyAtomicCounter` is a thread-safe monotonic counter that initialises lazily — the `itertools.count` object is only created on first call, protected by a module-level lock to prevent races during graph compilation.

### Source signatures (1.2.4)

```python
def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: GetNextVersion | None,
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
    """Apply writes from a step's tasks to checkpoint and channels.
    Returns the set of channels that were updated (used to schedule next tasks)."""


class LazyAtomicCounter:
    """Thread-safe monotonic counter initialised on first use.
    Used by visualisation (draw_graph) to generate stable, unique node IDs."""
    def __call__(self) -> int: ...
```

### `apply_writes` execution order

1. **Sort tasks** by `task_path_str(t.path[:3])` for deterministic write ordering.
2. **Update `versions_seen`**: record the version of each trigger channel seen by each task.
3. **Compute next version** from the current maximum channel version.
4. **Consume trigger channels**: after a task runs, its trigger channels are `.consume()`d so they don't re-trigger.
5. **Group writes by channel** and call `channel.update(values)` for each.
6. **Bump step for non-updated channels** — channels that `update(EMPTY_SEQ)` succeeds on are included in `updated_channels`.
7. **Finish step** — if the updated channels are disjoint from `trigger_to_nodes` (nothing will trigger next step), call `channel.finish()` on all channels to signal end-of-graph.

### Example 1: Walking through a single `apply_writes` call

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._checkpoint import empty_checkpoint, channels_from_checkpoint, create_checkpoint
from langgraph.pregel._algo import apply_writes
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.last_value import LastValue

# Build channels matching a simple state
specs = {
    "total": BinaryOperatorAggregate(int, operator.add),
    "name": LastValue(str),
}
checkpoint = empty_checkpoint()
channels, _ = channels_from_checkpoint(specs, checkpoint)

# Simulate a step: write total+=5 and name="Bob"
from langgraph.pregel._algo import PregelTaskWrites
from langgraph.constants import START

task_writes = [
    PregelTaskWrites(
        path=("my_node",),
        name="my_node",
        writes=[("total", 5), ("name", "Bob")],
        triggers=[],
    )
]

updated = apply_writes(
    checkpoint,
    channels,
    task_writes,
    get_next_version=None,
    trigger_to_nodes={},
)
print(f"updated channels: {updated}")  # {'total', 'name'}
print(f"total: {channels['total'].get()}")  # 5
print(f"name:  {channels['name'].get()}")   # Bob
```

### Example 2: `LazyAtomicCounter` — thread-safe lazy init

```python
from langgraph.pregel._algo import LazyAtomicCounter
import threading

counter = LazyAtomicCounter()

# First call initialises the internal itertools.count object
ids = []
def get_id():
    ids.append(counter())

threads = [threading.Thread(target=get_id) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# All 10 IDs are unique (no races)
print(len(set(ids)) == 10)  # True
print(sorted(ids))          # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (order may vary)
```

---

## 8 · `validate_graph` — compile-time graph structure validation

**Module:** `langgraph.pregel._validate`  
**Not exported publicly** — called automatically during `StateGraph.compile()`.

`validate_graph` performs a series of structural checks before a graph is compiled. Understanding what it validates helps you diagnose `ValueError` messages you see at compile time and explains which graph configurations are forbidden.

### Source signature (1.2.4)

```python
def validate_graph(
    nodes: Mapping[str, PregelNode],
    channels: dict[str, BaseChannel],
    managed: ManagedValueMapping,
    input_channels: str | Sequence[str],
    output_channels: str | Sequence[str],
    stream_channels: str | Sequence[str] | None,
    interrupt_after_nodes: All | Sequence[str],
    interrupt_before_nodes: All | Sequence[str],
) -> None: ...


def validate_keys(
    keys: Sequence[str],
    channels: Mapping[str, Any],
    managed: Mapping[str, Any] = {},
) -> None: ...
```

### Checks performed by `validate_graph`

| Check | Error raised |
|---|---|
| Channel or managed name is in `RESERVED` | `ValueError: Channel name 'X' is reserved` |
| Node name is in `RESERVED` | `ValueError: Node name 'X' is reserved` |
| Node reads a channel that doesn't exist | `ValueError: Node X reads channel 'Y' not in known channels` |
| No node subscribes to any input channel | `ValueError: Input channel X is not subscribed to by any node` |
| Output/stream channel not in known channels | `ValueError: Output channel 'X' not in known channels` |
| `interrupt_before/after_nodes` names unknown node | `ValueError: Node 'X' not found in graph` |

### Example 1: Triggering each validation error

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    x: int

# 1. Using a reserved channel/node name raises at compile time
# RESERVED names include: "__root__", "__start__", "__end__", "messages", etc.
# Example:
# builder = StateGraph(S)
# builder.add_node("__root__", lambda s: s)  # ValueError: reserved

# 2. interrupt_before with a node that doesn't exist
builder = StateGraph(S)
builder.add_node("real_node", lambda s: {"x": s["x"] + 1})
builder.add_edge(START, "real_node")
builder.add_edge("real_node", END)
try:
    builder.compile(interrupt_before=["nonexistent"])
except ValueError as e:
    print(f"Caught: {e}")
# ValueError: Node 'nonexistent' not found in graph

# 3. interrupt_after also validates
try:
    builder.compile(interrupt_after=["also_missing"])
except ValueError as e:
    print(f"Caught: {e}")
```

### Example 2: `validate_keys` — checking field names against known channels

`validate_keys` is used by `StateGraph` to verify that `input_fields`, `output_fields`, and similar lists reference channels that actually exist.

```python
from langgraph.pregel._validate import validate_keys
from langgraph.channels.last_value import LastValue

channels = {"x": LastValue(int), "y": LastValue(str)}

# Valid — all keys exist in channels
validate_keys(["x", "y"], channels)
print("OK")

# Invalid — "z" is not a known channel
try:
    validate_keys(["x", "z"], channels)
except ValueError as e:
    print(f"Caught: {e}")
# ValueError: Key 'z' not found in channels
```

---

## 9 · `draw_graph` + `Edge` + `TriggerEdge` — graph visualisation internals

**Module:** `langgraph.pregel._draw`  
**Not exported publicly** — called by `CompiledStateGraph.get_graph()`.

When you call `graph.get_graph()` or `graph.get_graph(xray=1)`, LangGraph internally calls `draw_graph(...)` which simulates a short execution of the Pregel loop to discover which edges are reachable. The result is a `langchain_core.runnables.graph.Graph` object that can be rendered to PNG, Mermaid, or ASCII art.

`Edge` and `TriggerEdge` are the internal named-tuple types used during the traversal to track discovered connections.

### Source signatures (1.2.4)

```python
class Edge(NamedTuple):
    source: str     # originating node name
    target: str     # destination node name
    conditional: bool   # True if this edge came from a conditional branch
    data: str | None    # branch label (Literal value) or None for unconditional

class TriggerEdge(NamedTuple):
    source: str
    conditional: bool
    data: str | None

def draw_graph(
    config: RunnableConfig,
    *,
    nodes: dict[str, PregelNode],
    specs: dict[str, BaseChannel | ManagedValueSpec],
    input_channels: str | Sequence[str],
    interrupt_after_nodes: All | Sequence[str],
    interrupt_before_nodes: All | Sequence[str],
    trigger_to_nodes: Mapping[str, Sequence[str]],
    checkpointer: Checkpointer,
    subgraphs: dict[str, Graph],
    limit: int = 250,
) -> Graph: ...
```

### Example 1: Getting the graph object and inspecting edges

```python
from langgraph.graph import StateGraph, START, END
from typing import Literal
from typing_extensions import TypedDict

class S(TypedDict):
    route: str

def router(state: S) -> Literal["path_a", "path_b"]:
    return "path_a" if state["route"] == "a" else "path_b"

def path_a(state: S) -> dict:
    return {}

def path_b(state: S) -> dict:
    return {}

graph = (
    StateGraph(S)
    .add_node("router", router)
    .add_node("path_a", path_a)
    .add_node("path_b", path_b)
    .add_edge(START, "router")
    .add_conditional_edges("router", router, ["path_a", "path_b"])
    .add_edge("path_a", END)
    .add_edge("path_b", END)
    .compile()
)

lg_graph = graph.get_graph()

print("Nodes:", list(lg_graph.nodes.keys()))

for edge in lg_graph.edges:
    print(f"  {edge.source} → {edge.target}"
          f"{'  (conditional)' if edge.conditional else ''}"
          f"{f'  label={edge.data!r}' if edge.data else ''}")
```

### Example 2: Rendering to Mermaid and PNG

```python
# Mermaid markdown (paste into mermaid.live)
mermaid = graph.get_graph().draw_mermaid()
print(mermaid)

# ASCII art (no dependencies)
ascii_art = graph.get_graph().draw_ascii()
print(ascii_art)

# PNG (requires `pip install pygraphviz` or `pip install grandalf`)
# graph.get_graph().draw_png("my_graph.png")
```

### Example 3: `xray` — expanding subgraph edges

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class Inner(TypedDict):
    val: int

class Outer(TypedDict):
    val: int

inner = (
    StateGraph(Inner)
    .add_node("inner_node", lambda s: {"val": s["val"] * 2})
    .add_edge(START, "inner_node")
    .add_edge("inner_node", END)
    .compile()
)

outer = (
    StateGraph(Outer)
    .add_node("outer_node", inner)   # inner graph as a node
    .add_edge(START, "outer_node")
    .add_edge("outer_node", END)
    .compile()
)

# xray=0 (default): outer_node is opaque
shallow = outer.get_graph(xray=0)
print("shallow nodes:", list(shallow.nodes.keys()))
# ['__start__', 'outer_node', '__end__']

# xray=1: expand one level of subgraphs
deep = outer.get_graph(xray=1)
print("deep nodes:", list(deep.nodes.keys()))
# ['__start__', 'outer_node:__start__', 'outer_node:inner_node', ..., '__end__']
```

---

## 10 · `ensure_message_ids` + `_is_message_dict` — message ID stability guarantee

**Module:** `langgraph.pregel._messages`  
**Not exported publicly** — called by `Pregel.put_writes()` before writes are submitted to the checkpointer.

Every `BaseMessage` stored in a LangGraph checkpoint must have a stable, non-`None` ID. Without stability, the same message appears with a *different* UUID on every `get_state()` replay — causing phantom duplicates in LangSmith traces and breaking the deduplication logic in `add_messages`.

`ensure_message_ids` is called synchronously on all pending writes before the background checkpoint thread is submitted, so the bytes serialised by the checkpointer always carry the stamped IDs.

`_is_message_dict` identifies dicts that represent messages (via their `"role"` or `"type"` key) so they can be stamped before `convert_to_messages` turns them into `BaseMessage` objects.

### Source signatures (1.2.4)

```python
_MESSAGE_ROLES: frozenset[str] = frozenset(
    {"user", "human", "assistant", "ai", "tool", "system", "function"}
)
_MESSAGE_TYPES: frozenset[str] = frozenset(
    {"human", "ai", "tool", "system", "function", "remove"}
)

def _is_message_dict(item: dict) -> bool:
    return item.get("role") in _MESSAGE_ROLES or item.get("type") in _MESSAGE_TYPES


def ensure_message_ids(value: Any) -> None:
    """Stamp stable UUIDs on message-like values before checkpointing.

    Handles three input shapes:
    - BaseMessage → assigns UUID if id is None.
    - Message-like dict (known role/type) → stamps 'id' in-place.
    - list of the above → converts dict items to typed BaseMessages and stamps.

    Mutates synchronously — safe because this runs before the background
    checkpoint thread is submitted.
    """
```

### Why this matters: the `add_messages` deduplication invariant

`add_messages` merges two lists by matching on `message.id`. If a message is stored without an ID, it gets a new UUID on every replay — creating duplicate "new" messages in the list on each `get_state()` call.

### Example 1: Observing ID stamping on raw dicts

```python
from langgraph.pregel._messages import ensure_message_ids, _is_message_dict

# Test _is_message_dict
print(_is_message_dict({"role": "user", "content": "hi"}))    # True (OpenAI style)
print(_is_message_dict({"type": "human", "content": "hi"}))   # True (LangChain style)
print(_is_message_dict({"name": "tool", "content": "hi"}))    # False — no role/type

# Test ensure_message_ids on a list of mixed types
from langchain_core.messages import HumanMessage, AIMessage

msgs = [
    {"role": "user", "content": "Hello"},        # dict — will get stamped
    HumanMessage(content="World", id=None),       # BaseMessage with no ID
    AIMessage(content="Hi", id="existing-id"),   # already has ID — unchanged
]

ensure_message_ids(msgs)

print(msgs[0].get("id"))        # now has a UUID (str)
print(msgs[1].id)               # now has a UUID (str)
print(msgs[2].id)               # still "existing-id"
```

### Example 2: Demonstrating ID stability across checkpoint round-trips

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]

def echo(state: State) -> dict:
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content="echo")]}

saver = MemorySaver()
graph = (
    StateGraph(State)
    .add_node("echo", echo)
    .add_edge(START, "echo")
    .add_edge("echo", END)
    .compile(checkpointer=saver)
)

config = {"configurable": {"thread_id": "test-stability"}}
graph.invoke({"messages": [HumanMessage("hi")]}, config)

# Get state twice — IDs must be identical
snap1 = graph.get_state(config)
snap2 = graph.get_state(config)

ids1 = [m.id for m in snap1.values["messages"]]
ids2 = [m.id for m in snap2.values["messages"]]
print(f"IDs stable: {ids1 == ids2}")  # True ← ensure_message_ids guarantee
print(f"Message count: {len(ids1)}")  # 2 (human + ai)
```

### Example 3: `_is_message_dict` as a guard in custom reducers

```python
from langgraph.pregel._messages import _is_message_dict
from langchain_core.messages import convert_to_messages

def safe_merge_messages(left: list, right: list) -> list:
    """A custom reducer that converts dicts before merging."""
    def normalise(items: list) -> list:
        result = []
        for item in items:
            if isinstance(item, dict) and _is_message_dict(item):
                result.extend(convert_to_messages([item]))
            else:
                result.append(item)
        return result

    merged = normalise(left) + normalise(right)
    # deduplicate by ID
    seen = {}
    for msg in merged:
        if hasattr(msg, "id") and msg.id:
            seen[msg.id] = msg
        else:
            seen[id(msg)] = msg
    return list(seen.values())


# Usage in state
# class State(TypedDict):
#     messages: Annotated[list, safe_merge_messages]
```

---

## Vol. index {#vol-index}

| Vol. | Classes covered |
|---|---|
| [1](./langgraph_class_deep_dives/) | StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore |
| [2](./langgraph_class_deep_dives_v2/) | RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl |
| [3](./langgraph_class_deep_dives_v3/) | interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig |
| [4](./langgraph_class_deep_dives_v4/) | set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node |
| [5](./langgraph_class_deep_dives_v5/) | RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt protocol |
| [6](./langgraph_class_deep_dives_v6/) | GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message, ProtocolEvent |
| [7](./langgraph_class_deep_dives_v7/) | PregelProtocol/StreamProtocol, BackgroundExecutor, AsyncBatchedBaseStore, get_text_at_path/tokenize_path, SerdeEvent, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec |
| [8](./langgraph_class_deep_dives_v8/) | ExecutionInfo/Runtime.heartbeat, ServerInfo, ReplayState, StreamMux, Call, ChannelWrite/ChannelWriteEntry, PregelRunner/FuturesDict, WritesProtocol, SyncPregelLoop/AsyncPregelLoop, DuplexStream |
| [9](./langgraph_class_deep_dives_v9/) | ToolCallRequest, Send+timeout, create_react_agent hooks, RetryPolicy chained, CachePolicy key_func, InMemoryStore raw embeddings, Command.PARENT, TimeoutPolicy.coerce(), entrypoint multi-policy |
| [10](./langgraph_class_deep_dives_v10/) | Durability modes, NodeError/NodeCancelledError, TaskPayload/TaskResultPayload, CheckpointPayload/CheckpointTask, Item/SearchItem, GetOp/PutOp/SearchOp/ListNamespacesOp/MatchCondition, UIMessage/RemoveUIMessage, StreamPart variants |
| [11](./langgraph_class_deep_dives_v11/) | InjectedState, InjectedStore, MessagesState, Overwrite, ToolOutputMixin, CheckpointMetadata, CheckpointTuple, StateUpdate, PersistentDict, DeltaChannelHistory |
| [12](./langgraph_class_deep_dives_v12/) | RemoteGraph/RemoteException, PostgresSaver/ShallowPostgresSaver, AsyncPostgresSaver, PostgresStore/PoolConfig, AsyncPostgresStore, ANNIndexConfig/HNSWConfig/IVFFlatConfig/PostgresIndexConfig, GraphRunStream/SubgraphRunStream, ToolCallWithContext/ToolInvocationError, LifecyclePayload/LifecycleTransformer, MessagesTransformer/CheckpointsTransformer/TasksTransformer |
| [13](./langgraph_class_deep_dives_v13/) | LastValue/LastValueAfterFinish, BranchSpec, BaseCache, get_config/get_store runtime accessors, ManagedValue/ManagedValueSpec, default_cache_key/_freeze, map_debug_tasks/results/checkpoint, GraphLifecycleStatus/callback manager factories, BaseStore abstract contract, _get_channels/annotation-to-channel inference |
| **14** | **StreamMessagesHandler/V2, PregelNode, ChannelRead, create_checkpoint/channels_from_checkpoint/copy_checkpoint, map_input/map_output_values/map_output_updates, _TimedAttemptScope/_AttemptContext/_AttemptEvent, apply_writes/LazyAtomicCounter, validate_graph, draw_graph/Edge/TriggerEdge, ensure_message_ids/_is_message_dict** |

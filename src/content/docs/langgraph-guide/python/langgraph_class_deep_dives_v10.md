---
title: "Class deep-dives Vol. 10 — 10 more LangGraph types"
description: "Source-verified deep dives into Durability checkpoint modes, NodeError/NodeCancelledError, TaskPayload/TaskResultPayload, CheckpointPayload/CheckpointTask, Item/SearchItem, GetOp/PutOp/SearchOp/ListNamespacesOp/MatchCondition, UIMessage/RemoveUIMessage, GraphOutput v2, StreamPart variants, and PregelExecutableTask/CacheKey — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 10"
  order: 41
---

# Class deep-dives Vol. 10 — 10 more LangGraph types

Verified against **`langgraph==1.2.4`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly (`/usr/local/lib/python3.11/dist-packages/langgraph`). All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

[→ Vol. 4 covers set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, error taxonomy](./langgraph_class_deep_dives_v4/)

[→ Vol. 5 covers RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt protocol](./langgraph_class_deep_dives_v5/)

[→ Vol. 6 covers GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message/ui_message_reducer, ProtocolEvent](./langgraph_class_deep_dives_v6/)

[→ Vol. 7 covers PregelProtocol/StreamProtocol, BackgroundExecutor/AsyncBackgroundExecutor, AsyncBatchedBaseStore/_dedupe_ops, get_text_at_path/tokenize_path, SerdeEvent/register_serde_event_listener, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec/node Protocols, identifier/get_runnable_for_task](./langgraph_class_deep_dives_v7/)

[→ Vol. 8 covers ExecutionInfo/Runtime.heartbeat, ServerInfo/BaseUser, ReplayState, StreamMux, Call (functional API internals), ChannelWrite/ChannelWriteEntry, PregelRunner/FuturesDict, WritesProtocol/PregelTaskWrites, SyncPregelLoop/AsyncPregelLoop, DuplexStream](./langgraph_class_deep_dives_v8/)

[→ Vol. 9 covers ToolCallRequest/override(), Send+timeout, create_react_agent pre/post hooks, RetryPolicy chained policies, CachePolicy custom key_func, InMemoryStore raw embeddings, context_schema+Runtime.context, Command.PARENT cross-subgraph routing, TimeoutPolicy.coerce(), entrypoint multi-policy retry](./langgraph_class_deep_dives_v9/)

[→ Vol. 11 covers InjectedState, InjectedStore, MessagesState, Overwrite, ToolOutputMixin, CheckpointMetadata, CheckpointTuple, StateUpdate, PersistentDict, DeltaChannelHistory](./langgraph_class_deep_dives_v11/)

---

## 1 · `Durability` — checkpoint write timing

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import Durability`

`Durability` is a `Literal["sync", "async", "exit"]` type alias that controls **when** checkpoint writes are flushed to the checkpointer. It replaced the deprecated `checkpoint_during: bool` parameter in langgraph 1.x.

### Source (1.2.4)

```python
Durability = Literal["sync", "async", "exit"]
"""Durability mode for the graph execution.

- 'sync':  Changes are persisted synchronously before the next step starts.
- 'async': Changes are persisted asynchronously while the next step executes.
- 'exit':  Changes are persisted only when the graph exits.
"""
```

The mode is accepted by `stream()`, `astream()`, `invoke()`, and `ainvoke()` on any compiled graph. The default when not specified is `"async"` (read from config or hardcoded fallback in `pregel/main.py`):

```python
# pregel/main.py – _defaults()
if durability is None:
    durability = config.get(CONF, {}).get(CONFIG_KEY_DURABILITY, "async")
```

### Trade-offs

| Mode | When checkpoint is written | Use case |
|------|---------------------------|----------|
| `"sync"` | Before the next step begins | Safest: zero data loss on crash between steps |
| `"async"` | Concurrently with the next step (default) | Balanced: hides checkpoint latency behind compute |
| `"exit"` | Only when the graph exits normally | Fastest: good for short, non-resumable workflows |

### Example 1: Explicit durability per invocation

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

class State(TypedDict):
    counter: int

def increment(state: State) -> State:
    return {"counter": state["counter"] + 1}

graph = (
    StateGraph(State)
    .add_node(increment)
    .add_edge(START, "increment")
    .add_edge("increment", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "demo"}}

# "sync" — checkpoint written before the next step (safest)
result = graph.invoke({"counter": 0}, config, durability="sync")
print(result)  # {'counter': 1}

# "exit" — checkpoint only written when the run completes (fastest)
result = graph.invoke({"counter": 0}, config, durability="exit")
print(result)  # {'counter': 1}
```

### Example 2: Durability via config key

You can bake the durability mode into a reusable config so callers don't need to remember to pass it:

```python
from langgraph._internal._constants import CONFIG_KEY_DURABILITY, CONF

# Embed durability into a base config
safe_config = {
    "configurable": {
        "thread_id": "safe-thread",
        CONFIG_KEY_DURABILITY: "sync",
    }
}

result = graph.invoke({"counter": 0}, safe_config)
print(result)  # {'counter': 1}
```

### Example 3: Streaming with durability

The `durability` parameter is equally available on `stream()`:

```python
for chunk in graph.stream(
    {"counter": 0},
    config,
    stream_mode="updates",
    durability="exit",       # persist only on exit — lower overhead
):
    print(chunk)
# {'increment': {'counter': 1}}
```

---

## 2 · `NodeError` and `NodeCancelledError`

**Module:** `langgraph.errors`  
**Exported as:** `from langgraph.errors import NodeError, NodeCancelledError`

`NodeError` is a frozen dataclass injected into **error-handler nodes** (registered via `add_node(..., error_handler=...)`). It carries the name of the failed node and the original exception, giving the handler full context to decide how to recover.

`NodeCancelledError` wraps a user-raised `asyncio.CancelledError` so it surfaces as an ordinary node failure rather than a silent teardown.

### Source (1.2.4)

```python
@dataclass(frozen=True, slots=True)
class NodeError:
    node: str           # Name of the node whose execution failed
    error: BaseException  # The exception raised by the failed node


class NodeCancelledError(Exception):
    """Raised when a node body raises asyncio.CancelledError itself."""
    node: str

    def __init__(self, node: str, message: str | None = None) -> None:
        super().__init__(message or f"Node {node!r} raised asyncio.CancelledError")
        self.node = node
```

### Example 1: Per-node error recovery with `NodeError`

```python
from typing import TypedDict

from langgraph.constants import END, START
from langgraph.errors import NodeError
from langgraph.graph import StateGraph
from langgraph.types import Command

class State(TypedDict):
    payload: str
    error_info: str

def flaky_api_call(state: State) -> State:
    if state["payload"] == "bad":
        raise ValueError("Upstream API rejected the request")
    return {"payload": "processed", "error_info": ""}

def api_error_handler(state: State, error: NodeError) -> Command:
    # NodeError.node  → the name of the failed node ("flaky_api_call")
    # NodeError.error → the original exception (ValueError)
    msg = f"[{error.node}] {type(error.error).__name__}: {error.error}"
    return Command(update={"error_info": msg})

graph = (
    StateGraph(State)
    .add_node("flaky_api_call", flaky_api_call, error_handler=api_error_handler)
    .add_edge(START, "flaky_api_call")
    .add_edge("flaky_api_call", END)
    .compile()
)

result = graph.invoke({"payload": "bad", "error_info": ""})
print(result)
# {'payload': 'bad', 'error_info': '[flaky_api_call] ValueError: Upstream API rejected the request'}
```

### Example 2: Distinguishing error types in a shared handler

```python
import httpx

def network_error_handler(state: State, error: NodeError) -> Command:
    match type(error.error):
        case httpx.TimeoutException:
            return Command(update={"error_info": f"{error.node}: timeout, will retry"})
        case httpx.HTTPStatusError:
            code = error.error.response.status_code
            return Command(update={"error_info": f"{error.node}: HTTP {code}"})
        case _:
            return Command(update={"error_info": f"{error.node}: unexpected — {error.error}"})
```

### Example 3: Understanding `NodeCancelledError`

`NodeCancelledError` is raised **automatically** by the retry layer when a node's own body raises `asyncio.CancelledError`. You will encounter it in retry policies and error handlers when debugging async workflows:

```python
import asyncio
from langgraph.errors import NodeCancelledError

async def fragile_node(state: State) -> State:
    # This simulates user code that raises CancelledError directly.
    # LangGraph converts it to NodeCancelledError so it flows through
    # the normal error path instead of being silently swallowed.
    raise asyncio.CancelledError("user triggered cancel")

def on_cancel(state: State, error: NodeError) -> Command:
    # error.error is a NodeCancelledError here, not asyncio.CancelledError
    return Command(update={"error_info": f"cancelled: {error.node}"})
```

---

## 3 · `TaskPayload` and `TaskResultPayload`

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import TaskPayload, TaskResultPayload`

These are the two `TypedDict` types emitted by `stream_mode="tasks"`. Every node execution generates a **start** event (`TaskPayload`) followed by a **result** event (`TaskResultPayload`). They share the same `id` so you can correlate them.

### Source (1.2.4)

```python
class TaskPayload(TypedDict):
    id: str            # Unique task ID (UUID)
    name: str          # Node name
    input: Any         # State passed into the node
    triggers: list[str]  # Channel writes that caused this task to run
    metadata: NotRequired[dict[str, Any]]  # langgraph_node, langgraph_step, etc.

class TaskResultPayload(TypedDict):
    id: str            # Same UUID as the matching TaskPayload
    name: str          # Node name
    error: str | None  # Stringified exception, or None on success
    interrupts: list[dict]  # Pending interrupt values
    result: dict[str, Any]  # Channel writes produced by this task
```

### Example 1: Consuming task events

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

class State(TypedDict):
    x: int

def add_ten(state: State) -> State:
    return {"x": state["x"] + 10}

def double(state: State) -> State:
    return {"x": state["x"] * 2}

graph = (
    StateGraph(State)
    .add_node(add_ten)
    .add_node(double)
    .add_edge(START, "add_ten")
    .add_edge("add_ten", "double")
    .add_edge("double", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "tasks-demo"}}

pending: dict[str, dict] = {}

for event in graph.stream({"x": 1}, config, stream_mode="tasks"):
    if "error" not in event:
        # TaskPayload — task is starting
        pending[event["id"]] = event
        print(f"START  {event['name']}  input={event['input']}  triggers={event['triggers']}")
    else:
        # TaskResultPayload — task finished
        start = pending.pop(event["id"])
        print(f"FINISH {event['name']}  result={event['result']}  error={event['error']}")

# START  add_ten  input={'x': 1}   triggers=['branch:to:add_ten']
# FINISH add_ten  result={'x': 11}  error=None
# START  double   input={'x': 11}  triggers=['branch:to:double']
# FINISH double   result={'x': 22}  error=None
```

### Example 2: Detecting node failures via `error` field

```python
from langgraph.errors import NodeError
from langgraph.types import Command

class State(TypedDict):
    value: str
    failed: bool

def risky(state: State) -> State:
    if state["value"] == "bad":
        raise RuntimeError("boom")
    return {}

def on_error(state: State, error: NodeError) -> Command:
    return Command(update={"failed": True})

graph = (
    StateGraph(State)
    .add_node("risky", risky, error_handler=on_error)
    .add_edge(START, "risky")
    .add_edge("risky", END)
    .compile()
)

for event in graph.stream({"value": "bad", "failed": False}, stream_mode="tasks"):
    if event.get("error"):
        print(f"Node '{event['name']}' failed: {event['error']}")
```

### Example 3: Correlating task events across subgraphs

```python
# When using subgraphs=True, each event gets a namespace tuple
for event in graph.stream(
    {"x": 0},
    config,
    stream_mode="tasks",
    subgraphs=True,
):
    namespace, payload = event
    print(f"[{' > '.join(namespace) or 'root'}] {payload.get('name', '?')}")
```

---

## 4 · `CheckpointPayload` and `CheckpointTask`

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import CheckpointPayload, CheckpointTask`

`stream_mode="checkpoints"` emits one `CheckpointPayload` per checkpoint written during a run. It gives you a complete snapshot of graph state, including which tasks were scheduled, their outcomes, and any pending interrupts — all keyed to a `RunnableConfig` you can use to resume the run later.

### Source (1.2.4)

```python
class CheckpointTask(TypedDict):
    id: str
    name: str
    error: NotRequired[str]    # present only when the task failed
    result: NotRequired[Any]   # present only when the task completed
    interrupts: NotRequired[list[dict]]  # present when interrupted/completed
    state: StateSnapshot | RunnableConfig | None

class CheckpointPayload(TypedDict, Generic[StateT]):
    config: RunnableConfig | None       # Config to resume/fetch this checkpoint
    metadata: CheckpointMetadata        # step, source, writes, etc.
    values: StateT                      # Full channel state at checkpoint time
    next: list[str]                     # Nodes scheduled for the next superstep
    parent_config: RunnableConfig | None  # Parent checkpoint config
    tasks: list[CheckpointTask]         # Tasks associated with this checkpoint
```

### Example 1: Inspecting checkpoints during a run

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

class State(TypedDict):
    step: int

def bump(state: State) -> State:
    return {"step": state["step"] + 1}

saver = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node(bump)
    .add_edge(START, "bump")
    .add_edge("bump", END)
    .compile(checkpointer=saver)
)

config = {"configurable": {"thread_id": "cp-inspect"}}

for cp in graph.stream({"step": 0}, config, stream_mode="checkpoints"):
    print(f"step={cp['metadata']['step']}  next={cp['next']}  values={cp['values']}")
    if cp["tasks"]:
        task = cp["tasks"][0]
        print(f"  task: name={task['name']}  result={task.get('result')}")

# step=0  next=['bump']  values={'step': 0}
#   task: name=__start__  result=None
# step=1  next=[]  values={'step': 1}
#   task: name=bump  result={'step': 1}
```

### Example 2: Resuming from a checkpoint config

```python
from langgraph.types import interrupt, Command

class ReviewState(TypedDict):
    data: str
    approved: bool

def review_node(state: ReviewState) -> ReviewState:
    decision = interrupt({"prompt": "Approve?", "data": state["data"]})
    return {"approved": decision}

review_graph = (
    StateGraph(ReviewState)
    .add_node("review_node", review_node)
    .add_edge(START, "review_node")
    .add_edge("review_node", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "review-1"}}
interrupted_cp = None

for cp in review_graph.stream(
    {"data": "contract.pdf", "approved": False},
    config,
    stream_mode="checkpoints",
):
    if cp["next"] == []:          # Final checkpoint after interrupt
        interrupted_cp = cp
        break

print("Interrupted at:", interrupted_cp["config"]["configurable"]["checkpoint_id"])

# Resume using the checkpoint config
review_graph.invoke(
    Command(resume=True),
    interrupted_cp["config"],
)
```

### Example 3: Accessing checkpoint metadata

```python
# CheckpointMetadata contains step number, source, and pending writes
for cp in graph.stream({"step": 0}, config, stream_mode="checkpoints"):
    meta = cp["metadata"]
    print(f"  source={meta.get('source')}  step={meta.get('step')}  writes={meta.get('writes')}")
```

---

## 5 · `Item` and `SearchItem`

**Module:** `langgraph.store.base`  
**Exported as:** `from langgraph.store.base import Item, SearchItem`

`Item` is the result type returned by `store.get()` and `store.put()` / `store.batch([GetOp(...)])`. `SearchItem` extends `Item` with an optional `score` float for semantic similarity search results.

### Source (1.2.4)

```python
class Item:
    __slots__ = ("value", "key", "namespace", "created_at", "updated_at")

    def __init__(self, *, value: dict[str, Any], key: str,
                 namespace: tuple[str, ...], created_at: datetime,
                 updated_at: datetime): ...

    def dict(self) -> dict: ...   # serialisable representation


class SearchItem(Item):
    __slots__ = ("score",)

    score: float | None   # cosine similarity (0.0–1.0), None if not ranked
```

`created_at` and `updated_at` are always timezone-aware `datetime` objects. If the store received ISO-format strings (e.g. from JSON deserialisation), they are converted via `datetime.fromisoformat()` automatically.

### Example 1: Reading items and inspecting timestamps

```python
from datetime import timezone

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(("projects", "acme"), "config", {"plan": "enterprise", "seats": 50})

item = store.get(("projects", "acme"), "config")
assert item is not None

print(item.namespace)    # ('projects', 'acme')
print(item.key)          # 'config'
print(item.value)        # {'plan': 'enterprise', 'seats': 50}
print(item.created_at)   # datetime(2026, ..., tzinfo=timezone.utc)
print(item.updated_at)   # same as created_at on first write

# Serialise the whole thing
print(item.dict())
# {'namespace': ['projects', 'acme'], 'key': 'config', 'value': {...},
#  'created_at': '2026-...', 'updated_at': '2026-...'}
```

### Example 2: Semantic search returning `SearchItem` with scores

```python
from langgraph.store.memory import InMemoryStore

# InMemoryStore with an in-process embedding function
store = InMemoryStore(
    index={
        "dims": 4,
        "embed": lambda texts: [[float(i) for i in range(4)] for _ in texts],
    }
)

store.put(("kb",), "doc1", {"text": "LangGraph is a graph-based agent framework"})
store.put(("kb",), "doc2", {"text": "Python is a general-purpose programming language"})
store.put(("kb",), "doc3", {"text": "Agents can call tools and maintain state"})

results = store.search(("kb",), query="agent frameworks")

for item in results:
    # item is a SearchItem — it has everything Item has, plus .score
    print(f"[{item.score:.4f}] {item.key}: {item.value['text']}")
```

### Example 3: Guarding against `None` items

```python
from langgraph.store.base import Item

def load_user_profile(store: InMemoryStore, user_id: str) -> dict | None:
    item: Item | None = store.get(("users",), user_id)
    if item is None:
        return None
    return item.value

# `store.get()` returns None when the key does not exist.
# Always check before accessing .value to avoid AttributeError.
profile = load_user_profile(store, "unknown-user")
print(profile)  # None
```

---

## 6 · `GetOp`, `PutOp`, `SearchOp`, `ListNamespacesOp` and `MatchCondition`

**Module:** `langgraph.store.base`  
**Exported as:** `from langgraph.store.base import GetOp, PutOp, SearchOp, ListNamespacesOp, MatchCondition`

`BaseStore.batch(ops)` accepts any mix of these `NamedTuple` operation types and returns a parallel list of results. This is the **only** way to issue multiple store operations atomically in a single round-trip — essential when you need to read-before-write without a race or when building custom store adapters.

### Source (1.2.4)

```python
class GetOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    refresh_ttl: bool = True    # refresh item TTL on read (ignored if no TTL)

class PutOp(NamedTuple):
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any] | None  # None → delete the item
    index: list[str] | bool | None = None  # fields to embed for vector search
    ttl: float | None = None      # seconds until expiry (requires store TTL support)

class SearchOp(NamedTuple):
    namespace_prefix: tuple[str, ...]
    filter: dict[str, Any] | None = None  # key-equality filter on item values
    limit: int = 10
    offset: int = 0
    query: str | None = None      # natural-language query (semantic search)
    refresh_ttl: bool = True

class MatchCondition(NamedTuple):
    match_type: Literal["prefix", "suffix"]
    path: tuple[str | Literal["*"], ...]  # "*" is a wildcard segment

class ListNamespacesOp(NamedTuple):
    match_conditions: tuple[MatchCondition, ...] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0
```

`batch()` returns a `list[Result]` where each slot is: `Item | None` for `GetOp`, `None` for `PutOp`, `list[SearchItem]` for `SearchOp`, and `list[tuple[str, ...]]` for `ListNamespacesOp`.

### Example 1: Atomic read-then-write

```python
from langgraph.store.base import GetOp, PutOp
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(("counters",), "page_views", {"count": 42})

# Read and write in one batch — avoids two separate round-trips
get_result, _ = store.batch([
    GetOp(namespace=("counters",), key="page_views"),
    PutOp(namespace=("audit",), key="read_log", value={"event": "page_view_read"}),
])

current = get_result.value["count"] if get_result else 0

store.batch([
    PutOp(namespace=("counters",), key="page_views", value={"count": current + 1}),
])
print(store.get(("counters",), "page_views").value)  # {'count': 43}
```

### Example 2: Filtered search with `SearchOp`

```python
from langgraph.store.base import SearchOp

store = InMemoryStore()
store.put(("docs",), "r1", {"type": "report", "status": "active", "title": "Q1"})
store.put(("docs",), "r2", {"type": "report", "status": "archived", "title": "Q4"})
store.put(("docs",), "n1", {"type": "note", "status": "active", "title": "TODO"})

(active_reports,) = store.batch([
    SearchOp(
        namespace_prefix=("docs",),
        filter={"type": "report", "status": "active"},
        limit=20,
    )
])

for item in active_reports:
    print(item.key, item.value["title"])  # r1 Q1
```

### Example 3: Discovering namespaces with `ListNamespacesOp` and `MatchCondition`

```python
from langgraph.store.base import ListNamespacesOp, MatchCondition

store = InMemoryStore()
for user in ("alice", "bob", "carol"):
    store.put(("users", user, "prefs"), "theme", {"value": "dark"})
    store.put(("users", user, "history"), "v1", {"items": []})

# List all namespaces that start with ("users",) and end with ("prefs",)
(namespaces,) = store.batch([
    ListNamespacesOp(
        match_conditions=(
            MatchCondition(match_type="prefix", path=("users",)),
            MatchCondition(match_type="suffix", path=("prefs",)),
        ),
    )
])

for ns in namespaces:
    print(ns)
# ('users', 'alice', 'prefs')
# ('users', 'bob', 'prefs')
# ('users', 'carol', 'prefs')
```

### Example 4: Deleting items with `PutOp(value=None)`

```python
# Setting value=None in a PutOp deletes the key
store.batch([
    PutOp(namespace=("docs",), key="r2", value=None),
])
assert store.get(("docs",), "r2") is None
```

---

## 7 · `UIMessage` and `RemoveUIMessage`

**Module:** `langgraph.graph.ui`  
**Exported as:** `from langgraph.graph.ui import UIMessage, RemoveUIMessage, AnyUIMessage`

`UIMessage` and `RemoveUIMessage` are the `TypedDict` types that underpin LangGraph's UI streaming protocol. `push_ui_message()` creates `UIMessage` instances; `delete_ui_message()` creates `RemoveUIMessage` instances; and `ui_message_reducer()` merges them into a running list using the shared `id` field.

### Source (1.2.4)

```python
class UIMessage(TypedDict):
    type: Literal["ui"]
    id: str                     # UUID, unique per rendered component instance
    name: str                   # UI component name (e.g. "ProgressBar", "Table")
    props: dict[str, Any]       # Component props passed to the renderer
    metadata: dict[str, Any]    # run_id, tags, merge flag, etc.

class RemoveUIMessage(TypedDict):
    type: Literal["remove-ui"]
    id: str                     # ID of the UIMessage to remove

AnyUIMessage = UIMessage | RemoveUIMessage
```

The `metadata["merge"]` flag (set by `push_ui_message(..., merge=True)`) instructs `ui_message_reducer()` to **shallow-merge** the incoming `props` with the existing message's `props` instead of replacing them wholesale. This is ideal for streaming incremental updates to a component.

### Example 1: Pushing and deleting UI messages from a node

```python
from typing import Annotated

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.ui import AnyUIMessage, push_ui_message, delete_ui_message, ui_message_reducer
from typing import TypedDict

class State(TypedDict):
    ui: Annotated[list[AnyUIMessage], ui_message_reducer]

def generate_report(state: State) -> State:
    # Push an initial "loading" spinner
    spinner = push_ui_message("Spinner", {"label": "Generating report…"})

    # … do work …

    # Replace it with the finished table
    delete_ui_message(spinner["id"])
    push_ui_message("ReportTable", {"rows": [{"col": "val"}]})
    return {}

graph = (
    StateGraph(State)
    .add_node(generate_report)
    .add_edge(START, "generate_report")
    .add_edge("generate_report", END)
    .compile()
)

result = graph.invoke({"ui": []})
print([m["name"] for m in result["ui"]])  # ['ReportTable']
```

### Example 2: Incremental prop merging with `merge=True`

```python
from langgraph.graph.ui import push_ui_message

def streaming_node(state: State) -> State:
    # Create the progress bar
    bar = push_ui_message("ProgressBar", {"pct": 0, "label": "Starting…"})

    for i in range(1, 4):
        # merge=True does a shallow merge of props — only `pct` changes
        push_ui_message(
            "ProgressBar",
            {"pct": i * 33},
            id=bar["id"],
            merge=True,
        )
    return {}
```

### Example 3: Reading UIMessage metadata fields

```python
from langgraph.graph.ui import UIMessage

def inspect_ui_message(msg: UIMessage) -> None:
    print("component:", msg["name"])
    print("id:       ", msg["id"])
    print("props:    ", msg["props"])
    # Standard metadata keys populated by push_ui_message:
    meta = msg["metadata"]
    print("run_id:   ", meta.get("run_id"))
    print("merge:    ", meta.get("merge", False))
    print("message_id:", meta.get("message_id"))  # linked AIMessage id, if any
```

### Example 4: Building a custom `ui_message_reducer`-aware state

```python
from typing import Annotated, TypedDict

from langgraph.graph.ui import AnyUIMessage, ui_message_reducer

class AppState(TypedDict):
    messages: list        # chat messages
    ui: Annotated[list[AnyUIMessage], ui_message_reducer]
    # ui_message_reducer handles add / remove / merge — never touch ui directly
```

---

## 8 · `GraphOutput` (`version="v2"`)

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import GraphOutput`

`GraphOutput` is a frozen dataclass returned by `invoke()` / `ainvoke()` when you pass `version="v2"`. It separates the **graph output value** (`.value`) from any **pending interrupts** (`.interrupts`) — avoiding the fragile `result["__interrupt__"]` dict-key pattern that is deprecated in v11.

### Source (1.2.4)

```python
@dataclass(frozen=True)
class GraphOutput(Generic[OutputT]):
    value: OutputT                         # Final channel values (dict / Pydantic / dataclass)
    interrupts: tuple[Interrupt, ...] = () # Interrupts pending at the time of return

    def __getitem__(self, key: str) -> Any: ...  # deprecated compat shim
    def __contains__(self, key: object) -> bool: ...  # deprecated compat shim
```

The `version="v2"` flag also changes the shape of `stream()` output — see Section 9.

### Example 1: Clean interrupt handling with `GraphOutput`

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command, GraphOutput, interrupt

class ApprovalState(TypedDict):
    doc: str
    approved: bool

def approval_node(state: ApprovalState) -> ApprovalState:
    decision = interrupt({"question": "Approve this document?", "doc": state["doc"]})
    return {"approved": decision}

graph = (
    StateGraph(ApprovalState)
    .add_node("approval_node", approval_node)
    .add_edge(START, "approval_node")
    .add_edge("approval_node", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "approval-1"}}

# First call — hits the interrupt
result: GraphOutput[ApprovalState] = graph.invoke(
    {"doc": "contract.pdf", "approved": False},
    config,
    version="v2",
)

print(type(result).__name__)     # GraphOutput
print(result.value)              # {'doc': 'contract.pdf', 'approved': False}
print(result.interrupts)         # (Interrupt(value={'question': '...', 'doc': '...'}, id='...'),)

# Resume — pass decision back via Command
final: GraphOutput[ApprovalState] = graph.invoke(
    Command(resume=True),
    config,
    version="v2",
)

print(final.value)       # {'doc': 'contract.pdf', 'approved': True}
print(final.interrupts)  # ()
```

### Example 2: `GraphOutput` when no interrupts occur

```python
from typing import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph

class State(TypedDict):
    n: int

def triple(state: State) -> State:
    return {"n": state["n"] * 3}

graph = (
    StateGraph(State)
    .add_node(triple)
    .add_edge(START, "triple")
    .add_edge("triple", END)
    .compile()
)

result = graph.invoke({"n": 7}, version="v2")

assert isinstance(result.value, dict)
assert result.value["n"] == 21
assert result.interrupts == ()
```

### Example 3: Iterating over multiple interrupts

```python
from langgraph.types import Interrupt

result: GraphOutput = graph.invoke(input, config, version="v2")

for interrupt_obj in result.interrupts:
    print(f"Interrupt id={interrupt_obj.id}")
    print(f"  payload: {interrupt_obj.value}")
    # Use interrupt_obj.id with Command(resume=..., resume_id=...) to
    # resolve specific interrupts when multiple are pending.
```

---

## 9 · `StreamPart` variants — typed v2 stream events

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import ValuesStreamPart, UpdatesStreamPart, TasksStreamPart, CheckpointStreamPart, MessagesStreamPart, CustomStreamPart, DebugStreamPart, StreamPart`

When you call `stream(..., version="v2")`, every emitted chunk is a typed `TypedDict` with a `type` discriminator key, an `ns` namespace tuple, and a `data` payload. You can exhaustively match on `type` instead of inferring structure from position.

### Source (1.2.4)

```python
class ValuesStreamPart(TypedDict, Generic[OutputT]):
    type: Literal["values"]
    ns: tuple[str, ...]     # () for root, ("node:<id>",) inside subgraphs
    data: OutputT           # Full state after each step

class UpdatesStreamPart(TypedDict):
    type: Literal["updates"]
    ns: tuple[str, ...]
    data: dict[str, Any]    # {node_name: updates_dict}

class TasksStreamPart(TypedDict):
    type: Literal["tasks"]
    ns: tuple[str, ...]
    data: TaskPayload | TaskResultPayload  # start or finish event

class CheckpointStreamPart(TypedDict, Generic[StateT]):
    type: Literal["checkpoints"]
    ns: tuple[str, ...]
    data: CheckpointPayload[StateT]

class MessagesStreamPart(TypedDict):
    type: Literal["messages"]
    ns: tuple[str, ...]
    data: tuple[BaseMessage, dict]   # (token message, metadata dict)

class CustomStreamPart(TypedDict):
    type: Literal["custom"]
    ns: tuple[str, ...]
    data: Any               # whatever StreamWriter wrote

class DebugStreamPart(TypedDict, Generic[StateT]):
    type: Literal["debug"]
    ns: tuple[str, ...]
    data: DebugPayload[StateT]

StreamPart = ValuesStreamPart | UpdatesStreamPart | MessagesStreamPart \
           | CustomStreamPart | CheckpointStreamPart | TasksStreamPart \
           | DebugStreamPart
```

### Example 1: Exhaustive `match` dispatch on `type`

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

class State(TypedDict):
    x: int

def step(state: State) -> State:
    return {"x": state["x"] + 1}

graph = (
    StateGraph(State)
    .add_node(step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "v2-stream"}}

for part in graph.stream(
    {"x": 0},
    config,
    stream_mode=["values", "updates", "tasks"],
    version="v2",
):
    match part["type"]:
        case "values":
            print("STATE   →", part["data"])
        case "updates":
            print("UPDATE  →", part["data"])
        case "tasks":
            d = part["data"]
            if "error" not in d:
                print(f"TASK START  {d['name']} input={d['input']}")
            else:
                print(f"TASK FINISH {d['name']} result={d['result']}")
```

### Example 2: Filtering by namespace for subgraph events

```python
for part in graph.stream(
    input,
    config,
    stream_mode="values",
    subgraphs=True,
    version="v2",
):
    if part["ns"] == ():
        print("Root state:", part["data"])
    else:
        print(f"Subgraph {part['ns']} state:", part["data"])
```

### Example 3: Consuming `MessagesStreamPart` token-by-token

```python
from langchain_core.messages import AIMessageChunk

for part in graph.stream(input, config, stream_mode="messages", version="v2"):
    if part["type"] == "messages":
        token_msg, metadata = part["data"]
        if isinstance(token_msg, AIMessageChunk):
            print(token_msg.content, end="", flush=True)
```

---

## 10 · `PregelExecutableTask` and `CacheKey`

**Module:** `langgraph.types`  
**Exported as:** `from langgraph.types import PregelExecutableTask, CacheKey`

`PregelExecutableTask` is the internal dataclass that represents a **runnable task** at execution time — one concrete node invocation within a superstep. `CacheKey` is the three-field `NamedTuple` that uniquely identifies a cache entry for that task.

Both types are primarily encountered when writing **custom store adapters**, **stream transformers**, or **debug tooling** that hooks into pregel internals.

### Source (1.2.4)

```python
class CacheKey(NamedTuple):
    ns: tuple[str, ...]   # Namespace path for the cache entry
    key: str              # Content-addressed key (hash of inputs)
    ttl: int | None       # Time-to-live in seconds; None = no expiry

@dataclass(slots=True, frozen=True)   # Python 3.11+: also weakref_slot=True
class PregelExecutableTask:
    name: str                         # Node name
    input: Any                        # Snapshot of state passed to the node
    proc: Runnable                    # The compiled node runnable
    writes: deque[tuple[str, Any]]    # Accumulated channel writes (mutated in-place)
    config: RunnableConfig            # Task-scoped config (includes metadata)
    triggers: Sequence[str]           # Channel names that caused this task
    retry_policy: Sequence[RetryPolicy]
    cache_key: CacheKey | None        # None when caching is disabled for this task
    id: str                           # UUID identifying this task instance
    path: tuple[str | int | tuple, ...]  # Namespace path inside the graph
    writers: Sequence[Runnable] = ()  # Post-processing write runnables
    subgraphs: Sequence[PregelProtocol] = ()
    timeout: TimeoutPolicy | None = None
```

### Example 1: Inspecting `CacheKey` structure

```python
from langgraph.types import CacheKey

# CacheKey is a NamedTuple — unpack or access by name
ck = CacheKey(
    ns=("langgraph", "cache", "writes", "my_task"),
    key="a3f9b12c...",    # xxhash of serialised inputs
    ttl=300,              # 5-minute expiry
)

print(ck.ns)   # ('langgraph', 'cache', 'writes', 'my_task')
print(ck.key)  # 'a3f9b12c...'
print(ck.ttl)  # 300

# Use as a dict key or in a set — NamedTuple is hashable
seen: set[CacheKey] = {ck}
```

### Example 2: Observing tasks via `stream_mode="debug"`

`PregelExecutableTask` fields map directly onto `TaskPayload` / `TaskResultPayload` in the stream. Use `stream_mode="debug"` to see both checkpoint and task events together:

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

class State(TypedDict):
    n: int

def double(state: State) -> State:
    return {"n": state["n"] * 2}

graph = (
    StateGraph(State)
    .add_node(double)
    .add_edge(START, "double")
    .add_edge("double", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "debug-demo"}}

for event in graph.stream({"n": 3}, config, stream_mode="debug"):
    etype = event["type"]
    payload = event["payload"]
    if etype == "task":
        # Mirrors PregelExecutableTask: name, input, triggers, id
        print(f"[step {event['step']}] TASK START  name={payload['name']}  "
              f"input={payload['input']}  triggers={payload['triggers']}")
    elif etype == "task_result":
        print(f"[step {event['step']}] TASK FINISH name={payload['name']}  "
              f"result={payload['result']}  error={payload['error']}")
    elif etype == "checkpoint":
        print(f"[step {event['step']}] CHECKPOINT  next={payload['next']}")
```

### Example 3: Using `cache_key` to audit caching behaviour

```python
from langgraph.func import entrypoint, task
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

cache = InMemoryCache()

@task(cache_policy=CachePolicy(ttl=60))
def expensive_computation(x: int) -> int:
    print(f"  computing {x}")
    return x * x

@entrypoint(cache=cache)
def workflow(inputs: list[int]) -> list[int]:
    futures = [expensive_computation(n) for n in inputs]
    return [f.result() for f in futures]

# First run: all tasks computed
print(workflow.invoke([2, 3, 4]))  # computing 2, 3, 4 → [4, 9, 16]

# Second run: results served from cache (no "computing" prints)
print(workflow.invoke([2, 3, 4]))  # → [4, 9, 16]

# CacheKey for each task: ns identifies the task function,
# key is a hash of the serialised argument(s), ttl=60
```

---

## Quick Reference

| Type | Module | Key use |
|------|--------|---------|
| `Durability` | `langgraph.types` | `"sync"/"async"/"exit"` checkpoint write timing on `stream()`/`invoke()` |
| `NodeError` | `langgraph.errors` | Injected into `error_handler=` nodes with `.node` and `.error` |
| `NodeCancelledError` | `langgraph.errors` | Wraps user-raised `asyncio.CancelledError` for normal error flow |
| `TaskPayload` | `langgraph.types` | Start event emitted by `stream_mode="tasks"` |
| `TaskResultPayload` | `langgraph.types` | Finish event emitted by `stream_mode="tasks"` |
| `CheckpointPayload` | `langgraph.types` | Full snapshot emitted by `stream_mode="checkpoints"` |
| `CheckpointTask` | `langgraph.types` | Task entry within `CheckpointPayload.tasks` |
| `Item` | `langgraph.store.base` | Result of `store.get()` / `GetOp` with timestamps |
| `SearchItem` | `langgraph.store.base` | `Item` plus `.score` float for semantic search results |
| `GetOp` | `langgraph.store.base` | Read-by-key in `BaseStore.batch()` |
| `PutOp` | `langgraph.store.base` | Write or delete in `BaseStore.batch()` |
| `SearchOp` | `langgraph.store.base` | Filtered / semantic search in `BaseStore.batch()` |
| `ListNamespacesOp` | `langgraph.store.base` | Namespace discovery in `BaseStore.batch()` |
| `MatchCondition` | `langgraph.store.base` | Prefix / suffix / wildcard namespace filter |
| `UIMessage` | `langgraph.graph.ui` | Typed UI update event pushed by `push_ui_message()` |
| `RemoveUIMessage` | `langgraph.graph.ui` | Removal event pushed by `delete_ui_message()` |
| `GraphOutput` | `langgraph.types` | Typed `invoke(..., version="v2")` return with `.value` + `.interrupts` |
| `ValuesStreamPart` | `langgraph.types` | v2 stream chunk for `stream_mode="values"` |
| `UpdatesStreamPart` | `langgraph.types` | v2 stream chunk for `stream_mode="updates"` |
| `TasksStreamPart` | `langgraph.types` | v2 stream chunk for `stream_mode="tasks"` |
| `CheckpointStreamPart` | `langgraph.types` | v2 stream chunk for `stream_mode="checkpoints"` |
| `MessagesStreamPart` | `langgraph.types` | v2 stream chunk for `stream_mode="messages"` |
| `CustomStreamPart` | `langgraph.types` | v2 stream chunk for `stream_mode="custom"` |
| `PregelExecutableTask` | `langgraph.types` | Execution-time task dataclass with all runtime fields |
| `CacheKey` | `langgraph.types` | `(ns, key, ttl)` identity tuple for task cache entries |

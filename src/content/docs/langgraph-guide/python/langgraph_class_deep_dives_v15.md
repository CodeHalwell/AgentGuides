---
title: "Class deep-dives Vol. 15 — Runtime, Store, Streaming & Error APIs"
description: "Source-verified deep dives into Runtime/ExecutionInfo/RunControl, BaseStore/Item/SearchItem, store batch ops (GetOp/SearchOp/PutOp/ListNamespacesOp), IndexConfig/TTLConfig, UIMessage/push_ui_message, StreamTransformer/ProtocolEvent, RemoteGraph, error hierarchy (NodeError/NodeTimeoutError/GraphDrained), IsLastStep/RemainingSteps, and HumanResponse — with multiple runnable examples for each class."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 15"
  order: 46
---

# Class deep-dives Vol. 15 — Runtime, Store, Streaming & Error APIs

Verified against **`langgraph==1.2.5`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `Runtime` + `ExecutionInfo` + `RunControl` + `ServerInfo` | `langgraph.runtime` |
| 2 | `BaseStore` + `Item` + `SearchItem` | `langgraph.store.base` |
| 3 | `GetOp` + `SearchOp` + `PutOp` + `ListNamespacesOp` + `MatchCondition` | `langgraph.store.base` |
| 4 | `IndexConfig` + `TTLConfig` | `langgraph.store.base` |
| 5 | `UIMessage` + `push_ui_message` + `delete_ui_message` | `langgraph.graph.ui` |
| 6 | `StreamTransformer` + `ProtocolEvent` | `langgraph.stream._types` |
| 7 | `RemoteGraph` | `langgraph.pregel.remote` |
| 8 | `NodeError` + `NodeTimeoutError` + `NodeCancelledError` + `GraphDrained` | `langgraph.errors` |
| 9 | `IsLastStep` + `RemainingSteps` | `langgraph.managed.is_last_step` |
| 10 | `HumanResponse` | `langgraph.prebuilt.interrupt` |

---

## 1 · `Runtime` + `ExecutionInfo` + `RunControl` + `ServerInfo`

**Module:** `langgraph.runtime`  
**Import:**
```python
from langgraph.runtime import Runtime, ExecutionInfo, RunControl, ServerInfo, get_runtime
```

Added in **v0.6.0**, `Runtime` is the unified injection point for everything a node needs beyond its state slice: typed context, a cross-thread store, a stream writer, heartbeat ticks, per-attempt metadata, and cooperative drain control. Declare it as a parameter on any node function and LangGraph injects it automatically.

### Source signature (1.2.5)

```python
@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    context: ContextT = field(default=None)
    store: BaseStore | None = field(default=None)
    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    heartbeat: Callable[[], None] = field(default=_no_op_heartbeat)
    previous: Any = field(default=None)
    execution_info: ExecutionInfo | None = field(default=None)
    server_info: ServerInfo | None = field(default=None)
    control: RunControl | None = field(default=None)
```

`ExecutionInfo` is a frozen dataclass injected into `runtime.execution_info`:

```python
@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    checkpoint_id: str
    checkpoint_ns: str
    task_id: str
    thread_id: str | None = None
    run_id: str | None = None
    node_attempt: int = 1          # 1-indexed retry counter
    node_first_attempt_time: float | None = None
```

`RunControl` is a cooperative drain signal:

```python
class RunControl:
    def request_drain(self, reason: str = "shutdown") -> None: ...
    @property
    def drain_requested(self) -> bool: ...
    @property
    def drain_reason(self) -> str | None: ...
```

### Example 1: Typed context + store access via `Runtime`

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class AppContext:
    user_id: str
    locale: str = "en"


class State(TypedDict):
    query: str
    answer: str


store = InMemoryStore()
# Pre-populate some user data
store.put(("users",), "alice", {"name": "Alice", "plan": "pro"})
store.put(("users",), "bob",   {"name": "Bob",   "plan": "free"})


def lookup_and_answer(state: State, runtime: Runtime[AppContext]) -> dict:
    user_id = runtime.context.user_id
    locale  = runtime.context.locale

    user_item = runtime.store.get(("users",), user_id) if runtime.store else None
    name = user_item.value["name"] if user_item else "stranger"
    plan = user_item.value["plan"] if user_item else "unknown"

    return {"answer": f"[{locale}] Hello {name} ({plan} plan): {state['query']}"}


graph = (
    StateGraph(State, context_schema=AppContext)
    .add_node("lookup", lookup_and_answer)
    .add_edge(START, "lookup")
    .add_edge("lookup", END)
    .compile(store=store)
)

result = graph.invoke(
    {"query": "What is my plan?", "answer": ""},
    context=AppContext(user_id="alice", locale="en-GB"),
)
print(result["answer"])
# [en-GB] Hello Alice (pro plan): What is my plan?
```

### Example 2: `ExecutionInfo` — retry-aware node logic

```python
import asyncio
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy


class State(TypedDict):
    result: str


attempt_log: list[int] = []


def flaky_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info
    attempt = info.node_attempt if info else 1
    attempt_log.append(attempt)

    if attempt < 3:
        raise ValueError(f"Simulated failure on attempt {attempt}")

    return {"result": f"succeeded on attempt {attempt}"}


graph = (
    StateGraph(State)
    .add_node(
        "flaky",
        flaky_node,
        retry=RetryPolicy(max_attempts=5),
    )
    .add_edge(START, "flaky")
    .add_edge("flaky", END)
    .compile()
)

output = graph.invoke({"result": ""})
print(output["result"])   # succeeded on attempt 3
print(attempt_log)        # [1, 2, 3]
```

### Example 3: `heartbeat` — keeping alive during long computation

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import TimeoutPolicy


class State(TypedDict):
    items_processed: int


def slow_batch(state: State, runtime: Runtime) -> dict:
    for i in range(10):
        time.sleep(0.05)        # simulate work
        runtime.heartbeat()     # resets the idle-timeout timer
    return {"items_processed": 10}


graph = (
    StateGraph(State)
    .add_node(
        "batch",
        slow_batch,
        # Without heartbeats this would fire after 0.1 s of silence
        timeout=TimeoutPolicy(idle_timeout=0.2),
    )
    .add_edge(START, "batch")
    .add_edge("batch", END)
    .compile()
)

result = graph.invoke({"items_processed": 0})
print(result["items_processed"])  # 10
```

### Example 4: `RunControl` — cooperative drain on SIGTERM

```python
import signal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime, get_runtime
from langgraph.types import Command


class State(TypedDict):
    step: int
    done: bool


def step_node(state: State, runtime: Runtime) -> Command:
    # Check if a drain was requested (e.g. SIGTERM)
    if runtime.drain_requested:
        return Command(update={"done": True}, goto=END)
    return Command(update={"step": state["step"] + 1})


# In production you'd wire this to SIGTERM:
# signal.signal(signal.SIGTERM, lambda *_: get_runtime().control.request_drain())
```

### Example 5: `get_runtime()` — accessing runtime outside the node signature

```python
from langgraph.runtime import get_runtime

def side_effect_node(state: dict) -> dict:
    runtime = get_runtime()
    if runtime and runtime.execution_info:
        print(f"Thread: {runtime.execution_info.thread_id}")
        print(f"Attempt: {runtime.execution_info.node_attempt}")
    return state
```

---

## 2 · `BaseStore` + `Item` + `SearchItem`

**Module:** `langgraph.store.base`  
**Import:**
```python
from langgraph.store.base import BaseStore, Item, SearchItem
from langgraph.store.memory import InMemoryStore  # concrete implementation
```

`BaseStore` is the abstract base for **cross-thread, cross-run memory**. Unlike checkpoint state (which is scoped to a single thread), a store is shared across all threads. Items live in hierarchical **namespaces** — tuples of strings that act like a folder path.

### `Item` fields

```
Item.namespace  : tuple[str, ...]   — e.g. ("users", "alice")
Item.key        : str               — unique within namespace
Item.value      : dict[str, Any]    — arbitrary JSON-serialisable data
Item.created_at : datetime
Item.updated_at : datetime
```

`SearchItem` extends `Item` with a `score: float | None` field set by vector-search implementations.

### Example 1: Basic CRUD with `InMemoryStore`

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# PUT — namespace must be non-empty, no dots in labels
store.put(("profiles", "users"), "alice", {"name": "Alice", "role": "admin"})
store.put(("profiles", "users"), "bob",   {"name": "Bob",   "role": "viewer"})

# GET
item = store.get(("profiles", "users"), "alice")
print(item.value)           # {"name": "Alice", "role": "admin"}
print(item.namespace)       # ("profiles", "users")
print(item.key)             # "alice"

# SEARCH with filter
results = store.search(("profiles", "users"), filter={"role": "admin"})
print([r.key for r in results])   # ["alice"]

# DELETE — implemented as put(value=None)
store.delete(("profiles", "users"), "bob")
print(store.get(("profiles", "users"), "bob"))   # None
```

### Example 2: Namespaced memory across threads

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage

store = InMemoryStore()
checkpointer = InMemorySaver()


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


def remember(state: ChatState, store: InMemoryStore) -> dict:
    user_id = state["user_id"]
    ns = ("memories", user_id)

    # recall previous facts
    memories = store.search(ns)
    context = "; ".join(m.value.get("fact", "") for m in memories)

    last_user = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    # store a new fact about the user's last message
    import hashlib
    key = hashlib.md5(last_user.encode()).hexdigest()[:8]
    store.put(ns, key, {"fact": f"user said: {last_user}"})

    response = f"(Remembered {len(memories)} facts). Context: {context!r}"
    return {"messages": [AIMessage(content=response)]}


graph = (
    StateGraph(ChatState)
    .add_node("remember", remember)
    .add_edge(START, "remember")
    .add_edge("remember", END)
    .compile(checkpointer=checkpointer, store=store)
)

cfg = {"configurable": {"thread_id": "t1"}}
graph.invoke({"messages": [HumanMessage("Hello!")], "user_id": "alice"}, cfg)
result = graph.invoke({"messages": [HumanMessage("I like coffee")], "user_id": "alice"}, cfg)
print(result["messages"][-1].content)
# (Remembered 1 facts). Context: 'user said: Hello!'
```

### Example 3: Listing namespaces

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(("docs", "reports", "2024"), "q1", {"title": "Q1 Report"})
store.put(("docs", "reports", "2024"), "q2", {"title": "Q2 Report"})
store.put(("docs", "wiki"), "home", {"title": "Home"})
store.put(("cache", "embeddings"), "emb1", {"vec": [0.1, 0.2]})

# All namespaces up to depth 2
namespaces = store.list_namespaces(max_depth=2)
print(namespaces)
# [("cache", "embeddings"), ("docs", "reports"), ("docs", "wiki")]

# Only under "docs"
docs_ns = store.list_namespaces(prefix=("docs",))
print(docs_ns)
# [("docs", "reports", "2024"), ("docs", "wiki")]
```

### Example 4: Async store operations

```python
import asyncio
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()


async def main():
    await store.aput(("async_ns",), "key1", {"value": 42})
    item = await store.aget(("async_ns",), "key1")
    print(item.value)   # {"value": 42}

    results = await store.asearch(("async_ns",), filter={"value": {"$gte": 40}})
    print(results[0].key)  # key1

    await store.adelete(("async_ns",), "key1")
    print(await store.aget(("async_ns",), "key1"))  # None


asyncio.run(main())
```

---

## 3 · `GetOp` + `SearchOp` + `PutOp` + `ListNamespacesOp` + `MatchCondition`

**Module:** `langgraph.store.base`  
**Import:**
```python
from langgraph.store.base import (
    GetOp, SearchOp, PutOp, ListNamespacesOp,
    MatchCondition, Op, Result,
)
```

`BaseStore.batch()` is the **only abstract method** — all convenience methods (`get`, `put`, `search`, `delete`, `list_namespaces`) delegate to it. Understanding the op types lets you build high-throughput pipelines that issue many operations in a single round-trip.

### Op types at a glance

| `NamedTuple` | Fields | Returned `Result` type |
|---|---|---|
| `GetOp` | `namespace`, `key`, `refresh_ttl` | `Item \| None` |
| `PutOp` | `namespace`, `key`, `value`, `index`, `ttl` | `None` |
| `SearchOp` | `namespace_prefix`, `filter`, `limit`, `offset`, `query`, `refresh_ttl` | `list[SearchItem]` |
| `ListNamespacesOp` | `match_conditions`, `max_depth`, `limit`, `offset` | `list[tuple[str, ...]]` |

`MatchCondition(match_type, path)` is used inside `ListNamespacesOp` to filter returned namespaces by prefix or suffix. Wildcards (`"*"`) are allowed in `path`.

### Example 1: Batching multiple ops in one call

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import GetOp, PutOp, SearchOp

store = InMemoryStore()

# Batch: two puts + one get — all executed atomically
results = store.batch([
    PutOp(("inventory",), "item_a", {"qty": 10, "category": "tools"}),
    PutOp(("inventory",), "item_b", {"qty": 5,  "category": "consumables"}),
    GetOp(("inventory",), "item_a"),
])

item_a = results[2]   # result for GetOp (third op)
print(item_a.value)   # {"qty": 10, "category": "tools"}
print(results[:2])    # [None, None]  — PutOp results are None
```

### Example 2: Compound search + list in one batch

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import SearchOp, ListNamespacesOp, MatchCondition

store = InMemoryStore()
for i in range(5):
    store.put(("products", "electronics"), f"prod_{i}", {"price": i * 10, "in_stock": i % 2 == 0})

results = store.batch([
    SearchOp(
        namespace_prefix=("products",),
        filter={"in_stock": True},
        limit=10,
    ),
    ListNamespacesOp(
        match_conditions=(MatchCondition(match_type="prefix", path=("products",)),),
        max_depth=2,
    ),
])

items, namespaces = results
print([r.key for r in items])    # ["prod_0", "prod_2", "prod_4"]
print(namespaces)                # [("products", "electronics")]
```

### Example 3: `PutOp` with `index=False` (skip vector indexing)

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import PutOp, GetOp

store = InMemoryStore()

store.batch([
    # This item will be stored but NOT indexed for semantic search
    PutOp(("secrets",), "api_key", {"key": "sk-xxx"}, index=False),
    # This item uses default indexing
    PutOp(("docs",), "readme", {"text": "Getting started guide"}, index=None),
])

# Direct fetch still works regardless of indexing
item = store.batch([GetOp(("secrets",), "api_key")])[0]
print(item.value["key"])   # sk-xxx
```

### Example 4: `MatchCondition` wildcards for namespace discovery

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import ListNamespacesOp, MatchCondition

store = InMemoryStore()
for user in ["alice", "bob", "carol"]:
    store.put(("users", user, "v1"), "profile", {"name": user})
    store.put(("users", user, "v2"), "profile", {"name": user, "extended": True})

# Find all namespaces ending with "v2" under any user
results = store.batch([
    ListNamespacesOp(
        match_conditions=(
            MatchCondition(match_type="prefix", path=("users",)),
            MatchCondition(match_type="suffix", path=("v2",)),
        ),
    )
])[0]

print(results)
# [("users","alice","v2"), ("users","bob","v2"), ("users","carol","v2")]
```

---

## 4 · `IndexConfig` + `TTLConfig`

**Module:** `langgraph.store.base`  
**Import:**
```python
from langgraph.store.base import IndexConfig, TTLConfig
from langgraph.store.memory import InMemoryStore
```

These two `TypedDict`s are the constructor-level knobs on any store implementation that supports vector search and automatic expiry.

### `IndexConfig` fields

| Field | Type | Description |
|---|---|---|
| `dims` | `int` | Embedding vector dimension |
| `embed` | `Embeddings \| EmbeddingsFunc \| AEmbeddingsFunc \| str` | How to embed text |
| `fields` | `list[str] \| None` | JSON-path fields to embed. `["$"]` embeds the whole value (default) |

### `TTLConfig` fields

| Field | Type | Description |
|---|---|---|
| `refresh_on_read` | `bool` | Whether `get`/`search` resets the expiry clock (default `True`) |
| `default_ttl` | `float \| None` | Minutes until expiry for new items. `None` = no expiry |
| `sweep_interval_minutes` | `int \| None` | How often the store purges expired items |

### Example 1: Semantic search with `IndexConfig`

```python
from langgraph.store.memory import InMemoryStore

# Fake embedding function for illustration — replace with a real one
def embed_fn(texts: list[str]) -> list[list[float]]:
    # Real usage: call an embedding API here
    return [[float(ord(c)) / 1000 for c in text[:4].ljust(4)] for text in texts]

store = InMemoryStore(
    index={
        "dims": 4,
        "embed": embed_fn,
        "fields": ["content"],   # only embed the "content" field
    }
)

store.put(("docs",), "intro", {"content": "Getting started with LangGraph"})
store.put(("docs",), "adv",   {"content": "Advanced graph patterns"})
store.put(("docs",), "mem",   {"content": "Memory and persistence"})

# Semantic search — needs a real embed_fn to rank meaningfully
results = store.search(("docs",), query="how to start")
for r in results:
    print(r.key, r.score)
```

### Example 2: Field-level indexing on `put`

Override the store's default `fields` per item using `PutOp(index=[...])` or `store.put(..., index=[...])`:

```python
from langgraph.store.memory import InMemoryStore

def embed_fn(texts):
    return [[float(ord(c)) / 1000 for c in t[:4].ljust(4)] for t in texts]

store = InMemoryStore(index={"dims": 4, "embed": embed_fn})

# Index only the summary field for this item
store.put(
    ("articles",),
    "article_1",
    {"title": "LangGraph Guide", "summary": "Covers graph basics", "body": "...very long..."},
    index=["summary"],   # override: only embed "summary"
)

# Disable indexing entirely for secrets
store.put(
    ("internal",),
    "secret",
    {"token": "abc123"},
    index=False,
)
```

### Example 3: `TTLConfig` — automatic expiry

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    ttl={
        "default_ttl": 60.0,          # items expire after 60 minutes by default
        "refresh_on_read": True,      # reset timer on every get/search
        "sweep_interval_minutes": 10, # purge expired items every 10 minutes
    }
)

# This item uses the store default (60 min)
store.put(("sessions",), "sess_1", {"user": "alice"})

# This item overrides to 5 minutes
store.put(("sessions",), "sess_2", {"user": "bob"}, ttl=5.0)

# This item never expires
store.put(("sessions",), "sess_3", {"user": "carol"}, ttl=None)
```

### Example 4: Nested field indexing with JSON-path syntax

```python
from langgraph.store.memory import InMemoryStore

def embed_fn(texts):
    return [[float(ord(c)) / 1000 for c in t[:4].ljust(4)] for t in texts]

store = InMemoryStore(
    index={
        "dims": 4,
        "embed": embed_fn,
        "fields": ["$"],    # embed whole document by default
    }
)

# For this item, index each message's content separately
store.put(
    ("convs",),
    "conv_1",
    {
        "messages": [
            {"role": "user",      "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    },
    index=["messages[*].content"],  # creates one vector per message
)
```

---

## 5 · `UIMessage` + `push_ui_message` + `delete_ui_message`

**Module:** `langgraph.graph.ui`  
**Import:**
```python
from langgraph.graph.ui import UIMessage, RemoveUIMessage, push_ui_message, delete_ui_message
```

`UIMessage` lets nodes stream **structured UI updates** to a frontend in real time. The pattern works alongside `stream_mode="custom"` or alongside the `ui` state key. The helper `push_ui_message()` both writes to the stream (for real-time delivery) and appends to graph state (for replay).

### `UIMessage` TypedDict fields

| Field | Type | Description |
|---|---|---|
| `type` | `Literal["ui"]` | Discriminator |
| `id` | `str` | Unique component ID (auto-generated if not provided) |
| `name` | `str` | Frontend component name |
| `props` | `dict[str, Any]` | Props to pass to the component |
| `metadata` | `dict[str, Any]` | Framework metadata (run_id, merge flag, etc.) |

### Source signature of `push_ui_message`

```python
def push_ui_message(
    name: str,
    props: dict[str, Any],
    *,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
    message: AnyMessage | None = None,
    state_key: str | None = "ui",
    merge: bool = False,
) -> UIMessage: ...
```

### Example 1: Streaming a progress bar to the UI

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage, push_ui_message, ui_message_reducer


class State(TypedDict):
    task: str
    ui: Annotated[list[UIMessage], ui_message_reducer]


def run_task(state: State) -> dict:
    # Push a "loading" component — visible immediately via stream
    msg = push_ui_message(
        name="progress-bar",
        props={"label": f"Processing: {state['task']}", "progress": 0},
    )

    # Simulate work in stages
    for pct in (25, 50, 75, 100):
        push_ui_message(
            name="progress-bar",
            props={"label": f"Processing: {state['task']}", "progress": pct},
            id=msg["id"],   # same ID → update the existing component
            merge=True,     # merge props instead of replacing
        )

    return {}


graph = (
    StateGraph(State)
    .add_node("task", run_task)
    .add_edge(START, "task")
    .add_edge("task", END)
    .compile()
)

for chunk in graph.stream({"task": "data export", "ui": []}, stream_mode="custom"):
    print(chunk)   # UIMessage dicts arrive as they're pushed
```

### Example 2: Associating a `UIMessage` with a LLM message

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.ui import UIMessage, push_ui_message, ui_message_reducer
from langchain_core.messages import AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    ui: Annotated[list[UIMessage], ui_message_reducer]


def agent_with_card(state: State) -> dict:
    response = AIMessage(content="Here is your weather forecast.")

    # Associate the UI card with the AI message by passing message=response
    push_ui_message(
        name="weather-card",
        props={"city": "London", "temp": "15°C", "condition": "cloudy"},
        message=response,  # links card to this message via message_id in metadata
    )

    return {"messages": [response]}


graph = (
    StateGraph(State)
    .add_node("agent", agent_with_card)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile()
)
```

### Example 3: Removing a UI component

```python
from langgraph.graph.ui import delete_ui_message

def cleanup_node(state: dict) -> dict:
    # Remove a previously pushed component by its ID
    delete_ui_message("component-uuid-1234")
    return {}
```

---

## 6 · `StreamTransformer` + `ProtocolEvent`

**Module:** `langgraph.stream._types`  
**Import:**
```python
from langgraph.stream._types import StreamTransformer, ProtocolEvent
```

`StreamTransformer` is the extension point for the **v3 streaming API**. Transformers register on a `StreamMux` and receive every `ProtocolEvent` (a uniform envelope around raw stream parts) before it reaches the caller. Use them to build custom projections, PII redaction, cost tracking, or moderation pipelines.

### `ProtocolEvent` structure

```python
class ProtocolEvent(TypedDict):
    type: Literal["event"]
    event_id: NotRequired[str]
    seq: NotRequired[int]      # monotonic; use for ordering, not timestamp
    method: str                # StreamMode: "values", "messages", "custom", etc.
    params: _ProtocolEventParams

class _ProtocolEventParams(TypedDict):
    namespace: list[str]
    timestamp: int             # wall-clock ms — not monotonic
    data: Any
    interrupts: NotRequired[tuple[Any, ...]]
```

### `StreamTransformer` interface

```python
class StreamTransformer(ABC):
    requires_async: ClassVar[bool] = False
    supports_sync: ClassVar[bool] = False
    required_stream_modes: ClassVar[tuple[str, ...]] = ()
    before_builtins: ClassVar[bool] = False

    def init(self) -> dict[str, Any]: ...   # return the projection dict
    def process(self, event: ProtocolEvent) -> bool: ...   # return False to suppress
    async def aprocess(self, event: ProtocolEvent) -> bool: ...
    def finalize(self) -> None: ...         # run ends normally
    async def afinalize(self) -> None: ...
    def fail(self, err: BaseException) -> None: ...
    async def afail(self, err: BaseException) -> None: ...
    def schedule(self, coro, *, on_error="log") -> asyncio.Task: ...
```

### Example 1: Counting stream events by mode

```python
from collections import defaultdict
from typing import Any
from langgraph.stream._types import StreamTransformer, ProtocolEvent


class EventCounter(StreamTransformer):
    """Count how many events arrive per stream mode."""

    required_stream_modes = ()   # compatible with any modes

    def init(self) -> dict[str, Any]:
        self._counts: dict[str, int] = defaultdict(int)
        return {"event_counts": self._counts}

    def process(self, event: ProtocolEvent) -> bool:
        self._counts[event["method"]] += 1
        return True   # keep the event in the main log

    def finalize(self) -> None:
        print("Stream event counts:", dict(self._counts))
```

### Example 2: Filtering sensitive keys from `values` events

```python
from typing import Any
from langgraph.stream._types import StreamTransformer, ProtocolEvent


SENSITIVE_KEYS = frozenset({"api_key", "password", "token"})


class RedactSensitiveFields(StreamTransformer):
    """Remove sensitive fields from 'values' stream events before they reach callers."""

    before_builtins = True  # must run before built-in transformers snapshot values
    required_stream_modes = ("values",)

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "values":
            data = event["params"].get("data")
            if isinstance(data, dict):
                for key in SENSITIVE_KEYS:
                    data.pop(key, None)
        return True
```

### Example 3: Async transformer with `schedule()`

```python
import asyncio
from typing import Any
from langgraph.stream._types import StreamTransformer, ProtocolEvent


class AsyncCostTracker(StreamTransformer):
    """Log token usage to an external system after each messages event."""

    requires_async = True
    required_stream_modes = ("messages",)

    def init(self) -> dict[str, Any]:
        self._token_total = 0
        return {}

    async def aprocess(self, event: ProtocolEvent) -> bool:
        if event["method"] == "messages":
            delta = event["params"].get("data", {})
            if isinstance(delta, dict):
                usage = delta.get("usage_metadata") or {}
                self._token_total += usage.get("total_tokens", 0)
        return True

    async def afinalize(self) -> None:
        # Fire-and-forget log to external system
        self.schedule(self._log_tokens(self._token_total))

    async def _log_tokens(self, total: int) -> None:
        await asyncio.sleep(0)   # replace with real async API call
        print(f"Total tokens used: {total}")
```

---

## 7 · `RemoteGraph`

**Module:** `langgraph.pregel.remote`  
**Import:**
```python
from langgraph.pregel.remote import RemoteGraph
```

`RemoteGraph` wraps the **LangGraph Server API** — it behaves identically to a local `CompiledStateGraph` but delegates all execution to a remote deployment. You can use it as a standalone runnable or embed it as a subgraph node in a local graph.

### Constructor signature

```python
RemoteGraph(
    assistant_id: str,      # graph_id or assistant name on the server
    /,
    *,
    url: str | None = None,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    client: LangGraphClient | None = None,
    sync_client: SyncLangGraphClient | None = None,
    config: RunnableConfig | None = None,
    name: str | None = None,
    distributed_tracing: bool = False,
)
```

### Example 1: Calling a remote graph synchronously

```python
from langgraph.pregel.remote import RemoteGraph

# Replace with your LangGraph Server deployment URL
remote = RemoteGraph(
    "my-agent",
    url="https://my-deployment.langsmith.app",
    api_key="lsv2_...",
)

# invoke / stream / ainvoke / astream all work exactly like a local graph
result = remote.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    config={"configurable": {"thread_id": "thread-1"}},
)
print(result)
```

### Example 2: `RemoteGraph` as a subgraph node

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import HumanMessage


class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    delegated_result: str


# The remote specialised agent
specialist = RemoteGraph(
    "specialist-agent",
    url="https://my-deployment.langsmith.app",
)


def delegate(state: OrchestratorState) -> dict:
    result = specialist.invoke(
        {"messages": state["messages"]},
        config={"configurable": {"thread_id": "specialist-1"}},
    )
    last_msg = result["messages"][-1]
    return {"delegated_result": last_msg.content}


orchestrator = (
    StateGraph(OrchestratorState)
    .add_node("delegate", delegate)
    .add_edge(START, "delegate")
    .add_edge("delegate", END)
    .compile()
)
```

### Example 3: Streaming from a remote graph

```python
import asyncio
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph(
    "summariser",
    url="https://my-deployment.langsmith.app",
)


async def stream_remote() -> None:
    async for chunk in remote.astream(
        {"text": "Summarise the history of computing"},
        stream_mode="messages",
        config={"configurable": {"thread_id": "t-42"}},
    ):
        print(chunk)


asyncio.run(stream_remote())
```

### Example 4: Passing thread state from parent to remote

```python
from langgraph.pregel.remote import RemoteGraph

remote = RemoteGraph(
    "tool-executor",
    url="https://my-deployment.langsmith.app",
)

# Resume an interrupted run with `Command`
from langgraph.types import Command

result = remote.invoke(
    Command(resume={"approved": True}),
    config={"configurable": {"thread_id": "thread-with-interrupt"}},
)
```

---

## 8 · `NodeError` + `NodeTimeoutError` + `NodeCancelledError` + `GraphDrained`

**Module:** `langgraph.errors`  
**Import:**
```python
from langgraph.errors import (
    NodeError, NodeTimeoutError, NodeCancelledError,
    GraphDrained, GraphRecursionError, InvalidUpdateError,
    EmptyInputError, TaskNotFound,
)
```

### Error hierarchy

```
Exception
├── GraphBubbleUp                   # internal signalling base
│   ├── GraphDrained                # cooperative SIGTERM drain completed
│   └── GraphInterrupt              # interrupt() — suppressed by root
│       └── [deprecated] NodeInterrupt
├── GraphRecursionError(RecursionError)  # recursion_limit exceeded
├── InvalidUpdateError              # concurrent LastValue write / bad return value
├── EmptyInputError                 # graph received empty input
├── TaskNotFound                    # distributed-mode task lookup failure
├── NodeCancelledError              # user node raised asyncio.CancelledError
└── NodeTimeoutError                # idle_timeout or run_timeout exceeded
```

`NodeError` is a **dataclass** (not an Exception) injected into error handler functions:

```python
@dataclass(frozen=True, slots=True)
class NodeError:
    node: str           # name of the failed node
    error: BaseException # the original exception
```

### Example 1: Per-node error handler with `NodeError`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import Command


class State(TypedDict):
    value: int
    status: str


def risky_node(state: State) -> dict:
    if state["value"] < 0:
        raise ValueError(f"negative value: {state['value']}")
    return {"status": "ok"}


def handle_risky_error(state: State, error: NodeError) -> Command:
    # error.node  → "risky"
    # error.error → ValueError("negative value: -1")
    print(f"Node '{error.node}' failed: {error.error}")
    return Command(
        update={"status": f"recovered: {error.error}"},
        goto=END,
    )


graph = (
    StateGraph(State)
    .add_node("risky", risky_node, error_handler=handle_risky_error)
    .add_edge(START, "risky")
    .add_edge("risky", END)
    .compile()
)

result = graph.invoke({"value": -1, "status": ""})
print(result["status"])   # recovered: negative value: -1
```

### Example 2: Catching `NodeTimeoutError`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.types import Command, TimeoutPolicy
import time


class State(TypedDict):
    result: str


def slow_node(state: State) -> dict:
    time.sleep(10)   # will exceed run_timeout
    return {"result": "done"}


def timeout_handler(state: State, error: NodeError) -> Command:
    if isinstance(error.error, NodeTimeoutError):
        nte: NodeTimeoutError = error.error
        print(f"Timeout! kind={nte.kind}, elapsed={nte.elapsed:.2f}s")
    return Command(update={"result": "timed out"}, goto=END)


graph = (
    StateGraph(State)
    .add_node(
        "slow",
        slow_node,
        error_handler=timeout_handler,
        timeout=TimeoutPolicy(run_timeout=0.1),
    )
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)

result = graph.invoke({"result": ""})
print(result["result"])   # timed out
```

### Example 3: `GraphDrained` — graceful SIGTERM handling

```python
import signal
from langgraph.errors import GraphDrained
from langgraph.runtime import get_runtime

def setup_sigterm_handler():
    def _handler(signum, frame):
        runtime = get_runtime()
        if runtime and runtime.control:
            runtime.control.request_drain("SIGTERM received")

    signal.signal(signal.SIGTERM, _handler)

# When drain is requested, LangGraph raises GraphDrained after the current
# superstep, saves the checkpoint, and the run can be resumed later.
try:
    result = graph.invoke({"step": 0, "done": False})
except GraphDrained as e:
    print(f"Graph drained gracefully: {e.reason}")
    # checkpoint has been saved; resume later with the same thread_id
```

### Example 4: `GraphRecursionError` — adjusting the recursion limit

```python
from langgraph.errors import GraphRecursionError

try:
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Loop forever"}]},
        config={"recursion_limit": 10},
    )
except GraphRecursionError as e:
    print("Hit the recursion limit — increase it or fix the loop")
    print(e)
```

---

## 9 · `IsLastStep` + `RemainingSteps`

**Module:** `langgraph.managed.is_last_step`  
**Import:**
```python
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps
```

These are **type aliases** backed by `ManagedValue` subclasses. Declare a parameter with one of these types and LangGraph injects the current loop position. They are the idiomatic way to prevent a node from exceeding the graph's `recursion_limit`.

### Source (1.2.5)

```python
class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

IsLastStep = Annotated[bool, IsLastStepManager]


class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step

RemainingSteps = Annotated[int, RemainingStepsManager]
```

Both are `Annotated` aliases: annotating a state field with them causes LangGraph to inject the value rather than reading it from the state dict.

### Example 1: `IsLastStep` — forced termination on final step

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep
from langchain_core.messages import HumanMessage, AIMessage


class AgentState(TypedDict):
    messages: Annotated[list, lambda a, b: a + b]
    is_last_step: IsLastStep   # injected by the framework


def agent_node(state: AgentState) -> dict:
    if state["is_last_step"]:
        # Forced exit before hitting GraphRecursionError
        return {
            "messages": [AIMessage(content="I've run out of steps. Final answer: unknown.")]
        }

    # Normal agent logic
    last_msg = state["messages"][-1].content
    return {
        "messages": [AIMessage(content=f"Thinking about: {last_msg}")]
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and "Final answer" in last.content:
        return END
    return "agent"


graph = (
    StateGraph(AgentState)
    .add_node("agent", agent_node)
    .add_edge(START, "agent")
    .add_conditional_edges("agent", should_continue)
    .compile()
)

result = graph.invoke(
    {"messages": [HumanMessage(content="Keep going")]},
    config={"recursion_limit": 4},
)
print(result["messages"][-1].content)
# I've run out of steps. Final answer: unknown.
```

### Example 2: `RemainingSteps` — proportional work

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.types import Command


class PlanState(TypedDict):
    plan: list[str]
    done: list[str]
    remaining_steps: RemainingSteps   # injected


def execute_node(state: PlanState) -> Command:
    steps_left = state["remaining_steps"]

    if not state["plan"] or steps_left <= 1:
        return Command(update={}, goto=END)

    # Take only what fits in the remaining budget
    safe_batch = state["plan"][: max(1, steps_left - 1)]
    return Command(
        update={
            "plan": state["plan"][len(safe_batch):],
            "done": state["done"] + safe_batch,
        }
    )


graph = (
    StateGraph(PlanState)
    .add_node("execute", execute_node)
    .add_edge(START, "execute")
    .add_conditional_edges(
        "execute",
        lambda s: END if not s["plan"] else "execute",
    )
    .compile()
)

result = graph.invoke(
    {"plan": ["a", "b", "c", "d", "e"], "done": []},
    config={"recursion_limit": 4},
)
print(result["done"])   # at most recursion_limit-1 items
```

### Example 3: Combining `IsLastStep` with `RetryPolicy`

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep
from langgraph.types import RetryPolicy


class State(TypedDict):
    counter: int
    is_last_step: IsLastStep


def loop_or_stop(state: State) -> dict | None:
    if state["is_last_step"]:
        print(f"Stopping at counter={state['counter']} (last step)")
        return {}
    return {"counter": state["counter"] + 1}


def keep_going(state: State) -> str:
    return "node" if state["counter"] < 100 else END


graph = (
    StateGraph(State)
    .add_node("node", loop_or_stop, retry=RetryPolicy(max_attempts=2))
    .add_edge(START, "node")
    .add_conditional_edges("node", keep_going)
    .compile()
)

graph.invoke({"counter": 0}, config={"recursion_limit": 5})
```

---

## 10 · `HumanResponse`

**Module:** `langgraph.prebuilt.interrupt`  
**Import:**
```python
from langgraph.prebuilt.interrupt import HumanResponse
```

`HumanResponse` is the **structured reply** that flows back into a graph when a human-in-the-loop interrupt is resumed. It lives alongside `interrupt()` (from `langgraph.types`) which is the modern replacement for the deprecated `NodeInterrupt`. The response type field tells your node exactly what the operator chose to do.

### `HumanResponse` TypedDict

```python
class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
```

| `type` | `args` | Meaning |
|--------|--------|---------|
| `"accept"` | `None` | User approved as-is |
| `"ignore"` | `None` | User skipped this step |
| `"response"` | `str` | User provided free-text feedback |
| `"edit"` | `ActionRequest` | User modified the payload |

### Example 1: Tool-approval flow

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.prebuilt.interrupt import HumanResponse
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    approved: bool


def agent(state: State) -> dict:
    return {
        "messages": [AIMessage(content="I want to call delete_user(user_id=42)")]
    }


def approval_gate(state: State) -> Command:
    last = state["messages"][-1]
    response: HumanResponse = interrupt(
        {
            "question": "Approve this action?",
            "action": last.content,
        }
    )

    if response["type"] == "accept":
        return Command(update={"approved": True}, goto="execute")
    elif response["type"] == "ignore":
        return Command(update={"approved": False}, goto=END)
    elif response["type"] == "response":
        # User gave feedback — add it to messages and loop back to agent
        feedback = response["args"]
        return Command(
            update={
                "messages": [HumanMessage(content=f"Feedback: {feedback}")],
                "approved": False,
            },
            goto="agent",
        )
    else:  # "edit"
        edited = response["args"]  # ActionRequest with updated args
        return Command(update={"approved": True}, goto="execute")


def execute(state: State) -> dict:
    return {"messages": [AIMessage(content="Action executed.")]}


graph = (
    StateGraph(State)
    .add_node("agent", agent)
    .add_node("approval_gate", approval_gate)
    .add_node("execute", execute)
    .add_edge(START, "agent")
    .add_edge("agent", "approval_gate")
    .add_edge("execute", END)
    .compile(checkpointer=InMemorySaver())
)

# First invocation — graph pauses at interrupt
thread_cfg = {"configurable": {"thread_id": "approval-1"}}
graph.invoke({"messages": [], "approved": False}, thread_cfg)

# Resume with "accept"
result = graph.invoke(
    Command(resume=HumanResponse(type="accept", args=None)),
    thread_cfg,
)
print(result["approved"])   # True
```

### Example 2: `"edit"` response — user modifies the action

```python
from langgraph.prebuilt.interrupt import HumanResponse, ActionRequest
from langgraph.types import Command

# When the user edits the payload, resume with an ActionRequest
edited_request = ActionRequest(
    action="delete_user",
    args={"user_id": 99},   # changed from 42 to 99
)

result = graph.invoke(
    Command(
        resume=HumanResponse(
            type="edit",
            args=edited_request,
        )
    ),
    {"configurable": {"thread_id": "approval-2"}},
)
```

### Example 3: Multi-interrupt loop — review every output

```python
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.prebuilt.interrupt import HumanResponse
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    accepted_count: int


def generate(state: State) -> dict:
    draft = f"Draft #{len(state['messages']) + 1}: some AI output"
    return {"messages": [AIMessage(content=draft)]}


def review(state: State) -> Command:
    last = state["messages"][-1]
    resp: HumanResponse = interrupt({"draft": last.content})

    if resp["type"] == "accept":
        if state["accepted_count"] + 1 >= 3:
            return Command(update={"accepted_count": state["accepted_count"] + 1}, goto=END)
        return Command(update={"accepted_count": state["accepted_count"] + 1}, goto="generate")
    elif resp["type"] == "response":
        return Command(
            update={"messages": [HumanMessage(content=resp["args"])]},
            goto="generate",
        )
    return Command(goto="generate")


graph = (
    StateGraph(State)
    .add_node("generate", generate)
    .add_node("review", review)
    .add_edge(START, "generate")
    .add_edge("generate", "review")
    .compile(checkpointer=InMemorySaver())
)
```

---

## Summary

| Class | Module | Use case |
|---|---|---|
| `Runtime` | `langgraph.runtime` | Unified injection of context, store, heartbeat, drain control |
| `ExecutionInfo` | `langgraph.runtime` | Per-attempt metadata (thread_id, checkpoint_id, attempt number) |
| `RunControl` | `langgraph.runtime` | Cooperative SIGTERM drain signalling |
| `BaseStore` | `langgraph.store.base` | Cross-thread, cross-run persistent key-value memory |
| `Item` / `SearchItem` | `langgraph.store.base` | Retrieved store items (with optional similarity score) |
| `GetOp` / `PutOp` / `SearchOp` / `ListNamespacesOp` | `langgraph.store.base` | Batch store operations for high-throughput scenarios |
| `IndexConfig` / `TTLConfig` | `langgraph.store.base` | Store-level vector search and expiry configuration |
| `UIMessage` + `push_ui_message` | `langgraph.graph.ui` | Stream real-time UI component updates from nodes |
| `StreamTransformer` | `langgraph.stream._types` | Custom stream projection, redaction, or side-effects |
| `RemoteGraph` | `langgraph.pregel.remote` | Embed a remote LangGraph Server deployment as a subgraph |
| `NodeError` / `NodeTimeoutError` / `GraphDrained` | `langgraph.errors` | Per-node error handlers and graceful drain |
| `IsLastStep` / `RemainingSteps` | `langgraph.managed.is_last_step` | Prevent GraphRecursionError in looping agents |
| `HumanResponse` | `langgraph.prebuilt.interrupt` | Structured accept/ignore/edit/response from human operators |

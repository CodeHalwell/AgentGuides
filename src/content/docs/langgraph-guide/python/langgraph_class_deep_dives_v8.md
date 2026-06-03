---
title: "Class deep-dives Vol. 8 — 10 more LangGraph types"
description: "Source-verified deep dives into ExecutionInfo/Runtime.heartbeat, ServerInfo/BaseUser, ReplayState, StreamMux, Call (functional API internals), ChannelWrite/ChannelWriteEntry, PregelRunner/FuturesDict, WritesProtocol/PregelTaskWrites, SyncPregelLoop/AsyncPregelLoop, and DuplexStream — with runnable examples for every feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 8"
  order: 32
---

# Class deep-dives Vol. 8 — 10 more LangGraph types

Verified against **`langgraph==1.2.4`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Every section was written by inspecting the installed package source directly (`/usr/local/lib/python3.11/dist-packages/langgraph`). All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

[→ Vol. 3 covers interrupt()/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig](./langgraph_class_deep_dives_v3/)

[→ Vol. 4 covers set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler on add_node, error taxonomy](./langgraph_class_deep_dives_v4/)

[→ Vol. 5 covers RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda/ensure_embeddings, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt protocol](./langgraph_class_deep_dives_v5/)

[→ Vol. 6 covers GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message/ui_message_reducer, ProtocolEvent](./langgraph_class_deep_dives_v6/)

[→ Vol. 7 covers PregelProtocol/StreamProtocol, BackgroundExecutor/AsyncBackgroundExecutor, AsyncBatchedBaseStore/_dedupe_ops, get_text_at_path/tokenize_path, SerdeEvent/register_serde_event_listener, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec/node Protocols, identifier/get_runnable_for_task](./langgraph_class_deep_dives_v7/)

---

## 1 · `ExecutionInfo` + `Runtime.heartbeat()`

**Module:** `langgraph.runtime`  
**Exported as:** `from langgraph.runtime import ExecutionInfo, Runtime`

`ExecutionInfo` is the frozen dataclass attached to `runtime.execution_info` in every node invocation. It carries all per-execution IDs and counters in one place. `Runtime.heartbeat()` is a callable field on `Runtime` that resets the *idle timeout* clock for the current node.

### `ExecutionInfo` source

```python
@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    checkpoint_id: str
    """ID of the checkpoint written after the previous step."""

    checkpoint_ns: str
    """Namespace of the checkpoint, used to isolate subgraph scopes."""

    task_id: str
    """The Pregel task ID executing this node invocation."""

    thread_id: str | None = None
    """Conversation thread identifier. None when no checkpointer is attached."""

    run_id: str | None = None
    """Run ID from RunnableConfig. None if not supplied by the caller."""

    node_attempt: int = 1
    """How many times this node has been attempted (1-indexed). Increments on retries."""

    node_first_attempt_time: float | None = None
    """Unix timestamp of the very first attempt. None on the first attempt itself."""

    def patch(self, **overrides: Any) -> ExecutionInfo:
        """Return a new ExecutionInfo with selected fields replaced.
        Useful in tests to simulate retries or specific thread IDs."""
        return replace(self, **overrides)
```

### `heartbeat` field

```python
@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    # ...
    heartbeat: Callable[[], None] = field(default=_no_op_heartbeat)
    """Record progress for the current node's idle_timeout.

    Call from inside long-running work that does not naturally emit
    writes, stream chunks, child tasks, or LangChain callback events,
    to prevent the node from being treated as idle. It is also the
    only progress signal honoured under TimeoutPolicy(refresh_on="heartbeat").
    Outside an idle-timed attempt this is a no-op.
    """
```

`heartbeat()` is a zero-argument callable — calling it tells the executor "this node is still making progress" and resets the idle-timeout clock. Outside an active idle-timed attempt it is a no-op, so calling it unconditionally is safe.

### Field reference

| Field | Type | Description |
|---|---|---|
| `checkpoint_id` | `str` | ID of the checkpoint from the previous step |
| `checkpoint_ns` | `str` | Subgraph namespace string (empty string for root graph) |
| `task_id` | `str` | Pregel task ID — use as an idempotency key for external calls |
| `thread_id` | `str \| None` | `None` when running without a checkpointer |
| `run_id` | `str \| None` | `None` when not supplied in `RunnableConfig` |
| `node_attempt` | `int` | 1-indexed retry count; 1 = first attempt |
| `node_first_attempt_time` | `float \| None` | Unix timestamp of first attempt; `None` on the first attempt |

### Pattern 1: idempotent external API calls

Use `task_id` as a stable idempotency key. The same logical task retains the same `task_id` across retries, so upstream services can safely reject duplicate requests.

```python
import time
import httpx
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    order_id: str
    payment_result: str | None


def charge_customer(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info
    attempt = info.node_attempt

    # Compute elapsed time across all retries
    elapsed = 0.0
    if attempt > 1 and info.node_first_attempt_time is not None:
        elapsed = time.time() - info.node_first_attempt_time
        print(f"Retry #{attempt} after {elapsed:.1f}s total")

    # task_id is stable across retries — safe idempotency key
    response = httpx.post(
        "https://payments.example.com/charge",
        json={"order": state["order_id"], "amount": 99.99},
        headers={"Idempotency-Key": info.task_id},
        timeout=5.0,
    )
    response.raise_for_status()
    return {"payment_result": response.json()["status"]}


builder = StateGraph(State)
builder.add_node(
    "charge",
    charge_customer,
    retry=RetryPolicy(max_attempts=3, backoff_factor=2.0),
)
builder.add_edge(START, "charge")
builder.add_edge("charge", END)
graph = builder.compile()
```

### Pattern 2: subgraph observability via `checkpoint_ns`

When running subgraphs, each subgraph has its own `checkpoint_ns`. Use it to distinguish which level of the graph hierarchy is executing.

```python
from langgraph.runtime import Runtime


def worker_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info
    # checkpoint_ns looks like "supervisor|subagent:task-abc123"
    depth = info.checkpoint_ns.count("|") + 1 if info.checkpoint_ns else 0
    print(f"[depth={depth}] thread={info.thread_id} task={info.task_id[:8]}")
    return {}
```

### Pattern 3: `heartbeat()` for long-running nodes

Pair `heartbeat()` with `TimeoutPolicy(idle_timeout=…, refresh_on="heartbeat")` to allow a node to run indefinitely as long as it keeps making progress.

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import TimeoutPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    items: list[str]
    processed: list[str]


async def batch_processor(state: State, runtime: Runtime) -> dict:
    results = []
    for item in state["items"]:
        # Do work that takes a while per item
        await asyncio.sleep(0.5)          # simulate per-item work
        results.append(f"done:{item}")
        runtime.heartbeat()               # reset the idle clock after each item
    return {"processed": results}


builder = StateGraph(State)
builder.add_node(
    "process",
    batch_processor,
    # Run for up to 2s without heartbeat before considering node idle
    # No hard run_timeout — so it can run as long as it keeps pinging
    timeout=TimeoutPolicy(idle_timeout=2.0, refresh_on="heartbeat"),
)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()
```

### Pattern 4: `patch()` for unit testing

`ExecutionInfo.patch()` returns a new frozen instance with some fields overridden. Use it to inject specific execution contexts in unit tests without running a full graph.

```python
import time
from langgraph.runtime import ExecutionInfo, Runtime


def make_test_runtime(thread_id: str = "test", attempt: int = 1) -> Runtime:
    first_attempt_time = time.time() - 10.0 if attempt > 1 else None
    info = ExecutionInfo(
        checkpoint_id="ckpt-test",
        checkpoint_ns="",
        task_id="task-test-001",
        thread_id=thread_id,
        node_attempt=attempt,
        node_first_attempt_time=first_attempt_time,
    )
    return Runtime(execution_info=info)


# Simulate a second-attempt run
runtime = make_test_runtime(attempt=2)
assert runtime.execution_info.node_attempt == 2

# Simulate promoting to a specific task_id
info2 = runtime.execution_info.patch(task_id="task-prod-abc")
assert info2.task_id == "task-prod-abc"
assert info2.node_attempt == 2   # other fields preserved
```

---

## 2 · `ServerInfo` + `BaseUser`

**Module:** `langgraph.runtime`  
**Exported as:** `from langgraph.runtime import ServerInfo`  
**`BaseUser` re-exported from:** `langgraph_sdk.auth.types`

`ServerInfo` is a frozen dataclass injected into `runtime.server_info` when the graph is running inside a **LangGraph Platform** (hosted) deployment. It is always `None` when running open-source LangGraph locally. `BaseUser` is a protocol that the authenticated user object implements.

### `ServerInfo` source

```python
@dataclass(frozen=True, slots=True)
class ServerInfo:
    """Metadata injected by LangGraph Server.
    None when running open-source LangGraph without LangSmith deployments."""

    assistant_id: str
    """The assistant ID for the current execution."""

    graph_id: str
    """The graph ID for the current execution."""

    user: BaseUser | None = None
    """The authenticated user, if any.

    Implements the BaseUser protocol from langgraph_sdk.auth.types,
    which supports both attribute access (user.identity) and
    dict-like access (user["identity"]).
    """
```

### `BaseUser` protocol

`BaseUser` (from `langgraph_sdk.auth.types`) is a protocol that the authenticated user object implements when LangGraph Platform auth middleware is active. It supports both attribute-style and dict-style access:

```python
# From langgraph_sdk.auth.types
class BaseUser(Protocol):
    @property
    def identity(self) -> str: ...
    # Also supports __getitem__ for dict-style access
    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...
```

### Guarding against OSS vs Platform

Always check `server_info is not None` before using it. Your graph code should work identically in both environments.

```python
from langgraph.runtime import Runtime, ServerInfo


def analytics_node(state: State, runtime: Runtime) -> dict:
    # Works in both OSS and LangGraph Platform
    server = runtime.server_info
    if server is not None:
        # Running inside LangGraph Platform deployment
        graph_id = server.graph_id
        assistant_id = server.assistant_id

        # Access the authenticated user (if auth middleware is configured)
        if server.user is not None:
            user_identity = server.user.identity
            # or: user_identity = server.user["identity"]
            print(f"Authenticated as: {user_identity} on graph {graph_id}")
        else:
            print(f"Running as anonymous on graph {graph_id}")
    else:
        # Running locally / in tests — no platform context available
        print("Running in OSS mode (no ServerInfo)")

    return {}
```

### Using `ServerInfo` for tenant isolation

In multi-tenant deployments, use `server_info.user.identity` to scope store namespaces per user.

```python
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


class State(TypedDict):
    query: str
    results: list[str]


def tenant_aware_search(state: State, runtime: Runtime) -> dict:
    # Determine the effective user identity
    user_id = "anonymous"
    if runtime.server_info and runtime.server_info.user:
        user_id = runtime.server_info.user.identity

    # Namespace all store operations under the user's identity
    if runtime.store:
        cached = runtime.store.get(("cache", user_id), state["query"])
        if cached:
            return {"results": cached.value["results"]}

        results = ["result-1", "result-2"]   # replace with real search
        runtime.store.put(
            ("cache", user_id),
            state["query"],
            {"results": results},
        )
        return {"results": results}

    return {"results": []}
```

### Testing with a mock `ServerInfo`

```python
from dataclasses import dataclass
from langgraph.runtime import ExecutionInfo, Runtime, ServerInfo


@dataclass
class MockUser:
    """Minimal BaseUser implementation for testing."""
    identity: str

    def __getitem__(self, key: str) -> str:
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def make_platform_runtime(user_id: str = "user-123") -> Runtime:
    return Runtime(
        execution_info=ExecutionInfo(
            checkpoint_id="ckpt-test",
            checkpoint_ns="",
            task_id="task-test",
        ),
        server_info=ServerInfo(
            assistant_id="asst-abc",
            graph_id="graph-xyz",
            user=MockUser(identity=user_id),
        ),
    )


runtime = make_platform_runtime("alice@example.com")
assert runtime.server_info.user.identity == "alice@example.com"
```

---

## 3 · `ReplayState`

**Module:** `langgraph._internal._replay`  
**Import:** `from langgraph._internal._replay import ReplayState`

`ReplayState` is an internal object used during **time-travel** (branching from a historical checkpoint). It tracks which subgraph namespaces have already been visited during the replay so that the right checkpoint is loaded at each level of the graph hierarchy.

### Source

```python
class ReplayState:
    """Tracks which subgraphs have already loaded their pre-replay checkpoint.

    During a parent replay, each subgraph's first invocation should restore the
    checkpoint from before the replay point. Subsequent invocations of the same
    subgraph (e.g. in a loop) should use normal checkpoint loading so they pick
    up freshly created checkpoints.
    """

    __slots__ = ("checkpoint_id", "_visited_ns")

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        self._visited_ns: set[str] = set()

    def _is_first_visit(self, checkpoint_ns: str) -> bool:
        """True the first time a subgraph namespace is seen.
        The task-id suffix is stripped so the same logical subgraph is
        recognized across loop iterations despite having a different task id."""
        # "sub_node:task_id" -> "sub_node"
        stable_ns = (
            checkpoint_ns.rsplit(NS_END, 1)[0]
            if NS_END in checkpoint_ns
            else checkpoint_ns
        )
        if stable_ns in self._visited_ns:
            return False
        self._visited_ns.add(stable_ns)
        return True

    def get_checkpoint(
        self,
        checkpoint_ns: str,
        checkpointer: BaseCheckpointSaver,
        checkpoint_config: RunnableConfig,
    ) -> CheckpointTuple | None:
        """Load the right checkpoint for a subgraph during replay.

        First call for a namespace: returns the latest checkpoint *before* the
        replay point. Subsequent calls: falls back to normal latest loading.
        """
        if self._is_first_visit(checkpoint_ns):
            for saved in checkpointer.list(
                checkpoint_config,
                before={"configurable": {"checkpoint_id": self.checkpoint_id}},
                limit=1,
            ):
                return saved
            return None
        return checkpointer.get_tuple(checkpoint_config)

    async def aget_checkpoint(...) -> CheckpointTuple | None:
        """Async version of get_checkpoint."""
```

### Why `ReplayState` exists

When you call `graph.invoke(None, config)` where `config` points at a historical checkpoint, LangGraph needs to:

1. Load each subgraph from its checkpoint *prior to* the replay point (not the latest).
2. For subsequent calls to the same subgraph (in a loop), load the *latest* checkpoint so the graph can make forward progress.

A single `ReplayState` instance is shared by reference across all derived configs within one parent execution. The `_visited_ns` set tracks which subgraph namespaces have already been seeded with the pre-replay state.

### Observing time-travel in action

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    step: int


def node_a(state: State) -> dict:
    return {"step": state["step"] + 1}


def node_b(state: State) -> dict:
    return {"messages": [AIMessage(f"Step {state['step']} complete")]}


checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "replay-demo"}}

# Run three times, producing checkpoints at step 1, 2, 3
for i in range(3):
    graph.invoke({"messages": [HumanMessage(f"msg-{i}")], "step": 0}, config)

# List all checkpoints — most recent first
history = list(graph.get_state_history(config))
print(f"Total checkpoints: {len(history)}")

# Replay from the second checkpoint (branch the timeline)
checkpoint_to_replay = history[-2]    # second-to-last
replay_config = checkpoint_to_replay.config

# Run from that historical point — ReplayState ensures subgraphs
# are loaded from the correct pre-replay checkpoint on first visit
result = graph.invoke(
    {"messages": [HumanMessage("branched run")]},
    replay_config,
)
print(result)
```

### Understanding `_is_first_visit` and the task-id suffix

Subgraph checkpoint namespaces look like `"sub_node:task-abc123"`. The `:task-abc123` suffix changes every loop iteration. `_is_first_visit` strips the suffix to get the stable namespace `"sub_node"`, ensuring the same logical subgraph is recognised across loop iterations.

```python
from langgraph._internal._replay import ReplayState
from langgraph._internal._constants import NS_END

state = ReplayState("ckpt-parent-001")

# First visit to "sub_node" — returns True regardless of task-id suffix
assert state._is_first_visit("sub_node:task-abc") is True

# Second visit to same logical subgraph (different task-id, same loop body)
assert state._is_first_visit("sub_node:task-xyz") is False

# A completely different subgraph — first visit
assert state._is_first_visit("other_node:task-111") is True
```

---

## 4 · `StreamMux`

**Module:** `langgraph.stream._mux`  
**Import:** `from langgraph.stream._mux import StreamMux`

`StreamMux` is the central event dispatcher for LangGraph's streaming infrastructure. It owns the main event log, routes events through a transformer pipeline (`StreamTransformer` instances), and auto-wires `StreamChannel` instances found in transformer projections so every `push()` also injects a `ProtocolEvent` into the main log.

### Source (condensed)

```python
class StreamMux:
    """Central event dispatcher for the streaming infrastructure."""

    def __init__(
        self,
        transformers: list[StreamTransformer] | None = None,
        *,
        is_async: bool = False,
        factories: list[TransformerFactory] | None = None,
        scope: tuple[str, ...] = (),
        _assign_seq: bool = True,
    ) -> None: ...

    def push(self, event: ProtocolEvent) -> None:
        """Route an event through all transformers, then append to the main log."""

    async def apush(self, event: ProtocolEvent) -> None:
        """Async variant — awaits each transformer's aprocess in order."""

    def close(self) -> None:
        """Finalize all transformers, close projections and the main event log."""

    async def aclose(self) -> None:
        """Async finalize — awaits scheduled tasks, then closes."""

    def fail(self, err: BaseException) -> None:
        """Fail all transformers, projections, and the main event log."""

    def bind_pump(self, fn: Callable[[], bool]) -> None:
        """Wire the sync pull callback onto every projection."""

    def bind_apump(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Wire the async pull callback onto every projection."""

    def _make_child(self, scope: tuple[str, ...]) -> StreamMux:
        """Build a mini-mux with the same factories scoped to a subgraph."""

    def transformer_by_key(self, key: str) -> StreamTransformer | None:
        """Return the transformer that contributed a projection key."""

    # Key attributes
    extensions: dict[str, Any]      # merged projection across all transformers
    native_keys: set[str]           # keys from _native=True transformers
    scope: tuple[str, ...]          # () for root, subgraph name for children
    is_async: bool
```

### How `StreamMux` fits in the execution pipeline

```
graph.astream(input, stream_mode="values")
  │
  ├── AsyncPregelLoop.__aenter__() builds a StreamMux with:
  │     factories=[ValuesTransformer, ...]   (one instance per scope)
  │
  ├── Each superstep:
  │     PregelRunner submits tasks → tasks call ChannelWrite
  │     ChannelWrite → StreamProtocol → StreamMux.push(ProtocolEvent)
  │     StreamMux routes through transformers → appends to main event log
  │
  └── AsyncGraphRunStream iterates the main event log (consumer)
```

### Using transformer factories vs pre-built instances

Factories propagate to child mini-muxes for nested subgraphs; pre-built instances are root-only.

```python
from langgraph.stream._mux import StreamMux
from langgraph.stream._types import StreamTransformer, ProtocolEvent


class AuditTransformer(StreamTransformer):
    """Log every event that passes through the mux."""

    def init(self) -> dict:
        return {}   # no projections — this transformer is side-effects only

    def process(self, event: ProtocolEvent) -> bool:
        method = event.get("method", "?")
        print(f"[audit] seq={event.get('seq')} method={method}")
        return True  # keep the event in the main log


# Factory form — propagates into subgraph mini-muxes
mux = StreamMux(
    factories=[lambda scope: AuditTransformer()],
    is_async=False,
)

# Pre-built form — root-only (cannot be cloned into child muxes)
audit = AuditTransformer()
audit.init()
mux2 = StreamMux(transformers=[audit], is_async=False)
```

### Injecting a custom `StreamTransformer` into a graph

The standard way to extend the streaming pipeline is via `compile(transformers=[...])` or the `GraphRunStream` API:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.stream._types import StreamTransformer, ProtocolEvent
from langgraph.stream.run_stream import GraphRunStream
from typing_extensions import TypedDict


class State(TypedDict):
    count: int


def counter(state: State) -> dict:
    return {"count": state["count"] + 1}


class CountingTransformer(StreamTransformer):
    """Count how many events pass through the mux."""

    def __init__(self, scope: tuple[str, ...]) -> None:
        self.event_count = 0
        self.scope = scope

    def init(self) -> dict:
        return {}

    def process(self, event: ProtocolEvent) -> bool:
        self.event_count += 1
        return True

    def finalize(self) -> None:
        print(f"scope={self.scope} total_events={self.event_count}")


builder = StateGraph(State)
builder.add_node("count", counter)
builder.add_edge(START, "count")
builder.add_edge("count", END)
graph = builder.compile()

# GraphRunStream exposes the mux's extensions dict
with GraphRunStream(
    graph,
    {"count": 0},
    stream_mode="values",
    factories=[CountingTransformer],
) as run:
    for chunk in run:
        pass
# prints: scope=() total_events=<n>
```

### Child muxes and subgraph scoping

`_make_child(scope)` builds a fresh mini-mux with the same factories but scoped to a subgraph namespace. The child inherits the pump bindings but does not assign `seq` numbers (to avoid mutating shared forwarded events).

```python
# The root mux uses scope=()
root = StreamMux(factories=[CountingTransformer], is_async=False)

# A subgraph "sub_node" gets its own child mux
child = root._make_child(scope=("sub_node",))
assert child.scope == ("sub_node",)
assert child._assign_seq is False  # child never assigns seq
```

---

## 5 · `Call` — functional API call object

**Module:** `langgraph.pregel._algo`  
**Import:** `from langgraph.pregel._algo import Call`

`Call` is the internal representation of an invocation created by `call()` from the functional API. When a `@task` function calls another function via `call()`, LangGraph wraps the invocation in a `Call` object that carries the callable, its arguments, and its per-call policies (retry, cache, timeout).

### Source

```python
class Call:
    __slots__ = (
        "func",
        "input",
        "retry_policy",
        "cache_policy",
        "callbacks",
        "timeout",
    )

    func: Callable
    input: tuple[tuple[Any, ...], dict[str, Any]]
    """Positional args and keyword args, stored as (args, kwargs)."""

    retry_policy: Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    callbacks: Callbacks
    timeout: TimeoutPolicy | None

    def __init__(
        self,
        func: Callable,
        input: tuple[tuple[Any, ...], dict[str, Any]],
        *,
        retry_policy: Sequence[RetryPolicy] | None,
        cache_policy: CachePolicy | None,
        callbacks: Callbacks,
        timeout: TimeoutPolicy | None = None,
    ) -> None: ...
```

### How `call()` creates `Call` objects

The public `call()` function (from `langgraph._internal._future`) is the user-facing way to create `Call` objects inside `@task` functions. It wraps the target callable and its arguments into a `Call` and schedules it within the current execution context.

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, CachePolicy
from langgraph.checkpoint.memory import InMemorySaver


@task(
    retry=RetryPolicy(max_attempts=3),
    cache=CachePolicy(ttl=60),
)
def fetch_data(url: str) -> dict:
    """Each call gets its own retry and cache policy."""
    import httpx
    return httpx.get(url).json()


@task
def transform(data: dict) -> str:
    return str(data.get("id", "unknown"))


@entrypoint(checkpointer=InMemorySaver())
def pipeline(urls: list[str]) -> list[str]:
    # fetch_data.submit() creates Call objects internally
    futures = [fetch_data.submit(u) for u in urls]
    raw = [f.result() for f in futures]
    return [transform(r) for r in raw]
```

### Understanding `Call` policies at the task level

The `retry_policy`, `cache_policy`, and `timeout` on `Call` override the defaults set on the `@task` decorator for that specific invocation. This lets you apply different policies to different invocations of the same task:

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy
from langgraph._internal._future import call     # low-level API
from langgraph.checkpoint.memory import InMemorySaver


@task
def http_call(url: str) -> dict:
    import httpx
    return httpx.get(url, timeout=5).json()


@entrypoint(checkpointer=InMemorySaver())
def graph_fn(urls: list[str]) -> list[dict]:
    results = []
    for url in urls:
        # Use call() to set per-invocation policies
        fut = call(
            http_call,
            args=(url,),
            kwargs={},
            retry_policy=[RetryPolicy(max_attempts=5, backoff_factor=1.5)],
            cache_policy=CachePolicy(ttl=300),
            timeout=TimeoutPolicy(run_timeout=10.0),
        )
        results.append(fut.result())
    return results
```

### Inspecting `Call` objects in tests

Since `Call` carries all invocation metadata, you can inspect it in unit tests or middleware to verify the policies applied to a specific invocation.

```python
from langgraph.pregel._algo import Call
from langgraph.types import RetryPolicy, CachePolicy

def make_call(func, *args, **kwargs) -> Call:
    """Helper that builds a Call object for testing."""
    return Call(
        func=func,
        input=(args, kwargs),
        retry_policy=[RetryPolicy(max_attempts=3)],
        cache_policy=CachePolicy(ttl=60),
        callbacks=[],
        timeout=None,
    )


def my_function(x: int) -> int:
    return x * 2


c = make_call(my_function, 21)
assert c.func is my_function
assert c.input == ((21,), {})
assert c.retry_policy[0].max_attempts == 3
assert c.cache_policy.ttl == 60
```

---

## 6 · `ChannelWrite` + `ChannelWriteEntry`

**Module:** `langgraph.pregel._write`  
**Import:** `from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry`

`ChannelWrite` is the runnable that sits at the tail of every compiled node's execution chain. It intercepts the node's return value and dispatches writes to the channel system via `CONFIG_KEY_SEND`. `ChannelWriteEntry` is the `NamedTuple` that describes a single channel write.

### `ChannelWriteEntry` source

```python
class ChannelWriteEntry(NamedTuple):
    channel: str
    """Channel name to write to."""

    value: Any = PASSTHROUGH
    """Value to write, or PASSTHROUGH to use the node's return value."""

    skip_none: bool = False
    """Whether to skip writing if the value resolves to None."""

    mapper: Callable | None = None
    """Optional transform applied to the value before writing to the channel."""
```

### `ChannelWrite` source (condensed)

```python
class ChannelWrite(RunnableCallable):
    """Implements the logic for sending writes to CONFIG_KEY_SEND."""

    writes: list[ChannelWriteEntry | ChannelWriteTupleEntry | Send]

    def __init__(
        self,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        *,
        tags: Sequence[str] | None = None,
    ): ...

    @staticmethod
    def do_write(
        config: RunnableConfig,
        writes: Sequence[ChannelWriteEntry | ChannelWriteTupleEntry | Send],
        allow_passthrough: bool = True,
    ) -> None:
        """Dispatch writes to the channel system imperatively."""

    @staticmethod
    def is_writer(runnable: Runnable) -> bool:
        """True if a runnable is a ChannelWrite or a custom write decorator."""

    @staticmethod
    def get_static_writes(runnable: Runnable) -> Sequence[tuple[str, Any, str | None]] | None:
        """Return statically-declared writes for graph analysis."""
```

### How node outputs become channel writes

When you return `{"count": 5}` from a node, the compiled graph wraps your node callable in a pipeline:

```
your_node_fn  →  ChannelWrite(writes=[ChannelWriteEntry("count", PASSTHROUGH)])
```

The `PASSTHROUGH` sentinel means "use whatever the previous step returned". The `ChannelWrite` replaces `PASSTHROUGH` with the actual return value at call time.

```python
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry, PASSTHROUGH

# This is what the compiled graph builds internally when you add a node
# that returns {"count": ..., "messages": ...}
write = ChannelWrite(
    writes=[
        ChannelWriteEntry(channel="count",    value=PASSTHROUGH),
        ChannelWriteEntry(channel="messages", value=PASSTHROUGH),
    ]
)
print(write.get_name())
# ChannelWrite<count,messages>
```

### Using `do_write` for direct imperative writes

`ChannelWrite.do_write()` is a static method you can call from within a node to write to channels imperatively, bypassing the return-value mechanism. This is useful when you need to write to channels that are not part of your node's return type.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: dict


def dual_write_node(state: State, config: RunnableConfig) -> dict:
    # Write metadata channel imperatively (outside the return dict)
    ChannelWrite.do_write(
        config,
        [ChannelWriteEntry(channel="metadata", value={"processed_at": "2026-01"})],
        allow_passthrough=False,
    )
    # Write messages via normal return
    return {"messages": [AIMessage("done")]}


builder = StateGraph(State)
builder.add_node("dual", dual_write_node)
builder.add_edge(START, "dual")
builder.add_edge("dual", END)
graph = builder.compile()
result = graph.invoke({"messages": [], "metadata": {}})
print(result["metadata"])   # {'processed_at': '2026-01'}
```

### `ChannelWriteEntry` with a `mapper`

Use `mapper` to transform a value before it is written to a channel. This is how the compiled graph handles state schemas that include transformation annotations.

```python
from langgraph.pregel._write import ChannelWriteEntry

# Write the length of a list to a "count" channel
entry = ChannelWriteEntry(
    channel="count",
    value=["a", "b", "c"],
    mapper=len,        # len(["a", "b", "c"]) == 3 is what gets written
)

# Write nothing if the value is None (useful for optional fields)
optional_entry = ChannelWriteEntry(
    channel="result",
    value=None,
    skip_none=True,    # this write is silently dropped
)
```

---

## 7 · `PregelRunner` + `FuturesDict`

**Module:** `langgraph.pregel._runner`  
**Import:** `from langgraph.pregel._runner import PregelRunner, FuturesDict`

`PregelRunner` is responsible for executing a set of Pregel tasks concurrently within a single superstep, committing their writes to channels, and yielding control to the `PregelLoop` when output is ready. `FuturesDict` is a `dict` subclass that tracks in-flight futures and signals a `threading.Event` when all futures are done.

### `FuturesDict` source

```python
class FuturesDict(Generic[F, E], dict[F, PregelExecutableTask | None]):
    event: E                                 # threading.Event or asyncio.Event
    callback: weakref.ref[Callable[...]]     # called when a future completes
    should_stop: Callable[[set[F]], bool]    # early-stop predicate
    counter: int                             # number of in-flight futures
    done: set[F]                             # futures that have resolved

    def __setitem__(self, key: F, value: PregelExecutableTask | None) -> None:
        """Register a new future and attach an on_done callback."""

    def on_done(self, task: PregelExecutableTask, fut: F) -> None:
        """Called automatically when a future resolves.
        Decrements the counter and sets event when all futures are done
        (or when the stop condition is met)."""
```

### `PregelRunner` source (condensed)

```python
class PregelRunner:
    """Responsible for executing a set of Pregel tasks concurrently,
    committing their writes, yielding control to caller when there is
    output to emit, and interrupting other tasks if appropriate."""

    def __init__(
        self,
        *,
        submit: weakref.ref[Submit],
        put_writes: weakref.ref[Callable[[str, Sequence[tuple[str, Any]]], None]],
        use_astream: bool = False,
        node_finished: Callable[[str], None] | None = None,
        node_error_handler_map: Mapping[str, str] | None = None,
        schedule_error_handler: Callable[...] | None = None,
        aschedule_error_handler: Callable[...] | None = None,
    ) -> None: ...

    def tick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        get_waiter: Callable[[], concurrent.futures.Future] | None = None,
        schedule_task: Callable[...],
    ) -> Iterator[None]:
        """Execute tasks concurrently; yield once per completed batch."""

    async def atick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        schedule_task: Callable[...],
    ) -> AsyncIterator[None]:
        """Async variant of tick()."""
```

### Execution flow through `PregelRunner.tick()`

```
PregelLoop.tick()
  │
  ├── Calls PregelRunner.tick(tasks)
  │     │
  │     ├── Submits each task to the thread pool via Submit
  │     │     (each task runs in its own thread)
  │     │
  │     ├── Waits on FuturesDict.event until all futures resolve
  │     │     or should_stop condition is met
  │     │
  │     ├── For each resolved future:
  │     │     • FuturesDict.on_done() → commit.callback()
  │     │     • Writes from task are committed to channels
  │     │
  │     └── yield → control returns to PregelLoop
  │
  └── PregelLoop reads updated channels, decides next superstep
```

### Error routing with `node_error_handler_map`

When `add_node(..., error_handler=handler_fn)` is used, `PregelRunner` builds a `node_error_handler_map` that routes failures to the handler node instead of propagating them as fatal exceptions. `_handled_exception_ids` tracks exception objects that have been routed so the stop predicate doesn't treat them as fatal.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import Command
from typing_extensions import TypedDict


class State(TypedDict):
    value: int
    error_msg: str | None


def risky_node(state: State) -> dict:
    if state["value"] < 0:
        raise ValueError(f"Negative value: {state['value']}")
    return {"value": state["value"] * 2}


def error_handler(state: State, error: NodeError) -> Command:
    # error.node = "risky", error.error = ValueError(...)
    return Command(
        update={"error_msg": f"handled: {error.error}", "value": 0},
        goto=END,
    )


builder = StateGraph(State)
builder.add_node("risky", risky_node, error_handler=error_handler)
builder.add_edge(START, "risky")
builder.add_edge("risky", END)
graph = builder.compile()

# Triggers the error handler path
result = graph.invoke({"value": -5, "error_msg": None})
print(result)  # {'value': 0, 'error_msg': 'handled: Negative value: -5'}
```

### `should_stop` predicate and early cancellation

When a fatal (unhandled) exception occurs in one task, `FuturesDict.should_stop` returns `True`, which sets the event immediately and causes `tick()` to wake up and propagate the error — without waiting for sibling tasks to finish.

```python
# From langgraph.pregel._runner
def _should_stop_others(
    done: set,
    *,
    handled_exception_ids: set[int],
) -> bool:
    """Return True if any resolved future has a fatal (unhandled) exception.
    Exceptions routed to a node error handler are excluded."""
    for fut in done:
        exc = _exception(fut)
        if exc is not None and id(exc) not in handled_exception_ids:
            return True
    return False
```

---

## 8 · `WritesProtocol` + `PregelTaskWrites`

**Module:** `langgraph.pregel._algo`  
**Import:** `from langgraph.pregel._algo import WritesProtocol, PregelTaskWrites`

`WritesProtocol` is the structural protocol that every object containing pending channel writes must implement. `PregelTaskWrites` is its simplest concrete implementation — a `NamedTuple` used for writes that don't originate from a runnable task (graph input, `update_state`, etc.).

### Source

```python
class WritesProtocol(Protocol):
    """Protocol for objects containing writes to be applied to checkpoint.
    Implemented by PregelTaskWrites and PregelExecutableTask."""

    @property
    def path(self) -> tuple[str | int | tuple, ...]: ...
    """Hierarchical path of the write origin. Used for deterministic sorting."""

    @property
    def name(self) -> str: ...
    """Name of the originating node or virtual write source."""

    @property
    def writes(self) -> Sequence[tuple[str, Any]]: ...
    """Sequence of (channel_name, value) pairs."""

    @property
    def triggers(self) -> Sequence[str]: ...
    """Channel names that triggered this write batch."""


class PregelTaskWrites(NamedTuple):
    """Simplest WritesProtocol implementation.
    Used for writes that don't originate from a runnable task."""

    path: tuple[str | int | tuple, ...]
    name: str
    writes: Sequence[tuple[str, Any]]
    triggers: Sequence[str]
```

### When `PregelTaskWrites` is used vs `PregelExecutableTask`

`PregelExecutableTask` (the full task object with a runnable, config, retry policy, etc.) implements `WritesProtocol` for node executions. `PregelTaskWrites` is used for simpler, non-runnable write batches:

| Use case | Type used |
|---|---|
| Normal node execution | `PregelExecutableTask` (implements `WritesProtocol`) |
| Graph input writes (first superstep) | `PregelTaskWrites` |
| `update_state()` calls | `PregelTaskWrites` |
| `bulk_update_state()` calls | `PregelTaskWrites` |
| Error handler synthetic writes | `PregelTaskWrites` |

### `apply_writes` and the `WritesProtocol`

`apply_writes` (also in `langgraph.pregel._algo`) processes a sequence of `WritesProtocol` objects to update checkpoint versions and channel state. Understanding the protocol helps when writing custom channel or checkpointer implementations.

```python
from langgraph.pregel._algo import PregelTaskWrites, apply_writes
from langgraph.graph import StateGraph, START, END
from langgraph.channels.last_value import LastValue
from langgraph.checkpoint.base import empty_checkpoint
from typing_extensions import TypedDict


class State(TypedDict):
    count: int
    label: str


# Simulate how update_state writes are structured internally
# path=() for root graph, name="__update_state__", triggers=[]
write_batch = PregelTaskWrites(
    path=(0,),               # synthetic step path
    name="__update_state__",
    writes=[("count", 42), ("label", "hello")],
    triggers=[],             # no triggers — this is an external write
)

# You can unpack it just like any NamedTuple
path, name, writes, triggers = write_batch
assert name == "__update_state__"
assert dict(writes) == {"count": 42, "label": "hello"}
```

### Sorting by `path` for deterministic write order

`apply_writes` sorts all `WritesProtocol` objects by `path` before applying them. This guarantees that writes from the null task (e.g. input, `update_state`) are applied before writes from running nodes, and that sibling node writes are always applied in the same order even when nodes run concurrently.

```python
from langgraph.pregel._algo import PregelTaskWrites

# Paths sort lexicographically: (0,) < (1,) < (1, 0)
writes = [
    PregelTaskWrites(path=(1, 0), name="node_a", writes=[("x", 10)], triggers=["start"]),
    PregelTaskWrites(path=(0,),   name="__input__", writes=[("x", 0)], triggers=[]),
    PregelTaskWrites(path=(1,),   name="node_b", writes=[("y", 20)], triggers=["start"]),
]
sorted_writes = sorted(writes, key=lambda t: t.path[:3])
print([w.name for w in sorted_writes])
# ['__input__', 'node_b', 'node_a']
```

---

## 9 · `SyncPregelLoop` + `AsyncPregelLoop`

**Module:** `langgraph.pregel._loop`  
**Import:** `from langgraph.pregel._loop import SyncPregelLoop, AsyncPregelLoop`

`PregelLoop` (and its two concrete subclasses) is the state machine that drives a single graph run from start to finish. It manages checkpoint loading/saving, task scheduling, interrupt handling, streaming, and cooperative drain.

### `PregelLoop` status machine

```python
class PregelLoop:
    status: Literal[
        "input",          # Waiting to read the first input
        "pending",        # Tasks are scheduled but not yet run
        "done",           # Graph has reached END or an exit condition
        "draining",       # RunControl.request_drain() was called; waiting to exit
        "interrupt_before",  # Graph interrupted before a node
        "interrupt_after",   # Graph interrupted after a node
        "out_of_steps",   # recursion_limit reached
    ]
```

State transitions:

```
         input
           │
           ▼
        pending ──► done
           │         ▲
           │    interrupt_before / interrupt_after
           │         ▲
           ▼         │
       [run tasks] ──┘
           │
           ▼
       draining ──► done
           │
           ▼
       out_of_steps ──► done (raises GraphRecursionError)
```

### `SyncPregelLoop` and `AsyncPregelLoop` as context managers

Both classes implement the context manager protocol. The `__enter__`/`__aenter__` method loads the checkpoint; `__exit__`/`__aexit__` saves the final checkpoint and cleans up.

```python
from langgraph.pregel._loop import SyncPregelLoop
# (Internal usage — shown for understanding; use graph.invoke() in production)

# SyncPregelLoop.__enter__ does:
# 1. Load checkpoint from checkpointer.get_tuple(config)
# 2. Initialize channels from checkpoint
# 3. Map graph input to channel writes
# 4. Set status = "pending" if there are tasks, "done" if not

# SyncPregelLoop.__exit__ does:
# 1. Save the final checkpoint (if durability="sync" or "exit")
# 2. Close the stream mux
# 3. Run post-run lifecycle callbacks
```

### Observing loop status changes

The `status` attribute is public. You can read it from outside the loop to understand the current execution phase. The `output` attribute holds the graph's final output once `status == "done"`.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    value: int
    approved: bool


def human_review(state: State) -> dict:
    approved = interrupt({"value": state["value"], "needs_approval": True})
    return {"approved": approved}


def process(state: State) -> dict:
    return {"value": state["value"] * 2}


builder = StateGraph(State)
builder.add_node("review", human_review)
builder.add_node("process", process)
builder.add_edge(START, "review")
builder.add_edge("review", "process")
builder.add_edge("process", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "loop-demo"}}

# First run — will interrupt at "review"
try:
    result = graph.invoke({"value": 10, "approved": False}, config)
except Exception:
    pass

# Check the current state — status would be "interrupt_before" or "interrupt_after"
snapshot = graph.get_state(config)
print(f"Next nodes: {snapshot.next}")         # ('review',) or ('process',)
print(f"Interrupted: {bool(snapshot.tasks)}")

# Resume by providing the interrupt answer
resumed = graph.invoke(
    {"approved": True},
    {**config, **{"configurable": {**config["configurable"], "checkpoint_id": None}}},
)
# Or simply:
graph.invoke(None, config)
```

### Cooperative drain via `RunControl`

`RunControl.request_drain()` sets a flag that the `PregelLoop` checks at each superstep boundary. When drain is requested, the loop transitions to `"draining"` and exits after the current superstep completes — no abrupt cancellation.

```python
import threading
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime, RunControl
from langgraph.types import TimeoutPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    steps: int


def long_node(state: State, runtime: Runtime) -> dict:
    for i in range(100):
        if runtime.drain_requested:
            print(f"Drain requested ({runtime.drain_reason}), stopping at step {i}")
            break
        runtime.heartbeat()
    return {"steps": i}


builder = StateGraph(State)
builder.add_node("work", long_node)
builder.add_edge(START, "work")
builder.add_edge("work", END)
graph = builder.compile()

# In a real deployment, RunControl is passed via the runtime at compile time.
# The graph exits cleanly when drain is signalled.
```

### `PregelLoop` field reference (partial)

| Field | Type | Description |
|---|---|---|
| `status` | `Literal[...]` | Current loop status (see state machine above) |
| `step` | `int` | Current superstep counter (0-indexed) |
| `stop` | `int` | Maximum superstep (from `recursion_limit`) |
| `tasks` | `dict[str, PregelExecutableTask]` | Tasks scheduled for the current superstep |
| `output` | `Any \| None` | Final output; populated when `status == "done"` |
| `channels` | `Mapping[str, BaseChannel]` | Live channel state |
| `checkpoint` | `Checkpoint` | Current checkpoint snapshot |
| `durability` | `Durability` | `"sync"`, `"async"`, or `"exit"` |

---

## 10 · `DuplexStream`

**Module:** `langgraph.pregel._loop`  
**Import:** `from langgraph.pregel._loop import DuplexStream`

`DuplexStream` is a function (not a class) that combines multiple `StreamProtocol` instances into a single `StreamProtocol` that fans out to all of them. When a `StreamChunk` arrives, `DuplexStream` dispatches it to every underlying protocol whose `modes` set includes the chunk's mode.

### Source

```python
def DuplexStream(*streams: StreamProtocol) -> StreamProtocol:
    def __call__(value: StreamChunk) -> None:
        for stream in streams:
            if value[1] in stream.modes:   # value[1] is the StreamMode
                stream(value)

    return StreamProtocol(__call__, {mode for s in streams for mode in s.modes})
```

The returned `StreamProtocol` has a `modes` set that is the **union** of all input protocols' modes. Each chunk is dispatched only to protocols that include its mode — protocols that don't care about a particular mode are skipped.

### Why `DuplexStream` exists

The graph can be asked to stream in multiple modes simultaneously (e.g. `stream_mode=["values", "updates"]`). Each mode maps to a separate `StreamProtocol`. `DuplexStream` merges them into a single callable that the internal streaming pipeline can treat as one protocol.

```python
from langgraph.pregel.protocol import StreamProtocol, StreamChunk
from langgraph.pregel._loop import DuplexStream

# Build two separate stream handlers
values_log: list[StreamChunk] = []
updates_log: list[StreamChunk] = []

values_proto = StreamProtocol(
    lambda chunk: values_log.append(chunk),
    modes={"values"},
)
updates_proto = StreamProtocol(
    lambda chunk: updates_log.append(chunk),
    modes={"updates"},
)

# Combine them — the duplex protocol handles both modes
duplex = DuplexStream(values_proto, updates_proto)
assert duplex.modes == {"values", "updates"}

# Simulate a chunk arriving in "values" mode
values_chunk: StreamChunk = ((), "values", {"count": 1})
duplex(values_chunk)
assert len(values_log) == 1    # received by values_proto
assert len(updates_log) == 0   # skipped by updates_proto

# Simulate a chunk in "updates" mode
updates_chunk: StreamChunk = ((), "updates", {"count": 1})
duplex(updates_chunk)
assert len(values_log) == 1    # skipped by values_proto
assert len(updates_log) == 1   # received by updates_proto
```

### Composing multiple stream sinks

`DuplexStream` is useful when you want to fan out a stream to multiple independent consumers — for example, a logging sink and a WebSocket sink.

```python
import json
from pathlib import Path
from langgraph.pregel.protocol import StreamProtocol, StreamChunk
from langgraph.pregel._loop import DuplexStream
from langgraph.graph import StateGraph, START, END
from langgraph.stream.run_stream import GraphRunStream
from typing_extensions import TypedDict


class State(TypedDict):
    count: int


def inc(state: State) -> dict:
    return {"count": state["count"] + 1}


# Sink 1: append to a log file
log_path = Path("/tmp/stream.log")
def log_sink(chunk: StreamChunk) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps({"mode": chunk[1], "data": str(chunk[2])}) + "\n")

# Sink 2: collect for later processing
collected: list[StreamChunk] = []
collect_sink = collected.append

values_proto = StreamProtocol(log_sink, modes={"values"})
collect_proto = StreamProtocol(collect_sink, modes={"values", "updates"})
duplex = DuplexStream(values_proto, collect_proto)

# Build a simple graph and stream with the duplex protocol
builder = StateGraph(State)
builder.add_node("inc", inc)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)
graph = builder.compile()

# GraphRunStream.stream_mode accepts the protocol directly in advanced usage
for chunk in graph.stream(
    {"count": 0},
    stream_mode=["values", "updates"],
):
    pass   # the graph's internal DuplexStream fans out to its own protocols

# Fan-out to your own sinks using the public streaming API:
for chunk in graph.stream({"count": 0}, stream_mode="values"):
    values_proto(((), "values", chunk))
    collect_proto(((), "values", chunk))
```

### Fan-in with `DuplexStream` + mode filtering

You can build a monitoring stream that subscribes to every mode and routes events differently based on their mode:

```python
from langgraph.pregel.protocol import StreamProtocol, StreamChunk
from langgraph.pregel._loop import DuplexStream
from langgraph.types import StreamMode


def make_routing_duplex(
    handler_map: dict[StreamMode, list[StreamProtocol]]
) -> StreamProtocol:
    """Build a DuplexStream that routes each mode to a different set of handlers."""
    all_protos: list[StreamProtocol] = []
    for mode, handlers in handler_map.items():
        for h in handlers:
            all_protos.append(h)
    return DuplexStream(*all_protos)


debug_log: list = []
metrics_log: list = []

debug_proto  = StreamProtocol(debug_log.append,   modes={"debug"})
values_proto = StreamProtocol(metrics_log.append, modes={"values"})
custom_proto = StreamProtocol(metrics_log.append, modes={"custom"})

duplex = make_routing_duplex({
    "debug":  [debug_proto],
    "values": [values_proto],
    "custom": [custom_proto],
})

assert "debug" in duplex.modes
assert "values" in duplex.modes
assert "custom" in duplex.modes
```

---

## Summary table

| Class / Function | Module | Available since | User-facing? |
|---|---|---|---|
| `ExecutionInfo` | `langgraph.runtime` | v0.6 (extended v1.2) | Yes — via `runtime.execution_info` |
| `Runtime.heartbeat()` | `langgraph.runtime` | v1.2 | Yes — call inside long-running nodes |
| `ServerInfo` | `langgraph.runtime` | v0.6 | Yes — populated on LangGraph Platform |
| `ReplayState` | `langgraph._internal._replay` | v1.0 | No — internal time-travel state |
| `StreamMux` | `langgraph.stream._mux` | v1.2 | Advanced — custom transformer pipelines |
| `Call` | `langgraph.pregel._algo` | v0.6 | No — internal functional API representation |
| `ChannelWrite` + `ChannelWriteEntry` | `langgraph.pregel._write` | v0.1 | Advanced — `do_write()` is useful |
| `PregelRunner` + `FuturesDict` | `langgraph.pregel._runner` | v0.1 | No — internal concurrency layer |
| `WritesProtocol` + `PregelTaskWrites` | `langgraph.pregel._algo` | v0.1 | No — internal write protocol |
| `SyncPregelLoop` / `AsyncPregelLoop` | `langgraph.pregel._loop` | v0.1 | No — internal execution state machine |
| `DuplexStream` | `langgraph.pregel._loop` | v0.1 | Advanced — multi-sink streaming |

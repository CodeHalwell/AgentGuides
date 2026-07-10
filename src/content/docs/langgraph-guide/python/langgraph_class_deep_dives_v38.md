---
title: "LangGraph Class Deep-Dives Vol. 38"
description: "Source-verified deep dives (langgraph==1.2.9) into 10 class groups: GraphCallbackHandler/GraphInterruptEvent/GraphResumeEvent (graph lifecycle observability via config callbacks), Runtime/RunControl/ExecutionInfo/ServerInfo (node runtime bundle — context/store/stream_writer/heartbeat injection and cooperative drain signalling), ToolRuntime/InjectedState/InjectedStore (tool-layer runtime injection and invisible argument annotations), StreamMux (central stream event dispatcher — transformer wiring, projection ownership, child mini-mux cloning), GraphRunStream/SubgraphRunStream (v3 caller-driven pump — native projection attributes, wire_pump handshake), CheckpointsTransformer/LifecycleTransformer/CustomTransformer (native stream transformers — scope-filtered channel log push, LifecyclePayload started/completed events), NodeBuilder (fluent Pregel node construction — subscribe_only/subscribe_to/read_from/do/write_to/add_retry_policies/add_cache_policy/set_timeout chaining), CachePolicy/CacheKey/TimeoutPolicy (node-level caching with key_func/ttl and idle/run timeout policies with refresh_on semantics), NodeTimeoutError/NodeCancelledError/GraphRecursionError (error taxonomy — idle vs run timeout kind, user-raised CancelledError wrapping, recursion_limit config), and UIMessage/RemoveUIMessage (generative UI streaming — type/id/name/props/metadata dict shape, remove-ui complement)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 38"
  order: 69
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.9` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `GraphCallbackHandler` · `GraphInterruptEvent` · `GraphResumeEvent`

**Module:** `langgraph.callbacks`

`GraphCallbackHandler` is the base class you subclass to observe **graph-level lifecycle transitions** — specifically the two events that are unique to LangGraph's interrupt/resume model and are not surfaced by the generic LangChain `BaseCallbackHandler` hierarchy. Instances are registered via `config["callbacks"]`; the internal `_GraphCallbackManager` filters the callback stack to only dispatch these events to handlers that inherit from `GraphCallbackHandler`.

**Key source facts** (from `langgraph/callbacks/__init__.py`):

- `GraphCallbackHandler` extends `langchain_core.callbacks.BaseCallbackHandler`. Both `on_interrupt` and `on_resume` default to a no-op `return None`, so you only override what you need.
- `GraphInterruptEvent` is a frozen dataclass carrying `run_id: UUID | None`, `status: GraphLifecycleStatus`, `checkpoint_id: str`, `checkpoint_ns: tuple[str, ...]`, and `interrupts: tuple[Interrupt, ...]`. The `interrupts` tuple holds every `Interrupt` object raised during the paused step.
- `GraphResumeEvent` is a frozen dataclass with the same `run_id`, `status`, `checkpoint_id`, and `checkpoint_ns` fields, emitted when the graph's Pregel loop detects it is resuming from a stored checkpoint rather than starting fresh.
- `_GraphCallbackManager.on_interrupt` / `on_resume` call `handle_event(self.handlers, ...)` — only handlers whose type is a subclass of `GraphCallbackHandler` receive the dispatch.
- Pass handlers as a list in `config["callbacks"]` or via `RunnableConfig`'s `callbacks` key. Multiple handlers are all invoked in registration order.

### Example 1 — log every interrupt to stdout with its checkpoint ID

```python
from uuid import UUID
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


class AuditHandler(GraphCallbackHandler):
    """Print interrupt and resume events with their checkpoint IDs."""

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        print(
            f"[INTERRUPT] run_id={event.run_id} "
            f"checkpoint={event.checkpoint_id[:8]}… "
            f"interrupts={len(event.interrupts)}"
        )

    def on_resume(self, event: GraphResumeEvent) -> None:
        print(
            f"[RESUME]    run_id={event.run_id} "
            f"checkpoint={event.checkpoint_id[:8]}…"
        )


class State(TypedDict):
    value: str


def review_step(state: State) -> dict:
    approved = interrupt({"question": "Approve?", "value": state["value"]})
    return {"value": f"approved:{approved}"}


builder = StateGraph(State)
builder.add_node("review", review_step)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["review"])

handler = AuditHandler()
cfg = {"configurable": {"thread_id": "t1"}, "callbacks": [handler]}

# First invoke — hits the interrupt
try:
    graph.invoke({"value": "hello"}, config=cfg)
except Exception:
    pass  # interrupt raises internally in some execution paths

# Resume — triggers on_resume on the handler
graph.invoke(None, config=cfg)
```

### Example 2 — collect interrupt payloads for structured inspection

```python
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent
from langgraph.types import Interrupt


class PayloadCollector(GraphCallbackHandler):
    def __init__(self) -> None:
        self.collected: list[tuple[str, list[Interrupt]]] = []

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        self.collected.append(
            (event.checkpoint_id, list(event.interrupts))
        )


from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


class S(TypedDict):
    step: int


def node_a(state: S) -> dict:
    interrupt({"msg": "pause at A"})
    return {}


collector = PayloadCollector()
builder = StateGraph(S)
builder.add_node("a", node_a)
builder.add_edge(START, "a")
builder.add_edge("a", END)

graph = builder.compile(
    checkpointer=MemorySaver(), interrupt_before=["a"]
)
cfg = {"configurable": {"thread_id": "t2"}, "callbacks": [collector]}
graph.invoke({"step": 0}, config=cfg)

print(f"Interrupts captured: {len(collector.collected)}")
for cid, interrupts in collector.collected:
    for iv in interrupts:
        print(f"  checkpoint={cid[:8]}… value={iv.value}")
```

### Example 3 — multiple handlers on the same graph run

```python
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent


class CountingHandler(GraphCallbackHandler):
    def __init__(self, name: str) -> None:
        self.name = name
        self.interrupt_count = 0
        self.resume_count = 0

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        self.interrupt_count += 1
        print(f"{self.name}: interrupt #{self.interrupt_count}")

    def on_resume(self, event: GraphResumeEvent) -> None:
        self.resume_count += 1
        print(f"{self.name}: resume #{self.resume_count}")


from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


class S2(TypedDict):
    x: int


def pause_node(state: S2) -> dict:
    interrupt("waiting")
    return {"x": state["x"] + 1}


h1, h2 = CountingHandler("H1"), CountingHandler("H2")

builder = StateGraph(S2)
builder.add_node("pause", pause_node)
builder.add_edge(START, "pause")
builder.add_edge("pause", END)
graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["pause"])

cfg = {"configurable": {"thread_id": "t3"}, "callbacks": [h1, h2]}
graph.invoke({"x": 0}, config=cfg)
# Resume
graph.invoke(None, config=cfg)

print(f"H1 resumes: {h1.resume_count}, H2 resumes: {h2.resume_count}")
```

---

## 2 · `Runtime` · `RunControl` · `ExecutionInfo` · `ServerInfo`

**Module:** `langgraph.runtime`

`Runtime` is the **convenience bundle** injected into graph nodes and middleware when a parameter named `runtime` carries a `Runtime[ContextT]` type annotation. It aggregates five orthogonal services so nodes avoid importing multiple individual helpers. `RunControl` is its cooperative-drain signal plane. `ExecutionInfo` provides read-only task metadata. `ServerInfo` carries LangGraph Server deployment metadata.

**Key source facts** (from `langgraph/runtime/__init__.py`):

- `Runtime` is a frozen `@dataclass(Generic[ContextT])` with fields: `context` (static per-run dependency injection), `store` (`BaseStore | None`), `stream_writer` (custom stream sink), `heartbeat` (refreshes `idle_timeout`), `previous` (functional API last return value), `execution_info` (`ExecutionInfo | None`), `server_info` (`ServerInfo | None`), and `control` (`RunControl | None`).
- `Runtime.merge(other)` produces a new `Runtime` taking non-default values from `other`, preserving `self` values where `other` carries the sentinel `_no_op_*` defaults.
- `Runtime.override(**overrides)` calls `dataclasses.replace(self, **overrides)` — convenient for test patching.
- `RunControl.__slots__ = ("_drain_reason",)`. `request_drain(reason="shutdown")` is a single attribute write — safe from any thread without a lock. `drain_requested` and `drain_reason` are read-only properties.
- `ExecutionInfo` is a frozen, slotted dataclass: `checkpoint_id`, `checkpoint_ns`, `task_id`, `thread_id | None`, `run_id | None`, `node_attempt: int` (1-indexed), `node_first_attempt_time: float | None`. `patch(**overrides)` returns a new instance via `dataclasses.replace`.
- `ServerInfo` carries `assistant_id`, `graph_id`, and `user: BaseUser | None`. Only populated when running under a LangGraph Server (cloud or self-hosted) deployment; `None` in open-source usage.

### Example 1 — inject `Runtime` for context + store access in a node

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class UserCtx:
    user_id: str


class State(TypedDict):
    greeting: str


store = InMemoryStore()
store.put(("profiles",), "alice", {"name": "Alice", "lang": "en"})


def greet(state: State, runtime: Runtime[UserCtx]) -> dict:
    uid = runtime.context.user_id
    profile = None
    if runtime.store:
        item = runtime.store.get(("profiles",), uid)
        if item:
            profile = item.value
    name = profile["name"] if profile else uid
    return {"greeting": f"Hello, {name}!"}


graph = (
    StateGraph(state_schema=State, context_schema=UserCtx)
    .add_node("greet", greet)
    .set_entry_point("greet")
    .set_finish_point("greet")
    .compile(store=store)
)

result = graph.invoke({}, context=UserCtx(user_id="alice"))
print(result["greeting"])  # Hello, Alice!
```

### Example 2 — use `ExecutionInfo` to log task metadata and detect retries

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime, ExecutionInfo


class State(TypedDict):
    attempts: list


def instrumented_node(state: State, runtime: Runtime) -> dict:
    info: ExecutionInfo | None = runtime.execution_info
    if info:
        attempt_label = f"attempt #{info.node_attempt}"
        print(
            f"task_id={info.task_id[:8]}… "
            f"thread={info.thread_id} "
            f"{attempt_label}"
        )
        return {"attempts": state.get("attempts", []) + [info.node_attempt]}
    return {"attempts": state.get("attempts", []) + [0]}


graph = (
    StateGraph(State)
    .add_node("work", instrumented_node)
    .add_edge(START, "work")
    .add_edge("work", END)
    .compile()
)

result = graph.invoke({"attempts": []})
print("attempts recorded:", result["attempts"])
```

### Example 3 — cooperative drain via `RunControl`

```python
import threading
from langgraph.runtime import RunControl


# RunControl is safe to use from any thread — drain is a single attribute write
control = RunControl()

print("drain_requested before:", control.drain_requested)  # False

def background_drain():
    control.request_drain(reason="graceful-shutdown")

t = threading.Thread(target=background_drain)
t.start()
t.join()

print("drain_requested after:", control.drain_requested)   # True
print("drain_reason:", control.drain_reason)               # graceful-shutdown

# Nodes check control.drain_requested and exit cleanly
if control.drain_requested:
    print(f"Stopping because: {control.drain_reason}")
```

---

## 3 · `ToolRuntime` · `InjectedState` · `InjectedStore`

**Module:** `langgraph.prebuilt.tool_node`

`ToolRuntime` is the **tool-layer counterpart to `Runtime`**, automatically injected into any `@tool`-decorated function that declares `runtime: ToolRuntime` as a parameter. Unlike `Runtime` (which node functions receive), `ToolRuntime` additionally carries `state`, `tool_call_id`, `config`, and `tools`. `InjectedState` and `InjectedStore` are `Annotated` markers that inject graph state or the store directly into individual tool parameters without exposing those values to the language model.

**Key source facts** (from `langgraph/prebuilt/tool_node.py`):

- `ToolRuntime` is a `@dataclass` that subclasses `_DirectlyInjectedToolArg` and is `Generic[ContextT, StateT]`. Fields: `state`, `context`, `config: RunnableConfig`, `stream_writer`, `tool_call_id: str | None`, `store: BaseStore | None`, `tools: list[BaseTool]`, `execution_info: ExecutionInfo | None`, `server_info: ServerInfo | None`.
- `ToolRuntime.emit_output_delta(delta)` reads the per-tool-call `StreamWriter` set by `StreamToolCallHandler` on a `ContextVar` at `on_tool_start` time and calls it with `delta`. Silent no-op when `stream_mode` does not include `"tools"`.
- `InjectedState(field=None)` — if `field` is a string, only that field's value is sliced from the state dict; if `None`, the whole state is injected. The parameter does **not** appear in the model's tool schema.
- `InjectedStore` — injects the `BaseStore` instance attached to the graph. Also invisible to the model.
- Detection logic in `ToolNode`: parameters annotated with `InjectedState`, `InjectedStore`, or typed as `ToolRuntime` are removed from the JSON schema before binding tool calls, then re-injected from the execution context at call time.

### Example 1 — access full graph state inside a tool with `ToolRuntime`

```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: list
    user_name: str


@tool
def personalized_tool(query: str, runtime: ToolRuntime) -> str:
    """Greet the user by name from graph state."""
    name = runtime.state.get("user_name", "stranger")
    return f"Hello {name}! You asked: {query}"


tool_node = ToolNode([personalized_tool])

# Simulate a state where the LLM has emitted a tool call
fake_tool_call = {
    "id": "call_001",
    "name": "personalized_tool",
    "args": {"query": "what's the weather?"},
    "type": "tool_call",
}
ai_msg = AIMessage(content="", tool_calls=[fake_tool_call])
state: AgentState = {
    "messages": [HumanMessage("hi"), ai_msg],
    "user_name": "Alice",
}

result = tool_node.invoke(state)
last_msg: ToolMessage = result["messages"][-1]
print(last_msg.content)  # Hello Alice! You asked: what's the weather?
```

### Example 2 — inject a specific state field with `InjectedState`

```python
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import InjectedState
from typing_extensions import TypedDict


class ChatState(TypedDict):
    messages: list
    locale: str


@tool
def locale_aware_response(
    text: str,
    locale: Annotated[str, InjectedState("locale")],  # only "locale" key injected
) -> str:
    """Format a response according to the user's locale."""
    return f"[{locale.upper()}] {text}"


tool_node = ToolNode([locale_aware_response])

fake_call = {
    "id": "call_002",
    "name": "locale_aware_response",
    "args": {"text": "Good morning"},  # 'locale' NOT sent by the model
    "type": "tool_call",
}
state: ChatState = {
    "messages": [AIMessage(content="", tool_calls=[fake_call])],
    "locale": "en-gb",
}

result = tool_node.invoke(state)
print(result["messages"][-1].content)  # [EN-GB] Good morning
```

### Example 3 — stream partial tool output with `emit_output_delta`

```python
import asyncio
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime
from typing_extensions import TypedDict


class S(TypedDict):
    messages: list


@tool
def streaming_tool(n: int, runtime: ToolRuntime) -> str:
    """Emit n deltas then return a final summary."""
    for i in range(n):
        # emit_output_delta is a no-op when stream_mode != "tools"
        runtime.emit_output_delta({"chunk": i, "partial": True})
    return f"done: {n} chunks"


tool_node = ToolNode([streaming_tool])

fake_call = {
    "id": "call_003",
    "name": "streaming_tool",
    "args": {"n": 3},
    "type": "tool_call",
}
state: S = {"messages": [AIMessage(content="", tool_calls=[fake_call])]}

# Outside a streaming context emit_output_delta is a silent no-op
result = tool_node.invoke(state)
print(result["messages"][-1].content)  # done: 3 chunks
```

---

## 4 · `StreamMux`

**Module:** `langgraph.stream._mux`

`StreamMux` is the **central routing hub** for LangGraph's streaming infrastructure. It owns the main event log (`StreamChannel[ProtocolEvent]`), orchestrates a pipeline of `StreamTransformer` instances, and merges their projection dicts so that events arrive at the right consumer channel. Child mini-muxes (one per subgraph scope) are cloned from the root via `_make_child`, inheriting `factories` but not pre-built `transformers`.

**Key source facts** (from `langgraph/stream/_mux.py`):

- Constructor accepts either `transformers` (pre-built; local to this mux) or `factories` (callables `(scope) -> StreamTransformer`; propagated to child mini-muxes via `_make_child`). Mixing both is valid.
- Transformers with `before_builtins = True` are registered before all others — used for content-mutation transformers (PII redaction, content filters) that must run before `MessagesTransformer` snapshots text fields.
- `extensions: dict[str, Any]` is the merged projection map across all registered transformers. Treat as read-only.
- `native_keys: set[str]` collects projection keys from transformers with `_native = True` — `GraphRunStream` promotes these to direct attributes.
- `push(method, params)` / `apush(method, params)` route a protocol event through the transformer chain, then append the result to the main event log.
- Projection key collisions between transformers raise `ValueError` at init time.

### Example 1 — inspect projection keys registered on a fresh mux

```python
from langgraph.stream._mux import StreamMux
from langgraph.stream.transformers import (
    ValuesTransformer,
    MessagesTransformer,
    CheckpointsTransformer,
    CustomTransformer,
)

mux = StreamMux(
    factories=[
        lambda scope: ValuesTransformer(scope),
        lambda scope: MessagesTransformer(scope),
        lambda scope: CheckpointsTransformer(scope),
        lambda scope: CustomTransformer(scope),
    ],
    is_async=False,
    scope=(),
)

print("All projection keys:", sorted(mux.extensions.keys()))
print("Native keys:", sorted(mux.native_keys))
# Native keys map 1:1 to GraphRunStream attributes like run.values, run.messages etc.
```

### Example 2 — create a child mux for a subgraph scope

```python
from langgraph.stream._mux import StreamMux
from langgraph.stream.transformers import ValuesTransformer, UpdatesTransformer


root_mux = StreamMux(
    factories=[
        lambda scope: ValuesTransformer(scope),
        lambda scope: UpdatesTransformer(scope),
    ],
    is_async=False,
    scope=(),
)

# Each subgraph gets its own mini-mux with the same transformer factories
# but a distinct scope tuple
child_mux = root_mux._make_child(scope=("subgraph_a",))

print("Root scope:", root_mux.scope)
print("Child scope:", child_mux.scope)
print("Child projection keys:", sorted(child_mux.extensions.keys()))
```

### Example 3 — observe v3 streaming with `stream_events`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    counter: int


def increment(state: S) -> dict:
    return {"counter": state["counter"] + 1}


graph = (
    StateGraph(S)
    .add_node("inc", increment)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile()
)

# stream_events(version="v3") returns a GraphRunStream backed by a StreamMux
try:
    run = graph.stream_events({"counter": 0}, version="v3")
    # Iterate values projection to drive the pump
    for snapshot in run.values:
        print("snapshot:", snapshot)
except Exception as exc:
    # v3 is marked @beta — may raise on older runtimes
    print(f"v3 not available: {type(exc).__name__}: {exc}")
```

---

## 5 · `GraphRunStream` · `SubgraphRunStream`

**Module:** `langgraph.stream.run_stream`

`GraphRunStream` is the **v3 streaming object** returned by `Pregel.stream_events(version="v3")`. Unlike the legacy streaming protocol — which returns a `Generator[StreamMode, None, None]` — v3 gives a single handle object whose named projection attributes (`run.values`, `run.messages`, `run.custom`, etc.) drive the graph forward as the caller iterates them. `SubgraphRunStream` is the nested variant (no independent pump) used for sub-graphs that share the root pump.

**Key source facts** (from `langgraph/stream/run_stream.py`):

- `GraphRunStream.__init__` receives `graph_iter`, `mux`, and `wire_pump=True`. It copies `mux.native_keys` to direct attributes via `setattr(self, key, mux.extensions[key])`, giving callers `run.values`, `run.checkpoints`, etc. without `extensions["values"]` indirection.
- `_wire_request_more(mux)` binds `_pump_next` as the mux's pull callback so that iterating any native projection channel automatically advances the graph.
- `_exhausted: bool` is set when `graph_iter` is consumed; further pump calls are no-ops.
- `_interrupted: bool` and `_interrupts: list` capture in-flight `Interrupt` exceptions from the pump so callers can inspect `run.interrupts` after a mid-step pause.
- `SubgraphRunStream` inherits `_SubgraphRunStreamMixin` and passes `wire_pump=False` — its pump is inherited from the parent mini-mux.
- Projections are **single-consumer** — iterating the same projection twice raises. Use `projection.tee(n)` for genuine fan-out.

### Example 1 — iterate `run.values` to drive a graph with v3 streaming

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    n: int


def double(state: S) -> dict:
    return {"n": state["n"] * 2}


graph = (
    StateGraph(S)
    .add_node("double", double)
    .add_edge(START, "double")
    .add_edge("double", END)
    .compile()
)

try:
    run = graph.stream_events({"n": 3}, version="v3")
    snapshots = list(run.values)
    print("values snapshots:", snapshots)
except Exception as exc:
    print(f"v3 not yet stable: {type(exc).__name__}")
```

### Example 2 — consume multiple projections in parallel via `tee`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    msg: str


def echo(state: S) -> dict:
    return {"msg": state["msg"].upper()}


graph = (
    StateGraph(S)
    .add_node("echo", echo)
    .add_edge(START, "echo")
    .add_edge("echo", END)
    .compile()
)

try:
    run = graph.stream_events({"msg": "hello"}, version="v3")
    # Without tee, iterating values twice raises. tee(2) creates two cursors.
    v1, v2 = run.values.tee(2)
    first = next(iter(v1))
    second = next(iter(v2))
    print("cursor 1:", first)
    print("cursor 2:", second)
except Exception as exc:
    print(f"v3 not yet stable: {type(exc).__name__}")
```

### Example 3 — check `run._exhausted` and `run._interrupted` state

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver


class S(TypedDict):
    value: int


def pause(state: S) -> dict:
    interrupt("paused")
    return {}


graph = (
    StateGraph(S)
    .add_node("pause", pause)
    .add_edge(START, "pause")
    .add_edge("pause", END)
    .compile(checkpointer=MemorySaver(), interrupt_before=["pause"])
)

cfg = {"configurable": {"thread_id": "v3-test"}}
try:
    run = graph.stream_events({"value": 1}, config=cfg, version="v3")
    list(run.values)  # pump until drained or interrupted
    print("exhausted:", run._exhausted)
    print("interrupted:", run._interrupted)
except Exception as exc:
    print(f"v3 not yet stable: {type(exc).__name__}")
```

---

## 6 · `CheckpointsTransformer` · `LifecycleTransformer` · `CustomTransformer`

**Module:** `langgraph.stream.transformers`

These three classes implement the **native stream transformer** pattern: each exposes a `StreamChannel` log under a projection key and overrides `process(event: ProtocolEvent) -> bool` to filter events by `method` and `namespace`. They all set `_native = True`, so `StreamMux` promotes their projection keys to direct attributes on `GraphRunStream`.

**Key source facts** (from `langgraph/stream/transformers.py`):

- All transformers subclass `StreamTransformer` and implement `init() -> dict[str, Any]` (returns `{key: StreamChannel}`), and `process(event) -> bool` (returning `True` continues the pipeline; returning `False` drops the event from later transformers).
- `CheckpointsTransformer` has `required_stream_modes = ("checkpoints",)` — events only arrive when the graph is compiled with a checkpointer and `stream_mode` includes `"checkpoints"`. It filters by `event["method"] == "checkpoints"` and `params["namespace"] == self._scope_list`.
- `LifecycleTransformer` extends `_TasksLifecycleBase` and pushes `LifecyclePayload` typed dicts with `status` (started/completed/failed), `graph_name`, `ns`, and optionally `trigger_call_id` / `result`. It tracks only subgraphs **strictly below** its own scope: `len(ns) > depth and ns[:depth] == self.scope`.
- `CustomTransformer` has `required_stream_modes = ("custom",)` and captures payloads written via `get_stream_writer()` inside nodes. The projection key is `"custom"`.
- `MessagesTransformer` (not shown in full here) uses `required_stream_modes = ("messages",)` and is the transformer-layer counterpart to `StreamMessagesHandler`.

### Example 1 — capture checkpoint events with `CheckpointsTransformer`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class S(TypedDict):
    x: int


def inc(state: S) -> dict:
    return {"x": state["x"] + 1}


graph = (
    StateGraph(S)
    .add_node("inc", inc)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile(checkpointer=MemorySaver())
)

cfg = {"configurable": {"thread_id": "cp-demo"}}

# stream_mode="checkpoints" uses CheckpointsTransformer under the hood
checkpoints = []
for event in graph.stream({"x": 0}, config=cfg, stream_mode="checkpoints"):
    checkpoints.append(event)

print(f"Checkpoint events received: {len(checkpoints)}")
if checkpoints:
    print("Keys in first checkpoint:", list(checkpoints[0].keys()))
```

### Example 2 — emit custom data from a node and read it via `CustomTransformer`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


class S(TypedDict):
    result: str


def compute(state: S) -> dict:
    writer = get_stream_writer()
    writer({"progress": 0.5, "message": "halfway"})
    writer({"progress": 1.0, "message": "done"})
    return {"result": "ok"}


graph = (
    StateGraph(S)
    .add_node("compute", compute)
    .add_edge(START, "compute")
    .add_edge("compute", END)
    .compile()
)

# stream_mode="custom" surfaces CustomTransformer payloads
custom_events = []
for event in graph.stream({"result": ""}, stream_mode="custom"):
    custom_events.append(event)

print(f"Custom events: {len(custom_events)}")
for ev in custom_events:
    print(" ", ev)
```

### Example 3 — combine multiple stream modes and route by mode

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import MemorySaver


class S(TypedDict):
    count: int


def step(state: S) -> dict:
    writer = get_stream_writer()
    writer({"step_count": state["count"]})
    return {"count": state["count"] + 1}


graph = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(checkpointer=MemorySaver())
)

cfg = {"configurable": {"thread_id": "multi-mode"}}

values_events, custom_events = [], []
for mode, event in graph.stream(
    {"count": 0}, config=cfg, stream_mode=["values", "custom"]
):
    if mode == "values":
        values_events.append(event)
    elif mode == "custom":
        custom_events.append(event)

print(f"values events: {len(values_events)}, custom events: {len(custom_events)}")
```

---

## 7 · `NodeBuilder`

**Module:** `langgraph.pregel.main`

`NodeBuilder` is the **fluent programmatic API** for constructing `PregelNode` instances without going through `StateGraph.add_node`. It is the low-level counterpart to the high-level graph builder, useful when building custom runtimes, middleware layers, or programmatically assembling Pregel graphs outside the `StateGraph` DSL.

**Key source facts** (from `langgraph/pregel/main.py`):

- `NodeBuilder` uses `__slots__` for all nine mutable fields: `_channels`, `_triggers`, `_tags`, `_metadata`, `_writes`, `_bound`, `_retry_policy`, `_cache_policy`, `_timeout`.
- `subscribe_only(channel)` — binds to a **single channel** (sets `_channels` as a bare `str`). Calling a second time raises `ValueError`. Adds the channel to `_triggers` automatically.
- `subscribe_to(*channels, read=True)` — appends to the `_channels` list (multi-channel). If `read=False`, channels trigger the node without appearing in its input dict.
- `do(node)` — wraps the callable with `coerce_to_runnable`; subsequent calls chain via `RunnableSeq`.
- `write_to(*channels, **kwargs)` — populates `_writes` as `ChannelWriteEntry` instances; callable kwargs become `ChannelWriteEntry(k, mapper=v)`.
- `add_retry_policies(*policies)` — appends `RetryPolicy` named-tuples to `_retry_policy`.
- `add_cache_policy(policy)` — sets `_cache_policy` to a `CachePolicy` instance.
- `set_timeout(timeout)` — normalises via `coerce_timeout_policy` and assigns to `_timeout`.
- `build()` — assembles and returns a `PregelNode(channels=…, triggers=…, writers=[ChannelWrite(self._writes)], …)`.

### Example 1 — build a node that subscribes to two channels and writes to a third

```python
from langgraph.pregel.main import NodeBuilder
from langgraph.pregel._write import ChannelWriteEntry


def merge_inputs(inputs: dict) -> dict:
    """Combine two input channels into one output."""
    a = inputs.get("channel_a", 0)
    b = inputs.get("channel_b", 0)
    return {"result": a + b}


node = (
    NodeBuilder()
    .subscribe_to("channel_a", "channel_b")
    .do(merge_inputs)
    .write_to("result")
    .build()
)

print("channels subscribed:", node.channels)
print("triggers:", node.triggers)
print("bound type:", type(node.bound).__name__)
```

### Example 2 — attach a retry policy and a cache policy

```python
from langgraph.pregel.main import NodeBuilder
from langgraph.types import RetryPolicy, CachePolicy


def flaky_step(state: dict) -> dict:
    return {"value": state.get("value", 0) + 1}


retry = RetryPolicy(initial_interval=0.1, max_attempts=3)
cache = CachePolicy(ttl=60)  # cache results for 60 seconds

node = (
    NodeBuilder()
    .subscribe_to("value")
    .do(flaky_step)
    .write_to("value")
    .add_retry_policies(retry)
    .add_cache_policy(cache)
    .build()
)

print("retry policies:", node.retry_policy)
print("cache policy ttl:", node.cache_policy.ttl)
```

### Example 3 — set a timeout and inspect the built `PregelNode`

```python
from datetime import timedelta
from langgraph.pregel.main import NodeBuilder
from langgraph.types import TimeoutPolicy


def slow_node(state: dict) -> dict:
    import time
    time.sleep(0.01)  # in production this could be an LLM call
    return {"done": True}


node = (
    NodeBuilder()
    .subscribe_to("trigger")
    .do(slow_node)
    .write_to("done")
    .set_timeout(TimeoutPolicy(run_timeout=30.0, idle_timeout=10.0))
    .meta("llm-call", env="prod")
    .build()
)

print("tags:", node.tags)
print("run_timeout:", node.timeout.run_timeout)
print("idle_timeout:", node.timeout.idle_timeout)
```

---

## 8 · `CachePolicy` · `CacheKey` · `TimeoutPolicy`

**Module:** `langgraph.types`

These three data classes configure **per-node execution policies** that are orthogonal to the graph's topology. `CachePolicy` memoises node results. `CacheKey` is the low-level key tuple used internally by the cache layer. `TimeoutPolicy` enforces hard (`run_timeout`) or activity-based (`idle_timeout`) time caps per node attempt.

**Key source facts** (from `langgraph/types.py`):

- `CachePolicy` is a frozen dataclass with `key_func: KeyFuncT = default_cache_key` and `ttl: int | None = None`. `default_cache_key` hashes the node's input with `pickle`. Supply a custom function to key on specific fields only.
- `CacheKey` is a `NamedTuple` with `ns: tuple[str, ...]`, `key: str`, and `ttl: int | None`. The cache layer constructs this from `CachePolicy.key_func(input)` plus the node's namespace.
- `TimeoutPolicy` is a frozen dataclass with `run_timeout: float | timedelta | None` (hard wall-clock cap, never refreshed), `idle_timeout: float | timedelta | None` (max time without observable progress), and `refresh_on: Literal["auto", "heartbeat"] = "auto"`.
- Under `refresh_on="auto"` the idle timer resets on any LangChain callback event (LLM token, chain end, tool start, etc.) and on explicit `runtime.heartbeat()` calls. Under `"heartbeat"` only explicit `heartbeat()` calls count.
- `TimeoutPolicy.coerce(value)` normalises a raw `float`, `timedelta`, or existing `TimeoutPolicy` into a `TimeoutPolicy` with positive-second float fields — used internally by `NodeBuilder.set_timeout`.
- `NodeTimeoutError` carries both `idle_timeout` and `run_timeout` (whichever is not `None`) along with `elapsed` and `kind: Literal["idle", "run"]` for precise error reporting.

### Example 1 — cache expensive node results with `CachePolicy`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

call_count = 0


class State(TypedDict):
    query: str
    result: str


def expensive_lookup(state: State) -> dict:
    global call_count
    call_count += 1
    print(f"  [lookup called #{call_count}]")
    return {"result": f"answer for: {state['query']}"}


cache = InMemoryCache()
graph = (
    StateGraph(State)
    .add_node(
        "lookup",
        expensive_lookup,
        cache=CachePolicy(ttl=300),  # cache for 5 minutes
    )
    .add_edge(START, "lookup")
    .add_edge("lookup", END)
    .compile(cache=cache)
)

# First call — cache miss
r1 = graph.invoke({"query": "capital of France", "result": ""})
print("r1:", r1["result"])

# Second call — cache hit, expensive_lookup not called again
r2 = graph.invoke({"query": "capital of France", "result": ""})
print("r2:", r2["result"])
print(f"Total actual calls: {call_count}")  # 1
```

### Example 2 — custom `key_func` to cache on a single field

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache


class S(TypedDict):
    user_id: str
    timestamp: float  # changes every call; must be excluded from cache key
    profile: dict


def fetch_profile(state: S) -> dict:
    print(f"  fetching profile for {state['user_id']}")
    return {"profile": {"name": f"User_{state['user_id']}", "tier": "gold"}}


# Only key on user_id — ignore timestamp
def user_id_key(state: dict) -> str:
    return state["user_id"]


cache = InMemoryCache()
graph = (
    StateGraph(S)
    .add_node("fetch", fetch_profile, cache=CachePolicy(key_func=user_id_key, ttl=60))
    .add_edge(START, "fetch")
    .add_edge("fetch", END)
    .compile(cache=cache)
)

import time
graph.invoke({"user_id": "u1", "timestamp": time.time(), "profile": {}})
# Second call with different timestamp — still a cache hit for user "u1"
r = graph.invoke({"user_id": "u1", "timestamp": time.time() + 5, "profile": {}})
print("cached profile:", r["profile"])
```

### Example 3 — set an idle timeout with `TimeoutPolicy`

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.errors import NodeTimeoutError


class S(TypedDict):
    done: bool


async def slow_async_node(state: S) -> dict:
    await asyncio.sleep(0.05)  # well within the timeout
    return {"done": True}


graph = (
    StateGraph(S)
    .add_node(
        "slow",
        slow_async_node,
        timeout=TimeoutPolicy(idle_timeout=5.0, run_timeout=30.0),
    )
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)


async def main():
    result = await graph.ainvoke({"done": False})
    print("done:", result["done"])


asyncio.run(main())
```

---

## 9 · `NodeTimeoutError` · `NodeCancelledError` · `GraphRecursionError`

**Module:** `langgraph.errors`

These three exceptions form the **node-failure error taxonomy** for non-business-logic failures. `NodeTimeoutError` fires when a `TimeoutPolicy` deadline is exceeded. `NodeCancelledError` wraps a user-raised `asyncio.CancelledError` so the run is reported as `error` rather than silently succeeding. `GraphRecursionError` fires when the Pregel loop exhausts `recursion_limit`.

**Key source facts** (from `langgraph/errors.py`):

- `NodeTimeoutError` does **not** inherit from `TimeoutError` (which is a subclass of `OSError`) — intentionally, so the default `RetryPolicy.retry_on` treats it as retryable. Fields: `node: str`, `timeout: float` (whichever threshold fired), `run_timeout: float | None`, `idle_timeout: float | None`, `elapsed: float`, `kind: Literal["idle", "run"]`. Constructors raise `ValueError` if the matching threshold kwarg is absent.
- `NodeCancelledError` carries `node: str`. The retry layer converts a user-raised `asyncio.CancelledError` in a node body to this type so it flows through the normal error path. Framework-initiated cancellation (the runner stopping sibling tasks) is left as `CancelledError` and silently tears down.
- `GraphRecursionError` extends `RecursionError` with no extra fields. Raised when the loop counter reaches `config["recursion_limit"]` (default 25). To raise it: run a graph with a deliberately short limit.
- `ErrorCode` is an `Enum` with symbolic names (e.g. `GRAPH_RECURSION_LIMIT`) used in error-page URLs in docstrings — links point to troubleshooting guides.
- `GraphInterrupt` (not shown in full) is a `GraphBubbleUp` raised when `interrupt()` is called; it is **not** an error — it is the normal control-flow mechanism for pausing execution.

### Example 1 — catch a `NodeTimeoutError` and inspect `kind` / `elapsed`

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.errors import NodeTimeoutError


class S(TypedDict):
    result: str


async def very_slow(state: S) -> dict:
    await asyncio.sleep(10)  # will time out
    return {"result": "never"}


graph = (
    StateGraph(S)
    .add_node("slow", very_slow, timeout=TimeoutPolicy(run_timeout=0.05))
    .add_edge(START, "slow")
    .add_edge("slow", END)
    .compile()
)


async def main():
    try:
        await graph.ainvoke({"result": ""})
    except NodeTimeoutError as exc:
        print(f"node: {exc.node}")
        print(f"kind: {exc.kind}")          # "run"
        print(f"run_timeout: {exc.run_timeout}")
        print(f"elapsed: {exc.elapsed:.3f}s")


asyncio.run(main())
```

### Example 2 — trigger `GraphRecursionError` with a short `recursion_limit`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.errors import GraphRecursionError


class S(TypedDict):
    n: int


def loop_node(state: S) -> dict:
    return {"n": state["n"] + 1}


builder = StateGraph(S)
builder.add_node("loop", loop_node)
builder.add_edge(START, "loop")
builder.add_edge("loop", "loop")  # infinite cycle

graph = builder.compile()

try:
    graph.invoke({"n": 0}, config={"recursion_limit": 5})
except GraphRecursionError as exc:
    print(f"Caught GraphRecursionError: {exc}")
    print("Increase recursion_limit in config to allow deeper graphs.")
```

### Example 3 — retry a node on `NodeTimeoutError` with `RetryPolicy`

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.errors import NodeTimeoutError

attempt_log: list[int] = []


class S(TypedDict):
    value: int


async def sometimes_slow(state: S) -> dict:
    attempt_log.append(len(attempt_log) + 1)
    if len(attempt_log) < 2:
        await asyncio.sleep(5)  # first attempt times out
    return {"value": 42}


graph = (
    StateGraph(S)
    .add_node(
        "work",
        sometimes_slow,
        timeout=TimeoutPolicy(run_timeout=0.05),
        retry=RetryPolicy(max_attempts=3, initial_interval=0.0, jitter=False),
    )
    .add_edge(START, "work")
    .add_edge("work", END)
    .compile()
)


async def main():
    try:
        result = await graph.ainvoke({"value": 0})
        print(f"Succeeded on attempt #{len(attempt_log)}: {result['value']}")
    except NodeTimeoutError as exc:
        print(f"Still timing out after {len(attempt_log)} attempts: {exc}")


asyncio.run(main())
```

---

## 10 · `UIMessage` · `RemoveUIMessage`

**Module:** `langgraph.graph.ui`

`UIMessage` and `RemoveUIMessage` power **generative UI streaming** in LangGraph. A node emits `UIMessage` dicts to the stream to tell a connected front-end to mount a named component with specific props. `RemoveUIMessage` tells the front-end to unmount a previously mounted component by ID. Both are `TypedDict` shapes, meaning they are plain dicts at runtime and can be emitted via `get_stream_writer()` or pushed through the `"ui"` / `"values"` stream channels.

**Key source facts** (from `langgraph/graph/ui.py`):

- `UIMessage` has five required keys: `type: Literal["ui"]`, `id: str` (unique identifier — used to correlate with `RemoveUIMessage`), `name: str` (the front-end component name to render), `props: dict[str, Any]` (component properties), and `metadata: dict[str, Any]` (optional out-of-band info such as node name, run ID).
- `RemoveUIMessage` has two keys: `type: Literal["remove-ui"]` and `id: str` (must match the `id` of a previously emitted `UIMessage`).
- These types are consumed by LangGraph's official front-end SDKs (e.g. `@langchain/langgraph-sdk` for React). In open-source usage, they appear as plain dicts in the stream and can be dispatched to any UI layer.
- The `id` field is critical: front-ends use it to maintain a keyed component registry. Emitting the same `id` twice with different `props` will update the component in place; emitting `RemoveUIMessage` with the same `id` will unmount it.
- There is no built-in validation of `name` values — the front-end is responsible for mapping names to components.

### Example 1 — emit a `UIMessage` from a node via the stream writer

```python
import uuid
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage
from langgraph.config import get_stream_writer


class S(TypedDict):
    query: str
    answer: str


def search_node(state: S) -> dict:
    writer = get_stream_writer()

    # Tell the UI to show a "searching" spinner
    spinner_id = str(uuid.uuid4())
    writer(
        UIMessage(
            type="ui",
            id=spinner_id,
            name="SearchSpinner",
            props={"query": state["query"]},
            metadata={"node": "search_node"},
        )
    )

    answer = f"Result for '{state['query']}'"

    # Tell the UI to remove the spinner once we have the answer
    from langgraph.graph.ui import RemoveUIMessage
    writer(RemoveUIMessage(type="remove-ui", id=spinner_id))

    return {"answer": answer}


graph = (
    StateGraph(S)
    .add_node("search", search_node)
    .add_edge(START, "search")
    .add_edge("search", END)
    .compile()
)

ui_events = []
for event in graph.stream(
    {"query": "LangGraph docs", "answer": ""},
    stream_mode="custom",
):
    ui_events.append(event)

print(f"UI events emitted: {len(ui_events)}")
for ev in ui_events:
    print(f"  type={ev.get('type')} name={ev.get('name', '-')} id={ev.get('id', '-')[:8]}…")
```

### Example 2 — stream a live data table that updates its props in place

```python
import uuid
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage
from langgraph.config import get_stream_writer


class S(TypedDict):
    rows: list


def live_table_node(state: S) -> dict:
    writer = get_stream_writer()
    table_id = "live-table-001"  # stable ID for in-place updates

    for batch in range(3):
        writer(
            UIMessage(
                type="ui",
                id=table_id,
                name="DataTable",
                props={"rows": list(range(batch * 2)), "loading": batch < 2},
                metadata={"batch": batch},
            )
        )

    return {"rows": list(range(6))}


graph = (
    StateGraph(S)
    .add_node("table", live_table_node)
    .add_edge(START, "table")
    .add_edge("table", END)
    .compile()
)

for event in graph.stream({"rows": []}, stream_mode="custom"):
    print(f"  UIMessage id={event.get('id')} props.loading={event.get('props', {}).get('loading')}")
```

### Example 3 — remove a UI component after a conditional branch

```python
import uuid
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage, RemoveUIMessage
from langgraph.config import get_stream_writer


class S(TypedDict):
    approved: bool
    banner_id: str


def show_banner(state: S) -> dict:
    writer = get_stream_writer()
    bid = str(uuid.uuid4())
    writer(
        UIMessage(
            type="ui",
            id=bid,
            name="ApprovalBanner",
            props={"message": "Awaiting approval…", "status": "pending"},
            metadata={},
        )
    )
    return {"banner_id": bid}


def approve(state: S) -> dict:
    writer = get_stream_writer()
    # Swap the banner to "approved" state by re-emitting with same id
    writer(
        UIMessage(
            type="ui",
            id=state["banner_id"],
            name="ApprovalBanner",
            props={"message": "Approved!", "status": "success"},
            metadata={},
        )
    )
    return {"approved": True}


def reject(state: S) -> dict:
    writer = get_stream_writer()
    # Remove the banner entirely on rejection
    writer(RemoveUIMessage(type="remove-ui", id=state["banner_id"]))
    return {"approved": False}


def route(state: S) -> str:
    return "approve" if state.get("approved") else "reject"


builder = StateGraph(S)
builder.add_node("show", show_banner)
builder.add_node("approve", approve)
builder.add_node("reject", reject)
builder.add_edge(START, "show")
builder.add_conditional_edges("show", route, {"approve": "approve", "reject": "reject"})
builder.add_edge("approve", END)
builder.add_edge("reject", END)

graph = builder.compile()

print("--- Approval path ---")
for ev in graph.stream({"approved": True, "banner_id": ""}, stream_mode="custom"):
    print(f"  type={ev['type']} name={ev.get('name', '-')}")

print("--- Rejection path ---")
for ev in graph.stream({"approved": False, "banner_id": ""}, stream_mode="custom"):
    print(f"  type={ev['type']} name={ev.get('name', '-')}")
```

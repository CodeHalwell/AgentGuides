---
title: "Class deep-dives Vol. 16 — v3 Streaming Protocol"
description: "Source-verified deep dives into the experimental v3 streaming API introduced in LangGraph 1.2.0: GraphRunStream, AsyncGraphRunStream, SubgraphRunStream, ValuesTransformer, UpdatesTransformer, CustomTransformer, MessagesTransformer, ChatModelStream, SubgraphTransformer, LifecycleTransformer, and StreamChannel — with multiple runnable examples for each class."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 16"
  order: 47
---

# Class deep-dives Vol. 16 — v3 Streaming Protocol

Verified against **`langgraph==1.2.5`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

> **Status:** The v3 streaming API (`stream_events(version="v3")` / `astream_events(version="v3")`) is marked `@beta` in the source. The API shape may change in a future minor release. All other classes covered here (`StreamChannel`, transformers) are stable.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `GraphRunStream` | `langgraph.stream.run_stream` |
| 2 | `AsyncGraphRunStream` | `langgraph.stream.run_stream` |
| 3 | `SubgraphRunStream` + `AsyncSubgraphRunStream` | `langgraph.stream.run_stream` |
| 4 | `ValuesTransformer` | `langgraph.stream.transformers` |
| 5 | `UpdatesTransformer` | `langgraph.stream.transformers` |
| 6 | `CustomTransformer` | `langgraph.stream.transformers` |
| 7 | `MessagesTransformer` + `ChatModelStream` | `langgraph.stream.transformers` |
| 8 | `SubgraphTransformer` | `langgraph.stream.transformers` |
| 9 | `LifecycleTransformer` + `LifecyclePayload` | `langgraph.stream.transformers` |
| 10 | `StreamChannel` | `langgraph.stream.stream_channel` |

---

## Background: The v3 Streaming Protocol

LangGraph's original streaming API (`graph.stream(stream_mode=...)`) yields raw dicts. The v2 API added typed `StreamPart` objects. The v3 API (beta, introduced in v1.2.0) takes a fundamentally different approach: instead of choosing a single stream mode, you get a **`GraphRunStream`** handle with multiple typed **projections** you can iterate independently or interleave.

```
stream_events(version="v3") ──► GraphRunStream
                                    │
                                    ├── run.values       (ValuesTransformer)
                                    ├── run.updates      (UpdatesTransformer)
                                    ├── run.custom       (CustomTransformer)
                                    ├── run.messages     (MessagesTransformer)
                                    ├── run.subgraphs    (SubgraphTransformer)
                                    ├── run.lifecycle    (LifecycleTransformer)
                                    ├── run.checkpoints  (CheckpointsTransformer)
                                    ├── run.tasks        (TasksTransformer)
                                    ├── run.debug        (DebugTransformer)
                                    └── run.output       (final state, no iteration)
```

No background thread is used. The caller's iteration on any projection drives the graph forward. The `StreamMux` dispatches each raw `ProtocolEvent` to all registered transformers.

---

## 1 · `GraphRunStream`

**Module:** `langgraph.stream.run_stream`  
**Import:**
```python
from langgraph.stream.run_stream import GraphRunStream
```

`GraphRunStream` is the sync handle returned by `graph.stream_events(version="v3")`. It owns a `StreamMux` with all built-in transformers pre-wired, and exposes every projection as a direct attribute (`run.values`, `run.messages`, etc.).

### Source signature (1.2.5)

```python
@beta(message="The v3 streaming protocol on Pregel is experimental.")
class GraphRunStream:
    # Direct attributes set by the mux's native transformers:
    values: StreamChannel[dict[str, Any]]
    updates: StreamChannel[dict[str, Any]]
    custom: StreamChannel[Any]
    messages: StreamChannel[ChatModelStream]
    subgraphs: StreamChannel[SubgraphRunStream]
    lifecycle: StreamChannel[LifecyclePayload]
    checkpoints: StreamChannel[dict[str, Any]]
    tasks: StreamChannel[dict[str, Any]]
    debug: StreamChannel[dict[str, Any]]

    # Key properties:
    @property
    def output(self) -> dict[str, Any] | None: ...     # drive to completion → final state
    @property
    def interrupted(self) -> bool: ...                  # drive to completion → was it interrupted?
    @property
    def interrupts(self) -> list[Any]: ...              # drive to completion → interrupt payloads

    def abort(self) -> None: ...                        # stop early, close the mux
    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]: ...  # merge projections
    def __iter__(self) -> Iterator[ProtocolEvent]: ...  # raw protocol events
    def __enter__(self) -> GraphRunStream: ...
    def __exit__(self, ...) -> None: ...
```

### Example 1: Iterating `run.values`

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    step: int


def node_a(state: State) -> dict:
    return {"step": state.get("step", 0) + 1}


def node_b(state: State) -> dict:
    return {"step": state["step"] + 1}


graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "v3-ex1"}}

# stream_events(version="v3") returns a GraphRunStream
run = graph.stream_events(
    {"messages": [HumanMessage(content="hi")], "step": 0},
    cfg,
    version="v3",
)

# Iterate run.values — each item is the full state after a node completes
for snapshot in run.values:
    print(f"step={snapshot['step']}")
# step=1
# step=2
```

### Example 2: Using the context manager and `run.output`

The context manager guarantees the graph is cleaned up even if you exit early. `.output` drives the graph all the way to completion and returns the final state snapshot.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    counter: int


def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}


graph = (
    StateGraph(State)
    .add_node("inc", increment)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"counter": 0},
    {"configurable": {"thread_id": "ctx-1"}},
    version="v3",
) as run:
    # .output drives the entire graph and returns the final snapshot
    final = run.output
    print(final)
# {"counter": 1}
```

### Example 3: `run.interrupted` and `run.interrupts`

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    approved: bool


def agent(state: State) -> dict:
    return {}


def gate(state: State) -> dict:
    answer = interrupt({"question": "Approve this action?"})
    return {"approved": answer}


graph = (
    StateGraph(State)
    .add_node("agent", agent)
    .add_node("gate", gate)
    .add_edge(START, "agent")
    .add_edge("agent", "gate")
    .add_edge("gate", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "interrupt-ex"}}

with graph.stream_events(
    {"messages": [HumanMessage(content="do it")], "approved": False},
    cfg,
    version="v3",
) as run:
    # Drives to completion (paused at interrupt)
    print("interrupted?", run.interrupted)     # True
    print("interrupts:", run.interrupts)        # [{"question": "Approve this action?"}]

# Resume the graph
with graph.stream_events(
    Command(resume=True),
    cfg,
    version="v3",
) as run:
    final = run.output
    print("approved:", final["approved"])       # True
```

### Example 4: `run.interleave` — merge projections in arrival order

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    val: int


def step(state: State) -> dict:
    return {"val": state["val"] + 1}


graph = (
    StateGraph(State)
    .add_node("s1", step)
    .add_node("s2", step)
    .add_edge(START, "s1")
    .add_edge("s1", "s2")
    .add_edge("s2", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"val": 0},
    {"configurable": {"thread_id": "interleave-1"}},
    version="v3",
) as run:
    for name, item in run.interleave("values", "updates"):
        if name == "values":
            print(f"[values] val={item['val']}")
        else:
            print(f"[updates] {item}")
# Items arrive in strict push-stamp order across projections.
```

### Example 5: Iterating raw `ProtocolEvent` objects

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    x: int


def inc(state: State) -> dict:
    return {"x": state["x"] + 1}


graph = (
    StateGraph(State)
    .add_node("inc", inc)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile()
)

run = graph.stream_events({"x": 0}, version="v3")
for event in run:
    # ProtocolEvent TypedDict: {"type": "event", "method": str, "params": {...}}
    print(event["method"], event["params"].get("namespace"))
```

---

## 2 · `AsyncGraphRunStream`

**Module:** `langgraph.stream.run_stream`  
**Import:**
```python
from langgraph.stream.run_stream import AsyncGraphRunStream
```

`AsyncGraphRunStream` is the async counterpart to `GraphRunStream`. It is returned by `await graph.astream_events(version="v3")` (note: the call itself is async) and must be used as an async context manager or with `async for`. No background task is created — async iteration on any projection drives the graph's event loop forward.

### Source signature (1.2.5)

```python
@beta(message="The v3 streaming protocol on Pregel is experimental.")
class AsyncGraphRunStream:
    # Same projection attributes as GraphRunStream, but async-iterable:
    values: StreamChannel[dict[str, Any]]      # async for snapshot in run.values
    updates: StreamChannel[dict[str, Any]]     # async for update in run.updates
    messages: StreamChannel[ChatModelStream]   # async for msg_stream in run.messages
    # ... same set of projections ...

    @property
    async def output(self) -> dict[str, Any] | None: ...
    @property
    async def interrupted(self) -> bool: ...
    @property
    async def interrupts(self) -> list[Any]: ...

    def abort(self) -> None: ...
    async def aabort(self) -> None: ...

    async def ainterleave(self, *names: str) -> AsyncIterator[tuple[str, Any]]: ...
    def __aiter__(self) -> AsyncIterator[ProtocolEvent]: ...
    async def __aenter__(self) -> AsyncGraphRunStream: ...
    async def __aexit__(self, ...) -> None: ...
```

### Example 1: Basic async streaming

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def chat(state: State) -> dict:
    last = state["messages"][-1].content
    return {"messages": [AIMessage(content=f"Echo: {last}")]}


graph = (
    StateGraph(State)
    .add_node("chat", chat)
    .add_edge(START, "chat")
    .add_edge("chat", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "async-v3-1"}}


async def main():
    async with await graph.astream_events(
        {"messages": [HumanMessage(content="Hello")]},
        cfg,
        version="v3",
    ) as run:
        async for snapshot in run.values:
            print("snapshot:", snapshot["messages"][-1].content)
        # After iteration exhausts, .output is already cached
        print("final:", (await run.output)["messages"][-1].content)


asyncio.run(main())
# snapshot: Echo: Hello
# final: Echo: Hello
```

### Example 2: Token streaming via `run.messages`

When the model is called with `stream_events(version="v3")`, `MessagesTransformer` yields one `ChatModelStream` per LLM invocation. Iterate each handle's `.text` projection for token-by-token text.

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def call_llm(state: State) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


graph = (
    StateGraph(State)
    .add_node("llm", call_llm)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "msg-stream-1"}}


async def main():
    async with await graph.astream_events(
        {"messages": [HumanMessage(content="Count to 3")]},
        cfg,
        version="v3",
    ) as run:
        async for msg_stream in run.messages:
            # msg_stream is a ChatModelStream / AsyncChatModelStream
            print(f"[LLM call in node: {msg_stream.node}]")
            async for text_delta in msg_stream.text:
                print(text_delta, end="", flush=True)
            print()  # newline after each message


asyncio.run(main())
```

### Example 3: Async interleave

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    n: int


def inc(state: State) -> dict:
    return {"n": state["n"] + 1}


graph = (
    StateGraph(State)
    .add_node("a", inc)
    .add_node("b", inc)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)


async def main():
    async with await graph.astream_events(
        {"n": 0},
        {"configurable": {"thread_id": "ai-1"}},
        version="v3",
    ) as run:
        async for name, item in run.ainterleave("values", "updates"):
            print(f"[{name}]", item)


asyncio.run(main())
```

---

## 3 · `SubgraphRunStream` + `AsyncSubgraphRunStream`

**Module:** `langgraph.stream.run_stream`  
**Import:**
```python
from langgraph.stream.run_stream import SubgraphRunStream, AsyncSubgraphRunStream
```

`SubgraphRunStream` extends `GraphRunStream` and represents a **discovered direct-child subgraph**. You never construct these directly — they are pushed into `run.subgraphs` by `SubgraphTransformer` as subgraph invocations are detected at runtime. Each handle exposes the same projection attributes as its parent but scoped to that subgraph's namespace.

### Key additional attributes

```python
class SubgraphRunStream(GraphRunStream):
    path: tuple[str, ...]         # namespace tuple identifying the subgraph
    graph_name: str | None        # the graph's name (if known)
    trigger_call_id: str | None   # the tool-call id that triggered this subgraph
    status: SubgraphStatus        # "started" | "completed" | "failed" | "cancelled"
    error: str | None             # set if status == "failed"
```

### Example 1: Iterating subgraph handles

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class InnerState(TypedDict):
    x: int


class OuterState(TypedDict):
    x: int
    result: int


def inner_node(state: InnerState) -> dict:
    return {"x": state["x"] * 2}


# Build the inner subgraph
inner = (
    StateGraph(InnerState)
    .add_node("double", inner_node)
    .add_edge(START, "double")
    .add_edge("double", END)
    .compile()
)


def outer_node(state: OuterState) -> dict:
    result = inner.invoke({"x": state["x"]})
    return {"result": result["x"]}


outer = (
    StateGraph(OuterState)
    .add_node("run_inner", outer_node)
    .add_edge(START, "run_inner")
    .add_edge("run_inner", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "sub-1"}}

with outer.stream_events(
    {"x": 5, "result": 0},
    cfg,
    version="v3",
) as run:
    for subgraph_handle in run.subgraphs:
        print(f"Subgraph path: {subgraph_handle.path}")
        print(f"Graph name:    {subgraph_handle.graph_name}")
        # Iterate the subgraph's own values projection
        for inner_snap in subgraph_handle.values:
            print(f"  inner state: {inner_snap}")
    print("Final:", run.output)
```

### Example 2: Checking subgraph status

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    n: int


def node(state: State) -> dict:
    return {"n": state["n"] + 1}


inner = (
    StateGraph(State)
    .add_node("inc", node)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile()
)


def outer_fn(state: State) -> dict:
    r = inner.invoke({"n": state["n"]})
    return {"n": r["n"]}


graph = (
    StateGraph(State)
    .add_node("outer", outer_fn)
    .add_edge(START, "outer")
    .add_edge("outer", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"n": 0},
    {"configurable": {"thread_id": "status-check"}},
    version="v3",
) as run:
    for handle in run.subgraphs:
        # Drain all events from this handle
        final_inner = handle.output
        print(f"status={handle.status}, result={final_inner}")
    _ = run.output
```

### Example 3: Async subgraph streaming

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class S(TypedDict):
    v: int


def inc(s: S) -> dict:
    return {"v": s["v"] + 10}


inner = (
    StateGraph(S).add_node("i", inc).add_edge(START, "i").add_edge("i", END).compile()
)


def wrap(s: S) -> dict:
    return {"v": inner.invoke({"v": s["v"]})["v"]}


outer = (
    StateGraph(S)
    .add_node("w", wrap)
    .add_edge(START, "w")
    .add_edge("w", END)
    .compile(checkpointer=InMemorySaver())
)


async def main():
    async with await outer.astream_events(
        {"v": 1},
        {"configurable": {"thread_id": "async-sub"}},
        version="v3",
    ) as run:
        async for handle in run.subgraphs:
            print(f"subgraph path={handle.path}")
            async for snap in handle.values:
                print(f"  inner snap: {snap}")
        print("outer output:", await run.output)


asyncio.run(main())
```

---

## 4 · `ValuesTransformer`

**Module:** `langgraph.stream.transformers`  
**Import:**
```python
from langgraph.stream.transformers import ValuesTransformer
```

`ValuesTransformer` is a native `StreamTransformer` that captures `stream_mode="values"` protocol events and surfaces them as `run.values` — a `StreamChannel[dict[str, Any]]` where each item is the full graph state after a node completes.

### Source signature (1.2.5)

```python
class ValuesTransformer(StreamTransformer):
    _native = True
    required_stream_modes = ("values",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None: ...
    def init(self) -> dict[str, Any]:
        return {"values": self._log}   # StreamChannel[dict]
    def process(self, event: ProtocolEvent) -> bool: ...
```

### Example 1: Programmatically register `ValuesTransformer`

You don't usually register transformers manually — `stream_events(version="v3")` does it. But you can add extra ones at the call site via `transformers=`:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.stream.transformers import ValuesTransformer
from typing_extensions import TypedDict


class State(TypedDict):
    count: int


def inc(state: State) -> dict:
    return {"count": state["count"] + 1}


graph = (
    StateGraph(State)
    .add_node("inc", inc)
    .add_edge(START, "inc")
    .add_edge("inc", END)
    .compile(checkpointer=InMemorySaver())
)

# ValuesTransformer is already registered by default in version="v3",
# but you can add a second one (e.g. scoped differently) via transformers=
run = graph.stream_events(
    {"count": 0},
    {"configurable": {"thread_id": "val-t-1"}},
    version="v3",
)

for snap in run.values:
    print("count =", snap["count"])   # 1
```

### Example 2: Custom scope to restrict snapshots

The `scope` parameter limits which `values` events reach the projection. Pass a non-empty tuple to only capture events from a specific subgraph namespace.

```python
# When building a transformer factory for a subgraph scope:
from langgraph.stream.transformers import ValuesTransformer

root_scope: tuple[str, ...] = ()
root_transformer = ValuesTransformer(scope=root_scope)

subgraph_scope: tuple[str, ...] = ("my-subgraph",)
sub_transformer = ValuesTransformer(scope=subgraph_scope)

# root_transformer.process() captures events where params["namespace"] == []
# sub_transformer.process() captures events where params["namespace"] == ["my-subgraph"]
```

### Example 3: Reading the latest snapshot via `run.output`

`run.output` drives the graph to completion and returns the final `_latest` values snapshot cached by `ValuesTransformer` (via `GraphRunStream._observe_event`).

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    items: list
    done: bool


def collect(state: State) -> dict:
    return {"items": state["items"] + ["collected"], "done": True}


graph = (
    StateGraph(State)
    .add_node("collect", collect)
    .add_edge(START, "collect")
    .add_edge("collect", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"items": [], "done": False},
    {"configurable": {"thread_id": "out-1"}},
    version="v3",
) as run:
    final = run.output          # drives to END
    print(final["items"])       # ["collected"]
    print(final["done"])        # True
```

---

## 5 · `UpdatesTransformer`

**Module:** `langgraph.stream.transformers`  
**Import:**
```python
from langgraph.stream.transformers import UpdatesTransformer
```

`UpdatesTransformer` captures `stream_mode="updates"` events and surfaces them on `run.updates`. Each item is a `dict` mapping node names to the updates they returned (`{"node_name": {key: value, ...}}`).

### Source signature (1.2.5)

```python
class UpdatesTransformer(StreamTransformer):
    _native = True
    required_stream_modes = ("updates",)

    def init(self) -> dict[str, Any]:
        return {"updates": self._log}   # StreamChannel[dict[str, Any]]
```

### Example 1: Per-node update inspection

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    a: int
    b: int


def step_a(state: State) -> dict:
    return {"a": state["a"] + 1}


def step_b(state: State) -> dict:
    return {"b": state["b"] + 10}


graph = (
    StateGraph(State)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .add_edge("step_b", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"a": 0, "b": 0},
    {"configurable": {"thread_id": "upd-1"}},
    version="v3",
) as run:
    for update in run.updates:
        # Each item: {"node_name": {changed_keys...}}
        node_name, delta = next(iter(update.items()))
        print(f"Node '{node_name}' changed: {delta}")
# Node 'step_a' changed: {'a': 1}
# Node 'step_b' changed: {'b': 10}
```

### Example 2: Branching — tracking which path was taken

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    value: int
    path_taken: str


def high_path(state: State) -> dict:
    return {"path_taken": "high"}


def low_path(state: State) -> dict:
    return {"path_taken": "low"}


def router(state: State) -> str:
    return "high" if state["value"] > 5 else "low"


graph = (
    StateGraph(State)
    .add_node("high", high_path)
    .add_node("low", low_path)
    .add_conditional_edges(START, router, {"high": "high", "low": "low"})
    .add_edge("high", END)
    .add_edge("low", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"value": 8, "path_taken": ""},
    {"configurable": {"thread_id": "branch-1"}},
    version="v3",
) as run:
    for update in run.updates:
        print("update:", update)    # {"high": {"path_taken": "high"}}
```

---

## 6 · `CustomTransformer`

**Module:** `langgraph.stream.transformers`  
**Import:**
```python
from langgraph.stream.transformers import CustomTransformer
```

`CustomTransformer` captures data emitted by nodes via `get_stream_writer()` and surfaces it on `run.custom`. This is the v3 equivalent of `stream_mode="custom"`.

### Source signature (1.2.5)

```python
class CustomTransformer(StreamTransformer):
    _native = True
    required_stream_modes = ("custom",)

    def init(self) -> dict[str, Any]:
        return {"custom": self._log}   # StreamChannel[Any]
```

### Example 1: Emitting custom events from a node

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict


class State(TypedDict):
    items: list[str]


def processor(state: State) -> dict:
    writer = get_stream_writer()
    results = []
    for i, item in enumerate(state["items"]):
        writer({"progress": i + 1, "of": len(state["items"]), "item": item})
        results.append(item.upper())
    return {"items": results}


graph = (
    StateGraph(State)
    .add_node("process", processor)
    .add_edge(START, "process")
    .add_edge("process", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"items": ["apple", "banana", "cherry"]},
    {"configurable": {"thread_id": "custom-1"}},
    version="v3",
) as run:
    for payload in run.custom:
        print(f"Progress: {payload['progress']}/{payload['of']} — {payload['item']}")
    print("Final:", run.output["items"])
# Progress: 1/3 — apple
# Progress: 2/3 — banana
# Progress: 3/3 — cherry
# Final: ['APPLE', 'BANANA', 'CHERRY']
```

### Example 2: Mixed projection — interleaving custom events with values

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict


class State(TypedDict):
    count: int


def counter(state: State) -> dict:
    writer = get_stream_writer()
    writer({"event": "counting", "from": state["count"]})
    return {"count": state["count"] + 1}


graph = (
    StateGraph(State)
    .add_node("counter", counter)
    .add_edge(START, "counter")
    .add_edge("counter", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"count": 0},
    {"configurable": {"thread_id": "mixed-1"}},
    version="v3",
) as run:
    for kind, item in run.interleave("custom", "values"):
        if kind == "custom":
            print(f"[custom] {item}")
        else:
            print(f"[values] count={item['count']}")
# [custom] {'event': 'counting', 'from': 0}
# [values] count=1
```

---

## 7 · `MessagesTransformer` + `ChatModelStream`

**Module:** `langgraph.stream.transformers`  
**Imports:**
```python
from langgraph.stream.transformers import MessagesTransformer, ChatModelStream
```

`MessagesTransformer` captures `stream_mode="messages"` protocol events and surfaces them on `run.messages` as a stream of `ChatModelStream` handles — one per LLM invocation. Each handle provides typed sub-projections (`.text`, `.reasoning`, `.tool_calls`, `.usage`, `.output`).

### `ChatModelStream` API

```python
class ChatModelStream:
    namespace: list[str] | None    # graph namespace of the LLM call
    node: str | None               # node name that made the call
    message_id: str | None         # AIMessage id

    @property
    def text(self) -> SyncTextProjection: ...       # Iterable[str] delta text
    @property
    def reasoning(self) -> SyncTextProjection: ...  # Iterable[str] reasoning deltas
    @property
    def tool_calls(self) -> SyncProjection: ...     # Iterable[ToolCallChunk]
    @property
    def output(self) -> AIMessage: ...              # blocking — drive to AIMessage
```

### Example 1: Token streaming with `ChatModelStream`

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


def llm_node(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "msg-1"}}

with graph.stream_events(
    {"messages": [HumanMessage(content="Say hello in French")]},
    cfg,
    version="v3",
) as run:
    for msg_stream in run.messages:
        print(f"[Node: {msg_stream.node}]")
        # .text is a SyncTextProjection: iterate for token deltas
        for token in msg_stream.text:
            print(token, end="", flush=True)
        print()
        # .output gives the assembled AIMessage once streaming is done
        print(f"Full: {msg_stream.output.content}")
```

### Example 2: Tool call streaming

When the model emits tool calls, they appear on `.tool_calls`. Each item is a `ToolCallChunk`.

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}, 22°C"


llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools([get_weather])


def llm_node(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


graph = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_node("tools", ToolNode([get_weather]))
    .add_edge(START, "llm")
    .add_conditional_edges("llm", tools_condition)
    .add_edge("tools", "llm")
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "tool-stream-1"}}

with graph.stream_events(
    {"messages": [HumanMessage(content="What's the weather in Paris?")]},
    cfg,
    version="v3",
) as run:
    for msg_stream in run.messages:
        print(f"[{msg_stream.node}]")
        # Collect any tool call chunks
        tc_chunks = list(msg_stream.tool_calls)
        if tc_chunks:
            print("Tool calls:", [c for c in tc_chunks if c.get("name")])
        else:
            for tok in msg_stream.text:
                print(tok, end="", flush=True)
        print()
```

### Example 3: Async message streaming

```python
import asyncio
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


async def llm_node(state: MessagesState) -> dict:
    return {"messages": [await llm.ainvoke(state["messages"])]}


graph = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile(checkpointer=InMemorySaver())
)


async def main():
    async with await graph.astream_events(
        {"messages": [HumanMessage(content="Tell me a joke")]},
        {"configurable": {"thread_id": "async-msg-1"}},
        version="v3",
    ) as run:
        async for msg_stream in run.messages:
            print(f"[{msg_stream.node}]", end=" ")
            async for token in msg_stream.text:
                print(token, end="", flush=True)
            print()


asyncio.run(main())
```

---

## 8 · `SubgraphTransformer`

**Module:** `langgraph.stream.transformers`  
**Import:**
```python
from langgraph.stream.transformers import SubgraphTransformer
```

`SubgraphTransformer` watches `tasks` protocol events to detect when a **direct-child subgraph** starts and finishes. For each discovered subgraph it builds a `SubgraphRunStream` (or async) handle with its own scoped mini-`StreamMux`, then pushes it into `run.subgraphs`. Grandchildren are discovered by the child handle's own `SubgraphTransformer` — the root only sees direct children.

### Source signature (1.2.5)

```python
class SubgraphTransformer(_TasksLifecycleBase):
    _native = True
    supports_sync = True
    required_stream_modes = ("tasks",)

    def init(self) -> dict[str, Any]:
        return {"subgraphs": self._log}   # StreamChannel[SubgraphRunStream]
```

### Example 1: Nested subgraph traversal

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class S(TypedDict):
    depth: int
    x: int


def leaf_node(s: S) -> dict:
    return {"x": s["x"] + 1}


# Level-2 subgraph
level2 = (
    StateGraph(S)
    .add_node("leaf", leaf_node)
    .add_edge(START, "leaf")
    .add_edge("leaf", END)
    .compile()
)


def mid_node(s: S) -> dict:
    r = level2.invoke({"depth": 2, "x": s["x"]})
    return {"x": r["x"]}


# Level-1 subgraph
level1 = (
    StateGraph(S)
    .add_node("mid", mid_node)
    .add_edge(START, "mid")
    .add_edge("mid", END)
    .compile()
)


def outer_node(s: S) -> dict:
    r = level1.invoke({"depth": 1, "x": s["x"]})
    return {"x": r["x"]}


# Outer graph
root = (
    StateGraph(S)
    .add_node("outer", outer_node)
    .add_edge(START, "outer")
    .add_edge("outer", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "nested-1"}}

with root.stream_events({"depth": 0, "x": 0}, cfg, version="v3") as run:
    for level1_handle in run.subgraphs:
        print(f"L1 subgraph path={level1_handle.path}")
        # Grandchildren are on the child handle's .subgraphs
        for level2_handle in level1_handle.subgraphs:
            print(f"  L2 subgraph path={level2_handle.path}")
            _ = level2_handle.output   # drain
        _ = level1_handle.output
    print("root output:", run.output)
```

### Example 2: SubgraphTransformer in a multi-agent graph

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage


def specialist_node(state: MessagesState) -> dict:
    return {"messages": [AIMessage(content="Specialist result")]}


specialist = (
    StateGraph(MessagesState)
    .add_node("specialist", specialist_node)
    .add_edge(START, "specialist")
    .add_edge("specialist", END)
    .compile()
)


def orchestrator_node(state: MessagesState) -> dict:
    result = specialist.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


orchestrator = (
    StateGraph(MessagesState)
    .add_node("orchestrate", orchestrator_node)
    .add_edge(START, "orchestrate")
    .add_edge("orchestrate", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "multi-agent-1"}}

with orchestrator.stream_events(
    {"messages": [HumanMessage(content="Do the task")]},
    cfg,
    version="v3",
) as run:
    for sub in run.subgraphs:
        print(f"Agent subgraph '{sub.graph_name}' at {sub.path}")
        for snap in sub.values:
            last = snap["messages"][-1]
            print(f"  agent said: {last.content}")
    print("orchestrator done:", run.output["messages"][-1].content)
```

---

## 9 · `LifecycleTransformer` + `LifecyclePayload`

**Module:** `langgraph.stream.transformers`  
**Imports:**
```python
from langgraph.stream.transformers import LifecycleTransformer, LifecyclePayload
```

`LifecycleTransformer` emits `LifecyclePayload` events whenever a subgraph starts or finishes (at **any** depth below the transformer's scope, unlike `SubgraphTransformer` which only sees direct children). Each payload is pushed to `run.lifecycle`.

### `LifecyclePayload` TypedDict

```python
class LifecyclePayload(TypedDict, total=False):
    event: SubgraphStatus           # "started" | "completed" | "failed" | "cancelled"
    namespace: list[str]            # namespace of the subgraph
    graph_name: NotRequired[str]    # optional graph name
    trigger_call_id: NotRequired[str]  # tool-call id that triggered the subgraph
    cause: NotRequired[LifecycleCause]
    error: NotRequired[str]         # only when event == "failed"
```

### Example 1: Monitoring all subgraph lifecycle events

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class S(TypedDict):
    v: int


def inc(s: S) -> dict:
    return {"v": s["v"] + 1}


inner = (
    StateGraph(S).add_node("i", inc).add_edge(START, "i").add_edge("i", END).compile()
)


def outer_fn(s: S) -> dict:
    r = inner.invoke({"v": s["v"]})
    return {"v": r["v"]}


graph = (
    StateGraph(S)
    .add_node("o", outer_fn)
    .add_edge(START, "o")
    .add_edge("o", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"v": 0},
    {"configurable": {"thread_id": "lifecycle-1"}},
    version="v3",
) as run:
    for payload in run.lifecycle:
        print(
            f"event={payload.get('event')} "
            f"ns={payload.get('namespace')} "
            f"graph={payload.get('graph_name')}"
        )
    _ = run.output
# event=started  ns=['o:0:i'] graph=...
# event=completed ns=['o:0:i'] graph=...
```

### Example 2: Error detection via lifecycle events

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class S(TypedDict):
    should_fail: bool
    v: int


def risky(s: S) -> dict:
    if s["should_fail"]:
        raise RuntimeError("inner node failed!")
    return {"v": s["v"] + 1}


inner = (
    StateGraph(S)
    .add_node("risky", risky)
    .add_edge(START, "risky")
    .add_edge("risky", END)
    .compile()
)


def safe_wrapper(s: S) -> dict:
    try:
        r = inner.invoke({"should_fail": s["should_fail"], "v": s["v"]})
        return {"v": r["v"]}
    except Exception:
        return {"v": -1}


graph = (
    StateGraph(S)
    .add_node("safe", safe_wrapper)
    .add_edge(START, "safe")
    .add_edge("safe", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"should_fail": True, "v": 0},
    {"configurable": {"thread_id": "lc-fail-1"}},
    version="v3",
) as run:
    for payload in run.lifecycle:
        if payload.get("event") == "failed":
            print(f"Subgraph failed: {payload.get('error')}")
        else:
            print(f"Lifecycle: {payload.get('event')}")
    _ = run.output
```

### Example 3: Async lifecycle monitoring

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class S(TypedDict):
    n: int


def inc(s: S) -> dict:
    return {"n": s["n"] + 5}


inner = (
    StateGraph(S).add_node("i", inc).add_edge(START, "i").add_edge("i", END).compile()
)


def wrap(s: S) -> dict:
    return {"n": inner.invoke({"n": s["n"]})["n"]}


root = (
    StateGraph(S)
    .add_node("w", wrap)
    .add_edge(START, "w")
    .add_edge("w", END)
    .compile(checkpointer=InMemorySaver())
)


async def main():
    async with await root.astream_events(
        {"n": 0},
        {"configurable": {"thread_id": "async-lc-1"}},
        version="v3",
    ) as run:
        async for payload in run.lifecycle:
            print(f"[lifecycle] {payload.get('event')} @ {payload.get('namespace')}")
        _ = await run.output


asyncio.run(main())
```

---

## 10 · `StreamChannel`

**Module:** `langgraph.stream.stream_channel`  
**Import:**
```python
from langgraph.stream.stream_channel import StreamChannel
```

`StreamChannel` is the **low-level single-consumer drainable queue** that underpins every projection (`run.values`, `run.messages`, etc.). You rarely construct it directly — the transformers do. Understanding it helps when you build custom transformers or need fan-out via `.tee()`.

### Key API

```python
class StreamChannel(Generic[T]):
    def __init__(self, name: str | None = None, *, maxlen: int | None = None) -> None: ...

    # Push (called by transformer):
    def push(self, item: T) -> None: ...

    # Lifecycle (called by mux):
    def close(self) -> None: ...
    def fail(self, error: BaseException) -> None: ...

    # Fan-out:
    def tee(self, n: int = 2) -> tuple[StreamChannel[T], ...]: ...
    async def atee(self, n: int = 2) -> tuple[StreamChannel[T], ...]: ...

    # Iteration (single-consumer):
    def __iter__(self) -> Iterator[T]: ...
    async def __aiter__(self) -> AsyncIterator[T]: ...
```

### Example 1: Building a custom transformer with `StreamChannel`

```python
from typing import Any
from langgraph.stream._types import StreamTransformer, ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel


class NodeTimingTransformer(StreamTransformer):
    """Record how many updates each node produces."""

    _native = True
    required_stream_modes = ("updates",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[dict[str, int]] = StreamChannel()
        self._counts: dict[str, int] = {}
        self._scope_list = list(scope)

    def init(self) -> dict[str, Any]:
        return {"node_counts": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "updates" and event["params"]["namespace"] == self._scope_list:
            data = event["params"]["data"]
            for node_name in data:
                self._counts[node_name] = self._counts.get(node_name, 0) + 1
            self._channel.push(dict(self._counts))
        return True

    def finalize(self) -> None:
        pass


# Use it
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    n: int


def node(s: State) -> dict:
    return {"n": s["n"] + 1}


graph = (
    StateGraph(State)
    .add_node("a", node)
    .add_node("b", node)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

with graph.stream_events(
    {"n": 0},
    {"configurable": {"thread_id": "custom-t-1"}},
    version="v3",
    transformers=[NodeTimingTransformer],
) as run:
    for counts in run.extensions["node_counts"]:
        print("Node update counts:", counts)
    _ = run.output
# Node update counts: {'a': 1}
# Node update counts: {'a': 1, 'b': 1}
```

### Example 2: Fan-out with `tee`

When you need two independent consumers of the same projection, `tee()` splits the channel. Both receive every item, but each has its own cursor.

```python
from langgraph.stream.stream_channel import StreamChannel

# Standalone demo (no graph needed)
ch: StreamChannel[str] = StreamChannel()
copy1, copy2 = ch.tee(2)

# In a real transformer you'd push to 'ch' and consumers iterate copy1/copy2.
```

### Example 3: Named channel — automatic protocol event forwarding

A channel constructed with a `name` (like `StreamChannel("lifecycle")`) is automatically wired by the `StreamMux` to also emit a `ProtocolEvent` with `method=name` (or `method="custom:<name>"` for non-native transformers) whenever `push()` is called. This is how `LifecycleTransformer` delivers its events over the wire to remote SDK clients.

```python
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream._types import StreamTransformer, ProtocolEvent
from typing import Any


class MyStatusTransformer(StreamTransformer):
    """Push a status payload visible to the wire protocol."""

    _native = True  # wire method is "my_status" (no "custom:" prefix)
    required_stream_modes = ("updates",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        # Named channel → forwarded as ProtocolEvent(method="my_status")
        self._channel: StreamChannel[dict] = StreamChannel("my_status")
        self._scope_list = list(scope)

    def init(self) -> dict[str, Any]:
        return {"my_status": self._channel}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "updates" and event["params"]["namespace"] == self._scope_list:
            self._channel.push({"node_ran": list(event["params"]["data"].keys())})
        return True
```

---

## Summary

| Class | Module | Use case |
|---|---|---|
| `GraphRunStream` | `langgraph.stream.run_stream` | Root v3 streaming handle; iterate `run.values`, `run.messages`, `run.output`, etc. |
| `AsyncGraphRunStream` | `langgraph.stream.run_stream` | Async twin; async iteration drives the graph without a background task |
| `SubgraphRunStream` | `langgraph.stream.run_stream` | In-process subgraph handle with own scoped projections; obtained from `run.subgraphs` |
| `ValuesTransformer` | `langgraph.stream.transformers` | Projects `values` events as full-state snapshots on `run.values` |
| `UpdatesTransformer` | `langgraph.stream.transformers` | Projects `updates` events as per-node deltas on `run.updates` |
| `CustomTransformer` | `langgraph.stream.transformers` | Projects `custom` events from `get_stream_writer()` on `run.custom` |
| `MessagesTransformer` | `langgraph.stream.transformers` | Projects `messages` events as `ChatModelStream` handles on `run.messages` |
| `ChatModelStream` | `langgraph.stream.transformers` | Per-LLM-call handle; `.text`, `.reasoning`, `.tool_calls`, `.output` projections |
| `SubgraphTransformer` | `langgraph.stream.transformers` | Discovers direct-child subgraphs; pushes `SubgraphRunStream` handles to `run.subgraphs` |
| `LifecycleTransformer` | `langgraph.stream.transformers` | Emits `LifecyclePayload` on every subgraph start/finish at all depths |
| `LifecyclePayload` | `langgraph.stream.transformers` | TypedDict carrying `event`, `namespace`, `graph_name`, `error` for lifecycle events |
| `StreamChannel` | `langgraph.stream.stream_channel` | Low-level single-consumer drainable queue; supports `tee()` for fan-out |

<a id="vol-index"></a>

## Vol. 1–15 index

| Volume | Classes covered |
|---|---|
| [Vol. 1](./langgraph_class_deep_dives/) | StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore |
| [Vol. 2](./langgraph_class_deep_dives_v2/) | RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl |
| [Vol. 3](./langgraph_class_deep_dives_v3/) | interrupt/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask, IndexConfig/TTLConfig |
| [Vol. 4](./langgraph_class_deep_dives_v4/) | set_node_defaults, add_sequence, input_schema/output_schema, context_schema/Runtime.context, get_stream_writer/StreamWriter, push_ui_message, entrypoint.final, REMOVE_ALL_MESSAGES, error_handler, error taxonomy |
| [Vol. 5](./langgraph_class_deep_dives_v5/) | RedisCache, EncryptedSerializer, JsonPlusSerializer, UntrackedValue, AnyValue, EmbeddingsLambda, BaseCheckpointSaver, typed StreamParts, task.clear_cache, HumanInterrupt |
| [Vol. 6](./langgraph_class_deep_dives_v6/) | GraphRunStream/AsyncGraphRunStream, StreamTransformer, StreamChannel, ValuesTransformer/CustomTransformer/UpdatesTransformer, GraphCallbackHandler, GraphInterruptEvent/GraphResumeEvent, GraphDrained, NodeTimeoutError, delete_ui_message, ProtocolEvent |
| [Vol. 7](./langgraph_class_deep_dives_v7/) | PregelProtocol/StreamProtocol, BackgroundExecutor, AsyncBatchedBaseStore, get_text_at_path/tokenize_path, SerdeEvent, BaseChannel, call()/SyncAsyncFuture, PregelScratchpad, StateNodeSpec, identifier/get_runnable_for_task |
| [Vol. 8](./langgraph_class_deep_dives_v8/) | ExecutionInfo/Runtime.heartbeat, ServerInfo/BaseUser, ReplayState, StreamMux, Call, ChannelWrite/ChannelWriteEntry, PregelRunner/FuturesDict, WritesProtocol/PregelTaskWrites, SyncPregelLoop/AsyncPregelLoop, DuplexStream |
| [Vol. 9](./langgraph_class_deep_dives_v9/) | ToolCallRequest.override(), Send+timeout, create_react_agent hooks, RetryPolicy chaining, CachePolicy key_func, InMemoryStore raw embeddings, context_schema+Runtime.context, Command.PARENT, TimeoutPolicy.coerce(), @entrypoint multi-policy retry |
| [Vol. 10](./langgraph_class_deep_dives_v10/) | Durability modes, NodeError/NodeCancelledError, TaskPayload, CheckpointPayload, Item/SearchItem, GetOp/PutOp/SearchOp/ListNamespacesOp/MatchCondition, UIMessage/RemoveUIMessage, GraphOutput v2, StreamPart variants, PregelExecutableTask/CacheKey |
| [Vol. 11](./langgraph_class_deep_dives_v11/) | InjectedState, InjectedStore, MessagesState, Overwrite, ToolOutputMixin, CheckpointMetadata, CheckpointTuple, StateUpdate, PersistentDict, DeltaChannelHistory |
| [Vol. 12](./langgraph_class_deep_dives_v12/) | RemoteGraph/RemoteException, PostgresSaver/ShallowPostgresSaver, AsyncPostgresSaver, PostgresStore/PoolConfig, AsyncPostgresStore, ANNIndexConfig/HNSWConfig/IVFFlatConfig, GraphRunStream/SubgraphRunStream, ToolCallWithContext/ToolInvocationError, LifecyclePayload/LifecycleTransformer, MessagesTransformer/CheckpointsTransformer/TasksTransformer |
| [Vol. 13](./langgraph_class_deep_dives_v13/) | (see page) |
| [Vol. 14](./langgraph_class_deep_dives_v14/) | (see page) |
| [Vol. 15](./langgraph_class_deep_dives_v15/) | Runtime, ExecutionInfo, RunControl, BaseStore, Item/SearchItem, GetOp/SearchOp/PutOp/ListNamespacesOp, IndexConfig/TTLConfig, UIMessage/push_ui_message, StreamTransformer/ProtocolEvent, RemoteGraph, NodeError/NodeTimeoutError/GraphDrained, IsLastStep/RemainingSteps, HumanResponse |
| **Vol. 16** (this page) | GraphRunStream, AsyncGraphRunStream, SubgraphRunStream, ValuesTransformer, UpdatesTransformer, CustomTransformer, MessagesTransformer/ChatModelStream, SubgraphTransformer, LifecycleTransformer/LifecyclePayload, StreamChannel |

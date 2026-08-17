---
title: "LangGraph Class Deep-Dives Vol. 46"
description: "Source-verified deep dives (langgraph==1.2.11) into 10 class groups: StreamTransformer + LifecycleTransformer (custom projection extension point, sync/async lanes, schedule()), ToolCallTransformer (tool lifecycle ToolCallStream handles, required_stream_modes), SubgraphTransformer (in-process SubgraphRunStream drill-down, child mini-mux scoping), TracePolicy (per-node process_inputs/process_outputs, PII redaction, payload summarisation), RetryPolicy (exponential backoff, custom retry_on callable, per-node vs per-task usage), TimeoutPolicy (run_timeout + idle_timeout dual-timer, refresh_on heartbeat, NodeTimeoutError), EncryptedSerializer (AES-EAX checkpoint encryption, from_pycryptodome_aes, CipherProtocol), InMemoryCache + BaseCache + CachePolicy (@task result reuse, TTL, compile(cache=)), InMemoryStore with semantic vector search (index= config, put(index=[...]), search(query=)), and ToolRuntime + InjectedState + InjectedStore (complete tool-injection API, field extraction, invisible-to-LLM params, runtime context access)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 46"
  order: 77
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.11`.

---

## 1 · `StreamTransformer` + `LifecycleTransformer` — custom stream projection extension point

**Module:** `langgraph.stream.transformers`

`StreamTransformer` is the abstract base class for all stream projections wired into the `StreamMux`. Subclasses observe raw protocol events and build typed derived outputs (`StreamChannel` instances, promises, counters, etc.). The mux calls `init()` once to collect the projection dict, then routes each `ProtocolEvent` through `process()` / `aprocess()` in registration order.

**Key source facts (`langgraph/stream/transformers.py`):**

- `init(self) -> dict[str, Any]` — return a dict whose values are the projection objects (typically `StreamChannel` instances). The mux stores these and, for transformers where `_native = True`, exposes them as direct attributes on the run stream.
- `process(self, event: ProtocolEvent) -> bool` — synchronous event handler. Return `True` to pass the event downstream; `False` to consume it without forwarding.
- `aprocess(self, event: ProtocolEvent) -> Awaitable[bool]` — async variant; overriding it opts the transformer into the async lane.
- `schedule(self, coro)` — dispatch async work from inside a sync `process` without blocking; the mux drains the coroutine before the next step. This also opts the transformer into the async lane automatically.
- `finalize()` / `fail(err)` — optional teardown hooks; default implementations are no-ops. `StreamChannel` projections are auto-closed by the mux so most transformers don't override these.
- `_native = True` — expose projection keys as direct attributes on the run stream (e.g. `run.lifecycle`, `run.tool_calls`).
- `LifecycleTransformer` is a built-in `_native` transformer that emits `LifecyclePayload` dicts to `run.lifecycle` for every subgraph start/finish event strictly below the transformer's scope.

### Example 1 — consume `run.lifecycle` to track subgraph durations

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
class State(TypedDict):
    value: int

def double(state: State) -> dict:
    return {"value": state["value"] * 2}

# Build a subgraph
sub = StateGraph(State)
sub.add_node("double", double)
sub.add_edge(START, "double")
sub.add_edge("double", END)
sub_compiled = sub.compile()

# Wire into parent
parent = StateGraph(State)
parent.add_node("sub", sub_compiled)
parent.add_edge(START, "sub")
parent.add_edge("sub", END)

# LifecycleTransformer is built-in to stream_events(version="v3") — compile() with no transformers arg
graph = parent.compile()

timings: list[dict] = []
start_times: dict[str, float] = {}

# stream_events(version="v3") returns a GraphRunStream with .lifecycle projection
run = graph.stream_events({"value": 3}, version="v3")
for event in run.lifecycle:
    ns = tuple(event["namespace"])
    if event["event"] == "started":
        start_times[str(ns)] = time.monotonic()
    elif event["event"] in ("completed", "failed", "interrupted", "drained"):
        started = start_times.pop(str(ns), None)
        if started is not None:
            timings.append({
                "namespace": ns,
                "status": event["event"],
                "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            })

for t in timings:
    print(t)
```

### Example 2 — write a custom `StreamTransformer` that counts node executions

```python
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream.transformers import StreamTransformer
from langgraph.stream.stream_channel import StreamChannel

class NodeCountTransformer(StreamTransformer):
    """Count how many times each named node executes."""

    _native = True  # expose as run.node_counts
    required_stream_modes = ("updates",)  # auto-enable the updates protocol in the mux

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[dict[str, int]] = StreamChannel()
        self._counts: dict[str, int] = {}

    def init(self) -> dict[str, Any]:
        return {"node_counts": self._channel}

    def process(self, event: dict) -> bool:
        # Updates events carry params["data"] — a dict mapping node name → update
        if event.get("method") == "updates":
            params = event.get("params", {})
            data = params.get("data", {})
            for node in data.keys():
                self._counts[node] = self._counts.get(node, 0) + 1
            if self._counts:
                self._channel.push(dict(self._counts))  # push snapshot after each step
        return True  # always pass events downstream

    def finalize(self) -> None:
        self._channel.close()


class State(TypedDict):
    steps: list[str]

def step_a(state: State) -> dict:
    return {"steps": state["steps"] + ["a"]}

def step_b(state: State) -> dict:
    return {"steps": state["steps"] + ["b"]}

graph = (
    StateGraph(State)
    .add_node("a", step_a)
    .add_node("b", step_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile(transformers=[NodeCountTransformer])
)

# stream_events(version="v3") exposes custom transformer projections via _native attributes
run = graph.stream_events({"steps": []}, version="v3")
final_counts: dict[str, int] = {}
for snapshot in run.node_counts:
    final_counts = snapshot

print("Node execution counts:", final_counts)
# Node execution counts: {'a': 1, 'b': 1}
```

### Example 3 — async transformer using `schedule()` for non-blocking side-effects

```python
import asyncio
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream.transformers import StreamTransformer
from langgraph.stream.stream_channel import StreamChannel

audit_log: list[dict] = []

class AsyncAuditTransformer(StreamTransformer):
    """Log every update event to an async audit store without blocking."""

    required_stream_modes = ("updates",)  # auto-enable the updates protocol in the mux

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._channel: StreamChannel[str] = StreamChannel()

    def init(self) -> dict[str, Any]:
        return {"audit": self._channel}

    def process(self, event: dict) -> bool:
        if event.get("method") == "updates":
            # schedule() fires a coroutine without holding up the pump
            self.schedule(self._log_event(event))
        return True

    async def _log_event(self, event: dict) -> None:
        await asyncio.sleep(0)  # simulate async I/O (DB write, HTTP call, etc.)
        # params["data"] is a dict mapping node name → update value
        data = event.get("params", {}).get("data", {})
        for node in data.keys():
            entry = {"method": "updates", "node": node}
            audit_log.append(entry)
            self._channel.push(f"logged: {node}")


class State(TypedDict):
    x: int

async def main():
    graph = (
        StateGraph(State)
        .add_node("inc", lambda s: {"x": s["x"] + 1})
        .add_edge(START, "inc")
        .add_edge("inc", END)
        .compile(transformers=[AsyncAuditTransformer])
    )
    # astream_events(version="v3") returns an AsyncGraphRunStream with the .audit projection
    run = graph.astream_events({"x": 0}, version="v3")
    async for _ in run.audit:
        pass
    print("Audit log:", audit_log)

asyncio.run(main())
```

---

## 2 · `ToolCallTransformer` — tool lifecycle as `ToolCallStream` handles

**Module:** `langgraph.prebuilt._tool_call_transformer`

`ToolCallTransformer` is a built-in `StreamTransformer` that converts raw `tools` channel protocol events into live `ToolCallStream` handles. Each handle tracks a single tool invocation from `tool-started` through `tool-output-delta` events to `tool-finished` or `tool-error`. Consumers iterate `run.tool_calls` to receive handles as they're created, then drain the handle's `output_deltas` for incremental output.

**Key source facts:**

- `required_stream_modes = ("tools",)` — the `tools` channel must be opted into at stream/invoke time for this transformer to receive events.
- `_native = True` — `run.tool_calls` is a first-class attribute on the run stream.
- `process()` returns `True` (pass-through) so the raw `tools` channel events still flow through to wire consumers.
- `ToolCallStream` exposes: `tool_call_id`, `tool_name`, `input`, `output` (final), `completed` (bool), `output_deltas` (async-iterable `StreamChannel`).
- Active handles are tracked in `_active: dict[str, ToolCallStream]`; `finalize()` closes any still-open handles after the run ends.
- Register by passing `transformers=[ToolCallTransformer]` to `compile()`.

### Example 1 — stream tool output deltas as they arrive

```python
import asyncio
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer

@tool
def count_words(text: str) -> str:
    """Count words in text."""
    count = len(text.split())
    return f"Word count: {count}"

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Minimal agent that calls count_words directly (no LLM needed for demo)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage

def agent_node(state: State) -> dict:
    # Simulate an LLM producing a tool call
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"name": "count_words", "args": {"text": "hello world foo"}, "id": "tc1", "type": "tool_call"}],
    )]}

def tool_node(state: State) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        result = count_words.invoke(tc["args"])
        results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"messages": results}

graph = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_node("tools", tool_node)
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    .compile(transformers=[ToolCallTransformer])
)

async def main():
    # astream_events(version="v3") returns AsyncGraphRunStream with .tool_calls projection
    run = graph.astream_events(
        {"messages": [HumanMessage(content="count these words")]},
        version="v3",
    )
    async for handle in run.tool_calls:
        print(f"Tool started: {handle.tool_name} (id={handle.tool_call_id})")
        print(f"  Input: {handle.input}")
        # Drain output deltas
        async for delta in handle.output_deltas:
            print(f"  Delta: {delta!r}")
        print(f"  Final output: {handle.output}")

asyncio.run(main())
```

### Example 2 — inspect tool calls synchronously after the run

```python
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def fake_agent(state: State) -> dict:
    return {"messages": [AIMessage(
        content="",
        tool_calls=[
            {"name": "add", "args": {"a": 3, "b": 4}, "id": "tc_add", "type": "tool_call"},
            {"name": "multiply", "args": {"a": 5, "b": 6}, "id": "tc_mul", "type": "tool_call"},
        ],
    )]}

def tool_exec(state: State) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        fn = {"add": add, "multiply": multiply}[tc["name"]]
        results.append(ToolMessage(content=str(fn.invoke(tc["args"])), tool_call_id=tc["id"]))
    return {"messages": results}

graph = (
    StateGraph(State)
    .add_node("agent", fake_agent)
    .add_node("tools", tool_exec)
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    .compile(transformers=[ToolCallTransformer])
)

# stream_events(version="v3") returns GraphRunStream with .tool_calls projection
completed_handles = []
run = graph.stream_events(
    {"messages": [HumanMessage(content="math")]},
    version="v3",
)
for handle in run.tool_calls:
    completed_handles.append(handle)

for h in completed_handles:
    print(f"{h.tool_name}({h.input}) → {h.output}")
# add({'a': 3, 'b': 4}) → 7
# multiply({'a': 5, 'b': 6}) → 30
```

### Example 3 — combine `ToolCallTransformer` with `LifecycleTransformer`

```python
import asyncio
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def fake_agent(state: State) -> dict:
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"name": "greet", "args": {"name": "Alice"}, "id": "tc1", "type": "tool_call"}],
    )]}

def tool_exec(state: State) -> dict:
    last = state["messages"][-1]
    results = [ToolMessage(content=greet.invoke(tc["args"]), tool_call_id=tc["id"]) for tc in last.tool_calls]
    return {"messages": results}

graph = (
    StateGraph(State)
    .add_node("agent", fake_agent)
    .add_node("tools", tool_exec)
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    # ToolCallTransformer must be registered at compile; LifecycleTransformer is built-in to v3
    .compile(transformers=[ToolCallTransformer])
)

async def main():
    # astream_events(version="v3") exposes both .tool_calls and .lifecycle projections
    run = graph.astream_events(
        {"messages": [HumanMessage(content="say hi")]},
        version="v3",
    )
    # Iterate tool_calls and lifecycle concurrently
    import asyncio
    async def drain_tools():
        async for handle in run.tool_calls:
            print(f"[TOOL] {handle.tool_name} → {handle.output}")

    async def drain_lifecycle():
        async for payload in run.lifecycle:
            print(f"[LIFECYCLE] {payload['event']} ns={payload['namespace']}")

    await asyncio.gather(drain_tools(), drain_lifecycle())

asyncio.run(main())
```

---

## 3 · `SubgraphTransformer` — in-process subgraph drill-down handles

**Module:** `langgraph.stream.transformers`

`SubgraphTransformer` discovers direct child subgraph invocations and builds `SubgraphRunStream` (sync) or `AsyncSubgraphRunStream` (async) handles. Each handle wraps a **child mini-mux** scoped to the subgraph's namespace, so consumers can iterate `handle.values`, `handle.messages`, `handle.lifecycle`, and `handle.subgraphs` (recursive grandchildren) on a per-subgraph basis.

**Key source facts:**

- Only discovers **direct children** — grandchildren appear on the child handle's own `subgraphs` log, not the root's. Depth is `len(scope) + 1`.
- The child mini-mux has its own `SubgraphTransformer`, enabling infinite recursive drill-down.
- `_native = True` → `run.subgraphs` is a direct attribute.
- `supports_sync = True` → works with both `stream()` and `astream()`.
- Each child handle's projections (`.values`, `.messages`, `.lifecycle`) are populated by the parent mux forwarding protocol events into the matching child mini-mux based on namespace prefix.

### Example 1 — drill into a subgraph's value stream

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def increment(state: State) -> dict:
    return {"count": state["count"] + 1}

# Build subgraph
sub = StateGraph(State)
sub.add_node("inc", increment)
sub.add_node("inc2", increment)
sub.add_edge(START, "inc")
sub.add_edge("inc", "inc2")
sub.add_edge("inc2", END)
sub_compiled = sub.compile()

# Wire into parent
parent = StateGraph(State)
parent.add_node("sub", sub_compiled)
parent.add_edge(START, "sub")
parent.add_edge("sub", END)
# SubgraphTransformer is built-in to stream_events(version="v3") — no transformers arg needed
graph = parent.compile()

async def main():
    # astream_events(version="v3") returns AsyncGraphRunStream with .subgraphs projection
    run = graph.astream_events({"count": 0}, version="v3")
    async for handle in run.subgraphs:
        print(f"Subgraph namespace: {handle.namespace}")
        async for value in handle.values:
            print(f"  subgraph value: {value}")
    # Also consume root values
    async for root_value in run.values:
        print(f"Root value: {root_value}")

asyncio.run(main())
```

### Example 2 — recursive grandchild subgraph drill-down

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    items: list[str]

def append_a(state: State) -> dict:
    return {"items": state["items"] + ["a"]}

def append_b(state: State) -> dict:
    return {"items": state["items"] + ["b"]}

# Innermost subgraph
inner = StateGraph(State)
inner.add_node("a", append_a)
inner.add_edge(START, "a")
inner.add_edge("a", END)
inner_compiled = inner.compile()

# Middle subgraph wraps inner
middle = StateGraph(State)
middle.add_node("inner", inner_compiled)
middle.add_node("b", append_b)
middle.add_edge(START, "inner")
middle.add_edge("inner", "b")
middle.add_edge("b", END)
middle_compiled = middle.compile()

# Root wraps middle
root = StateGraph(State)
root.add_node("middle", middle_compiled)
root.add_edge(START, "middle")
root.add_edge("middle", END)
# SubgraphTransformer is built-in to stream_events(version="v3")
graph = root.compile()

async def traverse(handle, depth: int = 0) -> None:
    indent = "  " * depth
    print(f"{indent}Subgraph: {handle.namespace}")
    # Grandchildren appear on handle.subgraphs (child mini-mux's own transformer)
    async for child in handle.subgraphs:
        await traverse(child, depth + 1)

async def main():
    run = graph.astream_events({"items": []}, version="v3")
    async for handle in run.subgraphs:
        await traverse(handle)

asyncio.run(main())
```

### Example 3 — subgraph message stream for a chat-style pipeline

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def classify(state: State) -> dict:
    return {"messages": [AIMessage(content="Classified as: question")]}

def respond(state: State) -> dict:
    return {"messages": [AIMessage(content="Here is the answer.")]}

# Sub-graph handles the response pipeline
response_sub = StateGraph(State)
response_sub.add_node("classify", classify)
response_sub.add_node("respond", respond)
response_sub.add_edge(START, "classify")
response_sub.add_edge("classify", "respond")
response_sub.add_edge("respond", END)
response_compiled = response_sub.compile()

# Root graph
root = StateGraph(State)
root.add_node("pipeline", response_compiled)
root.add_edge(START, "pipeline")
root.add_edge("pipeline", END)
# SubgraphTransformer is built-in to stream_events(version="v3")
graph = root.compile()

async def main():
    run = graph.astream_events(
        {"messages": [HumanMessage(content="What is LangGraph?")]},
        version="v3",
    )
    print("Root messages:")
    async for handle in run.subgraphs:
        print(f"  Messages from subgraph {handle.namespace}:")
        async for chunk, metadata in handle.messages:
            print(f"    [{metadata.get('langgraph_node')}] {chunk.content!r}")

asyncio.run(main())
```

---

## 4 · `TracePolicy` — per-node trace input/output transformation

**Module:** `langgraph.types`

`TracePolicy` is a frozen dataclass that controls what a node's LangSmith trace run records as its input and output. It acts as a sanitization layer: transform (summarise, redact, truncate) what is stored in the trace without affecting the actual data flowing through the graph.

**Key source facts (`langgraph/types.py`):**

- `process_inputs: Callable[[Any], Any] | None` — receives the node's raw input; returns the value written to the trace input field.
- `process_outputs: Callable[[Any], Any] | None` — receives the node's raw output; returns the value written to the trace output field.
- Scope is limited to the **node's own run** trace. Child runs created by a bound runnable and the root graph run are not affected.
- Pass to `add_node(..., trace_policy=TracePolicy(...))`. Plain function nodes use `trace=False` by default (no child runs).
- Not intended as a secrets redaction mechanism — use LangSmith's `hide_inputs`/`hide_outputs`/`anonymizer` for that.

### Example 1 — summarise large message history in traces

```python
from typing import Any
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import TracePolicy

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def compress_for_trace(value: Any) -> Any:
    """Keep only the last 2 messages in trace records."""
    if isinstance(value, dict) and "messages" in value:
        msgs = value["messages"]
        return {**value, "messages": msgs[-2:] if len(msgs) > 2 else msgs}
    return value

def my_llm_node(state: State) -> dict:
    # Simulate an LLM response
    return {"messages": [AIMessage(content="Here is a detailed answer.")]}

graph = StateGraph(State)
graph.add_node(
    "llm",
    my_llm_node,
    trace_policy=TracePolicy(
        process_inputs=compress_for_trace,
        process_outputs=compress_for_trace,
    ),
)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
compiled = graph.compile()

result = compiled.invoke({
    "messages": [HumanMessage(content=f"Message {i}") for i in range(10)],
})
print(f"Graph produced {len(result['messages'])} messages")
# Trace only stored the last 2 input messages
```

### Example 2 — redact sensitive fields from trace records

```python
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TracePolicy

class State(TypedDict):
    username: str
    api_key: str
    result: str

def _redact(value: Any) -> Any:
    """Remove api_key from whatever is being traced."""
    if isinstance(value, dict):
        return {k: ("***REDACTED***" if k == "api_key" else v) for k, v in value.items()}
    return value

def fetch_data(state: State) -> dict:
    # Uses state["api_key"] internally; we don't want it in traces
    return {"result": f"Data for {state['username']}"}

graph = StateGraph(State)
graph.add_node(
    "fetch",
    fetch_data,
    trace_policy=TracePolicy(process_inputs=_redact, process_outputs=_redact),
)
graph.add_edge(START, "fetch")
graph.add_edge("fetch", END)
compiled = graph.compile()

out = compiled.invoke({"username": "alice", "api_key": "sk-secret-123", "result": ""})
print(out["result"])
# The `fetch` node's own run trace records api_key=***REDACTED***
# Note: scope is the node's span only. The root graph run still receives the original
# invocation payload. For full-graph hiding use LangSmith's hide_inputs/hide_outputs
# or the anonymizer feature rather than relying on TracePolicy alone.
```

### Example 3 — omit outputs entirely for a privacy-sensitive node

```python
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TracePolicy

class State(TypedDict):
    pii_data: str
    processed: bool

def process_pii(state: State) -> dict:
    # Handles PII; neither input nor output should appear in traces
    return {"processed": True}

graph = StateGraph(State)
graph.add_node(
    "pii_processor",
    process_pii,
    trace_policy=TracePolicy(
        process_inputs=lambda _: {"pii_data": "<omitted>"},
        process_outputs=lambda _: {"processed": "<omitted>"},
    ),
)
graph.add_edge(START, "pii_processor")
graph.add_edge("pii_processor", END)
compiled = graph.compile()

result = compiled.invoke({"pii_data": "SSN: 123-45-6789", "processed": False})
print(result["processed"])  # True — real output flows through graph normally
```

---

## 5 · `RetryPolicy` — exponential backoff with custom `retry_on`

**Module:** `langgraph.types`

`RetryPolicy` is an immutable `NamedTuple` that configures per-node or per-task retry behaviour. LangGraph wraps the node in a retry loop: on failure it waits `initial_interval * backoff_factor^(attempt-1)` seconds (capped at `max_interval`, optionally jittered), then re-runs the node up to `max_attempts` total.

**Key source facts (`langgraph/types.py` + `langgraph/_internal/_retry.py`):**

- `initial_interval: float = 0.5` — wait before first retry (seconds).
- `backoff_factor: float = 2.0` — multiplier per attempt.
- `max_interval: float = 128.0` — ceiling on wait time.
- `max_attempts: int = 3` — total attempts including first try (not just retries).
- `jitter: bool = True` — add `random.uniform(0, interval)` noise to prevent thundering herd.
- `retry_on` — defaults to `default_retry_on` (retries on common transient exceptions). Can be a single exception class, a sequence, or `Callable[[Exception], bool]`.
- Wire to a node: `add_node("name", fn, retry_policy=RetryPolicy(...))`.
- Wire to a `@task`: `@task(retry_policy=RetryPolicy(...))`.
- Replaced deprecated `add_node(..., retry=...)` syntax (removed in v1.x).

### Example 1 — retry on transient HTTP errors

```python
import random
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

call_count = 0

class State(TypedDict):
    result: str

def flaky_api_call(state: State) -> dict:
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError(f"Transient failure on attempt {call_count}")
    return {"result": "success"}

graph = StateGraph(State)
graph.add_node(
    "api",
    flaky_api_call,
    retry_policy=RetryPolicy(
        initial_interval=0.01,   # fast for testing
        backoff_factor=2.0,
        max_attempts=5,
        jitter=False,
        retry_on=ConnectionError,
    ),
)
graph.add_edge(START, "api")
graph.add_edge("api", END)
compiled = graph.compile()

result = compiled.invoke({"result": ""})
print(result["result"])     # success
print(f"Total attempts: {call_count}")  # 3
```

### Example 2 — custom `retry_on` callable for selective retry logic

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

attempt = 0

class State(TypedDict):
    status_code: int

def api_node(state: State) -> dict:
    global attempt
    attempt += 1
    if attempt == 1:
        raise ValueError("HTTP 503: Service Unavailable")
    if attempt == 2:
        raise ValueError("HTTP 400: Bad Request")  # should NOT retry
    return {"status_code": 200}

def should_retry(exc: Exception) -> bool:
    """Only retry 5xx server errors, not 4xx client errors."""
    msg = str(exc)
    if "503" in msg or "502" in msg or "504" in msg:
        return True
    return False  # 4xx → don't retry, propagate immediately

graph = StateGraph(State)
graph.add_node(
    "call",
    api_node,
    retry_policy=RetryPolicy(
        initial_interval=0.01,
        max_attempts=5,
        retry_on=should_retry,
    ),
)
graph.add_edge(START, "call")
graph.add_edge("call", END)
compiled = graph.compile()

try:
    compiled.invoke({"status_code": 0})
except ValueError as e:
    print(f"Propagated: {e}")  # HTTP 400 — not retried
print(f"Attempts made: {attempt}")  # 2
```

### Example 3 — `RetryPolicy` on a Functional API `@task`

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

task_attempts = 0

@task(retry_policy=RetryPolicy(initial_interval=0.01, max_attempts=3, jitter=False))
async def unreliable_fetch(url: str) -> str:
    global task_attempts
    task_attempts += 1
    if task_attempts < 2:
        raise TimeoutError(f"Timeout fetching {url}")
    return f"Content from {url}"

@entrypoint(checkpointer=InMemorySaver())
async def workflow(url: str) -> str:
    content = await unreliable_fetch(url)
    return content

async def main():
    config = {"configurable": {"thread_id": "retry-demo"}}
    result = await workflow.ainvoke("https://example.com", config)
    print(result)               # Content from https://example.com
    print(f"Attempts: {task_attempts}")  # 2

asyncio.run(main())
```

---

## 6 · `TimeoutPolicy` — dual-timer node timeout with heartbeat

**Module:** `langgraph.types`

`TimeoutPolicy` is a frozen dataclass that configures per-node timeouts using two independent timers: a hard wall-clock cap (`run_timeout`) that never resets, and an idle cap (`idle_timeout`) that resets on progress signals or explicit heartbeats. Either or both may be set. When a timeout fires, `NodeTimeoutError` is raised; the node's `RetryPolicy` (if any) then decides whether to retry.

**Key source facts (`langgraph/types.py`):**

- `run_timeout: float | timedelta | None` — hard cap on total node attempt wall time. Never refreshed.
- `idle_timeout: float | timedelta | None` — max time without observable progress. Refreshed by `refresh_on`.
- `refresh_on: Literal["auto", "heartbeat"] = "auto"` — `"auto"` refreshes on standard graph callbacks and `runtime.heartbeat()`; `"heartbeat"` refreshes **only** on explicit `runtime.heartbeat()` calls.
- Pass a bare `float` to `add_node(..., timeout=5.0)` as a shorthand for `run_timeout=5.0`; internally coerced to `TimeoutPolicy`.
- Relies on asyncio cancellation — sync nodes that block the GIL won't be cancelled mid-execution.
- `runtime.heartbeat()` is available in nodes via the injected `Runtime` object (from `langgraph.runtime`).

### Example 1 — hard wall-clock cap with retry on timeout

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy, RetryPolicy

class State(TypedDict):
    result: str

attempt = 0

async def slow_node(state: State) -> dict:
    global attempt
    attempt += 1
    if attempt == 1:
        await asyncio.sleep(10)  # will be cancelled by timeout
    return {"result": f"ok on attempt {attempt}"}

graph = StateGraph(State)
graph.add_node(
    "slow",
    slow_node,
    timeout=TimeoutPolicy(run_timeout=0.05),  # 50ms hard cap
    retry_policy=RetryPolicy(initial_interval=0.01, max_attempts=3),
)
graph.add_edge(START, "slow")
graph.add_edge("slow", END)
compiled = graph.compile()

async def main():
    result = await compiled.ainvoke({"result": ""})
    print(result["result"])       # ok on attempt 2
    print(f"attempts: {attempt}") # 2

asyncio.run(main())
```

### Example 2 — idle timeout with explicit heartbeat

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy

class State(TypedDict):
    steps_done: int

async def long_running_node(state: State, runtime: Runtime) -> dict:
    """Simulate a node that does work in chunks and signals progress."""
    for i in range(5):
        await asyncio.sleep(0.02)    # 20ms of work per chunk
        runtime.heartbeat()          # resets the idle_timeout timer
    return {"steps_done": 5}

graph = StateGraph(State)
graph.add_node(
    "worker",
    long_running_node,
    timeout=TimeoutPolicy(
        idle_timeout=0.1,     # 100ms — reset by each heartbeat
        refresh_on="heartbeat",
    ),
)
graph.add_edge(START, "worker")
graph.add_edge("worker", END)
compiled = graph.compile()

async def main():
    result = await compiled.ainvoke({"steps_done": 0})
    print(result["steps_done"])  # 5 — heartbeats prevented idle timeout

asyncio.run(main())
```

### Example 3 — combine `run_timeout` and `idle_timeout`

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.errors import NodeTimeoutError

class State(TypedDict):
    output: str

async def mixed_timeout_node(state: State, runtime: Runtime) -> dict:
    # Heartbeat every 50ms; hard cap of 500ms; idle cap of 100ms
    for i in range(3):
        await asyncio.sleep(0.04)
        runtime.heartbeat()
    return {"output": "completed"}

graph = StateGraph(State)
graph.add_node(
    "work",
    mixed_timeout_node,
    timeout=TimeoutPolicy(
        run_timeout=1.0,      # hard 1s wall-clock cap
        idle_timeout=0.1,     # 100ms idle cap, reset by heartbeats
        refresh_on="heartbeat",
    ),
)
graph.add_edge(START, "work")
graph.add_edge("work", END)
compiled = graph.compile()

async def main():
    try:
        result = await compiled.ainvoke({"output": ""})
        print(result["output"])  # completed
    except NodeTimeoutError as e:
        print(f"Timed out: {e}")

asyncio.run(main())
```

---

## 7 · `EncryptedSerializer` — checkpoint AES encryption

**Module:** `langgraph.checkpoint.serde.encrypted`

`EncryptedSerializer` wraps any `SerializerProtocol` (defaulting to `JsonPlusSerializer`) with a `CipherProtocol` to encrypt checkpoint blobs at the serialization layer. Blob bytes are encrypted before being stored and decrypted transparently on load. This protects checkpoint data at rest without any application-level changes.

**Key source facts (`langgraph/checkpoint/serde/encrypted.py`):**

- Implements `SerializerProtocol`: `dumps_typed(obj) -> (type_str, bytes)` and `loads_typed((type_str, bytes)) -> obj`.
- The cipher name is appended to the type string as `"<typ>+<ciphername>"`, so the correct cipher is selected on load.
- `from_pycryptodome_aes(**kwargs)` is a factory that builds an AES-EAX authenticated cipher from a 16/24/32-byte key; reads `LANGGRAPH_AES_KEY` env var when no explicit `key` kwarg is provided. (Default mode is `AES.MODE_EAX` — not GCM; pass `mode=AES.MODE_GCM` to override.)
- Pass to `InMemorySaver(serde=EncryptedSerializer(...))` or any `BaseCheckpointSaver` that accepts a `serde` argument.
- Old unencrypted blobs (no `+` in type string) are loaded transparently, enabling zero-downtime migration.

### Example 1 — AES-EAX encrypted in-memory checkpointer

```python
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# 32-byte key → AES-256
aes_key = b"my-32-byte-secret-key-goes-here!"

class State(TypedDict):
    secret_data: str
    processed: bool

def handle_secret(state: State) -> dict:
    return {"processed": True}

graph = StateGraph(State)
graph.add_node("handler", handle_secret)
graph.add_edge(START, "handler")
graph.add_edge("handler", END)

# Wire AES encryption into the checkpointer
serde = EncryptedSerializer.from_pycryptodome_aes(key=aes_key)
checkpointer = InMemorySaver(serde=serde)
compiled = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "encrypted-thread"}}
result = compiled.invoke({"secret_data": "SSN: 123-45-6789", "processed": False}, config)
print(result["processed"])  # True

# Checkpoint blobs in the saver are AES-encrypted bytes
snapshot = compiled.get_state(config)
print(snapshot.values["processed"])  # True — decrypted on access
```

### Example 2 — environment-variable-driven key (with rotation caveat)

```python
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# Read AES key from environment at startup
os.environ["LANGGRAPH_AES_KEY"] = "another-32-byte-key-for-env-var!"

class State(TypedDict):
    value: int

def bump(state: State) -> dict:
    return {"value": state["value"] + 1}

serde = EncryptedSerializer.from_pycryptodome_aes()  # reads LANGGRAPH_AES_KEY
checkpointer = InMemorySaver(serde=serde)

graph = (
    StateGraph(State)
    .add_node("bump", bump)
    .add_edge(START, "bump")
    .add_edge("bump", END)
    .compile(checkpointer=checkpointer)
)
config = {"configurable": {"thread_id": "env-key-thread"}}
result = graph.invoke({"value": 0}, config)
print(result["value"])  # 1

# ⚠ Key rotation caveat: EncryptedSerializer stores no key-ID in the ciphertext type
# suffix — it only tags the cipher ("aes"). Replacing LANGGRAPH_AES_KEY while keeping
# existing checkpoints means the serializer has only the new key and cannot decrypt
# old blobs, which become permanently inaccessible.
#
# Safe rotation requires a multi-key migration strategy:
#   1. Keep the OLD key accessible (e.g. LANGGRAPH_AES_KEY_OLD).
#   2. Read each existing checkpoint through the old serde, re-encrypt with the new
#      serde, and write back — this must complete before the old key is removed.
#   3. Only retire the old key once all blobs have been migrated.
#
# This is a storage-layer concern outside EncryptedSerializer's scope; implement it
# in your checkpoint backend (database migration script, dual-read adapter, etc.).
```

### Example 3 — transparent migration from unencrypted to encrypted

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

class State(TypedDict):
    n: int

def step(state: State) -> dict:
    return {"n": state["n"] + 1}

graph_def = (
    StateGraph(State)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
)

# Phase 1: write a checkpoint with the unencrypted (default) serde
plain_saver = InMemorySaver()  # default JsonPlusSerializer
plain_graph = graph_def.compile(checkpointer=plain_saver)
config = {"configurable": {"thread_id": "migration-demo"}}
plain_graph.invoke({"n": 0}, config)

# Phase 2: migrate blobs to a new saver backed by the encrypted serde.
# In production this would be a database migration; here we copy InMemorySaver's
# internal storage directly so the same thread ID is accessible via enc_saver.
aes_key = b"migration-key-32bytes-exact!!!!!"
enc_serde = EncryptedSerializer.from_pycryptodome_aes(key=aes_key)
enc_saver = InMemorySaver(serde=enc_serde)

# Copy raw checkpoint storage so the old thread is accessible through enc_saver.
# EncryptedSerializer can fall back to plain-serde for blobs whose type tag lacks
# the '+aes' suffix, so existing unencrypted blobs are still readable.
enc_saver.storage = plain_saver.storage.copy()  # type: ignore[attr-defined]
enc_saver.writes = plain_saver.writes.copy()     # type: ignore[attr-defined]

enc_graph = graph_def.compile(checkpointer=enc_saver)

# Read the old (unencrypted) checkpoint through the new encrypted saver
snapshot = enc_graph.get_state(config)
print(snapshot.values["n"])  # 1 — old plaintext checkpoint is still readable

# Continue the thread; new checkpoints are written with AES encryption
result = enc_graph.invoke(None, config)
print(result["n"])  # 2 — subsequent checkpoint is encrypted
```

---

## 8 · `InMemoryCache` + `BaseCache` + `CachePolicy` — `@task` result reuse

**Module:** `langgraph.cache.memory`, `langgraph.cache.base`, `langgraph.types`

`BaseCache` is the abstract serialization-aware cache interface; `InMemoryCache` is the in-process implementation backed by a dict with optional TTL. Graphs and Functional API entrypoints accept a `cache=` argument at compile time; individual nodes or `@task` functions accept a `cache_policy=CachePolicy(...)` to control per-item TTL and whether results are cached at all.

**Key source facts:**

- `BaseCache.get(keys) / aget(keys)` — batch lookup, returns `{key: value}` for hits.
- `BaseCache.set(pairs) / aset(pairs)` — batch write with per-entry `(value, ttl_seconds | None)`.
- `InMemoryCache` uses `threading.RLock` for thread-safe sync access; `aget`/`aset` delegate to sync counterparts (no true async I/O).
- `CachePolicy(ttl: int | None = None)` — `ttl=None` means no expiry; `ttl=N` means N-second TTL.
- Wire to `@task`: `@task(cache_policy=CachePolicy(ttl=60))`.
- Wire to `StateGraph.compile(cache=InMemoryCache())` — applies to all `@task` calls within the graph.
- The `serde` parameter accepts any `SerializerProtocol` (default `JsonPlusSerializer(pickle_fallback=False)`). Swap in `EncryptedSerializer` to encrypt cached values.

### Example 1 — cache `@task` results across thread runs

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver

cache = InMemoryCache()
call_count = 0

@task(cache_policy=CachePolicy(ttl=300))  # cache for 5 minutes
async def expensive_computation(n: int) -> int:
    global call_count
    call_count += 1
    await asyncio.sleep(0.01)  # simulate work
    return n * n

@entrypoint(checkpointer=InMemorySaver(), cache=cache)
async def workflow(n: int) -> int:
    return await expensive_computation(n)

async def main():
    cfg = lambda tid: {"configurable": {"thread_id": tid}}
    r1 = await workflow.ainvoke(4, cfg("t1"))
    r2 = await workflow.ainvoke(4, cfg("t2"))  # different thread, same input → cache hit
    r3 = await workflow.ainvoke(5, cfg("t3"))  # different input → cache miss
    print(r1, r2, r3)        # 16 16 25
    print(f"Actual calls: {call_count}")  # 2 (not 3 — one cache hit)

asyncio.run(main())
```

### Example 2 — `InMemoryCache` with TTL expiry

```python
import asyncio
import time
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver

cache = InMemoryCache()
calls = []

@task(cache_policy=CachePolicy(ttl=1))  # 1-second TTL
async def ttl_task(key: str) -> str:
    calls.append(key)
    return f"result-for-{key}"

@entrypoint(checkpointer=InMemorySaver(), cache=cache)
async def wf(key: str) -> str:
    return await ttl_task(key)

async def main():
    cfg = lambda tid: {"configurable": {"thread_id": tid}}
    r1 = await wf.ainvoke("abc", cfg("th1"))  # miss → executes
    r2 = await wf.ainvoke("abc", cfg("th2"))  # hit  → from cache
    await asyncio.sleep(1.1)                   # TTL expires
    r3 = await wf.ainvoke("abc", cfg("th3"))  # miss → re-executes
    print(r1, r2, r3)           # result-for-abc x3
    print(f"Executions: {len(calls)}")  # 2 (cache hit on r2)

asyncio.run(main())
```

### Example 3 — clear cache by namespace

```python
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()

# Manually inspect set/get to understand the API
from langgraph.cache.base import Namespace, FullKey

ns_a = Namespace(("tasks", "expensive_computation"))
ns_b = Namespace(("tasks", "other_fn"))

key_a: FullKey = (ns_a, "hash-of-input-42")
key_b: FullKey = (ns_b, "hash-of-input-99")

cache.set({
    key_a: ({"result": 1764}, 60),   # TTL 60s
    key_b: ({"result": 9801}, None),  # no expiry
})

print(cache.get([key_a]))   # {key_a: {'result': 1764}}
print(cache.get([key_b]))   # {key_b: {'result': 9801}}

# Clear only namespace A
cache.clear(namespaces=[ns_a])
print(cache.get([key_a]))   # {} — evicted
print(cache.get([key_b]))   # {key_b: {'result': 9801}} — still present

# Clear everything
cache.clear()
print(cache.get([key_b]))   # {}
```

---

## 9 · `InMemoryStore` with semantic vector search — `index=`, `search(query=)`

**Module:** `langgraph.store.memory`

`InMemoryStore` is a fully in-process `BaseStore` implementation that supports both standard key-value CRUD and optional semantic (vector) search. Semantic search is disabled by default; enable it by passing an `index` config dict with embedding dimensions and an embedding callable.

**Key source facts (`langgraph/store/memory/__init__.py`):**

- `put(namespace, key, value, *, index=None | list[str])` — store an item; `index=[field_paths]` selects which string fields are embedded and indexed for search.
- `get(namespace, key, *, refresh_ttl=None)` — retrieve a single `Item` by key.
- `search(namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0)` — full-text or semantic search across a namespace prefix. `query` drives similarity search when an embedder is configured; `filter` is metadata equality matching.
- `delete(namespace, key)` — remove an item.
- `list_namespaces(*, prefix=None, suffix=None, max_depth=None, limit=100, offset=0)` — enumerate stored namespaces.
- `index={"dims": N, "embed": callable, "fields": ["field1", ...]}` — dims must match the embedder's output dimension; `fields` are JSONPath-like dot-paths into the value dict.
- `Item` has attributes: `namespace`, `key`, `value` (the stored dict), `created_at`, `updated_at`, `score` (similarity score when returned by `search`).

### Example 1 — basic key-value CRUD

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Store items in hierarchical namespaces
store.put(("users", "alice"), "profile", {"name": "Alice", "role": "admin"})
store.put(("users", "bob"), "profile", {"name": "Bob", "role": "user"})
store.put(("users", "alice"), "settings", {"theme": "dark", "lang": "en"})

# Retrieve by exact key
item = store.get(("users", "alice"), "profile")
print(item.value)  # {'name': 'Alice', 'role': 'admin'}
print(item.namespace)  # ('users', 'alice')

# Search within a namespace prefix (no embedder → keyword/filter only)
results = store.search(("users",), filter={"role": "admin"})
for r in results:
    print(r.key, r.value)  # profile {'name': 'Alice', 'role': 'admin'}

# Delete
store.delete(("users", "bob"), "profile")
print(store.get(("users", "bob"), "profile"))  # None
```

### Example 2 — semantic search with a custom embedder

```python
import math
from typing import Sequence
from langgraph.store.memory import InMemoryStore

# A tiny deterministic embedder for demonstration (not useful in practice)
VOCAB = ["python", "javascript", "agent", "graph", "memory", "tool"]

def toy_embed(texts: Sequence[str]) -> list[list[float]]:
    """Embed text as a unit vector over toy vocabulary."""
    embeddings = []
    for text in texts:
        vec = [float(w in text.lower()) for w in VOCAB]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        embeddings.append([x / norm for x in vec])
    return embeddings

store = InMemoryStore(index={
    "dims": len(VOCAB),
    "embed": toy_embed,
    "fields": ["text"],   # embed the "text" field of each stored item
})

# Store documents
docs = [
    ("docs", "d1", {"text": "Python agent with memory and tools"}),
    ("docs", "d2", {"text": "JavaScript graph rendering library"}),
    ("docs", "d3", {"text": "LangGraph: graph-based agent framework"}),
    ("docs", "d4", {"text": "Memory management in Python applications"}),
]
for ns_tail, key, value in docs:
    store.put(("docs",), key, value, index=["text"])

# Semantic search
results = store.search(("docs",), query="python agent memory", limit=3)
for r in results:
    print(f"[score={r.score:.3f}] {r.value['text']}")
```

### Example 3 — cross-thread long-term memory with `InjectedStore` in a node

```python
from typing import Any
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import InjectedStore

store = InMemoryStore()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

def remember_preference(state: State, store: Annotated[InMemoryStore, InjectedStore()]) -> dict:
    """Read user preferences from the store and inject into the reply."""
    item = store.get(("user_prefs",), state["user_id"])
    prefs = item.value if item else {}
    reply_text = f"Your prefs: {prefs}" if prefs else "No preferences stored yet."
    return {"messages": [AIMessage(content=reply_text)]}

def save_preference(state: State, store: Annotated[InMemoryStore, InjectedStore()]) -> dict:
    """Write the latest preference back to the store."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if last_human and "theme:" in last_human.content:
        theme = last_human.content.split("theme:")[-1].strip()
        existing = store.get(("user_prefs",), state["user_id"])
        prefs = existing.value if existing else {}
        prefs["theme"] = theme
        store.put(("user_prefs",), state["user_id"], prefs)
    return {}

graph = (
    StateGraph(State)
    .add_node("remember", remember_preference)
    .add_node("save", save_preference)
    .add_edge(START, "save")
    .add_edge("save", "remember")
    .add_edge("remember", END)
    .compile(store=store)
)

state1 = graph.invoke({
    "messages": [HumanMessage(content="set theme: dark")],
    "user_id": "alice",
})
print(state1["messages"][-1].content)  # Your prefs: {'theme': 'dark'}

state2 = graph.invoke({
    "messages": [HumanMessage(content="hello")],
    "user_id": "alice",
})
print(state2["messages"][-1].content)  # Your prefs: {'theme': 'dark'}
```

---

## 10 · `ToolRuntime` + `InjectedState` + `InjectedStore` — complete tool injection API

**Module:** `langgraph.prebuilt` (re-exported from `langgraph.prebuilt.tool_node`)

These three annotations give tools access to graph state, the persistent store, and the full runtime context — all invisible to the LLM's tool-calling interface. The LLM never sees these parameters in the schema; `ToolNode` injects them automatically at execution time.

**Key source facts:**

- `InjectedState(field: str | None = None)` — inject the full graph state dict, or a single field if `field` is specified. Use `Annotated[T, InjectedState("fieldname")]` to extract just one key.
- `InjectedStore()` — inject the `BaseStore` attached to the graph. Use `Annotated[InMemoryStore, InjectedStore()]`.
- `ToolRuntime` — inject a rich runtime context object. Just annotate a parameter as `runtime: ToolRuntime` (no `Annotated` wrapper needed). Provides: `state`, `tool_call_id`, `config`, `context`, `store`, `stream_writer`, `tools`.
- All three annotations are subclasses of `InjectedToolArg`; `ToolNode` strips them from the schema before sending to the LLM.
- Distinct from `Runtime` (injected into **nodes/middleware**): `ToolRuntime` adds `tool_call_id`, `state`, and `tools` — tool-specific attributes.

### Example 1 — `InjectedState` for a field-specific state injection

```python
from typing import Optional
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_user: str

@tool
def get_user_info(
    detail: str,
    current_user: Annotated[str, InjectedState("current_user")],  # extract one field
) -> str:
    """Get information about the current user. detail can be 'name' or 'role'."""
    if detail == "name":
        return f"Current user: {current_user}"
    return f"Role for {current_user}: admin"

def agent_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"name": "get_user_info", "args": {"detail": "name"}, "id": "tc1", "type": "tool_call"}],
    )]}

graph = (
    StateGraph(AgentState)
    .add_node("agent", agent_node)
    .add_node("tools", ToolNode([get_user_info]))
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    .compile()
)

result = graph.invoke({
    "messages": [HumanMessage(content="who am I?")],
    "current_user": "alice",
})
tool_msg = result["messages"][-1]
print(tool_msg.content)  # Current user: alice
```

### Example 2 — `InjectedStore` for persistent cross-session memory in a tool

```python
from typing import Optional
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

@tool
def remember(
    fact: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[InMemoryStore, InjectedStore()],
) -> str:
    """Remember a fact about the user. Stored persistently across sessions."""
    existing = store.get(("facts",), user_id)
    facts = existing.value.get("facts", []) if existing else []
    facts.append(fact)
    store.put(("facts",), user_id, {"facts": facts})
    return f"Remembered: {fact}"

@tool
def recall(
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[InMemoryStore, InjectedStore()],
) -> str:
    """Recall all stored facts about the current user."""
    item = store.get(("facts",), user_id)
    facts = item.value.get("facts", []) if item else []
    return f"Facts: {facts}" if facts else "No facts stored."

def agent_node(state: State) -> dict:
    return {"messages": [AIMessage(
        content="",
        tool_calls=[
            {"name": "remember", "args": {"fact": "likes dark mode"}, "id": "tc1", "type": "tool_call"},
        ],
    )]}

graph = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_node("tools", ToolNode([remember, recall]))
    .add_edge(START, "agent")
    .add_edge("agent", "tools")
    .add_edge("tools", END)
    .compile(store=store)
)

result = graph.invoke({
    "messages": [HumanMessage(content="remember something")],
    "user_id": "bob",
})
print(result["messages"][-1].content)  # Remembered: likes dark mode
```

### Example 3 — `ToolRuntime` for full runtime context access inside a tool

```python
import asyncio
from typing import Optional
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.store.memory import InMemoryStore

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

@tool
async def stream_report(
    topic: str,
    runtime: ToolRuntime,  # no Annotated needed — matched by name+type
) -> str:
    """Generate a streaming report on a topic, writing chunks as they arrive.

    Args:
        topic: The topic to report on.
    """
    # Access state without InjectedState
    user_id = runtime.state.get("user_id", "unknown")
    # Access the tool_call_id for correlation
    call_id = runtime.tool_call_id
    # Stream intermediate output to the caller
    for i in range(3):
        runtime.stream_writer({"chunk": i, "topic": topic, "user": user_id})
    # Access all available tools (for tool-calling chains)
    available = [t.name for t in (runtime.tools or [])]
    return f"Report on '{topic}' for user {user_id} (call={call_id}). Tools: {available}"

def fake_agent(state: State) -> dict:
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"name": "stream_report", "args": {"topic": "AI trends"}, "id": "tc_rep", "type": "tool_call"}],
    )]}

async def main():
    graph = (
        StateGraph(State)
        .add_node("agent", fake_agent)
        .add_node("tools", ToolNode([stream_report]))
        .add_edge(START, "agent")
        .add_edge("tools", END)
        .add_edge("agent", "tools")
        .compile()
    )

    chunks = []
    async for event in graph.astream(
        {"messages": [HumanMessage(content="report")], "user_id": "carol"},
        stream_mode="custom",
    ):
        chunks.append(event)

    # Final tool message
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content="report")], "user_id": "carol"},
    )
    print(state["messages"][-1].content)

asyncio.run(main())
```

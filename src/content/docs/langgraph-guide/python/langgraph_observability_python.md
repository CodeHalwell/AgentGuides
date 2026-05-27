---
title: "LangGraph Observability and Monitoring (Python)"
description: "Runtime.stream_writer for custom events, ExecutionInfo for structured logging, state history inspection, LangSmith tracing, and OpenTelemetry — source-verified for LangGraph 1.2.2."
framework: langgraph
language: python
---

# LangGraph Observability and Monitoring (Python)

Verified against **`langgraph==1.2.2`** (modules: `langgraph.runtime`, `langgraph.types`, `langgraph.graph.state`).

LangGraph provides several built-in mechanisms for observing what your graph is doing — without external instrumentation libraries:

| Mechanism | API | What you see |
|---|---|---|
| **Custom stream events** | `runtime.stream_writer(...)` | Arbitrary events from inside any node |
| **Execution metadata** | `runtime.execution_info` | Thread ID, task ID, attempt number, retry timing |
| **State snapshots** | `graph.get_state()` | Current state at any point |
| **Full state history** | `graph.get_state_history()` | Every checkpoint in the thread |
| **Debug streaming** | `stream_mode="debug"` | Node start/end events with timing |
| **LangSmith tracing** | Environment variables | Automatic traces in LangSmith |

---

## 1. `runtime.stream_writer` — emit custom events from nodes

`runtime.stream_writer` is a callable that emits any JSON-serialisable value on the `"custom"` stream channel. Clients reading the stream with `stream_mode="custom"` (or the combined `["values", "custom"]`) receive these events in real time.

### 1.1 Node emitting progress events

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_anthropic import ChatAnthropic


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str


model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


def research_node(state: State, runtime: Runtime) -> dict:
    """A multi-step node that emits progress events."""
    # Emit start event
    runtime.stream_writer({"event": "research_start", "query": state["query"]})

    # Step 1 — search
    runtime.stream_writer({"event": "step", "name": "web_search", "status": "running"})
    search_results = web_search(state["query"])
    runtime.stream_writer({"event": "step", "name": "web_search", "status": "done",
                           "result_count": len(search_results)})

    # Step 2 — summarize
    runtime.stream_writer({"event": "step", "name": "summarize", "status": "running"})
    response = model.invoke([
        HumanMessage(f"Summarize these results: {search_results}")
    ])
    runtime.stream_writer({"event": "step", "name": "summarize", "status": "done"})

    # Emit completion event
    runtime.stream_writer({"event": "research_done", "query": state["query"]})
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("research", research_node)
builder.add_edge(START, "research")
builder.add_edge("research", END)
graph = builder.compile(checkpointer=InMemorySaver())


# Consume custom events alongside state updates
config = {"configurable": {"thread_id": "obs-1"}}
for event in graph.stream(
    {"query": "Latest AI breakthroughs"},
    config,
    stream_mode=["values", "custom"],  # receive both state values AND custom events
):
    mode, data = event  # tuple: ("values", state_dict) or ("custom", your_payload)
    if mode == "custom":
        print(f"[custom] {data}")
    elif mode == "values":
        print(f"[state] messages: {len(data.get('messages', []))}")
```

### 1.2 Streaming with `get_runtime()` — when you can't annotate the function

If you're wrapping third-party code that has a fixed signature, use `get_runtime()` instead of parameter injection:

```python
from langgraph.runtime import get_runtime


def third_party_node(state: State) -> dict:
    """Fixed signature — can't add a runtime parameter."""
    runtime = get_runtime()
    runtime.stream_writer({"event": "third_party_start"})
    result = run_third_party_lib(state["query"])
    runtime.stream_writer({"event": "third_party_done"})
    return {"result": result}
```

> `get_runtime()` raises `RuntimeError` if called outside a Pregel task context (i.e. outside an active graph execution).

---

## 2. `ExecutionInfo` — structured execution metadata

`runtime.execution_info` provides structured access to IDs and counters that are otherwise scattered across `RunnableConfig` and the Pregel scratchpad. Use it for correlation IDs in logs and metrics.

```python
# Source-verified fields (langgraph.runtime.ExecutionInfo, langgraph 1.2.2):
@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    checkpoint_id:           str
    checkpoint_ns:           str
    task_id:                 str
    thread_id:               str | None          # None if no checkpointer
    run_id:                  str | None          # None if not in RunnableConfig
    node_attempt:            int = 1             # 1-indexed; increments on retries
    node_first_attempt_time: float | None = None # unix timestamp of first attempt
```

### 2.1 Structured logging with correlation IDs

```python
import logging
import time
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


def observable_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info

    logger.info(
        "Node started",
        extra={
            "thread_id":   info.thread_id,
            "task_id":     info.task_id,
            "run_id":      info.run_id,
            "attempt":     info.node_attempt,
        },
    )

    if info.node_attempt > 1:
        elapsed = time.time() - (info.node_first_attempt_time or time.time())
        logger.warning(
            "Node retry",
            extra={"attempt": info.node_attempt, "elapsed_since_first": elapsed},
        )
        # Emit a retry event to the custom stream
        runtime.stream_writer({"event": "retry", "attempt": info.node_attempt})

    result = do_work(state["query"])

    logger.info("Node completed", extra={"task_id": info.task_id})
    return {"result": result}
```

### 2.2 Emitting metrics per node

```python
from langgraph.runtime import Runtime
import time


def metered_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info
    start = time.monotonic()

    try:
        result = do_expensive_work(state["query"])
        status = "success"
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start
        # Emit a metric event — consumed by your monitoring pipeline
        runtime.stream_writer({
            "metric": "node_duration_seconds",
            "value": duration,
            "labels": {
                "thread_id": info.thread_id,
                "status":    status,
                "attempt":   info.node_attempt,
            },
        })

    return {"result": result}
```

---

## 3. State history inspection — `get_state` and `get_state_history`

`CompiledStateGraph` provides methods to inspect the state at any checkpoint:

### 3.1 Current state snapshot

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "inspect-1"}}

# Run the graph
graph.invoke({"query": "test"}, config)

# Get the current state
snapshot = graph.get_state(config)

print(f"Values:   {snapshot.values}")       # current state dict
print(f"Next:     {snapshot.next}")         # which nodes run next (empty = done)
print(f"Created:  {snapshot.created_at}")  # ISO timestamp of this checkpoint
print(f"Metadata: {snapshot.metadata}")    # step number, source, writes
```

### 3.2 Full checkpoint history (time-travel)

```python
# List every checkpoint for a thread in reverse chronological order
history = list(graph.get_state_history(config))
print(f"Total checkpoints: {len(history)}")

for i, snapshot in enumerate(history):
    cp_id = snapshot.config["configurable"]["checkpoint_id"]
    print(f"\nCheckpoint {i}: {cp_id}")
    print(f"  Next node: {snapshot.next}")
    print(f"  State keys: {list(snapshot.values.keys())}")
    print(f"  Step: {snapshot.metadata.get('step', '?')}")
```

### 3.3 Replay from a specific checkpoint (time-travel)

```python
# Get the state from 2 checkpoints ago
old_snapshot = history[2]
old_config = old_snapshot.config   # includes checkpoint_id

# Re-invoke from that historical checkpoint
result = graph.invoke(None, old_config)
print(f"Replayed from checkpoint: {result}")
```

### 3.4 `get_subgraphs` — inspect nested graph state

When your graph contains subgraphs, `aget_subgraphs()` (async) traverses the nested execution:

```python
async def inspect_subgraphs():
    config = {"configurable": {"thread_id": "sub-1"}}
    async for name, subgraph in graph.aget_subgraphs(recurse=True):
        state = await subgraph.aget_state(config)
        print(f"Subgraph '{name}': {list(state.values.keys())}")
```

---

## 4. Debug streaming — built-in node event tracing

`stream_mode="debug"` emits structured events for every node start, node end, and checkpoint write — no extra instrumentation required.

```python
config = {"configurable": {"thread_id": "debug-1"}}

for event in graph.stream({"query": "test"}, config, stream_mode="debug"):
    event_type = event.get("type")

    if event_type == "task":
        # A node task is about to start
        print(f"→ Starting: {event['payload']['name']} (step {event['step']})")

    elif event_type == "task_result":
        # A node task completed
        node = event["payload"]["name"]
        error = event["payload"].get("error")
        if error:
            print(f"✗ Failed:   {node} — {error}")
        else:
            print(f"✓ Done:     {node}")

    elif event_type == "checkpoint":
        print(f"⬛ Checkpoint written (step {event['step']})")
```

---

## 5. LangSmith tracing — automatic, zero-code

Set two environment variables and every graph run is automatically traced in LangSmith — node-by-node latency, inputs/outputs, errors, retry counts.

```bash
export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="my-agent"   # optional; defaults to "default"
```

No code changes needed. Every `graph.invoke()` and `graph.stream()` call will appear in the LangSmith UI with:

- Node-by-node execution timeline and latency
- Full input/output state at each node
- Error details and retry counts
- Thread and run IDs for correlation

### 5.1 Tagging runs for filtering

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "configurable": {"thread_id": "user-123"},
    "tags": ["production", "user-query"],       # filter runs by tag in LangSmith
    "metadata": {"user_id": "user-123",         # attach arbitrary metadata
                  "env": "prod"},
    "run_name": "customer_support_query",       # human-readable name in LangSmith
}

result = graph.invoke({"query": "help me"}, config)
```

---

## 6. OpenTelemetry — distributed tracing

For distributed tracing across services, use OpenTelemetry. Instead of wrapping node functions manually (which hides the true node boundary), use `stream_mode="debug"` events as the trigger for span creation:

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("langgraph-app")


async def run_with_otel_tracing(query: str) -> dict:
    config = {"configurable": {"thread_id": "otel-1"}}
    spans: dict[str, trace.Span] = {}
    result = {}

    async for event in graph.astream(
        {"query": query}, config, stream_mode="debug"
    ):
        event_type = event.get("type")

        if event_type == "task":
            node_name = event["payload"]["name"]
            span = tracer.start_span(
                f"langgraph.node.{node_name}",
                attributes={
                    "graph.node":    node_name,
                    "graph.step":    event["step"],
                    "graph.thread":  config["configurable"]["thread_id"],
                },
            )
            spans[node_name] = span

        elif event_type == "task_result":
            node_name = event["payload"]["name"]
            span = spans.pop(node_name, None)
            if span:
                error = event["payload"].get("error")
                if error:
                    span.set_status(Status(StatusCode.ERROR, error))
                else:
                    span.set_status(Status(StatusCode.OK))
                span.end()

        elif event_type == "checkpoint":
            result = event["payload"].get("values", {})

    return result
```

---

## 7. Monitoring checklist

| What to monitor | Signal | How to get it |
|---|---|---|
| Node latency | Duration between `task` and `task_result` debug events | `stream_mode="debug"` |
| Retry rate | `runtime.execution_info.node_attempt > 1` | `runtime.stream_writer({"metric": "retry"})` |
| Error rate | `task_result` events with `error` set | `stream_mode="debug"` |
| Active threads | Thread IDs in use | Track in your own DB keyed by `thread_id` |
| State size | `len(json.dumps(state.values))` | After each `get_state()` call |
| Token usage | LangSmith traces OR `usage_metadata` on AI messages | LangSmith or parse responses |
| Cache hit rate | Custom event from node before/after cache check | `runtime.stream_writer({"cache": "hit"})` |

---

## See also

- [`reference-runtime-and-managed-values.md`](/langgraph-guide/python/reference-runtime-and-managed-values/) — `Runtime`, `ExecutionInfo`, `ToolRuntime` full API reference
- [`reference-streaming-modes.md`](/langgraph-guide/python/reference-streaming-modes/) — all stream modes including `"debug"` and `"custom"`
- [`chapter-06-streaming-and-debugging.md`](/langgraph-guide/python/chapter-06-streaming-and-debugging/) — `get_state`, `update_state`, `get_state_history`
- [`langgraph_advanced_error_recovery.md`](/langgraph-guide/python/langgraph_advanced_error_recovery/) — `RetryPolicy`, `TimeoutPolicy`, `error_handler`

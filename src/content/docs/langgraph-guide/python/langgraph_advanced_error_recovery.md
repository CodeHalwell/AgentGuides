---
title: "LangGraph Advanced Error Handling and Recovery (Python)"
description: "Native RetryPolicy, TimeoutPolicy, node-level error_handler, dead-letter patterns, checkpoint-based resumption, and structured error types (NodeError, NodeTimeoutError, NodeCancelledError, GraphRecursionError, ErrorCode) — source-verified for LangGraph 1.2.11."
framework: langgraph
language: python
---

# LangGraph Advanced Error Handling and Recovery (Python)

Verified against **`langgraph==1.2.11`** (modules: `langgraph.types`, `langgraph.graph.state`, `langgraph.runtime`, `langgraph.errors`).

LangGraph provides first-class primitives for every layer of error handling, with no external retry library needed:

| Layer | Primitive | Scope |
|---|---|---|
| **Automatic retry** | `RetryPolicy` on `add_node` | Transient exceptions re-invoke the node |
| **Timeout** | `TimeoutPolicy` on `add_node` | Prevents nodes from hanging forever |
| **Node error handler** | `error_handler=` on `add_node` | Custom fallback logic per node |
| **Graph-wide handler** | `set_node_defaults(error_handler=...)` | One fallback for every node |
| **Dead-letter routing** | Conditional edge returning `END` | Graceful degradation without raising |
| **Checkpoint resumption** | `graph.invoke(None, config)` | Re-play from the last successful step |

---

## 1. `RetryPolicy` — automatic retry with backoff

`RetryPolicy` is a `NamedTuple` that wraps node execution in an automatic retry loop with configurable backoff and jitter. Verified source:

```python
# langgraph.types (source-verified, langgraph 1.2.2)
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5      # seconds before the first retry
    backoff_factor:   float = 2.0      # multiplier applied after each retry
    max_interval:     float = 128.0    # cap on the inter-retry wait (seconds)
    max_attempts:     int   = 3        # total attempts including the first
    jitter:           bool  = True     # add random jitter to intervals
    retry_on = default_retry_on        # default: httpx 5xx, transport errors
```

The default `retry_on` catches `httpx.HTTPStatusError` (5xx), `httpx.TransportError`, `ConnectionError`, and request timeouts. Anything else propagates immediately.

### 1.1 Basic retry on transient errors

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
import httpx


class State(TypedDict):
    query: str
    result: str
    error: str


def call_external_api(state: State) -> dict:
    """Hits an external API that may fail transiently."""
    resp = httpx.get(
        "https://api.example.com/search",
        params={"q": state["query"]},
        timeout=10.0,
    )
    resp.raise_for_status()
    return {"result": resp.json()["answer"]}


builder = StateGraph(State)
builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        initial_interval=1.0,
        backoff_factor=2.0,    # 1 s → 2 s → 4 s
        max_interval=30.0,
        max_attempts=4,        # 1 original + 3 retries
        jitter=True,
    ),
)
builder.add_edge(START, "api_call")
builder.add_edge("api_call", END)
graph = builder.compile()
```

### 1.2 Custom `retry_on` predicate

Pass a callable for fine-grained control — e.g. retry on 429 rate-limits but NOT on 4xx client errors.

```python
from langgraph.types import RetryPolicy
import httpx


def should_retry(exc: Exception) -> bool:
    """Retry on rate-limits and server errors only."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True
    return False


builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        initial_interval=2.0,
        max_attempts=5,
        retry_on=should_retry,
    ),
)
```

### 1.3 Ordered fallback retry sequence

Pass a **list** of `RetryPolicy` objects. The **first** policy whose `retry_on` matches the exception wins.

```python
from langgraph.types import RetryPolicy
import httpx


rate_limit_policy = RetryPolicy(
    initial_interval=1.0,
    max_attempts=3,
    retry_on=lambda e: isinstance(e, httpx.HTTPStatusError)
        and e.response.status_code == 429,
)

server_error_policy = RetryPolicy(
    initial_interval=5.0,
    backoff_factor=3.0,
    max_attempts=2,
    retry_on=lambda e: isinstance(e, httpx.HTTPStatusError)
        and e.response.status_code >= 500,
)

builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=[rate_limit_policy, server_error_policy],
)
```

### 1.4 Using `ExecutionInfo` for idempotent retries

When a node is retried, `runtime.execution_info.node_attempt` increments (1-indexed). Use this to issue idempotency keys so external APIs aren't double-charged.

```python
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
import time


async def idempotent_payment_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info

    if info.node_attempt > 1:
        elapsed = time.time() - (info.node_first_attempt_time or time.time())
        print(f"Retry #{info.node_attempt} after {elapsed:.1f}s")

    # task_id is stable across retries — safe idempotency key
    result = await post_payment(
        idempotency_key=info.task_id,
        amount=state["amount"],
    )
    return {"result": result}


builder.add_node(
    "payment",
    idempotent_payment_node,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0),
)
```

---

## 2. `TimeoutPolicy` — prevent nodes from hanging

`TimeoutPolicy` cancels an async node attempt if it runs too long. Two cancellation modes:

```python
# langgraph.types (source-verified, langgraph 1.2.2)
@dataclass
class TimeoutPolicy:
    run_timeout:  float | timedelta | None = None   # hard wall-clock cap
    idle_timeout: float | timedelta | None = None   # max time without progress
    refresh_on:   Literal["auto", "heartbeat"] = "auto"
```

> **Sync nodes cannot be timed out.** Only async nodes support `timeout=`. Use `asyncio.to_thread()` to make a blocking call cancellable.

### 2.1 Hard wall-clock timeout

```python
from langgraph.types import TimeoutPolicy, RetryPolicy


async def slow_llm_call(state: State) -> dict:
    import asyncio
    await asyncio.sleep(0)  # yield so cancellation can fire
    response = await model.ainvoke(state["messages"])
    return {"result": response.content}


builder.add_node(
    "llm",
    slow_llm_call,
    timeout=TimeoutPolicy(run_timeout=30.0),     # fail after 30 s
    retry_policy=RetryPolicy(max_attempts=2),    # retry once on timeout
)
```

### 2.2 Idle timeout with `runtime.heartbeat()`

Use `idle_timeout` when your node processes a stream — if no progress event arrives within the window, the node is cancelled even if the `run_timeout` hasn't fired yet.

```python
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.runtime import Runtime


async def streaming_node(state: State, runtime: Runtime) -> dict:
    chunks: list[str] = []
    async for chunk in call_streaming_api(state["query"]):
        chunks.append(chunk)
        runtime.heartbeat()   # reset the idle timer — we're still receiving data

    return {"result": "".join(chunks)}


builder.add_node(
    "stream_call",
    streaming_node,
    timeout=TimeoutPolicy(
        idle_timeout=30.0,        # cancel if no chunk arrives for 30 s
        refresh_on="heartbeat",   # ONLY runtime.heartbeat() resets the timer
    ),
    retry_policy=RetryPolicy(max_attempts=2),
)
```

---

## 3. Node-level `error_handler`

The `error_handler=` parameter on `add_node` specifies a **fallback node function** that runs if the main node raises an exception that is not handled by retries. The handler receives the same state plus the exception object — it can log, emit a metric, write a degraded result, or route to a dead-letter path.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


def fetch_node(state: State) -> dict:
    """Primary node — may fail after all retries are exhausted."""
    resp = httpx.get("https://api.example.com/data")
    resp.raise_for_status()
    return {"result": resp.json()["value"]}


def fetch_error_handler(state: State, exception: Exception) -> dict:
    """Fallback invoked if fetch_node raises after all retries.

    The handler MUST NOT raise — its exceptions fail the entire run.
    """
    print(f"fetch failed: {exception}")
    return {"result": "[unavailable]", "error": str(exception)}


builder = StateGraph(State)
builder.add_node(
    "fetch",
    fetch_node,
    retry_policy=RetryPolicy(max_attempts=3),
    error_handler=fetch_error_handler,   # runs only if all retries are exhausted
)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", END)

graph = builder.compile()
# Even when the API is down, graph.invoke returns {result: "[unavailable]", error: "..."}
result = graph.invoke({"query": "test"})
```

---

## 4. Graph-wide error handler with `set_node_defaults()`

Apply the same error handler to every node without repeating it on every `add_node`:

```python
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END


def global_error_handler(state: State, exception: Exception) -> dict:
    """Global fallback for any node that raises after retries."""
    import logging
    logging.error(
        "Node failed",
        extra={"error": str(exception), "state_keys": list(state.keys())},
    )
    return {"result": "[error]", "error": str(exception)}


builder = StateGraph(State)
builder.set_node_defaults(
    retry_policy=RetryPolicy(max_attempts=3),
    error_handler=global_error_handler,
)

# Both nodes inherit the retry + error_handler defaults
builder.add_node("fetch", fetch_node)
builder.add_node("process", process_node)

# Override the retry policy for a specific node; inherits error_handler
builder.add_node("critical", critical_node, retry_policy=RetryPolicy(max_attempts=5))

builder.add_edge(START, "fetch")
builder.add_edge("fetch", "process")
builder.add_edge("process", "critical")
builder.add_edge("critical", END)

graph = builder.compile()
```

---

## 5. Dead-letter routing with conditional edges

For more complex degradation — e.g. route to a fallback node instead of terminating — use a conditional edge that inspects the error field in state.

```python
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    query: str
    result: str
    error: str       # populated by the error handler


def route_after_fetch(state: State) -> str:
    if state.get("error"):
        return "dead_letter"    # error path
    return "process"            # happy path


def dead_letter_node(state: State) -> dict:
    """Log, alert, emit metric, and return a safe default."""
    send_alert(f"Dead-lettered request: {state['query']}, error: {state['error']}")
    return {"result": "[fallback response]"}


builder = StateGraph(State)
builder.add_node("fetch", fetch_node, error_handler=fetch_error_handler)
builder.add_node("process", process_node)
builder.add_node("dead_letter", dead_letter_node)

builder.add_edge(START, "fetch")
builder.add_conditional_edges("fetch", route_after_fetch)   # inspect error field
builder.add_edge("process", END)
builder.add_edge("dead_letter", END)

graph = builder.compile()
```

---

## 6. Checkpoint-based resumption

When a run fails mid-graph (e.g. the process crashes), you can resume from the **last successful checkpoint** by invoking with `None` as input and the same `thread_id`.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "resume-demo"}}

# First run — fails mid-graph at "process" node
try:
    graph.invoke({"query": "test"}, config)
except Exception as e:
    print(f"Run failed: {e}")

# Inspect the last saved state
snapshot = graph.get_state(config)
print(f"Last completed node(s): {snapshot.next}")  # which node is next to run
print(f"State so far: {snapshot.values}")

# Resume — re-invokes from the last checkpoint without re-running earlier nodes
result = graph.invoke(None, config)   # None = "continue from where we stopped"
print(result)
```

You can also **modify state before resuming** to fix bad data:

```python
# Fix the state that caused the failure
graph.update_state(
    config,
    {"query": "fixed_query"},   # overwrite the problematic field
)

# Now resume with corrected state
result = graph.invoke(None, config)
```

---

## 7. Complete resilient pipeline — all techniques combined

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy, TimeoutPolicy
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
import httpx
import time


class PipelineState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    result: str
    error: str


def should_retry(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, ConnectionError))


async def fetch_data(state: PipelineState, runtime: Runtime) -> dict:
    info = runtime.execution_info
    if info.node_attempt > 1:
        elapsed = time.time() - (info.node_first_attempt_time or time.time())
        runtime.stream_writer({"retry": info.node_attempt, "elapsed": elapsed})

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.example.com/data",
            params={"q": state["query"]},
        )
        resp.raise_for_status()
        return {"result": resp.json()["value"]}


def fetch_error_handler(state: PipelineState, exception: Exception) -> dict:
    return {"result": "[fetch unavailable]", "error": str(exception)}


async def process_data(state: PipelineState, runtime: Runtime) -> dict:
    chunks: list[str] = []
    async for chunk in analyze_stream(state["result"]):
        chunks.append(chunk)
        runtime.heartbeat()   # reset idle timer
    return {"result": "".join(chunks)}


def process_error_handler(state: PipelineState, exception: Exception) -> dict:
    return {"result": state.get("result", ""), "error": str(exception)}


def route(state: PipelineState) -> str:
    return "dead_letter" if state.get("error") else "process"


def dead_letter(state: PipelineState) -> dict:
    send_alert(f"Pipeline dead-letter: {state.get('error')}")
    return {}


builder = StateGraph(PipelineState)

builder.add_node(
    "fetch",
    fetch_data,
    retry_policy=RetryPolicy(
        initial_interval=1.0,
        max_attempts=4,
        retry_on=should_retry,
    ),
    timeout=TimeoutPolicy(run_timeout=20.0),
    error_handler=fetch_error_handler,
)

builder.add_node(
    "process",
    process_data,
    retry_policy=RetryPolicy(max_attempts=2),
    timeout=TimeoutPolicy(idle_timeout=30.0, refresh_on="heartbeat"),
    error_handler=process_error_handler,
)

builder.add_node("dead_letter", dead_letter)

builder.add_edge(START, "fetch")
builder.add_conditional_edges("fetch", route)
builder.add_edge("process", END)
builder.add_edge("dead_letter", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

---

## 6. Structured error types — catching and inspecting failures

LangGraph 1.2.11 defines a hierarchy of typed exceptions in `langgraph.errors`. Knowing which type is raised lets you write targeted `except` blocks in error handlers, retry predicates, and observability callbacks.

### `NodeError` — error handler context (source-verified)

`NodeError` is a frozen dataclass injected into any function registered as an `error_handler=`. It gives the handler structured access to what failed and where:

```python
from langgraph.errors import NodeError  # langgraph.errors, 1.2.11

# NodeError is a frozen dataclass with two fields:
#   node: str            — name of the failing node
#   error: BaseException — the raw exception that was raised
```

Use it to write generic handlers that log structured context:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
import logging

logger = logging.getLogger(__name__)


class State(TypedDict):
    data: str
    status: str


def flaky_node(state: State) -> dict:
    raise ValueError("upstream service unavailable")


def structured_error_handler(state: State, error: NodeError) -> dict:
    """NodeError.node and NodeError.error give full context without re-parsing."""
    logger.error(
        "node_failure",
        extra={"node": error.node, "exc_type": type(error.error).__name__},
    )
    return {"status": f"error in '{error.node}': {error.error}"}


graph = (
    StateGraph(State)
    .add_node("flaky", flaky_node, error_handler=structured_error_handler)
    .add_edge(START, "flaky")
    .add_edge("flaky", END)
    .compile()
)

result = graph.invoke({"data": "input", "status": ""})
print(result["status"])  # error in 'flaky': upstream service unavailable
```

### `NodeTimeoutError` — which timeout fired and how long it ran (source-verified)

When `TimeoutPolicy` cancels a node, LangGraph raises `NodeTimeoutError` (not the built-in `TimeoutError` — this is intentional so that `RetryPolicy`'s default `retry_on` treats it as retryable). Key attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `node` | `str` | Name of the timed-out node |
| `kind` | `"idle" \| "run"` | Whether `idle_timeout` or `run_timeout` fired |
| `timeout` | `float` | The threshold that was exceeded (seconds) |
| `elapsed` | `float` | How long the node actually ran (seconds) |
| `run_timeout` | `float \| None` | Configured `run_timeout` at the time of failure |
| `idle_timeout` | `float \| None` | Configured `idle_timeout` at the time of failure |

```python
import asyncio
from langgraph.errors import NodeTimeoutError
from langgraph.types import RetryPolicy, TimeoutPolicy

RETRYABLE_KINDS = {"run"}   # retry on run_timeout but not idle_timeout


def retry_on_run_timeout(exc: BaseException) -> bool:
    if isinstance(exc, NodeTimeoutError):
        return exc.kind in RETRYABLE_KINDS
    # fall back to the default behaviour for non-timeout errors
    from langgraph.types import default_retry_on
    return default_retry_on(exc)


async def slow_search(state: State) -> dict:
    await asyncio.sleep(60)   # simulate a slow operation
    return {"result": "done"}


graph = (
    StateGraph(State)
    .add_node(
        "search",
        slow_search,
        timeout=TimeoutPolicy(run_timeout=5.0, idle_timeout=2.0),
        retry_policy=RetryPolicy(
            max_attempts=2,
            retry_on=retry_on_run_timeout,  # only retry hard run_timeout, not idle
        ),
    )
    .add_edge(START, "search")
    .add_edge("search", END)
    .compile()
)
```

> `NodeTimeoutError` does **not** inherit from `asyncio.TimeoutError` or `TimeoutError`. If you have an `except TimeoutError` guard in your handler, it will not catch this exception — use `except NodeTimeoutError` explicitly.

### `NodeCancelledError` — asyncio.CancelledError from user code (source-verified)

`asyncio.CancelledError` is a `BaseException`, which the Pregel runner interprets as a framework-initiated cancellation signal (tearing down sibling tasks after a peer fails). When a node's own body raises it, LangGraph converts it into `NodeCancelledError` so it surfaces as a proper node failure rather than silently succeeding:

```python
# langgraph.errors — source-verified, 1.2.11
class NodeCancelledError(Exception):
    node: str   # the node whose body raised asyncio.CancelledError
```

Catch it in a `retry_on` predicate to decide whether cancellation should trigger a retry:

```python
import asyncio
from langgraph.errors import NodeCancelledError
from langgraph.types import RetryPolicy


def retry_policy_for_cancelled(exc: BaseException) -> bool:
    """Retry once if the node cancelled itself, but not on other errors."""
    return isinstance(exc, NodeCancelledError)


async def might_cancel(state: State) -> dict:
    # Simulate a node that may cancel itself due to an external signal
    if state.get("data") == "cancel":
        raise asyncio.CancelledError("external interrupt")
    return {"result": "ok"}


graph = (
    StateGraph(State)
    .add_node(
        "work",
        might_cancel,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=retry_policy_for_cancelled),
    )
    .add_edge(START, "work")
    .add_edge("work", END)
    .compile()
)
```

### `GraphRecursionError` — handling the recursion limit (source-verified)

`GraphRecursionError` is raised when the graph has exhausted the maximum number of steps (`recursion_limit`). In LangGraph 1.2.11 the default is **10007** (set via `DEFAULT_RECURSION_LIMIT` in `langgraph._internal._config`; override the baseline with the `LANGGRAPH_DEFAULT_RECURSION_LIMIT` environment variable). It inherits from the built-in `RecursionError`:

```python
class GraphRecursionError(RecursionError):
    pass   # carries an ErrorCode.GRAPH_RECURSION_LIMIT message
```

Two ways to handle it:

**Option 1 — raise the limit for long workflows:**

```python
result = graph.invoke(
    {"data": "input", "status": ""},
    {"recursion_limit": 100},   # second positional arg is the config
)
```

**Option 2 — catch it at the call site and resume from checkpoint:**

```python
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
cfg = {"configurable": {"thread_id": "long-run-1"}, "recursion_limit": 25}

try:
    result = graph.invoke({"data": "start", "status": ""}, cfg)
except GraphRecursionError:
    # The checkpoint was saved up to the last successful step.
    # Resume by passing None as input — LangGraph reloads from the checkpoint.
    result = graph.invoke(None, {**cfg, "recursion_limit": 50})
```

### `ErrorCode` — structured error taxonomy (source-verified)

`ErrorCode` is an `Enum` embedded in many LangGraph error messages. Match on it for programmatic error categorisation:

```python
from langgraph.errors import ErrorCode

# ErrorCode members (source-verified, 1.2.11):
# ErrorCode.GRAPH_RECURSION_LIMIT          — recursion_limit exceeded
# ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE — two threads wrote conflicting state
# ErrorCode.INVALID_GRAPH_NODE_RETURN_VALUE — node returned a non-dict / bad type
# ErrorCode.MULTIPLE_SUBGRAPHS             — ambiguous subgraph routing
# ErrorCode.INVALID_CHAT_HISTORY          — malformed message sequence
```

The code string is embedded in the exception message so you can detect it without importing `ErrorCode`:

```python
try:
    graph.invoke({"data": "x", "status": ""}, {"recursion_limit": 5})
except RecursionError as exc:
    if "GRAPH_RECURSION_LIMIT" in str(exc):
        print("Graph hit recursion limit — increase recursion_limit in config")
    raise
```

### Error type hierarchy summary

```
BaseException
 └── Exception
      ├── NodeTimeoutError       # run_timeout or idle_timeout exceeded
      ├── NodeCancelledError     # node body raised asyncio.CancelledError
      ├── EmptyInputError        # graph.invoke() received an empty input
      ├── RecursionError
      │    └── GraphRecursionError   # recursion_limit steps exhausted
      └── LookupError
           └── TaskNotFound     # resume targeted a non-existent interrupt id
```

---

## Comparison: native policies vs external libraries

| Scenario | LangGraph native | Why prefer native |
|---|---|---|
| Retry on HTTP errors | `RetryPolicy(retry_on=...)` | Integrated with the graph execution loop; works with checkpointers |
| Custom backoff | `RetryPolicy(initial_interval, backoff_factor, jitter)` | No extra dependency; same `NamedTuple` as `TimeoutPolicy` |
| Timeout | `TimeoutPolicy(run_timeout=..., idle_timeout=...)` | Cooperative cancellation via asyncio; heartbeat support |
| Global fallback | `set_node_defaults(error_handler=...)` | One line; no decorator boilerplate per node |
| Resume after crash | `graph.invoke(None, config)` | Requires checkpointer; saves/restores state automatically |

---

## See also

- [`reference-state-graph.md`](/langgraph-guide/python/reference-state-graph/) — `add_node`, `set_node_defaults` parameter reference
- [`langgraph_performance_optimization.md`](/langgraph-guide/python/langgraph_performance_optimization/) — `CachePolicy`, `RetryPolicy`, `TimeoutPolicy` in the performance context
- [`reference-runtime-and-managed-values.md`](/langgraph-guide/python/reference-runtime-and-managed-values/) — `Runtime.heartbeat()`, `ExecutionInfo.node_attempt`
- [`chapter-07-human-in-the-loop.md`](/langgraph-guide/python/chapter-07-human-in-the-loop/) — `interrupt()` and `Command(resume=...)` for human-gated recovery

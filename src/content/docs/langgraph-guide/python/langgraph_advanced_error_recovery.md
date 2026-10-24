---
title: "LangGraph Advanced Error Handling and Recovery (Python)"
description: "Native RetryPolicy, TimeoutPolicy, node-level error_handler, dead-letter patterns, checkpoint-based resumption, and structured error types (NodeError, NodeTimeoutError, NodeCancelledError, GraphRecursionError, ErrorCode) — source-verified."
framework: "langgraph"
language: "python"
sidebar:
  label: "Error handling & recovery"
  order: 52
---

Verified against **`langgraph==%%version=langgraph%%`** (modules: `langgraph.types`, `langgraph.graph.state`, `langgraph.runtime`, `langgraph.errors`).

LangGraph handles errors in layers. This page covers what happens after a node fails: error handlers, dead-letter routing, resuming from a checkpoint, and the exception types. Retry, timeout and cache options are documented once, in [Node policies](/langgraph-guide/python/reference-state-graph/#node-policies).

| Layer | Primitive | Scope |
|---|---|---|
| Automatic retry | `RetryPolicy` on `add_node` | Re-runs a node on transient exceptions |
| Timeout | `timeout=` / `TimeoutPolicy` on `add_node` (async nodes) | Cancels a hung attempt with `NodeTimeoutError` |
| Node error handler | `error_handler=` on `add_node` | Fallback once retries are exhausted |
| Graph-wide handler | `set_node_defaults(error_handler=...)` | One fallback for every node |
| Dead-letter routing | Handler returns `Command(goto="dead_letter")` | Continue on a recovery path instead of failing |
| Checkpoint resumption | `graph.invoke(None, config)` | Re-run from the failed step after a fix |
| Agent-level | `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `ToolRetryMiddleware`, `ToolErrorMiddleware` | `create_agent` agents ([Chapter 8](/langgraph-guide/python/chapter-08-middleware-hooks/#built-in-middleware)) |

## 1. Retries, timeouts and idempotency

Configure retries and timeouts as described in [Node policies](/langgraph-guide/python/reference-state-graph/#node-policies). Two points that matter for recovery:

- The default `retry_on` retries most exceptions but not `ValueError`, `TypeError`, other programming errors, or `OSError` (including the built-in `TimeoutError`). It does retry LangGraph's own `NodeTimeoutError`.
- A retried node runs again from the start, so side effects must be idempotent.

### Using `ExecutionInfo` for idempotent retries

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

## 2. Error handlers and dead-letter routing

An error handler runs when a node has failed and its retries are exhausted. Declare the failure parameter with the `NodeError` type (from `langgraph.errors`); LangGraph injects it by that annotation, so `error: Exception` does not work. `NodeError` has `.node` (the failed node's name) and `.error` (the exception).

The handler's return value decides what happens next:

- **A `Command(update=..., goto="node")`** applies the update and continues at that node. Use this for dead-letter routing.
- **A plain dict** applies the update and ends that branch: the failed node's outgoing edges are not followed.
- **Raising** fails the run.

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy


class State(TypedDict, total=False):
    query: str
    result: str
    error: str


async def fetch(state: State) -> dict:
    raise ConnectionError("api.example.com unreachable")     # fails on every attempt


async def process(state: State) -> dict:
    return {"result": state["result"].upper()}


async def dead_letter(state: State) -> dict:
    print(f"ALERT: {state['query']!r} dead-lettered: {state['error']}")
    return {"result": "[fallback response]"}


async def to_dead_letter(state: State, error: NodeError) -> Command:
    # Runs once retries are exhausted. A plain dict would end the branch here;
    # a Command chooses where the run continues.
    return Command(update={"error": f"{error.node}: {error.error!r}"}, goto="dead_letter")


graph = (
    StateGraph(State)
    .set_node_defaults(error_handler=to_dead_letter)          # every node
    .add_node("fetch", fetch, retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.01))
    .add_node("process", process)
    .add_node("dead_letter", dead_letter)
    .add_edge(START, "fetch")
    .add_edge("fetch", "process")
    .add_edge("process", END)
    .add_edge("dead_letter", END)
    .compile()
)

print(asyncio.run(graph.ainvoke({"query": "weather"})))
# ALERT: 'weather' dead-lettered: fetch: ConnectionError('api.example.com unreachable')
# {'query': 'weather', 'result': '[fallback response]', 'error': "fetch: ConnectionError('api.example.com unreachable')"}
```

Handlers do not catch failures from tasks that run in parallel with others in the same super-step (parallel branches or `Send` fan-out): the exception propagates and fails the run. In fan-out workers, catch exceptions inside the worker and write them to a state key with a list reducer (for example `errors: Annotated[list, operator.add]`).

If `set_node_defaults(timeout=...)` is also set, the handler gets the timeout too and must be `async def`.

## 3. Resuming from a checkpoint

With a checkpointer, a failed run leaves its last checkpoint behind, and the failed node is still pending. Fix the cause (or the state) and call `invoke(None, config)` to continue from that step; nodes that already finished do not run again.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict, total=False):
    query: str
    fetched: str
    result: str


calls = {"fetch": 0}


def fetch(state: State) -> dict:
    calls["fetch"] += 1
    return {"fetched": f"data for {state['query']}"}


def process(state: State) -> dict:
    if state["query"] == "bad":
        raise ValueError("cannot process 'bad'")
    return {"result": state["fetched"].upper()}


graph = (
    StateGraph(State)
    .add_node("fetch", fetch)
    .add_node("process", process)
    .add_edge(START, "fetch")
    .add_edge("fetch", "process")
    .add_edge("process", END)
    .compile(checkpointer=InMemorySaver())
)
config = {"configurable": {"thread_id": "resume-demo"}}

try:
    graph.invoke({"query": "bad"}, config)
except ValueError as exc:
    print("run failed:", exc)

snapshot = graph.get_state(config)
print(snapshot.next)              # ('process',): the node that failed, still pending
print(snapshot.values)            # {'query': 'bad', 'fetched': 'data for bad'}

# Fix the bad input, then resume with None: fetch does not run again.
graph.update_state(config, {"query": "good"})
print(graph.invoke(None, config))  # {'query': 'good', 'fetched': 'data for bad', 'result': 'DATA FOR BAD'}
print("fetch ran", calls["fetch"], "time(s)")   # 1
```

The same applies after a process crash: on restart, `invoke(None, config)` with the thread's config resumes from the last checkpoint written. How much work is redone depends on the [durability mode](/langgraph-guide/python/chapter-10-production/#durability-how-often-state-is-written).

## 4. Structured error types — catching and inspecting failures

LangGraph defines a hierarchy of typed exceptions in `langgraph.errors`. Knowing which type is raised lets you write targeted `except` blocks in error handlers, retry predicates, and observability callbacks.

### `NodeError` — error handler context (source-verified)

`NodeError` is a frozen dataclass injected into any function registered as an `error_handler=`. It gives the handler structured access to what failed and where:

```python
from langgraph.errors import NodeError  # langgraph.errors, %%version=langgraph%%

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
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeTimeoutError
from langgraph.types import RetryPolicy, TimeoutPolicy


class State(TypedDict):
    data: str
    result: str


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
# langgraph.errors — source-verified, %%version=langgraph%%
class NodeCancelledError(Exception):
    node: str   # the node whose body raised asyncio.CancelledError
```

Catch it in a `retry_on` predicate to decide whether cancellation should trigger a retry:

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeCancelledError
from langgraph.types import RetryPolicy


class State(TypedDict):
    data: str
    result: str


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

`GraphRecursionError` is raised when the graph has exhausted the maximum number of steps (`recursion_limit`). In LangGraph %%version=langgraph%% the default is **10007** (set via `DEFAULT_RECURSION_LIMIT` in `langgraph._internal._config`; override the baseline with the `LANGGRAPH_DEFAULT_RECURSION_LIMIT` environment variable). It inherits from the built-in `RecursionError`:

```python
class GraphRecursionError(RecursionError):
    pass   # carries an ErrorCode.GRAPH_RECURSION_LIMIT message
```

Two ways to handle it. Both use this loop, which needs 30 steps to finish:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class LoopState(TypedDict):
    count: int


def step(state: LoopState) -> dict:
    return {"count": state["count"] + 1}


loop_builder = (
    StateGraph(LoopState)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_conditional_edges("step", lambda s: END if s["count"] >= 30 else "step")
)
```

**Option 1 — raise the limit for long workflows:**

```python
loop = loop_builder.compile()
result = loop.invoke({"count": 0}, {"recursion_limit": 100})   # the config is the second argument
print(result)   # {'count': 30}
```

**Option 2 — catch it at the call site and resume from the checkpoint:**

```python
from langgraph.errors import GraphRecursionError

loop = loop_builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "long-run-1"}, "recursion_limit": 25}

try:
    result = loop.invoke({"count": 0}, cfg)
except GraphRecursionError:
    # The checkpoint was saved up to the last successful step.
    # Resume by passing None as input: the new limit applies to this invoke only.
    result = loop.invoke(None, {**cfg, "recursion_limit": 50})
print(result)   # {'count': 30}
```

### `ErrorCode` — structured error taxonomy (source-verified)

`ErrorCode` is an `Enum` whose **string value is embedded in LangGraph exception messages and troubleshooting URLs**. It is NOT exposed as an attribute on exception instances — code like `exc.error_code == ErrorCode.GRAPH_RECURSION_LIMIT` will not work. Prefer catching concrete exception types (`NodeTimeoutError`, `GraphRecursionError`, etc.); use message-string inspection only as a fallback.

```python
from langgraph.errors import ErrorCode

# ErrorCode members (source-verified, %%version=langgraph%%):
# ErrorCode.GRAPH_RECURSION_LIMIT          — recursion_limit exceeded
# ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE — parallel nodes wrote to a channel accepting
#                                             only one value per super-step (unannotated key
#                                             or second Overwrite in the same super-step)
# ErrorCode.INVALID_GRAPH_NODE_RETURN_VALUE — node returned a non-dict / bad type
# ErrorCode.MULTIPLE_SUBGRAPHS             — ambiguous subgraph routing
# ErrorCode.INVALID_CHAT_HISTORY          — malformed message sequence
```

The code string is embedded in the exception message — inspect it via `str(exc)`, not via an attribute:

```python
try:
    loop_builder.compile().invoke({"count": 0}, {"recursion_limit": 5})
except RecursionError as exc:
    if "GRAPH_RECURSION_LIMIT" in str(exc):
        print("Graph hit recursion limit — increase recursion_limit in config")
```

### Error type hierarchy summary

```text
BaseException
 └── Exception
      ├── NodeTimeoutError       # run_timeout or idle_timeout exceeded
      ├── NodeCancelledError     # node body raised asyncio.CancelledError
      ├── InvalidUpdateError     # e.g. two writes to a key without a reducer
      ├── EmptyInputError        # the graph received an empty input
      ├── TaskNotFound           # executor cannot find a task (distributed mode)
      ├── RecursionError
      │    └── GraphRecursionError   # recursion_limit steps exhausted
      └── GraphBubbleUp          # control flow, not failures: don't catch these broadly
           ├── GraphInterrupt    # interrupt() paused the run
           │    └── NodeInterrupt (deprecated)
           ├── ParentCommand     # Command(graph=Command.PARENT) travelling upwards
           └── GraphDrained      # RunControl.request_drain() stopped the run
```

`NodeError` (the error-handler argument) is a dataclass, not an exception. All of these live in `langgraph.errors`. A broad `except Exception:` inside a node also catches `GraphInterrupt`, which breaks `interrupt()`; re-raise `GraphBubbleUp` if you must catch everything.

---

## Comparison: native policies vs external libraries

| Scenario | LangGraph native | Why prefer native |
|---|---|---|
| Retry on HTTP errors | `RetryPolicy(retry_on=...)` | Integrated with the graph execution loop; works with checkpointers |
| Custom backoff | `RetryPolicy(initial_interval, backoff_factor, max_interval, jitter)` | No extra dependency; applies per attempt inside the graph loop |
| Timeout | `TimeoutPolicy(run_timeout=..., idle_timeout=...)` | Cooperative cancellation via asyncio; heartbeat support |
| Global fallback | `set_node_defaults(error_handler=...)` | One line; no decorator boilerplate per node |
| Resume after crash | `graph.invoke(None, config)` | Requires checkpointer; saves/restores state automatically |

---

## See also

- [`reference-state-graph.md`](/langgraph-guide/python/reference-state-graph/) — `add_node`, `set_node_defaults` parameter reference
- [`langgraph_performance_optimization.md`](/langgraph-guide/python/langgraph_performance_optimization/) — `CachePolicy`, `RetryPolicy`, `TimeoutPolicy` in the performance context
- [`reference-runtime-and-managed-values.md`](/langgraph-guide/python/reference-runtime-and-managed-values/) — `Runtime.heartbeat()`, `ExecutionInfo.node_attempt`
- [`chapter-07-human-in-the-loop.md`](/langgraph-guide/python/chapter-07-human-in-the-loop/) — `interrupt()` and `Command(resume=...)` for human-gated recovery

---
title: "LangGraph Performance Optimization (Python)"
description: "Native CachePolicy, RetryPolicy, TimeoutPolicy, async batching, and set_node_defaults — source-verified patterns for LangGraph 1.2.2."
framework: langgraph
language: python
---

# LangGraph Performance Optimization (Python)

Verified against **`langgraph==1.2.2`** (modules: `langgraph.types`, `langgraph.graph.state`, `langgraph.cache.memory`).

This guide covers the five levers you have for performance in LangGraph:

| Lever | API | When to reach for it |
|---|---|---|
| **Caching** | `CachePolicy` on `add_node` / `@task` | Idempotent nodes whose output depends only on input |
| **Retry with backoff** | `RetryPolicy` on `add_node` / `@task` | Transient I/O errors (HTTP 5xx, rate limits) |
| **Timeouts** | `TimeoutPolicy` on `add_node` / `@task` | Long-running async nodes that may hang |
| **Concurrency** | `graph.batch()`, `Send`, async `astream` | Independent tasks that can run in parallel |
| **Graph-wide defaults** | `set_node_defaults()` | Applying the same policy to every node without repeating it |

---

## 1. `CachePolicy` — built-in node caching

`CachePolicy` caches the output of a node by a hash of its input. When the same input arrives again within the TTL, LangGraph returns the cached result instead of re-running the node.

```python
# langgraph.types (source-verified, langgraph 1.2.2)
@dataclass
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key   # default: pickle-hash of full input
    ttl: int | None = None                   # seconds; None = never expires
```

### 1.1 Basic setup — `InMemoryCache`

You need a cache backend at compile time. For local development, use `InMemoryCache`. Production deployments can use a Redis-backed cache.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langchain_anthropic import ChatAnthropic


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    result: str


model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


def classify(state: State) -> dict:
    """Classify the query — expensive but pure/idempotent."""
    response = model.invoke(
        [HumanMessage(f"Classify this query in one word: {state['query']}")]
    )
    return {"result": response.content}


# Build graph with per-node CachePolicy
builder = StateGraph(State)
builder.add_node(
    "classify",
    classify,
    cache_policy=CachePolicy(ttl=300),   # 5-minute TTL per unique input
)
builder.add_edge(START, "classify")
builder.add_edge("classify", END)

# The cache= argument at compile time activates caching
cache = InMemoryCache()
graph = builder.compile(cache=cache)

config = {"configurable": {"thread_id": "perf-1"}}
# First call — hits the model
r1 = graph.invoke({"query": "What is photosynthesis?"}, config)
# Second call with identical input — served from cache (no model call)
r2 = graph.invoke({"query": "What is photosynthesis?"}, config)
```

### 1.2 Custom cache key function

By default, LangGraph hashes the entire input dict via pickle. Supply your own `key_func` to hash only the fields that actually matter — this increases cache hit rates when irrelevant fields change (e.g. a timestamp or session counter that varies but should not bust the cache).

```python
from langgraph.types import CachePolicy


def query_only_key(state: dict) -> str:
    """Cache by query text only — ignore unrelated state fields."""
    return state.get("query", "")


def summarize(state: State) -> dict:
    response = model.invoke([HumanMessage(f"Summarize: {state['query']}")])
    return {"result": response.content}


builder.add_node(
    "summarize",
    summarize,
    cache_policy=CachePolicy(
        key_func=query_only_key,
        ttl=3600,          # 1-hour TTL
    ),
)
```

### 1.3 Clearing the cache

Call `graph.aclear_cache(cache)` (async) or `graph.clear_cache(cache)` (sync) to wipe all cached entries — useful after model updates or data refreshes.

```python
import asyncio

async def refresh_and_run(query: str):
    await graph.aclear_cache(cache)
    result = await graph.ainvoke({"query": query}, config)
    return result
```

---

## 2. `RetryPolicy` — native retry with exponential backoff

`RetryPolicy` is a `NamedTuple` that configures automatic retries for a node when it raises an exception. No external library (e.g. `tenacity`) needed — LangGraph handles the backoff loop.

```python
# langgraph.types (source-verified, langgraph 1.2.2)
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5      # seconds before the first retry
    backoff_factor:   float = 2.0      # multiplier applied after each retry
    max_interval:     float = 128.0    # maximum inter-retry wait (seconds)
    max_attempts:     int   = 3        # total attempts including the first
    jitter:           bool  = True     # add random jitter to intervals
    retry_on = default_retry_on        # default: httpx 5xx, transport errors, timeouts
```

### 2.1 Retry on transient HTTP errors

```python
from langgraph.types import RetryPolicy
import httpx


def call_external_api(state: State) -> dict:
    """May fail transiently with 5xx or network errors."""
    resp = httpx.get(f"https://api.example.com/data?q={state['query']}")
    resp.raise_for_status()
    return {"result": resp.json()["answer"]}


builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=30.0,
        max_attempts=4,
        jitter=True,
    ),
)
```

### 2.2 Custom `retry_on` — callable predicate

For fine-grained control, pass a callable that returns `True` when a retry should happen.

```python
from langgraph.types import RetryPolicy
import httpx


def should_retry(exc: Exception) -> bool:
    """Retry on 429 (rate limit) and server errors, NOT on 4xx client errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True
    return False


def call_rate_limited_api(state: State) -> dict:
    resp = httpx.post("https://api.example.com/search", json={"q": state["query"]})
    resp.raise_for_status()
    return {"result": resp.json()}


builder.add_node(
    "search",
    call_rate_limited_api,
    retry_policy=RetryPolicy(
        initial_interval=2.0,
        max_attempts=5,
        retry_on=should_retry,
    ),
)
```

### 2.3 Ordered fallback policies (retry sequence)

Pass a **list** of `RetryPolicy` objects. LangGraph applies the **first policy whose `retry_on` matches** the raised exception.

```python
from langgraph.types import RetryPolicy
import httpx


# Quick retry for rate limits
rate_limit_policy = RetryPolicy(
    initial_interval=1.0,
    max_attempts=3,
    retry_on=lambda e: isinstance(e, httpx.HTTPStatusError)
        and e.response.status_code == 429,
)

# Slower retry for server errors
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
    retry_policy=[rate_limit_policy, server_error_policy],  # ordered fallbacks
)
```

### 2.4 `RetryPolicy` on `@task` (Functional API)

The same `RetryPolicy` type works on `@task` decorators in the Functional API.

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver
import httpx


@task(retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
def fetch_weather(city: str) -> dict:
    """Fetch weather data — retried up to 3 times on transient errors."""
    resp = httpx.get(f"https://api.weather.example.com/{city}")
    resp.raise_for_status()
    return resp.json()


@entrypoint(checkpointer=InMemorySaver())
def weather_pipeline(cities: list[str]) -> list[dict]:
    futures = [fetch_weather(c) for c in cities]  # parallel fan-out
    return [f.result() for f in futures]           # collect results
```

---

## 3. `TimeoutPolicy` — per-attempt cancellation

`TimeoutPolicy` caps how long an async node attempt can run. Two modes:

- **`run_timeout`** — hard wall-clock cap. Never refreshed by any signal.
- **`idle_timeout`** — cap on time without observable progress. Refreshed by LangGraph events or explicit `runtime.heartbeat()` calls.

> **Sync nodes cannot be timed out in-process.** Passing `timeout=` to a sync node raises `ValueError` at compile time. Wrap blocking code in `asyncio.to_thread()` to make it cancellable.

```python
# langgraph.types (source-verified, langgraph 1.2.2)
@dataclass
class TimeoutPolicy:
    run_timeout:  float | timedelta | None = None   # hard wall-clock cap per attempt
    idle_timeout: float | timedelta | None = None   # max time between progress signals
    refresh_on:   Literal["auto", "heartbeat"] = "auto"
    # "auto"      = refresh on graph progress signals AND heartbeat()
    # "heartbeat" = refresh ONLY on explicit runtime.heartbeat() calls
```

### 3.1 Hard wall-clock timeout

```python
from langgraph.types import TimeoutPolicy


async def slow_embedding(state: State) -> dict:
    import asyncio
    await asyncio.sleep(0)   # yield to event loop so cancellation can fire
    # ... expensive embedding work ...
    return {"result": "embeddings computed"}


builder.add_node(
    "embed",
    slow_embedding,
    timeout=TimeoutPolicy(run_timeout=10.0),   # fail after 10 s; no retry
)
```

### 3.2 Idle timeout with heartbeat

Use `idle_timeout` + `runtime.heartbeat()` when your node makes incremental progress (e.g. a streaming API). This detects stalls: if no chunk arrives within the window, the node is cancelled.

```python
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.runtime import Runtime


async def streaming_node(state: State, runtime: Runtime) -> dict:
    chunks: list[str] = []
    async for chunk in call_streaming_api(state["query"]):
        chunks.append(chunk)
        runtime.heartbeat()   # reset the idle timer — we're still making progress

    return {"result": "".join(chunks)}


builder.add_node(
    "stream_call",
    streaming_node,
    timeout=TimeoutPolicy(
        idle_timeout=30.0,       # fail if no chunk arrives for 30 s
        refresh_on="heartbeat",  # only runtime.heartbeat() resets the timer
    ),
    retry_policy=RetryPolicy(max_attempts=2),   # retry once on timeout
)
```

### 3.3 `TimeoutPolicy` on `@task`

```python
from langgraph.func import entrypoint, task
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver


@task(
    timeout=TimeoutPolicy(run_timeout=15.0),
    retry_policy=RetryPolicy(max_attempts=2),
)
async def run_analysis(data: dict) -> dict:
    """Must complete within 15 seconds; retried once on failure."""
    import asyncio
    await asyncio.sleep(0)   # allow cancellation
    # ... async analysis work ...
    return {"analysis": "done"}


@entrypoint(checkpointer=InMemorySaver())
async def pipeline(inputs: dict) -> dict:
    future = run_analysis(inputs)
    return future.result()
```

---

## 4. Graph-wide defaults — `set_node_defaults()`

Rather than repeating `retry_policy=`, `cache_policy=`, and `timeout=` on every `add_node` call, use `set_node_defaults()` to apply a baseline. Per-node values **always override** the defaults.

```python
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache


def handle_error_node(state: State, exception: Exception) -> dict:
    """Global fallback handler — called when any node raises unhandled."""
    return {"result": f"Error handled: {exception}"}


builder = StateGraph(State)

# Set graph-wide defaults — applies to every add_node that follows
builder.set_node_defaults(
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5),
    cache_policy=CachePolicy(ttl=600),         # 10-minute cache for all nodes
    error_handler=handle_error_node,            # global fallback on unhandled errors
    timeout=30.0,                               # 30 s hard cap on all async nodes
)

# All nodes below inherit the defaults above
builder.add_node("fetch", fetch_node)
builder.add_node("process", process_node)

# This node overrides retry_policy only; inherits the rest
builder.add_node(
    "critical",
    critical_node,
    retry_policy=RetryPolicy(max_attempts=5, initial_interval=2.0),
)

builder.add_edge(START, "fetch")
builder.add_edge("fetch", "process")
builder.add_edge("process", "critical")
builder.add_edge("critical", END)

graph = builder.compile(cache=InMemoryCache())
```

---

## 5. Concurrency — batch, async, and parallel branches

### 5.1 `graph.batch()` — multiple independent runs

Run the same graph on multiple inputs simultaneously. Each input gets its own thread ID.

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())

inputs = [
    {"query": "What is AI?"},
    {"query": "What is LangGraph?"},
    {"query": "What is Python?"},
]
configs = [
    {"configurable": {"thread_id": f"batch-{i}"}}
    for i in range(len(inputs))
]

# Sync batch
results = graph.batch(inputs, configs)

# Async batch — higher throughput for I/O-bound graphs
results = await graph.abatch(inputs, configs)
```

### 5.2 Parallel fan-out with `Send`

Use `Send` inside a conditional edge to run the same node concurrently with different inputs, then aggregate results.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
import operator


class MapState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]   # reducer: append all results


class ItemState(TypedDict):
    item: str


def fan_out(state: MapState) -> list[Send]:
    """Create one Send per item — all run in parallel."""
    return [Send("process_item", {"item": item}) for item in state["items"]]


def process_item(state: ItemState) -> dict:
    return {"results": [state["item"].upper()]}


def aggregate(state: MapState) -> dict:
    return {"results": state["results"]}


builder = StateGraph(MapState)
builder.add_node("process_item", process_item)
builder.add_node("aggregate", aggregate)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("process_item", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()
result = graph.invoke({"items": ["apple", "banana", "cherry"], "results": []})
print(result["results"])   # ['APPLE', 'BANANA', 'CHERRY'] (order may vary)
```

### 5.3 `add_sequence()` — concise linear pipelines

`add_sequence()` wires a list of nodes with edges in one call — no separate `add_node` + `add_edge` for each step.

```python
from langgraph.graph import StateGraph, START, END


def validate(state: State) -> dict: ...
def enrich(state: State) -> dict: ...
def persist(state: State) -> dict: ...


builder = StateGraph(State)

# Equivalent to add_node + add_edge for each step
builder.add_sequence([validate, enrich, persist])   # wires: validate → enrich → persist

builder.add_edge(START, "validate")   # wire START manually
builder.add_edge("persist", END)      # wire END manually

graph = builder.compile()
```

Use named tuples when function `__name__` is not descriptive enough:

```python
builder.add_sequence([
    ("validate", validate_input),
    ("enrich",   enrich_data),
    ("persist",  save_to_db),
])
```

### 5.4 Async streaming with `stream_mode="updates"`

For long-running graphs, stream partial results with `"updates"` mode — only what changed after each node is sent, minimizing data transfer.

```python
async def run_with_streaming(query: str) -> None:
    config = {"configurable": {"thread_id": "stream-perf"}}

    async for event in graph.astream(
        {"query": query},
        config,
        stream_mode="updates",   # only changed fields — much lighter than "values"
    ):
        for node_name, updates in event.items():
            print(f"{node_name} updated: {list(updates.keys())}")
```

---

## 6. Combining CachePolicy + RetryPolicy + TimeoutPolicy

A complete production-ready node that stacks all three policies — cache hits skip the API call entirely, retries handle transient failures, and the timeout prevents the node from hanging indefinitely.

```python
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
import httpx


def query_cache_key(state: dict) -> str:
    """Cache key: only the query text matters."""
    return state.get("query", "")


def should_retry(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, ConnectionError))


async def call_llm_api(state: State, runtime: Runtime) -> dict:
    """
    LLM API call with:
    - 5-minute result cache (skip the API when input repeats)
    - Up to 3 retries on rate-limit / server errors
    - 45-second hard cap per attempt
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.llm.example.com/generate",
            json={"prompt": state["query"]},
            timeout=40.0,
        )
        resp.raise_for_status()
        return {"result": resp.json()["text"]}


builder = StateGraph(State)
builder.add_node(
    "llm",
    call_llm_api,
    cache_policy=CachePolicy(key_func=query_cache_key, ttl=300),
    retry_policy=RetryPolicy(
        initial_interval=1.0,
        backoff_factor=2.0,
        max_attempts=3,
        retry_on=should_retry,
    ),
    timeout=TimeoutPolicy(run_timeout=45.0),
)
builder.add_edge(START, "llm")
builder.add_edge("llm", END)

cache = InMemoryCache()
graph = builder.compile(
    checkpointer=InMemorySaver(),
    cache=cache,
)
```

---

## Quick reference — policy constructors

```python
from langgraph.types import CachePolicy, RetryPolicy, TimeoutPolicy

# Cache: 10-minute TTL, default key (pickle hash of full input)
CachePolicy(ttl=600)

# Cache: 1-hour TTL, custom key function
CachePolicy(key_func=lambda s: s["query"], ttl=3600)

# Retry: 3 attempts, 1 s → 2 s → 4 s (with jitter, default)
RetryPolicy(initial_interval=1.0, backoff_factor=2.0, max_attempts=3)

# Retry: custom predicate, no jitter
RetryPolicy(max_attempts=5, jitter=False, retry_on=should_retry)

# Retry: ordered fallback sequence
[RetryPolicy(retry_on=rate_limit_check), RetryPolicy(retry_on=server_error_check)]

# Timeout: hard 30-second wall-clock cap
TimeoutPolicy(run_timeout=30.0)

# Timeout: idle cap (reset by heartbeat) + hard cap
TimeoutPolicy(run_timeout=60.0, idle_timeout=15.0, refresh_on="heartbeat")

# Graph-wide defaults (applied to every node)
builder.set_node_defaults(
    retry_policy=RetryPolicy(max_attempts=3),
    cache_policy=CachePolicy(ttl=600),
    timeout=30.0,
)
```

---

## See also

- [`reference-state-graph.md`](/langgraph-guide/python/reference-state-graph/) — `add_node`, `set_node_defaults`, `compile` signatures
- [`reference-functional-api.md`](/langgraph-guide/python/reference-functional-api/) — `@task` with `cache_policy` and `retry_policy`
- [`reference-runtime-and-managed-values.md`](/langgraph-guide/python/reference-runtime-and-managed-values/) — `Runtime.heartbeat()` for `idle_timeout`
- [`chapter-09-advanced-patterns.md`](/langgraph-guide/python/chapter-09-advanced-patterns/) — map-reduce with `Send`, `add_sequence`, `GraphOutput`
- [`langgraph_advanced_error_recovery.md`](/langgraph-guide/python/langgraph_advanced_error_recovery/) — `error_handler`, dead-letter patterns, checkpoint resumption

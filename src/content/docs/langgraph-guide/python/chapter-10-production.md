---
title: "Chapter 10 — Production & Troubleshooting"
description: "RetryPolicy, TimeoutPolicy, CachePolicy, Durability, IsLastStep/RemainingSteps, Topic channels, Send timeouts, Docker, CLI config, remote SDK, and fixes for common errors."
framework: langgraph
language: python
sidebar:
  label: "10 · Production & troubleshooting"
  order: 10
---

# Chapter 10 — Production & Troubleshooting

**What you'll learn:** the node-level reliability primitives (`RetryPolicy`, `TimeoutPolicy`, `CachePolicy`), checkpoint durability modes, loop-safeguard managed values (`IsLastStep`, `RemainingSteps`), the `Topic` channel for fan-in aggregation, per-Send timeouts, async execution, Docker deployment, CLI config, and troubleshooting the most common runtime errors.

Verified against **`langgraph==1.2.6`** (modules: `langgraph.types`, `langgraph.managed`, `langgraph.channels`).

**Time:** ~30 minutes.

> For the full deployment playbook (Kubernetes, cost optimization, disaster recovery, observability), continue to the [Production Guide](/langgraph-guide/python/langgraph_production_guide/) after this chapter.

---

## `RetryPolicy` — Automatic Node Retries

`RetryPolicy` is a `NamedTuple` that configures exponential-backoff retries for a node. Attach it when you add a node via `add_node(..., retry=...)`.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

class State(TypedDict):
    result: str
    attempts: int

def flaky_api_call(state: State) -> dict:
    """Simulates an API that sometimes fails."""
    import random
    if random.random() < 0.7:
        raise ConnectionError("Transient API failure")
    return {"result": "success", "attempts": state["attempts"] + 1}

# Default RetryPolicy: 3 attempts, 0.5s initial, 2× backoff, 128s cap, with jitter.
# Only retries on network/server errors by default (see retry_on below).
builder = StateGraph(State)
builder.add_node(
    "api_call",
    flaky_api_call,
    retry_policy=RetryPolicy(
        initial_interval=0.5,    # seconds before first retry
        backoff_factor=2.0,      # multiply interval by this after each attempt
        max_interval=30.0,       # cap interval at 30 seconds
        max_attempts=5,          # give up after 5 total attempts (not 5 retries)
        jitter=True,             # add random jitter to avoid thundering herd
    ),
)
builder.add_edge(START, "api_call")
builder.add_edge("api_call", END)
graph = builder.compile()
```

### Custom `retry_on` predicate

By default `RetryPolicy` retries on a built-in list of transient exceptions. Override `retry_on` with a callable, a single type, or a sequence of types:

```python
import httpx

def should_retry(exc: Exception) -> bool:
    """Retry on HTTP 429 (rate limit) or 5xx server errors only."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (ConnectionError, TimeoutError))

builder.add_node(
    "llm_call",
    my_llm_node,
    retry_policy=RetryPolicy(
        max_attempts=4,
        initial_interval=1.0,
        retry_on=should_retry,    # callable: receives the exception, returns bool
    ),
)

# Or pass exception types directly:
builder.add_node(
    "db_write",
    my_db_node,
    retry_policy=RetryPolicy(
        max_attempts=3,
        retry_on=(ConnectionError, OSError),   # tuple of types
    ),
)
```

---

## `TimeoutPolicy` — Hard and Idle Timeouts

`TimeoutPolicy` lets you cap how long a single node attempt can run. It has two independent timeout axes:

- **`run_timeout`** — hard wall-clock cap. Never refreshed by progress. The node is cancelled if it hasn't finished within this many seconds.
- **`idle_timeout`** — maximum time without observable progress. Refreshed automatically by LangGraph callbacks and by explicit `runtime.heartbeat()` calls.

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

# Hard 30-second cap — no matter what the node is doing
builder.add_node(
    "web_search",
    search_node,
    timeout=TimeoutPolicy(run_timeout=30.0),   # seconds as float
)

# Use timedelta for readability
builder.add_node(
    "long_analysis",
    analysis_node,
    timeout=TimeoutPolicy(
        run_timeout=timedelta(minutes=5),     # 5-minute wall-clock cap
        idle_timeout=timedelta(seconds=30),  # cancel if idle for 30 s
        refresh_on="auto",   # refresh idle_timeout on any graph callback (default)
    ),
)

# Shorthand: a plain float is treated as run_timeout
builder.add_node("quick_check", check_node, timeout=10.0)
```

`TimeoutPolicy` uses asyncio cancellation — it fires only when the event loop is released. CPU-bound blocking code (e.g. `time.sleep()`) will not be cancelled until it yields.

### Heartbeating for long-running nodes

When `refresh_on="heartbeat"`, the idle timeout resets only when you call `runtime.heartbeat()`:

```python
from langgraph.runtime import Runtime

async def long_running_node(state: State, runtime: Runtime) -> dict:
    for chunk in process_large_file_in_chunks(state["file_path"]):
        await do_async_work(chunk)
        runtime.heartbeat()   # signal "still alive" to reset idle_timeout
    return {"result": "done"}

builder.add_node(
    "processor",
    long_running_node,
    timeout=TimeoutPolicy(
        idle_timeout=60.0,
        refresh_on="heartbeat",   # only heartbeat() resets the idle timer
    ),
)
```

---

## `CachePolicy` — Node-Level Result Caching

`CachePolicy` caches the return value of a node based on its input. On a cache hit the node is skipped entirely and the cached result is used.

```python
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

def expensive_embedding_node(state: State) -> dict:
    """Compute text embeddings — slow and deterministic for the same input."""
    # ... call embedding API ...
    return {"embedding": [0.1, 0.2, 0.3]}

builder.add_node(
    "embed",
    expensive_embedding_node,
    cache_policy=CachePolicy(ttl=3600),   # cache hits valid for 1 hour
)

# Wire up the cache store at compile time
cache = InMemoryCache()
graph = builder.compile(cache=cache)
```

### Custom cache key function

By default the cache key is a hash of the node's input (via pickle). Supply `key_func` to control exactly what is hashed:

```python
from langgraph.types import CachePolicy

def query_cache_key(state: State) -> str:
    """Only the query text matters — ignore metadata fields."""
    return state.get("query", "")

builder.add_node(
    "llm_lookup",
    llm_node,
    cache_policy=CachePolicy(
        key_func=query_cache_key,
        ttl=600,   # 10 minutes
    ),
)
```

---

## `Durability` — Checkpoint Write Timing

The `Durability` type controls *when* a checkpoint is persisted to the saver relative to the next step. Pass it as `durability` on `invoke()`, `stream()`, `ainvoke()`, or `astream()` — **not** on `compile()`:

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "my-thread"}}

# "sync" (default): checkpoint confirmed before the next step starts.
# Safest; adds one checkpoint-write round-trip of latency per step.
result = graph.invoke(input, cfg, durability="sync")

# "async": checkpoint written in the background while the next step executes.
# Higher throughput; tiny window where a crash could lose the last checkpoint.
result = graph.invoke(input, cfg, durability="async")

# "exit": checkpoint written only when the graph exits (or raises).
# Fastest; only safe when you don't need per-step recovery.
result = graph.invoke(input, cfg, durability="exit")

# Works identically with stream / ainvoke / astream:
for event in graph.stream(input, cfg, durability="async", stream_mode="updates"):
    print(event)
```

Choose `"sync"` for human-in-the-loop flows (required for `interrupt()`), `"async"` for autonomous pipelines where max throughput matters, and `"exit"` for batch jobs that simply need a final snapshot.

---

## `IsLastStep` and `RemainingSteps` — Loop Safeguards

`IsLastStep` and `RemainingSteps` are **managed values** — LangGraph injects them automatically into any node that declares them as state fields. They protect against runaway loops when the graph hits its recursion limit (`recursion_limit`, default 25).

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # IsLastStep is True when this is the final allowed step
    is_last_step: IsLastStep
    # RemainingSteps is how many more steps the graph can take
    remaining_steps: RemainingSteps

def agent_node(state: AgentState) -> dict:
    if state["is_last_step"]:
        # Graceful shutdown — return a safe final message
        return {"messages": [{"role": "assistant", "content":
            "I've reached the step limit. Here's what I found so far..."}]}

    if state["remaining_steps"] <= 3:
        # Warn early — give the LLM a heads-up before hard stop
        system_hint = f"You have {state['remaining_steps']} steps remaining. Conclude soon."
    else:
        system_hint = ""

    # ... call LLM with optional system_hint ...
    return {"messages": []}

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile()

# Override the default recursion limit (default=25) at call time
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Research this topic"}]},
    config={"recursion_limit": 50},   # allow up to 50 steps
)
```

`IsLastStep` and `RemainingSteps` are read-only — you cannot write them back from a node. LangGraph manages them internally based on the step counter and `recursion_limit`.

---

## `Topic` Channel — Fan-In Accumulation

A `Topic` channel collects values from multiple parallel nodes and delivers them as a list. It is the fan-in counterpart to `Send`-based fan-out.

```python
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels import Topic   # langgraph.channels.topic.Topic
from langgraph.types import Send

class MapReduceState(TypedDict):
    subjects: list[str]
    # Annotated with operator.add so results from parallel workers accumulate
    results: Annotated[list[str], operator.add]

def fan_out(state: MapReduceState) -> list[Send]:
    """Dispatch one worker per subject in parallel."""
    return [Send("worker", {"subject": s}) for s in state["subjects"]]

def worker(state: dict) -> dict:
    subject = state["subject"]
    # Each worker returns its own result; operator.add merges them all
    return {"results": [f"Analysis of {subject}"]}

def aggregate(state: MapReduceState) -> dict:
    summary = f"Processed {len(state['results'])} topics"
    return {"results": [summary]}

builder = StateGraph(MapReduceState)
builder.add_node("worker", worker)
builder.add_node("aggregate", aggregate)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("worker", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()
result = graph.invoke({"subjects": ["AI", "ML", "NLP"], "results": []})
print(result["results"])
# ['Analysis of AI', 'Analysis of ML', 'Analysis of NLP', 'Processed 3 topics']
```

`Topic(accumulate=True)` retains values **across steps** (not just within one step). Use it for event logs or when you want a persistent running list rather than a per-step collection.

---

## `Send` with Timeout

`Send` accepts an optional `timeout` parameter (added in v1.2.x) to cap the execution of the dispatched task:

```python
from langgraph.types import Send, TimeoutPolicy
from datetime import timedelta

def dispatch_searches(state: State) -> list[Send]:
    return [
        Send(
            "search_node",
            {"query": q},
            timeout=timedelta(seconds=15),   # each search task gets 15 s
        )
        for q in state["queries"]
    ]
```

A float is also accepted as a shorthand `run_timeout`:

```python
Send("slow_node", {"data": payload}, timeout=30.0)
```

---

## `create_react_agent` Migration Notice

`create_react_agent` from `langgraph.prebuilt` is **deprecated** since langgraph 1.2.x. The replacement is `create_agent` from the `langchain` package:

```python
# Deprecated — still works but emits a DeprecationWarning
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model, tools)

# Preferred — use langchain.agents (requires langchain >= 0.3)
from langchain.agents import create_agent
agent = create_agent(model, tools)
```

`create_agent` adds a flexible middleware system (`AgentMiddleware`) with `wrap_tool_call` support at the agent level rather than the node level.

---

## Async Execution

All `invoke`/`stream` methods have async twins. Use them in FastAPI endpoints, asyncio scripts, or anywhere you need concurrent graph execution:

```python
import asyncio
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def async_node(state: State) -> dict:
    # Async I/O — doesn't block the event loop
    await asyncio.sleep(0.01)
    return {"messages": [{"role": "assistant", "content": "async reply"}]}

builder = StateGraph(State)
builder.add_node("respond", async_node)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile(checkpointer=InMemorySaver())

async def main():
    cfg = {"configurable": {"thread_id": "async-1"}}

    # ainvoke — single result
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        cfg,
    )
    print(result["messages"][-1].content)

    # astream — events as they arrive
    async for event in graph.astream(
        {"messages": [{"role": "user", "content": "Again"}]},
        cfg,
        stream_mode="updates",
    ):
        print(event)

    # abatch — multiple inputs concurrently
    results = await graph.abatch(
        [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(3)],
        [{"configurable": {"thread_id": f"b-{i}"}} for i in range(3)],
    )

asyncio.run(main())
```

---

## Production Deployment

### Docker Setup

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["langgraph", "run", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t my-agent:v1 .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  my-agent:v1
```

### CLI Configuration (`langgraph.json`)

```json
{
  "dependencies": ["langchain_anthropic", "langchain_tavily", "./agents"],
  "graphs": {
    "main_agent": "./agents.py:graph",
    "research_agent": "./agents.py:research_graph"
  },
  "env": "./.env",
  "python_version": "3.11"
}
```

### Remote Execution via SDK

```python
from langgraph_sdk import get_client
import asyncio

async def main():
    client = get_client(url="https://my-deployment.example.com")

    assistants = await client.assistants.search()
    assistant_id = assistants[0]["assistant_id"]
    thread = await client.threads.create()

    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input={"query": "Research AI trends"},
    ):
        if chunk.event == "messages/partial":
            print(chunk.data[0]["content"], end="", flush=True)

asyncio.run(main())
```

---

## Common Patterns Summary

| Pattern | Use Case | Key Idea |
|---|---|---|
| **Linear** | Simple pipelines | Node A → B → C → END |
| **Conditional** | Decision trees | `add_conditional_edges` with router fn |
| **Looping** | Iterations | Self-referencing edge + `IsLastStep` guard |
| **Supervisor** | Multi-agent | Central router dispatching to specialists |
| **Map-Reduce** | Parallel work | `Send` fan-out + `operator.add` fan-in |
| **ReAct** | Autonomous agent | Reason → Action → Observe loop |
| **Reflection** | Quality improvement | Self-critique → Refine loop |
| **Interrupt** | Human approval | `interrupt()` pause + `Command(resume=...)` |
| **Retry** | Resilience | `RetryPolicy` on flaky nodes |
| **Timeout** | Reliability | `TimeoutPolicy` run_timeout / idle_timeout |
| **Cache** | Performance | `CachePolicy` + `InMemoryCache` |

---

## Troubleshooting

### "Checkpointer must be provided for interrupts"

`interrupt()` requires a checkpointer. Always compile with one, even in tests:

```python
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())
```

### State not persisting across invocations

Missing or inconsistent `thread_id`:

```python
config = {"configurable": {"thread_id": "session-abc"}}
result = graph.invoke(input, config=config)   # same config every call
```

### `InvalidUpdateError`: "Can receive only one value per step"

A plain (`non-Annotated`) state key received updates from two nodes in the same step. Use `Annotated` with a reducer:

```python
import operator
from typing import Annotated

class State(TypedDict):
    # Wrong — crashes when two nodes write to "results" in the same step
    # results: list[str]

    # Correct — operator.add merges lists from parallel nodes
    results: Annotated[list[str], operator.add]
```

### Reducer function not working

`Annotated` must wrap the type, not the whole `TypedDict` field:

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Wrong
    # messages: list  →  add_messages never called

    # Correct
    messages: Annotated[list, add_messages]
```

### Tools not being called

The model must be bound to tools via `.bind_tools()`:

```python
model_with_tools = model.bind_tools(tools)
# Use model_with_tools inside the node — not the bare model
```

### Infinite loops / recursion limit exceeded

Add a safeguard using `RemainingSteps`:

```python
from langgraph.managed import RemainingSteps

class State(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: RemainingSteps   # auto-injected

def agent(state: State) -> dict:
    if state["remaining_steps"] <= 2:
        return {"messages": [{"role": "assistant", "content": "Wrapping up."}]}
    # normal LLM call ...
```

Or raise the limit at invocation time:

```python
graph.invoke(input, config={"recursion_limit": 100})
```

---

## Node-Level Reliability Quick Reference

| Feature | Import | How to add |
|---|---|---|
| Retry with backoff | `RetryPolicy` from `langgraph.types` | `add_node(..., retry_policy=RetryPolicy(...))` |
| Hard / idle timeout | `TimeoutPolicy` from `langgraph.types` | `add_node(..., timeout=TimeoutPolicy(...))` |
| Result caching | `CachePolicy` from `langgraph.types` | `add_node(..., cache_policy=CachePolicy(...))` + `compile(cache=...)` |
| Loop safeguard | `IsLastStep`, `RemainingSteps` from `langgraph.managed` | Add as state fields; auto-injected |
| Send with timeout | `Send` from `langgraph.types` | `Send(node, arg, timeout=30.0)` |
| Checkpoint timing | `Durability` from `langgraph.types` | `compile(..., durability="async")` |

---

## Next Steps

You've finished the Zero → Hero path. Where to go from here:

- **Build something real** — pick a recipe from the [Recipes collection](/langgraph-guide/python/langgraph_recipes/) (RAG, support router, research agent, doc pipeline, long-term memory chat).
- **Ship it** — read the full [Production Guide](/langgraph-guide/python/langgraph_production_guide/) covering Kubernetes, observability, cost tracking, and disaster recovery.
- **Scale it** — see [Performance Optimization](/langgraph-guide/python/langgraph_performance_optimization/) and [Observability](/langgraph-guide/python/langgraph_observability_python/).
- **Stream it** — the [FastAPI streaming server example](/langgraph-guide/python/langgraph_streaming_server_fastapi/) shows token-level SSE from a compiled graph.

Welcome to durable, stateful, production-grade agent systems.

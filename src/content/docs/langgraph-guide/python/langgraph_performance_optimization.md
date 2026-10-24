---
title: "LangGraph Performance (Python)"
description: "Making LangGraph graphs faster and cheaper: fewer model calls, parallel super-steps and Send fan-out with max_concurrency, batching, streaming, node caching, checkpoint durability and state size."
framework: "langgraph"
language: "python"
sidebar:
  label: "Performance"
---

Checked against `langgraph` %%version=langgraph%% and `langchain` %%version=langchain%% (%%versions-checked%%). This page lists the levers LangGraph gives you, roughly in order of effect. It quotes no benchmark figures: measure your own graph with LangSmith traces, which show the time spent in each node and model call.

## 1. Make fewer model calls

In most agents the model calls dominate run time and cost, so start there.

- **Cap the loop.** Set `recursion_limit` on every agent run (the default, 10007, does not stop a looping agent in practice). On `create_agent`, `ModelCallLimitMiddleware(run_limit=...)` and `ToolCallLimitMiddleware(...)` give finer limits.
- **Keep the context short.** `SummarizationMiddleware` and `ContextEditingMiddleware` shrink long histories; in a hand-built graph, trim messages before the model call.
- **Offer fewer tools.** `LLMToolSelectorMiddleware` picks a subset of a large tool list per call with a small model.
- **Use a smaller model for simple steps.** Route easy requests to `%%model=anthropic-small%%` (or similar) in a separate node, or switch per call in `wrap_model_call` with `request.override(model=...)`.

The middleware are described in [Chapter 8](/langgraph-guide/python/chapter-08-middleware-hooks/#built-in-middleware).

## 2. Run independent work in parallel

LangGraph runs every node that is ready in the same **super-step** concurrently: two nodes with edges from the same parent, or the targets of a list of `Send` objects. Sync nodes run in a thread pool; async nodes run on the event loop, which is the better choice for I/O-bound work.

`config["max_concurrency"]` caps how many tasks run at once, which keeps a large fan-out within a provider's rate limit:

```python
import asyncio, operator, time
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class State(TypedDict, total=False):
    urls: list[str]
    pages: Annotated[list[str], operator.add]
    summary: str


class FetchInput(TypedDict):
    url: str


running = {"now": 0, "peak": 0}


async def fetch(state: FetchInput) -> dict:
    running["now"] += 1
    running["peak"] = max(running["peak"], running["now"])
    await asyncio.sleep(0.1)                     # stands in for an HTTP or model call
    running["now"] -= 1
    return {"pages": [f"content of {state['url']}"]}


def fan_out(state: State) -> list[Send]:
    return [Send("fetch", {"url": u}) for u in state["urls"]]


def summarise(state: State) -> dict:
    return {"summary": f"{len(state['pages'])} pages"}


graph = (
    StateGraph(State)
    .add_node("fetch", fetch)
    .add_node("summarise", summarise)
    .add_conditional_edges(START, fan_out, ["fetch"])
    .add_edge("fetch", "summarise")
    .add_edge("summarise", END)
    .compile()
)

urls = [f"https://example.com/{i}" for i in range(20)]
start = time.perf_counter()
result = asyncio.run(graph.ainvoke({"urls": urls}, {"max_concurrency": 5}))
print(result["summary"], "peak concurrency:", running["peak"])   # 20 pages peak concurrency: 5
print(f"{time.perf_counter() - start:.1f}s")                      # about 0.4s: 4 waves of 5
```

Notes:

- Return the `Send` objects from a routing function (`add_conditional_edges`), or from a node as `Command(goto=[Send(...), ...])`; a node that returns a bare list of `Send` fails.
- `fetch` writes to `pages`, which has an `operator.add` reducer; without one, parallel writes to the same key raise `InvalidUpdateError`.
- `summarise` runs once, after every `fetch` task in the super-step has finished. For fan-ins across several steps, `add_node(..., defer=True)` delays a node until all other work has drained ([StateGraph reference](/langgraph-guide/python/reference-state-graph/)).
- `Send(..., timeout=...)` limits one task; see the [Command and Send reference](/langgraph-guide/python/reference-command-and-send/).

## 3. Batch independent runs

`batch`/`abatch` run many inputs through the graph; `max_concurrency` applies here too:

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict, total=False):
    text: str
    label: str


async def classify(state: State) -> dict:
    await asyncio.sleep(0.05)
    return {"label": "long" if len(state["text"]) > 20 else "short"}


graph = StateGraph(State).add_node("classify", classify).add_edge(START, "classify").compile()

inputs = [{"text": t} for t in ["hi", "a much longer piece of text here", "ok"]]
results = asyncio.run(graph.abatch(inputs, config={"max_concurrency": 2}))
print([r["label"] for r in results])   # ['short', 'long', 'short']

# With a checkpointer, each input needs its own thread:
saved = StateGraph(State).add_node("classify", classify).add_edge(START, "classify").compile(checkpointer=InMemorySaver())
configs = [{"configurable": {"thread_id": f"doc-{i}"}} for i in range(len(inputs))]
print(len(asyncio.run(saved.abatch(inputs, configs, durability="exit"))))   # 3
```

## 4. Stream for perceived latency

Streaming does not make a run finish sooner, but users see output earlier. `stream_mode="messages"` yields model tokens as they arrive; `"updates"` yields each node's output when it finishes; `"custom"` carries progress events from inside a node. See [Chapter 6](/langgraph-guide/python/chapter-06-streaming-and-debugging/) and the [streaming modes reference](/langgraph-guide/python/reference-streaming-modes/).

## 5. Cache deterministic steps

`add_node(..., cache_policy=CachePolicy(ttl=...))` with `compile(cache=InMemoryCache())` skips a node whose input it has seen before. It suits expensive deterministic work (retrieval, parsing, embeddings) rather than model calls whose output should vary. Options, keys and clearing are in [Node policies](/langgraph-guide/python/reference-state-graph/#node-policies); `SqliteCache` and `RedisCache` (shared across replicas) are in the [cache backends reference](/langgraph-guide/python/reference-cache-backends/).

Provider-side prompt caching (for example Anthropic's) is separate and is configured on the chat model, not in LangGraph.

## 6. Write fewer, smaller checkpoints

- **Durability.** The default, `durability="async"`, writes a checkpoint after every super-step without blocking the next one. `durability="exit"` writes only when the run ends or pauses, which suits short runs that need no mid-run recovery. Pass it to `invoke`/`stream`, not `compile()`. ([Chapter 10](/langgraph-guide/python/chapter-10-production/#durability-how-often-state-is-written).)
- **State size.** Every checkpoint serialises the channels that changed. Keep large documents, files and embeddings out of graph state: store them elsewhere (or in the [Store](/langgraph-guide/python/reference-store/)) and keep IDs in state.
- **Append-heavy channels.** A long message list or event log is re-serialised in full whenever it changes. `DeltaChannel` (beta, 1.2+) stores only each step's writes and rebuilds the value on read ([Channels reference](/langgraph-guide/python/reference-channels/#deltachannel-beta)).
- **Old checkpoints.** `PostgresSaver.prune(thread_ids, strategy="keep_latest")` drops all but the latest checkpoint of finished threads ([checkpointers reference](/langgraph-guide/python/reference-checkpointers/#keeping-history-small-prune)).
- **Narrow node inputs.** `add_node(..., input_schema=...)` gives a node only the keys it needs, which also keeps traces small.
- **Connection pooling.** With Postgres under concurrent load, give the saver an `AsyncConnectionPool` rather than a single connection ([checkpointers reference](/langgraph-guide/python/reference-checkpointers/)).

## 7. Bound slow calls

A hung HTTP or model call holds a worker. Put a `timeout=` (async nodes only) and a `RetryPolicy` on nodes that call external services; `TimeoutPolicy(idle_timeout=...)` cancels a call that stops producing tokens without capping a long but healthy one. See [Node policies](/langgraph-guide/python/reference-state-graph/#node-policies).

## Measuring

- Turn on LangSmith tracing (`LANGSMITH_TRACING=true`); each node and model call is a span with its duration and token counts.
- `stream_mode="debug"` (or `"tasks"`) shows when each task starts and finishes, which makes it easy to see whether branches overlap.
- For load tests, run a representative input set against `langgraph up` or a staging deployment with your usual load tool, and compare traces before and after each change.

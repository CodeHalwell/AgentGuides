---
title: "Functional API (`@entrypoint`, `@task`) — API reference"
description: "Turn ordinary Python functions into LangGraph workflows with @entrypoint and @task — the imperative alternative to StateGraph, with the same checkpointing, streaming, and interrupt semantics."
framework: langgraph
language: python
sidebar:
  label: "Ref · Functional API"
  order: 33
---

# Functional API — API reference

Verified against **`langgraph==1.2.4`** (module: `langgraph.func`).

The Functional API lets you author a graph as a plain Python function instead of explicitly building a `StateGraph`. The result is still a `Pregel` object with the same `invoke` / `stream` / `get_state` / `update_state` surface — so you get checkpointing, interrupts, streaming, and time travel for free.

Exports from `langgraph.func`: **`task`**, **`entrypoint`** (and nested `entrypoint.final`).

## Minimal runnable example

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver


@task
def fetch(url: str) -> str:
    # Pretend this hits the network.
    return f"content of {url}"


@task
def summarize(text: str) -> str:
    return f"summary: {text[:20]}"


@entrypoint(checkpointer=InMemorySaver())
def pipeline(urls: list[str]) -> list[str]:
    pages = [fetch(u) for u in urls]           # fetch futures, in parallel
    summaries = [summarize(p.result()) for p in pages]  # summarize futures
    return [s.result() for s in summaries]     # resolve before returning


cfg = {"configurable": {"thread_id": "run-1"}}
print(pipeline.invoke(["a", "b"], cfg))
# ['summary: content of a', 'summary: content of b']
```

Key observations:

- `@task` returns a `SyncAsyncFuture` when called. Call `.result()` to block, or collect futures and resolve later (parallelism).
- Tasks may **only** be called from inside an `@entrypoint` (or another graph node).
- Once decorated, `pipeline` behaves exactly like a compiled `StateGraph`: it has `invoke`, `stream`, `ainvoke`, `astream`, `get_state`, `update_state`, `get_state_history`.

## `@task`

```python
from langgraph.func import task
from langgraph.types import RetryPolicy, CachePolicy

@task(
    name="fetch_page",                          # default: function __name__
    retry_policy=RetryPolicy(max_attempts=3),
    cache_policy=CachePolicy(ttl=300),
)
def fetch(url: str) -> str: ...
```

Rules:

- Inputs and outputs must be **serializable** (JSON-plus via `JsonPlusSerializer`) when the entrypoint has a checkpointer — tasks are the unit the checkpointer caches.
- Async tasks require Python 3.11+.
- Using `retry_policy=<policy>` or `cache_policy=<policy>` turns on the same retry/cache machinery that `StateGraph` nodes use. Pass a sequence to `retry_policy` for ordered fallbacks.
- Calling `fetch("x")` inside an entrypoint returns a future (`SyncAsyncFuture[T]`). Outside an entrypoint it raises.
- `fetch.clear_cache(cache)` (or `aclear_cache`) wipes cached results for this task.

Deprecated kwarg: `retry=` (renamed to `retry_policy=` in v0.5 — still works with a warning).

### `task.clear_cache()` / `task.aclear_cache()`

When a task has a `cache_policy`, its results are keyed by the input hash. Use `clear_cache` to invalidate stale entries without rebuilding the whole graph:

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()


@task(cache_policy=CachePolicy(ttl=3600))
def embed(text: str) -> list[float]:
    return call_embedding_api(text)


# Inside a maintenance routine:
embed.clear_cache(cache)          # sync — wipes all cached results for `embed`
await embed.aclear_cache(cache)   # async variant
```

`clear_cache` only clears results for the decorated task, not the whole cache. It is a no-op if `cache_policy=None`.

### `timeout` on `@task` — `TimeoutPolicy`

The `timeout=` parameter accepts a plain `float` (seconds), `timedelta`, or a `TimeoutPolicy` object. Only async tasks support timeouts — sync tasks raise `ValueError` at decoration time.

```python
import asyncio
from datetime import timedelta
from langgraph.func import entrypoint, task
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver


@task(
    timeout=TimeoutPolicy(
        idle_timeout=15.0,      # abort if silent for 15 s
        refresh_on="heartbeat", # only runtime.heartbeat() resets the clock
    ),
    retry_policy=RetryPolicy(max_attempts=3),
)
async def scrape(url: str) -> str:
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            return await r.text()


@entrypoint(checkpointer=InMemorySaver())
async def crawl_pipeline(urls: list[str]) -> list[str]:
    futures = [scrape(u) for u in urls]
    return [f.result() for f in futures]
```

`TimeoutPolicy` fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `run_timeout` | `float \| timedelta \| None` | `None` | Hard wall-clock cap. Never refreshed. |
| `idle_timeout` | `float \| timedelta \| None` | `None` | Max time between progress signals. |
| `refresh_on` | `"auto" \| "heartbeat"` | `"auto"` | What resets the idle clock. |

### `runtime.heartbeat()` inside a task

When `idle_timeout` is set and `refresh_on="heartbeat"`, call `runtime.heartbeat()` from inside the task to prove it is still making progress. Access the `Runtime` via the `@entrypoint`'s injectable `runtime` parameter — tasks themselves do not receive `runtime` directly, but you can thread it through:

```python
from langgraph.runtime import Runtime
from langgraph.func import entrypoint, task
from langgraph.types import TimeoutPolicy
from langgraph.checkpoint.memory import InMemorySaver


@task(timeout=TimeoutPolicy(idle_timeout=30.0, refresh_on="heartbeat"))
async def long_task(items: list[str], heartbeat_fn) -> list[str]:
    results = []
    for item in items:
        results.append(await process(item))
        heartbeat_fn()   # reset idle clock after each item
    return results


@entrypoint(checkpointer=InMemorySaver())
async def pipeline(inp: dict, runtime: Runtime) -> dict:
    # Pass runtime.heartbeat as a callback so the task can signal progress
    future = long_task(inp["items"], runtime.heartbeat)
    return {"results": future.result()}
```

## `@entrypoint`

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

@entrypoint(
    checkpointer=InMemorySaver(),       # BaseCheckpointSaver | None
    store=InMemoryStore(),              # BaseStore | None
    cache=None,                         # BaseCache | None
    context_schema=Ctx,                 # type | None
    cache_policy=None,                  # CachePolicy | None (caches the whole workflow)
    retry_policy=None,                  # RetryPolicy | Sequence | None
)
def workflow(input_data: str) -> dict:
    ...
```

Signature rules for the decorated function:

- **Exactly one positional parameter** — the run's input. Use a dict or dataclass to pack multiple values.
- May also declare any of these **injectable** keyword parameters, which the runtime fills in automatically:

| Parameter | Type | Availability |
|---|---|---|
| `config` | `RunnableConfig` | Always |
| `previous` | whatever was saved last time | Only when `checkpointer` is set |
| `runtime` | `Runtime[ContextT]` | Always — exposes `context`, `store`, `stream_writer`, `previous`, `execution_info` |

Both sync and `async def` entrypoints are supported. Generator entrypoints (sync or async) are **not** supported — `@entrypoint` raises `NotImplementedError` when applied to a generator.

Deprecated kwargs that still work with warnings:

- `config_schema=` → use `context_schema=` (deprecated since v0.6).
- `retry=` → use `retry_policy=` (deprecated since v0.5).

## `entrypoint.final`

Return a value to the caller while saving a *different* value into the checkpoint:

```python
from typing import Any
from langgraph.func import entrypoint

@entrypoint(checkpointer=InMemorySaver())
def counter(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    return entrypoint.final(value=previous, save=2 * number)

cfg = {"configurable": {"thread_id": "t"}}
counter.invoke(3, cfg)   # returns 0 (previous was None), saves 6
counter.invoke(1, cfg)   # returns 6 (previous loaded), saves 2
counter.invoke(7, cfg)   # returns 2, saves 14
```

Type it as `entrypoint.final[R, S]` where `R` is the return type and `S` is the saved type.

## Interrupts, human-in-the-loop

`interrupt` and `Command` behave exactly as they do in a `StateGraph`:

```python
from langgraph.types import interrupt, Command

@task
def draft(topic: str) -> str:
    return f"Essay about {topic}"

@entrypoint(checkpointer=InMemorySaver())
def review_flow(topic: str) -> dict:
    essay = draft(topic).result()
    edit = interrupt({"question": "Edit this?", "essay": essay})
    return {"essay": essay, "edit": edit}

cfg = {"configurable": {"thread_id": "r1"}}
for chunk in review_flow.stream("cats", cfg):
    print(chunk)
# {'__interrupt__': (Interrupt(value=..., id=...),)}

# Resume with a human response
for chunk in review_flow.stream(Command(resume="Shorter, please."), cfg):
    print(chunk)
# {'review_flow': {'essay': '...', 'edit': 'Shorter, please.'}}
```

Resuming replays the entrypoint from the top. Task results that were already checkpointed **are not re-executed** — that's the cache guarantee tasks give you.

## Streaming

```python
# Default stream mode for entrypoints is 'updates'.
for update in pipeline.stream(["a", "b"], cfg):
    print(update)

# Token-level streaming from LLMs inside tasks:
for mode, data in pipeline.stream(["a", "b"], cfg, stream_mode=["updates", "messages"]):
    print(mode, data)
```

All stream modes from the [Streaming modes reference](./reference-streaming-modes/) apply — `values`, `updates`, `messages`, `custom`, `checkpoints`, `tasks`, `debug`.

## Custom streaming from inside a task

```python
@task
def with_progress(items: list[str]) -> list[str]:
    from langgraph.config import get_config
    # Or just declare runtime: Runtime in the signature.
    ...
```

Prefer the runtime injection:

```python
from langgraph.runtime import Runtime

@entrypoint(checkpointer=InMemorySaver())
def emit(inp: dict, runtime: Runtime) -> dict:
    runtime.stream_writer({"phase": "start"})
    ...
    runtime.stream_writer({"phase": "done"})
    return {"ok": True}

for ev in emit.stream({}, cfg, stream_mode="custom"):
    print(ev)
```

## Feature matrix: Functional API vs StateGraph

| Feature | `@entrypoint` + `@task` | `StateGraph` |
|---|---|---|
| Imperative control flow (if/loops/try) | Natural | Encode via conditional edges |
| Parallel fan-out | Collect task futures, resolve last | `Send`s or list-returning conditional edges |
| Shared mutable state | `previous` via checkpointer | Channels + reducers |
| Per-node retry/cache | On `@task` | On `add_node(...)` |
| Checkpointing, interrupts, streaming | Same | Same |
| Subgraphs | Add a compiled graph as a task | `add_node(sub_graph)` |
| Visualization | Limited (linear) | Full Mermaid / PNG |
| Best when | Logic is straight-line or branchy Python | Logic is a DAG with reducer semantics |

Both compile to `Pregel`, so they interoperate — you can pass an entrypoint as a node inside a `StateGraph`, or call a compiled `StateGraph` as a `@task`.

## Patterns

### 1. Map-reduce with futures

```python
@task
def score(item: str) -> float:
    return len(item) / 10.0

@entrypoint()
def max_score(items: list[str]) -> float:
    futures = [score(i) for i in items]
    return max(f.result() for f in futures)
```

### 2. Retry only transient errors

```python
import httpx
from langgraph.types import RetryPolicy

@task(retry_policy=RetryPolicy(
    max_attempts=5,
    retry_on=(httpx.TransportError, httpx.HTTPStatusError),
))
async def http_get(url: str) -> str:
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.text
```

### 3. Human review loop with resume

```python
@entrypoint(checkpointer=InMemorySaver())
def assisted_write(topic: str) -> dict:
    draft_text = draft(topic).result()
    corrections = interrupt({"draft": draft_text})
    final = revise(draft_text, corrections).result()
    return {"draft": draft_text, "final": final}
```

Stream once to collect the interrupt, call `assisted_write.stream(Command(resume=...), cfg)` to continue.

### 4. Running counter persisted across calls

```python
@entrypoint(checkpointer=InMemorySaver())
def hit(inp: None = None, *, previous: int | None = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    nxt = previous + 1
    return entrypoint.final(value=nxt, save=nxt)
```

### 5. Inject the store for long-term memory

```python
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

@entrypoint(checkpointer=InMemorySaver(), store=store)
def remember(inp: dict, runtime: Runtime) -> dict:
    uid = inp["user_id"]
    assert runtime.store is not None
    runtime.store.put(("mem", uid), "latest", {"text": inp["text"]})
    return {"saved": True}
```

### 6. Typed run-scoped context with `context_schema`

Pass per-run metadata (tenant ID, feature flags, DB connection) that should not be checkpointed:

```python
from dataclasses import dataclass
from langgraph.func import entrypoint, task
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class RunContext:
    tenant_id: str
    feature_flags: dict[str, bool]


@task
def process_item(item: str, tenant_id: str) -> str:
    return f"[{tenant_id}] processed: {item}"


@entrypoint(checkpointer=InMemorySaver(), context_schema=RunContext)
def pipeline(items: list[str], runtime: Runtime[RunContext]) -> list[str]:
    ctx = runtime.context   # type: RunContext — not persisted to checkpoint
    futures = [process_item(item, ctx.tenant_id) for item in items]
    return [f.result() for f in futures]


result = pipeline.invoke(
    ["a", "b", "c"],
    config={"configurable": {"thread_id": "run-1"}},
    context=RunContext(tenant_id="acme", feature_flags={"beta": True}),
)
# ['[acme] processed: a', '[acme] processed: b', '[acme] processed: c']
```

`context_schema` must be a dataclass, Pydantic model, or `TypedDict`. The context object is passed as the `context=` keyword argument to `invoke`/`stream` — separate from `config`. It is never persisted to checkpoints and never appears in `get_state()` output.

### 7. Cached task with manual cache invalidation

Cache expensive computations across runs; flush stale entries with `clear_cache`:

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver

cache = InMemoryCache()


@task(cache_policy=CachePolicy(ttl=3600))
def embed_document(text: str) -> list[float]:
    return slow_embedding_api(text)


@entrypoint(checkpointer=InMemorySaver(), cache=cache)
def index_pipeline(docs: list[str]) -> list[list[float]]:
    futures = [embed_document(d) for d in docs]
    return [f.result() for f in futures]


cfg = {"configurable": {"thread_id": "idx-1"}}
index_pipeline.invoke(["hello", "world"], cfg)     # computes embeddings
index_pipeline.invoke(["hello", "world"], cfg)     # served from cache

# Invalidate after model upgrade:
embed_document.clear_cache(cache)
index_pipeline.invoke(["hello", "world"], cfg)     # recomputes
```

### 8. Parallel tasks with per-task timeout and retry

```python
import asyncio
import httpx
from langgraph.func import entrypoint, task
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver


@task(
    timeout=TimeoutPolicy(run_timeout=10.0),
    retry_policy=RetryPolicy(
        max_attempts=3,
        retry_on=(httpx.TransportError, httpx.HTTPStatusError),
        backoff_factor=2.0,
    ),
)
async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


@entrypoint(checkpointer=InMemorySaver())
async def crawl(urls: list[str]) -> list[str]:
    futures = [fetch_url(u) for u in urls]
    results = []
    for f in futures:
        try:
            results.append(f.result())
        except Exception as e:
            results.append(f"ERROR: {e}")
    return results


cfg = {"configurable": {"thread_id": "crawl-1"}}
await crawl.ainvoke(["https://example.com", "https://httpbin.org/get"], cfg)
```

### 8. Named `@task` with a custom `key_func` for the cache

Override the cache key when the default pickle-hash is too broad or too narrow:

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver

cache = InMemoryCache()


def url_key(url: str) -> str:
    """Canonical cache key: strip protocol and trailing slash."""
    return url.removeprefix("https://").removeprefix("http://").rstrip("/")


@task(
    name="fetch",   # override the function name shown in traces
    cache_policy=CachePolicy(key_func=url_key, ttl=600),
)
def fetch_page(url: str) -> str:
    print(f"[network] fetching {url}")
    return f"content:{url}"


@entrypoint(checkpointer=InMemorySaver(), cache=cache)
def pipeline(urls: list[str]) -> list[str]:
    futures = [fetch_page(u) for u in urls]
    return [f.result() for f in futures]


cfg = {"configurable": {"thread_id": "cache-1"}}
pipeline.invoke(["https://example.com/", "http://example.com"], cfg)
# [network] fetching https://example.com/  — only one fetch; both URLs share the key
```

### 9. Typed `context_schema` on `@entrypoint`

Use `context_schema=` to pass typed, read-only run context (user ID, feature flags, DB connections) without putting it in the serializable input:

```python
from dataclasses import dataclass
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


@dataclass
class UserCtx:
    user_id: str
    is_premium: bool = False


store = InMemoryStore()


@task
def load_history(user_id: str) -> list[str]:
    """Load previous queries for this user from the store."""
    items = store.search(("history", user_id))
    return [i.value["query"] for i in items]


@entrypoint(
    checkpointer=InMemorySaver(),
    store=store,
    context_schema=UserCtx,
)
def answer(query: str, runtime: Runtime[UserCtx]) -> dict:
    ctx = runtime.context
    history = load_history(ctx.user_id).result()
    model = "claude-opus-4-7" if ctx.is_premium else "claude-haiku-4-5"

    response = f"[{model}] Answering '{query}' (history: {history})"

    # Save to cross-thread store
    if runtime.store:
        runtime.store.put(
            ("history", ctx.user_id),
            f"q-{len(query)}",
            {"query": query},
        )

    return {"response": response, "user_id": ctx.user_id}


cfg = {"configurable": {"thread_id": "session-42"}}

# context= is passed at call time — not serialized into the checkpoint
result = answer.invoke(
    "What is LangGraph?",
    cfg,
    context=UserCtx(user_id="alice", is_premium=True),
)
print(result["response"])
```

### 10. Entrypoint calling a compiled `StateGraph` as a task

Both Functional API and `StateGraph` compile to `Pregel`. You can use a compiled `StateGraph` inside a `@task`:

```python
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


# ── A compiled StateGraph sub-pipeline ──────────────────────────────────────

class CleanState(TypedDict):
    raw: str
    clean: str


def strip_node(state: CleanState) -> dict:
    return {"clean": state["raw"].strip().lower()}


cleaner = (
    StateGraph(CleanState)
    .add_node("strip", strip_node)
    .add_edge(START, "strip")
    .add_edge("strip", END)
    .compile()
)


# ── Wrap the StateGraph as a @task ──────────────────────────────────────────

@task
def clean_text(text: str) -> str:
    result = cleaner.invoke({"raw": text, "clean": ""})
    return result["clean"]


@entrypoint(checkpointer=InMemorySaver())
def process_batch(texts: list[str]) -> list[str]:
    futures = [clean_text(t) for t in texts]
    return [f.result() for f in futures]


cfg = {"configurable": {"thread_id": "batch-1"}}
result = process_batch.invoke(["  Hello World  ", "  LangGraph  "], cfg)
print(result)  # ['hello world', 'langgraph']
```

## Gotchas

- **Tasks are futures, not values.** A very common bug: `return tasks` instead of `return [t.result() for t in tasks]`. The future object serializes fine but the caller can't use it outside the graph.
- **Generators aren't allowed.** Authoring `yield` inside an entrypoint raises at decoration time. Use `stream_writer` + `stream_mode="custom"` instead.
- **`interrupt()` requires a checkpointer.** Pass `checkpointer=InMemorySaver()` even in tests — otherwise the raised `GraphInterrupt` escapes unhandled.
- **Resuming replays the whole entrypoint.** Any side effects outside `@task` (print, counters, file writes) run again. Put side effects in tasks, which are cached by the checkpointer.
- **`previous` is only available with a checkpointer.** Without one it defaults to `None` and does not update.
- **Don't `await` a sync task.** A `@task` on a sync function returns `SyncAsyncFuture`; call `.result()`.
- **Serializable I/O.** `JsonPlusSerializer` handles most dataclasses, Pydantic models, `datetime`, `UUID`, and LangChain messages. Avoid live connections, file handles, and locks in task outputs.
- **`context_schema=` data is NOT checkpointed.** Context is injected fresh at each `invoke`/`stream` call. The same `context=` must be passed on every call that resumes from an interrupt.
- **Calling a task outside an entrypoint raises `RuntimeError`.** Tasks are only valid inside `@entrypoint`-decorated functions or other graph nodes. Do not call them from test code, scripts, or top-level module scope without an active entrypoint.

## Breaking changes

| Version | Change |
|---|---|
| 1.2 | `timeout=` parameter added to `@task` and `@entrypoint` (async only). `TimeoutPolicy` dataclass introduced. `task.clear_cache()` / `task.aclear_cache()` added. |
| 1.0 | Functional API graduates out of experimental; `@entrypoint` / `@task` live in `langgraph.func`. |
| 0.6 | `config_schema=` on `@entrypoint` deprecated in favor of `context_schema=`. |
| 0.5 | `retry=` kwarg on `@task` / `@entrypoint` renamed to `retry_policy=`. |

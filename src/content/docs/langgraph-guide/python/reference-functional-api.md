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

Verified against **`langgraph==1.1.9`** (module: `langgraph.func`).

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

## Gotchas

- **Tasks are futures, not values.** A very common bug: `return tasks` instead of `return [t.result() for t in tasks]`. The future object serializes fine but the caller can't use it outside the graph.
- **Generators aren't allowed.** Authoring `yield` inside an entrypoint raises at decoration time. Use `stream_writer` + `stream_mode="custom"` instead.
- **`interrupt()` requires a checkpointer.** Pass `checkpointer=InMemorySaver()` even in tests — otherwise the raised `GraphInterrupt` escapes unhandled.
- **Resuming replays the whole entrypoint.** Any side effects outside `@task` (print, counters, file writes) run again. Put side effects in tasks, which are cached by the checkpointer.
- **`previous` is only available with a checkpointer.** Without one it defaults to `None` and does not update.
- **Don't `await` a sync task.** A `@task` on a sync function returns `SyncAsyncFuture`; call `.result()`.
- **Serializable I/O.** `JsonPlusSerializer` handles most dataclasses, Pydantic models, `datetime`, `UUID`, and LangChain messages. Avoid live connections, file handles, and locks in task outputs.

## Breaking changes

| Version | Change |
|---|---|
| 1.0 | Functional API graduates out of experimental; `@entrypoint` / `@task` live in `langgraph.func`. |
| 0.6 | `config_schema=` on `@entrypoint` deprecated in favor of `context_schema=`. |
| 0.5 | `retry=` kwarg on `@task` / `@entrypoint` renamed to `retry_policy=`. |

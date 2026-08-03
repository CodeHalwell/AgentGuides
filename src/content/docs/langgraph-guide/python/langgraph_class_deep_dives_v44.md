---
title: "LangGraph Class Deep-Dives Vol. 44"
description: "Source-verified deep dives (langgraph==1.2.10) into 10 class groups: BinaryOperatorAggregate (custom reducer channels, Overwrite bypass, operator.add/max/set.union patterns), Topic channel (accumulate vs per-step PubSub, fan-in aggregation, flatten behaviour), EphemeralValue (guard=True single-write enforcement, auto-clear after step, signal passing pattern), NamedBarrierValue (barrier synchronization, consume() reset, parallel-branch join), entrypoint decorator (checkpointer/store/cache wiring, previous parameter, entrypoint.final decoupled save), task + SyncAsyncFuture (parallel fan-out, cache_policy per-task, retry_policy on task, name override), RetryPolicy (NamedTuple fields, backoff formula, retry_on callable), TimeoutPolicy (run_timeout vs idle_timeout, refresh_on=auto vs heartbeat, runtime.heartbeat()), CachePolicy (key_func hash contract, ttl eviction, clear_cache() method), Send (map-reduce fan-out, per-Send timeout override, list composition), Command (update + goto + resume in one return, Command.PARENT subgraph bubble, Command with Send in goto)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 44"
  order: 75
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.10`.

---

## 1 · `BinaryOperatorAggregate` — custom reducer channels

**Module:** `langgraph.channels.binop`

`BinaryOperatorAggregate` is the channel type that `Annotated[T, reducer_fn]` fields compile to inside `StateGraph`. Every time one or more nodes write to the channel in a super-step, `update()` folds each incoming value through `self.operator(accumulator, value)`. A writer may bypass the operator entirely by wrapping the value in `Overwrite(value)`, which replaces the accumulator outright. Only **one** `Overwrite` is accepted per super-step; a second raises `InvalidUpdateError`.

**Key source facts (`langgraph/channels/binop.py`):**

- Constructor signature: `__init__(self, typ: type[Value], operator: Callable[[Value, Value], Value])`. The channel zero-initialises from `typ()` — `int()` → `0`, `list()` → `[]`, `set()` → `set()`.
- `update()` applies `operator` left-to-right over all values written in the same super-step. If the accumulator is still `MISSING` (channel never written) it adopts `values[0]` first.
- `Overwrite` check is done via `_get_overwrite(value)` — if `True` and there's already an earlier `Overwrite` in the same step, `InvalidUpdateError` is raised.
- `checkpoint()` returns the raw accumulated value; `from_checkpoint()` restores it so time-travel replays work correctly.
- Channel equality is based on `_operators_equal()` — two `operator.add` references compare equal (identity check). Any lambda operator also compares equal to **any** other operator (named or lambda), because all lambdas share `__name__ == "<lambda>"` and the comparator short-circuits to `True` on that condition alone.

### Example 1 — numeric aggregation with `operator.add`

```python
import operator
from typing import Annotated
from langgraph.graph import StateGraph, START, END

class CountState(dict):
    pass

# Annotated fields automatically compile to BinaryOperatorAggregate
from typing import TypedDict

class State(TypedDict):
    total: Annotated[int, operator.add]
    tags: Annotated[list[str], operator.add]   # list + list = concat

def node_a(state: State) -> dict:
    return {"total": 1, "tags": ["a"]}

def node_b(state: State) -> dict:
    return {"total": 10, "tags": ["b"]}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", END)
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"total": 0, "tags": []})
print(result)
# {'total': 11, 'tags': ['a', 'b']}  — both nodes ran in parallel, results accumulated
```

### Example 2 — set union reducer

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    seen_ids: Annotated[set[str], lambda a, b: a | b]

def fetch_batch_1(state: State) -> dict:
    return {"seen_ids": {"id-1", "id-2"}}

def fetch_batch_2(state: State) -> dict:
    return {"seen_ids": {"id-2", "id-3"}}

builder = StateGraph(State)
builder.add_node("b1", fetch_batch_1)
builder.add_node("b2", fetch_batch_2)
builder.add_edge(START, "b1")
builder.add_edge(START, "b2")
builder.add_edge("b1", END)
builder.add_edge("b2", END)

graph = builder.compile()
result = graph.invoke({"seen_ids": set()})
print(result["seen_ids"])
# {'id-1', 'id-2', 'id-3'} — union, id-2 deduplicated
```

### Example 3 — `Overwrite` to bypass the operator

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite

class State(TypedDict):
    counter: Annotated[int, operator.add]

def increment(state: State) -> dict:
    return {"counter": 5}           # normal: accumulated with operator.add

def reset_counter(state: State) -> dict:
    # Wrap in Overwrite to bypass operator.add and force-set the value
    return {"counter": Overwrite(0)}

builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_node("reset", reset_counter)
builder.add_edge(START, "inc")
builder.add_edge("inc", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
print(graph.invoke({"counter": 10}))
# {'counter': 0}  — Overwrite() bypassed operator.add; reset wins
```

---

## 2 · `Topic` — multi-value pub/sub channel

**Module:** `langgraph.channels.topic`

`Topic` collects **all** values written to it in a super-step into a `list`, making it useful when multiple parallel branches each contribute an item and you want to read the full list at once. Unlike `BinaryOperatorAggregate`, `Topic` handles any number of writers and flattens nested lists automatically via `_flatten()`.

**Key source facts (`langgraph/channels/topic.py`):**

- `__init__(self, typ: type[Value], accumulate: bool = False)`.
- `accumulate=False` (default): the list is cleared at the *start* of each super-step, so `get()` only returns values written in the **current** step.
- `accumulate=True`: values grow forever across all steps — useful as an append-only event log.
- `update()` calls `_flatten()` which handles both scalars and lists, so a node may write `"x"` or `["x", "y"]` to the same channel.
- `get()` raises `EmptyChannelError` when `self.values` is empty — route around it or use `is_available()` before reading.
- `ValueType` resolves to `Sequence[T]`; `UpdateType` resolves to `T | list[T]`.

### Example 1 — per-step fan-in collector with `Topic`

`Topic` is the right annotation when parallel nodes each contribute a **single item**. Declare the field as `Annotated[Sequence[str], Topic(str)]` and write a scalar string — the channel collects all writes from the same super-step into a sequence.

```python
from typing import Annotated, Sequence, TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END

class FanInState(TypedDict):
    # Topic(str) gathers one string per parallel writer into a sequence
    items: Annotated[Sequence[str], Topic(str)]

def worker_a(state: FanInState) -> dict:
    return {"items": "result-from-a"}   # single scalar, not a list

def worker_b(state: FanInState) -> dict:
    return {"items": "result-from-b"}

def aggregator(state: FanInState) -> dict:
    print("All items:", list(state["items"]))
    return {}

builder = StateGraph(FanInState)
builder.add_node("a", worker_a)
builder.add_node("b", worker_b)
builder.add_node("agg", aggregator)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", "agg")
builder.add_edge("b", "agg")
builder.add_edge("agg", END)

graph = builder.compile()
graph.invoke({"items": []})
# All items: ['result-from-a', 'result-from-b']
```

### Example 2 — `Topic` as accumulating event log (low-level Pregel)

```python
from langgraph.channels.topic import Topic
from langgraph.channels.last_value import LastValue
from langgraph.pregel.main import Pregel
from langgraph.pregel._read import ChannelRead
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.constants import START, END

# Build a graph that accumulates events across multiple invocations
event_log = Topic(str, accumulate=True)
status = LastValue(str)

def log_event(inputs):
    return {"events": inputs.get("event", "tick"), "status": "running"}

# Demonstrate Topic's accumulate=True behaviour directly
t = Topic(str, accumulate=True)
t.update(["event-1"])
t.update(["event-2", "event-3"])
print(list(t.get()))
# ['event-1', 'event-2', 'event-3']  — all retained across update() calls

# Reset: accumulate=False (default)
t2 = Topic(str, accumulate=False)
t2.update(["event-1"])
print(list(t2.get()))   # ['event-1']
t2.update(["event-2"])
print(list(t2.get()))   # ['event-2']  — event-1 cleared
```

### Example 3 — `Send` fan-out with `Topic` collecting parallel results

`Topic` is the right channel when parallel workers each contribute a **single item** and the graph needs to gather them all. Each worker writes one string; `Topic` accumulates every write from the same super-step into a sequence — no list-wrapping or `operator.add` needed.

```python
from typing import Annotated, Sequence, TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class OverallState(TypedDict):
    urls: list[str]
    # Topic(str) collects individual string writes from all parallel workers
    summaries: Annotated[Sequence[str], Topic(str)]

class WorkerState(TypedDict):
    url: str

def summarise(state: WorkerState) -> dict:
    # Write a single string — Topic gathers all parallel writes into a list
    return {"summaries": f"Summary of {state['url']}"}

def fan_out(state: OverallState) -> list:
    return [Send("summarise", {"url": u}) for u in state["urls"]]

builder = StateGraph(OverallState)
builder.add_node("summarise", summarise)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("summarise", END)

graph = builder.compile()
result = graph.invoke({"urls": ["http://a.com", "http://b.com"], "summaries": []})
print(list(result["summaries"]))
# ['Summary of http://a.com', 'Summary of http://b.com']
# Both strings were written in the same super-step; Topic collected them.
```

---

## 3 · `EphemeralValue` — per-step temporary state

**Module:** `langgraph.channels.ephemeral_value`

`EphemeralValue` stores a value for exactly **one super-step** then auto-clears to `MISSING`. It is the channel backing `START` (the input channel) in both `StateGraph` and `entrypoint`-based workflows. Its `guard=True` mode rejects multiple writers in the same step, enforcing single-producer semantics.

**Key source facts (`langgraph/channels/ephemeral_value.py`):**

- `__init__(self, typ: Any, guard: bool = True)`.
- `update(values)`: if `len(values) == 0`, clears the value and returns `True` if it was set. If `len(values) > 1` and `guard=True`, raises `InvalidUpdateError`.
- Clearing happens automatically because at the end of each super-step Pregel calls `update([])` on channels that received no writes — the channel self-clears.
- Use `guard=False` when multiple branches might each signal the same ephemeral key and you want the **last** write to win.
- `is_available()` returns `True` only while the value is set (same step).

### Example 1 — `START` channel is `EphemeralValue` under the hood

```python
from langgraph.channels.ephemeral_value import EphemeralValue

ev = EphemeralValue(str, guard=True)

# Write once — OK
ev.update(["hello"])
print(ev.get())           # 'hello'
print(ev.is_available())  # True

# Step ends — Pregel calls update([]) to clear
ev.update([])
print(ev.is_available())  # False

try:
    ev.get()
except Exception as e:
    print(type(e).__name__)   # EmptyChannelError
```

### Example 2 — guard=True prevents duplicate writes in one step

```python
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.errors import InvalidUpdateError

ev = EphemeralValue(str, guard=True)

try:
    ev.update(["value-a", "value-b"])   # Two writes in one step → error
except InvalidUpdateError as e:
    print("Caught:", e)
    # Caught: ... EphemeralValue(guard=True) can receive only one value per step.

# With guard=False: last write wins silently
ev2 = EphemeralValue(str, guard=False)
ev2.update(["value-a", "value-b"])
print(ev2.get())    # 'value-b'
```

### Example 3 — `EphemeralValue` checkpoint round-trip and restore

`EphemeralValue` participates in checkpointing via `checkpoint()` / `from_checkpoint()`, but is designed to hold a value for one super-step only. This example exercises the checkpoint/restore cycle directly and shows that a restored value persists until the *next* `update([])` clear:

```python
from langgraph.channels.ephemeral_value import EphemeralValue

ev = EphemeralValue(str, guard=True)

# Write a value (simulates a node writing to this channel in step N)
ev.update(["signal-from-node-A"])
print(ev.get())            # 'signal-from-node-A'
print(ev.is_available())   # True

# Checkpoint the value (e.g. before saving to a checkpointer)
saved = ev.checkpoint()
print("Checkpointed:", saved)   # 'signal-from-node-A'

# Restore from checkpoint into a fresh instance (simulates loading state)
ev2 = EphemeralValue(str, guard=True)
ev2_loaded = ev2.from_checkpoint(saved)
print(ev2_loaded.get())    # 'signal-from-node-A' — value restored

# Simulate next super-step starting: Pregel calls update([]) to auto-clear
ev2_loaded.update([])
print(ev2_loaded.is_available())  # False — cleared, as expected for ephemeral data

# The original instance is also cleared after its own next step
ev.update([])
print(ev.is_available())   # False
```

**Design note:** `EphemeralValue` works in **both** the low-level Pregel API and `StateGraph`. Pass a channel instance as the `Annotated` metadata and `StateGraph` uses it directly — no manual resets needed:

```python
from typing import Annotated
from langgraph.channels.ephemeral_value import EphemeralValue

class MyState(TypedDict):
    # auto-clears to MISSING after every super-step
    signal: Annotated[str | None, EphemeralValue(str)]
    result: str

# StateGraph compiles `signal` as EphemeralValue. A value written in super-step N
# is available to downstream nodes in super-step N+1; at the end of super-step N+1
# (if `signal` receives no further writes) Pregel calls update([]) and it clears.
# This makes EphemeralValue ideal for one-shot handoffs between sequential nodes.
```

The `START` channel is itself an `EphemeralValue(input_type)`, so the pattern is native to LangGraph at every level.

---

## 4 · `NamedBarrierValue` — synchronization barrier

**Module:** `langgraph.channels.named_barrier_value`

`NamedBarrierValue` holds a fixed set of **expected names** and becomes available only when *all* of them have been sent. It is LangGraph's built-in synchronisation primitive for joining parallel branches. After the barrier fires, `consume()` resets it so it can be re-armed for the next cycle.

**Key source facts (`langgraph/channels/named_barrier_value.py`):**

- `__init__(self, typ: type[Value], names: set[Value])`. `names` is the complete set of signal values that must arrive before the barrier opens.
- `update(values)`: adds each value to `self.seen`. Raises `InvalidUpdateError` for any value not in `self.names`.
- `get()`: returns `None` if `self.seen == self.names`; raises `EmptyChannelError` otherwise. The meaningful semantics are in *availability*, not the return value.
- `consume()`: if `seen == names`, resets `seen = set()` and returns `True` — used internally by Pregel to reset the barrier after it fires.
- `is_available()` is the correct predicate for conditional edges: `lambda s: s["barrier"]` won't work because `get()` returns `None`.

### Example 1 — join two parallel branches

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue

# Simulate two parallel branches both signalling "ready"
barrier = NamedBarrierValue(str, names={"branch_a", "branch_b"})

print(barrier.is_available())   # False — nothing received yet

barrier.update(["branch_a"])
print(barrier.is_available())   # False — still waiting for branch_b

barrier.update(["branch_b"])
print(barrier.is_available())   # True — all names received

# After firing, consume resets it
barrier.consume()
print(barrier.is_available())   # False — ready for next round
```

### Example 2 — invalid signal raises `InvalidUpdateError`

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.errors import InvalidUpdateError

barrier = NamedBarrierValue(str, names={"a", "b"})

try:
    barrier.update(["c"])   # "c" not in names
except InvalidUpdateError as e:
    print("Caught:", e)
    # Caught: At key '...': Value c not in {'a', 'b'}
```

### Example 3 — fan-in join using direct multi-source edges

In a `StateGraph`, the correct way to join parallel branches is with direct edges from each branch to a common downstream node. Conditional routing functions run *inside* each node's own super-step, before sibling writes are merged — so a conditional edge on `fetcher` cannot see `analyser`'s output. Using explicit edges creates an implicit barrier: `combine` only starts once **both** `fetcher` and `analyser` have completed and their writes have been folded into the merged state.

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    results: Annotated[list[str], operator.add]

def fetcher(state: State) -> dict:
    return {"results": ["fetch-result"]}

def analyser(state: State) -> dict:
    return {"results": ["analysis-result"]}

def combine(state: State) -> dict:
    # Runs only after BOTH fetcher and analyser complete — state is merged here
    print("All done. Results:", state["results"])
    return {}

builder = StateGraph(State)
builder.add_node("fetcher", fetcher)
builder.add_node("analyser", analyser)
builder.add_node("combine", combine)
builder.add_edge(START, "fetcher")
builder.add_edge(START, "analyser")
# Multi-source edge: "combine" is scheduled only after BOTH sources complete
builder.add_edge(["fetcher", "analyser"], "combine")
builder.add_edge("combine", END)

graph = builder.compile()
graph.invoke({"results": []})
# All done. Results: ['fetch-result', 'analysis-result']
```

---

## 5 · `entrypoint` + `entrypoint.final` — Functional API

**Module:** `langgraph.func`

`entrypoint` is a class-decorator that compiles a plain Python function into a full `Pregel` graph. It supports checkpointing, cross-thread stores, caching, retry, and timeout — without needing to declare state schemas or wire up edges. The decorated function's return value becomes the graph output. Use `entrypoint.final` to save a *different* value to the checkpoint than the one returned to the caller.

**Key source facts (`langgraph/func/__init__.py`):**

- `__init__` accepts `checkpointer`, `store`, `cache`, `context_schema`, `cache_policy`, `retry_policy`, `timeout`.
- Internally creates a `Pregel` with a single `PregelNode` that writes to `END` and `PREVIOUS` channels via two `ChannelWriteEntry` mappers (`_pluck_return_value`, `_pluck_save_value`).
- `previous` parameter (injectable): available when `checkpointer` is set; holds the *saved* value from the previous run on the same `thread_id`.
- Generator functions are explicitly rejected: `raise NotImplementedError("Generators are not supported in the Functional API.")`.
- `entrypoint.final(value=..., save=...)` is a dataclass. If `isinstance(return_value, entrypoint.final)`, `value` is returned to the caller and `save` is persisted.
- `context_schema` lets you pass a typed context object (replaces deprecated `config_schema`).

### Example 1 — basic entrypoint with checkpointing

```python
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@task
def fetch_data(query: str) -> str:
    # Simulate an API call — results cached in checkpoint
    return f"data for '{query}'"

@entrypoint(checkpointer=checkpointer)
def search_workflow(query: str, previous: Optional[str] = None) -> str:
    if previous:
        return f"Cached: {previous}"
    result = fetch_data(query).result()
    return result

config = {"configurable": {"thread_id": "thread-1"}}

first  = search_workflow.invoke("langgraph", config)
second = search_workflow.invoke("langgraph", config)   # same thread_id → previous is set

print(first)   # "data for 'langgraph'"
print(second)  # "Cached: data for 'langgraph'"
```

### Example 2 — `entrypoint.final` to decouple return value from saved state

```python
from typing import Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def counter(increment: int, *, previous: Any = None) -> entrypoint.final[str, int]:
    current = (previous or 0) + increment
    # Return a message to the caller, but save the raw number to checkpoint
    return entrypoint.final(value=f"Counter is now {current}", save=current)

config = {"configurable": {"thread_id": "cnt-1"}}
print(counter.invoke(5, config))   # "Counter is now 5"
print(counter.invoke(3, config))   # "Counter is now 8"  — previous=5 restored from checkpoint
print(counter.invoke(2, config))   # "Counter is now 10"
```

### Example 3 — interrupt inside `entrypoint` for human approval

```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@task
def draft_email(topic: str) -> str:
    return f"Dear user, here is info about {topic}. Regards, AI."

@entrypoint(checkpointer=checkpointer)
def email_workflow(topic: str) -> dict:
    draft = draft_email(topic).result()

    # Pause for human review — returns the resume value when workflow resumes
    approved = interrupt({
        "message": "Approve this email?",
        "draft": draft,
    })

    if approved:
        return {"status": "sent", "content": draft}
    return {"status": "cancelled", "content": None}

config = {"configurable": {"thread_id": "email-1"}}

# First run — pauses at interrupt
for event in email_workflow.stream("Python tips", config):
    print(event)   # shows '__interrupt__' event

# Resume with human approval
for event in email_workflow.stream(Command(resume=True), config):
    print(event)   # {'email_workflow': {'status': 'sent', 'content': '...'}}
```

---

## 6 · `task` + `SyncAsyncFuture` — parallel task fan-out

**Module:** `langgraph.func`

`task` is a decorator that turns any callable into a `_TaskFunction`. Calling a task returns a `SyncAsyncFuture` — a `concurrent.futures.Future` subclass that also implements `__await__` for use in async contexts. Tasks are checkpointed: their results are replayed on resume without re-executing the function, making them idempotent through interrupts.

**Key source facts (`langgraph/func/__init__.py`):**

- `task(func)` / `task(name=..., retry_policy=..., cache_policy=..., timeout=...)` — all kwargs are optional.
- Calling `my_task(args)` dispatches `_call_with_options(...)` and returns a `SyncAsyncFuture[T]`.
- `SyncAsyncFuture.__await__` is a generator that yields LangGraph's scheduler sentinel and then returns its result. It supports `await` directly but is **not** compatible with `asyncio.gather()` — `gather()` wraps awaitables in asyncio tasks which fail with `RuntimeError: Task got bad yield` on the sentinel. Await each future individually: `[await f for f in futures]`.
- Tasks can only be called from inside an `entrypoint` or a `StateGraph` node; calling outside raises `RuntimeError`.
- `cache_policy` on a task uses the same `CachePolicy` dataclass as nodes — `key_func` maps `(args, kwargs)` to a cache key string or bytes.
- Timeout on tasks requires Python ≥ 3.11 async; sync tasks cannot be cancelled safely.

### Example 1 — parallel task fan-out with `.result()`

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def score_document(doc: str) -> float:
    # Simulate scoring — runs in parallel via future
    return float(len(doc)) / 100

@entrypoint(checkpointer=InMemorySaver())
def rank_docs(docs: list[str]) -> list[tuple[str, float]]:
    futures = [score_document(d) for d in docs]
    scores  = [f.result() for f in futures]
    ranked  = sorted(zip(docs, scores), key=lambda x: -x[1])
    return ranked

config = {"configurable": {"thread_id": "rank-1"}}
result = rank_docs.invoke(["short", "a much longer document", "medium doc"], config)
for doc, score in result:
    print(f"{score:.2f}  {doc}")
```

### Example 2 — async task fan-out with `asyncio.gather`

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
async def async_fetch(url: str) -> str:
    await asyncio.sleep(0.01)   # simulate I/O
    return f"content from {url}"

@entrypoint(checkpointer=InMemorySaver())
async def crawl(urls: list[str]) -> list[str]:
    # Dispatch all tasks first — they run in parallel immediately
    futures = [async_fetch(u) for u in urls]
    # Collect results by awaiting each future sequentially.
    # NOTE: asyncio.gather() is incompatible with SyncAsyncFuture — it yields
    # LangGraph's scheduler sentinel, not an asyncio-native coroutine.
    results = [await f for f in futures]
    return results

async def main():
    config = {"configurable": {"thread_id": "crawl-1"}}
    pages = await crawl.ainvoke(["http://a.com", "http://b.com", "http://c.com"], config)
    print(pages)

asyncio.run(main())
# ['content from http://a.com', 'content from http://b.com', 'content from http://c.com']
```

### Example 3 — task with `retry_policy` and `name` override

```python
import random
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

attempt_counts: dict[str, int] = {}

@task(
    name="flaky_api_call",
    retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.1, backoff_factor=2.0),
)
def call_flaky_api(endpoint: str) -> str:
    attempt_counts[endpoint] = attempt_counts.get(endpoint, 0) + 1
    if attempt_counts[endpoint] < 3:
        raise ConnectionError(f"Timeout on {endpoint}")
    return f"Success from {endpoint}"

@entrypoint(checkpointer=InMemorySaver())
def resilient_workflow(endpoints: list[str]) -> list[str]:
    futures = [call_flaky_api(ep) for ep in endpoints]
    return [f.result() for f in futures]

config = {"configurable": {"thread_id": "retry-1"}}
results = resilient_workflow.invoke(["/api/data", "/api/meta"], config)
print(results)
print("Attempts:", attempt_counts)
# ['Success from /api/data', 'Success from /api/meta']
# Attempts: {'/api/data': 3, '/api/meta': 3}
```

---

## 7 · `RetryPolicy` + `TimeoutPolicy` — resilience configuration

**Module:** `langgraph.types`

`RetryPolicy` and `TimeoutPolicy` are independent but composable. `RetryPolicy` controls how many times a node or task is retried after failure and with what backoff. `TimeoutPolicy` controls how long a single attempt may run and whether idle time resets the clock.

**Key source facts (`langgraph/types.py`):**

- `RetryPolicy` is a `NamedTuple`. Backoff formula: `min(max_interval, initial_interval * backoff_factor ** (attempt - 1))`. With `jitter=True` a uniform random factor is applied.
- Default `retry_on` = `default_retry_on` (from `langgraph._internal._retry`): retries `ConnectionError`, HTTP 5xx (`httpx.HTTPStatusError` / `requests.HTTPError`), and any other exception **not** in the explicit blocklist (`ValueError`, `TypeError`, `ArithmeticError`, `ImportError`, `LookupError`, `NameError`, `SyntaxError`, `RuntimeError`, `ReferenceError`, `StopIteration`, `StopAsyncIteration`, `OSError`). Use a custom callable when you need different semantics.
- `TimeoutPolicy` is a frozen dataclass with `run_timeout: float | None`, `idle_timeout: float | None`, `refresh_on: Literal["auto", "heartbeat"]`.
- `run_timeout`: hard wall-clock cap, **never** refreshed. If exceeded, `NodeTimeoutError` is raised.
- `idle_timeout`: refreshed on each graph progress signal (with `"auto"`) or only on `runtime.heartbeat()` (with `"heartbeat"`).
- Pass `retry_policy=` and `timeout=` to `StateGraph.add_node(node, retry_policy=..., timeout=...)` or to `@entrypoint(retry_policy=..., timeout=...)`. Note: `retry=` is a deprecated alias for `retry_policy=`.

### Example 1 — `RetryPolicy` on a `StateGraph` node

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

call_count = 0

class State(TypedDict):
    result: str

def unreliable_node(state: State) -> dict:
    global call_count
    call_count += 1
    if call_count < 3:
        # ConnectionError is retried by default_retry_on; RuntimeError is not
        raise ConnectionError(f"Transient network failure #{call_count}")
    return {"result": "success after retries"}

policy = RetryPolicy(
    max_attempts=5,
    initial_interval=0.05,
    backoff_factor=2.0,
    jitter=False,
)

builder = StateGraph(State)
builder.add_node("work", unreliable_node, retry_policy=policy)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile()
result = graph.invoke({"result": ""})
print(result)          # {'result': 'success after retries'}
print(call_count)      # 3
```

### Example 2 — custom `retry_on` callable

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

class State(TypedDict):
    value: int

attempts = 0

class TransientError(Exception):
    pass

class PermanentError(Exception):
    pass

def risky_node(state: State) -> dict:
    global attempts
    attempts += 1
    if attempts == 1:
        raise TransientError("network blip")
    if attempts == 2:
        raise PermanentError("bad data — do not retry")
    return {"value": 42}

# Only retry TransientError, let PermanentError propagate
policy = RetryPolicy(
    max_attempts=3,
    retry_on=lambda e: isinstance(e, TransientError),
)

builder = StateGraph(State)
builder.add_node("risky", risky_node, retry_policy=policy)
builder.add_edge(START, "risky")
builder.add_edge("risky", END)

graph = builder.compile()

try:
    graph.invoke({"value": 0})
except PermanentError as e:
    print("Not retried:", e)   # Not retried: bad data — do not retry
```

### Example 3 — `TimeoutPolicy` with `idle_timeout` and heartbeat

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.runtime import get_runtime

class State(TypedDict):
    result: str

async def long_running_node(state: State) -> dict:
    runtime = get_runtime()
    for i in range(5):
        await asyncio.sleep(0.1)
        runtime.heartbeat()   # refreshes idle_timeout each iteration
    return {"result": "done"}

# idle_timeout=0.5s but heartbeat() keeps refreshing it every 0.1s
policy = TimeoutPolicy(idle_timeout=0.5, refresh_on="heartbeat")

builder = StateGraph(State)
builder.add_node("work", long_running_node, timeout=policy)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile()

async def run():
    result = await graph.ainvoke({"result": ""})
    print(result)   # {'result': 'done'} — heartbeat kept idle_timeout from firing

asyncio.run(run())
```

---

## 8 · `CachePolicy` — node-level result caching

**Module:** `langgraph.types`

`CachePolicy` enables memoisation of node or task outputs. When a node runs, LangGraph hashes the node's input through `key_func` and checks the cache. On a hit, the cached result is replayed without executing the node again. On a miss, the node runs and the result is stored with the optional `ttl`.

**Key source facts (`langgraph/types.py`):**

- `CachePolicy(key_func=default_cache_key, ttl: int | None = None)`.
- `default_cache_key` uses `pickle` hashing of the input — works for most serialisable types.
- Provide a custom `key_func` that receives the same positional and keyword arguments as the node function and must return `str | bytes`.
- `ttl=None` means the cache entry never expires; `ttl=60` expires after 60 seconds.
- Attach to a node: `builder.add_node("n", fn, cache_policy=CachePolicy())`.
- Attach to a task: `@task(cache_policy=CachePolicy(ttl=120))`.
- Clear per-task cache: `my_task.clear_cache(cache_instance)` (async: `await my_task.aclear_cache(...)`).

### Example 1 — caching an expensive node

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache

invocation_count = 0

class State(TypedDict):
    query: str
    answer: str

def expensive_search(state: State) -> dict:
    global invocation_count
    invocation_count += 1
    print(f"  (actual call #{invocation_count})")
    return {"answer": f"Result for: {state['query']}"}

cache = InMemoryCache()

builder = StateGraph(State)
builder.add_node("search", expensive_search, cache_policy=CachePolicy())
builder.add_edge(START, "search")
builder.add_edge("search", END)

graph = builder.compile(checkpointer=InMemorySaver(), cache=cache)

cfg1 = {"configurable": {"thread_id": "t1"}}
cfg2 = {"configurable": {"thread_id": "t2"}}

print("Run 1:", graph.invoke({"query": "langgraph", "answer": ""}, cfg1)["answer"])
print("Run 2:", graph.invoke({"query": "langgraph", "answer": ""}, cfg2)["answer"])
# (actual call #1)
# Run 1: Result for: langgraph
# Run 2: Result for: langgraph   — served from cache, no call #2
```

### Example 2 — custom `key_func` to ignore irrelevant fields

```python
import hashlib, json
from typing import TypedDict
from langgraph.types import CachePolicy

class State(TypedDict):
    query: str
    session_id: str   # differs per user but shouldn't bust the cache
    answer: str

def query_only_key(state: State) -> str:
    # Only include the query in the cache key — session_id is ignored
    payload = json.dumps({"query": state["query"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

policy = CachePolicy(key_func=query_only_key, ttl=300)   # 5 min TTL

# All users with the same query share the same cache entry
s1 = {"query": "LangGraph tutorial", "session_id": "user-A", "answer": ""}
s2 = {"query": "LangGraph tutorial", "session_id": "user-B", "answer": ""}

print(query_only_key(s1) == query_only_key(s2))   # True — same key
```

### Example 3 — per-task `CachePolicy` with TTL

```python
import time
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()
call_log: list[str] = []

@task(cache_policy=CachePolicy(ttl=5))   # cache results for 5 seconds
def fetch_price(symbol: str) -> float:
    call_log.append(symbol)
    return 42.0 if symbol == "BTC" else 1.0

@entrypoint(checkpointer=InMemorySaver(), cache=cache)
def price_workflow(symbols: list[str]) -> dict:
    futures = {s: fetch_price(s) for s in symbols}
    return {s: f.result() for s, f in futures.items()}

cfg = {"configurable": {"thread_id": "price-1"}}
print(price_workflow.invoke(["BTC", "ETH"], cfg))
print(price_workflow.invoke(["BTC", "ETH"], cfg))   # second call hits cache

print("Actual fetches:", call_log)   # each symbol fetched once
```

---

## 9 · `Send` — dynamic map-reduce dispatch

**Module:** `langgraph.types`

`Send` lets a conditional edge function return a **list** of `Send(node, state)` objects, one per parallel invocation. Each `Send` spawns an independent execution of the named node with the provided state. This is the standard pattern for map-reduce workflows in LangGraph.

**Key source facts (`langgraph/types.py`):**

- `Send(node: str, arg: Any, *, timeout: float | timedelta | TimeoutPolicy | None = None)`.
- `arg` is passed as the full state to the target node — it may be a dict, a Pydantic model, or anything the node accepts.
- Results are aggregated via each field's reducer (typically `Annotated[list[T], operator.add]`).
- `timeout` per `Send` overrides the target node's default timeout for that one task.
- A conditional edge function may return a mix of `Send` objects and plain strings — `Send` triggers node calls; strings navigate to existing node names.

### Example 1 — parallel map over a list

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class OverallState(TypedDict):
    items: list[str]
    processed: Annotated[list[str], operator.add]

class WorkerState(TypedDict):
    item: str

def process_item(state: WorkerState) -> dict:
    return {"processed": [state["item"].upper()]}

def fan_out(state: OverallState) -> list[Send]:
    return [Send("process", {"item": x}) for x in state["items"]]

builder = StateGraph(OverallState)
builder.add_node("process", process_item)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({"items": ["hello", "world", "langgraph"], "processed": []})
print(result["processed"])
# ['HELLO', 'WORLD', 'LANGGRAPH']
```

### Example 2 — conditional fan-out with per-`Send` timeout

Per-`Send` timeouts require an **async** node — LangGraph cannot cancel a running synchronous function, so it raises `sync_timeout_unsupported` if the target is sync. Use `async def` and `ainvoke()`.

```python
import asyncio
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, TimeoutPolicy

class State(TypedDict):
    tasks: list[dict]
    results: Annotated[list[str], operator.add]

class TaskState(TypedDict):
    task_id: str
    priority: str

# Must be async — sync nodes cannot be cancelled on timeout
async def run_task(state: TaskState) -> dict:
    return {"results": [f"done:{state['task_id']}"]}

def dispatch(state: State) -> list:
    sends = []
    for t in state["tasks"]:
        # High-priority tasks get a longer timeout budget
        timeout = TimeoutPolicy(run_timeout=30.0 if t["priority"] == "high" else 5.0)
        sends.append(Send("run_task", t, timeout=timeout))
    return sends

builder = StateGraph(State)
builder.add_node("run_task", run_task)
builder.add_conditional_edges(START, dispatch)
builder.add_edge("run_task", END)

graph = builder.compile()

async def main():
    result = await graph.ainvoke({
        "tasks": [
            {"task_id": "t1", "priority": "high"},
            {"task_id": "t2", "priority": "low"},
        ],
        "results": [],
    })
    print(result["results"])
    # ['done:t1', 'done:t2']

asyncio.run(main())
```

### Example 3 — two-phase map-reduce (map → aggregate → reduce)

```python
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class PipelineState(TypedDict):
    chunks: list[str]
    summaries: Annotated[list[str], operator.add]
    final: str

class ChunkState(TypedDict):
    text: str

def summarise_chunk(state: ChunkState) -> dict:
    return {"summaries": [f"Summary({state['text'][:10]})"]}

def aggregate(state: PipelineState) -> dict:
    combined = " | ".join(state["summaries"])
    return {"final": f"[{combined}]"}

def map_chunks(state: PipelineState) -> list:
    return [Send("summarise", {"text": c}) for c in state["chunks"]]

builder = StateGraph(PipelineState)
builder.add_node("summarise", summarise_chunk)
builder.add_node("aggregate", aggregate)
builder.add_conditional_edges(START, map_chunks)
builder.add_edge("summarise", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()
result = graph.invoke({
    "chunks": ["Introduction section", "Methods section", "Results section"],
    "summaries": [],
    "final": "",
})
print(result["final"])
# [Summary(Introducti) | Summary(Methods se) | Summary(Results se)]
```

---

## 10 · `Command` — combined state update + navigation

**Module:** `langgraph.types`

`Command` lets a node return a **single object** that simultaneously updates state, navigates to a specific next node, and optionally resumes a pending interrupt. It replaces the pattern of returning a dict (for state) separately from a conditional edge (for routing).

**Key source facts (`langgraph/types.py`):**

- `Command(update=..., goto=..., resume=..., graph=...)` — all fields are optional.
- `update`: applied to the graph's state exactly like a normal node return dict.
- `goto`: `str | list[str | Send]` — schedules additional routing. `Command(goto=...)` is **additive**: any static `add_edge` for the returning node is still executed alongside `goto`. To make routing fully dynamic, ensure no static `add_edge` exits from that node.
- `resume`: value(s) to resume a pending `interrupt()`. Map of interrupt IDs or single value for the next pending interrupt.
- `graph`: `None` (current graph, default) or `Command.PARENT` (send to the nearest parent graph — used in subgraphs).
- `_update_as_tuples()` supports dicts, `(key, value)` lists, Pydantic models with annotated keys, and a root `"__root__"` key for flat states.

### Example 1 — node that updates state and navigates in one return

```python
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class State(TypedDict):
    step: int
    log: list[str]

def router(state: State) -> Command[Literal["node_a", "node_b"]]:
    if state["step"] % 2 == 0:
        return Command(
            update={"log": state["log"] + ["routed→a"]},
            goto="node_a",
        )
    return Command(
        update={"log": state["log"] + ["routed→b"]},
        goto="node_b",
    )

def node_a(state: State) -> dict:
    return {"step": state["step"] + 1, "log": state["log"] + ["a-done"]}

def node_b(state: State) -> dict:
    return {"step": state["step"] + 1, "log": state["log"] + ["b-done"]}

builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "router")
builder.add_edge("node_a", END)
builder.add_edge("node_b", END)

graph = builder.compile()
print(graph.invoke({"step": 0, "log": []}))
# {'step': 1, 'log': ['routed→a', 'a-done']}
print(graph.invoke({"step": 1, "log": []}))
# {'step': 2, 'log': ['routed→b', 'b-done']}
```

### Example 2 — `Command.PARENT` to signal a parent graph from a subgraph

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class SubState(TypedDict):
    data: str

class ParentState(TypedDict):
    status: str
    data: str

def sub_worker(state: SubState) -> Command[Literal["__end__"]]:
    processed = state["data"].upper()
    # Write "status" to the PARENT graph's state, not this subgraph's state
    return Command(
        update={"status": "complete", "data": processed},
        goto=END,
        graph=Command.PARENT,
    )

sub_builder = StateGraph(SubState)
sub_builder.add_node("worker", sub_worker)
sub_builder.add_edge(START, "worker")
subgraph = sub_builder.compile()

def orchestrator(state: ParentState) -> dict:
    return {"data": state["data"]}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("sub", subgraph)
parent_builder.add_edge(START, "sub")
parent_builder.add_edge("sub", END)

parent = parent_builder.compile()
result = parent.invoke({"status": "pending", "data": "hello"})
print(result)
# {'status': 'complete', 'data': 'HELLO'}
```

### Example 3 — `Command` with `resume` to respond to an interrupt

```python
from langgraph.func import entrypoint
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def approval_workflow(request: str) -> dict:
    decision = interrupt({"request": request, "prompt": "Approve? (yes/no)"})
    if decision == "yes":
        return {"status": "approved", "request": request}
    return {"status": "rejected", "request": request}

config = {"configurable": {"thread_id": "approval-1"}}

# Stream until interrupt fires
events = list(approval_workflow.stream("Deploy to production", config))
print("Interrupted at:", [e for e in events if "__interrupt__" in e])

# Resume with Command(resume=...) — Command.resume sends the value back to interrupt()
events = list(approval_workflow.stream(Command(resume="yes"), config))
for e in events:
    if "approval_workflow" in e:
        print("Result:", e["approval_workflow"])
        # Result: {'status': 'approved', 'request': 'Deploy to production'}
```

---

## Reference — version compatibility

All examples are verified against:

| Package | Version |
|---|---|
| `langgraph` | `1.2.10` |
| `langgraph-checkpoint` | `4.1.1` |
| `langgraph-prebuilt` | `1.1.0` |

### Deprecated in v1.2.10

| Symbol | Replacement |
|---|---|
| `langgraph.prebuilt.HumanInterrupt` | `langchain.agents.interrupt.HumanInterrupt` |
| `langgraph.prebuilt.HumanInterruptConfig` | `langchain.agents.interrupt.HumanInterruptConfig` |
| `langgraph.prebuilt.ActionRequest` | `langchain.agents.interrupt.ActionRequest` |
| `langgraph.prebuilt.ValidationNode` | `create_agent` from `langchain.agents` with custom tool error handling |
| `@entrypoint(config_schema=...)` | `@entrypoint(context_schema=...)` |
| `@task(retry=...)` | `@task(retry_policy=...)` |

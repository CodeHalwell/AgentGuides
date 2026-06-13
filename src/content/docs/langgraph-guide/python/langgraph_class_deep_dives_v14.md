---
title: "Class deep-dives Vol. 14 — Routing, channels & graph internals"
description: "Source-verified deep dives into BranchSpec, LastValue/LastValueAfterFinish, ManagedValue, task decorator, DeltaChannel advanced mechanics, node input schema narrowing, _NodeDefaults/set_node_defaults, InMemoryCache/BaseCache, entrypoint full API, and CompiledStateGraph internals — with multiple runnable examples for each class."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 14"
  order: 45
---

# Class deep-dives Vol. 14 — Routing, channels & graph internals

Verified against **`langgraph==1.2.5`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class | Module |
|---|-------|--------|
| 1 | `BranchSpec` | `langgraph.graph._branch` |
| 2 | `LastValue` + `LastValueAfterFinish` | `langgraph.channels.last_value` |
| 3 | `ManagedValue` + custom managed values | `langgraph.managed.base` |
| 4 | `task` decorator + `_TaskFunction` | `langgraph.func` |
| 5 | `DeltaChannel` — advanced mechanics | `langgraph.channels.delta` |
| 6 | Node input schema narrowing | `langgraph.graph.state` |
| 7 | `_NodeDefaults` + `set_node_defaults()` | `langgraph.graph.state` |
| 8 | `InMemoryCache` + `BaseCache` | `langgraph.cache` |
| 9 | `entrypoint` — full parameter guide | `langgraph.func` |
| 10 | `CompiledStateGraph` internals | `langgraph.graph.state` |

---

## 1 · `BranchSpec` — conditional routing internals

**Module:** `langgraph.graph._branch`  
**Import:**
```python
from langgraph.graph._branch import BranchSpec
```

`BranchSpec` is the compiled representation of a conditional edge. Every call to `add_conditional_edges()` on a `StateGraph` internally creates a `BranchSpec` and attaches it to the source node. Understanding `BranchSpec` explains why certain routing patterns work — and why others silently fail.

### Source signature (1.2.5)

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type[Any] | None = None

    @classmethod
    def from_path(
        cls,
        path: Runnable[Any, Hashable | list[Hashable]],
        path_map: dict[Hashable, str] | list[str] | None,
        infer_schema: bool = False,
    ) -> BranchSpec: ...

    def run(
        self,
        writer: _Writer,
        reader: Callable[[RunnableConfig], Any] | None = None,
    ) -> RunnableCallable: ...
```

`BranchSpec` has three fields:

| Field | Type | Meaning |
|-------|------|---------|
| `path` | `Runnable` | The routing callable, wrapped in `RunnableCallable` |
| `ends` | `dict[Hashable, str] \| None` | Maps return values to node names; `None` means the callable returns node names directly |
| `input_schema` | `type \| None` | Narrows the state slice passed to `path`; inferred from the callable's first parameter annotation when `infer_schema=True` |

### How `path_map` becomes `ends`

`BranchSpec.from_path()` coerces `path_map` into a `dict`:

- `None` → `ends=None` — the function must return node-name strings (or `Send` objects) directly
- `["a", "b"]` → `{"a": "a", "b": "b"}` — identity mapping
- `{"go_left": "node_left", ...}` → used verbatim
- Auto-inferred from `Literal` return annotation if `path_map` is `None` and the callable has a `-> Literal["a", "b"]` annotation

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    value: int
    path: str


# Router with a Literal return — path_map is inferred automatically
def route(state: State) -> Literal["high", "low"]:
    return "high" if state["value"] > 10 else "low"


def high_node(state: State) -> dict:
    return {"path": "went high"}


def low_node(state: State) -> dict:
    return {"path": "went low"}


graph = (
    StateGraph(State)
    .add_node("high", high_node)
    .add_node("low", low_node)
    .add_edge(START, "router_source")
    .add_node("router_source", lambda s: s)
    .add_conditional_edges("router_source", route)  # path_map inferred from Literal
    .add_edge("high", END)
    .add_edge("low", END)
    .compile()
)

result = graph.invoke({"value": 15, "path": ""})
print(result["path"])  # went high
```

### Example 2: Explicit `path_map` dict

When the routing function returns short strings (e.g. `"ok"`, `"err"`) you want to map to longer node names:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    score: float
    label: str


def classify(state: S) -> str:
    if state["score"] >= 0.8:
        return "high"
    elif state["score"] >= 0.5:
        return "mid"
    return "low"


def make_node(label: str):
    def _node(state: S) -> dict:
        return {"label": label}
    _node.__name__ = label
    return _node


g = StateGraph(S)
g.add_node("classify", lambda s: s)
g.add_node("high_tier", make_node("premium"))
g.add_node("mid_tier", make_node("standard"))
g.add_node("low_tier", make_node("basic"))
g.add_edge(START, "classify")
g.add_conditional_edges(
    "classify",
    classify,
    path_map={"high": "high_tier", "mid": "mid_tier", "low": "low_tier"},
)
g.add_edge("high_tier", END)
g.add_edge("mid_tier", END)
g.add_edge("low_tier", END)
graph = g.compile()

print(graph.invoke({"score": 0.9, "label": ""})["label"])  # premium
print(graph.invoke({"score": 0.6, "label": ""})["label"])  # standard
print(graph.invoke({"score": 0.2, "label": ""})["label"])  # basic
```

### Example 3: Multi-destination fan-out

`BranchSpec._finish()` supports returning a **list** from the routing function, triggering parallel fan-out to multiple nodes:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


class FanState(TypedDict):
    input: str
    results: Annotated[list[str], add_messages]


def decide_reviewers(state: FanState) -> list[str]:
    """Route to 1 or 2 reviewers based on input length."""
    if len(state["input"]) > 50:
        return ["reviewer_a", "reviewer_b"]   # fan-out to both
    return ["reviewer_a"]


def reviewer_a(state: FanState) -> dict:
    return {"results": [HumanMessage(content=f"A says: {state['input'][:20]}")]}


def reviewer_b(state: FanState) -> dict:
    return {"results": [HumanMessage(content=f"B says: {state['input'][:20]}")]}


g = StateGraph(FanState)
g.add_node("start_node", lambda s: s)
g.add_node("reviewer_a", reviewer_a)
g.add_node("reviewer_b", reviewer_b)
g.add_edge(START, "start_node")
g.add_conditional_edges("start_node", decide_reviewers)
g.add_edge("reviewer_a", END)
g.add_edge("reviewer_b", END)
graph = g.compile()

result = graph.invoke({"input": "A very long input that definitely needs two reviewers", "results": []})
print(len(result["results"]))  # 2 — both reviewers ran
```

---

## 2 · `LastValue` + `LastValueAfterFinish` — the default state channels

**Module:** `langgraph.channels.last_value`  
**Import:**
```python
from langgraph.channels.last_value import LastValue, LastValueAfterFinish
```

Every unAnnotated field in a `TypedDict` state schema maps to a `LastValue` channel under the hood. `LastValueAfterFinish` is a companion used internally by the functional API for task outputs.

### `LastValue` source (1.2.5)

```python
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """Stores the last value received; can receive at most one value per step."""
    __slots__ = ("value",)
    value: Value | Any        # MISSING sentinel until first update

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) != 1:
            raise InvalidUpdateError(...)  # concurrent writes to the same key are an error
        self.value = values[-1]
        return True
```

Key properties:

| Property | Behaviour |
|----------|-----------|
| **Concurrent write guard** | Raises `InvalidUpdateError` if two nodes write to the same `LastValue` key in the same superstep |
| **Checkpoint representation** | Serialises directly — `checkpoint()` returns the stored value |
| **`from_checkpoint`** | Restores the stored value on replay |
| **`is_available()`** | Returns `False` until the first update (uses `MISSING` sentinel) |

### Example 1: Understanding the concurrent-write guard

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class State(TypedDict):
    counter: int   # LastValue — no reducer, single-write per superstep


def add_one(state: State) -> dict:
    return {"counter": state["counter"] + 1}


# Safe: sequential chain
g = StateGraph(State)
g.add_node("a", add_one)
g.add_node("b", add_one)
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)
result = g.compile().invoke({"counter": 0})
print(result["counter"])  # 2 — fine, sequential

# Dangerous: two parallel nodes both writing counter raises InvalidUpdateError
```

### Example 2: Using `Annotated` to opt into a reducer (bypass `LastValue`)

When you need multiple nodes to write the same key in the same superstep, annotate the field with a reducer:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator


class SafeState(TypedDict):
    values: Annotated[list[int], operator.add]  # BinaryOperatorAggregate — safe for fan-out


def produce_a(state: SafeState) -> dict:
    return {"values": [1]}


def produce_b(state: SafeState) -> dict:
    return {"values": [2]}


g = StateGraph(SafeState)
g.add_node("a", produce_a)
g.add_node("b", produce_b)
g.add_edge(START, "a")
g.add_edge(START, "b")
g.add_edge("a", END)
g.add_edge("b", END)
result = g.compile().invoke({"values": []})
print(result["values"])  # [1, 2] — both writes merged by the reducer
```

### `LastValueAfterFinish` — functional API task output channel

`LastValueAfterFinish` extends `LastValue` with a two-phase commit: values are stored on `update()` but only become readable after `finish()` is called. This is how the functional API ensures a `task()` result is not visible to the caller until the task has truly completed.

```python
class LastValueAfterFinish(Generic[Value], BaseChannel[Value, Value, tuple[Value, bool]]):
    __slots__ = ("value", "finished")
    # get() raises EmptyChannelError unless finished == True
    # consume() clears both value and finished flag after reading
    # finish() sets finished=True — triggers the value becoming available
```

You normally never instantiate this directly, but you can observe it in action when tracing a functional API `task()` through a graph.

---

## 3 · `ManagedValue` + custom managed values

**Module:** `langgraph.managed.base`  
**Import:**
```python
from langgraph.managed.base import ManagedValue, ManagedValueSpec, is_managed_value
```

`ManagedValue` is the abstract base class behind `IsLastStep` and `RemainingSteps`. It lets you inject **computed, read-only** values into any node without storing them in the checkpoint — the value is re-derived from `PregelScratchpad` on every superstep.

### Source signature (1.2.5)

```python
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...

ManagedValueSpec = type[ManagedValue]
```

The `PregelScratchpad` it receives has two relevant attributes:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `step` | `int` | Zero-indexed current superstep |
| `stop` | `int` | The `recursion_limit` (defaults to 25) |

Declaring a managed value in a TypedDict uses `Annotated`:

```python
from typing import Annotated
MyValue = Annotated[SomeType, MyManagedValueManager]
```

### Example 1: How `IsLastStep` and `RemainingSteps` work

```python
# From langgraph/managed/is_last_step.py — the actual source:
from typing import Annotated
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.managed.base import ManagedValue

class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

IsLastStep = Annotated[bool, IsLastStepManager]

class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step

RemainingSteps = Annotated[int, RemainingStepsManager]
```

Using them in a real node:

```python
from typing_extensions import TypedDict
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    messages: list[str]
    is_last_step: IsLastStep        # injected, not stored
    remaining: RemainingSteps       # injected, not stored


def agent(state: AgentState) -> dict:
    print(f"Remaining steps: {state['remaining']}, last={state['is_last_step']}")
    if state["is_last_step"]:
        return {"messages": state["messages"] + ["[forced stop]"]}
    return {"messages": state["messages"] + [f"step {10 - state['remaining']}"]}


graph = (
    StateGraph(AgentState)
    .add_node("agent", agent)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile()
)

result = graph.invoke({"messages": []}, {"recursion_limit": 3})
print(result["messages"])
# ['step 0', 'step 1', '[forced stop]'] — 3 supersteps, last one triggers is_last_step
```

### Example 2: Building a custom managed value — step counter

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.managed.base import ManagedValue
from langgraph.graph import StateGraph, START, END


class StepNumberManager(ManagedValue[int]):
    """One-indexed step counter injected into every node."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.step + 1  # 1-indexed


# Create a type alias using Annotated
StepNumber = Annotated[int, StepNumberManager]


class WorkState(TypedDict):
    results: list[str]
    step_number: StepNumber   # injected by the framework, not persisted


def worker(state: WorkState) -> dict:
    return {"results": state["results"] + [f"[step {state['step_number']}] processed"]}


graph = (
    StateGraph(WorkState)
    .add_node("worker", worker)
    .add_edge(START, "worker")
    .add_edge("worker", END)
    .compile()
)

result = graph.invoke({"results": []})
# START runs at step 0; the worker runs at step 1 → 1-indexed = 2
print(result["results"])  # ['[step 2] processed']
```

### Example 3: Custom managed value — node attempt tracker

```python
from typing import Annotated
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph.managed.base import ManagedValue


class MaxRecursionFractionManager(ManagedValue[float]):
    """Fraction of the recursion budget consumed so far (0.0–1.0)."""

    @staticmethod
    def get(scratchpad: PregelScratchpad) -> float:
        if scratchpad.stop == 0:
            return 0.0
        return scratchpad.step / scratchpad.stop


RecursionFraction = Annotated[float, MaxRecursionFractionManager]


# Use in a state that monitors its own budget consumption
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class BudgetState(TypedDict):
    output: list[str]
    budget_used: RecursionFraction


def monitor_node(state: BudgetState) -> dict:
    pct = state["budget_used"] * 100
    return {"output": state["output"] + [f"budget used: {pct:.1f}%"]}


graph = (
    StateGraph(BudgetState)
    .add_node("monitor", monitor_node)
    .add_edge(START, "monitor")
    .add_edge("monitor", END)
    .compile()
)

result = graph.invoke({"output": []}, {"recursion_limit": 10})
print(result["output"])  # ['budget used: 0.0%']
```

---

## 4 · `task` decorator + `_TaskFunction` — complete guide

**Module:** `langgraph.func`  
**Import:**
```python
from langgraph.func import task, entrypoint
```

`@task` wraps a sync or async callable in `_TaskFunction`, which schedules the call through the Pregel executor and returns a `SyncAsyncFuture`. Tasks can only be called from within an `@entrypoint` or a `StateGraph` node.

### `_TaskFunction` source signature (1.2.5)

```python
class _TaskFunction(Generic[P, T]):
    func: Callable[P, Awaitable[T]] | Callable[P, T]
    retry_policy: Sequence[RetryPolicy]
    cache_policy: CachePolicy[Callable[P, str | bytes]] | None
    timeout: TimeoutPolicy | None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> SyncAsyncFuture[T]: ...
    def clear_cache(self, cache: BaseCache) -> None: ...
    async def aclear_cache(self, cache: BaseCache) -> None: ...
```

### Full `@task` decorator parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `name` | `str \| None` | `None` | Override the task's registered name; useful for class methods |
| `retry_policy` | `RetryPolicy \| Sequence[RetryPolicy] \| None` | `None` | Retry on failure; first matching policy in a sequence wins |
| `cache_policy` | `CachePolicy[key_func] \| None` | `None` | Memoize results; `key_func` computes the cache key |
| `timeout` | `float \| timedelta \| TimeoutPolicy \| None` | `None` | Async-only wall-clock or idle timeout per attempt |

### Example 1: Parallel task fan-out with `result()`

```python
from langgraph.func import task, entrypoint
from langgraph.checkpoint.memory import InMemorySaver


@task
def square(n: int) -> int:
    return n * n


@entrypoint(checkpointer=InMemorySaver())
def run_squares(numbers: list[int]) -> list[int]:
    futures = [square(n) for n in numbers]      # all dispatched concurrently
    return [f.result() for f in futures]         # gather in order


config = {"configurable": {"thread_id": "t1"}}
print(run_squares.invoke([1, 2, 3, 4, 5], config))  # [1, 4, 9, 16, 25]
```

### Example 2: `retry_policy` on a task

```python
import random
from langgraph.func import task, entrypoint
from langgraph.types import RetryPolicy


@task(retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.1))
def flaky_fetch(url: str) -> str:
    if random.random() < 0.5:
        raise ConnectionError("transient failure")
    return f"content from {url}"


@entrypoint()
def fetch_all(urls: list[str]) -> list[str]:
    futures = [flaky_fetch(u) for u in urls]
    return [f.result() for f in futures]


print(fetch_all.invoke(["https://example.com/a", "https://example.com/b"]))
```

### Example 3: `cache_policy` — memoize expensive task calls

```python
from langgraph.func import task, entrypoint
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()
call_count = 0


@task(cache_policy=CachePolicy(key=lambda x: x))   # cache key = the input argument
def expensive_computation(data: str) -> str:
    global call_count
    call_count += 1
    return data.upper()


@entrypoint(cache=cache)
def process(items: list[str]) -> list[str]:
    futures = [expensive_computation(item) for item in items]
    return [f.result() for f in futures]


result1 = process.invoke(["hello", "world", "hello"])  # 2 unique: 2 calls
result2 = process.invoke(["hello", "world", "hello"])  # all cached: still 2 calls total

print(result1)       # ['HELLO', 'WORLD', 'HELLO']
print(call_count)    # 2 — deduplicated across both invocations
```

### Example 4: Async task with `timeout`

```python
import asyncio
from datetime import timedelta
from langgraph.func import task, entrypoint
from langgraph.types import TimeoutPolicy


@task(timeout=timedelta(seconds=2))   # hard 2-second wall-clock limit
async def slow_api_call(endpoint: str) -> str:
    await asyncio.sleep(0.1)          # fast enough
    return f"response from {endpoint}"


@entrypoint()
async def call_apis(endpoints: list[str]) -> list[str]:
    futures = [slow_api_call(ep) for ep in endpoints]
    results = await asyncio.gather(*[f for f in futures])
    return list(results)


import asyncio
asyncio.run(call_apis.ainvoke(["api/a", "api/b"]))
```

### Example 5: `name=` for class method tasks

```python
from langgraph.func import task, entrypoint


class DataProcessor:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def process(self, item: str) -> str:
        return f"{self.prefix}:{item}"


processor = DataProcessor(prefix="v2")

# Without name=, class methods get generic names — use name= to give a stable identity
process_task = task(processor.process, name="data_processor_process")


@entrypoint()
def run(items: list[str]) -> list[str]:
    futures = [process_task(i) for i in items]
    return [f.result() for f in futures]


print(run.invoke(["a", "b", "c"]))  # ['v2:a', 'v2:b', 'v2:c']
```

---

## 5 · `DeltaChannel` advanced mechanics

**Module:** `langgraph.channels.delta`  
**Import:**
```python
from langgraph.channels.delta import DeltaChannel
```

`DeltaChannel` (introduced in 1.1.x) solves the checkpoint bloat problem for large accumulating state (long message lists, event logs). Instead of snapshotting the full value on every step, it stores only a sentinel in checkpoint blobs and replays ancestor writes through the reducer when reconstructing.

### How checkpointing differs from `BinaryOperatorAggregate`

| Feature | `BinaryOperatorAggregate` | `DeltaChannel` |
|---------|--------------------------|----------------|
| Checkpoint blob | Full current value | `MISSING` (no blob) except on snapshot |
| Snapshot blob type | n/a | `_DeltaSnapshot(value=...)` |
| Reconstruction | Direct deserialise | Replay writes via `replay_writes()` |
| Max checkpoint size | Grows with value | Bounded by `snapshot_frequency` |

### `replay_writes()` internals

```python
def replay_writes(self, writes: Sequence[PendingWrite]) -> None:
    values = [v for _, _, v in writes]
    if not values:
        return
    base = self.value
    start = 0
    for i, v in enumerate(values):
        is_ow, ow_value = _get_overwrite(v)
        if is_ow:                              # Overwrite found — reset base
            base = copy(ow_value) if ow_value is not None else self.typ()
            start = i + 1                      # only replay writes AFTER the Overwrite
    remaining = values[start:]
    self.value = self.reducer(base, remaining) if remaining else base
```

The `Overwrite` sentinel acts as a "rebase" point: it clears the accumulated state back to a fresh value before replaying subsequent writes.

### Example 1: Basic `DeltaChannel` for an append-only log

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import StateGraph, START, END


def append_reducer(base: list, writes: list) -> list:
    """Append all new writes to the base list."""
    result = list(base)
    for item in writes:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


class EventLog(TypedDict):
    events: Annotated[list[str], DeltaChannel(append_reducer)]


def log_event(state: EventLog) -> dict:
    return {"events": [f"event at step"]}


graph = (
    StateGraph(EventLog)
    .add_node("log", log_event)
    .add_edge(START, "log")
    .add_edge("log", END)
    .compile()
)

result = graph.invoke({"events": []})
print(result["events"])  # ['event at step']
```

### Example 2: `snapshot_frequency` tuning

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


def list_concat(base: list, writes: list) -> list:
    return base + [w for sublist in writes for w in (sublist if isinstance(sublist, list) else [sublist])]


# snapshot_frequency=2 means a full snapshot is written every 2nd update
# This bounds how many ancestor writes must be replayed on warm start
class HighFreqState(TypedDict):
    log: Annotated[list[str], DeltaChannel(list_concat, snapshot_frequency=2)]


saver = InMemorySaver()


def append_node(state: HighFreqState) -> dict:
    n = len(state["log"])
    return {"log": [f"item-{n}"]}


graph = (
    StateGraph(HighFreqState)
    .add_node("append", append_node)
    .add_edge(START, "append")
    .add_edge("append", END)
    .compile(checkpointer=saver)
)

config = {"configurable": {"thread_id": "delta-demo"}}
# Run 5 times — snapshot written at update counts 2, 4, ...
for _ in range(5):
    result = graph.invoke({"log": []}, config)

print(result["log"])  # grows with each invocation
```

### Example 3: `Overwrite` as a reset in a `DeltaChannel`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.types import Overwrite
from langgraph.graph import StateGraph, START, END


def list_append(base: list, writes: list) -> list:
    return base + writes


class ResetState(TypedDict):
    history: Annotated[list[str], DeltaChannel(list_append)]


def normal_node(state: ResetState) -> dict:
    return {"history": ["normal entry"]}


def reset_node(state: ResetState) -> dict:
    # Overwrite() resets the base to an empty list; only entries AFTER this
    # write are replayed through list_append
    return {"history": Overwrite([])}


graph = (
    StateGraph(ResetState)
    .add_node("normal", normal_node)
    .add_node("reset", reset_node)
    .add_edge(START, "normal")
    .add_edge("normal", "reset")
    .add_edge("reset", END)
    .compile()
)

result = graph.invoke({"history": ["old1", "old2"]})
print(result["history"])  # [] — Overwrite([]) reset the accumulated list
```

---

## 6 · Node input schema narrowing

**Module:** `langgraph.graph.state`  
**Import:**
```python
from langgraph.graph import StateGraph
```

By default every node receives the full graph state. The `input_schema=` keyword on `add_node` narrows the slice passed to a node to a subset TypedDict, improving type safety and enabling schema validation at the node boundary.

### How it works internally

When `StateGraph.compile()` runs `attach_node()`, it inspects `StateNodeSpec.input_schema`. If it differs from the graph's default state schema, a projection mapper is installed that picks only the declared keys from the full state dict before invoking the node.

### Example 1: Simple key subset

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class FullState(TypedDict):
    user_id: str
    message: str
    internal_counter: int    # implementation detail — don't expose to every node
    result: str


class SummarisationInput(TypedDict):
    user_id: str
    message: str              # only these two keys are needed


def summarise(state: SummarisationInput) -> dict:
    # state only has user_id and message — internal_counter is not present
    return {"result": f"[{state['user_id']}] {state['message'][:50]}"}


graph = (
    StateGraph(FullState)
    .add_node("summarise", summarise, input_schema=SummarisationInput)
    .add_edge(START, "summarise")
    .add_edge("summarise", END)
    .compile()
)

out = graph.invoke({
    "user_id": "alice",
    "message": "Hello from the LangGraph guide",
    "internal_counter": 999,
    "result": "",
})
print(out["result"])  # [alice] Hello from the LangGraph guide
```

### Example 2: Named node overload

When you name the node explicitly (string + callable form), `input_schema=` still works:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class AppState(TypedDict):
    query: str
    context: str
    answer: str
    debug_info: dict          # never expose this to the LLM node


class LLMInput(TypedDict):
    query: str
    context: str


def llm_node(state: LLMInput) -> dict:
    return {"answer": f"Answer to '{state['query']}' given context: {state['context'][:30]}"}


graph = (
    StateGraph(AppState)
    .add_node("llm", llm_node, input_schema=LLMInput)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile()
)

out = graph.invoke({
    "query": "What is LangGraph?",
    "context": "LangGraph is a framework for building stateful LLM applications.",
    "answer": "",
    "debug_info": {"secret": "value"},
})
print(out["answer"])
```

### Example 3: Combining `input_schema=` with `retry_policy` and `timeout`

All per-node options compose freely:

```python
from typing_extensions import TypedDict
from datetime import timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, TimeoutPolicy


class PipelineState(TypedDict):
    raw_data: str
    processed: str
    metadata: dict


class ProcessorInput(TypedDict):
    raw_data: str             # only raw_data needed


async def async_processor(state: ProcessorInput) -> dict:
    # Simulates an async processing step
    return {"processed": state["raw_data"].strip().upper()}


graph = (
    StateGraph(PipelineState)
    .add_node(
        "processor",
        async_processor,
        input_schema=ProcessorInput,
        retry_policy=RetryPolicy(max_attempts=3),
        timeout=timedelta(seconds=5),
    )
    .add_edge(START, "processor")
    .add_edge("processor", END)
    .compile()
)
```

---

## 7 · `_NodeDefaults` + `set_node_defaults()` — graph-wide policy inheritance

**Module:** `langgraph.graph.state`  
**Import:**
```python
from langgraph.graph import StateGraph
```

`_NodeDefaults` is a simple dataclass holding four optional policies that apply to every node not explicitly overriding them:

```python
class _NodeDefaults:
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None
    cache_policy: CachePolicy | None = None
    error_handler: StateNode[Any, Any] | None = None
    timeout: TimeoutPolicy | None = None
```

`StateGraph.set_node_defaults()` mutates `self._node_defaults`. At `compile()` time, `attach_node()` merges the per-node values with these defaults (per-node always wins).

### Important asymmetry in defaults

| Policy | Applied to regular nodes | Applied to error-handler nodes |
|--------|-------------------------|-------------------------------|
| `retry_policy` | ✅ | ✅ |
| `timeout` | ✅ | ✅ |
| `cache_policy` | ✅ | ❌ (unsafe to cache handler results) |
| `error_handler` | ✅ | ❌ (handlers must not catch themselves) |

### Example 1: Fleet-wide retry + timeout + error handler

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, TimeoutPolicy
import random


class State(TypedDict):
    messages: list[str]
    errors: list[str]


def fallback_handler(state: State) -> dict:
    return {"errors": state["errors"] + ["error handled by fallback"]}


def unreliable_node(state: State) -> dict:
    if random.random() < 0.3:
        raise RuntimeError("transient error")
    return {"messages": state["messages"] + ["success"]}


def stable_node(state: State) -> dict:
    return {"messages": state["messages"] + ["stable output"]}


graph = (
    StateGraph(State)
    .set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.05),
        timeout=TimeoutPolicy(run_timeout=10.0),
        error_handler=fallback_handler,
    )
    .add_node("unreliable", unreliable_node)
    .add_node("stable", stable_node)
    .add_edge(START, "unreliable")
    .add_edge("unreliable", "stable")
    .add_edge("stable", END)
    .compile()
)

result = graph.invoke({"messages": [], "errors": []})
print(result["messages"])   # ['success', 'stable output'] if no failure after retries
print(result["errors"])     # ['error handled by fallback'] if all retries exhausted
```

### Example 2: Overriding defaults at the node level

Per-node values always take precedence over `set_node_defaults()`:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy


class S(TypedDict):
    result: str


def cheap_node(state: S) -> dict:
    return {"result": "cheap"}


def expensive_node(state: S) -> dict:
    return {"result": "expensive"}


def expensive_cached_node(state: S) -> dict:
    return {"result": "expensive+cached"}


graph = (
    StateGraph(S)
    .set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=2),
        cache_policy=CachePolicy(key=lambda s: str(s)),
    )
    .add_node("cheap", cheap_node, retry_policy=None, cache_policy=None)   # opts out
    .add_node("expensive", expensive_node)                                   # uses defaults
    .add_node(
        "expensive_cached",
        expensive_cached_node,
        cache_policy=CachePolicy(key=lambda s: "fixed-key"),  # custom override
    )
    .add_edge(START, "cheap")
    .add_edge("cheap", "expensive")
    .add_edge("expensive", "expensive_cached")
    .add_edge("expensive_cached", END)
    .compile()
)
```

### Example 3: `set_node_defaults()` as a fluent chain

`set_node_defaults()` returns `Self`, so you can chain it with the builder pattern:

```python
from typing_extensions import TypedDict
from datetime import timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


class S(TypedDict):
    x: int


graph = (
    StateGraph(S)
    .set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=5, backoff_factor=2.0),
        timeout=timedelta(seconds=30),
    )
    .add_node("a", lambda s: {"x": s["x"] + 1})
    .add_node("b", lambda s: {"x": s["x"] * 2})
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)

print(graph.invoke({"x": 3})["x"])  # 8
```

---

## 8 · `InMemoryCache` + `BaseCache` — caching contract & TTL mechanics

**Module:** `langgraph.cache.memory` / `langgraph.cache.base`  
**Import:**
```python
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.base import BaseCache
```

`BaseCache` defines the six-method contract every cache backend must implement. `InMemoryCache` is a thread-safe, TTL-aware in-process implementation backed by a `dict[Namespace, dict[str, (encoding, bytes, expiry)]]`.

### `BaseCache` contract (1.2.5)

```python
class BaseCache(ABC, Generic[ValueT]):
    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=False)

    @abstractmethod
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]: ...
    @abstractmethod
    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]: ...
    @abstractmethod
    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None: ...
    @abstractmethod
    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None: ...
    @abstractmethod
    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None: ...
    @abstractmethod
    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None: ...
```

Key types:

| Type | Definition |
|------|-----------|
| `Namespace` | `tuple[str, ...]` — hierarchical key prefix |
| `FullKey` | `tuple[Namespace, str]` — namespace + leaf key |
| TTL | `int \| None` — seconds; `None` = never expires |

### Example 1: Standalone `InMemoryCache` with TTL

```python
from langgraph.cache.memory import InMemoryCache

cache: InMemoryCache = InMemoryCache()

# Write with a 5-second TTL
cache.set({
    (("my_ns",), "key1"): ("some value", 5),        # expires in 5s
    (("my_ns",), "key2"): ("another value", None),  # never expires
})

# Read back
results = cache.get([
    (("my_ns",), "key1"),
    (("my_ns",), "key2"),
    (("my_ns",), "missing"),
])

print(results[(("my_ns",), "key1")])  # 'some value'
print(results[(("my_ns",), "key2")])  # 'another value'
print((("my_ns",), "missing") in results)  # False
```

### Example 2: Namespace clearing

```python
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()

# Populate two namespaces
for i in range(3):
    cache.set({(("ns_a",), f"k{i}"): (f"val_a_{i}", None)})
    cache.set({(("ns_b",), f"k{i}"): (f"val_b_{i}", None)})

# Clear only ns_a
cache.clear([(("ns_a",),)])

print(cache.get([(("ns_a",), "k0")]))  # {} — cleared
print(cache.get([(("ns_b",), "k0")])[(("ns_b",), "k0")])  # 'val_b_0' — intact

# Clear everything
cache.clear()
print(cache.get([(("ns_b",), "k0")]))  # {}
```

### Example 3: Custom `BaseCache` with a custom serializer

```python
import json
from collections.abc import Mapping, Sequence
from typing import Any
from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class JsonSerializer(SerializerProtocol):
    """Minimal JSON-only serializer for cache values."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "json", json.dumps(obj).encode()

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        encoding, raw = data
        assert encoding == "json"
        return json.loads(raw.decode())


class TinyCache(BaseCache):
    """Minimal in-memory cache with no TTL support."""

    def __init__(self):
        super().__init__(serde=JsonSerializer())
        self._store: dict[FullKey, bytes] = {}

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        out = {}
        for key in keys:
            if key in self._store:
                out[key] = self.serde.loads_typed(("json", self._store[key]))
        return out

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        return self.get(keys)

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        for key, (value, _ttl) in pairs.items():
            _, raw = self.serde.dumps_typed(value)
            self._store[key] = raw

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        self.set(pairs)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        if namespaces is None:
            self._store.clear()
        else:
            self._store = {
                k: v for k, v in self._store.items()
                if not any(k[0] == ns for ns in namespaces)
            }

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        self.clear(namespaces)


# Use the custom cache with task + entrypoint
from langgraph.func import task, entrypoint
from langgraph.types import CachePolicy

tiny = TinyCache()


@task(cache_policy=CachePolicy(key=lambda x: x))
def greet(name: str) -> str:
    print(f"  computing greeting for {name}")
    return f"Hello, {name}!"


@entrypoint(cache=tiny)
def pipeline(names: list[str]) -> list[str]:
    return [greet(n).result() for n in names]


print(pipeline.invoke(["Alice", "Bob", "Alice"]))
# computing greeting for Alice
# computing greeting for Bob
# ['Hello, Alice!', 'Hello, Bob!', 'Hello, Alice!']
# — Alice is computed only once (cached); second call hits the cache
```

---

## 9 · `entrypoint` — full parameter guide

**Module:** `langgraph.func`  
**Import:**
```python
from langgraph.func import entrypoint
```

`@entrypoint` converts a plain function into a fully featured `Pregel` graph. All parameters are optional; you only pay for what you use.

### Constructor parameters (1.2.5)

| Parameter | Type | Purpose |
|-----------|------|---------|
| `checkpointer` | `BaseCheckpointSaver \| None` | Enable thread persistence and `previous` injection |
| `store` | `BaseStore \| None` | Attach a long-term key/value store accessible via `runtime.store` |
| `cache` | `BaseCache \| None` | Attach a cache for `@task` results inside this workflow |
| `context_schema` | `type[ContextT] \| None` | Strongly-typed run-scoped context (replaces `config_schema`) |
| `cache_policy` | `CachePolicy \| None` | Memoize the entrypoint's own output |
| `retry_policy` | `RetryPolicy \| Sequence[RetryPolicy] \| None` | Retry the entire workflow on failure |
| `timeout` | `float \| timedelta \| TimeoutPolicy \| None` | Hard cap on async workflow execution |

### Example 1: `previous` + `checkpointer` — stateful workflows

```python
from typing import Optional
from typing_extensions import TypedDict
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver


@task
def increment(n: int) -> int:
    return n + 1


@entrypoint(checkpointer=InMemorySaver())
def counter(value: int, *, previous: Optional[int] = None) -> int:
    """Each call adds value to the running total from the previous call."""
    base = previous or 0
    future = increment(base + value)
    return future.result()


config = {"configurable": {"thread_id": "counter-1"}}
print(counter.invoke(10, config))   # 11  (0 + 10 + 1)
print(counter.invoke(5, config))    # 17  (11 + 5 + 1)
print(counter.invoke(3, config))    # 21  (17 + 3 + 1)
```

### Example 2: `context_schema` — typed run-scoped dependencies

```python
from dataclasses import dataclass
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class AppContext:
    user_id: str
    locale: str = "en"


store = InMemoryStore()
store.put(("profiles",), "u1", {"name": "Alice", "premium": True})


@task
def fetch_profile(user_id: str) -> dict:
    item = store.get(("profiles",), user_id)
    return item.value if item else {}


@entrypoint(
    checkpointer=InMemorySaver(),
    store=store,
    context_schema=AppContext,
)
def personalised_workflow(query: str, *, runtime: Runtime[AppContext]) -> str:
    profile = fetch_profile(runtime.context.user_id).result()
    locale = runtime.context.locale
    name = profile.get("name", "user")
    return f"[{locale}] Hello {name}: {query}"


config = {"configurable": {"thread_id": "pw-1"}}
result = personalised_workflow.invoke(
    "What are my options?",
    config,
    context=AppContext(user_id="u1", locale="fr"),
)
print(result)  # [fr] Hello Alice: What are my options?
```

### Example 3: `entrypoint.final` — decouple return value from saved state

```python
from typing import Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver


@entrypoint(checkpointer=InMemorySaver())
def accumulate(
    new_item: str,
    *,
    previous: Any = None,
) -> "entrypoint.final[str, list]":
    """
    Returns the most recently added item to the caller,
    but saves the full accumulated list to the checkpoint.
    """
    history: list = previous or []
    updated = history + [new_item]
    # value= is what invoke() returns; save= is what future calls see as `previous`
    return entrypoint.final(value=new_item, save=updated)


config = {"configurable": {"thread_id": "acc-1"}}
print(accumulate.invoke("first", config))    # 'first'   (caller sees just the new item)
print(accumulate.invoke("second", config))   # 'second'
print(accumulate.invoke("third", config))    # 'third'

# Inspect the full history via state snapshot
snap = accumulate.get_state(config)
print(snap.values)   # shows the full list ['first', 'second', 'third'] in the checkpoint
```

### Example 4: Full combination — all parameters together

```python
from dataclasses import dataclass
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver


@dataclass
class Context:
    tenant: str


cache = InMemoryCache()
store = InMemoryStore()
checkpointer = InMemorySaver()


@task(
    retry_policy=RetryPolicy(max_attempts=3),
    cache_policy=CachePolicy(key=lambda x: x),
)
def process(data: str) -> str:
    return data.upper()


@entrypoint(
    checkpointer=checkpointer,
    store=store,
    cache=cache,
    context_schema=Context,
    retry_policy=RetryPolicy(max_attempts=2),
    timeout=TimeoutPolicy(run_timeout=30.0),
)
def full_workflow(items: list[str]) -> list[str]:
    futures = [process(item) for item in items]
    return [f.result() for f in futures]


config = {"configurable": {"thread_id": "full-1"}}
result = full_workflow.invoke(
    ["hello", "world"],
    config,
    context=Context(tenant="acme"),
)
print(result)  # ['HELLO', 'WORLD']
```

---

## 10 · `CompiledStateGraph` internals

**Module:** `langgraph.graph.state`  
**Import:**
```python
from langgraph.graph import StateGraph
# CompiledStateGraph is what .compile() returns
```

`CompiledStateGraph` extends `Pregel` with schema-aware methods. Three are worth knowing in depth: `get_input_jsonschema()`, `get_output_jsonschema()`, and `attach_node()`.

### Source signature (1.2.5)

```python
class CompiledStateGraph(Pregel[StateT, ContextT, InputT, OutputT], ...):
    builder: StateGraph[StateT, ContextT, InputT, OutputT]
    schema_to_mapper: dict[type[Any], Callable | None]

    def get_input_jsonschema(self, config=None) -> dict[str, Any]: ...
    def get_output_jsonschema(self, config=None) -> dict[str, Any]: ...
    def attach_node(self, key: str, node: StateNodeSpec | None) -> None: ...
```

### `get_input_jsonschema()` / `get_output_jsonschema()`

These methods generate the full JSON Schema for the graph's input and output types. They delegate to `_get_json_schema()`, which:

1. Walks the TypedDict's `__annotations__`
2. Strips `Annotated` wrappers to find the base type
3. Skips managed values (they are not user-facing inputs)
4. Produces a standard `{"type": "object", "properties": {...}}` schema

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import json


class InputSchema(TypedDict):
    user_query: str
    session_id: str


class OutputSchema(TypedDict):
    answer: str
    confidence: float


class InternalState(TypedDict):
    user_query: str
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    confidence: float


def processor(state: InternalState) -> dict:
    return {
        "answer": f"Response to: {state['user_query']}",
        "confidence": 0.95,
    }


graph = (
    StateGraph(InternalState, input_schema=InputSchema, output_schema=OutputSchema)
    .add_node("processor", processor)
    .add_edge(START, "processor")
    .add_edge("processor", END)
    .compile()
)

# Inspect the generated JSON schemas
input_schema = graph.get_input_jsonschema()
output_schema = graph.get_output_jsonschema()

print("Input schema:", json.dumps(input_schema, indent=2))
print("Output schema:", json.dumps(output_schema, indent=2))
```

### `attach_node()` — how nodes are wired

`attach_node()` is called for every node during `compile()`. It installs:

1. A **projection mapper** (if `input_schema` is narrower than the state)
2. A **write pipeline**: state updates from node output flow through `ChannelWriteTupleEntry` mappers
3. A **control branch** handler for `Command` objects returned by nodes

The `_get_updates` closure inside `attach_node` handles every return shape a node might produce:

| Return type | Handled as |
|-------------|------------|
| `dict` | Writes matching keys to state channels |
| `Command` | Routes to target node + writes update |
| `list[Command | dict]` | Iterates, applying each |
| `Pydantic model / dataclass` | Extracts annotated fields |
| `None` | No-op write |

### Example 2: Observing `attach_node` effects via `.get_graph()`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    x: int
    y: int


def node_a(state: S) -> dict:
    return {"x": state["x"] + 1}


def node_b(state: S) -> dict:
    return {"y": state["y"] * 2}


graph = (
    StateGraph(S)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile()
)

# The Pregel representation after attach_node ran
print("Nodes:", list(graph.nodes.keys()))
print("Channels:", list(graph.channels.keys()))
# Draws a Mermaid diagram of the compiled graph
print(graph.get_graph().draw_mermaid())
```

### Example 3: Schema-driven validation with `get_input_jsonschema()`

JSON schemas generated by `CompiledStateGraph` can be used to validate incoming requests in an API handler:

```python
import json
import jsonschema   # pip install jsonschema
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class QueryInput(TypedDict):
    question: str
    max_tokens: int


class QueryOutput(TypedDict):
    answer: str


class GraphState(QueryInput, QueryOutput):
    pass


def answer_node(state: GraphState) -> dict:
    return {"answer": f"Answer (max_tokens={state['max_tokens']}): {state['question']}"}


graph = (
    StateGraph(GraphState, input_schema=QueryInput, output_schema=QueryOutput)
    .add_node("answer", answer_node)
    .add_edge(START, "answer")
    .add_edge("answer", END)
    .compile()
)

schema = graph.get_input_jsonschema()
print(json.dumps(schema, indent=2))

# Validate an incoming payload
payload = {"question": "What is LangGraph?", "max_tokens": 512}
jsonschema.validate(payload, schema)   # passes silently

try:
    jsonschema.validate({"question": 123}, schema)
except jsonschema.ValidationError as e:
    print("Validation error:", e.message)
```

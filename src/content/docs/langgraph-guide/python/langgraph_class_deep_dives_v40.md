---
title: "LangGraph Class Deep-Dives Vol. 40"
description: "Source-verified deep dives (langgraph==1.2.9) into 10 class groups: task/@task decorator and _TaskFunction (retry_policy/cache_policy/timeout/clear_cache, SyncAsyncFuture parallel futures), entrypoint decorator (previous/context_schema/store/cache params, single-param rule, injectable parameters), LastValue/LastValueAfterFinish (one-write-per-step constraint, EmptyChannelError, deferred visibility via finish()/consume()), BaseChannel generic protocol (ValueType/UpdateType/Checkpoint type params, checkpoint/from_checkpoint lifecycle, custom channel implementation), AgentState/AgentStatePydantic (add_messages reducer, RemainingSteps managed value, extending for custom fields), ToolCallRequest/ToolCallWrapper (immutable override() pattern, tool call interceptor chains, retry-on-error wrappers), ToolCallWithContext/ToolInvocationError (Send-based parallel dispatch with state context, LLM-focused validation error messages filtered to LLM-controlled args), ValidationNode (deprecated structured-output validator — schema list, format_error customisation, migration path to response_format), Pregel (the core BSP runtime — nodes/channels/stream_mode/checkpointer, get_state/get_state_history/update_state/bulk_update_state, draw_mermaid_png topology export), and RunnableSeq/_RunnableWithWriter/_RunnableWithStore (lightweight sequential runnable for node pipelines, typed Protocol classes for stream_writer/store injection, coerce_to_runnable conversion)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 40"
  order: 71
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.9` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `task` · `_TaskFunction`

**Module:** `langgraph.func`

`@task` wraps a sync or async callable in a `_TaskFunction` instance. Calling the wrapped function inside an `entrypoint` or `StateGraph` node returns a `SyncAsyncFuture[T]` — a lightweight handle whose `.result()` blocks (sync) or can be awaited (async) until the underlying computation finishes. When a `checkpointer` is attached, task inputs and outputs are serialised to the checkpoint after each super-step, so an interrupted entrypoint re-enters without re-executing already-completed tasks.

**Key source facts** (from `langgraph/func/__init__.py`):

- `_TaskFunction.__call__` delegates to `_call_with_options(self.func, args, kwargs, retry_policy=..., cache_policy=..., timeout=...)`. The wrapped callable is never invoked directly by `@task`.
- `retry_policy` accepts a single `RetryPolicy` or a `Sequence[RetryPolicy]`. Internally it is always normalised to `Sequence[RetryPolicy]` — an empty tuple means no retries.
- `cache_policy` is a `CachePolicy[Callable[P, str | bytes]]` where the generic parameter is a *key function* that maps the task's arguments to a cache key string or bytes. The cached result is stored under `(CACHE_NS_WRITES, identifier(func) or "__dynamic__")`.
- `timeout` accepts a bare `float` (seconds), `timedelta`, or a full `TimeoutPolicy`. It is coerced via `coerce_timeout_policy()`. **Only async tasks support timeouts** — passing a `timeout` to a sync task raises `sync_timeout_unsupported()` at decoration time.
- `_TaskFunction.clear_cache(cache)` / `aclear_cache(cache)` purge the cached result for this task from a `BaseCache` without touching other tasks.
- The `name` kwarg renames the underlying callable (used for tracing and cache namespacing) without changing the attribute on the original function — `_TaskFunction` patches a `functools.partial` for class-methods or sets `__name__` directly otherwise.

### Example 1 — parallel fan-out with `@task`

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver


@task
def square(n: int) -> int:
    return n * n


@entrypoint(checkpointer=InMemorySaver())
def compute(numbers: list[int]) -> list[int]:
    futures = [square(n) for n in numbers]
    return [f.result() for f in futures]


config = {"configurable": {"thread_id": "t1"}}
result = compute.invoke([1, 2, 3, 4, 5], config)
print(result)  # [1, 4, 9, 16, 25]
```

### Example 2 — task with `retry_policy` and `timeout` (async)

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

attempts = 0


@task(
    retry_policy=RetryPolicy(max_attempts=3, backoff_factor=0.1),
    timeout=5.0,
)
async def fetch_data(url: str) -> str:
    global attempts
    attempts += 1
    if attempts < 2:
        raise RuntimeError("transient error")
    await asyncio.sleep(0.01)
    return f"data from {url}"


@entrypoint(checkpointer=InMemorySaver())
async def pipeline(url: str) -> str:
    future = fetch_data(url)
    return await future


config = {"configurable": {"thread_id": "t2"}}
result = asyncio.run(pipeline.ainvoke("https://example.com", config))
print(result)           # data from https://example.com
print(f"Took {attempts} attempt(s)")  # Took 2 attempt(s)
```

### Example 3 — task with `cache_policy` and `clear_cache`

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver

cache = InMemoryCache()
call_count = 0


@task(cache_policy=CachePolicy(key_func=lambda x: str(x)))
def expensive(x: int) -> int:
    global call_count
    call_count += 1
    return x * 100


@entrypoint(checkpointer=InMemorySaver(), cache=cache)
def workflow(x: int) -> int:
    return expensive(x).result()


config = {"configurable": {"thread_id": "t3"}}
workflow.invoke(7, config)
workflow.invoke(7, config)  # second invoke hits cache
print(f"Computed {call_count} time(s)")  # Computed 1 time(s)

# Evict cache for this task
expensive.clear_cache(cache)
workflow.invoke(7, config)  # recomputed after eviction
print(f"Now computed {call_count} time(s)")  # Now computed 2 time(s)
```

---

## 2 · `entrypoint`

**Module:** `langgraph.func`

`@entrypoint(...)` is a decorator class that converts a plain function into a `Pregel` graph. The decorated function must accept **exactly one positional parameter** (the input) plus optional **injected** keyword parameters: `config`, `previous`, and `runtime`. The resulting `Pregel` object implements the full LangChain Runnable interface (`invoke` / `ainvoke` / `stream` / `astream` / `astream_events`).

**Key source facts** (from `langgraph/func/__init__.py`):

- The single-input rule is enforced at decoration time via `inspect.signature`. Generators and async generators raise `NotImplementedError`.
- `previous` is populated from the channel keyed `PREVIOUS` (a `LastValue` channel). If no checkpoint exists for the thread, `previous` is `MISSING` and the function's default value is used — typically `None`.
- `context_schema` (replacing deprecated `config_schema`) wires the context via `Runtime[ContextT]`. Callers supply context via the `context=` keyword argument to `invoke()` / `stream()` — separate from `config["configurable"]`, which carries only runnable/checkpoint settings such as `thread_id`.
- `store`, `cache`, `cache_policy`, `retry_policy`, and `timeout` are all passed directly to the internal `Pregel` constructor.
- `entrypoint.final(value=..., save=...)` decouples what the caller receives (`value`) from what is written to the `PREVIOUS` channel for the next invocation (`save`). The output type annotation `-> entrypoint.final[R, S]` lets type checkers infer both types; unparameterised `-> entrypoint.final` defaults both to `Any`.
- The `stream_mode` of the underlying `Pregel` is always `"updates"` and `stream_eager=True`, so partial results appear as soon as each task resolves.

### Example 1 — accumulating state with `previous`

```python
from typing import Optional
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver


@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: Optional[int] = None) -> int:
    current = previous or 0
    return current + increment


config = {"configurable": {"thread_id": "acc"}}
print(counter.invoke(5, config))   # 5
print(counter.invoke(3, config))   # 8
print(counter.invoke(10, config))  # 18
```

### Example 2 — `entrypoint.final` to separate return value from checkpoint

```python
from typing import Any
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver


@entrypoint(checkpointer=InMemorySaver())
def doubling_workflow(
    n: int, *, previous: Any = None
) -> "entrypoint.final[int, int]":
    saved = previous or 0
    # Return the last checkpoint value to the caller,
    # but save double-n for the next run.
    return entrypoint.final(value=saved, save=n * 2)


config = {"configurable": {"thread_id": "dbl"}}
print(doubling_workflow.invoke(3, config))   # 0  (previous was None)
print(doubling_workflow.invoke(4, config))   # 6  (3 * 2 saved previously)
print(doubling_workflow.invoke(10, config))  # 8  (4 * 2 saved previously)
```

### Example 3 — `context_schema` for typed run-time context

```python
from typing_extensions import TypedDict
from langgraph.func import entrypoint, task
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver


class RunContext(TypedDict):
    user_id: str
    role: str


@task
def greet(name: str) -> str:
    return f"Hello, {name}"


@entrypoint(
    checkpointer=InMemorySaver(),
    context_schema=RunContext,
)
def personalised(name: str, *, runtime: Runtime[RunContext]) -> str:
    ctx = runtime.context  # typed as RunContext
    greeting = greet(name).result()
    return f"[{ctx['role'].upper()}] {greeting} (uid={ctx['user_id']})"


config = {"configurable": {"thread_id": "ctx1"}}
print(personalised.invoke("Alice", config, context={"user_id": "u42", "role": "admin"}))
# [ADMIN] Hello, Alice (uid=u42)
```

---

## 3 · `LastValue` · `LastValueAfterFinish`

**Module:** `langgraph.channels.last_value`

`LastValue` is the default channel type for `StateGraph` fields not annotated with a reducer. It enforces **at most one write per super-step** — concurrent writes from multiple nodes raise `InvalidUpdateError`. `LastValueAfterFinish` extends this with a deferred-visibility gate: the value is only readable after `finish()` is called by the runtime, and `consume()` clears it after reading, making it useful for ephemeral handoff patterns.

**Key source facts** (from `langgraph/channels/last_value.py`):

- `LastValue.update(values)` raises `InvalidUpdateError` if `len(values) != 1`. The error message instructs using an `Annotated` key with a reducer.
- `LastValue.get()` raises `EmptyChannelError` when `self.value is MISSING` (before any write). `is_available()` is a cheap non-raising alternative.
- `LastValue.checkpoint()` returns `self.value`, which may be `MISSING` — callers that persist channels must handle `MISSING` as an absent entry.
- `LastValueAfterFinish.finish()` sets `self.finished = True` and returns `True` only if a value is already stored and not yet finished. The runtime calls `finish()` at the end of the super-step before exposing the channel to reads.
- `LastValueAfterFinish.consume()` clears both `self.value` and `self.finished` and returns `True` — the value disappears after one read, making it ideal for single-use signal channels.
- `LastValueAfterFinish.checkpoint()` serialises as `(value, finished)` tuple, preserving pending state across restarts.

### Example 1 — observing the one-write constraint

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import InvalidUpdateError


class State(TypedDict):
    value: int


def node_a(state: State) -> dict:
    return {"value": 10}


def node_b(state: State) -> dict:
    return {"value": 20}  # concurrent write → conflict


builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")  # both run in the same super-step
builder.add_edge("a", END)
builder.add_edge("b", END)

graph = builder.compile()
try:
    graph.invoke({"value": 0})
except InvalidUpdateError as e:
    print(f"Caught: {e}")
    # Caught: ... Can receive only one value per step ...
```

### Example 2 — inspecting channel availability with `is_available`

```python
from langgraph.channels.last_value import LastValue
from langgraph._internal._typing import MISSING


ch: LastValue[int] = LastValue(int, key="x")

print(ch.is_available())  # False — never updated

ch.update([42])
print(ch.is_available())  # True
print(ch.get())           # 42
print(ch.checkpoint())    # 42

ch2 = ch.from_checkpoint(ch.checkpoint())
print(ch2.get())          # 42 — restored from checkpoint
```

### Example 3 — `LastValueAfterFinish` deferred single-use signal

```python
from langgraph.channels.last_value import LastValueAfterFinish
from langgraph.errors import EmptyChannelError


signal: LastValueAfterFinish[str] = LastValueAfterFinish(str, key="signal")

signal.update(["ready"])
try:
    signal.get()  # not yet visible — finish() not called
except EmptyChannelError:
    print("Not yet available")

signal.finish()          # make it available
print(signal.get())      # ready

signal.consume()         # clear after one read
print(signal.is_available())  # False — consumed
```

---

## 4 · `BaseChannel`

**Module:** `langgraph.channels.base`

`BaseChannel[Value, Update, Checkpoint]` is the three-parameter generic ABC that all LangGraph channels implement. The type parameters express what you *read* (`Value`), what nodes *write* (`Update`), and what gets *serialised* (`Checkpoint`). For simple channels these are all the same type; `BinaryOperatorAggregate` has `Update == Value` but the binary operator changes them; `LastValueAfterFinish` uses `Checkpoint == tuple[Value, bool]` to persist the pending-finish state.

**Key source facts** (from `langgraph/channels/base.py`):

- `__slots__ = ("key", "typ")` — lightweight; `key` is the state-field name injected by `StateGraph` at build time, `typ` is the raw Python type annotation.
- `checkpoint()` defaults to `self.get()` — subclasses that store intermediate state (e.g. `LastValueAfterFinish`) must override it.
- `from_checkpoint(checkpoint)` is an abstract classmethod-style factory that returns a *new* channel instance pre-loaded with the checkpoint value. Returning `self` would break replay semantics.
- `copy()` defaults to `from_checkpoint(self.checkpoint())` but subclasses may override for efficiency (e.g. `LastValue.copy()` sets `value` directly).
- `is_available()` defaults to a try/except around `get()` — override with a cheap flag check for hot paths.
- Channels *must not* be shared between threads without copying; each `PregelRunner` task receives its own channel copy.

### Example 1 — implementing a min-value channel

```python
from collections.abc import Sequence
from typing import Any
from typing_extensions import Self
from langgraph.channels.base import BaseChannel
from langgraph.errors import EmptyChannelError
from langgraph._internal._typing import MISSING


class MinChannel(BaseChannel[int, int, int]):
    """Keeps the minimum value seen across concurrent writes."""

    __slots__ = ("value",)

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value: int | object = MISSING

    @property
    def ValueType(self) -> type[int]:
        return int

    @property
    def UpdateType(self) -> type[int]:
        return int

    def from_checkpoint(self, checkpoint: int | Any) -> Self:
        ch = self.__class__(self.typ, self.key)
        if checkpoint is not MISSING:
            ch.value = checkpoint
        return ch

    def update(self, values: Sequence[int]) -> bool:
        if not values:
            return False
        candidate = min(values)
        if self.value is MISSING or candidate < self.value:  # type: ignore[operator]
            self.value = candidate
            return True
        return False

    def get(self) -> int:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value  # type: ignore[return-value]

    def is_available(self) -> bool:
        return self.value is not MISSING


ch: MinChannel = MinChannel(int, key="min_score")
ch.update([5, 3, 8, 1, 9])
print(ch.get())        # 1 (minimum of all updates)

snapshot = ch.checkpoint()
restored = ch.from_checkpoint(snapshot)
print(restored.get())  # 1
```

### Example 2 — using a custom channel in `StateGraph`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    scores: Annotated[int, MinChannel(int)]  # use our custom channel


def score_a(state: State) -> dict:
    return {"scores": 7}


def score_b(state: State) -> dict:
    return {"scores": 3}


builder = StateGraph(State)
builder.add_node("a", score_a)
builder.add_node("b", score_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", END)
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"scores": 100})
print(result["scores"])  # 3 (min of 7, 3)
```

### Example 3 — inspecting channel type parameters

```python
from langgraph.channels.last_value import LastValue, LastValueAfterFinish
from langgraph.channels.binop import BinaryOperatorAggregate
import operator

lv: LastValue[str] = LastValue(str, "name")
print(lv.ValueType)    # <class 'str'>
print(lv.UpdateType)   # <class 'str'>

laf: LastValueAfterFinish[int] = LastValueAfterFinish(int, "counter")
print(laf.ValueType)   # <class 'int'>
print(laf.UpdateType)  # <class 'int'>

agg: BinaryOperatorAggregate[list, list] = BinaryOperatorAggregate(list, operator.add)
agg.key = "items"
print(agg.ValueType)   # <class 'list'>
print(agg.UpdateType)  # <class 'list'>
```

---

## 5 · `AgentState` · `AgentStatePydantic`

**Module:** `langgraph.prebuilt.chat_agent_executor`

`AgentState` is the `TypedDict` that `create_react_agent` uses by default. Its two fields are `messages: Annotated[Sequence[BaseMessage], add_messages]` and `remaining_steps: NotRequired[RemainingSteps]`. `remaining_steps` is backed by the `RemainingStepsManager` managed value, which auto-decrements each super-step and signals `IsLastStep` when it hits 1. `AgentStatePydantic` is a Pydantic `BaseModel` equivalent that was deprecated in v0.10.0 — the canonical replacement is `AgentState` (or the upstream `langchain.agents.AgentState`).

**Key source facts** (from `langgraph/prebuilt/chat_agent_executor.py`):

- `add_messages` is the reducer for `messages`: it appends new messages, merges by `id` for updates, and respects `RemoveMessage` objects.
- `remaining_steps: NotRequired[RemainingSteps]` means the field is optional in the initial state dict; the `RemainingStepsManager` initialises it to `recursion_limit - 1` on the first step.
- `RemainingSteps` is an alias for `int` — the managed value is the integer count itself, not a wrapper.
- `create_react_agent` accepts `state_schema` to substitute a custom schema; the custom schema must include a `messages` field with the `add_messages` reducer, and may include `remaining_steps` to enable the last-step guard.
- `tools_condition` checks `state["messages"][-1].tool_calls` — if present it routes to `"tools"`, otherwise to `END`.

### Example 1 — inspecting `AgentState` structure

```python
from typing import get_type_hints, get_args
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph.message import add_messages

hints = get_type_hints(AgentState, include_extras=True)
print(list(hints.keys()))    # ['messages', 'remaining_steps']

# Unpack the Annotated hint
msg_annotation = hints["messages"]
base_type, reducer = get_args(msg_annotation)
print(base_type)   # typing.Sequence[langchain_core.messages.base.BaseMessage]
print(reducer)     # <function add_messages at 0x...>
```

### Example 2 — custom state that extends `AgentState` with extra fields

```python
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps


class EnrichedState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps
    call_count: Annotated[int, operator.add]  # accumulate across steps


def count_step(state: EnrichedState) -> dict:
    return {"call_count": 1}  # each invocation adds 1


builder = StateGraph(EnrichedState)
builder.add_node("count", count_step)
builder.add_edge(START, "count")
builder.add_edge("count", END)

graph = builder.compile()
result = graph.invoke({"messages": [], "call_count": 0})
print(result["call_count"])  # 1
```

### Example 3 — using `remaining_steps` to guard the last super-step

```python
from typing import Sequence
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps, IsLastStep


class BoundedState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps
    is_last_step: IsLastStep


def safe_node(state: BoundedState) -> dict:
    if state["is_last_step"]:
        return {"messages": [{"role": "assistant", "content": "Stopping: step limit reached."}]}
    return {"messages": [{"role": "assistant", "content": "Working..."}]}


builder = StateGraph(BoundedState)
builder.add_node("work", safe_node)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile()
result = graph.invoke(
    {"messages": [HumanMessage(content="Go")]},
    config={"recursion_limit": 3},
)
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")
```

---

## 6 · `ToolCallRequest` · `ToolCallWrapper`

**Module:** `langgraph.prebuilt.tool_node`

`ToolCallRequest` is the immutable-by-convention dataclass that `ToolNode` passes to every registered *tool call wrapper*. A `ToolCallWrapper` is a `Callable[[ToolCallRequest, execute], ToolMessage | Command]` — it receives the request and an `execute` callable, and may call `execute` zero or more times (retry, short-circuit, or modify the request before executing). The `override()` method on `ToolCallRequest` returns a *new* instance with specific fields changed, following an immutable-update pattern.

**Key source facts** (from `langgraph/prebuilt/tool_node.py`):

- `ToolCallRequest` is a `@dataclass` with `__setattr__` overridden to emit a `DeprecationWarning` on direct assignment — use `override()` instead.
- `override(**overrides)` calls `dataclasses.replace(self, **overrides)`. Supported keys are `tool_call` and `state` (`TypedDict`-constrained by `_ToolCallRequestOverrides`).
- The `execute` callable passed to a `ToolCallWrapper` can be called **multiple times** with different `ToolCallRequest` objects — useful for retry-on-error patterns.
- `ToolNode` accepts `tool_call_wrappers: list[ToolCallWrapper]` and `async_tool_call_wrappers: list[AsyncToolCallWrapper]` (passed as `wrap_tool_call` / `awrap_tool_call` on the `ToolNode` constructor in newer releases or via `create_react_agent`'s `pre_tool_call_hook`).
- `AsyncToolCallWrapper` is `Callable[[ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]], Awaitable[ToolMessage | Command]]`.

### Example 1 — logging wrapper (passthrough interceptor)

```python
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from typing_extensions import TypedDict


def logging_wrapper(
    request: ToolCallRequest,
    execute,
) -> ToolMessage:
    print(f"[BEFORE] tool={request.tool_call['name']} args={request.tool_call['args']}")
    result = execute(request)
    print(f"[AFTER]  content={result.content!r}")
    return result


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


class S(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


node = ToolNode([multiply], wrap_tool_call=logging_wrapper)
builder = StateGraph(S)
builder.add_node("tools", node)
builder.add_edge(START, "tools")
builder.add_edge("tools", END)
g = builder.compile()

tool_call = {"name": "multiply", "args": {"a": 3, "b": 7}, "id": "c1", "type": "tool_call"}
result = g.invoke({"messages": [AIMessage(content="", tool_calls=[tool_call])]})
# [BEFORE] tool=multiply args={'a': 3, 'b': 7}
# [AFTER]  content='21'
print(result["messages"][-1].content)  # 21
```

### Example 2 — `override()` to modify arguments before execution

```python
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from typing_extensions import TypedDict


def clamp_value_wrapper(
    request: ToolCallRequest,
    execute,
) -> ToolMessage:
    """Clamp the 'value' argument to [0, 100] before executing."""
    args = dict(request.tool_call["args"])
    args["value"] = max(0, min(100, args.get("value", 0)))
    new_call = {**request.tool_call, "args": args}
    return execute(request.override(tool_call=new_call))


@tool
def set_volume(value: int) -> str:
    """Set the volume level."""
    return f"Volume set to {value}"


class S(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


node = ToolNode([set_volume], wrap_tool_call=clamp_value_wrapper)
builder = StateGraph(S)
builder.add_node("tools", node)
builder.add_edge(START, "tools")
builder.add_edge("tools", END)
g = builder.compile()

tool_call = {"name": "set_volume", "args": {"value": 150}, "id": "v1", "type": "tool_call"}
result = g.invoke({"messages": [AIMessage(content="", tool_calls=[tool_call])]})
print(result["messages"][-1].content)  # Volume set to 100
```

### Example 3 — retry wrapper with up to 3 attempts

```python
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from typing_extensions import TypedDict

call_count = 0


@tool
def flaky_api(query: str) -> str:
    """Query a flaky external API."""
    global call_count
    call_count += 1
    if call_count < 3:
        raise RuntimeError("API unavailable")
    return f"result for '{query}'"


def retry_wrapper(
    request: ToolCallRequest,
    execute,
) -> ToolMessage:
    last_error = None
    for attempt in range(3):
        try:
            return execute(request)
        except Exception as e:
            last_error = e
    return ToolMessage(
        content=f"Failed after 3 attempts: {last_error}",
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
    )


class S(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


node = ToolNode([flaky_api], handle_tool_errors=False, wrap_tool_call=retry_wrapper)
builder = StateGraph(S)
builder.add_node("tools", node)
builder.add_edge(START, "tools")
builder.add_edge("tools", END)
g = builder.compile()

tool_call = {"name": "flaky_api", "args": {"query": "data"}, "id": "r1", "type": "tool_call"}
result = g.invoke({"messages": [AIMessage(content="", tool_calls=[tool_call])]})
print(result["messages"][-1].content)  # result for 'data'
print(f"Total calls: {call_count}")    # Total calls: 3
```

---

## 7 · `ToolCallWithContext` · `ToolInvocationError`

**Module:** `langgraph.prebuilt.tool_node`

`ToolCallWithContext` is an internal `TypedDict` used when `create_react_agent` dispatches tool calls via the `Send` API for parallel execution — each `Send("tools", ToolCallWithContext(...))` carries a `tool_call` dict, a `__type` discriminator, and the current `state` snapshot. `ToolInvocationError` is the `ToolException` subclass raised by `ToolNode` when Pydantic validation fails on a tool's arguments — its message is filtered to only include errors for LLM-controlled arguments (not injected args like state/store).

**Key source facts** (from `langgraph/prebuilt/tool_node.py`):

- `ToolCallWithContext.__type` uses a double-underscore prefix (`"__type"`) to avoid collisions with user-defined state keys. The runtime checks this discriminator to detect whether a `Send` payload is a bare tool call dict or a context-augmented one.
- `ToolInvocationError` stores `tool_name`, `tool_kwargs`, `source` (the original `ValidationError`), and `filtered_errors` (errors for non-injected args only). Its `.message` attribute contains a formatted string using `TOOL_INVOCATION_ERROR_TEMPLATE`.
- `_filter_validation_errors(validation_error, injected_args)` removes errors for `InjectedState`, `InjectedStore`, and `ToolRuntime` parameters so the LLM only receives feedback about arguments *it* controls.
- `msg_content_output(output)` converts tool return values to `ToolMessage.content`: strings pass through, `list[dict]` with content block types pass through, and everything else is JSON-serialised with a `str()` fallback.
- `ToolNode` catches `ToolInvocationError` by default (`handle_tool_errors=True`) and returns the error message as a `ToolMessage` with `status="error"`.

### Example 1 — observing `ToolInvocationError` for invalid arguments

```python
from pydantic import BaseModel, field_validator
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode


class StrictArgs(BaseModel):
    count: int

    @field_validator("count")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("count must be positive")
        return v


@tool(args_schema=StrictArgs)
def batch_process(count: int) -> str:
    """Process items in a batch."""
    return f"Processed {count} items"


node = ToolNode([batch_process])

bad_call = {"name": "batch_process", "args": {"count": -5}, "id": "e1", "type": "tool_call"}
state = {"messages": [AIMessage(content="", tool_calls=[bad_call])]}
result = node.invoke(state)
msg = result["messages"][0]
print(f"status={msg.status}")      # status=error
print(f"content={msg.content!r}")  # contains the validation error for 'count'
```

### Example 2 — `msg_content_output` content normalisation

```python
from langgraph.prebuilt.tool_node import msg_content_output

# Plain string — passes through
print(msg_content_output("hello"))             # hello

# Content block list — passes through
blocks = [{"type": "text", "text": "world"}]
print(msg_content_output(blocks))              # [{'type': 'text', 'text': 'world'}]

# Arbitrary object — JSON-serialised
print(msg_content_output({"key": [1, 2, 3]})) # {"key": [1, 2, 3]}

# Non-serialisable — falls back to str()
class Unserializable:
    def __repr__(self): return "Unserializable()"
print(msg_content_output(Unserializable()))    # Unserializable()
```

### Example 3 — understanding `ToolCallWithContext` for Send-based dispatch

```python
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_results: Annotated[list, lambda a, b: a + b]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def dispatch_tools(state: State) -> list[Send]:
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return []
    return [
        Send(
            "run_tool",
            {
                "tool_call": tc,
                "__type": "tool_call_with_context",
                "state": state,
            },
        )
        for tc in last.tool_calls
    ]


def run_tool(payload: dict) -> dict:
    tc = payload["tool_call"]
    result = add.invoke(tc["args"])
    return {
        "tool_results": [result],
        "messages": [
            ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
        ],
    }


builder = StateGraph(State)
builder.add_node("dispatch", lambda s: None)
builder.add_conditional_edges("dispatch", dispatch_tools, ["run_tool"])
builder.add_node("run_tool", run_tool)
builder.add_edge("run_tool", END)
builder.add_edge(START, "dispatch")

graph = builder.compile()
tool_calls = [
    {"name": "add", "args": {"a": 1, "b": 2}, "id": "t1", "type": "tool_call"},
    {"name": "add", "args": {"a": 10, "b": 20}, "id": "t2", "type": "tool_call"},
]
result = graph.invoke({"messages": [AIMessage(content="", tool_calls=tool_calls)], "tool_results": []})
print(result["tool_results"])  # [3, 30]
```

---

## 8 · `ValidationNode`

**Module:** `langgraph.prebuilt.tool_validator`

`ValidationNode` validates tool-call arguments against Pydantic schemas without executing the tool. It is useful for structured-extraction pipelines where you want to re-prompt the model until it produces arguments that conform to a schema. **As of langgraph 0.10.0, `ValidationNode` is deprecated** — the canonical replacement is `create_react_agent(response_format=MySchema)` or tool-level argument validation via `ToolNode` with `handle_tool_errors=True`.

**Key source facts** (from `langgraph/prebuilt/tool_validator.py`):

- `ValidationNode.__init__` accepts a list of `BaseTool | type[BaseModel] | Callable`. Tools are extracted by their `args_schema`; callables get a schema auto-generated from their signature via `create_schema_from_function`.
- `_get_message` handles both list-of-messages and `{"messages": [...]}` input shapes.
- `_func` uses `get_executor_for_config(config)` for parallel validation — all tool calls in the last `AIMessage` are validated concurrently.
- On validation success, a `ToolMessage` with `content=model.model_dump_json()` is returned. On failure, a `ToolMessage` with `additional_kwargs={"is_error": True}` is returned, containing the formatted error string.
- `format_error` defaults to `_default_format_error` which combines `repr(error)` with the instruction `"Respond after fixing all validation errors."`.
- The re-prompt loop is driven by a conditional edge — `should_reprompt` scans backward through messages to find the last `AIMessage`, then checks for `is_error` tool messages above it.

### Example 1 — basic structured extraction with re-prompting (migration-ready)

```python
from typing import Literal
from pydantic import BaseModel, field_validator
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import warnings

# Suppress the deprecation warning for this demo
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langgraph.prebuilt import ValidationNode


class PickNumber(BaseModel):
    number: int

    @field_validator("number")
    @classmethod
    def only_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError("Only even numbers allowed")
        return v


class State(TypedDict):
    messages: Annotated[list, add_messages]


def fake_llm(state: State) -> dict:
    """Simulate an LLM that always picks 7 (odd — will fail) on first try."""
    msgs = state["messages"]
    # First call: pick an odd number; subsequent calls: pick even
    last_is_error = any(
        getattr(m, "additional_kwargs", {}).get("is_error")
        for m in msgs
        if hasattr(m, "additional_kwargs")
    )
    number = 8 if last_is_error else 7
    from langchain_core.messages import AIMessage
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "PickNumber",
                    "args": {"number": number},
                    "id": "pick1",
                    "type": "tool_call",
                }],
            )
        ]
    }


def should_validate(state: State) -> Literal["validate", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "validate"
    return "__end__"


def should_reprompt(state: State) -> Literal["llm", "__end__"]:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "ai":
            return "__end__"
        if getattr(msg, "additional_kwargs", {}).get("is_error"):
            return "llm"
    return "__end__"


builder = StateGraph(State)
builder.add_node("llm", fake_llm)
builder.add_node("validate", ValidationNode([PickNumber]))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_validate)
builder.add_conditional_edges("validate", should_reprompt)

graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage(content="Pick a number")]})
for msg in result["messages"]:
    print(f"{msg.type}: {getattr(msg, 'content', '')!r}")
```

### Example 2 — custom `format_error` for cleaner LLM feedback

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pydantic import BaseModel, field_validator, ValidationError
from langchain_core.messages import ToolCall
from langgraph.prebuilt import ValidationNode


class Temperature(BaseModel):
    celsius: float

    @field_validator("celsius")
    @classmethod
    def in_range(cls, v: float) -> float:
        if not (-273.15 <= v <= 1000):
            raise ValueError(f"Temperature {v} is out of physical range")
        return v


def concise_format_error(
    error: BaseException,
    call: ToolCall,
    schema: type[BaseModel],
) -> str:
    if isinstance(error, ValidationError):
        msgs = [e["msg"] for e in error.errors()]
        return "Fix: " + "; ".join(msgs)
    return f"Unexpected error: {error}"


node = ValidationNode([Temperature], format_error=concise_format_error)

from langchain_core.messages import AIMessage
state = {
    "messages": [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "Temperature",
                "args": {"celsius": -300},
                "id": "t1",
                "type": "tool_call",
            }],
        )
    ]
}
result = node.invoke(state)
err_msg = result["messages"][0]
print(err_msg.content)  # Fix: Temperature -300 is out of physical range
print(err_msg.additional_kwargs)  # {'is_error': True}
```

### Example 3 — migration: replacing `ValidationNode` with `response_format`

```python
# BEFORE (deprecated):
# from langgraph.prebuilt import ValidationNode
# node = ValidationNode([MySchema])

# AFTER — use create_react_agent with response_format:
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


class AnalysisResult(BaseModel):
    sentiment: str
    score: float
    summary: str


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# response_format instructs the agent to return a structured final message
# The agent will call the model with structured output after all tool calls
# are complete, ensuring the final response matches AnalysisResult.
# (Requires a model that supports structured output / tool calling)

# graph = create_react_agent(
#     model="claude-sonnet-5",
#     tools=[search],
#     response_format=AnalysisResult,
# )

# Demonstrate the schema structure the agent would use
print("response_format schema:")
print(AnalysisResult.model_json_schema())
# {'properties': {'sentiment': ..., 'score': ..., 'summary': ...}, ...}
```

---

## 9 · `Pregel`

**Module:** `langgraph.pregel.main`

`Pregel` is the compiled runtime that underlies every `StateGraph.compile()` and `@entrypoint(...)` call. It implements the **Bulk Synchronous Parallel** model: each step plans which nodes to run, executes them in parallel, then applies channel updates. You rarely construct `Pregel` directly — but inspecting and operating on the compiled object is central to time-travel, state injection, and topology analysis.

**Key source facts** (from `langgraph/pregel/main.py`):

- `Pregel.nodes: dict[str, PregelNode]` — the actor registry. Each `PregelNode` stores the bound runnable, its trigger channels, its output channels, and its retry/cache/timeout policies.
- `Pregel.channels: dict[str, BaseChannel]` — the full channel map, including `START`, `END`, state fields, and any managed-value channels.
- `Pregel.stream_mode: StreamMode | Sequence[StreamMode]` — the default mode(s) for `.stream()`. Overridable per-invocation.
- `Pregel.get_state(config, *, subgraphs=False) -> StateSnapshot` retrieves the latest checkpoint for a thread. `subgraphs=True` recursively includes child-graph state.
- `Pregel.get_state_history(config, *, filter=None, before=None, limit=None)` yields `StateSnapshot` objects in reverse chronological order — the foundation for time-travel.
- `Pregel.update_state(config, values, as_node=None) -> RunnableConfig` writes a state patch as if `as_node` had returned it, creating a new checkpoint. Returns the updated config (with new `checkpoint_id`).
- `Pregel.bulk_update_state(config, supersteps) -> RunnableConfig` applies multiple state updates atomically across multiple super-steps in one call.
- `Pregel.get_graph(config=None, *, xray=False) -> Graph` returns a drawable `Graph` for topology visualisation. `xray=True` recursively expands subgraphs.

### Example 1 — inspecting a compiled graph's channels and nodes

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    count: int
    label: str


def step_one(state: State) -> dict:
    return {"count": state["count"] + 1, "label": "done"}


builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_edge(START, "step_one")
builder.add_edge("step_one", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Inspect the Pregel object
print("Nodes:", list(graph.nodes.keys()))
# Nodes: ['step_one']

print("Channels:", list(graph.channels.keys()))
# Channels: ['__start__', 'count', 'label', '__end__', ...]

from langgraph.channels.last_value import LastValue
for name, ch in graph.channels.items():
    if isinstance(ch, LastValue):
        print(f"  {name}: LastValue[{ch.typ}]")
```

### Example 2 — time-travel with `get_state_history` and `update_state`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    steps: list


def add_step(state: State) -> dict:
    return {"steps": state["steps"] + [len(state["steps"]) + 1]}


builder = StateGraph(State)
builder.add_node("add_step", add_step)
builder.add_edge(START, "add_step")
builder.add_edge("add_step", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "history_demo"}}

# Run 3 times
for _ in range(3):
    graph.invoke({"steps": []}, config)

# Time-travel: inspect history
print("State history (newest first):")
for snapshot in graph.get_state_history(config):
    print(f"  step={snapshot.metadata.get('step')} values={snapshot.values}")

# Rewind to step 1 and patch state
history = list(graph.get_state_history(config))
checkpoint_at_step1 = history[-2]  # second-to-last is after step 1

patched_config = graph.update_state(
    checkpoint_at_step1.config,
    {"steps": [10, 20]},  # inject different values
)
result = graph.invoke(None, patched_config)
print("After time-travel patch:", result["steps"])  # [10, 20, 3]
```

### Example 3 — `get_graph` for topology export

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    x: int


def double(state: State) -> dict:
    return {"x": state["x"] * 2}


def triple(state: State) -> dict:
    return {"x": state["x"] * 3}


def router(state: State) -> str:
    return "double" if state["x"] < 10 else "triple"


builder = StateGraph(State)
builder.add_node("double", double)
builder.add_node("triple", triple)
builder.add_edge(START, "double")
builder.add_conditional_edges("double", router, {"double": "double", "triple": "triple"})
builder.add_edge("triple", END)

graph = builder.compile()
topology = graph.get_graph()

print("Nodes:")
for node in topology.nodes.values():
    print(f"  {node.id}")

print("Edges:")
for edge in topology.edges:
    print(f"  {edge.source} -> {edge.target}")

# Export as Mermaid markdown
mermaid = graph.get_graph().draw_mermaid()
print("\nMermaid diagram:")
print(mermaid)
```

---

## 10 · `RunnableSeq` · `_RunnableWithWriter` · `_RunnableWithStore`

**Module:** `langgraph._internal._runnable`

`RunnableSeq` is a lightweight, LangGraph-internal sequential runnable that chains steps without the overhead of `langchain_core.RunnableSequence`. `_RunnableWithWriter`, `_RunnableWithStore`, and `_RunnableWithWriterStore` are `Protocol` classes used to detect whether a callable's signature accepts `writer: StreamWriter` and/or `store: BaseStore` keyword arguments. The `RunnableCallable` wrapper (covered in Vol. 39) inspects these protocols to decide which arguments to inject at runtime.

**Key source facts** (from `langgraph/_internal/_runnable.py`):

- `RunnableSeq(*steps, name=None, trace_inputs=None)` requires at least 2 steps. It calls `coerce_to_runnable(step)` on each, converting plain callables to `RunnableCallable` instances.
- `RunnableSeq.invoke(input, config)` chains steps sequentially: each step's output is the next step's input. Config is passed unchanged.
- `RunnableSeq.ainvoke(input, config)` is the async equivalent.
- `_RunnableWithWriter.__call__(state, *, writer: StreamWriter) -> Output` — any callable matching this protocol receives the current graph's stream writer, enabling mid-node streaming via `writer({"my_key": chunk})`.
- `_RunnableWithStore.__call__(state, *, store: BaseStore) -> Output` — matching callables receive the graph's store for key-value lookups without going through `InjectedStore` annotation.
- `_RunnableWithConfigWriter` and `_RunnableWithConfigStore` extend this with `config: RunnableConfig` as well.
- `coerce_to_runnable(thing: RunnableLike) -> Runnable` converts callables to `RunnableCallable` and `RunnableSequence`s to `RunnableSeq`.

### Example 1 — `RunnableSeq` for multi-stage node pipelines

```python
from langgraph._internal._runnable import RunnableSeq, RunnableCallable


def normalise(text: str) -> str:
    return text.strip().lower()


def tokenise(text: str) -> list[str]:
    return text.split()


def count_words(tokens: list[str]) -> dict:
    return {"word_count": len(tokens), "unique": len(set(tokens))}


pipeline = RunnableSeq(normalise, tokenise, count_words, name="text_pipeline")

result = pipeline.invoke("  Hello World Hello  ", {})
print(result)
# {'word_count': 3, 'unique': 2}
```

### Example 2 — `_RunnableWithWriter` for mid-node streaming

```python
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter
from typing_extensions import TypedDict


class State(TypedDict):
    items: list[str]
    results: list[str]


def streaming_node(state: State, *, writer: StreamWriter) -> dict:
    """A node that streams partial results as it processes each item."""
    results = []
    for item in state["items"]:
        processed = item.upper()
        results.append(processed)
        writer({"partial": processed})  # stream each result as it's ready
    return {"results": results}


builder = StateGraph(State)
builder.add_node("process", streaming_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()

print("Streaming partial results:")
for chunk in graph.stream(
    {"items": ["alpha", "beta", "gamma"], "results": []},
    stream_mode="custom",
):
    print(f"  chunk: {chunk}")

result = graph.invoke({"items": ["alpha", "beta", "gamma"], "results": []})
print("Final results:", result["results"])
```

### Example 3 — `_RunnableWithStore` for store access without annotation

```python
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict


class State(TypedDict):
    user_id: str
    greeting: str


def personalise_node(state: State, *, store: BaseStore) -> dict:
    """Access the store directly via the _RunnableWithStore protocol."""
    namespace = ("preferences", state["user_id"])
    pref = store.get(namespace, "language")
    lang = pref.value if pref else "en"
    greetings = {"en": "Hello", "es": "Hola", "fr": "Bonjour"}
    return {"greeting": f"{greetings.get(lang, 'Hello')}, {state['user_id']}!"}


store = InMemoryStore()
store.put(("preferences", "alice"), "language", "fr")
store.put(("preferences", "bob"), "language", "es")

builder = StateGraph(State)
builder.add_node("personalise", personalise_node)
builder.add_edge(START, "personalise")
builder.add_edge("personalise", END)

graph = builder.compile(store=store)

print(graph.invoke({"user_id": "alice", "greeting": ""})["greeting"])
# Bonjour, alice!
print(graph.invoke({"user_id": "bob", "greeting": ""})["greeting"])
# Hola, bob!
print(graph.invoke({"user_id": "charlie", "greeting": ""})["greeting"])
# Hello, charlie!
```

---

## Summary

| # | Class / Symbol | Module | Key pattern |
|---|---|---|---|
| 1 | `task` · `_TaskFunction` | `langgraph.func` | `@task` → `SyncAsyncFuture`; parallel futures inside `entrypoint` |
| 2 | `entrypoint` | `langgraph.func` | `previous` accumulation; `entrypoint.final`; `context_schema` injection |
| 3 | `LastValue` · `LastValueAfterFinish` | `langgraph.channels.last_value` | One-write-per-step; deferred visibility via `finish()/consume()` |
| 4 | `BaseChannel` | `langgraph.channels.base` | `Value/Update/Checkpoint` generics; custom channel protocol |
| 5 | `AgentState` · `AgentStatePydantic` | `langgraph.prebuilt.chat_agent_executor` | `add_messages` reducer; `RemainingSteps`; `IsLastStep` guard |
| 6 | `ToolCallRequest` · `ToolCallWrapper` | `langgraph.prebuilt.tool_node` | Immutable `override()`; interceptor chains; retry wrappers |
| 7 | `ToolCallWithContext` · `ToolInvocationError` | `langgraph.prebuilt.tool_node` | `Send`-based parallel dispatch; LLM-filtered validation errors |
| 8 | `ValidationNode` | `langgraph.prebuilt.tool_validator` | Deprecated structured-output validator; `response_format` migration |
| 9 | `Pregel` | `langgraph.pregel.main` | BSP runtime; `get_state_history`; `update_state`; topology export |
| 10 | `RunnableSeq` · `_RunnableWithWriter` · `_RunnableWithStore` | `langgraph._internal._runnable` | Node pipelines; mid-node streaming; store access via protocol |

---
title: "Runtime, ToolRuntime & Managed Values — API reference"
description: "Runtime context injection, ToolRuntime for tools, ExecutionInfo, IsLastStep, RemainingSteps — how to access execution metadata and inject immutable context into nodes and tools."
framework: langgraph
language: python
sidebar:
  label: "Ref · Runtime & Managed Values"
  order: 39
---

# Runtime, ToolRuntime & Managed Values

Verified against **`langgraph==1.2.1`** (modules: `langgraph.runtime`, `langgraph.prebuilt.tool_node`, `langgraph.managed.is_last_step`).

This page covers the three mechanisms LangGraph provides for injecting execution context into node and tool functions without threading values through graph state:

| Mechanism | Where it applies | Added in |
|---|---|---|
| `Runtime[ContextT]` | Node functions | v0.6.0 |
| `ToolRuntime[ContextT, StateT]` | Tool functions called by `ToolNode` | v0.6.0 |
| `IsLastStep` / `RemainingSteps` | State schema fields (managed values) | v0.3 / v1.2 |

---

## `Runtime[ContextT]` — node-level injection

`Runtime` is a dataclass injected by the Pregel executor whenever a node function declares a `runtime` parameter. It bundles together every piece of execution context a node might need: the typed context object, the store, a stream writer, the previous state snapshot, rich execution metadata, and cooperative drain control.

### Class definition

```python
# langgraph.runtime
from dataclasses import dataclass
from typing import Generic, TypeVar, Any

ContextT = TypeVar("ContextT")

@dataclass
class Runtime(Generic[ContextT]):
    context: ContextT | None        # type-safe context from context_schema
    store: BaseStore | None         # graph's persistent store
    stream_writer: StreamWriter     # write events to the custom stream channel
    previous: Any                   # previous state (used for checkpointing diffs)
    execution_info: ExecutionInfo   # metadata about the current execution
    server_info: ServerInfo | None  # LangSmith Server info; None in OSS LangGraph
    control: RunControl | None      # cooperative drain / pause control
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `context` | `ContextT \| None` | The typed context object passed in `configurable["context"]`. `None` if no `context_schema` was set on the graph. |
| `store` | `BaseStore \| None` | The graph's persistent store (e.g. `InMemoryStore`, Postgres-backed store). `None` if no store was provided at compile time. |
| `stream_writer` | `StreamWriter` | Callable that emits a value on the `"custom"` stream channel. Equivalent to injecting `writer: StreamWriter` directly. |
| `previous` | `Any` | The previous channel state snapshot before this step ran. Useful for computing diffs without re-reading from the checkpointer. |
| `execution_info` | `ExecutionInfo` | Structured metadata: checkpoint ID, thread ID, task ID, run ID, retry count, and first-attempt timestamp. |
| `server_info` | `ServerInfo \| None` | Populated when running inside a LangSmith-hosted deployment. Always `None` in open-source LangGraph. |
| `control` | `RunControl \| None` | Cooperative drain handle. Allows a node to signal that it is safe to pause or cancel the run mid-execution. |

> **`config` is not on `Runtime`.** If you need `RunnableConfig` (e.g. to pass to `ChatModel.invoke`), add a separate `config: RunnableConfig` parameter to the node. Both parameters can coexist.

### Injection pattern

Declare `runtime` as a typed parameter. LangGraph matches by the `Runtime` annotation — the parameter name itself does not matter, but `runtime` is conventional.

```python
from dataclasses import dataclass
from typing import Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


@dataclass
class AppContext:
    user_id: str
    tenant_id: str


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def personalized_response(
    state: State,
    runtime: Runtime[AppContext],
    config: RunnableConfig,          # separate param — not on Runtime
) -> dict:
    # Type-safe context access
    user_id = runtime.context.user_id
    tenant_id = runtime.context.tenant_id

    # Read from the store
    prefs = runtime.store.get(("users", user_id), "preferences")
    theme = prefs.value.get("theme", "light") if prefs else "light"

    # Emit a progress event on the custom stream channel
    runtime.stream_writer({"status": "processing", "user": user_id})

    # Execution metadata
    print(f"attempt #{runtime.execution_info.node_attempt}, thread={runtime.execution_info.thread_id}")

    return {"messages": [AIMessage(f"Hello {user_id} (theme: {theme})")]}


store = InMemoryStore()
builder = StateGraph(State, context_schema=AppContext)
builder.add_node("respond", personalized_response)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
graph = builder.compile(checkpointer=InMemorySaver(), store=store)

# Pass context at call time via configurable
result = graph.invoke(
    {"messages": [HumanMessage("Hello")]},
    {"configurable": {
        "thread_id": "t1",
        "context": AppContext(user_id="u123", tenant_id="acme"),
    }},
)
```

---

## `ExecutionInfo` — execution metadata

`ExecutionInfo` is a frozen dataclass attached to `runtime.execution_info`. It gives structured access to IDs and counters that are otherwise scattered across `RunnableConfig` and the Pregel scratchpad.

### Class definition

```python
# langgraph.runtime
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    checkpoint_id: str
    checkpoint_ns: str
    task_id: str
    thread_id: str | None               # None if no checkpointer is attached
    run_id: str | None                  # None if not set in RunnableConfig
    node_attempt: int = 1               # 1-indexed retry count (1 = first attempt)
    node_first_attempt_time: float | None = None  # unix timestamp of first attempt
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `checkpoint_id` | `str` | ID of the checkpoint written after the previous step. |
| `checkpoint_ns` | `str` | Namespace of the checkpoint, used to isolate subgraphs. |
| `task_id` | `str` | ID of the Pregel task executing this node invocation. |
| `thread_id` | `str \| None` | Conversation thread identifier. `None` when no checkpointer is attached. |
| `run_id` | `str \| None` | Run ID from `RunnableConfig`. `None` if not supplied by the caller. |
| `node_attempt` | `int` | How many times this node has been attempted for the current step (1-indexed). Increments on retries. |
| `node_first_attempt_time` | `float \| None` | Unix timestamp of the very first attempt. Useful for computing total time spent across retries. |

### Usage example

```python
from langgraph.runtime import Runtime


def resilient_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info

    if info.node_attempt > 1:
        elapsed = time.time() - info.node_first_attempt_time
        print(f"Retry #{info.node_attempt} after {elapsed:.1f}s on thread {info.thread_id}")

    # Use task_id as an idempotency key for external API calls
    result = call_external_api(
        idempotency_key=info.task_id,
        payload=state["query"],
    )
    return {"result": result}
```

---

## `get_runtime()` — context-manager alternative

`get_runtime()` retrieves the current `Runtime` from a context variable set by the executor. It is an alternative to parameter injection for cases where you cannot add parameters to the function signature (e.g. when wrapping third-party code).

```python
# langgraph.runtime
from langgraph.runtime import get_runtime
```

### Usage

```python
from langgraph.runtime import get_runtime


def my_node(state: State) -> dict:
    # Equivalent to declaring `runtime: Runtime` as a parameter
    runtime = get_runtime()
    user_id = runtime.context.user_id if runtime.context else None
    runtime.stream_writer({"event": "started"})
    return {}
```

`get_runtime()` raises a `RuntimeError` if called outside of a LangGraph node execution context (i.e., outside an active Pregel task). Prefer parameter injection when the function signature is under your control — it is more explicit and easier to test.

---

## `ToolRuntime[ContextT, StateT]` — tool-level injection

`ToolRuntime` is a separate dataclass for **tool functions** invoked by `ToolNode`. It is distinct from `Runtime` — it provides the tool with access to the current graph state, the triggering tool-call ID, the store, and the typed context, but not execution metadata or drain control.

### Class definition

```python
# langgraph.prebuilt.tool_node
from dataclasses import dataclass
from typing import Generic, TypeVar

ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")

@dataclass
class ToolRuntime(Generic[ContextT, StateT]):
    state: StateT | None                    # current graph state snapshot
    tool_call_id: str | None                # ID of the ToolCall being executed
    config: RunnableConfig | None           # runnable config passed to the tool
    store: BaseStore | None                 # graph's persistent store
    context: ContextT | None               # type-safe context from context_schema
    stream_writer: StreamWriter | None      # write events to the custom stream channel
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `state` | `StateT \| None` | The current graph state at the time the tool is called. Lets tools read state without passing it through the tool call arguments. |
| `tool_call_id` | `str \| None` | The ID of the `ToolCall` message that triggered this tool invocation. |
| `config` | `RunnableConfig \| None` | The full `RunnableConfig`. Unlike `Runtime`, `ToolRuntime` includes `config` directly. |
| `store` | `BaseStore \| None` | The graph's persistent store. |
| `context` | `ContextT \| None` | The typed context object from `configurable["context"]`. |
| `stream_writer` | `StreamWriter \| None` | Emit events on the custom stream channel. |

### Usage example

```python
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime


@dataclass
class AppContext:
    user_id: str


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_tier: str


@tool
def fetch_user_data(query: str, runtime: ToolRuntime[AppContext, State]) -> str:
    """Fetch data for the current user."""
    # Read typed context
    user_id = runtime.context.user_id

    # Read graph state directly — no need to pass it via tool arguments
    tier = runtime.state["user_tier"] if runtime.state else "free"

    # Persist a result to the store
    runtime.store.put(
        ("users", user_id, "searches"),
        runtime.tool_call_id,
        {"query": query, "tier": tier},
    )

    # Emit a streaming event
    if runtime.stream_writer:
        runtime.stream_writer({"tool": "fetch_user_data", "query": query})

    return f"Results for {user_id} (tier={tier}): ..."


tool_node = ToolNode([fetch_user_data])
```

---

## `Runtime` vs `ToolRuntime` comparison

| Attribute | `Runtime[ContextT]` | `ToolRuntime[ContextT, StateT]` |
|---|---|---|
| **Used in** | Node functions | Tool functions (via `ToolNode`) |
| **Import** | `langgraph.runtime` | `langgraph.prebuilt.tool_node` |
| **`context`** | Yes | Yes |
| **`store`** | Yes | Yes |
| **`stream_writer`** | Yes (never `None`) | Yes (`None` if not available) |
| **`config`** | **No** — add `config: RunnableConfig` separately | Yes (included directly) |
| **`state`** | No — read from the `state` parameter | Yes — current graph state snapshot |
| **`tool_call_id`** | No | Yes |
| **`execution_info`** | Yes (`ExecutionInfo` dataclass) | No |
| **`previous`** | Yes — previous state snapshot | No |
| **`server_info`** | Yes (`None` in OSS) | No |
| **`control`** | Yes (`RunControl`) | No |
| **Generic type params** | `Runtime[ContextT]` | `ToolRuntime[ContextT, StateT]` |
| **Added in** | v0.6.0 | v0.6.0 |

---

## Managed values — `IsLastStep` & `RemainingSteps`

**Managed values** are state-field annotations that LangGraph fills in automatically from the Pregel executor's scratchpad. They are declared in the state schema like any other field, but the graph — not node code — writes them at each step. Nodes read them as ordinary state fields.

Two managed values ship with LangGraph out of the box:

| Type alias | Module | Type | Value |
|---|---|---|---|
| `IsLastStep` | `langgraph.managed.is_last_step` | `bool` | `True` when `step == recursion_limit - 1` |
| `RemainingSteps` | `langgraph.managed.is_last_step` | `int` | `recursion_limit - current_step` |

Both are `Annotated[T, ManagedValueManager]` type aliases. The graph rewrites them before every step — nodes must never write to them.

### Imports

```python
# Preferred import path (re-exported from the public managed package)
from langgraph.managed import IsLastStep, RemainingSteps

# Direct module import (also valid)
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps
```

### Minimal runnable example

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last_step: IsLastStep     # bool — injected by the graph
    remaining: RemainingSteps    # int  — injected by the graph


def my_node(state: State) -> dict:
    if state["is_last_step"]:
        return {"messages": [AIMessage("Max steps reached — returning early.")]}
    print(f"Steps remaining: {state['remaining']}")
    return {"messages": [AIMessage("Still going...")]}


builder = StateGraph(State)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_conditional_edges("my_node", lambda s: END if s["is_last_step"] else "my_node")

graph = builder.compile()
graph.invoke({"messages": [HumanMessage("Start")]})
```

### `IsLastStep`

```python
IsLastStep = Annotated[bool, IsLastStepManager]
```

`IsLastStep` is `True` exactly when `current_step == recursion_limit - 1`. Use it to detect imminent recursion-limit exhaustion and return a graceful partial result instead of raising `GraphRecursionError`.

```python
from langgraph.managed import IsLastStep


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last: IsLastStep


def agent(state: AgentState) -> dict:
    if state["is_last"]:
        return {"messages": [AIMessage("[truncated: recursion limit reached]")]}
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

Step-count details:

- Default recursion limit: **25** steps.
- Override per-call: `graph.invoke(input, {"recursion_limit": 50})`.
- `IsLastStep` becomes `True` at step **24** (default) or step **49** (`recursion_limit=50`).

### `RemainingSteps`

```python
RemainingSteps = Annotated[int, RemainingStepsManager]
```

`RemainingSteps` returns `recursion_limit - current_step` — how many steps are left. It decrements by 1 each step.

```python
from langgraph.managed import RemainingSteps


class PipelineState(TypedDict):
    items: list[str]
    processed: list[str]
    steps_left: RemainingSteps


def process_one(state: PipelineState) -> dict:
    if state["steps_left"] <= 2:
        # Flush remaining items — not enough steps to process individually
        return {"processed": state["processed"] + [f"[skipped: {len(state['items'])} items]"]}
    first, *rest = state["items"]
    return {"items": rest, "processed": state["processed"] + [first.upper()]}
```

---

## Import reference

| Symbol | Canonical import | Notes |
|---|---|---|
| `Runtime` | `from langgraph.runtime import Runtime` | |
| `get_runtime` | `from langgraph.runtime import get_runtime` | |
| `ExecutionInfo` | `from langgraph.runtime import ExecutionInfo` | Attached as `runtime.execution_info` |
| `ToolRuntime` | `from langgraph.prebuilt.tool_node import ToolRuntime` | |
| `IsLastStep` | `from langgraph.managed import IsLastStep` | Also at `langgraph.managed.is_last_step` |
| `RemainingSteps` | `from langgraph.managed import RemainingSteps` | Also at `langgraph.managed.is_last_step` |
| `StreamWriter` | `from langgraph.types import StreamWriter` | Type of `runtime.stream_writer` |
| `BaseStore` | `from langgraph.store.base import BaseStore` | Type of `runtime.store` |
| `InMemoryStore` | `from langgraph.store.memory import InMemoryStore` | Concrete store for local dev |
| `InMemorySaver` | `from langgraph.checkpoint.memory import InMemorySaver` | Concrete checkpointer for local dev |
| `context_schema` | `StateGraph(..., context_schema=MyContext)` | Graph constructor kwarg |

---

## Patterns

### ReAct agent with `Runtime` context and store

```python
from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


@dataclass
class UserContext:
    user_id: str
    locale: str = "en"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last: IsLastStep


def call_agent(state: AgentState, runtime: Runtime[UserContext]) -> dict:
    if state["is_last"]:
        return {"messages": [AIMessage("Step limit reached.")]}

    user_id = runtime.context.user_id
    history = runtime.store.get(("sessions", user_id), "history")
    prev_messages = history.value if history else []

    runtime.stream_writer({"event": "agent_start", "user": user_id})

    response = llm_with_tools.invoke(state["messages"] + prev_messages)

    runtime.store.put(
        ("sessions", user_id),
        "history",
        {"last_response": response.content},
    )
    return {"messages": [response]}


def router(state: AgentState) -> str:
    if state["is_last"]:
        return END
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


store = InMemoryStore()
builder = StateGraph(AgentState, context_schema=UserContext)
builder.add_node("agent", call_agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver(), store=store)

result = graph.invoke(
    {"messages": [HumanMessage("What is the weather?")]},
    {"configurable": {
        "thread_id": "session-42",
        "context": UserContext(user_id="alice", locale="en-GB"),
    }},
)
```

### Retry-aware node using `ExecutionInfo`

```python
import time
from langgraph.runtime import Runtime


def idempotent_node(state: State, runtime: Runtime) -> dict:
    info = runtime.execution_info

    if info.node_attempt > 1:
        elapsed = time.time() - (info.node_first_attempt_time or time.time())
        print(f"Retry {info.node_attempt} for task {info.task_id} after {elapsed:.1f}s")

    # Use task_id as idempotency key — safe to retry
    result = post_to_external_api(
        idempotency_key=info.task_id,
        data=state["payload"],
    )
    return {"result": result}
```

### Tool with `ToolRuntime` accessing state and context

```python
from dataclasses import dataclass
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolRuntime


@dataclass
class AppContext:
    api_key: str


class MyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    account_id: str


@tool
def look_up_account(query: str, runtime: ToolRuntime[AppContext, MyState]) -> str:
    """Look up account information."""
    # Context gives the API key without putting it in state
    api_key = runtime.context.api_key

    # State gives the current account_id without passing it via tool args
    account_id = runtime.state["account_id"] if runtime.state else "unknown"

    if runtime.stream_writer:
        runtime.stream_writer({"tool": "look_up_account", "account": account_id})

    return call_accounts_api(api_key=api_key, account_id=account_id, query=query)


tool_node = ToolNode([look_up_account])
```

### `get_runtime()` for code you cannot annotate

```python
from langgraph.runtime import get_runtime


def third_party_wrapper(state: State) -> dict:
    # This function signature is fixed — cannot add a runtime param
    runtime = get_runtime()
    runtime.stream_writer({"event": "third_party_start"})
    result = run_third_party_logic(state["data"])
    return {"result": result}
```

### Combining `RemainingSteps` with `Runtime`

```python
from langgraph.managed import RemainingSteps
from langgraph.runtime import Runtime


class PipelineState(TypedDict):
    stages: list[str]
    output: str
    steps_left: RemainingSteps


def pipeline_node(state: PipelineState, runtime: Runtime[AppContext]) -> dict:
    # Bail out early if steps are running low
    if state["steps_left"] <= 2:
        runtime.stream_writer({"warning": "low_steps", "remaining": state["steps_left"]})
        return {"output": state["output"] + " [pipeline aborted: low steps]"}

    stage, *rest = state["stages"]
    result = run_stage(stage, context=runtime.context)
    return {"stages": rest, "output": state["output"] + f"\n{stage}: {result}"}
```

---

## How managed values work (internals)

Each managed value is a subclass of `ManagedValue[T]` with a `get(scratchpad)` static method. The Pregel executor calls `get` before every step and injects the return value into the state the node sees — but does **not** persist it to a channel (so it never appears in checkpoints or reducer chains).

```python
# Simplified internals — do not import these directly
from langgraph._internal._scratchpad import PregelScratchpad

class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step
```

`PregelScratchpad.stop` is the recursion limit; `PregelScratchpad.step` is the 0-indexed current step.

---

## Gotchas

- **`config` is not on `Runtime`.** Add `config: RunnableConfig` as a separate parameter if you need it alongside `runtime: Runtime`.

- **`ToolRuntime` is not `Runtime`.** They are different classes from different modules. A node annotated with `runtime: ToolRuntime` will not receive injection — it must use `Runtime`. The inverse also holds.

- **Managed values are read-only.** Any node return dict that includes `IsLastStep` or `RemainingSteps` keys is silently ignored — the graph overwrites them before the next node sees them.

- **Managed values do not appear in checkpoints.** They are reconstructed from the scratchpad at runtime. You cannot read them from `StateSnapshot` or `get_state_history` results.

- **Provide a default for managed value fields.** Since they are never in the initial `invoke` input, declare them with a default matching their type:
  ```python
  class State(TypedDict, total=False):
      is_last_step: IsLastStep       # total=False makes the field optional
  # or with a dataclass / Pydantic model:
  class State(BaseModel):
      is_last_step: IsLastStep = False
      remaining: RemainingSteps = 25
  ```

- **`recursion_limit` is per-invoke, not per-graph.** Different calls to `graph.invoke` can use different limits. `IsLastStep` and `RemainingSteps` track whichever limit was active when the run started.

- **Step counter resets on each `invoke`.** Checkpointers save channel values but not the step counter. A new `invoke` on an existing thread always starts the step counter at 0.

- **`get_runtime()` raises outside a task context.** Do not call it from module-level code, background threads, or any code path that runs outside the Pregel executor.

---

## Version history

| Version | Change |
|---|---|
| 1.2.1 | `ExecutionInfo.node_first_attempt_time` field added. |
| 1.2 | `RemainingSteps` added alongside the existing `IsLastStep`. Both re-exported from `langgraph.managed`. |
| 0.6.0 | `Runtime`, `ToolRuntime`, and `get_runtime()` introduced. |
| 1.0 | `IsLastStep` and `RemainingSteps` moved from `langgraph.managed` to `langgraph.managed.is_last_step`; old import path still re-exported. |
| 0.3 | `IsLastStep` introduced as the first managed value. |

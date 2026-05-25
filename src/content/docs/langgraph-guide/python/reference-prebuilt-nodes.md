---
title: "ToolNode, InjectedState, InjectedStore, ToolRuntime, ToolCallTransformer — API reference"
description: "The prebuilt ToolNode executor, state/store injection annotations, ToolRuntime context, tools_condition router, ToolCallRequest interceptor, and ToolCallTransformer/ToolCallStream for per-tool streaming — with source-verified signatures for langgraph==1.2.1."
framework: langgraph
language: python
sidebar:
  label: "Ref · Prebuilt nodes"
  order: 36
---

# ToolNode, InjectedState, InjectedStore, ToolRuntime, ToolCallTransformer — API reference

Verified against **`langgraph==1.2.1`** / **`langgraph-prebuilt==1.1.0`** (modules: `langgraph.prebuilt.tool_node`, `langgraph.prebuilt.tool_validator`, `langgraph.prebuilt._tool_call_transformer`, `langgraph.prebuilt._tool_call_stream`).

`ToolNode` is LangGraph's prebuilt executor that takes a list of tools, reads the last AI message in state, runs every pending tool call in parallel, and writes back `ToolMessage` results. The surrounding helpers — `InjectedState`, `InjectedStore`, `ToolRuntime`, `tools_condition`, `ToolCallRequest`, `ToolCallTransformer`, and `ToolCallStream` — let tools read graph state, access the long-term store, stream partial output, intercept calls before execution, and consume per-tool-call streaming results in a structured way.

## Minimal runnable example

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


tools = [multiply]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
tool_node = ToolNode(tools)


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}
result = graph.invoke({"messages": [("user", "What is 6 times 7?")]}, config)
print(result["messages"][-1].content)  # 42
```

## Imports at a glance

| Symbol | Import path |
|---|---|
| `ToolNode` | `langgraph.prebuilt.tool_node` (also re-exported from `langgraph.prebuilt`) |
| `tools_condition` | `langgraph.prebuilt.tool_node` (also re-exported from `langgraph.prebuilt`) |
| `InjectedState` | `langgraph.prebuilt.tool_node` (also re-exported from `langgraph.prebuilt`) |
| `InjectedStore` | `langgraph.prebuilt.tool_node` (also re-exported from `langgraph.prebuilt`) |
| `ToolRuntime` | `langgraph.prebuilt.tool_node` |
| `ToolCallRequest` | `langgraph.prebuilt.tool_node` |
| `ToolCallTransformer` | `langgraph.prebuilt._tool_call_transformer` |
| `ToolCallStream` | `langgraph.prebuilt._tool_call_stream` |
| `ValidationNode` | `langgraph.prebuilt.tool_validator` (deprecated) |
| `MessagesState` | `langgraph.graph.message` |

The top-level `langgraph.prebuilt.__init__` re-exports `ToolNode`, `tools_condition`, `InjectedState`, and `InjectedStore`. `ToolRuntime`, `ToolCallRequest`, `ToolCallTransformer`, and `ToolCallStream` must be imported from their specific sub-modules directly.

## `ToolNode`

### Constructor

```python
ToolNode(
    tools: Sequence[BaseTool | Callable],
    *,
    name: str = "tools",
    tags: list[str] | None = None,
    handle_tool_errors: bool | str | Callable[..., str] | type[Exception] | tuple[type[Exception], ...] = True,
    messages_key: str = "messages",
    wrap_tool_call: ToolCallWrapper | None = None,
    awrap_tool_call: AsyncToolCallWrapper | None = None,
)
```

- `tools` — list of `BaseTool` instances or plain callables decorated with `@tool`. Callables are wrapped automatically.
- `name` — node name as registered in the graph. Defaults to `"tools"`.
- `tags` — LangChain run tags forwarded to each tool invocation for tracing.
- `handle_tool_errors` — controls exception handling; see the table below. Default is `True`.
- `messages_key` — the state key that holds the message list. Only relevant when state is a dict; ignored for list/direct-tool-call input formats.
- `wrap_tool_call` — sync interceptor called for every tool call before execution; receives a `ToolCallRequest` and an `execute` callable.
- `awrap_tool_call` — async variant of `wrap_tool_call`. Falls back to `wrap_tool_call` when not set.

### Property

```python
tool_node.tools_by_name  # dict[str, BaseTool]
```

Read-only mapping from tool name to the resolved `BaseTool`. Useful for inspecting schemas at startup.

### Input formats

`ToolNode` accepts three input formats:

**Format 1 — dict with messages key (most common)**

```python
{"messages": [AIMessage(content="...", tool_calls=[...])]}
```

The last message in the list is inspected for `tool_calls`. State dicts are the standard format when using `MessagesState` or any `TypedDict` state schema.

**Format 2 — list of messages**

```python
[AIMessage(content="...", tool_calls=[...])]
```

The node detects a list and treats it as the message sequence directly. Returns a list of `ToolMessage` objects (not a dict).

**Format 3 — direct tool calls**

```python
[{"name": "multiply", "args": {"a": 6, "b": 7}, "id": "tc_001", "type": "tool_call"}]
```

A list of raw tool-call dicts. The node skips message parsing and executes each entry directly.

### Output formats

| Input format | Normal tool output | Command tool output |
|---|---|---|
| Dict with messages key | `{"messages": [ToolMessage(...), ...]}` | `[Command(...), ...]` or mixed list |
| List of messages | `[ToolMessage(...), ...]` | `[Command(...), ...]` or mixed list |
| Direct tool calls | `[ToolMessage(...), ...]` | `[Command(...), ...]` or mixed list |

When any tool returns a `Command`, the node returns a list that may mix `ToolMessage` and `Command` objects. The graph runtime handles routing those commands to the appropriate next nodes.

### `handle_tool_errors`

| Value | Behavior |
|---|---|
| `True` | Catch all exceptions; return a default error string in a `ToolMessage` |
| `str` | Catch all exceptions; return this exact string as the error `ToolMessage` content |
| `type[Exception]` | Only catch this exception type; all others propagate |
| `tuple[type[Exception], ...]` | Only catch these exception types; all others propagate |
| `Callable[..., str]` | Catch all exceptions; call this formatter with `(exception,)` and use the return value as the error string |
| `False` | Disable error handling entirely; all exceptions propagate to the graph |

The default error template is: `"Error: {exception_repr}\n Please fix your mistakes."`.

### `wrap_tool_call` / `awrap_tool_call`

Both take a `ToolCallWrapper` signature:

```python
# sync
def my_wrapper(request: ToolCallRequest, execute: Callable) -> ToolMessage | Command:
    ...

# async
async def my_async_wrapper(request: ToolCallRequest, execute: Callable) -> ToolMessage | Command:
    ...
```

The `execute` callable runs the original tool and returns a `ToolMessage | Command`. You may call it, skip it, or replace it entirely. `awrap_tool_call` is used when the graph runs in async mode. If `awrap_tool_call` is not set, async execution falls back to `wrap_tool_call`.

## `ToolCallRequest`

A dataclass representing a single pending tool call as seen by `wrap_tool_call`.

```python
from dataclasses import dataclass
from langgraph.prebuilt.tool_node import ToolCallRequest

@dataclass
class ToolCallRequest:
    tool_call: ToolCall          # the raw tool call dict from the AI message
    tool: BaseTool | None        # resolved BaseTool, or None if name not found
    state: Any                   # current graph state (same object passed to ToolNode)
    runtime: ToolRuntime         # full runtime context (see below)
```

### `override(**overrides) -> ToolCallRequest`

Returns a new `ToolCallRequest` with the specified fields replaced. Direct attribute assignment on `ToolCallRequest` instances is deprecated — use `override` instead:

```python
modified = request.override(
    tool_call={**request.tool_call, "args": {"a": 10, "b": 2}}
)
result = execute(modified)
```

## `tools_condition`

```python
from langgraph.prebuilt import tools_condition

def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]
```

A built-in routing function for `add_conditional_edges`. Inspects the last message in state:

- Returns `"tools"` if the last `AIMessage` has a non-empty `tool_calls` list.
- Returns `"__end__"` otherwise.

```python
builder.add_conditional_edges("agent", tools_condition)
```

When `state` is a dict, `messages_key` controls which key holds the list (default `"messages"`). When `state` is a list, it is used directly. When `state` is a Pydantic model, the `messages` attribute is accessed.

> **Important:** `tools_condition` always returns the literal string `"tools"` or `"__end__"`. Your `ToolNode` must be registered with `name="tools"` (the default). If you name the node differently, add a `path_map` to remap the return value:
>
> ```python
> builder.add_conditional_edges(
>     "agent",
>     tools_condition,
>     {"tools": "my_tool_executor", "__end__": END},
> )
> ```

## `InjectedState`

```python
from langgraph.prebuilt import InjectedState

class InjectedState(InjectedToolArg):
    def __init__(self, field: str | None = None) -> None: ...
```

Annotates a tool parameter so that `ToolNode` injects the current graph state automatically. The argument is hidden from the model's tool schema — the LLM never sees or fills it.

**Inject the entire state dict:**

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState

@tool
def get_user_info(user_id: str, state: Annotated[dict, InjectedState()]) -> str:
    """Look up user info by ID."""
    return state.get("user_data", {}).get(user_id, "not found")
```

**Inject a single field from state:**

```python
@tool
def check_balance(state: Annotated[float, InjectedState("account_balance")]) -> str:
    """Check the current account balance."""
    return f"Balance: {state:.2f}"
```

When `InjectedState("foo")` is used, the tool receives `state["foo"]` directly, not the full dict. No `Annotated` nesting beyond the outer annotation is needed.

## `InjectedStore`

```python
from langgraph.prebuilt import InjectedStore

class InjectedStore(InjectedToolArg): ...
```

Annotates a tool parameter so that `ToolNode` injects the long-term `BaseStore` compiled into the graph. Like `InjectedState`, the parameter is hidden from the model's schema.

```python
from typing import Annotated
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

@tool
def save_preference(key: str, value: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Save a user preference to long-term memory."""
    store.put(("prefs",), key, {"value": value})
    return f"Saved {key}={value}"
```

Requirements:

- `langchain-core >= 0.3.8`
- Graph must be compiled with `store=` — e.g., `builder.compile(store=InMemoryStore())`

If the graph has no store compiled in, the injected value is `None` and tools that depend on it will fail at runtime.

## `ToolRuntime`

A dataclass injected into tools that declare a `runtime` parameter with type `ToolRuntime`. No `Annotated` wrapper is needed — `ToolNode` detects the parameter name `runtime` and the type annotation.

```python
from dataclasses import dataclass, field
from langgraph.prebuilt.tool_node import ToolRuntime

@dataclass
class ToolRuntime:
    state: StateT
    context: ContextT
    config: RunnableConfig
    stream_writer: StreamWriter
    tool_call_id: str | None
    store: BaseStore | None
    tools: list[BaseTool] = field(default_factory=list)
    execution_info: ExecutionInfo | None = None
    server_info: ServerInfo | None = None
```

Fields:

- `state` — the full current graph state, same as `InjectedState()`.
- `context` — the run-level context passed to `invoke(context=...)` (from `context_schema`).
- `config` — the `RunnableConfig` for the current run (thread\_id, callbacks, etc.).
- `stream_writer` — callable that writes deltas to the `tools` stream channel; see `emit_output_delta`.
- `tool_call_id` — the `id` field of the tool call currently being executed.
- `store` — the `BaseStore` compiled into the graph, or `None`.
- `tools` — the list of tools registered with this `ToolNode`.
- `execution_info` — `ExecutionInfo` with `checkpoint_id`, `thread_id`, `run_id`, `node_attempt`, `node_first_attempt_time`. Set by LangGraph Platform; `None` locally.
- `server_info` — `ServerInfo` set by LangGraph Platform only; `None` locally.

### `emit_output_delta`

```python
runtime.emit_output_delta(delta: Any) -> None
```

Streams a partial output chunk to the `tools` channel. Callers consuming `stream_mode="tools"` will receive each delta as it is emitted. This is a no-op if the graph was not invoked with `stream_mode="tools"` (or a superset that includes it).

`ToolRuntime` is tool-specific. It is distinct from `Runtime` (from `langgraph.runtime`), which is injected into graph **nodes**. Do not annotate node parameters with `ToolRuntime`.

## `ValidationNode` (deprecated)

```python
from langgraph.prebuilt.tool_validator import ValidationNode
```

Deprecated since **v1.0**. Validates tool call schemas without executing the tools. It was designed for structured output and extraction workflows where the LLM needs to be re-prompted when its tool arguments fail schema validation.

Migrate to `create_agent` from `langchain.agents`, which handles schema validation internally.

## Patterns

### 1. Basic ReAct loop

```python
from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


tools = [add, subtract]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def agent(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}
for chunk in graph.stream({"messages": [("user", "Add 3 and 4, then subtract 2")]}, config):
    print(chunk)
```

### 2. `handle_tool_errors` with a callable formatter

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


def format_error(exc: Exception) -> str:
    return f"Tool failed with {type(exc).__name__}: {exc}. Please retry with valid inputs."


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


tool_node = ToolNode(
    [divide],
    handle_tool_errors=format_error,
)
```

### 3. `InjectedState` — full state and field-only injection

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langchain_openai import ChatOpenAI


class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_name: str


# Inject the entire state
@tool
def greet_user(state: Annotated[dict, InjectedState()]) -> str:
    """Greet the current user by name."""
    return f"Hello, {state.get('user_name', 'stranger')}!"


# Inject a single field from state
@tool
def greet_field(name: Annotated[str, InjectedState("user_name")]) -> str:
    """Greet by injected user_name field."""
    return f"Hi, {name}!"


tools = [greet_user, greet_field]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def agent(state: AppState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(AppState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()
graph.invoke({"messages": [("user", "Greet me!")], "user_name": "Alice"})
```

### 4. `InjectedStore` — read/write long-term memory from a tool

```python
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition, InjectedStore
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI


@tool
def remember(key: str, value: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Store a fact for later."""
    store.put(("memory",), key, {"value": value})
    return f"Remembered: {key} = {value}"


@tool
def recall(key: str, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Recall a previously stored fact."""
    item = store.get(("memory",), key)
    return item.value["value"] if item else f"Nothing stored for {key}"


tools = [remember, recall]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
memory_store = InMemoryStore()


def agent(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# store= is required when any tool uses InjectedStore
graph = builder.compile(store=memory_store)
graph.invoke({"messages": [("user", "Remember that the answer is 42")]})
```

### 5. `ToolRuntime` — accessing state, streaming deltas, and reading `tool_call_id`

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI
import time


@tool
def slow_counter(n: int, runtime: ToolRuntime) -> str:
    """Count to n slowly, streaming each tick."""
    for i in range(1, n + 1):
        # emit_output_delta streams partial output when stream_mode includes "tools"
        runtime.emit_output_delta({"tick": i})
        time.sleep(0.1)
    # tool_call_id is the ID from the AIMessage tool_calls entry
    return f"Finished counting to {n} (call_id={runtime.tool_call_id})"


@tool
def read_state(runtime: ToolRuntime) -> str:
    """Return the number of messages in state."""
    msgs = runtime.state.get("messages", [])
    return f"State has {len(msgs)} messages"


tools = [slow_counter, read_state]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def agent(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# stream_mode="tools" receives deltas from emit_output_delta
for chunk in graph.stream(
    {"messages": [("user", "Count to 3")]},
    stream_mode=["updates", "tools"],
):
    print(chunk)
```

### 6. `wrap_tool_call` — interceptor that logs and modifies tool args

```python
import logging
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolNode, ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


def audit_wrapper(request: ToolCallRequest, execute):
    """Log every tool call; clamp numeric args before execution."""
    tool_name = request.tool_call["name"]
    original_args = request.tool_call["args"]
    logger.info("Tool call: %s args=%s", tool_name, original_args)

    # Clamp any integer arg above 100
    new_args = {
        k: min(v, 100) if isinstance(v, int) else v
        for k, v in original_args.items()
    }

    if new_args != original_args:
        logger.warning("Clamped args for %s: %s -> %s", tool_name, original_args, new_args)
        # Use override() — direct attribute assignment is deprecated
        request = request.override(
            tool_call={**request.tool_call, "args": new_args}
        )

    result = execute(request)
    logger.info("Tool result: %s", result)
    return result


@tool
def power(base: int, exponent: int) -> int:
    """Raise base to exponent."""
    return base ** exponent


tool_node = ToolNode([power], wrap_tool_call=audit_wrapper)
```

### 7. Tool-based access control with `wrap_tool_call`

Block tool execution entirely based on runtime state, returning a `ToolMessage` with an error instead of calling the tool:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, ToolCallRequest
from langgraph.prebuilt import tools_condition


class AppState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_role: str   # "admin" or "user"


@tool
def delete_record(record_id: str) -> str:
    """Delete a record from the database."""
    return f"Deleted record {record_id}"


@tool
def read_record(record_id: str) -> str:
    """Read a record from the database."""
    return f"Record {record_id}: data..."


def make_authz_wrapper(state_key: str = "user_role", required_role: str = "admin"):
    """Factory for an authorization wrapper that checks a state field."""

    def authz_wrapper(request: ToolCallRequest, execute):
        tool_name = request.tool_call["name"]
        role = request.state.get(state_key, "user") if isinstance(request.state, dict) else "user"

        # Only allow admin operations for privileged tools
        if tool_name == "delete_record" and role != required_role:
            return ToolMessage(
                content=f"Access denied: {tool_name} requires role '{required_role}' (you have '{role}')",
                tool_call_id=request.tool_call["id"],
            )
        return execute(request)

    return authz_wrapper


tools = [delete_record, read_record]
tool_node = ToolNode(tools, wrap_tool_call=make_authz_wrapper())
```

### 8. Dynamic tool list using `ToolRuntime.tools`

Access the list of all tools registered with the `ToolNode` from inside a tool:

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime, ToolNode


@tool
def list_available_tools(runtime: ToolRuntime) -> list[str]:
    """Return the names of all tools the agent can use."""
    return [t.name for t in runtime.tools]


@tool
def get_tool_description(tool_name: str, runtime: ToolRuntime) -> str:
    """Get the description of a specific tool by name."""
    for t in runtime.tools:
        if t.name == tool_name:
            return t.description or "(no description)"
    return f"Tool '{tool_name}' not found"


tool_node = ToolNode([list_available_tools, get_tool_description])
```

### 9. Async `awrap_tool_call` — non-blocking wrapper

When your graph runs in async mode and the wrapper itself does async work (e.g., an auth check against a remote service), use `awrap_tool_call`:

```python
import asyncio
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolNode, ToolCallRequest


async def async_authz_wrapper(request: ToolCallRequest, execute) -> ToolMessage:
    """Async wrapper: check authorization over the network before executing."""
    tool_name = request.tool_call["name"]

    # Simulate an async permission check (e.g., fetch from an auth service)
    await asyncio.sleep(0)   # replace with: allowed = await authz_service.check(tool_name)
    allowed = True

    if not allowed:
        return ToolMessage(
            content=f"Permission denied for {tool_name}",
            tool_call_id=request.tool_call["id"],
        )
    return await execute(request)   # execute is also async here


@tool
async def async_fetch(url: str) -> str:
    """Fetch a URL asynchronously."""
    return f"Content from {url}"


tool_node = ToolNode(
    [async_fetch],
    awrap_tool_call=async_authz_wrapper,   # used when graph runs in async mode
)
```

## `ToolCallTransformer` + `ToolCallStream`

`ToolCallTransformer` (module: `langgraph.prebuilt._tool_call_transformer`) is a built-in **`StreamTransformer`** that turns the raw `tools`-channel protocol events emitted during graph streaming into convenient **`ToolCallStream`** handles — one per tool invocation. It lets you consume per-tool incremental output (delta streaming), final output, and errors in a structured way without parsing raw event dicts.

> **Note:** `ToolCallTransformer` is **not** a base class to subclass; it is a concrete transformer you register with `compile()`. The raw `tools` protocol events and `ToolCallTransformer` are both part of `stream_mode="tools"` — see also the [Streaming modes reference](./reference-streaming-modes/#stream_modetools--per-tool-call-streaming).

### Registration

Pass `ToolCallTransformer` (the class itself, not an instance) to `compile()`:

```python
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer

graph = builder.compile(transformers=[ToolCallTransformer])
```

After registration the `tools` stream channel emits `ToolCallStream` objects instead of raw event dicts when you include `"tools"` in `stream_mode`.

### `ToolCallStream` fields

| Field | Type | Description |
|---|---|---|
| `tool_call_id` | `str` | Matches the `tool_call_id` from the `AIMessage`. |
| `tool_name` | `str` | Name of the tool being invoked. |
| `input` | `dict \| None` | Input arguments as received by the tool (from `on_tool_start`). `None` if not captured. |
| `output_deltas` | `StreamChannel[Any]` | Channel of incremental delta chunks. Iterate sync or async as they arrive. |
| `output` | `Any` | Terminal output from `tool-finished`. `None` until the tool completes successfully. |
| `error` | `str \| None` | Terminal error message from `tool-error`. `None` until the tool fails. |
| `completed` | `bool` | `True` once either `tool-finished` or `tool-error` has been observed. |

### Synchronous example

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer


@tool
def search(query: str) -> str:
    """Search the web for a query."""
    return f"Results for: {query}"


tools = [search]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
tool_node = ToolNode(tools)


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# Register the transformer — this projects raw tools events into ToolCallStream handles
graph = builder.compile(transformers=[ToolCallTransformer])

config = {"configurable": {"thread_id": "t1"}}

for run in graph.stream(
    {"messages": [("user", "Search for LangGraph docs")]},
    config,
    stream_mode="tools",
):
    for tc_stream in run.tool_calls:
        print(f"→ Tool started: {tc_stream.tool_name} (id={tc_stream.tool_call_id})")
        print(f"  Input: {tc_stream.input}")

        # Consume any delta chunks as they arrive
        for delta in tc_stream:
            print(f"  delta: {delta!r}")

        # After iteration the terminal state is populated
        if tc_stream.error:
            print(f"  ERROR: {tc_stream.error}")
        else:
            print(f"  Output: {tc_stream.output}")
```

### Asynchronous example

```python
import asyncio
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer


@tool
async def async_search(query: str) -> str:
    """Async search tool."""
    return f"Async results for: {query}"


tools = [async_search]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
tool_node = ToolNode(tools)


async def call_model(state: MessagesState) -> dict:
    return {"messages": [await llm.ainvoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(transformers=[ToolCallTransformer])

config = {"configurable": {"thread_id": "async-1"}}


async def main():
    async for run in graph.astream(
        {"messages": [("user", "Search for async patterns")]},
        config,
        stream_mode="tools",
    ):
        async for tc_stream in run.tool_calls:
            print(f"→ {tc_stream.tool_name} started (id={tc_stream.tool_call_id})")
            async for delta in tc_stream:
                print(f"  delta: {delta!r}")
            if tc_stream.error:
                print(f"  ERROR: {tc_stream.error}")
            else:
                print(f"  Final: {tc_stream.output}")


asyncio.run(main())
```

### Combining `"tools"` with other stream modes

`ToolCallTransformer` works when `"tools"` is included in a list of stream modes:

```python
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer

graph = builder.compile(transformers=[ToolCallTransformer])
config = {"configurable": {"thread_id": "multi"}}

for mode, data in graph.stream(
    {"messages": [("user", "Tell me the result of 6 × 7")]},
    config,
    stream_mode=["updates", "tools"],
):
    if mode == "updates":
        # Normal state-delta events
        print(f"[update] {list(data.keys())}")
    elif mode == "tools":
        # data is a run-level object; iterate its tool_calls
        for tc in data.tool_calls:
            print(f"[tool]   {tc.tool_name} → {tc.output}")
```

### How it works internally

`ToolCallTransformer` subscribes to the `tools` channel and handles four protocol events:

| Event | Action |
|---|---|
| `tool-started` | Creates a new `ToolCallStream(tool_call_id, tool_name, input)` and yields it on `run.tool_calls`. |
| `tool-output-delta` | Calls `tc_stream._push_delta(payload)` — the delta is queued on `output_deltas`. |
| `tool-finished` | Calls `tc_stream._finish(output)` — sets `output`, marks `completed=True`, closes `output_deltas`. |
| `tool-error` | Calls `tc_stream._fail(message)` — sets `error`, marks `completed=True`, closes `output_deltas`. |

`ToolCallStream` is not meant to be constructed directly — it is always produced by `ToolCallTransformer` as events flow through the stream mux.

## Gotchas

- **`InjectedState` / `InjectedStore` parameters are invisible to the model.** They are stripped from the JSON schema before it is sent to the LLM. Do not prompt the model to fill them in.
- **`InjectedStore` requires `store=` on compile.** If the graph was compiled without `store=`, the injected value is `None` and any tool that calls `store.get(...)` will raise `AttributeError`.
- **`ToolRuntime` is detected by parameter name `runtime` + type annotation.** Rename the parameter and injection silently stops. Keep the name exactly `runtime`.
- **`ToolRuntime` is not `Runtime`.** `Runtime` (from `langgraph.runtime`) is injected into graph nodes; `ToolRuntime` is injected into tools by `ToolNode`. Mixing them up causes type errors at runtime.
- **`emit_output_delta` is a no-op outside `stream_mode="tools"`.** If you invoke the graph without streaming or with a mode that excludes `"tools"`, calls to `emit_output_delta` do nothing and do not raise.
- **Direct attribute assignment on `ToolCallRequest` is deprecated.** Always use `request.override(...)` to produce a modified copy.
- **`wrap_tool_call` runs synchronously even in async mode** unless you also provide `awrap_tool_call`. Synchronous wrappers that do I/O in an async context will block the event loop.
- **`handle_tool_errors=False` means zero protection.** Any exception raised by a tool propagates directly into the graph and can halt execution. Use this only when you handle errors in the tool itself or upstream.
- **`ToolNode` runs all tool calls in the last AIMessage in parallel.** Tool ordering is not guaranteed. If your tools have side effects that must be sequenced, fan them out to separate nodes or serialize them in a custom wrapper.
- **`messages_key` must match your state schema.** If your messages key is not `"messages"`, set `messages_key=` on `ToolNode` and also pass `messages_key=` to `tools_condition`.

## Breaking changes

| Version | Change |
|---|---|
| 1.1.0 (prebuilt) | `ToolCallRequest.override()` introduced; direct attribute assignment deprecated. `awrap_tool_call` added. |
| 1.0.0 (prebuilt) | `ValidationNode` deprecated — use `create_agent` from `langchain.agents`. `AgentState` / `AgentStatePydantic` moved to `langchain.agents`. |
| 1.2.0 / prebuilt 1.1.0 | `ToolRuntime` dataclass introduced in `langgraph-prebuilt`; exposes `state`, `context`, `config`, `stream_writer`, `tool_call_id`, `store`, `tools`, `execution_info`, `server_info`. `emit_output_delta` added. `ToolCallTransformer` stream transformer and `ToolCallStream` handle added; enable per-tool-call structured streaming via `compile(transformers=[ToolCallTransformer])` + `stream_mode="tools"`. |
| 0.3.8 (langchain-core) | `InjectedStore` requires `langchain-core >= 0.3.8`. |

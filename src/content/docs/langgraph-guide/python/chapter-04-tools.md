---
title: "Chapter 4 — Tools"
description: "ToolNode, InjectedState, InjectedStore, ToolRuntime, Command-returning tools, custom error handling, parallel execution, and state injection patterns."
framework: langgraph
language: python
sidebar:
  label: "4 · Tools"
  order: 4
---

# Chapter 4 — Tools

**What you'll learn:** how to plug external capabilities into your graph — the built-in `ToolNode`, injecting graph state and the long-term store into tools, the new `ToolRuntime` all-in-one injection dataclass, routing from inside tool calls with `Command`, configuring fine-grained error handling, and understanding parallel tool execution.

Verified against **`langgraph==1.2.4`** (modules: `langgraph.prebuilt.tool_node`, `langgraph.types`).

**Time:** ~25 minutes.

> Prereqs: [Chapter 2 — Your first agent](/langgraph-guide/python/chapter-02-simple-agents/).

## Tool Integration

### Example 1: Basic ToolNode with `tools_condition`

`ToolNode` executes every `tool_call` in the last `AIMessage`, produces `ToolMessage` results, and returns them under the `messages` key. `tools_condition` routes to `"tools"` when tool calls are present, otherwise to `END`.

**Parallel execution:** `ToolNode` runs all `tool_calls` from a single `AIMessage` concurrently using a thread pool. If the model asks for weather in London *and* a stock price in the same response, both tools execute at the same time. Thread-safety matters if your tools share mutable state.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    prices = {"AAPL": 150.25, "GOOGL": 140.50}
    return f"{symbol}: ${prices.get(symbol, 'N/A')}"

tools = [get_weather, get_stock_price]

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# The model may emit two tool_calls in one AIMessage.
# ToolNode executes get_weather and get_stock_price in parallel.
result = graph.invoke({
    "messages": [{"role": "user", "content": "Weather in London and AAPL price?"}]
})
print(result["messages"][-1].content)
```

### Example 2: `handle_tool_errors` — all variants

`handle_tool_errors` controls what happens when a tool raises an exception. The following table shows every accepted value:

| Value | Behaviour |
|---|---|
| `True` *(default)* | Catch all exceptions; return a built-in error template as a `ToolMessage` |
| `False` | Disable error handling; exceptions propagate and crash the graph |
| `"<message>"` *(str)* | Catch all exceptions; return the fixed string as the error message |
| `ValueError` *(single type)* | Catch only `ValueError`; all other exceptions propagate |
| `(ValueError, ConnectionError)` *(tuple of types)* | Catch exactly those exception types |
| `Callable[..., str]` | Catch all exceptions; call the function with the exception to build the message |

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


@tool
def risky_lookup(item_id: str) -> str:
    """Look up an item by ID (may fail for unknown IDs)."""
    if not item_id.startswith("ITM-"):
        raise ValueError(f"ID must start with 'ITM-', got '{item_id}'")
    return f"Item {item_id}: found"


# ── Variant 1: True (default) ──────────────────────────────────────────────
# Catches all exceptions, returns the built-in error template.
node_default = ToolNode([risky_lookup])                          # handle_tool_errors=True


# ── Variant 2: fixed string ────────────────────────────────────────────────
# Every failure returns the same static message.
node_fixed_msg = ToolNode(
    [risky_lookup],
    handle_tool_errors="Tool failed. Please check your input and try again.",
)


# ── Variant 3: single exception type ──────────────────────────────────────
# Only ValueError is caught; ConnectionError and others propagate.
node_value_err = ToolNode(
    [risky_lookup],
    handle_tool_errors=ValueError,
)


# ── Variant 4: tuple of exception types ───────────────────────────────────
# Catches either ValueError or ConnectionError; anything else propagates.
node_multi_types = ToolNode(
    [risky_lookup],
    handle_tool_errors=(ValueError, ConnectionError),
)


# ── Variant 5: callable ────────────────────────────────────────────────────
# Catches all exceptions; the function receives the exception and returns
# the string that becomes the ToolMessage content.
def my_error_handler(e: Exception) -> str:
    if isinstance(e, ValueError):
        return f"Invalid argument: {e}. Please check the tool's input schema."
    if isinstance(e, ConnectionError):
        return "External service temporarily unavailable. Try again later."
    return f"Tool failed unexpectedly: {e}"

node_callable = ToolNode(
    [risky_lookup],
    handle_tool_errors=my_error_handler,
)


# ── Variant 6: False ───────────────────────────────────────────────────────
# No error handling. Exceptions crash the graph — useful during development
# when you want a full traceback rather than a silent ToolMessage error.
node_no_handling = ToolNode(
    [risky_lookup],
    handle_tool_errors=False,
)
```

### Example 3: Custom `messages_key` with `tools_condition`

`ToolNode` defaults to reading from and writing to `state["messages"]`. Use `messages_key` if your state schema stores messages under a different name. Pass the same key as the second argument to `tools_condition` so it inspects the right field.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic


class CustomerState(TypedDict):
    chat_history: Annotated[list, add_messages]   # not "messages"
    user_id: str


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"Order {order_id}: shipped"


model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools([lookup_order])


def agent(state: CustomerState) -> dict:
    return {"chat_history": [model.invoke(state["chat_history"])]}


builder = StateGraph(CustomerState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([lookup_order], messages_key="chat_history"))
builder.add_edge(START, "agent")

# tools_condition accepts a second argument: the key to inspect for tool_calls.
# Its full signature is:
#   tools_condition(state, messages_key="messages") -> Literal["tools", "__end__"]
builder.add_conditional_edges(
    "agent",
    lambda s: tools_condition(s, messages_key="chat_history"),
)
builder.add_edge("tools", "agent")

graph = builder.compile()
```

### Example 4: `InjectedState` — reading graph state inside a tool

Annotate a tool parameter with `InjectedState` and `ToolNode` will fill it with the current graph state. The parameter is hidden from the LLM's tool schema — the model cannot pass it.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic


class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    permissions: list[str]


@tool
def perform_action(
    action: str,
    state: Annotated[AppState, InjectedState],   # hidden from the LLM
) -> str:
    """Perform an action, checking permissions from state."""
    if action not in state["permissions"]:
        return f"Denied: user {state['user_id']} lacks '{action}' permission."
    return f"Action '{action}' executed for user {state['user_id']}."


model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools([perform_action])


def agent(state: AppState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(AppState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([perform_action]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()
result = graph.invoke({
    "messages": [{"role": "user", "content": "Please delete the record"}],
    "user_id": "alice",
    "permissions": ["read", "write"],  # "delete" is missing
})
print(result["messages"][-1].content)
# The tool returns a denial message; the model relays it.
```

### Example 5: `InjectedStore` — reading the long-term store inside a tool

Annotate a tool parameter with `InjectedStore` and `ToolNode` injects whatever store was passed to `compile(store=...)`. Like `InjectedState`, it's hidden from the LLM.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_anthropic import ChatAnthropic


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


@tool
def save_preference(
    preference: str,
    store: Annotated[BaseStore, InjectedStore()],   # hidden from the LLM
    state: Annotated[ChatState, InjectedState],     # hidden from the LLM
) -> str:
    """Save a user preference for future sessions."""
    user_id = state["user_id"]
    store.put(("prefs", user_id), preference, {"text": preference})
    return f"Saved preference for {user_id}: {preference}"


@tool
def recall_preferences(
    topic: str,
    store: Annotated[BaseStore, InjectedStore()],
    state: Annotated[ChatState, InjectedState],
) -> str:
    """Recall saved preferences relevant to a topic."""
    user_id = state["user_id"]
    items = store.search(("prefs", user_id), query=topic, limit=5)
    if not items:
        return "No relevant preferences found."
    return "\n".join(f"- {it.value['text']}" for it in items)


memory_store = InMemoryStore(
    index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
)

model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools(
    [save_preference, recall_preferences]
)


def agent(state: ChatState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(ChatState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([save_preference, recall_preferences]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(store=memory_store)
```

### Example 6: `ToolRuntime` — all-in-one injection (new in 1.2.1)

`ToolRuntime` is a dataclass introduced in LangGraph 1.2.1 that bundles *all* runtime context into a single parameter. When a tool declares `runtime: ToolRuntime`, `ToolNode` detects and injects it automatically — no `Annotated` wrapper needed. The parameter is invisible to the LLM.

Verified against the installed source (`langgraph-prebuilt==1.1.0`):

```python
@dataclass
class ToolRuntime(Generic[ContextT, StateT]):
    state: StateT | None                       # full graph state dict
    context: ContextT | None                   # typed context from context_schema
    config: RunnableConfig | None              # LangChain runnable config
    stream_writer: StreamWriter | None         # stream tokens mid-tool
    tool_call_id: str | None                   # ID of the triggering tool call
    store: BaseStore | None                    # store passed to compile(store=...)
    tools: list[BaseTool]                      # all tools registered with ToolNode
    execution_info: ExecutionInfo | None = None  # execution metadata
    server_info: ServerInfo | None = None        # LangSmith Platform info (None in OSS)
```

Use `ToolRuntime` when a single tool needs two or more of these values — it avoids stacking multiple `Annotated` parameters.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.tool_node import ToolRuntime   # new in 1.2.1
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langchain_anthropic import ChatAnthropic


class WorkspaceState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    project_id: str


@tool
def smart_search(
    query: str,
    runtime: ToolRuntime,   # injected automatically — invisible to the LLM
) -> str:
    """Search documents, log the call, and stream progress back to the caller."""
    # runtime.state is the full graph state dict; guard for None defensively
    state: WorkspaceState | None = runtime.state
    user_id = state["user_id"] if state else "anon"
    project_id = state["project_id"] if state else "unknown"

    store = runtime.store                          # long-term store
    tool_call_id = runtime.tool_call_id            # tracing / audit
    writer = runtime.stream_writer                 # mid-tool streaming

    # Optionally stream a progress token before the result arrives.
    if writer:
        writer({"type": "progress", "msg": f"Searching project {project_id}…"})

    # Query the long-term store with namespace isolation.
    if store:
        items = store.search(("docs", project_id), query=query, limit=5)
        results = [it.value.get("text", "") for it in items]
    else:
        results = []

    # Audit trail — tool_call_id ties this log entry to the conversation turn.
    print(f"[AUDIT] tool_call={tool_call_id} user={user_id} query={query!r}")

    if not results:
        return "No documents found."
    return "\n".join(f"• {r}" for r in results)


memory_store = InMemoryStore(
    index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
)

model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools([smart_search])


def agent(state: WorkspaceState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(WorkspaceState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode([smart_search]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(store=memory_store)

result = graph.invoke({
    "messages": [{"role": "user", "content": "Find docs about authentication"}],
    "user_id": "alice",
    "project_id": "proj-42",
})
print(result["messages"][-1].content)
```

**When to use `ToolRuntime` vs individual `Annotated` injections:**

| Need | Recommendation |
|---|---|
| Only state or only store | `Annotated[..., InjectedState]` / `Annotated[..., InjectedStore()]` — explicit and clear |
| Two or more of: state, store, tool_call_id, config, stream_writer | `ToolRuntime` — single parameter, less boilerplate |
| Need typed `context` | `ToolRuntime[MyContextType, MyStateType]` — use the generic form |

### Example 7: `Command`-returning tools — routing from inside a tool

A `@tool` that returns a `Command` lets the tool itself drive graph navigation. `ToolNode` unwraps the `Command` into state updates and `goto` signals, enabling agent hand-offs triggered by tool execution.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    assigned_to: str


@tool
def escalate_to_billing(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Escalate this conversation to the billing specialist."""
    # InjectedToolCallId is stripped from the model's schema — it's filled automatically.
    # A ToolMessage is required so the conversation history remains valid
    # (every AIMessage tool_call must be followed by a matching ToolMessage).
    return Command(
        goto="billing_agent",
        update={
            "assigned_to": "billing",
            "messages": [ToolMessage(
                content=f"Escalated to billing: {reason}",
                tool_call_id=tool_call_id,
            )],
        },
    )


@tool
def escalate_to_technical(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Escalate this conversation to the technical support team."""
    return Command(
        goto="technical_agent",
        update={
            "assigned_to": "technical",
            "messages": [ToolMessage(
                content=f"Escalated to technical: {reason}",
                tool_call_id=tool_call_id,
            )],
        },
    )


def billing_agent(state: SupportState) -> dict:
    return {"messages": [("assistant", "Billing team here. How can I help?")]}


def technical_agent(state: SupportState) -> dict:
    return {"messages": [("assistant", "Tech support here. Describe the issue.")]}


model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools(
    [escalate_to_billing, escalate_to_technical]
)


def triage_agent(state: SupportState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(SupportState)
builder.add_node("triage", triage_agent)
builder.add_node("tools", ToolNode([escalate_to_billing, escalate_to_technical]))
builder.add_node("billing_agent", billing_agent)
builder.add_node("technical_agent", technical_agent)

builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", tools_condition)
# ToolNode's Command goto drives us to billing_agent or technical_agent:
builder.add_edge("billing_agent", END)
builder.add_edge("technical_agent", END)

graph = builder.compile()
result = graph.invoke({
    "messages": [{"role": "user", "content": "I was double-charged on my invoice."}],
    "assigned_to": "triage",
})
print(result["assigned_to"])  # "billing"
```

### Example 8: `wrap_tool_call` interceptor and `ToolCallRequest.override()`

`wrap_tool_call` (and its async twin `awrap_tool_call`) lets you intercept every tool call before and after execution. Receive a `ToolCallRequest` (with `.tool_call`, `.tool`, `.state`, `.runtime`) and a callable `execute` — add logging, auth checks, or argument transformations without modifying the tools themselves.

**`ToolCallRequest.override()`** (new in 1.2.4) returns a new `ToolCallRequest` with specific fields replaced. Setting attributes directly on the instance is deprecated — always use `override()` for the immutable update pattern.

```python
from typing import Callable
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


def auditing_interceptor(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    # Pre-execution: auth check
    if tool_name == "delete_record" and not tool_args.get("confirmed"):
        return ToolMessage(
            content="Deletion requires confirmed=True.",
            tool_call_id=request.tool_call["id"],
        )

    # Execute the real tool
    result = execute(request)

    # Post-execution: audit log
    print(f"[AUDIT] {tool_name}({tool_args}) → {getattr(result, 'content', result)}")
    return result


tool_node = ToolNode(
    [get_weather, risky_lookup],
    wrap_tool_call=auditing_interceptor,
)
```

#### Mutating arguments with `override()`

Use `override()` to sanitise or transform arguments before execution without mutating the original request:

```python
from typing import Callable, Awaitable
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest


def sanitize_interceptor(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Redact PII from tool args before execution."""
    original_args = request.tool_call["args"]
    cleaned_args = {
        k: "[REDACTED]" if k in ("email", "phone", "ssn") else v
        for k, v in original_args.items()
    }

    if cleaned_args != original_args:
        # Build a new request — never mutate the original
        new_tool_call = {**request.tool_call, "args": cleaned_args}
        request = request.override(tool_call=new_tool_call)

    return execute(request)


# Async variant — Awaitable must be imported at module scope so the forward
# reference in the signature resolves correctly when get_type_hints() is called.
async def async_sanitize_interceptor(
    request: ToolCallRequest,
    execute: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
) -> ToolMessage:
    # override() works identically in async interceptors
    new_tool_call = {**request.tool_call, "args": {"cleaned": True}}
    return await execute(request.override(tool_call=new_tool_call))


tool_node = ToolNode(
    [my_tool],
    wrap_tool_call=sanitize_interceptor,
    awrap_tool_call=async_sanitize_interceptor,
)
```

> **Note:** `wrap_tool_call` overrides are not serialized to checkpoints and are not re-applied on resume. Put any stateful side effects in graph nodes rather than interceptors.

---

## Summary

| Feature | Class / Import | Key parameter / pattern |
|---|---|---|
| Basic tool execution | `ToolNode` from `langgraph.prebuilt` | `tools` |
| Parallel execution | `ToolNode` | automatic — all tool_calls in one AIMessage run concurrently |
| Custom error handling | `ToolNode` | `handle_tool_errors` (see table in Example 2) |
| Custom messages key | `ToolNode` | `messages_key` |
| Route based on messages key | `tools_condition` from `langgraph.prebuilt` | second arg `messages_key` |
| Read graph state in tool | `InjectedState` from `langgraph.prebuilt` | `Annotated[StateType, InjectedState]` |
| Read store in tool | `InjectedStore` from `langgraph.prebuilt` | `Annotated[BaseStore, InjectedStore()]` |
| All context in one param | `ToolRuntime` from `langgraph.prebuilt.tool_node` | `runtime: ToolRuntime` (no Annotated needed) |
| Route from a tool | `Command` from `langgraph.types` | return from `@tool` |
| Intercept tool calls | `ToolNode` | `wrap_tool_call` / `awrap_tool_call` |
| Immutable arg mutation in interceptors | `ToolCallRequest` from `langgraph.prebuilt.tool_node` | `.override(tool_call=..., state=...)` — returns a new instance |

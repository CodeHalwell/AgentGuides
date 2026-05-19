---
title: "Chapter 4 — Tools"
description: "ToolNode, InjectedState, InjectedStore, Command-returning tools, custom error handling, and state injection patterns."
framework: langgraph
language: python
sidebar:
  label: "4 · Tools"
  order: 4
---

# Chapter 4 — Tools

**What you'll learn:** how to plug external capabilities into your graph — the built-in `ToolNode`, injecting graph state and the long-term store into tools, routing from inside tool calls with `Command`, and configuring fine-grained error handling.

Verified against **`langgraph==1.2.0`** (modules: `langgraph.prebuilt.tool_node`, `langgraph.types`).

**Time:** ~20 minutes.

> Prereqs: [Chapter 2 — Your first agent](/langgraph-guide/python/chapter-02-simple-agents/).

## Tool Integration

### Example 1: Basic ToolNode with `tools_condition`

`ToolNode` executes every `tool_call` in the last `AIMessage`, produces `ToolMessage` results, and returns them under the `messages` key. `tools_condition` routes to `"tools"` when tool calls are present, otherwise to `END`.

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

result = graph.invoke({
    "messages": [{"role": "user", "content": "Weather in London and AAPL price?"}]
})
print(result["messages"][-1].content)
```

### Example 2: Custom `handle_tool_errors`

`handle_tool_errors` controls what happens when a tool raises. Pass a callable `(Exception) -> str` to return a custom error message back to the model instead of crashing.

```python
from langgraph.prebuilt import ToolNode

def my_error_handler(e: Exception) -> str:
    if isinstance(e, ValueError):
        return f"Invalid argument: {e}. Please check the tool's input schema."
    if isinstance(e, ConnectionError):
        return "External service temporarily unavailable. Try again later."
    return f"Tool failed: {e}"


@tool
def risky_lookup(item_id: str) -> str:
    """Look up an item by ID (may fail for unknown IDs)."""
    if not item_id.startswith("ITM-"):
        raise ValueError(f"ID must start with 'ITM-', got '{item_id}'")
    return f"Item {item_id}: found"


tool_node = ToolNode(
    [risky_lookup],
    handle_tool_errors=my_error_handler,
)

# Other options:
# handle_tool_errors=True            → catch all, use default template
# handle_tool_errors="Something went wrong"  → catch all, fixed message
# handle_tool_errors=ValueError      → only catch ValueError
# handle_tool_errors=(ValueError, ConnectionError)  → specific types
# handle_tool_errors=False           → let exceptions propagate
```

### Example 3: Custom `messages_key`

`ToolNode` defaults to reading from and writing to `state["messages"]`. Use `messages_key` if your state schema stores messages under a different name.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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
builder.add_conditional_edges(
    "agent",
    lambda s: "tools" if s["chat_history"][-1].tool_calls else END,
)
builder.add_edge("tools", "agent")

graph = builder.compile()
```

### Example 4: `InjectedState` — reading graph state inside a tool

Annotate a tool parameter with `InjectedState` and `ToolNode` will fill it with the current graph state. The parameter is hidden from the LLM's schema — the model cannot pass it.

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
    state: Annotated[AppState, InjectedState],
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
    store: Annotated[BaseStore, InjectedStore()],
    state: Annotated[ChatState, InjectedState],
) -> str:
    """Save a user preference for future sessions."""
    # Both store and state are injected automatically — the LLM only sees `preference`.
    # Namespace by user_id so preferences are isolated per user.
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

### Example 6: `Command`-returning tools — routing from inside a tool

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
            "messages": [ToolMessage(content=f"Escalated to billing: {reason}", tool_call_id=tool_call_id)],
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
            "messages": [ToolMessage(content=f"Escalated to technical: {reason}", tool_call_id=tool_call_id)],
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

### Example 7: `wrap_tool_call` interceptor

`wrap_tool_call` (and its async twin `awrap_tool_call`) lets you intercept every tool call before and after execution. Receive a `ToolCallRequest` (with `.tool_call`, `.tool`, `.state`, `.runtime`) and a callable `execute` — add logging, auth checks, or argument transformations without modifying the tools themselves.

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

> **Note:** `wrap_tool_call` overrides are not serialized to checkpoints and are not re-applied on resume. Put any stateful side effects in graph nodes rather than interceptors.

---

## Summary

| Feature | Class / Import | Key parameter |
|---|---|---|
| Basic tool execution | `ToolNode` from `langgraph.prebuilt` | `tools` |
| Custom error handling | `ToolNode` | `handle_tool_errors` |
| Custom messages key | `ToolNode` | `messages_key` |
| Read graph state in tool | `InjectedState` from `langgraph.prebuilt` | annotate parameter |
| Read store in tool | `InjectedStore` from `langgraph.prebuilt` | annotate parameter |
| Route from a tool | `Command` from `langgraph.types` | return from `@tool` |
| Intercept tool calls | `ToolNode` | `wrap_tool_call` / `awrap_tool_call` |


---
title: "Chapter 1 — Setup & Core Concepts"
description: "Install LangGraph, understand state, nodes, edges, and compilation — the four primitives everything else builds on."
framework: langgraph
language: python
sidebar:
  label: "1 · Setup & core concepts"
  order: 1
---

# Chapter 1 — Setup & Core Concepts

**What you'll learn:** install LangGraph, understand the mental model, and learn the four primitives — **state, nodes, edges, compilation** — that every graph builds on. Also covers new 1.2.1 additions: `MessagesState`, `REMOVE_ALL_MESSAGES`, `context_schema`, `add_sequence()`, `push_message()`, and the `add_messages` `format` parameter.

**Time:** ~20 minutes.

> This is the first chapter of the Zero → Hero path. Next chapter builds your first real agent on top of these primitives.

## Introduction & Fundamentals

### What is LangGraph?

LangGraph is a low-level orchestration framework for building stateful, long-running agent systems. Unlike high-level abstractions that hide complexity, LangGraph gives you full control over:

- **Agent behaviour** through explicit state management
- **Conditional logic** with fine-grained routing
- **Persistence** with durable execution across failures
- **Memory** both short-term (checkpoints) and long-term (stores)
- **Human oversight** through interrupts and approvals

Built by LangChain Inc, it's inspired by Google's Pregel and Apache Beam, providing production-grade infrastructure trusted by Klarna, Replit, and Elastic.

### Key Mental Model

Think of LangGraph as a **state machine with graphs**:

```
Initial State → Node A → Condition → [Node B or Node C] → Final State
                         ↓
                    Checkpoint saved
```

Each node is a Python function. State flows through edges. Conditions route based on logic. Checkpoints persist progress.

---

## Installation & Setup

### Basic Installation

LangGraph 1.2.4 is the current release. Install the core package alongside `langchain-core`:

```bash
# Core LangGraph (1.2.4)
pip install "langgraph>=1.2.4" langchain-core

# Async support
pip install aiosqlite

# For SQLite checkpointing (requires separate package)
pip install langgraph-checkpoint-sqlite

# For PostgreSQL checkpointing
pip install langgraph[postgres]
pip install psycopg2-binary

# LLM providers (example with Anthropic)
pip install langchain-anthropic

# Development & debugging
pip install langgraph-cli        # CLI tools
```

> **Note on checkpointers:** `InMemorySaver` is built into `langgraph` itself — no extra install needed. `SqliteSaver` requires the separate `langgraph-checkpoint-sqlite` package. PostgreSQL requires `langgraph[postgres]`.

### Project Structure

```
my-agent-project/
├── agent.py              # Main agent definitions
├── states.py             # State schemas
├── nodes.py              # Node implementations
├── tools.py              # Custom tools
├── checkpointer.py       # Persistence setup
├── langgraph.json        # CLI config
└── requirements.txt
```

### Minimal Setup Example

```python
# agent.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver  # built-in, no extra install
from typing_extensions import TypedDict

class State(TypedDict):
    message: str
    response: str

def process_node(state: State) -> dict:
    return {"response": f"Processed: {state['message']}"}

# Build graph
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# Compile with in-memory checkpointing
graph = builder.compile(checkpointer=InMemorySaver())

# Execute
result = graph.invoke(
    {"message": "Hello"},
    config={"configurable": {"thread_id": "user-1"}}
)
print(result)
# {'message': 'Hello', 'response': 'Processed: Hello'}
```

> **Import note:** The correct import for the in-memory checkpointer is `from langgraph.checkpoint.memory import InMemorySaver`. The older alias `MemorySaver` is deprecated — use `InMemorySaver` in all new code.

---

## Core Concepts

### 1. State Schema

State is the single source of truth for your graph. Define it with TypedDict or Pydantic:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # Merges new + old messages
    user_id: str
    context: dict
    should_continue: bool

# The add_messages reducer automatically appends new messages.
# If you pass {"messages": [new_msg]}, it merges with existing ones.
```

**Key insight**: The reducer function (like `add_messages`) defines how state updates combine with existing state.

Custom reducer example:

```python
from operator import add

class CounterState(TypedDict):
    count: Annotated[int, add]  # 5 + 3 = 8 (not replaced)
    last_update: str

class AppendListState(TypedDict):
    items: Annotated[list, lambda x, y: x + y]  # Custom append logic
```

### 2. Nodes

Nodes are Python functions that receive state and return updates:

```python
def my_node(state: State) -> dict:
    """Process state and return updates."""
    processed = transform(state["data"])
    return {
        "data": processed,
        "step_count": state.get("step_count", 0) + 1
    }

# Async nodes
async def async_node(state: State) -> dict:
    result = await expensive_operation(state["data"])
    return {"result": result}
```

**Critical**: Return only the fields you're updating. Other fields merge automatically.

### 3. Edges

Edges connect nodes and define control flow:

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

# Fixed edge: A → B always
builder.add_edge("node_a", "node_b")

# START/END pseudo-nodes
builder.add_edge(START, "node_a")      # Entry point
builder.add_edge("node_b", END)        # Exit point

# Conditional edge: Choose next node based on state
def should_continue(state: State) -> str:
    if state["counter"] > 5:
        return "finish"
    return "loop"

builder.add_conditional_edges(
    "decision",
    should_continue,
    {
        "finish": END,
        "loop": "decision"
    }
)
```

### 4. Compilation

The `.compile()` method turns your graph into an executable Pregel engine:

```python
# Built-in in-memory checkpointing (no extra install)
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())

# SQLite persistence (requires: pip install langgraph-checkpoint-sqlite)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)

# Without persistence (stateless)
graph = builder.compile()
```

### 5. Execution

Multiple ways to run your graph:

```python
# Synchronous - blocking
result = graph.invoke(
    {"message": "Hello"},
    config={"configurable": {"thread_id": "user-1"}}
)

# Streaming - get updates as they happen
for event in graph.stream(
    {"message": "Hello"},
    config={"configurable": {"thread_id": "user-1"}},
    stream_mode="values"  # or "updates" or "debug"
):
    print(event)

# Batch - process multiple inputs
results = graph.batch(
    [{"message": "A"}, {"message": "B"}],
    configs=[
        {"configurable": {"thread_id": f"user-{i}"}}
        for i in range(2)
    ]
)

# Asynchronous
import asyncio
async_result = await graph.ainvoke({"message": "Hello"}, config={...})

# Streaming async
async for event in graph.astream(...):
    print(event)
```

---

## What's New in LangGraph 1.2.1

The sections below document additions and changes introduced in LangGraph 1.2.1. All features are available when you install `langgraph>=1.2.1`.

---

### `MessagesState` — Built-in Messages Shorthand

Defining a TypedDict with an `add_messages`-annotated `messages` field is the single most common pattern in LangGraph. Version 1.2.1 ships `MessagesState` as a ready-made shorthand so you don't have to repeat that boilerplate.

**What it expands to under the hood:**

```python
from typing import Annotated
from langgraph.graph.message import add_messages, AnyMessage

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

**How to use it:**

```python
from langgraph.graph import StateGraph, START, END, MessagesState
# or equivalently:
# from langgraph.graph.message import MessagesState

from langchain_core.messages import HumanMessage, AIMessage

def chat_node(state: MessagesState) -> dict:
    # state["messages"] is a list of BaseMessage objects
    last = state["messages"][-1]
    reply = AIMessage(content=f"Echo: {last.content}")
    return {"messages": [reply]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage(content="Hello")]})
# result["messages"] contains the original HumanMessage + the new AIMessage
```

**Extending `MessagesState`** — add extra fields by subclassing or by creating a new TypedDict that includes `messages`:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

# Option A: extend with TypedDict inheritance
class AppState(MessagesState):
    user_id: str
    session_data: dict

# Option B: keep it explicit
from langgraph.graph.message import add_messages

class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_data: dict
```

---

### `REMOVE_ALL_MESSAGES` — Clear the Entire Message List

Previously, clearing all messages required iterating over every message and issuing individual `RemoveMessage` operations. LangGraph 1.2.1 adds the `REMOVE_ALL_MESSAGES` constant so you can wipe the list in a single operation.

**Import:**

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage
```

**The constant value** (for reference, you never need to use the raw string):

```python
REMOVE_ALL_MESSAGES = '__remove_all__'
```

**Example — reset messages between sessions:**

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage

def clear_history_node(state: MessagesState) -> dict:
    """Wipe the entire message history, preserving the current user message.

    add_messages processes [RemoveMessage(REMOVE_ALL_MESSAGES), current_msg]
    as: clear everything, then keep the messages that follow the sentinel.
    """
    current_msg = state["messages"][-1]  # keep the incoming user message
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), current_msg]}

def respond_node(state: MessagesState) -> dict:
    msgs = state["messages"]
    last = msgs[-1] if msgs else None
    content = f"Fresh start! You said: {last.content}" if last else "Fresh start!"
    return {"messages": [AIMessage(content=content)]}

builder = StateGraph(MessagesState)
builder.add_node("clear", clear_history_node)
builder.add_node("respond", respond_node)
builder.add_edge(START, "clear")
builder.add_edge("clear", "respond")
builder.add_edge("respond", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "session-1"}}

# First run — builds up history
graph.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)

# Second run — clear_history_node wipes all prior messages before responding
result = graph.invoke(
    {"messages": [HumanMessage(content="Starting fresh")]},
    config=config
)
# result["messages"] contains only the new HumanMessage + new AIMessage
```

> **When to use this:** session resets, conversation restarts, clearing stale context before a new task, or enforcing a token-budget ceiling by periodically wiping history.

---

### `context_schema` on `StateGraph` — Read-Only Runtime Context

LangGraph 1.2.1 introduces the `context_schema` constructor parameter to replace the older `config_schema`. It is designed for **read-only, immutable context** that nodes should be able to read but never write back to state: things like a `user_id`, an API key, a database connection, or a model provider choice.

Unlike regular state, context is never persisted to a checkpoint and cannot be updated by a node return value. It is injected at invocation time and stays constant for the lifetime of that run.

**Define a context schema** using a `dataclass` or `TypedDict`:

```python
from dataclasses import dataclass

@dataclass
class AppContext:
    user_id: str
    api_key: str
    model_provider: str = "anthropic"
```

**Wire it into the graph:**

```python
from langgraph.graph import StateGraph, START, END, MessagesState

builder = StateGraph(MessagesState, context_schema=AppContext)
```

**Access it inside nodes** via `langgraph.runtime.Runtime`:

```python
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage

def my_node(state: MessagesState, runtime: Runtime[AppContext]) -> dict:
    # runtime.context is typed as AppContext
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key

    # Use context values in your logic
    reply = AIMessage(content=f"Hello, user {user_id}!")
    return {"messages": [reply]}
```

**Pass context at invocation time** using the `context` keyword argument:

```python
graph = builder.compile()

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hi"}]},
    context={"user_id": "u-123", "api_key": "sk-...", "model_provider": "openai"}
)
```

**Full working example with a dataclass context:**

```python
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage

@dataclass
class AppContext:
    user_id: str
    api_key: str
    model_provider: str = "anthropic"

def call_model(state: MessagesState, runtime: Runtime[AppContext]) -> dict:
    user_id = runtime.context.user_id
    provider = runtime.context.model_provider
    last_msg = state["messages"][-1].content
    # In real code you'd call your LLM here using runtime.context.api_key
    reply = AIMessage(content=f"[{provider}] Hello {user_id}: {last_msg}")
    return {"messages": [reply]}

builder = StateGraph(MessagesState, context_schema=AppContext)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile(checkpointer=InMemorySaver())

result = graph.invoke(
    {"messages": [HumanMessage(content="What's my user ID?")]},
    config={"configurable": {"thread_id": "t-1"}},
    context={"user_id": "u-456", "api_key": "sk-test", "model_provider": "openai"},
)
for msg in result["messages"]:
    print(msg.content)
# [openai] Hello u-456: What's my user ID?
```

> **`context_schema` vs `config_schema`:** `config_schema` (deprecated) was for LangChain `RunnableConfig` style configuration that mixed runtime values with framework-level settings like `thread_id`. `context_schema` is a cleaner separation: graph-level `configurable` keys (like `thread_id`) stay in `config`, while your own runtime values live in `context`.

---

### `add_sequence()` — Chain Nodes Without Manual Edge Wiring

When you have a straight pipeline of nodes that should always run in order, calling `add_node` and `add_edge` for each pair is repetitive. The new `add_sequence()` method does both in one call.

**Before (verbose):**

```python
builder.add_node("fetch_context", fetch_context)
builder.add_node("call_model", call_model)
builder.add_node("save_conversation", save_conversation)
builder.add_edge("fetch_context", "call_model")
builder.add_edge("call_model", "save_conversation")
```

**After (with `add_sequence`):**

```python
builder.add_sequence([fetch_context, call_model, save_conversation])
# Registers all three nodes and wires fetch_context → call_model → save_conversation
```

Node names are inferred from the function name. You can override any name with a `(name, fn)` tuple:

```python
builder.add_sequence([
    ("fetch", fetch_context),       # explicit name
    call_model,                     # inferred: "call_model"
    ("persist", save_conversation), # explicit name
])
```

`add_sequence` returns `Self` for method chaining:

```python
builder = (
    StateGraph(MessagesState)
    .add_sequence([fetch_context, call_model, save_conversation])
    .add_edge(START, "fetch_context")
    .add_edge("save_conversation", END)
)
graph = builder.compile()
```

**Full example:**

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage

def fetch_context(state: MessagesState) -> dict:
    # Simulate fetching external context
    print("Step 1: Fetching context...")
    return {}

def call_model(state: MessagesState) -> dict:
    print("Step 2: Calling model...")
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"Response to: {last.content}")]}

def save_conversation(state: MessagesState) -> dict:
    print("Step 3: Saving conversation...")
    return {}

builder = StateGraph(MessagesState)
builder.add_sequence([fetch_context, call_model, save_conversation])
builder.add_edge(START, "fetch_context")
builder.add_edge("save_conversation", END)

graph = builder.compile(checkpointer=InMemorySaver())

from langchain_core.messages import HumanMessage
result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "seq-1"}}
)
```

---

### `add_messages` with `format="langchain-openai"`

The `add_messages` reducer now accepts a `format` keyword argument. Setting `format="langchain-openai"` instructs LangGraph to convert any raw dict messages (including those with Anthropic-style content blocks) into the OpenAI-compatible `BaseMessage` format.

This is useful when you want to pass messages directly to an OpenAI-compatible endpoint regardless of the original message format, or when you're mixing providers and need a normalised representation.

**Usage in a state schema:**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]
```

**What it does:** messages arriving as dicts with Anthropic-style `source`/`media_type` content blocks get converted to OpenAI-style `image_url` objects. Plain text messages are unaffected.

**Example:**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]

def chatbot_node(state: State) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's an image:",
                        "cache_control": {"type": "ephemeral"},  # Anthropic-style
                    },
                    {
                        "type": "image",
                        "source": {                              # Anthropic-style
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "1234",
                        },
                    },
                ],
            },
        ]
    }

builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")
graph = builder.compile()

result = graph.invoke({"messages": []})
# result["messages"] contains a HumanMessage with OpenAI-format content:
# HumanMessage(content=[
#     {"type": "text", "text": "Here's an image:"},
#     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,1234"}},
# ])
```

> **Requirement:** `format="langchain-openai"` requires `langchain-core>=0.3.11`.

---

### `push_message()` — Write Directly to the Messages Stream

`push_message()` lets you emit a message to the `messages` stream **immediately**, without waiting for the node to return. This is useful for intermediate status updates, streaming progress indicators, or partial responses while a long operation is running.

**Import:**

```python
from langgraph.graph.message import push_message
```

**Signature:**

```python
push_message(
    message: MessageLikeRepresentation | BaseMessageChunk,
    *,
    state_key: str = "messages",
) -> AnyMessage
```

- `message` — any message-like object: a `BaseMessage`, a `(role, content)` tuple, or a raw dict.
- `state_key` — defaults to `"messages"`. Pass `None` if you want to push to the stream without automatically writing to a channel.

**Example — progress indicator during a slow operation:**

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import push_message
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage
import time

def slow_research_node(state: MessagesState) -> dict:
    # Push an immediate status message to the stream
    push_message(AIMessage(content="Working on it...", id="status-1"))

    # ... do slow work ...
    time.sleep(2)

    # The final return value adds the real answer
    return {"messages": [AIMessage(content="Here is the full answer.")]}

builder = StateGraph(MessagesState)
builder.add_node("research", slow_research_node)
builder.add_edge(START, "research")
builder.add_edge("research", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Use stream_mode="messages" to receive push_message outputs in real time
for chunk in graph.stream(
    {"messages": [HumanMessage(content="Research quantum computing")]},
    config={"configurable": {"thread_id": "stream-1"}},
    stream_mode="messages",
):
    print(chunk)
```

> **When to use this:** real-time feedback for users during long-running nodes, streaming partial LLM output token by token, or emitting tool-call progress events without restructuring your node logic.

---

## Quick Reference: LangGraph 1.2.1 New Imports

```python
# Built-in messages shorthand
from langgraph.graph import MessagesState
from langgraph.graph.message import MessagesState  # equivalent

# Clear all messages constant
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage
# Usage: RemoveMessage(id=REMOVE_ALL_MESSAGES)

# Runtime context for context_schema
from langgraph.runtime import Runtime

# Push messages to stream mid-node
from langgraph.graph.message import push_message

# add_messages with format support
from langgraph.graph import add_messages
from langgraph.graph.message import add_messages  # equivalent
# Usage: Annotated[list, add_messages(format="langchain-openai")]

# In-memory checkpointer (no extra install required)
from langgraph.checkpoint.memory import InMemorySaver

# SQLite checkpointer (requires: pip install langgraph-checkpoint-sqlite)
from langgraph.checkpoint.sqlite import SqliteSaver
```

---

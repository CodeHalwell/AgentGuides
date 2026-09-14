---
title: "Chapter 2 — Your First Agent"
description: "Build linear pipelines, conditional routers, looping agents, ReAct tool-calling agents, and streaming outputs — six concrete patterns that cover most single-agent workloads. Verified against langgraph==1.2.11."
framework: langgraph
language: python
sidebar:
  label: "2 · Your first agent"
  order: 2
---

# Chapter 2 — Your First Agent

**What you'll learn:** six concrete agent patterns you can copy-paste and adapt — a linear chat pipeline, conditional routing, looping with a safeguard counter, a ReAct tool-calling agent with `create_react_agent`, `tools_condition` for routing, and streaming execution.

Verified against **`langgraph==1.2.11`**.

**Time:** ~25 minutes.

> Prereqs: [Chapter 1 — Setup & Core Concepts](/langgraph-guide/python/chapter-01-setup-and-core-concepts/).

---

## Example 1: Linear Chat Pipeline

A basic chatbot with no branching. Uses `MessagesState` — the built-in shorthand for agents that only need a `messages` list.

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

# Construct the model ONCE at module scope — reusing across node calls avoids
# unnecessary client-creation overhead in production.
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

def call_model(state: MessagesState) -> dict:
    """Call the LLM with the full message history."""
    system = SystemMessage(content="You are a concise assistant.")
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

# Compile with in-memory persistence so the thread retains history
graph = builder.compile(checkpointer=InMemorySaver())

# First turn
cfg = {"configurable": {"thread_id": "chat-1"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What is LangGraph?"}]},
    config=cfg,
)
print(result["messages"][-1].content)

# Second turn — history is automatically carried forward
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Give me one concrete example."}]},
    config=cfg,
)
print(result["messages"][-1].content)
```

`MessagesState` is equivalent to writing:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]
```

Use your own TypedDict when you need extra fields alongside `messages`.

---

## Example 2: Conditional Routing

Route based on message type. A classifier node sets `query_type`, and `add_conditional_edges` dispatches to the right handler.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    query: str
    query_type: str
    result: str

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


def classify_query(state: State) -> dict:
    query = state["query"].lower()
    if any(w in query for w in ["search", "find", "lookup", "who", "where"]):
        return {"query_type": "search"}
    elif any(w in query for w in ["calculate", "math", "solve", "%", "+"]):
        return {"query_type": "math"}
    else:
        return {"query_type": "general"}


def search_web(state: State) -> dict:
    return {"result": f"[search] Top results for: {state['query']}"}


def solve_math(state: State) -> dict:
    return {"result": f"[math] Computing: {state['query']}"}


def general_response(state: State) -> dict:
    response = model.invoke(state["query"])
    return {"result": response.content}


builder = StateGraph(State)
builder.add_node("classify", classify_query)
builder.add_node("search", search_web)
builder.add_node("math", solve_math)
builder.add_node("general", general_response)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    lambda s: s["query_type"],   # path function — returns the key to route on
    {"search": "search", "math": "math", "general": "general"},
)
for handler in ["search", "math", "general"]:
    builder.add_edge(handler, END)

graph = builder.compile()

print(graph.invoke({"query": "Who invented Python?", "query_type": "", "result": ""})["result"])
print(graph.invoke({"query": "Calculate 15% of 2000", "query_type": "", "result": ""})["result"])
```

---

## Example 3: Looping Agent with a Counter Safeguard

Agents that retry or refine their output loop back to an earlier node. A counter in state provides a hard exit.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    iteration: int
    data: str
    final_result: str


def process_step(state: LoopState) -> dict:
    processed = state["data"] + f" [step-{state['iteration']}]"
    return {"data": processed, "iteration": state["iteration"] + 1}


def should_continue(state: LoopState) -> str:
    # Hard cap: never run more than 3 iterations
    return "finish" if state["iteration"] >= 3 else "continue"


def finalize(state: LoopState) -> dict:
    return {"final_result": state["data"]}


builder = StateGraph(LoopState)
builder.add_node("process", process_step)
builder.add_node("finalize", finalize)
builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    should_continue,
    {"continue": "process", "finish": "finalize"},  # "process" loops back to itself
)
builder.add_edge("finalize", END)

graph = builder.compile()

result = graph.invoke({"iteration": 0, "data": "start", "final_result": ""})
print(result)
# {'iteration': 3, 'data': 'start [step-0] [step-1] [step-2]', 'final_result': 'start [step-0] [step-1] [step-2]'}
```

---

## Example 4: ReAct Agent with `create_react_agent`

`create_react_agent` from `langgraph.prebuilt` builds a full tool-calling loop for you — no boilerplate. It uses `MessagesState` internally and adds a `tools` node wired with `tools_condition`.

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    data = {"London": "12°C, overcast", "Tokyo": "24°C, sunny", "Paris": "18°C, partly cloudy"}
    return data.get(city, f"No data for {city!r}")


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies (stub)."""
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09, "GBP_USD": 1.27}
    key = f"{from_currency}_{to_currency}"
    rate = rates.get(key, 1.0)
    return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}"


agent = create_react_agent(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    tools=[get_weather, convert_currency],
    checkpointer=InMemorySaver(),
    prompt="You are a helpful travel assistant. Answer concisely.",
)

cfg = {"configurable": {"thread_id": "travel-1"}}

# The agent decides which tools to call and loops until it has an answer.
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Weather in London? Convert 100 GBP to USD."}]},
    config=cfg,
)
print(result["messages"][-1].content)
```

Key parameters on `create_react_agent`:

| Parameter | Type | Purpose |
|---|---|---|
| `model` | `str \| BaseChatModel \| Callable` | The LLM to use (static or dynamic) |
| `tools` | `list[BaseTool \| Callable]` | Tools the model may invoke |
| `prompt` | `str \| SystemMessage \| Callable` | System prompt (string shorthand or callable) |
| `checkpointer` | `BaseCheckpointSaver \| None` | Persistence for multi-turn threads |
| `store` | `BaseStore \| None` | Cross-thread long-term memory |
| `pre_model_hook` | `Callable \| Runnable \| None` | Transform state before each model call |
| `post_model_hook` | `Callable \| Runnable \| None` | Transform/inspect state after each model call |
| `response_format` | `dict \| type[BaseModel] \| None` | Force structured output on the final response |
| `interrupt_before` / `interrupt_after` | `list[str]` | Pause for human approval before/after named nodes |

---

## Example 5: Custom agent loop with `tools_condition`

When you need more control than `create_react_agent` gives you, build the agent graph manually. `tools_condition` is the standard routing function that checks whether the last `AIMessage` contains tool calls.

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for '{query}': ..."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}})   # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools([search, calculator])


def agent_node(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


# ToolNode runs all tool calls in the last AIMessage in parallel.
tool_node = ToolNode(tools=[search, calculator])

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
# tools_condition: if the last message has tool_calls → "tools", else → END
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")   # loop back after tool execution

graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "What is sqrt(144) + 5?"}]})
print(result["messages"][-1].content)
```

`tools_condition` is equivalent to:

```python
from langgraph.graph import END

def tools_condition(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END
```

---

## Example 6: Streaming Output

All LangGraph graphs expose a `.stream()` method. Choose the mode that fits your use case.

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = create_react_agent(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    tools=[multiply],
)

# --- Mode 1: "updates" (default) — only what each node changed ---
print("=== updates ===")
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is 12 × 34?"}]},
    stream_mode="updates",
):
    for node_name, update in chunk.items():
        print(f"{node_name}: {update}")

# --- Mode 2: "values" — full state snapshot after each step ---
print("\n=== values ===")
for snapshot in agent.stream(
    {"messages": [{"role": "user", "content": "What is 12 × 34?"}]},
    stream_mode="values",
):
    last = snapshot["messages"][-1]
    print(f"[{type(last).__name__}] {getattr(last, 'content', '')[:60]}")

# --- Mode 3: "messages" — token-level streaming for LLM text ---
print("\n=== messages (token streaming) ===")
for msg_chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "Explain LangGraph in one sentence."}]},
    stream_mode="messages",
):
    if hasattr(msg_chunk, "content") and msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
print()

# --- Mode 4: multiple modes at once ---
print("\n=== updates + messages ===")
for stream_mode, data in agent.stream(
    {"messages": [{"role": "user", "content": "What is 7 × 8?"}]},
    stream_mode=["updates", "messages"],
):
    if stream_mode == "messages":
        msg_chunk, _ = data
        if hasattr(msg_chunk, "content") and msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)
print()
```

**Streaming modes summary:**

| Mode | What you get | Best for |
|---|---|---|
| `"updates"` | Dict of `{node_name: state_changes}` after each node | Watching graph progress |
| `"values"` | Full state snapshot after each node | Inspecting the whole state at each step |
| `"messages"` | `(chunk, metadata)` tuples — one per LLM token | Token-level streaming to a UI |
| `"debug"` | Detailed execution trace with timing | Debugging and profiling |
| `"tasks"` | `(TasksStreamPart, ...)` — task-level events | Monitoring subgraph tasks |
| `"custom"` | Values written via `runtime.stream_writer(...)` | Custom progress signals from nodes |

Pass a list to receive multiple modes simultaneously; the first element of each yielded tuple identifies the mode.

---

## Quick-reference: which pattern to pick

| You need | Use |
|---|---|
| A simple chat loop | `StateGraph(MessagesState)` with one `call_model` node |
| Tool-calling without boilerplate | `create_react_agent(model, tools)` |
| Custom routing between tool-calling nodes | `ToolNode` + `tools_condition` + manual `StateGraph` |
| Branch on message content | `add_conditional_edges(source, path_fn, path_map)` |
| Retry / refine in a loop | Loop back with `add_conditional_edges("process", ..., {"continue": "process"})` |
| Token-level streaming to a UI | `graph.stream(..., stream_mode="messages")` |

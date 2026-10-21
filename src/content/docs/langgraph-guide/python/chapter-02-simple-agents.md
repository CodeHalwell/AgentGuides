---
title: "Chapter 2 — Your First Agent"
description: "Build linear pipelines, conditional routers, looping agents, tool-calling agents, and streaming outputs — six concrete patterns that cover most single-agent workloads."
framework: "langgraph"
language: "python"
sidebar:
  label: "2 · Your first agent"
  order: 2
---

**What you'll learn:** six concrete agent patterns you can copy-paste and adapt — a linear chat pipeline, conditional routing, looping with a safeguard counter, a tool-calling agent with `create_agent`, `tools_condition` for routing, and streaming execution.

Verified against **`langgraph==%%version=langgraph%%`**.

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
model = ChatAnthropic(model="claude-sonnet-5-5")

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

Route based on message content. A classifier node inspects keyword patterns and sets `query_type`, then `add_conditional_edges` dispatches to the right handler.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    query: str
    query_type: str
    result: str

model = ChatAnthropic(model="claude-sonnet-5-5")


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

## Example 4: Tool-calling agent with `create_agent`

`create_agent` from the `langchain` package (`pip install langchain`) builds the whole tool-calling loop: a model node, a tools node, and the routing between them. It returns a compiled LangGraph graph, so `invoke`, `stream`, checkpointers and interrupts work as in the examples above.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


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
    if key not in rates:
        return f"Unsupported currency pair: {from_currency} → {to_currency}. Supported: USD↔EUR, GBP→USD."
    return f"{amount} {from_currency} = {amount * rates[key]:.2f} {to_currency}"


agent = create_agent(
    "anthropic:%%model=anthropic%%",
    tools=[get_weather, convert_currency],
    system_prompt="You are a helpful travel assistant. Answer concisely.",
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "travel-1"}, "recursion_limit": 25}

# The agent decides which tools to call and loops until it has an answer.
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Weather in London? Convert 100 GBP to USD."}]},
    config=cfg,
)
print(result["messages"][-1].content)
```

`recursion_limit` caps the number of graph steps (each model call and each tools step counts). The default is 10007, which is far too high to stop a model that keeps calling tools, so set it for agent loops.

Main `create_agent` parameters:

| Parameter | Purpose |
|---|---|
| `model` | A `"provider:model"` string or a chat model instance |
| `tools` | Tools the model may call (`@tool` functions, `BaseTool`s, or provider tool dicts) |
| `system_prompt` | System prompt string or `SystemMessage` |
| `middleware` | Hooks around model and tool calls, and built-ins for approvals, summarisation, limits and retries ([Chapter 8](/langgraph-guide/python/chapter-08-middleware-hooks/)) |
| `response_format` | A Pydantic model or JSON schema for structured output; the result is in `result["structured_response"]` |
| `state_schema`, `context_schema` | Extra state fields (subclass `langchain.agents.AgentState`); typed run context passed as `invoke(..., context=...)` |
| `checkpointer`, `store` | Thread persistence and long-term memory ([Chapter 5](/langgraph-guide/python/chapter-05-memory/)) |

> **`create_react_agent` is deprecated.** Older code uses `langgraph.prebuilt.create_react_agent(model, tools, prompt=...)`. It still runs, with a `LangGraphDeprecatedSinceV10` warning, and is due for removal in LangGraph 2.0. Change `prompt=` to `system_prompt=` and move `pre_model_hook`/`post_model_hook` to middleware; [Chapter 8](/langgraph-guide/python/chapter-08-middleware-hooks/#migrating-from-create_react_agent-hooks) has the full mapping.

---

## Example 5: Custom agent loop with `tools_condition`

When you need more control than `create_agent` gives you, build the agent graph manually. `tools_condition` is the standard routing function that checks whether the last `AIMessage` contains tool calls.

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
    """Evaluate a simple arithmetic expression (stub — replace with a real math library in production)."""
    import ast, operator as op
    # Exponentiation (**) is intentionally excluded — unbounded exponents like
    # 2**1000000000 can exhaust memory before the exception handler fires.
    _ops = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
        ast.USub: op.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ops:
            return _ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ops:
            return _ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"Error: {e}"


model = ChatAnthropic(model="claude-sonnet-5-5").bind_tools([search, calculator])


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

result = graph.invoke({"messages": [{"role": "user", "content": "What is 12 * 12 + 5?"}]})
print(result["messages"][-1].content)
```

`tools_condition` is equivalent to:

```python
from langgraph.graph import END

def tools_condition(state):
    from langchain_core.messages import AIMessage
    last = state["messages"][-1]
    # Match LangGraph's real tools_condition: only an AIMessage with non-empty
    # tool_calls routes to tools; arbitrary objects with a tool_calls attribute do not.
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END
```

---

## Example 6: Streaming Output

All LangGraph graphs expose a `.stream()` method. Choose the mode that fits your use case.

```python
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = create_agent("anthropic:%%model=anthropic%%", tools=[multiply])

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

`stream()` defaults to `"updates"`. Pass a list of modes to get several at once; each item is then a `(mode, data)` tuple. [Chapter 6](/langgraph-guide/python/chapter-06-streaming-and-debugging/) covers every mode, including `"custom"`, `"debug"` and the typed `version="v2"` output.

---

## Quick-reference: which pattern to pick

| You need | Use |
|---|---|
| A simple chat loop | `StateGraph(MessagesState)` with one `call_model` node |
| Tool-calling without boilerplate | `create_agent(model, tools)` |
| Custom routing between tool-calling nodes | `ToolNode` + `tools_condition` + manual `StateGraph` |
| Branch on message content | `add_conditional_edges(source, path_fn, path_map)` |
| Retry / refine in a loop | Loop back with `add_conditional_edges("process", ..., {"continue": "process"})` |
| Token-level streaming to a UI | `graph.stream(..., stream_mode="messages")` |

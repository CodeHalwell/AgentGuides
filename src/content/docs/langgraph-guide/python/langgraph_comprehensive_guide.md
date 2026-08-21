---
title: "LangGraph: Comprehensive Technical Guide (Beginner to Expert)"
description: "Latest Version: LangGraph 1.2.11 (August 2026) Focus: Python Examples with practical, production-ready patterns Author Note: This guide progresses from fundamentals through advanced"
framework: langgraph
language: python
---

Latest: 1.2.11 | Updated: August 17, 2026
# LangGraph: Comprehensive Technical Guide (Beginner to Expert)

**Latest Version**: LangGraph 1.2.11 (August 2026)
**Focus**: Python examples with practical, production-ready patterns
**Author Note**: This guide progresses from fundamentals through advanced multi-agent architectures with real-world workflows.

> **Errata (April 2026).** An earlier draft of this page documented fabricated APIs (`langgraph.llm_hooks.pre_model_hook`, `langgraph.cache.cache_node`, `langgraph.graph.deferred`, `langgraph.prebuilt.command_tool`, `@tool(updates_state=True)`, `langgraph template` CLI subcommand). They are not in the installed package. See the [Errata section](#errata-removed-fabricated-sections) below for the real replacements. For middleware, read the dedicated [Chapter 8 — Middleware](/langgraph-guide/python/chapter-08-middleware-hooks/) page.

**What's real in v1.2.11 (verified August 2026):**
- `ToolRuntime` dataclass (`langgraph.prebuilt`) — injected into tools at execution time
- `ToolCallTransformer` abstract class (`langgraph.prebuilt`) — intercepts and transforms tool call arguments
- `InjectedState` / `InjectedStore` (`langgraph.prebuilt`) — inject graph state or the store into tools, invisible to the LLM
- `Overwrite` (`langgraph.types`) — bypass a reducer and replace a channel value directly
- `MessagesState` (`langgraph.graph`) — built-in TypedDict with `add_messages` reducer, ready to subclass
- `CheckpointMetadata` / `CheckpointTuple` (`langgraph.types`, `langgraph.checkpoint.base`) — inspect and traverse checkpoint history
- `BinaryOperatorAggregate` (`langgraph.channels.binop`) — underlying channel for `Annotated[T, reducer_fn]` fields; supports `Overwrite` bypass
- `Topic` channel (`langgraph.channels.topic`) — multi-value pub/sub with `accumulate` mode for event logs
- `EphemeralValue` channel (`langgraph.channels.ephemeral_value`) — per-step temporary state, auto-clears, used for `START` channel
- `NamedBarrierValue` (`langgraph.channels.named_barrier_value`) — synchronisation barrier that fires when all named writers signal
- `entrypoint` + `task` Functional API (`langgraph.func`) — build workflows without `StateGraph`; `entrypoint.final` for decoupled save value
- `RetryPolicy` / `TimeoutPolicy` / `CachePolicy` (`langgraph.types`) — per-node / per-task resilience and caching configuration
- `Send` with per-instance timeout override (`langgraph.types`) — dynamic map-reduce dispatch; `timeout` kwarg per `Send`
- `Command.PARENT` (`langgraph.types`) — signal parent graph's state from inside a subgraph
- Type-safe v2 streaming / invoke API (`version="v2"`)
- Pydantic / dataclass auto-coercion on input
- Python 3.10 – 3.14 support (Python 3.9 dropped)
- Cross-thread memory via `Store` + `InjectedStore`
- Fixed time-travel replays with interrupts and subgraphs

**Deprecated in v1.2.11:**
- `langgraph.prebuilt.HumanInterrupt` → `langchain.agents.interrupt.HumanInterrupt`
- `langgraph.prebuilt.HumanInterruptConfig` → `langchain.agents.interrupt.HumanInterruptConfig`
- `langgraph.prebuilt.ActionRequest` → `langchain.agents.interrupt.ActionRequest`
- `langgraph.prebuilt.ValidationNode` → use `create_agent` from `langchain.agents` with custom error handling
- `@entrypoint(config_schema=...)` → `@entrypoint(context_schema=...)`
- `add_node(..., retry=...)` → `add_node(..., retry_policy=...)`; `add_node(..., cache=...)` → `add_node(..., cache_policy=...)`

---

## Table of Contents

1. [Introduction & Fundamentals](#introduction--fundamentals)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Simple Agents](#simple-agents)
5. [Multi-Agent Systems](#multi-agent-systems)
6. [Tool Integration](#tool-integration)
7. [Memory & Persistence](#memory--persistence)
8. [Debugging & Visualization](#debugging--visualization)
9. [Type-Safe v2 API](#type-safe-v2-api-v11x)
10. [Human-in-the-Loop](#human-in-the-loop)
11. [Advanced Patterns](#advanced-patterns)
12. [Errata — removed fabricated sections](#errata-removed-fabricated-sections)
13. [Functional API](#functional-api-langgraph-10)
14. [Production Deployment](#production-deployment)
15. [Class & API Reference](#class--api-reference)

---

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

```bash
# Core LangGraph
pip install langgraph langchain-core

# Async support
pip install aiosqlite

# For database checkpointing
pip install langgraph[postgres]  # PostgreSQL support
pip install psycopg2-binary      # PostgreSQL adapter

# LLM providers (example with Anthropic)
pip install langchain-anthropic

# Development & debugging
pip install langgraph-cli        # CLI tools
```

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
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

class State(TypedDict):
    message: str
    response: str

def process_node(state: State):
    return {"response": f"Processed: {state['message']}"}

# Build graph
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# Compile with memory
graph = builder.compile(checkpointer=InMemorySaver())

# Execute
result = graph.invoke(
    {"message": "Hello"},
    config={"configurable": {"thread_id": "user-1"}}
)
print(result)
```


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

# The add_messages reducer automatically appends new messages
# If you pass {"messages": [new_msg]}, it merges with existing
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
from langgraph.checkpoint.sqlite import SqliteSaver

# Compile with persistence
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)

# Without persistence (in-memory only)
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

## Simple Agents

### Example 1: Linear Chat Pipeline

A basic chatbot with no branching:


```python
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_name: str

def fetch_user_context(state: State):
    """Load user info from database."""
    # Simulate DB lookup
    return {"user_name": "Alice"}

def call_model(state: State):
    """Call LLM with messages."""
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    
    system_prompt = f"You're helping {state['user_name']}. Be concise."
    
    response = model.invoke(state["messages"], system_prompt=system_prompt)
    return {"messages": [response]}

def save_conversation(state: State):
    """Persist messages to database."""
    # Save state["messages"] to DB
    return {}

# Build the graph
builder = StateGraph(State)
builder.add_node("fetch_context", fetch_user_context)
builder.add_node("model", call_model)
builder.add_node("save", save_conversation)

builder.add_edge(START, "fetch_context")
builder.add_edge("fetch_context", "model")
builder.add_edge("model", "save")
builder.add_edge("save", END)

# Compile with persistence
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())

# Use it
config = {"configurable": {"thread_id": "chat-session-1"}}
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    config=config
)

# Continue in same thread - context preserved
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What did you say before?"}]},
    config=config
)
```


### Example 2: Conditional Routing

Route based on message type:

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    query_type: str
    result: str

def classify_query(state: State) -> dict:
    """Determine query type."""
    query = state["query"].lower()
    
    if any(word in query for word in ["search", "find", "lookup"]):
        return {"query_type": "search"}
    elif any(word in query for word in ["calculate", "math", "solve"]):
        return {"query_type": "math"}
    else:
        return {"query_type": "general"}

def search_web(state: State) -> dict:
    """Handle search queries."""
    # Call search API
    result = f"Search results for: {state['query']}"
    return {"result": result}

def solve_math(state: State) -> dict:
    """Handle math queries."""
    result = f"Math answer for: {state['query']}"
    return {"result": result}

def general_response(state: State) -> dict:
    """Handle general queries."""
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    response = model.invoke(state["query"])
    return {"result": response.content}

# Build graph with conditional routing
builder = StateGraph(State)
builder.add_node("classify", classify_query)
builder.add_node("search", search_web)
builder.add_node("math", solve_math)
builder.add_node("general", general_response)

# Route based on classification
def route_to_handler(state: State) -> str:
    return state["query_type"]

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    route_to_handler,
    {
        "search": "search",
        "math": "math",
        "general": "general"
    }
)

# All handlers lead to END
for handler in ["search", "math", "general"]:
    builder.add_edge(handler, END)

graph = builder.compile()

# Test it
result = graph.invoke({"query": "What's the population of Tokyo?"})
print(result["result"])  # Routes to search

result = graph.invoke({"query": "Calculate 15% of 2000"})
print(result["result"])  # Routes to math
```

### Example 3: Looping with Counter

Agent that can loop (with limits):


```python
class LoopState(TypedDict):
    iteration: int
    data: str
    final_result: str

def process_step(state: LoopState) -> dict:
    """Do one iteration of processing."""
    processed = state["data"] + f" [step-{state['iteration']}]"
    return {
        "data": processed,
        "iteration": state["iteration"] + 1
    }

def should_continue(state: LoopState) -> str:
    """Decide whether to loop or finish."""
    if state["iteration"] >= 3:
        return "finish"
    return "continue"

def finalize(state: LoopState) -> dict:
    """Final processing."""
    return {"final_result": state["data"]}

builder = StateGraph(LoopState)
builder.add_node("process", process_step)
builder.add_node("finalize", finalize)

builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Loop back to self
        "finish": "finalize"
    }
)
builder.add_edge("finalize", END)

graph = builder.compile()

# Looping with safeguard
config = {"configurable": {"thread_id": "loop-test"}}
result = graph.invoke(
    {"iteration": 0, "data": "start"},
    config=config
)
print(result)
# Output: {'iteration': 3, 'data': 'start [step-0] [step-1] [step-2]', 'final_result': '...'}
```


### Example 4: Streaming Output

See the graph execute step-by-step:


```python
# Different streaming modes
config = {"configurable": {"thread_id": "stream-test"}}

# Mode 1: "values" - full state after each step
print("=== Streaming Values ===")
for event in graph.stream(
    {"iteration": 0, "data": "test"},
    config=config,
    stream_mode="values"
):
    print(f"State: {event}\n")

# Mode 2: "updates" - only what changed
print("\n=== Streaming Updates ===")
for event in graph.stream(
    {"iteration": 0, "data": "test"},
    config=config,
    stream_mode="updates"
):
    for node_name, updates in event.items():
        print(f"{node_name} updated: {updates}\n")

# Mode 3: "debug" - node execution trace
print("\n=== Debug Mode ===")
for event in graph.stream(
    {"iteration": 0, "data": "test"},
    config=config,
    stream_mode="debug"
):
    print(f"Debug: {event}\n")
```


---

## Multi-Agent Systems

### Example 1: Supervisor Pattern

One coordinator agent routing to specialists:


```python
from langchain_core.messages import BaseMessage
# Note: AgentExecutor and create_tool_calling_agent require `pip install langchain langchain-anthropic`
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from langchain_core.tools import tool
from typing import List

# Define specialized agents' tools
@tool
def research_tool(query: str) -> str:
    """Search the web for information."""
    return f"Research results for: {query}"

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate math expressions."""
    # In a real scenario, use a safe evaluation library
    return str(eval(expression))

# Helper function to create a specialist agent
def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Create agent runner function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [BaseMessage(type="human", content=result["output"], name=name)]}

# Create specialized agents
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
research_agent = create_agent(model, [research_tool], "You are a research specialist. Find accurate information.")
math_agent = create_agent(model, [calculator_tool], "You are a math specialist. Solve problems step-by-step.")

# Supervisor state
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

# Supervisor logic
def supervisor_node(state: SupervisorState) -> dict:
    """Analyze request and pick best agent."""
    last_message = state["messages"][-1]
    
    # If the last message is from an agent, the supervisor can decide to end the process
    if hasattr(last_message, 'name'):
        return {"next": "END"}

    prompt = f"""You manage two specialist agents:
- research_agent: For web searches, fact-finding, current info
- math_agent: For calculations and equations

Request: {last_message.content}

Which agent should handle this? Reply with ONLY the agent name or FINISH."""
    
    response = model.invoke(prompt)
    next_agent = response.content.strip()
    
    return {"next": next_agent}

# Build supervisor graph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("research_agent", lambda state: agent_node(state, research_agent, "research_agent"))
builder.add_node("math_agent", lambda state: agent_node(state, math_agent, "math_agent"))

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "research_agent": "research_agent",
        "math_agent": "math_agent",
        "FINISH": END,
    }
)

# Agents return to supervisor
builder.add_edge("research_agent", "supervisor")
builder.add_edge("math_agent", "supervisor")

supervisor_graph = builder.compile(checkpointer=InMemorySaver())

# Test it
config = {"configurable": {"thread_id": "supervisor-test"}}

result = supervisor_graph.invoke(
    {"messages": [{"role": "user", "content": "Research AI trends and calculate 25% of 1000"}]},
    config=config
)

print("Final response:", result["messages"][-1].content)
```


### Example 2: Parallel Worker Pattern

Fan-out to multiple workers, collect results:


```python
from langgraph.types import Send

class WorkflowState(TypedDict):
    tasks: list[dict]
    results: Annotated[dict, lambda x, y: {**x, **y}]  # Merge dicts

def split_tasks(state: WorkflowState) -> list[Send]:
    """Create parallel work for each task."""
    return [
        Send(
            "worker",
            {
                "task_id": task["id"],
                "task_data": task["data"]
            }
        )
        for task in state["tasks"]
    ]

def worker_node(state: WorkflowState) -> dict:
    """Process one task."""
    # Simulate work
    result = f"Processed: {state['task_data']}"
    return {"results": {state["task_id"]: result}}

def collect_results(state: WorkflowState) -> dict:
    """Aggregate all results."""
    summary = f"Completed {len(state['results'])} tasks"
    return {"results": {"summary": summary}}

# Build parallel graph
builder = StateGraph(WorkflowState)
builder.add_node("split", split_tasks)
builder.add_node("worker", worker_node)
builder.add_node("collect", collect_results)

# Fan-out: split → multiple workers
builder.add_conditional_edges(
    START,
    lambda _: "split"
)
builder.add_conditional_edges(
    "split",
    lambda _: ["worker"],  # All Send objects go to worker
    ["worker"]
)

# Fan-in: collect all results
builder.add_edge("worker", "collect")
builder.add_edge("collect", END)

parallel_graph = builder.compile()

# Test
result = parallel_graph.invoke({
    "tasks": [
        {"id": "task-1", "data": "data-a"},
        {"id": "task-2", "data": "data-b"},
        {"id": "task-3", "data": "data-c"}
    ]
})

print("Results:", result["results"])
# Output: {'task-1': 'Processed: data-a', 'task-2': 'Processed: data-b', ...}
```


### Example 3: Handoff Pattern

Agents handing off to each other mid-conversation:


```python
class HandoffState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    handoff_reason: str

def agent_a(state: HandoffState) -> dict:
    """First agent - handles initial request."""
    last_message = state["messages"][-1].content
    
    # Check if should handoff
    if "transfer" in last_message.lower():
        return {
            "current_agent": "agent_b",
            "handoff_reason": "User requested transfer",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Transferring to agent B..."
                }
            ]
        }
    
    # Normal response
    response = f"Agent A responds to: {last_message}"
    return {
        "current_agent": "agent_a",
        "messages": [{"role": "assistant", "content": response}]
    }

def agent_b(state: HandoffState) -> dict:
    """Second agent - takes over."""
    last_message = state["messages"][-1].content
    response = f"Agent B (now handling): {last_message}"
    return {
        "current_agent": "agent_b",
        "messages": [{"role": "assistant", "content": response}]
    }

def route_agent(state: HandoffState) -> str:
    """Route to current agent."""
    agent = state.get("current_agent", "agent_a")
    return agent

# Build handoff graph
builder = StateGraph(HandoffState)
builder.add_node("agent_a", agent_a)
builder.add_node("agent_b", agent_b)

builder.add_edge(START, "agent_a")
builder.add_conditional_edges(
    "agent_a",
    lambda state: "agent_b" if state.get("current_agent") == "agent_b" else "agent_a"
)
builder.add_edge("agent_b", END)

handoff_graph = builder.compile(checkpointer=InMemorySaver())

# Test handoff
config = {"configurable": {"thread_id": "handoff-test"}}

result = handoff_graph.invoke(
    {"messages": [{"role": "user", "content": "Help me"}], "current_agent": "agent_a"},
    config=config
)
print("Step 1:", result["messages"][-1].content)

result = handoff_graph.invoke(
    {"messages": [{"role": "user", "content": "Transfer me to another agent"}]},
    config=config
)
print("Step 2:", result["messages"][-1].content)
print("Current agent:", result["current_agent"])
```


---

## Tool Integration

### Example 1: Basic Tool Node

Using LangGraph's built-in `ToolNode`:

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# Define tools
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price."""
    prices = {"AAPL": 150.25, "GOOGL": 140.50}
    return f"{symbol}: ${prices.get(symbol, 'N/A')}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}: {subject}"

tools = [get_weather, get_stock_price, send_email]

# Create model with tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools(tools)

class ToolState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_results: list[str]

def agent_node(state: ToolState) -> dict:
    """Call model which may invoke tools."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build graph with tool handling
builder = StateGraph(ToolState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

# tools_condition: Routes to "tools" if tool_calls exist, else END
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

# After tools, return to agent for next iteration
builder.add_edge("tools", "agent")

tool_graph = builder.compile()

# Use it
result = tool_graph.invoke({
    "messages": [
        {"role": "user", "content": "What's the weather in London and AAPL stock price?"}
    ]
})

print("Final response:", result["messages"][-1].content)
```

### Example 2: Custom Tool Executor

Handle tool execution yourself for more control:

```python
from langchain_core.messages import ToolMessage
import json

class CustomToolState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_errors: Annotated[list, lambda x, y: x + y]

def execute_tools(state: CustomToolState) -> dict:
    """Manually execute tool calls with error handling."""
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, "tool_calls"):
        return {}
    
    tool_results = []
    errors = []
    
    for tool_call in last_message.tool_calls:
        try:
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            
            if tool_name == "get_weather":
                result = get_weather(args["city"])
            elif tool_name == "get_stock_price":
                result = get_stock_price(args["symbol"])
            else:
                result = "Tool not found"
            
            tool_results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )
        except Exception as e:
            errors.append(f"Tool {tool_name} failed: {str(e)}")
            tool_results.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call["id"]
                )
            )
    
    return {
        "messages": tool_results,
        "tool_errors": errors if errors else []
    }

# Build with custom tool executor
builder = StateGraph(CustomToolState)
builder.add_node("agent", agent_node)
builder.add_node("tools", execute_tools)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    lambda state: "tools" if hasattr(state["messages"][-1], "tool_calls") else END,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "agent")

custom_tool_graph = builder.compile()
```

### Example 3: Conditional Tool Usage

Only use tools when needed:

```python
class ConditionalToolState(TypedDict):
    query: str
    use_tools: bool
    result: str

def should_use_tools(state: ConditionalToolState) -> str:
    """Decide whether tools are needed."""
    query = state["query"].lower()
    
    needs_tools = any(
        word in query 
        for word in ["weather", "stock", "email", "current", "today"]
    )
    
    return "use_tools" if needs_tools else "direct_response"

def with_tools(state: ConditionalToolState) -> dict:
    """Process with tool calling."""
    # Call model with tools bound
    response = model_with_tools.invoke(state["query"])
    return {"result": response.content, "use_tools": True}

def without_tools(state: ConditionalToolState) -> dict:
    """Process without tools."""
    response = model.invoke(state["query"])
    return {"result": response.content, "use_tools": False}

builder = StateGraph(ConditionalToolState)
builder.add_node("route", should_use_tools)
builder.add_node("with_tools", with_tools)
builder.add_node("without_tools", without_tools)

builder.add_edge(START, "route")
builder.add_conditional_edges(
    "route",
    should_use_tools,
    {
        "use_tools": "with_tools",
        "direct_response": "without_tools"
    }
)
builder.add_edge("with_tools", END)
builder.add_edge("without_tools", END)

conditional_tool_graph = builder.compile()

# Test
result = conditional_tool_graph.invoke({"query": "What's the weather?"})
print("Used tools:", result["use_tools"])  # True

result = conditional_tool_graph.invoke({"query": "Tell me a joke"})
print("Used tools:", result["use_tools"])  # False
```

---

## Memory & Persistence

### Short-Term Memory: Checkpointers

Checkpointers save graph state automatically at each step. Resume from failures.

#### In-Memory (Development)

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# State persists within this Python process only
# Useful for development & testing
```

#### SQLite (Local Persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# File-based
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Or in-memory SQLite
checkpointer = SqliteSaver.from_conn_string(":memory:")

graph = builder.compile(checkpointer=checkpointer)
```

#### PostgreSQL (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg2

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db"
)

# Async version
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async_checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db"
)

graph = builder.compile(checkpointer=checkpointer)
```

### Using Checkpoints


```python
config = {"configurable": {"thread_id": "user-123"}}

# First invocation
result = graph.invoke(
    {"query": "Start process"},
    config=config
)

# Check current state
current_state = graph.get_state(config)
print(f"Next node: {current_state.next}")
print(f"Values: {current_state.values}")
print(f"Checkpoint ID: {current_state.config['configurable']['checkpoint_id']}")

# Continue in same thread - state restored from checkpoint
result = graph.invoke(
    {"query": "Continue"},
    config=config
)

# Get state history (time-travel debugging)
history = graph.get_state_history(config)

for i, checkpoint in enumerate(history):
    cp_id = checkpoint.config['configurable']['checkpoint_id']
    print(f"Step {i}: {cp_id}")
    print(f"  State: {checkpoint.values}")

# Resume from specific checkpoint (time-travel)
old_checkpoint_id = history[1].config['configurable']['checkpoint_id']
time_travel_config = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": old_checkpoint_id
    }
}

# Continue from that point in history
result = graph.invoke(
    {"query": "New direction"},
    config=time_travel_config
)
```


### Long-Term Memory: Store

Store provides cross-thread, persistent key-value storage with hierarchical namespaces:

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore

# In-memory for development
store = InMemoryStore()

# PostgreSQL for production (with vector search)
store = AsyncPostgresStore.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db"
)

# Store operations
namespace = ("users", "user-123", "preferences")

# Put data
await store.aput(
    namespace=namespace,
    key="theme",
    value={"dark_mode": True, "language": "en"}
)

# Get data
item = await store.aget(namespace, "theme")
print(item.value)  # {"dark_mode": True, ...}

# List all in namespace
items = await store.asearch(namespace_prefix=namespace)
for item in items:
    print(f"{item.key}: {item.value}")

# Delete
await store.adelete(namespace, "theme")

# Store with vector search for semantic retrieval
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
store_with_search = AsyncPostgresStore.from_conn_string(
    "postgresql://user:password@localhost/langgraph_db",
    embeddings=embeddings
)

# Store documents with embeddings
await store_with_search.aput(
    namespace=("docs", "kb"),
    key="api-guide",
    value={
        "title": "API Guide",
        "content": "LangGraph provides APIs for building stateful agents..."
    },
    index=["content"]  # Fields to embed
)

# Semantic search
results = await store_with_search.asearch(
    namespace_prefix=("docs",),
    query="how to build agents",
    limit=5
)

for result in results:
    print(f"Score: {result.score}, {result.value['title']}")
```

### Injecting Store into Nodes

Use LangGraph's dependency injection:

```python
from langgraph.prebuilt import InjectedStore
from typing import Annotated

def personalization_node(
    state: State,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Node that accesses store automatically."""
    user_id = state["user_id"]
    
    # Retrieve preferences
    namespace = ("users", user_id, "prefs")
    prefs_item = await store.aget(namespace, "theme")
    prefs = prefs_item.value if prefs_item else {}
    
    # Update if interaction changes preferences
    if state.get("user_voted_dark"):
        await store.aput(
            namespace,
            "theme",
            {"dark_mode": True, "last_updated": datetime.now().isoformat()}
        )
    
    return {"user_preferences": prefs}

# Compile with store
builder = StateGraph(State)
builder.add_node("personalize", personalization_node)

graph = builder.compile(store=store)
```

### Complete Memory Example


```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from datetime import datetime

class MemoryState(TypedDict):
    user_id: str
    message: str
    response: str
    conversation_history: Annotated[list, add_messages]

def store_memory_node(
    state: MemoryState,
    store: Annotated[InMemoryStore, InjectedStore]
) -> dict:
    """Store user preferences and conversation summary."""
    
    # Extract user preferences from conversation
    namespace = ("users", state["user_id"], "memory")
    
    # Save conversation turn
    await store.aput(
        namespace,
        f"turn-{datetime.now().isoformat()}",
        {
            "user_message": state["message"],
            "bot_response": state["response"]
        }
    )
    
    # Update user profile based on interactions
    profile_key = "profile"
    profile = await store.aget(namespace, profile_key)
    existing = profile.value if profile else {}
    
    updated_profile = {
        **existing,
        "total_turns": existing.get("total_turns", 0) + 1,
        "last_interaction": datetime.now().isoformat()
    }
    
    await store.aput(namespace, profile_key, updated_profile)
    
    return {}

# Build with memory
checkpointer = SqliteSaver.from_conn_string("memory.db")
store = InMemoryStore()

builder = StateGraph(MemoryState)
builder.add_node("respond", respond_node)
builder.add_node("remember", store_memory_node)

builder.add_edge(START, "respond")
builder.add_edge("respond", "remember")
builder.add_edge("remember", END)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)

# Use with persistence
config = {"configurable": {"thread_id": "user-alice"}}

for i in range(3):
    result = graph.invoke(
        {"user_id": "alice", "message": f"Message {i}"},
        config=config
    )
    print(result["response"])
    
# Multi-turn conversations remembered automatically
```


---

## Debugging & Visualization

### Graph Visualization

```python
from IPython.display import Image, display

# Get Mermaid diagram
diagram = graph.get_graph().draw_mermaid()
print(diagram)

# Display in Jupyter/Colab
display(Image(graph.get_graph().draw_mermaid_png()))

# ASCII art
print(graph.get_graph().draw_ascii())
```

Example output:

```
    ┌─────────────────────┐
    │      START          │
    └────────────┬────────┘
                 │
    ┌────────────▼────────────┐
    │     fetch_context       │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │     call_model          │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │      save_chat          │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │        END              │
    └─────────────────────────┘
```

### Streaming for Debugging


```python
# Debug mode shows node execution
config = {"configurable": {"thread_id": "debug-1"}}

for event in graph.stream(
    {"query": "test"},
    config=config,
    stream_mode="debug"
):
    print(f"Event: {event}")

# Output:
# {'type': 'task_start', 'timestamp': '...', 'step': 0, 'node': 'fetch_context'}
# {'type': 'task_end', 'timestamp': '...', 'step': 0, 'node': 'fetch_context', 'result': {...}}
# {'type': 'task_start', 'timestamp': '...', 'step': 1, 'node': 'call_model'}
# ...
```


## Type-Safe v2 API (v1.1.x)

### v2 Streaming

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    messages: list
    result: str

builder = StateGraph(State)
# ... add nodes and edges ...
graph = builder.compile()

# v2 streaming: opt-in with version="v2"
async for part in graph.astream(
    {"messages": [{"role": "user", "content": "Hello"}]},
    version="v2",  # Enables type-safe StreamPart output
):
    # part is a StreamPart with .type, .ns, and .data
    print(f"Type: {part.type}, Data: {part.data}")
```

### v2 Invoke

```python
# v2 invoke returns a GraphOutput instead of a dict
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    version="v2",
)

# GraphOutput has .value (final state) and .interrupts (any Human-in-the-Loop interrupts)
print(result.value)       # Final state dict
print(result.interrupts)  # List of interrupt points (if any)
```

### Pydantic/Dataclass Auto-Coercion

v1.1.x automatically coerces input dictionaries to the graph's state type on `invoke()`:

```python
from pydantic import BaseModel

class MyState(BaseModel):
    query: str
    result: str = ""

builder = StateGraph(MyState)
# ... graph setup ...
graph = builder.compile()

# Pass dict directly — auto-coerced to MyState
result = await graph.ainvoke({"query": "What is LangGraph?"})
print(type(result))  # MyState
```

### Getting State at Any Point


```python
# After partial execution
config = {"configurable": {"thread_id": "user-1"}}

# Start but intercept in middle
for event in graph.stream({"query": "test"}, config=config):
    pass

# Get state snapshot
state = graph.get_state(config)
print(f"Next node to run: {state.next}")
print(f"Current values: {state.values}")
print(f"Metadata: {state.metadata}")

# Modify state
graph.update_state(
    config,
    {"messages": [{"role": "system", "content": "Updated system prompt"}]}
)

# Continue from modified state
result = graph.invoke({"query": "continue"}, config=config)
```


### Checkpoint Inspection


```python
# List all checkpoints for a thread
config = {"configurable": {"thread_id": "user-1"}}

history = list(graph.get_state_history(config))
print(f"Total checkpoints: {len(history)}")

for i, snapshot in enumerate(history):
    cp_id = snapshot.config['configurable']['checkpoint_id']
    next_node = snapshot.next
    print(f"\nCheckpoint {i}: {cp_id}")
    print(f"  Next node(s): {next_node}")
    print(f"  State keys: {list(snapshot.values.keys())}")
```


### Batch Debugging


```python
# Process multiple and collect issues
inputs = [
    {"query": "Query 1"},
    {"query": "Query 2"},
    {"query": "Query 3"}
]

configs = [
    {"configurable": {"thread_id": f"batch-{i}"}}
    for i in range(len(inputs))
]

results = []
errors = []

for inp, cfg in zip(inputs, configs):
    try:
        result = graph.invoke(inp, config=cfg)
        results.append(result)
    except Exception as e:
        errors.append((cfg["configurable"]["thread_id"], str(e)))

print(f"Successful: {len(results)}/{len(inputs)}")
print(f"Failed: {len(errors)}")
for thread_id, error in errors:
    print(f"  {thread_id}: {error}")
```


---

## Human-in-the-Loop

### Basic Interrupts

Pause execution and request human input:


```python
from langgraph.types import interrupt, Command

class ApprovalState(TypedDict):
    action: str
    amount: float
    approved: bool
    approval_reason: str

def request_approval(state: ApprovalState) -> dict:
    """Pause and ask human for approval."""
    
    # Interrupt with information for human
    result = interrupt({
        "action": state["action"],
        "amount": state["amount"],
        "message": f"Approve {state['action']} for ${state['amount']}?"
    })
    
    # result contains human's response
    return {
        "approved": result.get("approved", False),
        "approval_reason": result.get("reason", "")
    }

def execute_action(state: ApprovalState) -> dict:
    """Execute if approved."""
    if state["approved"]:
        return {"action": f"Executed {state['action']}"}
    else:
        return {"action": f"Rejected {state['action']}"}

# Build with interrupts
builder = StateGraph(ApprovalState)
builder.add_node("request_approval", request_approval)
builder.add_node("execute", execute_action)

builder.add_edge(START, "request_approval")
builder.add_edge("request_approval", "execute")
builder.add_edge("execute", END)

# MUST compile with checkpointer for interrupts
checkpointer = InMemorySaver()
approval_graph = builder.compile(checkpointer=checkpointer)

# Usage
config = {"configurable": {"thread_id": "approval-1"}}

# Start - will interrupt
events = []
for event in approval_graph.stream(
    {"action": "transfer", "amount": 500.00},
    config=config
):
    events.append(event)
    
print(events)
# Output: [{'__interrupt__': (Interrupt(...), )}]

# Check if interrupted
state = approval_graph.get_state(config)
if state.next == ("__interrupt__",):
    print("Waiting for human approval")
    
    # Human decides
    human_decision = {
        "approved": True,
        "reason": "Amount looks reasonable"
    }
    
    # Resume with decision
    resume_events = list(approval_graph.stream(
        Command(resume=human_decision),
        config=config
    ))
    
    print(resume_events)  # Graph continues
```


### Multi-Step Approval Workflow


```python
from enum import Enum

class ApprovalStage(Enum):
    INITIAL_REVIEW = "initial"
    COMPLIANCE_CHECK = "compliance"
    FINAL_APPROVAL = "final"

class WorkflowApprovalState(TypedDict):
    action: str
    amount: float
    approval_stage: ApprovalStage
    approvals: Annotated[dict, lambda x, y: {**x, **y}]

def initial_review_node(state: WorkflowApprovalState) -> dict:
    """First level approval."""
    
    approval = interrupt({
        "stage": "INITIAL",
        "question": f"Review {state['action']} for ${state['amount']}?",
        "reviewer_type": "manager"
    })
    
    return {
        "approvals": {"initial": approval.get("approved")},
        "approval_stage": ApprovalStage.COMPLIANCE_CHECK
    }

def compliance_check_node(state: WorkflowApprovalState) -> dict:
    """Second level - compliance."""
    
    # Only ask if initial approved
    if not state["approvals"].get("initial"):
        return {
            "approval_stage": ApprovalStage.FINAL_APPROVAL,
            "approvals": {"compliance": False}
        }
    
    approval = interrupt({
        "stage": "COMPLIANCE",
        "question": "Compliance clearance needed",
        "reviewer_type": "compliance_officer"
    })
    
    return {
        "approvals": {"compliance": approval.get("approved")},
        "approval_stage": ApprovalStage.FINAL_APPROVAL
    }

def final_approval_node(state: WorkflowApprovalState) -> dict:
    """Executive final approval."""
    
    all_approved = all(state["approvals"].values())
    
    if not all_approved:
        return {"approvals": {"final": False}}
    
    approval = interrupt({
        "stage": "FINAL",
        "question": "Executive approval required",
        "reviewer_type": "executive"
    })
    
    return {"approvals": {"final": approval.get("approved")}}

def execute_if_approved(state: WorkflowApprovalState) -> dict:
    """Only run if all approvals granted."""
    
    all_approved = all(state["approvals"].values())
    
    if all_approved:
        # Execute action
        return {"action": f"EXECUTED: {state['action']}"}
    else:
        return {"action": f"REJECTED: {state['action']}"}

# Build workflow
builder = StateGraph(WorkflowApprovalState)
builder.add_node("initial", initial_review_node)
builder.add_node("compliance", compliance_check_node)
builder.add_node("final", final_approval_node)
builder.add_node("execute", execute_if_approved)

builder.add_edge(START, "initial")
builder.add_edge("initial", "compliance")
builder.add_edge("compliance", "final")
builder.add_edge("final", "execute")
builder.add_edge("execute", END)

approval_workflow = builder.compile(checkpointer=InMemorySaver())

# Multi-stage execution
config = {"configurable": {"thread_id": "multi-approval-1"}}

# Stage 1
stream_events(approval_workflow.stream(
    {"action": "hire", "amount": 80000},
    config=config
))

# Resume with manager approval
stream_events(approval_workflow.stream(
    Command(resume={"approved": True}),
    config=config
))

# Resume with compliance approval
stream_events(approval_workflow.stream(
    Command(resume={"approved": True}),
    config=config
))

# Resume with executive approval
stream_events(approval_workflow.stream(
    Command(resume={"approved": True}),
    config=config
))
```


### Interactive Debugging


```python
class DebugState(TypedDict):
    data: str
    step_result: str
    needs_adjustment: bool

def step_node(state: DebugState) -> dict:
    """Process data."""
    
    result = process(state["data"])
    
    # Ask if result is acceptable
    feedback = interrupt({
        "step": "Process",
        "result": result,
        "question": "Is this result acceptable? (yes/no/modify)"
    })
    
    if feedback["action"] == "modify":
        result = feedback["modified_result"]
        needs_adjustment = True
    else:
        needs_adjustment = feedback["action"] != "yes"
    
    return {
        "step_result": result,
        "needs_adjustment": needs_adjustment
    }

def decide_continue(state: DebugState) -> str:
    """Route based on feedback."""
    return "refine" if state["needs_adjustment"] else "finalize"

# Build interactive debug workflow
builder = StateGraph(DebugState)
builder.add_node("process", step_node)
builder.add_node("refine", refine_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "process")
builder.add_conditional_edges(
    "process",
    decide_continue,
    {"refine": "refine", "finalize": "finalize"}
)
builder.add_edge("refine", "process")
builder.add_edge("finalize", END)

debug_workflow = builder.compile(checkpointer=InMemorySaver())

# Interactive use
config = {"configurable": {"thread_id": "debug-session"}}

# Step through with feedback
stream_events(debug_workflow.stream(
    {"data": "raw_input"},
    config=config
))

# Human reviews and responds with modifications
stream_events(debug_workflow.stream(
    Command(resume={"action": "modify", "modified_result": "adjusted_output"}),
    config=config
))
```


---

## Advanced Patterns

### Pattern 1: ReAct (Reasoning + Acting)

The Reflection-Action pattern for autonomous agents, now built with modern LangChain components.

```python
# Note: AgentExecutor and create_tool_calling_agent require `pip install langchain langchain-anthropic`
# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for {query}..."

@tool
def calculator(expression: str) -> str:
    """Calculate expression."""
    return str(eval(expression))

tools = [search_web, calculator]

# Create the ReAct agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful research assistant. Think before acting."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
react_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Use it - the AgentExecutor automatically handles the ReAct loop
result = react_agent.invoke({
    "input": "Research population of Tokyo and calculate 15% of that",
    "chat_history": []
})

print(result["output"])
```

### Pattern 2: Tree-of-Thoughts

Explore multiple reasoning paths:


```python
from langgraph.types import Send

class ThoughtState(TypedDict):
    question: str
    thoughts: Annotated[list[dict], lambda x, y: x + y]
    best_thought: dict
    final_answer: str

def generate_thoughts(state: ThoughtState) -> list[Send]:
    """Generate multiple solution approaches."""
    
    num_paths = 3
    returns = []
    
    for i in range(num_paths):
        returns.append(
            Send("explore_thought", {
                "question": state["question"],
                "path_number": i
            })
        )
    
    return returns

def explore_thought(state: ThoughtState) -> dict:
    """Explore one reasoning path."""
    
    prompt = f"""
    Question: {state['question']}
    Path #{state.get('path_number', 0)}
    
    Provide your reasoning for this specific approach.
    """
    
    response = model.invoke(prompt)
    
    return {
        "thoughts": [{
            "path": state.get("path_number"),
            "reasoning": response.content,
            "quality_score": 0.8  # Could be evaluated
        }]
    }

def select_best(state: ThoughtState) -> dict:
    """Select the best thought."""
    
    if not state["thoughts"]:
        return {"best_thought": {}}
    
    best = max(state["thoughts"], key=lambda x: x.get("quality_score", 0))
    
    return {"best_thought": best}

def synthesize(state: ThoughtState) -> dict:
    """Synthesize best thought into answer."""
    
    best_reasoning = state["best_thought"].get("reasoning", "")
    
    prompt = f"""
    Best reasoning: {best_reasoning}
    
    Provide a final answer based on this reasoning.
    """
    
    response = model.invoke(prompt)
    
    return {"final_answer": response.content}

# Build tree-of-thoughts
builder = StateGraph(ThoughtState)
builder.add_node("generate", generate_thoughts)
builder.add_node("explore", explore_thought)
builder.add_node("select", select_best)
builder.add_node("synthesize", synthesize)

builder.add_conditional_edges(
    START,
    lambda _: "generate"
)
builder.add_conditional_edges(
    "generate",
    lambda _: ["explore"],
    ["explore"]
)
builder.add_edge("explore", "select")
builder.add_edge("select", "synthesize")
builder.add_edge("synthesize", END)

tot_graph = builder.compile()

# Use it
result = tot_graph.invoke({
    "question": "How should we approach climate change?"
})

print("Best thought:", result["best_thought"]["reasoning"])
print("Final answer:", result["final_answer"])
```


### Pattern 3: Self-Reflection

Agent critiques its own output:

```python
class ReflectionState(TypedDict):
    question: str
    initial_response: str
    critique: str
    refined_response: str
    reflection_count: int

def generate_response(state: ReflectionState) -> dict:
    """Generate initial response."""
    
    response = model.invoke(state["question"])
    
    return {
        "initial_response": response.content,
        "reflection_count": 0
    }

def self_critique(state: ReflectionState) -> dict:
    """Critique the response."""
    
    prompt = f"""
    Question: {state['question']}
    Response: {state['initial_response']}
    
    Critique this response. What could be improved?
    """
    
    critique = model.invoke(prompt)
    
    return {"critique": critique.content}

def should_refine(state: ReflectionState) -> str:
    """Decide if response needs refinement."""
    
    if state["reflection_count"] >= 2:
        return "done"
    
    # Check critique for issues
    if any(word in state["critique"].lower() 
           for word in ["incorrect", "missing", "unclear", "incomplete"]):
        return "refine"
    
    return "done"

def refine_response(state: ReflectionState) -> dict:
    """Create refined response based on critique."""
    
    prompt = f"""
    Original question: {state['question']}
    Your response: {state['initial_response']}
    Critique: {state['critique']}
    
    Provide an improved response addressing the critique.
    """
    
    refined = model.invoke(prompt)
    
    return {
        "refined_response": refined.content,
        "reflection_count": state["reflection_count"] + 1
    }

# Build reflection loop
builder = StateGraph(ReflectionState)
builder.add_node("generate", generate_response)
builder.add_node("critique", self_critique)
builder.add_node("refine", refine_response)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critique")

builder.add_conditional_edges(
    "critique",
    should_refine,
    {"refine": "refine", "done": END}
)

builder.add_edge("refine", "critique")  # Loop back for re-critique

reflection_graph = builder.compile()

# Use it
result = reflection_graph.invoke({
    "question": "Explain quantum computing to a child"
})

print("Initial:", result["initial_response"])
print("Refined:", result.get("refined_response", "No refinement needed"))
print("Reflection iterations:", result["reflection_count"])
```

### Pattern 4: Structured Output with Validation

```python
from pydantic import BaseModel, field_validator

class ResearchOutput(BaseModel):
    """Structured research output."""
    topic: str
    key_findings: list[str]
    sources: list[str]
    confidence_score: float
    
    @field_validator('confidence_score')
    def score_in_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Must be between 0 and 1')
        return v

class StructuredState(TypedDict):
    topic: str
    raw_research: str
    structured_output: ResearchOutput
    validation_passed: bool
    errors: list[str]

def research_node(state: StructuredState) -> dict:
    """Conduct research."""
    
    result = model.invoke(f"Research: {state['topic']}")
    
    return {"raw_research": result.content}

def structure_output(state: StructuredState) -> dict:
    """Parse into structured format."""
    
    prompt = f"""
    Research content: {state['raw_research']}
    
    Extract into JSON with fields:
    - topic
    - key_findings (list)
    - sources (list)
    - confidence_score (0-1)
    """
    
    response = model.invoke(prompt)
    
    try:
        import json
        parsed = json.loads(response.content)
        output = ResearchOutput(**parsed)
        return {
            "structured_output": output,
            "validation_passed": True,
            "errors": []
        }
    except Exception as e:
        return {
            "validation_passed": False,
            "errors": [str(e)]
        }

def decide_next(state: StructuredState) -> str:
    """Route based on validation."""
    if state["validation_passed"]:
        return "success"
    else:
        return "retry"

def retry_node(state: StructuredState) -> dict:
    """Re-attempt with error context."""
    
    prompt = f"""
    Previous errors: {', '.join(state['errors'])}
    Retry research on: {state['topic']}
    """
    
    result = model.invoke(prompt)
    
    return {"raw_research": result.content}

# Build validation graph
builder = StateGraph(StructuredState)
builder.add_node("research", research_node)
builder.add_node("structure", structure_output)
builder.add_node("retry", retry_node)

builder.add_edge(START, "research")
builder.add_edge("research", "structure")

builder.add_conditional_edges(
    "structure",
    decide_next,
    {"success": END, "retry": "retry"}
)

builder.add_edge("retry", "structure")  # Loop back

validation_graph = builder.compile()

# Use it
result = validation_graph.invoke({
    "topic": "AI safety"
})

if result["validation_passed"]:
    output = result["structured_output"]
    print(f"Topic: {output.topic}")
    print(f"Confidence: {output.confidence_score}")
    print(f"Findings: {output.key_findings}")
```

### Pattern 5: Caching and Memoization


```python
from functools import lru_cache
from langgraph.store.memory import InMemoryStore

class CacheState(TypedDict):
    query: str
    result: str
    cache_hit: bool

# Simple LRU cache for expensive operations
@lru_cache(maxsize=128)
def expensive_operation(query: str) -> str:
    """Simulate expensive operation."""
    import time
    time.sleep(1)
    return f"Result for {query}"

async def cached_operation_node(
    state: CacheState,
    store: Annotated[InMemoryStore, InjectedStore]
) -> dict:
    """Check cache before executing."""
    
    query = state["query"]
    namespace = ("cache", "results")
    
    # Check cache
    cached = await store.aget(namespace, query)
    
    if cached:
        return {
            "result": cached.value,
            "cache_hit": True
        }
    
    # Execute and cache
    result = expensive_operation(query)
    
    await store.aput(
        namespace,
        query,
        {"result": result, "timestamp": datetime.now().isoformat()}
    )
    
    return {
        "result": result,
        "cache_hit": False
    }

# Build with caching
builder = StateGraph(CacheState)
builder.add_node("process", cached_operation_node)

caching_graph = builder.compile(store=InMemoryStore())

# Usage
config = {"configurable": {"thread_id": "cache-test"}}

# First call - hits expensive operation
result = caching_graph.invoke({"query": "expensive"}, config=config)
print("Cache hit:", result["cache_hit"])  # False

# Second call - uses cache
result = caching_graph.invoke({"query": "expensive"}, config=config)
print("Cache hit:", result["cache_hit"])  # True
```


---


## Errata: removed fabricated sections

The following subsections appeared in earlier drafts of this guide under a "v1.0.3 Features" heading but do not match any real API in the installed `langgraph==1.2.0` package. They have been removed:

- **Node Caching** — `from langgraph.cache import cache_node, SemanticCache, CachePolicy` does not exist. For caching, use LangGraph's long-term `Store` (see [Memory & Persistence](#memory--persistence)) or plain `functools.lru_cache`.
- **Deferred Nodes** — `from langgraph.graph import deferred` and `@deferred(wait_for=[...])` are not real. Fan-in is native: edges from multiple sources into the same target wait for all upstream completions.
- **Pre/Post Model Hooks decorators** — `from langgraph.llm_hooks import pre_model_hook, post_model_hook` does not exist. The real middleware API lives in `langchain.agents.middleware` and is used via `langchain.agents.create_agent(middleware=[...])`. The older `langgraph.prebuilt.create_react_agent` function also accepts `pre_model_hook=` / `post_model_hook=` keyword arguments (not decorators). See [Chapter 8 — Middleware](/langgraph-guide/python/chapter-08-middleware-hooks/) for details.
- **Tools State Updates** — `@tool(updates_state=True)` returning `StateUpdate` is not a real decorator option. Have your node read the tool result and return the state update as a normal dict.
- **Command Tool for edgeless flows** — `command_tool`, `CommandRouter` are not real. Real equivalent: return a `langgraph.types.Command(goto="next_node", update={...})` from a node or a tool to drive routing.
- **LangGraph Templates CLI** — `langgraph template list|create|init|publish` is not a real subcommand. Use `langgraph new --template NAME` to scaffold from a template.

What's real and remains documented:

- **Cross-thread memory** — use `langgraph.store.postgres.AsyncPostgresStore` with hierarchical namespaces, and the `InjectedStore` annotation to inject the store into node signatures. Covered in [Chapter 5 — Memory & Persistence](/langgraph-guide/python/chapter-05-memory/#cross-thread-memory-v103).
- **Python 3.10 – 3.14** — LangGraph 1.1.x supports Python 3.10 through 3.14 (Python 3.9 was dropped in 1.1). Type-parameter syntax (PEP 695) works as-is; there's no LangGraph-specific coupling.
- **Type-safe v2 API** — opt in with `version="v2"` on `.invoke` / `.stream` / `.ainvoke` / `.astream`. Covered in [Type-Safe v2 API](#type-safe-v2-api-v11x) above.

## Functional API (LangGraph 1.0)

A simpler Python-native way to build workflows with automatic parallelization:


```python
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Optional

# Define parallelizable tasks
@task
def fetch_user_data(user_id: str) -> dict:
    """Get user info."""
    return {"user_id": user_id, "name": "Alice"}

@task
def fetch_orders(user_id: str) -> list[dict]:
    """Get user orders."""
    return [{"id": "1", "total": 99.99}]

@task
async def generate_recommendations(user_data: dict, orders: list) -> list[str]:
    """Generate recommendations (can be async)."""
    return ["Product A", "Product B"]

# Define entrypoint with automatic parallelization
@entrypoint(checkpointer=InMemorySaver())
def build_dashboard(user_id: str, *, previous: Optional[dict] = None) -> dict:
    """
    Build dashboard with parallel data fetching.
    
    Args:
        user_id: User to fetch data for
        previous: Return value from last invocation (enables state)
    
    Returns:
        Complete dashboard data
    """
    
    # Launch tasks in parallel - immediately get futures
    user_future = fetch_user_data(user_id)
    orders_future = fetch_orders(user_id)
    
    # Block and wait for results
    user_data = user_future.result()
    orders = orders_future.result()
    
    # Now generate recommendations using results
    recs_future = generate_recommendations(user_data, orders)
    recommendations = recs_future.result()
    
    # Can interrupt for human approval
    approved = interrupt({
        "recommendations": recommendations,
        "question": "Approve these recommendations?"
    })
    
    return {
        "user": user_data,
        "orders": orders,
        "recommendations": recommendations if approved else [],
        "status": "approved" if approved else "rejected"
    }

# Execute
config = {"configurable": {"thread_id": "user-session-1"}}

# Initial run - interrupts for approval
for result in build_dashboard.stream("user-123", config):
    print(result)

# Resume after human approval
for result in build_dashboard.stream(Command(resume=True), config):
    print(result)

# With previous state for stateful workflows
@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: Optional[int] = None) -> str:
    """Accumulate counter."""
    current = (previous or 0) + increment
    return f"Counter: {current}"

config = {"configurable": {"thread_id": "counter"}}
counter.invoke(5, config)    # "Counter: 5"
counter.invoke(3, config)    # "Counter: 8" (5+3)
```


---

## Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start LangGraph server
CMD ["langgraph", "run", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t my-agent:v1 .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  my-agent:v1
```

### CLI Configuration

```json
{
  "langgraph.json": {
    "dependencies": [
      "langchain_anthropic",
      "langchain_tavily",
      "./agents"
    ],
    "graphs": {
      "main_agent": "./agents.py:graph",
      "research_agent": "./agents.py:research_graph"
    },
    "env": "./.env",
    "python_version": "3.11"
  }
}
```

### Remote Execution via SDK

```python
from langgraph_sdk import get_client
import asyncio

async def main():
    client = get_client(url="https://my-deployment.langraph.app")
    
    # List available assistants (from langgraph.json graphs)
    assistants = await client.assistants.search()
    assistant_id = assistants[0]["assistant_id"]
    
    # Create conversation thread
    thread = await client.threads.create()
    
    # Stream execution
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input={"query": "Research AI trends"}
    ):
        if chunk.event == "messages/partial":
            print(chunk.data[0]["content"], end="", flush=True)
    
    # Get final state
    final_state = await client.threads.get_state(thread["thread_id"])
    print(f"\nFinal: {final_state}")

asyncio.run(main())
```

---

## New in v1.2.7 — Compiled Graph APIs

### Graph Visualization

`get_graph(xray=True)` expands compiled subgraphs inline, prefixing their node names with the parent node name. All visualization flows through `langchain_core.runnables.graph.Graph`:

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

# Build a subgraph
sub = StateGraph(State)
sub.add_node("sub_op", lambda s: {"value": s["value"] * 10})
sub.set_entry_point("sub_op")
sub.set_finish_point("sub_op")
compiled_sub = sub.compile()

# Embed in main graph
g = StateGraph(State)
g.add_node("scale", compiled_sub)
g.add_node("post",  lambda s: {"value": s["value"] + 1})
g.add_edge(START, "scale")
g.add_edge("scale", "post")
g.add_edge("post", END)
compiled = g.compile()

# Shallow view — subgraph appears as single "scale" node
print(sorted(compiled.get_graph().nodes.keys()))
# ['__end__', '__start__', 'post', 'scale']

# Deep view — subgraph internals exposed
print(sorted(compiled.get_graph(xray=True).nodes.keys()))
# ['__end__', '__start__', 'post', 'scale:sub_op']

# Generate Mermaid markdown for documentation
mermaid_md = compiled.get_graph().draw_mermaid()
print(mermaid_md[:200])
```

### Graph-as-Tool (Beta)

`compiled.as_tool()` converts any compiled graph into a LangChain `StructuredTool`:

```python
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

class SummaryState(TypedDict):
    text: str
    word_count: int

class SummaryInput(BaseModel):
    text: str = Field(description="Text to analyze")

g = StateGraph(SummaryState)
g.add_node("count", lambda s: {"word_count": len(s["text"].split())})
g.set_entry_point("count")
g.set_finish_point("count")
compiled = g.compile()

import warnings
from langchain_core._api import LangChainBetaWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=LangChainBetaWarning)
    word_count_tool = compiled.as_tool(
        args_schema=SummaryInput,
        name="word_count",
        description="Count words in a piece of text",
    )
result = word_count_tool.invoke({"text": "LangGraph is great for building agents"})
print("Word count:", result["word_count"])  # 7
```

### Subgraph Traversal

```python
# `compiled` is any CompiledStateGraph returned by g.compile()
# Iterate all subgraphs, optionally recursing into nested ones
for name, subgraph in compiled.get_subgraphs(recurse=True):
    print(f"  namespace={name!r}  type={type(subgraph).__name__}")
```

### Schema Introspection

```python
import json
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

class Ctx(BaseModel):
    user_id: str

class S(TypedDict):
    query: str
    answer: str

g = StateGraph(S, context_schema=Ctx)
g.add_node("n", lambda s: s)
g.set_entry_point("n")
g.set_finish_point("n")
compiled = g.compile()

print(json.dumps(compiled.get_input_jsonschema(),   indent=2))
print(json.dumps(compiled.get_output_jsonschema(),  indent=2))
print(json.dumps(compiled.get_context_jsonschema(), indent=2))
# Note: get_config_jsonschema() is deprecated — use get_context_jsonschema()
```

### Deferred Nodes

`add_node(defer=True)` ensures a node runs at graph quiescence — after all non-deferred work in the entire run has drained, not just the current super-step:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    items: Annotated[list[str], operator.add]
    summary: str

g = StateGraph(S)
g.add_node("worker_a", lambda s: {"items": ["a"]})
g.add_node("worker_b", lambda s: {"items": ["b"]})
g.add_node("summarise", lambda s: {"summary": f"got {s['items']}"}, defer=True)

g.add_edge(START, "worker_a")
g.add_edge(START, "worker_b")
g.add_edge("worker_a", "summarise")   # only one explicit edge — defer=True waits for worker_b too
g.add_edge("summarise", END)

result = g.compile().invoke({"items": [], "summary": ""})
print(result["summary"])  # got ['a', 'b']  — both items present when defer runs
```

### Cache Management

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

class S(TypedDict):
    x: int

cache = InMemoryCache()

g = StateGraph(S)
g.add_node("compute", lambda s: {"x": s["x"] * 2}, cache_policy=CachePolicy())
g.set_entry_point("compute")
g.set_finish_point("compute")
compiled = g.compile(cache=cache)

# Invalidate the entire graph's cache
if compiled.cache is not None:
    compiled.clear_cache()

# Invalidate only specific nodes
if compiled.cache is not None:
    compiled.clear_cache(nodes=["compute"])

# Async variant
import asyncio
if compiled.cache is not None:
    asyncio.run(compiled.aclear_cache(nodes=["compute"]))
```

### REMOVE_ALL_MESSAGES Sentinel

The sentinel is processed by `add_messages` — the state key **must** be declared with that reducer (via `Annotated[list, add_messages]` or `MessagesState`). Without it, `RemoveMessage` is stored as a plain message rather than triggering a history wipe.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.graph import StateGraph, START, END

class ConvState(TypedDict):
    # add_messages reducer is required — it is the one that interprets the sentinel
    messages: Annotated[list[AnyMessage], add_messages]

# Discard entire conversation history and start fresh
def reset_conversation(state: ConvState) -> dict:
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),          # wipe all history
            HumanMessage(content="New topic", id="n1"),     # first message of new context
        ]
    }
```

---

## Common Patterns Summary

| Pattern | Use Case | Key Idea |
|---------|----------|----------|
| **Linear** | Simple pipelines | Node A → B → C → END |
| **Conditional** | Decision trees | Routes based on state |
| **Looping** | Iterations | Self-referencing edges with exit condition |
| **Supervisor** | Multi-agent | Central router to specialists |
| **Parallel** | Concurrent work | Fan-out with Send, fan-in with collection |
| **ReAct** | Autonomous agent | Reason → Action → Observe loop |
| **Tree-of-Thoughts** | Complex reasoning | Multiple parallel thought paths |
| **Reflection** | Quality improvement | Self-critique → Refine loop |
| **Interrupt** | Human approval | Pause, wait, resume with Command |
| **Caching** | Performance | Store expensive results |
| **Deferred nodes** | End-of-run aggregation | `add_node(defer=True)` runs at quiescence, after all non-deferred work drains |
| **Graph-as-tool** | Multi-agent composition | `compiled.as_tool()` wraps graph as a StructuredTool |
| **History reset** | Context window management | `RemoveMessage(id=REMOVE_ALL_MESSAGES)` clears all messages |

---

## Troubleshooting

### Issue: "Checkpointer must be provided for interrupts"

**Cause**: Trying to use `interrupt()` without a checkpointer  
**Fix**: Always compile with a checkpointer when using interrupts:

```python
graph = builder.compile(checkpointer=InMemorySaver())
```

### Issue: State not persisting across invocations

**Cause**: Missing `thread_id` in config  
**Fix**: Always provide consistent `thread_id`:


```python
config = {"configurable": {"thread_id": "unique-id"}}
result = graph.invoke(input, config=config)  # Same config each time
```


### Issue: Reducer functions not working

**Cause**: Not using `Annotated` with reducer function  
**Fix**: Proper state schema:

```python
# Wrong
class State(TypedDict):
    messages: list

# Correct
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

### Issue: Tools not being called

**Cause**: Model not properly bound to tools  
**Fix**: Use `.bind_tools()`:

```python
model_with_tools = model.bind_tools(tools)
response = model_with_tools.invoke(messages)  # Works
```

### Issue: Infinite loops

**Cause**: Conditional edge always returns to same node  
**Fix**: Add iteration counter or state check:

```python
def should_continue(state) -> str:
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    return "process"
```

---

## Resources

- **Official Docs**: https://langchain-ai.github.io/langgraph/
- **GitHub**: https://github.com/langchain-ai/langgraph
- **Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Discord Community**: LangChain Discord

---

## Performance Tips

1. **Use async when possible**: `ainvoke()` and `astream()` for I/O-bound tasks
2. **Batch processing**: `graph.batch()` for multiple inputs
3. **Streaming**: Use `stream_mode="updates"` to reduce data transfer
4. **Checkpointer selection**: PostgreSQL > SQLite > In-Memory based on scale
5. **Cache expensive operations**: Store results in long-term Store
6. **Limit iterations**: Always set `MAX_ITERATIONS` to prevent runaway loops

---

## Next Steps

1. Start with simple linear graphs
2. Add conditional routing
3. Build multi-agent systems
4. Integrate tools
5. Add persistence with checkpointers
6. Deploy with CLI/Docker
7. Monitor with LangSmith

Good luck with your AI engineering journey! LangGraph gives you the low-level control to build sophisticated agent systems. Start small, iterate, and scale.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.2.7 | June 29, 2026 | Patch release. Version confirmed against installed `langgraph==1.2.7`, `langgraph-checkpoint==4.1.1`, `langgraph-prebuilt==1.1.0`. New deep-dive Vol. 29 covers 10 previously undocumented APIs: `Edge`/`TriggerEdge`/`draw_graph()` visualization internals, `get_graph(xray=)` subgraph expansion, `as_tool()` (beta) graph-to-tool conversion, `get_subgraphs(recurse=True)` namespace traversal, `get_input_jsonschema()`/`get_output_jsonschema()`/`get_context_jsonschema()` schema introspection, `clear_cache()`/`aclear_cache()` per-namespace cache invalidation, `_messages_delta_reducer` batch-invariant DeltaChannel reducer, `_add_messages_wrapper` partial-application decorator, `add_node(defer=True)` deferred end-of-run execution (quiescence), and `REMOVE_ALL_MESSAGES` bulk history reset. |
| 1.2.0 | May 12, 2026 | Minor release. Version confirmed against installed `langgraph==1.2.0` (`.routine-envs/check-0512-py`); `langgraph-checkpoint==4.1.0`, `langgraph-prebuilt==1.1.0`. New exports: `ToolRuntime`, `ToolCallTransformer` (both in `langgraph.prebuilt`). All core symbols (`StateGraph`, `END`, `START`, `CompiledStateGraph`, `MemorySaver`, `create_react_agent`, `ToolNode`, `StreamPart`, `Command`, `Send`, `Interrupt`, `interrupt`, `entrypoint`, `task`, `InMemoryStore`) verified with `-W error::DeprecationWarning`. |
| 1.1.10 | April 28, 2026 | Patch release. Version confirmed against installed `langgraph==1.1.10` (`.routine-envs/main-py-0428`); `langgraph-checkpoint==4.0.3`. All core symbols verified. |
| 1.1.9 | April 22, 2026 | Patch release; six source-verified reference pages added to the guide. |
| 1.1.8 | April 17, 2026 | Fixed strict `add_handler` type check that broke OpenTelemetry instrumentation; follows patch 1.1.7 (same day) |
| 1.1.7 | April 17, 2026 | Intermediate patch preceding 1.1.8; stability fixes |
| 1.1.6 | April 10, 2026 | Type-safe v2 streaming and invoke API (`version="v2"`); Pydantic/dataclass auto-coercion; Python 3.14 support; time-travel bug fixes with interrupts and subgraphs |
| 1.0.3 | November 2025 | Previous documented version |

---

## Class & API Reference

Source-verified reference for the classes, functions, and types developers actually
touch — consolidated from LangGraph's full class-by-class audit and re-verified
against the installed `langgraph==1.2.11` (`langgraph-checkpoint==4.2.0`,
`langgraph-prebuilt==1.1.0`, `langchain-core==1.6.0`). Each entry gives the module
path, the verified signature, why it matters, and a runnable example. Deeply
private, underscore-prefixed implementation details (`langgraph.pregel._algo`,
`langgraph._internal.*`, and similar) are intentionally **not** given individual
entries — they carry no compatibility guarantee and change between patch releases;
each section that has notable internals closes with a short, named pointer instead
of a full write-up.

Several symbols below are deprecated but still importable and functional in
1.2.11 — each carries an explicit **migrate to** note rather than being silently
dropped: `create_react_agent`, `AgentState`/`AgentStatePydantic`/
`AgentStateWithStructuredResponse`, `ValidationNode`, `MessageGraph`, and the
`HumanInterrupt`/`HumanInterruptConfig`/`ActionRequest` family all moved to
`langchain.agents` (or `langchain.agents.interrupt`) as of LangGraph 1.0, and each
still works with a `LangGraphDeprecatedSinceV10` warning.

### Graph Construction & State

#### `StateGraph`

**Module:** `langgraph.graph.state` (re-exported from `langgraph.graph`)

The declarative builder for a stateful graph. Declare a state schema (`TypedDict`,
dataclass, or Pydantic `BaseModel`), add nodes and edges, then `.compile()` to get a
runnable. Each state field maps to a channel: a plain field → `LastValue`; `Annotated[T,
reducer]` → `BinaryOperatorAggregate`; `Annotated[list[T], Topic(T)]` → `Topic`.

```python
StateGraph(
    state_schema: type[StateT],
    context_schema: type[ContextT] | None = None,
    *,
    input_schema: type[InputT] | None = None,
    output_schema: type[OutputT] | None = None,
)

.add_node(
    node: str | Callable, action: Callable | None = None, *,
    defer: bool = False, metadata: dict | None = None,
    input_schema: type | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    error_handler: Callable | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
    trace_policy: TracePolicy | None = None,
) -> Self

.add_edge(start_key: str | list[str], end_key: str) -> Self
.add_conditional_edges(source: str, path: Callable | Runnable,
                        path_map: dict | list[str] | None = None) -> Self
.compile(checkpointer=None, *, cache=None, store=None, interrupt_before=None,
         interrupt_after=None, debug=False, name=None,
         transformers=None) -> CompiledStateGraph
```

Key facts, cross-verified against source:
- `defer=True` schedules the node to run only once **all non-deferred work in the
  entire run has drained** — not merely the current super-step. It still needs an
  explicit incoming edge; with no edge it never runs.
- `destinations=` is a **visualization hint only** — it documents possible `Command`
  targets for `get_graph()`/`draw_mermaid()`; it has no effect on runtime routing.
- `error_handler` is added to the graph as its own node (`is_error_handler=True`
  internally); it cannot itself declare another `error_handler`.
- `set_node_defaults()`'s `retry_policy`/`timeout` apply to every node including
  error-handler nodes; its `cache_policy`/`error_handler` defaults apply to regular
  nodes only. A per-node value always wins over the graph-wide default. Not
  inherited by subgraphs.
- Nodes can declare extra keyword-only parameters — `config: RunnableConfig`,
  `store: BaseStore`, `writer: StreamWriter`, `runtime: Runtime[ContextT]` — and
  LangGraph auto-injects them by inspecting the signature at `add_node()` time; no
  manual wiring needed.
- `trace_policy` (see `TracePolicy` under Observability & Tracing) and
  `destinations` are commonly-missed kwargs.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache

class PipelineState(TypedDict):
    query: str
    results: Annotated[list[str], operator.add]
    cost: Annotated[float, operator.add]

class SearchInput(TypedDict):
    query: str  # node only sees this key

def web_search(state: SearchInput) -> dict:
    return {"results": [f"web:{state['query']}"], "cost": 0.001}

builder = StateGraph(PipelineState)
builder.add_node(
    "web_search", web_search, input_schema=SearchInput,
    retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.2, backoff_factor=2.0),
    cache_policy=CachePolicy(ttl=300),
)
builder.add_edge(START, "web_search")
builder.add_edge("web_search", END)
graph = builder.compile(checkpointer=InMemorySaver(), cache=InMemoryCache())
result = graph.invoke({"query": "langgraph docs", "results": [], "cost": 0.0})
```

#### State schemas — `TypedDict`, Pydantic `BaseModel`, `dataclass`

**Module:** `langgraph.graph.state`

`StateGraph` accepts any of three schema styles. A Pydantic `BaseModel` coerces dict
input through the constructor (running validators); LangGraph tracks
`model_fields_set` and writes back only the fields a node explicitly touched — a
node that returns `MyState(field_a="x")` leaves `field_b` untouched even though the
model has a default for it. Dataclasses behave the same way for partial updates.

```python
from pydantic import BaseModel, field_validator
from langgraph.graph import StateGraph, START, END

class AgentState(BaseModel):
    messages: list[str] = []
    count: int = 0

    @field_validator("count")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("count must be >= 0")
        return v

def increment(state: AgentState) -> AgentState:
    return AgentState(count=state.count + 1)   # messages untouched, stays []

graph = (
    StateGraph(AgentState)
    .add_node("increment", increment)
    .add_edge(START, "increment")
    .add_edge("increment", END)
    .compile()
)
print(graph.invoke({"count": "5"}))  # {'messages': [], 'count': 6} — str coerced to int
```

#### `CompiledStateGraph`

**Module:** `langgraph.graph.state`

The object `StateGraph.compile()` returns — a thin subclass of `Pregel` implementing
the full LangChain `Runnable` protocol plus state-management methods: `get_state`,
`update_state`, `bulk_update_state`, `get_state_history`, typed `stream`/`invoke`
overloads, `get_input_jsonschema()` / `get_output_jsonschema()` (used for API validation
at the graph's declared `input_schema`/`output_schema` boundary).

```python
class CompiledStateGraph(Pregel[StateT, ContextT, InputT, OutputT]):
    builder: StateGraph
    def get_input_jsonschema(self, config=None) -> dict: ...
    def get_output_jsonschema(self, config=None) -> dict: ...
```

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class ChatState(TypedDict):
    messages: list

def chat_node(state: ChatState) -> dict:
    return {"messages": state["messages"] + ["reply"]}

graph = StateGraph(ChatState).add_node("chat", chat_node).add_edge(START, "chat").add_edge("chat", END).compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "demo"}}
result = graph.invoke({"messages": ["hi"]}, cfg)
for chunk in graph.stream({"messages": ["hello"]}, cfg, stream_mode="updates"):
    print(chunk)
```

#### `add_sequence()`

**Module:** `langgraph.graph.state`

Shortcut for a linear pipeline: calls `add_node` for each item and wires
`add_edge(prev, next)` between consecutive nodes. Does **not** add `START`/`END`
edges — add those yourself. Items may be bare callables (name from `__name__`) or
`(name, callable)` tuples. Raises `ValueError` on an empty sequence or duplicate names.

```python
def add_sequence(
    self, nodes: Sequence[Callable | tuple[str, Callable]],
) -> Self
```

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    value: int

builder = StateGraph(S)
builder.add_sequence([
    ("double", lambda s: {"value": s["value"] * 2}),
    ("add_ten", lambda s: {"value": s["value"] + 10}),
])
builder.add_edge(START, "double")
builder.add_edge("add_ten", END)
graph = builder.compile()
print(graph.invoke({"value": 3})["value"])  # 16
```

#### `set_node_defaults()`

**Module:** `langgraph.graph.state`

Sets graph-wide fallback `retry_policy` / `cache_policy` / `error_handler` / `timeout`
applied to every node that doesn't specify its own (per-node values always win).
`retry_policy` and `timeout` apply even to error-handler nodes; `cache_policy` and
`error_handler` do not (caching a handler's output, or letting a handler catch itself,
is unsafe). Applied at `compile()` time; **not** inherited by subgraphs.

```python
.set_node_defaults(
    *, retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    error_handler: Callable | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
) -> Self
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

graph = (
    StateGraph(State)
    .set_node_defaults(retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.01))
    .add_node("a", lambda s: {"value": s["value"] + 1})
    .add_node("b", lambda s: {"value": s["value"] * 2}, retry_policy=RetryPolicy(max_attempts=2))
    .add_edge(START, "a").add_edge("a", "b").add_edge("b", END)
    .compile()
)
```

#### `input_schema` / `output_schema` — narrowing a graph or a node

**Module:** `langgraph.graph.state`

`StateGraph(state_schema, input_schema=..., output_schema=...)` narrows the public
input/output contract of the whole graph (useful when it's used as a subgraph or an
API boundary). `add_node(..., input_schema=...)` narrows just the slice of state one
node receives — only the declared keys are passed in.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class FullState(TypedDict):
    user_id: str; message: str; internal_counter: int; result: str

class LLMInput(TypedDict):
    user_id: str; message: str

def llm_node(state: LLMInput) -> dict:
    return {"result": f"[{state['user_id']}] {state['message']}"}

graph = (
    StateGraph(FullState)
    .add_node("llm", llm_node, input_schema=LLMInput)
    .add_edge(START, "llm").add_edge("llm", END)
    .compile()
)
```

#### `context_schema` + `Runtime.context`

**Modules:** `langgraph.graph.state`, `langgraph.runtime`

Declares a typed **read-only**, per-invocation context object injected into every node
via `runtime.context`. Unlike state it is never persisted to checkpoints, never
returned by `get_state()`, and never writable by nodes — use it for `user_id`, tenant
config, feature flags, or an authenticated principal.

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

@dataclass
class AppContext:
    user_id: str
    locale: str = "en"

class State(TypedDict):
    result: str

def personalise(state: State, runtime: Runtime[AppContext]) -> dict:
    return {"result": f"[{runtime.context.locale}] hi {runtime.context.user_id}"}

graph = StateGraph(State, context_schema=AppContext).add_node("p", personalise).add_edge(START, "p").add_edge("p", END).compile()
result = graph.invoke({"result": ""}, context=AppContext(user_id="alice", locale="en-GB"))
```

#### `add_messages` + `MessagesState` + `REMOVE_ALL_MESSAGES` + `push_message()`

**Module:** `langgraph.graph.message` (`MessagesState` also re-exported from `langgraph.graph`)

`add_messages` is the standard reducer for a `messages` list: appends new messages,
**upserts in place** when an incoming message shares an existing message's `id`
(the basis for "edit in place"), auto-assigns a UUID to ID-less messages, and
deletes via `RemoveMessage(id=...)`. The sentinel `REMOVE_ALL_MESSAGES`
(`"__remove_all__"`) clears the entire list in one write when included in the
update list — everything in existing state, and everything in the incoming batch
before the sentinel, is discarded; messages after the sentinel become the new
list. `format="langchain-openai"` normalises mixed dict/tuple/`BaseMessage` input
into OpenAI-compatible content blocks via `convert_to_openai_messages` (needs
`langchain-core>=0.3.11`). `MessagesState` is the one-field `TypedDict`
(`messages: Annotated[list[AnyMessage], add_messages]`) that ships as the
canonical chatbot state — subclass it to add fields.

`push_message()` emits a single message **immediately to the `"messages"` stream**
during node execution, before the node returns — and (unless `state_key=None`)
also writes it into the named state channel via the `add_messages`-compatible
reducer there. Useful for streaming intermediate progress without waiting for the
node to finish; must be called from inside a running node (reads the active
config via a ContextVar).

```python
def add_messages(left: Messages, right: Messages, *, format: Literal["langchain-openai"] | None = None) -> Messages
# add_messages() / add_messages(format=...) with no left/right returns a functools.partial,
# directly usable as an Annotated reducer: Annotated[list, add_messages(format="langchain-openai")]
REMOVE_ALL_MESSAGES: str = "__remove_all__"
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def push_message(message: MessageLikeRepresentation | BaseMessageChunk, *, state_key: str | None = "messages") -> AnyMessage
```

A separate, **experimental** batch-safe reducer, `_messages_delta_reducer`, exists
for pairing `add_messages`-style semantics with `DeltaChannel` — it is
batching-invariant but does *not* implement `REMOVE_ALL_MESSAGES` or auto-ID
assignment, unlike `add_messages`.

```python
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES

msgs = add_messages([HumanMessage("hi", id="1")], [AIMessage("hello", id="2")])
msgs = add_messages(msgs, [AIMessage("hello again", id="2")])       # in-place replace by id
msgs = add_messages(msgs, [RemoveMessage(id="1")])                   # delete by id
msgs = add_messages(msgs, [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage("fresh start")])
assert len(msgs) == 1
```

```python
from langchain_core.messages import AIMessageChunk
from langgraph.graph.message import push_message, MessagesState
from langgraph.graph import StateGraph, START, END

def long_running(state: MessagesState) -> dict:
    push_message(AIMessageChunk(content="Starting… ", id="progress-1"))
    return {"messages": [AIMessageChunk(content="Done.", id="final-1")]}

graph = StateGraph(MessagesState).add_node("work", long_running).add_edge(START, "work").add_edge("work", END).compile()
for chunk in graph.stream({"messages": [("user", "go")]}, stream_mode="messages"):
    msg, meta = chunk
```

#### `MessageGraph` (deprecated)

**Module:** `langgraph.graph.message`

The pre-1.0 graph type whose entire state was a bare `list[AnyMessage]`
(`Annotated[list[AnyMessage], add_messages]`). **Fully deprecated**, emits
`LangGraphDeprecatedSinceV10`. Migrate to `StateGraph(MessagesState)` —
behaviourally identical, just wraps the list in a `"messages"` key.

```python
class MessageGraph(StateGraph):
    def __init__(self) -> None:
        super().__init__(Annotated[list[AnyMessage], add_messages])
```

```python
# Migration: MessageGraph() -> StateGraph(MessagesState)
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage

def chatbot(state: MessagesState) -> dict:
    return {"messages": [AIMessage(content=f"Echo: {state['messages'][-1].content}")]}

graph = StateGraph(MessagesState).add_node("chatbot", chatbot).add_edge(START, "chatbot").add_edge("chatbot", END).compile()
```

#### `TAG_NOSTREAM`, `TAG_HIDDEN`

**Module:** `langgraph.constants`. Two `sys.intern()`d tag strings that affect
streaming/tracing visibility.

- `TAG_NOSTREAM` (`"nostream"`) — attach via `.with_config({"tags": ["nostream"]})` on
  a chat-model call inside a node to suppress its token stream from
  `stream_mode="messages"`; the model still runs and its final output is used
  normally.
- `TAG_HIDDEN` (`"langsmith:hidden"`) — suppresses a node from
  `stream_mode="debug"`/LangSmith traces. LangGraph applies this itself to
  internal book-keeping nodes (e.g. the `__start__` input-projection node);
  passing `tags=[TAG_HIDDEN]` to your own `add_node()` call has no effect in this
  version — there is no user-facing storage path for arbitrary node tags in
  `StateNodeSpec`.

```python
from langgraph.constants import TAG_HIDDEN
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

g = StateGraph(State)
g.add_node("compute", lambda s: {"value": s["value"] * 2})
g.add_edge(START, "compute"); g.add_edge("compute", END)
graph = g.compile()

hidden = [name for name, node in graph.nodes.items() if TAG_HIDDEN in (node.tags or [])]
print(hidden)  # ['__start__']
```

#### Graph visualization — `get_graph`, `draw_mermaid`, `xray`, `get_subgraphs`, schema introspection

**Module:** `langgraph.graph.state` / `langgraph.pregel.main`

`get_graph()` returns a `langchain_core.runnables.graph.Graph` — built by a
**dry-run simulation** of the Pregel loop from an empty checkpoint (no node
functions actually execute); it inspects the statically-declared `ChannelWrite`
targets to discover edges, so it works even for graphs whose conditional edges
depend on data.

```python
def get_graph(self, config=None, *, xray: int | bool = False) -> Graph: ...
def get_subgraphs(self, *, namespace: str | None = None, recurse: bool = False) -> Iterator[tuple[str, PregelProtocol]]: ...
```

- `xray=True` recursively expands **every** nested subgraph to its leaves;
  `xray=1`/`xray=2`/... expands only that many levels. Subgraph-internal node ids
  are prefixed `parent_node:child_node`.
- `graph.draw_mermaid()` returns Mermaid markdown; `graph.draw_mermaid_png()`
  defaults to calling the hosted Mermaid.ink API (`MermaidDrawMethod.API`) — pass
  `draw_method=MermaidDrawMethod.PYPPETEER` for a fully offline render.
- `get_subgraphs(namespace=None)` yields `(name, subgraph)` for every node whose
  compiled graph is embedded directly as a node value; `recurse=True` descends and
  prefixes namespaces (`"outer:inner"`). Nodes that merely *call* a compiled
  subgraph's `.invoke()` from inside a wrapper function are **not** discovered this
  way — only subgraphs wired in directly via `add_node("name", compiled_subgraph)`.
- `get_input_jsonschema()` / `get_output_jsonschema()` return `{}` when the schema
  is untyped; `get_context_jsonschema()` returns `None` (not `{}`) when no
  `context_schema` was passed to `StateGraph`. The older `get_config_jsonschema()`
  is deprecated since v1.0 in favor of `get_context_jsonschema()`.
- `Pregel.as_tool()` (beta) wraps any compiled graph as a `BaseTool` — see the
  Tools & Tool Calling section.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    x: int

sub = StateGraph(State)
sub.add_node("op", lambda s: {"x": s["x"] * 2})
sub.set_entry_point("op"); sub.set_finish_point("op")
compiled_sub = sub.compile()

main = StateGraph(State)
main.add_node("subgraph", compiled_sub)
main.add_edge(START, "subgraph"); main.add_edge("subgraph", END)
compiled_main = main.compile()

print(sorted(compiled_main.get_graph().nodes))          # shallow: ['__end__', '__start__', 'subgraph']
print(sorted(compiled_main.get_graph(xray=True).nodes))  # expanded: includes 'subgraph:op'
```

#### `BranchSpec` — conditional-edge internals

**Modules:** `langgraph.graph._branch`

`add_conditional_edges()` compiles into a `BranchSpec` NamedTuple stored on
`builder.branches[source_node][name]`.

```python
class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]
    ends: dict[Hashable, str] | None
    input_schema: type | None = None

    @classmethod
    def from_path(cls, path, path_map, infer_schema: bool = False) -> "BranchSpec": ...
```

- `ends=None` means open routing — the function's return value must itself be a
  node name (or a `Send` object).
- With `path_map=None` and a `Literal[...]` return annotation on the routing
  function, `ends` is auto-inferred from the `Literal` members.
- `add_conditional_edges` calls `from_path(..., infer_schema=True)`, so
  `input_schema` is populated from the router's parameter type when it's a
  `TypedDict`/dataclass; calling `from_path` directly defaults `infer_schema=False`.

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    score: int

def route(state: State) -> Literal["high", "low", "__end__"]:
    if state["score"] > 50: return "high"
    if state["score"] > 10: return "low"
    return "__end__"

builder = StateGraph(State)
builder.add_node("high", lambda s: {"score": s["score"] + 100})
builder.add_node("low", lambda s: {"score": s["score"] - 5})
builder.add_edge(START, "high")
builder.add_conditional_edges("high", route)

spec = builder.branches["high"]["route"]
print(spec.ends)  # {'high': 'high', 'low': 'low', '__end__': '__end__'}
```

#### Channels — `LastValue`, `BinaryOperatorAggregate` + `Overwrite`, `Topic`, `EphemeralValue`, `NamedBarrierValue` (+ `AfterFinish`), `AnyValue`, `UntrackedValue`, `DeltaChannel`

**Modules:** `langgraph.channels.*`; abstract base `langgraph.channels.base.BaseChannel`

Every state field maps to a channel type that governs what happens when zero, one, or
multiple parallel nodes write to it in the same super-step:

| Channel | Concurrent writes | Not written this step | Persists across steps |
|---|---|---|---|
| `LastValue` (default, unannotated field) | raises `InvalidUpdateError` | retains previous | yes |
| `LastValueAfterFinish` | same as `LastValue` | value unreadable until `finish()`; `consume()` resets value + finished flag | yes, as `(value, finished_bool)` |
| `BinaryOperatorAggregate` (`Annotated[T, reducer]`) | folded through `operator(a, b)` in arrival order (first write bootstraps the accumulator) | unchanged | yes |
| `Topic(T, accumulate=False)` | collects all → list (scalar or list writes both flatten) | resets to `[]` | no |
| `Topic(T, accumulate=True)` | collects all → list | unchanged | yes |
| `AnyValue` | last wins, **no** guard (assumes all equal) | clears to `MISSING` | no |
| `EphemeralValue(guard=True)` | raises `InvalidUpdateError` | clears | no |
| `EphemeralValue(guard=False)` | last wins | clears | no |
| `NamedBarrierValue(names={...})` | adds to seen-set; `is_available()` opens only when all seen (`get()` returns `None` — the signal is availability, not a value) | no value exposed until opened | resets (`seen=set()`) after `consume()` |
| `UntrackedValue` | `guard` controls same as `EphemeralValue` | never checkpointed (`checkpoint()` → `MISSING`) | no (always empty on resume) |
| `DeltaChannel` (beta) | reducer-folded, but only a sentinel is stored in checkpoints; full value rebuilt by replaying ancestor writes | n/a | yes (bounded by `snapshot_frequency`) |

`Overwrite(value)` (from `langgraph.types`) bypasses a `BinaryOperatorAggregate`'s
reducer and replaces the channel outright — at most one `Overwrite` per channel per
step (a second raises `InvalidUpdateError`); mixing one `Overwrite` with normal writes
is fine. It's detected in three forms: an `Overwrite` instance, a `{"__overwrite__":
v}` dict, or the JSON round-trip shape `{"value": v, "type": "__overwrite__"}`.

```python
class BinaryOperatorAggregate(BaseChannel[Value, Value, Value]):
    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]): ...
class Topic(BaseChannel[Sequence[Value], Value | list[Value], list[Value]]):
    def __init__(self, typ: type[Value], accumulate: bool = False) -> None: ...
class EphemeralValue(BaseChannel[Value, Value, Value]):
    def __init__(self, typ: Any, guard: bool = True) -> None: ...
class NamedBarrierValue(BaseChannel[Value, Value, set[Value]]):
    def __init__(self, typ: type[Value], names: set[Value]) -> None: ...
class UntrackedValue(BaseChannel[Value, Value, Value]):
    def __init__(self, typ: type[Value], guard: bool = True) -> None: ...
class DeltaChannel(BaseChannel):
    def __init__(self, reducer: Callable[[Any, Sequence[Any]], Any], typ: type | None = None,
                 *, snapshot_frequency: int = 1000) -> None: ...
class Overwrite:
    value: Any
```

`DeltaChannel`'s reducer receives the **whole batch** of writes for the step, not
one value at a time, and must be batching-invariant:
`reducer(reducer(s, xs), ys) == reducer(s, xs + ys)`. A `_DeltaSnapshot` blob is
written every `snapshot_frequency` updates or every
`DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` (env `LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT`,
default 5000) supersteps, bounding replay depth.

```python
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels import Topic
from langgraph.types import Overwrite

class State(TypedDict):
    total: Annotated[int, operator.add]                 # BinaryOperatorAggregate
    events: Annotated[Sequence[str], Topic(str)]         # cleared each step

def accumulate(state: State) -> dict:
    return {"total": 10, "events": "ran"}

def reset(state: State) -> dict:
    return {"total": Overwrite(0)}   # bypasses operator.add for this write

builder = StateGraph(State)
builder.add_node("acc", accumulate); builder.add_node("reset", reset)
builder.add_edge(START, "acc"); builder.add_edge("acc", "reset"); builder.add_edge("reset", END)
result = builder.compile().invoke({"total": 5, "events": []})
print(result["total"])  # 0 — Overwrite replaced the accumulated value
```

`NamedBarrierValueAfterFinish` (same module) adds a second gate on top of
`NamedBarrierValue`: the channel only opens once every name is seen **and** an explicit
`finish()` has been called — a "collect, then release" pattern used internally for
subgraph coordination; direct use is rare.

Writing a **custom channel**: subclass `BaseChannel[Value, Update, Checkpoint]` and
implement `ValueType`, `UpdateType`, `get()`, `update()`, `from_checkpoint()`;
override `checkpoint()` (default calls `get()`), `consume()` (return `True` to
self-clear after each read, as `EphemeralValue` does), and `is_available()` as needed.

#### Node representation internals — `StateNodeSpec`, `PregelNode`

**Modules:** `langgraph.graph._node`, `langgraph.pregel._read`

`add_node()` builds a `StateNodeSpec` (the declarative record: runnable, input_schema,
retry/cache/timeout policies, `is_error_handler`/`error_handler_node`, `defer`) and
stores it on `builder.nodes[name]` pre-compile. `compile()` turns each spec into a
`PregelNode` (channels subscribed, triggers, writers, bound runnable) stored on
`graph.nodes[name]`. Useful for introspecting a graph before or after compilation,
e.g. for custom linting or tracing tools.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

builder = StateGraph(State)
builder.add_node("n", lambda s: {"value": s["value"] + 1}, retry_policy=RetryPolicy(max_attempts=3))
spec = builder.nodes["n"]                 # StateNodeSpec, pre-compile
print(spec.retry_policy, spec.is_error_handler)
builder.add_edge(START, "n"); builder.add_edge("n", END)
graph = builder.compile()
pregel_node = graph.nodes["n"]            # PregelNode, post-compile
print(pregel_node.triggers, pregel_node.retry_policy)
```

#### `Pregel` + `NodeBuilder`

**Module:** `langgraph.pregel.main` (re-exported from `langgraph.pregel`)

`Pregel` is the runtime engine underlying every compiled graph — `CompiledStateGraph`
is a thin subclass. You rarely instantiate it directly; its constructor exposes the
full set of runtime knobs (`stream_eager`, `step_timeout`, `trigger_to_nodes`, etc.)
that `StateGraph.compile()` doesn't surface. `NodeBuilder` is the low-level fluent API
`add_node()`/`add_edge()` compile down to — useful for wiring a `Pregel` graph by hand
without `StateGraph`.

```python
Pregel(*, nodes: dict[str, PregelNode | NodeBuilder], channels: dict[str, BaseChannel | ManagedValueSpec] | None,
       output_channels: str | Sequence[str], input_channels: str | Sequence[str],
       checkpointer=None, store=None, cache=None, context_schema=None,
       interrupt_before_nodes=(), interrupt_after_nodes=(), name: str = "LangGraph", ...)

# NodeBuilder (verified 1.2.11 — note the real method names differ from some older docs)
nb = NodeBuilder()
nb.subscribe_only("channel")            # or .subscribe_to("ch1", "ch2", read=True)
nb.do(my_function)                      # set the node action
nb.write_to("output_channel")           # declare writes
nb.add_retry_policies(RetryPolicy(...)) # NOT with_retry_policy()
nb.add_cache_policy(CachePolicy(...))   # NOT with_cache_policy()
nb.set_timeout(30.0)                    # NOT with_timeout()
nb.meta("tag1", env="prod")             # tags + metadata combined; NOT with_tags()/with_metadata()
pregel_node = nb.build()                # -> PregelNode
```

```python
from langgraph.pregel.main import Pregel, NodeBuilder
from langgraph.channels.last_value import LastValue

node_double = NodeBuilder().subscribe_only("input").do(lambda x: {"result": x * 2}).write_to("result")
node_format = NodeBuilder().subscribe_only("result").do(lambda r: {"output": f"Result is {r}"}).write_to("output")
graph = Pregel(
    nodes={"double": node_double, "format": node_format},
    channels={"input": LastValue(int), "result": LastValue(int), "output": LastValue(str)},
    input_channels="input", output_channels="output",
)
print(graph.invoke(5))  # 'Result is 10'
```

**Internals (private, unverified-stability):** `apply_writes` / `prepare_next_tasks` /
`should_interrupt` / `validate_graph` / `validate_keys` (`langgraph.pregel._algo`,
`langgraph.pregel._validate`) implement compile-time validation and the core superstep
algorithm; `read_channel`/`read_channels`/`map_input`/`map_command` (`langgraph.pregel._io`)
form the I/O layer between the loop and channel state; `ChannelWrite`/`ChannelWriteEntry`/
`ChannelRead` (`langgraph.pregel._write`, `._read`) are the write/read runnables every
compiled node is wrapped in; `RunnableCallable`/`RunnableSeq` (`langgraph._internal._runnable`)
are LangGraph's own lightweight `Runnable` wrapper/pipeline (every node function passed
to `add_node` is wrapped in one); `DataclassLike`/`TypedDictLikeV1`/`TypedDictLikeV2`
(`langgraph._internal._typing`) are the structural protocols used to detect which kind
of schema a state class is; `get_field_default`/`get_cached_annotated_keys`
(`langgraph._internal._fields`) resolve default values and MRO-ordered field lists for
TypedDict/dataclass/Pydantic schemas; `Edge`/`TriggerEdge` (`langgraph.pregel._draw`) are
the `NamedTuple`s behind `get_graph().edges`, `draw_mermaid()`, `draw_ascii()`. All of
these live in underscore-prefixed ("private") modules as of the 1.x line and carry no
compatibility guarantee — treat them as debugging aids, not stable API.
### Checkpointing & Persistence

#### `InMemorySaver` (`BaseCheckpointSaver`)

**Module:** `langgraph.checkpoint.memory`. **Alias:** `MemorySaver`.

The in-process checkpoint backend. Stores, per thread/namespace/checkpoint id, a
`(checkpoint_bytes, metadata_bytes, parent_id)` tuple in `storage`, plus pending
task writes in `writes`, plus deduplicated channel values by version in `blobs`
(a channel whose value hasn't changed across checkpoints is stored once, not once
per checkpoint). `MemorySaver` additionally accepts a `filename=` for a
`PersistentDict`-backed on-disk store that survives process restarts (see below)
— plain `InMemorySaver()` is purely in-memory. `saver.list(config, filter=...,
limit=...)` and `saver.delete_thread(thread_id)` are the common inspection/cleanup
operations.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    x: int

saver = InMemorySaver()
graph = StateGraph(S).add_node("inc", lambda s: {"x": s["x"] + 1}).add_edge(START, "inc").add_edge("inc", END).compile(checkpointer=saver)
cfg = {"configurable": {"thread_id": "t1"}}
for _ in range(3):
    graph.invoke({"x": 1}, cfg)
for tup in saver.list(cfg):
    print(tup.metadata.get("step"), tup.checkpoint["id"][:8])
saver.delete_thread("t1")
```

#### `BaseCheckpointSaver` — building a custom backend

**Module:** `langgraph.checkpoint.base`

The abstract base every checkpoint backend implements. `V` is the version type
(`int`/`float`/`str`) used to order channel writes. Four methods make a working
backend: `get_tuple`, `list`, `put`, `put_writes` (plus async twins for production
use). Override `get_next_version(current, channel)` if your backend orders versions
differently from the default integer increment.

```python
class BaseCheckpointSaver(Generic[V]):
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None: ...
    def list(self, config, *, filter=None, before=None, limit=None) -> Iterator[CheckpointTuple]: ...
    def put(self, config, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig: ...
    def put_writes(self, config, writes: list[tuple[str, Any]], task_id: str, task_path: str = "") -> None: ...
```

```python
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, get_checkpoint_id

class DictSaver(BaseCheckpointSaver[int]):
    def __init__(self):
        super().__init__()
        self._store: dict[tuple, tuple] = {}

    def get_tuple(self, config):
        tid = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = get_checkpoint_id(config)
        entry = self._store.get((tid, ns, cid)) if cid else None
        if entry is None:
            return None
        cp, meta = entry
        return CheckpointTuple(config=config, checkpoint=cp, metadata=meta)

    def put(self, config, checkpoint, metadata, new_versions):
        c = config["configurable"]
        self._store[(c["thread_id"], c.get("checkpoint_ns", ""), checkpoint["id"])] = (checkpoint, metadata)
        return {"configurable": {**c, "checkpoint_id": checkpoint["id"]}}

    def list(self, config, *, filter=None, before=None, limit=None):
        return iter([])

    def put_writes(self, config, writes, task_id, task_path=""):
        pass
```

#### `get_checkpoint_id`, `get_checkpoint_metadata`

**Module:** `langgraph.checkpoint.base`.

```python
def get_checkpoint_id(config: RunnableConfig) -> str | None: ...
def get_checkpoint_metadata(config: RunnableConfig, *, metadata: CheckpointMetadata) -> CheckpointMetadata: ...
```

`get_checkpoint_id` reads `config["configurable"]["checkpoint_id"]`; it raises
`KeyError` if `"configurable"` itself is absent (call `ensure_config()` first, or
guard with `.get("configurable", {})`). `get_checkpoint_metadata` is the helper
custom checkpoint-saver authors call inside `put()` to merge the caller-supplied
metadata with `run_id` from the config before writing.

#### `CheckpointMetadata`, `CheckpointTuple`, `StateSnapshot`, `PregelTask` / `PregelExecutableTask`

**Modules:** `langgraph.types` (`CheckpointMetadata`, `StateSnapshot`, `PregelTask`,
`PregelExecutableTask`), `langgraph.checkpoint.base` (`CheckpointTuple`)

`CheckpointTuple` is what `BaseCheckpointSaver.get_tuple()`/`list()` return —
`(config, checkpoint, metadata, parent_config, pending_writes)`. `CheckpointMetadata`
rides alongside every checkpoint describing why it was created (`source`: `"input"` |
`"loop"` | `"update"` | `"fork"`; `step` — `-1` for the initial "input" checkpoint,
then `0, 1, 2, ...`; `parents`; `run_id`). `StateSnapshot` (returned by
`graph.get_state()` / yielded by `get_state_history()`) is the user-facing view:
`values`, `next` (nodes scheduled next; empty tuple means terminal), `config` (pass
back to `invoke()`/`update_state()` to fork here), `metadata`, `created_at`,
`parent_config` (walk this to traverse history without calling
`get_state_history()` again), `tasks` (`tuple[PregelTask, ...]`), `interrupts`.
`PregelTask` describes one node execution: `id`, `name`, `path`, `error` (set if
that task's last run raised), `interrupts`, `state` (subgraph snapshot if
applicable), `result`. `PregelExecutableTask` is the heavier, live in-flight form —
adds `input`, `proc` (the runnable), `writes` (a `deque` accumulating channel
writes during execution), `retry_policy`, `cache_key`, `timeout`, `triggers`.

```python
class CheckpointMetadata(TypedDict, total=False):
    source: Literal["input", "loop", "update", "fork"]; step: int
    parents: dict[str, str]; run_id: str
class CheckpointTuple(NamedTuple):
    config: RunnableConfig; checkpoint: Checkpoint; metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None; pending_writes: list | None = None
class StateSnapshot(NamedTuple):
    values: dict | Any; next: tuple[str, ...]; config: RunnableConfig
    metadata: CheckpointMetadata | None; created_at: str | None
    parent_config: RunnableConfig | None; tasks: tuple[PregelTask, ...]; interrupts: tuple[Interrupt, ...]
class PregelTask(NamedTuple):
    id: str; name: str; path: tuple; error: Exception | None = None
    interrupts: tuple[Interrupt, ...] = (); state: Any = None; result: Any = None
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    step: int; notes: str

def step_node(state: State) -> dict:
    val = interrupt(f"Approve step {state['step']}?")
    return {"step": state["step"] + 1, "notes": val}

graph = StateGraph(State).add_node("step", step_node).add_edge(START, "step").add_edge("step", END).compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}
list(graph.stream({"step": 0, "notes": ""}, cfg))
snap = graph.get_state(cfg)
print(snap.values, snap.interrupts, snap.next)          # {'step': 0, ...}  (Interrupt(...),)  ('step',)
list(graph.stream(Command(resume="approved"), cfg))
for task in graph.get_state(cfg).tasks:
    print(task.name, task.error)

# Walk history via parent_config without calling get_state_history again
hist = []
s = graph.get_state(cfg)
while s is not None:
    hist.append(s.values)
    s = graph.get_state(s.parent_config) if s.parent_config else None
```

#### `get_state`, `get_state_history`, `update_state` / `bulk_update_state` (time-travel)

**Module:** `langgraph.graph.state` (methods on `CompiledStateGraph`); `StateUpdate`
lives in `langgraph.types`.

```python
def get_state(self, config, *, subgraphs: bool = False) -> StateSnapshot: ...
def get_state_history(self, config, *, filter=None, before=None, limit=None) -> Iterator[StateSnapshot]: ...
def update_state(self, config, values, as_node: str | None = None, task_id: str | None = None) -> RunnableConfig: ...
def bulk_update_state(self, config, supersteps: Sequence[Sequence[StateUpdate]]) -> RunnableConfig: ...

class StateUpdate(NamedTuple):
    values: dict[str, Any] | None; as_node: str | None = None; task_id: str | None = None
```

Injects state externally into a checkpointed thread — for seeding, time-travel
patching, or resuming after human review. `update_state()` is a thin wrapper:
`bulk_update_state(config, [[StateUpdate(values, as_node, task_id)]])`.
`bulk_update_state` accepts a list of super-steps, each a list of `StateUpdate`
objects; each inner list's writes are folded through the field reducers *together*,
and the outer sequence orders supersteps atomically — useful for seeding test
fixtures without running any node. `get_state_history` yields **newest-first**;
`before=` (a config pointing at a checkpoint) restricts to strictly older history;
`limit=` caps the count. Both `update_state`/`bulk_update_state` require a
checkpointer — calling them on a graph compiled without one raises.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import StateUpdate
from typing_extensions import TypedDict

class State(TypedDict):
    status: str; result: str; messages: list[str]; count: int

graph = StateGraph(State).add_node("process", lambda s: {"status": "processing"}).add_edge(START, "process").add_edge("process", END).compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}
graph.invoke({"status": "pending", "result": "", "messages": [], "count": 0}, cfg)
new_cfg = graph.update_state(cfg, {"result": "approved", "status": "done"}, as_node="process")
print(graph.get_state(new_cfg).values)

graph.bulk_update_state(cfg, [
    [StateUpdate(values={"messages": ["seed-1"], "count": 10}, as_node="process")],
    [StateUpdate(values={"messages": ["seed-2"], "count": 20}, as_node="process")],
])
```

#### `GraphOutput` (`version="v2"`)

**Module:** `langgraph.types`

The typed return of `invoke()`/`ainvoke()` when called with `version="v2"`: separates
the graph's final value (`.value`) from any pending interrupts (`.interrupts`) instead
of overloading the return dict with a magic `"__interrupt__"` key. Dict-style access
(`output["key"]`, `"key" in output`) still works but is deprecated
(`LangGraphDeprecatedSinceV11`, removal targeted for v3.0) — use `.value`/`.interrupts`.

```python
class GraphOutput(Generic[OutputT]):
    value: OutputT
    interrupts: tuple[Interrupt, ...] = ()
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command, GraphOutput
from typing_extensions import TypedDict

class S(TypedDict):
    data: str; approved: bool

def review(state: S) -> dict:
    decision = interrupt({"prompt": f"Approve {state['data']}?"})
    return {"approved": decision == "yes"}

graph = StateGraph(S).add_node("review", review).add_edge(START, "review").add_edge("review", END).compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}
out: GraphOutput = graph.invoke({"data": "deploy", "approved": False}, cfg, version="v2")
if out.interrupts:
    print(out.interrupts[0].id, out.interrupts[0].value)
final: GraphOutput = graph.invoke(Command(resume="yes"), cfg, version="v2")
print(final.value["approved"])   # True
```

#### `Durability`

**Module:** `langgraph.types`

A `Literal["sync", "async", "exit"]` type alias controlling **when** checkpoint
writes are flushed, accepted by `invoke`/`ainvoke`/`stream`/`astream` (a per-call
knob, not a `compile()` knob). `"sync"` persists before the next step starts
(safest, resumable from any step, zero loss on crash); `"async"` persists
concurrently with the next step (a crash mid-step can lose that step's
checkpoint); `"exit"` persists only when the run ends or is interrupted (fastest,
no mid-run recovery).

```python
Durability = Literal["sync", "async", "exit"]
```

```python
result = graph.invoke({"counter": 0}, config, durability="sync")   # safest
result = graph.invoke({"counter": 0}, config, durability="exit")   # fastest
```

#### `JsonPlusSerializer`

**Module:** `langgraph.checkpoint.serde.jsonplus`

The default serializer for every `BaseCheckpointSaver`. Despite the name, uses
ormsgpack (binary MessagePack) as the primary encoding, falling back to a legacy
JSON-plus format for types msgpack can't handle. `pickle_fallback=True` is a
security risk (arbitrary deserialization) — never enable in production. Set
`LANGGRAPH_STRICT_MSGPACK=true` to restrict deserialization to a built-in
safe-type allowlist (recommended for production) — an unknown type then raises
`InvalidModuleError` instead of silently deserializing; extend the allowlist via
`allowed_msgpack_modules=[("myapp.models", "UserProfile")]` or
`serde.with_msgpack_allowlist([MyClass, ...])`. The allowlist is built by
`langgraph._internal._serde.build_serde_allowlist`, which recursively traverses
your Pydantic/dataclass/`TypedDict`/Enum state schemas plus a curated set of
`BaseMessage` types.

```python
JsonPlusSerializer(*, pickle_fallback: bool = False,
                    allowed_json_modules: Iterable[tuple[str, ...]] | Literal[True] | None = None,
                    allowed_msgpack_modules: AllowedMsgpackModules | Literal[True] | None = ...)
```

```python
import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"   # set before importing langgraph
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serde = JsonPlusSerializer(allowed_msgpack_modules=[("myapp.models", "UserProfile")])
```

#### `EncryptedSerializer` + `SerializerProtocol` / `CipherProtocol` / `SerializerCompat`

**Modules:** `langgraph.checkpoint.serde.encrypted` (`EncryptedSerializer`),
`langgraph.checkpoint.serde.base` (protocols)

`EncryptedSerializer` wraps any `SerializerProtocol` (default `JsonPlusSerializer`)
with a `CipherProtocol`, encrypting checkpoint bytes at rest; the type string gains a
`+ciphername` suffix (e.g. `"msgpack+aes"`) so `loads_typed` can distinguish encrypted
from plain blobs — `loads_typed` splits on the first `+` and falls through to the
inner serde unchanged when it's absent, so pre-existing unencrypted checkpoints
still read correctly after you turn encryption on (supports incremental rollout).
`from_pycryptodome_aes()` is a ready-made AES-EAX (authenticated) cipher factory
reading a 16/24/32-byte key from `key=` or the `LANGGRAPH_AES_KEY` env var —
**there is no key-ID stored alongside the ciphertext**, so rotating the key means
old blobs become unreadable unless you keep the old key around long enough to
re-encrypt everything. `SerializerProtocol` (`@runtime_checkable`) is just
`dumps_typed(obj) -> (type_str, bytes)` / `loads_typed((type_str, bytes)) -> obj` —
no inheritance required. `SerializerCompat` wraps an old-style `dumps`/`loads`-only
serde, using `type(obj).__name__` as the type tag.

```python
class EncryptedSerializer(SerializerProtocol):
    def __init__(self, cipher: CipherProtocol, serde: SerializerProtocol = JsonPlusSerializer()): ...
    @classmethod
    def from_pycryptodome_aes(cls, serde=JsonPlusSerializer(), **kwargs) -> EncryptedSerializer: ...
class CipherProtocol(Protocol):
    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]: ...
    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes: ...
```

```python
import os
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

serde = EncryptedSerializer.from_pycryptodome_aes(key=os.urandom(32))  # pip install pycryptodome
saver = InMemorySaver(serde=serde)
```

#### `register_serde_event_listener` + `SerdeEvent`

**Module:** `langgraph.checkpoint.serde.event_hooks`

Observability hook into the serde subsystem: `emit_serde_event` fires whenever
`JsonPlusSerializer` (de)serialises a type outside the msgpack allowlist, or a
blocked type. Subscribe with `register_serde_event_listener(fn)` (returns an
unregister callable) to audit-log or build a dynamic allowlist. Listener
exceptions are caught and logged per-listener (never propagated — a broken
listener cannot break serialization); all listeners are global (process-wide),
guarded by a lock with a short critical section.

```python
class SerdeEvent(TypedDict):
    kind: str; module: str; name: str; method: NotRequired[str]
    # kind in {"msgpack_unregistered_allowed", "msgpack_blocked", "msgpack_method_blocked"}
def register_serde_event_listener(listener: Callable[[SerdeEvent], None]) -> Callable[[], None]: ...
```

```python
from langgraph.checkpoint.serde.event_hooks import register_serde_event_listener, SerdeEvent

audit_log: list[SerdeEvent] = []
unregister = register_serde_event_listener(audit_log.append)
# ... run graphs ...
unregister()
```

#### `PersistentDict`

**Module:** `langgraph.checkpoint.memory`. Internal to `MemorySaver(filename=...)`.

A `defaultdict` subclass that pickles itself to disk atomically (via a `.tmp` file),
powering `MemorySaver` when you pass a `filename` so checkpoints survive process
restarts without a real database — good for local dev, reproducible test fixtures, and
lightweight single-process scripts. `flag="c"` (default) creates-or-opens; `"r"` is
read-only (`sync()` a no-op); `"n"` always overwrites.

```python
class PersistentDict(defaultdict):
    def __init__(self, *args, filename: str, **kwds): ...
    def sync(self) -> None: ...   # atomic flush to disk
    def close(self) -> None: ...  # sync() + clear()
```

#### `ReplayState` — time-travel subgraph coordination

**Module:** `langgraph._internal._replay` (private; used internally by time-travel replay)

When you `invoke(None, config)` at a historical `checkpoint_id`, LangGraph must load
each *subgraph* from its checkpoint immediately **before** the replay point on first
visit, then fall back to normal latest-checkpoint loading on subsequent visits within
the same loop (so the graph can still make forward progress). A single `ReplayState`
instance, shared by reference for the whole replayed run, tracks which subgraph
namespaces have already been seeded — stripping the `:task_id` suffix so repeated
loop iterations of the same subgraph are recognised as "already visited."

```python
class ReplayState:
    def __init__(self, checkpoint_id: str) -> None: ...
    def _is_first_visit(self, checkpoint_ns: str) -> bool: ...   # strips ":task_id"
    def get_checkpoint(self, checkpoint_ns, checkpointer, checkpoint_config) -> CheckpointTuple | None: ...
```

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    n: int

graph = StateGraph(State).add_node("inc", lambda s: {"n": s["n"] + 1}).add_edge(START, "inc").add_edge("inc", END).compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}
for _ in range(3):
    graph.invoke({"n": 0}, cfg)
history = list(graph.get_state_history(cfg))
old_cfg = history[-1].config
graph.invoke(None, old_cfg)   # re-replay from the oldest checkpoint; branches history
```

#### `PostgresSaver` / `ShallowPostgresSaver` / `AsyncPostgresSaver` — production checkpoint backends

**Modules:** `langgraph.checkpoint.postgres`, `.shallow`, `.aio` — **separate package
`langgraph-checkpoint-postgres`, not installed in the verification venv; signatures
below are per-source but not independently re-verified here.**

Persist checkpoints in PostgreSQL via `psycopg3` (+ optional `psycopg_pool`).
`PostgresSaver` stores full history (every step — enables `get_state_history()` /
time-travel); `ShallowPostgresSaver` stores only the latest row per thread+namespace
(constant storage, no history, best for high-volume production agents that don't need
replay — deprecated as of `langgraph-checkpoint-postgres` 2.0.20 in favor of
`PostgresSaver` + `durability="exit"`). `AsyncPostgresSaver` is the `asyncio`-native
twin. All three expose `.from_conn_string(uri, pipeline=True)` as a **context
manager** (not a factory that returns a saver directly), and require calling
`.setup()` once to create tables/run migrations.

```python
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

**Internals (private, unverified-stability):** `create_checkpoint` / `empty_checkpoint`
/ `delta_channels_to_snapshot` (`langgraph.pregel._checkpoint`) build the `Checkpoint`
TypedDict (`v`, `id`, `ts`, `channel_values`, `channel_versions`, `versions_seen`) at
the end of every superstep and decide when a `DeltaChannel` needs a full snapshot blob;
`PregelExecutableTask` + `CacheKey` (`langgraph.types`) are the execution-time task
dataclass and its `(ns, key, ttl)` cache-entry identity; `DeltaChannelHistory`
(`langgraph.checkpoint.base`, beta) is the per-channel write-history record a custom
checkpointer's `get_delta_channel_history()` returns for `DeltaChannel` reconstruction.

---

### Streaming & Transformers

#### `stream_mode` overview + `get_stream_writer()` / `StreamWriter`

**Module:** `langgraph.config` (`get_stream_writer`), `langgraph.types` (`StreamWriter` alias)

`graph.stream()`/`astream()` accept `stream_mode` as a string or list of:
`"values"` (full state per step), `"updates"` (per-node deltas), `"messages"`
(token-by-token `(chunk, metadata)`), `"custom"` (arbitrary payloads from
`get_stream_writer()`), `"debug"` (task/checkpoint trace events), `"tasks"`
(`TaskPayload`/`TaskResultPayload`), `"checkpoints"` (`CheckpointPayload`), and
`"tools"` (structured tool-call events — see Tools & Tool Calling). `get_stream_writer()`
returns the callable bound to the current node/task; calling it with any
JSON-serialisable value pushes to `"custom"` immediately, without touching graph
state — the caller sees it in real time via `stream_mode="custom"`. It is a safe
no-op outside an active run. `runtime.stream_writer` is the equivalent accessed
through `Runtime`.

```python
StreamWriter: TypeAlias = Callable[[Any], None]
def get_stream_writer() -> StreamWriter: ...
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from typing_extensions import TypedDict

class State(TypedDict):
    items: list[str]; processed: list[str]

def batch_processor(state: State) -> dict:
    writer = get_stream_writer()
    results = []
    for i, item in enumerate(state["items"]):
        writer({"progress": i + 1, "total": len(state["items"])})
        results.append(item.upper())
    return {"processed": results}

graph = StateGraph(State).add_node("p", batch_processor).add_edge(START, "p").add_edge("p", END).compile()
for chunk in graph.stream({"items": ["a", "b"], "processed": []}, stream_mode="custom"):
    print(chunk)
```

#### `TaskPayload` / `TaskResultPayload` (`stream_mode="tasks"`)

**Module:** `langgraph.types`

`stream_mode="tasks"` emits a start event (`TaskPayload`) then a finish event
(`TaskResultPayload`) per node execution, sharing an `id`. In the default (v1) API
each is yielded directly; distinguish start from result by the presence of `"input"`
(start-only) — `result` is always present on the finish event (possibly `{}`), so
check `error`/`interrupts` rather than `"result" in data` to detect failure.

```python
class TaskPayload(TypedDict):
    id: str; name: str; input: Any; triggers: list[str]; metadata: NotRequired[dict]
class TaskResultPayload(TypedDict):
    id: str; name: str; error: str | None; interrupts: list[dict]; result: dict[str, Any]
```

```python
for event in graph.stream({"x": 1}, stream_mode="tasks"):
    if "input" in event:
        print(f"START  {event['name']} input={event['input']}")
    else:
        print(f"FINISH {event['name']} result={event['result']} error={event['error']}")
```

#### `CheckpointPayload` + `CheckpointTask` (`stream_mode="checkpoints"`)

**Module:** `langgraph.types`

`stream_mode="checkpoints"` emits one `CheckpointPayload` per checkpoint write —
`config`, `metadata`, `values` (full state), `next`, `parent_config`, and
`tasks: list[CheckpointTask]` (each with `id`, `name`, and optional `error`/`result`/
`interrupts`). Handy for audit trails and progress dashboards without touching node code.

```python
class CheckpointTask(TypedDict):
    id: str; name: str; error: NotRequired[str]; result: NotRequired[Any]
    interrupts: NotRequired[list[dict]]; state: StateSnapshot | RunnableConfig | None
class CheckpointPayload(TypedDict, Generic[StateT]):
    config: RunnableConfig | None; metadata: CheckpointMetadata; values: StateT
    next: list[str]; parent_config: RunnableConfig | None; tasks: list[CheckpointTask]
```

```python
for cp in graph.stream({"step": 0}, config, stream_mode="checkpoints"):
    print(f"step={cp['metadata'].get('step')} next={cp['next']} values={cp['values']}")
```

#### Typed v2 `StreamPart` union

**Module:** `langgraph.types`

Passing `version="v2"` to `stream()` wraps every chunk in a typed `TypedDict` with a
`type` discriminator, `ns` (namespace tuple), and `data` — replacing bare
tuples/dicts with an exhaustively-`match`able union: `ValuesStreamPart`,
`UpdatesStreamPart`, `MessagesStreamPart`, `CustomStreamPart`, `CheckpointStreamPart`,
`TasksStreamPart`, `DebugStreamPart`.

```python
StreamPart = ValuesStreamPart[OutputT] | UpdatesStreamPart | MessagesStreamPart \
           | CustomStreamPart | CheckpointStreamPart[StateT] | TasksStreamPart | DebugStreamPart[StateT]
```

```python
for part in graph.stream({"x": 0}, config, stream_mode=["values", "updates"], version="v2"):
    match part["type"]:
        case "values":
            print("STATE →", part["data"])
        case "updates":
            print("UPDATE →", part["data"])
```

#### Generative UI — `push_ui_message()` / `delete_ui_message()` / `UIMessage` / `RemoveUIMessage` / `ui_message_reducer`

**Module:** `langgraph.graph.ui`

The UI-streaming protocol for frontends that render components from stream events.
`push_ui_message` both writes a `UIMessage` to the `"custom"` stream **and** (unless
`state_key=None`) applies it to a state key (default `"ui"`) via `ui_message_reducer`
— pass `merge=True` to shallow-merge `props` into an existing component with the same
`id` (for incremental updates like a progress bar) instead of replacing it wholesale.
`delete_ui_message(id)` emits a `RemoveUIMessage` tombstone; `ui_message_reducer`
raises `ValueError` if asked to remove an unknown `id`. `message=` links a UI
component to a specific `AIMessage` via `metadata["message_id"]`.

```python
def push_ui_message(name: str, props: dict, *, id: str | None = None,
                     metadata: dict | None = None, message: AnyMessage | None = None,
                     state_key: str | None = "ui", merge: bool = False) -> UIMessage
def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage
def ui_message_reducer(left, right) -> list[AnyUIMessage]
class UIMessage(TypedDict):
    type: Literal["ui"]; id: str; name: str; props: dict; metadata: dict
class RemoveUIMessage(TypedDict):
    type: Literal["remove-ui"]; id: str
```

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import push_ui_message, delete_ui_message, ui_message_reducer, AnyUIMessage

class State(TypedDict):
    result: str; ui: Annotated[list[AnyUIMessage], ui_message_reducer]

def task(state: State) -> dict:
    bar = push_ui_message("ProgressBar", {"pct": 0})
    push_ui_message("ProgressBar", {"pct": 100}, id=bar["id"], merge=True)
    delete_ui_message(bar["id"])
    return {"result": "done"}

graph = StateGraph(State).add_node("task", task).add_edge(START, "task").add_edge("task", END).compile()
result = graph.invoke({"result": "", "ui": []})
print(result["ui"])  # [] — component was removed
```

#### v3 streaming API — `GraphRunStream` / `AsyncGraphRunStream`, `SubgraphRunStream`, `StreamChannel`, `StreamMux`, `ProtocolEvent`

**Modules:** `langgraph.stream.run_stream`, `.transformers`, `.stream_channel`, `._mux`, `._types`

**Status: `@beta`** — `graph.stream_events(version="v3")` / `astream_events(version="v3")`
returns a caller-driven stream handle instead of a flat iterator. There is no
background thread: iterating any projection attribute (`run.values`, `run.messages`,
`run.custom`, …) is what pulls the graph forward, one raw `ProtocolEvent` at a time.
Under `version="v3"`, `stream_mode=` is rejected — the modes that run are derived
from the registered transformers instead.

```python
def stream_events(
    self, input, config=None, *, version: Literal["v1", "v2", "v3"] = "v2",
    interrupt_before=None, interrupt_after=None, control: RunControl | None = None,
    transformers: Sequence[Callable[[tuple[str,...]], Any]] | None = None,
) -> Any: ...

class GraphRunStream:                      # sync; AsyncGraphRunStream mirrors this API
    values: StreamChannel[dict]; updates: StreamChannel[dict]; custom: StreamChannel[Any]
    messages: StreamChannel[ChatModelStream]; subgraphs: StreamChannel[SubgraphRunStream]
    lifecycle: StreamChannel[LifecyclePayload]
    @property
    def output(self) -> dict | None: ...
    @property
    def interrupted(self) -> bool: ...
    def abort(self) -> None: ...
    def interleave(self, *names: str) -> Iterator[tuple[str, Any]]: ...
```

`ValuesTransformer` → `run.values` and `MessagesTransformer` → `run.messages` are
the only two **always-registered native** transformers along with `LifecycleTransformer`
→ `run.lifecycle` and `SubgraphTransformer` → `run.subgraphs`. Everything else is
**opt-in** — pass the transformer **class** (not an instance — the mux calls
`factory(scope)` itself, once per subgraph namespace) via `compile(transformers=[...])`
or `stream_events(transformers=[...])`:

| Transformer | Projection | Built-in? | Notes |
|---|---|---|---|
| `ValuesTransformer` | `run.values` | yes (native, always on) | Full state snapshot per step; also tracks `run.interrupted`/`run.interrupts`. |
| `UpdatesTransformer` | `run.updates` | opt-in | Per-node delta dict, `{node_name: update}`. |
| `CustomTransformer` | `run.custom` | opt-in | Payloads written via `get_stream_writer()` inside nodes. |
| `MessagesTransformer` | `run.messages` | yes (native, always on) | Yields one `ChatModelStream`/`AsyncChatModelStream` handle per LLM call, pushed at `message-start`; iterate `.text`/`.reasoning`/`.tool_calls`/`.output` while the model is still generating. Only populated for models actually invoked with v3 streaming — a plain `AIMessage` state write produces `values` snapshots, not token events. |
| `CheckpointsTransformer` | `run.checkpoints` | opt-in | Requires a checkpointer; needs `stream_mode` to include `"checkpoints"` on v1/v2. |
| `DebugTransformer` | `run.debug` | opt-in | Surfaces `stream_mode="debug"` events (`"task"`/`"task_result"`/`"checkpoint"`). |
| `TasksTransformer` | `run.tasks` | opt-in | Surfaces `stream_mode="tasks"` events. |
| `LifecycleTransformer` | `run.lifecycle` | yes (native, built into v3) | `LifecyclePayload` dicts (`event`, `namespace`, `graph_name`, `trigger_call_id`, `cause`, `error`) for every **child** subgraph/`@task` namespace start/finish strictly below the transformer's own scope; the root namespace itself is never reported. |
| `SubgraphTransformer` | `run.subgraphs` | yes (native, built into v3) | Discovers **direct child** subgraphs only (`len(ns) == len(scope)+1`); each handle wraps its own scoped mini-mux and exposes `.values`/`.messages`/`.lifecycle`/`.subgraphs` recursively for grandchildren. |
| `ToolCallTransformer` | `run.tool_calls` | opt-in (`langgraph.prebuilt._tool_call_transformer`) | See Tools & Tool Calling. |

`run.output` / `.interrupted` / `.interrupts` drive the run to completion and return
the result; `run.abort()` stops early and is idempotent; `run.interleave(*names)`
merges multiple projections into one `(name, item)` stream in strict arrival order
(a monotonic push-stamp, not round-robin — `ainterleave` for async). Each projection
is a single-consumer, pump-driven queue: iterating it twice raises `RuntimeError`;
use `projection.tee(n)`/`.atee(n)` for real fan-out to multiple consumers sharing
one buffer. `SubgraphRunStream` handles have `wire_pump=False` — iterating them
silently drives the *root* graph, not just the subgraph. Use the `with`/`async with`
form so `abort()`/cleanup happens even on early exit.

Registration order matters for content-mutating transformers: `before_builtins =
True` runs a transformer **before** `MessagesTransformer`/`ToolCallTransformer`, so
a PII-redaction transformer can rewrite message text before it's snapshotted — but
take care not to mutate `namespace`/`id`/`result`/`error`/`interrupts`, which
`LifecycleTransformer`/`SubgraphTransformer` depend on for bookkeeping.

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage

def chat(state: MessagesState) -> dict:
    return {"messages": [AIMessage(content="hi")]}

graph = StateGraph(MessagesState).add_node("chat", chat).add_edge(START, "chat").add_edge("chat", END).compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}
with graph.stream_events({"messages": [HumanMessage("hello")]}, cfg, version="v3") as run:
    for snapshot in run.values:
        print(snapshot["messages"][-1].content)
    print("final:", run.output)
```

**Writing a custom `StreamTransformer`** (base class in `langgraph.stream._types`):
implement `init()` (return `{key: StreamChannel(...)}` — this becomes `run.<key>` if
`_native = True`, else `run.extensions["<key>"]`) and `process(event) -> bool` (return
`False` to suppress the event from the main log). Class vars: `required_stream_modes`
(which raw modes the mux must emit for you), `before_builtins`, `requires_async`/
`schedule(coro)` for async-only work fired from a sync `process()`.

```python
from langgraph.stream._types import StreamTransformer, ProtocolEvent
from langgraph.stream.stream_channel import StreamChannel

class EventCounter(StreamTransformer):
    _native = True
    def __init__(self, scope=()):
        super().__init__(scope)
        self._log: StreamChannel[int] = StreamChannel()
        self._n = 0
    def init(self):
        return {"event_count": self._log}
    def process(self, event: ProtocolEvent) -> bool:
        self._n += 1
        self._log.push(self._n)
        return True

with graph.stream_events({"messages": []}, version="v3", transformers=[EventCounter]) as run:
    for n in run.event_count:
        print("events so far:", n)
```

`StreamChannel[T]` (`langgraph.stream.stream_channel`) is the drainable, single-
consumer queue backing every projection — a second `iter()`/`aiter()` on the same
channel raises `RuntimeError` (sync/async mode is locked in on first use); use
`.tee(n)`/`.atee(n)` to fan out to multiple independent consumers sharing one
buffer. Items are pump-driven — the channel does not eagerly buffer everything up
front; iterating pulls the next item only on demand. `StreamMux`
(`langgraph.stream._mux`) is the central dispatcher: it assigns a monotonic `seq`
to every `ProtocolEvent`, routes it through all registered transformers, and
auto-forwards named `StreamChannel` pushes back into the main event log;
`_make_child(scope)` builds a scoped mini-mux for a nested subgraph (propagated
only from transformer *factories*, not pre-built instances). `ProtocolEvent` is
the universal envelope: `{"type": "event", "seq": int, "method": StreamMode,
"params": {"namespace": [...], "timestamp": ..., "data": ..., "interrupts": [...]}}`
— `seq` (root-mux-only, monotonic) is the reliable ordering key, not `timestamp`
(wall-clock, can go backwards).

**Internals (private, unverified-stability):** `StreamMessagesHandler` /
`StreamMessagesHandlerV2` (`langgraph.pregel._messages`) are the LangChain callback
handlers that actually power `stream_mode="messages"` (dedup by message `id`,
`run_inline=True` for ordering); `_RemoteGraphRunStream` / `_ChannelProjection` /
`_ProjectionRegistry` (`langgraph.pregel._remote_run_stream`) adapt `RemoteGraph`'s SDK
events to the same `GraphRunStream` surface; `_SubgraphRunStreamMixin`
(`langgraph.stream.run_stream`) carries `path`/`graph_name`/`trigger_call_id`/`status`
metadata for subgraph handles and delegates pump calls to the parent.
### Store & Memory

#### `BaseStore`, `InMemoryStore`

**Modules:** `langgraph.store.base` (`BaseStore`), `langgraph.store.memory` (`InMemoryStore`)

Cross-thread, cross-run key-value (optionally vector-searchable) memory abstraction
— distinct from a checkpointer, which is scoped to one `thread_id`. Items live under
hierarchical `namespace` tuples (like a folder path) plus a `key`. `BaseStore` has
exactly **two** abstract methods — `batch(ops)` and `abatch(ops)` — every
convenience method (`get`, `put`, `search`, `delete`, `list_namespaces`, and their
async twins) delegates to one of them, so a custom store adapter only needs to
implement those two. Namespaces must be non-empty tuples of non-empty strings with
no dots, and the root segment must not be `"langgraph"` (violations raise
`InvalidNamespaceError`).

`InMemoryStore` is the built-in implementation, backing `_data[namespace][key] ->
Item` and, when configured, `_vectors[namespace][key][field_path]` for
cosine-similarity search. Without `index=`, `search()` only supports `filter=`
(exact-match); with it, both `filter=` and semantic `query=` work, and results are
`SearchItem`s carrying a cosine-similarity `.score`. `put(namespace, key, value,
index=...)`: `index=False` skips embedding for that item; `index=["field.path"]`
embeds only those JSON paths. Sync `batch()` embeds queries via a
`ThreadPoolExecutor` so it never blocks an async caller from a sync path;
`abatch()` uses `asyncio.gather`. `InMemoryStore.supports_ttl` is `False` — TTL
requires an adapter that opts in (e.g. `PostgresStore`); passing `ttl=` on
`InMemoryStore` raises `NotImplementedError`.

```python
class BaseStore(ABC):
    supports_ttl: bool = False
    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]: ...
    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]: ...

def __init__(self, *, index: IndexConfig | None = None) -> None: ...
```

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(("users", "alice"), "preferences", {"theme": "dark"})
item = store.get(("users", "alice"), "preferences")
print(item.value, item.namespace, item.key, item.created_at)
results = store.search(("users",), filter={"theme": "dark"})
print(store.list_namespaces(prefix=("users",)))
store.delete(("users", "alice"), "preferences")

# Semantic search — embed can be a plain sync/async callable, no LangChain needed
def embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t))] for t in texts]

sstore = InMemoryStore(index={"dims": 1, "embed": embed, "fields": ["text"]})
sstore.put(("docs",), "py", {"text": "Python programming guide"})
for hit in sstore.search(("docs",), query="python scripting", limit=2):
    print(hit.key, hit.score)
```

#### `Item` / `SearchItem`

**Module:** `langgraph.store.base`

`Item` is what `store.get()` returns: `value`, `key`, `namespace`, `created_at`,
`updated_at` (both always timezone-aware `datetime`s). `SearchItem` extends it with
`score: float | None` (cosine similarity, present on semantic-search results,
`None` when not ranked). `store.get()` returns `None` for a missing key — always
check before accessing `.value`.

```python
class Item:
    __slots__ = ("value", "key", "namespace", "created_at", "updated_at")
    def dict(self) -> dict: ...
class SearchItem(Item):
    __slots__ = ("score",)   # float | None — None when the search had no query, filter-only
```

#### `GetOp`, `PutOp`, `SearchOp`, `ListNamespacesOp`, `MatchCondition`

**Module:** `langgraph.store.base`

The batch-operation protocol every store implements. `SearchOp.filter` supports
`$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte` (a bare value means `$eq`);
`ListNamespacesOp.match_conditions` uses `MatchCondition(match_type="prefix" |
"suffix", path=(...))`, where `"*"` in the path matches one arbitrary segment.

```python
class GetOp(NamedTuple):
    namespace: tuple[str, ...]; key: str; refresh_ttl: bool = True
class PutOp(NamedTuple):
    namespace: tuple[str, ...]; key: str
    value: dict | None                      # None => delete
    index: Literal[False] | list[str] | None = None
    ttl: float | None = None                # minutes; requires supports_ttl
class SearchOp(NamedTuple):
    namespace_prefix: tuple[str, ...]
    filter: dict | None = None
    limit: int = 10; offset: int = 0
    query: str | None = None
    refresh_ttl: bool = True
class MatchCondition(NamedTuple):
    match_type: Literal["prefix", "suffix"]; path: tuple[str | Literal["*"], ...]
class ListNamespacesOp(NamedTuple):
    match_conditions: tuple[MatchCondition, ...] | None = None
    max_depth: int | None = None; limit: int = 100; offset: int = 0
```

```python
from langgraph.store.base import GetOp, PutOp, ListNamespacesOp, MatchCondition
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.batch([PutOp(("counters",), "views", {"count": 42})])
(get_result,) = store.batch([GetOp(("counters",), "views")])
print(get_result.value)

(namespaces,) = store.batch([
    ListNamespacesOp(match_conditions=(MatchCondition(match_type="prefix", path=("counters",)),))
])
print(namespaces)
```

#### `IndexConfig` + `TTLConfig`

**Module:** `langgraph.store.base`

The two config `TypedDict`s that unlock vector search and expiry on `InMemoryStore` /
`PostgresStore`. `fields` supports a small JSON-path syntax (`["$"]` embeds the whole
document — the default; `["a.b"]` nested; `["items[*].x"]` per-array-element;
`["{a,b}"]` multi-field). Per-item overrides: `store.put(..., index=["field"])` or
`index=False` to skip indexing that one item. `TTLConfig` has **four** fields —
`omit_expired` is easy to miss in older write-ups.

```python
class IndexConfig(TypedDict, total=False):
    dims: int; embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str; fields: list[str]
class TTLConfig(TypedDict, total=False):
    refresh_on_read: bool          # default True
    omit_expired: bool             # default False — exclude expired items from reads/lists
    default_ttl: float | None      # minutes; None = never
    sweep_interval_minutes: int | None
```

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={"dims": 16, "embed": my_embed, "fields": ["title", "chapters[*].content"]},
)
```

#### `EmbeddingsLambda` + `ensure_embeddings` + `get_text_at_path` / `tokenize_path`

**Module:** `langgraph.store.base.embed`

`ensure_embeddings()` normalises whatever you pass as `index["embed"]` — a LangChain
`Embeddings` instance, a plain sync/async callable `list[str] -> list[list[float]]`,
or a `"provider:model"` string (dispatched to `langchain.embeddings.init_embeddings`,
requires `langchain>=0.3.9`) — into a LangChain `Embeddings` instance, wrapping bare
callables in `EmbeddingsLambda`. `get_text_at_path`/`tokenize_path` implement the
`fields` JSON-path syntax used to pick which parts of a stored value get embedded:
dot paths (`"a.b"`), array indexing (`"[0]"`, `"[*]"`, `"[-1]"`), multi-field
selection (`"{a,b.c}"`), and `"$"` for the whole object serialized as JSON.

```python
def ensure_embeddings(embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str | None) -> Embeddings
class EmbeddingsLambda(Embeddings):
    def __init__(self, func: EmbeddingsFunc | AEmbeddingsFunc) -> None: ...
def get_text_at_path(obj: Any, path: str | list[str]) -> list[str]
def tokenize_path(path: str) -> list[str]
```

```python
from langgraph.store.base.embed import get_text_at_path

doc = {"title": "Guide", "sections": [{"heading": "Intro", "body": "..."}]}
print(get_text_at_path(doc, "sections[*].heading"))   # ['Intro']
print(get_text_at_path(doc, "{title,sections[*].heading}"))
```

#### `AsyncBatchedBaseStore`

**Module:** `langgraph.store.base.batch`

Base class for production async store adapters (Redis, Postgres, …) accessed from
inside a running event loop. Instead of one round-trip per `aget`/`aput`/`asearch`,
it queues ops on an `asyncio.Queue` drained by a background task that batches
everything queued in the same tick into one `abatch()` call, deduplicating repeated
reads and collapsing consecutive puts to the same key. A `@_check_loop` guard on
the sync methods raises `asyncio.InvalidStateError` if you call them from the same
event loop the store's background task owns (that would deadlock) — always use
the async variants (`aget`, `aput`, …) inside async code. `InMemoryStore` does
**not** inherit from this — it's a plain `BaseStore` — so use
`AsyncBatchedBaseStore` as the base only for stores whose backend genuinely
benefits from request coalescing (e.g. a store wrapping a single shared HTTP/DB
connection).

```python
class AsyncBatchedBaseStore(BaseStore):
    def __init__(self) -> None: ...   # starts a background asyncio.Task drainer
```

```python
import asyncio
async def safe():
    store = MyAsyncBatchedStore()
    await store.aput(("ns",), "key1", {"data": 1})   # correct
    item = await store.aget(("ns",), "key1")
    # store.get(("ns",), "key1")  # WRONG inside this loop — deadlocks
asyncio.run(safe())
```

#### `PostgresStore` / `AsyncPostgresStore` / `PoolConfig` / `ANNIndexConfig` / `HNSWConfig` / `IVFFlatConfig` / `PostgresIndexConfig`

**Module:** `langgraph.store.postgres` and `.base` — **separate package
`langgraph-checkpoint-postgres`, not installed in the verification venv; not
independently re-verified here.**

Durable, shared-across-threads key-value storage in Postgres with optional pgvector
ANN indexing. `PostgresIndexConfig` extends `IndexConfig` with `distance_type` and
`ann_index_config`, which is either `HNSWConfig` (proximity graph — best default,
consistent high recall, use when the dataset changes frequently) or `IVFFlatConfig`
(cluster-based — faster to build, better for very large, mostly-static datasets;
build the index *after* bulk-loading data). `PoolConfig` (`min_size`, `max_size`,
`kwargs`) configures the underlying `psycopg_pool.ConnectionPool`.

```python
class PoolConfig(TypedDict, total=False):
    min_size: int; max_size: int | None; kwargs: dict
class HNSWConfig(ANNIndexConfig, total=False):
    kind: Literal["hnsw"]; m: int; ef_construction: int
class IVFFlatConfig(ANNIndexConfig, total=False):
    kind: Literal["ivfflat"]; nlist: int
```

```python
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig, HNSWConfig

index: PostgresIndexConfig = {
    "dims": 1536, "embed": my_embed, "fields": ["text"],
    "distance_type": "cosine",
    "ann_index_config": HNSWConfig(kind="hnsw", m=16, ef_construction=64),
}
with PostgresStore.from_conn_string(DB_URI, index=index) as store:
    store.setup()
```

---

### Tools & Tool Calling

#### `ToolNode` — full API

**Module:** `langgraph.prebuilt.tool_node` (re-exported from `langgraph.prebuilt`)

Executes every tool call found in the last `AIMessage`, in parallel where possible.

```python
def __init__(
    self, tools: Sequence[BaseTool | Callable], *, name: str = "tools",
    tags: list[str] | None = None,
    handle_tool_errors: bool | str | Callable[..., str] | type[Exception] | tuple[type[Exception], ...] = _default_handle_tool_errors,
    messages_key: str = "messages",
    wrap_tool_call: ToolCallWrapper | None = None,
    awrap_tool_call: AsyncToolCallWrapper | None = None,
) -> None: ...
```

`handle_tool_errors` has five forms, verified: `True` (default-style template), a
fixed `str` (used verbatim), an exception `type` or `tuple[type, ...]` filter, or a
`Callable[[Exception], str]` formatter; `False` disables catching entirely (errors
propagate to the graph's `error_handler`, if any). **The actual default is not the
literal `True`** — it's the function `_default_handle_tool_errors`, which always
catches argument-validation errors (`ToolInvocationError`) but re-raises genuine
tool-execution errors, so the model can self-correct on bad arguments while a real
bug still surfaces. `messages_key` lets `ToolNode` read/write a differently-named
state field.

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

node = ToolNode([divide], handle_tool_errors="Tool failed; try different inputs.")
call = {"name": "divide", "args": {"a": 1.0, "b": 0.0}, "id": "c1", "type": "tool_call"}
result = node.invoke({"messages": [AIMessage(content="", tool_calls=[call])]})
print(result["messages"][0].content)  # Tool failed; try different inputs.
```

#### `wrap_tool_call` middleware — `ToolCallRequest`, `ToolCallWrapper`, `ToolInvocationError`

**Module:** `langgraph.prebuilt.tool_node`

`wrap_tool_call` (sync) / `awrap_tool_call` (async) is middleware around every tool
invocation — sanitise args, rate-limit, retry, cache, or short-circuit — via a
`(request: ToolCallRequest, execute) -> ToolMessage | Command` interceptor that may
call `execute(request)` zero, one, or many times.

```python
@dataclass
class ToolCallRequest:
    tool_call: ToolCall        # {"name", "args", "id", "type"}
    tool: BaseTool | None      # None if the model named an unregistered tool
    state: Any
    runtime: ToolRuntime

    def override(self, **overrides) -> "ToolCallRequest": ...   # dataclasses.replace; immutable-update pattern

ToolCallWrapper = Callable[[ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]], ToolMessage | Command]
```

- Direct attribute assignment on `ToolCallRequest` emits a `DeprecationWarning` —
  always go through `.override(tool_call=..., state=...)`, which returns a new
  instance, so interceptors never corrupt shared state across parallel tool calls.
- `request.tool is None` when the model hallucinated a tool name; `execute()` still
  returns an invalid-tool `ToolMessage` rather than raising, so branch on
  `request.tool is None` explicitly if you need different handling.
- `ToolInvocationError` (a `ToolException` subclass, raised internally on bad
  tool-call arguments) filters its message down to only the parameters the LLM
  actually controls — injected parameters (`InjectedState`/`InjectedStore`/
  `ToolRuntime`) are excluded, so the model isn't confused by validation errors on
  arguments it never supplied. `ToolNode` catches it by default and turns it into
  an error `ToolMessage` — override `handle_tool_errors` to change that behaviour.

```python
class ToolInvocationError(ToolException):
    message: str; tool_name: str; tool_kwargs: dict; source: ValidationError
```

```python
from typing import Callable
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

@tool
def set_volume(value: int) -> str:
    """Set the volume level."""
    return f"Volume set to {value}"

def clamp_value(request: ToolCallRequest, execute: Callable) -> ToolMessage:
    args = dict(request.tool_call["args"])
    args["value"] = max(0, min(100, args.get("value", 0)))
    return execute(request.override(tool_call={**request.tool_call, "args": args}))

node = ToolNode([set_volume], wrap_tool_call=clamp_value)
call = {"name": "set_volume", "args": {"value": 150}, "id": "v1", "type": "tool_call"}
result = node.invoke({"messages": [AIMessage(content="", tool_calls=[call])]})
print(result["messages"][0].content)  # Volume set to 100
```

#### `InjectedState`, `InjectedStore`, `ToolRuntime`

**Module:** `langgraph.prebuilt` (re-exports `langgraph.prebuilt.tool_node`)

Three annotation mechanisms that hide framework-supplied parameters from the
model's tool schema entirely, then fill them in at execution time. `InjectedState`
and `InjectedStore` are `InjectedToolArg` subclasses from `langchain_core`, so
annotate a parameter with them explicitly; `ToolRuntime` needs no `Annotated`
wrapper — a parameter simply typed `runtime: ToolRuntime` is matched by name+type
and auto-injected, bundling everything the other two (plus `get_stream_writer()`)
used to require separately.

```python
class InjectedState(InjectedToolArg):
    def __init__(self, field: str | None = None) -> None: ...   # None => inject the whole state dict; else state[field]
class InjectedStore(InjectedToolArg): ...                       # injects the compiled graph's BaseStore

@dataclass
class ToolRuntime(Generic[ContextT, StateT]):
    state: StateT; context: ContextT; config: RunnableConfig
    stream_writer: StreamWriter; tool_call_id: str | None
    store: BaseStore | None; tools: list[BaseTool]
    execution_info: ExecutionInfo | None; server_info: ServerInfo | None
    def emit_output_delta(self, delta: Any) -> None: ...   # partial output on the "tools" channel; silent no-op if not streaming
```

`execution_info` gives you `checkpoint_id`/`task_id`/`thread_id`/`node_attempt`
(1-indexed, increments on retry) for observability inside a tool; `server_info` is
populated only under LangGraph Server/Platform deployments (`None` locally).

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore, ToolNode
from langchain_core.messages import AIMessage

class AppState(TypedDict):
    messages: list
    user_name: str

@tool
def greet(greeting: str, name: Annotated[str, InjectedState("user_name")]) -> str:
    """Greet the user; name is injected from state, invisible to the model."""
    return f"{greeting}, {name}!"

node = ToolNode([greet])
call = {"name": "greet", "args": {"greeting": "Hello"}, "id": "g1", "type": "tool_call"}
result = node.invoke({"messages": [AIMessage(content="", tool_calls=[call])], "user_name": "Alice"})
print(result["messages"][0].content)  # Hello, Alice!
```

```python
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime

@tool
def analyse(query: str, runtime: ToolRuntime) -> str:
    """Analyse a query, streaming progress."""
    for step in ["planning", "searching", "synthesising"]:
        runtime.emit_output_delta({"step": step})
    return f"Complete analysis of: {query}"
```

#### `ToolCallWithContext`

**Module:** `langgraph.prebuilt.tool_node`. The internal `TypedDict` payload used
to dispatch a tool call in parallel via `Send`, while still carrying the state
snapshot at dispatch time: `{"tool_call": ToolCall, "__type": "tool_call_with_context",
"state": Any}`. `ToolNode` recognizes the `"__type"` discriminator (double-underscore
to avoid colliding with any user state key literally named `type`) and populates
`ToolRuntime.state` from the `"state"` field. This is what enables independently
interruptible, parallel per-tool-call execution in ReAct-style graphs.

#### `tools_condition`

**Module:** `langgraph.prebuilt.tool_node` (re-exported from `langgraph.prebuilt`)

The standard conditional-edge function for tool-calling loops. Inspects the last
message: an `AIMessage` with non-empty `.tool_calls` routes to `"tools"`, otherwise
`"__end__"`. Accepts a bare message list, a state dict (reads
`state[messages_key]`), or a Pydantic `BaseModel` (reads
`getattr(state, messages_key)`); pass `messages_key=` if your state doesn't use the
default `"messages"` field.

```python
def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]: ...
```

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

builder = StateGraph(State)
builder.add_node("model", lambda s: s)          # stand-in for a real LLM call
builder.add_node("tools", ToolNode([]))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)   # -> "tools" or END
builder.add_edge("tools", "model")
```

#### `ToolCallTransformer` + `ToolCallStream` (`stream_mode="tools"`)

**Modules:** `langgraph.prebuilt._tool_call_transformer` / `._tool_call_stream`
(re-exported: `ToolCallTransformer` from `langgraph.prebuilt`)

An opt-in `_native = True` transformer (`required_stream_modes = ("tools",)`) that
converts raw `"tools"`-channel protocol events into a live handle per tool call,
exposed on `run.tool_calls` — register at compile time
(`compile(transformers=[ToolCallTransformer])`), then iterate `run.tool_calls`
while streaming with `stream_mode="tools"`. Under the hood this is powered by
`StreamToolCallHandler` (`langgraph.pregel._tools`, private) which fires
`tool-started`/`tool-output-delta`/`tool-finished`/`tool-error` events; tag a tool
with `TAG_NOSTREAM` to suppress its events entirely. `process()` always returns
`True`, so raw `"tools"` channel events still flow to any other consumer alongside
the per-call handles.

```python
class ToolCallStream:
    tool_call_id: str; tool_name: str; input: dict | None
    output: Any               # set on tool-finished
    error: str | None         # set on tool-error
    completed: bool
    output_deltas: StreamChannel[Any]   # iterate sync (`for`) or async (`async for`)
```

```python
from langgraph.prebuilt import ToolNode, ToolCallTransformer
graph = builder.compile(transformers=[ToolCallTransformer])
async with graph.astream({"messages": []}, stream_mode="tools", version="v2") as run:
    async for tc in run.tool_calls:
        print("started:", tc.tool_name, tc.input)
        async for delta in tc.output_deltas:
            print("  delta:", delta)
        print("final:", tc.output)
```

#### `create_react_agent` (deprecated) — and the `AgentState` / `ValidationNode` migration

**Module:** `langgraph.prebuilt.chat_agent_executor` / `.tool_validator`

`create_react_agent` compiles a `"agent"` + `"tools"` ReAct loop. **Verified
deprecated** in the installed venv — the function carries
`@deprecated(category=LangGraphDeprecatedSinceV10)` — in favor of `create_agent`
from `langchain.agents`; it remains fully functional in 1.2.11 and existing code
keeps working (with a warning).

```python
def create_react_agent(
    model, tools, *, prompt=None, response_format=None,
    pre_model_hook=None, post_model_hook=None,
    state_schema=None, context_schema=None,
    checkpointer=None, store=None,
    interrupt_before=None, interrupt_after=None, debug=False,
    version: Literal["v1", "v2"] = "v2", name=None,
) -> CompiledStateGraph: ...
```

- `pre_model_hook(state) -> dict | None` can return `{"llm_input_messages": [...]}`
  to give the model a trimmed/annotated view for *this call only*, without
  mutating the persisted `messages` field (it bypasses the `add_messages` reducer).
- `post_model_hook(state) -> dict | Command | None` runs after the LLM call;
  useful for token-budget enforcement. If it returns `Command(goto="__end__")`
  while the last `AIMessage` still carries pending `tool_calls`, the prebuilt
  router will still try to schedule them unless you also strip/replace the
  message so the router sees no tool calls.
- `response_format=` adds a `generate_structured_response` node making a
  **separate** model call via `with_structured_output()` after the loop ends —
  extra latency/cost, result lands in `state["structured_response"]`.
- `version="v2"` (default) dispatches each tool call as an independent `Send`
  task so one tool's failure doesn't block sibling calls; `"v1"` runs them
  together in one `ToolNode` invocation.
- `AgentState` (the default `state_schema`) is `{"messages": Annotated[Sequence[BaseMessage],
  add_messages], "remaining_steps": NotRequired[RemainingSteps]}`; a custom
  `state_schema` must include a `remaining_steps` field or `create_react_agent`
  raises `ValueError`.
- `AgentState`, `AgentStatePydantic`, `AgentStateWithStructuredResponse` are each
  separately `@deprecated`, moved to `langchain.agents`; migrate to
  `response_format=` instead of a custom structured-response state class, or a
  hand-written `TypedDict`/Pydantic model with the same two fields for a custom
  `state_schema`.
- `ValidationNode` (schema-only tool-argument validator, no execution) is also
  `@deprecated` — migrate to `create_agent(response_format=...)` or tool-level
  `handle_tool_errors=` with a pydantic `args_schema` on the tool itself.

**Correction vs. some older write-ups:** `from langgraph.prebuilt import
AgentState` does **not** merely warn — it raises `ImportError` in the installed
1.2.11 (`AgentState` isn't re-exported from the `langgraph.prebuilt` package
`__init__` at all). Import it from `langgraph.prebuilt.chat_agent_executor`
instead.

```python
from typing import Annotated, Sequence, NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import create_react_agent

class MyAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]

agent = create_react_agent(model, tools=[], state_schema=MyAgentState)  # no deprecation warning
```

```python
# Modern equivalent of ValidationNode — pydantic args_schema + ToolNode.handle_tool_errors
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str; max_results: int = 10

@tool(args_schema=SearchParams)
def search(query: str, max_results: int = 10) -> str:
    """Search the web."""
    return f"Found {max_results} results"

tool_node = ToolNode([search], handle_tool_errors=True)
```

#### `ToolOutputMixin`

**Module:** re-exported from `langgraph.types`; actually defined in
`langchain_core.messages.tool` (some older write-ups place it in
`langgraph.prebuilt.tool_node` — that location is wrong, it never lived there).

An empty marker mixin. When a `BaseTool` returns an object, `ToolNode` uses it (not
its string coercion) directly only if it's a `ToolMessage`, a `Command`, or a list of
those — `ToolMessage` and `Command` both inherit `ToolOutputMixin`. Anything else is
`str()`-coerced and wrapped in a fresh `ToolMessage`. Subclass it yourself to
future-proof a custom structured tool-return type.

```python
from langgraph.types import ToolOutputMixin   # works: re-exported

class RichToolResult(ToolOutputMixin):
    def __init__(self, content: str, metadata: dict):
        self.content = content; self.metadata = metadata
    def __str__(self) -> str:
        return self.content
```

#### `Pregel.as_tool()` (beta)

**Module:** `langgraph.pregel.main` (delegates to
`langchain_core.tools.convert_runnable_to_tool`). Wraps any compiled graph as a
`BaseTool`.

```python
def as_tool(
    self, args_schema: type[BaseModel] | None = None, *,
    name: str | None = None, description: str | None = None,
    arg_types: dict[str, type] | None = None,
) -> BaseTool: ...
```

Schema is inferred from the graph's `TypedDict`/Pydantic input schema when
possible; use `arg_types={"key": type}` to expose only a subset of state keys as
tool arguments (avoids surfacing output-only fields as required inputs the model
must fill in). The returned tool's `.invoke()` calls `graph.invoke()`.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class SumState(TypedDict):
    a: int; b: int; result: int

graph = StateGraph(SumState)
graph.add_node("add", lambda s: {"result": s["a"] + s["b"]})
graph.add_edge(START, "add"); graph.add_edge("add", END)
compiled = graph.compile()

tool = compiled.as_tool(name="sum_graph", description="Add two integers.", arg_types={"a": int, "b": int})
print(tool.invoke({"a": 3, "b": 4}))  # {'a': 3, 'b': 4, 'result': 7}
```

**Internals (private, unverified-stability):** `_IdleProgressCallbackHandler`
(`langgraph.pregel._retry`) is the callback handler that resets a node's idle-timeout
clock on *every* LangChain event (LLM token, tool start/end, retriever hit, …) when
`TimeoutPolicy(refresh_on="auto")` is in effect.
---

### Caching

#### `CachePolicy` + `CacheKey`

**Module:** `langgraph.types`

`CachePolicy` memoises a node's or `@task`'s return value keyed on its input.
`key_func` defaults to a pickle-hash of the full input — supply your own to
normalise (case-fold, strip irrelevant fields, namespace by user) or to compute a
deterministic key. **Both a graph-level cache backend (`compile(cache=...)` /
`@entrypoint(cache=...)`) and a per-node/task `cache_policy=` are required
together** — either one alone silently does nothing.

```python
@dataclass(frozen=True)
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key   # (input) -> str|bytes; default: pickle hash
    ttl: int | None = None                   # seconds; None = never expires

class CacheKey(NamedTuple):
    ns: tuple[str, ...]; key: str; ttl: int | None
```

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
from typing_extensions import TypedDict

class State(TypedDict):
    query: str; result: str

call_log = []
def expensive(state: State) -> dict:
    call_log.append(state["query"])
    return {"result": f"answer:{state['query']}"}

def query_only_key(node_input: dict) -> str:
    return node_input.get("query", "")

builder = StateGraph(State)
builder.add_node("expensive", expensive, cache_policy=CachePolicy(key_func=query_only_key, ttl=60))
builder.add_edge(START, "expensive"); builder.add_edge("expensive", END)
graph = builder.compile(cache=InMemoryCache())
graph.invoke({"query": "foo", "result": ""})
graph.invoke({"query": "foo", "result": ""})   # served from cache
assert len(call_log) == 1
```

#### `BaseCache`, `InMemoryCache`

**Module:** `langgraph.cache.base` / `langgraph.cache.memory`

```python
class BaseCache(ABC, Generic[ValueT]):
    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=False)
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]: ...
    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None: ...
    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None: ...
    # + async twins aget/aset/aclear

class InMemoryCache(BaseCache):
    def __init__(self, *, serde: SerializerProtocol | None = None) -> None: ...
```

`FullKey = tuple[Namespace, str]`, `Namespace = tuple[str, ...]` — namespacing
gives per-task, per-user cache isolation without separate cache instances.
`InMemoryCache` protects its dict with a `threading.RLock`; TTL expiry is stored as
an absolute expiry timestamp on `set()` and checked lazily on `get()` (no
background sweep). `clear(namespaces=[ns])` evicts one namespace; `clear()` with
no args wipes everything.

```python
import time
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()
ns = ("pipeline", "step1")
cache.set({(ns, "run_42"): ({"result": 99}, 2)})   # 2-second TTL
print(cache.get([(ns, "run_42")]))    # {(('pipeline', 'step1'), 'run_42'): {'result': 99}}
time.sleep(2.1)
print(cache.get([(ns, "run_42")]))    # {}  — expired, lazily evicted
```

#### `RedisCache`

**Module:** `langgraph.cache.redis`. **Install:** `pip install redis`.

Drop-in Redis-backed replacement for `InMemoryCache` — uses `MGET`/pipelined
`SET`/`SETEX` for efficient batch reads/writes. When Redis is unreachable, operations
silently no-op (graph still runs correctly, just without caching). Keys are namespaced
`{prefix}{ns1}:{ns2}:...:{cache_key}`; default `serde` is `JsonPlusSerializer()` — wrap
it in `EncryptedSerializer` to encrypt cached values at rest.

```python
RedisCache(redis: Any, *, serde: SerializerProtocol | None = None, prefix: str = "langgraph:cache:")
```

```python
import redis
from langgraph.cache.redis import RedisCache

cache = RedisCache(redis.Redis(host="localhost", port=6379), prefix="myapp:")
graph = builder.compile(cache=cache)
```

#### `clear_cache()` / `aclear_cache()` — graph-level and task-level

**Modules:** `langgraph.pregel.main` (methods on `CompiledStateGraph`),
`langgraph.func` (methods on the `_TaskFunction` a `@task`-decorated function is
wrapped in)

`compiled_graph.clear_cache(nodes: Sequence[str] | None = None)` / `await
...aclear_cache(...)` invalidates cached **node** results for a graph (or specific
nodes) — but **not** `@task` caches, which live in a separate namespace; clear
those independently with `my_task.clear_cache(cache)` / `await
my_task.aclear_cache(cache)`. Calling `clear_cache()` on a graph compiled without
`cache=` raises `ValueError`. A task's cache namespace is `("langgraph", "cache",
"writes", "<function identifier>")`; a lambda or dynamic function with no stable
`__qualname__` falls back to `"__dynamic__"`.

```python
def clear_cache(self, cache: BaseCache) -> None: ...     # on a @task function
async def aclear_cache(self, cache: BaseCache) -> None: ...
```

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()

@task(cache_policy=CachePolicy(ttl=3600))
def fetch_data(key: str) -> str:
    return f"data_for_{key}"

@entrypoint(cache=cache)
def pipeline(key: str) -> str:
    return fetch_data(key).result()

pipeline.invoke("k1")
fetch_data.clear_cache(cache)   # next call for "k1" re-executes fetch_data only
```

---

### Error Handling, Retry & Timeout Policies

#### `RetryPolicy` (single and chained)

**Module:** `langgraph.types`

Controls node/task retry behaviour with exponential backoff. `add_node(...,
retry_policy=...)` and `@task(retry_policy=...)` accept either one `RetryPolicy` or a
**list** — LangGraph tries each in order and uses the first whose `retry_on` predicate
matches the raised exception (no match → the exception propagates immediately). This
lets you give rate-limit errors a slow, patient policy and network blips a fast one,
without a custom callable.

```python
class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    max_attempts: int = 3
    jitter: bool = True
    retry_on: type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool] = default_retry_on
```

Wait time per attempt is `min(max_interval, initial_interval * backoff_factor **
(attempt - 1))`, optionally jittered; `max_attempts` counts the **total** attempts
including the first try. `default_retry_on` is a blocklist strategy: always
retries `ConnectionError` and 5xx `httpx.HTTPStatusError`/`requests.HTTPError`;
never retries a fixed set of deterministic-failure types (`ValueError`,
`TypeError`, `ArithmeticError`, `ImportError`, `LookupError`, `NameError`,
`SyntaxError`, `RuntimeError`, `ReferenceError`, `StopIteration`,
`StopAsyncIteration`, `OSError`); retries **everything else** by default (so
new/unknown transient exception types from future SDK versions are retried
without any predicate update). Supply a `Callable[[Exception], bool]` for custom
logic, e.g. to also retry 429s or to never retry a specific application exception.

```python
import httpx
from langgraph.types import RetryPolicy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

rate_limit_policy = RetryPolicy(initial_interval=5.0, max_attempts=6,
    retry_on=lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429)
network_policy = RetryPolicy(initial_interval=0.5, max_attempts=3,
    retry_on=lambda e: isinstance(e, httpx.TransportError))

builder = StateGraph(State)
builder.add_node("fetch", my_fetch_fn, retry_policy=[rate_limit_policy, network_policy])
builder.add_edge(START, "fetch"); builder.add_edge("fetch", END)
graph = builder.compile()
```

#### `TimeoutPolicy` (`.coerce()`, `run_timeout` vs `idle_timeout`)

**Module:** `langgraph.types`

Two independent clocks per node/task attempt: `run_timeout` (hard wall-clock cap,
never refreshed) and `idle_timeout` (max time without observable progress).
`refresh_on="auto"` (default) resets the idle clock on any LangChain callback
event (LLM token, tool call, …) *and* explicit `runtime.heartbeat()`;
`"heartbeat"` restricts it to *only* explicit `heartbeat()` calls. A bare
`float`/`timedelta` passed anywhere a `TimeoutPolicy` is expected is coerced via
`TimeoutPolicy.coerce()` to `TimeoutPolicy(run_timeout=value)`.

```python
@dataclass(frozen=True)
class TimeoutPolicy:
    run_timeout: float | timedelta | None = None
    idle_timeout: float | timedelta | None = None
    refresh_on: Literal["auto", "heartbeat"] = "auto"
    @classmethod
    def coerce(cls, value: float | timedelta | TimeoutPolicy | None) -> TimeoutPolicy | None: ...
```

- **Timeouts rely on `asyncio` cancellation — they only fire reliably on `async
  def` nodes/tasks.** A sync node/task with a `timeout=` raises at setup time
  (`sync_timeout_unsupported`) in the current version — Python cannot safely
  pre-empt a running sync function; only `asyncio.Task.cancel()` at an `await`
  point is a safe cancellation point.
- Each **retry** attempt gets a fresh timeout clock — the timeout budget is
  per-attempt, not cumulative across retries.

```python
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

class State(TypedDict):
    processed: int

async def batch(state: State, runtime: Runtime) -> dict:
    for _ in range(10):
        await asyncio.sleep(0.05)
        runtime.heartbeat()   # keeps idle clock alive
    return {"processed": 10}

builder = StateGraph(State)
builder.add_node("batch", batch, timeout=TimeoutPolicy(idle_timeout=1.0, refresh_on="heartbeat"))
builder.add_edge(START, "batch"); builder.add_edge("batch", END)
graph = builder.compile()
```

#### `error_handler` on `add_node` / `set_node_defaults`

**Module:** `langgraph.graph.state`

A per-node (or graph-wide via `set_node_defaults(error_handler=...)`) fallback node
that runs when the node exhausts its retries and still raises. Retries always run
*first*; the handler fires only after they're exhausted. The handler function's
second parameter is a `NodeError` (see below), not the raw exception; the handler
itself is never retried or caught if it too raises.

```python
def error_handler(state: State, error: NodeError) -> dict | Command: ...
builder.add_node("risky", risky_fn, retry_policy=RetryPolicy(max_attempts=2), error_handler=error_handler)
```

```python
from langgraph.errors import NodeError
from langgraph.types import Command

def api_error_handler(state, error: NodeError) -> Command:
    return Command(update={"error_info": f"[{error.node}] {type(error.error).__name__}: {error.error}"})
```

#### `NodeError`, `NodeTimeoutError`, `NodeCancelledError`, `GraphDrained` + the exception hierarchy

**Module:** `langgraph.errors`

`NodeError` is **a frozen dataclass, not an exception** — `(node: str, error:
BaseException)` — injected as the second parameter into `error_handler` functions.
`NodeTimeoutError` deliberately does **not** inherit from the built-in
`TimeoutError` (an `OSError` subclass excluded by `default_retry_on`) so the
default `RetryPolicy` retries a timed-out attempt automatically; carries `node`,
`kind` (`"run"` | `"idle"`), `elapsed`, `run_timeout`, `idle_timeout`.
`NodeCancelledError` wraps a user-raised `asyncio.CancelledError` so it surfaces
through the normal error path rather than a silent teardown (framework-initiated
cancellation of sibling tasks is left as plain `CancelledError` and silently torn
down instead). `GraphDrained` is raised when `RunControl.request_drain()`
completes cooperatively (checkpoint already saved — the run can be resumed with
the same `thread_id`).

```
Exception
├── GraphBubbleUp                     — internal signalling; never catch in node code
│   ├── GraphInterrupt                — raised by interrupt()
│   ├── ParentCommand                 — Command.PARENT bubbling through a subgraph
│   └── GraphDrained                  — cooperative drain completed; resumable
├── GraphRecursionError(RecursionError)
├── InvalidUpdateError                — concurrent conflicting channel write / bad node return
├── EmptyInputError                   — graph invoked with empty input
├── TaskNotFound                      — distributed-mode task lookup failure
├── NodeCancelledError
└── NodeTimeoutError
```

```python
class NodeError:                      # frozen dataclass, not an Exception
    node: str; error: BaseException
class NodeTimeoutError(Exception):    # NOT a TimeoutError subclass
    node: str; kind: Literal["idle", "run"]; elapsed: float
    run_timeout: float | None; idle_timeout: float | None
class GraphDrained(Exception):        # (technically GraphBubbleUp)
    reason: str
```

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError, GraphRecursionError, GraphDrained

class State(TypedDict):
    value: int; error_info: str

def risky(state: State) -> dict:
    if state["value"] < 0:
        raise ValueError(f"negative: {state['value']}")
    return {"value": state["value"] * 2}

def handler(state: State, error: NodeError) -> dict:
    return {"value": 0, "error_info": f"{error.node} failed: {error.error}"}

g = StateGraph(State)
g.add_node("risky", risky, error_handler=handler)
g.add_edge(START, "risky"); g.add_edge("risky", END)
print(g.compile().invoke({"value": -1, "error_info": ""}))
# {'value': 0, 'error_info': 'risky failed: negative: -1'}

try:
    graph.invoke({"counter": 0}, config={"recursion_limit": 5})
except GraphRecursionError:
    print("hit recursion_limit — add a termination condition or raise the limit")
except GraphDrained as e:
    print(f"drained cooperatively: {e.reason} — resume with the same thread_id")
```

#### `GraphRecursionError`, `InvalidUpdateError`, `EmptyInputError`, `ErrorCode`

**Module:** `langgraph.errors`. `ErrorCode` is an `Enum` with exactly 5 members,
verified: `GRAPH_RECURSION_LIMIT`, `INVALID_CONCURRENT_GRAPH_UPDATE`,
`INVALID_GRAPH_NODE_RETURN_VALUE`, `MULTIPLE_SUBGRAPHS`, `INVALID_CHAT_HISTORY` —
embedded in exception messages / server troubleshooting URLs, but the OSS
exception classes don't expose it as an attribute, so branch on exception *type*,
not on an error code.

- `GraphRecursionError` is a `RecursionError` subclass, raised when the Pregel
  loop hits `config["recursion_limit"]` (default `DEFAULT_RECURSION_LIMIT`, 10007
  in this version — overridable via env `LANGGRAPH_DEFAULT_RECURSION_LIMIT` or
  per-call via `config={"recursion_limit": N}`). Fix by raising the limit, or
  short-circuiting with a `RemainingSteps`/`IsLastStep` managed value.
- `InvalidUpdateError` fires when a channel that only accepts one writer per step
  (`LastValue`, or a second `Overwrite` in the same step) receives two or more
  concurrent writes. It carries no structured `.error_code` attribute — the
  `ErrorCode` string is embedded in the message only.
- `EmptyInputError` fires on the first invocation of a thread when no state and
  no prior checkpoint exists to seed it.

```python
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START

builder = StateGraph(dict)
builder.add_node("loop", lambda s: {"n": s.get("n", 0) + 1})
builder.add_edge(START, "loop"); builder.add_edge("loop", "loop")
graph = builder.compile()
try:
    graph.invoke({}, config={"recursion_limit": 5})
except GraphRecursionError as e:
    print(f"Caught: {e}")
```

#### `GraphDrained`, `RunControl`

**Module:** `langgraph.errors` / `langgraph.runtime`. Cooperative-shutdown
primitives — distinct from a hard kill.

```python
class RunControl:
    __slots__ = ("_drain_reason",)
    def request_drain(self, reason: str = "shutdown") -> None: ...   # single attribute write; thread-safe with no lock
    @property
    def drain_requested(self) -> bool: ...
    @property
    def drain_reason(self) -> str | None: ...

class GraphDrained(GraphBubbleUp):
    def __init__(self, reason: str = "shutdown") -> None: ...
```

Pass your own `RunControl` via `graph.invoke(input, control=my_control)` (or to
`stream()`/`stream_events()`) so an external SIGTERM handler / background thread
can call `my_control.request_drain(reason)`; without an externally-supplied
`control=`, the executor's internal `RunControl` isn't reachable from outside.
Inside a node, check `runtime.drain_requested` / `runtime.drain_reason` (both
delegate to `runtime.control`) and return normally — do **not** raise
`GraphDrained` yourself; the engine checks the flag at the next super-step
boundary (after your node's writes are committed) and raises it there so the
checkpoint is safely saved first. `GraphDrained` is a `GraphBubbleUp` — like
`GraphInterrupt`, it is caught by the Pregel engine on the way out, not a bug
report; catch it at the call site to detect a graceful stop and resume the
thread later with `graph.invoke(None, config=...)`.

```python
import threading
from langgraph.runtime import RunControl

control = RunControl()
threading.Thread(target=lambda: control.request_drain("SIGTERM"), daemon=True).start()
# result = graph.invoke(inputs, config, control=control)
```

**Internals (private, unverified-stability):** `_TimedAttemptScope` / `_AttemptContext`
/ `_AttemptEvent` (`langgraph.pregel._retry`) implement the timeout-enforcement
boundary and retry-lifecycle event objects that `TimeoutPolicy`/`RetryPolicy` compile
down to.
---

### Human-in-the-loop & Interrupts

#### `interrupt()` + `Interrupt`

**Module:** `langgraph.types`

Pauses the current node, surfaces `value` to the caller, and waits for
`Command(resume=...)`. With a checkpointer attached, `invoke()`/`stream()` return
normally (the interrupt does not propagate to the caller as an exception) and the
pending payload is visible via `graph.get_state(config).interrupts` or the
`"__interrupt__"` key of the returned dict.

```python
def interrupt(value: Any) -> Any: ...

@dataclass(frozen=True, slots=True)
class Interrupt:
    value: Any
    id: str    # derived from an xxh3_128 hash of the checkpoint namespace — stable per node path/run
```

- On resume, the **entire node re-runs from the top**; `interrupt()` matches
  resume values by the call-order index within that node execution (or by `id`
  when resuming via a mapping), so multiple sequential `interrupt()` calls in one
  node each occupy their own slot and are answered one at a time across
  successive resumes, and already-resolved calls return immediately without
  re-pausing. Put side-effects that must run exactly once inside `@task`
  functions (memoised, skipped on replay), not directly above an `interrupt()`
  call.
- `Command(resume=value)` answers the next pending interrupt with `value`;
  `Command(resume={interrupt_id: value, ...})` answers **specific** interrupts by
  `.id` — required when several parallel tasks (e.g. `Send`-dispatched or
  `@task`-dispatched) each hold their own interrupt simultaneously.
- Works inside `@task` functions too, provided a checkpointer is attached.
  `NodeInterrupt` (the pre-1.0 mechanism) is fully removed/deprecated; use
  `interrupt()`.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    draft: str; approved: bool

def review(state: State) -> dict:
    decision = interrupt({"question": "Approve?", "draft": state["draft"]})
    return {"approved": decision == "approve"}

graph = StateGraph(State).add_node("review", review).add_edge(START, "review").add_edge("review", END).compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}

graph.invoke({"draft": "hello", "approved": False}, config)  # pauses
final = graph.invoke(Command(resume="approve"), config)      # resumes
print(final["approved"])  # True

# Resuming a specific interrupt by id (needed when multiple are pending at once)
snap = graph.get_state(config)
first_id = snap.interrupts[0].id if snap.interrupts else None
# graph.invoke(Command(resume={first_id: "approve"}), config)
```

`Command(resume=...)` is the only field needed to answer a pending `interrupt()`
— see the Prebuilt Nodes, Command & Send section for the full `Command`
reference; it can be combined with `update=`/`goto=` in the same `Command` if the
resuming node also needs to patch state or route explicitly.

#### `HumanResponse` (+ deprecated `HumanInterrupt` / `ActionRequest` / `HumanInterruptConfig`)

**Module:** `langgraph.prebuilt.interrupt` (moved to `langchain.agents.interrupt`
as of LangGraph 1.0; the `langgraph.prebuilt.interrupt` re-exports still work but
each emits `LangGraphDeprecatedSinceV10` — the `langchain` package itself isn't
installed in the verification venv, so the `langchain.agents.interrupt` import
path is confirmed only via the deprecation shim's pointer, not independently
re-verified). `HumanResponse` is the conventional shape a human-review resume
value takes; it remains a stable, non-deprecated `TypedDict`.

A structured HITL contract standardising what a human operator can do with a
paused graph: accept as-is, ignore/skip, send free-text feedback, or edit the
proposed action, and a matching response shape the node reads back after resume.

```python
class ActionRequest(TypedDict):
    action: str; args: dict
class HumanInterruptConfig(TypedDict):
    allow_ignore: bool; allow_respond: bool; allow_edit: bool; allow_accept: bool
class HumanInterrupt(TypedDict):
    action_request: ActionRequest; config: HumanInterruptConfig; description: str | None
class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
```

| `type` | `args` |
|---|---|
| `"accept"` / `"ignore"` | `None` |
| `"response"` | `str` (free-text feedback) |
| `"edit"` | `ActionRequest` (human-modified action/args) |

```python
from langgraph.types import interrupt, Command
try:
    from langchain.agents.interrupt import ActionRequest, HumanInterrupt, HumanInterruptConfig, HumanResponse
except ImportError:
    from langgraph.prebuilt.interrupt import ActionRequest, HumanInterrupt, HumanInterruptConfig, HumanResponse  # deprecated fallback

def request_approval(state) -> dict:
    request: HumanInterrupt = {
        "action_request": {"action": "run_shell", "args": {"cmd": state["command"]}},
        "config": {"allow_ignore": True, "allow_respond": True, "allow_edit": True, "allow_accept": True},
        "description": f"Approve running `{state['command']}`?",
    }
    response: HumanResponse = interrupt(request)
    return {"approved": response["type"] == "accept"}

# Resume: graph.invoke(Command(resume=HumanResponse(type="accept", args=None)), config)
```

---

### Functional API

#### `entrypoint`, `entrypoint.final`

**Module:** `langgraph.func`. `@entrypoint(...)` compiles a plain function into a
full `Pregel` graph — the decorated function must take exactly one positional
input parameter plus optional injectable keyword parameters (`config`,
`previous`, `runtime`). Generator functions raise `NotImplementedError`.

```python
class entrypoint(Generic[ContextT]):
    def __init__(self, checkpointer: BaseCheckpointSaver | None = None,
                 store: BaseStore | None = None, cache: BaseCache | None = None,
                 context_schema: type[ContextT] | None = None,
                 cache_policy: CachePolicy | None = None,
                 retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
                 timeout: float | timedelta | TimeoutPolicy | None = None) -> None: ...

    class final(Generic[R, S]):
        def __init__(self, *, value: R, save: S) -> None: ...
```

- `previous` is populated from the last **saved** value on the same `thread_id`
  (via a `LastValue` channel keyed `PREVIOUS`); on the first call it's `MISSING`
  and your function's own default (typically `None`) is used.
- `context_schema` (the modern name for the deprecated `config_schema`) types the
  `runtime.context` passed via the `context=` kwarg to `invoke()`/`stream()` —
  distinct from `config["configurable"]`, which only carries
  runnable/checkpoint settings like `thread_id`.
- `entrypoint.final(value=..., save=...)` decouples what the **caller** receives
  (`value`) from what gets written to the `previous` channel for the **next**
  call (`save`) — e.g. return a human-readable summary while persisting the full
  raw history.

```python
from typing import Optional
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: Optional[int] = None) -> entrypoint.final[str, int]:
    current = (previous or 0) + increment
    return entrypoint.final(value=f"Counter is now {current}", save=current)

config = {"configurable": {"thread_id": "cnt-1"}}
print(counter.invoke(5, config))   # Counter is now 5
print(counter.invoke(3, config))   # Counter is now 8  (previous=5 restored from checkpoint)
```

#### `task` + `SyncAsyncFuture`

**Module:** `langgraph.func`. `@task` wraps a sync or async callable so it runs as
an independently tracked, checkpointed Pregel sub-task when called from inside an
`entrypoint` (or a `StateGraph` node).

```python
def task(__func_or_none__=None, *, name: str | None = None,
          retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
          cache_policy: CachePolicy[Callable[..., str | bytes]] | None = None,
          timeout: float | timedelta | TimeoutPolicy | None = None) -> _TaskFunction: ...
```

Calling the decorated function returns a `SyncAsyncFuture[T]` — a
`concurrent.futures.Future` subclass whose `__await__` also makes it directly
`await`-able in async code. It supports `.result()` in sync contexts and
`await fut` individually in async contexts, but is **not** compatible with
`asyncio.gather()` (its `__await__` yields a scheduler sentinel meant for
LangGraph's own loop, which `gather()` rejects with `RuntimeError: Task got bad
yield`) — collect multiple futures with `[await f for f in futures]` or
`[f.result() for f in futures]`, not `gather`. Only `async` tasks support
`timeout=`; results are checkpointed, so resuming after an interrupt re-plays
already-completed task results instead of re-executing them.

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def square(n: int) -> int:
    return n * n

@entrypoint(checkpointer=InMemorySaver())
def compute(numbers: list[int]) -> list[int]:
    futures = [square(n) for n in numbers]     # dispatched in parallel
    return [f.result() for f in futures]

print(compute.invoke([1, 2, 3, 4], {"configurable": {"thread_id": "t1"}}))  # [1, 4, 9, 16]
```

#### `call()` + `SyncAsyncFuture`

**Modules:** `langgraph.pregel._call` (implementation); public re-export
`langgraph.types.call`

The low-level primitive `@task` compiles down to: `call(fn, *args, retry_policy=...,
cache_policy=..., timeout=..., **kwargs)` dispatches `fn` as a sub-task with its own
per-call policies (overriding the `@task` decorator's defaults for that one
invocation) and returns a `SyncAsyncFuture` — usable with `.result()` or `await`. Only
works inside an active Pregel execution (an `@entrypoint`/`@task`/node) — calling it
elsewhere raises a `KeyError`. `timeout` is async-only; passing it with a sync `func`
raises `NotImplementedError`.

```python
def call(func: Callable[..., T], *args,
         retry_policy: Sequence[RetryPolicy] | None = None,
         cache_policy: CachePolicy | None = None,
         timeout: float | timedelta | TimeoutPolicy | None = None, **kwargs) -> SyncAsyncFuture[T]
```

```python
from langgraph.types import call, RetryPolicy
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
import asyncio

@entrypoint(checkpointer=InMemorySaver())
async def pipeline(prompts: list[str]) -> list[str]:
    futures = [call(call_llm, p, retry_policy=[RetryPolicy(max_attempts=3)], timeout=30.0) for p in prompts]
    return list(await asyncio.gather(*futures))
```

**Internals (private, unverified-stability):** `PregelScratchpad`
(`langgraph._internal._scratchpad`) is the per-superstep execution context (`step`,
`stop`, `call_counter`, `interrupt_counter`, `resume`, `subgraph_counter`) that
`IsLastStep`/`RemainingSteps` and `interrupt()`'s resume matching read from;
`FunctionNonLocals`/`NonLocals` (`langgraph.pregel._utils`) do AST-based closure
analysis to detect which outer-scope names a `@task`/`@entrypoint` function captures;
`identifier`/`get_runnable_for_task`/`get_runnable_for_entrypoint`
(`langgraph.pregel._call`) resolve a callable's stable `module.qualname` for caching —
lambdas have no stable identifier and are therefore rebuilt (not cached) on every
compile.
---

### Runtime & Managed Values

#### `Runtime`, `ExecutionInfo`, `ServerInfo`, `BaseUser`, `get_runtime()`

**Module:** `langgraph.runtime`. `Runtime[ContextT]` is the convenience bundle
injected into any **node** (not tools — use `ToolRuntime` there) that declares a
`runtime: Runtime` parameter.

```python
@dataclass(frozen=True)
class Runtime(Generic[ContextT]):
    context: ContextT | None
    store: BaseStore | None
    stream_writer: StreamWriter
    heartbeat: Callable[[], None]
    previous: Any
    execution_info: ExecutionInfo | None
    server_info: ServerInfo | None
    control: RunControl | None

    def merge(self, other: "Runtime") -> "Runtime": ...      # other's non-default values win
    def override(self, **overrides) -> "Runtime": ...        # dataclasses.replace, for tests
    def patch_execution_info(self, **overrides) -> "Runtime": ...  # raises if execution_info is None
    @property
    def drain_requested(self) -> bool: ...                   # delegates to self.control

@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    checkpoint_id: str; checkpoint_ns: str; task_id: str
    thread_id: str | None; run_id: str | None
    node_attempt: int                     # 1-indexed; increments per retry
    node_first_attempt_time: float | None # fixed at the first attempt; unchanged on retries
    def patch(self, **overrides) -> "ExecutionInfo": ...

@dataclass(frozen=True, slots=True)
class ServerInfo:
    assistant_id: str; graph_id: str; user: BaseUser | None   # all None on open-source/local runs

def get_runtime(context_schema=None) -> Runtime: ...   # reads the active Runtime from the config ContextVar
```

`heartbeat` resets the idle-timeout clock — safe to call unconditionally, a no-op
outside an idle-timed attempt. `execution_info.task_id` is stable across retries —
use it as an idempotency key for external calls. `server_info` (`assistant_id`,
`graph_id`, `user: BaseUser | None`) is populated only when running inside
LangGraph Platform — always `None` in open-source/local runs; guard with `if
runtime.server_info is not None`. `BaseUser` is re-exported from
`langgraph_sdk.auth.types` (a protocol supporting both `user.identity` and
`user["identity"]` access), not defined in `langgraph` itself. `get_runtime()` is
an alternative to parameter injection for reading runtime state from a helper
function nested deep inside a node — functionally equivalent to injecting
`runtime: Runtime` directly, and only usable inside an active graph run.

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

@dataclass
class UserContext:
    user_id: str

class State(TypedDict):
    query: str; result: str

def node(state: State, runtime: Runtime[UserContext]) -> dict:
    info = runtime.execution_info
    return {"result": f"[{runtime.context.user_id}] attempt={info.node_attempt if info else 1}: {state['query']}"}

graph = StateGraph(State, context_schema=UserContext).add_node("n", node).add_edge(START, "n").add_edge("n", END).compile()
result = graph.invoke({"query": "hi", "result": ""}, context=UserContext(user_id="alice"))
```

#### `RunControl`

**Module:** `langgraph.runtime`

Cooperative graceful-shutdown signal (e.g. wired to `SIGTERM`). `request_drain(reason)`
sets a flag the loop checks at superstep boundaries; the run exits via `GraphDrained`
(see Error Handling) once the current superstep finishes — no abrupt cancellation, and
a checkpoint is saved so the run can resume later with the same `thread_id`.

```python
class RunControl:
    def request_drain(self, reason: str = "shutdown") -> None: ...
    @property
    def drain_requested(self) -> bool: ...
```

```python
from langgraph.runtime import RunControl
control = RunControl()
# elsewhere: control.request_drain(reason="SIGTERM")
# graph.invoke(input, config, control=control)  # raises GraphDrained at the next boundary
```

#### `ManagedValue` + `IsLastStep` / `RemainingSteps`

**Modules:** `langgraph.managed.base` (`ManagedValue`), `langgraph.managed.is_last_step`
(built-ins, also re-exported from `langgraph.managed`)

Managed values are scratchpad-derived fields the Pregel runtime injects into a
node's state **every step**, rather than storing them in a channel — a node
cannot write to one, and they never appear in checkpoint blobs.

```python
class ManagedValue(ABC, Generic[V]):
    @staticmethod
    @abstractmethod
    def get(scratchpad: PregelScratchpad) -> V: ...   # must be a @staticmethod; the class itself is the "spec"

IsLastStep = Annotated[bool, IsLastStepManager]         # True exactly when step == stop - 1
RemainingSteps = Annotated[int, RemainingStepsManager]  # stop - step, counting the current step
```

`PregelScratchpad.step` starts at 0 and increments per completed super-step;
`.stop` is the absolute step cutoff derived from `recursion_limit` (offset by any
prior checkpoint's step count on resume) — always compute remaining budget as
`stop - step`, never from the raw `config["recursion_limit"]` directly, since
`.stop` already accounts for where a resumed run picked up. `IsLastStep` becomes
`True` exactly once, giving a one-step warning window before `GraphRecursionError`
would fire on the next step. Writing a custom managed value just means subclassing
`ManagedValue[V]` with a `get(scratchpad) -> V` static method.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps

class State(TypedDict):
    value: int; is_last: IsLastStep; remaining_steps: RemainingSteps

def guarded(state: State) -> dict:
    if state["is_last"]:
        return {"value": state["value"]}   # bail out cleanly before the recursion limit hits
    return {"value": state["value"] + 1}

builder = StateGraph(State)
builder.add_node("work", guarded); builder.add_edge(START, "work")
builder.add_conditional_edges("work", lambda s: END if s["is_last"] else "work")
```

---

### Observability & Tracing

#### `TracePolicy`

**Module:** `langgraph.types`. A frozen dataclass, verified fields:
`(process_inputs, process_outputs)`.

```python
@dataclass(frozen=True)
class TracePolicy:
    process_inputs: Callable[[Any], Any] | None = None
    process_outputs: Callable[[Any], Any] | None = None
```

Attach via `add_node(..., trace_policy=TracePolicy(...))` to transform what a
node's own LangSmith trace span records — summarize a long message history,
redact a field — **without** affecting the actual data flowing through the graph.
Scope is limited to that node's own run span; the root graph run and any child
runs created by the bound runnable still see the original payload. This is a
sanitization/summarization convenience, not a secrets-redaction guarantee — use
LangSmith's `hide_inputs`/`hide_outputs`/anonymizer for that.

```python
from typing import Any
from langgraph.types import TracePolicy

def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("***" if k == "api_key" else v) for k, v in value.items()}
    return value

# builder.add_node("fetch", fetch_data, trace_policy=TracePolicy(process_inputs=redact, process_outputs=redact))
```

#### `GraphCallbackHandler`, `GraphInterruptEvent`, `GraphResumeEvent`

**Module:** `langgraph.callbacks`. `GraphCallbackHandler` extends
`langchain_core.callbacks.BaseCallbackHandler` with two graph-only lifecycle
hooks that a generic LangChain callback handler cannot observe.

```python
class GraphCallbackHandler(BaseCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> Any: ...   # default no-op
    def on_resume(self, event: GraphResumeEvent) -> Any: ...          # default no-op

@dataclass(frozen=True)
class GraphInterruptEvent:
    run_id: UUID | None; status: GraphLifecycleStatus
    checkpoint_id: str; checkpoint_ns: tuple[str, ...]
    interrupts: tuple[Interrupt, ...]

@dataclass(frozen=True)
class GraphResumeEvent:
    run_id: UUID | None; status: GraphLifecycleStatus
    checkpoint_id: str; checkpoint_ns: tuple[str, ...]

GraphLifecycleStatus = Literal["input", "pending", "done", "interrupt_before", "interrupt_after", "out_of_steps"]
```

Register instances via `config["callbacks"]`; an internal manager
(`_GraphCallbackManager` / `_AsyncGraphCallbackManager`) filters the callback stack
down to `GraphCallbackHandler` subclasses before dispatching these two events, so a
plain `BaseCallbackHandler` is silently skipped. Both methods may be `async def`
for async runs. `checkpoint_ns` lets a single handler distinguish root-graph
pauses from subgraph pauses by namespace depth.

```python
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent

class AuditHandler(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        print(f"paused at {event.checkpoint_id[:8]} ns={event.checkpoint_ns}")
    def on_resume(self, event: GraphResumeEvent) -> None:
        print(f"resumed at {event.checkpoint_id[:8]}")

# graph.invoke(inputs, config={"configurable": {...}, "callbacks": [AuditHandler()]})
```

#### `LangGraphDeprecationWarning` + subclasses

**Module:** `langgraph.warnings`

The deprecation-warning hierarchy every deprecated LangGraph API (`MessageGraph`,
`ValidationNode`, `create_react_agent`, `AgentState`, the `langgraph.prebuilt.interrupt`
re-exports, `GraphOutput` dict-access, …) emits — each subclass records `since` and
`expected_removal` as `(major, minor)` tuples, letting you filter by version range in
tests (`pytest.warns(LangGraphDeprecatedSinceV10)`) or promote them to errors
(`warnings.filterwarnings("error", category=LangGraphDeprecatedSinceV10)`) to assert a
codebase avoids a given deprecation window.

```python
class LangGraphDeprecationWarning(DeprecationWarning):
    def __init__(self, message, *, since: tuple[int, int], expected_removal: tuple[int, int] | None = None): ...
class LangGraphDeprecatedSinceV10(LangGraphDeprecationWarning): ...   # since=(1,0), removal=(2,0)
class LangGraphDeprecatedSinceV11(LangGraphDeprecationWarning): ...   # since=(1,1), removal=(3,0)
```

```python
import warnings, pytest
from langgraph.warnings import LangGraphDeprecatedSinceV10

def test_message_graph_emits_deprecation():
    with pytest.warns(LangGraphDeprecatedSinceV10):
        from langgraph.graph.message import MessageGraph
        MessageGraph()
```

**Internals (private, unverified-stability):** `StreamMessagesHandler` /
`StreamMessagesHandlerV2` (`langgraph.pregel._messages`, also listed under
Streaming & Transformers) are the concrete callback handlers powering
`stream_mode="messages"`; `_GraphCallbackManager` / `_AsyncGraphCallbackManager`
(`langgraph.callbacks`) are what actually calls `on_interrupt`/`on_resume` on
registered `GraphCallbackHandler` instances, built from `config["callbacks"]` via
`.configure()`.
---

### Prebuilt Nodes, Command & Send

#### `Command`

**Module:** `langgraph.types`

The universal node return type that simultaneously **updates state**, **routes**
(`goto`, replacing a static edge), **resumes an interrupt** (`resume`), and can
target the **parent graph** (`graph=Command.PARENT`, only valid from inside a
subgraph node — using it at the top level raises via `GraphBubbleUp`/
`ParentCommand`). `goto` accepts a single node name, a list mixing node names and
`Send` objects (parallel fan-out), or `END`.

```python
@dataclass
class Command(Generic[N], ToolOutputMixin):
    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
    graph: str | None = None            # None = this graph; Command.PARENT = parent
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

- `goto=` is **additive** with any static `add_edge` already leaving that node —
  both fire. For fully dynamic routing, don't also declare a static outgoing edge
  from the node.
- `graph=Command.PARENT` sends the update/goto to the **nearest enclosing parent
  graph** — the standard way for a subgraph node to write into parent state or
  route the parent; using it at the root graph raises (there is no parent to
  target).
- `update` accepts a plain dict, a list of `(key, value)` pairs, or a Pydantic
  model/dataclass instance (annotated-key extraction) — the same shapes a normal
  node return value accepts.

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from typing_extensions import TypedDict

class State(TypedDict):
    score: int; tier: str

def router(state: State) -> Command[Literal["premium", "standard"]]:
    tier = "premium" if state["score"] >= 90 else "standard"
    return Command(update={"tier": tier}, goto=tier)

def premium(state: State) -> dict:
    return {}
def standard(state: State) -> dict:
    return {}

builder = StateGraph(State)
builder.add_node("router", router); builder.add_node("premium", premium); builder.add_node("standard", standard)
builder.add_edge(START, "router"); builder.add_edge("premium", END); builder.add_edge("standard", END)
graph = builder.compile()

# Cross-subgraph: a node inside a subgraph escalating to its parent
def escalate(state: dict) -> Command:
    return Command(graph=Command.PARENT, update={"escalation_reason": "budget exceeded"}, goto="approval_node")

# Fan-out via Send objects in goto
def fan_out(state: dict) -> Command:
    return Command(goto=[Send("worker", {"item": i}) for i in state.get("items", [])])
```

#### `Send`

**Module:** `langgraph.types`

Routes execution to a named node with a **specific input**, bypassing shared state
— enables dynamic map-reduce fan-out where the branch count isn't known at
graph-build time. `Send` also accepts a per-task `timeout` (coerced through
`TimeoutPolicy.coerce()`), overriding the target node's default timeout for that
one fanned-out invocation.

```python
class Send:
    __slots__ = ("node", "arg", "timeout")
    def __init__(self, node: str, arg: Any, *, timeout: float | timedelta | TimeoutPolicy | None = None) -> None: ...
```

- `arg` becomes the full input to the target node (dict, Pydantic model,
  anything the node accepts). Results merge back into the parent state via each
  field's reducer — typically `Annotated[list[T], operator.add]` (or `Topic(T)`
  when each worker contributes a single scalar item rather than a list).
- `Send` is hashable and compares structurally on `(node, arg, timeout)` **only
  if `arg` is hashable** — passing a plain `dict` (the common case) makes that
  `Send` instance unhashable; don't put such `Send` objects in a set/dict key.
- Per-`Send` `timeout=` requires the target node to be `async`, like all node
  timeouts.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class MapState(TypedDict):
    urls: list[str]; results: Annotated[list[str], operator.add]

def distribute(state: MapState) -> list[Send]:
    return [Send("scrape", {"url": u, "result": ""}, timeout=10.0) for u in state["urls"]]

def scrape(state: dict) -> dict:
    return {"results": [f"content of {state['url']}"]}

builder = StateGraph(MapState)
builder.add_node("scrape", scrape)
builder.add_conditional_edges(START, distribute)
builder.add_edge("scrape", END)
result = builder.compile().invoke({"urls": ["a.com", "b.com"], "results": []})
```

#### `RemoteGraph` + `RemoteException` + `get_client` / `get_sync_client`

**Module:** `langgraph.pregel.remote`

A `PregelProtocol` implementation wrapping any LangGraph Server-compatible HTTP
API (LangSmith deployment, self-hosted `langgraph-cli` server, or another graph in
the same process via ASGI loopback) so it behaves exactly like a local compiled
graph — pass it to `add_node()` as a subgraph, or call
`invoke`/`stream`/`get_state`/`update_state`/`get_state_history`/`get_graph`
directly. `api_key` falls back to `LANGGRAPH_API_KEY` / `LANGSMITH_API_KEY` /
`LANGCHAIN_API_KEY`; at least one of `url`, `client`, or `sync_client` is
required. `RemoteException` is a plain `Exception` raised when the remote server
returns an error response — catch it to distinguish remote failures from local
ones.

```python
class RemoteGraph(PregelProtocol):
    def __init__(self, assistant_id: str, /, *, url: str | None = None, api_key: str | None = None,
                 headers: dict[str, str] | None = None, client=None, sync_client=None,
                 config: RunnableConfig | None = None, name: str | None = None,
                 distributed_tracing: bool = False): ...

def get_client(*, url: str | None = None, api_key: str | ... = NOT_PROVIDED,
               headers=None, timeout=None) -> LangGraphClient: ...
def get_sync_client(*, url=None, ...) -> SyncLangGraphClient: ...
```

`get_client(url=None)` attempts an in-process ASGI loopback (no network hop) when
no `url` is given; the `api_key` sentinel (`NOT_PROVIDED`, distinct from `None`)
triggers auto-loading from `LANGGRAPH_API_KEY` → `LANGSMITH_API_KEY` →
`LANGCHAIN_API_KEY` — pass `api_key=None` explicitly to skip that lookup entirely
(e.g. for local dev with no auth). `distributed_tracing=True` propagates LangSmith
`x-parent-*` headers so the remote run's trace links back to the parent's.

```python
from langgraph.pregel.remote import RemoteGraph, RemoteException

remote = RemoteGraph("my_agent", url="http://localhost:2024", api_key="local-key")
try:
    result = remote.invoke({"messages": [{"role": "user", "content": "hi"}]},
                            config={"configurable": {"thread_id": "t1"}})
except RemoteException as e:
    print(f"remote graph failed: {e}")

# Embed a remote deployment as a node in a local orchestration graph:
builder.add_node("research", remote)
```

**Internals (private, unverified-stability):** `PregelProtocol` + `StreamProtocol`
(`langgraph.pregel.protocol`) are the abstract executor interface both `Pregel` and
`RemoteGraph` implement and the slim `(modes, __call__)` stream-mode-filter struct
behind them, respectively — annotate a parameter as `PregelProtocol` to write code
that works with local or remote graphs interchangeably. `BackgroundExecutor` /
`AsyncBackgroundExecutor` / `Submit` (`langgraph.pregel._executor`) run parallel node
execution (thread pool for sync, asyncio tasks for async); `PregelRunner` /
`FuturesDict` (`langgraph.pregel._runner`), `SyncPregelLoop` / `AsyncPregelLoop` /
`DuplexStream` (`langgraph.pregel._loop`), and `WritesProtocol` / `PregelTaskWrites`
(`langgraph.pregel._algo`) implement the per-superstep task-scheduling/write-commit
machinery underneath every `invoke()`/`stream()` call.

---

### Corrections vs. older third-party write-ups

A short list of symbols and signatures that differ from what circulates in older
blog posts, outdated docs, or earlier drafts of this reference — the installed
`langgraph==1.2.11` source is the tiebreaker throughout this section:

- **`ToolOutputMixin` at `langgraph.prebuilt.tool_node`** — never actually lived there;
  it's `langchain_core.messages.tool.ToolOutputMixin`, re-exported (unofficially) via
  `langgraph.types`. Not removed, just mis-located in some older docs.
- **`NodeBuilder.with_retry_policy()` / `.with_cache_policy()` / `.with_timeout()` /
  `.with_tags()` / `.with_metadata()`** — these method names never existed in the
  installed version; the real API is `add_retry_policies()`, `add_cache_policy()`,
  `set_timeout()`, and a combined `meta(*tags, **metadata)`.
- **`add_node(..., retry=...)`** — several older examples use a `retry=` kwarg; the
  real parameter name has always been `retry_policy=`.
- **`ToolNode(handle_tool_errors=True)` as a literal default** — the real default
  value is the function `_default_handle_tool_errors`, not the literal `True`
  (see Tools & Tool Calling above).
- **`from langgraph.prebuilt import AgentState`** — raises `ImportError` in
  1.2.11; it was never re-exported from the `langgraph.prebuilt` package
  `__init__`. Import from `langgraph.prebuilt.chat_agent_executor` instead.
- **`SqliteSaver.from_conn_string(...)` / `PostgresSaver.from_conn_string(...)`**
  — these are **context managers** (`with ... as saver:`), not factories that
  return a saver directly.
- **`Interrupt.ns`, `Interrupt.when`, `Interrupt.resumable`** — removed; only
  `Interrupt.value` and `Interrupt.id` remain on the dataclass.
- **`checkpoint_during=False`** on `invoke`/`stream` — deprecated in favor of
  `durability="exit"` (or `"sync"` / `"async"`).
- **`ShallowPostgresSaver`** — deprecated as of `langgraph-checkpoint-postgres`
  2.0.20; use `PostgresSaver` + `durability="exit"`.

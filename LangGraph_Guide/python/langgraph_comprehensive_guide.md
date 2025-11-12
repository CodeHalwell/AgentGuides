Latest: 1.0.3
# LangGraph: Comprehensive Technical Guide (Beginner to Expert)

**Latest Version**: LangGraph 1.0+  
**Focus**: Python Examples with practical, production-ready patterns  
**Author Note**: This guide progresses from fundamentals through advanced multi-agent architectures with real-world workflows.

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
9. [Human-in-the-Loop](#human-in-the-loop)
10. [Advanced Patterns](#advanced-patterns)

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

{% raw %}
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
{% endraw %}

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

{% raw %}
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
{% endraw %}

---

## Simple Agents

### Example 1: Linear Chat Pipeline

A basic chatbot with no branching:

{% raw %}
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
{% endraw %}

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

{% raw %}
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
{% endraw %}

### Example 4: Streaming Output

See the graph execute step-by-step:

{% raw %}
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
{% endraw %}

---

## Multi-Agent Systems

### Example 1: Supervisor Pattern

One coordinator agent routing to specialists:

{% raw %}
```python
from langchain_core.messages import BaseMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
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
{% endraw %}

### Example 2: Parallel Worker Pattern

Fan-out to multiple workers, collect results:

{% raw %}
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
{% endraw %}

### Example 3: Handoff Pattern

Agents handing off to each other mid-conversation:

{% raw %}
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
{% endraw %}

---

## Tool Integration

### Example 1: Basic Tool Node

Using LangGraph's built-in `ToolNode`:

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

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

{% raw %}
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
{% endraw %}

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

{% raw %}
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
{% endraw %}

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

{% raw %}
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
{% endraw %}

### Getting State at Any Point

{% raw %}
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
{% endraw %}

### Checkpoint Inspection

{% raw %}
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
{% endraw %}

### Batch Debugging

{% raw %}
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
{% endraw %}

---

## Human-in-the-Loop

### Basic Interrupts

Pause execution and request human input:

{% raw %}
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
{% endraw %}

### Multi-Step Approval Workflow

{% raw %}
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
{% endraw %}

### Interactive Debugging

{% raw %}
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
{% endraw %}

---

## Advanced Patterns

### Pattern 1: ReAct (Reasoning + Acting)

The Reflection-Action pattern for autonomous agents, now built with modern LangChain components.

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool

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

{% raw %}
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
{% endraw %}

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

{% raw %}
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
{% endraw %}

---

## Functional API (LangGraph 1.0)

A simpler Python-native way to build workflows with automatic parallelization:

{% raw %}
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
{% endraw %}

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

{% raw %}
```python
config = {"configurable": {"thread_id": "unique-id"}}
result = graph.invoke(input, config=config)  # Same config each time
```
{% endraw %}

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
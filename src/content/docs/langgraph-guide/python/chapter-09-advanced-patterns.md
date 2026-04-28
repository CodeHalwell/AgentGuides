---
title: "Chapter 9 — Advanced Patterns"
description: "ReAct, Tree-of-Thoughts, self-reflection, structured output, node caching, deferred nodes, Command tool, templates, and the Functional API."
framework: langgraph
language: python
sidebar:
  label: "9 · Advanced patterns"
  order: 9
---

# Chapter 9 — Advanced Patterns

**What you'll learn:** the patterns you reach for when simple graphs aren't enough. Reasoning-action loops (ReAct), multi-path exploration (Tree-of-Thoughts), self-critique (Reflection), validation loops (Structured output), plus v1 features — node caching, deferred nodes, the Command tool for edgeless flows, templates, and the declarative Functional API.

**Time:** ~40 minutes. Most of this is reference — skim for patterns you need.

> Prereqs: [Chapter 3 — Multi-agent systems](/langgraph-guide/python/chapter-03-multi-agent/) and [Chapter 4 — Tools](/langgraph-guide/python/chapter-04-tools/).

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

## Further built-in features

Below this point, earlier drafts documented several "v1.0.3+ features" that do not exist in the installed `langgraph==1.1.10` package. The following were removed after verifying against the installed library:

- **Node Caching** — `langgraph.cache.cache_node`, `SemanticCache`, `CachePolicy` are not real. For caching today, use LangGraph's long-term `Store` (see [Chapter 5 — Memory & persistence](/langgraph-guide/python/chapter-05-memory/)) or wrap expensive tool calls with Python's own `functools.lru_cache` / a Redis client (the "Caching and Memoization" pattern above is the real approach).
- **Deferred Nodes** — `@deferred(wait_for=[...])` and `langgraph.graph.deferred` do not exist. Fan-in is already native: an edge from multiple sources into the same target waits for all upstream nodes to complete before that target runs (see [Chapter 3 — Parallel Worker Pattern](/langgraph-guide/python/chapter-03-multi-agent/#example-2-parallel-worker-pattern)).
- **Tools with State Updates** — `@tool(updates_state=True)` accepting a `state` parameter and returning `StateUpdate` is not a real API. To let a tool update graph state, wrap the tool in a node that reads state, calls the tool, and returns the `state` update dict as normal.
- **Command Tool for edgeless flows** — `command_tool` / `CommandRouter` do not exist. The real way to let the agent control routing from a tool call is to use `langgraph.types.Command` as a tool return value; the graph routes based on the `goto` field.
- **LangGraph Templates CLI** — `langgraph template list|create|init|publish` is not real. The actual command is `langgraph new --template NAME` (run `langgraph --help` for the full CLI). A curated template gallery lives at <https://github.com/langchain-ai/langgraph/tree/main/examples>.

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

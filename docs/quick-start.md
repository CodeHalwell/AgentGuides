---
layout: default
title: "Quick Start"
description: "Get started with AI agents in minutes"
---

# 🚀 Quick Start Guide

Get up and running with AI agents in just a few minutes.

---

## Choose Your Path

### 👤 Path 1: Simple Single Agent (5 minutes)

**Use SmolAgents for the fastest start**

```bash
pip install smolagents
```

```python
from smolagents import CodeAgent, tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

agent = CodeAgent(tools=[add_numbers])
result = agent.run("What is 15 + 27?")
print(result)
```

**Next Steps**:
1. Read [SmolAgents README](../SmolAgents_Guide/README.md)
2. Try more [SmolAgents Recipes](../SmolAgents_Guide/smolagents_recipes.md)
3. Deep dive: [SmolAgents Comprehensive Guide](../SmolAgents_Guide/smolagents_comprehensive_guide.md)

---

### 👥 Path 2: Multi-Agent Team (10 minutes)

**Use CrewAI for team-based agents**

```bash
pip install crewai
```

```python
from crewai import Agent, Task, Crew

# Define agents with different roles
analyst = Agent(
    role="Data Analyst",
    goal="Provide insights from data",
    backstory="You are an experienced data analyst"
)

writer = Agent(
    role="Content Writer",
    goal="Write clear reports",
    backstory="You are a professional writer"
)

# Define tasks
analysis_task = Task(
    description="Analyse the Q3 sales data",
    agent=analyst
)

report_task = Task(
    description="Write a summary report",
    agent=writer,
    depends_on=[analysis_task]
)

# Create and run crew
crew = Crew(agents=[analyst, writer], tasks=[analysis_task, report_task])
result = crew.kickoff()
print(result)
```

**Next Steps**:
1. Read [CrewAI README](../CrewAI_Guide/README.md)
2. Try more [CrewAI Recipes](../CrewAI_Guide/crewai_recipes.md)
3. Deep dive: [CrewAI Comprehensive Guide](../CrewAI_Guide/crewai_comprehensive_guide.md)

---

### 🔗 Path 3: Multi-Agent with Handoffs (15 minutes)

**Use OpenAI Agents SDK for powerful coordination**

```bash
pip install openai-agents
export OPENAI_API_KEY=sk-your-key
```

```python
from agents import Agent, Runner

# Create specialised agents
support_agent = Agent(
    name="Support",
    instructions="You are a customer support specialist"
)

technical_agent = Agent(
    name="Technical",
    instructions="You are a technical specialist"
)

# Agents can handoff to each other
support_agent.instructions += "\nHandoff to technical if complex"

# Run
result = await Runner.run(support_agent, "I'm having a technical issue")
print(result.final_output)
```

**Next Steps**:
1. Read [OpenAI Agents SDK README](../OpenAI_Agents_SDK_Guides/README.md)
2. Try more [OpenAI Recipes](../OpenAI_Agents_SDK_Guides/openai_agents_sdk_recipes.md)
3. Deep dive: [OpenAI Comprehensive Guide](../OpenAI_Agents_SDK_Guides/openai_agents_sdk_comprehensive_guide.md)

---

### 📚 Path 4: Knowledge Retrieval (RAG) (15 minutes)

**Use LlamaIndex for RAG systems**

```bash
pip install llama-index
```

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of France?")
print(response)
```

**Next Steps**:
1. Read [LlamaIndex README](../LlamaIndex_Guide/README.md)
2. Try more [LlamaIndex Recipes](../LlamaIndex_Guide/llamaindex_recipes.md)
3. Deep dive: [LlamaIndex Comprehensive Guide](../LlamaIndex_Guide/llamaindex_comprehensive_guide.md)

---

### 🌊 Path 5: Complex Workflows (20 minutes)

**Use LangGraph for sophisticated control flow**

```bash
pip install langgraph langchain-openai
```

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    messages: list[str]
    current_step: str

# Create graph
builder = StateGraph(State)

def step_1(state: State) -> State:
    state["current_step"] = "completed_step_1"
    return state

def step_2(state: State) -> State:
    state["current_step"] = "completed_step_2"
    return state

builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_edge("step_1", "step_2")
builder.set_entry_point("step_1")

graph = builder.compile()
result = graph.invoke({"messages": [], "current_step": "start"})
print(result)
```

**Next Steps**:
1. Read [LangGraph README](../LangGraph_Guide/README.md)
2. Try more [LangGraph Recipes](../LangGraph_Guide/python/langgraph_recipes.md)
3. Deep dive: [LangGraph Comprehensive Guide](../LangGraph_Guide/python/langgraph_comprehensive_guide.md)

---

## Common Tasks

### Add a Tool to Your Agent

```python
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return results

agent = CodeAgent(tools=[search_web])
```

[Learn More →](../OpenAI_Agents_SDK_Guides/openai_agents_sdk_comprehensive_guide.md#tools)

---

### Implement Memory

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Now the graph remembers between runs
```

[Learn More →](../LangGraph_Guide/python/langgraph_comprehensive_guide.md#memory)

---

### Add Error Handling

```python
from smolagents import tool
import logging

@tool
def risky_operation() -> str:
    """An operation that might fail."""
    try:
        return do_something()
    except Exception as e:
        logging.error(f"Operation failed: {e}")
        return "Operation failed, trying alternative approach"
```

[Learn More →](../OpenAI_Agents_SDK_Guides/openai_agents_sdk_production_guide.md#error-handling)

---

### Deploy to Production

```bash
# Containerise your agent
docker build -t my-agent .
docker run -p 8000:8000 my-agent

# Deploy to cloud
aws ecs run-task --cluster my-cluster --task-definition my-agent
```

[Learn More →](../OpenAI_Agents_SDK_Guides/openai_agents_sdk_production_guide.md#deployment)

---

## Comparison: Which Path for You?

| Need | Recommendation | Time | Complexity |
|------|----------------|------|-----------|
| Simple automation | SmolAgents | 5 min | Low |
| Team of agents | CrewAI | 10 min | Medium |
| Flexible coordination | OpenAI Agents SDK | 15 min | Medium |
| RAG/Search | LlamaIndex | 15 min | Medium |
| Complex workflows | LangGraph | 20 min | High |

---

## Next Steps

After your quick start:

1. **Explore Recipes** - Real-world implementations
2. **Read Comprehensive Guide** - Full features and concepts
3. **Study Production Guide** - Deployment and scaling
4. **Review Diagrams** - Architecture and patterns

---

## Common Patterns

### Chain Multiple Agents

```python
# CrewAI
task1 = Task(description="...", agent=agent1)
task2 = Task(description="...", agent=agent2, depends_on=[task1])

# LangGraph
builder.add_edge("agent1_node", "agent2_node")

# OpenAI Agents SDK
agent1 can handoff to agent2
```

---

### Parallel Execution

```python
# CrewAI
tasks = [task1, task2, task3]  # No dependencies = parallel

# LangGraph
builder.add_edge("start", ["agent1", "agent2"])  # Parallel
builder.add_edge(["agent1", "agent2"], "join_node")  # Merge results

# Python
import asyncio
await asyncio.gather(agent1.run(...), agent2.run(...))
```

---

### Conditional Routing

```python
# LangGraph (best support)
def route(state):
    if state["needs_help"]:
        return "escalate"
    else:
        return "resolve"

builder.add_conditional_edges("classify", route)

# OpenAI Agents SDK
if condition:
    agent.handoff_to(specialist_agent)

# CrewAI
# Use task dependencies
```

---

## Troubleshooting

### Agent Isn't Using Tools

```python
# ✅ Correct: Tools are passed
agent = Agent(tools=[my_tool])

# ❌ Wrong: Tools not passed
agent = Agent()
```

[See Tool Guide →](../OpenAI_Agents_SDK_Guides/openai_agents_sdk_comprehensive_guide.md#tools)

---

### Memory Not Working

```python
# ✅ Correct: Checkpointer configured
graph = builder.compile(checkpointer=MemorySaver())

# ❌ Wrong: No checkpointer
graph = builder.compile()
```

[See Memory Guide →](../LangGraph_Guide/python/langgraph_comprehensive_guide.md#memory)

---

### Streaming Not Working

```python
# ✅ Correct: Use streaming API
for event in agent.stream(input):
    print(event)

# ❌ Wrong: Not streaming
output = agent.run(input)
```

[See Streaming Guide →](../OpenAI_Agents_SDK_Guides/openai_agents_sdk_comprehensive_guide.md#streaming)

---

## Resources

- 📖 [All Guides](guides.md)
- 🔄 [Framework Comparison](frameworks.md)
- 📚 [Browse by Use Case](guides.md#-quick-reference)
- 🏠 [Home](index.md)

---

## Get Help

1. **Check the relevant guide** - Comprehensive guides have troubleshooting sections
2. **Search the Recipes** - Your use case is probably covered
3. **Review Production Guide** - For deployment issues
4. **Check official docs** - Links provided in each guide

---

**Ready?** Pick a path above and start building! 🚀


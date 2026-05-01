---
title: "Microsoft Agent Framework Python - Recipes and Code Patterns"
description: "This document provides a collection of practical, copy-paste-ready Python recipes for building common agentic patterns with the Microsoft Agent Framework for Python."
framework: microsoft-agent-framework
language: python
---

# Microsoft Agent Framework Python - Recipes and Code Patterns

All recipes on this page target the real `agent_framework` package (verified against `agent-framework-core==1.2.2`). The primary agent class is `Agent`; chat clients come from `agent_framework.openai`, `agent_framework.foundry`, `agent_framework.anthropic`, etc. The tool decorator is `@tool` from `agent_framework`. Multi-turn state is managed via `agent.create_session()`; workflow orchestration uses `WorkflowBuilder` from `agent_framework`.

This document provides a collection of practical, copy-paste-ready Python recipes for building common agentic patterns with the Microsoft Agent Framework for Python.

**Target Platform:** Python 3.10+
**Framework Version:** 1.0+

---

## Beginner Recipes

These recipes are for developers new to the framework and cover fundamental concepts.

### Recipe 1: Simple Chat Agent (Python)

This is the "Hello, World!" of the Agent Framework — a basic conversational agent that maintains history across turns.

```python
# simple_chat_agent.py
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def run_interactive():
    # Construct the agent directly with a chat client.
    # Swap OpenAIChatClient for FoundryChatClient / AnthropicClient / etc.
    agent = Agent(
        client=OpenAIChatClient(),  # reads OPENAI_API_KEY from env
        instructions="You are a friendly AI assistant. Keep your responses concise.",
    )

    # Create a session so follow-up turns see prior history.
    session = agent.create_session()

    print("Chat Agent Initialized. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        response = await agent.run(user_input, session=session)
        print(f"Assistant: {response.text}")

if __name__ == "__main__":
    asyncio.run(run_interactive())
```

### Recipe 2: Agent with a Single Tool

This recipe shows how to add a simple tool to an agent, allowing it to perform an action.

```python
# agent_with_tool.py
import asyncio
from datetime import datetime, timezone
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

# 1. Define the tool
@tool(description="Gets the current UTC date and time.")
def get_current_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

async def run_tool_agent():
    # 2. Create the agent with the tool attached
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You have a tool to get the current time. Use it when asked.",
        tools=[get_current_time],
    )

    response = await agent.run("What time is it right now?")
    print(f"Assistant: {response.text}")

if __name__ == "__main__":
    asyncio.run(run_tool_agent())
```

### Recipe 3: Basic Error Handling in Tools

This recipe demonstrates how to handle errors within a tool gracefully.

```python
# error_handling_tool.py
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

@tool(description="Divides two numbers.")
def divide(numerator: float, denominator: float) -> str:
    if denominator == 0:
        # Raising an exception will pass the error message to the LLM
        raise ValueError("Cannot divide by zero. Please ask the user for a non-zero denominator.")
    return str(numerator / denominator)

async def run_error_handling():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a math assistant.",
        tools=[divide],
    )

    # Trigger the error
    response = await agent.run("What is 10 divided by 0?")
    print(f"Assistant: {response.text}")
    # Expected: "I cannot divide by zero. Could you please provide a different number?"

if __name__ == "__main__":
    asyncio.run(run_error_handling())
```

---

## Intermediate Recipes

These recipes cover multi-agent systems, memory, and more complex tool interactions.

### Recipe 4: Simple Multi-Agent Workflow (Sequential)

This recipe shows how to chain two agents together: a researcher and a summarizer.

```python
# sequential_workflow.py
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

class SequentialWorkflow:
    def __init__(self, client: OpenAIChatClient):
        self.client = client

    async def run(self, topic: str):
        # 1. Define agents
        researcher = Agent(
            client=self.client,
            instructions="You are a world-class researcher. Find detailed information on the given topic.",
        )
        summarizer = Agent(
            client=self.client,
            instructions="You are a skilled editor. Summarize the provided text into a single, concise paragraph.",
        )

        # 2. Execute sequential flow — pass the researcher's output directly to the summarizer.
        print("Researching...")
        research = await researcher.run(f"Please research the topic: {topic}")

        print("Summarizing...")
        summary = await summarizer.run(
            f"Please summarize the following research findings:\n\n{research.text}"
        )

        return summary.text

async def main():
    workflow = SequentialWorkflow(OpenAIChatClient())
    result = await workflow.run("Quantum Computing")
    print(f"Result:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
```

For a graph-based variant with checkpointing and fan-in / fan-out edges, use `WorkflowBuilder` from `agent_framework` — see the [Workflows & Declarative Agents page](../microsoft_agent_framework_graphs_declarative/).

### Recipe 5: Agent with Multiple Tools and a Router

This recipe demonstrates a router agent that decides which specialist agent should handle a request.

```python
# router_agent.py
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

class RouterWorkflow:
    def __init__(self, client: OpenAIChatClient):
        self.router = Agent(
            client=client,
            instructions="You are a request router. Determine if a query is 'Billing' or 'Technical'. Respond with only one word.",
        )
        self.billing = Agent(
            client=client,
            instructions="You are a billing support specialist.",
        )
        self.tech = Agent(
            client=client,
            instructions="You are a technical support specialist.",
        )

    async def handle_request(self, user_input: str) -> str:
        # 1. Route the request (stateless single-turn call).
        route_response = await self.router.run(user_input)
        route = route_response.text

        # 2. Select target agent.
        target_agent = self.billing if "Billing" in route else self.tech

        # 3. Process with target agent.
        response = await target_agent.run(user_input)
        return response.text

async def main():
    workflow = RouterWorkflow(OpenAIChatClient())
    response = await workflow.handle_request("My invoice is incorrect.")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Recipe 6: Memory Integration with Azure AI Search

This recipe shows how to expose Azure AI Search as a retrieval tool so an agent can ground answers in a long-term knowledge index. `agent-framework-core==1.2.2` does not ship a dedicated `AzureAISearchMemory` class — the idiomatic pattern is a `@tool`-decorated function that wraps the `azure-search-documents` client.

```python
# memory_agent.py
import asyncio
import os
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
INDEX = "agent-memory"

async def _search(query: str, top: int = 3) -> list[str]:
    credential = DefaultAzureCredential()
    async with SearchClient(endpoint=ENDPOINT, index_name=INDEX, credential=credential) as client:
        results = []
        async for doc in await client.search(search_text=query, top=top):
            results.append(doc.get("content", ""))
        return results

@tool(description="Retrieve relevant documents from the agent's knowledge base.")
async def search_knowledge(query: Annotated[str, "The search query"]) -> str:
    hits = await _search(query)
    return "\n\n".join(hits) if hits else "No relevant documents found."

async def run_memory_agent():
    agent = Agent(
        client=FoundryChatClient(),
        instructions="You answer questions using the search_knowledge tool when helpful.",
        tools=[search_knowledge],
    )
    response = await agent.run("What languages does the Agent Framework support?")
    print(response.text)

if __name__ == "__main__":
    asyncio.run(run_memory_agent())
```

---

## Advanced Recipes

These recipes cover more complex scenarios like RAG and custom tool development.

### Recipe 7: RAG (Retrieval-Augmented Generation) Agent

This builds on the memory recipe to create a full RAG agent that can ingest and query documents.

```python
# rag_agent.py
import asyncio
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

class KnowledgeBase:
    def __init__(self):
        self.docs: dict[str, str] = {}

    def add_doc(self, doc_id: str, content: str) -> None:
        self.docs[doc_id] = content

    def search(self, query: str) -> str:
        # Simple keyword search for demonstration
        results = [content for content in self.docs.values() if query.lower() in content.lower()]
        return "\n\n".join(results) if results else "No relevant documents found."

kb = KnowledgeBase()
kb.add_doc("doc1", "Microsoft Agent Framework unifies Semantic Kernel and AutoGen.")

@tool(description="Search the knowledge base for information.")
def search_knowledge(query: Annotated[str, "The search query"]) -> str:
    return kb.search(query)

async def run_rag():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Use the search_knowledge tool to answer questions.",
        tools=[search_knowledge],
    )
    response = await agent.run("What does the framework unify?")
    print(response.text)

if __name__ == "__main__":
    asyncio.run(run_rag())
```

### Recipe 8: Complex Multi-Agent Orchestration with `WorkflowBuilder`

This recipe uses `WorkflowBuilder` — the real graph-based orchestration engine in `agent-framework-core` — for a robust and explicit multi-agent collaboration. Each agent becomes an executor node; edges define the flow.

```python
# workflow_orchestration.py
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

async def run_workflow(topic: str):
    client = OpenAIChatClient()

    # 1. Define agents
    researcher = Agent(client=client, instructions="You are a researcher. Produce bullet-point findings.")
    analyst = Agent(client=client, instructions="You are a data analyst. Extract insights from findings.")
    writer = Agent(client=client, instructions="You are a technical writer. Turn insights into a short report.")

    # 2. Build a sequential workflow: researcher -> analyst -> writer
    workflow = (
        WorkflowBuilder(start_executor=researcher)
        .add_edge(researcher, analyst)
        .add_edge(analyst, writer)
        .build()
    )

    # 3. Execute — the starting message flows through each agent.
    result = await workflow.run(f"Write a brief report on: {topic}")
    print(result.get_outputs()[-1])

if __name__ == "__main__":
    asyncio.run(run_workflow("AI Agents"))
```

---

## Integration Recipes

These recipes show how to integrate the Agent Framework with other Python technologies.

### Recipe 9: Exposing an Agent via FastAPI

This recipe shows how to create a web API endpoint to interact with an agent.

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

app = FastAPI()
agent: Agent | None = None

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup() -> None:
    global agent
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful API assistant.",
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    response = await agent.run(request.message)
    return {"reply": response.text}

# Run with: uvicorn main:app --reload
```

### Recipe 10: Event-Driven Agents with Azure Functions

This recipe demonstrates how to trigger an agent workflow from an Azure Queue Storage message using the Python v2 programming model.

```python
# function_app.py
import azure.functions as func
import logging
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

app = func.FunctionApp()

# In production, cache the chat client / agent across invocations (warm start)
# rather than reconstructing on every message.
_client = FoundryChatClient()
_agent = Agent(client=_client, instructions="Process the task.")

@app.queue_trigger(arg_name="msg", queue_name="agent-tasks", connection="AzureWebJobsStorage")
async def process_queue_item(msg: func.QueueMessage):
    user_query = msg.get_body().decode("utf-8")
    logging.info(f"Python Queue trigger processed message: {user_query}")

    response = await _agent.run(user_query)
    logging.info(f"Agent response: {response.text}")
```

---

## Troubleshooting Patterns

These patterns help with debugging and monitoring your agents.

### Pattern 11: Debugging Agent Execution with Streaming

Stream the agent's thoughts and tool calls in real-time to understand its decision-making process.

```python
# debug_streaming.py
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def debug_stream():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Helpful assistant.",
    )

    async for update in agent.run_stream("Tell me a joke."):
        # Each update is an AgentResponseUpdate with incremental text and/or tool calls.
        if update.tool_calls:
            for call in update.tool_calls:
                print(f"\033[90m[Tool Call: {call.name}({call.arguments})]\033[0m")  # Gray
        if update.text:
            print(f"\033[92m{update.text}\033[0m", end="", flush=True)  # Green
    print()

if __name__ == "__main__":
    asyncio.run(debug_stream())
```


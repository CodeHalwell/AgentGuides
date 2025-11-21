# Microsoft Agent Framework Python - Recipes and Code Patterns

This document provides a collection of practical, copy-paste-ready Python recipes for building common agentic patterns with the Microsoft Agent Framework for Python.

**Target Platform:** Python 3.10+
**Framework Version:** 1.0+

---

## Beginner Recipes

These recipes are for developers new to the framework and cover fundamental concepts.

### Recipe 1: Simple Chat Agent (Python)

This is the "Hello, World!" of the Agent Framework—a basic conversational agent that maintains history.

```python
# simple_chat_agent.py
import asyncio
from microsoft.agents.ai import AgentFactory, ChatAgent

async def run_interactive():
    factory = AgentFactory()
    
    # Create the agent
    agent = await factory.create_agent(
        ChatAgent,
        instructions="You are a friendly AI assistant. Keep your responses concise."
    )

    print("Chat Agent Initialized. Type 'exit' to quit.")
    
    # Create a thread for the conversation
    thread = await agent.create_thread()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ('exit', 'quit'):
            break

        # Invoke the agent
        response = await thread.invoke(user_input)
        print(f"Assistant: {response.get_content()}")

if __name__ == "__main__":
    asyncio.run(run_interactive())
```

### Recipe 2: Agent with a Single Tool

This recipe shows how to add a simple tool to an agent, allowing it to perform an action.

```python
# agent_with_tool.py
import asyncio
from datetime import datetime, timezone
from microsoft.agents.ai import AgentFactory, ChatAgent
from microsoft.agents.ai.tool import ai_function

# 1. Define the tool
@ai_function(description="Gets the current UTC date and time.")
def get_current_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

async def run_tool_agent():
    factory = AgentFactory()

    # 2. Create the agent and add the tool
    agent = await factory.create_agent(
        ChatAgent,
        instructions="You have a tool to get the current time. Use it when asked.",
        tools=[get_current_time]
    )

    thread = await agent.create_thread()
    response = await thread.invoke("What time is it right now?")
    print(f"Assistant: {response.get_content()}")

if __name__ == "__main__":
    asyncio.run(run_tool_agent())
```

### Recipe 3: Basic Error Handling in Tools

This recipe demonstrates how to handle errors within a tool gracefully.

```python
# error_handling_tool.py
import asyncio
from microsoft.agents.ai import AgentFactory, ChatAgent
from microsoft.agents.ai.tool import ai_function

@ai_function(description="Divides two numbers.")
def divide(numerator: float, denominator: float) -> str:
    if denominator == 0:
        # Raising an exception will pass the error message to the LLM
        raise ValueError("Cannot divide by zero. Please ask the user for a non-zero denominator.")
    return str(numerator / denominator)

async def run_error_handling():
    factory = AgentFactory()
    agent = await factory.create_agent(
        ChatAgent,
        instructions="You are a math assistant.",
        tools=[divide]
    )

    thread = await agent.create_thread()
    
    # Trigger the error
    response = await thread.invoke("What is 10 divided by 0?")
    print(f"Assistant: {response.get_content()}")
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
from microsoft.agents.ai import AgentFactory, ChatAgent

class SequentialWorkflow:
    def __init__(self, factory: AgentFactory):
        self.factory = factory

    async def run(self, topic: str):
        # 1. Define agents
        researcher = await self.factory.create_agent(
            ChatAgent,
            instructions="You are a world-class researcher. Find detailed information on the given topic."
        )
        summarizer = await self.factory.create_agent(
            ChatAgent,
            instructions="You are a skilled editor. Summarize the provided text into a single, concise paragraph."
        )

        # 2. Execute sequential flow
        # Use a single thread to pass context between agents
        thread = await researcher.create_thread()

        # Step 1: Researcher gathers information
        print("Researching...")
        await thread.invoke(f"Please research the topic: {topic}")
        
        # Step 2: Summarizer is invoked on the same thread. It sees the researcher's output.
        print("Summarizing...")
        summary_response = await summarizer.invoke(
            thread=thread, 
            message="Please summarize your findings."
        )

        return summary_response.get_content()

async def main():
    factory = AgentFactory()
    workflow = SequentialWorkflow(factory)
    result = await workflow.run("Quantum Computing")
    print(f"Result:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Recipe 5: Agent with Multiple Tools and a Router

This recipe demonstrates a router agent that decides which specialist agent should handle a request.

```python
# router_agent.py
import asyncio
from microsoft.agents.ai import AgentFactory, ChatAgent

class RouterWorkflow:
    def __init__(self, factory: AgentFactory):
        self.factory = factory
        self.router = None
        self.billing = None
        self.tech = None

    async def initialize(self):
        self.router = await self.factory.create_agent(
            ChatAgent,
            instructions="You are a request router. Determine if a query is 'Billing' or 'Technical'. Respond with only one word."
        )
        self.billing = await self.factory.create_agent(
            ChatAgent, 
            instructions="You are a billing support specialist."
        )
        self.tech = await self.factory.create_agent(
            ChatAgent, 
            instructions="You are a technical support specialist."
        )

    async def handle_request(self, user_input: str) -> str:
        # 1. Route the request
        router_thread = await self.router.create_thread()
        route_response = await router_thread.invoke(user_input)
        route = route_response.get_content()

        # 2. Select target agent
        if "Billing" in route:
            target_agent = self.billing
        else:
            target_agent = self.tech

        # 3. Process with target agent
        # Create a fresh thread for the actual conversation
        conversation_thread = await target_agent.create_thread()
        await conversation_thread.add_message(role="user", content=user_input)
        
        response = await conversation_thread.invoke("Provide a helpful response.")
        return response.get_content()

async def main():
    factory = AgentFactory()
    workflow = RouterWorkflow(factory)
    await workflow.initialize()
    
    response = await workflow.handle_request("My invoice is incorrect.")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Recipe 6: Memory Integration with Azure AI Search

This recipe shows how to configure an agent to use Azure AI Search for long-term memory and RAG.

```python
# memory_agent.py
import asyncio
import os
from azure.identity.aio import DefaultAzureCredential
from microsoft.agents.ai import AgentFactory, ChatAgent
from microsoft.agents.ai.memory.azure import AzureAISearchMemory

async def run_memory_agent():
    factory = AgentFactory()
    
    # Configure Azure AI Search Memory
    credential = DefaultAzureCredential()
    memory = AzureAISearchMemory(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        credential=credential,
        index_name="agent-memory"
    )
    
    # Create an agent with memory access
    # Note: In Python, memory is often injected via tools or specific RAG patterns
    # This example assumes a RAG-enabled agent pattern
    
    agent = await factory.create_agent(
        ChatAgent,
        instructions="You answer questions based on your knowledge base.",
        memory=memory # Hypothetical direct integration or via tools
    )
    
    # Ingest data
    await memory.upsert("doc-1", "The Agent Framework supports Python and .NET.")
    
    # Query
    thread = await agent.create_thread()
    response = await thread.invoke("What languages are supported?")
    print(response.get_content())

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
from typing import Annotated
from microsoft.agents.ai.tool import ai_function

class KnowledgeBase:
    def __init__(self):
        self.docs = {}

    def add_doc(self, id: str, content: str):
        self.docs[id] = content

    def search(self, query: str) -> str:
        # Simple keyword search for demonstration
        results = [content for content in self.docs.values() if query.lower() in content.lower()]
        return "\n\n".join(results) if results else "No relevant documents found."

kb = KnowledgeBase()
kb.add_doc("doc1", "Microsoft Agent Framework unifies Semantic Kernel and AutoGen.")

@ai_function(description="Search the knowledge base for information.")
def search_knowledge(query: Annotated[str, "The search query"]) -> str:
    return kb.search(query)

async def run_rag():
    factory = AgentFactory()
    agent = await factory.create_agent(
        ChatAgent,
        instructions="Use the search_knowledge tool to answer questions.",
        tools=[search_knowledge]
    )
    
    thread = await agent.create_thread()
    response = await thread.invoke("What does the framework unify?")
    print(response.get_content())
```

### Recipe 8: Complex Multi-Agent Orchestration with `Workflow`

This recipe uses the `Workflow` engine for a more robust and explicit multi-agent collaboration.

```python
# workflow_orchestration.py
import asyncio
from microsoft.agents.ai import AgentFactory, ChatAgent
from microsoft.agents.ai.orchestration import Workflow

async def run_workflow(topic: str):
    factory = AgentFactory()
    
    # 1. Define agents
    researcher = await factory.create_agent(ChatAgent, instructions="You are a researcher.")
    analyst = await factory.create_agent(ChatAgent, instructions="You are a data analyst.")
    writer = await factory.create_agent(ChatAgent, instructions="You are a technical writer.")

    # 2. Create workflow
    workflow = Workflow("ResearchPaper")

    # 3. Add steps
    step1 = workflow.add_agent(researcher)
    step2 = workflow.add_agent(analyst)
    step3 = workflow.add_agent(writer)

    # 4. Define flow
    workflow.add_edge(step1, step2)
    workflow.add_edge(step2, step3)

    # 5. Execute
    result = await workflow.execute(context={"topic": topic})
    print(result.get_last_message().content)

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
from microsoft.agents.ai import AgentFactory, ChatAgent
import asyncio

app = FastAPI()
factory = AgentFactory()
agent = None

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup():
    global agent
    agent = await factory.create_agent(
        ChatAgent,
        instructions="You are a helpful API assistant."
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    thread = await agent.create_thread()
    response = await thread.invoke(request.message)
    return {"reply": response.get_content()}

# Run with: uvicorn main:app --reload
```

### Recipe 10: Event-Driven Agents with Azure Functions

This recipe demonstrates how to trigger an agent workflow from an Azure Queue Storage message using the Python v2 programming model.

```python
# function_app.py
import azure.functions as func
import logging
from microsoft.agents.ai import AgentFactory, ChatAgent

app = func.FunctionApp()

@app.queue_trigger(arg_name="msg", queue_name="agent-tasks", connection="AzureWebJobsStorage")
async def process_queue_item(msg: func.QueueMessage):
    logging.info(f"Python Queue trigger processed message: {msg.get_body().decode('utf-8')}")
    
    user_query = msg.get_body().decode('utf-8')
    
    # Initialize agent (in production, use a singleton or cached factory)
    factory = AgentFactory()
    agent = await factory.create_agent(ChatAgent, instructions="Process the task.")
    
    thread = await agent.create_thread()
    response = await thread.invoke(user_query)
    
    logging.info(f"Agent response: {response.get_content()}")
```

---

## Troubleshooting Patterns

These patterns help with debugging and monitoring your agents.

### Pattern 11: Debugging Agent Execution with Streaming

Stream the agent's thoughts and tool calls in real-time to understand its decision-making process.

```python
# debug_streaming.py
import asyncio
from microsoft.agents.ai import AgentFactory, ChatAgent

async def debug_stream():
    factory = AgentFactory()
    agent = await factory.create_agent(ChatAgent, instructions="Helpful assistant.")
    thread = await agent.create_thread()

    async for message in thread.stream("Tell me a joke."):
        if message.role == "tool":
            print(f"\033[90m[Tool Call: {message.content}]\033[0m") # Gray
        elif message.role == "assistant":
            print(f"\033[92m[Assistant]: {message.content}\033[0m") # Green

if __name__ == "__main__":
    asyncio.run(debug_stream())
```

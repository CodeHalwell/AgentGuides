Latest: 1.1.0
# Microsoft Agent Framework Python - Comprehensive Technical Guide

**Release:** October 2025
**Framework Version:** 1.0+
**Target Platform:** Python 3.10+

---

## Introduction

This guide provides a comprehensive technical overview of the Microsoft Agent Framework for Python, designed for developers building advanced AI agents and multi-agent systems.

### Framework Overview

The Microsoft Agent Framework is an open-source SDK that unifies the capabilities of **Semantic Kernel** and **AutoGen**. It offers a single, cohesive platform for Python developers to build everything from simple conversational bots to complex, orchestrated multi-agent systems.

-   **Inheritance from Semantic Kernel:** It brings enterprise-grade features, including a robust plugin/tool system, memory management, and a wide array of connectors.
-   **Inheritance from AutoGen:** It incorporates sophisticated multi-agent orchestration, group chat coordination, and flexible conversation patterns.

The framework is designed with a Python-first approach, embracing `asyncio` for scalability and integrating seamlessly with the rich Python data science and web development ecosystems.

### Key Objectives

-   **Unified SDK:** A single, Pythonic library for all agent development needs.
-   **Production-Ready:** Built-in features for observability, security, and scalable deployment.
-   **Extensibility:** A modular design that allows for custom agents, tools, and memory backends.
-   **Azure Integration:** Deep, native integration with Azure AI services while remaining platform-agnostic.

---

## Core Fundamentals

### Architecture Principles

The framework's architecture is layered to promote modularity and ease of use.

```
+-----------------------------------+
|      Application Layer            |
| (Your Agents, FastAPI/Flask APIs) |
+-----------------------------------+
|      Orchestration Layer          |
| (Workflows, GroupChatManager)     |
+-----------------------------------+
|      Agent Abstraction Layer      |
| (Agent, ChatAgent, AgentThread)   |
+-----------------------------------+
|      Core Components Layer          |
| (Tools, Memory, LLM Providers)    |
+-----------------------------------+
|      Integration Layer            |
| (Azure, OpenAI, Custom Connectors)|
+-----------------------------------+
```

### Installation

Setting up your Python environment is straightforward.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install the core package
pip install microsoft-agents-ai --pre

# 3. Install provider-specific packages
pip install microsoft-agents-ai-azure --pre
pip install azure-identity
```

### Authentication and Configuration

Manage credentials securely using environment variables and `azure-identity`.

**1. Environment Variables:**

Create a `.env` file in your project root.

```
# .env
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"
```

**2. Loading Configuration:**

Use a library like `python-dotenv` to load these variables.

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
```

**3. Using `DefaultAzureCredential` (Recommended):**

For production, rely on managed identities and `DefaultAzureCredential`.

```python
from azure.identity.aio import DefaultAzureCredential

# This will automatically use the managed identity of the host,
# environment variables, or local Azure CLI login.
credential = DefaultAzureCredential()
```

### Environment Setup & Basic Usage

```python
# main.py
import asyncio
from microsoft.agents.ai import AgentFactory
from microsoft.agents.ai.azure import add_azure_openai
from azure.identity.aio import DefaultAzureCredential
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT

async def main():
    # Create the agent factory and configure it
    factory = AgentFactory()
    
    # Use DefaultAzureCredential for secure, passwordless auth
    credential = DefaultAzureCredential()
    
    factory.add_azure_openai(
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        credential=credential
    )

    # Create a simple chat agent
    agent = await factory.create_agent(
        "ChatAgent",
        instructions="You are a helpful AI assistant for Python developers."
    )

    # Interact with the agent
    thread = await agent.create_thread()
    response = await thread.invoke("What are decorators in Python?")
    
    print(response.get_content())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Simple Agents

### `Agent` vs. `ChatAgent`

-   **`Agent`**: A stateless agent for single-turn interactions.
-   **`ChatAgent`**: A stateful agent that manages conversation history within a thread. This is the most common type.

### Creating a `ChatAgent`

```python
import asyncio
from microsoft.agents.ai import AgentFactory, ChatAgent

async def run_chat_agent(factory: AgentFactory):
    agent = await factory.create_agent(
        ChatAgent,
        instructions="You are a friendly and helpful assistant."
    )
    
    thread = await agent.create_thread()
    
    print("Starting conversation... (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = await thread.invoke(user_input)
        print(f"Assistant: {response.get_content()}")
```

### Agent Lifecycle

Agents are designed to be managed via the `AgentFactory`, which handles resource allocation and disposal. Using `async with` ensures that resources like HTTP clients and credentials are properly managed.

---

## Multi-Agent Systems

### Orchestration Patterns

-   **Sequential Workflow:** The output of one agent is passed as the input to the next.
-   **Router/Dispatcher:** A primary agent routes tasks to specialized agents based on the query.
-   **Group Chat:** Multiple agents collaborate in a shared conversation, moderated by a manager.

### Example: Router Pattern

```python
import asyncio
from microsoft.agents.ai import AgentFactory, ChatAgent

class RouterWorkflow:
    def __init__(self, factory: AgentFactory):
        self._factory = factory
        self._router = None
        self._billing_agent = None
        self._tech_agent = None

    async def initialize(self):
        self._router = await self._factory.create_agent(
            ChatAgent,
            instructions="You are a router. Classify the user's query as 'Billing' or 'Technical'. Respond with only one of those words."
        )
        self._billing_agent = await self._factory.create_agent(
            ChatAgent,
            instructions="You are a billing support expert."
        )
        self._tech_agent = await self._factory.create_agent(
            ChatAgent,
            instructions="You are a technical support expert."
        )

    async def handle_request(self, user_query: str):
        router_thread = await self._router.create_thread()
        route_response = await router_thread.invoke(user_query)
        route = route_response.get_content()

        if "Billing" in route:
            target_agent = self._billing_agent
        else:
            target_agent = self._tech_agent
            
        conversation_thread = await target_agent.create_thread()
        await conversation_thread.add_message(content=user_query, role="user")
        final_response = await conversation_thread.invoke("Address the user's query.")
        
        return final_response.get_content()

# --- Usage ---
# workflow = RouterWorkflow(factory)
# await workflow.initialize()
# response = await workflow.handle_request("I have a problem with my invoice.")
# print(response)
```

---

## Tools Integration

### Defining and Using Tools

Tools are standard Python functions decorated with `@ai_function` to expose them to an agent.

```python
from microsoft.agents.ai.tool import ai_function
from typing import Annotated

@ai_function(description="Get the current time in a specified timezone.")
async def get_current_time(
    timezone: Annotated[str, "The IANA timezone name, e.g., 'America/New_York'."]
) -> str:
    from datetime import datetime
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        return f"The current time in {timezone} is {datetime.now(tz).strftime('%H:%M:%S')}."
    except zoneinfo.ZoneInfoNotFoundError:
        return "Unknown timezone."

# --- Attaching the tool to an agent ---
# agent = await factory.create_agent(
#     ChatAgent,
#     instructions="You can get the current time.",
#     tools=[get_current_time]
# )
# response = await agent.create_thread().invoke("What time is it in New York?")
```

### Built-in Azure Tools

The `microsoft-agents-ai-azure` package provides tools for interacting with Azure services like Azure AI Search.

---

## Structured Output

Force an agent to respond in a specific JSON schema using Pydantic models.

```python
from pydantic import BaseModel, Field
from typing import List

class UserProfile(BaseModel):
    """A model to hold structured user information."""
    name: str = Field(description="The user's full name.")
    age: int = Field(description="The user's age.")
    interests: List[str] = Field(description="A list of the user's interests.")

async def extract_structured_data(factory: AgentFactory, text: str) -> UserProfile:
    agent = await factory.create_agent(
        ChatAgent,
        instructions="Extract user profile information from the text provided."
    )
    thread = await agent.create_thread()
    
    # Pass the Pydantic model as the expected response type
    response = await thread.invoke(text, response_type=UserProfile)
    
    return response.get_content()

# --- Usage ---
# text_blob = "My name is Jane Doe, I'm 28, and I love hiking and programming in Python."
# profile = await extract_structured_data(factory, text_blob)
# print(profile.model_dump_json(indent=2))
```

This is a foundational part of the comprehensive guide. I will continue to generate the remaining sections.
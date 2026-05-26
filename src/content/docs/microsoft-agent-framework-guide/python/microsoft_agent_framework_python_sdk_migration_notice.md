---
title: "Migration Notice: agent-framework → azure-ai-agents"
description: "Critical correction: the correct Python package for Azure AI Agents is azure-ai-agents (not agent-framework). This document explains what changed and how to update your code."
framework: microsoft-agent-framework
language: python
---

# Migration Notice: `agent-framework` → `azure-ai-agents`

> **Important correction:** Several earlier guides in this documentation set described a Python package called `agent-framework` (or `agent-framework-core`). That package does not match the official, publicly available Azure AI Agents SDK on PyPI. The correct package is **`azure-ai-agents`**, maintained by Microsoft as part of the Azure SDK for Python. This page explains the differences and shows you exactly how to update your code.

---

## The Short Version

| | Old (incorrect) | New (correct) |
|---|---|---|
| **PyPI package** | `agent-framework` / `agent-framework-core` | `azure-ai-agents` |
| **Install command** | `pip install agent-framework` | `pip install azure-ai-agents azure-identity` |
| **Top-level import** | `from agent_framework import Agent` | `from azure.ai.agents import AgentsClient` |
| **Entry-point class** | `Agent` | `AgentsClient` |
| **Version referenced** | 1.6.0 (fictional) | 1.1.0 (real, on PyPI) |
| **PyPI URL** | N/A | https://pypi.org/project/azure-ai-agents/ |
| **Source of truth** | — | [Azure SDK for Python on GitHub](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-agents) |

---

## Why This Matters

The `agent-framework` package described in earlier guides was based on a prototype or fictional API surface. If you try to run code from those guides, you will encounter an `ImportError` because neither `agent-framework` nor `agent-framework-core` exists as documented on PyPI. Any code that imports from `agent_framework` will fail immediately unless you have a private internal package by that name.

The `azure-ai-agents` package is Microsoft's official, generally available Python SDK for the Azure AI Agents service, published and maintained under the [Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python) umbrella.

---

## Before and After: Side-by-Side Comparison

### Installation

```bash
# WRONG — this package does not exist as documented on PyPI
pip install agent-framework

# CORRECT — official Microsoft SDK, version 1.1.0
pip install azure-ai-agents azure-identity
```

### Imports and Client Initialisation

```python
# WRONG — fictional package and class names
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

client = FoundryChatClient(
    endpoint="https://...",
    subscription_id="...",
)
agent = Agent(client=client, instructions="You are a helpful assistant.")
response = await agent.run("hello")
```

```python
# CORRECT — azure-ai-agents v1.1.0
import os
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="my-assistant",
    instructions="You are a helpful assistant.",
)

thread = client.threads.create()

client.messages.create(
    thread_id=thread.id,
    role="user",
    content="hello",
)

run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    for content in msg.content:
        if hasattr(content, "text"):
            print(content.text.value)
```

---

## Key Class and API Mapping

The table below maps every major concept from the old `agent-framework` guides to its equivalent in `azure-ai-agents` 1.1.0.

| Old (`agent-framework`) | New (`azure-ai-agents` 1.1.0) | Notes |
|---|---|---|
| `Agent` (entry point) | `AgentsClient` | The client manages all resources |
| `FoundryChatClient` | Removed — use `AgentsClient` directly | `AgentsClient` is both client and orchestrator |
| `Agent(client=..., instructions=...)` | `client.create_agent(model=..., instructions=...)` | Agent is now a remote resource, not a local object |
| `agent.run("...")` | `client.runs.create_and_process(thread_id, agent_id=agent.id)` | Runs are tied to threads |
| `@tool` decorator | `FunctionTool(functions={my_func})` | Pass `tools=[FunctionTool(...)]` to `create_agent` |
| `InMemoryHistoryProvider` | `client.threads` (server-side history) | Thread history is managed by Azure, not locally |
| `WorkflowBuilder` | `client.runs.stream()` with `AgentEventHandler` | Use event-driven streaming for complex flows |
| `MCPStdioTool` | Not directly available | Use `OpenApiTool` for HTTP-based tools, or wrap external tools as Python functions via `FunctionTool` |
| `AzureCliCredential` from `agent_framework.auth` | `AzureCliCredential` from `azure.identity` | Standard Azure Identity library |
| `agent.session` / `AgentSession` | `client.threads` + `client.messages` | Sessions map to threads; messages are retrieved separately |
| Async: `await agent.run(...)` | Async: `from azure.ai.agents.aio import AgentsClient` | Use the `.aio` submodule for async |

---

## Complete Migration Example

Here is a realistic before-and-after showing a simple single-turn agent with a custom function tool.

### Before (fictional `agent-framework` API)

```python
# WRONG — do not use this code
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.tools import tool

@tool
def get_weather(location: str) -> str:
    """Return the current weather for a location."""
    return f"Sunny, 22°C in {location}"

async def main():
    client = FoundryChatClient(
        endpoint="https://myresource.services.ai.azure.com",
        subscription_id="...",
    )
    agent = Agent(
        client=client,
        name="weather-bot",
        instructions="You are a weather assistant.",
        tools=[get_weather],
    )
    response = await agent.run("What is the weather in London?")
    print(response.text)

asyncio.run(main())
```

### After (official `azure-ai-agents` v1.1.0)

```python
# CORRECT — azure-ai-agents v1.1.0
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FunctionTool, ListSortOrder
from azure.identity import DefaultAzureCredential


def get_weather(location: str) -> str:
    """Return the current weather for a location."""
    return f"Sunny, 22°C in {location}"


def main():
    client = AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )

    # Wrap your Python function in a FunctionTool
    weather_tool = FunctionTool(functions={get_weather})

    # Create the agent with the tool attached
    agent = client.create_agent(
        model=os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
        name="weather-bot",
        instructions="You are a weather assistant.",
        tools=weather_tool.definitions,
    )

    # Create a thread and post a user message
    thread = client.threads.create()
    client.messages.create(
        thread_id=thread.id,
        role="user",
        content="What is the weather in London?",
    )

    # Run the agent; enable_auto_function_calls handles tool dispatch automatically
    run = client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
        tool_resources=weather_tool.resources,
        enable_auto_function_calls=True,
    )

    print(f"Run finished with status: {run.status}")

    # Retrieve the agent's response
    messages = client.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING,
    )
    for msg in messages:
        if msg.text_messages:
            print(f"{msg.role}: {msg.text_messages[-1].text.value}")

    # Clean up
    client.delete_agent(agent.id)
    client.threads.delete(thread.id)


if __name__ == "__main__":
    main()
```

Key changes to note:

- The `@tool` decorator is replaced by `FunctionTool(functions={get_weather})`. The function's docstring and type hints are used automatically to generate the tool schema that is sent to the model.
- `enable_auto_function_calls=True` tells the SDK to call your Python function and submit the result back to the model automatically, so you do not need to write a polling loop that handles `REQUIRES_ACTION` status yourself.
- There is no longer a single `response.text` — instead, you iterate over the thread's messages and read `msg.text_messages[-1].text.value`.

---

## Async Migration

If you were using the async variant of the fictional API, the async migration is straightforward:

```python
# CORRECT async pattern — azure-ai-agents v1.1.0
import os
import asyncio
from azure.ai.agents.aio import AgentsClient  # note: .aio submodule
from azure.ai.agents.models import ListSortOrder
from azure.identity.aio import DefaultAzureCredential  # note: .aio submodule


async def main() -> None:
    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = await client.create_agent(
            model=os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
            name="async-agent",
            instructions="You are a helpful assistant.",
        )

        thread = await client.threads.create()

        await client.messages.create(
            thread_id=thread.id,
            role="user",
            content="hello",
        )

        run = await client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
        )
        print(f"Run status: {run.status}")

        messages = client.messages.list(
            thread_id=thread.id,
            order=ListSortOrder.ASCENDING,
        )
        async for msg in messages:
            if msg.text_messages:
                print(f"{msg.role}: {msg.text_messages[-1].text.value}")

        await client.delete_agent(agent.id)


asyncio.run(main())
```

You will also need `aiohttp` installed for the async client:

```bash
pip install aiohttp
```

---

## Common Import Errors After Migration

### `ModuleNotFoundError: No module named 'agent_framework'`

You have the old import in your code. Replace every `from agent_framework import ...` or `import agent_framework` with the correct `azure.ai.agents` equivalents shown above.

### `ModuleNotFoundError: No module named 'azure.ai.agents'`

Run `pip install azure-ai-agents` in your active Python environment. Double-check that you are using the same environment your script runs in.

### `ImportError: cannot import name 'FoundryChatClient' from 'azure.ai.agents'`

`FoundryChatClient` does not exist in `azure-ai-agents`. Use `AgentsClient` directly.

### `AttributeError: 'AgentsClient' object has no attribute 'run'`

The `Agent` object in the old API had a `.run()` method. In `azure-ai-agents`, running an agent means creating a `ThreadRun` via `client.runs.create_and_process(thread_id=..., agent_id=...)`. Make sure you have created a thread and posted a message to it first.

---

## Further Reading

- [Installation & Quickstart](./microsoft_agent_framework_python_installation_and_quickstart) — the correct quickstart guide for `azure-ai-agents` 1.1.0
- [Official PyPI page](https://pypi.org/project/azure-ai-agents/)
- [Azure SDK for Python — azure-ai-agents README](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-agents)
- [Microsoft Learn: Azure AI Agents quickstart (Python)](https://learn.microsoft.com/azure/ai-services/agents/quickstart?pivots=programming-language-python-azure)
- [API reference](https://aka.ms/azsdk/azure-ai-agents/python/reference)

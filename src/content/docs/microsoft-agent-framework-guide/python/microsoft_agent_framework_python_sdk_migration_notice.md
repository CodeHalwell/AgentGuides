---
title: "azure-ai-agents — Integration Add-on Reference"
description: "azure-ai-agents is an optional Microsoft add-on for the Azure AI Agents service, not a replacement for agent-framework. This page explains when to use it alongside the framework."
framework: microsoft-agent-framework
language: python
---

# `azure-ai-agents` — Integration Add-on Reference

> **Clarification:** `azure-ai-agents` is **not** a replacement for `agent-framework`. It is an optional Microsoft SDK for direct access to the Azure AI Agents service REST API. Use it alongside `agent-framework` when you need low-level control over Azure AI Agents service resources (threads, runs, vector stores). For everyday agent development, use `agent-framework` directly.

---

## Package Summary

| | `agent-framework` | `azure-ai-agents` |
|---|---|---|
| **Role** | Primary agent framework | Azure AI Agents service add-on |
| **Install** | `pip install agent-framework` | `pip install azure-ai-agents azure-identity` |
| **Import root** | `agent_framework` | `azure.ai.agents` |
| **Entry-point class** | `Agent` | `AgentsClient` |
| **Use case** | All agent development | Direct Azure AI Agents service access |

---

## When to use `azure-ai-agents`

Use `azure-ai-agents` alongside `agent-framework` when you need:

- **Direct thread/run lifecycle management** — `client.threads.create()`, `client.runs.create_and_process()`
- **Azure AI Agents service tools** — Code Interpreter, File Search, Bing Grounding, Azure AI Search via `AgentsClient`
- **Vector store management** — `VectorStore`, `VectorStoreFileBatch` for file-backed retrieval
- **Connected agent orchestration** — `ConnectedAgentTool` for multi-agent topologies on Azure

For everything else — tool decoration, middleware, sessions, MCP, A2A, declarative agents, functional workflows — use `agent-framework` directly.

---

## Side-by-side: `agent-framework` vs `azure-ai-agents`

### agent-framework (primary framework)

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

async def main():
    client = FoundryChatClient(
        endpoint="https://myresource.services.ai.azure.com",
        subscription_id="...",
    )
    agent = Agent(client=client, instructions="You are a helpful assistant.")
    response = await agent.run("hello")
    print(response.text)

asyncio.run(main())
```

### azure-ai-agents (add-on, direct service access)

```python
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
client.messages.create(thread_id=thread.id, role="user", content="hello")
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

messages = client.messages.list(thread_id=thread.id)
for msg in messages:
    for content in msg.content:
        if hasattr(content, "text"):
            print(content.text.value)

client.delete_agent(agent.id)
client.threads.delete(thread.id)
```

---

## Installation

```bash
# Primary framework (required)
pip install agent-framework

# Add-on for Azure AI Agents service access (optional)
pip install azure-ai-agents azure-identity
```

Provider-specific `agent-framework` sub-packages:

```bash
pip install agent-framework-azure-ai       # Azure AI Foundry chat client
pip install agent-framework-openai         # OpenAI / Azure OpenAI chat clients
pip install agent-framework-a2a --pre      # A2A protocol (pre-release)
pip install agent-framework-declarative --pre  # Declarative workflows (pre-release)
```

---

## Further Reading

- [Comprehensive Python guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/) — full `agent-framework` API reference
- [Model providers](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/) — `FoundryChatClient`, `OpenAIChatClient`, `AnthropicClient`
- [azure-ai-agents integration — class reference Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives/) — `AgentsClient`, `FunctionTool`, `ToolSet`, and more
- [Official `azure-ai-agents` PyPI page](https://pypi.org/project/azure-ai-agents/)

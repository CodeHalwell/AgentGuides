---
title: "Microsoft Agent Framework (Python) — azure-ai-agents Add-on"
description: "azure-ai-agents is an optional Microsoft add-on for the Azure AI Agents service, not a replacement for agent-framework. This page explains when to use it alongside the framework."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "azure-ai-agents add-on"
  order: 90
---

> **Clarification:** `azure-ai-agents` is **not** a replacement for `agent-framework`. It is an optional Microsoft SDK for direct access to the Azure AI Agents service REST API. Use it alongside `agent-framework` when you need low-level control over Azure AI Agents service resources (threads, runs, vector stores). For everyday agent development, use `agent-framework` directly.

> **Deprecation:** `azure-ai-agents` (1.1.0) targets the **classic** Foundry Agent Service: the threads, messages and runs API. Microsoft Learn marks Agents (classic) as deprecated, to be retired on 31 March 2027, and says that models later than GPT-5 need the new agents. New code should use the new Foundry Agent Service, which `agent-framework` reaches through `FoundryChatClient` and `FoundryAgent` (`agent-framework-foundry`, built on `azure-ai-projects` 2.x). See Microsoft's [migration guide](https://learn.microsoft.com/azure/foundry/agents/how-to/migrate).

---

## Package Summary

| | `agent-framework` | `azure-ai-agents` |
|---|---|---|
| **Role** | Primary agent framework | Classic Azure AI Agents service SDK (deprecated service) |
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
from azure.identity import AzureCliCredential

async def main():
    client = FoundryChatClient(
        project_endpoint="https://myresource.services.ai.azure.com/api/projects/my-project",
        model="%%model=openai-small%%",
        credential=AzureCliCredential(),
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
    model=os.environ["MODEL_DEPLOYMENT_NAME"],   # a deployment the classic service supports
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
pip install agent-framework-foundry        # Microsoft Foundry chat client (FoundryChatClient)
pip install agent-framework-openai         # OpenAI / Azure OpenAI chat clients
pip install agent-framework-a2a --pre      # A2A protocol (beta)
pip install agent-framework-declarative    # Declarative YAML agents and workflows
```

---

## Further Reading

- [Comprehensive Python guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/) — the `agent-framework` core, with an API map
- [Model providers](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/) — `FoundryChatClient`, `OpenAIChatClient`, `AnthropicClient`
- [`azure-ai-agents` reference on Microsoft Learn](https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme) — `AgentsClient`, threads, runs, tool sets and vector stores
- [Official `azure-ai-agents` PyPI page](https://pypi.org/project/azure-ai-agents/)

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-25 | 1.22.0 | Added the deprecation of the classic Agents service that `azure-ai-agents` targets (retirement on 31 March 2027, per Microsoft Learn), and replaced the current model placeholder in the classic example with a deployment name, because the classic service does not support models after GPT-5. |

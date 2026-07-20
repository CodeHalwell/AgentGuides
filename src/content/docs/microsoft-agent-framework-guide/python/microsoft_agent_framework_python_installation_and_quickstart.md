---
title: "agent-framework (Python) — Installation & Quickstart"
description: "Get started with agent-framework 1.10.0: installation, authentication, your first agent, tools, sessions, and async patterns."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — Installation & Quickstart

This guide covers everything you need to get your first agent running in Python using `agent-framework` (version 1.10.0).

---

## Package Information

| Property | Value |
|---|---|
| Package name | `agent-framework` |
| Current version | `1.10.0` |
| Import root | `agent_framework` |
| Python requirement | 3.10 or later |

```bash
pip install agent-framework
```

Provider-specific sub-packages (install whichever chat backend you need):

```bash
pip install agent-framework-azure-ai       # Azure AI Foundry (FoundryChatClient)
pip install agent-framework-openai         # OpenAI / Azure OpenAI (OpenAIChatClient)
pip install agent-framework-a2a --pre      # A2A protocol (pre-release)
pip install agent-framework-declarative --pre  # Declarative YAML agents (pre-release)
```

---

## Architecture Overview

The framework is organised around a small number of core primitives:

1. **`Agent`** — the primary class. Wraps a chat client, holds instructions, and orchestrates tool calls.
2. **Chat client** — provider-specific wrapper (`FoundryChatClient`, `OpenAIChatClient`, `AnthropicClient`, …). Passed to `Agent(client=...)`.
3. **`@tool`** — decorator that exposes a Python function to the agent. Type hints and docstring become the tool schema automatically.
4. **`AgentSession`** — multi-turn state. Passed to `agent.run(prompt, session=session)` to maintain conversation history.
5. **`WorkflowBuilder`** — directed-graph orchestration for multi-agent pipelines.
6. **`@workflow` / `@step`** — functional workflow API; write multi-step pipelines as plain async Python without graph wiring.

---

## Your First Agent

The following example creates a minimal single-turn agent using OpenAI as the chat backend.

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant. Answer concisely.",
    )
    response = await agent.run("What is the capital of France?")
    print(response.text)


asyncio.run(main())
```

### Using Azure AI Foundry

Replace `OpenAIChatClient` with `FoundryChatClient` for Azure-hosted models:

```python
import asyncio
import os
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient


async def main() -> None:
    client = FoundryChatClient(
        endpoint=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        deployment=os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o"),
    )
    agent = Agent(
        client=client,
        instructions="You are a helpful assistant.",
    )
    response = await agent.run("Summarise the agent-framework in one sentence.")
    print(response.text)


asyncio.run(main())
```

---

## Adding Tools

Use the `@tool` decorator to expose Python functions. The framework reads type hints and the docstring to build the tool schema automatically.

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient


@tool
def get_weather(location: str) -> str:
    """Return the current weather for a location."""
    return f"Sunny, 22°C in {location}"


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a weather assistant.",
        tools=[get_weather],
    )
    response = await agent.run("What is the weather in London?")
    print(response.text)


asyncio.run(main())
```

---

## Multi-Turn Sessions

Use `agent.create_session()` to maintain conversation history across multiple turns:

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    )

    session = agent.create_session()

    response1 = await agent.run("My name is Alice.", session=session)
    print(response1.text)

    response2 = await agent.run("What is my name?", session=session)
    print(response2.text)  # Agent remembers: Alice


asyncio.run(main())
```

---

## Streaming Responses

Pass `stream=True` to `agent.run()` to receive token-by-token output:

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    )

    async for chunk in await agent.run("Explain asyncio in Python.", stream=True):
        print(chunk.text, end="", flush=True)
    print()


asyncio.run(main())
```

---

## Common Errors

### `ModuleNotFoundError: No module named 'agent_framework'`

Run `pip install agent-framework` in your active environment. For provider-specific classes, install the matching sub-package (e.g. `pip install agent-framework-openai`).

### `ModuleNotFoundError: No module named 'agent_framework.foundry'`

Install the Azure AI Foundry provider: `pip install agent-framework-azure-ai`.

### `AttributeError: 'Agent' object has no attribute 'run'`

You may have an old version of the package. Upgrade with `pip install --upgrade agent-framework`.

---

## Next Steps

- [Core fundamentals](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/#core-fundamentals) — full architecture and installation reference
- [Model providers](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/) — all chat client constructors with real signatures
- [Tools](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_tools/) — `@tool`, `FunctionTool`, approval gates, schemas, runtime context
- [MCP integration](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_mcp/) — `MCPStdioTool`, `MCPStreamableHTTPTool`, `MCPWebsocketTool`
- [Multi-agent orchestration](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/) — `WorkflowBuilder`, sequential, concurrent, handoff, group chat
- [Class deep dives Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — source-verified deep dives into `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`

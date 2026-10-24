---
title: "Microsoft Agent Framework (Python) — Installation and Quick Start"
description: "Get started with agent-framework: installation, authentication, your first agent, tools, sessions, and async patterns."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "Installation & quick start"
---

This guide covers everything you need to get your first agent running in Python using `agent-framework` (version %%version=agent-framework%%, checked %%versions-checked%%).

---

## Package Information

| Property | Value |
|---|---|
| Package name | `agent-framework` |
| Current version | `%%version=agent-framework%%` |
| Import root | `agent_framework` |
| Python requirement | 3.10 or later |

```bash
pip install agent-framework
```

The `agent-framework` meta package depends on `agent-framework-core[all]`, so it pulls in every integration package (OpenAI, Foundry, Anthropic, A2A, Redis, DevUI and so on, including the beta ones). That is convenient for learning but heavy for a deployed service. For a smaller install, take the core and only the integrations you use:

```bash
pip install agent-framework-core           # Agent, tools, sessions, workflows
pip install agent-framework-openai         # OpenAI / Azure OpenAI (OpenAIChatClient)
pip install agent-framework-foundry        # Microsoft Foundry (FoundryChatClient)
pip install agent-framework-a2a --pre      # A2A protocol (beta)
pip install agent-framework-declarative    # Declarative YAML agents
```

The import paths are the same either way (`agent_framework.openai`, `agent_framework.foundry`, ...); a namespace whose package is not installed raises a `ModuleNotFoundError` naming the package to install. Two integrations have platform limits: `agent-framework-github-copilot` needs Python 3.11+, and `agent-framework-hyperlight` installs only on Linux x86-64 and Windows AMD64.

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

### Using Microsoft Foundry

Replace `OpenAIChatClient` with `FoundryChatClient` for Azure-hosted models:

```python
import asyncio
import os
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential


async def main() -> None:
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ.get("FOUNDRY_MODEL", "%%model=openai%%"),
        credential=AzureCliCredential(),
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

### `ModuleNotFoundError: The 'agent-framework-foundry' package is not installed`

The `agent_framework.foundry`, `agent_framework.openai`, `agent_framework.anthropic` (and so on) namespaces ship with the core package and load their integration package on first use. Install the package the message names: `pip install agent-framework-foundry`. (The older `agent-framework-azure-ai` package stopped at 1.0.0rc6 and does not provide `FoundryChatClient`.)

### `ImportError: cannot import name 'ChatAgent'` (or `ChatMessage`, `ai_function`)

The code was written for a 2025 preview. Those names were renamed before 1.0: `ChatAgent` → `Agent`, `ChatMessage` → `Message`, `@ai_function` → `@tool`. See the list of [breaking changes](/microsoft-agent-framework-guide/python/#breaking-changes-worth-knowing).

### `SettingNotFoundError` when creating the client

`OpenAIChatClient()` with no arguments reads `OPENAI_API_KEY` and `OPENAI_MODEL` (or `OPENAI_CHAT_MODEL`) from the environment. A missing model gives "Model must be specified via the 'model' parameter or the 'OPENAI_CHAT_MODEL', 'OPENAI_MODEL' environment variable"; a missing key gives the less obvious "Exactly one of 'base_url', 'endpoint' must be provided", because without a key the client cannot tell which service you mean. Set the variables, or pass `model=` and `api_key=` explicitly. A `.env` file is read only when you pass `env_file_path=`.

---

## Next Steps

- [Comprehensive guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/) — configuration, agents, streaming, sessions, workflows and more on one page
- [Model providers](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/) — all chat client constructors with real signatures
- [Tools](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_tools/) — `@tool`, `FunctionTool`, approval gates, schemas, runtime context
- [MCP integration](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_mcp/) — `MCPStdioTool`, `MCPStreamableHTTPTool`, `MCPWebsocketTool`
- [Multi-agent orchestration](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/) — `WorkflowBuilder`, sequential, concurrent, handoff, group chat
- [API map](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/#api-map) — which page documents each class

Latest: 0.11.5 | Updated: April 2026
# AG2 (AutoGen) Comprehensive Guide

**Version:** 0.11.5
**Last Updated:** April 2026
**Focus:** Modern AutoGen (AG2) Framework

## Overview

AG2 (formerly AutoGen) is the next generation of the AutoGen framework, designed for building advanced multi-agent systems. It introduces a more modular architecture, improved orchestration, and enhanced tool integration compared to the legacy AutoGen.

## Key Features

*   **Modular Architecture:** Decoupled components for flexible agent design.
*   **Enhanced Orchestration:** sophisticated conversation management.
*   **Tool Integration:** Seamless integration with custom tools and MCP.
*   **Human-in-the-Loop:** Built-in support for human oversight and intervention.

## Installation

```bash
pip install ag2
```

## Basic Usage

```python
# Compatibility note: pre-0.11.5 imports used `from autogen import ...`
# As of 0.11.x, the canonical package is `ag2` (or `autogen` as an alias).
# Use `from autogen.beta import ...` for the new event-driven API.
from ag2 import Agent, GroupChat, GroupChatManager

# Define agents
user_proxy = Agent(
    name="User_Proxy",
    system_message="A human admin.",
    human_input_mode="ALWAYS"
)

coder = Agent(
    name="Coder",
    system_message="You are a skilled Python developer."
)

# Create a group chat
groupchat = GroupChat(agents=[user_proxy, coder], messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat)

# Start the conversation
user_proxy.initiate_chat(manager, message="Write a Python script to fetch stock prices.")
```

## Advanced Concepts

### Custom Agents

You can create custom agents by subclassing the base `Agent` class and overriding methods like `generate_reply`.

### Tool Use

AG2 supports defining tools as Python functions and registering them with agents.

```python
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

coder.register_function(function_map={"get_weather": get_weather})
```

### Group Chat Management

The `GroupChatManager` handles the flow of messages between agents. You can customize the speaker selection logic.

## Migration from Legacy AutoGen

If you are migrating from the legacy `pyautogen` package, note the following changes:
*   Package name: `pyautogen` -> `ag2`
*   Import paths may have changed.
*   Some deprecated classes have been removed.

## autogen.beta: Event-Driven Architecture (v0.11.x+)

`autogen.beta` is a ground-up redesign of AG2 with a streaming, event-driven architecture. Every conversation runs on a **MemoryStream** — a pub/sub channel that enables parallel agent execution.

```python
import asyncio
from autogen.beta import ConversableAgent, MemoryStream

async def main():
    # Create agents with the new event-driven system
    assistant = ConversableAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={"model": "gpt-4o"},
    )

    user_proxy = ConversableAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
    )

    # MemoryStream enables pub/sub event handling
    stream = MemoryStream()

    @stream.subscribe("message")
    async def on_message(event):
        print(f"[{event.sender}]: {event.content}")

    result = await user_proxy.a_initiate_chat(
        assistant,
        message="Solve this problem: ...",
        stream=stream,
    )

asyncio.run(main())
```

## A2A Protocol Support (v0.11.x+)

AG2 now supports Google's Agent-to-Agent (A2A) 1.0 protocol for cross-framework agent communication:

```python
from autogen.interop import A2AServer, A2AClient

# Expose an AG2 agent via A2A
agent = AssistantAgent("research_agent", llm_config={"model": "gpt-4o"})
server = A2AServer(agent, port=8080)
await server.start()

# Connect to an external A2A agent (any framework)
client = A2AClient("http://external-agent:8080")
response = await client.send_task("Analyse this dataset", attachments=[...])
print(response.result)
```

## AG2 CLI (v0.11.x+)

```bash
# Initialise a new AG2 project
ag2 init my-agent-project

# Run an agent directly from CLI
ag2 run --agent research_agent --task "Research latest AI news"

# Start the web UI
ag2 ui
```

## QuickResearchTool (v0.11.x+)

New built-in tool for parallel web research:

```python
from autogen.tools import QuickResearchTool
from autogen import AssistantAgent

research_tool = QuickResearchTool(
    max_sources=5,
    parallel=True,
)

agent = AssistantAgent(
    name="researcher",
    llm_config={"model": "gpt-4o"},
    tools=[research_tool],
)

result = await agent.a_run("What are the latest developments in quantum computing?")
```

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.11.5 | April 5, 2026 | Security fixes (CVE-2026-23745, CVE-2026-23950, CVE-2026-24842); `QuickResearchTool` for parallel research; Gemini client streaming; `RemyxCodeExecutor` for containerised execution |
| 0.11.0 | March 2026 | `autogen.beta` event-driven redesign (MemoryStream pub/sub); A2A 1.0 protocol; AG2 CLI (`ag2 init`, `ag2 run`); `A2UIAgent`; `GroupToolExecutor` async handler |
| 0.3.2 | November 2025 | Previous documented version |

## Resources

*   [Official Documentation](https://github.com/ag2ai/ag2)
*   [GitHub Repository](https://github.com/ag2ai/ag2)

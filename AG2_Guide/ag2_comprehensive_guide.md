# AG2 (AutoGen) Comprehensive Guide

**Version:** 0.3.2
**Last Updated:** November 2025
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

## Resources

*   [Official Documentation](https://github.com/ag2ai/ag2)
*   [GitHub Repository](https://github.com/ag2ai/ag2)

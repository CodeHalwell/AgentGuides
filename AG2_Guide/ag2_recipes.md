# AG2 (AutoGen) Recipes

**Version:** 0.3.2
**Last Updated:** November 2025
**Focus:** Ready-to-Use Examples

## Overview

This collection of recipes provides copy-pasteable code for common AG2 use cases.

## Recipe 1: Simple Chatbot

A basic agent that chats with the user.

```python
from ag2 import Agent, UserProxyAgent

assistant = Agent(name="assistant", system_message="You are a helpful AI assistant.")
user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="ALWAYS")

user_proxy.initiate_chat(assistant, message="Hello!")
```

## Recipe 2: Code Execution Agent

An agent that can write and execute Python code.

```python
from ag2 import Agent, UserProxyAgent

assistant = Agent(
    name="coder",
    system_message="You are a python expert. Write code to solve problems."
)
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA stock price YTD.")
```

## Recipe 3: Two-Agent Debate

Two agents debating a topic.

```python
from ag2 import Agent, GroupChat, GroupChatManager

pro_agent = Agent(name="Pro", system_message="Argue in favor of remote work.")
con_agent = Agent(name="Con", system_message="Argue against remote work.")
judge = Agent(name="Judge", system_message="Evaluate the debate.")

groupchat = GroupChat(agents=[pro_agent, con_agent, judge], messages=[], max_round=6)
manager = GroupChatManager(groupchat=groupchat)

pro_agent.initiate_chat(manager, message="Let's debate remote work.")
```

## Recipe 4: Tool Use with Calculator

An agent that uses a calculator tool.

```python
from ag2 import Agent, UserProxyAgent

def calculator(a: int, b: int, op: str) -> int:
    if op == "+": return a + b
    elif op == "-": return a - b
    elif op == "*": return a * b
    elif op == "/": return a // b
    else: return 0

assistant = Agent(name="assistant", system_message="Use the calculator tool.")
assistant.register_function(function_map={"calculator": calculator})

user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER")

user_proxy.initiate_chat(assistant, message="Calculate 123 * 456")
```

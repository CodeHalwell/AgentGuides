---
title: "AG2 Classic - Comprehensive Technical Guide"
description: "Beginner-to-expert guide to the AG2 Classic `import autogen` API — ConversableAgent, LLMConfig, GroupChat, nested and sequential chats, tools and code execution. Not Microsoft AutoGen 0.4+."
framework: "autogen"
language: "python"
---

> **This page covers AG2 Classic** — the `import autogen` API (`ConversableAgent`, `GroupChat`, `initiate_chat`) maintained by the AG2 community fork and published on PyPI as `autogen` (and as `ag2` up to 0.14). It is **not** Microsoft AutoGen 0.4+ (`autogen_agentchat`), which is covered on the [AgentChat & Core page](/autogen-guide/python/autogen_agentchat_guide/). Code here was checked against `autogen` %%version=autogen%%, the Classic line, which is in maintenance mode (security and critical fixes only). AG2's current framework is `ag2` 1.x — see the [AG2 guide](/ag2-guide/).

**A beginner-to-expert tutorial for multi-agent systems with the AG2 Classic API.**

---

## Background: AutoGen, AG2 and AG2 Classic

- **AutoGen 0.2** (`pyautogen`, `import autogen`) was developed at Microsoft Research in 2023–2024.
- In **November 2024** AutoGen's original creators forked the 0.2 line as **AG2** under the `ag2ai` organisation (Apache 2.0, open governance), keeping the `import autogen` API. Microsoft was not rebranding its project: it continued `microsoft/autogen` separately and shipped the incompatible 0.4 rewrite in January 2025, which it has since archived in favour of Microsoft Agent Framework.
- In **July 2026** AG2 1.0 replaced this API in the `ag2` package with a new async framework (`import ag2`). The API on this page continues as **AG2 Classic** in the `autogen` package.

The API is broadly compatible with AutoGen 0.2 code; see the [AutoGen 0.2 → AG2 Classic migration guide](/autogen-guide/python/autogen_to_ag2_migration_guide/).

---

## Table of Contents

1. [Introduction and Overview](#introduction-and-overview)
2. [Core Fundamentals](#core-fundamentals)
3. [Simple Agents](#simple-agents)
4. [Multi-Agent Systems](#multi-agent-systems)
5. [Tools and Function Integration](#tools-and-function-integration)
6. [Structured Output and Data Validation](#structured-output-and-data-validation)
7. [Agentic Patterns and Workflows](#agentic-patterns-and-workflows)
8. [Memory Systems and Context Management](#memory-systems-and-context-management)
9. [Context Engineering](#context-engineering)
10. [Later AG2 Classic features](#later-ag2-classic-features)
11. [Migration from AutoGen 0.2](#migration-from-autogen-to-ag2)

---

## Introduction and Overview

### What is AG2 Classic?

AG2 Classic is a framework for building applications from **conversable agents** that message each other to solve tasks.

**Key Characteristics:**

- **Multi-Agent Orchestration**: Two-agent chats, group chats with speaker selection, nested and sequential chats, swarms.
- **Human-in-the-Loop**: `human_input_mode` lets a human approve, steer or take over turns.
- **LLM Provider Flexibility**: OpenAI, Azure OpenAI, Anthropic, Gemini, Bedrock, Mistral, Groq, Ollama and others through `LLMConfig` entries.
- **Code Execution**: Run generated Python and shell code locally or in Docker.
- **Tool Integration**: Register Python functions as tools with `register_function`.

### Classic vs the alternatives

| | AG2 Classic (this page) | AG2 1.x | Microsoft AutoGen 0.4–0.7 |
|---|---|---|---|
| Install | `pip install "autogen[openai]"` | `pip install "ag2[openai]"` | `pip install autogen-agentchat "autogen-ext[openai]"` |
| Import | `import autogen` | `import ag2` | `import autogen_agentchat` |
| Status | Maintenance mode | Active | Archived |

---

## Core Fundamentals

### Installation and Setup

#### Python Version Requirements

AG2 Classic requires **Python 3.10 or later**. Verify your Python version:

```bash
python --version
```

#### Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install "autogen[openai]"      # AG2 Classic with the OpenAI client
# or, equivalently for existing projects, the last Classic release of ag2:
# pip install "ag2[openai]<1.0"
```

Do **not** install `autogen-agentchat` or `pyautogen` for this API: both install Microsoft AutoGen 0.4+, which uses different imports. And do not upgrade to `ag2>=1.0`, which no longer ships the `autogen` package.

#### Troubleshooting Installation Issues

**Issue: `ModuleNotFoundError: No module named 'autogen'`**

**Solution**: You have `ag2>=1.0`, `autogen-agentchat` or `pyautogen` installed instead of Classic. Run `pip show autogen ag2` and install `autogen` (or `ag2<1.0`) in a fresh virtual environment.

### Configuring LLM Providers

#### Understanding the Configuration List

In modern AutoGen, LLM configuration is managed via a list of dictionaries passed to an agent's `llm_config` parameter. This approach simplifies setup and allows for easy fallback between different models or API keys.

#### Method 1: Single Model Configuration

The most common method is to define the configuration directly in your code.

```python
import autogen

config_list = [
    {
        "model": "gpt-4",
        "api_key": "sk-your-api-key-here",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "cache_seed": 42,  # Use a seed for caching and reproducibility
}

# Use with an agent
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)
```

#### Method 2: File-Based Configuration (OAI_CONFIG_LIST.json)

For better security and portability, store configurations in a JSON file. By default, AutoGen looks for a file named `OAI_CONFIG_LIST.json` in the working directory.

**`OAI_CONFIG_LIST.json` file:**
```json
[
    {
        "model": "gpt-4",
        "api_key": "sk-your-key-here"
    },
    {
        "model": "gpt-3.5-turbo",
        "api_key": "sk-your-other-key-here"
    }
]
```

**Load it in your code:**
```python
import autogen

# AutoGen automatically finds and loads this file if the path is not specified
config_list = autogen.llm_config.config.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4"] # Optional: filter for specific models
    }
)

llm_config = {"config_list": config_list}
```

#### Method 3: Environment Variables

You can also load credentials from environment variables.

```bash
export OPENAI_API_KEY="sk-your-api-key"
```

**`OAI_CONFIG_LIST.json` file (without api_key):**
```json
[
    {
        "model": "gpt-4"
    }
]
```
AutoGen will automatically use the `OPENAI_API_KEY` environment variable.

#### Method 4: Azure OpenAI Configuration

```python
config_list_azure = [
    {
        "model": "your-deployment-name", # The deployment name you chose for your model
        "api_key": "your-azure-api-key",
        "base_url": "https://your-resource-name.openai.azure.com",
        "api_type": "azure",
        "api_version": "2024-02-01",
    }
]
```

#### Method 5: Anthropic Claude Configuration

```python
config_list_claude = [
    {
        "model": "claude-3-5-sonnet-20240620",
        "api_key": "your-anthropic-api-key",
        "api_type": "anthropic",
    }
]
```

#### Method 6: Google Gemini Configuration

```python
config_list_gemini = [
    {
        "model": "gemini-pro",
        "api_key": "your-google-api-key",
        "api_type": "google",
    }
]
```

#### Method 7: Local LLMs with Ollama

```python
config_list_ollama = [
    {
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama", # The API key can be any string
    }
]
```


### Core Agent Classes

#### ConversableAgent: The Foundation

The `ConversableAgent` is the base class for all agents. It handles message passing, LLM communication, and reply generation.

**Minimal Example:**

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Create a conversable agent
agent = autogen.ConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

print("Agent created:", agent.name)
```

**Key Parameters:**

```python
agent = autogen.ConversableAgent(
    name="assistant",
    system_message="You are helpful.",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Options: NEVER, TERMINATE, ALWAYS
    max_consecutive_auto_reply=10,
    code_execution_config=False,  # or a dictionary for configuration
    function_map=None, # Deprecated, use register_for_llm and register_for_execution
    description="An assistant agent"
)
```

**Complete Working Example:**

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Create agents for code discussion
coder = autogen.ConversableAgent(
    name="coder",
    system_message="You are an expert Python developer. Write clean, well-documented code.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

reviewer = autogen.ConversableAgent(
    name="reviewer",
    system_message="You are a senior code reviewer. Analyse code for quality, security, and best practices. Do not write code.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Start conversation
chat_result = reviewer.initiate_chat(
    recipient=coder,
    message="Write a Python function that validates email addresses using regex.",
    max_turns=2,
    summary_method="last_msg"
)

# Print summary
print(chat_result.summary)
```

#### AssistantAgent: Pre-configured for Task Solving

`AssistantAgent` is a subclass of `ConversableAgent` that is optimized for task solving, with default system prompts and configurations.

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Create assistant
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a problem-solving assistant. Provide clear, step-by-step solutions."
)

print(f"Assistant '{assistant.name}' created.")
```

#### UserProxyAgent: Human-in-the-Loop and Code Execution

`UserProxyAgent` represents a human user or acts as a code executor in automated workflows.

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Create assistant
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

# Create user proxy with code execution
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Initiate task
user_proxy.initiate_chat(
    assistant,
    message="Create a Python script that reads CSV data and generates summary statistics."
)
```

**Human Input Modes:**

- `"NEVER"`: Never ask for human input; agent operates autonomously.
- `"TERMINATE"`: Ask for input only when a termination condition is met.
- `"ALWAYS"`: Ask for human input after every agent reply.

**Code Execution Configuration:**

```python
# Local execution (no Docker)
code_config_local = {
    "work_dir": "coding",
    "use_docker": False,
}

# Docker execution (isolated and secure)
code_config_docker = {
    "work_dir": "coding",
    "use_docker": True,  # Docker must be installed and running
}
```


---

## Simple Agents

### Creating Your First Agent

#### The Absolute Minimum Example

```python
import autogen

# 1. Configure LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key"
    }
]
llm_config = {"config_list": config_list}


# 2. Create agent
agent = autogen.ConversableAgent(
    name="my_agent",
    llm_config=llm_config
)

# 3. Use it
print(f"Agent '{agent.name}' is ready!")
```

#### Single-Agent Task Solving

A single agent receiving and responding to messages:

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Create problem solver
solver = autogen.ConversableAgent(
    name="problem_solver",
    system_message="You are an expert problem solver. Break down complex problems into manageable steps.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Initiate a chat with itself to solve a problem
chat_result = solver.initiate_chat(
    recipient=solver,
    message="How would you optimise a slow Python function that processes large datasets?",
    max_turns=1
)
```

#### Agent Roles and System Messages

**Example 1: Researcher Agent**

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

researcher = autogen.ConversableAgent(
    name="researcher",
    system_message="""You are an expert researcher specializing in data science and machine learning.
Your role is to:
1. Research topics thoroughly
2. Find credible sources
3. Summarize findings concisely
4. Highlight key insights and trends""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)
```

**Example 2: Technical Writer Agent**

```python
writer = autogen.ConversableAgent(
    name="technical_writer",
    system_message="""You are a senior technical writer known for creating clear, concise documentation.
Your expertise includes:
- Writing for diverse audiences (beginners to experts)
- Creating step-by-step tutorials
- Explaining complex concepts simply
- Using examples effectively
- Reviewing documentation for clarity""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)
```

**Example 3: Code Quality Agent**

```python
code_reviewer = autogen.ConversableAgent(
    name="code_reviewer",
    system_message="""You are a meticulous code quality assurance specialist.
When reviewing code, you:
1. Check for bugs and logic errors
2. Verify adherence to coding standards
3. Assess performance and scalability
4. Suggest improvements with examples
5. Consider security implications
Do NOT write code; only review and critique.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)
```

### Agent Configuration Deep Dive

#### Human Input Mode Patterns

**Pattern 1: Completely Autonomous (NEVER)**

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

autonomous_agent = autogen.ConversableAgent(
    name="autonomous",
    system_message="You solve problems independently.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=15
)
```

Use case: Automated systems, background processing, cost-sensitive tasks.

**Pattern 2: Termination-Based (TERMINATE)**

```python
interactive_agent = autogen.ConversableAgent(
    name="interactive",
    system_message="You work collaboratively with humans.",
    llm_config=llm_config,
    human_input_mode="TERMINATE",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").upper()
)
```

Use case: Decision-making workflows, human oversight required.

**Pattern 3: Always Interactive (ALWAYS)**

```python
manual_approval_agent = autogen.ConversableAgent(
    name="manual",
    system_message="You require human approval for each action.",
    llm_config=llm_config,
    human_input_mode="ALWAYS"
)
```

#### Temperature and Creativity Settings

**Conservative (Deterministic) Configuration**

```python
config_list_conservative = autogen.llm_config.config.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={"model": ["gpt-4"]}
)
llm_config_conservative = {
    "config_list": config_list_conservative,
    "temperature": 0.0,
}

fact_checker = autogen.ConversableAgent(
    name="fact_checker",
    system_message="You verify facts and provide accurate information.",
    llm_config=llm_config_conservative
)
```

**Creative Configuration**

```python
config_list_creative = autogen.llm_config.config.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={"model": ["gpt-4"]}
)
llm_config_creative = {
    "config_list": config_list_creative,
    "temperature": 1.0,
    "top_p": 0.95
}

brainstorm_agent = autogen.ConversableAgent(
    name="brainstorm",
    system_message="You generate diverse, unconventional ideas.",
    llm_config=llm_config_creative
)
```

#### Token Limits and Cost Control

```python
# Minimal responses (cost-conscious)
llm_config_minimal = {
    "config_list": config_list,
    "max_tokens": 500
}

# Comprehensive responses
llm_config_comprehensive = {
    "config_list": config_list,
    "max_tokens": 4000
}
```

### Code Execution Environments

#### Local Execution (No Docker)

**Advantages**: Fast, simple setup, direct filesystem access
**Disadvantages**: Potential security risks, full system access

```python
import autogen
from autogen.coding import LocalCommandLineCodeExecutor

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir="./code_execution", timeout=60),
        "last_n_messages": "auto",
    },
    human_input_mode="NEVER",
)
```

**Example: Execute Python Code Locally**

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="Write Python code to solve problems."
)

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={"executor": autogen.coding.LocalCommandLineCodeExecutor(work_dir="code")},
    human_input_mode="NEVER"
)

executor.initiate_chat(
    assistant,
    message="Write a Python script that calculates Fibonacci numbers up to 10."
)
```

#### Docker Execution (Isolated and Secure)

**Advantages**: Secure, sandboxed, reproducible
**Disadvantages**: Slightly slower, requires Docker installation

```python
import autogen
from autogen.coding import DockerCommandLineCodeExecutor

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={
        "executor": DockerCommandLineCodeExecutor(image="python:3.12-slim", work_dir="./workspace", timeout=120),
    },
    human_input_mode="NEVER"
)
```

**Example: Execute Code in Docker Container**

```python
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={
        "executor": autogen.coding.DockerCommandLineCodeExecutor(image="python:3.12-slim", work_dir="./code"),
    },
    human_input_mode="NEVER"
)

executor.initiate_chat(
    assistant,
    message="Create a script using matplotlib to plot a sine wave and save it as 'sine.png'."
)
```

#### Custom Code Executor

```python
from autogen.coding import LocalCommandLineCodeExecutor
import autogen

# Create custom executor with specific configuration
custom_executor = LocalCommandLineCodeExecutor(
    work_dir="./custom_workspace",
    timeout=90,
)

executor_agent = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={"executor": custom_executor},
    human_input_mode="NEVER"
)
```

### Function Calling and Tool Use

#### Simple Function Registration

In modern AutoGen, you register functions (tools) to two components: the agent that calls the tool (`register_for_llm`) and the agent that executes it (`register_for_execution`).

```python
from datetime import datetime
from typing import Annotated
import autogen

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Define tool function with type hints
def get_current_time(timezone: Annotated[str, "Timezone (e.g., 'UTC', 'EST')"]) -> str:
    """Get the current time in a specified timezone."""
    # Note: This is simplified; use pytz for real implementation
    return f"Current time in {timezone}: {datetime.now().strftime('%H:%M:%S')}"

# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You can access current time in different timezones. Use the provided tools.",
    llm_config=llm_config
)

# The user_proxy agent will execute the function
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False, # No code execution needed for this example
)

# Register the function for the LLM to find and the executor to run
assistant.register_for_llm(name="get_current_time", description="Get the current time in a specified timezone.")(get_current_time)
user_proxy.register_for_execution(name="get_current_time")(get_current_time)


# Use it
user_proxy.initiate_chat(
    assistant,
    message="What is the current time in EST and UTC?",
)
```

#### Complex Function with Multiple Parameters

```python
from typing import Annotated, List
import autogen
import statistics

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

def calculate_statistics(
    numbers: Annotated[List[float], "List of numbers to analyse"],
    include_median: Annotated[bool, "Whether to include median (default: True)"] = True,
    decimal_places: Annotated[int, "Decimal places for rounding (default: 2)"] = 2
) -> dict:
    """Calculate statistics for a list of numbers."""
    stats = {
        "mean": round(statistics.mean(numbers), decimal_places),
        "min": round(min(numbers), decimal_places),
        "max": round(max(numbers), decimal_places),
        "count": len(numbers)
    }
    
    if include_median:
        stats["median"] = round(statistics.median(numbers), decimal_places)
    
    return stats

assistant = autogen.AssistantAgent(
    name="analyst",
    system_message="You analyse numerical data using available tools.",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Register the function for both agents
assistant.register_for_llm(name="calculate_statistics", description="Calculate mean, median, min, max for numerical data")(calculate_statistics)
user_proxy.register_for_execution(name="calculate_statistics")(calculate_statistics)

user_proxy.initiate_chat(
    assistant,
    message="Calculate statistics for these numbers: 10, 20, 30, 40, 50",
)
```

### Human-in-the-Loop Patterns

#### Interactive Decision Making

```python
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

analyst = ConversableAgent(
    name="analyst",
    system_message="You analyse data and provide recommendations.",
    llm_config=llm_config
)

human = UserProxyAgent(
    name="human",
    human_input_mode="TERMINATE",  # Get human input when needed
    is_termination_msg=lambda msg: "APPROVED" in msg.get("content", "").upper()
)

# Start conversation - will prompt for human input
human.initiate_chat(
    analyst,
    message="Analyse this customer data and recommend actions. End with 'APPROVED' when done."
)
```

#### Approval Workflow

```python
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

proposer = ConversableAgent(
    name="proposer",
    system_message="You propose solutions and await approval.",
    llm_config=llm_config
)

approver = UserProxyAgent(
    name="approver",
    human_input_mode="ALWAYS",  # Ask for input on every turn
    system_message="Review proposals and provide approval or feedback."
)

approver.initiate_chat(
    proposer,
    message="Propose a solution to reduce database query times by 50%."
)
```

---

## Multi-Agent Systems

### Two-Agent Conversations

#### Basic Two-Agent Setup

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

agent_a = ConversableAgent(
    name="alice",
    system_message="You are Alice, a curious learner.",
    llm_config=llm_config
)

agent_b = ConversableAgent(
    name="bob",
    system_message="You are Bob, an experienced teacher.",
    llm_config=llm_config
)

# Initiate conversation
response = agent_a.initiate_chat(
    agent_b,
    message="Explain machine learning to me like I'm 10 years old."
)

response.process()
print(response.summary)
```

#### Passing Conversation in Reverse

```python
# Continue conversation with other agent initiating
response = agent_b.run(
    recipient=agent_a,
    message="Now explain deep learning.",
    max_turns=5
)

response.process()
print("Next topic summary:", response.summary)
```

### Group Chat: Multiple Agents

#### Creating a Group Chat

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Create specialized agents
planner = ConversableAgent(
    name="planner",
    system_message="You plan project tasks and timelines.",
    llm_config=llm_config
)

developer = ConversableAgent(
    name="developer",
    system_message="You implement planned tasks using code.",
    llm_config=llm_config
)

tester = ConversableAgent(
    name="tester",
    system_message="You test implementations and report issues.",
    llm_config=llm_config
)

# Create group chat
groupchat = GroupChat(
    agents=[planner, developer, tester],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"  # LLM decides next speaker
)

# Create manager
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Start conversation
planner.initiate_chat(
    manager,
    message="We need to build a REST API for a todo application. Let's plan this."
)
```

#### Complete Working Example: Content Creation Team

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Research agent
researcher = ConversableAgent(
    name="researcher",
    system_message="You research topics deeply and provide well-researched information.",
    llm_config=llm_config
)

# Writer agent
writer = ConversableAgent(
    name="writer",
    system_message="You write clear, engaging content based on research.",
    llm_config=llm_config
)

# Editor agent
editor = ConversableAgent(
    name="editor",
    system_message="You edit content for clarity, consistency, and quality. Provide feedback, don't rewrite.",
    llm_config=llm_config
)

# Create group chat
groupchat = GroupChat(
    agents=[researcher, writer, editor],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Initiate project
researcher.initiate_chat(
    manager,
    message="Create an article on 'The Evolution of Machine Learning in Healthcare' suitable for technical professionals."
)
```

### Speaker Selection Strategies

#### Strategy 1: Automatic (LLM-based Decision)

```python
# LLM decides who should speak next based on context
groupchat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method="auto"  # Intelligent selection
)
```

#### Strategy 2: Round-Robin (Sequential)

```python
# Agents speak in turn
groupchat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method="round_robin"  # A1 → A2 → A3 → A1...
)
```

#### Strategy 3: Random Selection

```python
groupchat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method="random"
)
```

#### Strategy 4: Manual Selection with Custom Function

```python
def custom_speaker_selector(last_speaker, agents, messages):
    """Custom logic to select next speaker."""
    # Example: Prefer 'expert' agent after any other agent
    for agent in agents:
        if agent.name == "expert" and last_speaker.name != "expert":
            return agent
    
    # Fallback: round-robin
    import random
    return random.choice([a for a in agents if a != last_speaker])

groupchat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method="manual",
    select_speaker_message_function=custom_speaker_selector
)
```

#### Preventing Repetitive Speakers

```python
groupchat = GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method="auto",
    allow_repeat_speaker=False  # No consecutive repeats
)
```

### Nested Chats: Hierarchical Conversations

#### Basic Nested Chat

```python
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Main agents
task_planner = ConversableAgent(
    name="planner",
    system_message="You plan high-level tasks.",
    llm_config=llm_config
)

implementer = UserProxyAgent(
    name="implementer",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER"
)

# Nested review agents
reviewer = ConversableAgent(
    name="reviewer",
    system_message="You review code for quality and suggest improvements.",
    llm_config=llm_config
)

improver = ConversableAgent(
    name="improver",
    system_message="You refactor code based on feedback.",
    llm_config=llm_config
)

# Register nested chats
implementer.register_nested_chats(
    chat_queue=[
        {
            "recipient": reviewer,
            "message": "Review this code for quality",
            "summary_method": "last_msg",
            "max_turns": 2
        },
        {
            "recipient": improver,
            "message": "Improve the code based on review",
            "summary_method": "reflection_with_llm",
            "max_turns": 2
        }
    ],
    trigger=task_planner,  # run the review chain whenever the planner messages the implementer
)

# Start main conversation (nested chats trigger automatically)
task_planner.initiate_chat(
    implementer,
    message="Create a Python script to sort a list of dictionaries by multiple keys."
)
```

### Sequential Chats: Multi-Stage Workflows

```python
from autogen import ConversableAgent, initiate_chats, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Stage 1 agent
ideator = ConversableAgent(
    name="ideator",
    system_message="You brainstorm innovative product ideas.",
    llm_config=llm_config
)

# Stage 2 agent
validator = ConversableAgent(
    name="validator",
    system_message="You validate ideas for feasibility and market potential.",
    llm_config=llm_config
)

# Stage 3 agent
planner = ConversableAgent(
    name="planner",
    system_message="You create implementation plans for validated ideas.",
    llm_config=llm_config
)

# Coordinator
coordinator = ConversableAgent(
    name="coordinator",
    system_message="You coordinate the idea development process.",
    llm_config=llm_config
)

# Define chat sequence
chat_sequence = [
    {
        "sender": coordinator,
        "recipient": ideator,
        "message": "Generate 3 innovative ideas for an e-commerce startup.",
        "max_turns": 2,
        "summary_method": "last_msg"
    },
    {
        "sender": coordinator,
        "recipient": validator,
        "message": "Validate these ideas for market potential.",
        "max_turns": 2,
        "summary_method": "reflection_with_llm",
        "carryover": "Use the ideas from the previous stage."
    },
    {
        "sender": coordinator,
        "recipient": planner,
        "message": "Create detailed implementation plans.",
        "max_turns": 2,
        "summary_method": "reflection_with_llm",
        "carryover": "Build on the validated ideas."
    }
]

# Execute sequential chats
results = initiate_chats(chat_sequence)

for i, result in enumerate(results):
    print(f"\n=== Stage {i+1} ===")
    print(result.summary)
```

### Agent State Management

#### Sharing Context with ContextVariables

```python
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.group.context_variables import ContextVariables

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Create shared context
context = ContextVariables()
context.set("project_name", "E-commerce Platform")
context.set("budget_usd", 50000)
context.set("deadline_months", 6)

# Create agents with context
project_manager = ConversableAgent(
    name="pm",
    system_message="Access context: project_name, budget_usd, deadline_months",
    llm_config=llm_config,
    context_variables=context
)

lead_developer = ConversableAgent(
    name="dev_lead",
    system_message="Develop with awareness of budget and timeline constraints.",
    llm_config=llm_config,
    context_variables=context
)

# Access context within agents
print(f"Project: {context.get('project_name')}")
print(f"Budget: ${context.get('budget_usd')}")
print(f"Timeline: {context.get('deadline_months')} months")

# Use agents
project_manager.initiate_chat(
    lead_developer,
    message="Design the architecture considering our constraints."
)
```

---

## Tools and Function Integration

### Function Registration with @register_function

[Content continues with extensive tool integration examples...]

(Continuing in next part due to length...)

---

## Structured Output and Data Validation

### JSON Mode and Pydantic Models

To get structured output, you can instruct the LLM via the system message to return JSON, and then validate it with a Pydantic model.

```python
from pydantic import BaseModel, Field
from typing import List
import autogen
import json

config_list = autogen.llm_config.config.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Define structured response schema
class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    in_stock: bool = Field(description="Whether in stock")

class ProductRecommendation(BaseModel):
    products: List[Product]
    reason: str = Field(description="Why these products are recommended")

# Instruct the agent to use the schema in the system message
system_prompt = f"""You are a helpful product recommender.
Please respond with a JSON object that conforms to the following Pydantic schema:
{ProductRecommendation.model_json_schema()}
"""

# Create agent with the schema in its prompt
agent = autogen.ConversableAgent(
    name="recommender",
    system_message=system_prompt,
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    llm_config=False,
)

# Initiate the chat
chat_result = user_proxy.initiate_chat(
    agent,
    message="Recommend laptops suitable for machine learning development.",
    max_turns=1
)

# Parse structured response
response_text = chat_result.summary
try:
    # The response from the LLM should be a JSON string
    recommendations = ProductRecommendation.model_validate_json(response_text)
    for product in recommendations.products:
        print(f"- {product.name}: ${product.price} (In stock: {product.in_stock})")
    print(f"\nReason: {recommendations.reason}")
except Exception as e:
    print(f"Failed to validate JSON response: {e}")
    print(f"Raw response: {response_text}")

```

---

## Agentic Patterns and Workflows

### ReAct (Reasoning and Acting) Implementation

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Agent that thinks and acts
react_agent = AssistantAgent(
    name="react_agent",
    system_message="""You solve problems using the ReAct (Reasoning + Acting) pattern:
1. Thought: Analyse the problem
2. Action: Choose a tool/action
3. Observation: Observe results
4. Repeat until solved

Clearly label each thought, action, and observation.""",
    llm_config=llm_config
)

executor = UserProxyAgent(
    name="executor",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER"
)

executor.initiate_chat(
    react_agent,
    message="Calculate the optimal dimensions for a rectangular garden with area 100 m² and minimal perimeter."
)
```

### Self-Refinement Loops

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Initial generator
generator = ConversableAgent(
    name="generator",
    system_message="Generate solutions to problems.",
    llm_config=llm_config
)

# Critic
critic = ConversableAgent(
    name="critic",
    system_message="Critique solutions and suggest improvements. Identify weaknesses and edge cases.",
    llm_config=llm_config
)

# Refiner
refiner = ConversableAgent(
    name="refiner",
    system_message="Refine solutions based on critical feedback.",
    llm_config=llm_config
)

# Start self-refinement loop
print("=== Initial Generation ===")
response = generator.run(
    recipient=critic,
    message="Design an algorithm to detect anomalies in time series data.",
    max_turns=2
)

print("\n=== Refinement ===")
response = critic.run(
    recipient=refiner,
    message=f"Improve this based on the previous discussion: {response.summary}",
    max_turns=2
)
```

---

## Memory Systems and Context Management

### Conversation History Management

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

agent1 = ConversableAgent(
    name="agent1",
    system_message="You remember and reference previous conversations.",
    llm_config=llm_config
)

agent2 = ConversableAgent(
    name="agent2",
    system_message="You build on previous context.",
    llm_config=llm_config
)

# First conversation
response1 = agent2.run(
    recipient=agent1,
    message="We're building a machine learning model for customer churn prediction.",
    max_turns=3
)

# Drive the run to completion, then read its history
response1.process()
print("Previous conversation:")
for msg in response1.messages:
    print(f"{msg['name']}: {msg['content'][:100]}...")

# Second conversation (with context from first)
response2 = agent2.run(
    recipient=agent1,
    message="Given our earlier discussion, what features should we prioritise?",
    max_turns=3
)
```

---

## Context Engineering

### System Message Design

#### Role Definition Example

```python
system_messages = {
    "domain_expert": """You are a domain expert with 20+ years of experience.
Your role:
- Provide authoritative answers
- Explain complex concepts clearly
- Cite relevant research and standards
- Identify gaps in understanding""",
    
    "brainstormer": """You are a creative brainstormer.
Your role:
- Generate diverse ideas
- Challenge assumptions
- Think outside the box
- Build on others' ideas""",
    
    "skeptic": """You are a thoughtful sceptic.
Your role:
- Question assumptions
- Identify potential problems
- Consider edge cases
- Play devil's advocate"""
}
```

### Few-Shot Prompting

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

few_shot_system_message = """You are a data analyst. Here are examples of how to analyse problems:

Example 1:
- Data: Sales by region for Q1
- Analysis: Calculate growth rate, identify top performers
- Output: Summary table with rankings

Example 2:
- Data: Customer satisfaction scores
- Analysis: Group by segment, calculate averages, identify trends
- Output: Visualisation with insights

Now, analyse new data following this pattern."""

analyst = ConversableAgent(
    name="analyst",
    system_message=few_shot_system_message,
    llm_config=llm_config
)
```

---

## Later AG2 Classic features

Earlier drafts of this section described "2025 enhancements" such as `context_truncation_strategy=` on `ConversableAgent` and an `autogen.telemetry.TelemetryConfig` class. Neither exists in the Classic package, and AutoGen Studio is a Microsoft AutoGen 0.4+ tool rather than part of AG2, so that material has been removed. The features below are real in `autogen` %%version=autogen%%.

### `LLMConfig`

`LLMConfig` replaced bare `config_list` dicts as the preferred way to configure models:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig({"api_type": "openai", "model": "gpt-4.1-mini"}, temperature=0.2)
# or: LLMConfig.from_json(path="OAI_CONFIG_LIST.json")

agent = ConversableAgent("assistant", system_message="You are helpful.", llm_config=llm_config)
```

### `run()` and group-chat patterns

`ConversableAgent.run(...)` and `autogen.agentchat.run_group_chat(...)` return a `RunResponse` you can iterate for events, an alternative to `initiate_chat`. Group orchestration is described with patterns from `autogen.agentchat.group.patterns` (`AutoPattern`, `RoundRobinPattern`, `DefaultPattern`, …) plus handoffs and `ContextVariables`:

```python
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns import AutoPattern

llm_config = LLMConfig({"api_type": "openai", "model": "gpt-4.1-mini"})
planner = ConversableAgent("planner", system_message="Plan the work.", llm_config=llm_config)
writer = ConversableAgent("writer", system_message="Write the result.", llm_config=llm_config)
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)

pattern = AutoPattern(
    initial_agent=planner,
    agents=[planner, writer],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)
# response = run_group_chat(pattern=pattern, messages="Draft a release note for v2.", max_rounds=8)
# response.process()   # print events to the console
```

Swarm helpers moved to `autogen.agentchat.contrib.swarm_agent` in 0.13.0.

### A2A and MCP

- `autogen.a2a.A2aAgentServer(agent, url=...)` serves a Classic agent over A2A; `A2aRemoteAgent(url=..., name=...)` wraps a remote A2A agent as a participant (extra: `autogen[a2a]`).
- `autogen.mcp.create_toolkit(session)` turns an MCP client session into tools (extra: `autogen[mcp]`).

### `autogen.beta`

From 0.11 Classic shipped `autogen.beta`, a preview of the event-driven API. It became the top-level `ag2` package in AG2 1.0; new work should use that directly — see the [AG2 guide](/ag2-guide/).

### Cost tracking

`autogen.gather_usage_summary([agent, ...])` totals token usage and cost across agents after a run; `ChatResult.cost` holds per-chat figures.

---

## Migration from AutoGen to AG2

This section is about moving **AutoGen 0.2** code (`pyautogen`, `import autogen`) onto **AG2 Classic**. Code written for Microsoft AutoGen 0.4+ (`autogen_agentchat`) uses a different API and does not migrate this way.

### Steps

1. **Change the distribution, not the imports.** Uninstall `pyautogen` (which now installs Microsoft's `autogen-agentchat`) and install `autogen` (AG2 Classic), or `ag2<1.0`:

   ```bash
   pip uninstall -y pyautogen
   pip install "autogen[openai]"
   ```

2. **Keep `import autogen`.** Classic installs the `autogen` module, so most 0.2 code runs unchanged.
3. **Prefer `LLMConfig`.** Plain `{"config_list": [...]}` dicts still work; `LLMConfig({...})` or `LLMConfig.from_json(path="OAI_CONFIG_LIST")` is the current form.
4. **Pass code executors as objects.** `code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir=...)}` (or `DockerCommandLineCodeExecutor`); mixing `executor` with `work_dir`/`use_docker` keys raises `ValueError`.
5. **Move swarm imports.** Since 0.13.0, `SwarmAgent` and `initiate_swarm_chat` live in `autogen.agentchat.contrib.swarm_agent`; new code should use group-chat patterns (`autogen.agentchat.group.patterns`).

### Where to go after Classic

Classic is in maintenance mode. For new AG2 work, port to `ag2` 1.x (`import ag2`); the [AG2 guide](/ag2-guide/ag2_comprehensive_guide/#ag2-classic-and-migrating-from-0x) has a porting checklist. For the full 0.2 → Classic walkthrough, see the [migration guide](./autogen_to_ag2_migration_guide/).

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| autogen 0.14.1 (AG2 Classic) | October 21, 2026 | Reframed as an AG2 Classic page (it never described Microsoft AutoGen 0.4+). Removed the false "Microsoft AutoGen was rebranded as AG2" narrative, the invented `context_truncation_strategy`, `TelemetryConfig`, `get_cost_stats` and AutoGen Studio "2025" material; fixed `LLMConfig.from_json` calls (keyword `path=`), code-executor configuration and a nested-chat trigger that broke the page's code fence. Code blocks executed against `autogen` 0.14.1 with a dummy key. |
| 0.11.5 | April 5, 2026 | Earlier revision. |

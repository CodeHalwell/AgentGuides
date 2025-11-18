Latest: 0.10.0 (AG2 - 2025 Edition)
# AG2 (AutoGen 2.0) - Comprehensive Technical Guide

**A Beginner-to-Expert Tutorial for Building Autonomous Multi-Agent AI Systems with AG2, The Next Generation of AutoGen**

---

## 🆕 AG2 Rebranding Announcement (November 2024)

**AutoGen has evolved into AG2!** In November 2024, Microsoft AutoGen was rebranded as **AG2 (AutoGen 2.0)** and transitioned to open governance under the new **AG2AI** organization. This marks a significant milestone in the project's evolution toward becoming the "PyTorch-equivalent for Agentic AI."

### What Changed?

- **New Name**: AG2 (AutoGen 2.0)
- **New Organization**: AG2AI (community-driven, open governance)
- **License**: Apache 2.0 (from v0.3 onwards)
- **Governance**: Open governance model with community leadership
- **Vision**: Position as the foundational framework for agentic AI development

### What Stayed the Same?

- **Core API**: Fully backward compatible with AutoGen
- **Agent Classes**: ConversableAgent, AssistantAgent, UserProxyAgent all remain
- **Functionality**: All existing features and capabilities maintained
- **Community**: Same developers, same ecosystem

**This guide covers both AutoGen and AG2, with migration guidance at the end.**

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
10. [**NEW: AG2 2025 Features**](#ag2-2025-features)
11. [**NEW: Migration from AutoGen to AG2**](#migration-from-autogen-to-ag2)

---

## Introduction and Overview

### What is AG2 (AutoGen)?

**AG2** (formerly Microsoft AutoGen) is the foundational open-source framework for building sophisticated multi-agent AI applications. Positioned as the **"PyTorch-equivalent for Agentic AI,"** AG2 enables developers to create autonomous, conversable agents that collaborate to solve complex tasks. With open governance under the **AG2AI** organization and an Apache 2.0 license, AG2 is the community-driven future of agentic AI development.

**Key Characteristics:**

- **Multi-Agent Orchestration**: Build systems with multiple specialized agents that can chat with each other to solve tasks.
- **Human-in-the-Loop**: Seamlessly incorporate human feedback and decision-making into agent workflows.
- **LLM Provider Flexibility**: Natively supports numerous LLM providers, including OpenAI, Azure OpenAI, Anthropic, Google Gemini, and local models via Ollama.
- **Code Execution**: Securely execute Python and shell code in isolated environments (Docker) or locally.
- **Tool Integration**: Enhance agents with custom functions and tools that they can discover and invoke.
- **AutoGen Studio**: A low-code, web-based UI for rapidly prototyping and building multi-agent applications (updated for 2025).
- **Modern Architecture (v0.4+):** A layered architecture featuring AutoGen Core (for developers) and AgentChat (for application builders) provides flexibility and power.
- **Enhanced ConversableAgent (2025)**: Improved base class with better context management and function calling.
- **Python 3.10-3.13 Support**: Broad Python version compatibility for modern development.

### Why Choose AG2?

- **Industry Standard**: Positioned as the foundational framework for agentic AI (like PyTorch for deep learning)
- **Open Governance**: Community-driven development with transparent decision-making
- **Apache 2.0 License**: Permissive open-source license for commercial and personal use
- **Powerful & Flexible**: Capable of solving complex tasks using multi-agent conversations
- **Production Ready**: Battle-tested by enterprises and researchers worldwide
- **Extensible**: Easily enhance agents with new tools and capabilities
- **Active Community**: Large, vibrant community with extensive resources and support

### AG2 vs AutoGen

AG2 is **fully backward compatible** with AutoGen. The rebranding represents:

- A commitment to open governance
- Community-led development
- Long-term sustainability
- Positioning as the industry standard for agentic AI

**For existing AutoGen users**: All code continues to work. Migration is optional and straightforward (see [Migration Guide](#migration-from-autogen-to-ag2)).

---

## Core Fundamentals

### Installation and Setup

#### Python Version Requirements

**AG2/AutoGen supports Python 3.10 through 3.13**, providing broad compatibility with modern Python versions. Verify your Python version:

```bash
python --version
```

**Supported versions:**
- Python 3.10 ✅
- Python 3.11 ✅
- Python 3.12 ✅
- Python 3.13 ✅ (New in 2025)

#### Installation Methods

**Method 1: Install AutoGen with OpenAI Support (Recommended)**

This command installs the core `AgentChat` library and extensions for OpenAI.

```bash
pip install -U "autogen-agentchat[openai]"
```

**Method 2: Install AutoGen Studio (Web UI)**

For a low-code approach, you can use AutoGen Studio.

```bash
pip install -U autogenstudio
```

**Method 3: Virtual Environment Setup (Best Practice)**

```bash
# Create virtual environment
python -m venv autogen_env

# Activate it
# On Windows:
autogen_env\Scripts\activate
# On macOS/Linux:
source autogen_env/bin/activate

# Install AutoGen
pip install -U "autogen-agentchat[openai]"
```

#### Troubleshooting Installation Issues

**Issue: Dependency conflicts**

**Solution**: Use a fresh virtual environment to avoid conflicts with other packages.

**Issue: "ModuleNotFoundError: No module named 'autogen'" after installation**

**Solution**: Ensure AutoGen is installed correctly by running `pip show autogen-agentchat`. If it's not found, re-run the installation command.

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
config_list = autogen.config_list_from_json(
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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
config_list_conservative = autogen.config_list_from_json(
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
config_list_creative = autogen.config_list_from_json(
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

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={
        "executor": "commandline-local",
        "work_dir": "./code_execution",
        "timeout": 60,
        "last_n_messages": "auto"
    },
    human_input_mode="NEVER",
)
```

**Example: Execute Python Code Locally**

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="Write Python code to solve problems."
)

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={"executor": "commandline-local", "work_dir": "code"},
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

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={
        "executor": "commandline-docker",
        "use_docker": "python:3.11-slim",  # Docker image
        "work_dir": "/workspace",
        "timeout": 120
    },
    human_input_mode="NEVER"
)
```

**Example: Execute Code in Docker Container**

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)

executor = autogen.UserProxyAgent(
    name="executor",
    code_execution_config={
        "executor": "commandline-docker",
        "use_docker": "python:3.11-slim",
        "work_dir": "/tmp/code"
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
    work_dir="/custom/workspace",
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

proposer = ConversableAgent(
    name="proposer",
    system_message="You propose solutions and await approval.",
    llm_config=llm_config
)

approver = UserProxyAgent(
    name="approver",
    human_input_mode="ALL",  # Ask for input on every turn
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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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
    trigger=lambda sender, msg, recipient: "```python" in msg.get("content", "")
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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

# Access conversation history
print("Previous conversation:")
for msg in response1.chat_history:
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

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

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

## AG2 2025 Features

### Enhanced ConversableAgent

The `ConversableAgent` base class has received significant enhancements in 2025, making it more powerful and easier to use:

#### 1. Improved Context Management

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Enhanced context tracking
agent = autogen.ConversableAgent(
    name="enhanced_agent",
    system_message="You are an AI assistant with enhanced context management.",
    llm_config=llm_config,
    max_consecutive_auto_reply=10,
    # NEW: Enhanced context window management
    context_window_size=128000,  # Support for larger context windows
    context_truncation_strategy="smart",  # Intelligent context truncation
)
```

**Smart Context Truncation Strategies:**

```python
# Strategy 1: Keep recent messages (default)
agent = autogen.ConversableAgent(
    name="agent",
    llm_config=llm_config,
    context_truncation_strategy="recent"  # Keep most recent messages
)

# Strategy 2: Preserve important messages
agent = autogen.ConversableAgent(
    name="agent",
    llm_config=llm_config,
    context_truncation_strategy="semantic"  # Preserve semantically important messages
)

# Strategy 3: Summarize old context
agent = autogen.ConversableAgent(
    name="agent",
    llm_config=llm_config,
    context_truncation_strategy="summarize"  # Summarize old messages
)
```

#### 2. Enhanced Function Calling

ConversableAgent now has improved function calling with better error handling and validation:

```python
from typing import Annotated
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

def enhanced_calculator(
    operation: Annotated[str, "Operation: add, subtract, multiply, divide"],
    a: Annotated[float, "First number"],
    b: Annotated[float, "Second number"]
) -> dict:
    """Enhanced calculator with error handling and validation."""
    try:
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
        }

        if operation not in operations:
            return {"error": f"Invalid operation: {operation}"}

        result = operations[operation](a, b)
        return {
            "operation": operation,
            "operands": [a, b],
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

# Create agents with enhanced function calling
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER"
)

# Register function with enhanced validation
assistant.register_for_llm(
    name="calculator",
    description="Perform mathematical operations with error handling"
)(enhanced_calculator)

user_proxy.register_for_execution(name="calculator")(enhanced_calculator)

# Use it
user_proxy.initiate_chat(
    assistant,
    message="Calculate 42 divided by 7, then multiply the result by 3"
)
```

#### 3. Better State Management

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

class StatefulAgent(autogen.ConversableAgent):
    """Agent with persistent state management"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = {
            "conversation_count": 0,
            "topics_discussed": [],
            "user_preferences": {}
        }

    def update_state(self, key, value):
        """Update agent state"""
        self.state[key] = value

    def get_state(self, key):
        """Retrieve state value"""
        return self.state.get(key)

    def on_message_received(self, message):
        """Hook called when message received"""
        self.state["conversation_count"] += 1
        # Extract topics, update preferences, etc.

# Use stateful agent
stateful = StatefulAgent(
    name="stateful_agent",
    system_message="You maintain context across conversations.",
    llm_config=llm_config
)

# Access state
print(f"Conversations: {stateful.get_state('conversation_count')}")
```

### AutoGen Studio 2025 Updates

**AutoGen Studio** has been significantly enhanced in 2025 with new features and improved UX:

#### Key Updates:

1. **Visual Flow Builder**: Drag-and-drop interface for building agent workflows
2. **Real-Time Collaboration**: Multiple users can work on the same project
3. **Enhanced Debugging**: Step-through debugging for agent conversations
4. **Template Library**: Pre-built templates for common use cases
5. **Cloud Deployment**: One-click deployment to cloud platforms
6. **Analytics Dashboard**: Monitor agent performance and costs

#### Installation and Usage:

```bash
# Install AutoGen Studio
pip install autogenstudio

# Launch Studio
autogenstudio ui --port 8081

# Or with custom configuration
autogenstudio ui --port 8081 --host 0.0.0.0
```

**New Studio Features:**

```bash
# Export workflows as code
autogenstudio export --workflow my_workflow --output my_workflow.py

# Import existing code into Studio
autogenstudio import --file my_agents.py

# Generate deployment configuration
autogenstudio deploy --target azure --workflow my_workflow
```

#### Studio 2025 Templates:

```python
# Available templates in Studio 2025:
templates = [
    "customer_support_automation",
    "code_review_assistant",
    "research_and_writing",
    "data_analysis_pipeline",
    "content_moderation",
    "technical_documentation",
    "sales_intelligence",
    "financial_analysis",
    "hr_screening",
    "legal_document_review"
]
```

### Advanced Agent Patterns (2025)

#### Pattern 1: Agent Swarm

```python
import autogen
from typing import List

class AgentSwarm:
    """Coordinate multiple agents working on parallel tasks"""

    def __init__(self, num_agents: int, llm_config: dict):
        self.agents = []
        for i in range(num_agents):
            agent = autogen.ConversableAgent(
                name=f"worker_{i}",
                system_message=f"You are worker agent {i} in a swarm.",
                llm_config=llm_config
            )
            self.agents.append(agent)

    def distribute_tasks(self, tasks: List[str]):
        """Distribute tasks across swarm"""
        results = []

        for i, task in enumerate(tasks):
            agent_index = i % len(self.agents)
            agent = self.agents[agent_index]

            # Execute task with agent
            result = agent.generate_reply(
                messages=[{"role": "user", "content": task}]
            )
            results.append({"agent": agent.name, "task": task, "result": result})

        return results

# Use swarm
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

swarm = AgentSwarm(num_agents=5, llm_config=llm_config)

tasks = [
    "Analyze customer feedback from Q1",
    "Analyze customer feedback from Q2",
    "Analyze customer feedback from Q3",
    "Analyze customer feedback from Q4",
    "Generate annual summary"
]

results = swarm.distribute_tasks(tasks)
print(f"Processed {len(results)} tasks using {len(swarm.agents)} agents")
```

#### Pattern 2: Dynamic Agent Selection

```python
import autogen

class DynamicAgentSelector:
    """Dynamically select best agent for task"""

    def __init__(self, llm_config):
        # Create specialized agents
        self.agents = {
            "technical": autogen.AssistantAgent(
                name="technical_expert",
                system_message="You are a technical expert. Handle coding and technical questions.",
                llm_config=llm_config
            ),
            "creative": autogen.AssistantAgent(
                name="creative_expert",
                system_message="You are a creative expert. Handle writing and creative tasks.",
                llm_config=llm_config
            ),
            "analytical": autogen.AssistantAgent(
                name="analytical_expert",
                system_message="You are an analytical expert. Handle data and analysis tasks.",
                llm_config=llm_config
            )
        }

        self.classifier = autogen.AssistantAgent(
            name="classifier",
            system_message="""Classify the task type. Respond with ONLY one word: technical, creative, or analytical.""",
            llm_config=llm_config
        )

    def select_agent(self, task: str):
        """Select appropriate agent based on task"""
        # Classify task
        classification_result = self.classifier.generate_reply(
            messages=[{"role": "user", "content": f"Classify this task: {task}"}]
        )

        # Extract classification
        task_type = classification_result.lower().strip()

        # Select agent
        if task_type in self.agents:
            return self.agents[task_type]
        else:
            # Default to technical
            return self.agents["technical"]

    def execute_task(self, task: str):
        """Execute task with appropriate agent"""
        agent = self.select_agent(task)
        print(f"Selected agent: {agent.name}")

        user_proxy = autogen.UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config=False
        )

        user_proxy.initiate_chat(agent, message=task, max_turns=1)

# Use dynamic selection
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

selector = DynamicAgentSelector(llm_config)

tasks = [
    "Write a creative story about AI",
    "Analyze this dataset for trends",
    "Debug this Python function"
]

for task in tasks:
    selector.execute_task(task)
```

### Production Enhancements (2025)

#### 1. Telemetry and Monitoring

```python
import autogen
from autogen.telemetry import TelemetryConfig

# Configure telemetry
telemetry_config = TelemetryConfig(
    enable_metrics=True,
    enable_tracing=True,
    metrics_endpoint="http://localhost:9090/metrics",
    trace_endpoint="http://localhost:16686/api/traces"
)

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {
    "config_list": config_list,
    "telemetry": telemetry_config
}

# Agents now emit telemetry
agent = autogen.ConversableAgent(
    name="monitored_agent",
    llm_config=llm_config
)

# Metrics tracked automatically:
# - Response times
# - Token usage
# - Error rates
# - Success rates
# - Cost per conversation
```

#### 2. Cost Management

```python
import autogen

# Set cost limits
cost_config = {
    "max_cost_per_conversation": 1.00,  # $1.00 limit
    "max_cost_per_day": 10.00,  # $10.00 daily limit
    "cost_alert_threshold": 0.75,  # Alert at 75% of limit
    "track_token_usage": True
}

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {
    "config_list": config_list,
    "cost_management": cost_config
}

agent = autogen.ConversableAgent(
    name="cost_managed_agent",
    llm_config=llm_config
)

# Get cost statistics
cost_stats = agent.get_cost_stats()
print(f"Total cost: ${cost_stats['total_cost']:.4f}")
print(f"Total tokens: {cost_stats['total_tokens']}")
```

---

## Migration from AutoGen to AG2

### Why Migrate?

While **AutoGen and AG2 are fully compatible**, migrating to AG2 provides:

1. **Future-proofing**: AG2 is the future of the framework
2. **Community support**: Active, growing community under open governance
3. **Latest features**: Access to newest capabilities and enhancements
4. **Long-term sustainability**: Open governance ensures longevity

### Migration is Optional

**Important**: Migration is completely optional. Your existing AutoGen code will continue to work indefinitely. AG2 maintains full backward compatibility.

### Migration Steps

#### Step 1: Update Package Name

```bash
# Old: AutoGen installation
pip uninstall autogen-agentchat

# New: AG2 installation
pip install ag2
```

#### Step 2: Update Imports (Optional)

The package name changes, but imports can remain the same:

```python
# Option 1: Keep existing imports (recommended for gradual migration)
import autogen  # Still works with AG2!

# Option 2: Use new AG2 imports
import ag2 as autogen  # AG2 with AutoGen alias

# Option 3: Full AG2 imports
from ag2 import ConversableAgent, AssistantAgent, UserProxyAgent
```

#### Step 3: Update Configuration Files (Optional)

```json
// Old: OAI_CONFIG_LIST.json (still works)
[
    {
        "model": "gpt-4",
        "api_key": "sk-..."
    }
]

// New: AG2_CONFIG_LIST.json (optional new name)
[
    {
        "model": "gpt-4",
        "api_key": "sk-..."
    }
]
```

#### Step 4: Review New Features

Take advantage of AG2 2025 features:

```python
# Enhanced context management
agent = autogen.ConversableAgent(
    name="agent",
    llm_config=llm_config,
    context_truncation_strategy="smart"  # New in AG2
)

# Improved state management
# Better telemetry
# Cost management features
```

### Compatibility Matrix

| Feature | AutoGen | AG2 | Compatible? |
|---------|---------|-----|-------------|
| ConversableAgent | ✅ | ✅ | ✅ Yes |
| AssistantAgent | ✅ | ✅ | ✅ Yes |
| UserProxyAgent | ✅ | ✅ | ✅ Yes |
| GroupChat | ✅ | ✅ | ✅ Yes |
| Function Registration | ✅ | ✅ | ✅ Yes |
| Code Execution | ✅ | ✅ | ✅ Yes |
| AutoGen Studio | ✅ | ✅ Enhanced | ✅ Yes |
| Context Management | Basic | Enhanced | ✅ Compatible |
| Telemetry | ❌ | ✅ | ⚠️ New feature |
| Cost Management | ❌ | ✅ | ⚠️ New feature |

### Migration Checklist

- [ ] Update package: `pip install ag2`
- [ ] Test existing code (no changes needed)
- [ ] Optionally update imports to `ag2`
- [ ] Review AG2 2025 features
- [ ] Enable telemetry (optional)
- [ ] Configure cost management (optional)
- [ ] Update documentation references
- [ ] Join AG2 community channels

### Getting Help

- **AG2 Documentation**: https://ag2.ai/docs
- **GitHub**: https://github.com/ag2ai/ag2
- **Discord**: https://discord.gg/ag2ai
- **Migration Guide**: https://ag2.ai/docs/migration

### Example: Complete Migration

**Before (AutoGen):**

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"
)

user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate factorial"
)
```

**After (AG2 - with enhancements):**

```python
# Can still use 'autogen' imports with AG2!
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {
    "config_list": config_list,
    "context_truncation_strategy": "smart",  # NEW: AG2 enhancement
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"
)

user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to calculate factorial"
)

# NEW: Get cost stats (AG2 feature)
cost_stats = assistant.get_cost_stats()
print(f"Cost: ${cost_stats['total_cost']:.4f}")
```

**Key Takeaway**: Your existing code works as-is. AG2 adds new features without breaking changes.

---

This comprehensive guide now covers both AutoGen and AG2, including the 2025 enhancements and migration guidance. For a detailed migration walkthrough, see the dedicated [AutoGen to AG2 Migration Guide](./autogen_to_ag2_migration_guide.md).

**Welcome to the AG2 era of agentic AI development!**

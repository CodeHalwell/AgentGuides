# AG2 (AutoGen 0.2.x Community Fork) - Comprehensive Technical Guide

**A Beginner-to-Expert Tutorial for Building Autonomous Multi-Agent AI Systems**

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

---

## Introduction and Overview

### What is AG2?

AG2 (formerly known as AutoGen) is an open-source programming framework designed for building sophisticated AI agent systems. It enables developers to create autonomous agents capable of collaborating with each other and with humans to solve complex tasks. AG2 is a community-driven fork that maintains compatibility with AutoGen 0.2.x whilst providing enhanced features, improved stability, and vibrant community support.

**Key Characteristics:**

- **Multi-Agent Orchestration**: Build systems with multiple specialized agents working together
- **Human-in-the-Loop**: Seamlessly incorporate human feedback and decision-making
- **LLM Provider Flexibility**: Support for 10+ LLM providers (OpenAI, Anthropic, Google Gemini, local Ollama, and more)
- **Code Execution**: Execute Python and shell code in isolated environments or locally
- **Tool Integration**: Register custom functions as tools that agents can discover and invoke
- **Structured Output**: Generate responses conforming to specific schemas using Pydantic models
- **Protocol Support**: A2A (Agent-to-Agent) protocol for distributed, REST-based agent communication
- **Production-Ready**: Comprehensive logging, cost tracking, and deployment patterns

### Why Choose AG2 Over Alternatives?

- **Stability**: Based on the battle-tested AutoGen 0.2.x architecture
- **Simplicity**: Intuitive APIs with less complexity than AutoGen 0.4
- **Community-Driven**: Active community contributing enhancements and maintaining the project
- **Governance**: Open governance model with transparent decision-making
- **Compatibility**: Easier migration path from AutoGen 0.2.34

---

## Core Fundamentals

### Installation and Setup

#### Python Version Requirements

AG2 requires Python version **>= 3.10 and < 3.14**. Verify your Python version:

```bash
python --version
```

#### Installation Methods

**Method 1: Install AG2 with OpenAI Support (Recommended)**

On Windows/Linux:
```bash
pip install ag2[openai]
```

On macOS:
```bash
pip install 'ag2[openai]'
```

**Method 2: Install AG2 with Minimal Dependencies**

```bash
pip install ag2
```

This installs only the core framework without optional dependencies. Additional features can be installed as needed.

**Method 3: Install with Multiple Provider Support**

```bash
pip install ag2[openai,anthropic,google]
```

**Method 4: Install from Source (Development)**

```bash
git clone https://github.com/ag2ai/ag2.git
cd ag2
pip install -e ".[dev,openai]"
```

**Method 5: Virtual Environment Setup (Best Practice)**

```bash
# Create virtual environment
python -m venv ag2_env

# Activate it
# On Windows:
ag2_env\Scripts\activate
# On macOS/Linux:
source ag2_env/bin/activate

# Install AG2
pip install ag2[openai]
```

#### Troubleshooting Installation Issues

**Issue: "Python version not compatible" error**

```python
# Check your Python version
import sys
print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
# Must be 3.10, 3.11, 3.12, or 3.13
```

**Solution**: Install the correct Python version from [python.org](https://www.python.org/downloads/).

**Issue: Dependency conflicts**

**Solution**: Use `pip install --force-reinstall ag2[openai]` or create a fresh virtual environment.

**Issue: "ModuleNotFoundError: No module named 'autogen'" after installation**

**Solution**: Ensure AG2 is installed correctly by running `pip show ag2`.

### Configuring LLM Providers

#### Understanding LLMConfig

The `LLMConfig` class centralises configuration for LLM providers, supporting temperature, token limits, tool definitions, and provider routing.

#### Method 1: Single Model Configuration

```python
from autogen import LLMConfig, ConversableAgent

# Create LLMConfig for OpenAI GPT-4
config = LLMConfig({
    "model": "gpt-4",
    "api_key": "sk-your-api-key-here",
    "temperature": 0.7,
    "max_tokens": 2000
})

# Use with agent
agent = ConversableAgent(
    name="assistant",
    llm_config=config,
    human_input_mode="NEVER"
)
```

#### Method 2: File-Based Configuration (OAI_CONFIG_LIST)

Create a file named `OAI_CONFIG_LIST`:

```json
[
    {
        "model": "gpt-4",
        "api_key": "sk-your-key-here",
        "organization": "your-org-id"
    },
    {
        "model": "gpt-4o-mini",
        "api_key": "sk-your-key-here"
    }
]
```

Load it in your code:

```python
from autogen import LLMConfig

# Load from file
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Or specify model explicitly
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST", model="gpt-4")
```

#### Method 3: Multiple Models with Fallback

```python
from autogen import LLMConfig

config = LLMConfig(
    {"model": "gpt-4", "api_key": "sk-key1"},
    {"model": "gpt-4o-mini", "api_key": "sk-key2"},
    temperature=0.7,
    max_tokens=2000,
    routing_method="round_robin"  # Distribute load across models
)
```

#### Method 4: Azure OpenAI Configuration

```python
from autogen import LLMConfig
import os

config = LLMConfig({
    "model": "gpt-4",
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "base_url": "https://your-resource.openai.azure.com",
    "api_type": "azure",
    "api_version": "2024-02-15-preview"
})
```

#### Method 5: Anthropic Claude Configuration

```python
from autogen import LLMConfig
import os

config = LLMConfig({
    "model": "claude-3-5-sonnet-20241022",
    "api_key": os.getenv("ANTHROPIC_API_KEY")
})
```

#### Method 6: Google Gemini Configuration

```python
from autogen import LLMConfig
import os

config = LLMConfig({
    "model": "gemini-pro",
    "api_key": os.getenv("GOOGLE_API_KEY")
})
```

#### Method 7: Local Ollama Configuration

```python
from autogen import LLMConfig

config = LLMConfig({
    "model": "llama2",  # or mistral, neural-chat, etc.
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama"
})
```

#### Method 8: Environment Variables

```python
from autogen import LLMConfig
import os

config = LLMConfig({
    "model": "gpt-4",
    "api_key": os.getenv("OPENAI_API_KEY"),  # Loaded from environment
    "organization": os.getenv("OPENAI_ORG_ID")
})
```

### Core Agent Classes

#### ConversableAgent: The Foundation

The `ConversableAgent` is the base class for all agents. It handles message passing, LLM communication, and reply generation.

**Minimal Example:**

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Create a conversable agent
agent = ConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Initiate conversation
print("Agent created:", agent.name)
```

**Key Parameters:**

```python
agent = ConversableAgent(
    name="assistant",  # Agent identifier
    system_message="You are helpful.",  # Role definition
    llm_config=llm_config,  # LLM configuration
    human_input_mode="NEVER",  # Options: NEVER, TERMINATE, ALL
    max_consecutive_auto_reply=10,  # Max replies before stopping
    default_auto_reply="I cannot process this.",  # Fallback response
    code_execution_config=False,  # Enable/disable code execution
    function_map={},  # Map of function names to callables
    description="An assistant agent"  # For documentation
)
```

**Complete Working Example:**

```python
from autogen import ConversableAgent, LLMConfig

# Configure LLM
llm_config = LLMConfig({
    "model": "gpt-4",
    "api_key": "sk-your-key"
})

# Create agents for code discussion
coder = ConversableAgent(
    name="coder",
    system_message="You are an expert Python developer. Write clean, well-documented code.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5
)

reviewer = ConversableAgent(
    name="reviewer",
    system_message="You are a senior code reviewer. Analyse code for quality, security, and best practices. Do not write code.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5
)

# Start conversation
response = reviewer.run(
    recipient=coder,
    message="Write a Python function that validates email addresses using regex.",
    max_turns=8
)

# Process and display results
response.process()
print("Summary:", response.summary)
print("\nChat History:")
for msg in response.chat_history:
    print(f"\n{msg['name']}: {msg['content'][:200]}...")
```

#### AssistantAgent: Pre-configured for Task Solving

`AssistantAgent` is optimised for solving tasks with language models, with code execution disabled by default.

```python
from autogen import AssistantAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Create assistant
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a problem-solving assistant. Provide clear, step-by-step solutions."
)

print(f"Assistant '{assistant.name}' created with LLM: {assistant.llm_config}")
```

**Key Differences from ConversableAgent:**

- `human_input_mode` defaults to "NEVER" (vs. "ALL" in ConversableAgent)
- `code_execution_config` defaults to `False` (vs. potentially enabled)
- Optimised for generating responses without executing code

#### UserProxyAgent: Human-in-the-Loop and Code Execution

`UserProxyAgent` represents a human user or acts as a code executor in automated workflows.

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

# Create assistant
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

# Create user proxy with code execution
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",  # Ask for input when termination condition met
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Set to True for isolated execution
        "executor": "commandline-local"
    },
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").upper()
)

# Initiate task
user_proxy.initiate_chat(
    assistant,
    message="Create a Python script that reads CSV data and generates summary statistics."
)
```

**Human Input Modes:**

- `"NEVER"`: Never ask for human input; agent operates autonomously
- `"TERMINATE"`: Ask for input when termination conditions are met
- `"ALL"`: Ask for input on every agent reply

**Code Execution Configuration:**

```python
# Local execution (no Docker)
code_config_local = {
    "executor": "commandline-local",
    "work_dir": "/tmp/coding",
    "timeout": 60
}

# Docker execution (isolated)
code_config_docker = {
    "executor": "commandline-docker",
    "use_docker": "python:3.11-slim",
    "work_dir": "/tmp/coding",
    "timeout": 120
}

# Custom executor instance
from autogen.coding import LocalCommandLineCodeExecutor
executor = LocalCommandLineCodeExecutor(
    work_dir="/custom/path",
    timeout=90
)
code_config_custom = {"executor": executor}
```

### Version Compatibility and Migration

#### Migrating from AutoGen 0.2.34 to AG2

**Step 1: Update Package Reference**

```bash
# Remove old package
pip uninstall autogen-agentchat

# Install AG2
pip install ag2[openai]
```

**Step 2: Update Imports**

Old (AutoGen 0.2.34):
```python
from autogen import ConversableAgent
from autogen.oai import OpenAIWrapper
```

New (AG2):
```python
from autogen import ConversableAgent
from autogen import LLMConfig
```

**Step 3: Update Configuration**

Old:
```python
from autogen.oai import OpenAIWrapper
config_list = [{"model": "gpt-4", "api_key": "sk-xxx"}]
model_client = OpenAIWrapper(config_list=config_list)
```

New:
```python
from autogen import LLMConfig
llm_config = LLMConfig({"model": "gpt-4", "api_key": "sk-xxx"})
```

**Step 4: Test Your Code**

Create a test script to verify compatibility:

```python
from autogen import ConversableAgent, LLMConfig

# Test basic functionality
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
agent = ConversableAgent("test_agent", llm_config=llm_config)
print("Migration successful!")
```

**Step 5: Handle Deprecated Features**

Check for deprecated methods and update:

```python
# Old deprecated method
# response = agent.send("message", recipient)

# New method
response = agent.run(
    recipient=other_agent,
    message="message"
)
```

#### Comparison: AG2 vs AutoGen 0.4

| Feature | AG2 | AutoGen 0.4 |
|---------|-----|------------|
| Architecture | Synchronous, familiar | Async, event-driven |
| Migration from 0.2.x | Easy, maintains compatibility | Breaking changes |
| Learning Curve | Gentle, well-documented | Steeper, new paradigm |
| Stability | Battle-tested | Newer, evolving |
| Community Support | Active community fork | Microsoft-maintained |
| Use Case | Production systems, existing users | New projects, advanced features |

#### Key Differences in Usage

**AG2 Example:**
```python
# Simple, synchronous
response = agent.run(recipient=other_agent, message="task")
response.process()
```

**AutoGen 0.4 Example (for reference):**
```python
# Async, event-based (simplified representation)
# await async_agent.run(message="task")
```

---

## Simple Agents

### Creating Your First Agent

#### The Absolute Minimum Example

```python
from autogen import ConversableAgent, LLMConfig

# 1. Configure LLM
llm_config = LLMConfig({
    "model": "gpt-4",
    "api_key": "your-api-key"
})

# 2. Create agent
agent = ConversableAgent(
    name="my_agent",
    llm_config=llm_config
)

# 3. Use it
print(f"Agent '{agent.name}' is ready!")
```

#### Single-Agent Task Solving

A single agent receiving and responding to messages:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

# Create problem solver
solver = ConversableAgent(
    name="problem_solver",
    system_message="You are an expert problem solver. Break down complex problems into manageable steps.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Create a dummy recipient for .run() method
dummy = ConversableAgent(
    name="dummy",
    llm_config=False,  # No LLM needed
    human_input_mode="NEVER"
)

# Solve a problem
response = dummy.run(
    recipient=solver,
    message="How would you optimise a slow Python function that processes large datasets?"
)

response.process()
print(response.summary)
```

#### Agent Roles and System Messages

**Example 1: Researcher Agent**

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

researcher = ConversableAgent(
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
writer = ConversableAgent(
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
code_reviewer = ConversableAgent(
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
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

autonomous_agent = ConversableAgent(
    name="autonomous",
    system_message="You solve problems independently.",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Never ask for input
    max_consecutive_auto_reply=15  # Allow many replies before stopping
)
```

Use case: Automated systems, background processing, cost-sensitive tasks.

**Pattern 2: Termination-Based (TERMINATE)**

```python
interactive_agent = ConversableAgent(
    name="interactive",
    system_message="You work collaboratively with humans.",
    llm_config=llm_config,
    human_input_mode="TERMINATE",  # Ask for input when termination conditions met
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").upper()
)
```

Use case: Decision-making workflows, human oversight required.

**Pattern 3: Always Interactive (ALL)**

```python
# This requires human input on every exchange
# Rarely used in practice; reserved for sensitive operations
manual_approval_agent = ConversableAgent(
    name="manual",
    system_message="You require human approval for each action.",
    llm_config=llm_config,
    human_input_mode="ALL"  # Ask on every reply
)
```

#### Temperature and Creativity Settings

**Conservative (Deterministic) Configuration**

```python
config_conservative = LLMConfig({
    "model": "gpt-4",
    "api_key": "sk-xxx",
    "temperature": 0.0,  # Always pick best probability
    "max_tokens": 1000
})

fact_checker = ConversableAgent(
    name="fact_checker",
    system_message="You verify facts and provide accurate information.",
    llm_config=config_conservative
)
```

**Balanced Configuration**

```python
config_balanced = LLMConfig({
    "model": "gpt-4",
    "api_key": "sk-xxx",
    "temperature": 0.7,  # Some creativity
    "max_tokens": 1500
})

creative_agent = ConversableAgent(
    name="creative",
    system_message="You generate ideas and creative solutions.",
    llm_config=config_balanced
)
```

**Creative Configuration**

```python
config_creative = LLMConfig({
    "model": "gpt-4",
    "api_key": "sk-xxx",
    "temperature": 1.0,  # Maximum randomness
    "max_tokens": 2000,
    "top_p": 0.95
})

brainstorm_agent = ConversableAgent(
    name="brainstorm",
    system_message="You generate diverse, unconventional ideas.",
    llm_config=config_creative
)
```

#### Token Limits and Cost Control

```python
# Minimal responses (cost-conscious)
config_minimal = LLMConfig({
    "model": "gpt-4-turbo",
    "api_key": "sk-xxx",
    "max_tokens": 500  # Limit output
})

# Balanced responses
config_balanced = LLMConfig({
    "model": "gpt-4-turbo",
    "api_key": "sk-xxx",
    "max_tokens": 2000
})

# Comprehensive responses
config_comprehensive = LLMConfig({
    "model": "gpt-4-turbo",
    "api_key": "sk-xxx",
    "max_tokens": 4000
})
```

### Code Execution Environments

#### Local Execution (No Docker)

**Advantages**: Fast, simple setup, direct filesystem access
**Disadvantages**: Potential security risks, full system access

```python
from autogen import UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

executor = UserProxyAgent(
    name="executor",
    code_execution_config={
        "executor": "commandline-local",
        "work_dir": "./code_execution",
        "timeout": 60,
        "last_n_messages": "auto"
    },
    human_input_mode="NEVER",
    llm_config=False
)
```

**Example: Execute Python Code Locally**

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="Write Python code to solve problems."
)

executor = UserProxyAgent(
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
from autogen import UserProxyAgent

executor = UserProxyAgent(
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
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

assistant = AssistantAgent("assistant", llm_config=llm_config)

executor = UserProxyAgent(
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
from autogen import UserProxyAgent

# Create custom executor with specific configuration
custom_executor = LocalCommandLineCodeExecutor(
    work_dir="/custom/workspace",
    timeout=90,
    functions=[],  # Allowed functions (empty = all)
    silent=False  # Print execution output
)

executor_agent = UserProxyAgent(
    name="executor",
    code_execution_config={"executor": custom_executor},
    human_input_mode="NEVER"
)
```

### Function Calling and Tool Use

#### Simple Function Registration

```python
from datetime import datetime
from typing import Annotated
from autogen import ConversableAgent, register_function, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

# Define tool function with type hints
def get_current_time(timezone: Annotated[str, "Timezone (e.g., 'UTC', 'EST')"]) -> str:
    """Get the current time in a specified timezone."""
    # Note: This is simplified; use pytz for real implementation
    return f"Current time in {timezone}: 14:30:45"

# Create agents
assistant = ConversableAgent(
    name="assistant",
    system_message="You can access current time in different timezones.",
    llm_config=llm_config
)

executor = ConversableAgent(
    name="executor",
    human_input_mode="NEVER",
    llm_config=False
)

# Register function
register_function(
    get_current_time,
    caller=assistant,
    executor=executor,
    description="Retrieve current time for a specific timezone"
)

# Use it
executor.initiate_chat(
    assistant,
    message="What is the current time in EST and UTC?",
    max_turns=3
)
```

#### Complex Function with Multiple Parameters

```python
from typing import Annotated, List
from autogen import ConversableAgent, register_function, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

def calculate_statistics(
    numbers: Annotated[List[float], "List of numbers to analyse"],
    include_median: Annotated[bool, "Whether to include median (default: True)"] = True,
    decimal_places: Annotated[int, "Decimal places for rounding (default: 2)"] = 2
) -> dict:
    """Calculate statistics for a list of numbers."""
    import statistics
    
    stats = {
        "mean": round(statistics.mean(numbers), decimal_places),
        "min": round(min(numbers), decimal_places),
        "max": round(max(numbers), decimal_places),
        "count": len(numbers)
    }
    
    if include_median:
        stats["median"] = round(statistics.median(numbers), decimal_places)
    
    return stats

assistant = ConversableAgent(
    name="analyst",
    system_message="You analyse numerical data using available tools.",
    llm_config=llm_config
)

executor = ConversableAgent(
    name="executor",
    human_input_mode="NEVER",
    llm_config=False
)

register_function(
    calculate_statistics,
    caller=assistant,
    executor=executor,
    description="Calculate mean, median, min, max for numerical data"
)

executor.initiate_chat(
    assistant,
    message="Calculate statistics for these numbers: 10, 20, 30, 40, 50",
    max_turns=3
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

```python
from pydantic import BaseModel, Field
from typing import List
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

# Define structured response schema
class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    in_stock: bool = Field(description="Whether in stock")

class ProductRecommendation(BaseModel):
    products: List[Product]
    reason: str = Field(description="Why these products are recommended")

# Create agent with schema
agent = ConversableAgent(
    name="recommender",
    system_message="Recommend products matching the user's needs.",
    llm_config=LLMConfig(
        {"model": "gpt-4", "api_key": "sk-xxx"},
        response_format=ProductRecommendation
    )
)

executor = ConversableAgent(
    name="executor",
    human_input_mode="NEVER",
    llm_config=False
)

response = executor.initiate_chat(
    agent,
    message="Recommend laptops suitable for machine learning development.",
    max_turns=1
)

# Parse structured response
import json
response_content = response.chat_history[-1]["content"]
recommendations = ProductRecommendation.model_validate_json(response_content)

for product in recommendations.products:
    print(f"{product.name}: ${product.price} (In stock: {product.in_stock})")
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

This comprehensive guide covers the core and intermediate topics. Continue reading in the production guide for advanced deployment strategies, cost optimisation, and enterprise patterns.


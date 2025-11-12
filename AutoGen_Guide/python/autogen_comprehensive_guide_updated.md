Latest: 0.10.0
# Microsoft AutoGen - Comprehensive Technical Guide

**A Beginner-to-Expert Tutorial for Building Autonomous Multi-Agent AI Systems with the Official Microsoft AutoGen Framework**

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

### What is AutoGen?

AutoGen is an open-source framework from Microsoft designed for building sophisticated applications using multi-agent workflows. It enables developers to create autonomous, conversable agents that can collaborate to solve complex tasks. With a focus on simplifying the orchestration, automation, and optimization of complex LLM workflows, AutoGen has become a leading tool for multi-agent system development.

**Key Characteristics:**

- **Multi-Agent Orchestration**: Build systems with multiple specialized agents that can chat with each other to solve tasks.
- **Human-in-the-Loop**: Seamlessly incorporate human feedback and decision-making into agent workflows.
- **LLM Provider Flexibility**: Natively supports numerous LLM providers, including OpenAI, Azure OpenAI, Anthropic, Google Gemini, and local models via Ollama.
- **Code Execution**: Securely execute Python and shell code in isolated environments (Docker) or locally.
- **Tool Integration**: Enhance agents with custom functions and tools that they can discover and invoke.
- **AutoGen Studio**: A low-code, web-based UI for rapidly prototyping and building multi-agent applications.
- **Modern Architecture (v0.4+):** A layered architecture featuring AutoGen Core (for developers) and AgentChat (for application builders) provides flexibility and power.

### Why Choose AutoGen?

- **Powerful & Flexible**: Capable of solving complex tasks using multi-agent conversations.
- **Human-in-the-loop**: Supports varying levels of human involvement.
- **Extensible**: Easily enhance agents with new tools and capabilities.
- **Open Source**: Backed by Microsoft Research with a growing community.

---

## Core Fundamentals

### Installation and Setup

#### Python Version Requirements

AutoGen requires Python version **>= 3.10**. Verify your Python version:

```bash
python --version
```

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
        "work_dir": "coding",
        "use_docker": False,
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
    code_execution_config={"work_dir": "code"},
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
        "work_dir": "coding",
        "use_docker": True,
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
        "work_dir": "/app/coding", # Mount inside docker
        "use_docker": True,
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
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

analyst = autogen.ConversableAgent(
    name="analyst",
    system_message="You analyse data and provide recommendations.",
    llm_config=llm_config
)

human = autogen.UserProxyAgent(
    name="human",
    human_input_mode="TERMINATE",
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
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

proposer = autogen.ConversableAgent(
    name="proposer",
    system_message="You propose solutions and await approval.",
    llm_config=llm_config
)

approver = autogen.UserProxyAgent(
    name="approver",
    human_input_mode="ALWAYS",
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
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

agent_a = autogen.ConversableAgent(
    name="alice",
    system_message="You are Alice, a curious learner.",
    llm_config=llm_config
)

agent_b = autogen.ConversableAgent(
    name="bob",
    system_message="You are Bob, an experienced teacher.",
    llm_config=llm_config
)

# Initiate conversation
chat_result = agent_a.initiate_chat(
    agent_b,
    message="Explain machine learning to me like I'm 10 years old."
)
```

#### Passing Conversation in Reverse

```python
# Continue conversation with other agent initiating
chat_result_2 = agent_b.initiate_chat(
    recipient=agent_a,
    message="Now explain deep learning.",
    max_turns=2
)
```

### Group Chat: Multiple Agents

#### Creating a Group Chat

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Create specialized agents
planner = autogen.ConversableAgent(
    name="planner",
    system_message="You plan project tasks and timelines.",
    llm_config=llm_config
)

developer = autogen.ConversableAgent(
    name="developer",
    system_message="You implement planned tasks using code.",
    llm_config=llm_config
)

tester = autogen.ConversableAgent(
    name="tester",
    system_message="You test implementations and report issues.",
    llm_config=llm_config
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[planner, developer, tester],
    messages=[],
    max_round=10,
)

# Create manager
manager = autogen.GroupChatManager(
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
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Research agent
researcher = autogen.ConversableAgent(
    name="researcher",
    system_message="You research topics deeply and provide well-researched information.",
    llm_config=llm_config
)

# Writer agent
writer = autogen.ConversableAgent(
    name="writer",
    system_message="You write clear, engaging content based on research.",
    llm_config=llm_config
)

# Editor agent
editor = autogen.ConversableAgent(
    name="editor",
    system_message="You edit content for clarity, consistency, and quality. Provide feedback, don't rewrite.",
    llm_config=llm_config
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[researcher, writer, editor],
    messages=[],
    max_round=12,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

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
groupchat = autogen.GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method="auto"  # Intelligent selection
)
```

#### Strategy 2: Round-Robin (Sequential)

```python
# Agents speak in turn
groupchat = autogen.GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method="round_robin"  # A1 → A2 → A3 → A1...
)
```

#### Strategy 3: Random Selection

```python
groupchat = autogen.GroupChat(
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

groupchat = autogen.GroupChat(
    agents=[agent1, agent2, agent3],
    messages=[],
    max_round=15,
    speaker_selection_method=custom_speaker_selector
)
```

#### Preventing Repetitive Speakers

```python
groupchat = autogen.GroupChat(
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
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Main agents
task_planner = autogen.ConversableAgent(
    name="planner",
    system_message="You plan high-level tasks.",
    llm_config=llm_config
)

implementer = autogen.UserProxyAgent(
    name="implementer",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER"
)

# Nested review agents
reviewer = autogen.ConversableAgent(
    name="reviewer",
    system_message="You review code for quality and suggest improvements.",
    llm_config=llm_config
)

improver = autogen.ConversableAgent(
    name="improver",
    system_message="You refactor code based on feedback.",
    llm_config=llm_config
)

# Register nested chats
implementer.register_nested_chats(
    [
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
    trigger=lambda sender: "```python" in implementer.last_message(sender)["content"]
)

# Start main conversation (nested chats trigger automatically)
task_planner.initiate_chat(
    implementer,
    message="Create a Python script to sort a list of dictionaries by multiple keys."
)
```

### Sequential Chats: Multi-Stage Workflows

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Stage 1 agent
ideator = autogen.ConversableAgent(
    name="ideator",
    system_message="You brainstorm innovative product ideas.",
    llm_config=llm_config
)

# Stage 2 agent
validator = autogen.ConversableAgent(
    name="validator",
    system_message="You validate ideas for feasibility and market potential.",
    llm_config=llm_config
)

# Stage 3 agent
planner = autogen.ConversableAgent(
    name="planner",
    system_message="You create implementation plans for validated ideas.",
    llm_config=llm_config
)

# Coordinator
coordinator = autogen.UserProxyAgent(
    name="coordinator",
    human_input_mode="NEVER"
)

# Define chat sequence
chat_sequence = [
    {
        "recipient": ideator,
        "message": "Generate 3 innovative ideas for an e-commerce startup.",
        "max_turns": 1,
        "summary_method": "last_msg"
    },
    {
        "recipient": validator,
        "message": "Validate these ideas for market potential.",
        "max_turns": 1,
        "summary_method": "last_msg",
    },
    {
        "recipient": planner,
        "message": "Create detailed implementation plans.",
        "max_turns": 1,
        "summary_method": "last_msg",
    }
]

# Execute sequential chats
coordinator.initiate_chats(chat_sequence)
```

---
## Agentic Patterns and Workflows

### ReAct (Reasoning and Acting) Implementation

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Agent that thinks and acts
react_agent = autogen.AssistantAgent(
    name="react_agent",
    system_message="""You solve problems using the ReAct (Reasoning + Acting) pattern.
1. Thought: Analyse the problem.
2. Action: Choose a tool/action.
3. Observation: Observe results.
4. Repeat until solved.

Clearly label each thought, action, and observation.""",
    llm_config=llm_config
)

executor = autogen.UserProxyAgent(
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
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Initial generator
generator = autogen.ConversableAgent(
    name="generator",
    system_message="Generate solutions to problems.",
    llm_config=llm_config
)

# Critic
critic = autogen.ConversableAgent(
    name="critic",
    system_message="Critique solutions and suggest improvements. Identify weaknesses and edge cases.",
    llm_config=llm_config
)

# Refiner
refiner = autogen.ConversableAgent(
    name="refiner",
    system_message="Refine solutions based on critical feedback.",
    llm_config=llm_config
)

# Start self-refinement loop
groupchat = autogen.GroupChat(
    agents=[generator, critic, refiner],
    messages=[
        {
            "role": "user",
            "content": "Design an algorithm to detect anomalies in time series data."
        }
    ],
    max_round=3
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
generator.initiate_chat(manager, message="Start the refinement process.")
```

---

## Memory Systems and Context Management

### Conversation History Management

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

agent1 = autogen.ConversableAgent(
    name="agent1",
    system_message="You remember and reference previous conversations.",
    llm_config=llm_config
)

agent2 = autogen.ConversableAgent(
    name="agent2",
    system_message="You build on previous context.",
    llm_config=llm_config
)

# First conversation
chat_result_1 = agent2.initiate_chat(
    agent1,
    message="We're building a machine learning model for customer churn prediction.",
    max_turns=2
)

# Access conversation history
print("Previous conversation:")
for msg in chat_result_1.chat_history:
    print(f"{msg['role']}: {msg['content'][:100]}...")

# Second conversation carries the context
chat_result_2 = agent2.initiate_chat(
    agent1,
    message="Given our earlier discussion, what features should we prioritise?",
    max_turns=2,
    # The history is automatically carried over in the agent objects
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
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

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

analyst = autogen.ConversableAgent(
    name="analyst",
    system_message=few_shot_system_message,
    llm_config=llm_config
)
```

---

This comprehensive guide covers the core and intermediate topics. Continue reading in the production guide for advanced deployment strategies, cost optimisation, and enterprise patterns.
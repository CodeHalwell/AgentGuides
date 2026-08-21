---
title: "Google Agent Development Kit (ADK) - Comprehensive Technical Guide"
description: "Version: 1.1 Last Updated: August 21, 2026 Framework: Google Agent Development Kit (ADK) Target Audience: Beginner to Advanced Developers"
framework: google-adk
language: python
---

Latest: 2.7.1 | Updated: August 21, 2026
# Google Agent Development Kit (ADK) - Comprehensive Technical Guide

**Version:** 1.1  
**Last Updated:** August 21, 2026  
**Framework:** Google Agent Development Kit (ADK)  
**Target Audience:** Beginner to Advanced Developers

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Fundamentals](#core-fundamentals)
3. [Simple Agents](#simple-agents)
4. [Multi-Agent Systems](#multi-agent-systems)
5. [Tools Integration](#tools-integration)
6. [Structured Output](#structured-output)
7. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
8. [Agentic Patterns](#agentic-patterns)
9. [Memory Systems](#memory-systems)
10. [Context Engineering](#context-engineering)
11. [Google Cloud Integration](#google-cloud-integration)
12. [Gemini-Specific Features](#gemini-specific-features)
13. [Vertex AI](#vertex-ai)
14. [Advanced Topics](#advanced-topics)
15. [Class & API Reference](#class--api-reference)

---

## Introduction

The Google Agent Development Kit (ADK) is an open-source, code-first Python framework designed to simplify the creation, evaluation, and deployment of sophisticated AI agents. ADK is optimised for integration with Google's Gemini models and the broader Google Cloud ecosystem, whilst maintaining a model-agnostic and framework-flexible approach that allows developers to use alternative language models and deployment platforms.

ADK addresses the fundamental challenges of agent development by providing:

- **Structured Agent Framework:** Clear abstractions for defining agents, tools, and orchestration patterns
- **Multi-Agent Orchestration:** Built-in support for hierarchical, sequential, parallel, and loop-based agent coordination
- **Rich Tool Ecosystem:** Pre-built integrations with Google services (Search, BigQuery, Vertex AI) and support for custom tools
- **Session Management:** Comprehensive state management with support for resumable agents
- **Evaluation Framework:** Tools for testing and evaluating agent behaviour against defined criteria
- **Production Readiness:** Built-in observability, authentication, and deployment options for Cloud Run and Vertex AI

### Key Advantages

1. **Code-First Philosophy:** Agents are defined through Python code, enabling version control, testing, and CI/CD integration
2. **Scalability:** Supports everything from simple single-agent assistants to complex multi-agent systems
3. **Google Cloud Integration:** Native support for Vertex AI, Cloud Run, Firestore, BigQuery, and other GCP services
4. **Gemini Optimisation:** Full leverage of Gemini 2.5 capabilities including multimodal inputs, context caching, and function calling
5. **Developer Experience:** CLI tools, web-based development UI, and comprehensive documentation

---

## Core Fundamentals

### Installation and Setup

#### Installing the SDK

The first step in using ADK is installing the `google-adk` package:

```bash
# Create a virtual environment (recommended)
python3 -m venv adk_env
source adk_env/bin/activate  # On Windows: adk_env\Scripts\activate

# Install the ADK package
pip install google-adk>=2.0.0

# Verify installation
python -c "import google.adk; print('ADK installed successfully')"
```

#### Installing Additional Dependencies

Depending on your use case, you may need additional packages:

```bash
# For structured outputs with Pydantic
pip install pydantic

# For Google Cloud services
pip install google-cloud-firestore
pip install google-cloud-bigquery
pip install google-cloud-storage

# For development and testing
pip install pytest
pip install pytest-asyncio

# For async support
pip install aiohttp
```

#### Creating a requirements.txt

For production deployments, create a `requirements.txt` file:

```
google-adk>=1.0.0
google-genai>=0.3.0
pydantic>=2.0
google-cloud-firestore>=2.14.0
google-cloud-bigquery>=3.13.0
google-cloud-storage>=2.10.0
python-dotenv>=1.0.0
```

**Cost Considerations:**
- The `google-adk` package itself is free
- Using Gemini models incurs costs based on input/output tokens (typically $0.075 per 1M input tokens, $0.30 per 1M output tokens for Gemini 2.5 Flash)
- Google Cloud services (Firestore, BigQuery, etc.) have their own pricing models

### Google Cloud Project Setup

#### Creating a Google Cloud Project

1. Navigate to [Google Cloud Console](https://console.cloud.google.com/)
2. Click the project dropdown in the top navigation bar
3. Click "New Project"
4. Enter project name: `adk-agents-project` (or your preferred name)
5. Click "Create"
6. Wait for project creation (this may take a few minutes)

#### Enabling Required APIs

For a typical ADK application, enable the following APIs:

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  firestore.googleapis.com \
  storage-api.googleapis.com \
  bigquery.googleapis.com \
  secretmanager.googleapis.com
```

#### Setting Up Authentication

**Option 1: Application Default Credentials (Development)**

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and authenticate
gcloud init
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**Option 2: Service Account (Production)**

```bash
# Create service account
gcloud iam service-accounts create adk-agent-sa \
  --display-name="ADK Agent Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:adk-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:adk-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/datastore.user"

# Create and download key
gcloud iam service-accounts keys create adk-key.json \
  --iam-account=adk-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/adk-key.json"
```

### ADK Architecture and Design Principles

#### Core Architecture Components

ADK is built on the following architectural principles:

```
┌─────────────────────────────────────────────────┐
│            Application Layer                     │
│  (User interactions, API endpoints)              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│            Runner Layer                          │
│  (Orchestration, execution, session management)  │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│            Agent Layer                           │
│  (Agent definition, reasoning, planning)         │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
    ┌───▼──┐  ┌────▼───┐  ┌──▼────┐
    │Tools │  │ Models │  │Memory │
    └──────┘  └────────┘  └───────┘
```

#### Design Principles

1. **Modularity:** Each component (agents, tools, services) is independently testable and replaceable
2. **Separation of Concerns:** Clear boundaries between agent logic, tool execution, and infrastructure
3. **Async-First:** All I/O operations support asynchronous execution for improved performance
4. **Observable:** Built-in telemetry for monitoring agent execution and debugging
5. **Extensible:** Easy to add custom agents, tools, and services
6. **Type-Safe:** Full type hints for improved IDE support and runtime safety

### Core Classes: Agent, LlmAgent, Runner

#### The Agent Class

The `Agent` class is the base abstraction for all agents in ADK:

```python
from google.adk import Agent
from google.adk.tools import google_search
from google.genai import types

# Create a basic agent
search_agent = Agent(
    name="web_researcher",
    model="gemini-2.5-flash",
    description="Researches topics using web search",
    instruction="You are a helpful research assistant. Use web search to find accurate, current information.",
    tools=[google_search],
    max_iterations=5,
    max_total_tokens=4096
)
```

**Key Properties:**

- `name`: Unique identifier for the agent
- `model`: The LLM model to use (e.g., "gemini-2.5-flash", "gemini-2.5-pro")
- `description`: Human-readable description of agent capabilities
- `instruction`: System prompt defining agent behaviour and constraints
- `tools`: List of tools the agent can invoke
- `max_iterations`: Maximum reasoning steps before timeout
- `max_total_tokens`: Maximum tokens for a single invocation

#### The LlmAgent Class

`LlmAgent` is a specialised subclass optimised for language model interactions:

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search, url_context

# Create an LLM agent with advanced features
researcher = LlmAgent(
    name="research_specialist",
    model="gemini-2.5-pro",
    description="Conducts in-depth research with content extraction",
    instruction="""You are a research specialist. Your responsibilities:
    1. Use google_search to find relevant sources
    2. Use url_context to extract and analyse content
    3. Synthesise information into comprehensive summaries
    4. Cite all sources accurately""",
    tools=[google_search, url_context],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
    ),
)

# Define sub-agents for hierarchical structure
summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Create concise, accurate summaries of provided text"
)

fact_checker = LlmAgent(
    name="fact_checker",
    model="gemini-2.5-pro",
    instruction="Verify claims and identify any factual inaccuracies"
)

# Create coordinator that uses sub-agents
coordinator = LlmAgent(
    name="research_coordinator",
    model="gemini-2.5-pro",
    instruction="Coordinate research, summarisation, and fact-checking",
    sub_agents=[summariser, fact_checker]
)
```

#### The Runner Class

`Runner` manages agent execution, handling session management and lifecycle:

```python
from google.adk import Runner, Agent
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.genai import types
import asyncio

# Create runner with in-memory session service (for development)
async_runner = Runner(
    app_name="research_app",
    agent=search_agent,
    session_service=InMemorySessionService()
)

# Create runner with DatabaseSessionService for production (SQLite, Postgres, etc.)
async def create_production_runner():
    runner = Runner(
        app_name="research_app_prod",
        agent=search_agent,
        session_service=DatabaseSessionService(
            db_url="postgresql+asyncpg://user:pass@localhost/adk"
        )
    )
    return runner

# Execute agent asynchronously
async def execute_agent():
    # Create content message
    user_message = types.Content(
        role='user',
        parts=[types.Part(text="What are the latest developments in quantum computing?")]
    )

    # Run agent
    async for event in async_runner.run_async(
        user_id="user123",
        session_id="session456",
        new_message=user_message
    ):
        if event.content:
            print(f"[{event.author}]: {event.content}")

# Run in event loop
asyncio.run(execute_agent())
```

### Gemini Model Configuration

#### Available Models

ADK supports multiple Gemini models with different capabilities and costs:

| Model | Input Cost | Output Cost | Context Window | Best For |
|-------|-----------|-----------|-----------------|----------|
| gemini-2.5-flash | $0.075/1M | $0.30/1M | 1M tokens | Fast responses, tool calling |
| gemini-2.5-pro | $3/1M | $12/1M | 1M tokens | Complex reasoning, quality |
| gemini-1.5-flash | $0.075/1M | $0.30/1M | 1M tokens | Cost-effective tasks |
| gemini-1.5-pro | $3/1M | $12/1M | 1M tokens | Advanced reasoning |

#### Model Configuration

```python
from google.adk import Agent
from google.genai import types

# Configure with Flash model for speed
fast_agent = Agent(
    name="quick_responder",
    model="gemini-2.5-flash",
    description="Provides quick responses",
    instruction="Respond concisely and quickly",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=1024,
    ),
)

# Configure with Pro model for complex reasoning
reasoner = Agent(
    name="complex_reasoner",
    model="gemini-2.5-pro",
    description="Handles complex reasoning tasks",
    instruction="Provide detailed, step-by-step reasoning",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=4096,
    ),
)

# Configure with safety settings
safe_agent = Agent(
    name="safe_responder",
    model="gemini-2.5-flash",
    instruction="Provide helpful information safely",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ],
    ),
)
```

#### Temperature and Sampling Parameters

- **temperature** (0.0 - 1.0): Controls randomness
  - 0.0: Deterministic, same response every time
  - 0.5: Balanced creativity and consistency
  - 1.0: Maximum creativity, varied responses
  
- **top_p** (0.0 - 1.0): Nucleus sampling - considers top-p cumulative probability
  - 0.9: Consider only top 90% probability mass
  - 0.5: More conservative, less random

- **top_k** (1 - 40): Consider only top-k most likely tokens
  - Smaller values (10-20): More focused, deterministic
  - Larger values (40): More diverse, creative

### API Keys and Credentials

#### Using API Keys

While Application Default Credentials are recommended for production, you can also use API keys:

```python
import os
from google.adk import Agent
from google.genai import Client

# Set API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# Initialise client with API key
client = Client(api_key=os.environ["GOOGLE_API_KEY"])

# Create agent that uses the client
api_agent = Agent(
    name="api_agent",
    model="gemini-2.5-flash",
    description="Agent using API key authentication",
    instruction="You are a helpful assistant"
)
```

#### Managing Credentials with .env Files

For development, use environment files:

```python
# .env file
GOOGLE_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GEMINI_API_KEY=your-api-key

# Python code
from dotenv import load_dotenv
import os

load_dotenv()

project_id = os.getenv("GOOGLE_PROJECT_ID")
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
api_key = os.getenv("GEMINI_API_KEY")
```

#### Using Secret Manager for Production

```bash
# Create secret
gcloud secrets create gemini-api-key \
  --replication-policy="user-managed" \
  --locations=us-central1

# Add secret version
echo "YOUR_API_KEY" | gcloud secrets versions add gemini-api-key --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member="serviceAccount:adk-agent-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Simple Agents

### Creating Basic Agents with LlmAgent

#### The Simplest Agent

```python
from google.adk.agents import LlmAgent

# Minimal agent configuration
chatbot = LlmAgent(
    name="chatbot",
    model="gemini-2.5-flash",
    instruction="You are a friendly chatbot. Answer user questions helpfully."
)

print(f"Created agent: {chatbot.name}")
print(f"Description: {chatbot.description or 'No description'}")
```

#### Agent with Tools

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# Agent with web search capability
web_agent = LlmAgent(
    name="web_assistant",
    model="gemini-2.5-flash",
    description="Searches the web for current information",
    instruction="""You are a web search assistant. When asked about current events,
    weather, or recent information, use google_search to find accurate, up-to-date information.
    Always cite your sources.""",
    tools=[google_search]
)
```

#### Agent with Custom Configuration

```python
from google.adk.agents import LlmAgent
from google.genai import types

# Fully configured agent
configured_agent = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-pro",
    description="Analyses data and provides insights",
    instruction="""You are a data analyst. When given data:
    1. Identify key patterns and trends
    2. Calculate relevant statistics
    3. Suggest actionable insights
    4. Present findings clearly""",
    tools=[],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
        top_p=0.9,
    ),
)
```

### Agent Configuration and Initialization

#### Configuration Parameters

```python
from google.adk import Agent
from google.genai import types

# Comprehensive configuration
agent = Agent(
    # Identity
    name="research_assistant",
    model="gemini-2.5-flash",
    description="Conducts research and summarises findings",
    
    # Behaviour
    instruction="""You are a research specialist with expertise in multiple domains.
    Your approach:
    - Gather comprehensive information from multiple perspectives
    - Identify and synthesise patterns
    - Present findings with appropriate caveats
    - Always distinguish between facts and opinions""",
    
    # Capabilities
    tools=[],  # Tools will be added separately
    
    # Model behaviour
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=2048,
    ),
)
```

### System Instructions

#### Crafting Effective System Instructions

System instructions define the agent's personality, expertise, and constraints:

```python
from google.adk.agents import LlmAgent

# Well-structured system instruction
instruction = """You are a professional technical writer specialising in API documentation.

EXPERTISE:
- REST API design and documentation
- OpenAPI/Swagger specifications
- Developer experience best practices
- Technical accuracy and clarity

RESPONSIBILITIES:
1. Understand complex technical concepts
2. Translate them into clear documentation
3. Provide practical code examples
4. Consider the target audience

CONSTRAINTS:
- Use professional, clear language
- Avoid jargon without explanation
- Provide concrete examples
- Cite sources for technical claims

OUTPUT FORMAT:
- Structure documentation hierarchically
- Use code blocks for examples
- Include parameter descriptions
- Provide usage scenarios"""

doc_writer = LlmAgent(
    name="api_doc_writer",
    model="gemini-2.5-flash",
    description="Writes professional API documentation",
    instruction=instruction
)
```

#### Role-Based Instructions

```python
from google.adk.agents import LlmAgent

# Customer support agent
support_agent = LlmAgent(
    name="support_specialist",
    model="gemini-2.5-flash",
    instruction="""You are a patient, helpful customer support specialist.

ROLE:
- Listen carefully to customer concerns
- Provide clear, actionable solutions
- Escalate when necessary
- Follow up to ensure satisfaction

TONE:
- Empathetic and professional
- Avoid technical jargon
- Acknowledge frustration
- Be proactive"""
)

# Code review agent
review_agent = LlmAgent(
    name="code_reviewer",
    model="gemini-2.5-pro",
    instruction="""You are an experienced code reviewer focused on quality and maintainability.

REVIEW CRITERIA:
1. Code correctness - Does it work as intended?
2. Readability - Can other developers understand it?
3. Performance - Are there efficiency concerns?
4. Security - Are there potential vulnerabilities?
5. Testing - Is the code adequately tested?

FEEDBACK STYLE:
- Constructive and specific
- Offer alternatives, not just criticism
- Praise good practices
- Link to relevant documentation"""
)
```

### Simple Function Calling

#### Basic Function Tool

```python
from google.adk.agents import LlmAgent
import math

# Define a simple tool function
def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle.
    
    Args:
        radius: The radius of the circle in units
        
    Returns:
        The area of the circle in square units
    """
    return math.pi * radius ** 2

# Create agent with the tool
math_agent = LlmAgent(
    name="math_helper",
    model="gemini-2.5-flash",
    instruction="You are a helpful math assistant. Use tools to perform calculations.",
    tools=[calculate_circle_area]
)
```

#### Tool with Multiple Parameters

```python
from google.adk.agents import LlmAgent
from typing import List

def calculate_statistics(numbers: List[float]) -> dict:
    """Calculate statistics for a list of numbers.
    
    Args:
        numbers: List of numerical values
        
    Returns:
        Dictionary with mean, median, and standard deviation
    """
    import statistics
    return {
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
        "count": len(numbers)
    }

stats_agent = LlmAgent(
    name="stats_calculator",
    model="gemini-2.5-flash",
    instruction="Calculate statistical measures when asked about data",
    tools=[calculate_statistics]
)
```

#### Async Tool Functions

```python
from google.adk.agents import LlmAgent
import asyncio
import aiohttp

async def fetch_weather(city: str) -> dict:
    """Fetch weather information for a city (async).
    
    Args:
        city: City name
        
    Returns:
        Weather information dictionary
    """
    # Simulated async API call
    await asyncio.sleep(1)
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 65
    }

weather_agent = LlmAgent(
    name="weather_assistant",
    model="gemini-2.5-flash",
    instruction="Provide weather information for requested locations",
    tools=[fetch_weather]
)
```

### Synchronous Execution

#### Running Agents Synchronously

While ADK is async-first, you can run agents synchronously:

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk import Runner
from google.genai import types

# Create agent and runner
agent = LlmAgent(
    name="sync_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions helpfully"
)

runner = Runner(
    app_name="sync_app",
    agent=agent,
    session_service=InMemorySessionService()
)

# Wrapper function for synchronous execution
def run_agent_sync(user_id: str, query: str) -> str:
    """Run agent synchronously."""
    async def _run():
        message = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )
        
        response_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id="session_1",
            new_message=message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        
        return response_text
    
    return asyncio.run(_run())

# Use synchronously
response = run_agent_sync("user123", "What is Python?")
print(response)
```

### Error Handling

#### Basic Error Handling

```python
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

agent = LlmAgent(
    name="error_safe_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions helpfully"
)

runner = Runner(
    app_name="error_handling_app",
    agent=agent,
    session_service=InMemorySessionService()
)

async def run_with_error_handling():
    try:
        message = types.Content(
            role='user',
            parts=[types.Part(text="What is 2+2?")]
        )
        
        async for event in runner.run_async(
            user_id="user123",
            session_id="session_1",
            new_message=message
        ):
            if event.content:
                print(f"Response: {event.content}")
    
    except ValueError as e:
        print(f"Validation error: {e}")
    except TimeoutError as e:
        print(f"Execution timeout: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

# Execute
asyncio.run(run_with_error_handling())
```

#### Tool Error Handling

```python
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

def divide_numbers(a: float, b: float, tool_context: ToolContext) -> float:
    """Safely divide two numbers.
    
    Args:
        a: Numerator
        b: Denominator
        tool_context: Tool execution context
        
    Returns:
        Result of division
        
    Raises:
        ValueError: If denominator is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Create agent with tool
calculator = LlmAgent(
    name="calculator",
    model="gemini-2.5-flash",
    instruction="Help with mathematical calculations. Handle errors gracefully.",
    tools=[divide_numbers]
)
```

---

## Multi-Agent Systems

### Agent Composition and Nesting

#### Creating Nested Agent Hierarchies

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search

# Define leaf agents (no sub-agents)
researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    description="Conducts research using web search",
    instruction="Search for and compile information on given topics",
    tools=[google_search]
)

summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    description="Summarises information",
    instruction="Create concise, clear summaries of provided information"
)

# Define parent agent that coordinates children
content_creator = LlmAgent(
    name="content_creator",
    model="gemini-2.5-pro",
    description="Creates comprehensive content by coordinating research and summarisation",
    instruction="Coordinate research and summarisation to produce high-quality content",
    sub_agents=[researcher, summariser]
)

# Can continue nesting - create a higher-level coordinator
publishing_agent = LlmAgent(
    name="publisher",
    model="gemini-2.5-pro",
    description="Manages content creation for publication",
    instruction="Oversee content creation process",
    sub_agents=[content_creator]
)
```

#### Deep Hierarchies

```python
from google.adk.agents import LlmAgent

# Layer 1: Specialists
qa_agent = LlmAgent(name="qa_specialist", model="gemini-2.5-flash", instruction="Ensure quality")
qa_agent = LlmAgent(name="qa_specialist", model="gemini-2.5-flash", 
    instruction="Ensure quality standards are met")
writer = LlmAgent(name="writer", model="gemini-2.5-flash", 
    instruction="Create original content")

# Layer 2: Team leads
content_lead = LlmAgent(
    name="content_lead",
    model="gemini-2.5-pro",
    instruction="Lead content team",
    sub_agents=[writer, qa_agent]
)

# Layer 3: Department head
content_manager = LlmAgent(
    name="content_manager",
    model="gemini-2.5-pro",
    instruction="Manage content department",
    sub_agents=[content_lead]
)
```

### Parent-Child Agent Relationships

#### Direct Child Invocation

```python
from google.adk.agents import LlmAgent

# Parent agent that explicitly coordinates children
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-pro",
    description="Coordinates specialised agents",
    instruction="""You coordinate between research and analysis agents.
    When given a task:
    1. Ask the researcher agent to gather information
    2. Pass information to the analyst agent
    3. Synthesise final response""",
    sub_agents=[researcher, analyst]
)
```

#### Communication Patterns

```python
from google.adk.agents import LlmAgent

# Sequential delegation
sequencer = LlmAgent(
    name="sequencer",
    model="gemini-2.5-pro",
    instruction="""Handle tasks sequentially:
    1. Pass task to planner for strategy
    2. Pass plan to executor for implementation
    3. Pass results to reviewer for quality check""",
    sub_agents=[planner, executor, reviewer]
)

# Parallel delegation
paralleliser = LlmAgent(
    name="paralleliser",
    model="gemini-2.5-pro",
    instruction="""Distribute work to specialists simultaneously:
    - Data analyst examines data trends
    - Market researcher studies competition
    - Writer creates content
    Combine their outputs into comprehensive report""",
    sub_agents=[analyst, researcher, writer]
)
```

### Agent Orchestration Patterns

#### Sequential Agent Workflow

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools import google_search

# Define workflow steps
step1_research = LlmAgent(
    name="research",
    model="gemini-2.5-flash",
    instruction="Research the topic thoroughly",
    tools=[google_search]
)

step2_organise = LlmAgent(
    name="organise",
    model="gemini-2.5-flash",
    instruction="Organise research into logical structure"
)

step3_draft = LlmAgent(
    name="draft_writer",
    model="gemini-2.5-pro",
    instruction="Write comprehensive draft based on research"
)

step4_edit = LlmAgent(
    name="editor",
    model="gemini-2.5-flash",
    instruction="Edit for clarity, coherence, and accuracy"
)

# Create sequential workflow
article_workflow = SequentialAgent(
    name="article_creation_workflow",
    description="Multi-step article creation process",
    sub_agents=[step1_research, step2_organise, step3_draft, step4_edit],
    instruction="Execute each step in sequence, passing results forward"
)
```

#### Parallel Agent Execution

```python
from google.adk.agents import ParallelAgent, LlmAgent

# Define agents to run in parallel
parser = LlmAgent(
    name="parser",
    model="gemini-2.5-flash",
    instruction="Parse and structure the input"
)

analyser = LlmAgent(
    name="analyser",
    model="gemini-2.5-pro",
    instruction="Analyse the input for patterns"
)

classifier = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction="Classify the input"
)

# Create parallel execution
multi_processor = ParallelAgent(
    name="parallel_processor",
    description="Process input through multiple perspectives simultaneously",
    sub_agents=[parser, analyser, classifier],
    instruction="Execute all agents in parallel and combine results"
)
```

#### Loop Agent for Iterative Tasks

```python
from google.adk.agents import LoopAgent, LlmAgent
from google.adk.tools import exit_loop

# Define iterative agent
problem_solver = LlmAgent(
    name="problem_solver",
    model="gemini-2.5-pro",
    instruction="""Solve the problem step by step:
    1. State your current understanding
    2. Identify remaining issues
    3. Make progress towards solution
    4. Call exit_loop when complete""",
    tools=[exit_loop],
    max_iterations=1  # Reset per loop iteration
)

# Create loop wrapper
iterative_solution = LoopAgent(
    name="iterative_problem_solver",
    description="Solve complex problems through iteration",
    sub_agents=[problem_solver],
    max_iterations=10,
    instruction="Keep iterating until solution found"
)
```

### Distributed Agent Architectures

#### Agent-to-Agent Communication

```python
from google.adk import Agent
from google.adk.tools import google_search
from google.genai import types

# Define agents that can communicate
agent_a = Agent(
    name="agent_a",
    model="gemini-2.5-flash",
    instruction="You handle user requests and delegate to specialist agents",
    tools=[google_search]
)

agent_b = Agent(
    name="agent_b",
    model="gemini-2.5-flash",
    instruction="You are a specialist. Receive requests from other agents"
)

# Communication through shared context
async def agent_communication():
    """Demonstrate inter-agent communication."""
    from google.adk.sessions import InMemorySessionService
    from google.adk import Runner
    
    # Run agent A
    runner_a = Runner(
        app_name="agent_a_app",
        agent=agent_a,
        session_service=InMemorySessionService()
    )
    
    # Message from user to Agent A
    user_message = types.Content(
        role='user',
        parts=[types.Part(text="What's the latest news?")]
    )
    
    response_from_a = ""
    async for event in runner_a.run_async(
        user_id="user123",
        session_id="session_1",
        new_message=user_message
    ):
        if event.content:
            response_from_a = event.content.parts[0].text if event.content.parts else ""
    
    # Pass Agent A's response to Agent B
    if response_from_a:
        runner_b = Runner(
            app_name="agent_b_app",
            agent=agent_b,
            session_service=InMemorySessionService()
        )
        
        b_message = types.Content(
            role='user',
            parts=[types.Part(text=f"Process this for specialisation: {response_from_a}")]
        )
        
        async for event in runner_b.run_async(
            user_id="user123",
            session_id="session_2",
            new_message=b_message
        ):
            if event.content:
                print(f"Agent B response: {event.content.parts[0].text}")
```

### Hierarchical Agent Structures

#### Tree-Based Architecture

```python
from google.adk.agents import LlmAgent

# Root coordinator
root = LlmAgent(
    name="ceo_agent",
    model="gemini-2.5-pro",
    instruction="Provide executive overview and delegate to departments"
)

# Department level
dept_engineering = LlmAgent(
    name="engineering_director",
    model="gemini-2.5-pro",
    instruction="Manage engineering operations"
)

dept_marketing = LlmAgent(
    name="marketing_director",
    model="gemini-2.5-pro",
    instruction="Manage marketing operations"
)

# Team level (Engineering)
team_backend = LlmAgent(
    name="backend_lead",
    model="gemini-2.5-flash",
    instruction="Lead backend development"
)

team_frontend = LlmAgent(
    name="frontend_lead",
    model="gemini-2.5-flash",
    instruction="Lead frontend development"
)

# Team level (Marketing)
team_content = LlmAgent(
    name="content_lead",
    model="gemini-2.5-flash",
    instruction="Lead content creation"
)

team_analytics = LlmAgent(
    name="analytics_lead",
    model="gemini-2.5-flash",
    instruction="Lead analytics"
)

# Build hierarchy
dept_engineering.sub_agents = [team_backend, team_frontend]
dept_marketing.sub_agents = [team_content, team_analytics]
root.sub_agents = [dept_engineering, dept_marketing]
```

#### Forest-Based Architecture

```python
from google.adk.agents import LlmAgent

# Multiple independent hierarchies

# Financial System
cfo = LlmAgent(name="cfo", model="gemini-2.5-pro", instruction="Financial management")
controller = LlmAgent(name="controller", model="gemini-2.5-flash", instruction="Accounting")
treasurer = LlmAgent(name="treasurer", model="gemini-2.5-flash", instruction="Cash management")
cfo.sub_agents = [controller, treasurer]

# Operations System
coo = LlmAgent(name="coo", model="gemini-2.5-pro", instruction="Operations management")
supply_chain = LlmAgent(name="supply_chain", model="gemini-2.5-flash", instruction="Supply chain")
logistics = LlmAgent(name="logistics", model="gemini-2.5-flash", instruction="Logistics")
coo.sub_agents = [supply_chain, logistics]

# Use both systems with coordination
coordinator = LlmAgent(
    name="enterprise_coordinator",
    model="gemini-2.5-pro",
    instruction="Coordinate across CFO and COO lines",
    sub_agents=[cfo, coo]
)
```

---

## Tools Integration

### Function Declarations

#### Simple Function Declaration

```python
from google.adk.agents import LlmAgent

def get_current_time() -> str:
    """Get the current time in ISO format.
    
    Returns:
        Current time as ISO format string
    """
    from datetime import datetime
    return datetime.now().isoformat()

# Agent with tool
assistant = LlmAgent(
    name="time_assistant",
    model="gemini-2.5-flash",
    instruction="Provide the current time when asked",
    tools=[get_current_time]
)
```

#### Function with Parameters

```python
from google.adk.agents import LlmAgent
from typing import Optional

def search_product_database(
    product_name: str,
    category: Optional[str] = None,
    max_results: int = 10
) -> list:
    """Search for products in database.
    
    Args:
        product_name: Name or partial name of product
        category: Optional product category filter
        max_results: Maximum results to return (default 10)
        
    Returns:
        List of matching products
    """
    # Simulated database search
    return [
        {
            "name": product_name,
            "category": category or "General",
            "price": 99.99,
            "rating": 4.5
        }
    ]

product_agent = LlmAgent(
    name="product_finder",
    model="gemini-2.5-flash",
    instruction="Help customers find products using the search tool",
    tools=[search_product_database]
)
```

#### Type-Annotated Functions

```python
from google.adk.agents import LlmAgent
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class DataPoint:
    timestamp: str
    value: float
    status: str

def process_data(
    data_points: List[DataPoint],
    operation: str  # "average", "sum", "max", "min"
) -> Dict[str, float]:
    """Process data points with specified operation.
    
    Args:
        data_points: List of data points to process
        operation: Operation to perform
        
    Returns:
        Dictionary with results
    """
    values = [dp.value for dp in data_points]
    
    if operation == "average":
        return {"result": sum(values) / len(values)}
    elif operation == "sum":
        return {"result": sum(values)}
    elif operation == "max":
        return {"result": max(values)}
    elif operation == "min":
        return {"result": min(values)}
    
    return {"result": 0}

data_agent = LlmAgent(
    name="data_processor",
    model="gemini-2.5-flash",
    instruction="Process data using appropriate operations",
    tools=[process_data]
)
```

### Tool Registration

#### Dynamic Tool Registration

```python
from google.adk import Agent
from google.adk.tools import FunctionTool

# Create individual tools
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def square_number(n: float) -> float:
    """Square a number."""
    return n ** 2

# Create tools
add_tool = FunctionTool(func=add_numbers)
multiply_tool = FunctionTool(func=multiply_numbers)
square_tool = FunctionTool(func=square_number)

# Register with agent
calculator = Agent(
    name="calculator",
    model="gemini-2.5-flash",
    instruction="Perform mathematical operations",
    tools=[add_tool, multiply_tool, square_tool]
)
```

#### Tool Grouping

```python
from google.adk import Agent
from google.adk.tools import ToolSet

class MathToolSet(ToolSet):
    """Collection of mathematical tools."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

math_tools = MathToolSet()

agent = Agent(
    name="math_assistant",
    model="gemini-2.5-flash",
    instruction="Help with mathematical calculations",
    tools=math_tools.get_tools()
)
```

### Google-Specific Tools

#### Google Search

```python
from google.adk import Agent
from google.adk.tools import google_search

search_agent = Agent(
    name="web_searcher",
    model="gemini-2.5-flash",
    description="Searches the web for information",
    instruction="""You are a web search assistant. When asked for current information,
    use google_search to find accurate, up-to-date results. Always cite sources.""",
    tools=[google_search]
)

# The agent can now use google_search to:
# - Find current events and news
# - Look up recent information
# - Verify facts
# - Discover new information not in training data
```

#### Google Calendar (Simulated Integration)

```python
from google.adk import Agent
from google.adk.tools import calendar_schedule_event, calendar_list_events
from typing import Optional

# Create calendar agent
calendar_agent = Agent(
    name="calendar_assistant",
    model="gemini-2.5-flash",
    description="Manages calendar events",
    instruction="""You are a calendar assistant. Help users schedule events,
    check availability, and manage their calendar. Use calendar tools to
    create events and retrieve schedule information.""",
    tools=[calendar_schedule_event, calendar_list_events]
)

# The agent can handle:
# - "Schedule a meeting for tomorrow at 2pm"
# - "What's on my calendar next week?"
# - "Find a time slot for a meeting with three people"
```

#### BigQuery Integration

```python
from google.adk import Agent
from google.adk.integrations.bigquery import BigQueryToolset

# Initialise BigQuery toolset (no project_id arg — uses ADC credentials)
bq_tools = BigQueryToolset()

# Create analytics agent
analytics_agent = Agent(
    name="data_analyst",
    model="gemini-2.5-pro",
    description="Analyses data using BigQuery",
    instruction="""You are a data analyst with access to BigQuery.
    When asked about data:
    1. Write appropriate SQL queries
    2. Execute queries to get results
    3. Analyse and interpret results
    4. Provide actionable insights""",
    tools=[bq_tools],
)

# The agent can:
# - Execute SQL queries
# - Analyse datasets
# - Generate reports
# - Identify trends
```

### Custom Tool Creation

#### Basic Custom Tool

```python
from google.adk import Agent
from typing import List
import statistics

def calculate_advanced_stats(numbers: List[float]) -> dict:
    """Calculate comprehensive statistics.
    
    Args:
        numbers: List of numerical values
        
    Returns:
        Dictionary of statistical measures
    """
    if not numbers:
        return {"error": "Empty list provided"}
    
    sorted_nums = sorted(numbers)
    
    return {
        "count": len(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "mode": statistics.mode(numbers) if len(set(numbers)) < len(numbers) else None,
        "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0,
        "variance": statistics.variance(numbers) if len(numbers) > 1 else 0,
        "min": min(numbers),
        "max": max(numbers),
        "range": max(numbers) - min(numbers),
        "q1": sorted_nums[len(sorted_nums) // 4],
        "q3": sorted_nums[3 * len(sorted_nums) // 4]
    }

# Create agent with custom tool
stats_agent = Agent(
    name="statistics_expert",
    model="gemini-2.5-flash",
    instruction="Provide statistical analysis of datasets",
    tools=[calculate_advanced_stats]
)
```

#### Tool with Context

```python
from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from typing import List

def add_to_history(
    item: str,
    category: str,
    tool_context: ToolContext
) -> str:
    """Add item to history tracked in tool context.
    
    Args:
        item: Item to add
        category: Category for organisation
        tool_context: Provides access to session state
        
    Returns:
        Confirmation message
    """
    # Initialise history if needed
    if "history" not in tool_context.state:
        tool_context.state["history"] = {}
    
    if category not in tool_context.state["history"]:
        tool_context.state["history"][category] = []
    
    # Add item
    tool_context.state["history"][category].append(item)
    
    return f"Added '{item}' to {category}. Total items: {len(tool_context.state['history'][category])}"

history_agent = Agent(
    name="history_tracker",
    model="gemini-2.5-flash",
    instruction="Track items and maintain history",
    tools=[add_to_history]
)
```

### Tool Schemas with Parameters

#### Detailed Parameter Schemas

```python
from google.adk import Agent
from typing import Literal, List
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Database query request."""
    query_type: Literal["SELECT", "INSERT", "UPDATE", "DELETE"]
    table_name: str = Field(..., description="Name of the table")
    columns: List[str] = Field(default_factory=list, description="Columns to retrieve")
    where_clause: str = Field(default="", description="WHERE clause conditions")
    limit: int = Field(default=100, description="Result limit")

def execute_query(request: QueryRequest) -> dict:
    """Execute database query.
    
    Args:
        request: Query request parameters
        
    Returns:
        Query results or status
    """
    return {
        "status": "success",
        "query_type": request.query_type,
        "rows_affected": 5,
        "results": []
    }

db_agent = Agent(
    name="database_manager",
    model="gemini-2.5-pro",
    instruction="Execute database queries safely",
    tools=[execute_query]
)
```

#### Enum and Literal Parameters

```python
from google.adk import Agent
from typing import Literal
from enum import Enum

class ReportFormat(str, Enum):
    """Available report formats."""
    PDF = "pdf"
    JSON = "json"
    EXCEL = "excel"
    HTML = "html"

def generate_report(
    report_type: Literal["sales", "inventory", "customer"],
    format: ReportFormat = ReportFormat.PDF,
    include_charts: bool = True
) -> dict:
    """Generate business report.
    
    Args:
        report_type: Type of report to generate
        format: Output format (pdf, json, excel, html)
        include_charts: Whether to include visualisations
        
    Returns:
        Report data or file location
    """
    return {
        "report_type": report_type,
        "format": format.value,
        "has_charts": include_charts,
        "file_path": f"/reports/{report_type}_report.{format.value}"
    }

reporting_agent = Agent(
    name="report_generator",
    model="gemini-2.5-flash",
    instruction="Generate reports in requested formats",
    tools=[generate_report]
)
```

### Error Handling in Tools

#### Try-Except Pattern

```python
from google.adk import Agent
from typing import Optional

def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0
) -> dict:
    """Safely divide with error handling.
    
    Args:
        numerator: Number to divide
        denominator: Divisor
        default: Default value if error occurs
        
    Returns:
        Result dictionary with status
    """
    try:
        if denominator == 0:
            return {
                "status": "error",
                "error_type": "ZeroDivisionError",
                "message": "Cannot divide by zero",
                "result": default
            }
        
        result = numerator / denominator
        return {
            "status": "success",
            "result": result
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
            "result": default
        }

calc_agent = Agent(
    name="safe_calculator",
    model="gemini-2.5-flash",
    instruction="Perform calculations safely",
    tools=[safe_divide]
)
```

#### Validation Error Handling

```python
from google.adk import Agent
from pydantic import ValidationError, BaseModel, Field
from typing import Optional

class DataInput(BaseModel):
    """Validated data input."""
    values: list = Field(..., min_items=1)
    operation: str
    max_size: int = Field(default=1000000)

def validate_and_process(data_dict: dict) -> dict:
    """Validate input and process.
    
    Args:
        data_dict: Input data to validate
        
    Returns:
        Processing result or error
    """
    try:
        validated = DataInput(**data_dict)
        return {
            "status": "success",
            "message": f"Processing {len(validated.values)} items"
        }
    
    except ValidationError as e:
        return {
            "status": "validation_error",
            "errors": e.errors(),
            "message": "Input validation failed"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

validator_agent = Agent(
    name="data_validator",
    model="gemini-2.5-flash",
    instruction="Validate data before processing",
    tools=[validate_and_process]
)
```

### Async Tool Execution

#### Async Tool Functions

```python
from google.adk import Agent
import asyncio

async def fetch_data_async(url: str, timeout: int = 10) -> dict:
    """Fetch data from URL asynchronously.
    
    Args:
        url: URL to fetch from
        timeout: Request timeout in seconds
        
    Returns:
        Response data
    """
    try:
        # Simulated async fetch
        await asyncio.sleep(1)
        return {
            "status": "success",
            "url": url,
            "data": {"sample": "data"},
            "fetch_time": 1000
        }
    
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "url": url,
            "error": "Request timed out"
        }

async_agent = Agent(
    name="async_data_fetcher",
    model="gemini-2.5-flash",
    instruction="Fetch data asynchronously from sources",
    tools=[fetch_data_async]
)
```

#### Tool with Streaming Response

```python
from google.adk import Agent
from typing import AsyncGenerator

async def stream_data(source: str) -> AsyncGenerator[str, None]:
    """Stream data from source asynchronously.
    
    Args:
        source: Data source identifier
        
    Yields:
        Data chunks
    """
    for i in range(5):
        await asyncio.sleep(0.5)
        yield f"Chunk {i}: Data from {source}"

streaming_agent = Agent(
    name="data_streamer",
    model="gemini-2.5-flash",
    instruction="Stream data from sources",
    tools=[stream_data]
)
```

---

## Structured Output

### Response Schemas

#### Defining JSON Schemas

```python
from google.adk import Agent
from pydantic import BaseModel, Field
from typing import List, Optional

class PersonSchema(BaseModel):
    """Schema for person response."""
    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=0, le=150, description="Age in years")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")

class CompanySchema(BaseModel):
    """Schema for company response."""
    name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Industry sector")
    employee_count: int = Field(..., ge=1, description="Number of employees")
    founded_year: int = Field(..., description="Year founded")
    headquarters: str = Field(..., description="Headquarters location")

structured_agent = Agent(
    name="data_extractor",
    model="gemini-2.5-flash",
    instruction="Extract and structure information as JSON",
    # Response schema would be used in generation settings
)
```

#### Complex Nested Schemas

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    """Address information."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str

class Contact(BaseModel):
    """Contact information."""
    email: str
    phone: Optional[str] = None
    addresses: List[Address]

class Organization(BaseModel):
    """Organization information."""
    name: str
    industry: str
    founded: datetime
    headquarters: Address
    contacts: List[Contact]
    employee_count: int
    revenue_millions: Optional[float] = None
```

### JSON Mode Configuration

#### Enabling JSON Mode

```python
from google.adk import Agent
from google.genai import types
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    """Analysis result structure."""
    topic: str
    key_points: list[str]
    sentiment: str
    confidence: float

# output_schema enforces structured JSON output (disables tool use)
json_agent = Agent(
    name="json_analyst",
    model="gemini-2.5-flash",
    instruction="Analyse text and return JSON-formatted results",
    output_schema=AnalysisResult,
    generate_content_config=types.GenerateContentConfig(
        max_output_tokens=2048,
    ),
)
```

### Pydantic Models for Validation

#### Output Validation with Pydantic

```python
from google.adk import Agent
from pydantic import BaseModel, validator
from typing import List

class Article(BaseModel):
    """Validated article structure."""
    title: str
    content: str
    author: str
    tags: List[str]
    word_count: int
    
    @validator('title')
    def title_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v
    
    @validator('tags')
    def valid_tags(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
        return v

article_agent = Agent(
    name="article_writer",
    model="gemini-2.5-flash",
    instruction="Write articles that conform to Article schema"
)

# Validate agent output
def process_agent_output(output: dict) -> Article:
    try:
        return Article(**output)
    except Exception as e:
        print(f"Validation failed: {e}")
        return None
```

### Output Parsing Strategies

#### Parse and Validate

```python
from google.adk import Agent
import json
from typing import Optional

def parse_agent_output(
    raw_output: str,
    expected_format: str = "json"
) -> Optional[dict]:
    """Parse agent output to expected format.
    
    Args:
        raw_output: Raw output from agent
        expected_format: Expected format (json, yaml, xml)
        
    Returns:
        Parsed output or None if parsing fails
    """
    try:
        if expected_format == "json":
            return json.loads(raw_output)
        
        # Add other format handlers as needed
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    
    return None

# Use in agent workflow
parsing_agent = Agent(
    name="parser",
    model="gemini-2.5-flash",
    instruction="Output structured information as valid JSON"
)
```

### Schema Enforcement

#### Strict Schema Validation

```python
from google.adk import Agent
from pydantic import BaseModel, ValidationError, Field
from typing import Optional
import json

class StrictResponseSchema(BaseModel):
    """Strictly enforced response schema."""
    action: str = Field(..., pattern="^(create|read|update|delete)$")
    resource_type: str = Field(..., pattern="^[a-z_]+$")
    resource_id: Optional[int] = Field(None, ge=1)
    status: str = Field(..., pattern="^(success|error|pending)$")
    message: str = Field(..., min_length=1, max_length=500)

def enforce_schema(agent_output: str) -> StrictResponseSchema:
    """Enforce strict schema on output.
    
    Args:
        agent_output: Raw agent output
        
    Returns:
        Validated schema object
        
    Raises:
        ValidationError: If output doesn't match schema
    """
    try:
        data = json.loads(agent_output)
        return StrictResponseSchema(**data)
    
    except json.JSONDecodeError:
        raise ValueError("Output is not valid JSON")
    
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")

strict_agent = Agent(
    name="strict_responder",
    model="gemini-2.5-flash",
    instruction="Respond with strictly formatted JSON following the defined schema"
)
```

### Complex Nested Structures

#### Multi-Level Nested Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class NestedSchema(BaseModel):
    """Deeply nested schema example."""
    
    class Metadata(BaseModel):
        created: datetime
        updated: datetime
        version: str
    
    class Tag(BaseModel):
        name: str
        value: str
    
    class ContentBlock(BaseModel):
        type: str
        data: Dict[str, any]
        tags: List[Tag]
    
    class Author(BaseModel):
        name: str
        email: str
        bio: Optional[str] = None
    
    # Main structure
    title: str
    authors: List[Author]
    content: List[ContentBlock]
    metadata: Metadata
    status: str
    related_ids: List[int] = Field(default_factory=list)

# Example usage
complex_agent = Agent(
    name="complex_content_creator",
    model="gemini-2.5-pro",
    instruction="Generate complex, nested content structures"
)
```

---

## Model Context Protocol (MCP)

### MCP in ADK

#### Understanding MCP

Model Context Protocol (MCP) is a standardised protocol that enables:

- **Tool Sharing:** Expose ADK agent tools as standardised MCP resources
- **Client-Server Communication:** Allow external systems to access agent capabilities
- **Standardised Interfaces:** Work with any MCP-compatible client

```python
from google.adk import Agent

# ADK agents can be exposed via MCP
mcp_enabled_agent = Agent(
    name="mcp_agent",
    model="gemini-2.5-flash",
    description="Agent exposed via Model Context Protocol",
    instruction="Serve requests through MCP interface"
)
```

### Exposing ADK Agents via MCP

#### MCP Server Setup

```python
from google.adk import Agent
from google.adk.mcp import MCPServer, MCPResource

# Create agents
agent = Agent(
    name="main_agent",
    model="gemini-2.5-flash",
    instruction="Main agent"
)

# Set up MCP server
mcp_server = MCPServer(
    name="adk_agent_server",
    version="1.0.0"
)

# Register agents as MCP resources
mcp_server.add_resource(
    MCPResource(
        name="main_agent",
        description="Main ADK agent via MCP",
        agent=agent
    )
)

# Start server
async def start_mcp_server():
    await mcp_server.start(
        host="localhost",
        port=5000,
        ssl_enabled=False
    )
```

#### MCP Client Usage

```python
from google.adk.mcp import MCPClient

# Connect to MCP server
client = MCPClient(
    server_url="http://localhost:5000",
    name="adk_client"
)

# Use agent through MCP
async def use_agent_via_mcp():
    from google.genai import types
    
    message = types.Content(
        role='user',
        parts=[types.Part(text="What is 2+2?")]
    )
    
    response = await client.call_resource(
        resource_name="main_agent",
        method="run",
        parameters={"message": message}
    )
    
    return response
```

### Tool Sharing

#### Expose Tools via MCP

```python
from google.adk.mcp import MCPServer, MCPTool

def calculate_total(items: list[float], tax_rate: float) -> float:
    """Calculate total with tax."""
    subtotal = sum(items)
    tax = subtotal * tax_rate
    return subtotal + tax

# Create MCP tool
mcp_tool = MCPTool(
    name="calculate_total",
    description="Calculate total with tax",
    function=calculate_total
)

# Register with MCP server
mcp_server.add_tool(mcp_tool)
```

#### Shared Tool Registry

```python
from google.adk.mcp import MCPToolRegistry

# Create registry
registry = MCPToolRegistry()

# Register multiple tools
registry.register("calculate_total", calculate_total)
registry.register("apply_discount", apply_discount)
registry.register("format_currency", format_currency)

# Create server with registry
mcp_server = MCPServer(
    name="shared_tools_server",
    tool_registry=registry
)
```

### Context Management

#### Managing Context Over MCP

```python
from google.adk.mcp import MCPContextManager

# Create context manager
context_manager = MCPContextManager()

# Store context
context_manager.set("user_id", "user123")
context_manager.set("session_id", "sess456")
context_manager.set("preferences", {"language": "en"})

# Retrieve context
user_id = context_manager.get("user_id")
preferences = context_manager.get("preferences")

# Use in MCP calls
async def make_context_aware_call():
    response = await mcp_server.call_with_context(
        resource="agent",
        context=context_manager.get_all()
    )
    return response
```

### Integration Patterns

#### MCP with Multiple Agents

```python
from google.adk.mcp import MCPServer

# Create server with multiple agents
server = MCPServer(name="multi_agent_server")

# Register different agents
for agent_config in agent_configs:
    server.add_resource(
        MCPResource(
            name=agent_config["name"],
            description=agent_config["description"],
            agent=create_agent(agent_config)
        )
    )

# Clients can access multiple agents through single MCP server
```

#### MCP Gateway Pattern

```python
from google.adk.mcp import MCPGateway

# Create gateway
gateway = MCPGateway(
    name="adk_gateway",
    default_model="gemini-2.5-flash"
)

# Add multiple backend servers
gateway.add_backend_server("server_1", "http://backend1:5000")
gateway.add_backend_server("server_2", "http://backend2:5000")

# Route requests to appropriate backend
# Clients connect to gateway instead of individual servers
```

---

## Agentic Patterns

### ReAct with Gemini

#### ReAct Pattern Implementation

The ReAct (Reasoning + Acting) pattern enables agents to think through problems and take actions:

```python
from google.adk import Agent
from google.adk.tools import google_search
from typing import List, Dict

# Define ReAct agent
react_agent = Agent(
    name="react_reasoner",
    model="gemini-2.5-pro",
    description="Reasons and acts to solve problems",
    instruction="""You are a reasoning agent. For each problem:

1. **THOUGHT**: Think about the current state and what you need to know.
2. **ACTION**: Decide what to do - call a tool or make a conclusion.
3. **OBSERVATION**: Observe the results of your action.
4. **REASONING**: Update your understanding based on observations.
5. **FINAL ANSWER**: Provide your conclusion when you have enough information.

Always follow this pattern explicitly in your reasoning.""",
    tools=[google_search],
    max_iterations=5
)

# Example execution traces show the agent's thinking:
# THOUGHT: I need to find information about recent AI developments.
# ACTION: I'll search for "latest AI breakthroughs 2025"
# OBSERVATION: Found articles about new transformer architectures
# REASONING: The information shows progress in efficiency improvements
# THOUGHT: I should search for more specific details
# ACTION: Searching for "transformer efficiency 2025"
# FINAL ANSWER: Based on my research, recent developments include...
```

#### Chain of Thought with Gemini

```python
from google.adk import Agent

cot_agent = Agent(
    name="cot_thinker",
    model="gemini-2.5-pro",
    instruction="""Solve problems using chain-of-thought reasoning:

For each problem, think step-by-step:
- Step 1: What is the problem asking?
- Step 2: What information do I have?
- Step 3: What approach should I take?
- Step 4: Implement the approach step-by-step
- Step 5: Verify the solution
- Step 6: Present the final answer

Show your complete reasoning process.""",
    max_iterations=10
)
```

### Function Calling Workflows

#### Multi-Step Function Calling

```python
from google.adk import Agent

def search_product(name: str) -> dict:
    """Search for product by name."""
    return {"product_id": "123", "name": name, "price": 99.99}

def check_inventory(product_id: str) -> dict:
    """Check inventory for product."""
    return {"product_id": product_id, "in_stock": True, "quantity": 50}

def process_order(product_id: str, quantity: int) -> dict:
    """Process order for product."""
    return {"order_id": "ORD123", "status": "confirmed", "total": 999.90}

order_agent = Agent(
    name="order_processor",
    model="gemini-2.5-pro",
    instruction="""Process customer orders:
1. Search for product customer wants
2. Check if product is in stock
3. If in stock, process the order
4. Confirm order details to customer""",
    tools=[search_product, check_inventory, process_order]
)
```

### Multi-Step Reasoning

#### Complex Problem Decomposition

```python
from google.adk import Agent

def analyze_market_trends() -> dict:
    """Analyse market trends."""
    return {"trend": "growth", "rate": "5.2%"}

def evaluate_competition() -> dict:
    """Evaluate competition."""
    return {"competitors": 3, "market_share": "25%"}

def assess_resources() -> dict:
    """Assess available resources."""
    return {"budget": 1000000, "team": 50}

def generate_strategy() -> dict:
    """Generate business strategy."""
    return {"strategy": "expansion", "timeline": "12 months"}

strategy_agent = Agent(
    name="strategic_planner",
    model="gemini-2.5-pro",
    instruction="""Develop comprehensive business strategy:

1. Analyze market trends to understand landscape
2. Evaluate competition to identify opportunities
3. Assess available resources for realistic planning
4. Generate strategy based on analysis
5. Provide implementation roadmap

Reason through each step carefully.""",
    tools=[
        analyze_market_trends,
        evaluate_competition,
        assess_resources,
        generate_strategy
    ],
    max_iterations=10
)
```

### Planning and Execution

#### Plan-Then-Execute Pattern

```python
from google.adk.agents import SequentialAgent, LlmAgent

# Planner agent
planner = LlmAgent(
    name="planner",
    model="gemini-2.5-pro",
    instruction="""Create detailed execution plans. For each task:
1. Identify all subtasks
2. Order them logically
3. Estimate resources needed
4. Identify dependencies
5. Create timeline"""
)

# Executor agent
executor = LlmAgent(
    name="executor",
    model="gemini-2.5-flash",
    instruction="""Execute plans step-by-step:
1. Receive plan from planner
2. Execute each step in order
3. Track progress
4. Report status
5. Flag any issues"""
)

# Manager agent
manager = LlmAgent(
    name="manager",
    model="gemini-2.5-pro",
    instruction="Coordinate planning and execution"
)

# Create workflow
manager.sub_agents = [planner, executor]
```

### Self-Reflection Loops

#### Reflection Pattern

```python
from google.adk import Agent

def initial_attempt(problem: str) -> str:
    """Make initial attempt."""
    return "Initial answer"

def check_correctness(answer: str) -> dict:
    """Check if answer is correct."""
    return {
        "is_correct": False,
        "issues": ["Missing detail", "Incomplete reasoning"]
    }

def refine_answer(answer: str, issues: list) -> str:
    """Refine answer based on issues."""
    return "Refined answer"

reflective_agent = Agent(
    name="self_reflective_agent",
    model="gemini-2.5-pro",
    instruction="""Solve problems with self-reflection:

1. Generate initial answer
2. Check answer for correctness
3. If issues found, refine answer
4. Re-check refined answer
5. Iterate until satisfied

Always reflect on your work before finalising.""",
    tools=[initial_attempt, check_correctness, refine_answer]
)
```

### Autonomous Task Completion

#### Full Autonomy Agent

```python
from google.adk import Agent

autonomous_agent = Agent(
    name="autonomous_task_completer",
    model="gemini-2.5-pro",
    description="Completes tasks fully autonomously",
    instruction="""You are autonomous in completing assigned tasks:

1. Understand the task completely
2. Break down into subtasks
3. Execute each subtask
4. Verify completion
5. Handle any issues that arise
6. Report on completion

Work independently until the task is fully complete.""",
    max_iterations=20,
    max_total_tokens=8192
)
```

---

## Memory Systems

### Conversation History Management

#### In-Memory Conversation History

```python
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
import asyncio

async def manage_conversation_history():
    """Manage conversation history in memory."""
    from google.adk import Agent, Runner
    
    agent = Agent(
        name="conversation_agent",
        model="gemini-2.5-flash",
        instruction="Be a helpful assistant that remembers context"
    )
    
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="conversation_app",
        agent=agent,
        session_service=session_service
    )
    
    # Create session
    session = await session_service.create_session(
        app_name="conversation_app",
        user_id="user123",
        state={"conversation": []}
    )
    
    # Send multiple messages
    messages = [
        "Hello, my name is Alice",
        "What did I just tell you?",
        "Can you remember my name?"
    ]
    
    for msg in messages:
        user_message = types.Content(
            role='user',
            parts=[types.Part(text=msg)]
        )
        
        async for event in runner.run_async(
            user_id="user123",
            session_id=session.id,
            new_message=user_message
        ):
            if event.content:
                # Agent remembers previous context
                print(f"Agent: {event.content.parts[0].text}")
```

#### Persistent Conversation Storage

```python
from google.adk.sessions import DatabaseSessionService

async def setup_persistent_storage():
    """Set up persistent conversation storage."""
    
    # DatabaseSessionService supports SQLite, Postgres, MySQL, Spanner
    session_service = DatabaseSessionService(
        db_url="postgresql+asyncpg://user:pass@localhost/adk"
    )
    
    # Create session (automatically persisted)
    session = await session_service.create_session(
        app_name="persistent_app",
        user_id="user456",
        state={
            "preferences": {"language": "en"},
            "history": []
        }
    )
    
    # Session data is persisted to the DB and survives restarts
    retrieved = await session_service.get_session(
        app_name="persistent_app",
        user_id="user456",
        session_id=session.id
    )
    
    return retrieved
```

### Context Caching with Gemini

#### Static Context Caching

```python
from google.adk import Agent
from google.adk.agents import ContextCacheConfig

# Large static context that doesn't change
LARGE_SYSTEM_CONTEXT = """You are an expert in multiple domains:
- Software Engineering
- Data Science
- Cloud Architecture
- DevOps
- Security

You have extensive knowledge of these domains and can provide detailed,
accurate information. Always cite relevant best practices and standards.""" * 20  # Make it large

cache_agent = Agent(
    name="cached_agent",
    model="gemini-2.5-pro",
    instruction="Provide expert advice on technical topics",
    static_instruction=LARGE_SYSTEM_CONTEXT,
    max_total_tokens=8192
)

# The large system context is cached on first use
# Subsequent requests reuse cached context, reducing latency and cost
```

#### Dynamic Caching Configuration

```python
from google.adk.agents import ContextCacheConfig

cache_config = ContextCacheConfig(
    enable_auto_caching=True,
    ttl_seconds=3600,  # Cache for 1 hour
    min_cache_size=1000,  # Minimum tokens for caching
    max_cache_entries=10,  # Maximum concurrent cached contexts
)

cached_agent = Agent(
    name="dynamic_cache_agent",
    model="gemini-2.5-pro",
    instruction="Respond to queries",
    cache_config=cache_config
)
```

### Firestore for Persistent Memory

#### Storing Agent Memory in Firestore

```python
from google.cloud import firestore
from typing import Dict, List

class FirestoreMemoryService:
    """Custom memory service using Firestore."""
    
    def __init__(self, project_id: str, collection: str = "agent_memory"):
        self.db = firestore.Client(project=project_id)
        self.collection = collection
    
    async def save_memory(
        self,
        agent_id: str,
        memory_type: str,
        content: Dict
    ) -> str:
        """Save memory to Firestore."""
        doc_ref = self.db.collection(self.collection).document()
        doc_ref.set({
            "agent_id": agent_id,
            "type": memory_type,
            "content": content,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return doc_ref.id
    
    async def retrieve_memory(
        self,
        agent_id: str,
        memory_type: str
    ) -> List[Dict]:
        """Retrieve memories of type."""
        docs = self.db.collection(self.collection).where(
            "agent_id", "==", agent_id
        ).where(
            "type", "==", memory_type
        ).stream()
        
        return [doc.to_dict() for doc in docs]
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict
    ) -> None:
        """Update existing memory."""
        self.db.collection(self.collection).document(memory_id).update(updates)

# Usage
memory_service = FirestoreMemoryService(project_id="my-gcp-project")

# Save memory
await memory_service.save_memory(
    agent_id="agent_123",
    memory_type="conversation",
    content={"topic": "Python", "duration": 300}
)

# Retrieve memories
memories = await memory_service.retrieve_memory(
    agent_id="agent_123",
    memory_type="conversation"
)
```

### Vector Search with Vertex AI

#### Vector-based Semantic Search

```python
from google.adk.memory import VectorSearchMemory
from google.cloud import aiplatform
import numpy as np

class SemanticMemoryService:
    """Use vector search for semantic memory retrieval."""
    
    def __init__(self, project_id: str, index_endpoint_id: str):
        self.project_id = project_id
        self.index_endpoint_id = index_endpoint_id
        self.client = aiplatform.MatchingEngineIndexEndpointClient()
    
    async def store_semantic_memory(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: dict
    ) -> str:
        """Store text with embedding."""
        # Store in vector index
        doc_id = str(hash(text))
        # Implementation depends on vector store
        return doc_id
    
    async def retrieve_similar_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list:
        """Retrieve semantically similar memories."""
        # Query vector store for similar embeddings
        similar_docs = []  # Results from search
        return similar_docs

# Usage for RAG-style memory retrieval
semantic_memory = SemanticMemoryService(
    project_id="my-gcp-project",
    index_endpoint_id="my-index-endpoint"
)
```

### Custom Memory Implementations

#### Abstract Memory Service

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class CustomMemoryService(ABC):
    """Base class for custom memory implementations."""
    
    @abstractmethod
    async def save(self, key: str, value: Dict) -> None:
        """Save memory."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Dict]:
        """Retrieve memory."""
        pass
    
    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List memory keys."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete memory."""
        pass

# Implementation using Redis
class RedisMemoryService(CustomMemoryService):
    """Memory service using Redis."""
    
    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)
    
    async def save(self, key: str, value: Dict) -> None:
        """Save to Redis."""
        import json
        await self.redis.set(key, json.dumps(value))
    
    async def retrieve(self, key: str) -> Optional[Dict]:
        """Retrieve from Redis."""
        import json
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern."""
        return await self.redis.keys(pattern)
    
    async def delete(self, key: str) -> None:
        """Delete from Redis."""
        await self.redis.delete(key)
```

### Memory Lifecycle Management

#### Memory Expiration and Cleanup


```python
from datetime import datetime, timedelta
from typing import Dict

class MemoryLifecycleManager:
    """Manage memory lifecycle with expiration."""
    
    def __init__(self, memory_service: CustomMemoryService):
        self.memory_service = memory_service
    
    async def save_with_ttl(
        self,
        key: str,
        value: Dict,
        ttl_minutes: int = 60
    ) -> None:
        """Save memory with expiration."""
        expiration = datetime.now() + timedelta(minutes=ttl_minutes)
        value['_expires_at'] = expiration.isoformat()
        await self.memory_service.save(key, value)
    
    async def retrieve_if_valid(self, key: str) -> Optional[Dict]:
        """Retrieve memory only if not expired."""
        data = await self.memory_service.retrieve(key)
        
        if not data:
            return None
        
        if '_expires_at' in data:
            expires_at = datetime.fromisoformat(data['_expires_at'])
            if datetime.now() > expires_at:
                await self.memory_service.delete(key)
                return None
        
        return data
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memories."""
        keys = await self.memory_service.list_keys()
        deleted_count = 0
        
        for key in keys:
            if await self.retrieve_if_valid(key) is None:
                deleted_count += 1
        
        return deleted_count

# Usage
lifecycle_manager = MemoryLifecycleManager(memory_service)

# Save memory that expires in 30 minutes
await lifecycle_manager.save_with_ttl(
    "user_session_123",
    {"preferences": {...}},
    ttl_minutes=30
)

# Retrieve only if not expired
session_data = await lifecycle_manager.retrieve_if_valid("user_session_123")
```


---

## Context Engineering

### System Instruction Design

#### Effective System Instruction Structure

```python
COMPREHENSIVE_SYSTEM_INSTRUCTION = """You are an expert technical assistant with deep knowledge of cloud computing.

## YOUR ROLE
You provide accurate, detailed technical guidance on AWS, Google Cloud, and Azure platforms.

## EXPERTISE AREAS
- Cloud Architecture
- Serverless Computing
- Containerisation
- DevOps and CI/CD
- Security Best Practices
- Cost Optimisation

## HOW YOU OPERATE
1. Understand the specific use case and constraints
2. Provide multiple approaches when appropriate
3. Explain trade-offs between options
4. Include code examples and configuration samples
5. Reference best practices and documentation

## COMMUNICATION STYLE
- Clear and professional
- Avoid unnecessary jargon
- Explain concepts for learners
- Provide actionable recommendations

## LIMITATIONS
- Acknowledge when information might be outdated
- Recommend checking official documentation
- Admit when you don't have sufficient information
- Suggest consulting with specialists for critical decisions

## OUTPUT FORMAT
- Structure responses hierarchically
- Use code blocks for examples
- Provide step-by-step instructions
- Include configuration examples
- Link to relevant documentation
"""

expert_agent = Agent(
    name="cloud_expert",
    model="gemini-2.5-pro",
    instruction=COMPREHENSIVE_SYSTEM_INSTRUCTION,
    description="Expert cloud computing assistant"
)
```

### Few-Shot Prompting

#### Providing Examples for Better Performance

```python
FEW_SHOT_INSTRUCTION = """You are a code reviewer. Review code for quality, security, and performance.

## EXAMPLE 1: Good Review
User: "Review this Python function"
Code:
```python
def calculate_total(items: list[float]) -> float:
    '''Calculate total of items with validation.'''
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if not items:
        raise ValueError("items list cannot be empty")
    return sum(items)
```

Review:
✓ **Strengths:**
- Type hints for clarity
- Comprehensive error handling
- Clear docstring
- Validates inputs

✓ **Suggestions:**
- Consider using sum() directly if validated elsewhere
- Could add logging for debugging

## EXAMPLE 2: Code with Issues
User: "Review this code"
Code:
```python
def process_data(data):
    result = []
    for item in data:
        try:
            result.append(item * 2)
        except:
            pass
    return result
```

Review:
✗ **Issues Found:**
- No type hints - unclear what data should be
- Bare except clause - silently ignores errors
- No documentation
- Could be replaced with list comprehension

**Recommended Fix:**
```python
def process_data(data: list[int]) -> list[int]:
    '''Double each item in the list.'''
    return [item * 2 for item in data]
```

---

Now review the provided code following these examples:
"""

reviewer = Agent(
    name="code_reviewer",
    model="gemini-2.5-pro",
    instruction=FEW_SHOT_INSTRUCTION
)
```

### Context Caching Strategies

#### Optimal Context Caching

```python
from google.adk import Agent
from google.adk.agents import ContextCacheConfig

# Strategy 1: Static long-form content caching
LARGE_DOCUMENTATION = """
[Comprehensive API documentation - large content that doesn't change]
""" * 100

static_cache_agent = Agent(
    name="doc_agent",
    model="gemini-2.5-pro",
    static_instruction=LARGE_DOCUMENTATION,
    instruction="Answer questions about the API",
    cache_config=ContextCacheConfig(
        enable_auto_caching=True,
        ttl_seconds=86400  # 24 hours
    )
)

# Strategy 2: Dynamic context caching with large examples
LARGE_CONTEXT_WITH_EXAMPLES = """
You are a code generator.

## EXAMPLES:
[Many code examples - could be hundreds of lines]

## GUIDELINES:
[Detailed guidelines - could be thousands of words]
"""

example_cache_agent = Agent(
    name="code_gen",
    model="gemini-2.5-pro",
    static_instruction=LARGE_CONTEXT_WITH_EXAMPLES,
    instruction="Generate code based on requirements",
    cache_config=ContextCacheConfig(
        enable_auto_caching=True,
        ttl_seconds=3600,
        min_cache_size=1000
    )
)
```

### Prompt Engineering for Gemini

#### Multi-Turn Conversation Optimization

```python
OPTIMISED_INSTRUCTION = """You are a conversational AI assistant optimised for Gemini.

## CONVERSATION CHARACTERISTICS
- Maintain context across multiple turns
- Reference previous messages naturally
- Build on earlier information
- Clarify ambiguities proactively

## RESPONSE STRATEGY
- First response: Comprehensive, sets context
- Subsequent responses: Build on established context
- Use pronouns to reference previous statements
- Acknowledge and incorporate user feedback

## GEMINI OPTIMISATIONS
- Leverage reasoning steps for complex problems
- Use function calling for concrete actions
- Maintain conversational flow even with tools
- Provide detailed explanations when requested
"""

conversational_agent = Agent(
    name="conversation_specialist",
    model="gemini-2.5-pro",
    instruction=OPTIMISED_INSTRUCTION
)
```

### Context Window Optimization

#### Efficient Context Usage

```python
from google.adk import Agent
from google.genai import types

efficient_agent = Agent(
    name="efficient_agent",
    model="gemini-2.5-flash",  # Flash for efficiency
    instruction="Provide concise, direct answers",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=1024,
        stop_sequences=["END", "###"],
    ),
)
```

### Dynamic Context Injection

#### Runtime Context Addition

```python
from google.adk import Agent
from google.genai import types

async def inject_runtime_context():
    """Inject dynamic context at runtime."""
    
    base_agent = Agent(
        name="contextual_agent",
        model="gemini-2.5-pro",
        instruction="Respond to user queries"
    )
    
    # Gather runtime context
    runtime_context = {
        "current_user": "alice@example.com",
        "user_tier": "premium",
        "request_time": "2025-01-15T10:30:00Z",
        "available_features": ["feature_a", "feature_b", "feature_c"]
    }
    
    # Create enhanced message with context
    context_str = f"Context: {runtime_context}\n\n"
    user_query = "What features do I have access to?"
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=context_str + user_query)]
    )
    
    # Agent responds with awareness of runtime context
    return message
```

---

## Google Cloud Integration

### Vertex AI Integration

#### Using Vertex AI Models

```python
from google.adk import Agent
from google.genai import types
import vertexai

# Initialise Vertex AI
vertexai.init(project="my-gcp-project", location="us-central1")

# Use Vertex AI's Gemini models
vertex_agent = Agent(
    name="vertex_agent",
    model="gemini-2.5-flash",  # Available through Vertex AI
    description="Agent using Vertex AI models",
    instruction="Process user requests with Vertex AI",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=2048,
    ),
)
```

#### Vertex AI Monitoring Integration


```python
from google.cloud import monitoring_v3

class VertexAIMonitoring:
    """Monitor agent performance in Vertex AI."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = monitoring_v3.MetricServiceClient()
    
    async def log_agent_metric(
        self,
        agent_name: str,
        metric_name: str,
        value: float
    ) -> None:
        """Log custom metric for agent."""
        project_name = f"projects/{self.project_id}"
        
        time_series = monitoring_v3.TimeSeries()
        time_series.metric.type = f"custom.googleapis.com/{metric_name}"
        
        now = self.get_now_proto3()
        point = monitoring_v3.Point(
            {"interval": {"end_time": now}, "value": {"double_value": value}}
        )
        time_series.points = [point]
        
        self.client.create_time_series(name=project_name, time_series=[time_series])
```


### Cloud Run Deployment

#### Containerised Agent Deployment

```dockerfile
# Dockerfile for ADK agent
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
```

#### Cloud Run Deployment Script

```bash
#!/bin/bash
set -e

# Configuration
PROJECT_ID="my-gcp-project"
SERVICE_NAME="adk-agent-service"
REGION="us-central1"

# Build and push image
gcloud builds submit \
  --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --project ${PROJECT_ID}

# Deploy to Cloud Run
gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --project ${PROJECT_ID} \
  --set-env-vars "GOOGLE_PROJECT_ID=${PROJECT_ID}"
```

### Cloud Functions for Agents

#### Serverless Agent

```python
# main.py - Cloud Function
from google.adk import Agent, Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types
import json

agent = Agent(
    name="serverless_agent",
    model="gemini-2.5-flash",
    instruction="Process requests in serverless environment"
)

session_service = DatabaseSessionService(db_url="postgresql+asyncpg://user:pass@/adk?host=/cloudsql/project:region:instance")

async def agent_request(request):
    """Handle Cloud Function request."""
    try:
        request_json = request.get_json()
        user_id = request_json.get('user_id')
        message_text = request_json.get('message')
        
        runner = Runner(
            app_name="serverless_app",
            agent=agent,
            session_service=session_service
        )
        
        message = types.Content(
            role='user',
            parts=[types.Part(text=message_text)]
        )
        
        response_text = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=f"session_{user_id}",
            new_message=message
        ):
            if event.content and event.content.parts:
                response_text += event.content.parts[0].text
        
        return {"response": response_text}, 200
    
    except Exception as e:
        return {"error": str(e)}, 500

# Deploy with: gcloud functions deploy agent_request --runtime python311 --trigger-http
```

### Firestore for State

#### Storing Agent State

```python
from google.cloud import firestore

class AgentStateManager:
    """Manage agent state in Firestore."""
    
    def __init__(self, project_id: str):
        self.db = firestore.Client(project=project_id)
        self.collection = "agent_states"
    
    async def save_agent_state(
        self,
        agent_id: str,
        state: dict
    ) -> None:
        """Save agent state."""
        self.db.collection(self.collection).document(agent_id).set({
            "state": state,
            "timestamp": firestore.SERVER_TIMESTAMP
        }, merge=True)
    
    async def load_agent_state(self, agent_id: str) -> dict:
        """Load agent state."""
        doc = self.db.collection(self.collection).document(agent_id).get()
        return doc.to_dict()["state"] if doc.exists else {}

# Usage
state_manager = AgentStateManager(project_id="my-gcp-project")

# Save state
await state_manager.save_agent_state(
    "agent_123",
    {"current_task": "processing", "progress": 75}
)

# Load state
state = await state_manager.load_agent_state("agent_123")
```

### Cloud Storage for Artifacts

#### Store and Retrieve Artifacts

```python
from google.cloud import storage
from google.adk.artifacts import ArtifactService

class CloudStorageArtifactService(ArtifactService):
    """Store artifacts in Cloud Storage."""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
    
    async def save_artifact(
        self,
        artifact_id: str,
        content: bytes,
        content_type: str
    ) -> str:
        """Save artifact to Cloud Storage."""
        blob = self.bucket.blob(artifact_id)
        blob.upload_from_string(content, content_type=content_type)
        return blob.public_url
    
    async def retrieve_artifact(self, artifact_id: str) -> bytes:
        """Retrieve artifact from Cloud Storage."""
        blob = self.bucket.blob(artifact_id)
        return blob.download_as_bytes()

# Usage
artifact_service = CloudStorageArtifactService(
    project_id="my-gcp-project",
    bucket_name="agent-artifacts"
)

# Save artifact
url = await artifact_service.save_artifact(
    "report_2025_01",
    b"Report content",
    "text/plain"
)

# Retrieve artifact
content = await artifact_service.retrieve_artifact("report_2025_01")
```

### BigQuery for Analytics

#### Logging Agent Interactions

```python
from google.cloud import bigquery
from google.adk.tools.bigquery_toolset import BigQueryToolset

class AgentAnalytics:
    """Track agent interactions in BigQuery."""
    
    def __init__(self, project_id: str, dataset_id: str):
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.table_id = f"{project_id}.{dataset_id}.agent_interactions"
    
    async def log_interaction(
        self,
        agent_id: str,
        user_id: str,
        query: str,
        response: str,
        execution_time: float
    ) -> None:
        """Log interaction to BigQuery."""
        rows_to_insert = [{
            "agent_id": agent_id,
            "user_id": user_id,
            "query": query,
            "response": response,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }]
        
        errors = self.client.insert_rows_json(self.table_id, rows_to_insert)
        if errors:
            print(f"Errors inserting rows: {errors}")

# Usage
analytics = AgentAnalytics(
    project_id="my-gcp-project",
    dataset_id="agent_analytics"
)

# Log interaction
await analytics.log_interaction(
    agent_id="agent_123",
    user_id="user_456",
    query="What is the weather?",
    response="The weather is sunny and warm.",
    execution_time=1.23
)
```

### Secret Manager for Credentials

#### Secure Credential Management


```python
from google.cloud import secretmanager

class SecureCredentialManager:
    """Manage credentials securely."""
    
    def __init__(self, project_id: str):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id
    
    def create_secret(self, secret_id: str, secret_value: str) -> str:
        """Create a secret."""
        parent = f"projects/{self.project_id}"
        
        secret = self.client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}},
            }
        )
        
        # Add version
        self.client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )
        
        return secret.name
    
    def access_secret(self, secret_id: str, version: str = "latest") -> str:
        """Access a secret."""
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")

# Usage
credential_manager = SecureCredentialManager(project_id="my-gcp-project")

# Store API key securely
credential_manager.create_secret(
    "gemini_api_key",
    "your-api-key"
)

# Retrieve when needed
api_key = credential_manager.access_secret("gemini_api_key")
```


---

## Gemini-Specific Features

### Multimodal Inputs

#### Processing Images

```python
from google.adk import Agent
from google.genai import types
import base64

vision_agent = Agent(
    name="vision_analyst",
    model="gemini-2.5-vision",
    description="Analyzes images and multimedia",
    instruction="Analyse images and describe what you see"
)

async def analyse_image(image_path: str):
    """Analyse image using ADK agent."""
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    
    runner = Runner(
        app_name="vision_app",
        agent=vision_agent,
        session_service=InMemorySessionService()
    )
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    # Create message with image
    message = types.Content(
        role='user',
        parts=[
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png",
                    data=image_data
                )
            ),
            types.Part(text="What's in this image?")
        ]
    )
    
    async for event in runner.run_async(
        user_id="user123",
        session_id="session1",
        new_message=message
    ):
        if event.content:
            print(f"Analysis: {event.content.parts[0].text}")
```

#### Processing Video

```python
from google.genai import types

video_agent = Agent(
    name="video_analyst",
    model="gemini-2.5-vision",
    instruction="Analyse videos and extract key information"
)

async def analyse_video(video_uri: str):
    """Analyse video using ADK."""
    message = types.Content(
        role='user',
        parts=[
            types.Part(
                video_metadata=types.VideoMetadata(
                    start_offset=types.Duration(seconds=0),
                    end_offset=types.Duration(seconds=60)
                )
            ),
            types.Part(text="Summarise the key events in this video")
        ]
    )
    
    # Process video through agent
    # Note: Video processing requires proper file upload setup
```

### Grounding with Google Search

#### Grounded Generation

```python
from google.adk import Agent
from google.adk.tools import google_search

grounded_agent = Agent(
    name="grounded_responder",
    model="gemini-2.5-flash",
    description="Provides grounded responses using web search",
    instruction="""When answering questions about current events, recent information,
    or facts that might change, use google_search to ground your response in
    current information. Always cite sources.""",
    tools=[google_search]
)

# The agent grounds responses using real-time web search results
# This ensures information is current and verifiable
```

### Code Execution

#### Executing Code Within Agents

```python
from google.adk import Agent
from google.adk.code_executors import BuiltInCodeExecutor

code_agent = Agent(
    name="code_executor",
    model="gemini-2.5-flash",
    description="Executes code to solve problems",
    instruction="When needed, write and execute Python code to solve problems",
    code_executor=BuiltInCodeExecutor(),
    max_iterations=10
)

# Agent can now:
# - Write Python code
# - Execute it safely
# - See results
# - Iterate based on results
```

### Function Calling

#### Structured Function Invocation

```python
from google.adk import Agent
from typing import List

def get_weather(city: str, date: str) -> dict:
    """Get weather for a city on a specific date."""
    return {
        "city": city,
        "date": date,
        "temperature": 72,
        "condition": "Sunny"
    }

def book_flight(origin: str, destination: str, date: str) -> dict:
    """Book a flight."""
    return {
        "status": "booked",
        "flight_id": "ABC123",
        "confirmation": "Email sent"
    }

function_agent = Agent(
    name="travel_assistant",
    model="gemini-2.5-pro",
    instruction="Help plan trips by checking weather and booking flights",
    tools=[get_weather, book_flight]
)

# Agent uses function calling to structure its reasoning and actions
```

### Context Caching

#### Caching Strategies for Gemini

```python
from google.adk import Agent
from google.adk.agents import ContextCacheConfig

# Expensive cached context
LARGE_KNOWLEDGE_BASE = """
[Extensive domain knowledge - could be 100KB of text]
""" * 1000

cached_agent = Agent(
    name="knowledge_agent",
    model="gemini-2.5-pro",
    static_instruction=LARGE_KNOWLEDGE_BASE,
    instruction="Answer questions about the knowledge base",
    cache_config=ContextCacheConfig(
        enable_auto_caching=True,
        ttl_seconds=86400
    )
)

# Benefits:
# - First request: ~150ms (includes cache creation)
# - Subsequent requests: ~50ms (uses cache)
# - Cost savings: 90% reduction in token costs for cached content
```

### Safety Settings

#### Configuring Gemini Safety

```python
from google.adk import Agent
from google.genai import types

safe_agent = Agent(
    name="safe_assistant",
    model="gemini-2.5-flash",
    instruction="Respond helpfully while maintaining safety",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.9,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ],
    ),
)
```

### Model Selection

#### Choosing Between Models

```python
from google.adk import Agent

# Fast responses, lower cost
flash_agent = Agent(
    name="quick_agent",
    model="gemini-2.5-flash",
    instruction="Provide quick responses",
    # Typical: 50-200ms latency, $0.075/1M input tokens
)

# Better quality, slightly slower
pro_agent = Agent(
    name="quality_agent",
    model="gemini-2.5-pro",
    instruction="Provide high-quality, detailed responses",
    # Typical: 100-500ms latency, $3/1M input tokens
)

# Best quality, slowest
ultra_agent = Agent(
    name="expert_agent",
    model="gemini-2.5-ultra",
    instruction="Provide expert-level responses",
    # Typical: 500ms-2s latency, $15/1M input tokens
)
```

---

## Vertex AI

### Model Garden Integration

[Content continues with extensive coverage of Vertex AI integration, custom deployments, vector search, feature stores, and monitoring...]

---

## Advanced Topics

### Custom Agent Types

#### Creating Agent Variants

```python
from google.adk import Agent
from abc import ABC, abstractmethod

class SpecialisedAgent(Agent, ABC):
    """Base class for specialised agents."""
    
    @abstractmethod
    def get_specialty(self) -> str:
        """Return agent specialty."""
        pass

class DataAnalystAgent(SpecialisedAgent):
    """Agent specialising in data analysis."""
    
    def __init__(self, name: str, project_id: str):
        super().__init__(
            name=name,
            model="gemini-2.5-pro",
            instruction="Analyse data and provide insights",
            description="Data analysis specialist"
        )
        self.project_id = project_id
    
    def get_specialty(self) -> str:
        return "Data Analysis"

class CodeReviewAgent(SpecialisedAgent):
    """Agent specialising in code review."""
    
    def __init__(self, name: str):
        super().__init__(
            name=name,
            model="gemini-2.5-pro",
            instruction="Review code for quality and best practices",
            description="Code review specialist"
        )
    
    def get_specialty(self) -> str:
        return "Code Review"
```

---

## Graph-Based Agent Workflows (available since v1.25.0; use `Workflow` in 2.x)

`GraphAgent` enables stateful, graph-based multi-agent orchestration with visual tooling:

```python
from google.adk import Agent
from google.adk.graph import GraphAgent, GraphEdge

# Define individual specialist agents
research_agent = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="Research the given topic thoroughly.",
)

analysis_agent = Agent(
    name="analyser",
    model="gemini-2.0-flash",
    instruction="Analyse the research and extract key insights.",
)

writer_agent = Agent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="Write a clear report based on the analysis.",
)

# Build the graph
graph = GraphAgent(
    name="report_pipeline",
    agents=[research_agent, analysis_agent, writer_agent],
    edges=[
        GraphEdge(from_agent="researcher", to_agent="analyser"),
        GraphEdge(from_agent="analyser", to_agent="writer"),
    ],
)

# Run the graph
result = await graph.run("Write a report on quantum computing trends in 2026")
print(result.final_output)
```

---

## Task API (v1.28.0+)

The Task API provides structured task management for complex, multi-step agent workflows:

```python
from google.adk import Agent
from google.adk.tasks import TaskManager, Task, TaskStatus

agent = Agent(name="task_agent", model="gemini-2.0-flash")
task_manager = TaskManager(agent)

# Create and track tasks
task = await task_manager.create_task(
    title="Market Analysis Report",
    description="Analyse Q1 2026 market data",
    priority="high",
)

# Run the task
await task_manager.run_task(task.id)

# Check status
status = await task_manager.get_status(task.id)
print(f"Status: {status}")  # TaskStatus.COMPLETED
```

---

## Session Rewind (v1.29.0+)

Rewind a session to replay from a prior state — useful for debugging and exploring alternative paths:

```python
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

# ... run agent session ...

# Rewind to before the last 2 turns
rewound_session = await session_service.rewind(
    session_id="session_123",
    steps_back=2,
)

# Continue from the rewound point
result = await agent.run(
    "Try a different approach",
    session=rewound_session,
)
```

---

## Class & API Reference

**Verified against `google-adk==2.7.1`** (live introspection — `model_fields`, `inspect.signature`, `__mro__`, package `__all__` lists — cross-checked against the installed package source, consolidating 48 sequential "class deep dive" volumes down to one deduplicated, source-verified reference covering 12 subject areas).

If you're coming from an older volume of this guide, the callouts below marked **Correction** or **Import path note** flag real breaking changes versus earlier `google-adk` releases — most notably:

- `EventsCompactionConfig` moved to the private `google.adk.apps._configs` module and is no longer re-exported from `google.adk.apps` (only `App` and `ResumabilityConfig` still are).
- Every A2A class (`RemoteA2aAgent`, `A2aAgentExecutor`, `AgentCardBuilder`, `to_a2a`, ...) lives behind a private submodule path — the entire `google.adk.a2a` package tree has empty `__init__.py` files.
- GCS and Firestore session/memory/artifact integrations relocated out of `google.adk.tools`/`google.adk.sessions`/`google.adk.memory` into a dedicated `google.adk.integrations.*` subpackage.
- `LoopAgent`, `ParallelAgent`, and `SequentialAgent` are `@deprecated` in favor of `Workflow` (a verified `DeprecationWarning` now fires on instantiation) — they still work today, but new code should prefer `Workflow` with `edges`/`@node`.
- `BaseToolset`, `SqliteSessionService`, `Trigger`, and `Graph` are no longer re-exported at their package root — import from their submodules directly.

### Agents & Context

#### `BaseAgent`
Every concrete agent (`LlmAgent`, `LoopAgent`, `ParallelAgent`, `SequentialAgent`, custom agents) inherits from this abstract Pydantic base. It wires together lifecycle methods (`run_async`, `run_live`), the before/after callback chain, sub-agent tree management, and optional per-agent typed state (`BaseAgentState`). It is itself a `BaseNode` subclass, so any agent can appear directly in a `Workflow` graph.

```python
class BaseAgent(BaseNode, abc.ABC):
    name: str                                     # must be a valid Python identifier, not "user"
    description: str = ''                         # one-liner used by parent-agent routing
    parent_agent: Optional[BaseAgent] = Field(default=None, init=False, exclude=True)
    sub_agents: list[BaseAgent] = Field(default_factory=list)
    before_agent_callback: Optional[BeforeAgentCallback] = None
    after_agent_callback: Optional[AfterAgentCallback] = None
```

`model_post_init` sets `parent_agent` on every sub-agent passed at construction time; a sub-agent that already has a different parent raises `ValueError`. Assigning to `sub_agents` *after* construction skips this wiring — always pass `sub_agents=[...]` in the constructor. `clone(update={...})` deep-copies an agent (and its sub-agent tree) for building variants without mutating the original; `find_agent(name)` / `find_sub_agent(name)` walk the tree by name.

```python
from google.adk.agents import LlmAgent

base = LlmAgent(name="analyst", model="gemini-2.5-flash", instruction="You are a data analyst.")
variant = base.clone(update={"name": "strict_analyst"})
print(variant.name, base.name)  # strict_analyst analyst (original untouched)
```

#### LlmAgent
Why it matters: the workhorse agent class. In 2.7.1 `LlmAgent` (aliased `Agent`) sits on a unified class hierarchy with the workflow graph system — `LlmAgent -> BaseAgent -> google.adk.workflow._base_node.BaseNode -> pydantic.BaseModel`. This is a real architectural change versus every one of the 25 source docs (all written against 2.1.0–2.3.0), which describe `LlmAgent`/`BaseAgent` as graph-agnostic. Because of this MRO, every `LlmAgent` instance now directly carries workflow-node fields (`rerun_on_resume`, `retry_config`, `timeout`, `wait_for_output`, `parallel_worker`, `state_schema`) even when the agent is never used inside a `Workflow`.

```python
from google.adk.agents import LlmAgent  # also exported as Agent

agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Answer questions using the search tool.",
    tools=[...],
    # workflow-node fields inherited via BaseNode, usable even standalone:
    retry_config=None,
    timeout=None,
)
```
```python
print(LlmAgent.__mro__)
# (LlmAgent, BaseAgent, google.adk.workflow._base_node.BaseNode,
#  pydantic.main.BaseModel, abc.ABC, object)
```

#### `LlmAgent` — configuration knobs
**Source:** `google.adk.agents.llm_agent` (verified 2.7.1)

Beyond the constructor fields already covered by batch A, four knobs are worth calling out on their own because they change request shape rather than behavior:

- **`static_instruction`** (`types.ContentUnion | None`) — content that never changes between requests, sent as `system_instruction` verbatim so Gemini's implicit/explicit context cache can reuse it. When set, `instruction` (still dynamic, `{var}`-templated) is appended to the *user* content instead of the system prompt. When `static_instruction` is `None` (default), `instruction` is the system prompt as usual.
- **`generate_content_config`** (`types.GenerateContentConfig | None`) — raw generation params (temperature, safety settings, `response_mime_type`, `thinking_config`). `planner.thinking_config` wins over `generate_content_config.thinking_config` when both are set.
- **`include_contents: Literal['default', 'none'] = 'default'`** — `'none'` means the model sees zero prior session history, operating only on `instruction` + the current turn. Used for deterministic, stateless workflow nodes and parallel-fan-out workers.
- **`LlmAgent.set_default_model(model)` / `set_default_live_model(model)`** — `@classmethod`s overriding the class-level fallback model (`str` or `BaseLlm`) used when an agent omits `model=`. Verified defaults: `DEFAULT_MODEL = 'gemini-3.5-flash'`, `DEFAULT_LIVE_MODEL = 'gemini-live-2.5-flash-native-audio'`.

```python
from google.adk.agents import LlmAgent

LlmAgent.set_default_model("gemini-2.5-pro")           # app-wide default
LlmAgent.set_default_live_model("gemini-live-2.5-flash-native-audio")

legal_agent = LlmAgent(
    name="legal_drafter",
    static_instruction="You are a legal drafting assistant. Never give specific legal advice.",
    instruction="Current matter type: {matter_type}. Jurisdiction: {jurisdiction}.",
    include_contents="none",
)
```

#### `ManagedAgent`
**Source:** `google.adk.agents._managed_agent` (re-exported from `google.adk.agents`)

Wraps Google's **Managed Agents API** (`interactions.create`) so a server-hosted agent (identified by `agent_id`) runs without local inference. Verified 2.7.1 constructor adds `api_client: Optional[Client]` for pre-built clients and restricts `tools` to `list[Union[types.Tool, BaseTool, RemoteMcpServer]]` (a `Callable`/`FunctionTool` in this list raises at runtime — client-executed tools aren't supported). Interactions always stream (`background=True`); non-streaming polling isn't implemented. The API is served only from the `global` location — an enterprise client pinned elsewhere is rejected at construction.

```python
from google.adk.agents import ManagedAgent
from google.genai import types

agent = ManagedAgent(
    name="managed_search",
    agent_id="antigravity-preview-05-2026",
    tools=[types.Tool(google_search=types.GoogleSearch())],  # server-side tool only
)
```

Set `mode="single_turn"` to place a `ManagedAgent` inside a parent `LlmAgent.sub_agents` list — ADK wraps it as a `_SingleTurnAgentTool` automatically, same as any other single-turn sub-agent.

#### `LangGraphAgent`
**Source:** `google.adk.agents.langgraph_agent`

Wraps a compiled LangGraph `CompiledStateGraph` as a native `BaseAgent`, bridging ADK's `Event`-based history with LangGraph's `messages` state. Marked a "concept implementation" in the source docstring — expect API changes.

```python
class LangGraphAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    graph: CompiledStateGraph
    instruction: str = ''      # injected as SystemMessage on the first turn only
```

`_run_async_impl` passes `configurable.thread_id = ctx.session.id` on every `graph.invoke()` call. Memory strategy depends on whether the graph was compiled with a checkpointer:

| `graph.checkpointer` | Message strategy |
|---|---|
| `None` | `_get_conversation_with_agent` — replays the *entire* session (user + this agent's turns) from `session.events` |
| Set (e.g. `MemorySaver`) | `_get_last_human_messages` — only the latest block of user messages; LangGraph's own checkpointer owns history |

```python
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents.langgraph_agent import LangGraphAgent

def call_model(state: MessagesState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.set_entry_point("model")
builder.add_edge("model", END)

agent = LangGraphAgent(
    name="lg_agent",
    graph=builder.compile(checkpointer=MemorySaver()),   # multi-turn memory
    instruction="You are a helpful assistant that remembers context.",
)
```

#### `AntigravityAgent` + `convert_step_to_events`
**Source:** `google.adk.labs.antigravity._antigravity_agent`, `._event_converter` — `@experimental`

Wraps a pre-configured `google.antigravity.Agent` as an ADK `BaseAgent`, delegating each turn to the Antigravity SDK runner and converting its trajectory steps into ADK events. Because the Antigravity harness owns session lifecycle, the agent must be a **standalone root** — both `model_post_init` (rejects a non-empty `sub_agents`) and `__setattr__` (rejects being assigned a `parent_agent`) enforce this.

```python
def convert_step_to_events(
    step: sdk_types.Step, *, ctx: InvocationContext, author: str,
    seen_tool_calls: set[str], seen_tool_results: set[str], streaming: bool = False,
) -> list[Event]: ...
```

Mapping rules: SSE partial deltas → `thinking_delta` becomes a `Part(thought=True)`, `content_delta` becomes text; final model text is emitted only when `step.is_complete_response`; tool calls are deduplicated on `call.id or f'{step_index}-{call.name}'`; tool-response events are authored as the **tool name**, not the agent name.

Trajectories persist across turns via `config.save_dir` — the harness writes a random-ID trajectory on turn 1, which the agent renames to `{session_id}_{agent_name}` and replays via `config.conversation_id` on later turns. `_trajectory_files.save_resume_step_index` prevents re-emitting already-yielded steps.

```python
from google.adk.labs.antigravity import AntigravityAgent
from google.antigravity import LocalAgentConfig

config = LocalAgentConfig(model="gemini-2.0-flash", save_dir="~/.adk_antigravity_trajs")
agent = AntigravityAgent(name="my_ag_agent", config=config)   # root-only; no sub_agents
```

#### `Context` — the unified write API (`ToolContext` = `CallbackContext`)
**Source:** `google.adk.agents.context`

In 2.x, `ToolContext` and `CallbackContext` are both plain aliases for `Context` (`tools/tool_context.py: ToolContext = Context`; `agents/callback_context.py: CallbackContext = Context`). `Context` extends `ReadonlyContext` with every write capability available inside tools, callbacks, and `@node` functions.

| Property / method | Purpose |
|---|---|
| `ctx.state[key] = val` | Mutate session state (`app:`, `user:`, `temp:` prefixes apply) |
| `ctx.route = "key"` | Set the conditional edge for the current workflow step |
| `ctx.output = val` | Set the node's explicit output (raises if the function also returns a value) |
| `await ctx.save_artifact/load_artifact/list_artifacts/get_artifact_version` | Artifact CRUD |
| `await ctx.search_memory(query)` | Search long-term memory |
| `await ctx.add_events_to_memory(events=...)` / `add_session_to_memory()` / `add_memory(memories=...)` | Ingest into long-term memory |
| `ctx.request_confirmation(hint=None)` | Trigger HITL confirmation from inside a tool |
| `await ctx.run_node(node, node_input, run_id=...)` | Dynamically invoke another workflow node |
| `yield RequestInput(interrupt_id=..., message=...)` | Pause a workflow node for user input |

```python
from google.adk.tools.tool_context import ToolContext

def update_preferences(preferred_language: str, tool_context: ToolContext) -> dict:
    tool_context.state["session_theme"] = "dark"                 # session-scoped
    tool_context.state["user:preferred_language"] = preferred_language  # persists across sessions
    tool_context.state["app:last_pref_update"] = "2026-07-02"     # visible to all users/sessions
    return {"updated": True}
```

#### `ReadonlyContext`
**Source:** `google.adk.agents.readonly_context` (verified 2.7.1)

Base class for `Context`; passed to `InstructionProvider` callables, `before_agent_callback`, and `BaseToolset.get_tools()`. Enforces read-only access at the API level: `state` is a `MappingProxyType`, so a mutation attempt raises `TypeError` instead of silently corrupting session state.

```python
class ReadonlyContext:
    @property
    def user_content(self) -> types.Content | None: ...
    @property
    def invocation_id(self) -> str: ...
    @property
    def agent_name(self) -> str: ...                       # "unknown" when agent is None
    @property
    def state(self) -> MappingProxyType[str, Any]: ...      # READ-ONLY
    @property
    def session(self) -> Session: ...
    @property
    def user_id(self) -> str: ...
    @property
    def run_config(self) -> RunConfig | None: ...
    @property
    def custom_metadata(self) -> Mapping[str, Any]: ...
    def get_credential(self, key: str) -> AuthCredential | None: ...
```

All heavier type imports (`InvocationContext`, `Session`, `RunConfig`, `AuthCredential`) live inside `if TYPE_CHECKING:` blocks, so importing `ReadonlyContext` carries no extra runtime cost.

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

class RoleBasedToolset(BaseToolset):
    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None):
        is_admin = readonly_context is not None and readonly_context.state.get("role") == "admin"
        return self._all_tools + (self._admin_tools if is_admin else [])
```

#### InvocationContext
Why it matters: the per-`run_async` call context threading session, run config, and branch info through the whole agent/tool/callback call graph. Unchanged from the docs.

```python
from google.adk.agents.invocation_context import InvocationContext
```

### Tools & Toolsets

#### `FunctionTool`
**Source:** `google.adk.tools.function_tool` (verified 2.7.1)

Wraps any callable (sync/async, function or callable object) as an ADK tool.

```python
class FunctionTool(BaseTool):
    def __init__(self, func: Callable[..., Any], *,
                 require_confirmation: Union[bool, Callable[..., bool]] = False): ...
```

- **Pydantic auto-coercion** — a `dict` argument from the LLM is converted to a `BaseModel` instance *before* the function runs, when the parameter is annotated as a `BaseModel` subclass (also `Optional[BaseModel]`, `list[BaseModel]`).
- **Mandatory-args guard** — params without defaults (excluding `*args`/`**kwargs`) are required; a missing one short-circuits to `{"error": "...missing parameters..."}` without calling the function.
- **`require_confirmation`** — `bool` or a callable invoked with the tool's preprocessed keyword arguments, returning `bool`. When truthy: calls `tool_context.request_confirmation(hint=...)`, sets `skip_summarization=True`, and returns an error dict pending the next turn's `ToolConfirmation` response.
- **Context injection** — a parameter literally named `tool_context`, or typed `ToolContext`/`Context`, is injected automatically and stripped from the LLM-visible declaration.

```python
from pydantic import BaseModel
from google.adk.tools import FunctionTool

class Order(BaseModel):
    item_id: str
    quantity: int

def place_order(order: Order) -> dict:
    """Place a product order."""
    return {"order_id": f"ORD-{order.item_id}", "status": "confirmed"}

order_tool = FunctionTool(func=place_order)

# require_confirmation as a predicate — only gates destructive calls
def needs_confirm(table: str, where_clause: str) -> bool:
    return where_clause.strip() in ("1=1", "true")

delete_tool = FunctionTool(func=lambda table, where_clause: {"rows": 0}, require_confirmation=needs_confirm)
```

#### `LongRunningFunctionTool`
**Source:** `google.adk.tools.long_running_tool`

A thin `FunctionTool` subclass for async operations that return before the work finishes.

```python
class LongRunningFunctionTool(FunctionTool):
    def __init__(self, func: Callable):
        super().__init__(func)
        self.is_long_running = True
```

`_get_declaration()` appends a guardrail instruction to the tool's description: *"This is a long-running operation. Do not call this tool again if it has already returned some intermediate or pending status."* The function itself must return immediately with a `{"status": "pending", ...}` dict; real work is offloaded via `asyncio.create_task`, and a **second, regular** `FunctionTool` acts as the companion poll tool that reads progress the background task writes to shared state.

```python
import asyncio, time
from google.adk.tools import LongRunningFunctionTool, FunctionTool
from google.adk.tools.tool_context import ToolContext

_jobs: dict[str, dict] = {}

async def _run_export(job_id: str) -> None:
    await asyncio.sleep(5)
    _jobs[job_id] = {"status": "done"}

async def start_export(dataset: str, tool_context: ToolContext) -> dict:
    """Start an async data export job."""
    job_id = f"exp-{dataset}-{int(time.time())}"
    _jobs[job_id] = {"status": "pending"}
    tool_context.state[f"export_job:{dataset}"] = job_id
    asyncio.create_task(_run_export(job_id))          # fire-and-forget — do NOT await
    return {"status": "pending", "job_id": job_id}

async def check_export(dataset: str, tool_context: ToolContext) -> dict:
    """Check the status of a dataset export job."""
    return _jobs.get(tool_context.state.get(f"export_job:{dataset}"), {"error": "not found"})

export_tool = LongRunningFunctionTool(func=start_export)
poll_tool = FunctionTool(func=check_export)
```

#### `ToolConfirmation` — human-in-the-loop tool gate
**Source:** `google.adk.tools.tool_confirmation` (verified 2.7.1 — signature and fields unchanged since 2.3.0)

```python
@experimental(FeatureName.TOOL_CONFIRMATION)
class ToolConfirmation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=alias_generators.to_camel,   # camelCase on the wire
        populate_by_name=True,
    )
    hint: str = ""
    confirmed: bool = False
    payload: Optional[Any] = None

    @classmethod
    def from_response_dict(cls, response: dict[str, Any]) -> "ToolConfirmation":
        """Handles both a direct dict and the ADK client's {'response': json_string} wrapper."""
        if response and len(response) == 1 and "response" in response:
            return cls.model_validate(json.loads(response["response"]))
        return cls.model_validate(response)
```

The interrupt mechanism lives in `FunctionTool`/`ToolContext`, not in the tool function's return value — returning a `ToolConfirmation` from a plain function has no special effect. Two patterns trigger it: `FunctionTool(func, require_confirmation=True)` (recommended — see above), or calling `tool_context.request_confirmation(hint=..., payload=...)` directly inside a function that accepts `tool_context`, then branching on `tool_context.tool_confirmation` on the resumed call.

```python
from google.adk.tools.tool_context import ToolContext

async def delete_record(record_id: str, tool_context: ToolContext) -> dict:
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(hint=f"Delete record '{record_id}'?")
        return {"pending": "Awaiting confirmation."}
    if not tool_context.tool_confirmation.confirmed:
        return {"status": "cancelled"}
    return {"status": "deleted", "id": record_id}
```

#### BaseTool
Why it matters: the abstract root every tool ultimately implements. **Correction**: `BaseTool` in 2.7.1 carries a `response_scheduling` field not documented in any of the 25 volumes, controlling whether a tool's result is returned immediately or scheduled asynchronously into the event stream — relevant when building custom long-running tools without using `LongRunningFunctionTool`.

```python
from google.adk.tools import BaseTool
```

#### `AgentTool`
**Source:** `google.adk.tools.agent_tool` (verified 2.7.1)

Wraps any `BaseAgent`/`LlmAgent` as a callable tool for another agent — the standard multi-agent composition primitive. Import path and basic constructor are unchanged from earlier docs.

```python
class AgentTool(BaseTool):
    def __init__(
        self, agent: BaseAgent, skip_summarization: bool = False, *,
        include_plugins: bool = True, propagate_grounding_metadata: bool = False,
    ): ...
```

- `skip_summarization` — return the sub-agent's raw output without an LLM summarization pass (useful with `output_schema`).
- `include_plugins` (default `True`) — propagates the parent runner's plugins into the sub-agent's inner runner; set `False` to give a sensitive sub-agent (e.g. one that processes PII) an isolated plugin/logging environment.
- `propagate_grounding_metadata` (default `False`) — forwards grounding/citation metadata from a search-grounded sub-agent up to the parent's event stream.

```python
from google.adk.tools import AgentTool

tool = AgentTool(agent=some_sub_agent)

coordinator_tools = [
    AgentTool(sensitive_data_agent, include_plugins=False),
    AgentTool(search_specialist, propagate_grounding_metadata=True),
]
```

#### BaseToolset
Why it matters: the abstract base for every toolset (dynamic collections of tools, e.g. from an MCP server or an OpenAPI spec). **Correction**: `BaseToolset` is not re-exported from `google.adk.tools` in 2.7.1 (`from google.adk.tools import BaseToolset` now raises `ImportError`) — it must be imported from its submodule. Only one method, `get_tools`, is actually abstract; prefix injection is handled by the `@final` `get_tools_with_prefix()`.

```python
from google.adk.tools.base_toolset import BaseToolset
# constructor: BaseToolset(*, tool_filter=None, tool_name_prefix=None)
```

#### McpToolset (also aliased MCPToolset)
Why it matters: exposes tools from any MCP server (stdio, SSE, or streamable-HTTP transport) as ADK tools. Both spellings (`McpToolset`, `MCPToolset`) are exported from `google.adk.tools.__all__` as aliases of the same class. Requires the optional `mcp` package (`pip install google-adk[mcp]` / `[extensions]`) — not importable in an environment lacking it, which is expected and not a docs bug.

```python
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams

toolset = McpToolset(
    connection_params=StdioConnectionParams(command="npx", args=["-y", "some-mcp-server"]),
)
```

#### GCSToolset / GCSAdminToolset
Why it matters: **Correction** — every source doc that mentions these places them under `google.adk.tools`; in 2.7.1 they have moved to `google.adk.integrations.gcs.storage_toolset.GCSToolset` and `google.adk.integrations.gcs.admin_toolset.GCSAdminToolset` respectively, part of a broader trend of cloud-specific toolsets relocating into an `integrations` subpackage.

```python
from google.adk.integrations.gcs.storage_toolset import GCSToolset
from google.adk.integrations.gcs.admin_toolset import GCSAdminToolset
```

#### SpannerToolset / SpannerAdminToolset / BigQueryToolset / BigtableToolset / PubSubToolset
Why it matters: these remain at their documented `google.adk.tools.<service>` paths in 2.7.1 (confirmed via source grep), but each requires its corresponding Google Cloud client library extra (`google-cloud-spanner`, `google-cloud-bigquery`, `google-cloud-bigtable`, `google-api-core`) to actually import — none are installed in the bare verification venv, so import failures there reflect missing optional dependencies, not code drift.

```python
from google.adk.tools.spanner import SpannerToolset, SpannerAdminToolset
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigtable import BigtableToolset
from google.adk.tools.pubsub import PubSubToolset
```

#### `PubSubToolset` + PubSub client factory + credentials
**Source:** `google.adk.tools.pubsub.pubsub_toolset`, `.message_tool`, `.config`, `.client`, `.pubsub_credentials`

`@experimental(FeatureName.PUBSUB_TOOLSET)` `BaseToolset` exposing three Pub/Sub operations as `GoogleTool`-wrapped functions.

```python
class PubSubToolset(BaseToolset):
    def __init__(self, *, tool_filter: ToolPredicate | list[str] | None = None,
                 credentials_config: PubSubCredentialsConfig | None = None,
                 pubsub_tool_config: PubSubToolConfig | None = None): ...
```

| Tool | Key behavior |
|---|---|
| `publish_message(topic_name, message, attributes=None, ordering_key="")` | `enable_message_ordering` auto-set when `ordering_key` is non-empty; returns `{"message_id": ...}` |
| `pull_messages(subscription_name, max_messages=1, auto_ack=False)` | binary payload falls back to base64 if UTF-8 decode fails |
| `acknowledge_messages(subscription_name, ack_ids)` | returns `{"status": "SUCCESS"}` or an error dict |

`PubSubToolConfig(project_id: str | None = None)` pins the GCP project (`extra="forbid"`); `PubSubCredentialsConfig` extends `BaseGoogleCredentialsConfig`, defaulting `scopes` to `("https://www.googleapis.com/auth/pubsub",)` and using token-cache key `"pubsub_token_cache"`.

Underneath, `get_publisher_client()` / `get_subscriber_client()` (module `tools.pubsub.client`) cache one client per `(id(credentials), user_agent, ...)` key with a 30-minute TTL and a `threading.Lock`, using `BatchSettings(max_messages=1)` to effectively disable publisher-side batching. Call `cleanup_clients()` (or `PubSubToolset.close()`, which delegates to it) on shutdown to close cached gRPC transports.

```python
from google.adk.tools.pubsub import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
    tool_filter=["publish_message"],   # expose only publish
)
```

#### `SpannerToolset` + `SpannerAdminToolset` + settings
**Source:** `google.adk.tools.spanner.spanner_toolset`, `.admin_toolset`, `.settings`, `.spanner_credentials`

Both `@experimental` (`FeatureName.SPANNER_TOOLSET` / `SPANNER_ADMIN_TOOLSET`), wrapping Spanner operations as `GoogleTool` instances sharing a `SpannerCredentialsConfig` + `SpannerToolSettings`.

```python
class SpannerToolset(BaseToolset):
    def __init__(self, *, tool_filter=None,
                 credentials_config: SpannerCredentialsConfig | None = None,
                 spanner_tool_settings: SpannerToolSettings | None = None): ...
```

5 metadata tools (`spanner_list_table_names/indexes/index_columns/named_schemas`, `spanner_get_table_schema`) are always present; `spanner_execute_sql` and `spanner_similarity_search` are added only when `Capabilities.DATA_READ in settings.capabilities` (the default); `spanner_vector_store_similarity_search` requires `vector_store_settings` to be set. `SpannerAdminToolset` exposes 7 separate instance/database management tools.

```python
class SpannerToolSettings(BaseModel):
    capabilities: list[Capabilities] = [Capabilities.DATA_READ]
    max_executed_query_result_rows: int = 50
    query_result_mode: QueryResultMode = QueryResultMode.DEFAULT   # or DICT_LIST for {col: val} rows
    database_role: str | None = None
    vector_store_settings: SpannerVectorStoreSettings | None = None

class SpannerVectorStoreSettings(BaseModel):
    project_id: str; instance_id: str; database_id: str; table_name: str
    content_column: str; embedding_column: str; vector_length: int
    vertex_ai_embedding_model_name: str
    nearest_neighbors_algorithm: Literal["EXACT_NEAREST_NEIGHBORS", "APPROXIMATE_NEAREST_NEIGHBORS"] = "EXACT_NEAREST_NEIGHBORS"
    top_k: int = 4
    distance_type: str = "COSINE"          # COSINE | DOT_PRODUCT | EUCLIDEAN
    vector_search_index_settings: "VectorSearchIndexSettings | None" = None  # ANN only

class VectorSearchIndexSettings(BaseModel):
    index_name: str
    tree_depth: int = 2      # 2 or 3 (3 for >100M rows)
    num_leaves: int = 1000   # recommended: num_rows / 1000
    num_branches: int | None = None   # only for tree_depth=3
```

A `@model_validator` on `SpannerVectorStoreSettings` enforces `vector_length > 0` and that any `primary_key_columns` appear among the declared columns.

```python
from google.adk.tools.spanner import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings, QueryResultMode

toolset = SpannerToolset(
    spanner_tool_settings=SpannerToolSettings(
        max_executed_query_result_rows=100, query_result_mode=QueryResultMode.DICT_LIST,
    ),
)
```

#### ApplicationIntegrationToolset / ToolboxToolset / APIHubToolset
Why it matters: no cloud-extra dependency required; all three import cleanly in the bare venv and match their documented constructors.

```python
from google.adk.tools.application_integration_tool import ApplicationIntegrationToolset
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.tools.apihub_tool import APIHubToolset
```

#### `ToolboxToolset`
**Source:** `google.adk.tools.toolbox_toolset` (verified 2.7.1 signature)

Delegates to `toolbox_adk.ToolboxToolset` (`pip install "google-adk[toolbox]"`) to bridge an ADK agent to an [MCP Toolbox for Databases](https://github.com/googleapis/mcp-toolbox-sdk-python) server.

```python
class ToolboxToolset(BaseToolset):
    def __init__(self, server_url: str, toolset_name: str | None = None,
                 tool_names: list[str] | None = None,
                 auth_token_getters: Mapping[str, Callable[[], str]] | None = None,
                 bound_params: Mapping[str, Union[Callable[[], Any], Any]] | None = None,
                 credentials: CredentialConfig | None = None,
                 additional_headers: Mapping[str, str] | None = None, **kwargs): ...
```

`toolset_name` and `tool_names` combine as a **union**, not an intersection; omit both to load every tool on the server. `bound_params` binds a param (static value or zero-arg callable, called fresh per invocation) so it's never exposed to or settable by the LLM. `get_tools()`/`close()` are fully delegated to the underlying package; if it isn't installed, the constructor raises `ImportError` immediately (not at first use).

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset

toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    toolset_name="customer-tools",
    bound_params={"user_id": lambda: get_current_user_id(), "region": "us-east1"},
)
```

#### `OpenAPIToolset`
**Source:** `google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset` (verified 2.7.1)

Parses an OpenAPI 3.x spec (dict, JSON, or YAML string) and generates one `RestApiTool` per operation, eagerly, at construction time — `get_tools()` only applies `tool_filter`.

```python
class OpenAPIToolset(BaseToolset):
    def __init__(self, *, spec_dict: dict | None = None, spec_str: str | None = None,
                 spec_str_type: Literal["json", "yaml"] = "json",
                 auth_scheme: AuthScheme | None = None, auth_credential: AuthCredential | None = None,
                 credential_key: str | None = None,
                 tool_filter: ToolPredicate | list[str] | None = None,
                 tool_name_prefix: str | None = None,
                 ssl_verify: bool | str | ssl.SSLContext | None = None,
                 header_provider: Callable[[ReadonlyContext], dict[str, str]] | None = None,
                 httpx_client_factory: HttpxClientFactory | None = None,
                 preserve_property_names: bool = False): ...
```

- **`preserve_property_names`** controls only the **LLM-facing** argument name; the outgoing HTTP body always uses the spec's `original_name` regardless of this flag. Default `False` snake_cases property names for the LLM (`firstName` → `first_name` argument, still sent as `firstName` in the body).
- **`ssl_verify`** — `True`/`False`/a CA-bundle path/an `ssl.SSLContext`; `configure_ssl_verify_all(...)` can swap it after construction.
- **`header_provider`** — `(ReadonlyContext) -> dict[str, str]`, called per request for dynamic auth/correlation headers.
- **`httpx_client_factory`** — zero-arg callable returning a fresh `httpx.AsyncClient`; called on every request (not cached), enabling proxies, HTTP/2, custom transports.

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

toolset = OpenAPIToolset(
    spec_dict=my_spec_dict,
    ssl_verify="/etc/ssl/certs/corporate-ca.pem",
    tool_filter=["list_pets", "create_pet"],
    tool_name_prefix="petstore",
)
```

#### `OperationParser` + `ApiParameter` + `rename_python_keywords`
**Source:** `google.adk.tools.openapi_tool.openapi_spec_parser.operation_parser`, `.common.common`

`OperationParser` converts one OpenAPI `Operation` into `ApiParameter` instances used to generate the Python function signature, docstring, and JSON Schema for a `RestApiTool`.

```python
class OperationParser:
    def __init__(self, operation: Operation | dict | str, should_parse: bool = True, *,
                 preserve_property_names: bool = False): ...
    @classmethod
    def load(cls, operation, *, params: list[ApiParameter]) -> "OperationParser": ...  # inject pre-built params
```

`ApiParameter.model_post_init` derives `py_name` via `_to_snake_case(original_name)` then `rename_python_keywords()` (`"if"` → `"param_if"`); falls back to a location-derived default (`query_param`, `path_param`, ...) when both are empty. `_dedupe_param_names()` appends a numeric suffix to any collision.

```python
from google.adk.tools.openapi_tool.common.common import rename_python_keywords
assert rename_python_keywords("if") == "param_if"
assert rename_python_keywords("user_id") == "user_id"
```

#### `UrlContextTool`
**Source:** `google.adk.tools.url_context_tool` (verified 2.7.1)

A model-side built-in: it never produces a Python function call, instead injecting `types.Tool(url_context=types.UrlContext())` into the request so Gemini 2+ fetches URLs referenced in the conversation itself.

```python
class UrlContextTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name='url_context', description='url_context')

    async def process_llm_request(self, *, tool_context, llm_request) -> None:
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []
        if is_gemini_model(llm_request.model) or _is_managed_agent(llm_request):
            llm_request.config.tools.append(types.Tool(url_context=types.UrlContext()))
        else:
            raise ValueError(f'Url context tool is not supported for model {llm_request.model}')

url_context = UrlContextTool()   # module-level singleton — import and use directly
```

The `is_gemini_model` guard can be bypassed for testing via `GOOGLE_ADK_DISABLE_GEMINI_MODEL_ID_CHECK` (env var name confirmed in 2.7.1 source; the analogous `ADK_DISABLE_GEMINI_MODEL_ID_CHECK` name is used by the general model-family check in `model_name_utils`, a separate flag — see Runner & Execution Internals).

```python
from google.adk.agents import LlmAgent
from google.adk.tools.url_context_tool import url_context
from google.adk.tools import google_search

agent = LlmAgent(name="researcher", model="gemini-2.5-flash", tools=[google_search, url_context])
```

#### `EnterpriseWebSearchTool`
**Source:** `google.adk.tools.enterprise_search_tool`

Like `UrlContextTool`, this injects a Gemini built-in (`types.Tool(enterprise_web_search=types.EnterpriseWebSearch())`) rather than making HTTP calls itself. Exported as the module-level singleton `enterprise_web_search` (via `google.adk.tools`). Raises `ValueError` on non-Gemini models, and on Gemini 1.x combined with any other tool (Gemini 1.x is fine as the *sole* tool).

```python
from google.adk.agents import LlmAgent
from google.adk.tools import enterprise_web_search

agent = LlmAgent(name="search_agent", model="gemini-2.0-flash", tools=[enterprise_web_search])
```

#### Pre-built singleton tools (google_search, url_context, exit_loop, etc.)
Why it matters: a whole family of built-ins — `google_search`, `url_context`, `exit_loop`, `load_memory`, `preload_memory`, `load_artifacts`, `transfer_to_agent`, `enterprise_web_search`, `google_maps_grounding`, `get_user_choice`, `request_input` — are pre-instantiated module-level singletons, not classes you construct. Passing `google_search` directly into an agent's `tools=[...]` list is correct usage; there is no `GoogleSearchTool()` call needed.

```python
from google.adk.tools import google_search, url_context, exit_loop, transfer_to_agent
agent = LlmAgent(name="a", model="gemini-2.5-flash", tools=[google_search])
```

#### `VertexAiLoadProfilesTool`
**Source:** `google.adk.tools.vertex_ai_load_profiles_tool`

A `FunctionTool` subclass fetching a user's structured profiles from `VertexAiMemoryBankService.retrieve_profiles()` and surfacing them as a zero-argument tool call result (the `tool_context` parameter is stripped automatically since it's context-typed).

```python
class VertexAiLoadProfilesTool(FunctionTool):
    def __init__(self, memory_service: VertexAiMemoryBankService):
        super().__init__(self.load_profiles)
        self._memory_service = memory_service

    async def load_profiles(self, tool_context: ToolContext) -> dict[str, Any]:
        profiles = await self._memory_service.retrieve_profiles(
            app_name=tool_context.session.app_name, user_id=tool_context.user_id)
        return {"profiles": [p.profile for p in profiles if p.profile]}
```

#### DataAgentToolset / SkillToolset / EnvironmentToolset
Why it matters: all three remain present, at `google.adk.tools.data_agent.data_agent_toolset`, `google.adk.tools.skill_toolset`, and `google.adk.tools.environment._environment_toolset` respectively — the last one is intentionally private-underscored in its module name, so import it as a full path even though the class itself is public.

```python
from google.adk.tools.data_agent.data_agent_toolset import DataAgentToolset
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.environment._environment_toolset import EnvironmentToolset
```

#### `DataAgentToolset`
**Source:** `google.adk.tools.data_agent` — `@experimental(FeatureName.DATA_AGENT_TOOLSET)`

Wraps three `GoogleTool`-wrapped operations for **Gemini Data Analytics Agents**: `list_accessible_data_agents`, `get_data_agent_info`, `ask_data_agent`. `DataAgentCredentialsConfig` extends `BaseGoogleCredentialsConfig` and accepts exactly one of `credentials`, `external_access_token_key`, or a `client_id`/`client_secret` OAuth pair (scopes default to `bigquery`; token-cache key `"data_agent_token_cache"`). `DataAgentToolConfig(max_query_result_rows=50)` caps `ask_data_agent`'s row count.

```python
import google.auth
from google.adk.tools.data_agent import DataAgentToolset, DataAgentCredentialsConfig

creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/bigquery"])
toolset = DataAgentToolset(credentials_config=DataAgentCredentialsConfig(credentials=creds))
```

#### ComputerUseToolset
Why it matters: **Correction** — no class named `ComputerUseToolset` exists in `google.adk.tools.computer_use` in 2.7.1; the module instead exposes `ComputerUseTool` and a `BaseComputer` abstraction (grounding computer-use actions to a concrete computer backend). Treat "ComputerUseToolset" mentions in older docs as referring to `ComputerUseTool` plus a computer implementation, not a distinct toolset class.

```python
from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.adk.tools.computer_use.base_computer import BaseComputer
```

#### `GCPSkillRegistry` + `ApiRegistry` + `AgentRegistry`
**Source:** `google.adk.integrations.skill_registry.gcp_skill_registry`, `.api_registry.api_registry`, `.agent_registry.agent_registry`

Three Vertex AI-backed discovery/proxy clients:

- **`GCPSkillRegistry(project_id=None, location=None)`** implements the `SkillRegistry` ABC over a lazily-created `vertexai.Client`; falls back to `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` env vars. `get_skill(name=)` base64-decodes and unzips the skill filesystem (off the event loop, via `asyncio.to_thread`); `search_skills(query=)` returns `list[Frontmatter]` (discovery metadata only).
- **`ApiRegistry(api_registry_project_id, location="global", header_provider=None)`** fetches all registered MCP servers from Cloud API Registry at construction time and exposes `get_toolset(mcp_server_name, tool_filter=None, tool_name_prefix=None) -> McpToolset` per server.
- **`AgentRegistry(project_id, location, header_provider=None)`** — both args are required (no env fallback; raises `ValueError` if either is falsy). `get_mcp_toolset(mcp_server_name, auth_scheme=None, ...)` auto-resolves a `GcpAuthProviderScheme` from IAM bindings when no scheme is given (requires `GcpAuthProvider` registered via `CredentialManager.register_auth_provider` and the `agent-identity` extra). Also exposes `list_endpoints`, `get_endpoint`, `get_model_name`.

```python
from google.adk.integrations.agent_registry.agent_registry import AgentRegistry
from google.adk.auth.credential_manager import CredentialManager
from google.adk.integrations.agent_identity.gcp_auth_provider import GcpAuthProvider

CredentialManager.register_auth_provider(GcpAuthProvider())
registry = AgentRegistry(project_id="my-project", location="us-central1")
toolset = registry.get_mcp_toolset(mcp_server_name="projects/my-project/locations/us-central1/agents/my-agent/mcpServers/default")
```

#### `EnvironmentSimulationFactory` + `EnvironmentSimulationEngine` + `ToolSpecMockStrategy`
**Source:** `google.adk.tools.environment_simulation.*` — `@experimental(FeatureName.ENVIRONMENT_SIMULATION)`

Lets you test agents **without calling real tools**. `EnvironmentSimulationFactory.create_callback(config)` / `.create_plugin(config)` each build one shared `EnvironmentSimulationEngine`, whose app-scoped `_state_store` dict is captured by that single callback/plugin instance (create a fresh one per session for isolation). `EnvironmentSimulationEngine.simulate(tool, args, tool_context)` is used as a `before_tool_callback`: it returns `None` (real tool runs) for any tool not present in `EnvironmentSimulationConfig.tool_simulation_configs`.

```python
class EnvironmentSimulationConfig(BaseModel):
    simulation_model: str
    tool_simulation_configs: list[ToolSimulationConfig]
    environment_data: str | None = None    # seed context (e.g. a DB snapshot) for the mock LLM
```

`ToolSpecMockStrategy` (selected via `MockStrategy.MOCK_STRATEGY_TOOL_SPEC`) is the production strategy: it prompts an LLM with the tool's schema, the shared state store, and the tool-connection map (which params are created vs consumed across tools), asking for a JSON response and a realistic 404 when a consumed ID isn't in the state store yet. `TracingMockStrategy` (`MOCK_STRATEGY_TRACING`) exists but is **deprecated** — its `mock()` always returns `{"status": "error", "error_message": "Not implemented"}`; pass historical call traces via `EnvironmentSimulationConfig.tracing` instead, which `ToolSpecMockStrategy` uses as prompt context.

`agent_simulator` (the pre-2.4.0 module path) is now deprecated, emits a `DeprecationWarning`, and re-exports `EnvironmentSimulationFactory as AgentSimulatorFactory` for compatibility — migrate to `google.adk.tools.environment_simulation`.

```python
from google.adk.tools.environment_simulation import EnvironmentSimulationFactory
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, MockStrategy,
)

config = EnvironmentSimulationConfig(
    simulation_model="gemini-2.5-flash",
    tool_simulation_configs=[
        ToolSimulationConfig(tool_name="create_ticket", mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC),
    ],
)
agent.before_tool_callback = EnvironmentSimulationFactory.create_callback(config)
```


### Workflows & Graph Orchestration

#### `BaseNode` — the workflow node contract
**Source:** `google.adk.workflow._base_node`

Every graph node (including `BaseAgent`) inherits from this Pydantic `BaseModel`.

```python
class BaseNode(BaseModel):
    name: str                            # must be a valid Python identifier
    description: str = ''
    rerun_on_resume: bool = False        # True → re-execute from scratch on HITL resume
    wait_for_output: bool = False        # True → WAITING state without emitting output/route yet
    retry_config: RetryConfig | None = None
    timeout: float | None = None
    input_schema: SchemaType | None = None
    output_schema: SchemaType | None = None
    state_schema: type[BaseModel] | None = None
```

`run()` is `@final` (subclasses cannot override it) and normalizes whatever `_run_impl()` yields: `None` is skipped, an `Event` passes through (validating `event.output` if set), a `RequestInput` becomes a HITL interrupt event, and any other value is wrapped as `Event(output=validated_value)`. Subclasses override `_run_impl()` only. `START = BaseNode(name='__START__')` is a sentinel that is never executed — `Workflow` seeds its direct successors directly.

```python
from google.adk.workflow._base_node import BaseNode
from google.adk.agents.context import Context

class EchoNode(BaseNode):
    name: str = "echo"
    async def _run_impl(self, *, ctx: Context, node_input):
        yield f"Echo: {node_input}"
```

#### `Node` — user-subclassable workflow node
**Source:** `google.adk.workflow._node`

The recommended base for custom node logic: implement `run_node_impl` and optionally set `parallel_worker=True` to fan out over a list input.

```python
class Node(BaseNode):
    parallel_worker: bool = False        # frozen=True — set only at construction
    max_parallel_workers: int | None = None   # only valid with parallel_worker=True; must be >= 1
```

When `parallel_worker=True`, `model_post_init` clones the node (with `parallel_worker=False` on the clone, to prevent infinite recursion) and wraps it in `_ParallelWorker`, which calls `run_node_impl` once per list element, capped at `max_parallel_workers` concurrently.

```python
from typing import AsyncGenerator, Any
from google.adk.workflow._node import Node
from google.adk.tools.tool_context import ToolContext

class SentimentNode(Node):
    async def run_node_impl(self, *, ctx: ToolContext, node_input: Any) -> AsyncGenerator[Any, None]:
        yield {"text": node_input[:30], "score": len(node_input) % 5}

parallel_sentiment = SentimentNode(name="sentiment", parallel_worker=True, max_parallel_workers=4)
```

#### `FunctionNode` — `parameter_binding` + parallel fan-out
**Source:** `google.adk.workflow._function_node`

```python
class FunctionNode(BaseNode):
    parameter_binding: Literal['state', 'node_input'] = 'state'
```

- `'state'` (default) — the wrapped function's parameters are read directly from `ctx.state` by name.
- `'node_input'` — the function behaves like a tool: the caller passes a dict, and ADK infers `input_schema`/`output_schema` from the function's type hints, coercing a `dict` to a `BaseModel` annotation, `types.Content` to `str` (joined text, non-text parts dropped with a log), and anything else via `TypeAdapter(...).validate_python(...)`. `_PASSTHROUGH_OUTPUT_TYPES = (types.Content, Event, RequestInput)` skip validation on output.
- `parallel_worker=True` (same semantics as `Node`, above) makes the wrapped function run once per list element.
- Both sync and async generators are supported as the function body (sync generators are wrapped via `_sync_to_async_gen()`).

```python
from pydantic import BaseModel
from google.adk.workflow._function_node import FunctionNode

class SearchRequest(BaseModel):
    query: str; max_results: int = 10

async def search(request: SearchRequest) -> dict:
    return {"hits": await run_search(request.query, request.max_results)}

search_node = FunctionNode(func=search, name="search", parameter_binding="node_input")
```

#### `Workflow` — graph-based orchestration node
**Source:** `google.adk.workflow._workflow` (verified 2.7.1)

Replaces `SequentialAgent`/`ParallelAgent`/`LoopAgent` for branching pipelines. Declare `edges` using the chain syntax (below) and the engine fans them out to parallel `NodeRunner` tasks, replays/resumes from session events, and enforces `max_concurrency`.

```python
class Workflow(BaseNode):
    rerun_on_resume: bool = True         # default True, unlike plain BaseNode
    edges: list[EdgeItem] = []
    max_concurrency: int | None = None   # only throttles STATIC (edge-triggered) nodes
    graph: Graph | None = None
```

`model_post_init` builds `self.graph = Graph.from_edge_items(self.edges)` (if `edges` were given and `graph` wasn't) and calls `self.graph.validate_graph()`, then validates that any `FunctionNode` parameter names match `state_schema` fields. `max_concurrency` deliberately excludes dynamic nodes (those scheduled via `ctx.run_node()`) — throttling them could deadlock the event loop.

`state_schema` (a Pydantic `BaseModel` *class*, inherited from `BaseNode`) is Workflow-specific in effect: `ctx.state` mutations are validated against its fields (raising `StateSchemaError` on mismatch), and matching field names become **injected parameters** in `@node` functions automatically.

```python
from google.adk.workflow import Workflow
from google.adk.workflow._base_node import START
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._function_node import FunctionNode

wf = Workflow(
    name="enrichment",
    edges=[
        ("START", (enrich_a, enrich_b, enrich_c)),   # fan-out
        (enrich_a, merge), (enrich_b, merge), (enrich_c, merge),
    ],
    max_concurrency=2,
)
```

#### `JoinNode` — all-predecessors barrier
**Source:** `google.adk.workflow._join_node` (verified 2.7.1)

```python
class JoinNode(BaseNode):
    @property
    def _requires_all_predecessors(self) -> bool:
        return True   # overrides BaseNode's default of False

    def _validate_input_data(self, data):
        # when input_schema is set, validates EACH predecessor's contribution independently
        if self.input_schema and isinstance(data, dict):
            return {k: self._validate_schema(v, self.input_schema) for k, v in data.items()}
        return super()._validate_input_data(data)

    async def _run_impl(self, *, ctx, node_input):
        yield Event(output=node_input, branch=ctx._invocation_context.branch)  # pure pass-through
```

Runs only after every predecessor edge has delivered — the natural counterpart to a fan-out. `node_input` is a `dict` keyed by branch identifier (usually the predecessor node's name, but a custom `Trigger(branch=...)` can differ — inspect actual keys rather than assuming).

```python
from google.adk.workflow._join_node import JoinNode
join = JoinNode(name="merge")
# downstream: def combine(node_input: dict): node_input["fetch_weather"], node_input["fetch_news"]
```

#### `Trigger` — routing data model for workflow edges
**Source:** `google.adk.workflow._trigger` (verified 2.7.1)

```python
class Trigger(BaseModel):
    model_config = ConfigDict(ser_json_bytes='base64')
    input: Any = None                    # payload forwarded to the downstream node
    use_sub_branch: bool = False         # isolate the downstream node's event history
    branch: str | None = None            # inherited from the predecessor node
    isolation_scope: str | None = None   # explicit scope tag for state partitioning
```

```python
from google.adk.workflow._trigger import Trigger
def make_tenant_trigger(tenant_id, payload):
    return Trigger(input=payload, use_sub_branch=True, isolation_scope=f"tenant:{tenant_id}")
```

**Import path note:** `Trigger` is not re-exported at the `google.adk.workflow` package level in 2.7.1 (only from the private `google.adk.workflow._trigger` submodule) despite several older docs showing a bare `from google.adk.workflow import Trigger`.

#### `Graph` + `Edge` + chain syntax (`parse_edge_items`)
**Source:** `google.adk.workflow._graph`, `.utils._graph_parser`

```python
RouteValue: TypeAlias = bool | int | str
NodeLike: TypeAlias = BaseNode | BaseTool | Callable[..., Any] | Literal["START"]
RoutingMap: TypeAlias = dict[RouteValue, NodeLike | tuple[NodeLike, ...]]
ChainElement: TypeAlias = NodeLike | tuple[NodeLike, ...] | RoutingMap
EdgeItem: TypeAlias = Edge | tuple[ChainElement, ...]
DEFAULT_ROUTE = "__DEFAULT__"

class Edge(BaseModel):
    from_node: BaseNode; to_node: BaseNode; route: RouteValue | list[RouteValue] | None = None

class Graph(BaseModel):
    nodes: list[BaseNode] = []
    edges: list[Edge] = []

    @classmethod
    def from_edge_items(cls, edge_items: list[EdgeItem]) -> "Graph": ...

    def get_next_pending_nodes(self, node_name, routes_to_match) -> list[str]:
        # 1. edges with route=None always fire
        # 2. edges matching any value in routes_to_match fire
        # 3. if nothing matched and a DEFAULT_ROUTE edge exists, it fires
        ...
```

`nodes` is auto-inferred from `edges` (deduplicated by `id(node)`) — passing `nodes` explicitly raises. A chain tuple `(a, b, c)` expands to unconditional edges `a→b, b→c`; a `dict` in chain position `i+1` creates conditional fan-out edges from element `i` (values may themselves be tuples for parallel fan-out on one route). `DEFAULT_ROUTE` works as an edge-side fallback tag; a node may also explicitly `ctx.route = DEFAULT_ROUTE` — since `"__DEFAULT__"` never matches a specific route, the fallback edge still fires, same net effect as emitting no route.

```python
wf = Workflow(name="branching", edges=[
    ("START", classifier, {"A": path_a, "B": path_b, DEFAULT_ROUTE: fallback}),
    (path_a, merge), (path_b, merge), (fallback, merge),
])
```

**Import path note:** `Edge` is re-exported publicly from `google.adk.workflow`, but `Graph` itself is **not** in `workflow.__all__` — import it from the private `google.adk.workflow._graph` submodule.

#### Graph validation suite
**Source:** `google.adk.workflow.utils._graph_validation`

`validate_graph(nodes, edges)` runs automatically at `Workflow` construction (`model_post_init` → `_build_graph`), delegating to 7 checks and returning the set of terminal node names:

```python
def validate_graph(nodes, edges) -> set[str]:
    node_names = _validate_duplicate_node_names(nodes)     # Counter-based duplicate detection
    _validate_start_node(node_names)
    _validate_connectivity(edges, node_names)              # BFS from START; unreachable nodes → ValueError
    _validate_duplicate_edges(edges)
    _validate_default_routes(edges)                        # DEFAULT_ROUTE can't combine with other routes in a list
    _detect_unconditional_cycles(edges, node_names)         # DFS over route=None edges only
    _validate_static_schemas(edges)
    _validate_chat_agent_wiring(edges)                      # chat-mode LlmAgent must follow START directly
    return _compute_terminal_nodes(nodes, edges)
```

A cycle is only an error if it contains **no** conditional (routed) edge — a router can always break an unconditional loop. `_validate_chat_agent_wiring` rejects a `mode='chat'` `LlmAgent` placed after any node other than `START`, because chat-mode agents rely on conversational history rather than a structured `node_input`.

```python
import pytest
from google.adk.workflow import Workflow, START, Edge

def test_unconditional_cycle_detected():
    with pytest.raises(ValueError, match="Unconditional cycle detected"):
        Workflow(name="bad", edges=[(START, a, b), Edge(from_node=b, to_node=a)])
```

#### `RetryConfig` — exponential-backoff retry for workflow nodes
**Source:** `google.adk.workflow._retry_config` (verified 2.7.1)

Attach to any `BaseNode` (`LlmAgent`, `FunctionNode`, custom nodes) via `retry_config=`.

```python
class RetryConfig(BaseModel):
    max_attempts: int | None = None     # None → runtime default of 5 (including the first attempt)
    initial_delay: float | None = None  # None → 1.0 s
    max_delay: float | None = None      # None → 60.0 s
    backoff_factor: float | None = None # None → 2.0
    jitter: float | None = None         # None → 1.0 (= ±100% symmetric randomness); 0.0 disables it
    exceptions: list[str | type[BaseException]] | None = None  # None → retry on ALL exceptions

    @field_validator('exceptions', mode='before')
    @classmethod
    def _normalize_exceptions(cls, v):
        # exception classes are converted to their __name__ string for uniform runtime checking
        ...
```

`_should_retry_node` (in `workflow.utils._retry_utils`) checks `node_state.attempt_count` (1-based) against `max_attempts` and, if `exceptions` is set, the failing exception's class name. `_get_retry_delay` computes `min(initial_delay * backoff_factor ** (attempt_count - 1), max_delay)`, then applies a **symmetric additive** jitter offset in `[-jitter*delay, +jitter*delay]` (so with the default `jitter=1.0`, the first retry falls anywhere in `[0, 2 × initial_delay]`).

```python
from google.adk.workflow import RetryConfig
flaky = FunctionNode(name="api_caller", func=call_external_api, retry_config=RetryConfig(
    max_attempts=4, initial_delay=1.0, backoff_factor=2.0, jitter=0.5,
    exceptions=["HTTPStatusError", "TimeoutException"],
))
```

Because `LlmAgent`/`BaseAgent` share the same `BaseNode` MRO as workflow nodes, `RetryConfig` is directly usable on agents too (`LlmAgent(..., retry_config=RetryConfig(...))`), not just on `FunctionNode`/custom nodes.

#### `DynamicNodeScheduler` — `ctx.run_node()` internals
**Source:** `google.adk.workflow._dynamic_node_scheduler`, `._schedule_dynamic_node` (Protocol)

Implements `ctx.run_node()`: scheduling a child node at runtime rather than through static edges.

```python
class ScheduleDynamicNode(Protocol):
    def __call__(self, ctx: Context, node: Any, node_input: Any, *,
                 node_name: str | None = None, use_as_output: bool = False,
                 run_id: str, use_sub_branch: bool = False,
                 override_branch: str | None = None,
                 override_isolation_scope: str | None = None) -> Awaitable[Context]: ...
    # ctx.run_node() unwraps the returned Context and hands back only child_ctx.output (Any)

@dataclass(kw_only=True)
class DynamicNodeRun:
    state: NodeState; output: Any = None
    task: asyncio.Task | None = None
    transfer_to_agent: str | None = None
    recovered_state: "_ChildScanState | None" = None   # rehydrated from session events

@dataclass(kw_only=True)
class DynamicNodeState:
    runs: dict[str, DynamicNodeRun] = field(default_factory=dict)  # keyed by node_path, e.g. "/wf@1/child@1"
    interrupt_ids: set[str] = field(default_factory=set)
    replay_manager: ReplayManager = field(default_factory=ReplayManager)
```

Three execution paths: **Fresh** (no prior events → run normally); **Completed/dedup** (session events show the node already finished, or a task for the same `node_path` is already running → return the cached/in-flight result without re-executing); **Waiting/resume** (prior events show an interrupt → rehydrate and either fast-forward past resolved interrupts or bubble unresolved ones up). `node_input` is validated against `node.input_schema` first; a validation error is re-raised with the node name folded into the title for clearer debugging. A `ReplayManager` chronological sequence barrier (`advance_sequence`/`wait_sequence`) ensures dynamic node results replay in the same order as the original run.

```python
@node(rerun_on_resume=True)   # required whenever a node calls ctx.run_node()
async def orchestrate(ctx):
    summary = await ctx.run_node(researcher, node_input="quantum computing", run_id="run-001")
    return summary   # run_node() returns the child's output directly, not a Context
```

#### `build_node` + `is_node_like`
**Source:** `google.adk.workflow.utils._workflow_graph_utils`

Converts any `NodeLike` value into a concrete `BaseNode`; called internally by the edge parser and by `_dispatch_task_fc` (chat-mode task delegation, below).

```python
def build_node(node_like: NodeLike, *, name: str | None = None,
                rerun_on_resume: bool | None = None, retry_config: RetryConfig | None = None,
                timeout: float | None = None, auth_config: Any = None,
                parameter_binding: Literal['state', 'node_input'] = 'state') -> BaseNode: ...
```

| Input | Output |
|---|---|
| `"START"` | the `START` singleton |
| `LlmAgent` | a clone with `rerun_on_resume=True` forced; `mode` defaulted to `'chat'` if `parent_agent` is set, else `'single_turn'`; `'task'`/`'chat'` modes get `wait_for_output=True` |
| other `BaseNode` | `model_copy(update=kwargs)` if overrides given, else returned as-is |
| `BaseTool` | wrapped in `_ToolNode` |
| `Callable` | wrapped in `FunctionNode` (`rerun_on_resume` defaults `False`) |

```python
standalone = build_node(LlmAgent(name="s", model="gemini-2.5-flash"))
print(standalone.mode)   # 'single_turn'
```

#### `run_llm_agent_as_node` + helpers — `LlmAgent` as a workflow node
**Source:** `google.adk.workflow._llm_agent_wrapper`

The async generator the workflow engine calls whenever an `LlmAgent` sits in a `Workflow` graph, dispatching on three modes.

| Mode | Behavior |
|---|---|
| `single_turn` | one `run_async` call; `include_contents` defaults to `'none'` unless the agent set it explicitly; output extracted via `process_llm_agent_output` on every event |
| `chat` | an outer `while True` dispatch loop re-enters `agent.run_async` after each task-delegation function call, letting a coordinator sequentially delegate to multiple sub-agents across LLM rounds |
| `task` | waits for the `finish_task` function-call/response handshake before promoting a result; retries on validation failure |

Only these three values are accepted; anything else raises `ValueError` immediately. `prepare_llm_agent_context` shallow-copies the `InvocationContext` for `single_turn` mode (isolating its events under `isolation_scope`, `ic.session.model_copy(deep=False)` for a cheap independent session reference) — `chat`/`task` reuse the original context unchanged. `process_llm_agent_output` skips partial/FC/non-model events, strips `part.thought` parts before concatenating text, validates against `output_schema` if set, and writes `output_key` into `ctx.actions.state_delta`.

```python
summariser = LlmAgent(name="summariser", model="gemini-2.5-flash",
                       instruction="Summarise node_input.", output_key="summary")
# mode is left unset here; build_node defaults it to 'single_turn' for a standalone node
wf = Workflow(name="summarise_wf", edges=[(START, summariser)])
```

#### Chat-mode task-delegation dispatch loop
**Source:** `google.adk.workflow._llm_agent_wrapper`

When `agent.mode == 'chat'`, three phases run: **(1)** `_find_unresolved_task_delegations` scans session events authored by the coordinator or `'user'` for `_TaskAgentTool` function calls with no matching function response by ID — deliberately **not** filtered by `isolation_scope`, since a chat coordinator's conversation spans user turns. **(2)** the live loop re-enters `agent.run_async`, and on each task-delegation FC calls `_dispatch_task_fc` (which passes `run_id=fc.id`, making delegation idempotent across resumes) then synthesizes a `FunctionResponse` event and loops. **(3)** if no task FC appears, the LLM is done and the loop returns. Sub-agents passed via `sub_agents=[...]` with `mode="task"` are auto-wrapped as `_TaskAgentTool` by `LlmAgent`'s model validator — this is the only path that triggers the dispatch machinery (plain `tools=[AgentTool(...)]` does not).

```python
researcher = LlmAgent(name="researcher", model="gemini-2.5-flash", mode="task",
                       instruction="Research the topic and return key facts.")
writer = LlmAgent(name="writer", model="gemini-2.5-flash", mode="task",
                   instruction="Write a report based on research findings.")
coordinator = LlmAgent(name="coordinator", model="gemini-2.5-pro", mode="chat",
                        instruction="Call 'researcher' first, then 'writer'.",
                        sub_agents=[researcher, writer])
```

#### Task-mode `FinishTask` FC/FR handshake
**Source:** `google.adk.workflow._llm_agent_wrapper`

In `task` mode the LLM calls a special `finish_task` function when structured output is ready; the wrapper waits for `FinishTaskTool`'s success `FunctionResponse` before promoting a result, giving the tool a chance to reject invalid output and let the LLM retry. `_is_finish_task_success_fr(event)` returns `True` only when the response equals `FINISH_TASK_SUCCESS_RESULT` (a validation-error response returns `False`, so the loop keeps going). If `output_schema` is a primitive (`str`/`int`/`float`/`bool`), `FinishTaskTool` wraps the value under a sentinel `_wrapper_key`, which the loop unwraps so `event.output` is the bare primitive.

`mode='task'` agents **cannot** be placed directly as static workflow graph nodes — `Workflow.__init__` raises `ValueError`. They must be dispatched either as entries in a `mode='chat'` coordinator's `sub_agents` (auto-wrapped as `_TaskAgentTool`) or dynamically via `ctx.run_node()` from a `FunctionNode`. Note also: `output_key` has **no effect** in task mode — the task branch never calls `process_llm_agent_output`; the result only ever surfaces via `event.output`.

```python
from pydantic import BaseModel
class TravelPlan(BaseModel):
    destination: str; days: int

planner = LlmAgent(name="planner", model="gemini-2.5-pro", mode="task",
                    instruction="Create a travel plan.", output_schema=TravelPlan)
coordinator = LlmAgent(name="coordinator", model="gemini-2.5-flash", mode="chat",
                        instruction="Ask the planner for a plan, then summarise.",
                        sub_agents=[planner])
```

#### HITL workflow utilities
**Source:** `google.adk.workflow.utils._workflow_hitl_utils`

```python
REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'
REQUEST_CREDENTIAL_FUNCTION_CALL_NAME = 'adk_request_credential'

def create_request_input_event(request_input: RequestInput) -> Event: ...
    # sets long_running_tool_ids=[request_input.interrupt_id]; response_schema → JSON schema
def create_request_input_response(interrupt_id: str, response: Mapping[str, Any]) -> types.Part: ...
def create_auth_request_event(auth_config: AuthConfig, interrupt_id: str) -> Event: ...
async def process_auth_resume(response_data: Any, auth_config: AuthConfig, state: State) -> None: ...
    # tries AuthConfig.model_validate(response_data) first; on failure the fallback is
    # auth-type-specific: API_KEY treats response_data as the raw key string (an
    # AuthCredential dict here would be stringified, not parsed); OAuth2/OIDC calls
    # AuthCredential.model_validate(response_data) instead.
def has_auth_credential(auth_config: AuthConfig, state: State) -> bool: ...
```

The correct pattern for a node that requests input is: wrap it `FunctionNode(func=..., rerun_on_resume=True)`, check `ctx.resume_inputs` **at the top** of the function body first (populated on resume), and only `yield create_request_input_event(...)` on the first run. Without `rerun_on_resume=True`, the replay interceptor fast-forwards by setting the node's output directly from the resolved response — the generator body never re-executes, so a resume-handling branch placed after the `yield` never runs.

```python
async def approval_gate(ctx):
    response = ctx.resume_inputs.get("approval-001")
    if response is not None:
        if response.get("result") != "approve":
            raise ValueError("Rejected by human operator.")
        return
    yield create_request_input_event(RequestInput(
        interrupt_id="approval-001", message="Approve this action?",
        response_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    ))

gate_node = FunctionNode(func=approval_gate, rerun_on_resume=True)
```

#### `resolve_and_derive_transfer_context` — agent transfer routing
**Source:** `google.adk.workflow.utils._transfer_utils`

Resolves the target agent and derives the `Context` it should run under, in one pass, whenever `transfer_to_agent` fires.

```python
def resolve_and_derive_transfer_context(
    target_name: str, current_agent: BaseAgent, root_agent: BaseAgent,
    curr_ctx: Context, curr_parent_ctx: Context | None,
) -> tuple[BaseAgent, Context | None] | tuple[None, None]: ...
```

| Case | Condition | Result |
|---|---|---|
| SELF | `target.name == current.name` | raises `ValueError` |
| CHILD | `target.parent_agent.name == current.name` | `(target, curr_ctx)` — nests deeper |
| SIBLING | same `parent_agent` as current | `(target, curr_parent_ctx)` — shares parent context |
| PARENT | `current.parent_agent.name == target.name` | walks the context chain for a matching node name; falls back to the outermost root context if none matches |
| Unrelated | none of the above, but target exists | `(target, None)` |

`sub_agents` **must** be passed at construction time (not assigned afterward) — only then does `model_post_init` set `parent_agent` on each sub-agent, which the SIBLING/PARENT checks depend on. A grandparent transfer (leaf → root, skipping the direct parent) is **not** Case "PARENT" (that requires a *direct* parent match) — it falls through to the unrelated case.

#### `LoopAgent` / `ParallelAgent` / `SequentialAgent` → `Workflow` migration
**Source:** `google.adk.agents.loop_agent`, `.parallel_agent`, `.sequential_agent` (verified 2.7.1 — all three still exist and are fully functional, but decorated `@deprecated`)

All three carry the same verified deprecation notice: *"...deprecated in favor of Workflow and will be removed in a future version. Workflow cannot yet be used as an LlmAgent sub-agent."* — that caveat means the migration isn't always drop-in yet when the pipeline itself needs to be nested inside another `LlmAgent`'s `sub_agents`.

- **`LoopAgent`** stops when any event has `actions.escalate=True`, or after `max_iterations` full passes (each pass calls `ctx.reset_sub_agent_states(name)` to restart sub-agent state fresh). Migration: a `Workflow` routing map where the loop body routes back to itself (`ctx.route = "continue"`) or to a terminal node (`ctx.route = "done"`); track the iteration count manually in `ctx.state`.
- **`ParallelAgent`** isolates each sub-agent's **event history** per branch (`branch = "{parent}.{sub}"`) but does **not** namespace `state_delta` — parallel branches writing the same state key overwrite each other; use unique `output_key`s. `run_live` is not implemented (raises `NotImplementedError`). Migration: a `Workflow` fan-out (nested tuple in `edges`) plus a `JoinNode`.
- **`SequentialAgent`** persists `SequentialAgentState.current_sub_agent` for resume; if a saved sub-agent name is no longer in the list, `_get_start_index` logs a warning and restarts from index 0. Migration: a `Workflow` chain (`edges=[(START, a, b, c)]`).

```python
# LoopAgent(name="refine", sub_agents=[critic], max_iterations=5)  — deprecated
# equivalent:
@node(rerun_on_resume=True)
async def shorten_draft(node_input, ctx):
    count = ctx.state.get("iters", 0) + 1
    ctx.state["iters"] = count
    ctx.route = "done" if len(node_input.split()) <= 100 or count >= 5 else "continue"
    return node_input[:500]

refine_workflow = Workflow(name="refine", edges=[
    (START, shorten_draft, {"continue": shorten_draft, "done": publish}),
])
```

#### `AutoFlow` vs `SingleFlow`
**Source:** `google.adk.flows.llm_flows.auto_flow`, `.single_flow`

Every `LlmAgent` has an `_llm_flow` attribute set at construction, controlling the ordered request/response processor pipeline. `SingleFlow` assembles the standard pipeline (basic → auth → confirmation → instructions → identity → compaction → contents → context-cache → interactions → NL-planning → code-execution → output-schema, request-side; NL-planning → code-execution → basic, response-side). `AutoFlow(SingleFlow)` adds exactly one processor: `agent_transfer.request_processor`, which injects the `transfer_to_agent` function declaration and handles the model's transfer response by setting `ctx.actions.transfer_to_agent`.

`AutoFlow` is chosen whenever the agent has `sub_agents`, or in any configuration other than `disallow_transfer_to_parent=True` **and** `disallow_transfer_to_peers=True` **and** no `sub_agents` (which gets `SingleFlow`). Transfer directions `AutoFlow` allows: parent→sub-agent, sub-agent→parent, and sub-agent→peer (only when the parent is an `LlmAgent` and `disallow_transfer_to_peers=False`, the default).

```python
specialist = LlmAgent(name="tax-specialist", model="gemini-2.5-flash",
                       instruction="Handle tax questions only.",
                       disallow_transfer_to_peers=True)   # blocks lateral transfer, keeps SingleFlow-like routing
```

#### `_AgentTransferLlmRequestProcessor` — agent-transfer pipeline stage
**Source:** `google.adk.flows.llm_flows.agent_transfer`

Runs on every LLM call for `LlmAgent` instances (a no-op for agents lacking `disallow_transfer_to_parent`, i.e. non-`LlmAgent` `BaseAgent`s). Transfer targets: sub-agents *not* in `single_turn`/`task` mode; the parent (unless `disallow_transfer_to_parent=True`); peers (other sub-agents of the parent, unless `disallow_transfer_to_peers=True`). No routing instructions are injected at all when `agent.mode in ('task', 'single_turn')` — purely functional sub-agents stay silent about routing. Module-level singleton `request_processor = _AgentTransferLlmRequestProcessor()`.

#### `ReplayManager` — unified replay orchestrator
**Source:** `google.adk.workflow.utils._replay_manager`

Consolidates event rehydration, replay interception, and sequence-barrier synchronization for both static and dynamic workflow nodes.

```python
class ReplayManager:
    def scan_workflow_events(self, ctx: Context) -> tuple[dict[str, "_ChildScanState"], list[str]]: ...
    def prepare_parent_sequence_barrier(self, ctx: Context, parent_path: str) -> "ReplaySequenceBarrier": ...
    async def advance_sequence(self, parent_path: str, key: str) -> None: ...
    async def wait_sequence(self, parent_path: str, key: str) -> None: ...
```

`scan_workflow_events()` rehydrates completed child executions and computes the chronological completion order (last terminal event wins per node path), producing a `ReplaySequenceBarrier` that forces replaying dynamic nodes to fire in the same order as the original run — preventing non-deterministic divergence on resume.

#### NodeInterruptedError / NodeTimeoutError / DynamicNodeFailError
Why it matters: the exception types a workflow raises for HITL interruption, node timeout, and dynamic-node failure respectively. `NodeTimeoutError` is public (`workflow.__all__`); the other two remain importable only from the private `google.adk.workflow._errors` module — confirmed still present and importable in 2.7.1, contrary to no explicit doc claim of removal but worth flagging since they're easy to miss given they're absent from `__all__`.

```python
from google.adk.workflow._errors import NodeInterruptedError, DynamicNodeFailError
from google.adk.workflow import NodeTimeoutError  # this one is public
```

### Streaming & Live Sessions

#### `LiveRequest` + `LiveRequestQueue`
**Source:** `google.adk.agents.live_request_queue` (verified 2.7.1)

The write side of a bidirectional (`Runner.run_live()`) agent session.

```python
class LiveRequest(BaseModel):
    model_config = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')
    content: Optional[types.Content] = None          # turn-by-turn text
    blob: Optional[types.Blob] = None                # realtime audio/video chunk
    activity_start: Optional[types.ActivityStart] = None
    activity_end: Optional[types.ActivityEnd] = None
    audio_stream_end: bool = False                   # confirmed 2.7.1 field, not documented pre-2.5.0
    close: bool = False                              # shutdown sentinel
    partial: bool = False                             # incomplete turn (accumulate before forwarding)
    state_delta: Optional[dict[str, Any]] = None      # always applied, regardless of which other field is set

class LiveRequestQueue:
    def send_content(self, content: types.Content, partial: bool = False) -> None: ...
    def send_realtime(self, blob: types.Blob) -> None: ...
    def send_activity_start(self) -> None: ...
    def send_activity_end(self) -> None: ...
    def send(self, req: LiveRequest) -> None: ...
    async def get(self) -> LiveRequest: ...
    def close(self) -> None: ...   # enqueues LiveRequest(close=True) as a sentinel
```

Priority order when multiple fields are set on one `LiveRequest` (highest first): `activity_start > activity_end > blob > content`; `state_delta` is always applied on top regardless. Prefer separate `send_*` calls over hand-building compound requests. All `send_*` methods call `put_nowait`, so they're safe from sync code or a different async task.

```python
queue = LiveRequestQueue()
queue.send_activity_start()
for chunk in audio_chunks:
    queue.send_realtime(types.Blob(mime_type="audio/pcm;rate=16000", data=chunk))
queue.send_activity_end()
queue.send(LiveRequest(
    content=types.Content(role="user", parts=[types.Part(text="Use my saved preferences.")]),
    state_delta={"user_language": "fr"},   # atomically applied alongside the content turn
))
queue.close()
```

#### `ActiveStreamingTool`
**Source:** `google.adk.agents.active_streaming_tool` (verified 2.7.1)

Tracks the two resources a currently-executing **streaming tool** needs during a live session: the background task and the input queue feeding it.

```python
class ActiveStreamingTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    task: Optional[asyncio.Task[Any]] = None
    stream: Optional[LiveRequestQueue] = None
```

ADK detects a tool parameter annotated `input_stream: LiveRequestQueue`, creates the queue, registers the `ActiveStreamingTool`, and starts feeding it from the live model stream automatically — your function only needs to declare the parameter.

```python
async def live_transcription_tool(input_stream: LiveRequestQueue, tool_context: ToolContext):
    parts = []
    while True:
        req = await input_stream.get()
        if req.close:
            break
        if req.blob:
            parts.append(f"[audio:{len(req.blob.data)}bytes]")
    return {"transcript": " ".join(parts)}

def cancel_if_running(active: ActiveStreamingTool) -> None:
    if active.task and not active.task.done():
        active.task.cancel()
    if active.stream:
        active.stream.close()
```

#### `TranscriptionEntry`
**Source:** `google.adk.agents.transcription_entry`

Typed container for accumulating raw audio/video blobs alongside model-generated `Content` for later transcription in a live session.

```python
class TranscriptionEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    role: Optional[str] = None            # "user"/"model" for speech turns; None for function calls
    data: Union[types.Blob, types.Content] # raw media OR a structured Content object
```

```python
transcript = [
    TranscriptionEntry(role="user", data=types.Blob(mime_type="audio/pcm;rate=16000", data=b"...")),
    TranscriptionEntry(role="model", data=types.Content(role="model", parts=[types.Part(text="I heard you.")])),
    TranscriptionEntry(role=None, data=types.Content(parts=[types.Part(function_call=fc)])),  # FC: role is None
]
```

#### `BaseLlmConnection`
**Source:** `google.adk.models.base_llm_connection`

Abstract base for live, bidirectional LLM connections consumed via `LiveRequestQueue`. `GeminiLlmConnection` (below) is the concrete implementation.

| Method | Direction | Purpose |
|---|---|---|
| `send_history(history)` | client → model | push full conversation history right after setup |
| `send_content(content)` | client → model | send a turn-completing user `Content`; model replies immediately |
| `_send_content(content, *, partial=False)` | client → model | partial-update hook; default delegates to `send_content` |
| `send_realtime(blob)` | client → model | push a raw audio/video chunk; VAD decides when to respond |
| `receive()` | model → client | `AsyncGenerator[LlmResponse, None]` |
| `close()` | — | tear down the transport |

Drain `receive()` with `Aclosing` (`google.adk.utils.context_utils`) so `aclose()` fires even if the consumer breaks out of the loop early:

```python
from google.adk.utils.context_utils import Aclosing

async def drain_connection(conn: BaseLlmConnection) -> list[str]:
    texts = []
    async with Aclosing(conn.receive()) as gen:
        async for resp in gen:
            if resp.content and resp.content.parts:
                texts += [p.text for p in resp.content.parts if p.text]
    return texts
```

#### `GeminiLlmConnection`
**Source:** `google.adk.models.gemini_llm_connection`

The `BaseLlmConnection` implementation for Gemini's live BIDI API, wrapping `google.genai.live.AsyncSession`.

```python
class GeminiLlmConnection(BaseLlmConnection):
    def __init__(self, gemini_session: live.AsyncSession,
                 api_backend: GoogleLLMVariant = GoogleLLMVariant.VERTEX_AI,
                 model_version: str | None = None): ...
```

`send_history()` always strips audio parts first (the Live API can't replay audio — only text/FC/FR survive), then either sends the filtered history directly via `send_live_content()`, or — for Gemini 3.1 Flash on Vertex AI, which doesn't support `history_config` in the SDK — collapses all prior turns into one user-role message prefixed `"Previous conversation history:\n"` with `ROLE: text` lines, avoiding a `1007` protocol error from mixed-role turn ordering.

#### `StreamingMode` + `PROGRESSIVE_SSE_STREAMING`
**Source:** `google.adk.agents.run_config` (`StreamingMode`), `google.adk.utils.streaming_utils` (`StreamingResponseAggregator`)

```python
class StreamingMode(Enum):
    NONE = None    # single aggregated content per turn (default)
    SSE  = 'sse'   # progressive partial events + a final aggregated event
    BIDI = 'bidi'  # reserved — actual bidi streaming uses runner.run_live(), not run_async()
```

| `event.partial` | Content | Meaning |
|---|---|---|
| `True` | text parts | streaming chunk — display for typewriter effect |
| `True` | function_call parts | in-flight FC argument accumulation — usually skip in UI |
| `False` | any | final aggregated response |

`StreamingResponseAggregator` has two code paths gated by `is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)` (default **on**): the new **progressive** mode accumulates parts in arrival order (text/FC/other interleave correctly, flushed on `close()`); the legacy **non-progressive** mode concatenates text chunks and only emits a merged event once a non-text chunk arrives. In progressive mode, streaming function-call args arrive as JSONPath fragments (`fc.partial_args: list[PartialArg(json_path, value)]`) applied incrementally to `_current_fc_args` until `fc.will_continue` is `False`.

Because progressive mode emits **both** partial chunks and a final aggregated text event, naively printing every event double-prints text — check `event.partial` and skip the final duplicate:

```python
async for event in runner.run_async(..., run_config=RunConfig(streaming_mode=StreamingMode.SSE)):
    if event.partial and event.content:
        has_text = any(p.text for p in event.content.parts)
        has_fc = any(p.function_call for p in event.content.parts)
        if has_text and not has_fc:
            print("".join(p.text or "" for p in event.content.parts), end="", flush=True)
    elif not event.partial and event.get_function_calls():
        for fc in event.get_function_calls():
            print(f"\n→ calling {fc.name}({fc.args})")
```

#### `ToolThreadPoolConfig`
**Source:** `google.adk.agents.run_config` (verified 2.7.1)

```python
class ToolThreadPoolConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_workers: int = Field(default=4, ge=1)
```

Routes tool execution to a `ThreadPoolExecutor` so the event loop stays responsive to live audio/interruptions instead of stalling on a blocking tool call. One pool is created per event loop and shared across concurrent invocations on it; it's torn down when the loop closes. Async tools run inside a **new** event loop within the worker thread (catching accidental blocking I/O in `async def` tools). A cancelled invocation drops queued-but-unstarted tool calls, but a tool already executing in a thread cannot be interrupted (threads aren't cancellable).

GIL caveat: helps blocking I/O and C extensions that release the GIL (network, file, DB, numpy/hashlib/PIL); does **not** help pure-Python CPU-bound loops — use `ProcessPoolExecutor` or chunk the work with `await asyncio.sleep(0)` for those.

```python
run_config = RunConfig(tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8))
```

#### `RunConfig` — per-invocation runtime configuration
**Source:** `google.adk.agents.run_config` (verified 2.7.1 — the field set has grown substantially release over release; this reflects the current, most complete shape)

Passed to `runner.run_async(..., run_config=...)` (and `run_live`) to control everything that varies per call without touching the agent definition. `extra='forbid'` — an unknown field raises `ValidationError` rather than being silently dropped.

```python
class RunConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    # --- Audio / Live ---
    speech_config: Optional[types.SpeechConfig] = None
    response_modalities: Optional[list[types.Modality]] = None   # defaults to AUDIO if unset
    avatar_config: Optional[types.AvatarConfig] = None
    realtime_input_config: Optional[types.RealtimeInputConfig] = None
    explicit_vad_signal: Optional[bool] = None
    output_audio_transcription: Optional[types.AudioTranscriptionConfig] = None
    input_audio_transcription: Optional[types.AudioTranscriptionConfig] = None
    translation_config: Optional[types.TranslationConfig] = None
    enable_affective_dialog: Optional[bool] = None
    proactivity: Optional[types.ProactivityConfig] = None
    save_live_blob: bool = False           # persist live video/audio to the artifact service
    save_live_audio: bool = False
    save_input_blobs_as_artifacts: bool = False   # DEPRECATED — use SaveFilesAsArtifactsPlugin instead

    # --- Streaming ---
    streaming_mode: StreamingMode = StreamingMode.NONE
    support_cfc: bool = False              # Compositional Function Calling (SSE only; routes through the Live API)

    # --- Session management ---
    session_resumption: Optional[types.SessionResumptionConfig] = None
    history_config: Optional[types.HistoryConfig] = None
    context_window_compression: Optional[types.ContextWindowCompressionConfig] = None
    get_session_config: Optional[GetSessionConfig] = None

    # --- Per-turn context injection ---
    model_input_context: list[types.Content] | None = None
    # injected into the LLM request for THIS invocation only — never persisted to the session

    include_thoughts_from_other_agents: bool = False
    # default False: sub-agent thought parts are stripped when reformatted as this agent's context

    # --- Execution limits ---
    max_llm_calls: int = 500               # <=0 → unlimited (logs a warning); sys.maxsize raises ValueError

    # --- Observability ---
    labels: Optional[dict[str, str]] = None            # billing/attribution labels
    http_options: Optional[types.HttpOptions] = None    # custom headers/timeout for this invocation's API calls
    custom_metadata: Optional[dict[str, Any]] = None
    telemetry: TelemetryConfig | None = None            # per-request OTel override

    tool_thread_pool_config: Optional[ToolThreadPoolConfig] = None
```

When `max_llm_calls` is exceeded, `Runner` raises `LlmCallsLimitExceededError` (see `_InvocationCostManager` below) — catch it to return a graceful "reached my processing limit" message.

```python
run_config = RunConfig(
    streaming_mode=StreamingMode.SSE, max_llm_calls=50,
    labels={"env": "production", "user_tier": "premium"},
    model_input_context=[types.Content(role="user", parts=[types.Part(text="[CONTEXT] AAPL: $189.42")])],
    get_session_config=GetSessionConfig(num_recent_events=30),
)
```

### Memory, Sessions & Artifacts

#### InMemorySessionService / DatabaseSessionService / VertexAiSessionService
Why it matters: the three baseline session backends, all exported from `google.adk.sessions.__all__`. `DatabaseSessionService` requires the optional `sqlalchemy` extra (`pip install google-adk[db]`) — its `ImportError` in a bare venv is an expected optional-dependency gate, not a docs bug.

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService, VertexAiSessionService
```

#### SqliteSessionService
Why it matters: a lightweight file-backed session store. **Correction**: not listed in `google.adk.sessions.__all__` in 2.7.1 (several docs show a top-level `from google.adk.sessions import SqliteSessionService`, which now fails) — must be imported from its own submodule.

```python
from google.adk.sessions.sqlite_session_service import SqliteSessionService
```

#### FirestoreSessionService / FirestoreMemoryService
Why it matters: **Correction** — these have moved out of `google.adk.sessions`/`google.adk.memory` entirely into a new `google.adk.integrations.firestore` subpackage, alongside the GCS toolset relocation noted above. This is a real, repeat-across-docs import-path break: every source volume mentioning them shows the old `google.adk.sessions`/`google.adk.memory` path.

```python
from google.adk.integrations.firestore.firestore_session_service import FirestoreSessionService
from google.adk.integrations.firestore.firestore_memory_service import FirestoreMemoryService
```

#### `InMemoryMemoryService`
**Source:** `google.adk.memory.in_memory_memory_service`

Implements `BaseMemoryService` with **keyword matching**, not semantic vector search — prototyping/testing only.

- Storage: `dict[str, dict[str, list[Event]]]` keyed `"{app_name}/{user_id}"` → `session_id` → events.
- Search: `_extract_words_lower()` splits on `\w+` and lower-cases; **any** query word matching any event word is a hit (OR logic, not AND).
- `add_events_to_memory()` dedups on `event.id` via a `set`.
- Events added with no `session_id` land under the sentinel key `"__unknown_session_id__"`.

```python
memory = InMemoryMemoryService()
await memory.add_session_to_memory(session)
result = await memory.search_memory(app_name="app", user_id="u1", query="Paris tower")
```

#### InMemoryMemoryService / VertexAiRagMemoryService / VertexAiMemoryBankService
Why it matters: the standard memory-service tier, all confirmed present and unchanged at `google.adk.memory`.

```python
from google.adk.memory import InMemoryMemoryService, VertexAiRagMemoryService, VertexAiMemoryBankService
```

#### `ToolContext.add_memory` / `add_events_to_memory` / `add_session_to_memory`
**Source:** `google.adk.tools.tool_context` (the `Context` class)

Three memory-write methods on `Context`, in increasing order of directness:

```python
async def add_memory(self, *, memories: Sequence[MemoryEntry],
                      custom_metadata: Mapping[str, object] | None = None) -> None: ...
async def add_events_to_memory(self, *, events: Sequence[Event],
                                custom_metadata: Mapping[str, object] | None = None) -> None: ...
async def add_session_to_memory(self) -> None: ...
```

| Method | Use when | Support |
|---|---|---|
| `add_session_to_memory()` | ingest the entire session at run end | all services |
| `add_events_to_memory(events=...)` | persist only the current turn's delta | `InMemoryMemoryService`, `VertexAiMemoryBankService`, `VertexAiRagMemoryService`; `NotImplementedError` elsewhere |
| `add_memory(memories=...)` | write structured `MemoryEntry` facts directly, bypassing event extraction | Vertex AI Memory Bank; `NotImplementedError` on `InMemoryMemoryService` |

```python
class MemoryEntry(BaseModel):
    content: types.Content
    custom_metadata: dict[str, Any] = {}
    id: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[str] = None
```

```python
async def save_user_preference(category: str, value: str, tool_context: ToolContext) -> dict:
    entry = MemoryEntry(
        content=types.Content(role="user", parts=[types.Part(text=f"User prefers {value} for {category}.")]),
        custom_metadata={"category": category, "value": value},
    )
    await tool_context.add_memory(memories=[entry])
    return {"saved": True}
```

#### InMemoryArtifactService / GcsArtifactService / FileArtifactService
Why it matters: all three remain exported from `google.adk.artifacts.__all__` unchanged — `FileArtifactService` in particular is confirmed still present, contradicting nothing (some earlier drafts of this summary flagged it as at-risk; it is not).

```python
from google.adk.artifacts import InMemoryArtifactService, GcsArtifactService, FileArtifactService
```

#### `InMemoryArtifactService` + `FileArtifactService`
**Source:** `google.adk.artifacts.in_memory_artifact_service`, `.file_artifact_service`

Both implement `BaseArtifactService` with the same versioning/scoping contract; `InMemoryArtifactService` for tests, `FileArtifactService` for local-disk persistence.

- **Scoping** — session-scoped by default (`{app}/{user}/{session}/{filename}`); `filename` starting with `"user:"` forces **user scope** (persists across sessions, `session_id` argument is ignored for that call); a `session_id=None` on `save_artifact` also implies user scope.
- **Versioning** — 0-based ints; each `save_artifact` call appends a new version (equal to the current version count before the append). `load_artifact(version=None)` returns the latest.
- `FileArtifactService(root_dir)` stores each version as `root/users/{user}/[sessions/{session}/]artifacts/{path}/versions/{n}/{filename}` plus a `metadata.json` sidecar (`FileArtifactVersion`), and rejects path-traversal filenames (`"../../x"`, absolute paths, empty names) via `InputValidationError`. Separators in a filename create nested directories.

```python
svc = InMemoryArtifactService()
v0 = await svc.save_artifact(app_name="app", user_id="u1", session_id="s1",
                              filename="notes.txt", artifact=types.Part(text="Draft 1"))
v1 = await svc.save_artifact(app_name="app", user_id="u1", session_id="s1",
                              filename="notes.txt", artifact=types.Part(text="Final"))
latest = await svc.load_artifact(app_name="app", user_id="u1", session_id="s1", filename="notes.txt")
# latest.text == "Final"; svc.list_versions(...) == [0, 1]

# user-scoped, cross-session
await svc.save_artifact(app_name="crm", user_id="alice", session_id="s1",
                         filename="user:profile.json", artifact=types.Part(text='{"tier":"gold"}'))
```

#### `ArtifactVersion` + `BaseArtifactService`
**Source:** `google.adk.artifacts.base_artifact_service`

```python
class ArtifactVersion(BaseModel):
    version: int
    canonical_uri: str        # backend URI, e.g. gs:// or file://
    custom_metadata: dict[str, Any]
    create_time: float
    mime_type: Optional[str]
```

`BaseArtifactService` methods: `save_artifact(...) -> int`, `load_artifact(...) -> Optional[types.Part]`, `list_artifact_keys(...) -> list[str]`, `delete_artifact(...) -> None`, `list_versions(...) -> list[int]`, `list_artifact_versions(...) -> list[ArtifactVersion]`, `get_artifact_version(...) -> Optional[ArtifactVersion]`. Scope rule: `session_id=None` → user-scoped; a real `session_id` → session-scoped.

#### `ParsedArtifactUri` + `parse_artifact_uri` + `get_artifact_uri` + `is_artifact_ref`
**Source:** `google.adk.artifacts.artifact_util`

Canonical encoder/decoder for `artifact://` URIs used by `ToolContext.save_artifact` and the artifact services above.

```python
class ParsedArtifactUri(NamedTuple):
    app_name: str; user_id: str; session_id: str | None; filename: str; version: int

def parse_artifact_uri(uri: str) -> ParsedArtifactUri | None: ...   # None for malformed/non artifact:// URIs
def get_artifact_uri(app_name, user_id, filename, version, session_id=None) -> str: ...
def is_artifact_ref(part: types.Part) -> bool: ...   # True if part.file_data.file_uri starts with "artifact://"
```

Two forms: session-scoped `artifact://apps/{app}/users/{user}/sessions/{session}/artifacts/{filename}/versions/{version}`; user-scoped (no `session_id`) omits the `sessions/{session}` segment.

```python
uri = get_artifact_uri("my_app", "user-123", "report.pdf", version=2, session_id="sess-abc")
parsed = parse_artifact_uri(uri)
assert parsed.session_id == "sess-abc" and parsed.version == 2
```

#### State / StateSchemaError
Why it matters: `State` is the typed, schema-validated session-state container; `StateSchemaError` is raised on a schema mismatch. Both remain exported from `google.adk.sessions`, unchanged.

```python
from google.adk.sessions import State, StateSchemaError
```

#### `BaseSessionService` + `GetSessionConfig` + `ListSessionsResponse`
**Source:** `google.adk.sessions.base_session_service`

The ABC every session backend implements (`InMemorySessionService`, `SqliteSessionService`, `DatabaseSessionService`, `VertexAiSessionService`, `PerAgentDatabaseSessionService`).

```python
class GetSessionConfig(BaseModel):
    num_recent_events: Optional[int] = None   # None → all; 0 → none; N → last N
    after_timestamp: Optional[float] = None   # None → all; float → events with timestamp >= value
```

| Method | Returns |
|---|---|
| `create_session(...)` | `Session` |
| `get_session(..., config=None)` | `Optional[Session]` — `config` filters events |
| `list_sessions(...)` | `ListSessionsResponse` — events/state stripped |
| `delete_session(...)` | `None` |
| `get_user_state(...)` | `dict` (user-scoped, shared across sessions) |
| `append_event(...)` | `Event` |

```python
lean = await svc.get_session(app_name="app", user_id="u1", session_id=sid,
                              config=GetSessionConfig(num_recent_events=0))
```

#### Session migration pipeline — `MIGRATIONS`, `upgrade()`
**Source:** `google.adk.sessions.migration.migration_runner`, `.migrate_from_sqlalchemy_pickle`, `._schema_check_utils`

Migrates a legacy pickle-based session schema (V0) to the current JSON schema (V1).

```python
MIGRATIONS = {SCHEMA_VERSION_0_PICKLE: (SCHEMA_VERSION_1_JSON, migrate_from_sqlalchemy_pickle.migrate)}
LATEST_VERSION = SCHEMA_VERSION_1_JSON

def upgrade(source_db_url: str, dest_db_url: str, allow_unsafe_unpickling: bool = False) -> None: ...
```

`upgrade()` raises `RuntimeError` if `source == dest` (no in-place migration); detects the current version via `get_db_schema_version()` and returns immediately if already latest; multi-step chains use temp SQLite files, deleted in a `finally` block even on failure. `allow_unsafe_unpickling=True` is required to migrate V0 data (which used Python pickle for `EventActions`) — only pass it for a source DB you trust. `to_sync_url()` strips async driver prefixes (`sqlite+aiosqlite://` → `sqlite://`) so you can reuse the same URL your `DatabaseSessionService` uses.

```python
from google.adk.sessions.migration.migration_runner import upgrade
upgrade(source_db_url="sqlite:///old_sessions.db", dest_db_url="sqlite:///new_sessions_v1.db",
        allow_unsafe_unpickling=True)
```

#### V1 session schema — `StorageSession` / `StorageEvent` / `StorageMetadata` / `DynamicJSON` / `PreciseTimestamp`
**Source:** `google.adk.sessions.schemas.v1`, `.shared`

The current on-disk representation for `DatabaseSessionService`/`SqliteSessionService`, replacing V0's pickled `EventActions` column with a single `event_data` JSON column.

- **`DynamicJSON(TypeDecorator)`** — dialect-adaptive: PostgreSQL → native `JSONB`; MySQL → `LONGTEXT` with manual `json.dumps`/`loads` (avoids truncation errors); everything else (SQLite, ...) → `TEXT` + JSON (de)serialization.
- **`PreciseTimestamp(TypeDecorator)`** — MySQL → `DATETIME(fsp=6)` (microsecond precision); elsewhere the SQLAlchemy default.
- **`StorageMetadata`** (`adk_internal_metadata` table) — single row `key="schema_version", value="1"`; this is how `get_db_schema_version()` detects V1 without inspecting columns.
- **`StorageEvent.from_event(session, event)`** serializes via `event.model_dump(exclude_none=True, mode="json")`; **`.to_event()`** deserializes with the primary-key columns (`id`, `invocation_id`, `timestamp`) explicitly overriding any duplicate fields inside `event_data`.
- `StorageSession` cascades `all, delete-orphan` on its `StorageEvent` relationship — deleting a session auto-deletes its events.

#### `PerAgentDatabaseSessionService` + `PerAgentFileArtifactService`
**Source:** `google.adk.cli.utils.local_storage`

Route session/artifact operations to each agent's own `.adk` folder (`<agent_dir>/.adk/session.db`, `.adk/artifacts/`), with an optional `app_name_to_dir` mapping to alias a logical app name to a different on-disk directory (only the directory changes — the logical `app_name` passed downstream is unchanged). `_get_service(app_name)` acquires an `asyncio.Lock()` before lazily creating each per-agent service, to avoid duplicate creation under concurrent requests. App names starting with `"__"` (built-in agents) route to a shared `agents_root/.adk/` location instead of the per-agent tree. `PerAgentFileArtifactService` falls back to a **read-only** legacy shared store (`agents_root/.adk/artifacts/`, pre-per-agent layout) for `load_artifact`/`list_artifact_keys`/`list_versions`; `save_artifact` never writes there, and `delete_artifact` deletes from both stores so a deleted artifact can't reappear via the fallback.

```python
from google.adk.cli.utils.local_storage import create_local_session_service
session_service = create_local_session_service(base_dir=agents_root, per_agent=True)
```

#### `DotAdkFolder` + `dot_adk_folder_for_agent`
**Source:** `google.adk.cli.utils.dot_adk_folder`

Path-safe accessor for the `.adk` sub-folder inside an agent's working directory.

```python
class DotAdkFolder:
    dot_adk_dir: Path        # cached_property: agent_dir / ".adk" — NOT auto-created
    artifacts_dir: Path      # cached_property: dot_adk_dir / "artifacts"
    session_db_path: Path    # cached_property: dot_adk_dir / "session.db"
```

`dot_adk_folder_for_agent(agents_root, app_name)` resolves both `agents_root` and `agents_root / app_name` via `Path.resolve()` and asserts the latter `is_relative_to` the former — blocking path-traversal via `app_name="../../../etc"` (raises `ValueError`).

#### `inject_session_state` + `_is_valid_state_name`
**Source:** `google.adk.utils.instructions_utils`

Populates instruction templates at call time from live session state and artifact content — designed for `InstructionProvider` callables.

| Placeholder | Source | Missing behavior |
|---|---|---|
| `{var_name}` | `session.state[var_name]` | raises `KeyError` |
| `{var_name?}` | `session.state[var_name]` | replaced with `''` |
| `{artifact.file_name}` | `artifact_service.load_artifact(filename)` | raises `KeyError` |
| `{artifact.file_name?}` | `artifact_service.load_artifact(filename)` | replaced with `''` |

`_is_valid_state_name` accepts a bare identifier or a `app:`/`user:`/`temp:`-prefixed identifier; anything else (`"2bad"`, `"app:"`) is left **unchanged** in the template rather than raising.

```python
async def personalised_instruction(ctx: ReadonlyContext) -> str:
    return await inject_session_state(
        "You are assisting {user:name}. Optional tone: {user:preferred_tone?}.", ctx)
```


### Callbacks, Plugins & Middleware

#### BasePlugin
Why it matters: the plugin lifecycle hook interface. **Correction**: 2.7.1 exposes **14** hook methods, more than any individual source doc describes (most cover 12–13): `before_run_callback`, `after_run_callback`, `before_agent_callback`, `after_agent_callback`, `before_model_callback`, `after_model_callback`, `before_tool_callback`, `after_tool_callback`, `on_event_callback`, `on_user_message_callback`, `on_tool_error_callback`, `on_model_error_callback`, and two undocumented-in-every-volume additions: `on_agent_error_callback` and `on_run_error_callback` — dedicated error hooks at the agent and top-level run scope, filling a gap the docs' `on_tool_error_callback`/`on_model_error_callback` pair didn't cover.

```python
from google.adk.plugins import BasePlugin

class MyPlugin(BasePlugin):
    async def on_agent_error_callback(self, *, agent, callback_context, error):
        ...  # new in 2.7.1 relative to the source docs
```

#### PluginManager
Why it matters: registers plugins and drives early-exit hook chaining (first plugin to return non-`None` from a hook short-circuits the rest). Unchanged from the docs.

```python
from google.adk.plugins import PluginManager
```

#### ReflectAndRetryToolPlugin / ReflectAndRetryModelPlugin
Why it matters: built-in plugins that catch a tool/model error, feed it back to the LLM as a reflection prompt, and retry. **Note**: `google.adk.plugins.__all__` in 2.7.1 lists `ReflectAndRetryModelPlugin` alongside `ReflectAndRetryToolPlugin` — most source docs only describe the tool-retry variant; the model-retry sibling applies the same reflect-and-retry pattern to raw model-call failures.

```python
from google.adk.plugins import ReflectAndRetryToolPlugin, ReflectAndRetryModelPlugin
```

#### DebugLoggingPlugin / LoggingPlugin
Why it matters: built-in verbose tracing plugins for local development. Both remain exported; `LoggingPlugin` is the lighter-weight sibling of `DebugLoggingPlugin`.

```python
from google.adk.plugins import DebugLoggingPlugin
```

#### `LoggingPlugin`
**Source:** `google.adk.plugins.logging_plugin`

Implements all 12 `BasePlugin` hooks and prints a structured, ANSI-grey, emoji-prefixed trace to stdout — the first thing to add when debugging locally. Every callback returns `None`, so it never short-circuits the pipeline. `_log(msg)` prepends `[{self.name}]`; there is no Python `logging` integration by default (subclass and override `_log` to route elsewhere).

| Hook | Logs |
|---|---|
| `on_user_message_callback` | invocation/session/user/app IDs, root agent, user content, branch |
| `before_run_callback` / `after_run_callback` | invocation start/end, starting agent, final response, errors |
| `before_agent_callback` / `after_agent_callback` | agent name starting/finished |
| `before_model_callback` | model name, agent name, first 200 chars of system instruction, available tool names |
| `after_model_callback` | error code/message, content, `partial`, `turn_complete`, token usage |
| `on_model_error_callback` | the exception |
| `before_tool_callback` / `after_tool_callback` | tool name, agent name, FC id, arguments (and result) |
| `on_tool_error_callback` | the exception |
| `on_event_callback` | event id, author, content, `is_final_response()`, FC/FR names, `long_running_tool_ids` |

```python
from google.adk.plugins.logging_plugin import LoggingPlugin
app = App(name="demo", root_agent=agent, plugins=[LoggingPlugin(name="debug")])
```

To redirect to Python's `logging` instead of stdout, subclass and override `_log`:

```python
import logging
logger = logging.getLogger("adk.trace")

class StructuredLoggingPlugin(LoggingPlugin):
    def _log(self, message: str) -> None:
        logger.debug(message.replace("\033[90m", "").replace("\033[0m", ""))
```

#### `BigQueryAgentAnalyticsPlugin` + `BigQueryLoggerConfig` + `EventData`
**Source:** `google.adk.plugins.bigquery_agent_analytics_plugin`

Streams agent lifecycle events to BigQuery via the **BigQuery Write API**. Handles multiple concurrent asyncio event loops (e.g. multi-threaded servers) by keeping a `dict[asyncio.AbstractEventLoop, _LoopState]`, each with its own `BigQueryWriteAsyncClient`, `BatchProcessor`, and write stream; stale closed loops are cleaned up lazily.

```python
plugin = BigQueryAgentAnalyticsPlugin(
    project_id="my-gcp-project", dataset_id="agent_analytics",
    config=BigQueryLoggerConfig(table_id="agent_events", view_prefix="agent_", max_content_length=8192),
)
app = App(name="analytics_demo", root_agent=agent, plugins=[plugin])
```

```python
@dataclass(kw_only=True)
class EventData:
    latency_ms: Optional[int] = None
    model: Optional[str] = None
    usage_metadata: Any = None
    status: str = "OK"
    source_event: Optional["Event"] = None
    adk_extras: dict[str, Any] = field(default_factory=dict)  # placed INSIDE attributes.adk.* JSON
```

Content longer than `max_content_length` is offloaded to GCS via `GCSOffloader` (a `HybridContentParser` translates between a GCS URI and inline text on read).

#### `EnsureRetryOptionsPlugin` + `add_default_retry_options_if_not_present`
**Source:** `google.adk.evaluation._retry_options_utils`

Injects `types.HttpRetryOptions` into every `LlmRequest` before it's sent, protecting long-running eval batches (or any deployment) from transient rate-limit/gateway errors.

```python
_DEFAULT_HTTP_RETRY_OPTIONS  # attempts=7, initial_delay=5.0, max_delay=120.0, exp_base=2.0
                             # retried codes: 408, 429, 500, 502, 503, 504
```

Backoff: `min(max_delay, initial_delay * exp_base**n)` — with defaults, 5s/10s/20s/40s/80s/120s/120s across 7 attempts. `add_default_retry_options_if_not_present(llm_request)` is the idempotent standalone function (only assigns if `retry_options is None`); `EnsureRetryOptionsPlugin` wraps it as an app-wide `before_model_callback`.

```python
app = App(name="eval_app", root_agent=my_agent, plugins=[EnsureRetryOptionsPlugin()])
```

#### `_RequestIntercepterPlugin`
**Source:** `google.adk.evaluation.request_intercepter_plugin` — internal use only; documented here to show the pattern, not as a stable dependency.

Couples an `LlmRequest` with its `LlmResponse` via a UUID stashed in `custom_metadata`, so eval tooling can look up "what was actually sent" for any response it observes.

```python
class _RequestIntercepterPlugin(BasePlugin):
    async def before_model_callback(self, *, callback_context, llm_request):
        request_id = str(uuid.uuid4())
        self._llm_requests_cache[request_id] = llm_request
        callback_context.state[_LLM_REQUEST_ID_KEY] = request_id
    async def after_model_callback(self, *, callback_context, llm_response):
        llm_response.custom_metadata = llm_response.custom_metadata or {}
        llm_response.custom_metadata[_LLM_REQUEST_ID_KEY] = callback_context.state.get(_LLM_REQUEST_ID_KEY)
    def get_model_request(self, llm_response) -> Optional[LlmRequest]:
        rid = (llm_response.custom_metadata or {}).get(_LLM_REQUEST_ID_KEY)
        return self._llm_requests_cache.get(rid)
```

Note `Event` **is** an `LlmResponse` subclass — pass the event itself to `get_model_request`, not `event.llm_response`.

#### `FeatureName` + `FeatureConfig` + `temporary_feature_override` + `@experimental`/`@stable`/`@working_in_progress`
**Source:** `google.adk.features._feature_registry`, `._feature_decorator`

ADK's feature-flag system, gating experimental APIs across the whole codebase.

```python
class FeatureStage(Enum):
    WIP = "wip"; EXPERIMENTAL = "experimental"; STABLE = "stable"

@dataclass
class FeatureConfig:
    stage: FeatureStage
    default_on: bool = False
```

Priority for `is_feature_enabled(name)`: **(1)** programmatic `override_feature_enabled(name, bool)` **(2)** env vars `ADK_ENABLE_<NAME>` / `ADK_DISABLE_<NAME>` **(3)** the registry default. `temporary_feature_override(name, enabled)` is a context manager that saves/restores the override — safe for sequential or nested single-threaded use, but `_FEATURE_OVERRIDES` is a plain unlocked `dict`, so two concurrent tasks toggling the *same* flag can restore each other's values out of order.

The `@experimental`/`@stable`/`@working_in_progress` decorators register a `FeatureConfig` on first use and wrap `__init__`/the function to call `check_feature_enabled()` at invocation time, raising `RuntimeError` if disabled. `@experimental` sets `default_on=False`; `@stable` sets `default_on=True`; `@working_in_progress` always starts disabled.

```python
from google.adk.features import FeatureName, override_feature_enabled, temporary_feature_override

override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, True)   # permanent

with temporary_feature_override(FeatureName.PUBSUB_TOOLSET, True):      # scoped, e.g. in a test
    toolset = PubSubToolset(...)
```

```bash
ADK_ENABLE_PROGRESSIVE_SSE_STREAMING=1 python my_agent.py    # env-var toggle, no code change
```

### Evaluation

#### AgentEvaluator
Why it matters: the top-level entry point for running an eval set against an agent and asserting on pass/fail criteria. Unchanged at `google.adk.evaluation.agent_evaluator.AgentEvaluator`, re-exported from the package root.

```python
from google.adk.evaluation import AgentEvaluator
await AgentEvaluator.evaluate(agent_module="my_agent", eval_dataset_file_path_or_dir="evals/")
```

#### TrajectoryEvaluator
Why it matters: scores whether an agent's tool-call sequence matches an expected trajectory. **Note**: not re-exported at `google.adk.evaluation` package level in 2.7.1 (`from google.adk.evaluation import TrajectoryEvaluator` fails) — import from its submodule.

```python
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
```

#### EvalConfig / CustomMetricConfig
Why it matters: declares which metrics run against an eval set and lets you register `CustomMetricConfig` instances alongside built-ins. Field set confirmed: `criteria`, `custom_metrics`, `user_simulator_config`, `live_model_config` — matches the source docs.

```python
from google.adk.evaluation.eval_config import EvalConfig
```

#### `EvalCase` + `Invocation` + `IntermediateData` + `EvalSet`
**Source:** `google.adk.evaluation.eval_case`, `.eval_set`

The fundamental data model. `EvalSet` groups `EvalCase`s under an ID; each `EvalCase` is one or more conversation turns (`Invocation`s) with expected responses/rubrics.

```
EvalBaseModel
├── IntermediateData    — tool_uses, tool_responses, intermediate_responses
├── InvocationEvent      — author, content
├── InvocationEvents     — list[InvocationEvent]
├── Invocation           — user_content, final_response, intermediate_data
├── SessionInput         — app_name, user_id, state
└── EvalCase             — eval_id, conversation, session_input, rubric
```

```python
class Invocation(EvalBaseModel):
    invocation_id: str = ""
    user_content: genai_types.Content
    final_response: Optional[genai_types.Content] = None
    intermediate_data: Optional[Union[IntermediateData, InvocationEvents]] = None
    creation_timestamp: float = 0.0

class IntermediateData(EvalBaseModel):
    tool_uses: list[genai_types.FunctionCall] = []
    tool_responses: list[genai_types.FunctionResponse] = []
    intermediate_responses: list[tuple[str, list[genai_types.Part]]] = []  # (author_name, parts)

class EvalSet(BaseModel):
    eval_set_id: str
    eval_cases: list[EvalCase]
    creation_timestamp: float = 0.0
```

`EvalCase` enforces an XOR between `conversation=` (fixed turns) and `conversation_scenario=` (a generated multi-turn plan, see `ConversationScenario` below) — set one, not both.

```python
case = EvalCase(eval_id="order_query_001", conversation=[Invocation(
    user_content=types.Content(role="user", parts=[types.Part(text="Order #1234 status?")]),
    final_response=types.Content(role="model", parts=[types.Part(text="Shipped, arrives tomorrow.")]),
    intermediate_data=IntermediateData(
        tool_uses=[types.FunctionCall(name="get_order_status", args={"order_id": "1234"})],
    ),
)])
```

#### `InMemoryEvalSetsManager`
**Source:** `google.adk.evaluation.in_memory_eval_sets_manager`

Implements the full `EvalSetsManager` interface over nested dicts — the simplest way to run eval pipelines in tests without a DB/GCS bucket. Maintains `_eval_sets` and `_eval_cases` both keyed by `app_name` first (multi-app isolation); `_ensure_app_exists` avoids `KeyError` without a separate `create_app` call. `create_eval_set`/`add_eval_case` raise `ValueError` on a duplicate ID; missing lookups raise `NotFoundError` (`google.adk.errors.not_found_error`) or return `None`/`[]` depending on the method. No locking — fine for single-threaded test runners.

```python
manager = InMemoryEvalSetsManager()
manager.create_eval_set(app_name="search_agent", eval_set_id="baseline_v1")
manager.add_eval_case("search_agent", "baseline_v1", case)
```

#### `EvalMetric` + `BaseCriterion` + `JudgeModelOptions` + `PrebuiltMetrics`
**Source:** `google.adk.evaluation.eval_metrics`

```python
class EvalStatus(Enum):
    PASSED = 1; FAILED = 2; NOT_EVALUATED = 3

class JudgeModelOptions(EvalBaseModel):
    judge_model: str = "gemini-2.5-flash"
    judge_model_config: Optional[genai_types.GenerateContentConfig] = None
    num_samples: int = 5

class BaseCriterion(BaseModel):        # camelCase JSON aliases (alias_generator=to_camel)
    threshold: float
    include_intermediate_responses_in_final: bool = False

class LlmAsAJudgeCriterion(BaseCriterion):
    judge_model_options: JudgeModelOptions = JudgeModelOptions()

class RubricsBasedCriterion(BaseCriterion):
    judge_model_options: JudgeModelOptions = JudgeModelOptions()
    rubrics: list[Rubric] = []

class EvalMetric(EvalBaseModel):
    metric_name: str
    threshold: float
    criterion: Optional[BaseCriterion] = None
    custom_function_path: Optional[str] = None
```

`PrebuiltMetrics` (values used as `EvalConfig.criteria` keys or `metric_name`): `TOOL_TRAJECTORY_AVG_SCORE`, `RESPONSE_EVALUATION_SCORE`, `RESPONSE_MATCH_SCORE`, `SAFETY_V1`, `FINAL_RESPONSE_MATCH_V2`, `RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1`, `HALLUCINATIONS_V1`, `RUBRIC_BASED_TOOL_USE_QUALITY_V1`, `PER_TURN_USER_SIMULATOR_QUALITY_V1`, `MULTI_TURN_TASK_SUCCESS_V1`, `MULTI_TURN_TRAJECTORY_QUALITY_V1`, `MULTI_TURN_TOOL_USE_QUALITY_V1`, `RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1`.

```python
metric = EvalMetric(metric_name=PrebuiltMetrics.RESPONSE_MATCH_SCORE.value, threshold=0.5)  # ROUGE-1, no judge model
```

#### RougeEvaluator / FinalResponseMatchV2Evaluator / HallucinationsV1Evaluator
Why it matters: the built-in response-quality metric family — ROUGE-based final-response match (v1), an improved semantic v2 matcher, and a hallucination detector. All remain at their documented submodule paths.

```python
from google.adk.evaluation.final_response_match_v1 import RougeEvaluator
from google.adk.evaluation.final_response_match_v2 import FinalResponseMatchV2Evaluator
from google.adk.evaluation.hallucinations_v1 import HallucinationsV1Evaluator
```

#### `ResponseEvaluator`
**Source:** `google.adk.evaluation.response_evaluator`

Concrete `Evaluator` supporting two built-in metrics: **coherence** (1–5, via Vertex AI `PrebuiltMetric.COHERENCE`) and **ROUGE-1 match** (0–1, against a golden response).

```python
class ResponseEvaluator(Evaluator):
    def __init__(self, threshold=None, metric_name=None, eval_metric: EvalMetric | None = None): ...
```

Pass either `eval_metric` **or** both `threshold`+`metric_name` — mixing raises `ValueError`. Valid `metric_name`: `"response_evaluation_score"` (coherence) or `"response_match_score"` (ROUGE-1).

#### Rubric-based evaluators (RubricBasedFinalResponseQualityV1Evaluator, RubricBasedToolUseV1Evaluator, RubricBasedMultiTurnTrajectoryEvaluator)
Why it matters: LLM-as-judge evaluators scored against explicit rubrics rather than reference outputs — useful when there's no single "correct" trajectory. All confirmed present at their documented paths under `google.adk.evaluation`.

```python
from google.adk.evaluation.rubric_based_final_response_quality_v1 import RubricBasedFinalResponseQualityV1Evaluator
```

#### `SafetyEvaluatorV1` + `MultiTurn*V1Evaluator`s + `RubricBasedFinalResponseQualityV1Evaluator`
**Source:** `google.adk.evaluation.safety_evaluator`, `.multi_turn_task_success_evaluator`, `.multi_turn_tool_use_quality_evaluator`, `.multi_turn_trajectory_quality_evaluator`, `.rubric_based_final_response_quality_v1`

All reference-free, score `[0, 1]`, require `GOOGLE_CLOUD_PROJECT`+`GOOGLE_CLOUD_LOCATION` or `GOOGLE_API_KEY`, and delegate to the Vertex Gen AI Eval SDK via `_VertexAiEvalFacade` (below).

| Class | Backing metric | Measures |
|---|---|---|
| `SafetyEvaluatorV1` | `PrebuiltMetric.SAFETY` (single-turn) | harmlessness of each response independently |
| `MultiTurnTaskSuccessV1Evaluator` | `RubricMetric.MULTI_TURN_TASK_SUCCESS` | did the agent ultimately achieve the user's goal |
| `MultiTurnToolUseQualityV1Evaluator` | `RubricMetric.MULTI_TURN_TOOL_USE_QUALITY` | were function calls made correctly/in order |
| `MultiTurnTrajectoryQualityV1Evaluator` | `RubricMetric.MULTI_TURN_TRAJECTORY_QUALITY` | was the path to the goal sensible and efficient |
| `RubricBasedFinalResponseQualityV1Evaluator` | LLM-judge with a 10240-token silent-thinking budget | user-defined rubrics against the final response, `yes`/`no` → `1.0`/`0.0` |

The `V1` suffix is a versioning convention — future alternative strategies may ship as `V2`, etc. All multi-turn evaluators mark every turn but the **last** `NOT_EVALUATED`; only the final turn in a conversation gets a numeric score.

```python
metric = EvalMetric(metric_name=PrebuiltMetrics.SAFETY_V1.value, threshold=0.7)
evaluator = SafetyEvaluatorV1(eval_metric=metric)
result = evaluator.evaluate_invocations(actual_invocations=invocations)
```

#### `_VertexAiEvalFacade` hierarchy
**Source:** `google.adk.evaluation.vertex_ai_eval_facade`

The private bridge every Vertex-backed evaluator above shares.

- **`__init__`** — reads `GOOGLE_CLOUD_PROJECT`/`LOCATION`/`GOOGLE_API_KEY` from env; API key takes priority; otherwise both project and location must be set or `ValueError` is raised with remediation instructions.
- **`_perform_eval(dataset, metrics)`** — the only method that actually calls `client.evals.evaluate(...)`; isolated for unit-test patching.
- **`_get_score(eval_result)`** — reads `summary_metrics[0].mean_score`; returns `None` on empty results, a non-float, or NaN — preventing NaN from propagating into `EvaluationResult.overall_score`.
- **`_SingleTurnVertexAiEvalFacade`** — one `evals.evaluate()` call per `Invocation`, built as a single-row `pandas.DataFrame({"prompt", "reference", "response"})`.
- **`_MultiTurnVertexiAiEvalFacade`** — one `evals.evaluate()` call for the *entire conversation*, assembled as a Vertex `EvalCase(agent_data=AgentData(agents={...}, turns=[ConversationTurn, ...]))`. Marks turns `0..N-2` as `NOT_EVALUATED`; the final turn carries the score. Discards its `conversation_scenario` argument immediately (`del conversation_scenario`) — it has **no effect** on the judge input despite appearing in the signature.

#### `AppDetails` + `AgentDetails`
**Source:** `google.adk.evaluation.app_details`

A lightweight snapshot of the agent hierarchy (names, instructions, tool declarations) that eval backends and judge prompts reference without a live runner.

```python
class AgentDetails(BaseModel):
    name: str; instructions: str; tool_declarations: list[...] = []

class AppDetails(BaseModel):
    agent_details: dict[str, AgentDetails]
    def get_developer_instructions(self, agent_name: str) -> str: ...
    def get_tools_by_agent_name(self) -> dict[str, list]: ...
```

#### GcsEvalSetsManager / LocalEvalSetsManager
Why it matters: persistence backends for eval sets (and their `*Results` counterparts) — local filesystem vs. GCS-backed storage for CI pipelines. Import paths unchanged from the docs.

```python
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
```

#### `ConversationScenario` + `ConversationGenerationConfig` + `ScenarioGenerator`
**Source:** `google.adk.evaluation.conversation_scenarios`, `._vertex_ai_scenario_generation_facade`

AI-assisted eval-set generation: describe an agent and let Gemini generate realistic multi-turn `ConversationScenario`s.

```python
class ConversationScenario(BaseModel):
    starting_prompt: str
    conversation_plan: str
    user_persona: UserPersona | str | None = None   # a string resolves against the persona registry

class ConversationGenerationConfig(BaseModel):
    count: int
    model_name: str
    generation_instruction: str | None = None
    environment_context: str | None = None          # seeds ground-truth backend state into the prompt

class ScenarioGenerator:
    def __init__(self) -> None: ...   # same auth rules as _VertexAiEvalFacade
    def generate_scenarios(self, agent: BaseAgent, config: ConversationGenerationConfig) -> list[ConversationScenario]: ...
        # synchronous — no await; uses AgentInfo.load_from_agent(agent) for context
```

Feed generated scenarios into `EvalCase(eval_id=..., conversation_scenario=s)` (not `conversation=`) and evaluate with a reference-free multi-turn metric (`response_match_score` needs a golden response scenarios don't provide).

```python
generator = ScenarioGenerator()
scenarios = generator.generate_scenarios(
    agent=travel_agent,
    config=ConversationGenerationConfig(count=5, model_name="gemini-2.5-flash",
                                         environment_context="Flights SFO→LAX 08:00 ($129)."),
)
eval_set = EvalSet(eval_set_id="generated", eval_cases=[
    EvalCase(eval_id=f"g{i}", conversation_scenario=s) for i, s in enumerate(scenarios)
])
```

#### `UserSimulator` + `NextUserMessage` + `Status` + `BaseUserSimulatorConfig`
**Source:** `google.adk.evaluation.simulation.user_simulator` — `@experimental`

Abstract base for automated multi-turn evaluation: implementations receive the conversation history and return the next simulated user turn.

```python
class Status(enum.Enum):
    SUCCESS = "success"; TURN_LIMIT_REACHED = "turn_limit_reached"
    STOP_SIGNAL_DETECTED = "stop_signal_detected"; NO_MESSAGE_GENERATED = "no_message_generated"

class NextUserMessage(EvalBaseModel):
    status: Status
    user_message: Optional[genai_types.Content] = None
    # @model_validator: user_message is non-None IFF status == SUCCESS

class UserSimulator(ABC):
    def __init__(self, config, config_type):
        self._config = config_type.model_validate(config.model_dump())
    async def get_next_user_message(self, events: list[Event]) -> NextUserMessage: ...
    def get_simulation_evaluator(self) -> Optional[Evaluator]: ...
```

#### `StaticUserSimulator` + `LlmBackedUserSimulator`
**Source:** `google.adk.evaluation.simulation.static_user_simulator`, `.llm_backed_user_simulator`

`StaticUserSimulator(static_conversation: StaticConversation)` plays back a pre-recorded `list[Invocation]` — no LLM, `get_simulation_evaluator()` always `None`. `events` is accepted but ignored; the index (`invocation_idx`) advances every call regardless of what the agent said, and returns `STOP_SIGNAL_DETECTED` once exhausted.

`LlmBackedUserSimulator` drives a real LLM (default `gemini-2.5-flash`, thinking enabled) guided by a `ConversationScenario`, stopping on a `</finished>` signal in the model's text.

```python
class LlmBackedUserSimulatorConfig(BaseUserSimulatorConfig):
    model: str = "gemini-2.5-flash"
    max_allowed_invocations: int = 20     # runaway guard; -1 = unlimited (not recommended)
    custom_instructions: str | None = None
    # field_validator requires {{ stop_signal }}, {{ conversation_plan }}, AND
    # {{ conversation_history }} all present in custom_instructions if set
```

#### LlmBackedUserSimulator / StaticUserSimulator / UserPersona
Why it matters: drives multi-turn conversation simulation for evals — either an LLM improvising as a persona-constrained user, or a fixed scripted transcript. Field sets and import paths unchanged.

```python
from google.adk.evaluation.llm_backed_user_simulator import LlmBackedUserSimulator
from google.adk.evaluation.static_user_simulator import StaticUserSimulator
```

#### `AgentOptimizer[T, U]` + `Sampler[T]` + `OptimizerResult`
**Source:** `google.adk.optimization.agent_optimizer`, `.sampler`, `.data_types` (verified 2.7.1)

```python
class Sampler(ABC, Generic[SamplingResultT]):
    TRAIN_SET: ClassVar[Literal["train"]] = "train"
    VALIDATION_SET: ClassVar[Literal["validation"]] = "validation"
    @abstractmethod
    def get_train_example_ids(self) -> list[str]: ...
    @abstractmethod
    def get_validation_example_ids(self) -> list[str]: ...
    @abstractmethod
    async def sample_and_score(self, candidate: Agent, example_set="validation",
                                batch: list[str] | None = None,
                                capture_full_eval_data: bool = False) -> SamplingResultT: ...

class AgentOptimizer(ABC, Generic[SamplingResultT, AgentWithScoresT]):
    @abstractmethod
    async def optimize(self, initial_agent: Agent, sampler: Sampler[SamplingResultT]) -> OptimizerResult[AgentWithScoresT]: ...

class SamplingResult(BaseModel):
    scores: dict[str, float]                        # uid → score, higher is better
class UnstructuredSamplingResult(SamplingResult):
    data: Optional[dict[str, dict[str, Any]]] = None # uid → raw eval artefacts, for reflection-based optimizers
class AgentWithScores(BaseModel):
    optimized_agent: Agent
    overall_score: Optional[float] = None
class OptimizerResult(BaseModel, Generic[AgentWithScoresT]):
    optimized_agents: list[AgentWithScoresT]         # Pareto front — not strictly ordered
```

`capture_full_eval_data=True` signals the sampler must also return trajectories/tool-calls (needed by reflection-based optimizers like GEPA); with `False` only `scores` is required.

#### `LocalEvalSampler` + `LocalEvalSamplerConfig`
**Source:** `google.adk.optimization.local_eval_sampler`

The `Sampler` implementation for running evaluation **locally** (no Vertex AI) during prompt optimization, backed by `LocalEvalService`.

```python
class LocalEvalSamplerConfig(BaseModel):
    eval_config: EvalConfig
    app_name: str
    train_eval_set: str
    train_eval_case_ids: list[str] | None = None
    validation_eval_set: str | None = None            # defaults to the train set
    validation_eval_case_ids: list[str] | None = None
```

#### `SimplePromptOptimizer`
**Source:** `google.adk.optimization.simple_prompt_optimizer`

Gradient-free, LLM-driven iterative prompt improver: sample a mini-batch → score the current best → feed score+prompt into `_OPTIMIZER_PROMPT_TEMPLATE` → generate a candidate → score it → keep if better.

```python
class SimplePromptOptimizerConfig(BaseModel):
    optimizer_model: str = "gemini-2.5-flash"
    model_configuration: GenerateContentConfig = GenerateContentConfig(
        thinking_config=ThinkingConfig(include_thoughts=True, thinking_budget=10240))
    num_iterations: int = 10
    batch_size: int = 5
```

The optimizer LLM is instructed to output only the improved prompt text — no markdown — so the raw response becomes the new `instruction` directly.

#### `GEPARootAgentOptimizer` (+ `GEPARootAgentPromptOptimizer`)
**Source:** `google.adk.optimization.gepa_root_agent_optimizer`, `.gepa_root_agent_prompt_optimizer` — requires `pip install gepa` separately (`@experimental`)

Both modules exist in 2.7.1 — `gepa_root_agent_optimizer.py` is the current, broader optimizer (treats both `agent_prompt` and any `SkillToolset` skill instructions as separate optimization targets); `gepa_root_agent_prompt_optimizer.py` is an older, prompt-only variant kept for compatibility. Prefer `GEPARootAgentOptimizer`.

```python
class GEPARootAgentOptimizerConfig(BaseModel):
    optimizer_model: str = "gemini-3.5-flash"
    model_configuration: GenerateContentConfig = GenerateContentConfig(
        thinking_config=ThinkingConfig(include_thoughts=True, thinking_level=ThinkingLevel.HIGH))
    max_metric_calls: int = 100
    reflection_minibatch_size: int = 3
    run_dir: str | None = None      # enables checkpoint-based resume

class GEPARootAgentOptimizerResult(OptimizerResult[AgentWithScores]):
    gepa_result: dict[str, Any] | None = None   # raw GEPA library result (Pareto-front history, etc.)
```

An `_AgentGEPAAdapter` is created dynamically inside `_create_agent_gepa_adapter_class()` so `gepa` is imported only when actually used — zero startup cost for users without the extra installed. Each GEPA evaluation call bridges to ADK's async `sampler.sample_and_score` via `asyncio.run_coroutine_threadsafe` from a thread-pool.

```python
optimizer = GEPARootAgentOptimizer(config=GEPARootAgentOptimizerConfig(max_metric_calls=30, run_dir="/tmp/gepa-run"))
result = await optimizer.optimize(agent, MyEvalSampler())
```

#### Environment simulation for eval — see Tools & Toolsets
`EnvironmentSimulationFactory` / `EnvironmentSimulationEngine` / `ToolSpecMockStrategy` (documented under Tools & Toolsets) are most often used from evaluation harnesses to run an agent deterministically against mocked tool responses instead of live services — see that section for the full API.

### A2A Protocol

#### RemoteA2aAgent
Why it matters: the client-side agent that proxies to a remote A2A-protocol server as if it were a local sub-agent. **Correction (significant)**: `RemoteA2aAgent` is **not** re-exported from `google.adk.agents` in 2.7.1 (`from google.adk.agents import RemoteA2aAgent` raises `ImportError`, contradicting every source doc that shows this exact import). It must be imported from its private submodule, and doing so additionally requires the optional `a2a` SDK package (`pip install google-adk[a2a]`) which is not installed by default.

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
# requires: pip install "google-adk[a2a]"
```

#### `A2aAgentExecutor` + `A2aAgentExecutorConfig` + `ExecuteInterceptor`
**Source:** `google.adk.a2a.executor.a2a_agent_executor`, `.config` — both `@a2a_experimental`

Adapts any ADK `Runner` to the `a2a.server.agent_execution.AgentExecutor` interface so it can be hosted in an [A2A](https://github.com/google/a2a) server.

```python
class A2aAgentExecutor(AgentExecutor):
    def __init__(self, *, runner: Runner | Callable[..., Runner | Awaitable[Runner]],
                 config: A2aAgentExecutorConfig | None = None,
                 use_legacy: bool = False, force_new_version: bool = False): ...
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None: ...
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None: ...
```

`runner` may be a live `Runner` **or** a zero/one-arg callable (sync or async) returning one — resolved and cached on first `execute()` call, useful for deferred/async initialization. Execution flow: resolve runner → convert `RequestContext` → `AgentRunRequest` via `config.request_converter` → ensure the session exists → publish `working` status → drain `runner.run_async()`, converting each ADK event via `config.event_converter` → publish the final `completed`/`failed`/`input_required` status.

```python
class A2aAgentExecutorConfig(BaseModel):
    a2a_part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part
    gen_ai_part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part
    request_converter: A2ARequestToAgentRunRequestConverter = convert_a2a_request_to_agent_run_request
    event_converter: AdkEventToA2AEventsConverter = legacy_convert_event_to_a2a_events
    adk_event_converter: AdkEventToA2AEventsConverterImpl = convert_event_to_a2a_events_impl
    execute_interceptors: Optional[list[ExecuteInterceptor]] = None

@dataclasses.dataclass
class ExecuteInterceptor:
    before_agent: Optional[Callable[[RequestContext], Awaitable[RequestContext]]] = None
    after_event: Optional[Callable[[ExecutorContext, A2AEvent, Event],
                                    Awaitable[Union[A2AEvent, list[A2AEvent], None]]]] = None
    after_agent: Optional[Callable[[ExecutorContext, TaskStatusUpdateEvent],
                                    Awaitable[TaskStatusUpdateEvent]]] = None
```

`ExecuteInterceptor` is a plain **dataclass** of optional hooks, not a base class to subclass. `before_agent` runs before `ExecutorContext` exists (receives the raw `RequestContext`); `after_event`/`after_agent` receive the fully-resolved `ExecutorContext`. Returning `None` from `after_event` drops that event from the A2A stream entirely.

```python
async def filter_internal_events(executor_ctx, a2a_event, adk_event):
    if adk_event.get_function_calls() or adk_event.get_function_responses():
        return None   # hide tool-call/response events from the A2A client
    return a2a_event

config = A2aAgentExecutorConfig(execute_interceptors=[ExecuteInterceptor(after_event=filter_internal_events)])
executor = A2aAgentExecutor(runner=my_runner, config=config)
```

**Import path note:** every `__init__.py` in the entire `google.adk.a2a` subpackage tree is empty in 2.7.1 — every A2A class without exception must be imported from its full private submodule path (as shown above), never from a package root.

#### to_a2a()
Why it matters: one-call convenience factory that wraps an `LlmAgent` into a ready-to-run Starlette A2A server, skipping manual `A2aAgentExecutor` wiring for the common case.

```python
from google.adk.a2a.utils.agent_to_a2a import to_a2a
app = to_a2a(agent=my_agent, port=8080)
```

#### AgentCardBuilder
Why it matters: constructs the A2A "agent card" (capability/skill manifest) a server advertises to clients. Same private-submodule-only import rule applies.

```python
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
```

#### TaskResultAggregator
Why it matters: reduces a stream of ADK events into the 5-state A2A task lifecycle (submitted/working/input-required/completed/failed), the piece that actually maps ADK's event model onto A2A's task state machine.

```python
from google.adk.a2a.converters.event_converter import TaskResultAggregator
```

#### `ExecutorContext`
**Source:** `google.adk.a2a.executor.executor_context`

A lightweight, immutable context the A2A executor creates once per request and passes to `after_event`/`after_agent` interceptor hooks (not `before_agent`, which only has the raw `RequestContext`).

```python
class ExecutorContext:
    def __init__(self, app_name: str, user_id: str, session_id: str, runner: Runner): ...
    @property
    def app_name(self) -> str: ...
    @property
    def user_id(self) -> str: ...
    @property
    def session_id(self) -> str: ...
    @property
    def runner(self) -> Runner: ...
```

#### `A2aRemoteAgentConfig` + `RequestInterceptor` + `ParametersConfig`
**Source:** `google.adk.a2a.agent.config`

The client-side counterpart to `A2aAgentExecutorConfig`: configuration passed to `RemoteA2aAgent`, controlling how A2A messages/tasks/artifacts/parts convert to ADK events and letting you intercept or abort outbound requests.

- Five converter fields (`a2a_message_converter`, `a2a_task_converter`, `a2a_status_update_converter`, `a2a_artifact_update_converter`, `a2a_part_converter`) all default to the standard converters.
- `request_interceptors: list[RequestInterceptor] | None` — hooks run in order before/after each request.
- `RequestInterceptor.before_request` — `(InvocationContext, A2AMessage, ParametersConfig) → (A2AMessage | Event, ParametersConfig)`. Returning an **`Event`** instead of an `A2AMessage` aborts the call and returns that event directly, without contacting the remote agent.
- `RequestInterceptor.after_request` — `(InvocationContext, A2AEvent, Event) → Event | None`; returning `None` suppresses the event.
- `ParametersConfig` carries `request_metadata: dict | None` and `client_call_context: ClientCallContext | None`.
- `__deepcopy__` is custom: callable fields are copied by reference (not deep-copied), everything else is deep-copied — avoids trying to pickle a lambda/function.

```python
async def rate_limit_interceptor(ctx, message, params):
    if quota_exceeded():
        return Event(invocation_id=ctx.invocation_id, author="system",
                      content=types.Content(parts=[types.Part.from_text("Quota exceeded.")])), params
    return message, params

config = A2aRemoteAgentConfig(request_interceptors=[RequestInterceptor(before_request=rate_limit_interceptor)])
```

#### `convert_event_to_a2a_events` + `AdkEventToA2AEventsConverter`
**Source:** `google.adk.a2a.converters.from_adk_event`

Converts ADK `Event`s into A2A protocol events for streaming, tracking artifact lifecycle across streaming chunks.

```python
A2AUpdateEvent = Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]
AdkEventToA2AEventsConverter = Callable[
    [Event, Optional[Dict[str, str]], Optional[str], Optional[str], GenAIPartToA2APartConverter],
    List[A2AUpdateEvent],
]
```

`agents_artifacts: dict[str, str]` is a **stateful** `{agent_name: artifact_id}` dict maintained across calls by the caller. Logic per event: if an artifact for this agent is already tracked, `append = event.partial` and the entry is dropped once `partial` goes `False` (last chunk); otherwise a new `artifact_id` is minted, tracked only if `partial` is still `True` (more chunks coming).

`create_error_status_event(event, task_id=None, context_id=None)` builds a `TaskStatusUpdateEvent(state=TaskState.failed, final=True)` from `event.error_message` (or a default message), with `_add_event_metadata` stamping `invocation_id`, `author`, `event_id`, `branch`, citation/grounding/custom/usage metadata, and `error_code` onto `status.message.metadata` / `artifact.metadata`.

```python
agents_artifacts: dict = {}
events = convert_event_to_a2a_events(event, agents_artifacts, task_id="t1", context_id="c1")
```

#### `AgentRunRequest` + `convert_a2a_request_to_agent_run_request`
**Source:** `google.adk.a2a.converters.request_converter`

The ADK-side model populated from an incoming A2A `RequestContext`, decoupling the wire format from `Runner.run_async()`'s signature.

```python
@a2a_experimental
class AgentRunRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    invocation_id: Optional[str] = None
    new_message: Optional[genai_types.Content] = None
    state_delta: Optional[dict[str, Any]] = None
    run_config: Optional[RunConfig] = None

A2A_METADATA_KEY = 'a2a_metadata'

def convert_a2a_request_to_agent_run_request(request, part_converter=...) -> AgentRunRequest:
    # user_id: request.call_context.user.user_name if present, else f"A2A_USER_{request.context_id}"
    # session_id = request.context_id (A2A context == ADK session)
    # request.metadata is wrapped in run_config.custom_metadata[A2A_METADATA_KEY]
    # raises ValueError if request.message is None
```

`A2ARequestToAgentRunRequestConverter = Callable[[RequestContext, A2APartToGenAIPartConverter], AgentRunRequest]` is the type alias for a custom drop-in replacement (assign to `A2aAgentExecutorConfig.request_converter`).

#### `LongRunningFunctions` + `handle_user_input` (A2A bridge)
**Source:** `google.adk.a2a.converters.long_running_functions`

Tracks function-call/response pairs marked long-running during an A2A task exchange, emitting a `TaskStatusUpdateEvent` when the remote agent pauses awaiting external input.

- `process_event()` deep-copies the event, strips any `FunctionCall` part whose `.id` is in `event.long_running_tool_ids` into an internal buffer (skips partial events); matching `FunctionResponse` parts are likewise stripped and buffered, but the ID is **not** removed from the tracked set — `has_long_running_function_calls()` stays `True` until the caller drives the lifecycle forward.
- `_task_state` starts `input_required`; if the buffered function call's name is `REQUEST_EUC_FUNCTION_CALL_NAME` (end-user-credentials request), it flips to `auth_required` (last call wins).
- `create_long_running_function_call_event(task_id, context_id)` converts the buffered parts back to A2A parts, tags DataParts carrying FC metadata with `A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY`, and wraps them in a `TaskStatusUpdateEvent(final=True)`.
- `handle_user_input(context)` — a guard: if the current task is `input_required`/`auth_required` but the incoming user message contains **no** `DataPart` typed `FUNCTION_RESPONSE`, it re-emits a `TaskStatusUpdateEvent` re-asserting the same state, reminding the client a function response is still expected.

#### `include_artifacts_in_a2a_event_interceptor`
**Source:** `google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event`

A module-level `ExecuteInterceptor` instance whose `after_event` hook translates ADK `artifact_delta` entries into A2A `TaskArtifactUpdateEvent`s, so artifacts saved during an ADK turn surface over the A2A streaming protocol. For each `(filename, version)` in `adk_event.actions.artifact_delta`, it loads the artifact via the runner's `artifact_service`, converts the part, and emits an additional `TaskArtifactUpdateEvent` alongside the original status event — then clears `artifact_delta` to prevent double emission on repeated calls. Returns the original event unchanged (not wrapped in a list) when there's nothing to add.

```python
config = A2aAgentExecutorConfig(execute_interceptors=[include_artifacts_in_a2a_event_interceptor])
executor = A2aAgentExecutor(runner=my_runner, config=config)
```

#### `InteractionsRequestProcessor` + `_find_previous_interaction_state`
**Source:** `google.adk.flows.llm_flows.interactions_processor`

Enables stateful multi-turn conversations over Gemini's **Interactions API**, where turns chain via `previous_interaction_id` instead of resending full history.

```python
def _find_previous_interaction_state(
    events: list[Event], *, agent_name: str, current_branch: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Scans events in reverse for the last interaction_id authored by agent_name within
    current_branch. Returns (interaction_id, environment_id)."""
```

Branch matching: root branch (`current_branch is None`) includes events with a falsy `event.branch`; a named branch includes events matching that branch **or** with a falsy branch. `InteractionsRequestProcessor.run_async()` activates only when the agent's `canonical_model` is a `Gemini` instance with `use_interactions_api=True`; it sets `llm_request.previous_interaction_id` and yields no events.

```python
model = Gemini(model="gemini-2.5-flash", use_interactions_api=True)
agent = LlmAgent(name="stateful_agent", model=model, instruction="Remember context across turns.")
```

### MCP Integration

#### McpToolset / MCPSessionManager
Why it matters: `McpToolset` (see Tools & Toolsets above) delegates the actual protocol handshake to `MCPSessionManager`, which manages the underlying MCP client session lifecycle across stdio/SSE/streamable-HTTP transports. Both require the optional `mcp` package; failing to import them in a bare venv reflects that missing extra, not a code change.

```python
from google.adk.tools.mcp_tool.mcp_session_manager import (
    MCPSessionManager, StdioConnectionParams, SseConnectionParams, StreamableHTTPConnectionParams,
)
```

#### `McpToolset` — `header_provider`, `use_mcp_resources`, `progress_callback`
**Source:** `google.adk.tools.mcp_tool.mcp_toolset`

Three features worth calling out beyond a plain connection:

- **`header_provider`** — `(ReadonlyContext) -> dict[str, str]`, called on every MCP request. Header **precedence differs by code path**: at session creation (`get_tools`/`get_resources`) provider headers merge first and auth headers win on a key collision; on individual tool calls (`_run_async_impl`) auth headers merge first and **provider** headers win. Avoid setting the same header key (e.g. `Authorization`) from both sources. Only works over HTTP-based transports (`SseConnectionParams`/`StreamableHTTPConnectionParams`) — for `StdioConnectionParams`, `MCPSessionManager._merge_headers` returns `None` and provider headers are discarded.
- **`use_mcp_resources`** — when `True`, appends a `LoadMcpResourceTool` so the model can fetch named MCP *resources* (not tools) from the server.
- **`progress_callback`** — accepts a plain `ProgressFnT(progress, total, message)` **or** a factory `(tool_name, callback_context, **kwargs) -> ProgressFnT`, giving per-tool callbacks that can read/write session state via `callback_context`.

```python
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import SseConnectionParams

def tenant_headers(ctx) -> dict[str, str]:
    return {"X-Tenant-ID": ctx.state.get("user:tenant_id", "default")}

toolset = McpToolset(
    connection_params=SseConnectionParams(url="https://mcp.example.com/sse", timeout=10.0),
    header_provider=tenant_headers,
    use_mcp_resources=True,
)
```

#### `McpTool` + `ProgressCallbackFactory`
**Source:** `google.adk.tools.mcp_tool.mcp_tool` (alias `MCPTool`)

The `BaseTool` wrapper `McpToolset.get_tools()` returns for each `mcp.types.Tool` entry — handles JSON Schema → Gemini `FunctionDeclaration` conversion, auth injection via `BaseAuthenticatedTool`, a graceful error boundary on transport crashes, and optional progress reporting.

```python
@runtime_checkable
class ProgressCallbackFactory(Protocol):
    def __call__(self, tool_name: str, *, callback_context: CallbackContext | None = None,
                 **kwargs: Any) -> ProgressFnT | None: ...
```

Return `None` from the factory to suppress progress reporting for a specific (e.g. noisy) tool name.

```python
def selective_progress_factory(tool_name, *, callback_context=None, **kwargs):
    if tool_name in {"slow_export", "batch_process"}:
        return None
    async def on_progress(progress, total, message):
        print(f"[{tool_name}] {progress}/{total}: {message}")
    return on_progress
```

#### `to_mcp_server` — expose any ADK agent as an MCP server
**Source:** `google.adk.tools.mcp_tool._agent_to_mcp` — `@experimental(FeatureName.MCP_AGENT_SERVER)`

The MCP counterpart of `to_a2a`: wraps any `BaseAgent` in a `FastMCP` server so MCP hosts (Claude Code, IDE extensions, any MCP client) can drive it via the standard protocol.

```python
def to_mcp_server(agent: BaseAgent, *, name: Optional[str] = None,
                   instructions: Optional[str] = None, runner: Optional[Runner] = None) -> FastMCP: ...
```

One ADK session is kept per MCP connection, keyed in a `weakref.WeakKeyDictionary` on `ctx.session` — entries drop automatically on connection GC, no explicit cleanup needed. Part→ContentBlock mapping: `part.text` → `TextContent`; `image/*` inline data → `ImageContent`; `audio/*` inline data → `AudioContent`; other inline data → `EmbeddedResource`; function calls/thoughts are skipped. Intermediate (non-final) text is forwarded as MCP **progress notifications** (`ctx.report_progress`) so a host can stream partial output.

```python
from google.adk.tools.mcp_tool._agent_to_mcp import to_mcp_server

server = to_mcp_server(agent, instructions="An AI assistant powered by Gemini.")
server.run(transport="stdio")            # or: server.streamable_http_app() for HTTP hosting
```

#### `McpInstructionProvider`
**Source:** `google.adk.agents.mcp_instruction_provider`

Implements the `InstructionProvider` protocol (`Callable[[ReadonlyContext], str | Awaitable[str]]`) by fetching a **named MCP Prompt** at request time, so a centralized MCP server can own agent instructions and updates propagate without redeploying the agent.

```python
class McpInstructionProvider(InstructionProvider):
    def __init__(self, connection_params: Any, prompt_name: str, errlog: TextIO = sys.stderr): ...
    async def __call__(self, context: ReadonlyContext) -> str: ...
```

Runtime steps: open/reuse an `MCPSessionManager` session → `session.list_prompts()` to discover the prompt's required argument names → pull matching values from `context.state` by key → `session.get_prompt(name, arguments=...)` → concatenate all `text`-type message contents. Raises `ValueError` if the prompt is empty or the server returns no messages — wrap in a fallback provider for resilience.

```python
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from mcp import StdioServerParameters

provider = McpInstructionProvider(
    connection_params=StdioServerParameters(command="python", args=["-m", "my_mcp_server"]),
    prompt_name="agent_system_prompt",
)
agent = LlmAgent(name="dynamic_agent", model="gemini-2.5-pro", instruction=provider)
```

#### LoadMcpResourceTool
Why it matters: a tool that fetches a named resource from a connected MCP server on demand, for cases where an agent needs to pull a resource mid-conversation rather than at startup.

```python
from google.adk.tools.mcp_tool.load_mcp_resource_tool import LoadMcpResourceTool
```

#### MCP transport parameters — `SseConnectionParams` / `StreamableHTTPConnectionParams` / `StdioConnectionParams`
**Source:** `google.adk.tools.mcp_tool.mcp_session_manager`

```python
class StdioConnectionParams(BaseModel):
    server_params: StdioServerParameters   # from mcp.StdioServerParameters
    timeout: float = 5.0

class SseConnectionParams(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0
    httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client

class StreamableHTTPConnectionParams(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0
    terminate_on_close: bool = True
    httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client
```

`CheckableMcpHttpClientFactory` is a `@runtime_checkable Protocol`; inject a custom factory for a private CA, proxy, or connection pool. See MCP Integration for `McpToolset`/`McpTool` usage of these.


### IAM, Security & Auth

#### `AuthCredential` + `AuthCredentialTypes` + `HttpAuth` + `ServiceAccount`
**Source:** `google.adk.auth.auth_credential` (verified 2.7.1)

The unified data class for every credential shape ADK understands. Field aliases are camelCase (`BaseModelWithConfig` uses `alias_generator=to_camel`); the actual `__init__` signature exposes the camelCase names directly (e.g. `authType=`, `apiKey=`) alongside Python-side snake_case attribute access.

```python
class AuthCredentialTypes(str, Enum):
    API_KEY = "apiKey"; HTTP = "http"; OAUTH2 = "oauth2"
    OPEN_ID_CONNECT = "openIdConnect"; SERVICE_ACCOUNT = "serviceAccount"

class OAuth2Auth(BaseModelWithConfig):
    client_id: str | None = None; client_secret: str | None = None
    auth_uri: str | None = None; nonce: str | None = None; state: str | None = None
    redirect_uri: str | None = None; auth_response_uri: str | None = None; auth_code: str | None = None
    access_token: str | None = None; refresh_token: str | None = None; id_token: str | None = None
    expires_at: int | None = None; expires_in: int | None = None; audience: str | None = None
    code_verifier: str | None = None; code_challenge_method: Literal["S256"] | None = None   # PKCE

class ServiceAccount(BaseModelWithConfig):
    service_account_credential: ServiceAccountCredential | None = None
    scopes: list[str] | None = None
    use_default_credential: bool | None = False
    use_id_token: bool | None = False        # for Cloud Run / Cloud Functions
    audience: str | None = None              # required when use_id_token=True

class AuthCredential(BaseModelWithConfig):
    auth_type: AuthCredentialTypes
    resource_ref: str | None = None          # future: Secret Manager reference
    api_key: str | None = None
    http: HttpAuth | None = None
    service_account: ServiceAccount | None = None
    oauth2: OAuth2Auth | None = None
```

OIDC credentials use `auth_type=OPEN_ID_CONNECT` but still populate the same `oauth2: OAuth2Auth` field (only the type tag differs).

```python
sa_cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(use_default_credential=True,
                                    scopes=["https://www.googleapis.com/auth/cloud-platform"]),
)
```

#### AuthConfig / AuthHandler
Why it matters: `AuthConfig` declares what auth scheme + credential a tool needs; `AuthHandler` drives the actual exchange/refresh flow at request time. `AuthConfig` lives at `google.adk.auth.auth_tool.AuthConfig` (re-exported from `google.adk.auth`); `AuthHandler` is submodule-only.

```python
from google.adk.auth import AuthConfig
from google.adk.auth.auth_handler import AuthHandler
```

#### `AuthHandler`
**Source:** `google.adk.auth.auth_handler`

Drives the three-phase OAuth flow: build a redirect URI, store the credential once the browser flow completes, exchange the code for a token. Used internally by `_AuthLlmRequestProcessor` but usable directly for custom toolsets.

```python
class AuthHandler:
    def __init__(self, auth_config: AuthConfig): ...
    def generate_auth_request(self) -> AuthConfig: ...
    def generate_auth_uri(self) -> AuthCredential: ...    # uses authlib; PKCE verifier auto-generated (48 chars) if absent
    async def parse_and_store_auth_response(self, state: State) -> None: ...
    async def exchange_auth_token(self) -> AuthCredential: ...
    def get_auth_response(self, state: State) -> AuthCredential | None: ...
```

Credentials are stored in session state under `"temp:" + auth_config.credential_key`.

#### `CredentialManager`
**Source:** `google.adk.auth.credential_manager` — `@experimental`

Orchestrates the 8-step credential lifecycle: `0` rehydrate + delegate a `CustomAuthScheme` to its registered `BaseAuthProvider`; `1` validate config; `2` short-circuit if already ready (API_KEY/HTTP); `3` load an existing processed credential from the credential service; `4` else load from an auth response in context; `5` else check client-credentials flow, or return `None` to trigger OAuth consent; `6` exchange (service account → token; OAuth2 code → token); `7` refresh if expired; `8` save if modified.

```python
class CredentialManager:
    def __init__(self, auth_config: AuthConfig): ...
    async def get_auth_credential(self) -> AuthCredential | None: ...
    def register_credential_exchanger(self, auth_type: AuthCredentialTypes, exchanger: BaseCredentialExchanger) -> None: ...
    @classmethod
    def register_auth_provider(cls, provider: BaseAuthProvider) -> None: ...   # class-level, threading.Lock-protected
```

A `CustomAuthScheme` credential raises `ValueError` from `get_auth_credential` unless a matching `BaseAuthProvider` was registered — register providers once at startup.

```python
CredentialManager.register_auth_provider(MyCustomAuthProvider())
```

#### `OAuth2DiscoveryManager` + `AuthorizationServerMetadata` + `ProtectedResourceMetadata`
**Source:** `google.adk.auth.oauth2_discovery`

Implements metadata auto-discovery per **RFC8414** (authorization server metadata) and **RFC9728** (protected resource metadata), used internally by `CredentialManager` when OAuth flow URLs are missing. For a non-root issuer path, three endpoints are tried in order: `/.well-known/oauth-authorization-server{path}` (RFC8414 path-insertion), `/.well-known/openid-configuration{path}` (OIDC path-insertion), `{path}/.well-known/openid-configuration` (OIDC path-appending). A root issuer URL tries only the base forms.

```python
class AuthorizationServerMetadata(BaseModel):
    issuer: str; authorization_endpoint: str; token_endpoint: str
    scopes_supported: list[str] | None = None; registration_endpoint: str | None = None

class ProtectedResourceMetadata(BaseModel):
    resource: str; authorization_servers: list[str] = []
```

#### BaseCredentialExchanger / OAuth2CredentialExchanger / CredentialExchangerRegistry
Why it matters: pluggable credential-exchange layer — converts a raw auth response (e.g. an OAuth2 authorization code) into a usable access credential, with a registry so custom exchangers can be registered per auth scheme.

```python
from google.adk.auth.exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger
```

#### `ServiceAccountCredentialExchanger`
**Source:** `google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger`

Exchanges a service-account credential for either an **access token** (default; OAuth2-scoped Cloud API calls) or an **ID token** (Cloud Run/Functions, verified via `Authorization: Bearer <id_token>`) — controlled by `ServiceAccount.use_id_token`. `use_default_credential=True` skips the JSON key entirely and falls back to Application Default Credentials.

```python
class ServiceAccountCredentialExchanger(BaseAuthCredentialExchanger):
    def exchange_credential(self, auth_scheme: AuthScheme, auth_credential: AuthCredential | None = None) -> AuthCredential: ...
    # use_id_token=True → IDTokenCredentials.from_service_account_info(), populates `audience`
    # use_id_token=False → service_account.Credentials.from_service_account_info(), uses `scopes`
```

#### `AutoAuthCredentialExchanger`
**Source:** `google.adk.tools.openapi_tool.auth.credential_exchangers.auto_auth_credential_exchanger` — `@experimental`

A convenience dispatcher selecting the right exchanger by `auth_credential.auth_type`.

```python
class AutoAuthCredentialExchanger(BaseAuthCredentialExchanger):
    def __init__(self, custom_exchangers: dict[str, type[BaseAuthCredentialExchanger]] | None = None):
        self.exchangers = {
            AuthCredentialTypes.OAUTH2: OAuth2CredentialExchanger,
            AuthCredentialTypes.OPEN_ID_CONNECT: OAuth2CredentialExchanger,
            AuthCredentialTypes.SERVICE_ACCOUNT: ServiceAccountCredentialExchanger,
        }
        if custom_exchangers:
            self.exchangers.update(custom_exchangers)
```

Any other `auth_type` passes the credential through unchanged; `None`/no credential returns `None`. `custom_exchangers` both adds new types and overrides the built-in three.

```python
exchanger = AutoAuthCredentialExchanger(custom_exchangers={AuthCredentialTypes.API_KEY: ApiKeyRefresher})
```

#### `OAuthGrantType` + `ExtendedOAuth2` + `OpenIdConnectWithConfig`
**Source:** `google.adk.auth.auth_schemes`

Extended FastAPI/OpenAPI security-scheme models supporting grant-type introspection and endpoint auto-discovery.

```python
class OAuthGrantType(str, Enum):
    CLIENT_CREDENTIALS = "client_credentials"; AUTHORIZATION_CODE = "authorization_code"
    IMPLICIT = "implicit"; PASSWORD = "password"
    @staticmethod
    def from_flow(flow: OAuthFlows) -> "OAuthGrantType": ...   # inspects which flow field is populated

@experimental
class ExtendedOAuth2(OAuth2):
    issuer_url: str | None = None    # enables endpoint auto-discovery when tokenUrl is empty

class OpenIdConnectWithConfig(SecurityBase):
    authorization_endpoint: str; token_endpoint: str
    userinfo_endpoint: str | None = None; revocation_endpoint: str | None = None
    scopes: list[str] | None = None
```

#### BaseCredentialService / InMemoryCredentialService / SessionStateCredentialService
Why it matters: where exchanged/refreshed credentials are persisted between turns — in-memory for dev, session-state-backed for anything that must survive process restarts without an external store.

```python
from google.adk.auth.credential_service.session_state_credential_service import SessionStateCredentialService
```

#### `ToolContextCredentialStore` + `AuthPreparationResult` + `ToolAuthHandler`
**Source:** `google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler`

Implements the credential lifecycle for `RestApiTool` calls. `ToolContextCredentialStore` persists exchanged credentials in session state keyed by a 16-char SHA-256 prefix of the `(auth_scheme, auth_credential)` pair — with OAuth2 volatile fields (`auth_uri`, `state`, `auth_response_uri`, `auth_code`, `access_token`, `refresh_token`, `expires_at`, `expires_in`) zeroed **before** hashing, so the key is stable regardless of which token is currently active.

```python
class AuthPreparationResult(BaseModel):
    state: Literal["pending", "done"]        # pending → user must complete an OAuth flow
    auth_scheme: AuthScheme | None = None
    auth_credential: AuthCredential | None = None   # may be None even when state=="done" (consumed downstream, or a failed exchange)
```

`prepare_auth_credentials` 5-step pipeline: `1` no scheme → `done` immediately; `2` look up an existing credential, refreshing a stale OAuth2 token if found; `3` if none/still-pending, call `get_auth_response()`; `4` if still none, call `request_credential()` and return `pending`; `5` exchange (SA token / OAuth2 bearer) → `done`.

#### `token_to_scheme_credential` + `openid_dict_to_scheme_credential` + `credential_to_param`
**Source:** `google.adk.tools.openapi_tool.auth.auth_helpers`

Factory functions producing matched `(AuthScheme, AuthCredential)` pairs and converting them back into HTTP request parameters.

| Function | Input | Output |
|---|---|---|
| `token_to_scheme_credential` | `"apikey"` or `"oauth2Token"` + location + value | `(APIKey \| HTTPBearer, AuthCredential)` |
| `openid_dict_to_scheme_credential` | discovery config dict + scopes + credential dict | `(OpenIdConnectWithConfig, AuthCredential)` |
| `openid_url_to_scheme_credential` | OIDC discovery URL + scopes + credential dict | same, auto-fetches config |
| `service_account_dict_to_scheme_credential` | SA JSON dict + scopes | `(HTTPBearer, AuthCredential)` |
| `credential_to_param` | `(AuthScheme, AuthCredential)` | `(ApiParameter, {header_value_dict})` — ready to inject into an outgoing request |

```python
scheme, credential = token_to_scheme_credential("oauth2Token", credential_value="ya29.my-token")
param, headers = credential_to_param(scheme, credential)   # {"Authorization": "Bearer ya29.my-token"}
```

#### AuthenticatedFunctionTool / BaseAuthenticatedTool
Why it matters: the tool-level integration point — wraps a function tool so ADK automatically injects a resolved credential before invocation, pausing the run for user auth if none is cached yet. Constructor signature confirmed unchanged via live introspection.

```python
from google.adk.tools import AuthenticatedFunctionTool
```

#### `_AuthLlmRequestProcessor` + `_store_auth_and_collect_resume_targets`
**Source:** `google.adk.auth.auth_preprocessor`

The resume-side of auth: when a user-authored event contains `adk_request_credential` function responses, this processor stores the credentials and re-executes the tools that originally needed them. Module-level singleton `request_processor = _AuthLlmRequestProcessor()` runs before each LLM call.

`TOOLSET_AUTH_CREDENTIAL_ID_PREFIX = '_adk_toolset_auth_'` marks toolset-level auth (pre-tool-listing) which does **not** map to a resumable function call and is excluded from `tools_to_resume`. `_store_auth_and_collect_resume_targets` scans events for matching FCs, stores credentials via `AuthHandler`, and collects the original FC IDs to re-execute.

#### `MtlsEndpoint` + mTLS utilities
**Source:** `google.adk.utils._mtls_utils`

Centralizes mTLS endpoint selection for Google API service clients (Secret Manager, Parameter Manager, etc.).

```python
class MtlsEndpoint(str, Enum):
    AUTO = "auto"; ALWAYS = "always"; NEVER = "never"

def get_api_endpoint(location: str, default_template: str, mtls_template: str) -> str: ...
    # reads GOOGLE_API_USE_MTLS_ENDPOINT (defaults AUTO) and use_client_cert_effective()
def is_non_mtls_googleapis_endpoint(url: str) -> bool: ...   # host ends .googleapis.com but not .mtls.googleapis.com
def effective_googleapis_endpoint(url: str) -> str: ...      # rewrites to the .mtls. variant
```

#### GcpAuthProvider / AuthProviderRegistry
Why it matters: resolves ambient GCP credentials (ADC, workload identity, etc.) for tools that need to call Google Cloud APIs without an explicit per-user OAuth flow.

```python
from google.adk.auth.providers.gcp_auth_provider import GcpAuthProvider
```

#### `PubSubCredentialsConfig` + `SpannerCredentialsConfig` + `DataAgentCredentialsConfig`
**Source:** `google.adk.tools.pubsub.pubsub_credentials`, `google.adk.tools.spanner.spanner_credentials`, `google.adk.tools.data_agent`

All three extend `BaseGoogleCredentialsConfig` (which enforces mutual exclusivity between `credentials`, `external_access_token_key`, and `client_id`/`client_secret` auth modes) and add a service-specific default scope + session-state token-cache key: Pub/Sub → `("https://www.googleapis.com/auth/pubsub",)` / `"pubsub_token_cache"`; Spanner → `SPANNER_DEFAULT_SCOPE` (admin+data scopes) / `"spanner_token_cache"`; Data Agent → BigQuery scope / `"data_agent_token_cache"`. See the corresponding toolset entries under Tools & Toolsets for constructor usage.

### Caching & Performance

#### ContextCacheConfig / GeminiContextCacheManager
Why it matters: configures Gemini's implicit/explicit context caching (system instruction + tool schema reuse across turns) to cut token cost on long-running agents. Field set confirmed unchanged from the docs.

```python
from google.adk.models import ContextCacheConfig  # re-exported alongside model classes in some builds; confirm via your model module if this import fails, and fall back to google.adk.models.context_cache_config
```

**Verified via live introspection (2.7.1):** `GeminiContextCacheManager` is real and lives at `google.adk.models.gemini_context_cache_manager.GeminiContextCacheManager` (constructor: `__init__(self, genai_client)`; methods `handle_context_caching`, `populate_cache_metadata_in_response`, `cleanup_cache`) — the internal engine `ContextCacheConfig` configures. `ContextCacheConfig` itself is **not** importable from `google.adk.models` in 2.7.1; its confirmed path is `google.adk.agents.context_cache_config.ContextCacheConfig`.

#### `ContextCacheConfig`
**Source:** `google.adk.agents.context_cache_config` (verified 2.7.1)

Attached to `App.context_cache_config` to enable Gemini's [context caching](https://ai.google.dev/gemini-api/docs/caching) for every `LlmAgent` in the app, reusing the stable prefix (system instruction + tool schemas + history) across turns.

```python
@experimental(FeatureName.AGENT_CONFIG)
class ContextCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_intervals: int = Field(default=10, ge=1, le=100)   # reuse the same cache this many invocations, then refresh
    ttl_seconds: int = Field(default=1800, gt=0)              # cache lifetime, default 30 min
    min_tokens: int = Field(default=0, ge=0)                   # gate: only cache if prior prompt >= this many tokens
    create_http_options: types.HttpOptions | None = None       # timeout for CachedContent.create(); on timeout, proceeds uncached

    @property
    def ttl_string(self) -> str:
        return f"{self.ttl_seconds}s"
```

**Verified constraints (from the 2.7.1 docstring, more precise than earlier releases' "4096-token floor" description):** caching begins on the **second turn at the earliest** — the first request has no prior token count to gate on. Gemini enforces its own model-specific minimum regardless of `min_tokens`: **2048 tokens for Gemini 2.5, 4096 tokens for Gemini 3**. Short or single-turn sessions are therefore never cached.

```python
app = App(name="finance_app", root_agent=agent, context_cache_config=ContextCacheConfig(
    cache_intervals=20, ttl_seconds=3600, min_tokens=3000,
    create_http_options=types.HttpOptions(timeout=8000),   # 8s cache-creation timeout, ms
))
```

#### CacheMetadata
Why it matters: a frozen model recording whether a given LLM call actually hit the cache (`active` state) versus merely computed a cache-eligible fingerprint (`fingerprint-only` state), exposing an `expire_soon` property for proactive cache refresh. Field set confirmed identical to the source docs via live `model_fields` introspection.

```python
from google.adk.models.llm_response import CacheMetadata
if response.cache_metadata and response.cache_metadata.expire_soon:
    ...  # proactively refresh the cache before it lapses
```

#### `CachePerformanceAnalyzer`
**Source:** `google.adk.utils.cache_performance_analyzer` — `@experimental`

Reads a session's event history and computes cache-utilization metrics so you can tune `cache_intervals`/`ttl_seconds`/`min_tokens` empirically rather than by guessing.

```python
class CachePerformanceAnalyzer:
    def __init__(self, session_service: BaseSessionService): ...
    async def analyze_agent_cache_performance(
        self, session_id: str, user_id: str, app_name: str, agent_name: str,
    ) -> dict[str, Any]: ...
```

Returns `{"status": "no_cache_data"}` when nothing was recorded, else `{"status": "active", "requests_with_cache", "avg_invocations_used", "latest_cache", "cache_refreshes", "total_invocations", "total_prompt_tokens", "total_cached_tokens", "cache_hit_ratio_percent", "cache_utilization_ratio_percent", "avg_cached_tokens_per_request", "total_requests", "requests_with_cache_hits"}`. `cache_hit_ratio_percent = cached_tokens / prompt_tokens × 100` (a healthy large-system-prompt agent is typically 60–80%); `cache_utilization_ratio_percent = requests_with_cache_hits / total_requests × 100` (drops when the cache expires too often or the session is too short to warm up — the first turn is never cached). `cache_refreshes` counts distinct cache resource names; a high refresh count relative to `requests_with_cache` signals the TTL is too short for the traffic pattern.

```python
analyzer = CachePerformanceAnalyzer(session_service)
stats = await analyzer.analyze_agent_cache_performance(session_id, user_id, app_name, agent_name)
if stats["status"] == "active":
    print(f"Hit ratio: {stats['cache_hit_ratio_percent']:.1f}%")
```

#### EventsCompactionConfig
Why it matters: configures sliding-window + token-threshold compaction of old events to keep long conversations within context limits, with HITL-safety guards so a compaction pass never discards an un-acknowledged human-in-the-loop request. **Correction (repeats across nearly every source volume)**: the standard `from google.adk.apps import App, EventsCompactionConfig, ResumabilityConfig` import now fails — `google.adk.apps.__all__` in 2.7.1 only exports `App` and `ResumabilityConfig`. `EventsCompactionConfig` moved to the private `google.adk.apps._configs` module and is not re-exported.

```python
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps._configs import EventsCompactionConfig  # NOT from google.adk.apps directly

config = EventsCompactionConfig(
    compaction_interval=20, event_retention_size=10, overlap_size=2, token_threshold=8000,
)
```

#### ResumabilityConfig
Why it matters: governs whether an `App`'s runs can be paused and resumed later (e.g. across a process restart, mid-HITL). **Correction**: one source volume (`_v18.md`) claims a two-field shape (`handle`, `rerun_on_resume`); live introspection shows `ResumabilityConfig.model_fields` is a **single field**, `is_resumable: bool = False` — matching the other volumes, not `_v18.md`. Use the single-field shape.

```python
from google.adk.apps import ResumabilityConfig
ResumabilityConfig(is_resumable=True)
```

#### `EventsCompactionConfig` + `ResumabilityConfig`
**Source:** `google.adk.apps._configs` (verified 2.7.1 signatures)

Both attach to `App(...)`.

```python
@experimental
class ResumabilityConfig(BaseModel):
    is_resumable: bool = False
    # "resumability": pause on a long-running function call, resume from the last event.
    # Best-effort: resumed tool calls must be idempotent (at-least-once); any
    # temporary/in-memory state is lost on pause.

@experimental
class EventsCompactionConfig(BaseModel):
    summarizer: Optional[BaseEventsSummarizer] = None
    compaction_interval: Optional[int] = Field(default=None, gt=0)
    overlap_size: Optional[int] = Field(default=None, ge=0)
    token_threshold: Optional[int] = Field(default=None, gt=0)
    event_retention_size: Optional[int] = Field(default=None, ge=0)
```

A `@model_validator` requires `token_threshold` + `event_retention_size` to be set **together**, and `compaction_interval` + `overlap_size` to be set **together**; at least one complete pair must be present. `compaction_interval` counts **new user-initiated invocations**, not tokens. Both trigger pairs may coexist — ADK checks the token-threshold trigger first; sliding-window fires only if the token check doesn't.

```python
app = App(name="durable_app", root_agent=root,
          events_compaction_config=EventsCompactionConfig(token_threshold=20_000, event_retention_size=5),
          resumability_config=ResumabilityConfig(is_resumable=True))
```

#### `LlmEventSummarizer` + `BaseEventsSummarizer`
**Source:** `google.adk.apps.llm_event_summarizer`, `.base_events_summarizer`

`BaseEventsSummarizer` is the `@experimental` ABC any custom compactor implements:

```python
class BaseEventsSummarizer(abc.ABC):
    @abc.abstractmethod
    async def maybe_summarize_events(self, *, events: list[Event]) -> Optional[Event]:
        """Return None to skip compaction, or an Event with actions.compaction populated
        (EventCompaction(start_timestamp, end_timestamp, compacted_content)). The runner
        appends the returned event to the session — do not persist it yourself."""
```

`LlmEventSummarizer` is the built-in implementation ADK auto-creates from the root `LlmAgent.canonical_model` when `EventsCompactionConfig.summarizer` is `None`.

```python
class LlmEventSummarizer(BaseEventsSummarizer):
    _DEFAULT_PROMPT_TEMPLATE: str          # instructs: state the primary language, list exact
                                            # tool names called, produce a concise summary
    _MAX_TOOL_CONTENT_CHARS: int = 2000    # tool call args/responses truncated to this length

    def __init__(self, llm: BaseLlm, prompt_template: Optional[str] = None): ...
```

`_format_events_for_prompt` renders: `"{author} (thought): {text}"` for thought parts (skipped if the event is itself a compaction, to avoid summarizing summaries); `"{author}: {text}"` for plain text; `"{author} called tool: {name}({args})"` / `"Tool response from {name}: {response}"` for function calls/responses (args/response truncated at 2000 chars).

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini
summarizer = LlmEventSummarizer(llm=Gemini(model="gemini-2.5-flash"))  # cheaper model for compaction
```

#### `CompactionRequestProcessor`
**Source:** `google.adk.flows.llm_flows.compaction`

A `BaseLlmRequestProcessor` that compacts session events **before** the contents processor runs, activated only when `EventsCompactionConfig` has token-threshold fields set (`_has_token_threshold_config` guard — a purely sliding-window config is a no-op here). Delegates to `_run_compaction_for_token_threshold_config()` (module `apps.compaction`) and, on success, sets `invocation_context.token_compaction_checked = True` to prevent redundant compaction within the same invocation. Module exports a ready-to-use singleton: `request_processor = CompactionRequestProcessor()` — no need to instantiate it yourself.

#### `ContextCacheRequestProcessor`
**Source:** `google.adk.flows.llm_flows.context_cache_processor`

Enables Gemini context caching for agents with `ContextCacheConfig` configured. No-ops immediately if `invocation_context.context_cache_config is None`; otherwise sets `llm_request.cache_config`, scans session events for the most recent `CacheMetadata` and the previous cacheable-content token count (via `_find_cache_info_from_events()`), and populates `llm_request.cache_metadata` / `cacheable_contents_token_count` so the model-specific cache manager can decide whether to reuse or refresh the cache. Both the existing cache name and the previous prompt token count come from past `Event` objects in the session — there's no external cache-state store.

### Runner & Execution Internals

#### `Runner`
**Source:** `google.adk.runners`

The single public entry point for all agent execution: turn-by-turn (`run_async`), live bidirectional (`run_live`), and a synchronous wrapper (`run`).

```python
Runner(
    *, app: Optional[App] = None, app_name: Optional[str] = None, agent: Optional[BaseAgent] = None,
    node: Any = None, plugins: Optional[list[BasePlugin]] = None,   # deprecated; use App instead
    artifact_service: Optional[BaseArtifactService] = None,
    session_service: BaseSessionService,                            # required
    memory_service: Optional[BaseMemoryService] = None,
    credential_service: Optional[BaseCredentialService] = None,
    plugin_close_timeout: float = 5.0,
    auto_create_session: bool = False,
)

async def run_async(self, *, user_id, session_id, invocation_id=None, new_message=None,
                     state_delta=None, run_config=None, yield_user_message=False) -> AsyncGenerator[Event, None]: ...
```

**Root-agent mode guard** — if `agent.mode is None`, `run_async` forces it to `"chat"` (an `LlmAgent` root must be chat mode); a root already in `"task"` mode raises `ValueError`. `_find_agent_to_run` walks session events backwards to find the last incomplete sub-agent invocation for resume, bypassed when a task sub-agent exists (the coordinator always gets the message instead). In `run_live`, events that are *yielded but not saved* (inline audio blobs) are distinguished from events that are *saved and yielded* (file-data references, usage metadata, transcription, function calls) — inline audio streams without polluting the session.

```python
runner = Runner(agent=agent, app_name="demo", session_service=InMemorySessionService(), auto_create_session=True)
async for event in runner.run_async(user_id="u1", session_id="session-42", new_message=msg,
                                     state_delta={"user_name": "Alice"}):
    if event.is_final_response():
        print(event.content.parts[0].text)
```

#### `InMemoryRunner.run_debug`
**Source:** `google.adk.runners` (`InMemoryRunner.run_debug`, verified 2.7.1 signature name/shape unchanged since 2.4.0)

Developer-convenience wrapper over `run_async`: auto-creates a session, accepts `str | list[str]`, and optionally prints agent output.

```python
async def run_debug(self, user_messages: str | list[str], *,
                     user_id: str = "debug_user_id", session_id: str = "debug_session_id",
                     run_config: RunConfig | None = None, quiet: bool = False,
                     verbose: bool = False) -> list[Event]: ...
```

`quiet=True` suppresses console output (still returns events); `verbose=True` also prints tool calls/results; reusing the same `session_id` across calls continues the conversation.

```python
runner = InMemoryRunner(agent=agent)
events = await runner.run_debug("What is the capital of France?")
final = next((e for e in reversed(events) if e.is_final_response()), None)
```

#### `SlackRunner`
**Source:** `google.adk.integrations.slack.slack_runner` — `pip install "google-adk[slack]"`

Wraps any `Runner` with a [Slack Bolt](https://slack.dev/bolt-python/) `AsyncApp`, handling `app_mention` and DM events over **Socket Mode** (no public webhook URL needed).

```python
class SlackRunner:
    def __init__(self, runner: Runner, slack_app: AsyncApp): ...
    async def start(self, app_token: str): ...
```

Session ID convention: `thread_ts = event.get("thread_ts") or event.get("ts")`; `session_id = f"{channel_id}-{thread_ts}"` — every Slack event carries `ts`, so threaded replies share the parent's `thread_ts` while top-level messages use their own, giving each thread (or unthreaded message) an isolated ADK session. A `"_Thinking..._"` placeholder is posted immediately and updated in-place once the first text part arrives.

```python
slack_runner = SlackRunner(runner=runner, slack_app=AsyncApp(token=os.environ["SLACK_BOT_TOKEN"]))
await slack_runner.start(app_token=os.environ["SLACK_APP_TOKEN"])
```

#### `get_fast_api_app`
**Source:** `google.adk.cli.fast_api`

Constructs the `FastAPI` app behind `adk api_server`/`adk web`.

```python
def get_fast_api_app(
    *, agents_dir: str, agent_loader: Optional[BaseAgentLoader] = None,
    session_service_uri: Optional[str] = None, artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None, eval_storage_uri: Optional[str] = None,
    allow_origins: Optional[list[str]] = None, web: bool, a2a: bool = False,
    task_store_uri: Optional[str] = None, host: str = "127.0.0.1", port: int = 8000,
    url_prefix: Optional[str] = None, trace_to_cloud: bool = False, otel_to_cloud: bool = False,
    reload_agents: bool = False, lifespan=None, extra_plugins: Optional[list[str]] = None,
    auto_create_session: bool = False,
    trigger_sources: Optional[list[Literal["pubsub", "eventarc"]]] = None,
    default_llm_model: Optional[str] = None, express_mode: bool = False,
) -> FastAPI: ...
```

| URI scheme | Backend |
|---|---|
| `memory://` | explicitly ephemeral in-memory |
| `None` (default) | local SQLite under `.adk/` when the working directory is writable; falls back to in-memory on a read-only FS (e.g. Cloud Run) |
| `sqlite:///path.db` | SQLite via `aiosqlite` |
| `postgresql://...` | PostgreSQL via SQLAlchemy async |
| `agentengine://resource-id` | Vertex AI Agent Engine |
| `gs://bucket/prefix` | GCS artifact service |

Passing `session_service_uri=None` does **not** guarantee ephemeral storage — on a writable dev machine ADK persists to `.adk/`; pass `memory://` explicitly for a guaranteed-ephemeral dev backend. `a2a=True` adds `/a2a/**` endpoints; `trigger_sources=["pubsub", "eventarc"]` adds the trigger endpoints described by `TriggerRouter` below.

#### `TriggerRouter` + `PubSubTriggerRequest` + `EventarcTriggerRequest`
**Source:** `google.adk.cli.trigger_routes`

Registers `/apps/{app_name}/trigger/pubsub` and `/apps/{app_name}/trigger/eventarc` on a FastAPI app, letting agents process Pub/Sub push messages and Eventarc CloudEvents without a pre-created session. `DEFAULT_TRIGGER_SOURCES = []` — no endpoints register by default; opt in explicitly.

- A single `asyncio.Semaphore(max_concurrent)` (default 10, override via `ADK_TRIGGER_MAX_CONCURRENT`) gates concurrent invocations across both endpoints; excess requests queue.
- `_run_agent_with_retry` retries up to `max_retries` (default 3) on 429/`RESOURCE_EXHAUSTED`, delay `min(base_delay * 2**attempt, max_delay) + jitter(0..delay*0.5)`; exhaustion raises `TransientError` → HTTP 500, signaling the upstream service to retry at the delivery level.
- `/trigger/eventarc` handles both CloudEvents *structured* mode (full event in the JSON body) and *binary* mode (`ce-*` headers + a Pub/Sub-wrapped body), inspecting `req.message`/`req.data` then falling back to raw serialization.

```python
router = TriggerRouter(server, trigger_sources=["pubsub"], max_concurrent=5, max_retries=2)
router.register(app)   # POST /apps/{app_name}/trigger/pubsub
```

#### `ServiceRegistry` + `load_services_module`
**Source:** `google.adk.cli.service_registry`

Maps URI scheme prefixes to factory callables for session/artifact/memory/A2A-task-store services; `get_service_registry()` returns the singleton, extensible via a `services.py` or `services.yaml` in the agent directory (dual registration — YAML loads first, then `services.py`; on a scheme collision the Python file wins).

Built-in schemes: sessions `memory://`→`InMemorySessionService`, `sqlite://`→`SqliteSessionService`, `agentengine://`→`VertexAiSessionService`, `postgresql://`/`mysql://`→`DatabaseSessionService`; artifacts `memory://`→`InMemoryArtifactService`, `gs://`→`GcsArtifactService`, `file://`→`FileArtifactService` (path-traversal-checked); memory `memory://`→`InMemoryMemoryService`, `rag://`→`VertexAiRagMemoryService`, `agentengine://`→`VertexAiMemoryBankService`; A2A task stores `memory://`→`InMemoryTaskStore`, `postgresql+asyncpg://`/`mysql+aiomysql://`/`sqlite+aiosqlite://`→`DatabaseTaskStore`. `sqlite://` with no path falls back to `InMemorySessionService`; a short-form `agentengine://<id>` (no slashes) reads `GOOGLE_CLOUD_PROJECT`/`LOCATION` from env, while a full resource name skips that lookup.

```python
# my_agent/services.py
from google.adk.cli.service_registry import get_service_registry
get_service_registry().register_session_service("myscheme", my_custom_factory)
```

#### `MockModel` + `AgentTestRunner`
**Source:** `google.adk.cli.agent_test_runner`

pytest infrastructure for offline, deterministic testing. `MockModel` is a `BaseLlm` subclass returning pre-scripted `LlmResponse` objects.

```python
EXCLUDED_EVENT_FIELDS = {"id", "timestamp", "invocation_id", "model_version", "finish_reason",
                          "usage_metadata", "avg_logprobs", "cache_metadata", "logprobs_result",
                          "citation_metadata"}   # excluded from snapshot-test equality — non-deterministic fields

def get_test_files(target_folder: str | None = None) -> list[pytest.ParameterSet]: ...
    # walks target_folder (or $ADK_TEST_FOLDER) for tests/*.json; includes a file only if its
    # parent dir looks like an agent dir (agent.py / __init__.py / root_agent.yaml present);
    # a "_xfail.json" suffix is auto-marked xfail
```

```python
model = MockModel(responses=[LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Paris.")]))])
agent = LlmAgent(name="greeter", model=model, instruction="Greet the user by name.")
```

#### Event / EventActions
Why it matters: `Event` is the atomic unit of everything that happens during a run (model output, tool call/result, state delta); `EventActions` carries the side-effect payload (state changes, artifact deltas, transfer/escalate flags) attached to an event. **Correction**: `Event` (which extends `LlmResponse`) carries two fields absent from every source doc — `environment_id` and `voice_activity` — reflecting newer live-session/environment-simulation support added after the docs were written. `EventActions`' 14-field shape matches the docs exactly (no drift there).

```python
from google.adk.events import Event, EventActions
# Event.model_fields now includes: environment_id, voice_activity (undocumented in sources)
```

#### LlmRequest / LlmResponse
Why it matters: the normalized request/response envelope every `BaseLlm` implementation consumes/produces, decoupling agent logic from any specific model API shape. `LlmResponse.model_fields` shows the same two new fields as `Event` (`environment_id`, `voice_activity`), which is expected since `Event` subclasses `LlmResponse`; every other field matches the docs.

```python
from google.adk.models import LlmRequest, LlmResponse
```

#### `LlmRequest`
**Source:** `google.adk.models.llm_request`

The mutable container every `BaseLlmRequestProcessor` populates before the model call.

```python
class LlmRequest(BaseModel):
    model: Optional[str] = None
    contents: list[types.Content] = []
    config: types.GenerateContentConfig
    tools_dict: dict[str, BaseTool] = {}                 # excluded from serialization
    cache_config: Optional[ContextCacheConfig] = None
    previous_interaction_id: Optional[str] = None        # Interactions API chaining

    def append_instructions(self, instructions: list[str] | types.Content) -> list[types.Content]: ...
    def append_tools(self, tools: list[BaseTool]) -> None: ...
    def set_output_schema(self, output_schema: SchemaType, *, base_model: SchemaType = None) -> None: ...
```

`append_tools` merges new `FunctionDeclaration`s into the **single** existing `types.Tool(function_declarations=[...])` in `config.tools`, avoiding duplicate wrapper objects. `append_instructions` concatenates text with double newlines into `system_instruction`; passing a `Content` with an inline image splits it — text goes to the system instruction, the image becomes a prepended user-content reference.

#### BaseLlm / LLMRegistry
Why it matters: the abstract model interface and the registry that resolves a model-name string (e.g. `"gemini-2.5-flash"`) to a concrete `BaseLlm` implementation — the extension point for adding a custom or third-party model backend.

```python
from google.adk.models import BaseLlm, LLMRegistry
```

#### `LLMRegistry`
**Source:** `google.adk.models.registry`

The global registry mapping model-name strings to `BaseLlm` classes. `resolve(model_name)` is `@lru_cache(maxsize=32)`-decorated and returns the class (not an instance); `new_llm(model_name)` instantiates it. Supports prefix routing (`"litellm:gpt-4o"` → strip the `litellm:` prefix, match against the class name) and lazy provider loading so heavy optional imports (e.g. `litellm`) aren't paid for unless that prefix is actually used.

```python
class LLMRegistry:
    @staticmethod
    def register(llm_cls: type[BaseLlm]) -> None: ...   # reads llm_cls.supported_models() for regex patterns

    @staticmethod
    @lru_cache(maxsize=32)
    def resolve(model: str) -> type[BaseLlm]: ...
```

```python
from google.adk.models.registry import LLMRegistry
from google.adk.models.lite_llm import LiteLlm

LLMRegistry.register(LiteLlm)
llm = LLMRegistry().new_llm("litellm:gpt-4o")
print(type(llm).__name__)  # LiteLlm
```

#### LiteLlm
Why it matters: routes to 100+ non-Gemini providers via the LiteLLM library. Confirmed still at `google.adk.models.lite_llm.LiteLlm`; requires the optional `litellm` extra (`pip install google-adk[extensions]`) — its `ImportError` in the bare verification venv is the expected optional-dependency gate.

```python
from google.adk.models.lite_llm import LiteLlm
model = LiteLlm(model="claude-opus-4-5-20250514")
```

#### AnthropicLlm / Claude
Why it matters: **Correction** — `AnthropicLlm` itself is not in `google.adk.models.__all__` (only its subclass `Claude` is publicly re-exported); import the base class from its submodule if you need it directly. Both classes exist and are unchanged in shape from the docs.

```python
from google.adk.models import Claude               # public
from google.adk.models.anthropic_llm import AnthropicLlm  # base class, submodule only
```

#### ApigeeLlm
Why it matters: routes model calls through an Apigee API-management gateway (auth, rate limiting, observability) in front of the underlying Gemini call. Confirmed unchanged; now documented as subclassing `Gemini` directly (`class ApigeeLlm(Gemini)`), not `BaseLlm` — a detail worth getting right if you're overriding its methods.

```python
from google.adk.models import ApigeeLlm
```

#### Gemma / Gemma3Ollama
Why it matters: local/open-weight Gemma model support, including an Ollama-backed variant for fully local inference. Both remain exported from `google.adk.models`.

```python
from google.adk.models import Gemma, Gemma3Ollama
```

#### `OpenAILlm` (labs)
**Source:** `google.adk.labs.openai._openai_llm` — install `openai` separately; import path is `google.adk.labs.openai`.

Experimental `BaseLlm` that swaps Gemini for any OpenAI Chat Completions-compatible model without changing agent code.

```python
class OpenAILlm(BaseLlm):
    model: str = "gpt-4o"
    max_tokens: int = 4096

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"gpt-.*", r"o1-.*", r"o3-.*"]
```

Translates ADK↔OpenAI concepts: `Content(role="model")` → `{"role": "assistant"}`; `Part.function_call` → OpenAI `tool_calls`; a Pydantic `response_schema` → `response_format: {"type": "json_schema", ...}`. JSON Schema `type` values are lowercased recursively (`_update_type_string`) because Gemini emits uppercase type strings that OpenAI's strict validator rejects. Streaming accumulates text and per-index tool-call argument fragments, yielding one final non-partial `LlmResponse`.

```python
from google.adk.agents import LlmAgent
from google.adk.labs.openai import OpenAILlm
from google.adk.models.registry import LLMRegistry

LLMRegistry.register(OpenAILlm)
agent = LlmAgent(name="gpt-agent", model="gpt-4o", instruction="You are a helpful assistant.")
# requires OPENAI_API_KEY in the environment
```

#### `Gemini` model class
**Source:** `google.adk.models.google_llm` (verified 2.7.1)

The concrete `BaseLlm` for all Gemini models; `LLMRegistry` resolves a bare model string like `"gemini-2.5-flash"` to this class.

```python
class Gemini(BaseLlm):
    model: str = 'gemini-2.5-flash'
    client_kwargs: Optional[dict[str, Any]] = None         # forwarded verbatim to google.genai.Client()
    base_url: Optional[str] = None                          # override the AI platform base URL
    speech_config: Optional[types.SpeechConfig] = None       # TTS voice/encoding for Live-mode sessions
    use_interactions_api: bool = False                       # route through client.aio.interactions
    retry_options: Optional[types.HttpRetryOptions] = None   # HTTP-level retry policy
```

Override the `@cached_property api_client` in a subclass to pass any `google.genai.Client` constructor argument ADK doesn't expose as a field (`project`, `location`, `credentials`, `http_options`, `enterprise`, ...).

```python
class RegionalGemini(Gemini):
    @cached_property
    def api_client(self) -> Client:
        return Client(vertexai=True, project="my-gcp-project", location="europe-west4")
```

#### `generate_content_via_interactions`
**Source:** `google.adk.models.interactions_utils`

An async generator routing LLM requests through Gemini's **Interactions API** instead of `generate_content`. When `llm_request.previous_interaction_id` is set, only the latest user content is sent (not the full conversation) — reducing payload size; both streaming and non-streaming are supported. Enabled by constructing `Gemini(model=..., use_interactions_api=True)` (see below).

#### Model name utilities
**Source:** `google.adk.utils.model_name_utils`

Branches on model family without hard-coding string prefixes; normalizes both simple names and Vertex AI path forms (`projects/.../publishers/google/models/gemini-2.5-flash`).

```python
def extract_model_name(model_string: str) -> str: ...   # strips Vertex/apigee/"models/" prefixes
def is_gemini_model(model_string) -> bool: ...
def is_gemini_1_model(model_string) -> bool: ...          # matches ^gemini-1\.\d+
def is_gemini_eap_or_2_or_above(model_string) -> bool: ...
def is_gemini_3_1_flash_live(model_string) -> bool: ...
def is_gemini_3_5_live_translate(model_string) -> bool: ...
```

`ADK_DISABLE_GEMINI_MODEL_ID_CHECK=true` bypasses the `^gemini-` prefix check for internal/non-public model IDs; `is_gemini_model_id_check_disabled()` reads that flag (downstream consumers like `output_schema_utils` check it before applying model-family-based routing guards).

#### BuiltInCodeExecutor / UnsafeLocalCodeExecutor / VertexAiCodeExecutor
Why it matters: the three code-executor tiers usable without any extra install — native Gemini code execution, an unsandboxed local subprocess (dev/trusted-code only), and a managed Vertex AI executor. All confirmed present and unchanged at `google.adk.code_executors`.

```python
from google.adk.code_executors import BuiltInCodeExecutor, UnsafeLocalCodeExecutor, VertexAiCodeExecutor
```

#### ContainerCodeExecutor / GkeCodeExecutor / AgentEngineSandboxCodeExecutor
Why it matters: the three sandboxed-execution tiers (Docker container, gVisor-isolated GKE Job, managed Agent Engine sandbox). All three remain exported from `google.adk.code_executors.__all__`; `ContainerCodeExecutor` additionally requires the `[extensions]` optional dependency group to actually construct (its `ImportError` in the bare venv is expected).

```python
from google.adk.code_executors import ContainerCodeExecutor, GkeCodeExecutor, AgentEngineSandboxCodeExecutor
```

#### CodeExecutorContext
Why it matters: per-invocation state for a code executor (working directory, installed packages, execution history) — shared across all executor tiers regardless of sandbox backend.

```python
from google.adk.code_executors import CodeExecutorContext
```

#### `_CodeExecutionRequestProcessor` / `_CodeExecutionResponseProcessor` + `DataFileUtil`
**Source:** `google.adk.flows.llm_flows._code_execution`

```python
@dataclass
class DataFileUtil:
    extension: str; loader_code_template: str   # "{filename}" placeholder
_DATA_FILE_UTIL_MAP = {"text/csv": DataFileUtil(extension=".csv", loader_code_template="pd.read_csv('{filename}')")}
```

Only CSV is supported at present. The request processor skips entirely for agents with no `code_executor`; for `BuiltInCodeExecutor` it just calls `code_executor.process_llm_request(llm_request)` and returns — the `DataFileUtil` CSV-optimization path (replacing inline CSV with a `data_N_M.csv` placeholder + pandas-exploration output) applies only to **external** `BaseCodeExecutor` subclasses with `optimize_data_file=True` (e.g. `UnsafeLocalCodeExecutor` fixes both `stateful` and `optimize_data_file` to `False`, so its CSV reaches the model unprocessed). The response processor: for `BuiltInCodeExecutor`, saves `image/*` inline data parts to the artifact service and replaces them with `"Saved as artifact: {file_name}."`; for external executors, extracts and runs the first code block in the reply, yields separate code/result events, and sets `llm_response.content = None` to continue the generation loop — tracking `error_count` via `code_executor_context` and stopping once it reaches `code_executor.error_retry_attempts`.

#### BuiltInPlanner / PlanReActPlanner
Why it matters: `BuiltInPlanner` delegates planning to the model's native thinking/planning mode; `PlanReActPlanner` implements an explicit ReAct-style plan-then-act loop for models without native planning. Both confirmed unchanged at `google.adk.planners`.

```python
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
```

#### `_BasicLlmRequestProcessor` + `_build_basic_request`
**Source:** `google.adk.flows.llm_flows.basic`

The first request processor in the standard flow: sets `llm_request.model` from `agent.canonical_model`, deep-copies `agent.generate_content_config` into `llm_request.config` (or a fresh `GenerateContentConfig()`), conditionally applies `output_schema` (skipped in task mode; only set if the agent has no tools or the model supports structured output alongside tools — checked via `can_use_output_schema_with_tools(model)`), and populates all live-connect fields (`response_modalities`, `speech_config`, etc.) from `invocation_context.run_config`, not from `agent.generate_content_config`. Module-level singleton `request_processor = _BasicLlmRequestProcessor()`.

#### `_OutputSchemaRequestProcessor`
**Source:** `google.adk.flows.llm_flows._output_schema_processor`

Bridges `output_schema` + `tools` on models that can't natively use both together (`can_use_output_schema_with_tools()` returns `False`). No-ops if the agent lacks either `output_schema` or `tools`, if the model *can* combine them, or if `agent.mode == 'task'` (task agents use `FinishTaskTool` for typed output instead). When active, it injects a `SetModelResponseTool` and an instruction forcing the model to call it for its final answer; `get_structured_model_response(event)` scans for a `set_model_response` function response and JSON-serializes its payload (unwrapping a `{"result": ...}` envelope if present).

#### `_NlPlanningRequestProcessor` + response processor
**Source:** `google.adk.flows.llm_flows._nl_planning`

Module-level singletons inserted into the pipeline whenever an `LlmAgent` has a `planner`. For a `BuiltInPlanner`, the request side calls `planner.apply_thinking_config(llm_request)`; for a `PlanReActPlanner`, it calls `planner.build_planning_instruction(...)` and then strips thought parts from history (`_remove_thought_from_request` sets `part.thought = None` on every part, preventing prior reasoning from accumulating in context). On the response side, `BuiltInPlanner` is a no-op (Gemini's extended thinking is already embedded in the response parts); `PlanReActPlanner` validates the model followed the required plan-prefix format.

#### `_IdentityLlmRequestProcessor`
**Source:** `google.adk.flows.llm_flows.identity`

The simplest processor: injects `'You are an agent. Your internal name is "{agent.name}".'` (plus `' The description about you is "{agent.description}".'` when non-empty) into the system prompt on every call, **except** when `agent.mode == 'single_turn'` — task-delegation sub-agents in that mode get no identity injection at all. Module-level singleton `request_processor`.

#### TelemetryContext
Why it matters: **Note** — two distinct classes share this name in 2.7.1: `google.adk.telemetry.node_tracing.TelemetryContext` (workflow-node span context) and `google.adk.telemetry._instrumentation.TelemetryContext` (general OpenTelemetry instrumentation context). Neither is re-exported from `google.adk.telemetry` package root; pick the submodule matching your use case rather than assuming a single shared class.

```python
from google.adk.telemetry.node_tracing import TelemetryContext as NodeTelemetryContext
from google.adk.telemetry._instrumentation import TelemetryContext as InstrumentationContext
```

#### `TelemetryContext` + `start_as_current_node_span`
**Source:** `google.adk.telemetry.node_tracing`

```python
GEN_AI_WORKFLOW_NESTED = "gen_ai.workflow.nested"
_ENTRYPOINT_WORKFLOW_KEY = context_api.create_key("adk-entrypoint-workflow-active")

@dataclass(frozen=True)
class TelemetryContext:
    otel_context: context_api.Context
    _associated_event_ids: list[str] = field(default_factory=list)
    def add_event(self, event: Event) -> None: ...

@asynccontextmanager
async def start_as_current_node_span(context: Context, node: BaseNode) -> AsyncIterator[TelemetryContext]: ...
```

Dispatches to `invoke_agent {name}` (agents emit this span themselves), `invoke_workflow {name}` (for `Workflow` nodes), or `invoke_node {name}` (plain `BaseNode`s — no OTel semconv standard yet). `_ENTRYPOINT_WORKFLOW_KEY` is set once the first workflow span opens; a nested workflow (e.g. an agent-as-tool spinning up a sub-workflow) checks this key and emits `gen_ai.workflow.nested=True`. After the span closes, associated event IDs are stamped onto `gcp.vertex.agent.associated_event_ids`.

#### `trace_call_llm` + `trace_tool_call`
**Source:** `google.adk.telemetry.tracing`

Write OTel span attributes for every LLM/tool invocation. `trace_call_llm` sets `gen_ai.system="gcp.vertex.agent"`, `gen_ai.request.model`, invocation/session IDs, and `gcp.vertex.agent.llm_request`/`llm_response` (JSON, or `"{}"` when content capture is disabled via `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false`, default `true`). `trace_tool_call` sets `gen_ai.operation.name="execute_tool"`, `gen_ai.tool.name/type/description`, and `error.type` (an explicit `Exception` wins over a passed `error_type` string if both are given). `GCP_MCP_SERVER_DESTINATION_ID` — when a `BaseTool.custom_metadata` carries this key, `trace_tool_call` copies it to a span attribute that [AppHub](https://cloud.google.com/app-hub/docs/) uses to associate the call with a destination GCP resource.

#### `ApiServerSpanExporter` + `InMemoryExporter`
**Source:** `google.adk.cli.api_server`

Two `SpanExporter`s used by the ADK API server to feed spans to in-memory dicts without an external backend. `ApiServerSpanExporter` keeps only `"call_llm"`, `"send_data"`, and `"execute_tool*"` spans, storing each (as a dict of attributes + trace/span ID) under `trace_dict[event_id]` — spans without an event ID are dropped. `InMemoryExporter` keeps **all** spans in `self._spans` and separately indexes `trace_dict[session_id] → [trace_id, ...]` (duplicate trace IDs within a session deduplicated); `get_finished_spans(session_id)` filters by that index. Both `force_flush()` synchronously return `True`; `InMemoryExporter.clear()` clears `_spans` but leaves `trace_dict` intact.

#### `SerializedBaseModel`
**Source:** `google.adk.utils._serialized_base_model`

A thin `pydantic.BaseModel` subclass enforcing camelCase JSON serialization for custom web-facing/storage models. Note: `Event` and `Session` do **not** inherit from it — their `model_dump_json()` does not default to `by_alias=True`.

```python
class SerializedBaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        alias_generator=alias_generators.to_camel, populate_by_name=True, use_attribute_docstrings=True)
    def model_dump_json(self, **kwargs) -> str:
        kwargs.setdefault('by_alias', True)   # camelCase output by default
        return super().model_dump_json(**kwargs)
```

`populate_by_name=True` lets Python code set fields snake_case while the wire format is camelCase; `use_attribute_docstrings=True` promotes plain field docstrings (not just `Field(description=...)`) into JSON Schema, powering auto-generated OpenAPI docs.

#### `print_event` + content utilities
**Source:** `google.adk.utils._debug_output`, `google.adk.utils.content_utils`

```python
_ARGS_MAX_LEN = 50; _RESPONSE_MAX_LEN = 100; _CODE_OUTPUT_MAX_LEN = 100

def print_event(event: Event, *, verbose: bool = False) -> None: ...
    # verbose=False: only text parts, prefixed "{author} > "; consecutive text parts are
    # buffered and flushed together to avoid repeating the prefix.
    # verbose=True: also prints function_call/function_response/executable_code/
    # code_execution_result/inline_data/file_data, each truncated per the constants above.

def is_audio_part(part: types.Part) -> bool: ...          # inline_data or file_data mime_type startswith "audio/"
def filter_audio_parts(content: types.Content) -> types.Content | None: ...  # None if the whole Content was audio
def extract_text_from_content(content: types.Content | None) -> str: ...    # concatenates non-thought text; '' for None
```

`filter_audio_parts` is what `GeminiLlmConnection.send_history()` uses to strip audio before replaying history to the Live API. `extract_text_from_content` backs `BaseNode._validate_input_data`/`_validate_output_data`'s `Content → str` coercion.

```python
print_event(event)                 # quiet: agent replies only
print_event(event, verbose=True)   # also shows tool calls/results
```

#### `plot_workflow_graph`
**Source:** `google.adk.cli.utils.graph_visualization`

Renders an ADK agent/`Workflow` graph as a Graphviz DOT/SVG diagram with live `NodeStatus` color overlays, powering the ADK web UI's workflow view.

```python
def plot_workflow_graph(app_info: dict[str, Any], agent_state: dict[str, Any] | None = None,
                         format: str = "svg", dark_mode: bool = True) -> str | bytes: ...
```

When `app_info["root_agent"]["graph"]` is empty (an `LlmAgent` without a `Workflow`), it recursively traverses `sub_agents` and lists `tools` as metadata to build a synthetic graph — giving a visual overview even for multi-agent trees that don't use `Workflow`. `NodeStatus` colors (dark mode): `COMPLETED`→`#16A34A`, `RUNNING`→`#D97706`, `FAILED`→`#EF4444`, default→`#1E293B`, START→`#059669`, END→`#DC2626`.

#### `E2BEnvironment` + `DaytonaEnvironment` + `LocalEnvironment`/`BaseEnvironment`/`ExecutionResult`
**Source:** `google.adk.integrations.e2b._e2b_environment`, `google.adk.integrations.daytona._daytona_environment`, `google.adk.environment._local_environment`/`._base_environment`

Three `BaseEnvironment` implementations backing `EnvironmentToolset` (`@experimental`, remote ones need their `pip install google-adk[e2b]`/`[daytona]` extras).

```python
@dataclass
class ExecutionResult:
    exit_code: int; stdout: str; stderr: str; timed_out: bool
```

- **`LocalEnvironment()`** — `asyncio.create_subprocess_shell` + `asyncio.to_thread` file I/O. `initialize()` creates a temp dir (`adk_workspace_` prefix) unless `working_dir` is supplied; `close()` removes it only if it was auto-created. On timeout, `proc.kill()` sets `timed_out=True` and still collects whatever stdout/stderr was captured.
- **`E2BEnvironment(image="base", timeout=300, api_key=None, env_vars=None)`** — every op that finds a still-running sandbox calls `sandbox.set_timeout(self._timeout)` (TTL keepalive); if the sandbox has expired, a fresh one is created transparently (installed packages/files are lost). `_resolve_path()` anchors relative paths under `/home/user` but passes an **absolute** path through unchanged — don't rely on it as a traversal guard for user-supplied absolute paths.
- **`DaytonaEnvironment(image=None, timeout=300, api_key=None, api_url=None, env_vars=None)`** — `initialize()` creates one `AsyncSandbox` (idempotent — a second call is a no-op); `close()` deletes it. `DaytonaError` timeouts map to `ExecutionResult(exit_code=-1, timed_out=True)`; `DaytonaNotFoundError` on read maps to `FileNotFoundError`. Paths resolve under `_SANDBOX_HOME = "/workspaces"`.

```python
env = DaytonaEnvironment(timeout=300)
await env.initialize()
toolset = EnvironmentToolset(environment=env)
```

#### `SandboxClient`
**Source:** `google.adk.integrations.vmaas.sandbox_client` — `@experimental(FeatureName.COMPUTER_USE)`

Drives a Vertex AI Computer Use sandbox browser via Chrome DevTools Protocol commands.

```python
class SandboxClient:
    def __init__(self, vertexai_client: vertexai.Client, sandbox: Any, access_token: str): ...
    async def make_cdp_request(self, command: str, params: dict | None = None) -> dict: ...
    async def make_cdp_batch_request(self, commands: list[dict], stop_on_error: bool = True) -> list[dict]: ...
        # tries POST /cdps first; falls back to sequential make_cdp_request on 404
    async def get_screenshot(self, max_retries: int = 3) -> bytes: ...
        # retries transient CDP errors (e.g. "Execution context was destroyed" mid-navigation)
```

Modifier-key bitmask for `Input.dispatchKeyEvent`: `CONTROL=2, ALT=1, SHIFT=8, COMMAND=4, SUPER=4`.

#### Platform abstractions — injectable time/UUID/thread
**Source:** `google.adk.platform.time`, `.uuid`, `.thread`

Thin `ContextVar`-based wrappers around `time.time()`, `uuid.uuid4()`, and `threading.Thread` so tests can inject deterministic values without monkey-patching stdlib. `set_time_provider`/`set_id_provider` (with matching `reset_*`) store a callable in a `ContextVar`, so overrides in one async task don't leak into a concurrent sibling task.

```python
adk_time.set_time_provider(lambda: 1_700_000_000.0)
try:
    assert adk_time.get_time() == 1_700_000_000.0
finally:
    adk_time.reset_time_provider()
```

#### `find_context_parameter` + `Aclosing`
**Source:** `google.adk.utils.context_utils`

```python
@functools.lru_cache(maxsize=1024)
def find_context_parameter(func) -> str | None: ...
    # inspects inspect.signature + typing.get_type_hints (falls back to raw annotations if that
    # raises); identity-checks each param's annotation against Context (ToolContext/CallbackContext
    # are aliases, so any of them match); returns the parameter NAME, or None if no match.

Aclosing = contextlib.aclosing   # module-level alias, kept for backward compatibility
```

`Aclosing` guarantees `aclose()` fires on an async generator even when the consumer breaks out of the loop early — used throughout the runner and `BaseLlmConnection` implementations to avoid leaking resources.

#### `SchemaType` + `_schema_utils`
**Source:** `google.adk.utils._schema_utils`

Normalizes the four schema forms ADK accepts (`type[BaseModel]`, `list[BaseModel]`, `dict`, `google.genai.types.Schema`) under one alias `SchemaType = types.SchemaUnion`.

```python
def is_basemodel_schema(schema) -> bool: ...
def is_list_of_basemodel(schema) -> bool: ...
def schema_to_json_schema(schema) -> dict: ...      # dict passes through; else TypeAdapter(schema).json_schema()
def validate_schema(schema, json_text: str) -> Any:
    # BaseModel schema  → schema.model_validate_json(json_text).model_dump(exclude_none=True)
    # list[BaseModel]   → TypeAdapter(schema).validate_json(json_text), each item .model_dump(exclude_none=True)
    # else              → json.loads(json_text)
```

#### `AgentInfo` + `get_agents_dict` + `get_tools_info`
**Source:** `google.adk.utils.agent_info`

Runtime inspection of an agent tree — for debugging, docs generation, or a management UI.

```python
class AgentInfo(pydantic.BaseModel):
    name: str; description: str; instruction: str
    tools: list[types.Tool]; sub_agents: list[str]   # names only, not nested AgentInfo objects

async def get_tools_info(tools: list[ToolUnion]) -> list[types.Tool]: ...
    # BaseTool → tool._get_declaration(); BaseToolset → await toolset.get_tools() then each declaration;
    # plain callable → wrapped in FunctionTool, then declaration
async def get_agents_dict(agent: LlmAgent) -> dict[str, AgentInfo]: ...  # DFS, name-guarded against re-visits
```

#### `LlmAgentConfig` + `AgentRefConfig` + `CodeConfig` + `ToolConfig` + `ToolArgsConfig`
**Source:** `google.adk.agents.llm_agent_config`, `.common_configs`, `google.adk.tools.tool_configs`

The `@experimental(AGENT_CONFIG)`, `@deprecated` (favoring plain Python) YAML declarative agent DSL. `AgentLoader._load_from_yaml_config` discovers `agents_dir/{agent_name}/root_agent.yaml` when no Python module exists for that agent name.

```python
class CodeConfig(BaseModel, extra="forbid"):
    name: str                              # FQN, e.g. "my_pkg.tools.my_tool"

class AgentRefConfig(BaseModel, extra="forbid"):
    config_path: str | None = None         # relative YAML path
    code: str | None = None                # FQN of a Python agent instance
    # exactly one of the two must be set

class ToolArgsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")   # open key-value bag for tool constructor args

class ToolConfig(BaseModel, extra="forbid"):
    name: str
    args: ToolArgsConfig | None = None
```

Five `ToolConfig` reference patterns: an ADK built-in by name (`"google_search"`); an FQN to a pre-built tool instance; an FQN to a tool class + `args`; an FQN to a factory function `(args: ToolArgsConfig) -> BaseTool` + `args`; an FQN to a plain Python function (auto-wrapped in `FunctionTool`). `LlmAgentConfig` fields include `model`/`model_code` (mutually exclusive), `instruction`, `static_instruction`, `tools: list[ToolConfig]`, the four callback list fields, `output_schema`/`output_key`, `include_contents`, `generate_content_config`, `disallow_transfer_to_parent/peers`.

#### `AgentBuilderAssistant`
**Source:** `google.adk.cli.built_in_agents.adk_agent_builder_assistant`

A factory producing a fully-configured `LlmAgent` (served as the built-in `"__adk_agent_builder_assistant"`) that can write/read/validate YAML agent configs.

```python
class AgentBuilderAssistant:
    @staticmethod
    def create_agent(model: str = "gemini-2.5-pro") -> LlmAgent: ...
```

Ships 9 `FunctionTool`s (`read_config_files`, `write_config_files`, `explore_project`, `read_files`, `write_files`, `delete_files`, `cleanup_unused_files`, `search_adk_source`, `search_adk_knowledge`) plus 2 `AgentTool`-wrapped sub-agents (`google_search_agent`, `url_context_agent`, since ADK's built-in search/URL-context tools are themselves sub-agents). `instruction` is a callable that resolves the working directory from `session.state` at runtime and injects a compact schema reference (9 core config definitions, pruned generation-config fields) into the system prompt. The module exposes `root_agent = AgentBuilderAssistant.create_agent()` so `AgentLoader` can discover it without explicit registration.

#### `yaml_utils` — `load_yaml_file` + `dump_pydantic_to_yaml` + `_MultilineDumper`
**Source:** `google.adk.utils.yaml_utils`

```python
def load_yaml_file(file_path: str | Path) -> Any: ...    # yaml.safe_load; raises FileNotFoundError
def dump_pydantic_to_yaml(model: BaseModel, file_path, *, indent=2, sort_keys=True,
                           exclude_none=True, exclude_defaults=True, exclude=None) -> None: ...
```

`_MultilineDumper` (a `yaml.SafeDumper` subclass) forces consistent sequence indentation (`indentless=False`), uses `|` literal-block style for any string containing `\n`/`"`/`'`, sets `width=1_000_000` (disables PyYAML's line-wrapping — important for long instruction strings), and `allow_unicode=True`. `sort_keys=True` by default gives deterministic diffs for version-controlled config files.

#### `env_utils` — `is_env_enabled` + `is_enterprise_mode_enabled`
**Source:** `google.adk.utils.env_utils`

```python
def is_env_enabled(env_var_name: str, default: str = '0') -> bool: ...
    # os.environ.get(name, default).lower() in ('true', '1'); all other values (incl. "yes"/"on") are falsy
def is_enterprise_mode_enabled() -> bool: ...
    # checks GOOGLE_GENAI_USE_ENTERPRISE first; falls back to the deprecated GOOGLE_GENAI_USE_VERTEXAI
    # with a DeprecationWarning; False if neither is set
```

`GOOGLE_GENAI_USE_VERTEXAI` is deprecated in favor of `GOOGLE_GENAI_USE_ENTERPRISE`, which gates broader enterprise routing beyond just Vertex AI (checked by `GoogleLLMVariant`, model-name utilities, and several tools).

#### `_InvocationCostManager` + `LlmCallsLimitExceededError` + `RealtimeCacheEntry`
**Source:** `google.adk.agents.invocation_context`

`_InvocationCostManager` is a `BaseModel` tracking `_number_of_llm_calls: int` as a private attribute; `increment_and_enforce_llm_calls_limit(run_config)` increments **before** checking, so the limit is enforced strictly at the call that pushes over it. `LlmCallsLimitExceededError` is raised only when `run_config.max_llm_calls > 0` and the count exceeds it — `max_llm_calls <= 0` disables the check entirely. `InvocationContext.increment_llm_call_count()` is the public entry point.

`RealtimeCacheEntry(role: str, data: types.Blob, timestamp: float)` (`arbitrary_types_allowed=True` for the `Blob`) accumulates audio blob chunks in a live bidirectional session for context-caching before they're flushed to the model.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 2.7.1 | August 21, 2026 | **Documentation consolidation.** Added a new [Class & API Reference](#class--api-reference) section (12 subject areas), merging and deduplicating the 48 sequential "class deep dives" volumes into this single page; those volume pages have been removed. Version banner and frontmatter bumped to `google-adk==2.7.1`; content cross-checked via live introspection against the installed package, confirming (among other things) that `LoopAgent`/`ParallelAgent`/`SequentialAgent` now carry a verified `@deprecated` notice, `EventsCompactionConfig` lives at the private `google.adk.apps._configs`, and GCS/Firestore integrations moved under `google.adk.integrations.*`. |
| 2.3.0 | June 22 – 29, 2026 | Version bump to 2.3.0. Class deep dives vol. 24 added (`LlmAgent` mode system, `ContextCacheConfig`, `State` + `StateSchemaError`, `DatabaseSessionService`, `VertexAiSessionService`, `VertexAiMemoryBankService`, `Event` + `NodeInfo`, `EventActions` + `EventCompaction`, `ReadonlyContext`, `InvocationContext`; all source-verified against `google-adk==2.3.0`). Class deep dives vol. 30 and vol. 31 added (21 additional classes across pipeline processors, registry, environment simulation, and integration layers). Memory & Artifacts and MCP & A2A guides enhanced with 4 and 5 complete runnable examples respectively. |
| 2.1.0 | May 23, 2026 | Minor feature release. `RunConfig` gains `tool_thread_pool_config` (`ToolThreadPoolConfig`), `context_window_compression`, `get_session_config`, `enable_affective_dialog`, `proactivity`, `session_resumption`. `BaseNode.state_schema`, `input_schema`, `output_schema` documented as first-class fields. `Context.add_memory()` for explicit memory entries. `BasePlugin.on_model_error_callback` formally documented. Class deep dives page added. Core symbols verified against installed `google-adk==2.1.0` (`.routine-envs/check-0523-google-adk`) with `-W error::DeprecationWarning`; all PASS. 27 top-level exports confirmed. |
| 2.0.0 (GA) | May 20, 2026 | GA stable release confirmed. `pip install google-adk` (no `--pre` required). Core symbols (`google.adk.agents.Agent`, `google.adk.agents.LlmAgent`, `google.adk.runners.Runner`, `google.adk.sessions.InMemorySessionService`, `google.adk.tools.FunctionTool`, `google.adk.tools.ToolContext`, `google.adk.memory.InMemoryMemoryService`, `google.adk.artifacts.InMemoryArtifactService`) verified against installed `google-adk==2.0.0` (`.routine-envs/check-0520-adk`) with `-W error::DeprecationWarning`; all PASS. |
| 2.0.0 | May 19, 2026 | **Stable major release.** `SequentialAgent`, `ParallelAgent`, `LoopAgent` are officially deprecated in favour of `Workflow`. `ToolContext` is now an alias for `Context` (`agents/context.py`). `LongRunningFunctionTool` moved to `tools/long_running_tool.py`. `McpToolset` gained `credential_key` parameter for shared credential service namespacing. `RunConfig` adds `ToolThreadPoolConfig` for live-mode tool concurrency, `custom_metadata` merged into every event, and `get_session_config` for selective session event loading. Plugins gained `before_run_callback`, `on_event_callback`, `after_run_callback`, and `close` lifecycle hooks. `GlobalInstructionPlugin` replaces the deprecated `LlmAgent.global_instruction` field. `SaveFilesAsArtifactsPlugin` replaces the deprecated `RunConfig.save_input_blobs_as_artifacts`. `Workflow.max_concurrency` limits graph-scheduled parallel nodes. |
| 1.33.0 | May 9, 2026 | Minor stable release. Version confirmed against installed `google-adk 1.33.0`; `google.adk.agents.Agent`, `google.adk.agents.LlmAgent`, `google.adk.tools.FunctionTool` verified with `-W error::DeprecationWarning` — all PASS. |
| 1.32.0 | May 1, 2026 | Stable patch release. Version confirmed against installed `google-adk 1.32.0` (`.routine-envs/check-googadk-0501`); `google.adk.agents.Agent` import verified with `-W error::DeprecationWarning`. |
| 1.31.1 | April 2026 | Patch release; stability improvements. |
| 1.31.0 | April 17, 2026 | Overhauled Web UI: live chat interface, session display names, structured execution traces, Graph View canvas, event filtering (by message/tool/error type), computer-use visualisation; memory bank event ingestion; Vertex AI Agent Engine Sandbox for computer use; Firestore database support; session ID tracking in LLM responses; user-agent headers for Parameter Manager and Secret Manager clients; minimum MCP version raised to 1.24.0; `FunctionDeclaration` JSON schema fallback improved; BigQuery plugin fixes (data transfers, metadata); console URL path corrections after Agent Engine deployment; event callback timing fix (plugin modifications now persist correctly) |
| 1.30.0 | April 13, 2026 | A2A 1.0 spec compliance; `AgentEngineSandboxCodeExecutor`; YAML agent config support; authentication provider support in agent registry; Parameter Manager integration; Gemma 4 model support; artifact service integration via interceptor; `TaskStatusUpdateEvent` emission; live avatar support; BigQuery tools promoted to stable; path traversal validation for user/session IDs |
| 1.29.0 | April 2026 | Session rewind for replay/debugging; `MCPToolset` async-first API (legacy sync API deprecated) |
| 1.28.0 | March 2026 | Task API for structured task management |
| 1.25.0 | February 2026 | `GraphAgent` for graph-based multi-agent workflows; Web UI for graph visualisation |
| 1.18.0 | November 2025 | Previous documented version |

---

*This comprehensive guide covers all major aspects of Google ADK. For specific implementations and advanced patterns, refer to the Production Guide, Recipes, and official ADK documentation.*


Latest: 1.18.0
# Google Agent Development Kit (ADK) - Comprehensive Technical Guide

**Version:** 1.0  
**Last Updated:** 2025  
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
pip install google-adk

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
    temperature=0.7,
    top_p=0.95,
    top_k=40
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
from google.adk.sessions import InMemorySessionService, FirestoreSessionService
from google.genai import types
import asyncio

# Create runner with in-memory session service (for development)
async_runner = Runner(
    app_name="research_app",
    agent=search_agent,
    session_service=InMemorySessionService()
)

# Create runner with Firestore for production
async def create_production_runner():
    runner = Runner(
        app_name="research_app_prod",
        agent=search_agent,
        session_service=FirestoreSessionService(
            project_id="your-project-id",
            collection_name="agent_sessions"
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
from google.adk.agents import ModelConfig

# Configure with Flash model for speed
fast_agent = Agent(
    name="quick_responder",
    model="gemini-2.5-flash",
    description="Provides quick responses",
    instruction="Respond concisely and quickly",
    model_config=ModelConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024
    )
)

# Configure with Pro model for complex reasoning
reasoner = Agent(
    name="complex_reasoner",
    model="gemini-2.5-pro",
    description="Handles complex reasoning tasks",
    instruction="Provide detailed, step-by-step reasoning",
    model_config=ModelConfig(
        temperature=0.3,  # Lower temperature for consistency
        top_p=0.9,
        top_k=20,
        max_output_tokens=4096,
        candidate_count=3  # Multiple candidates for diversity
    )
)

# Configure with safety settings
safe_agent = Agent(
    name="safe_responder",
    model="gemini-2.5-flash",
    instruction="Provide helpful information safely",
    model_config=ModelConfig(
        temperature=0.7,
        safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    )
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
from google.adk.agents import LlmAgent, ModelConfig

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
    model_config=ModelConfig(
        temperature=0.3,
        max_output_tokens=2048,
        top_p=0.9
    ),
    max_iterations=10,
    max_total_tokens=8192
)
```

### Agent Configuration and Initialization

#### Configuration Parameters

```python
from google.adk import Agent
from google.adk.agents import ModelConfig

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
    model_config=ModelConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        candidate_count=1
    ),
    
    # Execution constraints
    max_iterations=5,
    max_total_tokens=4096,
    timeout_seconds=30
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
from google.adk.tools.bigquery_toolset import BigQueryToolset

# Initialise BigQuery tools
bq_tools = BigQueryToolset(project_id="my-gcp-project")

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
    tools=bq_tools.get_tools()
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
from google.adk.agents import ModelConfig
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    """Analysis result structure."""
    topic: str
    key_points: list[str]
    sentiment: str
    confidence: float

json_agent = Agent(
    name="json_analyst",
    model="gemini-2.5-flash",
    instruction="Analyse text and return JSON-formatted results",
    model_config=ModelConfig(
        response_format="json_object",
        max_output_tokens=2048
    )
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
from google.adk.sessions import FirestoreSessionService
from google.cloud import firestore

async def setup_persistent_storage():
    """Set up persistent conversation storage."""
    
    # Initialise Firestore session service
    session_service = FirestoreSessionService(
        project_id="my-gcp-project",
        collection_name="agent_conversations"
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
    
    # Session data is automatically saved to Firestore
    # Can be retrieved later even after restart
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
from google.adk.sessions import FirestoreSessionService
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

{% raw %}
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
{% endraw %}

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
from google.adk.agents import ModelConfig

# Configuration for efficient context usage
efficient_config = ModelConfig(
    temperature=0.7,
    top_p=0.95,
    max_output_tokens=1024,  # Reasonable limit
    stop_sequences=["END", "###"]  # Early stopping
)

efficient_agent = Agent(
    name="efficient_agent",
    model="gemini-2.5-flash",  # Flash for efficiency
    instruction="Provide concise, direct answers",
    model_config=efficient_config,
    max_total_tokens=4096
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
from google.adk.agents import ModelConfig
import vertexai

# Initialise Vertex AI
vertexai.init(project="my-gcp-project", location="us-central1")

# Use Vertex AI's Gemini models
vertex_agent = Agent(
    name="vertex_agent",
    model="gemini-2.5-flash",  # Available through Vertex AI
    description="Agent using Vertex AI models",
    instruction="Process user requests with Vertex AI",
    model_config=ModelConfig(
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=2048
    )
)
```

#### Vertex AI Monitoring Integration

{% raw %}
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
{% endraw %}

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
from google.adk.sessions import FirestoreSessionService
from google.genai import types
import json

agent = Agent(
    name="serverless_agent",
    model="gemini-2.5-flash",
    instruction="Process requests in serverless environment"
)

session_service = FirestoreSessionService(project_id="my-gcp-project")

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
from google.adk.sessions import FirestoreSessionService

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

{% raw %}
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
{% endraw %}

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
from google.adk.agents import ModelConfig

safety_config = ModelConfig(
    temperature=0.5,
    top_p=0.9,
    safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_LOW_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
)

safe_agent = Agent(
    name="safe_assistant",
    model="gemini-2.5-flash",
    instruction="Respond helpfully while maintaining safety",
    model_config=safety_config
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

*This comprehensive guide covers all major aspects of Google ADK. For specific implementations and advanced patterns, refer to the Production Guide, Recipes, and official ADK documentation.*

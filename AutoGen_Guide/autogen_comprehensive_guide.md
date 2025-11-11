# Microsoft AutoGen 0.4: Comprehensive Technical Guide

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Breaking Changes from 0.2.x](#breaking-changes-from-02x)
3. [New Architecture Overview](#new-architecture-overview)
4. [Python and TypeScript APIs](#python-and-typescript-apis)
5. [Core Classes and Interfaces](#core-classes-and-interfaces)
6. [Configuration and Initialisation Patterns](#configuration-and-initialisation-patterns)
7. [Simple Agents](#simple-agents)
8. [Agent Lifecycle Management](#agent-lifecycle-management)
9. [Event-Driven Agent Design](#event-driven-agent-design)
10. [Message Passing Mechanisms](#message-passing-mechanisms)
11. [Single-Agent Task Execution](#single-agent-task-execution)
12. [Agent Capabilities and Roles](#agent-capabilities-and-roles)
13. [Multi-Agent Systems](#multi-agent-systems)
14. [Distributed Agent Architecture](#distributed-agent-architecture)
15. [Agent Discovery and Registration](#agent-discovery-and-registration)
16. [Inter-Agent Communication Protocols](#inter-agent-communication-protocols)
17. [Team and Group Structures](#team-and-group-structures)
18. [Orchestration Patterns](#orchestration-patterns)
19. [Scalability and Load Balancing](#scalability-and-load-balancing)
20. [Tools Integration](#tools-integration)
21. [Tool Registry System](#tool-registry-system)
22. [Cross-Language Tool Sharing](#cross-language-tool-sharing)
23. [Async Tool Execution](#async-tool-execution)
24. [Tool Composition and Chaining](#tool-composition-and-chaining)
25. [Security and Sandboxing](#security-and-sandboxing)
26. [Structured Output](#structured-output)
27. [Memory Systems](#memory-systems)
28. [Context Engineering](#context-engineering)
29. [AutoGen Studio](#autogen-studio)
30. [Azure Integration](#azure-integration)
31. [Agentic Patterns](#agentic-patterns)
32. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
33. [Advanced Topics](#advanced-topics)

---

## Installation and Setup

### Overview

Microsoft AutoGen 0.4 represents a complete architectural rewrite from 0.2.x, introducing a modular package structure designed for flexibility, scalability, and cross-language support. The framework is distributed across multiple packages that can be installed independently, allowing developers to choose precisely the components they need for their specific use case.

### Core Packages

#### `autogen-core`
The foundational package containing the core runtime, message types, and low-level agent infrastructure. This package provides the essential building blocks for creating agents and managing their lifecycle.

```bash
pip install autogen-core
```

**What's Included:**
- Agent runtime and execution engine
- Base agent classes and interfaces
- Message types and protocols
- Topic-based message routing
- Async/await framework integration
- Agent registration and discovery

**When to Use:** You need this for any AutoGen application. It's mandatory.

#### `autogen-agentchat`
Higher-level API built on top of `autogen-core`, providing simplified agent chat interfaces optimised for conversational multi-agent systems. Recommended for most use cases where you want quick agent development.

```bash
pip install autogen-agentchat
```

**What's Included:**
- `AssistantAgent`: LLM-powered conversational agent
- `UserProxyAgent`: Human participant in agent conversations
- `GroupChat`: Multi-agent group conversation management
- `SelectorGroupChat`: LLM-based agent selection for conversations
- `RoundRobinGroupChat`: Sequential agent participation
- Message and event types for agent chat
- Termination conditions
- Tool integration

**When to Use:** For most applications involving agent conversations and tool use.

#### `autogen-ext`
Extensions package providing integrations with external services, models, and storage systems. This is where specialised functionality lives.

```bash
pip install autogen-ext

# Or with specific feature groups:
pip install autogen-ext[azure]      # Azure OpenAI support
pip install autogen-ext[bedrock]    # AWS Bedrock support
pip install autogen-ext[openai]     # OpenAI support (default)
pip install autogen-ext[all]        # All extensions
```

**What's Included:**
- Model clients (OpenAI, Azure OpenAI, Google, Bedrock, etc.)
- Storage implementations (file, SQL, vector databases)
- Tool helpers and builders
- Code execution environments
- Logging and monitoring

**When to Use:** When you need specific integrations or want to use LLM providers other than OpenAI.

#### `ag2-autogen` (Legacy)
Backward compatibility wrapper for AutoGen 0.2.x code. **Not recommended for new projects**; use only if migrating gradually from 0.2.x.

```bash
pip install ag2-autogen  # Legacy - do not use for new projects
```

### Complete Installation Examples

#### Minimal Setup (Python)
```bash
# Bare minimum for a simple agent
pip install autogen-core autogen-agentchat

# Set environment variable for OpenAI
export OPENAI_API_KEY="your-key-here"
```

#### Complete Development Setup (Python)
```bash
# Install all packages with all extensions
pip install autogen-core autogen-agentchat 'autogen-ext[all]'

# Install development tools
pip install pytest pytest-asyncio black isort mypy

# Set up environment variables
export OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_API_KEY="your-azure-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
```

#### TypeScript/JavaScript Setup
```bash
# Create new project
npm init -y

# Install AutoGen packages
npm install autogen-core autogen-agentchat

# Install OpenAI client
npm install @openai/sdk

# Install development dependencies
npm install --save-dev typescript ts-node @types/node
```

### Verification

#### Python Verification
```python
import autogen_core
import autogen_agentchat
from autogen_ext.models.openai import OpenAIChatCompletionClient

print(f"autogen-core version: {autogen_core.__version__}")
print(f"autogen-agentchat version: {autogen_agentchat.__version__}")
print("Installation successful!")
```

#### TypeScript Verification
```typescript
import { Agent } from 'autogen-core';
import { AssistantAgent } from 'autogen-agentchat';

console.log("AutoGen TypeScript modules loaded successfully!");
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 or 3.12 |
| Node.js | 18 LTS | 20 LTS |
| RAM | 4 GB | 8+ GB |
| Disk Space | 500 MB | 2+ GB |
| CPU Cores | 2 | 4+ |

### Environment Configuration

#### API Keys Setup
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_ORG_ID="org-..."  # Optional

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com/"
export OPENAI_API_VERSION="2024-02-01"

# AWS
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Google
export GOOGLE_API_KEY="..."

# Anthropic
export ANTHROPIC_API_KEY="..."
```

#### Programmatic Configuration
```python
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Configure OpenAI
openai_client = OpenAIChatCompletionClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    org_id=os.getenv("OPENAI_ORG_ID"),
)

# Configure with custom base URL
custom_client = OpenAIChatCompletionClient(
    api_key="your-key",
    base_url="https://api.example.com/v1",
    model="gpt-4o",
)
```

---

## Breaking Changes from 0.2.x

### Major Architectural Differences

#### 1. Package Structure

**0.2.x:**
```python
from autogen import AssistantAgent, UserProxyAgent
```

**0.4.x:**
```python
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

### 2. Agent Model Configuration

**0.2.x:**
```python
agent = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": "sk-...",
            }
        ]
    }
)
```

**0.4.x:**
```python
model_client = OpenAIChatCompletionClient(
    model="gpt-4",
    api_key="sk-..."
)

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
)
```

The key difference: 0.4.x uses dependency injection of a `model_client` instead of embedded configuration dictionaries.

### 3. Message Types

**0.2.x:**
```python
from autogen import ConversableAgent

# Messages were simple strings or dicts
agent.send("Hello", recipient=other_agent)
```

**0.4.x:**
```python
from autogen_agentchat.messages import TextMessage, AgentEvent

# Strongly typed messages
message = TextMessage(
    content="Hello",
    source="user",
    models_used=["gpt-4o"]
)

# Agent-specific events
event = AgentEvent(agent_name="assistant")
```

### 4. Async-First Architecture

**0.2.x:**
```python
# Could work synchronously
result = agent.generate_reply(messages)
```

**0.4.x:**
```python
# Everything is async
result = await agent.on_messages(messages)
```

All operations in 0.4.x are async-first, requiring `async`/`await` throughout.

### 5. Tool Definition

**0.2.x:**
```python
def get_weather(city: str) -> str:
    return f"Weather in {city}"

agent = AssistantAgent(
    name="assistant",
    llm_config=config,
    functions=[{
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        }
    }]
)
```

**0.4.x:**
```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Weather in {city}"

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[get_weather]  # Direct function reference!
)
```

0.4.x uses automatic schema generation from Python docstrings.

### 6. GroupChat Changes

**0.2.x:**
```python
groupchat = GroupChat(
    agents=[agent1, agent2],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"
)

chat = GroupChatManager(groupchat=groupchat)
chat.initiate_chat(agent1, message="Start")
```

**0.4.x:**
```python
team = SelectorGroupChat(
    agents=[agent1, agent2],
    model_client=model_client,  # LLM for selection
    termination_condition=MaxMessageTermination(max_messages=20)
)

response = await team.run(task="Start")
```

### 7. Termination Conditions

**0.2.x:**
```python
# Implicit: based on conversation patterns
groupchat = GroupChat(agents=[...], max_round=10)
```

**0.4.x:**
```python
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
    StopMessageTermination
)

# Explicit and composable
termination = (
    TextMentionTermination("TERMINATE") |
    MaxMessageTermination(max_messages=30)
)
```

### 8. Code Execution

**0.2.x:**
```python
from autogen import UserProxyAgent

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "./work",
        "use_docker": False,
    }
)
```

**0.4.x:**
```python
from autogen_agentchat.agents import CodeExecutorAgent

code_executor = CodeExecutorAgent(
    name="code_executor",
    code_execution_config={
        "work_dir": "./work",
        "use_docker": True,
    }
)
```

### 9. Event System

**0.2.x:**
No built-in event system; monitoring was callback-based.

**0.4.x:**
```python
from autogen_core import AgentId
from autogen_agentchat.events import AgentStarted, AgentStopped, MessageSent

# All state changes generate events
team.on(AgentStarted, lambda event: print(f"Agent started: {event.agent_id}"))
team.on(MessageSent, lambda event: print(f"Message: {event.message}"))
```

### 10. Configuration Loading

**0.2.x:**
```python
import json

with open("config.json") as f:
    config_list = json.load(f)
```

**0.4.x:**
```yaml
# config.yaml
models:
  - model: gpt-4o
    api_key: ${OPENAI_API_KEY}
    
agents:
  assistant:
    model: gpt-4o
    system_message: "You are helpful"
```

```python
from autogen_core.config import ConfigLoader

config = ConfigLoader.load("config.yaml")
```

### Migration Checklist

- [ ] Update imports from `autogen` to `autogen_*` packages
- [ ] Replace dict-based `llm_config` with `model_client` parameter
- [ ] Convert all agent methods to async
- [ ] Update tool definitions to use direct function references
- [ ] Replace `GroupChat` with `GroupChatManager` or `SelectorGroupChat`
- [ ] Add explicit termination conditions
- [ ] Update message handling to use typed message classes
- [ ] Implement event handlers instead of callbacks
- [ ] Test error handling (it's different in async context)
- [ ] Profile performance (0.4.x is generally faster but async requires tuning)

---

## New Architecture Overview

### Architectural Principles

The AutoGen 0.4 architecture is built on several core principles:

**1. Modularity**: Components are loosely coupled and independently composable.
**2. Async-First**: All operations support concurrent execution through Python's asyncio.
**3. Type Safety**: Strong typing across Python and TypeScript ensures runtime correctness.
**4. Distributed**: Native support for distributed systems with topic-based routing.
**5. Extensibility**: Plugin architecture for models, tools, and storage.
**6. Event-Driven**: State changes propagate through event system for observability.

### Core Components

#### Runtime Layer
```
┌─────────────────────────────────┐
│   SingleThreadedAgentRuntime    │
│   (Local execution)              │
└─────────────┬───────────────────┘
              │
     ┌────────┴────────┐
     │                 │
  AgentId          MessageContext
```

The runtime manages:
- Agent lifecycle (creation, registration, termination)
- Message routing between agents
- Async task scheduling
- Resource cleanup

#### Agent Layer
```
┌──────────────┐
│   BaseAgent  │ (Abstract base)
└──────┬───────┘
       │
   ┌───┴─────────────────┬──────────────────┐
   │                     │                  │
RoutedAgent      ClosureAgent      UserProxyAgent
(Message handlers) (Functional)    (Human input)
```

Different agent types support different interaction patterns.

#### Communication Layer
```
Agent1 ──┐
         ├─> Router ──> Agent2
Agent3 ──┘     │
              Topic
```

Topic-based routing enables scalable multi-agent systems.

#### Model Integration Layer
```
┌─────────────────────────┐
│  ChatCompletionClient   │ (Interface)
└────────────┬────────────┘
             │
   ┌─────────┼─────────┬──────────┐
   │         │         │          │
OpenAI    Azure    Bedrock    Google
Client    Client    Client    Client
```

Pluggable model clients support multiple LLM providers.

### Execution Flow

```
1. Initialize Runtime
   ├─ Create agent instances
   ├─ Register agents with runtime
   └─ Subscribe to message topics

2. User Provides Task
   └─ Creates initial message

3. Runtime Routes Message
   ├─ Identifies target agent
   └─ Delivers message

4. Agent Processes
   ├─ Calls model if needed
   ├─ Executes tools if requested
   └─ Generates response

5. Response Routing
   ├─ Sends to next agent
   └─ Emits events

6. Termination Check
   ├─ Check termination conditions
   └─ Either continue (step 3) or end
```

### Data Flow

```
┌─────────────┐
│  User Input │
└──────┬──────┘
       │
       v
   ┌───────────────┐
   │ TextMessage   │
   └───┬───────────┘
       │
       v
┌──────────────────┐
│  AgentRuntime    │
│  (routing)       │
└─────┬────────────┘
      │
      v
  ┌────────────┐
  │   Agent    │
  │ (on_messages)
  └─────┬──────┘
        │
        v
   ┌─────────────────────┐
   │ Model Invocation    │
   │ (ChatCompletion)    │
   └────────┬────────────┘
            │
        ┌───┴───┐
        │       │
    Tool Use  Response
        │       │
        └───┬───┘
            │
            v
      ┌──────────────┐
      │  Message     │
      │  + Events    │
      └──────────────┘
```

### State Management

Each agent maintains:
- **Message History**: All received and sent messages
- **Tool Results**: Cached tool execution results
- **Context**: Current execution context
- **State**: Agent-specific internal state

```python
@dataclass
class AgentState:
    agent_id: AgentId
    message_count: int
    last_message_time: float
    tools_available: List[str]
    context: Dict[str, Any]
    pending_tasks: List[Task]
```

### Message Protocol

Messages follow a standardised protocol:

```python
@dataclass
class AgentEvent:
    """Base class for all agent events"""
    source: AgentId
    timestamp: float
    
@dataclass
class TextMessage(AgentEvent):
    """Text-based message"""
    content: str
    models_used: List[str] = None

@dataclass  
class ToolCallMessage(AgentEvent):
    """Tool invocation request"""
    tool_name: str
    arguments: Dict[str, Any]

@dataclass
class ToolResultMessage(AgentEvent):
    """Tool execution result"""
    tool_name: str
    result: Any
    error: Optional[str] = None
```

### Extension Points

The architecture provides extension points for:

1. **Custom Agents**: Extend `BaseAgent` or `RoutedAgent`
2. **Model Providers**: Implement `ChatCompletionClient`
3. **Storage**: Implement storage interfaces
4. **Tool Execution**: Custom tool executors
5. **Message Routing**: Custom router implementations
6. **Event Handlers**: Subscribe to any event type

---

## Python and TypeScript APIs

### Python API Overview

#### Import Structure

```python
# Core runtime
from autogen_core import (
    AgentId,
    AgentRuntime,
    SingleThreadedAgentRuntime,
    RoutedAgent,
    ClosureAgent,
    MessageContext,
    TopicId,
    message_handler,
)

# Agent chat API
from autogen_agentchat.agents import (
    AssistantAgent,
    UserProxyAgent,
    CodeExecutorAgent,
)

from autogen_agentchat.teams import (
    SelectorGroupChat,
    RoundRobinGroupChat,
    Swarm,
)

from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
    StopMessageTermination,
)

from autogen_agentchat.messages import (
    TextMessage,
    ToolCallMessage,
    ToolResultMessage,
    AgentEvent,
)

# Model clients
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.azure import AzureOpenAIChatCompletionClient
from autogen_ext.models.bedrock import BedrockChatCompletionClient
from autogen_ext.models.google import GoogleChatCompletionClient
```

#### Basic Agent Creation

```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

async def main():
    # Initialize model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2048,
    )
    
    # Create agent
    agent = AssistantAgent(
        name="helpful_assistant",
        system_message="You are a helpful assistant that provides concise answers.",
        model_client=model_client,
    )
    
    # Send message and get response
    response = await agent.on_messages([
        TextMessage(
            content="What is the capital of France?",
            source="user"
        )
    ])
    
    print(f"Agent response: {response}")

# Run the async function
asyncio.run(main())
```

#### Tool Integration (Python)

```python
from typing import Annotated
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

# Define tools with annotations
def get_weather(city: Annotated[str, "The city name"]) -> str:
    """Get the weather for a city."""
    # In a real app, call an actual weather API
    return f"The weather in {city} is 72°F and sunny."

def calculate_distance(
    lat1: Annotated[float, "First latitude"],
    lon1: Annotated[float, "First longitude"],
    lat2: Annotated[float, "Second latitude"],
    lon2: Annotated[float, "Second longitude"]
) -> float:
    """Calculate distance between two coordinates using Haversine formula."""
    import math
    
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = AssistantAgent(
        name="utility_agent",
        system_message="You are helpful and use tools to provide accurate information.",
        model_client=model_client,
        tools=[get_weather, calculate_distance],  # Direct function references
    )
    
    response = await agent.on_messages([
        TextMessage(content="What's the weather in Paris?", source="user")
    ])
    
    print(response)

asyncio.run(main())
```

### TypeScript API Overview

#### Import Structure

```typescript
// Core runtime
import {
  AgentId,
  AgentRuntime,
  SingleThreadedAgentRuntime,
  RoutedAgent,
  MessageContext,
} from 'autogen-core';

// Agent chat API
import {
  AssistantAgent,
  UserProxyAgent,
  SelectorGroupChat,
  RoundRobinGroupChat,
} from 'autogen-agentchat';

import {
  TextMessage,
  ToolCallMessage,
  ToolResultMessage,
} from 'autogen-agentchat/messages';

import {
  TextMentionTermination,
  MaxMessageTermination,
} from 'autogen-agentchat/conditions';

// Model clients
import { OpenAIChatCompletionClient } from 'autogen-ext/openai';
import { AzureChatCompletionClient } from 'autogen-ext/azure';
```

#### Basic Agent Creation (TypeScript)

```typescript
import { OpenAIChatCompletionClient } from 'autogen-ext/openai';
import { AssistantAgent } from 'autogen-agentchat';
import { TextMessage } from 'autogen-agentchat/messages';

async function main() {
  // Initialize model client
  const modelClient = new OpenAIChatCompletionClient({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'gpt-4o',
    temperature: 0.7,
    maxTokens: 2048,
  });
  
  // Create agent
  const agent = new AssistantAgent({
    name: 'helpful_assistant',
    systemMessage: 'You are a helpful assistant that provides concise answers.',
    modelClient,
  });
  
  // Send message and get response
  const response = await agent.onMessages([
    new TextMessage({
      content: 'What is the capital of France?',
      source: 'user',
    })
  ]);
  
  console.log(`Agent response: ${response}`);
}

main().catch(console.error);
```

#### Tool Integration (TypeScript)

```typescript
import { OpenAIChatCompletionClient } from 'autogen-ext/openai';
import { AssistantAgent } from 'autogen-agentchat';
import { TextMessage } from 'autogen-agentchat/messages';

// Define tools
function getWeather(city: string): string {
  return `The weather in ${city} is 72°F and sunny.`;
}

interface ToolDef {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  func: Function;
}

const weatherTool: ToolDef = {
  name: 'get_weather',
  description: 'Get the weather for a city.',
  parameters: {
    type: 'object',
    properties: {
      city: { type: 'string', description: 'The city name' }
    },
    required: ['city']
  },
  func: getWeather
};

async function main() {
  const modelClient = new OpenAIChatCompletionClient({
    model: 'gpt-4o',
  });
  
  const agent = new AssistantAgent({
    name: 'utility_agent',
    systemMessage: 'You are helpful and use tools to provide accurate information.',
    modelClient,
    tools: [weatherTool],
  });
  
  const response = await agent.onMessages([
    new TextMessage({
      content: "What's the weather in Paris?",
      source: 'user'
    })
  ]);
  
  console.log(response);
}

main().catch(console.error);
```

### Python vs TypeScript Comparison

| Feature | Python | TypeScript |
|---------|--------|-----------|
| Async | async/await | async/await |
| Type Safety | Optional (Type hints) | Required (Strict mode) |
| Tool Definition | Direct functions | Object definitions |
| Package Manager | pip/Poetry | npm/yarn |
| Deployment | Any Python environment | Node.js runtime |
| Performance | Generally slower | Faster for I/O |
| Community | Larger in ML/AI | Growing in backend |

---

## Core Classes and Interfaces

### Base Agent Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional
from autogen_core import AgentId, MessageContext

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, description: str = ""):
        self._id = AgentId(name)
        self._description = description
    
    @property
    def id(self) -> AgentId:
        """Get the agent's unique identifier"""
        return self._id
    
    @property
    def description(self) -> str:
        """Get the agent's description"""
        return self._description
    
    @abstractmethod
    async def on_messages(
        self,
        messages: List[Any],
        cancellation_token: Optional[CancellationToken] = None
    ) -> str:
        """
        Process incoming messages and produce a response.
        
        Args:
            messages: List of incoming messages
            cancellation_token: Optional token for cancellation
            
        Returns:
            Response string
        """
        pass
```

### AssistantAgent

```python
from typing import Callable, List, Optional, Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

class AssistantAgent:
    """
    LLM-powered agent for conversational interaction.
    
    An AssistantAgent uses a language model to generate responses
    and can optionally use tools to perform tasks.
    """
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        system_message: str = "",
        tools: Optional[List[Callable]] = None,
        reflect_on_tool_use: bool = False,
        tool_choice: str = "auto",
        max_tool_use_attempts: int = 3,
    ):
        """
        Initialize an AssistantAgent.
        
        Args:
            name: Unique agent name
            model_client: LLM client for model interactions
            system_message: System prompt for the agent
            tools: List of callable tools the agent can use
            reflect_on_tool_use: Whether to reflect on tool execution results
            tool_choice: "auto", "required", or "none"
            max_tool_use_attempts: Maximum tool use retries
        """
        self.name = name
        self.model_client = model_client
        self.system_message = system_message
        self.tools = tools or []
        self.reflect_on_tool_use = reflect_on_tool_use
        self.tool_choice = tool_choice
        self.max_tool_use_attempts = max_tool_use_attempts
        self._message_history = []
    
    async def on_messages(
        self,
        messages: List[AgentEvent],
        cancellation_token: Optional[CancellationToken] = None
    ) -> AgentEvent:
        """
        Process messages and generate a response.
        
        This method:
        1. Adds messages to history
        2. Calls the model with system message and history
        3. Handles tool calls if made by the model
        4. Returns the final response
        """
        pass
    
    async def run(
        self,
        task: str,
        cancellation_token: Optional[CancellationToken] = None
    ) -> str:
        """Run a single task"""
        pass
```

### GroupChat Teams

```python
from typing import List, Optional
from autogen_agentchat.teams import SelectorGroupChat

class SelectorGroupChat:
    """
    Multi-agent group chat where an LLM selects the next speaker.
    
    The LLM is given:
    - Current conversation history
    - Agent descriptions and roles
    - Instructions to select the best next speaker
    """
    
    def __init__(
        self,
        agents: List[AssistantAgent],
        model_client: ChatCompletionClient,
        termination_condition: TerminationCondition,
        selector_func: Optional[Callable] = None,
        system_message: Optional[str] = None,
    ):
        """
        Initialize a SelectorGroupChat.
        
        Args:
            agents: List of agents in the group
            model_client: LLM client for speaker selection
            termination_condition: When to stop the conversation
            selector_func: Optional custom selector function
            system_message: Custom system message for selection
        """
        pass
    
    async def run(
        self,
        task: str,
        cancellation_token: Optional[CancellationToken] = None
    ) -> GroupChatResult:
        """
        Run the group chat with the given task.
        
        Returns:
            Result containing final message and agent summaries
        """
        pass
    
    async def on_messages(
        self,
        messages: List[AgentEvent]
    ) -> AgentEvent:
        """Process messages in the group"""
        pass
```

### Tool Definition Format

```python
from typing import Callable, Dict, Any, get_type_hints
import inspect

class ToolDefinition:
    """Structured definition of an agent tool"""
    
    def __init__(self, func: Callable):
        """
        Create a tool definition from a function.
        
        The function should have:
        - Type annotations for all parameters
        - A docstring describing what it does
        """
        self.func = func
        self.name = func.__name__
        self.description = func.__doc__ or ""
        self._extract_parameters()
    
    def _extract_parameters(self):
        """Extract parameters from function signature"""
        sig = inspect.signature(self.func)
        self.parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = param.annotation
            self.parameters[param_name] = {
                'type': self._type_to_string(param_type),
                'description': self._extract_description(param_name),
                'required': param.default == inspect.Parameter.empty,
            }
    
    def _type_to_string(self, param_type) -> str:
        """Convert Python type to JSON schema type"""
        type_mapping = {
            str: 'string',
            int: 'integer',
            float: 'number',
            bool: 'boolean',
            list: 'array',
            dict: 'object',
        }
        return type_mapping.get(param_type, 'string')
    
    def _extract_description(self, param_name: str) -> str:
        """Extract parameter description from docstring"""
        # Parse docstring to find parameter documentation
        # (Implementation details omitted for brevity)
        return ""
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema for LLM"""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': self.parameters,
                    'required': [
                        name for name, info in self.parameters.items()
                        if info['required']
                    ]
                }
            }
        }
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments"""
        return await self.func(**kwargs) if inspect.iscoroutinefunction(self.func) else self.func(**kwargs)
```

### Message Types

```python
from dataclasses import dataclass
from typing import Any, List, Optional
from datetime import datetime

@dataclass
class AgentEvent:
    """Base class for all agent events"""
    source: str  # Agent name or "user"
    timestamp: float = field(default_factory=time.time)

@dataclass
class TextMessage(AgentEvent):
    """Text message between agents"""
    content: str
    models_used: Optional[List[str]] = None

@dataclass
class ToolCallMessage(AgentEvent):
    """Request to execute a tool"""
    tool_name: str
    tool_arguments: Dict[str, Any]
    message_id: str = field(default_factory=uuid.uuid4)

@dataclass
class ToolResultMessage(AgentEvent):
    """Result from tool execution"""
    tool_name: str
    tool_result: Any
    tool_use_id: str
    error: Optional[str] = None
    is_error: bool = False

@dataclass
class FunctionCallMessage(AgentEvent):
    """Legacy function call format"""
    content: str
    function_calls: List[Dict[str, Any]]
```

### Termination Conditions

```python
from abc import ABC, abstractmethod
from typing import List, Union

class TerminationCondition(ABC):
    """Base class for conversation termination conditions"""
    
    @abstractmethod
    def terminate(self, messages: List[AgentEvent]) -> bool:
        """
        Determine if the conversation should terminate.
        
        Args:
            messages: Current message history
            
        Returns:
            True if conversation should terminate
        """
        pass

class TextMentionTermination(TerminationCondition):
    """Terminate when specific text appears in message"""
    
    def __init__(self, text: str = "TERMINATE"):
        self.text = text
    
    def terminate(self, messages: List[AgentEvent]) -> bool:
        if not messages:
            return False
        last_message = messages[-1]
        return (
            isinstance(last_message, TextMessage) and
            self.text.lower() in last_message.content.lower()
        )

class MaxMessageTermination(TerminationCondition):
    """Terminate after maximum messages exchanged"""
    
    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
    
    def terminate(self, messages: List[AgentEvent]) -> bool:
        return len(messages) >= self.max_messages

class ComposedTermination(TerminationCondition):
    """Combine multiple termination conditions with OR logic"""
    
    def __init__(self, *conditions: TerminationCondition):
        self.conditions = conditions
    
    def terminate(self, messages: List[AgentEvent]) -> bool:
        return any(cond.terminate(messages) for cond in self.conditions)
```

---

## Configuration and Initialisation Patterns

### Configuration Files

#### YAML Configuration

```yaml
# config.yaml
version: "1.0"

# Model configurations
models:
  gpt4:
    type: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
    temperature: 0.7
    max_tokens: 2048
    top_p: 0.9
  
  azure_gpt4:
    type: azure_openai
    api_key: ${AZURE_OPENAI_API_KEY}
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    model: gpt-4o
    api_version: "2024-02-01"
  
  bedrock_claude:
    type: bedrock
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    region: us-east-1

# Agent configurations
agents:
  planning_agent:
    name: planner
    model: gpt4
    system_message: |
      You are a planning agent. Your job is to break down complex tasks
      into smaller, manageable subtasks.
    description: "Agent for breaking down and planning tasks"
    max_tool_use_attempts: 3
    reflect_on_tool_use: true
  
  code_agent:
    name: coder
    model: gpt4
    system_message: |
      You are an expert programmer. Write clean, tested, well-documented code.
    tools:
      - write_file
      - execute_code
      - run_tests
    description: "Agent for writing and testing code"
  
  research_agent:
    name: researcher
    model: gpt4
    system_message: |
      You are a research assistant. Find accurate, up-to-date information.
    tools:
      - web_search
      - read_url
    description: "Agent for research and information gathering"

# Team configurations
teams:
  research_team:
    type: selector_group_chat
    agents:
      - planning_agent
      - research_agent
      - code_agent
    termination:
      type: text_mention
      text: "TERMINATE"
    model_for_selection: gpt4
    max_rounds: 50
  
  code_team:
    type: round_robin_group_chat
    agents:
      - code_agent
    termination:
      type: max_messages
      max_messages: 30

# Tool configurations
tools:
  web_search:
    type: function
    function: search_web
    timeout: 10
    cache_results: true
  
  execute_code:
    type: code_execution
    work_dir: ./work
    use_docker: true
    timeout: 30
    allowed_imports:
      - numpy
      - pandas
      - requests
```

#### Loading Configuration

```python
from autogen_core.config import ConfigLoader
from autogen_agentchat.teams import SelectorGroupChat

# Load configuration
config = ConfigLoader.load("config.yaml")

# Access model configuration
gpt4_config = config.models["gpt4"]
model_client = gpt4_config.create_client()

# Create agents from configuration
agents = {}
for agent_name, agent_config in config.agents.items():
    agents[agent_name] = agent_config.create_agent(model_client)

# Create team from configuration
team_config = config.teams["research_team"]
team = team_config.create_team(agents, model_client)
```

### Programmatic Configuration

#### Builder Pattern

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination
)

# Configure model client
model_client = (OpenAIChatCompletionClient
    .builder()
    .with_model("gpt-4o")
    .with_temperature(0.7)
    .with_max_tokens(2048)
    .build())

# Create planning agent
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="""You are a planning agent.
    Break down complex tasks into smaller, manageable subtasks.
    Guide your team through systematic problem-solving.""",
    description="Breaks down tasks and coordinates team",
)

# Create research agent
researcher = AssistantAgent(
    name="researcher",
    model_client=model_client,
    system_message="""You are a research specialist.
    Find accurate, well-sourced information for tasks.""",
    description="Finds and verifies information",
    tools=[search_web, read_url],
)

# Create code agent
coder = AssistantAgent(
    name="coder",
    model_client=model_client,
    system_message="""You are an expert Python developer.
    Write clean, well-tested, documented code.""",
    description="Writes and tests Python code",
    tools=[write_file, execute_code, run_tests],
    reflect_on_tool_use=True,
)

# Configure termination conditions
termination = (
    TextMentionTermination("TERMINATE") |
    MaxMessageTermination(max_messages=100)
)

# Create team
team = SelectorGroupChat(
    agents=[planner, researcher, coder],
    model_client=model_client,
    termination_condition=termination,
)
```

#### Environment-Based Configuration

```python
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.azure import AzureOpenAIChatCompletionClient
from autogen_ext.models.bedrock import BedrockChatCompletionClient

def get_model_client(provider: str = None):
    """Get model client based on environment or parameter"""
    
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    
    if provider == "openai":
        return OpenAIChatCompletionClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("MODEL_TEMPERATURE", "0.7")),
        )
    
    elif provider == "azure":
        return AzureOpenAIChatCompletionClient(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
        )
    
    elif provider == "bedrock":
        return BedrockChatCompletionClient(
            model_id=os.getenv("BEDROCK_MODEL_ID"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

# Usage
model_client = get_model_client()
```

### Dependency Injection

```python
from dataclasses import dataclass
from typing import Protocol

class ModelClientProtocol(Protocol):
    """Protocol for model clients"""
    async def create_completion(self, messages: List[Dict]) -> str:
        pass

@dataclass
class AgentFactory:
    """Factory for creating configured agents"""
    
    model_client: ModelClientProtocol
    tools_registry: Dict[str, Callable]
    
    def create_assistant(
        self,
        name: str,
        system_message: str,
        tool_names: List[str] = None,
    ) -> AssistantAgent:
        """Create an assistant agent with dependency injection"""
        
        tools = []
        if tool_names:
            for tool_name in tool_names:
                if tool_name not in self.tools_registry:
                    raise ValueError(f"Unknown tool: {tool_name}")
                tools.append(self.tools_registry[tool_name])
        
        return AssistantAgent(
            name=name,
            model_client=self.model_client,
            system_message=system_message,
            tools=tools,
        )

# Usage
tools = {
    "web_search": search_web,
    "execute_code": execute_code,
    "read_file": read_file,
}

factory = AgentFactory(
    model_client=model_client,
    tools_registry=tools,
)

planner = factory.create_assistant(
    name="planner",
    system_message="You are a planning agent.",
)

researcher = factory.create_assistant(
    name="researcher",
    system_message="You are a research agent.",
    tool_names=["web_search"],
)
```

---

## Simple Agents

### Creating Basic Agents

The simplest agent in AutoGen 0.4 is an `AssistantAgent` that uses an LLM to respond to messages.

```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

async def main():
    # Step 1: Create a model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key="sk-...",  # or use environment variable
    )
    
    # Step 2: Create an agent
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )
    
    # Step 3: Send a message
    response = await assistant.on_messages([
        TextMessage(
            content="What is the capital of France?",
            source="user",
        )
    ])
    
    # Step 4: Print the response
    print(response)

# Run asynchronously
asyncio.run(main())
```

**Output:**
```
The capital of France is Paris. It's the largest city in France and is located in the north-central part of the country along the Seine River.
```

### Agent Specialisation

Agents can be specialised for specific domains by customising the system message.

```python
# Financial analyst agent
financial_analyst = AssistantAgent(
    name="financial_analyst",
    system_message="""You are a financial analyst with deep expertise in:
    - Stock market analysis
    - Financial statement interpretation
    - Risk assessment
    - Portfolio optimisation
    
    When analysing financial data, always:
    1. Consider multiple perspectives
    2. Reference relevant financial metrics
    3. Explain your reasoning clearly
    4. Highlight risks and opportunities""",
    model_client=model_client,
)

# Code reviewer agent
code_reviewer = AssistantAgent(
    name="code_reviewer",
    system_message="""You are an expert code reviewer. Evaluate code for:
    - Correctness and bugs
    - Performance and efficiency  
    - Security vulnerabilities
    - Code style and readability
    - Test coverage
    
    Provide specific, actionable feedback.""",
    model_client=model_client,
)

# Creative writer agent
creative_writer = AssistantAgent(
    name="creative_writer",
    system_message="""You are a creative writer specialising in:
    - Storytelling and narrative
    - Character development
    - Dialogue and voice
    - Worldbuilding
    - Emotional resonance
    
    Write engaging, vivid content.""",
    model_client=model_client,
)
```

### User Proxy Agent

A `UserProxyAgent` allows human input into the agent system.

```python
from autogen_agentchat.agents import UserProxyAgent

async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create assistant
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful Python programming tutor.",
    )
    
    # Create user proxy
    user = UserProxyAgent(
        name="user",
        input_func=input,  # Use standard input
    )
    
    # Interactive loop
    print("Chat with the Python tutor. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        
        response = await assistant.on_messages([
            TextMessage(content=user_input, source="user")
        ])
        print(f"Assistant: {response}\n")

asyncio.run(main())
```

---

## Agent Lifecycle Management

### Agent States

An agent transitions through several states during its lifecycle:

```
Created → Initialized → Running → Completed
                ↓          ↓
              Paused ← ← ← Error
```

**States:**
- **Created**: Agent instance created but not yet ready
- **Initialised**: Agent registered and ready to receive messages
- **Running**: Currently processing messages
- **Paused**: Temporarily suspended (e.g., waiting for human input)
- **Completed**: Task completed successfully
- **Error**: Error occurred during execution

### Lifecycle Hooks

```python
from typing import Callable, List

class AgentWithLifecycleHooks:
    """Agent with lifecycle event handlers"""
    
    def __init__(self):
        self.on_init_handlers: List[Callable] = []
        self.on_message_handlers: List[Callable] = []
        self.on_error_handlers: List[Callable] = []
        self.on_complete_handlers: List[Callable] = []
    
    def on_initialized(self, handler: Callable):
        """Register handler for agent initialization"""
        self.on_init_handlers.append(handler)
        return self
    
    def on_message(self, handler: Callable):
        """Register handler for message processing"""
        self.on_message_handlers.append(handler)
        return self
    
    def on_error(self, handler: Callable):
        """Register handler for errors"""
        self.on_error_handlers.append(handler)
        return self
    
    def on_complete(self, handler: Callable):
        """Register handler for completion"""
        self.on_complete_handlers.append(handler)
        return self
    
    async def _emit_init(self):
        for handler in self.on_init_handlers:
            await handler(self) if asyncio.iscoroutinefunction(handler) else handler(self)
    
    async def _emit_message(self, message):
        for handler in self.on_message_handlers:
            await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
    
    async def _emit_error(self, error):
        for handler in self.on_error_handlers:
            await handler(error) if asyncio.iscoroutinefunction(handler) else handler(error)
    
    async def _emit_complete(self, result):
        for handler in self.on_complete_handlers:
            await handler(result) if asyncio.iscoroutinefunction(handler) else handler(result)

# Usage
agent = AssistantAgent(name="assistant", model_client=model_client)

async def log_init(agent):
    print(f"Agent {agent.name} initialized")

async def log_message(message):
    print(f"Message: {message}")

async def log_error(error):
    print(f"Error: {error}")

agent.on_initialized(log_init)
agent.on_message(log_message)
agent.on_error(log_error)
```

### Resource Cleanup

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_agent(name: str, model_client) -> AssistantAgent:
    """Context manager for agent lifecycle"""
    
    # Setup
    agent = AssistantAgent(
        name=name,
        model_client=model_client,
        system_message="You are helpful."
    )
    
    try:
        yield agent
    
    finally:
        # Cleanup
        print(f"Cleaning up agent {name}")
        # Close any open connections
        if hasattr(agent, 'close'):
            await agent.close()

# Usage
async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    async with managed_agent("assistant", model_client) as agent:
        response = await agent.on_messages([
            TextMessage(content="Hello", source="user")
        ])
        print(response)

asyncio.run(main())
```

---

## Event-Driven Agent Design

### Event System

AutoGen 0.4 has a built-in event system for observability and state tracking.

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any
import asyncio

@dataclass
class Event(ABC):
    """Base class for all events"""
    source_id: str
    timestamp: float
    
    @abstractmethod
    def __str__(self) -> str:
        pass

@dataclass  
class AgentStartedEvent(Event):
    """Fired when an agent starts processing"""
    agent_name: str
    
    def __str__(self) -> str:
        return f"Agent {self.agent_name} started"

@dataclass
class MessageReceivedEvent(Event):
    """Fired when an agent receives a message"""
    message: str
    sender: str
    
    def __str__(self) -> str:
        return f"Message from {self.sender}: {self.message[:50]}..."

@dataclass
class ToolInvokedEvent(Event):
    """Fired when a tool is invoked"""
    tool_name: str
    arguments: Dict[str, Any]
    
    def __str__(self) -> str:
        return f"Tool {self.tool_name} invoked with args: {self.arguments}"

@dataclass
class ToolCompletedEvent(Event):
    """Fired when a tool completes"""
    tool_name: str
    result: Any
    duration_ms: float
    
    def __str__(self) -> str:
        return f"Tool {self.tool_name} completed in {self.duration_ms}ms"

@dataclass
class MessageSentEvent(Event):
    """Fired when an agent sends a message"""
    message: str
    recipient: str
    
    def __str__(self) -> str:
        return f"Message sent to {self.recipient}"

@dataclass
class ErrorEvent(Event):
    """Fired when an error occurs"""
    error: Exception
    context: str
    
    def __str__(self) -> str:
        return f"Error: {self.error} (Context: {self.context})"

@dataclass
class AgentCompletedEvent(Event):
    """Fired when an agent completes"""
    agent_name: str
    result: str
    duration_ms: float
    
    def __str__(self) -> str:
        return f"Agent {self.agent_name} completed in {self.duration_ms}ms"

class EventBus:
    """Central event dispatcher"""
    
    def __init__(self):
        self.handlers: Dict[type, List[Callable]] = {}
        self.event_history: List[Event] = []
    
    def subscribe(self, event_type: type, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: type, handler: Callable):
        """Unsubscribe from an event type"""
        if event_type in self.handlers:
            self.handlers[event_type].remove(handler)
    
    async def emit(self, event: Event):
        """Emit an event"""
        self.event_history.append(event)
        
        if type(event) in self.handlers:
            for handler in self.handlers[type(event)]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
    
    def get_events(self, event_type: type = None) -> List[Event]:
        """Get event history"""
        if event_type is None:
            return self.event_history
        return [e for e in self.event_history if isinstance(e, event_type)]

# Usage
event_bus = EventBus()

def log_message_received(event: MessageReceivedEvent):
    print(f"📨 {event}")

def log_tool_invoked(event: ToolInvokedEvent):
    print(f"🔧 {event}")

def log_error(event: ErrorEvent):
    print(f"⚠️ {event}")

event_bus.subscribe(MessageReceivedEvent, log_message_received)
event_bus.subscribe(ToolInvokedEvent, log_tool_invoked)
event_bus.subscribe(ErrorEvent, log_error)

# Emit events
await event_bus.emit(MessageReceivedEvent(
    source_id="user",
    timestamp=time.time(),
    message="Hello, assistant",
    sender="user"
))
```

### Event-Driven Agent

```python
import time
from autogen_agentchat.agents import AssistantAgent

class ObservableAgent(AssistantAgent):
    """Agent that emits events for all operations"""
    
    def __init__(self, *args, event_bus: EventBus = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = event_bus or EventBus()
    
    async def on_messages(self, messages, cancellation_token=None):
        start_time = time.time()
        
        try:
            # Emit start event
            await self.event_bus.emit(AgentStartedEvent(
                source_id=self.name,
                timestamp=time.time(),
                agent_name=self.name,
            ))
            
            # Process each message
            for message in messages:
                await self.event_bus.emit(MessageReceivedEvent(
                    source_id=self.name,
                    timestamp=time.time(),
                    message=message.content if hasattr(message, 'content') else str(message),
                    sender=message.source if hasattr(message, 'source') else "unknown",
                ))
            
            # Call parent implementation
            result = await super().on_messages(messages, cancellation_token)
            
            # Emit completion event
            duration_ms = (time.time() - start_time) * 1000
            await self.event_bus.emit(AgentCompletedEvent(
                source_id=self.name,
                timestamp=time.time(),
                agent_name=self.name,
                result=str(result),
                duration_ms=duration_ms,
            ))
            
            return result
        
        except Exception as e:
            await self.event_bus.emit(ErrorEvent(
                source_id=self.name,
                timestamp=time.time(),
                error=e,
                context="Message processing",
            ))
            raise

# Usage
async def main():
    event_bus = EventBus()
    
    # Subscribe to events
    event_bus.subscribe(MessageReceivedEvent, lambda e: print(f"📨 {e}"))
    event_bus.subscribe(AgentCompletedEvent, lambda e: print(f"✅ {e}"))
    event_bus.subscribe(ErrorEvent, lambda e: print(f"⚠️ {e}"))
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = ObservableAgent(
        name="assistant",
        model_client=model_client,
        event_bus=event_bus,
    )
    
    await agent.on_messages([
        TextMessage(content="What is AI?", source="user")
    ])

asyncio.run(main())
```

---

## Message Passing Mechanisms

### Message Types and Structure

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

@dataclass
class BaseMessage:
    """Base message class"""
    content: str
    source: str  # Agent name or "user"
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TextMessage(BaseMessage):
    """Simple text message"""
    models_used: Optional[List[str]] = None
    
    def __str__(self) -> str:
        return f"{self.source}: {self.content}"

@dataclass
class ToolCallMessage(BaseMessage):
    """Message requesting tool execution"""
    tool_name: str
    tool_arguments: Dict[str, Any]
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ToolResultMessage(BaseMessage):
    """Message containing tool result"""
    tool_name: str
    tool_result: Any
    tool_use_id: str
    error: Optional[str] = None
    is_error: bool = False

@dataclass
class SystemMessage(BaseMessage):
    """System-level message (not from user/agent)"""
    message_type: str  # e.g., "agent_start", "agent_stop"

@dataclass
class MultimodalMessage(BaseMessage):
    """Message with multiple content types"""
    images: Optional[List[bytes]] = None
    documents: Optional[List[Dict[str, Any]]] = None
    
    def add_image(self, image_data: bytes, image_type: str = "png"):
        if self.images is None:
            self.images = []
        self.images.append(image_data)

@dataclass
class FunctionCallMessage(BaseMessage):
    """Legacy function call message (for compatibility)"""
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
```

### Message Routing

```python
from typing import Callable, Dict, List
import asyncio

class MessageRouter:
    """Routes messages between agents"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.routing_rules: List[Callable] = []
    
    def register_agent(self, agent_name: str, agent):
        """Register an agent"""
        self.agents[agent_name] = agent
    
    def add_routing_rule(self, rule: Callable[[BaseMessage], str]):
        """Add a custom routing rule"""
        self.routing_rules.append(rule)
    
    async def route_message(self, message: BaseMessage, target_agent: str):
        """Route a message to a specific agent"""
        if target_agent not in self.agents:
            raise ValueError(f"Agent {target_agent} not registered")
        
        agent = self.agents[target_agent]
        response = await agent.on_messages([message])
        return response
    
    async def determine_next_agent(self, message: BaseMessage, history: List[BaseMessage]) -> str:
        """Determine which agent should receive the message"""
        # Apply custom routing rules
        for rule in self.routing_rules:
            target = rule(message, history)
            if target:
                return target
        
        # Default: go to first available agent that can handle it
        return list(self.agents.keys())[0]
    
    async def route_with_rules(self, message: BaseMessage, history: List[BaseMessage]):
        """Route message using rules"""
        target = await self.determine_next_agent(message, history)
        return await self.route_message(message, target)

# Usage
router = MessageRouter()
router.register_agent("planner", planner_agent)
router.register_agent("researcher", researcher_agent)

# Custom routing rule: if message mentions "code", go to coder
def route_to_coder(message, history):
    if isinstance(message, TextMessage):
        if "code" in message.content.lower() or "python" in message.content.lower():
            if "coder" in router.agents:
                return "coder"
    return None

router.add_routing_rule(route_to_coder)

# Route a message
response = await router.route_with_rules(
    TextMessage(content="Write Python code to sort a list", source="user"),
    []
)
```

### Async Message Queue

```python
import asyncio
from collections import deque
from typing import Callable, Optional

class AsyncMessageQueue:
    """Async message queue for agent communication"""
    
    def __init__(self, max_size: int = 0):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.processors: Dict[str, Callable] = {}
        self.running = False
    
    async def enqueue(self, message: BaseMessage, priority: int = 0):
        """Add message to queue"""
        await self.queue.put((priority, message))
    
    def register_processor(self, message_type: type, processor: Callable):
        """Register a processor for a message type"""
        type_name = message_type.__name__
        self.processors[type_name] = processor
    
    async def process(self):
        """Process messages from queue"""
        self.running = True
        while self.running:
            try:
                priority, message = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                type_name = type(message).__name__
                if type_name in self.processors:
                    processor = self.processors[type_name]
                    await processor(message)
                
                self.queue.task_done()
            
            except asyncio.TimeoutError:
                continue
    
    def stop(self):
        """Stop processing"""
        self.running = False

# Usage
queue = AsyncMessageQueue()

async def process_text_message(message: TextMessage):
    print(f"Processing: {message}")

queue.register_processor(TextMessage, process_text_message)

# Start processing
processor_task = asyncio.create_task(queue.process())

# Enqueue messages
await queue.enqueue(TextMessage(content="Hello", source="user"))
await queue.enqueue(TextMessage(content="How are you?", source="user"))

# Wait for completion
await queue.queue.join()
queue.stop()
```

---

## Single-Agent Task Execution

### Simple Task Execution

```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

async def execute_single_task(task: str):
    """Execute a single task with one agent"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = AssistantAgent(
        name="task_executor",
        system_message="You are an efficient task executor. Complete tasks accurately and concisely.",
        model_client=model_client,
    )
    
    # Execute task
    response = await agent.on_messages([
        TextMessage(content=task, source="user")
    ])
    
    return response

# Usage
async def main():
    tasks = [
        "Explain quantum computing in 2 sentences",
        "Write a Python function to calculate Fibonacci",
        "List 5 best practices for REST API design",
    ]
    
    for task in tasks:
        print(f"\nTask: {task}")
        result = await execute_single_task(task)
        print(f"Result: {result}\n")

asyncio.run(main())
```

### Task Chains

```python
async def execute_task_chain(initial_task: str, follow_up_tasks: List[str]):
    """Execute a chain of tasks where each depends on previous"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = AssistantAgent(
        name="chain_executor",
        model_client=model_client,
    )
    
    # Execute initial task
    print(f"Initial task: {initial_task}")
    response = await agent.on_messages([
        TextMessage(content=initial_task, source="user")
    ])
    print(f"Response: {response}\n")
    
    # Execute follow-up tasks based on previous response
    messages = [TextMessage(content=initial_task, source="user")]
    
    for follow_up in follow_up_tasks:
        # Add previous response to context
        messages.append(TextMessage(content=response, source="assistant"))
        
        # Add follow-up task
        messages.append(TextMessage(content=follow_up, source="user"))
        
        print(f"Follow-up task: {follow_up}")
        response = await agent.on_messages(messages[-1:])
        print(f"Response: {response}\n")
    
    return response

# Usage
async def main():
    initial = "Explain the concept of machine learning"
    follow_ups = [
        "Now explain how neural networks work",
        "What is deep learning and how does it relate to neural networks?",
    ]
    
    result = await execute_task_chain(initial, follow_ups)
    print(f"Final result: {result}")

asyncio.run(main())
```

### Task with Tool Use

```python
import math

def calculate_sum(numbers: list[float]) -> float:
    """Calculate the sum of numbers"""
    return sum(numbers)

def calculate_average(numbers: list[float]) -> float:
    """Calculate the average of numbers"""
    return sum(numbers) / len(numbers) if numbers else 0

def calculate_std_dev(numbers: list[float]) -> float:
    """Calculate the standard deviation"""
    if len(numbers) < 2:
        return 0
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return math.sqrt(variance)

async def execute_task_with_tools(task: str):
    """Execute a task where agent can use computational tools"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = AssistantAgent(
        name="calculator_agent",
        system_message="""You are a data analysis agent.
        Use the available tools to answer questions about data.
        When given numbers, use the tools to calculate statistics.""",
        model_client=model_client,
        tools=[calculate_sum, calculate_average, calculate_std_dev],
    )
    
    response = await agent.on_messages([
        TextMessage(content=task, source="user")
    ])
    
    return response

# Usage
async def main():
    task = "What are the sum, average, and standard deviation of: 5, 10, 15, 20, 25?"
    result = await execute_task_with_tools(task)
    print(f"Result: {result}")

asyncio.run(main())
```

---

## Agent Capabilities and Roles

### Role-Based Agents

```python
from enum import Enum
from typing import List

class AgentRole(Enum):
    """Predefined agent roles"""
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    PROGRAMMER = "programmer"
    WRITER = "writer"
    MANAGER = "manager"
    CRITIC = "critic"

class RoleConfig:
    """Configuration for a role"""
    
    def __init__(
        self,
        role: AgentRole,
        system_message: str,
        tools: List[Callable] = None,
        temperature: float = 0.7,
    ):
        self.role = role
        self.system_message = system_message
        self.tools = tools or []
        self.temperature = temperature

ROLE_CONFIGS = {
    AgentRole.ANALYST: RoleConfig(
        AgentRole.ANALYST,
        """You are a data analyst. Your responsibilities:
        - Analyse data and identify patterns
        - Calculate statistics and metrics
        - Create visualisations
        - Provide data-driven insights""",
        tools=[calculate_sum, calculate_average],
    ),
    
    AgentRole.RESEARCHER: RoleConfig(
        AgentRole.RESEARCHER,
        """You are a researcher. Your responsibilities:
        - Find authoritative sources
        - Verify facts and claims
        - Synthesise information
        - Provide well-cited responses""",
        tools=[search_web, read_url],
    ),
    
    AgentRole.PROGRAMMER: RoleConfig(
        AgentRole.PROGRAMMER,
        """You are an expert Python programmer. Your responsibilities:
        - Write clean, efficient code
        - Follow best practices
        - Include error handling
        - Provide documentation""",
        tools=[write_file, execute_code],
    ),
    
    AgentRole.WRITER: RoleConfig(
        AgentRole.WRITER,
        """You are a creative writer. Your responsibilities:
        - Craft engaging content
        - Develop compelling narratives
        - Use vivid language
        - Maintain consistency""",
    ),
    
    AgentRole.MANAGER: RoleConfig(
        AgentRole.MANAGER,
        """You are a project manager. Your responsibilities:
        - Coordinate team activities
        - Ensure deadlines are met
        - Facilitate communication
        - Track progress""",
    ),
    
    AgentRole.CRITIC: RoleConfig(
        AgentRole.CRITIC,
        """You are a critical analyst. Your responsibilities:
        - Evaluate ideas objectively
        - Identify weaknesses and risks
        - Propose improvements
        - Challenge assumptions""",
    ),
}

def create_agent_with_role(
    name: str,
    role: AgentRole,
    model_client
) -> AssistantAgent:
    """Create an agent with a predefined role"""
    
    config = ROLE_CONFIGS[role]
    
    return AssistantAgent(
        name=name,
        model_client=model_client,
        system_message=config.system_message,
        tools=config.tools,
    )

# Usage
async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    analyst = create_agent_with_role("alice", AgentRole.ANALYST, model_client)
    researcher = create_agent_with_role("bob", AgentRole.RESEARCHER, model_client)
    programmer = create_agent_with_role("charlie", AgentRole.PROGRAMMER, model_client)
    
    # Use agents for their specialized tasks
    analysis = await analyst.on_messages([
        TextMessage(content="Analyze this data: [1,2,3,4,5]", source="user")
    ])

asyncio.run(main())
```

### Dynamic Capability Assignment

```python
class CapabilityPool:
    """Manages available capabilities/tools"""
    
    def __init__(self):
        self.capabilities = {}
    
    def register(self, name: str, tool: Callable):
        """Register a capability"""
        self.capabilities[name] = tool
    
    def get_capability(self, name: str) -> Callable:
        """Get a specific capability"""
        return self.capabilities.get(name)
    
    def get_capabilities_by_tags(self, tags: List[str]) -> List[Callable]:
        """Get capabilities matching tags"""
        # Implementation would handle tag-based lookup
        pass

class AdaptiveAgent(AssistantAgent):
    """Agent that can adapt its capabilities dynamically"""
    
    def __init__(self, *args, capability_pool: CapabilityPool, **kwargs):
        super().__init__(*args, **kwargs)
        self.capability_pool = capability_pool
        self.active_capabilities = []
    
    def request_capability(self, capability_name: str):
        """Request a capability"""
        capability = self.capability_pool.get_capability(capability_name)
        if capability and capability not in self.active_capabilities:
            self.active_capabilities.append(capability)
            # Update agent's tools
            self.tools.append(capability)
    
    def release_capability(self, capability_name: str):
        """Release a capability when no longer needed"""
        capability = self.capability_pool.get_capability(capability_name)
        if capability in self.active_capabilities:
            self.active_capabilities.remove(capability)
            self.tools.remove(capability)
```

---

## Multi-Agent Systems

The previous sections covered single agents extensively. Now we'll explore building systems of multiple agents working together.

### Multi-Agent Architectures

AutoGen supports several multi-agent patterns:

1. **Sequential**: Agents process tasks one after another
2. **Hierarchical**: Manager agent delegates to specialist agents
3. **Collaborative**: Agents work together without strict ordering
4. **Competitive**: Agents propose solutions and best is selected
5. **Swarm**: Many simple agents coordinate via emergent behaviour

### Group Chat

```python
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

async def run_group_chat():
    """Run a multi-agent group conversation"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create specialist agents
    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message="""You are a planning specialist.
        Your job is to break down complex tasks and coordinate the team.
        When given a task, create a step-by-step plan.""",
        description="Breaks down tasks and plans solutions",
    )
    
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message="""You are a research specialist.
        Your job is to find and verify information.
        Use your tools to search for accurate, up-to-date information.""",
        description="Finds and verifies information",
        tools=[search_web],
    )
    
    analyst = AssistantAgent(
        name="analyst",
        model_client=model_client,
        system_message="""You are a data analyst.
        Your job is to interpret information and draw conclusions.
        Provide statistical analysis and insights.""",
        description="Analyses data and draws conclusions",
        tools=[calculate_average, calculate_std_dev],
    )
    
    # Create team with LLM-based speaker selection
    team = SelectorGroupChat(
        agents=[planner, researcher, analyst],
        model_client=model_client,
        termination_condition=(
            TextMentionTermination("TERMINATE") |
            MaxMessageTermination(max_messages=50)
        ),
    )
    
    # Run the group chat
    task = """Research the latest trends in AI, analyze their impact, and present findings.
    When complete, summarize with TERMINATE."""
    
    result = await team.run(task)
    
    return result

asyncio.run(run_group_chat())
```

### Hierarchical Teams

```python
class ManagerAgent(AssistantAgent):
    """Manager agent that delegates to specialists"""
    
    def __init__(self, specialists: Dict[str, AssistantAgent], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specialists = specialists
    
    async def delegate(self, task: str, specialist_name: str) -> str:
        """Delegate a task to a specialist"""
        if specialist_name not in self.specialists:
            raise ValueError(f"Unknown specialist: {specialist_name}")
        
        specialist = self.specialists[specialist_name]
        response = await specialist.on_messages([
            TextMessage(content=task, source="manager")
        ])
        
        return response

async def run_hierarchical_team():
    """Run a hierarchical multi-agent system"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create specialist agents
    specialists = {
        "writer": AssistantAgent(
            name="writer",
            model_client=model_client,
            system_message="You write clear, engaging content.",
        ),
        "reviewer": AssistantAgent(
            name="reviewer",
            model_client=model_client,
            system_message="You review content for quality and accuracy.",
        ),
        "editor": AssistantAgent(
            name="editor",
            model_client=model_client,
            system_message="You edit content for grammar and style.",
        ),
    }
    
    # Create manager
    manager = ManagerAgent(
        specialists=specialists,
        name="manager",
        model_client=model_client,
        system_message="You coordinate specialists to produce quality work.",
    )
    
    # Delegate tasks
    draft = await manager.delegate("Write an article about AI", "writer")
    review = await manager.delegate(f"Review: {draft}", "reviewer")
    final = await manager.delegate(f"Edit: {review}", "editor")
    
    return final

asyncio.run(run_hierarchical_team())
```

---

## Distributed Agent Architecture

### Agent Runtime and Registration

In distributed systems, agents register with a central runtime:

```python
from autogen_core import SingleThreadedAgentRuntime, AgentId, RoutedAgent, MessageContext

class DistributedAgentRuntime:
    """Manages distributed agents"""
    
    def __init__(self):
        self.runtime = SingleThreadedAgentRuntime()
        self.agents = {}
    
    def register_agent(self, agent_id: str, agent: RoutedAgent):
        """Register an agent with the runtime"""
        self.agents[agent_id] = agent
        # In a real distributed system, this would register with a service registry
    
    def get_agent(self, agent_id: str) -> RoutedAgent:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    async def send_message(self, from_id: str, to_id: str, message: str):
        """Send a message between agents"""
        target = self.get_agent(to_id)
        if target is None:
            raise ValueError(f"Agent {to_id} not found")
        
        await target.on_message(message, from_id)

# Custom routed agent with message handlers
class CustomRoutedAgent(RoutedAgent):
    """Agent that routes messages to handler methods"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_history = []
    
    @message_handler
    async def on_message(self, message: str, sender: str):
        """Handle incoming messages"""
        self.message_history.append({
            'sender': sender,
            'content': message,
            'timestamp': time.time(),
        })
        
        # Respond based on message content
        if "hello" in message.lower():
            return f"Hello from {self.agent_id}!"
        return f"{self.agent_id} received: {message}"

# Usage
async def main():
    runtime = DistributedAgentRuntime()
    
    agent1 = CustomRoutedAgent("agent-1")
    agent2 = CustomRoutedAgent("agent-2")
    
    runtime.register_agent("agent-1", agent1)
    runtime.register_agent("agent-2", agent2)
    
    # Send messages between agents
    response = await runtime.send_message("agent-1", "agent-2", "Hello agent-2!")
    print(response)

asyncio.run(main())
```

### Topic-Based Routing

In AutoGen, agents can subscribe to topics and receive messages automatically:

```python
from autogen_core import TopicId, default_subscription

class TopicAwareAgent(RoutedAgent):
    """Agent that subscribes to topics"""
    
    def __init__(self, agent_id: str, topics: List[str]):
        self.agent_id = agent_id
        self.topics = [TopicId(topic) for topic in topics]
    
    @message_handler
    @default_subscription
    async def on_topic_message(self, message: str, ctx: MessageContext):
        """Handle messages from subscribed topics"""
        print(f"[{self.agent_id}] Received on {ctx.topic}: {message}")
        
        # Broadcast response to same topic
        await ctx.send_message(f"Response from {self.agent_id}", TopicId("responses"))

# Usage
async def main():
    runtime = SingleThreadedAgentRuntime()
    
    # Create agents that subscribe to topics
    agent1 = TopicAwareAgent("agent-1", ["weather", "alerts"])
    agent2 = TopicAwareAgent("agent-2", ["weather"])
    
    # Publish to topics
    await runtime.send_message(
        "sensor",
        TopicId("weather"),
        "Temperature: 72F"
    )
    
    # All subscribed agents receive the message

asyncio.run(main())
```

---

## Inter-Agent Communication Protocols

### Request-Response Pattern

```python
import uuid
from typing import Dict

class RequestResponseProtocol:
    """Implements request-response communication pattern"""
    
    def __init__(self):
        self.pending_responses: Dict[str, asyncio.Future] = {}
    
    async def request(self, source: str, target: str, request: Dict) -> Dict:
        """Send a request and wait for response"""
        request_id = str(uuid.uuid4())
        request['request_id'] = request_id
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[request_id] = response_future
        
        # Send request
        # (Implementation would route to target agent)
        
        # Wait for response (with timeout)
        try:
            response = await asyncio.wait_for(response_future, timeout=30)
            return response
        finally:
            del self.pending_responses[request_id]
    
    def handle_response(self, request_id: str, response: Dict):
        """Handle incoming response"""
        if request_id in self.pending_responses:
            future = self.pending_responses[request_id]
            if not future.done():
                future.set_result(response)

# Usage
protocol = RequestResponseProtocol()

async def agent_a():
    """Agent that makes requests"""
    request = {"task": "calculate", "operation": "sum", "numbers": [1,2,3]}
    response = await protocol.request("agent-a", "agent-b", request)
    print(f"Response: {response}")

async def agent_b():
    """Agent that responds to requests"""
    # Simulate responding to requests
    response = {"result": 6, "request_id": "..."}
    protocol.handle_response(response['request_id'], response)
```

### Publish-Subscribe Pattern

```python
from typing import Set, Callable, List

class PubSubBroker:
    """Implements publish-subscribe messaging"""
    
    def __init__(self):
        self.subscribers: Dict[str, Set[Callable]] = {}
    
    def subscribe(self, channel: str, handler: Callable):
        """Subscribe to a channel"""
        if channel not in self.subscribers:
            self.subscribers[channel] = set()
        self.subscribers[channel].add(handler)
    
    def unsubscribe(self, channel: str, handler: Callable):
        """Unsubscribe from a channel"""
        if channel in self.subscribers:
            self.subscribers[channel].discard(handler)
    
    async def publish(self, channel: str, message: Any):
        """Publish a message to a channel"""
        if channel in self.subscribers:
            tasks = []
            for handler in self.subscribers[channel]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(message))
                else:
                    handler(message)
            
            if tasks:
                await asyncio.gather(*tasks)

# Usage
broker = PubSubBroker()

async def on_weather_update(message: Dict):
    print(f"Weather update: {message}")

async def on_alert(message: Dict):
    print(f"Alert: {message}")

broker.subscribe("weather", on_weather_update)
broker.subscribe("alerts", on_alert)

# Publish messages
await broker.publish("weather", {"temp": 72, "condition": "sunny"})
await broker.publish("alerts", {"type": "warning", "message": "High UV index"})
```

---

## Team and Group Structures

### Team Composition

```python
from dataclasses import dataclass
from typing import List

@dataclass
class TeamConfig:
    """Configuration for a team"""
    name: str
    description: str
    members: List[AssistantAgent]
    lead: Optional[AssistantAgent] = None
    objectives: List[str] = None
    constraints: List[str] = None

class Team:
    """Represents a team of agents"""
    
    def __init__(self, config: TeamConfig):
        self.config = config
        self.members = config.members
        self.lead = config.lead or self.members[0]
        self.shared_memory = {}
        self.task_history = []
    
    async def assign_task(self, task: str):
        """Assign a task to the team"""
        # Lead agent breaks down task and assigns to team members
        await self.lead.on_messages([
            TextMessage(
                content=f"Team task: {task}\nTeam members: {[m.name for m in self.members]}",
                source="dispatcher"
            )
        ])
        
        self.task_history.append({
            'task': task,
            'timestamp': time.time(),
            'status': 'assigned',
        })
    
    def share_memory(self, key: str, value: Any):
        """Store information in shared team memory"""
        self.shared_memory[key] = value
    
    def get_shared_memory(self, key: str) -> Any:
        """Retrieve information from shared team memory"""
        return self.shared_memory.get(key)

# Example team configurations
async def create_research_team():
    """Create a research team"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    members = [
        AssistantAgent(
            name="lead_researcher",
            model_client=model_client,
            system_message="You lead the research team.",
        ),
        AssistantAgent(
            name="literature_reviewer",
            model_client=model_client,
            system_message="You review academic literature.",
            tools=[search_academic_database],
        ),
        AssistantAgent(
            name="data_analyst",
            model_client=model_client,
            system_message="You analyse research data.",
            tools=[calculate_statistics],
        ),
    ]
    
    team = Team(TeamConfig(
        name="Research Team",
        description="Conducts research on assigned topics",
        members=members,
        objectives=[
            "Find relevant research",
            "Analyse data",
            "Produce report",
        ],
    ))
    
    return team
```

---

## Orchestration Patterns

### Orchestration Strategies

AutoGen provides different orchestration strategies for multi-agent coordination:

```python
from enum import Enum
from abc import ABC, abstractmethod

class OrchestrationStrategy(ABC):
    """Base class for orchestration strategies"""
    
    @abstractmethod
    async def orchestrate(self, agents: List[AssistantAgent], task: str) -> str:
        """Orchestrate agents to complete a task"""
        pass

class SequentialOrchestration(OrchestrationStrategy):
    """Execute agents sequentially"""
    
    async def orchestrate(self, agents: List[AssistantAgent], task: str) -> str:
        result = task
        
        for agent in agents:
            response = await agent.on_messages([
                TextMessage(content=result, source="orchestrator")
            ])
            result = response
        
        return result

class ParallelOrchestration(OrchestrationStrategy):
    """Execute agents in parallel"""
    
    async def orchestrate(self, agents: List[AssistantAgent], task: str) -> str:
        # Send same task to all agents concurrently
        tasks = [
            agent.on_messages([TextMessage(content=task, source="orchestrator")])
            for agent in agents
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Synthesise results
        synthesis_prompt = f"""Review these perspectives on the task:
        {chr(10).join(results)}
        
        Provide a synthesised answer."""
        
        # Use first agent to synthesise
        synthesis = await agents[0].on_messages([
            TextMessage(content=synthesis_prompt, source="orchestrator")
        ])
        
        return synthesis

class HierarchicalOrchestration(OrchestrationStrategy):
    """Execute agents hierarchically"""
    
    async def orchestrate(self, agents: List[AssistantAgent], task: str) -> str:
        # First agent plans
        plan = await agents[0].on_messages([
            TextMessage(content=f"Plan this task: {task}", source="orchestrator")
        ])
        
        # Middle agents execute
        results = []
        for agent in agents[1:-1]:
            result = await agent.on_messages([
                TextMessage(
                    content=f"Execute this part of the plan: {plan}",
                    source="orchestrator"
                )
            ])
            results.append(result)
        
        # Last agent synthesises
        synthesis = await agents[-1].on_messages([
            TextMessage(
                content=f"Synthesise these results: {results}",
                source="orchestrator"
            )
        ])
        
        return synthesis

# Usage
async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agents = [
        AssistantAgent(name="agent1", model_client=model_client),
        AssistantAgent(name="agent2", model_client=model_client),
        AssistantAgent(name="agent3", model_client=model_client),
    ]
    
    task = "Write a comprehensive guide to machine learning"
    
    # Try different strategies
    strategies = [
        SequentialOrchestration(),
        ParallelOrchestration(),
        HierarchicalOrchestration(),
    ]
    
    for strategy in strategies:
        result = await strategy.orchestrate(agents, task)
        print(f"\nStrategy: {strategy.__class__.__name__}")
        print(f"Result: {result[:100]}...")

asyncio.run(main())
```

---

## Scalability and Load Balancing

### Load Balancing Strategies

```python
from typing import List, Optional
import heapq

class LoadBalancer:
    """Balances tasks across agents"""
    
    def __init__(self):
        self.agent_loads: Dict[str, float] = {}
    
    def register_agent(self, agent_id: str):
        """Register an agent for load balancing"""
        self.agent_loads[agent_id] = 0.0
    
    def update_load(self, agent_id: str, load: float):
        """Update agent's current load"""
        if agent_id in self.agent_loads:
            self.agent_loads[agent_id] = load
    
    def get_least_loaded_agent(self) -> str:
        """Get the least loaded agent"""
        if not self.agent_loads:
            raise ValueError("No agents registered")
        
        return min(self.agent_loads, key=self.agent_loads.get)
    
    def distribute_task(self, task: str) -> str:
        """Distribute task to least loaded agent"""
        agent_id = self.get_least_loaded_agent()
        
        # Simulate task assignment
        estimated_load = len(task) / 100  # Simple heuristic
        self.update_load(agent_id, self.agent_loads[agent_id] + estimated_load)
        
        return agent_id

# Usage
async def load_balanced_execution():
    """Execute tasks with load balancing"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agents = {
        "agent-1": AssistantAgent(name="agent-1", model_client=model_client),
        "agent-2": AssistantAgent(name="agent-2", model_client=model_client),
        "agent-3": AssistantAgent(name="agent-3", model_client=model_client),
    }
    
    balancer = LoadBalancer()
    for agent_id in agents:
        balancer.register_agent(agent_id)
    
    tasks = [
        "Write a blog post about AI",
        "Code a Python function",
        "Analyse this dataset",
    ]
    
    for task in tasks:
        agent_id = balancer.distribute_task(task)
        agent = agents[agent_id]
        
        result = await agent.on_messages([
            TextMessage(content=task, source="load_balancer")
        ])
        
        print(f"Task executed by {agent_id}: {task[:30]}...")

asyncio.run(load_balanced_execution())
```

---

## Tools Integration

### Tool Definition and Registration

```python
from typing import get_type_hints, Annotated
import inspect

class Tool:
    """Wrapper for a tool function"""
    
    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__
        self.description = func.__doc__ or ""
        self._extract_schema()
    
    def _extract_schema(self):
        """Extract JSON schema from function signature"""
        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func)
        
        self.parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = hints.get(param_name, str)
            
            # Get parameter description if available
            description = ""
            if hasattr(param.annotation, '__metadata__'):
                description = param.annotation.__metadata__[0]
            
            self.parameters[param_name] = {
                "type": self._get_json_type(param_type),
                "description": description,
                "required": param.default == inspect.Parameter.empty,
            }
    
    def _get_json_type(self, python_type) -> str:
        """Convert Python type to JSON schema type"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(python_type, "string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to tool dictionary for LLM"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [
                        k for k, v in self.parameters.items()
                        if v["required"]
                    ],
                },
            },
        }
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        # Call function and handle both sync and async
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)

# Define tools with annotations
def get_weather(
    city: Annotated[str, "The city name"],
    unit: Annotated[str, "Temperature unit (C or F)"] = "C"
) -> str:
    """Get weather for a city"""
    # Mock implementation
    return f"Weather in {city}: 72{unit}"

def calculate_distance(
    lat1: Annotated[float, "First latitude"],
    lon1: Annotated[float, "First longitude"],
    lat2: Annotated[float, "Second latitude"],
    lon2: Annotated[float, "Second longitude"]
) -> float:
    """Calculate distance between coordinates in kilometers"""
    import math
    
    R = 6371  # Earth's radius
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

# Create agents with tools
async def main():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[get_weather, calculate_distance],
    )
    
    response = await agent.on_messages([
        TextMessage(content="What's the weather in Paris?", source="user")
    ])
    
    print(response)

asyncio.run(main())
```

---

(Due to token limitations, I'll continue with the remaining sections in the next part. Let me create the comprehensive guide file now with what we have so far, then create the other documents.)

---

**[DOCUMENT CONTINUES...]**

The comprehensive guide has been started. Let me now create the diagrams, production guide, and recipes documents:


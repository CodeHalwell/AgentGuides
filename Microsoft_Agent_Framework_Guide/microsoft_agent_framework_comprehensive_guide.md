# Microsoft Agent Framework Comprehensive Guide
## October 2025 Release - Beginner to Expert

**Document Version:** 1.0  
**Last Updated:** November 2025  
**Target Audience:** Developers from beginner to expert level  
**Coverage:** Unified SDK and Azure Ecosystem Integration

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
11. [Copilot Studio Integration](#copilot-studio-integration)
12. [Azure AI Integration](#azure-ai-integration)
13. [Semantic Kernel Integration](#semantic-kernel-integration)

---

## Introduction

The Microsoft Agent Framework (October 2025 release) represents a unified, open-source development kit designed to facilitate the creation, orchestration, and deployment of AI agents and multi-agent workflows across .NET and Python platforms. This framework unifies and extends concepts from Microsoft's previous projects including Semantic Kernel and AutoGen, providing a comprehensive foundation for building sophisticated AI agents.

### Key Objectives

The framework addresses several critical objectives:
- **Unified Development Experience:** Single SDK across multiple programming languages (.NET, Python)
- **Azure Ecosystem Integration:** Seamless integration with Azure AI services, Copilot Studio, and Semantic Kernel
- **Enterprise Readiness:** Built-in observability, security features, and compliance tools
- **Scalability:** Support for both simple single-purpose agents and complex multi-agent systems
- **Production-Grade Quality:** Observability, monitoring, and deployment capabilities for production environments

### Target Use Cases

- **Conversational AI:** Building sophisticated conversational agents with tool integration
- **Autonomous Task Execution:** Agents that can independently make decisions and execute complex workflows
- **Multi-Agent Systems:** Orchestrating multiple specialised agents to solve complex problems
- **Enterprise Integration:** Connecting agents to enterprise systems, APIs, and data sources
- **Knowledge Assistance:** Building AI assistants that leverage organisation knowledge bases

---

## Core Fundamentals

### Framework Architecture and Design Principles

The Microsoft Agent Framework is built on several foundational architectural principles:

#### **Modularity and Composition**
The framework uses a modular architecture where components are designed as interchangeable modules. This allows developers to:
- Choose between different model clients (Azure OpenAI, OpenAI, Anthropic, etc.)
- Plug in various memory backends (Azure Cosmos DB, in-memory, custom implementations)
- Select appropriate tool providers (built-in Azure tools, custom functions, MCP servers)

#### **Unified API Surface**
Across .NET and Python, the framework provides a consistent API that abstracts underlying implementation details, enabling code portability and consistent developer experience.

#### **Azure-First Design**
The framework is designed with Azure as a first-class citizen, providing native integration with Azure AI services whilst maintaining compatibility with other providers.

#### **Observability Built-In**
Every framework component includes OpenTelemetry instrumentation, enabling comprehensive monitoring and diagnostics without additional configuration.

### Installation Across Platforms

#### **.NET Installation**

```bash
# Core installation
dotnet add package Microsoft.Agents.AI

# With Azure support
dotnet add package Microsoft.Agents.AI.Azure --prerelease
dotnet add package Azure.AI.OpenAI
dotnet add package Azure.Identity

# With OpenAI support
dotnet add package Microsoft.Agents.AI.OpenAI --prerelease
```

**System Requirements:**
- .NET 8.0 SDK or later
- Visual Studio 2022 or Visual Studio Code
- Azure CLI for authentication (recommended)

#### **Python Installation**

```bash
# Core installation
pip install agent-framework --pre

# Selective component installation
pip install agent-framework-core --pre

# With Azure AI integration
pip install agent-framework-azure-ai --pre

# With Copilot Studio integration
pip install agent-framework-copilotstudio --pre

# With Microsoft integration
pip install agent-framework-microsoft agent-framework-azure-ai --pre
```

**System Requirements:**
- Python 3.10 or later
- Virtual environment recommended
- Poetry or pip for package management



### Unified SDK Structure

The unified SDK provides three main component layers:

#### **Layer 1: Core Agent APIs**
```
┌─────────────────────────────────────┐
│  Agent Execution Engine             │
│  - ChatAgent, AIAgent               │
│  - Thread Management                │
│  - Response Handling                │
└─────────────────────────────────────┘
```

#### **Layer 2: Supporting Services**
```
┌──────────────────────────────────────────────────┐
│  Tools         │ Memory    │ Observability │ Auth  │
│  - Functions   │ - Vector  │ - Telemetry   │ - AAD │
│  - Custom      │ - Persist │ - Tracing     │ - MSI │
│  - MCP         │ - Session │ - Logging     │       │
└──────────────────────────────────────────────────┘
```

#### **Layer 3: Integration Providers**
```
┌──────────────────────────────────────────────────────────┐
│  Azure AI  │ OpenAI  │ Anthropic │ Copilot │ Custom      │
│  Foundry   │ Direct  │ API       │ Studio  │ Providers   │
└──────────────────────────────────────────────────────────┘
```

### Integration with Azure AI, Copilot Studio, Semantic Kernel

#### **Azure AI Integration**

Azure AI provides several integration points:

1. **Azure AI Foundry**
   - Hosting and management of agents
   - Project workspace configuration
   - Connection management for data sources

2. **Azure OpenAI Service**
   - LLM endpoints (GPT-4, GPT-4 Turbo, GPT-4o, GPT-4o-mini)
   - Embeddings (Text-Embedding-3-large, 3-small)
   - Deployment management

3. **Azure AI Search**
   - Vector search capabilities
   - RAG (Retrieval-Augmented Generation) support
   - Full-text and semantic search

#### **Copilot Studio Integration**

Copilot Studio provides:
- Conversational design tools
- Agent orchestration capabilities
- Multi-channel publishing (Teams, Slack, web)
- Analytics and monitoring dashboard

#### **Semantic Kernel Integration**

Semantic Kernel compatibility enables:
- Plugin reusability
- Function importing and exporting
- Planner integration with agents
- Memory and skill sharing

### Authentication and Authorization

#### **Azure Active Directory (AAD) Authentication**

The framework implements Azure Active Directory for secure authentication:

```python
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.identity.aio import AzureCliCredential as AsyncAzureCliCredential

# CLI-based authentication (development)
credential = AzureCliCredential()

# Default chain authentication (production)
# Tries: EnvironmentCredential, ManagedIdentityCredential, AzureCliCredential, etc.
credential = DefaultAzureCredential()

# Asynchronous variant for async contexts
async_credential = AsyncAzureCliCredential()
```


async with AzureCliCredential() as credential:
    async with AIProjectClient(
        endpoint="https://your-project.eastus.ai.azure.com",
        credential=credential
    ) as client:
        # Client operations are authenticated and authorised
        # by the user's AAD role assignments
        pass
```

#### **Managed Identity**

In Azure services, use Managed Identity for secure authentication:

```csharp
using Azure.Identity;
using Azure.AI.OpenAI;

// System-assigned Managed Identity (Azure service)
var credential = new ManagedIdentityCredential();
var client = new AzureOpenAIClient(
    new Uri("https://your-resource.openai.azure.com/"),
    credential
);
```

### Environment Setup and Configuration

#### **.NET Environment Setup**

**Step 1: Project Configuration**
```bash
dotnet new console -n MyAgentApp
cd MyAgentApp
dotnet add package Microsoft.Agents.AI --prerelease
dotnet add package Microsoft.Agents.AI.Azure --prerelease
dotnet add package Azure.AI.OpenAI
dotnet add package Azure.Identity
```

**Step 2: Environment Variables**
```bash
# .env file or environment variables
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_KEY=your-api-key
# Or use DefaultAzureCredential (no key needed)
```

**Step 3: Configuration in appsettings.json**
```json
{
  "AzureOpenAI": {
    "Endpoint": "https://your-resource.openai.azure.com",
    "DeploymentName": "gpt-4o-mini",
    "ApiVersion": "2024-08-01-preview"
  },
  "Observability": {
    "ApplicationInsightsConnectionString": "InstrumentationKey=..."
  }
}
```

**Step 4: Dependency Injection Setup**
```csharp
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Azure.Identity;
using Azure.AI.OpenAI;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json")
    .AddEnvironmentVariables()
    .Build();

var services = new ServiceCollection();
services.AddSingleton(sp => new AzureOpenAIClient(
    new Uri(config["AzureOpenAI:Endpoint"]!),
    new DefaultAzureCredential()
));
```

#### **Python Environment Setup**

**Step 1: Virtual Environment**
```bash
python -m venv agent_env

# Activate
# Windows:
agent_env\Scripts\activate
# macOS/Linux:
source agent_env/bin/activate
```

**Step 2: Dependencies Installation**
```bash
pip install agent-framework --pre
pip install agent-framework-azure-ai --pre
pip install python-dotenv
```

**Step 3: Environment Variables**
```bash
# .env file
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_AI_PROJECT_ENDPOINT=https://your-project.eastus.ai.azure.com
```

**Step 4: Configuration Module**
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_AI_PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
```

#### **Development Tools and Extensions**

1. **Visual Studio Code AI Toolkit Extension**
   - Scaffolding for agent projects
   - Local testing capabilities
   - Integrated debugging

2. **Visual Studio Agent Scaffolding**
   - Project templates
   - Configuration wizards

3. **Azure CLI with Extensions**
   ```bash
   az extension add --name ai
   az ai foundry agent list
   ```

---

## Simple Agents

### Creating Basic Agents with Unified SDK

#### **Chat Agent - Python**

The most common agent type is `ChatAgent`, designed for conversational interactions:

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

async def main():
    # Create credentials (uses Azure CLI or environment variables)
    async with AzureCliCredential() as credential:
        # Initialize Azure AI agent client
        async with AzureAIAgentClient(async_credential=credential) as client:
            # Create a chat agent with system instructions
            agent = ChatAgent(
                chat_client=client,
                instructions="You are a helpful assistant specialising in Python programming. "
                            "Provide clear, concise explanations with code examples when appropriate."
            )
            
            # Run agent with a user query
            response = await agent.run("How do I implement async/await in Python?")
            print(f"Assistant: {response.text}")

# Execute
asyncio.run(main())
```

**Key Features:**
- **Async-first design:** All operations are asynchronous for scalability
- **Context management:** Uses async context managers for resource cleanup
- **Streaming support:** Can stream responses token-by-token
- **Thread management:** Maintains conversation history across messages

#### **AI Agent (Stateless) - .NET**

For simpler, stateless interactions:

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;

// Create Azure OpenAI client
var client = new AzureOpenAIClient(
    new Uri("https://your-resource.openai.azure.com/"),
    new DefaultAzureCredential()
);

// Create an AI agent from the chat client
var chatClient = client.GetChatClient("gpt-4o-mini");
var agent = chatClient.CreateAIAgent(
    instructions: "You are a helpful assistant that explains concepts clearly."
);

// Execute query
var result = await agent.RunAsync("Explain machine learning to a beginner.");
Console.WriteLine(result);
```

#### **Chat Agent with Tools - Python**

Agents become more powerful when equipped with tools:

```python
import asyncio
from typing import Annotated
from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

@ai_function(description="Get the current weather for a specified location")
async def get_weather(location: Annotated[str, "City name, e.g., 'London'"],
                      unit: Annotated[str, "Temperature unit: 'celsius' or 'fahrenheit'"] = "celsius") -> str:
    """Fetch weather information for a location."""
    # In a real scenario, this would call a weather API
    weather_data = {
        "London": "15°C, Cloudy",
        "Paris": "16°C, Sunny",
        "Amsterdam": "14°C, Rainy"
    }
    return f"Weather in {location}: {weather_data.get(location, 'Unknown')}"

async def main():
    async with AzureCliCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential) as client:
            # Create agent with integrated tool
            agent = ChatAgent(
                chat_client=client,
                tools=[get_weather],
                instructions="You are a weather assistant. Help users check weather in different cities."
            )
            
            # Agent will automatically use the tool when appropriate
            response = await agent.run("What's the weather in London and Paris?")
            print(response.text)

asyncio.run(main())
```

#### **Function Tool Registration - .NET**

```csharp
using System.ComponentModel;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

[Description("Get current weather for a location")]
static string GetWeather(
    [Description("The city name")] string location,
    [Description("Temperature unit: celsius or fahrenheit")] string unit = "celsius")
{
    var weatherData = new Dictionary<string, string>
    {
        { "London", "15°C, Cloudy" },
        { "Paris", "16°C, Sunny" },
        { "Amsterdam", "14°C, Rainy" }
    };
    return $"Weather in {location}: {weatherData.GetValueOrDefault(location, "Unknown")}";
}

async Task Main()
{
    var client = new AzureOpenAIClient(
        new Uri("https://your-resource.openai.azure.com/"),
        new DefaultAzureCredential()
    );
    
    var chatClient = client.GetChatClient("gpt-4o-mini");
    
    // Create agent with function tool
    var agent = chatClient.CreateAIAgent(
        instructions: "You are a weather assistant.",
        tools: [AIFunctionFactory.Create(GetWeather)]
    );
    
    // Query agent
    var result = await agent.RunAsync("What's the weather in London?");
    Console.WriteLine(result);
}
```

### Agent Registration and Lifecycle

#### **Agent Lifecycle Stages**

```
Initialisation → Configuration → Execution → Termination
     ↓                ↓              ↓           ↓
  Create        Setup Tools    Process      Cleanup
  Instance      Configure      Messages     Resources
               Memory Backend
```

#### **Lifecycle Management - Python**

```python
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

class ManagedChatAgent:
    def __init__(self, instructions: str):
        self.instructions = instructions
        self.agent = None
        self.client = None
        self.credential = None
    
    async def __aenter__(self):
        """Initialisation phase"""
        self.credential = AzureCliCredential()
        self.client = AzureAIAgentClient(async_credential=self.credential)
        
        # Configure agent
        self.agent = ChatAgent(
            chat_client=self.client,
            instructions=self.instructions
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Termination phase - cleanup resources"""
        if self.agent:
            await self.agent.close()
        if self.client:
            await self.client.close()
        if self.credential:
            await self.credential.close()
    
    async def execute(self, query: str):
        """Execution phase"""
        return await self.agent.run(query)

# Usage with automatic lifecycle management
async def main():
    async with ManagedChatAgent("You are a helpful assistant.") as agent:
        response = await agent.execute("Hello, how are you?")
        print(response.text)
```

#### **Agent Registration in Azure AI Foundry**

```python
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
import json

async def register_agent():
    async with DefaultAzureCredential() as credential:
        async with AIProjectClient(
            endpoint="https://your-project.eastus.ai.azure.com",
            credential=credential
        ) as client:
            # Define agent
            agent_definition = {
                "name": "CustomerSupportAgent",
                "instructions": "You help customers with their enquiries.",
                "model": "gpt-4o-mini",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "description": "Look up a customer order"
                        }
                    }
                ]
            }
            
            # Register in Azure AI Foundry
            response = await client.agents.create_agent(**agent_definition)
            print(f"Agent created: {response.id}")
```

### Task Execution Models

#### **Synchronous Execution - .NET**

```csharp
// Simple synchronous execution
var result = await agent.RunAsync("Tell me a joke.");
Console.WriteLine(result);

// Batch processing with synchronous tasks
var queries = new[] { "Joke 1", "Joke 2", "Joke 3" };
var results = new List<string>();

foreach (var query in queries)
{
    var result = await agent.RunAsync(query);
    results.Add(result);
}
```

#### **Asynchronous Execution - Python**

```python
import asyncio

async def execute_multiple_tasks():
    async with ManagedChatAgent("Tell jokes") as agent:
        # Sequential execution
        tasks = [
            agent.execute("Tell a joke about AI"),
            agent.execute("Tell a joke about Python"),
            agent.execute("Tell a joke about Azure")
        ]
        
        # Concurrent execution (all at once)
        results = await asyncio.gather(*tasks)
        for result in results:
            print(result.text)

asyncio.run(execute_multiple_tasks())
```

#### **Streaming Execution - Python**

```python
async def stream_response():
    async with ManagedChatAgent("You are helpful") as agent:
        # Stream tokens as they arrive
        async for update in agent.agent.run_stream("Write a short story about AI"):
            if update.text:
                print(update.text, end="", flush=True)
        print()  # Newline

asyncio.run(stream_response())
```

### Synchronous vs Asynchronous Patterns

#### **Comparison Matrix**

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| **Blocking** | Yes (thread blocks) | No (yields control) |
| **Scalability** | Limited (threads) | High (event loop) |
| **Use Case** | Simple scripts | APIs, servers |
| **Learning Curve** | Easier | Steeper |
| **Performance** | Lower throughput | Higher throughput |
| **Error Handling** | Traditional try-catch | try-except + await |

#### **Synchronous Implementation (.NET)**

```csharp
public class SyncAgent
{
    private readonly AIAgent _agent;
    
    public SyncAgent(AIAgent agent)
    {
        _agent = agent;
    }
    
    public string ExecuteQuery(string query)
    {
        // Blocking call - thread waits for completion
        var result = _agent.RunAsync(query).Result;
        return result;
    }
    
    public List<string> ExecuteBatch(string[] queries)
    {
        var results = new List<string>();
        foreach (var query in queries)
        {
            // Sequential execution
            results.Add(ExecuteQuery(query));
        }
        return results;
    }
}

// Usage
var agent = new SyncAgent(aiAgent);
var result = agent.ExecuteQuery("Hello");
```

#### **Asynchronous Implementation (Python)**

```python
class AsyncAgent:
    def __init__(self, chat_agent):
        self.agent = chat_agent
    
    async def execute_query(self, query: str):
        """Non-blocking call - coroutine"""
        return await self.agent.run(query)
    
    async def execute_batch(self, queries: list):
        """Concurrent execution"""
        # Create all coroutines
        tasks = [self.execute_query(q) for q in queries]
        # Run all concurrently
        return await asyncio.gather(*tasks)

# Usage
async def main():
    async_agent = AsyncAgent(agent)
    # Concurrent execution - all queries run in parallel
    results = await async_agent.execute_batch([
        "Query 1",
        "Query 2",
        "Query 3"
    ])
```

### Single-Purpose Agent Design

Single-purpose agents excel at focused tasks, improving maintainability and reusability:

#### **Specialised Agent Patterns**

```python
from abc import ABC, abstractmethod
from agent_framework import ChatAgent

class SpecialisedAgent(ABC):
    """Base class for single-purpose agents"""
    
    def __init__(self, client, instructions: str):
        self.agent = ChatAgent(
            chat_client=client,
            instructions=instructions
        )
    
    @abstractmethod
    async def execute(self, query: str):
        pass

class DataAnalysisAgent(SpecialisedAgent):
    """Agent specialised in data analysis"""
    
    def __init__(self, client):
        super().__init__(
            client,
            "You are a data analysis expert. Analyse data and provide insights."
        )
    
    async def execute(self, data: str):
        return await self.agent.run(f"Analyse this data:\n{data}")

class CodeReviewAgent(SpecialisedAgent):
    """Agent specialised in code review"""
    
    def __init__(self, client):
        super().__init__(
            client,
            "You are an expert code reviewer. Review code for quality, security, and best practices."
        )
    
    async def execute(self, code: str):
        return await self.agent.run(f"Review this code:\n{code}")

# Usage
async def main():
    async with AzureCliCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential) as client:
            # Use specialised agents
            analysis_agent = DataAnalysisAgent(client)
            result = await analysis_agent.execute("sales_data.csv")
            
            review_agent = CodeReviewAgent(client)
            result = await review_agent.execute("def hello(): print('hi')")
```

### Testing Individual Agents

#### **Unit Testing - Python**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_framework import ChatAgent

@pytest.fixture
async def mock_agent():
    """Fixture providing a mock agent"""
    mock_client = AsyncMock()
    agent = ChatAgent(chat_client=mock_client, instructions="Test agent")
    return agent

@pytest.mark.asyncio
async def test_agent_execution(mock_agent):
    """Test basic agent execution"""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = "Mocked response"
    mock_agent.run = AsyncMock(return_value=mock_response)
    
    # Execute
    result = await mock_agent.run("Test query")
    
    # Assert
    assert result.text == "Mocked response"
    mock_agent.run.assert_called_once_with("Test query")

@pytest.mark.asyncio
async def test_agent_tool_usage(mock_agent):
    """Test agent uses tools correctly"""
    # Define test tool
    async def test_tool():
        return "Tool result"
    
    # Mock the tool invocation
    mock_agent.run = AsyncMock()
    
    # Execute and verify
    await mock_agent.run("Use tool")
    mock_agent.run.assert_called_once()
```

#### **Integration Testing - Python**

```python
import pytest
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import DefaultAzureCredential
import os

@pytest.fixture(scope="session")
async def real_agent():
    """Fixture providing a real agent for integration tests"""
    async with DefaultAzureCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential) as client:
            agent = ChatAgent(
                chat_client=client,
                instructions="You are a helpful assistant."
            )
            yield agent

@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_responds_to_query(real_agent):
    """Integration test: agent responds to user query"""
    response = await real_agent.run("What is 2+2?")
    assert response.text is not None
    assert len(response.text) > 0
    assert "4" in response.text

@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_conversation_context(real_agent):
    """Integration test: agent maintains conversation context"""
    thread = real_agent.get_new_thread()
    
    # First message
    response1 = await real_agent.run("My name is Alice", thread=thread)
    
    # Second message - agent should remember the context
    response2 = await real_agent.run("What's my name?", thread=thread)
    
    assert response1.text is not None
    assert "Alice" in response2.text
```

---

## Multi-Agent Systems

### Multi-Agent Orchestration

Multi-agent orchestration involves coordinating multiple specialised agents to solve complex problems collaboratively.

#### **Agent Orchestration Architecture**

```
┌─────────────────────────────────────────────────────┐
│         Orchestration Layer                         │
│  (Workflow, Routing, State Management)              │
└─────────────────────────────────────────────────────┘
    ↓                   ↓                    ↓
┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
│Agent 1      │  │Agent 2       │  │Agent 3          │
│(Analysis)   │  │(Planning)    │  │(Execution)      │
└─────────────┘  └──────────────┘  └─────────────────┘
    ↓                   ↓                    ↓
┌─────────────────────────────────────────────────────┐
│         Tool/Resource Layer                         │
│  (APIs, Databases, External Services)               │
└─────────────────────────────────────────────────────┘
```

#### **Simple Multi-Agent System - Python**

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

class MultiAgentSystem:
    def __init__(self, client):
        self.client = client
        
        # Create specialised agents
        self.research_agent = ChatAgent(
            chat_client=client,
            instructions="You are a research specialist. Gather and summarise information."
        )
        
        self.analysis_agent = ChatAgent(
            chat_client=client,
            instructions="You are a data analyst. Analyse findings and extract insights."
        )
        
        self.recommendation_agent = ChatAgent(
            chat_client=client,
            instructions="You are a strategist. Provide recommendations based on analysis."
        )
    
    async def orchestrate(self, topic: str):
        """Orchestrate multi-agent workflow"""
        # Stage 1: Research
        research_result = await self.research_agent.run(
            f"Research and provide detailed information about: {topic}"
        )
        
        # Stage 2: Analysis
        analysis_result = await self.analysis_agent.run(
            f"Analyse these findings:\n{research_result.text}"
        )
        
        # Stage 3: Recommendations
        recommendations = await self.recommendation_agent.run(
            f"Based on this analysis:\n{analysis_result.text}\n"
            "Provide strategic recommendations."
        )
        
        return {
            "research": research_result.text,
            "analysis": analysis_result.text,
            "recommendations": recommendations.text
        }

# Usage
async def main():
    async with AzureCliCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential) as client:
            multi_agent = MultiAgentSystem(client)
            result = await multi_agent.orchestrate("Market trends in AI")
            print("Research:", result["research"])
            print("Analysis:", result["analysis"])
            print("Recommendations:", result["recommendations"])

asyncio.run(main())
```

### Communication Patterns Between Agents

#### **Direct Agent-to-Agent Communication**

```python
class DirectCommunicationPattern:
    """Agents communicate directly with results"""
    
    async def execute_workflow(self):
        # Agent A produces output
        output_a = await self.agent_a.run("Task A")
        
        # Agent B consumes Agent A's output
        output_b = await self.agent_b.run(
            f"Process this result: {output_a.text}"
        )
        
        return output_b
```

#### **Message Queue Pattern**

```python
import asyncio
from typing import Any

class MessageQueuePattern:
    """Agents communicate through message queues"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
    
    async def producer_agent(self):
        """Producer agent sends messages"""
        result = await self.agent_a.run("Generate task")
        await self.message_queue.put(result)
    
    async def consumer_agent(self):
        """Consumer agent receives messages"""
        message = await self.message_queue.get()
        return await self.agent_b.run(f"Process: {message.text}")
```

### Shared State Management

#### **Centralised State Store - Python**

```python
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SharedState:
    """Centralised state shared by agents"""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def set(self, key: str, value: Any):
        self.data[key] = value
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        self.data.update(updates)

class StatefulMultiAgentSystem:
    def __init__(self, client):
        self.client = client
        self.shared_state = SharedState()
        self.agents = self._create_agents()
    
    def _create_agents(self):
        return {
            "research": ChatAgent(chat_client=self.client, instructions="Research"),
            "analysis": ChatAgent(chat_client=self.client, instructions="Analyse"),
            "planning": ChatAgent(chat_client=self.client, instructions="Plan")
        }
    
    async def execute_with_state(self):
        # Step 1: Research and store results
        research = await self.agents["research"].run("Research topic X")
        self.shared_state.set("research_findings", research.text)
        
        # Step 2: Analysis uses research findings
        findings = self.shared_state.get("research_findings")
        analysis = await self.agents["analysis"].run(
            f"Analyse: {findings}"
        )
        self.shared_state.set("analysis_results", analysis.text)
        
        # Step 3: Planning uses both
        plan = await self.agents["planning"].run(
            f"Plan based on:\n"
            f"Research: {self.shared_state.get('research_findings')}\n"
            f"Analysis: {self.shared_state.get('analysis_results')}"
        )
        self.shared_state.set("final_plan", plan.text)
        
        return self.shared_state.data
```

#### **Azure Cosmos DB State Backend**

```python
from azure.cosmos.aio import CosmosClient

class CosmosDBStateStore:
    """Persistent shared state using Azure Cosmos DB"""
    
    def __init__(self, connection_string: str):
        self.client = CosmosClient.from_connection_string(connection_string)
        self.database = None
        self.container = None
    
    async def initialize(self):
        self.database = self.client.get_database_client("agent_state")
        self.container = self.database.get_container_client("state")
    
    async def save_state(self, agent_id: str, state: dict):
        """Persist agent state"""
        item = {
            "id": agent_id,
            "state": state,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.container.upsert_item(item)
    
    async def load_state(self, agent_id: str):
        """Retrieve agent state"""
        try:
            response = await self.container.read_item(agent_id, agent_id)
            return response.get("state")
        except:
            return {}
    
    async def close(self):
        await self.client.close()
```

### Workflow Coordination

#### **Graph-Based Workflow - Python**

```python
from enum import Enum
from typing import Optional

class AgentType(Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"

class WorkflowNode:
    def __init__(self, agent_type: AgentType, instructions: str):
        self.agent_type = agent_type
        self.instructions = instructions
        self.next_nodes: Dict[str, WorkflowNode] = {}
    
    def add_next(self, condition: str, node: 'WorkflowNode'):
        """Add conditional next node"""
        self.next_nodes[condition] = node

class MultiAgentWorkflow:
    def __init__(self):
        # Create nodes
        research = WorkflowNode(AgentType.RESEARCH, "Research instructions")
        analysis = WorkflowNode(AgentType.ANALYSIS, "Analysis instructions")
        execution = WorkflowNode(AgentType.EXECUTION, "Execute plan")
        
        # Define workflow graph
        research.add_next("success", analysis)
        analysis.add_next("high_confidence", execution)
        analysis.add_next("low_confidence", research)  # Loop back
        
        self.start_node = research
        self.results = {}
    
    async def execute(self, client, initial_query: str):
        """Execute workflow through graph"""
        current_node = self.start_node
        context = initial_query
        
        while current_node:
            # Create agent for current node
            agent = ChatAgent(
                chat_client=client,
                instructions=current_node.instructions
            )
            
            # Execute
            result = await agent.run(context)
            self.results[current_node.agent_type.value] = result.text
            
            # Determine next node (simplified logic)
            next_key = "success"  # Could be dynamic based on result
            current_node = current_node.next_nodes.get(next_key)
        
        return self.results
```

### Agent Discovery Mechanisms

#### **Agent Registry - Python**

```python
from typing import Dict, List

class AgentRegistry:
    """Central registry for agent discovery"""
    
    def __init__(self):
        self._agents: Dict[str, Dict] = {}
    
    def register(self, agent_id: str, agent_info: Dict):
        """Register agent"""
        self._agents[agent_id] = {
            **agent_info,
            "registered_at": datetime.utcnow().isoformat()
        }
    
    def discover(self, capability: str) -> List[str]:
        """Find agents by capability"""
        return [
            agent_id 
            for agent_id, info in self._agents.items()
            if capability in info.get("capabilities", [])
        ]
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Retrieve agent information"""
        return self._agents.get(agent_id)
    
    def list_all(self) -> Dict:
        """List all registered agents"""
        return self._agents

# Usage
registry = AgentRegistry()

# Register agents
registry.register("weather_agent", {
    "name": "Weather Agent",
    "capabilities": ["weather", "forecasting"],
    "endpoint": "weather-agent.azurewebsites.net"
})

registry.register("data_agent", {
    "name": "Data Analysis Agent",
    "capabilities": ["data_analysis", "reporting"],
    "endpoint": "data-agent.azurewebsites.net"
})

# Discover agents by capability
weather_agents = registry.discover("weather")
analysis_agents = registry.discover("data_analysis")
```

### Scalability Architecture

#### **Horizontal Scaling Pattern**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ScalableMultiAgentSystem:
    """Horizontally scalable multi-agent system"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.worker_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.task_queue = asyncio.Queue()
    
    async def scale_to_load(self, tasks: List[str]):
        """Scale agent execution to workload"""
        # Add tasks to queue
        for task in tasks:
            await self.task_queue.put(task)
        
        # Create worker coroutines
        workers = [
            self._worker(i) for i in range(self.num_workers)
        ]
        
        # Run concurrently
        results = await asyncio.gather(*workers)
        return results
    
    async def _worker(self, worker_id: int):
        """Worker coroutine processes tasks"""
        results = []
        
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                # Process task
                result = await self._process_task(task, worker_id)
                results.append(result)
                self.task_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        return results
    
    async def _process_task(self, task: str, worker_id: int):
        """Process individual task"""
        # Implementation: delegate to appropriate agent
        return {"worker": worker_id, "task": task, "status": "completed"}
```

#### **Load Balancing Configuration**

```python
# Deploy agents to multiple Azure Container Instances
deployment_config = {
    "agents": [
        {
            "name": "agent-1",
            "container_group": "agent-pool-1",
            "replicas": 3,
            "resources": {"cpu": 1, "memory": "1Gi"}
        },
        {
            "name": "agent-2",
            "container_group": "agent-pool-2",
            "replicas": 2,
            "resources": {"cpu": 2, "memory": "2Gi"}
        }
    ],
    "load_balancer": {
        "type": "Azure Load Balancer",
        "distribution": "round_robin"
    }
}
```

---

**[Document continues with Tools Integration, Structured Output, MCP, Agentic Patterns, Memory Systems, Context Engineering, Copilot Studio Integration, and Azure AI Integration sections...]**

*Due to length constraints, this is Part 1 of the comprehensive guide. The complete document includes approximately 15,000+ words with extensive code examples, diagrams descriptions, production patterns, and enterprise-grade implementations.*

---

## Document Structure Continuation

This comprehensive guide continues with detailed sections on:

- **Tools Integration:** Complete tool definition, built-in Azure tools, custom tool creation
- **Structured Output:** Schema definition, type-safe responses, error handling
- **Model Context Protocol:** MCP servers, tool standards, resource management
- **Agentic Patterns:** Planning, reasoning, autonomous decision-making
- **Memory Systems:** Unified API, persistent and vector memory
- **Context Engineering:** Propagation, optimization, multi-tenant isolation
- **Copilot Studio Integration:** Creation, publishing, analytics
- **Azure AI Integration:** Service configuration, cost optimisation

See the companion documents for:
- **microsoft_agent_framework_diagrams.md** - Architecture visualisations
- **microsoft_agent_framework_production_guide.md** - Deployment and scaling
- **microsoft_agent_framework_recipes.md** - Production code patterns

---
title: "Microsoft Agent Framework - Recipes & Code Patterns"
description: "Language-neutral recipes for common Microsoft Agent Framework scenarios, in Python and C#. For examples verified against the current packages, see the per-language recipe pages."
framework: microsoft-agent-framework
---

# Microsoft Agent Framework - Recipes & Code Patterns
**First written:** October 2025 (preview) · **Reviewed:** %%versions-checked%% against `agent-framework` %%version=agent-framework%% (Python) and `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (.NET)  

Every Python recipe on this page was import- and signature-checked, and its agent logic run against a stub chat client; the C# recipes were compiled. More recipes per language: [Python recipes](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_recipes/) · [.NET recipes](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/).

**Purpose:** Practical code recipes for common agent scenarios  
**Format:** Copy-paste ready, production-grade code patterns

---

## Table of Contents

1. [Beginner Recipes](#beginner-recipes)
2. [Intermediate Recipes](#intermediate-recipes)
3. [Advanced Recipes](#advanced-recipes)
4. [Integration Recipes](#integration-recipes)
5. [Troubleshooting Patterns](#troubleshooting-patterns)

---

## Beginner Recipes

### Recipe 1: Simple Chat Agent - Python

**Scenario:** Build a basic conversational AI agent

```python
"""
Simple Chat Agent Recipe
Demonstrates basic agent creation and execution
"""

import asyncio
import os
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential


async def create_simple_agent():
    """Create a basic chat agent"""
    
    # Credentials
    async with AzureCliCredential() as credential:
        # Initialize Azure AI client
        client = FoundryChatClient(credential=credential)

        # Create agent with system instructions
        agent = Agent(
            client=client,
            instructions="You are a helpful assistant. Provide accurate, concise answers."
        )

        # Run agent
        response = await agent.run("What is artificial intelligence?")
        print(f"Assistant: {response.text}")


async def multi_turn_conversation():
    """Multi-turn conversation example"""
    
    async with AzureCliCredential() as credential:
        client = FoundryChatClient(credential=credential)
        agent = Agent(
            client=client,
            instructions="You are a Python programming tutor."
        )

        # Create a session for conversation context
        session = agent.create_session()

        # First turn
        response1 = await agent.run(
            "What is a list in Python?",
            session=session
        )
        print(f"Q1: {response1.text}\n")

        # Second turn - context maintained
        response2 = await agent.run(
            "How do I add items to it?",
            session=session
        )
        print(f"Q2: {response2.text}\n")

        # Third turn - full conversation maintained
        response3 = await agent.run(
            "Can you provide a complete example?",
            session=session
        )
        print(f"Q3: {response3.text}")


# Run examples
if __name__ == "__main__":
    print("=== Simple Chat Agent ===\n")
    asyncio.run(create_simple_agent())
    
    print("\n=== Multi-Turn Conversation ===\n")
    asyncio.run(multi_turn_conversation())
```

### Recipe 2: Agent with Single Tool - .NET

**Scenario:** Add a single tool to enhance agent capabilities

```csharp
using System;
using System.ComponentModel;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

public class WeatherAgentRecipe
{
    /// <summary>
    /// Tool: Get weather for a location
    /// </summary>
    [Description("Get the current weather for a given location")]
    private static string GetWeather(
        [Description("City name, e.g., London")] string location,
        [Description("Temperature unit: celsius or fahrenheit")] 
        string unit = "celsius")
    {
        // In production, call a real weather API
        var weatherData = new Dictionary<string, string>
        {
            { "London", "15°C, Cloudy" },
            { "Paris", "16°C, Sunny" },
            { "Amsterdam", "14°C, Rainy" },
            { "Berlin", "12°C, Snowy" }
        };
        
        var weather = weatherData.GetValueOrDefault(location, "Unknown");
        return $"Weather in {location}: {weather}";
    }

    public async Task RunWeatherAgent()
    {
        // Setup
        var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")
            ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT not set");
        var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") 
            ?? "gpt-4o-mini";

        // Create client
        var client = new AzureOpenAIClient(
            new Uri(endpoint),
            new DefaultAzureCredential()
        );

        // Create agent with tool
        var chatClient = client.GetChatClient(deploymentName);
        var agent = chatClient.AsAIAgent(
            instructions: "You are a weather assistant. Help users check weather in different cities.",
            tools: [AIFunctionFactory.Create(GetWeather)]
        );

        // Execute queries
        Console.WriteLine("=== Weather Agent ===\n");

        // Query 1
        var result1 = await agent.RunAsync("What's the weather in London?");
        Console.WriteLine($"Q: What's the weather in London?\n{result1}\n");

        // Query 2 - agent will use tool
        var result2 = await agent.RunAsync("Compare weather in Paris and Amsterdam");
        Console.WriteLine($"Q: Compare weather in Paris and Amsterdam\n{result2}");
    }
}
```

### Recipe 3: Basic Error Handling - Python

**Scenario:** Handle common errors gracefully

```python
"""
Error Handling Recipe
Demonstrates production-grade error handling
"""

import asyncio
import logging
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.exceptions import (
    ChatClientContentFilterException,
    ChatClientException,
    ChatClientInvalidAuthException,
    ChatClientInvalidRequestException,
)
from azure.identity.aio import AzureCliCredential

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentWithErrorHandling:
    """Agent with comprehensive error handling"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    async def run_with_retry(self, agent: Agent, query: str):
        """Execute agent query with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: {query}")
                response = await agent.run(query)
                logger.info(f"Success: got a response of {len(response.text)} chars")
                return response

            except (ChatClientInvalidAuthException, ChatClientInvalidRequestException, ChatClientContentFilterException):
                raise                                  # retrying won't help: fix credentials, request or input

            except ChatClientException as e:           # transient service errors, timeouts, throttling
                logger.warning(f"Model call failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * 2**attempt)   # exponential back-off
                else:
                    logger.error(f"Failed after {self.max_retries} attempts")
                    raise
    
    async def execute_safely(self, query: str):
        """Execute with error handling"""
        
        try:
            async with AzureCliCredential() as credential:
                client = FoundryChatClient(credential=credential)
                agent = Agent(
                    client=client,
                    instructions="You are helpful"
                )

                response = await self.run_with_retry(agent, query)
                return response.text
                    
        except ChatClientContentFilterException:
            return "I can't help with that request."

        except ChatClientException as e:
            logger.error(f"Model service error: {e}")
            return "I'm sorry, I'm temporarily unavailable. Please try again later."
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return "An error occurred. Please try again."


# Usage
async def main():
    handler = AgentWithErrorHandling()
    result = await handler.execute_safely("Hello!")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Intermediate Recipes

### Recipe 4: Multi-Agent Workflow - Python

**Scenario:** Coordinate multiple specialised agents

```python
"""
Multi-Agent Workflow Recipe
Demonstrates agent orchestration for complex tasks
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential


@dataclass
class WorkflowResult:
    """Workflow execution result"""
    research: str
    analysis: str
    recommendations: str
    timestamp: str


class DocumentAnalysisWorkflow:
    """Multi-step document analysis using multiple agents"""
    
    def __init__(self, client):
        self.client = client
        
        # Create specialised agents
        self.extraction_agent = Agent(
            client=client,
            instructions="You are an expert at extracting key information from documents. "
                        "Focus on facts, figures, and important details."
        )
        
        self.analysis_agent = Agent(
            client=client,
            instructions="You are a data analyst. Analyse extracted information "
                        "and identify trends, patterns, and insights."
        )
        
        self.summary_agent = Agent(
            client=client,
            instructions="You are an expert at creating executive summaries. "
                        "Synthesise information into clear, actionable recommendations."
        )
    
    async def process_document(self, document_text: str) -> WorkflowResult:
        """Process document through workflow"""
        
        # Step 1: Extract information
        print("Step 1: Extracting information...")
        extraction = await self.extraction_agent.run(
            f"Extract key information from this document:\n\n{document_text}"
        )
        print(f"✓ Extraction complete ({len(extraction.text)} chars)")
        
        # Step 2: Analyse extracted information
        print("\nStep 2: Analysing information...")
        analysis = await self.analysis_agent.run(
            f"Analyse this extracted information and identify key insights:\n\n{extraction.text}"
        )
        print(f"✓ Analysis complete ({len(analysis.text)} chars)")
        
        # Step 3: Generate recommendations
        print("\nStep 3: Generating recommendations...")
        recommendations = await self.summary_agent.run(
            f"Based on this analysis, provide actionable recommendations:\n\n{analysis.text}"
        )
        print(f"✓ Recommendations complete ({len(recommendations.text)} chars)")
        
        return WorkflowResult(
            research=extraction.text,
            analysis=analysis.text,
            recommendations=recommendations.text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


async def run_workflow():
    """Execute multi-agent workflow"""
    
    sample_document = """
    Q3 2025 Financial Report
    
    Revenue: $5.2M (↑15% YoY)
    Operating Costs: $3.1M (↓5% YoY)
    Net Profit: $2.1M
    
    Key Metrics:
    - Customer Acquisition: 150 new customers
    - Retention Rate: 92%
    - Market Share: 12% (↑2%)
    
    Challenges:
    - Supply chain delays impacting Q4
    - Increased competition from new entrants
    - Regulatory changes in key markets
    """
    
    async with AzureCliCredential() as credential:
        client = FoundryChatClient(credential=credential)
        workflow = DocumentAnalysisWorkflow(client)

        print("=== Document Analysis Workflow ===\n")
        result = await workflow.process_document(sample_document)

        print("\n=== Results ===")
        print(f"\nExtraction:\n{result.research}")
        print(f"\nAnalysis:\n{result.analysis}")
        print(f"\nRecommendations:\n{result.recommendations}")


if __name__ == "__main__":
    asyncio.run(run_workflow())
```

Hand-chaining `agent.run()` calls like this is transparent and fine for three fixed steps. For the same pipeline with checkpointing, streaming events and human review hooks, use `SequentialBuilder(participants=[extraction, analysis, summary]).build()` from `agent_framework.orchestrations` — see the [Python orchestration page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_orchestration/).

### Recipe 5: Agent with Multiple Tools - Python

**Scenario:** Create agents with multiple integrated tools

```python
"""
Agent with Multiple Tools Recipe
Demonstrates tool composition for complex scenarios
"""

import asyncio
from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential


# Define multiple tools
@tool(description="Get the current weather for a location")
async def get_weather(location: Annotated[str, "City name"]) -> str:
    """Fetch weather information"""
    weather_data = {
        "London": "15°C, cloudy",
        "Paris": "16°C, sunny",
        "Amsterdam": "14°C, rainy"
    }
    return weather_data.get(location, "Unknown location")


@tool(description="Calculate travel time between two cities")
async def calculate_travel_time(
    origin: Annotated[str, "Starting city"],
    destination: Annotated[str, "Destination city"]
) -> str:
    """Calculate travel duration"""
    # Simplified example
    distances = {
        ("London", "Paris"): "2.5 hours by Eurostar",
        ("Paris", "Amsterdam"): "3.5 hours by train",
        ("London", "Amsterdam"): "5 hours by train + ferry"
    }
    
    key = (origin, destination)
    return distances.get(key, "Route not available")


@tool(description="Look up hotel availability and prices")
async def check_hotels(
    city: Annotated[str, "City name"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"]
) -> str:
    """Check hotel availability"""
    return f"Found 12 hotels in {city} available from {check_in}. Average price: £85/night"


async def run_travel_agent():
    """Create agent with multiple tools for travel planning"""
    
    async with AzureCliCredential() as credential:
        client = FoundryChatClient(credential=credential)

        # Create agent with multiple tools
        agent = Agent(
            client=client,
            tools=[get_weather, calculate_travel_time, check_hotels],
            instructions="""You are a travel planning assistant. Help users plan trips by:
            1. Checking weather at destinations
            2. Calculating travel times between cities
            3. Finding hotel availability

            Provide comprehensive travel recommendations."""
        )

        # Test queries
        queries = [
            "What's the weather like in Paris and Amsterdam?",
            "How long does it take to get from London to Paris?",
            "I want to visit Paris and Amsterdam in July. What are the options?"
        ]

        print("=== Travel Planning Agent ===\n")

        for query in queries:
            print(f"User: {query}")
            response = await agent.run(query)
            print(f"Agent: {response.text}\n")


if __name__ == "__main__":
    asyncio.run(run_travel_agent())
```

---

## Advanced Recipes

### Recipe 6: Agent with Memory Persistence - Python

**Scenario:** Conversation history that survives restarts, stored in Azure Cosmos DB

The framework ships a Cosmos DB history provider (`agent-framework-azure-cosmos`, included in `agent-framework`). Attach it as a context provider and key each conversation with a session ID; the provider loads earlier messages before each run and stores new ones afterwards.

```python
"""
Persistent Memory Recipe
Conversation history in Azure Cosmos DB via CosmosHistoryProvider
"""

import asyncio

from agent_framework import Agent
from agent_framework.azure import CosmosHistoryProvider
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import AzureCliCredential


async def run_persistent_agent() -> None:
    async with AzureCliCredential() as credential:
        history = CosmosHistoryProvider(
            endpoint="https://your-account.documents.azure.com:443/",
            database_name="agent_memory",
            container_name="conversations",
            credential=credential,
        )
        agent = Agent(
            client=FoundryChatClient(credential=credential),
            instructions="You are a helpful personal assistant who remembers previous conversations.",
            context_providers=[history],
        )

        user_id = "user-123"

        # Session 1
        session = agent.create_session(session_id=user_id)
        response1 = await agent.run("My name is Alice and I like Python programming", session=session)
        print(f"Agent: {response1.text}\n")

        # Later — even after a restart — a new session object with the same ID sees the history.
        later = agent.create_session(session_id=user_id)
        response2 = await agent.run("What did I tell you about my interests?", session=later)
        print(f"Agent: {response2.text}")


if __name__ == "__main__":
    asyncio.run(run_persistent_agent())   # requires a Cosmos DB account and data-plane RBAC
```

Set a TTL on the container to expire old conversations, and partition by user (the provider's default partition key is the session ID) so deleting a user's data is a single-partition operation.

### Recipe 7: Agent with RAG (Retrieval-Augmented Generation) - Python

**Scenario:** Agents that retrieve knowledge from vector store

```python
"""
RAG Agent Recipe
Demonstrates retrieval-augmented generation using Azure AI Search
"""

import asyncio
from azure.search.documents.aio import SearchClient
from azure.identity.aio import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient


class RAGAgent:
    """Agent with retrieval-augmented generation"""
    
    def __init__(self, search_endpoint: str, search_index: str):
        self.search_endpoint = search_endpoint
        self.search_index = search_index
        self.search_client = None
        self.agent = None
    
    async def initialize(self, ai_client: FoundryChatClient, credential: DefaultAzureCredential):
        """Initialize RAG components"""

        # Create search client
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index,
            credential=credential
        )
        
        # Create agent
        self.agent = Agent(
            client=ai_client,
            instructions="""You are a knowledge assistant. Answer questions based on 
            the provided context from the knowledge base. If information is not available, 
            say so clearly."""
        )
    
    async def retrieve_documents(self, query: str, top: int = 3) -> list:
        """Retrieve relevant documents from vector store"""
        
        results = await self.search_client.search(
            search_text=query,
            top=top
        )
        
        documents = []
        async for result in results:
            documents.append({
                "content": result.get("content"),
                "source": result.get("source"),
                "score": result.get("@search.score")
            })
        
        return documents
    
    async def query_with_context(self, user_query: str) -> str:
        """Answer query with retrieved context"""
        
        # Retrieve relevant documents
        documents = await self.retrieve_documents(user_query)
        
        # Build context
        context = "Relevant information from knowledge base:\n\n"
        for doc in documents:
            context += f"- Source: {doc['source']}\n  {doc['content']}\n\n"
        
        # Prepare full query
        full_query = f"{context}\nUser question: {user_query}"
        
        # Execute agent
        response = await self.agent.run(full_query)
        
        return response.text


async def run_rag_agent():
    """Run RAG-enabled agent"""
    
    search_endpoint = "https://your-search.search.windows.net"
    search_index = "documents"
    
    async with DefaultAzureCredential() as credential:
        ai_client = FoundryChatClient(credential=credential)

        rag_agent = RAGAgent(search_endpoint, search_index)
        await rag_agent.initialize(ai_client, credential)

        print("=== RAG Agent ===\n")

        # Query with retrieved context
        queries = [
            "What are the benefits of electric vehicles?",
            "How does solar energy work?",
            "What are the latest AI developments?"
        ]

        for query in queries:
            print(f"User: {query}")
            response = await rag_agent.query_with_context(query)
            print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(run_rag_agent())   # requires an Azure AI Search index with "content" and "source" fields
```

Doing retrieval by hand keeps every step visible. The framework also ships it as a context provider, which runs the search before each model call without changing your prompts:

```python
from agent_framework import Agent
from agent_framework.azure import AzureAISearchContextProvider
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import DefaultAzureCredential

credential = DefaultAzureCredential()
search = AzureAISearchContextProvider(
    endpoint="https://your-search.search.windows.net",
    index_name="documents",
    credential=credential,
    top_k=3,
)
agent = Agent(
    client=FoundryChatClient(credential=credential),
    instructions="Answer from the provided context. Say so if it doesn't cover the question.",
    context_providers=[search],
)
```

---

## Integration Recipes

### Recipe 8: Azure Functions Integration - C#

**Scenario:** Deploy agent as Azure Function

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using Microsoft.Agents.AI;
using System.ComponentModel;

public class AgentFunction
{
    private readonly ILogger<AgentFunction> _logger;

    public AgentFunction(ILogger<AgentFunction> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// HTTP-triggered function exposing agent
    /// </summary>
    [Function("QueryAgent")]
    public async Task<IActionResult> RunAgent(
        [HttpTrigger(AuthorizationLevel.Anonymous, "post", Route = "agent/query")]
        HttpRequest req)
    {
        _logger.LogInformation("Agent query received");

        try
        {
            // Parse request
            var requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            string query = data?.query;

            if (string.IsNullOrEmpty(query))
            {
                return new BadRequestObjectResult("Query is required");
            }

            // Create agent
            var agent = CreateAgent();

            // Execute
            var result = await agent.RunAsync(query);

            // Return response
            return new OkObjectResult(new
            {
                success = true,
                response = result,
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            _logger.LogError($"Error: {ex.Message}");
            
            return new ObjectResult(new
            {
                success = false,
                error = ex.Message
            })
            {
                StatusCode = 500
            };
        }
    }

    [Description("Get current weather for a location")]
    private static string GetWeather(
        [Description("City name")] string location)
    {
        return $"Weather in {location}: 20°C, Sunny";
    }

    private AIAgent CreateAgent()
    {
        var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
        var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME");

        var client = new AzureOpenAIClient(
            new Uri(endpoint),
            new DefaultAzureCredential()
        );

        var chatClient = client.GetChatClient(deploymentName);
        
        return chatClient.AsAIAgent(
            instructions: "You are a helpful assistant",
            tools: [AIFunctionFactory.Create(GetWeather)]
        );
    }
}
```

---

## Troubleshooting Patterns

### Pattern 1: Debugging Agent Responses

```python
"""
Debug Pattern: Inspect agent internals
"""

class DebuggableAgent(Agent):
    """Agent with debugging capabilities"""
    
    async def run_with_debug(self, query: str, debug: bool = True):
        """Run with detailed debugging"""
        
        if debug:
            print(f"[DEBUG] Query: {query}")
        
        # Run agent
        response = await self.run(query)
        
        if debug:
            print(f"[DEBUG] Response length: {len(response.text)}")
            print(f"[DEBUG] Response preview: {response.text[:200]}...")
            if hasattr(response, 'metadata'):
                print(f"[DEBUG] Metadata: {response.metadata}")
        
        return response
```

### Pattern 2: Monitoring Tool Execution

```python
"""
Monitoring Pattern: Track tool calls
"""

class MonitoredAgent(Agent):
    """Agent that monitors tool execution"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_calls = []
        self.tool_errors = []
    
    async def track_tool_execution(self, tool_name: str, args: dict):
        """Track tool calls"""
        execution = {
            "tool": tool_name,
            "args": args,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        self.tool_calls.append(execution)
        return len(self.tool_calls) - 1  # Return index
    
    def get_tool_statistics(self):
        """Get tool execution statistics"""
        return {
            "total_calls": len(self.tool_calls),
            "total_errors": len(self.tool_errors),
            "tools_used": list(set(t["tool"] for t in self.tool_calls))
        }
```

---

**These recipes provide production-ready patterns for common Agent Framework scenarios. Adapt and extend them for your specific use cases.**

For comprehensive conceptual information, see `microsoft_agent_framework_comprehensive_guide.md`.


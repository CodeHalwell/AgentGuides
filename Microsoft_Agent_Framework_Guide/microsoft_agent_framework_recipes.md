# Microsoft Agent Framework - Recipes & Code Patterns
## October 2025 Release - Production-Ready Examples

**Document Version:** 1.0  
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
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential


async def create_simple_agent():
    """Create a basic chat agent"""
    
    # Credentials
    async with AzureCliCredential() as credential:
        # Initialize Azure AI client
        async with AzureAIAgentClient(async_credential=credential) as client:
            
            # Create agent with system instructions
            agent = ChatAgent(
                chat_client=client,
                instructions="You are a helpful assistant. Provide accurate, concise answers."
            )
            
            # Run agent
            response = await agent.run("What is artificial intelligence?")
            print(f"Assistant: {response.text}")


async def multi_turn_conversation():
    """Multi-turn conversation example"""
    
    async with AzureCliCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential) as client:
            agent = ChatAgent(
                chat_client=client,
                instructions="You are a Python programming tutor."
            )
            
            # Create a thread for conversation context
            thread = agent.get_new_thread()
            
            # First turn
            response1 = await agent.run(
                "What is a list in Python?",
                thread=thread
            )
            print(f"Q1: {response1.text}\n")
            
            # Second turn - context maintained
            response2 = await agent.run(
                "How do I add items to it?",
                thread=thread
            )
            print(f"Q2: {response2.text}\n")
            
            # Third turn - full conversation maintained
            response3 = await agent.run(
                "Can you provide a complete example?",
                thread=thread
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
        var agent = chatClient.CreateAIAgent(
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
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from azure.core.exceptions import AzureError

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
    
    async def run_with_retry(self, agent: ChatAgent, query: str):
        """Execute agent query with retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: {query}")
                response = await agent.run(query)
                logger.info(f"Success: Got response of {len(response.text)} chars")
                return response
                
            except AzureError as e:
                logger.warning(f"Azure error on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts")
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                raise
    
    async def execute_safely(self, query: str):
        """Execute with error handling"""
        
        try:
            async with AzureCliCredential() as credential:
                async with AzureAIAgentClient(async_credential=credential) as client:
                    agent = ChatAgent(
                        chat_client=client,
                        instructions="You are helpful"
                    )
                    
                    response = await self.run_with_retry(agent, query)
                    return response.text
                    
        except AzureError as e:
            logger.error(f"Azure service error: {e}")
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
from enum import Enum
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
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
        self.extraction_agent = ChatAgent(
            chat_client=client,
            instructions="You are an expert at extracting key information from documents. "
                        "Focus on facts, figures, and important details."
        )
        
        self.analysis_agent = ChatAgent(
            chat_client=client,
            instructions="You are a data analyst. Analyse extracted information "
                        "and identify trends, patterns, and insights."
        )
        
        self.summary_agent = ChatAgent(
            chat_client=client,
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
            timestamp=datetime.utcnow().isoformat()
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
        async with AzureAIAgentClient(async_credential=credential) as client:
            workflow = DocumentAnalysisWorkflow(client)
            
            print("=== Document Analysis Workflow ===\n")
            result = await workflow.process_document(sample_document)
            
            print("\n=== Results ===")
            print(f"\nExtraction:\n{result.research}")
            print(f"\nAnalysis:\n{result.analysis}")
            print(f"\nRecommendations:\n{result.recommendations}")


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(run_workflow())
```

### Recipe 5: Agent with Multiple Tools - Python

**Scenario:** Create agents with multiple integrated tools

```python
"""
Agent with Multiple Tools Recipe
Demonstrates tool composition for complex scenarios
"""

import asyncio
from typing import Annotated
from agent_framework import ChatAgent, ai_function
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential


# Define multiple tools
@ai_function(description="Get the current weather for a location")
async def get_weather(location: Annotated[str, "City name"]) -> str:
    """Fetch weather information"""
    weather_data = {
        "London": "15°C, cloudy",
        "Paris": "16°C, sunny",
        "Amsterdam": "14°C, rainy"
    }
    return weather_data.get(location, "Unknown location")


@ai_function(description="Calculate travel time between two cities")
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


@ai_function(description="Look up hotel availability and prices")
async def check_hotels(
    city: Annotated[str, "City name"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"]
) -> str:
    """Check hotel availability"""
    return f"Found 12 hotels in {city} available from {check_in}. Average price: £85/night"


async def run_travel_agent():
    """Create agent with multiple tools for travel planning"""
    
    async with AzureCliCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential) as client:
            
            # Create agent with multiple tools
            agent = ChatAgent(
                chat_client=client,
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

**Scenario:** Agents with persistent conversation history

```python
"""
Persistent Memory Recipe
Demonstrates long-term context maintenance using Azure Cosmos DB
"""

import asyncio
import json
from datetime import datetime
from azure.cosmos.aio import CosmosClient
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential


class PersistentMemoryAgent:
    """Agent with Cosmos DB-backed memory"""
    
    def __init__(self, cosmos_connection: str):
        self.cosmos_client = CosmosClient.from_connection_string(cosmos_connection)
        self.database = None
        self.container = None
    
    async def initialize(self):
        """Initialize Cosmos DB"""
        self.database = self.cosmos_client.get_database_client("agent_memory")
        self.container = self.database.get_container_client("conversations")
    
    async def save_conversation_turn(self, user_id: str, query: str, response: str):
        """Save conversation turn to persistent storage"""
        
        item = {
            "id": f"{user_id}-{datetime.utcnow().timestamp()}",
            "user_id": user_id,
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
            "ttl": 7776000  # 90 days
        }
        
        await self.container.create_item(body=item)
    
    async def load_conversation_history(self, user_id: str, limit: int = 10) -> str:
        """Load recent conversation history"""
        
        query = "SELECT * FROM c WHERE c.user_id = @user_id ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        
        items = []
        async for item in self.container.query_items(
            query=query,
            parameters=[
                {"name": "@user_id", "value": user_id},
                {"name": "@limit", "value": limit}
            ]
        ):
            items.append(item)
        
        # Format history for context
        history = []
        for item in reversed(items):  # Reverse to chronological order
            history.append(f"User: {item['query']}")
            history.append(f"Agent: {item['response']}")
        
        return "\n".join(history)
    
    async def execute_with_context(self, user_id: str, query: str, agent: ChatAgent) -> str:
        """Execute agent query with historical context"""
        
        # Load history
        history = await self.load_conversation_history(user_id)
        
        # Prepare context
        full_query = f"Previous conversation:\n{history}\n\nNew query: {query}"
        
        # Execute
        response = await agent.run(full_query)
        
        # Save turn
        await self.save_conversation_turn(user_id, query, response.text)
        
        return response.text
    
    async def close(self):
        """Cleanup"""
        await self.cosmos_client.close()


async def run_persistent_agent():
    """Run agent with persistent memory"""
    
    cosmos_connection = "DefaultEndpointsProtocol=https://..."  # Your connection string
    
    agent_store = PersistentMemoryAgent(cosmos_connection)
    await agent_store.initialize()
    
    try:
        async with AzureCliCredential() as credential:
            async with AzureAIAgentClient(async_credential=credential) as client:
                agent = ChatAgent(
                    chat_client=client,
                    instructions="You are a helpful personal assistant who remembers previous conversations."
                )
                
                # Simulate multi-session conversation
                user_id = "user-123"
                
                print("=== Persistent Memory Agent ===\n")
                
                # Session 1
                print("Session 1:")
                response1 = await agent_store.execute_with_context(
                    user_id, 
                    "My name is Alice and I like Python programming",
                    agent
                )
                print(f"Alice: {response1}\n")
                
                # Later session - agent remembers
                print("Session 2 (later):")
                response2 = await agent_store.execute_with_context(
                    user_id,
                    "What did I tell you about my interests?",
                    agent
                )
                print(f"Alice: {response2}")
                
    finally:
        await agent_store.close()


if __name__ == "__main__":
    # asyncio.run(run_persistent_agent())  # Requires Cosmos DB setup
    print("Run with valid Cosmos DB connection string")
```

### Recipe 7: Agent with RAG (Retrieval-Augmented Generation) - Python

**Scenario:** Agents that retrieve knowledge from vector store

```python
"""
RAG Agent Recipe
Demonstrates retrieval-augmented generation using Azure AI Search
"""

import asyncio
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import Vector
from azure.identity.aio import DefaultAzureCredential
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient


class RAGAgent:
    """Agent with retrieval-augmented generation"""
    
    def __init__(self, search_endpoint: str, search_index: str):
        self.search_endpoint = search_endpoint
        self.search_index = search_index
        self.search_client = None
        self.agent = None
    
    async def initialize(self, ai_client: AzureAIAgentClient):
        """Initialize RAG components"""
        
        credential = DefaultAzureCredential()
        
        # Create search client
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index,
            credential=credential
        )
        
        # Create agent
        self.agent = ChatAgent(
            chat_client=ai_client,
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
        async with AzureAIAgentClient(async_credential=credential) as ai_client:
            
            rag_agent = RAGAgent(search_endpoint, search_index)
            await rag_agent.initialize(ai_client)
            
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
    # asyncio.run(run_rag_agent())  # Requires Azure AI Search setup
    print("Run with valid Azure AI Search configuration")
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
        
        return chatClient.CreateAIAgent(
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

class DebuggableAgent(ChatAgent):
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

class MonitoredAgent(ChatAgent):
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

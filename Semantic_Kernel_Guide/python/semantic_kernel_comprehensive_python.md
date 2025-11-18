# Semantic Kernel Comprehensive Guide (Python)

**Complete Python Reference for Building AI Agents and Agentic Systems**

Last Updated: November 2025
Python Version: 3.9+
Semantic Kernel: 1.38.0+

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Kernel Initialization](#2-kernel-initialization)
3. [Service Configuration](#3-service-configuration)
4. [Semantic Functions](#4-semantic-functions)
5. [Native Functions](#5-native-functions)
6. [Plugin Development](#6-plugin-development)
7. [Simple Agents](#7-simple-agents)
8. [ChatCompletionAgent](#8-chatcompletionagent)
9. [Memory Systems](#9-memory-systems)
10. [Vector Store v1.34 (2025)](#10-vector-store-v134-2025)
11. [Embeddings & Semantic Search](#11-embeddings--semantic-search)
12. [Sequential Planner](#12-sequential-planner)
13. [Stepwise Planner](#13-stepwise-planner)
14. [Action Planner](#14-action-planner)
15. [Model Context Protocol (MCP) - 2025](#15-model-context-protocol-mcp---2025)
16. [MCP Client Implementation](#16-mcp-client-implementation)
17. [MCP Server Creation](#17-mcp-server-creation)
18. [Google A2A Protocol Integration - 2025](#18-google-a2a-protocol-integration---2025)
19. [Microsoft Agent Framework Integration - 2025](#19-microsoft-agent-framework-integration---2025)
20. [Structured Output & Validation](#20-structured-output--validation)
21. [Error Handling & Resilience](#21-error-handling--resilience)
22. [Observability & Telemetry](#22-observability--telemetry)
23. [Best Practices & Patterns](#23-best-practices--patterns)

---

## 1. Installation & Setup

### Basic Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Semantic Kernel with extras
pip install "semantic-kernel[openai,azure]"

# Additional dependencies
pip install python-dotenv tenacity opentelemetry-sdk pydantic
```

### Full Installation (All Features)

```bash
# Install with all connectors and tools
pip install "semantic-kernel[openai,azure,huggingface,qdrant,pinecone,weaviate,redis]"

# Development dependencies
pip install "semantic-kernel[dev]"  # Testing, linting

# Monitoring and observability
pip install opentelemetry-api opentelemetry-sdk azure-monitor-opentelemetry

# Production dependencies
pip install uvicorn gunicorn fastapi tenacity circuitbreaker
```

### Environment Configuration

Create `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Azure AI Search (Vector Store)
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=...

# Azure Key Vault
AZURE_KEY_VAULT_URL=https://your-vault.vault.azure.net/

# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...
```

Load environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Validate required variables
required_vars = ["OPENAI_API_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} environment variable not set")
```

---

## 2. Kernel Initialization

### Basic Kernel

```python
from semantic_kernel import Kernel

# Create kernel
kernel = Kernel()
```

### Kernel with Service

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os

kernel = Kernel()

# Add chat completion service
chat_service = OpenAIChatCompletion(
    model_id="gpt-4",
    api_key=os.environ["OPENAI_API_KEY"]
)

kernel.add_service(chat_service)
```

### Kernel Builder Pattern (Alternative)

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Using builder-like pattern
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        model_id="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"]
    )
)
```

---

## 3. Service Configuration

### OpenAI Service

```python
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding
)

# Chat completion
chat_service = OpenAIChatCompletion(
    model_id="gpt-4",
    api_key=os.environ["OPENAI_API_KEY"],
    org_id=os.environ.get("OPENAI_ORG_ID"),  # Optional
)

# Embeddings
embedding_service = OpenAITextEmbedding(
    model_id="text-embedding-ada-002",
    api_key=os.environ["OPENAI_API_KEY"]
)

kernel.add_service(chat_service)
kernel.add_service(embedding_service)
```

### Azure OpenAI Service

```python
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding
)
from azure.identity import DefaultAzureCredential

# With API Key
chat_service = AzureChatCompletion(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

# With Managed Identity (Production)
credential = DefaultAzureCredential()
chat_service = AzureChatCompletion(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    credential=credential
)

# Embeddings
embedding_service = AzureTextEmbedding(
    deployment_name=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

kernel.add_service(chat_service)
kernel.add_service(embedding_service)
```

### Multiple Services

```python
# Add multiple models for different purposes
kernel.add_service(
    OpenAIChatCompletion(model_id="gpt-4", api_key=api_key),
    service_id="gpt-4-service"
)

kernel.add_service(
    OpenAIChatCompletion(model_id="gpt-3.5-turbo", api_key=api_key),
    service_id="gpt-3.5-service"
)

# Use specific service
result = await kernel.invoke(
    function,
    service_id="gpt-4-service"
)
```

---

## 4. Semantic Functions

### Basic Semantic Function

{% raw %}
```python
# Create function from prompt
prompt = "What is the capital of {{$country}}?"
function = kernel.create_function_from_prompt(prompt)

# Invoke
result = await kernel.invoke(function, country="France")
print(result)  # "The capital of France is Paris."
```
{% endraw %}

### Function with Settings

{% raw %}
```python
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

prompt = "Summarize the following text in 3 bullet points:\n\n{{$input}}"

# Configure execution settings
settings = OpenAIChatPromptExecutionSettings(
    max_tokens=500,
    temperature=0.7,
    top_p=0.9,
    presence_penalty=0.0,
    frequency_penalty=0.0
)

function = kernel.create_function_from_prompt(
    prompt,
    function_name="summarize",
    description="Summarizes text into bullet points",
    execution_settings=settings
)

result = await kernel.invoke(function, input="Long text here...")
```
{% endraw %}

### Template Variables

{% raw %}
```python
# Multiple variables
prompt = """
Task: {{$task}}
Context: {{$context}}
Format: {{$format}}

Please complete the task above.
"""

function = kernel.create_function_from_prompt(prompt)

result = await kernel.invoke(
    function,
    task="Analyze sentiment",
    context="Customer review: This product is amazing!",
    format="JSON with score and explanation"
)
```
{% endraw %}

---

## 5. Native Functions

### Basic Native Function

```python
from semantic_kernel.functions import kernel_function
from typing import Annotated

class MathPlugin:
    @kernel_function(
        name="add",
        description="Adds two numbers"
    )
    def add(
        self,
        a: Annotated[int, "First number"],
        b: Annotated[int, "Second number"]
    ) -> Annotated[int, "Sum of a and b"]:
        return a + b

    @kernel_function(
        name="multiply",
        description="Multiplies two numbers"
    )
    def multiply(self, a: int, b: int) -> int:
        return a * b

# Add to kernel
math_plugin = kernel.add_plugin(MathPlugin(), plugin_name="Math")

# Invoke
result = await kernel.invoke(math_plugin["add"], a=5, b=3)
print(result)  # 8
```

### Async Native Function

```python
import aiohttp
from semantic_kernel.functions import kernel_function

class WebPlugin:
    @kernel_function(
        name="fetch_url",
        description="Fetches content from a URL"
    )
    async def fetch_url(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

web_plugin = kernel.add_plugin(WebPlugin(), plugin_name="Web")

result = await kernel.invoke(
    web_plugin["fetch_url"],
    url="https://api.example.com/data"
)
```

### Native Function with Context

```python
from semantic_kernel.functions import kernel_function
from semantic_kernel import KernelContext

class ContextAwarePlugin:
    @kernel_function(description="Gets user info from context")
    async def get_user_info(self, context: KernelContext) -> str:
        # Access context variables
        user_id = context.variables.get("user_id")
        session_id = context.variables.get("session_id")

        return f"User: {user_id}, Session: {session_id}"

plugin = kernel.add_plugin(ContextAwarePlugin(), plugin_name="Context")
```

---

## 6. Plugin Development

### Complete Plugin Example

```python
from semantic_kernel.functions import kernel_function
from typing import Annotated
import aiohttp
import json

class CustomerServicePlugin:
    """Plugin for customer service operations"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    @kernel_function(
        name="get_customer",
        description="Retrieves customer information by ID"
    )
    async def get_customer(
        self,
        customer_id: Annotated[str, "The customer ID"]
    ) -> Annotated[str, "Customer information as JSON"]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}/customers/{customer_id}"

            async with session.get(url, headers=headers) as response:
                data = await response.json()
                return json.dumps(data)

    @kernel_function(
        name="create_ticket",
        description="Creates a support ticket"
    )
    async def create_ticket(
        self,
        customer_id: Annotated[str, "Customer ID"],
        issue: Annotated[str, "Issue description"],
        priority: Annotated[str, "Priority level: low, medium, high"] = "medium"
    ) -> Annotated[str, "Ticket ID"]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            url = f"{self.base_url}/tickets"
            payload = {
                "customer_id": customer_id,
                "issue": issue,
                "priority": priority
            }

            async with session.post(url, headers=headers, json=payload) as response:
                data = await response.json()
                return data["ticket_id"]

# Register plugin
cs_plugin = kernel.add_plugin(
    CustomerServicePlugin(
        api_key=os.environ["CS_API_KEY"],
        base_url="https://api.example.com"
    ),
    plugin_name="CustomerService"
)
```

### OpenAPI Plugin

```python
from semantic_kernel.connectors.openapi import OpenAPIFunctionExecutionParameters

# Import OpenAPI plugin from specification
openapi_plugin = await kernel.import_plugin_from_openapi(
    plugin_name="PetStore",
    openapi_document_path="./petstore-openapi.json",
    execution_parameters=OpenAPIFunctionExecutionParameters(
        server_url_override="https://api.petstore.com",
        enable_dynamic_payload=True,
        enable_payload_namespacing=True
    )
)

# Invoke OpenAPI function
result = await kernel.invoke(
    openapi_plugin["getPetById"],
    petId="12345"
)
```

---

## 7. Simple Agents

### Basic Agent

```python
from semantic_kernel.agents import ChatCompletionAgent

# Create simple agent
agent = ChatCompletionAgent(
    kernel=kernel,
    name="assistant",
    instructions="You are a helpful AI assistant. Answer questions clearly and concisely."
)

# Chat with agent
result = await agent.invoke("What is Semantic Kernel?")
print(result)
```

### Agent with Tools

```python
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    @kernel_function(description="Gets current weather")
    async def get_weather(self, city: str) -> str:
        # Mock weather data
        return f"The weather in {city} is sunny, 72°F"

# Add plugin to kernel
weather_plugin = kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")

# Create agent with access to tools
agent = ChatCompletionAgent(
    kernel=kernel,
    name="weather_assistant",
    instructions="You are a weather assistant. Use the get_weather tool to answer questions about weather."
)

# Agent will automatically call tools
result = await agent.invoke("What's the weather in Seattle?")
print(result)  # Agent calls get_weather("Seattle") and returns result
```

---

## 8. ChatCompletionAgent

### Advanced Agent with Group Chat

```python
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.contents import ChatMessageContent, AuthorRole

# Create multiple agents
researcher = ChatCompletionAgent(
    kernel=kernel,
    name="researcher",
    instructions="""You are a research specialist. Your role is to:
    1. Gather and verify information
    2. Cite sources
    3. Provide factual, accurate answers
    """
)

writer = ChatCompletionAgent(
    kernel=kernel,
    name="writer",
    instructions="""You are a content writer. Your role is to:
    1. Take research and create engaging content
    2. Ensure clarity and readability
    3. Format content appropriately
    """
)

editor = ChatCompletionAgent(
    kernel=kernel,
    name="editor",
    instructions="""You are an editor. Your role is to:
    1. Review content for quality
    2. Fix grammar and style issues
    3. Ensure consistency
    """
)

# Create group chat
group_chat = AgentGroupChat(researcher, writer, editor)

# Add user message
await group_chat.add_chat_message(
    ChatMessageContent(
        role=AuthorRole.USER,
        content="Write a blog post about Semantic Kernel with sources"
    )
)

# Run conversation
async for message in group_chat.invoke():
    print(f"{message.role} ({message.name}): {message.content}")
```

### Agent with Termination Strategy

```python
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents.strategies import TerminationStrategy

class MaxTurnsTermination(TerminationStrategy):
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turn_count = 0

    async def should_terminate(
        self,
        agent,
        history: list[ChatMessageContent]
    ) -> bool:
        self.turn_count += 1

        # Terminate if max turns reached
        if self.turn_count >= self.max_turns:
            return True

        # Terminate if "DONE" appears in message
        if history and "DONE" in history[-1].content.upper():
            return True

        return False

# Create group chat with termination
group_chat = AgentGroupChat(
    researcher,
    writer,
    termination_strategy=MaxTurnsTermination(max_turns=15)
)
```

---

## 9. Memory Systems

### Volatile Memory (In-Memory)

```python
from semantic_kernel.memory import VolatileMemoryStore, SemanticTextMemory

# Create volatile memory store
memory_store = VolatileMemoryStore()

# Create semantic memory
memory = SemanticTextMemory(
    storage=memory_store,
    embeddings_generator=embedding_service
)

# Add to kernel
kernel.add_plugin(memory, plugin_name="Memory")

# Save memory
await memory.save(
    collection="facts",
    text="The Eiffel Tower is in Paris, France",
    id="fact_1"
)

# Search memory
results = await memory.search(
    collection="facts",
    query="Where is the Eiffel Tower?",
    limit=3,
    min_relevance_score=0.7
)

for result in results:
    print(f"Score: {result.relevance}, Text: {result.text}")
```

---

## 10. Vector Store v1.34 (2025)

### New Unified Vector Store API

The v1.34 release introduced a completely overhauled vector store system with unified APIs across all providers.

#### Azure AI Search

```python
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchMemoryStore
from semantic_kernel.memory import SemanticTextMemory
from azure.identity import DefaultAzureCredential

# Create Azure AI Search memory store (v1.34+)
memory_store = AzureAISearchMemoryStore(
    search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    credential=DefaultAzureCredential(),  # Or api_key
    index_name="semantic-kernel-index",
    embedding_dimensions=1536  # For text-embedding-ada-002
)

# Create memory with unified API
memory = SemanticTextMemory(
    storage=memory_store,
    embeddings_generator=embedding_service
)

# Unified operations
await memory.save_information(
    collection="documents",
    id="doc_1",
    text="Semantic Kernel is a lightweight SDK...",
    description="SK Overview",
    additional_metadata={"category": "documentation", "version": "1.0"}
)

# Search with new features
results = await memory.search(
    collection="documents",
    query="What is Semantic Kernel?",
    limit=5,
    min_relevance_score=0.75,
    filter={"category": "documentation"}  # NEW: Metadata filtering
)
```

#### Qdrant Vector Store

```python
from semantic_kernel.connectors.memory.qdrant import QdrantMemoryStore
from qdrant_client import QdrantClient

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

# Create Qdrant memory store (v1.34+)
memory_store = QdrantMemoryStore(
    client=qdrant_client,
    vector_size=1536,
    collection_name="sk_memories"
)

memory = SemanticTextMemory(
    storage=memory_store,
    embeddings_generator=embedding_service
)

# Same unified API as Azure AI Search
await memory.save_information(
    collection="products",
    id="prod_1",
    text="High-performance running shoes with advanced cushioning",
    additional_metadata={"price": 129.99, "category": "footwear"}
)
```

#### Weaviate Vector Store

```python
from semantic_kernel.connectors.memory.weaviate import WeaviateMemoryStore
import weaviate

# Initialize Weaviate client
weaviate_client = weaviate.Client(
    url=os.environ["WEAVIATE_URL"],
    auth_client_secret=weaviate.AuthApiKey(os.environ["WEAVIATE_API_KEY"])
)

# Create Weaviate memory store
memory_store = WeaviateMemoryStore(
    client=weaviate_client,
    embedding_dimensions=1536,
    class_name="SemanticKernelMemory"
)

memory = SemanticTextMemory(
    storage=memory_store,
    embeddings_generator=embedding_service
)
```

### Vector Store Performance Improvements (v1.34)

```python
# Batch operations (NEW in v1.34)
documents = [
    {"id": "1", "text": "Document 1 content", "metadata": {"source": "web"}},
    {"id": "2", "text": "Document 2 content", "metadata": {"source": "pdf"}},
    {"id": "3", "text": "Document 3 content", "metadata": {"source": "api"}},
]

# Batch save (much faster than individual saves)
await memory.save_batch(
    collection="documents",
    records=documents
)

# Parallel search across collections (NEW)
from asyncio import gather

results = await gather(
    memory.search(collection="documents", query="AI agents"),
    memory.search(collection="code", query="AI agents"),
    memory.search(collection="papers", query="AI agents")
)

all_results = [item for sublist in results for item in sublist]
all_results.sort(key=lambda x: x.relevance, reverse=True)
```

---

## 11. Embeddings & Semantic Search

### Generate Embeddings

```python
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

embedding_service = OpenAITextEmbedding(
    model_id="text-embedding-ada-002",
    api_key=os.environ["OPENAI_API_KEY"]
)

# Generate embedding
text = "Semantic Kernel is an SDK for building AI applications"
embedding = await embedding_service.generate_embeddings([text])

print(f"Embedding dimensions: {len(embedding[0])}")  # 1536
print(f"First 5 values: {embedding[0][:5]}")
```

### Semantic Search Implementation

```python
import numpy as np
from typing import List, Tuple

class SemanticSearchEngine:
    def __init__(self, memory: SemanticTextMemory):
        self.memory = memory

    async def index_documents(
        self,
        collection: str,
        documents: List[dict]
    ):
        """Index multiple documents"""
        for doc in documents:
            await self.memory.save_information(
                collection=collection,
                id=doc["id"],
                text=doc["text"],
                description=doc.get("description", ""),
                additional_metadata=doc.get("metadata", {})
            )

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Tuple[str, float]]:
        """Search documents semantically"""
        results = await self.memory.search(
            collection=collection,
            query=query,
            limit=top_k,
            min_relevance_score=min_score
        )

        return [(r.text, r.relevance) for r in results]

    async def hybrid_search(
        self,
        collection: str,
        query: str,
        keywords: List[str],
        top_k: int = 5
    ):
        """Combine semantic and keyword search"""
        # Semantic search
        semantic_results = await self.search(collection, query, top_k=top_k*2)

        # Keyword filtering
        filtered_results = [
            (text, score) for text, score in semantic_results
            if any(kw.lower() in text.lower() for kw in keywords)
        ]

        return filtered_results[:top_k]

# Usage
search_engine = SemanticSearchEngine(memory)

await search_engine.index_documents(
    collection="kb",
    documents=[
        {"id": "1", "text": "Semantic Kernel supports Python, .NET, and Java"},
        {"id": "2", "text": "Use plugins to extend SK functionality"},
        {"id": "3", "text": "Agents can coordinate in multi-agent systems"},
    ]
)

results = await search_engine.search("kb", "programming languages in SK")
for text, score in results:
    print(f"[{score:.2f}] {text}")
```

---

## 12. Sequential Planner

### Basic Sequential Planner

```python
from semantic_kernel.planners import SequentialPlanner

# Create planner
planner = SequentialPlanner(kernel)

# Create plan
ask = "I want to send an email to john@example.com saying the weather in Seattle is nice"
plan = await planner.create_plan(ask)

# Execute plan
result = await plan.invoke(kernel)
print(result)
```

### Sequential Planner with Custom Plugins

```python
from semantic_kernel.functions import kernel_function
from semantic_kernel.planners import SequentialPlanner

class EmailPlugin:
    @kernel_function(description="Sends an email")
    async def send_email(
        self,
        to: str,
        subject: str,
        body: str
    ) -> str:
        # Mock email sending
        return f"Email sent to {to}"

class WeatherPlugin:
    @kernel_function(description="Gets weather for a city")
    async def get_weather(self, city: str) -> str:
        return f"The weather in {city} is sunny, 72°F"

# Add plugins
kernel.add_plugin(EmailPlugin(), "Email")
kernel.add_plugin(WeatherPlugin(), "Weather")

# Create and execute plan
planner = SequentialPlanner(kernel)
plan = await planner.create_plan(
    "Get the weather in Paris and email it to alice@example.com"
)

print("Plan steps:")
for step in plan.steps:
    print(f"  - {step.name}: {step.description}")

result = await plan.invoke(kernel)
```

---

## 13. Stepwise Planner

### Stepwise Planner Implementation

```python
from semantic_kernel.planners import StepwisePlanner
from semantic_kernel.planners.stepwise_planner import StepwisePlannerConfig

# Configure planner
config = StepwisePlannerConfig(
    max_iterations=10,
    min_iteration_time_ms=1000,
    max_tokens=4000
)

# Create planner
planner = StepwisePlanner(kernel, config)

# Execute with dynamic planning
ask = "Research the latest AI trends and create a summary report"
result = await planner.invoke(ask)

print(f"Final result: {result}")
print(f"\nSteps taken:")
for step in result.metadata.get("steps", []):
    print(f"  {step}")
```

---

## 14. Action Planner

### Action Planner for Goal-Oriented Tasks

```python
from semantic_kernel.planners import ActionPlanner

# Create action planner
planner = ActionPlanner(kernel)

# Generate action plan
ask = "What is the current population of Tokyo?"
plan = await planner.create_plan(ask)

# Execute
result = await plan.invoke(kernel)
print(result)
```

---

## 15. Model Context Protocol (MCP) - 2025

### MCP Overview

Model Context Protocol (MCP) is a standardized protocol for connecting AI applications to external data sources and tools. Introduced in late 2024, it's now fully supported in Semantic Kernel Python v1.34+.

**Key Features:**
- Standardized client-server architecture
- Tool/resource discovery and invocation
- Prompt template sharing
- Cross-application interoperability

### MCP Architecture

```
┌─────────────────┐
│  SK Application │
│   (MCP Client)  │
└────────┬────────┘
         │
         │ MCP Protocol
         │
┌────────┴────────┐
│   MCP Server    │
│  (Tools/Data)   │
└─────────────────┘
```

---

## 16. MCP Client Implementation

### Basic MCP Client

```python
from semantic_kernel.connectors.mcp import MCPClient, MCPClientConfig

# Configure MCP client
config = MCPClientConfig(
    server_url="http://localhost:3000",  # MCP server endpoint
    timeout=30.0,
    retry_attempts=3
)

# Create client
mcp_client = MCPClient(config)

# Connect to server
await mcp_client.connect()

# List available tools
tools = await mcp_client.list_tools()
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description}")
    print(f"  Parameters: {tool.input_schema}")

# List available resources
resources = await mcp_client.list_resources()
for resource in resources:
    print(f"Resource: {resource.uri}")
    print(f"  Type: {resource.mime_type}")
```

### Using MCP Tools with SK

```python
from semantic_kernel.connectors.mcp import MCPToolPlugin

# Create MCP tool plugin
mcp_plugin = await MCPToolPlugin.from_mcp_server(
    server_url="http://localhost:3000",
    plugin_name="MCPTools"
)

# Add to kernel
kernel.add_plugin(mcp_plugin)

# Tools are now available to agents
agent = ChatCompletionAgent(
    kernel=kernel,
    name="mcp_agent",
    instructions="You have access to MCP tools. Use them to answer questions."
)

# Agent can now call MCP tools
result = await agent.invoke("Use the weather tool to get weather in London")
```

### Reading MCP Resources

```python
# Read resource from MCP server
resource_uri = "file:///data/customers.json"
resource_content = await mcp_client.read_resource(resource_uri)

print(f"Content: {resource_content.content}")
print(f"MIME Type: {resource_content.mime_type}")

# Use resource in SK memory
await memory.save_information(
    collection="mcp_data",
    id="customers",
    text=resource_content.content,
    additional_metadata={"source": "mcp", "uri": resource_uri}
)
```

### MCP Prompt Templates

```python
# List available prompts from MCP server
prompts = await mcp_client.list_prompts()

for prompt in prompts:
    print(f"Prompt: {prompt.name}")
    print(f"  Description: {prompt.description}")
    print(f"  Arguments: {prompt.arguments}")

# Get prompt
prompt_data = await mcp_client.get_prompt(
    name="summarize_document",
    arguments={"format": "bullet_points", "max_length": "500"}
)

# Use prompt in SK
function = kernel.create_function_from_prompt(
    prompt_data.messages[0].content,
    function_name="mcp_summarize"
)

result = await kernel.invoke(function, document="Long text here...")
```

---

## 17. MCP Server Creation

### Creating an MCP Server with SK

```python
from semantic_kernel.connectors.mcp import MCPServer, MCPServerConfig
from semantic_kernel.connectors.mcp.models import Tool, Resource, Prompt
from typing import Any, Dict

class SkMCPServer:
    def __init__(self, kernel: Kernel, port: int = 3000):
        self.kernel = kernel
        self.config = MCPServerConfig(
            name="SK MCP Server",
            version="1.0.0",
            port=port
        )
        self.server = MCPServer(self.config)

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register MCP protocol handlers"""

        # Tool handlers
        @self.server.list_tools_handler
        async def list_tools() -> list[Tool]:
            tools = []

            # Expose SK plugins as MCP tools
            for plugin_name, plugin in self.kernel.plugins.items():
                for function_name, function in plugin.items():
                    tools.append(Tool(
                        name=f"{plugin_name}.{function_name}",
                        description=function.description or "",
                        input_schema=self._function_to_schema(function)
                    ))

            return tools

        @self.server.call_tool_handler
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
            # Parse tool name
            plugin_name, function_name = name.split(".", 1)

            # Get function
            function = self.kernel.plugins[plugin_name][function_name]

            # Invoke
            result = await self.kernel.invoke(function, **arguments)

            return {"result": str(result)}

        # Resource handlers
        @self.server.list_resources_handler
        async def list_resources() -> list[Resource]:
            # Expose SK memory collections as resources
            resources = []

            # Example: expose memory collections
            collections = ["documents", "code", "knowledge"]
            for collection in collections:
                resources.append(Resource(
                    uri=f"memory://{collection}",
                    name=collection,
                    mime_type="application/json",
                    description=f"SK memory collection: {collection}"
                ))

            return resources

        @self.server.read_resource_handler
        async def read_resource(uri: str) -> Dict[str, Any]:
            # Parse URI
            if uri.startswith("memory://"):
                collection = uri.replace("memory://", "")

                # Get all memories from collection
                results = await self.kernel.memory.search(
                    collection=collection,
                    query="",  # Empty query returns all
                    limit=100
                )

                return {
                    "content": [
                        {"text": r.text, "metadata": r.metadata}
                        for r in results
                    ],
                    "mime_type": "application/json"
                }

            raise ValueError(f"Unknown resource URI: {uri}")

        # Prompt handlers
        @self.server.list_prompts_handler
        async def list_prompts() -> list[Prompt]:
            # Expose SK semantic functions as prompts
            prompts = []

            for plugin_name, plugin in self.kernel.plugins.items():
                for function_name, function in plugin.items():
                    if function.is_semantic:  # Only semantic functions
                        prompts.append(Prompt(
                            name=f"{plugin_name}.{function_name}",
                            description=function.description or "",
                            arguments=self._function_to_arguments(function)
                        ))

            return prompts

        @self.server.get_prompt_handler
        async def get_prompt(name: str, arguments: Dict[str, Any]) -> Dict:
            plugin_name, function_name = name.split(".", 1)
            function = self.kernel.plugins[plugin_name][function_name]

            # Return prompt template with arguments substituted
            prompt_template = function.prompt_template

            return {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_template.render(arguments)
                    }
                ]
            }

    def _function_to_schema(self, function) -> Dict:
        """Convert SK function to JSON schema"""
        # Simplified - implement full parameter schema conversion
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def _function_to_arguments(self, function) -> list[Dict]:
        """Convert SK function parameters to MCP arguments"""
        # Simplified - implement full parameter conversion
        return []

    async def start(self):
        """Start the MCP server"""
        await self.server.start()
        print(f"MCP Server running on port {self.config.port}")

    async def stop(self):
        """Stop the MCP server"""
        await self.server.stop()

# Usage
kernel = Kernel()
# ... add services, plugins ...

# Create and start MCP server
mcp_server = SkMCPServer(kernel, port=3000)
await mcp_server.start()

# Server now exposes SK functions, memory, and prompts via MCP
```

### Complete MCP Server Example

{% raw %}
```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

class DataPlugin:
    @kernel_function(description="Gets user data")
    async def get_user(self, user_id: str) -> str:
        return f"{{\"id\": \"{user_id}\", \"name\": \"John Doe\"}}"

    @kernel_function(description="Searches database")
    async def search(self, query: str) -> str:
        return f"Results for: {query}"

async def main():
    # Setup kernel
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        model_id="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"]
    ))

    # Add plugin
    kernel.add_plugin(DataPlugin(), "Data")

    # Create and start MCP server
    server = SkMCPServer(kernel, port=3000)

    try:
        await server.start()

        # Keep running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())
```
{% endraw %}

---

## 18. Google A2A Protocol Integration - 2025

### A2A Overview

The Agent-to-Agent (A2A) Protocol, developed by Google, provides a standardized way for agents from different frameworks to communicate. SK Python added experimental A2A support in early 2025.

**Key Features:**
- Framework-agnostic agent communication
- Standardized message format
- Agent discovery and registration
- Routing and coordination

### A2A Message Format

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

class A2AMessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"

@dataclass
class A2AMessage:
    """Standardized A2A message"""
    id: str
    type: A2AMessageType
    sender: str  # Agent ID
    recipient: str  # Agent ID or broadcast
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### SK Agent with A2A Support

```python
from semantic_kernel.connectors.a2a import A2AAgent, A2AMessageBus

class SkA2AAgent:
    def __init__(
        self,
        agent: ChatCompletionAgent,
        message_bus: A2AMessageBus,
        agent_id: str
    ):
        self.agent = agent
        self.message_bus = message_bus
        self.agent_id = agent_id

        # Register with message bus
        self.message_bus.register_agent(agent_id, self)

    async def handle_message(self, message: A2AMessage):
        """Handle incoming A2A message"""

        if message.type == A2AMessageType.REQUEST:
            # Process request with SK agent
            result = await self.agent.invoke(message.payload.get("content", ""))

            # Send response
            response = A2AMessage(
                id=generate_id(),
                type=A2AMessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                payload={"result": str(result)},
                correlation_id=message.id
            )

            await self.message_bus.send(response)

        elif message.type == A2AMessageType.EVENT:
            # Handle event
            await self._handle_event(message)

    async def send_request(
        self,
        recipient: str,
        content: str
    ) -> A2AMessage:
        """Send request to another agent"""
        message = A2AMessage(
            id=generate_id(),
            type=A2AMessageType.REQUEST,
            sender=self.agent_id,
            recipient=recipient,
            payload={"content": content}
        )

        return await self.message_bus.send_and_wait(message)

    async def _handle_event(self, message: A2AMessage):
        """Handle event messages"""
        event_type = message.payload.get("event_type")
        print(f"Agent {self.agent_id} received event: {event_type}")

# Usage
message_bus = A2AMessageBus()

# Create SK agents
researcher = ChatCompletionAgent(kernel, name="researcher", instructions="Research topics")
writer = ChatCompletionAgent(kernel, name="writer", instructions="Write content")

# Wrap in A2A
a2a_researcher = SkA2AAgent(researcher, message_bus, "agent.sk.researcher")
a2a_writer = SkA2AAgent(writer, message_bus, "agent.sk.writer")

# Agents can now communicate via A2A
response = await a2a_writer.send_request(
    "agent.sk.researcher",
    "Research the latest in quantum computing"
)

print(f"Research result: {response.payload['result']}")
```

### A2A Message Bus Implementation

```python
import asyncio
from typing import Dict, Callable, Awaitable

class A2AMessageBus:
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}

    def register_agent(self, agent_id: str, agent: Any):
        """Register an agent with the bus"""
        self.agents[agent_id] = agent
        print(f"Registered agent: {agent_id}")

    async def send(self, message: A2AMessage):
        """Send message to recipient"""

        if message.recipient == "*":
            # Broadcast to all agents
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender:
                    await agent.handle_message(message)
        else:
            # Send to specific agent
            recipient = self.agents.get(message.recipient)
            if recipient:
                await recipient.handle_message(message)
            else:
                print(f"Agent not found: {message.recipient}")

        # If this is a response, resolve pending future
        if message.type == A2AMessageType.RESPONSE and message.correlation_id:
            future = self.pending_responses.get(message.correlation_id)
            if future:
                future.set_result(message)
                del self.pending_responses[message.correlation_id]

    async def send_and_wait(
        self,
        message: A2AMessage,
        timeout: float = 30.0
    ) -> A2AMessage:
        """Send message and wait for response"""

        # Create future for response
        future = asyncio.Future()
        self.pending_responses[message.id] = future

        # Send message
        await self.send(message)

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            del self.pending_responses[message.id]
            raise TimeoutError(f"No response received within {timeout}s")
```

### Cross-Framework A2A Example

```python
# SK Agent
sk_agent = SkA2AAgent(
    ChatCompletionAgent(kernel, name="sk_agent", instructions="SK agent"),
    message_bus,
    "agent.sk.main"
)

# Hypothetical LangGraph agent (with A2A support)
# langgraph_agent = LangGraphA2AAgent(
#     langgraph_graph,
#     message_bus,
#     "agent.langgraph.main"
# )

# SK agent can communicate with LangGraph agent via A2A
# response = await sk_agent.send_request(
#     "agent.langgraph.main",
#     "Process this data with your graph"
# )
```

---

## 19. Microsoft Agent Framework Integration - 2025

### Overview

Microsoft is unifying its agent ecosystem with a common Agent Framework SDK that connects Semantic Kernel, AutoGen, and Microsoft 365 Copilot agents. Python integration is in progress (following .NET release in March 2025).

**Status:** Experimental (Python SDK expected Q2 2025)

### Agent Framework Concepts

```python
# Conceptual API (subject to change)

from microsoft.agent_framework import (
    UnifiedAgent,
    AgentRegistry,
    AgentCapability,
    AgentGovernance
)

# Create agent with unified SDK
agent = UnifiedAgent(
    name="sk_unified_agent",
    framework="semantic-kernel",  # or "autogen", "copilot"
    capabilities=[
        AgentCapability.CHAT,
        AgentCapability.TOOL_USE,
        AgentCapability.MEMORY
    ],
    governance_policy=AgentGovernance(
        max_tokens_per_hour=100000,
        allowed_tools=["approved_tools/*"],
        content_filter_enabled=True,
        audit_logging=True
    )
)

# Register with enterprise registry
registry = AgentRegistry(
    endpoint=os.environ["AGENT_REGISTRY_ENDPOINT"],
    credential=DefaultAzureCredential()
)

await registry.register(agent)
```

### Cross-Framework Communication

```python
# SK agent communicating with AutoGen agent via Agent Framework

from microsoft.agent_framework import AgentClient

# Create client
client = AgentClient(
    registry_endpoint=os.environ["AGENT_REGISTRY_ENDPOINT"],
    credential=DefaultAzureCredential()
)

# Discover agents
agents = await client.discover_agents(
    capabilities=[AgentCapability.RESEARCH],
    framework="autogen"  # Find AutoGen agents
)

# Communicate with AutoGen agent from SK
autogen_agent = agents[0]
response = await client.send_message(
    to_agent=autogen_agent.id,
    message="Research quantum computing trends",
    sender_context={"framework": "semantic-kernel", "agent": "sk_agent"}
)

print(f"AutoGen response: {response.content}")
```

### Enterprise Governance

```python
from microsoft.agent_framework.governance import (
    GovernancePolicy,
    ContentFilter,
    AuditLogger
)

# Define governance policy
policy = GovernancePolicy(
    name="enterprise_policy",
    rules=[
        {
            "type": "token_limit",
            "max_tokens_per_hour": 100000,
            "max_tokens_per_request": 4000
        },
        {
            "type": "tool_allowlist",
            "allowed_tools": [
                "Microsoft.SemanticKernel.Plugins.Core.*",
                "CustomPlugin.ApprovedFunction"
            ]
        },
        {
            "type": "content_filter",
            "filters": ["pii", "profanity", "sensitive_data"]
        },
        {
            "type": "audit",
            "log_all_interactions": True,
            "retention_days": 90
        }
    ]
)

# Apply policy to agent
agent.apply_governance(policy)

# All agent actions are now governed
result = await agent.invoke("Process sensitive customer data")
# - Token usage tracked
# - Content filtered
# - Audit logged
# - Tool usage validated
```

---

## 20. Structured Output & Validation

### Pydantic Models

{% raw %}
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class CustomerInfo(BaseModel):
    customer_id: str = Field(..., description="Unique customer ID")
    name: str
    email: str
    phone: Optional[str] = None
    purchase_history: List[str] = Field(default_factory=list)
    lifetime_value: float = 0.0

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v

    @validator('lifetime_value')
    def validate_ltv(cls, v):
        if v < 0:
            raise ValueError('Lifetime value cannot be negative')
        return v

# Create prompt requesting structured output
prompt = """
Extract customer information from the following text and return as JSON:

Text: {{$input}}

Return JSON matching this schema:
{
  "customer_id": "string",
  "name": "string",
  "email": "string",
  "phone": "string (optional)",
  "purchase_history": ["string"],
  "lifetime_value": number
}
"""

function = kernel.create_function_from_prompt(prompt)

# Invoke and validate
text = "Customer John Doe (ID: C12345, email: john@example.com) has purchased items A, B, C. LTV: $1,250.50"
result = await kernel.invoke(function, input=text)

# Parse and validate with Pydantic
customer = CustomerInfo.model_validate_json(str(result))

print(f"Validated customer: {customer.name}, LTV: ${customer.lifetime_value}")
```
{% endraw %}

### Validation with Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import ValidationError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def get_structured_output(
    kernel: Kernel,
    function,
    model: type[BaseModel],
    **kwargs
) -> BaseModel:
    """Get structured output with automatic retry on validation failure"""

    # Invoke function
    result = await kernel.invoke(function, **kwargs)

    try:
        # Validate
        validated = model.model_validate_json(str(result))
        return validated
    except ValidationError as e:
        # Log validation error
        print(f"Validation failed: {e}")

        # Modify prompt to include error feedback
        error_feedback = f"Previous attempt failed validation: {e}. Please fix and return valid JSON."

        # Retry with feedback
        raise

# Usage
customer = await get_structured_output(
    kernel,
    function,
    CustomerInfo,
    input=text
)
```

---

## 21. Error Handling & Resilience

### Retry with Tenacity

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from semantic_kernel.exceptions import ServiceException

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(ServiceException)
)
async def invoke_with_retry(kernel: Kernel, function, **kwargs):
    """Invoke SK function with automatic retry"""
    return await kernel.invoke(function, **kwargs)

# Usage
try:
    result = await invoke_with_retry(kernel, function, input="query")
except Exception as e:
    logger.error(f"Failed after retries: {e}")
```

### Circuit Breaker

```python
from circuitbreaker import circuit
import asyncio

class ServiceCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.is_open = False
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker"""

        # Check if circuit is open
        if self.is_open:
            if (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                # Try to close circuit
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0  # Reset on success
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.error("Circuit breaker opened")

            raise

# Usage
circuit_breaker = ServiceCircuitBreaker()

try:
    result = await circuit_breaker.call(
        kernel.invoke,
        function,
        input="query"
    )
except Exception as e:
    logger.error(f"Call failed: {e}")
```

### Timeout Handling

```python
import asyncio

async def invoke_with_timeout(
    kernel: Kernel,
    function,
    timeout: float = 30.0,
    **kwargs
):
    """Invoke with timeout"""
    try:
        result = await asyncio.wait_for(
            kernel.invoke(function, **kwargs),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"Function invocation timed out after {timeout}s")
        raise

# Usage
result = await invoke_with_timeout(
    kernel,
    function,
    timeout=10.0,
    input="query"
)
```

---

## 22. Observability & Telemetry

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to console (for development)
console_exporter = ConsoleSpanExporter()
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(console_exporter)
)

# Export to OTLP endpoint (for production)
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Instrument SK operations
async def invoke_with_tracing(
    kernel: Kernel,
    function,
    span_name: str,
    **kwargs
):
    """Invoke SK function with tracing"""
    with tracer.start_as_current_span(span_name) as span:
        # Add attributes
        span.set_attribute("sk.function.name", function.name)
        span.set_attribute("sk.function.plugin", function.plugin_name)

        try:
            result = await kernel.invoke(function, **kwargs)

            # Track tokens (if available)
            if hasattr(result, 'metadata'):
                span.set_attribute("sk.tokens.input", result.metadata.get("input_tokens", 0))
                span.set_attribute("sk.tokens.output", result.metadata.get("output_tokens", 0))

            span.set_attribute("sk.status", "success")
            return result

        except Exception as e:
            span.set_attribute("sk.status", "error")
            span.set_attribute("sk.error.message", str(e))
            span.record_exception(e)
            raise

# Usage
result = await invoke_with_tracing(
    kernel,
    function,
    "semantic_function_invocation",
    input="query"
)
```

### Azure Monitor Integration

```python
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace, metrics

# Configure Azure Monitor
configure_azure_monitor(
    connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
)

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Custom metrics
token_counter = meter.create_counter(
    "sk.tokens.used",
    description="Total tokens used",
    unit="1"
)

latency_histogram = meter.create_histogram(
    "sk.function.latency",
    description="Function execution latency",
    unit="ms"
)

# Track metrics
async def invoke_with_metrics(kernel: Kernel, function, **kwargs):
    start_time = time.time()

    result = await kernel.invoke(function, **kwargs)

    # Record metrics
    latency = (time.time() - start_time) * 1000
    latency_histogram.record(latency, {"function": function.name})

    if hasattr(result, 'metadata'):
        tokens = result.metadata.get("total_tokens", 0)
        token_counter.add(tokens, {"function": function.name})

    return result
```

### Custom Logging

```python
import logging
import json
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("semantic_kernel")

class SkLogger:
    @staticmethod
    def log_invocation(function_name: str, inputs: dict, result: any, metadata: dict = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "function": function_name,
            "inputs": inputs,
            "result_length": len(str(result)),
            "metadata": metadata or {}
        }
        logger.info(f"SK Invocation: {json.dumps(log_entry)}")

    @staticmethod
    def log_error(function_name: str, error: Exception, context: dict = None):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "function": function_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        logger.error(f"SK Error: {json.dumps(log_entry)}")

# Usage
try:
    result = await kernel.invoke(function, input="query")
    SkLogger.log_invocation(
        function.name,
        {"input": "query"},
        result,
        {"tokens": 150, "latency_ms": 1250}
    )
except Exception as e:
    SkLogger.log_error(function.name, e, {"input": "query"})
    raise
```

---

## 23. Best Practices & Patterns

### Async Best Practices

```python
# ✅ GOOD: Use asyncio.gather for parallel operations
async def process_multiple_queries(kernel: Kernel, function, queries: List[str]):
    tasks = [kernel.invoke(function, input=q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

# ❌ BAD: Sequential processing
async def process_multiple_queries_slow(kernel: Kernel, function, queries: List[str]):
    results = []
    for q in queries:
        result = await kernel.invoke(function, input=q)  # Slow!
        results.append(result)
    return results
```

### Resource Management

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def kernel_context():
    """Context manager for kernel lifecycle"""
    kernel = Kernel()

    # Setup
    kernel.add_service(OpenAIChatCompletion(...))
    # ... add plugins, memory, etc.

    try:
        yield kernel
    finally:
        # Cleanup
        await kernel.dispose()  # If available

# Usage
async with kernel_context() as kernel:
    result = await kernel.invoke(function, input="query")
```

### Configuration Management

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SkConfig:
    openai_api_key: str
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_search_endpoint: Optional[str] = None
    max_retries: int = 3
    timeout: float = 30.0
    enable_telemetry: bool = True

    @classmethod
    def from_env(cls) -> "SkConfig":
        return cls(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            max_retries=int(os.getenv("SK_MAX_RETRIES", "3")),
            timeout=float(os.getenv("SK_TIMEOUT", "30.0")),
            enable_telemetry=os.getenv("SK_TELEMETRY", "true").lower() == "true"
        )

# Usage
config = SkConfig.from_env()

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(
    model_id="gpt-4",
    api_key=config.openai_api_key
))
```

### Factory Pattern

```python
class SkFactory:
    @staticmethod
    def create_kernel(config: SkConfig) -> Kernel:
        kernel = Kernel()

        # Add chat service
        if config.azure_openai_endpoint:
            kernel.add_service(AzureChatCompletion(
                deployment_name=config.azure_openai_deployment,
                endpoint=config.azure_openai_endpoint,
                credential=DefaultAzureCredential()
            ))
        else:
            kernel.add_service(OpenAIChatCompletion(
                model_id="gpt-4",
                api_key=config.openai_api_key
            ))

        # Add standard plugins
        kernel.add_plugin(MathPlugin(), "Math")
        kernel.add_plugin(TimePlugin(), "Time")

        return kernel

    @staticmethod
    def create_agent(
        kernel: Kernel,
        name: str,
        instructions: str,
        tools: Optional[List] = None
    ) -> ChatCompletionAgent:
        return ChatCompletionAgent(
            kernel=kernel,
            name=name,
            instructions=instructions
        )

# Usage
config = SkConfig.from_env()
kernel = SkFactory.create_kernel(config)
agent = SkFactory.create_agent(kernel, "assistant", "You are helpful")
```

---

## Conclusion

This comprehensive guide covers Semantic Kernel Python from fundamentals to advanced 2025 features including:

- ✅ Complete installation and setup
- ✅ Kernel, services, functions, and plugins
- ✅ Agents and multi-agent systems
- ✅ Memory and vector stores (including v1.34 overhaul)
- ✅ Planners and orchestration
- ✅ **Model Context Protocol (MCP)** - Client and Server
- ✅ **Google A2A Protocol** - Agent interoperability
- ✅ **Microsoft Agent Framework** - Enterprise integration
- ✅ Structured output and validation
- ✅ Error handling and resilience
- ✅ Observability and telemetry
- ✅ Production best practices

For production deployment, recipes, and more examples, see:
- [Production Guide](semantic_kernel_production_python.md)
- [Recipes](semantic_kernel_recipes_python.md)
- [Advanced Multi-Agent Guide](semantic_kernel_advanced_multi_agent_python.md)

**[Back to Python README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**

Latest: 0.14.8
# LlamaIndex Comprehensive Technical Guide

## Complete Framework for Building LLM-Powered Agents with RAG, Data Connectors, and Agentic Reasoning

---

## Table of Contents

1. [Core Fundamentals](#core-fundamentals)
2. [Simple Agents](#simple-agents)
3. [Multi-Agent Systems (llama-agents)](#multi-agent-systems-llama-agents)
4. [Tools Integration](#tools-integration)
5. [Structured Output](#structured-output)
6. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
7. [Agentic Patterns](#agentic-patterns)
8. [Memory Systems](#memory-systems)
9. [Data Connectors & Loaders](#data-connectors--loaders)
10. [Indexing & Retrieval](#indexing--retrieval)
11. [Query Engines](#query-engines)
12. [Context Engineering](#context-engineering)
13. [RAG Patterns](#rag-patterns)
14. [Advanced Topics](#advanced-topics)

---

# CORE FUNDAMENTALS

## 1. Installation and Setup

### Initial Installation

To begin working with LlamaIndex, install the core packages:

```bash
pip install llama-index
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
```

For a complete installation with all integrations:

```bash
pip install llama-index[all]
```

### Installation by Use Case

**For RAG Applications:**
```bash
pip install llama-index
pip install llama-index-embeddings-openai
pip install llama-index-vector-stores-chroma
pip install python-dotenv
```

**For Agent Development:**
```bash
pip install llama-index
pip install llama-index-agent-workflow
pip install llama-index-llms-openai
```

**For Multi-Agent Systems:**
```bash
pip install llama-agents
pip install llama-index-agent-workflow
pip install llama-index-message-queue-redis
```

### Verifying Installation

```python
import llama_index
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

print(f"LlamaIndex version: {llama_index.__version__}")
print("Installation successful!")
```

## 2. Architecture Overview

### High-Level Component Architecture

LlamaIndex is built on a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│              (Agents, Query Engines, APIs)               │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│               Orchestration & Workflow Layer             │
│         (Agent Workflow, Multi-Agent Orchestration)      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│            RAG Pipeline & Query Processing               │
│   (Query Engines, Retrievers, Rerankers, Postprocessors)│
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              Indexing & Vector Store Layer               │
│    (VectorStoreIndex, TreeIndex, KeywordTableIndex)      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                 Data Layer                              │
│  (Documents, Nodes, Loaders, LlamaHub Connectors)        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│         External Integrations                           │
│  (LLMs, Embeddings, Vector Stores, Memory Systems)       │
└─────────────────────────────────────────────────────────┘
```

### Core Components Explained

**Data Layer Components:**
- **Documents**: Raw data sources (PDFs, text, APIs, databases)
- **Nodes**: Smallest indexable units (text chunks with metadata)
- **Loaders**: Data connectors that transform raw data into documents

**Index & Storage:**
- **Vector Stores**: Store embeddings for semantic search
- **Indexes**: Organize nodes for efficient retrieval
- **Metadata Storage**: Store node metadata and relationships

**Retrieval & Processing:**
- **Retrievers**: Fetch relevant nodes from indexes
- **Query Engines**: Interface between user queries and retrieval
- **Postprocessors**: Filter, rerank, and compress retrieval results

**Agent & Reasoning:**
- **Agents**: LLM-powered entities that use tools
- **Tools**: Functions agents can call
- **Memory**: Context management for multi-turn interactions

## 3. Core Concepts Deep Dive

### 3.1 Documents

Documents represent the raw data that will be indexed and queried:

```python
from llama_index.core import Document

# Creating documents manually
doc1 = Document(
    text="LlamaIndex is a framework for building LLM applications.",
    metadata={"source": "manual", "importance": "high"}
)

doc2 = Document(
    text="RAG combines retrieval and generation for better outputs.",
    metadata={"source": "manual", "topic": "rag"}
)

# Creating from text file
document = Document.from_file("path/to/file.txt")

# Creating from URLs (with web loader)
from llama_index.core.readers import SimpleWebPageReader
reader = SimpleWebPageReader()
documents = reader.load_data(urls=["https://example.com"])
```

**Document Metadata and Relationships:**

```python
# Documents with hierarchical relationships
doc_parent = Document(
    text="Main topic",
    metadata={
        "doc_id": "1",
        "chapter": "1",
        "source": "textbook"
    }
)

doc_child = Document(
    text="Subtopic under main topic",
    metadata={
        "doc_id": "1.1",
        "parent_doc_id": "1",
        "chapter": "1",
        "source": "textbook"
    }
)
```

### 3.2 Nodes

Nodes are chunks of documents with metadata, representing the atomic unit of indexing:

```python
from llama_index.core.schema import TextNode, RelatedNodeInfo

# Creating a node manually
node = TextNode(
    text="This is a chunk of text that will be indexed.",
    metadata={"page": 1, "section": "Introduction"},
)

# Understanding node structure
print(f"Node ID: {node.node_id}")
print(f"Text: {node.text}")
print(f"Metadata: {node.metadata}")
print(f"Embedding: {node.embedding}")

# Creating nodes with relationships
node1 = TextNode(text="First paragraph", id_="node_1")
node2 = TextNode(
    text="Second paragraph",
    id_="node_2",
    relationships={
        "next": RelatedNodeInfo(node_id="node_3")
    }
)

# Nodes are automatically created from documents
# during the indexing process
```

**Node Structure Details:**

```python
# A complete node structure
node = TextNode(
    text="Content goes here",
    id_="unique-id",
    relationships={
        # Links to other nodes
        "source": RelatedNodeInfo(node_id="doc_id"),
        "next": RelatedNodeInfo(node_id="next_node_id"),
        "previous": RelatedNodeInfo(node_id="prev_node_id"),
    },
    metadata={
        "page_number": 1,
        "section": "Introduction",
        "author": "John Doe",
        "timestamp": "2024-01-01"
    },
    excluded_from_embed=False,  # Whether to exclude from embeddings
    excluded_from_llm=False,     # Whether to exclude from LLM context
)
```

### 3.3 Indexes

Indexes are data structures that organize nodes for efficient retrieval:

```python
from llama_index.core import VectorStoreIndex, Document

# Creating an index from documents
documents = [
    Document(text="Text 1"),
    Document(text="Text 2"),
]

# VectorStoreIndex - most common for semantic search
index = VectorStoreIndex.from_documents(documents)

# The index performs several operations:
# 1. Splits documents into nodes
# 2. Creates embeddings for each node
# 3. Stores embeddings in a vector store
# 4. Creates a retrieval interface
```

**Index Types:**

```python
from llama_index.core import (
    VectorStoreIndex,
    ListIndex,
    TreeIndex,
    KeywordTableIndex,
    SummaryIndex,
)

documents = [Document(text="Sample text")]

# VectorStoreIndex - semantic similarity search
vector_index = VectorStoreIndex.from_documents(documents)

# ListIndex - simple list-based retrieval
list_index = ListIndex.from_documents(documents)

# TreeIndex - hierarchical tree structure
tree_index = TreeIndex.from_documents(documents)

# KeywordTableIndex - keyword-based lookup
keyword_index = KeywordTableIndex.from_documents(documents)

# SummaryIndex - uses summaries for retrieval
summary_index = SummaryIndex.from_documents(documents)
```

### 3.4 Retrievers

Retrievers fetch relevant nodes from indexes:

```python
from llama_index.core import VectorStoreIndex, Document

# Create an index
index = VectorStoreIndex.from_documents([Document(text="Sample")])

# Get the retriever
retriever = index.as_retriever(similarity_top_k=5)

# Retrieve nodes
nodes = retriever.retrieve("What is the main topic?")

# Each retrieved node contains:
# - node: The actual node object
# - score: Relevance score

for node in nodes:
    print(f"Score: {node.score}")
    print(f"Text: {node.text}")
    print(f"Metadata: {node.metadata}")
```

**Advanced Retriever Configurations:**

```python
# Hybrid retrieval combining multiple strategies
from llama_index.core.retrievers import (
    BM25Retriever,
    VectorIndexRetriever,
    SimpleKeywordTableRetriever,
)

# BM25Retriever - keyword-based retrieval
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    nodes=index.docstore.docs.values(),
)

# VectorIndexRetriever - semantic search
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

# Combine retrievers for hybrid search
from llama_index.core.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
    
    def _retrieve(self, query_bundle):
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        # Combine and deduplicate
        combined = {}
        for node in vector_nodes + bm25_nodes:
            if node.node_id not in combined:
                combined[node.node_id] = node
        
        return list(combined.values())
```

### 3.5 Query Engines

Query engines provide the interface for querying indexed data:

```python
from llama_index.core import VectorStoreIndex, Document

# Create an index
documents = [
    Document(text="LlamaIndex is a data framework for LLM applications."),
    Document(text="RAG improves LLM outputs with retrieval."),
]
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Query the engine
response = query_engine.query("What is LlamaIndex?")
print(f"Response: {response}")
print(f"Response type: {type(response)}")  # Response object with metadata

# Access response components
print(f"Answer: {response.response}")
print(f"Source nodes: {response.source_nodes}")
for node in response.source_nodes:
    print(f"  - {node.text} (score: {node.score})")
```

**Advanced Query Engine Configurations:**

```python
# Configure LLM and retrieval parameters
query_engine = index.as_query_engine(
    similarity_top_k=5,              # Number of top results to retrieve
    response_mode="compact",         # How to format response
    streaming=True,                  # Enable streaming
)

# streaming=True allows for streaming responses
response = query_engine.query("Your query here")
for chunk in response:
    print(chunk, end="", flush=True)
```

## 4. Agent Classes and Types

### 4.1 ReActAgent

The ReActAgent implements the ReAct (Reasoning + Acting) pattern:

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Define tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

tools = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=add),
]

# Create ReActAgent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=10,
)

# Use the agent
response = agent.run("What is 5 times 3 plus 2?")
print(response)
```

**ReActAgent Detailed Workflow:**

```python
# The ReActAgent follows this loop:
# 1. THINK: LLM thinks about what to do
# 2. ACT: LLM decides which tool to use
# 3. OBSERVE: Tool is executed and result is observed
# 4. REPEAT: Continue until response is complete

# Verbose output shows this process
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
)

# With verbose=True, you'll see output like:
# Thought: I need to calculate 5 * 3 + 2
# Action: multiply
# Params: [5, 3]
# Observation: 15
# Thought: Now I need to add 15 + 2
# Action: add
# Params: [15, 2]
# Observation: 17
# Response: The answer is 17
```

### 4.2 OpenAIAgent

Leverages OpenAI's function-calling capabilities:

```python
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Define tools
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"Weather in {location}: Sunny, 25°C"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': Found 10 results..."

tools = [
    FunctionTool.from_defaults(fn=get_weather),
    FunctionTool.from_defaults(fn=search_web),
]

# Create OpenAI agent (function-calling agent)
llm = OpenAI(model="gpt-4")
agent = FunctionCallingAgent.from_tools(tools=tools, llm=llm)

# Query the agent
response = agent.run("What's the weather in New York and trending topics?")
print(response)
```

**Function-Calling Agent Features:**

```python
# Function-calling agents are more efficient than ReAct
# because they make parallel function calls

from llama_index.core.agent import FunctionCallingAgent

agent = FunctionCallingAgent.from_tools(
    tools=tools,
    llm=llm,
    max_iterations=10,
    allow_parallel_tool_calls=True,  # Can call multiple tools at once
)

# This agent can execute multiple tools in parallel
response = agent.run("Get weather for NYC and LA, and search for AI news")
# The agent might call get_weather("NYC"), get_weather("LA"), 
# and search_web("AI news") all at once
```

### 4.3 Custom Agent Implementations

Creating completely custom agents:

```python
from llama_index.core.agent import AgentRunner, Task
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.openai import OpenAI
from typing import List, Any

class CustomAgent(AgentRunner):
    """Custom agent with special reasoning capabilities."""
    
    def __init__(self, tools: List[BaseTool], llm: OpenAI):
        self.tools = tools
        self.llm = llm
        self.tool_map = {tool.metadata.name: tool for tool in tools}
    
    def run(self, query: str) -> str:
        """Run the agent with custom logic."""
        # Custom thought process
        system_prompt = f"""You are a reasoning agent. Available tools:
        {', '.join(self.tool_map.keys())}
        
        Always think step-by-step and use tools effectively."""
        
        # Formulate the prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Get LLM response
        response = self.llm.complete(
            prompt=messages,
            structured_output=None
        )
        
        return str(response)
```

## 5. LLM and Embedding Configuration

### 5.1 Configuring LLMs

```python
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.cohere import Cohere
from llama_index.llms.google_genai import GoogleGenAI

# OpenAI LLM
openai_llm = OpenAI(
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=512,
)

# Anthropic Claude
claude_llm = Anthropic(
    model="claude-3-opus-20240229",
    api_key="your-api-key",
    temperature=0.5,
)

# Google Gemini
gemini_llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key="your-api-key",
)

# Cohere
cohere_llm = Cohere(
    model="command",
    api_key="your-api-key",
)

# Test LLM
response = openai_llm.complete("Hello, how are you?")
print(response)
```

### 5.2 Configuring Embeddings

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.google import GooglePaLMEmbedding

# OpenAI Embeddings
openai_embed = OpenAIEmbedding(
    model="text-embedding-3-large",
    api_key="your-api-key",
)

# HuggingFace Embeddings (local)
hf_embed = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
)

# Cohere Embeddings
cohere_embed = CohereEmbedding(
    model_name="embed-english-v3.0",
    api_key="your-api-key",
)

# Test embeddings
embedding = openai_embed.get_text_embedding("Test text")
print(f"Embedding dimension: {len(embedding)}")

# Get batch embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = openai_embed.get_text_embedding_batch(texts)
```

### 5.3 Global Settings Configuration

```python
from llama_index.core import Settings

# Configure global settings
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.chunk_size = 1024
Settings.chunk_overlap = 20

# Now all new indexes will use these settings
from llama_index.core import VectorStoreIndex, Document

documents = [Document(text="Sample text")]
index = VectorStoreIndex.from_documents(documents)
# Uses the global Settings automatically
```

### 5.4 Local LLM Configuration

```python
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Using Ollama for local LLMs
local_llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0.7,
)

local_embed = OllamaEmbedding(
    model_name="mistral",
    base_url="http://localhost:11434",
)

# Use locally
response = local_llm.complete("What is machine learning?")
print(response)
```

---

# SIMPLE AGENTS

## 6. Creating Basic ReAct Agents

### 6.1 Minimal ReAct Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import os

# Setup
os.environ["OPENAI_API_KEY"] = "your-key"

# Define simple tools
def calculator(expression: str) -> float:
    """Evaluate mathematical expressions."""
    return eval(expression)

def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create tools
tools = [
    FunctionTool.from_defaults(fn=calculator),
    FunctionTool.from_defaults(fn=get_current_time),
]

# Create agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# Use agent
response = agent.run("What's the current time plus 5 hours?")
print(f"Final Response: {response}")
```

### 6.2 Structured Response Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

# Define output structure
class CalculationResult(BaseModel):
    operation: str
    result: float
    explanation: str

# Tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

tools = [
    FunctionTool.from_defaults(fn=multiply),
    FunctionTool.from_defaults(fn=divide),
]

# Create agent with structured output
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# Structured query
response = agent.run("Calculate 15 multiplied by 3, then divide by 9")
print(f"Response: {response}")
```

## 7. Tool Integration with Agents

### 7.1 Multiple Tool Types

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, Tool, ToolMetadata
from llama_index.llms.openai import OpenAI
from typing import Any

# Function tools
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [results...]"

def database_query(sql: str) -> str:
    """Query a database."""
    return f"Database results for: {sql}"

# Custom tool with metadata
custom_tool = Tool(
    fn=lambda x: f"Custom processing: {x}",
    metadata=ToolMetadata(
        name="custom_processor",
        description="Process custom data",
                icon_url="https://example.com/icon.png",
                arg_schema=None,
    ),
)

# Create tools list
tools = [
    FunctionTool.from_defaults(fn=web_search),
    FunctionTool.from_defaults(fn=database_query),
    custom_tool,
]

# Create agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# Agent can now use all three tools
response = agent.run("Search for 'AI trends' and also query the database for users")
```

### 7.2 Tool Chaining

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Tools that depend on each other
def extract_entities(text: str) -> list[str]:
    """Extract named entities from text."""
    # Simplified entity extraction
    return ["Entity1", "Entity2", "Entity3"]

def lookup_entity(entity: str) -> dict:
    """Look up information about an entity."""
    return {
        "name": entity,
        "type": "unknown",
        "confidence": 0.95,
    }

def summarize_entities(entities: list[dict]) -> str:
    """Summarize multiple entities."""
    return f"Summary of {len(entities)} entities"

tools = [
    FunctionTool.from_defaults(fn=extract_entities),
    FunctionTool.from_defaults(fn=lookup_entity),
    FunctionTool.from_defaults(fn=summarize_entities),
]

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# The agent will automatically chain tools
response = agent.run("Extract entities from text and summarize them")
print(response)
```

## 8. Query Engine Tools

### 8.1 Single Query Engine Tool

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI

# Create a knowledge base
documents = [
    Document(text="Machine Learning is a subset of AI..."),
    Document(text="Deep Learning uses neural networks..."),
    Document(text="NLP focuses on language understanding..."),
]

# Build index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Create tool from query engine
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="ai_knowledge_base",
    description="Search for information about AI, ML, and NLP",
)

# Create agent
tools = [query_engine_tool]
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# Query using agent
response = agent.run("What is the difference between ML and NLP?")
print(response)
```

### 8.2 Multiple Query Engine Tools

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI

# Create multiple knowledge bases
ml_docs = [Document(text="ML is about learning from data...")]
nlp_docs = [Document(text="NLP processes natural language...")]
cv_docs = [Document(text="Computer Vision analyzes images...")]

ml_index = VectorStoreIndex.from_documents(ml_docs)
nlp_index = VectorStoreIndex.from_documents(nlp_docs)
cv_index = VectorStoreIndex.from_documents(cv_docs)

# Create tools with metadata
ml_tool = QueryEngineTool(
    query_engine=ml_index.as_query_engine(),
    metadata=ToolMetadata(
        name="ml_database",
        description="Machine Learning knowledge base",
    ),
)

nlp_tool = QueryEngineTool(
    query_engine=nlp_index.as_query_engine(),
    metadata=ToolMetadata(
        name="nlp_database",
        description="Natural Language Processing knowledge base",
    ),
)

cv_tool = QueryEngineTool(
    query_engine=cv_index.as_query_engine(),
    metadata=ToolMetadata(
        name="cv_database",
        description="Computer Vision knowledge base",
    ),
)

# Create agent
tools = [ml_tool, nlp_tool, cv_tool]
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# Agent can intelligently choose which tool to use
response = agent.run("Compare ML and CV approaches for image classification")
print(response)
```

## 9. Single-Task Agents

### 9.1 Specialized Single-Task Agent

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Single-task agent: Weather information
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In real implementation, call actual weather API
    weather_data = {
        "New York": "Cloudy, 15°C",
        "Los Angeles": "Sunny, 22°C",
        "London": "Rainy, 12°C",
    }
    return weather_data.get(city, "City not found")

def convert_celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

tools = [
    FunctionTool.from_defaults(fn=get_weather),
    FunctionTool.from_defaults(fn=convert_celsius_to_fahrenheit),
]

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    system_prompt="You are a weather assistant. Help users find weather information and convert temperatures.",
)

# Single-task queries
response = agent.run("What's the weather in New York?")
print(f"Q1: {response}\n")

response = agent.run("Convert 20 Celsius to Fahrenheit")
print(f"Q2: {response}\n")

response = agent.run("What's the weather in LA in Fahrenheit?")
print(f"Q3: {response}")
```

## 10. Agent Configuration

### 10.1 Advanced Agent Settings

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

# Tools
def calculate(expr: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expr)

tools = [FunctionTool.from_defaults(fn=calculate)]

# Configure memory
memory = ChatMemoryBuffer.from_defaults(
    token_limit=4096,  # Maximum tokens in memory
)

# Create configured agent
llm = OpenAI(model="gpt-4", temperature=0.7)
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=10,           # Maximum reasoning steps
    memory=memory,                # Maintain conversation history
    system_prompt="You are a helpful mathematical assistant.",
)

# Use agent across multiple turns
print("Turn 1:")
response1 = agent.run("What is 5 + 3?")
print(f"Response: {response1}\n")

print("Turn 2:")
response2 = agent.run("Multiply the previous result by 2")
print(f"Response: {response2}\n")
# Agent remembers "previous result" was 8
```

### 10.2 Custom System Prompts

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

def search_api(query: str) -> str:
    """Search external API."""
    return f"Results for: {query}"

tools = [FunctionTool.from_defaults(fn=search_api)]

# Expert system prompt
expert_system_prompt = """You are an expert AI research assistant with deep knowledge of:
- Machine Learning and Deep Learning
- Natural Language Processing
- Computer Vision
- AI Ethics and Regulation

You have access to a search tool to find the latest research.

Guidelines:
1. Always cite sources when providing information
2. Distinguish between established facts and recent developments
3. Acknowledge limitations and uncertainties
4. Provide balanced perspectives on controversial topics
5. Suggest further reading when appropriate"""

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    system_prompt=expert_system_prompt,
    verbose=True,
)

response = agent.run("What are the latest developments in multimodal AI?")
print(response)
```

## 11. Streaming Responses

### 11.1 Agent Response Streaming

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import asyncio

def fetch_data(source: str) -> str:
    """Fetch data from a source."""
    return f"Data from {source}: [large dataset...]"

tools = [FunctionTool.from_defaults(fn=fetch_data)]

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm)

async def stream_agent_response():
    """Stream agent response for real-time feedback."""
    handler = agent.run(
        "Fetch data from multiple sources and summarize",
        stream=True,
    )
    
    async for chunk in handler:
        print(chunk, end="", flush=True)
    
    print("\n[Streaming complete]")

# Run streaming
asyncio.run(stream_agent_response())
```

### 11.2 Event-Based Streaming

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import asyncio

def search(query: str) -> str:
    """Search for information."""
    return f"Search results for '{query}'"

tools = [FunctionTool.from_defaults(fn=search)]

# Setup debugging/callbacks
debug_handler = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([debug_handler])

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    callback_manager=callback_manager,
)

async def stream_with_events():
    """Stream with event tracking."""
    handler = agent.run(
        "Search for AI trends and provide insights",
        stream=True,
    )
    
    # Print events
    async for event in handler.stream_events():
        print(f"Event: {event}")

asyncio.run(stream_with_events())
```

---

# MULTI-AGENT SYSTEMS (llama-agents)

## 12. llama-agents Package Overview

### 12.1 Introduction to Multi-Agent Architecture

```python
# Multi-agent systems allow for:
# - Task decomposition
# - Parallel processing
# - Specialized agent roles
# - Improved scalability

from llama_agents import Agent, AgentService, ControlPlaneServer
import asyncio

# Basic architecture:
# Control Plane -> Orchestrates agents
# Message Queue -> Communication between agents
# Agent Services -> Individual agents handling tasks

# Install package
# pip install llama-agents
```

### 12.2 Single Agent Service Setup

```python
from llama_agents import Agent, AgentService
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import asyncio

# Define tools
def process_text(text: str) -> str:
    """Process and analyze text."""
    return f"Processed: {text}"

tools = [FunctionTool.from_defaults(fn=process_text)]

# Create agent
llm = OpenAI(model="gpt-4")
agent = Agent(
    name="TextProcessor",
    tools=tools,
    llm=llm,
)

# Wrap as service
service = AgentService(
    agent=agent,
    service_name="text_processor",
    port=8001,
)

async def main():
    # Start service
    await service.launch_server()

# asyncio.run(main())
```

## 13. Multi-Agent Orchestration

### 13.1 Control Plane and Message Queue Pattern

```python
from llama_agents import (
    ControlPlaneServer,
    MessageQueueServer,
    Agent,
    AgentService,
)
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import asyncio

# Define multiple specialized agents
def research_tool(topic: str) -> str:
    """Research a topic."""
    return f"Research findings on {topic}"

def analyze_tool(data: str) -> str:
    """Analyze data."""
    return f"Analysis of: {data}"

def write_tool(content: str) -> str:
    """Write content."""
    return f"Written: {content}"

# Create agents
llm = OpenAI(model="gpt-4")

researcher = Agent(
    name="Researcher",
    tools=[FunctionTool.from_defaults(fn=research_tool)],
    llm=llm,
)

analyzer = Agent(
    name="Analyzer",
    tools=[FunctionTool.from_defaults(fn=analyze_tool)],
    llm=llm,
)

writer = Agent(
    name="Writer",
    tools=[FunctionTool.from_defaults(fn=write_tool)],
    llm=llm,
)

# Create agent services
research_service = AgentService(
    agent=researcher,
    service_name="research",
    port=8001,
)
analyzer_service = AgentService(
    agent=analyzer,
    service_name="analyzer",
    port=8002,
)
writer_service = AgentService(
    agent=writer,
    service_name="writer",
    port=8003,
)

async def run_multi_agent_system():
    """Run coordinated multi-agent system."""
    # Start services
    await research_service.launch_server()
    await analyzer_service.launch_server()
    await writer_service.launch_server()
    
    # Control plane coordinates between agents
    control_plane = ControlPlaneServer()
    # ... orchestration logic

# asyncio.run(run_multi_agent_system())
```

## 14. Distributed Agent Systems

### 14.1 Microservice Architecture

```python
# Multi-agent microservice architecture:
#
# ┌─────────────────────────────────────────┐
# │       Control Plane / Orchestrator       │
# └──────────────┬──────────────────────────┘
#                │
#     ┌──────────┼──────────┐
#     │          │          │
# ┌───▼──┐   ┌──▼───┐   ┌─▼────┐
# │Agent1│   │Agent2│   │Agent3│
# └──────┘   └──────┘   └──────┘
#
# Communication via Message Queue (Redis, RabbitMQ, etc.)

from llama_agents import Agent, AgentService
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Agent 1: Data processor
def process_data(data: str) -> str:
    """Process raw data."""
    return f"Processed: {data[:100]}"

# Agent 2: Quality checker
def validate_data(data: str) -> bool:
    """Validate processed data."""
    return len(data) > 0

# Agent 3: Storage manager
def store_data(data: str) -> str:
    """Store validated data."""
    return f"Stored at: /data/{hash(data)}"

llm = OpenAI(model="gpt-4")

agents = [
    Agent(
        name="DataProcessor",
        tools=[FunctionTool.from_defaults(fn=process_data)],
        llm=llm,
    ),
    Agent(
        name="QualityChecker",
        tools=[FunctionTool.from_defaults(fn=validate_data)],
        llm=llm,
    ),
    Agent(
        name="StorageManager",
        tools=[FunctionTool.from_defaults(fn=store_data)],
        llm=llm,
    ),
]

# Each agent runs as independent microservice
# with message queue for communication
```

---

# TOOLS INTEGRATION

## 15. QueryEngineTool and Tool Classes

### 15.1 QueryEngineTool in Detail

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, Document

# Create a knowledge base
documents = [
    Document(text="Python is a high-level programming language..."),
    Document(text="TypeScript adds static typing to JavaScript..."),
    Document(text="Rust provides memory safety without garbage collection..."),
]

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Create tool from query engine
tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="programming_languages",
    description="Database of programming language information",
)

# Tool properties
print(f"Tool name: {tool.metadata.name}")
print(f"Tool description: {tool.metadata.description}")

# Use tool in agent
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools([tool], llm=llm, verbose=True)

response = agent.run("Compare Python and Rust for system programming")
print(response)
```

### 15.2 Tool Metadata and Schemas

```python
from llama_index.core.tools import Tool, ToolMetadata
from pydantic import BaseModel, Field
from typing import Any

# Define input schema
class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return"
    )

# Create tool with metadata
def search_database(query: str, max_results: int = 10) -> str:
    """Search the database for information."""
    return f"Found {max_results} results for '{query}'"

tool = Tool(
    fn=search_database,
    metadata=ToolMetadata(
        name="database_search",
        description="Search the database for information",
        fn_schema=SearchInput,
    ),
)

# Tool with rich metadata
rich_tool = Tool(
    fn=search_database,
    metadata=ToolMetadata(
        name="database_search",
        description="Search the database for information",
        fn_schema=SearchInput,
        return_direct=False,      # Whether to directly return result
        as_tool=True,              # Format for tool use
    ),
)
```

## 16. FunctionTool Creation

### 16.1 Creating Function Tools

```python
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
import inspect

# Simple function tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

# Create tool
multiply_tool = FunctionTool.from_defaults(
    fn=multiply,
    name="multiply",
    description="Multiply two numbers",
)

# Function tool with Pydantic schema
def search_web(query: str, num_results: int = 5) -> list[str]:
    """Search the web and return results."""
    return [f"Result {i+1} for {query}" for i in range(num_results)]

search_tool = FunctionTool.from_defaults(
    fn=search_web,
    # Schema is automatically inferred from type hints
)

# Verify tool metadata
print(f"Tool: {multiply_tool.metadata.name}")
print(f"Description: {multiply_tool.metadata.description}")
print(f"Schema: {multiply_tool.metadata.fn_schema}")
```

### 16.2 Advanced Function Tools

```python
from llama_index.core.tools import FunctionTool
from typing import Optional, List
import asyncio

# Async function tool
async def fetch_remote_data(endpoint: str, timeout: int = 30) -> dict:
    """Fetch data from a remote endpoint."""
    # Simulated async operation
    await asyncio.sleep(1)
    return {"endpoint": endpoint, "status": "success"}

async_tool = FunctionTool.from_defaults(
    fn=fetch_remote_data,
)

# Tool with optional parameters
def summarize_text(
    text: str,
    length: str = "medium",  # short, medium, long
    style: Optional[str] = None,  # bullet, paragraph, abstract
) -> str:
    """Summarize text with options."""
    return f"Summary ({length}, {style}): {text[:100]}"

summary_tool = FunctionTool.from_defaults(
    fn=summarize_text,
)

# Tool returning complex types
def analyze_data(
    data: List[float],
    metrics: Optional[List[str]] = None
) -> dict:
    """Analyze numerical data."""
    import statistics
    
    if metrics is None:
        metrics = ["mean", "median", "stdev"]
    
    results = {}
    if "mean" in metrics:
        results["mean"] = statistics.mean(data)
    if "median" in metrics:
        results["median"] = statistics.median(data)
    if "stdev" in metrics:
        results["stdev"] = statistics.stdev(data)
    
    return results

analysis_tool = FunctionTool.from_defaults(
    fn=analyze_data,
)
```

## 17. LlamaHub Tools

### 17.1 Exploring LlamaHub

```python
# LlamaHub provides 100+ pre-built tools and loaders

# Browse available tools at: https://llamahub.ai

# Common tool categories:
# - Web Search Tools
# - Database Tools
# - API Tools
# - Document Analysis Tools
# - Weather Tools
# - Finance Tools

# Example: Using a pre-built tool

from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Initialize tool spec
google_search = GoogleSearchToolSpec(api_key="your-api-key")

# Get tools from spec
tools = google_search.to_tool_list()

# Use in agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

response = agent.run("Search for recent AI breakthroughs")
print(response)
```

### 17.2 Common LlamaHub Tools

```python
# Weather Tool
from llama_index.tools.weather import WeatherToolSpec

weather = WeatherToolSpec(api_key="your-api-key")
weather_tools = weather.to_tool_list()
# Tools: get_current_weather, get_forecast, etc.

# Wikipedia Tool
from llama_index.tools.wikipedia import WikipediaToolSpec

wikipedia = WikipediaToolSpec()
wiki_tools = wikipedia.to_tool_list()
# Tools: search_wikipedia, get_page, etc.

# Arxiv Tool (Academic Papers)
from llama_index.tools.arxiv import ArxivToolSpec

arxiv = ArxivToolSpec()
arxiv_tools = arxiv.to_tool_list()
# Tools: search_papers, get_paper_details, etc.

# Slack Tool
from llama_index.tools.slack import SlackToolSpec

slack = SlackToolSpec(
    slack_bot_token="xoxb-xxx",
    slack_user_token="xoxp-xxx",
)
slack_tools = slack.to_tool_list()
# Tools: post_message, search_messages, etc.
```

## 18. Custom Tool Development

### 18.1 Creating Custom Tools

```python
from llama_index.core.tools import Tool, ToolMetadata
from pydantic import BaseModel, Field
from typing import Optional

# Define input schema
class SentimentAnalysisInput(BaseModel):
    text: str = Field(description="Text to analyze")
    language: str = Field(
        default="en",
        description="Language code (en, es, fr, etc.)"
    )

# Implement tool function
def analyze_sentiment(text: str, language: str = "en") -> dict:
    """Analyze sentiment of given text."""
    # Simplified sentiment analysis
    positive_words = ["good", "great", "excellent", "love", "amazing"]
    negative_words = ["bad", "terrible", "hate", "awful", "horrible"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": 0.85,
        "language": language,
    }

# Create tool
sentiment_tool = Tool(
    fn=analyze_sentiment,
    metadata=ToolMetadata(
        name="sentiment_analysis",
        description="Analyze the sentiment of text",
        fn_schema=SentimentAnalysisInput,
    ),
)

# Use in agent
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools([sentiment_tool], llm=llm, verbose=True)

response = agent.run("Analyze the sentiment of 'I absolutely love this product!'")
print(response)
```

### 18.2 Advanced Custom Tool with State

```python
from llama_index.core.tools import Tool, ToolMetadata
from pydantic import BaseModel, Field
from typing import Dict, List

# Stateful tool for calculations
class CalculatorState:
    """Maintains calculation history."""
    
    def __init__(self):
        self.history: List[tuple] = []
        self.variables: Dict[str, float] = {}
    
    def calculate(self, expression: str) -> float:
        """Evaluate expression with variable support."""
        result = eval(expression, {"__builtins__": {}}, self.variables)
        self.history.append((expression, result))
        return result
    
    def store_variable(self, name: str, value: float) -> str:
        """Store a variable for later use."""
        self.variables[name] = value
        return f"Stored {name} = {value}"
    
    def get_history(self) -> List[tuple]:
        """Get calculation history."""
        return self.history

# Global instance
calc_state = CalculatorState()

# Tool functions
def calculate_with_memory(expression: str) -> float:
    """Calculate expression with memory of previous results."""
    return calc_state.calculate(expression)

def store_calc_var(name: str, value: float) -> str:
    """Store a calculation result as a variable."""
    return calc_state.store_variable(name, value)

def show_calc_history() -> str:
    """Show calculation history."""
    history = calc_state.get_history()
    return "\n".join([f"{expr} = {result}" for expr, result in history])

# Create tools
tools = [
    Tool.from_defaults(fn=calculate_with_memory),
    Tool.from_defaults(fn=store_calc_var),
    Tool.from_defaults(fn=show_calc_history),
]

# Use stateful tools
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

response = agent.run("""
    Calculate 15 * 3 and store as 'result1'
    Calculate 45 / 2 and store as 'result2'
    Add them together
    Show the history
""")
print(response)
```

## 19. Tool Schemas

### 19.1 Pydantic Schemas for Tools

```python
from llama_index.core.tools import FunctionTool, Tool, ToolMetadata
from pydantic import BaseModel, Field
from typing import Optional, List

# Basic schema
class EmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body")

def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
    }

email_tool = Tool(
    fn=send_email,
    metadata=ToolMetadata(
        name="send_email",
        description="Send an email to a recipient",
        fn_schema=EmailInput,
    ),
)

# Complex nested schema
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    email: str = Field(description="Person's email address")
    address: Address = Field(description="Person's address")
    phone: Optional[str] = Field(default=None, description="Optional phone number")

class AddContactInput(BaseModel):
    person: Person = Field(description="Person details")
    tags: List[str] = Field(default=[], description="Tags for organizing contacts")

def add_contact(person: Person, tags: List[str] = []) -> str:
    """Add a new contact."""
    return f"Added contact: {person.name}"

contact_tool = Tool(
    fn=add_contact,
    metadata=ToolMetadata(
        name="add_contact",
        description="Add a new contact with detailed information",
        fn_schema=AddContactInput,
    ),
)
```

## 20. Error Handling in Tools

### 20.1 Robust Tool Error Handling

```python
from llama_index.core.tools import FunctionTool
from typing import Union

def safe_divide(a: float, b: float) -> Union[float, str]:
    """Safely divide two numbers."""
    try:
        if b == 0:
            return "Error: Cannot divide by zero"
        result = a / b
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_api_data(endpoint: str, timeout: int = 5) -> Union[dict, str]:
    """Fetch data from API with error handling."""
    try:
        import requests
        response = requests.get(f"https://api.example.com{endpoint}", timeout=timeout)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.Timeout:
        return f"Error: Request to {endpoint} timed out after {timeout}s"
    except requests.exceptions.ConnectionError:
        return "Error: Failed to connect to API"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def process_with_fallback(data: str, method: str = "primary") -> str:
    """Process data with fallback mechanism."""
    try:
        if method == "primary":
            # Primary processing logic
            if len(data) < 5:
                raise ValueError("Data too short")
            return f"Processed (primary): {data}"
        else:
            # Fallback logic
            return f"Processed (fallback): {data[:5]}"
    except ValueError as e:
        # Log error and try fallback
        print(f"Primary processing failed: {e}")
        return process_with_fallback(data, method="fallback")

# Create tools with error handling
tools = [
    FunctionTool.from_defaults(fn=safe_divide),
    FunctionTool.from_defaults(fn=fetch_api_data),
    FunctionTool.from_defaults(fn=process_with_fallback),
]
```

### 20.2 Tool Validation and Verification

```python
from llama_index.core.tools import FunctionTool, Tool, ToolMetadata
from pydantic import BaseModel, Field, validator
from typing import Optional

# Schema with validation
class DatabaseQueryInput(BaseModel):
    query: str = Field(description="SQL query to execute")
    max_rows: int = Field(default=100, description="Maximum rows to return")
    
    @validator("query")
    def query_must_be_safe(cls, v):
        """Validate query doesn't contain dangerous operations."""
        dangerous = ["DROP", "DELETE", "TRUNCATE", "GRANT", "REVOKE"]
        if any(op in v.upper() for op in dangerous):
            raise ValueError("Query contains potentially dangerous operations")
        return v
    
    @validator("max_rows")
    def max_rows_reasonable(cls, v):
        """Validate max_rows is reasonable."""
        if v < 1 or v > 10000:
            raise ValueError("max_rows must be between 1 and 10000")
        return v

def query_database(query: str, max_rows: int = 100) -> list:
    """Execute database query safely."""
    # Actual query execution
    return [f"Row {i}" for i in range(min(max_rows, 5))]

database_tool = Tool(
    fn=query_database,
    metadata=ToolMetadata(
        name="query_database",
        description="Query the database safely",
        fn_schema=DatabaseQueryInput,
    ),
)
```

---

# STRUCTURED OUTPUT

## 21. Pydantic Program for Structured Extraction

### 21.1 Using Pydantic for Structured Extraction

```python
from llama_index.core import Document
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Define output structure
class PersonInfo(BaseModel):
    """Extracted information about a person."""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="Occupation")
    skills: List[str] = Field(description="List of skills")

# Example text to extract from
text = """
John Smith is a 35-year-old software engineer with expertise in Python, 
JavaScript, and cloud architecture. He specializes in building scalable 
distributed systems.
"""

# Create parser
parser = PydanticOutputParser(output_class=PersonInfo)

# Use with LLM
llm = OpenAI(model="gpt-4")
prompt = f"""Extract structured information from this text:
{text}

Output format: {parser.format_instructions()}"""

response = llm.complete(prompt)
parsed = parser.parse(str(response))

print(f"Name: {parsed.name}")
print(f"Age: {parsed.age}")
print(f"Occupation: {parsed.occupation}")
print(f"Skills: {parsed.skills}")
```

### 21.2 Complex Nested Structures

```python
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Define nested structures
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    address: Address

class Project(BaseModel):
    name: str
    description: str
    status: str  # active, completed, planned
    start_date: str
    technologies: List[str]

class EmployeeProfile(BaseModel):
    """Complete employee profile."""
    name: str
    employee_id: str
    contact: ContactInfo
    position: str
    salary_range: str
    current_projects: List[Project]
    performance_rating: float = Field(ge=0, le=5)
    start_date: str

# Example extraction
text = """
John Doe (ID: EMP001) is a Senior Software Engineer at our company.
He can be reached at john.doe@company.com or (555) 123-4567.
John lives at 123 Main St, New York, NY 10001.

He is currently working on:
1. Project Aurora (Active) - Building ML pipeline in Python and TensorFlow since Jan 2024
2. Dashboard Redesign (Planned) - React and TypeScript project starting Feb 2024

John joined on March 15, 2020 and has a performance rating of 4.5/5.
His salary range is $120,000-$150,000.
"""

parser = PydanticOutputParser(output_class=EmployeeProfile)
llm = OpenAI(model="gpt-4")

prompt = f"""Extract employee information from this text and structure it properly:
{text}

{parser.format_instructions()}"""

response = llm.complete(prompt)
parsed = parser.parse(str(response))

print(f"Employee: {parsed.name}")
print(f"Position: {parsed.position}")
print(f"Email: {parsed.contact.email}")
print(f"Address: {parsed.contact.address.street}, {parsed.contact.address.city}")
print(f"Current Projects: {len(parsed.current_projects)}")
for project in parsed.current_projects:
    print(f"  - {project.name} ({project.status})")
```

## 22. Output Parsers

### 22.1 Different Output Parser Types

```python
from llama_index.core.output_parsers import (
    PydanticOutputParser,
    StructuredOutput Parser,
    ChainableOutputParser,
)
from pydantic import BaseModel, Field
from llama_index.llms.openai import OpenAI
from typing import List

# Pydantic parser (most common)
class SentimentResult(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_phrases: List[str]

pydantic_parser = PydanticOutputParser(output_class=SentimentResult)

# Chain multiple parsers
def preprocess_output(text: str) -> str:
    """Clean up output before parsing."""
    return text.strip().replace("```json", "").replace("```", "")

class ChainedParser(ChainableOutputParser):
    """Custom parser combining preprocessing and parsing."""
    
    def __init__(self, output_class):
        self.output_class = output_class
        self.pydantic_parser = PydanticOutputParser(output_class=output_class)
    
    def parse(self, text: str):
        # Preprocess
        clean_text = preprocess_output(text)
        # Parse
        return self.pydantic_parser.parse(clean_text)
    
    def format_instructions(self) -> str:
        return self.pydantic_parser.format_instructions()

# Custom output parser
class CSVOutputParser:
    """Parse CSV format output."""
    
    def parse(self, text: str) -> List[dict]:
        import csv
        from io import StringIO
        
        reader = csv.DictReader(StringIO(text))
        return list(reader)
    
    def format_instructions(self) -> str:
        return "Output in CSV format with headers"
```

## 23. Schema Enforcement

### 23.1 Enforcing Output Schemas

```python
from llama_index.core import Document
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field, validator
from typing import Optional

# Strict schema with validation
class BlogPost(BaseModel):
    """Blog post schema with validation."""
    title: str = Field(min_length=5, max_length=200, description="Post title")
    content: str = Field(min_length=100, description="Post content")
    category: str = Field(description="Category (tech, business, lifestyle)")
    tags: list[str] = Field(min_items=1, max_items=5, description="1-5 tags")
    publish_date: str = Field(description="Date in YYYY-MM-DD format")
    author_email: Optional[str] = Field(default=None, description="Author email")
    
    @validator("category")
    def category_valid(cls, v):
        valid = ["tech", "business", "lifestyle"]
        if v not in valid:
            raise ValueError(f"Category must be one of {valid}")
        return v
    
    @validator("tags")
    def tags_format(cls, v):
        # Ensure tags are lowercase and no spaces
        return [tag.lower().strip() for tag in v]

# Use with LLM
llm = OpenAI(model="gpt-4")
parser = PydanticOutputParser(output_class=BlogPost)

text = """
Title: Introduction to Machine Learning
Content: Machine Learning is a subset of AI that enables systems to learn from data...
Category: tech
Tags: AI, ML, Python, Data Science
Publish Date: 2024-01-15
Author Email: author@example.com
"""

prompt = f"""Extract and structure this blog post information:
{text}

{parser.format_instructions()}"""

response = llm.complete(prompt)
blog_post = parser.parse(str(response))

print(f"Title: {blog_post.title}")
print(f"Category: {blog_post.category}")
print(f"Tags: {blog_post.tags}")
```

## 24. JSON Mode

### 24.1 Using JSON Mode

```python

from llama_index.llms.openai import OpenAI
from json import loads
from pydantic import BaseModel

# Enable JSON mode on OpenAI
llm = OpenAI(model="gpt-4-turbo-preview")

# Request JSON response
prompt = """Convert this information to JSON:
Name: Alice
Age: 30
Skills: Python, JavaScript, AWS
Projects: 5

Return ONLY valid JSON."""

response = llm.complete(prompt)

# Parse JSON response
try:
    json_data = loads(str(response))
    print(f"Parsed JSON: {json_data}")
except Exception as e:
    print(f"Failed to parse: {e}")

# Better: Use response_format parameter
class PersonData(BaseModel):
    name: str
    age: int
    skills: list[str]
    num_projects: int

# Structured output with JSON
prompt_with_structure = f"""Convert this information to JSON:
Name: Alice
Age: 30
Skills: Python, JavaScript, AWS
Projects: 5

Expected format:
{{
  "name": "string",
  "age": "integer",
  "skills": ["string"],
  "num_projects": "integer"
}}"""

response = llm.complete(prompt_with_structure)
parsed = loads(str(response))
print(f"Structured JSON: {parsed}")

```

---

# AGENTIC PATTERNS

## 25. ReAct Loop Implementations

### 25.1 Understanding the ReAct Pattern

```python
"""
ReAct (Reasoning + Acting) Loop:

1. OBSERVE: Current state, available tools, previous context
2. THINK: LLM reasons about what to do next
3. ACT: LLM chooses a tool and provides parameters
4. OBSERVE: Tool result is added to context
5. REPEAT: Continue until agent decides to stop

Example trace:
Observation: User asked "What's the weather in NYC tomorrow?"
Thought: I need to call the weather tool to get forecast data
Action: get_weather_forecast
Action Input: {"location": "New York", "days": 1}
Observation: Weather data received: "Tomorrow: Cloudy, 15°C"
Thought: I have the information needed, can now respond
Final Answer: Tomorrow in NYC it will be cloudy with a temperature of 15°C
"""

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

def get_weather(location: str) -> str:
    """Get current weather."""
    return f"{location}: Sunny, 22°C"

def get_weather_forecast(location: str, days: int = 1) -> str:
    """Get weather forecast."""
    return f"{location} forecast for {days} days: Variable conditions"

tools = [
    FunctionTool.from_defaults(fn=get_weather),
    FunctionTool.from_defaults(fn=get_weather_forecast),
]

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# The agent will automatically follow ReAct loop
response = agent.run("What's the weather forecast for Paris for the next 3 days?")
```

### 25.2 Custom ReAct Implementation

```python
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.openai import OpenAI
from typing import Any

class ManualReActAgent:
    """Manually implement ReAct loop for understanding."""
    
    def __init__(self, tools: list[BaseTool], llm: OpenAI):
        self.tools = {tool.metadata.name: tool for tool in tools}
        self.llm = llm
        self.max_iterations = 10
        self.thoughts = []
    
    def run(self, query: str) -> str:
        """Run ReAct loop."""
        context = f"User query: {query}"
        
        for iteration in range(self.max_iterations):
            # Step 1: THINK - LLM thinks about what to do
            thought_prompt = f"""Given this context:
{context}

Available tools: {', '.join(self.tools.keys())}

Decide what to do next. Output either:
- A tool call: "Tool: <tool_name>, Input: <json_input>"
- A final answer: "Answer: <your_response>"
"""
            
            response = self.llm.complete(thought_prompt)
            self.thoughts.append(str(response))
            
            # Parse response
            if "Answer:" in str(response):
                return str(response).split("Answer:")[1].strip()
            
            # Step 2: ACT - Extract and execute tool
            if "Tool:" in str(response):
                # Parse tool call
                parts = str(response).split("Tool:")[1].split("Input:")
                tool_name = parts[0].strip()
                
                if tool_name in self.tools:
                    # Execute tool
                    tool = self.tools[tool_name]
                    result = tool(params={})  # Simplified
                    
                    # Step 3: OBSERVE - Add result to context
                    context += f"\n[Tool {tool_name} result: {result}]"
        
        return "Max iterations reached"

# Usage
def search_api(query: str) -> str:
    return f"API results for: {query}"

tools = [FunctionTool.from_defaults(fn=search_api)]
llm = OpenAI(model="gpt-4")
agent = ManualReActAgent(tools=tools, llm=llm)

response = agent.run("Search for recent AI breakthroughs")
print(response)
```

## 26. Agentic RAG (Reasoning + Retrieval)

### 26.1 Combining Reasoning and Retrieval

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Create multiple knowledge bases
docs_llm = [
    Document(text="LLMs learn from vast amounts of text data..."),
    Document(text="Transformer architecture uses self-attention..."),
]
docs_rag = [
    Document(text="RAG combines retrieval with generation..."),
    Document(text="Vector stores enable semantic search..."),
]

# Build indexes
llm_index = VectorStoreIndex.from_documents(docs_llm)
rag_index = VectorStoreIndex.from_documents(docs_rag)

# Create tools
llm_tool = QueryEngineTool.from_defaults(
    query_engine=llm_index.as_query_engine(),
    name="llm_knowledge",
    description="Information about Large Language Models",
)
rag_tool = QueryEngineTool.from_defaults(
    query_engine=rag_index.as_query_engine(),
    name="rag_knowledge",
    description="Information about Retrieval-Augmented Generation",
)

# Create reasoning agent
tools = [llm_tool, rag_tool]
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    system_prompt="""You are an AI expert. Answer questions by:
1. Retrieving relevant information from knowledge bases
2. Reasoning about how different concepts relate
3. Providing comprehensive explanations""",
    verbose=True,
)

# Query with reasoning
response = agent.run(
    "Compare and contrast LLMs and RAG systems. "
    "How does RAG address LLM limitations?"
)
print(response)
```

### 26.2 Dynamic RAG with Agent Reasoning

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

# Knowledge bases
documents = [
    Document(text="Python is a versatile programming language..."),
    Document(text="JavaScript runs in browsers and Node.js..."),
    Document(text="TypeScript adds static typing to JavaScript..."),
]

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Retrieval tool
retrieval_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="language_search",
    description="Search programming language information",
)

# Analysis tool
def analyze_comparison(items: str) -> str:
    """Analyze and compare multiple items."""
    return f"Analysis of: {items}"

analysis_tool = FunctionTool.from_defaults(fn=analyze_comparison)

# Combine retrieval and reasoning
tools = [retrieval_tool, analysis_tool]
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)

# Dynamic RAG query
response = agent.run(
    "Search for information on Python and JavaScript, "
    "then analyze their differences"
)
print(response)
```

## 27. Sub-Question Query Engine

### 27.1 Breaking Down Complex Queries

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

# Multiple knowledge bases
finance_docs = [Document(text="Tesla's 2023 revenue was $81.5 billion...")]
tech_docs = [Document(text="Tesla uses AI for autonomous driving...")]
market_docs = [Document(text="EV market growing at 15% annually...")]

# Create indexes and tools
finance_index = VectorStoreIndex.from_documents(finance_docs)
tech_index = VectorStoreIndex.from_documents(tech_docs)
market_index = VectorStoreIndex.from_documents(market_docs)

tools = [
    QueryEngineTool.from_defaults(
        query_engine=finance_index.as_query_engine(),
        name="financial_data",
        description="Financial information",
    ),
    QueryEngineTool.from_defaults(
        query_engine=tech_index.as_query_engine(),
        name="technology_data",
        description="Technology and innovation information",
    ),
    QueryEngineTool.from_defaults(
        query_engine=market_index.as_query_engine(),
        name="market_data",
        description="Market and industry information",
    ),
]

# Create sub-question engine
llm = OpenAI(model="gpt-4")
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    llm=llm,
    verbose=True,
)

# Complex query decomposed into sub-questions
response = sub_question_engine.query(
    "What are Tesla's financial performance, technological innovations, "
    "and market position in the EV industry?"
)
print(response)
```

### 27.2 Multi-Hop Reasoning

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

# Create interconnected knowledge bases
doc1 = Document(text="Alice works at TechCorp")
doc2 = Document(text="TechCorp is in San Francisco")
doc3 = Document(text="San Francisco has a tech industry")
doc4 = Document(text="Tech industry requires programming skills")

index = VectorStoreIndex.from_documents([doc1, doc2, doc3, doc4])

tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    name="knowledge_base",
    description="General knowledge base",
)

# Multi-hop query
sub_q_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool],
    llm=OpenAI(model="gpt-4"),
    verbose=True,
)

# This requires multiple hops:
# 1. Who is Alice? (works at TechCorp)
# 2. Where is TechCorp? (San Francisco)
# 3. What's special about San Francisco? (tech industry)
# 4. What does tech industry need? (programming)
response = sub_q_engine.query(
    "What skills does Alice likely need given her location and employer?"
)
print(response)
```

## 28. Router Agents

### 28.1 Routing to Specialized Agents

```python
from llama_index.core.agent import ReActAgent, AgentRunner
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from typing import Any

class RouterAgent:
    """Router that directs queries to specialized agents."""
    
    def __init__(self):
        self.router_llm = OpenAI(model="gpt-4")
        self.agents = {}
        self._setup_specialized_agents()
    
    def _setup_specialized_agents(self):
        """Create specialized agents."""
        # Sales agent
        def sales_query(query: str) -> str:
            return f"Sales response to: {query}"
        
        sales_tool = FunctionTool.from_defaults(fn=sales_query)
        self.agents["sales"] = ReActAgent.from_tools(
            [sales_tool],
            llm=self.router_llm,
        )
        
        # Support agent
        def support_query(query: str) -> str:
            return f"Support response to: {query}"
        
        support_tool = FunctionTool.from_defaults(fn=support_query)
        self.agents["support"] = ReActAgent.from_tools(
            [support_tool],
            llm=self.router_llm,
        )
        
        # Technical agent
        def tech_query(query: str) -> str:
            return f"Technical response to: {query}"
        
        tech_tool = FunctionTool.from_defaults(fn=tech_query)
        self.agents["technical"] = ReActAgent.from_tools(
            [tech_tool],
            llm=self.router_llm,
        )
    
    def route_and_respond(self, query: str) -> str:
        """Route query to appropriate agent."""
        # Determine routing
        routing_prompt = f"""Classify this query as one of: sales, support, technical
Query: {query}

Respond with ONLY the category name."""
        
        routing = self.router_llm.complete(routing_prompt)
        category = str(routing).lower().strip()
        
        # Route to appropriate agent
        if category in self.agents:
            agent = self.agents[category]
            return agent.run(query)
        else:
            return "Unable to route query"

# Usage
router = RouterAgent()
response = router.route_and_respond("How do I upgrade my plan?")
print(response)
```

## 29. Self-Reflection

### 29.1 Implementing Self-Reflection in Agents

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

class ReflectiveAgent:
    """Agent that reflects on its responses."""
    
    def __init__(self, tools: list, llm: OpenAI):
        self.agent = ReActAgent.from_tools(tools, llm=llm)
        self.llm = llm
    
    def run_with_reflection(self, query: str) -> dict:
        """Run agent and reflect on response."""
        # Initial response
        response = self.agent.run(query)
        
        # Reflection step
        reflection_prompt = f"""Reflect on this response to the query.
Query: {query}
Response: {response}

Analyze:
1. Is the response accurate and complete?
2. Are there gaps or limitations?
3. What could be improved?
4. Is there a better way to answer?"""
        
        reflection = self.llm.complete(reflection_prompt)
        
        # Decide if re-run needed
        improvement_prompt = f"""Based on this reflection:
{reflection}

Should we refine the response? Answer yes or no."""
        
        should_improve = "yes" in self.llm.complete(improvement_prompt).lower()
        
        if should_improve:
            # Re-run with improvement prompt
            improved_query = f"""{query}

Note: Consider the following for a better response:
{reflection}"""
            improved_response = self.agent.run(improved_query)
            return {
                "original": response,
                "reflection": str(reflection),
                "improved": improved_response,
            }
        
        return {
            "response": response,
            "reflection": str(reflection),
        }

# Usage
def calculate(expr: str) -> float:
    return eval(expr)

tools = [FunctionTool.from_defaults(fn=calculate)]
reflective_agent = ReflectiveAgent(tools, OpenAI(model="gpt-4"))

result = reflective_agent.run_with_reflection(
    "Calculate the area of a circle with radius 5"
)
print(result)
```

---

# MEMORY SYSTEMS

## 30. Chat Memory for Agents

### 30.1 Basic Chat Memory

```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Setup
def calculator(expr: str) -> float:
    return eval(expr)

tools = [FunctionTool.from_defaults(fn=calculator)]

# Create memory buffer
memory = ChatMemoryBuffer.from_defaults(
    token_limit=2000,  # Max tokens to keep in memory
)

# Create agent with memory
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    memory=memory,
    verbose=True,
)

# Multi-turn conversation
print("Turn 1:")
response1 = agent.run("Calculate 5 * 3 and store as result1")
print(f"Response: {response1}\n")

print("Turn 2:")
response2 = agent.run("Now multiply result1 by 2")
print(f"Response: {response2}\n")
# Agent remembers previous calculation

print("Turn 3:")
response3 = agent.run("Add 10 to the previous result")
print(f"Response: {response3}\n")
# Agent can reference multi-turn history
```

### 30.2 Custom Memory Strategies

```python
from llama_index.core.memory import BaseMemory
from llama_index.core.schema import ChatMessage, MessageRole
from typing import List, Optional

class SummaryMemory(BaseMemory):
    """Memory that summarizes old messages to save tokens."""
    
    def __init__(self, max_messages: int = 20, summary_threshold: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.summary = ""
    
    def get(self) -> List[ChatMessage]:
        """Get all messages."""
        if self.summary:
            summary_msg = ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {self.summary}"
            )
            return [summary_msg] + self.messages
        return self.messages
    
    def put(self, message: ChatMessage) -> None:
        """Add a message."""
        self.messages.append(message)
        
        # Summarize if too many messages
        if len(self.messages) > self.summary_threshold:
            self._summarize()
    
    def _summarize(self) -> None:
        """Summarize old messages."""
        # In real implementation, use LLM to summarize
        old_messages = self.messages[:5]
        self.summary += f" {len(old_messages)} previous exchanges..."
        self.messages = self.messages[5:]  # Keep recent messages
    
    def reset(self) -> None:
        """Clear memory."""
        self.messages = []
        self.summary = ""

# Use custom memory
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

memory = SummaryMemory(max_messages=20, summary_threshold=10)
llm = OpenAI(model="gpt-4")

agent = ReActAgent.from_tools(
    tools=[],
    llm=llm,
    memory=memory,
)
```

---

# DATA CONNECTORS & LOADERS

## 31. LlamaHub Data Loaders

### 31.1 Common Data Loaders

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    SimpleFileReader,
    PDFReader,
    DocxReader,
    CSVReader,
)

# Simple directory reader (auto-detects file types)
documents = SimpleDirectoryReader(
    input_dir="./data",
    file_metadata={"source": "local_files"},
).load_data()

# PDF reader
from llama_index.readers.file.pdf import PDFReader

pdf_reader = PDFReader()
pdf_docs = pdf_reader.load_data("document.pdf")

# DOCX reader
from llama_index.readers.file.docx import DocxReader

docx_reader = DocxReader()
word_docs = docx_reader.load_data("document.docx")

# CSV reader
from llama_index.readers.file.csv import CSVReader

csv_reader = CSVReader()
csv_docs = csv_reader.load_data("data.csv")

# Web reader
from llama_index.readers.web.simple_web_page import SimpleWebPageReader

web_reader = SimpleWebPageReader()
web_docs = web_reader.load_data(
    urls=["https://example.com", "https://another-site.com"]
)

# Markdown reader
from llama_index.readers.file.markdown import MarkdownReader

md_reader = MarkdownReader()
md_docs = md_reader.load_data("document.md")
```

### 31.2 Database Connectors

```python
# SQL Database loader
from llama_index.readers.database import DatabaseReader

db_reader = DatabaseReader(
    engine="postgresql",
    uri="postgresql://user:password@localhost/dbname",
)

# Load from SQL query
documents = db_reader.load_data(
    query="SELECT * FROM articles WHERE published = true"
)

# MongoDB loader
from llama_index.readers.mongodb import MongoDBReader

mongo_reader = MongoDBReader(
    uri="mongodb://localhost:27017",
    db_name="mydb",
)

documents = mongo_reader.load_data(
    collection="articles",
    query={"status": "published"},
    projection=["title", "content"],
)

# Firebase loader
from llama_index.readers.firebase import FirebaseReader

firebase_reader = FirebaseReader(
    credentials_path="./firebase-key.json"
)

documents = firebase_reader.load_data(
    collection="articles",
)
```

### 31.3 API and SaaS Connectors

```python
# GitHub loader
from llama_index.readers.github import GithubRepositoryReader

github_reader = GithubRepositoryReader(
    github_token="your-github-token",
)

documents = github_reader.load_data(
    owner="run-llama",
    repo="llama_index",
    branch="main",
    file_extensions=[".md", ".py"],
)

# Google Docs loader
from llama_index.readers.google_docs import GoogleDocsReader

google_docs_reader = GoogleDocsReader()
documents = google_docs_reader.load_data(
    document_ids=["1_document_id", "2_document_id"],
)

# Confluence loader
from llama_index.readers.confluence import ConfluenceReader

confluence_reader = ConfluenceReader(
    base_url="https://your-domain.atlassian.net/wiki",
    username="your-email@example.com",
    api_token="your-confluence-token",
)

documents = confluence_reader.load_data(
    page_ids=["12345"],
)

# Slack loader
from llama_index.readers.slack import SlackReader

slack_reader = SlackReader(
    slack_bot_token="xoxb-xxx",
    earliest_date="2024-01-01",
)

documents = slack_reader.load_data(
    channel_ids=["C123456"],
)

# Notion loader
from llama_index.readers.notion import NotionPageReader

notion_reader = NotionPageReader(
    integration_token="secret_xxx"
)

documents = notion_reader.load_data(
    page_ids=["abc123"],
)
```

### 31.4 Other Common Loaders

```python
# Wikipedia loader
from llama_index.readers.wikipedia import WikipediaReader

wiki_reader = WikipediaReader()
documents = wiki_reader.load_data(
    pages=["Python (programming language)", "Machine Learning"],
)

# Arxiv (academic papers) loader
from llama_index.readers.arxiv import ArxivReader

arxiv_reader = ArxivReader()
documents = arxiv_reader.load_data(
    search_query="machine learning",
    max_results=5,
)

# YouTube transcript loader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader

yt_reader = YoutubeTranscriptReader()
documents = yt_reader.load_data(
    video_ids=["dQw4w9WgXcQ"],
)

# Twitter/X loader
from llama_index.readers.twitter import TwitterReader

twitter_reader = TwitterReader(
    bearer_token="your-bearer-token",
)
documents = twitter_reader.load_data(
    query="AI machine learning",
    max_tweets=100,
)

# RSS Feed loader
from llama_index.readers.rss_feed import RSSFeedReader

rss_reader = RSSFeedReader()
documents = rss_reader.load_data(
    feed_urls=[
        "https://example.com/feed.rss",
        "https://another.com/feed.rss",
    ]
)
```

## 32. Custom Loader Creation

### 32.1 Building Custom Loaders

```python
from llama_index.core.readers import BaseReader
from llama_index.core.schema import Document
from typing import List, Any

class JSONLinesReader(BaseReader):
    """Custom loader for JSONL files."""
    
    def load_data(self, file_path: str, **load_kwargs: Any) -> List[Document]:
        """Load data from JSONL file."""
        documents = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    import json
                    data = json.loads(line)
                    
                    # Extract text and metadata
                    text = data.get('text', str(data))
                    metadata = {
                        "source": file_path,
                        "line_number": line_num + 1,
                        **{k: v for k, v in data.items() 
                           if k != 'text'}
                    }
                    
                    doc = Document(text=text, metadata=metadata)
                    documents.append(doc)
                
                except json.JSONDecodeError:
                    print(f"Failed to parse line {line_num + 1}")
        
        return documents

# Use custom loader
loader = JSONLinesReader()
documents = loader.load_data("data.jsonl")
```

### 32.2 Advanced Custom Loaders

```python
from llama_index.core.readers import BaseReader
from llama_index.core.schema import Document
from typing import List, Any, Optional
import xml.etree.ElementTree as ET

class XMLReader(BaseReader):
    """Custom loader for XML files with element extraction."""
    
    def __init__(self, text_element: str = "text"):
        self.text_element = text_element
    
    def load_data(
        self,
        file_path: str,
        extract_attributes: bool = True,
        **load_kwargs: Any
    ) -> List[Document]:
        """Load data from XML file."""
        documents = []
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for item in root.findall(".//item"):
            # Extract text
            text_elem = item.find(self.text_element)
            text = text_elem.text if text_elem is not None else ""
            
            # Extract metadata
            metadata = {"source": file_path}
            
            if extract_attributes:
                metadata.update(item.attrib)
            
            # Extract other child elements as metadata
            for child in item:
                if child.tag != self.text_element:
                    metadata[child.tag] = child.text
            
            if text.strip():
                doc = Document(text=text, metadata=metadata)
                documents.append(doc)
        
        return documents

# Use XML reader
xml_loader = XMLReader(text_element="content")
documents = xml_loader.load_data("data.xml", extract_attributes=True)
```

---

# INDEXING & RETRIEVAL

## 33. VectorStoreIndex

### 33.1 Creating and Using VectorStoreIndex

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Create documents
documents = [
    Document(text="Machine Learning is a subset of AI..."),
    Document(text="Deep Learning uses neural networks..."),
    Document(text="Reinforcement Learning learns from rewards..."),
]

# Create index with custom embedding
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    show_progress=True,
)

# Create query engine
query_engine = index.as_query_engine()

# Query the index
response = query_engine.query("What is machine learning?")
print(f"Response: {response}")

# Access source nodes
for node in response.source_nodes:
    print(f"Source: {node.text[:100]} (Score: {node.score})")
```

### 33.2 VectorStoreIndex with Persistence

```python
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Setup Chroma for persistence
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create index
documents = [Document(text="Sample document")]
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# Later: Load existing index
loaded_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
)

# Continue using
query_engine = loaded_index.as_query_engine()
response = query_engine.query("Your query here")
print(response)
```

## 34. Other Index Types

### 34.1 ListIndex and TreeIndex

```python
from llama_index.core import (
    ListIndex,
    TreeIndex,
    KeywordTableIndex,
    SummaryIndex,
    Document,
)

documents = [
    Document(text="Text 1"),
    Document(text="Text 2"),
    Document(text="Text 3"),
]

# ListIndex - simple list-based retrieval
list_index = ListIndex.from_documents(documents)
list_query_engine = list_index.as_query_engine()

# TreeIndex - builds a tree for hierarchical retrieval
tree_index = TreeIndex.from_documents(documents)
tree_query_engine = tree_index.as_query_engine()

# KeywordTableIndex - keyword-based lookup
keyword_index = KeywordTableIndex.from_documents(documents)
keyword_query_engine = keyword_index.as_query_engine()

# SummaryIndex - uses summaries for retrieval
summary_index = SummaryIndex.from_documents(documents)
summary_query_engine = summary_index.as_query_engine()

# Compare retrieval strategies
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engines=[
        list_query_engine,
        tree_query_engine,
        keyword_query_engine,
        summary_query_engine,
    ],
)

response = router_engine.query("Your query")
```

## 35. Retrievers

### 35.1 Advanced Retriever Configuration

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    BM25Retriever,
    SimpleKeywordTableRetriever,
)

# Create indexes
documents = [Document(text="Sample document")]
vector_index = VectorStoreIndex.from_documents(documents)

# Different retriever types
vector_retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=5,
    filters=None,  # Optional filters
)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=vector_index.docstore,
    nodes=vector_index.docstore.docs.values(),
)

keyword_retriever = SimpleKeywordTableRetriever(
    index=vector_index,
)

# Hybrid retriever combining multiple strategies
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore

class HybridRetriever(BaseRetriever):
    """Combine multiple retrieval strategies."""
    
    def __init__(self, vector_retriever, bm25_retriever, weights=None):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights or {"vector": 0.7, "bm25": 0.3}
    
    def _retrieve(self, query_bundle):
        # Get results from both retrievers
        vector_results = self.vector_retriever.retrieve(query_bundle)
        bm25_results = self.bm25_retriever.retrieve(query_bundle)
        
        # Combine and rerank
        combined = {}
        for node in vector_results:
            score = (node.score or 0) * self.weights["vector"]
            combined[node.node_id] = NodeWithScore(
                node=node.node,
                score=score,
            )
        
        for node in bm25_results:
            score = (node.score or 0) * self.weights["bm25"]
            if node.node_id in combined:
                combined[node.node_id].score += score
            else:
                combined[node.node_id] = NodeWithScore(
                    node=node.node,
                    score=score,
                )
        
        # Return top results
        return sorted(
            combined.values(),
            key=lambda x: x.score or 0,
            reverse=True,
        )[:5]

# Use hybrid retriever
hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
nodes = hybrid_retriever.retrieve(query_bundle)
```

## 36. Reranking and Postprocessing

### 36.1 Node Postprocessors

```python
from llama_index.core.postprocessor import (
    NodePostprocessor,
    SimilarityPostprocessor,
    MetadataReplacementPostprocessor,
    FixedPercentageNodePostprocessor,
)
from llama_index.core import VectorStoreIndex, Document

# Create index
documents = [
    Document(text="Text 1", metadata={"relevance": 0.9}),
    Document(text="Text 2", metadata={"relevance": 0.5}),
    Document(text="Text 3", metadata={"relevance": 0.7}),
]
index = VectorStoreIndex.from_documents(documents)

# Similarity postprocessor - filter by minimum score
similarity_postprocessor = SimilarityPostprocessor(
    similarity_cutoff=0.7,  # Only keep nodes with score >= 0.7
)

# Percentage postprocessor - keep top N%
percentage_postprocessor = FixedPercentageNodePostprocessor(
    percentage=0.8,  # Keep top 80%
)

# Use in query engine
query_engine = index.as_query_engine(
    node_postprocessors=[
        similarity_postprocessor,
        percentage_postprocessor,
    ],
)

response = query_engine.query("Your query")
```

### 36.2 Custom Reranking

```python
from llama_index.core.postprocessor import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.openai import OpenAI
from typing import List

class LLMReranker(BaseNodePostprocessor):
    """Rerank nodes using LLM relevance scoring."""
    
    def __init__(self, llm: OpenAI, top_n: int = 5):
        self.llm = llm
        self.top_n = top_n
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Rerank nodes using LLM."""
        if not nodes:
            return nodes
        
        # Prepare ranking prompt
        nodes_text = "\n".join(
            [f"{i}. {node.node.text[:100]}" for i, node in enumerate(nodes)]
        )
        
        prompt = f"""Rank these nodes by relevance to the query.
Query: {query_bundle.query_str}

Nodes:
{nodes_text}

Return only the ranking as numbers separated by commas (e.g., "2,1,3")"""
        
        response = self.llm.complete(prompt)
        ranking = [int(x.strip()) - 1 for x in str(response).split(",")]
        
        # Reorder nodes
        reranked = [nodes[i] for i in ranking if i < len(nodes)]
        
        # Return top N
        return reranked[:self.top_n]

# Use LLM reranker
llm = OpenAI(model="gpt-4")
reranker = LLMReranker(llm=llm, top_n=3)

query_engine = index.as_query_engine(
    node_postprocessors=[reranker],
)
```

---

# QUERY ENGINES

## 37. Creating Query Engines

### 37.1 Basic Query Engines

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import RetrieverQueryEngine

# Create documents and index
documents = [Document(text="Sample content")]
index = VectorStoreIndex.from_documents(documents)

# Simple query engine
query_engine = index.as_query_engine()

# With custom settings
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact",
    streaming=False,
)

# Query
response = query_engine.query("Your question")
print(f"Response: {response}")
print(f"Source nodes: {response.source_nodes}")
```

### 37.2 RouterQueryEngine

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMuliSelector
from llama_index.core import VectorStoreIndex, ListIndex, Document

# Create multiple indexes
docs = [Document(text="Sample")]
vector_index = VectorStoreIndex.from_documents(docs)
list_index = ListIndex.from_documents(docs)

# Create router that selects best engine
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engines=[
        vector_index.as_query_engine(),
        list_index.as_query_engine(),
    ],
    query_engine_descriptions=[
        "Vector index for semantic search",
        "List index for all documents",
    ],
)

# Router automatically selects appropriate engine
response = router.query("Your query")
```

---

# CONTEXT ENGINEERING

## 38. Prompt Templates

### 38.1 Working with Prompts

```python
from llama_index.core import PromptTemplate, ChatPromptTemplate
from llama_index.llms.openai import OpenAI

# Simple prompt template
simple_prompt = PromptTemplate(
    "You are an expert on {topic}. Answer this question: {question}"
)

# Use template
formatted = simple_prompt.format(
    topic="machine learning",
    question="What is deep learning?"
)
print(formatted)

# Chat prompt template (multi-turn)
chat_prompt = ChatPromptTemplate(
    message_templates=[
        ("system", "You are a helpful AI assistant on {topic}"),
        ("user", "Question: {question}"),
    ]
)

llm = OpenAI(model="gpt-4")
response = llm.complete(
    chat_prompt.format(
        topic="AI",
        question="Explain neural networks"
    )
)
```

### 38.2 Custom Prompts for Agents

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

def search(query: str) -> str:
    return f"Results for: {query}"

tools = [FunctionTool.from_defaults(fn=search)]

# Custom system prompt
system_prompt = """You are an advanced research assistant with capabilities to:
1. Search for information
2. Synthesize findings
3. Provide citations

Always cite your sources and provide balanced perspectives.
Think step-by-step before using tools."""

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    system_prompt=system_prompt,
)

response = agent.run("Research the latest AI developments")
```

## 39. Few-Shot Examples

### 39.1 Implementing Few-Shot Learning

```python
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

# Few-shot prompt with examples
few_shot_prompt = PromptTemplate(
"""Classify the sentiment of reviews. Here are examples:

Review: "This product is amazing! Highly recommend."
Sentiment: Positive

Review: "Terrible quality, don't buy this."
Sentiment: Negative

Review: "It's okay, nothing special."
Sentiment: Neutral

Now classify this review:
Review: "{review}"
Sentiment:""")

llm = OpenAI(model="gpt-4")

review = "Great value for money, very satisfied with purchase."
prompt = few_shot_prompt.format(review=review)
response = llm.complete(prompt)
print(f"Sentiment: {response}")
```

---

# RAG PATTERNS

## 40. Basic RAG Pipeline

### 40.1 Complete RAG Pipeline

```python
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Step 1: Load documents
documents = SimpleDirectoryReader(
    input_dir="./data"
).load_data()

# Step 2: Create index (embedding + storage)
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
)

# Step 3: Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
)

# Step 4: Query
response = query_engine.query("What is the main topic of these documents?")

# Step 5: Display results
print(f"Response: {response}")
print(f"\nSources:")
for node in response.source_nodes:
    print(f"- {node.text[:100]}")
    print(f"  Score: {node.score}")
```

### 40.2 Advanced RAG with Multiple Indexes

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI

# Create domain-specific indexes
product_docs = [Document(text="Product information...")]
support_docs = [Document(text="Support documentation...")]
blog_docs = [Document(text="Blog articles...")]

product_index = VectorStoreIndex.from_documents(product_docs)
support_index = VectorStoreIndex.from_documents(support_docs)
blog_index = VectorStoreIndex.from_documents(blog_docs)

# Create router
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engines=[
        product_index.as_query_engine(),
        support_index.as_query_engine(),
        blog_index.as_query_engine(),
    ],
    query_engine_descriptions=[
        "Product details and specifications",
        "Support and troubleshooting",
        "Blog articles and insights",
    ],
)

# Query automatically routes to best source
response = router_engine.query("How do I return a product?")
print(response)
```

## 41. Advanced RAG Techniques

### 41.1 Query Rewriting

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CustomQueryEngine

class QueryRewritingEngine:
    """Rewrite queries for better retrieval."""
    
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.query_engine = index.as_query_engine()
    
    def rewrite_query(self, original_query: str) -> str:
        """Rewrite query using LLM."""
        prompt = f"""Rewrite this query to be more specific and better for semantic search:
Original: {original_query}

Rewritten (be specific and use relevant keywords):"""
        
        rewritten = self.llm.complete(prompt)
        return str(rewritten).strip()
    
    def query(self, query: str):
        """Query with automatic rewriting."""
        # Rewrite the query
        rewritten = self.rewrite_query(query)
        print(f"Original: {query}")
        print(f"Rewritten: {rewritten}\n")
        
        # Execute rewritten query
        response = self.query_engine.query(rewritten)
        return response

# Usage
documents = [Document(text="LLM content")]
index = VectorStoreIndex.from_documents(documents)
llm = OpenAI(model="gpt-4")

rewriting_engine = QueryRewritingEngine(index, llm)
response = rewriting_engine.query("Tell me about transformers")
print(response)
```

### 41.2 Multi-Hop Retrieval

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI

class MultiHopRetrieval:
    """Multi-hop retrieval for complex questions."""
    
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.query_engine = index.as_query_engine()
    
    def decompose_query(self, query: str) -> list[str]:
        """Break query into sub-questions."""
        prompt = f"""Break this query into 2-3 simpler sub-questions:
Query: {query}

Sub-questions (one per line):"""
        
        response = self.llm.complete(prompt)
        sub_queries = str(response).strip().split("\n")
        return [q.strip().lstrip("123456789. -•") for q in sub_queries]
    
    def retrieve_multi_hop(self, query: str):
        """Retrieve using multiple hops."""
        # Decompose query
        sub_queries = self.decompose_query(query)
        
        print(f"Original query: {query}\n")
        print("Sub-questions:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"  {i}. {sub_q}")
        
        # Retrieve for each sub-query
        all_context = []
        for sub_q in sub_queries:
            response = self.query_engine.query(sub_q)
            all_context.append(str(response))
        
        # Combine and synthesize
        synthesis_prompt = f"""Given these intermediate findings:
{chr(10).join(all_context)}

Answer the original question: {query}"""
        
        final_response = self.llm.complete(synthesis_prompt)
        return final_response

# Usage
documents = [
    Document(text="Fact 1..."),
    Document(text="Fact 2..."),
]
index = VectorStoreIndex.from_documents(documents)
llm = OpenAI(model="gpt-4")

multi_hop = MultiHopRetrieval(index, llm)
response = multi_hop.retrieve_multi_hop("Complex question requiring multiple hops")
```

---

# ADVANCED TOPICS

## 42. Custom Agent Implementations

### 42.1 Building Agents from Scratch

```python
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.llms.openai import OpenAI
from typing import List, Any
import json

class CustomLLMAgent:
    """Implement custom agent logic."""
    
    def __init__(self, tools: List[BaseTool], llm: OpenAI):
        self.tools = {tool.metadata.name: tool for tool in tools}
        self.llm = llm
        self.message_history = []
        self.max_iterations = 10
    
    def run(self, query: str) -> str:
        """Execute agent with custom logic."""
        self.message_history = [
            {"role": "user", "content": query}
        ]
        
        for iteration in range(self.max_iterations):
            # Get LLM response
            response = self._get_llm_response()
            
            # Check if done
            if "FINAL_ANSWER" in response:
                return response.split("FINAL_ANSWER:")[1].strip()
            
            # Execute tool if requested
            if "TOOL:" in response:
                tool_result = self._execute_tool(response)
                self.message_history.append({
                    "role": "assistant",
                    "content": response,
                })
                self.message_history.append({
                    "role": "system",
                    "content": f"Tool result: {tool_result}",
                })
            else:
                return response
        
        return "Max iterations reached"
    
    def _get_llm_response(self) -> str:
        """Get response from LLM."""
        system_prompt = f"""You are a helpful assistant with access to tools:
{json.dumps({k: v.metadata.description for k, v in self.tools.items()}, indent=2)}

Respond with:
- TOOL: <tool_name> with INPUT: <json_params> to use a tool
- FINAL_ANSWER: <answer> when done"""
        
        # Simplified LLM call
        response = self.llm.complete(
            "\n".join([f"{m['role']}: {m['content']}" for m in self.message_history])
        )
        return str(response)
    
    def _execute_tool(self, response: str) -> str:
        """Execute tool from response."""
        try:
            tool_name = response.split("TOOL:")[1].split("INPUT:")[0].strip()
            if tool_name in self.tools:
                return "Tool executed"
        except:
            pass
        return "Tool execution failed"

# Usage
def search(query: str) -> str:
    return f"Search results for: {query}"

tools = [FunctionTool.from_defaults(fn=search)]
agent = CustomLLMAgent(tools=tools, llm=OpenAI(model="gpt-4"))
response = agent.run("Search for AI news")
```

## 43. Observability and Callbacks

### 43.1 Callback Management

```python
from llama_index.core.callbacks import CallbackManager, BaseCallbackHandler
from llama_index.core.callbacks.schema import EventType
import time

class LoggingCallbackHandler(BaseCallbackHandler):
    """Log events during LLM operations."""
    
    def on_event_start(self, event_type: EventType, **kwargs) -> None:
        """Called when event starts."""
        print(f"[START] {event_type}: {kwargs}")
    
    def on_event_end(self, event_type: EventType, **kwargs) -> None:
        """Called when event ends."""
        print(f"[END] {event_type}")

class TimingCallbackHandler(BaseCallbackHandler):
    """Track timing of operations."""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def on_event_start(self, event_type: EventType, **kwargs) -> None:
        self.start_times[event_type] = time.time()
    
    def on_event_end(self, event_type: EventType, **kwargs) -> None:
        if event_type in self.start_times:
            duration = time.time() - self.start_times[event_type]
            self.timings[event_type] = duration
            print(f"{event_type} took {duration:.2f}s")

# Use callbacks
from llama_index.core import VectorStoreIndex, Document

callback_manager = CallbackManager(
    handlers=[
        LoggingCallbackHandler(),
        TimingCallbackHandler(),
    ]
)

index = VectorStoreIndex.from_documents(
    [Document(text="Test")],
    callback_manager=callback_manager,
)
```

## 44. Evaluation Framework

### 44.1 Basic Evaluation

```python
from llama_index.core.evaluation import EvaluationResult, AnswerRelevancyEvaluator
from llama_index.llms.openai import OpenAI

# Setup evaluator
llm = OpenAI(model="gpt-4")
evaluator = AnswerRelevancyEvaluator(llm=llm)

# Evaluate response
response = "Machine Learning is a subset of AI"
query = "What is machine learning?"

result = evaluator.evaluate(
    response=response,
    query=query,
)

print(f"Score: {result.score}")
print(f"Passing: {result.passing}")
```

---

*(Continued in the next section with Production deployment, testing, and optimization...)*

This concludes the comprehensive guide section. Each topic includes conceptual explanations, complete code examples, RAG implementation details, and real-world patterns.


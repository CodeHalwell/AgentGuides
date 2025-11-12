Latest: 2.19.0
# Haystack Comprehensive Technical Guide: From Fundamentals to Expert-Level Agentic AI

## Table of Contents

1. [Part I: Core Fundamentals](#part-i-core-fundamentals)
   - Installation and Setup
   - Haystack 2.x Architecture Overview
   - Components and Pipelines
   - Agent Concepts and Paradigms
   - Provider-Agnostic Design Philosophy
   - Configuration Patterns and Best Practices

2. [Part II: Simple Agents](#part-ii-simple-agents)
   - Creating Agents with the Agent Class
   - Tool Integration and Function Calling
   - Single-Purpose Agents
   - Conversational Agents with Memory
   - Agent Configuration and Customisation
   - Error Handling and Resilience

3. [Part III: Multi-Agent Systems](#part-iii-multi-agent-systems)
   - Multi-Agent Pipeline Design Patterns
   - Agent Coordination and Communication
   - Router and Dispatcher Components
   - Conditional Routing and Flow Control
   - Parallel Execution and Concurrency
   - Agent Collaboration Patterns

4. [Part IV: Tools Integration](#part-iv-tools-integration)
   - Tool Components Architecture
   - Custom Tool Creation
   - OpenAPI Tool Integration
   - Function Calling Mechanisms
   - Tool Validation and Verification
   - Error Recovery Strategies

5. [Part V: Structured Output and Validation](#part-v-structured-output-and-validation)
   - Output Adapters and Transformation
   - Schema Validation Framework
   - JSON Output Generation
   - Pydantic Integration
   - Custom Output Formats
   - Parsing Strategies and Error Handling

6. [Part VI: Model Context Protocol (MCP)](#part-vi-model-context-protocol-mcp)
   - MCP in Haystack Ecosystem
   - Custom MCP Component Development
   - Tool Exposure Through MCP
   - Context Management
   - Integration Patterns

7. [Part VII: Agentic Patterns](#part-vii-agentic-patterns)
   - ReAct Agent Loops and Reasoning Chains
   - Planning Components and Goal Decomposition
   - Self-Correction Mechanisms
   - Multi-Step Reasoning Frameworks
   - Reflection Patterns
   - Autonomous Workflows

8. [Part VIII: Memory Systems](#part-viii-memory-systems)
   - Conversation Memory Components
   - Document Stores for Memory Management
   - Memory Retrievers
   - Session Management
   - Persistent Memory Infrastructure
   - Custom Memory Store Implementation

9. [Part IX: Document Stores](#part-ix-document-stores)
   - Supported Store Types
   - Configuration and Initialisation
   - Indexing Strategies
   - Retrieval Methods and Query Execution
   - Hybrid Search Techniques
   - Filters, Metadata, and Advanced Queries

10. [Part X: Pipelines](#part-x-pipelines)
    - Pipeline Creation and Composition
    - Component Connections
    - Conditional Branching
    - Loops in Pipelines
    - Error Handling in Pipeline Execution
    - Pipeline Visualisation and Debugging

11. [Part XI: Retrievers and Generators](#part-xi-retrievers-and-generators)
    - Retriever Components and Types
    - Generator Components
    - RAG Pipeline Construction
    - Prompt Builders and Templates
    - Output Validation
    - Streaming Responses

12. [Part XII: Context Engineering](#part-xii-context-engineering)
    - PromptBuilder Component Deep Dive
    - Dynamic Prompt Construction
    - Template Management
    - Context Optimisation
    - Few-Shot Learning Examples
    - Prompt Versioning and Rollbacks

13. [Part XIII: Observability and Monitoring](#part-xiii-observability-and-monitoring)
    - Tracing and Logging Infrastructure
    - Component Instrumentation
    - Pipeline Monitoring
    - Performance Metrics Collection
    - Custom Tracers
    - Integration with Observability Platforms

14. [Part XIV: Advanced Topics](#part-xiv-advanced-topics)
    - Custom Components Development
    - Component Testing and Validation
    - Pipeline Optimisation Techniques
    - Provider Switching
    - Evaluation Pipelines
    - CI/CD Integration

---

# PART I: CORE FUNDAMENTALS

## Installation and Setup

### Basic Installation

Haystack is available on PyPI and can be installed using pip. The main package `haystack-ai` provides core functionality, with additional packages for specific integrations.

```bash
# Basic installation
pip install haystack-ai

# Installation with specific version
pip install haystack-ai==2.16.0

# Installation with extras (includes common integrations)
pip install haystack-ai[all]

# Development installation from source
pip install git+https://github.com/deepset-ai/haystack.git@main
```

### Virtual Environment Setup

For production deployments, always use a virtual environment:

```bash
# Create virtual environment
python -m venv haystack_env

# Activate environment
# On Windows
haystack_env\Scripts\activate
# On macOS/Linux
source haystack_env/bin/activate

# Install Haystack
pip install haystack-ai
```

### Verifying Installation

```python
import haystack
from haystack import Pipeline
from haystack.components import component

print(f"Haystack version: {haystack.__version__}")

# Test basic component creation
@component
def simple_component(text: str) -> dict:
    return {"result": text.upper()}

print("Installation verified successfully!")
```

### Docker Setup

For containerised environments:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir haystack-ai

COPY . .

CMD ["python", "app.py"]
```

### Optional Dependencies

Different use cases require different dependencies:

```bash
# For Elasticsearch support
pip install haystack-ai[elasticsearch]

# For Weaviate support
pip install haystack-ai[weaviate]

# For OpenSearch support
pip install haystack-ai[opensearch]

# For Pinecone support
pip install haystack-ai[pinecone]

# For Qdrant support
pip install haystack-ai[qdrant]

# For together with LLM integrations
pip install haystack-ai[openai]
pip install haystack-ai[anthropic]
pip install haystack-ai[huggingface]

# Install all commonly used integrations
pip install "haystack-ai[elasticsearch,weaviate,pinecone,qdrant,openai,anthropic,huggingface]"
```

## Haystack 2.x Architecture Overview

Haystack 2.x represents a complete architectural redesign focused on modularity, flexibility, and production-readiness. The framework is built on several foundational principles:

### Architectural Principles

**1. Component-Based Design**: Everything in Haystack is a component. Components are decorated Python classes that can be connected into pipelines. Each component has explicit inputs and outputs.

```python
from haystack import component

@component
class DocumentProcessor:
    """
    A simple component that processes documents.
    """
    
    @component.output_types(processed_documents=list)
    def run(self, documents: list[str]) -> dict:
        """
        Process a list of documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            Dictionary with processed documents
        """
        processed = [doc.strip().lower() for doc in documents]
        return {"processed_documents": processed}

# Create instance
processor = DocumentProcessor()

# Component information
print(f"Inputs: {processor.__haystack_input__}")
print(f"Outputs: {processor.__haystack_output__}")
```

**2. Pipeline-Centric Execution**: Pipelines are directed acyclic graphs (DAGs) of components. Data flows through connected components, transforming at each step.

{% raw %}
```python
from haystack import Pipeline, component

# Define multiple components
@component
class InputValidator:
    @component.output_types(valid_input=str)
    def run(self, query: str) -> dict:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return {"valid_input": query}

@component
class TextTransformer:
    @component.output_types(transformed=str)
    def run(self, text: str) -> dict:
        return {"transformed": text.upper()}

@component
class ResultFormatter:
    @component.output_types(formatted_result=str)
    def run(self, text: str) -> dict:
        return {"formatted_result": f"Result: {text}"}

# Create pipeline
pipeline = Pipeline()
pipeline.add_component("validator", InputValidator())
pipeline.add_component("transformer", TextTransformer())
pipeline.add_component("formatter", ResultFormatter())

# Connect components
pipeline.connect("validator.valid_input", "transformer.text")
pipeline.connect("transformer.transformed", "formatter.text")

# Run pipeline
result = pipeline.run({"validator": {"query": "hello world"}})
print(result)  # Output: {'formatter': {'formatted_result': 'Result: HELLO WORLD'}}
```
{% endraw %}

**3. Provider-Agnostic Integration**: Haystack supports multiple LLM providers, vector databases, and tools. Components can be swapped to use different providers without changing pipeline structure.

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat import AnthropicChatGenerator

# Create generator with OpenAI
openai_generator = OpenAIChatGenerator(
    model="gpt-4o",
    api_key="sk-your-key"
)

# Create generator with Anthropic
anthropic_generator = AnthropicChatGenerator(
    model="claude-3-opus",
    api_key="your-api-key"
)

# Both can be used interchangeably in pipelines
```

**4. Type-Safe Configuration**: Components use Python type hints extensively, enabling IDE autocomplete, static analysis, and runtime validation.

{% raw %}
```python
from haystack import component
from typing import Optional

@component
class ConfigurableComponent:
    def __init__(
        self,
        threshold: float = 0.5,
        max_results: int = 10,
        enable_caching: bool = True,
        optional_param: Optional[str] = None
    ):
        self.threshold = threshold
        self.max_results = max_results
        self.enable_caching = enable_caching
        self.optional_param = optional_param
    
    @component.output_types(results=dict)
    def run(self, input_data: dict) -> dict:
        return {"results": {"threshold": self.threshold}}
```
{% endraw %}

### Architectural Layers

```
┌─────────────────────────────────────────────────┐
│           Application Layer                      │
│    (Agents, RAG Applications, Workflows)         │
├─────────────────────────────────────────────────┤
│           Pipeline Layer                         │
│    (DAG Execution, Routing, Branching)           │
├─────────────────────────────────────────────────┤
│           Component Layer                        │
│    (Retrievers, Generators, Tools, etc.)         │
├─────────────────────────────────────────────────┤
│           Integration Layer                      │
│    (LLM Providers, Document Stores, APIs)        │
├─────────────────────────────────────────────────┤
│           Data Layer                             │
│    (Documents, Embeddings, Vector Indexes)       │
└─────────────────────────────────────────────────┘
```

## Components and Pipelines

### Understanding Components

A component is the fundamental building block in Haystack. It's a Python class decorated with `@component` that:

1. Takes input through a `run()` method
2. Performs processing
3. Outputs results via return dictionary
4. Is reusable across multiple pipelines
5. Has explicit input and output types

```python
from haystack import component, logging
from typing import Optional
import logging as std_logging

# Set up logging
logger = std_logging.getLogger(__name__)

@component
class AdvancedTextProcessor:
    """
    An advanced text processing component with multiple output types.
    This component demonstrates best practices for component design.
    """
    
    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 1000,
        remove_special_chars: bool = False
    ):
        """
        Initialize the text processor.
        
        Args:
            min_length: Minimum allowed text length
            max_length: Maximum allowed text length
            remove_special_chars: Whether to remove special characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_special_chars = remove_special_chars
        logger.info(f"Initialised TextProcessor with min={min_length}, max={max_length}")
    
    @component.output_types(
        processed_text=str,
        char_count=int,
        word_count=int,
        metadata=dict
    )
    def run(self, text: str) -> dict:
        """
        Process input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with multiple output types
            
        Raises:
            ValueError: If text doesn't meet length requirements
        """
        if len(text) < self.min_length:
            raise ValueError(f"Text too short: {len(text)} < {self.min_length}")
        
        if len(text) > self.max_length:
            raise ValueError(f"Text too long: {len(text)} > {self.max_length}")
        
        processed = text.strip()
        if self.remove_special_chars:
            processed = ''.join(c for c in processed if c.isalnum() or c.isspace())
        
        word_count = len(processed.split())
        char_count = len(processed)
        
        metadata = {
            "original_length": len(text),
            "processed_length": char_count,
            "word_count": word_count,
            "special_chars_removed": self.remove_special_chars
        }
        
        logger.debug(f"Processed text: {len(text)} -> {char_count} chars")
        
        return {
            "processed_text": processed,
            "char_count": char_count,
            "word_count": word_count,
            "metadata": metadata
        }

# Usage
processor = AdvancedTextProcessor(min_length=3, max_length=500)
result = processor.run("Hello world")
print(result)
# Output: {
#     'processed_text': 'Hello world',
#     'char_count': 11,
#     'word_count': 2,
#     'metadata': {...}
# }
```

### Creating Pipelines

Pipelines are constructed by adding components and connecting their inputs/outputs:

{% raw %}
```python
from haystack import Pipeline

# Create empty pipeline
pipeline = Pipeline()

# Add components
pipeline.add_component("processor", AdvancedTextProcessor())
pipeline.add_component("formatter", ResultFormatter())

# Connect outputs to inputs
# Format: "source_component.output_name" -> "target_component.input_name"
pipeline.connect("processor.processed_text", "formatter.text")

# Run pipeline
result = pipeline.run({"processor": {"text": "sample text"}})

# Access results
print(result["formatter"]["formatted_result"])
```
{% endraw %}

### Complex Pipeline with Branching

```python
from haystack import Pipeline, component

@component
class Classifier:
    @component.output_types(category=str, confidence=float)
    def run(self, text: str) -> dict:
        if len(text) > 10:
            return {"category": "long", "confidence": 0.9}
        return {"category": "short", "confidence": 0.8}

@component
class LongTextHandler:
    @component.output_types(result=str)
    def run(self, text: str) -> dict:
        return {"result": f"Long text processed: {text[:50]}..."}

@component
class ShortTextHandler:
    @component.output_types(result=str)
    def run(self, text: str) -> dict:
        return {"result": f"Short text: {text}"}

# Create pipeline with branching
pipeline = Pipeline()
pipeline.add_component("classifier", Classifier())
pipeline.add_component("long_handler", LongTextHandler())
pipeline.add_component("short_handler", ShortTextHandler())

# Add conditional connections (we'll handle this in the Conditional Branching section)
```

## Agent Concepts in Haystack

Agents in Haystack are autonomous systems that leverage LLMs and external tools to solve complex queries iteratively. Unlike simple components, agents maintain state, make decisions, and iterate until goals are reached.

### Agent Architecture

```
┌─────────────────────────────────────────────┐
│            User Query                        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│        Agent State Initialisation           │
│  - Load conversation history                │
│  - Prepare tool descriptions                │
│  - Set system prompt                        │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│        LLM Processing                       │
│  - Generate reasoning and action plan       │
│  - Produce tool calls or final response     │
└────────────────┬────────────────────────────┘
                 │
         ┌───────┴──────┐
         │              │
         ▼              ▼
   ┌──────────┐   ┌──────────────┐
   │Execute   │   │Final Answer  │
   │Tools     │   │Reached       │
   │          │   │              │
   └────┬─────┘   └──────┬───────┘
        │                │
        ▼                ▼
   ┌──────────┐     ┌──────────────┐
   │Update    │     │Return Result │
   │State     │     │              │
   └────┬─────┘     └──────────────┘
        │
        └──────────────┬──────────────┘
                       │
                   (Loop until
                    exit condition)
```

### Agent Fundamentals

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from haystack.dataclasses import ChatMessage
import json

# Define tools for the agent
def add_numbers(a: int, b: int) -> dict:
    """Add two numbers together."""
    return {"result": a + b}

def multiply_numbers(a: int, b: int) -> dict:
    """Multiply two numbers."""
    return {"result": a * b}

def get_current_time() -> dict:
    """Get current timestamp."""
    from datetime import datetime
    return {"time": datetime.now().isoformat()}

# Convert functions to Tool objects
tools = [
    Tool(function=add_numbers, description="Adds two integers"),
    Tool(function=multiply_numbers, description="Multiplies two integers"),
    Tool(function=get_current_time, description="Gets the current time")
]

# Initialise LLM generator
chat_generator = OpenAIChatGenerator(
    model="gpt-4o",
    api_key="your-api-key"
)

# Create agent
agent = Agent(
    tools=tools,
    llm=chat_generator,
    system_prompt="You are a helpful mathematical assistant. Use tools to solve problems.",
)

# Run agent
query = "What is 15 + 7 multiplied by 3?"
result = agent.run(query)
print(result)
```

### Agent State Management

Agents maintain internal state throughout their execution lifecycle:

```python
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, AgentState

# Agents have internal state tracking
class StateTrackingAgent(Agent):
    """
    Extended agent with visible state management.
    Demonstrates how agents maintain state across iterations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_history = []
    
    def run(self, query: str, max_iterations: int = 10) -> dict:
        """
        Run agent with state tracking.
        
        Args:
            query: User query
            max_iterations: Maximum iterations before timeout
            
        Returns:
            Final result with history
        """
        # Initialize state
        messages = [ChatMessage.from_user(query)]
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get LLM response
            llm_response = self._get_llm_response(messages)
            
            # Track execution
            self.execution_history.append({
                "iteration": iteration,
                "response": llm_response,
                "message_count": len(messages)
            })
            
            # Check exit condition
            if self._should_exit(llm_response):
                return {
                    "result": llm_response,
                    "iterations": iteration,
                    "history": self.execution_history
                }
            
            # Process tool calls or continue loop
            messages.append(ChatMessage.from_assistant(llm_response))
        
        return {
            "result": "Max iterations reached",
            "iterations": iteration,
            "history": self.execution_history
        }
    
    def _get_llm_response(self, messages: list) -> str:
        """Get response from LLM."""
        # Implementation would call LLM
        pass
    
    def _should_exit(self, response: str) -> bool:
        """Check if agent should exit."""
        # Implementation would check exit conditions
        pass
```

## Provider-Agnostic Design Philosophy

Haystack's provider-agnostic design is one of its strongest features. This allows applications to be built once and run with different providers by simply swapping components.

### Understanding Provider Abstraction

```python
from haystack.components.generators.chat import OpenAIChatGenerator, AnthropicChatGenerator, HuggingFaceAPIChatGenerator

# All generators implement the same interface
class CombinedLLMPipeline:
    """
    Pipeline that can use different LLM providers interchangeably.
    """
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize with specified provider.
        
        Args:
            provider: One of 'openai', 'anthropic', 'huggingface'
        """
        self.provider = provider
        self.generator = self._create_generator(provider)
    
    def _create_generator(self, provider: str):
        """Create appropriate generator based on provider."""
        if provider == "openai":
            return OpenAIChatGenerator(model="gpt-4o")
        elif provider == "anthropic":
            return AnthropicChatGenerator(model="claude-3-opus")
        elif provider == "huggingface":
            return HuggingFaceAPIChatGenerator(model="mistral")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate(self, messages: list) -> dict:
        """Generate response using configured provider."""
        return self.generator.run(messages=messages)

# Usage - same code works with different providers
for provider in ["openai", "anthropic", "huggingface"]:
    pipeline = CombinedLLMPipeline(provider=provider)
    response = pipeline.generate([{"role": "user", "content": "Hello"}])
    print(f"{provider}: {response}")
```

### Vector Database Provider Abstraction

```python
from haystack.document_stores.types import DocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.weaviate import WeaviateDocumentStore
from haystack.document_stores.qdrant import QdrantDocumentStore

class VectorStoreWrapper:
    """
    Wrapper that abstracts vector store provider.
    """
    
    def __init__(self, store_type: str = "elasticsearch", **kwargs):
        """
        Initialize vector store.
        
        Args:
            store_type: Type of store ('elasticsearch', 'weaviate', 'qdrant')
            **kwargs: Additional configuration
        """
        self.store = self._create_store(store_type, **kwargs)
    
    def _create_store(self, store_type: str, **kwargs) -> DocumentStore:
        """Create appropriate document store."""
        if store_type == "elasticsearch":
            return ElasticsearchDocumentStore(
                hosts="localhost",
                index="documents",
                **kwargs
            )
        elif store_type == "weaviate":
            return WeaviateDocumentStore(**kwargs)
        elif store_type == "qdrant":
            return QdrantDocumentStore(
                url="http://localhost:6333",
                collection_name="documents",
                **kwargs
            )
        else:
            raise ValueError(f"Unknown store type: {store_type}")
    
    def write_documents(self, documents: list):
        """Write documents to store."""
        return self.store.write_documents(documents)
    
    def query(self, query: str, top_k: int = 10):
        """Query documents."""
        return self.store.query_by_text(query, top_k=top_k)

# Usage - easily switch providers
for store_type in ["elasticsearch", "weaviate", "qdrant"]:
    store = VectorStoreWrapper(store_type=store_type)
    # Same interface across different providers
```

## Configuration Patterns and Best Practices

### Environment-Based Configuration

```python
import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class HaystackConfig:
    """
    Configuration container for Haystack applications.
    Loads from environment variables with sensible defaults.
    """
    
    # LLM Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # Document Store Configuration
    doc_store_type: str = os.getenv("DOC_STORE_TYPE", "elasticsearch")
    doc_store_host: str = os.getenv("DOC_STORE_HOST", "localhost")
    doc_store_port: int = int(os.getenv("DOC_STORE_PORT", "9200"))
    doc_store_index: str = os.getenv("DOC_STORE_INDEX", "haystack_docs")
    
    # Agent Configuration
    agent_system_prompt: str = os.getenv(
        "AGENT_SYSTEM_PROMPT",
        "You are a helpful assistant."
    )
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    agent_timeout: int = int(os.getenv("AGENT_TIMEOUT", "30"))
    
    # Observability
    enable_tracing: bool = os.getenv("ENABLE_TRACING", "true").lower() == "true"
    logging_level: str = os.getenv("LOGGING_LEVEL", "INFO")
    
    def validate(self):
        """Validate configuration."""
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY must be set")
        
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        
        if self.agent_max_iterations < 1:
            raise ValueError("AGENT_MAX_ITERATIONS must be at least 1")

# Usage
config = HaystackConfig()
config.validate()
print(f"Using {config.llm_provider} with model {config.llm_model}")
```

### YAML Configuration Files

```yaml
# config.yaml
haystack:
  llm:
    provider: openai
    model: gpt-4o
    temperature: 0.7
    max_tokens: 4096
  
  document_store:
    type: elasticsearch
    host: localhost
    port: 9200
    index: haystack_documents
    embedding_model: text-embedding-3-small
  
  agent:
    system_prompt: "You are a helpful AI assistant."
    max_iterations: 10
    timeout: 30
    exit_conditions:
      - type: token_count
        value: 5000
      - type: iteration_count
        value: 10
  
  observability:
    tracing_enabled: true
    logging_level: INFO
    metrics_export: prometheus
```

```python
import yaml
from pathlib import Path

def load_config_from_yaml(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Usage
config = load_config_from_yaml("config.yaml")
haystack_config = config.get("haystack", {})
```

---

# PART II: SIMPLE AGENTS

## Creating Agents with the Agent Class

The `Agent` class is the foundation for building autonomous AI systems in Haystack. Unlike components which perform discrete tasks, agents maintain state and iterate to solve complex problems.

### Basic Agent Structure

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from haystack.dataclasses import ChatMessage

# Step 1: Define tools
def weather_tool(location: str) -> dict:
    """Get current weather for a location."""
    # In real application, call actual weather API
    weather_data = {
        "London": {"temperature": 15, "condition": "Rainy"},
        "New York": {"temperature": 25, "condition": "Sunny"},
        "Tokyo": {"temperature": 20, "condition": "Cloudy"}
    }
    return weather_data.get(location, {"error": "Location not found"})

def search_tool(query: str) -> dict:
    """Search for information."""
    # Simulated search
    return {"results": [f"Result for '{query}' #1", f"Result for '{query}' #2"]}

# Step 2: Create Tool objects
tools = [
    Tool(function=weather_tool, description="Get weather for a location"),
    Tool(function=search_tool, description="Search for information")
]

# Step 3: Initialise LLM
llm = OpenAIChatGenerator(model="gpt-4o", api_key="sk-...")

# Step 4: Create agent
agent = Agent(
    tools=tools,
    llm=llm,
    system_prompt="You are a helpful assistant with access to weather and search tools.",
)

# Step 5: Run agent
response = agent.run(
    query="What's the weather in London and any recent news?",
    max_iterations=10
)

print(response)
```

### Agent with Custom Configuration

```python
from haystack.components.agents import Agent, AgentConfig
from haystack.dataclasses import ChatMessage

# Create custom configuration
config = AgentConfig(
    max_iterations=15,
    timeout_seconds=60,
    enable_streaming=True,
    temperature=0.5,
    top_p=0.9
)

# Create agent with custom config
agent = Agent(
    tools=tools,
    llm=llm,
    system_prompt="You are a careful, methodical assistant.",
    config=config
)

# Run with custom parameters
result = agent.run(
    query="Analyse the following data and provide insights",
    max_iterations=config.max_iterations
)
```

### Stateful Agent for Conversations

```python
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from typing import Optional

class ConversationalAgent:
    """
    Agent that maintains conversation state across multiple turns.
    """
    
    def __init__(self, agent: Agent, system_prompt: str):
        self.agent = agent
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    def add_user_message(self, query: str):
        """Add user message to history."""
        self.conversation_history.append(
            ChatMessage.from_user(query)
        )
    
    def add_assistant_message(self, response: str):
        """Add assistant message to history."""
        self.conversation_history.append(
            ChatMessage.from_assistant(response)
        )
    
    def run(self, query: str) -> str:
        """
        Run agent with conversation context.
        
        Args:
            query: User query for this turn
            
        Returns:
            Agent response
        """
        self.add_user_message(query)
        
        # Run agent with full conversation history
        result = self.agent.run(
            query=query,
            messages=self.conversation_history
        )
        
        # Extract response
        response = result.get("response", "")
        self.add_assistant_message(response)
        
        return response
    
    def get_history(self) -> list:
        """Get full conversation history."""
        return self.conversation_history
    
    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []

# Usage
conversational_agent = ConversationalAgent(
    agent=agent,
    system_prompt="Remember the context of our conversation."
)

# First turn
response1 = conversational_agent.run("What are my favourite colours?")
# Agent: "I don't have information about your favourite colours yet."

# Second turn
response2 = conversational_agent.run("My favourite colour is blue.")
# Agent stores this in history

# Third turn
response3 = conversational_agent.run("What did I just tell you?")
# Agent can reference previous messages: "You told me your favourite colour is blue."
```

## Tool Integration and Function Calling

Tools are how agents interact with external systems. They bridge the gap between LLM reasoning and real-world actions.

### Creating Custom Tools

```python
from haystack.tools import Tool
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Simple tool - function with type hints
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: int,
    compounds_per_year: int = 1
) -> dict:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount
        rate: Interest rate (as percentage)
        time: Time period in years
        compounds_per_year: Compounding frequency
        
    Returns:
        Dictionary with calculation results
    """
    amount = principal * (1 + rate / 100 / compounds_per_year) ** (compounds_per_year * time)
    interest = amount - principal
    
    return {
        "principal": principal,
        "rate": rate,
        "time": time,
        "final_amount": round(amount, 2),
        "interest_earned": round(interest, 2)
    }

# Register as tool
compound_interest_tool = Tool(
    function=calculate_compound_interest,
    description="Calculates compound interest given principal, rate, and time period"
)

# Complex tool - database query
class DatabaseQueryTool:
    """Tool for querying a database."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        logger.info(f"Initialised DatabaseQueryTool with {connection_string}")
    
    def query(self, sql: str, params: Optional[dict] = None) -> dict:
        """
        Execute SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        # Simulate database query
        logger.debug(f"Executing query: {sql}")
        
        # In real application, execute actual query
        return {
            "query": sql,
            "rows": 10,
            "results": [{"id": 1, "name": "Example"}]
        }

# Create instance and tool
db_tool_instance = DatabaseQueryTool("postgresql://localhost/mydb")
database_tool = Tool(
    function=db_tool_instance.query,
    description="Execute database queries"
)

# Web API tool with error handling
import requests
from requests.exceptions import RequestException

def call_external_api(
    endpoint: str,
    method: str = "GET",
    payload: Optional[dict] = None,
    timeout: int = 10
) -> dict:
    """
    Call external API with error handling.
    
    Args:
        endpoint: API endpoint URL
        method: HTTP method (GET, POST, etc.)
        payload: Request payload for POST/PUT
        timeout: Request timeout in seconds
        
    Returns:
        API response or error information
    """
    try:
        if method == "GET":
            response = requests.get(endpoint, timeout=timeout)
        elif method == "POST":
            response = requests.post(endpoint, json=payload, timeout=timeout)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        response.raise_for_status()
        
        return {
            "status": response.status_code,
            "data": response.json(),
            "success": True
        }
    
    except RequestException as e:
        logger.error(f"API call failed: {str(e)}")
        return {
            "status": getattr(e.response, 'status_code', None),
            "error": str(e),
            "success": False
        }

api_tool = Tool(
    function=call_external_api,
    description="Call external REST APIs"
)
```

### Tool Validation and Error Handling

{% raw %}
```python
from haystack.tools import Tool
from typing import Callable
import inspect

class ToolValidator:
    """Validates tools and their function signatures."""
    
    @staticmethod
    def validate_tool_function(func: Callable) -> dict:
        """
        Validate that function is suitable for use as a tool.
        
        Returns:
            Validation results
        """
        issues = []
        
        # Check for type hints
        sig = inspect.signature(func)
        if not all(param.annotation != inspect.Parameter.empty 
                   for param in sig.parameters.values()):
            issues.append("Not all parameters have type hints")
        
        if sig.return_annotation == inspect.Signature.empty:
            issues.append("Return type not specified")
        
        # Check for docstring
        if not func.__doc__:
            issues.append("Function missing docstring")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "parameters": list(sig.parameters.keys()),
            "return_type": str(sig.return_annotation)
        }

# Validate tools before use
validation_result = ToolValidator.validate_tool_function(calculate_compound_interest)
print(validation_result)

# Tool with input validation
def validated_calculator(expression: str) -> dict:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Calculation result
    """
    import re
    
    # Only allow safe characters
    if not re.match(r'^[0-9+\-*/().]+$', expression):
        return {"error": "Invalid characters in expression", "success": False}
    
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

calculator_tool = Tool(
    function=validated_calculator,
    description="Evaluate mathematical expressions safely"
)
```
{% endraw %}

### Tool Invocation in Agents

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool

# Create tools
tools = [
    Tool(function=calculate_compound_interest, description="Calculate compound interest"),
    Tool(function=validated_calculator, description="Evaluate mathematical expressions"),
]

# Create agent
llm = OpenAIChatGenerator(model="gpt-4o")
agent = Agent(tools=tools, llm=llm)

# The agent will automatically call tools when needed
response = agent.run(
    query="If I invest £1000 at 5% interest for 10 years with quarterly compounding, "
          "how much will I have? Then calculate the monthly return rate."
)

print(response)
```

## Single-Purpose Agents

Single-purpose agents are specialised agents designed to solve one specific type of problem efficiently.

### Customer Support Agent

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from typing import Optional

class CustomerSupportAgent:
    """
    Agent specialised for customer support tasks.
    """
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.support_history = []
        self._setup_tools()
        self._setup_agent()
    
    def _load_knowledge_base(self) -> dict:
        """Load customer support knowledge base."""
        return {
            "returns": "Items can be returned within 30 days for full refund.",
            "shipping": "Standard shipping takes 5-7 business days.",
            "warranty": "All products come with a 1-year warranty.",
            "contact": "Support available 9AM-5PM GMT, Monday-Friday."
        }
    
    def _setup_tools(self):
        """Set up support tools."""
        
        def lookup_knowledge(topic: str) -> dict:
            """Look up information in knowledge base."""
            result = self.knowledge_base.get(
                topic.lower(),
                "Information not found in knowledge base."
            )
            return {"topic": topic, "answer": result}
        
        def check_order_status(order_id: str) -> dict:
            """Check order status (simulated)."""
            # In real application, query actual database
            return {
                "order_id": order_id,
                "status": "Shipped",
                "tracking": "TRK123456789"
            }
        
        def create_support_ticket(issue: str, priority: str = "normal") -> dict:
            """Create support ticket."""
            return {
                "ticket_id": "TICKET-001",
                "issue": issue,
                "priority": priority,
                "created": True
            }
        
        self.tools = [
            Tool(
                function=lookup_knowledge,
                description="Lookup information in customer support knowledge base"
            ),
            Tool(
                function=check_order_status,
                description="Check status of a customer order"
            ),
            Tool(
                function=create_support_ticket,
                description="Create a support ticket for unresolved issues"
            ),
        ]
    
    def _setup_agent(self):
        """Initialise the agent."""
        llm = OpenAIChatGenerator(model="gpt-4o")
        self.agent = Agent(
            tools=self.tools,
            llm=llm,
            system_prompt="""You are a helpful customer support specialist. 
            Your goal is to resolve customer issues efficiently using available tools.
            Always be polite and professional. 
            If you cannot resolve an issue, create a support ticket."""
        )
    
    def handle_query(self, query: str) -> dict:
        """
        Handle customer query.
        
        Args:
            query: Customer's question or issue
            
        Returns:
            Support response
        """
        response = self.agent.run(query=query, max_iterations=5)
        self.support_history.append({"query": query, "response": response})
        return response
    
    def get_history(self) -> list:
        """Get support interaction history."""
        return self.support_history

# Usage
support_agent = CustomerSupportAgent()
response = support_agent.handle_query("Can I return my order? My order ID is ORD-12345.")
print(response)
```

### Data Analysis Agent

```python
import pandas as pd
import numpy as np
from haystack.components.agents import Agent
from haystack.tools import Tool

class DataAnalysisAgent:
    """
    Agent specialised in data analysis tasks.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self._setup_tools()
        self._setup_agent()
    
    def _setup_tools(self):
        """Set up data analysis tools."""
        
        def describe_data() -> dict:
            """Get statistical summary of data."""
            return {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "dtypes": self.df.dtypes.to_dict(),
                "summary": self.df.describe().to_dict()
            }
        
        def column_statistics(column: str) -> dict:
            """Get statistics for a specific column."""
            if column not in self.df.columns:
                return {"error": f"Column {column} not found"}
            
            col_data = self.df[column]
            return {
                "column": column,
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "missing": int(col_data.isna().sum())
            }
        
        def filter_data(column: str, condition: str, value) -> dict:
            """Filter data based on conditions."""
            try:
                if condition == "equals":
                    filtered = self.df[self.df[column] == value]
                elif condition == "greater_than":
                    filtered = self.df[self.df[column] > value]
                elif condition == "less_than":
                    filtered = self.df[self.df[column] < value]
                else:
                    return {"error": f"Unknown condition: {condition}"}
                
                return {
                    "original_rows": len(self.df),
                    "filtered_rows": len(filtered),
                    "sample": filtered.head(5).to_dict(orient='records')
                }
            except Exception as e:
                return {"error": str(e)}
        
        def correlate_columns(col1: str, col2: str) -> dict:
            """Calculate correlation between two columns."""
            try:
                corr = self.df[col1].corr(self.df[col2])
                return {
                    "column1": col1,
                    "column2": col2,
                    "correlation": float(corr),
                    "interpretation": self._interpret_correlation(corr)
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.tools = [
            Tool(function=describe_data, description="Get statistical summary of dataset"),
            Tool(function=column_statistics, description="Get statistics for a specific column"),
            Tool(function=filter_data, description="Filter data based on column values"),
            Tool(function=correlate_columns, description="Calculate correlation between columns"),
        ]
    
    def _setup_agent(self):
        """Initialise the agent."""
        llm = OpenAIChatGenerator(model="gpt-4o")
        self.agent = Agent(
            tools=self.tools,
            llm=llm,
            system_prompt="""You are a data analyst. Analyse the provided dataset using available tools.
            Provide insights, patterns, and recommendations based on the data."""
        )
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient."""
        if abs(corr) < 0.3:
            return "Weak correlation"
        elif abs(corr) < 0.7:
            return "Moderate correlation"
        else:
            return "Strong correlation"
    
    def analyse(self, query: str) -> dict:
        """
        Analyse data based on query.
        
        Args:
            query: Analysis query
            
        Returns:
            Analysis results
        """
        return self.agent.run(query=query, max_iterations=10)

# Usage
data = pd.DataFrame({
    'Sales': [100, 150, 200, 180, 220],
    'Marketing': [10, 15, 20, 18, 25],
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May']
})

analyst = DataAnalysisAgent(data)
results = analyst.analyse("What's the relationship between marketing spend and sales?")
print(results)
```

## Conversational Agents with Memory

Conversational agents maintain context across multiple turns, enabling natural multi-turn dialogue.

### Basic Conversation Memory

```python
from haystack.dataclasses import ChatMessage
from typing import List
from datetime import datetime

class ConversationMemory:
    """
    Manages conversation history with metadata.
    """
    
    def __init__(self, max_messages: int = 100):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
        self.created_at = datetime.now()
        self.metadata = {}
    
    def add_message(self, role: str, content: str, metadata: dict = None):
        """Add message to conversation."""
        message = ChatMessage(role=role, content=content)
        message.metadata = metadata or {}
        message.timestamp = datetime.now()
        
        self.messages.append(message)
        
        # Maintain maximum message limit
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_last_n_messages(self, n: int = 10) -> List[ChatMessage]:
        """Get last N messages."""
        return self.messages[-n:]
    
    def get_context(self) -> str:
        """Get conversation context as string."""
        context_parts = []
        for msg in self.get_last_n_messages(20):
            context_parts.append(f"{msg.role}: {msg.content}")
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []
    
    def get_summary(self) -> dict:
        """Get conversation summary."""
        return {
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "duration": (datetime.now() - self.created_at).total_seconds(),
            "roles": {
                "user": sum(1 for m in self.messages if m.role == "user"),
                "assistant": sum(1 for m in self.messages if m.role == "assistant")
            }
        }

class ConversationalAgentWithMemory:
    """
    Agent with built-in conversation memory.
    """
    
    def __init__(self, agent: Agent, memory: ConversationMemory = None):
        self.agent = agent
        self.memory = memory or ConversationMemory()
    
    def chat(self, user_message: str) -> str:
        """
        Chat with agent, maintaining memory.
        
        Args:
            user_message: User's message
            
        Returns:
            Agent's response
        """
        # Add user message to memory
        self.memory.add_message("user", user_message)
        
        # Get conversation context
        context = self.memory.get_context()
        
        # Run agent with context
        prompt = f"Previous conversation:\n{context}\n\nRespond to: {user_message}"
        response = self.agent.run(query=prompt)
        
        # Extract response text (may vary by agent implementation)
        response_text = str(response)
        
        # Add assistant response to memory
        self.memory.add_message("assistant", response_text)
        
        return response_text
    
    def get_conversation_history(self) -> List[dict]:
        """Get full conversation history."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in self.memory.messages
        ]
    
    def reset_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

# Usage
llm = OpenAIChatGenerator(model="gpt-4o")
agent = Agent(tools=[], llm=llm)
memory_agent = ConversationalAgentWithMemory(agent=agent)

# Multi-turn conversation
responses = []
responses.append(memory_agent.chat("Hi, my name is Alice"))
responses.append(memory_agent.chat("What did I just tell you?"))
responses.append(memory_agent.chat("Tell me a joke"))

print("Conversation history:")
for turn in memory_agent.get_conversation_history():
    print(f"{turn['role']}: {turn['content']}")
```

## Agent Configuration and Customisation

Agents can be customised extensively to meet specific requirements.

### Advanced Agent Configuration

```python
from haystack.components.agents import Agent, AgentConfig
from typing import Optional, Callable
from enum import Enum

class ExitStrategy(Enum):
    """Strategies for agent exit conditions."""
    EXACT_MATCH = "exact_match"
    KEYWORD = "keyword"
    ITERATION_LIMIT = "iteration_limit"
    TOKEN_LIMIT = "token_limit"
    CUSTOM = "custom"

class AdvancedAgentConfig:
    """
    Advanced configuration for agents with fine-grained control.
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        timeout_seconds: int = 60,
        temperature: float = 0.7,
        top_p: float = 1.0,
        exit_strategy: ExitStrategy = ExitStrategy.ITERATION_LIMIT,
        enable_logging: bool = True,
        enable_tracing: bool = False,
        custom_exit_check: Optional[Callable] = None
    ):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.top_p = top_p
        self.exit_strategy = exit_strategy
        self.enable_logging = enable_logging
        self.enable_tracing = enable_tracing
        self.custom_exit_check = custom_exit_check
    
    def validate(self) -> bool:
        """Validate configuration."""
        assert 0 <= self.temperature <= 2, "Temperature must be 0-2"
        assert 0 < self.top_p <= 1, "top_p must be 0-1"
        assert self.max_iterations > 0, "max_iterations must be positive"
        assert self.timeout_seconds > 0, "timeout_seconds must be positive"
        return True

def create_configurable_agent(
    tools: list,
    llm,
    config: AdvancedAgentConfig
) -> Agent:
    """
    Create agent with advanced configuration.
    
    Args:
        tools: List of tools
        llm: Language model
        config: Configuration object
        
    Returns:
        Configured agent
    """
    config.validate()
    
    # Set LLM parameters
    llm.temperature = config.temperature
    llm.top_p = config.top_p
    
    # Create agent
    agent = Agent(
        tools=tools,
        llm=llm,
        system_prompt="You are a helpful assistant.",
        temperature=config.temperature
    )
    
    # Store config for later reference
    agent._config = config
    agent._iteration_count = 0
    agent._start_time = None
    
    return agent

# Usage
advanced_config = AdvancedAgentConfig(
    max_iterations=20,
    timeout_seconds=120,
    temperature=0.5,
    exit_strategy=ExitStrategy.TOKEN_LIMIT
)

llm = OpenAIChatGenerator(model="gpt-4o")
agent = create_configurable_agent(
    tools=[],
    llm=llm,
    config=advanced_config
)
```

## Error Handling and Resilience

Robust agents handle errors gracefully and recover when possible.

### Comprehensive Error Handling

```python
from haystack.components.agents import Agent
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class ErrorHandlingAgent:
    """
    Agent with comprehensive error handling and recovery.
    """
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.error_log = []
        self.retry_policy = {
            "max_retries": 3,
            "backoff_factor": 2.0,
            "timeout": 30
        }
    
    def run_with_retry(
        self,
        query: str,
        max_retries: Optional[int] = None
    ) -> dict:
        """
        Run agent with retry logic.
        
        Args:
            query: Query to run
            max_retries: Maximum number of retries
            
        Returns:
            Result or error information
        """
        max_retries = max_retries or self.retry_policy["max_retries"]
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                logger.info(f"Running agent query (attempt {attempt + 1}/{max_retries})")
                result = self.agent.run(query=query, max_iterations=10)
                logger.info("Agent query succeeded")
                return {"success": True, "result": result}
            
            except TimeoutError as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
                last_error = e
                attempt += 1
                if attempt < max_retries:
                    import time
                    wait_time = self.retry_policy["backoff_factor"] ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                self.error_log.append({
                    "attempt": attempt + 1,
                    "error": str(e),
                    "type": type(e).__name__
                })
                last_error = e
                attempt += 1
        
        return {
            "success": False,
            "error": str(last_error),
            "attempts": max_retries,
            "error_log": self.error_log
        }
    
    def get_error_statistics(self) -> dict:
        """Get error statistics."""
        if not self.error_log:
            return {"error_count": 0}
        
        error_types = {}
        for entry in self.error_log:
            error_type = entry["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "last_error": self.error_log[-1] if self.error_log else None
        }

# Usage
error_handling_agent = ErrorHandlingAgent(agent=agent)
result = error_handling_agent.run_with_retry(
    query="Execute a complex analysis",
    max_retries=5
)
print(result)
```

---

# PART III: MULTI-AGENT SYSTEMS

Multi-agent systems enable complex problem-solving by coordinating multiple specialised agents.

## Multi-Agent Pipeline Design Patterns

### Sequential Agent Pipeline

```python
from haystack import Pipeline, component
from haystack.components.agents import Agent

class SequentialMultiAgentPipeline:
    """
    Pipeline where agents execute sequentially, with each agent processing output from previous agent.
    """
    
    def __init__(self):
        self.pipeline = Pipeline()
        self.agents = {}
    
    def add_agent(self, name: str, agent: Agent, position: int):
        """Add agent to pipeline."""
        self.agents[name] = {"agent": agent, "position": position}
    
    def build_pipeline(self):
        """Build sequential pipeline."""
        # Sort agents by position
        sorted_agents = sorted(
            self.agents.items(),
            key=lambda x: x[1]["position"]
        )
        
        previous_name = None
        for name, agent_info in sorted_agents:
            self.pipeline.add_component(name, agent_info["agent"])
            
            if previous_name:
                # Connect previous agent output to current agent input
                self.pipeline.connect(f"{previous_name}.output", f"{name}.input")
            
            previous_name = name
    
    def run(self, initial_input: dict) -> dict:
        """Run sequential pipeline."""
        return self.pipeline.run(initial_input)

# Example: Document Processing Pipeline
@component
class DocumentAgentWrapper:
    """Wraps agent for use in pipeline."""
    
    def __init__(self, agent: Agent, name: str):
        self.agent = agent
        self.name = name
    
    @component.output_types(output=dict)
    def run(self, input: dict) -> dict:
        result = self.agent.run(query=str(input))
        return {"output": result}
```

### Parallel Agent Execution

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ParallelMultiAgentPipeline:
    """
    Executes multiple agents in parallel for different aspects of a problem.
    """
    
    def __init__(self, max_workers: int = 4):
        self.agents: Dict[str, Agent] = {}
        self.max_workers = max_workers
    
    def add_agent(self, name: str, agent: Agent):
        """Add agent to parallel pipeline."""
        self.agents[name] = agent
    
    def run_parallel(self, queries: Dict[str, str]) -> Dict[str, dict]:
        """
        Run agents in parallel.
        
        Args:
            queries: Dictionary mapping agent names to their queries
            
        Returns:
            Results from all agents
        """
        results = {}
        errors = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_agent = {
                executor.submit(
                    self._run_agent_safely,
                    agent_name,
                    queries.get(agent_name, "")
                ): agent_name
                for agent_name in self.agents.keys()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    logger.info(f"Agent {agent_name} completed successfully")
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {str(e)}")
                    errors[agent_name] = str(e)
        
        return {
            "results": results,
            "errors": errors,
            "success_count": len(results),
            "error_count": len(errors)
        }
    
    def _run_agent_safely(self, agent_name: str, query: str) -> dict:
        """Run single agent with error handling."""
        agent = self.agents[agent_name]
        try:
            result = agent.run(query=query, max_iterations=10)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Usage
parallel_pipeline = ParallelMultiAgentPipeline(max_workers=4)

# Add specialist agents
analysis_agent = Agent(tools=[], llm=llm, system_prompt="You are a data analyst")
writing_agent = Agent(tools=[], llm=llm, system_prompt="You are a writer")
coding_agent = Agent(tools=[], llm=llm, system_prompt="You are a programmer")

parallel_pipeline.add_agent("analyst", analysis_agent)
parallel_pipeline.add_agent("writer", writing_agent)
parallel_pipeline.add_agent("coder", coding_agent)

# Run in parallel
results = parallel_pipeline.run_parallel({
    "analyst": "Analyse the sales data for Q4",
    "writer": "Write a summary of Q4 performance",
    "coder": "Implement the Q4 dashboard"
})

print(results)
```

### Router-Based Agent Dispatching

```python
from enum import Enum
from typing import Optional

class AgentRole(Enum):
    """Roles that agents can fulfil."""
    DATA_ANALYST = "data_analyst"
    CUSTOMER_SERVICE = "customer_service"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL = "general"

class IntelligentAgentRouter:
    """
    Routes queries to appropriate agent based on content analysis.
    """
    
    def __init__(self):
        self.agents: Dict[AgentRole, Agent] = {}
        self.routing_keywords = {
            AgentRole.DATA_ANALYST: ["analyse", "data", "statistics", "trend"],
            AgentRole.CUSTOMER_SERVICE: ["order", "refund", "complaint", "support"],
            AgentRole.TECHNICAL_SUPPORT: ["bug", "error", "crash", "technical", "code"],
            AgentRole.GENERAL: []
        }
    
    def register_agent(self, role: AgentRole, agent: Agent):
        """Register agent for specific role."""
        self.agents[role] = agent
    
    def determine_route(self, query: str) -> AgentRole:
        """
        Determine appropriate agent for query.
        
        Args:
            query: User query
            
        Returns:
            Appropriate agent role
        """
        query_lower = query.lower()
        
        # Check keywords for each role
        for role, keywords in self.routing_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return role
        
        # Default to general agent
        return AgentRole.GENERAL
    
    def route_and_process(self, query: str) -> dict:
        """
        Route query to appropriate agent and process.
        
        Args:
            query: User query
            
        Returns:
            Routing and processing results
        """
        # Determine route
        route = self.determine_route(query)
        logger.info(f"Routed query to: {route.value}")
        
        # Get agent
        agent = self.agents.get(route)
        if not agent:
            return {
                "success": False,
                "error": f"No agent registered for role {route.value}"
            }
        
        # Process with appropriate agent
        try:
            result = agent.run(query=query, max_iterations=10)
            return {
                "success": True,
                "route": route.value,
                "result": result
            }
        except Exception as e:
            logger.error(f"Agent processing failed: {str(e)}")
            return {
                "success": False,
                "route": route.value,
                "error": str(e)
            }

# Usage
router = IntelligentAgentRouter()

# Register specialist agents
data_agent = Agent(tools=[], llm=llm, system_prompt="You are a data analyst")
support_agent = Agent(tools=[], llm=llm, system_prompt="You are customer support")
tech_agent = Agent(tools=[], llm=llm, system_prompt="You are technical support")

router.register_agent(AgentRole.DATA_ANALYST, data_agent)
router.register_agent(AgentRole.CUSTOMER_SERVICE, support_agent)
router.register_agent(AgentRole.TECHNICAL_SUPPORT, tech_agent)

# Route and process various queries
queries = [
    "Can you analyse our sales trends?",
    "I need to return my order",
    "I'm getting an error in the application"
]

for query in queries:
    result = router.route_and_process(query)
    print(f"Query: {query}")
    print(f"Route: {result['route']}")
    print(f"Result: {result['result']}\n")
```

---

Due to token constraints, I'll now create the remaining comprehensive files. Let me continue with creating the other essential guides:

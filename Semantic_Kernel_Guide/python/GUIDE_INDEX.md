# Semantic Kernel Python - Complete Guide Index

**Quick Reference and Navigation for All Python Documentation**

Last Updated: November 2025

---

## Table of Contents

- [By Topic](#by-topic)
- [By Skill Level](#by-skill-level)
- [By Use Case](#by-use-case)
- [By Feature (2025)](#by-feature-2025)
- [Quick Reference](#quick-reference)

---

## By Topic

### Installation & Setup
- [README.md](README.md) → Quick Start
- [Comprehensive Guide](semantic_kernel_comprehensive_python.md) → Section 1: Installation & Configuration
- [Production Guide](semantic_kernel_production_python.md) → Section 1: Environment Setup

### Core Concepts

#### Kernel & Services
- **Comprehensive Guide → Section 2:** Kernel Initialization
- **Comprehensive Guide → Section 3:** Service Configuration (OpenAI, Azure)
- **Recipes → Basic Setup Examples**

#### Functions & Plugins
- **Comprehensive Guide → Section 4:** Semantic Functions
- **Comprehensive Guide → Section 5:** Native Functions
- **Comprehensive Guide → Section 6:** Plugin Development
- **Recipes → Plugin Examples:** Calculator, HTTP Client, Data Processing

#### Agents
- **Comprehensive Guide → Section 7:** Simple Agents
- **Comprehensive Guide → Section 8:** ChatCompletionAgent
- **Advanced Multi-Agent Guide:** Complete multi-agent reference
- **Recipes → Agent Examples:** Research Assistant, Debate System

#### Memory & Vector Stores
- **Comprehensive Guide → Section 9:** Memory Systems
- **Comprehensive Guide → Section 10:** Vector Store v1.34 (NEW)
- **Comprehensive Guide → Section 11:** Embeddings & Semantic Search
- **Recipes → RAG Implementation**

#### Planners & Orchestration
- **Comprehensive Guide → Section 12:** Sequential Planner
- **Comprehensive Guide → Section 13:** Stepwise Planner
- **Comprehensive Guide → Section 14:** Action Planner
- **Advanced Multi-Agent Guide → Orchestration Patterns**

### Advanced Topics

#### Multi-Agent Systems
- **[Advanced Multi-Agent Guide](semantic_kernel_advanced_multi_agent_python.md):** Complete reference
  - AgentGroupChat
  - Termination Strategies
  - Role Routing
  - Capability Routing
  - Human-in-the-Loop Handoffs
  - Resilience Patterns

#### Middleware & Guardrails
- **[Middleware Guide](semantic_kernel_middleware_python.md):** Complete reference
  - Execution Wrapper Pattern
  - Policy Enforcement
  - Content Filtering
  - PII Detection
  - Custom Middleware Development

#### Streaming
- **[Streaming Server Guide](semantic_kernel_streaming_server_fastapi.md):** FastAPI implementation
  - Server-Sent Events (SSE)
  - WebSocket alternatives
  - Incremental response streaming
  - Production deployment

### 2025 Features

#### Model Context Protocol (MCP)
- **Comprehensive Guide → Section 15:** MCP Overview & Architecture
- **Comprehensive Guide → Section 16:** MCP Client Implementation (Python)
- **Comprehensive Guide → Section 17:** MCP Server Creation
- **Recipes → MCP Integration Examples**

#### Vector Store Overhaul (v1.34)
- **Comprehensive Guide → Section 10:** Vector Store v1.34 Features
  - Unified API
  - Performance Improvements
  - New Connectors
- **Recipes → Vector Store Migration Guide**

#### Google A2A Protocol
- **Comprehensive Guide → Section 18:** A2A Protocol Integration
  - Agent-to-Agent Communication
  - Message Routing
  - Interoperability Patterns
- **Recipes → A2A Examples**

#### Microsoft Agent Framework Integration
- **Comprehensive Guide → Section 19:** Agent Framework Integration
  - Unified Agent SDK
  - Cross-Framework Communication
  - Enterprise Governance

### Production & Deployment

#### Deployment Strategies
- **Production Guide → Section 2:** Docker Containerization
- **Production Guide → Section 3:** Kubernetes Deployment
- **Production Guide → Section 4:** Azure Container Apps
- **Production Guide → Section 5:** Azure Functions
- **Streaming Server Guide:** FastAPI Production Deployment

#### Performance & Optimization
- **Production Guide → Section 6:** Caching Strategies
- **Production Guide → Section 7:** Batching & Parallelization
- **Production Guide → Section 8:** Async Optimization
- **Production Guide → Section 9:** Token Management

#### Monitoring & Observability
- **Production Guide → Section 10:** OpenTelemetry Integration
- **Production Guide → Section 11:** Application Insights
- **Production Guide → Section 12:** Logging & Metrics
- **Production Guide → Section 13:** Custom Telemetry

#### Security
- **Production Guide → Section 14:** Azure Key Vault Integration
- **Production Guide → Section 15:** Input Validation
- **Production Guide → Section 16:** Secrets Management
- **Production Guide → Section 17:** Authentication & Authorization
- **Middleware Guide:** Guardrails & Policy Enforcement

#### Error Handling & Resilience
- **Production Guide → Section 18:** Retry Patterns (tenacity)
- **Production Guide → Section 19:** Circuit Breakers
- **Production Guide → Section 20:** Graceful Degradation
- **Advanced Multi-Agent Guide → Resilience Section**

#### Testing
- **Production Guide → Section 21:** Unit Testing
- **Production Guide → Section 22:** Integration Testing
- **Production Guide → Section 23:** End-to-End Testing
- **Production Guide → Section 24:** Mocking Strategies

### Recipes & Examples

#### Basic Recipes
- **Recipes → Section 1:** Q&A System
- **Recipes → Section 2:** Content Summarizer
- **Recipes → Section 3:** Translation Service
- **Recipes → Section 4:** Sentiment Analyzer

#### Plugin Recipes
- **Recipes → Section 5:** Custom Calculator Plugin
- **Recipes → Section 6:** HTTP Request Plugin
- **Recipes → Section 7:** Data Processing Plugin
- **Recipes → Section 8:** Database Plugin

#### Memory & Retrieval Recipes
- **Recipes → Section 9:** Document QA System
- **Recipes → Section 10:** RAG Implementation
- **Recipes → Section 11:** Similarity Search
- **Recipes → Section 12:** Semantic Caching

#### Multi-Agent Recipes
- **Recipes → Section 13:** Debate System
- **Recipes → Section 14:** Research Assistant
- **Recipes → Section 15:** Collaborative Problem Solving
- **Recipes → Section 16:** Code Review Agent Team

#### Advanced Recipes
- **Recipes → Section 17:** ReAct Agent
- **Recipes → Section 18:** Dynamic Planning
- **Recipes → Section 19:** Streaming Chat Server
- **Recipes → Section 20:** MCP Integration

### Diagrams & Visualizations

- **[Diagrams Guide](semantic_kernel_diagrams_python.md):**
  - Architecture Diagrams
  - Request Flow Diagrams
  - Multi-Agent Orchestration Patterns
  - Memory Pipeline Visualizations
  - Async Execution Patterns
  - MCP Architecture
  - A2A Protocol Flow

---

## By Skill Level

### Beginner (New to Semantic Kernel)

**Week 1: Fundamentals**
1. [README.md](README.md) → Quick Start
2. Comprehensive Guide → Sections 1-3 (Installation, Kernel, Services)
3. Recipes → Sections 1-4 (Basic Examples)

**Week 2: Building Blocks**
4. Comprehensive Guide → Sections 4-6 (Functions & Plugins)
5. Comprehensive Guide → Section 7-8 (Simple Agents)
6. Recipes → Sections 5-8 (Plugin Recipes)

**Practice Projects:**
- Build a simple chatbot
- Create a Q&A system with memory
- Develop a custom plugin

### Intermediate (Familiar with SK Basics)

**Weeks 3-4: Advanced Patterns**
1. Comprehensive Guide → Sections 9-11 (Memory & Vector Stores)
2. Comprehensive Guide → Sections 12-14 (Planners)
3. Advanced Multi-Agent Guide → Sections 1-3
4. Recipes → Sections 9-12 (RAG & Memory)

**Weeks 5-6: Multi-Agent Systems**
5. Advanced Multi-Agent Guide → Complete
6. Recipes → Sections 13-16 (Multi-Agent Examples)
7. Middleware Guide → Guardrails & Policy

**Practice Projects:**
- Implement RAG system
- Build multi-agent debate system
- Create research assistant with tools

### Advanced (Production-Ready Systems)

**Weeks 7-8: Production Deployment**
1. Production Guide → Sections 1-13 (Deployment to Monitoring)
2. Production Guide → Sections 14-20 (Security to Resilience)
3. Streaming Server Guide → Production Patterns

**Weeks 9-10: 2025 Features**
4. Comprehensive Guide → Sections 15-17 (MCP)
5. Comprehensive Guide → Section 18 (A2A Protocol)
6. Comprehensive Guide → Section 19 (Agent Framework)
7. Recipes → Sections 17-20 (Advanced Patterns)

**Practice Projects:**
- Deploy production SK system to Kubernetes
- Implement MCP server and client
- Build cross-framework agent system with A2A
- Create enterprise monitoring dashboard

---

## By Use Case

### Document Processing & RAG
**Goal:** Upload documents, extract information, answer questions

**Read:**
1. Comprehensive Guide → Section 9-11 (Memory & Embeddings)
2. Comprehensive Guide → Section 10 (Vector Store v1.34)
3. Recipes → Section 10 (RAG Implementation)
4. Production Guide → Section 6 (Caching)

**Code Examples:**
- Recipes → Document QA System
- Recipes → Similarity Search
- Recipes → Semantic Caching

### Multi-Agent Collaboration
**Goal:** Multiple agents working together on complex tasks

**Read:**
1. Comprehensive Guide → Section 7-8 (Agents)
2. Advanced Multi-Agent Guide → Complete
3. Recipes → Sections 13-16 (Multi-Agent Examples)

**Code Examples:**
- Advanced Multi-Agent Guide → AgentGroupChat
- Recipes → Debate System
- Recipes → Research Assistant
- Recipes → Collaborative Problem Solving

### API Integration & Automation
**Goal:** Connect to external APIs, automate workflows

**Read:**
1. Comprehensive Guide → Section 6 (Plugins)
2. Recipes → Sections 5-8 (Plugin Examples)
3. Production Guide → Section 18-19 (Error Handling)

**Code Examples:**
- Recipes → HTTP Request Plugin
- Recipes → Database Plugin
- Recipes → OpenAPI Integration

### Streaming Chat Applications
**Goal:** Real-time chat with streaming responses

**Read:**
1. Streaming Server Guide → Complete
2. Production Guide → Section 8 (Async Optimization)
3. Recipes → Section 19 (Streaming Chat)

**Code Examples:**
- Streaming Server Guide → FastAPI SSE
- Recipes → WebSocket Chat
- Production Guide → Async Patterns

### Enterprise Deployment
**Goal:** Scalable, secure, production-ready deployment

**Read:**
1. Production Guide → Complete
2. Comprehensive Guide → Section 19 (Agent Framework)
3. Middleware Guide → Security & Guardrails

**Code Examples:**
- Production Guide → Kubernetes Deployment
- Production Guide → Azure Integration
- Middleware Guide → Policy Enforcement

---

## By Feature (2025)

### Model Context Protocol (MCP)

**What:** Standardized protocol for connecting AI applications to data sources and tools

**Documentation:**
- Comprehensive Guide → Section 15: MCP Overview
- Comprehensive Guide → Section 16: MCP Client (Python SDK)
- Comprehensive Guide → Section 17: MCP Server Creation
- Recipes → Section 20: MCP Integration Examples

**Key Topics:**
- MCP Client: Connect to MCP servers, access tools/resources
- MCP Server: Expose SK functions as MCP tools
- MCP Protocol: Understanding messages, requests, responses
- Integration: Using MCP with SK agents and planners

**Status:** Available in Python v1.34+ (November 2024)

### Vector Store v1.34 Overhaul

**What:** Unified, high-performance vector store API

**Documentation:**
- Comprehensive Guide → Section 10: Vector Store Architecture
- Recipes → Vector Store Migration Guide
- Production Guide → Section 6: Vector Store Caching

**Key Topics:**
- Unified API: Consistent interface across providers
- Performance: Faster embedding and retrieval
- New Connectors: Azure AI Search, Qdrant, Weaviate, Pinecone
- Migration: Upgrading from older vector store implementations

**Status:** Released in Python v1.34 (November 2024)

### Google A2A Protocol

**What:** Agent-to-Agent communication protocol for interoperability

**Documentation:**
- Comprehensive Guide → Section 18: A2A Protocol
- Recipes → A2A Integration Examples
- Advanced Multi-Agent Guide → A2A Routing

**Key Topics:**
- A2A Messages: Standard message format
- Agent Discovery: Finding and connecting to agents
- Routing: Intelligent message routing between agents
- Interoperability: SK agents talking to non-SK agents

**Status:** Integration available (experimental)

### Microsoft Agent Framework Integration

**What:** Unified SDK connecting SK with broader Microsoft agent ecosystem

**Documentation:**
- Comprehensive Guide → Section 19: Agent Framework
- Production Guide → Section 25: Enterprise Governance

**Key Topics:**
- Unified SDK: Common APIs across SK, AutoGen, Agent Framework
- Cross-Framework: SK agents working with AutoGen agents
- Enterprise Features: Governance, compliance, monitoring
- Deployment: Enterprise-scale agent orchestration

**Status:** Integration in progress (March 2025 for .NET, Python following)

---

## Quick Reference

### Common Commands

```bash
# Installation
pip install "semantic-kernel[openai,azure]" python-dotenv tenacity

# Run examples
python -m examples.basic_chat
python -m examples.multi_agent

# Run tests
pytest tests/

# Development
pip install -e ".[dev]"
pre-commit install
```

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://....openai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4

# Azure AI Search (Vector Store)
AZURE_SEARCH_ENDPOINT=https://....search.windows.net
AZURE_SEARCH_API_KEY=...

# Monitoring
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=...
```

### Key Imports

```python
# Kernel
from semantic_kernel import Kernel

# AI Services
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion

# Agents
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat

# Functions
from semantic_kernel.functions import kernel_function

# Memory
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchMemoryStore
```

### Common Patterns

```python
# Create kernel
kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(...))

# Create function
@kernel_function
async def my_function(input: str) -> str:
    return f"Processed: {input}"

# Create agent
agent = ChatCompletionAgent(
    kernel=kernel,
    name="assistant",
    instructions="You are a helpful assistant"
)

# Invoke
result = await kernel.invoke(function, input="Hello")
```

---

## Navigation

**[← Back to Python README](README.md)** | **[↑ Main Guide](../README.md)**

---

## Document Quick Links

- **[README.md](README.md)** - Start here
- **[semantic_kernel_comprehensive_python.md](semantic_kernel_comprehensive_python.md)** - Complete reference
- **[semantic_kernel_production_python.md](semantic_kernel_production_python.md)** - Production deployment
- **[semantic_kernel_recipes_python.md](semantic_kernel_recipes_python.md)** - Code examples
- **[semantic_kernel_diagrams_python.md](semantic_kernel_diagrams_python.md)** - Visual guides
- **[semantic_kernel_advanced_multi_agent_python.md](semantic_kernel_advanced_multi_agent_python.md)** - Multi-agent systems
- **[semantic_kernel_middleware_python.md](semantic_kernel_middleware_python.md)** - Middleware & guardrails
- **[semantic_kernel_streaming_server_fastapi.md](semantic_kernel_streaming_server_fastapi.md)** - Streaming server

---

**Last Updated:** November 2025
**Semantic Kernel Python Version:** 1.38.0+

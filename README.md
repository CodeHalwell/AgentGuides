# Agent Guides: Comprehensive Technical Documentation Collection

A comprehensive collection of technical guides for building production-ready AI agents with various frameworks and SDKs.

## 📚 Available Guides

### OpenAI Agents SDK Guides ⭐ NEW
Complete documentation for building multi-agent systems with the OpenAI Agents SDK.

**Location**: `OpenAI_Agents_SDK_Guides/`

**Contents**:
- **README.md** - Guide overview and quick start
- **openai_agents_sdk_comprehensive_guide.md** - Complete reference covering:
  - Core primitives (Agent, Runner, Handoff, Guardrail, Session)
  - Simple agents and multi-agent systems
  - Tools integration and structured outputs
  - Model Context Protocol (MCP) integration
  - Agentic patterns and guardrails
  - Memory systems and context engineering
  - Responses API integration and streaming
  - Tracing and observability
  - Real-time experiences and model providers

- **openai_agents_sdk_production_guide.md** - Production deployment covering:
  - Deployment architectures (monolithic, microservices, serverless)
  - Scalability and performance optimisation
  - Error handling and resilience patterns
  - Monitoring and observability
  - Security and safety
  - Cost optimisation
  - Testing strategies
  - CI/CD integration
  - Real-world examples

- **openai_agents_sdk_diagrams.md** - Visual architecture guides:
  - Agent lifecycle diagrams
  - Multi-agent interaction patterns
  - Message flow diagrams
  - Session management architecture
  - Tool integration patterns
  - Guardrail integration
  - MCP integration architecture
  - Production deployment topologies
  - Error handling and scalability patterns

- **openai_agents_sdk_recipes.md** - Practical implementations:
  - Customer service agents (airline, e-commerce)
  - Research and knowledge retrieval
  - Financial analysis systems
  - Code generation and review
  - Multi-language translation
  - Content moderation
  - Personal assistants
  - Team collaboration tools
  - Data analysis pipelines
  - Enterprise document processing

### CrewAI Guides
Framework for orchestrating multiple AI agents with specialised roles.

**Location**: `CrewAI_Guide/`

### AG2 (Formerly AutoGen) Guides
Framework for building multi-agent conversations with automatic message routing.

**Location**: `AG2_Guide/`

### LangGraph Guides
Graph-based agent orchestration framework for complex workflows.

**Location**: `LangGraph_Guide/`

### Haystack Guides
End-to-end NLP framework for building production search systems.

**Location**: `Haystack_Guide/`

### LlamaIndex Guides
Data framework for connecting LLMs with your data.

**Location**: `LlamaIndex_Guide/`

### PydanticAI Guides
Structured AI framework with type-safe agent development.

**Location**: `PydanticAI_Guide/`

### Amazon Bedrock Agents Guides
Build agents using Amazon's managed foundation models.

**Location**: `Amazon_Bedrock_Agents_Guide/`

### Microsoft Agent Framework Guides
Build agents using Microsoft's framework and services.

**Location**: `Microsoft_Agent_Framework_Guide/`

### Google ADK Guides
Google's Agent Development Kit for building intelligent agents.

**Location**: `Google_ADK_Guide/`

### SmolAgents Guides ⭐ NEW
Lightweight Python framework for building agents that think in code.

**Location**: `SmolAgents_Guide/`

**Contents**:
- **README.md** - Overview and quick start guide
- **smolagents_comprehensive_guide.md** - Complete reference covering:
  - Design philosophy (~1,000 lines of core code)
  - CodeAgent paradigm (write Python code instead of JSON)
  - ToolCallingAgent for traditional workflows
  - Tool creation (@tool decorator and Tool subclass)
  - Model configuration (100+ LLM providers)
  - Multi-agent orchestration patterns
  - Code execution sandboxing
  - Memory and state management
  - MCP integration
  - Advanced patterns and optimisations

- **smolagents_production_guide.md** - Production deployment covering:
  - Performance optimisation
  - Cost management and token budgeting
  - Monitoring and observability
  - Security best practices
  - Error handling and resilience
  - Deployment options (Docker, Kubernetes)
  - Scaling strategies
  - Testing and QA
  - W&B Weave integration

- **smolagents_diagrams.md** - Visual architecture guides:
  - Framework architecture
  - CodeAgent vs ToolCallingAgent comparison
  - Execution flows
  - Multi-agent orchestration
  - Tool integration patterns
  - Memory management
  - Deployment architectures

- **smolagents_recipes.md** - Practical implementations:
  - Data analysis agents
  - Web research agents
  - Business intelligence workflows
  - Content generation
  - Code review agents
  - Multi-agent pipelines
  - API integration patterns
  - Custom tool creation
  - Error handling patterns

## 🎯 Key Features

Each guide collection includes:

1. **Comprehensive Guide** - Complete reference with all features, concepts, and advanced patterns
2. **Production Guide** - Deployment strategies, scaling, monitoring, and operational patterns
3. **Diagrams Guide** - Visual architecture and flow diagrams for better understanding
4. **Recipes Guide** - Practical, copy-paste ready code examples for common use cases
5. **README** - Quick start and overview of the framework

## 🚀 Getting Started

### For OpenAI Agents SDK

Start with the quick introduction in `OpenAI_Agents_SDK_Guides/README.md`:

```bash
cd OpenAI_Agents_SDK_Guides
cat README.md
```

Then explore:
1. Comprehensive Guide for concepts and features
2. Recipes for practical implementations
3. Production Guide for deployment patterns
4. Diagrams for architecture understanding

### Installation

```bash
pip install openai-agents
export OPENAI_API_KEY=sk-your-key
```

### First Agent

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant"
    )
    
    result = await Runner.run(agent, "What is machine learning?")
    print(result.final_output)

asyncio.run(main())
```

## 📖 Documentation Structure

Each guide uses consistent formatting:

- **Clear Sections**: Organised by complexity and use case
- **Complete Examples**: Every concept includes working code
- **Best Practices**: Production-ready patterns throughout
- **Diagrams**: Visual representations of architectures
- **Cross-References**: Links between related topics

## 🔍 Finding What You Need

### By Use Case
- **Customer Service**: See OpenAI_Agents_SDK_Guides/openai_agents_sdk_recipes.md
- **Research/Knowledge**: LlamaIndex_Guide for RAG, OpenAI_Agents_SDK_Guides for multi-agent workflows
- **Code Generation**: PydanticAI_Guide, LangGraph_Guide
- **Financial Analysis**: OpenAI_Agents_SDK_Guides/openai_agents_sdk_recipes.md

### By Complexity
- **Beginner**: Start with README files and "Simple Agents" sections
- **Intermediate**: Read comprehensive guides and diagrams
- **Advanced**: Study production guides and advanced patterns

### By Framework
- **OpenAI**: OpenAI_Agents_SDK_Guides/
- **Multi-Agent Orchestration**: CrewAI_Guide/, AG2_Guide/
- **Graph-Based Workflows**: LangGraph_Guide/
- **Data Integration**: LlamaIndex_Guide/
- **Structured Outputs**: PydanticAI_Guide/

## 💾 File Organization

```
AgentGuides/
├── OpenAI_Agents_SDK_Guides/
│   ├── README.md
│   ├── openai_agents_sdk_comprehensive_guide.md
│   ├── openai_agents_sdk_production_guide.md
│   ├── openai_agents_sdk_diagrams.md
│   └── openai_agents_sdk_recipes.md
├── CrewAI_Guide/
├── AG2_Guide/
├── LangGraph_Guide/
├── Haystack_Guide/
├── LlamaIndex_Guide/
├── PydanticAI_Guide/
├── Amazon_Bedrock_Agents_Guide/
├── Microsoft_Agent_Framework_Guide/
├── Google_ADK_Guide/
└── README.md (this file)
```

## 🔑 Key Concepts Across Frameworks

### Agents
Autonomous entities with instructions, tools, and decision-making capabilities.

### Tools/Functions
Executable functions agents can call to interact with external systems.

### Handoffs/Delegation
Mechanisms for transferring control between specialised agents.

### Sessions/Memory
Conversation history and state management.

### Guardrails
Safety mechanisms for input/output validation.

### Tracing/Observability
Monitoring and debugging agent execution.

## 📝 British English Spelling

All documentation uses British English spelling conventions (e.g., 'optimisation' instead of 'optimization').

## 🤝 Contributing

To add new guides or update existing ones:
1. Follow the established guide structure
2. Include comprehensive examples
3. Add diagrams where helpful
4. Ensure production-ready code
5. Use British English spelling

## 📞 Support

For framework-specific questions:
- Review the official documentation linked in each guide
- Check the comprehensive guide's advanced topics
- Refer to recipes for similar implementations

## 📈 Framework Comparison

| Framework | Best For | Complexity | Community |
|-----------|----------|-----------|-----------|
| OpenAI Agents SDK | Multi-agent systems with lightweight primitives | Medium | Growing |
| CrewAI | Orchestrated role-based agents | Medium | Active |
| AG2 (AutoGen) | Research and experimentation | Medium-High | Large |
| LangGraph | Complex workflows and graphs | High | Growing |
| LlamaIndex | RAG and data indexing | Medium | Active |
| PydanticAI | Type-safe structured outputs | Low-Medium | Growing |

## 📅 Last Updated

- **OpenAI Agents SDK Guides**: November 2024
- **SmolAgents Guides**: November 2024
- **Other Frameworks**: Various dates (see individual guides)

## ⚠️ Version Notes

- All code examples are tested with recent framework versions
- Check individual READMEs for specific version requirements
- Update dependencies regularly for security and new features

---

This collection represents comprehensive, production-ready documentation for building AI agents across multiple frameworks. Each guide is designed to be both a learning resource and a reference for implementation.


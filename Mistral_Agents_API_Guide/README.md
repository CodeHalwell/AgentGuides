# Mistral Agents API Comprehensive Guide

## MAY 27, 2025 LAUNCH - AGENTS API NOW AVAILABLE

Mistral AI has officially launched the **Agents API** on May 27, 2025, bringing autonomous generative AI capabilities to production. This major platform release introduces powerful built-in connectors, multi-agent orchestration, and Model Context Protocol (MCP) support.

Welcome to the **Mistral Agents API Comprehensive Guide** – the most extensive, expert-level resource for building sophisticated AI agents using Mistral's platform. This guide covers everything from foundational concepts to advanced production deployments, with special emphasis on **built-in orchestration**, **persistent memory via Conversations API**, **5 powerful built-in connectors**, **MCP integration**, and **eliminating external framework dependencies**.

## 📚 Guide Structure

This comprehensive documentation suite consists of eight detailed documents covering the May 2025 launch features:

### 1. **mistral_agents_api_comprehensive_guide.md** (Core Knowledge)
The ultimate reference covering all aspects of the Mistral Agents API:
- **Core Fundamentals**: Platform setup, API keys, architecture, supported models, and configuration
- **Simple Agents**: Basic agent creation, instructions, single-task execution, synchronous calls
- **Multi-Agent Systems**: Built-in orchestration, coordination patterns, delegation mechanisms
- **Tools Integration**: Web search, code execution, image generation, document retrieval
- **Structured Output**: JSON schema, Pydantic integration, type enforcement
- **Model Context Protocol (MCP)**: Support, custom servers, tool exposure patterns
- **Agentic Patterns**: Multi-step execution, planning, reasoning, self-correction, ReAct implementations
- **Conversations API**: Persistent memory, multi-turn dialogues, state management
- **Memory Systems**: Built-in persistence, context preservation, retrieval strategies
- **Context Engineering**: Prompt design, few-shot patterns, dynamic construction
- **Deployment Patterns**: Direct platform deployment, scaling, cost management
- **Advanced Topics**: Streaming, rate limiting, monitoring, testing, security

### 2. **mistral_agents_api_diagrams.md** (Visual Architecture)
Comprehensive visual representations of:
- Agent lifecycle and deployment architecture
- Conversation flow and memory management
- Multi-agent orchestration patterns
- Tool execution workflows
- Request/response processing flows
- MCP integration patterns
- Memory persistence architecture

### 3. **mistral_agents_api_production_guide.md** (Enterprise Deployment)
Production-ready best practices including:
- Infrastructure setup and configuration
- Scaling strategies and load balancing
- Monitoring, logging, and observability
- Error handling and recovery
- Security hardening and compliance
- Cost optimisation strategies
- Performance tuning
- CI/CD integration
- Database design for conversation storage
- Rate limiting implementation

### 4. **mistral_agents_api_recipes.md** (Ready-to-Use Patterns)
Copy-paste ready code examples for:
- Building web search agents
- Multi-agent systems without external frameworks
- Creating custom tools
- Implementing RAG patterns
- Building chatbots with persistent memory
- Implementing hierarchical agent structures
- Creating specialised task agents
- Real-world application examples

### 5. **mistral_agents_api_connectors_guide.md** (NEW - May 2025)
Comprehensive guide to all 5 built-in connectors:
- **Python Code Execution**: Secure sandboxed Python environment
- **Image Generation**: Black Forest Lab FLUX1.1 [pro] Ultra integration
- **Web Search**: Standard + Premium (AFP, Associated Press) sources
- **Document Library/RAG**: Mistral Cloud document access and retrieval
- **Persistent Memory**: Server-side conversation state management
- Configuration, examples, and best practices for each connector

### 6. **mistral_agents_api_orchestration_guide.md** (NEW - May 2025)
Multi-agent collaboration and orchestration patterns:
- Sequential agent pipelines
- Parallel agent execution
- Hierarchical agent structures
- Complex problem-solving workflows
- Agent handoff mechanisms
- State management across agents

### 7. **mistral_agents_api_mcp_guide.md** (NEW - May 2025)
Model Context Protocol (MCP) integration:
- Anthropic MCP implementation details
- Standardized third-party tool connections
- Custom MCP server development
- Tool exposure patterns
- Best practices for MCP integration

### 8. **README.md** (This File)
Navigation and overview of the entire documentation suite.

---

## 🚀 Quick Start

### Installation

```bash
# Install the Mistral AI SDK with agents support
pip install "mistralai[agents]"
```

### Set Up Your API Key

```bash
# On Linux/Mac
export MISTRAL_API_KEY="your-api-key-here"

# On Windows PowerShell
$env:MISTRAL_API_KEY="your-api-key-here"
```

### Create Your First Agent (30 seconds)

```python
import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# Create a simple agent
agent = client.beta.agents.create(
    model="mistral-medium-2505",
    name="My First Agent",
    description="A helpful assistant"
)

# Start a conversation
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Hello! What can you help me with?"
)

print(conversation.outputs[-1].content)
```

---

## 🎯 Core Concepts at a Glance

### **Agents**
Persistent AI entities with pre-configured models, instructions, tools, and completion parameters. Agents maintain identity across conversations and can be versioned.

### **Conversations**
Stateful interaction threads with full history management. Each conversation maintains context and can be resumed, restarted, or branched. Conversations provide **persistent memory** across all interactions.

### **Tools**
Extended capabilities including web search, code execution, image generation, and document retrieval. Tools can be built-in or custom-defined.

### **Memory**
Built-in persistent storage via the Conversations API. Your entire conversation history is maintained server-side, enabling true long-term context retention.

### **Orchestration**
Native multi-agent coordination through handoff mechanisms and conversation management—no external frameworks required.

---

## 📖 Documentation Guide

### For **Beginners**
Start with:
1. Read "Core Fundamentals" in [mistral_agents_api_comprehensive_guide](mistral_agents_api_comprehensive_guide)
2. Review "Simple Agents" section
3. Follow "Quick Start" recipes in [mistral_agents_api_recipes](mistral_agents_api_recipes)

### For **Developers**
Focus on:
1. "Tools Integration" and "Structured Output" sections
2. "Agentic Patterns" for advanced workflows
3. Production guide for deployment considerations
4. Recipes for real-world implementations

### For **DevOps/SREs**
Prioritise:
1. [mistral_agents_api_production_guide](mistral_agents_api_production_guide) entirely
2. "Deployment Patterns" section
3. Monitoring and scaling sections
4. Security and compliance recommendations

### For **Architects**
Review:
1. System design diagrams in [mistral_agents_api_diagrams](mistral_agents_api_diagrams)
2. "Multi-Agent Systems" section
3. Advanced orchestration patterns
4. Production deployment strategies

---

## 🔑 Key Features

### ✨ **Built-in Orchestration**
- Multi-agent coordination without external frameworks
- Native agent handoff mechanisms
- Shared conversation context management
- Hierarchical agent structures

### 💾 **Persistent Memory**
- Server-side conversation history
- Full context retrieval and replay
- Conversation branching and restart capabilities
- Long-term context windows

### 🛠️ **5 Built-in Connectors** (May 2025 Launch)
- **Python Code Execution**: Secure sandboxed Python environment for autonomous code generation and execution
- **Image Generation**: Black Forest Lab FLUX1.1 [pro] Ultra for high-quality image creation
- **Web Search**: Standard search + Premium sources (AFP, Associated Press) for authoritative information
- **Document Library/RAG**: Native Mistral Cloud document access for knowledge retrieval
- **Persistent Memory**: Server-side conversation state management for long-term context
- **Custom Tools**: Define your own function-based tools
- **MCP Integration**: Anthropic Model Context Protocol for standardized third-party tool connections

### 📊 **Structured Output**
- JSON schema validation
- Pydantic model integration
- Type-safe responses
- Complex data structure support

### 🔄 **Streaming Support**
- Real-time response streaming
- Server-Sent Events (SSE) integration
- Progressive tool execution feedback
- Streamed conversation events

### 🛡️ **Enterprise Ready**
- API key authentication
- Rate limiting and quota management
- Error recovery and retry logic
- Production monitoring and logging

---

## 📋 Supported Models (May 2025)

| Model | Alias | Best For | Context Window | Use Case |
|-------|-------|----------|-----------------|----------|
| `mistral-medium-2505` | `mistral-medium-latest` | Balanced performance | Large | General purpose, web search, connectors |
| `mistral-large-latest` | Latest flagship | Complex reasoning | Extra large | Intricate logic, detailed analysis, orchestration |

## 📊 Performance Metrics (May 2025 Launch)

**SimpleQA Benchmark Results with Web Search Connector:**
- **Mistral Large**: 23% → **75%** (+52 percentage points improvement)
- **Mistral Medium**: 22.08% → **82.32%** (+60.24 percentage points improvement)

The Web Search connector dramatically improves factual accuracy and current information retrieval, making agents significantly more reliable for knowledge-intensive tasks.

---

## 🌐 Mistral Platform vs External Frameworks

### Why Use Mistral Agents API Natively?

**No External Dependencies**
- Built-in orchestration
- Managed persistence
- Native streaming
- Server-side state management

**Direct Deployment**
- Deploy on Mistral platform
- No orchestration layer needed
- Unified API surface
- Built-in monitoring

**Persistent Memory Out of the Box**
- Conversation history storage
- Context retrieval
- State management
- Multi-turn dialogue support

**Cost Effective**
- No extra infrastructure
- Managed service model
- Pay-per-token pricing
- Efficient resource usage

---

## 🎓 Learning Path

```
Start Here (Foundation)
    ↓
    ├─→ Core Fundamentals
    ├─→ Simple Agents
    └─→ Your First Agent (Quick Start Recipe)
         ↓
    Intermediate (Building Blocks)
    ├─→ Tools Integration
    ├─→ Structured Output
    └─→ Conversations API
         ↓
    Advanced (Production)
    ├─→ Multi-Agent Systems
    ├─→ Agentic Patterns
    ├─→ Production Deployment
    └─→ Enterprise Scaling
```

---

## 🔍 Navigation Tips

- **Search Within Docs**: Use Ctrl+F / Cmd+F for keyword search
- **Code Examples**: All sections include complete, runnable code
- **Copy-Paste Ready**: Recipes section contains production-ready snippets
- **Extensive Links**: Each section references related topics
- **Inline Explanations**: Every code example includes detailed comments

---

## 📞 Support Resources

- **Official Docs**: https://docs.mistral.ai/agents
- **API Reference**: https://docs.mistral.ai/api
- **SDK Repository**: https://github.com/mistralai/client-python
- **Community**: Mistral AI Discord and Forums

---

## 🎯 Key Takeaways

1. **Mistral Agents API provides everything you need for sophisticated agent development**
2. **No external frameworks required** – orchestration is built-in
3. **Persistent memory via Conversations API** enables true multi-turn intelligence
4. **Server-side state management** simplifies deployment and scaling
5. **Native tool integration** for web search, code execution, image generation, and more
6. **Production-ready** with streaming, error handling, and monitoring

---

## 🎉 What's New in May 2025 Launch

### Major Features
1. **Agents API Official Launch** (May 27, 2025)
   - Production-ready autonomous agent capabilities
   - Full platform integration

2. **5 Built-in Connectors**
   - Python Code Execution (secure sandboxed environment)
   - Image Generation (Black Forest Lab FLUX1.1 [pro] Ultra)
   - Web Search (Standard + Premium: AFP, Associated Press)
   - Document Library/RAG (Mistral Cloud integration)
   - Persistent Memory (server-side conversation state)

3. **Agent Orchestration**
   - Multi-agent collaboration frameworks
   - Sequential, parallel, and hierarchical patterns
   - Complex problem-solving workflows

4. **Model Context Protocol (MCP)**
   - Anthropic MCP implementation
   - Standardized third-party tool connections
   - Custom MCP server support

5. **Performance Improvements**
   - SimpleQA: Mistral Large 23% → 75% with web search
   - SimpleQA: Mistral Medium 22.08% → 82.32% with web search

## 📝 Document Versions

**Latest Version**: 2.0 (May 2025 Launch Edition)

All code examples tested against:
- Mistral AI Python SDK v1.9.11+
- Models: `mistral-medium-2505`, `mistral-large-latest`
- API Version: v1
- Connectors: All 5 built-in connectors
- MCP: Anthropic MCP integration

---

## 🎓 Table of Contents by Document

### Comprehensive Guide Contents
1. Core Fundamentals
2. Simple Agents
3. Multi-Agent Systems
4. Tools Integration
5. Structured Output
6. Model Context Protocol (MCP)
7. Agentic Patterns
8. Conversations API
9. Memory Systems
10. Context Engineering
11. Deployment Patterns
12. Web Search Tool
13. Code Execution
14. Image Generation
15. Document Retrieval
16. Advanced Topics

### Production Guide Contents
1. Infrastructure Setup
2. Scaling Strategies
3. Monitoring & Observability
4. Error Handling & Recovery
5. Security & Compliance
6. Cost Optimisation
7. Performance Tuning
8. CI/CD Integration
9. Database Design
10. Rate Limiting

### Diagrams Contents
1. Agent Lifecycle
2. Conversation Flow
3. Multi-Agent Orchestration
4. Tool Execution Workflow
5. Request/Response Processing
6. MCP Integration
7. Memory Persistence
8. Deployment Architecture

### Recipes Contents
1. Web Search Agent
2. Multi-Agent System
3. Custom Tools
4. RAG Implementation
5. Persistent Chatbot
6. Hierarchical Agents
7. Specialised Task Agents
8. Real-World Applications

### Connectors Guide Contents (NEW - May 2025)
1. Python Code Execution Connector
2. Image Generation Connector
3. Web Search Connector (Standard + Premium)
4. Document Library/RAG Connector
5. Persistent Memory Connector
6. Connector Best Practices
7. Security Considerations
8. Performance Optimization

### Orchestration Guide Contents (NEW - May 2025)
1. Sequential Agent Pipelines
2. Parallel Agent Execution
3. Hierarchical Agent Structures
4. Agent Handoff Mechanisms
5. State Management
6. Complex Workflows
7. Error Handling in Orchestration
8. Production Patterns

### MCP Guide Contents (NEW - May 2025)
1. MCP Architecture
2. Anthropic MCP Implementation
3. Custom MCP Servers
4. Tool Exposure Patterns
5. Integration Best Practices
6. Security and Authentication
7. Debugging MCP Connections
8. Production Deployment

---

**Happy building with Mistral Agents API! 🚀**


## Advanced Guides
- [mistral_agents_api_advanced_python.md](mistral_agents_api_advanced_python.md)
- [mistral_agents_api_connectors_guide.md](mistral_agents_api_connectors_guide.md) - NEW May 2025
- [mistral_agents_api_orchestration_guide.md](mistral_agents_api_orchestration_guide.md) - NEW May 2025
- [mistral_agents_api_mcp_guide.md](mistral_agents_api_mcp_guide.md) - NEW May 2025

## Streaming Examples
- [mistral_streaming_server_fastapi.md](mistral_streaming_server_fastapi.md)

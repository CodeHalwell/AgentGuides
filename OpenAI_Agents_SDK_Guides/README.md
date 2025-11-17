# OpenAI Agents SDK Complete Guide Collection (2025 Edition)

**🎯 PRODUCTION-READY | Official Swarm Replacement | Python 3.9+**

Welcome to the comprehensive guide collection for the OpenAI Agents SDK, the **official production-ready replacement** for the experimental Swarm framework. Build enterprise-grade multi-agent AI applications with confidence.

## 🚀 Critical 2025 Updates

### ⭐ **NEW: Swarm Migration Guide**
OpenAI has officially deprecated Swarm in favor of the Agents SDK. **[Complete Migration Guide →](openai_agents_sdk_swarm_migration_guide.md)**

### ⭐ **NEW: 2025 Features Guide**
- Built-in tracing and visualization
- Provider-agnostic support (100+ LLMs)
- Enhanced guardrails and session management
- MCP integration
- Fine-tuning and evaluation integration
**[2025 Features Guide →](openai_agents_sdk_2025_features.md)**

---

## 📚 Guide Documents

This collection contains comprehensive guides:

### 0. **[🆕 Swarm Migration Guide](openai_agents_sdk_swarm_migration_guide.md)** ⭐
**CRITICAL FOR SWARM USERS**: Complete migration path from experimental Swarm to production Agents SDK. Includes:
- Why migrate? (Production stability, active maintenance, enterprise features)
- Side-by-side code comparisons
- Breaking changes and solutions
- Step-by-step migration checklist
- Rollback strategy

### 0.5 **[🆕 2025 Features Guide](openai_agents_sdk_2025_features.md)** ⭐
Latest critical features and improvements:
- Built-in tracing and visualization
- Provider-agnostic support (100+ LLMs via LiteLLM)
- Enhanced guardrails system
- MCP integration (filesystem, git, HTTP)
- Evaluation and fine-tuning integration
- Production-ready session backends

### 1. **[Comprehensive Guide](openai_agents_sdk_comprehensive_guide.md)**
The complete reference covering all aspects of the OpenAI Agents SDK from fundamental concepts to advanced patterns. Includes:
- Core installation and setup procedures
- Fundamental concepts and design philosophy
- All primitive types and their use cases (Agents, Handoffs, Guardrails, Sessions, Tools, Runner)
- Simple and complex agent patterns
- Complete code examples for every feature
- Best practices and architectural considerations

### 2. **[Production Guide](openai_agents_sdk_production_guide)**
Practical patterns and strategies for deploying agents to production environments. Covers:
- Deployment architectures and patterns
- Scalability and performance optimisation
- Error handling and resilience strategies
- Monitoring, observability, and tracing
- Security and safety considerations
- Cost optimisation techniques
- Testing strategies and CI/CD integration
- Real-world deployment examples

### 3. **[Diagrams Guide](openai_agents_sdk_diagrams)**
Visual representations and architecture diagrams for common patterns. Includes:
- Agent lifecycle diagrams
- Multi-agent interaction patterns
- Message flow diagrams
- Session and memory management flows
- Tool and guardrail integration patterns
- MCP server integration architectures
- Production deployment topologies

### 4. **[Recipes Guide](openai_agents_sdk_recipes)**
Practical, ready-to-use code examples for common use cases. Features:
- Customer service agent implementations
- Research and knowledge retrieval agents
- Financial analysis workflows
- Code generation and review agents
- Multi-language translation services
- Content moderation and analysis systems
- Personal assistant implementations
- Team collaboration workflows

## 🚀 Quick Start

### Installation

```bash
pip install openai-agents
```

### Basic Agent

```python
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)

result = Runner.run_sync(agent, "What is 2 + 2?")
print(result.final_output)
```

## 🎯 Who Should Use This Guide?

- **Beginners**: Start with the Comprehensive Guide's fundamentals section
- **Developers**: Reference the Recipes Guide for common patterns
- **DevOps/SRE**: Review the Production Guide for deployment strategies
- **Architects**: Study the Diagrams Guide for system design patterns
- **Advanced Users**: Explore all guides for deep customisation opportunities

## 📖 Key Concepts (2025 Edition)

### Why Agents SDK Over Swarm?

| Feature | Swarm (❌ Deprecated) | Agents SDK (✅ Production) |
|---------|----------------------|---------------------------|
| **Status** | Experimental | Production-Ready |
| **Guardrails** | None | Built-in Input/Output |
| **Tracing** | Basic | Comprehensive + Visualization |
| **Sessions** | Manual | Automatic (SQLite/Redis/OpenAI) |
| **Providers** | OpenAI only | 100+ via LiteLLM |
| **MCP Support** | None | First-class |
| **Maintenance** | Deprecated | Active Development |

### Core Primitives (The Six Building Blocks)
1. **Agent**: An LLM configured with instructions, tools, and guardrails
2. **Handoff**: Mechanism for transferring control between specialised agents
3. **Tool**: Functions agents can call with automatic Pydantic schema generation
4. **Guardrail**: Input/output validation for safety and compliance
5. **Session**: Automatic conversation history management (multiple backends)
6. **Runner**: The orchestrator executing the agent loop

### Design Philosophy
The SDK emphasises:
- **Simplicity**: Minimal abstractions and intuitive APIs
- **Customisation**: Extensible architecture for specific needs
- **Production-Readiness**: Built-in support for tracing, sessions, and error handling
- **Model Agnosticity**: Support for OpenAI and 100+ other LLM providers via LiteLLM
- **Active Maintenance**: Regular updates, security patches, new features

## 🔗 Model Context Protocol (MCP)

The SDK includes first-class support for the Model Context Protocol, enabling:
- Integration with MCP servers (filesystem, git, web tools, etc.)
- Hosted MCP connectors with approval workflows
- Custom MCP server creation
- SSE streaming and secure remote execution

## 📊 Session Management

Choose from multiple session backends:
- **SQLite**: File-based storage (default, suitable for most applications)
- **Redis**: Distributed, high-performance session storage
- **SQLAlchemy**: Support for any database backend
- **OpenAI Backend**: Managed session storage via OpenAI's infrastructure

## 🛡️ Safety and Compliance

The SDK provides comprehensive guardrail support:
- **Input Guardrails**: Validate and filter user inputs before processing
- **Output Guardrails**: Validate and filter agent responses before returning
- **Custom Validation**: Create domain-specific safety checks
- **Integration**: Works seamlessly with Pydantic validation

## 📈 Observability

Built-in features include:
- **Tracing**: Visualise and debug agent flows
- **Structured Logging**: Comprehensive event tracking
- **Token Counting**: Monitor usage and costs
- **Integration**: Connect to Langfuse, AgentOps, Braintrust, and more

## 🔗 Additional Resources

- **Official GitHub**: https://github.com/openai/openai-agents-python
- **Examples Repository**: https://github.com/openai/openai-agents-python/tree/main/examples
- **OpenAI Documentation**: https://openai.github.io/openai-agents-python/
- **API Reference**: Available within the examples and documentation

## 📝 File Structure

Each guide is designed to be:
- **Self-contained**: Can be read independently
- **Cross-referenced**: Links between guides for related topics
- **Code-heavy**: Practical examples accompany every concept
- **Verbose**: Detailed explanations for comprehensive understanding

## 🎓 Learning Path

### For Beginners
1. Read the Quick Start section in this README
2. Review "Core Fundamentals" in the Comprehensive Guide
3. Try examples from the Recipes Guide
4. Understand Production considerations from the Production Guide

### For Experienced Developers
1. Skim the Comprehensive Guide's fundamentals
2. Deep-dive into advanced patterns and MCP integration
3. Review Production Guide for deployment strategies
4. Reference Recipes Guide for specific use cases

### For DevOps/SRE Personnel
1. Focus on the Production Guide entirely
2. Review deployment architectures in the Diagrams Guide
3. Understand monitoring and observability patterns
4. Plan scaling strategies and cost optimisation

## 💡 Key Features Summary

| Feature | Description |
|---------|-------------|
| **Lightweight Primitives** | Agent, Handoff, Guardrail, Session, Tool |
| **Multi-Agent Systems** | Handoff mechanisms for agent delegation |
| **Tool Integration** | Automatic schema generation with Pydantic |
| **Structured Outputs** | Pydantic-based output validation |
| **Session Memory** | Multiple backends (SQLite, Redis, SQLAlchemy) |
| **MCP Support** | First-class Model Context Protocol integration |
| **Streaming** | Token-level and item-level event streaming |
| **Guardrails** | Input and output validation with custom logic |
| **Tracing** | Built-in observability and debugging |
| **Model Agnostic** | Support for OpenAI and 100+ other providers |
| **Error Handling** | Comprehensive exception hierarchy |
| **Async/Await** | Full async support with sync alternatives |

## 📞 Support and Community

For questions, issues, or contributions:
- Open issues on [GitHub](https://github.com/openai/openai-agents-python)
- Check existing examples in the repository
- Review error messages and tracing output
- Consult the comprehensive guides for advanced patterns

---

## 🆕 2025 Critical Features

### 1. Built-in Tracing & Visualization
```python
from agents import Agent, Runner, trace

with trace("Customer Support Workflow", group_id="batch_001"):
    result = await Runner.run(agent, query)
# View at https://platform.openai.com/traces
```

### 2. Provider-Agnostic (100+ LLMs)
```python
# Use Claude, Gemini, Llama, Mistral, or any LiteLLM-supported model
agent = Agent(
    name="Claude Agent",
    model="litellm/anthropic/claude-3-5-sonnet-20240620"
)
```

### 3. Guardrails System
```python
@input_guardrail
async def safety_check(ctx, agent, input_data):
    # Validate inputs before processing
    return GuardrailFunctionOutput(...)
```

### 4. Enhanced Session Management
```python
from agents import SQLiteSession, RedisSession, OpenAIConversationsSession

# Choose backend: SQLite, Redis, SQLAlchemy, or OpenAI
session = SQLiteSession("user_123", "conversations.db")
```

### 5. MCP Integration
```python
from agents.mcp import MCPServerStdio

async with MCPServerStdio(name="Filesystem", params={...}) as server:
    agent = Agent(mcp_servers=[server])
```

---

**Note**: This guide collection focuses on the **Python** implementation of the OpenAI Agents SDK. For **JavaScript/TypeScript**, refer to the [TypeScript Guide Collection](../OpenAI_Agents_SDK_TypeScript_Guide/).

**Last Updated**: January 2025
**SDK Version**: Latest (v0.2.9+)
**Status**: Production-Ready | Official Swarm Replacement


## Streaming Examples
- [openai_agents_streaming_server_fastapi.md](openai_agents_streaming_server_fastapi.md)

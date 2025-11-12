# Microsoft Agent Framework - .NET Guide Collection
## October 2025 Release - Enterprise-Grade Documentation

**Release Date:** October 2025  
**Framework Status:** Unified SDK (unifying Semantic Kernel + AutoGen)  
**Platform:** .NET 8.0+  
**Latest Version:** 1.0+

---

## 📚 Documentation Structure

This comprehensive guide collection is designed to take you from beginner to expert in Microsoft Agent Framework for .NET. Each document serves a specific purpose:

### **1. [microsoft_agent_framework_dotnet_comprehensive_guide.md](./microsoft_agent_framework_dotnet_comprehensive_guide.md)**
**Complete conceptual reference** - ~15,000+ words

- **Core Fundamentals:** Architecture, installation, unified SDK, authentication
- **Simple Agents:** Basic creation, lifecycle, task execution patterns
- **Multi-Agent Systems:** Orchestration, communication, state management, scalability
- **Tools Integration:** Definition, registration, built-in Azure tools, custom creation
- **Structured Output:** Schema validation, type-safe responses, error handling
- **Model Context Protocol (MCP):** Server creation, tool standards, ecosystem integration
- **Agentic Patterns:** Planning, reasoning, autonomous decision-making, reflection
- **Memory Systems:** Unified API, persistent & vector memory, lifecycle management
- **Context Engineering:** Propagation, optimisation, multi-tenant isolation
- **Copilot Studio Integration:** Creation, publishing, analytics
- **Azure AI Integration:** Service configuration, optimisation strategies
- **Semantic Kernel Integration:** Plugin compatibility, migration

**Best For:** Learning concepts, understanding architecture, reference material

---

### **2. [microsoft_agent_framework_dotnet_diagrams.md](./microsoft_agent_framework_dotnet_diagrams.md)**
**Visual architecture reference** - System flows and topology diagrams

- **System Architecture:** Layered design, component relationships
- **Agent Lifecycle:** State machine, request/response flows
- **Multi-Agent Orchestration:** Sequential and branching workflows
- **Tool Integration:** Execution pipeline, provider architecture
- **Memory Systems:** Multi-tier architecture, access patterns
- **Azure Integration:** Service ecosystem, interconnections
- **Deployment:** Containerisation, multi-environment topology
- **Authentication & Security:** Flow diagrams, RBAC model
- **Data Flow:** Complete request-response cycle with error handling

**Best For:** Visual learners, architecture planning, system design

---

### **3. [microsoft_agent_framework_dotnet_production_guide.md](./microsoft_agent_framework_dotnet_production_guide.md)**
**Enterprise deployment & operations** - ~12,000+ words

- **Production Deployment:** Azure Container Apps, Kubernetes, CI/CD
- **Scaling Strategies:** Horizontal/vertical scaling, caching, rate limiting
- **Monitoring & Observability:** Application Insights, custom metrics, alerting
- **Security Best Practices:** Secrets management, network security, encryption
- **High Availability & Disaster Recovery:** Multi-region, backups, circuit breakers
- **Cost Optimisation:** Analysis framework, resource optimisation
- **Performance Tuning:** Connection pooling, batch processing, TPL patterns
- **Enterprise Governance:** Compliance, auditing, policy enforcement

**Best For:** DevOps engineers, infrastructure teams, production deployments

---

### **4. [microsoft_agent_framework_dotnet_recipes.md](./microsoft_agent_framework_dotnet_recipes.md)**
**Copy-paste ready code patterns** - ~5,000+ words

- **Beginner Recipes:** Simple chat agent, single tool, error handling
- **Intermediate Recipes:** Multi-agent workflow, multiple tools, memory persistence
- **Advanced Recipes:** RAG integration, complex orchestration
- **Integration Recipes:** Azure Functions, Logic Apps, event-driven patterns
- **Troubleshooting Patterns:** Debugging, monitoring, performance analysis

**Best For:** Developers building solutions, copy-paste code patterns, problem-solving

---

## 🚀 Quick Start Paths

### **Path 1: Beginner (Learning)**
1. Start with README (this file)
2. Read: [microsoft_agent_framework_dotnet_comprehensive_guide](microsoft_agent_framework_dotnet_comprehensive_guide) → Core Fundamentals + Simple Agents sections
3. View: [microsoft_agent_framework_dotnet_diagrams](microsoft_agent_framework_dotnet_diagrams) → System Architecture section
4. Code: [microsoft_agent_framework_dotnet_recipes](microsoft_agent_framework_dotnet_recipes) → Beginner Recipes
5. Practice: Build simple chat agent following Recipe 1

**Estimated Time:** 4-6 hours

### **Path 2: Intermediate (Building)**
1. Prerequisites: Complete Beginner path
2. Read: [microsoft_agent_framework_dotnet_comprehensive_guide](microsoft_agent_framework_dotnet_comprehensive_guide) → Multi-Agent + Tools sections
3. Code: [microsoft_agent_framework_dotnet_recipes](microsoft_agent_framework_dotnet_recipes) → Intermediate Recipes
4. Build: Multi-agent workflow following Recipe 4
5. View: [microsoft_agent_framework_dotnet_diagrams](microsoft_agent_framework_dotnet_diagrams) → Multi-Agent Orchestration

**Estimated Time:** 6-8 hours

### **Path 3: Advanced (Production)**
1. Prerequisites: Complete Intermediate path
2. Read: [microsoft_agent_framework_dotnet_production_guide](microsoft_agent_framework_dotnet_production_guide) → All sections
3. View: [microsoft_agent_framework_dotnet_diagrams](microsoft_agent_framework_dotnet_diagrams) → Deployment Architecture
4. Code: [microsoft_agent_framework_dotnet_recipes](microsoft_agent_framework_dotnet_recipes) → Advanced Recipes + Integration
5. Design: Production architecture for your use case

**Estimated Time:** 10-12 hours

### **Path 4: Quick Reference**
1. Need a specific feature? Use Table of Contents
2. Search across all documents for your scenario
3. Check [microsoft_agent_framework_dotnet_recipes](microsoft_agent_framework_dotnet_recipes) for code examples
4. Reference [microsoft_agent_framework_dotnet_diagrams](microsoft_agent_framework_dotnet_diagrams) for architecture

---

## 📋 Feature Matrix

| Feature | Beginner Guide | Comprehensive | Production | Recipes | Diagrams |
|---------|--------|---------------|-----------|---------|----------|
| **Concepts** | ✓ | ✓✓✓ | ✓ | ✓ | ✓ |
| **Code Examples** | Limited | Extensive | Infrastructure | ✓✓✓ | - |
| **Architecture** | Basic | Detailed | Enterprise | - | ✓✓✓ |
| **Deployment** | - | Overview | ✓✓✓ | - | ✓ |
| **Security** | - | Overview | ✓✓✓ | Patterns | ✓ |
| **Scaling** | - | Brief | ✓✓✓ | - | ✓ |

---

## 🔧 Installation & Setup

### **.NET Quick Start**

```bash
# Create console application
dotnet new console -n MyAgentApp
cd MyAgentApp

# Add packages
dotnet add package Microsoft.Agents.AI --prerelease
dotnet add package Azure.AI.OpenAI
dotnet add package Azure.Identity

# With Azure support
dotnet add package Microsoft.Agents.AI.Azure --prerelease

# Verify installation
dotnet build
```

### **System Requirements**

- .NET 8.0 SDK or later
- Visual Studio 2022 or Visual Studio Code
- Azure CLI for authentication (recommended)
- C# 12.0 language features

### **Environment Variables**

```bash
# .env file or environment variables
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_KEY=your-api-key
# Or use DefaultAzureCredential (no key needed)
```

### **Configuration File (appsettings.json)**

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

---

## 📖 Key Concepts at a Glance

### **Agent**
A conversational AI entity powered by LLMs, capable of understanding queries, making decisions, and using tools. Can be stateful (ChatAgent) or stateless (AIAgent).

### **Tool**
Functions that agents can invoke to perform specific tasks. Examples: API calls, database queries, custom functions. Defined using `AIFunctionFactory` or attributes.

### **Thread**
A conversation context that maintains history and state. Enables multi-turn interactions with agents.

### **Memory**
Persistent storage for agent state, conversation history, and knowledge. Multiple backends supported (Cosmos DB, SQL Server, Azure AI Search).

### **Orchestration**
Coordination of multiple agents to solve complex problems. Supports sequential, parallel, and conditional workflows.

### **Model Context Protocol (MCP)**
Standard interface for exposing tools and resources to agents, enabling integration across different systems.

---

## 🎯 Common Use Cases

### **Customer Support Agent**
- **Guide:** Comprehensive (Tools Integration) + Recipes (Recipe 4)
- **Deployment:** Production Guide (Azure Container Apps)
- **Components:** ChatAgent + multiple tools + persistent memory

### **Data Analysis Agent**
- **Guide:** Comprehensive (Agentic Patterns) + Recipes (RAG Recipe)
- **Deployment:** Production Guide (Batch Processing)
- **Components:** Multi-agent workflow + Azure Search

### **Autonomous Task Execution**
- **Guide:** Comprehensive (Multi-Agent Systems) + Production Guide (Orchestration)
- **Deployment:** Production Guide (Kubernetes)
- **Components:** Multi-agent orchestration + Circuit Breakers

### **Knowledge Assistant**
- **Guide:** Comprehensive (Memory Systems) + Recipes (RAG Recipe)
- **Deployment:** Production Guide (High Availability)
- **Components:** RAG + persistent memory + Azure AI Search

---

## 💡 Best Practices Summary

### **Development**
✓ Start with simple agents, add complexity gradually  
✓ Test agents independently before orchestrating  
✓ Use strong typing and attributes for tools  
✓ Implement comprehensive error handling  
✓ Leverage async/await patterns throughout  

### **Production**
✓ Use Azure Key Vault for secrets  
✓ Implement observability from day one  
✓ Design for horizontal scaling  
✓ Plan for multi-region deployment  
✓ Automate security scanning in CI/CD  

### **Operations**
✓ Monitor all agent executions  
✓ Track token usage and costs  
✓ Maintain audit trails  
✓ Plan regular disaster recovery tests  
✓ Implement gradual rollout strategies  

---

## 🔗 Reference Links

### **Official Microsoft Resources**
- [Microsoft Agent Framework GitHub](https://github.com/microsoft/agent-framework)
- [.NET Samples Repository](https://github.com/microsoft/Agent-Framework-Samples/tree/main/dotnet)
- [Microsoft Learn - Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
- [Azure AI Foundry Documentation](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
- [Microsoft 365 Agents SDK for .NET](https://github.com/microsoft/agents-for-net)

### **Related Technologies**
- [Semantic Kernel .NET](https://learn.microsoft.com/en-us/semantic-kernel/get-started/quick-start-dotnet)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

### **.NET-Specific Resources**
- [C# Async Programming](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/async/)
- [Dependency Injection in .NET](https://docs.microsoft.com/en-us/dotnet/core/extensions/dependency-injection)
- [.NET Configuration](https://docs.microsoft.com/en-us/dotnet/core/extensions/configuration)

---

## 📞 Support & Community

### **Getting Help**
1. **Check the Troubleshooting Patterns** in Recipes document
2. **Search GitHub Issues:** microsoft/agent-framework
3. **Review Microsoft Q&A:** `tag:agent-framework`
4. **Join Microsoft Learn Community**

### **Reporting Issues**
- **Bug Reports:** [GitHub Issues](https://github.com/microsoft/agent-framework/issues)
- **Feature Requests:** GitHub Discussions
- **Documentation Issues:** This repository

---

## 📄 Document Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 2025 | Initial .NET-focused documentation release |

---

## 🎓 Learning Outcomes

After completing all four documents, you'll be able to:

**Knowledge:**
- ✓ Understand Agent Framework architecture and design principles
- ✓ Explain different agent types and execution models
- ✓ Describe memory systems and context management
- ✓ Understand tool integration and MCP standards
- ✓ Master async/await patterns for agent systems
- ✓ Understand .NET dependency injection and configuration

**Skills:**
- ✓ Create and deploy agents in .NET/C#
- ✓ Build multi-agent orchestrated systems
- ✓ Integrate with Azure services and custom APIs
- ✓ Implement RAG and advanced patterns
- ✓ Monitor, secure, and scale agent applications
- ✓ Write production-ready async .NET code

**Capabilities:**
- ✓ Design enterprise-grade agent applications
- ✓ Deploy to production with high availability
- ✓ Optimise for cost and performance
- ✓ Implement compliance and governance requirements
- ✓ Troubleshoot and debug agent systems

---

## 🏆 Expert Tips

1. **Start Small, Scale Fast:** Begin with simple single-purpose agents before multi-agent orchestration
2. **Embrace Async:** Leverage .NET's Task-based async throughout for better performance
3. **Monitor from Day One:** Implement comprehensive observability early—debugging in production is expensive
4. **Test Tool Calls:** Always test tools independently before integrating with agents
5. **Plan for Failure:** Implement circuit breakers, retries, and graceful degradation
6. **Use Strong Typing:** Leverage C#'s type system for compile-time safety
7. **Security First:** Never hardcode secrets; use Key Vault from the start
8. **Cost Awareness:** Monitor token usage and implement caching strategies early
9. **Dependency Injection:** Use DI containers for better testability and lifecycle management
10. **Configuration:** Separate configuration from code using appsettings.json and environment variables

---

## 📞 Questions?

**Refer to the appropriate document:**
- **"What is...?" or "How does...?"** → Comprehensive Guide
- **"What does architecture look like?"** → Diagrams
- **"How do I deploy to production?"** → Production Guide
- **"Show me code..."** → Recipes
- **"Where do I start?"** → This README

---

## 🔐 Security & Compliance

All code examples in this documentation follow security best practices:
- ✓ No hardcoded secrets or API keys
- ✓ Use Azure Identity for authentication
- ✓ Implement RBAC and least privilege
- ✓ Encrypt sensitive data
- ✓ Audit all operations

---

## 🎯 .NET-Specific Considerations

### **Async Best Practices**
- Use `async`/`await` throughout
- Avoid `.Result` and `.Wait()` - prefer async all the way
- Use `Task.WhenAll()` for concurrent operations
- Implement proper cancellation with `CancellationToken`

### **Type Safety**
- Use record types for immutable data
- Leverage nullable reference types
- Use attributes for metadata (`[Description]`, etc.)

### **Dependency Injection**
- Register services with appropriate lifetimes
- Use `IOptions<T>` for configuration
- Implement proper disposal patterns

### **Testing**
- Use xUnit or NUnit with Moq
- Mock Azure services with `Azure.Core.TestFramework`
- Implement integration tests separately

---

## 📜 License

These documentation materials are provided as-is for educational and reference purposes. Refer to Microsoft's official documentation and license terms for production use.

---

**Last Updated:** November 2025  
**Maintained By:** AI Documentation Team  
**Status:** Actively Maintained  
**Next Review:** Q2 2026

---

## 🚀 Ready to Get Started?

Choose your path above and dive in! Start with the [Comprehensive Guide](./microsoft_agent_framework_dotnet_comprehensive_guide.md) if this is your first time.

**Happy building! ⚡🎯**


## Advanced Guides
- [microsoft_agent_framework_dotnet_advanced.md](microsoft_agent_framework_dotnet_advanced.md)

## Streaming Examples
- [microsoft_agent_streaming_server_dotnet.md](microsoft_agent_streaming_server_dotnet.md)

# Microsoft Agent Framework - Complete Guide Collection
## April 2026 - GA 1.0 Release

**GA Release Date:** April 3–7, 2026 (previously Preview since October 2025)
**Framework Status:** ✅ Production-Ready — GA 1.0 with Long-Term Support
**Supported Platforms:** .NET, Python
**Latest Version:** 1.0 GA

> **Status Update:** Microsoft Agent Framework has graduated from Public Preview to **GA 1.0** as of April 3–7, 2026. The `--prerelease` flag is no longer needed; install with `pip install agent-framework`.

## ⚠️ Migration Notes from Preview to GA 1.0

- **`ChatClientAgentOptions.Instructions` removed** — pass instructions directly to the `ChatClientAgent` constructor.
- **Dependency version floor changed** — packages now require `>=1.0.0,<2`; RC installs are incompatible.
- **Package name changes** — use `agent-framework`, `agent-framework-core`, `agent-framework-openai` (not the preview names).
- **Migration from standalone Semantic Kernel or AutoGen** — see the [official migration guide](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-semantic-kernel/). Microsoft estimates 2–4 hours for a typical app.

---

## 🌟 NEW in April 2026 - GA 1.0 Features

### **Production-Ready with Stable APIs and LTS Commitment**
Agent Framework 1.0 is now ready for production with stable APIs across both .NET and Python.

### **First-Party Service Connectors**
Swap LLM providers with a single line via the `IChatClient` interface:
- Azure OpenAI, OpenAI, Anthropic Claude, Amazon Bedrock, Google Gemini, Ollama

### **Unified Orchestration**
Multi-agent patterns built-in: sequential, concurrent, handoff, group chat, and Magentic-One.

### **Azure App Service Support**
Deploy multi-agent apps to Azure App Service with native integration.

---

## 🌟 Features Added in October 2025 Preview

### **Agent2Agent (A2A) Protocol** 🔥
Enable seamless collaboration between agents across different frameworks (OpenAI SDK, Claude SDK, LangGraph, Google ADK). Built-in authentication, cross-framework messaging, and remote agent communication.

👉 **[Complete A2A Protocol Guide](./microsoft_agent_framework_a2a_protocol.md)**

### **Graph-Based Workflows** 🔥
Build complex agent orchestration with directed graphs. Features include streaming support, checkpointing, human-in-the-loop integration, and time-travel debugging for workflow analysis.

👉 **[Graph-Based Workflows & Declarative Definitions Guide](./microsoft_agent_framework_graphs_declarative.md)**

### **Declarative Agent Definitions** 🔥
Configure agents using YAML/JSON for version control and reusability. Supports templates, validation, and environment-based configuration.

👉 **[Graph-Based Workflows & Declarative Definitions Guide](./microsoft_agent_framework_graphs_declarative.md)**

### **Enhanced Enterprise Features** 🔥
- **OpenTelemetry Instrumentation:** Full distributed tracing, metrics, and logging
- **Azure AI Content Safety:** Real-time harmful content detection and blocking
- **Entra ID Authentication:** Managed Identity, Workload Identity for Kubernetes
- **Compliance:** HIPAA, GDPR, SOX, PCI-DSS, FedRAMP configurations

👉 **[Enterprise Features 2025 Guide](./microsoft_agent_framework_enterprise_2025.md)**

### **GA Status**
- **Current Status:** GA 1.0 — Production-Ready (April 3–7, 2026)
- **Production Readiness:** Stable APIs, LTS commitment
- **Migration Support:** Comprehensive guides from Semantic Kernel and AutoGen at [learn.microsoft.com](https://learn.microsoft.com/en-us/agent-framework/migration-guide/)

---

## 📚 Documentation Structure

This comprehensive guide collection is designed to take you from beginner to expert in Microsoft Agent Framework. Each document serves a specific purpose:

### **1. [microsoft_agent_framework_comprehensive_guide.md](./microsoft_agent_framework_comprehensive_guide.md)**
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

### **2. Language-Specific Guides**

#### **[.NET Guide](./dotnet/README.md)**
Comprehensive documentation for C#/.NET developers, including installation, configuration, and advanced patterns.

#### **[Python Guide](./python/README.md)**
Complete guide for Python developers, covering installation, async patterns, and data science integration.

---

### **3. [microsoft_agent_framework_diagrams.md](./microsoft_agent_framework_diagrams.md)**
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

### **4. [microsoft_agent_framework_production_guide.md](./microsoft_agent_framework_production_guide.md)**
**Enterprise deployment & operations** - ~12,000+ words

- **Production Deployment:** Azure Container Apps, Kubernetes, CI/CD
- **Scaling Strategies:** Horizontal/vertical scaling, caching, rate limiting
- **Monitoring & Observability:** Application Insights, custom metrics, alerting
- **Security Best Practices:** Secrets management, network security, encryption
- **High Availability & Disaster Recovery:** Multi-region, backups, circuit breakers
- **Cost Optimisation:** Analysis framework, resource optimisation
- **Performance Tuning:** Connection pooling, batch processing
- **Enterprise Governance:** Compliance, auditing, policy enforcement

**Best For:** DevOps engineers, infrastructure teams, production deployments

---

### **5. [microsoft_agent_framework_recipes.md](./microsoft_agent_framework_recipes.md)**
**Copy-paste ready code patterns** - ~5,000+ words

- **Beginner Recipes:** Simple chat agent, single tool, error handling
- **Intermediate Recipes:** Multi-agent workflow, multiple tools, memory persistence
- **Advanced Recipes:** RAG integration, complex orchestration
- **Integration Recipes:** Azure Functions, Logic Apps, event-driven patterns
- **Troubleshooting Patterns:** Debugging, monitoring, performance analysis

**Best For:** Developers building solutions, copy-paste code patterns, problem-solving

---

### **6. [microsoft_agent_framework_a2a_protocol.md](./microsoft_agent_framework_a2a_protocol.md)** 🆕
**Agent2Agent Protocol specification** - ~8,000+ words

- **Core Concepts:** Message structure, protocol components, framework registration
- **Cross-Framework Interoperability:** OpenAI SDK, Claude SDK, LangGraph, Google ADK
- **Authentication:** OAuth2, Entra ID, mTLS, API keys
- **Message Formats:** Request/response, events, errors
- **Implementation Examples:** Complete Python and .NET implementations
- **Security:** End-to-end encryption, message signing, rate limiting
- **Production Patterns:** Circuit breakers, message queues, monitoring

**Best For:** Building cross-framework agent systems, enterprise integration

---

### **7. [microsoft_agent_framework_graphs_declarative.md](./microsoft_agent_framework_graphs_declarative.md)** 🆕
**Graph workflows and declarative configuration** - ~10,000+ words

- **Graph-Based Workflows:** Directed graphs, conditional routing, parallel execution
- **Declarative Definitions:** YAML/JSON configuration, templates, validation
- **Streaming:** Real-time output streaming, token-by-token responses
- **Checkpointing:** Save/resume workflows, failure recovery
- **Human-in-the-Loop:** Approval workflows, interactive decision points
- **Time-Travel Debugging:** Navigate execution history, replay workflows
- **Production Patterns:** Error handling, performance optimization

**Best For:** Complex orchestration, configuration-driven development, debugging

---

### **8. [microsoft_agent_framework_enterprise_2025.md](./microsoft_agent_framework_enterprise_2025.md)** 🆕
**Enterprise features and compliance** - ~9,000+ words

- **OpenTelemetry Instrumentation:** Distributed tracing, metrics, custom spans
- **Azure AI Content Safety:** Harmful content detection, PII redaction
- **Entra ID Authentication:** Managed Identity, Workload Identity, multi-tenant
- **Compliance:** HIPAA, GDPR, SOX, PCI-DSS, FedRAMP configurations
- **Enterprise Governance:** Policy enforcement, cost management, model governance
- **Advanced Security:** End-to-end encryption, network security, zero trust
- **Audit Logging:** Comprehensive auditing, SIEM integration

**Best For:** Enterprise deployments, regulated industries, security teams

---

## 🚀 Quick Start Paths

### **Path 1: Beginner (Learning)**
1. Start with README (this file)
2. Read: [microsoft_agent_framework_comprehensive_guide](microsoft_agent_framework_comprehensive_guide) → Core Fundamentals + Simple Agents sections
3. View: [microsoft_agent_framework_diagrams](microsoft_agent_framework_diagrams) → System Architecture section
4. Code: [microsoft_agent_framework_recipes](microsoft_agent_framework_recipes) → Beginner Recipes
5. Practice: Build simple chat agent following Recipe 1

**Estimated Time:** 4-6 hours

### **Path 2: Intermediate (Building)**
1. Prerequisites: Complete Beginner path
2. Read: [microsoft_agent_framework_comprehensive_guide](microsoft_agent_framework_comprehensive_guide) → Multi-Agent + Tools sections
3. Code: [microsoft_agent_framework_recipes](microsoft_agent_framework_recipes) → Intermediate Recipes
4. Build: Multi-agent workflow following Recipe 4
5. View: [microsoft_agent_framework_diagrams](microsoft_agent_framework_diagrams) → Multi-Agent Orchestration

**Estimated Time:** 6-8 hours

### **Path 3: Advanced (Production)**
1. Prerequisites: Complete Intermediate path
2. Read: [microsoft_agent_framework_production_guide](microsoft_agent_framework_production_guide) → All sections
3. View: [microsoft_agent_framework_diagrams](microsoft_agent_framework_diagrams) → Deployment Architecture
4. Code: [microsoft_agent_framework_recipes](microsoft_agent_framework_recipes) → Advanced Recipes + Integration
5. Design: Production architecture for your use case

**Estimated Time:** 10-12 hours

### **Path 4: Quick Reference**
1. Need a specific feature? Use Table of Contents
2. Search across all documents for your scenario
3. Check [microsoft_agent_framework_recipes](microsoft_agent_framework_recipes) for code examples
4. Reference [microsoft_agent_framework_diagrams](microsoft_agent_framework_diagrams) for architecture

---

## 📋 Feature Matrix

| Feature | Comprehensive | Production | Recipes | Diagrams | A2A Protocol | Graphs/Declarative | Enterprise 2025 |
|---------|---------------|-----------|---------|----------|--------------|-------------------|----------------|
| **Core Concepts** | ✓✓✓ | ✓ | ✓ | ✓ | ✓✓ | ✓✓✓ | ✓ |
| **Code Examples** | ✓✓✓ | ✓✓ | ✓✓✓ | - | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| **Architecture** | ✓✓✓ | ✓✓✓ | - | ✓✓✓ | ✓✓ | ✓✓✓ | ✓✓ |
| **Deployment** | ✓ | ✓✓✓ | - | ✓ | - | - | ✓ |
| **Security** | ✓ | ✓✓✓ | ✓ | ✓ | ✓✓✓ | ✓ | ✓✓✓ |
| **Cross-Framework** | - | - | - | - | ✓✓✓ | - | - |
| **Orchestration** | ✓✓ | ✓ | ✓✓ | ✓✓ | - | ✓✓✓ | - |
| **Compliance** | - | ✓✓ | - | - | - | - | ✓✓✓ |
| **Observability** | ✓ | ✓✓✓ | ✓ | - | ✓ | ✓✓ | ✓✓✓ |

---

## 🔧 Installation & Setup

### **Python Quick Start**

```bash
# Create virtual environment
python -m venv agent_env
source agent_env/bin/activate  # On Windows: agent_env\Scripts\activate

# Install Agent Framework (GA - no --pre flag needed)
pip install agent-framework

# With Azure integration
pip install agent-framework-azure-ai

# Verify installation
python -c "import agent_framework; print('✓ Framework installed')"
```

### **.NET Quick Start**

```bash
# Create console application
dotnet new console -n MyAgentApp
cd MyAgentApp

# Add packages
dotnet add package Microsoft.Agents.AI
dotnet add package Azure.AI.OpenAI
dotnet add package Azure.Identity
```



---

## 📖 Key Concepts at a Glance

### **Agent**
A conversational AI entity powered by LLMs, capable of understanding queries, making decisions, and using tools. Can be stateful (ChatAgent) or stateless (AIAgent).

### **Tool**
Functions that agents can invoke to perform specific tasks. Examples: API calls, database queries, custom functions.

### **Thread**
A conversation context that maintains history and state. Enables multi-turn interactions with agents.

### **Memory**
Persistent storage for agent state, conversation history, and knowledge. Multiple backends supported (Cosmos DB, SQL, Azure AI Search).

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
✓ Use type hints and descriptive tool names  
✓ Implement comprehensive error handling  

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
- [Microsoft Learn - Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
- [Azure AI Foundry Documentation](https://learn.microsoft.com/en-us/azure/ai-services/agents/)

### **Related Technologies**
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)

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
| 1.0 | Nov 2025 | Initial comprehensive documentation release |

---

## 🎓 Learning Outcomes

After completing all four documents, you'll be able to:

**Knowledge:**
- ✓ Understand Agent Framework architecture and design principles
- ✓ Explain different agent types and execution models
- ✓ Describe memory systems and context management
- ✓ Understand tool integration and MCP standards

**Skills:**
- ✓ Create and deploy agents in Python, .NET
- ✓ Build multi-agent orchestrated systems
- ✓ Integrate with Azure services and custom APIs
- ✓ Implement RAG and advanced patterns
- ✓ Monitor, secure, and scale agent applications

**Capabilities:**
- ✓ Design enterprise-grade agent applications
- ✓ Deploy to production with high availability
- ✓ Optimise for cost and performance
- ✓ Implement compliance and governance requirements
- ✓ Troubleshoot and debug agent systems

---

## 🏆 Expert Tips

1. **Start Small, Scale Fast:** Begin with simple single-purpose agents before multi-agent orchestration
2. **Monitor from Day One:** Implement comprehensive observability early—debugging in production is expensive
3. **Test Tool Calls:** Always test tools independently before integrating with agents
4. **Plan for Failure:** Implement circuit breakers, retries, and graceful degradation
5. **Optimise Incrementally:** Profile before optimising; measure impact of each change
6. **Security First:** Never hardcode secrets; use Key Vault from the start
7. **Document Decisions:** Record why you chose specific patterns for future reference
8. **Cost Awareness:** Monitor token usage and implement caching strategies early

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

## 📜 License

These documentation materials are provided as-is for educational and reference purposes. Refer to Microsoft's official documentation and license terms for production use.

---

**Last Updated:** April 16, 2026  
**Maintained By:** AI Documentation Team  
**Status:** Actively Maintained — GA 1.0  
**Next Review:** Q3 2026

---

## 🚀 Ready to Get Started?

Choose your path above and dive in! Start with the [Comprehensive Guide](./microsoft_agent_framework_comprehensive_guide.md) if this is your first time.

**Happy building! 🎯**

---

## 📋 Revision History

| Date | Version | Changes |
|------|---------|---------|
| April 16, 2026 | 1.0 GA | Updated to GA 1.0; `ChatClientAgentOptions.Instructions` removed; `--prerelease` flags removed; first-party connectors (Azure OpenAI, OpenAI, Anthropic, Bedrock, Gemini, Ollama); Azure App Service support |
| November 2025 | 1.0 Preview | Initial guide; preview release; multi-agent orchestration; .NET and Python SDKs |

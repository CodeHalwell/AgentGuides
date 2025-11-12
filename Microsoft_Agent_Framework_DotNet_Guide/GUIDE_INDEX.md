# Microsoft Agent Framework .NET Guide - Complete Index
## Your Navigation Hub for Learning Agent Development in C# and .NET

**Last Updated:** November 2025  
**Guide Version:** 1.0  
**.NET Version:** 8.0+

---

## 🎯 Quick Navigation

### **For Beginners: Start Here**
1. Read **[README.md](README.md)** - Overview and setup (10 min)
2. Study **[Comprehensive Guide - Introduction & Core Fundamentals](microsoft_agent_framework_dotnet_comprehensive_guide.md#introduction)** (1-2 hours)
3. Try **[Recipes - Beginner Section](microsoft_agent_framework_dotnet_recipes.md#beginner-recipes)** - Copy-paste examples (30 min)

### **For Intermediate Developers**
1. Review **[Comprehensive Guide - Multi-Agent Systems](microsoft_agent_framework_dotnet_comprehensive_guide.md#multi-agent-systems)** (2-3 hours)
2. Explore **[Diagrams - Multi-Agent Orchestration](microsoft_agent_framework_dotnet_diagrams.md#multi-agent-orchestration)** (30 min)
3. Implement **[Recipes - Intermediate Section](microsoft_agent_framework_dotnet_recipes.md#intermediate-recipes)** (2-4 hours)

### **For Production Deployment**
1. Study **[Production Guide - Complete](microsoft_agent_framework_dotnet_production_guide.md)** (4-6 hours)
2. Review **[Diagrams - Deployment Architecture](microsoft_agent_framework_dotnet_diagrams.md#deployment-architecture)** (1 hour)
3. Apply **[Recipes - Integration Section](microsoft_agent_framework_dotnet_recipes.md#integration-recipes)** (varies)

---

## 📚 Document Catalog

### **1. README.md** (Quick Start)
**Purpose:** Entry point, installation, quick reference  
**Length:** ~300 lines  
**Time to Read:** 10-15 minutes  
**Best For:** Getting started, installation, overview

**Key Sections:**
- Installation & Setup
- Quick Start Paths (Beginner, Intermediate, Advanced)
- Key Concepts at a Glance
- .NET-Specific Considerations

**When to Use:**
- ✓ First time with Agent Framework
- ✓ Need installation instructions
- ✓ Want learning path guidance
- ✓ Quick reference for setup

---

### **2. microsoft_agent_framework_dotnet_comprehensive_guide.md**
**Purpose:** Complete conceptual reference and learning resource  
**Length:** ~12,000+ lines (~15,000 words)  
**Time to Read:** 8-12 hours (study multiple sessions)  
**Best For:** Deep learning, understanding concepts, reference material

**Key Sections:**
1. **Introduction** (Lines 1-50)
   - Framework overview
   - Key objectives
   - Target use cases

2. **Core Fundamentals** (Lines 51-400)
   - Architecture principles
   - Installation
   - Authentication
   - Environment setup
   - Dependency injection

3. **Simple Agents** (Lines 401-900)
   - Creating basic agents
   - AIAgent vs ChatAgent
   - Lifecycle management
   - Task execution models
   - Testing strategies

4. **Multi-Agent Systems** (Lines 901-1400)
   - Orchestration patterns
   - Communication patterns
   - Shared state management
   - Workflow coordination
   - Scalability architecture

5. **Tools Integration** (Lines 1401-1900)
   - Tool definition with `AIFunctionFactory`
   - Attributes and metadata
   - Built-in Azure tools
   - Custom tool creation
   - Error handling

6. **Structured Output** (Lines 1901-2200)
   - Record types
   - Schema validation
   - Type-safe responses

7. **Model Context Protocol (MCP)** (Lines 2201-2500)
   - MCP server implementation
   - Tool standards
   - Resource management

8. **Agentic Patterns** (Lines 2501-2900)
   - Planning and reasoning
   - Autonomous decision-making
   - Reflection patterns

9. **Memory Systems** (Lines 2901-3300)
   - Memory backends
   - Persistent storage
   - Vector memory
   - SQL Server integration
   - Cosmos DB integration

10. **Context Engineering** (Lines 3301-3600)
    - Context propagation
    - AsyncLocal patterns
    - Multi-tenant isolation

11. **Copilot Studio Integration** (Lines 3601-3800)
    - .NET SDK integration
    - Agent publishing

12. **Azure AI Integration** (Lines 3801-4000)
    - Azure services
    - Cost optimization
    - Service configuration

13. **Semantic Kernel Integration** (Lines 4001-4200)
    - Plugin compatibility
    - Migration strategies
    - Shared patterns

**When to Use:**
- ✓ Learning agent concepts
- ✓ Understanding architecture
- ✓ Need detailed explanations
- ✓ Reference material
- ✓ Teaching others

---

### **3. microsoft_agent_framework_dotnet_recipes.md**
**Purpose:** Copy-paste ready C# code examples for common scenarios  
**Length:** ~5,000+ lines  
**Time to Use:** Varies by recipe (15min - 2 hours each)  
**Best For:** Practical implementation, learning by example, quick solutions

**Recipe Categories:**

#### **Beginner Recipes** (3 recipes)
- Recipe 1: Simple Chat Agent (C#)
- Recipe 2: Agent with Single Tool
- Recipe 3: Basic Error Handling

**Use When:**
- Starting with Agent Framework
- Need basic examples
- Learning fundamentals

#### **Intermediate Recipes** (4 recipes)
- Recipe 4: Multi-Agent Workflow
- Recipe 5: Agent with Multiple Tools
- Recipe 6: Memory Integration
- Recipe 7: Streaming Responses

**Use When:**
- Building production features
- Coordinating multiple agents
- Implementing persistence

#### **Advanced Recipes** (3 recipes)
- Recipe 8: RAG (Retrieval-Augmented Generation)
- Recipe 9: Complex Multi-Agent Orchestration
- Recipe 10: Custom Tool Development

**Use When:**
- Building advanced features
- Need RAG capabilities
- Creating custom integrations

#### **Integration Recipes** (4 recipes)
- Recipe 11: Azure Functions Integration
- Recipe 12: Event-Driven Architecture
- Recipe 13: ASP.NET Core Integration
- Recipe 14: Azure Logic Apps Integration

**Use When:**
- Integrating with Azure services
- Building serverless solutions
- Creating APIs

#### **Troubleshooting Patterns** (5 patterns)
- Pattern 1: Debugging Agent Execution
- Pattern 2: Performance Monitoring
- Pattern 3: Error Recovery
- Pattern 4: Token Usage Optimization
- Pattern 5: Load Testing

**Use When:**
- Debugging issues
- Optimizing performance
- Monitoring production

---

### **4. microsoft_agent_framework_dotnet_production_guide.md**
**Purpose:** Enterprise deployment, scaling, security, operations  
**Length:** ~12,000+ lines  
**Time to Read:** 6-10 hours  
**Best For:** DevOps, infrastructure, production deployment

**Key Sections:**
1. **Production Deployment**
   - Azure Container Apps
   - Docker containers
   - Kubernetes
   - CI/CD with Azure DevOps

2. **Scaling Strategies**
   - Horizontal scaling
   - Vertical scaling
   - Caching strategies
   - Rate limiting

3. **Monitoring & Observability**
   - Application Insights
   - Custom metrics
   - Logging with ILogger
   - Tracing
   - Alerting

4. **Security Best Practices**
   - Secrets management
   - Network security
   - Encryption
   - Authentication
   - Authorization

5. **High Availability & Disaster Recovery**
   - Multi-region deployment
   - Backup strategies
   - Circuit breakers (Polly)
   - Failover procedures

6. **Cost Optimization**
   - Resource optimization
   - Token usage management
   - Caching strategies
   - Reserved instances

7. **Performance Tuning**
   - TPL optimization
   - Connection pooling
   - Batch processing
   - Memory management

8. **Enterprise Governance**
   - Compliance
   - Auditing
   - Policy enforcement
   - Data governance

---

### **5. microsoft_agent_framework_dotnet_diagrams.md**
**Purpose:** Visual architecture reference and system design  
**Length:** ~1,000+ lines  
**Time to Review:** 1-2 hours  
**Best For:** Visual learners, architecture planning, presentations

**Diagram Categories:**
1. **System Architecture** - High-level framework design
2. **Agent Lifecycle** - State machines and flows
3. **Multi-Agent Orchestration** - Communication patterns
4. **Tool Integration** - Execution pipelines
5. **Memory Systems** - Storage architecture
6. **Azure Integration** - Service ecosystems
7. **Deployment** - Infrastructure topology
8. **Security** - Authentication flows
9. **Data Flow** - Request-response cycles

---

## 🎓 Learning Paths

### **Path 1: Absolute Beginner (8-12 hours total)**

**Goal:** Build your first agent and understand basics

1. **[README.md](README.md)** - Setup & Overview (1 hour)
   - Install .NET SDK and tools
   - Setup Azure credentials
   - Verify installation

2. **[Comprehensive Guide - Introduction](microsoft_agent_framework_dotnet_comprehensive_guide.md#introduction)** (30 min)
   - Understand what Agent Framework is
   - Learn key objectives

3. **[Comprehensive Guide - Core Fundamentals](microsoft_agent_framework_dotnet_comprehensive_guide.md#core-fundamentals)** (2 hours)
   - Architecture principles
   - Authentication
   - Dependency Injection
   - Configuration

4. **[Recipes - Recipe 1: Simple Chat Agent](microsoft_agent_framework_dotnet_recipes.md#recipe-1-simple-chat-agent---csharp)** (1 hour)
   - Copy and run first example
   - Understand code structure

5. **[Comprehensive Guide - Simple Agents](microsoft_agent_framework_dotnet_comprehensive_guide.md#simple-agents)** (2 hours)
   - Learn agent creation patterns
   - Understand lifecycle

6. **[Recipes - Recipe 2: Agent with Single Tool](microsoft_agent_framework_dotnet_recipes.md#recipe-2-agent-with-single-tool)** (1-2 hours)
   - Add tool to your agent
   - Test tool invocation

7. **Practice Project** (2-3 hours)
   - Build custom agent for your use case
   - Add one custom tool
   - Test thoroughly

**Outcome:** You can create basic agents with tools

---

### **Path 2: Intermediate Developer (15-20 hours total)**

**Goal:** Build multi-agent systems and integrate with Azure

**Prerequisites:** Complete Beginner path

1. **[Comprehensive Guide - Multi-Agent Systems](microsoft_agent_framework_dotnet_comprehensive_guide.md#multi-agent-systems)** (4 hours)
   - Orchestration patterns
   - Communication patterns
   - State management

2. **[Diagrams - Multi-Agent Orchestration](microsoft_agent_framework_dotnet_diagrams.md#multi-agent-orchestration)** (30 min)
   - Visualize patterns
   - Understand flows

3. **[Recipes - Recipe 4: Multi-Agent Workflow](microsoft_agent_framework_dotnet_recipes.md#recipe-4-multi-agent-workflow)** (2 hours)
   - Implement multi-agent system
   - Test coordination

4. **[Comprehensive Guide - Tools Integration](microsoft_agent_framework_dotnet_comprehensive_guide.md#tools-integration)** (3 hours)
   - Advanced tool patterns
   - Azure tools
   - Custom tools

5. **[Recipes - Recipe 5: Multiple Tools](microsoft_agent_framework_dotnet_recipes.md#recipe-5-agent-with-multiple-tools)** (2 hours)
   - Implement complex tools
   - Test integration

6. **[Comprehensive Guide - Memory Systems](microsoft_agent_framework_dotnet_comprehensive_guide.md#memory-systems)** (3 hours)
   - Persistent storage
   - Vector memory
   - SQL Server/Cosmos DB

7. **Practice Project** (5-8 hours)
   - Build 3-agent orchestrated system
   - Add persistent memory
   - Integrate with Azure service

**Outcome:** You can build production-ready multi-agent systems

---

### **Path 3: Advanced/Production (25-35 hours total)**

**Goal:** Deploy enterprise-grade agent systems to production

**Prerequisites:** Complete Intermediate path

1. **[Production Guide - Complete Read](microsoft_agent_framework_dotnet_production_guide.md)** (6 hours)
   - Deployment strategies
   - Security
   - Scaling
   - Monitoring

2. **[Diagrams - Deployment Architecture](microsoft_agent_framework_dotnet_diagrams.md#deployment-architecture)** (1 hour)
   - Infrastructure patterns
   - Network topology

3. **[Comprehensive Guide - Agentic Patterns](microsoft_agent_framework_dotnet_comprehensive_guide.md#agentic-patterns)** (4 hours)
   - Advanced reasoning
   - Planning patterns
   - Autonomous agents

4. **[Recipes - Recipe 8: RAG Implementation](microsoft_agent_framework_dotnet_recipes.md#recipe-8-rag-retrieval-augmented-generation)** (4 hours)
   - Build RAG system
   - Integrate with Azure AI Search

5. **[Recipes - Integration Recipes](microsoft_agent_framework_dotnet_recipes.md#integration-recipes)** (6 hours)
   - Azure Functions
   - Event-driven patterns
   - ASP.NET Core integration

6. **[Production Guide - Security](microsoft_agent_framework_dotnet_production_guide.md#security-best-practices)** (2 hours)
   - Implement security
   - Key Vault integration

7. **[Production Guide - Monitoring](microsoft_agent_framework_dotnet_production_guide.md#monitoring--observability)** (2 hours)
   - Setup Application Insights
   - Custom metrics
   - Alerting

8. **Production Project** (15-20 hours)
   - Deploy to Azure
   - Implement monitoring
   - Setup CI/CD
   - Load testing
   - Security hardening

**Outcome:** You can deploy and operate production agent systems

---

## 🔍 Topic Index

### **By Concept**

#### **Agents**
- Creating agents: [Comprehensive Guide - Simple Agents](microsoft_agent_framework_dotnet_comprehensive_guide.md#simple-agents)
- AIAgent vs ChatAgent: [Comprehensive Guide - Core Fundamentals](microsoft_agent_framework_dotnet_comprehensive_guide.md#unified-sdk-structure)
- Lifecycle: [Comprehensive Guide - Agent Lifecycle](microsoft_agent_framework_dotnet_comprehensive_guide.md#agent-lifecycle-management)
- Testing: [Comprehensive Guide - Testing](microsoft_agent_framework_dotnet_comprehensive_guide.md#testing-individual-agents)

#### **Tools**
- Defining tools: [Comprehensive Guide - Tools Integration](microsoft_agent_framework_dotnet_comprehensive_guide.md#tools-integration)
- `AIFunctionFactory`: [Recipes - Tool Examples](microsoft_agent_framework_dotnet_recipes.md)
- Azure tools: [Comprehensive Guide - Built-in Tools](microsoft_agent_framework_dotnet_comprehensive_guide.md#tools-integration)
- Custom tools: [Recipes - Custom Tool Development](microsoft_agent_framework_dotnet_recipes.md#recipe-10-custom-tool-development)

#### **Multi-Agent Systems**
- Orchestration: [Comprehensive Guide - Multi-Agent Systems](microsoft_agent_framework_dotnet_comprehensive_guide.md#multi-agent-systems)
- Communication patterns: [Comprehensive Guide - Communication](microsoft_agent_framework_dotnet_comprehensive_guide.md#communication-patterns-between-agents)
- State management: [Comprehensive Guide - Shared State](microsoft_agent_framework_dotnet_comprehensive_guide.md#shared-state-management)
- Workflows: [Comprehensive Guide - Workflow Coordination](microsoft_agent_framework_dotnet_comprehensive_guide.md#workflow-coordination)

#### **Dependency Injection**
- Configuration: [Comprehensive Guide - DI Setup](microsoft_agent_framework_dotnet_comprehensive_guide.md#core-fundamentals)
- Service lifetimes: [Comprehensive Guide - DI](microsoft_agent_framework_dotnet_comprehensive_guide.md#core-fundamentals)
- IOptions pattern: [Recipes - Configuration](microsoft_agent_framework_dotnet_recipes.md)

#### **Memory & Persistence**
- Memory systems: [Comprehensive Guide - Memory Systems](microsoft_agent_framework_dotnet_comprehensive_guide.md#memory-systems)
- SQL Server integration: [Recipes - Memory Integration](microsoft_agent_framework_dotnet_recipes.md#recipe-6-memory-integration)
- Cosmos DB: [Comprehensive Guide - Cosmos DB State](microsoft_agent_framework_dotnet_comprehensive_guide.md#cosmos-db-state-backend)
- Vector memory: [Comprehensive Guide - Memory Systems](microsoft_agent_framework_dotnet_comprehensive_guide.md#memory-systems)

---

## 🔧 By Task

### **"I want to..."**

#### **...build my first agent**
→ [Recipes - Recipe 1: Simple Chat Agent](microsoft_agent_framework_dotnet_recipes.md#recipe-1-simple-chat-agent---csharp)

#### **...add tools to an agent**
→ [Recipes - Recipe 2: Agent with Single Tool](microsoft_agent_framework_dotnet_recipes.md#recipe-2-agent-with-single-tool)

#### **...coordinate multiple agents**
→ [Recipes - Recipe 4: Multi-Agent Workflow](microsoft_agent_framework_dotnet_recipes.md#recipe-4-multi-agent-workflow)

#### **...implement RAG**
→ [Recipes - Recipe 8: RAG Implementation](microsoft_agent_framework_dotnet_recipes.md#recipe-8-rag-retrieval-augmented-generation)

#### **...deploy to production**
→ [Production Guide - Complete](microsoft_agent_framework_dotnet_production_guide.md)

#### **...integrate with Azure Functions**
→ [Recipes - Recipe 11: Azure Functions](microsoft_agent_framework_dotnet_recipes.md#recipe-11-azure-functions-integration)

#### **...handle errors properly**
→ [Recipes - Recipe 3: Error Handling](microsoft_agent_framework_dotnet_recipes.md#recipe-3-basic-error-handling)

#### **...add memory to agents**
→ [Recipes - Recipe 6: Memory Integration](microsoft_agent_framework_dotnet_recipes.md#recipe-6-memory-integration)

#### **...setup dependency injection**
→ [Comprehensive Guide - DI Setup](microsoft_agent_framework_dotnet_comprehensive_guide.md#core-fundamentals)

#### **...understand the architecture**
→ [Diagrams - System Architecture](microsoft_agent_framework_dotnet_diagrams.md#system-architecture)

#### **...optimize performance**
→ [Production Guide - Performance Tuning](microsoft_agent_framework_dotnet_production_guide.md#performance-tuning)

#### **...setup monitoring**
→ [Production Guide - Monitoring](microsoft_agent_framework_dotnet_production_guide.md#monitoring--observability)

#### **...secure my agents**
→ [Production Guide - Security](microsoft_agent_framework_dotnet_production_guide.md#security-best-practices)

---

## 💡 Tips for Using These Guides

### **For Learning**
1. Start with README for setup
2. Read Comprehensive Guide sections in order
3. Try each Recipe as you learn concepts
4. Use Diagrams to visualize concepts
5. Reference Production Guide when ready to deploy

### **For Reference**
1. Use GUIDE_INDEX (this file) to find topics
2. Bookmark frequently used sections
3. Use Visual Studio search within documents
4. Keep Recipes open while coding

### **For Teaching**
1. Use Diagrams for explanations
2. Share relevant Recipe links
3. Reference Comprehensive Guide for depth
4. Use Production Guide for best practices

---

## 📞 Getting Help

### **Within These Guides**
- Search this index for your topic
- Check Troubleshooting Patterns in Recipes
- Review relevant Diagrams for visualization

### **External Resources**
- [Microsoft Agent Framework GitHub](https://github.com/microsoft/agent-framework)
- [.NET Samples Repository](https://github.com/microsoft/Agent-Framework-Samples/tree/main/dotnet)
- [Microsoft Learn](https://learn.microsoft.com/en-us/agent-framework/)
- [GitHub Issues](https://github.com/microsoft/agent-framework/issues)
- [Microsoft 365 Agents SDK for .NET](https://github.com/microsoft/agents-for-net)

---

## 🎯 Quick Reference Card

| Need | Document | Section |
|------|----------|---------|
| Install | README.md | Installation & Setup |
| First Agent | Recipes | Recipe 1 |
| Add Tool | Recipes | Recipe 2 |
| Multi-Agent | Comprehensive | Multi-Agent Systems |
| Memory | Recipes | Recipe 6 |
| RAG | Recipes | Recipe 8 |
| Deploy | Production Guide | Production Deployment |
| Monitor | Production Guide | Monitoring |
| Secure | Production Guide | Security |
| Optimize | Production Guide | Performance Tuning |
| DI Setup | Comprehensive | Core Fundamentals |

---

## 🔑 .NET-Specific Best Practices

### **Async/Await**
- Always use `async`/`await` - never `.Result` or `.Wait()`
- Use `Task.WhenAll()` for concurrent operations
- Implement `CancellationToken` support
- [Guide Reference](microsoft_agent_framework_dotnet_comprehensive_guide.md#async-patterns)

### **Dependency Injection**
- Register services with appropriate lifetimes
- Use `IOptions<T>` for configuration
- Implement `IDisposable`/`IAsyncDisposable`
- [Guide Reference](microsoft_agent_framework_dotnet_comprehensive_guide.md#dependency-injection)

### **Configuration**
- Use `appsettings.json` for non-secret config
- Use Azure Key Vault for secrets
- Leverage environment-specific configs
- [Guide Reference](microsoft_agent_framework_dotnet_comprehensive_guide.md#configuration)

### **Testing**
- Use xUnit or NUnit
- Mock with Moq
- Use `Azure.Core.TestFramework`
- [Guide Reference](microsoft_agent_framework_dotnet_comprehensive_guide.md#testing)

---

**Last Updated:** November 2025  
**Guide Maintainer:** AI Documentation Team  
**Feedback:** Please submit issues or suggestions via GitHub

**Happy Building! ⚡🤖**


### Advanced Guides
- microsoft_agent_framework_dotnet_advanced.md

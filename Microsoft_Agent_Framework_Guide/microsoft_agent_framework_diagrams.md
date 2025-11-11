# Microsoft Agent Framework - Architecture Diagrams & Visualisations
## October 2025 Release

**Document Version:** 1.0  
**Purpose:** Visual representation of Agent Framework architecture and system design

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Agent Lifecycle](#agent-lifecycle)
3. [Multi-Agent Orchestration](#multi-agent-orchestration)
4. [Tool Integration Architecture](#tool-integration-architecture)
5. [Memory Systems Architecture](#memory-systems-architecture)
6. [Azure Integration](#azure-integration)
7. [Deployment Architecture](#deployment-architecture)
8. [Authentication & Security Flow](#authentication--security-flow)
9. [Data Flow Patterns](#data-flow-patterns)

---

## System Architecture

### High-Level Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  User Applications / Interfaces                           │  │
│  │  (Web Apps, Chat Clients, APIs, Teams Bots)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            MICROSOFT AGENT FRAMEWORK CORE LAYER                  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            Agent Execution Engine                          │ │
│  │  ┌──────────────┐  ┌────────────┐  ┌────────────────────┐ │ │
│  │  │ ChatAgent    │  │ AIAgent    │  │ Custom Agents      │ │ │
│  │  │              │  │ (Stateless)│  │ (Extended Runtime) │ │ │
│  │  └──────────────┘  └────────────┘  └────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Supporting Services & Middleware                  │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ Tools &  │  │ Memory   │  │ Observ-  │  │ Auth &   │ │ │
│  │  │ Functions│  │ Management  │ability    │  │ Security │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Integration & Extensibility Layer                 │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ Plugin   │  │ MCP      │  │ Custom   │  │ Workflow │ │ │
│  │  │ System   │  │ Integration│ Providers │  │ Builder  │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            EXTERNAL INTEGRATIONS & BACKENDS                      │
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │  LLM Providers │  │ Data Sources   │  │ Azure Services │    │
│  │  ┌──────────┐  │  │ ┌──────────┐  │  │ ┌──────────┐  │    │
│  │  │Azure    │  │  │ │Azure AI  │  │  │ │CosmosDB  │  │    │
│  │  │OpenAI   │  │  │ │Search    │  │  │ │          │  │    │
│  │  ├──────────┤  │  │ ├──────────┤  │  │ ├──────────┤  │    │
│  │  │OpenAI   │  │  │ │APIs      │  │  │ │Logic Apps│  │    │
│  │  │          │  │  │ ├──────────┤  │  │ ├──────────┤  │    │
│  │  ├──────────┤  │  │ │Custom    │  │  │ │Functions │  │    │
│  │  │Anthropic│  │  │ │Databases │  │  │ │          │  │    │
│  │  └──────────┘  │  │ └──────────┘  │  │ └──────────┘  │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Layered Architecture Detail

```
Layer 1 (Presentation)
    ↓
┌─────────────────────────────┐
│   Copilot Studio            │  Multi-channel UI
│   Web App APIs              │
│   Mobile Interfaces         │
└─────────────────────────────┘
    ↓
Layer 2 (Application)
    ↓
┌─────────────────────────────┐
│   ChatAgent / AIAgent       │  Core agent types
│   Workflow Orchestration    │
│   Task Management           │
└─────────────────────────────┘
    ↓
Layer 3 (Service)
    ↓
┌─────────────────────────────┐
│   Tools Service             │  Function execution
│   Memory Service            │
│   Auth Service              │
│   Observability Service     │
└─────────────────────────────┘
    ↓
Layer 4 (Data)
    ↓
┌─────────────────────────────┐
│   Azure AI Search           │  Vector/full-text search
│   Azure Cosmos DB           │  Persistent state
│   Azure Blob Storage        │  Large data
│   Custom Databases          │
└─────────────────────────────┘
```

---

## Agent Lifecycle

### Complete Agent Lifecycle State Machine

```
                    ┌──────────────────┐
                    │   Initialisation  │
                    │   (Create Agent)  │
                    └────────┬──────────┘
                             ↓
                    ┌──────────────────┐
                    │  Configuration   │
                    │ (Setup Tools,    │
                    │  Memory, Auth)   │
                    └────────┬──────────┘
                             ↓
        ┌────────────────────────────────────────────┐
        │                                            │
        ↓                                            ↓
┌──────────────────┐                        ┌──────────────────┐
│  Ready/Idle      │                        │ Context Check    │
│ (Awaiting Input) │                        │ (Load State)     │
└────────┬─────────┘                        └────────┬─────────┘
         │                                           │
         └──────────────────┬──────────────────────┘
                            ↓
                   ┌────────────────────┐
                   │  Execute Query     │
                   │ (Invoke Agent)     │
                   └─────────┬──────────┘
                             ↓
                   ┌────────────────────┐
                   │ Tool Invocation    │
                   │ (if needed)        │
                   └─────────┬──────────┘
                             ↓
                   ┌────────────────────┐
                   │ Generate Response  │
                   │ (LLM Call)         │
                   └─────────┬──────────┘
                             ↓
                   ┌────────────────────┐
                   │ Response Processing│
                   │ (Format Output)    │
                   └─────────┬──────────┘
                             ↓
        ┌────────────────────────────────────────────┐
        │                                            │
        ↓                                            ↓
┌──────────────────┐                        ┌──────────────────┐
│ Return Response  │                        │ Save State       │
│ (to User)        │                        │ (Persist Memory) │
└─────────┬────────┘                        └────────┬─────────┘
          │                                         │
          └──────────────────┬────────────────────┘
                             ↓
                   ┌────────────────────┐
                   │  Ready/Idle        │
                   │  (Await Next Input)│
                   └─────────┬──────────┘
                             ↓
                   ┌────────────────────┐
                   │ Termination        │
                   │ (Cleanup, Close)   │
                   └────────────────────┘
```

### Request/Response Flow

```
Client/User
    ↓
┌─────────────────────────────────┐
│  1. Send Query/Request          │
└─────────────┬───────────────────┘
              ↓
        ┌─────────────────────┐
        │ Agent Framework     │
        │ Entry Point         │
        └────────┬────────────┘
                 ↓
        ┌─────────────────────────────┐
        │ 2. Load Thread/Context      │
        │    (From Memory Store)      │
        └────────┬────────────────────┘
                 ↓
        ┌─────────────────────────────┐
        │ 3. Prepare Message          │
        │    (Add System Instructions)│
        └────────┬────────────────────┘
                 ↓
        ┌─────────────────────────────┐
        │ 4. Call LLM                 │
        │    (Azure OpenAI/Provider)  │
        └────────┬────────────────────┘
                 ↓
     ┌──────────────────────────────┐
     │ LLM Returns:                 │
     │ - Text response OR           │
     │ - Tool call request          │
     └────────┬─────────────────────┘
              ↓
     ┌────────────────────────────────────┐
     │ 5. Decision: Tool call?            │
     └────────┬──────────────┬────────────┘
              │ YES          │ NO
              ↓              ↓
        ┌──────────────┐  ┌─────────────┐
        │ 6a. Execute  │  │ 7. Format   │
        │ Requested    │  │ Response    │
        │ Tool         │  └────────┬────┘
        └──────┬───────┘           │
               ↓                   │
        ┌──────────────┐           │
        │ 6b. Collect  │           │
        │ Tool Result  │           │
        └──────┬───────┘           │
               ↓                   │
        ┌──────────────┐           │
        │ 6c. Add      │           │
        │ Result to    │           │
        │ Context      │           │
        └──────┬───────┘           │
               ↓                   │
        ┌──────────────┐           │
        │ 6d. Call     │           │
        │ LLM Again    │           │
        └──────┬───────┘           │
               │                   │
               └───────┬───────────┘
                       ↓
              ┌─────────────────────┐
              │ 8. Save to History  │
              │ & Memory            │
              └─────────┬───────────┘
                        ↓
              ┌─────────────────────┐
              │ 9. Return Response  │
              │ to Client           │
              └─────────────────────┘
```

---

## Multi-Agent Orchestration

### Simple Sequential Workflow

```
Input Query
    ↓
┌─────────────────────────┐
│ Agent 1: Validation     │  ← Validates input, extracts parameters
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Agent 2: Research       │  ← Gathers information
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Agent 3: Analysis       │  ← Processes findings
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Agent 4: Synthesis      │  ← Combines results
└──────────┬──────────────┘
           ↓
Output Response
```

### Complex Workflow with Branching

```
                      Input
                        ↓
              ┌─────────────────────┐
              │ Router Agent        │
              │ (Classify Task)     │
              └┬──────────────────┬─┘
         Type A│                  │Type B
               ↓                  ↓
     ┌──────────────────┐  ┌──────────────────┐
     │ Path A Agents    │  │ Path B Agents    │
     │                  │  │                  │
     │ A1 → A2 → A3    │  │ B1 → B2 → B3    │
     └────────┬─────────┘  └────────┬─────────┘
              │                     │
              └──────────┬──────────┘
                         ↓
              ┌──────────────────────┐
              │ Aggregator Agent     │
              │ (Combine Results)    │
              └──────────┬───────────┘
                         ↓
                    Output
```

### Conditional Branching with Feedback

```
                    Start
                      ↓
              ┌─────────────────┐
              │ Agent: Analyse  │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
            /─┤ Confidence?     │─\
           /  └─────────────────┘  \
        High /                       \ Low
          /                           \
         ↓                             ↓
    ┌─────────────┐            ┌─────────────────┐
    │ Execute     │            │ Request More    │
    │ Plan        │            │ Information     │
    └────┬────────┘            └────────┬────────┘
         ↓                              ↓
    ┌─────────────┐            ┌─────────────────┐
    │ Verify      │            │ Re-analyse      │
    │ Results     │            │ with New Data   │
    └────┬────────┘            └────────┬────────┘
         ↓                              ↓
    ┌─────────────┐            ┌─────────────────┐
    │ Return      │            └────────┬────────┘
    │ Final Result│                     │
    └─────────────┘                     │
                       ┌────────────────┘
                       │
                    Success!
```

---

## Tool Integration Architecture

### Tool Execution Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                      Agent Query                              │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────────────┐
        │ LLM Decision: Need Tools?              │
        └─────────────┬──────────────────────────┘
                      ↓
         ┌────────────────────────────────────────┐
         │ Tool Selection & Parameter Extraction  │
         └──────────┬─────────────────────────────┘
                    ↓
        ┌──────────────────────────────────────────┐
        │  Tool Validation                         │
        │  - Check permissions                     │
        │  - Validate parameters                   │
        │  - Check resource availability          │
        └──────────┬────────────────────────────────┘
                   ↓
        ┌──────────────────────────────────────────┐
        │ Route to Tool Handler                    │
        └─┬────────────────────────────────────────┘
          │
    ┌─────┴───────┬──────────────┬────────────┐
    ↓             ↓              ↓            ↓
┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
│ Function   │ │ Azure      │ │ MCP      │ │ Custom   │
│ Tool       │ │ Service    │ │ Server   │ │ Tool     │
│            │ │            │ │          │ │          │
│ Execute    │ │ API Call   │ │ Query    │ │ Handler  │
│ Local      │ │ (Search,   │ │ Resource │ │ Invoke   │
│ Function   │ │ Functions) │ │ & Tools  │ │ External │
└─────┬──────┘ └────┬───────┘ └────┬─────┘ └────┬─────┘
      │             │              │            │
      └──────────┬──┴──────────┬───┴───────────┘
                 ↓             ↓
            ┌──────────────────────────┐
            │ Aggregate Results        │
            │ Format for LLM           │
            └────────┬─────────────────┘
                     ↓
            ┌──────────────────────────┐
            │ Return Tool Result       │
            │ to Agent Context         │
            └────────┬─────────────────┘
                     ↓
            ┌──────────────────────────┐
            │ Continue Agent Loop      │
            │ (Possibly More Tools)    │
            └──────────────────────────┘
```

### Tool Types and Providers

```
┌────────────────────────────────────────────────────────────────┐
│                    TOOL ECOSYSTEM                               │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Built-in Tools (Framework Provided)                            │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Azure Search  │ Logic Apps │ Functions │ Knowledge     │   │
│  │               │            │           │ Retrieval     │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ├──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Function Tools (User-Defined Functions)                │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ @ai_function decorated methods                   │  │  │
│  │  │ Python functions with type hints                 │  │  │
│  │  │ C# methods with [Function] attributes            │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ├──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  MCP Tools (Model Context Protocol)                     │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ Remote MCP Servers                              │  │  │
│  │  │ File System Operations                          │  │  │
│  │  │ Web Search                                      │  │  │
│  │  │ Custom Protocol Servers                         │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ├──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Custom Tools                                           │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ Third-party APIs                                │  │  │
│  │  │ Enterprise Systems Integration                  │  │  │
│  │  │ Database Operations                             │  │  │
│  │  │ Custom Business Logic                           │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Memory Systems Architecture

### Multi-Tier Memory Architecture

```
┌────────────────────────────────────────────────────────────┐
│            SESSION LAYER (Ephemeral)                        │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Current Conversation History                         │ │
│  │ Short-term Context                                   │ │
│  │ In-Memory Storage (LRU Cache)                        │ │
│  │ TTL: Duration of conversation                        │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│            PERSISTENT LAYER                                 │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Message History                                       │ │
│  │ User Preferences                                      │ │
│  │ Agent State                                           │ │
│  │ Storage: Azure Cosmos DB / SQLite                    │ │
│  │ TTL: Configurable (days to years)                    │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│            VECTOR MEMORY LAYER (RAG)                        │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Embeddings                                           │ │
│  │ Semantic Search Index                                │ │
│  │ Document Chunks & Metadata                           │ │
│  │ Storage: Azure AI Search                             │ │
│  │ Use Case: Knowledge retrieval, similarity search     │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│            REFERENCE LAYER (Knowledge Base)                 │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Company Policies                                      │ │
│  │ Product Catalogs                                      │ │
│  │ FAQ Documents                                         │ │
│  │ Storage: Azure Blob, SharePoint, Databases           │ │
│  │ TTL: Long-term / Evergreen                           │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### Memory Access Pattern

```
Query
  ↓
┌─────────────────────────────────────┐
│ 1. Check Session Cache              │  ← Fast lookup (milliseconds)
└──┬───────────────────────────────────┘
   │ Hit: Return cached context
   │ Miss: Continue
   ↓
┌──────────────────────────────────────┐
│ 2. Query Persistent Memory           │  ← Database lookup (< 100ms)
│    (Azure Cosmos DB)                 │
└──┬───────────────────────────────────┘
   │ Found: Update session cache
   │ Not found: Continue
   ↓
┌──────────────────────────────────────┐
│ 3. Vector Semantic Search            │  ← Similarity matching
│    (Azure AI Search)                 │
└──┬───────────────────────────────────┘
   │ Found relevant chunks: 
   │ Add to context
   │ Not found: Continue
   ↓
┌──────────────────────────────────────┐
│ 4. Reference Knowledge Base          │  ← Document retrieval
└──┬───────────────────────────────────┘
   │
   ↓
┌──────────────────────────────────────┐
│ 5. Aggregate & Pass to LLM           │
└──────────────────────────────────────┘
```

---

## Azure Integration

### Azure Services Ecosystem

```
┌────────────────────────────────────────────────────────────────┐
│                  MICROSOFT AGENT FRAMEWORK                      │
└────────────────────────────┬─────────────────────────────────────┘
                             ↓
          ┌──────────────────────────────────────┐
          │   Azure AI Foundry                    │
          │  (Project Hub & Management)           │
          └──────────────────────────────────────┘
                             ↓
        ┌────────────────────────────────────────┐
        │          Connectors & Services          │
        └────────────────────────────────────────┘
                        ↓↓↓↓↓↓↓
     ┌──────────┬──────────┬──────────┬──────────┐
     ↓          ↓          ↓          ↓          ↓
┌────────────────────────────────────────────────────────┐
│ Compute     │ Data     │ Intelligence │ Integration   │
├────────────────────────────────────────────────────────┤
│ • Container │ • Search │ • OpenAI    │ • Logic Apps  │
│   Apps      │ • Cosmos │ • Cognitive │ • API Mgmt    │
│ • Functions │ • Blob   │ • Translator│ • Service Bus │
│ • App Svc   │ • SQL    │ • Vision    │ • Event Grid  │
└────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────┐
│ Management & Monitoring                               │
├────────────────────────────────────────────────────────┤
│ • Azure Monitor         • Application Insights        │
│ • Log Analytics         • Azure Policy                │
│ • Key Vault             • RBAC                        │
└────────────────────────────────────────────────────────┘
```

### Azure AI Integration Detail

```
┌────────────────────────────────────────────────────────────┐
│         AZURE OPENAI SERVICE                               │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Deployment: gpt-4o, gpt-4o-mini, gpt-4-turbo            │
│  Purpose: LLM inference, embeddings                        │
│  Integration: ChatClient, EmbeddingsClient                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Request Flow:                                        │ │
│  │ Agent Query → Framework → Azure OpenAI API           │ │
│  │ Response ← Framework ← LLM Response                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│         AZURE AI SEARCH                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Capabilities: Full-text, vector, hybrid search            │
│  Purpose: RAG, semantic search, knowledge retrieval        │
│  Integration: Vector memory provider                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Search Flow:                                         │ │
│  │ Query → Generate Embedding → Vector Search → Results │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│         AZURE COSMOS DB                                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Capabilities: Multi-model, globally distributed            │
│  Purpose: Persistent state, conversation history           │
│  Integration: Memory provider backend                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Storage Flow:                                        │ │
│  │ State → Serialise → Write to Cosmos → Persistent    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Containerised Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT TARGET                        │
│                   (Azure Container Apps)                     │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                Container Environment                  │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │        Agent Framework Container               │ │ │
│  │  │  ┌────────────────────────────────────────────┐ │ │ │
│  │  │  │ Runtime (Python / .NET)                    │ │ │ │
│  │  │  ├────────────────────────────────────────────┤ │ │ │
│  │  │  │ Agent Framework Library                    │ │ │ │
│  │  │  ├────────────────────────────────────────────┤ │ │ │
│  │  │  │ Custom Agent Code                          │ │ │ │
│  │  │  ├────────────────────────────────────────────┤ │ │ │
│  │  │  │ Dependencies (OpenAI, Azure SDKs)          │ │ │ │
│  │  │  └────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  │                      ↑                               │ │
│  │                      │ (Requests)                    │ │
│  │                      │                               │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │  Load Balancer                                  │ │ │
│  │  │  (Azure Load Balancer / Application Gateway)    │ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                               │
│  Replica Set: N instances (auto-scaling based on metrics)   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Pod 1  │ Pod 2  │ Pod 3  │ ... │ Pod N               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Environment Deployment

```
Development Environment
    ↓
┌──────────────────────────────────────────┐
│ Local / Dev Azure Subscription           │
│ Single instance                          │
│ Minimal replicas                         │
│ Dev databases                            │
└──────────────────────────────────────────┘
    ↓
Staging Environment
    ↓
┌──────────────────────────────────────────┐
│ Staging Azure Subscription               │
│ 2-3 replicas                             │
│ Production-like config                   │
│ Clone of production data (anonymised)    │
└──────────────────────────────────────────┘
    ↓
Production Environment
    ↓
┌──────────────────────────────────────────┐
│ Production Azure Subscription            │
│ N replicas (auto-scaling)                │
│ Full monitoring & alerts                 │
│ Production data with backups             │
│ Disaster recovery replicas               │
└──────────────────────────────────────────┘
```

---

## Authentication & Security Flow

### Authentication Pipeline

```
Client Request
    ↓
┌────────────────────────────────────────┐
│ 1. Receive Request                     │
└────────┬───────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 2. Extract Authentication Credential   │
│    (Bearer Token / API Key)            │
└────────┬───────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 3. Validate Token                      │
│    - Check signature                   │
│    - Verify expiration                 │
│    - Check revocation list             │
└────────┬───────────────────────────────┘
         │ Invalid → Reject (401)
         │ Valid → Continue
         ↓
┌────────────────────────────────────────┐
│ 4. Extract User/Service Identity       │
│    (From AAD Token Claims)             │
└────────┬───────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 5. Check RBAC Authorisation            │
│    - Agent execution role?             │
│    - Tool access permissions?          │
│    - Data access scope?                │
└────────┬───────────────────────────────┘
         │ Unauthorised → Reject (403)
         │ Authorised → Continue
         ↓
┌────────────────────────────────────────┐
│ 6. Create Security Context             │
│    (Attach to request)                 │
└────────┬───────────────────────────────┘
         ↓
┌────────────────────────────────────────┐
│ 7. Process Request                     │
│    (With security context)             │
└────────────────────────────────────────┘
```

### RBAC Model

```
┌────────────────────────────────────────────────────────────┐
│              Azure Active Directory                         │
│                                                             │
│  User/Service Principal                                     │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ RBAC Role Assignment                              │   │
│  │                                                    │   │
│  │ ├─ Agent Reader                                   │   │
│  │ │  └─ Read agent definitions & logs              │   │
│  │ │                                                 │   │
│  │ ├─ Agent Contributor                             │   │
│  │ │  └─ Create, modify agents & tools              │   │
│  │ │                                                 │   │
│  │ ├─ Agent User                                     │   │
│  │ │  └─ Execute agents                             │   │
│  │ │                                                 │   │
│  │ └─ Azure AI Project Admin                         │   │
│  │    └─ Full project & subscription management      │   │
│  │                                                    │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Access Control Evaluation                          │   │
│  └────────────────────────────────────────────────────┘   │
│           ↓                                                 │
│  Allowed / Denied                                          │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Data Flow Patterns

### Complete Request-Response Cycle

```
External System/User
         ↓
┌────────────────────────────────────────────────────┐
│ 1. INGRESS                                         │
│    REST API / WebSocket / Direct SDK Call          │
└────────┬─────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ 2. AUTHENTICATION & AUTHORISATION                 │
│    Validate credentials, check permissions         │
└────────┬─────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ 3. REQUEST PARSING & VALIDATION                   │
│    Extract intent, validate schema                 │
└────────┬─────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ 4. AGENT FRAMEWORK PROCESSING                     │
│    ├─ Load agent                                   │
│    ├─ Load context/memory                          │
│    ├─ Format prompt                                │
│    └─ Invoke LLM                                   │
└────────┬─────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ 5. LLM INTERACTION                                │
│    ├─ Send messages to Azure OpenAI                │
│    ├─ Receive response                             │
│    └─ Parse tool calls (if any)                    │
└────────┬─────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────┐
    │ Tool Calls? ──No──┐            │
    │     │             ↓            │
    │    Yes       ┌──────────────┐  │
    │     │        │ Format Output│  │
    │     ↓        └──────────────┘  │
    │ ┌──────────────────────────┐   │
    │ │ 6. TOOL EXECUTION        │   │
    │ │ Execute tools, collect   │   │
    │ │ results, add to context  │   │
    │ │ Go back to step 4        │   │
    │ └──────────────────────────┘   │
    │                                │
    └────────────┬───────────────────┘
                 ↓
┌────────────────────────────────────────────────────┐
│ 7. STATE PERSISTENCE                              │
│    Save conversation history & memory              │
└────────┬─────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ 8. RESPONSE FORMATTING                            │
│    Format response, apply transformations          │
└────────┬─────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────┐
│ 9. EGRESS                                          │
│    Return response via API / WebSocket             │
└────────┬─────────────────────────────────────────┘
         ↓
External System/User
```

### Error Handling Flow

```
Operation
    ↓
┌──────────────────────────────────────┐
│ Error Occurs?                        │
└──────┬───────────────────────────────┘
       │ No → Continue
       │ Yes
       ↓
┌──────────────────────────────────────┐
│ Error Classification                 │
├──────────────────────────────────────┤
│ ├─ Retriable (Network timeout)       │
│ ├─ Non-retriable (Auth failure)      │
│ └─ Partial (Tool failed)             │
└──┬────────────────────────────────────┘
   │
   ├─ Retriable
   │  ↓
   │  ┌──────────────────────────────┐
   │  │ Exponential Backoff Retry    │
   │  │ (up to max attempts)         │
   │  └──────┬───────────────────────┘
   │         │
   │         ├─ Success → Resume
   │         └─ Failure → Non-retriable
   │
   ├─ Non-retriable
   │  ↓
   │  ┌──────────────────────────────┐
   │  │ Log Error & Metrics          │
   │  │ Notify user / fallback       │
   │  └──────────────────────────────┘
   │
   └─ Partial
      ↓
      ┌──────────────────────────────┐
      │ Continue with available info │
      │ Mark failed components       │
      └──────────────────────────────┘
```

---

## Visualisation Summary

These diagrams illustrate:

1. **System Architecture:** Multi-layer design enabling scalability and flexibility
2. **Agent Lifecycle:** Complete state machine from creation to termination
3. **Multi-Agent Workflows:** Sequential and branching orchestration patterns
4. **Tool Integration:** Pipeline for tool discovery, validation, and execution
5. **Memory Systems:** Multi-tier architecture for context management
6. **Azure Integration:** Ecosystem of services supporting agents
7. **Deployment:** Containerised, scalable production setup
8. **Security:** Authentication and authorisation flows
9. **Data Flow:** Complete request/response cycle with error handling

---

**For detailed implementation, refer to the companion documents:**
- `microsoft_agent_framework_comprehensive_guide.md` - Core concepts & code
- `microsoft_agent_framework_production_guide.md` - Deployment & scaling
- `microsoft_agent_framework_recipes.md` - Practical code patterns


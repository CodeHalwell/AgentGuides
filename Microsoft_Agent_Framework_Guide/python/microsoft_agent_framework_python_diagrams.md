# Microsoft Agent Framework Python - Architecture Diagrams

This document provides visual references for the architecture, data flow, and deployment topology of Python-based agent systems.

**Target Platform:** Python 3.10+

---

## 1. System Architecture

### High-Level Layered Design

```mermaid
graph TD
    User[User / Client App] --> API[FastAPI / Flask Interface]
    
    subgraph "Agent Framework (Python)"
        API --> Orchestrator[Orchestrator / Workflow]
        Orchestrator --> AgentA[ChatAgent A]
        Orchestrator --> AgentB[ChatAgent B]
        
        AgentA --> Memory[Memory Store]
        AgentA --> Tools[Tools / Plugins]
        
        Tools --> PythonFunc[Python Functions]
        Tools --> AzureAI[Azure AI Services]
    end
    
    AgentA --> LLM[LLM Service (OpenAI)]
    Memory --> VectorDB[Vector Database]
```

---

## 2. Agent Lifecycle (AsyncIO)

The lifecycle of a Python agent within an `asyncio` event loop.

```mermaid
sequenceDiagram
    participant App as Application (Main Loop)
    participant Factory as AgentFactory
    participant Agent as ChatAgent
    participant Thread as AgentThread
    participant LLM as LLM Service

    App->>Factory: create_agent(instructions, tools)
    Factory-->>Agent: Returns Agent Instance
    
    App->>Agent: create_thread()
    Agent-->>Thread: Returns Thread Instance
    
    loop Conversation Loop
        App->>Thread: invoke(user_input)
        activate Thread
        Thread->>Thread: Add User Message to History
        Thread->>LLM: Send History + System Prompt
        LLM-->>Thread: Response (Content or Tool Call)
        
        alt Tool Call Required
            Thread->>Thread: Execute Python Tool (await)
            Thread->>LLM: Send Tool Result
            LLM-->>Thread: Final Response
        end
        
        Thread-->>App: Return AgentResponse
        deactivate Thread
    end
```

---

## 3. Multi-Agent Orchestration (Router Pattern)

A common pattern where a router agent dispatches tasks to specialized agents.

```mermaid
graph LR
    Input[User Query] --> Router[Router Agent]
    
    Router -->|Classifies as Billing| Billing[Billing Agent]
    Router -->|Classifies as Tech| Tech[Tech Support Agent]
    
    Billing -->|Uses| CRM[CRM Tool]
    Tech -->|Uses| Logs[Log Analysis Tool]
    
    Billing --> Output[Final Response]
    Tech --> Output
```

---

## 4. Deployment Architecture (Azure Container Apps)

Recommended production topology for Python agents.

```mermaid
graph TD
    Internet((Internet)) --> FrontDoor[Azure Front Door]
    FrontDoor --> ACAEnv[Azure Container Apps Environment]
    
    subgraph "ACA Environment (VNet)"
        Ingress[Ingress Controller] --> Service[Python Agent Service (Gunicorn)]
        
        Service -->|Scale Out| Replica1[Replica 1]
        Service -->|Scale Out| Replica2[Replica 2]
        Service -->|Scale Out| ReplicaN[Replica N]
    end
    
    Replica1 --> KeyVault[Azure Key Vault]
    Replica1 --> OpenAI[Azure OpenAI]
    Replica1 --> Cosmos[Cosmos DB (Memory)]
    
    KEDA[KEDA Scaler] -->|Monitors| Service
    KEDA -.->|Metrics| AzureMonitor[Azure Monitor]
```

---

## 5. RAG Data Flow

Retrieval-Augmented Generation flow using Python and Azure AI Search.

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant SearchTool as Search Tool
    participant VectorDB as Azure AI Search
    participant LLM

    User->>Agent: "How do I configure authentication?"
    Agent->>LLM: Process Intent
    LLM-->>Agent: Call Tool: search_knowledge("authentication")
    
    Agent->>SearchTool: execute("authentication")
    SearchTool->>VectorDB: Vector Search
    VectorDB-->>SearchTool: Top k Documents
    SearchTool-->>Agent: Return Document Chunks
    
    Agent->>LLM: Generate Answer with Context
    LLM-->>Agent: "You can configure authentication using..."
    Agent-->>User: Final Answer
```

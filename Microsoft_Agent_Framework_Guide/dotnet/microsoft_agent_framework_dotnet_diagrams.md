# Microsoft Agent Framework .NET - Architecture & Workflow Diagrams

This document provides visual representations of the key architectural concepts, data flows, and interaction patterns within the Microsoft Agent Framework for .NET.

---

## 1. System Architecture

### High-Level Framework Design

This diagram shows the layered architecture of the framework, from the core components up to the application layer.

```mermaid
graph TD
    subgraph Your Application
        A[APIs, Services, UI]
    end

    subgraph Agent Framework
        B(Orchestration Layer<br/><i>Workflows, GroupChat</i>)
        C(Agent Abstraction Layer<br/><i>AIAgent, ChatAgent, AgentThread</i>)
        D(Core Components<br/><i>Tools, Memory, State</i>)
        E(LLM & Integration Layer<br/><i>AzureOpenAI, Connectors</i>)
    end

    A --> B
    B --> C
    C --> D
    D --> E
```

---

## 2. Agent Lifecycle

### State Machine for a `ChatAgent`

This diagram illustrates the lifecycle of a stateful `ChatAgent` as it processes messages within a thread.

```mermaid
graph TD
    Start((Create Thread)) --> Idle
    Idle -- InvokeAsync(userInput) --> Processing
    
    subgraph Processing
        direction LR
        A[Get History] --> B{LLM Inference}
        B -- Tool Needed? --> C{Select Tool}
        C -- Execute --> D[Tool Output]
        D --> B
        B -- No Tool --> E[Generate Response]
    end

    Processing -- Returns AgentResponse --> Idle
    Idle -- "InvokeAsync(...)" --> Processing
    Idle -- End of Session --> End((Thread Terminated))
```

---

## 3. Multi-Agent Orchestration

### Sequential Workflow

Agent A's output becomes the input for Agent B.

```mermaid
graph LR
    Input --> AgentA[Agent A<br/><i>e.g., Researcher</i>]
    AgentA -- OutputA --> AgentB[Agent B<br/><i>e.g., Summarizer</i>]
    AgentB -- Final Output --> Result
```

### Router (Broadcast) Pattern

A router agent delegates a task to the most appropriate specialist agent.

```mermaid
graph TD
    UserInput --> Router{Router Agent}
    Router -- "Is it a billing question?" --> Billing[Billing Agent]
    Router -- "Is it a technical issue?" --> TechSupport[Tech Support Agent]
    Router -- "Is it a sales inquiry?" --> Sales[Sales Agent]

    Billing --> FinalResponse
    TechSupport --> FinalResponse
    Sales --> FinalResponse
```

---

## 4. Tool Integration

### Tool Execution Pipeline

This shows how an agent selects and executes a tool.

```mermaid
graph TD
    Agent[Agent Logic] -- "User asks a question" --> LLM{LLM Decides}
    LLM -- "Tool 'GetCurrentWeather' should be called" --> ToolSelection
    
    subgraph Tool Execution
        ToolSelection[Select 'GetCurrentWeather' Tool]
        ToolSelection -- "Invoke with params: { city: 'Seattle' }" --> CSharpTool(C# Method<br/><i>GetCurrentWeather("Seattle")</i>)
        CSharpTool -- "Returns 'Rainy, 55°F'" --> ToolResult
    end

    ToolResult -- "Provide tool output back to LLM" --> LLM
    LLM -- "Formulate final answer" --> AgentResponse[Agent Response to User]
```

---

## 5. Memory Systems

### Multi-Tier Memory Architecture

Agents can access different layers of memory for context.

```mermaid
graph TD
    subgraph Agent
        A[Agent Core Logic]
    end

    subgraph Memory Tiers
        B(Short-Term Memory<br/><i>Current Conversation Thread</i>)
        C(Working Memory<br/><i>Recent Conversations, User Profile</i>)
        D(Long-Term Memory<br/><i>Vector DB for RAG, Knowledge Base</i>)
    end

    A <--> B
    A --> C
    A --> D
```

---

## 6. Azure Integration

### Azure Service Ecosystem

This diagram shows how the Agent Framework integrates with key Azure services.

```mermaid
graph TD
    subgraph Agent Host
        A[Azure Container Apps / AKS]
    end

    subgraph AI Services
        B[Azure OpenAI<br/><i>LLM Models</i>]
        C[Azure AI Search<br/><i>Vector Memory for RAG</i>]
        D[Azure Content Safety<br/><i>Moderation</i>]
    end

    subgraph Data & State
        E[Azure Cosmos DB<br/><i>Thread State, History</i>]
        F[Azure SQL Database<br/><i>Relational Memory</i>]
    end

    subgraph Observability
        G[Application Insights<br/><i>Logging, Tracing, Metrics</i>]
    end
    
    subgraph Security
        H[Azure Key Vault<br/><i>Secrets Management</i>]
    end

    A --> B
    A --> C
    A --> E
    A --> G
    A --> H
    B --> D
```

---

## 7. Deployment Architecture

### Production Topology on Azure

A typical high-availability deployment on Azure.

```mermaid
graph TD
    User[User] --> LB{Azure Load Balancer}
    
    subgraph VNet
        subgraph AppSubnet
            LB --> AppGW(Azure Application Gateway<br/><i>WAF</i>)
            AppGW --> AgentService1[Agent Service<br/>Instance 1]
            AppGW --> AgentService2[Agent Service<br/>Instance 2]
            AppGW --> AgentService3[Agent Service<br/>Instance 3]
        end

        subgraph DataSubnet
            AgentService1 --> DB[(Azure Cosmos DB)]
            AgentService2 --> DB
            AgentService3 --> DB
            AgentService1 --> AISearch[(Azure AI Search)]
        end

        subgraph IntegrationSubnet
            AgentService1 --> AzureOpenAI[Azure OpenAI Service]
            AgentService1 --> KeyVault[Azure Key Vault]
        end
    end

    style AppSubnet fill:#cde4ff
    style DataSubnet fill:#e2d1ff
    style IntegrationSubnet fill:#f8d7da
```

---

## 8. Security & Authentication Flow

### JWT-Based Authentication Flow

This shows a typical flow for securing an agent API.

```mermaid
sequenceDiagram
    participant Client
    participant AgentAPI as Agent API
    participant AuthServer as Authentication Server

    Client->>AuthServer: Request Token (username, password)
    AuthServer-->>Client: Returns JWT

    Client->>AgentAPI: Invoke Agent (with JWT in Header)
    AgentAPI->>AgentAPI: Validate JWT
    alt Is Valid
        AgentAPI->>AgentAPI: Process Request
        AgentAPI-->>Client: Agent Response
    else Is Invalid
        AgentAPI-->>Client: 401 Unauthorized
    end
```

---

## 9. Data Flow

### Complete Request-Response Cycle

This diagram traces a single user request through the entire system.

```mermaid
graph TD
    A[User Request via API] --> B{Middleware<br/><i>Auth, Logging</i>}
    B --> C{Agent Thread Manager}
    C -- Get or Create Thread --> D[State Backend<br/><i>e.g., Cosmos DB</i>]
    D -- Returns Thread State --> C
    C --> E{Agent Logic}
    E -- "Needs knowledge" --> F[Memory System<br/><i>e.g., Azure AI Search</i>]
    F -- "Returns relevant docs" --> E
    E -- "Needs tool" --> G[Tool Executor]
    G -- "Invokes C# method" --> H(External API / DB)
    H -- "Returns data" --> G
    G -- "Returns tool output" --> E
    E -- "LLM prompt with context" --> I[LLM Service<br/><i>Azure OpenAI</i>]
    I -- "LLM completion" --> E
    E -- "Final response" --> J{Update Thread State}
    J --> D
    E -- "Final response" --> K[Format Output]
    K --> A
```
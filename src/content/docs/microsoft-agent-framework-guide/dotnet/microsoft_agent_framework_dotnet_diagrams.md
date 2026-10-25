---
title: "Microsoft Agent Framework (.NET) — Architecture Diagrams"
description: "Diagrams of the Microsoft Agent Framework for .NET: package layers, the ChatClientAgent run loop with sessions, workflow supersteps, orchestration patterns, tool calls, memory, hosting endpoints, Azure integration and deployment."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Diagrams"
  order: 18
---

Visual summaries of the main concepts of the Microsoft Agent Framework for .NET. The workflow and hosting diagrams at the end link to the pages that explain them.

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
        H(Hosting<br/><i>AddAIAgent, OpenAI / A2A / AG-UI endpoints</i>)
        B(Workflows and orchestrations<br/><i>WorkflowBuilder, AgentWorkflowBuilder</i>)
        C(Agents<br/><i>AIAgent, ChatClientAgent, AgentSession</i>)
        D(Tools, context providers, middleware<br/><i>AIFunction, AIContextProvider, AIAgentBuilder</i>)
        E(Model clients<br/><i>IChatClient: OpenAI, Azure OpenAI, Foundry, Anthropic, ...</i>)
    end

    A --> H
    A --> B
    H --> C
    B --> C
    C --> D
    D --> E
```

---

## 2. Agent Lifecycle

### A `ChatClientAgent` run

An agent is stateless; the conversation lives in an `AgentSession` that the caller keeps (or a session store keeps for it) between runs.

```mermaid
graph TD
    Start((CreateSessionAsync)) --> Idle
    Idle -- RunAsync(userInput) --> Processing
    
    subgraph Processing
        direction LR
        A[Get History] --> B{LLM Inference}
        B -- Tool Needed? --> C{Select Tool}
        C -- Execute --> D[Tool Output]
        D --> B
        B -- No Tool --> E[Generate Response]
    end

    Processing -- Returns AgentResponse --> Idle
    Idle -- "RunAsync(...)" --> Processing
    Idle -- SerializeSessionAsync --> Stored((Session stored))
    Stored -- DeserializeSessionAsync --> Idle
```

---

## 3. Multi-Agent Orchestration

### Sequential Workflow

Each agent receives the conversation so far and adds its reply (`AgentWorkflowBuilder.BuildSequential`); with `chainOnlyAgentResponses` it sees only the previous agent's reply. See the [orchestrations page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_orchestrations/).

```mermaid
graph LR
    Input --> AgentA[Agent A<br/><i>e.g., Researcher</i>]
    AgentA -- OutputA --> AgentB[Agent B<br/><i>e.g., Summariser</i>]
    AgentB -- Final Output --> Result
```

### Router Pattern

A router classifies the request (structured output, or a switch edge in a workflow) and one specialist answers. For specialists that pass the conversation between themselves, use a handoff workflow instead.

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
        B(Short-Term Memory<br/><i>AgentSession: the current conversation</i>)
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
        E[Azure Cosmos DB / Blob Storage<br/><i>Sessions, history, checkpoints</i>]
        F[Microsoft Foundry<br/><i>Hosted agents, evaluation</i>]
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
    A --> F
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

    Client->>AuthServer: Sign in (OAuth 2.0 / Entra ID)
    AuthServer-->>Client: Returns JWT

    Client->>AgentAPI: Invoke Agent (with JWT in Header)
    AgentAPI->>AgentAPI: Validate JWT (RequireAuthorization)
    alt Is Valid
        AgentAPI->>AgentAPI: Isolation key from the sub claim, load that user's session
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
    B --> C{Session store<br/><i>AgentSessionStore</i>}
    C -- Get or create session --> D[State backend<br/><i>e.g. Azure Blob, Redis</i>]
    D -- Returns session --> C
    C --> E{Agent Logic}
    E -- "Needs knowledge" --> F[Memory System<br/><i>e.g., Azure AI Search</i>]
    F -- "Returns relevant docs" --> E
    E -- "Needs tool" --> G[Tool Executor]
    G -- "Invokes C# method" --> H(External API / DB)
    H -- "Returns data" --> G
    G -- "Returns tool output" --> E
    E -- "LLM prompt with context" --> I[LLM Service<br/><i>Azure OpenAI</i>]
    I -- "LLM completion" --> E
    E -- "Final response" --> J{Save session}
    J --> D
    E -- "Final response" --> K[Format Output]
    K --> A
```

---

## 10. Workflow Supersteps

### How a workflow run proceeds

Messages sent during a superstep are delivered at the start of the next; executors that receive messages in the same superstep run concurrently, and a checkpoint can be taken at the end of each. See the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/).

```mermaid
sequenceDiagram
    participant Caller
    participant Runner as InProcessExecution
    participant A as Executor A
    participant B as Executor B
    participant C as Executor C
    participant Port as RequestPort

    Caller->>Runner: RunStreamingAsync(workflow, input, checkpoints)
    Note over Runner: Superstep 0
    Runner->>A: input
    A-->>Runner: SendMessageAsync (fan-out)
    Runner-->>Caller: SuperStepCompletedEvent (checkpoint)
    Note over Runner: Superstep 1
    par
        Runner->>B: message
    and
        Runner->>C: message
    end
    B-->>Runner: QueueStateUpdateAsync, SendMessageAsync
    C-->>Runner: SendMessageAsync
    Note over Runner: Superstep 2
    Runner->>Port: message
    Port-->>Caller: RequestInfoEvent
    Caller->>Runner: SendResponseAsync(response)
    Note over Runner: Superstep 3
    Runner->>A: response routed on
    A-->>Caller: WorkflowOutputEvent
```

---

## 11. Hosting Endpoints

### One registration, several protocols

Agents and workflows registered with `AddAIAgent` / `AddWorkflow` are keyed services; each `Map...` call exposes them in one protocol. See the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/).

```mermaid
graph LR
    subgraph Clients
        O[OpenAI SDK clients]
        P[Agents on other frameworks]
        U[CopilotKit / AG-UI front end]
        D[Developer in a browser]
    end

    subgraph App[ASP.NET Core app]
        R[MapOpenAIResponses / MapOpenAIChatCompletions]
        A2[MapA2AJsonRpc / MapA2AHttpJson]
        AG[MapAGUIServer]
        DV[MapDevUI<br/><i>development only</i>]
        K[(Keyed AIAgent services<br/>AddAIAgent, AddWorkflow.AddAsAIAgent)]
        S[(Session store<br/>isolated per user)]
    end

    O --> R --> K
    P --> A2 --> K
    U --> AG --> K
    D --> DV --> R
    K --> S
```

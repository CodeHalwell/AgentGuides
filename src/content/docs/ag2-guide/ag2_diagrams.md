---
title: "AG2 Architecture Diagrams"
description: "How an AG2 1.x turn, tool loop, human-input pause and Network workflow channel fit together."
framework: "ag2"
language: "python"
---

> Diagrams describe the `ag2` %%version=ag2%% (1.x) architecture. Classic (`autogen`) group chats are shown only for comparison.

## One turn: `agent.ask()` and the tool loop

```mermaid
sequenceDiagram
    participant App
    participant Agent
    participant Model as Model provider
    participant Tool
    participant Stream as MemoryStream

    App->>Agent: await agent.ask("...")
    Agent->>Model: messages + tool schemas
    Model-->>Agent: ToolCallEvent
    Agent-->>Stream: ToolCallEvent
    Agent->>Tool: execute (Context injected)
    Tool-->>Agent: ToolResultEvent
    Agent->>Model: messages + tool result
    Model-->>Agent: ModelResponse (final)
    Agent-->>Stream: ModelResponse
    Agent-->>App: AgentReply (.body, await .content())
```

## Human in the loop

```mermaid
sequenceDiagram
    participant Model
    participant Agent
    participant Tool
    participant Hook as hitl_hook
    participant Human

    Model-->>Agent: call publish_plan(plan)
    Agent->>Tool: run
    Tool->>Agent: await context.input("Approve?")
    Agent->>Hook: HumanInputRequest
    Hook->>Human: prompt (CLI / UI / queue)
    Human-->>Hook: "yes"
    Hook-->>Agent: HumanMessage
    Agent-->>Tool: "yes"
    Tool-->>Agent: "Published."
```

## Network: hub, agents and a workflow channel

```mermaid
graph TD
    Hub["Hub<br/>registry · WAL · audit"] --- Store[(KnowledgeStore<br/>memory / disk / sqlite / redis)]
    Researcher[researcher] -->|register| Hub
    Writer[writer] -->|register| Hub
    Editor[editor] -->|register| Hub
    Researcher -->|"open(type=workflow, graph)"| Channel{{workflow channel}}
    Channel -->|"TransitionGraph.sequence"| Writer
    Writer --> Channel
    Channel --> Editor
    Editor --> Channel
    Channel -->|EV_CHANNEL_CLOSED| Hub
```

## Classic GroupChat vs 1.x workflow channel

```mermaid
graph LR
    subgraph Classic["AG2 Classic (autogen)"]
        M[GroupChatManager] --> A1[ConversableAgent A]
        M --> B1[ConversableAgent B]
        A1 --> M
        B1 --> M
    end
    subgraph New["ag2 1.x"]
        H[Hub + WorkflowAdapter] --> A2[Agent A]
        H --> B2[Agent B]
        G[TransitionGraph<br/>JSON-serialisable] -.-> H
    end
```

## Interop surfaces

```mermaid
graph LR
    MCPS[(MCP servers)] -->|MCPToolkit| Agent
    Agent -->|"A2AServer.build_jsonrpc / build_rest / build_grpc"| A2AClients[A2A clients]
    RemoteA2A[(Remote A2A agent)] -->|"A2AConfig(card_url=...)"| Agent
    Agent -->|"ag2.mcp.MCPServer"| MCPClients[MCP clients]
    Agent -->|ag2.ag_ui| UI[AG-UI front-ends]
```

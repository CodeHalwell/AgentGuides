---
title: "Mistral Agents API: Comprehensive Visual Architecture Guide"
description: "Agent lifecycle, conversation flow, streaming, handoff and tool-execution diagrams for Mistral's Agents and Conversations APIs."
framework: mistral-agents-api
---

Diagrams of the Agents and Conversations APIs. Endpoint names, entry types and event names match `mistralai` %%version=mistralai%% (checked %%versions-checked%%). Parts 5 to 7 sketch how such a service is typically built and deployed; they are illustrations, not a description of Mistral's internal systems.

---

## Part 1: Comprehensive Diagrams

### 1. Agent Lifecycle Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LIFECYCLE                          │
└─────────────────────────────────────────────────────────────────┘

CREATION PHASE
──────────────
    │
    ▼
┌──────────────────────────┐
│   POST /v1/agents        │
├──────────────────────────┤
│ - Model selection        │
│ - Instructions/Prompt    │
│ - Tools configuration    │
│ - Completion parameters  │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│   Agent Created          │
│   (version 0)            │
├──────────────────────────┤
│ agent_id: ag_xxx         │
│ status: active           │
│ version: 0               │
└──────────────────────────┘

USAGE PHASE
──────────
    │
    ├─► GET /v1/agents/{id}          → Retrieve agent metadata
    │
    ├─► PATCH /v1/agents/{id}        → Update (creates a new version)
    │
    └─► POST /v1/conversations       → Start conversations
            │
            ├─► POST /v1/conversations/{id}          → Continue
            │
            ├─► POST /v1/conversations/{id}/restart  → Branch
            │
            └─► GET /v1/conversations/{id}/history   → Retrieve history

DELETION
────────
    │
    ▼
┌──────────────────────────┐
│   Agent Deleted          │
│   DELETE /v1/agents/{id} │
├──────────────────────────┤
│ Old conversations still  │
│ accessible and           │
│ replayable               │
└──────────────────────────┘
```

### 2. Conversation Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATION FLOW                            │
└─────────────────────────────────────────────────────────────────┘

CLIENT                                    MISTRAL SERVER
──────                                    ──────────────

User Query
    │
    ▼
┌─────────────┐
│POST /convers│
│ations      │
├─────────────┤
│ agent_id    │
│ inputs      │
│ stream: true│
└─────────────┘
    │
    │────────────────────────┐
    │                        ▼
    │              ┌──────────────────────┐
    │              │  Load Agent Config   │
    │              │  + instructions      │
    │              │  + tools            │
    │              │  + parameters       │
    │              └──────────────────────┘
    │                        │
    │                        ▼
    │              ┌──────────────────────┐
    │              │  Load Conversation  │
    │              │  History (if exists)│
    │              └──────────────────────┘
    │                        │
    │                        ▼
    │              ┌──────────────────────┐
    │              │  LLM Forward Pass    │
    │              │  + system prompt     │
    │              │  + history           │
    │              │  + new input         │
    │              └──────────────────────┘
    │                        │
    │              ┌─────────┴──────────┐
    │              ▼                    ▼
    │        ┌──────────┐        ┌──────────────┐
    │        │ Response │        │ Tool Call(s) │
    │        │ Generation        │ (optional)   │
    │        └──────────┘        └──────────────┘
    │              │                    │
    │              │              ┌─────┴──────────┐
    │              │              ▼                ▼
    │              │         ┌────────┐    ┌──────────────┐
    │              │         │Execute │    │ Execute      │
    │              │         │Tool 1  │    │ Tool N       │
    │              │         └────────┘    └──────────────┘
    │              │              │                │
    │              │              └────────┬───────┘
    │              │                       ▼
    │              │          ┌─────────────────────────┐
    │              │          │ Tool Results            │
    │              │          │ Appended to History     │
    │              │          └─────────────────────────┘
    │              │                       │
    │              │          ┌────────────┴──────────────┐
    │              │          │                           │
    │              │          ▼                           ▼
    │              │    ┌─────────────┐          ┌──────────────┐
    │              │    │ More Tools? │          │ Final Output │
    │              │    │ (loop back) │          │ Generation   │
    │              │    └─────────────┘          └──────────────┘
    │              │           │                        │
    │              └───────────┼────────────────────────┘
    │                          ▼
    │              ┌──────────────────────┐
    │              │ Store Entry in DB:   │
    │              │ - Role: assistant    │
    │              │ - Content            │
    │              │ - Tool executions    │
    │              │ - Metadata           │
    │              └──────────────────────┘
    │
    │◄─ SSE: event: conversation.response.started
    │◄─ SSE: event: message.output.delta
    │◄─ SSE: event: message.output.delta
    │   ... (streaming chunks)
    │◄─ SSE: event: conversation.response.done
    ▼
Display Response
```

### 3. Multi-Agent Orchestration Pattern

```
┌────────────────────────────────────────────────────────────────┐
│         MULTI-AGENT ORCHESTRATION WITHOUT FRAMEWORKS            │
└────────────────────────────────────────────────────────────────┘

SEQUENTIAL PIPELINE
───────────────────
    User Query
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              AGENT 1: Data Cleaner                      │
    │  Raw Data  → Clean & Normalize → Cleaned Data         │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              AGENT 2: Analyzer                          │
    │  Cleaned Data → Extract Insights → Analysis Results    │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              AGENT 3: Report Writer                     │
    │  Analysis → Format Report → Final Report               │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    User Gets Final Report


PARALLEL PROCESSING
───────────────────
                     User Query
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
        ┌─────┐        ┌─────┐        ┌─────┐
        │Agent│        │Agent│        │Agent│
        │ 1   │        │ 2   │        │ 3   │
        └─────┘        └─────┘        └─────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                  ┌────────────────┐
                  │  Synthesize    │
                  │  Results       │
                  └────────────────┘
                          │
                          ▼
                    Final Output


HIERARCHICAL STRUCTURE
──────────────────────
                      Manager Agent
                      (Coordinator)
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         Specialist 1   Specialist 2   Specialist 3
         (Research)     (Analysis)     (Reporting)
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                     Synthesized Output
```

### 4. Tool Execution Workflow

```
┌────────────────────────────────────────────────────────────┐
│              TOOL EXECUTION WORKFLOW                        │
└────────────────────────────────────────────────────────────┘

LLM Generation
      │
      ▼
┌──────────────────────────┐
│ Decision: Tool needed?   │
└──────────────────────────┘
      │
      ├─ No ──→ Generate Response
      │         │
      │         ▼
      │    Return to User
      │
      └─ Yes ──→ Tool Call Generation
                 │
                 ▼
            ┌──────────────────────┐
            │ Extract Tool Name &  │
            │ Parameters           │
            └──────────────────────┘
                 │
                 ▼
            ┌──────────────────────┐
            │ Validate Parameters  │
            │ Against Schema       │
            └──────────────────────┘
                 │
                 ├─ Invalid ──→ Error Response
                 │              │
                 │              ▼
                 │         Inform Agent
                 │
                 └─ Valid ──→ Execute Tool
                            │
                    ┌───────┴────────┐
                    ▼                ▼
                ┌────────┐      ┌───────────┐
                │Success │      │Exception  │
                └────────┘      └───────────┘
                    │                │
                    ▼                ▼
            ┌──────────────┐    ┌──────────────┐
            │Tool Result   │    │Error Message │
            └──────────────┘    └──────────────┘
                    │                │
                    └────────┬────────┘
                             ▼
                    ┌──────────────────────┐
                    │ Append to Conversation
                    │ History              │
                    └──────────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Continue LLM Forward │
                    │ Pass with Result    │
                    └──────────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ More tools needed?   │
                    └──────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                   Yes               No
                    │                 │
          (Loop back to         (Generate Final
           Tool Call)            Response)
```

### 5. Request/Response Processing Pipeline

```
┌────────────────────────────────────────────────────────────┐
│           REQUEST/RESPONSE PROCESSING                      │
└────────────────────────────────────────────────────────────┘

CLIENT SIDE
───────────
  ┌────────────────────┐
  │ Application Logic  │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Format Request:    │
  │ - agent_id         │
  │ - inputs           │
  │ - stream: bool     │
  │ - store: bool      │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ HTTP Request       │
  │ POST /conversations│
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Streaming Handler  │
  │ (if stream=true)   │
  └────────────────────┘

API GATEWAY
───────────
           │
           ▼
  ┌────────────────────┐
  │ Authentication     │
  │ (API Key Verify)   │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Rate Limiting      │
  │ (Quota Check)      │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Route to Worker    │
  └────────────────────┘

WORKER PROCESSING
─────────────────
           │
           ▼
  ┌────────────────────┐
  │ Load Agent Config  │
  │ (from DB)          │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Load Conversation  │
  │ History (if exists)│
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Execute LLM        │
  │ Pipeline           │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Process Tool Calls │
  │ (if any)           │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Store in Database  │
  │ - Entry data       │
  │ - Metadata         │
  │ - Timing info      │
  └────────────────────┘

RESPONSE DELIVERY
─────────────────
           │
           ▼
  ┌────────────────────┐
  │ Stream Events?     │
  └────────────────────┘
           │
    ┌──────┴──────┐
    │             │
   Yes           No
    │             │
    ▼             ▼
┌────────┐   ┌─────────┐
│SSE     │   │Standard │
│Chunked │   │Response │
└────────┘   └─────────┘
    │             │
    └──────┬──────┘
           ▼
  ┌────────────────────┐
  │ Return to Client   │
  │ - conversation_id  │
  │ - outputs[]        │
  │ - usage tokens     │
  └────────────────────┘
           │
           ▼
CLIENT APPLICATION
```

### 6. Memory Persistence Architecture

```
┌────────────────────────────────────────────────────────────┐
│            MEMORY PERSISTENCE ARCHITECTURE                 │
└────────────────────────────────────────────────────────────┘

CONVERSATION LIFECYCLE & STORAGE
─────────────────────────────────

Turn 1: User Input
    │
    ▼
┌──────────────────────────┐
│ VOLATILE (In Processing) │
├──────────────────────────┤
│ - Received: "Hello"      │
│ - In LLM pipeline        │
└──────────────────────────┘
    │
    ▼
LLM Processing
    │
    ▼
┌──────────────────────────┐
│ Generated Response       │
│ + Tool Executions        │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│         PERSISTENT DATABASE                   │
├──────────────────────────────────────────────┤
│ Conversation Entry:                          │
│ ├─ ID: msg_xxx                              │
│ ├─ Type: message.input                      │
│ ├─ Role: user                               │
│ ├─ Content: "Hello"                         │
│ ├─ Created_at: timestamp                    │
│ └─ Conversation_id: conv_yyy                │
│                                              │
│ Conversation Entry:                          │
│ ├─ ID: msg_yyy                              │
│ ├─ Type: message.output                     │
│ ├─ Role: assistant                          │
│ ├─ Content: "Hi! How can I help?"           │
│ ├─ Created_at: timestamp                    │
│ ├─ Completed_at: timestamp                  │
│ └─ Agent_id: ag_zzz                         │
└──────────────────────────────────────────────┘

Turn 2: Continuation (Hours/Days Later)
    │
    ▼
┌──────────────────────────┐
│ New user input: "..."    │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ LOAD FULL HISTORY                            │
├──────────────────────────────────────────────┤
│ GET /conversations/{conv_id}/history        │
│   Returns: All entries (chronological)       │
│   ├─ User input 1                           │
│   ├─ Assistant response 1                   │
│   ├─ Tool execution 1                       │
│   ├─ User input 2                           │
│   └─ ... (all history)                      │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Context Reconstructed    │
│ Full conversation loaded │
└──────────────────────────┘
    │
    ▼
LLM Processes
New Input + Full History
    │
    ▼
┌──────────────────────────────────────────────┐
│ NEW ENTRIES STORED                           │
├──────────────────────────────────────────────┤
│ ├─ User input (entry)                       │
│ ├─ Assistant output (entry)                 │
│ └─ Tool executions (entries)                │
└──────────────────────────────────────────────┘

CONVERSATION BRANCHES
──────────────────────
                  Original Conv
                       │
           ┌───────────┼───────────┐
           │           │           │
        Turn 1       Turn 2      Turn 3
           │           │           │
           ▼           ▼           ▼
      [Entry]     [Entry]     [Entry]
                       │
              Restart from Turn 1
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
      [Entry]                  [New Branch]
      (Resume)                 (Alternative
                                Path)
```

### 7. Conversation State Management

```
┌────────────────────────────────────────────────────────────┐
│           CONVERSATION STATE MACHINE                        │
└────────────────────────────────────────────────────────────┘

States:
─────
[CREATED]
    │
    ├─→ new conversation initialized
    ├─→ agent linked
    ├─→ ready for first turn
    │
    ▼
[ACTIVE]
    │
    ├─→ processing messages
    ├─→ generating responses
    ├─→ executing tools
    │
    ▼
[AWAITING_INPUT]
    │
    ├─→ response completed
    ├─→ waiting for user input
    ├─→ context maintained
    │
    ▼
[PROCESSING]
    │
    ├─→ executing LLM forward pass
    ├─→ may call tools
    ├─→ generating response
    │
    ▼
[COMPLETED_TURN]
    │
    ├─→ response generated
    ├─→ entries stored
    ├─→ ready for next turn or restart
    │
    ▼
[DELETED]
    │
    ├─→ DELETE /v1/conversations/{id}
    ├─→ history no longer retrievable

Transitions:
───────────
CREATED → ACTIVE (on first message)
ACTIVE ↔ AWAITING_INPUT (normal conversation flow)
Any → PROCESSING (when processing message)
PROCESSING → AWAITING_INPUT/COMPLETED_TURN
Any → DELETED (explicit deletion)
COMPLETED_TURN → new conversation (via restart from an entry)
```

### 8. API Integration Points

```
┌────────────────────────────────────────────────────────────┐
│              MISTRAL AGENTS API ENDPOINTS                  │
└────────────────────────────────────────────────────────────┘

AGENT MANAGEMENT
────────────────
┌──────────────────────────────────────────────┐
│ POST   /v1/agents                            │
│ GET    /v1/agents                            │
│ GET    /v1/agents/{agent_id}                 │
│ PATCH  /v1/agents/{agent_id}                 │
│ DELETE /v1/agents/{agent_id}                 │
└──────────────────────────────────────────────┘

CONVERSATION MANAGEMENT
───────────────────────
┌──────────────────────────────────────────────┐
│ POST   /v1/conversations (start new)          │
│ POST   /v1/conversations/{conv_id}           │
│        (continue/append)                     │
│ POST   /v1/conversations/{conv_id}/restart   │
│        (restart from entry)                  │
│ GET    /v1/conversations (list all)          │
│ GET    /v1/conversations/{conv_id}           │
│        (get metadata)                        │
│ GET    /v1/conversations/{conv_id}/history   │
│        (full entries)                        │
│ GET    /v1/conversations/{conv_id}/messages  │
│        (only messages, filtered)             │
└──────────────────────────────────────────────┘

STREAMING ENDPOINTS
───────────────────
┌──────────────────────────────────────────────┐
│ POST   /v1/conversations?stream=true         │
│ POST   /v1/conversations/{conv_id}?stream=true
│ POST   /v1/conversations/{conv_id}/restart?  │
│        stream=true                           │
│                                              │
│ Returns: Server-Sent Events (SSE) stream     │
│ Events:                                      │
│  - conversation.response.started             │
│  - message.output.delta                      │
│  - tool.execution.started                    │
│  - tool.execution.done                       │
│  - conversation.response.done                │
└──────────────────────────────────────────────┘
```

### 9. Agent Configuration Schema

```
┌────────────────────────────────────────────────────────────┐
│          AGENT CONFIGURATION STRUCTURE                     │
└────────────────────────────────────────────────────────────┘

Agent Object:
{
  "id": "ag_xxxxxxxxxxxx",           # Unique identifier
  "name": "string",                   # Human-readable name
  "description": "string",            # Purpose description
  "model": "mistral-medium-3-5",      # Model to use
  "instructions": "string",           # System prompt
  "version": 1,                       # Version number
  "created_at": "2026-10-21T...",     # Creation timestamp
  "updated_at": "2026-10-21T...",     # Last update
  "tools": [                          # Available tools
    {"type": "web_search"},
    {"type": "function", "function": {...}},
    ...
  ],
  "completion_args": {                # Model parameters
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": 2048,
    "presence_penalty": 0,
    "frequency_penalty": 0
  },
  "handoffs": [...]                   # Agent handoff config
}
```

---

## Part 2: Sequence Diagrams

### Web Search Agent Interaction

```
Client          Mistral API        Agent Logic       Web Search     LLM
  │                 │                  │               API          │
  ├─ POST /conversations           │                  │           │
  │ {agent_id, inputs}             │                  │           │
  │─────────────────────────────────>                 │           │
  │                 │                  │               │           │
  │                 ├──Load Agent Config              │           │
  │                 ├──Load Conversation History      │           │
  │                 │                                  │           │
  │                 │  Agent Routing                  │           │
  │                 ├─────────────────────────────────────────────>
  │                 │                                  │        Process
  │                 │                                  │        w/Context
  │                 │                                  │           │
  │                 │                                  │  <─────────
  │                 │                         Needs web search    │
  │                 │                                  │           │
  │                 ├─────────────────web_search()─────────────────>
  │                 │                      │           │           │
  │                 │                      │           Search      │
  │                 │                      │           Executed    │
  │                 │                      │<──────────────────────
  │                 │                      Results                 │
  │                 │                      │           │           │
  │                 │  Append to History   │           │           │
  │                 ├────────────────────────────────────────────────>
  │                 │                      │           │        Continue
  │                 │                      │           │        w/Results
  │                 │                      │           │           │
  │                 │                      │           │  <─────────
  │                 │                   Final Response │           │
  │                 │                      │           │           │
  │  <─────────────Response────────────────            │           │
  │                 │                      │           │           │
```

### Multi-Agent Handoff Pattern

```
Client          Manager Agent      Specialist 1    Specialist 2    DB
  │                 │                  │               │           │
  ├─ Query ─────────────────────────────>             │           │
  │                 │                                  │           │
  │                 ├─Analyze Request──>              │           │
  │                 │                                  │           │
  │                 │<─Needs Research───              │           │
  │                 │                                  │           │
  │                 ├─ Delegate to Specialist 1 ──────>           │
  │                 │                                  │           │
  │                 │                   Research ─────────────────>
  │                 │                     Done        │           │
  │                 │                  <──────────────            
  │                 │                 ┌────────────┐              │
  │                 │                 │ Store      │              │
  │                 │                 │ Results    │─────────────>
  │                 │                 └────────────┘              │
  │                 │                   Results                   │
  │                 │                   <──────────              
  │                 │<─Research Complete─────────               │
  │                 │                                 │           │
  │                 ├─ Delegate to Specialist 2 ──────────────────>
  │                 │                 Analyse                 │
  │                 │                     <──────────────────────
  │                 │<─Analysis Complete──────                   │
  │                 │                                 │           │
  │                 ├─ Synthesize Results ────────────>        │
  │                 │                    Combine              │
  │                 │                    <───────────           │
  │                 │                                 │           │
  │<──Final Report─────────────────────────────────  │           │
  │                 │                 │               │           │
```

### Streaming Conversation Flow


```
Client                          Mistral Server
  │                                  │
  ├─ POST /conversations?stream=true │
  │────────────────────────────────────>
  │                                  │
  │                          Process Request
  │                                  │
  │<─ SSE: conversation.response.started
  │                                  │
  │<─ SSE: message.output.delta "The"│
  │                                  │
  │<─ SSE: message.output.delta " weather"
  │                                  │
  │<─ SSE: message.output.delta " is sunny"
  │                                  │
  │                          (Tool needed)
  │                                  │
  │<─ SSE: tool.execution.started    │
  │   {name: "web_search"}           │
  │                                  │
  │                    Execute tool in background
  │                                  │
  │<─ SSE: tool.execution.done       │
  │   {name: "web_search"}           │
  │                                  │
  │<─ SSE: message.output.delta " and 25°C"
  │                                  │
  │<─ SSE: conversation.response.done│
  │   {tokens: {...}}                │
  │                                  │
```


### Persistent Memory Retrieval

```
Turn 1 (Day 1):
  Client ─ "What's your name?" ─> Agent
  Agent ─ "I'm Alice" ──────────> DB
  
Turn 2 (Day 2):
  Client ─ "What did you say earlier?" ─> Agent
           │
           ├─ Load Conversation History from DB
           │  ├─ Entry 1: User: "What's your name?"
           │  └─ Entry 2: Agent: "I'm Alice"
           │
           ├─ Include in Context
           │  "Remember the conversation history:
           │   User: What's your name?
           │   Agent: I'm Alice"
           │
           └─ Process with LLM
           
  Agent ─ "I said my name is Alice" ──> Client
```

---

## Part 3: Data Structure Diagrams

### Conversation Entry Object

```
ConversationEntry {
  ├─ id: string (unique)
  │  └─ msg_xxxxxxxxx
  │
  ├─ type: string
  │  ├─ message.input      (user message)
  │  ├─ message.output     (agent response)
  │  ├─ tool.execution     (built-in tool ran)
  │  ├─ function.call      (your function requested)
  │  ├─ function.result    (your function's result)
  │  └─ agent.handoff      (conversation moved to another agent)
  │
  ├─ role: string
  │  ├─ user
  │  ├─ assistant
  │  └─ tool
  │
  ├─ content: string | object | array
  │  └─ May include tool references
  │
  ├─ created_at: ISO8601 timestamp
  ├─ completed_at: ISO8601 timestamp (optional)
  │
  ├─ agent_id: string (optional)
  │  └─ ag_xxxxxxxxx
  │
  ├─ model: string (optional)
  │  └─ mistral-medium-3-5
  │
  ├─ conversation_id: string
  │  └─ conv_xxxxxxxxx
  │
  └─ metadata: object (optional)
     └─ Custom fields
}
```

### Tool Definition Structure

```
Tool {
  ├─ type: string
  │  ├─ web_search         (web search)
  │  ├─ web_search_premium (Premium search)
  │  ├─ code_interpreter   (Python sandbox)
  │  ├─ image_generation   (images as files)
  │  ├─ document_library   (RAG over Libraries)
  │  ├─ connector          (registered MCP server)
  │  └─ function           (Custom tool)
  │
  ├─ (if type == "function")
  │  └─ function {
  │      ├─ name: string
  │      │  └─ "get_weather"
  │      │
  │      ├─ description: string
  │      │  └─ "Get current weather..."
  │      │
  │      └─ parameters: JSONSchema
  │         ├─ type: "object"
  │         ├─ properties: {
  │         │  └─ location: {type: "string"}
  │         │
  │         └─ required: ["location"]
  │     }
  │
  └─ (if type == "web_search")
     └─ No parameters needed
}
```

---

## Part 4: Processing Pipeline Flows

### Request Processing Pipeline

```
REQUEST → VALIDATION → AUTH → RATE LIMIT → QUEUE → WORKER
    │         │         │        │          │        │
    ├─> Parse  ├─> Check ├─> API  ├─> Check ├─> Route ├─> Load
    │   JSON   │  Format │ Key    │  Quota  │  to    │  Agent
    │          │         │        │         │  Pool  │
    └─────────────────────────────────────────────────────>
                                                      │
                                                      ▼
    <─ RESPONSE ◄─ SERIALIZE ◄─ PROCESS ◄─ GENERATE ◄─ STREAM
         │           │            │           │
         ├─> Format  ├─> JSON    ├─> LLM    ├─> Tools
         │   Headers │  Encode   │  Forward  │  Exec
         └────────────────────────────────────>
```

### LLM Forward Pass with Tools

```
┌─ System Prompt (Agent Instructions)
├─ Conversation History (Full Context)
├─ New User Input
└─ Tools Specification
         │
         ▼
┌──────────────────────┐
│  LLM Processing      │
│  (Attention passes)  │
└──────────────────────┘
         │
    ┌────┴─────┐
    │           │
    ▼           ▼
 Response   Tool Call
    │           │
    │       ┌───┴────────────────┐
    │       ▼                    ▼
    │    Valid Schema?       Error
    │       │                  │
    │      Yes               Return
    │       │                Error
    │       ▼
    │  Execute Tool
    │       │
    │       ▼
    │  Tool Result
    │       │
    │       ▼
    │  Append to History
    │       │
    │       ▼
    │  Loop back to LLM?
    │       │
    │    ┌──┴──┐
    │    │     │
    │   Yes   No
    │    │     │
    └────┼─────┘
         │
         ▼
    Return Output
```

---

## Part 5: System Architecture

### High-Level System Design

```
┌──────────────────────────────────────────────────────────────┐
│                        CLIENT APPLICATIONS                    │
│ (Web, Mobile, Desktop, CLI, etc.)                            │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    MISTRAL API GATEWAY                       │
│ ├─ Authentication (API Key verification)                   │
│ ├─ Rate Limiting (Token bucket algorithm)                  │
│ ├─ Request Validation (Schema validation)                  │
│ └─ Routing (Direct to appropriate service)                 │
└──────────────────────────────────────────────────────────────┘
         │                   │                  │
         ▼                   ▼                  ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐
│   Agents Service ││ Conversations    ││  Tools Service   │
├──────────────────┤├──────────────────┤├──────────────────┤
│ ├─ Agent CRUD   ││ ├─ Start         ││ ├─ Web Search    │
│ ├─ Versioning  ││ ├─ Continue      ││ ├─ Code Exec     │
│ ├─ Config      ││ ├─ Restart       ││ ├─ Image Gen     │
│ └─ Metadata    ││ ├─ History       ││ ├─ Doc Library   │
│                ││ └─ List          ││ └─ Custom Tools  │
└────────┬────────┘└────────┬─────────┘└────────┬─────────┘
         │                  │                   │
         └──────────────────┼───────────────────┘
                            ▼
         ┌──────────────────────────────────────┐
         │      LLM EXECUTION ENGINE            │
         │ ├─ Model Selection                  │
         │ ├─ Context Assembly                 │
         │ ├─ Token Counting                   │
         │ └─ Streaming Output                 │
         └──────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    ┌─────────┐        ┌──────────┐      ┌─────────┐
    │mistral- │        │mistral-  │      │mistral- │
    │small    │        │medium    │      │large    │
    └─────────┘        └──────────┘      └─────────┘
    (Model Pool / Load Balancer)
         │
         └──────────────────┬──────────────────┐
                            ▼                  ▼
                      ┌─────────────┐  ┌────────────────┐
                      │Database     │  │Cache Layer     │
                      │├─ Agents    │  │├─ Conversation │
                      │├─ Conversat │  ││  Context      │
                      ││ ions       │  │├─ Agent Config │
                      │├─ Entries   │  │└─ Tool Results │
                      │└─ Metadata  │  └────────────────┘
                      └─────────────┘
```

### Conversation State Database Schema

```
AGENTS TABLE
────────────
id (PK)                    │ VARCHAR
name                       │ VARCHAR
description                │ TEXT
model                      │ VARCHAR
instructions               │ TEXT
version                    │ INT
created_at                 │ TIMESTAMP
updated_at                 │ TIMESTAMP
tools (JSON)               │ JSONB
completion_args (JSON)     │ JSONB
owner_id                   │ VARCHAR (FK)
status                     │ ENUM


CONVERSATIONS TABLE
───────────────────
id (PK)                    │ VARCHAR
agent_id (FK)              │ VARCHAR
created_at                 │ TIMESTAMP
updated_at                 │ TIMESTAMP
name                       │ VARCHAR
description                │ TEXT
owner_id                   │ VARCHAR (FK)
status                     │ ENUM
metadata (JSON)            │ JSONB


CONVERSATION_ENTRIES TABLE
──────────────────────────
id (PK)                    │ VARCHAR
conversation_id (FK)       │ VARCHAR
type                       │ ENUM
role                       │ ENUM
content                    │ TEXT/JSONB
created_at                 │ TIMESTAMP
completed_at               │ TIMESTAMP
agent_id                   │ VARCHAR (FK)
model                      │ VARCHAR
tool_name                  │ VARCHAR
tool_call_id               │ VARCHAR
metadata (JSON)            │ JSONB
parent_entry_id            │ VARCHAR (FK, nullable)
token_count                │ INT


INDEXES
───────
idx_conv_agent             │ CONVERSATIONS(agent_id)
idx_entries_conv           │ ENTRIES(conversation_id)
idx_entries_created        │ ENTRIES(created_at)
idx_owner_conv             │ CONVERSATIONS(owner_id)
idx_agent_owner            │ AGENTS(owner_id)
```

---

## Part 6: Error Handling Flow

```
REQUEST PROCESSING
        │
        ▼
    ┌─────────┐
    │ Error?  │
    └────┬────┘
         │
     ┌───┴───┐
     │       │
    No      Yes
     │       │
     ▼       ▼
  Continue ┌──────────────────────────┐
           │ Error Type?              │
           └──────────┬───────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
      ┌─────┐    ┌────────┐    ┌──────────┐
      │Auth │    │Validation  │   Rate   │
      │Error│    │Error       │  Limit   │
      └──┬──┘    └────┬───────┘  └────┬───┘
         │            │               │
         ▼            ▼               ▼
      401        422           429
      Unauthorized Invalid       Too Many
                  Request        Requests
         │            │            │
         └────────────┼────────────┘
                      ▼
             ┌──────────────────────┐
             │ Format Error         │
             │ Response             │
             │ {                    │
             │  "status": "code",   │
             │  "error": "message", │
             │  "type": "type_code" │
             │ }                    │
             └──────────────────────┘
                      │
                      ▼
             Return to Client
```

---

## Part 7: Deployment Architecture

### Single-Region Deployment

```
┌─────────────────────────────────────────────────────┐
│                    REGION: us-east-1                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │        Load Balancer / API Gateway         │   │
│  └────────────────────────────────────────────┘   │
│                     │                              │
│  ┌──────────────────┼──────────────────────────┐  │
│  │                  │                          │  │
│  ▼                  ▼                          ▼  │
│ ┌──────┐         ┌──────┐                ┌──────┐ │
│ │Worker│         │Worker│      ...       │Worker│ │
│ │Pool-1│         │Pool-2│                │Pool-N│ │
│ └──────┘         └──────┘                └──────┘ │
│  (Auto-scaling)                                    │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │    Distributed Cache (Redis Cluster)      │   │
│  │  - Session state                          │   │
│  │  - Rate limit counters                    │   │
│  │  - Frequently accessed config              │   │
│  └────────────────────────────────────────────┘   │
│                     │                              │
│                     ▼                              │
│  ┌────────────────────────────────────────────┐   │
│  │      Primary Database (PostgreSQL)         │   │
│  │  - Agents                                  │   │
│  │  - Conversations                           │   │
│  │  - Entries                                 │   │
│  └────────────────────────────────────────────┘   │
│           │               │                        │
│           ▼               ▼                        │
│      ┌──────────┐    ┌───────────┐               │
│      │ Hot      │    │ Warm      │               │
│      │ Replicas │    │ Backups   │               │
│      └──────────┘    └───────────┘               │
│                                                   │
└─────────────────────────────────────────────────┘
```

---

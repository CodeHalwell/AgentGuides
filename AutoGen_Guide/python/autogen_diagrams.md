# AG2 Architecture and Design Diagrams

**Visual Representations of AG2 Patterns and Workflows**

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Agent Communication Patterns](#agent-communication-patterns)
3. [Multi-Agent Orchestration](#multi-agent-orchestration)
4. [Nested Chats Architecture](#nested-chats-architecture)
5. [Sequential Chat Workflow](#sequential-chat-workflow)
6. [Group Chat Speaker Selection](#group-chat-speaker-selection)
7. [Tool Integration Flow](#tool-integration-flow)
8. [Code Execution Models](#code-execution-models)
9. [State Management Architecture](#state-management-architecture)
10. [Production Deployment Patterns](#production-deployment-patterns)

---

## Core Architecture

### AG2 Framework Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      AG2 Framework                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Agent Communication Layer                  │   │
│  │                                                         │   │
│  │  ConversableAgent ←────────────────→ ConversableAgent  │   │
│  │  (Core base class for all agents)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Specialised Agent Types                         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  AssistantAgent         UserProxyAgent  GroupChatMgr   │   │
│  │  (LLM-based problem     (Human proxy +  (Multi-agent   │   │
│  │   solving)              code execution)  orchestration)│   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Message & Execution Layer                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  ┌──────────────────┐  ┌──────────────────────────┐   │   │
│  │  │ LLM Providers    │  │  Code Execution         │   │   │
│  │  │ • OpenAI         │  │  • Local execution      │   │   │
│  │  │ • Anthropic      │  │  • Docker execution     │   │   │
│  │  │ • Google Gemini  │  │  • Custom executors     │   │   │
│  │  │ • Ollama         │  │  • Jupyter notebooks    │   │   │
│  │  └──────────────────┘  └──────────────────────────┘   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Integration & Protocol Layer                    │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  Function Registry  |  A2A Protocol  |  RAG System     │   │
│  │  (Tool calling)     |  (REST APIs)   |  (Documents)    │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Hierarchy

```
                          ConversableAgent
                                 △
                    ┌────────────┼────────────┐
                    │            │            │
            AssistantAgent  UserProxyAgent  GroupChatManager
            
  - LLM-enabled        - Code execution    - Multi-agent
  - Problem solving    - Human-in-loop     - Orchestration
  - Message replies    - External tools    - Speaker selection
```

---

## Agent Communication Patterns

### Two-Agent Conversation Flow

```
┌─────────────────────────────────────────┐
│         Two-Agent Conversation          │
└─────────────────────────────────────────┘

                 Agent A
                   △
                   │ (1) Send message
                   │
                   ↓
                 Agent B
                   △
                   │ (2) Process & reply
                   │
                   ↓
                 Agent A
                   △
                   │ (3) Review reply
                   │
                   ↓
                 Agent B
                   ...
                   
Termination:
- max_turns reached
- Termination condition met
- TERMINATE keyword found
```

### Agent Message Flow

```
          ┌─────────────────────────────────────┐
          │      Agent A initiates_chat          │
          └─────────────────────────────────────┘
                          │
                          ↓
            ┌─────────────────────────────┐
            │  Message Sent to Agent B    │
            │  • Content                  │
            │  • Role (assistant/user)    │
            │  • Name                     │
            │  • Timestamp                │
            └─────────────────────────────┘
                          │
                          ↓
            ┌─────────────────────────────┐
            │  Agent B Processes Message  │
            │  1. Extract content         │
            │  2. Check termination       │
            │  3. Generate reply          │
            └─────────────────────────────┘
                          │
                          ↓
            ┌─────────────────────────────┐
            │  Reply Sent to Agent A      │
            │  • Generated content        │
            │  • Role = "assistant"       │
            │  • Name of Agent B          │
            └─────────────────────────────┘
                          │
                          ↓
            ┌─────────────────────────────┐
            │  Conversation Summary       │
            │  Chat history, messages,    │
            │  summary text               │
            └─────────────────────────────┘
```

---

## Multi-Agent Orchestration

### Group Chat Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  GroupChatManager                       │
│  (Central orchestration point)                          │
└─────────────────────────────────────────────────────────┘
                          △
                          │ Coordinates
                    ┌─────┴─────────┬─────────┐
                    │               │         │
                    ↓               ↓         ↓
                ┌────────┐    ┌────────┐  ┌────────┐
                │Agent A │    │Agent B │  │Agent C │
                │Specialist   Specialist Specialist
                │#1       │    │#2       │  │#3       │
                └────────┘    └────────┘  └────────┘

Speaker Selection Loop:
1. Analyse chat context
2. Determine next speaker (auto/round-robin/manual)
3. Request message from selected agent
4. Broadcast to all agents
5. Check termination condition
6. Repeat
```

### Speaker Selection Strategies

```
┌──────────────────────────────────────────────────┐
│      Speaker Selection Methods                   │
├──────────────────────────────────────────────────┤
│                                                  │
│  AUTO (LLM-based)                                │
│  ┌─────────────────────────────────────┐        │
│  │ LLM decides best next speaker       │        │
│  │ Based on conversation context       │        │
│  │ Intelligent turn taking             │        │
│  └─────────────────────────────────────┘        │
│          ↓                                       │
│  Next Speaker: Dynamically selected             │
│                                                  │
│  ROUND-ROBIN                                     │
│  ┌─────────────────────────────────────┐        │
│  │ Agent A → Agent B → Agent C → A...  │        │
│  │ Deterministic turn sequence         │        │
│  │ Predictable execution               │        │
│  └─────────────────────────────────────┘        │
│          ↓                                       │
│  Next Speaker: Sequential rotation              │
│                                                  │
│  RANDOM                                          │
│  ┌─────────────────────────────────────┐        │
│  │ Randomly select next speaker        │        │
│  │ Unpredictable dialogue flow         │        │
│  │ Less control, more variation        │        │
│  └─────────────────────────────────────┘        │
│          ↓                                       │
│  Next Speaker: Random selection                 │
│                                                  │
│  MANUAL                                          │
│  ┌─────────────────────────────────────┐        │
│  │ Custom function selects speaker     │        │
│  │ Full programmatic control           │        │
│  │ Complex orchestration logic         │        │
│  └─────────────────────────────────────┘        │
│          ↓                                       │
│  Next Speaker: Custom logic result              │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Nested Chats Architecture

### Hierarchical Chat Triggering

```
                    ┌─────────────────────────┐
                    │    Main Conversation    │
                    │   Agent A ←→ Agent B    │
                    └─────────────────────────┘
                              │
                              │ (Trigger condition met)
                              ↓
                    ┌─────────────────────────┐
                    │   Nested Chat Level 1   │
                    │   Agent B ←→ Reviewer   │
                    └─────────────────────────┘
                              │
                              │ (Summary: "code quality ok")
                              ↓
                    ┌─────────────────────────┐
                    │   Nested Chat Level 2   │
                    │   Reviewer ←→ Improver  │
                    └─────────────────────────┘
                              │
                              │ (Summary: "improvements applied")
                              ↓
                    ┌─────────────────────────┐
                    │  Resume Main Conversation│
                    │   (Incorporate results)  │
                    └─────────────────────────┘
```

### Nested Chat Trigger Conditions

```
Message Generated
       │
       ↓
┌──────────────────────────┐
│ Check Trigger Condition  │
│ (Function: is_triggered) │
└──────────────────────────┘
       │
       ├─ YES ──→ ┌─────────────────────────┐
       │          │ Execute Nested Chat     │
       │          │ • Queue configured      │
       │          │ • Message prepared      │
       │          │ • Summary method set    │
       │          │ • Max turns limited     │
       │          └─────────────────────────┘
       │                    │
       │                    ↓
       │          ┌─────────────────────────┐
       │          │ Embed Results in Main   │
       │          │ Chat Context            │
       │          └─────────────────────────┘
       │
       └─ NO ──→ Continue Main Conversation
```

---

## Sequential Chat Workflow

### Multi-Stage Pipeline

```
Stage 1: Research
┌─────────────────────────┐
│  Coordinator → Researcher │
│  "Research topic X"       │
└─────────────────────────┘
         │
         ↓ Summary: "Key findings..."
         │
         ↓ Carryover: Use findings

Stage 2: Analysis
┌─────────────────────────┐
│  Coordinator → Analyst    │
│  "Analyse findings"       │
└─────────────────────────┘
         │
         ↓ Summary: "Analysis results..."
         │
         ↓ Carryover: Use analysis

Stage 3: Writing
┌─────────────────────────┐
│  Coordinator → Writer     │
│  "Write report"           │
└─────────────────────────┘
         │
         ↓ Summary: "Report draft..."
         │
         ↓ Carryover: Review draft

Stage 4: Editing
┌─────────────────────────┐
│  Coordinator → Editor     │
│  "Edit report"            │
└─────────────────────────┘
         │
         ↓ Final output
```

---

## Group Chat Speaker Selection

### Auto Selection Decision Tree

```
                    ┌─────────────────────┐
                    │ Current Turn Ended  │
                    └─────────────────────┘
                              │
                              ↓
                    ┌─────────────────────┐
                    │ Check Termination   │
                    │ Conditions          │
                    └─────────────────────┘
                         YES│
                    ┌──────┴──────┐
                    │ END         │
                    └─────────────┘
                         NO│
                    ┌──────┴──────────────────────┐
                    │ Build context string        │
                    │ (Recent messages, names)    │
                    └─────────────────────────────┘
                              │
                              ↓
                    ┌─────────────────────┐
                    │ Send to LLM with    │
                    │ prompt:             │
                    │ "Who should speak?" │
                    └─────────────────────┘
                              │
                              ↓
                    ┌─────────────────────┐
                    │ Parse LLM response  │
                    │ Extract agent name  │
                    └─────────────────────┘
                              │
                              ↓
                    ┌─────────────────────┐
                    │ Select agent        │
                    │ Request message     │
                    └─────────────────────┘
                              │
                              ↓
                    ┌─────────────────────┐
                    │ Continue loop       │
                    └─────────────────────┘
```

---

## Tool Integration Flow

### Function Registration and Execution

```
┌─────────────────────────────────────────────────────────┐
│           Tool Integration Flow                         │
└─────────────────────────────────────────────────────────┘

1. DEFINITION
   ┌──────────────────────────┐
   │ Define function          │
   │ with type hints          │
   │ and docstring            │
   └──────────────────────────┘
           │
           ↓

2. REGISTRATION
   ┌──────────────────────────┐
   │ register_function()      │
   │ • Function               │
   │ • Caller agent           │
   │ • Executor agent         │
   │ • Description            │
   └──────────────────────────┘
           │
           ↓

3. DISCOVERY
   ┌──────────────────────────┐
   │ LLM discovers tool       │
   │ through schema           │
   │ (Auto-generated)         │
   └──────────────────────────┘
           │
           ↓

4. INVOCATION
   ┌──────────────────────────┐
   │ LLM decides to call tool │
   │ • Generates parameters   │
   │ • Formats call           │
   └──────────────────────────┘
           │
           ↓

5. EXECUTION
   ┌──────────────────────────┐
   │ Executor runs function   │
   │ • Input validation       │
   │ • Error handling         │
   │ • Return result          │
   └──────────────────────────┘
           │
           ↓

6. FEEDBACK
   ┌──────────────────────────┐
   │ Result sent to LLM       │
   │ • Success or error       │
   │ • Continue generation    │
   └──────────────────────────┘
```

### Tool Execution Modes

```
┌───────────────────────────────────────────────┐
│        Tool Execution Architecture            │
└───────────────────────────────────────────────┘

┌─────────────────────────────┐
│  Caller Agent (LLM-based)   │
│  • Decides when to call     │
│  • Generates parameters     │
│  • Processes results        │
└─────────────────────────────┘
         ↑        │
         │        │ Request with params
         │        ↓
         │  ┌─────────────────────────┐
         │  │  Executor Agent         │
         │  │  • Receives request     │
         │  │  • Executes function    │
         │  │  • Returns result       │
         │  └─────────────────────────┘
         │        │
         └────────┘ Result

Two possibilities:
A) Executor = same as caller (single agent with function)
B) Executor = different agent (specialised executor)
```

---

## Code Execution Models

### Local vs Docker Execution

```
┌──────────────────────────────────────────────────────────┐
│              Code Execution Models                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  LOCAL EXECUTION                                         │
│  ┌──────────────────────────────────────┐               │
│  │ Assistant generates code             │               │
│  │           │                          │               │
│  │           ↓                          │               │
│  │ Local command-line executor          │               │
│  │ • Direct file system access          │               │
│  │ • Same Python environment            │               │
│  │ • Fast execution                     │               │
│  │ • Security: Full system access       │               │
│  │           │                          │               │
│  │           ↓                          │               │
│  │ Result returned to agent             │               │
│  └──────────────────────────────────────┘               │
│                                                          │
│  DOCKER EXECUTION (Recommended)                          │
│  ┌──────────────────────────────────────┐               │
│  │ Assistant generates code             │               │
│  │           │                          │               │
│  │           ↓                          │               │
│  │ Docker command-line executor         │               │
│  │ • Spawn container                    │               │
│  │ • Image: python:3.11-slim (etc.)     │               │
│  │ • Isolated filesystem                │               │
│  │ • Resource limits (memory, CPU)      │               │
│  │ • Security: Sandboxed environment    │               │
│  │           │                          │               │
│  │           ↓                          │               │
│  │ Container executes code              │               │
│  │           │                          │               │
│  │           ↓                          │               │
│  │ Capture stdout/stderr                │               │
│  │           │                          │               │
│  │           ↓                          │               │
│  │ Return output to agent               │               │
│  │           │                          │               │
│  │           ↓                          │               │
│  │ Container cleaned up                 │               │
│  └──────────────────────────────────────┘               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## State Management Architecture

### Context Variables and Persistence

```
┌─────────────────────────────────────────────┐
│         State Management in AG2             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │     ContextVariables()               │  │
│  │  (Shared state dictionary)           │  │
│  ├──────────────────────────────────────┤  │
│  │                                      │  │
│  │  context.set("key", value)           │  │
│  │  context.get("key")                  │  │
│  │  context.update(dict)                │  │
│  │                                      │  │
│  └──────────────────────────────────────┘  │
│           △  △  △  △  △  △                  │
│           │  │  │  │  │  │                  │
│    ┌──────┘  │  │  │  │  └──────┐          │
│    │         │  │  │  │         │          │
│    ↓         ↓  ↓  ↓  ↓         ↓          │
│  Agent1    Agent2 Agent3 ... AgentN       │
│  (Reads/   (Reads/ (Reads/ (Reads/        │
│   Writes)   Writes) Writes) Writes)       │
│                                             │
└─────────────────────────────────────────────┘

Scope: Multi-agent conversations, group chats
Lifetime: Duration of agents and conversations
Access: Thread-safe dictionary operations
```

---

## Production Deployment Patterns

### REST API Deployment (A2A Protocol)

```
┌──────────────────────────────────────────────────┐
│         REST API Deployment Architecture         │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │        Client Application                │   │
│  │  (External service, web app, etc.)       │   │
│  └──────────────────────────────────────────┘   │
│              │                                   │
│              │ HTTP/REST Calls                   │
│              │ POST /message                     │
│              │ GET /status                       │
│              ↓                                   │
│  ┌──────────────────────────────────────────┐   │
│  │      A2A Server                          │   │
│  │  (REST API wrapper)                      │   │
│  │  • Parse HTTP requests                   │   │
│  │  • Route to agent                        │   │
│  │  • Return JSON responses                 │   │
│  └──────────────────────────────────────────┘   │
│              │                                   │
│              ↓                                   │
│  ┌──────────────────────────────────────────┐   │
│  │      AG2 Agent                           │   │
│  │  • Process messages                      │   │
│  │  • Generate responses                    │   │
│  │  • Maintain state                        │   │
│  └──────────────────────────────────────────┘   │
│              │                                   │
│              ↓                                   │
│  ┌──────────────────────────────────────────┐   │
│  │    External Resources                    │   │
│  │  • APIs, databases, tools                │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Docker Deployment Stack

```
┌────────────────────────────────────────────────────┐
│           Docker Deployment Stack                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  Load Balancer (nginx/HAProxy)                     │
│          │                                         │
│  ┌───────┴───────────────────────┐                │
│  │                               │                │
│  ↓                               ↓                │
│ ┌──────────────────┐  ┌──────────────────┐       │
│ │  AG2 Container 1 │  │  AG2 Container 2 │  ...  │
│ │  (Port 8000)     │  │  (Port 8001)     │       │
│ │  • Agent A       │  │  • Agent B       │       │
│ │  • Agent B       │  │  • Tools         │       │
│ │  • Tools         │  │                  │       │
│ └──────────────────┘  └──────────────────┘       │
│        │                      │                   │
│  ┌─────┴──────────────────────┴─────┐            │
│  │      Shared Services              │            │
│  ├───────────────────────────────────┤            │
│  │ • Redis (cache/sessions)          │            │
│  │ • PostgreSQL (persistence)        │            │
│  │ • Message Queue (async tasks)     │            │
│  └───────────────────────────────────┘            │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Complex Workflow Integration

### End-to-End Content Creation System

```
┌─────────────────────────────────────────────────────────────┐
│         Content Creation System (Recipe #4)                 │
│                    Sequential Pipeline                      │
└─────────────────────────────────────────────────────────────┘

Phase 1: Ideation
┌──────────────────┐
│ Idea Generator   │  ──→ Generate 3 blog post angles
│                  │      [Ideas exchanged between agents]
└──────────────────┘
         │
         ↓ Summary: Selected angle
         │
Phase 2: Research
┌──────────────────┐
│ Researcher       │  ──→ Gather facts, statistics, insights
│                  │      [Deep research phase]
└──────────────────┘
         │
         ↓ Summary: Research findings
         │
Phase 3: Writing
┌──────────────────┐
│ Writer           │  ──→ Create 800-word blog post
│                  │      [Incorporate research findings]
└──────────────────┘
         │
         ↓ Summary: First draft
         │
Phase 4: Editing
┌──────────────────┐
│ Editor           │  ──→ Polish and refine
│                  │      [Grammar, clarity, engagement]
└──────────────────┘
         │
         ↓
   Final Blog Post

Carryover pattern allows context from each phase to inform the next!
```

---

## Summary: Architecture at a Glance

| Component | Role | Examples |
|-----------|------|----------|
| **Agents** | Communication endpoints | ConversableAgent, AssistantAgent |
| **Orchestrators** | Multi-agent coordination | GroupChat, GroupChatManager |
| **Execution** | Task processing | LocalExecutor, DockerExecutor |
| **LLM Layer** | Intelligence | OpenAI, Anthropic, Gemini, Ollama |
| **Tools** | Extended capabilities | Functions, APIs, external services |
| **State** | Shared context | ContextVariables, chat history |
| **Deployment** | Production readiness | REST API, Docker, async patterns |

This architectural flexibility allows AG2 to scale from simple two-agent conversations to complex enterprise systems with dozens of specialised agents, multiple orchestration layers, and integrated external systems.

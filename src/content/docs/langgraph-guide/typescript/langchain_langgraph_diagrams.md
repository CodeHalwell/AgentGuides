---
title: "LangChain.js and LangGraph.js Diagrams and Visualisations"
description: "Comprehensive visual representations of architectures, workflows, and patterns for TypeScript-based LLM applications."
framework: "langgraph"
language: "typescript"
sidebar:
  label: "Diagrams"
  order: 45
---

Comprehensive visual representations of architectures, workflows, and patterns for TypeScript-based LLM applications.

---

## Architecture Diagrams

### LangChain.js Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                             │
│  (Agents, Chains, Retrievers, Memory Systems)                       │
└──────┬─────────────────────────────────────────────────────────────┘
       │
┌──────┴─────────────────────────────────────────────────────────────┐
│              LangChain.js Core (@langchain/core)                    │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  Chat Models    │  │  Prompt      │  │  Output Parsers     │ │
│  │  - OpenAI       │  │  Templates   │  │  - JSON Parser      │ │
│  │  - Anthropic    │  │  - Variables │  │  - Structured output│ │
│  │  - Google       │  │  - Templates │  │  - String Parser    │ │
│  └─────────────────┘  └──────────────┘  └─────────────────────┘ │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  Tools           │  │  Retrievers  │  │  Memory Systems     │ │
│  │  - Structured    │  │  - Vector DB │  │  - Checkpointers    │ │
│  │  - Dynamic       │  │  - Document  │  │    (thread state)   │ │
│  │  - Custom        │  │  - Search    │  │    (@langchain/     │ │
│  │                  │  │              │  │     langgraph-      │ │
│  │                  │  │              │  │     checkpoint)     │ │
│  │                  │  │              │  │  - Stores (long-term)│ │
│  └──────────────────┘  └──────────────┘  └─────────────────────┘ │
│                                                                   │
│  LCEL (LangChain Expression Language)                             │
│  - Composition | Streaming | Invoking | Batching                │
└──────────────────────────────────────────────────────────────────┘
       │
┌──────┴─────────────────────────────────────────────────────────────┐
│              LLM Provider Integrations                              │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  OpenAI    │  │Anthropic│  │  Google  │  │  Other Providers │ │
│  │  API       │  │  API    │  │ Vertex   │  │  (Ollama, etc)   │ │
│  └────────────┘  └─────────┘  └──────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### LangGraph.js Execution Model

```
┌────────────────────────────────────────────────────────────────┐
│                     StateGraph Execution                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input State                                                  │
│       │                                                        │
│       ▼                                                        │
│  ┌─────────────┐                                              │
│  │   START     │ (Special node)                              │
│  └──────┬──────┘                                              │
│         │                                                      │
│         ▼                                                      │
│  ┌──────────────────────┐                                     │
│  │  Node 1              │  ┌──────────────────────┐           │
│  │  TypeScript Function │──▶│  Node 2              │           │
│  │  Process & Update    │  │  TypeScript Function │           │
│  │  State               │  │  Process & Update    │           │
│  └────────┬─────────────┘  │  State               │           │
│           │                └──┬───────────────────┘           │
│           │ (Conditional Edge)│                               │
│           │                   │                               │
│           ├──────────────┬────┘                               │
│           │              │                                     │
│           ▼              ▼                                     │
│  ┌──────────────┐  ┌──────────────┐                           │
│  │ Node 3       │  │ Node 4       │                           │
│  │ Alternative  │  │ Alternative  │                           │
│  │ Path         │  │ Path         │                           │
│  └──────┬───────┘  └──────┬───────┘                           │
│         │                 │                                   │
│         └────────┬────────┘                                   │
│                  ▼                                            │
│           ┌─────────────┐                                     │
│           │     END     │ (Special node)                      │
│           └─────────────┘                                     │
│                  │                                            │
│                  ▼                                            │
│           Output State                                        │
└────────────────────────────────────────────────────────────────┘
```

### Agent Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Lifecycle                          │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   1. Initialize  │
│   ────────────── │
│   - Create Model │
│   - Set Tools    │
│   - createAgent  │
│     (+ middleware)│
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  2. Think                    │
│  ────────────────────────────│
│  LLM analyzes input          │
│  Decides next action         │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  3. Action Selection         │
│  ────────────────────────────│
│  Choose tool from available: │
│  - Tool 1                    │
│  - Tool 2                    │
│  - Tool 3                    │
│  - or no tool call (answer)  │
└────────┬─────────────────────┘
         │
      ┌──┴──────────────────────┐
      │                         │
      ▼                         ▼
┌──────────────────┐  ┌──────────────────┐
│ Action == TOOL   │  │ Action == ANSWER │
│ ────────────────│  │ ──────────────────│
│ 4. Invoke Tool  │  │ 5. Return Result │
│ Get Result      │  │ and Exit         │
└────────┬─────────┘  └──────────────────┘
         │                    │
         ▼                    │
┌──────────────────┐         │
│ 6. Observation  │         │
│ ────────────────│         │
│ Add tool result │         │
│ to context      │         │
└────────┬─────────┘         │
         │                   │
         └───────┬───────────┘
                 │
         Repeat until the model replies without tool calls
                 │
                 ▼
         ┌──────────────────┐
         │  7. Complete     │
         │  ──────────────  │
         │  Return output   │
         └──────────────────┘
```

---

## State Flow Diagrams

### Simple State Progression

```
User Input
    │
    ▼
┌──────────────────────────────────┐
│ State: {                         │
│   input: string                  │
│   processing: false              │
│   output: null                   │
│ }                                │
└──────────────────────────────────┘
    │
    ▼
    Validate → Parse → Process
    │          │          │
    ▼          ▼          ▼
┌──────────────────────────────────┐
│ State: {                         │
│   input: "processed"             │
│   processing: true               │
│   output: null                   │
│ }                                │
└──────────────────────────────────┘
    │
    ▼
    Generate → Format
    │          │
    ▼          ▼
┌──────────────────────────────────┐
│ State: {                         │
│   input: "processed"             │
│   processing: false              │
│   output: "result"               │
│ }                                │
└──────────────────────────────────┘
    │
    ▼
Output to User
```

### Conditional Branching State Flow

```
                    Initial State
                         │
                         ▼
                  ┌───────────────┐
                  │ Classify Input│
                  └───────┬───────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Priority│      │ Question│      │ Feedback│
    │ Request │      │         │      │         │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
         ▼                ▼                ▼
    Priority        Escalate to         Route to
    Processing      Knowledge Base      Analytics
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Format Output │
                  └───────┬───────┘
                          │
                          ▼
                   Return to User
```

### Multi-Agent State Evolution

```
      Initial State
          │
          ▼
┌─────────────────────────────────┐
│ {                               │
│   task: "Analyse market trends" │
│   agentResponses: {}            │
│   status: "pending"             │
│ }                               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Agent 1 (Research) Executes     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ {                               │
│   task: "Analyse market trends" │
│   agentResponses: {             │
│     research: "2024 trends..."  │
│   }                             │
│   status: "research_complete"   │
│ }                               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Agent 2 (Analysis) Executes     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ {                               │
│   task: "Analyse market trends" │
│   agentResponses: {             │
│     research: "2024 trends...", │
│     analysis: "Key findings..." │
│   }                             │
│   status: "analysis_complete"   │
│ }                               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Agent 3 (Synthesis) Executes    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ {                               │
│   task: "Analyse market trends" │
│   agentResponses: {             │
│     research: "2024 trends...", │
│     analysis: "Key findings...",│
│     synthesis: "Final report"   │
│   }                             │
│   status: "complete"            │
│ }                               │
└─────────────────────────────────┘
```

---

## Agent Communication Patterns

### Supervisor Pattern

```
                    Input Task
                         │
                         ▼
                  ┌──────────────┐
                  │  Supervisor  │ Determines which
                  │   Decides    │ agent to route to
                  └──────┬───────┘
                         │
         ┌───────────┬───┴───┬───────────┐
         │           │       │           │
         ▼           ▼       ▼           ▼
    ┌───────┐  ┌────────┐  ┌─────────┐  ┌─────┐
    │Agent 1│  │Agent 2 │  │Agent 3  │  │FINAL│
    │Spec.1 │  │Spec.2  │  │Spec.3   │  │ANSW.│
    └───┬───┘  └───┬────┘  └────┬────┘  └──┬──┘
        │          │            │          │
        └─────┬────┴────┬───────┴──────────┘
              │         │
              ▼         ▼
          ┌────────────────────┐
          │ Supervisor Routes  │
          │ back to agents     │
          │ for refinement     │
          └────────────────────┘
              │
              ▼
         ┌──────────────┐
         │ Final Output │
         └──────────────┘
```

### Hierarchical Multi-Agent System

```
                        ┌───────────────────┐
                        │  Top Coordinator  │
                        └─────────┬─────────┘
                                  │
                 ┌────────────┬────┴────┬────────────┐
                 │            │         │            │
                 ▼            ▼         ▼            ▼
            ┌────────┐    ┌────────┐  ┌────────┐  ┌──────────┐
            │Manager │    │Manager │  │Manager │  │Manager   │
            │Squad 1 │    │Squad 2 │  │Squad 3 │  │Squad 4   │
            └───┬────┘    └───┬────┘  └───┬────┘  └────┬─────┘
                │             │           │            │
          ┌─────┴──────┐  ┌──┴──────┐  ┌─┴──────┐  ┌──┴──────┐
          │ │ │        │  │ │ │     │  │ │ │    │  │ │ │ │   │
          ▼ ▼ ▼        ▼  ▼ ▼ ▼     ▼  ▼ ▼ ▼    ▼  ▼ ▼ ▼ ▼   ▼
        ┌─┐┌─┐┌─┐    ┌─┐┌─┐┌─┐   ┌─┐┌─┐┌─┐   ┌─┐┌─┐┌─┐┌─┐
        │W││W││W│    │W││W││W│   │W││W││W│   │W││W││W││W│
        └─┘└─┘└─┘    └─┘└─┘└─┘   └─┘└─┘└─┘   └─┘└─┘└─┘└─┘
        Worker     Worker      Worker      Worker
        Nodes      Nodes       Nodes       Nodes

        W = Individual worker agent
```

### Sequential Agent Handoff

```
Start Task
    │
    ▼
┌──────────────┐
│ Agent A      │ Starts processing
│ (Planning)   │
└──────┬───────┘
       │
       ├─ Executes planning phase
       │
       ▼
   Agent A determines it needs Agent B
       │
       ▼
┌──────────────┐
│ Handoff to   │ Passes context/state
│ Agent B      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Agent B      │ Receives context
│ (Analysis)   │ Continues from A
└──────┬───────┘
       │
       ├─ Executes analysis phase
       │
       ▼
   Agent B determines it needs Agent C
       │
       ▼
┌──────────────┐
│ Handoff to   │ Passes enriched context
│ Agent C      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Agent C      │ Receives full context
│ (Execution)  │ Completes task
└──────┬───────┘
       │
       ▼
   Final Result
```

---

## Memory and State Persistence Architecture

### Memory System Architecture

LangGraph has two memory layers. (The older LangChain `BufferMemory` / `ConversationSummaryMemory` classes are legacy and are not part of `@langchain/core`.)

```
┌────────────────────────────────────────────────────────────┐
│                 Your graph / createAgent agent             │
└───────────────┬──────────────────────────────┬─────────────┘
                │                              │
                ▼                              ▼
   ┌────────────────────────────┐  ┌─────────────────────────────┐
   │ Checkpointer (short-term)  │  │ Store (long-term)           │
   │ state of one thread,       │  │ JSON documents by namespace │
   │ saved every super-step     │  │ shared across threads,      │
   │ keyed by thread_id         │  │ optional vector search      │
   │                            │  │                             │
   │ MemorySaver (tests)        │  │ InMemoryStore (dev)         │
   │ SqliteSaver / PostgresSaver│  │ PostgresStore               │
   └────────────────────────────┘  └─────────────────────────────┘
        compile({ checkpointer })        compile({ store })
```

Trimming or summarising long histories is done in the graph (a node, or `summarizationMiddleware` on `createAgent`), not by a memory class.

### Checkpoint Persistence Architecture

```
┌──────────────────────────────────────────────────────────┐
│            LangGraph Execution                           │
└──────────────────────────────────────────────────────────┘

Graph Compilation
    │
    ▼
Execution with Checkpointer
    │
    ├─ Super-step 1 (Node 1) → save checkpoint
    │
    ├─ Super-step 2 (Nodes 2 and 3, parallel) → save checkpoint
    │
    ├─ ...
    │
    └─ Complete

Checkpoint Storage Options:

┌────────────────────────────────────────────────────┐
│         Checkpoint Storage Backends                │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────┐  ┌──────────────────────┐      │
│  │ MemorySaver  │  │ SqliteSaver          │      │
│  │ ────────────│  │ ────────────────────  │      │
│  │ • In-memory │  │ • Local SQLite DB     │      │
│  │ • Fast      │  │ • Persistent local    │      │
│  │ • Volatile  │  │ • Good for dev        │      │
│  └──────────────┘  └──────────────────────┘      │
│                                                    │
│  ┌─────────────────────┐  ┌───────────────────┐  │
│  │ PostgresSaver       │  │ Custom Saver      │  │
│  │ ───────────────────│  │ ──────────────────│  │
│  │ • PostgreSQL DB     │  │ • User-defined    │  │
│  │ • Production ready  │  │ • Any backend     │  │
│  │ • Scalable          │  │ • Full control    │  │
│  └─────────────────────┘  └───────────────────┘  │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## RAG Architecture

### Retrieval-Augmented Generation Pipeline

```
User Query
    │
    ▼
┌────────────────────────────────┐
│ Query Embedding                │
│ Convert text to vector         │
└────────┬─────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Vector Similarity Search       │
│ Find similar documents         │
└────────┬─────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Retrieve Documents             │
│ Top-K results from VectorStore │
└────────┬─────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Context Assembly               │
│ Combine retrieved docs with    │
│ original query                 │
└────────┬─────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ LLM Generation                 │
│ Generate response using        │
│ context and query              │
└────────┬─────────────────────┘
         │
         ▼
     Final Response

    Data Flow:
    ────────
    Documents
         │
         ▼
    ┌─────────────────┐
    │ Text Splitter   │
    │ Chunk documents │
    └────┬────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Embedding Model │
    │ Generate vectors│
    └────┬────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Vector Store    │
    │ Store & Index   │
    │ (Pinecone,      │
    │  Chroma,        │
    │  Weaviate,      │
    │  Supabase)      │
    └─────────────────┘
```

### Multi-Query Retrieval Pattern

```
        Query
          │
          ├─────────────────────────┐
          │                         │
          ▼                         ▼
    Original Query          Query Expansion
                            via LLM
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
                Query 1      Query 2      Query 3
                
                    │            │            │
                    └────────────┼────────────┘
                                 │
                                 ▼
                       Vector Search (All 3)
                                 │
                                 ▼
                    ┌────────────┬────────────┐
                    │            │            │
                    ▼            ▼            ▼
                Results-1   Results-2   Results-3
                    │            │            │
                    └────────────┼────────────┘
                                 │
                                 ▼
                    Deduplicate & Rerank
                                 │
                                 ▼
                         Final Results
```

---

## Control Flow Diagrams

### Conditional Edge Routing

```
    State
      │
      ▼
  ┌───────────────────────┐
  │ Routing Function      │
  │ Evaluates state       │
  │ Determines next node  │
  └─────────┬─────────────┘
            │
     ┌──────┴──────────────────┐
     │ Return routing decision │
     │                         │
     └──┬──┬──┬──┬──┬──┬──┬──┘
        │  │  │  │  │  │  │
        ▼  ▼  ▼  ▼  ▼  ▼  ▼
      Path1 Path2 Path3 ...

Routing Logic Example:
────────────────────

if (state.priority === 'high') {
  return 'urgent_handler';
} else if (state.type === 'question') {
  return 'qa_agent';
} else if (state.error_count > 3) {
  return END; // the END constant ('__end__'), not the string 'END'
} else {
  return 'default_handler';
}
```

### Loop Detection and Control

```
┌──────────────┐
│ Enter Loop   │
│ iteration: 0 │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Check MAX_ITERATIONS │
│ iteration < max      │◄─────────────────┐
└──────┬───────────────┘                  │
       │                                  │
    YES│                                  │
       ▼                                  │
┌──────────────┐                          │
│ Execute Node │                          │
│ iteration ++  │                         │
└──────┬───────┘                          │
       │                                  │
       ▼                                  │
┌──────────────────────┐                  │
│ Conditional Edge     │                  │
│ Check continue cond. │                  │
└─┬────────────────────┘                  │
  │                                       │
  ├─ true  ──────────────────────────────┘
  │
  └─ false
       │
       ▼
┌──────────────┐
│ Exit Loop    │
│ Return       │
└──────────────┘
```

### Human-in-the-Loop Interruption Flow

```
    Node runs and calls interrupt(payload)
          │
          ▼
    ┌──────────────────────────────┐
    │ Run pauses; checkpoint saved │  needs a checkpointer and a thread_id
    │ invoke() returns state with  │
    │ result[INTERRUPT] = [payload]│
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ Human / caller decides       │  can also inspect getState() or
    │ (approve, edit, reject ...)  │  change it with updateState()
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ invoke(new Command({ resume: │
    │   value }), same thread)     │
    └──────────────┬───────────────┘
                   │
                   ▼
    The interrupted node re-runs from its start;
    interrupt() now returns `value`
                   │
                   ▼
    Graph continues to END (or the next interrupt)
```

---

## Performance and Scaling Diagrams

### Token Management Architecture

```
Application Request
      │
      ▼
┌──────────────────────────┐
│ Token Counter            │
│ Estimate input tokens    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Check Budget             │
│ Used + Estimated <= Max? │
└──────┬────────────────────┘
       │
    YES│
       ▼
┌──────────────────────────┐
│ Execute with LLM         │
│ Stream responses         │
└──────┬────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Count Actual Output      │
│ Tokens                   │
└──────┬────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Update Token Counter     │
│ used += actual_output    │
└──────┬────────────────────┘
       │
       NO│
       ▼
┌──────────────────────────┐
│ Reject Request           │
│ Budget exhausted         │
└──────────────────────────┘
```

### Streaming Architecture

```
Application Request
      │
      ▼
┌──────────────────────────┐
│ Open Connection          │
│ Setup streaming          │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Start LLM Streaming      │
│ Receive tokens           │
└──────┬───────────────────┘
       │
       ▼ (Per Token)
┌──────────────────────────┐
│ Token Received           │
└──────┬───────────────────┘
       │
       ├─ Send to Client
       │
       ├─ Update UI
       │
       └─ Continue...
       │
       ▼ (Final)
┌──────────────────────────┐
│ Stream Complete          │
│ Final result aggregated  │
└──────────────────────────┘
```

---

## Deployment Architecture

### Production Deployment Architecture

```
┌────────────────────────────────────────────────────┐
│              Frontend Layer                        │
│  (Next.js, React, Browser)                        │
└──────────────────────┬─────────────────────────────┘
                       │ HTTP/WebSocket
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
┌──────────────────────────┐  ┌──────────────────┐
│ API Layer                │  │ WebSocket Server │
│ - Express.js             │  │ - Real-time      │
│ - Next.js Routes         │  │ - Streaming      │
│ - Request validation     │  │ - Event updates  │
└──────────┬───────────────┘  └────────┬─────────┘
           │                          │
           └──────────────┬───────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Core Logic   │  │ Integration  │  │ Persistence  │
│ - Agents     │  │ - LLM APIs   │  │ - PostgreSQL │
│ - Chains     │  │ - External   │  │ - Redis      │
│ - Tools      │  │   Services   │  │ - VectorDB   │
└──────────────┘  └──────────────┘  └──────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Monitoring   │  │ Logging      │  │ Observability│
│ - Prometheus │  │ - Winston    │  │ - LangSmith  │
│ - Metrics    │  │ - Structured │  │ - Tracing    │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Type System Diagrams

### Main types

```
StateGraph<State, Update, Nodes>            (builder; node names are tracked in the type)
│   .addNode / .addEdge / .addConditionalEdges / .addSequence
│   .compile({ checkpointer?, store?, cache? })
└── CompiledStateGraph                       (a Pregel runnable)
        invoke / stream / getState / getStateHistory / updateState

createAgent({ model, tools, middleware })    (from "langchain")
└── ReactAgent                               (wraps a compiled LangGraph graph)
        invoke / stream

tool(fn, { name, description, schema })      (from "@langchain/core/tools")
└── StructuredTool / DynamicStructuredTool

State definitions (any one of)
│
├── Annotation.Root({ key: Annotation<T>(...) })
│   ├── Annotation<T>                      (last value wins)
│   └── Annotation<T>({ reducer, default })
│
├── Zod object schema  (z.object({...}))
│
└── new StateSchema({ ... })
    ├── Standard Schema field (Zod 4 / Valibot / ArkType)
    ├── new ReducedValue(schema, { reducer, inputSchema? })
    ├── new UntrackedValue<T>()
    └── MessagesValue
```



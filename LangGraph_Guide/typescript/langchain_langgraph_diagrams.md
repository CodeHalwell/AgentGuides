# LangChain.js and LangGraph.js Diagrams and Visualisations

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
│  │  - Anthropic    │  │  - Variables │  │  - Zod Parser       │ │
│  │  - Google       │  │  - Templates │  │  - String Parser    │ │
│  └─────────────────┘  └──────────────┘  └─────────────────────┘ │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │  Tools           │  │  Retrievers  │  │  Memory Systems     │ │
│  │  - Structured    │  │  - Vector DB │  │  - Buffer Memory    │ │
│  │  - Dynamic       │  │  - Document  │  │  - Window Memory    │ │
│  │  - Custom        │  │  - Search    │  │  - Summary Memory   │ │
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
│   - Configure    │
│   Executor       │
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
│  - FINAL_ANSWER              │
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
         Repeat until FINAL_ANSWER
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

```
┌─────────────────────────────────────────────────────┐
│        Application / Conversation Layer             │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         │             │              │
         ▼             ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌────────────┐
    │ Buffer   │  │ Window   │  │ Summary    │
    │ Memory   │  │ Memory   │  │ Memory     │
    │          │  │          │  │            │
    │ Stores   │  │ Stores   │  │ Stores     │
    │ all msgs │  │ last N   │  │ compressed │
    │          │  │ messages │  │ version    │
    └────┬─────┘  └────┬─────┘  └─────┬──────┘
         │             │              │
         └─────────────┼──────────────┘
                       │
         ┌─────────────┼──────────────────┐
         │             │                  │
         ▼             ▼                  ▼
    ┌──────────┐  ┌──────────────┐  ┌─────────────┐
    │ Entity   │  │ Vector Store │  │ Custom      │
    │ Memory   │  │ Memory       │  │ Memory      │
    │          │  │              │  │             │
    │ Tracks  │  │ Stores       │  │ User-defined│
    │ entities │  │ embeddings   │  │ storage     │
    │ across   │  │ for semantic │  │ logic       │
    │ convs   │  │ search       │  │             │
    └──────────┘  └──────────────┘  └─────────────┘
         │              │                  │
         └──────────────┼──────────────────┘
                        │
         ┌──────────────┼──────────────────┐
         │              │                  │
         ▼              ▼                  ▼
    ┌──────────┐  ┌──────────┐  ┌────────────┐
    │ In-Memory│  │ LangSmith│  │ Persistent │
    │ Storage  │  │ Backend  │  │ Database   │
    │          │  │          │  │            │
    │ Fast but │  │ Observable│  │ Durable    │
    │ volatile │  │ via API   │  │ storage    │
    └──────────┘  └──────────┘  └────────────┘
```

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
    ├─ Execute Node 1 → Save Checkpoint 1
    │
    ├─ Execute Node 2 → Save Checkpoint 2
    │
    ├─ Execute Node 3 → Save Checkpoint 3
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
  return 'END';
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
    Workflow Execution
          │
          ▼
    ┌─────────────┐
    │ Execute     │
    │ Node        │
    └──────┬──────┘
           │
           ▼
    ┌──────────────────┐
    │ Check for        │
    │ interrupt()      │
    │ call?            │
    └────┬────────┬────┘
         │        │
        YES      NO
         │        │
         ▼        │
    ┌──────────┐  │
    │ PAUSE    │  │
    │Execution │  │
    └────┬─────┘  │
         │        │
         ▼        │
    ┌────────────────────────┐
    │ Human Inspector        │
    │ - View state           │
    │ - Make decision        │
    │ - Modify state         │
    │ - Approve/Reject       │
    └────┬───────────────────┘
         │
         ▼
    ┌──────────────────┐
    │ Resume with      │
    │ new state        │
    └────┬─────────────┘
         │
         ├─────────────┐
         │             │
         ▼             ▼
    Continue      Abort
         │             │
         │             ▼
         │         Error State
         │
         ▼
    Continue Execution
         │
         ▼
    Return Result
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

### TypeScript Type Hierarchy for Agents

```
Agent<InputType, OutputType>
│
├── SimpleAgent
│   └── Extends: Agent<string, string>
│
├── StructuredAgent<TInput, TOutput>
│   ├── Extends: Agent<TInput, TOutput>
│   └── Uses: Zod schemas for validation
│
└── MultiAgentOrchestrator<TState>
    ├── Extends: Agent<TState, TState>
    ├── Manages: Agent[]
    └── Routes: Conditional logic

ToolSpecification<TInput, TOutput>
│
├── Tool<TInput, TOutput>
│   ├── name: string
│   ├── description: string
│   ├── schema: ZodSchema<TInput>
│   └── func: (input: TInput) => Promise<TOutput>
│
└── DynamicStructuredTool<TInput, TOutput>
    ├── Extends: Tool<TInput, TOutput>
    ├── Callbacks: RunManager
    └── Error handling: Comprehensive

StateSchema
│
├── Annotation<T>
│   ├── value_type: string
│   ├── default: () => T
│   └── reducer: (x: T, y: T) => T
│
└── Record<key, Annotation<T>>
    └── Used in: StateGraph definition
```

---

This diagrams document provides visual reference for all major architectural patterns, workflows, and system designs in LangChain.js and LangGraph.js applications. Use these diagrams alongside the comprehensive guide for a complete understanding of the system architecture.


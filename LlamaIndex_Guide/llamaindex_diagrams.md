# LlamaIndex Architecture and System Diagrams

Comprehensive visual documentation of LlamaIndex components, workflows, and patterns.

---

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Data Pipeline](#data-pipeline)
3. [RAG Systems](#rag-systems)
4. [Agent Workflows](#agent-workflows)
5. [Multi-Agent Systems](#multi-agent-systems)
6. [Query Processing](#query-processing)
7. [Index Types](#index-types)
8. [Memory and Context](#memory-and-context)
9. [Advanced Patterns](#advanced-patterns)

---

## Core Architecture

### LlamaIndex System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Agents, Query Engines, Chat Interfaces, APIs)              │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐   ┌─────▼─────┐   ┌────▼────┐
    │  Agents │   │   Query   │   │  Chat   │
    │ (ReAct, │   │  Engines  │   │Interface│
    │Function)│   │(Router,   │   │         │
    └────┬────┘   │Sub-Qs)    │   └────┬────┘
         │        └─────┬─────┘        │
         └────────────┬─┴──────────────┘
                      │
         ┌────────────▼──────────────┐
         │  RAG Pipeline & Tools     │
         │  (Retrievers, Rerankers,  │
         │   Postprocessors)         │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │  Indexing & Storage       │
         │  (VectorStoreIndex,       │
         │   TreeIndex, ListIndex)   │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │  Data Layer               │
         │  (Documents, Nodes,       │
         │   Loaders, Embeddings)    │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │ External Integrations     │
         │ (LLMs, Vector Stores,     │
         │  Memory Systems)          │
         └──────────────────────────┘
```

### Component Communication

```
       ┌─────────────┐
       │   User      │
       │  Query      │
       └──────┬──────┘
              │
       ┌──────▼──────────┐
       │  Query Engine   │
       │  (Orchestrator) │
       └──────┬──────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼──┐ ┌───▼──┐ ┌───▼──┐
│Ret.  │ │Post  │ │Output│
│      │ │Proc. │ │Form. │
└───┬──┘ └───┬──┘ └───┬──┘
    │        │        │
    └────┬───┴───┬────┘
         │       │
    ┌────▼──┐ ┌─▼────┐
    │Vector │ │ LLM  │
    │Store  │ │      │
    └───────┘ └──────┘
```

---

## Data Pipeline

### Document Processing Pipeline

```
Input Sources
    │
    ├─ PDFs              ┐
    ├─ Text Files        │
    ├─ Web Pages         ├─▶ Data Loaders  ┐
    ├─ Databases         │    (100+)       │
    ├─ APIs              │                  ├─▶ Documents
    └─ Custom            │                  │   (Structured
                         │                  │    Text + Meta)
                         │                  │
                         └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Text Splitting   │
                    │ (Chunk Strategy) │
                    └──────┬───────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │ Create Nodes     │
                    │ (Add Metadata)   │
                    └──────┬───────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │ Generate         │
                    │ Embeddings       │
                    └──────┬───────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │ Store in Vector  │
                    │ Store & Index    │
                    └──────────────────┘
```

### Node Structure

```
┌─────────────────────────────────┐
│         TextNode                │
├─────────────────────────────────┤
│ • node_id: "unique-id"          │
│ • text: "Content..."            │
│ • embedding: [0.1, 0.2, ...]    │
│ • metadata: {                   │
│     page: 1,                    │
│     source: "doc.pdf",          │
│     section: "intro"            │
│   }                             │
│ • relationships: {              │
│     next: node_2,               │
│     source: doc_1               │
│   }                             │
└─────────────────────────────────┘
```

---

## RAG Systems

### Complete RAG Pipeline

```
┌────────────────────────┐
│  Question/Query        │
└────────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Query Processing   │
    │ (Expansion,        │
    │  Rewriting)        │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Retrieval          │
    │ (Semantic Search   │
    │  + Ranking)        │
    └────────┬───────────┘
             │
    ┌────────▼───────────┐
    │ Retrieved Context  │
    │ + User Query       │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ LLM Processing     │
    │ (Generation)       │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Generated Response │
    │ + Citations        │
    └────────────────────┘
```

### Retrieval Architecture

```
User Query
    │
    ▼
Embedding → Vector
    │
    ▼
Vector Store Search (Similarity)
    │
    ├─▶ Top-K Similar Nodes
    │
    ▼
Postprocessors
    │
    ├─▶ Similarity Threshold Filter
    ├─▶ Reranking (LLM/Semantic)
    ├─▶ Deduplication
    └─▶ Metadata Filtering
    │
    ▼
Final Ranked Nodes
    │
    ▼
Context Building (with Metadata)
```

### Hybrid Search Architecture

```
Query
  │
  ├─ Vector Search Branch         Keyword Search Branch
  │  │                            │
  │  ├─ Embedding Generation      ├─ BM25 Tokenization
  │  │  │                         │  │
  │  │  ├─ Vector Store Lookup    │  ├─ Inverted Index Lookup
  │  │  │  │                      │  │  │
  │  │  │  └─▶ Similarity Score   │  │  └─▶ Relevance Score
  │  │  │                         │  │
  │  └─▶ Results Set A            └─▶ Results Set B
  │
  ├─ Merge Results
  │  (Weighted Combination)
  │
  ▼
Unified Ranked Results
```

---

## Agent Workflows

### ReAct Loop

```
┌─────────────────────┐
│ User Query/Goal     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ OBSERVE             │
│ - Current State     │
│ - Available Tools   │
│ - Context           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ THINK               │
│ LLM Reasons About   │
│ What To Do Next     │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │ Decision     │
    ├──────────────┤
    │ Use Tool?    │
    └──┬────────┬──┘
       │        │
       YES     NO
       │        │
       ▼        ▼
    ┌───────┐ ┌─────────────┐
    │ ACT   │ │ Generate    │
    │       │ │ Final       │
    │Execute│ │ Answer      │
    │Tool   │ │             │
    └───┬───┘ └─────────────┘
        │            │
        └─────┬──────┘
              │
              ▼
         ┌─────────────┐
         │ Loop Check  │
         ├─────────────┤
         │ Max Iters?  │
         │ Done?       │
         └──┬─────┬────┘
            │     │
           YES   NO
            │     │
            │     └──▶ Loop Back
            │          to OBSERVE
            ▼
       ┌─────────────┐
       │ Return      │
       │ Response    │
       └─────────────┘
```

### Function Calling Agent

```
Query
  │
  ▼
LLM with Function Definitions
  │
  ├─ Parse Available Functions
  │
  ▼
Parallel Function Calls (Async)
  │
  ├─▶ Function 1 ──▶ Result 1
  ├─▶ Function 2 ──▶ Result 2
  └─▶ Function 3 ──▶ Result 3
  │
  ▼
Aggregate Results
  │
  ▼
Final Synthesis
```

### Tool Integration

```
                    Agent
                      │
        ┌─────────────┼─────────────┐
        │             │             │
    ┌───▼────┐   ┌───▼────┐   ┌───▼────┐
    │ Tool 1 │   │ Tool 2 │   │ Tool 3 │
    │        │   │        │   │        │
    │┌──────┐│   │┌──────┐│   │┌──────┐│
    ││Input ││   ││Input ││   ││Input ││
    │└───┬──┘│   │└───┬──┘│   │└───┬──┘│
    │    │   │   │    │   │   │    │   │
    │    └─┬─┘   │    └─┬─┘   │    └─┬─┘
    │      │     │      │     │      │
    └──┬───┘     └──┬───┘     └──┬───┘
       │            │            │
       ├────────────┼────────────┤
       │            │            │
       ▼            ▼            ▼
    Result1     Result2      Result3
       │            │            │
       └────────────┼────────────┘
                    │
                    ▼
           Combined Results
                    │
                    ▼
           Agent Response
```

---

## Multi-Agent Systems

### Multi-Agent Architecture

```
┌───────────────────────────────────────┐
│      Control Plane / Orchestrator      │
│  (Task Distribution, Coordination)     │
└────────────┬────────────────────────┬──┘
             │                        │
    ┌────────▼────────┐      ┌────────▼────────┐
    │ Message Queue   │◀────▶│ State Manager   │
    │ (Redis/RabbitMQ)│      │ (Consensus)     │
    └────────┬────────┘      └────────┬────────┘
             │                        │
    ┌────────┼─────────────────────────┼────────┐
    │        │                         │        │
┌───▼──┐ ┌──▼────┐                ┌──▼───┐ ┌──▼──┐
│Agent1│ │Agent 2│ ... Agent N    │Monit.│ │Cache│
│      │ │       │                │      │ │     │
│Spec  │ │Special│                │     │ │    │
│Task1 │ │Task2  │                │     │ │    │
└──────┘ └───────┘                └──────┘ └─────┘
```

### Agent Communication Flow

```
Agent A                             Agent B
  │                                   │
  ├─▶ "I need data on topic X"       │
  │                                   │
  │  ◀─ "Requesting from service" ───┤
  │                                   │
  │  ┌─ Message Queue (RabbitMQ)     │
  │  │ ├─ Topic: retrieve_data       │
  │  │ ├─ Priority: high             │
  │  │ └─ Callback_queue: agent_a    │
  │  │                                │
  │  └──────────────────────────────▶│
  │                                   │
  │                          Process  │
  │                           Data    │
  │                                   │
  │  ◀────────────────────────────────┤
  │     "Data ready at /storage/x"    │
  │                                   │
  Continue Processing                 Done
```

---

## Query Processing

### Query Engine Selection Flow

```
User Query
    │
    ▼
Router Engine
    │
    ├─ Query Analysis
    │  ├─ Domain detection
    │  ├─ Complexity analysis
    │  └─ Intent extraction
    │
    ├─ Engine Candidates
    │  ├─ Vector Search Engine
    │  ├─ Keyword Search Engine
    │  ├─ Hierarchical Engine
    │  └─ Hybrid Engine
    │
    ▼
LLM Selection Logic
    │
    ├─ Score each engine
    ├─ Rank by relevance
    └─ Select best match
    │
    ▼
Selected Query Engine
    │
    ▼
Retrieval + Generation
    │
    ▼
Response
```

### Sub-Question Decomposition

```
Complex Query
    │
    ▼
LLM Decomposition
    │
    ├─▶ Sub-Question 1
    ├─▶ Sub-Question 2
    ├─▶ Sub-Question 3
    └─▶ Sub-Question 4
    │
    ├──────────────────┐
    │ Parallel Query   │
    │ Execution        │
    │ (async/await)    │
    └──┬───────────────┘
       │
       ├─▶ [Results 1] ──────┐
       ├─▶ [Results 2] ──────┼─▶ Results Pool
       ├─▶ [Results 3] ──────┤
       └─▶ [Results 4] ──────┘
              │
              ▼
       LLM Synthesis
              │
              ▼
       Final Answer
```

---

## Index Types

### Index Comparison

```
┌──────────────────────────────────────────────────────┐
│              Index Type Comparison                    │
├──────────────────────────────────────────────────────┤
│                                                       │
│ VectorStoreIndex                                     │
│ ├─ Fast semantic search                              │
│ ├─ Requires embeddings                               │
│ ├─ Memory: Moderate                                  │
│ └─ Use: General RAG, semantic queries                │
│                                                       │
│ ListIndex                                            │
│ ├─ Simple sequential retrieval                       │
│ ├─ No embeddings needed                              │
│ ├─ Memory: Low                                       │
│ └─ Use: Small datasets, similarity scoring           │
│                                                       │
│ TreeIndex                                            │
│ ├─ Hierarchical decomposition                        │
│ ├─ Creates tree structure                            │
│ ├─ Memory: Moderate-High                             │
│ └─ Use: Complex documents, multi-level queries       │
│                                                       │
│ KeywordTableIndex                                    │
│ ├─ Keyword-based lookup                              │
│ ├─ Inverted index structure                          │
│ ├─ Memory: Low-Moderate                              │
│ └─ Use: Keyword-heavy queries, exact matches         │
│                                                       │
│ SummaryIndex                                         │
│ ├─ Summary-based retrieval                           │
│ ├─ LLM-generated summaries                           │
│ ├─ Memory: High                                      │
│ └─ Use: Collection-level queries                     │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Vector Store Integration

```
Index Creation
    │
    ├─▶ Document Processing
    │   ├─ Chunking
    │   └─ Embedding
    │
    ├─▶ Vector Generation
    │   ├─ OpenAI Embeddings
    │   ├─ HuggingFace Embeddings
    │   └─ Local Models
    │
    ▼
Vector Store Options
    │
    ├─ Chroma (Local/Vector DB)
    ├─ Pinecone (Cloud)
    ├─ Weaviate (Distributed)
    ├─ Supabase (PostgreSQL+pgvector)
    ├─ Milvus (Scalable)
    └─ Many more...
    │
    ▼
Index Ready for Queries
```

---

## Memory and Context

### Chat Memory Buffer

```
┌──────────────────────────────────┐
│  ChatMemoryBuffer                │
├──────────────────────────────────┤
│                                  │
│  Max Tokens: 2000                │
│                                  │
│  Turn 1:                         │
│  ├─ User: "Hello"                │
│  └─ Assistant: "Hi there!"       │
│                                  │
│  Turn 2:                         │
│  ├─ User: "What did I ask?"      │
│  └─ Assistant: "You said hello"  │
│                                  │
│  ... (recent context maintained)
│                                  │
│  Token Count:                    │
│  ├─ Current: 1850 / 2000         │
│  ├─ When full: Prune oldest      │
│  └─ Keep recent turns            │
│                                  │
└──────────────────────────────────┘
```

### Memory Types

```
                    Memory Systems
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼──┐         ┌───▼──┐        ┌───▼──┐
    │ Chat │         │Vector│        │Summary│
    │Memory│         │Store │        │Memory │
    │Buffer│         │Memory│        │      │
    └──┬───┘         └──┬───┘        └──┬───┘
       │                │                │
       ├─ Last N       ├─ Semantic    ├─ Condensed
       │  messages     │  search      │  history
       │                │  over past   │
       ├─ Fast         │  turns       ├─ Compact
       │  retrieval    │              │  storage
       │                ├─ Chroma,    │
       └─ Limited      │  Pinecone,   └─ LLM
         context       │  Weaviate    │  generated
                       │              │
                       └─ Flexible   └─ Summarized
                          retention
```

---

## Advanced Patterns

### Agentic RAG

```
Query
  │
  ▼
Agent Reasoning
  │
  ├─ Plan retrieval strategy
  ├─ Determine if retrieval needed
  └─ Plan reasoning steps
  │
  ▼
Conditional Retrieval
  │
  ├─ Query 1: Search for context
  ├─ Evaluate results
  ├─ Query 2: Fetch specific details
  └─ Combine all retrieved data
  │
  ▼
LLM Reasoning + Generation
  │
  ├─ Reason over retrieved data
  ├─ Apply multi-step logic
  ├─ Verify answers
  └─ Generate response
  │
  ▼
Response with Verification
```

### Self-Reflection Loop

```
Initial Response Generation
          │
          ▼
    ┌──────────────┐
    │ Reflection   │
    │ Analysis     │
    │ (LLM)        │
    └──────┬───────┘
           │
        ┌──▼──┐
        │Pass?│
        └──┬──┘
           │
      ┌────┴────┐
      │          │
     YES        NO
      │          │
      ▼          ▼
   Return    ┌──────────────┐
             │ Improvement  │
             │ Suggestions  │
             │ (LLM)        │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ Regenerate   │
             │ Response     │
             │ (Improved)   │
             └──────┬───────┘
                    │
                    ▼
                 Return
```

### Tool Chaining

```
Task
 │
 ▼
[Tool A] ──▶ Output A
 │
 │ (Depends on A)
 ▼
[Tool B] ──▶ Output B
 │
 │ (Depends on B)
 ▼
[Tool C] ──▶ Output C
 │
 │ (Aggregate)
 ▼
Final Result
```

### Router Pattern

```
             Query
               │
               ▼
          Router Agent
               │
        ┌──────┼──────┬──────┐
        │      │      │      │
        ▼      ▼      ▼      ▼
    [Sales] [Support] [Technical] [Product]
    Agent   Agent     Agent       Agent
        │      │      │      │
        └──────┼──────┼──────┘
               │
               ▼
          Route Query to
        Selected Agent
               │
               ▼
            Response
```

---

## Performance Optimization

### Retrieval Optimization

```
Query
  │
  ▼
Preprocessing (5ms)
  │
  ├─ Normalization
  └─ Expansion
  │
  ▼
Embedding (10ms)
  │
  ▼
Vector Search (50ms)
  │
  ├─ Fast approximate search
  ├─ Top-K candidates
  └─ Initial ranking
  │
  ▼
Reranking (20ms)
  │
  ├─ Semantic reranking
  ├─ Relevance scoring
  └─ Final ranking
  │
  ▼
Generation (100ms)
  │
  ▼
Total: ~185ms
```

### Caching Strategy

```
Incoming Query
    │
    ▼
┌─────────────────┐
│ Cache Check     │
│ (Fast)          │
└────┬────────────┘
     │
  ┌──▼──┐
  │Hit? │
  └──┬──┘
     │
  ┌──┴──┐
  │    │
 YES  NO
  │    │
  │    └─▶ Cache Miss
  │       │
  │       ├─ Retrieve
  │       ├─ Generate
  │       ├─ Store in Cache
  │       │
  ▼       ▼
Return Cached   Return Generated
Response        Response + Cache
```

---

This diagrams document provides visual representations of all major LlamaIndex components and workflows. Each diagram illustrates the flow of data and control through different aspects of the framework.


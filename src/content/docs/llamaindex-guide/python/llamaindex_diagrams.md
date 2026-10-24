---
title: "LlamaIndex Architecture and System Diagrams"
description: "Text diagrams of LlamaIndex components, data flow, agent loops and retrieval patterns."
sidebar:
  label: "Diagrams"
framework: "llamaindex"
language: "python"
---

Visual reference for LlamaIndex components, workflows and patterns. Class names match `llama-index-core` %%version=llama-index%%; see the [comprehensive guide](/llamaindex-guide/python/llamaindex_comprehensive_guide/) for code.

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
    │(Function│   │  Engines  │   │ Engines │
    │ ReAct)  │   │(Router,   │   │         │
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
         │ SummaryIndex, PropGraph)  │
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
  │  (reciprocal-rank fusion, QueryFusionRetriever)
  │
  ▼
Unified Ranked Results
```

---

## Agent Workflows

### ReAct Loop (`ReActAgent`)

`ReActAgent` prompts the LLM to write its reasoning as text in a fixed format, parses that text, runs the tool it names, and feeds the result back as an observation. Any LLM works, because no native tool-calling support is needed.

```
┌─────────────────────┐
│ User message        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ LLM writes                          │◀────────────────┐
│   Thought: I need the price         │                 │
│   Action: get_price                 │                 │
│   Action Input: {"item": "pen"}     │                 │
└──────────┬──────────────────────────┘                 │
           │ parse the text                             │
           ▼                                            │
    ┌──────────────┐                                    │
    │ Action named?│                                    │
    └──┬────────┬──┘                                    │
      YES       NO                                      │
       │        │                                       │
       ▼        ▼                                       │
┌─────────────┐ ┌─────────────────────────┐             │
│ Run the tool│ │ LLM wrote               │             │
│ (Action)    │ │   Answer: ...           │             │
└──────┬──────┘ │ Return it as the        │             │
       │        │ AgentOutput             │             │
       ▼        └─────────────────────────┘             │
┌─────────────────────┐                                 │
│ Observation: result │ ─ added to the history ─────────┘
└─────────────────────┘
  (stops with an error after max_iterations rounds, 20 by default)
```

### Function Calling Agent (`FunctionAgent`)

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

### Multi-Agent Architecture (`AgentWorkflow`)

```
                    user_msg
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│ AgentWorkflow (one Context per run)              │
│                                                  │
│   shared memory  ◀──────────────▶  shared state  │
│        ▲                              ▲          │
│        │                              │          │
│  ┌─────┴──────┐  handoff()  ┌─────────┴──┐       │
│  │ root agent │ ──────────▶ │ specialist │       │
│  │ (triage)   │ ◀────────── │ agent      │       │
│  └─────┬──────┘  handoff()  └─────┬──────┘       │
│        │ tools                    │ tools        │
│        ▼                          ▼              │
│   FunctionTool,              QueryEngineTool,    │
│   MCP tools ...              FunctionTool ...    │
└──────────────────────────────────────────────────┘
                       │
                       ▼
              AgentOutput + event stream
```

Each agent may only hand off to the names in its `can_handoff_to`. The 2024 `llama-agents` design (control plane, message queue, agent services) is retired; separate-process deployment is now done with LlamaAgents / `llamactl`.

### Orchestrator Pattern (agents as tools)

```
Orchestrator FunctionAgent
  │
  ├─▶ tool: research(topic) ──▶ researcher.run(...) ──▶ notes
  │                                                       │
  │  ◀────────────────────────────────────────────────────┘
  │
  ├─▶ tool: write(notes) ─────▶ writer.run(...) ──────▶ draft
  │                                                       │
  │  ◀────────────────────────────────────────────────────┘
  ▼
Final answer (orchestrator keeps control throughout)
```

---

## Query Processing

### Query Engine Selection Flow (`RouterQueryEngine`)

The router does not analyse the query itself. It shows a selector LLM the description of each `QueryEngineTool` and asks which to use, then forwards the query. Descriptions are the routing signal, so write them carefully.

```
User Query
    │
    ▼
RouterQueryEngine
    │
    ▼
Selector (LLMSingleSelector or LLMMultiSelector)
    │   prompt: the query + a numbered list of tool descriptions
    │   LLM replies with a choice (or choices) and a reason
    ▼
Chosen QueryEngineTool(s)
    │
    ├─ single selector: that engine's response is returned
    └─ multi selector: each chosen engine answers, then a
       summariser LLM call combines the answers
    ▼
Response (the selection is in response.metadata["selector_result"])
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

### Chat engine: `condense_plus_context`

```
Turn N: user message
    │
    ▼
Condense (LLM call): chat history + message  ──▶  standalone question
    │
    ▼
Retriever(standalone question)  ──▶  nodes (post-processors applied)
    │
    ▼
System prompt = instructions + retrieved context
    │
    ▼
LLM(system prompt + chat history + user message)  ──▶  answer
    │
    ▼
Memory: user message and answer appended
```

On the first turn the history is empty, so the condense call is skipped and the message is used as written.

### Small-to-big retrieval

```
Indexing                              Querying
────────                              ────────
HierarchicalNodeParser                Query
  parent 2048 ─ child 512 ─ leaf 128    │
        │                               ▼
        ├─ all levels ──▶ docstore    Vector search over leaves only
        └─ leaves only ─▶ vector      (AutoMergingRetriever)
                          index         │
                                        ▼
                                      More than half of a parent's leaves hit?
                                        ├─ yes: replace them with the parent
                                        └─ no:  keep the leaves
                                        ▼
                                      Context for the LLM
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
│ SummaryIndex (formerly ListIndex)                    │
│ ├─ Stores nodes as a list; returns all of them       │
│ ├─ No embeddings needed                              │
│ ├─ Cost: every node goes to the LLM per query        │
│ └─ Use: "summarise everything", small collections    │
│                                                       │
│ PropertyGraphIndex                                   │
│ ├─ Entities and relations extracted at ingestion     │
│ ├─ Graph + vector retrieval                          │
│ └─ Use: relationship-heavy questions                 │
│                                                       │
│ MultiModalVectorStoreIndex                           │
│ ├─ Separate text and image embeddings                │
│ └─ Use: image + text retrieval (multimodal=True)     │
│                                                       │
└──────────────────────────────────────────────────────┘
```

`TreeIndex` and `KeywordTableIndex` still exist but are rarely the right choice; for keyword matching combine a vector retriever with BM25 (above). Choosing a vector store is covered in the [production guide](/llamaindex-guide/python/llamaindex_production_guide/#2-index-persistence-and-vector-store-choice).

---

## Memory and Context

### Agent Memory (`Memory`)

```
┌──────────────────────────────────────────────┐
│ Memory(session_id="user-123")                │
├──────────────────────────────────────────────┤
│ Short-term: chat history (SQL-backed)        │
│   token_limit × chat_history_token_ratio     │
│   ├─ User: "Hello"                           │
│   ├─ Assistant: "Hi there!"                  │
│   └─ ...                                     │
│          │ overflow (token_flush_size)       │
│          ▼                                   │
│ Long-term memory blocks                      │
│   ├─ StaticMemoryBlock   (fixed facts)       │
│   ├─ FactExtractionMemoryBlock (LLM facts)   │
│   └─ VectorMemoryBlock   (retrieved turns)   │
│          │                                   │
│          ▼                                   │
│ Rendered into the prompt on each turn        │
└──────────────────────────────────────────────┘
```

If you pass no memory, agents use a plain token-bounded `ChatMemoryBuffer`. Pass `memory=Memory.from_defaults(...)` to `run()` for the long-term blocks shown above.

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

For runnable versions of these flows see the [comprehensive guide](/llamaindex-guide/python/llamaindex_comprehensive_guide/) and [advanced agents](/llamaindex-guide/python/llamaindex_advanced_agents_python/).


---
title: "Haystack Architecture and Pattern Diagrams"
description: "Architecture diagrams for Haystack 3 — components, pipeline execution, the agent loop, multi-agent coordination and memory/retrieval."
framework: "haystack"
language: "python"
---

> Reflects `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%). Component names follow Haystack 3: ChatGenerators only, tools executed by `Agent`, a single `Pipeline` class with `run`, `run_async` and `stream`.

## Table of Contents

1. [Haystack 3 Core Architecture](#haystack-3-core-architecture)
2. [Component Types and Interactions](#component-types-and-interactions)
3. [Pipeline Execution Flow](#pipeline-execution-flow)
4. [Agent Loop Architecture](#agent-loop-architecture)
5. [Multi-Agent Coordination Patterns](#multi-agent-coordination-patterns)
6. [Memory and Retrieval Systems](#memory-and-retrieval-systems)
7. [Document Store Integration](#document-store-integration)
8. [RAG Pipeline Architecture](#rag-pipeline-architecture)
9. [Observability Stack](#observability-stack)
10. [Production Deployment Architecture](#production-deployment-architecture)

---

## Haystack 3 Core Architecture

### System Layers

```
┌──────────────────────────────────────────────────────────────┐
│                    Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │    Agents    │  │ RAG Systems  │  │   Pipelines  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────────────┐
│                  Pipeline Orchestration Layer                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  DAG Execution Engine                                  │   │
│  │  - Component Registration  - Data Flow  - Routing     │   │
│  │  - Branching Logic        - Error Handling             │   │
│  └────────────────────────────────────────────────────────┘   │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────────────┐
│                  Component Layer                              │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│  │  Retrievers    │ │ ChatGenerators │ │     Tools      │   │
│  └────────────────┘ └────────────────┘ └────────────────┘   │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│  │ PromptBuilders │ │    Routers     │ │  Validators    │   │
│  └────────────────┘ └────────────────┘ └────────────────┘   │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────────────┐
│                 Integration Layer                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │ LLM Providers│ │ Vector Stores│ │  APIs/Tools  │         │
│  └──────────────┘ └──────────────┘ └──────────────┘         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │ OpenAI       │ │ Elasticsearch│ │ HTTP Clients │         │
│  │ Anthropic    │ │ Weaviate     │ │ DB Connectors│         │
│  │ Hugging Face │ │ Pinecone     │ │ File Systems │         │
│  └──────────────┘ └──────────────┘ └──────────────┘         │
└──────────────────┬───────────────────────────────────────────┘
                   │
┌──────────────────┴───────────────────────────────────────────┐
│                  Data Layer                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │  Documents   │ │  Embeddings  │ │  Indexes     │         │
│  └──────────────┘ └──────────────┘ └──────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

### Component Anatomy

```
┌─────────────────────────────────────────────┐
│           Haystack Component                 │
├─────────────────────────────────────────────┤
│  @component                                 │
│  class MyComponent:                         │
│      def __init__(self, ...):               │
│          # Initialization                  │
│          # Validation                      │
│          # State Setup                     │
│                                             │
│      @component.output_types(              │
│          output1=type,                     │
│          output2=type                      │
│      )                                      │
│      def run(self, input1, input2) -> dict:│
│          # Processing Logic                │
│          return {                          │
│              "output1": value1,            │
│              "output2": value2             │
│          }                                  │
├─────────────────────────────────────────────┤
│  Inputs:     [input1, input2, ...]         │
│  Outputs:    [output1, output2, ...]       │
│  State:      [internal_state]              │
│  Metadata:   [name, type, version]         │
└─────────────────────────────────────────────┘
```

---

## Component Types and Interactions

### Component Hierarchy

```
┌────────────────────────────────────────────────────────┐
│                Haystack Component                       │
│                  (Base Class)                           │
└─────────────────┬──────────────────────────────────────┘
                  │
    ┌─────────────┼──────────────┬───────────────┐
    │             │              │               │
    ▼             ▼              ▼               ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│Retrievers│ │  Chat    │ │  Routers │ │ Builders &   │
│          │ │Generators│ │          │ │ Joiners      │
└──────────┘ └──────────┘ └──────────┘ └──────────────┘
    │             │              │               │
    ├─────────────┼──────────────┼───────────────┤
    │             │              │               │
    ▼             ▼              ▼               ▼
┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐
│BM25      │ │OpenAI    │ │Conditional│ │PromptBuilder │
│Embedding │ │Anthropic*│ │LLMMessages│ │ChatPrompt-   │
│MultiQuery│ │Fallback  │ │Metadata   │ │ Builder      │
│          │ │Mock      │ │FileType   │ │DocumentJoiner│
└──────────┘ └──────────┘ └───────────┘ └──────────────┘
  * integration package (haystack_integrations.*)
└──────────┘ └──────────┘ └──────────┘ └──────────────┘
```

### Data Flow Between Components

```
Component A                    Component B
┌─────────────────┐          ┌─────────────────┐
│                 │          │                 │
│  @output_types( │          │  def run(self,  │
│    result=str   │          │    input: str   │
│  )              │          │  ):             │
│  def run():     │          │      ...        │
│    ...          │          │                 │
│    return {     │          │  @output_types( │
│      "result":  ├─────────►│    output=int   │
│        "text"   │          │  )              │
│    }            │          │                 │
│                 │          │                 │
└─────────────────┘          └─────────────────┘
     ▲                                │
     │                                │
     │ Connection definition in       │
     │ pipeline.connect(              │
     │ "A.result", "B.input")         │
     │                                ▼
     │                          ┌─────────────────┐
     │                          │ Component C     │
     │                          │                 │
     │                          │  def run(self,  │
     └──────────────────────────┤    data: int    │
                                │  ):             │
                                │      ...        │
                                └─────────────────┘
```

---

## Pipeline Execution Flow

### Simple Linear Pipeline

```
Input Query
    │
    ▼
┌────────────────────┐
│   Validator        │  ✓ Validates input format
│   Component        │  ✓ Checks for empty values
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Transformer      │  ✓ Normalises text
│   Component        │  ✓ Preprocesses data
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Processor        │  ✓ Main processing logic
│   Component        │  ✓ Complex transformations
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Formatter        │  ✓ Structures output
│   Component        │  ✓ Adds metadata
└────────┬───────────┘
         │
         ▼
    Output
```

### Branching Pipeline with Conditional Routing

```
          Input
            │
            ▼
      ┌─────────────┐
      │  Classifier │  ← Routes based on input type
      └─────┬───────┘
            │
       ┌────┴────────┐
       │             │
    Type A         Type B
       │             │
       ▼             ▼
  ┌─────────┐   ┌─────────┐
  │Handler A│   │Handler B│
  └────┬────┘   └────┬────┘
       │             │
       └────┬────────┘
            │
            ▼
       ┌──────────────┐
       │ Aggregator   │  ← Combines results
       │ Component    │
       └──────┬───────┘
              │
              ▼
          Output
```

### Parallel Execution Pipeline

```
                Input
                  │
                  ▼
        ┌─────────────────┐
        │  Splitter       │  ← Distributes work
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
 Worker 1     Worker 2     Worker 3
 ┌───────┐   ┌───────┐   ┌───────┐
 │Process│   │Process│   │Process│
 │Task 1 │   │Task 2 │   │Task 3 │
 └────┬──┘   └────┬──┘   └────┬──┘
      │           │           │
      └───────────┼───────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  Aggregator     │  ← Combines results
        │  Component      │
        └────────┬────────┘
                 │
                 ▼
             Output
```

---

## Agent Loop Architecture

### Agent Execution Flow

```
agent.run(messages=[...], **prompt_variables, **state_values)
  │
  ▼
┌──────────────────────────────────────┐
│ before_run hooks                     │
│ Render system_prompt / user_prompt   │
│ Initialise State (messages, step_count,
│   token_usage, tool_call_counts, ...)│
└──────────────────┬───────────────────┘
                   │
      ┌────────────▼─────────────┐
      │ before_llm hooks         │◄─────────────────────────┐
      │ (a hook may set stop_run)│                          │
      └────────────┬─────────────┘                          │
                   ▼                                        │
      ┌──────────────────────────┐                          │
      │ chat_generator.run(      │                          │
      │   messages, tools)       │                          │
      └────────────┬─────────────┘                          │
                   │                                        │
         tool calls in reply?                               │
          │ no               │ yes                          │
          ▼                  ▼                              │
   exit_reason="text"  ┌──────────────────────────┐         │
          │            │ before_tool hooks        │         │
          │            │ (e.g. ConfirmationHook)  │         │
          │            │ run tools concurrently   │         │
          │            │ (tool_concurrency_limit) │         │
          │            │ outputs_to_state → State │         │
          │            │ after_tool hooks         │         │
          │            └────────────┬─────────────┘         │
          │                         │                       │
          │         tool in exit_conditions?                │
          │            │ yes                │ no            │
          │            ▼                    └─ step_count < │
          │     exit_reason=<tool>             max_agent_steps
          │            │                                    │
          ▼            ▼                                    │
      ┌──────────────────────────┐   continue_run=True      │
      │ on_exit hooks            │──────────────────────────┘
      └────────────┬─────────────┘
                   ▼
      ┌──────────────────────────┐
      │ after_run hooks          │
      │ return {messages,        │
      │  last_message, step_count│
      │  token_usage,            │
      │  tool_call_counts,       │
      │  exit_reason, ...state}  │
      └──────────────────────────┘
```

### Agent State Transitions

```
            ┌─────────────────────────┐
            │   IDLE STATE            │
            │  (Awaiting input)       │
            └────────────┬────────────┘
                         │
                    User provides query
                         │
                         ▼
            ┌─────────────────────────┐
            │ PROCESSING STATE        │
            │ (Thinking & Planning)   │
            └────────┬───────┬────────┘
                     │       │
           Tool Call │       │ Direct Answer
                     │       │
                     ▼       ▼
            ┌──────────────────────────┐
            │ TOOL_EXECUTION STATE     │
            │ (Running tools)          │
            └────────┬────────────────┘
                     │
              ┌──────▼──────┐
              │ Tool Result │
              └──────┬──────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │ CHECK EXIT CONDITIONS           │
    │ - max_agent_steps reached?      │
    │ - exit-condition tool called?   │
    │ - text reply (no tool calls)?   │
    └──┬────────────────────────┬─────┘
       │                        │
    No │                        │ Yes
       │                        │
       ▼                        ▼
    Loop Back      ┌─────────────────────┐
                   │  COMPLETE STATE     │
                   │  (Done)             │
                   └─────────────────────┘
```

---

## Multi-Agent Coordination Patterns

### Sequential Agent Pipeline

```
Query
  │
  ▼
┌─────────────────┐
│   Agent 1       │
│  (Analyser)     │  ✓ Analyses problem
│                 │  ✓ Outputs analysis
└────────┬────────┘
         │
    (Output 1)
         │
         ▼
┌─────────────────┐
│   Agent 2       │
│  (Planner)      │  ✓ Creates plan
│                 │  ✓ Based on analysis
└────────┬────────┘
         │
    (Output 2)
         │
         ▼
┌─────────────────┐
│   Agent 3       │
│  (Executor)     │  ✓ Executes plan
│                 │  ✓ Produces results
└────────┬────────┘
         │
    (Output 3)
         │
         ▼
      Result
```

### Parallel Agent Execution with Aggregation

```
                    Query
                      │
                      ▼
              ┌───────────────────┐
              │  Task Distributor │
              └─────────┬─────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    Task 1         Task 2         Task 3
    [Agent A]      [Agent B]      [Agent C]
    │              │              │
    │ (Parallel    │ (Parallel    │ (Parallel
    │  Execution)  │  Execution)  │  Execution)
    │              │              │
    ▼              ▼              ▼
   Result A      Result B      Result C
    │              │              │
    └──────────────┼──────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Result Aggregator   │
        │  - Combine results   │
        │  - Resolve conflicts │
        │  - Format output     │
        └──────────┬───────────┘
                   │
                   ▼
               Final Result
```

### Master-Worker Agent Pattern

```
        ┌──────────────────────┐
        │   Master Agent       │
        │  - Orchestrates work │
        │  - Monitors progress │
        │  - Handles failures  │
        └──────────┬───────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
  Worker 1     Worker 2     Worker 3
  ┌────────┐  ┌────────┐  ┌────────┐
  │Task 1  │  │Task 2  │  │Task 3  │
  │ Agent  │  │ Agent  │  │ Agent  │
  │        │  │        │  │        │
  │[Work]  │  │[Work]  │  │[Work]  │
  │        │  │        │  │        │
  └────┬───┘  └────┬───┘  └────┬───┘
       │           │           │
       └───────────┼───────────┘
                   │
                   ▼ (Results + Status)
        ┌──────────────────────┐
        │   Master Agent       │
        │  - Aggregates results│
        │  - Produces output   │
        └──────────────────────┘
```

---

## Memory and Retrieval Systems

### Conversation Memory Architecture

```
┌─────────────────────────────────────────┐
│      Conversation Memory Buffer          │
├─────────────────────────────────────────┤
│                                          │
│  Turn 1:                                 │
│  User: "Hello, what's the time?"        │
│  Assistant: "It's 2:30 PM"              │
│                                          │
│  Turn 2:                                 │
│  User: "What did I just ask?"           │
│  Assistant: "You asked what time it is"  │
│                                          │
│  Turn 3:                                 │
│  User: "Can you remember my name?"      │
│  Assistant: "I don't have your name"    │
│                                          │
│  [Maximum capacity: e.g., 100 messages] │
│  [Oldest messages purged when exceeded]  │
└─────────────────────────────────────────┘
     │                          │
     ▼                          ▼
┌──────────────────┐  ┌──────────────────┐
│Short-term Memory │  │Persistent Storage│
│(In-Memory Store) │  │(Database/File)   │
│                  │  │                  │
│Fast Access       │  │Long-term Recall │
│Limited Size      │  │Unlimited Size    │
└──────────────────┘  └──────────────────┘
```

### Semantic Memory with Vector Database

```
Input Query
    │
    ▼
┌─────────────────────┐
│   Embedding Model   │  "Convert text to embeddings"
└────────────┬────────┘
             │
             ▼ (Vector)
┌──────────────────────────────┐
│   Vector Database            │
│  (e.g., Weaviate)            │
│                              │
│  ┌──────────────────────┐    │
│  │ Document 1: [0.1, 0.2] │  │
│  │ Document 2: [0.2, 0.3] │  │
│  │ Document 3: [0.05, 0.25]│ │
│  └──────────────────────┘    │
└──────────┬───────────────────┘
           │
           ▼ (Similarity Search)
┌──────────────────────┐
│ Top K Similar Docs   │
│                      │
│ Doc 1 - Similarity 0.95
│ Doc 3 - Similarity 0.87
│ Doc 5 - Similarity 0.82
└──────────┬───────────┘
           │
           ▼
   Retrieved Documents
```

---

## Document Store Integration

### Multi-Store Architecture

```
┌─────────────────────────────────────────────────────┐
│         Haystack Document Abstraction Layer         │
└────────────┬────────────────┬──────────────┬────────┘
             │                │              │
       Write Documents    Query              Index Management
             │                │              │
    ┌────────▼────────┐   ┌────▼────┐   ┌──▼──────────┐
    │Document Storage │   │Retrieval │   │Embedding    │
    │Interface        │   │Interface │   │Management   │
    └────────┬────────┘   └────┬────┘   └──┬──────────┘
             │                │              │
    ┌────────┴────────────────┴──────────────┴─────────┐
    │                                                   │
    ▼                           ▼                      ▼
┌──────────────┐      ┌─────────────────┐     ┌──────────────┐
│Elasticsearch │      │    Weaviate     │     │   Pinecone   │
│              │      │                 │     │              │
│- Full text   │      │- Vector DB      │     │- Vector DB   │
│- Hybrid search│      │- Semantic search│     │- Semantic    │
└──────────────┘      └─────────────────┘     └──────────────┘

┌──────────────┐      ┌─────────────────┐     ┌──────────────┐
│   Qdrant     │      │    Milvus       │     │    Chroma    │
│              │      │                 │     │              │
│- Vector DB   │      │- Vector DB      │     │- Local/Cloud │
│- Filtering   │      │- Performance    │     │- Lightweight │
└──────────────┘      └─────────────────┘     └──────────────┘
```

---

## RAG Pipeline Architecture

### Complete RAG System

```
┌─────────────────────────────────────────────────────────────┐
│                  QUERY PROCESSING PATH                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User Query: "What is the return policy?"                    │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────────────┐                                   │
│   │ Query Processor     │ - Normalise  - Expand             │
│   └────────────┬────────┘                                   │
│                │                                             │
│                ▼                                             │
│   ┌─────────────────────────────────┐                       │
│   │  Embedding Model                │ - Convert to vector   │
│   │  (text-embedding-3-small)       │                       │
│   └────────────┬────────────────────┘                       │
│                │                                             │
│                ▼                                             │
│   ┌─────────────────────────────────┐                       │
│   │ Vector Search                   │ - Similarity search   │
│   │ (Weaviate / Elasticsearch)      │ - Return top-k docs  │
│   └────────────┬────────────────────┘                       │
│                │                                             │
│                ▼                                             │
│   Retrieved Documents:                                      │
│   1. "Returns accepted within 30 days"                      │
│   2. "Full refund process"                                  │
│   3. "Return shipping instructions"                         │
│                │                                             │
└────────────────┼──────────────────────────────────────────┘
                 │
┌────────────────┴──────────────────────────────────────────┐
│              GENERATION PATH                               │
├────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌────────────────────────────────┐                       │
│   │  Prompt Builder                │                       │
│   │                                │  System Prompt:       │
│   │  Format:                       │  "You are helpful"    │
│   │  - System prompt               │                       │
│   │  - Retrieved context           │  Context:             │
│   │  - User query                  │  [Retrieved docs]     │
│   └────────────┬────────────────────┘                       │
│                │                                             │
│                ▼                                             │
│   ┌────────────────────────────────┐                       │
│   │  ChatGenerator                 │  e.g. OpenAIChat-     │
│   │  (Generate Response)           │  Generator            │
│   └────────────┬────────────────────┘                       │
│                │                                             │
│                ▼                                             │
│   Generated Answer:                                        │
│   "We accept returns within 30 days for full refund..."    │
│                │                                             │
│                ▼                                             │
│   ┌────────────────────────────────┐                       │
│   │  Output Validator              │  - Check format       │
│   │                                │  - Validate schema    │
│   │  Optional:                     │  - Ensure completeness│
│   │  - JSON Formatting             │                       │
│   │  - Citation Extraction         │                       │
│   └────────────┬────────────────────┘                       │
│                │                                             │
│                ▼                                             │
│   Final Response (with citations)                          │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

---

## Observability Stack

### Monitoring Architecture

```
┌──────────────────────────────────────────────────────────┐
│          Haystack Application with Instrumentation       │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  tracing.enable_tracing(OpenTelemetryTracer(...))         │
│  or a connector: OpenTelemetryConnector / Langfuse...    │
│                                                           │
│  Spans: haystack.pipeline.run                             │
│           └ haystack.component.run                        │
│         haystack.agent.run                                │
│           └ haystack.agent.step                           │
│               ├ haystack.agent.step.llm                   │
│               └ haystack.agent.step.tool (one per call)   │
│                    │                                     │
└────────────────────┼──────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │    Tracing Collector         │
        │  (e.g., OpenTelemetry)       │
        │                              │
        │  Collects:                   │
        │  - Event timestamps          │
        │  - Component metrics         │
        │  - Error information         │
        └──────────┬────────────────┬──┘
                   │                │
        ┌──────────▼──┐    ┌────────▼──┐
        │  Metrics    │    │  Traces   │
        │  Storage    │    │  Storage  │
        └──────────┬──┘    └────────┬──┘
                   │                │
    ┌──────────────▼────────────────▼──────┐
    │  Observability Backend                │
    │  - Prometheus (Metrics)               │
    │  - Jaeger (Traces)                    │
    │  - ELK Stack (Logs)                   │
    └──────────────┬───────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │  Dashboards & Alerting             │
    │  - Grafana (Metrics)               │
    │  - Jaeger UI (Traces)              │
    │  - Custom Alerts                   │
    └────────────────────────────────────┘
```

---

## Production Deployment Architecture

### Container-Based Deployment

```
┌──────────────────────────────────────────────────────────┐
│             Kubernetes Cluster                            │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────────────────────────┐               │
│  │  API Service (Load Balanced)         │               │
│  │  - Ingress Controller                │               │
│  │  - SSL/TLS                           │               │
│  │  - Rate Limiting                     │               │
│  └──────────┬───────────────────────────┘               │
│             │                                            │
│     ┌───────┴───────┬───────┬───────┐                   │
│     │               │       │       │                   │
│     ▼               ▼       ▼       ▼                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ Pod 1   │  │ Pod 2   │  │ Pod 3   │ (Replicas)    │
│  │Haystack │  │Haystack │  │Haystack │               │
│  │Agent    │  │Agent    │  │Agent    │               │
│  └────┬────┘  └────┬────┘  └────┬────┘               │
│       │            │            │                     │
│       └────────────┼────────────┘                     │
│                    │                                  │
├────────────────────┼──────────────────────────────────┤
│  Storage Layer     │                                  │
│                    ▼                                  │
│  ┌──────────────────────────────┐                    │
│  │ Redis Cache                  │                    │
│  └──────────────────────────────┘                    │
│                                                       │
│  ┌──────────────────────────────┐                    │
│  │ PostgreSQL / Database        │                    │
│  └──────────────────────────────┘                    │
│                                                       │
│  ┌──────────────────────────────┐                    │
│  │ Vector Store (Weaviate)      │                    │
│  └──────────────────────────────┘                    │
│                                                       │
├──────────────────────────────────────────────────────┤
│  External Services                                   │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ OpenAI   │  │ Anthropic│  │ Hugging  │          │
│  │ API      │  │ API      │  │ Face API │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### Microservices Architecture

```
┌─────────────────────────────────────────────────────────┐
│              API Gateway                                 │
│         (Authentication & Routing)                       │
└────────────┬────────────────────────────────────────────┘
             │
       ┌─────┴──────┬──────────┬───────────┐
       │            │          │           │
       ▼            ▼          ▼           ▼
  ┌────────────┐┌────────┐┌─────────┐┌──────────┐
  │ Agent      ││RAG     ││Search   ││Analytics │
  │Service     ││Service ││Service  ││Service   │
  │            ││        ││         ││          │
  │- Process   ││- Index ││- Query  ││- Metrics │
  │- Tools     ││- Embed ││- Score  ││- Logs    │
  │- LLM calls ││- Store ││- Return ││- Traces  │
  └─────┬──────┘└──┬─────┘└────┬────┘└──────────┘
        │          │           │
        └──────────┼───────────┘
                   │
        ┌──────────▼──────────┐
        │ Message Queue       │
        │ (RabbitMQ/Kafka)    │
        │ - Event streaming   │
        │ - Async processing  │
        └─────────────────────┘
```

---

## Advanced Pattern Diagrams

### ReAct Loop with Self-Correction

```
                 Start
                   │
                   ▼
          ┌────────────────────┐
          │ 1. REASON          │
          │ - Analyse problem  │
          │ - Break into steps │
          │ - Plan approach    │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │ 2. ACT             │
          │ - Execute tools    │
          │ - Collect results  │
          │ - Gather feedback  │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │ 3. OBSERVE         │
          │ - Analyse results  │
          │ - Check correctness│
          │ - Identify gaps    │
          └────────┬───────────┘
                   │
              ┌────▼────┐
              │Result   │
              │Correct? │
              └────┬────┘
                   │
            ┌──────┴───────┐
            │              │
          Yes              No
            │              │
            ▼              ▼
         Return      Self-Correct
         Result      (Adjust plan)
            │              │
            │         ┌────▼───────────┐
            │         │ 4. REFLECT     │
            │         │ - Review error │
            │         │ - Adjust logic │
            │         │ - Generate fix │
            │         └─────┬──────────┘
            │               │
            │         (Loop back to 1)
            │         (with corrections)
            │
            ▼
           End
```

### Hierarchical Agent System

```
                    ┌──────────────────┐
                    │  Orchestrator     │
                    │  Agent (Main)     │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Strategist   │ │ Coordinator  │ │ Executor     │
    │ Agent        │ │ Agent        │ │ Agent        │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
         Plans          Schedules        Executes
           │                │                │
           └────────────────┼────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
          ┌────────┐  ┌────────┐  ┌────────┐
          │Worker  │  │Worker  │  │Worker  │
          │Agent 1 │  │Agent 2 │  │Agent 3 │
          └────────┘  └────────┘  └────────┘
```

---

This diagram document provides comprehensive visual representations of Haystack's architecture and patterns across all levels of complexity.


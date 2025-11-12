
# Semantic Kernel: Architecture and Pattern Diagrams

This document contains comprehensive visual diagrams illustrating Semantic Kernel's architecture, component interactions, and various patterns.

---

## 1. Semantic Kernel Architecture Overview

### 1.1 Core Component Hierarchy


```
┌─────────────────────────────────────────────────────────────────┐
│                     Semantic Kernel                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Kernel (Orchestrator)                │  │
│  │  - Manages lifecycle                                    │  │
│  │  - Routes requests                                      │  │
│  │  - Coordinates services                                 │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│        ┌──────────────────┼──────────────────┐                │
│        │                  │                  │                │
│  ┌─────▼────────┐  ┌──────▼───────┐  ┌──────▼────────┐      │
│  │  Plugins     │  │   Services   │  │     Memory    │      │
│  │              │  │              │  │               │      │
│  │ • Semantic   │  │ • OpenAI     │  │ • Volatile    │      │
│  │ • Native     │  │ • Azure      │  │ • Vector      │      │
│  │ • OpenAPI    │  │ • Hugging    │  │ • Persistent  │      │
│  │              │  │   Face       │  │               │      │
│  └──────────────┘  └──────────────┘  └───────────────┘      │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Connectors & Infrastructure                 │  │
│  │                                                          │  │
│  │  • Azure AI Search  • Qdrant  • Weaviate               │  │
│  │  • OpenAI           • Hugging Face                     │  │
│  │  • Logging          • Telemetry                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Request Processing Pipeline

```
┌────────────────┐
│  User Input    │
└────────┬───────┘
         │
         ▼
┌────────────────────────┐
│  Parse Request         │
│  - Extract parameters  │
│  - Validate input      │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Locate Function       │
│  - Find plugin         │
│  - Validate params     │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Prepare Context       │
│  - Load memory         │
│  - Build variables     │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Invoke Function       │
│  - Execute logic       │
│  - Call LLM if needed  │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Parse Result          │
│  - Format output       │
│  - Validate schema     │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Update Memory         │
│  - Store context       │
│  - Update embeddings   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  Return Result         │
└────────────────────────┘
```

---

## 2. Plugin Architecture

### 2.1 Plugin Lifecycle

```
┌────────────────────────────────────────────────────┐
│              Plugin Lifecycle                       │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. Creation                                       │
│     ┌──────────────────────────────────────────┐  │
│     │ Define Plugin Class                      │  │
│     │ - Implement methods                      │  │
│     │ - Add descriptions                       │  │
│     │ - Set parameters                         │  │
│     └──────────────────────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  2. Registration                                   │
│     ┌──────────────────────────────────────────┐  │
│     │ Import into Kernel                       │  │
│     │ - kernel.ImportPluginFromType<T>        │  │
│     │ - Assign plugin name                    │  │
│     └──────────────────────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  3. Discovery                                      │
│     ┌──────────────────────────────────────────┐  │
│     │ Kernel Inspects Plugin                   │  │
│     │ - Extracts metadata                      │  │
│     │ - Builds function catalog                │  │
│     └──────────────────────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  4. Invocation                                     │
│     ┌──────────────────────────────────────────┐  │
│     │ Call Function                            │  │
│     │ - Resolve parameters                     │  │
│     │ - Execute function                       │  │
│     └──────────────────────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  5. Result Handling                                │
│     ┌──────────────────────────────────────────┐  │
│     │ Process Result                           │  │
│     │ - Format output                          │  │
│     │ - Log execution                          │  │
│     │ - Update memory                          │  │
│     └──────────────────────────────────────────┘  │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 2.2 Plugin Types and Relationships

```
┌─────────────────────────────────────────────────────────┐
│                    Plugin Types                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │         Semantic Plugins                       │   │
│  │  (Prompt-Based)                                │   │
│  │                                                │   │
│  │  + Leverage LLM capabilities                  │   │
│  │  + Template-based prompts                     │   │
│  │  + Parameter injection                        │   │
│  │  + Cross-model compatibility                  │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │         Native Plugins                         │   │
│  │  (Code-Based)                                  │   │
│  │                                                │   │
│  │  + Direct code execution                      │   │
│  │  + Full language capabilities                 │   │
│  │  + Deterministic results                      │   │
│  │  + Type safety (especially C#)               │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │         OpenAPI Plugins                        │   │
│  │  (External API Integration)                    │   │
│  │                                                │   │
│  │  + REST API integration                       │   │
│  │  + Automated function discovery               │   │
│  │  + Schema-based invocation                    │   │
│  │  + External service consumption               │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Multi-Agent Orchestration Patterns

### 3.1 Master-Worker Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                  Master-Worker Pattern                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                        ┌─────────────┐                       │
│                        │    Master   │                       │
│                        │    Agent    │                       │
│                        └──────┬──────┘                       │
│                               │                              │
│                ┌──────────────┼──────────────┐               │
│                │              │              │               │
│    ┌───────────▼──────┐ ┌─────▼─────────┐ ┌──▼────────────┐ │
│    │ Worker Agent 1   │ │ Worker Agent 2│ │Worker Agent 3 │ │
│    │ (Task A)         │ │ (Task B)      │ │ (Task C)      │ │
│    │ - Specialised    │ │ - Specialised │ │ - Specialised │ │
│    │ - Parallel       │ │ - Parallel    │ │ - Parallel    │ │
│    └────────┬─────────┘ └────────┬──────┘ └──┬────────────┘ │
│             │                    │             │              │
│             └────────────────────┼─────────────┘              │
│                                  │                            │
│                          ┌───────▼────────┐                  │
│                          │ Results        │                  │
│                          │ Aggregation    │                  │
│                          └───────┬────────┘                  │
│                                  │                            │
│                          ┌───────▼────────┐                  │
│                          │ Final Output   │                  │
│                          └────────────────┘                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Peer-to-Peer Communication Pattern

```
┌────────────────────────────────────────────────────────────┐
│           Peer-to-Peer Communication Pattern               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│   ┌──────────┐                                            │
│   │ Agent A  │◄─────┐                                     │
│   │          │      │                                     │
│   └────┬─────┘      │                                     │
│        │            │                                     │
│        ├───────────►│◄──────────┐                        │
│        │     ┌──────▼────┐      │                        │
│        │     │ Agent B   │      │                        │
│        │     │           │      │                        │
│        │     └──────┬─────┘      │                        │
│        │            │            │                        │
│        ├────────────┼────────────┤                        │
│        │            │            │                        │
│        │            ▼            ▼                        │
│        │     ┌──────────┐  ┌──────────┐                  │
│        └────►│ Agent C  │  │ Agent D  │                  │
│             │          │  │          │                   │
│             └──────┬───┘  └────┬─────┘                   │
│                    │           │                         │
│                    └─────┬─────┘                         │
│                          │                              │
│                  ┌───────▼────────┐                     │
│                  │ Consensus      │                     │
│                  │ Result         │                     │
│                  └────────────────┘                     │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### 3.3 Hierarchical Orchestration Pattern

```
┌─────────────────────────────────────────────────────────┐
│        Hierarchical Orchestration Pattern               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│              ┌────────────────────┐                     │
│              │   Orchestrator     │                     │
│              │   (Main Task)      │                     │
│              └─────────┬──────────┘                     │
│                        │                                │
│          ┌─────────────┼─────────────┐                  │
│          │             │             │                  │
│    ┌─────▼──────┐ ┌────▼─────┐ ┌───▼────────┐          │
│    │ Subtask 1  │ │Subtask 2 │ │Subtask 3   │          │
│    │ Coordinator│ │Coordinator│ │Coordinator│          │
│    └────┬──────┘ └────┬─────┘ └───┬────────┘          │
│         │             │             │                   │
│    ┌────┴────┐   ┌────┴────┐   ┌───┴────┐             │
│    │          │   │         │   │        │             │
│ ┌──▼─┐  ┌──┐ │┌──▼──┐ ┌──┐ │┌──▼─┐ ┌──┐│             │
│ │Ag1 │  │Ag2│ │Ag3  │ │Ag4│ │Ag5 │ │Ag6││             │
│ └────┘  └──┘ │└─────┘ └──┘ │└────┘ └──┘│             │
│              │              │           │             │
│              └──────────────┴───────────┘             │
│                                                       │
│              Results Flow Upward                    │
│              ▲          ▲          ▲                 │
│              │          │          │                 │
│    ┌─────────┼──────────┼──────────┼────────┐        │
│    │         │          │          │        │        │
│    │    ┌────┴────┐ ┌───┴────┐ ┌──┴───┐    │        │
│    │    │Subtask  │ │Subtask │ │Task  │    │        │
│    │    │1 Result │ │2 Result│ │3 Result   │        │
│    │    └─────────┘ └────────┘ └───────┘    │        │
│    │                                        │        │
│    │           ┌──────────────────┐         │        │
│    │           │ Final Result     │         │        │
│    │           └──────────────────┘         │        │
│    │                                        │        │
│    └────────────────────────────────────────┘        │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 4. Memory System Architecture

### 4.1 Memory Store Hierarchy

```
┌────────────────────────────────────────────────────┐
│            Memory Store Architecture               │
├────────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │         Kernel                              │ │
│  │  add_memory_store()                         │ │
│  └─────────────┬───────────────────────────────┘ │
│                │                                 │
│    ┌───────────┴──────────────┐                 │
│    │                          │                 │
│    ▼                          ▼                 │
│ ┌──────────────┐       ┌─────────────────────┐ │
│ │ Volatile     │       │ Persistent Stores   │ │
│ │ MemoryStore  │       │                     │ │
│ │              │       │ ┌────────────────┐  │ │
│ │ • In-memory  │       │ │Azure AI Search │  │ │
│ │ • Fast       │       │ └────────────────┘  │ │
│ │ • Temporary  │       │                     │ │
│ │              │       │ ┌────────────────┐  │ │
│ │              │       │ │    Qdrant      │  │ │
│ │              │       │ └────────────────┘  │ │
│ │              │       │                     │ │
│ │              │       │ ┌────────────────┐  │ │
│ │              │       │ │  Weaviate      │  │ │
│ │              │       │ └────────────────┘  │ │
│ │              │       │                     │ │
│ │              │       │ ┌────────────────┐  │ │
│ │              │       │ │ Custom Store   │  │ │
│ │              │       │ └────────────────┘  │ │
│ └──────────────┘       └─────────────────────┘ │
│                                                │
└────────────────────────────────────────────────┘
```

### 4.2 Embedding and Search Pipeline

```
┌─────────────────────────────────────────────────────────┐
│      Embedding and Semantic Search Pipeline            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐                                  │
│  │ Input Text       │                                  │
│  │ "User feedback"  │                                  │
│  └────────┬─────────┘                                  │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────────┐                              │
│  │ Text Embedding       │                              │
│  │ Service              │                              │
│  │ • OpenAI             │                              │
│  │ • Azure OpenAI       │                              │
│  │ • Hugging Face       │                              │
│  └────────┬─────────────┘                              │
│           │                                             │
│           ▼                                             │
│  ┌────────────────────┐                                │
│  │ Vector             │                                │
│  │ [0.24, 0.51, ...] │                                │
│  │ [0.33, 0.12, ...] │                                │
│  │ [0.19, 0.44, ...] │                                │
│  └────────┬───────────┘                                │
│           │                                             │
│           ▼                                             │
│  ┌───────────────────────┐                             │
│  │ Memory Store          │                             │
│  │ • Collections         │                             │
│  │ • Persist vectors     │                             │
│  │ • Store metadata      │                             │
│  └────────┬──────────────┘                             │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────────┐                              │
│  │ Similarity Search     │                              │
│  │ • Query vector       │                              │
│  │ • Vector DB search   │                              │
│  │ • Ranked results     │                              │
│  └────────┬─────────────┘                              │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────────┐                              │
│  │ Retrieved Context    │                              │
│  │ Top K memories       │                              │
│  │ Ready for LLM        │                              │
│  └──────────────────────┘                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Planner Architecture

### 5.1 Planner Types and Execution

```
┌──────────────────────────────────────────────────────────┐
│          Planner Types and Execution Flow               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  INPUT: Goal/Task                                        │
│  ┌────────────────┐                                      │
│  │ "Send email"   │                                      │
│  │ "Generate"     │                                      │
│  │ "Report"       │                                      │
│  └────────┬───────┘                                      │
│           │                                              │
│     ┌─────┴──────────────────┐                           │
│     │ Planner Type?          │                           │
│     └──────┬────────────────┬─┴─────┬──────┐             │
│            │                │       │      │             │
│        ┌───▼───┐        ┌───▼──┐ ┌──▼──┐ ┌──▼──┐       │
│        │Action │        │Step  │ │Seq  │ │Hand │       │
│        │Planner│        │Planner│ │Plan │ │Plan │       │
│        └───┬───┘        └───┬──┘ └──┬──┘ └──┬──┘       │
│            │                │       │      │            │
│        ┌───▼────────┐       │       │      │            │
│        │Generates   │       │       │      │            │
│        │  actions   │       │       │      │            │
│        │  greedily  │       │       │      │            │
│        └───┬────────┘       │       │      │            │
│            │                │       │      │            │
│        ┌───▼─────┐      ┌───▼──┐ ┌──▼──┐ ┌──▼──┐      │
│        │Execute  │      │Exec  │ │Exec │ │Call │      │
│        │Actions  │      │Step  │ │Seq  │ │Human│      │
│        │         │      │      │ │     │ │     │      │
│        └────┬────┘      └───┬──┘ └──┬──┘ └──┬──┘      │
│             │                │       │      │         │
│        ┌────▼──────┐         │       │      │         │
│        │Converge   │         │       │      │         │
│        │To Goal    │         │       │      │         │
│        │           │         │       │      │         │
│        └────┬──────┘     ┌───▼──┐ ┌──▼──┐ ┌──▼──┐    │
│             │             │Done  │ │Done │ │Wait │    │
│        OUTPUT: Result    └──────┘ └─────┘ └─────┘    │
│             │                                        │
│        ┌────▼──────┐                                  │
│        │Final Plan │                                  │
│        └───────────┘                                  │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### 5.2 Plan Execution State Machine

```
┌──────────────────────────────────────────────────────────┐
│      Plan Execution State Machine                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│          ┌──────────────────┐                            │
│          │   CREATED        │                            │
│          │ Plan generated   │                            │
│          └────────┬─────────┘                            │
│                   │                                      │
│                   ▼                                      │
│          ┌──────────────────┐                            │
│          │   VALIDATING     │                            │
│          │ Check functions  │                            │
│          │ exist            │                            │
│          └────────┬─────────┘                            │
│                   │                                      │
│           ┌───────┴────────┐                             │
│           │                │                             │
│       Valid            Invalid                           │
│           │                │                             │
│           ▼                ▼                             │
│    ┌──────────┐      ┌──────────┐                       │
│    │EXECUTING │      │FAILED    │                       │
│    └────┬─────┘      │Validation│                       │
│         │            │Failed    │                       │
│         │            └──────────┘                       │
│         ▼                                                │
│    ┌───────────┐                                        │
│    │STEPS      │                                        │
│    │Execute    │                                        │
│    │each step  │                                        │
│    └────┬──────┘                                        │
│         │                                                │
│    ┌────┴────────────┐                                  │
│    │                 │                                  │
│  Success         Error/Exception                        │
│    │                 │                                  │
│    ▼                 ▼                                  │
│┌───────┐         ┌──────────┐                          │
││NEXT   │         │ERROR     │                          │
││STEP   │         │Retry?    │                          │
│└───┬───┘         └─────┬────┘                          │
│    │                   │                               │
│    │          ┌────────┴────────┐                      │
│    │          │                 │                      │
│    │        Retry           Abort                      │
│    │          │                 │                      │
│    │          ▼                 ▼                      │
│    │     ┌────────┐         ┌──────────┐             │
│    │     │STEPS   │         │FAILED    │             │
│    │     │        │         │Plan      │             │
│    │     └────┬───┘         │Failed    │             │
│    │          │             └──────────┘             │
│    │          │                                      │
│    ▼          ▼                                      │
│  ┌──────────────┐                                    │
│  │ ALL COMPLETE │                                    │
│  └────────┬─────┘                                    │
│           │                                          │
│           ▼                                          │
│    ┌────────────┐                                    │
│    │SUCCEEDED  │                                    │
│    │Return     │                                    │
│    │Result     │                                    │
│    └───────────┘                                    │
│                                                    │
└──────────────────────────────────────────────────┘
```

---

## 6. Function Invocation Flow

### 6.1 Complete Invocation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│         Function Invocation Complete Flow              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. REQUEST PREPARATION                                 │
│    ┌──────────────────────────────┐                    │
│    │ Parse Plugin:Function         │                   │
│    │ "Math:Add"                    │                   │
│    └────────┬─────────────────────┘                    │
│             │                                           │
│ 2. PARAMETER RESOLUTION                                │
│    ┌────────▼──────────────────────┐                   │
│    │ Get Arguments                  │                   │
│    │ { "a": 5, "b": 3 }            │                   │
│    └────────┬──────────────────────┘                   │
│             │                                           │
│ 3. TYPE COERCION                                       │
│    ┌────────▼──────────────────────┐                   │
│    │ Convert to Expected Types      │                   │
│    │ int: 5, int: 3                │                   │
│    └────────┬──────────────────────┘                   │
│             │                                           │
│ 4. INVOCATION                                          │
│    ┌────────▼──────────────────────┐                   │
│    │ Execute Function               │                   │
│    │ Result: 8                      │                   │
│    └────────┬──────────────────────┘                   │
│             │                                           │
│ 5. RESULT FORMATTING                                   │
│    ┌────────▼──────────────────────┐                   │
│    │ Serialize Result               │                   │
│    │ "8"                            │                   │
│    └────────┬──────────────────────┘                   │
│             │                                           │
│ 6. LOGGING & TELEMETRY                                 │
│    ┌────────▼──────────────────────┐                   │
│    │ Log: Success                   │                   │
│    │ Duration: 10ms                 │                   │
│    │ Tokens: 5                      │                   │
│    └────────┬──────────────────────┘                   │
│             │                                           │
│ 7. RETURN                                              │
│    ┌────────▼──────────────────────┐                   │
│    │ Return Result to Caller        │                   │
│    │ { value: 8, success: true }   │                   │
│    └──────────────────────────────┘                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Structured Output Schema

### 7.1 Output Validation Pipeline

```
┌──────────────────────────────────────────────────┐
│      Structured Output Validation               │
├──────────────────────────────────────────────────┤
│                                                 │
│  ┌────────────────┐                             │
│  │ LLM Response   │                             │
│  │ (Raw Text)     │                             │
│  └────────┬───────┘                             │
│           │                                     │
│           ▼                                     │
│  ┌────────────────────┐                         │
│  │ Parse JSON/Schema  │                         │
│  │ Extract structure  │                         │
│  └────────┬───────────┘                         │
│           │                                     │
│           ▼                                     │
│  ┌────────────────────┐                         │
│  │ Type Validation    │                         │
│  │ Check types match  │                         │
│  │ schema             │                         │
│  └────────┬───────────┘                         │
│           │                                     │
│    ┌──────┴──────┐                              │
│    │             │                              │
│  Valid       Invalid                            │
│    │             │                              │
│    ▼             ▼                              │
│ ┌───────┐  ┌─────────────┐                     │
│ │Field  │  │Validation   │                     │
│ │Validity│  │Error: retry │                     │
│ │Check  │  │or handle    │                     │
│ └───┬───┘  └─────────────┘                     │
│     │                                           │
│     ▼                                           │
│ ┌──────────────┐                                │
│ │Required      │                                │
│ │Fields        │                                │
│ │Present?      │                                │
│ └────┬─────────┘                                │
│      │                                           │
│  ┌───┴──────┐                                   │
│  │          │                                   │
│ Yes        No                                  │
│  │          │                                   │
│  ▼          ▼                                   │
│ ┌──┐   ┌──────────┐                            │
│ │✓ │   │Set       │                            │
│ │ │   │defaults? │                            │
│ └─┘   └──┬───────┘                             │
│          │                                     │
│      ┌───┴───┐                                  │
│      │       │                                  │
│    Set    No defaults                          │
│   default │                                     │
│      │    ▼                                     │
│      │  ┌──────────┐                            │
│      │  │Validation│                            │
│      │  │Failed    │                            │
│      │  └──────────┘                            │
│      │                                          │
│      ▼                                          │
│  ┌──────────────┐                               │
│  │Return Typed  │                               │
│  │Object        │                               │
│  │(Pydantic/C#) │                               │
│  └──────────────┘                               │
│                                                 │
└──────────────────────────────────────────────────┘
```

---

## 8. Azure Integration Architecture

### 8.1 Azure Services Integration

```
┌────────────────────────────────────────────────────┐
│     Semantic Kernel with Azure Services           │
├────────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │         Semantic Kernel                     │ │
│  │                                             │ │
│  │  ┌───────────────────────────────────────┐ │ │
│  │  │ Chat Completion                       │ │ │
│  │  └────────┬─────────────────────────────┘ │ │
│  │           │                               │ │
│  │           ▼                               │ │
│  │  ┌───────────────────────────────────────┐ │ │
│  │  │ Azure OpenAI Service                  │ │ │
│  │  │ • gpt-4                               │ │ │
│  │  │ • gpt-4-turbo                         │ │ │
│  │  │ • gpt-3.5-turbo                       │ │ │
│  │  └───┬──────────────────────────────────┘ │ │
│  │      │                                    │ │
│  │      ▼                                    │ │
│  │  Azure Managed Identity                  │ │
│  │  or API Key Auth                         │ │
│  │                                          │ │
│  │  ┌───────────────────────────────────────┐ │ │
│  │  │ Embeddings                            │ │ │
│  │  └────────┬─────────────────────────────┘ │ │
│  │           │                               │ │
│  │           ▼                               │ │
│  │  ┌───────────────────────────────────────┐ │ │
│  │  │ Azure OpenAI Embeddings               │ │ │
│  │  │ • text-embedding-3-small              │ │ │
│  │  │ • text-embedding-3-large              │ │ │
│  │  └───┬──────────────────────────────────┘ │ │
│  │      │                                    │ │
│  │      ▼                                    │ │
│  │  ┌───────────────────────────────────────┐ │ │
│  │  │ Azure AI Search                       │ │ │
│  │  │ Vector Store for Memory               │ │ │
│  │  │ • Collections                         │ │ │
│  │  │ • Vector Search                       │ │ │
│  │  │ • Semantic Ranking                    │ │ │
│  │  └────────────────────────────────────────┘ │ │
│  │                                             │ │
│  │  ┌────────────────────────────────────────┐ │ │
│  │  │ Additional Azure Services              │ │ │
│  │  │ • Key Vault (Secrets)                 │ │ │
│  │  │ • App Configuration                   │ │ │
│  │  │ • Application Insights (Telemetry)    │ │ │
│  │  │ • Functions (Deployment)              │ │ │
│  │  └────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└────────────────────────────────────────────────────┘
```

---

## 9. ReAct Pattern Flow

### 9.1 Reasoning and Acting Loop

```
┌─────────────────────────────────────────────────────────┐
│           ReAct (Reasoning + Acting) Pattern            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Start                                                  │
│  ├─ Thought: "I need to find the stock price"          │
│  │                                                      │
│  ├─ Action: "Use StockPrice plugin"                    │
│  │ └─ Action Input: {"symbol": "AAPL"}                 │
│  │                                                      │
│  ├─ Observation: "$245.50"                              │
│  │                                                      │
│  ├─ Thought: "Stock price is rising, let me check      │
│  │           historical data"                           │
│  │                                                      │
│  ├─ Action: "Use HistoricalData plugin"                │
│  │ └─ Action Input: {"symbol": "AAPL",                 │
│  │                   "days": 30}                        │
│  │                                                      │
│  ├─ Observation: "[240, 242, 243, 245, 245.50]"        │
│  │                                                      │
│  ├─ Thought: "Stock is in uptrend. I have enough        │
│  │           information. Let me provide analysis."     │
│  │                                                      │
│  └─ Final Answer: "AAPL is at $245.50 showing           │
│     uptrend over 30 days"                               │
│                                                         │
│  ┌──────────────────────────────────────────────┐      │
│  │  Loop Control                                │      │
│  │  • Max iterations: 10                        │      │
│  │  • Exit on: Final Answer                     │      │
│  │  • Exit on: Error after 3 retries            │      │
│  │  • Exit on: Time limit exceeded              │      │
│  └──────────────────────────────────────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Prompt Template Structure

### 10.1 Template Variable Substitution

{% raw %}
```

┌────────────────────────────────────────────────┐
│     Prompt Template Variable Flow              │
├────────────────────────────────────────────────┤
│                                                │
│  Template:                                     │
│  ┌────────────────────────────────────────┐  │
│  │ "Translate {{$language}} text:        │  │
│  │  {{$input}} to French"                 │  │
│  └────┬──────────────────────────────────┘  │
│       │                                      │
│  Variables Dictionary:                       │
│  ┌────────────────────────────────────────┐  │
│  │ {                                      │  │
│  │   "language": "English",               │  │
│  │   "input": "Hello, world!"             │  │
│  │ }                                      │  │
│  └────┬──────────────────────────────────┘  │
│       │                                      │
│       ▼                                      │
│  Substitution:                               │
│  ┌────────────────────────────────────────┐  │
│  │ "Translate English text:               │  │
│  │  Hello, world! to French"              │  │
│  └────────────────────────────────────────┘  │
│                                               │
└────────────────────────────────────────────────┘

```
{% endraw %}

---

This comprehensive diagram document illustrates the key architectural patterns and flows in Semantic Kernel, helping developers understand how components interact and orchestrate together.





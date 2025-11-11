# Microsoft AutoGen 0.4: Architecture and Diagrams

This document provides visual representations of AutoGen's architecture, components, and data flows.

## Core Architecture Diagrams

### 1. AutoGen Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (Your code using agents, teams, orchestration)             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              AutoGen Agent Chat Layer                        │
├──────────────────────────────────────────────────────────────┤
│ • AssistantAgent       • UserProxyAgent   • CodeExecutorAgent│
│ • GroupChat Managers   • Termination Conditions             │
│ • Message Types        • Event System                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              AutoGen Core Layer                              │
├──────────────────────────────────────────────────────────────┤
│ • Runtime Execution Engine                                   │
│ • Agent Registration & Lifecycle                             │
│ • Message Routing & Topics                                   │
│ • Context Management                                         │
│ • Event Emission                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│              Extensions Layer (autogen-ext)                  │
├──────────────────────────────────────────────────────────────┤
│ Model Clients  │ Storage    │ Tool Helpers │ Code Execution  │
│  • OpenAI      │ • SQL      │  • Builders  │  • Sandboxes    │
│  • Azure       │ • Vector   │  • Schemas   │  • Executors    │
│  • Bedrock     │ • File     │              │  • Timeouts     │
│  • Google      │            │              │                 │
└──────────────────────────────────────────────────────────────┘
```

### 2. Agent Lifecycle

```
                    Agent Creation
                          │
                          v
                    ┌─────────────┐
                    │   Created   │
                    └─────────────┘
                          │
                          v
                    ┌─────────────┐
                    │ Initialised │ ◄──┐ (Register with runtime)
                    └─────────────┘    │
                          │            │
                  ┌───────┴──────┐     │
                  v              v     │
            ┌─────────┐      ┌──────┐  │
            │ Running │─────►│Error │──┴─► (Retry)
            └────┬────┘      └──────┘
                 │
            ┌────┴────┐
            v         v
         ┌──────┐  ┌─────────┐
         │Paused│  │Completed│
         └──┬───┘  └─────────┘
            │
            v
         ┌─────────┐
         │ Resumed │
         └────┬────┘
              │
              v (Continue)
          ┌─────────┐
          │ Running │ (again)
          └─────────┘
```

### 3. Single Agent Message Processing Flow

```
┌─────────────┐
│User Message │
└──────┬──────┘
       │
       v
┌──────────────────────┐
│Add to Message History│
└──────┬───────────────┘
       │
       v
┌──────────────────────────────────────────┐
│   Prepare LLM Prompt                      │
├──────────────────────────────────────────┤
│ • System Message                          │
│ • Message History                         │
│ • Available Tools                         │
│ • Additional Context                      │
└──────┬───────────────────────────────────┘
       │
       v
┌──────────────────────┐
│Call LLM Model        │
│(Chat Completion)     │
└──────┬───────────────┘
       │
       v
┌────────────────────────────┐
│Parse LLM Response          │
└────┬───────────┬──────┬────┘
     │           │      │
Text │    Tool   │ Error│
     │    Call   │      │
     v           v      v
┌────────┐  ┌──────────────┐  ┌────────┐
│Return  │  │Execute Tool  │  │ Retry  │
│Message │  │              │  │ Logic  │
└────────┘  └──────┬───────┘  └────────┘
                   │
                   v
            ┌─────────────┐
            │Tool Result  │
            └──────┬──────┘
                   │
                   v (If reflect_on_tool_use)
            ┌──────────────────┐
            │Call Model Again  │
            │with Tool Result  │
            └──────┬───────────┘
                   │
                   v
            ┌──────────────┐
            │Final Response│
            └──────────────┘
```

### 4. Multi-Agent Communication Flow

```
                          Team/Group Chat
                                 │
                   ┌─────────────┬─────────────┐
                   │             │             │
              ┌────v────┐  ┌─────v─────┐ ┌────v────┐
              │ Agent 1  │  │ Agent 2   │ │ Agent 3 │
              │(Planner) │  │(Researcher)│ │(Coder) │
              └────┬─────┘  └─────┬─────┘ └────┬────┘
                   │             │             │
                   └─────────────┼─────────────┘
                                 │
                                 v
                    ┌────────────────────────┐
                    │ Speaker Selection      │
                    │ • LLM-based OR         │
                    │ • Rule-based OR        │
                    │ • Round-robin OR       │
                    │ • Custom Function      │
                    └────────┬───────────────┘
                             │
                    ┌────────v────────┐
                    │Selected Agent   │
                    │Generates Reply  │
                    └────────┬────────┘
                             │
                    ┌────────v────────────┐
                    │Broadcast to Group   │
                    └────────┬────────────┘
                             │
                    ┌────────v──────────────┐
                    │Check Termination      │
                    │Condition              │
                    └────┬──────────┬───────┘
                      No │          │ Yes
                    ┌────v─┐      ┌─v──────┐
                    │Loop  │      │Complete│
                    │Again │      │        │
                    └──────┘      └────────┘
```

### 5. Tool Execution Architecture

```
┌─────────────────────┐
│ Agent on_messages() │
└──────────┬──────────┘
           │
           v
┌──────────────────────────────────┐
│ Model generates Tool Call(s)      │
│ (Function name + arguments)       │
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│ Tool Call Handler                 │
│ • Validate arguments              │
│ • Check tool exists               │
│ • Measure execution time          │
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│ Execute Tool Function             │
│ • Handle async/sync               │
│ • Apply timeout                   │
│ • Catch exceptions                │
└──────────┬────────────────────────┘
           │
      ┌────┴────┐
      │          │
   Success    Error
      │          │
      v          v
┌─────────┐  ┌──────────────┐
│ Result  │  │ Error Result │
└────┬────┘  └──────┬───────┘
     │              │
     └──────┬───────┘
            │
            v
┌─────────────────────────────────┐
│ Return ToolResultMessage to LLM │
│ • Include result or error       │
│ • Tool use ID for reference     │
└──────────┬──────────────────────┘
           │
           v
┌─────────────────────────────────┐
│ Model can:                       │
│ • Generate response              │
│ • Call another tool              │
│ • Request clarification          │
└─────────────────────────────────┘
```

### 6. Distributed Agent Architecture

```
                    ┌─────────────────────┐
                    │  Agent Runtime      │
                    │  (Orchestrator)     │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────v────┐  ┌──────v────┐  ┌────v──────┐
         │  Agent 1  │  │  Agent 2  │  │  Agent 3  │
         │ (Local)   │  │(Remote 1) │  │(Remote 2) │
         └──────┬────┘  └──────┬────┘  └────┬──────┘
                │              │             │
                └──────────────┼─────────────┘
                               │
                    ┌──────────v──────────┐
                    │ Message Broker      │
                    │ (Topic-based)       │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
             Topic A         Topic B        Topic C
                │              │              │
         ┌──────v────┐  ┌──────v────┐  ┌────v──────┐
         │Subscriber │  │Subscriber │  │Subscriber │
         │    Set 1  │  │    Set 2  │  │    Set 3  │
         └───────────┘  └───────────┘  └───────────┘
```

### 7. Memory System Architecture

```
┌──────────────────────────────────────────┐
│         Agent Memory Hierarchy            │
└──────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         v            v            v
    ┌────────┐  ┌──────────┐ ┌──────────┐
    │ Working│  │ Episodic │ │Semantic  │
    │ Memory │  │ Memory   │ │ Memory   │
    │ (L1)   │  │ (L2)     │ │ (L3)     │
    └────┬───┘  └────┬─────┘ └────┬─────┘
         │           │            │
         v           v            v
    ┌────────────────────────────────────┐
    │    Shared Team Memory              │
    │  (Accessible to all agents)        │
    └─────────────┬──────────────────────┘
                  │
    ┌─────────────┼──────────────┐
    │             │              │
    v             v              v
┌────────┐  ┌──────────┐  ┌────────────┐
│Vector  │  │Document  │  │Relational  │
│Database│  │Store     │  │Database    │
└────────┘  └──────────┘  └────────────┘
```

### 8. Event System Flow

```
┌─────────────────────┐
│     Application     │
│  (Agent Operation)  │
└──────────┬──────────┘
           │
           v
┌──────────────────────────────────┐
│  Event Generated                 │
│  • Type: AgentStarted, etc.      │
│  • Timestamp                     │
│  • Metadata                      │
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│  Event Bus / Dispatcher          │
│  (Central event hub)             │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┬─────────┬──────────┐
    │             │         │          │
    v             v         v          v
┌────────┐  ┌─────────┐ ┌──────┐ ┌──────┐
│Logging │  │Monitoring│Tracing│Storage│
│Handler │  │ Handler  │Handler│Handler│
└────────┘  └─────────┘ └──────┘ └──────┘
    │             │         │          │
    └─────────────┴─────────┴──────────┘
                  │
                  v
          (External Systems)
          • Logs
          • Metrics
          • Traces
          • Files
```

### 9. Structured Output Pipeline

```
┌──────────────────────┐
│ Define Schema        │
│ (Python dataclass)   │
└──────────┬───────────┘
           │
           v
┌──────────────────────┐
│ Generate JSON Schema │
│ (from Python types)  │
└──────────┬───────────┘
           │
           v
┌──────────────────────────────┐
│ Pass to LLM                   │
│ (as function parameter)       │
└──────────┬───────────────────┘
           │
           v
┌──────────────────────────────┐
│ LLM Generates JSON            │
│ (Structured output)           │
└──────────┬───────────────────┘
           │
           v
┌──────────────────────────────┐
│ Parse JSON                    │
│ • Validate against schema     │
│ • Type conversion             │
└──────────┬───────────────────┘
           │
           v
┌──────────────────────────────┐
│ Create Typed Object           │
│ (Python dataclass instance)   │
└──────────┬───────────────────┘
           │
           v
┌──────────────────────────────┐
│ Use Structured Data           │
│ • Type-safe operations        │
│ • IDE autocomplete            │
│ • Validation built-in         │
└──────────────────────────────┘
```

### 10. Model Context Protocol (MCP) Integration

```
┌─────────────────────────────┐
│   AutoGen Agent             │
└──────────┬──────────────────┘
           │
           v
┌──────────────────────────────┐
│  MCP Client                  │
│  (Tool consumer)             │
└──────────┬───────────────────┘
           │
    ┌──────v──────┐
    │ Send Tool   │
    │ Requests    │
    └──────┬──────┘
           │
           v
┌──────────────────────────────┐
│  MCP Transport Layer         │
│  • Stdio                     │
│  • HTTP                      │
│  • WebSocket                 │
└──────────┬───────────────────┘
           │
           v
┌──────────────────────────────┐
│  MCP Server                  │
│  • Tool provider             │
│  • Resource provider         │
│  • Capability exposer        │
└──────────┬───────────────────┘
           │
    ┌──────┴───────────────┐
    │                      │
    v                      v
┌─────────────┐      ┌──────────┐
│ Tools       │      │Resources │
│ (Functions) │      │ (Data)   │
└─────────────┘      └──────────┘
```

### 11. Request-Response Pattern (Multi-Agent)

```
┌──────────┐  Request    ┌──────────┐
│Agent A   ├────────────►│Agent B   │
│(Requester)             │(Handler) │
└──────────┘  Response   └────┬─────┘
     ▲        (with ID)        │
     │                         v
     │         ┌───────────────────┐
     │         │Process Request    │
     │         │Perform Task       │
     │         └────────┬──────────┘
     │                  │
     │                  v
     │         ┌───────────────────┐
     │         │Generate Response  │
     │         │(with same ID)     │
     │         └────────┬──────────┘
     │                  │
     └──────────────────┘
```

### 12. Publish-Subscribe Pattern (Multi-Agent)

```
┌──────────────────────────────────────┐
│         Message Broker               │
│     (Topic-Based)                    │
└──────────┬───────────────────────────┘
           │
    ┌──────┴──────┬───────┬─────────┐
    │             │       │         │
Topic:         Task    Alert   Update
Weather         │       │         │
    │           │       │         │
    v           v       v         v
 ┌────┐     ┌────┐  ┌───┐    ┌────┐
 │ A1 │     │ B1 │  │ C1│    │ D1 │
 │ A2 │  ┌─►│ B2 │  │ C2│    │    │
 │    │  │  │    │  │   │    │    │
 └────┘  │  └────┘  └───┘    └────┘
 Sub1    │  Sub2     Sub3     Sub4
         │
    ┌────┴──────┐
    │ Publishers│
    │ (Producers)
    └─────┬─────┘
          │
      ┌───┴────┐
      v        v
    ┌──┐    ┌──┐
    │P1│    │P2│
    └──┘    └──┘
```

### 13. Code Execution Flow

```
┌─────────────────────┐
│ Agent generates     │
│ Python code         │
└──────────┬──────────┘
           │
           v
┌──────────────────────────────┐
│ Code Extraction              │
│ (Parse from LLM response)    │
└──────────┬───────────────────┘
           │
           v
┌──────────────────────────────┐
│ Security Validation          │
│ • Check allowed imports      │
│ • Scan for dangerous calls   │
└──────────┬───────────────────┘
           │
           v
    ┌──────┴──────┐
    │Use Docker   │Use Direct
    │(Sandbox)    │(Local)
    └──┬──────┬───┘
       │      │
       v      v
   ┌──────┐ ┌─────┐
   │Docker│ │Python
   │Exec  │ │Proc │
   └───┬──┘ └──┬──┘
       │       │
       └───┬───┘
           │
           v
    ┌──────────────────┐
    │ Capture Output   │
    │ • stdout         │
    │ • stderr         │
    │ • Return value   │
    └────────┬─────────┘
             │
             v
    ┌──────────────────┐
    │ Apply Timeout    │
    │ (if exceeded)    │
    │ Kill process     │
    └────────┬─────────┘
             │
             v
    ┌──────────────────┐
    │ Return Result    │
    │ to Agent         │
    └──────────────────┘
```

## Orchestration Patterns

### 14. Sequential Orchestration

```
Task
 │
 v
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Agent 1    │─────►│  Agent 2    │─────►│  Agent 3    │
│  (Plan)     │      │  (Execute)  │      │  (Synthesis)│
└─────────────┘      └─────────────┘      └─────────────┘
 Output: Plan    Output: Execution    Output: Result
                   Details
```

### 15. Parallel Orchestration

```
                    Task
                     │
         ┌───────────┼───────────┐
         │           │           │
         v           v           v
    ┌────────┐   ┌────────┐   ┌────────┐
    │Agent 1 │   │Agent 2 │   │Agent 3 │
    └────┬───┘   └────┬───┘   └────┬───┘
         │           │           │
         └───────────┼───────────┘
                     │
                     v
            ┌──────────────────┐
            │ Synthesis Agent  │
            │ (Combines results)
            └────────┬─────────┘
                     │
                     v
                  Result
```

### 16. Hierarchical Orchestration

```
           Task
            │
            v
       ┌────────────┐
       │  Manager   │
       │  (Plan &   │
       │  Delegate) │
       └─────┬──────┘
             │
    ┌────────┼────────┐
    │        │        │
    v        v        v
 ┌────┐  ┌────┐  ┌────┐
 │ S1 │  │ S2 │  │ S3 │
 │    │  │    │  │    │
 └─┬──┘  └─┬──┘  └─┬──┘
   │       │       │
   └───────┼───────┘
           │
           v
      ┌────────────┐
      │ Synthesise │
      │ Results    │
      └─────┬──────┘
            │
            v
         Result
```

## State Transitions

### 17. Multi-Agent Conversation State Machine

```
┌─────────────────────────────┐
│   Initialisation            │
│ • Create agents             │
│ • Register with runtime     │
│ • Setup topics              │
└──────────────┬──────────────┘
               │
               v
        ┌──────────────┐
        │ Waiting for  │
        │ First Message│
        └──────┬───────┘
               │
     ┌─────────v────────┐
     │ First message    │
     │ received         │
     └─────────┬────────┘
               │
               v
        ┌──────────────────┐
        │ Agent Processing │
        │ (Message routing)│
        └────────┬─────────┘
                 │
         ┌───────┴─────────┐
         │                 │
         v                 v
    ┌─────────┐       ┌──────────────┐
    │ Continue│       │Termination   │
    │ (More   │       │Condition Met │
    │ messages)       │              │
    └────┬────┘       └──────┬───────┘
         │                   │
         v                   v
    (Loop to          ┌──────────────────┐
     agent            │ Finalisation     │
     processing)      │ • Cleanup        │
                      │ • Collect results│
                      │ • Close resources
                      └──────┬───────────┘
                             │
                             v
                        ┌──────────────┐
                        │ Complete     │
                        │ Return Result│
                        └──────────────┘
```

## Data Structure Diagrams

### 18. Message Object Hierarchy

```
┌──────────────────────┐
│  BaseMessage         │
│  • content: str      │
│  • source: str       │
│  • timestamp: float  │
│  • message_id: str   │
└──────────┬───────────┘
           │
    ┌──────┴─────────┬──────────┬─────────┐
    │                │          │         │
    v                v          v         v
┌──────────┐  ┌──────────┐  ┌──────┐  ┌─────────┐
│TextMsg   │  │ToolCall │  │Tool  │  │System   │
│          │  │Message  │  │Result│  │Message  │
│• models  │  │• tool   │  │• err │  │• type   │
│_used     │  │_name    │  │or    │  │         │
└──────────┘  │• tool   │  │• is_ │  └─────────┘
              │_arguments  │error │
              │• tool_call │      │
              │_id        │      │
              └──────────┘  └──────┘
```

### 19. Agent Type Hierarchy

```
┌──────────────────────┐
│   BaseAgent (ABC)    │
│  • name: str         │
│  • description: str  │
│  • on_messages()     │
└──────────┬───────────┘
           │
    ┌──────┴──────┬────────────────┐
    │             │                │
    v             v                v
┌─────────┐  ┌──────────────┐  ┌──────┐
│Assistant│  │UserProxy     │  │Code  │
│Agent    │  │Agent         │  │Executor
│• model_ │  │• input_func  │  │• work
│client   │  │• code_exe    │  │_dir
│• tools  │  │_config       │  │      │
│• system │  │              │  │      │
│_message │  │              │  │      │
└─────────┘  └──────────────┘  └──────┘
```

## Deployment Architecture

### 20. Local Deployment

```
┌─────────────────────────────────────┐
│     Local Machine                   │
├─────────────────────────────────────┤
│ ┌───────────────────────────────┐   │
│ │  Python / Node.js Runtime     │   │
│ ├───────────────────────────────┤   │
│ │ ┌─────────────────────────────┤   │
│ │ │  AutoGen Application        │   │
│ │ ├─────────────────────────────┤   │
│ │ │ • Agents                    │   │
│ │ │ • Teams                     │   │
│ │ │ • Orchestration             │   │
│ │ └─────────────────────────────┤   │
│ │ ┌─────────────────────────────┤   │
│ │ │  Local Storage              │   │
│ │ │ • SQLite / JSON files       │   │
│ │ │ • Vector DB (Faiss, etc)    │   │
│ │ └─────────────────────────────┤   │
│ └───────────────────────────────┘   │
│         │                            │
│         v                            │
│  ┌─────────────────┐                 │
│  │  LLM API Call   │◄────────────┐   │
│  │  (OpenAI, etc)  │             │   │
│  └─────────────────┘             │   │
│         │            Network      │   │
│         └──────────────────────────┘   │
└─────────────────────────────────────┘
```

### 21. Cloud Deployment (Azure)

```
┌─────────────────────────────────────────┐
│         Azure Cloud                     │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │  Azure Container Instances / AKS    │ │
│ │  ┌─────────────────────────────────┐ │ │
│ │  │ Docker Container                │ │ │
│ │  │ • AutoGen Application           │ │ │
│ │  │ • Python Runtime                │ │ │
│ │  └─────────────────────────────────┘ │ │
│ └──────────────┬──────────────────────┘ │
│                │                        │
│        ┌───────┴────────┐               │
│        v                v               │
│ ┌────────────────┐ ┌──────────────────┐ │
│ │Azure Cosmos DB │ │Azure OpenAI      │ │
│ │(Persistence)  │ │(LLM Access)      │ │
│ └────────────────┘ └──────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
```

## Message Flow Examples

### 22. Complex Multi-Agent Task Flow

```
User: "Research AI trends and write a summary"
        │
        v
┌────────────────────────────────────┐
│ Planning Agent                      │
│ • Breaks down task                  │
│ • Creates plan: research + write    │
└────────────────────────────────────┘
        │ Plan created
        │
        ├─────────────────────────┐
        │                         │
        v                         v
┌─────────────────┐       ┌────────────────┐
│ Research Agent  │       │ Writing Agent  │
│ • Searches web  │       │ • Drafts text  │
│ • Collects data │◄──────│ • Waits for    │
│                 │       │   research     │
└─────────────────┘       └────────────────┘
        │ Research data
        │
        v
│ Research: [5 findings] ──► Writing Agent
│                              (Continues)
│                                │
│                                v
│                           ┌───────────────┐
│                           │ Review Agent  │
│                           │ • Checks text │
│                           │ • Suggests    │
│                           │   improvements
│                           └───────────────┘
│                                │
│                                v
│                           ┌───────────────┐
│                           │ Final Summary │
│                           │ Returned to   │
│                           │ User          │
│                           └───────────────┘
```

---

**End of Diagrams Document**

This document provides comprehensive visual representations of AutoGen 0.4's architecture, making it easier to understand how components interact and data flows through the system.


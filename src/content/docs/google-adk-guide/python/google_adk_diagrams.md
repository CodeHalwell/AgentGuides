---
title: "Google ADK Architecture Diagrams (Python)"
description: "Text diagrams of Google ADK's runtime, agents, workflows, sessions, memory and deployment, with links to the pages that explain each part."
framework: "google-adk"
language: "python"
sidebar:
  label: "Diagrams"
  order: 99
---

---

## ADK Core Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│     (adk web / api_server, your FastAPI app, A2A, CLI)      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Runner (App + plugins)                   │
│  • Drives the root agent or Workflow for one invocation     │
│  • Loads and saves sessions and state                       │
│  • Calls plugins around agents, models and tools            │
│  • Yields events to the caller                              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
         ┌──────────┐   ┌─────────┐   ┌─────────┐
         │ Agent    │   │ Tools   │   │ Models  │
         │ Logic    │   │ Executor│   │ API     │
         └──────────┘   └─────────┘   └─────────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │ Sessions │  │ Artifacts│  │ Memory   │
         │ Service  │  │ Service  │  │ Service  │
         └──────────┘  └──────────┘  └──────────┘
```

---

## Agent Hierarchy

```text
                    ┌────────────────────┐
                    │  Root Coordinator  │
                    │     Agent          │
                    └────────┬───────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Specialist1 │  │  Specialist2 │  │  Specialist3 │
    │   (Research) │  │  (Analysis)  │  │  (Writing)   │
    └──────────────┘  └──────────────┘  └──────────────┘
          │                  │                  │
    ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
    │           │      │           │      │           │
    ▼           ▼      ▼           ▼      ▼           ▼
  ┌───┐       ┌───┐  ┌───┐       ┌───┐  ┌───┐       ┌───┐
  │ A │       │ B │  │ C │       │ D │  │ E │       │ F │
  └───┘       └───┘  └───┘       └───┘  └───┘       └───┘
 (Tools)    (Tools) (Tools)     (Tools) (Tools)     (Tools)
```

---

## Workflow chain

`Workflow(edges=[(START, research, analysis, writer, editor)])`: each node receives the previous node's output.

```text
User Query
    │
    ▼
┌──────────────────┐
│ Research Agent   │
│ • Search web     │
│ • Gather data    │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Analysis Agent   │
│ • Process data   │
│ • Extract key    │
│   points         │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Writer Agent     │
│ • Draft content  │
│ • Format output  │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Editor Agent     │
│ • Review         │
│ • Polish         │
└─────────┬────────┘
          │
          ▼
      Result
```

---

## Workflow fan-out and join

`Workflow(edges=[(START, (parser, analyser, checker), JoinNode(name="join"), aggregate)])`: the branches run concurrently and the `JoinNode` passes `{branch: output}` to the next node.

```text
              User Input
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │Parser  │ │Analyser│ │Checker │
    │        │ │        │ │        │
    │Process │ │Identify│ │Verify  │
    │input   │ │patterns│ │results │
    └────┬───┘ └───┬────┘ └───┬────┘
         │         │         │
         └─────────┼─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Aggregator Agent │
          │ • Combine results│
          │ • Generate final │
          │   response       │
          └──────────┬───────┘
                     │
                     ▼
                  Output
```

---

## ReAct Loop Pattern

```text
                    ┌─────────────────┐
                    │  Start Problem  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   THOUGHT:      │
                    │ What do I know? │
                    │ What do I need? │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   ACTION:       │
                    │ Call a tool or  │
                    │ make conclusion │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  OBSERVATION:   │
                    │ Tool result /   │
                    │ feedback        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  REASONING:     │
                    │ Interpret what  │
                    │ I learned       │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                                 │
            ▼                                 ▼
    ┌──────────────┐                  ┌──────────────┐
    │ More Work    │                  │ Final Answer │
    │ Needed?      │                  │ Ready        │
    └──────┬───────┘                  └──────┬───────┘
           │                                 │
      Yes  │                                 │  No
           └─────────────┬─────────────────┘
                         │
         (Loop back to THOUGHT)
```

---

## Tool Invocation Flow

```text
        ┌──────────────────┐
        │   Agent Decides  │
        │ to Call Tool     │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Construct Tool   │
        │ Call with Params │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Validate Inputs  │
        │ Check Schemas    │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Execute Tool     │
        │ (Sync/Async)     │
        └────────┬─────────┘
                 │
         ┌───────┴────────┐
         │                │
         ▼                ▼
    ┌────────┐        ┌─────────┐
    │Success │        │ Error   │
    └────┬───┘        └────┬────┘
         │                 │
         ▼                 ▼
    ┌─────────┐        ┌─────────┐
    │ Result  │        │ Handle  │
    │Feedback │        │ Error   │
    └────┬────┘        └────┬────┘
         │                  │
         └──────────┬───────┘
                    │
                    ▼
         ┌────────────────────┐
         │ Update Agent State │
         │ Continue Reasoning │
         └────────────────────┘
```

---

## Session Management Flow

```text
┌──────────────────────────────────────────────────────┐
│               Session Lifecycle                       │
└──────────────────────────────────────────────────────┘

    Create Session
           │
           ▼
    ┌─────────────────┐
    │ Initialize      │
    │ State           │
    │ - User ID       │
    │ - Session ID    │
    │ - Metadata      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Store Session   │
    │ (Memory/DB)     │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
 ┌────────┐      ┌────────┐
 │Agent   │  ◄───┤Session │
 │Call 1  │      │Updated │
 └────────┘      └────────┘
    │
    ▼
 ┌────────┐
 │Agent   │
 │Call 2  │
 └────────┘
    │
    ▼
 ┌────────┐
 │Agent   │
 │Call N  │
 └────────┘
    │
    ▼
 Retrieve/Delete Session
```

---

## Memory Architecture

Three stores with different lifetimes. The agent sees session state and events directly; long-term memory is reached through a tool (`load_memory`, `preload_memory`) or `ctx.search_memory()`, and is filled by calling `add_session_to_memory()` (for example from an `after_agent_callback`).

```text
                       ┌──────────────────────┐
                       │      LlmAgent        │
                       └──────────┬───────────┘
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
 ┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
 │ Context window  │   │ Session             │   │ Memory service       │
 │ (one model call)│   │ events + state      │   │ (across sessions)    │
 │                 │   │ app: / user: /      │   │ load_memory tool,    │
 │ instruction,    │   │ session / temp:     │   │ preload_memory tool, │
 │ history, tools  │   │                     │   │ ctx.search_memory()  │
 └─────────────────┘   └──────────┬──────────┘   └──────────┬───────────┘
                                  │                         │
                                  ▼                         ▼
                      ┌───────────────────────┐  ┌───────────────────────┐
                      │ Session service       │  │ InMemoryMemoryService │
                      │ InMemory, Sqlite,     │  │ SqliteMemoryService   │
                      │ Database (SQLAlchemy),│  │ VertexAiMemoryBank    │
                      │ VertexAi (Agent       │  │ VertexAiRag           │
                      │ Platform), Firestore  │  │                       │
                      └───────────────────────┘  └───────────────────────┘

 Artifacts (files, images) sit beside these in an artifact service:
 InMemory, File, or GCS, scoped to a session or (with a "user:" filename) to the user.
```

See [Memory and artifacts](/google-adk-guide/python/memory-and-artifacts/) and [Runner, App and Sessions](/google-adk-guide/python/runner-and-sessions/).

---

## Google Cloud Integration Architecture

```text
┌───────────────────────────────────────────────────────────┐
│                      ADK Application                      │
└──────────────────────────────┬────────────────────────────┘
                               │
     ┌─────────────────────────┼──────────────────────────┐
     │                         │                          │
     ▼                         ▼                          ▼
┌───────────────┐     ┌─────────────────────┐    ┌───────────────────┐
│ Models        │     │ Built-in and Google │    │ Runtime           │
│ Gemini API or │     │ Cloud tools         │    │ Agent Engine,     │
│ Vertex AI     │     │ google_search,      │    │ Cloud Run, GKE    │
│ (Agent        │     │ VertexAiSearchTool, │    │                   │
│ Platform)     │     │ BigQuery, Spanner,  │    │                   │
│               │     │ Pub/Sub toolsets    │    │                   │
└───────────────┘     └─────────────────────┘    └───────────────────┘
     │                         │                          │
     ▼                         ▼                          ▼
┌───────────────┐     ┌─────────────────────┐    ┌───────────────────┐
│ Sessions and  │     │ Artifacts           │    │ Observability     │
│ Memory Bank   │     │ Cloud Storage       │    │ Cloud Trace,      │
│ (Agent        │     │ (GcsArtifactService)│    │ Monitoring and    │
│ Platform);    │     │                     │    │ Logging via       │
│ Cloud SQL via │     │                     │    │ --otel_to_cloud   │
│ Database-     │     │                     │    │                   │
│ SessionService│     │                     │    │                   │
└───────────────┘     └─────────────────────┘    └───────────────────┘
```

---

## Multi-Agent Orchestration

There is no separate routing service or tool broker: the agents themselves do the routing. An `LlmAgent` with `sub_agents` can hand the conversation to one of them (`transfer_to_agent`), wrap another agent as a tool (`AgentTool`), or the whole system can be a `Workflow` graph with fixed edges.

```text
                    ┌──────────────────────────┐
                    │  Coordinator (LlmAgent)  │
                    │  sub_agents=[...]        │
                    └────────────┬─────────────┘
                                 │
         ┌───────────────────────┼─────────────────────────┐
         │                       │                         │
         ▼                       ▼                         ▼
┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ transfer_to_agent│   │ AgentTool(agent)     │   │ Workflow edges       │
│ the sub-agent    │   │ called like a tool;  │   │ fixed order, fan-out │
│ takes over the   │   │ result returns to    │   │ and join, routing on │
│ conversation     │   │ the caller           │   │ a node's output      │
└──────────────────┘   └──────────────────────┘   └──────────────────────┘
         │                       │                         │
         └───────────────────────┼─────────────────────────┘
                                 ▼
                ┌───────────────────────────────────┐
                │ One session shared by all agents: │
                │ events, state (output_key), and   │
                │ the same artifact/memory services │
                └───────────────────────────────────┘
```

---

## Deployment Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    Cloud Build                           │
│  • Test                                                  │
│  • Build                                                 │
│  • Push to Registry                                      │
└──────────────────────┬─────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    ┌─────────────┐        ┌─────────────┐
    │ Cloud Run   │        │ Vertex AI   │
    │ (Serverless)│        │ Agent Engine│
    └──────┬──────┘        └──────┬──────┘
           │                     │
           ▼                     ▼
    ┌─────────────┐        ┌─────────────┐
    │ US-Central1 │        │ EU-West1    │
    │ (Region 1)  │        │ (Region 2)  │
    └──────┬──────┘        └──────┬──────┘
           │                     │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Load Balancer       │
           │ (Traffic routing)   │
           └──────────┬──────────┘
                      │
                      ▼
              ┌─────────────────┐
              │  End Users      │
              └─────────────────┘
```

---

## Monitoring Stack

```text
┌──────────────────────────────────────────────────────────┐
│                   ADK Application                        │
└──────────────────────────┬───────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌──────────┐       ┌───────────┐     ┌─────────┐
   │  Logging │       │ Metrics   │     │ Tracing │
   │ (Logs)   │       │(Monitoring)     │(Trace)  │
   └────┬─────┘       └─────┬─────┘     └────┬────┘
        │                   │               │
        └───────────────────┼───────────────┘
                            │
                            ▼
         ┌──────────────────────────────────┐
         │ Cloud Logging Dashboard          │
         │ Cloud Monitoring Dashboard       │
         │ Cloud Trace Dashboard            │
         └──────────────┬───────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │ Alerting & Notification          │
         │ • Email                          │
         │ • SMS                            │
         │ • Slack                          │
         │ • PagerDuty                      │
         └──────────────────────────────────┘
```

---

## Security Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    Request Flow                          │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │ API Gateway / Load Balancer      │
        │ - SSL/TLS Termination            │
        │ - DDoS Protection                │
        └─────────────────┬────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │ Authentication                   │
        │ - JWT Validation                 │
        │ - Service Account                │
        │ - OAuth2                         │
        └─────────────────┬────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │ Authorization (RBAC)             │
        │ - Role Check                     │
        │ - Permission Check               │
        │ - Resource ACL                   │
        └─────────────────┬────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │ Input Validation                 │
        │ - Schema Validation              │
        │ - Content Scanning               │
        │ - Rate Limiting                  │
        └─────────────────┬────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │ ADK Agent Processing             │
        │ - Execute with security context  │
        │ - Encrypted communication        │
        │ - Audit logging                  │
        └─────────────────┬────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │ Response Encryption              │
        │ - HTTPS/SSL                      │
        │ - Field-level encryption         │
        └──────────────────────────────────┘
```

---

## Cost Optimisation Flow

```text
┌────────────────────────────────────────────────────────┐
│              Query Analysis                            │
└────────────────────┬─────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ▼                        ▼
    ┌─────────┐              ┌──────────┐
    │Simple?  │              │Complex?  │
    │(routing)│              │(analysis)│
    └────┬────┘              └────┬─────┘
         │                        │
         ▼                        ▼
    ┌───────────┐            ┌──────────┐
    │Small model│            │Larger    │
    │           │            │model     │
    └─────┬─────┘            └────┬─────┘
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
          ┌────────────────────┐
          │ Execute & Log Cost │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │ Update Budget      │
          │ Track Spending     │
          └────────────────────┘
```

---

## Testing Architecture

```text
┌────────────────────────────────────────────────────────┐
│              Test Suite Organization                    │
└────────────────────┬─────────────────────────────────┘
                     │
         ┌───────────┼────────────┬─────────────┐
         │           │            │             │
         ▼           ▼            ▼             ▼
    ┌────────┐  ┌────────┐   ┌────────┐   ┌─────────┐
    │ Unit   │  │Integration│Performance│Smoke     │
    │Tests   │  │Tests      │Tests      │Tests     │
    └───┬────┘  └────┬─────┘ └───┬──────┘ └────┬────┘
        │            │           │             │
        ▼            ▼           ▼             ▼
    ┌─────────────────────────────────────────────────┐
    │           Continuous Integration                │
    │  • Run on each commit                           │
    │  • Generate coverage reports                    │
    │  • Block on failures                            │
    │  • Create test reports                          │
    └──────────────────┬────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────────┐
    │      Deploy to Production (if passing)          │
    └──────────────────────────────────────────────────┘
```

---

## Data Flow: End-to-End Request

```text
1. User Request
   │
   ▼
2. API Gateway
   │ ├─ Validate request
   │ ├─ Check authentication
   │ └─ Apply rate limiting
   │
   ▼
3. Session Service
   │ ├─ Load/create session
   │ └─ Restore state
   │
   ▼
4. Agent
   │ ├─ Receive input
   │ ├─ Process with reasoning
   │ ├─ Call tools as needed
   │ └─ Generate response
   │
   ▼
5. Tool Execution
   │ ├─ Validate inputs
   │ ├─ Execute (possibly calling external services)
   │ └─ Return results
   │
   ▼
6. Response Generation
   │ ├─ Format response
   │ ├─ Validate output schema
   │ └─ Prepare for return
   │
   ▼
7. Session Update
   │ ├─ Update state
   │ ├─ Store artifacts
   │ └─ Log interaction
   │
   ▼
8. Response Return
   │ ├─ Encrypt response
   │ ├─ Add metadata
   │ └─ Send to client
   │
   ▼
9. Monitoring
   │ ├─ Log interaction
   │ ├─ Record metrics
   │ ├─ Track costs
   │ └─ Update dashboards
   │
   ▼
User Receives Response
```

---

*These diagrams provide visual representations of key ADK concepts and architectures. Refer to the comprehensive and production guides for detailed explanations.*


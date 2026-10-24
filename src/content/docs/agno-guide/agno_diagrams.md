---
title: "Agno: diagrams"
description: "Mermaid diagrams of Agno's moving parts: the agent run loop, pausing and resuming for human input, the four team modes, workflow primitives, AgentOS and its clients, and what is stored where."
framework: "agno"
language: "python"
sidebar:
  label: "Diagrams"
  order: 60
---

Checked against **agno %%version=agno%%** (checked %%versions-checked%%). These diagrams simplify behaviour that is verified on the linked pages; the pages are the reference.

## The agent run loop

What happens inside `agent.run()` ([comprehensive guide](/agno-guide/agno_comprehensive_guide/)):

```mermaid
sequenceDiagram
    participant App
    participant Agent
    participant DB as db
    participant Model
    participant Tool

    App->>Agent: run(input, session_id, user_id)
    Agent->>DB: load session (history, session_state)
    Agent->>Agent: resolve dependencies, run pre_hooks / guardrails
    Agent->>Agent: build system message (instructions, memories, state, knowledge refs)
    loop until the model answers without tool calls
        Agent->>Model: messages + tool schemas
        Model-->>Agent: tool calls or text
        Agent->>Tool: execute (tool_hooks wrap each call)
        Tool-->>Agent: result
    end
    Agent->>Agent: parse output_schema, run post_hooks
    Agent->>DB: save run (and memories / summary if enabled)
    Agent-->>App: RunOutput (content, tools, metrics, status)
```

A guardrail that raises `InputCheckError` ends the run before the first model call with `status=error`; an `OutputCheckError` sets `status=error` after it.

## Pausing for a person

Tools with `requires_confirmation`, `requires_user_input` or `external_execution` ([human-in-the-loop](/agno-guide/agno_comprehensive_guide/#human-in-the-loop)):

```mermaid
sequenceDiagram
    participant App
    participant Agent
    participant Model
    participant Person

    App->>Agent: run("Delete /tmp/report.txt")
    Agent->>Model: messages + tools
    Model-->>Agent: call delete_file(path)
    Agent-->>App: RunOutput(status=PAUSED, requirements)
    App->>Person: show tool name and arguments
    Person-->>App: approve / reject / supply input
    App->>App: req.confirm() / req.reject(note) / req.provide_user_input(...)
    App->>Agent: continue_run(run_id, requirements)
    Agent->>Agent: run the tool, or tell the model it was rejected
    Agent->>Model: messages + tool result
    Model-->>Agent: final answer
    Agent-->>App: RunOutput(status=COMPLETED)
```

Over HTTP the same exchange is `POST /agents/{id}/runs` followed by `POST /agents/{id}/runs/{run_id}/continue` ([AgentOS](/agno-guide/agno_agentos/#human-in-the-loop-over-http)).

## Team modes

([Teams](/agno-guide/agno_teams/#modes))

```mermaid
flowchart TB
    subgraph coordinate["coordinate (default)"]
        C0[User] --> CL[Leader]
        CL -- delegate_task_to_member --> CA[Member A]
        CL -- delegate_task_to_member --> CB[Member B]
        CA --> CL
        CB --> CL
        CL --> C1[Leader's answer]
    end
    subgraph route
        R0[User] --> RL[Leader]
        RL -- picks one --> RA[Member A]
        RA --> R1[Member's answer returned as is]
    end
    subgraph broadcast
        B0[User] --> BL[Leader]
        BL -- delegate_task_to_members --> BA[Member A]
        BL -- same task --> BB[Member B]
        BA --> BL
        BB --> BL
        BL --> B1[Combined answer]
    end
    subgraph tasks
        T0[User] --> TL[Leader]
        TL -- create_task --> TQ[(Task list in session_state)]
        TL -- execute_task / execute_tasks_parallel --> TA[Members]
        TA --> TQ
        TQ --> TL
        TL -- mark_all_complete or max_iterations --> T1[Answer]
    end
```

## Workflow primitives

([Workflows](/agno-guide/agno_workflows/))

```mermaid
flowchart LR
    I([input]) --> S1[Step: agent / team / function]
    S1 --> P{{Parallel}}
    P --> P1[Step]
    P --> P2[Step]
    P1 --> J[Step: read get_step_content of the Parallel]
    P2 --> J
    J --> C{Condition}
    C -- true --> CT[steps]
    C -- false --> CF[else_steps]
    CT --> L
    CF --> L
    subgraph L[Loop until end_condition or max_iterations]
        L1[Step] --> L2[Step]
    end
    L --> R{Router}
    R -- selector picks --> RA[choice A]
    R -.-> RB[choice B]
    RA --> O([WorkflowRunOutput])
```

Any step can carry `human_review=HumanReview(...)`, which pauses the run before (or after) it until `continue_run()`.

## AgentOS and its clients

([AgentOS](/agno-guide/agno_agentos/))

```mermaid
flowchart LR
    subgraph clients[Clients]
        UI[os.agno.com control plane<br/>or your own UI]
        HTTP[curl / httpx / AgentOSClient / RemoteAgent]
        MCPC[MCP clients]
        CHAT[Slack, Telegram, WhatsApp]
        PROTO[AG-UI front ends, A2A agents]
    end
    subgraph os[AgentOS: FastAPI app in your infrastructure]
        AUTH[OS_SECURITY_KEY or JWT Authorization<br/>user_isolation]
        REST["REST + SSE<br/>/agents /teams /workflows /sessions /memories /knowledge /traces ..."]
        MCPS["/mcp"]
        IF["interfaces: /slack /telegram /whatsapp /agui /a2a"]
        Q[background queue, scheduler]
        COMP[Agents, Teams, Workflows]
    end
    DB[(Your database:<br/>sessions, runs, memories,<br/>knowledge contents, evals, traces, schedules)]
    VDB[(Vector store)]
    LLM[Model providers]

    UI --> AUTH
    HTTP --> AUTH
    MCPC --> MCPS
    CHAT --> IF
    PROTO --> IF
    AUTH --> REST
    REST --> COMP
    MCPS --> COMP
    IF --> COMP
    Q --> COMP
    COMP --> LLM
    COMP --> DB
    COMP --> VDB
```

The control plane is a browser application that calls your AgentOS directly; your data stays in your database.

## What is stored where

```mermaid
erDiagram
    SESSION ||--o{ RUN : "has (agno_runs, one row per run since 3.0)"
    SESSION {
        string session_id
        string session_type "agent / team / workflow"
        string user_id
        json session_data "includes session_state"
        json summary
    }
    RUN {
        string run_id
        string session_id
        string user_id
        string run_type
        int run_index
        string status
        json run_data "messages, tools, metrics"
    }
    USER_MEMORY {
        string memory_id
        string user_id
        string memory
        json topics
    }
    KNOWLEDGE_CONTENT {
        string id
        string name
        json metadata
        string status
        string user_id
    }
    TRACE ||--o{ SPAN : contains
    EVAL_RUN {
        string run_id
        string eval_type
        json eval_data
    }
    SCHEDULE {
        string name
        string cron_expr
        string timezone
        string endpoint
        json payload
        int next_run_at
    }
```

All of these live in the one `db` you give the agent, team, workflow or AgentOS; table names can be changed on the db constructor. Knowledge vectors live in the vector store, with the contents table tracking what was inserted.

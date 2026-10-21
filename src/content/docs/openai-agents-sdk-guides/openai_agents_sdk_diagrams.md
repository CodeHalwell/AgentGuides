---
title: "OpenAI Agents SDK: Diagrams and Architecture Guide"
description: "Text diagrams of how the Python OpenAI Agents SDK works: the run loop, sessions, handoffs, guardrails, tools, MCP and approvals."
sidebar:
  label: "Diagrams"
  order: 8
framework: "openai-agents-sdk"
language: "python"
---

> Checked against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%).

Text diagrams of what happens inside a run. Code for each step is in the [comprehensive guide](/openai-agents-sdk-guides/openai_agents_sdk_comprehensive_guide/).

## Table of Contents

1. [Agent Lifecycle](#agent-lifecycle)
2. [Multi-Agent Interaction Patterns](#multi-agent-interaction-patterns)
3. [Message Flow Diagrams](#message-flow-diagrams)
4. [Session Management](#session-management)
5. [Tool Integration Patterns](#tool-integration-patterns)
6. [Guardrail Integration](#guardrail-integration)
7. [MCP Integration Architecture](#mcp-integration-architecture)
8. [Approvals](#approvals)
9. [Real-World Patterns](#real-world-patterns)

---

## Agent Lifecycle

### Agent Execution Loop

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Execution Flow                  │
└─────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   User Input         │
    │   + Instructions     │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Format as Input     │
    │  (with history)      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐      ┌───────────────────────────┐
    │  Call LLM Model      │ ◄──► │ Input guardrails (first   │
    │  (with tools schema) │      │ agent only; in parallel   │
    └──────────┬───────────┘      │ unless run_in_parallel=   │
               │                  │ False, which runs first)  │
               │                  └───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────┐
    │  Parse LLM Response                      │
    ├──────────────────────────────────────────┤
    │  ├─ Tool Calls → Process & Return Results│
    │  ├─ Handoff → Switch Agent & Continue   │
    │  └─ Final Output → Return to User        │
    └──────────┬───────────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
    Has More   Reached Final
    Work?      Output?
    │          │
    ├─Yes──────┤
    │          │
    ▼          ▼
  Loop      Output Guardrails
             (validation)
               │
               ▼
          Return Result
```

### Session State Management

```
┌──────────────────────────────────────────────────────────┐
│           Session Lifecycle with Conversation             │
└──────────────────────────────────────────────────────────┘

Turn 1:
  User: "My favourite colour is blue"
     │
     ▼
  Session Storage: [user_msg_1]
     │
     ▼
  Agent Response: "Noted, blue is your favourite"
     │
     ▼
  Session Storage: [user_msg_1, assistant_msg_1]

Turn 2:
  User: "What's my favourite colour?"
     │
     ▼
  Load from Session: [user_msg_1, assistant_msg_1]
     │
     ▼
  Add new User message
  Session Storage: [user_msg_1, assistant_msg_1, user_msg_2]
     │
     ▼
  Agent (with context): "Your favourite colour is blue"
     │
     ▼
  Session Storage: [user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2]
```

---

## Multi-Agent Interaction Patterns

### Simple Handoff Pattern

```
┌────────────────────────────────────────────────────────────┐
│            Simple Agent Handoff Architecture               │
└────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌─────────────────────┐
│  Triage Agent       │
│  - Determines type  │
│  - Routes request   │
└──────┬──────────────┘
       │
       ├─ Billing Issue ──────────┐
       │                          │
       ├─ Technical Issue ────────┤
       │                          │
       └─ General Question ───────┤
                                  │
         ┌────────────────────────┴──────┐
         │                               │
         ▼                               ▼
    ┌──────────────┐            ┌──────────────┐
    │ Billing Agent│            │ Technical    │
    │              │            │ Support      │
    │ - Process    │            │              │
    │ - Calculate  │            │ - Diagnose   │
    │ - Resolve    │            │ - Solutions  │
    └──────────────┘            └──────────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
                    Final Response
```

### Chain of Responsibility Pattern

```
┌────────────────────────────────────────────────────────────┐
│         Chain of Responsibility - Agent Delegation          │
└────────────────────────────────────────────────────────────┘

User Query: "Complex Financial Analysis"
    │
    ▼
┌──────────────────────┐
│ Data Collection      │
│ Agent                │ Fetch data sources
├──────────────────────┤
│ Tools: Web search,   │
│ file retrieval       │
└──────┬───────────────┘
       │ "Here's the data"
       ▼
┌──────────────────────┐
│ Preprocessing Agent  │
│                      │ Clean & structure data
├──────────────────────┤
│ Tools: Data          │
│ transformations      │
└──────┬───────────────┘
       │ "Data prepared"
       ▼
┌──────────────────────┐
│ Analysis Agent       │
│                      │ Run complex analysis
├──────────────────────┤
│ Tools: Calculations, │
│ statistical models   │
└──────┬───────────────┘
       │ "Analysis complete"
       ▼
┌──────────────────────┐
│ Reporting Agent      │
│                      │ Generate formatted report
├──────────────────────┤
│ Tools: Formatting,   │
│ export options       │
└──────┬───────────────┘
       │
       ▼
   Final Report
```

### Hub-and-Spoke Pattern

```
┌────────────────────────────────────────────────────────────┐
│          Hub-and-Spoke Multi-Agent Architecture            │
└────────────────────────────────────────────────────────────┘

                ┌─────────────────────┐
                │   Orchestrator      │
                │   Agent (Hub)       │
                │                     │
                │ - Route requests    │
                │ - Aggregate results │
                │ - Manage workflow   │
                └─────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Agent 1 │      │ Agent 2 │      │ Agent 3 │
    │ Search  │      │ Analysis│      │ Format  │
    └─────────┘      └─────────┘      └─────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
                  Orchestrator collects
                  and integrates results
```

---

## Message Flow Diagrams

### Standard Agent Query Flow

```
┌────────────────────────────────────────────────────────────┐
│            Message Flow: Query to Response                  │
└────────────────────────────────────────────────────────────┘

User                Agent              LLM              Tool
  │                  │                 │                 │
  │─ Query ─────────>│                 │                 │
  │                  │─ Format ───────>│                 │
  │                  │   (with context)│                 │
  │                  │                 │                 │
  │                  │<─ Response ─────│                 │
  │                  │ (with tool call)│                 │
  │                  │                 │                 │
  │                  │─────────────────────── Call ─────>│
  │                  │                                    │
  │                  │<───────────────── Result ─────────│
  │                  │                                    │
  │                  │─ Re-call ────────>│ (with result) │
  │                  │                   │               │
  │                  │<─ Final Output ───│               │
  │                  │                   │               │
  │<─ Response ──────│                   │               │
  │                  │                   │               │
```

### Handoff Message Flow

```
┌────────────────────────────────────────────────────────────┐
│          Message Flow: Agent Handoff Process                │
└────────────────────────────────────────────────────────────┘

User           Agent 1 (Triage)      Agent 2 (Specialist)
  │                │                        │
  │─ Query ───────>│                        │
  │                │                        │
  │                │─ Evaluate ──>│         │
  │                │   (LLM)      │         │
  │                │                        │
  │                │<── Decision ──│        │
  │                │   "Handoff"           │
  │                │                        │
  │                │─ Transfer History ───>│
  │                │                        │
  │                │                  Process Query
  │                │                        │
  │                │<─── Final Response ───│
  │                │                        │
  │<─ Response ────│                        │
  │                │                        │
```

### Guardrail Validation Flow

```
User Input
   │
   ├──────────────────────────────┐
   ▼                              ▼
Input guardrails             First model call
(run_in_parallel=True)       (starts at the same time)
   │
   ├─ tripwire? ──► cancel run, raise InputGuardrailTripwireTriggered
   │
   ▼
Agent loop continues (tools, handoffs, more model calls)
   │
   ▼
Final output from the last agent
   │
   ▼
Output guardrails
   │
   ├─ tripwire? ──► raise OutputGuardrailTripwireTriggered (no automatic retry)
   │
   ▼
Return result
```

With `@input_guardrail(run_in_parallel=False)` the guardrail finishes before the first model call starts, so a tripwire costs no model tokens.

---

## Session Management

### SQLite Session Persistence

```
┌────────────────────────────────────────────────────────────┐
│         Session Storage: SQLite Architecture                │
└────────────────────────────────────────────────────────────┘

                 In-Memory
              Conversation
                Context
                   │
                   ▼
          ┌──────────────────┐
          │  SQLiteSession   │
          │  (Agent SDK)     │
          └────────┬─────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    Write On       Auto-Load On
    Message        Agent Run
        │               │
        ▼               ▼
   ┌───────────────────────────┐
   │   SQLite Database         │
   │   conversations.db        │
   ├───────────────────────────┤
   │ agent_sessions            │
   │ agent_messages            │
   │ (session_id, message_data)│
   └───────────────────────────┘
        ▲                   │
        │                   └──────────> Persistent Storage
        │                                (File System)
        └─── Query & Retrieve ──────────
             For Context
```

### Multi-Backend Session Support

```
┌────────────────────────────────────────────────────────────┐
│       Session Storage: Multiple Backend Options             │
└────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  Agent/Runner       │
│                     │
│  session=...        │
└──────────┬──────────┘
           │
    ┌──────┼──────┬──────────┬────────────┐
    │      │      │          │            │
    ▼      ▼      ▼          ▼            ▼
  SQLite Redis SQLAlchemy  OpenAI    SQLite ":memory:"
  │       │     │          │        │
  │       │     │          │        │
  ├──FS   ├─CPU │          │        └─ RAM
  │       │     │          │        (testing)
  │       │     │          │
  │       │     ├─ PostgreSQL
  │       │     ├─ MySQL
  │       │     └─ Others
  │       │
  │       └─ Distributed
  │          Cache Layer
  │
  └─ Local File
     Storage
```

---

## Tool Integration Patterns

### Function Tool Execution Flow

```
┌────────────────────────────────────────────────────────────┐
│        Function Tool: Definition to Execution               │
└────────────────────────────────────────────────────────────┘

Step 1: Tool Definition
┌──────────────────────────┐
│ @function_tool           │
│ def get_weather():       │
│   ...                    │
└──────────┬───────────────┘
           │
           ▼
Step 2: Schema Generation
┌──────────────────────────┐
│ Pydantic Inspection      │
├──────────────────────────┤
│ - Extract parameters     │
│ - Get type hints         │
│ - Generate JSON schema   │
└──────────┬───────────────┘
           │
           ▼
Step 3: Register with Agent
┌──────────────────────────┐
│ Agent(tools=[...])       │
├──────────────────────────┤
│ - Add to tools list      │
│ - Include in LLM prompt  │
└──────────┬───────────────┘
           │
           ▼
Step 4: LLM Decision
┌──────────────────────────┐
│ LLM determines need      │
│ tool and calls it        │
└──────────┬───────────────┘
           │
           ▼
Step 5: Tool Execution
┌──────────────────────────┐
│ Runner calls tool        │
│ with LLM arguments       │
└──────────┬───────────────┘
           │
           ▼
Step 6: Return Result
┌──────────────────────────┐
│ Tool output appended     │
│ to message history       │
└──────────┬───────────────┘
           │
           ▼
Step 7: Continue Loop
┌──────────────────────────┐
│ LLM uses result to       │
│ craft response           │
└──────────────────────────┘
```

### Hosted Tools Architecture

```
┌────────────────────────────────────────────────────────────┐
│        Hosted Tools: OpenAI and External Services           │
└────────────────────────────────────────────────────────────┘

Agent with
Hosted Tools
    │
    ├─ WebSearchTool()
    │  └──> Real-time web search
    │       Results returned
    │
    ├─ FileSearchTool()
    │  └──> Vector database search
    │       Semantic matching
    │
    ├─ CodeInterpreterTool(tool_config=...)
    │  └──> Python in an OpenAI-hosted container
    │       Results captured
    │
    ├─ ImageGenerationTool(tool_config=...)
    │  └──> Image generation; image returned as base64
    │
    └─ HostedMCPTool(tool_config=...)
       └──> OpenAI calls a remote MCP server

Run in your process instead: ComputerTool(computer=...) drives a
screen you implement; ShellTool / ApplyPatchTool act on a machine you control.
```

---

## Guardrail Integration

### Where Each Guardrail Runs

```
Runner.run(starting_agent, input)
   │
   ├─ input guardrails ............ starting agent only (+ RunConfig.input_guardrails)
   │
   ├─ for each function tool call:
   │     tool input guardrails .... before the tool (allow / reject_content / raise)
   │     needs_approval ........... pause the run for a person
   │     tool executes
   │     tool output guardrails ... before the model sees the result
   │
   ├─ handoff input_filter ........ when control moves to another agent
   │
   └─ output guardrails ........... final agent only (+ RunConfig.output_guardrails)
```

Several guardrails at the same level run concurrently; the first tripwire stops the run.

---

## MCP Integration Architecture

### MCP Server Integration Pattern

```
┌────────────────────────────────────────────────────────────┐
│    Model Context Protocol: Agent-Server Integration         │
└────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   OpenAI Agent                            │
│                                                           │
│  ├─ Agent                                               │
│  ├─ Tools (function tools + MCP tools)                  │
│  └─ Runner                                              │
└─────────────────┬───────────────────────────────────────┘
                  │ Uses resources from
                  │
        ┌─────────┴──────────┬──────────────┐
        │                    │              │
        ▼                    ▼              ▼
    MCP Server A        MCP Server B    MCP Server C
    (Filesystem)       (Git)           (Web Tools)
        │                  │               │
        ├─ List files      ├─ Git log      ├─ Search
        ├─ Read file       ├─ Git diff     ├─ Fetch URL
        └─ Write file      └─ Git blame    └─ Parse HTML
```

### Stdio vs HTTP MCP

```
┌────────────────────────────────────────────────────────────┐
│    MCP Server Types: Local vs Remote                        │
└────────────────────────────────────────────────────────────┘

Local MCP (Stdio):
  Agent
    │
    ├─ MCPServerStdio
    │  │
    │  └─ subprocess.spawn
    │     (local process)
    │
    └─ Tools from local
       process

Remote MCP (HTTP/SSE):
  Agent
    │
    ├─ MCPServerStreamableHttp
    │  │
    │  └─ HTTP/SSE connection
    │     (remote server)
    │
    └─ Tools from remote
       endpoint
```

---

## Approvals

### Pausing for a Human

```
Runner.run(agent, input)
   │
   ▼
Model calls a tool marked needs_approval
   │
   ▼
Run stops; result.interruptions = [ToolApprovalItem, ...]
   │
   ▼
state = result.to_state()  ──► state.to_string() ──► your database
   │                                                    │
   │            (minutes or days later, any process)    │
   ▼                                                    ▼
state = await RunState.from_string(agent, saved)
state.approve(item)  or  state.reject(item, rejection_message=...)
   │
   ▼
Runner.run(agent, state)  ──► tool runs (or the model is told it was rejected)
   │
   ▼
Loop continues to the final output
```

---

## Real-World Patterns

### Financial Research Workflow

```
┌────────────────────────────────────────────────────────────┐
│     Use Case: Financial Analysis Multi-Agent Workflow       │
└────────────────────────────────────────────────────────────┘

User: "Analyse stock XYZ"
    │
    ▼
Collector Agent
  ├─ Web search for news
  ├─ Fetch financial data
  └─ Retrieve analyst reports
    │
    ▼ Data collected
    │
Processor Agent
  ├─ Parse financial metrics
  ├─ Calculate ratios
  └─ Identify trends
    │
    ▼ Processed data
    │
Analyser Agent
  ├─ SWOT analysis
  ├─ Risk assessment
  └─ Valuation models
    │
    ▼ Analysis complete
    │
Reporter Agent
  ├─ Format findings
  ├─ Generate visuals
  └─ Produce recommendation
    │
    ▼
Final Report
```

### Customer Support Triage

```
┌────────────────────────────────────────────────────────────┐
│     Use Case: Customer Service Support Triage               │
└────────────────────────────────────────────────────────────┘

Customer Query
    │
    ▼
Initial Assessment Agent
  ├─ Classify issue
  ├─ Set priority
  └─ Route to specialist
    │
    ├─ Billing ────────> Billing Agent
    │  - Invoices          - Account access
    │  - Refunds           - Payment processing
    │                      - Escalation
    │
    ├─ Technical ─────> Technical Agent
    │  - Bugs              - Troubleshooting
    │  - Features          - Documentation
    │  - Integration       - Solutions
    │
    └─ General ───────> Knowledge Agent
       - FAQs             - Documentation
       - Getting started  - Best practices

Agent Handles Request
    │
    └─> Resolution or Escalation
        to Human Agent
```

This comprehensive diagrams guide provides visual clarity for understanding OpenAI Agents SDK architectures and patterns.


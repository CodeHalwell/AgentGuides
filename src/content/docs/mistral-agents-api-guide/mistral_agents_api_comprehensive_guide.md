---
title: "Mistral Agents API: Comprehensive Technical Guide"
description: "> BREAKING: Mistral SDK v2.0.1 (March 12, 2026) is NOT backwards-compatible with v1.x. See the migration guide for full details."
framework: mistral-agents-api
---

Latest: 2.4.9 | Updated: June 4, 2026
# Mistral Agents API: Comprehensive Technical Guide

> **BREAKING (v2.0.1, March 2026)**: The v2 SDK is NOT backwards-compatible with v1.x. See the migration guide for full details. Current stable: **v2.4.9**.

This document provides comprehensive visual representations of Mistral Agents API architecture, data flows, and patterns.

## 1. Agent Lifecycle Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LIFECYCLE                          │
└─────────────────────────────────────────────────────────────────┘

CREATION PHASE
──────────────
    │
    ▼
┌──────────────────────────┐
│   POST /v1/agents        │
├──────────────────────────┤
│ - Model selection        │
│ - Instructions/Prompt    │
│ - Tools configuration    │
│ - Completion parameters  │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│   Agent Created          │
│   (v1, Immutable)        │
├──────────────────────────┤
│ agent_id: ag_xxx         │
│ status: active           │
│ version: 1               │
└──────────────────────────┘

USAGE PHASE
──────────
    │
    ├─► GET /v1/agents/{id}          → Retrieve agent metadata
    │
    ├─► PATCH /v1/agents/{id}        → Update configuration (v2, v3...)
    │
    └─► POST /v1/conversations       → Start conversations
            │
            ├─► POST /v1/conversations/{id}          → Continue
            │
            ├─► POST /v1/conversations/{id}/restart  → Branch
            │
            └─► GET /v1/conversations/{id}/history   → Retrieve history

DELETION/ARCHIVAL
─────────────────
    │
    ▼
┌──────────────────────────┐
│   Agent Archived         │
│   (Historical Reference) │
├──────────────────────────┤
│ Old conversations still  │
│ accessible and           │
│ replayable               │
└──────────────────────────┘
```

## 2. Conversation Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATION FLOW                            │
└─────────────────────────────────────────────────────────────────┘

CLIENT                                    MISTRAL SERVER
──────                                    ──────────────

User Query
    │
    ▼
┌─────────────┐
│POST /convers│
│ations      │
├─────────────┤
│ agent_id    │
│ inputs      │
│ stream: true│
└─────────────┘
    │
    │────────────────────────┐
    │                        ▼
    │              ┌──────────────────────┐
    │              │  Load Agent Config   │
    │              │  + instructions      │
    │              │  + tools            │
    │              │  + parameters       │
    │              └──────────────────────┘
    │                        │
    │                        ▼
    │              ┌──────────────────────┐
    │              │  Load Conversation  │
    │              │  History (if exists)│
    │              └──────────────────────┘
    │                        │
    │                        ▼
    │              ┌──────────────────────┐
    │              │  LLM Forward Pass    │
    │              │  + system prompt     │
    │              │  + history           │
    │              │  + new input         │
    │              └──────────────────────┘
    │                        │
    │              ┌─────────┴──────────┐
    │              ▼                    ▼
    │        ┌──────────┐        ┌──────────────┐
    │        │ Response │        │ Tool Call(s) │
    │        │ Generation        │ (optional)   │
    │        └──────────┘        └──────────────┘
    │              │                    │
    │              │              ┌─────┴──────────┐
    │              │              ▼                ▼
    │              │         ┌────────┐    ┌──────────────┐
    │              │         │Execute │    │ Execute      │
    │              │         │Tool 1  │    │ Tool N       │
    │              │         └────────┘    └──────────────┘
    │              │              │                │
    │              │              └────────┬───────┘
    │              │                       ▼
    │              │          ┌─────────────────────────┐
    │              │          │ Tool Results            │
    │              │          │ Appended to History     │
    │              │          └─────────────────────────┘
    │              │                       │
    │              │          ┌────────────┴──────────────┐
    │              │          │                           │
    │              │          ▼                           ▼
    │              │    ┌─────────────┐          ┌──────────────┐
    │              │    │ More Tools? │          │ Final Output │
    │              │    │ (loop back) │          │ Generation   │
    │              │    └─────────────┘          └──────────────┘
    │              │           │                        │
    │              └───────────┼────────────────────────┘
    │                          ▼
    │              ┌──────────────────────┐
    │              │ Store Entry in DB:   │
    │              │ - Role: assistant    │
    │              │ - Content            │
    │              │ - Tool executions    │
    │              │ - Metadata           │
    │              └──────────────────────┘
    │
    │◄─ SSE: event: conversation.response.started
    │◄─ SSE: event: message.output.delta
    │◄─ SSE: event: message.output.delta
    │   ... (streaming chunks)
    │◄─ SSE: event: conversation.response.done
    ▼
Display Response
```

## 3. Multi-Agent Orchestration Pattern

```
┌────────────────────────────────────────────────────────────────┐
│         MULTI-AGENT ORCHESTRATION WITHOUT FRAMEWORKS            │
└────────────────────────────────────────────────────────────────┘

SEQUENTIAL PIPELINE
───────────────────
    User Query
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              AGENT 1: Data Cleaner                      │
    │  Raw Data  → Clean & Normalize → Cleaned Data         │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              AGENT 2: Analyzer                          │
    │  Cleaned Data → Extract Insights → Analysis Results    │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              AGENT 3: Report Writer                     │
    │  Analysis → Format Report → Final Report               │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    User Gets Final Report


PARALLEL PROCESSING
───────────────────
                     User Query
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
        ┌─────┐        ┌─────┐        ┌─────┐
        │Agent│        │Agent│        │Agent│
        │ 1   │        │ 2   │        │ 3   │
        └─────┘        └─────┘        └─────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                  ┌────────────────┐
                  │  Synthesize    │
                  │  Results       │
                  └────────────────┘
                          │
                          ▼
                    Final Output


HIERARCHICAL STRUCTURE
──────────────────────
                      Manager Agent
                      (Coordinator)
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         Specialist 1   Specialist 2   Specialist 3
         (Research)     (Analysis)     (Reporting)
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                     Synthesized Output
```

## 4. Tool Execution Workflow

```
┌────────────────────────────────────────────────────────────┐
│              TOOL EXECUTION WORKFLOW                        │
└────────────────────────────────────────────────────────────┘

LLM Generation
      │
      ▼
┌──────────────────────────┐
│ Decision: Tool needed?   │
└──────────────────────────┘
      │
      ├─ No ──→ Generate Response
      │         │
      │         ▼
      │    Return to User
      │
      └─ Yes ──→ Tool Call Generation
                 │
                 ▼
            ┌──────────────────────┐
            │ Extract Tool Name &  │
            │ Parameters           │
            └──────────────────────┘
                 │
                 ▼
            ┌──────────────────────┐
            │ Validate Parameters  │
            │ Against Schema       │
            └──────────────────────┘
                 │
                 ├─ Invalid ──→ Error Response
                 │              │
                 │              ▼
                 │         Inform Agent
                 │
                 └─ Valid ──→ Execute Tool
                            │
                    ┌───────┴────────┐
                    ▼                ▼
                ┌────────┐      ┌───────────┐
                │Success │      │Exception  │
                └────────┘      └───────────┘
                    │                │
                    ▼                ▼
            ┌──────────────┐    ┌──────────────┐
            │Tool Result   │    │Error Message │
            └──────────────┘    └──────────────┘
                    │                │
                    └────────┬────────┘
                             ▼
                    ┌──────────────────────┐
                    │ Append to Conversation
                    │ History              │
                    └──────────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Continue LLM Forward │
                    │ Pass with Result    │
                    └──────────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ More tools needed?   │
                    └──────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                   Yes               No
                    │                 │
          (Loop back to         (Generate Final
           Tool Call)            Response)
```

## 5. Request/Response Processing Pipeline

```
┌────────────────────────────────────────────────────────────┐
│           REQUEST/RESPONSE PROCESSING                      │
└────────────────────────────────────────────────────────────┘

CLIENT SIDE
───────────
  ┌────────────────────┐
  │ Application Logic  │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Format Request:    │
  │ - agent_id         │
  │ - inputs           │
  │ - stream: bool     │
  │ - store: bool      │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ HTTP Request       │
  │ POST /conversations│
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Streaming Handler  │
  │ (if stream=true)   │
  └────────────────────┘

API GATEWAY
───────────
           │
           ▼
  ┌────────────────────┐
  │ Authentication     │
  │ (API Key Verify)   │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Rate Limiting      │
  │ (Quota Check)      │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Route to Worker    │
  └────────────────────┘

WORKER PROCESSING
─────────────────
           │
           ▼
  ┌────────────────────┐
  │ Load Agent Config  │
  │ (from DB)          │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Load Conversation  │
  │ History (if exists)│
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Execute LLM        │
  │ Pipeline           │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Process Tool Calls │
  │ (if any)           │
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ Store in Database  │
  │ - Entry data       │
  │ - Metadata         │
  │ - Timing info      │
  └────────────────────┘

RESPONSE DELIVERY
─────────────────
           │
           ▼
  ┌────────────────────┐
  │ Stream Events?     │
  └────────────────────┘
           │
    ┌──────┴──────┐
    │             │
   Yes           No
    │             │
    ▼             ▼
┌────────┐   ┌─────────┐
│SSE     │   │Standard │
│Chunked │   │Response │
└────────┘   └─────────┘
    │             │
    └──────┬──────┘
           ▼
  ┌────────────────────┐
  │ Return to Client   │
  │ - conversation_id  │
  │ - outputs[]        │
  │ - usage tokens     │
  └────────────────────┘
           │
           ▼
CLIENT APPLICATION
```

## 6. Memory Persistence Architecture

```
┌────────────────────────────────────────────────────────────┐
│            MEMORY PERSISTENCE ARCHITECTURE                 │
└────────────────────────────────────────────────────────────┘

CONVERSATION LIFECYCLE & STORAGE
─────────────────────────────────

Turn 1: User Input
    │
    ▼
┌──────────────────────────┐
│ VOLATILE (In Processing) │
├──────────────────────────┤
│ - Received: "Hello"      │
│ - In LLM pipeline        │
└──────────────────────────┘
    │
    ▼
LLM Processing
    │
    ▼
┌──────────────────────────┐
│ Generated Response       │
│ + Tool Executions        │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│         PERSISTENT DATABASE                   │
├──────────────────────────────────────────────┤
│ Conversation Entry:                          │
│ ├─ ID: msg_xxx                              │
│ ├─ Type: message.input                      │
│ ├─ Role: user                               │
│ ├─ Content: "Hello"                         │
│ ├─ Created_at: timestamp                    │
│ └─ Conversation_id: conv_yyy                │
│                                              │
│ Conversation Entry:                          │
│ ├─ ID: msg_yyy                              │
│ ├─ Type: message.output                     │
│ ├─ Role: assistant                          │
│ ├─ Content: "Hi! How can I help?"           │
│ ├─ Created_at: timestamp                    │
│ ├─ Completed_at: timestamp                  │
│ └─ Agent_id: ag_zzz                         │
└──────────────────────────────────────────────┘

Turn 2: Continuation (Hours/Days Later)
    │
    ▼
┌──────────────────────────┐
│ New user input: "..."    │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ LOAD FULL HISTORY                            │
├──────────────────────────────────────────────┤
│ GET /conversations/{conv_id}/history        │
│   Returns: All entries (chronological)       │
│   ├─ User input 1                           │
│   ├─ Assistant response 1                   │
│   ├─ Tool execution 1                       │
│   ├─ User input 2                           │
│   └─ ... (all history)                      │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Context Reconstructed    │
│ Full conversation loaded │
└──────────────────────────┘
    │
    ▼
LLM Processes
New Input + Full History
    │
    ▼
┌──────────────────────────────────────────────┐
│ NEW ENTRIES STORED                           │
├──────────────────────────────────────────────┤
│ ├─ User input (entry)                       │
│ ├─ Assistant output (entry)                 │
│ └─ Tool executions (entries)                │
└──────────────────────────────────────────────┘

CONVERSATION BRANCHES
──────────────────────
                  Original Conv
                       │
           ┌───────────┼───────────┐
           │           │           │
        Turn 1       Turn 2      Turn 3
           │           │           │
           ▼           ▼           ▼
      [Entry]     [Entry]     [Entry]
                       │
              Restart from Turn 1
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
      [Entry]                  [New Branch]
      (Resume)                 (Alternative
                                Path)
```

## 7. Conversation State Management

```
┌────────────────────────────────────────────────────────────┐
│           CONVERSATION STATE MACHINE                        │
└────────────────────────────────────────────────────────────┘

States:
─────
[CREATED]
    │
    ├─→ new conversation initialized
    ├─→ agent linked
    ├─→ ready for first turn
    │
    ▼
[ACTIVE]
    │
    ├─→ processing messages
    ├─→ generating responses
    ├─→ executing tools
    │
    ▼
[AWAITING_INPUT]
    │
    ├─→ response completed
    ├─→ waiting for user input
    ├─→ context maintained
    │
    ▼
[PROCESSING]
    │
    ├─→ executing LLM forward pass
    ├─→ may call tools
    ├─→ generating response
    │
    ▼
[COMPLETED_TURN]
    │
    ├─→ response generated
    ├─→ entries stored
    ├─→ ready for next turn or restart
    │
    ▼
[ARCHIVED]
    │
    ├─→ no new messages accepted
    ├─→ history retrievable
    ├─→ can restart from any entry

Transitions:
───────────
CREATED → ACTIVE (on first message)
ACTIVE ↔ AWAITING_INPUT (normal conversation flow)
Any → PROCESSING (when processing message)
PROCESSING → AWAITING_INPUT/COMPLETED_TURN
Any → ARCHIVED (explicit archival)
ARCHIVED → ACTIVE (via restart mechanism)
```

## 8. API Integration Points

```
┌────────────────────────────────────────────────────────────┐
│              MISTRAL AGENTS API ENDPOINTS                  │
└────────────────────────────────────────────────────────────┘

AGENT MANAGEMENT
────────────────
┌──────────────────────────────────────────────┐
│ POST   /v1/agents                            │
│ GET    /v1/agents                            │
│ GET    /v1/agents/{agent_id}                 │
│ PATCH  /v1/agents/{agent_id}                 │
│ DELETE /v1/agents/{agent_id}                 │
└──────────────────────────────────────────────┘

CONVERSATION MANAGEMENT
───────────────────────
┌──────────────────────────────────────────────┐
│ POST   /v1/conversations (start new)          │
│ POST   /v1/conversations/{conv_id}           │
│        (continue/append)                     │
│ POST   /v1/conversations/{conv_id}/restart   │
│        (restart from entry)                  │
│ GET    /v1/conversations (list all)          │
│ GET    /v1/conversations/{conv_id}           │
│        (get metadata)                        │
│ GET    /v1/conversations/{conv_id}/history   │
│        (full entries)                        │
│ GET    /v1/conversations/{conv_id}/messages  │
│        (only messages, filtered)             │
└──────────────────────────────────────────────┘

STREAMING ENDPOINTS
───────────────────
┌──────────────────────────────────────────────┐
│ POST   /v1/conversations?stream=true         │
│ POST   /v1/conversations/{conv_id}?stream=true
│ POST   /v1/conversations/{conv_id}/restart?  │
│        stream=true                           │
│                                              │
│ Returns: Server-Sent Events (SSE) stream     │
│ Events:                                      │
│  - conversation.response.started             │
│  - message.output.delta                      │
│  - tool.execution.started                    │
│  - tool.execution.completed                  │
│  - conversation.response.done                │
└──────────────────────────────────────────────┘
```

## 9. Agent Configuration Schema

```
┌────────────────────────────────────────────────────────────┐
│          AGENT CONFIGURATION STRUCTURE                     │
└────────────────────────────────────────────────────────────┘

Agent Object:
{
  "id": "ag_xxxxxxxxxxxx",           # Unique identifier
  "name": "string",                   # Human-readable name
  "description": "string",            # Purpose description
  "model": "mistral-medium-2505",     # Model to use
  "instructions": "string",           # System prompt
  "version": 1,                       # Version number
  "created_at": "2025-06-16T...",     # Creation timestamp
  "updated_at": "2025-06-16T...",     # Last update
  "tools": [                          # Available tools
    {
      "type": "web_search",
      "function": {...}
    },
    ...
  ],
  "completion_args": {                # Model parameters
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": 2048,
    "presence_penalty": 0,
    "frequency_penalty": 0
  },
  "handoffs": [...]                   # Agent handoff config
}
```

---

## 10. New in v2.4.7: Guardrails and Handoff Execution

*Source: `mistralai==2.4.7` installed at `.routine-envs/mistralai`. Symbols inspected: `GuardrailConfig`, `ModerationLlmv1Config`, `ModerationLlmv2Config` (from `mistralai.client.models`), `Conversations.start()`, `BetaAgents.create()`, `Agents.complete()` (from `mistralai.client`).*

### 10.1 Guardrails (`GuardrailConfig`)

Attach content moderation to individual agents or conversations. The `guardrails` parameter accepts a list of `GuardrailConfig` objects, each of which configures a Mistral moderation model.

```python
import os
from mistralai.client import Mistral
from mistralai.client.models import (
    GuardrailConfig,
    ModerationLlmv2Config,
)

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Attach guardrails when creating an agent
agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="SafeAssistant",
    instructions="You are a helpful assistant.",
    guardrails=[
        GuardrailConfig(
            block_on_error=True,           # return HTTP 403 on moderation error
            moderation_llm_v2=ModerationLlmv2Config(
                # uses mistral-moderation-2603 by default — omit model_name unless overriding
                ignore_other_categories=False,
            ),
        )
    ],
)
```

`GuardrailConfig` fields (source: `mistralai/client/models/guardrailconfig.py`, v2.4.7):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `block_on_error` | `bool` | `False` | Return HTTP 403 and block the request on a moderation server error |
| `moderation_llm_v1` | `ModerationLlmv1Config \| None` | `None` | Use Mistral Moderation v1 (`mistral-moderation-2411`) |
| `moderation_llm_v2` | `ModerationLlmv2Config \| None` | `None` | Use Mistral Moderation v2 (`mistral-moderation-2603`) |

`ModerationLlmv2Config` fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"mistral-moderation-2603"` | Override model name. Omit in general use. |
| `custom_category_thresholds` | `ModerationLlmv2CategoryThresholds \| None` | `None` | Per-category threshold overrides |
| `ignore_other_categories` | `bool` | `False` | If `True`, only evaluate categories in `custom_category_thresholds` |
| `action` | `ModerationLLMAction \| None` | `None` | Action on detection |

You can also attach `guardrails` on a per-conversation basis via `conversations.start()`:

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Tell me how to do something unsafe.",
    guardrails=[
        GuardrailConfig(
            block_on_error=True,
            moderation_llm_v2=ModerationLlmv2Config(),
        )
    ],
)
```

**Pitfall:** `guardrails` and `moderation_llm_v1` / `moderation_llm_v2` are mutually exclusive with one another within a single `GuardrailConfig`. Providing both `moderation_llm_v1` and `moderation_llm_v2` in the same config object is technically permitted but the platform applies them sequentially — use one per config entry for clarity.

---

### 10.2 Handoff Execution Mode (`handoff_execution`)

All `conversations` methods (`start`, `append`, `restart`, and their streaming variants) now accept a `handoff_execution` parameter that controls where agent-to-agent handoffs are executed.

```python
# Source: mistralai/client/models/conversationrequest.py v2.4.7
# ConversationRequestHandoffExecution = Literal["client", "server"]

# Server-side handoffs: Mistral platform routes automatically
conversation = client.beta.conversations.start(
    agent_id=router_agent.id,
    inputs="I need help with a billing issue.",
    handoff_execution="server",   # platform resolves handoffs without a client round-trip
)

# Client-side handoffs: inspect response and route yourself
conversation = client.beta.conversations.start(
    agent_id=router_agent.id,
    inputs="I need help with a billing issue.",
    handoff_execution="client",
)
# Check conversation.outputs for handoff events and dispatch manually
```

| Value | Behaviour |
|-------|-----------|
| `"server"` | Mistral platform executes agent-to-agent handoffs internally. The conversation response includes the final reply from the target agent. |
| `"client"` | Platform returns a deferred handoff event in the conversation outputs. Your code inspects `AgentHandoffStartedEvent` / `AgentHandoffDoneEvent` and dispatches the next call. |

`AgentHandoffStartedEvent` and `AgentHandoffDoneEvent` are in `mistralai.client.models` as of v2.4.7. Use `client` mode when you need to log, audit, or transform state between hops.

---

### 10.3 Prompt Cache Key (`prompt_cache_key`)

The GA `agents.complete()` endpoint (not `beta.agents`) now accepts `prompt_cache_key: str | None` to tag a completion for server-side prompt caching. Reuse the same key across calls with identical system prompts to reduce latency on repeated tasks.

```python
# GA Agents API — note: client.agents.complete(), not client.beta.agents
response = client.agents.complete(
    agent_id=agent.id,
    messages=[{"role": "user", "content": "Summarise this document..."}],
    prompt_cache_key="doc-summary-v1",   # reused on subsequent calls with the same prompt
)
```

**When to use:** Long, stable system prompts or few-shot examples that are sent repeatedly. The `prompt_cache_key` must be identical across calls for the cache to hit. It has no effect when the prompt content differs.

---

**End of Diagrams Documentation**

All diagrams use ASCII art for clarity and can be copied/shared easily. For more detailed visual representation, refer to the comprehensive guide's code examples and the production guide's architecture sections.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 2.4.9 | June 4, 2026 | Patch release. Version bumped 2.4.8 → 2.4.9; `Latest:` header and title updated; revision history entry added. `mistralai==2.4.9` confirmed latest stable on PyPI (2026-06-04); `from mistralai.client import Mistral` is the correct import path. | Claude routine |
| 2.4.8 | May 29, 2026 | Patch release. Version bumped 2.4.7 → 2.4.8; `Latest:` header and title updated; revision history entry added. `from mistralai.client import Mistral` verified against installed `mistralai==2.4.8` (`.routine-envs/check-0529-mistral`); `from mistralai import Mistral` still raises `ImportError`. | Claude routine |
| 2.4.7 | May 27, 2026 | Minor feature release. Version bumped 2.4.5 → 2.4.7. New `guardrails` parameter on `beta.agents.create()` and `conversations.start()` documented (see §10 below). New `handoff_execution` parameter on all Conversations API methods documented. New `prompt_cache_key` on `agents.complete()` documented. Verified against installed `mistralai 2.4.7` (`.routine-envs/mistralai`); `from mistralai.client import Mistral` confirmed; `from mistralai import Mistral` still raises `ImportError`. | Claude routine |
| 2.4.5 | May 9, 2026 | Patch release. Version confirmed against installed `mistralai 2.4.5` (`.routine-envs/check-0509-py`); `from mistralai.client import Mistral` import verified. Note: `from mistralai import Mistral` fails (top-level is a namespace package); correct import path remains `from mistralai.client import Mistral`. |
| 2.4.4 | May 1, 2026 | Patch release. Version confirmed against installed `mistralai 2.4.4` (`.routine-envs/check-mistral2-0501`); `from mistralai.client import Mistral` import verified with `-W error::DeprecationWarning`. |
| 2.4.3 | April 28, 2026 | Patch release. Version confirmed against installed `mistralai 2.4.3` (`.routine-envs/main-py-0428`); `from mistralai.client import Mistral` import verified. |
| 2.4.2 | April 23, 2026 | Patch release. Version bump confirmed against PyPI `mistralai 2.4.2`. |
| 2.4.1 | April 22, 2026 | Header updated from stale 2.0.1 to current 2.4.1; confirmed correct `from mistralai.client import Mistral` import path. |
| 2.4.0 | April 2026 | Azure AI and Google Cloud deployment targets; Python 3.10+ minimum. |
| 2.0.1 | March 12, 2026 | **BREAKING v2 rewrite**: stateful conversation API redesigned; TypeScript SDK now ESM-only, requires Zod v4; full Agents API with MCP tools, Code Interpreter, Premium Web Search; v1.x incompatible |
| 1.9.11 | November 2025 | Previous documented version |


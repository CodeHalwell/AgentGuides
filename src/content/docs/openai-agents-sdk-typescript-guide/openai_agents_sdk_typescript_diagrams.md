---
title: "OpenAI Agents SDK TypeScript: Architecture Diagrams & Visual Guides"
description: "Text diagrams of how the TypeScript OpenAI Agents SDK works: the run loop, handoffs, sessions, tools, retries, structured output and approvals."
sidebar:
  label: "Diagrams"
  order: 8
framework: "openai-agents-sdk"
language: "typescript"
---

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%).

Text diagrams of what happens inside a run. Code for each step is in the [comprehensive guide](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_comprehensive_guide/).

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenAI Agents SDK                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Agent      │      │   Runner     │      │   Handoff    │  │
│  ├──────────────┤      ├──────────────┤      ├──────────────┤  │
│  │ • Name       │      │ • Execute    │      │ • Trigger    │  │
│  │ • Instructions│      │ • Manage     │      │ • Agent      │  │
│  │ • Tools      │      │ • State      │      │ • Input      │  │
│  │ • Guardrails │      │ • Output     │      │ • Callback   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                     │                     │           │
│         │                     ▼                     │           │
│         │              ┌──────────────┐             │           │
│         └─────────────►│   Session    │◄────────────┘           │
│                        ├──────────────┤                         │
│                        │ • History    │                         │
│                        │ • State      │                         │
│                        │ • Memory     │                         │
│                        │ • Metadata   │                         │
│                        └──────────────┘                         │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Tool       │    │  Guardrail   │    │   Output     │   │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤   │
│  │ • Function   │    │ • Validation │    │ • Schema     │   │
│  │ • Parameters │    │ • Constraints│    │ • Type-safe  │   │
│  │ • Execute    │    │ • Enforcement│    │ • Structured │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │   Model Layer    │
                   ├──────────────────┤
                   │ • OpenAI models  │
                   │ • aisdk() models │
                   │ • Custom Model   │
                   └──────────────────┘
```

---

## Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Execution Flow                        │
└─────────────────────────────────────────────────────────────────┘

  User Input
      │
      ▼
  ┌─────────────────┐
  │  Input Guard    │ ◄──── First agent only. Runs alongside the
  │  (Guardrail)    │       first model call unless runInParallel: false
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Build Context      │ ◄──── Include instructions
  │  + History          │       Include tools
  │  + Instructions     │       Session memory
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Call LLM           │ ◄──── Send prompt to model
  │  (Model Layer)      │
  └────────┬────────────┘
           │
           ▼
      ┌─────────┬──────────┐
      │          │          │
   Text      Tool Call  Handoff
      │          │          │
      ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────┐
  │ Output │ │Execute │ │Delegate│
  │Guardrail│ │  Tool  │ │ Agent  │
  └───┬────┘ └───┬────┘ └───┬────┘
      │          │          │
      └──────────┼──────────┘
                 │
                 ▼
         ┌──────────────┐
         │ Update State │ ◄──── Record in session
         │  + Messages  │       Track tokens
         │  + Results   │
         └──────┬───────┘
                │
                ▼
           Final Output
```

---

## Multi-Agent Handoff Pattern

```
┌──────────────────────────────────────────────────────────────┐
│              Multi-Agent Handoff System                      │
└──────────────────────────────────────────────────────────────┘

                      User Query
                          │
                          ▼
                  ┌─────────────────┐
                  │ Triage Agent    │
                  │ (Main Router)   │
                  └────────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │Billing  │       │Technical│       │General  │
   │Support  │       │Support  │       │Support  │
   │Agent    │       │Agent    │       │Agent    │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                 │                 │
        ▼                 ▼                 ▼
   Specialised        Specialised        Specialised
   Response          Response           Response

Routing is the model's decision: it sees one transfer_to_<agent> tool per
specialist, with that agent's handoffDescription, and calls the best match.
```

---

## Session & Memory Management

```
run(agent, input, { session })
   │
   ▼
session.getItems()  ──► prior history + new input ──► agent loop
                                                        │
                                                        ▼
                                            new items (messages, tool calls,
                                            tool results, handoffs)
                                                        │
                                                        ▼
                                              session.addItems(newItems)

Next run with the same session id starts from the stored history.

Session implementations:
  ├─ MemorySession                     (in process; tests, single instance)
  ├─ OpenAIConversationsSession        (stored by OpenAI)
  ├─ OpenAIResponsesCompactionSession  (wraps another session, compacts history)
  └─ your class implementing Session   (Redis, Postgres, DynamoDB...)
```

---

## Tool Integration Pattern

```
┌────────────────────────────────────────────────────────────┐
│              Tool Integration Architecture                │
└────────────────────────────────────────────────────────────┘

       Agent Processing
              │
              ▼
    ┌─────────────────────┐
    │ LLM Decides         │
    │ Tool is Needed      │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │ Select Tool & Parameters    │
    │ ├─ Tool name               │
    │ ├─ Arguments               │
    │ └─ Validation schema       │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │ Validate Parameters         │
    │ (Type checking)             │
    └──────────┬──────────────────┘
               │
        ┌──────┴───────┐
        │              │
     Valid         Invalid
        │              │
        ▼              ▼
    ┌────────┐   ┌──────────┐
    │ Execute│   │Error     │
    │ Tool   │   │Response  │
    └───┬────┘   └──────────┘
        │
        ▼
    ┌──────────────────────┐
    │ Format Result        │
    │ ├─ Success output    │
    │ ├─ Error handling    │
    │ └─ Type conversion   │
    └──────────┬───────────┘
               │
               ▼
        Tool Result to Agent
```

---

## Error Handling & Resilience Pattern

```
Model call
   │
   ├─ fails (network, timeout, 429/5xx)
   │      │
   │      ▼
   │   modelSettings.retry.policy decides (retryPolicies.*)
   │      ├─ retry ──► wait (backoff, or the server's Retry-After) ──► call again
   │      └─ give up ──► error thrown from run()
   │
   ├─ tool throws ──► errorFunction formats it ──► returned to the model as the tool result
   │
   ├─ maxTurns reached ──► errorHandlers.maxTurns, else MaxTurnsExceededError
   │
   ├─ guardrail tripwire ──► Input/OutputGuardrailTripwireTriggered
   │
   └─ signal aborted ──► run rejects; in-flight request cancelled
```

Retries happen inside one `run()` and only when replaying the request is safe.

---

## Structured Output Processing

```
Agent with outputType: z.object(...)
   │
   ▼
Zod schema ──► JSON Schema (strict) sent with the request
   │
   ▼
Model returns JSON constrained to the schema
   │
   ▼
SDK parses and validates with Zod
   │
   ├─ valid   ──► result.finalOutput typed as z.infer<typeof schema>
   └─ invalid ──► ModelBehaviorError, or errorHandlers.invalidFinalOutput
```

---

## Approvals

```
run(agent, input)
   │
   ▼
Model calls a tool with needsApproval
   │
   ▼
Run stops: result.interruptions = [RunToolApprovalItem, ...]
   │
   ▼
result.state.toString() ──► your database
   │
   │   (later, any process)
   ▼
RunState.fromString(agent, saved)
state.approve(item)  or  state.reject(item, { message })
   │
   ▼
run(agent, state) ──► tool runs (or the model hears it was rejected) ──► final output
```

---

## Type Safety Flow

```
┌────────────────────────────────────────────────────────┐
│          TypeScript Type Safety Flow                  │
└────────────────────────────────────────────────────────┘

  TypeScript Source
      │
      ▼
  ┌──────────────────────┐
  │ Type Checking        │ ◄──── Compile-time validation
  │ (tsc)                │
  └──────┬───────────────┘
         │
    ┌────┴────┐
    │         │
  Errors   Continue
    │         │
    ▼         ▼
  Fix     ┌──────────────────┐
   │      │ Runtime Validation
   │      │ (Zod schemas)
   │      └────┬─────────────┘
   │           │
   │           ▼
   │      ┌──────────────────┐
   │      │ API Layer        │ ◄──── Runtime safety
   │      │ (Type guards)    │
   │      └────┬─────────────┘
   │           │
   └───────────┴──────────────► Safe Execution

Type Layers:
  1. TypeScript (Compile-time)
  2. Zod (Runtime schema)
  3. API validation (Input/Output)
  4. Business logic (Domain types)
```

---

## Component Interaction Diagram

```
┌────────────────────────────────────────────────────────┐
│      OpenAI Agents SDK Component Interactions        │
└────────────────────────────────────────────────────────┘

  User Application
      │
      ├─────────────────┬──────────────┬───────────────┐
      │                 │              │               │
      ▼                 ▼              ▼               ▼
   Agent          Runner           Handoff        Tool
   Creation       Execution        Delegation      Invocation
      │                 │              │               │
      │     ┌───────────┘              │               │
      │     │          ┌───────────────┘               │
      │     │          │           ┌───────────────────┘
      │     │          │           │
      ▼     ▼          ▼           ▼
   ┌─────────────────────────────────┐
   │        Session Layer             │
   │ (Conversation State & Memory)    │
   └─────────────────────────────────┘
      │
      ▼
   ┌─────────────────────────────────┐
   │    Guardrail Layer               │
   │ (Validation & Safety)            │
   └─────────────────────────────────┘
      │
      ▼
   ┌─────────────────────────────────┐
   │    Model Layer                   │
   │ (LLM Integration)                │
   └─────────────────────────────────┘
```

---

The Python guide has the same flows for the Python SDK: [Python diagrams](/openai-agents-sdk-guides/openai_agents_sdk_diagrams/).


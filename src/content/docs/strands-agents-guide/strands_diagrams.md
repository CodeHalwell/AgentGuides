---
title: "Strands Agents: Diagrams"
description: "Text diagrams of how Strands Agents works in both SDKs: the agent loop and where hooks fire, tool execution, structured output, sessions, interrupts, multi-agent patterns and deployment on AWS."
framework: "strands"
sidebar:
  label: "Diagrams"
  order: 9
---

Diagrams of what happens inside the Python and TypeScript SDKs (Python `strands-agents` %%version=strands-agents%%, TypeScript `@strands-agents/sdk` %%version=@strands-agents/sdk%%). Names are given as Python / TypeScript where they differ. Code for each step is in the [Python](/strands-agents-guide/python/strands_comprehensive_guide/) and [TypeScript](/strands-agents-guide/typescript/strands_typescript_comprehensive_guide/) comprehensive guides.

## The agent loop

One call to `agent(...)` / `agent.invoke(...)`:

```text
 agent("prompt", invocation_state=..., limits=...)
   │
   ├─ BeforeInvocationEvent ............ hooks may rewrite input or cancel
   ├─ append user message to agent.messages ── MessageAddedEvent
   │
   │  ┌──────────────────────── event loop cycle ─────────────────────────┐
   │  │                                                                   │
   │  │  conversation manager trims history if needed                     │
   │  │  BeforeModelCallEvent ........... hooks may cancel                │
   │  │  model.stream(messages, tool_specs, system_prompt)               │
   │  │     └─ text / tool-use deltas → stream_async / stream() events    │
   │  │  AfterModelCallEvent ............ hooks may set retry             │
   │  │                                                                   │
   │  │  stop_reason == tool_use ?                                        │
   │  │     yes ─► BeforeToolsEvent                                       │
   │  │            for each tool call (concurrently by default):          │
   │  │               BeforeToolCallEvent ... cancel / rewrite / interrupt│
   │  │               run tool  (error → error result, loop continues)    │
   │  │               AfterToolCallEvent .... replace result / retry      │
   │  │            AfterToolsEvent ........... may end the turn           │
   │  │            append tool results ─────────────────────► next cycle  │
   │  │     no ──► leave the loop                                         │
   │  └───────────────────────────────────────────────────────────────────┘
   │
   ├─ structured output requested and not produced? force the output tool once
   ├─ session manager saves messages and state
   ├─ AfterInvocationEvent ............. hooks may resume with new input
   ▼
 AgentResult(stop_reason, message, metrics, structured_output, interrupts)
```

The loop also stops on `limits` (turns, tokens), cancellation, an interrupt, or `max_tokens` (Python raises `MaxTokensReachedException`). Throttling is retried inside the model call by the retry strategy; a context overflow makes the conversation manager trim and the call repeat.

## Tool execution

```text
 model asks for:  [get_order(ORD-1), get_invoice(INV-7), refund(ORD-1, 250)]
                         │
            ┌────────────┼──────────────────┐        ConcurrentToolExecutor (default)
            ▼            ▼                  ▼
       get_order    get_invoice          refund
            │            │                  │
            │            │        tool_context.interrupt("approval")
            │            │                  │  ─► loop stops, stop_reason = interrupt
            ▼            ▼                  ▼
       {status: success} {status: error}   (resumed later with interruptResponse)
                         │
          all results appended as one user message ─► next model call
```

With `SequentialToolExecutor` the three run one after another in the order the model gave.

## Structured output

```text
 agent(prompt, structured_output_model=Invoice)        TS: { structuredOutputSchema: Invoice }
   │
   ├─ tool list = your tools + output tool (schema = Invoice)
   │     Python: tool named after the model class      TS: strands_structured_output
   │
   ├─ model may call ordinary tools first ... loop as usual
   │
   ├─ model calls the output tool ──► validate ──► result.structured_output
   │                                     │
   │                                  invalid → error result, model retries
   │
   └─ model ends without it ──► one more call with tool choice forced
                                   └─ still nothing → StructuredOutputException / StructuredOutputError
```

## Sessions and state

```text
            ┌──────────────────────── Agent ─────────────────────────┐
            │  messages            ◄── trimmed by conversation manager│
            │  state / appState    ◄── tools and hooks read and write │
            │  (interrupt state)                                      │
            └───────────────┬─────────────────────────────────────────┘
                            │ restore on create, save after each call
                            ▼
                   Session manager
            ┌───────────────┼──────────────────────────┬──────────────────────┐
            ▼               ▼                          ▼                      ▼
     FileSessionManager  S3SessionManager   SnapshotSessionManager /   AgentCoreMemory-
     (one file per       (one object per    TS SessionManager           SessionManager
      message)            message)          + LocalFileStorage | S3Storage (Python, bedrock-agentcore)

  invocation_state / invocationState: per call only, never saved, never shown to the model
```

## Interrupts across requests

```text
 request 1                          storage                       request 2 (hours later)
 ─────────                          ───────                       ──────────────────────
 agent = make(session "case-9")
 agent("refund £250")
   tool → interrupt("approval") ──► snapshot incl. pending ───┐
 return {pending: [id, reason]}       interrupt                │
                                                               │   agent = make(session "case-9")
                                                               └─► agent([{interruptResponse:
                                                                        {interruptId: id,
                                                                         response: "approve"}}])
                                                                     tool runs again,
                                                                     interrupt() returns "approve"
                                                                     → "Refunded £250"
```

## Multi-agent patterns

### Agents as tools

```text
                 ┌──────────────────────┐
  user ────────► │  supervisor agent    │ ◄── decides which specialist to call
                 └───┬──────────────┬───┘
         tool call   │              │   tool call (input: "question")
                     ▼              ▼
          ┌──────────────┐   ┌──────────────┐
          │ sales agent  │   │billing agent │   each has its own tools and model
          └──────────────┘   └──────────────┘
                     │              │
                     └──── answers as tool results ────► supervisor writes the reply
```

### Swarm

```text
            handoff_to_agent / {agentId}         handoff
  task ──► ┌────────────┐ ───────────────► ┌────────────┐ ──────► ┌────────────┐
           │ researcher │                  │  analyst   │         │  reviewer  │ ──► final answer
           └────────────┘ ◄─────────────── └────────────┘         └────────────┘   (no handoff)
                 shared: original task, handoff history, context passed so far
                 limits: max_handoffs / max_iterations / timeouts   (TS: maxSteps, timeout in ms)
```

Python members hand off with a `handoff_to_agent` tool; TypeScript members return structured output with an `agentId`.

### Graph

```text
                      ┌───────────┐
                      │  collect  │ ◄──────────────┐
                      └─────┬─────┘                │ condition: validation failed
                            ▼                      │
                      ┌───────────┐ ───────────────┘
                      │ validate  │
                      └─────┬─────┘
                            │ condition: validation passed
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼          fan-out: run in parallel
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ market   │  │technical │  │regulatory│
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             └─────────────┼─────────────┘          fan-in: waits for all three
                           ▼
                     ┌───────────┐
                     │  writer   │ ──► GraphResult / MultiAgentResult
                     └───────────┘
  each node receives: original task + outputs of the nodes feeding it
```

### A2A

```text
  ┌────────── process A ──────────┐                 ┌────────── process B ──────────┐
  │ local Agent                   │   HTTP (A2A)    │ A2AServer(agent_factory=...)  │
  │   tools: [make_a2a_client] ───┼────────────────►│   /.well-known/agent-card.json│
  │ or Graph node: A2AAgent(url) ─┼────────────────►│   one Agent per A2A context   │
  └───────────────────────────────┘                 └───────────────────────────────┘
```

## Deployment on AWS

```text
                       ┌───────────────────── AgentCore Runtime ─────────────────────┐
  client ─ SigV4/JWT ─►│ session A ─► microVM ─► BedrockAgentCoreApp ─► Strands Agent │──► Bedrock models
  (InvokeAgentRuntime) │ session B ─► microVM ─► BedrockAgentCoreApp ─► Strands Agent │──► tools / MCP / Gateway
                       └──────────────┬──────────────────────────────────────────────┘
                                      ├─► AgentCore Memory (session manager / memory store)
                                      ├─► AgentCore Identity (third-party API keys, OAuth)
                                      └─► CloudWatch / X-Ray (OpenTelemetry spans)

  alternatives:  API Gateway ─► Lambda (short, stateless; sessions in S3)
                 ALB ─► ECS Fargate / EKS (FastAPI or Express; sessions in S3 or DynamoDB)
```

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | Py 1.62.0 / TS 1.22.0 | New page for the standalone Strands guide; architecture diagrams moved from the former Bedrock Strands page and extended to cover hooks, structured output, sessions, interrupts and A2A for both SDKs. |

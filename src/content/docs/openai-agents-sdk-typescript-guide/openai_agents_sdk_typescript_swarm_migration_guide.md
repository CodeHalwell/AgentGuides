---
title: "OpenAI Agents SDK with TypeScript: Coming from Swarm"
description: "Swarm was a Python-only experiment; how its ideas map onto the TypeScript Agents SDK for teams porting a Swarm prototype."
framework: "openai-agents-sdk"
language: "typescript"
sidebar:
  label: "Coming from Swarm"
  order: 7
---

> Checked against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%).

There was never a TypeScript Swarm. [openai/swarm](https://github.com/openai/swarm) was a small, Python-only educational library, and it is no longer developed. If you are porting a Swarm prototype to TypeScript you are really doing two things at once, changing language and changing framework. The [Python migration guide](/openai-agents-sdk-guides/openai_agents_sdk_swarm_migration_guide/) covers the framework change in Python; this page maps Swarm's ideas directly onto the TypeScript SDK.

## Concept mapping

| Swarm (Python) | TypeScript Agents SDK |
|----------------|-----------------------|
| `Swarm().run(agent, messages)` | `await run(agent, input)` or `new Runner().run(...)` |
| `Agent(name, instructions, functions)` | `new Agent({ name, instructions, tools })` |
| Plain Python function as a tool | `tool({ name, description, parameters: z.object(...), execute })` |
| A function that returns an `Agent` | `handoffs: [otherAgent]` |
| `context_variables` dict | `run(agent, input, { context })`; tools read `runContext.context` |
| `instructions=lambda context_variables: ...` | `instructions: (runContext) => ...` |
| `response.messages` | `result.history`, or a `session` |
| `response.agent` | `result.lastAgent` |
| `max_turns=float("inf")` (default) | `maxTurns: 10` (default); `null` for no limit |
| `stream=True` | `run(agent, input, { stream: true })` |
| `debug=True` | Tracing, on by default |
| Chat Completions API | Responses API by default; `setOpenAIAPI('chat_completions')` to switch |

## The same triage agent in TypeScript

A Swarm triage agent with one tool and one handoff, written for the TypeScript SDK:

```typescript
import { Agent, RunContext, run, tool } from '@openai/agents';
import { z } from 'zod';

interface Customer {
  name: string;
}

const getOrder = tool({
  name: 'get_order',
  description: 'Look up an order for the current customer',
  parameters: z.object({ orderId: z.string() }),
  execute: async ({ orderId }, runContext?: RunContext<Customer>) =>
    `Order ${orderId} for ${runContext?.context.name ?? 'unknown'} has shipped.`,
});

const refunds = new Agent<Customer>({
  name: 'Refunds',
  handoffDescription: 'Handles refund requests',
  instructions: 'Handle refunds.',
});

const triage = new Agent<Customer>({
  name: 'Triage',
  instructions: (runContext) =>
    `Help ${runContext.context.name}. Hand refund requests to Refunds.`,
  tools: [getOrder],
  handoffs: [refunds],
});

const result = await run(triage, 'Where is order 42?', { context: { name: 'Alice' } });
console.log(result.lastAgent?.name, result.finalOutput);
```

## Checklist

- Rewrite each Swarm function as a `tool()` with a Zod schema; the schema replaces Swarm's signature introspection.
- Replace `context_variables` with a typed context object passed to `run()`.
- Replace "return an agent" functions with `handoffs`; use `agent.asTool()` where the caller should keep control.
- Set `maxTurns` deliberately (Swarm had no limit) and `model` on each agent.
- Choose where history lives: `MemorySession`, `OpenAIConversationsSession`, or your own `Session`.
- Add what Swarm lacked where you need it: `outputType`, guardrails, [approvals](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/#human-in-the-loop-approvals), and offline tests with [`ScriptedModel`](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_production_guide/#testing-without-the-api).

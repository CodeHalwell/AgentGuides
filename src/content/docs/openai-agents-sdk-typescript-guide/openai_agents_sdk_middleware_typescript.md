---
title: "OpenAI Agents SDK Middleware and Guardrails (TypeScript)"
description: "Implement middleware, guardrails, and tracing with the OpenAI Agents SDK in TypeScript."
framework: openai-agents-sdk-typescript
---

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%).

## Goals
- Add request/response middlewares to Agent and Runner
- Enforce policy/PII guardrails and human-in-the-loop
- Trace tokens, latency, and tool IO with OpenTelemetry

## Middleware Pattern

```ts
type Next = (input: string) => Promise<string>;
type Middleware = (input: string, next: Next) => Promise<string>;

const policyMw: Middleware = async (input, next) => {
  if (/ssn|credit card/i.test(input)) throw new Error("Policy violation");
  return next(input);
};

const redactMw: Middleware = async (input, next) => {
  const out = await next(input);
  return out.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "***-**-****");
};

function chain(mws: Middleware[], handler: Next): Next {
  return mws.reduceRight<Next>((acc, mw) => (input) => mw(input, acc), handler);
}
```

## Integrate with Agent/Runner

```ts
import { Agent, run } from "@openai/agents";

const agent = new Agent({ name: "Assistant", instructions: "Answer concisely." });

async function llmHandler(input: string): Promise<string> {
  const result = await run(agent, input);
  return result.finalOutput ?? "";
}

const wrapped = chain([policyMw, redactMw], llmHandler);

export async function handle(input: string) {
  return wrapped(input);
}
```

For checks that belong inside the agent loop, prefer the SDK's own hooks: `inputGuardrails` / `outputGuardrails` on the agent, tool-level guardrails on `tool()`, and `needsApproval` for actions a human must sign off.

## HITL
- Mark sensitive tools with `needsApproval`; the run returns `result.interruptions`, and `JSON.stringify(result.state)` lets you persist it while a human decides, then resume with `RunState.fromString(agent, saved)`.

## Tracing
- Create spans per agent step; add attributes: model, tokens, tool names.


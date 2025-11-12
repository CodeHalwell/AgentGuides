---
layout: default
title: "OpenAI Agents SDK Middleware and Guardrails (TypeScript)"
description: "Implement middleware, guardrails, and tracing with the OpenAI Agents SDK in TypeScript."
---

# OpenAI Agents SDK Middleware and Guardrails (TypeScript)

Last verified: 2025-11

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
  return mws.reduceRight((acc, mw) => (input) => mw(input, acc), handler);
}
```

## Integrate with Agent/Runner

```ts
import { Agent, Runner } from "openai-agents"; // adjust to actual package

async function llmHandler(input: string) {
  // invoke agent/runner here and return stringified content
  return `OK: ${input}`;
}

const wrapped = chain([policyMw, redactMw], llmHandler);

export async function run(input: string) {
  return wrapped(input);
}
```

## HITL
- On guardrail failure, persist state and handoff to a human queue.

## Tracing
- Create spans per agent step; add attributes: model, tokens, tool names.

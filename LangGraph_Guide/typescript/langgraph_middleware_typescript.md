---
layout: default
title: "LangGraph Middleware and Guardrails (TypeScript)"
description: "Implement request/response middlewares, guardrails, and tracing in LangGraph TypeScript."
---

# LangGraph Middleware and Guardrails (TypeScript)

Last verified: 2025-11 • Source: langgraph-js templates

## Wrapper Pattern

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

## Integrate with Nodes

```ts
import { StateGraph } from "@langchain/langgraph";

type State = { input: string; output?: string };

async function llmHandler(text: string): Promise<string> {
  // call your LLM here
  return `OK: ${text}`;
}

const wrapped = chain([policyMw, redactMw], llmHandler);

function node(state: State): State {
  // execute wrapped LLM
  return { ...state, output: "" };
}

const g = new StateGraph<State>({ channels: { input: "input", output: "output" } });
g.addNode("n1", node);
```

## Tracing

- Use OpenTelemetry SDK for Node; wrap middlewares with spans
- Add attributes: graph, node, attempt, tokens


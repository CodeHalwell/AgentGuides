---
title: "LangGraph Middleware and Guardrails (TypeScript)"
description: "Implement request/response middlewares, guardrails, and tracing in LangGraph TypeScript."
framework: "langgraph"
language: "typescript"
sidebar:
  label: "Middleware & guardrails"
---

> Verified against `@langchain/langgraph` %%version=@langchain/langgraph%% and `langchain` %%version=langchainjs%% (checked %%versions-checked%%).

Middleware belongs to `createAgent`; a hand-written `StateGraph` has none, because each node is your own function. There are two common shapes.

- **Agents built with `createAgent`** (from `langchain`) take `middleware: [...]`. Use `createMiddleware` for your own, or the built-ins (`piiMiddleware`, `modelCallLimitMiddleware`, `toolCallLimitMiddleware`, `humanInTheLoopMiddleware`, `modelRetryMiddleware`, `summarizationMiddleware`, `openAIModerationMiddleware` and others).
- **Hand-written graphs** compose plain wrapper functions around the model call, as below.

## Agent middleware

```ts
import { createAgent, createMiddleware, piiMiddleware } from "langchain";

const policy = createMiddleware({
  name: "Policy",
  beforeModel: (state) => {
    const last = state.messages.at(-1);
    if (/ssn|credit card/i.test(String(last?.content))) {
      throw new Error("Policy violation");
    }
    return undefined;
  },
});

const agent = createAgent({
  model: "openai:%%model=openai-small%%",
  tools: [],
  middleware: [
    policy,
    piiMiddleware("email", { strategy: "redact" }), // redact e-mail addresses
  ],
});
```

For approvals before a tool runs, use `humanInTheLoopMiddleware`; a complete example is in [Interrupts, Send, Command and subgraphs](/langgraph-guide/typescript/langgraph_control_flow_typescript/#approvals-in-createagent-agents).

## Wrapper pattern for your own nodes

```ts
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";

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

// Use the wrapped handler inside a node

const State = Annotation.Root({
  input: Annotation<string>,
  output: Annotation<string>,
});

async function llmHandler(text: string): Promise<string> {
  // call your LLM here
  return `OK: ${text}`;
}

const wrapped = chain([policyMw, redactMw], llmHandler);

const graph = new StateGraph(State)
  .addNode("n1", async (state) => ({ output: await wrapped(state.input) }))
  .addEdge(START, "n1")
  .addEdge("n1", END)
  .compile();

console.log(await graph.invoke({ input: "hello" }));   // { input: 'hello', output: 'OK: hello' }
```

## Tracing

- Wrap middlewares in OpenTelemetry spans (see [Observability](/langgraph-guide/typescript/langgraph_observability_typescript/)), or set `LANGSMITH_TRACING=true` to trace LangGraph runs in LangSmith without code changes.
- Useful span attributes: graph name, node name (`config.metadata.langgraph_node`), attempt number, token usage.

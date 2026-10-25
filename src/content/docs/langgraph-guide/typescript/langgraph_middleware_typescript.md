---
title: "LangGraph Middleware and Guardrails (TypeScript)"
description: "Agent middleware in LangChain.js (custom hooks, built-ins, execution order) and the wrapper pattern for hand-written LangGraph TypeScript graphs."
framework: "langgraph"
language: "typescript"
sidebar:
  label: "Middleware & guardrails"
  order: 25
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

## Writing your own middleware

`createMiddleware({ name, ... })` accepts any of these hooks:

| Hook | Runs | Receives | May return |
|------|------|----------|------------|
| `beforeAgent` | once, at the start of `invoke` | `(state, runtime)` | a state update, `undefined`, or `{ jumpTo }` |
| `beforeModel` | before each model call | `(state, runtime)` | the same |
| `wrapModelCall` | around each model call | `(request, handler)` | the `AIMessage` (or a `Command`) |
| `afterModel` | after each model call, before tools run | `(state, runtime)` | a state update, `undefined`, or `{ jumpTo }` |
| `wrapToolCall` | around each tool call | `(request, handler)` | a `ToolMessage` or a `Command` |
| `afterAgent` | once, at the end | `(state, runtime)` | a state update or `undefined` |

A middleware can also declare `stateSchema` (extra state keys, persisted with the thread), `contextSchema` (per-run values passed as `context` on `invoke`, not persisted) and `tools` (extra tools it adds to the agent). To jump, write the hook as `{ hook, canJumpTo: ["end"] }` and return `{ jumpTo: "end" }`; the targets are `"model"`, `"tools"` and `"end"`.

The example below uses all three kinds: a budget kept in middleware state, a model-call wrapper that sets the system prompt and logs latency, and a tool-call wrapper that refuses a tool based on run context. `fakeModel()` stands in for a real model.

```ts
import { createAgent, createMiddleware, fakeModel, tool, ToolMessage } from "langchain";
import { AIMessage } from "@langchain/core/messages";
import * as z from "zod";

// 1. Middleware with its own state: count model calls and stop after a budget.
const MAX_MODEL_CALLS = 1; // low so the example hits it

const budget = createMiddleware({
  name: "Budget",
  stateSchema: z.object({ modelCalls: z.number().default(0) }),
  beforeModel: {
    canJumpTo: ["end"],
    hook: (state) => {
      if (state.modelCalls >= MAX_MODEL_CALLS) {
        return {
          jumpTo: "end",
          messages: [new AIMessage("Stopping: model-call budget used up.")],
        };
      }
      return { modelCalls: state.modelCalls + 1 };
    },
  },
});

// 2. Wrap every model call: add a system prompt and log latency.
const timing = createMiddleware({
  name: "Timing",
  wrapModelCall: async (request, handler) => {
    const started = Date.now();
    const response = await handler({
      ...request,
      systemPrompt: "Answer briefly. Use tools when asked about orders.",
    });
    console.log(`model call: ${Date.now() - started} ms, ${request.messages.length} messages in`);
    return response;
  },
});

// 3. Wrap every tool call: block a tool for read-only users.
const readOnly = createMiddleware({
  name: "ReadOnly",
  contextSchema: z.object({ role: z.enum(["admin", "viewer"]) }),
  wrapToolCall: async (request, handler) => {
    if (request.toolCall.name === "refund" && request.runtime.context.role !== "admin") {
      return new ToolMessage({
        content: "Refunds need an admin.",
        tool_call_id: request.toolCall.id ?? "",
      });
    }
    return handler(request);
  },
});

const refund = tool(async ({ orderId }) => `refunded ${orderId}`, {
  name: "refund",
  description: "Refund an order",
  schema: z.object({ orderId: z.string() }),
});

const model = fakeModel()
  .respondWithTools([{ name: "refund", args: { orderId: "42" } }])
  .respond(new AIMessage("I could not refund order 42."));

const agent = createAgent({
  model,
  tools: [refund],
  middleware: [budget, timing, readOnly],
});

const result = await agent.invoke(
  { messages: [{ role: "user", content: "Refund order 42" }] },
  { context: { role: "viewer" } },
);
for (const m of result.messages) {
  const calls = AIMessage.isInstance(m) ? m.tool_calls?.map((c) => c.name) : undefined;
  console.log(m.type, calls?.length ? calls : m.content);
}
console.log("modelCalls:", result.modelCalls);
```

Output:

```text
model call: 3 ms, 1 messages in
human Refund order 42
ai [ 'refund' ]
tool Refunds need an admin.
ai Stopping: model-call budget used up.
modelCalls: 1
```

In `wrapModelCall`, pass a modified copy of `request` to `handler` (as with `systemPrompt` here) rather than mutating it; `request` also carries `model`, `messages`, `tools`, `state` and `runtime`, so the same hook can swap models or filter tools per call. Not calling `handler` at all skips the model and lets you return a cached or canned `AIMessage`.

### Execution order

With `middleware: [A, B]`, `before*` hooks run in list order (A then B), `after*` hooks in reverse (B then A), and `wrap*` hooks nest with the first entry outermost (A's wrapper calls B's, which calls the model). Put cheap guards that may short-circuit early in the list, and wrappers that should see the final request (logging, caching) late.

## Built-in middleware

All of these are exported from `langchain`:

| Middleware | What it does |
|------------|--------------|
| `modelCallLimitMiddleware({ threadLimit, runLimit, exitBehavior })` | Caps model calls per thread and/or per run; `exitBehavior` is `"end"` or `"error"`. |
| `toolCallLimitMiddleware({ toolName, threadLimit, runLimit, exitBehavior })` | The same for tool calls, for one tool or all; `exitBehavior` defaults to `"continue"` (blocked calls get an error `ToolMessage`). |
| `modelRetryMiddleware(...)`, `toolRetryMiddleware(...)` | Retry with exponential back-off. Defaults: `maxRetries: 2`, `initialDelayMs: 1000`, `backoffFactor: 2`, `maxDelayMs: 60000`, `jitter: true`. `toolRetryMiddleware` takes `tools` to limit it to some tools and `onFailure` (`"continue"`, the default, returns the error to the model; `"error"` rethrows). |
| `modelFallbackMiddleware(...models)` | Tries the given models in order when the primary model call fails (models as instances or `"provider:model"` strings). |
| `toolErrorMiddleware({ onError, tools })` | Turns tool exceptions into error `ToolMessage`s: `onError(error, request)` returns the content to show the model, or nothing to rethrow. |
| `humanInTheLoopMiddleware({ interruptOn })` | Pauses before chosen tools for approve / edit / reject decisions. |
| `summarizationMiddleware(...)` | Summarises old messages when the history gets long. |
| `contextEditingMiddleware({ edits })` | Edits the context before model calls; `ClearToolUsesEdit` clears old tool results. |
| `piiMiddleware(type, { strategy })`, `piiRedactionMiddleware(...)` | Detect PII; `strategy` is `"block"`, `"redact"`, `"mask"` or `"hash"`. |
| `openAIModerationMiddleware(...)` | Runs OpenAI's moderation endpoint on input and/or output (`checkInput`, `checkOutput`). |
| `llmToolSelectorMiddleware(...)` | Uses a small model to pick relevant tools before the main call. |
| `dynamicSystemPromptMiddleware(fn)` | Builds the system prompt from state and context each call. |
| `todoListMiddleware()` | Gives the agent a `write_todos` tool and todo state for multi-step plans. |
| `toolEmulatorMiddleware(...)` | Has a model emulate tool results, for testing without real tools. |
| `anthropicPromptCachingMiddleware(...)`, `bedrockPromptCachingMiddleware(...)` | Add provider prompt-cache markers. |
| `providerToolSearchMiddleware(...)` | Defers large tool sets to provider-side tool search. |

A typical production stack combines retries with limits. Here the tool fails twice and succeeds on the third attempt:

```ts
import {
  createAgent, fakeModel, tool,
  modelCallLimitMiddleware, toolCallLimitMiddleware, toolRetryMiddleware,
} from "langchain";
import { AIMessage } from "@langchain/core/messages";
import * as z from "zod";

let attempts = 0;
const lookup = tool(
  async ({ sku }) => {
    attempts += 1;
    if (attempts < 3) throw new Error("inventory service timed out");
    return `${sku}: 12 in stock`;
  },
  { name: "lookup", description: "Look up stock for a SKU", schema: z.object({ sku: z.string() }) },
);

const model = fakeModel()
  .respondWithTools([{ name: "lookup", args: { sku: "A-1" } }])
  .respond(new AIMessage("A-1 has 12 in stock."));

const agent = createAgent({
  model,
  tools: [lookup],
  middleware: [
    // Retry failing tools up to 2 more times with exponential back-off.
    toolRetryMiddleware({ maxRetries: 2, initialDelayMs: 100, jitter: false }),
    // At most 5 tool calls per run for `lookup`.
    toolCallLimitMiddleware({ toolName: "lookup", runLimit: 5 }),
    // At most 10 model calls per run; end the run instead of throwing.
    modelCallLimitMiddleware({ runLimit: 10, exitBehavior: "end" }),
  ],
});

const result = await agent.invoke({ messages: [{ role: "user", content: "Stock for A-1?" }] });
console.log(result.messages.at(-2)?.content, "| attempts:", attempts);
console.log(result.messages.at(-1)?.content);
```

Output:

```text
A-1: 12 in stock | attempts: 3
A-1 has 12 in stock.
```

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

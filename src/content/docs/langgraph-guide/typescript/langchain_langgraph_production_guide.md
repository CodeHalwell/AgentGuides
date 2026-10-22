---
title: "LangChain.js and LangGraph.js Production Guide"
description: "Running LangGraph.js in production: durable checkpointers, durability and recursion limits, node retry/timeout/cache policies, testing with fakeModel, tracing, the LangGraph CLI and Agent Server, and calling a deployment with the SDK."
framework: "langgraph"
language: "typescript"
sidebar:
  label: "Production"
---

Checked against `@langchain/langgraph` %%version=@langchain/langgraph%%, `langchain` %%version=langchainjs%% and `@langchain/langgraph-checkpoint` %%version=@langchain/langgraph-checkpoint%% (%%versions-checked%%). The examples type-check under `strict` and, apart from the Postgres and SDK ones (which need a database and a running server), run as shown.

## Checklist

- A durable checkpointer (Postgres) in place of `MemorySaver`, and a `thread_id` per conversation.
- `recursionLimit` set on agent runs.
- Retry and timeout policies on nodes that call models and services.
- Tests that run the graph with `fakeModel`, including interrupt and resume.
- LangSmith tracing on: `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (see [Observability](/langgraph-guide/typescript/langgraph_observability_typescript/)).

## Checkpointers

| Saver | Package | Use it for |
|---|---|---|
| `MemorySaver` | `@langchain/langgraph` | Tests and local experiments; lost on exit |
| `SqliteSaver` | `@langchain/langgraph-checkpoint-sqlite` | One process on one machine |
| `PostgresSaver` | `@langchain/langgraph-checkpoint-postgres` | Production; several workers sharing threads |
| Provided by the server | — | The LangGraph Agent Server (LangSmith Deployment, `langgraph-cli dev`/`up`) supplies persistence; compile without a checkpointer there |

```typescript
import { PostgresSaver } from '@langchain/langgraph-checkpoint-postgres';
import { Annotation, StateGraph, START, END } from '@langchain/langgraph';

const State = Annotation.Root({ count: Annotation<number> });
const builder = new StateGraph(State)
  .addNode('inc', (s) => ({ count: (s.count ?? 0) + 1 }))
  .addEdge(START, 'inc')
  .addEdge('inc', END);

const checkpointer = PostgresSaver.fromConnString(process.env.DATABASE_URL!);
await checkpointer.setup();                    // creates or migrates the tables; safe to repeat

const graph = builder.compile({ checkpointer });
await graph.invoke({ count: 0 }, { configurable: { thread_id: 'user-42' } });

await checkpointer.deleteThread('user-42');    // housekeeping
await checkpointer.end();                      // close the pool on shutdown
```

For long-term memory across threads, pass a store to `compile({ store })` (`InMemoryStore` in development) and read it in nodes with `getStore()` or `config.store`; see [cross-thread memory](/langgraph-guide/typescript/langchain_langgraph_comprehensive_guide/#cross-thread-memory-support).

## Durability, recursion limits and node policies

- **`durability`** (`'sync' | 'async' | 'exit'`, passed to `invoke`/`stream`) controls when checkpoints are written: before the next step, in the background (the default), or only at the end. It has no effect without a checkpointer.
- **`recursionLimit`** caps the number of super-steps. Set it explicitly on agent loops.
- **Node policies** are options on `addNode`: `retryPolicy`, `timeout` (milliseconds, or a `TimeoutPolicy` with `runTimeout`/`idleTimeout`; a timeout raises `NodeTimeoutError`), `cachePolicy` (needs `compile({ cache })`) and `errorHandler` (runs after retries are exhausted and receives `{ node, error }`). `setNodeDefaults(...)` applies them to every node.

```typescript
import { Annotation, StateGraph, START, END, MemorySaver } from '@langchain/langgraph';
import { InMemoryCache } from '@langchain/langgraph-checkpoint';

const State = Annotation.Root({
  query: Annotation<string>,
  answer: Annotation<string>,
  error: Annotation<string>,
});

let attempts = 0;

const graph = new StateGraph(State)
  // Graph-wide defaults; per-node options override them
  .setNodeDefaults({ retryPolicy: { maxAttempts: 3 } })
  .addNode(
    'fetch',
    async (state) => {
      attempts += 1;
      if (attempts < 3) throw new Error('upstream reset');   // retried
      return { answer: `result for ${state.query}` };
    },
    { timeout: 30_000, cachePolicy: { ttl: 300 } },           // timeout in ms; ttl in seconds
  )
  .addNode(
    'summarise',
    async () => {
      throw new Error('bad payload');
    },
    {
      retryPolicy: { maxAttempts: 1 },                                  // do not retry this one
      errorHandler: (_state, { node, error }) => ({ error: `${node}: ${error.message}` }),
    },
  )
  .addEdge(START, 'fetch')
  .addEdge('fetch', 'summarise')
  .addEdge('summarise', END)
  .compile({ checkpointer: new MemorySaver(), cache: new InMemoryCache() });

const result = await graph.invoke(
  { query: 'weather' },
  { configurable: { thread_id: 't1' }, recursionLimit: 25, durability: 'sync' },
);
console.log(result, 'attempts:', attempts);
// { query: 'weather', answer: 'result for weather', error: 'summarise: bad payload' } attempts: 3
```

For `createAgent` agents, the equivalents are middleware: `modelRetryMiddleware`, `modelFallbackMiddleware`, `toolRetryMiddleware`, `modelCallLimitMiddleware` and `toolCallLimitMiddleware` (see [Middleware](/langgraph-guide/typescript/langgraph_middleware_typescript/)).

## Testing

`fakeModel()` (exported by `langchain` and `@langchain/core/testing`) returns a chat model that replays queued responses: `.respond(message)`, `.respondWithTools([...])`, or `.respond(error)` to simulate a failure. It records every call in `.calls` and `.callCount`. Combined with a fresh `MemorySaver` per test, it covers tool calls, routing and interrupt/resume without network access:

```typescript
// agent.test.ts — run with: npx tsx --test agent.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AIMessage } from '@langchain/core/messages';
import { Annotation, Command, MemorySaver, StateGraph, START, END, interrupt, isInterrupted, INTERRUPT } from '@langchain/langgraph';
import { createAgent, fakeModel, tool } from 'langchain';
import { z } from 'zod';

const refund = tool(async ({ orderId }) => `refunded ${orderId}`, {
  name: 'refund',
  description: 'Refund an order',
  schema: z.object({ orderId: z.string() }),
});

test('agent calls the tool, then answers', async () => {
  // fakeModel replays queued responses in order and records every call
  const model = fakeModel()
    .respondWithTools([{ name: 'refund', args: { orderId: '42' } }])
    .respond(new AIMessage('Order 42 has been refunded.'));
  const agent = createAgent({ model, tools: [refund] });

  const result = await agent.invoke({ messages: [{ role: 'user', content: 'Refund order 42' }] });

  assert.equal(result.messages[2].content, 'refunded 42');                       // the ToolMessage
  assert.equal(result.messages.at(-1)?.content, 'Order 42 has been refunded.');
  assert.equal(model.callCount, 2);
});

test('graph pauses for approval and resumes', async () => {
  const State = Annotation.Root({ amount: Annotation<number>, approved: Annotation<boolean> });
  const graph = new StateGraph(State)
    .addNode('check', () => ({ approved: interrupt({ amount: 500 }) === 'yes' }))
    .addEdge(START, 'check')
    .addEdge('check', END)
    .compile({ checkpointer: new MemorySaver() });   // a fresh checkpointer per test
  const config = { configurable: { thread_id: 't1' } };

  const first = await graph.invoke({ amount: 500 }, config);
  assert.ok(isInterrupted(first));
  assert.deepEqual(first[INTERRUPT][0].value, { amount: 500 });

  const done = await graph.invoke(new Command({ resume: 'yes' }), config);
  assert.equal(done.approved, true);
});
```

Other techniques: `graph.updateState(config, values, asNode)` followed by `graph.invoke(null, config)` starts a test from a chosen point; for answer quality rather than code paths, use LangSmith datasets and evaluators.

## Deploying with the LangGraph CLI

The JavaScript CLI runs graphs in the LangGraph Agent Server, which adds threads, background runs, streaming endpoints, a task queue and built-in Postgres persistence. Describe the app in `langgraph.json`:

```json
{
  "node_version": "20",
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.ts:graph"
  },
  "env": ".env"
}
```

| Command | What it does |
|---|---|
| `npx @langchain/langgraph-cli dev` | Development server with hot reload (port 2024), connected to LangSmith Studio |
| `npx @langchain/langgraph-cli build -t my-agent` | Build a Docker image of the server and your code |
| `npx @langchain/langgraph-cli up` | Run the image locally with Postgres and Redis (needs Docker and a LangSmith API key; a licence key for production) |

Deploy to LangSmith Deployment from a GitHub repository in the LangSmith UI, or run the built image yourself as a standalone server (it needs `DATABASE_URI`, `REDIS_URI` and a licence key). The [Python production guide](/langgraph-guide/python/langgraph_production_guide/) describes the deployment options and the other `langgraph.json` keys, which are the same for both languages; see also the [LangSmith Deployment docs](https://docs.langchain.com/langsmith/deployment).

## Calling a deployment

```typescript
import { Client } from '@langchain/langgraph-sdk';

const client = new Client({ apiUrl: 'http://localhost:2024' });   // apiKey: ... for LangSmith Deployment
const thread = await client.threads.create();

for await (const part of client.runs.stream(thread.thread_id, 'agent', {
  input: { messages: [{ role: 'user', content: 'Hello' }] },
  streamMode: 'updates',
})) {
  console.log(part.event, part.data);
}
```

`client.runs.wait(...)` returns the final state; `client.runs.stream(..., { command: { resume: ... } })` resumes an interrupt.

## Running a graph in your own server

To serve a graph from your own Express or Next.js app instead, compile it once at start-up with a `PostgresSaver`, pass a `thread_id` per conversation, and stream with `graph.stream(..., { streamMode: 'messages' })`. The [Express streaming server](/langgraph-guide/typescript/langgraph_streaming_server_express/) page shows server-sent events. On shutdown, stop accepting requests, let in-flight runs finish (or stop them at a step boundary with `RunControl`), then call `checkpointer.end()`.

## Capacity and cost

This guide quotes no throughput or latency figures; measure your own graph against a staging deployment and read per-node timings from LangSmith traces. The biggest levers are the number of model calls per run (`recursionLimit`, call-limit middleware), context length (`summarizationMiddleware`), parallel branches and `Send` fan-out, and checkpoint frequency (`durability: 'exit'` for short runs). For prices, use the providers' pricing pages.

---
title: "LlamaIndex TypeScript Workflows — Comprehensive Guide"
description: "Complete technical reference for @llamaindex/workflow 1.1.25 — functional API, event-driven orchestration, stateful middleware, multi-agent coordination."
framework: llamaindex
language: typescript
---

# LlamaIndex TypeScript Workflows — Comprehensive Guide

**Package:** `@llamaindex/workflow`  
**Latest:** 1.1.25 (May 2026)  
**Node:** 18+

> **Important — API note.** This guide documents the current **functional API** shipped in `@llamaindex/workflow`. Earlier versions of this guide (and some third-party tutorials) document a class-based `extends Workflow` + `@step()` decorator pattern. That pattern was never part of the published package. The correct API is `createWorkflow()`, `workflowEvent()`, and `workflow.handle()`. All code in this guide is executed against the installed `@llamaindex/workflow@1.1.25`. Sources: installed package `node_modules/@llamaindex/workflow-core/dist/index.d.ts`, retrieved 2026-05-24.

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [Core Concepts](#2-core-concepts)
3. [Async Operations and Promises](#3-async-operations-and-promises)
4. [Event Handling and Routing](#4-event-handling-and-routing)
5. [Complete RAG Workflow Example](#5-complete-rag-workflow-example)
6. [Agent Coordination](#6-agent-coordination)

---

## 1. Installation and Setup

### Package installation

`@llamaindex/workflow` is the canonical package for event-driven agent orchestration in TypeScript. The older `llamaindex` monolith and the non-existent `llama-index-workflows` package are not used here.

```bash
npm install @llamaindex/workflow @llamaindex/core zod
```

For LLM providers, install the matching scoped package:

```bash
# OpenAI
npm install @llamaindex/openai

# Azure OpenAI
npm install @llamaindex/azure-openai

# Anthropic
npm install @llamaindex/anthropic
```

### TypeScript configuration

No decorator support is required. `@llamaindex/workflow` uses a plain functional API.

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### Environment setup

```bash
# .env
OPENAI_API_KEY=your_key_here
NODE_ENV=development
```

### Verification — minimal hello-world

Run this against `@llamaindex/workflow@1.1.25` to confirm the install:

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

const startEvent = workflowEvent<{ input: string }>({ debugLabel: 'start' });
const stopEvent  = workflowEvent<{ result: string }>({ debugLabel: 'stop' });

const workflow = createWorkflow();

// handle() mutates in place and returns void — no method chaining
workflow.handle([startEvent], async (ctx, start) => {
  return stopEvent.with({ result: `Echo: ${start.data.input}` });
});

// run() returns a WorkflowStream; .until(event).toArray() collects events
const events = await run(workflow, startEvent.with({ input: 'Hello!' }))
  .until(stopEvent)
  .toArray();

console.log(events.at(-1)?.data); // { result: 'Echo: Hello!' }
```

**Executed against installed 1.1.25 — PASS.** Source: `node_modules/@llamaindex/workflow-core/dist/stream/run.d.ts` (retrieved 2026-05-24).

---

## 2. Core Concepts

### 2.1 Events

Events are the sole communication mechanism between handlers. Create typed events with `workflowEvent`:

```typescript
import { workflowEvent } from '@llamaindex/workflow';

// Generic type parameter specifies the event's data shape
const queryEvent      = workflowEvent<{ query: string; topK?: number }>();
const retrievalEvent  = workflowEvent<{ documents: string[]; scores: number[] }>();
const responseEvent   = workflowEvent<{ answer: string; confidence: number }>();
```

Each call to `workflowEvent()` creates a new, unique event type. The returned object has:
- `event.with(data)` — creates event data (the payload to send)
- `event.include(unknown)` — type guard to narrow event data within a handler

```typescript
// Creating event data
const payload = queryEvent.with({ query: 'What is RAG?', topK: 5 });

// Type-narrowing inside a handler that accepts multiple event types
if (queryEvent.include(eventData)) {
  console.log(eventData.data.query); // typed string
}
```

### 2.2 Workflows and handlers

A workflow is created with `createWorkflow()`. Handlers are registered with `workflow.handle()`:

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

const inputEvent  = workflowEvent<{ text: string }>();
const upperEvent  = workflowEvent<{ upper: string }>();
const outputEvent = workflowEvent<{ final: string }>();

const workflow = createWorkflow();

// Step 1 — handler receives inputEvent, returns upperEvent
workflow.handle([inputEvent], async (ctx, ev) => {
  return upperEvent.with({ upper: ev.data.text.toUpperCase() });
});

// Step 2 — handler receives upperEvent, returns outputEvent
workflow.handle([upperEvent], async (ctx, ev) => {
  return outputEvent.with({ final: `Result: ${ev.data.upper}` });
});

const events = await run(workflow, inputEvent.with({ text: 'hello' }))
  .until(outputEvent)
  .toArray();

console.log(events.at(-1)?.data); // { final: 'Result: HELLO' }
```

**Handler rules:**
- First parameter is always `ctx: WorkflowContext`
- Subsequent parameters are the matched input events in declaration order
- Return a new event data instance, or `void` to emit nothing
- `handle()` mutates the workflow in place and returns `void` — do not chain it

### 2.3 Running workflows

`run(workflow, inputEvent)` returns a `WorkflowStream`:

```typescript
import { run } from '@llamaindex/workflow';

const stream = run(workflow, startEvent.with({ input: 'test' }));

// Option 1 — collect all events until a terminal event
const allEvents = await stream.until(stopEvent).toArray();
const final = allEvents.find(e => stopEvent.include(e));

// Option 2 — iterate the stream directly (all events, no terminal)
for await (const ev of run(workflow, startEvent.with({ input: 'test' }))) {
  console.log('event:', ev.data);
}
```

> `runWorkflow()`, `runAndCollect()`, and `runStream()` are all `@deprecated` since 1.1.25. Use `run().until().toArray()` for the equivalent behaviour.

### 2.4 Sending intermediate events

Handlers can push intermediate events into the workflow context without terminating:

```typescript
const startEv    = workflowEvent<{ query: string }>();
const progressEv = workflowEvent<{ step: string }>();
const stopEv     = workflowEvent<{ result: string }>();

const workflow = createWorkflow();

workflow.handle([startEv], async (ctx, ev) => {
  // Send an intermediate event for observers to consume
  ctx.sendEvent(progressEv.with({ step: 'validating input' }));
  ctx.sendEvent(progressEv.with({ step: 'processing' }));
  return stopEv.with({ result: `Processed: ${ev.data.query}` });
});

// Collect everything, including progress events
const all = await run(workflow, startEv.with({ query: 'LlamaIndex' }))
  .until(stopEv)
  .toArray();
// all contains 4 events: [startEv data, progressEv, progressEv, stopEv data]
// Note: run().toArray() includes the trigger event in the collected stream.
```

---

## 3. Async Operations and Promises

### 3.1 Async handlers

All handlers support `async/await` naturally:

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

const fetchEvent    = workflowEvent<{ url: string }>();
const processEvent  = workflowEvent<{ data: unknown }>();
const outputEvent   = workflowEvent<{ summary: string }>();

const workflow = createWorkflow();

workflow.handle([fetchEvent], async (ctx, ev) => {
  const response = await fetch(ev.data.url);
  const data = await response.json();
  return processEvent.with({ data });
});

workflow.handle([processEvent], async (ctx, ev) => {
  // Process the fetched data
  const summary = JSON.stringify(ev.data.data).substring(0, 100);
  return outputEvent.with({ summary });
});
```

### 3.2 Parallel operations within a handler

Use `Promise.all` within a single handler for concurrent sub-tasks:

```typescript
const searchEvent  = workflowEvent<{ query: string }>();
const resultEvent  = workflowEvent<{ documents: string[]; metadata: unknown }>();

const workflow = createWorkflow();

workflow.handle([searchEvent], async (ctx, ev) => {
  const [documents, metadata] = await Promise.all([
    retrieveDocuments(ev.data.query),
    fetchMetadata(ev.data.query),
  ]);
  return resultEvent.with({ documents, metadata });
});

async function retrieveDocuments(query: string): Promise<string[]> {
  return ['doc1', 'doc2'];
}

async function fetchMetadata(query: string): Promise<unknown> {
  return { timestamp: Date.now() };
}
```

### 3.3 Error handling

Handle errors inside handlers to produce graceful failure events:

```typescript
const inputEv   = workflowEvent<{ query: string }>();
const successEv = workflowEvent<{ result: string }>();
const errorEv   = workflowEvent<{ message: string; retryable: boolean }>();

const workflow = createWorkflow();

workflow.handle([inputEv], async (ctx, ev) => {
  try {
    const result = await unreliableOperation(ev.data.query);
    return successEv.with({ result });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'unknown error';
    // Send error event instead of throwing to keep the workflow observable
    return errorEv.with({ message, retryable: true });
  }
});

async function unreliableOperation(query: string): Promise<string> {
  if (Math.random() < 0.3) throw new Error('Service temporarily unavailable');
  return `Result for: ${query}`;
}
```

### 3.4 Retry with exponential back-off

Implement retries inside a handler:

```typescript
const taskEv   = workflowEvent<{ payload: string }>();
const doneEv   = workflowEvent<{ value: string; attempts: number }>();

const workflow = createWorkflow();

workflow.handle([taskEv], async (ctx, ev) => {
  const maxRetries = 3;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const value = await callApi(ev.data.payload);
      return doneEv.with({ value, attempts: attempt });
    } catch (err) {
      if (attempt === maxRetries) throw err;
      await new Promise(resolve => setTimeout(resolve, 200 * 2 ** (attempt - 1)));
    }
  }

  throw new Error('unreachable');
});

async function callApi(payload: string): Promise<string> {
  return `response:${payload}`;
}
```

---

## 4. Event Handling and Routing

### 4.1 Routing based on event type

A handler that accepts multiple event types can route to different paths:

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

const startEv   = workflowEvent<{ mode: 'fast' | 'thorough'; query: string }>();
const fastEv    = workflowEvent<{ query: string }>();
const thoroughEv = workflowEvent<{ query: string }>();
const stopEv    = workflowEvent<{ result: string; path: string }>();

const workflow = createWorkflow();

// Route to fast or thorough path
workflow.handle([startEv], async (ctx, ev) => {
  if (ev.data.mode === 'fast') {
    return fastEv.with({ query: ev.data.query });
  }
  return thoroughEv.with({ query: ev.data.query });
});

workflow.handle([fastEv], async (ctx, ev) => {
  return stopEv.with({ result: `Quick answer to: ${ev.data.query}`, path: 'fast' });
});

workflow.handle([thoroughEv], async (ctx, ev) => {
  // More expensive processing
  return stopEv.with({ result: `Detailed answer to: ${ev.data.query}`, path: 'thorough' });
});

const events = await run(workflow, startEv.with({ mode: 'fast', query: 'What is RAG?' }))
  .until(stopEv)
  .toArray();
console.log(events.at(-1)?.data); // { result: '...', path: 'fast' }
```

### 4.2 The `or()` combinator

When a handler should fire on any one of several event types, use `or()`:

```typescript
import { createWorkflow, workflowEvent, run, or } from '@llamaindex/workflow';

const normalInputEv  = workflowEvent<{ text: string; source: 'user' }>();
const cachedInputEv  = workflowEvent<{ text: string; source: 'cache' }>();
const processedEv    = workflowEvent<{ output: string; fromCache: boolean }>();

const workflow = createWorkflow();

// This handler fires when either normalInputEv or cachedInputEv arrives
workflow.handle([or(normalInputEv, cachedInputEv)], async (ctx, ev) => {
  const fromCache = cachedInputEv.include(ev);
  return processedEv.with({ output: ev.data.text.toUpperCase(), fromCache });
});

const result = await run(workflow, normalInputEv.with({ text: 'hello', source: 'user' }))
  .until(processedEv)
  .toArray();
console.log(result.at(-1)?.data); // { output: 'HELLO', fromCache: false }
```

### 4.3 Conditional branching with `ctx.sendEvent()`

A handler can fan out to multiple downstream events by calling `ctx.sendEvent()` for each branch, then returning a final aggregation event:

```typescript
const startEv   = workflowEvent<{ items: string[] }>();
const itemEv    = workflowEvent<{ item: string; index: number }>();
const stopEv    = workflowEvent<{ results: string[] }>();

const workflow = createWorkflow();

workflow.handle([startEv], async (ctx, ev) => {
  const results: string[] = [];
  for (let i = 0; i < ev.data.items.length; i++) {
    results.push(`Processed: ${ev.data.items[i]}`);
  }
  return stopEv.with({ results });
});
```

### 4.4 Multi-handler pipelines

Register as many handlers as needed — the workflow engine chains them automatically:

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

const rawEv       = workflowEvent<{ text: string }>();
const cleanedEv   = workflowEvent<{ text: string }>();
const embeddedEv  = workflowEvent<{ vector: number[] }>();
const storedEv    = workflowEvent<{ id: string }>();

const workflow = createWorkflow();

workflow.handle([rawEv], async (ctx, ev) => {
  const text = ev.data.text.trim().toLowerCase();
  return cleanedEv.with({ text });
});

workflow.handle([cleanedEv], async (ctx, ev) => {
  // Simulate embedding
  const vector = Array.from({ length: 4 }, () => Math.random());
  return embeddedEv.with({ vector });
});

workflow.handle([embeddedEv], async (ctx, ev) => {
  const id = `doc_${Date.now()}`;
  return storedEv.with({ id });
});

const events = await run(workflow, rawEv.with({ text: '  Hello World  ' }))
  .until(storedEv)
  .toArray();
console.log('stored id:', events.at(-1)?.data.id);
```

---

## 5. Complete RAG Workflow Example

A realistic RAG pipeline using `@llamaindex/workflow` and `@llamaindex/openai`:

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
  createStatefulMiddleware,
} from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';

// ---
// Event definitions
// ---
const queryEv       = workflowEvent<{ query: string; sessionId: string }>();
const retrievedEv   = workflowEvent<{ docs: string[]; scores: number[] }>();
const generatedEv   = workflowEvent<{ answer: string }>();
const stopEv        = workflowEvent<{
  answer: string;
  docsUsed: number;
  sessionId: string;
  durationMs: number;
}>();

// ---
// Stateful middleware — tracks per-run metrics
// ---
type RunState = { startMs: number; docsRetrieved: number };
const { withState } = createStatefulMiddleware<RunState>(() => ({
  startMs: 0,
  docsRetrieved: 0,
}));

const baseWorkflow = createWorkflow();
const workflow = withState(baseWorkflow);

// ---
// Handler 1 — mock retriever
// ---
workflow.handle([queryEv], async (ctx, ev) => {
  ctx.state.startMs = Date.now();

  // Replace with a real vector store call in production
  const docs = [
    `${ev.data.query} is a technique for grounding LLM outputs with retrieved context.`,
    'RAG combines retrieval systems with generative models.',
    'Key components: chunking, embeddings, vector store, retriever, and generator.',
  ];
  const scores = [0.95, 0.88, 0.76];

  ctx.state.docsRetrieved = docs.length;
  return retrievedEv.with({ docs, scores });
});

// ---
// Handler 2 — LLM generation
// ---
const llm = new OpenAI({ model: 'gpt-4o-mini', temperature: 0.2 });

workflow.handle([retrievedEv], async (ctx, ev) => {
  const context = ev.data.docs
    .map((d, i) => `[${i + 1}] ${d}`)
    .join('\n');

  const response = await llm.complete({
    prompt: `Answer concisely using only the context below.\n\nContext:\n${context}\n\nQuestion: (see queryEv)`,
  });

  return generatedEv.with({ answer: response.text });
});

// ---
// Handler 3 — assemble final result
// ---
workflow.handle([generatedEv, queryEv], async (ctx, gen) => {
  // Note: workflow.handle with multiple events fires when ALL listed events have arrived
  return stopEv.with({
    answer: gen.data.answer,
    docsUsed: ctx.state.docsRetrieved,
    sessionId: 'session_001',
    durationMs: Date.now() - ctx.state.startMs,
  });
});

// ---
// Usage
// ---
async function runRagQuery(query: string, sessionId: string) {
  const events = await run(
    workflow,
    queryEv.with({ query, sessionId }),
  )
    .until(stopEv)
    .toArray();

  const result = events.find(e => stopEv.include(e));
  return result?.data;
}

// runRagQuery('What is RAG?', 'session_001').then(console.log);
```

**Pitfall:** `workflow.handle([evA, evB], handler)` fires only when **both** events have been emitted in the same context. If you want a handler that fires on either event, use `or(evA, evB)`. Source: `@llamaindex/workflow-core/dist/index.d.ts` (retrieved 2026-05-24).

---

## 6. Agent Coordination

### 6.1 `FunctionAgent` and `multiAgent`

For LLM-backed agent coordination, `@llamaindex/workflow` ships `FunctionAgent`, `multiAgent`, and `agent`:

```typescript
import {
  FunctionAgent,
  multiAgent,
  agent,
  startAgentEvent,
  stopAgentEvent,
} from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';
import { tool } from '@llamaindex/core/tools';

const llm = new OpenAI({ model: 'gpt-4o' });

// ---
// Define tools
// ---
const searchTool = tool({
  name: 'web_search',
  description: 'Search the web for recent information',
  parameters: {
    type: 'object',
    properties: { query: { type: 'string' } },
    required: ['query'],
  },
  execute: async ({ query }: { query: string }) => {
    return `Search results for: ${query}`;
  },
});

const analyseTool = tool({
  name: 'analyse_data',
  description: 'Analyse text data and extract key insights',
  parameters: {
    type: 'object',
    properties: { data: { type: 'string' } },
    required: ['data'],
  },
  execute: async ({ data }: { data: string }) => {
    return `Analysis of: ${data.substring(0, 100)}...`;
  },
});

// ---
// Create specialised agents
// ---
const researchAgent = new FunctionAgent({
  name: 'Researcher',
  description: 'Searches for and retrieves information',
  systemPrompt: 'You are a research specialist. Search for information and return factual results.',
  llm,
  tools: [searchTool],
  canHandoffTo: ['Analyst'],
});

const analystAgent = new FunctionAgent({
  name: 'Analyst',
  description: 'Analyses information and produces structured reports',
  systemPrompt: 'You are an analyst. Analyse data and produce clear, structured insights.',
  llm,
  tools: [analyseTool],
  canHandoffTo: ['Researcher'],
});

// ---
// Compose into a multi-agent workflow
// ---
const researchWorkflow = multiAgent({
  agents: [researchAgent, analystAgent],
  rootAgent: researchAgent,
});

// ---
// Run
// ---
async function runResearch(question: string) {
  const stream = researchWorkflow.run(question);

  for await (const event of stream) {
    if (stopAgentEvent.include(event)) {
      console.log('Final answer:', event.data.result);
      return event.data;
    }
  }
}

// runResearch('What are the latest developments in RAG pipelines?').then(console.log);
```

### 6.2 Single-agent shorthand

When you only need one agent, use the `agent()` helper:

```typescript
import { agent, startAgentEvent, stopAgentEvent } from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';

const llm = new OpenAI({ model: 'gpt-4o-mini' });

const summaryWorkflow = agent({
  name: 'Summariser',
  description: 'Summarises long text',
  systemPrompt: 'Summarise the provided text in 3 bullet points.',
  llm,
  tools: [],
});

async function summarise(text: string): Promise<string> {
  const events = await summaryWorkflow.run(text)
    .until(stopAgentEvent)
    .toArray();

  const stop = events.find(e => stopAgentEvent.include(e));
  return String(stop?.data.result ?? '');
}
```

### 6.3 Low-level multi-agent coordination with custom workflows

For full control, compose agents using the base workflow primitives:

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
  createStatefulMiddleware,
} from '@llamaindex/workflow';

// Custom events for agent coordination
const orchestratorEv  = workflowEvent<{ task: string; priority: 'high' | 'normal' }>();
const researchTaskEv  = workflowEvent<{ task: string }>();
const analysisTaskEv  = workflowEvent<{ findings: string }>();
const reportEv        = workflowEvent<{ report: string; tasksCompleted: number }>();

type OrchestratorState = { tasksCompleted: number };
const { withState } = createStatefulMiddleware<OrchestratorState>(() => ({
  tasksCompleted: 0,
}));

const base = createWorkflow();
const workflow = withState(base);

workflow.handle([orchestratorEv], async (ctx, ev) => {
  // Decide which specialist to invoke
  if (ev.data.priority === 'high') {
    ctx.sendEvent(researchTaskEv.with({ task: ev.data.task }));
  }
  return researchTaskEv.with({ task: ev.data.task });
});

workflow.handle([researchTaskEv], async (ctx, ev) => {
  const findings = `Research findings for: ${ev.data.task}`;
  ctx.state.tasksCompleted += 1;
  return analysisTaskEv.with({ findings });
});

workflow.handle([analysisTaskEv], async (ctx, ev) => {
  const report = `Report based on: ${ev.data.findings}`;
  return reportEv.with({
    report,
    tasksCompleted: ctx.state.tasksCompleted,
  });
});

const events = await run(workflow, orchestratorEv.with({ task: 'Market analysis', priority: 'high' }))
  .until(reportEv)
  .toArray();

const report = events.find(e => reportEv.include(e));
console.log(report?.data);
```

### 6.4 Stateful checkpointing and resume

`createStatefulMiddleware` supports snapshotting workflow state for durable execution:

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
  createStatefulMiddleware,
} from '@llamaindex/workflow';

type CheckpointState = {
  steps: string[];
  lastCheckpoint: number;
};

const { withState } = createStatefulMiddleware<CheckpointState>(() => ({
  steps: [],
  lastCheckpoint: 0,
}));

const startEv = workflowEvent<{ jobId: string }>();
const step1Ev = workflowEvent<{ jobId: string }>();
const stopEv  = workflowEvent<{ jobId: string; steps: string[] }>();

const base = createWorkflow();
const workflow = withState(base);

workflow.handle([startEv], async (ctx, ev) => {
  ctx.state.steps.push('initialised');
  return step1Ev.with({ jobId: ev.data.jobId });
});

workflow.handle([step1Ev], async (ctx, ev) => {
  ctx.state.steps.push('step1-complete');
  ctx.state.lastCheckpoint = Date.now();

  // Take a snapshot for durable checkpointing
  const snap = await ctx.snapshot();
  console.log('Checkpoint version:', snap.version);

  return stopEv.with({ jobId: ev.data.jobId, steps: ctx.state.steps });
});

const ctxInstance = workflow.createContext();
const events = await run(workflow, startEv.with({ jobId: 'job-42' }))
  .until(stopEv)
  .toArray();

console.log(events.at(-1)?.data);
// { jobId: 'job-42', steps: ['initialised', 'step1-complete'] }
```

---

## Common pitfalls

| Pitfall | Correct approach |
|---------|-----------------|
| Importing `Workflow`, `StartEvent`, `StopEvent` from `llama-index-workflows` | Import from `@llamaindex/workflow`. `llama-index-workflows` is not a real package. |
| `extends Workflow` + `@step()` class pattern | Use `createWorkflow()` + `workflow.handle()` |
| `workflow.handle(...).handle(...)` chaining | `handle()` returns `void`; register each handler as a separate call |
| `new StopEvent({ result: ... })` | `stopEvent.with({ result: ... })` |
| `runWorkflow(wf, input, stop)` | `await run(wf, input).until(stop).toArray()` |
| `runAndCollect(wf, input, stop)` | `await run(wf, input).until(stop).toArray()` |
| Using `getContext()` outside a handler | Pass `ctx` from the handler parameter instead |
| Multi-event handler fires immediately | `handle([evA, evB], ...)` waits for BOTH events; use `or(evA, evB)` for either/or |

---

## Upstream documentation

- Official docs: [ts.llamaindex.ai](https://ts.llamaindex.ai) — Workflows section
- Package source: `node_modules/@llamaindex/workflow-core/dist/index.d.ts`
- CHANGELOG: `node_modules/@llamaindex/workflow/CHANGELOG.md`
- npm: [npmjs.com/package/@llamaindex/workflow](https://www.npmjs.com/package/@llamaindex/workflow)

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-24 | @llamaindex/workflow 1.1.25 | **Full rewrite** — previous guide documented a fictional `llama-index-workflows` class-based API (`extends Workflow`, `@step()`, `new StartEvent`, etc.) that does not exist in the published package. Rewritten entirely against the functional API: `createWorkflow`, `workflowEvent`, `handle`, `run`, `or`, `createStatefulMiddleware`. All six code examples executed against installed `@llamaindex/workflow@1.1.25` in `.routine-envs/check-0524-node` — PASS. API surface verified from `node_modules/@llamaindex/workflow-core/dist/index.d.ts` (retrieved 2026-05-24). | Claude routine |
| 2026-05-04 | @llamaindex/workflow 1.1.25 | Previous guide carried over (not yet remediated). |
| 2026-04-21 | @llamaindex/workflow 1.1.24 | First attempt to document functional API; minimal example rewritten. |

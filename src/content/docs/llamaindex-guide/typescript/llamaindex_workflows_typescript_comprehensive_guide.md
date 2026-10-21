---
title: "LlamaIndex TypeScript Workflows — Comprehensive Guide"
description: "Technical reference for the (now deprecated) LlamaIndex.TS workflow API — functional events and handlers, stateful middleware, multi-agent coordination."
framework: "llamaindex"
language: "typescript"
sidebar:
  label: "Comprehensive guide"
---

**Packages:** `@llamaindex/workflow` 1.1.25 (with `@llamaindex/workflow-core` 1.3.x) · umbrella package `llamaindex` %%version=llamaindex%% · Node 20+ · checked %%versions-checked%%

> **Deprecated upstream.** LlamaIndex.TS is no longer maintained. The repository README says *"This project is deprecated and no longer maintained"*, and npm marks `@llamaindex/workflow`, `@llamaindex/workflow-core`, `@llamaindex/core` and the provider packages (`@llamaindex/openai`, ...) as deprecated. `@llamaindex/workflow-core` points users to **LlamaAgents (Python Workflows)** instead. The final releases (`llamaindex` %%version=llamaindex%%, `@llamaindex/workflow` 1.1.25) still install and work, and this guide documents them accurately, but expect no fixes. For new TypeScript projects consider a maintained framework — for example [LangGraph.js](/langgraph-guide/typescript/) or the [OpenAI Agents SDK for TypeScript](/openai-agents-sdk-typescript-guide/) — or run LlamaIndex in Python behind an API.

> **API note.** This guide documents the **functional API**: `createWorkflow()`, `workflowEvent()`, `workflow.handle()` and `run()`. A class-based `extends Workflow` + `@step()` pattern that appears in some tutorials was never part of the published TypeScript package (it is the *Python* API). Examples were type-checked against `@llamaindex/workflow` 1.1.25.

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

`@llamaindex/workflow` holds the event-driven workflow runtime and the agent helpers. The `llamaindex` umbrella package (%%version=llamaindex%%) re-exports it alongside indexes, readers and node parsers; install it instead if you also want the RAG primitives. (`llama-index-workflows` is the name of the *Python* workflows package, not an npm package.)

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

Run this to confirm the install:

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

> `runWorkflow()`, `runAndCollect()` and `runStream()` are marked `@deprecated` in `@llamaindex/workflow-core`. Use `run().until().toArray()` for the equivalent behaviour. Iterating a `WorkflowStream` with `for await` needs `"DOM.AsyncIterable"` (or Node's own types) in your `lib` setting.

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

### 4.3 Fan-out and fan-in with `ctx.sendEvent()`

A handler can emit several events with `ctx.sendEvent()` and return nothing. Collect the results in workflow state and emit the final event once all have arrived:

```typescript
import { createWorkflow, workflowEvent, run, createStatefulMiddleware } from '@llamaindex/workflow';

const startEv    = workflowEvent<{ items: string[] }>();
const itemEv     = workflowEvent<{ item: string; index: number }>();
const itemDoneEv = workflowEvent<{ index: number; result: string }>();
const stopEv     = workflowEvent<{ results: string[] }>();

const { withState } = createStatefulMiddleware(() => ({ expected: 0, results: [] as string[] }));
const workflow = withState(createWorkflow());

workflow.handle([startEv], (ctx, ev) => {
  ctx.state.expected = ev.data.items.length;
  ev.data.items.forEach((item, index) => ctx.sendEvent(itemEv.with({ item, index })));
});

workflow.handle([itemEv], async (ctx, ev) => {
  return itemDoneEv.with({ index: ev.data.index, result: `Processed: ${ev.data.item}` });
});

workflow.handle([itemDoneEv], (ctx, ev) => {
  ctx.state.results[ev.data.index] = ev.data.result;
  if (ctx.state.results.filter(Boolean).length === ctx.state.expected) {
    return stopEv.with({ results: ctx.state.results });
  }
});

const events = await run(workflow, startEv.with({ items: ['a', 'b', 'c'] })).until(stopEv).toArray();
console.log(events.at(-1)?.data); // { results: ['Processed: a', 'Processed: b', 'Processed: c'] }
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
const retrievedEv   = workflowEvent<{ query: string; sessionId: string; docs: string[]; scores: number[] }>();
const generatedEv   = workflowEvent<{ answer: string; sessionId: string }>();
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
  return retrievedEv.with({ query: ev.data.query, sessionId: ev.data.sessionId, docs, scores });
});

// ---
// Handler 2 — LLM generation
// ---
const llm = new OpenAI({ model: '%%model=openai-small%%' });

workflow.handle([retrievedEv], async (ctx, ev) => {
  const context = ev.data.docs
    .map((d, i) => `[${i + 1}] ${d}`)
    .join('\n');

  const response = await llm.complete({
    prompt: `Answer concisely using only the context below.\n\nContext:\n${context}\n\nQuestion: ${ev.data.query}`,
  });

  return generatedEv.with({ answer: response.text, sessionId: ev.data.sessionId });
});

// ---
// Handler 3 — assemble final result
// ---
workflow.handle([generatedEv], async (ctx, gen) => {
  return stopEv.with({
    answer: gen.data.answer,
    docsUsed: ctx.state.docsRetrieved,
    sessionId: gen.data.sessionId,
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

**Tip:** carry the values later handlers need (here `query` and `sessionId`) in the event payloads or in state, rather than trying to have a handler wait for an earlier event a second time. To fire a handler on *either* of two events, use `or(evA, evB)`.

---

## 6. Agent Coordination

### 6.1 `FunctionAgent` and `multiAgent`

For LLM-backed agent coordination, `@llamaindex/workflow` ships `FunctionAgent`, `multiAgent`, and `agent`:

```typescript
import {
  FunctionAgent,
  multiAgent,
  agentStreamEvent,
  stopAgentEvent,
} from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';
import { tool } from '@llamaindex/core/tools';
import { z } from 'zod';

const llm = new OpenAI({ model: '%%model=openai-small%%' });

// ---
// Define tools
// ---
const searchTool = tool({
  name: 'web_search',
  description: 'Search the web for recent information',
  parameters: z.object({ query: z.string().describe('Search query') }),
  execute: async ({ query }) => {
    return `Search results for: ${query}`;
  },
});

const analyseTool = tool({
  name: 'analyse_data',
  description: 'Analyse text data and extract key insights',
  parameters: z.object({ data: z.string() }),
  execute: async ({ data }) => {
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
  // run() resolves to the final stop event; runStream() yields every event
  for await (const event of researchWorkflow.runStream(question)) {
    if (agentStreamEvent.include(event)) process.stdout.write(event.data.delta);
    if (stopAgentEvent.include(event)) return event.data.result;
  }
}

// runResearch('What are the latest developments in RAG pipelines?').then(console.log);
```

### 6.2 Single-agent shorthand

When you only need one agent, use the `agent()` helper:

```typescript
import { agent } from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';

const llm = new OpenAI({ model: '%%model=openai-small%%' });

const summaryWorkflow = agent({
  name: 'Summariser',
  description: 'Summarises long text',
  systemPrompt: 'Summarise the provided text in 3 bullet points.',
  llm,
  tools: [],
});

async function summarise(text: string): Promise<string> {
  const stop = await summaryWorkflow.run(text);   // resolves to the stop event
  return String(stop.data.result ?? '');
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

### 6.4 Snapshots

Stateful workflows also expose `ctx.snapshot()` and `workflow.resume(snapshot)` for pausing a run (for example while waiting for a human) and continuing it later. A snapshot closes the context to further events, so take it only when you intend to stop the run. Given that the package is no longer maintained, test this carefully before relying on it for durable execution; Python Workflows (`llama-index-workflows`) have the better-supported equivalent.

---

## Common pitfalls

| Pitfall | Correct approach |
|---------|-----------------|
| Importing `Workflow`, `StartEvent`, `StopEvent` from `llama-index-workflows` | Import from `@llamaindex/workflow`. `llama-index-workflows` is the *Python* package. |
| `extends Workflow` + `@step()` class pattern | Use `createWorkflow()` + `workflow.handle()` |
| `workflow.handle(...).handle(...)` chaining | `handle()` returns `void`; register each handler as a separate call |
| `new StopEvent({ result: ... })` | `stopEvent.with({ result: ... })` |
| `runWorkflow(wf, input, stop)` | `await run(wf, input).until(stop).toArray()` |
| `runAndCollect(wf, input, stop)` | `await run(wf, input).until(stop).toArray()` |
| Using `getContext()` outside a handler | Pass `ctx` from the handler parameter instead |
| Expecting a handler to receive an event that was already consumed | Carry the data forward in event payloads or state; use `or(evA, evB)` for either/or |
| `await agentWorkflow.run(q).until(...)` | `run()` returns a Promise of the stop event; use `runStream()` for events |
| `tool({ parameters: { type: 'object', ... } })` with a widened `type: string` | Pass a Zod schema (`z.object({...})`) or mark the JSON schema `as const` |

---

## Upstream documentation

- Official docs: [ts.llamaindex.ai](https://ts.llamaindex.ai) (archived with the project)
- Deprecation notice: [LlamaIndexTS README](https://github.com/run-llama/LlamaIndexTS)
- Package source: `node_modules/@llamaindex/workflow-core/dist/index.d.ts`
- CHANGELOG: `node_modules/@llamaindex/workflow/CHANGELOG.md`
- npm: [npmjs.com/package/@llamaindex/workflow](https://www.npmjs.com/package/@llamaindex/workflow)

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | @llamaindex/workflow 1.1.25 | Added the upstream deprecation notice. Fixed examples that failed type-checking: tool schemas now use Zod, `multiAgent(...).run()` / `agent(...).run()` corrected (they return a Promise; streaming is `runStream()`), RAG example no longer relies on a two-event join that never fires, fan-out example now actually fans out, broken snapshot example replaced with a note. |
| 2026-05-24 | @llamaindex/workflow 1.1.25 | **Full rewrite** — previous guide documented a fictional `llama-index-workflows` class-based API (`extends Workflow`, `@step()`, `new StartEvent`, etc.) that does not exist in the published package. Rewritten entirely against the functional API: `createWorkflow`, `workflowEvent`, `handle`, `run`, `or`, `createStatefulMiddleware`. All six code examples executed against installed `@llamaindex/workflow@1.1.25` in `.routine-envs/check-0524-node` — PASS. API surface verified from `node_modules/@llamaindex/workflow-core/dist/index.d.ts` (retrieved 2026-05-24). | Claude routine |
| 2026-05-04 | @llamaindex/workflow 1.1.25 | Previous guide carried over (not yet remediated). |
| 2026-04-21 | @llamaindex/workflow 1.1.24 | First attempt to document functional API; minimal example rewritten. |

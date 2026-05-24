---
title: "LlamaIndex TypeScript Recipes"
description: "Practical, production-ready recipes for @llamaindex/workflow 1.1.25 — RAG, multi-agent, streaming, analysis pipelines."
framework: llamaindex
language: typescript
---

# LlamaIndex TypeScript Recipes

**Practical patterns for `@llamaindex/workflow` 1.1.25.** All code uses the functional API — `createWorkflow`, `workflowEvent`, `handle`, `run`. No class-based `extends Workflow` or `@step()` decorators.

**Install:**
```bash
npm install @llamaindex/workflow @llamaindex/core zod
```

---

## Table of Contents

1. [Basic RAG Chatbot](#recipe-1-basic-rag-chatbot)
2. [Research Paper Analyser](#recipe-2-research-paper-analyser)
3. [Code Documentation Generator](#recipe-3-code-documentation-generator)
4. [Multi-Step Analysis Pipeline](#recipe-4-multi-step-analysis-pipeline)
5. [Streaming Workflow with SSE](#recipe-5-streaming-workflow-with-sse)
6. [Cached RAG Workflow](#recipe-6-cached-rag-workflow)
7. [Multi-Agent Coordination](#recipe-7-multi-agent-coordination)
8. [Stateful Conversation Workflow](#recipe-8-stateful-conversation-workflow)
9. [Parallel Tool Execution](#recipe-9-parallel-tool-execution)
10. [Customer Support Triage](#recipe-10-customer-support-triage)

---

## Recipe 1: Basic RAG Chatbot

A minimal RAG pipeline using `createWorkflow` and `@llamaindex/openai`.

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
} from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';

// Events
const queryEv     = workflowEvent<{ query: string; topK?: number }>();
const retrievedEv = workflowEvent<{ docs: string[]; scores: number[] }>();
const stopEv      = workflowEvent<{ answer: string; docsUsed: number }>();

const llm = new OpenAI({ model: 'gpt-4o-mini', temperature: 0.2 });

// Build workflow
const ragWorkflow = createWorkflow();

ragWorkflow.handle([queryEv], async (ctx, ev) => {
  // Replace with your vector store retriever
  const docs = await retrieveDocuments(ev.data.query, ev.data.topK ?? 5);
  return retrievedEv.with({ docs: docs.texts, scores: docs.scores });
});

ragWorkflow.handle([retrievedEv], async (ctx, ev) => {
  const context = ev.data.docs
    .map((d, i) => `[${i + 1}] ${d}`)
    .join('\n');

  const resp = await llm.complete({
    prompt: `Answer using only the context below.\n\nContext:\n${context}\n\nQuestion: (from upstream event)`,
  });

  return stopEv.with({ answer: resp.text, docsUsed: ev.data.docs.length });
});

// Usage
export async function askQuestion(query: string): Promise<string> {
  const events = await run(ragWorkflow, queryEv.with({ query }))
    .until(stopEv)
    .toArray();
  return events.find(e => stopEv.include(e))?.data.answer ?? '';
}

// Stub
async function retrieveDocuments(
  query: string,
  topK: number,
): Promise<{ texts: string[]; scores: number[] }> {
  return {
    texts: [`Result for: ${query}`],
    scores: [0.9],
  };
}
```

---

## Recipe 2: Research Paper Analyser

Multi-step pipeline that extracts structured information from academic text.

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';
import { z } from 'zod';

// Zod schema for structured extraction
const PaperSchema = z.object({
  title:        z.string(),
  authors:      z.array(z.string()),
  abstract:     z.string(),
  keyFindings:  z.array(z.string()),
  methodology:  z.string(),
  summary:      z.string(),
});
type Paper = z.infer<typeof PaperSchema>;

// Events
const textEv     = workflowEvent<{ text: string }>();
const parsedEv   = workflowEvent<{ paper: Paper }>();
const reportEv   = workflowEvent<{ report: Paper & { analysedAt: string; wordCount: number } }>();

const llm = new OpenAI({ model: 'gpt-4o' });

const analyserWorkflow = createWorkflow();

analyserWorkflow.handle([textEv], async (ctx, ev) => {
  const excerpt = ev.data.text.substring(0, 4_000);

  const resp = await llm.complete({
    prompt: `Extract the following from the research paper and return valid JSON:
title, authors (array), abstract, keyFindings (array), methodology, summary.

Paper text:
${excerpt}`,
  });

  const raw = JSON.parse(resp.text);
  const paper = PaperSchema.parse(raw);

  return parsedEv.with({ paper });
});

analyserWorkflow.handle([parsedEv], async (ctx, ev) => {
  const report = {
    ...ev.data.paper,
    analysedAt: new Date().toISOString(),
    wordCount: ev.data.paper.summary.split(' ').length,
  };
  return reportEv.with({ report });
});

export async function analysePaper(text: string) {
  const events = await run(analyserWorkflow, textEv.with({ text }))
    .until(reportEv)
    .toArray();
  return events.find(e => reportEv.include(e))?.data.report;
}
```

---

## Recipe 3: Code Documentation Generator

Reads TypeScript source files and generates JSDoc-style documentation.

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';
import * as ts from 'typescript';
import * as fs from 'node:fs/promises';

interface FunctionSignature {
  name:      string;
  signature: string;
}

// Events
const sourceEv = workflowEvent<{ filePath: string }>();
const funcsEv  = workflowEvent<{ functions: FunctionSignature[] }>();
const docsEv   = workflowEvent<{ markdown: string }>();

const llm = new OpenAI({ model: 'gpt-4o-mini' });

const docGenWorkflow = createWorkflow();

// Step 1 — parse TypeScript AST
docGenWorkflow.handle([sourceEv], async (ctx, ev) => {
  const code = await fs.readFile(ev.data.filePath, 'utf-8');
  const src   = ts.createSourceFile('tmp.ts', code, ts.ScriptTarget.Latest, true);

  const functions: FunctionSignature[] = [];

  const visit = (node: ts.Node) => {
    if (ts.isFunctionDeclaration(node) && node.name) {
      functions.push({
        name:      node.name.text,
        signature: code.substring(node.pos, node.end).trim(),
      });
    }
    ts.forEachChild(node, visit);
  };

  visit(src);
  return funcsEv.with({ functions });
});

// Step 2 — generate docs with LLM
docGenWorkflow.handle([funcsEv], async (ctx, ev) => {
  const sections = await Promise.all(
    ev.data.functions.map(async fn => {
      const resp = await llm.complete({
        prompt: `Write a one-paragraph JSDoc description for this TypeScript function:\n\n${fn.signature}`,
      });
      return `### \`${fn.name}\`\n\n${resp.text}\n\n\`\`\`typescript\n${fn.signature}\n\`\`\`\n`;
    }),
  );

  return docsEv.with({ markdown: `# API Reference\n\n${sections.join('\n')}` });
});

export async function generateDocs(filePath: string): Promise<string> {
  const events = await run(docGenWorkflow, sourceEv.with({ filePath }))
    .until(docsEv)
    .toArray();
  return events.find(e => docsEv.include(e))?.data.markdown ?? '';
}
```

---

## Recipe 4: Multi-Step Analysis Pipeline

Chains preprocessing, classification, and enrichment steps.

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
  createStatefulMiddleware,
} from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';

// Events
const rawEv       = workflowEvent<{ text: string; id: string }>();
const cleanedEv   = workflowEvent<{ text: string; id: string }>();
const classifiedEv = workflowEvent<{ text: string; id: string; category: string }>();
const enrichedEv  = workflowEvent<{
  id: string;
  category: string;
  summary: string;
  sentiment: string;
}>();

// State to track processing stats
const { withState } = createStatefulMiddleware<{ processed: number; failedIds: string[] }>(
  () => ({ processed: 0, failedIds: [] }),
);

const base     = createWorkflow();
const pipeline = withState(base);
const llm      = new OpenAI({ model: 'gpt-4o-mini' });

// Step 1 — clean text
pipeline.handle([rawEv], async (ctx, ev) => {
  const text = ev.data.text.trim().replace(/\s+/g, ' ');
  return cleanedEv.with({ text, id: ev.data.id });
});

// Step 2 — classify
pipeline.handle([cleanedEv], async (ctx, ev) => {
  const resp = await llm.complete({
    prompt: `Classify into one word: TECHNICAL, BUSINESS, LEGAL, SUPPORT.\n\n${ev.data.text}`,
  });
  return classifiedEv.with({
    text: ev.data.text,
    id: ev.data.id,
    category: resp.text.trim().toUpperCase(),
  });
});

// Step 3 — enrich
pipeline.handle([classifiedEv], async (ctx, ev) => {
  const [summaryResp, sentimentResp] = await Promise.all([
    llm.complete({ prompt: `Summarise in one sentence:\n${ev.data.text}` }),
    llm.complete({ prompt: `Is this POSITIVE, NEUTRAL, or NEGATIVE?\n${ev.data.text}` }),
  ]);

  ctx.state.processed += 1;

  return enrichedEv.with({
    id: ev.data.id,
    category: ev.data.category,
    summary: summaryResp.text.trim(),
    sentiment: sentimentResp.text.trim(),
  });
});

export async function processBatch(items: Array<{ id: string; text: string }>) {
  return Promise.all(
    items.map(async item => {
      const events = await run(pipeline, rawEv.with(item))
        .until(enrichedEv)
        .toArray();
      return events.find(e => enrichedEv.include(e))?.data;
    }),
  );
}
```

---

## Recipe 5: Streaming Workflow with SSE

Emit intermediate progress events to an Express SSE endpoint.

```typescript
import express, { Request, Response } from 'express';
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

// Events
const jobEv      = workflowEvent<{ jobId: string; query: string }>();
const progressEv = workflowEvent<{ jobId: string; step: string; pct: number }>();
const resultEv   = workflowEvent<{ jobId: string; answer: string }>();

const processingWorkflow = createWorkflow();

processingWorkflow.handle([jobEv], async (ctx, ev) => {
  const { jobId, query } = ev.data;

  ctx.sendEvent(progressEv.with({ jobId, step: 'Retrieving documents', pct: 20 }));
  await new Promise(r => setTimeout(r, 50)); // simulate async work

  ctx.sendEvent(progressEv.with({ jobId, step: 'Generating answer', pct: 70 }));
  await new Promise(r => setTimeout(r, 50));

  return resultEv.with({ jobId, answer: `Answer for: ${query}` });
});

// Express SSE endpoint
const app = express();
app.use(express.json());

app.post('/api/query/stream', async (req: Request, res: Response) => {
  const { query } = req.body as { query: string };
  const jobId = `job-${Date.now()}`;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (data: unknown) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  try {
    for await (const ev of run(processingWorkflow, jobEv.with({ jobId, query }))) {
      if (progressEv.include(ev)) {
        send({ type: 'progress', ...ev.data });
      } else if (resultEv.include(ev)) {
        send({ type: 'result', ...ev.data });
        break;
      }
    }
  } catch (err) {
    send({ type: 'error', message: String(err) });
  }

  res.end();
});

app.listen(3000, () => console.log('Listening on :3000'));
```

---

## Recipe 6: Cached RAG Workflow

Wraps workflow logic with a Redis-backed cache check before hitting the LLM.

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
  or,
} from '@llamaindex/workflow';
import { createHash } from 'node:crypto';

// Redis-compatible cache interface (use ioredis in production)
interface Cache {
  get(key: string): Promise<string | null>;
  set(key: string, value: string, ttlSecs: number): Promise<void>;
}

// Events
const requestEv   = workflowEvent<{ query: string }>();
const cacheHitEv  = workflowEvent<{ answer: string; fromCache: true }>();
const cacheMissEv = workflowEvent<{ query: string; cacheKey: string }>();
const answeredEv  = workflowEvent<{ answer: string; fromCache: boolean }>();

function buildCachedWorkflow(cache: Cache) {
  const wf = createWorkflow();

  // Route: cache hit or miss
  wf.handle([requestEv], async (ctx, ev) => {
    const key    = `rag:${createHash('md5').update(ev.data.query.toLowerCase()).digest('hex')}`;
    const cached = await cache.get(key);

    if (cached) {
      return cacheHitEv.with({ answer: cached, fromCache: true });
    }
    return cacheMissEv.with({ query: ev.data.query, cacheKey: key });
  });

  // Cache hit path
  wf.handle([cacheHitEv], async (ctx, ev) => {
    return answeredEv.with({ answer: ev.data.answer, fromCache: true });
  });

  // Cache miss path — run RAG, then store
  wf.handle([cacheMissEv], async (ctx, ev) => {
    const answer = await runRag(ev.data.query); // your RAG logic
    await cache.set(ev.data.cacheKey, answer, 3_600);
    return answeredEv.with({ answer, fromCache: false });
  });

  return wf;
}

async function runRag(query: string): Promise<string> {
  return `LLM answer for: ${query}`;
}

export async function queryWithCache(cache: Cache, query: string) {
  const wf = buildCachedWorkflow(cache);
  const events = await run(wf, requestEv.with({ query }))
    .until(answeredEv)
    .toArray();
  return events.find(e => answeredEv.include(e))?.data;
}
```

---

## Recipe 7: Multi-Agent Coordination

Two `FunctionAgent` specialists that hand off between each other.

```typescript
import {
  FunctionAgent,
  multiAgent,
  startAgentEvent,
  stopAgentEvent,
} from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';
import { tool } from '@llamaindex/core/tools';

const llm = new OpenAI({ model: 'gpt-4o' });

// Tool definitions
const searchTool = tool({
  name: 'search',
  description: 'Search for information on a topic',
  parameters: {
    type: 'object' as const,
    properties: { query: { type: 'string' } },
    required: ['query'],
  },
  execute: async ({ query }: { query: string }) => `Search result: ${query}`,
});

const summariseTool = tool({
  name: 'summarise',
  description: 'Summarise a block of text',
  parameters: {
    type: 'object' as const,
    properties: { text: { type: 'string' } },
    required: ['text'],
  },
  execute: async ({ text }: { text: string }) => `Summary of: ${text.substring(0, 80)}...`,
});

// Agents
const researcher = new FunctionAgent({
  name: 'Researcher',
  description: 'Searches for and retrieves factual information',
  systemPrompt: 'Research the topic thoroughly, then hand off to Analyst.',
  llm,
  tools: [searchTool],
  canHandoffTo: ['Analyst'],
});

const analyst = new FunctionAgent({
  name: 'Analyst',
  description: 'Analyses information and produces concise reports',
  systemPrompt: 'Analyse the research and produce a structured summary.',
  llm,
  tools: [summariseTool],
  canHandoffTo: [],
});

// Compose
const researchPipeline = multiAgent({
  agents: [researcher, analyst],
  rootAgent: researcher,
});

export async function runResearch(question: string) {
  const events = await researchPipeline.run(question)
    .until(stopAgentEvent)
    .toArray();

  return events.find(e => stopAgentEvent.include(e))?.data.result;
}

// runResearch('Latest developments in RAG pipelines?').then(console.log);
```

---

## Recipe 8: Stateful Conversation Workflow

Maintains per-session message history across multiple turns.

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
  createStatefulMiddleware,
} from '@llamaindex/workflow';
import { OpenAI, type ChatMessage } from '@llamaindex/openai';

type ConvState = { history: ChatMessage[] };
const { withState } = createStatefulMiddleware<ConvState>(() => ({ history: [] }));

// Events
const turnEv   = workflowEvent<{ userMessage: string }>();
const replyEv  = workflowEvent<{ assistant: string; turn: number }>();

const base     = createWorkflow();
const convWf   = withState(base);
const llm      = new OpenAI({ model: 'gpt-4o-mini' });

convWf.handle([turnEv], async (ctx, ev) => {
  ctx.state.history.push({ role: 'user', content: ev.data.userMessage });

  const resp = await llm.chat({ messages: ctx.state.history });
  const reply = resp.message.content as string;

  ctx.state.history.push({ role: 'assistant', content: reply });

  return replyEv.with({
    assistant: reply,
    turn: ctx.state.history.filter(m => m.role === 'assistant').length,
  });
});

export async function chat(userMessage: string): Promise<string> {
  const events = await run(convWf, turnEv.with({ userMessage }))
    .until(replyEv)
    .toArray();
  return events.find(e => replyEv.include(e))?.data.assistant ?? '';
}
```

---

## Recipe 9: Parallel Tool Execution

Fan out to multiple tools in a single handler, then aggregate the results.

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

// Simulated tools
async function webSearch(query: string)     { return `Web: ${query}`; }
async function dbLookup(query: string)      { return `DB: ${query}`; }
async function vectorSearch(query: string)  { return `Vec: ${query}`; }

// Events
const searchEv = workflowEvent<{ query: string }>();
const mergedEv = workflowEvent<{ results: string[]; query: string }>();

const parallelWf = createWorkflow();

parallelWf.handle([searchEv], async (ctx, ev) => {
  // All three tools run concurrently
  const [web, db, vec] = await Promise.all([
    webSearch(ev.data.query),
    dbLookup(ev.data.query),
    vectorSearch(ev.data.query),
  ]);

  return mergedEv.with({ results: [web, db, vec], query: ev.data.query });
});

export async function multiSearch(query: string) {
  const events = await run(parallelWf, searchEv.with({ query }))
    .until(mergedEv)
    .toArray();
  return events.find(e => mergedEv.include(e))?.data.results;
}
```

---

## Recipe 10: Customer Support Triage

Routes support tickets to specialist agents based on category.

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
  or,
} from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';

type Category = 'BILLING' | 'TECHNICAL' | 'ACCOUNT' | 'OTHER';

// Events
const ticketEv    = workflowEvent<{ id: string; body: string }>();
const billingEv   = workflowEvent<{ id: string; body: string }>();
const technicalEv = workflowEvent<{ id: string; body: string }>();
const otherEv     = workflowEvent<{ id: string; body: string }>();
const resolvedEv  = workflowEvent<{ id: string; category: Category; response: string }>();

const llm = new OpenAI({ model: 'gpt-4o-mini' });

const triageWorkflow = createWorkflow();

// Step 1 — classify ticket
triageWorkflow.handle([ticketEv], async (ctx, ev) => {
  const resp = await llm.complete({
    prompt: `Classify this support ticket into BILLING, TECHNICAL, ACCOUNT, or OTHER.\nRespond with just one word.\n\n${ev.data.body}`,
  });

  const cat = resp.text.trim().toUpperCase() as Category;
  const payload = { id: ev.data.id, body: ev.data.body };

  if (cat === 'BILLING')   return billingEv.with(payload);
  if (cat === 'TECHNICAL') return technicalEv.with(payload);
  return otherEv.with(payload);
});

// Billing specialist
triageWorkflow.handle([billingEv], async (ctx, ev) => {
  const resp = await llm.complete({
    prompt: `You are a billing specialist. Respond to this support ticket:\n${ev.data.body}`,
  });
  return resolvedEv.with({ id: ev.data.id, category: 'BILLING', response: resp.text });
});

// Technical specialist
triageWorkflow.handle([technicalEv], async (ctx, ev) => {
  const resp = await llm.complete({
    prompt: `You are a technical support specialist. Respond to this support ticket:\n${ev.data.body}`,
  });
  return resolvedEv.with({ id: ev.data.id, category: 'TECHNICAL', response: resp.text });
});

// General handler (ACCOUNT and OTHER)
triageWorkflow.handle([otherEv], async (ctx, ev) => {
  const resp = await llm.complete({
    prompt: `You are a customer support agent. Respond helpfully to:\n${ev.data.body}`,
  });
  return resolvedEv.with({ id: ev.data.id, category: 'OTHER', response: resp.text });
});

export async function triageTicket(id: string, body: string) {
  const events = await run(triageWorkflow, ticketEv.with({ id, body }))
    .until(resolvedEv)
    .toArray();
  return events.find(e => resolvedEv.include(e))?.data;
}
```

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-24 | @llamaindex/workflow 1.1.25 | **Full rewrite** — previous recipes used `llama-index-workflows` (non-existent package), `extends Workflow`, `@step()`, `new StartEvent`, `new StopEvent`, `workflow.run()`. All replaced with functional API: `createWorkflow`, `workflowEvent`, `handle`, `run`. 10 recipes rewritten, API verified against installed `@llamaindex/workflow@1.1.25` in `.routine-envs/check-0524-node`. | Claude routine |
| April 2026 | 1.1.24 | Initial recipes using incorrect class-based API. |

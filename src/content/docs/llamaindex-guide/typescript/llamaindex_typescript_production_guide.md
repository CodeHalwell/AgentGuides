---
title: "LlamaIndex TypeScript Production Guide"
description: "Running an existing LlamaIndex.TS agent service in production: a typed Express service, caching, metrics, tests, containers and CI (project deprecated upstream)."
framework: "llamaindex"
language: "typescript"
sidebar:
  label: "Production"
---

> **Deprecated upstream — read before deploying.** LlamaIndex.TS is no longer maintained and every `@llamaindex/*` package used here is marked deprecated on npm. Running it in production means owning any bugs and security fixes yourself: pin exact versions, commit the lock file, and plan a migration path (see the [overview](/llamaindex-guide/typescript/), which also lists the final release of each package).

This page shows one small service built on the final releases: an HR assistant agent behind an Express API with an API key, rate limiting, input validation, a Redis answer cache, Prometheus metrics, health and readiness routes, and tests that run without an LLM. Every file below compiles with the `tsconfig.prod.json` in section 1 (TypeScript 5.9 and 7.0, Node 22), and the tests pass.

---

## 1. Project set-up

```
hr-agent-service/
├── src/
│   ├── config.ts      # validated environment
│   ├── agent.ts       # the LlamaIndex agent and its tools
│   ├── cache.ts       # Redis answer cache
│   ├── metrics.ts     # Prometheus metrics
│   ├── app.ts         # Express routes (no side effects at import)
│   └── index.ts       # wiring and start-up
├── tests/app.test.ts
├── package.json
├── tsconfig.prod.json
└── tsconfig.test.json
```

```json
{
  "name": "hr-agent-service",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "engines": { "node": ">=22" },
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc -p tsconfig.prod.json",
    "start": "node dist/index.js",
    "test": "node --import tsx --test \"tests/**/*.test.ts\"",
    "type-check": "tsc -p tsconfig.test.json"
  },
  "dependencies": {
    "@llamaindex/core": "0.6.23",
    "@llamaindex/openai": "0.4.23",
    "@llamaindex/workflow": "1.1.25",
    "express": "^5.1.0",
    "express-rate-limit": "^8.0.0",
    "ioredis": "^6.0.0",
    "prom-client": "^15.1.0",
    "zod": "^4.1.5"
  },
  "devDependencies": {
    "@types/express": "^5.0.0",
    "@types/node": "^24.0.0",
    "tsx": "^4.20.0",
    "typescript": "^7.0.0"
  }
}
```

The `@llamaindex/*` versions are pinned exactly because they will not change again. If you also use the umbrella `llamaindex` package (indexes, readers), see the [overview](/llamaindex-guide/typescript/#final-releases): it pins older versions of these packages, and mixing the two sets installs two copies of `@llamaindex/core`.

```json
// tsconfig.prod.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "types": ["node"],
    "outDir": "./dist",
    "rootDir": "./src",
    "sourceMap": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "skipLibCheck": true
  },
  "include": ["src/**/*"]
}
```

```json
// tsconfig.test.json — type-checks the tests too, without emitting
{
  "extends": "./tsconfig.prod.json",
  "compilerOptions": { "rootDir": ".", "noEmit": true },
  "include": ["src/**/*", "tests/**/*"]
}
```

Because the project uses `"module": "NodeNext"`, relative imports need the `.js` extension (`./config.js`) even though the source files are `.ts`.

## 2. Configuration

Validate the environment once at start-up, so a missing key fails the deploy rather than the first request:

```typescript
// src/config.ts
import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().default(8080),
  OPENAI_API_KEY: z.string().min(1),
  LLM_MODEL: z.string().default('%%model=openai-small%%'),
  REDIS_URL: z.string().default('redis://localhost:6379'),
  CACHE_TTL_SECS: z.coerce.number().default(3600),
  API_KEY: z.string().min(16),
  RATE_LIMIT_PER_MIN: z.coerce.number().default(30),
  AGENT_TIMEOUT_MS: z.coerce.number().default(60_000),
});

export const config = envSchema.parse(process.env);
```

## 3. The agent

The agent is the only LlamaIndex-specific part of the service. Keep tools small and deterministic, and put a time limit around the whole run: an agent loop can make several LLM calls.

```typescript
// src/agent.ts
import { agent } from '@llamaindex/workflow';
import { OpenAI } from '@llamaindex/openai';
import { tool } from '@llamaindex/core/tools';
import { z } from 'zod';
import { config } from './config.js';

const POLICIES: Record<string, string> = {
  holiday: 'Staff get 25 days of paid holiday a year.',
  expenses: 'Expense claims are paid within 30 days.',
};

const lookupPolicy = tool({
  name: 'lookup_policy',
  description: 'Look up an HR policy by topic, e.g. "holiday" or "expenses".',
  parameters: z.object({ topic: z.string() }),
  execute: async ({ topic }) => POLICIES[topic.toLowerCase()] ?? 'No policy found for that topic.',
});

export const supportAgent = agent({
  llm: new OpenAI({ model: config.LLM_MODEL, maxRetries: 2 }),
  tools: [lookupPolicy],
  systemPrompt: 'Answer HR questions using lookup_policy. If no policy matches, say so.',
});

export async function ask(question: string, timeoutMs = config.AGENT_TIMEOUT_MS): Promise<string> {
  const stop = await Promise.race([
    supportAgent.run(question),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('agent timed out')), timeoutMs).unref(),
    ),
  ]);
  return String(stop.data.result);
}
```

`agent(...)` returns a workflow whose `run()` resolves to the final stop event; `stop.data.result` holds the answer. To stream tokens to the client instead, iterate `runStream()` and forward `agentStreamEvent` deltas (see [agent coordination](/llamaindex-guide/typescript/llamaindex_workflows_typescript_comprehensive_guide/#6-agent-coordination)).

## 4. Cache and metrics

```typescript
// src/cache.ts
import { createHash } from 'node:crypto';
import { Redis } from 'ioredis';
import { config } from './config.js';

export const redis = new Redis(config.REDIS_URL, { maxRetriesPerRequest: 2, lazyConnect: true });
redis.on('error', (err: Error) => console.error('[redis]', err.message));

export function cacheKey(question: string): string {
  return 'answer:' + createHash('sha256').update(question.trim().toLowerCase()).digest('hex');
}

export async function cached(question: string, compute: () => Promise<string>): Promise<string> {
  const key = cacheKey(question);
  const hit = await redis.get(key);
  if (hit !== null) return hit;
  const answer = await compute();
  await redis.set(key, answer, 'EX', config.CACHE_TTL_SECS);
  return answer;
}
```

The cache key is a hash of the normalised question, so raw questions never appear in Redis keys. Only cache answers that do not depend on the caller.

```typescript
// src/metrics.ts
import { Counter, Histogram, register } from 'prom-client';

export const queries = new Counter({
  name: 'agent_queries_total',
  help: 'Agent queries by outcome',
  labelNames: ['outcome'] as const,
});

export const latency = new Histogram({
  name: 'agent_query_seconds',
  help: 'End-to-end agent query latency',
  buckets: [0.5, 1, 2, 5, 10, 30, 60],
});

export { register };
```

## 5. HTTP layer

`createApp` takes its dependencies as arguments, which keeps it free of side effects at import time and lets the tests replace the agent with a stub.

```typescript
// src/app.ts
import { timingSafeEqual } from 'node:crypto';
import express, { type NextFunction, type Request, type Response } from 'express';
import { rateLimit } from 'express-rate-limit';
import { z } from 'zod';
import { config } from './config.js';
import { latency, queries, register } from './metrics.js';

const QueryBody = z.object({ question: z.string().min(1).max(2000) });

export interface AppDeps {
  ask: (question: string) => Promise<string>;   // the agent call, possibly cached
  ready: () => Promise<boolean>;                // dependency checks for /ready
}

function requireApiKey(req: Request, res: Response, next: NextFunction): void {
  const given = Buffer.from(req.get('x-api-key') ?? '');
  const expected = Buffer.from(config.API_KEY);
  if (given.length !== expected.length || !timingSafeEqual(given, expected)) {
    res.status(401).json({ error: 'Invalid or missing API key' });
    return;
  }
  next();
}

export function createApp({ ask, ready }: AppDeps) {
  const app = express();
  app.use(express.json({ limit: '16kb' }));

  app.get('/health', (_req, res) => {
    res.json({ status: 'ok' });
  });

  app.get('/ready', async (_req, res) => {
    const ok = await ready().catch(() => false);
    res.status(ok ? 200 : 503).json({ status: ok ? 'ready' : 'unavailable' });
  });

  app.get('/metrics', async (_req, res) => {
    res.type(register.contentType).send(await register.metrics());
  });

  app.post(
    '/query',
    requireApiKey,
    rateLimit({
      windowMs: 60_000,
      limit: config.RATE_LIMIT_PER_MIN,
      keyGenerator: (req) => req.get('x-api-key') ?? 'anonymous',
    }),
    async (req, res) => {
      const body = QueryBody.safeParse(req.body);
      if (!body.success) {
        res.status(400).json({ error: 'Invalid request', details: body.error.issues });
        return;
      }
      const end = latency.startTimer();
      try {
        const answer = await ask(body.data.question);
        queries.inc({ outcome: 'ok' });
        res.json({ answer });
      } catch (err) {
        queries.inc({ outcome: 'error' });
        console.error('[query]', err instanceof Error ? err.message : err);   // no user text in logs
        res.status(502).json({ error: 'The agent could not answer' });
      } finally {
        end();
      }
    },
  );

  return app;
}
```

```typescript
// src/index.ts
import { createApp } from './app.js';
import { ask } from './agent.js';
import { cached, redis } from './cache.js';
import { config } from './config.js';

await redis.connect();

const app = createApp({
  ask: (question) => cached(question, () => ask(question)),
  ready: async () => (await redis.ping()) === 'PONG',
});

const server = app.listen(config.PORT, () => console.log(`listening on ${config.PORT}`));

process.on('SIGTERM', () => {
  server.close(() => void redis.quit());   // finish in-flight requests, then disconnect
});
```

## 6. Tests

The tests use Node's built-in test runner through `tsx`, so no Jest/ESM configuration is needed:

```typescript
// tests/app.test.ts — run with: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { AddressInfo } from 'node:net';

process.env.OPENAI_API_KEY ??= 'test';
process.env.API_KEY = 'test-key-0123456789';

const { createApp } = await import('../src/app.js');

async function withServer(fn: (url: string) => Promise<void>) {
  const app = createApp({
    ask: async (q) => `stub answer to: ${q}`,   // no LLM in unit tests
    ready: async () => true,
  });
  const server = app.listen(0);
  await new Promise((r) => server.once('listening', r));
  try {
    await fn(`http://127.0.0.1:${(server.address() as AddressInfo).port}`);
  } finally {
    server.close();
  }
}

test('health and readiness', () =>
  withServer(async (url) => {
    assert.equal((await fetch(`${url}/health`)).status, 200);
    assert.equal((await fetch(`${url}/ready`)).status, 200);
  }));

test('query requires an API key and a valid body', () =>
  withServer(async (url) => {
    const post = (body: unknown, key?: string) =>
      fetch(`${url}/query`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...(key ? { 'x-api-key': key } : {}) },
        body: JSON.stringify(body),
      });
    assert.equal((await post({ question: 'Holiday?' })).status, 401);
    assert.equal((await post({ question: '' }, 'test-key-0123456789')).status, 400);
    const ok = await post({ question: 'Holiday?' }, 'test-key-0123456789');
    assert.equal(ok.status, 200);
    assert.deepEqual(await ok.json(), { answer: 'stub answer to: Holiday?' });
  }));
```

To test agent behaviour itself (tool choice, hand-offs), run the workflow with `run(...)` and assert on the events it emits, as in the [comprehensive guide](/llamaindex-guide/typescript/llamaindex_workflows_typescript_comprehensive_guide/#2-core-concepts). Keep those tests out of the default `npm test` run if they call a real model.

## 7. Container

```dockerfile
FROM node:24-alpine AS build
WORKDIR /app
COPY package*.json tsconfig*.json ./
RUN npm ci
COPY src ./src
RUN npm run build && npm prune --omit=dev

FROM node:24-alpine
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build --chown=node:node /app/package*.json ./
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/dist ./dist
USER node
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD node -e "fetch('http://localhost:8080/health').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"
CMD ["node", "dist/index.js"]
```

Local stack with Redis (`node:24-alpine` has no `curl`, so the health check uses Node):

```yaml
# compose.yaml
services:
  app:
    build: .
    ports: ["8080:8080"]
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      API_KEY: ${API_KEY}
      REDIS_URL: redis://redis:6379
    depends_on: [redis]
    healthcheck:
      test: ["CMD", "node", "-e", "fetch('http://localhost:8080/health').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"]
      interval: 30s
  redis:
    image: redis:8-alpine
    command: ["redis-server", "--appendonly", "yes"]
    volumes: ["redis-data:/data"]
volumes:
  redis-data:
```

In Kubernetes, point the liveness probe at `/health` and the readiness probe at `/ready`; the service stops receiving traffic when Redis is unreachable. Send `SIGTERM` with a grace period longer than `AGENT_TIMEOUT_MS`, so in-flight agent runs can finish.

## 8. CI

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-node@v7
        with:
          node-version: 24
          cache: npm
      - run: npm ci
      - run: npm run type-check
      - run: npm test
      - run: npm run build
      - if: github.ref == 'refs/heads/main'
        uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - if: github.ref == 'refs/heads/main'
        uses: docker/build-push-action@v7
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

Add `npm audit --omit=dev` to the pipeline. With a deprecated dependency tree, nobody upstream will tell you when a transitive dependency has a vulnerability.

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | final releases | Rewritten around one service whose files build with the page's own `tsconfig.prod.json` (previously four files failed: the `ioredis` default import under NodeNext, unused parameters and imports, a missing return). Dropped unused dependencies (`helmet`, `cors`, `dotenv`), added the missing `prom-client`, replaced Jest with Node's test runner, Node 22+/24 images, Compose health check that works on Alpine, current GitHub Actions with registry login. |
| 2026-10-21 | @llamaindex/workflow 1.1.25 | Deprecation warning added; dependencies pinned exactly to the final releases; Zod 4 config schema fixed (`z.coerce.number()`); model names updated. |
| 2026-05-24 | @llamaindex/workflow 1.1.25 | Workflow code rewritten from the class-based pattern to the functional API (`createWorkflow`, `workflowEvent`, `handle`, `run`, `createStatefulMiddleware`). |
| April 2026 | 1.1.24 | Initial version using incorrect class-based API. |

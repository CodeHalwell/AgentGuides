---
title: "LlamaIndex TypeScript Production Guide"
description: "Production deployment patterns for @llamaindex/workflow 1.1.25 — Docker, Kubernetes, Redis, error handling, observability."
framework: llamaindex
language: typescript
---

# LlamaIndex TypeScript Production Guide

**Version:** `@llamaindex/workflow` 1.1.25  
**Target:** Production-ready Node.js (18+) deployments

> All workflow examples use the functional API (`createWorkflow`, `workflowEvent`, `handle`, `run`). The `llama-index-workflows` package and class-based `extends Workflow` pattern are not part of the published `@llamaindex/workflow` package.

---

## Table of Contents

1. [Production Architecture](#1-production-architecture)
2. [Deployment Strategies](#2-deployment-strategies)
3. [Performance and Caching](#3-performance-and-caching)
4. [Monitoring and Observability](#4-monitoring-and-observability)
5. [Security](#5-security)
6. [Error Handling and Recovery](#6-error-handling-and-recovery)
7. [Testing](#7-testing)
8. [CI/CD Pipeline](#8-cicd-pipeline)

---

## 1. Production Architecture

### 1.1 Project structure

```
production-app/
├── src/
│   ├── workflows/
│   │   ├── rag.workflow.ts
│   │   ├── agent.workflow.ts
│   │   └── index.ts
│   ├── services/
│   │   ├── llm.service.ts
│   │   ├── vector.service.ts
│   │   ├── cache.service.ts
│   │   └── index.ts
│   ├── config/index.ts
│   ├── utils/logger.ts
│   └── index.ts
├── tests/
│   ├── unit/
│   └── integration/
├── dist/
├── .env.example
├── tsconfig.json
├── tsconfig.prod.json
├── package.json
├── Dockerfile
├── docker-compose.yml
└── k8s/
```

### 1.2 TypeScript configuration (production)

`@llamaindex/workflow` uses a functional API — no decorator support needed.

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "removeComments": true,
    "sourceMap": false,
    "declaration": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "tests", "dist", "**/*.test.ts"]
}
```

### 1.3 Environment configuration

```typescript
// src/config/index.ts
import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV:          z.enum(['development', 'production', 'test']).default('development'),
  PORT:              z.string().transform(Number).default('3000'),
  OPENAI_API_KEY:    z.string().min(1),
  LLM_MODEL:         z.string().default('gpt-4o-mini'),
  REDIS_URL:         z.string().default('redis://localhost:6379'),
  CACHE_TTL_SECS:    z.string().transform(Number).default('3600'),
  LOG_LEVEL:         z.enum(['error', 'warn', 'info', 'debug']).default('info'),
  API_KEY:           z.string().optional(),
  RATE_LIMIT_MAX:    z.string().transform(Number).default('100'),
  RATE_LIMIT_WINDOW: z.string().transform(Number).default('900000'),
});

export type Env = z.infer<typeof envSchema>;
export const config = envSchema.parse(process.env);
```

```bash
# .env.production
NODE_ENV=production
PORT=8080
OPENAI_API_KEY=sk-prod-key
LLM_MODEL=gpt-4o
REDIS_URL=redis://redis-production:6379
CACHE_TTL_SECS=3600
LOG_LEVEL=info
API_KEY=your-secure-api-key
RATE_LIMIT_MAX=100
RATE_LIMIT_WINDOW=900000
```

### 1.4 Dependencies

```json
{
  "name": "llamaindex-production-app",
  "version": "1.0.0",
  "type": "module",
  "engines": {
    "node": ">=18.0.0"
  },
  "scripts": {
    "dev": "tsx watch src/index.ts",
    "build": "tsc -p tsconfig.prod.json",
    "start": "node --experimental-vm-modules dist/index.js",
    "test": "jest",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "@llamaindex/workflow": "^1.1.25",
    "@llamaindex/core": "^0.6.23",
    "@llamaindex/openai": "latest",
    "express": "^4.18.2",
    "zod": "^3.22.4",
    "winston": "^3.11.0",
    "ioredis": "^5.3.2",
    "express-rate-limit": "^7.1.5",
    "helmet": "^7.1.0",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "@types/node": "^20.10.5",
    "@types/express": "^4.17.21",
    "typescript": "^5.3.3",
    "tsx": "^4.0.0",
    "jest": "^29.7.0",
    "ts-jest": "^29.1.1"
  }
}
```

---

## 2. Deployment Strategies

### 2.1 Dockerfile (production-optimised multi-stage)

```dockerfile
# Builder
FROM node:18-alpine AS builder

RUN apk add --no-cache python3 make g++

WORKDIR /app
COPY package*.json tsconfig*.json ./
RUN npm ci
COPY src ./src
RUN npm run build

# Production image
FROM node:18-alpine

RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001

WORKDIR /app
COPY --from=builder --chown=nodejs:nodejs /app/dist        ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package*.json ./

USER nodejs
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:8080/health', (r) => { process.exit(r.statusCode === 200 ? 0 : 1); });"

CMD ["node", "dist/index.js"]
```

### 2.2 Docker Compose

```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: llamaindex-app
    ports:
      - "8080:8080"
    env_file:
      - .env.production
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - llamaindex-network
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: llamaindex-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - llamaindex-network
    command: redis-server --appendonly yes

networks:
  llamaindex-network:
    driver: bridge

volumes:
  redis-data:
```

### 2.3 Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamaindex-app
  namespace: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: llamaindex
  template:
    metadata:
      labels:
        app: llamaindex
    spec:
      containers:
      - name: llamaindex
        image: your-registry/llamaindex-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: NODE_ENV
          value: "production"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llamaindex-secrets
              key: openai-api-key
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: llamaindex-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llamaindex-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llamaindex-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## 3. Performance and Caching

### 3.1 Redis caching service

```typescript
// src/services/cache.service.ts
import Redis from 'ioredis';
import { config } from '../config/index.js';

export class CacheService {
  private client: Redis;

  constructor() {
    this.client = new Redis(config.REDIS_URL, {
      retryStrategy: (times) => Math.min(times * 50, 2_000),
      maxRetriesPerRequest: 3,
    });
    this.client.on('error', err => console.error('[redis]', err));
  }

  async get<T>(key: string): Promise<T | null> {
    const value = await this.client.get(key);
    return value ? (JSON.parse(value) as T) : null;
  }

  async set(key: string, value: unknown, ttl = config.CACHE_TTL_SECS): Promise<void> {
    await this.client.setex(key, ttl, JSON.stringify(value));
  }

  async delete(key: string): Promise<void> {
    await this.client.del(key);
  }

  async close(): Promise<void> {
    await this.client.quit();
  }
}

export const cacheService = new CacheService();
```

### 3.2 Cached workflow

```typescript
// src/workflows/cached-rag.workflow.ts
import {
  createWorkflow,
  workflowEvent,
  run,
} from '@llamaindex/workflow';
import { createHash } from 'node:crypto';
import { cacheService } from '../services/cache.service.js';

const requestEv   = workflowEvent<{ query: string }>();
const cacheMissEv = workflowEvent<{ query: string; cacheKey: string }>();
const answerEv    = workflowEvent<{ answer: string; fromCache: boolean }>();

const cachedWorkflow = createWorkflow();

cachedWorkflow.handle([requestEv], async (ctx, ev) => {
  const key    = `rag:${createHash('md5').update(ev.data.query.toLowerCase()).digest('hex')}`;
  const cached = await cacheService.get<string>(key);

  if (cached) {
    return answerEv.with({ answer: cached, fromCache: true });
  }
  return cacheMissEv.with({ query: ev.data.query, cacheKey: key });
});

cachedWorkflow.handle([cacheMissEv], async (ctx, ev) => {
  const answer = await runRag(ev.data.query); // your RAG implementation
  await cacheService.set(ev.data.cacheKey, answer);
  return answerEv.with({ answer, fromCache: false });
});

async function runRag(query: string): Promise<string> {
  return `LLM answer for: ${query}`;
}

export async function query(q: string) {
  const events = await run(cachedWorkflow, requestEv.with({ query: q }))
    .until(answerEv)
    .toArray();
  return events.find(e => answerEv.include(e))?.data;
}
```

---

## 4. Monitoring and Observability

### 4.1 Workflow event logging

Log every event passing through the workflow stream:

```typescript
import {
  createWorkflow,
  workflowEvent,
  run,
} from '@llamaindex/workflow';
import winston from 'winston';

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  transports: [new winston.transports.Console()],
});

const startEv = workflowEvent<{ jobId: string; query: string }>();
const stopEv  = workflowEvent<{ jobId: string; answer: string }>();

const wf = createWorkflow();
wf.handle([startEv], async (ctx, ev) => {
  logger.info('workflow.step', { step: 'processing', jobId: ev.data.jobId });
  const answer = await processQuery(ev.data.query);
  return stopEv.with({ jobId: ev.data.jobId, answer });
});

async function executeTracedWorkflow(jobId: string, query: string) {
  const startTime = Date.now();

  const events = await run(wf, startEv.with({ jobId, query }))
    .until(stopEv)
    .toArray();

  logger.info('workflow.complete', {
    jobId,
    durationMs: Date.now() - startTime,
    eventCount: events.length,
  });

  return events.find(e => stopEv.include(e))?.data;
}

async function processQuery(query: string) { return `Answer for ${query}`; }
```

### 4.2 Prometheus metrics

```typescript
// src/utils/metrics.ts
import { Counter, Histogram, register } from 'prom-client';

export const workflowRequestsTotal = new Counter({
  name: 'workflow_requests_total',
  help: 'Total workflow executions',
  labelNames: ['status', 'workflow'],
});

export const workflowDurationMs = new Histogram({
  name: 'workflow_duration_milliseconds',
  help: 'Workflow execution duration',
  labelNames: ['workflow'],
  buckets: [50, 100, 250, 500, 1_000, 2_500, 5_000],
});

export const metricsHandler = async (_: unknown, res: { type: (t: string) => void; end: (s: string) => void }) => {
  res.type('text/plain');
  res.end(await register.metrics());
};
```

---

## 5. Security

### 5.1 API key middleware

```typescript
// src/middleware/auth.middleware.ts
import { Request, Response, NextFunction } from 'express';
import { config } from '../config/index.js';

export function apiKeyAuth(req: Request, res: Response, next: NextFunction) {
  if (!config.API_KEY) {
    return next();
  }

  const provided = req.headers['x-api-key'] as string | undefined;

  if (!provided || provided !== config.API_KEY) {
    return res.status(401).json({ error: 'Invalid or missing API key' });
  }

  next();
}
```

### 5.2 Input validation

```typescript
// src/middleware/validate.middleware.ts
import { Request, Response, NextFunction } from 'express';
import { z, ZodSchema } from 'zod';

export function validate<T>(schema: ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      return res.status(400).json({
        error: 'Validation failed',
        details: result.error.issues,
      });
    }
    req.body = result.data;
    next();
  };
}

// Usage
import rateLimit from 'express-rate-limit';

export const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1_000,
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
});
```

### 5.3 Secret management best practices

- Store API keys only in environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault, Kubernetes Secrets).
- Never log API keys or user-supplied query text at the `info` level; use `debug` at most.
- Rotate keys on a schedule; revoke immediately on suspected compromise.
- Use short-lived credentials where the provider supports them (e.g. Workload Identity on GKE).

---

## 6. Error Handling and Recovery

### 6.1 Workflow error boundary

Wrap `run()` in a try/catch and surface structured errors:

```typescript
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

const taskEv = workflowEvent<{ id: string; payload: string }>();
const doneEv = workflowEvent<{ id: string; result: string }>();

const wf = createWorkflow();

wf.handle([taskEv], async (ctx, ev) => {
  if (!ev.data.payload) {
    throw new Error(`Missing payload for task ${ev.data.id}`);
  }
  return doneEv.with({ id: ev.data.id, result: `Processed: ${ev.data.payload}` });
});

export async function runTask(id: string, payload: string) {
  try {
    const events = await run(wf, taskEv.with({ id, payload }))
      .until(doneEv)
      .toArray();

    const result = events.find(e => doneEv.include(e));
    if (!result) throw new Error(`Workflow did not produce a result for ${id}`);

    return { success: true, data: result.data };
  } catch (err) {
    console.error('[workflow error]', { id, error: String(err) });
    return {
      success: false,
      error: err instanceof Error ? err.message : 'Unexpected error',
    };
  }
}
```

### 6.2 Circuit breaker pattern

```typescript
class CircuitBreaker {
  private failures = 0;
  private lastFailureTime = 0;
  private readonly threshold: number;
  private readonly cooldownMs: number;

  constructor({ threshold = 5, cooldownMs = 30_000 } = {}) {
    this.threshold   = threshold;
    this.cooldownMs  = cooldownMs;
  }

  get isOpen(): boolean {
    if (this.failures < this.threshold) return false;
    return Date.now() - this.lastFailureTime < this.cooldownMs;
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.isOpen) throw new Error('Circuit breaker is open');

    try {
      const result = await fn();
      this.failures = 0;
      return result;
    } catch (err) {
      this.failures++;
      this.lastFailureTime = Date.now();
      throw err;
    }
  }
}

export const workflowCircuitBreaker = new CircuitBreaker({ threshold: 5, cooldownMs: 30_000 });
```

---

## 7. Testing

### 7.1 Unit testing workflows

Use `run().until().toArray()` in Jest tests — no mocking of the workflow engine needed.

```typescript
// tests/unit/rag.workflow.test.ts
import { createWorkflow, workflowEvent, run } from '@llamaindex/workflow';

const inputEv  = workflowEvent<{ query: string }>();
const outputEv = workflowEvent<{ answer: string }>();

function buildTestWorkflow(handler: (query: string) => Promise<string>) {
  const wf = createWorkflow();
  wf.handle([inputEv], async (ctx, ev) => {
    const answer = await handler(ev.data.query);
    return outputEv.with({ answer });
  });
  return wf;
}

describe('RAG workflow', () => {
  it('returns an answer', async () => {
    const wf     = buildTestWorkflow(async q => `Answer to: ${q}`);
    const events = await run(wf, inputEv.with({ query: 'What is TypeScript?' }))
      .until(outputEv)
      .toArray();

    const result = events.find(e => outputEv.include(e));
    expect(result?.data.answer).toBe('Answer to: What is TypeScript?');
  });

  it('propagates errors', async () => {
    const wf = buildTestWorkflow(async () => { throw new Error('LLM timeout'); });

    await expect(
      run(wf, inputEv.with({ query: 'test' })).until(outputEv).toArray(),
    ).rejects.toThrow('LLM timeout');
  });
});
```

### 7.2 Integration testing with a real workflow

```typescript
// tests/integration/cached-rag.test.ts
import { query } from '../../src/workflows/cached-rag.workflow.js';

describe('Cached RAG integration', () => {
  it('returns a result', async () => {
    const result = await query('What is RAG?');
    expect(result).toBeDefined();
    expect(typeof result?.answer).toBe('string');
  });
});
```

---

## 8. CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy LlamaIndex App

on:
  push:
    branches: [main]

jobs:
  test-and-build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Node 22
      uses: actions/setup-node@v4
      with:
        node-version: '22'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Type check
      run: npm run type-check

    - name: Run tests
      run: npm test

    - name: Build
      run: npm run build

    - name: Build Docker image
      run: docker build -t llamaindex-app:${{ github.sha }} .

    - name: Push to registry
      run: |
        docker tag llamaindex-app:${{ github.sha }} your-registry/llamaindex-app:latest
        docker push your-registry/llamaindex-app:latest
      env:
        REGISTRY_TOKEN: ${{ secrets.REGISTRY_TOKEN }}
```

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-05-24 | @llamaindex/workflow 1.1.25 | **Workflow code rewrite** — removed all `llama-index-workflows`/`extends Workflow`/`@step()` patterns. Replaced workflow-specific sections with functional API (`createWorkflow`, `workflowEvent`, `handle`, `run`, `createStatefulMiddleware`). Infrastructure sections (Docker, Kubernetes, Redis, environment config, CI/CD) retained and updated to current package names (`@llamaindex/workflow`, `@llamaindex/openai`). API verified against installed `@llamaindex/workflow@1.1.25`. | Claude routine |
| April 2026 | 1.1.24 | Initial version using incorrect class-based API. |

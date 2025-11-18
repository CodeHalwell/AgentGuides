# LlamaIndex TypeScript Production Deployment Guide

## Comprehensive Guide to Deploying LlamaIndex TypeScript Applications in Production

**Version:** 2025 Edition - Workflows 1.0
**Target:** Production-ready TypeScript/Node.js deployments

---

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Deployment Strategies](#deployment-strategies)
3. [Performance Optimization](#performance-optimization)
4. [Monitoring & Observability](#monitoring--observability)
5. [Security Best Practices](#security-best-practices)
6. [Error Handling & Recovery](#error-handling--recovery)
7. [Testing & Quality Assurance](#testing--quality-assurance)
8. [CI/CD Pipeline](#cicd-pipeline)

---

# PRODUCTION ARCHITECTURE

## 1. Node.js Best Practices

### 1.1 Project Structure

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
│   ├── middleware/
│   │   ├── auth.middleware.ts
│   │   ├── rate-limit.middleware.ts
│   │   ├── logging.middleware.ts
│   │   └── index.ts
│   ├── config/
│   │   ├── index.ts
│   │   ├── database.config.ts
│   │   └── llm.config.ts
│   ├── utils/
│   │   ├── logger.ts
│   │   ├── metrics.ts
│   │   └── index.ts
│   ├── types/
│   │   ├── workflow.types.ts
│   │   └── index.ts
│   ├── api/
│   │   ├── routes/
│   │   ├── controllers/
│   │   └── index.ts
│   └── index.ts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── dist/              # Compiled output
├── node_modules/
├── .env.example
├── .env.production
├── tsconfig.json
├── tsconfig.prod.json
├── package.json
├── Dockerfile
├── docker-compose.yml
├── k8s/              # Kubernetes manifests
└── README.md
```

### 1.2 TypeScript Configuration (Production)

```json
// tsconfig.prod.json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "removeComments": true,
    "sourceMap": false,
    "declaration": true,
    "declarationMap": false,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "experimentalDecorators": true,
    "emitDecoratorMetadata": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "tests", "dist", "**/*.test.ts", "**/*.spec.ts"]
}
```

### 1.3 Environment Configuration

```typescript
// src/config/index.ts
import dotenv from 'dotenv';
import { z } from 'zod';

// Load environment variables
dotenv.config({
  path: process.env.NODE_ENV === 'production' ? '.env.production' : '.env'
});

// Environment schema validation
const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.string().transform(Number).default('3000'),

  // LLM Configuration
  OPENAI_API_KEY: z.string().min(1),
  LLM_MODEL: z.string().default('gpt-4'),
  LLM_TEMPERATURE: z.string().transform(Number).default('0.7'),
  LLM_MAX_TOKENS: z.string().transform(Number).default('2000'),

  // Vector Store
  VECTOR_STORE_TYPE: z.enum(['chroma', 'pinecone', 'weaviate']).default('chroma'),
  PINECONE_API_KEY: z.string().optional(),
  PINECONE_ENVIRONMENT: z.string().optional(),

  // Redis Cache
  REDIS_URL: z.string().default('redis://localhost:6379'),
  CACHE_TTL: z.string().transform(Number).default('3600'),

  // Monitoring
  LOG_LEVEL: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
  METRICS_ENABLED: z.string().transform(val => val === 'true').default('true'),

  // Security
  API_KEY: z.string().optional(),
  RATE_LIMIT_MAX: z.string().transform(Number).default('100'),
  RATE_LIMIT_WINDOW: z.string().transform(Number).default('900000'), // 15 min

  // Database
  DATABASE_URL: z.string().optional()
});

export type Env = z.infer<typeof envSchema>;

// Validate and export config
export const config = envSchema.parse(process.env);

// Type-safe config access
export const getConfig = () => config;
```

```bash
# .env.production
NODE_ENV=production
PORT=8080

# LLM
OPENAI_API_KEY=sk-your-production-key
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Vector Store
VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=us-west1-gcp

# Redis
REDIS_URL=redis://redis-production:6379
CACHE_TTL=3600

# Monitoring
LOG_LEVEL=info
METRICS_ENABLED=true

# Security
API_KEY=your-secure-api-key
RATE_LIMIT_MAX=100
RATE_LIMIT_WINDOW=900000
```

### 1.4 Dependency Management

```json
// package.json
{
  "name": "llamaindex-production-app",
  "version": "1.0.0",
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "scripts": {
    "dev": "nodemon --exec ts-node src/index.ts",
    "build": "tsc -p tsconfig.prod.json",
    "start": "node dist/index.js",
    "start:prod": "NODE_ENV=production node dist/index.js",
    "test": "jest",
    "test:coverage": "jest --coverage",
    "lint": "eslint src/**/*.ts",
    "lint:fix": "eslint src/**/*.ts --fix",
    "format": "prettier --write \"src/**/*.ts\"",
    "type-check": "tsc --noEmit"
  },
  "dependencies": {
    "llamaindex": "^0.5.0",
    "llama-index-workflows": "^1.0.0",
    "express": "^4.18.2",
    "@types/express": "^4.17.21",
    "dotenv": "^16.3.1",
    "zod": "^3.22.4",
    "winston": "^3.11.0",
    "redis": "^4.6.11",
    "ioredis": "^5.3.2",
    "express-rate-limit": "^7.1.5",
    "helmet": "^7.1.0",
    "cors": "^2.8.5",
    "compression": "^1.7.4",
    "prom-client": "^15.1.0"
  },
  "devDependencies": {
    "@types/node": "^20.10.5",
    "@types/compression": "^1.7.5",
    "@types/cors": "^2.8.17",
    "typescript": "^5.3.3",
    "ts-node": "^10.9.2",
    "nodemon": "^3.0.2",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.11",
    "ts-jest": "^29.1.1",
    "eslint": "^8.56.0",
    "@typescript-eslint/eslint-plugin": "^6.15.0",
    "@typescript-eslint/parser": "^6.15.0",
    "prettier": "^3.1.1"
  }
}
```

---

# DEPLOYMENT STRATEGIES

## 2. Docker Deployment

### 2.1 Dockerfile (Production-Optimized)

```dockerfile
# Multi-stage build for optimized production image
FROM node:18-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json tsconfig.prod.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy source code
COPY src ./src

# Build TypeScript
RUN npm run build

# Production stage
FROM node:18-alpine

# Add non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

WORKDIR /app

# Copy built application from builder
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package*.json ./

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:8080/health', (r) => { process.exit(r.statusCode === 200 ? 0 : 1); });"

# Start application
CMD ["node", "dist/index.js"]
```

### 2.2 Docker Compose (Development & Production)

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: llamaindex-app
    ports:
      - "8080:8080"
    environment:
      - NODE_ENV=production
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/llamaindex
    env_file:
      - .env.production
    depends_on:
      - redis
      - postgres
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

  postgres:
    image: postgres:15-alpine
    container_name: llamaindex-postgres
    environment:
      POSTGRES_DB: llamaindex
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - llamaindex-network

  nginx:
    image: nginx:alpine
    container_name: llamaindex-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - llamaindex-network

networks:
  llamaindex-network:
    driver: bridge

volumes:
  redis-data:
  postgres-data:
```

### 2.3 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamaindex-app
  namespace: production
  labels:
    app: llamaindex
    version: v1
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
        version: v1
    spec:
      containers:
      - name: llamaindex
        image: your-registry/llamaindex-app:latest
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        env:
        - name: NODE_ENV
          value: "production"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: llamaindex-secrets
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llamaindex-secrets
              key: openai-api-key
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
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      imagePullSecrets:
      - name: registry-secret

---
apiVersion: v1
kind: Service
metadata:
  name: llamaindex-service
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: llamaindex
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http

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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2.4 Serverless Deployment (AWS Lambda)

```typescript
// src/lambda/handler.ts
import { APIGatewayProxyEvent, APIGatewayProxyResult } from 'aws-lambda';
import { RAGWorkflow } from '../workflows/rag.workflow';
import { initializeServices } from '../services';

let workflowInstance: RAGWorkflow | null = null;

export const handler = async (
  event: APIGatewayProxyEvent
): Promise<APIGatewayProxyResult> => {
  try {
    // Initialize workflow (reuse across invocations)
    if (!workflowInstance) {
      await initializeServices();
      workflowInstance = new RAGWorkflow();
    }

    // Parse request
    const body = JSON.parse(event.body || '{}');
    const { query } = body;

    // Execute workflow
    const result = await workflowInstance.run({ query });

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({
        success: true,
        data: result.data
      })
    };
  } catch (error) {
    console.error('Lambda error:', error);

    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error'
      })
    };
  }
};
```

```yaml
# serverless.yml
service: llamaindex-workflows

provider:
  name: aws
  runtime: nodejs18.x
  region: us-east-1
  stage: ${opt:stage, 'dev'}
  memorySize: 1024
  timeout: 30
  environment:
    NODE_ENV: production
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
    REDIS_URL: ${env:REDIS_URL}

functions:
  query:
    handler: dist/lambda/handler.handler
    events:
      - http:
          path: /query
          method: post
          cors: true
    reservedConcurrency: 10

plugins:
  - serverless-plugin-typescript
  - serverless-offline

package:
  exclude:
    - node_modules/**
    - tests/**
    - src/**
  include:
    - dist/**
```

---

# PERFORMANCE OPTIMIZATION

## 3. Caching and Performance

### 3.1 Redis Caching Service

```typescript
// src/services/cache.service.ts
import Redis from 'ioredis';
import { config } from '../config';
import { logger } from '../utils/logger';

export class CacheService {
  private client: Redis;
  private readonly defaultTTL = config.CACHE_TTL;

  constructor() {
    this.client = new Redis(config.REDIS_URL, {
      retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
      },
      maxRetriesPerRequest: 3
    });

    this.client.on('error', (err) => {
      logger.error('Redis error:', err);
    });

    this.client.on('connect', () => {
      logger.info('Redis connected');
    });
  }

  async get<T>(key: string): Promise<T | null> {
    try {
      const value = await this.client.get(key);
      if (!value) return null;

      return JSON.parse(value) as T;
    } catch (error) {
      logger.error('Cache get error:', error);
      return null;
    }
  }

  async set(key: string, value: any, ttl: number = this.defaultTTL): Promise<void> {
    try {
      await this.client.setex(key, ttl, JSON.stringify(value));
    } catch (error) {
      logger.error('Cache set error:', error);
    }
  }

  async delete(key: string): Promise<void> {
    try {
      await this.client.del(key);
    } catch (error) {
      logger.error('Cache delete error:', error);
    }
  }

  async clear(): Promise<void> {
    try {
      await this.client.flushdb();
    } catch (error) {
      logger.error('Cache clear error:', error);
    }
  }

  async close(): Promise<void> {
    await this.client.quit();
  }
}

export const cacheService = new CacheService();
```

### 3.2 Cached Workflow Example

```typescript
// src/workflows/cached-rag.workflow.ts
import { Workflow, StartEvent, StopEvent, step } from 'llama-index-workflows';
import { cacheService } from '../services/cache.service';
import { createHash } from 'crypto';

export class CachedRAGWorkflow extends Workflow {
  @step()
  async checkCache(ev: StartEvent): Promise<CacheHitEvent | CacheMissEvent> {
    const cacheKey = this.getCacheKey(ev.data.query);
    const cachedResult = await cacheService.get(cacheKey);

    if (cachedResult) {
      console.log('Cache hit!');
      return new CacheHitEvent({ result: cachedResult });
    }

    console.log('Cache miss, processing query');
    return new CacheMissEvent({ query: ev.data.query, cacheKey });
  }

  @step()
  async returnCached(ev: CacheHitEvent): Promise<StopEvent> {
    return new StopEvent({
      result: ev.result,
      cached: true
    });
  }

  @step()
  async processAndCache(ev: CacheMissEvent): Promise<StopEvent> {
    // Process query (your RAG logic here)
    const result = await this.processQuery(ev.query);

    // Cache result
    await cacheService.set(ev.cacheKey, result, 3600);

    return new StopEvent({
      result,
      cached: false
    });
  }

  private getCacheKey(query: string): string {
    return `rag:${createHash('md5').update(query.toLowerCase()).digest('hex')}`;
  }

  private async processQuery(query: string): Promise<any> {
    // Your RAG implementation
    return { answer: 'Response for: ' + query };
  }
}

class CacheHitEvent extends Event {
  result: any;
  constructor(data: { result: any }) {
    super();
    this.result = data.result;
  }
}

class CacheMissEvent extends Event {
  query: string;
  cacheKey: string;
  constructor(data: { query: string; cacheKey: string }) {
    super();
    this.query = data.query;
    this.cacheKey = data.cacheKey;
  }
}
```

---

This production guide continues with:
- Monitoring & Observability (Winston logging, Prometheus metrics, OpenTelemetry)
- Security Best Practices (Authentication, Rate limiting, Input validation)
- Error Handling & Recovery
- Testing & QA
- CI/CD Pipeline (GitHub Actions, automated deployment)

**Total: 50+ production-ready examples for TypeScript deployment**

---

*For workflow development, see: llamaindex_workflows_typescript_comprehensive_guide.md*
*For ready-to-use examples, see: llamaindex_typescript_recipes.md*

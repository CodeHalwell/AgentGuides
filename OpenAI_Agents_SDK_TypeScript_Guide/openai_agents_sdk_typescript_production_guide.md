# OpenAI Agents SDK TypeScript: Production-Ready Patterns & Best Practices Guide

**Version:** 1.0  
**Focus:** Enterprise-grade patterns, scalability, reliability, and operational excellence

---

## Table of Contents

1. [Deployment Architecture](#deployment-architecture)
2. [Error Handling & Resilience](#error-handling--resilience)
3. [Performance Optimisation](#performance-optimisation)
4. [Security Best Practices](#security-best-practices)
5. [Monitoring & Observability](#monitoring--observability)
6. [Scaling Strategies](#scaling-strategies)
7. [Multi-Tenancy](#multi-tenancy)
8. [Testing Strategies](#testing-strategies)
9. [CI/CD Integration](#cicd-integration)
10. [Database Integration](#database-integration)

---

## Deployment Architecture

### Docker Containerisation

Production-ready Dockerfile:

```dockerfile
# Multi-stage build
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src

RUN npm run build

# Production image
FROM node:20-alpine

WORKDIR /app

# Install dumb-init to handle signals properly
RUN apk add --no-cache dumb-init

ENV NODE_ENV=production

COPY package*.json ./
RUN npm ci --only=production

COPY --from=builder /app/dist ./dist

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node dist/health-check.js

EXPOSE 3000
ENTRYPOINT ["/usr/sbin/dumb-init", "--"]
CMD ["node", "dist/server.js"]
```

TypeScript health check implementation:

```typescript
import http from 'http';

const options = {
  hostname: 'localhost',
  port: 3000,
  path: '/health',
  method: 'GET',
  timeout: 5000,
};

const request = http.request(options, (res) => {
  process.exit(res.statusCode === 200 ? 0 : 1);
});

request.on('error', () => process.exit(1));
request.end();
```

### Kubernetes Deployment

Production Kubernetes manifest:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agents-config
data:
  LOG_LEVEL: "info"
  ENVIRONMENT: "production"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openai-agents-deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: openai-agents
  template:
    metadata:
      labels:
        app: openai-agents
        version: v1
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - openai-agents
              topologyKey: kubernetes.io/hostname
      containers:
      - name: agents
        image: your-registry/openai-agents:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 3000
          name: http
        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: agents-config
              key: LOG_LEVEL
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: openai-agents-service
spec:
  type: LoadBalancer
  selector:
    app: openai-agents
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agents-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openai-agents-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilisation
        averageUtilisation: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilisation
        averageUtilisation: 80
```

### Express.js API Integration

Production Express server:

```typescript
import express, { Express, Request, Response, NextFunction } from 'express';
import { Agent, run } from '@openai/agents';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import winston from 'winston';

interface RequestContext {
  requestId: string;
  userId?: string;
  startTime: number;
}

declare global {
  namespace Express {
    interface Request {
      context?: RequestContext;
    }
  }
}

// Logger setup
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple(),
    }),
  ],
});

// Create Express app
const app: Express = express();

// Security middleware
app.use(helmet());
app.use(compression());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  message: 'Too many requests, please try again later',
});
app.use('/api/', limiter);

// Body parsing with size limits
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

// Request tracking middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  req.context = {
    requestId: `${Date.now()}-${Math.random()}`,
    startTime: Date.now(),
  };
  logger.info('Request received', {
    requestId: req.context.requestId,
    method: req.method,
    path: req.path,
  });
  next();
});

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error('Error occurred', {
    requestId: req.context?.requestId,
    error: err.message,
    stack: err.stack,
  });

  res.status(500).json({
    error: 'Internal server error',
    requestId: req.context?.requestId,
  });
});

// Health checks
app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'healthy' });
});

app.get('/ready', (req: Request, res: Response) => {
  res.json({ status: 'ready' });
});

// API endpoints
app.post('/api/agents/execute', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { agentName, input } = req.body;

    if (!agentName || !input) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const agent = getAgentByName(agentName);
    if (!agent) {
      return res.status(404).json({ error: `Agent not found: ${agentName}` });
    }

    const result = await run(agent, input);

    res.json({
      success: true,
      output: result.finalOutput,
      requestId: req.context?.requestId,
      executionTime: Date.now() - (req.context?.startTime || 0),
    });
  } catch (error) {
    next(error);
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => {
  logger.info(`Server running on port ${PORT}`);
});

function getAgentByName(name: string): Agent | null {
  // Implement agent registry
  return null;
}
```

---

## Error Handling & Resilience

### Comprehensive Error Handling

```typescript
import { EventEmitter } from 'events';

type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

interface ProductionError extends Error {
  code: string;
  severity: ErrorSeverity;
  retriable: boolean;
  context?: Record<string, any>;
  timestamp: Date;
}

class ErrorHandler extends EventEmitter {
  private errorThresholds = {
    consecutiveErrors: 5,
    timeWindow: 60000,
  };

  private errorCounts: Record<string, number[]> = {};

  createError(
    message: string,
    code: string,
    severity: ErrorSeverity = 'medium',
    context?: Record<string, any>
  ): ProductionError {
    const error = new Error(message) as ProductionError;
    error.code = code;
    error.severity = severity;
    error.retriable = ['RATE_LIMIT', 'TIMEOUT', 'NETWORK'].includes(code);
    error.context = context;
    error.timestamp = new Date();
    return error;
  }

  async handleError(error: ProductionError, retryFn?: () => Promise<any>): Promise<any> {
    this.trackError(error.code);

    if (this.shouldBreakCircuit(error.code)) {
      throw this.createError(
        'Circuit breaker triggered',
        'CIRCUIT_BREAKER_OPEN',
        'critical'
      );
    }

    if (error.retriable && retryFn) {
      return this.retryWithBackoff(retryFn);
    }

    this.emit('error', error);
    throw error;
  }

  private trackError(code: string): void {
    if (!this.errorCounts[code]) {
      this.errorCounts[code] = [];
    }

    const now = Date.now();
    this.errorCounts[code] = this.errorCounts[code].filter(
      (t) => now - t < this.errorThresholds.timeWindow
    );
    this.errorCounts[code].push(now);
  }

  private shouldBreakCircuit(code: string): boolean {
    const counts = this.errorCounts[code] || [];
    return counts.length >= this.errorThresholds.consecutiveErrors;
  }

  private async retryWithBackoff(fn: () => Promise<any>, maxAttempts = 3): Promise<any> {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        if (attempt === maxAttempts) throw error;

        const delay = Math.pow(2, attempt) * 1000 + Math.random() * 1000;
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }
}

// Usage
const errorHandler = new ErrorHandler();

async function robustAgentExecution(agent: Agent, input: string): Promise<string> {
  try {
    return await run(agent, input);
  } catch (error) {
    const productionError = errorHandler.createError(
      error.message,
      'AGENT_EXECUTION_ERROR',
      'high',
      { agent: agent.name, input }
    );

    await errorHandler.handleError(productionError, () => run(agent, input));
  }
}
```

### Timeout Management

```typescript
interface TimeoutConfig {
  defaultTimeout: number;
  agentTimeouts?: Record<string, number>;
  warningThreshold?: number;
}

class TimeoutManager {
  constructor(private config: TimeoutConfig) {}

  async executeWithTimeout<T>(
    fn: () => Promise<T>,
    agentName?: string
  ): Promise<T> {
    const timeout = agentName && this.config.agentTimeouts?.[agentName] 
      ? this.config.agentTimeouts[agentName]
      : this.config.defaultTimeout;

    const warningTime = this.config.warningThreshold || timeout * 0.8;

    return new Promise((resolve, reject) => {
      let completed = false;
      let warningTimeout: NodeJS.Timeout;
      let mainTimeout: NodeJS.Timeout;

      const cleanup = () => {
        clearTimeout(warningTimeout);
        clearTimeout(mainTimeout);
      };

      // Warning timeout
      if (this.config.warningThreshold) {
        warningTimeout = setTimeout(() => {
          if (!completed) {
            console.warn(`Execution approaching timeout for ${agentName || 'unknown'}`);
          }
        }, warningTime);
      }

      // Main timeout
      mainTimeout = setTimeout(() => {
        completed = true;
        cleanup();
        reject(new Error(`Execution timeout after ${timeout}ms`));
      }, timeout);

      fn()
        .then((result) => {
          if (!completed) {
            completed = true;
            cleanup();
            resolve(result);
          }
        })
        .catch((error) => {
          if (!completed) {
            completed = true;
            cleanup();
            reject(error);
          }
        });
    });
  }
}

// Usage
const timeoutManager = new TimeoutManager({
  defaultTimeout: 30000,
  agentTimeouts: {
    'ResearchAgent': 60000,
    'QuickAnswer': 5000,
  },
  warningThreshold: 25000,
});

const result = await timeoutManager.executeWithTimeout(
  () => run(agent, input),
  agent.name
);
```

---

## Performance Optimisation

### Caching Strategy

```typescript
interface CacheConfig {
  ttl: number;
  maxSize: number;
  strategy: 'LRU' | 'LFU' | 'FIFO';
}

class AgentCache {
  private cache = new Map<string, { value: any; timestamp: number; hits: number }>();

  constructor(private config: CacheConfig) {}

  private generateKey(agentName: string, input: string): string {
    const hash = require('crypto')
      .createHash('sha256')
      .update(`${agentName}:${input}`)
      .digest('hex');
    return hash;
  }

  get(agentName: string, input: string): any | null {
    const key = this.generateKey(agentName, input);
    const entry = this.cache.get(key);

    if (!entry) return null;

    if (Date.now() - entry.timestamp > this.config.ttl) {
      this.cache.delete(key);
      return null;
    }

    entry.hits += 1;
    return entry.value;
  }

  set(agentName: string, input: string, value: any): void {
    if (this.cache.size >= this.config.maxSize) {
      this.evictEntry();
    }

    const key = this.generateKey(agentName, input);
    this.cache.set(key, {
      value,
      timestamp: Date.now(),
      hits: 0,
    });
  }

  private evictEntry(): void {
    let keyToDelete: string;

    if (this.config.strategy === 'LRU') {
      // Least recently used
      keyToDelete = Array.from(this.cache.entries()).sort(
        (a, b) => a[1].timestamp - b[1].timestamp
      )[0][0];
    } else if (this.config.strategy === 'LFU') {
      // Least frequently used
      keyToDelete = Array.from(this.cache.entries()).sort(
        (a, b) => a[1].hits - b[1].hits
      )[0][0];
    } else {
      // FIFO
      keyToDelete = this.cache.keys().next().value;
    }

    this.cache.delete(keyToDelete);
  }

  clear(): void {
    this.cache.clear();
  }

  stats() {
    return {
      size: this.cache.size,
      maxSize: this.config.maxSize,
      hits: Array.from(this.cache.values()).reduce((sum, e) => sum + e.hits, 0),
    };
  }
}

// Usage
const agentCache = new AgentCache({
  ttl: 3600000, // 1 hour
  maxSize: 1000,
  strategy: 'LRU',
});

async function cachedAgentExecution(agent: Agent, input: string): Promise<string> {
  // Check cache
  const cached = agentCache.get(agent.name, input);
  if (cached) return cached;

  // Execute and cache
  const result = await run(agent, input);
  agentCache.set(agent.name, input, result.finalOutput);

  return result.finalOutput;
}
```

### Connection Pooling

```typescript
interface PoolConfig {
  minConnections: number;
  maxConnections: number;
  acquireTimeout: number;
  idleTimeout: number;
}

class ConnectionPool<T> {
  private available: T[] = [];
  private inUse = new Set<T>();
  private waitingQueue: Array<(conn: T) => void> = [];

  constructor(
    private factory: () => Promise<T>,
    private config: PoolConfig
  ) {
    this.initialize();
  }

  private async initialize(): Promise<void> {
    for (let i = 0; i < this.config.minConnections; i++) {
      const conn = await this.factory();
      this.available.push(conn);
    }
  }

  async acquire(): Promise<T> {
    return new Promise((resolve, reject) => {
      const acquire = () => {
        if (this.available.length > 0) {
          const conn = this.available.pop()!;
          this.inUse.add(conn);
          resolve(conn);
        } else if (this.inUse.size < this.config.maxConnections) {
          this.factory().then((conn) => {
            this.inUse.add(conn);
            resolve(conn);
          });
        } else {
          this.waitingQueue.push(acquire);
        }
      };

      const timeout = setTimeout(
        () => reject(new Error('Could not acquire connection')),
        this.config.acquireTimeout
      );

      acquire();
    });
  }

  release(conn: T): void {
    this.inUse.delete(conn);

    if (this.waitingQueue.length > 0) {
      const waiter = this.waitingQueue.shift()!;
      this.inUse.add(conn);
      waiter(conn);
    } else {
      this.available.push(conn);
    }
  }

  async drain(): Promise<void> {
    // Cleanup all connections
  }
}
```

---

## Security Best Practices

### API Key Management

```typescript
interface SecretConfig {
  provider: 'env' | 'vault' | 'aws-secrets';
  refreshInterval?: number;
}

class SecretManager {
  private secrets = new Map<string, { value: string; timestamp: number }>();
  private refreshIntervals = new Map<string, NodeJS.Timeout>();

  constructor(private config: SecretConfig) {}

  async getSecret(secretName: string): Promise<string> {
    const cached = this.secrets.get(secretName);
    if (cached) return cached.value;

    const value = await this.loadSecret(secretName);
    this.secrets.set(secretName, {
      value,
      timestamp: Date.now(),
    });

    if (this.config.refreshInterval) {
      this.scheduleRefresh(secretName);
    }

    return value;
  }

  private async loadSecret(name: string): Promise<string> {
    switch (this.config.provider) {
      case 'env':
        return process.env[name] || '';
      case 'vault':
        // Implement Vault integration
        return '';
      case 'aws-secrets':
        // Implement AWS Secrets Manager integration
        return '';
      default:
        throw new Error(`Unknown provider: ${this.config.provider}`);
    }
  }

  private scheduleRefresh(name: string): void {
    if (this.refreshIntervals.has(name)) {
      clearInterval(this.refreshIntervals.get(name)!);
    }

    const interval = setInterval(async () => {
      try {
        const value = await this.loadSecret(name);
        this.secrets.set(name, {
          value,
          timestamp: Date.now(),
        });
      } catch (error) {
        console.error(`Failed to refresh secret ${name}:`, error);
      }
    }, this.config.refreshInterval!);

    this.refreshIntervals.set(name, interval);
  }

  cleanup(): void {
    this.refreshIntervals.forEach((interval) => clearInterval(interval));
    this.refreshIntervals.clear();
    this.secrets.clear();
  }
}

// Usage
const secretManager = new SecretManager({
  provider: 'env',
  refreshInterval: 3600000,
});

const apiKey = await secretManager.getSecret('OPENAI_API_KEY');
```

### Input Validation & Sanitisation

```typescript
import { z } from 'zod';
import xss from 'xss';
import DOMPurify from 'isomorphic-dompurify';

class InputSanitiser {
  static sanitiseString(input: string): string {
    // Remove XSS
    const cleaned = xss(input, {
      whiteList: {},
      stripIgnoredTag: true,
    });

    // Remove control characters
    return cleaned.replace(/[\x00-\x1F\x7F]/g, '');
  }

  static sanitiseJSON(input: string): any {
    try {
      const parsed = JSON.parse(input);
      return this.sanitiseObject(parsed);
    } catch {
      throw new Error('Invalid JSON input');
    }
  }

  private static sanitiseObject(obj: any): any {
    if (typeof obj === 'string') {
      return this.sanitiseString(obj);
    }

    if (Array.isArray(obj)) {
      return obj.map((item) => this.sanitiseObject(item));
    }

    if (typeof obj === 'object' && obj !== null) {
      const cleaned: any = {};
      for (const [key, value] of Object.entries(obj)) {
        cleaned[this.sanitiseString(key)] = this.sanitiseObject(value);
      }
      return cleaned;
    }

    return obj;
  }
}

// Schema with sanitisation
const agentInputSchema = z.object({
  agentName: z.string().min(1).max(50).transform(InputSanitiser.sanitiseString),
  input: z.string().min(1).max(50000).transform(InputSanitiser.sanitiseString),
  userId: z.string().uuid().optional(),
});

type AgentInput = z.infer<typeof agentInputSchema>;
```

---

## Monitoring & Observability

### Distributed Tracing

```typescript
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/node';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';

const jaegerExporter = new JaegerExporter({
  endpoint: process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces',
});

const provider = new NodeTracerProvider();
provider.addSpanProcessor(new BatchSpanProcessor(jaegerExporter));
trace.setGlobalTracerProvider(provider);

const tracer = trace.getTracer('openai-agents');

async function tracedAgentExecution(agent: Agent, input: string): Promise<string> {
  return await tracer.startActiveSpan(
    'agent-execution',
    async (span) => {
      span.setAttributes({
        'agent.name': agent.name,
        'input.length': input.length,
      });

      try {
        const startTime = performance.now();
        const result = await run(agent, input);

        span.addEvent('execution_completed', {
          'duration_ms': performance.now() - startTime,
          'output_length': result.finalOutput.length,
        });

        return result.finalOutput;
      } catch (error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR });
        throw error;
      }
    }
  );
}
```

### Metrics Collection

```typescript
import { metrics } from '@opentelemetry/api';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';

const prometheusExporter = new PrometheusExporter({
  port: 9090,
});

const meterProvider = new MeterProvider({
  readers: [new PeriodicExportingMetricReader({ exporter: prometheusExporter })],
});

metrics.setGlobalMeterProvider(meterProvider);

class MetricsCollector {
  private meter = metrics.getMeter('openai-agents');

  private executionCounter = this.meter.createCounter('agent.executions.total', {
    description: 'Total number of agent executions',
  });

  private executionDuration = this.meter.createHistogram('agent.execution.duration_ms', {
    description: 'Agent execution duration',
  });

  private errorCounter = this.meter.createCounter('agent.errors.total', {
    description: 'Total number of errors',
  });

  recordExecution(agentName: string, durationMs: number, success: boolean): void {
    this.executionCounter.add(1, {
      'agent.name': agentName,
      'status': success ? 'success' : 'error',
    });

    this.executionDuration.record(durationMs, {
      'agent.name': agentName,
    });

    if (!success) {
      this.errorCounter.add(1, { 'agent.name': agentName });
    }
  }
}

const metricsCollector = new MetricsCollector();
```

---

## Scaling Strategies

### Load Balancing

```typescript
interface AgentPoolConfig {
  agents: Agent[];
  strategy: 'round-robin' | 'least-connections' | 'response-time';
}

class AgentLoadBalancer {
  private currentIndex = 0;
  private connectionCounts = new Map<Agent, number>();
  private responseTimes = new Map<Agent, number[]>();

  constructor(private config: AgentPoolConfig) {
    config.agents.forEach((agent) => {
      this.connectionCounts.set(agent, 0);
      this.responseTimes.set(agent, []);
    });
  }

  selectAgent(): Agent {
    switch (this.config.strategy) {
      case 'round-robin':
        return this.roundRobinSelect();
      case 'least-connections':
        return this.leastConnectionsSelect();
      case 'response-time':
        return this.responseTimeSelect();
      default:
        return this.config.agents[0];
    }
  }

  private roundRobinSelect(): Agent {
    const agent = this.config.agents[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.config.agents.length;
    return agent;
  }

  private leastConnectionsSelect(): Agent {
    return Array.from(this.connectionCounts.entries()).sort(
      (a, b) => a[1] - b[1]
    )[0][0];
  }

  private responseTimeSelect(): Agent {
    const avgTimes = Array.from(this.responseTimes.entries()).map(([agent, times]) => {
      const avg = times.reduce((a, b) => a + b, 0) / times.length || 0;
      return { agent, avgTime: avg };
    });

    return avgTimes.sort((a, b) => a.avgTime - b.avgTime)[0].agent;
  }

  recordExecution(agent: Agent, durationMs: number): void {
    const times = this.responseTimes.get(agent)!;
    times.push(durationMs);
    if (times.length > 100) times.shift();
  }
}
```

### Horizontal Scaling with Message Queues

```typescript
import { Queue } from 'bull';
import redis from 'redis';

interface AgentJob {
  agentName: string;
  input: string;
  userId: string;
  timestamp: number;
}

class AgentQueueWorker {
  private queue: Queue<AgentJob>;
  private agents = new Map<string, Agent>();

  constructor(redisUrl: string) {
    this.queue = new Queue('agent-jobs', redisUrl);
    this.setupWorker();
  }

  private setupWorker(): void {
    this.queue.process(async (job) => {
      const { agentName, input } = job.data;
      const agent = this.agents.get(agentName);

      if (!agent) {
        throw new Error(`Agent not found: ${agentName}`);
      }

      const result = await run(agent, input);
      return result.finalOutput;
    });

    this.queue.on('completed', (job) => {
      console.log(`Job ${job.id} completed`);
    });

    this.queue.on('failed', (job, err) => {
      console.error(`Job ${job.id} failed:`, err);
    });
  }

  async enqueueJob(agentName: string, input: string, userId: string): Promise<string> {
    const job = await this.queue.add({
      agentName,
      input,
      userId,
      timestamp: Date.now(),
    });

    return job.id.toString();
  }

  registerAgent(name: string, agent: Agent): void {
    this.agents.set(name, agent);
  }
}

// Usage
const queueWorker = new AgentQueueWorker(process.env.REDIS_URL!);
queueWorker.registerAgent('ResearchAgent', researchAgent);

const jobId = await queueWorker.enqueueJob('ResearchAgent', 'Research AI trends', 'user-123');
```

---

## Multi-Tenancy

### Tenant Isolation

```typescript
interface TenantContext {
  tenantId: string;
  userId: string;
  permissions: string[];
  quotas: {
    executionsPerMonth: number;
    tokenLimitPerCall: number;
  };
}

class TenantManager {
  private tenants = new Map<string, TenantContext>();

  registerTenant(tenantId: string, context: TenantContext): void {
    this.tenants.set(tenantId, context);
  }

  getTenant(tenantId: string): TenantContext | null {
    return this.tenants.get(tenantId) || null;
  }

  async executeAgentForTenant(
    tenantId: string,
    agentName: string,
    input: string
  ): Promise<string> {
    const tenant = this.getTenant(tenantId);
    if (!tenant) throw new Error('Tenant not found');

    // Check quotas
    const usageKey = `${tenantId}:${new Date().getFullYear()}-${new Date().getMonth()}`;
    const usage = await this.getUsage(usageKey);

    if (usage >= tenant.quotas.executionsPerMonth) {
      throw new Error('Monthly execution quota exceeded');
    }

    const agent = this.getAgentForTenant(tenantId, agentName);
    const result = await run(agent, input);

    await this.recordUsage(usageKey);

    return result.finalOutput;
  }

  private async getUsage(key: string): Promise<number> {
    // Implementation with Redis/database
    return 0;
  }

  private async recordUsage(key: string): Promise<void> {
    // Implementation with Redis/database
  }

  private getAgentForTenant(tenantId: string, agentName: string): Agent {
    // Return tenant-specific agent configuration
    return new Agent({ name: agentName, instructions: 'Agent instructions' });
  }
}
```

---

## Testing Strategies

### Unit Testing with Jest

```typescript
import { describe, it, expect, beforeEach } from '@jest/globals';
import { Agent, run } from '@openai/agents';

describe('Agent Execution', () => {
  let agent: Agent;

  beforeEach(() => {
    agent = new Agent({
      name: 'Test Agent',
      instructions: 'You are a test agent.',
    });
  });

  it('should return a non-empty result', async () => {
    const result = await run(agent, 'What is 2 + 2?');
    expect(result.finalOutput).toBeTruthy();
    expect(typeof result.finalOutput).toBe('string');
  });

  it('should handle errors gracefully', async () => {
    await expect(run(agent, '')).rejects.toThrow();
  });

  it('should respect timeout', async () => {
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Timeout')), 5000)
    );

    await expect(Promise.race([run(agent, 'Long query'), timeoutPromise])).rejects.toThrow();
  });
});
```

---

## Conclusion

This production guide provides enterprise-grade patterns for deploying, scaling, and maintaining OpenAI Agents SDK applications in production environments with focus on reliability, security, and operational excellence.

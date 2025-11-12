# Claude Agent SDK (TypeScript) - Production Implementation Guide

## Production-Ready Architecture and Deployment

This guide covers hardened, production-ready patterns for deploying Claude Agent SDK applications at scale.

### Error Handling Strategies for Production

**Comprehensive Error Management:**

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import * as Sentry from '@sentry/node';
import pino from 'pino';

// Initialize error tracking
Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 0.1
});

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true
    }
  }
});

// Error classification
type ErrorSeverity = 'critical' | 'high' | 'medium' | 'low';

interface ProductionError {
  id: string;
  code: string;
  message: string;
  severity: ErrorSeverity;
  context: Record<string, unknown>;
  timestamp: number;
  stackTrace?: string;
  userId?: string;
  sessionId?: string;
}

class ProductionErrorHandler {
  private errorCounts: Map<string, number> = new Map();
  private circuitBreakerThreshold = 5;
  private circuitBreakerWindow = 60000; // 1 minute

  async handleQueryError(
    error: Error,
    context: Record<string, unknown>
  ): Promise<ProductionError> {
    const prodError: ProductionError = {
      id: this.generateErrorId(),
      code: this.classifyErrorCode(error),
      message: error.message,
      severity: this.determineSeverity(error),
      context,
      timestamp: Date.now(),
      stackTrace: error.stack
    };

    // Log error
    logger.error(prodError, 'Query execution failed');

    // Report to error tracking
    Sentry.captureException(error, {
      contexts: context as Record<string, unknown>,
      level: this.mapSeverityToSentryLevel(prodError.severity),
      tags: {
        errorCode: prodError.code,
        errorId: prodError.id
      }
    });

    // Check circuit breaker
    this.updateCircuitBreaker(prodError.code);

    // Store error for later analysis
    await this.storeError(prodError);

    return prodError;
  }

  private classifyErrorCode(error: Error): string {
    if (error.message.includes('authentication')) {
      return 'AUTHENTICATION_ERROR';
    } else if (error.message.includes('permission')) {
      return 'PERMISSION_DENIED';
    } else if (error.message.includes('timeout')) {
      return 'TIMEOUT_ERROR';
    } else if (error.message.includes('rate limit')) {
      return 'RATE_LIMIT_EXCEEDED';
    } else if (error.message.includes('budget')) {
      return 'BUDGET_EXCEEDED';
    }
    return 'UNKNOWN_ERROR';
  }

  private determineSeverity(error: Error): ErrorSeverity {
    const code = this.classifyErrorCode(error);

    const severityMap: Record<string, ErrorSeverity> = {
      'AUTHENTICATION_ERROR': 'critical',
      'PERMISSION_DENIED': 'high',
      'TIMEOUT_ERROR': 'medium',
      'RATE_LIMIT_EXCEEDED': 'medium',
      'BUDGET_EXCEEDED': 'high',
      'UNKNOWN_ERROR': 'low'
    };

    return severityMap[code] || 'low';
  }

  private mapSeverityToSentryLevel(severity: ErrorSeverity): 'fatal' | 'error' | 'warning' | 'info' {
    const levelMap: Record<ErrorSeverity, 'fatal' | 'error' | 'warning' | 'info'> = {
      'critical': 'fatal',
      'high': 'error',
      'medium': 'warning',
      'low': 'info'
    };

    return levelMap[severity];
  }

  private generateErrorId(): string {
    return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private updateCircuitBreaker(errorCode: string): void {
    const count = this.errorCounts.get(errorCode) || 0;
    this.errorCounts.set(errorCode, count + 1);

    if (count >= this.circuitBreakerThreshold) {
      logger.warn(`Circuit breaker triggered for ${errorCode}`);
      Sentry.captureMessage(`Circuit breaker triggered for ${errorCode}`, 'warning');

      setTimeout(() => {
        this.errorCounts.set(errorCode, 0);
      }, this.circuitBreakerWindow);
    }
  }

  private async storeError(error: ProductionError): Promise<void> {
    // Store in database or error tracking system
    logger.debug({ error }, 'Storing error for analysis');
  }

  isCircuitBreakerOpen(errorCode: string): boolean {
    const count = this.errorCounts.get(errorCode) || 0;
    return count >= this.circuitBreakerThreshold;
  }
}

// Usage
const errorHandler = new ProductionErrorHandler();

async function produceQueryWithErrorHandling(prompt: string) {
  try {
    const response = query({ prompt });

    for await (const message of response) {
      if (message.type === 'error') {
        const prodError = await errorHandler.handleQueryError(
          new Error(message.error),
          { prompt, messageType: message.type }
        );

        if (errorHandler.isCircuitBreakerOpen(prodError.code)) {
          throw new Error('Circuit breaker is open. Service temporarily unavailable.');
        }
      }
    }
  } catch (error) {
    const prodError = await errorHandler.handleQueryError(
      error as Error,
      { prompt }
    );

    // Gracefully handle or rethrow
    if (prodError.severity === 'critical') {
      throw error;
    }
  }
}
```

### Rate Limiting and Quota Management

**Production-Grade Rate Limiting:**

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import Redis from 'ioredis';

interface RateLimitConfig {
  requestsPerMinute: number;
  requestsPerHour: number;
  tokensPerDay: number;
  maxConcurrentRequests: number;
}

class RateLimiter {
  private redis: Redis;
  private config: RateLimitConfig;

  constructor(config: RateLimitConfig) {
    this.redis = new Redis(process.env.REDIS_URL);
    this.config = config;
  }

  async checkRateLimit(userId: string): Promise<{
    allowed: boolean;
    remaining: number;
    resetTime: number;
  }> {
    const keys = {
      minute: `ratelimit:${userId}:minute`,
      hour: `ratelimit:${userId}:hour`,
      day: `ratelimit:${userId}:day`,
      concurrent: `ratelimit:${userId}:concurrent`
    };

    const [minuteCount, hourCount, dayTokens, concurrentRequests] = await Promise.all([
      this.redis.incr(keys.minute),
      this.redis.incr(keys.hour),
      this.redis.incrby(keys.day, 0),
      this.redis.incr(keys.concurrent)
    ]);

    // Set expiry on first increment
    await this.redis.expire(keys.minute, 60);
    await this.redis.expire(keys.hour, 3600);
    await this.redis.expire(keys.day, 86400);

    const allowed =
      minuteCount <= this.config.requestsPerMinute &&
      hourCount <= this.config.requestsPerHour &&
      dayTokens <= this.config.tokensPerDay &&
      concurrentRequests <= this.config.maxConcurrentRequests;

    const resetTime = Math.min(
      (await this.redis.ttl(keys.minute)) * 1000,
      (await this.redis.ttl(keys.hour)) * 1000
    );

    return {
      allowed,
      remaining: this.config.requestsPerMinute - minuteCount,
      resetTime: resetTime > 0 ? Date.now() + resetTime : 0
    };
  }

  async trackTokenUsage(userId: string, tokens: number): Promise<void> {
    const key = `ratelimit:${userId}:day:tokens`;
    await this.redis.incrby(key, tokens);
    await this.redis.expire(key, 86400);
  }

  async decrementConcurrent(userId: string): Promise<void> {
    const key = `ratelimit:${userId}:concurrent`;
    await this.redis.decr(key);
  }

  async cleanup(): Promise<void> {
    await this.redis.quit();
  }
}

// Middleware for Express
function rateLimitMiddleware(limiter: RateLimiter) {
  return async (req: any, res: any, next: any) => {
    const userId = req.user?.id || req.ip;
    const { allowed, remaining, resetTime } = await limiter.checkRateLimit(userId);

    res.setHeader('X-RateLimit-Remaining', remaining);
    res.setHeader('X-RateLimit-Reset', resetTime);

    if (!allowed) {
      return res.status(429).json({
        error: 'Rate limit exceeded',
        resetTime: new Date(resetTime).toISOString()
      });
    }

    req.limiter = limiter;
    next();
  };
}

// Usage
const rateLimiter = new RateLimiter({
  requestsPerMinute: 30,
  requestsPerHour: 500,
  tokensPerDay: 1000000,
  maxConcurrentRequests: 5
});
```

### Cost Optimisation and Budget Management

**Cost Control Implementation:**

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import type { QueryOptions } from '@anthropic-ai/claude-agent-sdk';

interface CostTracker {
  userId: string;
  dailyBudget: number;
  monthlyBudget: number;
  spent: number;
  monthlySpent: number;
}

class CostOptimiser {
  private trackers: Map<string, CostTracker> = new Map();

  // Model pricing (update based on current rates)
  private modelPricing = {
    'claude-sonnet-4-5': {
      input: 0.003,
      output: 0.015,
      cacheCreation: 0.00375,
      cacheRead: 0.0003
    },
    'claude-3-5-sonnet': {
      input: 0.003,
      output: 0.015
    },
    'claude-opus': {
      input: 0.015,
      output: 0.075
    },
    'claude-3-5-haiku': {
      input: 0.00008,
      output: 0.0004
    }
  };

  selectOptimalModel(
    complexity: 'simple' | 'moderate' | 'complex',
    budget: number
  ): string {
    // Select model based on complexity and budget
    if (complexity === 'simple' && budget < 1) {
      return 'claude-3-5-haiku';
    } else if (complexity === 'moderate' && budget < 5) {
      return 'claude-3-5-sonnet';
    } else if (complexity === 'complex' || budget >= 10) {
      return 'claude-sonnet-4-5';
    }

    return 'claude-3-5-sonnet';
  }

  async queryWithBudgetControl(
    prompt: string,
    userId: string,
    dailyBudget: number = 10.0
  ): Promise<string> {
    const tracker = this.getOrCreateTracker(userId, dailyBudget);

    // Check if budget exceeded
    if (tracker.spent >= tracker.dailyBudget) {
      throw new Error(
        `Daily budget of $${tracker.dailyBudget} exceeded. ` +
        `Already spent: $${tracker.spent.toFixed(4)}`
      );
    }

    // Calculate remaining budget
    const remaining = tracker.dailyBudget - tracker.spent;

    // Estimate tokens and cost
    const estimatedInputTokens = Math.ceil(prompt.length / 4);
    const estimatedCost = (estimatedInputTokens * 0.003) / 1000; // approximate

    if (estimatedCost > remaining) {
      throw new Error(
        `Estimated cost $${estimatedCost.toFixed(4)} exceeds remaining budget $${remaining.toFixed(4)}`
      );
    }

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        maxBudgetUsd: remaining
      }
    });

    let output = '';
    let actualCost = 0;

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      } else if (message.type === 'system' && message.cost) {
        actualCost += message.cost;
      }
    }

    tracker.spent += actualCost;
    tracker.monthlySpent += actualCost;

    return output;
  }

  private getOrCreateTracker(userId: string, dailyBudget: number): CostTracker {
    if (!this.trackers.has(userId)) {
      this.trackers.set(userId, {
        userId,
        dailyBudget,
        monthlyBudget: dailyBudget * 30,
        spent: 0,
        monthlySpent: 0
      });
    }

    return this.trackers.get(userId)!;
  }

  getCostReport(userId: string): {
    dailyBudget: number;
    dailySpent: number;
    dailyRemaining: number;
    monthlyBudget: number;
    monthlySpent: number;
    monthlyRemaining: number;
    projectedMonthly: number;
  } {
    const tracker = this.trackers.get(userId);

    if (!tracker) {
      throw new Error(`No tracker found for user ${userId}`);
    }

    const dailyRemaining = tracker.dailyBudget - tracker.spent;
    const monthlyRemaining = tracker.monthlyBudget - tracker.monthlySpent;
    const projectedMonthly = tracker.monthlySpent * (30 / (new Date().getDate()));

    return {
      dailyBudget: tracker.dailyBudget,
      dailySpent: tracker.spent,
      dailyRemaining,
      monthlyBudget: tracker.monthlyBudget,
      monthlySpent: tracker.monthlySpent,
      monthlyRemaining,
      projectedMonthly
    };
  }

  resetDailyBudget(userId: string): void {
    const tracker = this.trackers.get(userId);
    if (tracker) {
      tracker.spent = 0;
    }
  }
}

// Usage
const costOptimiser = new CostOptimiser();

const output = await costOptimiser.queryWithBudgetControl(
  'Analyse this code for performance issues',
  'user123',
  5.0
);

const report = costOptimiser.getCostReport('user123');
console.log('Cost Report:', report);
```

### Monitoring and Logging

**Production-Grade Monitoring:**

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import pino from 'pino';
import prometheus from 'prom-client';

// Setup logging
const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-transport-http'
  }
});

// Prometheus metrics
const metrics = {
  queriesTotal: new prometheus.Counter({
    name: 'agent_queries_total',
    help: 'Total number of agent queries',
    labelNames: ['model', 'status']
  }),

  queryDuration: new prometheus.Histogram({
    name: 'agent_query_duration_seconds',
    help: 'Duration of agent queries',
    labelNames: ['model'],
    buckets: [0.1, 0.5, 1, 5, 10]
  }),

  tokensUsed: new prometheus.Counter({
    name: 'agent_tokens_used_total',
    help: 'Total tokens used',
    labelNames: ['model', 'type']
  }),

  costTotal: new prometheus.Counter({
    name: 'agent_cost_usd_total',
    help: 'Total cost in USD',
    labelNames: ['model']
  }),

  concurrentQueries: new prometheus.Gauge({
    name: 'agent_concurrent_queries',
    help: 'Current number of concurrent queries'
  }),

  errors: new prometheus.Counter({
    name: 'agent_errors_total',
    help: 'Total number of errors',
    labelNames: ['error_code']
  })
};

class ProductionMonitor {
  async executeMonitoredQuery(
    prompt: string,
    model: string = 'claude-sonnet-4-5'
  ): Promise<{ output: string; metrics: Record<string, number> }> {
    const startTime = Date.now();
    metrics.concurrentQueries.inc();

    const queryMetrics = {
      duration: 0,
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      cost: 0
    };

    try {
      const response = query({
        prompt,
        options: { model }
      });

      let output = '';

      for await (const message of response) {
        if (message.type === 'assistant') {
          output += message.content;
          queryMetrics.outputTokens += Math.ceil(message.content.length / 4);
        } else if (message.type === 'error') {
          metrics.errors.labels(message.error.type).inc();
          logger.error({ error: message.error }, 'Query error');
        }
      }

      queryMetrics.duration = Date.now() - startTime;
      queryMetrics.inputTokens = Math.ceil(prompt.length / 4);
      queryMetrics.totalTokens = queryMetrics.inputTokens + queryMetrics.outputTokens;

      // Estimate cost
      const pricing = this.getPricing(model);
      queryMetrics.cost =
        (queryMetrics.inputTokens * pricing.input +
          queryMetrics.outputTokens * pricing.output) / 1000;

      // Update metrics
      metrics.queriesTotal.labels(model, 'success').inc();
      metrics.queryDuration.labels(model).observe(queryMetrics.duration / 1000);
      metrics.tokensUsed.labels(model, 'input').inc(queryMetrics.inputTokens);
      metrics.tokensUsed.labels(model, 'output').inc(queryMetrics.outputTokens);
      metrics.costTotal.labels(model).inc(queryMetrics.cost);

      logger.info(
        {
          model,
          duration: queryMetrics.duration,
          tokens: queryMetrics.totalTokens,
          cost: queryMetrics.cost
        },
        'Query completed'
      );

      return { output, metrics: queryMetrics };
    } catch (error) {
      metrics.queriesTotal.labels(model, 'error').inc();
      metrics.errors.labels('query_exception').inc();

      logger.error(
        {
          error: error instanceof Error ? error.message : String(error),
          model,
          duration: Date.now() - startTime
        },
        'Query failed'
      );

      throw error;
    } finally {
      metrics.concurrentQueries.dec();
    }
  }

  private getPricing(model: string): { input: number; output: number } {
    const pricing: Record<string, { input: number; output: number }> = {
      'claude-sonnet-4-5': { input: 3, output: 15 },
      'claude-3-5-sonnet': { input: 3, output: 15 },
      'claude-opus': { input: 15, output: 75 },
      'claude-3-5-haiku': { input: 0.08, output: 0.4 }
    };

    return pricing[model] || { input: 3, output: 15 };
  }

  getMetrics(): string {
    return prometheus.register.metrics();
  }
}

// Usage with Express
import express from 'express';

const app = express();
const monitor = new ProductionMonitor();

app.post('/query', async (req, res) => {
  try {
    const { prompt, model } = req.body;
    const { output, metrics } = await monitor.executeMonitoredQuery(prompt, model);

    res.json({ output, metrics });
  } catch (error) {
    res.status(500).json({ error: (error as Error).message });
  }
});

app.get('/metrics', (_req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end(monitor.getMetrics());
});

app.listen(3000);
```

### Performance Tuning

**Performance Optimisation Strategies:**

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

interface PerformanceConfig {
  enableCaching: boolean;
  enablePromptCaching: boolean;
  maxConcurrentRequests: number;
  requestTimeout: number;
  modelSelectionStrategy: 'cost' | 'speed' | 'balanced';
}

class PerformanceTuner {
  private cache: Map<string, { result: string; timestamp: number }> = new Map();
  private cacheMaxAge = 3600000; // 1 hour

  async optimisedQuery(
    prompt: string,
    config: PerformanceConfig
  ): Promise<{ result: string; fromCache: boolean; executionTime: number }> {
    const startTime = Date.now();

    // Check cache
    if (config.enableCaching) {
      const cached = this.getFromCache(prompt);
      if (cached) {
        return {
          result: cached,
          fromCache: true,
          executionTime: Date.now() - startTime
        };
      }
    }

    // Select optimal model
    const model = this.selectModel(config.modelSelectionStrategy);

    // Execute with timeout
    const timeoutPromise = new Promise<never>((_resolve, reject) => {
      setTimeout(
        () => reject(new Error('Query timeout')),
        config.requestTimeout
      );
    });

    const queryPromise = (async () => {
      const response = query({
        prompt,
        options: {
          model,
          maxTokens: this.getMaxTokens(config.modelSelectionStrategy)
        }
      });

      let result = '';

      for await (const message of response) {
        if (message.type === 'assistant') {
          result += message.content;
        }
      }

      return result;
    })();

    const result = await Promise.race([queryPromise, timeoutPromise]);

    // Cache result
    if (config.enableCaching) {
      this.cacheResult(prompt, result);
    }

    return {
      result,
      fromCache: false,
      executionTime: Date.now() - startTime
    };
  }

  private selectModel(strategy: string): string {
    switch (strategy) {
      case 'cost':
        return 'claude-3-5-haiku';
      case 'speed':
        return 'claude-3-5-haiku';
      case 'balanced':
      default:
        return 'claude-sonnet-4-5';
    }
  }

  private getMaxTokens(strategy: string): number {
    switch (strategy) {
      case 'cost':
        return 2048;
      case 'speed':
        return 2048;
      case 'balanced':
      default:
        return 4096;
    }
  }

  private getFromCache(prompt: string): string | null {
    const cached = this.cache.get(prompt);

    if (!cached) {
      return null;
    }

    // Check if expired
    if (Date.now() - cached.timestamp > this.cacheMaxAge) {
      this.cache.delete(prompt);
      return null;
    }

    return cached.result;
  }

  private cacheResult(prompt: string, result: string): void {
    this.cache.set(prompt, {
      result,
      timestamp: Date.now()
    });
  }

  clearCache(): void {
    this.cache.clear();
  }

  getCacheSize(): number {
    return this.cache.size;
  }
}

// Usage
const tuner = new PerformanceTuner();

const { result, fromCache, executionTime } = await tuner.optimisedQuery(
  'What is TypeScript?',
  {
    enableCaching: true,
    enablePromptCaching: true,
    maxConcurrentRequests: 10,
    requestTimeout: 30000,
    modelSelectionStrategy: 'balanced'
  }
);

console.log(`Result: ${result}`);
console.log(`From cache: ${fromCache}`);
console.log(`Execution time: ${executionTime}ms`);
```

### Deployment Patterns (Docker, Kubernetes)

**Docker Deployment:**

```dockerfile
# Dockerfile
FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy source
COPY src ./src
COPY tsconfig.json ./

# Build
RUN npm run build

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"

# Run
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

**Kubernetes Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  template:
    metadata:
      labels:
        app: agent-api
    spec:
      containers:
      - name: agent-api
        image: myregistry.azurecr.io/agent-api:1.0.0
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-secrets
              key: api-key
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
  name: agent-api-service
  namespace: production
spec:
  selector:
    app: agent-api
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-api
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

### Load Balancing and High Availability

**Production HA Setup:**

```typescript
import express from 'express';
import http from 'http';
import cluster from 'cluster';
import os from 'os';

const app = express();
const PORT = process.env.PORT || 3000;
const numCPUs = os.cpus().length;

if (cluster.isPrimary) {
  console.log(`Primary ${process.pid} is running`);

  // Fork workers
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker) => {
    console.log(`Worker ${worker.process.pid} died`);
    // Restart worker
    cluster.fork();
  });
} else {
  // Worker process
  app.get('/health', (_req, res) => {
    res.json({ status: 'ok', pid: process.pid });
  });

  app.post('/query', async (req, res) => {
    // Handle query
    res.json({ result: 'processed' });
  });

  const server = http.createServer(app);

  server.listen(PORT, () => {
    console.log(`Worker ${process.pid} listening on port ${PORT}`);
  });

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully');
    server.close(() => {
      process.exit(0);
    });
  });
}
```

### Environment Variable Management

**Configuration Management:**

```typescript
import { z } from 'zod';
import dotenv from 'dotenv';

dotenv.config();

// Schema for environment variables
const EnvSchema = z.object({
  NODE_ENV: z.enum(['development', 'staging', 'production']).default('development'),
  PORT: z.coerce.number().default(3000),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // API Configuration
  ANTHROPIC_API_KEY: z.string().min(1),
  MAX_REQUEST_TIMEOUT: z.coerce.number().default(30000),
  MAX_RETRIES: z.coerce.number().default(3),

  // Cost Management
  MAX_DAILY_BUDGET_USD: z.coerce.number().default(100),
  MAX_MONTHLY_BUDGET_USD: z.coerce.number().default(3000),

  // Rate Limiting
  REQUESTS_PER_MINUTE: z.coerce.number().default(30),
  REQUESTS_PER_HOUR: z.coerce.number().default(500),

  // Database
  DATABASE_URL: z.string().optional(),
  REDIS_URL: z.string().optional(),

  // Monitoring
  SENTRY_DSN: z.string().optional(),
  PROMETHEUS_ENABLED: z.coerce.boolean().default(true),

  // Authentication
  JWT_SECRET: z.string().optional(),
  API_KEY: z.string().optional()
});

type Environment = z.infer<typeof EnvSchema>;

let env: Environment;

export function initEnvironment(): Environment {
  try {
    env = EnvSchema.parse(process.env);
    return env;
  } catch (error) {
    console.error('Invalid environment configuration:', error);
    process.exit(1);
  }
}

export function getEnv(): Environment {
  if (!env) {
    throw new Error('Environment not initialised. Call initEnvironment() first.');
  }
  return env;
}

// Usage
const environment = initEnvironment();
console.log(`Running in ${environment.NODE_ENV} mode`);
console.log(`API timeout: ${environment.MAX_REQUEST_TIMEOUT}ms`);
console.log(`Daily budget: $${environment.MAX_DAILY_BUDGET_USD}`);
```

### Security Hardening

**Security Best Practices:**

```typescript
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import cors from 'cors';
import express from 'express';

const app = express();

// Security headers
app.use(helmet());

// CORS configuration
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || [],
  credentials: true,
  methods: ['GET', 'POST'],
  maxAge: 3600
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // 100 requests per windowMs
  message: 'Too many requests from this IP'
});

app.use('/api/', limiter);

// Input validation
import { z } from 'zod';

const QueryRequestSchema = z.object({
  prompt: z.string().min(1).max(10000),
  model: z.string().optional(),
  temperature: z.number().min(0).max(2).optional(),
  maxTokens: z.number().min(1).max(8192).optional()
});

app.post('/api/query', express.json(), async (req, res) => {
  try {
    const request = QueryRequestSchema.parse(req.body);

    // Process request safely
    // ...

    res.json({ success: true });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.errors });
    }
    res.status(500).json({ error: 'Internal server error' });
  }
});

// API key validation
function validateApiKey(req: any, res: any, next: any) {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey || apiKey !== process.env.API_KEY) {
    return res.status(401).json({ error: 'Invalid API key' });
  }

  next();
}

app.use('/api/protected/', validateApiKey);

// Sensitive data logging
function sanitiseForLogging(obj: any): any {
  const sensitiveKeys = ['password', 'apiKey', 'secret', 'token'];

  const sanitised = { ...obj };

  for (const key of sensitiveKeys) {
    if (key in sanitised) {
      sanitised[key] = '[REDACTED]';
    }
  }

  return sanitised;
}
```

---

## Advanced Production Patterns

### Database Integration

**Production-Ready Database Layer:**

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import { Prisma, PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

interface StoredQuery {
  id: string;
  userId: string;
  prompt: string;
  response: string;
  model: string;
  tokensUsed: number;
  costUsd: number;
  createdAt: Date;
  executedAt: Date;
}

class QueryDatabase {
  async storeQuery(data: Omit<StoredQuery, 'id' | 'createdAt'>): Promise<StoredQuery> {
    return prisma.agentQuery.create({
      data: {
        ...data,
        id: `query_${Date.now()}`
      }
    });
  }

  async getUserQueryHistory(userId: string, limit: number = 50): Promise<StoredQuery[]> {
    return prisma.agentQuery.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: limit
    });
  }

  async aggregateUserMetrics(userId: string) {
    return prisma.agentQuery.aggregate({
      where: { userId },
      _count: true,
      _sum: {
        tokensUsed: true,
        costUsd: true
      },
      _avg: {
        costUsd: true
      }
    });
  }
}

// Usage
const queryDb = new QueryDatabase();

async function trackQuery(prompt: string, userId: string) {
  const response = query({ prompt });

  let output = '';
  let tokensUsed = 0;
  let cost = 0;

  for await (const message of response) {
    if (message.type === 'assistant') {
      output += message.content;
      tokensUsed += Math.ceil(message.content.length / 4);
    }
  }

  await queryDb.storeQuery({
    userId,
    prompt,
    response: output,
    model: 'claude-sonnet-4-5',
    tokensUsed,
    costUsd: cost,
    executedAt: new Date()
  });
}
```

This production guide provides the foundation for hardened, scalable Claude Agent SDK deployments.


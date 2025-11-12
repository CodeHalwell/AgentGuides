# LangChain.js and LangGraph.js Production Guide

**Deployment, Operations, Monitoring, and Best Practices**

---

## Table of Contents

1. [Environment Configuration](#environment-configuration)
2. [Performance Optimisation](#performance-optimisation)
3. [Security Best Practices](#security-best-practices)
4. [Database and Persistence](#database-and-persistence)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Error Handling and Recovery](#error-handling-and-recovery)
7. [Deployment Patterns](#deployment-patterns)
8. [Scaling Considerations](#scaling-considerations)
9. [Cost Optimisation](#cost-optimisation)
10. [Testing Strategies](#testing-strategies)
11. [CI/CD Integration](#cicd-integration)
12. [LangGraph Studio Setup](#langgraph-studio-setup)

---

## Environment Configuration

### Multi-Environment Setup

```typescript
// config/environment.ts
import dotenv from 'dotenv';

dotenv.config();

type Environment = 'development' | 'staging' | 'production';

export interface EnvironmentConfig {
  env: Environment;
  nodeEnv: string;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  
  // API Configuration
  apis: {
    openai: {
      apiKey: string;
      model: string;
      maxTokens: number;
      timeout: number;
    };
    anthropic: {
      apiKey: string;
      model: string;
    };
  };
  
  // Database Configuration
  database: {
    postgres: {
      host: string;
      port: number;
      username: string;
      password: string;
      database: string;
      ssl: boolean;
    };
    redis: {
      host: string;
      port: number;
      password?: string;
    };
  };
  
  // LangChain Configuration
  langchain: {
    langsmithKey?: string;
    langsmithProject?: string;
    tracingEnabled: boolean;
  };
  
  // Application Configuration
  app: {
    port: number;
    corsOrigins: string[];
    rateLimitWindow: number;
    rateLimitRequests: number;
  };
}

function validateEnvironmentVariables(): EnvironmentConfig {
  const env = (process.env.NODE_ENV || 'development') as Environment;
  
  // Validate required variables based on environment
  const requiredVars: Record<Environment, string[]> = {
    development: ['OPENAI_API_KEY'],
    staging: ['OPENAI_API_KEY', 'POSTGRES_URL', 'LANGSMITH_API_KEY'],
    production: [
      'OPENAI_API_KEY',
      'POSTGRES_URL',
      'REDIS_URL',
      'LANGSMITH_API_KEY',
      'DATABASE_SSL',
    ],
  };
  
  const missing = requiredVars[env].filter((v) => !process.env[v]);
  if (missing.length > 0) {
    throw new Error(
      `Missing required environment variables for ${env}: ${missing.join(', ')}`
    );
  }
  
  return {
    env,
    nodeEnv: process.env.NODE_ENV || 'development',
    logLevel: (process.env.LOG_LEVEL || 'info') as any,
    
    apis: {
      openai: {
        apiKey: process.env.OPENAI_API_KEY!,
        model: process.env.OPENAI_MODEL || 'gpt-4-turbo',
        maxTokens: parseInt(process.env.OPENAI_MAX_TOKENS || '4000'),
        timeout: parseInt(process.env.OPENAI_TIMEOUT || '30000'),
      },
      anthropic: {
        apiKey: process.env.ANTHROPIC_API_KEY || '',
        model: process.env.ANTHROPIC_MODEL || 'claude-3-opus-20240229',
      },
    },
    
    database: {
      postgres: {
        host: process.env.POSTGRES_HOST || 'localhost',
        port: parseInt(process.env.POSTGRES_PORT || '5432'),
        username: process.env.POSTGRES_USER || 'postgres',
        password: process.env.POSTGRES_PASSWORD || '',
        database: process.env.POSTGRES_DB || 'langchain_app',
        ssl: process.env.DATABASE_SSL === 'true',
      },
      redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
        password: process.env.REDIS_PASSWORD,
      },
    },
    
    langchain: {
      langsmithKey: process.env.LANGSMITH_API_KEY,
      langsmithProject: process.env.LANGSMITH_PROJECT,
      tracingEnabled: process.env.LANGSMITH_TRACING_V2 === 'true',
    },
    
    app: {
      port: parseInt(process.env.PORT || '3000'),
      corsOrigins: (process.env.CORS_ORIGINS || 'http://localhost:3000').split(','),
      rateLimitWindow: parseInt(process.env.RATE_LIMIT_WINDOW || '60000'),
      rateLimitRequests: parseInt(process.env.RATE_LIMIT_REQUESTS || '100'),
    },
  };
}

export const config = validateEnvironmentVariables();
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY src ./src
COPY dist ./dist

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"

# Start application
CMD ["node", "dist/index.js"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: production
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      POSTGRES_URL: postgresql://user:password@postgres:5432/langchain
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: langchain
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
```

---

## Performance Optimisation

### Caching Strategy

```typescript
// utils/cache.ts
import Redis from 'ioredis';

interface CacheOptions {
  ttl: number;
  keyPrefix: string;
}

export class CacheManager {
  private redis: Redis;
  
  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl);
  }
  
  private generateKey(prefix: string, ...parts: string[]): string {
    return [prefix, ...parts].join(':');
  }
  
  async get<T>(prefix: string, ...parts: string[]): Promise<T | null> {
    const key = this.generateKey(prefix, ...parts);
    const value = await this.redis.get(key);
    return value ? JSON.parse(value) : null;
  }
  
  async set<T>(
    value: T,
    ttl: number,
    prefix: string,
    ...parts: string[]
  ): Promise<void> {
    const key = this.generateKey(prefix, ...parts);
    await this.redis.setex(key, ttl, JSON.stringify(value));
  }
  
  async delete(prefix: string, ...parts: string[]): Promise<void> {
    const key = this.generateKey(prefix, ...parts);
    await this.redis.del(key);
  }
  
  async invalidatePattern(pattern: string): Promise<void> {
    const keys = await this.redis.keys(pattern);
    if (keys.length > 0) {
      await this.redis.del(...keys);
    }
  }
}

// Usage in agent execution
export async function executeAgentWithCaching(
  input: string,
  cache: CacheManager,
  agent: any
): Promise<string> {
  // Check cache first
  const cachedResult = await cache.get<string>(
    'agent_results',
    input
  );
  
  if (cachedResult) {
    console.log('Cache hit');
    return cachedResult;
  }
  
  // Execute agent
  const result = await agent.invoke({ input });
  
  // Cache result for 1 hour
  await cache.set(
    result,
    3600,
    'agent_results',
    input
  );
  
  return result;
}
```

### Token Optimisation

```typescript
// utils/tokenCounter.ts
import { encoding_for_model } from 'js-tiktoken';

export class TokenCounter {
  private encoding: any;
  
  constructor(modelName: string = 'gpt-4') {
    this.encoding = encoding_for_model(modelName);
  }
  
  countTokens(text: string): number {
    return this.encoding.encode(text).length;
  }
  
  countMessagesTokens(messages: any[]): number {
    let tokensPerMessage = 4; // for gpt-4
    let tokenCount = 0;
    
    for (const message of messages) {
      tokenCount += tokensPerMessage;
      for (const key in message) {
        tokenCount += this.countTokens(message[key]);
      }
    }
    
    return tokenCount;
  }
  
  estimateCost(
    inputTokens: number,
    outputTokens: number,
    model: string
  ): number {
    const pricing: Record<string, { input: number; output: number }> = {
      'gpt-4-turbo': { input: 0.01, output: 0.03 },
      'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
      'claude-3-opus': { input: 0.015, output: 0.075 },
    };
    
    const rates = pricing[model] || { input: 0.001, output: 0.002 };
    return (inputTokens * rates.input + outputTokens * rates.output) / 1000;
  }
}

// Usage in agent
export async function executeWithTokenTracking(
  agent: any,
  input: string,
  counter: TokenCounter
): Promise<{ result: string; tokensUsed: number; costUSD: number }> {
  const inputTokens = counter.countTokens(input);
  
  console.log(`Input tokens: ${inputTokens}`);
  
  if (inputTokens > 8000) {
    console.warn('Input exceeds recommended token limit');
    // Consider summarisation or truncation
  }
  
  const startTime = Date.now();
  const result = await agent.invoke({ input });
  const duration = Date.now() - startTime;
  
  const outputTokens = counter.countTokens(result);
  const totalTokens = inputTokens + outputTokens;
  const cost = counter.estimateCost(inputTokens, outputTokens, 'gpt-4-turbo');
  
  console.log(`
    Total tokens: ${totalTokens}
    Cost (USD): $${cost.toFixed(4)}
    Duration: ${duration}ms
  `);
  
  return {
    result,
    tokensUsed: totalTokens,
    costUSD: cost,
  };
}
```

### Batch Processing

```typescript
// utils/batchProcessor.ts
interface BatchConfig {
  batchSize: number;
  maxConcurrency: number;
  timeoutPerItem: number;
}

export class BatchProcessor {
  constructor(private config: BatchConfig) {}
  
  async processBatch<T, R>(
    items: T[],
    processor: (item: T) => Promise<R>
  ): Promise<(R | Error)[]> {
    const results: (R | Error)[] = [];
    
    for (let i = 0; i < items.length; i += this.config.batchSize) {
      const batch = items.slice(i, i + this.config.batchSize);
      
      const batchPromises = batch.map((item) =>
        Promise.race([
          processor(item),
          new Promise<Error>((_, reject) =>
            setTimeout(
              () => reject(new Error('Processing timeout')),
              this.config.timeoutPerItem
            )
          ),
        ]).catch((error) => error)
      );
      
      // Process with concurrency limit
      const concurrencyLimited = this.limitConcurrency(
        batchPromises,
        this.config.maxConcurrency
      );
      
      const batchResults = await Promise.all(concurrencyLimited);
      results.push(...batchResults);
      
      // Progress logging
      console.log(`Processed ${Math.min(i + this.config.batchSize, items.length)}/${items.length}`);
    }
    
    return results;
  }
  
  private async limitConcurrency<T>(
    promises: Promise<T>[],
    concurrency: number
  ): Promise<T[]> {
    const results: T[] = [];
    const executing: Promise<any>[] = [];
    
    for (const promise of promises) {
      const exec = promise.then((result) => {
        executing.splice(executing.indexOf(exec), 1);
        return result;
      });
      
      results.push(exec as any);
      
      if (promises.length >= concurrency) {
        if (executing.length >= concurrency) {
          await Promise.race(executing);
        }
        executing.push(exec);
      }
    }
    
    return Promise.all(results);
  }
}
```

---

## Security Best Practices

### API Key Management

```typescript
// security/secretManager.ts
import * as crypto from 'crypto';

export class SecretManager {
  private encryptionKey: Buffer;
  
  constructor(masterKey: string) {
    // Derive encryption key from master key
    this.encryptionKey = crypto
      .pbkdf2Sync(masterKey, 'salt', 100000, 32, 'sha256')
      .slice(0, 32);
  }
  
  encrypt(plaintext: string): string {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-gcm', this.encryptionKey, iv);
    
    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const tag = cipher.getAuthTag();
    
    return `${iv.toString('hex')}:${encrypted}:${tag.toString('hex')}`;
  }
  
  decrypt(ciphertext: string): string {
    const [ivHex, encrypted, tagHex] = ciphertext.split(':');
    
    const iv = Buffer.from(ivHex, 'hex');
    const tag = Buffer.from(tagHex, 'hex');
    
    const decipher = crypto.createDecipheriv('aes-256-gcm', this.encryptionKey, iv);
    decipher.setAuthTag(tag);
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }
}

// Vault-like storage for API keys
export class APIKeyVault {
  private vault: Map<string, string> = new Map();
  private secretManager: SecretManager;
  
  constructor(masterKey: string) {
    this.secretManager = new SecretManager(masterKey);
  }
  
  storeKey(keyName: string, apiKey: string): void {
    const encrypted = this.secretManager.encrypt(apiKey);
    this.vault.set(keyName, encrypted);
  }
  
  retrieveKey(keyName: string): string | null {
    const encrypted = this.vault.get(keyName);
    if (!encrypted) return null;
    
    return this.secretManager.decrypt(encrypted);
  }
  
  // Use with environment-specific retrieval
  getApiKey(service: string, environment: string): string {
    const key = this.retrieveKey(`${service}_${environment}`);
    if (!key) {
      throw new Error(`API key not found for ${service} in ${environment}`);
    }
    return key;
  }
}
```

### Input Validation and Sanitisation

```typescript
// security/inputValidator.ts
import { z } from 'zod';
import DOMPurify from 'isomorphic-dompurify';

export class InputValidator {
  private static readonly maxInputLength = 50000;
  private static readonly allowedCharacters = /^[a-zA-Z0-9\s\-_.,!?'"():\n]+$/;
  
  static validateUserInput(input: string): string {
    // Length check
    if (input.length > this.maxInputLength) {
      throw new Error(`Input exceeds maximum length of ${this.maxInputLength}`);
    }
    
    // Character check
    if (!this.allowedCharacters.test(input)) {
      throw new Error('Input contains disallowed characters');
    }
    
    // Sanitise HTML content
    const sanitised = DOMPurify.sanitize(input, {
      ALLOWED_TAGS: [],
      ALLOWED_ATTR: [],
    });
    
    return sanitised.trim();
  }
  
  static validateAgentInput(schema: z.ZodSchema, input: any): any {
    try {
      return schema.parse(input);
    } catch (error) {
      throw new Error(`Validation failed: ${error}`);
    }
  }
}

// Rate limiting
export class RateLimiter {
  private requests: Map<string, number[]> = new Map();
  
  constructor(private windowMs: number, private maxRequests: number) {}
  
  isAllowed(clientId: string): boolean {
    const now = Date.now();
    const requests = this.requests.get(clientId) || [];
    
    // Remove old requests outside window
    const recentRequests = requests.filter((t) => now - t < this.windowMs);
    
    if (recentRequests.length >= this.maxRequests) {
      return false;
    }
    
    recentRequests.push(now);
    this.requests.set(clientId, recentRequests);
    
    return true;
  }
}
```

### CORS and Authentication

```typescript
// security/auth.ts
import express from 'express';
import jwt from 'jsonwebtoken';

export interface AuthToken {
  userId: string;
  scope: string[];
  iat: number;
  exp: number;
}

export class AuthManager {
  constructor(private jwtSecret: string) {}
  
  generateToken(userId: string, scope: string[] = []): string {
    const payload: AuthToken = {
      userId,
      scope,
      iat: Math.floor(Date.now() / 1000),
      exp: Math.floor(Date.now() / 1000) + 86400, // 24 hours
    };
    
    return jwt.sign(payload, this.jwtSecret);
  }
  
  verifyToken(token: string): AuthToken {
    return jwt.verify(token, this.jwtSecret) as AuthToken;
  }
  
  middleware() {
    return (req: express.Request, res: express.Response, next: express.NextFunction) => {
      const token = req.headers.authorization?.replace('Bearer ', '');
      
      if (!token) {
        return res.status(401).json({ error: 'No token provided' });
      }
      
      try {
        const decoded = this.verifyToken(token);
        (req as any).user = decoded;
        next();
      } catch (error) {
        return res.status(401).json({ error: 'Invalid token' });
      }
    };
  }
}

// CORS configuration
export function configureCORS(app: express.Application, allowedOrigins: string[]): void {
  app.use((req, res, next) => {
    const origin = req.headers.origin;
    
    if (allowedOrigins.includes(origin || '')) {
      res.header('Access-Control-Allow-Origin', origin);
      res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      res.header('Access-Control-Allow-Credentials', 'true');
    }
    
    if (req.method === 'OPTIONS') {
      res.sendStatus(200);
    } else {
      next();
    }
  });
}
```

---

## Database and Persistence

### PostgreSQL Checkpoint Saver

```typescript
// persistence/postgresCheckpointer.ts
import { Pool, Client } from 'pg';
import { BaseCheckpointSaver, Checkpoint } from '@langchain/langgraph';

export class PostgresCheckpointSaver extends BaseCheckpointSaver {
  private pool: Pool;
  
  constructor(connectionString: string) {
    super();
    this.pool = new Pool({ connectionString });
  }
  
  async initialize(): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query(`
        CREATE TABLE IF NOT EXISTS checkpoints (
          thread_id VARCHAR(255) NOT NULL,
          checkpoint_id VARCHAR(255) NOT NULL,
          checkpoint JSONB NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (thread_id, checkpoint_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_checkpoints_created 
        ON checkpoints(created_at);
      `);
    } finally {
      client.release();
    }
  }
  
  async get_checkpoint(config: any): Promise<Checkpoint | null> {
    const { thread_id, checkpoint_id } = config.configurable;
    
    const result = await this.pool.query(
      'SELECT checkpoint FROM checkpoints WHERE thread_id = $1 AND checkpoint_id = $2',
      [thread_id, checkpoint_id]
    );
    
    return result.rows[0]?.checkpoint || null;
  }
  
  async put_checkpoint(config: any, checkpoint: Checkpoint): Promise<void> {
    const { thread_id, checkpoint_id } = config.configurable;
    
    await this.pool.query(
      `INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint)
       VALUES ($1, $2, $3)
       ON CONFLICT (thread_id, checkpoint_id)
       DO UPDATE SET checkpoint = EXCLUDED.checkpoint`,
      [thread_id, checkpoint_id, JSON.stringify(checkpoint)]
    );
  }
  
  async list_checkpoints(thread_id: string): Promise<Checkpoint[]> {
    const result = await this.pool.query(
      'SELECT checkpoint FROM checkpoints WHERE thread_id = $1 ORDER BY created_at DESC',
      [thread_id]
    );
    
    return result.rows.map((row) => row.checkpoint);
  }
}

// Usage
const checkpointer = new PostgresCheckpointSaver(process.env.POSTGRES_URL!);
await checkpointer.initialize();

const graph = stateGraph.compile({
  checkpointer,
});
```

### SQLite for Local Development

```typescript
// persistence/sqliteCheckpointer.ts
import Database from 'better-sqlite3';
import { BaseCheckpointSaver, Checkpoint } from '@langchain/langgraph';

export class SqliteCheckpointSaver extends BaseCheckpointSaver {
  private db: Database.Database;
  
  constructor(dbPath: string = './checkpoints.db') {
    super();
    this.db = new Database(dbPath);
    this.initialize();
  }
  
  private initialize(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        checkpoint_id TEXT NOT NULL,
        checkpoint TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (thread_id, checkpoint_id)
      );
      
      CREATE INDEX IF NOT EXISTS idx_checkpoints_created 
      ON checkpoints(created_at);
    `);
  }
  
  async get_checkpoint(config: any): Promise<Checkpoint | null> {
    const { thread_id, checkpoint_id } = config.configurable;
    
    const stmt = this.db.prepare(
      'SELECT checkpoint FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?'
    );
    const result = stmt.get(thread_id, checkpoint_id) as any;
    
    return result ? JSON.parse(result.checkpoint) : null;
  }
  
  async put_checkpoint(config: any, checkpoint: Checkpoint): Promise<void> {
    const { thread_id, checkpoint_id } = config.configurable;
    
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_id, checkpoint)
      VALUES (?, ?, ?)
    `);
    stmt.run(thread_id, checkpoint_id, JSON.stringify(checkpoint));
  }
  
  async list_checkpoints(thread_id: string): Promise<Checkpoint[]> {
    const stmt = this.db.prepare(
      'SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC'
    );
    const results = stmt.all(thread_id) as any[];
    
    return results.map((row) => JSON.parse(row.checkpoint));
  }
}
```

---

## Monitoring and Observability

### LangSmith Integration

```typescript
// monitoring/langsmithClient.ts
import { Client } from 'langsmith';

export class LangSmithMonitoring {
  private client: Client;
  
  constructor(apiKey: string, projectName?: string) {
    this.client = new Client({
      apiKey,
      apiUrl: 'https://api.smith.langchain.com',
      projectName,
    });
  }
  
  async trackAgentExecution(
    agentName: string,
    input: string,
    output: string,
    metadata: Record<string, any> = {}
  ): Promise<void> {
    await this.client.createRun({
      name: agentName,
      run_type: 'agent',
      inputs: { input },
      outputs: { output },
      extra: {
        metadata,
        timestamp: new Date().toISOString(),
      },
    });
  }
  
  async trackToolCall(
    toolName: string,
    input: any,
    output: any,
    duration: number
  ): Promise<void> {
    await this.client.createRun({
      name: toolName,
      run_type: 'tool',
      inputs: input,
      outputs: output,
      extra: {
        duration,
        timestamp: new Date().toISOString(),
      },
    });
  }
}

// Callback handler for LangChain integration
import { BaseCallbackHandler } from '@langchain/core/callbacks';

export class LangSmithCallbackHandler extends BaseCallbackHandler {
  name = 'langsmith_handler';
  
  constructor(private langsmith: LangSmithMonitoring) {
    super();
  }
  
  async handleChainEnd(output: any): Promise<void> {
    console.log('Chain completed:', output);
  }
  
  async handleToolEnd(output: string): Promise<void> {
    console.log('Tool completed:', output);
  }
}
```

### Structured Logging

```typescript
// monitoring/logger.ts
import winston from 'winston';

export function createLogger(serviceName: string): winston.Logger {
  return winston.createLogger({
    defaultMeta: { service: serviceName },
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.json(),
    ),
    transports: [
      new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple(),
        ),
      }),
      new winston.transports.File({
        filename: 'error.log',
        level: 'error',
      }),
      new winston.transports.File({
        filename: 'combined.log',
      }),
    ],
  });
}

// Usage
const logger = createLogger('agent-service');

async function executeAgentWithLogging(
  agent: any,
  input: string,
  logger: winston.Logger
): Promise<string> {
  const startTime = Date.now();
  
  logger.info('Agent execution started', { input });
  
  try {
    const result = await agent.invoke({ input });
    const duration = Date.now() - startTime;
    
    logger.info('Agent execution completed', {
      duration,
      inputLength: input.length,
      outputLength: (result as string).length,
    });
    
    return result;
  } catch (error) {
    const duration = Date.now() - startTime;
    
    logger.error('Agent execution failed', {
      duration,
      error: error instanceof Error ? error.message : 'Unknown error',
      stack: error instanceof Error ? error.stack : undefined,
    });
    
    throw error;
  }
}
```

---

## Deployment Patterns

### Next.js Deployment with Vercel

```typescript
// pages/api/agent.ts
import { NextApiRequest, NextApiResponse } from 'next';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';

// Initialize at module level for reuse across requests
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const agent = createReactAgent({
  llm: model,
  tools: [], // Add your tools
});

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  // Enable streaming response
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  const { input } = req.body;
  
  try {
    const stream = await agent.stream({ input });
    
    for await (const event of stream) {
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    }
    
    res.write('data: [DONE]\n\n');
    res.end();
  } catch (error) {
    console.error('Agent error:', error);
    res.write(
      `data: ${JSON.stringify({
        error: error instanceof Error ? error.message : 'Unknown error',
      })}\n\n`
    );
    res.end();
  }
}
```

### Express.js with Hot Reload

```typescript
// server.ts
import express from 'express';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';

const app = express();
app.use(express.json());

// Agent instance management
let agent: any = null;
let lastUpdateTime = Date.now();

async function initializeAgent(): Promise<void> {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  
  agent = createReactAgent({
    llm: model,
    tools: [],
  });
  
  lastUpdateTime = Date.now();
  console.log('Agent initialised');
}

// Hot reload endpoint
app.post('/reload', async (req, res) => {
  try {
    await initializeAgent();
    res.json({ status: 'reloaded', timestamp: lastUpdateTime });
  } catch (error) {
    res.status(500).json({ error: error });
  }
});

app.post('/agent', async (req, res) => {
  if (!agent) {
    return res.status(500).json({ error: 'Agent not initialised' });
  }
  
  const { input } = req.body;
  
  try {
    const result = await agent.invoke({ input });
    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: error });
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

// Initialize and start
(async () => {
  await initializeAgent();
  
  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(`Server running on port ${port}`);
  });
})();
```

---

[Document continues with additional sections...]

This production guide provides comprehensive coverage of deployment, operations, security, and best practices for running LangChain.js and LangGraph.js in production environments. Refer to specific sections for your deployment scenario.

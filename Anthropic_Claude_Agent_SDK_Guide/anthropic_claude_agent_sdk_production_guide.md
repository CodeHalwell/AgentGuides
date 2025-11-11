# Claude Agent SDK - Production Deployment Guide

> **Enterprise-Ready Patterns, Deployment Strategies, and Operational Excellence**

---

## Table of Contents

1. [Error Handling & Resilience](#error-handling--resilience)
2. [Cost Optimisation & Budgeting](#cost-optimisation--budgeting)
3. [Monitoring & Observability](#monitoring--observability)
4. [Performance Tuning](#performance-tuning)
5. [Deployment Strategies](#deployment-strategies)
6. [Security & Compliance](#security--compliance)
7. [Scaling & High Availability](#scaling--high-availability)
8. [Troubleshooting](#troubleshooting)

---

## Error Handling & Resilience

### Comprehensive Error Handling

```typescript
// TypeScript - Production-grade error handling
import { query } from '@anthropic-ai/claude-agent-sdk';
import { Logger } from './logger';

enum ErrorSeverity {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low'
}

interface ErrorContext {
  timestamp: Date;
  sessionId?: string;
  prompt: string;
  toolName?: string;
  userId?: string;
  attempt: number;
  severity: ErrorSeverity;
  originalError: Error;
  metadata: Record<string, any>;
}

class ProductionQueryHandler {
  private logger: Logger;
  private maxRetries = 3;
  private backoffMultiplier = 2;

  async executeWithErrorHandling(
    prompt: string,
    context: Partial<ErrorContext> = {}
  ): Promise<string> {
    let attempt = 1;
    let lastError: Error | null = null;

    while (attempt <= this.maxRetries) {
      try {
        return await this.attemptQuery(prompt, { ...context, attempt });
      } catch (error) {
        lastError = error as Error;

        const errorContext: ErrorContext = {
          timestamp: new Date(),
          attempt,
          severity: this.determineSeverity(error),
          originalError: lastError,
          prompt,
          ...context,
          metadata: {
            errorType: error instanceof Error ? error.constructor.name : typeof error,
            errorMessage: error instanceof Error ? error.message : String(error)
          }
        };

        // Log the error
        this.logger.error(`Query failed (attempt ${attempt}/${this.maxRetries})`, errorContext);

        // Determine if we should retry
        if (!this.shouldRetry(error, attempt)) {
          this.logger.error(`Query failed permanently`, errorContext);
          throw new ProductionError(
            `Query failed after ${attempt} attempt(s)`,
            errorContext
          );
        }

        // Calculate backoff
        const waitMs = Math.min(
          1000 * Math.pow(this.backoffMultiplier, attempt - 1),
          30000
        );

        this.logger.info(`Retrying after ${waitMs}ms...`, { attempt, waitMs });
        await new Promise(resolve => setTimeout(resolve, waitMs));

        attempt++;
      }
    }

    throw new ProductionError(
      `Query failed after maximum retries`,
      {
        timestamp: new Date(),
        attempt: this.maxRetries,
        severity: ErrorSeverity.CRITICAL,
        originalError: lastError!,
        prompt,
        ...context,
        metadata: { finalError: lastError?.message }
      }
    );
  }

  private async attemptQuery(
    prompt: string,
    context: ErrorContext
  ): Promise<string> {
    let result = '';
    const response = query({
      prompt,
      options: { model: "claude-3-5-sonnet-20241022" }
    });

    for await (const message of response) {
      if (message.type === 'assistant') {
        if (typeof message.content === 'string') {
          result += message.content;
        }
      } else if (message.type === 'error') {
        throw new ApiError(
          message.error?.message || 'Unknown API error',
          message.error?.code,
          message.error?.type
        );
      }
    }

    return result;
  }

  private determineSeverity(error: any): ErrorSeverity {
    if (error.code === 'AUTHENTICATION_FAILED') return ErrorSeverity.CRITICAL;
    if (error.code === 'CONTEXT_LENGTH_EXCEEDED') return ErrorSeverity.HIGH;
    if (error.code === 'RATE_LIMIT_EXCEEDED') return ErrorSeverity.MEDIUM;
    if (error.code === 'TIMEOUT') return ErrorSeverity.MEDIUM;
    return ErrorSeverity.LOW;
  }

  private shouldRetry(error: any, attempt: number): boolean {
    // Don't retry authentication errors
    if (error.code === 'AUTHENTICATION_FAILED') return false;

    // Don't retry after max attempts
    if (attempt >= this.maxRetries) return false;

    // Retry on transient errors
    const transientErrors = ['RATE_LIMIT_EXCEEDED', 'TIMEOUT', 'ECONNREFUSED'];
    return transientErrors.includes(error.code);
  }
}

class ProductionError extends Error {
  constructor(
    message: string,
    public context: ErrorContext
  ) {
    super(message);
    this.name = 'ProductionError';
  }
}

class ApiError extends Error {
  constructor(
    message: string,
    public code?: string,
    public type?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
```

### Graceful Degradation

```python
# Python - Graceful fallback strategies
import asyncio
from claude_agent_sdk import query, ClaudeSDKError
from typing import Optional

class FallbackStrategy:
    """Handle failures with fallback models and strategies."""

    async def query_with_fallback(
        self,
        prompt: str,
        primary_model: str = "claude-3-5-sonnet-20241022",
        fallback_models: Optional[list] = None
    ) -> str:
        fallback_models = fallback_models or [
            "claude-3-5-opus-20241022",  # More capable but slower
            "claude-3-5-haiku-20241022"   # Faster but less capable
        ]

        models_to_try = [primary_model] + fallback_models

        for model in models_to_try:
            try:
                print(f"Attempting with {model}...")
                result = ""

                async for message in query(
                    prompt=prompt,
                    options=ClaudeAgentOptions(model=model)
                ):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                result += block.text

                return result

            except ClaudeSDKError as e:
                print(f"Failed with {model}: {e}")
                if model == models_to_try[-1]:
                    raise
                continue
            except asyncio.TimeoutError:
                print(f"Timeout with {model}, trying fallback...")
                continue

        raise Exception("All models failed")

    async def query_with_caching(
        self,
        prompt: str,
        cache_key: Optional[str] = None
    ) -> str:
        """Use caching to reduce costs and improve performance."""
        cache_key = cache_key or hash(prompt)

        # Check cache first
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        # Query if not cached
        result = await self.query_with_fallback(prompt)

        # Store in cache
        await self._store_in_cache(cache_key, result)

        return result
```

---

## Cost Optimisation & Budgeting

### Budget Management

```typescript
// TypeScript - Comprehensive cost tracking and budgeting
interface BudgetConfig {
  maxDailyUSD: number;
  maxSessionUSD: number;
  maxMonthlyUSD: number;
  alertThresholdPercent: number;
}

interface CostMetrics {
  sessionCost: number;
  dailyCost: number;
  monthlyCost: number;
  estimatedTotalTokens: number;
}

class CostManager {
  private config: BudgetConfig;
  private sessionMetrics: Map<string, CostMetrics> = new Map();

  constructor(config: BudgetConfig) {
    this.config = config;
  }

  async queryWithBudgetControl(
    prompt: string,
    sessionId: string
  ): Promise<string> {
    // Check remaining budget
    const dailyRemaining = await this.getRemainingDailyBudget();
    const sessionRemaining = this.getSessionBudgetRemaining(sessionId);

    if (dailyRemaining <= 0) {
      throw new Error('Daily budget exceeded');
    }

    if (sessionRemaining <= 0) {
      throw new Error('Session budget exceeded');
    }

    // Set a safety limit for this query (assume ~1K tokens = $0.015)
    const estimatedQueryCost = Math.min(dailyRemaining, sessionRemaining);

    const response = query({
      prompt,
      options: {
        model: "claude-3-5-sonnet-20241022",
        maxBudgetUsd: estimatedQueryCost  // Stop if exceeds budget
      }
    });

    let result = '';
    let queryTokens = 0;
    let queryCost = 0;

    for await (const message of response) {
      if (message.type === 'assistant') {
        if (typeof message.content === 'string') {
          result += message.content;
          // Rough token count: 1 char ≈ 0.25 tokens
          queryTokens += message.content.length * 0.25;
        }
      } else if (message.type === 'system' && message.usage) {
        queryTokens = message.usage.input_tokens + message.usage.output_tokens;
        queryCost = this.calculateCost(message.usage);
      }
    }

    // Update metrics
    this.updateMetrics(sessionId, queryCost, queryTokens);

    // Check thresholds
    await this.checkBudgetAlerts();

    return result;
  }

  private calculateCost(usage: { input_tokens: number; output_tokens: number }): number {
    // Claude 3.5 Sonnet pricing (as of 2024)
    const inputCostPer1k = 0.003;
    const outputCostPer1k = 0.015;

    return (
      (usage.input_tokens / 1000) * inputCostPer1k +
      (usage.output_tokens / 1000) * outputCostPer1k
    );
  }

  private updateMetrics(sessionId: string, cost: number, tokens: number) {
    const current = this.sessionMetrics.get(sessionId) || {
      sessionCost: 0,
      dailyCost: 0,
      monthlyCost: 0,
      estimatedTotalTokens: 0
    };

    current.sessionCost += cost;
    current.dailyCost += cost;
    current.monthlyCost += cost;
    current.estimatedTotalTokens += tokens;

    this.sessionMetrics.set(sessionId, current);
  }

  private async getRemainingDailyBudget(): Promise<number> {
    // In production, fetch from database
    const dailyUsage = 0; // Get from DB
    return this.config.maxDailyUSD - dailyUsage;
  }

  private getSessionBudgetRemaining(sessionId: string): number {
    const metrics = this.sessionMetrics.get(sessionId);
    if (!metrics) return this.config.maxSessionUSD;
    return this.config.maxSessionUSD - metrics.sessionCost;
  }

  private async checkBudgetAlerts() {
    for (const [sessionId, metrics] of this.sessionMetrics.entries()) {
      const usage = metrics.sessionCost / this.config.maxSessionUSD;
      if (usage > (this.config.alertThresholdPercent / 100)) {
        console.warn(`Budget alert: Session ${sessionId} at ${(usage * 100).toFixed(1)}%`);
      }
    }
  }
}
```

### Model Selection for Cost-Performance

```python
# Python - Choose models based on requirements
from claude_agent_sdk import query, ClaudeAgentOptions

class ModelSelector:
    """Select appropriate models based on task complexity and budget."""

    MODEL_PROFILES = {
        "claude-3-5-haiku-20241022": {
            "cost_tier": "economy",
            "best_for": ["simple_qa", "classification", "basic_generation"],
            "input_cost_per_1k": 0.00080,
            "output_cost_per_1k": 0.0024,
            "latency_ms": 200
        },
        "claude-3-5-sonnet-20241022": {
            "cost_tier": "mid",
            "best_for": ["general_purpose", "coding", "analysis", "orchestration"],
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.015,
            "latency_ms": 500
        },
        "claude-3-5-opus-20241022": {
            "cost_tier": "premium",
            "best_for": ["complex_reasoning", "multi_step_tasks", "research"],
            "input_cost_per_1k": 0.015,
            "output_cost_per_1k": 0.075,
            "latency_ms": 1000
        }
    }

    def select_model(
        self,
        task_type: str,
        budget_priority: str = "balanced",
        latency_requirement: int = 5000
    ) -> str:
        """
        Select best model for task.
        budget_priority: 'cost' (cheapest), 'balanced' (default), 'quality' (best)
        """

        candidates = []

        # Filter by task type
        for model, profile in self.MODEL_PROFILES.items():
            if task_type in profile["best_for"]:
                candidates.append((model, profile))

        if not candidates:
            # Default to Sonnet if task not specifically listed
            return "claude-3-5-sonnet-20241022"

        # Filter by latency requirement
        candidates = [
            (m, p) for m, p in candidates
            if p["latency_ms"] <= latency_requirement
        ]

        if not candidates:
            return "claude-3-5-sonnet-20241022"

        # Sort by priority
        if budget_priority == "cost":
            return min(candidates, key=lambda x: x[1]["input_cost_per_1k"])[0]
        elif budget_priority == "quality":
            return max(candidates, key=lambda x: x[1]["cost_tier"])[0]
        else:  # balanced
            return candidates[len(candidates) // 2][0]

selector = ModelSelector()

# Usage
async def process_task(task_type: str):
    model = selector.select_model(
        task_type=task_type,
        budget_priority="balanced",
        latency_requirement=2000
    )

    async for message in query(
        prompt="Process this task",
        options=ClaudeAgentOptions(model=model)
    ):
        print(message)
```

---

## Monitoring & Observability

### Comprehensive Logging

```typescript
// TypeScript - Production-grade logging
import pino from 'pino';

interface LogContext {
  sessionId?: string;
  userId?: string;
  requestId?: string;
  timestamp: Date;
  duration?: number;
  tokensUsed?: number;
  costUSD?: number;
  [key: string]: any;
}

class ProductionLogger {
  private logger = pino({
    level: process.env.LOG_LEVEL || 'info',
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true,
        singleLine: false
      }
    }
  });

  logQueryStart(prompt: string, context: LogContext) {
    this.logger.info(
      {
        event: 'query_start',
        prompt: prompt.substring(0, 100),
        ...context
      },
      'Query started'
    );
  }

  logQueryComplete(result: string, context: LogContext & { tokensUsed: number; costUSD: number }) {
    this.logger.info(
      {
        event: 'query_complete',
        resultLength: result.length,
        ...context
      },
      `Query completed in ${context.duration}ms, cost: $${context.costUSD.toFixed(4)}`
    );
  }

  logToolUse(toolName: string, input: any, context: LogContext) {
    this.logger.debug(
      {
        event: 'tool_use',
        tool: toolName,
        inputSize: JSON.stringify(input).length,
        ...context
      },
      `Tool used: ${toolName}`
    );
  }

  logError(error: Error, context: LogContext) {
    this.logger.error(
      {
        event: 'error',
        error: {
          message: error.message,
          stack: error.stack,
          name: error.name
        },
        ...context
      },
      error.message
    );
  }

  logMetrics(metrics: {
    avgResponseTime: number;
    queriesPerHour: number;
    errorRate: number;
    costPerQuery: number;
  }, context: LogContext) {
    this.logger.info(
      {
        event: 'metrics',
        metrics,
        ...context
      },
      'Performance metrics'
    );
  }
}
```

### Metrics Collection

```python
# Python - Metrics collection and reporting
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

@dataclass
class QueryMetrics:
    timestamp: datetime
    duration_ms: float
    tokens_input: int
    tokens_output: int
    cost_usd: float
    model: str
    success: bool
    error_type: Optional[str] = None

class MetricsCollector:
    def __init__(self, retention_hours: int = 24):
        self.metrics: List[QueryMetrics] = []
        self.retention_hours = retention_hours
        self._cleanup_old_metrics()

    def record_query(
        self,
        duration_ms: float,
        tokens_input: int,
        tokens_output: int,
        model: str,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        cost = self._calculate_cost(tokens_input, tokens_output, model)
        metric = QueryMetrics(
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost,
            model=model,
            success=success,
            error_type=error_type
        )
        self.metrics.append(metric)

    def get_summary(self, minutes: int = 60) -> Dict:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [m for m in self.metrics if m.timestamp > cutoff]

        if not recent:
            return {"no_data": True}

        successful = [m for m in recent if m.success]
        failed = [m for m in recent if not m.success]

        return {
            "total_queries": len(recent),
            "successful": len(successful),
            "failed": len(failed),
            "error_rate": len(failed) / len(recent) if recent else 0,
            "avg_duration_ms": sum(m.duration_ms for m in recent) / len(recent),
            "total_cost_usd": sum(m.cost_usd for m in recent),
            "total_tokens": sum(m.tokens_input + m.tokens_output for m in recent),
            "queries_per_minute": len(recent) / minutes
        }

    def _calculate_cost(self, tokens_in: int, tokens_out: int, model: str) -> float:
        # Pricing as of 2024
        pricing = {
            "claude-3-5-sonnet-20241022": {
                "input": 0.003 / 1000,
                "output": 0.015 / 1000
            },
            "claude-3-5-haiku-20241022": {
                "input": 0.00080 / 1000,
                "output": 0.0024 / 1000
            },
            "claude-3-5-opus-20241022": {
                "input": 0.015 / 1000,
                "output": 0.075 / 1000
            }
        }
        rates = pricing.get(model, pricing["claude-3-5-sonnet-20241022"])
        return (tokens_in * rates["input"]) + (tokens_out * rates["output"])

    def _cleanup_old_metrics(self):
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff]
```

---

## Performance Tuning

### Context Optimization

```typescript
// TypeScript - Optimize context window usage
class ContextOptimiser {
  private maxContextTokens = 200000;  // Claude 3.5 limit
  private reserveTokens = 5000;       // Reserve for output

  optimisePrompt(
    basePrompt: string,
    examples: string[] = [],
    instructions: string = ""
  ): string {
    const reserved = this.reserveTokens;
    const available = this.maxContextTokens - reserved;

    // Calculate token estimates (rough: 1 token ≈ 4 chars)
    const estimatedTokens = (text: string) => Math.ceil(text.length / 4);

    let tokensUsed = 0;
    let optimisedPrompt = basePrompt;
    tokensUsed += estimatedTokens(basePrompt);

    // Add instructions if space allows
    if (instructions && tokensUsed + estimatedTokens(instructions) < available * 0.7) {
      optimisedPrompt += `\n\nInstructions:\n${instructions}`;
      tokensUsed += estimatedTokens(instructions);
    }

    // Add examples in priority order
    for (const example of examples) {
      const exampleTokens = estimatedTokens(example);
      if (tokensUsed + exampleTokens < available * 0.8) {
        optimisedPrompt += `\n\n${example}`;
        tokensUsed += exampleTokens;
      }
    }

    console.log(`Context optimised: ${tokensUsed} tokens used, ${((tokensUsed/available)*100).toFixed(1)}% of budget`);

    return optimisedPrompt;
  }

  // Compress context for long conversations
  compressConversationHistory(history: Array<{role: string; content: string}>, targetTokens = 50000): string {
    if (history.length === 0) return '';

    // Keep recent messages verbatim
    const recentMessages = history.slice(-5);
    let compressed = recentMessages
      .map(m => `${m.role}: ${m.content}`)
      .join('\n\n');

    const recentTokens = Math.ceil(compressed.length / 4);

    if (recentTokens > targetTokens) {
      // If recent alone exceeds target, summarise
      return this.summariseMessages(recentMessages);
    }

    // Add older messages if space allows
    const budgetLeft = targetTokens - recentTokens;
    const olderMessages = history.slice(0, -5);

    if (olderMessages.length > 0) {
      const olderSummary = this.summariseMessages(olderMessages);
      compressed = `[Earlier conversation summary]\n${olderSummary}\n\n[Recent messages]\n${compressed}`;
    }

    return compressed;
  }

  private summariseMessages(messages: Array<{role: string; content: string}>): string {
    return `Conversation included ${messages.length} messages discussing: ${
      messages.map(m => m.content.substring(0, 30)).join(', ')
    }...`;
  }
}
```

---

## Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile for Claude Agent SDK application
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY src/ ./src/
COPY dist/ ./dist/

# Set environment variables
ENV NODE_ENV=production
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"

# Expose port
EXPOSE 3000

# Start application
CMD ["node", "dist/index.js"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-agent
  labels:
    app: claude-agent
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: claude-agent
  template:
    metadata:
      labels:
        app: claude-agent
    spec:
      containers:
      - name: agent
        image: myregistry.azurecr.io/claude-agent:latest
        ports:
        - containerPort: 3000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: claude-secrets
              key: api-key
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 1
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
                  - claude-agent
              topologyKey: kubernetes.io/hostname
```

---

## Security & Compliance

### API Key Management

```typescript
// TypeScript - Secure API key handling
import { SecretClient } from "@azure/keyvault-secrets";
import { DefaultAzureCredential } from "@azure/identity";

class SecureApiKeyManager {
  private secretClient: SecretClient;

  constructor(vaultUrl: string) {
    const credential = new DefaultAzureCredential();
    this.secretClient = new SecretClient(vaultUrl, credential);
  }

  async getApiKey(keyName: string): Promise<string> {
    const secret = await this.secretClient.getSecret(keyName);
    return secret.value || '';
  }

  async rotateApiKey(keyName: string, newKey: string) {
    await this.secretClient.setSecret(keyName, newKey);
    console.log(`API key ${keyName} rotated`);
  }

  // Use environment variable as fallback
  getApiKeySync(): string {
    return process.env.ANTHROPIC_API_KEY || '';
  }
}

// Usage in agent
async function createSecureAgent() {
  const keyManager = new SecureApiKeyManager(
    process.env.KEY_VAULT_URL || ''
  );

  const apiKey = await keyManager.getApiKey('anthropic-api-key');

  if (!apiKey) {
    throw new Error('Failed to retrieve API key');
  }

  // Use API key in queries
  const response = query({
    prompt: "Your prompt here",
    options: { model: "claude-3-5-sonnet-20241022" }
  });
}
```

### Input Validation & Sanitisation

```python
# Python - Secure input handling
from typing import Any
import re

class InputValidator:
    """Validate and sanitise agent inputs."""

    MAX_PROMPT_LENGTH = 100000
    MAX_PROMPT_LINES = 1000
    DANGEROUS_PATTERNS = [
        r'system\s*prompt\s*override',
        r'ignore\s+previous\s+instructions',
        r'<\s*script',
        r'__.*__',  # Python magic methods in prompts
    ]

    def validate_prompt(self, prompt: str) -> bool:
        """Check if prompt is safe."""

        # Length checks
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt too long: {len(prompt)} > {self.MAX_PROMPT_LENGTH}")

        if prompt.count('\n') > self.MAX_PROMPT_LINES:
            raise ValueError(f"Too many lines: {prompt.count(chr(10))} > {self.MAX_PROMPT_LINES}")

        # Pattern checks for injection attempts
        lower_prompt = prompt.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, lower_prompt, re.IGNORECASE):
                raise ValueError(f"Dangerous pattern detected: {pattern}")

        return True

    def sanitise_tool_input(self, tool_name: str, input_data: dict) -> dict:
        """Sanitise inputs before passing to tools."""

        if tool_name == "Bash":
            # Prevent command injection
            command = input_data.get("command", "")
            if any(char in command for char in [";", "|", "&", ">", "<", "$"]):
                raise ValueError("Potentially dangerous bash command")

        if tool_name == "Write":
            # Prevent directory traversal
            path = input_data.get("file_path", "")
            if ".." in path or path.startswith("/"):
                raise ValueError("Path traversal detected")

        return input_data

    def validate_mcp_server(self, server_config: dict) -> bool:
        """Validate MCP server configurations."""

        if server_config.get("type") == "stdio":
            # Only allow whitelisted commands
            command = server_config.get("command")
            if not command.startswith("npx") and not command.startswith("node"):
                raise ValueError(f"Command not whitelisted: {command}")

        return True

validator = InputValidator()

# Usage
async def safe_agent_query(prompt: str):
    # Validate input
    validator.validate_prompt(prompt)

    async for message in query(prompt=prompt):
        print(message)
```

---

## Scaling & High Availability

### Load Balancing

```typescript
// TypeScript - Load balancing across multiple agent instances
interface AgentNode {
  id: string;
  url: string;
  healthScore: number;
  activeConnections: number;
}

class LoadBalancer {
  private nodes: Map<string, AgentNode> = new Map();
  private healthCheckInterval = 30000;  // 30 seconds

  registerNode(node: AgentNode) {
    this.nodes.set(node.id, node);
    this.startHealthCheck(node.id);
  }

  selectNode(): AgentNode {
    const healthyNodes = Array.from(this.nodes.values())
      .filter(n => n.healthScore > 0.5)
      .sort((a, b) => {
        // Prefer nodes with less load and better health
        const scoreA = a.healthScore - (a.activeConnections * 0.1);
        const scoreB = b.healthScore - (b.activeConnections * 0.1);
        return scoreB - scoreA;
      });

    if (healthyNodes.length === 0) {
      throw new Error('No healthy nodes available');
    }

    return healthyNodes[0];
  }

  async route(prompt: string): Promise<string> {
    const node = this.selectNode();

    try {
      node.activeConnections++;

      const response = await fetch(`${node.url}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      if (!response.ok) {
        node.healthScore *= 0.9;  // Reduce health score
      }

      const result = await response.json();
      node.healthScore = Math.min(node.healthScore + 0.1, 1);  // Improve health

      return result.result;

    } finally {
      node.activeConnections--;
    }
  }

  private startHealthCheck(nodeId: string) {
    setInterval(async () => {
      const node = this.nodes.get(nodeId);
      if (!node) return;

      try {
        const response = await fetch(`${node.url}/health`, {
          timeout: 5000
        });
        node.healthScore = response.ok ? 1 : 0.5;
      } catch {
        node.healthScore = 0;
      }
    }, this.healthCheckInterval);
  }
}
```

---

## Troubleshooting

### Common Issues & Solutions

```markdown
## Issue: Rate Limiting (429 errors)

**Symptoms:**
- Frequent "RATE_LIMIT_EXCEEDED" errors
- Requests fail after initial success
- Error: "You've exceeded your rate limit"

**Solutions:**
1. **Implement exponential backoff:**
   - Wait: 1s, 2s, 4s, 8s... (max 30s)
   - Maximum retries: 3-5

2. **Add request batching:**
   - Group multiple requests
   - Space them over time

3. **Use a queue system:**
   - Queue requests at controlled rate
   - Process sequentially or with delays

4. **Request higher limits:**
   - Contact Anthropic support
   - Provide usage statistics

---

## Issue: Context Length Exceeded

**Symptoms:**
- Error: "CONTEXT_LENGTH_EXCEEDED"
- Agent fails on complex tasks
- Large document analysis fails

**Solutions:**
1. **Break into smaller tasks:**
   ```typescript
   // Instead of one large request
   const chunks = splitDocument(largeDoc, 100);
   for (const chunk of chunks) {
     const analysis = await agent.analyse(chunk);
   }
   ```

2. **Summarise conversation history:**
   - Keep only recent N messages
   - Summarise older conversation

3. **Use compression:**
   - Remove redundant text
   - Compress long lists
   - Abbreviate known terms

4. **Switch to larger context model:**
   - All Claude models: 200K tokens

---

## Issue: Timeouts

**Symptoms:**
- Requests hang indefinitely
- No response after 30+ seconds
- Connection resets

**Solutions:**
1. **Add timeout handling:**
   ```typescript
   const timeout = setTimeout(() => {
     throw new Error('Query timeout');
   }, 30000);
   ```

2. **Simplify complex prompts:**
   - Break into steps
   - Reduce context

3. **Check network:**
   - Latency from your region
   - ISP restrictions

4. **Use streaming:**
   - Get partial results faster
   - Don't wait for complete response

---

## Issue: Inconsistent Results

**Symptoms:**
- Same prompt gives different outputs
- Non-deterministic behaviour
- Unpredictable formatting

**Solutions:**
1. **Set explicit temperature:**
   ```typescript
   // Lower temperature = more consistent
   // 0.0 = deterministic, 1.0 = creative
   options: { temperature: 0.0 }
   ```

2. **Use structured output:**
   - Specify exact JSON format
   - Provide schema validation

3. **Add examples:**
   - Few-shot prompting
   - Show expected format

4. **Post-process outputs:**
   - Validate structure
   - Retry if invalid
```

---

**This production guide ensures your Claude Agent applications run reliably, efficiently, and cost-effectively at scale.**


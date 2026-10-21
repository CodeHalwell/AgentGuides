---
title: "OpenAI Agents SDK TypeScript: Features and Updates (2025–2026)"
description: "Type-safe agents, human-in-the-loop approvals, tracing, guardrails, MCP and provider-agnostic models in the TypeScript SDK, plus what changed in 2026."
framework: openai-agents-sdk-typescript
---

# OpenAI Agents SDK TypeScript: Features and Updates (2025–2026)

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%). First written in January 2025. TypeScript 5+, Node.js 22+.

---

## Overview

The OpenAI Agents SDK for TypeScript is the **official production-ready framework** for building type-safe agentic AI applications. This page covers the main features of the TypeScript SDK; [What changed in 2026](#what-changed-in-2026) summarises the recent releases.

---

## Table of Contents

1. [Production-Ready Status](#production-ready-status)
2. [TypeScript-Specific Features](#typescript-specific-features)
3. [Human-in-the-Loop Approvals](#human-in-the-loop-approvals)
4. [Built-in Tracing and Observability](#built-in-tracing-and-observability)
5. [Guardrails System](#guardrails-system)
6. [Handoffs and Delegation](#handoffs-and-delegation)
7. [MCP Integration](#mcp-integration)
8. [Provider-Agnostic Support](#provider-agnostic-support)
9. [Type-Safe Tools](#type-safe-tools)
10. [Production Features](#production-features)
11. [What changed in 2026](#what-changed-in-2026)

---

## Production-Ready Status

### Why Choose the TypeScript Agents SDK?

- ✅ **TypeScript-First Design**: Full type safety, excellent IDE support
- ✅ **Production-Ready**: Stable API, comprehensive testing
- ✅ **Active Maintenance**: Regular updates and feature additions
- ✅ **Enterprise Features**: Observability, security, scalability
- ✅ **No Experimental Code**: Unlike Swarm, fully production-supported

### Swarm Replacement

**The Agents SDK is the official production replacement for experimental Swarm implementations.**

```typescript
// BEFORE: Swarm (a Python-only experiment; TypeScript-style pseudo-code)
import { Swarm, Agent } from 'swarm';

const client = new Swarm();
const agent = new Agent({ name: 'Assistant' });
const result = client.run({ agent, messages: [...] });

// AFTER: Agents SDK (Production, TypeScript-First)
import { Agent, run } from '@openai/agents';

const agent = new Agent({ name: 'Assistant' });
const result = await run(agent, 'Hello');
console.log(result.finalOutput);
```

---

## TypeScript-Specific Features

### 1. Full Type Inference

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

// Schema defines types automatically
const UserSchema = z.object({
  id: z.string().uuid(),
  name: z.string(),
  email: z.string().email(),
  age: z.number().min(18).max(120),
});

type User = z.infer<typeof UserSchema>; // Automatic type inference

const getUserTool = tool({
  name: 'get_user',
  description: 'Fetch user information',
  parameters: z.object({
    userId: z.string().uuid(),
  }),
  execute: async ({ userId }): Promise<User> => {
    // Return type is fully typed
    return {
      id: userId,
      name: 'John Doe',
      email: 'john@example.com',
      age: 30,
    };
  },
});

const agent = new Agent({
  name: 'User Agent',
  tools: [getUserTool],
});

// Full type safety throughout
const result = await run(agent, 'Get user info');
```

### 2. Generic Agent Patterns

```typescript
import { Agent } from '@openai/agents';
import { z } from 'zod';

// Generic typed agent creator
function createTypedAgent<TOutput extends z.ZodObject<any>>(config: {
  name: string;
  instructions: string;
  outputType: TOutput;
  tools?: any[];
}) {
  return new Agent({
    name: config.name,
    instructions: config.instructions,
    outputType: config.outputType,
    tools: config.tools,
  });
}

// Usage with type inference
const SentimentSchema = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
});

type SentimentAnalysis = z.infer<typeof SentimentSchema>;

const sentimentAgent = createTypedAgent({
  name: 'Sentiment Analyzer',
  instructions: 'Analyze sentiment of text',
  outputType: SentimentSchema,
});

// finalOutput is already parsed and typed from the schema
const result = await run(sentimentAgent, 'I love TypeScript!');
const analysis: SentimentAnalysis | undefined = result.finalOutput;
```

### 3. Type-Safe Context Injection

```typescript
import { Agent, run } from '@openai/agents';

interface UserContext {
  userId: string;
  subscription: 'free' | 'premium' | 'enterprise';
  preferences: {
    language: string;
    timezone: string;
  };
}

const agent = new Agent({
  name: 'Personalized Assistant',
  instructions: 'Provide personalized assistance based on user context',
});

async function executeWithContext(
  agent: Agent,
  input: string,
  context: UserContext
): Promise<string> {
  // Type-safe context passing
  const result = await run(agent, input, {
    context,
  });

  return result.finalOutput ?? '';
}

// Usage
const response = await executeWithContext(
  agent,
  'What features do I have?',
  {
    userId: 'user_123',
    subscription: 'premium',
    preferences: {
      language: 'en',
      timezone: 'UTC',
    },
  }
);
```

---

## Human-in-the-Loop Approvals

### Approval Workflow

Mark a tool with `needsApproval` (a boolean or an async predicate). When the model calls it, the run pauses and returns the pending calls in `result.interruptions`; approve or reject each one on `result.state`, then resume by passing the state back to `run()`:

```typescript
import { Agent, run, tool, RunState } from '@openai/agents';
import { z } from 'zod';

// Critical action requiring human approval
const deleteUserTool = tool({
  name: 'delete_user',
  description: 'Delete user account permanently',
  parameters: z.object({
    userId: z.string(),
    reason: z.string().min(20),
  }),
  needsApproval: true,
  execute: async ({ userId, reason }) => {
    return { status: 'deleted', userId, timestamp: new Date().toISOString() };
  },
});

// Conditional approval: only permanent bans need a human
const banUserTool = tool({
  name: 'ban_user',
  description: 'Ban user from platform',
  parameters: z.object({
    userId: z.string(),
    duration: z.enum(['temporary', 'permanent']),
  }),
  needsApproval: async (_ctx, { duration }) => duration === 'permanent',
  execute: async ({ userId, duration }) => ({ status: 'banned', userId, duration }),
});

const adminAgent = new Agent({
  name: 'Admin Agent',
  instructions: 'Manage users with appropriate caution',
  tools: [deleteUserTool, banUserTool],
});

let result = await run(
  adminAgent,
  'Delete user account user_123 due to repeated Terms of Service violations',
);

while (result.interruptions.length > 0) {
  for (const interruption of result.interruptions) {
    console.log(`Approval required: ${interruption.name}`, interruption.arguments);

    // In production: show UI, post to Slack, write to an audit log...
    const approved = await getUserConfirmation(interruption);
    if (approved) result.state.approve(interruption);
    else result.state.reject(interruption);
  }
  result = await run(adminAgent, result.state);
}

console.log(result.finalOutput);
```

### Pausing across requests

Approvals often arrive minutes or hours later, from another process. Serialise the state, store it, and restore it when the decision comes in:

```typescript
// When the run pauses
const serialised = JSON.stringify(result.state);
await db.saveRunState(runId, serialised);

// Later, e.g. in a Slack button handler
const state = await RunState.fromString(adminAgent, await db.loadRunState(runId));
for (const interruption of state.getInterruptions()) {
  state.approve(interruption);
}
const resumed = await run(adminAgent, state);
```

---

## Built-in Tracing and Observability

### Workflow Visualization

```typescript
import { Agent, run, withTrace } from '@openai/agents';

const agent = new Agent({
  name: 'Data Processor',
  instructions: 'Process and analyze data',
});

// Group related operations
await withTrace('Customer Data Pipeline', async () => {
  const step1 = await run(agent, 'Validate customer data');
  const step2 = await run(agent, 'Enrich customer profiles');
  const step3 = await run(agent, 'Generate insights');

  return { step1, step2, step3 };
}, {
  metadata: {
    environment: 'production',
    pipeline_version: '2.0',
    timestamp: new Date().toISOString(),
  },
});

// View traces at: https://platform.openai.com/traces
```

### Integration with Observability Platforms

```typescript
import { Agent, run, withTrace } from '@openai/agents';
import { Logger } from 'winston';
import { Tracer } from '@opentelemetry/api';

interface ObservabilityConfig {
  logger: Logger;
  tracer: Tracer;
  metrics: MetricsCollector;
}

async function observableAgentExecution(
  agent: Agent,
  input: string,
  config: ObservabilityConfig
): Promise<string> {
  const span = config.tracer.startSpan('agent_execution');

  try {
    config.logger.info('Starting agent execution', { agent: agent.name, input });

    const startTime = Date.now();

    const result = await withTrace(
      `Execute ${agent.name}`,
      () => run(agent, input),
      {
        metadata: {
          agent_name: agent.name,
          input_length: input.length,
        },
      }
    );

    const duration = Date.now() - startTime;

    config.metrics.recordExecutionTime(agent.name, duration);
    config.logger.info('Agent execution completed', {
      agent: agent.name,
      duration,
    });

    span.setStatus({ code: 0 });
    return result.finalOutput;
  } catch (error) {
    span.setStatus({ code: 2, message: error.message });
    config.logger.error('Agent execution failed', { error });
    throw error;
  } finally {
    span.end();
  }
}
```

### Performance Metrics

```typescript
import { Agent, run } from '@openai/agents';

interface ExecutionMetrics {
  duration: number;
  tokenCount: number;
  toolCalls: number;
  errorRate: number;
}

class MetricsCollector {
  private metrics: Map<string, ExecutionMetrics[]> = new Map();

  async executeWithMetrics(
    agent: Agent,
    input: string
  ): Promise<{ result: string; metrics: ExecutionMetrics }> {
    const startTime = Date.now();
    let tokenCount = 0;
    let toolCalls = 0;

    try {
      const result = await run(agent, input);

      const metrics: ExecutionMetrics = {
        duration: Date.now() - startTime,
        tokenCount: result.runContext.usage.totalTokens,
        toolCalls: result.newItems.filter((i) => i.type === 'tool_call_item').length,
        errorRate: 0,
      };

      this.recordMetrics(agent.name, metrics);

      return {
        result: result.finalOutput,
        metrics,
      };
    } catch (error) {
      const metrics: ExecutionMetrics = {
        duration: Date.now() - startTime,
        tokenCount: 0,
        toolCalls: 0,
        errorRate: 1,
      };

      this.recordMetrics(agent.name, metrics);
      throw error;
    }
  }

  private recordMetrics(agentName: string, metrics: ExecutionMetrics): void {
    if (!this.metrics.has(agentName)) {
      this.metrics.set(agentName, []);
    }
    this.metrics.get(agentName)!.push(metrics);
  }

  getAverageMetrics(agentName: string): ExecutionMetrics | null {
    const agentMetrics = this.metrics.get(agentName);
    if (!agentMetrics || agentMetrics.length === 0) return null;

    const avg = agentMetrics.reduce(
      (acc, m) => ({
        duration: acc.duration + m.duration,
        tokenCount: acc.tokenCount + m.tokenCount,
        toolCalls: acc.toolCalls + m.toolCalls,
        errorRate: acc.errorRate + m.errorRate,
      }),
      { duration: 0, tokenCount: 0, toolCalls: 0, errorRate: 0 }
    );

    const count = agentMetrics.length;

    return {
      duration: avg.duration / count,
      tokenCount: avg.tokenCount / count,
      toolCalls: avg.toolCalls / count,
      errorRate: avg.errorRate / count,
    };
  }
}

// Usage
const collector = new MetricsCollector();
const { result, metrics } = await collector.executeWithMetrics(agent, 'Query');
console.log(`Duration: ${metrics.duration}ms, Tokens: ${metrics.tokenCount}`);
```

---

## Guardrails System

### Input Validation Guardrails

A guardrail is an object with a `name` and an `execute` function returning `{ tripwireTriggered, outputInfo }`. When a tripwire fires, `run()` throws `InputGuardrailTripwireTriggered` (or `OutputGuardrailTripwireTriggered`):

```typescript
import { Agent, run, InputGuardrail, InputGuardrailTripwireTriggered } from '@openai/agents';

const toText = (input: unknown) => (typeof input === 'string' ? input : JSON.stringify(input));

// PII detection guardrail
const piiGuardrail: InputGuardrail = {
  name: 'pii_detector',
  execute: async ({ input }) => {
    const text = toText(input);
    const piiPatterns = [
      /\b\d{3}-\d{2}-\d{4}\b/, // SSN
      /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/, // Credit card
    ];
    const found = piiPatterns.some((pattern) => pattern.test(text));
    return { tripwireTriggered: found, outputInfo: { found } };
  },
};

// Guardrails can call a cheap model themselves
const topicGuardrail: InputGuardrail = {
  name: 'off_topic',
  runInParallel: false, // block the main agent until this finishes
  execute: async ({ input, context }) => {
    const checker = new Agent({
      name: 'Topic checker',
      model: 'gpt-5.6-luna',
      instructions: 'Answer "yes" if the request is about our product, otherwise "no".',
    });
    const verdict = await run(checker, toText(input), { context });
    return { tripwireTriggered: verdict.finalOutput?.trim().toLowerCase() !== 'yes', outputInfo: verdict.finalOutput };
  },
};

const agent = new Agent({
  name: 'Secure Agent',
  instructions: 'Process user requests securely',
  inputGuardrails: [piiGuardrail, topicGuardrail],
});

try {
  await run(agent, 'My SSN is 123-45-6789');
} catch (error) {
  if (error instanceof InputGuardrailTripwireTriggered) {
    console.error('Input rejected by guardrail:', error.result.guardrail.name);
  } else {
    throw error;
  }
}
```

### Output Content Filtering

```typescript
import { Agent, run, OutputGuardrail, OutputGuardrailTripwireTriggered } from '@openai/agents';

// Sensitive information filter
const sensitiveDataGuardrail: OutputGuardrail = {
  name: 'sensitive_data_filter',
  execute: async ({ agentOutput }) => {
    const text = String(agentOutput);
    const leaked = [/api[_-]?key/i, /password\s*[:=]/i, /secret/i].some((p) => p.test(text));
    return { tripwireTriggered: leaked, outputInfo: { leaked } };
  },
};

const agent = new Agent({
  name: 'Filtered Agent',
  instructions: 'Provide helpful responses',
  outputGuardrails: [sensitiveDataGuardrail],
});

try {
  const result = await run(agent, 'Show me the deployment config');
  console.log(result.finalOutput);
} catch (error) {
  if (error instanceof OutputGuardrailTripwireTriggered) {
    console.log('Response withheld by content policy.');
  } else {
    throw error;
  }
}
```

Guardrails can also be attached to individual function tools (`inputGuardrails` / `outputGuardrails` on `tool()`, built with `defineToolInputGuardrail` / `defineToolOutputGuardrail`).

---

## Handoffs and Delegation

### Type-Safe Handoff Patterns

```typescript
import { Agent, Handoff, run } from '@openai/agents';

// Define specialist agents
const billingSpecialist = new Agent({
  name: 'Billing Specialist',
  handoffDescription: 'Handles billing and payment inquiries',
  instructions: 'Resolve billing issues professionally and accurately',
});

const technicalSupport = new Agent({
  name: 'Technical Support',
  handoffDescription: 'Handles technical issues and troubleshooting',
  instructions: 'Provide step-by-step technical assistance',
});

const accountManagement = new Agent({
  name: 'Account Management',
  handoffDescription: 'Handles account changes and settings',
  instructions: 'Help with account configuration and settings',
});

// Triage agent with typed handoffs
const triageAgent = new Agent({
  name: 'Customer Service Triage',
  instructions: `Route customer inquiries to appropriate specialist:
    - Billing/payment issues → Billing Specialist
    - Technical problems → Technical Support
    - Account changes → Account Management`,
  handoffs: [billingSpecialist, technicalSupport, accountManagement],
});

// Execute with automatic routing
const result = await run(triageAgent, 'I was charged twice this month');
console.log(`Routed to: ${result.lastAgent?.name}`);
console.log(`Response: ${result.finalOutput}`);
```

### Conditional Handoff Logic

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

// Route based on classification
const classifyQueryTool = tool({
  name: 'classify_query',
  description: 'Classify customer query type',
  parameters: z.object({
    query: z.string(),
  }),
  execute: async ({ query }): Promise<{
    type: 'billing' | 'technical' | 'account' | 'general';
    confidence: number;
  }> => {
    const query_lower = query.toLowerCase();

    if (
      query_lower.includes('charge') ||
      query_lower.includes('payment') ||
      query_lower.includes('invoice')
    ) {
      return { type: 'billing', confidence: 0.95 };
    }

    if (
      query_lower.includes('error') ||
      query_lower.includes('crash') ||
      query_lower.includes('bug')
    ) {
      return { type: 'technical', confidence: 0.9 };
    }

    if (
      query_lower.includes('password') ||
      query_lower.includes('settings') ||
      query_lower.includes('profile')
    ) {
      return { type: 'account', confidence: 0.85 };
    }

    return { type: 'general', confidence: 0.6 };
  },
});

const routerAgent = new Agent({
  name: 'Smart Router',
  instructions: 'Classify query and route to appropriate specialist',
  tools: [classifyQueryTool],
  handoffs: [billingSpecialist, technicalSupport, accountManagement],
});
```

---

## MCP Integration

### Filesystem Operations

```typescript
import { Agent, run, MCPServerStdio } from '@openai/agents';

async function filesystemExample() {
  const filesystemServer = new MCPServerStdio({
    name: 'filesystem',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-filesystem', './project'],
  });
  await filesystemServer.connect();

  try {
    const agent = new Agent({
      name: 'File Manager',
      instructions: 'Help manage project files and directories',
      mcpServers: [filesystemServer],
    });

    const result = await run(agent, 'List all TypeScript files in src/');
    console.log(result.finalOutput);
  } finally {
    await filesystemServer.close();
  }
}
```

### Git Integration

```typescript
import { Agent, run, MCPServerStdio } from '@openai/agents';

async function gitExample() {
  const gitServer = new MCPServerStdio({
    name: 'git',
    fullCommand: 'uvx mcp-server-git --repository ./repo',
  });
  await gitServer.connect();

  try {
    const agent = new Agent({
      name: 'Git Assistant',
      instructions: 'Help with git operations and code review',
      mcpServers: [gitServer],
    });

    const result = await run(agent, 'Show recent commits and their changes');
    console.log(result.finalOutput);
  } finally {
    await gitServer.close();
  }
}
```

### HTTP MCP Servers

```typescript
import { Agent, run, MCPServerStreamableHttp, hostedMcpTool } from '@openai/agents';

async function httpMCPExample() {
  // Local SDK-managed connection to a Streamable HTTP server
  const apiServer = new MCPServerStreamableHttp({
    name: 'api_server',
    url: 'https://api.example.com/mcp',
    requestInit: {
      headers: { Authorization: `Bearer ${process.env.API_TOKEN}` },
    },
    // Discovery stops after 64 pages by default since 0.20.0
    // maxListPages: 200,
  });
  await apiServer.connect();

  try {
    const agent = new Agent({
      name: 'API Assistant',
      instructions: 'Interact with external APIs via MCP',
      mcpServers: [apiServer],
    });

    const result = await run(agent, 'Fetch latest data from the API');
    console.log(result.finalOutput);
  } finally {
    await apiServer.close();
  }
}

// Alternatively, let the Responses API call a public MCP server itself
const hostedAgent = new Agent({
  name: 'Docs Assistant',
  instructions: 'Answer questions using the docs server.',
  tools: [hostedMcpTool({ serverLabel: 'docs', serverUrl: 'https://example.com/mcp' })],
});
```

`connectMcpServers()` connects several servers at once and closes them together.

---

## Provider-Agnostic Support

### Multiple LLM Providers

OpenAI models are referenced by name. Other providers go through the Vercel AI SDK adapter (`npm install @openai/agents-extensions` plus the provider package, e.g. `@ai-sdk/anthropic`). There is no `litellm/` model prefix in the TypeScript SDK.

```typescript
import { Agent, run } from '@openai/agents';
import { aisdk } from '@openai/agents-extensions/ai-sdk';
import { anthropic } from '@ai-sdk/anthropic';
import { google } from '@ai-sdk/google';

// Configure different providers
const providers = {
  openai: new Agent({
    name: 'OpenAI Assistant',
    model: 'gpt-5.6-sol',
    instructions: 'Answer clearly and concisely.',
  }),

  claude: new Agent({
    name: 'Claude Assistant',
    model: aisdk(anthropic('claude-sonnet-4-5')),
    instructions: 'Answer clearly and concisely.',
  }),

  gemini: new Agent({
    name: 'Gemini Assistant',
    model: aisdk(google('gemini-2.5-flash')),
    instructions: 'Answer clearly and concisely.',
  }),
};

// Compare responses across providers
async function compareProviders(query: string) {
  const results = await Promise.all(
    Object.entries(providers).map(async ([name, agent]) => {
      const result = await run(agent, query);
      return {
        provider: name,
        response: result.finalOutput,
      };
    })
  );

  return results;
}

// Usage
const comparison = await compareProviders('Explain TypeScript generics');
comparison.forEach(({ provider, response }) => {
  console.log(`\n${provider.toUpperCase()}:\n${response}\n`);
});
```

---

## Type-Safe Tools

### Zod-Powered Validation

```typescript
import { tool } from '@openai/agents';
import { z } from 'zod';

// Complex validation with Zod
const createOrderTool = tool({
  name: 'create_order',
  description: 'Create a new order',
  parameters: z.object({
    customerId: z.string().uuid(),
    items: z.array(
      z.object({
        productId: z.string(),
        quantity: z.number().int().positive(),
        price: z.number().positive(),
      })
    ).min(1).max(100),
    shippingAddress: z.object({
      street: z.string(),
      city: z.string(),
      state: z.string().length(2),
      zipCode: z.string().regex(/^\d{5}(-\d{4})?$/),
      country: z.string().length(2),
    }),
    paymentMethod: z.enum(['credit_card', 'debit_card', 'paypal', 'apple_pay']),
  }).refine(
    (data) => {
      const total = data.items.reduce((sum, item) => sum + item.price * item.quantity, 0);
      return total > 0;
    },
    { message: 'Order total must be greater than 0' }
  ),
  execute: async (order) => {
    // Full type safety in execution
    const total = order.items.reduce(
      (sum, item) => sum + item.price * item.quantity,
      0
    );

    return {
      orderId: `ORD-${Date.now()}`,
      customerId: order.customerId,
      total,
      status: 'confirmed',
    };
  },
});
```

---

## Production Features

### Error Handling

`run()` has no timeout or retry options; use an `AbortSignal` for timeouts, model-level `retry` settings for transient API failures, and the SDK's error classes to tell failures apart:

```typescript
import {
  Agent,
  run,
  MaxTurnsExceededError,
  ModelBehaviorError,
  ModelRefusalError,
  InputGuardrailTripwireTriggered,
} from '@openai/agents';

async function robustExecution(agent: Agent, input: string): Promise<string> {
  try {
    const result = await run(agent, input, {
      signal: AbortSignal.timeout(30_000), // 30 second timeout
      maxTurns: 8,
    });

    return result.finalOutput ?? '';
  } catch (error) {
    if (error instanceof MaxTurnsExceededError) {
      console.error('Agent exceeded its turn limit');
    } else if (error instanceof ModelRefusalError) {
      console.error('Model refused the request');
    } else if (error instanceof ModelBehaviorError) {
      console.error('Model produced invalid output');
    } else if (error instanceof InputGuardrailTripwireTriggered) {
      console.error('Blocked by input guardrail');
    } else if (error instanceof Error && error.name === 'TimeoutError') {
      console.error('Agent execution timed out');
    } else {
      console.error('Unexpected error:', error);
    }

    throw error;
  }
}
```

### Cost Tracking

```typescript
import { Agent, run } from '@openai/agents';

interface UsageMetrics {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
}

// USD per 1M tokens — placeholder figures; load current prices from config
const PRICES: Record<string, { input: number; output: number }> = {
  'gpt-5.6-luna': { input: 0, output: 0 },
  'gpt-5.6-sol': { input: 0, output: 0 },
};

class CostTracker {
  private totalCost = 0;
  private usageHistory: UsageMetrics[] = [];

  async executeWithTracking(
    agent: Agent,
    input: string
  ): Promise<{ result: string; metrics: UsageMetrics }> {
    const result = await run(agent, input);
    const usage = result.runContext.usage;
    const price = PRICES[String(agent.model)] ?? { input: 0, output: 0 };

    const metrics: UsageMetrics = {
      inputTokens: usage.inputTokens,
      outputTokens: usage.outputTokens,
      totalTokens: usage.totalTokens,
      cost: (usage.inputTokens * price.input + usage.outputTokens * price.output) / 1_000_000,
    };

    this.totalCost += metrics.cost;
    this.usageHistory.push(metrics);

    return {
      result: result.finalOutput ?? '',
      metrics,
    };
  }

  getTotalCost(): number {
    return this.totalCost;
  }
}
```

---

## What changed in 2026

The packages moved from 0.3 to %%version=@openai/agents%% during 2026; all `@openai/agents*` packages share one version number. Minor releases may contain breaking changes. Highlights:

| Release | What changed |
|---------|--------------|
| 0.8 | Human-in-the-loop approvals with serialisable `RunState`; browser bundle improvements; Zod 4. |
| 0.9–0.11 | `OpenAIConversationsSession`, `OpenAIResponsesCompactionSession`, `MemorySession`; `shellTool`, `applyPatchTool`, `toolSearchTool`; `MCPServerStreamableHttp`. |
| 0.13 | Default Realtime model `gpt-realtime-2.1`. |
| 0.14 | Sensitive model/tool data no longer logged by default; default task and turn tracing. |
| 0.15 | Requires the `openai` Node SDK v7.2+; default model `gpt-5.6-luna`; newer MCP protocol negotiation. |
| 0.16 | Testing utilities: scripted models, sandbox sessions, Realtime (`@openai/agents/testing`). |
| 0.17–0.19 | Security hardening: blocked tool output redacted from replay state; sandbox file operations hardened; resumed MCP calls bound to their original server; agent-tool stream backlog capped at 1024 events. |
| 0.20 | MCP tool discovery capped at 64 pages by default (`maxListPages`); model calls after approval resumes count towards `maxTurns`. |
| 0.21 | Runloop sandbox provider removed; usage-detail values need narrowing before arithmetic. |

Sandbox agents are available in TypeScript too, via `@openai/agents/sandbox` (local backends) and `@openai/agents-extensions/sandbox/*` (hosted providers).

---

## Summary

The TypeScript Agents SDK provides:

✅ **TypeScript-First** design with full type safety
✅ **Human-in-the-Loop** approvals with resumable run state
✅ **Built-in Tracing** and observability
✅ **Guardrails** for input/output validation
✅ **Handoffs** for agent delegation
✅ **MCP Integration** for filesystem, git, HTTP
✅ **Provider-Agnostic** support via the Vercel AI SDK adapter
✅ **Type-Safe Tools** with Zod validation
✅ **Production Features** (error handling, cost tracking)

---

**Next:** [Swarm Migration Guide](./openai_agents_sdk_typescript_swarm_migration_guide/)


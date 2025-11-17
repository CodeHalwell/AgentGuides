# OpenAI Agents SDK TypeScript: Critical 2025 Features and Updates

**Status:** Production-Ready
**Last Updated:** January 2025
**SDK Version:** Latest
**Language:** TypeScript
**TypeScript Version:** 5.0+

---

## Overview

The OpenAI Agents SDK for TypeScript is the **official production-ready framework** for building type-safe agentic AI applications. This guide covers all critical 2025 features that make TypeScript Agents SDK the definitive choice for modern development.

---

## Table of Contents

1. [Production-Ready Status](#production-ready-status)
2. [TypeScript-Specific 2025 Features](#typescript-specific-2025-features)
3. [Human-in-the-Loop Approvals](#human-in-the-loop-approvals)
4. [Built-in Tracing and Observability](#built-in-tracing-and-observability)
5. [Guardrails System](#guardrails-system)
6. [Handoffs and Delegation](#handoffs-and-delegation)
7. [MCP Integration](#mcp-integration)
8. [Provider-Agnostic Support](#provider-agnostic-support)
9. [Type-Safe Tools](#type-safe-tools)
10. [Production Features](#production-features)

---

## Production-Ready Status

### Why Choose TypeScript Agents SDK (2025)?

- ✅ **TypeScript-First Design**: Full type safety, excellent IDE support
- ✅ **Production-Ready**: Stable API, comprehensive testing
- ✅ **Active Maintenance**: Regular updates and feature additions
- ✅ **Enterprise Features**: Observability, security, scalability
- ✅ **No Experimental Code**: Unlike Swarm, fully production-supported

### Swarm Replacement

**The Agents SDK is the official production replacement for experimental Swarm implementations.**

```typescript
// BEFORE: Swarm-like (Experimental)
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

## TypeScript-Specific 2025 Features

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
function createTypedAgent<TOutput extends z.ZodSchema>(config: {
  name: string;
  instructions: string;
  outputSchema: TOutput;
  tools?: any[];
}): Agent {
  return new Agent({
    name: config.name,
    instructions: config.instructions,
    outputSchema: config.outputSchema,
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
  outputSchema: SentimentSchema,
});

// Full type safety in results
const result = await run(sentimentAgent, 'I love TypeScript!');
const analysis: SentimentAnalysis = JSON.parse(result.finalOutput);
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

  return result.finalOutput;
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

### NEW 2025: Approval Workflow

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

// Critical action requiring human approval
const deleteUserTool = tool({
  name: 'delete_user',
  description: 'Delete user account permanently',
  parameters: z.object({
    userId: z.string().uuid(),
    reason: z.string().min(20),
  }),
  requiresApproval: true, // NEW: Human approval required
  execute: async ({ userId, reason }) => {
    return {
      status: 'deleted',
      userId,
      timestamp: new Date().toISOString(),
    };
  },
});

const moderatorTool = tool({
  name: 'ban_user',
  description: 'Ban user from platform',
  parameters: z.object({
    userId: z.string().uuid(),
    duration: z.enum(['temporary', 'permanent']),
  }),
  requiresApproval: true,
  execute: async ({ userId, duration }) => {
    return { status: 'banned', userId, duration };
  },
});

const adminAgent = new Agent({
  name: 'Admin Agent',
  instructions: 'Manage users with appropriate caution',
  tools: [deleteUserTool, moderatorTool],
});

// Approval handler
interface ApprovalRequest {
  toolName: string;
  parameters: Record<string, any>;
  agentName: string;
}

interface ApprovalResponse {
  approved: boolean;
  reason?: string;
  modifiedParameters?: Record<string, any>;
}

async function customApprovalHandler(
  request: ApprovalRequest
): Promise<ApprovalResponse> {
  console.log(`\n🔔 APPROVAL REQUIRED`);
  console.log(`Tool: ${request.toolName}`);
  console.log(`Agent: ${request.agentName}`);
  console.log(`Parameters:`, request.parameters);

  // In production: Show UI, send notification, log to audit system
  const approved = await getUserConfirmation(request);

  if (approved) {
    return {
      approved: true,
      reason: 'Approved by administrator',
    };
  } else {
    return {
      approved: false,
      reason: 'Rejected by administrator',
    };
  }
}

// Execute with approval workflow
const result = await run(
  adminAgent,
  'Delete user account user_123 due to Terms of Service violation',
  {
    approvalHandler: customApprovalHandler,
  }
);

// Slack Integration Example
async function slackApprovalHandler(
  request: ApprovalRequest
): Promise<ApprovalResponse> {
  const slackMessage = {
    channel: '#admin-approvals',
    text: 'Action Requires Approval',
    blocks: [
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `*${request.toolName}* requires approval\n\`\`\`${JSON.stringify(request.parameters, null, 2)}\`\`\``,
        },
      },
      {
        type: 'actions',
        elements: [
          {
            type: 'button',
            text: { type: 'plain_text', text: 'Approve' },
            value: 'approve',
            action_id: 'approve_action',
          },
          {
            type: 'button',
            text: { type: 'plain_text', text: 'Reject' },
            value: 'reject',
            action_id: 'reject_action',
          },
        ],
      },
    ],
  };

  // Send to Slack and wait for response
  const approval = await sendSlackApprovalRequest(slackMessage);

  return {
    approved: approval.action === 'approve',
    reason: approval.reason,
  };
}
```

### Conditional Approval Rules

```typescript
interface ApprovalRules {
  requireApprovalIf: (params: any) => boolean;
  approvers: string[];
  timeout: number;
}

const approvalRules: Record<string, ApprovalRules> = {
  delete_user: {
    requireApprovalIf: (params) => params.userId.startsWith('admin_'),
    approvers: ['senior_admin@company.com'],
    timeout: 3600000, // 1 hour
  },
  modify_billing: {
    requireApprovalIf: (params) => params.amount > 1000,
    approvers: ['finance@company.com'],
    timeout: 7200000, // 2 hours
  },
};

async function ruleBasedApprovalHandler(
  request: ApprovalRequest
): Promise<ApprovalResponse> {
  const rules = approvalRules[request.toolName];

  if (!rules) {
    return { approved: true }; // No rules = auto-approve
  }

  if (!rules.requireApprovalIf(request.parameters)) {
    return { approved: true }; // Doesn't meet threshold
  }

  // Request approval from designated approvers
  const approval = await requestApprovalFrom(rules.approvers, request, rules.timeout);

  return {
    approved: approval.approved,
    reason: approval.reason,
  };
}
```

---

## Built-in Tracing and Observability

### Workflow Visualization

```typescript
import { Agent, run, trace } from '@openai/agents';

const agent = new Agent({
  name: 'Data Processor',
  instructions: 'Process and analyze data',
});

// Group related operations
await trace('Customer Data Pipeline', async () => {
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
import { Agent, run, trace } from '@openai/agents';
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

    const result = await trace(
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
        tokenCount: result.usage?.totalTokens || 0,
        toolCalls: result.toolCallCount || 0,
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

```typescript
import { Agent, run } from '@openai/agents';
import { z } from 'zod';

interface InputGuardrail<T> {
  name: string;
  schema?: z.ZodSchema<T>;
  validate: (input: string) => Promise<boolean>;
  onViolation?: (input: string) => string | Promise<string>;
}

// Email validation guardrail
const emailGuardrail: InputGuardrail<{ email: string }> = {
  name: 'email_validator',
  schema: z.object({
    email: z.string().email(),
  }),
  validate: async (input: string) => {
    const emailPattern = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i;
    return emailPattern.test(input);
  },
  onViolation: (input) => {
    return 'Invalid email format provided. Please provide a valid email address.';
  },
};

// PII detection guardrail
const piiGuardrail: InputGuardrail<string> = {
  name: 'pii_detector',
  validate: async (input: string) => {
    const piiPatterns = [
      /\b\d{3}-\d{2}-\d{4}\b/, // SSN
      /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/, // Credit card
    ];

    return !piiPatterns.some(pattern => pattern.test(input));
  },
  onViolation: () => {
    return 'Input contains PII and has been blocked for security reasons.';
  },
};

const agent = new Agent({
  name: 'Secure Agent',
  instructions: 'Process user requests securely',
  inputGuardrails: [emailGuardrail, piiGuardrail],
});

// Execute with guardrails
try {
  const result = await run(agent, 'My SSN is 123-45-6789');
} catch (error) {
  console.error('Input rejected by guardrail:', error.message);
}
```

### Output Content Filtering

```typescript
import { Agent, run } from '@openai/agents';

interface OutputGuardrail {
  name: string;
  check: (output: string) => boolean | Promise<boolean>;
  filter?: (output: string) => string | Promise<string>;
}

// Profanity filter
const profanityGuardrail: OutputGuardrail = {
  name: 'profanity_filter',
  check: async (output: string) => {
    const bannedWords = ['profanity1', 'profanity2', 'offensive'];
    return !bannedWords.some(word => output.toLowerCase().includes(word));
  },
  filter: (output: string) => {
    return 'Response has been filtered due to content policy violation.';
  },
};

// Sensitive information filter
const sensitiveDataGuardrail: OutputGuardrail = {
  name: 'sensitive_data_filter',
  check: async (output: string) => {
    const sensitivePatterns = [
      /api[_-]?key/i,
      /password/i,
      /secret/i,
      /token/i,
    ];

    return !sensitivePatterns.some(pattern => pattern.test(output));
  },
  filter: (output: string) => {
    // Redact sensitive information
    return output
      .replace(/api[_-]?key\s*[:=]\s*[\w-]+/gi, 'api_key: [REDACTED]')
      .replace(/password\s*[:=]\s*[\w-]+/gi, 'password: [REDACTED]');
  },
};

const agent = new Agent({
  name: 'Filtered Agent',
  instructions: 'Provide helpful responses',
  outputGuardrails: [profanityGuardrail, sensitiveDataGuardrail],
});
```

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
console.log(`Routed to: ${result.currentAgent?.name}`);
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
import { Agent, run } from '@openai/agents';
import { createMCPServer } from '@openai/agents/mcp';

async function filesystemExample() {
  const filesystemServer = await createMCPServer({
    name: 'filesystem',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-filesystem', './project'],
  });

  const agent = new Agent({
    name: 'File Manager',
    instructions: 'Help manage project files and directories',
    mcpServers: [filesystemServer],
  });

  const result = await run(agent, 'List all TypeScript files in src/');
  console.log(result.finalOutput);

  await filesystemServer.close();
}
```

### Git Integration

```typescript
import { createMCPServer } from '@openai/agents/mcp';

async function gitExample() {
  const gitServer = await createMCPServer({
    name: 'git',
    transport: 'stdio',
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-git', './repo'],
  });

  const agent = new Agent({
    name: 'Git Assistant',
    instructions: 'Help with git operations and code review',
    mcpServers: [gitServer],
  });

  const result = await run(agent, 'Show recent commits and their changes');
  console.log(result.finalOutput);

  await gitServer.close();
}
```

### HTTP MCP Servers

```typescript
import { createMCPServer } from '@openai/agents/mcp';

async function httpMCPExample() {
  const apiServer = await createMCPServer({
    name: 'api_server',
    transport: 'http',
    url: 'https://api.example.com/mcp',
    headers: {
      'Authorization': `Bearer ${process.env.API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    timeout: 30000,
  });

  const agent = new Agent({
    name: 'API Assistant',
    instructions: 'Interact with external APIs via MCP',
    mcpServers: [apiServer],
  });

  const result = await run(agent, 'Fetch latest data from the API');
  console.log(result.finalOutput);

  await apiServer.close();
}
```

---

## Provider-Agnostic Support

### Multiple LLM Providers

```typescript
import { Agent, run } from '@openai/agents';

// Configure different providers
const providers = {
  openai: new Agent({
    name: 'OpenAI Assistant',
    model: 'gpt-4o',
    instructions: 'You are GPT-4',
  }),

  claude: new Agent({
    name: 'Claude Assistant',
    model: 'litellm/anthropic/claude-3-5-sonnet-20240620',
    instructions: 'You are Claude 3.5 Sonnet',
  }),

  gemini: new Agent({
    name: 'Gemini Assistant',
    model: 'litellm/gemini/gemini-2.0-flash',
    instructions: 'You are Gemini 2.0',
  }),

  llama: new Agent({
    name: 'Llama Assistant',
    model: 'litellm/replicate/meta-llama/llama-2-70b-chat',
    instructions: 'You are Llama 2',
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

```typescript
import { Agent, run } from '@openai/agents';

async function robustExecution(agent: Agent, input: string): Promise<string> {
  try {
    const result = await run(agent, input, {
      timeout: 30000, // 30 second timeout
      maxRetries: 3,
      retryDelay: 1000,
    });

    return result.finalOutput;
  } catch (error) {
    if (error.code === 'TIMEOUT') {
      console.error('Agent execution timed out');
    } else if (error.code === 'RATE_LIMIT') {
      console.error('Rate limit exceeded');
    } else {
      console.error('Unexpected error:', error);
    }

    throw error;
  }
}
```

### Cost Tracking

```typescript
interface UsageMetrics {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
}

class CostTracker {
  private totalCost = 0;
  private usageHistory: UsageMetrics[] = [];

  async executeWithTracking(
    agent: Agent,
    input: string
  ): Promise<{ result: string; metrics: UsageMetrics }> {
    const result = await run(agent, input);

    const metrics: UsageMetrics = {
      inputTokens: result.usage?.inputTokens || 0,
      outputTokens: result.usage?.outputTokens || 0,
      totalTokens: result.usage?.totalTokens || 0,
      cost: this.calculateCost(result.usage),
    };

    this.totalCost += metrics.cost;
    this.usageHistory.push(metrics);

    return {
      result: result.finalOutput,
      metrics,
    };
  }

  private calculateCost(usage: any): number {
    // GPT-4 pricing example
    const inputCost = (usage?.inputTokens || 0) * 0.00003;
    const outputCost = (usage?.outputTokens || 0) * 0.00006;
    return inputCost + outputCost;
  }

  getTotalCost(): number {
    return this.totalCost;
  }
}
```

---

## Summary

TypeScript Agents SDK (2025) provides:

✅ **TypeScript-First** design with full type safety
✅ **Human-in-the-Loop** approvals (NEW 2025)
✅ **Built-in Tracing** and observability
✅ **Guardrails** for input/output validation
✅ **Handoffs** for agent delegation
✅ **MCP Integration** for filesystem, git, HTTP
✅ **Provider-Agnostic** support (100+ LLMs)
✅ **Type-Safe Tools** with Zod validation
✅ **Production Features** (error handling, cost tracking)

**Recommendation**: Use TypeScript Agents SDK for all production agentic applications.

---

**Last Updated:** January 2025
**SDK Version:** Latest
**TypeScript Version:** 5.0+
**Next:** [Swarm Migration Guide](openai_agents_sdk_typescript_swarm_migration_guide.md)

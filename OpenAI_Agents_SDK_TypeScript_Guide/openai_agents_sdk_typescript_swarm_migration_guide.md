# OpenAI Agents SDK TypeScript: Complete Migration Guide from Swarm

**Status:** Production-Ready (2025)
**Swarm Status:** Experimental - Deprecated
**Migration Priority:** HIGH
**Last Updated:** January 2025
**Language:** TypeScript

---

## Executive Summary

The **OpenAI Agents SDK for TypeScript** is the official, production-ready framework for building agentic AI applications. This guide helps teams migrate from experimental Swarm implementations to the robust, type-safe Agents SDK.

**Key Decision Factors:**

- ✅ **Agents SDK**: Production-ready, TypeScript-first, comprehensive features
- ❌ **Swarm**: Experimental, limited TypeScript support, no production guarantees

---

## Table of Contents

1. [Why Migrate to TypeScript Agents SDK?](#why-migrate-to-typescript-agents-sdk)
2. [2025 Features and Improvements](#2025-features-and-improvements)
3. [Side-by-Side Comparison](#side-by-side-comparison)
4. [TypeScript Migration Checklist](#typescript-migration-checklist)
5. [Code Migration Examples](#code-migration-examples)
6. [Type Safety Improvements](#type-safety-improvements)
7. [Testing Strategy](#testing-strategy)
8. [Production Deployment](#production-deployment)

---

## Why Migrate to TypeScript Agents SDK?

### Swarm Limitations (Experimental)

1. **No TypeScript-First Design**: Limited type definitions, weak inference
2. **Experimental Status**: No production guarantees, unstable API
3. **Limited Features**: No guardrails, tracing, or session management
4. **Poor IDE Support**: Minimal autocomplete and type checking

### Agents SDK Advantages (Production)

1. **TypeScript-First**: Full type safety, excellent IDE support
2. **Production-Ready**: Stable API, comprehensive testing
3. **2025 Features**:
   - ✨ Human-in-the-loop approvals (NEW)
   - ✨ Built-in tracing and visualization
   - ✨ Guardrails for input/output validation
   - ✨ MCP (Model Context Protocol) integration
   - ✨ Provider-agnostic support (100+ LLMs)
4. **Enterprise-Grade**: Scalability, observability, security

---

## 2025 Features and Improvements

### 1. Human-in-the-Loop Approvals (NEW 2025)

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

// Tool requiring human approval
const criticalActionTool = tool({
  name: 'delete_user_data',
  description: 'Delete user data (requires approval)',
  parameters: z.object({
    userId: z.string(),
    reason: z.string(),
  }),
  requiresApproval: true, // NEW: Human approval required
  execute: async ({ userId, reason }) => {
    // This will pause and request human approval
    return { status: 'deleted', userId, reason };
  },
});

const agent = new Agent({
  name: 'Admin Agent',
  instructions: 'Manage user data with caution',
  tools: [criticalActionTool],
});

// Approval workflow
const result = await run(agent, 'Delete user data for user_123', {
  approvalHandler: async (toolCall) => {
    // Custom approval logic
    console.log(`Approval requested for: ${toolCall.name}`);
    const approved = await getUserApproval(toolCall);
    return { approved, reason: 'User confirmed' };
  },
});
```

### 2. Built-in Tracing and Visualization

```typescript
import { Agent, run, trace } from '@openai/agents';

// Group related runs with tracing
await trace('Customer Support Workflow', async () => {
  const result1 = await run(triageAgent, query1);
  const result2 = await run(specialistAgent, query2);

  return { result1, result2 };
}, {
  metadata: {
    environment: 'production',
    customer_id: 'cust_123',
  },
});

// View traces at https://platform.openai.com/traces
```

### 3. Guardrails (Input/Output Validation)

```typescript
import { Agent, run, inputGuardrail, outputGuardrail } from '@openai/agents';
import { z } from 'zod';

// Input validation guardrail
const emailGuardrail = inputGuardrail({
  name: 'validate_email',
  schema: z.object({
    email: z.string().email(),
  }),
  onViolation: (input) => {
    throw new Error(`Invalid email: ${input.email}`);
  },
});

// Output content filter
const contentFilterGuardrail = outputGuardrail({
  name: 'filter_profanity',
  check: (output: string) => {
    const bannedWords = ['profanity1', 'profanity2'];
    return !bannedWords.some(word => output.toLowerCase().includes(word));
  },
  onViolation: (output) => {
    return 'Response blocked due to content policy violation';
  },
});

const agent = new Agent({
  name: 'Safe Assistant',
  instructions: 'Help users safely',
  inputGuardrails: [emailGuardrail],
  outputGuardrails: [contentFilterGuardrail],
});
```

### 4. Model Context Protocol (MCP) Integration

```typescript
import { Agent, run } from '@openai/agents';
import { createMCPServer } from '@openai/agents/mcp';

// Filesystem MCP integration
const filesystemServer = await createMCPServer({
  name: 'filesystem',
  transport: 'stdio',
  command: 'npx',
  args: ['-y', '@modelcontextprotocol/server-filesystem', './project'],
});

const agent = new Agent({
  name: 'File Assistant',
  instructions: 'Help manage files using MCP',
  mcpServers: [filesystemServer],
});

const result = await run(agent, 'List all TypeScript files in the project');
```

### 5. Provider-Agnostic Support

```typescript
import { Agent, run } from '@openai/agents';

// Use Claude
const claudeAgent = new Agent({
  name: 'Claude Assistant',
  model: 'litellm/anthropic/claude-3-5-sonnet-20240620',
  instructions: 'You are Claude',
});

// Use Gemini
const geminiAgent = new Agent({
  name: 'Gemini Assistant',
  model: 'litellm/gemini/gemini-2.0-flash',
  instructions: 'You are Gemini',
});

// Use Llama
const llamaAgent = new Agent({
  name: 'Llama Assistant',
  model: 'litellm/replicate/meta-llama/llama-2-70b-chat',
  instructions: 'You are Llama',
});

// Switch providers seamlessly
const result = await run(claudeAgent, 'Explain TypeScript generics');
```

### 6. Enhanced Session Management

```typescript
import { Agent, run, Session } from '@openai/agents';

interface ConversationMetadata {
  userId: string;
  tags: string[];
}

// Type-safe session
const session = new Session<ConversationMetadata>({
  id: 'user_123',
  metadata: {
    userId: 'user_123',
    tags: ['support', 'billing'],
  },
  maxMessages: 50,
  ttl: 3600, // 1 hour
});

const agent = new Agent({
  name: 'Support Agent',
  instructions: 'Provide customer support',
});

// Turn 1
const result1 = await run(agent, 'My name is Alice', { session });

// Turn 2 - automatic context
const result2 = await run(agent, 'What is my name?', { session });
// Response: "Your name is Alice"
```

---

## Side-by-Side Comparison

### Basic Agent Creation

```typescript
// ============================================
// SWARM (Experimental - Python-based)
// ============================================
// Note: Swarm has limited TypeScript support
import { Swarm, Agent } from 'swarm';

const client = new Swarm();

const agent = new Agent({
  name: 'Assistant',
  instructions: 'You are helpful',
});

const result = client.run({
  agent,
  messages: [{ role: 'user', content: 'Hello' }],
});

// ============================================
// AGENTS SDK (Production TypeScript)
// ============================================
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'Assistant',
  instructions: 'You are helpful',
});

const result = await run(agent, 'Hello');
console.log(result.finalOutput);
```

### Type-Safe Tools

```typescript
// ============================================
// SWARM (Limited Types)
// ============================================
function getWeather(location: string): string {
  return `Weather in ${location}`;
}

const agent = new Agent({
  name: 'Weather',
  functions: [getWeather],
});

// ============================================
// AGENTS SDK (Full Type Safety)
// ============================================
import { tool } from '@openai/agents';
import { z } from 'zod';

interface WeatherResult {
  location: string;
  temperature: number;
  condition: string;
}

const getWeatherTool = tool({
  name: 'get_weather',
  description: 'Get weather information',
  parameters: z.object({
    location: z.string().describe('City name'),
    units: z.enum(['celsius', 'fahrenheit']).default('celsius'),
  }),
  execute: async ({ location, units }): Promise<WeatherResult> => {
    return {
      location,
      temperature: 72,
      condition: 'Sunny',
    };
  },
});

const agent = new Agent({
  name: 'Weather',
  tools: [getWeatherTool],
});
```

### Agent Handoffs

```typescript
// ============================================
// SWARM
// ============================================
function transferToSpecialist() {
  return specialistAgent;
}

const triageAgent = new Agent({
  name: 'Triage',
  functions: [transferToSpecialist],
});

// ============================================
// AGENTS SDK
// ============================================
import { Agent, Handoff } from '@openai/agents';

const specialistAgent = new Agent({
  name: 'Specialist',
  handoffDescription: 'Handles specialized queries',
  instructions: 'Provide expert assistance',
});

const triageAgent = new Agent({
  name: 'Triage',
  instructions: 'Route to appropriate specialist',
  handoffs: [specialistAgent],
});

const result = await run(triageAgent, 'I need specialist help');
// Automatically routes to specialistAgent
```

---

## TypeScript Migration Checklist

### Pre-Migration

- [ ] **Audit Current Implementation**: Document all agents and workflows
- [ ] **TypeScript Setup**: Ensure TypeScript 5.0+ installed
- [ ] **Dependencies Review**: Check for Swarm-specific packages
- [ ] **Test Coverage**: Ensure critical paths are tested
- [ ] **Type Definitions**: Plan type-safe interfaces

### Migration Steps

- [ ] **Install Agents SDK**: `npm install @openai/agents zod`
- [ ] **Update tsconfig.json**: Enable strict mode
- [ ] **Convert Imports**: Update from Swarm to Agents SDK
- [ ] **Type Tools**: Add Zod schemas for tool parameters
- [ ] **Add Guardrails**: Implement input/output validation
- [ ] **Enable Tracing**: Add observability
- [ ] **Session Management**: Implement type-safe sessions
- [ ] **Update Tests**: Migrate to Jest with async/await
- [ ] **Human-in-the-Loop**: Add approval workflows where needed

### Post-Migration

- [ ] **Type Coverage**: Verify 100% type safety
- [ ] **Performance Testing**: Check response times
- [ ] **Error Handling**: Validate error scenarios
- [ ] **Documentation**: Update API docs
- [ ] **Monitoring**: Set up production observability
- [ ] **Team Training**: Educate developers on new patterns

---

## Code Migration Examples

### Example 1: Customer Service System

**Before (Swarm-like):**

```typescript
// Limited type safety
interface Message {
  role: string;
  content: string;
}

function processRefund(bookingId: string): string {
  return `Refund processed: ${bookingId}`;
}

const refundAgent = {
  name: 'Refund',
  functions: [processRefund],
};

const client = { run: (config: any) => ({ messages: [] }) };
const result = client.run({
  agent: refundAgent,
  messages: [{ role: 'user', content: 'Refund my booking' }],
});
```

**After (Agents SDK 2025):**

```typescript
import { Agent, run, tool, Session } from '@openai/agents';
import { z } from 'zod';

// Type-safe tool definition
interface RefundResult {
  status: 'approved' | 'denied';
  bookingId: string;
  amount: number;
  confirmationNumber: string;
}

const processRefundTool = tool({
  name: 'process_refund',
  description: 'Process refund request',
  parameters: z.object({
    bookingId: z.string().regex(/^BK-\d{6}$/),
    reason: z.string().min(10),
  }),
  execute: async ({ bookingId, reason }): Promise<RefundResult> => {
    return {
      status: 'approved',
      bookingId,
      amount: 99.99,
      confirmationNumber: `REF-${Date.now()}`,
    };
  },
});

const refundAgent = new Agent({
  name: 'Refund Specialist',
  handoffDescription: 'Handles refund requests',
  instructions: 'Process refund requests with empathy and efficiency',
  tools: [processRefundTool],
});

const triageAgent = new Agent({
  name: 'Customer Service',
  instructions: 'Route customer requests appropriately',
  handoffs: [refundAgent],
});

// Type-safe execution with session
interface CustomerContext {
  customerId: string;
  tier: 'basic' | 'premium' | 'enterprise';
}

async function handleCustomerRequest(
  customerId: string,
  query: string
): Promise<string> {
  const session = new Session<CustomerContext>({
    id: customerId,
    metadata: {
      customerId,
      tier: 'premium',
    },
  });

  const result = await run(triageAgent, query, { session });
  return result.finalOutput;
}

// Usage
const response = await handleCustomerRequest(
  'cust_123',
  'I need a refund for booking BK-123456'
);
```

### Example 2: Research Workflow with Structured Output

**Before (Swarm-like):**

```typescript
function searchWeb(query: string): string {
  return `Results for ${query}`;
}

const researcher = {
  name: 'Researcher',
  functions: [searchWeb],
};

// Manual JSON parsing, no validation
const result = client.run({
  agent: researcher,
  messages: [{ role: 'user', content: 'Research AI' }],
});
```

**After (Agents SDK 2025):**

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

// Structured output schema
const ResearchOutputSchema = z.object({
  topic: z.string(),
  summary: z.string(),
  keyFindings: z.array(z.string()),
  sources: z.array(z.object({
    title: z.string(),
    url: z.string().url(),
    relevance: z.number().min(0).max(1),
  })),
  confidence: z.number().min(0).max(1),
});

type ResearchOutput = z.infer<typeof ResearchOutputSchema>;

// Web search tool with validation
const webSearchTool = tool({
  name: 'search_web',
  description: 'Search the web for information',
  parameters: z.object({
    query: z.string(),
    numResults: z.number().max(10).default(5),
  }),
  execute: async ({ query, numResults }) => {
    // Actual implementation
    return {
      results: [],
      query,
    };
  },
});

const researcher = new Agent({
  name: 'Research Agent',
  instructions: `Research topics thoroughly.
    Provide structured output with sources and confidence scores.`,
  tools: [webSearchTool],
  outputSchema: ResearchOutputSchema,
});

// Type-safe execution
async function conductResearch(topic: string): Promise<ResearchOutput> {
  const result = await run(researcher, `Research: ${topic}`);

  // Automatic validation
  const research = ResearchOutputSchema.parse(
    JSON.parse(result.finalOutput)
  );

  return research;
}

// Usage with full type safety
const research = await conductResearch('AI trends 2025');
console.log(`Topic: ${research.topic}`);
console.log(`Confidence: ${research.confidence}`);
research.keyFindings.forEach(finding => console.log(`- ${finding}`));
```

### Example 3: Multi-Agent Workflow with Tracing

**Before (Swarm-like):**

```typescript
// No tracing, manual orchestration
const result1 = client.run({ agent: agent1, messages: [...] });
const result2 = client.run({ agent: agent2, messages: [...] });
const result3 = client.run({ agent: agent3, messages: [...] });
```

**After (Agents SDK 2025):**

```typescript
import { Agent, run, trace } from '@openai/agents';

const dataCollector = new Agent({
  name: 'Data Collector',
  instructions: 'Collect required data',
});

const analyzer = new Agent({
  name: 'Analyzer',
  instructions: 'Analyze collected data',
});

const reporter = new Agent({
  name: 'Reporter',
  instructions: 'Generate report',
});

interface WorkflowResult {
  data: any;
  analysis: any;
  report: string;
}

async function executeAnalysisWorkflow(
  query: string
): Promise<WorkflowResult> {
  // Built-in tracing for entire workflow
  return trace('Analysis Workflow', async () => {
    // Step 1: Collect data
    const dataResult = await run(dataCollector, query);

    // Step 2: Analyze
    const analysisResult = await run(
      analyzer,
      `Analyze: ${dataResult.finalOutput}`
    );

    // Step 3: Generate report
    const reportResult = await run(
      reporter,
      `Create report: ${analysisResult.finalOutput}`
    );

    return {
      data: dataResult.finalOutput,
      analysis: analysisResult.finalOutput,
      report: reportResult.finalOutput,
    };
  }, {
    metadata: {
      workflow: 'analysis',
      query,
      timestamp: new Date().toISOString(),
    },
  });
}

// Execution with automatic tracing
const result = await executeAnalysisWorkflow('Market analysis for Q1 2025');
// View detailed trace at https://platform.openai.com/traces
```

---

## Type Safety Improvements

### Generic Agent Factory

```typescript
import { Agent } from '@openai/agents';
import { z } from 'zod';

interface TypedAgentConfig<TOutput extends z.ZodSchema> {
  name: string;
  instructions: string;
  outputSchema: TOutput;
  tools?: any[];
}

function createTypedAgent<TOutput extends z.ZodSchema>(
  config: TypedAgentConfig<TOutput>
): Agent {
  return new Agent({
    name: config.name,
    instructions: config.instructions,
    outputSchema: config.outputSchema,
    tools: config.tools,
  });
}

// Usage with type inference
const AnalysisSchema = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  score: z.number(),
});

const analysisAgent = createTypedAgent({
  name: 'Sentiment Analyzer',
  instructions: 'Analyze sentiment',
  outputSchema: AnalysisSchema,
});

// Type-safe result
type AnalysisResult = z.infer<typeof AnalysisSchema>;
```

### Type-Safe Tool Builder

```typescript
import { tool } from '@openai/agents';
import { z } from 'zod';

class ToolBuilder<
  TParams extends z.ZodSchema,
  TResult
> {
  constructor(
    private name: string,
    private description: string,
    private parameters: TParams,
    private handler: (params: z.infer<TParams>) => Promise<TResult>
  ) {}

  build() {
    return tool({
      name: this.name,
      description: this.description,
      parameters: this.parameters,
      execute: this.handler,
    });
  }
}

// Usage
const weatherTool = new ToolBuilder(
  'get_weather',
  'Get weather information',
  z.object({
    location: z.string(),
    units: z.enum(['celsius', 'fahrenheit']),
  }),
  async ({ location, units }) => ({
    location,
    temperature: 72,
    units,
  })
).build();
```

---

## Testing Strategy

### Jest Configuration

```typescript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
  ],
};
```

### Unit Tests

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

describe('Agents SDK Migration', () => {
  describe('Basic Agent', () => {
    it('should create and execute agent', async () => {
      const agent = new Agent({
        name: 'Test Agent',
        instructions: 'You are a test agent',
      });

      const result = await run(agent, 'What is 2+2?');

      expect(result.finalOutput).toBeDefined();
      expect(result.finalOutput).toContain('4');
    });
  });

  describe('Tools', () => {
    it('should execute tool with validation', async () => {
      const addTool = tool({
        name: 'add',
        description: 'Add two numbers',
        parameters: z.object({
          a: z.number(),
          b: z.number(),
        }),
        execute: async ({ a, b }) => a + b,
      });

      const agent = new Agent({
        name: 'Math Agent',
        tools: [addTool],
      });

      const result = await run(agent, 'Add 5 and 3');
      expect(result.finalOutput).toContain('8');
    });
  });

  describe('Handoffs', () => {
    it('should delegate to specialist agent', async () => {
      const specialist = new Agent({
        name: 'Specialist',
        handoffDescription: 'Handles complex queries',
        instructions: 'Provide expert assistance',
      });

      const triage = new Agent({
        name: 'Triage',
        instructions: 'Route to specialist',
        handoffs: [specialist],
      });

      const result = await run(triage, 'I need specialist help');

      expect(result.currentAgent?.name).toBe('Specialist');
    });
  });

  describe('Guardrails', () => {
    it('should validate input', async () => {
      const agent = new Agent({
        name: 'Safe Agent',
        instructions: 'Help users safely',
        inputGuardrails: [
          {
            name: 'content_filter',
            check: (input: string) => !input.includes('unsafe'),
          },
        ],
      });

      await expect(
        run(agent, 'This is unsafe content')
      ).rejects.toThrow();
    });
  });
});
```

---

## Production Deployment

### Docker Configuration

```dockerfile
# Multi-stage build for TypeScript
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

ENV NODE_ENV=production

COPY package*.json ./
RUN npm ci --only=production

COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["node", "dist/server.js"]
```

### Express.js Integration

```typescript
import express from 'express';
import { Agent, run } from '@openai/agents';

const app = express();
app.use(express.json());

const agent = new Agent({
  name: 'API Agent',
  instructions: 'Handle API requests',
});

app.post('/api/agents/execute', async (req, res) => {
  try {
    const { input } = req.body;

    const result = await run(agent, input);

    res.json({
      success: true,
      output: result.finalOutput,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

app.listen(3000, () => {
  console.log('Agent API running on port 3000');
});
```

---

## Conclusion

The OpenAI Agents SDK for TypeScript provides:

✅ **Production-ready** stability and TypeScript-first design
✅ **2025 Features**: Human-in-the-loop, tracing, guardrails, MCP
✅ **Full Type Safety**: Zod schemas, type inference, IDE support
✅ **Provider Flexibility**: 100+ LLM providers
✅ **Enterprise Features**: Observability, security, scalability

**Migration Recommendation**: Migrate all Swarm implementations to Agents SDK for production TypeScript applications.

---

**Last Updated:** January 2025
**Guide Version:** 1.0
**Agents SDK Version:** Latest
**TypeScript Version:** 5.0+

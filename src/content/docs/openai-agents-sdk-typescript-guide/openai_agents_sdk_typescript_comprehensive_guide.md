---
title: "OpenAI Agents SDK with TypeScript: Comprehensive Technical Guide"
description: "Agents, runner, handoffs, guardrails, sessions, tools and structured output with the OpenAI Agents SDK for TypeScript."
framework: "openai-agents-sdk"
language: "typescript"
---

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%). Node.js 22+ (also Deno, Bun and Cloudflare Workers with `nodejs_compat`), Zod 4, `openai` v7.

---

## Table of Contents

1. [Core Fundamentals](#core-fundamentals)
2. [Simple Agents](#simple-agents)
3. [Multi-Agent Systems](#multi-agent-systems)
4. [Tools Integration](#tools-integration)
5. [Structured Output](#structured-output)
6. [Summary](#summary)

**Continue elsewhere in this guide:** [Features and updates](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_2025_features/) for MCP, realtime and the Responses API · [Middleware & guardrails](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/) · [Streaming server (Express)](/openai-agents-sdk-typescript-guide/openai_agents_streaming_server_express/) · [Production guide](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_production_guide/) for deployment, tracing and testing · [Recipes](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_recipes/).

---

## Core Fundamentals

### Installation and Setup

The OpenAI Agents SDK provides a lightweight framework for building agentic AI applications with minimal abstractions. `@openai/agents` bundles the core, the OpenAI provider and realtime support; Zod is a peer dependency:

```bash
npm install @openai/agents@%%version=@openai/agents%% zod
# Optional: non-OpenAI models via the Vercel AI SDK, Twilio/Cloudflare realtime transports, hosted sandboxes
npm install @openai/agents-extensions@%%version=@openai/agents%% @ai-sdk/anthropic
```

> **Note:** The AI SDK adapter lives in `@openai/agents-extensions` and is imported from the `ai-sdk` subpath: `import { aisdk } from '@openai/agents-extensions/ai-sdk'`. The `@openai/agents*` packages are versioned together, so keep them on the same version.

```bash
# Additional development dependencies
npm install dotenv
npm install --save-dev typescript @types/node
```

Create a `.env` file to store your API credentials:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### TypeScript Configuration

The SDK ships ES modules; a modern `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

Set `"type": "module"` in `package.json` so top-level `await` works in the examples below.

### Design Philosophy

The OpenAI Agents SDK embodies a philosophy of **lightweight primitives over heavy abstractions**. Rather than providing a monolithic framework, it offers core building blocks:

- **Agent**: An LLM-equipped entity with instructions and tools
- **Runner**: Executes agents and manages their lifecycle
- **Handoff**: Enables task delegation between agents
- **Guardrail**: Validates inputs and outputs
- **Session**: Maintains conversation history and state
- **Tool**: Extends agent capabilities with functions or APIs

This minimalist approach provides flexibility and clarity whilst avoiding unnecessary complexity.

### Evolution from Swarm to Agents SDK

The Agents SDK represents the production-ready evolution of OpenAI's experimental Swarm project. Key improvements include:

- **Enhanced Stability**: Production-grade reliability
- **Better Type Safety**: Improved TypeScript support
- **Refined API**: Cleaner, more intuitive interfaces
- **Performance Optimisation**: Optimised for real-world scenarios
- **Built-in Tracing**: Visibility into agent execution
- **Model Flexibility**: Support for various LLM providers

### Core Primitives Deep Dive

#### Agent

An `Agent` is configuration: a name, instructions, a model, and optionally tools, handoffs, guardrails and an output type. Sampling parameters go in `modelSettings`:

```typescript
import { Agent } from '@openai/agents';

const researchAgent = new Agent({
  name: 'Research Assistant',
  instructions: `You are a thorough research assistant.
    - Provide detailed, well-sourced information
    - Always cite your sources
    - Acknowledge uncertainty when appropriate
    - Break down complex topics into understandable parts`,
  model: '%%model=openai%%',
  modelSettings: { maxTokens: 2048 },
});
```

If `model` is omitted the SDK uses its default (`gpt-5.6-luna` at the time of writing, since 0.15.0) or the `OPENAI_DEFAULT_MODEL` environment variable.

#### Runner

`run()` executes an agent loop — model calls, tool calls, handoffs and guardrails — until a final output is produced. A `Runner` instance holds defaults shared by many runs:

```typescript
import { Agent, Runner, run, MaxTurnsExceededError } from '@openai/agents';

// One-off run with the default runner
const result = await run(researchAgent, 'What are the latest AI trends?', {
  maxTurns: 10,            // null disables the limit
  // session, context, signal, previousResponseId, conversationId, errorHandlers ...
});
console.log(result.finalOutput);
console.log(result.lastAgent?.name);          // agent that produced the output
console.log(result.runContext.usage);         // { requests, inputTokens, outputTokens, totalTokens, ... }

// Shared configuration
const runner = new Runner({
  model: '%%model=openai-small%%',          // overrides every agent's model
  workflowName: 'Research',
  traceIncludeSensitiveData: false,
});

try {
  const r = await runner.run(researchAgent, 'Summarise the field in three bullets.');
  console.log(r.finalOutput);
} catch (error) {
  if (error instanceof MaxTurnsExceededError) console.error('Agent looped');
  else throw error;
}
```

#### Handoff

Handoffs let one agent delegate the conversation to another. List agents directly, or wrap them with `handoff()` to override the tool name, add input data or filter history:

```typescript
import { Agent, handoff } from '@openai/agents';
import { z } from 'zod';

const billingAgent = new Agent({
  name: 'Billing Specialist',
  handoffDescription: 'Billing, invoices and payments',
  instructions: 'Handle all billing-related inquiries with accuracy and professionalism.',
});

const technicalAgent = new Agent({
  name: 'Technical Support',
  handoffDescription: 'Bugs, errors and crashes',
  instructions: 'Resolve technical issues with clear, step-by-step solutions.',
});

const customerServiceAgent = Agent.create({
  name: 'Customer Service Triage',
  instructions: 'Route customers to the appropriate specialist.',
  handoffs: [
    billingAgent,
    handoff(technicalAgent, {
      toolNameOverride: 'transfer_to_technical',
      toolDescriptionOverride: 'Transfer customer to technical support',
      inputType: z.object({ summary: z.string() }),
      onHandoff: (_ctx, input) => console.log('Escalation summary:', input?.summary),
    }),
  ],
});
```

`Agent.create()` (rather than `new Agent()`) keeps the output type correct when handoff targets have different output types.

#### Guardrail

Guardrails are objects with a `name` and an `execute` function that returns `{ tripwireTriggered, outputInfo }`. Input guardrails run on the user input of the first agent; output guardrails run on the final output:

```typescript
import {
  Agent,
  run,
  InputGuardrail,
  OutputGuardrail,
  InputGuardrailTripwireTriggered,
  OutputGuardrailTripwireTriggered,
} from '@openai/agents';
import { z } from 'zod';

const noEmailsGuardrail: InputGuardrail = {
  name: 'no_email_addresses',
  execute: async ({ input }) => {
    const text = typeof input === 'string' ? input : JSON.stringify(input);
    const found = /[\w.+-]+@[\w-]+\.[\w.]+/.test(text);
    return { tripwireTriggered: found, outputInfo: { found } };
  },
};

const responseSchema = z.object({
  status: z.enum(['success', 'error', 'pending']),
  message: z.string(),
});

const statusGuardrail: OutputGuardrail<typeof responseSchema> = {
  name: 'no_error_status',
  execute: async ({ agentOutput }) => ({
    tripwireTriggered: agentOutput.status === 'error',
    outputInfo: { status: agentOutput.status },
  }),
};

const agent = new Agent({
  name: 'Support',
  instructions: 'Answer support questions.',
  outputType: responseSchema,
  inputGuardrails: [noEmailsGuardrail],
  outputGuardrails: [statusGuardrail],
});

try {
  await run(agent, 'My email is jane@example.com, please reset my password');
} catch (e) {
  if (e instanceof InputGuardrailTripwireTriggered) console.log('Input blocked');
  else if (e instanceof OutputGuardrailTripwireTriggered) console.log('Output blocked');
  else throw e;
}
```

#### Session

Pass a session to `run()` and the SDK loads prior items before the turn and stores new ones afterwards. `MemorySession` is in-process; `OpenAIConversationsSession` stores history server-side; implement the `Session` interface for your own store:

```typescript
import { Agent, MemorySession, run } from '@openai/agents';

const agent = new Agent({ name: 'Tutor', instructions: 'Explain concepts simply.' });
const session = new MemorySession();

const result1 = await run(agent, 'What is machine learning?', { session });
console.log('Turn 1:', result1.finalOutput);

// Second turn - maintains context
const result2 = await run(agent, 'Explain neural networks in that context.', { session });
console.log('Turn 2:', result2.finalOutput);

const history = await session.getItems();
console.log('Stored items:', history.length);
```

#### Tool

`tool()` builds a function tool from a Zod schema; the parameters are validated before `execute` runs:

```typescript
import { tool } from '@openai/agents';
import { z } from 'zod';

// Type-safe tool definition
const calculateTool = tool({
  name: 'calculate',
  description: 'Performs mathematical operations',
  parameters: z.object({
    operation: z.enum(['add', 'subtract', 'multiply', 'divide']),
    a: z.number(),
    b: z.number(),
  }),
  execute: async ({ operation, a, b }) => {
    switch (operation) {
      case 'add': return a + b;
      case 'subtract': return a - b;
      case 'multiply': return a * b;
      case 'divide': return b !== 0 ? a / b : 'Error: Division by zero';
      default: return 'Unknown operation';
    }
  },
});

// Async tool with external API call
const weatherTool = tool({
  name: 'get_weather',
  description: 'Retrieves weather information for a location',
  parameters: z.object({
    location: z.string().describe('City and country or coordinates'),
    units: z.enum(['celsius', 'fahrenheit']).default('celsius'),
  }),
  execute: async ({ location, units }) => {
    // Mock API call
    return `Weather in ${location}: 22°C, Partly cloudy`;
  },
});

// Tool that needs human approval before it runs
const databaseQueryTool = tool({
  name: 'query_database',
  description: 'Executes safe database queries',
  parameters: z.object({
    query: z.string().describe('SQL query (SELECT only)'),
  }),
  needsApproval: async (_ctx, { query }) => !query.toUpperCase().startsWith('SELECT'),
  execute: async ({ query }) => {
    // Mock database execution
    return { success: true, rows: [] };
  },
  // Thrown errors are reported back to the model by default; customise with errorFunction
});
```

### Model Configuration

OpenAI models are referenced by name. Other providers go through the AI SDK adapter in `@openai/agents-extensions`:

```typescript
import { Agent } from '@openai/agents';
import { aisdk } from '@openai/agents-extensions/ai-sdk';
import { anthropic } from '@ai-sdk/anthropic';

// OpenAI (Responses API by default)
const openaiAgent = new Agent({
  name: 'OpenAI Assistant',
  instructions: 'Be concise.',
  model: '%%model=openai%%',
  modelSettings: { maxTokens: 2048, reasoning: { effort: 'low' } },
});

// Anthropic via the Vercel AI SDK
const claudeAgent = new Agent({
  name: 'Claude Assistant',
  instructions: 'Be concise.',
  model: aisdk(anthropic('%%model=anthropic%%')),
  modelSettings: { temperature: 0.7 },
});
```

To use Chat Completions instead of Responses for OpenAI models, call `setOpenAIAPI('chat_completions')` once at startup.

### Environment Setup Best Practices

```typescript
import dotenv from 'dotenv';

dotenv.config();

interface EnvironmentConfig {
  openaiApiKey: string;
  anthropicApiKey?: string;
  environment: 'development' | 'staging' | 'production';
  debug: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
}

function loadEnvironmentConfig(): EnvironmentConfig {
  const config: EnvironmentConfig = {
    openaiApiKey: process.env.OPENAI_API_KEY || '',
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    environment: (process.env.NODE_ENV as any) || 'development',
    debug: process.env.DEBUG === 'true',
    logLevel: (process.env.LOG_LEVEL as any) || 'info',
  };

  if (!config.openaiApiKey) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }

  return config;
}

const envConfig = loadEnvironmentConfig();
```

### TypeScript Configuration Patterns

```typescript
// Generic agent factory with type safety
type AgentCallback<T> = (result: T) => Promise<void> | void;

interface TypedAgentConfig<T> {
  name: string;
  instructions: string;
  outputSchema?: z.ZodSchema<T>;
  onComplete?: AgentCallback<T>;
}

async function createTypedAgent<T>(
  config: TypedAgentConfig<T>
): Promise<Agent> {
  const agent = new Agent({
    name: config.name,
    instructions: config.instructions,
  });

  return agent;
}

// Usage with type inference
interface ResearchOutput {
  title: string;
  findings: string[];
  sources: string[];
}

const typedAgent = await createTypedAgent<ResearchOutput>({
  name: 'Research Agent',
  instructions: 'Conduct thorough research.',
  onComplete: async (result) => {
    console.log(`Research: ${result.title}`);
    result.findings.forEach(f => console.log(`- ${f}`));
  },
});
```

---

## Simple Agents

### Creating Basic Agents

The simplest agent requires just a name and instructions:

```typescript
import { Agent, run } from '@openai/agents';

async function createBasicAgent(): Promise<void> {
  const basicAgent = new Agent({
    name: 'General Assistant',
    instructions: 'You are a helpful, friendly assistant. Provide clear, concise answers.',
  });

  const result = await run(basicAgent, 'What is the capital of France?');
  console.log(result.finalOutput);
  // Output: "The capital of France is Paris."
}

createBasicAgent().catch(console.error);
```

### Agent Name and Instructions Parameters

Agent identity and behaviour are defined through these core parameters:

```typescript
interface AgentIdentity {
  name: string;
  instructions: string;
  personalityTraits?: string[];
  expertise?: string[];
}

// Specialized agent example
const expertiseAgent: AgentIdentity = {
  name: 'Data Science Expert',
  instructions: `You are a data science expert with 10+ years of experience.
    Specialise in:
    - Machine learning algorithms
    - Statistical analysis
    - Data visualisation
    - Python and R programming
    
    When answering questions:
    1. Explain concepts from first principles
    2. Provide practical examples
    3. Discuss trade-offs and alternatives
    4. Suggest relevant tools and libraries`,
  personalityTraits: ['analytical', 'patient', 'practical'],
  expertise: ['ML', 'Statistics', 'Python', 'SQL'],
};

const dataAgent = new Agent({
  name: expertiseAgent.name,
  instructions: expertiseAgent.instructions,
});
```

### Type-Safe Agent Configuration

Use TypeScript interfaces to ensure configuration completeness:

```typescript
interface StrictAgentConfig {
  name: string;
  instructions: string;
  model?: '%%model=openai%%' | '%%model=openai-small%%';
  temperature?: number & { __brand: 'temperature' };
  maxTokens?: number;
  tools?: Tool[];
}

// Type guard for valid temperature
function createTemperature(value: number): number & { __brand: 'temperature' } {
  if (value < 0 || value > 2) {
    throw new Error('Temperature must be between 0 and 2');
  }
  return value as number & { __brand: 'temperature' };
}

const strictConfig: StrictAgentConfig = {
  name: 'Strict Agent',
  instructions: 'Follow instructions precisely.',
  model: '%%model=openai-small%%',
  temperature: createTemperature(0.7),
  maxTokens: 1024,
};

// Sampling options belong in modelSettings on the real Agent
const { temperature, maxTokens, ...rest } = strictConfig;
const strictAgent = new Agent({ ...rest, modelSettings: { temperature, maxTokens } });
```

### run() and Runner.run() Patterns

Agents have no `run()` method of their own; pass them to `run()` or to a `Runner`:

```typescript
// Pattern 1: Shared Runner with run-wide defaults
async function directRun(): Promise<void> {
  const agent = new Agent({
    name: 'Direct Agent',
    instructions: 'Answer briefly.',
  });

  const runner = new Runner({ workflowName: 'direct' });
  const result = await runner.run(agent, 'What is AI?');
  console.log(result.finalOutput);
}

// Pattern 2: Runner with multiple agents
async function multiAgentRun(): Promise<void> {
  const agent1 = new Agent({
    name: 'Agent 1',
    instructions: 'Explain concepts.',
  });

  const agent2 = new Agent({
    name: 'Agent 2',
    instructions: 'Provide examples.',
  });

  const result1 = await run(agent1, 'What is blockchain?');
  const result2 = await run(agent2, result1.finalOutput ?? '');
  
  console.log('Final result:', result2.finalOutput);
}

// Pattern 3: Batch execution
async function batchExecution(): Promise<void> {
  const agent = new Agent({
    name: 'Batch Agent',
    instructions: 'Process tasks.',
  });

  const queries = [
    'What is Python?',
    'What is JavaScript?',
    'What is Go?',
  ];

  const results = await Promise.all(
    queries.map(query => run(agent, query))
  );

  results.forEach((result, idx) => {
    console.log(`Query ${idx + 1}: ${result.finalOutput}`);
  });
}
```

### Dynamic System Prompts

Modify behaviour based on context:

```typescript
interface DynamicPromptContext {
  userRole?: string;
  expertise?: 'beginner' | 'intermediate' | 'expert';
  language?: string;
  formality?: 'casual' | 'formal' | 'technical';
}

function generateDynamicInstructions(context: DynamicPromptContext): string {
  let instructions = 'You are a helpful assistant.';

  if (context.userRole) {
    instructions += ` You are speaking to a ${context.userRole}.`;
  }

  switch (context.expertise) {
    case 'beginner':
      instructions += ' Explain concepts simply, avoiding jargon.';
      break;
    case 'intermediate':
      instructions += ' Provide balanced explanations with some technical detail.';
      break;
    case 'expert':
      instructions += ' Assume advanced knowledge and provide technical depth.';
      break;
  }

  if (context.formality === 'casual') {
    instructions += ' Use conversational language.';
  } else if (context.formality === 'technical') {
    instructions += ' Use precise, technical language.';
  }

  return instructions;
}

async function adaptiveAgent(context: DynamicPromptContext): Promise<void> {
  const agent = new Agent({
    name: 'Adaptive Agent',
    instructions: generateDynamicInstructions(context),
  });

  const result = await run(agent, 'Explain machine learning.');
  console.log(result.finalOutput);
}

// Usage
await adaptiveAgent({ expertise: 'beginner', formality: 'casual' });
```

### Single-Turn Conversations

Handle independent queries:

```typescript
async function singleTurnConversation(): Promise<void> {
  const agent = new Agent({
    name: 'Question Answerer',
    instructions: 'Provide accurate, concise answers to questions.',
  });

  // Each query is independent
  const queries = [
    'What year was JavaScript created?',
    'Who invented the web?',
    'What is the fastest land animal?',
  ];

  for (const query of queries) {
    const result = await run(agent, query);
    console.log(`Q: ${query}`);
    console.log(`A: ${result.finalOutput}\n`);
  }
}

singleTurnConversation().catch(console.error);
```

### Multi-Turn Dialogues

Maintain context across turns:

```typescript
import { Agent, MemorySession, run } from '@openai/agents';

interface ConversationTurn {
  userInput: string;
  agentResponse: string;
  timestamp: Date;
}

async function multiTurnDialogue(): Promise<ConversationTurn[]> {
  const agent = new Agent({
    name: 'Conversational Agent',
    instructions: `You are a helpful assistant that remembers context.
      Maintain coherent conversation by referencing previous messages.
      Build on prior knowledge to deepen understanding.`,
  });

  const session = new MemorySession({ sessionId: 'conv-001' });
  const conversation: ConversationTurn[] = [];

  const exchanges = [
    'My name is Alex and I love Python programming.',
    'What should I learn next after Python?',
    'How can I apply machine learning with what I know?',
    'Show me a simple example.',
  ];

  for (const userInput of exchanges) {
    const result = await run(agent, userInput, { session });
    
    conversation.push({
      userInput,
      agentResponse: result.finalOutput ?? '',
      timestamp: new Date(),
    });

    console.log(`User: ${userInput}`);
    console.log(`Agent: ${result.finalOutput}\n`);
  }

  return conversation;
}

await multiTurnDialogue();
```

### Streaming Outputs

Handle large responses in real-time:

```typescript
import { run } from '@openai/agents';

async function streamingOutput(): Promise<void> {
  const agent = new Agent({
    name: 'Streaming Agent',
    instructions: 'Provide detailed, thorough responses.',
  });

  const input = 'Explain the history of artificial intelligence in detail.';
  
  try {
    const result = await run(agent, input, { stream: true });

    // toTextStream() yields only the text deltas
    for await (const chunk of result.toTextStream()) {
      process.stdout.write(chunk);
    }
    await result.completed;

    console.log('\n--- Streaming complete ---');
  } catch (error) {
    console.error('Streaming error:', error);
  }
}

// Real-time streaming to WebSocket
async function streamToWebSocket(
  agent: Agent,
  input: string,
  ws: WebSocket
): Promise<void> {
  const result = await run(agent, input, { stream: true });

  // Iterating the result itself yields every event (raw model events,
  // run items, agent updates); pick out text deltas
  for await (const event of result) {
    if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') {
      ws.send(JSON.stringify({ type: 'stream', content: event.data.delta }));
    }
  }
  await result.completed;

  ws.send(JSON.stringify({ type: 'complete', finalOutput: result.finalOutput }));
}
```

### Error Handling with TypeScript

Comprehensive error management:

```typescript
class AgentError extends Error {
  code: 'VALIDATION_ERROR' | 'RATE_LIMIT' | 'API_ERROR' | 'TIMEOUT' | 'UNKNOWN' = 'UNKNOWN';
  details?: Record<string, any>;
}

class AgentErrorHandler {
  private static readonly MAX_RETRIES = 3;
  private static readonly RETRY_DELAY = 1000;

  static async executeWithRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number = this.MAX_RETRIES
  ): Promise<T> {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        if (attempt === maxRetries) throw error;

        const delay = this.RETRY_DELAY * Math.pow(2, attempt - 1);
        console.log(`Retry attempt ${attempt} after ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw new Error('Max retries exceeded');
  }

  static createAgentError(
    message: string,
    code: AgentError['code'],
    details?: Record<string, any>
  ): AgentError {
    const error = new AgentError(message);
    error.code = code;
    error.details = details;
    return error;
  }

  static handleError(error: unknown): void {
    if (error instanceof AgentError) {
      console.error(`[${error.code}] ${error.message}`, error.details);
    } else if (error instanceof Error) {
      console.error(`Unknown error: ${error.message}`);
    } else {
      console.error('Unexpected error:', error);
    }
  }
}

// Usage
async function robustAgentExecution(): Promise<void> {
  const agent = new Agent({
    name: 'Robust Agent',
    instructions: 'Handle errors gracefully.',
  });

  try {
    const result = await AgentErrorHandler.executeWithRetry(
      () => run(agent, 'What is TypeScript?')
    );
    console.log(result.finalOutput);
  } catch (error) {
    AgentErrorHandler.handleError(error);
  }
}
```

### Response Type Definitions

Define and enforce response schemas:

```typescript
import { z } from 'zod';

// Response schema definitions
const questionAnswerSchema = z.object({
  question: z.string(),
  answer: z.string(),
  confidence: z.number().min(0).max(1),
  sources: z.string().array().optional(),
});

type QuestionAnswer = z.infer<typeof questionAnswerSchema>;

const analysisSchema = z.object({
  summary: z.string(),
  keyPoints: z.string().array(),
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  actionItems: z.string().array().optional(),
});

type Analysis = z.infer<typeof analysisSchema>;

// Type-safe response parsing
async function parseAgentResponse<T extends z.ZodSchema>(
  agent: Agent,
  input: string,
  schema: T
): Promise<z.infer<T> | null> {
  try {
    const result = await run(agent, input);

    // In production, you'd parse JSON from the output
    const parsed = JSON.parse(result.finalOutput);
    return schema.parse(parsed);
  } catch (error) {
    console.error('Response parsing failed:', error);
    return null;
  }
}

// Usage
async function typedResponse(): Promise<void> {
  const agent = new Agent({
    name: 'Analysis Agent',
    instructions: 'Respond in JSON format.',
  });

  const response = await parseAgentResponse(
    agent,
    'Analyse this text...',
    analysisSchema
  );

  if (response) {
    console.log('Summary:', response.summary);
    console.log('Sentiment:', response.sentiment);
  }
}
```

---

## Multi-Agent Systems

### Handoff Mechanisms Between Agents

Seamless delegation patterns:

```typescript
import { Agent, Handoff } from '@openai/agents';

interface MultiAgentSystem {
  agents: Record<string, Agent>;
  handoffs: Handoff[];
  router: (input: string) => Agent;
}

// Define specialist agents
const weatherAgent = new Agent({
  name: 'Weather Expert',
  instructions: 'Provide detailed weather information and forecasts.',
  handoffDescription: 'Transfer for weather-related queries',
});

const newsAgent = new Agent({
  name: 'News Reporter',
  instructions: 'Provide current news and information.',
  handoffDescription: 'Transfer for news-related queries',
});

const financeAgent = new Agent({
  name: 'Finance Advisor',
  instructions: 'Provide financial information and advice.',
  handoffDescription: 'Transfer for financial queries',
});

// Create router
const routeToSpecialist = (input: string): Agent => {
  const lowerInput = input.toLowerCase();

  if (/weather|rain|sunny|forecast|temperature/.test(lowerInput)) {
    return weatherAgent;
  } else if (/news|latest|breaking|article|report/.test(lowerInput)) {
    return newsAgent;
  } else if (/stocks|crypto|investment|portfolio|dividend/.test(lowerInput)) {
    return financeAgent;
  }

  return triagedAgent; // Default
};

// Create triage agent with handoffs
const triagedAgent = Agent.create({
  name: 'Universal Assistant',
  instructions: 'Route queries to the most appropriate specialist.',
  handoffs: [weatherAgent, newsAgent, financeAgent],
});

// Execute with automatic routing
async function executeWithHandoff(query: string): Promise<string> {
  const result = await run(triagedAgent, query);
  return result.finalOutput;
}
```

### Type-Safe Agent Delegation Patterns

```typescript
interface DelegationResult<T> {
  delegatedAgent: Agent;
  result: T;
  executionTime: number;
}

type AgentHandler<T> = (input: string) => Promise<T>;

async function delegateWithType<T>(
  input: string,
  handlers: Record<string, AgentHandler<T>>,
  selector: (input: string) => keyof typeof handlers
): Promise<DelegationResult<T>> {
  const startTime = performance.now();
  const handlerKey = selector(input);
  const handler = handlers[handlerKey as string];

  if (!handler) {
    throw new Error(`No handler found for ${String(handlerKey)}`);
  }

  const result = await handler(input);

  return {
    delegatedAgent: { name: String(handlerKey) } as Agent,
    result,
    executionTime: performance.now() - startTime,
  };
}

// Usage
const handlers = {
  technical: async (input: string) => {
    const agent = new Agent({
      name: 'Technical',
      instructions: 'Solve technical problems.',
    });
    const result = await run(agent, input);
    return result.finalOutput;
  },
  support: async (input: string) => {
    const agent = new Agent({
      name: 'Support',
      instructions: 'Provide customer support.',
    });
    const result = await run(agent, input);
    return result.finalOutput;
  },
};

const delegation = await delegateWithType(
  'System is crashing',
  handlers,
  (input) => /crash|error|bug/.test(input) ? 'technical' : 'support'
);
```

### Message Filtering During Handoffs

Control what history the receiving agent sees with `inputFilter` on `handoff()`. The filter receives the history and new items and returns a trimmed copy; `removeAllTools` from `@openai/agents-core/extensions` strips tool calls and results:

```typescript
import { Agent, handoff, HandoffInputData } from '@openai/agents';
import { removeAllTools } from '@openai/agents-core/extensions';

const MAX_HISTORY = 10;

// Keep only the last few items and drop tool chatter
function trimHistory(data: HandoffInputData): HandoffInputData {
  const withoutTools = removeAllTools(data);
  const history = Array.isArray(withoutTools.inputHistory)
    ? withoutTools.inputHistory.slice(-MAX_HISTORY)
    : withoutTools.inputHistory;
  return { ...withoutTools, inputHistory: history };
}

const specialist = new Agent({
  name: 'Specialist',
  instructions: 'Answer using only the recent conversation.',
});

const frontDesk = new Agent({
  name: 'Front Desk',
  instructions: 'Hand complex questions to the specialist.',
  handoffs: [handoff(specialist, { inputFilter: trimHistory })],
});
```

### Routing Logic and Conditional Handoffs

Sophisticated routing strategies:

```typescript
interface RoutingRule<T> {
  name: string;
  condition: (input: string) => boolean;
  handler: (input: string) => Promise<T>;
  priority: number; // Higher priority evaluated first
}

class SmartRouter<T> {
  private rules: RoutingRule<T>[] = [];

  addRule(rule: RoutingRule<T>): void {
    this.rules.push(rule);
    this.rules.sort((a, b) => b.priority - a.priority);
  }

  async route(input: string): Promise<T> {
    for (const rule of this.rules) {
      if (rule.condition(input)) {
        console.log(`Routing to: ${rule.name}`);
        return rule.handler(input);
      }
    }

    throw new Error('No routing rule matched');
  }
}

// Setup router
const router = new SmartRouter<string>();

router.addRule({
  name: 'Emergency Handler',
  priority: 1000,
  condition: (input) => /urgent|emergency|critical/.test(input),
  handler: async (input) => {
    const emergency = new Agent({
      name: 'Emergency',
      instructions: 'Handle urgent issues immediately.',
    });
    const result = await run(emergency, input);
    return result.finalOutput;
  },
});

router.addRule({
  name: 'Technical Support',
  priority: 10,
  condition: (input) => /bug|error|crash|technical/.test(input),
  handler: async (input) => {
    const technical = new Agent({
      name: 'Technical',
      instructions: 'Resolve technical issues.',
    });
    const result = await run(technical, input);
    return result.finalOutput;
  },
});

router.addRule({
  name: 'General Support',
  priority: 1,
  condition: () => true, // Always matches
  handler: async (input) => {
    const general = new Agent({
      name: 'General',
      instructions: 'Provide general assistance.',
    });
    const result = await run(general, input);
    return result.finalOutput;
  },
});

// Usage
const response = await router.route('There is a critical bug in production!');
```

### Parallel Agent Execution

Execute multiple agents concurrently:

```typescript
interface ParallelExecutionConfig {
  agents: Agent[];
  input: string;
  timeout?: number;
  failFast?: boolean;
}

async function executeParallel(
  config: ParallelExecutionConfig
): Promise<any[]> {
  const promises = config.agents.map(agent =>
    Promise.race([
      run(agent, config.input),
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error('Timeout')),
          config.timeout || 30000
        )
      ),
    ])
  );

  if (config.failFast) {
    return Promise.all(promises);
  }

  return Promise.allSettled(promises).then(results =>
    results.map(r => r.status === 'fulfilled' ? r.value : r.reason)
  );
}

// Usage
const agents = [
  new Agent({ name: 'Agent A', instructions: 'Analyse data.' }),
  new Agent({ name: 'Agent B', instructions: 'Validate results.' }),
  new Agent({ name: 'Agent C', instructions: 'Summarise findings.' }),
];

const results = await executeParallel({
  agents,
  input: 'Analyse this dataset',
  timeout: 20000,
  failFast: false,
});

results.forEach((result, idx) => {
  console.log(`Agent ${idx}: ${result.finalOutput || 'Error'}`);
});
```

### Agent as Tools Pattern

Use agents as callable tools:

```typescript
const summariserAgent = new Agent({
  name: 'Summariser',
  instructions: 'Create concise, accurate summaries.',
});

const translatorAgent = new Agent({
  name: 'Translator',
  instructions: 'Translate text between languages.',
});

// Create tools from agents
const summariseTool = tool({
  name: 'summarise_text',
  description: 'Create a summary of the provided text',
  parameters: z.object({
    text: z.string(),
    maxLength: z.number().optional(),
  }),
  execute: async ({ text, maxLength }) => {
    const prompt = maxLength 
      ? `Summarise this in ${maxLength} words:\n${text}`
      : `Summarise this:\n${text}`;
    
    const result = await run(summariserAgent, prompt);
    return result.finalOutput;
  },
});

const translateTool = tool({
  name: 'translate_text',
  description: 'Translate text to another language',
  parameters: z.object({
    text: z.string(),
    targetLanguage: z.string(),
  }),
  execute: async ({ text, targetLanguage }) => {
    const result = await run(
      translatorAgent,
      `Translate to ${targetLanguage}:\n${text}`
    );
    return result.finalOutput;
  },
});

// Use in main agent
const mainAgent = new Agent({
  name: 'Content Processor',
  instructions: 'Process content using available tools.',
  tools: [summariseTool, translateTool],
});
```

### Multi-Agent Workflows

Complex orchestrated workflows:

```typescript
interface WorkflowStep {
  name: string;
  agent: Agent;
  input: string | ((previousResult: any) => string);
  condition?: (result: any) => boolean;
}

class Workflow {
  private steps: WorkflowStep[] = [];
  private results: Record<string, any> = {};

  addStep(step: WorkflowStep): this {
    this.steps.push(step);
    return this;
  }

  async execute(): Promise<Record<string, any>> {
    for (const step of this.steps) {
      if (step.condition && !step.condition(this.results)) {
        console.log(`Skipping ${step.name} - condition not met`);
        continue;
      }

      const input = typeof step.input === 'function'
        ? step.input(this.results)
        : step.input;

      console.log(`Executing: ${step.name}`);
      const result = await run(step.agent, input);
      this.results[step.name] = result.finalOutput;
    }

    return this.results;
  }
}

// Define workflow
const workflow = new Workflow()
  .addStep({
    name: 'Research',
    agent: new Agent({
      name: 'Researcher',
      instructions: 'Conduct research.',
    }),
    input: 'Research artificial intelligence trends',
  })
  .addStep({
    name: 'Analyse',
    agent: new Agent({
      name: 'Analyst',
      instructions: 'Analyse provided information.',
    }),
    input: (prev) => `Analyse these findings: ${prev.Research}`,
    condition: (prev) => prev.Research?.length > 0,
  })
  .addStep({
    name: 'Summarise',
    agent: new Agent({
      name: 'Summariser',
      instructions: 'Create summaries.',
    }),
    input: (prev) => `Summarise: ${prev.Analyse}`,
  });

const workflowResults = await workflow.execute();
console.log('Workflow completed:', workflowResults);
```

### Customer Service Examples

Real-world customer service implementation:

```typescript
const billingAgent = new Agent({
  name: 'Billing Support',
  instructions: `Handle billing inquiries with accuracy and empathy.
    - Verify customer identity
    - Explain charges clearly
    - Process refunds when appropriate
    - Escalate complex issues`,
});

const technicalAgent = new Agent({
  name: 'Technical Support',
  instructions: `Resolve technical issues systematically.
    - Gather system information
    - Follow troubleshooting steps
    - Provide clear guidance
    - Escalate if needed`,
});

const triageAgent = Agent.create({
  name: 'Support Triage',
  instructions: `Route customers to appropriate department.
    - Identify issue type
    - Ensure customer satisfaction
    - Maintain professional tone`,
  handoffs: [billingAgent, technicalAgent],
});

async function customerServiceWorkflow(
  customerMessage: string
): Promise<void> {
  console.log(`Customer: ${customerMessage}`);
  
  const result = await run(triageAgent, customerMessage);
  console.log(`Support: ${result.finalOutput}`);

  if (result.lastAgent?.name === 'Billing Support') {
    console.log('Routed to Billing Department');
  } else if (result.lastAgent?.name === 'Technical Support') {
    console.log('Routed to Technical Department');
  }
}

// Usage
await customerServiceWorkflow('I was charged twice last month!');
```

---

## Tools Integration

### Function Tools with Automatic Schema Generation

```typescript
import { tool } from '@openai/agents';
import { z } from 'zod';

// Simple calculator tool
const addTool = tool({
  name: 'add_numbers',
  description: 'Add two numbers together',
  parameters: z.object({
    a: z.number().describe('First number'),
    b: z.number().describe('Second number'),
  }),
  execute: async ({ a, b }) => a + b,
});

// Complex tool with validation
const emailTool = tool({
  name: 'send_email',
  description: 'Send an email to a recipient',
  parameters: z.object({
    to: z.string().email('Invalid email'),
    subject: z.string().min(1).max(200),
    body: z.string().min(1).max(5000),
    cc: z.string().email().optional(),
    attachments: z.string().array().optional(),
  }),
  execute: async ({ to, subject, body, cc, attachments }) => {
    // Mock email sending
    return {
      success: true,
      messageId: `msg_${Date.now()}`,
      sentTo: to,
      timestamp: new Date(),
    };
  },
});

// Database query tool with safety checks
const queryDatabaseTool = tool({
  name: 'query_database',
  description: 'Execute a READ-ONLY database query',
  parameters: z.object({
    query: z.string().describe('SQL SELECT query'),
    limit: z.number().max(1000).default(100),
  }),
  execute: async ({ query, limit }) => {
    if (!query.toUpperCase().startsWith('SELECT')) {
      return { error: 'Only SELECT queries are allowed' };
    }

    // Mock database execution
    return {
      success: true,
      rowCount: 42,
      rows: [],
    };
  },
});

// File operation tool
const readFileTool = tool({
  name: 'read_file',
  description: 'Read contents of a file',
  parameters: z.object({
    filepath: z.string(),
    encoding: z.enum(['utf-8', 'base64']).default('utf-8'),
  }),
  execute: async ({ filepath, encoding }) => {
    // Mock file reading
    return {
      filepath,
      content: 'File content here...',
      size: 1024,
      encoding,
    };
  },
});
```

### Tool Definition Patterns

```typescript
// Generic tool builder for type safety
class ToolBuilder<P extends z.ZodObject<any>, R> {
  private name: string;
  private description: string;
  private parameters: P;
  private execute: (input: z.infer<P>) => Promise<R>;

  constructor(config: {
    name: string;
    description: string;
    parameters: P;
    execute: (input: z.infer<P>) => Promise<R>;
  }) {
    this.name = config.name;
    this.description = config.description;
    this.parameters = config.parameters;
    this.execute = config.execute;
  }

  build() {
    return tool({
      name: this.name,
      description: this.description,
      parameters: this.parameters,
      execute: this.execute,
    });
  }
}

// Usage
const userLookupTool = new ToolBuilder({
  name: 'lookup_user',
  description: 'Look up user information',
  parameters: z.object({
    userId: z.string().regex(/^[0-9]+$/),
    fields: z.string().array().optional(),
  }),
  execute: async ({ userId, fields }) => {
    return {
      id: userId,
      name: 'John Doe',
      email: 'john@example.com',
    };
  },
}).build();
```

### Type-Safe Parameter Validation

```typescript
// Advanced validation with transformations
const dateRangeTool = tool({
  name: 'query_date_range',
  description: 'Query data within a date range',
  parameters: z.object({
    startDate: z.string().pipe(z.coerce.date()),
    endDate: z.string().pipe(z.coerce.date()),
    format: z.enum(['json', 'csv', 'xml']).default('json'),
  }).refine(
    (data) => data.startDate < data.endDate,
    { message: 'Start date must be before end date', path: ['startDate'] }
  ),
  execute: async ({ startDate, endDate, format }) => {
    console.log(`Querying from ${startDate} to ${endDate} as ${format}`);
    return { success: true, count: 100 };
  },
});

// Discriminated union for flexible inputs
const searchTool = tool({
  name: 'search',
  description: 'Search with different strategies',
  parameters: z.discriminatedUnion('type', [
    z.object({
      type: z.literal('keyword'),
      query: z.string(),
      limit: z.number().default(10),
    }),
    z.object({
      type: z.literal('advanced'),
      filters: z.record(z.string(), z.any()),
      sortBy: z.string().optional(),
    }),
  ]),
  execute: async (input) => {
    if (input.type === 'keyword') {
      return { results: [], query: input.query };
    } else {
      return { results: [], filters: input.filters };
    }
  },
});
```

### Type Hints and Interface Definitions

```typescript
interface DatabaseConnection {
  host: string;
  port: number;
  username: string;
  password: string;
  database: string;
}

interface QueryResult<T> {
  success: boolean;
  data?: T[];
  error?: string;
  executionTime: number;
}

const createDatabaseTool = <T extends Record<string, any>>(
  connection: DatabaseConnection,
  tableName: string
) => {
  return tool({
    name: `query_${tableName}`,
    description: `Query ${tableName} from database`,
    parameters: z.object({
      where: z.record(z.string(), z.any()).optional(),
      limit: z.number().optional(),
    }),
    execute: async ({ where, limit }): Promise<QueryResult<T>> => {
      const start = performance.now();
      
      try {
        // Mock execution
        const data: T[] = [];
        const executionTime = performance.now() - start;
        
        return { success: true, data, executionTime };
      } catch (error) {
        return {
          success: false,
          error: error.message,
          executionTime: performance.now() - start,
        };
      }
    },
  });
};
```

### OAI Hosted Tools Overview

```typescript
// Web Search Tool
const webSearchTool = tool({
  name: 'web_search',
  description: 'Search the web for information',
  parameters: z.object({
    query: z.string(),
    numResults: z.number().max(10).default(5),
    language: z.string().default('en'),
  }),
  execute: async ({ query, numResults, language }) => {
    // Using OpenAI web search
    return {
      query,
      results: [
        { title: 'Result 1', url: 'https://...', snippet: 'Description' },
      ],
      language,
    };
  },
});

// File Search Tool
const fileSearchTool = tool({
  name: 'search_files',
  description: 'Search through uploaded documents',
  parameters: z.object({
    query: z.string(),
    fileType: z.enum(['pdf', 'txt', 'md', 'docx']).optional(),
  }),
  execute: async ({ query, fileType }) => {
    return {
      query,
      matches: [],
      fileType,
    };
  },
});

// Code Interpreter Tool
const interpretCodeTool = tool({
  name: 'interpret_code',
  description: 'Execute and interpret code',
  parameters: z.object({
    language: z.enum(['python', 'javascript', 'sql']),
    code: z.string(),
  }),
  execute: async ({ language, code }) => {
    return {
      output: 'Execution result',
      language,
      success: true,
    };
  },
});

// Image Generation Tool
const generateImageTool = tool({
  name: 'generate_image',
  description: 'Generate images with DALL-E',
  parameters: z.object({
    prompt: z.string(),
    size: z.enum(['256x256', '512x512', '1024x1024']).default('1024x1024'),
    quality: z.enum(['standard', 'hd']).default('standard'),
  }),
  execute: async ({ prompt, size, quality }) => {
    return {
      url: 'https://example.com/image.png',
      prompt,
      size,
      quality,
    };
  },
});
```

### Custom Tool Creation with TypeScript

```typescript
// Advanced tool with error handling and retries
class RetryableToolBuilder {
  private maxRetries = 3;
  private retryDelay = 1000;

  withRetry<P, R>(
    toolDef: {
      name: string;
      description: string;
      parameters: z.ZodSchema<P>;
      execute: (input: P) => Promise<R>;
    }
  ) {
    return tool({
      name: toolDef.name,
      description: toolDef.description,
      parameters: toolDef.parameters,
      execute: async (input) => {
        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
          try {
            return await toolDef.execute(input);
          } catch (error) {
            if (attempt === this.maxRetries) throw error;
            await new Promise(resolve =>
              setTimeout(resolve, this.retryDelay * attempt)
            );
          }
        }
      },
    });
  }
}

// Tool with caching
class CachedToolBuilder {
  private cache = new Map<string, { value: any; timestamp: number }>();
  private ttl = 5 * 60 * 1000; // 5 minutes

  withCache<P, R>(
    toolDef: {
      name: string;
      description: string;
      parameters: z.ZodSchema<P>;
      execute: (input: P) => Promise<R>;
    },
    keyFn: (input: P) => string
  ) {
    return tool({
      name: toolDef.name,
      description: toolDef.description,
      parameters: toolDef.parameters,
      execute: async (input) => {
        const key = keyFn(input);
        const cached = this.cache.get(key);

        if (cached && Date.now() - cached.timestamp < this.ttl) {
          return cached.value;
        }

        const result = await toolDef.execute(input);
        this.cache.set(key, { value: result, timestamp: Date.now() });

        return result;
      },
    });
  }
}

// Usage
const apiTool = new CachedToolBuilder().withCache(
  {
    name: 'fetch_user_data',
    description: 'Fetch user data from API',
    parameters: z.object({ userId: z.string() }),
    execute: async ({ userId }) => {
      // API call
      return { id: userId, name: 'User' };
    },
  },
  ({ userId }) => `user-${userId}`
);
```

### Error Handling in Tools

```typescript
interface ToolExecutionResult<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

const robustToolTemplate = <P extends z.ZodObject<any>, T>(config: {
  name: string;
  description: string;
  parameters: P;
  execute: (input: z.infer<P>) => Promise<T>;
}): any => {
  return tool({
    name: config.name,
    description: config.description,
    parameters: config.parameters,
    execute: async (input): Promise<ToolExecutionResult<T>> => {
      try {
        // Validate input
        const validatedInput = config.parameters.parse(input);

        // Execute with timeout
        const result = await Promise.race([
          config.execute(validatedInput),
          new Promise((_, reject) =>
            setTimeout(
              () => reject(new Error('Tool execution timeout')),
              30000
            )
          ),
        ]);

        return { success: true, data: result };
      } catch (error) {
        if (error instanceof z.ZodError) {
          return {
            success: false,
            error: {
              code: 'VALIDATION_ERROR',
              message: 'Invalid tool parameters',
              details: error.issues,
            },
          };
        }

        return {
          success: false,
          error: {
            code: 'EXECUTION_ERROR',
            message: error.message || 'Tool execution failed',
          },
        };
      }
    },
  });
};
```

### Async Tool Execution Patterns

```typescript
// Promise-based execution
const asyncDataFetchTool = tool({
  name: 'fetch_data',
  description: 'Fetch data asynchronously',
  parameters: z.object({ endpoint: z.string() }),
  execute: async ({ endpoint }) => {
    const response = await fetch(endpoint);
    return response.json();
  },
});

// Stream-based tool
const streamProcessingTool = tool({
  name: 'process_stream',
  description: 'Process streaming data',
  parameters: z.object({ source: z.string() }),
  execute: async ({ source }) => {
    const chunks: string[] = [];

    const response = await fetch(source);
    const reader = response.body?.getReader();

    if (reader) {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(new TextDecoder().decode(value));
      }
    }

    return { totalChunks: chunks.length, data: chunks.join('') };
  },
});

// Concurrent tool execution
const concurrentToolExecution = tool({
  name: 'fetch_multiple',
  description: 'Fetch from multiple sources concurrently',
  parameters: z.object({ urls: z.string().array() }),
  execute: async ({ urls }) => {
    const results = await Promise.allSettled(
      urls.map(url => fetch(url).then(r => r.json()))
    );

    return {
      successful: results.filter(r => r.status === 'fulfilled').length,
      failed: results.filter(r => r.status === 'rejected').length,
      results: results.map(r =>
        r.status === 'fulfilled' ? r.value : { error: r.reason.message }
      ),
    };
  },
});
```

### Tool Result Typing

```typescript
interface ToolResult {
  toolName: string;
  success: boolean;
  output: any;
  executionTime: number;
  metadata?: Record<string, any>;
}

const createTypedTool = <P extends z.ZodObject<any>, R>(config: {
  name: string;
  description: string;
  parameters: P;
  execute: (input: z.infer<P>) => Promise<R>;
  outputSchema?: z.ZodSchema<R>;
}): any => {
  return tool({
    ...config,
    execute: async (input): Promise<ToolResult> => {
      const start = performance.now();

      try {
        const result = await config.execute(input);

        if (config.outputSchema) {
          config.outputSchema.parse(result);
        }

        return {
          toolName: config.name,
          success: true,
          output: result,
          executionTime: performance.now() - start,
        };
      } catch (error) {
        return {
          toolName: config.name,
          success: false,
          output: null,
          executionTime: performance.now() - start,
          metadata: { error: error.message },
        };
      }
    },
  });
};
```

---

## Structured Output

### Output Schema Definition with TypeScript Types

```typescript
import { z } from 'zod';

// Basic structured output
const analysisResultSchema = z.object({
  summary: z.string(),
  keyPoints: z.string().array(),
  sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']),
  confidence: z.number().min(0).max(1),
});

type AnalysisResult = z.infer<typeof analysisResultSchema>;

// Complex nested structure
const researchReportSchema = z.object({
  title: z.string(),
  abstract: z.string(),
  sections: z.array(z.object({
    title: z.string(),
    content: z.string(),
    subsections: z.array(z.object({
      title: z.string(),
      content: z.string(),
      citations: z.array(z.object({
        author: z.string(),
        year: z.number(),
        title: z.string(),
      })).optional(),
    })).optional(),
  })),
  conclusion: z.string(),
  bibliography: z.array(z.string()),
  metadata: z.object({
    author: z.string(),
    createdAt: z.coerce.date(),
    version: z.string(),
  }),
});

type ResearchReport = z.infer<typeof researchReportSchema>;

// Discriminated union for multiple output types
const documentTypeSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('report'),
    content: z.string(),
    metrics: z.record(z.string(), z.number()),
  }),
  z.object({
    type: z.literal('summary'),
    content: z.string(),
    length: z.enum(['short', 'medium', 'long']),
  }),
  z.object({
    type: z.literal('checklist'),
    items: z.array(z.object({
      description: z.string(),
      completed: z.boolean(),
    })),
  }),
]);

type DocumentType = z.infer<typeof documentTypeSchema>;

// Generic schema factory
function createPaginatedSchema<T extends z.ZodSchema>(itemSchema: T) {
  return z.object({
    items: itemSchema.array(),
    total: z.number(),
    page: z.number(),
    pageSize: z.number(),
    hasMore: z.boolean(),
  });
}

const paginatedResults = createPaginatedSchema(
  z.object({
    id: z.string(),
    title: z.string(),
  })
);

type PaginatedResults = z.infer<typeof paginatedResults>;
```

### Structured Outputs with Reasoning Content

```typescript
const reasoningOutputSchema = z.object({
  reasoning: z.array(z.object({
    step: z.number(),
    thought: z.string(),
    justification: z.string(),
  })),
  conclusion: z.string(),
  confidence: z.number().min(0).max(1),
  alternativeApproaches: z.string().array().optional(),
});

type ReasoningOutput = z.infer<typeof reasoningOutputSchema>;

// Usage with agent
async function generateReasonedOutput(
  agent: Agent,
  query: string
): Promise<ReasoningOutput> {
  const systemPrompt = `Provide your reasoning in the following JSON format:
{
  "reasoning": [
    {"step": 1, "thought": "...", "justification": "..."},
    {"step": 2, "thought": "...", "justification": "..."}
  ],
  "conclusion": "...",
  "confidence": 0.95,
  "alternativeApproaches": ["...", "..."]
}`;

  const result = await run(agent, query);
  
  try {
    const parsed = JSON.parse(result.finalOutput);
    return reasoningOutputSchema.parse(parsed);
  } catch {
    throw new Error('Failed to parse reasoned output');
  }
}
```

### JSON Mode Configuration

```typescript
interface JSONModeConfig {
  enabled: boolean;
  strict: boolean;
  schema?: z.ZodSchema;
}

// Agent with JSON mode
const jsonAgent = new Agent({
  name: 'JSON Agent',
  instructions: `Always respond with valid JSON.
    Format your response as:
    {
      "status": "success",
      "data": {},
      "timestamp": "ISO date string"
    }`,
  model: '%%model=openai-small%%',
  modelSettings: { temperature: 0 },
});

// Helper to ensure JSON output
async function ensureJsonOutput<T extends z.ZodSchema>(
  agent: Agent,
  query: string,
  schema: T
): Promise<z.infer<T>> {
  const result = await run(agent, query);

  try {
    const parsed = JSON.parse(result.finalOutput);
    return schema.parse(parsed);
  } catch (error) {
    console.error('JSON parsing failed:', error);
    throw error;
  }
}
```

### Type-Safe Output Parsing

```typescript
class OutputParser<T> {
  constructor(private schema: z.ZodSchema<T>) {}

  parse(output: string): T | Error {
    try {
      // Try direct JSON parse
      const json = JSON.parse(output);
      return this.schema.parse(json);
    } catch (jsonError) {
      try {
        // Try extracting JSON from markdown code blocks
        const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);
        if (jsonMatch) {
          const json = JSON.parse(jsonMatch[1]);
          return this.schema.parse(json);
        }
      } catch (markdownError) {
        // Try extracting JSON from raw text
        const rawMatch = output.match(/\{[\s\S]*\}/);
        if (rawMatch) {
          const json = JSON.parse(rawMatch[0]);
          return this.schema.parse(json);
        }
      }

      return new Error(`Failed to parse output: ${output}`);
    }
  }

  parseAsync(output: string): Promise<T> {
    return Promise.resolve().then(() => {
      const result = this.parse(output);
      if (result instanceof Error) throw result;
      return result;
    });
  }
}

// Usage
const resultParser = new OutputParser(analysisResultSchema);
const parsed = resultParser.parse(agentOutput);
```

### Validation Strategies

```typescript
// Strategy 1: Retry on validation failure
async function parseWithRetry<T extends z.ZodSchema>(
  agent: Agent,
  query: string,
  schema: T,
  maxRetries = 3
): Promise<z.infer<T>> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const result = await run(agent, query);

    try {
      const parsed = JSON.parse(result.finalOutput);
      return schema.parse(parsed);
    } catch (error) {
      if (attempt === maxRetries) throw error;

      const improvedQuery = `${query}

Please ensure your response is valid JSON matching this structure:
${JSON.stringify(z.toJSONSchema(schema), null, 2)}`;

      // Retry with corrected prompt
      continue;
    }
  }

  throw new Error('Max retries exceeded');
}

// Strategy 2: Partial validation with fallback
async function parseWithFallback<T extends z.ZodSchema>(
  agent: Agent,
  query: string,
  schema: T,
  fallback: z.infer<T>
): Promise<z.infer<T>> {
  try {
    const result = await run(agent, query);
    const parsed = JSON.parse(result.finalOutput);
    return schema.parse(parsed);
  } catch (error) {
    console.warn('Validation failed, using fallback:', error);
    return fallback;
  }
}

// Strategy 3: Progressive validation
async function progressiveValidation<T extends z.ZodSchema>(
  output: string,
  schema: T
): Promise<Partial<z.infer<T>>> {
  const parsed = JSON.parse(output);
  const result: any = {};

  const shape = schema instanceof z.ZodObject ? schema.shape : {};

  for (const [key, fieldSchema] of Object.entries(shape)) {
    try {
      result[key] = (fieldSchema as z.ZodSchema).parse(parsed[key]);
    } catch {
      console.warn(`Failed to validate ${key}, skipping`);
    }
  }

  return result;
}
```

### Complex Nested Structures with Interfaces

```typescript
// E-commerce product structure
interface ProductReview {
  rating: number;
  text: string;
  author: string;
  helpfulCount: number;
}

interface ProductVariant {
  sku: string;
  color?: string;
  size?: string;
  price: number;
  stock: number;
}

interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  category: string;
  variants: ProductVariant[];
  reviews: ProductReview[];
  metadata: {
    createdAt: Date;
    updatedAt: Date;
    tags: string[];
  };
}

const productSchema: z.ZodSchema<Product> = z.object({
  id: z.string().uuid(),
  name: z.string().min(1).max(200),
  description: z.string().max(5000),
  price: z.number().positive(),
  category: z.string(),
  variants: z.array(z.object({
    sku: z.string(),
    color: z.string().optional(),
    size: z.string().optional(),
    price: z.number().positive(),
    stock: z.number().nonnegative(),
  })),
  reviews: z.array(z.object({
    rating: z.number().min(1).max(5),
    text: z.string(),
    author: z.string(),
    helpfulCount: z.number().nonnegative(),
  })),
  metadata: z.object({
    createdAt: z.coerce.date(),
    updatedAt: z.coerce.date(),
    tags: z.string().array(),
  }),
});
```

### Type Enforcement with Generics

```typescript
// Generic response wrapper
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
  timestamp: Date;
}

// Type-safe agent factory
class TypedAgentFactory {
  static create<T>(
    name: string,
    instructions: string,
    outputSchema: z.ZodSchema<T>
  ): {
    execute: (input: string) => Promise<ApiResponse<T>>;
  } {
    const agent = new Agent({ name, instructions });

    return {
      execute: async (input: string): Promise<ApiResponse<T>> => {
        try {
          const result = await run(agent, input);
          const parsed = JSON.parse(result.finalOutput);
          const validated = outputSchema.parse(parsed);

          return {
            success: true,
            data: validated,
            timestamp: new Date(),
          };
        } catch (error) {
          return {
            success: false,
            error: {
              code: 'PARSE_ERROR',
              message: error.message,
            },
            timestamp: new Date(),
          };
        }
      },
    };
  }
}

// Usage
const typedAgent = TypedAgentFactory.create(
  'Analysis Agent',
  'Provide structured analysis',
  analysisResultSchema
);

const response = await typedAgent.execute('Analyse this data');
if (response.success && response.data) {
  console.log('Analysis:', response.data.summary);
}
```

---

## Summary

This comprehensive guide provides an extensive foundation for building production-ready agentic applications with TypeScript and the OpenAI Agents SDK. Each section includes:

- **Conceptual Explanations**: Understanding the 'why' behind each feature
- **Complete TypeScript Code Examples**: With full type annotations
- **Production-Ready Patterns**: Battle-tested approaches
- **Real-World Use Cases**: Practical scenarios and implementations
- **Type Safety Best Practices**: Leveraging TypeScript's capabilities

The guide emphasises lightweight primitives, model-agnostic design, and built-in observability throughout all implementations.

---

### Realtime Voice Agents

Realtime agents live in `@openai/agents/realtime` (re-exported by `@openai/agents`). A `RealtimeSession` connects over WebRTC in the browser or WebSocket on the server; the default model has been `gpt-realtime-2.1` since 0.13.0.

```typescript
import { RealtimeAgent, RealtimeSession } from '@openai/agents/realtime';

const agent = new RealtimeAgent({
  name: 'Voice Assistant',
  instructions: 'You are a helpful voice assistant.',
});

const session = new RealtimeSession(agent, {
  transport: 'websocket', // server-side; browsers default to WebRTC
});

session.on('audio', (event) => {
  playAudio(event.data); // ArrayBuffer of PCM16 audio
});

await session.connect({ apiKey: process.env.OPENAI_API_KEY! });

// Stream microphone audio in as you capture it
session.sendAudio(recordedPcm);
```

In the browser, mint a short-lived client secret on your server and pass it as `apiKey` rather than exposing your API key.

---

### Breaking changes by version

The packages use `0.Y.Z` versioning: a minor bump may break public interfaces. Changes since 0.11 most likely to affect existing code, from the upstream changelogs:

| Version | Change |
|---------|--------|
| 0.12.0 | Realtime tool dispatch across handoffs made unambiguous. |
| 0.13.0 | Default Realtime model `gpt-realtime-2.1`. |
| 0.14.0 | Sensitive model and tool data are no longer logged by default (programmatic opt-in); default task and turn tracing with a per-run opt-out. |
| 0.15.0 | Requires `openai` (Node SDK) v7.2+; default model `gpt-5.6-luna`; MCP 2026-07-28 protocol negotiation with legacy fallback. |
| 0.16.0 | Scripted model, sandbox session and Realtime testing utilities (`@openai/agents/testing`). |
| 0.17.0 | Blocked tool outputs are redacted from replay state. |
| 0.18.0 | Unix-local and Docker sandbox file operations hardened (optional Python support; 10 MB editor limit for Docker). |
| 0.19.0 | Agent-tool streaming callbacks capped at 1024 pending events (`onStreamMaxPendingEvents`); resumed MCP calls bound to their original server; default tool-parse failures redacted. |
| 0.20.0 | Automatic MCP tool discovery limited to 64 pages (`maxListPages`); model calls after approval resumes count towards `maxTurns`. |
| 0.21.0 | Runloop sandbox provider removed; usage-detail values need narrowing to numbers before arithmetic. |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| %%version=@openai/agents%% | %%versions-checked%% | Re-verified against the current release (last revised for 0.11.6). Rewrote the Core Primitives section against the real API (it previously wrapped invented config types); fixed install commands (`openai-agents` and `@openai/agents-extensions-vercel-ai-sdk` are not the package names); replaced non-existent APIs (`new Session()`, `agent.run()`, `result.currentAgent`, iterating a stream result as text, `RealtimeRunner`); Zod 4 fixes (`z.record` key type, `error.issues`); added a breaking-changes table; updated model names. |
| 0.11.6 | May 29, 2026 | Version bump; key symbols re-checked. |
| 0.8.0 | March 2026 | Realtime section added. |
| 0.3.2 | November 2025 | First version of this guide. |

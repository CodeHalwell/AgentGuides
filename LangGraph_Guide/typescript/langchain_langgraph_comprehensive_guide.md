Latest: 1.0.2
# LangChain.js and LangGraph.js Comprehensive Technical Guide

**Beginner to Expert Level | TypeScript-Native Implementation | Production-Ready Patterns**

---

## Table of Contents

1. [Core Fundamentals](#core-fundamentals)
2. [Simple Agents (LangChain.js)](#simple-agents-langchainjs)
3. [Simple Agents (LangGraph.js)](#simple-agents-langgraphjs)
4. [Multi-Agent Systems](#multi-agent-systems)
5. [Tools Integration](#tools-integration)
6. [Structured Output](#structured-output)
7. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
8. [Agentic Patterns](#agentic-patterns)
9. [Memory Systems (LangChain.js)](#memory-systems-langchainjs)
10. [State Management (LangGraph.js)](#state-management-langgraphjs)
11. [LangGraph Checkpointing](#langgraph-checkpointing)
12. [Conditional Logic (LangGraph.js)](#conditional-logic-langgraphjs)
13. [Context Engineering](#context-engineering)
14. [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
15. [Human-in-the-Loop (LangGraph.js)](#human-in-the-loop-langgraphjs)
16. [LangGraph Studio](#langgraph-studio)
17. [Streaming](#streaming)
18. [Chains and Sequences](#chains-and-sequences)
19. [Callbacks and Tracing](#callbacks-and-tracing)
20. [TypeScript Patterns](#typescript-patterns)
21. [Deployment Patterns](#deployment-patterns)
22. [Advanced Topics](#advanced-topics)

---

## Core Fundamentals

### Installation and Package Setup

LangChain.js and LangGraph.js are modular frameworks designed to work seamlessly with TypeScript. The installation process depends on your specific use case, but we'll cover the most common scenarios.

#### Step 1: Project Initialisation

First, initialise your Node.js project with TypeScript support:

```bash
# Create a new directory
mkdir my-langchain-project
cd my-langchain-project

# Initialise npm project
npm init -y

# Install TypeScript and necessary tooling
npm install --save-dev typescript @types/node ts-node

# Initialise TypeScript configuration
npx tsc --init
```

#### Step 2: Update TypeScript Configuration

Create or update your `tsconfig.json` with appropriate settings for modern TypeScript:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020"],
    "moduleResolution": "node",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

#### Step 3: Install Core Packages

```bash
# Core LangChain.js packages
npm install @langchain/core @langchain/community

# LangGraph.js for state orchestration
npm install @langchain/langgraph

# Popular LLM provider integrations
npm install @langchain/openai
npm install @langchain/anthropic
npm install @langchain/google-vertexai

# Utilities and validation
npm install zod dotenv
npm install --save-dev @types/dotenv
```

#### Step 4: Environment Configuration

Create a `.env` file in your project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4-turbo

# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-your-api-key-here

# LangSmith Observability (Optional)
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=my-project
LANGSMITH_TRACING_V2=true

# Other Configuration
NODE_ENV=development
LOG_LEVEL=info
```

Create a file to load these safely:

```typescript
// src/config.ts
import dotenv from 'dotenv';

dotenv.config();

export const config = {
  openai: {
    apiKey: process.env.OPENAI_API_KEY || '',
    model: process.env.OPENAI_MODEL || 'gpt-4-turbo',
  },
  anthropic: {
    apiKey: process.env.ANTHROPIC_API_KEY || '',
  },
  langsmith: {
    apiKey: process.env.LANGSMITH_API_KEY,
    project: process.env.LANGSMITH_PROJECT,
  },
  nodeEnv: process.env.NODE_ENV || 'development',
  logLevel: process.env.LOG_LEVEL || 'info',
};
```

### TypeScript-First Architecture

LangChain.js and LangGraph.js are built with TypeScript at their core, offering several advantages:

#### Type Safety Benefits

```typescript
// ✅ GOOD: Fully typed component
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, AIMessage } from '@langchain/core/messages';

interface ConversationContext {
  userId: string;
  history: (HumanMessage | AIMessage)[];
  metadata: Record<string, unknown>;
}

async function processConversation(
  context: ConversationContext
): Promise<AIMessage> {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0.7,
  });

  const response = await model.invoke(context.history);
  
  // TypeScript ensures response is AIMessage
  if (response instanceof AIMessage) {
    console.log('Received valid AI message');
  }
  
  return response;
}
```

#### Generics for Flexibility

```typescript
// Define a generic agent execution function
interface AgentInput<T> {
  data: T;
  userId: string;
}

interface AgentOutput<R> {
  result: R;
  executionTime: number;
  tokensUsed?: number;
}

async function executeAgent<T, R>(
  input: AgentInput<T>,
  processor: (data: T) => Promise<R>
): Promise<AgentOutput<R>> {
  const startTime = Date.now();
  
  try {
    const result = await processor(input.data);
    const executionTime = Date.now() - startTime;
    
    return {
      result,
      executionTime,
    };
  } catch (error) {
    throw new Error(`Agent execution failed for user ${input.userId}: ${error}`);
  }
}

// Usage with different data types
const numberResult = await executeAgent(
  { data: 42, userId: 'user-123' },
  async (n) => n * 2
);

const stringResult = await executeAgent(
  { data: 'hello', userId: 'user-456' },
  async (s) => s.toUpperCase()
);
```

### Relationship Between LangChain.js and LangGraph.js

LangChain.js and LangGraph.js serve different but complementary purposes:

| Aspect | LangChain.js | LangGraph.js |
|--------|-------------|-------------|
| **Primary Purpose** | Building blocks for LLM applications | Stateful orchestration framework |
| **Abstraction Level** | Higher-level components and chains | Lower-level graph execution engine |
| **State Management** | Stateless or basic memory | First-class stateful support |
| **Workflow Type** | Sequential or branching chains | Complex multi-step workflows |
| **Best For** | Simple agents, RAG pipelines | Multi-agent systems, long-running workflows |
| **Persistence** | Optional, via custom handlers | Built-in, via CheckpointSaver |

#### Integration Pattern

```typescript
// LangChain.js components power LangGraph.js workflows
import { StateGraph, START, END } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

// LangChain.js tool
const calculator = tool(
  {
    name: 'calculator',
    description: 'Perform arithmetic operations',
    schema: z.object({
      operation: z.enum(['add', 'subtract', 'multiply', 'divide']),
      a: z.number(),
      b: z.number(),
    }),
  },
  ({ operation, a, b }) => {
    switch (operation) {
      case 'add': return a + b;
      case 'subtract': return a - b;
      case 'multiply': return a * b;
      case 'divide': return b !== 0 ? a / b : 0;
    }
  }
);

// LangChain.js model
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// LangGraph.js state structure
interface MathState {
  problem: string;
  solution?: number;
  steps: string[];
}

// LangGraph.js workflow using LangChain.js components
const graph = new StateGraph<MathState>({
  channels: {
    problem: { value_type: 'string' },
    solution: { value_type: 'number', optional: true },
    steps: { value_type: 'array', default: () => [] },
  },
});

graph.addNode('solve', async (state) => {
  state.steps.push('Solving: ' + state.problem);
  // Use LangChain.js model and tools here
  return state;
});

graph.addEdge(START, 'solve');
graph.addEdge('solve', END);

const workflow = graph.compile();
```

### Core Classes and Concepts

#### ChatModels

ChatModels are the foundation of LLM interactions in LangChain.js. They wrap API calls to language models and provide a unified interface.

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { HumanMessage, SystemMessage, AIMessage } from '@langchain/core/messages';

// OpenAI ChatModel
const openaiModel = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4-turbo',
  temperature: 0.7,
  maxTokens: 2000,
  topP: 0.9,
});

// Anthropic ChatModel
const anthropicModel = new ChatAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  model: 'claude-3-opus-20240229',
  temperature: 0.5,
});

// Invoke with message array
const messages = [
  new SystemMessage('You are a helpful code assistant'),
  new HumanMessage('How do I read a file in TypeScript?'),
];

const response = await openaiModel.invoke(messages);
console.log(response.content); // "To read a file in TypeScript, you can use..."

// Streaming invocation
const stream = await openaiModel.stream(messages);
for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

#### PromptTemplates

Prompt templates enable dynamic, reusable prompt construction with variable substitution.

```typescript
import { PromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts';

// String-based PromptTemplate
const simpleTemplate = new PromptTemplate({
  inputVariables: ['topic', 'level'],
  template: `Explain {topic} at a {level} level of complexity.`,
});

const formatted = await simpleTemplate.format({
  topic: 'quantum computing',
  level: 'beginner',
});
// Output: "Explain quantum computing at a beginner level of complexity."

// ChatPromptTemplate for multi-message prompts
const chatTemplate = ChatPromptTemplate.fromMessages([
  [
    'system',
    'You are an expert {field} professional. Your role is to help users understand {field} concepts.',
  ],
  ['user', 'Question: {question}'],
  ['user', 'Context: {context}'],
]);

const chatFormatted = await chatTemplate.formatMessages({
  field: 'machine learning',
  question: 'What is overfitting?',
  context: 'We are building a classification model',
});
```

#### OutputParsers

Output parsers transform model responses into structured, usable formats.

```typescript
import { StringOutputParser } from '@langchain/core/output_parsers';
import { z } from 'zod';
import { ZodOutputParser } from 'langchain/output_parsers';

// Simple string parser
const simpleParser = new StringOutputParser();
const textResult = await simpleParser.parse('Some LLM output');

// Zod schema parser for structured output
const personSchema = z.object({
  name: z.string(),
  age: z.number(),
  email: z.string().email(),
  hobbies: z.array(z.string()),
});

const zodParser = new ZodOutputParser(personSchema);

const jsonOutput = `{
  "name": "Alice Johnson",
  "age": 28,
  "email": "alice@example.com",
  "hobbies": ["reading", "hiking", "programming"]
}`;

const parsed = await zodParser.parse(jsonOutput);
console.log(parsed.name); // TypeScript knows this is a string
console.log(parsed.age); // TypeScript knows this is a number
```

#### Tools

Tools extend agent capabilities by providing access to external functionalities.

```typescript
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

// Define tools with Zod schemas
const weatherTool = tool(
  {
    name: 'get_weather',
    description: 'Get the current weather for a location',
    schema: z.object({
      location: z.string().describe('City or location name'),
      unit: z.enum(['celsius', 'fahrenheit']).default('celsius'),
    }),
  },
  async ({ location, unit }) => {
    // In a real application, call an actual weather API
    return `Weather in ${location}: 22°${unit === 'celsius' ? 'C' : 'F'}, Sunny`;
  }
);

// Tool with validation and error handling
const databaseTool = tool(
  {
    name: 'query_database',
    description: 'Query application database for user information',
    schema: z.object({
      userId: z.string().describe('The unique user identifier'),
      fields: z.array(z.string()).describe('Fields to retrieve'),
    }),
  },
  async ({ userId, fields }) => {
    try {
      // Validate user ID format
      if (!userId.match(/^user_[0-9a-f]{8}$/)) {
        throw new Error('Invalid user ID format');
      }
      
      // Simulate database query
      const userData = {
        userId,
        name: 'John Doe',
        email: 'john@example.com',
        createdAt: '2024-01-01',
      };
      
      return fields
        .filter((f) => f in userData)
        .map((f) => `${f}: ${userData[f as keyof typeof userData]}`)
        .join(', ');
    } catch (error) {
      return `Error querying database: ${error}`;
    }
  }
);

// List available tools (commonly used in agent setup)
const tools = [weatherTool, databaseTool];
const toolDescriptions = tools
  .map((t) => `- ${t.name}: ${t.description}`)
  .join('\n');

console.log('Available tools:');
console.log(toolDescriptions);
```

#### Agents

Agents are autonomous entities that use language models and tools to solve problems.

```typescript
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { AgentExecutor } from '@langchain/core/agents';

// Agents are created by combining a model, tools, and execution strategy
const basicAgent = createReactAgent({
  llm: openaiModel,
  tools: [weatherTool, databaseTool],
});

// Execute the agent
const result = await basicAgent.invoke({
  input: 'What is the weather in London and who is user_12345678?',
});

console.log(result.output);
```

### LangGraph Classes

#### StateGraph

StateGraph is the core orchestration primitive in LangGraph.js for building stateful workflows.

```typescript
import { StateGraph, START, END } from '@langchain/langgraph';
import { Annotation } from '@langchain/langgraph';

// Define state with Annotation for full type safety
const StateAnnotation = Annotation.Root({
  input: Annotation<string>,
  intermediateSteps: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => x.concat(y),
  }),
  output: Annotation<string | null>({ default: () => null }),
});

// Create graph with type-safe state
const graph = new StateGraph(StateAnnotation);

// Define nodes as TypeScript functions
graph.addNode('validateInput', (state) => {
  if (state.input.length === 0) {
    return { ...state, output: 'Error: empty input' };
  }
  return { ...state, intermediateSteps: ['Input validated'] };
});

graph.addNode('processData', (state) => {
  const result = state.input.toUpperCase();
  return {
    ...state,
    intermediateSteps: [...state.intermediateSteps, 'Data processed'],
    output: result,
  };
});

// Connect nodes with edges
graph.addEdge(START, 'validateInput');
graph.addEdge('validateInput', 'processData');
graph.addEdge('processData', END);

// Compile and execute
const runnable = graph.compile();
const result = await runnable.invoke({ input: 'hello' });

console.log(result.output); // "HELLO"
console.log(result.intermediateSteps); // ['Input validated', 'Data processed']
```

#### MessageGraph

MessageGraph specialises in workflows centred around message passing.

```typescript
import { MessageGraph } from '@langchain/langgraph';
import { HumanMessage, AIMessage } from '@langchain/core/messages';

const messageGraph = new MessageGraph();

messageGraph.addNode('respondAgent', (state) => {
  // State is an array of messages
  const lastMessage = state[state.length - 1];
  return [
    new AIMessage({
      content: 'Response to: ' + lastMessage.content,
    }),
  ];
});

messageGraph.addEdge('respondAgent', END);
messageGraph.setEntryPoint('respondAgent');

const messageRunnable = messageGraph.compile();
const messages = [new HumanMessage('Hello there')];
const output = await messageRunnable.invoke(messages);
```

#### Command

Command objects allow dynamic control flow within graph execution.

```typescript
import { Command } from '@langchain/langgraph';

interface WorkflowState {
  counter: number;
  shouldContinue: boolean;
}

graph.addNode('increment', (state: WorkflowState) => {
  const newCount = state.counter + 1;
  
  if (newCount > 5) {
    // Command for immediate termination
    return new Command({
      goto: END,
      update: { counter: newCount },
    });
  }
  
  return { counter: newCount, shouldContinue: true };
});

graph.addNode('process', (state: WorkflowState) => {
  console.log('Processing with counter:', state.counter);
  return state;
});

graph.addEdge(START, 'increment');
graph.addEdge('increment', 'process');
graph.addEdge('process', 'increment');
```

---

## Simple Agents (LangChain.js)

Agents in LangChain.js are autonomous systems that make decisions about which tools to use to accomplish tasks.

### Creating Basic Agents with createReactAgent

The ReAct (Reasoning + Acting) paradigm is the most common agent pattern. It interleaves thought steps with tool invocations.

```typescript
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

// Define tools
const addTool = tool(
  {
    name: 'add',
    description: 'Add two numbers',
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  },
  ({ a, b }) => a + b
);

const multiplyTool = tool(
  {
    name: 'multiply',
    description: 'Multiply two numbers',
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  },
  ({ a, b }) => a * b
);

// Create agent
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0.7,
});

const agent = createReactAgent({
  llm: model,
  tools: [addTool, multiplyTool],
});

// Execute
const result = await agent.invoke({
  input: 'What is (5 + 3) * 2?',
});

console.log('Agent result:', result);
```

### Agent Types

#### ReAct Agents

ReAct agents are the most versatile, suitable for complex reasoning tasks.

```typescript
// Already demonstrated above - createReactAgent implements ReAct pattern
```

#### OpenAI Functions Agents

These agents use OpenAI's function-calling API for more reliable tool invocation.

```typescript
import { createOpenAIFunctionsAgent } from '@langchain/langgraph/prebuilt';
import { AgentExecutor } from '@langchain/core/agents';

const functionsAgent = createOpenAIFunctionsAgent({
  llm: model,
  tools: [addTool, multiplyTool],
});

const executor = new AgentExecutor({
  agent: functionsAgent,
  tools: [addTool, multiplyTool],
  verbose: true,
});

const result = await executor.invoke({
  input: 'Calculate 10 + 5 and then multiply by 2',
});
```

#### Structured Chat Agents

These agents work with JSON-formatted tool calls and are particularly good for consistent output formatting.

```typescript
import { createStructuredChatAgent } from '@langchain/langgraph/prebuilt';

const chatAgent = createStructuredChatAgent({
  llm: model,
  tools: [addTool, multiplyTool],
});
```

### Tool Integration and Usage

#### Creating Custom Tools

```typescript
import { DynamicStructuredTool } from '@langchain/core/tools';

// More advanced tool creation with validation
const databaseTool = new DynamicStructuredTool({
  name: 'query_users',
  description: 'Query user database with filtering options',
  schema: z.object({
    age_min: z.number().describe('Minimum age to filter'),
    age_max: z.number().describe('Maximum age to filter'),
    limit: z.number().default(10).describe('Number of results'),
  }),
  func: async (input) => {
    // Validate inputs
    if (input.age_min < 0 || input.age_max < 0) {
      throw new Error('Age values must be positive');
    }
    if (input.age_min > input.age_max) {
      throw new Error('age_min must be less than or equal to age_max');
    }
    
    // Simulate database query
    return `Found ${Math.floor(Math.random() * 100)} users between ${input.age_min} and ${input.age_max}`;
  },
});
```

#### Async Tool Execution

```typescript
const apiTool = tool(
  {
    name: 'call_external_api',
    description: 'Call an external API with a URL',
    schema: z.object({
      url: z.string().url(),
      method: z.enum(['GET', 'POST']).default('GET'),
    }),
  },
  async ({ url, method }) => {
    try {
      const response = await fetch(url, { method });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      return { error: `API call failed: ${error}` };
    }
  }
);
```

### Single-Task Agent Execution

```typescript
// Simple one-off agent execution
async function executeSimpleTask(prompt: string): Promise<string> {
  const agent = createReactAgent({
    llm: model,
    tools: [addTool, multiplyTool],
  });

  const result = await agent.invoke({ input: prompt });
  return result.output || '';
}

const result = await executeSimpleTask('Multiply 7 by 8');
```

### Agent Executors and Configuration

```typescript
import { AgentExecutor } from '@langchain/core/agents';

const executor = new AgentExecutor({
  agent: basicAgent,
  tools: [addTool, multiplyTool],
  maxIterations: 10, // Prevent infinite loops
  returnIntermediateSteps: true, // Capture thought process
  handleParsingErrors: true, // Gracefully handle malformed tool calls
  verbose: true, // Enable detailed logging
});

const result = await executor.invoke({
  input: 'Your task here',
});

console.log('Final output:', result.output);
console.log('Steps taken:', result.intermediateSteps);
```

### Streaming Responses

```typescript
// Stream tokens as they arrive
async function streamAgentResponse(prompt: string): Promise<void> {
  const stream = await agent.stream({
    input: prompt,
  });

  for await (const event of stream) {
    if (event.type === 'tool_start') {
      console.log(`Using tool: ${event.tool}`);
    } else if (event.type === 'tool_end') {
      console.log(`Tool result: ${event.result}`);
    } else if (event.type === 'agent_message') {
      console.log(`Agent: ${event.message.content}`);
    }
  }
}
```

### Error Handling

```typescript
async function robustAgentExecution(prompt: string): Promise<string> {
  try {
    const result = await agent.invoke({ input: prompt });
    return result.output || 'No output generated';
  } catch (error) {
    if (error instanceof Error) {
      if (error.message.includes('API')) {
        console.error('LLM API error:', error.message);
        return 'Unable to connect to LLM service. Please try again.';
      } else if (error.message.includes('Tool')) {
        console.error('Tool execution error:', error.message);
        return 'Tool execution failed. Please check your input.';
      }
    }
    console.error('Unexpected agent error:', error);
    return 'An unexpected error occurred.';
  }
}
```

---

## Simple Agents (LangGraph.js)

LangGraph.js provides more granular control over agent workflows through explicit graph definition.

### Creating Basic StateGraph Instances

```typescript
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { z } from 'zod';

// Define state schema
const agentStateSchema = z.object({
  input: z.string(),
  thoughts: z.array(z.string()).default(() => []),
  actions: z.array(z.object({
    tool: z.string(),
    input: z.any(),
    timestamp: z.number(),
  })).default(() => []),
  result: z.string().optional(),
  isComplete: z.boolean().default(false),
});

type AgentState = z.infer<typeof agentStateSchema>;

// Create state graph
const graph = new StateGraph<AgentState>({
  channels: {
    input: { value_type: 'string' },
    thoughts: { 
      value_type: 'array',
      default: () => [],
    },
    actions: { 
      value_type: 'array',
      default: () => [],
    },
    result: { value_type: 'string', optional: true },
    isComplete: { value_type: 'boolean', default: () => false },
  },
});
```

### Node Definitions with TypeScript Functions

Nodes are the fundamental building blocks of LangGraph workflows.

```typescript
// Node that analyzes the input
graph.addNode('analyzeInput', async (state: AgentState) => {
  console.log('Analysing input:', state.input);
  
  const thoughts = [
    ...state.thoughts,
    `Received input: ${state.input}`,
  ];
  
  return {
    ...state,
    thoughts,
  };
});

// Node that selects appropriate tools
graph.addNode('selectTool', async (state: AgentState) => {
  const thoughts = [
    ...state.thoughts,
    'Determining appropriate tool to use...',
  ];
  
  let selectedTool = 'calculator';
  if (state.input.includes('search')) {
    selectedTool = 'search';
  }
  
  return {
    ...state,
    thoughts,
    actions: [
      ...state.actions,
      {
        tool: selectedTool,
        input: state.input,
        timestamp: Date.now(),
      },
    ],
  };
});

// Node that executes selected tool
graph.addNode('executeToolAction', async (state: AgentState) => {
  const lastAction = state.actions[state.actions.length - 1];
  
  let result = '';
  if (lastAction.tool === 'calculator') {
    // Simulate tool execution
    result = 'Calculation result: 42';
  } else if (lastAction.tool === 'search') {
    result = 'Search results: 10 items found';
  }
  
  return {
    ...state,
    thoughts: [
      ...state.thoughts,
      `Executed ${lastAction.tool}: ${result}`,
    ],
    result,
    isComplete: true,
  };
});
```

### State Schemas with TypeScript Interfaces

Proper state definition is crucial for type safety and workflow clarity.

```typescript
import { Annotation } from '@langchain/langgraph';

// Using Annotation for better type support
const StateAnnotation = Annotation.Root({
  messages: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  currentNode: Annotation<string>,
  metadata: Annotation<Record<string, any>>({
    default: () => ({}),
  }),
  iterationCount: Annotation<number>({
    default: () => 0,
  }),
});

// For complex state with nested structures
interface ComplexAgentState {
  // Input data
  query: string;
  context: {
    userId: string;
    sessionId: string;
    createdAt: Date;
  };
  
  // Processing state
  tokenCount: number;
  processingSteps: Array<{
    name: string;
    status: 'pending' | 'running' | 'complete' | 'failed';
    duration: number;
  }>;
  
  // Output
  response: string | null;
  confidence: number;
}
```

### Single-Node Workflows

Sometimes you need just one processing step:

```typescript
const simpleGraph = new StateGraph<AgentState>();

simpleGraph.addNode('process', async (state: AgentState) => {
  return {
    ...state,
    result: `Processed: ${state.input.toUpperCase()}`,
    isComplete: true,
  };
});

simpleGraph.addEdge(START, 'process');
simpleGraph.addEdge('process', END);

const compiled = simpleGraph.compile();
const output = await compiled.invoke({ input: 'hello world' });
```

### Edge Connections

Edges define the control flow between nodes:

```typescript
// Direct unconditional edge
graph.addEdge('analyzeInput', 'selectTool');

// Conditional edge with routing function
graph.addConditionalEdges(
  'selectTool',
  (state: AgentState) => {
    if (state.isComplete) {
      return 'end';
    }
    return 'executeToolAction';
  },
  {
    executeToolAction: 'executeToolAction',
    end: END,
  }
);

// Edge with multiple paths
const routingFunction = (state: AgentState): string => {
  if (state.input.includes('urgent')) {
    return 'priorityHandler';
  } else if (state.input.includes('question')) {
    return 'questionHandler';
  }
  return 'defaultHandler';
};

graph.addConditionalEdges(
  'routeInput',
  routingFunction,
  {
    priorityHandler: 'priorityHandler',
    questionHandler: 'questionHandler',
    defaultHandler: 'defaultHandler',
  }
);
```

### Compilation and Execution

```typescript
// Basic compilation
const compiled = graph.compile();

// Execution with input
const result = await compiled.invoke({
  input: 'Calculate 5 + 3',
  thoughts: [],
  actions: [],
});

console.log('Final result:', result.result);
console.log('Thought process:', result.thoughts);

// Streaming execution
const stream = compiled.stream({
  input: 'Find information about TypeScript',
  thoughts: [],
  actions: [],
});

for await (const step of stream) {
  console.log('Step:', step);
}
```

### MemorySaver for Checkpointing

```typescript
import { MemorySaver } from '@langchain/langgraph';

// Enable in-memory checkpointing
const memory = new MemorySaver();
const compiledWithMemory = graph.compile({
  checkpointer: memory,
});

// Execute with thread for persistent state
const result = await compiledWithMemory.invoke(
  { input: 'hello' },
  { configurable: { thread_id: 'user_123' } }
);

// Resume from checkpoint
const resumed = await compiledWithMemory.invoke(
  { input: 'continue from before' },
  { configurable: { thread_id: 'user_123' } }
);
```

---

## Multi-Agent Systems

Multi-agent systems enable complex task distribution and collaborative problem-solving.

### Multi-Agent Orchestration with LangGraph

```typescript
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';

// Define shared state for all agents
const MultiAgentStateAnnotation = Annotation.Root({
  task: Annotation<string>,
  agentResponses: Annotation<Record<string, string>>({
    default: () => ({}),
    reducer: (x, y) => ({ ...x, ...y }),
  }),
  supervisorDecision: Annotation<string | null>({
    default: () => null,
  }),
  finalResult: Annotation<string | null>({
    default: () => null,
  }),
  messageLog: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

const multiAgentGraph = new StateGraph(MultiAgentStateAnnotation);

// Define individual agent nodes
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0.7,
});

// Research agent
multiAgentGraph.addNode('researchAgent', async (state) => {
  const response = await model.invoke(
    `Research the following topic and provide key findings: ${state.task}`
  );
  
  return {
    agentResponses: {
      research: response.content as string,
    },
    messageLog: [`Research agent completed analysis`],
  };
});

// Analysis agent
multiAgentGraph.addNode('analysisAgent', async (state) => {
  const response = await model.invoke(
    `Analyse the following information: ${state.agentResponses.research}`
  );
  
  return {
    agentResponses: {
      ...state.agentResponses,
      analysis: response.content as string,
    },
    messageLog: [`Analysis agent completed review`],
  };
});

// Writing agent
multiAgentGraph.addNode('writingAgent', async (state) => {
  const response = await model.invoke(
    `Based on this research and analysis, write a comprehensive summary:\nResearch: ${state.agentResponses.research}\nAnalysis: ${state.agentResponses.analysis}`
  );
  
  return {
    agentResponses: {
      ...state.agentResponses,
      writing: response.content as string,
    },
    messageLog: [`Writing agent completed summary`],
  };
});
```

### Supervisor Patterns with Agent Nodes

```typescript
// Supervisor node that coordinates agents
multiAgentGraph.addNode('supervisor', async (state) => {
  const coordinationPrompt = `
    You are a supervisor managing research, analysis, and writing agents.
    Task: ${state.task}
    
    Current status:
    - Research completed: ${'research' in state.agentResponses}
    - Analysis completed: ${'analysis' in state.agentResponses}
    - Writing completed: ${'writing' in state.agentResponses}
    
    Respond with: NEXT_AGENT followed by the agent name (research, analysis, writing, or COMPLETE).
  `;
  
  const response = await model.invoke(coordinationPrompt);
  const nextAgent = response.content as string;
  
  return {
    supervisorDecision: nextAgent,
    messageLog: [`Supervisor routed to: ${nextAgent}`],
  };
});

// Conditional routing based on supervisor decision
multiAgentGraph.addEdge(START, 'supervisor');

multiAgentGraph.addConditionalEdges(
  'supervisor',
  (state) => {
    const decision = state.supervisorDecision || 'END';
    if (decision.includes('research')) {
      return 'researchAgent';
    } else if (decision.includes('analysis')) {
      return 'analysisAgent';
    } else if (decision.includes('writing')) {
      return 'writingAgent';
    }
    return 'END';
  },
  {
    researchAgent: 'researchAgent',
    analysisAgent: 'analysisAgent',
    writingAgent: 'writingAgent',
    END: END,
  }
);

multiAgentGraph.addEdge('researchAgent', 'supervisor');
multiAgentGraph.addEdge('analysisAgent', 'supervisor');
multiAgentGraph.addEdge('writingAgent', 'supervisor');
```

### Agent-to-Agent Communication

```typescript
// Agents communicate through shared state
interface CommunicationState {
  messageQueue: Array<{
    from: string;
    to: string;
    content: string;
    timestamp: number;
  }>;
  agentStates: Record<string, any>;
}

const commGraph = new StateGraph<CommunicationState>();

commGraph.addNode('agent1', async (state) => {
  // Agent 1 processes and sends message to Agent 2
  const message = {
    from: 'agent1',
    to: 'agent2',
    content: 'Here are my findings...',
    timestamp: Date.now(),
  };
  
  return {
    messageQueue: [...state.messageQueue, message],
    agentStates: {
      ...state.agentStates,
      agent1: { status: 'completed', output: 'processed' },
    },
  };
});

commGraph.addNode('agent2', async (state) => {
  // Agent 2 receives message from queue
  const messagesForMe = state.messageQueue.filter((m) => m.to === 'agent2');
  
  // Process messages
  const response = {
    from: 'agent2',
    to: 'agent1',
    content: 'I received and processed your findings',
    timestamp: Date.now(),
  };
  
  return {
    messageQueue: [...state.messageQueue, response],
    agentStates: {
      ...state.agentStates,
      agent2: { status: 'processed', inputCount: messagesForMe.length },
    },
  };
});
```

### Shared State Management

```typescript
// Shared state that persists across agent executions
interface SharedWorkflowState {
  // Global configuration
  config: {
    maxRetries: number;
    timeout: number;
    region: string;
  };
  
  // Shared resources
  resources: {
    vectorStore: any;
    cache: Map<string, any>;
    tokenBudget: number;
  };
  
  // Coordination
  executionPlan: string[];
  currentIndex: number;
  
  // Results accumulation
  results: Record<string, any>;
}

const sharedStateGraph = new StateGraph<SharedWorkflowState>();

sharedStateGraph.addNode('consumeResource', async (state) => {
  // Check and consume shared resource
  const tokensAvailable = state.resources.tokenBudget;
  
  if (tokensAvailable < 100) {
    throw new Error('Insufficient token budget');
  }
  
  return {
    ...state,
    resources: {
      ...state.resources,
      tokenBudget: tokensAvailable - 100,
    },
  };
});
```

### Hierarchical Agent Structures

```typescript
// Parent-child agent hierarchy
interface HierarchicalState {
  level: number; // 0 = top level, 1+ = sub-agents
  parentId: string | null;
  childrenIds: string[];
  result: string;
}

const createAgentAtLevel = (
  level: number,
  parentId: string | null = null
): StateGraph<HierarchicalState> => {
  const graph = new StateGraph<HierarchicalState>();
  
  graph.addNode('process', async (state) => {
    if (state.level === 0) {
      // Top-level coordinator
      return {
        ...state,
        result: 'Coordinating sub-agents',
        childrenIds: ['child1', 'child2'],
      };
    } else {
      // Sub-agent
      return {
        ...state,
        result: `Processed at level ${state.level}`,
      };
    }
  });
  
  return graph;
};
```

---

[Continuing with remaining sections...]

## Tools Integration

### Tool Definition with DynamicStructuredTool

```typescript
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';

// Comprehensive tool with validation, error handling, and metadata
const advancedSearchTool = new DynamicStructuredTool({
  name: 'semantic_search',
  description: 'Search through documents using semantic similarity',
  schema: z.object({
    query: z.string().describe('Search query in natural language'),
    topK: z.number().default(5).describe('Number of top results to return'),
    filters: z.object({
      source: z.string().optional(),
      dateRange: z.object({
        start: z.date().optional(),
        end: z.date().optional(),
      }).optional(),
    }).optional(),
  }),
  func: async (input, runManager) => {
    try {
      // Add logging callback
      await runManager?.handleToolStart(
        { name: 'semantic_search' },
        input.query,
      );
      
      // Simulate semantic search
      const results = Array.from({ length: input.topK }).map((_, i) => ({
        id: `doc_${i}`,
        score: 0.95 - i * 0.1,
        content: `Relevant document ${i + 1}`,
      }));
      
      await runManager?.handleToolEnd(JSON.stringify(results));
      return JSON.stringify(results);
    } catch (error) {
      await runManager?.handleToolError(error);
      throw error;
    }
  },
});
```

### Custom Tool Creation with TypeScript

```typescript
// Custom tool combining multiple operations
class DataProcessingTool extends DynamicStructuredTool {
  private cache: Map<string, any> = new Map();
  
  constructor() {
    super({
      name: 'process_data',
      description: 'Process and transform data with caching',
      schema: z.object({
        operation: z.enum(['sum', 'average', 'transform']),
        data: z.array(z.number()),
        useCache: z.boolean().default(true),
      }),
      func: this.execute.bind(this),
    });
  }
  
  private async execute(input: {
    operation: string;
    data: number[];
    useCache: boolean;
  }): Promise<string> {
    const cacheKey = `${input.operation}_${JSON.stringify(input.data)}`;
    
    if (input.useCache && this.cache.has(cacheKey)) {
      console.log('Using cached result');
      return JSON.stringify(this.cache.get(cacheKey));
    }
    
    let result: number;
    switch (input.operation) {
      case 'sum':
        result = input.data.reduce((a, b) => a + b, 0);
        break;
      case 'average':
        result = input.data.reduce((a, b) => a + b, 0) / input.data.length;
        break;
      case 'transform':
        result = input.data.map((x) => x * 2).reduce((a, b) => a + b, 0);
        break;
      default:
        throw new Error(`Unknown operation: ${input.operation}`);
    }
    
    if (input.useCache) {
      this.cache.set(cacheKey, result);
    }
    
    return JSON.stringify(result);
  }
}

const processingTool = new DataProcessingTool();
```

### Zod Schemas for Validation

```typescript
// Complex, nested Zod schemas
const resourceSchema = z.object({
  id: z.string().uuid().describe('Unique resource identifier'),
  type: z.enum(['compute', 'storage', 'network']),
  specifications: z.object({
    cpu: z.number().positive().optional(),
    memory: z.number().positive().optional(),
    storage: z.number().positive().optional(),
  }),
  metadata: z.record(z.string(), z.any()).optional(),
});

const resourceManagementTool = new DynamicStructuredTool({
  name: 'manage_resources',
  description: 'Manage cloud resources with validation',
  schema: z.object({
    action: z.enum(['create', 'update', 'delete']),
    resource: resourceSchema,
  }),
  func: async ({ action, resource }) => {
    // Tool receives fully validated data
    console.log(`${action.toUpperCase()} resource:`, resource.id);
    return `Successfully ${action}d resource`;
  },
});
```

### Tool Error Handling

```typescript
// Tool with comprehensive error handling
const robustAPITool = tool(
  {
    name: 'call_robust_api',
    description: 'Call external API with error recovery',
    schema: z.object({
      endpoint: z.string().url(),
      timeout: z.number().default(30000),
      retries: z.number().default(3),
    }),
  },
  async ({ endpoint, timeout, retries }) => {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        const response = await fetch(endpoint, {
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          if (attempt === retries) {
            return {
              error: true,
              status: response.status,
              message: `API returned ${response.status} after ${retries} retries`,
            };
          }
          // Retry for 5xx errors
          if (response.status >= 500) {
            await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
            continue;
          }
        }
        
        return {
          error: false,
          data: await response.json(),
        };
      } catch (error) {
        if (attempt === retries) {
          return {
            error: true,
            message: `API call failed after ${retries} retries: ${error}`,
          };
        }
        
        if (error instanceof Error && error.name === 'AbortError') {
          // Retry on timeout
          console.log(`Timeout (${timeout}ms), retrying...`);
          continue;
        }
        
        // For other errors, wait before retrying
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      }
    }
    
    return { error: true, message: 'Exhausted all retries' };
  }
);
```

### Async Tool Execution

```typescript
// Tools that handle long-running operations
const longRunningTool = tool(
  {
    name: 'background_job',
    description: 'Execute long-running background job',
    schema: z.object({
      jobId: z.string(),
      priority: z.enum(['low', 'medium', 'high']).default('medium'),
    }),
  },
  async ({ jobId, priority }, runManager) => {
    try {
      // Report progress
      await runManager?.handleToolStart(
        { name: 'background_job' },
        jobId,
      );
      
      // Simulate async job execution with progress updates
      for (let i = 1; i <= 5; i++) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        console.log(`Job ${jobId} progress: ${i * 20}%`);
      }
      
      await runManager?.handleToolEnd(`Job ${jobId} completed`);
      return `Job completed successfully`;
    } catch (error) {
      await runManager?.handleToolError(error);
      throw error;
    }
  }
);
```

### LangChain Community Tools

```typescript
// Using tools from @langchain/community
import { SerpAPITool } from '@langchain/community/tools/google_serper';
import { Calculator } from '@langchain/community/tools/calculator';

// SerpAPI for web search
const searchTool = new SerpAPITool({
  apiKey: process.env.SERP_API_KEY,
});

// Calculator tool
const calculatorTool = new Calculator();

// Integrate into agent
const agent = createReactAgent({
  llm: model,
  tools: [searchTool, calculatorTool],
});
```

### Integration with External APIs

```typescript
// Complex integration with multiple external services
class IntegratedToolSuite {
  private tools: Map<string, DynamicStructuredTool> = new Map();
  
  constructor(
    private apiKeys: Record<string, string>,
  ) {
    this.initializeTools();
  }
  
  private initializeTools(): void {
    // GitHub API tool
    this.tools.set(
      'github',
      new DynamicStructuredTool({
        name: 'github_api',
        description: 'Query GitHub repositories and commits',
        schema: z.object({
          action: z.enum(['search', 'commits', 'issues']),
          owner: z.string().optional(),
          repo: z.string().optional(),
          query: z.string().optional(),
        }),
        func: async (input) => {
          const headers = {
            Authorization: `Bearer ${this.apiKeys.github}`,
          };
          
          let endpoint = 'https://api.github.com/search/repositories';
          if (input.action === 'commits') {
            endpoint = `https://api.github.com/repos/${input.owner}/${input.repo}/commits`;
          }
          
          const response = await fetch(endpoint, { headers });
          return await response.json();
        },
      })
    );
    
    // Stripe API tool
    this.tools.set(
      'stripe',
      new DynamicStructuredTool({
        name: 'stripe_api',
        description: 'Query Stripe payment information',
        schema: z.object({
          action: z.enum(['balance', 'charges', 'customers']),
          limit: z.number().optional(),
        }),
        func: async (input) => {
          const headers = {
            Authorization: `Bearer ${this.apiKeys.stripe}`,
          };
          
          const endpoint = `https://api.stripe.com/v1/${input.action}`;
          const params = new URLSearchParams();
          if (input.limit) {
            params.append('limit', input.limit.toString());
          }
          
          const response = await fetch(`${endpoint}?${params}`, {
            headers,
          });
          return await response.json();
        },
      })
    );
  }
  
  getTools(): DynamicStructuredTool[] {
    return Array.from(this.tools.values());
  }
}
```

## Structured Output

### JsonOutputParser and StructuredOutputParser

```typescript
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { z } from 'zod';

// JSON output parser with schema
const jsonParser = new JsonOutputParser<Record<string, any>>();

// Define expected output structure
const analysisSchema = z.object({
  summary: z.string(),
  keyPoints: z.array(z.string()),
  sentiment: z.enum(['positive', 'neutral', 'negative']),
  confidence: z.number().min(0).max(1),
});

type AnalysisOutput = z.infer<typeof analysisSchema>;

// Use parser with model
const chain = ChatPromptTemplate.fromTemplate(
  'Analyse this text: {text}'
)
  .pipe(model)
  .pipe(jsonParser);

const result = await chain.invoke({ text: 'Your text here' });
console.log(result); // Fully typed as Record<string, any>
```

### Zod Schemas for Type Safety

```typescript
// Comprehensive schema with validation
const articleSchema = z.object({
  title: z.string().min(5).max(200),
  authors: z.array(z.object({
    name: z.string(),
    email: z.string().email(),
    affiliation: z.string().optional(),
  })),
  publishedDate: z.date(),
  content: z.string().min(100),
  tags: z.array(z.string()).min(1).max(10),
  metrics: z.object({
    views: z.number().int().nonnegative(),
    likes: z.number().int().nonnegative(),
    shares: z.number().int().nonnegative(),
  }),
  metadata: z.record(z.string(), z.any()).optional(),
});

type Article = z.infer<typeof articleSchema>;

// Parser that validates against schema
const articleParser = new JsonOutputParser<Article>();
```

### withStructuredOutput() Method

```typescript
// Modern approach using withStructuredOutput
const extractionModel = model.withStructuredOutput(
  z.object({
    entities: z.array(z.object({
      text: z.string(),
      type: z.enum(['PERSON', 'ORG', 'LOCATION']),
    })),
    sentiment: z.enum(['positive', 'neutral', 'negative']),
    keywords: z.array(z.string()),
  })
);

const extraction = await extractionModel.invoke(
  'Text to analyse for entities and sentiment'
);

// Result is fully typed
console.log(extraction.entities); // Fully typed as Entity[]
console.log(extraction.sentiment); // Fully typed as 'positive' | 'neutral' | 'negative'
```

### Output Validation Strategies

```typescript
// Multi-layer validation
async function validateAndRefine<T>(
  output: unknown,
  schema: z.ZodSchema<T>,
  model: ChatOpenAI,
  maxRetries: number = 3,
): Promise<T> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // First validation attempt
      return schema.parse(output);
    } catch (error) {
      if (attempt === maxRetries) {
        throw new Error(`Validation failed after ${maxRetries} attempts`);
      }
      
      // Ask model to correct output
      const correctionPrompt = `
        The following output failed validation:
        ${JSON.stringify(output)}
        
        Error: ${error}
        
        Schema requirements:
        ${JSON.stringify(schema)}
        
        Please provide corrected output that matches the schema.
      `;
      
      const correctedOutput = await model.invoke(correctionPrompt);
      output = JSON.parse(correctedOutput.content as string);
    }
  }
  
  throw new Error('Validation validation exhausted');
}
```

### Complex Nested Structures

```typescript
// Deeply nested type-safe structures
const complexDataSchema = z.object({
  project: z.object({
    id: z.string().uuid(),
    name: z.string(),
    organisation: z.object({
      id: z.string(),
      name: z.string(),
      metadata: z.record(z.string(), z.any()),
    }),
    team: z.array(z.object({
      memberId: z.string(),
      role: z.enum(['lead', 'developer', 'tester']),
      contribution: z.object({
        commits: z.number(),
        linesOfCode: z.number(),
        reviewsCompleted: z.number(),
      }),
    })),
    timelines: z.object({
      started: z.date(),
      deadline: z.date(),
      phases: z.array(z.object({
        name: z.string(),
        startDate: z.date(),
        endDate: z.date(),
        status: z.enum(['pending', 'in_progress', 'completed']),
      })),
    }),
  }),
});

type ComplexProject = z.infer<typeof complexDataSchema>;
```

---

## Type-Safe .stream() Method (v0.3+)

### Overview

LangGraph v0.3+ introduces a powerful type-safe streaming API that provides real-time state updates and values during graph execution. The `.stream()` method returns different types of information based on the `streamMode` parameter, enabling developers to observe and respond to workflow execution in real-time.

### Stream Modes

```typescript
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';

// Define state with proper typing
const WorkflowStateAnnotation = Annotation.Root({
  input: Annotation<string>,
  steps: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  output: Annotation<string | null>({
    default: () => null,
  }),
  metadata: Annotation<Record<string, any>>({
    default: () => ({}),
  }),
});

type WorkflowState = typeof WorkflowStateAnnotation.State;

const graph = new StateGraph(WorkflowStateAnnotation);

// Add processing nodes
graph.addNode('preprocess', async (state: WorkflowState) => {
  await new Promise((resolve) => setTimeout(resolve, 100));
  return {
    steps: ['Preprocessing completed'],
    metadata: { preprocessed: true },
  };
});

graph.addNode('analyze', async (state: WorkflowState) => {
  await new Promise((resolve) => setTimeout(resolve, 100));
  return {
    steps: ['Analysis completed'],
    metadata: { ...state.metadata, analyzed: true },
  };
});

graph.addNode('generate', async (state: WorkflowState) => {
  await new Promise((resolve) => setTimeout(resolve, 100));
  return {
    output: `Processed: ${state.input}`,
    steps: ['Generation completed'],
  };
});

graph.addEdge(START, 'preprocess');
graph.addEdge('preprocess', 'analyze');
graph.addEdge('analyze', 'generate');
graph.addEdge('generate', END);

const workflow = graph.compile();
```

### StreamMode: "values" - Complete State Updates

Returns the complete state after each node execution, ideal for tracking full workflow state.

```typescript
async function streamWithValues(): Promise<void> {
  const stream = workflow.stream(
    {
      input: 'Hello world',
      steps: [],
      output: null,
      metadata: {},
    },
    {
      streamMode: 'values',
    }
  );

  console.log('=== Streaming with VALUES mode ===');

  for await (const state of stream) {
    // state contains the complete updated state after each node
    console.log('Current state:', {
      steps: state.steps,
      output: state.output,
      metadata: state.metadata,
    });
  }
}

// Example output:
// Current state: { steps: ['Preprocessing completed'], output: null, metadata: { preprocessed: true } }
// Current state: { steps: ['Preprocessing completed', 'Analysis completed'], output: null, metadata: { preprocessed: true, analyzed: true } }
// Current state: { steps: ['Preprocessing completed', 'Analysis completed', 'Generation completed'], output: 'Processed: Hello world', metadata: { preprocessed: true, analyzed: true } }
```

### StreamMode: "updates" - Incremental Updates Only

Returns only the updates made by each node, perfect for tracking changes without full state overhead.

```typescript
async function streamWithUpdates(): Promise<void> {
  const stream = workflow.stream(
    {
      input: 'Hello world',
      steps: [],
      output: null,
      metadata: {},
    },
    {
      streamMode: 'updates',
    }
  );

  console.log('=== Streaming with UPDATES mode ===');

  for await (const update of stream) {
    // update contains only the changes made by the current node
    console.log('Node update:', update);
  }
}

// Example output:
// Node update: { preprocess: { steps: ['Preprocessing completed'], metadata: { preprocessed: true } } }
// Node update: { analyze: { steps: ['Analysis completed'], metadata: { analyzed: true } } }
// Node update: { generate: { output: 'Processed: Hello world', steps: ['Generation completed'] } }
```

### StreamMode: "debug" - Complete Execution Information

Returns detailed execution metadata including node names, timestamps, and state transitions.

```typescript
async function streamWithDebug(): Promise<void> {
  const stream = workflow.stream(
    {
      input: 'Hello world',
      steps: [],
      output: null,
      metadata: {},
    },
    {
      streamMode: 'debug',
    }
  );

  console.log('=== Streaming with DEBUG mode ===');

  for await (const debugInfo of stream) {
    // debugInfo contains execution metadata
    console.log('Debug info:', {
      type: debugInfo.type,
      node: debugInfo.node,
      state: debugInfo.state,
      timestamp: new Date().toISOString(),
    });
  }
}

// Example output:
// Debug info: { type: 'task', node: 'preprocess', state: {...}, timestamp: '2025-01-15T10:30:00.000Z' }
// Debug info: { type: 'task', node: 'analyze', state: {...}, timestamp: '2025-01-15T10:30:00.100Z' }
// Debug info: { type: 'task', node: 'generate', state: {...}, timestamp: '2025-01-15T10:30:00.200Z' }
```

### StreamMode: "messages" - Message-Focused Streaming

Specifically designed for conversational workflows, returns only message updates.

```typescript
import { BaseMessage, HumanMessage, AIMessage } from '@langchain/core/messages';

const ChatStateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

const chatGraph = new StateGraph(ChatStateAnnotation);

chatGraph.addNode('chatbot', async (state) => {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const response = await model.invoke(state.messages);

  return {
    messages: [response],
  };
});

chatGraph.addEdge(START, 'chatbot');
chatGraph.addEdge('chatbot', END);

const chatWorkflow = chatGraph.compile();

async function streamWithMessages(): Promise<void> {
  const stream = chatWorkflow.stream(
    {
      messages: [new HumanMessage('What is TypeScript?')],
    },
    {
      streamMode: 'messages',
    }
  );

  console.log('=== Streaming with MESSAGES mode ===');

  for await (const message of stream) {
    // message contains only message updates
    if (message instanceof AIMessage) {
      console.log('AI Response:', message.content);
    }
  }
}
```

### Advanced Streaming: Multiple Modes

You can stream using multiple modes simultaneously for comprehensive monitoring.

```typescript
async function streamWithMultipleModes(): Promise<void> {
  const stream = workflow.stream(
    {
      input: 'Complex workflow',
      steps: [],
      output: null,
      metadata: {},
    },
    {
      streamMode: ['values', 'updates', 'debug'],
    }
  );

  console.log('=== Streaming with MULTIPLE modes ===');

  for await (const chunk of stream) {
    // chunk contains data from all requested modes
    console.log('Stream chunk:', {
      hasValues: 'values' in chunk,
      hasUpdates: 'updates' in chunk,
      hasDebug: 'debug' in chunk,
    });

    if ('values' in chunk) {
      console.log('Complete state:', chunk.values);
    }

    if ('updates' in chunk) {
      console.log('Node updates:', chunk.updates);
    }

    if ('debug' in chunk) {
      console.log('Debug info:', chunk.debug);
    }
  }
}
```

### Real-Time Progress Tracking

Practical example of using streaming for progress monitoring in UI applications.

```typescript
interface ProgressUpdate {
  step: string;
  progress: number;
  total: number;
  message: string;
}

async function* streamWithProgress(
  input: string
): AsyncGenerator<ProgressUpdate, void, unknown> {
  const stream = workflow.stream(
    { input, steps: [], output: null, metadata: {} },
    { streamMode: 'updates' }
  );

  let completed = 0;
  const total = 3; // Total number of nodes

  for await (const update of stream) {
    const nodeName = Object.keys(update)[0];
    completed++;

    yield {
      step: nodeName,
      progress: completed,
      total,
      message: `Completed ${nodeName}: ${Math.round((completed / total) * 100)}%`,
    };
  }
}

// Usage in Express.js API
async function handleProgressStream(req: any, res: any): Promise<void> {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  try {
    for await (const progress of streamWithProgress(req.body.input)) {
      res.write(`data: ${JSON.stringify(progress)}\n\n`);
    }

    res.write('data: {"done": true}\n\n');
    res.end();
  } catch (error) {
    res.write(`data: {"error": "${error}"}\n\n`);
    res.end();
  }
}
```

### Type-Safe Streaming with Custom State

Demonstrates full type safety throughout the streaming process.

```typescript
interface CustomWorkflowState {
  userId: string;
  requestId: string;
  processingStages: Array<{
    stage: string;
    status: 'pending' | 'processing' | 'complete' | 'failed';
    timestamp: number;
  }>;
  results: Record<string, any>;
  errors: string[];
}

const CustomStateAnnotation = Annotation.Root({
  userId: Annotation<string>,
  requestId: Annotation<string>,
  processingStages: Annotation<CustomWorkflowState['processingStages']>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  results: Annotation<Record<string, any>>({
    default: () => ({}),
    reducer: (x, y) => ({ ...x, ...y }),
  }),
  errors: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

const customGraph = new StateGraph(CustomStateAnnotation);

customGraph.addNode('validate', async (state) => {
  return {
    processingStages: [{
      stage: 'validation',
      status: 'complete' as const,
      timestamp: Date.now(),
    }],
  };
});

customGraph.addNode('process', async (state) => {
  return {
    processingStages: [{
      stage: 'processing',
      status: 'complete' as const,
      timestamp: Date.now(),
    }],
    results: {
      processedData: 'Processed successfully',
    },
  };
});

customGraph.addEdge(START, 'validate');
customGraph.addEdge('validate', 'process');
customGraph.addEdge('process', END);

const customWorkflow = customGraph.compile();

async function streamCustomWorkflow(
  userId: string,
  requestId: string
): Promise<void> {
  const stream = customWorkflow.stream(
    {
      userId,
      requestId,
      processingStages: [],
      results: {},
      errors: [],
    },
    {
      streamMode: 'values',
    }
  );

  for await (const state of stream) {
    // TypeScript knows the exact shape of state
    console.log('User:', state.userId);
    console.log('Request:', state.requestId);
    console.log('Stages:', state.processingStages.map((s) => s.stage));
    console.log('Results:', state.results);
  }
}
```

### Error Handling in Streams

Robust error handling patterns for streaming workflows.

```typescript
async function streamWithErrorHandling(): Promise<void> {
  try {
    const stream = workflow.stream(
      {
        input: 'Test input',
        steps: [],
        output: null,
        metadata: {},
      },
      {
        streamMode: 'updates',
      }
    );

    for await (const update of stream) {
      try {
        // Process update
        console.log('Processing update:', update);

        // Check for error conditions in state
        const nodeName = Object.keys(update)[0];
        const nodeUpdate = update[nodeName];

        if (nodeUpdate.error) {
          throw new Error(`Node ${nodeName} failed: ${nodeUpdate.error}`);
        }
      } catch (nodeError) {
        console.error('Error processing node update:', nodeError);
        // Continue to next update or break based on error severity
      }
    }
  } catch (streamError) {
    console.error('Fatal stream error:', streamError);
    // Handle complete stream failure
  }
}
```

---

## .addNode() and .addSequence() Methods

### Overview

LangGraph v1.0+ introduces convenient builder methods that significantly reduce boilerplate when constructing workflows. The `.addNode()` and `.addSequence()` methods provide ergonomic APIs for defining graph structures with less code while maintaining full type safety.

### .addNode() Method

The `.addNode()` method simplifies node creation with support for inline function definitions and automatic edge management.

```typescript
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';

interface SimpleState {
  input: string;
  count: number;
  messages: string[];
}

const StateAnnotation = Annotation.Root({
  input: Annotation<string>,
  count: Annotation<number>({ default: () => 0 }),
  messages: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

const graph = new StateGraph(StateAnnotation);

// Traditional approach (verbose)
function processNodeOld(state: SimpleState): Partial<SimpleState> {
  return {
    count: state.count + 1,
    messages: [`Processed: ${state.input}`],
  };
}
graph.addNode('process', processNodeOld);

// New .addNode() approach (concise)
graph
  .addNode('process', (state) => ({
    count: state.count + 1,
    messages: [`Processed: ${state.input}`],
  }))
  .addNode('validate', (state) => ({
    messages: state.count > 0 ? ['Valid'] : ['Invalid'],
  }))
  .addNode('finalize', (state) => ({
    messages: [`Final count: ${state.count}`],
  }));
```

### Chaining .addNode() Calls

The builder pattern allows chaining multiple node definitions for cleaner code.

```typescript
const DataProcessingStateAnnotation = Annotation.Root({
  rawData: Annotation<string>,
  cleanedData: Annotation<string | null>({ default: () => null }),
  validatedData: Annotation<string | null>({ default: () => null }),
  transformedData: Annotation<string | null>({ default: () => null }),
  errors: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

const dataGraph = new StateGraph(DataProcessingStateAnnotation);

dataGraph
  .addNode('clean', (state) => {
    const cleaned = state.rawData.trim().toLowerCase();
    return {
      cleanedData: cleaned,
    };
  })
  .addNode('validate', (state) => {
    const isValid = state.cleanedData && state.cleanedData.length > 0;

    if (!isValid) {
      return {
        errors: ['Validation failed: empty data'],
      };
    }

    return {
      validatedData: state.cleanedData,
    };
  })
  .addNode('transform', (state) => {
    if (!state.validatedData) {
      return {
        errors: ['Cannot transform invalid data'],
      };
    }

    return {
      transformedData: state.validatedData.toUpperCase(),
    };
  })
  .addEdge(START, 'clean')
  .addEdge('clean', 'validate')
  .addEdge('validate', 'transform')
  .addEdge('transform', END);

const dataWorkflow = dataGraph.compile();
```

### .addSequence() Method

The `.addSequence()` method creates a linear sequence of nodes with automatic edge connections, perfect for pipelines.

```typescript
const PipelineStateAnnotation = Annotation.Root({
  input: Annotation<string>,
  stage1Result: Annotation<string | null>({ default: () => null }),
  stage2Result: Annotation<string | null>({ default: () => null }),
  stage3Result: Annotation<string | null>({ default: () => null }),
  finalOutput: Annotation<string | null>({ default: () => null }),
});

const pipelineGraph = new StateGraph(PipelineStateAnnotation);

// Traditional approach: manually adding nodes and edges
pipelineGraph.addNode('stage1', (state) => ({
  stage1Result: `Stage1: ${state.input}`,
}));
pipelineGraph.addNode('stage2', (state) => ({
  stage2Result: `Stage2: ${state.stage1Result}`,
}));
pipelineGraph.addNode('stage3', (state) => ({
  stage3Result: `Stage3: ${state.stage2Result}`,
}));
pipelineGraph.addEdge(START, 'stage1');
pipelineGraph.addEdge('stage1', 'stage2');
pipelineGraph.addEdge('stage2', 'stage3');
pipelineGraph.addEdge('stage3', END);

// New .addSequence() approach (much cleaner)
const sequenceGraph = new StateGraph(PipelineStateAnnotation);

sequenceGraph.addSequence([
  {
    name: 'stage1',
    handler: (state) => ({
      stage1Result: `Stage1: ${state.input}`,
    }),
  },
  {
    name: 'stage2',
    handler: (state) => ({
      stage2Result: `Stage2: ${state.stage1Result}`,
    }),
  },
  {
    name: 'stage3',
    handler: (state) => ({
      stage3Result: `Stage3: ${state.stage2Result}`,
    }),
  },
  {
    name: 'finalize',
    handler: (state) => ({
      finalOutput: state.stage3Result,
    }),
  },
]);

const sequenceWorkflow = sequenceGraph.compile();
```

### Complex Pipeline with .addSequence()

Real-world example of a document processing pipeline using .addSequence().

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { Document } from '@langchain/core/documents';

interface DocumentProcessingState {
  document: string;
  chunks: string[];
  summaries: string[];
  keywords: string[];
  finalReport: string | null;
}

const DocStateAnnotation = Annotation.Root({
  document: Annotation<string>,
  chunks: Annotation<string[]>({ default: () => [] }),
  summaries: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  keywords: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  finalReport: Annotation<string | null>({ default: () => null }),
});

const docGraph = new StateGraph(DocStateAnnotation);

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0.3,
});

docGraph.addSequence([
  {
    name: 'chunk',
    handler: async (state) => {
      // Split document into chunks
      const chunkSize = 1000;
      const chunks: string[] = [];

      for (let i = 0; i < state.document.length; i += chunkSize) {
        chunks.push(state.document.slice(i, i + chunkSize));
      }

      return { chunks };
    },
  },
  {
    name: 'summarize',
    handler: async (state) => {
      // Summarize each chunk
      const summaries: string[] = [];

      for (const chunk of state.chunks) {
        const response = await model.invoke(
          `Summarize this text in one sentence: ${chunk}`
        );
        summaries.push(response.content as string);
      }

      return { summaries };
    },
  },
  {
    name: 'extractKeywords',
    handler: async (state) => {
      // Extract keywords from summaries
      const response = await model.invoke(
        `Extract 5 keywords from these summaries: ${state.summaries.join(' ')}`
      );

      const keywords = (response.content as string)
        .split(',')
        .map((k) => k.trim());

      return { keywords };
    },
  },
  {
    name: 'generateReport',
    handler: async (state) => {
      const report = `
Document Analysis Report
========================

Chunks Processed: ${state.chunks.length}
Summaries Generated: ${state.summaries.length}

Key Findings:
${state.summaries.map((s, i) => `${i + 1}. ${s}`).join('\n')}

Keywords: ${state.keywords.join(', ')}
      `.trim();

      return { finalReport: report };
    },
  },
]);

const docWorkflow = docGraph.compile();

// Usage
async function processDocument(document: string): Promise<string> {
  const result = await docWorkflow.invoke({
    document,
    chunks: [],
    summaries: [],
    keywords: [],
    finalReport: null,
  });

  return result.finalReport || '';
}
```

### Mixing .addNode() and .addSequence()

You can combine both methods for flexible workflow construction.

```typescript
const HybridStateAnnotation = Annotation.Root({
  input: Annotation<string>,
  preprocessed: Annotation<string | null>({ default: () => null }),
  analyzed: Annotation<string | null>({ default: () => null }),
  enhanced: Annotation<string | null>({ default: () => null }),
  validated: Annotation<string | null>({ default: () => null }),
  output: Annotation<string | null>({ default: () => null }),
});

const hybridGraph = new StateGraph(HybridStateAnnotation);

// Add individual nodes with custom logic
hybridGraph
  .addNode('preprocess', async (state) => {
    // Complex preprocessing
    const preprocessed = state.input.trim().toLowerCase();
    return { preprocessed };
  })
  .addNode('validate', async (state) => {
    // Validation logic
    if (!state.enhanced || state.enhanced.length === 0) {
      throw new Error('Invalid data');
    }
    return { validated: state.enhanced };
  });

// Add a sequence of simple transformations
hybridGraph.addSequence([
  {
    name: 'analyze',
    handler: async (state) => ({
      analyzed: `Analyzed: ${state.preprocessed}`,
    }),
  },
  {
    name: 'enhance',
    handler: async (state) => ({
      enhanced: `Enhanced: ${state.analyzed}`,
    }),
  },
]);

// Connect everything
hybridGraph.addEdge(START, 'preprocess');
hybridGraph.addEdge('preprocess', 'analyze'); // Start of sequence
hybridGraph.addEdge('enhance', 'validate'); // End of sequence to validate
hybridGraph.addEdge('validate', END);

const hybridWorkflow = hybridGraph.compile();
```

### Conditional Sequences

Using .addSequence() with conditional routing.

```typescript
const ConditionalStateAnnotation = Annotation.Root({
  inputType: Annotation<'text' | 'data' | 'image'>,
  input: Annotation<string>,
  processed: Annotation<string | null>({ default: () => null }),
  result: Annotation<string | null>({ default: () => null }),
});

const conditionalGraph = new StateGraph(ConditionalStateAnnotation);

// Define separate sequences for different input types
const textSequence = [
  {
    name: 'processText',
    handler: async (state: any) => ({
      processed: `Text processed: ${state.input}`,
    }),
  },
  {
    name: 'finalizeText',
    handler: async (state: any) => ({
      result: `Text result: ${state.processed}`,
    }),
  },
];

const dataSequence = [
  {
    name: 'processData',
    handler: async (state: any) => ({
      processed: `Data processed: ${state.input}`,
    }),
  },
  {
    name: 'finalizeData',
    handler: async (state: any) => ({
      result: `Data result: ${state.processed}`,
    }),
  },
];

// Add routing node
conditionalGraph.addNode('route', (state) => state);

// Add sequences
conditionalGraph.addSequence(textSequence);
conditionalGraph.addSequence(dataSequence);

// Setup conditional edges
conditionalGraph.addEdge(START, 'route');

conditionalGraph.addConditionalEdges(
  'route',
  (state) => {
    if (state.inputType === 'text') {
      return 'processText';
    } else {
      return 'processData';
    }
  },
  {
    processText: 'processText',
    processData: 'processData',
  }
);

conditionalGraph.addEdge('finalizeText', END);
conditionalGraph.addEdge('finalizeData', END);

const conditionalWorkflow = conditionalGraph.compile();
```

---

## Node Caching

### Overview

LangGraph v1.0+ introduces node caching to skip redundant computations when the same node is invoked with identical inputs. This feature dramatically improves performance for workflows with repeated operations or deterministic transformations.

### Basic Node Caching

Enable caching at the node level to automatically cache results.

```typescript
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';

interface CachedState {
  input: string;
  expensiveResult: string | null;
  computeCount: number;
}

const CachedStateAnnotation = Annotation.Root({
  input: Annotation<string>,
  expensiveResult: Annotation<string | null>({ default: () => null }),
  computeCount: Annotation<number>({ default: () => 0 }),
});

const graph = new StateGraph(CachedStateAnnotation);

// Expensive computation that benefits from caching
graph.addNode(
  'expensiveComputation',
  async (state) => {
    console.log('Performing expensive computation...');

    // Simulate expensive operation
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const result = `Computed: ${state.input.toUpperCase()}`;

    return {
      expensiveResult: result,
      computeCount: state.computeCount + 1,
    };
  },
  {
    // Enable caching for this node
    cache: true,
  }
);

graph.addEdge(START, 'expensiveComputation');
graph.addEdge('expensiveComputation', END);

const workflow = graph.compile();

// First invocation - performs computation
const result1 = await workflow.invoke({
  input: 'test',
  expensiveResult: null,
  computeCount: 0,
});
console.log('First result:', result1.expensiveResult); // Takes 2 seconds
console.log('Compute count:', result1.computeCount); // 1

// Second invocation with same input - uses cache
const result2 = await workflow.invoke({
  input: 'test',
  expensiveResult: null,
  computeCount: 0,
});
console.log('Second result:', result2.expensiveResult); // Instant
console.log('Compute count:', result2.computeCount); // 0 (cached)
```

### Custom Cache Key Functions

Define custom cache key generation for fine-grained control.

```typescript
interface DataState {
  userId: string;
  timestamp: number;
  data: Record<string, any>;
  processedData: Record<string, any> | null;
}

const DataStateAnnotation = Annotation.Root({
  userId: Annotation<string>,
  timestamp: Annotation<number>,
  data: Annotation<Record<string, any>>,
  processedData: Annotation<Record<string, any> | null>({ default: () => null }),
});

const dataGraph = new StateGraph(DataStateAnnotation);

dataGraph.addNode(
  'processData',
  async (state) => {
    console.log('Processing data for user:', state.userId);

    // Complex data transformation
    const processed = Object.entries(state.data).reduce(
      (acc, [key, value]) => ({
        ...acc,
        [key]: typeof value === 'string' ? value.toUpperCase() : value,
      }),
      {}
    );

    return {
      processedData: processed,
    };
  },
  {
    cache: true,
    // Custom cache key - only userId and data matter, not timestamp
    cacheKey: (state: DataState) => {
      return `${state.userId}:${JSON.stringify(state.data)}`;
    },
  }
);

dataGraph.addEdge(START, 'processData');
dataGraph.addEdge('processData', END);

const dataWorkflow = dataGraph.compile();

// These will use the same cache despite different timestamps
await dataWorkflow.invoke({
  userId: 'user123',
  timestamp: Date.now(),
  data: { name: 'john', age: 30 },
  processedData: null,
});

await dataWorkflow.invoke({
  userId: 'user123',
  timestamp: Date.now() + 1000, // Different timestamp
  data: { name: 'john', age: 30 }, // Same data
  processedData: null,
}); // Uses cache
```

### Cache TTL (Time-To-Live)

Set expiration times for cached results.

```typescript
interface ApiState {
  endpoint: string;
  response: string | null;
  cachedAt: number | null;
}

const ApiStateAnnotation = Annotation.Root({
  endpoint: Annotation<string>,
  response: Annotation<string | null>({ default: () => null }),
  cachedAt: Annotation<number | null>({ default: () => null }),
});

const apiGraph = new StateGraph(ApiStateAnnotation);

apiGraph.addNode(
  'fetchApi',
  async (state) => {
    console.log('Fetching from API:', state.endpoint);

    // Simulate API call
    const response = await fetch(state.endpoint);
    const data = await response.text();

    return {
      response: data,
      cachedAt: Date.now(),
    };
  },
  {
    cache: true,
    // Cache expires after 5 minutes
    cacheTTL: 5 * 60 * 1000, // 5 minutes in milliseconds
  }
);

apiGraph.addEdge(START, 'fetchApi');
apiGraph.addEdge('fetchApi', END);

const apiWorkflow = apiGraph.compile();

// First call - fetches from API
await apiWorkflow.invoke({
  endpoint: 'https://api.example.com/data',
  response: null,
  cachedAt: null,
});

// Call within 5 minutes - uses cache
await apiWorkflow.invoke({
  endpoint: 'https://api.example.com/data',
  response: null,
  cachedAt: null,
});

// Wait 5 minutes...
await new Promise((resolve) => setTimeout(resolve, 5 * 60 * 1000));

// Call after TTL expires - fetches again
await apiWorkflow.invoke({
  endpoint: 'https://api.example.com/data',
  response: null,
  cachedAt: null,
});
```

### Distributed Caching with Redis

Implement distributed caching for multi-instance deployments.

```typescript
import Redis from 'ioredis';

interface CacheConfig {
  redis: Redis;
  keyPrefix: string;
  defaultTTL: number;
}

class RedisCacheManager {
  constructor(private config: CacheConfig) {}

  private generateKey(prefix: string, ...parts: string[]): string {
    return `${this.config.keyPrefix}:${prefix}:${parts.join(':')}`;
  }

  async get<T>(cacheKey: string): Promise<T | null> {
    const value = await this.config.redis.get(cacheKey);
    return value ? JSON.parse(value) : null;
  }

  async set<T>(cacheKey: string, value: T, ttl?: number): Promise<void> {
    const expiry = ttl || this.config.defaultTTL;
    await this.config.redis.setex(
      cacheKey,
      Math.floor(expiry / 1000),
      JSON.stringify(value)
    );
  }

  async invalidate(pattern: string): Promise<void> {
    const keys = await this.config.redis.keys(pattern);
    if (keys.length > 0) {
      await this.config.redis.del(...keys);
    }
  }
}

// Node with Redis caching
interface MLState {
  inputText: string;
  embedding: number[] | null;
}

const MLStateAnnotation = Annotation.Root({
  inputText: Annotation<string>,
  embedding: Annotation<number[] | null>({ default: () => null }),
});

const mlGraph = new StateGraph(MLStateAnnotation);

const cacheManager = new RedisCacheManager({
  redis: new Redis(process.env.REDIS_URL),
  keyPrefix: 'langgraph',
  defaultTTL: 3600 * 1000, // 1 hour
});

mlGraph.addNode(
  'generateEmbedding',
  async (state) => {
    const cacheKey = `embedding:${state.inputText}`;

    // Check cache first
    const cached = await cacheManager.get<number[]>(cacheKey);
    if (cached) {
      console.log('Using cached embedding');
      return { embedding: cached };
    }

    console.log('Generating new embedding');

    // Generate embedding (expensive operation)
    const embedding = await generateEmbedding(state.inputText);

    // Cache the result
    await cacheManager.set(cacheKey, embedding);

    return { embedding };
  }
);

async function generateEmbedding(text: string): Promise<number[]> {
  // Simulate expensive embedding generation
  await new Promise((resolve) => setTimeout(resolve, 1000));
  return Array.from({ length: 1536 }, () => Math.random());
}

mlGraph.addEdge(START, 'generateEmbedding');
mlGraph.addEdge('generateEmbedding', END);

const mlWorkflow = mlGraph.compile();
```

### Selective Cache Invalidation

Invalidate cache entries based on specific conditions.

```typescript
interface UserDataState {
  userId: string;
  operation: 'read' | 'write' | 'delete';
  data: Record<string, any> | null;
  result: string | null;
}

const UserDataStateAnnotation = Annotation.Root({
  userId: Annotation<string>,
  operation: Annotation<'read' | 'write' | 'delete'>,
  data: Annotation<Record<string, any> | null>({ default: () => null }),
  result: Annotation<string | null>({ default: () => null }),
});

const userGraph = new StateGraph(UserDataStateAnnotation);

class CacheInvalidationManager {
  constructor(private cache: RedisCacheManager) {}

  async invalidateUserCache(userId: string): Promise<void> {
    // Invalidate all cache entries for this user
    await this.cache.invalidate(`*:user:${userId}:*`);
    console.log(`Invalidated cache for user: ${userId}`);
  }

  async invalidatePattern(pattern: string): Promise<void> {
    await this.cache.invalidate(pattern);
  }
}

const invalidationManager = new CacheInvalidationManager(cacheManager);

userGraph
  .addNode('processOperation', async (state) => {
    const cacheKey = `user:${state.userId}:data`;

    if (state.operation === 'read') {
      // Try to read from cache
      const cached = await cacheManager.get<Record<string, any>>(cacheKey);
      if (cached) {
        return { result: 'Data retrieved from cache', data: cached };
      }

      // Fetch from database
      const data = await fetchUserData(state.userId);
      await cacheManager.set(cacheKey, data);

      return { result: 'Data retrieved from database', data };
    } else if (state.operation === 'write') {
      // Write to database
      await writeUserData(state.userId, state.data!);

      // Invalidate cache
      await invalidationManager.invalidateUserCache(state.userId);

      return { result: 'Data written, cache invalidated' };
    } else if (state.operation === 'delete') {
      // Delete from database
      await deleteUserData(state.userId);

      // Invalidate cache
      await invalidationManager.invalidateUserCache(state.userId);

      return { result: 'Data deleted, cache invalidated' };
    }

    return { result: 'Unknown operation' };
  });

async function fetchUserData(userId: string): Promise<Record<string, any>> {
  // Simulate database fetch
  return { userId, name: 'John Doe', email: 'john@example.com' };
}

async function writeUserData(
  userId: string,
  data: Record<string, any>
): Promise<void> {
  // Simulate database write
  console.log('Writing to database:', userId, data);
}

async function deleteUserData(userId: string): Promise<void> {
  // Simulate database delete
  console.log('Deleting from database:', userId);
}

userGraph.addEdge(START, 'processOperation');
userGraph.addEdge('processOperation', END);

const userWorkflow = userGraph.compile();
```

### Cache Performance Monitoring

Track cache hit/miss ratios and performance metrics.

```typescript
interface CacheMetrics {
  hits: number;
  misses: number;
  totalRequests: number;
  avgComputeTime: number;
  avgCacheTime: number;
}

class CacheMonitor {
  private metrics: CacheMetrics = {
    hits: 0,
    misses: 0,
    totalRequests: 0,
    avgComputeTime: 0,
    avgCacheTime: 0,
  };

  recordHit(duration: number): void {
    this.metrics.hits++;
    this.metrics.totalRequests++;

    // Update average cache retrieval time
    this.metrics.avgCacheTime =
      (this.metrics.avgCacheTime * (this.metrics.hits - 1) + duration) /
      this.metrics.hits;
  }

  recordMiss(computeDuration: number): void {
    this.metrics.misses++;
    this.metrics.totalRequests++;

    // Update average computation time
    this.metrics.avgComputeTime =
      (this.metrics.avgComputeTime * (this.metrics.misses - 1) + computeDuration) /
      this.metrics.misses;
  }

  getMetrics(): CacheMetrics & { hitRate: number } {
    const hitRate =
      this.metrics.totalRequests > 0
        ? this.metrics.hits / this.metrics.totalRequests
        : 0;

    return {
      ...this.metrics,
      hitRate,
    };
  }

  logMetrics(): void {
    const metrics = this.getMetrics();
    console.log('=== Cache Performance Metrics ===');
    console.log(`Total Requests: ${metrics.totalRequests}`);
    console.log(`Cache Hits: ${metrics.hits}`);
    console.log(`Cache Misses: ${metrics.misses}`);
    console.log(`Hit Rate: ${(metrics.hitRate * 100).toFixed(2)}%`);
    console.log(`Avg Cache Time: ${metrics.avgCacheTime.toFixed(2)}ms`);
    console.log(`Avg Compute Time: ${metrics.avgComputeTime.toFixed(2)}ms`);
    console.log(
      `Time Saved: ${((metrics.avgComputeTime - metrics.avgCacheTime) * metrics.hits).toFixed(2)}ms`
    );
  }
}

const cacheMonitor = new CacheMonitor();

// Use monitor in nodes
mlGraph.addNode('monitoredComputation', async (state) => {
  const cacheKey = `computation:${state.inputText}`;
  const startTime = Date.now();

  const cached = await cacheManager.get<string>(cacheKey);

  if (cached) {
    const duration = Date.now() - startTime;
    cacheMonitor.recordHit(duration);
    return { result: cached };
  }

  // Perform computation
  const result = await expensiveComputation(state.inputText);
  const duration = Date.now() - startTime;

  cacheMonitor.recordMiss(duration);
  await cacheManager.set(cacheKey, result);

  return { result };
});

async function expensiveComputation(input: string): Promise<string> {
  await new Promise((resolve) => setTimeout(resolve, 500));
  return `Computed: ${input}`;
}

// Log metrics periodically
setInterval(() => {
  cacheMonitor.logMetrics();
}, 60000); // Every minute
```

---

## Deferred Nodes

### Overview

Deferred nodes in LangGraph v1.0+ allow you to delay the execution of specific nodes until all upstream paths have completed. This is particularly useful for aggregation operations, final processing steps, or scenarios where you need to ensure all parallel branches have finished before proceeding.

### Basic Deferred Node

Define a node that waits for all upstream nodes to complete before executing.

```typescript
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';

interface ParallelProcessingState {
  input: string;
  branchAResult: string | null;
  branchBResult: string | null;
  branchCResult: string | null;
  aggregatedResult: string | null;
}

const ParallelStateAnnotation = Annotation.Root({
  input: Annotation<string>,
  branchAResult: Annotation<string | null>({ default: () => null }),
  branchBResult: Annotation<string | null>({ default: () => null }),
  branchCResult: Annotation<string | null>({ default: () => null }),
  aggregatedResult: Annotation<string | null>({ default: () => null }),
});

const graph = new StateGraph(ParallelStateAnnotation);

// Define parallel processing branches
graph
  .addNode('branchA', async (state) => {
    await new Promise((resolve) => setTimeout(resolve, 100));
    return {
      branchAResult: `Branch A processed: ${state.input}`,
    };
  })
  .addNode('branchB', async (state) => {
    await new Promise((resolve) => setTimeout(resolve, 200));
    return {
      branchBResult: `Branch B processed: ${state.input}`,
    };
  })
  .addNode('branchC', async (state) => {
    await new Promise((resolve) => setTimeout(resolve, 150));
    return {
      branchCResult: `Branch C processed: ${state.input}`,
    };
  });

// Deferred aggregation node - waits for all branches
graph.addNode(
  'aggregate',
  async (state) => {
    console.log('Aggregating results from all branches');

    const allResults = [
      state.branchAResult,
      state.branchBResult,
      state.branchCResult,
    ].filter(Boolean);

    return {
      aggregatedResult: `Aggregated: ${allResults.join(' | ')}`,
    };
  },
  {
    // Mark as deferred - will wait for all upstream paths
    deferred: true,
  }
);

// Setup edges - all branches flow to aggregate
graph.addEdge(START, 'branchA');
graph.addEdge(START, 'branchB');
graph.addEdge(START, 'branchC');

graph.addEdge('branchA', 'aggregate');
graph.addEdge('branchB', 'aggregate');
graph.addEdge('branchC', 'aggregate');

graph.addEdge('aggregate', END);

const workflow = graph.compile();

// Execute workflow
const result = await workflow.invoke({
  input: 'test data',
  branchAResult: null,
  branchBResult: null,
  branchCResult: null,
  aggregatedResult: null,
});

console.log('Final result:', result.aggregatedResult);
// Output: "Aggregated: Branch A processed: test data | Branch B processed: test data | Branch C processed: test data"
```

### Fan-Out/Fan-In Pattern with Deferred Nodes

Implement the common fan-out/fan-in pattern for parallel processing and aggregation.

```typescript
interface DataAnalysisState {
  dataset: number[];
  sumResult: number | null;
  avgResult: number | null;
  maxResult: number | null;
  minResult: number | null;
  statistics: {
    sum: number;
    avg: number;
    max: number;
    min: number;
    count: number;
  } | null;
}

const AnalysisStateAnnotation = Annotation.Root({
  dataset: Annotation<number[]>,
  sumResult: Annotation<number | null>({ default: () => null }),
  avgResult: Annotation<number | null>({ default: () => null }),
  maxResult: Annotation<number | null>({ default: () => null }),
  minResult: Annotation<number | null>({ default: () => null }),
  statistics: Annotation<DataAnalysisState['statistics']>({ default: () => null }),
});

const analysisGraph = new StateGraph(AnalysisStateAnnotation);

// Parallel analysis nodes
analysisGraph
  .addNode('calculateSum', async (state) => {
    const sum = state.dataset.reduce((a, b) => a + b, 0);
    return { sumResult: sum };
  })
  .addNode('calculateAvg', async (state) => {
    const avg = state.dataset.reduce((a, b) => a + b, 0) / state.dataset.length;
    return { avgResult: avg };
  })
  .addNode('calculateMax', async (state) => {
    const max = Math.max(...state.dataset);
    return { maxResult: max };
  })
  .addNode('calculateMin', async (state) => {
    const min = Math.min(...state.dataset);
    return { minResult: min };
  });

// Deferred aggregation node
analysisGraph.addNode(
  'generateStatistics',
  async (state) => {
    console.log('Generating final statistics report');

    return {
      statistics: {
        sum: state.sumResult!,
        avg: state.avgResult!,
        max: state.maxResult!,
        min: state.minResult!,
        count: state.dataset.length,
      },
    };
  },
  {
    deferred: true,
  }
);

// Fan-out from START to all analysis nodes
analysisGraph.addEdge(START, 'calculateSum');
analysisGraph.addEdge(START, 'calculateAvg');
analysisGraph.addEdge(START, 'calculateMax');
analysisGraph.addEdge(START, 'calculateMin');

// Fan-in from all analysis nodes to aggregation
analysisGraph.addEdge('calculateSum', 'generateStatistics');
analysisGraph.addEdge('calculateAvg', 'generateStatistics');
analysisGraph.addEdge('calculateMax', 'generateStatistics');
analysisGraph.addEdge('calculateMin', 'generateStatistics');

analysisGraph.addEdge('generateStatistics', END);

const analysisWorkflow = analysisGraph.compile();

// Execute
const dataset = [10, 20, 30, 40, 50];
const analysis = await analysisWorkflow.invoke({
  dataset,
  sumResult: null,
  avgResult: null,
  maxResult: null,
  minResult: null,
  statistics: null,
});

console.log('Statistics:', analysis.statistics);
// Output: { sum: 150, avg: 30, max: 50, min: 10, count: 5 }
```

### Conditional Deferred Execution

Combine deferred nodes with conditional logic for complex workflows.

```typescript
interface MultiStageState {
  stage: 'preprocessing' | 'processing' | 'postprocessing';
  paths: string[];
  pathAResult: string | null;
  pathBResult: string | null;
  pathCResult: string | null;
  finalResult: string | null;
}

const MultiStageAnnotation = Annotation.Root({
  stage: Annotation<'preprocessing' | 'processing' | 'postprocessing'>,
  paths: Annotation<string[]>({ default: () => [] }),
  pathAResult: Annotation<string | null>({ default: () => null }),
  pathBResult: Annotation<string | null>({ default: () => null }),
  pathCResult: Annotation<string | null>({ default: () => null }),
  finalResult: Annotation<string | null>({ default: () => null }),
});

const multiStageGraph = new StateGraph(MultiStageAnnotation);

// Router node
multiStageGraph.addNode('router', (state) => state);

// Parallel paths
multiStageGraph
  .addNode('pathA', async (state) => ({
    paths: [...state.paths, 'A'],
    pathAResult: 'Path A completed',
  }))
  .addNode('pathB', async (state) => ({
    paths: [...state.paths, 'B'],
    pathBResult: 'Path B completed',
  }))
  .addNode('pathC', async (state) => ({
    paths: [...state.paths, 'C'],
    pathCResult: 'Path C completed',
  }));

// Deferred merger node
multiStageGraph.addNode(
  'merge',
  async (state) => {
    const results = [
      state.pathAResult,
      state.pathBResult,
      state.pathCResult,
    ].filter(Boolean);

    return {
      finalResult: `Merged ${results.length} paths: ${results.join(', ')}`,
    };
  },
  {
    deferred: true,
  }
);

// Conditional routing from router
multiStageGraph.addEdge(START, 'router');

multiStageGraph.addConditionalEdges(
  'router',
  (state) => {
    if (state.stage === 'preprocessing') {
      return 'all';
    } else if (state.stage === 'processing') {
      return 'ab';
    } else {
      return 'single';
    }
  },
  {
    all: ['pathA', 'pathB', 'pathC'],
    ab: ['pathA', 'pathB'],
    single: ['pathA'],
  }
);

// All paths lead to deferred merge
multiStageGraph.addEdge('pathA', 'merge');
multiStageGraph.addEdge('pathB', 'merge');
multiStageGraph.addEdge('pathC', 'merge');

multiStageGraph.addEdge('merge', END);

const multiStageWorkflow = multiStageGraph.compile();

// Test different stages
const result1 = await multiStageWorkflow.invoke({
  stage: 'preprocessing',
  paths: [],
  pathAResult: null,
  pathBResult: null,
  pathCResult: null,
  finalResult: null,
});
console.log('Preprocessing:', result1.finalResult);
// Output: "Merged 3 paths: Path A completed, Path B completed, Path C completed"

const result2 = await multiStageWorkflow.invoke({
  stage: 'processing',
  paths: [],
  pathAResult: null,
  pathBResult: null,
  pathCResult: null,
  finalResult: null,
});
console.log('Processing:', result2.finalResult);
// Output: "Merged 2 paths: Path A completed, Path B completed"
```

### Error Handling with Deferred Nodes

Implement robust error handling for deferred aggregation.

```typescript
interface ResilientState {
  tasks: string[];
  results: Array<{ task: string; result: string; error?: string }>;
  summary: string | null;
}

const ResilientStateAnnotation = Annotation.Root({
  tasks: Annotation<string[]>,
  results: Annotation<ResilientState['results']>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  summary: Annotation<string | null>({ default: () => null }),
});

const resilientGraph = new StateGraph(ResilientStateAnnotation);

// Dynamic task processors
const createTaskNode = (taskId: string) => {
  return async (state: ResilientState) => {
    try {
      // Simulate task processing with potential failure
      if (Math.random() < 0.3) {
        throw new Error(`Task ${taskId} failed randomly`);
      }

      await new Promise((resolve) => setTimeout(resolve, 100));

      return {
        results: [{
          task: taskId,
          result: `Task ${taskId} completed successfully`,
        }],
      };
    } catch (error) {
      return {
        results: [{
          task: taskId,
          result: '',
          error: error instanceof Error ? error.message : 'Unknown error',
        }],
      };
    }
  };
};

// Add task nodes
resilientGraph
  .addNode('task1', createTaskNode('task1'))
  .addNode('task2', createTaskNode('task2'))
  .addNode('task3', createTaskNode('task3'));

// Deferred summary node with error handling
resilientGraph.addNode(
  'summarize',
  async (state) => {
    const successful = state.results.filter((r) => !r.error);
    const failed = state.results.filter((r) => r.error);

    const summary = `
Execution Summary:
------------------
Total Tasks: ${state.results.length}
Successful: ${successful.length}
Failed: ${failed.length}

Successful Tasks:
${successful.map((r) => `- ${r.task}: ${r.result}`).join('\n')}

${failed.length > 0 ? `Failed Tasks:\n${failed.map((r) => `- ${r.task}: ${r.error}`).join('\n')}` : ''}
    `.trim();

    return { summary };
  },
  {
    deferred: true,
  }
);

// Connect all tasks in parallel
resilientGraph.addEdge(START, 'task1');
resilientGraph.addEdge(START, 'task2');
resilientGraph.addEdge(START, 'task3');

resilientGraph.addEdge('task1', 'summarize');
resilientGraph.addEdge('task2', 'summarize');
resilientGraph.addEdge('task3', 'summarize');

resilientGraph.addEdge('summarize', END);

const resilientWorkflow = resilientGraph.compile();

// Execute with error resilience
const result = await resilientWorkflow.invoke({
  tasks: ['task1', 'task2', 'task3'],
  results: [],
  summary: null,
});

console.log(result.summary);
```

### Timeout Handling for Deferred Nodes

Add timeout protection for deferred nodes waiting on slow upstream paths.

```typescript
interface TimeoutState {
  input: string;
  fastResult: string | null;
  slowResult: string | null;
  timedOut: boolean;
  finalResult: string | null;
}

const TimeoutStateAnnotation = Annotation.Root({
  input: Annotation<string>,
  fastResult: Annotation<string | null>({ default: () => null }),
  slowResult: Annotation<string | null>({ default: () => null }),
  timedOut: Annotation<boolean>({ default: () => false }),
  finalResult: Annotation<string | null>({ default: () => null }),
});

const timeoutGraph = new StateGraph(TimeoutStateAnnotation);

timeoutGraph
  .addNode('fastPath', async (state) => {
    await new Promise((resolve) => setTimeout(resolve, 100));
    return { fastResult: `Fast: ${state.input}` };
  })
  .addNode('slowPath', async (state) => {
    await new Promise((resolve) => setTimeout(resolve, 5000)); // 5 seconds
    return { slowResult: `Slow: ${state.input}` };
  });

timeoutGraph.addNode(
  'aggregateWithTimeout',
  async (state) => {
    // This node is deferred but has timeout logic
    const hasAllResults = state.fastResult && state.slowResult;

    if (!hasAllResults && state.timedOut) {
      return {
        finalResult: `Partial results (timeout): Fast=${state.fastResult}, Slow=TIMEOUT`,
      };
    }

    return {
      finalResult: `All results: Fast=${state.fastResult}, Slow=${state.slowResult}`,
    };
  },
  {
    deferred: true,
    timeout: 3000, // Wait maximum 3 seconds for all upstream nodes
  }
);

timeoutGraph.addEdge(START, 'fastPath');
timeoutGraph.addEdge(START, 'slowPath');

timeoutGraph.addEdge('fastPath', 'aggregateWithTimeout');
timeoutGraph.addEdge('slowPath', 'aggregateWithTimeout');

timeoutGraph.addEdge('aggregateWithTimeout', END);

const timeoutWorkflow = timeoutGraph.compile();

try {
  const result = await timeoutWorkflow.invoke({
    input: 'test',
    fastResult: null,
    slowResult: null,
    timedOut: false,
    finalResult: null,
  });

  console.log(result.finalResult);
} catch (error) {
  console.error('Workflow timeout:', error);
}
```

---

## Pre/Post Model Hooks

### Overview

LangGraph v1.0+ introduces pre and post-model hooks that allow you to inject custom logic before and after model invocations. This enables request transformation, response modification, logging, caching, and other cross-cutting concerns without modifying core node logic.

### Basic Model Hooks

Define hooks that execute before and after model calls.

```typescript
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, AIMessage } from '@langchain/core/messages';

interface HookContext {
  modelName: string;
  temperature: number;
  timestamp: number;
  metadata: Record<string, any>;
}

class ModelHookManager {
  private preHooks: Array<(messages: any[], context: HookContext) => Promise<any[]>> = [];
  private postHooks: Array<(response: any, context: HookContext) => Promise<any>> = [];

  registerPreHook(
    hook: (messages: any[], context: HookContext) => Promise<any[]>
  ): void {
    this.preHooks.push(hook);
  }

  registerPostHook(
    hook: (response: any, context: HookContext) => Promise<any>
  ): void {
    this.postHooks.push(hook);
  }

  async executePreHooks(
    messages: any[],
    context: HookContext
  ): Promise<any[]> {
    let processedMessages = messages;

    for (const hook of this.preHooks) {
      processedMessages = await hook(processedMessages, context);
    }

    return processedMessages;
  }

  async executePostHooks(
    response: any,
    context: HookContext
  ): Promise<any> {
    let processedResponse = response;

    for (const hook of this.postHooks) {
      processedResponse = await hook(processedResponse, context);
    }

    return processedResponse;
  }
}

const hookManager = new ModelHookManager();

// Pre-hook: Add system message
hookManager.registerPreHook(async (messages, context) => {
  console.log('[PRE-HOOK] Adding system message');

  const systemMessage = {
    role: 'system',
    content: `You are a helpful assistant. Current time: ${new Date(context.timestamp).toISOString()}`,
  };

  return [systemMessage, ...messages];
});

// Pre-hook: Log request
hookManager.registerPreHook(async (messages, context) => {
  console.log('[PRE-HOOK] Logging request:', {
    model: context.modelName,
    messageCount: messages.length,
    timestamp: context.timestamp,
  });

  return messages;
});

// Post-hook: Log response
hookManager.registerPostHook(async (response, context) => {
  console.log('[POST-HOOK] Logging response:', {
    model: context.modelName,
    responseLength: response.content?.length || 0,
    timestamp: context.timestamp,
  });

  return response;
});

// Post-hook: Transform response
hookManager.registerPostHook(async (response, context) => {
  console.log('[POST-HOOK] Transforming response');

  // Add metadata to response
  return {
    ...response,
    metadata: {
      ...response.metadata,
      processedAt: Date.now(),
      modelUsed: context.modelName,
    },
  };
});

// Model wrapper with hooks
async function invokeModelWithHooks(
  model: ChatOpenAI,
  messages: any[],
  context: HookContext
): Promise<any> {
  // Execute pre-hooks
  const processedMessages = await hookManager.executePreHooks(messages, context);

  // Invoke model
  const response = await model.invoke(processedMessages);

  // Execute post-hooks
  const processedResponse = await hookManager.executePostHooks(response, context);

  return processedResponse;
}

// Usage
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0.7,
});

const result = await invokeModelWithHooks(
  model,
  [new HumanMessage('What is TypeScript?')],
  {
    modelName: 'gpt-4-turbo',
    temperature: 0.7,
    timestamp: Date.now(),
    metadata: {},
  }
);

console.log('Final result:', result);
```

### Request Transformation Hooks

Transform requests before they reach the model.

```typescript
interface MessageTransformer {
  name: string;
  transform: (messages: any[]) => Promise<any[]>;
}

class RequestTransformationPipeline {
  private transformers: MessageTransformer[] = [];

  addTransformer(transformer: MessageTransformer): void {
    this.transformers.push(transformer);
  }

  async transform(messages: any[]): Promise<any[]> {
    let transformed = messages;

    for (const transformer of this.transformers) {
      console.log(`Applying transformer: ${transformer.name}`);
      transformed = await transformer.transform(transformed);
    }

    return transformed;
  }
}

const pipeline = new RequestTransformationPipeline();

// Transformer: Inject context
pipeline.addTransformer({
  name: 'context-injector',
  transform: async (messages) => {
    const contextMessage = {
      role: 'system',
      content: 'Additional context: You are helping with a TypeScript project.',
    };

    return [contextMessage, ...messages];
  },
});

// Transformer: Message sanitization
pipeline.addTransformer({
  name: 'sanitizer',
  transform: async (messages) => {
    return messages.map((msg) => ({
      ...msg,
      content:
        typeof msg.content === 'string'
          ? msg.content.replace(/<script>/g, '')
          : msg.content,
    }));
  },
});

// Transformer: Add few-shot examples
pipeline.addTransformer({
  name: 'few-shot-examples',
  transform: async (messages) => {
    const examples = [
      { role: 'user', content: 'What is a type?' },
      {
        role: 'assistant',
        content: 'A type is a way to define the shape of data in TypeScript.',
      },
    ];

    // Insert examples before last user message
    const lastUserIndex = messages
      .map((m, i) => ({ m, i }))
      .filter(({ m }) => m.role === 'user')
      .pop()?.i;

    if (lastUserIndex !== undefined) {
      return [
        ...messages.slice(0, lastUserIndex),
        ...examples,
        ...messages.slice(lastUserIndex),
      ];
    }

    return messages;
  },
});

// Usage in graph node
interface ChatState {
  messages: any[];
  response: any;
}

const ChatStateAnnotation = Annotation.Root({
  messages: Annotation<any[]>({ default: () => [] }),
  response: Annotation<any>({ default: () => null }),
});

const chatGraph = new StateGraph(ChatStateAnnotation);

chatGraph.addNode('chat', async (state) => {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  // Transform messages through pipeline
  const transformedMessages = await pipeline.transform(state.messages);

  // Invoke model with transformed messages
  const response = await model.invoke(transformedMessages);

  return { response };
});

chatGraph.addEdge(START, 'chat');
chatGraph.addEdge('chat', END);

const chatWorkflow = chatGraph.compile();
```

### Response Caching Hook

Implement response caching as a post-hook.

```typescript
import Redis from 'ioredis';

class ResponseCacheHook {
  private redis: Redis;
  private ttl: number;

  constructor(redisUrl: string, ttl: number = 3600) {
    this.redis = new Redis(redisUrl);
    this.ttl = ttl;
  }

  private generateCacheKey(messages: any[]): string {
    const messageStr = JSON.stringify(messages);
    return `model:response:${Buffer.from(messageStr).toString('base64')}`;
  }

  async checkCache(messages: any[]): Promise<any | null> {
    const cacheKey = this.generateCacheKey(messages);
    const cached = await this.redis.get(cacheKey);

    if (cached) {
      console.log('[CACHE] Hit - returning cached response');
      return JSON.parse(cached);
    }

    console.log('[CACHE] Miss - will invoke model');
    return null;
  }

  async cacheResponse(messages: any[], response: any): Promise<void> {
    const cacheKey = this.generateCacheKey(messages);
    await this.redis.setex(cacheKey, this.ttl, JSON.stringify(response));
    console.log('[CACHE] Stored response');
  }

  async invalidateCache(pattern: string = '*'): Promise<void> {
    const keys = await this.redis.keys(`model:response:${pattern}`);
    if (keys.length > 0) {
      await this.redis.del(...keys);
      console.log(`[CACHE] Invalidated ${keys.length} entries`);
    }
  }
}

const cacheHook = new ResponseCacheHook(process.env.REDIS_URL || 'redis://localhost:6379');

// Model invocation with caching
async function invokeWithCache(
  model: ChatOpenAI,
  messages: any[]
): Promise<any> {
  // Check cache (pre-hook behavior)
  const cached = await cacheHook.checkCache(messages);
  if (cached) {
    return cached;
  }

  // Invoke model
  const response = await model.invoke(messages);

  // Cache response (post-hook behavior)
  await cacheHook.cacheResponse(messages, response);

  return response;
}
```

### Logging and Monitoring Hooks

Comprehensive logging for all model interactions.

```typescript
interface ModelInvocationLog {
  id: string;
  timestamp: number;
  modelName: string;
  inputTokens: number;
  outputTokens: number;
  duration: number;
  success: boolean;
  error?: string;
  metadata: Record<string, any>;
}

class ModelMonitoringHooks {
  private logs: ModelInvocationLog[] = [];

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async preInvoke(messages: any[], context: HookContext): Promise<any[]> {
    const invocationId = this.generateId();

    console.log(`[MONITOR] Starting invocation ${invocationId}`);
    console.log(`[MONITOR] Model: ${context.modelName}`);
    console.log(`[MONITOR] Messages: ${messages.length}`);

    // Store invocation start time in context
    context.metadata.invocationId = invocationId;
    context.metadata.startTime = Date.now();

    return messages;
  }

  async postInvoke(response: any, context: HookContext): Promise<any> {
    const duration = Date.now() - (context.metadata.startTime || 0);
    const invocationId = context.metadata.invocationId;

    const log: ModelInvocationLog = {
      id: invocationId,
      timestamp: Date.now(),
      modelName: context.modelName,
      inputTokens: response.usage?.input_tokens || 0,
      outputTokens: response.usage?.output_tokens || 0,
      duration,
      success: true,
      metadata: context.metadata,
    };

    this.logs.push(log);

    console.log(`[MONITOR] Completed invocation ${invocationId}`);
    console.log(`[MONITOR] Duration: ${duration}ms`);
    console.log(`[MONITOR] Tokens: ${log.inputTokens + log.outputTokens}`);

    return response;
  }

  async onError(error: Error, context: HookContext): Promise<void> {
    const duration = Date.now() - (context.metadata.startTime || 0);
    const invocationId = context.metadata.invocationId;

    const log: ModelInvocationLog = {
      id: invocationId,
      timestamp: Date.now(),
      modelName: context.modelName,
      inputTokens: 0,
      outputTokens: 0,
      duration,
      success: false,
      error: error.message,
      metadata: context.metadata,
    };

    this.logs.push(log);

    console.error(`[MONITOR] Failed invocation ${invocationId}`);
    console.error(`[MONITOR] Error: ${error.message}`);
  }

  getMetrics(): {
    totalInvocations: number;
    successRate: number;
    avgDuration: number;
    totalTokens: number;
  } {
    const successful = this.logs.filter((l) => l.success);

    return {
      totalInvocations: this.logs.length,
      successRate: this.logs.length > 0 ? successful.length / this.logs.length : 0,
      avgDuration:
        this.logs.length > 0
          ? this.logs.reduce((sum, l) => sum + l.duration, 0) / this.logs.length
          : 0,
      totalTokens: this.logs.reduce(
        (sum, l) => sum + l.inputTokens + l.outputTokens,
        0
      ),
    };
  }

  exportLogs(): ModelInvocationLog[] {
    return [...this.logs];
  }
}

const monitoringHooks = new ModelMonitoringHooks();

// Integrated model call with monitoring
async function monitoredModelCall(
  model: ChatOpenAI,
  messages: any[],
  context: HookContext
): Promise<any> {
  try {
    // Pre-invoke hook
    const processedMessages = await monitoringHooks.preInvoke(messages, context);

    // Model invocation
    const response = await model.invoke(processedMessages);

    // Post-invoke hook
    const processedResponse = await monitoringHooks.postInvoke(response, context);

    return processedResponse;
  } catch (error) {
    // Error hook
    await monitoringHooks.onError(error as Error, context);
    throw error;
  }
}

// Periodic metrics logging
setInterval(() => {
  const metrics = monitoringHooks.getMetrics();
  console.log('=== Model Metrics ===');
  console.log(`Total Invocations: ${metrics.totalInvocations}`);
  console.log(`Success Rate: ${(metrics.successRate * 100).toFixed(2)}%`);
  console.log(`Avg Duration: ${metrics.avgDuration.toFixed(2)}ms`);
  console.log(`Total Tokens: ${metrics.totalTokens}`);
}, 60000); // Every minute
```

### Content Moderation Hook

Implement content filtering as a pre/post hook.

```typescript
interface ModerationResult {
  flagged: boolean;
  categories: string[];
  severity: 'low' | 'medium' | 'high';
}

class ContentModerationHook {
  private blockedPatterns: RegExp[] = [
    /\b(password|api[_-]?key|secret|token)\s*[:=]\s*\S+/i,
    /\b\d{3}-\d{2}-\d{4}\b/, // SSN pattern
    /\b\d{16}\b/, // Credit card pattern
  ];

  async moderateInput(messages: any[]): Promise<ModerationResult> {
    const flaggedCategories: string[] = [];

    for (const message of messages) {
      const content = typeof message.content === 'string' ? message.content : '';

      // Check for sensitive patterns
      for (const pattern of this.blockedPatterns) {
        if (pattern.test(content)) {
          flaggedCategories.push('sensitive-data');
          break;
        }
      }

      // Check for inappropriate content
      if (this.containsInappropriateContent(content)) {
        flaggedCategories.push('inappropriate');
      }
    }

    return {
      flagged: flaggedCategories.length > 0,
      categories: flaggedCategories,
      severity: flaggedCategories.length > 1 ? 'high' : flaggedCategories.length === 1 ? 'medium' : 'low',
    };
  }

  private containsInappropriateContent(content: string): boolean {
    // Implement your inappropriate content detection
    const inappropriatePatterns = [
      // Add your patterns
    ];

    return inappropriatePatterns.some((pattern) => pattern.test(content));
  }

  async moderateOutput(response: any): Promise<ModerationResult> {
    const content = typeof response.content === 'string' ? response.content : '';

    // Check if response contains sensitive info
    const containsSensitiveData = this.blockedPatterns.some((pattern) =>
      pattern.test(content)
    );

    return {
      flagged: containsSensitiveData,
      categories: containsSensitiveData ? ['sensitive-data'] : [],
      severity: containsSensitiveData ? 'high' : 'low',
    };
  }
}

const moderationHook = new ContentModerationHook();

// Model call with moderation
async function moderatedModelCall(
  model: ChatOpenAI,
  messages: any[]
): Promise<any> {
  // Pre-hook: Moderate input
  const inputModeration = await moderationHook.moderateInput(messages);

  if (inputModeration.flagged) {
    throw new Error(
      `Input flagged for moderation: ${inputModeration.categories.join(', ')}`
    );
  }

  // Invoke model
  const response = await model.invoke(messages);

  // Post-hook: Moderate output
  const outputModeration = await moderationHook.moderateOutput(response);

  if (outputModeration.flagged) {
    console.warn('Output contains sensitive content, filtering...');

    return {
      ...response,
      content:
        'I cannot provide that information as it may contain sensitive data.',
    };
  }

  return response;
}
```

---

## Cross-Thread Memory Support

### Overview

LangGraph v1.0+ introduces cross-thread memory support, allowing memory persistence across different conversation threads. This enables shared context, user preferences, and knowledge to be accessible across multiple independent conversations.

### Basic Cross-Thread Memory

Implement shared memory across multiple threads.

```typescript
import { MemorySaver } from '@langchain/langgraph';

interface UserMemory {
  userId: string;
  preferences: Record<string, any>;
  facts: string[];
  conversationHistory: Array<{
    threadId: string;
    timestamp: number;
    summary: string;
  }>;
}

class CrossThreadMemoryManager {
  private userMemories: Map<string, UserMemory> = new Map();

  getUserMemory(userId: string): UserMemory {
    if (!this.userMemories.has(userId)) {
      this.userMemories.set(userId, {
        userId,
        preferences: {},
        facts: [],
        conversationHistory: [],
      });
    }

    return this.userMemories.get(userId)!;
  }

  updatePreferences(
    userId: string,
    preferences: Record<string, any>
  ): void {
    const memory = this.getUserMemory(userId);
    memory.preferences = {
      ...memory.preferences,
      ...preferences,
    };
  }

  addFact(userId: string, fact: string): void {
    const memory = this.getUserMemory(userId);
    if (!memory.facts.includes(fact)) {
      memory.facts.push(fact);
    }
  }

  addConversationSummary(
    userId: string,
    threadId: string,
    summary: string
  ): void {
    const memory = this.getUserMemory(userId);
    memory.conversationHistory.push({
      threadId,
      timestamp: Date.now(),
      summary,
    });
  }

  getContextForThread(userId: string, threadId: string): string {
    const memory = this.getUserMemory(userId);

    const context = `
User Preferences:
${Object.entries(memory.preferences)
  .map(([key, value]) => `- ${key}: ${value}`)
  .join('\n')}

Known Facts:
${memory.facts.map((f) => `- ${f}`).join('\n')}

Previous Conversations:
${memory.conversationHistory
  .filter((c) => c.threadId !== threadId)
  .map((c) => `- ${new Date(c.timestamp).toISOString()}: ${c.summary}`)
  .join('\n')}
    `.trim();

    return context;
  }
}

const crossThreadMemory = new CrossThreadMemoryManager();

// Graph with cross-thread memory
interface ConversationState {
  userId: string;
  threadId: string;
  messages: any[];
  userContext: string;
  response: any;
}

const ConversationStateAnnotation = Annotation.Root({
  userId: Annotation<string>,
  threadId: Annotation<string>,
  messages: Annotation<any[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  userContext: Annotation<string>({ default: () => '' }),
  response: Annotation<any>({ default: () => null }),
});

const conversationGraph = new StateGraph(ConversationStateAnnotation);

// Load user context from cross-thread memory
conversationGraph.addNode('loadContext', async (state) => {
  const context = crossThreadMemory.getContextForThread(
    state.userId,
    state.threadId
  );

  return { userContext: context };
});

// Process with context
conversationGraph.addNode('processWithContext', async (state) => {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const systemMessage = {
    role: 'system',
    content: `You are a helpful assistant. Here is what we know about the user:\n\n${state.userContext}`,
  };

  const messages = [systemMessage, ...state.messages];
  const response = await model.invoke(messages);

  return { response };
});

// Extract and save learnings
conversationGraph.addNode('savelearnings', async (state) => {
  // Extract user preferences from conversation
  const responseText =
    typeof state.response?.content === 'string' ? state.response.content : '';

  // Simple preference extraction (in production, use a model for this)
  if (responseText.toLowerCase().includes('prefer')) {
    crossThreadMemory.addFact(
      state.userId,
      `Expressed preference in conversation ${state.threadId}`
    );
  }

  return state;
});

conversationGraph.addEdge(START, 'loadContext');
conversationGraph.addEdge('loadContext', 'processWithContext');
conversationGraph.addEdge('processWithContext', 'saveLearnings');
conversationGraph.addEdge('saveLearnings', END);

const conversationWorkflow = conversationGraph.compile();

// Usage across multiple threads
const userId = 'user123';

// Thread 1
await conversationWorkflow.invoke({
  userId,
  threadId: 'thread1',
  messages: [{ role: 'user', content: 'I prefer TypeScript over JavaScript' }],
  userContext: '',
  response: null,
});

crossThreadMemory.updatePreferences(userId, { language: 'TypeScript' });
crossThreadMemory.addConversationSummary(
  userId,
  'thread1',
  'User expressed preference for TypeScript'
);

// Thread 2 - automatically has context from Thread 1
await conversationWorkflow.invoke({
  userId,
  threadId: 'thread2',
  messages: [{ role: 'user', content: 'Can you help me with my project?' }],
  userContext: '',
  response: null,
});
```

### Persistent Cross-Thread Memory with Database

Implement persistent cross-thread memory using PostgreSQL.

```typescript
import { Pool } from 'pg';

class DatabaseCrossThreadMemory {
  private pool: Pool;

  constructor(connectionString: string) {
    this.pool = new Pool({ connectionString });
    this.initialize();
  }

  private async initialize(): Promise<void> {
    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS user_memory (
        user_id VARCHAR(255) PRIMARY KEY,
        preferences JSONB DEFAULT '{}',
        facts TEXT[] DEFAULT '{}',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS conversation_summaries (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        thread_id VARCHAR(255) NOT NULL,
        summary TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user_memory(user_id)
      );

      CREATE INDEX IF NOT EXISTS idx_user_summaries
      ON conversation_summaries(user_id, timestamp DESC);
    `);
  }

  async getUserMemory(userId: string): Promise<UserMemory> {
    const result = await this.pool.query(
      'SELECT * FROM user_memory WHERE user_id = $1',
      [userId]
    );

    if (result.rows.length === 0) {
      // Create new user memory
      await this.pool.query(
        'INSERT INTO user_memory (user_id) VALUES ($1)',
        [userId]
      );

      return {
        userId,
        preferences: {},
        facts: [],
        conversationHistory: [],
      };
    }

    const row = result.rows[0];

    // Get conversation history
    const summaries = await this.pool.query(
      'SELECT thread_id, summary, timestamp FROM conversation_summaries WHERE user_id = $1 ORDER BY timestamp DESC LIMIT 10',
      [userId]
    );

    return {
      userId,
      preferences: row.preferences || {},
      facts: row.facts || [],
      conversationHistory: summaries.rows.map((s) => ({
        threadId: s.thread_id,
        timestamp: new Date(s.timestamp).getTime(),
        summary: s.summary,
      })),
    };
  }

  async updatePreferences(
    userId: string,
    preferences: Record<string, any>
  ): Promise<void> {
    await this.pool.query(
      `INSERT INTO user_memory (user_id, preferences)
       VALUES ($1, $2)
       ON CONFLICT (user_id)
       DO UPDATE SET
         preferences = user_memory.preferences || EXCLUDED.preferences,
         updated_at = CURRENT_TIMESTAMP`,
      [userId, JSON.stringify(preferences)]
    );
  }

  async addFact(userId: string, fact: string): Promise<void> {
    await this.pool.query(
      `INSERT INTO user_memory (user_id, facts)
       VALUES ($1, ARRAY[$2])
       ON CONFLICT (user_id)
       DO UPDATE SET
         facts = array_append(user_memory.facts, $2),
         updated_at = CURRENT_TIMESTAMP`,
      [userId, fact]
    );
  }

  async addConversationSummary(
    userId: string,
    threadId: string,
    summary: string
  ): Promise<void> {
    await this.pool.query(
      'INSERT INTO conversation_summaries (user_id, thread_id, summary) VALUES ($1, $2, $3)',
      [userId, threadId, summary]
    );
  }

  async getRelevantContext(
    userId: string,
    currentThreadId: string,
    query: string
  ): Promise<string> {
    const memory = await this.getUserMemory(userId);

    // In production, use vector search for semantic relevance
    const relevantSummaries = memory.conversationHistory
      .filter((c) => c.threadId !== currentThreadId)
      .slice(0, 3);

    const context = `
User Profile:
${Object.entries(memory.preferences)
  .map(([k, v]) => `- ${k}: ${v}`)
  .join('\n')}

Known Facts:
${memory.facts.slice(0, 5).map((f) => `- ${f}`).join('\n')}

Relevant Previous Conversations:
${relevantSummaries
  .map((s) => `- ${new Date(s.timestamp).toLocaleDateString()}: ${s.summary}`)
  .join('\n')}
    `.trim();

    return context;
  }
}

const dbMemory = new DatabaseCrossThreadMemory(process.env.DATABASE_URL!);
```

### Shared Knowledge Base

Implement a shared knowledge base accessible across all threads.

```typescript
interface KnowledgeEntry {
  id: string;
  userId: string;
  category: string;
  content: string;
  source: string; // thread_id where it was learned
  confidence: number;
  createdAt: number;
  updatedAt: number;
}

class SharedKnowledgeBase {
  private knowledge: Map<string, KnowledgeEntry[]> = new Map();

  addKnowledge(
    userId: string,
    category: string,
    content: string,
    source: string,
    confidence: number = 1.0
  ): void {
    if (!this.knowledge.has(userId)) {
      this.knowledge.set(userId, []);
    }

    const entries = this.knowledge.get(userId)!;

    const entry: KnowledgeEntry = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      userId,
      category,
      content,
      source,
      confidence,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    entries.push(entry);
  }

  getKnowledgeByCategory(userId: string, category: string): KnowledgeEntry[] {
    const entries = this.knowledge.get(userId) || [];
    return entries
      .filter((e) => e.category === category)
      .sort((a, b) => b.confidence - a.confidence);
  }

  getAllKnowledge(userId: string): KnowledgeEntry[] {
    return this.knowledge.get(userId) || [];
  }

  updateConfidence(entryId: string, newConfidence: number): void {
    for (const entries of this.knowledge.values()) {
      const entry = entries.find((e) => e.id === entryId);
      if (entry) {
        entry.confidence = newConfidence;
        entry.updatedAt = Date.now();
        break;
      }
    }
  }

  searchKnowledge(userId: string, query: string): KnowledgeEntry[] {
    const entries = this.knowledge.get(userId) || [];
    const queryLower = query.toLowerCase();

    return entries
      .filter((e) => e.content.toLowerCase().includes(queryLower))
      .sort((a, b) => b.confidence - a.confidence);
  }
}

const sharedKB = new SharedKnowledgeBase();

// Node that extracts and stores knowledge
conversationGraph.addNode('extractKnowledge', async (state) => {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  // Ask model to extract learnable facts
  const extractionPrompt = `
Based on this conversation, extract any facts or preferences about the user.
Format as JSON array: [{"category": "preference|fact|skill", "content": "the fact"}]

Conversation:
${JSON.stringify(state.messages)}
  `;

  const extraction = await model.invoke([{ role: 'user', content: extractionPrompt }]);

  try {
    const facts = JSON.parse(extraction.content as string);

    for (const fact of facts) {
      sharedKB.addKnowledge(
        state.userId,
        fact.category,
        fact.content,
        state.threadId,
        0.8
      );
    }
  } catch (error) {
    console.error('Failed to extract knowledge:', error);
  }

  return state;
});

// Node that retrieves relevant knowledge
conversationGraph.addNode('retrieveKnowledge', async (state) => {
  const lastMessage = state.messages[state.messages.length - 1];
  const query = typeof lastMessage?.content === 'string' ? lastMessage.content : '';

  const relevantKnowledge = sharedKB.searchKnowledge(state.userId, query);

  const knowledgeContext = relevantKnowledge
    .slice(0, 5)
    .map((k) => `- [${k.category}] ${k.content} (confidence: ${k.confidence})`)
    .join('\n');

  return {
    userContext: `${state.userContext}\n\nRelevant Knowledge:\n${knowledgeContext}`,
  };
});
```

---

## Continuation

Document continues with remaining v1.0+ features:

- **Tools State Updates** - Tools can directly update graph state
- **Command Tool** - Dynamic agent flows
- **LangGraph Templates** - Common agentic use cases

---

**End of Core v1.0+ Features** - Additional documentation in subsequent files.

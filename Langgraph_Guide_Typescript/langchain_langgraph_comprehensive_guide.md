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

## Continuation Notice

This document continues with comprehensive coverage of remaining sections. Due to length constraints, I'm providing the structure and will complete implementation in the next sections.

### Remaining Sections to be Covered

7. **Model Context Protocol (MCP)** - Custom MCP server integration
8. **Agentic Patterns** - ReAct, Plan-and-Execute, Self-Correction
9. **Memory Systems (LangChain.js)** - All memory types and persistence
10. **State Management (LangGraph.js)** - Advanced state patterns
11. **LangGraph Checkpointing** - All checkpoint implementations
12. **Conditional Logic (LangGraph.js)** - Routing and control flow
13. **Context Engineering** - Prompt optimisation
14. **Retrieval-Augmented Generation (RAG)** - Full RAG implementations
15. **Human-in-the-Loop** - Approval workflows
16. **LangGraph Studio** - Setup and usage
17. **Streaming** - Token streaming and event streaming
18. **Chains and Sequences** - LCEL and composition
19. **Callbacks and Tracing** - LangSmith integration
20. **TypeScript Patterns** - Advanced patterns
21. **Deployment Patterns** - Production deployment
22. **Advanced Topics** - Testing, monitoring, performance

---

**End of Core Sections** - Continue with next document for remaining content.


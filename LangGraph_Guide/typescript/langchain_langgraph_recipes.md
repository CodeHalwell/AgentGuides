# LangChain.js and LangGraph.js Recipes

**Production-Ready Code Examples and Practical Implementations**

---

## Table of Contents

1. [Basic Chatbot](#basic-chatbot)
2. [Research Agent](#research-agent)
3. [Document Analysis Agent](#document-analysis-agent)
4. [Multi-Agent Supervisor](#multi-agent-supervisor)
5. [RAG Chatbot](#rag-chatbot)
6. [Code Review Agent](#code-review-agent)
7. [Data Analysis Pipeline](#data-analysis-pipeline)
8. [Human-in-the-Loop Approval](#human-in-the-loop-approval)
9. [Streaming Chat API](#streaming-chat-api)
10. [Error Recovery Agent](#error-recovery-agent)

---

## Basic Chatbot

### Simple Conversational Chatbot with Memory

```typescript
// recipes/basicChatbot.ts
import { ChatOpenAI } from '@langchain/openai';
import { BufferMemory } from '@langchain/core/memory';
import { ConversationChain } from '@langchain/core/chains';
import { PromptTemplate } from '@langchain/core/prompts';
import { HumanMessage } from '@langchain/core/messages';

export async function basicChatbot(): Promise<void> {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0.7,
  });

  const memory = new BufferMemory({
    memoryKey: 'history',
    inputKey: 'input',
  });

  const template = `You are a helpful and friendly assistant.

Previous conversation:
{history}

Current input: {input}`;

  const prompt = new PromptTemplate({
    inputVariables: ['history', 'input'],
    template,
  });

  const chain = new ConversationChain({
    llm: model,
    memory,
    prompt,
  });

  // Run conversation
  const conversations = [
    'What is TypeScript?',
    'How does it differ from JavaScript?',
    'Can you give me an example of using types?',
  ];

  for (const userInput of conversations) {
    console.log(`\nUser: ${userInput}`);
    const response = await chain.call({ input: userInput });
    console.log(`Assistant: ${response.response}`);
  }
}

// Usage
// basicChatbot().catch(console.error);
```

### LangGraph-based Chatbot with Stateful Messages

```typescript
// recipes/langraphChatbot.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { HumanMessage, AIMessage, BaseMessage } from '@langchain/core/messages';

const ChatStateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

export async function langgraphChatbot(): Promise<void> {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0.7,
  });

  const graph = new StateGraph(ChatStateAnnotation);

  graph.addNode('chatNode', async (state) => {
    const response = await model.invoke(state.messages);
    return {
      messages: [response],
    };
  });

  graph.addEdge(START, 'chatNode');
  graph.addEdge('chatNode', END);

  const compiled = graph.compile();

  const messages: BaseMessage[] = [];

  const userInputs = [
    'Hello, how are you?',
    'Tell me about LangChain',
  ];

  for (const userInput of userInputs) {
    messages.push(new HumanMessage(userInput));
    const result = await compiled.invoke({ messages });
    
    const lastMessage = result.messages[result.messages.length - 1];
    console.log(`User: ${userInput}`);
    console.log(`Assistant: ${lastMessage.content}`);
    
    messages.push(...result.messages);
  }
}
```

---

## Research Agent

### Multi-step Research with Tool Usage

```typescript
// recipes/researchAgent.ts
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

export async function researchAgent(): Promise<void> {
  // Define research tools
  const searchWikipedia = tool(
    {
      name: 'search_wikipedia',
      description: 'Search Wikipedia for information about a topic',
      schema: z.object({
        query: z.string().describe('Search query'),
      }),
    },
    async ({ query }) => {
      // Simulate Wikipedia search
      return `Wikipedia results for "${query}": [Mock search results]`;
    }
  );

  const synthesizeInformation = tool(
    {
      name: 'synthesise_information',
      description: 'Combine information from multiple sources',
      schema: z.object({
        sources: z.array(z.string()).describe('Information sources'),
        topic: z.string().describe('Topic to synthesise'),
      }),
    },
    async ({ sources, topic }) => {
      return `Synthesised report on ${topic}: [Summary combining all sources]`;
    }
  );

  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const agent = createReactAgent({
    llm: model,
    tools: [searchWikipedia, synthesizeInformation],
  });

  const result = await agent.invoke({
    input: 'Research the history of TypeScript and provide a comprehensive report',
  });

  console.log('Research Result:', result.output);
}
```

---

## Document Analysis Agent

### Process and Analyse Documents

```typescript
// recipes/documentAnalysisAgent.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';

interface DocumentAnalysisState {
  document: string;
  extractedEntities: string[];
  summary: string;
  sentiment: 'positive' | 'negative' | 'neutral';
}

const DocumentStateAnnotation = Annotation.Root({
  document: Annotation<string>,
  extractedEntities: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
  summary: Annotation<string>({ default: () => '' }),
  sentiment: Annotation<'positive' | 'negative' | 'neutral'>({
    default: () => 'neutral',
  }),
});

export async function documentAnalysisAgent(): Promise<void> {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const graph = new StateGraph(DocumentStateAnnotation);

  // Entity extraction node
  graph.addNode('extractEntities', async (state) => {
    const prompt = `Extract named entities (people, organisations, locations) from this text:\n\n${state.document}`;
    const response = await model.invoke(prompt);
    return {
      extractedEntities: [response.content as string],
    };
  });

  // Summarisation node
  graph.addNode('summarise', async (state) => {
    const prompt = `Summarise this document in 2-3 sentences:\n\n${state.document}`;
    const response = await model.invoke(prompt);
    return {
      summary: response.content as string,
    };
  });

  // Sentiment analysis node
  graph.addNode('analyseSentiment', async (state) => {
    const prompt = `Analyse the sentiment of this text. Respond with only: positive, negative, or neutral.\n\n${state.document}`;
    const response = await model.invoke(prompt);
    const sentiment = (response.content as string).toLowerCase().trim() as any;
    return {
      sentiment: ['positive', 'negative', 'neutral'].includes(sentiment) 
        ? sentiment 
        : 'neutral',
    };
  });

  graph.addEdge(START, 'extractEntities');
  graph.addEdge('extractEntities', 'summarise');
  graph.addEdge('summarise', 'analyseSentiment');
  graph.addEdge('analyseSentiment', END);

  const compiled = graph.compile();

  const sampleDocument = `
    Apple Inc. announced today that Tim Cook will continue as CEO.
    The company reported strong Q4 earnings, with revenue exceeding expectations.
    Microsoft and Google are also showing strong performance in the market.
  `;

  const result = await compiled.invoke({
    document: sampleDocument,
    extractedEntities: [],
    summary: '',
    sentiment: 'neutral',
  });

  console.log('Document Analysis Results:');
  console.log('Entities:', result.extractedEntities);
  console.log('Summary:', result.summary);
  console.log('Sentiment:', result.sentiment);
}
```

---

## Multi-Agent Supervisor

### Coordinated Multi-Agent System

```typescript
// recipes/multiAgentSupervisor.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';

interface SupervisorState {
  task: string;
  agentResults: Record<string, string>;
  supervisorDecision: string;
  finalOutput: string;
}

const SupervisorStateAnnotation = Annotation.Root({
  task: Annotation<string>,
  agentResults: Annotation<Record<string, string>>({
    default: () => ({}),
    reducer: (x, y) => ({ ...x, ...y }),
  }),
  supervisorDecision: Annotation<string>({ default: () => '' }),
  finalOutput: Annotation<string>({ default: () => '' }),
});

export async function multiAgentSupervisor(): Promise<void> {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const graph = new StateGraph(SupervisorStateAnnotation);

  // Supervisor node
  graph.addNode('supervisor', async (state) => {
    const coordinationPrompt = `
      You are a supervisor managing multiple agents.
      Task: ${state.task}
      
      Current agent results: ${JSON.stringify(state.agentResults)}
      
      Decide which agent should work next or if we should finish.
      Respond with: AGENT_RESEARCH, AGENT_ANALYSIS, AGENT_WRITING, or COMPLETE
    `;

    const response = await model.invoke(coordinationPrompt);
    return {
      supervisorDecision: (response.content as string).trim(),
    };
  });

  // Research agent node
  graph.addNode('researchAgent', async (state) => {
    const response = await model.invoke(`Research this topic: ${state.task}`);
    return {
      agentResults: {
        research: response.content as string,
      },
    };
  });

  // Analysis agent node
  graph.addNode('analysisAgent', async (state) => {
    const research = state.agentResults.research || '';
    const response = await model.invoke(`Analyse this research: ${research}`);
    return {
      agentResults: {
        analysis: response.content as string,
      },
    };
  });

  // Writing agent node
  graph.addNode('writingAgent', async (state) => {
    const research = state.agentResults.research || '';
    const analysis = state.agentResults.analysis || '';
    const response = await model.invoke(
      `Write a summary combining this research and analysis:\nResearch: ${research}\nAnalysis: ${analysis}`
    );
    return {
      finalOutput: response.content as string,
    };
  });

  // Edge routing
  graph.addEdge(START, 'supervisor');

  graph.addConditionalEdges(
    'supervisor',
    (state) => {
      if (state.supervisorDecision.includes('RESEARCH')) {
        return 'researchAgent';
      } else if (state.supervisorDecision.includes('ANALYSIS')) {
        return 'analysisAgent';
      } else if (state.supervisorDecision.includes('WRITING')) {
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

  graph.addEdge('researchAgent', 'supervisor');
  graph.addEdge('analysisAgent', 'supervisor');
  graph.addEdge('writingAgent', 'supervisor');

  const compiled = graph.compile();

  const result = await compiled.invoke({
    task: 'Provide a comprehensive analysis of artificial intelligence trends in 2024',
    agentResults: {},
    supervisorDecision: '',
    finalOutput: '',
  });

  console.log('Final Output:', result.finalOutput);
}
```

---

## RAG Chatbot

### Retrieval-Augmented Generation Implementation

```typescript
// recipes/ragChatbot.ts
import { ChatOpenAI } from '@langchain/openai';
import { MemoryVectorStore } from '@langchain/core/vectorstores';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';

export async function ragChatbot(): Promise<void> {
  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = new MemoryVectorStore(embeddings);

  // Add sample documents to the vector store
  const documents = [
    new Document({
      pageContent: 'TypeScript is a typed superset of JavaScript',
      metadata: { source: 'typescript-docs' },
    }),
    new Document({
      pageContent: 'LangChain provides tools for building LLM applications',
      metadata: { source: 'langchain-docs' },
    }),
    new Document({
      pageContent: 'LangGraph enables orchestration of multi-agent systems',
      metadata: { source: 'langgraph-docs' },
    }),
  ];

  await vectorStore.addDocuments(documents);

  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  // RAG function
  async function ragQuery(query: string): Promise<string> {
    // Retrieve relevant documents
    const relevantDocs = await vectorStore.similaritySearch(query, 2);

    // Format context from documents
    const context = relevantDocs
      .map((doc) => doc.pageContent)
      .join('\n');

    // Create augmented prompt
    const augmentedPrompt = `
      Context from documents:
      ${context}
      
      Question: ${query}
      
      Provide an answer based on the provided context.
    `;

    const response = await model.invoke(augmentedPrompt);
    return response.content as string;
  }

  // Test RAG
  const question = 'What is TypeScript?';
  const answer = await ragQuery(question);

  console.log(`Question: ${question}`);
  console.log(`Answer: ${answer}`);
}
```

---

## Streaming Chat API

### Real-time Streaming Responses

```typescript
// recipes/streamingAPI.ts
import express from 'express';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';

export function setupStreamingAPI(): express.Application {
  const app = express();
  app.use(express.json());

  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const agent = createReactAgent({
    llm: model,
    tools: [],
  });

  app.post('/stream', async (req, res) => {
    const { message } = req.body;

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    try {
      // Stream from agent
      const stream = await agent.stream({ input: message });

      for await (const chunk of stream) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }

      res.write('data: [DONE]\n\n');
      res.end();
    } catch (error) {
      res.write(
        `data: ${JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' })}\n\n`
      );
      res.end();
    }
  });

  return app;
}
```

---

## Error Recovery Agent

### Robust Error Handling and Retry Logic

```typescript
// recipes/errorRecoveryAgent.ts
import { ChatOpenAI } from '@langchain/openai';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
}

export async function errorRecoveryAgent(): Promise<void> {
  // Unreliable tool that sometimes fails
  const unreliableTool = tool(
    {
      name: 'unreliable_operation',
      description: 'Performs an operation that may fail',
      schema: z.object({
        operation: z.string(),
      }),
    },
    async ({ operation }) => {
      const random = Math.random();
      if (random < 0.5) {
        throw new Error('Random operation failure');
      }
      return `Successfully completed: ${operation}`;
    }
  );

  // Error recovery wrapper
  async function executeWithRecovery<T>(
    fn: () => Promise<T>,
    config: RetryConfig
  ): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= config.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        console.log(`Attempt ${attempt} failed: ${lastError.message}`);

        if (attempt < config.maxRetries) {
          const delay = Math.min(
            config.initialDelay * Math.pow(2, attempt - 1),
            config.maxDelay
          );

          console.log(`Retrying in ${delay}ms...`);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw new Error(`Failed after ${config.maxRetries} attempts: ${lastError?.message}`);
  }

  // Usage
  const config: RetryConfig = {
    maxRetries: 3,
    initialDelay: 100,
    maxDelay: 5000,
  };

  try {
    const result = await executeWithRecovery(
      async () => {
        // Simulate operation
        return 'Success';
      },
      config
    );

    console.log('Result:', result);
  } catch (error) {
    console.error('Operation failed:', error);
  }
}
```

---

## Advanced Streaming Agent (v0.3+)

### Real-Time Progress Tracking with Type-Safe Streaming

```typescript
// recipes/advancedStreamingAgent.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import express from 'express';

interface AnalysisState {
  documentUrl: string;
  downloadStatus: string | null;
  processStatus: string | null;
  analysisResult: string | null;
  error: string | null;
}

const AnalysisStateAnnotation = Annotation.Root({
  documentUrl: Annotation<string>,
  downloadStatus: Annotation<string | null>({ default: () => null }),
  processStatus: Annotation<string | null>({ default: () => null }),
  analysisResult: Annotation<string | null>({ default: () => null }),
  error: Annotation<string | null>({ default: () => null }),
});

const analysisGraph = new StateGraph(AnalysisStateAnnotation);

analysisGraph
  .addNode('download', async (state) => {
    // Simulate document download
    await new Promise((resolve) => setTimeout(resolve, 1000));
    return {
      downloadStatus: `Downloaded from ${state.documentUrl}`,
    };
  })
  .addNode('process', async (state) => {
    // Simulate processing
    await new Promise((resolve) => setTimeout(resolve, 1500));
    return {
      processStatus: 'Document processed successfully',
    };
  })
  .addNode('analyze', async (state) => {
    const model = new ChatOpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    const response = await model.invoke(
      `Analyze this document: ${state.processStatus}`
    );

    return {
      analysisResult: response.content as string,
    };
  });

analysisGraph.addEdge(START, 'download');
analysisGraph.addEdge('download', 'process');
analysisGraph.addEdge('process', 'analyze');
analysisGraph.addEdge('analyze', END);

const analysisWorkflow = analysisGraph.compile();

// Express endpoint with Server-Sent Events
const app = express();
app.use(express.json());

app.post('/analyze', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const { documentUrl } = req.body;

  try {
    const stream = analysisWorkflow.stream(
      {
        documentUrl,
        downloadStatus: null,
        processStatus: null,
        analysisResult: null,
        error: null,
      },
      {
        streamMode: 'updates',
      }
    );

    for await (const update of stream) {
      const nodeName = Object.keys(update)[0];
      const nodeData = update[nodeName];

      res.write(
        `data: ${JSON.stringify({
          node: nodeName,
          update: nodeData,
          timestamp: new Date().toISOString(),
        })}\n\n`
      );
    }

    res.write('data: {"done": true}\n\n');
    res.end();
  } catch (error) {
    res.write(
      `data: ${JSON.stringify({
        error: error instanceof Error ? error.message : 'Unknown error',
      })}\n\n`
    );
    res.end();
  }
});

export default app;
```

---

## Cached RAG Pipeline

### Implementing Node Caching for Expensive Embeddings

```typescript
// recipes/cachedRagPipeline.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from '@langchain/core/vectorstores';
import { Document } from '@langchain/core/documents';
import Redis from 'ioredis';

interface RAGState {
  query: string;
  embedding: number[] | null;
  retrievedDocs: Document[];
  answer: string | null;
}

const RAGStateAnnotation = Annotation.Root({
  query: Annotation<string>,
  embedding: Annotation<number[] | null>({ default: () => null }),
  retrievedDocs: Annotation<Document[]>({ default: () => [] }),
  answer: Annotation<string | null>({ default: () => null }),
});

// Redis cache for embeddings
const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');

async function getCachedEmbedding(text: string): Promise<number[] | null> {
  const cached = await redis.get(`embedding:${text}`);
  return cached ? JSON.parse(cached) : null;
}

async function cacheEmbedding(text: string, embedding: number[]): Promise<void> {
  await redis.setex(
    `embedding:${text}`,
    3600, // 1 hour TTL
    JSON.stringify(embedding)
  );
}

const ragGraph = new StateGraph(RAGStateAnnotation);

// Cached embedding generation node
ragGraph.addNode('generateEmbedding', async (state) => {
  // Check cache first
  const cached = await getCachedEmbedding(state.query);
  if (cached) {
    console.log('Using cached embedding');
    return { embedding: cached };
  }

  console.log('Generating new embedding');
  const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const embedding = await embeddings.embedQuery(state.query);

  // Cache for future use
  await cacheEmbedding(state.query, embedding);

  return { embedding };
});

// Retrieve relevant documents
ragGraph.addNode('retrieve', async (state) => {
  // Simulate vector store search
  const vectorStore = new MemoryVectorStore(
    new OpenAIEmbeddings({
      apiKey: process.env.OPENAI_API_KEY,
    })
  );

  const docs = await vectorStore.similaritySearch(state.query, 3);

  return { retrievedDocs: docs };
});

// Generate answer
ragGraph.addNode('generateAnswer', async (state) => {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const context = state.retrievedDocs
    .map((doc) => doc.pageContent)
    .join('\n\n');

  const response = await model.invoke(
    `Context:\n${context}\n\nQuestion: ${state.query}\n\nAnswer:`
  );

  return { answer: response.content as string };
});

ragGraph.addEdge(START, 'generateEmbedding');
ragGraph.addEdge('generateEmbedding', 'retrieve');
ragGraph.addEdge('retrieve', 'generateAnswer');
ragGraph.addEdge('generateAnswer', END);

export const cachedRAGWorkflow = ragGraph.compile();
```

---

## Parallel Data Processing with Deferred Aggregation

### Fan-Out/Fan-In Pattern for Scalable Processing

```typescript
// recipes/parallelDataProcessing.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';

interface DataBatch {
  id: string;
  data: any[];
}

interface ParallelProcessingState {
  batches: DataBatch[];
  processedBatches: Map<string, any>;
  aggregatedResult: any | null;
  errors: Array<{ batchId: string; error: string }>;
}

const ParallelStateAnnotation = Annotation.Root({
  batches: Annotation<DataBatch[]>,
  processedBatches: Annotation<Map<string, any>>({
    default: () => new Map(),
  }),
  aggregatedResult: Annotation<any | null>({ default: () => null }),
  errors: Annotation<Array<{ batchId: string; error: string }>>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

function createProcessingGraph() {
  const graph = new StateGraph(ParallelStateAnnotation);

  // Create dynamic processing nodes for each batch
  const createBatchProcessor = (batchIndex: number) => {
    return async (state: ParallelProcessingState) => {
      const batch = state.batches[batchIndex];

      try {
        // Simulate processing
        await new Promise((resolve) => setTimeout(resolve, 1000));

        const processed = batch.data.map((item) => ({
          ...item,
          processed: true,
          timestamp: Date.now(),
        }));

        const newMap = new Map(state.processedBatches);
        newMap.set(batch.id, processed);

        return {
          processedBatches: newMap,
        };
      } catch (error) {
        return {
          errors: [{
            batchId: batch.id,
            error: error instanceof Error ? error.message : 'Unknown error',
          }],
        };
      }
    };
  };

  // Add batch processing nodes
  graph.addNode('batch0', createBatchProcessor(0));
  graph.addNode('batch1', createBatchProcessor(1));
  graph.addNode('batch2', createBatchProcessor(2));

  // Deferred aggregation node
  graph.addNode(
    'aggregate',
    async (state) => {
      const allProcessed = Array.from(state.processedBatches.values()).flat();

      return {
        aggregatedResult: {
          totalItems: allProcessed.length,
          successful: allProcessed.filter((item) => item.processed).length,
          errors: state.errors.length,
          data: allProcessed,
        },
      };
    },
    {
      deferred: true, // Wait for all batch processors
    }
  );

  // Fan-out: Start all batch processors in parallel
  graph.addEdge(START, 'batch0');
  graph.addEdge(START, 'batch1');
  graph.addEdge(START, 'batch2');

  // Fan-in: All processors feed into aggregator
  graph.addEdge('batch0', 'aggregate');
  graph.addEdge('batch1', 'aggregate');
  graph.addEdge('batch2', 'aggregate');

  graph.addEdge('aggregate', END);

  return graph.compile();
}

export const parallelProcessor = createProcessingGraph();

// Usage
async function processLargeDatasset(dataset: any[]): Promise<any> {
  const batchSize = Math.ceil(dataset.length / 3);

  const batches: DataBatch[] = [
    { id: 'batch0', data: dataset.slice(0, batchSize) },
    { id: 'batch1', data: dataset.slice(batchSize, batchSize * 2) },
    { id: 'batch2', data: dataset.slice(batchSize * 2) },
  ];

  const result = await parallelProcessor.invoke({
    batches,
    processedBatches: new Map(),
    aggregatedResult: null,
    errors: [],
  });

  return result.aggregatedResult;
}
```

---

## Content Moderation Pipeline with Pre/Post Hooks

### Implementing Safety Checks Around Model Calls

```typescript
// recipes/moderatedChatbot.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';

interface ModeratedChatState {
  userMessage: string;
  moderatedInput: string | null;
  aiResponse: string | null;
  moderatedOutput: string | null;
  flags: string[];
}

const ModeratedStateAnnotation = Annotation.Root({
  userMessage: Annotation<string>,
  moderatedInput: Annotation<string | null>({ default: () => null }),
  aiResponse: Annotation<string | null>({ default: () => null }),
  moderatedOutput: Annotation<string | null>({ default: () => null }),
  flags: Annotation<string[]>({
    default: () => [],
    reducer: (x, y) => [...x, ...y],
  }),
});

const moderatedGraph = new StateGraph(ModeratedStateAnnotation);

// Pre-hook: Input moderation
moderatedGraph.addNode('moderateInput', async (state) => {
  const sensitivePatterns = [
    /\b\d{3}-\d{2}-\d{4}\b/, // SSN
    /\b\d{16}\b/, // Credit card
    /\b(password|api[_-]?key)\s*[:=]/i,
  ];

  const flags: string[] = [];
  let moderatedInput = state.userMessage;

  for (const pattern of sensitivePatterns) {
    if (pattern.test(state.userMessage)) {
      flags.push('sensitive-data-detected');
      moderatedInput = state.userMessage.replace(pattern, '[REDACTED]');
    }
  }

  if (flags.length > 0) {
    console.warn('Input moderation flags:', flags);
  }

  return {
    moderatedInput,
    flags,
  };
});

// Model invocation
moderatedGraph.addNode('generateResponse', async (state) => {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const response = await model.invoke(state.moderatedInput || state.userMessage);

  return {
    aiResponse: response.content as string,
  };
});

// Post-hook: Output moderation
moderatedGraph.addNode('moderateOutput', async (state) => {
  let moderatedOutput = state.aiResponse || '';

  // Check for leaked sensitive info
  const sensitivePatterns = [
    /\b\d{3}-\d{2}-\d{4}\b/,
    /\b\d{16}\b/,
  ];

  const flags: string[] = [];

  for (const pattern of sensitivePatterns) {
    if (pattern.test(moderatedOutput)) {
      flags.push('output-contains-sensitive-data');
      moderatedOutput =
        'I apologize, but I cannot provide that information as it may contain sensitive data.';
    }
  }

  return {
    moderatedOutput,
    flags,
  };
});

moderatedGraph.addEdge(START, 'moderateInput');
moderatedGraph.addEdge('moderateInput', 'generateResponse');
moderatedGraph.addEdge('generateResponse', 'moderateOutput');
moderatedGraph.addEdge('moderateOutput', END);

export const moderatedChatbot = moderatedGraph.compile();
```

---

## Cross-Thread Memory Multi-User Support

### Building User Context Across Conversations

```typescript
// recipes/multiUserMemorySystem.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { Pool } from 'pg';

interface UserConversationState {
  userId: string;
  threadId: string;
  message: string;
  userProfile: Record<string, any>;
  conversationHistory: string[];
  response: string | null;
}

const UserConversationAnnotation = Annotation.Root({
  userId: Annotation<string>,
  threadId: Annotation<string>,
  message: Annotation<string>,
  userProfile: Annotation<Record<string, any>>({ default: () => ({}) }),
  conversationHistory: Annotation<string[]>({ default: () => [] }),
  response: Annotation<string | null>({ default: () => null }),
});

class MultiUserMemorySystem {
  private pool: Pool;

  constructor(connectionString: string) {
    this.pool = new Pool({ connectionString });
  }

  async getUserProfile(userId: string): Promise<Record<string, any>> {
    const result = await this.pool.query(
      'SELECT profile FROM user_profiles WHERE user_id = $1',
      [userId]
    );

    return result.rows[0]?.profile || {};
  }

  async getConversationHistory(
    userId: string,
    threadId: string
  ): Promise<string[]> {
    const result = await this.pool.query(
      `SELECT summary FROM conversation_summaries
       WHERE user_id = $1 AND thread_id != $2
       ORDER BY created_at DESC LIMIT 5`,
      [userId, threadId]
    );

    return result.rows.map((row) => row.summary);
  }

  async saveConversation(
    userId: string,
    threadId: string,
    message: string,
    response: string
  ): Promise<void> {
    await this.pool.query(
      `INSERT INTO conversation_summaries (user_id, thread_id, summary)
       VALUES ($1, $2, $3)`,
      [userId, threadId, `User: ${message}\nAssistant: ${response}`]
    );
  }

  async updateUserProfile(
    userId: string,
    updates: Record<string, any>
  ): Promise<void> {
    await this.pool.query(
      `INSERT INTO user_profiles (user_id, profile)
       VALUES ($1, $2)
       ON CONFLICT (user_id)
       DO UPDATE SET profile = user_profiles.profile || EXCLUDED.profile`,
      [userId, JSON.stringify(updates)]
    );
  }
}

const memorySystem = new MultiUserMemorySystem(process.env.DATABASE_URL!);

const conversationGraph = new StateGraph(UserConversationAnnotation);

conversationGraph.addNode('loadUserContext', async (state) => {
  const [profile, history] = await Promise.all([
    memorySystem.getUserProfile(state.userId),
    memorySystem.getConversationHistory(state.userId, state.threadId),
  ]);

  return {
    userProfile: profile,
    conversationHistory: history,
  };
});

conversationGraph.addNode('generateResponse', async (state) => {
  const model = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const context = `
User Profile: ${JSON.stringify(state.userProfile, null, 2)}

Previous Conversations:
${state.conversationHistory.join('\n\n')}

Current Message: ${state.message}
  `.trim();

  const response = await model.invoke(context);

  return {
    response: response.content as string,
  };
});

conversationGraph.addNode('saveContext', async (state) => {
  await memorySystem.saveConversation(
    state.userId,
    state.threadId,
    state.message,
    state.response!
  );

  // Extract and save preferences if mentioned
  if (state.message.toLowerCase().includes('prefer')) {
    await memorySystem.updateUserProfile(state.userId, {
      lastPreferenceUpdate: new Date().toISOString(),
    });
  }

  return state;
});

conversationGraph.addEdge(START, 'loadUserContext');
conversationGraph.addEdge('loadUserContext', 'generateResponse');
conversationGraph.addEdge('generateResponse', 'saveContext');
conversationGraph.addEdge('saveContext', END);

export const multiUserConversation = conversationGraph.compile();
```

---

## Production Monitoring Dashboard

### Complete Observability for LangGraph Workflows

```typescript
// recipes/monitoringDashboard.ts
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';
import express from 'express';

interface MetricsState {
  requestId: string;
  operation: string;
  startTime: number;
  endTime: number | null;
  duration: number | null;
  success: boolean;
  error: string | null;
}

class MonitoringDashboard {
  private metrics: MetricsState[] = [];
  private app: express.Application;

  constructor() {
    this.app = express();
    this.setupRoutes();
  }

  recordMetric(metric: MetricsState): void {
    this.metrics.push(metric);

    // Keep only last 1000 metrics
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-1000);
    }
  }

  getMetrics(): {
    total: number;
    successful: number;
    failed: number;
    avgDuration: number;
    recentErrors: Array<{ requestId: string; error: string }>;
  } {
    const successful = this.metrics.filter((m) => m.success);
    const failed = this.metrics.filter((m) => !m.success);

    const totalDuration = this.metrics
      .filter((m) => m.duration !== null)
      .reduce((sum, m) => sum + (m.duration || 0), 0);

    return {
      total: this.metrics.length,
      successful: successful.length,
      failed: failed.length,
      avgDuration:
        this.metrics.length > 0 ? totalDuration / this.metrics.length : 0,
      recentErrors: failed
        .slice(-10)
        .map((m) => ({ requestId: m.requestId, error: m.error || 'Unknown' })),
    };
  }

  private setupRoutes(): void {
    this.app.get('/metrics', (req, res) => {
      res.json(this.getMetrics());
    });

    this.app.get('/metrics/detailed', (req, res) => {
      res.json(this.metrics.slice(-100));
    });
  }

  listen(port: number): void {
    this.app.listen(port, () => {
      console.log(`Monitoring dashboard running on port ${port}`);
    });
  }
}

export const dashboard = new MonitoringDashboard();

// Middleware to track all workflow executions
export function createMonitoredWorkflow<T>(
  workflow: any,
  operationName: string
) {
  return async (input: T): Promise<any> => {
    const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const startTime = Date.now();

    try {
      const result = await workflow.invoke(input);
      const endTime = Date.now();

      dashboard.recordMetric({
        requestId,
        operation: operationName,
        startTime,
        endTime,
        duration: endTime - startTime,
        success: true,
        error: null,
      });

      return result;
    } catch (error) {
      const endTime = Date.now();

      dashboard.recordMetric({
        requestId,
        operation: operationName,
        startTime,
        endTime,
        duration: endTime - startTime,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      throw error;
    }
  };
}
```

---

These recipes provide practical, production-ready implementations that you can adapt to your specific use cases. Each recipe demonstrates key patterns and best practices for building with LangChain.js and LangGraph.js, including the latest v1.0+ and v0.3+ features such as type-safe streaming, node caching, deferred nodes, model hooks, and cross-thread memory support.
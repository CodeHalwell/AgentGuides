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

These recipes provide practical, production-ready implementations that you can adapt to your specific use cases. Each recipe demonstrates key patterns and best practices for building with LangChain.js and LangGraph.js.

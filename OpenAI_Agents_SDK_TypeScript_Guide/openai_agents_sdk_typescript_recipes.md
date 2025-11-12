# OpenAI Agents SDK TypeScript: Practical Recipes & Code Examples

**Version:** 1.0  
**Focus:** Ready-to-use implementations, real-world patterns, copy-paste recipes

---

## Table of Contents

1. [Basic Agent Recipes](#basic-agent-recipes)
2. [Multi-Agent Workflows](#multi-agent-workflows)
3. [Data Processing](#data-processing)
4. [Customer Service](#customer-service)
5. [Content Generation](#content-generation)
6. [Research & Analysis](#research--analysis)
7. [Integration Patterns](#integration-patterns)
8. [Advanced Orchestration](#advanced-orchestration)

---

## Basic Agent Recipes

### Recipe 1: Simple Q&A Agent

```typescript
import { Agent, run } from '@openai/agents';

async function createQAAgent() {
  const qaAgent = new Agent({
    name: 'Q&A Assistant',
    instructions: `You are a knowledgeable Q&A assistant. 
      - Answer questions accurately and concisely
      - Provide relevant examples when helpful
      - Acknowledge when you're uncertain`,
  });

  const question = 'What are the benefits of TypeScript?';
  const result = await run(qaAgent, question);
  
  console.log(`Q: ${question}`);
  console.log(`A: ${result.finalOutput}`);
  
  return result.finalOutput;
}

// Run: createQAAgent();
```

### Recipe 2: Translation Agent with Multiple Languages

```typescript
import { Agent, run } from '@openai/agents';

async function translateWithAgent(text: string, targetLanguage: string) {
  const translator = new Agent({
    name: 'Translator',
    instructions: `You are an expert translator. Translate the provided text to ${targetLanguage} while:
      - Preserving tone and context
      - Maintaining technical accuracy
      - Handling idioms appropriately`,
  });

  const result = await run(translator, `Translate to ${targetLanguage}: "${text}"`);
  return result.finalOutput;
}

// Usage
const english = 'Hello, how can I help you today?';
const spanish = await translateWithAgent(english, 'Spanish');
const french = await translateWithAgent(english, 'French');
const german = await translateWithAgent(english, 'German');

console.log(`Spanish: ${spanish}`);
console.log(`French: ${french}`);
console.log(`German: ${german}`);
```

### Recipe 3: Content Classification Agent

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

async function classifyContentAgent() {
  const classificationTool = tool({
    name: 'classify_content',
    description: 'Classify content into categories',
    parameters: z.object({
      text: z.string(),
    }),
    execute: async ({ text }) => {
      // Return classifications
      return {
        category: 'technology',
        sentiment: 'positive',
        isSpam: false,
      };
    },
  });

  const classifier = new Agent({
    name: 'Content Classifier',
    instructions: 'Classify user-provided content into appropriate categories.',
    tools: [classificationTool],
  });

  const content = 'This is an amazing new AI framework!';
  const result = await run(classifier, `Classify: ${content}`);
  
  return result.finalOutput;
}
```

---

## Multi-Agent Workflows

### Recipe 4: Research & Summary Workflow

```typescript
import { Agent, run } from '@openai/agents';

async function researchAndSummariseWorkflow(topic: string) {
  // Researcher agent
  const researcher = new Agent({
    name: 'Researcher',
    instructions: `Research and gather information about the given topic.
      - Find key facts and recent developments
      - Identify main concepts
      - Note authoritative sources`,
  });

  // Summariser agent
  const summariser = new Agent({
    name: 'Summariser',
    instructions: `Create a concise summary of research findings.
      - Extract key points
      - Keep it under 500 words
      - Organise logically`,
  });

  // Step 1: Research
  console.log('Step 1: Researching...');
  const researchResult = await run(researcher, `Research: ${topic}`);
  console.log('Research findings:', researchResult.finalOutput.substring(0, 200) + '...');

  // Step 2: Summarise
  console.log('\nStep 2: Summarising...');
  const summaryResult = await run(
    summariser,
    `Summarise these research findings about ${topic}:\n${researchResult.finalOutput}`
  );
  console.log('Summary:', summaryResult.finalOutput);

  return summaryResult.finalOutput;
}

// Usage
await researchAndSummariseWorkflow('Quantum Computing');
```

### Recipe 5: Customer Support Routing

```typescript
import { Agent, run } from '@openai/agents';

async function customerSupportRouter(customerQuery: string) {
  // Department agents
  const billingAgent = new Agent({
    name: 'Billing Support',
    instructions: 'Handle billing and payment inquiries professionally.',
  });

  const technicalAgent = new Agent({
    name: 'Technical Support',
    instructions: 'Resolve technical issues step-by-step.',
  });

  const generalAgent = new Agent({
    name: 'General Support',
    instructions: 'Provide general assistance and information.',
  });

  // Routing agent
  const routerAgent = Agent.create({
    name: 'Support Router',
    instructions: `Route customer inquiries to the appropriate department:
      - Billing issues → Billing Support
      - Technical problems → Technical Support
      - Everything else → General Support`,
    handoffs: [billingAgent, technicalAgent, generalAgent],
  });

  const result = await run(routerAgent, customerQuery);
  
  console.log(`Query: ${customerQuery}`);
  console.log(`Routed to: ${result.currentAgent?.name || 'General'}`);
  console.log(`Response: ${result.finalOutput}`);

  return result;
}

// Usage examples
await customerSupportRouter('Why was I charged twice?');
await customerSupportRouter('The app keeps crashing on startup');
await customerSupportRouter('How do I reset my password?');
```

### Recipe 6: Parallel Processing Pipeline

```typescript
import { Agent, run } from '@openai/agents';

async function parallelProcessingPipeline(document: string) {
  // Create specialized agents
  const sentimentAnalyser = new Agent({
    name: 'Sentiment Analyser',
    instructions: 'Analyse the sentiment of the provided text.',
  });

  const keywordExtractor = new Agent({
    name: 'Keyword Extractor',
    instructions: 'Extract key topics and keywords from the text.',
  });

  const readabilityAnalyser = new Agent({
    name: 'Readability Analyser',
    instructions: 'Analyse readability and suggest improvements.',
  });

  // Execute in parallel
  const [sentiment, keywords, readability] = await Promise.all([
    run(sentimentAnalyser, `Analyse sentiment: ${document}`),
    run(keywordExtractor, `Extract keywords: ${document}`),
    run(readabilityAnalyser, `Analyse readability: ${document}`),
  ]);

  return {
    sentiment: sentiment.finalOutput,
    keywords: keywords.finalOutput,
    readability: readability.finalOutput,
  };
}

// Usage
const analysis = await parallelProcessingPipeline('Your document text here...');
console.log('Analysis Results:', analysis);
```

---

## Data Processing

### Recipe 7: Data Validation Agent

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

async function dataValidationAgent(jsonData: string) {
  const validateTool = tool({
    name: 'validate_data',
    description: 'Validate data structure and content',
    parameters: z.object({
      data: z.string(),
      rules: z.string(),
    }),
    execute: async ({ data, rules }) => {
      try {
        const parsed = JSON.parse(data);
        return { valid: true, data: parsed };
      } catch {
        return { valid: false, error: 'Invalid JSON' };
      }
    },
  });

  const validator = new Agent({
    name: 'Data Validator',
    instructions: `Validate provided data against requirements.
      - Check structure
      - Verify types
      - Ensure completeness`,
    tools: [validateTool],
  });

  const result = await run(validator, `Validate this data: ${jsonData}`);
  return result.finalOutput;
}

// Usage
const testData = JSON.stringify({
  name: 'John Doe',
  email: 'john@example.com',
  age: 30,
});

await dataValidationAgent(testData);
```

### Recipe 8: CSV to Structured Format

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

async function csvToStructuredAgent(csvData: string) {
  const parseTool = tool({
    name: 'parse_csv',
    description: 'Parse CSV data',
    parameters: z.object({
      csv: z.string(),
    }),
    execute: async ({ csv }) => {
      const lines = csv.trim().split('\n');
      const headers = lines[0].split(',');
      const rows = lines.slice(1).map(line =>
        Object.fromEntries(
          headers.map((header, idx) => [header.trim(), line.split(',')[idx]?.trim()])
        )
      );
      return { headers, rows, count: rows.length };
    },
  });

  const parser = new Agent({
    name: 'CSV Parser',
    instructions: 'Convert CSV data to structured format.',
    tools: [parseTool],
  });

  const result = await run(parser, `Parse this CSV:\n${csvData}`);
  return result.finalOutput;
}

// Usage
const csv = `name,email,department
Alice Johnson,alice@company.com,Engineering
Bob Smith,bob@company.com,Sales
Carol White,carol@company.com,HR`;

await csvToStructuredAgent(csv);
```

---

## Customer Service

### Recipe 9: FAQ System with Agent

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

const faqDatabase = {
  'refund-policy': 'Refunds accepted within 30 days of purchase...',
  'shipping-time': 'Standard shipping takes 5-7 business days...',
  'warranty': 'All products come with a 1-year warranty...',
};

async function faqAgent(userQuestion: string) {
  const lookupFAQTool = tool({
    name: 'lookup_faq',
    description: 'Look up answers in FAQ database',
    parameters: z.object({
      query: z.string(),
    }),
    execute: async ({ query }) => {
      const key = Object.keys(faqDatabase).find(k =>
        query.toLowerCase().includes(k.replace('-', ' '))
      );
      return key ? { found: true, answer: faqDatabase[key as keyof typeof faqDatabase] } : { found: false };
    },
  });

  const faqAssistant = new Agent({
    name: 'FAQ Assistant',
    instructions: 'Answer customer questions using the FAQ database when available.',
    tools: [lookupFAQTool],
  });

  const result = await run(faqAssistant, userQuestion);
  return result.finalOutput;
}

// Usage
await faqAgent('What is your refund policy?');
await faqAgent('How long does shipping take?');
await faqAgent('Do you offer warranty?');
```

### Recipe 10: Appointment Scheduling Assistant

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

const appointments: Array<{ id: string; time: string; client: string }> = [];

async function appointmentSchedulingAgent(userRequest: string) {
  const scheduleTool = tool({
    name: 'schedule_appointment',
    description: 'Schedule an appointment',
    parameters: z.object({
      clientName: z.string(),
      time: z.string(),
      reason: z.string(),
    }),
    execute: async ({ clientName, time, reason }) => {
      const id = `apt-${Date.now()}`;
      appointments.push({ id, time, client: clientName });
      return { success: true, appointmentId: id, confirmationTime: time };
    },
  });

  const checkAvailabilityTool = tool({
    name: 'check_availability',
    description: 'Check available time slots',
    parameters: z.object({
      date: z.string(),
    }),
    execute: async ({ date }) => {
      // Mock availability
      return {
        available: ['09:00', '10:00', '14:00', '15:00', '16:00'],
      };
    },
  });

  const scheduler = new Agent({
    name: 'Appointment Scheduler',
    instructions: 'Help customers schedule appointments by finding available slots and confirming.',
    tools: [scheduleTool, checkAvailabilityTool],
  });

  const result = await run(scheduler, userRequest);
  return result.finalOutput;
}

// Usage
await appointmentSchedulingAgent('I need to schedule a consultation for next Tuesday morning');
```

---

## Content Generation

### Recipe 11: Blog Post Generator

```typescript
import { Agent, run } from '@openai/agents';

async function blogPostGenerator(topic: string, wordCount: number = 1000) {
  const outlineAgent = new Agent({
    name: 'Outline Creator',
    instructions: 'Create detailed blog post outlines with clear sections.',
  });

  const writerAgent = new Agent({
    name: 'Blog Writer',
    instructions: 'Write engaging blog posts with SEO in mind.',
  });

  // Step 1: Create outline
  const outlineResult = await run(outlineAgent, `Create an outline for a ${wordCount}-word blog post about: ${topic}`);

  // Step 2: Write blog post
  const blogResult = await run(
    writerAgent,
    `Write a ${wordCount}-word blog post about ${topic} using this outline:\n${outlineResult.finalOutput}`
  );

  return {
    topic,
    outline: outlineResult.finalOutput,
    content: blogResult.finalOutput,
  };
}

// Usage
const blog = await blogPostGenerator('The Future of Artificial Intelligence', 1500);
console.log('Blog Post:', blog.content);
```

### Recipe 12: Social Media Content Creator

```typescript
import { Agent, run } from '@openai/agents';

async function socialMediaContentCreator(topic: string) {
  const tweetAgent = new Agent({
    name: 'Tweet Creator',
    instructions: 'Create engaging tweets within 280 characters.',
  });

  const linkedinAgent = new Agent({
    name: 'LinkedIn Strategist',
    instructions: 'Create professional LinkedIn posts.',
  });

  const instagramAgent = new Agent({
    name: 'Instagram Specialist',
    instructions: 'Create creative Instagram captions.',
  });

  const [tweet, linkedin, instagram] = await Promise.all([
    run(tweetAgent, `Create a tweet about: ${topic}`),
    run(linkedinAgent, `Create a LinkedIn post about: ${topic}`),
    run(instagramAgent, `Create an Instagram caption about: ${topic}`),
  ]);

  return {
    twitter: tweet.finalOutput,
    linkedin: linkedin.finalOutput,
    instagram: instagram.finalOutput,
  };
}

// Usage
const content = await socialMediaContentCreator('Sustainable Technology');
console.log('Social Media Content:', content);
```

---

## Research & Analysis

### Recipe 13: Market Analysis Agent

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

async function marketAnalysisAgent(industry: string) {
  const trendTool = tool({
    name: 'research_trends',
    description: 'Research market trends',
    parameters: z.object({
      industry: z.string(),
      period: z.string(),
    }),
    execute: async ({ industry }) => ({
      trends: ['AI adoption', 'Cost optimisation', 'Sustainability'],
    }),
  });

  const competitorTool = tool({
    name: 'analyse_competitors',
    description: 'Analyse competitor landscape',
    parameters: z.object({
      industry: z.string(),
    }),
    execute: async ({ industry }) => ({
      topCompetitors: ['Company A', 'Company B', 'Company C'],
    }),
  });

  const analyst = new Agent({
    name: 'Market Analyst',
    instructions: 'Provide comprehensive market analysis.',
    tools: [trendTool, competitorTool],
  });

  const result = await run(analyst, `Analyse the ${industry} market for investment opportunities`);
  return result.finalOutput;
}

// Usage
await marketAnalysisAgent('Technology');
```

### Recipe 14: Code Review Agent

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

async function codeReviewAgent(code: string) {
  const analyseCodeTool = tool({
    name: 'analyse_code',
    description: 'Analyse code quality',
    parameters: z.object({
      code: z.string(),
    }),
    execute: async ({ code }) => ({
      issues: ['Missing error handling', 'Potential performance issue'],
      score: 7.5,
    }),
  });

  const reviewer = new Agent({
    name: 'Code Reviewer',
    instructions: `Review code for:
      - Best practices
      - Performance issues
      - Security vulnerabilities
      - Readability improvements`,
    tools: [analyseCodeTool],
  });

  const result = await run(reviewer, `Review this code:\n\`\`\`typescript\n${code}\n\`\`\``);
  return result.finalOutput;
}

// Usage
const sampleCode = `
function calculateTotal(items) {
  let total = 0;
  for (let i = 0; i < items.length; i++) {
    total += items[i].price;
  }
  return total;
}
`;

await codeReviewAgent(sampleCode);
```

---

## Integration Patterns

### Recipe 15: Webhook Handler with Agent

```typescript
import { Agent, run } from '@openai/agents';
import express from 'express';

const app = express();
app.use(express.json());

const eventProcessor = new Agent({
  name: 'Event Processor',
  instructions: 'Process and respond to incoming events.',
});

app.post('/webhook', async (req, res) => {
  const { eventType, data } = req.body;

  try {
    const result = await run(
      eventProcessor,
      `Process this event: ${eventType}\nData: ${JSON.stringify(data)}`
    );

    res.json({
      success: true,
      response: result.finalOutput,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log('Webhook server ready'));
```

### Recipe 16: Scheduled Agent Tasks

```typescript
import { Agent, run } from '@openai/agents';
import cron from 'node-cron';

const reportGenerator = new Agent({
  name: 'Report Generator',
  instructions: 'Generate daily reports and summaries.',
});

// Run every day at 8 AM
cron.schedule('0 8 * * *', async () => {
  console.log('Generating daily report...');
  const result = await run(reportGenerator, 'Generate today\'s summary report');
  console.log('Report:', result.finalOutput);
});

// Run every hour
cron.schedule('0 * * * *', async () => {
  console.log('Running hourly task...');
  const result = await run(reportGenerator, 'Check system health');
  console.log('Health check:', result.finalOutput);
});
```

---

## Advanced Orchestration

### Recipe 17: Complex Workflow with Conditions

```typescript
import { Agent, run } from '@openai/agents';

async function complexWorkflow(input: string) {
  const analyserAgent = new Agent({
    name: 'Analyser',
    instructions: 'Analyse input and determine next steps.',
  });

  const escalationAgent = new Agent({
    name: 'Escalator',
    instructions: 'Handle urgent matters.',
  });

  const routineAgent = new Agent({
    name: 'Routine Handler',
    instructions: 'Handle routine tasks.',
  });

  // Step 1: Analyse
  const analysis = await run(analyserAgent, `Analyse: ${input}`);

  // Step 2: Route based on analysis
  let result;
  if (analysis.finalOutput.includes('urgent') || analysis.finalOutput.includes('critical')) {
    result = await run(escalationAgent, input);
  } else {
    result = await run(routineAgent, input);
  }

  return result.finalOutput;
}

// Usage
await complexWorkflow('Server is down!');
await complexWorkflow('Update documentation');
```

### Recipe 18: Feedback Loop with Refinement

```typescript
import { Agent, run } from '@openai/agents';

async function refinementWorkflow(initialQuery: string, maxIterations = 3) {
  const analyser = new Agent({
    name: 'Analyser',
    instructions: 'Analyse and critique responses.',
  });

  const writer = new Agent({
    name: 'Writer',
    instructions: 'Create and refine content.',
  });

  let current = initialQuery;
  let iteration = 0;

  while (iteration < maxIterations) {
    iteration++;
    console.log(`Iteration ${iteration}`);

    // Generate content
    const writeResult = await run(writer, current);
    console.log('Content:', writeResult.finalOutput.substring(0, 100) + '...');

    // Get feedback
    const feedback = await run(
      analyser,
      `Critique this response and suggest improvements:\n${writeResult.finalOutput}`
    );

    if (feedback.finalOutput.includes('excellent') || feedback.finalOutput.includes('perfect')) {
      console.log('Content approved!');
      return writeResult.finalOutput;
    }

    // Prepare for next iteration
    current = `Improve based on this feedback:\n${feedback.finalOutput}`;
  }

  return 'Final version after refinement';
}

// Usage
await refinementWorkflow('Write an introduction to machine learning');
```

---

These recipes provide practical, copy-paste implementations for common use cases with the OpenAI Agents SDK.

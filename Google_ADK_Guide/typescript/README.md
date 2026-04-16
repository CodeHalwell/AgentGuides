# Google Agent Development Kit (ADK) for TypeScript

**Status:** Available (announced December 2025)
**Package:** `@google/adk` (npm)
**Official Repository:** https://github.com/google/adk-js
**Documentation:** https://google.github.io/adk-docs/get-started/typescript/

---

## Overview

Google ADK for TypeScript brings the same capabilities and developer experience as ADK for Python to the TypeScript/JavaScript ecosystem. It was officially announced in December 2025, making ADK a multi-language framework supporting **Python, TypeScript, Go, and Java**.

ADK for TypeScript uses the same features, development UI (`adk web`), and interface as the Python ADK — TypeScript developers can apply best practices like version control, automated testing, and CI/CD integration directly to their agent code.

---

## Installation

```bash
npm install @google/adk
```

```bash
# Or with Yarn
yarn add @google/adk
```

---

## Quick Start

### Hello World Agent

```typescript
import { LlmAgent } from '@google/adk/agents';
import { GoogleSearch } from '@google/adk/tools';
import { Runner } from '@google/adk';

// Create a simple agent
const agent = new LlmAgent({
  name: 'MyAssistant',
  model: 'gemini-2.0-flash',
  description: 'A helpful assistant',
  instruction: 'You are a helpful AI assistant.',
  tools: [new GoogleSearch()]
});

// Run the agent
const runner = new Runner({ agent });
const response = await runner.runAsync({
  userId: 'user_123',
  sessionId: 'session_001',
  newMessage: 'What is the latest news about AI?'
});

for await (const event of response) {
  if (event.isFinalResponse()) {
    console.log(event.content?.parts?.[0]?.text);
  }
}
```

---

## Core Concepts

### Agents

Agents are the primary building blocks. Define agent behavior using TypeScript classes:

```typescript
import { LlmAgent } from '@google/adk/agents';

const researchAgent = new LlmAgent({
  name: 'ResearchAgent',
  model: 'gemini-2.0-flash',
  instruction: `You are a research assistant. 
  Use available tools to gather accurate information.`,
  tools: []
});
```

### Tools

Define custom tools using TypeScript functions with typed parameters:

```typescript
import { FunctionTool } from '@google/adk/tools';
import { z } from 'zod';

const weatherTool = new FunctionTool({
  name: 'get_weather',
  description: 'Get current weather for a location',
  schema: z.object({
    location: z.string().describe('City name or coordinates'),
    units: z.enum(['celsius', 'fahrenheit']).optional()
  }),
  handler: async ({ location, units = 'celsius' }) => {
    // Implementation
    return { temperature: 22, condition: 'sunny', location };
  }
});
```

### Multi-Agent Systems

Compose agents into hierarchical systems:

```typescript
import { LlmAgent } from '@google/adk/agents';
import { AgentTool } from '@google/adk/tools';

// Specialist agents
const dataAgent = new LlmAgent({
  name: 'DataAgent',
  model: 'gemini-2.0-flash',
  instruction: 'Analyze data and provide insights.'
});

const reportAgent = new LlmAgent({
  name: 'ReportAgent',
  model: 'gemini-2.0-flash',
  instruction: 'Generate professional reports from data insights.'
});

// Coordinator agent delegates to specialists
const coordinatorAgent = new LlmAgent({
  name: 'Coordinator',
  model: 'gemini-2.0-flash',
  instruction: 'Coordinate data analysis and reporting tasks.',
  tools: [
    new AgentTool({ agent: dataAgent }),
    new AgentTool({ agent: reportAgent })
  ]
});
```

### Sessions and State

```typescript
import { InMemorySessionService } from '@google/adk/sessions';
import { Runner } from '@google/adk';

const sessionService = new InMemorySessionService();
const runner = new Runner({
  agent,
  sessionService
});

// Sessions persist conversation context
const session = await sessionService.createSession({
  appName: 'my-app',
  userId: 'user_123'
});
```

---

## Streaming

ADK for TypeScript supports real-time streaming of agent responses:

```typescript
const stream = await runner.runAsync({ userId, sessionId, newMessage });

for await (const event of stream) {
  if (event.content?.parts) {
    for (const part of event.content.parts) {
      if (part.text) {
        process.stdout.write(part.text);
      }
    }
  }
}
```

---

## MCP Integration

Connect any MCP-compatible service:

```typescript
import { MCPToolset } from '@google/adk/tools';

const mcpTools = new MCPToolset({
  serverParams: {
    command: 'npx',
    args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/workspace']
  }
});

const agent = new LlmAgent({
  name: 'FileAgent',
  model: 'gemini-2.0-flash',
  instruction: 'Help with file operations.',
  tools: [mcpTools]
});
```

---

## A2A Protocol

ADK TypeScript supports Agent-to-Agent (A2A) 1.0 for cross-framework communication:

```typescript
import { A2AServer } from '@google/adk/a2a';

const server = new A2AServer({
  agent,
  port: 8080,
  capabilities: { streaming: true }
});

await server.start();
```

---

## Development Tools

### ADK Web UI

```bash
# Start the development UI
npx @google/adk web
```

The web UI provides:
- Agent testing and debugging
- Workflow graph visualization
- Session replay and inspection

### ADK CLI

```bash
# Run an agent
npx @google/adk run agent.ts

# Deploy to Vertex AI Agent Engine
npx @google/adk deploy --target vertex-ai
```

---

## Differences from Python ADK

| Feature | Python ADK | TypeScript ADK |
|---------|------------|----------------|
| Same agent primitives | ✅ | ✅ |
| Same tool patterns | ✅ | ✅ |
| MCP integration | ✅ | ✅ |
| A2A protocol | ✅ | ✅ |
| Vertex AI integration | ✅ | ✅ |
| Cloud Run deployment | ✅ | ✅ (via `adk deploy`) |
| Type-safe schema | Pydantic | Zod |
| Async runtime | asyncio | Node.js EventEmitter |

---

## Production Deployment

### Cloud Run (Node.js)

```typescript
import express from 'express';
import { Runner, HttpHandler } from '@google/adk';

const app = express();
const runner = new Runner({ agent });

app.post('/run', HttpHandler.fromRunner(runner));

app.listen(process.env.PORT || 8080, () => {
  console.log('Agent server running');
});
```

### Vertex AI Agent Engine

```bash
# Deploy via CLI
npx @google/adk deploy \
  --target vertex-ai \
  --project my-gcp-project \
  --region us-central1
```

---

## Resources

- **Official Docs:** https://google.github.io/adk-docs/get-started/typescript/
- **GitHub:** https://github.com/google/adk-js
- **npm:** https://www.npmjs.com/package/@google/adk
- **Blog Post:** https://developers.googleblog.com/introducing-agent-development-kit-for-typescript-build-ai-agents-with-the-power-of-a-code-first-approach/

---

## 📋 Revision History

| Date | Version | Changes |
|------|---------|---------|
| April 16, 2026 | Initial | New TypeScript ADK guide created; TypeScript support announced December 2025 |

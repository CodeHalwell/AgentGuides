---
title: "Strands Agents TypeScript SDK"
description: "Strands Agents for TypeScript on Amazon Bedrock: agents, Zod-typed tools, model providers, structured output, streaming, hooks, sessions, multi-agent graphs and swarms, and MCP."
framework: "amazon-bedrock-agents"
language: "typescript"
---

The TypeScript SDK for Strands Agents (`@strands-agents/sdk`) is the Node.js counterpart to the Python `strands-agents` package: the same model-driven agent loop, with tools defined by Zod schemas and fully typed inputs. It runs on Amazon Bedrock by default and also supports Anthropic, OpenAI, Google and Vercel AI SDK providers.

> **Verified against:** `@strands-agents/sdk` %%version=@strands-agents/sdk%% (checked %%versions-checked%%). Every example on this page compiles under `tsc --strict` against that installed package. Node.js 22 or later is required.

The Python version of this material is the [Strands SDK guide](/amazon-bedrock-agents-guide/bedrock_strands_sdk_guide/). For hosting a Strands agent on AgentCore Runtime, see the [AgentCore TypeScript guide](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/).

---

## Table of Contents

1. [Installation](#installation)
2. [Agents](#agents)
3. [Model providers](#model-providers)
4. [Tools](#tools)
5. [Structured output](#structured-output)
6. [Streaming](#streaming)
7. [Hooks and plugins](#hooks-and-plugins)
8. [Conversation history and sessions](#conversation-history-and-sessions)
9. [Multi-agent patterns](#multi-agent-patterns)
10. [MCP tools](#mcp-tools)
11. [Differences from the Python SDK](#differences-from-the-python-sdk)

---

## Installation

```bash
npm install @strands-agents/sdk@%%version=@strands-agents/sdk%% zod
npm install --save-dev typescript @types/node
```

The package is ESM-only. Use `"type": "module"` in `package.json` and `"module": "NodeNext"` (or `"ESNext"` with a bundler) in `tsconfig.json`; the examples below use top-level `await`.

Several providers and integrations are optional peer dependencies, installed only when you use them:

| Feature | Import path | Extra package |
|---------|-------------|---------------|
| Amazon Bedrock | `@strands-agents/sdk` | none (bundled) |
| Anthropic API | `@strands-agents/sdk/models/anthropic` | `@anthropic-ai/sdk` |
| OpenAI | `@strands-agents/sdk/models/openai` | `openai` |
| Google Gemini | `@strands-agents/sdk/models/google` | `@google/genai` |
| MCP clients | `@strands-agents/sdk` (`McpClient`) | `@modelcontextprotocol/client` |
| S3 session storage | `@strands-agents/sdk/storage` | `@aws-sdk/client-s3` |

AWS credentials come from the standard AWS SDK for JavaScript chain (environment variables, shared config, SSO, instance or task role).

---

## Agents

An `Agent` combines a model, a system prompt and tools. `invoke()` runs the agent loop to completion and returns an `AgentResult`; `toString()` gives the final text.

```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'

const agent = new Agent({
  model: new BedrockModel({
    modelId: 'us.anthropic.claude-sonnet-5-5', // cross-Region inference profile
    region: 'us-east-1',
    maxTokens: 4096,
  }),
  systemPrompt: 'You are a concise assistant for AWS architects.',
  printer: false, // the default (true) also writes the response to stdout
})

const result = await agent.invoke('In two sentences, what is Amazon Bedrock AgentCore?')

console.log(result.toString())
console.log(result.stopReason) // e.g. 'endTurn'
console.log(agent.messages.length) // conversation history is kept on the agent
```

Points worth knowing:

- **Model shorthand.** `model` also accepts a string, which becomes `new BedrockModel({ modelId })`. With no model at all, the SDK uses a default Bedrock model and logs a warning that the default may change between releases; set `modelId` explicitly.
- **State.** `agent.messages` holds the conversation. `appState` is durable key-value state that tools and hooks can read; `invocationState` (passed to `invoke`) is per-call state that is not stored.
- **Concurrency.** An agent handles one invocation at a time; a second concurrent call throws `ConcurrentInvocationError`. Create one agent per conversation or session.
- **Limits and cancellation.** `invoke(prompt, { limits: { turns: 5 }, cancelSignal })` bounds the loop; `agent.cancel()` stops an in-flight call.

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent({
  model: 'us.anthropic.claude-haiku-5-5',
  appState: { tenantId: 'acme' },
  printer: false,
})

const controller = new AbortController()
setTimeout(() => controller.abort(), 60_000)

const result = await agent.invoke('Summarise our support policy in three bullet points.', {
  invocationState: { requestId: 'req-123' },
  cancelSignal: controller.signal,
  limits: { turns: 5 },
})

console.log(result.toString())
console.log(agent.appState.get('tenantId')) // 'acme'
```

---

## Model providers

### Amazon Bedrock

`BedrockModel` wraps the Bedrock Runtime Converse API. Model IDs are Bedrock model IDs or inference profile IDs; the Claude 5.5 family is normally called through a cross-Region profile such as `us.anthropic.claude-sonnet-5-5` (or the `global.` prefix where available).

```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'

const model = new BedrockModel({
  modelId: 'us.anthropic.claude-sonnet-5-5',
  region: 'us-west-2',
  maxTokens: 8192,
  temperature: 0.2,
  // Prompt caching for the system prompt and tool definitions
  cacheConfig: { strategy: 'auto', systemPromptTTL: true, toolsTTL: true },
  // Optional Bedrock Guardrail
  guardrailConfig: {
    guardrailIdentifier: 'gr-abc123',
    guardrailVersion: '1',
    trace: 'enabled',
  },
})

const agent = new Agent({ model, printer: false })
console.log((await agent.invoke('Hello')).toString())
```

Other options include `topP`, `stopSequences`, `additionalRequestFields` (for model-specific request fields), `clientConfig` (passed to the AWS SDK `BedrockRuntimeClient`), `requestTimeout` and `apiKey` (Bedrock API keys instead of SigV4). `model.updateConfig({ ... })` changes settings between calls.

### Anthropic, OpenAI and others

Non-Bedrock providers live under subpath imports so that their SDKs stay optional:

```typescript
import { Agent } from '@strands-agents/sdk'
import { AnthropicModel } from '@strands-agents/sdk/models/anthropic'
import { OpenAIModel } from '@strands-agents/sdk/models/openai'

const claude = new AnthropicModel({
  apiKey: process.env.ANTHROPIC_API_KEY,
  modelId: 'claude-sonnet-5-5',
  maxTokens: 4096,
})

const gpt = new OpenAIModel({
  api: 'chat', // or 'responses'
  apiKey: process.env.OPENAI_API_KEY,
})

const agentA = new Agent({ model: claude, printer: false })
const agentB = new Agent({ model: gpt, printer: false })
```

`@strands-agents/sdk/models/google` (`GoogleModel`) and `@strands-agents/sdk/models/vercel` (any Vercel AI SDK language model) follow the same pattern. A custom provider extends the abstract `Model` class and implements `stream()`, `getConfig()` and `updateConfig()`.

---

## Tools

### Zod tools

`tool()` takes a Zod schema; the callback's `input` is inferred from it, and the model's arguments are validated before your code runs. The return value can be a string, number, object or array; it is serialised into the tool result.

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const getOrderStatus = tool({
  name: 'get_order_status',
  description: 'Look up the shipping status of an order by its ID.',
  inputSchema: z.object({
    orderId: z.string().describe('Order ID, e.g. ORD-1042'),
    includeHistory: z.boolean().default(false).describe('Include past status changes'),
  }),
  callback: async ({ orderId, includeHistory }) => {
    // input is typed as { orderId: string; includeHistory: boolean }
    const status = { orderId, status: 'shipped', carrier: 'DHL' }
    return includeHistory ? { ...status, history: ['created', 'packed', 'shipped'] } : status
  },
})

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  tools: [getOrderStatus],
  printer: false,
})

console.log((await agent.invoke('Where is order ORD-1042?')).toString())
```

If the callback throws, the error is returned to the model as an error tool result rather than ending the run, so the model can recover or explain.

### Tool context and state

The callback's second argument is a `ToolContext` with the calling agent, the tool-use block and the invocation state:

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import type { ToolContext } from '@strands-agents/sdk'
import { z } from 'zod'

interface AppState {
  tenantId: string
}

const listInvoices = tool({
  name: 'list_invoices',
  description: 'List recent invoices for the current tenant.',
  inputSchema: z.object({ limit: z.number().int().min(1).max(50).default(10) }),
  callback: async (input, context?: ToolContext) => {
    if (!context) throw new Error('Tool context is required')
    const tenantId = context.agent.appState.get<AppState>('tenantId') // durable state
    const requestId = context.invocationState.requestId as string | undefined // per call
    return { tenantId, requestId, toolUseId: context.toolUse.toolUseId, invoices: [], limit: input.limit }
  },
})

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  appState: { tenantId: 'acme' },
  tools: [listInvoices],
  printer: false,
})

await agent.invoke('Show my last five invoices.', { invocationState: { requestId: 'req-42' } })
```

### Streaming tools and JSON Schema tools

A callback can be an async generator: each `yield` is surfaced as a `toolStreamUpdateEvent` in `agent.stream()`, and the `return` value is the tool result. If you prefer plain JSON Schema, `FunctionTool` takes one directly (inputs are then `unknown`).

```typescript
import { FunctionTool, tool } from '@strands-agents/sdk'
import { z } from 'zod'

export const exportReport = tool({
  name: 'export_report',
  description: 'Export a report and report progress.',
  inputSchema: z.object({ reportId: z.string() }),
  callback: async function* (input): AsyncGenerator<string, string, unknown> {
    yield 'Collecting data...'
    yield 'Rendering PDF...'
    return `s3://reports/${input.reportId}.pdf`
  },
})

export const ping = new FunctionTool({
  name: 'ping',
  description: 'Check that a host name resolves.',
  inputSchema: {
    type: 'object',
    properties: { host: { type: 'string' } },
    required: ['host'],
  },
  callback: (input: unknown) => {
    const { host } = input as { host: string }
    return `ok: ${host}`
  },
})
```

### Vended tools

The SDK ships optional ready-made tools under `@strands-agents/sdk/vended-tools/*`, including `notebook`, `fileEditor` (`file-editor`), `httpRequest` (`http-request`), `bash`, `shell`, `python-repl`, `web-fetch` and `subagent`. They are not loaded unless imported. Shell, file and code tools execute on the host running the agent: give them a sandbox or a locked-down container in production.

```typescript
import { Agent } from '@strands-agents/sdk'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  tools: [httpRequest, notebook],
  printer: false,
})
```

---

## Structured output

Pass a Zod schema as `structuredOutputSchema` (on the agent or per call). The agent validates the model's answer against it and retries with the validation errors if needed; if validation still fails it throws `StructuredOutputError`. `result.structuredOutput` is typed as `unknown`, so parse it with the same schema to get a typed value.

```typescript
import { Agent, StructuredOutputError } from '@strands-agents/sdk'
import { z } from 'zod'

const Ticket = z.object({
  category: z.enum(['billing', 'technical', 'account']),
  priority: z.enum(['low', 'medium', 'high']),
  summary: z.string().max(200),
})

const agent = new Agent({ model: 'us.anthropic.claude-haiku-5-5', printer: false })

try {
  const result = await agent.invoke(
    'Customer says: "I was charged twice this month and need a refund urgently."',
    { structuredOutputSchema: Ticket }
  )
  const ticket = Ticket.parse(result.structuredOutput) // typed as z.infer<typeof Ticket>
  console.log(ticket.category, ticket.priority)
} catch (error) {
  if (error instanceof StructuredOutputError) {
    console.error('Model did not produce a valid ticket:', error.message)
  } else {
    throw error
  }
}
```

---

## Streaming

`agent.stream()` is an async generator of typed events, with the `AgentResult` as its return value. Model tokens arrive wrapped in `modelStreamUpdateEvent`; lifecycle events (`beforeToolCallEvent`, `afterToolCallEvent`, `toolResultEvent` and so on) are interleaved. Narrowing on `type` gives full type information.

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const getTime = tool({
  name: 'get_time',
  description: 'Current UTC time',
  inputSchema: z.object({}),
  callback: () => new Date().toISOString(),
})

const agent = new Agent({ model: 'us.anthropic.claude-sonnet-5-5', tools: [getTime], printer: false })

const stream = agent.stream('What time is it in UTC? Then write a haiku about it.')
let next = await stream.next()
while (!next.done) {
  const event = next.value
  if (
    event.type === 'modelStreamUpdateEvent' &&
    event.event.type === 'modelContentBlockDeltaEvent' &&
    event.event.delta.type === 'textDelta'
  ) {
    process.stdout.write(event.event.delta.text)
  } else if (event.type === 'beforeToolCallEvent') {
    console.log(`\n[tool] ${event.toolUse.name}`)
  }
  next = await stream.next()
}

const result = next.value // AgentResult
console.log('\nstop reason:', result.stopReason)
```

A plain `for await (const event of agent.stream(...))` works too if you do not need the final result.

---

## Hooks and plugins

Hooks are typed callbacks registered against event classes with `agent.addHook()`. Several events are writable: setting `cancel` on `BeforeToolCallEvent` or `BeforeModelCallEvent` blocks the call (a string becomes the message returned to the model), and setting `retry` on `AfterModelCallEvent` or `AfterToolCallEvent` re-runs it.

```typescript
import {
  Agent,
  AfterInvocationEvent,
  AfterToolCallEvent,
  BeforeToolCallEvent,
  tool,
} from '@strands-agents/sdk'
import { z } from 'zod'

const deleteBucket = tool({
  name: 'delete_bucket',
  description: 'Delete an S3 bucket',
  inputSchema: z.object({ bucket: z.string() }),
  callback: ({ bucket }) => `deleted ${bucket}`,
})

const agent = new Agent({ model: 'us.anthropic.claude-sonnet-5-5', tools: [deleteBucket], printer: false })

// Guardrail: block destructive tools for read-only callers
agent.addHook(BeforeToolCallEvent, (event) => {
  const readOnly = event.invocationState.readOnly === true
  if (readOnly && event.toolUse.name.startsWith('delete_')) {
    event.cancel = `Tool ${event.toolUse.name} is not allowed for read-only users.`
  }
})

// Audit log
agent.addHook(AfterToolCallEvent, (event) => {
  console.log(JSON.stringify({ tool: event.toolUse.name, failed: event.error !== undefined }))
})

agent.addHook(AfterInvocationEvent, (event) => {
  console.log(`messages in history: ${event.agent.messages.length}`)
})

await agent.invoke('Delete the bucket called scratch-data.', { invocationState: { readOnly: true } })
```

`addHook()` returns a cleanup function that removes the hook. To package hooks and tools together, implement the `Plugin` interface and pass it in `plugins`:

```typescript
import { Agent, BeforeModelCallEvent } from '@strands-agents/sdk'
import type { LocalAgent, Plugin } from '@strands-agents/sdk'

class ModelCallCounter implements Plugin {
  readonly name = 'model-call-counter'
  count = 0

  initAgent(agent: LocalAgent): void {
    agent.addHook(BeforeModelCallEvent, () => {
      this.count += 1
    })
  }
}

const counter = new ModelCallCounter()
const agent = new Agent({ model: 'us.anthropic.claude-haiku-5-5', plugins: [counter], printer: false })
await agent.invoke('Hi')
console.log(`model calls: ${counter.count}`)
```

The SDK also has a middleware API (`agent.addMiddleware()` with `InvokeModelStage` / `ExecuteToolStage`) for wrapping model and tool calls, and retry strategies (`ModelRetryStrategy`, `ExponentialBackoff`) for throttling.

---

## Conversation history and sessions

By default the agent keeps a sliding window of history and trims it when the model reports a context-window overflow. Configure it explicitly with a conversation manager:

```typescript
import { Agent, SlidingWindowConversationManager } from '@strands-agents/sdk'

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  conversationManager: new SlidingWindowConversationManager({ windowSize: 40 }),
  printer: false,
})
```

`SummarizingConversationManager` summarises older turns instead of dropping them, and `NullConversationManager` disables trimming.

To persist an agent's messages and state between processes, attach a `SessionManager` with a storage backend. `LocalFileStorage` and `S3Storage` come from `@strands-agents/sdk/storage`:

```typescript
import { Agent, SessionManager } from '@strands-agents/sdk'
import { S3Storage } from '@strands-agents/sdk/storage'
import { S3Client } from '@aws-sdk/client-s3'

const session = new SessionManager({
  sessionId: 'user-123',
  storage: new S3Storage('my-agent-sessions', {
    prefix: 'prod/',
    s3Client: new S3Client({ region: 'us-west-2' }),
  }),
})

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  sessionManager: session,
  printer: false,
})

await agent.invoke('My name is Priya.') // restored and saved automatically
```

For long-term memory that is extracted and searched rather than replayed, use AgentCore Memory through `bedrock-agentcore` (see [AgentCore Memory](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/#memory)).

---

## Multi-agent patterns

### Agents as tools

An `Agent` can be placed directly in another agent's `tools`; its `name` and `description` become the tool definition. Use `asTool()` to rename it or to keep its context between calls.

```typescript
import { Agent } from '@strands-agents/sdk'

const researcher = new Agent({
  name: 'researcher',
  description: 'Finds and summarises factual information on a topic.',
  model: 'us.anthropic.claude-sonnet-5-5',
  systemPrompt: 'You research topics and report concise, sourced facts.',
  printer: false,
})

const editor = new Agent({
  name: 'editor',
  description: 'Rewrites draft text for clarity and tone.',
  model: 'us.anthropic.claude-haiku-5-5',
  systemPrompt: 'You edit text to be clear and concise.',
  printer: false,
})

const orchestrator = new Agent({
  model: 'us.anthropic.claude-opus-5-5',
  systemPrompt: 'Use the researcher for facts and the editor to polish the final answer.',
  tools: [researcher, editor.asTool({ name: 'polish_text', preserveContext: true })],
  printer: false,
})

console.log((await orchestrator.invoke('Write a short brief on AWS Graviton for a CFO.')).toString())
```

### Graph

`Graph` runs agents as nodes in a directed graph you define. Nodes are identified by agent `id`; a node runs once all its incoming edges have completed, and independent nodes run in parallel. An edge can carry a `handler` that decides at run time whether it is taken.

```typescript
import { Agent, Graph } from '@strands-agents/sdk'

const model = 'us.anthropic.claude-sonnet-5-5'

const researcher = new Agent({ id: 'researcher', model, systemPrompt: 'Gather key facts.', printer: false })
const analyst = new Agent({ id: 'analyst', model, systemPrompt: 'Analyse the facts for risks.', printer: false })
const writer = new Agent({ id: 'writer', model, systemPrompt: 'Write a one-page summary.', printer: false })

const graph = new Graph({
  nodes: [researcher, analyst, writer],
  edges: [
    ['researcher', 'analyst'],
    ['analyst', 'writer'],
  ],
  maxSteps: 10,
})

const result = await graph.invoke('Assess moving our batch jobs to AWS Fargate.')
console.log(result.status)
for (const node of result.results) {
  console.log(node.nodeId, node.status, `${node.duration}ms`)
}
```

### Swarm

In a `Swarm` the agents decide the routing: each node can hand off to another node or finish. `start` names the entry node and `maxSteps` bounds the number of handoffs.

```typescript
import { Agent, Swarm } from '@strands-agents/sdk'

const model = 'us.anthropic.claude-sonnet-5-5'

const triage = new Agent({
  id: 'triage',
  model,
  description: 'Classifies the request and hands off to the right specialist.',
  systemPrompt: 'Classify the request, then hand off. Do not answer it yourself.',
  printer: false,
})
const billing = new Agent({
  id: 'billing',
  model,
  description: 'Answers billing and invoice questions.',
  systemPrompt: 'Answer billing questions. Do not hand off.',
  printer: false,
})
const technical = new Agent({
  id: 'technical',
  model,
  description: 'Answers technical support questions.',
  systemPrompt: 'Answer technical questions. Do not hand off.',
  printer: false,
})

const swarm = new Swarm({ nodes: [triage, billing, technical], start: 'triage', maxSteps: 4 })

const result = await swarm.invoke('Why was I charged twice in September?')
console.log(result.status, result.results.map((r) => r.nodeId))
```

Both `Graph` and `Swarm` have `stream()` for node-level events, accept a `sessionManager` to persist progress, and are also exported from `@strands-agents/sdk/multiagent` together with their event classes (`BeforeNodeCallEvent`, `MultiAgentHandoffEvent` and so on).

---

## MCP tools

`McpClient` connects to a Model Context Protocol server and exposes its tools. Pass the client itself in `tools`; the agent connects and lists the server's tools when it initialises (on the first `invoke()`, or call `await agent.initialize()` to do it up front). It needs `@modelcontextprotocol/client` installed.

```typescript
import { Agent, McpClient } from '@strands-agents/sdk'
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio'

// Local server over stdio
const awsDocs = new McpClient({
  transport: new StdioClientTransport({
    command: 'uvx',
    args: ['awslabs.aws-documentation-mcp-server@latest'],
  }),
})

// Remote server over Streamable HTTP, with tool names prefixed
const internal = new McpClient({
  url: 'https://mcp.example.com/mcp',
  headers: { Authorization: `Bearer ${process.env.MCP_TOKEN ?? ''}` },
  prefix: 'internal',
})

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  tools: [awsDocs, internal],
  printer: false,
})

try {
  console.log((await agent.invoke('How do I enable S3 Object Lock?')).toString())
} finally {
  await awsDocs.disconnect()
  await internal.disconnect()
}
```

`toolFilters` limits which server tools are exposed, `McpClient.loadServers()` builds clients from an `mcpServers`-style JSON config, and `tasksConfig` enables the MCP tasks extension for long-running tools. An AgentCore Gateway endpoint is an MCP server, so it is reached with the same client and an OAuth bearer token.

---

## Differences from the Python SDK

The TypeScript SDK covers the core of the Python SDK, but some things differ or are missing as of %%version=@strands-agents/sdk%%:

| Python | TypeScript |
|--------|------------|
| `@tool` decorator on a function | `tool({ name, description, inputSchema, callback })` with Zod, or `FunctionTool` with JSON Schema |
| `agent("prompt")` | `await agent.invoke('prompt')` |
| `agent.stream_async()` | `agent.stream()` (async generator, returns `AgentResult`) |
| Callback handlers | No callback-handler API; use `stream()` or hooks |
| `Workflow` pattern / `strands_tools.workflow` | Not in the TypeScript SDK; use `Graph` |
| `strands-agents-tools` package | Smaller set of vended tools under `@strands-agents/sdk/vended-tools/*` |
| `A2AServer` / `A2AAgent` | Available under `@strands-agents/sdk/a2a`, built on `@a2a-js/sdk` 0.3.x |
| `AgentCoreMemorySessionManager` (in `bedrock-agentcore`) | AgentCore Memory is a Strands `MemoryStore` (`AgentCoreMemoryStore`), not a session manager |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | `@strands-agents/sdk` 1.21.0 | New page. All examples compiled with `tsc --strict` against the installed package. |

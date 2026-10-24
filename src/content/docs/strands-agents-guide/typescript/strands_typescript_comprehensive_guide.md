---
title: "Strands Agents (TypeScript): Comprehensive Guide"
description: "The TypeScript Strands Agents SDK from the ground up: project set-up, the agent loop, invoke and stream, model providers, Zod tools, structured output, conversation management, state, sessions, limits and errors."
framework: "strands"
language: "typescript"
sidebar:
  label: "Comprehensive guide"
  order: 1
---

Checked against `@strands-agents/sdk` %%version=@strands-agents/sdk%% (%%versions-checked%%). Every example on this page compiles with `tsc --strict` and was run on Node.js 22 with the Bedrock Runtime client replaced by an offline stub, so the SDK code paths are real and the model replies are canned.

Topic pages: [tools and MCP](/strands-agents-guide/typescript/strands_typescript_tools_and_mcp/), [hooks, interventions and interrupts](/strands-agents-guide/typescript/strands_typescript_hooks_and_interrupts/), [multi-agent and A2A](/strands-agents-guide/typescript/strands_typescript_multi_agent/), [production](/strands-agents-guide/typescript/strands_typescript_production/) and [recipes](/strands-agents-guide/typescript/strands_typescript_recipes/).

## Project set-up

```bash
npm install @strands-agents/sdk@%%version=@strands-agents/sdk%% zod
npm install --save-dev typescript @types/node
```

Node.js 22 or later. The package is ESM-only, so use `"type": "module"` in `package.json` and a NodeNext (or bundler) module setting:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "skipLibCheck": true
  }
}
```

`skipLibCheck` matters: the SDK's type declarations reference optional peer packages (OpenAI, Anthropic, Google, OpenTelemetry, A2A) that you will not all have installed. Install a peer only when you import the matching subpath:

| Feature | Import | Peer package |
|---------|--------|--------------|
| Amazon Bedrock | `@strands-agents/sdk` | none (bundled) |
| Anthropic API | `@strands-agents/sdk/models/anthropic` | `@anthropic-ai/sdk` |
| OpenAI | `@strands-agents/sdk/models/openai` | `openai` |
| Google Gemini | `@strands-agents/sdk/models/google` | `@google/genai` |
| Vercel AI SDK providers | `@strands-agents/sdk/models/vercel` | `@ai-sdk/provider` and a provider package |
| MCP | `@strands-agents/sdk` (`McpClient`) | `@modelcontextprotocol/client` |
| S3 storage | `@strands-agents/sdk/storage` | `@aws-sdk/client-s3` |
| A2A | `@strands-agents/sdk/a2a`, `.../a2a/express` | `@a2a-js/sdk` (0.3.x), `express` |
| OpenTelemetry | `@strands-agents/sdk/telemetry` | `@opentelemetry/*` packages |

AWS credentials come from the standard AWS SDK for JavaScript chain (environment, shared config, SSO, instance or task role). Set `AWS_REGION` or pass `region`.

A first agent:

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const wordCount = tool({
  name: 'word_count',
  description: 'Count the words in a piece of text.',
  inputSchema: z.object({ text: z.string().describe('The text to count') }),
  callback: ({ text }) => text.split(/\s+/).filter(Boolean).length,
})

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%', // Bedrock cross-Region inference profile
  systemPrompt: 'You are a concise assistant.',
  tools: [wordCount],
})

const result = await agent.invoke("How many words are in 'the quick brown fox'?")
console.log(result.stopReason) // 'endTurn'
console.log(result.toString()) // the final text
```

By default the agent also prints its response to stdout (`printer: true`). Pass `printer: false` in servers and tests.

## The agent loop

The loop is the same as in Python: append the input to `agent.messages`, call the model with the history, system prompt and tool specs, run any tools it asks for (concurrently by default), append their results, and repeat until the model answers without tools. There is no fixed iteration cap; set `limits` (below). The [diagrams page](/strands-agents-guide/strands_diagrams/#the-agent-loop) shows where hooks fire.

- A tool that throws produces an error tool result; the model sees it and can recover.
- Throttling and transient model errors are retried by the agent's retry strategy.
- A context-window overflow makes the conversation manager trim history and the call is retried.

### AgentResult

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', printer: false })
const result = await agent.invoke('Name three Yorkshire towns.')

console.log(result.stopReason) // 'endTurn'
console.log(result.toString()) // text of the last message
console.log(result.lastMessage.role) // 'assistant'
console.log(result.metrics?.accumulatedUsage) // token usage
console.log(agent.messages.length) // history is kept on the agent
```

| Field | Meaning |
|-------|---------|
| `stopReason` | `endTurn`, `maxTokens`, `stopSequence`, `guardrailIntervened`, `contentFiltered`, `interrupt`, `cancelled`, `limitTurns`, `limitTotalTokens`, `limitOutputTokens`, `toolUse`, ... (camelCase, unlike Python's snake_case) |
| `lastMessage` | The final assistant `Message` |
| `structuredOutput` | The validated value when you passed a schema |
| `interrupts` | Pending interrupts when `stopReason === 'interrupt'` |
| `metrics` | `AgentMetrics` (usage, cycles, tool metrics) |
| `invocationState` | The per-call state object, after tools and hooks ran |

## The Agent class

| Option | Purpose |
|--------|---------|
| `model` | A `Model`, a Bedrock model ID string, or a `ModelRouter`. Default: a Bedrock model, with a warning |
| `systemPrompt` | String or system content blocks |
| `tools` | Tools, `McpClient`s, or other agents |
| `name`, `description`, `id` | Identity; `id` names graph and swarm nodes and keys sessions |
| `messages` | Initial history |
| `appState` | Durable JSON state (not sent to the model) |
| `printer` | Print the response to stdout (default `true`) |
| `conversationManager` | History trimming (default sliding window) |
| `sessionManager` | Persistence |
| `structuredOutputSchema` | Default Zod schema for every call |
| `plugins`, `interventions` | Extensions and policy handlers |
| `retryStrategy` | Retry strategy, a list of them, or `null` for none |
| `toolExecutor` | `ConcurrentToolExecutor` (default) or `SequentialToolExecutor` |
| `traceAttributes` | Extra OpenTelemetry span attributes |

Hooks are added after construction with `agent.addHook()`, not through a constructor option (see [hooks](/strands-agents-guide/typescript/strands_typescript_hooks_and_interrupts/)).

### invoke and stream

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', printer: false })

// Run to completion
const result = await agent.invoke('Summarise the plot of Hamlet in one sentence.')
console.log(result.toString())

// Stream typed events; the generator's return value is the AgentResult
for await (const event of agent.stream('And Macbeth?')) {
  if (
    event.type === 'modelStreamUpdateEvent' &&
    event.event.type === 'modelContentBlockDeltaEvent' &&
    event.event.delta.type === 'textDelta'
  ) {
    process.stdout.write(event.event.delta.text)
  }
}
```

The prompt can be a string, an array of content blocks (`TextBlock`, `ImageBlock`, `DocumentBlock` or their plain-data forms), messages, or interrupt responses. An image:

```typescript
import { Agent, ImageBlock, TextBlock } from '@strands-agents/sdk'

const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]) // stand-in bytes
const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', printer: false })

const result = await agent.invoke([
  new TextBlock('What does this chart show?'),
  new ImageBlock({ format: 'png', source: { bytes: png } }),
])
console.log(result.toString())
```

An agent runs one invocation at a time; a second concurrent call throws `ConcurrentInvocationError`. Create an agent per conversation.

## Model providers

### Amazon Bedrock

A string `model` becomes `new BedrockModel({ modelId })`. For settings, construct it yourself:

```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'

const model = new BedrockModel({
  modelId: 'us.anthropic.%%model=anthropic%%',
  region: 'eu-west-1',
  maxTokens: 4096,
  temperature: 0.2,
  cacheConfig: { strategy: 'auto' }, // prompt caching for the system prompt and tools
  guardrailConfig: { guardrailIdentifier: 'gr-abc123', guardrailVersion: '1', trace: 'enabled' },
})

const agent = new Agent({ model, printer: false })
model.updateConfig({ temperature: 0 })
console.log((await agent.invoke('Hello')).toString())
```

Model IDs are Bedrock model IDs or cross-Region inference profiles such as `us.anthropic.%%model=anthropic-small%%` (prefixes `us.`, `eu.`, `apac.`, `global.` depending on model and Region). Other options: `topP`, `stopSequences`, `additionalRequestFields`, `clientConfig` (passed to `BedrockRuntimeClient`), `apiKey` (Bedrock API keys), `requestTimeout`.

### Anthropic, OpenAI, Google and the Vercel AI SDK

```typescript
import { Agent } from '@strands-agents/sdk'
import { AnthropicModel } from '@strands-agents/sdk/models/anthropic'
import { GoogleModel } from '@strands-agents/sdk/models/google'
import { OpenAIModel } from '@strands-agents/sdk/models/openai'

const claude = new AnthropicModel({
  apiKey: process.env.ANTHROPIC_API_KEY ?? 'sk-ant-...',
  modelId: '%%model=anthropic%%',
  maxTokens: 4096,
})

const gptResponses = new OpenAIModel({ apiKey: process.env.OPENAI_API_KEY ?? 'sk-...', modelId: '%%model=openai%%' })
const gptChat = new OpenAIModel({ api: 'chat', apiKey: 'sk-...', modelId: '%%model=openai%%' })

// Any OpenAI-compatible server
const local = new OpenAIModel({
  api: 'chat',
  apiKey: 'unused',
  modelId: 'my-model',
  clientConfig: { baseURL: 'http://localhost:8000/v1' },
})

const gemini = new GoogleModel({ apiKey: process.env.GEMINI_API_KEY ?? 'key', modelId: '%%model=gemini%%' })

const agent = new Agent({ model: claude, printer: false })
```

| Class | Import | Notes |
|-------|--------|-------|
| `BedrockModel` | `@strands-agents/sdk` | Converse API |
| `AnthropicModel` | `@strands-agents/sdk/models/anthropic` | `apiKey` or `ANTHROPIC_API_KEY`; `client`/`clientConfig`; `params` |
| `OpenAIModel` | `@strands-agents/sdk/models/openai` | **Defaults to the Responses API**; `api: 'chat'` for Chat Completions. `stateful: true` keeps state server-side (Responses only) |
| `GoogleModel` | `@strands-agents/sdk/models/google` | `apiKey`; `params`; `builtInTools` |
| `VercelModel` | `@strands-agents/sdk/models/vercel` | Wraps any Vercel AI SDK language model (Ollama, Mistral, Groq and others through their AI SDK providers) |

There are no TypeScript equivalents of Python's `LiteLLMModel`, `OllamaModel`, `MistralModel`, `LlamaCppModel`, `SageMakerAIModel` or `WriterModel`; use an OpenAI-compatible endpoint or the Vercel AI SDK provider for those services.

### Routing and fallback

```typescript
import { Agent, BedrockModel, ModelRouter } from '@strands-agents/sdk'

const router = new ModelRouter([
  new BedrockModel({ modelId: 'us.anthropic.%%model=anthropic%%' }),
  new BedrockModel({ modelId: 'eu.anthropic.%%model=anthropic%%', region: 'eu-west-1' }),
])
const agent = new Agent({ model: router, printer: false })
```

`new ModelRouter(models, options)` takes the candidates in order. The default `FallbackStrategy` uses the first model and moves on when a call fails; `ClassifierStrategy` picks per call. `auxModel` moves the SDK's own side calls (summaries, classifiers, judges) to a cheaper model.

### Writing a provider

Extend the abstract `Model` class and implement `stream()` (an async iterable of `ModelStreamEvent`s), `getConfig()` and `updateConfig()`. The [scripted test model](/strands-agents-guide/typescript/strands_typescript_production/#a-scripted-model) is a complete example.

## Tools in brief

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const convertTemperature = tool({
  name: 'convert_temperature',
  description: 'Convert a temperature between Celsius and Fahrenheit.',
  inputSchema: z.object({
    value: z.number().describe('The temperature to convert'),
    to: z.enum(['C', 'F']).describe('The unit to convert to'),
  }),
  callback: ({ value, to }) => {
    const result = to === 'F' ? (value * 9) / 5 + 32 : ((value - 32) * 5) / 9
    return `${result.toFixed(1)}°${to}`
  },
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [convertTemperature], printer: false })
console.log(agent.tools.map((t) => t.name)) // [ 'convert_temperature' ]
const direct = await agent.tool.convert_temperature!.invoke({ value: 0, to: 'F' }) // bypasses the model
console.log(direct.status, JSON.stringify(direct.content))
```

The callback's input is typed from the schema and validated before your code runs. More, including tool context, streaming tools, vended tools and MCP, is on the [tools page](/strands-agents-guide/typescript/strands_typescript_tools_and_mcp/).

## Structured output

Pass a Zod schema as `structuredOutputSchema` on the agent or per call. The value is on `result.structuredOutput`, typed as the schema's output in general but best re-parsed for a precise type:

```typescript
import { Agent, StructuredOutputError } from '@strands-agents/sdk'
import { z } from 'zod'

const Ticket = z.object({
  category: z.enum(['billing', 'technical', 'account']),
  priority: z.enum(['low', 'medium', 'high']),
  summary: z.string().max(200),
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic-small%%', printer: false })

try {
  const result = await agent.invoke('Customer: "I was charged twice and need a refund today."', {
    structuredOutputSchema: Ticket,
  })
  const ticket = Ticket.parse(result.structuredOutput)
  console.log(ticket.category, ticket.priority)
} catch (error) {
  if (error instanceof StructuredOutputError) {
    console.error('no valid ticket:', error.message)
  } else {
    throw error
  }
}
```

The SDK adds a tool named `strands_structured_output` with your schema, forces it if the model answers without it, and retries with the validation errors. `StructuredOutputError` means it still failed.

## Streaming events

`agent.stream()` yields typed events with a `type` discriminator; narrowing on it gives full types. The ones you will use:

| `type` | Contents |
|--------|----------|
| `modelStreamUpdateEvent` | `event`: a raw model event (`modelContentBlockDeltaEvent` with `textDelta`, `toolUseInputDelta` or `reasoningContentDelta`, and so on) |
| `beforeToolCallEvent` / `afterToolCallEvent` | `toolUse`, and on the after event `result` and `error` |
| `toolStreamUpdateEvent` | Progress from a streaming tool |
| `toolResultEvent` | A tool result block |
| `modelMessageEvent` | A complete assistant message |
| `agentResultEvent` | The final result |

The hook event classes double as stream events, so the same names appear in both places. To get the result as well as the events, drive the generator by hand:

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const getTime = tool({
  name: 'get_time',
  description: 'Current UTC time',
  inputSchema: z.object({}),
  callback: () => new Date().toISOString(),
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [getTime], printer: false })
const stream = agent.stream('What time is it in UTC?')

let next = await stream.next()
while (!next.done) {
  const event = next.value
  if (event.type === 'beforeToolCallEvent') console.log(`\n[tool] ${event.toolUse.name}`)
  if (
    event.type === 'modelStreamUpdateEvent' &&
    event.event.type === 'modelContentBlockDeltaEvent' &&
    event.event.delta.type === 'textDelta'
  ) {
    process.stdout.write(event.event.delta.text)
  }
  next = await stream.next()
}
console.log('\nstop reason:', next.value.stopReason)
```

There is no callback-handler API as in Python; use `stream()` or hooks.

## Conversation management

```typescript
import {
  Agent,
  NullConversationManager,
  SlidingWindowConversationManager,
  SummarizingConversationManager,
} from '@strands-agents/sdk'

const chat = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  conversationManager: new SlidingWindowConversationManager({ windowSize: 20 }),
  printer: false,
})

const support = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  conversationManager: new SummarizingConversationManager({ summaryRatio: 0.4, preserveRecentMessages: 8 }),
  printer: false,
})

const raw = new Agent({ model: 'us.anthropic.%%model=anthropic%%', conversationManager: new NullConversationManager(), printer: false })
```

The sliding window (the default, 40 messages) keeps recent messages without splitting tool calls from results; the summarising manager replaces old messages with a summary when the context overflows; the null manager never trims.

## State

`appState` is durable JSON state on the agent, saved by the session manager; `invocationState` is per call, may hold any value, and reaches tools and hooks:

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import type { ToolContext } from '@strands-agents/sdk'
import { z } from 'zod'

interface AppState {
  tier: string
  cart?: string[]
}

const myOrders = tool({
  name: 'my_orders',
  description: "List the current customer's orders.",
  inputSchema: z.object({}),
  callback: (_input, context?: ToolContext) => {
    if (!context) throw new Error('tool context required')
    const customer = context.invocationState.customerId as string
    const tier = context.agent.appState.get<AppState>('tier') // typed as string | undefined
    return `${customer} (${tier}): ORD-1, ORD-2`
  },
})

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  tools: [myOrders],
  appState: { tier: 'gold' },
  printer: false,
})

agent.appState.set('cart', ['A-1'])
console.log(agent.appState.get('cart'))
await agent.invoke('What have I ordered?', { invocationState: { customerId: 'C-456' } })
```

## Sessions

A `SessionManager` saves the agent's messages and state as snapshots and restores them when an agent with the same session ID is created. Give it a storage backend from `@strands-agents/sdk/storage`:

```typescript
import { Agent, SessionManager } from '@strands-agents/sdk'
import { LocalFileStorage, S3Storage } from '@strands-agents/sdk/storage'

function agentFor(sessionId: string): Agent {
  return new Agent({
    model: 'us.anthropic.%%model=anthropic%%',
    sessionManager: new SessionManager({ sessionId, storage: new LocalFileStorage('./sessions') }),
    printer: false,
  })
}

await agentFor('user-42').invoke('My name is Priya.')
const later = agentFor('user-42')
await later.initialize() // loads the saved snapshot
console.log(later.messages.length) // 2

// The same with S3 (needs @aws-sdk/client-s3)
const inS3 = new SessionManager({ sessionId: 'user-42', storage: new S3Storage('agent-sessions', { prefix: 'prod/' }) })
```

Session IDs may contain only lowercase letters, digits, hyphens and underscores; anything else throws when the `SessionManager` is created (agent `id`s are checked the same way when a snapshot is saved), so normalise user or tenant IDs first. This is stricter than Python, which only rejects path separators. `saveLatestOn` controls when the latest snapshot is written (`'invocation'` by default), and `snapshotTrigger` writes immutable checkpoints you can restore later. Graphs and swarms accept a `sessionManager` too. For long-term memory that is extracted and searched rather than replayed, the SDK has a `MemoryManager` with pluggable stores, and AgentCore Memory is available through `bedrock-agentcore` (see the [AgentCore TypeScript guide](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/)).

## Limits, cancellation and errors

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', printer: false })

const result = await agent.invoke('Research this thoroughly', {
  limits: { turns: 8, totalTokens: 50_000 },
  cancelSignal: AbortSignal.timeout(60_000),
})
if (result.stopReason.startsWith('limit') || result.stopReason === 'cancelled') {
  console.log('stopped early:', result.stopReason)
}
```

`agent.cancel()` stops the current call from elsewhere. Errors exported from the package root include `ModelThrottledError`, `ContextWindowOverflowError`, `MaxTokensError`, `StructuredOutputError`, `ConcurrentInvocationError`, `ToolValidationError`, `ToolNotFoundError` and `ModelError`.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.22.0 | New page in the standalone Strands guide, expanding the TypeScript Strands page from the Amazon Bedrock guide: project set-up, `AgentResult`, provider table (OpenAI defaults to Responses), routing, stream events, conversation managers, state, sessions with storage backends, limits and errors. Examples type-check under `strict` and run against an offline Bedrock stub. |

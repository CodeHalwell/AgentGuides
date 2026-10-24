---
title: "Strands Agents (TypeScript): Production Guide"
description: "Running TypeScript Strands agents in production: AgentCore Runtime, Lambda, Express on Fargate; one agent per session; limits, timeouts and retries; OpenTelemetry tracing; dependency conflicts; testing with a scripted model; and where the TypeScript SDK lags Python."
framework: "strands"
language: "typescript"
sidebar:
  label: "Production"
  order: 6
---

Checked against `@strands-agents/sdk` %%version=@strands-agents/sdk%% and `bedrock-agentcore` (npm) %%version=bedrock-agentcore-ts%% (%%versions-checked%%). Examples compile under `tsc --strict`; the Express, Lambda and test examples were run locally against an offline Bedrock stub, and the AgentCore example was run through its local HTTP server. Nothing was deployed.

## Hosting options

| Host | Fits | Notes |
|------|------|-------|
| **AgentCore Runtime** | Most agents on AWS | `bedrock-agentcore` provides the `/invocations` and `/ping` server; Node.js 22 (`NODE_22` for direct code deployment) |
| **AWS Lambda** | Short, stateless requests | 15-minute limit; bundle with esbuild; response streaming needs a streaming handler |
| **ECS Fargate, EKS, App Runner** | Containers you already run | Express or Fastify; you own scaling, timeouts and session storage |

The rules from the [Python production guide](/strands-agents-guide/python/strands_production_guide/) apply unchanged: one agent per conversation, history in a `SessionManager`, every run bounded, traces exported.

## AgentCore Runtime

```typescript
import { BedrockAgentCoreApp } from 'bedrock-agentcore/runtime'
import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'

const agents = new Map<string, Agent>()

function agentFor(sessionId: string): Agent {
  let agent = agents.get(sessionId)
  if (!agent) {
    agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', printer: false })
    agents.set(sessionId, agent)
  }
  return agent
}

const app = new BedrockAgentCoreApp({
  invocationHandler: {
    requestSchema: z.object({ prompt: z.string().min(1) }),
    process: async function* (request, context) {
      const agent = agentFor(context.sessionId)
      for await (const event of agent.stream(request.prompt, { limits: { turns: 10 } })) {
        if (
          event.type === 'modelStreamUpdateEvent' &&
          event.event.type === 'modelContentBlockDeltaEvent' &&
          event.event.delta.type === 'textDelta'
        ) {
          yield { data: { text: event.event.delta.text } }
        }
      }
    },
  },
})

app.run() // 0.0.0.0:8080
```

Install with `npm install bedrock-agentcore@%%version=bedrock-agentcore-ts%% @strands-agents/sdk@%%version=@strands-agents/sdk%% zod`. The request context, health status, memory, identity and deployment are covered in the [AgentCore TypeScript guide](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/#runtime-bedrockagentcoreapp) and its [production page](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_production/).

**Dependency conflict.** `bedrock-agentcore` %%version=bedrock-agentcore-ts%% declares an optional peer `@a2a-js/sdk >= 1.0.0`, while the Strands A2A module needs `@a2a-js/sdk` 0.3.x. A project that installs both `@a2a-js/sdk` and `bedrock-agentcore` fails `npm install` with `ERESOLVE`. If you do not need Strands' own A2A classes, leave `@a2a-js/sdk` out; otherwise keep the A2A server in a separate package.

## AWS Lambda

```typescript
import { Agent, SessionManager } from '@strands-agents/sdk'
import { S3Storage } from '@strands-agents/sdk/storage'

interface ChatEvent {
  sessionId: string
  prompt: string
}

export const handler = async (event: ChatEvent) => {
  const agent = new Agent({
    model: 'us.anthropic.%%model=anthropic%%',
    sessionManager: new SessionManager({
      sessionId: event.sessionId,
      storage: new S3Storage('agent-sessions', { prefix: 'lambda/' }),
    }),
    printer: false,
  })
  const result = await agent.invoke(event.prompt, {
    limits: { turns: 10 },
    cancelSignal: AbortSignal.timeout(50_000), // stay inside the function timeout
  })
  return { answer: result.toString(), stopReason: result.stopReason }
}
```

Build the agent inside the handler: consecutive events in one execution environment can belong to different users. Bundle with esbuild (`--platform=node --format=esm --target=node22`) and mark the optional peers you do not use as external.

## Express on Fargate or EKS

```typescript
import express from 'express'
import { Agent, SessionManager } from '@strands-agents/sdk'
import { S3Storage } from '@strands-agents/sdk/storage'

const app = express()
app.use(express.json())

function agentFor(sessionId: string): Agent {
  return new Agent({
    model: 'us.anthropic.%%model=anthropic%%',
    sessionManager: new SessionManager({ sessionId, storage: new S3Storage('agent-sessions', { prefix: 'chat/' }) }),
    printer: false,
  })
}

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' })
})

app.post('/chat', async (req, res) => {
  const { sessionId, prompt } = req.body as { sessionId: string; prompt: string }
  const controller = new AbortController()
  res.on('close', () => controller.abort()) // client went away: stop the agent

  res.setHeader('Content-Type', 'text/plain; charset=utf-8')
  const agent = agentFor(sessionId)
  for await (const event of agent.stream(prompt, { limits: { turns: 12 }, cancelSignal: controller.signal })) {
    if (
      event.type === 'modelStreamUpdateEvent' &&
      event.event.type === 'modelContentBlockDeltaEvent' &&
      event.event.delta.type === 'textDelta'
    ) {
      res.write(event.event.delta.text)
    }
  }
  res.end()
})

const server = app.listen(Number(process.env.PORT ?? 8080))
export { app, server }
```

Give the task role `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the models and inference profiles you use, and S3 access to the session bucket. Raise the load balancer's idle timeout above your longest stream. Concurrent requests for the same session each load and then save the snapshot, so serialise them per session if clients can send them in parallel.

## Limits, timeouts and retries

| Level | Setting |
|-------|---------|
| Agent loop | `limits: { turns, totalTokens, outputTokens }` and `cancelSignal` on each `invoke()` / `stream()` |
| Model | `maxTokens`; `clientConfig` on `BedrockModel` for the AWS SDK client (retries, request handler timeouts) |
| Multi-agent | `maxSteps`, `timeout`, `nodeTimeout` on `Graph` and `Swarm` (milliseconds; all unbounded by default) |
| Tools | Forward `context.cancelSignal` to `fetch` and other cancellable calls |
| Retries | `retryStrategy: new DefaultModelRetryStrategy({ maxAttempts, backoff })` ([details](/strands-agents-guide/typescript/strands_typescript_hooks_and_interrupts/#retry-strategies)); the default is 6 attempts, 4 s to 240 s |

The AWS SDK client retries on its own as well, so the two layers multiply; for interactive requests shorten the Strands strategy or set `clientConfig: { maxAttempts: 1 }` on the model.

## Tracing

Telemetry lives in `@strands-agents/sdk/telemetry` (not the package root). That module imports six OpenTelemetry packages, so install all of them, within the ranges the SDK declares as peers:

```bash
npm install "@opentelemetry/api@^1.9.0" "@opentelemetry/resources@^2.8.0" \
  "@opentelemetry/sdk-trace-base@^2.6.1" "@opentelemetry/sdk-metrics@^2.6.1" \
  "@opentelemetry/exporter-trace-otlp-http@^0.219.0" "@opentelemetry/exporter-metrics-otlp-http@^0.219.0"
```

Installing the newest versions without ranges fails: `@opentelemetry/resources` 3.x needs API 1.10, while current `sdk-trace-base` 2.x accepts only API below 1.10.

```typescript
import { Agent } from '@strands-agents/sdk'
import { setupTracer } from '@strands-agents/sdk/telemetry'

// OTLP over HTTP; reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS
setupTracer({ exporters: { otlp: true, console: false } })

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  traceAttributes: { 'session.id': 's-123', 'user.id': 'u-17' },
  printer: false,
})
await agent.invoke('Hello')
```

Call `setupTracer` once at start-up. To use your own provider, pass `setupTracer({ provider })` (you then register its context manager and propagators yourself). `setupMeter({ exporters: { otlp: true } })` exports metrics. One call produces `invoke_agent <name>` with an `execute_agent_loop_cycle` span per cycle, each containing `chat` and `execute_tool <tool>` spans; attributes follow the OpenTelemetry GenAI conventions, as in Python (whose cycle span is named `execute_event_loop_cycle`).

## Testing without the cloud

### A scripted model

```typescript
import { Agent, Model, tool } from '@strands-agents/sdk'
import type { BaseModelConfig, Message, ModelStreamEvent, StreamOptions } from '@strands-agents/sdk'
import { strict as assert } from 'node:assert'
import { z } from 'zod'

type Turn = string | { name: string; input: Record<string, unknown> }

/** Replays canned turns: a string is a text reply, an object is one tool call. */
class ScriptedModel extends Model<BaseModelConfig> {
  private config: BaseModelConfig = { modelId: 'scripted' }
  readonly requests: Message[][] = []

  constructor(private readonly turns: Turn[]) {
    super()
  }

  updateConfig(config: BaseModelConfig): void {
    this.config = { ...this.config, ...config }
  }

  getConfig(): BaseModelConfig {
    return this.config
  }

  async *stream(messages: Message[], _options?: StreamOptions): AsyncIterable<ModelStreamEvent> {
    this.requests.push([...messages])
    const turn = this.turns.shift() ?? 'OK'
    yield { type: 'modelMessageStartEvent', role: 'assistant' }
    if (typeof turn === 'string') {
      yield { type: 'modelContentBlockStartEvent' }
      yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: turn } }
      yield { type: 'modelContentBlockStopEvent' }
      yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
    } else {
      const toolUseId = `t${this.requests.length}`
      yield { type: 'modelContentBlockStartEvent', start: { type: 'toolUseStart', name: turn.name, toolUseId } }
      yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'toolUseInputDelta', input: JSON.stringify(turn.input) } }
      yield { type: 'modelContentBlockStopEvent' }
      yield { type: 'modelMessageStopEvent', stopReason: 'toolUse' }
    }
    yield { type: 'modelMetadataEvent', usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }
  }
}

const add = tool({
  name: 'add',
  description: 'Add two numbers.',
  inputSchema: z.object({ a: z.number(), b: z.number() }),
  callback: ({ a, b }) => a + b,
})

// The agent calls the tool, then answers
const model = new ScriptedModel([{ name: 'add', input: { a: 2, b: 3 } }, '2 + 3 = 5'])
const agent = new Agent({ model, tools: [add], printer: false })
const result = await agent.invoke('What is 2 + 3?')
assert.equal(result.toString(), '2 + 3 = 5')
assert.equal(model.requests.length, 2)

// Structured output is a call to the strands_structured_output tool
const Answer = z.object({ value: z.number() })
const scripted = new ScriptedModel([{ name: 'strands_structured_output', input: { value: 5 } }])
const typed = await new Agent({ model: scripted, printer: false }).invoke('2 + 3?', { structuredOutputSchema: Answer })
assert.deepEqual(typed.structuredOutput, { value: 5 })

console.log('tests passed')
```

The same model drives graphs and swarms (script `strands_structured_output` calls with an `agentId` to route a swarm), hooks, interventions and interrupts. To exercise `BedrockModel` itself, stub `BedrockRuntimeClient.prototype.send` from `@aws-sdk/client-bedrock-runtime` to return a `{ stream }` of Converse events; that is how every example in this guide was run.

## Where TypeScript differs from Python

As of %%version=@strands-agents/sdk%% against Python %%version=strands-agents%%:

| Area | Python | TypeScript |
|------|--------|------------|
| Tools | `@tool` with docstrings; Pydantic parameters arrive as dicts | `tool()` with Zod; nested input arrives parsed and typed |
| Calling | `agent("...")`, `invoke_async`, `stream_async` | `await agent.invoke()`, `agent.stream()` (returns the result) |
| Streaming output | Callback handlers or `stream_async` | `stream()` or hooks; no callback handlers |
| Hooks | `hooks=[...]` in the constructor, `agent.add_hook()` | `agent.addHook(EventClass, fn)` only |
| Providers | Bedrock, Anthropic, OpenAI (Chat and Responses), Gemini, LiteLLM, Ollama, Mistral, llama.cpp, Llama API, SageMaker, Writer | Bedrock, Anthropic, OpenAI (Responses by default, or Chat), Google, Vercel AI SDK |
| Tool library | `strands-agents-tools` plus `strands.vended_tools` | Vended tools only; no `use_aws`, `workflow`, `retrieve`, browser or code-interpreter wrappers |
| Multi-agent | `GraphBuilder`, `Swarm` with handoff tool, `workflow` tool; bounded defaults | `Graph({ nodes, edges })`, `Swarm` with structured-output handoffs; unbounded defaults, milliseconds |
| Agent as tool | `as_tool(name, description, preserve_context, delegate)` | `asTool({ name, description, preserveContext })` |
| Sessions | File, S3, snapshot and AgentCore Memory session managers | `SessionManager` with `LocalFileStorage`, `S3Storage` or `InMemoryStorage`; AgentCore Memory as a `MemoryManager` store |
| Evals | `strands-agents-evals` | No TypeScript evals package |
| Real-time voice | `BidiAgent` | Not available |

## Checklist

- Model ID set explicitly.
- `printer: false` in servers.
- One agent per conversation; history in a `SessionManager`.
- `limits` and a `cancelSignal` on every call; `maxSteps` and `timeout` on every graph and swarm.
- Retry layers sized for the use.
- `setupTracer` at start-up with session and user trace attributes.
- Optional peers pinned to the ranges the SDK declares.
- Unit tests on a scripted model.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.22.0 | New page: AgentCore Runtime, Lambda and Express hosting, limits and retries, tracing set-up from the `telemetry` subpath, the OpenTelemetry and `@a2a-js/sdk` peer conflicts found while installing, a scripted test model, and a Python comparison table. |

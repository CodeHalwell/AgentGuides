---
title: "Bedrock AgentCore TypeScript SDK"
description: "The bedrock-agentcore npm package: BedrockAgentCoreApp for AgentCore Runtime, AgentCore Memory for Strands, Code Interpreter and Browser tools, and Identity credential wrappers."
framework: "amazon-bedrock-agents"
language: "typescript"
---

# Bedrock AgentCore TypeScript SDK

The `bedrock-agentcore` npm package is the TypeScript counterpart of the Python `bedrock-agentcore` SDK. It provides an AgentCore Runtime-compatible HTTP server, a Strands memory store backed by AgentCore Memory, clients and Strands/Vercel AI tool wrappers for Code Interpreter and Browser, and helpers for AgentCore Identity.

> **Verified against:** `bedrock-agentcore` %%version=bedrock-agentcore-ts%% with `@strands-agents/sdk` %%version=@strands-agents/sdk%% (checked %%versions-checked%%). Every example compiles under `tsc --strict`; the Runtime server example was also run locally.

> **Pre-1.0 package.** The TypeScript SDK is at 0.x, and its Strands tool integrations for Code Interpreter and Browser are published under `experimental/` import paths. Expect breaking changes between minor versions and pin the version you test against.

For the AgentCore services themselves (what Runtime, Memory, Gateway and Identity are, and the boto3 control-plane calls), see the [AgentCore comprehensive guide](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/). Deployment from Node is on the [production page](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_production/).

---

## Table of Contents

1. [Installation and modules](#installation-and-modules)
2. [Runtime: BedrockAgentCoreApp](#runtime-bedrockagentcoreapp)
3. [Memory](#memory)
4. [Code Interpreter](#code-interpreter)
5. [Browser](#browser)
6. [Identity](#identity)
7. [A2A, WebSocket and other Runtime features](#a2a-websocket-and-other-runtime-features)
8. [What the TypeScript SDK does not cover](#what-the-typescript-sdk-does-not-cover)

---

## Installation and modules

```bash
npm install bedrock-agentcore@%%version=bedrock-agentcore-ts%% @strands-agents/sdk@%%version=@strands-agents/sdk%% zod
```

Node.js 20 or later is required by `bedrock-agentcore`; Strands needs Node.js 22, so use 22 for both (it is also the `NODE_22` runtime that AgentCore direct code deployment offers). The package is ESM and is imported by subpath:

| Import path | Contents |
|-------------|----------|
| `bedrock-agentcore/runtime` | `BedrockAgentCoreApp`, `RuntimeClient` (presigned WebSocket URLs, interactive shell), request/handler types |
| `bedrock-agentcore/runtime/a2a` | `serveA2A`, `buildAgentCard`, `getContext` for the A2A protocol |
| `bedrock-agentcore/memory/strands` | `AgentCoreMemoryStore`, `createAgentCoreMemoryStores` for Strands `MemoryManager` |
| `bedrock-agentcore/code-interpreter` | `CodeInterpreter` client |
| `bedrock-agentcore/experimental/code-interpreter/strands` | `CodeInterpreterTools` (Strands tools) |
| `bedrock-agentcore/browser`, `bedrock-agentcore/browser/playwright` | `Browser` session client, `PlaywrightBrowser` |
| `bedrock-agentcore/experimental/browser/strands` | `BrowserTools` (Strands tools) |
| `bedrock-agentcore/identity` | `withApiKey`, `withAccessToken` |
| `bedrock-agentcore/web-search` | `WebSearchClient` for a Gateway web-search connector target |

`@strands-agents/sdk`, `playwright`, `ai` (Vercel AI SDK), `express` and `@a2a-js/sdk` are optional peer dependencies; install the ones the modules you import need.

---

## Runtime: BedrockAgentCoreApp

AgentCore Runtime calls your container or code bundle over a small HTTP contract: `GET /ping` for health and `POST /invocations` for requests, on port 8080. `BedrockAgentCoreApp` (built on Fastify) implements that contract. You supply an `invocationHandler` whose `process` function receives the parsed request body and a `RequestContext`; an optional Zod `requestSchema` validates and types the body, and invalid requests get a 400.

### A Strands agent on Runtime

`process` can return a value (sent as JSON) or be an async generator (sent as Server-Sent Events). This example streams Strands text deltas and keeps one agent per session so each session keeps its conversation:

```typescript
import { BedrockAgentCoreApp } from 'bedrock-agentcore/runtime'
import { Agent, BedrockModel, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const lookupOrder = tool({
  name: 'lookup_order',
  description: 'Look up an order by ID.',
  inputSchema: z.object({ orderId: z.string() }),
  callback: ({ orderId }) => ({ orderId, status: 'shipped' }),
})

// One agent per runtime session. On AgentCore Runtime each session gets its
// own microVM, so in production this map holds a single entry.
const agents = new Map<string, Agent>()

function agentFor(sessionId: string): Agent {
  let agent = agents.get(sessionId)
  if (!agent) {
    agent = new Agent({
      model: new BedrockModel({ modelId: 'us.anthropic.claude-sonnet-5-5', region: 'us-west-2' }),
      systemPrompt: 'You are a helpful order-support assistant.',
      tools: [lookupOrder],
      printer: false,
    })
    agents.set(sessionId, agent)
  }
  return agent
}

const app = new BedrockAgentCoreApp({
  invocationHandler: {
    requestSchema: z.object({ prompt: z.string().min(1) }),
    process: async function* (request, context) {
      // request is typed as { prompt: string }
      const agent = agentFor(context.sessionId)
      for await (const event of agent.stream(request.prompt)) {
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

app.run() // listens on 0.0.0.0:8080
```

Run it locally with `npx tsx main.ts` and call it the way Runtime does. A session ID is required (the header, or a `sessionId` field in the body); without one the server returns 400, as it does when the body fails `requestSchema`.

```bash
curl -s http://localhost:8080/ping
# {"status":"Healthy","time_of_last_update":1792620000}

curl -N -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -H 'X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: local-session-0000000000000000000001' \
  -d '{"prompt": "Where is order ORD-7?"}'
# data: {"text":"Order ORD-7 has "}
# data: {"text":"shipped."}
```

A streaming handler only streams to callers that send `Accept: text/event-stream`; anyone else gets HTTP 406. When invoking a deployed runtime, set `accept: 'text/event-stream'` on `InvokeAgentRuntimeCommand` (see the [production page](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_production/#invoking-a-deployed-runtime)). If the generator throws, the server sends an `event: error` message and closes the stream.

A non-streaming handler is simpler, and is what you want when the caller expects one JSON document:

```typescript
import { BedrockAgentCoreApp } from 'bedrock-agentcore/runtime'
import { Agent } from '@strands-agents/sdk'
import { z } from 'zod'

const agent = new Agent({ model: 'us.anthropic.claude-haiku-5-5', printer: false })

const app = new BedrockAgentCoreApp({
  invocationHandler: {
    requestSchema: z.object({ prompt: z.string() }),
    process: async (request, context) => {
      context.log.info({ requestId: context.requestId }, 'invocation')
      const result = await agent.invoke(request.prompt)
      return { result: result.toString(), sessionId: context.sessionId }
    },
  },
})

app.run({ port: Number(process.env.PORT ?? 8080) })
```

### The request context

| Field | Meaning |
|-------|---------|
| `sessionId` | From `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id`; one microVM per session on Runtime |
| `headers` | Forwardable headers: `Authorization` and `X-Amzn-Bedrock-AgentCore-Runtime-Custom-*` (custom headers must be on the runtime's request-header allowlist) |
| `workloadAccessToken` | Workload identity token used by the Identity wrappers |
| `requestId` | Request ID for tracing (generated if absent) |
| `oauth2CallbackUrl` | OAuth2 callback URL for user-federation flows |
| `log` | Fastify logger scoped to the request |

Runtime does not tell the agent who the end user is. If you need an actor ID (for example for memory), pass it in the payload or a custom header.

### Health status and background work

Runtime reclaims idle sessions. `/ping` reports `HealthyBusy` while tracked background tasks are running, which keeps the session alive. Wrap long-running work with `app.asyncTask()`, or use `addAsyncTask()` / `completeAsyncTask()` by hand; `pingHandler` overrides the status entirely.

```typescript
import { BedrockAgentCoreApp } from 'bedrock-agentcore/runtime'
import { z } from 'zod'

async function rebuildIndex(datasetId: string): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 120_000)) // stand-in for real work
  console.log(`rebuilt ${datasetId}`)
}

const app = new BedrockAgentCoreApp({
  invocationHandler: {
    requestSchema: z.object({ datasetId: z.string() }),
    process: async (request) => {
      // Return immediately; /ping reports HealthyBusy until the task completes
      const taskId = app.addAsyncTask('rebuild-index', { datasetId: request.datasetId })
      void rebuildIndex(request.datasetId).finally(() => app.completeAsyncTask(taskId))
      return { status: 'started' }
    },
  },
})

app.run()
```

`app.getAsyncTaskInfo()` returns the active task count and running jobs, for logging or diagnostics.

---

## Memory

AgentCore Memory is exposed to TypeScript as a Strands `MemoryStore`. `createAgentCoreMemoryStores()` builds one store per namespace; give them to a Strands `MemoryManager` and the agent gets a `search_memory` tool for recall, while conversation turns are written back with `CreateEvent` so that the memory resource's strategies (semantic, summary, user preference, episodic) can extract long-term records on the server.

```typescript
import { BedrockAgentCoreApp } from 'bedrock-agentcore/runtime'
import { createAgentCoreMemoryStores } from 'bedrock-agentcore/memory/strands'
import { Agent, MemoryManager } from '@strands-agents/sdk'
import { z } from 'zod'

const memoryId = process.env.MEMORY_SUPPORTMEMORY_ID // injected by the AgentCore CLI deploy
if (!memoryId) throw new Error('MEMORY_SUPPORTMEMORY_ID is not set')

const agents = new Map<string, Agent>()

function agentFor(actorId: string, sessionId: string): Agent {
  const key = `${actorId}:${sessionId}`
  let agent = agents.get(key)
  if (!agent) {
    const stores = createAgentCoreMemoryStores({
      memoryId: memoryId!,
      actorId,
      sessionId,
      namespaces: [
        { namespace: '/users/{actorId}/facts', writable: true },
        { namespace: '/users/{actorId}/preferences' },
      ],
      extraction: true, // write turns back to AgentCore Memory
      region: 'us-west-2',
    })
    agent = new Agent({
      model: 'us.anthropic.claude-sonnet-5-5',
      memoryManager: new MemoryManager({ stores }),
      printer: false,
    })
    agents.set(key, agent)
  }
  return agent
}

const app = new BedrockAgentCoreApp({
  invocationHandler: {
    requestSchema: z.object({ prompt: z.string(), userId: z.string().optional() }),
    process: async (request, context) => {
      const actorId =
        context.headers['x-amzn-bedrock-agentcore-runtime-custom-actor-id'] ??
        request.userId ??
        context.sessionId
      const agent = agentFor(actorId, context.sessionId)
      try {
        return { result: (await agent.invoke(request.prompt)).toString() }
      } finally {
        // Make sure buffered CreateEvent calls land before the microVM can be reclaimed
        await agent.memoryManager?.flush()
      }
    },
  },
})

app.run()
```

Things to get right:

- **The memory resource must already exist.** The store never creates it. Create it with `agentcore add memory` (or CloudFormation/CDK, or the control-plane API); when a memory and a runtime are deployed together by the CLI, the memory ID arrives as `MEMORY_<NAME>_ID`.
- **Namespaces must match the strategies' templates.** Query with the same template the memory was provisioned with (for example `/users/{actorId}/facts`); the store resolves `{actorId}` and `{sessionId}` and throws on any other placeholder. A namespace that includes `{sessionId}` is only recalled within that session.
- **Recall is eventually consistent.** Extraction runs asynchronously on the server, so a fact written in this turn may not be searchable in the next.
- **At most one writable namespace.** Writes go to a single event stream per actor and session; mark the write target with `writable: true`.

To read or write events and records directly, without Strands, use the AWS SDK client (`@aws-sdk/client-bedrock-agentcore`: `CreateEventCommand`, `ListEventsCommand`, `RetrieveMemoryRecordsCommand`). There is no TypeScript equivalent of the Python `MemoryClient` or of `AgentCoreMemorySessionManager`.

---

## Code Interpreter

`CodeInterpreterTools` gives a Strands agent three tools backed by a managed, sandboxed Code Interpreter session: `executeCode` (Python, JavaScript or TypeScript), `executeCommand` (shell) and `fileOperations` (read, write, list, remove). A session starts on first use; stop it when you are done so you are not billed until the timeout.

```typescript
import { CodeInterpreterTools } from 'bedrock-agentcore/experimental/code-interpreter/strands'
import { Agent } from '@strands-agents/sdk'

const codeInterpreter = new CodeInterpreterTools({ region: 'us-west-2' })

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  systemPrompt: 'Use the code tools to compute answers rather than estimating.',
  tools: codeInterpreter.tools,
  printer: false,
})

try {
  const result = await agent.invoke('What is the standard deviation of 3, 7, 7, 19, 24? Show your code.')
  console.log(result.toString())
} finally {
  await codeInterpreter.stopSession()
}
```

The `CodeInterpreter` client can be used on its own. Its methods return the service's output as a string:

```typescript
import { CodeInterpreter } from 'bedrock-agentcore/code-interpreter'

const interpreter = new CodeInterpreter({ region: 'us-west-2' })

await interpreter.startSession({ sessionName: 'analysis', timeout: 900 })
try {
  await interpreter.writeFiles({ files: [{ path: 'data.csv', content: 'x,y\n1,2\n3,4\n' }] })
  const output = await interpreter.executeCode({
    language: 'python',
    code: 'import pandas as pd\nprint(pd.read_csv("data.csv").describe())',
  })
  console.log(output)
} finally {
  await interpreter.stopSession()
}
```

Pass `identifier` to use a custom Code Interpreter (for example one with VPC networking) instead of the default `aws.codeinterpreter.v1`.

---

## Browser

`BrowserTools` gives a Strands agent `navigate`, `click`, `type`, `getText`, `getHtml`, `screenshot` and `evaluate` tools that drive a managed AgentCore Browser session over Playwright. It needs the `playwright` package installed (only the client library is used; no local browser is launched).

```typescript
import { BrowserTools } from 'bedrock-agentcore/experimental/browser/strands'
import { Agent } from '@strands-agents/sdk'

const browser = new BrowserTools({ region: 'us-west-2' })

const agent = new Agent({
  model: 'us.anthropic.claude-sonnet-5-5',
  systemPrompt: 'Use the browser tools to read web pages. Quote what you find.',
  tools: browser.tools,
  printer: false,
})

try {
  const result = await agent.invoke('Open https://aws.amazon.com/bedrock/agentcore/ and list the services it names.')
  console.log(result.toString())
} finally {
  await browser.stopSession()
}
```

For scripted automation without a model, use `PlaywrightBrowser` directly:

```typescript
import { PlaywrightBrowser } from 'bedrock-agentcore/browser/playwright'

const browser = new PlaywrightBrowser({ region: 'us-west-2' })

try {
  await browser.startSession({ viewport: { width: 1280, height: 800 } })
  await browser.navigate({ url: 'https://example.com' })
  const heading = await browser.getText({ selector: 'h1' })
  console.log(heading)
  console.log('Live view:', await browser.generateLiveViewUrl())
} finally {
  await browser.stopSession()
}
```

`startSession()` also accepts `proxyConfiguration`, `extensions` (from S3) and `profileConfiguration` for persisted browser profiles. A React live-view component is available from `bedrock-agentcore/browser/live-view`.

---

## Identity

AgentCore Identity holds API keys and OAuth2 credentials for outbound calls. Inside a Runtime handler, `withApiKey` and `withAccessToken` wrap an async function whose **last** parameter is the credential; the wrapper fetches it using the request's workload access token and returns a function without that parameter.

```typescript
import { BedrockAgentCoreApp } from 'bedrock-agentcore/runtime'
import { withAccessToken, withApiKey } from 'bedrock-agentcore/identity'
import { z } from 'zod'

// API key stored in an AgentCore Identity API-key credential provider
const searchTickets = withApiKey({ providerName: 'zendesk-api-key' })(
  async (query: string, apiKey: string): Promise<unknown> => {
    const response = await fetch(`https://example.zendesk.com/api/v2/search.json?query=${encodeURIComponent(query)}`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    })
    return response.json()
  }
)

// OAuth2 machine-to-machine token from an OAuth2 credential provider
const listRepos = withAccessToken({ providerName: 'github', scopes: ['repo'], authFlow: 'M2M' })(
  async (org: string, token: string): Promise<unknown> => {
    const response = await fetch(`https://api.github.com/orgs/${org}/repos`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    return response.json()
  }
)

const app = new BedrockAgentCoreApp({
  invocationHandler: {
    requestSchema: z.object({ query: z.string(), org: z.string() }),
    process: async (request) => ({
      tickets: await searchTickets(request.query),
      repos: await listRepos(request.org),
    }),
  },
})

app.run()
```

When called inside a request, the wrappers read `workloadAccessToken` from the request context automatically; outside a handler, pass `workloadIdentityToken` in the config. For `USER_FEDERATION` (three-legged OAuth), add `onAuthUrl` to receive the consent URL to show the user, and `callbackUrl`. The wrappers are plain functions, so they work equally inside a Strands tool callback. Credential providers and workload identities are created with the control plane (`@aws-sdk/client-bedrock-agentcore-control`) or `agentcore add credential`.

---

## A2A, WebSocket and other Runtime features

- **A2A protocol.** `serveA2A()` from `bedrock-agentcore/runtime/a2a` serves an `@a2a-js/sdk` `AgentExecutor` on the A2A contract (JSON-RPC at `POST /`, agent card at `/.well-known/agent-card.json`, port 9000). It requires `@a2a-js/sdk` 1.x and `express`. Note that Strands' own A2A module (`@strands-agents/sdk/a2a`) is built on `@a2a-js/sdk` 0.3.x, so the two cannot share one install as of these versions; npm reports a peer-dependency conflict. Pick one A2A stack per service.
- **WebSocket.** Pass `websocketHandler` to `BedrockAgentCoreApp` to serve bidirectional streaming on `GET /ws`. `RuntimeClient.generateWsConnection()` and `generatePresignedUrl()` produce signed connection details for callers.
- **Custom content types.** `config.contentTypeParsers` registers Fastify parsers for non-JSON payloads.
- **Interactive shell.** `RuntimeClient.openShell()` opens a shell session into a running runtime session, for debugging.

---

## What the TypeScript SDK does not cover

Compared with the Python `bedrock-agentcore` SDK at the time of writing:

| Area | Python SDK | TypeScript SDK |
|------|-----------|----------------|
| Runtime app | `BedrockAgentCoreApp` with `@app.entrypoint` | `BedrockAgentCoreApp` with `invocationHandler.process` |
| Memory | `MemoryClient`; `AgentCoreMemorySessionManager` for Strands | Strands `MemoryStore` only; use `@aws-sdk/client-bedrock-agentcore` for raw calls |
| Gateway | `GatewayClient` | No Gateway module ("coming soon" in the package README); reach a gateway as an MCP server with Strands `McpClient`, manage it with `@aws-sdk/client-bedrock-agentcore-control` or `agentcore add gateway` |
| Observability | OpenTelemetry via ADOT (`opentelemetry-instrument`) | No SDK module ("coming soon"); Strands emits OpenTelemetry spans and metrics, configured through `@strands-agents/sdk/telemetry` |
| Code Interpreter, Browser | `code_session`, `browser_session` and clients | Clients plus Strands and Vercel AI tools (the Strands tools are under `experimental/`) |
| Identity | `@requires_access_token`, `@requires_api_key` decorators | `withAccessToken`, `withApiKey` wrappers |
| Policy, Evaluations, Payments | Not covered by this site's Python guide | No SDK modules; use the AWS SDK for JavaScript clients or the AgentCore CLI |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | `bedrock-agentcore` 0.4.6 | New page. Examples compiled with `tsc --strict`; Runtime server run locally against a stub model. |

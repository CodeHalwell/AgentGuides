---
title: "Strands Agents (TypeScript): Tools and MCP"
description: "Tools in the TypeScript Strands SDK: Zod tools with typed input, JSON Schema tools, tool context and per-request state, streaming tools, direct calls, executors, vended tools, sandboxes and MCP clients."
framework: "strands"
language: "typescript"
sidebar:
  label: "Tools and MCP"
  order: 2
---

Checked against `@strands-agents/sdk` %%version=@strands-agents/sdk%% (%%versions-checked%%). Examples compile under `tsc --strict` and were run against an offline Bedrock stub; the MCP example ran against a real local stdio server.

## Zod tools

`tool()` takes a name, a description, a Zod schema and a callback. The callback's input is inferred from the schema, and the model's arguments are validated before the callback runs. `.describe()` on each field becomes the parameter description the model reads:

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const getOrderStatus = tool({
  name: 'get_order_status',
  description: 'Look up the shipping status of an order by its ID.',
  inputSchema: z.object({
    orderId: z.string().regex(/^ORD-\d+$/).describe('Order ID, e.g. ORD-1042'),
    includeHistory: z.boolean().default(false).describe('Include past status changes'),
  }),
  callback: async ({ orderId, includeHistory }) => {
    // input is typed as { orderId: string; includeHistory: boolean }
    const status = { orderId, status: 'shipped', carrier: 'DPD' }
    return includeHistory ? { ...status, history: ['created', 'packed', 'shipped'] } : status
  },
})

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  tools: [getOrderStatus],
  printer: false,
})

console.log((await agent.invoke('Where is order ORD-1042?')).toString())
```

Unlike Python, nested objects arrive as validated, typed values (Zod parses them), so a `z.object` parameter is safe to use directly.

**Return values.** Strings and numbers become a text block; objects and arrays become a JSON content block. **Errors.** If the callback throws, or validation fails, the model receives an error tool result and the loop continues.

## Tool context

The callback's second argument is a `ToolContext`:

| Field | Contents |
|-------|----------|
| `toolUse` | The call (`toolUseId`, `name`, `input`) |
| `agent` | The calling agent (`appState`, `messages`, ...) |
| `invocationState` | The per-call state passed to `invoke()` |
| `cancelSignal` | An `AbortSignal` for cancellation; forward it to `fetch` and other APIs |
| `interrupt({ name, reason })` | Pause for human input ([interrupts](/strands-agents-guide/typescript/strands_typescript_hooks_and_interrupts/#interrupts)) |

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
    const response = await Promise.resolve({ invoices: [] as string[] }) // e.g. fetch(url, { signal: context.cancelSignal })
    return { tenantId, requestId, limit: input.limit, ...response }
  },
})

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  appState: { tenantId: 'acme' },
  tools: [listInvoices],
  printer: false,
})

await agent.invoke('Show my last five invoices.', { invocationState: { requestId: 'req-42' } })
```

Identity and authorisation data belongs in `invocationState`, where the model cannot see or change it.

## Streaming tools and JSON Schema tools

A callback written as an async generator streams progress: each `yield` appears as a `toolStreamUpdateEvent` in `agent.stream()`, and the generator's **return value** is the tool result (the last yielded value is not, unlike Python):

```typescript
import { Agent, FunctionTool, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const exportReport = tool({
  name: 'export_report',
  description: 'Export a report to PDF.',
  inputSchema: z.object({ reportId: z.string() }),
  callback: async function* (input): AsyncGenerator<string, string, unknown> {
    yield 'Collecting data...'
    yield 'Rendering PDF...'
    return `s3://reports/${input.reportId}.pdf`
  },
})

// Plain JSON Schema instead of Zod: input arrives as unknown
const ping = new FunctionTool({
  name: 'ping',
  description: 'Check that a host name resolves.',
  inputSchema: { type: 'object', properties: { host: { type: 'string' } }, required: ['host'] },
  callback: (input: unknown) => `ok: ${(input as { host: string }).host}`,
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [exportReport, ping], printer: false })

for await (const event of agent.stream('Export report R-9')) {
  if (event.type === 'toolStreamUpdateEvent') console.log('progress:', event.event.data)
}
```

`ZodTool` is the class behind `tool()`, and `Tool` is the abstract base for fully custom tools.

## Direct calls

`agent.tool.<name>` gives a handle for calling a tool without the model. `invoke()` returns the `ToolResultBlock`; `stream()` yields progress first. Calls are recorded in the history (as an assistant tool use, the tool result and a short assistant note) unless you pass `{ recordDirectToolCall: false }`:

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const loadProfile = tool({
  name: 'load_profile',
  description: 'Load a user profile.',
  inputSchema: z.object({ userId: z.string() }),
  callback: ({ userId }) => ({ userId, name: 'Priya', plan: 'pro' }),
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [loadProfile], printer: false })

const result = await agent.tool.load_profile!.invoke({ userId: 'u-17' })
console.log(result.status, JSON.stringify(result.content))
await agent.tool.load_profile!.invoke({ userId: 'u-18' }, { recordDirectToolCall: false })
console.log(agent.messages.length) // 3: the first call (tool use, result, and a note) was recorded; the second was not
```

## Executors

Tool calls from one model turn run concurrently by default. To run them in order:

```typescript
import { Agent, SequentialToolExecutor } from '@strands-agents/sdk'

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  toolExecutor: new SequentialToolExecutor(),
  printer: false,
})
```

## Vended tools

The SDK ships optional tools under `@strands-agents/sdk/vended-tools/*`. They are not loaded unless imported:

| Import path | Exports |
|-------------|---------|
| `vended-tools/shell` | `makeShell(sandbox?, options?)` (`makeBash` is deprecated) |
| `vended-tools/file-editor` | `fileEditor`, `makeFileEditor` |
| `vended-tools/http-request` | `httpRequest` |
| `vended-tools/web-fetch` | `webFetch`, `makeWebFetch` (needs `turndown`) |
| `vended-tools/notebook` | `notebook`, `makeNotebook` |
| `vended-tools/python-repl` | `pythonRepl`, `makePythonRepl` |
| `vended-tools/sleep`, `vended-tools/handoff-to-user` | `sleep`, `handoffToUser` |
| `vended-tools/subagent`, `vended-tools/swarm` | Let the model spawn sub-agents |
| `vended-tools/mcp-router`, `vended-tools/a2a-client` | Factories for MCP routing and A2A calls |

```typescript
import { Agent } from '@strands-agents/sdk'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'
import { makeShell } from '@strands-agents/sdk/vended-tools/shell'

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  tools: [makeShell(), fileEditor, httpRequest],
  printer: false,
})
console.log(agent.tools.map((t) => t.name))
```

Shell, file and code tools run on the host unless the agent has a sandbox. `DockerSandbox` from `@strands-agents/sdk/sandbox/docker` (`new DockerSandbox({ container, workingDir, user })`) and an SSH sandbox from `@strands-agents/sdk/sandbox/ssh` route them elsewhere; pass one as the agent's `sandbox` option.

There is no TypeScript `strands-agents-tools` package: the vended tools are the whole set. Python-only tools such as `use_aws`, `workflow`, `retrieve` and the AgentCore browser and code interpreter wrappers have no TypeScript counterpart; write a Zod tool around the AWS SDK call you need.

## MCP

`McpClient` connects to a Model Context Protocol server and exposes its tools. Install `@modelcontextprotocol/client`, and pass the client in `tools`; the agent connects and lists tools when it initialises (on the first `invoke()`, or `await agent.initialize()`).

This connects to the stdio inventory server from the [Python MCP section](/strands-agents-guide/python/strands_tools_and_mcp/#a-local-stdio-server) (any MCP server works the same way), filters out one tool and prefixes the rest:

```typescript
import { Agent, McpClient } from '@strands-agents/sdk'
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio'

const inventory = new McpClient({
  transport: new StdioClientTransport({ command: 'python3', args: ['inventory_server.py'] }),
  toolFilters: { rejected: ['delete_sku'] },
  prefix: 'inv',
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [inventory], printer: false })

try {
  await agent.initialize()
  console.log(agent.tools.map((t) => t.name)) // [ 'inv_stock_level' ]
  console.log((await agent.invoke('How many A-1 are in stock?')).toString())
} finally {
  await inventory.disconnect()
}
```

For a Streamable HTTP server, pass `url` (and `headers`, or `auth` for OAuth client credentials, or `authProvider`):

```typescript
import { Agent, McpClient } from '@strands-agents/sdk'

const internal = new McpClient({
  url: 'https://mcp.example.com/mcp',
  headers: { Authorization: `Bearer ${process.env.MCP_TOKEN ?? ''}` },
  continueOnError: true, // start the agent even if this server is down
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [internal], printer: false })
```

Other options: `toolFilters` (`allowed` / `rejected` lists of names, `RegExp`s or callbacks), `elicitationCallback`, `progressCallback`, `tasksConfig` for long-running tools, `logHandler`. `McpClient.loadServers(config)` builds clients from an `mcpServers`-style JSON file or object. The client can also be used directly: `listTools()`, `callTool()`, `listPrompts()`, `getPrompt()`, `listResources()`, `readResource()`. AgentCore Gateway is an MCP server, reached the same way with an OAuth token.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.22.0 | New page, from the tools and MCP sections of the TypeScript Strands page in the Amazon Bedrock guide. Adds tool context fields, the generator return-value rule, direct calls, executors, the vended tools table and sandboxes; MCP example run against a local stdio server. |

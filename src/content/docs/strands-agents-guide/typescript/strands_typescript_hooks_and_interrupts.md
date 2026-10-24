---
title: "Strands Agents (TypeScript): Hooks, Interventions and Interrupts"
description: "Extending the TypeScript Strands agent loop: typed hook events and their writable fields, plugins, intervention handlers, human-in-the-loop interrupts, retry strategies and middleware."
framework: "strands"
language: "typescript"
sidebar:
  label: "Hooks and interrupts"
  order: 3
---

Checked against `@strands-agents/sdk` %%version=@strands-agents/sdk%% (%%versions-checked%%). Examples compile under `tsc --strict` and were run against an offline Bedrock stub.

The layers are the same as in Python: hooks observe and adjust the loop, plugins bundle hooks and tools, intervention handlers make policy decisions, and interrupts pause for a human. The main API difference is that hooks are registered with `agent.addHook(EventClass, callback)` rather than passed to the constructor.

## Hook events

| Event | Fires | Writable |
|-------|-------|----------|
| `InitializedEvent` | After the agent initialises | |
| `BeforeInvocationEvent` | Start of each `invoke()` / `stream()` | `cancel` |
| `AfterInvocationEvent` | End of each call | `resume` (start another call with this input) |
| `MessageAddedEvent` | A message was appended | |
| `BeforeModelCallEvent` | Before each model call | `cancel` |
| `AfterModelCallEvent` | After each model call (`stopData` or `error`, `attemptCount`) | `retry` |
| `BeforeToolsEvent` | Before a batch of tool calls | `cancel` |
| `BeforeToolCallEvent` | Before each tool call | `toolUse`, `selectedTool`, `cancel` |
| `AfterToolCallEvent` | After each tool call (`result`, `error`) | `result`, `retry` |
| `AfterToolsEvent` | After the batch | `endTurn` |

Each event has `agent` and `invocationState`. The fields that are not writable are `readonly` in the type definitions, so the compiler stops you. `After*` callbacks run in reverse registration order.

```typescript
import { Agent, AfterInvocationEvent, AfterToolCallEvent, BeforeToolCallEvent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const deleteBucket = tool({
  name: 'delete_bucket',
  description: 'Delete an S3 bucket',
  inputSchema: z.object({ bucket: z.string() }),
  callback: ({ bucket }) => `deleted ${bucket}`,
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [deleteBucket], printer: false })

// Block destructive tools for read-only callers; the model receives the message as an error result
agent.addHook(BeforeToolCallEvent, (event) => {
  if (event.invocationState.readOnly === true && event.toolUse.name.startsWith('delete_')) {
    event.cancel = `Tool ${event.toolUse.name} is not allowed for read-only users.`
  }
})

// Audit log
agent.addHook(AfterToolCallEvent, (event) => {
  console.log(JSON.stringify({ tool: event.toolUse.name, status: event.result.status, failed: event.error !== undefined }))
})

const removeCounter = agent.addHook(AfterInvocationEvent, (event) => {
  console.log(`messages in history: ${event.agent.messages.length}`)
})

await agent.invoke('Delete the bucket called scratch-data.', { invocationState: { readOnly: true } })
removeCounter() // addHook returns a function that removes the hook
```

`addHook(Event, callback, { order })` sets priority (lower runs first). Callbacks may be async.

### Retrying a tool call

```typescript
import { Agent, AfterToolCallEvent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

let attempts = 0
const flakyLookup = tool({
  name: 'flaky_lookup',
  description: 'Look up a key in a flaky service.',
  inputSchema: z.object({ key: z.string() }),
  callback: ({ key }) => {
    attempts += 1
    if (attempts < 3) throw new Error('service unavailable')
    return `value for ${key}`
  },
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [flakyLookup], printer: false })

agent.addHook(AfterToolCallEvent, (event) => {
  const retries = (event.invocationState.toolRetries as number | undefined) ?? 0
  if (event.error && retries < 3) {
    event.invocationState.toolRetries = retries + 1
    event.retry = true
  }
})

await agent.invoke("Look up 'alpha'")
console.log(`tool ran ${attempts} times`)
```

## Plugins

A plugin is any object with a `name` and an `initAgent(agent)` method that registers hooks; an optional `getTools()` adds tools:

```typescript
import { Agent, BeforeModelCallEvent, tool } from '@strands-agents/sdk'
import type { LocalAgent, Plugin } from '@strands-agents/sdk'
import { z } from 'zod'

class ModelCallCounter implements Plugin {
  readonly name = 'model-call-counter'
  count = 0

  initAgent(agent: LocalAgent): void {
    agent.addHook(BeforeModelCallEvent, () => {
      this.count += 1
    })
  }

  getTools() {
    return [
      tool({
        name: 'model_call_count',
        description: 'How many model calls this agent has made.',
        inputSchema: z.object({}),
        callback: () => this.count,
      }),
    ]
  }
}

const counter = new ModelCallCounter()
const agent = new Agent({ model: 'us.anthropic.%%model=anthropic-small%%', plugins: [counter], printer: false })
await agent.invoke('Hi')
console.log(`model calls: ${counter.count}`, agent.tools.map((t) => t.name))
```

Ready-made plugins live under `@strands-agents/sdk/vended-plugins/*`: `skills` (Agent Skills from `SKILL.md` folders), `context-offloader`, `context-injector` and `goal`.

## Interventions

Subclass `InterventionHandler`, set `name`, and override `beforeInvocation`, `beforeModelCall`, `afterModelCall`, `beforeToolCall` or `afterToolCall`. Return an action from `InterventionActions`: `proceed()`, `deny(reason)`, `guide(feedback)`, `confirm(prompt)` (before tool calls only) or `transform(fn)`:

```typescript
import { Agent, BeforeToolCallEvent, InterventionActions, InterventionHandler, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const { confirm, deny, proceed } = InterventionActions

const deleteBucket = tool({
  name: 'delete_bucket',
  description: 'Delete an S3 bucket',
  inputSchema: z.object({ bucket: z.string() }),
  callback: ({ bucket }) => `deleted ${bucket}`,
})

class Approvals extends InterventionHandler {
  readonly name = 'approvals'

  override beforeToolCall(event: BeforeToolCallEvent) {
    if (!event.toolUse.name.startsWith('delete_')) return proceed()
    if (event.invocationState.readOnly === true) return deny('read-only caller')
    return confirm(`Allow ${event.toolUse.name} with ${JSON.stringify(event.toolUse.input)}?`)
  }
}

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  tools: [deleteBucket],
  interventions: [new Approvals()],
  printer: false,
})

let result = await agent.invoke('Delete the bucket called scratch-data')
if (result.stopReason === 'interrupt' && result.interrupts?.[0]) {
  const pending = result.interrupts[0]
  console.log(pending.name, '-', pending.reason)
  result = await agent.invoke([{ interruptResponse: { interruptId: pending.id, response: 'yes' } }])
}
console.log(result.toString())
```

Handlers run in order and a `deny` stops evaluation, so put cheap checks first. `onError` (`'throw'` by default, `'deny'`, `'proceed'`) decides what a failing handler does. `@strands-agents/sdk/vended-interventions/hitl`, `/cedar` and `/steering` provide ready-made handlers.

## Interrupts

An interrupt pauses the agent with `stopReason === 'interrupt'` and `result.interrupts` (each with `id`, `name`, `reason`). Resume by invoking the agent with `interruptResponse` blocks; the paused tool or hook runs again and `interrupt()` returns the response.

### From a tool

```typescript
import { Agent, tool } from '@strands-agents/sdk'
import type { ToolContext } from '@strands-agents/sdk'
import { z } from 'zod'

const refund = tool({
  name: 'refund',
  description: 'Refund an order.',
  inputSchema: z.object({ orderId: z.string(), amount: z.number().describe('Amount in GBP') }),
  callback: ({ orderId, amount }, context?: ToolContext) => {
    if (!context) throw new Error('tool context required')
    if (amount > 100) {
      const decision = context.interrupt<string>({ name: 'refund-approval', reason: { orderId, amount } })
      if (decision !== 'approve') return `Refund of £${amount} for ${orderId} was not approved`
    }
    return `Refunded £${amount} for ${orderId}`
  },
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [refund], printer: false })
let result = await agent.invoke('Refund £250 on ORD-7')

while (result.stopReason === 'interrupt') {
  const responses = (result.interrupts ?? []).map((pending) => ({
    interruptResponse: { interruptId: pending.id, response: 'approve' }, // from a UI, ticket or queue
  }))
  result = await agent.invoke(responses)
}
console.log(result.toString())
```

Code before `interrupt()` runs again on resume, so keep side effects after it. With a `SessionManager`, the pending interrupt is saved with the agent, so a new agent with the same session ID can be resumed in a later request.

### From a hook

`BeforeToolCallEvent` and `BeforeToolsEvent` implement the same `interrupt()`:

```typescript
import { Agent, BeforeToolCallEvent, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const sendEmail = tool({
  name: 'send_email',
  description: 'Send an email.',
  inputSchema: z.object({ to: z.string(), body: z.string() }),
  callback: ({ to }) => `sent to ${to}`,
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [sendEmail], printer: false })

agent.addHook(BeforeToolCallEvent, (event) => {
  if (event.toolUse.name !== 'send_email') return
  const answer = event.interrupt<string>({ name: 'approve-email', reason: event.toolUse.input })
  if (answer !== 'yes') event.cancel = 'Email not approved'
})

const first = await agent.invoke('Email ops@example.com that the deploy finished')
const id = first.interrupts?.[0]?.id
if (id) {
  const resumed = await agent.invoke([{ interruptResponse: { interruptId: id, response: 'yes' } }])
  console.log(resumed.stopReason, resumed.toString())
}
```

## Retry strategies

The default `retryStrategy` is a `DefaultModelRetryStrategy`: 6 attempts with exponential backoff from 4 seconds to 240 seconds, for `ModelThrottledError` only. Shorten it for interactive use:

```typescript
import { Agent, DefaultModelRetryStrategy, ExponentialBackoff } from '@strands-agents/sdk'

const agent = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  retryStrategy: new DefaultModelRetryStrategy({
    maxAttempts: 3,
    backoff: new ExponentialBackoff({ baseMs: 500, maxMs: 4_000, jitter: 'full' }),
  }),
  printer: false,
})
```

`retryStrategy: null` disables retries. `ConstantBackoff` and `LinearBackoff` are the other backoff shapes; subclass `ModelRetryStrategy` (and override `isRetryable`) to retry other errors. A strategy instance belongs to one agent.

## Middleware

Middleware wraps a stage: `InvokeModelStage` (a model call) or `ExecuteToolStage` (a tool call). A wrap handler is an async generator that delegates to `next` with `yield*` and can act before and after; `.Input` and `.Output` sub-stages transform the context or the result:

```typescript
import { Agent, ExecuteToolStage, InvokeModelStage, tool } from '@strands-agents/sdk'
import { z } from 'zod'

const lookup = tool({
  name: 'lookup',
  description: 'Look up a term.',
  inputSchema: z.object({ term: z.string() }),
  callback: ({ term }) => `${term}: defined`,
})

const agent = new Agent({ model: 'us.anthropic.%%model=anthropic%%', tools: [lookup], printer: false })
const timings: number[] = []

agent.addMiddleware(InvokeModelStage, async function* (context, next) {
  const started = performance.now()
  const result = yield* next(context)
  timings.push(performance.now() - started)
  return result
})

agent.addMiddleware(ExecuteToolStage.Input, (context) => {
  console.log('running tool', context.toolUse.name)
  return context
})

await agent.invoke("Define 'idempotent'")
console.log(`${timings.length} model call(s)`)
```

`addMiddleware` returns a function that removes the handler.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.22.0 | New page, from the hooks section of the TypeScript Strands page in the Amazon Bedrock guide. Adds the event table with writable fields, tool retries, plugins with tools, intervention handlers, interrupts from tools and hooks, retry strategy defaults and middleware. |

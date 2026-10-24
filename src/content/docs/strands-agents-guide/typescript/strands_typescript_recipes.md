---
title: "Strands Agents (TypeScript): Recipes"
description: "Complete TypeScript Strands programs: a support agent with structured triage and sessions, an approval workflow that resumes in a later request, a research graph with parallel branches, and batch extraction with bounded concurrency."
framework: "strands"
language: "typescript"
sidebar:
  label: "Recipes"
  order: 5
---

Checked against `@strands-agents/sdk` %%version=@strands-agents/sdk%% (%%versions-checked%%). Each recipe is a complete ES module that compiles under `tsc --strict`; all were run against an offline Bedrock stub, so they execute, though a real model's answers will differ. The [Express streaming server](/strands-agents-guide/typescript/strands_typescript_production/#express-on-fargate-or-eks) is on the production page.

## Support agent with triage and memory

```typescript
import { Agent, SessionManager, tool } from '@strands-agents/sdk'
import type { ToolContext } from '@strands-agents/sdk'
import { LocalFileStorage } from '@strands-agents/sdk/storage'
import { z } from 'zod'

const ORDERS: Record<string, { status: string; carrier?: string }> = {
  'ORD-1': { status: 'shipped', carrier: 'DPD' },
  'ORD-2': { status: 'processing' },
}

const Triage = z.object({
  category: z.enum(['order', 'billing', 'technical', 'other']),
  urgency: z.enum(['low', 'normal', 'high']),
  summary: z.string().describe('One sentence'),
})

const orderStatus = tool({
  name: 'order_status',
  description: "Look up the status of one of the customer's orders.",
  inputSchema: z.object({ orderId: z.string().describe('Order ID, for example ORD-1') }),
  callback: ({ orderId }, context?: ToolContext) => {
    const customer = context?.invocationState.customerId as string // set by the server, not the model
    const order = ORDERS[orderId]
    if (!order) throw new Error(`No order ${orderId} for customer ${customer}`)
    return { orderId, ...order }
  },
})

const triageAgent = new Agent({
  model: 'us.anthropic.%%model=anthropic-small%%',
  systemPrompt: 'Classify customer support messages.',
  printer: false,
})

function supportAgent(customerId: string): Agent {
  return new Agent({
    model: 'us.anthropic.%%model=anthropic%%',
    systemPrompt: 'You are a support agent for Example Ltd. Use tools; never guess order details.',
    tools: [orderStatus],
    // session IDs may contain only lowercase letters, digits, hyphens and underscores
    sessionManager: new SessionManager({ sessionId: `support-${customerId.toLowerCase()}`, storage: new LocalFileStorage('./sessions') }),
    printer: false,
  })
}

async function handle(customerId: string, message: string) {
  const triage = Triage.parse((await triageAgent.invoke(message, { structuredOutputSchema: Triage })).structuredOutput)
  triageAgent.messages.length = 0 // the classifier keeps no history
  if (triage.urgency === 'high') return { triage, answer: 'Passed to a human agent.' }
  const result = await supportAgent(customerId).invoke(message, { invocationState: { customerId } })
  return { triage, answer: result.toString() }
}

console.log(await handle('C-42', 'Where is my order ORD-1?'))
console.log(await handle('C-42', 'And when will it arrive?')) // same session, history restored
```

## Approval workflow across requests

The first request stops at an interrupt and returns its ID; the decision arrives in a later request, and a new agent with the same session ID resumes the paused tool call:

```typescript
import { Agent, SessionManager, tool } from '@strands-agents/sdk'
import type { ToolContext } from '@strands-agents/sdk'
import { LocalFileStorage } from '@strands-agents/sdk/storage'
import { z } from 'zod'

const refund = tool({
  name: 'refund',
  description: 'Refund an order.',
  inputSchema: z.object({ orderId: z.string(), amount: z.number().describe('Amount in GBP') }),
  callback: ({ orderId, amount }, context?: ToolContext) => {
    if (!context) throw new Error('tool context required')
    if (amount > 100) {
      const decision = context.interrupt<string>({ name: 'manager-approval', reason: { orderId, amount } })
      if (decision !== 'approve') return `Refund for ${orderId} declined by manager`
    }
    return `Refunded £${amount.toFixed(2)} on ${orderId}`
  },
})

function agentFor(caseId: string): Agent {
  return new Agent({
    model: 'us.anthropic.%%model=anthropic%%',
    tools: [refund],
    sessionManager: new SessionManager({ sessionId: caseId, storage: new LocalFileStorage('./sessions') }),
    printer: false,
  })
}

async function start(caseId: string, request: string) {
  const result = await agentFor(caseId).invoke(request)
  if (result.stopReason === 'interrupt') {
    return { status: 'waiting', pending: (result.interrupts ?? []).map((i) => ({ id: i.id, name: i.name, reason: i.reason })) }
  }
  return { status: 'done', answer: result.toString() }
}

async function decide(caseId: string, interruptId: string, decision: 'approve' | 'deny') {
  const result = await agentFor(caseId).invoke([{ interruptResponse: { interruptId, response: decision } }])
  return { status: result.stopReason, answer: result.toString() }
}

const ticket = await start('case-981', 'Please refund £250 on ORD-7, it arrived broken.')
console.log(JSON.stringify(ticket))
if (ticket.status === 'waiting' && ticket.pending?.[0]) {
  // ... later, in another request, the manager approves
  console.log(await decide('case-981', ticket.pending[0].id, 'approve'))
}
```

## Research graph with parallel branches

```typescript
import { Agent, Graph } from '@strands-agents/sdk'
import { z } from 'zod'

const FAST = 'us.anthropic.%%model=anthropic-small%%'
const MAIN = 'us.anthropic.%%model=anthropic%%'

const Brief = z.object({ title: z.string(), keyPoints: z.array(z.string()), risks: z.array(z.string()) })

const researcher = (focus: string) =>
  new Agent({ id: focus, model: FAST, systemPrompt: `Research the ${focus} angle of the topic. Be factual and brief.`, printer: false })

const graph = new Graph({
  nodes: [
    new Agent({ id: 'scope', model: FAST, systemPrompt: 'Restate the research question and list what to check.', printer: false }),
    researcher('market'),
    researcher('technical'),
    researcher('regulatory'),
    new Agent({ id: 'writer', model: MAIN, systemPrompt: 'Write a short, balanced brief from the research.', printer: false }),
  ],
  edges: [
    ['scope', 'market'],
    ['scope', 'technical'],
    ['scope', 'regulatory'],
    ['market', 'writer'],
    ['technical', 'writer'],
    ['regulatory', 'writer'],
  ],
  maxConcurrency: 3,
  timeout: 10 * 60_000,
  nodeTimeout: 3 * 60_000,
})

const result = await graph.invoke('Should a UK retailer adopt drone delivery in 2027?')
const draft = result.content.map((block) => (block.type === 'textBlock' ? block.text : '')).join('')

const formatter = new Agent({ model: FAST, printer: false })
const brief = Brief.parse(
  (await formatter.invoke(`Turn this into a brief:\n\n${draft}`, { structuredOutputSchema: Brief })).structuredOutput,
)
console.log(brief.title, brief.keyPoints.length, 'points;', 'tokens:', result.usage.totalTokens)
```

## Batch extraction with bounded concurrency

Each document gets its own agent (an agent runs one call at a time), and a small worker pool keeps the request rate under your Bedrock quota:

```typescript
import { Agent, StructuredOutputError } from '@strands-agents/sdk'
import { z } from 'zod'

const Invoice = z.object({ supplier: z.string(), invoiceNumber: z.string(), totalGbp: z.number() })
type Invoice = z.infer<typeof Invoice>

const documents = [
  'Invoice 1001 from Acme Ltd. Total due: £120.00',
  'Northwind Traders, invoice NW-77, amount £89.50 inc VAT',
  'Receipt from Café Nero, £4.20',
]

async function extract(text: string): Promise<Invoice | null> {
  const agent = new Agent({
    model: 'us.anthropic.%%model=anthropic-small%%',
    systemPrompt: 'Extract invoice fields exactly as written.',
    printer: false,
  })
  try {
    const result = await agent.invoke(text, { structuredOutputSchema: Invoice })
    return Invoice.parse(result.structuredOutput)
  } catch (error) {
    if (error instanceof StructuredOutputError) return null
    throw error
  }
}

async function mapWithLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let next = 0
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next++
      results[index] = await fn(items[index]!)
    }
  })
  await Promise.all(workers)
  return results
}

const invoices = await mapWithLimit(documents, 4, extract)
invoices.forEach((invoice, i) => console.log(invoice ?? 'FAILED', '<-', documents[i]!.slice(0, 40)))
```

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.22.0 | New page: support agent with triage and sessions, approvals across requests, research graph and batch extraction. |

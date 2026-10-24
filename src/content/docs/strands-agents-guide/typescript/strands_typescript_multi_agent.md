---
title: "Strands Agents (TypeScript): Multi-Agent Systems and A2A"
description: "Coordinating TypeScript Strands agents: agents as tools, Graph with conditional edges and loops, Swarm handoffs through structured output, orchestrator streaming and hooks, and the A2A server and client."
framework: "strands"
language: "typescript"
sidebar:
  label: "Multi-agent and A2A"
  order: 4
---

Checked against `@strands-agents/sdk` %%version=@strands-agents/sdk%% (%%versions-checked%%). Examples compile under `tsc --strict` and were run against an offline Bedrock stub that scripts each agent; the A2A example ran a real local A2A server.

The patterns match Python (agents as tools, graph, swarm, A2A), with three differences that matter in practice:

- **No defaults bound a run.** `Graph` and `Swarm` default `maxSteps`, `timeout` and `nodeTimeout` to `Infinity`. Set them.
- **Timeouts are in milliseconds**, not seconds.
- **Swarm handoffs use structured output**, not a `handoff_to_agent` tool, and there is no `workflow` tool or `GraphBuilder`: graphs are declared as `nodes` and `edges`.

## Agents as tools

An `Agent` can go straight into another agent's `tools`; its `name` and `description` become the tool definition. `asTool({ name, description, preserveContext })` renames it or keeps its conversation between calls (by default it is reset before each call):

```typescript
import { Agent } from '@strands-agents/sdk'

const researcher = new Agent({
  name: 'researcher',
  description: 'Finds and summarises factual information on a topic.',
  model: 'us.anthropic.%%model=anthropic%%',
  systemPrompt: 'You research topics and report concise, sourced facts.',
  printer: false,
})

const editor = new Agent({
  name: 'editor',
  description: 'Rewrites draft text for clarity and tone.',
  model: 'us.anthropic.%%model=anthropic-small%%',
  systemPrompt: 'You edit text to be clear and concise.',
  printer: false,
})

const orchestrator = new Agent({
  model: 'us.anthropic.%%model=anthropic-flagship%%',
  systemPrompt: 'Use the researcher for facts and polish_text to tidy the final answer.',
  tools: [researcher, editor.asTool({ name: 'polish_text', preserveContext: true })],
  printer: false,
})

console.log(orchestrator.tools.map((t) => t.name)) // [ 'researcher', 'polish_text' ]
console.log((await orchestrator.invoke('Write a short brief on AWS Graviton for a CFO.')).toString())
```

Python's `as_tool(delegate=True)` has no TypeScript equivalent in this release.

## Graph

`new Graph({ nodes, edges })` runs agents as nodes, identified by their `id`. A node runs once its incoming edges are satisfied; independent nodes run in parallel (bounded by `maxConcurrency`). An edge is a `[source, target]` pair, or `{ source, target, handler }` where `handler(state)` decides whether it is taken, which makes loops possible:

```typescript
import { Agent, Graph } from '@strands-agents/sdk'
import type { MultiAgentState } from '@strands-agents/sdk/multiagent'

const model = 'us.anthropic.%%model=anthropic%%'
const node = (id: string, systemPrompt: string) => new Agent({ id, model, systemPrompt, printer: false })

function textOf(state: MultiAgentState, nodeId: string): string {
  const content = state.node(nodeId)?.content ?? []
  return content.map((block) => (block.type === 'textBlock' ? block.text : '')).join('')
}

const passed = (state: MultiAgentState) => textOf(state, 'validate').trim().toUpperCase().startsWith('PASS')

const graph = new Graph({
  nodes: [
    node('collect', 'Collect and structure the raw data described in the task.'),
    node('validate', 'Check the data for completeness. Start your reply with PASS or FAIL.'),
    node('analyse', 'Analyse the data and identify trends.'),
    node('report', 'Write a concise report of the key findings.'),
  ],
  edges: [
    ['collect', 'validate'],
    { source: 'validate', target: 'analyse', handler: passed },
    { source: 'validate', target: 'collect', handler: (state) => !passed(state) }, // retry loop
    ['analyse', 'report'],
  ],
  sources: ['collect'],
  maxSteps: 10, // bound the loop
  timeout: 15 * 60_000, // milliseconds, whole graph
  nodeTimeout: 5 * 60_000,
})

const result = await graph.invoke('Trend analysis of Q4 sales from s3://sales-bucket/q4.csv')
console.log(result.status)
for (const nodeResult of result.results) {
  console.log(nodeResult.nodeId, nodeResult.status, `${nodeResult.duration}ms`)
}
```

`sources` names the entry nodes (otherwise nodes with no incoming edges are used, which a loop back to `collect` would hide). `result` is a `MultiAgentResult`: `status` (`COMPLETED`, `FAILED`, `CANCELLED` or `INTERRUPTED`, from the `Status` constant), `results` (one `NodeResult` per execution, with `nodeId`, `status`, `content`, `duration`, `usage`, `error`), `content` (the final output), `usage` and `duration`. A node can also be a `Swarm` or another `Graph`.

## Swarm

In a `Swarm` the agents route the work. After each turn the active agent returns structured output with an optional `agentId` (who to hand off to), a `message` and `context`; no `agentId` means it is finished. `start` names the entry agent:

```typescript
import { Agent, Swarm } from '@strands-agents/sdk'

const model = 'us.anthropic.%%model=anthropic%%'

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

const swarm = new Swarm({
  nodes: [triage, billing, technical],
  start: 'triage',
  maxSteps: 4,
  timeout: 5 * 60_000,
})

const result = await swarm.invoke('Why was I charged twice in September?')
console.log(result.status, result.results.map((r) => r.nodeId))
```

When `maxSteps` is reached the swarm throws (`max_steps=<4> | swarm reached step limit`), so catch it if a long ping-pong is possible, or use `repetitiveHandoffDetectionWindow` with `repetitiveHandoffMinUniqueAgents` to stop loops between two agents earlier.

## Streaming and hooks for orchestrators

`graph.stream()` and `swarm.stream()` yield orchestrator events: `beforeNodeCallEvent`, `nodeStreamUpdateEvent` (with `inner.event`, the agent's own stream event), `nodeResultEvent`, `multiAgentHandoffEvent` (`source`, `targets`), `afterNodeCallEvent` and `multiAgentResultEvent`.

```typescript
import { Agent, Graph } from '@strands-agents/sdk'
import { BeforeNodeCallEvent } from '@strands-agents/sdk/multiagent'

const model = 'us.anthropic.%%model=anthropic%%'
const graph = new Graph({
  nodes: [
    new Agent({ id: 'draft', model, systemPrompt: 'Draft a release note.', printer: false }),
    new Agent({ id: 'review', model, systemPrompt: 'Review the draft.', printer: false }),
  ],
  edges: [['draft', 'review']],
  timeout: 120_000,
})

graph.addHook(BeforeNodeCallEvent, (event) => {
  console.log('->', event.nodeId)
})

for await (const event of graph.stream('Version 2.4 adds CSV export.')) {
  if (event.type === 'multiAgentHandoffEvent') console.log('handoff', event.source, '->', event.targets)
  if (event.type === 'multiAgentResultEvent') console.log('done:', event.result.status)
}
```

`BeforeNodeCallEvent.cancel` stops a node, and `BeforeNodeCallEvent.interrupt()` pauses the whole orchestrator for a person. Multi-agent plugins (`plugins: [...]`) bundle such hooks.

## A2A

Install `@a2a-js/sdk` (0.3.x) and `express`.

### Serving an agent

```typescript
import { Agent } from '@strands-agents/sdk'
import { A2AExpressServer } from '@strands-agents/sdk/a2a/express'

// One agent per A2A context (conversation)
const agentFactory = (contextId: string) =>
  new Agent({
    model: 'us.anthropic.%%model=anthropic%%',
    systemPrompt: `You answer geography questions. Conversation ${contextId}.`,
    printer: false,
  })

const server = new A2AExpressServer({
  agentFactory,
  name: 'geography',
  description: 'Answers questions about countries, capitals and borders.',
  host: '127.0.0.1',
  port: 9000,
})

await server.serve() // agent card at /.well-known/agent-card.json
```

The factory runs once per A2A context (and once at start-up), and the server keeps up to `maxContexts` agents. Passing a single `agent` is deprecated: it serialises every caller's requests through one conversation. `createMiddleware()` returns an Express router to mount in an existing app instead of calling `serve()`, and `httpUrl` advertises the public URL behind a load balancer. The SDK logs that its A2A module is experimental and may change with the underlying `@a2a-js/sdk`.

### Calling a remote agent

```typescript
import { Agent, Graph } from '@strands-agents/sdk'
import { A2AAgent } from '@strands-agents/sdk/a2a'
import { makeA2AClient } from '@strands-agents/sdk/vended-tools/a2a-client'

const remote = new A2AAgent({ url: 'http://127.0.0.1:9000' })

const card = await remote.getAgentCard()
console.log(card.name, '-', card.description)

const result = await remote.invoke('What is the capital of France?')
console.log(result.toString())

// A remote agent can be a graph or swarm node, like a local one
const graph = new Graph({
  nodes: [remote, new Agent({ id: 'summarise', model: 'us.anthropic.%%model=anthropic%%', printer: false })],
  edges: [[remote.id, 'summarise']],
  timeout: 120_000,
})

// For the model to call remote agents itself, give it the vended A2A client tool
const local = new Agent({
  model: 'us.anthropic.%%model=anthropic%%',
  tools: [makeA2AClient({ allowedEndpoints: ['http://127.0.0.1:9000'] })],
  printer: false,
})
console.log(local.tools.map((t) => t.name))
```

`A2AAgent` implements the same invokable-agent interface as `Agent`, so it works as a `Graph` or `Swarm` node, but it is not a tool; `makeA2AClient` restricts the model to the endpoints you list.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.22.0 | New page, from the multi-agent section of the TypeScript Strands page in the Amazon Bedrock guide. Adds the unbounded-defaults warning, conditional edges and loops, result fields, swarm step-limit behaviour, orchestrator streaming and hooks, and the A2A server and client. |

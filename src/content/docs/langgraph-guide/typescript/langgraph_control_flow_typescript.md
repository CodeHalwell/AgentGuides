---
title: "LangGraph (TypeScript): interrupts, Command, Send, subgraphs and time travel"
description: "Control flow in LangGraph.js: pausing for human input with interrupt and Command, routing with Command, fan-out with Send, subgraphs and Command.PARENT, the Functional API, approvals in createAgent agents, and time travel."
framework: "langgraph"
language: "typescript"
sidebar:
  label: "Interrupts, Send, Command & subgraphs"
  order: 15
---

Checked against `@langchain/langgraph` %%version=@langchain/langgraph%% and `langchain` %%version=langchainjs%% (%%versions-checked%%). Every example compiles under `tsc --strict` and runs as shown (with the [recommended tsconfig](/langgraph-guide/typescript/langchain_langgraph_comprehensive_guide/#installation-and-package-setup) and `"type": "module"`). The Python guide covers the same features in [Chapter 7](/langgraph-guide/python/chapter-07-human-in-the-loop/) and the [Command and Send reference](/langgraph-guide/python/reference-command-and-send/).

## Human-in-the-loop with `interrupt`

`interrupt(value)` pauses the run inside a node and hands `value` to the caller. The graph must have a checkpointer, and you resume on the same `thread_id` with `new Command({ resume })`; the value you pass becomes the return value of `interrupt()`.

```typescript
import {
  Annotation, Command, INTERRUPT, MemorySaver, StateGraph, START, END, interrupt, isInterrupted,
} from '@langchain/langgraph';

const State = Annotation.Root({
  amount: Annotation<number>,
  approved: Annotation<boolean>,
});

const graph = new StateGraph(State)
  .addNode('approve', (state) => {
    // Pauses the run; the value is shown to the caller.
    const answer = interrupt({ question: `Approve a refund of ${state.amount}?` });
    return { approved: answer === 'yes' };
  })
  .addEdge(START, 'approve')
  .addEdge('approve', END)
  .compile({ checkpointer: new MemorySaver() });   // required to resume

const config = { configurable: { thread_id: 'refund-1' } };

const paused = await graph.invoke({ amount: 120 }, config);
if (isInterrupted(paused)) {
  console.log(paused[INTERRUPT][0].value);           // { question: 'Approve a refund of 120?' }
}

const snapshot = await graph.getState(config);
console.log(snapshot.next);                          // [ 'approve' ]

const done = await graph.invoke(new Command({ resume: 'yes' }), config);
console.log(done);                                   // { amount: 120, approved: true }
```

Things to know:

- **The node re-runs from the top on resume.** Code before `interrupt()` runs again, so keep it free of side effects (or move side effects after the interrupt, or into a `task`).
- **`isInterrupted(result)`** narrows the result type so `result[INTERRUPT]` type-checks; `graph.getState(config)` also shows the pending interrupts in `snapshot.tasks`.
- **Several interrupts in one node** are matched to resume values in order. With interrupts pending in parallel nodes, resume with a map from interrupt ID to value: `new Command({ resume: { [id]: value } })`.
- **Without a checkpointer** `interrupt()` throws `No checkpointer set` (error code `MISSING_CHECKPOINTER`), and `invoke(new Command({ resume }))` throws as well. Compile with at least a `MemorySaver`.

## Approvals in `createAgent` agents

For an agent built with `createAgent`, `humanInTheLoopMiddleware` adds the interrupt for you, before the tools you name run:

```typescript
import { createAgent, humanInTheLoopMiddleware, tool, fakeModel } from 'langchain';
import { AIMessage } from '@langchain/core/messages';
import { Command, MemorySaver } from '@langchain/langgraph';
import { z } from 'zod';

const sendEmail = tool(async ({ to }) => `sent to ${to}`, {
  name: 'send_email',
  description: 'Send an e-mail',
  schema: z.object({ to: z.string(), body: z.string() }),
});

// Scripted model so the example runs offline; use 'anthropic:%%model=anthropic%%' in real code.
const model = fakeModel()
  .respondWithTools([{ name: 'send_email', args: { to: 'sam@example.com', body: 'Hi' } }])
  .respond(new AIMessage('Done.'));

const agent = createAgent({
  model,
  tools: [sendEmail],
  middleware: [humanInTheLoopMiddleware({ interruptOn: { send_email: true } })],
  checkpointer: new MemorySaver(),   // required for the pause
});

const config = { configurable: { thread_id: 'hitl-1' } };
const paused = await agent.invoke({ messages: [{ role: 'user', content: 'E-mail Sam' }] }, config);
console.log(paused.__interrupt__?.[0].value);   // { actionRequests: [ { name: 'send_email', args: {...} } ], ... }

const done = await agent.invoke(new Command({ resume: { decisions: [{ type: 'approve' }] } }), config);
console.log(done.messages.at(-1)?.content);     // Done.
```

Each decision is `{ type: 'approve' }`, `{ type: 'edit', editedAction: { name, args } }`, `{ type: 'reject', message }` or `{ type: 'respond', message }`, one per pending tool call. More middleware is on the [middleware page](/langgraph-guide/typescript/langgraph_middleware_typescript/).

### One interrupt per tool call

By default the middleware raises a single interrupt for all gated calls in a model turn. Pass `interruptMode: 'per_call'` to raise one interrupt per gated call instead, so each can be answered separately (added in `langchain` 1.5.18). Each interrupt value has `type: 'tool_approval'`, plus `tool_call_id`, `name`, `args` and `description`. Filter them with `isToolApprovalInterrupt`, and resume with a map from interrupt ID to a single decision:

```typescript
import { createAgent, humanInTheLoopMiddleware, isToolApprovalInterrupt, tool, fakeModel } from 'langchain';
import { AIMessage } from '@langchain/core/messages';
import { Command, MemorySaver } from '@langchain/langgraph';
import { z } from 'zod';

const sendEmail = tool(async ({ to }) => `sent to ${to}`, {
  name: 'send_email',
  description: 'Send an e-mail',
  schema: z.object({ to: z.string(), body: z.string() }),
});

const model = fakeModel()
  .respondWithTools([
    { name: 'send_email', args: { to: 'sam@example.com', body: 'Hi' } },
    { name: 'send_email', args: { to: 'kim@example.com', body: 'Hi' } },
  ])
  .respond(new AIMessage('Done.'));

const agent = createAgent({
  model,
  tools: [sendEmail],
  middleware: [humanInTheLoopMiddleware({ interruptOn: { send_email: true }, interruptMode: 'per_call' })],
  checkpointer: new MemorySaver(),
});

const config = { configurable: { thread_id: 'hitl-2' } };
const paused = await agent.invoke({ messages: [{ role: 'user', content: 'Mail both' }] }, config);
const pending = (paused.__interrupt__ ?? []).filter(isToolApprovalInterrupt);
console.log(pending.length);   // 2

const resume = Object.fromEntries(pending.map((i) => [i.id, { type: 'approve' }]));
const done = await agent.invoke(new Command({ resume }), config);
console.log(done.messages.at(-1)?.content);   // Done.
```

In this mode the middleware checks each answer before saving it. An `edit` decision (`{ type: 'edit', edited_action: { name, args } }`, with a snake-case key) cannot change the tool name, and its `args` must match the tool's schema. An invalid answer throws a `ZodError` and is not saved, so the interrupt can be answered again. If you send several answers together and one is invalid, the tools for the others may already have run, so validate first or answer one at a time. List this middleware before any tool-retry or error-handling middleware, and keep earlier middleware free of side effects, because it runs again for each paused call on resume.

## Routing with `Command`

A node can return a `Command` that updates state and chooses the next node in one step. Declare the possible targets with `ends` so the builder can type-check the graph and draw the edges:

```typescript
import { Annotation, Command, StateGraph, START, END } from '@langchain/langgraph';

const State = Annotation.Root({
  query: Annotation<string>,
  route: Annotation<string>,
  answer: Annotation<string>,
});

const graph = new StateGraph(State)
  // `ends` lists where this node's Command can go (needed for typing and the diagram)
  .addNode(
    'triage',
    (state) =>
      new Command({
        update: { route: state.query.includes('refund') ? 'billing' : 'general' },
        goto: state.query.includes('refund') ? 'billing' : 'general',
      }),
    { ends: ['billing', 'general'] },
  )
  .addNode('billing', () => ({ answer: 'Billing will contact you.' }))
  .addNode('general', () => ({ answer: 'Here is our FAQ.' }))
  .addEdge(START, 'triage')
  .addEdge('billing', END)
  .addEdge('general', END)
  .compile();

console.log(await graph.invoke({ query: 'I want a refund' }));
// { query: 'I want a refund', route: 'billing', answer: 'Billing will contact you.' }
```

`goto` adds to any static edges from the node; it does not replace them. Leave out static edges from nodes that route with `Command`.

## Fan-out with `Send`

Return `Send` objects from a routing function to run one task per item, each with its own input. The tasks run in parallel in one super-step, and the next node runs once they have all finished:

```typescript
import { Annotation, Send, StateGraph, START, END } from '@langchain/langgraph';

const State = Annotation.Root({
  topics: Annotation<string[]>,
  // Parallel workers append here, so the field needs a reducer
  notes: Annotation<string[]>({ reducer: (a, b) => a.concat(b), default: () => [] }),
  summary: Annotation<string>,
});

// Each Send carries its own input, which need not match the graph state
const WorkerInput = Annotation.Root({ topic: Annotation<string> });

const graph = new StateGraph(State)
  .addNode(
    'research',
    async (input: typeof WorkerInput.State) => ({ notes: [`notes on ${input.topic}`] }),
    { input: WorkerInput },
  )
  .addNode('summarise', (state) => ({ summary: `${state.notes.length} topics covered` }))
  // A routing function returning Send objects fans out one task per item
  .addConditionalEdges(START, (state) => state.topics.map((topic) => new Send('research', { topic })), [
    'research',
  ])
  .addEdge('research', 'summarise')   // runs once, after every research task
  .addEdge('summarise', END)
  .compile();

const result = await graph.invoke({ topics: ['pricing', 'support', 'security'] });
console.log(result.notes, result.summary);   // [ 'notes on pricing', ... ] 3 topics covered
```

The field the workers write to needs a reducer; without one, parallel writes raise `InvalidUpdateError`.

## Subgraphs and `Command.PARENT`

A compiled graph can be a node in another graph. When parent and child share state keys, add the compiled child directly; otherwise call it from a node function and map the state yourself. A child node can jump to a node in the parent with `Command.PARENT`:

```typescript
import { Annotation, Command, MessagesAnnotation, StateGraph, START, END } from '@langchain/langgraph';
import { AIMessage, HumanMessage } from '@langchain/core/messages';

// Parent and child share the `messages` key, so the child can be added as a node directly.
// MessagesAnnotation's reducer merges by message ID, so the child's returned history is not duplicated.
const Shared = Annotation.Root({
  ...MessagesAnnotation.spec,
  escalated: Annotation<boolean>,
});

const specialist = new StateGraph(Shared)
  .addNode('work', (state) => {
    if (String(state.messages.at(-1)?.content).includes('complex')) {
      // Jump to a node in the *parent* graph and update the parent's state
      return new Command({ graph: Command.PARENT, goto: 'humanReview', update: { escalated: true } });
    }
    return { messages: [new AIMessage('specialist: done')] };
  })
  .addEdge(START, 'work')
  .addEdge('work', END)
  .compile();

const parent = new StateGraph(Shared)
  .addNode('specialist', specialist, { ends: ['humanReview'] })   // a compiled graph is a node
  .addNode('humanReview', () => ({ messages: [new AIMessage('human: reviewed')] }))
  .addEdge(START, 'specialist')
  .addEdge('specialist', END)
  .addEdge('humanReview', END)
  .compile();

const simple = await parent.invoke({ messages: [new HumanMessage('simple task')] });
console.log(simple.messages.map((m) => m.content));    // [ 'simple task', 'specialist: done' ]
const complex = await parent.invoke({ messages: [new HumanMessage('complex task')] });
console.log(complex.messages.map((m) => m.content), complex.escalated);   // [ 'complex task', 'human: reviewed' ] true
```

The parent lists the child's possible jump targets in `ends`. With a checkpointer on the parent, subgraphs share it automatically; `graph.getState(config, { subgraphs: true })` includes their state.

## Functional API: `entrypoint` and `task`

The Functional API builds the same durable workflows without declaring a graph. `entrypoint` wraps the workflow; `task` wraps a unit of work whose result is checkpointed, so it is not repeated when the workflow resumes after an interrupt:

```typescript
import { Command, MemorySaver, entrypoint, interrupt, task } from '@langchain/langgraph';

// A task's result is saved in the checkpoint, so it does not run again on resume.
const draftEmail = task('draftEmail', async (topic: string) => {
  return `Dear customer, about ${topic}: ...`;
});

const sendEmail = task('sendEmail', async (body: string) => `sent: ${body.slice(0, 20)}...`);

const workflow = entrypoint(
  { name: 'emailWorkflow', checkpointer: new MemorySaver() },
  async (topic: string) => {
    const draft = await draftEmail(topic);
    const decision = interrupt({ draft, question: 'Send this e-mail?' });   // pause for review
    if (decision !== 'send') return { status: 'cancelled' };
    return { status: await sendEmail(draft) };
  },
);

const config = { configurable: { thread_id: 'email-1' } };
for await (const chunk of await workflow.stream('your refund', config)) {
  console.log(chunk);   // { draftEmail: '...' } then { __interrupt__: [ ... ] }
}
console.log(await workflow.invoke(new Command({ resume: 'send' }), config));
// { status: 'sent: Dear customer, about...' }

// Parallel tasks: start them all, then await together
const double = task('double', async (n: number) => n * 2);
const fanOut = entrypoint({ name: 'fanOut' }, async (ns: number[]) => Promise.all(ns.map((n) => double(n))));
console.log(await fanOut.invoke([1, 2, 3]));   // [ 2, 4, 6 ]
```

Calling several tasks before awaiting them runs them in parallel. An entrypoint is a Pregel graph, so `invoke`, `stream`, checkpointers and `getState` work as they do for a `StateGraph`.

## Time travel: history, replay and forks

With a checkpointer, every super-step is saved. `getStateHistory` lists the checkpoints (newest first). Invoking with `null` and a checkpoint's config re-runs from that point; `updateState` on an old checkpoint creates a fork you can resume:

```typescript
import { Annotation, MemorySaver, StateGraph, START, END } from '@langchain/langgraph';

const State = Annotation.Root({
  topic: Annotation<string>,
  joke: Annotation<string>,
});

const graph = new StateGraph(State)
  .addNode('pickTopic', () => ({ topic: 'cats' }))
  .addNode('writeJoke', (state) => ({ joke: `A joke about ${state.topic}` }))
  .addEdge(START, 'pickTopic')
  .addEdge('pickTopic', 'writeJoke')
  .addEdge('writeJoke', END)
  .compile({ checkpointer: new MemorySaver() });

const config = { configurable: { thread_id: 'tt-1' } };
await graph.invoke({}, config);

// 1. Browse history (newest first)
const history = [];
for await (const snapshot of graph.getStateHistory(config)) history.push(snapshot);
for (const s of history) console.log(s.next, s.values);

// 2. Pick the checkpoint taken just before writeJoke ran
const beforeJoke = history.find((s) => s.next.includes('writeJoke'))!;

// 3. Fork: change state at that checkpoint, then resume from the fork
const forkConfig = await graph.updateState(beforeJoke.config, { topic: 'dogs' });
const forked = await graph.invoke(null, forkConfig);
console.log(forked.joke);   // A joke about dogs

// Replaying the original checkpoint without changes re-runs writeJoke from there
const replayed = await graph.invoke(null, beforeJoke.config);
console.log(replayed.joke);   // A joke about cats
```

The forked run and the original are separate branches of the same thread; `getStateHistory` shows both.

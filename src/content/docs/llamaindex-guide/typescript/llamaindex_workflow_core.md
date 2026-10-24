---
title: "Workflows engine (@llamaindex/workflow-core)"
description: "The event-driven engine under LlamaIndex.TS workflows and agents: events and handlers, joins, stateful middleware, streaming, human-in-the-loop with snapshots, retries, validation and testing."
framework: "llamaindex"
language: "typescript"
sidebar:
  label: "Workflows engine"
  order: 250
---

> Verified against `@llamaindex/workflow-core` %%version=@llamaindex/workflow-core%% (checked %%versions-checked%%). Every example was type-checked with `tsc --strict` and run with Node 22.

> **Deprecated upstream.** npm marks `@llamaindex/workflow-core` as deprecated and points to LlamaAgents (Python Workflows) instead. The final release installs and works; there will be no fixes. The Python counterpart is documented on the [Workflows (Python)](/llamaindex-guide/python/llamaindex_workflows/) page.

`@llamaindex/workflow-core` is the small engine that `@llamaindex/workflow` (and therefore `agent(...)` and `multiAgent(...)`) is built on. It has no LLM code at all: events, handlers, a context per run, an event stream, and optional middleware. Use it directly when you want event-driven orchestration in TypeScript without the rest of LlamaIndex.TS; if you already use `@llamaindex/workflow`, the core functions (`createWorkflow`, `workflowEvent`, `or`, `run`, `createStatefulMiddleware`, `zodEvent`) are re-exported from its root and the [comprehensive guide](/llamaindex-guide/typescript/llamaindex_workflows_typescript_comprehensive_guide/) applies. `@llamaindex/workflow` 1.1.25 depends on a 1.3.x release of this package.

---

## 1. Installation and entry points

```bash
npm install @llamaindex/workflow-core@%%version=@llamaindex/workflow-core%%
npm install zod        # only for zodEvent
npm install p-retry    # only for pRetryHandler
```

The package is ESM-first (a CommonJS build of the root and middleware is included). Functionality is split across subpath exports, and the root does **not** export `run`:

| Import from | Exports |
|---|---|
| `@llamaindex/workflow-core` | `createWorkflow`, `workflowEvent`, `or`, `WorkflowStream`, `extendContext`, `isWorkflowEvent`, `isWorkflowEventData`, `eventSource`, `getContext` (deprecated) |
| `@llamaindex/workflow-core/stream/run` | `run`; deprecated `runWorkflow`, `runAndCollect`, `runStream` |
| `@llamaindex/workflow-core/middleware/state` | `createStatefulMiddleware` (state, `snapshot()`, `resume()`) |
| `@llamaindex/workflow-core/middleware/validation` | `withValidation` (typed `strictHandle`) |
| `@llamaindex/workflow-core/middleware/trace-events` | `withTraceEvents`, `runOnce`, `createHandlerDecorator`, `getEventOrigins` |
| `@llamaindex/workflow-core/util/zod` | `zodEvent` (needs `zod` 3.25+ or 4) |
| `@llamaindex/workflow-core/util/p-retry` | `pRetryHandler` (needs `p-retry`) |
| `@llamaindex/workflow-core/stream/consumer`, `/filter`, `/find`, `/until` | Function forms of the stream helpers (`collect`, `nothing`, `filter`, `find`, `until`) |
| `@llamaindex/workflow-core/hono`, `/next`, `/mcp`, `/observable` | `createHonoHandler`, `createNextHandler`, `mcpTool`, `toObservable` (need `hono`, `next`, `@modelcontextprotocol/sdk` or `rxjs`) |

A `tsconfig.json` that works with the examples below:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "skipLibCheck": true,
    "lib": ["ES2022", "DOM", "DOM.AsyncIterable"],
    "types": ["node"]
  }
}
```

`skipLibCheck` is needed: the package's own `middleware/state.d.ts` fails to type-check against its root declarations. `DOM.AsyncIterable` lets you `for await` over a `WorkflowStream`.

---

## 2. Events, handlers and running

An event type is created with `workflowEvent<Payload>()`; `.with(payload)` makes an instance, `.include(x)` is a type guard. A workflow is a set of handlers registered with `handle([events], fn)`. A handler receives the context and the matched event, and may return a new event, which is sent on its behalf.

```typescript
import { createWorkflow, workflowEvent } from '@llamaindex/workflow-core';
import { run } from '@llamaindex/workflow-core/stream/run';

const startEvent = workflowEvent<{ name: string }>({ debugLabel: 'start' });
const stopEvent = workflowEvent<string>({ debugLabel: 'stop' });

const workflow = createWorkflow();

workflow.handle([startEvent], (ctx, start) => {
  return stopEvent.with(`Hello, ${start.data.name}!`);
});

const events = await run(workflow, startEvent.with({ name: 'Ada' }))
  .until(stopEvent)
  .toArray();
console.log(events.length, events.at(-1)?.data); // 2 Hello, Ada!

// The same, by hand: create a context, read its stream, send the start event
const { stream, sendEvent } = workflow.createContext();
sendEvent(startEvent.with({ name: 'Grace' }));
const stop = await stream.untilEvent(stopEvent);
console.log(stop.data); // Hello, Grace!
```

- There are no built-in start and stop events. Any event can start a run, and the caller decides which event means "finished" (`until(stopEvent)`).
- `run(workflow, event)` creates a context, sends the event and returns the context's stream. The stream includes the event you sent and **does not end on its own**: always bound it with `until(...)`, `untilEvent(...)` or `take(n)`.
- `handle()` returns `void`; register each handler with a separate call.
- `uniqueId` (optional in `workflowEvent({ uniqueId })`) identifies an event type across a network boundary; see `toResponse()` below.

### Branching, loops and `or()`

A handler returns whichever event fits; returning an earlier event type loops. `or(a, b)` accepts either event in one handler:

```typescript
import { createWorkflow, workflowEvent, or } from '@llamaindex/workflow-core';
import { run } from '@llamaindex/workflow-core/stream/run';

const startEvent = workflowEvent<{ topic: string }>();
const draftEvent = workflowEvent<{ text: string; attempt: number }>();
const reviewEvent = workflowEvent<{ text: string; attempt: number }>();
const rejectedEvent = workflowEvent<{ reason: string }>();
const stopEvent = workflowEvent<{ text: string } | { error: string }>();

const workflow = createWorkflow();

workflow.handle([startEvent], (ctx, ev) => {
  if (ev.data.topic.trim() === '') return rejectedEvent.with({ reason: 'empty topic' });
  return draftEvent.with({ text: ev.data.topic, attempt: 1 });
});

workflow.handle([draftEvent], async (ctx, ev) => {
  return reviewEvent.with({ text: `${ev.data.text} (draft ${ev.data.attempt})`, attempt: ev.data.attempt });
});

workflow.handle([reviewEvent], (ctx, ev) => {
  if (ev.data.attempt < 3) return draftEvent.with({ text: ev.data.text, attempt: ev.data.attempt + 1 });
  return stopEvent.with({ text: ev.data.text });
});

workflow.handle([rejectedEvent], (ctx, ev) => stopEvent.with({ error: ev.data.reason }));

// A logger that sees both drafts and reviews
workflow.handle([or(draftEvent, reviewEvent)], () => {
  console.log('draft or review');
});

const ok = await run(workflow, startEvent.with({ topic: 'Workflows' })).until(stopEvent).toArray();
console.log(ok.at(-1)?.data); // { text: 'Workflows (draft 1) (draft 2) (draft 3)' }
```

Every handler whose accepted events match receives the event — in this example the logger and the `draftEvent` / `reviewEvent` handlers all run. There is no graph validation: an event nobody handles is simply not handled.

### Joins: waiting for several events

A handler that lists several events fires once one of each is waiting, and receives them in the order listed:

```typescript
const askEvent = workflowEvent<{ question: string }>();
const queryEvent = workflowEvent<string>();
const chunksEvent = workflowEvent<string[]>();
const answerEvent = workflowEvent<string>();

const rag = createWorkflow();
rag.handle([askEvent], async (ctx, ev) => {
  ctx.sendEvent(queryEvent.with(ev.data.question));
  await new Promise((r) => setTimeout(r, 20));      // stands in for retrieval
  ctx.sendEvent(chunksEvent.with(['c1', 'c2']));
});
rag.handle([queryEvent, chunksEvent], (ctx, query, chunks) => {
  return answerEvent.with(`${query.data}: ${chunks.data.join(', ')}`);
});

const answer = await run(rag, askEvent.with({ question: 'Q' })).untilEvent(answerEvent);
console.log(answer.data); // Q: c1, c2
```

`ctx.sendEvent(...events)` emits events without returning, which is also how a handler fans out.

---

## 3. State with `createStatefulMiddleware`

The state middleware adds `ctx.state` to every handler's context. The optional initialiser builds the state; when it takes an argument, `createContext(input)` requires one.

```typescript
import { createWorkflow, workflowEvent } from '@llamaindex/workflow-core';
import { createStatefulMiddleware } from '@llamaindex/workflow-core/middleware/state';

const startEvent = workflowEvent<{ items: string[] }>();
const jobEvent = workflowEvent<string>();
const doneEvent = workflowEvent<{ item: string; length: number }>();
const resultEvent = workflowEvent<Record<string, number>>();

const { withState } = createStatefulMiddleware(() => ({
  expected: 0,
  results: {} as Record<string, number>,
}));
const workflow = withState(createWorkflow());

workflow.handle([startEvent], (ctx, ev) => {
  ctx.state.expected = ev.data.items.length;
  for (const item of ev.data.items) ctx.sendEvent(jobEvent.with(item));   // fan out
});

workflow.handle([jobEvent], async (ctx, ev) => {
  await new Promise((r) => setTimeout(r, 10 * ev.data.length));          // runs concurrently
  return doneEvent.with({ item: ev.data, length: ev.data.length });
});

workflow.handle([doneEvent], (ctx, ev) => {                                 // fan in
  ctx.state.results[ev.data.item] = ev.data.length;
  if (Object.keys(ctx.state.results).length === ctx.state.expected) {
    return resultEvent.with(ctx.state.results);
  }
});

const { stream, sendEvent, state } = workflow.createContext();
sendEvent(startEvent.with({ items: ['ccc', 'a', 'bb'] }));
const result = await stream.untilEvent(resultEvent);
console.log(result.data, state.expected); // { a: 1, bb: 2, ccc: 3 } 3
```

State is a plain mutable object shared by all handlers of one context; each `createContext()` gets its own. Keep it JSON-serialisable if you want snapshots (section 5).

---

## 4. Streaming

Everything sent in a context appears on its stream. `WorkflowStream` is a `ReadableStream` with workflow helpers: `until`, `untilEvent`, `filter`, `map`, `take`, `forEach`, `toArray`, `on`, `toResponse`.

```typescript
import { createWorkflow, workflowEvent } from '@llamaindex/workflow-core';
import { run } from '@llamaindex/workflow-core/stream/run';

const startEvent = workflowEvent<{ prompt: string }>();
const tokenEvent = workflowEvent<{ delta: string }>();
const progressEvent = workflowEvent<string>();
const stopEvent = workflowEvent<{ text: string }>();

const workflow = createWorkflow();
workflow.handle([startEvent], async (ctx, ev) => {
  ctx.sendEvent(progressEvent.with('generating'));
  let text = '';
  for (const word of ev.data.prompt.split(' ')) {      // stands in for LLM tokens
    await new Promise((r) => setTimeout(r, 5));
    text += word + ' ';
    ctx.sendEvent(tokenEvent.with({ delta: word + ' ' }));
  }
  return stopEvent.with({ text: text.trim() });
});

// Iterate until the stop event
for await (const ev of run(workflow, startEvent.with({ prompt: 'events all the way down' })).until(stopEvent)) {
  if (tokenEvent.include(ev)) process.stdout.write(ev.data.delta);
  else if (progressEvent.include(ev)) console.log(`[${ev.data}]`);
  else if (stopEvent.include(ev)) console.log('\nfinal:', ev.data.text);
}

// Callbacks with on(), then wait for the result
const { stream, sendEvent } = workflow.createContext();
const off = stream.on(tokenEvent, (ev) => process.stdout.write(ev.data.delta));
sendEvent(startEvent.with({ prompt: 'shout this' }));
const final = await stream.untilEvent(stopEvent);
off();

// Just the tokens
const tokens = await run(workflow, startEvent.with({ prompt: 'a b c' }))
  .until(stopEvent)
  .filter(tokenEvent)
  .map((ev) => ev.data.delta)
  .toArray();                                          // [ 'a ', 'b ', 'c ' ]

// Newline-delimited JSON for an HTTP response
const response = run(workflow, startEvent.with({ prompt: 'x y' })).until(stopEvent).toResponse();
```

- **Read `ctx.stream` before sending.** The stream is created when first accessed and only sees events sent after that. `run()` and the destructuring `const { stream, sendEvent } = workflow.createContext()` both do this in the right order; `ctx.sendEvent(...)` followed by `ctx.stream.untilEvent(...)` can miss events that handlers emit synchronously.
- `untilEvent()` and `toArray()` consume the stream. To react to several events in turn (such as a request and then the final result), use one `for await` loop rather than consecutive `untilEvent()` calls.
- `toResponse()` writes one JSON object per event: `{"data": ..., "uniqueId": ...}`. `WorkflowStream.fromResponse(response, { startEvent, tokenEvent, ... })` turns it back into events on the client, matching on `uniqueId` — give events an explicit `uniqueId` when client and server are built separately.
- `createHonoHandler(workflow, getStart, stopEvent)` and `createNextHandler(workflow, getStart, stopEvent)` wrap a workflow as a Hono route or a Next.js route handler; `mcpTool(workflow, startEvent, stopEvent)` exposes one as an MCP tool. They were not exercised for this page.

---

## 5. Human-in-the-loop

There are no special HITL events: define a request event that no handler consumes, so it only reaches the caller, and a response event that the caller sends back.

```typescript
import { createWorkflow, workflowEvent } from '@llamaindex/workflow-core';
import { createStatefulMiddleware, type SnapshotData } from '@llamaindex/workflow-core/middleware/state';

const startEvent = workflowEvent<{ title: string }>({ debugLabel: 'start' });
const approvalRequest = workflowEvent<{ question: string }>({ debugLabel: 'approvalRequest' });
const approvalResponse = workflowEvent<{ approved: boolean }>({ debugLabel: 'approvalResponse' });
const stopEvent = workflowEvent<string>({ debugLabel: 'stop' });

const { withState } = createStatefulMiddleware((input: { owner: string }) => ({
  title: '',
  owner: input.owner,
}));
const workflow = withState(createWorkflow());

workflow.handle([startEvent], (ctx, ev) => {
  ctx.state.title = ev.data.title;
  return approvalRequest.with({ question: `Publish '${ev.data.title}'?` });
});

workflow.handle([approvalResponse], (ctx, ev) => {
  const verdict = ev.data.approved ? 'published' : 'discarded';
  return stopEvent.with(`${ctx.state.title} (${ctx.state.owner}): ${verdict}`);
});
```

Within one process, answer on the same context:

```typescript
const { stream, sendEvent } = workflow.createContext({ owner: 'ada' });
sendEvent(startEvent.with({ title: 'Release notes' }));
for await (const ev of stream.until(stopEvent)) {
  if (approvalRequest.include(ev)) {
    console.log(ev.data.question);
    sendEvent(approvalResponse.with({ approved: true }));   // e.g. after a UI prompt
  } else if (stopEvent.include(ev)) {
    console.log(ev.data);                                     // Release notes (ada): published
  }
}
```

When the answer may take longer than the process lives, take a snapshot once the request appears and resume from it later:

```typescript
// Process 1: run until the request, then snapshot
const ctx = workflow.createContext({ owner: 'grace' });
const firstStream = ctx.stream;                  // subscribe before sending anything
ctx.sendEvent(startEvent.with({ title: 'Roadmap' }));
await firstStream.untilEvent(approvalRequest);
const saved = JSON.stringify(await ctx.snapshot());   // store this in a database

// Process 2: resume and deliver the answer
const resumed = workflow.resume(JSON.parse(saved) as SnapshotData);
console.log(resumed.state);                      // { title: 'Roadmap', owner: 'grace' }
const resumedStream = resumed.stream;
resumed.sendEvent(approvalResponse.with({ approved: false }));
const done = await resumedStream.untilEvent(stopEvent);
console.log(done.data);                          // Roadmap (grace): discarded
```

How snapshots behave:

- `snapshot()` waits for running async handlers, then returns `{ queue, unrecoverableQueue, version, state }`. `state` is `JSON.stringify(ctx.state)`, so it must be JSON-safe.
- Taking a snapshot **locks the context**: handlers stop running in it and events sent afterwards only produce a warning. Snapshot only when you are about to abandon that context, and only once per context.
- `queue` holds pending events that some handler accepts; they are re-sent on `resume()`. Events no handler accepts — the approval request above — go to `unrecoverableQueue` and are not replayed, so store whatever you need to show the user yourself.
- `version` is a hash of the registered handlers. `resume()` does not check it; compare it yourself if you deploy changes while runs are paused.
- `resume()` must be called on a workflow built with the same `withState(...)` and the same handlers registered in the same order.

---

## 6. Errors, retries and timeouts

The engine has no retry policy, timeout or failure event of its own, and errors behave differently depending on how a handler fails:

- A **synchronous** throw aborts the context: `ctx.signal` is aborted with the error as `signal.reason`. The stream does not end or reject.
- A rejected **async** handler is not caught by the engine. In Node it surfaces as an unhandled rejection, which terminates the process by default.

So catch errors inside handlers and turn them into events:

```typescript
const startEvent = workflowEvent<string>();
const stopEvent = workflowEvent<string>();
const errorEvent = workflowEvent<{ message: string }>();

const workflow = createWorkflow();
workflow.handle([startEvent], async (ctx, ev) => {
  try {
    return stopEvent.with(await callUpstream(ev.data));
  } catch (err) {
    return errorEvent.with({ message: (err as Error).message });
  }
});

const last = (await run(workflow, startEvent.with('x'))
  .until((ev) => stopEvent.include(ev) || errorEvent.include(ev))
  .toArray()).at(-1);
if (errorEvent.include(last)) console.error(last.data.message);
```

### Retries with `pRetryHandler`

`pRetryHandler(handler, options)` wraps a handler in [`p-retry`](https://www.npmjs.com/package/p-retry); `options` are p-retry's (`retries`, `minTimeout`, `factor`, ...):

```typescript
import { pRetryHandler } from '@llamaindex/workflow-core/util/p-retry';

const fetchEvent = workflowEvent<{ url: string }>();
const fetchedEvent = workflowEvent<{ body: string }>();

workflow.handle(
  [fetchEvent],
  pRetryHandler(
    async (ctx, ev) => fetchedEvent.with({ body: await flakyFetch(ev.data.url) }),
    { retries: 4, minTimeout: 500, factor: 2 },
  ),
);
```

When the retries run out the wrapped handler rejects, with the consequence described above; catch inside the handler if you want an error event instead.

### Timeouts

Race the result against a timer, and check `ctx.signal` in long handlers:

```typescript
async function runWithTimeout(ms: number) {
  const { stream, sendEvent } = workflow.createContext();
  sendEvent(startEvent.with('z'));
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`timed out after ${ms} ms`)), ms);
  });
  try {
    return await Promise.race([stream.untilEvent(stopEvent), timeout]);
  } finally {
    clearTimeout(timer);
  }
}
```

The handlers keep running after the timeout fires; the engine has no way to cancel them other than the abort signal they choose to check.

---

## 7. Validated events and typed outputs

`zodEvent(schema)` creates an event type that validates its payload when `.with()` is called (a Zod error is thrown on bad input). `withValidation(workflow, rules)` adds `strictHandle`, whose `sendEvent` only accepts the outputs declared for the handler's inputs:

```typescript
import { createWorkflow, workflowEvent } from '@llamaindex/workflow-core';
import { withValidation } from '@llamaindex/workflow-core/middleware/validation';
import { zodEvent } from '@llamaindex/workflow-core/util/zod';
import { z } from 'zod';

const signupEvent = zodEvent(z.object({ email: z.string().email() }), { debugLabel: 'signup' });
const acceptedEvent = workflowEvent<string>();
const rejectedEvent = workflowEvent<string>();

const workflow = withValidation(createWorkflow(), [
  [[signupEvent], [acceptedEvent]],
  [[signupEvent], [rejectedEvent]],
]);

workflow.strictHandle([signupEvent], (sendEvent, ctx, ev) => {
  if (ev.data.email.endsWith('@example.com')) sendEvent(acceptedEvent.with(ev.data.email));
  else sendEvent(rejectedEvent.with(ev.data.email));
  // sendEvent(signupEvent.with(...)) does not compile: not a declared output
});
```

Each rule is `[inputs, outputs]`, and `outputs` is the exact tuple of events sent together in one `sendEvent` call. Alternatives therefore need one rule each, as above; a single rule `[[signupEvent], [acceptedEvent, rejectedEvent]]` means "send both at once" and logs `Invalid input detected` at run time when only one is sent (the event is still delivered).

---

## 8. Nested workflows and testing

A handler can run another workflow and await its result:

```typescript
const summariseStart = workflowEvent<string>();
const summariseStop = workflowEvent<string>();
const summariser = createWorkflow();
summariser.handle([summariseStart], (ctx, ev) => summariseStop.with(ev.data.slice(0, 200)));

const docEvent = workflowEvent<{ text: string }>();
const summaryEvent = workflowEvent<string>();
const pipeline = createWorkflow();
pipeline.handle([docEvent], async (ctx, ev) => {
  const summary = await run(summariser, summariseStart.with(ev.data.text)).untilEvent(summariseStop);
  return summaryEvent.with(`summary: ${summary.data}`);
});
```

Workflows are plain objects, so tests need no harness. Build the workflow in a function so each test gets fresh handlers, run it and assert on the collected events:

```typescript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createWorkflow, workflowEvent } from '@llamaindex/workflow-core';
import { run } from '@llamaindex/workflow-core/stream/run';

const startEvent = workflowEvent<number>();
const progressEvent = workflowEvent<number>();
const stopEvent = workflowEvent<number>();

function buildWorkflow() {
  const workflow = createWorkflow();
  workflow.handle([startEvent], (ctx, ev) => {
    for (let i = 1; i <= ev.data; i++) ctx.sendEvent(progressEvent.with(i));
    return stopEvent.with(ev.data * 2);
  });
  return workflow;
}

test('doubles the input and reports progress', async () => {
  const events = await run(buildWorkflow(), startEvent.with(3)).until(stopEvent).toArray();
  assert.equal(events.filter((e) => progressEvent.include(e)).length, 3);
  const last = events.at(-1);
  assert.ok(stopEvent.include(last));
  assert.equal(last.data, 6);
});
```

Run it with `npx tsx --test`, or any test runner that handles ESM.

---

## 9. Compared with Python Workflows

| | `@llamaindex/workflow-core` | Python `llama-index-workflows` |
|---|---|---|
| Definition | Functional: `createWorkflow()` + `handle([events], fn)` | Class with `@step` methods |
| Start / stop | Any event; caller chooses the terminal event | `StartEvent` / `StopEvent` (or typed subclasses) |
| Graph validation | None | Checked before the first run |
| State | `createStatefulMiddleware`, plain object | `ctx.store`, typed Pydantic state |
| Human-in-the-loop | Your own request/response events; `snapshot()` / `resume()` | `InputRequiredEvent` / `HumanResponseEvent`, `wait_for_event`, `Context.to_dict()` |
| Retries, timeouts | `pRetryHandler`, your own timer | `retry_policy`, `@catch_error`, `timeout=` |
| Serving | Hono / Next.js helpers, `toResponse()` | `WorkflowServer`, `llamactl` |
| Maintenance | Deprecated | Maintained |

---

## Common pitfalls

| Symptom | Cause and fix |
|---|---|
| `run` is not exported from `@llamaindex/workflow-core` | Import it from `@llamaindex/workflow-core/stream/run`. |
| `await ...toArray()` never resolves | The stream does not end by itself. Add `.until(stopEvent)` or `.take(n)`. |
| `untilEvent` misses an event, or the script exits with "unsettled top-level await" | The stream was read after the event was sent. Access `ctx.stream` (or destructure it) before `sendEvent`. |
| `Stream ended without matching event` on a second `untilEvent` | The first call consumed the stream. Use one `for await` loop. |
| Process exits on a handler error | An async handler rejected. Catch inside the handler and return an error event. |
| Events sent after `snapshot()` do nothing | Snapshots lock the context. Resume from the snapshot instead. |
| `tsc` errors inside `node_modules/@llamaindex/workflow-core` | Set `"skipLibCheck": true`. |
| `context.send(...)` / `context.waitFor(...)` from the package's doc comments do not exist | Use `sendEvent(...)` and `stream.untilEvent(...)`. |

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | @llamaindex/workflow-core 1.3.4 | New page for the engine package: subpath exports, joins, state middleware, streaming, human-in-the-loop with `snapshot()` / `resume()`, error behaviour, `pRetryHandler`, `withValidation` / `zodEvent`, testing. Examples type-checked with `tsc --strict` and run with Node 22. |

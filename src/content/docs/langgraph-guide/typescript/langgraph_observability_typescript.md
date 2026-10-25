---
title: "LangGraph Observability and Monitoring (TypeScript)"
description: "Tracing, callbacks and stream-based monitoring for LangGraph TypeScript, with LangSmith and OpenTelemetry."
framework: "langgraph"
language: "typescript"
sidebar:
  label: "Observability"
  order: 35
---


> Verified against `@langchain/langgraph` %%version=@langchain/langgraph%% (checked %%versions-checked%%). Upstream: https://github.com/langchain-ai/langgraphjs

There are three layers to choose from. LangSmith tracing needs no code. Callbacks give you hooks on every run, node, model and tool call inside the process. Stream modes such as `"tasks"` let a caller watch progress without touching the graph. OpenTelemetry fits in by wrapping node functions.

## LangSmith (no code changes)

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (optionally `LANGSMITH_PROJECT`). Every graph run, node, model call and tool call is then traced automatically.

To make traces easier to find, pass `runName`, `tags` and `metadata` in the run config. They are attached to the root run and inherited by child runs, and callbacks receive them too (see the next example).

Callbacks, including the LangSmith tracer, run in the background by default. In short-lived environments (serverless functions, scripts that exit straight away) either set `LANGCHAIN_CALLBACKS_BACKGROUND=false` or call `await awaitAllCallbacks()` (from `@langchain/core/callbacks/promises`) before the process ends, otherwise the last traces can be lost.

There is no per-node trace-redaction option in the JavaScript packages yet (Python has `TracePolicy`; see [Trace policy](/langgraph-guide/python/reference-trace-policy/)). To keep sensitive values out of traces, leave them out of state or pass them through `context` instead.

## Callbacks

A callback handler sees every runnable in the run: the graph itself, each node, and any model or tool call inside a node. Pass handlers in `callbacks` on `invoke`/`stream`, or bind them with `graph.withConfig({ callbacks: [...] })`. The `runName` argument tells you which node a chain event belongs to:

```ts
import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { ChainValues } from "@langchain/core/utils/types";
import type { Serialized } from "@langchain/core/load/serializable";
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import * as z from "zod";

class TimingHandler extends BaseCallbackHandler {
  name = "timing";
  private starts = new Map<string, { name: string; t: number }>();

  async handleChainStart(
    _chain: Serialized, _inputs: ChainValues, runId: string,
    _parentRunId?: string, _tags?: string[], _metadata?: Record<string, unknown>,
    _runType?: string, runName?: string,
  ) {
    this.starts.set(runId, { name: runName ?? "chain", t: performance.now() });
  }

  async handleChainEnd(_outputs: ChainValues, runId: string) {
    const s = this.starts.get(runId);
    if (s) console.log(`${s.name}: ${(performance.now() - s.t).toFixed(1)} ms`);
    this.starts.delete(runId);
  }

  async handleChainError(err: Error, runId: string) {
    const s = this.starts.get(runId);
    console.log(`${s?.name ?? runId} failed: ${err.message}`);
    this.starts.delete(runId);
  }
}

const State = new StateSchema({ n: z.number() });
const graph = new StateGraph(State)
  .addNode("double", async (s) => ({ n: s.n * 2 }))
  .addNode("inc", async (s) => ({ n: s.n + 1 }))
  .addEdge(START, "double")
  .addEdge("double", "inc")
  .addEdge("inc", END)
  .compile();

const result = await graph.invoke(
  { n: 3 },
  {
    callbacks: [new TimingHandler()],
    runName: "double-then-inc",
    tags: ["demo"],
    metadata: { userId: "u-42" },
  },
);
console.log(result);
```

Output (timings vary):

```text
__start__: 13.4 ms
double: 3.5 ms
inc: 1.7 ms
double-then-inc: 57.8 ms
{ n: 7 }
```

The `__start__` entry is the graph's internal input step; filter it out if you only want your own nodes.

### Interrupt and resume events

`GraphCallbackHandler` (exported from `@langchain/langgraph` since 1.4.17) adds two graph-level hooks on top of `BaseCallbackHandler`: `handleInterrupt` fires when the root graph pauses, and `handleResume` fires before a run continues from a checkpoint. Unlike other callbacks these are always awaited. Set `raiseError = true` on the handler if an error inside it should fail the run.

```ts
import {
  GraphCallbackHandler,
  type GraphInterruptEvent,
  type GraphResumeEvent,
  StateGraph, StateSchema, START, END, MemorySaver, Command, interrupt,
} from "@langchain/langgraph";
import * as z from "zod";

class InterruptLogger extends GraphCallbackHandler {
  handleInterrupt(event: GraphInterruptEvent) {
    console.log("paused", event.status, event.checkpointId,
      event.interrupts.map((i) => i.value));
  }
  handleResume(event: GraphResumeEvent) {
    console.log("resumed", event.status, event.checkpointId);
  }
}

const State = new StateSchema({ draft: z.string(), approved: z.boolean().default(false) });

const graph = new StateGraph(State)
  .addNode("review", async (s) => {
    const ok = interrupt({ question: "Approve?", draft: s.draft }) as boolean;
    return { approved: ok };
  })
  .addEdge(START, "review")
  .addEdge("review", END)
  .compile({ checkpointer: new MemorySaver() });

const config = {
  configurable: { thread_id: "t1" },
  callbacks: [new InterruptLogger()],
};

await graph.invoke({ draft: "Hello" }, config);
console.log(await graph.invoke(new Command({ resume: true }), config));
```

Output:

```text
paused pending 1f1d0087-... [ { question: 'Approve?', draft: 'Hello' } ]
resumed pending 1f1d0087-...
{ draft: 'Hello', approved: true }
```

This is a convenient place to notify a reviewer or record how long runs sit waiting for human input. The event carries `runId`, `status`, `checkpointId`, `checkpointNs` and, for interrupts, the `interrupts` list.

## Watching progress with stream modes

When the caller, rather than the graph, should do the monitoring, stream with `streamMode: "tasks"`. Each node produces a start event (with `input` and `triggers`) and a finish event (with `result`), both carrying the task `id` and `name`:

```ts
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import * as z from "zod";

const State = new StateSchema({ n: z.number() });
const graph = new StateGraph(State)
  .addNode("double", async (s) => ({ n: s.n * 2 }))
  .addNode("inc", async (s) => ({ n: s.n + 1 }))
  .addEdge(START, "double")
  .addEdge("double", "inc")
  .addEdge("inc", END)
  .compile();

for await (const chunk of await graph.stream({ n: 1 }, { streamMode: "tasks" })) {
  if ("result" in chunk) {
    console.log("finished", chunk.name, chunk.result);
  } else {
    console.log("started", chunk.name, chunk.input);
  }
}
```

Output:

```text
started double { n: 1 }
finished double { n: 2 }
started inc { n: 2 }
finished inc { n: 3 }
```

If a node throws, its finish event still arrives (with an empty `result`) and the stream then rejects with the error, so wrap the loop in `try`/`catch`. `streamMode: "debug"` wraps the same task events in `{ type: "task" | "task_result", ... }` objects and, when a checkpointer is attached, adds `type: "checkpoint"` events; `"checkpoints"` emits only the checkpoint events. Modes can be combined, for example `streamMode: ["updates", "tasks"]`, in which case each chunk is a `[mode, payload]` tuple.

## OpenTelemetry

LangGraph does not create OpenTelemetry spans itself. The simplest approach is to wrap node functions. `startActiveSpan` makes the span current, so spans created by libraries the node calls (HTTP clients, database drivers) nest under it:

```ts
import { trace, SpanStatusCode } from "@opentelemetry/api";
import { StateGraph, StateSchema, START, END } from "@langchain/langgraph";
import * as z from "zod";

const tracer = trace.getTracer("langgraph-ts");

function traced<S, R>(name: string, fn: (state: S) => Promise<R>) {
  return async (state: S): Promise<R> =>
    tracer.startActiveSpan(`node:${name}`, async (span) => {
      try {
        return await fn(state);
      } catch (e) {
        const err = e as Error;
        span.recordException(err);
        span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
        throw e;
      } finally {
        span.end();
      }
    });
}

const State = new StateSchema({
  question: z.string(),
  docs: z.array(z.string()).default(() => []),
});

const retrieve = async (state: typeof State.State) => ({
  docs: [`doc about ${state.question}`],
});

const graph = new StateGraph(State)
  .addNode("retrieve", traced("retrieve", retrieve))
  .addEdge(START, "retrieve")
  .addEdge("retrieve", END)
  .compile();

console.log(await graph.invoke({ question: "tides" }));
```

`@opentelemetry/api` is a no-op until an SDK is registered: set one up at process start (for example with `@opentelemetry/sdk-node` and an OTLP exporter) to send the spans somewhere.

## Metrics

Useful numbers to record, and where to get them:

- **Per-node duration**: from `handleChainStart`/`handleChainEnd` in a callback handler (as above) or from the span wrapper.
- **Errors and retries**: `handleChainError` fires for each failed attempt, including attempts that a `retryPolicy` later retries successfully, so count errors per node name there (pair the `runId` with the name recorded in `handleChainStart`).
- **Timeouts**: a node with a `timeout` that runs over fails with `NodeTimeoutError`; check with `isNodeTimeoutError(err)` in `handleChainError` or around `invoke`.
- **Time waiting for humans**: the gap between `handleInterrupt` and the matching `handleResume`.
- **Token usage**: `handleLLMEnd` receives the model output; with LangSmith enabled, token counts and costs are already recorded per run.

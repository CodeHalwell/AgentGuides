---
title: "LangGraph Observability and Monitoring (TypeScript)"
description: "Tracing and metrics for LangGraph TypeScript with OpenTelemetry."
framework: langgraph
language: typescript
---

# LangGraph Observability and Monitoring (TypeScript)


> Verified against `@langchain/langgraph` %%version=@langchain/langgraph%% (checked %%versions-checked%%). Upstream: https://github.com/langchain-ai/langgraphjs

## LangSmith (no code changes)

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (optionally `LANGSMITH_PROJECT`). Every graph run, node, model call and tool call is then traced automatically. Per-node tracing can be tuned with the `tracePolicy` node option (added in 1.4.15).

## Tracing

```ts
import { context, trace, SpanStatusCode } from "@opentelemetry/api";

const tracer = trace.getTracer("langgraph-ts");

function traced<T extends any[], R>(name: string, fn: (...a: T) => Promise<R>) {
  return async (...a: T) => {
    const span = tracer.startSpan(name);
    try {
      const out = await fn(...a);
      span.end();
      return out;
    } catch (e: any) {
      span.recordException(e);
      span.setStatus({ code: SpanStatusCode.ERROR, message: e?.message });
      span.end();
      throw e;
    }
  };
}
```

Wrap a node function with it, e.g. `graph.addNode("retrieve", traced("retrieve", retrieveNode))`.

## Callbacks

For graph-wide hooks without wrapping every node, pass a callback handler in the run config. `GraphCallbackHandler` (exported from `@langchain/langgraph` since 1.4.17) offers typed handlers for graph lifecycle events; the generic `BaseCallbackHandler` from `@langchain/core/callbacks/base` works too:

```ts
import { BaseCallbackHandler } from "@langchain/core/callbacks/base";

class TimingHandler extends BaseCallbackHandler {
  name = "timing";
  private starts = new Map<string, number>();
  async handleChainStart(_chain: unknown, _inputs: unknown, runId: string) {
    this.starts.set(runId, Date.now());
  }
  async handleChainEnd(_outputs: unknown, runId: string) {
    const t = this.starts.get(runId);
    if (t) console.log(`run ${runId} took ${Date.now() - t} ms`);
  }
}

await graph.invoke(input, { callbacks: [new TimingHandler()] });
```

## Metrics

- Use OTLP exporters; record per-node durations, retries (`retryPolicy`) and timeouts (`NodeTimeoutError`).

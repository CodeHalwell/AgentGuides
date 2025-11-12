---
layout: default
title: "LangGraph Observability and Monitoring (TypeScript)"
description: "Tracing and metrics for LangGraph TypeScript with OpenTelemetry."
---

# LangGraph Observability and Monitoring (TypeScript)


Latest: 1.0.2
Upstream: https://github.com/langchain-ai/langgraph

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

## Metrics
- Use OTLP exporters; record per-node durations and attempts
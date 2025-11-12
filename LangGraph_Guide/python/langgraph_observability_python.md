---
layout: default
title: "LangGraph Observability and Monitoring (Python)"
description: "Tracing, metrics, and logging patterns for LangGraph Python."
---

# LangGraph Observability and Monitoring (Python)

Last verified: 2025-11 • Source: langchain-ai/langgraph

## Tracing
- Wrap node execution with OpenTelemetry spans
- Attributes: graph, node, attempt, duration, tokens, errors

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

def traced_node(fn):
    def wrapper(state):
        with tracer.start_as_current_span(fn.__name__) as span:
            span.set_attribute("graph.node", fn.__name__)
            try:
                out = fn(state)
                return out
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise
    return wrapper
```

## Metrics
- Emit per-node counters and histograms (attempts, latency)
- Export via Prometheus or OTLP

## Logs
- Structured logs with run/trace IDs and node names


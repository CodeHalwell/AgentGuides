---
layout: default
title: "LlamaIndex Observability & Monitoring (Python)"
description: "Tracing, metrics, and logging patterns for LlamaIndex."
---

# LlamaIndex Observability & Monitoring (Python)


Latest: 0.14.8
Upstream: https://github.com/run-llama/llama_index/releases | https://pypi.org/project/llama-index/

## Tracing
- Wrap query engine invocations with OTEL spans; include index name, retriever type, tokens

## Metrics
- Histogram for latency; counters for calls, failures; gauge for cache hit rate

## Logs
- Structured logs with request ID and source nodes used
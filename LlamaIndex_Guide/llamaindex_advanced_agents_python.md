---
layout: default
title: "LlamaIndex Advanced Agents (Python)"
description: "Multi-agent orchestration with LlamaIndex tools, observability, and recovery patterns."
---

# LlamaIndex Advanced Agents (Python)

Last verified: 2025-11 • Source: https://github.com/run-llama/llama_index

## Patterns
- Tool-augmented agents with query engines and retrievers
- Specialized agents per domain (RAG, web, code) with a router
- HITL escalation on low confidence

## Router Skeleton

```python
def route(task: str) -> str:
    if "docs" in task.lower():
        return "rag_agent"
    if "web" in task.lower():
        return "web_agent"
    return "general_agent"
```

## Observability
- Track query latency, token usage, retriever stats; export OTEL

## Recovery
- Retry transient retriever/LLM errors; dead-letter on repeated failure


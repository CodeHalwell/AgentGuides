---
layout: default
title: "LangGraph Performance Optimization (Python)"
description: "Techniques for batching, caching, and parallelism in LangGraph Python."
---

# LangGraph Performance Optimization (Python)


Latest: 1.0.3
Upstream: https://github.com/langchain-ai/langgraph

## Techniques
- Cache intermediate results by state keys to avoid re-computation
- Batch LLM calls when possible; stream responses
- Parallel branches with care; manage rate limits and retries

## Example: Cached Node

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def embed(text: str) -> list[float]:
    ...

def node(state: dict) -> dict:
    vec = embed(state["text"])  # reuses cache
    return {**state, "vec": vec}
```
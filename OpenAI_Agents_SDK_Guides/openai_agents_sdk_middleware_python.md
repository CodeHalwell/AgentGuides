---
layout: default
title: "OpenAI Agents SDK Middleware & Guardrails (Python)"
description: "Add middleware, guardrails, and tracing to Python Agents SDK apps."
---

# OpenAI Agents SDK Middleware & Guardrails (Python)


Latest: 6.8.1
Upstream: https://github.com/openai/openai-python | https://platform.openai.com/docs

## Middleware Chain

```python
from typing import Awaitable, Callable
Next = Callable[[str], Awaitable[str]]
Middleware = Callable[[str, Next], Awaitable[str]]

async def policy_mw(inp: str, next_call: Next) -> str:
    if "ssn" in inp.lower():
        raise ValueError("Policy violation")
    return await next_call(inp)
```

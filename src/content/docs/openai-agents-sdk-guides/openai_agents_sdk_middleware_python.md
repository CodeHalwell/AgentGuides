---
title: "OpenAI Agents SDK Middleware & Guardrails (Python)"
description: "Add middleware, guardrails, and tracing to Python Agents SDK apps."
framework: openai-agents-sdk
---

# OpenAI Agents SDK Middleware & Guardrails (Python)


Latest: openai-agents 0.17.4 | Updated: May 27, 2026
Upstream: https://github.com/openai/openai-agents-python | https://platform.openai.com/docs

> The middleware and guardrail shapes below are framework-agnostic — they wrap any call, whether you dispatch through `openai-agents` or the base `openai` SDK.

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

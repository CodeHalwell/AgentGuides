---
layout: default
title: "Semantic Kernel Middleware (Python)"
description: "Implement request/response middleware, guardrails, and policy enforcement in SK Python."
---

# Semantic Kernel Middleware (Python)

Last verified: 2025-11 • Source: SK Python samples and docs

## Goals
- Implement middleware-like hooks around function/agent execution
- Add guardrails (content policy, PII) and safety filters
- Capture metrics, traces, and redaction consistently

## Execution Wrapper Pattern

```python
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

Middleware = Callable[[str, Callable[[str], Awaitable[str]]], Awaitable[str]]

async def policy_guardrail(input_text: str) -> None:
    banned = ["ssn", "credit card"]
    if any(b in input_text.lower() for b in banned):
        raise ValueError("Policy violation: sensitive data detected")

async def redact(text: str) -> str:
    return text.replace("123-45-6789", "***-**-****")

async def middleware_stack(middlewares: list[Middleware], handler: Callable[[str], Awaitable[str]]):
    async def call_chain(text: str) -> str:
        next_handler = handler
        for mw in reversed(middlewares):
            current = next_handler
            next_handler = (lambda t, mw=mw, current=current: mw(t, current))
        return await next_handler(text)
    return call_chain

async def llm_handler(text: str) -> str:
    # call SK function/agent here
    return f"OK: {text}"

async def guardrail_mw(text: str, next_call):
    await policy_guardrail(text)
    return await next_call(text)

async def redact_mw(text: str, next_call):
    out = await next_call(text)
    return await redact(out)

async def run():
    stack = await middleware_stack([guardrail_mw, redact_mw], llm_handler)
    print(await stack("draft contract for ssn 123-45-6789"))
```

Integrate this wrapper into your SK orchestrations before invoking semantic/native functions.

## Function-Level Middleware in SK

Wrap `kernel.invoke_*` calls with the stack to get consistent enforcement:

```python
async def invoke_with_mw(kernel, func, input_text: str) -> str:
    async def handler(text: str):
        return await kernel.invoke_async(func, input_text=text)
    stack = await middleware_stack([guardrail_mw, redact_mw], handler)
    return await stack(input_text)
```

## Telemetry

- Add timing, token usage, and tool IO tags; export via OpenTelemetry
- Use correlation IDs per request and include in logs/traces

## Testing

- Unit tests for guardrail decisions and redaction
- Integration tests using a mocked SK service to verify wrapper order

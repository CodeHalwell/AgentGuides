---
title: "Microsoft Agent Framework (Python) — Middleware"
description: "Agent, chat, and function middleware in agent-framework-core 1.1.0 — real signatures, short-circuit patterns, telemetry, retries, and redaction."
framework: microsoft-agent-framework
language: python
---

# Middleware — Python

Middleware is how you intercept agent runs without subclassing `Agent`. Three levels wrap three different call sites:

| Middleware | Wraps | Context class | Use for |
|---|---|---|---|
| **Agent** | A whole `agent.run(...)` call | `AgentContext` | Auth, rate limiting, logging, system-prompt injection, high-level retries |
| **Chat** | A single model request inside the tool loop | `ChatContext` | Per-call observability, prompt caching, token accounting, response rewriting |
| **Function** | A single tool invocation | `FunctionInvocationContext` | Argument validation, PII redaction, approval gates, per-tool telemetry |

All three ship in `agent_framework`; imports below are stable in `agent-framework-core==1.1.0`.

## The `call_next` contract

Every middleware receives a `context` and a zero-argument `call_next`:

```python
async def mw(context, call_next):
    # 1. Code here runs BEFORE the wrapped call
    await call_next()          # advance the pipeline
    # 2. Code here runs AFTER — inspect or mutate context.result
```

Three ways to end execution:

1. **Normal flow** — `await call_next()`, then optionally mutate `context.result`.
2. **Short-circuit** — set `context.result = ...` and **return without calling `call_next`**. Downstream middleware and the actual model / tool call are skipped.
3. **Hard termination** — `raise MiddlewareTermination("reason", result=...)`. Unwinds the pipeline; the agent returns the attached result (or re-raises if none).

## Decorator form

Use the matching decorator to tag a plain function. The tag tells the agent which pipeline the function belongs to, so `middleware=[...]` can mix and match.

```python
from collections.abc import Awaitable, Callable
from agent_framework import (
    Agent,
    AgentContext,
    ChatContext,
    FunctionInvocationContext,
    agent_middleware,
    chat_middleware,
    function_middleware,
)
from agent_framework.openai import OpenAIChatClient


@agent_middleware
async def log_run(context: AgentContext, call_next: Callable[[], Awaitable[None]]) -> None:
    print(f"[{context.agent.name}] received {len(context.messages)} messages")
    await call_next()
    print(f"[{context.agent.name}] returned {context.result.text[:60]}")


@chat_middleware
async def count_tokens(context: ChatContext, call_next):
    before = sum(len((m.text or "")) for m in context.messages)
    await call_next()
    print(f"chars in={before} out={len(context.result.text)}")


@function_middleware
async def log_tool(context: FunctionInvocationContext, call_next):
    print(f"tool call: {context.function.name}({context.arguments})")
    await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    middleware=[log_run, count_tokens, log_tool],
)
```

Order inside `middleware=[...]` is outer-to-inner. `log_run` wraps everything; `count_tokens` wraps each individual model call (so it fires once per tool loop iteration); `log_tool` wraps each tool invocation.

## Class form

Pick the class form when the middleware holds state (retry counts, budgets, token totals) or needs configuration.

```python
from agent_framework import AgentMiddleware, AgentContext, MiddlewareTermination


class BudgetGuard(AgentMiddleware):
    def __init__(self, max_runs: int) -> None:
        self.remaining = max_runs

    async def process(self, context: AgentContext, call_next) -> None:
        if self.remaining <= 0:
            raise MiddlewareTermination("budget exhausted")
        self.remaining -= 1
        await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    instructions="...",
    middleware=[BudgetGuard(max_runs=20)],
)
```

Mix decorator-style and class-style freely — both land in the same pipeline.

## Agent-level vs run-level

`middleware=` on the `Agent` constructor is always active. You can layer additional middleware on a single call via `agent.run(..., middleware=[...])`:

```python
await agent.run(
    "Summarise the attached doc",
    middleware=[count_tokens],      # scoped to this one run
)
```

Run-level middleware runs **inside** agent-level middleware (outer-to-inner: ctor, then run).

## Short-circuiting — block a request

Skip the model entirely when the input fails a policy check:

```python
from agent_framework import AgentMiddleware, AgentResponse, Message, Role, TextContent


class ProfanityBlock(AgentMiddleware):
    async def process(self, context: AgentContext, call_next) -> None:
        last = context.messages[-1] if context.messages else None
        if last and "sensitive-term" in (last.text or "").lower():
            context.result = AgentResponse(
                messages=[Message(role=Role.ASSISTANT, contents=[TextContent("Blocked by policy.")])],
            )
            return                       # do NOT call call_next
        await call_next()
```

## Retrying a failed tool call

Function middleware is the natural place for per-tool retries:

```python
import asyncio
from agent_framework import FunctionMiddleware, FunctionInvocationContext


class RetryOnError(FunctionMiddleware):
    def __init__(self, attempts: int = 3, backoff: float = 0.5) -> None:
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self.attempts = attempts
        self.backoff = backoff

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        last_exc: Exception | None = None
        for i in range(self.attempts):
            try:
                await call_next()
                return
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(self.backoff * (2**i))
        assert last_exc is not None  # attempts >= 1 guarantees we saw at least one exception
        raise last_exc  # give up
```

## Redacting sensitive outputs

Chat middleware sees the finalised `ChatResponse` after the model call. Rewrite it in place:

```python
import re
from agent_framework import ChatMiddleware, ChatContext, ChatResponse

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class Redactor(ChatMiddleware):
    """Redacts emails from non-streaming responses.

    `ChatContext.result` is only a `ChatResponse` when `context.stream is False`.
    For streaming calls, register a `stream_transform_hooks` entry instead.
    """

    async def process(self, context: ChatContext, call_next) -> None:
        await call_next()
        if context.stream or not isinstance(context.result, ChatResponse):
            return
        for msg in context.result.messages:
            for content in msg.contents:
                if getattr(content, "text", None):
                    content.text = EMAIL_RE.sub("[email]", content.text)
```

`ChatContext.result` is a `ChatResponse` for non-streaming calls and a `ResponseStream[ChatResponseUpdate, ChatResponse]` for streaming. For streaming, append to `context.stream_transform_hooks` to rewrite each `ChatResponseUpdate` as it arrives:

```python
def redact_update(update):
    for content in update.contents or []:
        if getattr(content, "text", None):
            content.text = EMAIL_RE.sub("[email]", content.text)
    return update


class StreamingRedactor(ChatMiddleware):
    async def process(self, context: ChatContext, call_next) -> None:
        context.stream_transform_hooks.append(redact_update)
        await call_next()
```

## Passing per-run data through the pipeline

All three context classes expose a mutable `metadata` dict. Use it to hand data down the chain or up to the caller:

```python
@agent_middleware
async def tag_tenant(context, call_next):
    context.metadata["tenant_id"] = context.kwargs.get("tenant_id", "default")
    await call_next()


@function_middleware
async def log_tenant(context, call_next):
    print(f"tool invoked for tenant={context.metadata.get('tenant_id')}")
    await call_next()


await agent.run("...", function_invocation_kwargs={"tenant_id": "acme"})
```

`function_invocation_kwargs` in the outer `agent.run(...)` call surfaces as `context.kwargs` inside function middleware, so any runtime secrets, tenant IDs, or request-scoped state flow through cleanly.

## Emitting OpenTelemetry spans

Agent-framework auto-emits `agent_framework.*` spans without any middleware. Use middleware only when you want a business-level span around it:

```python
from opentelemetry import trace
from agent_framework import AgentMiddleware

tracer = trace.get_tracer("myapp.agent")


class TraceRun(AgentMiddleware):
    async def process(self, context, call_next) -> None:
        with tracer.start_as_current_span(
            "agent.run",
            attributes={"agent.name": context.agent.name, "msg_count": len(context.messages)},
        ) as span:
            await call_next()
            span.set_attribute("response.length", len(context.result.text))
```

## Context quick reference

### `AgentContext`

| Field | Type | Notes |
|---|---|---|
| `agent` | `SupportsAgentRun` | The invoked agent. |
| `messages` | `list[Message]` | Full input (incl. system/instructions). |
| `session` | `AgentSession \| None` | Bound session, if any. |
| `tools` | override | Run-scoped tool override. |
| `options` | `Mapping[str, Any]` | Merged `ChatOptions` dict. |
| `stream` | `bool` | True for streaming runs. |
| `metadata` | `dict` | Shared across middleware in this run. |
| `result` | `AgentResponse \| ResponseStream` | Set to short-circuit. |
| `kwargs` | `dict` | Run-level keyword args. |
| `client_kwargs` | `dict` | Forwarded to chat client. |
| `function_invocation_kwargs` | `dict` | Forwarded to tool invocation. |

### `ChatContext`

| Field | Notes |
|---|---|
| `client` | `SupportsChatGetResponse` — the underlying chat client. |
| `messages`, `options`, `stream`, `metadata`, `result`, `kwargs` | As above. |
| `function_invocation_kwargs` | Only to tool invocation. |
| `stream_transform_hooks` / `stream_result_hooks` / `stream_cleanup_hooks` | Hook sequences for streaming pipelines. |

### `FunctionInvocationContext`

| Field | Notes |
|---|---|
| `function` | `FunctionTool` being invoked. |
| `arguments` | Parsed args (Pydantic model or dict). |
| `session`, `metadata`, `result`, `kwargs` | As above. |

## Breaking changes (2026 line)

- `middleware=` now **requires a list**. A single instance raises `TypeError`.
- `AgentRunResponse` / `AgentRunResponseUpdate` renamed to `AgentResponse` / `AgentResponseUpdate`.
- `AggregateContextProvider` removed — compose providers directly.

The `context_providers` parameter on `Agent` is still **plural** in 1.1.0, contrary to earlier drafts of this guide.

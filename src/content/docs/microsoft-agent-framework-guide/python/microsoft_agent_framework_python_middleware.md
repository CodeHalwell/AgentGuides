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
from agent_framework import AgentMiddleware, AgentResponse, Content, Message


class ProfanityBlock(AgentMiddleware):
    async def process(self, context: AgentContext, call_next) -> None:
        last = context.messages[-1] if context.messages else None
        if last and "sensitive-term" in (last.text or "").lower():
            context.result = AgentResponse(
                messages=[
                    Message(
                        role="assistant",
                        contents=[Content.from_text("Blocked by policy.")],
                    )
                ],
            )
            return                       # do NOT call call_next
        await call_next()
```

`agent_framework` ships a single unified `Content` class — construct text content via `Content.from_text(...)`, images via `Content.from_uri(...)`, errors via `Content.from_error(...)`, etc. There are no separate `TextContent`/`ImageContent` classes.

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

## Streaming hooks on `ChatContext`

When `context.stream is True`, `context.result` is a `ResponseStream[ChatResponseUpdate, ChatResponse]` — you can't just rewrite it the way you'd rewrite a `ChatResponse`. Instead, register hooks that run at three distinct points of the stream's lifecycle:

| Hook | Fires | Use for |
|---|---|---|
| `stream_transform_hooks` | Once per yielded `ChatResponseUpdate` | Mask PII / inject metadata / rewrite tokens as they stream |
| `stream_result_hooks` | Once on the finalised `ChatResponse` (after the stream completes) | Final-pass cleanup, audit logging, trace linking |
| `stream_cleanup_hooks` | Once after the stream is fully consumed (before the finaliser) | Flush a metric, close a span, release a lock |

Each list accepts sync **or** async callables. Add to it before calling `call_next()` so the hook runs against the stream the underlying client returns.

```python
import logging
import re
from agent_framework import ChatMiddleware, ChatContext, ChatResponseUpdate

logger = logging.getLogger(__name__)
PHONE_RE = re.compile(r"\+?\d[\d -]{8,}\d")


class StreamingPiiRedactor(ChatMiddleware):
    async def process(self, context: ChatContext, call_next) -> None:
        # Transform every chunk as it arrives.
        async def redact_chunk(update: ChatResponseUpdate) -> ChatResponseUpdate:
            for content in update.contents or []:
                text = getattr(content, "text", None)
                if text:
                    content.text = PHONE_RE.sub("[redacted-phone]", text)
            return update

        # Run a final pass on the assembled response (in case anything slipped through).
        async def final_pass(response):
            for msg in response.messages:
                for content in msg.contents:
                    text = getattr(content, "text", None)
                    if text:
                        content.text = PHONE_RE.sub("[redacted-phone]", text)
            return response

        # Always clean up — even if the consumer aborts mid-stream.
        async def close_span():
            logger.info("chat.stream.completed middleware=pii")

        context.stream_transform_hooks.append(redact_chunk)
        context.stream_result_hooks.append(final_pass)
        context.stream_cleanup_hooks.append(close_span)
        await call_next()
```

A few practical notes:

- **Hook order matters** — hooks run in the order they were registered. Stack the cheap deterministic redactor before the expensive LLM-based moderation hook so the cleaner output reaches the moderator.
- **Sync hooks are fine** — the framework `await`s anything that returns an awaitable and otherwise calls the hook directly.
- **Don't mix `context.result = ...` with hooks** — for streaming, set the hooks; the framework wires them into the live stream. Setting `context.result` to a fresh `ResponseStream` wholesale only makes sense for synthetic short-circuit responses.
- **Mirroring for non-streaming.** `AgentContext` exposes the same trio (`stream_transform_hooks` / `stream_result_hooks` / `stream_cleanup_hooks`) for agent-level streaming — register them there if you want the redactor to apply across every chat call inside one agent run.

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

## `ChatMiddleware` — caching the model call

Because chat middleware wraps the actual model invocation (every iteration of the tool loop), it's the right place to short-circuit the network call when you already have the answer:

```python
import hashlib
import json
import time
from agent_framework import ChatContext, ChatMiddleware, ChatResponse, Message


class MemoryChatCache(ChatMiddleware):
    """In-memory chat-call cache keyed on the (messages, options) pair.

    Skips downstream chat clients on a cache hit by setting ``context.result``
    *without* calling ``call_next()``. Streaming calls fall through to the
    real client — caching streams without re-yielding chunks needs more care.
    """

    def __init__(self, ttl_seconds: float = 300) -> None:
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, ChatResponse]] = {}

    @staticmethod
    def _key(messages: list[Message], options: dict) -> str:
        blob = json.dumps(
            {
                "m": [(m.role, m.text or "") for m in messages],
                "o": {k: v for k, v in options.items() if isinstance(v, (str, int, float, bool))},
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    async def process(self, context: ChatContext, call_next) -> None:
        if context.stream:
            await call_next()
            return

        key = self._key(list(context.messages), dict(context.options or {}))
        cached = self._store.get(key)
        if cached and cached[0] >= time.monotonic():
            context.result = cached[1]              # short-circuit: skip the model
            return

        await call_next()
        if isinstance(context.result, ChatResponse):
            self._store[key] = (time.monotonic() + self.ttl, context.result)
```

Two things this pattern leans on that aren't immediately obvious:

- **Setting `context.result` before `call_next()` skips the model.** No tool loop iteration is consumed. This is different from agent middleware, which short-circuits the entire run (but the model would still see the cached response on the next agent call).
- **Stream caching is opt-in.** The pattern above bypasses cache lookups when `context.stream is True`. For streaming caches, accumulate chunks in a `stream_transform_hooks` callback and replay them via a synthetic `ResponseStream` — the [streaming hooks section](#streaming-hooks-on-chatcontext) shows the hook-based shape.

## `agent_middleware` decorator with stateful closures

The `@agent_middleware` decorator marks a plain function for the agent pipeline, but the function can still capture mutable state via a closure — handy when you want middleware-as-config without writing a class:

```python
from agent_framework import Agent, AgentContext, agent_middleware


def make_concurrency_limiter(max_inflight: int):
    """Limit how many runs can be in flight concurrently for this agent."""
    import asyncio
    semaphore = asyncio.Semaphore(max_inflight)

    @agent_middleware
    async def limiter(context: AgentContext, call_next) -> None:
        async with semaphore:
            await call_next()

    return limiter


def make_run_counter():
    """Count completed runs by name — useful for cheap traffic dashboards."""
    counts: dict[str, int] = {}

    @agent_middleware
    async def counter(context: AgentContext, call_next) -> None:
        await call_next()
        counts[context.agent.name or "<unnamed>"] = counts.get(context.agent.name or "<unnamed>", 0) + 1

    counter.counts = counts                         # expose for inspection
    return counter


limiter = make_concurrency_limiter(max_inflight=4)
counter = make_run_counter()

agent = Agent(client=client, name="research", middleware=[limiter, counter])
# After running:
print(counter.counts)
```

`@agent_middleware` (and its `@chat_middleware` / `@function_middleware` siblings) sets a `_middleware_type` attribute on the function so the framework routes it into the right pipeline — that's the only difference between the bare callable and an explicit `AgentMiddleware` subclass. Mixing decorated functions with class-based middleware in the same `middleware=[...]` list is fully supported.

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

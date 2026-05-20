---
title: "Microsoft Agent Framework (Python) — Middleware"
description: "Agent, chat, and function middleware in agent-framework-core 1.5.0 — real signatures, short-circuit patterns, telemetry, retries, and redaction."
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

All three ship in `agent_framework`; imports below are stable in `agent-framework-core==1.5.0`.

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

### Hard termination via `MiddlewareTermination`

`MiddlewareTermination` is a control-flow exception that unwinds the entire pipeline immediately. Code that runs **after `await call_next()` under normal flow** is skipped in every outer middleware — but `try`/`finally` blocks and `async with` cleanup still execute as the exception propagates, so context-manager-based metric flushing, span-closing, or lock release still works as expected. `MiddlewareTermination` carries an optional `result=` payload that becomes the agent's final response.

```python
from agent_framework import (
    AgentMiddleware,
    AgentContext,
    AgentResponse,
    Content,
    Message,
    MiddlewareTermination,
)


class HardBudgetGuard(AgentMiddleware):
    """Reject the call outright when over a per-tenant quota.

    Compared to `context.result = ...; return`, raising `MiddlewareTermination`
    skips every other middleware's after-call code too — useful when you don't
    want token-counting, metric, or cache-write middleware to mistakenly
    record a request as successful.
    """

    def __init__(self, quota: dict[str, int]) -> None:
        self.quota = quota

    async def process(self, context: AgentContext, call_next) -> None:
        tenant = context.metadata.get("tenant_id")
        if tenant and self.quota.get(tenant, 0) <= 0:
            raise MiddlewareTermination(
                f"tenant {tenant} over quota",
                result=AgentResponse(
                    messages=[
                        Message(
                            role="assistant",
                            contents=[Content.from_text("Quota exhausted.")],
                        )
                    ],
                ),
            )
        await call_next()
```

When `MiddlewareTermination(result=R)` is raised the agent returns `R` as if it were the normal result. Raised without `result=`, it becomes a regular exception you can catch and translate at the call site:

```python
try:
    response = await agent.run("Hello")
except MiddlewareTermination as exc:
    print(f"blocked: {exc}")  # exc.result is None
```

## Caching tool results — `FunctionMiddleware`

Pure or read-mostly tools shouldn't re-run when the model asks the same question twice in a row. Function middleware sees every tool call and can short-circuit on a cache hit — either by setting `context.result` and returning, or by raising `MiddlewareTermination(result=...)` when you want to unwind the whole pipeline. The example below uses the `MiddlewareTermination` form so the result is delivered all the way back to the agent without further middleware running:

```python
import json
from typing import Any
from agent_framework import (
    Agent,
    FunctionMiddleware,
    FunctionInvocationContext,
    MiddlewareTermination,
)
from agent_framework.openai import OpenAIChatClient


class FunctionCallCache(FunctionMiddleware):
    """Memoise idempotent tools across a single agent run (or longer).

    Keyed on (tool_name, arguments) — argument order is normalised by JSON
    sort_keys. Only caches tools tagged `kind="readonly"`; everything else
    falls through.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _key(name: str, arguments: dict[str, Any]) -> str:
        return f"{name}::{json.dumps(arguments, sort_keys=True, default=str)}"

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        # Only cache tools that explicitly opt in via @tool(kind="readonly")
        if context.function.kind != "readonly":
            await call_next()
            return

        key = self._key(context.function.name, context.arguments or {})
        if key in self._cache:
            # Short-circuit: skip the wrapped call entirely.
            raise MiddlewareTermination(
                "cache hit",
                result=self._cache[key],
            )

        await call_next()
        if context.result is not None:
            self._cache[key] = context.result


agent = Agent(
    client=OpenAIChatClient(),
    instructions="…",
    middleware=[FunctionCallCache()],
)
```

The `kind` filter is a contract with the tool authors: opt in with `@tool(kind="readonly")` and the cache covers you; otherwise the call goes through. Two extensions you'll commonly want:

- Add a TTL by storing `(time.monotonic(), value)` tuples and checking expiry on hit.
- Move the cache to Redis to share across replicas — same shape, swap the dict for an async client.

## Per-tool circuit breaker

A flaky downstream API can burn through the agent's iteration budget retrying the same broken tool. Trip a per-tool circuit after N consecutive failures, then refuse subsequent calls for a cool-down window:

```python
import time
from collections import defaultdict
from agent_framework import (
    FunctionMiddleware,
    FunctionInvocationContext,
    MiddlewareTermination,
)


class CircuitBreaker(FunctionMiddleware):
    """Open the circuit on N consecutive failures; refuse calls for cool_down seconds."""

    def __init__(self, *, threshold: int = 3, cool_down: float = 60.0) -> None:
        self.threshold = threshold
        self.cool_down = cool_down
        self._failures: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}

    def _is_open(self, tool: str) -> bool:
        opened = self._opened_at.get(tool)
        if opened is None:
            return False
        if time.monotonic() - opened > self.cool_down:
            # Half-open: clear state and allow the next call through.
            self._opened_at.pop(tool, None)
            self._failures[tool] = 0
            return False
        return True

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        tool = context.function.name
        if self._is_open(tool):
            raise MiddlewareTermination(
                f"{tool} circuit open — refusing call",
                result=f"[{tool} unavailable; try again later]",
            )

        try:
            await call_next()
        except Exception:
            self._failures[tool] += 1
            if self._failures[tool] >= self.threshold:
                self._opened_at[tool] = time.monotonic()
            raise
        else:
            self._failures[tool] = 0          # reset on success
```

`MiddlewareTermination(result=...)` lets you hand the model a synthetic answer when the circuit is open — much friendlier than letting the exception propagate. The model can either retry a different tool or apologise to the user; either way, you stop hammering the broken backend.

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

## Rewriting tool results — `FunctionInvocationContext.result`

`FunctionInvocationContext.result` is `None` until you `await call_next()`; afterwards it holds whatever the tool returned. You can both **observe** it (for logging) and **override** it before it lands back in the chat loop. This is the cleanest way to redact, truncate, or normalise tool output without touching the tool itself.

```python
import json
from agent_framework import FunctionMiddleware, FunctionInvocationContext


class TruncateLargeJSON(FunctionMiddleware):
    """Cap a tool's JSON return value at N characters so an oversized DB row
    or web-scrape doesn't blow the model's context window.

    `context.result` is whatever the tool returned — could be a string, dict,
    Pydantic model, dataclass, or `None`. Cast carefully before mutating.
    """

    def __init__(self, max_chars: int = 4_000) -> None:
        self.max_chars = max_chars

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        await call_next()

        # Serialise dict / list / model into JSON for length checks; pass strings through.
        as_text = (
            context.result
            if isinstance(context.result, str)
            else json.dumps(context.result, default=str)
        )
        if len(as_text) > self.max_chars:
            context.result = (
                as_text[: self.max_chars]
                + f"\n…[truncated {len(as_text) - self.max_chars} chars]"
            )
```

`FunctionInvocationContext` also exposes a `metadata` dict that's shared across **every middleware in the same invocation**. Use it to forward a request-id, a tenant marker, or a timing measurement from one middleware to another without touching the tool's signature:

```python
import logging
import time
from agent_framework import (
    Agent,
    FunctionMiddleware,
    FunctionInvocationContext,
)
from agent_framework.openai import OpenAIChatClient

log = logging.getLogger(__name__)


class StartTimer(FunctionMiddleware):
    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        context.metadata["my_app.t0"] = time.monotonic()
        await call_next()


class RecordLatency(FunctionMiddleware):
    """Reads the timestamp another middleware put on `context.metadata`.

    Both middleware run on the same `FunctionInvocationContext` instance, so
    metadata flows freely. Just be careful with key names — namespace yours
    (`my_app.t0`, `my_app.tenant`) so middleware from different teams don't
    clobber each other.

    Replace ``log.info`` with your real metrics emitter (Prometheus, OpenTelemetry,
    StatsD, …); it's left as a logger here so the snippet runs without extra deps.
    """

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        await call_next()
        elapsed_ms = (time.monotonic() - context.metadata["my_app.t0"]) * 1_000
        log.info("tool.%s.latency_ms=%.2f", context.function.name, elapsed_ms)


agent = Agent(
    client=OpenAIChatClient(),
    middleware=[StartTimer(), RecordLatency(), TruncateLargeJSON(max_chars=8_000)],
)
```

The `arguments` field on `FunctionInvocationContext` is **already validated** — it's either a Pydantic model instance (when the tool's schema produced one) or a `Mapping[str, Any]`. You can mutate it before `call_next()` to inject defaults, or use it to short-circuit when the args fail an external policy check by raising `MiddlewareTermination(result=...)`.

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
- **Mirroring for non-streaming.** `AgentContext` exposes the same trio (`stream_transform_hooks` / `stream_result_hooks` / `stream_cleanup_hooks`) for agent-level streaming — register them there if you want the redactor to apply across the whole agent run, not per-model-call.

## Streaming hooks on `AgentContext`

`AgentContext` streaming hooks operate at the **outer agent run boundary** — they see the assembled `AgentResponseUpdate` stream that the caller consumes, not individual model call chunks. This makes them the right layer for:

- Injecting run-level spans that open when the first token arrives and close when the consumer finishes
- Transforming the final `AgentResponse` (e.g. post-processing structured output)
- Releasing run-scoped locks or quotas once the consumer is done

The three hooks on `AgentContext` behave exactly like their `ChatContext` equivalents:

| Hook | Fires | Receives |
|---|---|---|
| `stream_transform_hooks` | Once per `AgentResponseUpdate` yielded to the caller | `AgentResponseUpdate` — return the (optionally mutated) update |
| `stream_result_hooks` | Once, on the final `AgentResponse` after the stream completes | `AgentResponse` — return the (optionally mutated) response |
| `stream_cleanup_hooks` | Once, after the stream is fully consumed | Nothing — `() -> None` or `() -> Awaitable[None]` |

```python
import asyncio
import logging
import time
from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    AgentResponseUpdate,
)
from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)


class AgentStreamingInstrumentor(AgentMiddleware):
    """Attach timing and word-count telemetry to every streaming agent run.

    Hooks registered on AgentContext fire once per run — unlike ChatContext
    hooks, which fire once per model call inside the tool loop. Use this layer
    when you need a single outer span that covers all LLM calls and tool
    invocations made during one agent.run().
    """

    async def process(self, context: AgentContext, call_next) -> None:
        if not context.stream:
            await call_next()
            return

        run_start = time.monotonic()
        word_count = 0

        def count_words(update: AgentResponseUpdate) -> AgentResponseUpdate:
            nonlocal word_count
            if update.text:
                word_count += len(update.text.split())
            return update

        async def record_result(response: AgentResponse) -> AgentResponse:
            elapsed_ms = (time.monotonic() - run_start) * 1_000
            logger.info(
                "agent.streaming_run.completed",
                extra={
                    "agent": context.agent.name,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "output_words": word_count,
                    "finish_reasons": response.finish_reasons,
                },
            )
            return response

        async def cleanup() -> None:
            # Fires even if the consumer breaks out of the stream early.
            logger.debug("agent.stream.consumer.done agent=%s", context.agent.name)

        context.stream_transform_hooks.append(count_words)
        context.stream_result_hooks.append(record_result)
        context.stream_cleanup_hooks.append(cleanup)
        await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    middleware=[AgentStreamingInstrumentor()],
)


async def main():
    # The hooks only activate for streaming runs:
    stream = agent.run("Explain transformer attention in 3 paragraphs.", stream=True)
    async for update in stream:
        if update.text:
            print(update.text, end="", flush=True)
    # AgentStreamingInstrumentor logs elapsed_ms and output_words once here.


asyncio.run(main())
```

Key difference from `ChatContext` hooks: if the agent makes 3 tool calls and 4 LLM roundtrips to answer your question, `ChatContext.stream_transform_hooks` would fire for every token from every one of those 4 model calls. `AgentContext.stream_transform_hooks` fires for every token in the **outer** `ResponseStream` that the caller iterates — typically the same tokens, but scoped to one accounting boundary instead of four.

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
from typing import Any
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
    def _key(messages: list[Message], options: dict[str, Any]) -> str:
        # Use Message.to_dict() so non-text content (function calls, images,
        # tool results) and structured option values participate in the key.
        # `default=str` covers anything not natively JSON-serialisable
        # (e.g. enum values, datetimes) so distinct objects produce distinct keys.
        blob = json.dumps(
            {
                "m": [m.to_dict() for m in messages],
                "o": options,
            },
            sort_keys=True,
            default=str,
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

## Building a synthetic `ChatResponse`

When chat middleware wants to satisfy a request from a cache, mock, or fallback model, hand back a fully-formed `ChatResponse` rather than a raw string. Downstream code (token accounting, telemetry, structured-output parsing) treats it identically to a real model response.

```python
from agent_framework import (
    ChatMiddleware,
    ChatContext,
    ChatResponse,
    Content,
    Message,
    UsageDetails,
)


class CacheLookup(ChatMiddleware):
    def __init__(self, cache: dict[str, str]) -> None:
        self.cache = cache

    async def process(self, context: ChatContext, call_next) -> None:
        key = context.messages[-1].text if context.messages else ""
        cached = self.cache.get(key)
        if cached is not None and not context.stream:
            context.result = ChatResponse(
                messages=[Message(role="assistant", contents=[Content.from_text(cached)])],
                model=context.options.get("model") if context.options else None,
                finish_reason="stop",
                usage_details=UsageDetails(input_token_count=0, output_token_count=0),
            )
            return                                      # short-circuit
        await call_next()
```

When you need to assemble a `ChatResponse` from a stream (e.g. you're proxying a streaming provider and want to give non-streaming consumers the joined result), `ChatResponse.from_updates(updates)` and `ChatResponse.from_update_generator(async_iter)` consolidate `ChatResponseUpdate` chunks into a single response, including structured-output parsing when you pass `output_format_type=`.

```python
from agent_framework import ChatResponse

# Sync path: list of updates already collected
final = ChatResponse.from_updates(updates)

# Async path: forward an async iterator straight from a streaming client
final = await ChatResponse.from_update_generator(client.get_streaming_response("hi"))

# Structured output: pass a Pydantic model and read `final.value`
final = await ChatResponse.from_update_generator(
    client.get_streaming_response("Extract", response_format=Address),
    output_format_type=Address,
)
print(final.text)   # raw text
print(final.value)  # parsed Address instance, lazily validated
```

`final.value` triggers parsing on first access — if the response text doesn't match the schema you'll see a `pydantic.ValidationError` (or `ValueError` for non-Pydantic JSON-schema formats), which is exactly the boundary you want for structured-output retries.

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
| `compaction_strategy` | `CompactionStrategy \| None` | Per-run compaction override. |
| `tokenizer` | `TokenizerProtocol \| None` | Per-run tokenizer override. |
| `metadata` | `dict` | Shared across middleware in this run. |
| `result` | `AgentResponse \| ResponseStream` | Set to short-circuit. |
| `kwargs` | `dict` | Run-level keyword args. |
| `client_kwargs` | `dict` | Forwarded to chat client. |
| `function_invocation_kwargs` | `dict` | Forwarded to tool invocation. |
| `stream_transform_hooks` | `list[Callable]` | Per-update transformers (streaming runs). |
| `stream_result_hooks` | `list[Callable]` | Hooks on the joined `AgentResponse` after streaming. |
| `stream_cleanup_hooks` | `list[Callable]` | Cleanup callbacks once the stream is consumed. |

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

The `context_providers` parameter on `Agent` is still **plural** in 1.4.0, contrary to earlier drafts of this guide.

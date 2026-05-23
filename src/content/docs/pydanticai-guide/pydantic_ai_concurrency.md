---
title: "PydanticAI: Concurrency Limiting"
description: "ConcurrencyLimiter, ConcurrencyLimit, and AbstractConcurrencyLimiter — cap parallel model requests with optional backpressure and OpenTelemetry observability."
framework: pydanticai
language: python
---

# Concurrency Limiting

Verified against **pydantic-ai==1.102.0** — source module: `pydantic_ai.concurrency`.

`ConcurrencyLimiter` caps the number of simultaneous model API calls without any external dependencies. When the cap is reached, additional callers queue and wait. Optionally set `max_queued` to reject requests once the queue depth exceeds a threshold.

## Minimal runnable example

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter

limiter = ConcurrencyLimiter(max_running=3)

agent = Agent('openai:gpt-4o')

async def main():
    # Run 10 queries but allow at most 3 in flight at once
    prompts = [f'What is {i} squared?' for i in range(10)]
    results = await asyncio.gather(
        *[agent.run(p, model_settings={'concurrency_limiter': limiter}) for p in prompts]
    )
    for r in results:
        print(r.output)

asyncio.run(main())
```

## `ConcurrencyLimiter` constructor

```python
from pydantic_ai.concurrency import ConcurrencyLimiter, ConcurrencyLimit

# Simple: max 5 concurrent requests, unlimited queue
limiter = ConcurrencyLimiter(max_running=5)

# With backpressure: raise ConcurrencyLimitExceeded if > 10 tasks are already waiting
limiter = ConcurrencyLimiter(max_running=5, max_queued=10)

# Named (shows in OTel spans and error messages)
limiter = ConcurrencyLimiter(max_running=5, name='openai-prod')

# From a ConcurrencyLimit config object
config = ConcurrencyLimit(max_running=5, max_queued=10)
limiter = ConcurrencyLimiter.from_limit(config, name='openai-prod')
```

## `ConcurrencyLimit` — configuration object

```python
from pydantic_ai.concurrency import ConcurrencyLimit

# Serialisable config — useful when limits come from env vars or config files
config = ConcurrencyLimit(max_running=10, max_queued=50)
```

## Observability properties

```python
limiter = ConcurrencyLimiter(max_running=5, name='my-limiter')

print(limiter.max_running)      # 5
print(limiter.running_count)    # current concurrent calls
print(limiter.waiting_count)    # tasks queued right now
print(limiter.available_count)  # free slots
print(limiter.name)             # 'my-limiter'
```

When a task must wait, `ConcurrencyLimiter.acquire()` creates an OpenTelemetry span named `"waiting for <source> concurrency"` — it shows up in your trace waterfall automatically.

## Backpressure with `max_queued`

```python
from pydantic_ai.exceptions import ConcurrencyLimitExceeded
from pydantic_ai.concurrency import ConcurrencyLimiter

strict_limiter = ConcurrencyLimiter(max_running=2, max_queued=5)

try:
    # If 5 tasks are already waiting, this raises immediately
    await agent.run('prompt', model_settings={'concurrency_limiter': strict_limiter})
except ConcurrencyLimitExceeded as e:
    print(f'Rejected: {e}')
    # Handle gracefully — e.g. return a 429 to the caller
```

## Sharing a limiter across agents

A single `ConcurrencyLimiter` can be shared so the *total* concurrent calls across all agents stays bounded:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter

shared = ConcurrencyLimiter(max_running=10, name='shared-pool')

summariser = Agent('openai:gpt-4o-mini')
analyst    = Agent('openai:gpt-4o')

async def process(text: str):
    summary = await summariser.run(
        f'Summarise: {text}',
        model_settings={'concurrency_limiter': shared},
    )
    analysis = await analyst.run(
        f'Analyse: {summary.output}',
        model_settings={'concurrency_limiter': shared},
    )
    return analysis.output

# At most 10 model calls run at once across both agents
results = await asyncio.gather(*[process(doc) for doc in documents])
```

## Per-provider concurrency limiting

Use separate limiters per provider to avoid hitting provider-specific rate limits:

```python
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter

openai_limiter   = ConcurrencyLimiter(max_running=20, name='openai')
anthropic_limiter = ConcurrencyLimiter(max_running=5,  name='anthropic')

openai_agent    = Agent('openai:gpt-4o')
anthropic_agent = Agent('anthropic:claude-opus-4-5')

# At call sites, pass the appropriate limiter
async def call_openai(prompt):
    return await openai_agent.run(
        prompt,
        model_settings={'concurrency_limiter': openai_limiter},
    )

async def call_anthropic(prompt):
    return await anthropic_agent.run(
        prompt,
        model_settings={'concurrency_limiter': anthropic_limiter},
    )
```

## Custom limiter — Redis-backed distributed limiting

Subclass `AbstractConcurrencyLimiter` to integrate with external rate limiters:

```python
import asyncio
from pydantic_ai.concurrency import AbstractConcurrencyLimiter

class RedisLimiter(AbstractConcurrencyLimiter):
    """Distributed concurrency limiting backed by Redis."""

    def __init__(self, redis_client, key: str, max_running: int):
        self._redis = redis_client
        self._key = key
        self._max = max_running
        self._local_sem = asyncio.Semaphore(max_running)

    async def acquire(self, source: str) -> None:
        # Local semaphore prevents local over-subscription
        await self._local_sem.acquire()
        # Remote lock prevents cross-process over-subscription
        while True:
            count = int(await self._redis.get(self._key) or 0)
            if count < self._max:
                await self._redis.incr(self._key)
                return
            await asyncio.sleep(0.05)

    def release(self) -> None:
        self._local_sem.release()
        asyncio.create_task(self._redis.decr(self._key))
```

## `AnyConcurrencyLimit` type alias

The `model_settings` dict accepts all these forms interchangeably:

```python
from pydantic_ai.concurrency import AnyConcurrencyLimit

# AnyConcurrencyLimit = int | ConcurrencyLimit | AbstractConcurrencyLimiter | None

# int shorthand — creates a ConcurrencyLimiter(max_running=N) internally
settings = {'concurrency_limiter': 5}

# ConcurrencyLimit config object
settings = {'concurrency_limiter': ConcurrencyLimit(max_running=5, max_queued=20)}

# Pre-created limiter instance (for sharing)
settings = {'concurrency_limiter': shared_limiter}

# None — no limiting (default)
settings = {'concurrency_limiter': None}
```

## FastAPI integration — request-level limiting

Use a shared limiter with backpressure to protect your service under load:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

limiter = ConcurrencyLimiter(max_running=10, max_queued=20, name='api')
agent = Agent('openai:gpt-4o')

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.limiter = limiter
    yield

app = FastAPI(lifespan=lifespan)

@app.post('/ask')
async def ask(prompt: str):
    try:
        result = await agent.run(
            prompt,
            model_settings={'concurrency_limiter': app.state.limiter},
        )
        return {'output': result.output}
    except ConcurrencyLimitExceeded:
        raise HTTPException(status_code=429, detail='Too many concurrent requests')
```

## Monitoring active slots

```python
import asyncio
from pydantic_ai.concurrency import ConcurrencyLimiter

limiter = ConcurrencyLimiter(max_running=5, name='monitor-demo')

async def status_reporter():
    while True:
        print(
            f'running={limiter.running_count}  '
            f'waiting={limiter.waiting_count}  '
            f'available={limiter.available_count}'
        )
        await asyncio.sleep(1.0)
```

## `normalize_to_limiter` — utility for library authors

`normalize_to_limiter` converts any `AnyConcurrencyLimit` value into a concrete `AbstractConcurrencyLimiter` instance (or `None`). Use it in custom model wrappers or toolsets that accept user-provided concurrency config:

```python
from pydantic_ai.concurrency import (
    normalize_to_limiter,
    AnyConcurrencyLimit,
    ConcurrencyLimit,
    ConcurrencyLimiter,
)

def setup_limiter(config: AnyConcurrencyLimit, name: str) -> None:
    limiter = normalize_to_limiter(config, name=name)
    if limiter is None:
        print('No concurrency limiting configured')
    else:
        print(f'Limiter: {type(limiter).__name__}, max_running={limiter.max_running}')

setup_limiter(None, 'test')                              # No limiting
setup_limiter(5, 'test')                                 # Creates ConcurrencyLimiter(5)
setup_limiter(ConcurrencyLimit(5, max_queued=20), 'test') # Creates ConcurrencyLimiter(5, max_queued=20)
setup_limiter(ConcurrencyLimiter(10), 'test')            # Returns as-is
```

## `get_concurrency_context` — async context manager wrapper

`get_concurrency_context` returns an async context manager that acquires and releases the limiter. Passing `None` returns a no-op manager. This is the internal primitive used by PydanticAI's agent graph — useful when building custom runners:

```python
import asyncio
from pydantic_ai.concurrency import (
    ConcurrencyLimiter,
    get_concurrency_context,
    normalize_to_limiter,
    AnyConcurrencyLimit,
)

async def run_with_limit(work, limit: AnyConcurrencyLimit = None):
    """Run an async callable under an optional concurrency limit."""
    limiter = normalize_to_limiter(limit, name='custom-runner')
    async with get_concurrency_context(limiter, source='custom-runner'):
        return await work()

async def main():
    limiter = ConcurrencyLimiter(max_running=3)

    tasks = [
        run_with_limit(lambda: asyncio.sleep(0.1), limit=limiter)
        for _ in range(10)
    ]
    await asyncio.gather(*tasks)
    print('All tasks completed with concurrency ≤ 3')

asyncio.run(main())
```

## How OTel spans are created

When a task must **wait** (no free slots), `ConcurrencyLimiter.acquire()` creates an OpenTelemetry span automatically:

- **Span name**: `"waiting for <limiter-name-or-source> concurrency"`
- **Attributes**: `source`, `waiting_count`, `max_running`, `limiter_name` (if set), `max_queued` (if set)

This appears in your distributed trace waterfall as a latency contributor — you can see how long tasks spend waiting for a slot without any extra instrumentation code.

```python
from pydantic_ai.concurrency import ConcurrencyLimiter

# Name appears in the OTel span and in ConcurrencyLimitExceeded error messages
limiter = ConcurrencyLimiter(max_running=5, name='openai-gpt4o-pool')
```

## Reference

| Symbol | Module | Notes |
|---|---|---|
| `ConcurrencyLimiter` | `pydantic_ai.concurrency` | Main limiter class with OTel observability |
| `ConcurrencyLimit` | `pydantic_ai.concurrency` | Serialisable config dataclass |
| `AbstractConcurrencyLimiter` | `pydantic_ai.concurrency` | ABC for custom (e.g. Redis-backed) limiters |
| `AnyConcurrencyLimit` | `pydantic_ai.concurrency` | Type alias: `int \| ConcurrencyLimit \| AbstractConcurrencyLimiter \| None` |
| `normalize_to_limiter` | `pydantic_ai.concurrency` | Normalise any `AnyConcurrencyLimit` → limiter instance or `None` |
| `get_concurrency_context` | `pydantic_ai.concurrency` | Async context manager wrapping acquire/release |
| `ConcurrencyLimitExceeded` | `pydantic_ai.exceptions` | Raised when queue depth exceeds `max_queued` |

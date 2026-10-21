---
title: "Mistral Agents API: Production Deployment Guide"
description: "Running services on Mistral's Agents API in production: configuration, concurrency, retries, observability, storage, rate limits, security and CI/CD."
framework: "mistral-agents-api"
language: "python"
---

**Verified against `mistralai` %%version=mistralai%%** (checked %%versions-checked%%).

The agent loop, built-in tools and conversation storage run on Mistral's side, so your service is mostly an API client: it needs sensible concurrency, retries, observability, its own record of which conversations belong to whom, and careful handling of secrets.

---

## 1. Configuration

```bash
# Credentials (one of)
MISTRAL_API_KEY=...
MISTRAL_SA_TOKEN_PATH=/var/run/secrets/mistral/token   # service account; wins over MISTRAL_API_KEY

# Your service
AGENT_ID_SUPPORT=ag_...           # created by a deploy script, not at request time
AGENT_VERSION_SUPPORT=prod        # version alias
MISTRAL_SERVER=eu                 # "eu", "us" or "global"
MISTRAL_TIMEOUT_MS=60000
MISTRAL_MAX_CONCURRENCY=16
```

Load and validate it once at start-up:

```python
import os
from pydantic import BaseModel

class Settings(BaseModel):
    agent_id_support: str
    agent_version_support: str = "prod"
    mistral_server: str = "eu"
    mistral_timeout_ms: int = 60_000
    mistral_max_concurrency: int = 16

settings = Settings(**{k.lower(): v for k, v in os.environ.items() if k.lower() in Settings.model_fields})
```

**Create agents in a deploy step, not per request.** A script (or CI job) creates or updates agents, sets a version alias such as `prod`, and writes the agent IDs into configuration. Requests then start conversations against `agent_version="prod"`, and a rollback is `update_version()` or moving the alias.

```python
# deploy_agents.py
import os
from mistralai.client import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

agent = client.beta.agents.update(
    agent_id=os.environ["AGENT_ID_SUPPORT"],
    instructions=open("prompts/support.md").read(),
    version_message=os.environ.get("GIT_SHA", "manual"),
)
client.beta.agents.create_version_alias(agent_id=agent.id, alias="prod", version=agent.version)
print("support agent now at version", agent.version)
```

---

## 2. Client, concurrency and retries

Create one client per process and reuse it; it pools HTTP connections. Bound concurrency explicitly.

```python
import asyncio
from mistralai.client import Mistral
from mistralai.client.utils import BackoffStrategy, RetryConfig

client = Mistral(
    server=settings.mistral_server,
    timeout_ms=settings.mistral_timeout_ms,
    retry_config=RetryConfig(
        "backoff",
        BackoffStrategy(initial_interval=500, max_interval=20_000, exponent=2.0, max_elapsed_time=90_000),
        retry_connection_errors=True,
    ),
)
limit = asyncio.Semaphore(settings.mistral_max_concurrency)

async def start_conversation(inputs: str, metadata: dict[str, str]):
    async with limit:
        return await client.beta.conversations.start_async(
            agent_id=settings.agent_id_support,
            agent_version=settings.agent_version_support,
            inputs=inputs,
            metadata=metadata,
        )
```

The SDK's retry policy retries connection errors and the status codes it considers transient. Retrying `start()` can create a second conversation if the first request actually succeeded; for idempotency, record the conversation ID before acting on the reply, and retry `append()` only after checking the history.

For a custom transport (proxies, custom TLS), pass `httpx2` clients; since v3 the SDK no longer uses `httpx`:

```python
import httpx2
from mistralai.client import Mistral

client = Mistral(
    client=httpx2.Client(timeout=60, limits=httpx2.Limits(max_connections=50)),
    async_client=httpx2.AsyncClient(timeout=60, limits=httpx2.Limits(max_connections=50)),
)
```

### Circuit breaker

When Mistral (or your network) is down, fail fast instead of piling up requests:

```python
import time
from mistralai.client import errors

class CircuitBreaker:
    def __init__(self, threshold: int = 5, reset_after: float = 30.0):
        self.threshold, self.reset_after = threshold, reset_after
        self.failures, self.opened_at = 0, 0.0

    def allow(self) -> bool:
        return self.failures < self.threshold or time.monotonic() - self.opened_at > self.reset_after

    def record(self, ok: bool) -> None:
        if ok:
            self.failures = 0
        else:
            self.failures += 1
            if self.failures == self.threshold:
                self.opened_at = time.monotonic()

breaker = CircuitBreaker()

async def guarded_start(inputs: str):
    if not breaker.allow():
        raise RuntimeError("Mistral temporarily unavailable")
    try:
        response = await start_conversation(inputs, metadata={})
    except errors.MistralError as e:
        breaker.record(ok=e.status_code < 500 and e.status_code != 429)
        raise
    breaker.record(ok=True)
    return response
```

---

## 3. Observability

Log one structured record per call. `response.usage` has token counts, including `connector_tokens` for built-in tools; `outputs` shows which tools ran and which agents handled the request.

```python
import json
import logging
import time

log = logging.getLogger("agents")

async def observed_start(inputs: str, user_id: str):
    started = time.monotonic()
    response = await start_conversation(inputs, metadata={"user_id": user_id})
    log.info(json.dumps({
        "event": "conversation.start",
        "conversation_id": response.conversation_id,
        "user_id": user_id,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "connector_tokens": response.usage.connector_tokens,
        "tools": [e.name for e in response.outputs if e.type == "tool.execution"],
        "handoffs": [e.next_agent_name for e in response.outputs if e.type == "agent.handoff"],
    }))
    return response
```

The SDK depends on `opentelemetry-api`; install `mistralai[telemetry]` to export its spans with the OTLP exporter alongside your service's own traces.

Alert on: error rate by status code, p95 latency, tokens per conversation (a rising trend usually means a prompt or tool regression), and 429s.

---

## 4. Storing conversation references

Mistral stores the conversation; you store who it belongs to.

```sql
CREATE TABLE agent_conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    agent_version   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    prompt_tokens   INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active'      -- active | closed | deleted
);
CREATE INDEX ON agent_conversations (user_id, last_used_at DESC);
```

Check ownership before every `append()`, so one user cannot continue another's conversation by guessing an ID. Use conversation `metadata` for search and filtering (`conversations.list(metadata=...)`), but keep authorisation in your own database. When a user deletes their data, call `client.beta.conversations.delete(conversation_id=...)` as well as removing your rows.

---

## 5. Rate limiting your users

Mistral enforces workspace limits; you should enforce per-user limits so one user cannot exhaust them:

```python
import time

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate, self.capacity = rate_per_sec, capacity
        self.tokens, self.updated = float(capacity), time.monotonic()

    def take(self, n: int = 1) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

buckets: dict[str, TokenBucket] = {}

def allow(user_id: str) -> bool:
    return buckets.setdefault(user_id, TokenBucket(rate_per_sec=0.2, capacity=5)).take()
```

In a multi-instance deployment, keep the buckets in Redis.

---

## 6. Performance

- **Stream** with `start_stream_async()` for user-facing requests; time to first token matters more than total time.
- **Choose models per agent.** Routers and classifiers rarely need more than `mistral-small-latest`.
- **Keep instructions stable** so the platform can cache the prompt; put per-request data in `inputs`, not `instructions`.
- **Trim tools.** Every tool definition costs prompt tokens on every turn.
- **Use `store=False`** for one-shot requests you will never append to.
- **Batch offline work** with the Batch API (`client.batch`) rather than conversations when no tools or state are involved.

---

## 7. Security

- **Secrets.** Prefer service-account tokens (`MISTRAL_SA_TOKEN_PATH`, re-read on every request so rotation needs no restart) over long-lived API keys. If the variable is set but the file is missing, every request fails with `ServiceAccountTokenError`; unset it rather than leaving it dangling.
- **Data residency.** Use `server="eu"` or `server="us"` to pin the region.
- **Guardrails.** Attach `GuardrailConfig` with `ModerationLlmv2Config` to public-facing agents.
- **Prompt injection.** Web search results, documents and connector outputs are untrusted. Use `requires_confirmation` on tools that write or send, and never put credentials in instructions.
- **Least privilege for connectors.** Scope credentials (`consumer_scope="user"` where possible) and expose only needed tools with `include`.
- **Access control.** Map your users to conversations in your database (section 4) and check it on every request.

---

## 8. Deployment

```dockerfile
FROM python:3.12-slim
RUN useradd --create-home app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt      # pins mistralai==%%version=mistralai%%
COPY --chown=app:app . .
USER app
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Because the heavy work happens at Mistral, instances are light: scale on concurrent requests and keep `MISTRAL_MAX_CONCURRENCY` × replicas below your workspace rate limit.

---

## 9. CI/CD

```yaml
name: agents
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt pytest
      - run: pytest -q                       # unit tests with the client mocked
  deploy-agents:
    needs: test
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: python deploy_agents.py         # creates a new agent version and moves the "prod" alias
        env:
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
          AGENT_ID_SUPPORT: ${{ vars.AGENT_ID_SUPPORT }}
          GIT_SHA: ${{ github.sha }}
```

Run a small evaluation set against the new agent version before moving the alias, and keep the previous version number so rollback is one call.

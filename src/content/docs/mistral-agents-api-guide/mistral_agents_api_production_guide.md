---
title: "Mistral Agents API: Production Deployment Guide"
description: "Running services on Mistral's Agents API in production: configuration, concurrency, retries, observability, storage, rate limits, security and CI/CD."
framework: "mistral-agents-api"
language: "python"
sidebar:
  label: "Production"
  order: 7
---

**Verified against `mistralai` %%version=mistralai%%** (checked %%versions-checked%%).

The agent loop, built-in tools and conversation storage run on Mistral's side, so your service is mostly an API client: it needs sensible concurrency, retries, observability, its own record of which conversations belong to whom, and careful handling of secrets.

---

## 1. Configuration

```bash
# Credentials (one of)
MISTRAL_API_KEY=...
MISTRAL_SA_TOKEN_PATH=/var/run/secrets/mistral/token   # Mistral service-account token file; wins over MISTRAL_API_KEY

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

### OpenTelemetry

The SDK can emit OpenTelemetry spans for its API calls (operation, model, token usage and, unless redacted, the messages and tool calls), but only once you opt in, with `configure_telemetry()` or the `MISTRAL_SDK_TELEMETRY` environment variable (`dedicated`, `global` or `false`). A tracer provider on its own is not enough. Install the extra first:

```bash
pip install "mistralai[telemetry]==%%version=mistralai%%"
```

To send the SDK's spans through your own pipeline, next to your service's spans, use `global` mode:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from mistralai.extra.observability import RedactingSpanExporter, configure_telemetry

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(
    RedactingSpanExporter(OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces"))
))
trace.set_tracer_provider(provider)

configure_telemetry(client, provider="global")   # SDK spans now use the global provider

tracer = trace.get_tracer("support-service")
with tracer.start_as_current_span("answer_support_question") as span:
    response = client.beta.conversations.start(agent_id=settings.agent_id_support, inputs="Where is ORD-1234?")
    span.set_attribute("conversation.id", response.conversation_id)
```

The SDK's own spans then appear as children of `answer_support_question`. In `global` mode (or when you pass your own `TracerProvider`) the SDK does not redact anything, which is why the exporter above is wrapped in `RedactingSpanExporter`; its default policy masks secrets and common personal data (emails, card-like numbers, IP addresses) in span attributes.

`configure_telemetry(client)` with no `provider` uses `dedicated` mode instead: the SDK creates its own exporter, redacts by default (`redaction=` takes a policy, a callback or `False`) and sends the spans to Mistral's telemetry endpoint, authenticated with the client's API key.

### Mistral's traces and evaluations

Spans sent in `dedicated` mode are stored by Mistral, and `client.beta.observability` (beta) works with them: `traces` and `spans` search and aggregate them, `judges` defines LLM judges that score conversations, `datasets` builds evaluation sets (including from spans or an uploaded file), and `evaluations` runs evaluation pipelines over them. This is one way to run the evaluation set mentioned under [CI/CD](#10-deployment-and-cicd) before moving a version alias; a plain pytest suite over a fixed set of prompts is another.

### What to alert on

Alert on: error rate by status code, p95 latency, tokens per conversation (a rising trend usually means a prompt or tool regression), and 429s.

---

## 4. Testing

Unit-test your code against scripted API responses rather than the live API. The client accepts any `httpx2` client, so an `httpx2.MockTransport` can return canned conversation entries, for example a `function.call` followed by the final `message.output`:

```python
# test_support.py
import json
import httpx2
from mistralai.client import Mistral
from mistralai.client.models import FunctionCallEntry, FunctionResultEntry

# --- code under test (normally imported from your service) ---
def get_order_status(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}

def answer_order_question(client: Mistral, agent_id: str, question: str, max_turns: int = 5) -> str:
    response = client.beta.conversations.start(agent_id=agent_id, inputs=question)
    for _ in range(max_turns):
        call = response.outputs[-1]
        if not isinstance(call, FunctionCallEntry):
            return response.outputs[-1].content
        args = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
        response = client.beta.conversations.append(
            conversation_id=response.conversation_id,
            inputs=[FunctionResultEntry(tool_call_id=call.tool_call_id, result=json.dumps(get_order_status(**args)))],
        )
    raise RuntimeError("too many tool calls")

# --- test helpers ---
NOW = "2026-10-21T00:00:00Z"

def entry(kind: str, **fields) -> dict:
    return {"object": "entry", "type": kind, "id": f"{kind}-1", "created_at": NOW, **fields}

def scripted_client(turns: list[list[dict]]) -> tuple[Mistral, list[dict]]:
    """A client whose conversation calls return `turns` in order; also returns the request bodies sent."""
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content or b"{}"))
        return httpx2.Response(200, json={
            "object": "conversation.response",
            "conversation_id": "conv_test",
            "outputs": turns.pop(0),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    return Mistral(api_key="test", client=httpx2.Client(transport=httpx2.MockTransport(handler))), sent

def test_order_question_runs_the_tool() -> None:
    client, sent = scripted_client([
        [entry("function.call", tool_call_id="tc1", name="get_order_status", arguments='{"order_id": "ORD-1234"}')],
        [entry("message.output", role="assistant", content="ORD-1234 has shipped.")],
    ])
    assert answer_order_question(client, "ag_test", "Where is ORD-1234?") == "ORD-1234 has shipped."
    assert sent[1]["inputs"][0]["tool_call_id"] == "tc1"
    assert json.loads(sent[1]["inputs"][0]["result"])["status"] == "shipped"
```

For async code pass `async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))` as well. Script handoffs (`agent.handoff` entries), pending approvals (`function.call` with `"confirmation_status": "pending"`) and errors (`httpx2.Response(429, ...)`) the same way. In TypeScript, pass a `fetcher` to `HTTPClient`; see [TypeScript production](/mistral-agents-api-guide/typescript/production/#testing).

Keep a small set of live tests as well, run against a staging agent version before you move the `prod` alias: mocks prove your loop handles the entries, not that the agent's instructions produce them.

---

## 5. Cost

- **Model tokens.** `usage.prompt_tokens` and `usage.completion_tokens` are billed at the agent's model rate. Routers, classifiers and extractors rarely need more than `mistral-small-latest`.
- **Tool tokens and calls.** `usage.connector_tokens` counts the tokens that tool results (search results, retrieved passages, code output) added to the context; they are included in `total_tokens`. `usage.connectors` counts calls per tool (for example `{"web_search": 1}`). Built-in tools are also priced per call (image generation per image) on top of tokens; check Mistral's pricing page.
- **History grows.** Stored conversations save you resending the transcript, but the model still reads the whole history on every `append()`, so prompt tokens rise with each turn. Start a new conversation, or `restart()` from an earlier entry, when the topic changes.
- **Cap the work.** Set `completion_args={"max_tokens": ...}` on agents, bound function-call loops, and give each agent only the tools it needs (every tool definition is sent on every turn).
- **Batch offline work.** The Batch API (`client.batch.jobs`) is priced below synchronous requests and suits bulk classification or extraction with no tools or conversation state.
- **Track it.** Log usage per conversation (section 3) and keep running totals per user (section 6), so a prompt or tool regression shows up as a cost trend before it shows up on the invoice.

---

## 6. Storing conversation references

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

## 7. Rate limiting your users

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

## 8. Performance

- **Stream** with `start_stream_async()` for user-facing requests; time to first token matters more than total time.
- **Choose models per agent.** Routers and classifiers rarely need more than `mistral-small-latest`.
- **Keep instructions stable** so the platform can cache the prompt; put per-request data in `inputs`, not `instructions`.
- **Trim tools.** Every tool definition costs prompt tokens on every turn.
- **Use `store=False`** for one-shot requests you will never append to.

---

## 9. Security

- **Secrets.** Prefer Mistral service-account tokens (`MISTRAL_SA_TOKEN_PATH`, re-read on every request so rotation needs no restart) over long-lived API keys. If the variable is set but the file is missing, every request fails with `ServiceAccountTokenError`; unset it rather than leaving it dangling.
- **Data residency.** Use `server="eu"` or `server="us"` to pin the region.
- **Guardrails.** Attach `GuardrailConfig` with `ModerationLlmv2Config` to public-facing agents.
- **Prompt injection.** Web search results, documents and connector outputs are untrusted. Use `requires_confirmation` on tools that write or send, and never put credentials in instructions.
- **Least privilege for connectors.** Scope credentials (`consumer_scope="user"` where possible) and expose only needed tools with `include`.
- **Access control.** Map your users to conversations in your database (section 6) and check it on every request.

---

## 10. Deployment and CI/CD

This is the one deployment set-up for the Python pages in this guide; the [FastAPI streaming server](/mistral-agents-api-guide/mistral_streaming_server_fastapi/) and a [chat-completions service](/mistral-agents-api-guide/mistral_agents_api_advanced_python/) deploy the same way.

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

On Kubernetes, mount the Mistral service-account token from a Secret as a file and point `MISTRAL_SA_TOKEN_PATH` at it:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistral-agent
spec:
  replicas: 2
  selector:
    matchLabels: { app: mistral-agent }
  template:
    metadata:
      labels: { app: mistral-agent }
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/mistral-agent:1.0.0
          env:
            - name: MISTRAL_SA_TOKEN_PATH
              value: /var/run/secrets/mistral/token
            - name: AGENT_ID_SUPPORT
              valueFrom: { configMapKeyRef: { name: mistral-config, key: agentIdSupport } }
          volumeMounts:
            - { name: mistral-token, mountPath: /var/run/secrets/mistral, readOnly: true }
          ports: [{ containerPort: 8000 }]
          readinessProbe:
            httpGet: { path: /healthz, port: 8000 }
      volumes:
        - name: mistral-token
          secret:
            secretName: mistral-sa-token          # holds the Mistral token under the key "token"
            items: [{ key: token, path: token }]
```

The token is a Mistral service-account token (see `client.beta.service_accounts`), not the pod's own Kubernetes service-account token, which Mistral does not accept. Kubernetes updates a mounted Secret in place (unless it is mounted with `subPath`), and the SDK re-reads the file on each request, so rotating the Secret needs no restart. With an API key instead, set `MISTRAL_API_KEY` from a `secretKeyRef`.

Because the heavy work happens at Mistral, instances are light: scale on concurrent requests and keep `MISTRAL_MAX_CONCURRENCY` × replicas below your workspace rate limit.

CI runs the mocked tests from section 4, then creates the new agent version and moves the alias:

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
      - run: pytest -q                       # the mocked tests from section 4
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

Run a small evaluation set against the new agent version before moving the alias, and keep the previous version number so rollback is one call. Build and push the container image in the same workflow with your usual Docker actions.

---
title: "Microsoft Agent Framework Python - Production Guide"
description: "This guide provides enterprise-level best practices for deploying, managing, and scaling Python applications built with the Microsoft Agent Framework."
framework: microsoft-agent-framework
language: python
---

# Microsoft Agent Framework Python - Production Guide

This guide provides enterprise-level best practices for deploying, managing, and scaling Python applications built with the Microsoft Agent Framework.

**Target Audience:** DevOps Engineers, Infrastructure Architects, Senior Python Developers
**Platform:** Python 3.10+

---

## 1. Production Deployment

### Recommended Host: Azure Container Apps

For most scenarios, **Azure Container Apps (ACA)** provides the best balance of scalability, ease of use, and power for hosting agent applications. It offers serverless containers, built-in Dapr integration for microservices, and KEDA-based scaling.

### Strategy 1: Docker & Azure Container Apps

**Step 1: Dockerize Your Application**

Create a `Dockerfile` in your project root. We recommend using a lightweight base image.

```dockerfile
# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if needed (e.g., build tools)
# RUN apt-get update && apt-get install -y build-essential

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the port
EXPOSE 8000

# Run with Gunicorn/Uvicorn for production performance
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
```

**Step 2: Build and Push to Azure Container Registry (ACR)**

```bash
# Log in to Azure
az login

# Create an Azure Container Registry
az acr create --resource-group MyResourceGroup --name myagentregistry --sku Basic

# Build and push the image
az acr build --registry myagentregistry --image my-python-agent:v1 .
```

**Step 3: Deploy to Azure Container Apps**

Use the Azure CLI or Bicep.

```bash
az containerapp create \
  --name my-agent-service \
  --resource-group MyResourceGroup \
  --environment my-aca-env \
  --image myagentregistry.azurecr.io/my-python-agent:v1 \
  --target-port 8000 \
  --ingress 'external' \
  --registry-server myagentregistry.azurecr.io \
  --min-replicas 1 \
  --max-replicas 10 \
  --cpu 1.0 --memory 2.0Gi
```

### Strategy 2: CI/CD with GitHub Actions

Automate your deployment process.

```yaml
# .github/workflows/deploy.yml
name: Deploy Python Agent to ACA

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: 'Az CLI login'
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}

    - name: 'Build and push image'
      run: |
        az acr build --registry myagentregistry --image my-python-agent:${{ github.sha }} .

    - name: 'Deploy to Azure Container Apps'
      uses: azure/container-apps-deploy-action@v1
      with:
        imageToDeploy: myagentregistry.azurecr.io/my-python-agent:${{ github.sha }}
        containerAppName: my-agent-service
        resourceGroup: MyResourceGroup
```

---

## 2. Scaling Strategies

### Horizontal Scaling (Scale Out)

-   **Stateless Agents:** Design your agents to be stateless. Persist conversation history to Cosmos DB or Redis.
-   **KEDA-Based Autoscaling:** Use KEDA scalers in ACA.
    -   **HTTP Scaler:** Scale based on incoming request rate.
    -   **Azure Queue Storage Scaler:** Scale based on the number of messages in a queue (for async background workers).

### Caching Strategies

-   **LLM Response Caching:** Cache identical prompts to save costs.
-   **Redis:** Use `redis-py` for high-performance caching.

```python
# caching.py
import redis.asyncio as redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

async def get_cached_response(prompt_hash: str):
    val = await r.get(prompt_hash)
    return json.loads(val) if val else None

async def cache_response(prompt_hash: str, response: dict):
    await r.set(prompt_hash, json.dumps(response), ex=3600) # Expire in 1 hour
```

---

## 3. Monitoring & Observability

### OpenTelemetry Integration

The Microsoft Agent Framework for Python has native support for OpenTelemetry.

**1. Install Packages:**

```bash
pip install opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-azure-monitor \
    opentelemetry-instrumentation-aiohttp
```

**2. Configure Tracing:**

```python
# telemetry.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
import os

def configure_telemetry():
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Export to Azure Monitor (Application Insights)
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    exporter = AzureMonitorTraceExporter(connection_string=connection_string)
    span_processor = BatchSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer
```

### Structured Logging

Use the standard `logging` module, but configure it to output JSON or integrate with Azure Monitor.

```python
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string='InstrumentationKey=...'))
logger.setLevel(logging.INFO)

logger.info("Processing request", extra={"custom_dimension": "value"})
```

---

## 4. Security Best Practices

### Secrets Management

Use **Azure Key Vault** to store API keys and connection strings.

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

key_vault_url = os.getenv("KEY_VAULT_URL")
credential = DefaultAzureCredential()
client = SecretClient(vault_url=key_vault_url, credential=credential)

secret = client.get_secret("OpenAIKey")
openai_key = secret.value
```

### Authentication

-   **Managed Identity:** Always use `DefaultAzureCredential` in production. It automatically handles authentication for Azure services without hardcoded credentials.
-   **Network Security:** Deploy ACA in a VNet to isolate traffic.

---

## 5. High Availability & Disaster Recovery

-   **Multi-Region:** Deploy to paired regions (e.g., East US 2 and Central US).
-   **Traffic Manager:** Use Azure Front Door to route traffic.
-   **Resiliency:** Use libraries like `tenacity` for retries.

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_openai_with_retry():
    # Make LLM call
    pass
```

---

## 6. Cost Optimization

-   **Model Selection:** Use `gpt-4o-mini` for routing and simple tasks.
-   **Token Monitoring:** Log token usage metadata from responses.
-   **Budget Alerts:** Set up Azure Cost Management alerts.

---

## 7. Performance Tuning

-   **AsyncIO:** Python's `asyncio` is critical for I/O-bound agent workloads. Avoid blocking calls (`time.sleep`, synchronous requests) in the main event loop.
-   **Gunicorn Configuration:** Adjust the number of workers based on CPU cores (`2 * cores + 1`).
-   **Connection Pooling:** Reuse `aiohttp.ClientSession` instances instead of creating a new one for every request.

```python
# Efficient session management
import aiohttp

session = None

async def get_session():
    global session
    if session is None:
        session = aiohttp.ClientSession()
    return session
```

---

## 8. Local Interactive Testing Before Deploying

Before pushing to ACA, smoke-test the agent against a real model using `agent-framework-devui`. It's a built-in FastAPI/Uvicorn server (re-exported as `agent_framework.devui`) that exposes any agent or workflow over an OpenAI-compatible HTTP API plus a web UI.

```python
# devui_app.py
from agent_framework import Agent
from agent_framework.devui import serve
from agent_framework.foundry import FoundryChatClient


def build_agent() -> Agent:
    return Agent(
        client=FoundryChatClient(),                    # picks up FOUNDRY_PROJECT_ENDPOINT / FOUNDRY_MODEL
        instructions="You are an internal support agent.",
    )


if __name__ == "__main__":
    serve(
        entities=[build_agent()],
        host="127.0.0.1",
        port=8080,
        auto_open=True,
        instrumentation_enabled=True,                  # surfaces OpenTelemetry traces in the UI
    )
```

Hardening before sharing the URL with a teammate:

- Use `mode="user"` instead of the default `"developer"` — generic error messages, admin endpoints disabled.
- `auth_enabled=True` enforces a Bearer token on every request. Pass `DEVUI_AUTH_TOKEN` in the environment to keep the same token across restarts; otherwise the server auto-generates one and prints it on startup.
- Network-exposing the server (`host="0.0.0.0"`) without auth raises a loud startup warning — DevUI is a development harness, not a production runtime.

For directory-style discovery (one folder per agent), use `serve(entities_dir="./agents", ...)` or run `devui ./agents --port 8080 --auth` from the CLI.

---

## 9. Choosing the Right Chat Client for Production

Three first-party clients dominate Azure deployments:

| Client | Best for | Notes |
|---|---|---|
| `OpenAIChatClient` (Azure mode) | Standard Azure OpenAI deployments — direct endpoint + key/Entra | Smallest dependency footprint; reads `AZURE_OPENAI_*` env vars |
| `agent_framework.foundry.FoundryChatClient` | Microsoft Foundry projects — model deployment, evaluation, private networking | Reads `FOUNDRY_PROJECT_ENDPOINT` / `FOUNDRY_MODEL`; supports `AIProjectClient` reuse for OAuth helpers |
| `agent_framework.foundry.FoundryAgent` | Service-managed agents — identity, threads, tool definitions live in Foundry | Use when you want the agent to outlive your process / be visible in the Foundry portal |

`FoundryChatClient` constructor accepts either `(project_endpoint=, model=, credential=)` or a pre-built `project_client=AIProjectClient(...)`. The latter is the right choice when your service already shares an `AIProjectClient` for evaluations or memory:

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

project = AIProjectClient(
    endpoint="https://corp-foundry.services.ai.azure.com",
    credential=DefaultAzureCredential(),
)

client = FoundryChatClient(project_client=project, model="gpt-4o-mini")
agent = Agent(client=client, instructions="...")
```

All three clients implement the same `SupportsChatGetResponse` protocol, so you can swap them without touching agent / tool / middleware code.

---

## 10. Operational patterns specific to `agent_framework`

### Reuse one `Agent` per process

`Agent` is cheap to call but moderately expensive to construct (chat client, middleware pipelines, context provider registration). Build one at startup and pass it through dependency injection:

```python
# app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from agent_framework import Agent, InMemoryHistoryProvider
from agent_framework.foundry import FoundryChatClient


_agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = Agent(
        client=FoundryChatClient(),
        instructions="You are the customer-support agent.",
        context_providers=[InMemoryHistoryProvider()],
    )
    yield
    # Nothing to dispose — the chat client cleans up its httpx pools on GC.


app = FastAPI(lifespan=lifespan)


def get_agent() -> Agent:
    assert _agent is not None
    return _agent


@app.post("/chat")
async def chat(message: str, agent: Agent = Depends(get_agent)):
    return await agent.run(message)
```

Per-request state (conversation history, user context) lives in the `AgentSession`, not the agent — create a session per user/conversation and pass `session=` on each `agent.run(...)`.

### Per-tenant scoping

For SaaS deployments, scope per-tenant data via `additional_properties` on the run call rather than per-tenant agent instances:

```python
await agent.run(
    user_message,
    session=session,
    additional_properties={"tenant_id": tenant_id, "gen_ai.conversation.id": correlation_id},
    function_invocation_kwargs={"tenant_id": tenant_id},   # forwarded to tools that opt in via **kwargs
)
```

Tools and context providers that declare `**kwargs` see the runtime data; `additional_properties` propagate to telemetry spans for trace correlation. One agent instance, N tenants — no extra memory footprint.

### Graceful timeouts on workflows

`Workflow.run(...)` doesn't enforce a wall-clock deadline. Wrap it in `asyncio.wait_for` so a stuck tool or runaway Magentic loop doesn't tie up your worker pool:

```python
import asyncio
import logging
from agent_framework import CheckpointStorage, Workflow

logger = logging.getLogger(__name__)


async def safe_run(
    workflow: Workflow,
    user_input: str,
    *,
    storage: CheckpointStorage,
    timeout_s: float = 60.0,
):
    try:
        return await asyncio.wait_for(workflow.run(user_input), timeout=timeout_s)
    except asyncio.TimeoutError:
        # Save the latest checkpoint so a human can resume manually.
        latest = await storage.get_latest(workflow_name=workflow.name)
        logger.warning(
            "workflow_timeout checkpoint_id=%s",
            latest.checkpoint_id if latest else None,
        )
        raise
```

Combine with `MagenticBuilder(max_round_count=20, max_stall_count=3)` so the orchestrator gives up internally before you ever hit the outer timeout.

### Concurrency limits

Front-line `OpenAIChatClient` / `FoundryChatClient` traffic is rate-limited by your model deployment. Add a semaphore at the agent boundary so a flood of requests doesn't all hit the model at once:

```python
import asyncio
from agent_framework import AgentMiddleware, AgentContext


class ConcurrencyLimit(AgentMiddleware):
    def __init__(self, max_concurrent: int) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)

    async def process(self, context: AgentContext, call_next) -> None:
        async with self._sem:
            await call_next()


agent = Agent(
    client=FoundryChatClient(),
    instructions="…",
    middleware=[ConcurrencyLimit(max_concurrent=8)],
)
```

For workflow-level limits put the semaphore in your handler instead of around `agent.run` — workflow supersteps execute concurrently, so you want one semaphore per executor type.

### Health checks that don't burn tokens

A liveness probe that calls `agent.run("ping")` costs real money on every probe interval. Keep `/healthz` as a lightweight local liveness signal — it answers "is this Python process still running?" — and let the orchestrator's network probe answer "is the upstream model up?":

```python
@app.get("/healthz")
async def healthz(agent: Agent = Depends(get_agent)):
    # Liveness only — does NOT touch the chat client. The process is up if this returns.
    return {"status": "ok", "agent": agent.name, "version": agent.id}
```

If you genuinely need a readiness probe that proves the model deployment is reachable, run a separate `/readyz` that issues a token-free call (e.g. an Azure OpenAI `models` listing or an `AIProjectClient.connections.list()` call) on startup, caches the result for 60 seconds, and returns 503 until the probe succeeds. Never make the model probe synchronous on every probe — the cost adds up fast at K8s default 10-second intervals.

### Checkpoint hygiene

`FileCheckpointStorage` and Cosmos / Redis backends accumulate over time. Add a scheduled task that prunes old checkpoints per workflow:

```python
async def prune_checkpoints(storage, *, workflow_name: str, keep: int = 50) -> int:
    ids = await storage.list_checkpoint_ids(workflow_name=workflow_name)
    deleted = 0
    for old in ids[:-keep]:
        if await storage.delete(old):
            deleted += 1
    return deleted
```

Run it nightly per workflow name. Aim for `keep` ≥ longest plausible HITL pause (don't auto-delete checkpoints with `pending_request_info_events`).

### Circuit breaker for upstream model outages

When the model API is degraded, fail fast rather than waiting for every request to time out:

```python
import asyncio
import time
from agent_framework import ChatMiddleware, ChatContext, MiddlewareTermination


class ModelCircuitBreaker(ChatMiddleware):
    """A simple async-safe circuit breaker.

    `ChatMiddleware.process` runs concurrently across requests, so the failure
    counter and open-until timestamp are guarded by an `asyncio.Lock`. The lock
    is only held while reading/mutating state — never around `call_next()` —
    so it does not serialise model traffic.
    """

    def __init__(self, *, fail_threshold: int = 5, recover_seconds: float = 30.0) -> None:
        self._fail_count = 0
        self._open_until: float = 0.0
        self._fail_threshold = fail_threshold
        self._recover_seconds = recover_seconds
        self._lock = asyncio.Lock()

    async def process(self, context: ChatContext, call_next) -> None:
        async with self._lock:
            if time.time() < self._open_until:
                raise MiddlewareTermination("model unavailable — circuit open")

        try:
            await call_next()
        except Exception:
            async with self._lock:
                self._fail_count += 1
                if self._fail_count >= self._fail_threshold:
                    self._open_until = time.time() + self._recover_seconds
            raise
        else:
            async with self._lock:
                self._fail_count = 0
```

Pair with a retry middleware that catches `MiddlewareTermination("model unavailable...")` and routes to a fallback agent (or a static answer). Two layers, single circuit-breaker source of truth.

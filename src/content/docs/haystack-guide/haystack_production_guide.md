---
title: "Haystack Production Deployment and Operations Guide"
description: "Deploying Haystack 3 in production — readiness checks, serialised pipelines, Docker, Kubernetes, FastAPI services, scaling and caching."
framework: "haystack"
language: "python"
---

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%).

## Table of Contents

1. [Production Readiness Checklist](#production-readiness-checklist)
2. [Deployment Strategies](#deployment-strategies)
3. [Containerisation and Docker](#containerisation-and-docker)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [API Service Development](#api-service-development)
6. [Scaling Strategies](#scaling-strategies)
7. [Caching and Performance](#caching-and-performance)

For tracing, logging and error-recovery patterns, continue in [Observability](/haystack-guide/haystack_observability_python/); for resilience inside the agent loop, see [Error Handling and Resilience](/haystack-guide/haystack_comprehensive_guide/#error-handling-and-resilience).

---

## Production Readiness Checklist

### Before you deploy

| Area | Check |
|------|-------|
| Versions | Pin `haystack-ai` and every integration package; Haystack 3 integrations are released independently of the core |
| Secrets | API keys passed as `Secret.from_env_var(...)`, never as strings in code or YAML |
| Start-up | `pipeline.warm_up()` at start-up so missing keys and unreachable stores fail fast (from 3.0, keys are resolved at warm-up, not construction) |
| Shutdown | `pipeline.close()` / `await pipeline.close_async()` to release HTTP clients and connections |
| Serialisation | Pipelines loaded from YAML use an explicit `allowed_modules` list, never `unsafe=True` for files you did not write |
| Prompts | Template variables are required by default; integration tests cover every input path |
| Agents | `max_agent_steps` set deliberately; `exit_reason` logged; tools with side effects behind a confirmation hook |
| Tracing | A tracer is enabled explicitly (nothing is auto-enabled in 3.x); content tracing only where PII rules allow |
| Logging | Decide whether Haystack's handler should propagate to your root logger (see [Observability](/haystack-guide/haystack_observability_python/#logging)) |
| Data | If you index with auto-generated IDs and upgraded from 2.x, re-index documents with non-empty `meta` (their IDs changed) |

### A start-up check

```python
import logging
import os

from haystack import Pipeline

logger = logging.getLogger(__name__)

REQUIRED_ENV = ["OPENAI_API_KEY"]


def check_ready(pipeline: Pipeline) -> None:
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    # Creates API clients and connects to stores; raises on bad credentials or config
    pipeline.warm_up()

    # Connections are type-checked by connect(); log the entry points and outputs for the record
    logger.info("Pipeline ready: inputs=%s outputs=%s", pipeline.inputs(), pipeline.outputs())
```

For a deeper smoke test, run the pipeline once with `MockChatGenerator` substituted for the real generator in CI: it exercises every connection and prompt template without network calls.

---

## Deployment Strategies

### Pipelines as configuration

Build pipelines in code, serialise them to YAML, and ship the YAML with the release. The same artefact then runs in every environment:

```python
from haystack import Pipeline

# build.py — run in CI
pipeline = build_rag_pipeline()          # your factory function
with open("pipelines/rag.yaml", "w") as fp:
    pipeline.dump(fp)
```

```python
# service start-up
from haystack import Pipeline

with open("pipelines/rag.yaml") as fp:
    pipeline = Pipeline.load(fp, allowed_modules=["myapp.components.*"])
```

Components from `haystack`, `haystack_integrations` and `haystack_experimental` are trusted by default. Your own components and tool functions must be allowed per call (as above), process-wide with `haystack.core.serialization.allow_deserialization_module("myapp.*")`, or through the `HAYSTACK_DESERIALIZATION_ALLOWLIST` environment variable.

### Blue-green and canary releases

Haystack services are stateless (conversation history and documents live in external stores), so standard Kubernetes strategies apply unchanged:

- **Blue-green**: run two Deployments (`haystack-blue`, `haystack-green`) and switch the Service selector once the new one passes smoke tests.
- **Canary**: route a small percentage of traffic to the new version with your ingress or service mesh (for example NGINX `canary-weight`, Istio `VirtualService` weights), and compare error rate, latency, `exit_reason` distribution and token usage before promoting.

Treat a model change, a prompt change and a Haystack upgrade as three separate releases; each can change behaviour on its own.

---

## Containerisation and Docker

### Production Dockerfile

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN useradd -m -u 1000 haystack
WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY --chown=haystack:haystack . .
USER haystack

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt`:

```text
haystack-ai==%%version=haystack-ai%%
elasticsearch-haystack        # pin the versions you tested
opentelemetry-haystack
fastapi
uvicorn[standard]
redis
```

### Docker Compose for Development

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ELASTICSEARCH_URL: http://elasticsearch:9200
      REDIS_URL: redis://redis:6379/0
    depends_on: [elasticsearch, redis]

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.15.0
    environment:
      discovery.type: single-node
      xpack.security.enabled: "false"
      ES_JAVA_OPTS: -Xms1g -Xmx1g
    ports: ["9200:9200"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

---

## Kubernetes Deployment

### Kubernetes Manifests

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: haystack
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: haystack-config
  namespace: haystack
data:
  ELASTICSEARCH_URL: http://elasticsearch:9200
  REDIS_URL: redis://redis:6379/0
  HAYSTACK_DESERIALIZATION_ALLOWLIST: "myapp.*"
---
# Create with: kubectl -n haystack create secret generic haystack-secrets --from-literal=OPENAI_API_KEY=...
apiVersion: apps/v1
kind: Deployment
metadata:
  name: haystack-api
  namespace: haystack
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
  selector:
    matchLabels: { app: haystack-api }
  template:
    metadata:
      labels: { app: haystack-api }
    spec:
      securityContext: { runAsNonRoot: true, runAsUser: 1000 }
      containers:
        - name: api
          image: registry.example.com/haystack-api:1.4.2   # your image tag
          ports: [{ containerPort: 8000 }]
          envFrom:
            - configMapRef: { name: haystack-config }
            - secretRef: { name: haystack-secrets }
          resources:
            requests: { cpu: 500m, memory: 1Gi }
            limits: { cpu: "2", memory: 2Gi }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 30
            periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: haystack-api
  namespace: haystack
spec:
  selector: { app: haystack-api }
  ports: [{ port: 80, targetPort: 8000 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: haystack-api
  namespace: haystack
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: haystack-api }
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
```

LLM-bound services spend most of their time waiting on the network, so CPU is a weak scaling signal. If you can, scale on in-flight requests or request latency via a custom metric instead.

---

## API Service Development

### FastAPI Integration

Build the agent once at start-up, warm it up, and call `run_async` per request so one worker can serve many concurrent requests.

```python
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class AgentResponse(BaseModel):
    answer: str
    exit_reason: str | None
    steps: int
    tool_calls: dict[str, int]
    total_tokens: int | None
    seconds: float


agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%", timeout=60, max_retries=2),
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        max_agent_steps=10,
    )
    agent.warm_up()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=AgentResponse)
async def query(q: Query) -> AgentResponse:
    start = time.perf_counter()
    try:
        result = await agent.run_async(messages=[ChatMessage.from_user(q.text)])
    except Exception:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=502, detail="Upstream model error")
    return AgentResponse(
        answer=result["last_message"].text or "",
        exit_reason=result.get("exit_reason"),
        steps=result["step_count"],
        tool_calls=result["tool_call_counts"],
        total_tokens=result["token_usage"].get("total_tokens"),
        seconds=round(time.perf_counter() - start, 3),
    )
```

For token streaming over server-sent events, see the [streaming server](/haystack-guide/haystack_streaming_server_fastapi/) page.

deepset's open-source **Hayhooks** project (`pip install hayhooks`) can serve pipelines and agents as REST and OpenAI-compatible endpoints without writing the FastAPI layer yourself; consult its documentation for the current deployment format.

---

## Scaling Strategies

### Horizontal Scaling with Load Balancing

- **Keep workers stateless.** Store conversation history (`ChatMessage.to_dict()`), user state and documents outside the process so any replica can serve any request.
- **Prefer async.** `Agent.run_async` and `Pipeline.run_async` let one Uvicorn worker overlap many LLM calls. Use a few workers per pod (`--workers 2-4`) for CPU-bound pre-processing.
- **Bound concurrency.** `Pipeline.run_async(..., concurrency_limit=N)` limits parallel components per request; `tool_concurrency_limit` limits parallel tool calls per agent step. Add a semaphore at the API layer to protect provider rate limits.
- **Separate indexing from querying.** Run indexing pipelines as batch jobs (Kubernetes `Job`/`CronJob`) so ingestion spikes do not affect query latency.
- **Spread provider risk.** `FallbackChatGenerator` fails over between providers or regions.

```python
import asyncio

MAX_IN_FLIGHT = asyncio.Semaphore(32)   # per process

async def run_limited(messages):
    async with MAX_IN_FLIGHT:
        return await agent.run_async(messages=messages)
```

---

## Caching and Performance

### Multi-Layer Caching Strategy

1. **Response cache** for repeated identical questions (Redis, short TTL).
2. **Embedding / indexing cache** so unchanged documents are not re-embedded: `CacheChecker` looks up documents by a metadata field and splits inputs into hits and misses.
3. **Provider-side prompt caching**: keep long system prompts and tool schemas stable and at the start of the prompt so providers can reuse them.

```python
import hashlib
import json

from redis.asyncio import Redis

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
TTL_SECONDS = 3600


def cache_key(text: str, model: str = "%%model=openai-small%%", prompt_version: str = "v3") -> str:
    digest = hashlib.sha256(f"{model}|{prompt_version}|{text.strip().lower()}".encode()).hexdigest()
    return f"answer:{digest}"


async def cached_answer(text: str) -> dict:
    key = cache_key(text)
    if (hit := await redis.get(key)) is not None:
        return {**json.loads(hit), "from_cache": True}

    result = await run_limited([ChatMessage.from_user(text)])
    payload = {"answer": result["last_message"].text, "exit_reason": result.get("exit_reason")}
    if result.get("exit_reason") == "text":          # only cache clean completions
        await redis.set(key, json.dumps(payload), ex=TTL_SECONDS)
    return {**payload, "from_cache": False}
```

Include the model name and a prompt version in the key so a deployment that changes either does not serve stale answers. Do not cache answers that depend on user-specific tool results.

Skipping already-indexed files with `CacheChecker`:

```python
from haystack import Document
from haystack.components.caching import CacheChecker
from haystack.document_stores.in_memory import InMemoryDocumentStore

store = InMemoryDocumentStore()
store.write_documents([Document(content="old", meta={"url": "https://example.com/a"})])

checker = CacheChecker(document_store=store, cache_field="url")
result = checker.run(items=["https://example.com/a", "https://example.com/b"])
print(result["misses"])   # ['https://example.com/b'] — only these need fetching and indexing
```

---

**Not covered here.** Security hardening, rate limiting, database operations, governance and disaster recovery depend on your platform more than on Haystack. For tracing and logging see [Observability](/haystack-guide/haystack_observability_python/).

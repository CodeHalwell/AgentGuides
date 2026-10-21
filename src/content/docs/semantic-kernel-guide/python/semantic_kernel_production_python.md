---
title: "Semantic Kernel Production Guide (Python)"
description: "A production-shaped FastAPI service for Semantic Kernel in Python: shared AI service with a kernel per request, OpenTelemetry and token accounting, retries and timeouts, a caching filter and offline tests."
framework: "semantic-kernel"
language: "python"
sidebar:
  label: "Production"
  order: 6
---

> Targets `semantic-kernel` %%version=semantic-kernel%% on Python 3.10+ (checked %%versions-checked%%). The modules below were run together: the tests pass against a stub chat service, and the token counters were read back from an in-memory OpenTelemetry reader.

This page is the Python implementation of the [production guide](/semantic-kernel-guide/semantic_kernel_production_guide/), which covers the reasoning (deployment shape, security, scaling, cost) and the release checklist. Here: a small FastAPI service you can copy, with each Semantic Kernel-specific decision explained.

```bash
pip install "semantic-kernel==%%version=semantic-kernel%%" fastapi "uvicorn[standard]" azure-monitor-opentelemetry
pip install pytest httpx      # for the tests
```

Semantic Kernel already depends on `openai`, `pydantic`, `azure-identity` and the OpenTelemetry API and SDK; let it choose their versions and lock the result (`uv lock`, `pip-compile`).

```
app.py          FastAPI app: shared service, kernel per request
settings.py     configuration from the environment
telemetry.py    exporter set-up and a token-usage filter
cache.py        optional exact-match cache filter
tests/test_app.py
```

---

## 1. Settings

```python
# settings.py
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    azure_openai_endpoint: str
    azure_openai_deployment: str
    appinsights_connection_string: str | None
    request_timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_openai_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            appinsights_connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
        )
```

Secrets do not appear here: the service authenticates to Azure OpenAI with managed identity, and the Application Insights connection string is injected by the platform (Container Apps, App Service or a Kubernetes secret).

---

## 2. The app: one service, one kernel per request

```python
# app.py
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from fastapi import Depends, FastAPI, HTTPException, Request
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.exceptions import KernelInvokeException
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt

from settings import Settings
from telemetry import register_filters, setup_telemetry

logger = logging.getLogger("app")

# Prompt functions are immutable and can be shared; plugins and filters are added per kernel.
REPLY = KernelFunctionFromPrompt(
    function_name="reply",
    plugin_name="chat",
    prompt="You are a concise support assistant.\n\nCustomer: {{$input}}",
)


def create_chat_service(settings: Settings) -> AzureChatCompletion:
    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_ad_token_provider=get_bearer_token_provider(   # managed identity in Azure, `az login` locally
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"),
        api_version="2024-10-21",
        max_retries=2,      # the OpenAI SDK retries 429/5xx with back-off; don't add a second layer
        timeout=30.0,       # per attempt
    )
    return AzureChatCompletion(deployment_name=settings.azure_openai_deployment, async_client=client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    setup_telemetry(settings.appinsights_connection_string)
    app.state.settings = settings
    # One AI service per process: it holds the HTTP connection pool and is safe to share.
    app.state.chat_service = create_chat_service(settings)
    yield


app = FastAPI(lifespan=lifespan)


def get_kernel(request: Request) -> Kernel:
    """A fresh kernel per request, built around the shared service."""
    kernel = Kernel()
    kernel.add_service(request.app.state.chat_service)
    register_filters(kernel)
    return kernel


class ChatRequest(BaseModel):
    message: str
    tenant: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}                           # liveness: never calls the model


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request, kernel: Annotated[Kernel, Depends(get_kernel)]):
    try:
        result = await asyncio.wait_for(
            kernel.invoke(REPLY, KernelArguments(input=body.message, tenant=body.tenant)),
            timeout=request.app.state.settings.request_timeout_s,
        )
    except (KernelInvokeException, asyncio.TimeoutError):
        logger.exception("chat failed")
        raise HTTPException(status_code=503, detail="The assistant is unavailable.")   # no exception text
    return ChatResponse(reply=str(result))
```

What matters here:

- **The AI service is created once**, in the lifespan handler, and kept on `app.state`. It owns the HTTP connection pool and the token provider, and is safe to share.
- **The kernel is built per request** in `get_kernel`. A `Kernel` is a cheap container; building one per request means plugins or filters that a request adds (a tenant's tools, a per-user approval filter) cannot leak into another request. This is the same rule as registering `Kernel` as transient in .NET.
- **Retries happen once, in the OpenAI SDK.** `max_retries` and `timeout` on the `AsyncAzureOpenAI` client control them; the request-level `asyncio.wait_for` bounds the whole call, including any tool-calling round trips.
- **Errors are logged, not returned.** `KernelInvokeException` wraps the connector's `ServiceResponseException`; the client sees a generic 503.

For agents, build the `ChatCompletionAgent` inside the handler in the same way (`ChatCompletionAgent(kernel=kernel, ...)`): construction is cheap, and the conversation state belongs in a thread you load from a store, not in the agent object. The [streaming server](/semantic-kernel-guide/python/semantic_kernel_streaming_server_fastapi/) shows that shape with SSE.

---

## 3. Telemetry and token usage

```python
# telemetry.py
from opentelemetry import metrics

from semantic_kernel.filters import FilterTypes, FunctionInvocationContext

meter = metrics.get_meter("app.sk")
prompt_tokens = meter.create_counter("app.llm.tokens.prompt", unit="{token}")
completion_tokens = meter.create_counter("app.llm.tokens.completion", unit="{token}")


def setup_telemetry(connection_string: str | None) -> None:
    """Export SK's spans, metrics and logs. Call once, before the first request."""
    if connection_string:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=connection_string, logger_name="semantic_kernel")


async def record_token_usage(context: FunctionInvocationContext, next) -> None:
    """Function-invocation filter: count tokens per function and tenant.

    Python SK has no token-usage metric of its own, so read the usage the
    connector attaches to each completion's metadata.
    """
    await next(context)
    if context.result is None:
        return
    tenant = str((context.arguments or {}).get("tenant", "unknown"))
    attrs = {"function": context.function.fully_qualified_name, "tenant": tenant}
    for completion in context.result.metadata.get("metadata", []):
        usage = completion.get("usage")
        if usage is not None:
            prompt_tokens.add(usage.prompt_tokens or 0, attrs)
            completion_tokens.add(usage.completion_tokens or 0, attrs)


def register_filters(kernel) -> None:
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, record_token_usage)
```

Semantic Kernel instruments itself with OpenTelemetry under the `semantic_kernel` namespace. In Python that means:

- **Spans** for each function invocation and, with `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true` set in the environment, GenAI semantic-convention spans for model calls, including the `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` attributes. The `..._SENSITIVE=true` variant also records prompts and completions; leave it off in production.
- **Metrics**: only `semantic_kernel.function.invocation.duration` and `semantic_kernel.function.streaming.duration` histograms. The token-usage metrics (`semantic_kernel.function.invocation.token_usage.*`) exist only in the .NET SDK.

So token accounting in Python is yours to record. The filter above reads the `usage` (a `CompletionUsage`) that the OpenAI connectors attach to every completion's metadata and adds it to counters with function and tenant attributes. Agents call the chat service directly, not through a kernel function, so for agent turns read `response.message.metadata["usage"]` instead (the [resilient agent recipe](/semantic-kernel-guide/python/semantic_kernel_recipes_python/#6-resilient-agent-with-monitoring) does).

---

## 4. Caching deterministic functions

```python
# cache.py
import hashlib
import json

from semantic_kernel.filters import FunctionInvocationContext
from semantic_kernel.functions import FunctionResult

CACHEABLE = {"text-classify"}            # deterministic functions only
_cache: dict[str, str] = {}              # use Redis or similar when you run several replicas


async def cache_filter(context: FunctionInvocationContext, next) -> None:
    name = context.function.fully_qualified_name
    if name not in CACHEABLE:
        await next(context)
        return
    args = {k: str(v) for k, v in (context.arguments or {}).items()}
    key = hashlib.sha256(json.dumps([name, args], sort_keys=True).encode()).hexdigest()
    if key in _cache:
        context.result = FunctionResult(function=context.function.metadata, value=_cache[key])
        return                            # skip the model call
    await next(context)
    if context.result is not None:
        _cache[key] = str(context.result)
```

Register it in `get_kernel` with `kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, cache_filter)`. Setting `context.result` and returning without calling `next` skips the model call. Cache only functions whose output should not vary (classification, extraction) and include everything that changes the answer in the key; an exact-match cache misses near-duplicate questions by design.

---

## 5. Tests

```python
# tests/test_app.py
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.completion_usage import CompletionUsage
from semantic_kernel.contents import AuthorRole, ChatMessageContent

import app as app_module


class StubChat(ChatCompletionClientBase):
    """Offline chat service: canned reply plus usage metadata, like a real connector."""

    async def _inner_get_chat_message_contents(self, chat_history, settings):
        return [ChatMessageContent(
            role=AuthorRole.ASSISTANT,
            content="Have you tried turning it off and on again?",
            metadata={"usage": CompletionUsage(prompt_tokens=12, completion_tokens=9)},
        )]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "test")
    monkeypatch.setattr(app_module, "create_chat_service", lambda settings: StubChat(ai_model_id="stub"))
    with TestClient(app_module.app) as c:
        yield c


def test_chat(client):
    r = client.post("/chat", json={"message": "My router is broken", "tenant": "acme"})
    assert r.status_code == 200
    assert "turning it off" in r.json()["reply"]


def test_kernel_per_request(client):
    request = SimpleNamespace(app=client.app)
    k1, k2 = app_module.get_kernel(request), app_module.get_kernel(request)
    assert k1 is not k2
    assert k1.get_service() is k2.get_service()
```

Subclassing `ChatCompletionClientBase` gives a service the kernel accepts anywhere a real one goes, including behind agents; `unittest.mock.AsyncMock` does not. Replacing `create_chat_service` keeps the test offline and avoids acquiring an Azure token. Add a `tests/conftest.py` that puts the project root on `sys.path` if you keep tests in a subfolder.

---

## 6. Container

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

The service needs `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, optionally `APPLICATIONINSIGHTS_CONNECTION_STRING` and `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS`, and an identity with the *Cognitive Services OpenAI User* role on the Azure OpenAI resource (on AKS, workload identity; `AZURE_CLIENT_ID` selects a user-assigned identity). Point liveness probes at `/health`, which never calls the model. Scale on concurrency and watch the provider's tokens-per-minute quota, which usually runs out before CPU does.

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | 1.45.0 | Rewritten. The previous modules did not import (`Config`, `ChatRequest`, `logger` and a `health` module were undefined), built one kernel for the whole app despite the kernel-per-request advice, and claimed Python SK emits token-usage metrics (only .NET does). Generic Kubernetes manifests and checklists replaced by a link to the language-neutral production guide. |

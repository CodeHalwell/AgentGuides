---
title: "Semantic Kernel Production Guide (Python)"
description: "Production deployment, monitoring, resilience and testing for Semantic Kernel Python applications."
framework: "semantic-kernel"
language: "python"
---

**Production deployment, monitoring, resilience and testing for Python**

> Targets `semantic-kernel` %%version=semantic-kernel%% on Python 3.10+ (checked %%versions-checked%%). Semantic Kernel is supported, but new agent features land in its successor, [Microsoft Agent Framework](/microsoft-agent-framework-guide/python/); the deployment patterns here apply to both.

---

## Overview

This guide covers production deployment of Semantic Kernel Python applications including Docker/Kubernetes deployment, monitoring, security, performance optimization, and operational best practices.

**See also:** [general production guide](/semantic-kernel-guide/semantic_kernel_production_guide/) for language-agnostic patterns.

---

## Quick Start: Production Checklist

- [ ] Containerize application (Docker)
- [ ] Configure environment variables and secrets (Azure Key Vault)
- [ ] Implement structured logging and OpenTelemetry
- [ ] Add error handling, retries, and circuit breakers
- [ ] Configure horizontal scaling (Kubernetes HPA)
- [ ] Set up monitoring (Application Insights)
- [ ] Implement rate limiting and throttling
- [ ] Add health checks and readiness probes
- [ ] Configure CI/CD pipeline
- [ ] Perform load testing

---

## Docker Containerization

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```txt
semantic-kernel[azure]==%%version=semantic-kernel%%
fastapi
uvicorn[standard]
tenacity
circuitbreaker
azure-monitor-opentelemetry
azure-identity
azure-keyvault-secrets
```

Pin Semantic Kernel exactly and lock the rest (`pip-compile`, `uv lock`); SK already depends on `openai`, `pydantic` and the OpenTelemetry API/SDK, so do not pin those separately to versions it does not support. (There is no `openai` extra — the OpenAI connector is always installed.)

### Build and Run

```bash
# Build
docker build -t sk-python-app:latest .

# Run locally
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e AZURE_KEY_VAULT_URL=$AZURE_KEY_VAULT_URL \
  sk-python-app:latest

# Push to registry
docker tag sk-python-app:latest myregistry.azurecr.io/sk-python-app:latest
docker push myregistry.azurecr.io/sk-python-app:latest
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sk-python-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sk-python-app
  template:
    metadata:
      labels:
        app: sk-python-app
        version: v1
    spec:
      containers:
      - name: app
        image: myregistry.azurecr.io/sk-python-app:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: AZURE_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: azure-identity
              key: client-id
        - name: APPLICATIONINSIGHTS_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: app-insights
              key: connection-string
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
      serviceAccountName: sk-python-app
---
apiVersion: v1
kind: Service
metadata:
  name: sk-python-app
  namespace: production
spec:
  selector:
    app: sk-python-app
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sk-python-app-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sk-python-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy

```bash
# Create namespace
kubectl create namespace production

# Create secrets
kubectl create secret generic azure-identity \
  --from-literal=client-id=$AZURE_CLIENT_ID \
  -n production

kubectl create secret generic app-insights \
  --from-literal=connection-string=$APPLICATIONINSIGHTS_CONNECTION_STRING \
  -n production

# Deploy
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n production
kubectl logs -f deployment/sk-python-app -n production
```

---

## Production Application Structure

```python
# main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from semantic_kernel import Kernel
from config import Config
from monitoring import setup_monitoring
from health import HealthCheck

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.config = Config.from_env()
    app.state.kernel = create_kernel(app.state.config)
    app.state.chat_function = app.state.kernel.add_function(
        plugin_name="chat", function_name="reply", prompt="{{$input}}",
    )
    app.state.health = HealthCheck()

    setup_monitoring(app.state.config)

    yield

    # Shutdown: nothing to dispose on the Kernel itself; close clients you created

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    # Check dependencies
    if not app.state.health.check_openai():
        raise HTTPException(status_code=503, detail="OpenAI unavailable")
    return {"status": "ready"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = await app.state.kernel.invoke(
            app.state.chat_function,
            input=request.message
        )
        return {"response": str(result)}
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail="Internal error")   # don't leak exception text

def create_kernel(config: "Config") -> Kernel:
    from azure.identity import DefaultAzureCredential
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(
        deployment_name=config.azure_openai_deployment,
        endpoint=config.azure_openai_endpoint,
        credential=DefaultAzureCredential(),       # managed identity in production
    ))
    return kernel
```

```python
# config.py
from dataclasses import dataclass
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import os

@dataclass
class Config:
    openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    app_insights_connection_string: str
    max_retries: int = 3
    timeout: float = 30.0

    @classmethod
    def from_env(cls):
        # Use Key Vault in production
        if os.getenv("AZURE_KEY_VAULT_URL"):
            return cls.from_key_vault()
        else:
            return cls.from_environment()

    @classmethod
    def from_key_vault(cls):
        vault_url = os.environ["AZURE_KEY_VAULT_URL"]
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)

        return cls(
            openai_api_key=client.get_secret("openai-api-key").value,
            azure_openai_endpoint=client.get_secret("azure-openai-endpoint").value,
            azure_openai_deployment=client.get_secret("azure-openai-deployment").value,
            app_insights_connection_string=client.get_secret("app-insights-connection-string").value,
        )

    @classmethod
    def from_environment(cls):
        return cls(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            azure_openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_openai_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            app_insights_connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
        )
```

---

## Monitoring with OpenTelemetry

Semantic Kernel emits OpenTelemetry spans and metrics itself (function invocations, model calls, token usage). Turn on the GenAI diagnostics with `SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS=true` (add `..._SENSITIVE=true` only in development — it records prompts and completions), configure an exporter, and add your own metrics on top.

```python
# monitoring.py
from opentelemetry import trace, metrics
from azure.monitor.opentelemetry import configure_azure_monitor

def setup_monitoring(config: Config):
    """Configure OpenTelemetry with Azure Monitor"""

    # Configure Azure Monitor
    configure_azure_monitor(
        connection_string=config.app_insights_connection_string,
        logger_name="semantic_kernel",
    )

    # Get tracer and meter
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)

    # Create custom metrics
    request_counter = meter.create_counter(
        "sk.requests.total",
        description="Total requests",
        unit="1"
    )

    token_counter = meter.create_counter(
        "sk.tokens.used",
        description="Tokens used",
        unit="1"
    )

    latency_histogram = meter.create_histogram(
        "sk.request.duration",
        description="Request duration",
        unit="ms"
    )

    return tracer, meter
```

```python
# Instrumented kernel invocation
from opentelemetry import trace
import time

tracer = trace.get_tracer(__name__)

async def invoke_with_monitoring(kernel, function, **kwargs):
    with tracer.start_as_current_span("sk_invoke") as span:
        span.set_attribute("sk.function", function.name)
        span.set_attribute("sk.plugin", function.plugin_name)

        start = time.time()

        try:
            result = await kernel.invoke(function, **kwargs)

            duration_ms = (time.time() - start) * 1000
            span.set_attribute("sk.duration_ms", duration_ms)
            span.set_attribute("sk.status", "success")

            # Record metrics
            latency_histogram.record(duration_ms, {"function": function.name})

            # Token usage is recorded by SK's own metrics
            # (semantic_kernel.function.invocation.token_usage.*); no need to parse it here.

            return result

        except Exception as e:
            span.set_attribute("sk.status", "error")
            span.record_exception(e)
            raise
```

---

## Error Handling & Resilience

```python
# resilience.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from circuitbreaker import circuit
from semantic_kernel.exceptions import ServiceResponseException
import asyncio
import logging

logger = logging.getLogger(__name__)

class ResilientKernel:
    def __init__(self, kernel, max_retries=3, circuit_threshold=5):
        self.kernel = kernel
        self.max_retries = max_retries
        self.circuit_threshold = circuit_threshold

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # retry only service failures, not bugs in your own code
        retry=retry_if_exception_type(ServiceResponseException),
        reraise=True,
    )
    async def invoke_with_retry(self, function, **kwargs):
        """Invoke with automatic retry"""
        return await self.kernel.invoke(function, **kwargs)

    async def invoke_with_timeout(self, function, timeout=30.0, **kwargs):
        """Invoke with timeout"""
        try:
            return await asyncio.wait_for(
                self.kernel.invoke(function, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout after {timeout}s")
            raise

    @circuit(failure_threshold=5, recovery_timeout=60, expected_exception=ServiceResponseException)
    async def invoke_with_circuit_breaker(self, function, **kwargs):
        """Invoke with circuit breaker"""
        return await self.kernel.invoke(function, **kwargs)

    async def invoke_safe(self, function, timeout=60.0, **kwargs):
        """Timeout around retry around a single call"""
        return await asyncio.wait_for(self.invoke_with_retry(function, **kwargs), timeout=timeout)
```

The OpenAI connectors already retry rate-limit and 5xx responses inside the OpenAI SDK; keep application-level retries few, or you multiply them.

---

## Performance Optimization

### Caching

An exact-match cache like this helps for repeated, deterministic prompts (classification, extraction). It is not a semantic cache — near-duplicate questions miss.

```python
# caching.py
import logging
from typing import Dict, Any
import hashlib
import json

logger = logging.getLogger(__name__)

class PromptCache:
    def __init__(self, max_size=1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size

    def _make_key(self, function_name: str, **kwargs) -> str:
        """Generate cache key"""
        content = json.dumps({"func": function_name, "args": kwargs}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def get_or_invoke(self, kernel, function, **kwargs):
        """Get from cache or invoke"""
        key = self._make_key(function.name, **kwargs)

        if key in self.cache:
            logger.info(f"Cache hit for {function.name}")
            return self.cache[key]

        result = await kernel.invoke(function, **kwargs)

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = result
        return result
```

### Batching

```python
# batching.py
import asyncio
from typing import Any, List

class BatchProcessor:
    def __init__(self, kernel, function, batch_size=10):
        self.kernel = kernel
        self.function = function
        self.batch_size = batch_size

    async def process_batch(self, items: List[str]) -> List[Any]:
        """Process items in parallel batches"""
        results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]

            # Process batch in parallel
            tasks = [
                self.kernel.invoke(self.function, input=item)
                for item in batch
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results
```

---

## Testing

```python
# tests/test_kernel.py
import pytest
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents import AuthorRole, ChatMessageContent

class StubChat(ChatCompletionClientBase):
    """Offline chat service returning a canned reply."""
    async def _inner_get_chat_message_contents(self, chat_history, settings):
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, content="Mocked response")]

@pytest.fixture
def kernel() -> Kernel:
    k = Kernel()
    k.add_service(StubChat(ai_model_id="stub"))
    return k

@pytest.mark.asyncio
async def test_simple_invocation(kernel):
    function = kernel.add_function(plugin_name="t", function_name="echo", prompt="Test prompt")
    result = await kernel.invoke(function)
    assert str(result) == "Mocked response"
```

`unittest.mock.AsyncMock` does not satisfy the service interface the kernel looks up; subclassing `ChatCompletionClientBase` does, and also works behind agents.

---

## Security Best Practices

1. **Use Azure Key Vault for secrets**
2. **Enable Managed Identity in production**
3. **Implement rate limiting**
4. **Validate all inputs**
5. **Use HTTPS everywhere**
6. **Enable audit logging**
7. **Regular dependency updates**
8. **Content filtering and guardrails**

See: [Middleware Guide](/semantic-kernel-guide/python/semantic_kernel_middleware_python/) for guardrails implementation.

---

## Additional Resources

- [Comprehensive Guide](/semantic-kernel-guide/python/semantic_kernel_comprehensive_python/) - Complete reference
- [Recipes](/semantic-kernel-guide/python/semantic_kernel_recipes_python/) - Code examples
- [General Production Guide](/semantic-kernel-guide/semantic_kernel_production_guide/) - Language-agnostic patterns
- [Streaming Server Guide](/semantic-kernel-guide/python/semantic_kernel_streaming_server_fastapi/) - FastAPI patterns

**[Back to Python overview](/semantic-kernel-guide/python/)**


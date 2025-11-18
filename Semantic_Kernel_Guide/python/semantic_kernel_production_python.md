# Semantic Kernel Production Guide (Python)

**Production Deployment, Monitoring, and Best Practices for Python**

Last Updated: November 2025
Python Version: 3.9+
Semantic Kernel: 1.38.0+

---

## Overview

This guide covers production deployment of Semantic Kernel Python applications including Docker/Kubernetes deployment, monitoring, security, performance optimization, and operational best practices.

**See Also:** [../semantic_kernel_production_guide.md](../semantic_kernel_production_guide.md) for language-agnostic patterns.

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
FROM python:3.11-slim

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
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```txt
semantic-kernel[azure,openai]==1.38.0
fastapi==0.104.0
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
tenacity==8.2.3
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
azure-monitor-opentelemetry==1.1.0
azure-identity==1.15.0
azure-keyvault-secrets==4.7.0
pydantic==2.5.0
```

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
    app.state.kernel = await create_kernel(app.state.config)
    app.state.health = HealthCheck()

    setup_monitoring(app.state.config)

    yield

    # Shutdown
    await app.state.kernel.dispose()

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
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
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

```python
# monitoring.py
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter, AzureMonitorMetricExporter
import logging

def setup_monitoring(config: Config):
    """Configure OpenTelemetry with Azure Monitor"""

    # Configure Azure Monitor
    configure_azure_monitor(
        connection_string=config.app_insights_connection_string,
        logger_name="semantic_kernel"
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

            if hasattr(result, 'metadata'):
                tokens = result.metadata.get("total_tokens", 0)
                token_counter.add(tokens, {"function": function.name})
                span.set_attribute("sk.tokens", tokens)

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
        retry=retry_if_exception_type(Exception)
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

    @circuit(failure_threshold=5, recovery_timeout=60, expected_exception=Exception)
    async def invoke_with_circuit_breaker(self, function, **kwargs):
        """Invoke with circuit breaker"""
        return await self.kernel.invoke(function, **kwargs)

    async def invoke_safe(self, function, timeout=30.0, **kwargs):
        """Invoke with all resilience patterns"""
        return await self.invoke_with_timeout(
            function,
            timeout=timeout,
            **kwargs
        )
```

---

## Performance Optimization

### Caching

```python
# caching.py
from functools import lru_cache
from typing import Dict, Any
import hashlib
import json

class SemanticCache:
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
from typing import List

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
from unittest.mock import Mock, AsyncMock
from semantic_kernel import Kernel

@pytest.fixture
async def mock_kernel():
    kernel = Kernel()
    # Mock AI service
    mock_service = AsyncMock()
    mock_service.complete_async.return_value = "Mocked response"
    kernel.add_service(mock_service)
    return kernel

@pytest.mark.asyncio
async def test_simple_invocation(mock_kernel):
    function = mock_kernel.create_function_from_prompt("Test prompt")
    result = await mock_kernel.invoke(function)
    assert result is not None

@pytest.mark.asyncio
async def test_retry_on_failure():
    kernel = ResilientKernel(mock_kernel, max_retries=3)

    # Should retry and eventually succeed
    result = await kernel.invoke_with_retry(function, input="test")
    assert result is not None
```

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

See: [Middleware Guide](semantic_kernel_middleware_python.md) for guardrails implementation.

---

## Additional Resources

- [Comprehensive Guide](semantic_kernel_comprehensive_python.md) - Complete reference
- [Recipes](semantic_kernel_recipes_python.md) - Code examples
- [General Production Guide](../semantic_kernel_production_guide.md) - Language-agnostic patterns
- [Streaming Server Guide](semantic_kernel_streaming_server_fastapi.md) - FastAPI patterns

**[Back to Python README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**

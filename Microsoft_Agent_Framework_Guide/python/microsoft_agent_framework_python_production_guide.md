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

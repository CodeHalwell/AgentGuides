---
title: "Microsoft Agent Framework - Production & Deployment Guide"
description: "Deploying Microsoft Agent Framework services — containers, Azure Container Apps and AKS, scaling, monitoring, security and disaster recovery."
framework: "microsoft-agent-framework"
---

**Target Audience:** DevOps, Infrastructure, and Senior Engineers  
**First written:** October 2025 (preview) · **Reviewed:** %%versions-checked%% against `agent-framework` %%version=agent-framework%% (Python) and `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (.NET)  
**Focus:** Enterprise-grade deployment, scaling, monitoring, and security

---

## Table of Contents

1. [Production Deployment](#production-deployment)
2. [Scaling Strategies](#scaling-strategies)
3. [Monitoring & Observability](#monitoring--observability)
4. [Security Best Practices](#security-best-practices)
5. [High Availability & Disaster Recovery](#high-availability--disaster-recovery)
6. [Cost Optimisation](#cost-optimisation)
7. [Performance Tuning](#performance-tuning)
8. [Enterprise Governance](#enterprise-governance)

---

## Production Deployment

### Azure Container Apps Deployment

#### **Infrastructure as Code - Bicep**

```bicep
param environment string = 'prod'
param location string = resourceGroup().location
param agentImageUri string
param replicaCount int = 3

resource containerApp 'Microsoft.App/containerApps@2022-03-01' = {
  name: 'agent-framework-${environment}'
  location: location
  properties: {
    managedEnvironmentId: containerEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      dapr: {
        enabled: true
        appId: 'agent-framework'
      }
      secrets: [
        {
          name: 'azure-openai-key'
          value: listKeys(openaiAccount.id, '2023-05-01').keys[0].value
        }
        {
          name: 'cosmos-connection'
          value: 'AccountEndpoint=${cosmosAccount.properties.documentEndpoint};'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'agent-framework'
          image: agentImageUri
          resources: {
            cpu: '1'
            memory: '2Gi'
          }
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openaiAccount.properties.endpoint
            }
            {
              name: 'AZURE_OPENAI_KEY'
              secretRef: 'azure-openai-key'
            }
            {
              name: 'COSMOS_CONNECTION'
              secretRef: 'cosmos-connection'
            }
          ]
          probes: [
            {
              type: 'liveness'
              httpGet: {
                path: '/health/live'
                port: 8080
              }
              initialDelaySeconds: 30
              periodSeconds: 10
            }
            {
              type: 'readiness'
              httpGet: {
                path: '/health/ready'
                port: 8080
              }
              initialDelaySeconds: 10
              periodSeconds: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: replicaCount
        maxReplicas: replicaCount * 3
        rules: [
          {
            name: 'cpu-scaling'
            custom: {
              type: 'cpu'
              metadata: {
                type: 'Utilization'
                value: '70'
              }
            }
          }
          {
            name: 'memory-scaling'
            custom: {
              type: 'memory'
              metadata: {
                type: 'Utilization'
                value: '80'
              }
            }
          }
        ]
      }
    }
  }
}
```

#### **Docker Configuration**

```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0 as base
WORKDIR /app
EXPOSE 8080

FROM mcr.microsoft.com/dotnet/sdk:8.0 as build
WORKDIR /src
COPY ["AgentFramework.csproj", "./"]
RUN dotnet restore "AgentFramework.csproj"
COPY . .
RUN dotnet build "AgentFramework.csproj" -c Release -o /app/build

FROM build as publish
RUN dotnet publish "AgentFramework.csproj" -c Release -o /app/publish

FROM base as final
WORKDIR /app
COPY --from=publish /app/publish .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health/live || exit 1

ENTRYPOINT ["dotnet", "AgentFramework.dll"]
```

#### **Azure CLI Deployment**

```bash
#!/bin/bash

# Variables
RESOURCE_GROUP="rg-agent-framework"
CONTAINER_APP_NAME="agent-framework-app"
CONTAINER_APP_ENV="agent-env"
IMAGE_URI="your-registry.azurecr.io/agent-framework:latest"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location eastus

# Create container app environment
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location eastus

# Create container app
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $IMAGE_URI \
  --target-port 8080 \
  --ingress 'external' \
  --min-replicas 3 \
  --max-replicas 10 \
  --cpu 1.0 \
  --memory 2Gi \
  --set-env-vars \
    AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT \
    ENVIRONMENT=production

# Configure scaling rules
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --yaml - <<EOF
apiVersion: apps/core.io/v1alpha1
kind: ContainerApp
metadata:
  name: $CONTAINER_APP_NAME
  namespace: $RESOURCE_GROUP
spec:
  template:
    scale:
      minReplicas: 3
      maxReplicas: 10
      rules:
      - name: cpu-scaling
        custom:
          query: cpu
          metadata:
            type: Utilization
            value: '70'
      - name: http-scaling
        http:
          metadata:
            concurrentRequests: '100'
EOF
```

### Kubernetes Deployment

#### **Helm Chart Values**

```yaml
# helm values for Agent Framework
replicaCount: 3

image:
  repository: your-registry.azurecr.io/agent-framework
  pullPolicy: IfNotPresent
  tag: "1.0.0"

imagePullSecrets:
  - name: acr-secret

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: agent-framework.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: agent-framework-tls
      hosts:
        - agent-framework.example.com

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilisationPercentage: 70
  targetMemoryUtilisationPercentage: 80

healthChecks:
  liveness:
    enabled: true
    initialDelaySeconds: 30
    periodSeconds: 10
  readiness:
    enabled: true
    initialDelaySeconds: 10
    periodSeconds: 5

env:
  - name: AZURE_OPENAI_ENDPOINT
    valueFrom:
      secretKeyRef:
        name: agent-secrets
        key: azure-openai-endpoint
  - name: COSMOS_CONNECTION
    valueFrom:
      secretKeyRef:
        name: agent-secrets
        key: cosmos-connection

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values:
                  - agent-framework
          topologyKey: kubernetes.io/hostname
```

#### **Kubernetes Manifest**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-framework

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: agent-framework
data:
  config.json: |
    {
      "environment": "production",
      "logging": {
        "level": "Information",
        "format": "json"
      }
    }

---
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: agent-framework
type: Opaque
stringData:
  azure-openai-endpoint: https://your-resource.openai.azure.com
  cosmos-connection: AccountEndpoint=https://...;AccountKey=...

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-framework
  namespace: agent-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-framework
  template:
    metadata:
      labels:
        app: agent-framework
    spec:
      serviceAccountName: agent-framework
      containers:
      - name: agent-framework
        image: your-registry.azurecr.io/agent-framework:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: AZURE_OPENAI_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: azure-openai-endpoint
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: agent-framework-service
  namespace: agent-framework
spec:
  type: LoadBalancer
  selector:
    app: agent-framework
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-framework-hpa
  namespace: agent-framework
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-framework
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilisation
        averageUtilisation: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilisation
        averageUtilisation: 80
```

---

## Scaling Strategies

Agent services are I/O-bound: almost all of a request's time is spent waiting for the model. Two consequences follow. Scale on **concurrent requests**, not CPU; and remember that the model deployment's tokens-per-minute (TPM) quota is usually the real ceiling, long before your containers are busy.

### Horizontal scaling

In Azure Container Apps, scale on HTTP concurrency:

```bicep
scale: {
  minReplicas: 2           // keep two warm for availability
  maxReplicas: 20          // cap cost, and stay inside the model's TPM quota
  rules: [
    {
      name: 'http-concurrency'
      http: {
        metadata: {
          concurrentRequests: '20'
        }
      }
    }
  ]
}
```

On AKS, the equivalent is a KEDA `ScaledObject` on request rate, or an HPA on a custom "in-flight requests" metric.

### Vertical sizing

Each replica needs little CPU (it is mostly awaiting HTTP responses) but enough memory for in-flight conversations and any local tokenisation. 0.5–1 vCPU and 1–2 GiB is a sensible starting point; measure, then adjust the `resources` block. Container Apps does not resize a running replica — change the template and roll a new revision.

### Concurrency limits inside a replica

Bound the number of simultaneous model calls per replica so a burst queues locally instead of tripping the model's rate limit:

```python
import asyncio

from agent_framework import Agent, AgentContext, AgentMiddleware


class ConcurrencyLimit(AgentMiddleware):
    """Allow at most `limit` agent runs at once in this process; the rest wait their turn."""

    def __init__(self, limit: int) -> None:
        self._gate = asyncio.Semaphore(limit)

    async def process(self, context: AgentContext, call_next) -> None:
        async with self._gate:
            await call_next()


limiter = ConcurrencyLimit(limit=16)   # share one instance across all agents in the process
# agent = Agent(client=..., instructions=..., middleware=[limiter])
```

### Caching

Caching full answers is only safe for deterministic, non-personalised questions (FAQ-style lookups, classification of identical inputs). Do it in middleware, keyed on the agent and the exact input, and keep the TTL short. In a multi-replica deployment, use Redis rather than process memory.

```python
import hashlib
import time

from agent_framework import AgentContext, AgentMiddleware, AgentResponse


class ResponseCache(AgentMiddleware):
    def __init__(self, ttl_seconds: float = 300) -> None:
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, AgentResponse]] = {}

    def _key(self, context: AgentContext) -> str:
        text = "\n".join(f"{m.role}:{m.text}" for m in context.messages)
        return hashlib.sha256(f"{context.agent.name}|{text}".encode()).hexdigest()

    async def process(self, context: AgentContext, call_next) -> None:
        if context.session is not None or context.stream:
            await call_next()          # never cache conversational or streaming runs
            return

        key = self._key(context)
        hit = self._store.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl:
            context.result = hit[1]    # short-circuit: the model is not called
            return

        await call_next()
        if isinstance(context.result, AgentResponse):
            self._store[key] = (time.monotonic(), context.result)
```

Provider-side prompt caching (automatic on recent OpenAI and Azure OpenAI models for long, repeated prefixes) is the cheaper win: keep stable instructions and tool definitions at the start of the prompt and it applies without any code.

---

## Monitoring & Observability

### Application Insights

`configure_azure_monitor()` (package `azure-monitor-opentelemetry`) sets up traces, metrics and logs to Application Insights; `enable_instrumentation()` turns on the framework's GenAI spans and metrics inside it.

```python
import os

from agent_framework.observability import enable_instrumentation
from azure.monitor.opentelemetry import configure_azure_monitor


def setup_observability() -> None:
    configure_azure_monitor(connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"])
    enable_instrumentation(enable_sensitive_data=False)   # spans carry metadata, not prompt text

# Call once at start-up, before creating agents. Every agent.run() then produces
# invoke_agent / chat / execute_tool spans and gen_ai.client.token.usage and
# gen_ai.client.operation.duration metrics, tagged with agent and model names.
```

For .NET, wrap agents with `.AsBuilder().UseOpenTelemetry(sourceName)` and add the Azure Monitor exporter; see the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#net-opentelemetry-integration).

### Custom metrics

The built-in metrics cover model calls. Add business metrics — per-tenant usage, outcome counts — with middleware so every agent reports them the same way.

```python
import time

from agent_framework import AgentContext, AgentMiddleware, AgentResponse
from opentelemetry import metrics

meter = metrics.get_meter("contoso.agents")
runs = meter.create_counter("agent.runs", unit="1", description="Agent runs by outcome")
duration = meter.create_histogram("agent.run.duration", unit="ms", description="Agent run duration")
tokens = meter.create_counter("agent.tokens", unit="1", description="Tokens used, by tenant")


class AgentMetricsMiddleware(AgentMiddleware):
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    async def process(self, context: AgentContext, call_next) -> None:
        attrs = {"agent": context.agent.name or "unnamed", "tenant": self.tenant_id}
        start = time.perf_counter()
        try:
            await call_next()
            runs.add(1, {**attrs, "outcome": "success"})
            if isinstance(context.result, AgentResponse) and context.result.usage_details:
                tokens.add(context.result.usage_details.get("total_token_count") or 0, attrs)
        except Exception:
            runs.add(1, {**attrs, "outcome": "error"})
            raise
        finally:
            duration.record((time.perf_counter() - start) * 1000, attrs)


# await agent.run(question, middleware=[AgentMetricsMiddleware(tenant_id=claims["tid"])])
```

### Log aggregation

With `configure_azure_monitor()` in place, standard `logging` records are exported too; `extra=` fields become custom dimensions you can query. Log identifiers and sizes, not prompt content.

```python
import logging

logger = logging.getLogger("contoso.agents")
logger.setLevel(logging.INFO)


def log_agent_execution(agent_name: str, request_id: str, result_text: str, elapsed_ms: float) -> None:
    logger.info(
        "agent run completed",
        extra={"agent": agent_name, "request_id": request_id, "result_length": len(result_text), "elapsed_ms": elapsed_ms},
    )


def log_agent_error(agent_name: str, request_id: str, error: Exception) -> None:
    logger.error(
        "agent run failed",
        extra={"agent": agent_name, "request_id": request_id, "error_type": type(error).__name__},
        exc_info=error,
    )
```

### Alerting

Create alerts as Azure Monitor log search alerts over the exported telemetry. Useful starting points (KQL against the Application Insights workspace):

```kusto
// Error rate above 5% over 5 minutes (from the agent.runs counter above)
customMetrics
| where name == "agent.runs"
| summarize errors = sumif(valueSum, tostring(customDimensions.outcome) == "error"), total = sum(valueSum) by bin(timestamp, 5m)
| extend error_rate = errors / total
| where error_rate > 0.05

// p99 model latency above 5 s (framework span)
dependencies
| where name startswith "chat"
| summarize p99 = percentile(duration, 99) by bin(timestamp, 10m)
| where p99 > 5000

// Throttling from the model endpoint
dependencies
| where name startswith "chat" and resultCode == "429"
| summarize throttled = count() by bin(timestamp, 5m)
| where throttled > 10
```

```bash
az monitor scheduled-query create \
  --name agent-high-error-rate --resource-group rg-agents \
  --scopes "$APP_INSIGHTS_ID" --evaluation-frequency 5m --window-size 5m \
  --condition "count 'ErrorRate' > 0" \
  --condition-query ErrorRate="customMetrics | where name == 'agent.runs' | summarize e=sumif(valueSum, tostring(customDimensions.outcome)=='error'), t=sum(valueSum) | where e/t > 0.05" \
  --action-groups "$ACTION_GROUP_ID"
```

---

## Security Best Practices

### Secrets management

The best secret is the one you don't have: use managed identity for Azure OpenAI / Foundry, Cosmos DB, Storage, AI Search and Key Vault, and disable key-based auth on those resources. For the secrets that remain (third-party API keys used by tools), read them from Key Vault at start-up with the async client and cache them in memory.

```python
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient


async def load_tool_secrets(vault_url: str, names: list[str]) -> dict[str, str]:
    """Fetch the named secrets once at start-up."""
    async with DefaultAzureCredential() as credential, SecretClient(vault_url=vault_url, credential=credential) as client:
        return {name: (await client.get_secret(name)).value for name in names}

# secrets = await load_tool_secrets("https://your-keyvault.vault.azure.net/", ["weather-api-key"])
```

In Container Apps you can also reference Key Vault secrets directly from the app's `secrets` configuration (`keyVaultUrl` plus the app's managed identity), so they arrive as environment variables without any code.

### Network security

Put the model, data and search services behind private endpoints with public network access disabled, and integrate the Container Apps environment (or AKS cluster) with the same virtual network. Then restrict inbound traffic to the front door:

```bicep
resource agentNsg 'Microsoft.Network/networkSecurityGroups@2023-09-01' = {
  name: 'agent-framework-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowHttpsFromFrontDoor'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: 'AzureFrontDoor.Backend'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
        }
      }
      {
        name: 'DenyAllInbound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
          access: 'Deny'
          priority: 4096
          direction: 'Inbound'
        }
      }
    ]
  }
}
```

### Data encryption

Azure services encrypt data at rest by default, and can use customer-managed keys where policy requires it. Application-level encryption is worth adding for one thing in particular: **serialised conversation sessions** you store yourself, which contain the full conversation.

```python
import json

from agent_framework import AgentSession
from cryptography.fernet import Fernet


class EncryptedSessionStore:
    """Encrypts AgentSession state before it leaves the process. Keep the key in Key Vault."""

    def __init__(self, key: bytes, backend: dict[str, bytes]) -> None:
        self._fernet = Fernet(key)
        self._backend = backend               # stand-in for Blob Storage / Redis / Cosmos DB

    def save(self, session: AgentSession) -> None:
        payload = json.dumps(session.to_dict()).encode()
        self._backend[session.session_id] = self._fernet.encrypt(payload)

    def load(self, session_id: str) -> AgentSession | None:
        blob = self._backend.get(session_id)
        if blob is None:
            return None
        return AgentSession.from_dict(json.loads(self._fernet.decrypt(blob)))


# store = EncryptedSessionStore(Fernet.generate_key(), backend={})
# store.save(session); restored = store.load(session.session_id)
```

---

## High Availability & Disaster Recovery

### Multi-region deployment

- **Model endpoints:** deploy the model in two regions (or use a Global / Data Zone deployment, which Azure routes across regions for you). Put Azure API Management or Front Door in front if you need explicit failover between deployments.
- **Compute:** run the agent service in two regions behind Azure Front Door with health probes.
- **State:** Cosmos DB with multi-region writes (or a failover region) for sessions and workflow checkpoints; geo-redundant storage for anything in Blob Storage.
- **Targets:** decide RPO and RTO explicitly. With Cosmos DB multi-region and stateless compute, an RPO of seconds and an RTO of a few minutes is realistic.

### Failing over between model deployments

The simplest model-level failover is two agents with the same instructions on different deployments, trying the second when the first fails with a service error:

```python
from agent_framework import Agent, AgentResponse
from agent_framework.exceptions import ChatClientException
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential

credential = DefaultAzureCredential()
INSTRUCTIONS = "You are a support assistant."

primary = Agent(
    client=OpenAIChatClient(model="gpt-4o-mini", azure_endpoint="https://contoso-uksouth.openai.azure.com", credential=credential),
    name="support", instructions=INSTRUCTIONS,
)
secondary = Agent(
    client=OpenAIChatClient(model="gpt-4o-mini", azure_endpoint="https://contoso-swedencentral.openai.azure.com", credential=credential),
    name="support", instructions=INSTRUCTIONS,
)


async def run_with_failover(question: str) -> AgentResponse:
    try:
        return await primary.run(question)
    except ChatClientException:
        return await secondary.run(question)
```

Server-side conversation state (Responses API `previous_response_id`, Foundry threads) lives in one region. If you need conversations to survive a regional failover, keep history client-side (in the session or a Cosmos DB history provider) rather than on the service.

### Backups

What needs backing up is the state you own: stored sessions, chat history and workflow checkpoints. Keep them in Cosmos DB with **continuous backup** (point-in-time restore) or in Blob Storage with soft delete and versioning, rather than writing your own backup loop. Agents themselves are code and configuration; rebuild them from source control.

### Circuit breaker

When the model endpoint is failing, a circuit breaker stops every request from waiting for its own timeout. As chat middleware it protects every agent on the client:

```python
import time

from agent_framework import ChatContext, ChatMiddleware
from agent_framework.exceptions import ChatClientException


class CircuitOpenError(Exception):
    pass


class CircuitBreaker(ChatMiddleware):
    def __init__(self, failure_threshold: int = 5, reset_after_seconds: float = 60) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after = reset_after_seconds
        self.failures = 0
        self.opened_at: float | None = None

    async def process(self, context: ChatContext, call_next) -> None:
        if self.opened_at is not None:
            if time.monotonic() - self.opened_at < self.reset_after:
                raise CircuitOpenError("model endpoint circuit is open; failing fast")
            self.opened_at = None              # half-open: let one call through

        try:
            await call_next()
        except ChatClientException:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
            raise
        else:
            self.failures = 0


breaker = CircuitBreaker()
# client = OpenAIChatClient(middleware=[breaker])   # every agent using this client is protected
```

The OpenAI SDK underneath already retries transient errors with back-off (`max_retries`, default 2); the breaker is for sustained outages, where retrying only adds load.

---

## Cost Optimisation

### Where the money goes

For most agent services, model tokens dominate; compute, storage and networking are small by comparison. So cost work is mostly token work:

- **Measure** tokens per request, per agent and per tenant (the `agent.tokens` metric above, or the built-in `gen_ai.client.token.usage`).
- **Right-size the model:** route simple tasks (classification, extraction, short answers) to a small model and keep the large one for reasoning.
- **Trim context:** long histories are re-sent every turn. Use compaction (`compaction_strategy=` on `Agent`) or a history limit.
- **Use prompt caching:** stable instructions and tool definitions first.
- **Cap output:** set `max_tokens` in `default_options` where answers should be short.

```python
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity.aio import ManagedIdentityCredential

credential = ManagedIdentityCredential()

classifier = Agent(
    client=FoundryChatClient(model="gpt-4o-mini", credential=credential),
    name="classifier",
    instructions="Classify the request as billing, technical or general. Reply with one word.",
    default_options={"max_tokens": 5, "temperature": 0},
)

specialist = Agent(
    client=FoundryChatClient(model="gpt-4o", credential=credential),
    name="specialist",
    instructions="You resolve complex technical support cases.",
)


async def answer(question: str) -> str:
    category = (await classifier.run(question)).text.strip().lower()
    if category == "technical":
        return (await specialist.run(question)).text
    return f"Routing to the {category} team."
```

Then set budgets: Azure Cost Management budgets with alerts on the resource group, and per-deployment TPM quotas as a hard ceiling.

### Compute

Use the Container Apps consumption profile (scale to the minimum replica count when idle), set `maxReplicas` to what the model quota can actually serve, and review p95 CPU and memory monthly to right-size the container.

---

## Performance Tuning

### Client reuse and connection pooling

Create chat clients once per process and share them between agents; each client holds an HTTP connection pool. Creating a client (or a credential) per request defeats pooling and adds a token fetch to every call. To tune the pool, pass your own `AsyncOpenAI` client with a configured `httpx` client:

```python
import httpx
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI

http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=httpx.Timeout(60.0, connect=5.0),
)

shared_client = OpenAIChatClient(
    model="gpt-4o-mini",
    async_client=AsyncOpenAI(http_client=http_client, max_retries=3),
)
# Every Agent(client=shared_client, ...) reuses the same pool.
```

The same applies to Cosmos DB, AI Search and Storage clients used by tools and context providers: one instance per process.

### Batch processing

Run independent requests concurrently, bounded so a large batch doesn't exceed the model's rate limit, and collect failures instead of aborting the batch:

```python
import asyncio

from agent_framework import Agent, AgentResponse


async def batch_agent_executions(
    agent: Agent, queries: list[str], max_concurrency: int = 10
) -> list[AgentResponse | BaseException]:
    gate = asyncio.Semaphore(max_concurrency)

    async def one(query: str) -> AgentResponse:
        async with gate:
            return await agent.run(query)

    return await asyncio.gather(*(one(q) for q in queries), return_exceptions=True)

# results = await batch_agent_executions(agent, queries)
# failed = [q for q, r in zip(queries, results) if isinstance(r, BaseException)]
```

For large offline jobs where latency doesn't matter, the Azure OpenAI Batch API is roughly half the price of synchronous calls; call it directly through the `openai` SDK rather than through an agent.

---

## Enterprise Governance

### Compliance and auditing

Audit records for agent runs and tool calls are best produced by middleware, so no code path can skip them. The [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#audit-and-compliance-logging) has complete `AgentAuditMiddleware` and `ToolAuditMiddleware` examples, plus Microsoft Purview integration for DLP and the Sentinel query that reads the resulting records.

For retention, send audit logs to a Log Analytics workspace with the retention your regulator requires, and archive to Storage with an immutability (WORM) policy where records must be tamper-proof.

### Policy enforcement

Use Azure Policy for the platform rules, assigned at the subscription or management group so new resources inherit them. Built-in definitions cover most of what agent workloads need:

- Azure AI services should have key access disabled (managed identity only).
- Azure AI services should restrict network access / use private link.
- Cosmos DB accounts should disable public network access.
- Require tags (`cost-centre`, `environment`, `owner`) on resource groups.
- Allowed locations, for data residency.

```bash
# Example: require managed identity (no keys) on Azure AI services in a subscription
az policy assignment create \
  --name deny-ai-local-auth \
  --display-name "Azure AI services should have key access disabled" \
  --policy 71ef260a-8f18-47b7-abcb-62d0673d94dc \
  --scope "/subscriptions/$SUBSCRIPTION_ID"
```

Policies inside the agent — which tools a user may call, which models a team may use, per-tenant budgets — are middleware; see [Enterprise Governance](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/#enterprise-governance).

---

For code-level detail, see:

- [Comprehensive guide](/microsoft-agent-framework-guide/microsoft_agent_framework_comprehensive_guide/) — core concepts
- [Recipes](/microsoft-agent-framework-guide/microsoft_agent_framework_recipes/) — practical examples
- [Diagrams](/microsoft-agent-framework-guide/microsoft_agent_framework_diagrams/) — architecture visualisations

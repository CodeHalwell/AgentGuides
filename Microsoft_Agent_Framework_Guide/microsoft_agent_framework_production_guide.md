# Microsoft Agent Framework - Production & Deployment Guide
## October 2025 Release

**Document Version:** 1.0  
**Target Audience:** DevOps, Infrastructure, and Senior Engineers  
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
              query: 'cpu'
              metadata: {
                type: 'Utilization'
                value: '70'
              }
            }
          }
          {
            name: 'memory-scaling'
            custom: {
              query: 'memory'
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

### Horizontal Scaling Configuration

```python
# Kubernetes HPA configuration class
class HorizontalScalingConfig:
    """Production-grade horizontal scaling configuration"""
    
    def __init__(self):
        self.min_replicas = 3  # Minimum availability
        self.max_replicas = 20  # Cost control
        
        # CPU-based scaling
        self.cpu_target = 70  # Scale up at 70% CPU
        self.cpu_scale_up_period = 60  # Check every 60s
        self.cpu_scale_down_period = 300  # Conservative downscale
        
        # Memory-based scaling
        self.memory_target = 80  # Scale up at 80% memory
        
        # Request-based scaling
        self.concurrent_requests_target = 100
        self.requests_per_second_target = 500
```

### Vertical Scaling

```bicep
# Vertical scaling via container resource adjustment
resource scalingPolicy 'Microsoft.Insights/autoscaleSettings@2021-05-01-preview' = {
  name: 'agent-vertical-scaling'
  location: location
  properties: {
    enabled: true
    targetResourceUri: containerApp.id
    profiles: [
      {
        name: 'Auto scale based on Load'
        capacity: {
          minimum: '1'
          maximum: '4'
          default: '1'
        }
        rules: [
          {
            metricTrigger: {
              metricName: 'CpuPercentage'
              metricResourceUri: containerApp.id
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT5M'
              timeAggregation: 'Average'
              operator: 'GreaterThan'
              threshold: 70
            }
            scaleAction: {
              direction: 'Increase'
              type: 'ChangeCount'
              value: '1'
              cooldown: 'PT5M'
            }
          }
        ]
      }
    ]
  }
}
```

### Caching & Rate Limiting

```python
from functools import lru_cache
from datetime import timedelta
import asyncio

class ProductionCachingStrategy:
    """Multi-level caching for production performance"""
    
    def __init__(self):
        self.local_cache = {}
        self.cache_ttl = timedelta(minutes=5)
        self.rate_limiter = RateLimiter(
            calls_per_second=1000,
            time_window=1
        )
    
    async def get_cached_response(self, key: str):
        """Retrieve from cache with TTL"""
        if key in self.local_cache:
            cached_item = self.local_cache[key]
            if datetime.utcnow() < cached_item['expiry']:
                return cached_item['value']
        return None
    
    async def cache_response(self, key: str, value: Any):
        """Store in cache with TTL"""
        self.local_cache[key] = {
            'value': value,
            'expiry': datetime.utcnow() + self.cache_ttl
        }
    
    async def apply_rate_limit(self):
        """Apply rate limiting to prevent abuse"""
        await self.rate_limiter.wait()
```

---

## Monitoring & Observability

### Application Insights Configuration

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.azuremonitor import AzureMonitorTraceExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry import configure_azure_monitor

class ProductionObservability:
    """Enterprise-grade observability setup"""
    
    def __init__(self, connection_string: str):
        # Configure Azure Monitor
        configure_azure_monitor(connection_string=connection_string)
        
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # Create metrics
        self.request_counter = self.meter.create_counter(
            name="agent.requests.total",
            description="Total number of requests",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="agent.request.duration",
            description="Request duration in ms",
            unit="ms"
        )
        
        self.error_counter = self.meter.create_counter(
            name="agent.errors.total",
            description="Total number of errors",
            unit="1"
        )
    
    async def trace_request(self, request_id: str, query: str):
        """Trace individual request"""
        with self.tracer.start_as_current_span(f"agent_request_{request_id}") as span:
            span.set_attribute("request.id", request_id)
            span.set_attribute("request.query", query)
            
            # Track duration
            start = time.perf_counter()
            try:
                # Process request
                duration = (time.perf_counter() - start) * 1000
                self.request_duration.record(duration)
                self.request_counter.add(1)
            except Exception as e:
                self.error_counter.add(1)
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                raise
```

### Custom Metrics

```python
class AgentMetrics:
    """Track agent-specific metrics"""
    
    def __init__(self, meter):
        self.meter = meter
        
        # Agent lifecycle metrics
        self.agents_created = meter.create_counter(
            "agent.created.total",
            description="Number of agents created"
        )
        
        self.agents_executed = meter.create_counter(
            "agent.executed.total",
            description="Number of agent executions"
        )
        
        # Tool metrics
        self.tools_invoked = meter.create_counter(
            "agent.tools.invoked.total",
            description="Number of tool invocations"
        )
        
        self.tool_duration = meter.create_histogram(
            "agent.tool.duration",
            description="Tool execution duration (ms)"
        )
        
        # Memory metrics
        self.memory_operations = meter.create_counter(
            "agent.memory.operations.total",
            description="Memory operations (read/write)"
        )
        
        # LLM metrics
        self.llm_calls = meter.create_counter(
            "agent.llm.calls.total",
            description="Number of LLM API calls"
        )
        
        self.token_usage = meter.create_histogram(
            "agent.tokens.used",
            description="Token usage per request"
        )
```

### Log Aggregation

```python
import logging
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter

class ProductionLogger:
    """Production-grade logging"""
    
    def __init__(self):
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create logger
        self.logger = logging.getLogger("agent_framework")
        
        # Add structured logging
        self.logger.info(
            "Agent Framework initialised",
            extra={
                "component": "core",
                "version": "1.0.0",
                "environment": "production"
            }
        )
    
    def log_agent_execution(self, agent_id: str, query: str, result: str):
        """Log agent execution"""
        self.logger.info(
            f"Agent {agent_id} executed",
            extra={
                "agent_id": agent_id,
                "query": query[:100],  # Truncate for security
                "result_length": len(result),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_error(self, agent_id: str, error: Exception):
        """Log errors with full context"""
        self.logger.error(
            f"Agent {agent_id} error: {str(error)}",
            extra={
                "agent_id": agent_id,
                "error_type": type(error).__name__,
                "error_details": str(error)
            },
            exc_info=True
        )
```

### Alerting Strategy

```yaml
# Azure Monitor Alert Rules
alerts:
  - name: HighErrorRate
    description: Alert when error rate exceeds 5%
    condition:
      metric: agent.errors.total
      threshold: 5
      operator: GreaterThan
      window: 5m
    actions:
      - type: email
        recipients:
          - ops-team@company.com
      - type: webhook
        url: https://alerts.company.com/hooks/agent-errors

  - name: HighLatency
    description: Alert when p99 latency exceeds 5s
    condition:
      metric: agent.request.duration
      percentile: 99
      threshold: 5000
      operator: GreaterThan
      window: 10m
    actions:
      - type: pagerduty
        service_id: agent-framework-service

  - name: HighMemoryUsage
    description: Alert when memory usage exceeds 85%
    condition:
      metric: container_memory_usage_bytes
      threshold: 85
      operator: GreaterThan
      window: 5m
    actions:
      - type: auto_scale
        action: increase_replicas
```

---

## Security Best Practices

### Secrets Management

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

class SecureSecretsManagement:
    """Production-grade secrets management"""
    
    def __init__(self, vault_url: str):
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
    
    async def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Key Vault"""
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret: {secret_name}")
            raise
    
    async def set_secret(self, secret_name: str, value: str, tags: dict = None):
        """Store secret in Key Vault"""
        self.client.set_secret(secret_name, value, tags=tags)
    
    async def rotate_secret(self, secret_name: str):
        """Initiate secret rotation"""
        # Implementation triggers Azure Key Vault rotation
        pass

# Usage
secrets_manager = SecureSecretsManagement(
    vault_url="https://your-keyvault.vault.azure.net/"
)

# Load secrets
azure_openai_key = await secrets_manager.get_secret("azure-openai-key")
cosmos_connection = await secrets_manager.get_secret("cosmos-connection-string")
```

### Network Security

```bicep
# Network Security Group for Agent Framework
resource agentNsg 'Microsoft.Network/networkSecurityGroups@2021-02-01' = {
  name: 'agent-framework-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'AllowHTTPSInbound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowAzureServicesInbound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: 'AzureCloud'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 110
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
          priority: 1000
          direction: 'Inbound'
        }
      }
    ]
  }
}
```

### Data Encryption

```python
from cryptography.fernet import Fernet
from azure.storage.blob.aio import BlobClient

class DataEncryption:
    """Production data encryption"""
    
    def __init__(self, encryption_key: str):
        self.cipher_suite = Fernet(encryption_key.encode())
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive information"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive information"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    async def upload_encrypted_blob(self, container: str, blob_name: str, data: str):
        """Upload encrypted data to blob storage"""
        encrypted = self.encrypt_sensitive_data(data)
        
        async with BlobClient.from_connection_string(
            conn_str=self.connection_string,
            container_name=container,
            blob_name=blob_name
        ) as blob_client:
            await blob_client.upload_blob(encrypted, overwrite=True)
```

---

## High Availability & Disaster Recovery

### Multi-Region Deployment

```yaml
# Multi-region configuration
primary_region: eastus
secondary_region: westus

deployment:
  primary:
    location: eastus
    replicas: 3
    database: cosmos-db-primary
    storage: blob-primary
  secondary:
    location: westus
    replicas: 3
    database: cosmos-db-secondary
    storage: blob-secondary

failover_strategy:
  type: automatic
  detection_threshold: 30s
  switching_time: 60s
  
recovery_point_objective: 1 minute
recovery_time_objective: 5 minutes
```

### Backup Strategy

```python
class BackupStrategy:
    """Production backup and recovery"""
    
    async def backup_agent_state(self, agent_id: str):
        """Regular state backups"""
        state = await self.load_agent_state(agent_id)
        
        # Create backup
        backup_id = f"backup-{agent_id}-{datetime.utcnow().isoformat()}"
        await self.storage_client.upload_blob(
            container="agent-backups",
            name=backup_id,
            data=json.dumps(state)
        )
        
        return backup_id
    
    async def restore_from_backup(self, backup_id: str):
        """Restore from backup"""
        blob_client = await self.storage_client.get_blob_client(
            container="agent-backups",
            blob=backup_id
        )
        
        backup_data = await blob_client.download_blob()
        return json.loads(await backup_data.readall())
    
    async def scheduled_backup_job(self):
        """Run periodic backups"""
        while True:
            agents = await self.list_all_agents()
            for agent in agents:
                await self.backup_agent_state(agent.id)
            
            # Backup every hour
            await asyncio.sleep(3600)
```

### Circuit Breaker Pattern

```python
from enum import Enum
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Fail fast
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Resilience pattern for production"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
    
    async def execute(self, func, *args, **kwargs):
        """Execute with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.error(f"Circuit breaker opened: {e}")
            
            raise
```

---

## Cost Optimisation

### Cost Analysis Framework

```python
class CostAnalysis:
    """Track and optimise costs"""
    
    def __init__(self, subscription_id: str):
        self.subscription_id = subscription_id
        self.cost_management_client = CostManagementClient(...)
    
    async def calculate_monthly_cost(self):
        """Calculate cost breakdown"""
        costs = {
            "azure_openai": await self._calculate_llm_costs(),
            "compute": await self._calculate_compute_costs(),
            "storage": await self._calculate_storage_costs(),
            "networking": await self._calculate_networking_costs()
        }
        
        costs["total"] = sum(costs.values())
        return costs
    
    async def _calculate_llm_costs(self):
        """LLM API costs"""
        # Estimate based on:
        # - Token usage
        # - Model type
        # - API pricing
        
        tokens_per_month = await self._estimate_token_usage()
        model_prices = {
            "gpt-4": 0.03,  # per 1K tokens
            "gpt-4o": 0.015,
            "gpt-4o-mini": 0.0005
        }
        
        return sum(
            tokens_per_month.get(model, 0) * model_prices[model]
            for model in model_prices
        )
    
    async def _calculate_compute_costs(self):
        """Container instance costs"""
        # Based on CPU, memory, running hours
        cpu_hours = 3 * 24 * 30  # 3 replicas, 24/7, 30 days
        memory_gb_hours = 6 * 24 * 30  # 2GB per replica
        
        return (cpu_hours * 0.048) + (memory_gb_hours * 0.0050)
    
    async def optimise_costs(self):
        """Generate cost optimisation recommendations"""
        recommendations = []
        
        # Check reserved instances
        if self.compute_utilisation < 60:
            recommendations.append({
                "area": "compute",
                "recommendation": "Reduce minimum replicas",
                "potential_savings": "15-20%"
            })
        
        # Check LLM model efficiency
        if self.token_efficiency < 80:
            recommendations.append({
                "area": "llm",
                "recommendation": "Use gpt-4o-mini for simpler tasks",
                "potential_savings": "70%"
            })
        
        return recommendations
```

### Resource Optimisation

```python
class ResourceOptimisation:
    """Optimise resource allocation"""
    
    async def right_size_compute(self):
        """Adjust compute resources based on actual usage"""
        
        # Analyze historical metrics
        usage_metrics = await self.get_usage_metrics(days=30)
        
        # Calculate optimal sizing
        p95_cpu = numpy.percentile(usage_metrics['cpu'], 95)
        p95_memory = numpy.percentile(usage_metrics['memory'], 95)
        
        # Add 20% buffer
        recommended_cpu = p95_cpu * 1.2
        recommended_memory = p95_memory * 1.2
        
        # Update resource limits
        await self.update_container_app_resources(
            cpu=recommended_cpu,
            memory=recommended_memory
        )
    
    async def optimise_caching(self):
        """Maximise cache hit rates"""
        
        # Analyse cache patterns
        stats = await self.get_cache_statistics()
        
        if stats['hit_rate'] < 0.7:
            # Increase cache TTL or size
            await self.update_cache_config(
                ttl=600,  # Increase to 10 minutes
                max_size=1000  # Increase to 1000 items
            )
```

---

## Performance Tuning

### Connection Pooling

```python
from asyncpg import create_pool

class OptimisedConnections:
    """Optimised connection management"""
    
    async def init_pools(self):
        """Initialise connection pools"""
        
        # Azure OpenAI client pool
        self.openai_pool = aiohttp.ClientSession()
        
        # Cosmos DB pool
        self.cosmos_pool = await create_pool(
            dsn="cosmosdb://...",
            min_size=10,
            max_size=50
        )
        
        # Azure Search pool
        self.search_client = SearchClient(...)
    
    async def cleanup(self):
        """Clean up pools"""
        await self.openai_pool.close()
        await self.cosmos_pool.close()
```

### Batch Processing

```python
class BatchProcessing:
    """Efficient batch operations"""
    
    async def batch_agent_executions(self, queries: List[str], batch_size: int = 10):
        """Process queries in batches"""
        
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            
            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self.agent.run(query) for query in batch],
                return_exceptions=True
            )
            
            results.extend(batch_results)
        
        return results
    
    async def batch_vector_upsert(self, embeddings: List[Dict], batch_size: int = 100):
        """Batch insert embeddings"""
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            
            # Upsert to search index
            await self.search_client.upload_documents(batch)
```

---

## Enterprise Governance

### Compliance & Auditing

```python
class ComplianceFramework:
    """Enterprise compliance tracking"""
    
    async def log_data_access(self, user_id: str, resource_id: str, action: str):
        """Audit trail for data access"""
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "resource_id": resource_id,
            "action": action,
            "ip_address": self.request_context.client_ip,
            "session_id": self.request_context.session_id
        }
        
        # Store in immutable log
        await self.audit_log_storage.append(audit_entry)
    
    async def compliance_report(self, start_date: datetime, end_date: datetime):
        """Generate compliance report"""
        
        audit_entries = await self.audit_log_storage.query(
            start=start_date,
            end=end_date
        )
        
        report = {
            "period": f"{start_date} to {end_date}",
            "total_accesses": len(audit_entries),
            "unique_users": len(set(e['user_id'] for e in audit_entries)),
            "sensitive_access_count": len([
                e for e in audit_entries
                if e['resource_id'].startswith('sensitive_')
            ]),
            "anomalies": self._detect_anomalies(audit_entries)
        }
        
        return report
```

### Policy Enforcement

```yaml
# Azure Policy for Agent Framework governance
policies:
  - name: EnforceTagging
    description: Require specific tags on all resources
    rules:
      - resource_type: containerApp
        required_tags:
          - cost-centre
          - environment
          - owner
          - application
  
  - name: EnforceEncryption
    description: Enforce encryption at rest
    rules:
      - service: cosmosdb
        encryption_enabled: true
      - service: blob_storage
        encryption_type: "AES-256"
  
  - name: EnforceNetworkPolicy
    description: Enforce network security
    rules:
      - no_public_ips: true
      - require_service_endpoints: true
      - allowed_inbound_ports: [443]
```

---

**This production guide provides enterprise-ready patterns for deploying and managing Agent Framework applications at scale.**

For detailed code examples and implementation patterns, refer to:
- `microsoft_agent_framework_comprehensive_guide.md` - Core concepts
- `microsoft_agent_framework_recipes.md` - Practical code examples
- `microsoft_agent_framework_diagrams.md` - Architecture visualisations

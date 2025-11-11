# Microsoft AutoGen 0.4: Production Deployment Guide

## Introduction

This guide covers enterprise-ready patterns, best practices, and operational considerations for deploying AutoGen applications to production environments.

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Deployment Strategies](#deployment-strategies)
3. [Security and Authentication](#security-and-authentication)
4. [Performance Optimisation](#performance-optimisation)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Error Handling and Resilience](#error-handling-and-resilience)
7. [Scalability Patterns](#scalability-patterns)
8. [Cost Management](#cost-management)
9. [High Availability](#high-availability)
10. [Load Balancing](#load-balancing)
11. [Disaster Recovery](#disaster-recovery)
12. [Security Best Practices](#security-best-practices)

---

## Production Architecture

### Reference Architecture for Enterprise Deployments

```yaml
# architecture.yaml
Application Layer:
  - API Gateway
    - Rate limiting
    - Request validation
    - Authentication
  
  - Load Balancer
    - Health checks
    - Auto-scaling
    - Circuit breaking

  - Service Mesh
    - Service discovery
    - Traffic management
    - Security policies

Processing Layer:
  - AutoGen Cluster
    - Multiple agent runtimes
    - Distributed coordination
    - State management

  - Background Job Queue
    - Task scheduling
    - Retry logic
    - Dead letter handling

Data Layer:
  - Primary Database
    - Agent state
    - Conversation history
    - User data
  
  - Cache Layer (Redis)
    - Session cache
    - Model response cache
    - Rate limit counters
  
  - Vector Database
    - Embeddings
    - Semantic search
    - Memory storage

Monitoring Layer:
  - Logging (ELK/Datadog)
  - Metrics (Prometheus)
  - Tracing (Jaeger/Zipkin)
  - Alerting (PagerDuty)
```

### Recommended Infrastructure

**Minimum Production Setup:**
- 2+ replicas of AutoGen service
- Load balancer (L7 aware)
- Managed database (PostgreSQL or Azure Cosmos DB)
- Redis cluster (3+ nodes)
- Centralized logging
- Monitoring with alerts

**Optimal Production Setup:**
- Kubernetes cluster (AKS, EKS, GKE)
- Service mesh (Istio)
- API gateway (Kong, Azure API Management)
- Multi-region deployment
- Disaster recovery site
- Enterprise monitoring stack

---

## Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autogen-agent
  labels:
    app: autogen
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  
  selector:
    matchLabels:
      app: autogen
  
  template:
    metadata:
      labels:
        app: autogen
    
    spec:
      containers:
      - name: autogen
        image: your-registry/autogen:latest
        imagePullPolicy: Always
        
        ports:
        - containerPort: 8000
          name: http
        
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-credentials
              key: api-key
        
        - name: LOG_LEVEL
          value: "INFO"
        
        - name: ENVIRONMENT
          value: "production"
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 2
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/.cache
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
      
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
                  - autogen
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: autogen-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  
  selector:
    app: autogen

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: autogen-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autogen-agent
  
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
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### Azure Container Instances

```python
# deploy_to_azure.py
import asyncio
from azure.identity import DefaultAzureCredential
from azure.containerregistry import ContainerRegistryClient
from azure.containerinstances import ContainerInstancesClient
from azure.containerinstances.models import (
    ContainerGroup, Container, ResourceRequests, ResourceLimits,
    EnvironmentVariable, Port
)

async def deploy_to_azure():
    """Deploy AutoGen to Azure Container Instances"""
    
    credential = DefaultAzureCredential()
    
    # Create container group
    container_group_name = "autogen-prod"
    resource_group = "my-resource-group"
    
    client = ContainerInstancesClient(credential)
    
    # Define container
    container = Container(
        name="autogen",
        image="your-registry.azurecr.io/autogen:latest",
        resources=ResourceRequests(
            cpu=2.0,
            memory_in_gb=4.0,
        ),
        ports=[Port(port=8000)],
        environment_variables=[
            EnvironmentVariable(
                name="OPENAI_API_KEY",
                secure_value="your-key-from-keyvault",
            ),
            EnvironmentVariable(
                name="LOG_LEVEL",
                value="INFO",
            ),
        ],
    )
    
    # Create container group
    container_group = ContainerGroup(
        location="eastus",
        containers=[container],
        os_type="Linux",
        restart_policy="Always",
        ip_address={
            "ports": [{"port": 8000, "protocol": "TCP"}],
            "type": "Public",
            "dns_name_label": "autogen-prod",
        },
    )
    
    # Deploy
    response = client.container_groups.create_or_update(
        resource_group,
        container_group_name,
        container_group,
    )
    
    return response

asyncio.run(deploy_to_azure())
```

---

## Security and Authentication

### API Authentication

```python
# security.py
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

security = HTTPBearer()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        
        return user_id
    
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

def create_access_token(user_id: str, expires_delta: timedelta = None):
    """Create JWT token"""
    if expires_delta is None:
        expires_delta = timedelta(hours=1)
    
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

### API Rate Limiting

```python
# rate_limiting.py
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.util import get_remote_address
from redis.asyncio import Redis
from fastapi import Request

async def init_rate_limiter():
    """Initialize rate limiter with Redis"""
    redis = await Redis.from_url("redis://localhost:6379")
    await FastAPILimiter.init(redis, identifier=get_remote_address)

from fastapi_limiter.depends import RateLimiter

@app.post("/agent/chat")
@limiter.limit("100/minute")
async def chat(request: Request):
    """Chat endpoint with rate limit"""
    pass
```

### Secret Management

```python
# secrets.py
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

class SecretManager:
    """Manage secrets from Azure Key Vault"""
    
    def __init__(self, vault_url: str):
        credential = DefaultAzureCredential()
        self.client = SecretClient(
            vault_url=vault_url,
            credential=credential
        )
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Key Vault"""
        secret = self.client.get_secret(secret_name)
        return secret.value
    
    def get_all_secrets(self, prefix: str) -> dict:
        """Get all secrets with prefix"""
        secrets = {}
        properties = self.client.list_properties_of_secrets()
        
        for secret_property in properties:
            if secret_property.name.startswith(prefix):
                secret = self.client.get_secret(secret_property.name)
                secrets[secret_property.name] = secret.value
        
        return secrets

# Usage
secret_manager = SecretManager(
    vault_url="https://your-keyvault.vault.azure.net/"
)

openai_key = secret_manager.get_secret("openai-api-key")
```

---

## Performance Optimisation

### Model Response Caching

```python
# caching.py
from functools import wraps
import hashlib
import json
from typing import Any, Callable
import asyncio

class ResponseCache:
    """Cache LLM model responses"""
    
    def __init__(self, redis_client, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    def _generate_key(self, prompt: str, model: str, params: dict) -> str:
        """Generate cache key from prompt and parameters"""
        key_data = f"{model}:{prompt}:{json.dumps(params, sort_keys=True)}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"llm_response:{key_hash}"
    
    async def get(self, prompt: str, model: str, params: dict) -> Any:
        """Get cached response"""
        key = self._generate_key(prompt, model, params)
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None
    
    async def set(self, prompt: str, model: str, params: dict, response: Any):
        """Cache response"""
        key = self._generate_key(prompt, model, params)
        await self.redis.setex(
            key,
            self.ttl,
            json.dumps(response)
        )

def cache_llm_response(cache: ResponseCache):
    """Decorator to cache LLM responses"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract cache parameters
            model = kwargs.get("model") or args[0] if args else None
            prompt = kwargs.get("prompt") or args[1] if len(args) > 1 else None
            
            # Try cache first
            if model and prompt:
                cached = await cache.get(prompt, model, kwargs)
                if cached:
                    return cached
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Cache result
            if model and prompt:
                await cache.set(prompt, model, kwargs, result)
            
            return result
        
        return wrapper
    return decorator

# Usage
@cache_llm_response(response_cache)
async def call_model(model: str, prompt: str, temperature: float = 0.7):
    """Call LLM with caching"""
    pass
```

### Connection Pooling

```python
# connections.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Database connection pooling
engine = create_async_engine(
    "postgresql+asyncpg://user:password@host/dbname",
    echo=False,
    pool_size=20,           # Number of persistent connections
    max_overflow=10,        # Additional connections under load
    pool_pre_ping=True,     # Verify connections before using
    pool_recycle=3600,      # Recycle connections hourly
)

async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Redis connection pooling
from redis.asyncio import ConnectionPool, Redis

pool = ConnectionPool.from_url(
    "redis://localhost:6379",
    max_connections=50,
    decode_responses=True,
)

redis_client = Redis(connection_pool=pool)
```

### Agent Concurrency

```python
# concurrency.py
import asyncio
from typing import List
import semaphore

class ConcurrentAgentExecutor:
    """Execute multiple agents concurrently with limits"""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_agent(self, agent, task: str):
        """Execute agent with concurrency limit"""
        async with self.semaphore:
            return await agent.on_messages([
                TextMessage(content=task, source="executor")
            ])
    
    async def execute_many(
        self,
        tasks: List[tuple],  # [(agent, task), ...]
    ) -> List[str]:
        """Execute multiple agent tasks concurrently"""
        coroutines = [
            self.execute_agent(agent, task)
            for agent, task in tasks
        ]
        
        return await asyncio.gather(*coroutines)

# Usage
executor = ConcurrentAgentExecutor(max_concurrent=20)

tasks = [
    (agent1, "Write summary"),
    (agent2, "Analyse data"),
    (agent3, "Generate report"),
]

results = await executor.execute_many(tasks)
```

---

## Monitoring and Observability

### Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, "user_id"):
            log_obj["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        
        return json.dumps(log_obj)

# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger("autogen")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info(
    "Agent processing task",
    extra={
        "user_id": "user-123",
        "request_id": "req-456",
        "agent_name": "planner",
    }
)
```

### Metrics and Monitoring

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
agent_messages_total = Counter(
    'agent_messages_total',
    'Total messages processed by agents',
    ['agent_name', 'status']
)

agent_processing_seconds = Histogram(
    'agent_processing_seconds',
    'Time taken to process messages',
    ['agent_name'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0)
)

llm_calls_total = Counter(
    'llm_calls_total',
    'Total LLM API calls',
    ['model', 'status']
)

llm_tokens_used = Counter(
    'llm_tokens_used',
    'Total tokens used in LLM calls',
    ['model', 'token_type']
)

active_agents = Gauge(
    'active_agents',
    'Number of active agents',
    ['agent_type']
)

# Usage
async def process_with_metrics(agent, task: str):
    """Process task with metrics collection"""
    
    start_time = time.time()
    
    try:
        result = await agent.on_messages([
            TextMessage(content=task, source="user")
        ])
        
        duration = time.time() - start_time
        
        agent_processing_seconds.labels(
            agent_name=agent.name
        ).observe(duration)
        
        agent_messages_total.labels(
            agent_name=agent.name,
            status="success"
        ).inc()
        
        return result
    
    except Exception as e:
        agent_messages_total.labels(
            agent_name=agent.name,
            status="error"
        ).inc()
        raise
```

### Distributed Tracing

```python
# tracing.py
from jaeger_client import Config
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

def init_tracer(service_name: str):
    """Initialize Jaeger tracer"""
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(
        SimpleSpanProcessor(jaeger_exporter)
    )
    
    return trace_provider.get_tracer(service_name)

tracer = init_tracer("autogen-service")

async def trace_agent_execution(agent_name: str, task: str):
    """Trace agent execution"""
    
    with tracer.start_as_current_span(f"agent.{agent_name}") as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("task", task[:100])  # Truncate for spans
        
        # Your agent logic here
        result = await agent.on_messages([
            TextMessage(content=task, source="user")
        ])
        
        span.set_attribute("result.length", len(str(result)))
        
        return result
```

---

## Error Handling and Resilience

### Retry Strategies

```python
# retries.py
import asyncio
from typing import Callable, Any, TypeVar, Optional
import random

T = TypeVar('T')

class RetryConfig:
    """Retry configuration"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            delay *= (0.5 + random.random())
        
        return delay

async def retry_with_config(
    func: Callable[..., T],
    config: RetryConfig,
    *args,
    **kwargs,
) -> T:
    """Retry async function with exponential backoff"""
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        
        except Exception as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                delay = config.calculate_delay(attempt)
                print(f"Retry attempt {attempt + 1} after {delay}s")
                await asyncio.sleep(delay)
    
    raise last_exception

# Usage
async def call_unstable_api():
    """Call API that might fail"""
    result = await retry_with_config(
        agent.on_messages,
        RetryConfig(max_attempts=3),
        messages=[TextMessage(content="task", source="user")]
    )
    return result
```

### Circuit Breaker

```python
# circuit_breaker.py
from enum import Enum
import time
from typing import Callable

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Prevent cascading failures with circuit breaker pattern"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        
        if self.state == CircuitState.OPEN:
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
        
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            
            raise

# Usage
breaker = CircuitBreaker(failure_threshold=3)

async def safe_agent_call():
    """Call agent with circuit breaker protection"""
    return await breaker.call(
        agent.on_messages,
        [TextMessage(content="task", source="user")]
    )
```

---

## Scalability Patterns

### Horizontal Scaling with Message Queues

```python
# message_queue.py
import asyncio
from typing import Any
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

class TaskQueue:
    """Queue tasks for processing by multiple workers"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
    
    async def start(self):
        """Start producer and consumer"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers
        )
        await self.producer.start()
    
    async def stop(self):
        """Stop producer and consumer"""
        await self.producer.stop()
    
    async def enqueue_task(self, topic: str, task: dict):
        """Enqueue a task"""
        await self.producer.send_and_wait(
            topic,
            json.dumps(task).encode()
        )
    
    async def consume_tasks(self, topic: str):
        """Consume tasks from queue"""
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="autogen-workers",
            value_deserializer=lambda m: json.loads(m.decode())
        )
        
        await consumer.start()
        
        try:
            async for message in consumer:
                yield message.value
        
        finally:
            await consumer.stop()

# Usage
queue = TaskQueue()
await queue.start()

# Producer
await queue.enqueue_task("agent_tasks", {
    "agent": "researcher",
    "task": "Research quantum computing",
})

# Consumer (runs on worker instance)
async for task in queue.consume_tasks("agent_tasks"):
    agent = agents[task["agent"]]
    result = await agent.on_messages([
        TextMessage(content=task["task"], source="queue")
    ])
```

---

## Cost Management

### LLM Usage Tracking

```python
# cost_tracking.py
from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class TokenUsage:
    """Track token usage for cost calculation"""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CostTracker:
    """Track and report LLM costs"""
    
    # Pricing per 1K tokens (update with current rates)
    PRICING = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    }
    
    def __init__(self):
        self.usage_log = []
        self.total_cost = 0.0
    
    def log_usage(self, usage: TokenUsage):
        """Log token usage"""
        
        prompt_cost = (usage.prompt_tokens / 1000) * self.PRICING[usage.model]["prompt"]
        completion_cost = (usage.completion_tokens / 1000) * self.PRICING[usage.model]["completion"]
        
        total_cost = prompt_cost + completion_cost
        self.total_cost += total_cost
        
        self.usage_log.append({
            "model": usage.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_cost_summary(self) -> Dict:
        """Get cost summary"""
        return {
            "total_cost": self.total_cost,
            "log_entries": len(self.usage_log),
            "average_cost_per_request": self.total_cost / len(self.usage_log) if self.usage_log else 0,
        }

# Usage in model client wrapper
class CostTrackingOpenAIClient:
    """OpenAI client with cost tracking"""
    
    def __init__(self, base_client, cost_tracker: CostTracker):
        self.base_client = base_client
        self.cost_tracker = cost_tracker
    
    async def create_chat_completion(self, **kwargs):
        """Create completion with cost tracking"""
        
        response = await self.base_client.create_chat_completion(**kwargs)
        
        # Track usage
        usage = response.get("usage", {})
        self.cost_tracker.log_usage(TokenUsage(
            model=kwargs.get("model"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        ))
        
        return response
```

---

## High Availability

### Failover Strategy

```python
# failover.py
from typing import List
import asyncio

class FailoverAgent:
    """Agent with failover to backup models"""
    
    def __init__(
        self,
        primary_client,
        backup_clients: List,
        name: str = "failover_agent",
    ):
        self.primary_client = primary_client
        self.backup_clients = backup_clients
        self.name = name
    
    async def on_messages(self, messages: List, cancellation_token=None):
        """Process messages with failover"""
        
        clients = [self.primary_client] + self.backup_clients
        last_error = None
        
        for i, client in enumerate(clients):
            try:
                print(f"Attempting with client {i}")
                
                # Call agent with current client
                agent = AssistantAgent(
                    name=self.name,
                    model_client=client,
                )
                
                result = await agent.on_messages(messages, cancellation_token)
                return result
            
            except Exception as e:
                last_error = e
                print(f"Client {i} failed: {e}")
                
                if i < len(clients) - 1:
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
        
        raise last_error

# Usage
primary = OpenAIChatCompletionClient(model="gpt-4o")
backup1 = OpenAIChatCompletionClient(model="gpt-4-turbo")
backup2 = AzureOpenAIChatCompletionClient(model="gpt-4o", endpoint="...")

failover_agent = FailoverAgent(
    primary_client=primary,
    backup_clients=[backup1, backup2],
    name="failover_assistant"
)
```

---

## Load Balancing

### Agent Load Distribution

```python
# load_balancing.py
from typing import List
import random

class LoadBalancedTeam:
    """Distribute tasks across multiple agent instances"""
    
    def __init__(self, agents: List[AssistantAgent]):
        self.agents = agents
        self.agent_loads = {agent.name: 0 for agent in agents}
    
    def get_least_loaded_agent(self) -> AssistantAgent:
        """Get agent with lowest current load"""
        agent_name = min(
            self.agent_loads,
            key=self.agent_loads.get
        )
        return next(a for a in self.agents if a.name == agent_name)
    
    async def process_task(self, task: str):
        """Process task on least loaded agent"""
        
        agent = self.get_least_loaded_agent()
        self.agent_loads[agent.name] += 1
        
        try:
            result = await agent.on_messages([
                TextMessage(content=task, source="load_balancer")
            ])
            return result
        
        finally:
            self.agent_loads[agent.name] -= 1
```

---

## Disaster Recovery

### Backup and Restore

```python
# disaster_recovery.py
import asyncio
import json
from datetime import datetime
import aiofiles

class DisasterRecovery:
    """Backup and restore agent state"""
    
    def __init__(self, backup_path: str = "./backups"):
        self.backup_path = backup_path
    
    async def backup_agent_state(self, agent_name: str, state: dict):
        """Backup agent state"""
        
        timestamp = datetime.now().isoformat()
        backup_file = f"{self.backup_path}/{agent_name}_{timestamp}.json"
        
        async with aiofiles.open(backup_file, "w") as f:
            await f.write(json.dumps(state, indent=2))
        
        print(f"Backup created: {backup_file}")
    
    async def restore_agent_state(self, agent_name: str, timestamp: str) -> dict:
        """Restore agent state from backup"""
        
        backup_file = f"{self.backup_path}/{agent_name}_{timestamp}.json"
        
        async with aiofiles.open(backup_file, "r") as f:
            content = await f.read()
        
        return json.loads(content)
    
    async def create_database_backup(self, db_connection):
        """Create database backup"""
        
        timestamp = datetime.now().isoformat()
        backup_file = f"{self.backup_path}/db_backup_{timestamp}.sql"
        
        # Execute backup command (implementation specific to DB)
        # This is a placeholder
        print(f"Database backup: {backup_file}")

# Usage
dr = DisasterRecovery()

# Create backup
await dr.backup_agent_state("planner", {
    "messages": [...],
    "state": {...},
})

# Restore from backup
state = await dr.restore_agent_state("planner", "2024-01-15T10:30:00")
```

---

**Production Deployment Best Practices Summary:**

✅ Always use separate environments (dev, staging, prod)
✅ Implement comprehensive logging and monitoring
✅ Use automated deployments (CI/CD)
✅ Have disaster recovery procedures
✅ Implement security best practices
✅ Monitor costs and optimise usage
✅ Plan for scalability from the start
✅ Use infrastructure-as-code
✅ Implement proper error handling
✅ Test failover procedures regularly

---


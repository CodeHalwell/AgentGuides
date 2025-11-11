# Google Agent Development Kit (ADK) - Production Guide

**Version:** 1.0  
**Focus:** Enterprise Deployment, Reliability, Performance, Security

---

## Table of Contents

1. [Production Deployment Patterns](#production-deployment-patterns)
2. [Scalability and Performance](#scalability-and-performance)
3. [Reliability and Fault Tolerance](#reliability-and-fault-tolerance)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Cost Optimisation](#cost-optimisation)
6. [Security Best Practices](#security-best-practices)
7. [Testing Strategies](#testing-strategies)
8. [CI/CD Pipeline with Cloud Build](#cicd-pipeline-with-cloud-build)
9. [Enterprise Patterns](#enterprise-patterns)
10. [Disaster Recovery](#disaster-recovery)
11. [Performance Tuning](#performance-tuning)
12. [SLA Management](#sla-management)

---

## Production Deployment Patterns

### Cloud Run Deployment

#### Multi-Region Deployment

```bash
#!/bin/bash
set -e

PROJECT_ID="production-project"
SERVICE_NAME="adk-agents"
REGIONS=("us-central1" "europe-west1" "asia-southeast1")

# Build image
gcloud builds submit \
  --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --project ${PROJECT_ID}

# Deploy to multiple regions
for REGION in "${REGIONS[@]}"; do
  echo "Deploying to ${REGION}..."
  gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 4 \
    --max-instances 100 \
    --min-instances 10 \
    --timeout 900 \
    --project ${PROJECT_ID} \
    --set-env-vars \
      GOOGLE_PROJECT_ID=${PROJECT_ID},\
      REGION=${REGION},\
      ENVIRONMENT=production
done

# Set up load balancer
gcloud compute backend-services create ${SERVICE_NAME}-backend \
  --protocol=HTTPS \
  --global

for REGION in "${REGIONS[@]}"; do
  NEG_NAME="${SERVICE_NAME}-neg-${REGION}"
  gcloud compute network-endpoint-groups create ${NEG_NAME} \
    --region=${REGION} \
    --network-endpoint-type=SERVERLESS \
    --cloud-run-service=${SERVICE_NAME}
  
  gcloud compute backend-services add-backend ${SERVICE_NAME}-backend \
    --instance-group=${NEG_NAME} \
    --instance-group-region=${REGION} \
    --global
done
```

#### Production-Ready Docker Configuration

```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Add non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy built packages
COPY --from=builder /root/.local /home/appuser/.local

# Copy application
COPY --chown=appuser:appuser . .

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Vertex AI Agent Engine Deployment

```python
# Deploy agent to Vertex AI Agent Engine
import vertexai
from google.cloud import aiplatform_v1

def deploy_agent_to_vertex():
    """Deploy ADK agent to Vertex AI Agent Engine."""
    
    vertexai.init(project="production-project", location="us-central1")
    
    # Create agent configuration
    agent_config = {
        "display_name": "Production ADK Agent",
        "instructions": """You are a production agent responsible for:
        - Processing customer requests
        - Maintaining system stability
        - Providing accurate information
        - Handling errors gracefully""",
        "model": "gemini-2.5-pro",
        "tools": [
            # Tool definitions
        ],
        "temperature": 0.3,  # Lower temp for consistency
        "top_p": 0.9,
        "max_output_tokens": 2048
    }
    
    # Deploy
    client = aiplatform_v1.AgentsClient()
    agent = client.create_agent(
        parent="projects/production-project/locations/us-central1",
        agent=agent_config
    )
    
    return agent

# Deploy with auto-scaling
deployment_config = {
    "machine_type": "n2-highmem-4",
    "replicas": {
        "min": 3,
        "max": 20
    },
    "autoscaling_metric": "cpu_utilization",
    "autoscaling_target": 70
}
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adk-agent-deployment
  namespace: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: adk-agent
  template:
    metadata:
      labels:
        app: adk-agent
        version: v1.0.0
    spec:
      serviceAccountName: adk-agent-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: agent
        image: gcr.io/production-project/adk-agent:latest
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        
        env:
        - name: GOOGLE_PROJECT_ID
          valueFrom:
            configMapKeyRef:
              name: adk-config
              key: project-id
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: /var/secrets/google/key.json
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        volumeMounts:
        - name: google-cloud-key
          mountPath: /var/secrets/google
      
      volumes:
      - name: google-cloud-key
        secret:
          secretName: adk-agent-key
      
      nodeSelector:
        workload: agents
      
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
                  - adk-agent
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: adk-agent-service
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: adk-agent
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: adk-agent-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: adk-agent-deployment
  minReplicas: 5
  maxReplicas: 50
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

---

## Scalability and Performance

### Load Testing and Benchmarking

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from google.adk import Agent, Runner
from google.adk.sessions import FirestoreSessionService
from google.genai import types

class LoadTestSuite:
    """Comprehensive load testing for ADK agents."""
    
    def __init__(self, agent: Agent, concurrent_users: int = 100):
        self.agent = agent
        self.concurrent_users = concurrent_users
        self.session_service = FirestoreSessionService(project_id="test-project")
        self.results = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0,
            "min_latency": float('inf'),
            "max_latency": 0,
            "avg_latency": 0
        }
    
    async def test_single_request(self, user_id: str, query: str) -> dict:
        """Test single agent request."""
        start_time = time.time()
        try:
            runner = Runner(
                app_name="load_test_app",
                agent=self.agent,
                session_service=self.session_service
            )
            
            message = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )
            
            async for event in runner.run_async(
                user_id=user_id,
                session_id=f"session_{user_id}",
                new_message=message
            ):
                if event.content:
                    pass  # Process response
            
            latency = time.time() - start_time
            return {
                "status": "success",
                "latency": latency
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    async def run_load_test(self, duration_seconds: int = 60):
        """Run sustained load test."""
        import random
        
        test_queries = [
            "What is 2+2?",
            "List the steps to learn Python",
            "Explain machine learning",
            "How does cloud computing work?",
            "What are microservices?"
        ]
        
        start_time = time.time()
        tasks = []
        
        while time.time() - start_time < duration_seconds:
            for i in range(self.concurrent_users):
                user_id = f"load_test_user_{i}"
                query = random.choice(test_queries)
                task = self.test_single_request(user_id, query)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            self._update_results(results)
            tasks = []
            await asyncio.sleep(1)
        
        self._print_results()
    
    def _update_results(self, test_results: list):
        """Update test results."""
        for result in test_results:
            self.results["total_requests"] += 1
            
            if result["status"] == "success":
                self.results["successful"] += 1
                latency = result["latency"]
            else:
                self.results["failed"] += 1
                latency = result["latency"]
            
            self.results["total_time"] += latency
            self.results["min_latency"] = min(self.results["min_latency"], latency)
            self.results["max_latency"] = max(self.results["max_latency"], latency)
    
    def _print_results(self):
        """Print test results."""
        avg_latency = self.results["total_time"] / max(self.results["total_requests"], 1)
        
        print(f"""
        === Load Test Results ===
        Total Requests: {self.results["total_requests"]}
        Successful: {self.results["successful"]}
        Failed: {self.results["failed"]}
        Success Rate: {(self.results["successful"]/max(self.results["total_requests"], 1)*100):.2f}%
        
        Min Latency: {self.results["min_latency"]*1000:.2f}ms
        Max Latency: {self.results["max_latency"]*1000:.2f}ms
        Avg Latency: {avg_latency*1000:.2f}ms
        """)

# Usage
async def run_benchmarks():
    agent = Agent(
        name="benchmark_agent",
        model="gemini-2.5-flash",
        instruction="Respond quickly and accurately"
    )
    
    tester = LoadTestSuite(agent, concurrent_users=50)
    await tester.run_load_test(duration_seconds=300)

# asyncio.run(run_benchmarks())
```

### Connection Pooling

```python
from google.cloud import firestore
from google.adk.sessions import FirestoreSessionService
import asyncio

class PooledSessionService:
    """Session service with connection pooling."""
    
    def __init__(self, project_id: str, pool_size: int = 10):
        self.project_id = project_id
        self.pool_size = pool_size
        self.session_services = [
            FirestoreSessionService(project_id=project_id)
            for _ in range(pool_size)
        ]
        self.current_index = 0
        self.lock = asyncio.Lock()
    
    async def get_service(self):
        """Get next service from pool."""
        async with self.lock:
            service = self.session_services[self.current_index]
            self.current_index = (self.current_index + 1) % self.pool_size
            return service
    
    async def create_session(self, app_name: str, user_id: str, state: dict = None):
        """Create session using pooled service."""
        service = await self.get_service()
        return await service.create_session(app_name, user_id, state or {})
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib
from typing import Optional
import json

class ResponseCache:
    """Cache agent responses for common queries."""
    
    def __init__(self, max_cache_size: int = 10000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}
    
    def _get_cache_key(self, query: str, agent_id: str) -> str:
        """Generate cache key for query."""
        key_str = f"{agent_id}:{query}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, query: str, agent_id: str) -> Optional[str]:
        """Retrieve cached response."""
        import time
        
        key = self._get_cache_key(query, agent_id)
        
        if key in self.cache:
            if time.time() - self.access_times[key] < self.ttl_seconds:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    async def set(self, query: str, agent_id: str, response: str) -> None:
        """Cache response."""
        import time
        
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        key = self._get_cache_key(query, agent_id)
        self.cache[key] = response
        self.access_times[key] = time.time()

# Usage with agent
cache = ResponseCache(max_cache_size=5000, ttl_seconds=3600)

async def cached_agent_call(agent, query: str):
    """Call agent with caching."""
    cached_response = await cache.get(query, agent.name)
    
    if cached_response:
        return cached_response
    
    # Call agent (implementation)
    response = "Agent response"
    
    await cache.set(query, agent.name, response)
    return response
```

---

## Reliability and Fault Tolerance

### Retry Logic with Exponential Backoff

```python
import asyncio
from typing import Callable, TypeVar, Optional
import random

T = TypeVar('T')

class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

async def retry_with_backoff(
    func: Callable,
    *args,
    config: RetryConfig = None,
    **kwargs
) -> T:
    """Execute function with exponential backoff retry."""
    config = config or RetryConfig()
    
    last_exception = None
    delay = config.initial_delay
    
    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        
        except Exception as e:
            last_exception = e
            
            if attempt == config.max_retries:
                raise
            
            # Calculate delay
            if config.jitter:
                jitter = random.uniform(0, 0.1 * delay)
                actual_delay = min(delay + jitter, config.max_delay)
            else:
                actual_delay = min(delay, config.max_delay)
            
            print(f"Attempt {attempt + 1} failed. Retrying in {actual_delay:.2f}s...")
            await asyncio.sleep(actual_delay)
            
            # Exponential backoff
            delay *= config.exponential_base

# Usage
async def call_agent():
    from google.adk import Agent, Runner
    from google.adk.sessions import InMemorySessionService
    
    agent = Agent(name="test", model="gemini-2.5-flash", instruction="Test")
    runner = Runner(
        app_name="retry_test",
        agent=agent,
        session_service=InMemorySessionService()
    )
    
    config = RetryConfig(max_retries=5, initial_delay=2.0)
    
    result = await retry_with_backoff(
        runner.run_async,
        user_id="user123",
        session_id="session1",
        config=config
    )
    
    return result
```

### Circuit Breaker Pattern

```python
from enum import Enum
import asyncio
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure > timedelta(seconds=self.recovery_timeout)

# Usage
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

async def resilient_agent_call(agent, query):
    async def call():
        # Agent call logic
        pass
    
    return await breaker.call(call)
```

### Graceful Degradation

```python
class GracefulDegradation:
    """Provide fallback responses when agent fails."""
    
    def __init__(self, agent, fallback_responses: dict = None):
        self.agent = agent
        self.fallback_responses = fallback_responses or {}
    
    async def execute(self, query: str) -> str:
        """Execute with fallback."""
        try:
            # Try agent
            response = await self._call_agent(query)
            return response
        
        except Exception as e:
            print(f"Agent failed: {e}. Using fallback...")
            
            # Return fallback response
            return self.fallback_responses.get(
                query,
                "I apologize, but I'm temporarily unavailable. Please try again later."
            )
    
    async def _call_agent(self, query: str) -> str:
        """Call agent (implementation)."""
        pass

# Usage
degradation = GracefulDegradation(
    agent=my_agent,
    fallback_responses={
        "What is your name?": "I'm an ADK agent.",
        "How are you?": "I'm operating normally."
    }
)

response = await degradation.execute("What is your name?")
```

---

## Monitoring and Observability

### Cloud Logging Integration

```python
import google.cloud.logging
import logging

def setup_cloud_logging(project_id: str):
    """Set up Cloud Logging for ADK agents."""
    
    # Set up logging client
    logging_client = google.cloud.logging.Client(project=project_id)
    logging_client.setup_logging()
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    return logger

# Usage in agent code
logger = setup_cloud_logging("production-project")

def log_agent_execution(agent_name: str, user_id: str, query: str, response: str, latency: float):
    """Log agent execution."""
    
    logger.info(
        "Agent executed",
        extra={
            "labels": {
                "agent_name": agent_name,
                "user_id": user_id,
                "environment": "production"
            },
            "json_fields": {
                "query": query,
                "response": response,
                "latency_ms": latency * 1000
            }
        }
    )

# Structured logging example
def log_agent_error(agent_name: str, error: Exception, context: dict):
    """Log agent error."""
    
    logger.error(
        f"Agent {agent_name} failed",
        extra={
            "labels": {
                "severity": "ERROR",
                "agent_name": agent_name
            },
            "json_fields": {
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context
            }
        }
    )
```

### Custom Metrics with Cloud Monitoring

```python
from google.cloud import monitoring_v3
import time

class AgentMetricsCollector:
    """Collect custom metrics for ADK agents."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.metric_client = monitoring_v3.MetricServiceClient()
        self.query_client = monitoring_v3.QueryServiceClient()
    
    def record_latency(self, agent_name: str, latency_ms: float):
        """Record agent latency metric."""
        
        time_series = monitoring_v3.TimeSeries()
        time_series.metric.type = "custom.googleapis.com/adk/agent_latency"
        time_series.metric.labels["agent_name"] = agent_name
        
        now = time.time()
        point = monitoring_v3.Point({
            "interval": {
                "end_time": {"seconds": int(now)}
            },
            "value": {
                "double_value": latency_ms
            }
        })
        
        time_series.points = [point]
        
        self.metric_client.create_time_series(
            name=f"projects/{self.project_id}",
            time_series=[time_series]
        )
    
    def record_errors(self, agent_name: str):
        """Record agent error count."""
        
        time_series = monitoring_v3.TimeSeries()
        time_series.metric.type = "custom.googleapis.com/adk/agent_errors"
        time_series.metric.labels["agent_name"] = agent_name
        
        now = time.time()
        point = monitoring_v3.Point({
            "interval": {
                "end_time": {"seconds": int(now)}
            },
            "value": {
                "int64_value": 1
            }
        })
        
        time_series.points = [point]
        
        self.metric_client.create_time_series(
            name=f"projects/{self.project_id}",
            time_series=[time_series]
        )

# Usage
metrics = AgentMetricsCollector(project_id="production-project")

async def monitored_agent_call(agent, query: str):
    start_time = time.time()
    try:
        # Call agent
        response = "result"
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_latency(agent.name, latency_ms)
        return response
    
    except Exception as e:
        metrics.record_errors(agent.name)
        raise
```

### Distributed Tracing

```python
from google.cloud import trace_v2
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.gcp_trace import CloudTraceExporter

def setup_distributed_tracing(project_id: str):
    """Set up distributed tracing for ADK."""
    
    # Create tracer provider
    tracer_provider = TracerProvider()
    
    # Add Cloud Trace exporter
    cloud_trace_exporter = CloudTraceExporter(project_id=project_id)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(cloud_trace_exporter)
    )
    
    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    return trace.get_tracer(__name__)

# Usage
tracer = setup_distributed_tracing("production-project")

async def traced_agent_call(agent, query: str):
    """Execute agent with tracing."""
    
    with tracer.start_as_current_span("agent_call") as span:
        span.set_attribute("agent.name", agent.name)
        span.set_attribute("query", query)
        
        try:
            # Call agent
            response = "result"
            span.set_attribute("status", "success")
            return response
        
        except Exception as e:
            span.set_attribute("status", "error")
            span.set_attribute("error.type", type(e).__name__)
            raise
```

---

## Cost Optimisation

### Token Usage Tracking

```python
from typing import Dict
import json
from datetime import datetime

class TokenUsageTracker:
    """Track token usage for cost analysis."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.usage_log = []
        self.daily_summary = {}
    
    def log_token_usage(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cost: float
    ):
        """Log token usage."""
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "cost": cost
        }
        
        self.usage_log.append(entry)
        
        # Update daily summary
        today = datetime.now().date().isoformat()
        if today not in self.daily_summary:
            self.daily_summary[today] = {
                "total_tokens": 0,
                "total_cost": 0,
                "requests": 0,
                "by_agent": {}
            }
        
        summary = self.daily_summary[today]
        summary["total_tokens"] += entry["total_tokens"]
        summary["total_cost"] += cost
        summary["requests"] += 1
        
        if agent_name not in summary["by_agent"]:
            summary["by_agent"][agent_name] = {
                "tokens": 0,
                "cost": 0,
                "requests": 0
            }
        
        summary["by_agent"][agent_name]["tokens"] += entry["total_tokens"]
        summary["by_agent"][agent_name]["cost"] += cost
        summary["by_agent"][agent_name]["requests"] += 1
    
    def get_cost_estimate(self, days: int = 7) -> Dict:
        """Get cost estimate for period."""
        
        total_cost = sum(
            summary["total_cost"]
            for summary in list(self.daily_summary.values())[-days:]
        )
        
        avg_daily_cost = total_cost / max(days, 1)
        estimated_monthly = avg_daily_cost * 30
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "avg_daily_cost": avg_daily_cost,
            "estimated_monthly": estimated_monthly
        }

# Model pricing configuration
PRICING = {
    "gemini-2.5-flash": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000
    },
    "gemini-2.5-pro": {
        "input": 3.0 / 1_000_000,
        "output": 12.0 / 1_000_000
    }
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for tokens."""
    
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")
    
    pricing = PRICING[model]
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

# Usage
tracker = TokenUsageTracker("production-project")

async def track_agent_cost(agent, query: str):
    """Execute agent and track cost."""
    
    # Simulate response with token counts
    response = await agent.execute(query)
    
    cost = calculate_cost(
        model=agent.model,
        input_tokens=len(query.split()) * 2,  # Rough estimate
        output_tokens=len(response.split()) * 2
    )
    
    tracker.log_token_usage(
        agent_name=agent.name,
        input_tokens=len(query.split()) * 2,
        output_tokens=len(response.split()) * 2,
        model=agent.model,
        cost=cost
    )
    
    return response
```

### Model Selection Strategy

```python
class CostOptimizedModelSelector:
    """Select optimal model based on requirements and cost."""
    
    MODELS = {
        "gemini-2.5-flash": {"cost": 1, "quality": 8, "speed": 10},
        "gemini-2.5-pro": {"cost": 40, "quality": 10, "speed": 7},
        "gemini-1.5-flash": {"cost": 1, "quality": 7, "speed": 10}
    }
    
    @staticmethod
    def select_model(
        max_cost: float = 1.0,
        min_quality: int = 7,
        min_speed: int = 7
    ) -> str:
        """Select model matching criteria."""
        
        candidates = []
        
        for model, metrics in CostOptimizedModelSelector.MODELS.items():
            if (metrics["cost"] <= max_cost and
                metrics["quality"] >= min_quality and
                metrics["speed"] >= min_speed):
                candidates.append((model, metrics["cost"]))
        
        if not candidates:
            return "gemini-2.5-pro"  # Fallback
        
        # Return cheapest candidate
        return min(candidates, key=lambda x: x[1])[0]

# Usage
selector = CostOptimizedModelSelector()

# For simple queries - prioritise cost
simple_model = selector.select_model(max_cost=5, min_quality=5)

# For complex tasks - prioritise quality
complex_model = selector.select_model(max_cost=100, min_quality=9)
```

---

## Security Best Practices

### Input Validation and Sanitisation

```python
import re
from typing import Optional

class InputValidator:
    """Validate and sanitise user inputs."""
    
    def __init__(self):
        self.max_length = 10000
        self.max_tokens = 2000
        self.sql_injection_pattern = re.compile(r"(SELECT|INSERT|UPDATE|DELETE)", re.IGNORECASE)
        self.xss_pattern = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
    
    def validate_input(self, input_text: str) -> tuple[bool, Optional[str]]:
        """Validate input for security."""
        
        # Check length
        if len(input_text) > self.max_length:
            return False, "Input exceeds maximum length"
        
        # Check for SQL injection patterns
        if self.sql_injection_pattern.search(input_text):
            return False, "Potentially malicious SQL detected"
        
        # Check for XSS patterns
        if self.xss_pattern.search(input_text):
            return False, "Potentially malicious script detected"
        
        # Check for empty input
        if not input_text.strip():
            return False, "Input cannot be empty"
        
        return True, None
    
    def sanitise_input(self, input_text: str) -> str:
        """Remove potentially harmful characters."""
        
        # Remove HTML tags
        sanitised = re.sub(r"<[^>]+>", "", input_text)
        
        # Remove potentially harmful unicode characters
        sanitised = "".join(
            c for c in sanitised
            if ord(c) < 1114112 and c.isprintable()
        )
        
        return sanitised

# Usage
validator = InputValidator()

async def secure_agent_call(agent, user_input: str):
    """Call agent with input validation."""
    
    is_valid, error_msg = validator.validate_input(user_input)
    
    if not is_valid:
        return {"error": error_msg}
    
    sanitised_input = validator.sanitise_input(user_input)
    
    # Call agent with sanitised input
    return await agent.execute(sanitised_input)
```

### Access Control

```python
from enum import Enum
from typing import Set

class AgentPermission(Enum):
    """Permissions for agent operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"

class AccessControl:
    """Implement role-based access control."""
    
    def __init__(self):
        self.user_roles = {}
        self.role_permissions = {
            "admin": {
                AgentPermission.READ,
                AgentPermission.WRITE,
                AgentPermission.DELETE,
                AgentPermission.EXECUTE
            },
            "user": {
                AgentPermission.READ,
                AgentPermission.EXECUTE
            },
            "viewer": {
                AgentPermission.READ
            }
        }
    
    def assign_role(self, user_id: str, role: str):
        """Assign role to user."""
        if role not in self.role_permissions:
            raise ValueError(f"Unknown role: {role}")
        self.user_roles[user_id] = role
    
    def check_permission(
        self,
        user_id: str,
        permission: AgentPermission
    ) -> bool:
        """Check if user has permission."""
        
        if user_id not in self.user_roles:
            return False
        
        role = self.user_roles[user_id]
        return permission in self.role_permissions.get(role, set())
    
    def require_permission(self, user_id: str, permission: AgentPermission):
        """Decorator to require permission."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.check_permission(user_id, permission):
                    raise PermissionError(f"User {user_id} lacks {permission.value} permission")
                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Usage
access_control = AccessControl()
access_control.assign_role("user123", "admin")
access_control.assign_role("user456", "user")

# Check permission
if access_control.check_permission("user123", AgentPermission.DELETE):
    print("User can delete")
```

### Rate Limiting

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    """Implement rate limiting for agent access."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_log = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limit."""
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Remove old requests
        self.request_log[user_id] = [
            req_time for req_time in self.request_log[user_id]
            if req_time > one_minute_ago
        ]
        
        # Check limit
        if len(self.request_log[user_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.request_log[user_id].append(now)
        return True
    
    def get_remaining(self, user_id: str) -> int:
        """Get remaining requests for user."""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        recent_requests = [
            req_time for req_time in self.request_log[user_id]
            if req_time > one_minute_ago
        ]
        
        return max(0, self.requests_per_minute - len(recent_requests))

# Usage
limiter = RateLimiter(requests_per_minute=60)

async def rate_limited_agent_call(agent, user_id: str, query: str):
    """Call agent with rate limiting."""
    
    if not limiter.is_allowed(user_id):
        remaining = limiter.get_remaining(user_id)
        return {"error": f"Rate limit exceeded. Try again in 1 minute."}
    
    return await agent.execute(query)
```

---

## Testing Strategies

### Unit Testing

```python
import pytest
from google.adk import Agent

@pytest.fixture
def test_agent():
    """Create test agent."""
    return Agent(
        name="test_agent",
        model="gemini-2.5-flash",
        instruction="You are a helpful test agent"
    )

def test_agent_creation(test_agent):
    """Test agent creation."""
    assert test_agent.name == "test_agent"
    assert test_agent.model == "gemini-2.5-flash"

def test_agent_initialization():
    """Test agent initialization with tools."""
    def test_tool(input_str: str) -> str:
        return f"Result: {input_str}"
    
    agent = Agent(
        name="tool_agent",
        model="gemini-2.5-flash",
        tools=[test_tool],
        instruction="Use the test tool"
    )
    
    assert len(agent.tools) == 1

@pytest.mark.asyncio
async def test_agent_execution(test_agent):
    """Test agent execution."""
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    
    runner = Runner(
        app_name="test",
        agent=test_agent,
        session_service=InMemorySessionService()
    )
    
    message = types.Content(
        role='user',
        parts=[types.Part(text="What is 2+2?")]
    )
    
    response = None
    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=message
    ):
        if event.content:
            response = event.content
    
    assert response is not None
```

### Integration Testing

```python
import pytest
from google.cloud import firestore

@pytest.fixture
def firestore_db():
    """Firestore test instance."""
    return firestore.Client()

@pytest.mark.asyncio
async def test_agent_with_firestore(firestore_db):
    """Test agent with Firestore session management."""
    from google.adk.sessions import FirestoreSessionService
    from google.adk import Agent, Runner
    
    session_service = FirestoreSessionService(
        project_id="test-project",
        db=firestore_db
    )
    
    agent = Agent(
        name="test_agent",
        model="gemini-2.5-flash",
        instruction="Test agent"
    )
    
    # Create session
    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        state={"test": "data"}
    )
    
    assert session.user_id == "test_user"
    assert session.state["test"] == "data"
    
    # Retrieve session
    retrieved = await session_service.get_session(
        app_name="test_app",
        user_id="test_user",
        session_id=session.id
    )
    
    assert retrieved.id == session.id
```

### Performance Testing

```python
import pytest
import time
import asyncio

@pytest.mark.asyncio
async def test_agent_latency():
    """Test agent response latency."""
    from google.adk import Agent
    
    agent = Agent(
        name="perf_test",
        model="gemini-2.5-flash",
        instruction="Respond quickly"
    )
    
    start = time.time()
    
    # Execute agent (simplified)
    result = await agent.execute("Test query")
    
    elapsed = time.time() - start
    
    # Should respond within 5 seconds
    assert elapsed < 5.0, f"Latency {elapsed}s exceeds threshold"

@pytest.mark.asyncio
async def test_concurrent_agent_calls():
    """Test agent under concurrent load."""
    from google.adk import Agent
    
    agent = Agent(
        name="concurrent_test",
        model="gemini-2.5-flash",
        instruction="Test agent"
    )
    
    # Simulate 10 concurrent calls
    tasks = [
        agent.execute(f"Query {i}")
        for i in range(10)
    ]
    
    start = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    assert len(results) == 10
    # Should complete reasonably
    assert elapsed < 60.0
```

---

## CI/CD Pipeline with Cloud Build

### Cloud Build Configuration

```yaml
# cloudbuild.yaml
steps:
  # Step 1: Run tests
  - name: 'gcr.io/cloud-builders/python'
    id: 'run-tests'
    entrypoint: bash
    args:
      - -c
      - |
        pip install -r requirements-dev.txt
        python -m pytest tests/ -v --cov=src

  # Step 2: Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    id: 'build-image'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/adk-agent:$SHORT_SHA'
      - '-t'
      - 'gcr.io/$PROJECT_ID/adk-agent:latest'
      - '.'

  # Step 3: Push image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    id: 'push-image'
    args:
      - 'push'
      - 'gcr.io/$PROJECT_ID/adk-agent:$SHORT_SHA'

  # Step 4: Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gke-deploy'
    id: 'deploy-cloud-run'
    args:
      - run
      - deploy
      - adk-agent
      - '--image'
      - 'gcr.io/$PROJECT_ID/adk-agent:$SHORT_SHA'
      - '--platform'
      - managed
      - '--region'
      - us-central1
      - '--allow-unauthenticated'

  # Step 5: Run smoke tests
  - name: 'gcr.io/cloud-builders/python'
    id: 'smoke-tests'
    entrypoint: bash
    args:
      - -c
      - |
        pip install requests
        python tests/smoke_tests.py

images:
  - 'gcr.io/$PROJECT_ID/adk-agent:$SHORT_SHA'
  - 'gcr.io/$PROJECT_ID/adk-agent:latest'

timeout: '3600s'
options:
  machineType: 'N1_HIGHCPU_8'
  logging: CLOUD_LOGGING_ONLY
```

---

## Enterprise Patterns

### Multi-Tenant Agent Architecture

```python
class MultiTenantAgentManager:
    """Manage agents for multiple tenants."""
    
    def __init__(self):
        self.tenant_agents = {}
    
    def register_tenant_agent(self, tenant_id: str, agent: Agent):
        """Register agent for tenant."""
        self.tenant_agents[tenant_id] = agent
    
    async def execute_for_tenant(
        self,
        tenant_id: str,
        query: str,
        user_id: str
    ) -> str:
        """Execute agent for specific tenant."""
        
        if tenant_id not in self.tenant_agents:
            raise ValueError(f"No agent registered for tenant {tenant_id}")
        
        agent = self.tenant_agents[tenant_id]
        
        # Execute with tenant-specific context
        response = await agent.execute(
            f"[Tenant: {tenant_id}] {query}"
        )
        
        return response
    
    def get_tenant_config(self, tenant_id: str) -> dict:
        """Get configuration for tenant."""
        return {
            "tenant_id": tenant_id,
            "agent_name": self.tenant_agents.get(tenant_id).name,
            "model": self.tenant_agents.get(tenant_id).model
        }

# Usage
manager = MultiTenantAgentManager()

# Register agents for different tenants
manager.register_tenant_agent("tenant1", agent1)
manager.register_tenant_agent("tenant2", agent2)

# Execute for specific tenant
response = await manager.execute_for_tenant(
    tenant_id="tenant1",
    query="What's my balance?",
    user_id="user123"
)
```

---

*This production guide provides comprehensive patterns and practices for deploying ADK agents at scale. Refer to specific sections for detailed implementation guidance.*


# Pydantic AI: Production Deployment & Architecture Guide

**Version:** 1.0.0  
**Focus:** Enterprise-grade deployment, scaling, monitoring, and architectural patterns

---

## Production Architecture Patterns

### Multi-Tier Agent Architecture

```python
"""
Enterprise-grade agent system with three tiers:
1. API Layer - FastAPI endpoints
2. Agent Layer - Business logic agents
3. Infrastructure Layer - Databases, caches, external services
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pydantic_ai import Agent
from dataclasses import dataclass
import logging

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ INFRASTRUCTURE LAYER ============

@dataclass
class InfrastructureDependencies:
    """All infrastructure dependencies."""
    database_url: str
    redis_url: str
    api_keys: dict
    logger: logging.Logger
    environment: str  # 'development', 'staging', 'production'

async def get_infrastructure() -> InfrastructureDependencies:
    """Factory for infrastructure dependencies."""
    import os
    return InfrastructureDependencies(
        database_url=os.getenv('DATABASE_URL', 'postgresql://localhost/agent_db'),
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
        api_keys={'openai': os.getenv('OPENAI_API_KEY')},
        logger=logger,
        environment=os.getenv('ENVIRONMENT', 'development')
    )

# ============ AGENT LAYER ============

class AgentProvider:
    """Factory for creating configured agents."""
    
    @staticmethod
    def create_support_agent(deps: InfrastructureDependencies) -> Agent:
        """Create support/customer service agent."""
        return Agent(
            'openai:gpt-4o',
            deps_type=InfrastructureDependencies,
            instructions='''
            You are a professional customer support agent.
            - Always be polite and professional
            - Prioritize customer satisfaction
            - Escalate complex issues appropriately
            - Log all interactions for quality assurance
            ''',
            name='SupportAgent'
        )
    
    @staticmethod
    def create_data_analyst_agent(deps: InfrastructureDependencies) -> Agent:
        """Create data analysis agent."""
        return Agent(
            'openai:gpt-4o',
            deps_type=InfrastructureDependencies,
            instructions='''
            You are a data analysis expert.
            - Provide accurate insights from data
            - Explain findings clearly
            - Suggest actionable recommendations
            - Use visualizations when helpful
            ''',
            name='DataAnalystAgent'
        )

# ============ API LAYER ============

app = FastAPI(
    title='Pydantic AI Enterprise',
    version='1.0.0',
    docs_url='/api/docs'
)

class SupportRequest(BaseModel):
    """Type-safe request model."""
    customer_id: int
    issue: str
    priority: str = 'normal'  # normal, high, critical

class AgentResponse(BaseModel):
    """Type-safe response model."""
    response: str
    suggested_actions: list[str]
    escalated: bool
    tokens_used: int

@app.post('/api/support/query', response_model=AgentResponse)
async def support_query(
    request: SupportRequest,
    deps: InfrastructureDependencies = Depends(get_infrastructure)
) -> AgentResponse:
    """
    Handle customer support queries through agent.
    
    Type-safe request/response with full validation.
    """
    try:
        agent = AgentProvider.create_support_agent(deps)
        
        result = await agent.run(
            f"Customer {request.customer_id}: {request.issue}",
            deps=deps
        )
        
        return AgentResponse(
            response=result.output,
            suggested_actions=['escalate', 'provide_docs'],
            escalated=request.priority == 'critical',
            tokens_used=result.usage().total_tokens
        )
    
    except Exception as e:
        deps.logger.error(f"Support query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Containerized Deployment

```dockerfile
# Dockerfile for Pydantic AI agent service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run with gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", "main:app"]
```

```yaml
# docker-compose.yml for full stack
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/agents
      REDIS_URL: redis://redis:6379
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ENVIRONMENT: production
      LOGFIRE_TOKEN: ${LOGFIRE_TOKEN}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: agents
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pydantic-ai-agent
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pydantic-ai-agent
  template:
    metadata:
      labels:
        app: pydantic-ai-agent
    spec:
      containers:
      - name: agent
        image: your-registry/pydantic-ai-agent:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
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
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: pydantic-ai-agent-service
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: pydantic-ai-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

## Observability & Monitoring

### Logfire Integration

```python
"""
Production-grade observability with Pydantic Logfire.
Provides automatic tracing, logging, and performance monitoring.
"""

import logfire
from pydantic_ai import Agent

# Configure Logfire
logfire.configure(
    service_name='pydantic-ai-agent',
    service_version='1.0.0',
    environment='production',
    send_to_logfire='always'  # 'always', 'if-token-present', 'never'
)

# Instrument Pydantic AI automatically
logfire.instrument_pydantic_ai()

# Instrument other libraries
logfire.instrument_httpx()
logfire.instrument_sqlalchemy()
logfire.instrument_asyncpg()

# Create agent
agent = Agent('openai:gpt-4o')

# Custom spans for business logic
@logfire.span('process_customer_request')
async def handle_request(customer_id: int, query: str):
    """Trace entire customer request flow."""
    
    # Add context to trace
    logfire.set_attribute('customer_id', customer_id)
    logfire.set_attribute('query_length', len(query))
    
    # Run agent (automatically traced)
    result = await agent.run(query)
    
    # Log custom metrics
    logfire.info(
        'agent_response_generated',
        response_length=len(result.output),
        tokens_used=result.usage().total_tokens
    )
    
    return result

# Performance monitoring
@logfire.span('expensive_operation')
async def monitor_performance():
    """Monitor performance with automatic timing."""
    import time
    start = time.time()
    
    # Your operation here
    await agent.run('Complex query')
    
    duration = time.time() - start
    logfire.warn(f'slow_operation', duration=duration)
```

### Metrics & Dashboards

```python
"""
Prometheus metrics for Pydantic AI agents.
Integrates with Grafana for visualization.
"""

from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
agent_requests = Counter(
    'pydantic_ai_requests_total',
    'Total agent requests',
    ['agent_name', 'status']
)

agent_latency = Histogram(
    'pydantic_ai_request_duration_seconds',
    'Request latency in seconds',
    ['agent_name'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)

tokens_used = Counter(
    'pydantic_ai_tokens_used_total',
    'Total tokens consumed',
    ['agent_name', 'token_type']
)

active_requests = Gauge(
    'pydantic_ai_active_requests',
    'Currently active requests',
    ['agent_name']
)

# Middleware to track metrics
async def metrics_middleware(agent_name: str):
    """Track metrics for agent calls."""
    
    async def wrapper(agent_run):
        active_requests.labels(agent_name=agent_name).inc()
        start_time = time.time()
        
        try:
            result = await agent_run()
            agent_requests.labels(agent_name=agent_name, status='success').inc()
            
            # Track token usage
            usage = result.usage()
            tokens_used.labels(agent_name=agent_name, token_type='input').inc(
                usage.input_tokens
            )
            tokens_used.labels(agent_name=agent_name, token_type='output').inc(
                usage.output_tokens
            )
            
            return result
        
        except Exception as e:
            agent_requests.labels(agent_name=agent_name, status='error').inc()
            raise
        
        finally:
            duration = time.time() - start_time
            agent_latency.labels(agent_name=agent_name).observe(duration)
            active_requests.labels(agent_name=agent_name).dec()
    
    return wrapper
```

## Error Handling & Resilience

### Comprehensive Error Strategy

```python
"""
Production error handling with:
- Graceful degradation
- Automatic retries
- Error categorization
- User-friendly messages
"""

from pydantic_ai import Agent, ModelRetry
from enum import Enum
import asyncio
from typing import Optional

class ErrorCategory(str, Enum):
    """Error classification for monitoring."""
    RATE_LIMIT = 'rate_limit'
    TIMEOUT = 'timeout'
    VALIDATION = 'validation'
    EXTERNAL_SERVICE = 'external_service'
    INTERNAL = 'internal'

class AgentError(Exception):
    """Base agent error with category."""
    def __init__(self, message: str, category: ErrorCategory, retryable: bool = False):
        self.message = message
        self.category = category
        self.retryable = retryable
        super().__init__(message)

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Exponential backoff retry strategy."""
    
    for attempt in range(max_retries):
        try:
            return await func()
        
        except AgentError as e:
            if not e.retryable or attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            import random
            jitter = random.uniform(0, delay * 0.1)
            
            await asyncio.sleep(delay + jitter)
        
        except Exception as e:
            # Unexpected error
            raise AgentError(
                f"Unexpected error: {str(e)}",
                ErrorCategory.INTERNAL
            )

# Agent with comprehensive error handling
agent = Agent('openai:gpt-4o')

@agent.output_validator
async def validate_and_retry(ctx, output):
    """Validate output and request retry if needed."""
    
    if not output or len(str(output)) == 0:
        raise ModelRetry('Empty response. Please provide a complete answer.')
    
    return output

async def safe_agent_run(query: str) -> Optional[str]:
    """Safe agent execution with error handling."""
    
    try:
        async def run_agent():
            return await agent.run(query)
        
        result = await retry_with_backoff(run_agent)
        return result.output
    
    except AgentError as e:
        logger.error(f"{e.category.value}: {e.message}")
        
        # Return user-friendly fallback
        fallback_messages = {
            ErrorCategory.RATE_LIMIT: "Service is busy. Please try again in a moment.",
            ErrorCategory.TIMEOUT: "Request timed out. Please try again.",
            ErrorCategory.VALIDATION: "Invalid input. Please check your request.",
            ErrorCategory.EXTERNAL_SERVICE: "External service unavailable. Retrying...",
            ErrorCategory.INTERNAL: "An internal error occurred. Please contact support.",
        }
        
        return fallback_messages.get(e.category, "An error occurred.")
    
    except Exception as e:
        logger.exception(f"Unexpected error in agent run: {e}")
        return None
```

## Token Management & Cost Control

```python
"""
Production token tracking for cost control and optimization.
"""

from pydantic_ai import Agent, UsageLimits, RunUsage
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TokenBudget:
    """Track token usage and costs."""
    daily_limit: int
    monthly_limit: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    tokens_used_today: int = 0
    tokens_used_month: int = 0

class TokenTracker:
    """Track and enforce token budgets."""
    
    def __init__(self, budget: TokenBudget):
        self.budget = budget
        self.last_reset = datetime.now()
    
    def check_budget(self, estimated_tokens: int) -> bool:
        """Check if request fits within budget."""
        
        # Reset daily count if needed
        if (datetime.now() - self.last_reset).days >= 1:
            self.budget.tokens_used_today = 0
            self.last_reset = datetime.now()
        
        can_use_today = (
            self.budget.tokens_used_today + estimated_tokens <= 
            self.budget.daily_limit
        )
        
        can_use_month = (
            self.budget.tokens_used_month + estimated_tokens <= 
            self.budget.monthly_limit
        )
        
        return can_use_today and can_use_month
    
    def record_usage(self, usage: RunUsage):
        """Record actual token usage."""
        total = usage.input_tokens + usage.output_tokens
        self.budget.tokens_used_today += total
        self.budget.tokens_used_month += total
    
    def estimate_cost(self, usage: RunUsage) -> float:
        """Calculate cost of request."""
        input_cost = (usage.input_tokens / 1000) * self.budget.cost_per_1k_input
        output_cost = (usage.output_tokens / 1000) * self.budget.cost_per_1k_output
        return input_cost + output_cost

# Usage
tracker = TokenTracker(TokenBudget(
    daily_limit=100_000,
    monthly_limit=5_000_000,
    cost_per_1k_input=0.005,
    cost_per_1k_output=0.015
))

agent = Agent('openai:gpt-4o')

async def cost_controlled_run(query: str) -> Optional[str]:
    """Run agent with cost controls."""
    
    # Estimate tokens (rough estimate)
    estimated_tokens = len(query.split()) * 2 + 200
    
    if not tracker.check_budget(estimated_tokens):
        return "Budget limit reached. Please try again later."
    
    # Run with token limits
    result = await agent.run(
        query,
        usage_limits=UsageLimits(total_tokens_limit=estimated_tokens + 100)
    )
    
    # Record actual usage and cost
    tracker.record_usage(result.usage())
    cost = tracker.estimate_cost(result.usage())
    
    logger.info(f"Request cost: ${cost:.4f}")
    
    return result.output
```

## Caching & Performance

```python
"""
Production caching strategies for agent responses.
Reduces latency and costs significantly.
"""

from redis import Redis
import hashlib
import json
from pydantic_ai import Agent

redis_client = Redis.from_url('redis://localhost:6379')

class AgentCacheManager:
    """Manage caching for agent responses."""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl  # Cache TTL in seconds
    
    def _generate_cache_key(self, query: str, agent_name: str) -> str:
        """Generate deterministic cache key."""
        key_data = f"{agent_name}:{query}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_cached_response(
        self,
        query: str,
        agent_name: str
    ) -> Optional[str]:
        """Get response from cache if available."""
        
        key = self._generate_cache_key(query, agent_name)
        cached = redis_client.get(key)
        
        if cached:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached.decode('utf-8')
        
        return None
    
    async def cache_response(
        self,
        query: str,
        response: str,
        agent_name: str
    ):
        """Cache agent response."""
        
        key = self._generate_cache_key(query, agent_name)
        redis_client.setex(key, self.ttl, response)
        
        logger.info(f"Cached response for: {query[:50]}...")
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries by pattern."""
        for key in redis_client.scan_iter(match=pattern):
            redis_client.delete(key)

# Usage with cache
cache_manager = AgentCacheManager(ttl=3600)
agent = Agent('openai:gpt-4o')

async def run_with_caching(query: str, agent_name: str = 'default') -> str:
    """Run agent with caching."""
    
    # Try cache first
    cached_response = await cache_manager.get_cached_response(query, agent_name)
    if cached_response:
        return cached_response
    
    # Run agent if not cached
    result = await agent.run(query)
    response = result.output
    
    # Cache for future use
    await cache_manager.cache_response(query, response, agent_name)
    
    return response
```

## Scaling Strategies

### Horizontal Scaling with Load Balancing

```python
"""
Load balancing configuration for multiple agent instances.
"""

# nginx.conf - Reverse proxy and load balancer
upstream pydantic_agents {
    least_conn;  # Least connection load balancing
    server agent1:8000 weight=1;
    server agent2:8000 weight=1;
    server agent3:8000 weight=1;
    
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    location /api/agent/ {
        proxy_pass http://pydantic_agents;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://pydantic_agents/health;
    }
}
```

### Queue-Based Processing for Long-Running Tasks

```python
"""
Use message queues (Celery, RabbitMQ) for long-running agent tasks.
"""

from celery import Celery
from pydantic_ai import Agent

app = Celery('agent_tasks', broker='redis://localhost:6379')

@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def run_analysis_agent(self, data: str):
    """Long-running agent task with Celery."""
    
    try:
        agent = Agent('openai:gpt-4o')
        result = agent.run_sync(data)
        
        # Store result
        store_result(self.request.id, result.output)
        
        return result.output
    
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

# FastAPI endpoint
from fastapi import BackgroundTasks

@app.post('/api/analyze-async')
async def analyze_async(data: str, background_tasks: BackgroundTasks):
    """Submit long task to queue."""
    
    task = run_analysis_agent.delay(data)
    
    return {
        'task_id': task.id,
        'status': 'queued',
        'check_url': f'/api/task/{task.id}'
    }

@app.get('/api/task/{task_id}')
async def get_task_result(task_id: str):
    """Poll for task result."""
    
    from celery.result import AsyncResult
    
    result = AsyncResult(task_id, app=app)
    
    return {
        'task_id': task_id,
        'status': result.state,
        'result': result.result if result.ready() else None
    }
```

---

(Continues with more production patterns, best practices, and operational guidance)


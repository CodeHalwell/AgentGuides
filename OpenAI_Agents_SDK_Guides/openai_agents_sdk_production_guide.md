# OpenAI Agents SDK: Production Guide

A comprehensive guide to deploying, scaling, and operating agents in production environments. This guide covers deployment architectures, observability, security, performance optimisation, and operational best practices.

## Table of Contents

1. [Deployment Architectures](#deployment-architectures)
2. [Scalability and Performance](#scalability-and-performance)
3. [Error Handling and Resilience](#error-handling-and-resilience)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Security and Safety](#security-and-safety)
6. [Cost Optimisation](#cost-optimisation)
7. [Testing Strategies](#testing-strategies)
8. [CI/CD Integration](#cicd-integration)
9. [Database and Session Management](#database-and-session-management)
10. [Rate Limiting and Quotas](#rate-limiting-and-quotas)
11. [Multi-Tenancy](#multi-tenancy)
12. [Real-World Deployment Examples](#real-world-deployment-examples)

---

## Deployment Architectures

### Monolithic Service Deployment

Single service handling all agent logic:

```python
# app.py - FastAPI-based agent service
from fastapi import FastAPI, HTTPException
from agents import Agent, Runner, SQLiteSession
from pydantic import BaseModel
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Service")

class QueryRequest(BaseModel):
    user_id: str
    query: str
    agent_name: str = "default"

class QueryResponse(BaseModel):
    user_id: str
    response: str
    tokens_used: int

# Initialize agents at startup
agents = {}

@app.on_event("startup")
async def startup():
    agents["default"] = Agent(
        name="Default Assistant",
        instructions="Provide helpful assistance"
    )
    agents["research"] = Agent(
        name="Research Assistant",
        instructions="Conduct thorough research"
    )
    logger.info("Agents initialised")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        agent = agents.get(request.agent_name, agents["default"])
        session = SQLiteSession(request.user_id, "sessions.db")
        
        result = await Runner.run(
            agent,
            request.query,
            session=session,
            max_turns=5
        )
        
        logger.info(f"Query processed for user {request.user_id}")
        
        return QueryResponse(
            user_id=request.user_id,
            response=result.final_output,
            tokens_used=getattr(result, "usage", {}).get("total_tokens", 0)
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agents": list(agents.keys())}
```

Deploy with:

```bash
pip install fastapi uvicorn openai-agents
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Microservices Architecture

Separate services for different agent types:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Triage agent service
  triage-service:
    build: ./services/triage
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      SERVICE_PORT: 8001
    ports:
      - "8001:8001"
    depends_on:
      - postgres
  
  # Billing agent service
  billing-service:
    build: ./services/billing
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      SERVICE_PORT: 8002
      DATABASE_URL: postgresql://user:pass@postgres:5432/billing
    ports:
      - "8002:8002"
    depends_on:
      - postgres
  
  # Technical support service
  support-service:
    build: ./services/support
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      SERVICE_PORT: 8003
    ports:
      - "8003:8003"
    depends_on:
      - postgres
  
  # API Gateway
  api-gateway:
    build: ./gateway
    environment:
      TRIAGE_SERVICE: http://triage-service:8001
      BILLING_SERVICE: http://billing-service:8002
      SUPPORT_SERVICE: http://support-service:8003
    ports:
      - "8000:8000"
    depends_on:
      - triage-service
      - billing-service
      - support-service
  
  # Session storage
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: agents
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # Cache layer
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Serverless Deployment

AWS Lambda-based agent execution:

```python
# lambda_handler.py - AWS Lambda handler
from agents import Agent, Runner, SQLiteSession
import json
import asyncio
import os

def get_agent_by_type(agent_type):
    """Factory for creating agents."""
    agents = {
        "support": Agent(
            name="Support Agent",
            instructions="Provide customer support"
        ),
        "billing": Agent(
            name="Billing Agent",
            instructions="Handle billing inquiries"
        )
    }
    return agents.get(agent_type, agents["support"])

async def run_agent(event):
    """Execute agent in Lambda environment."""
    user_id = event.get("user_id", "anonymous")
    query = event.get("query", "")
    agent_type = event.get("agent_type", "support")
    
    agent = get_agent_by_type(agent_type)
    
    # Use in-memory session in Lambda (or S3/DynamoDB for persistence)
    session = SQLiteSession(user_id, ":memory:")
    
    result = await Runner.run(
        agent,
        query,
        session=session,
        max_turns=3
    )
    
    return {
        "user_id": user_id,
        "response": result.final_output,
        "agent": agent.name
    }

def lambda_handler(event, context):
    """AWS Lambda entry point."""
    try:
        result = asyncio.run(run_agent(event))
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
```

Deploy with Terraform:

```hcl
# main.tf
resource "aws_lambda_function" "agent_function" {
  filename            = "lambda.zip"
  function_name       = "openai-agent-processor"
  role               = aws_iam_role.lambda_role.arn
  handler            = "lambda_handler.lambda_handler"
  runtime            = "python3.11"
  timeout            = 300
  memory_size        = 1024

  environment {
    variables = {
      OPENAI_API_KEY = var.openai_api_key
    }
  }
}

resource "aws_api_gateway_rest_api" "agent_api" {
  name = "agent-api"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id           = aws_api_gateway_rest_api.agent_api.id
  resource_id           = aws_api_gateway_resource.resource.id
  http_method           = "POST"
  type                  = "AWS_PROXY"
  integration_http_method = "POST"
  uri                   = aws_lambda_function.agent_function.invoke_arn
}
```

### Kubernetes Deployment

Container orchestration with Kubernetes:

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  namespace: production

spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  selector:
    matchLabels:
      app: agent-service
  
  template:
    metadata:
      labels:
        app: agent-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    
    spec:
      containers:
      - name: agent-service
        image: myregistry.azurecr.io/agent-service:latest
        imagePullPolicy: Always
        
        ports:
        - containerPort: 8000
          name: http
        
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database-url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 2
        
        volumeMounts:
        - name: config
          mountPath: /etc/config
      
      volumes:
      - name: config
        configMap:
          name: agent-config

---
apiVersion: v1
kind: Service
metadata:
  name: agent-service

spec:
  selector:
    app: agent-service
  
  ports:
  - port: 80
    targetPort: 8000
    name: http
  
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-service-hpa

spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-service
  
  minReplicas: 3
  maxReplicas: 10
  
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

### Horizontal Scaling

Distribute load across multiple instances:

```python
# load_balanced_client.py
import aiohttp
import asyncio
from typing import List

class LoadBalancedAgentClient:
    def __init__(self, service_urls: List[str]):
        self.service_urls = service_urls
        self.current_index = 0
    
    def get_next_url(self) -> str:
        """Round-robin service selection."""
        url = self.service_urls[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.service_urls)
        return url
    
    async def query(self, user_id: str, query: str) -> dict:
        """Send query to least-loaded service."""
        url = f"{self.get_next_url()}/query"
        
        async with aiohttp.ClientSession() as session:
            payload = {"user_id": user_id, "query": query}
            async with session.post(url, json=payload) as resp:
                return await resp.json()

async def main():
    client = LoadBalancedAgentClient([
        "http://agent-1:8000",
        "http://agent-2:8000",
        "http://agent-3:8000"
    ])
    
    # Distribute queries
    tasks = [
        client.query(f"user_{i}", "Query")
        for i in range(100)
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} queries")
```

### Connection Pooling

Reuse HTTP connections efficiently:

```python
# connection_pool.py
import aiohttp
from typing import Optional

class ConnectionPoolManager:
    def __init__(self, max_connections: int = 100):
        self.connector = aiohttp.TCPConnector(limit=max_connections)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(connector=self.connector)
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

async def main():
    async with ConnectionPoolManager() as session:
        # Reuse same session for multiple requests
        tasks = []
        for i in range(1000):
            task = session.post(
                "http://agent-service/query",
                json={"query": f"Query {i}"}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        print(f"Completed {len(responses)} requests with pooling")
```

### Caching Strategies

Cache agent responses:

```python
# cache_layer.py
from functools import wraps
import hashlib
import json
import redis
from typing import Any, Callable

class CacheLayer:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
    
    def _make_key(self, agent_name: str, query: str) -> str:
        """Generate cache key."""
        combined = f"{agent_name}:{query}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def cached_agent_run(self, ttl: int = 3600):
        """Decorator for caching agent runs."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(agent_name: str, query: str, *args, **kwargs) -> Any:
                cache_key = self._make_key(agent_name, query)
                
                # Check cache
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
                
                # Run agent
                result = await func(agent_name, query, *args, **kwargs)
                
                # Cache result
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result)
                )
                
                return result
            
            return wrapper
        return decorator

cache = CacheLayer()

@cache.cached_agent_run(ttl=3600)
async def run_cached_agent(agent_name: str, query: str):
    """Run agent with caching."""
    from agents import Agent, Runner
    
    agent = Agent(name=agent_name)
    result = await Runner.run(agent, query)
    
    return {"response": result.final_output}
```

---

## Error Handling and Resilience

### Comprehensive Exception Handling

Handle all error scenarios:

```python
# error_handler.py
from agents import (
    Agent, Runner,
    MaxTurnsExceeded,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    ModelBehaviorError
)
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class AgentExecutionError(Exception):
    """Base exception for agent execution."""
    pass

class AgentTimeoutError(AgentExecutionError):
    """Agent execution timed out."""
    pass

class AgentValidationError(AgentExecutionError):
    """Guardrail validation failed."""
    pass

async def run_agent_with_error_handling(
    agent: Agent,
    query: str,
    max_retries: int = 3
) -> Optional[str]:
    """Run agent with comprehensive error handling."""
    
    for attempt in range(max_retries):
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                Runner.run(agent, query),
                timeout=30.0
            )
            return result.final_output
        
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt + 1}: Timeout")
            if attempt == max_retries - 1:
                raise AgentTimeoutError("Agent execution timed out")
            await asyncio.sleep(2 ** attempt)
        
        except InputGuardrailTripwireTriggered as e:
            logger.warning(f"Input validation failed: {e}")
            raise AgentValidationError("Input failed validation")
        
        except OutputGuardrailTripwireTriggered as e:
            logger.warning(f"Output validation failed: {e}")
            raise AgentValidationError("Output failed validation")
        
        except MaxTurnsExceeded:
            logger.error("Maximum turns exceeded")
            raise AgentExecutionError("Agent exceeded maximum turns")
        
        except ModelBehaviorError as e:
            logger.error(f"Model error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
        
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            raise AgentExecutionError(f"Unexpected error: {str(e)}")
    
    return None
```

### Circuit Breaker Pattern

Prevent cascading failures:

```python
# circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitState.OPEN:
            # Check if we should try recovery
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to retry."""
        if not self.last_failure_time:
            return True
        
        elapsed = datetime.now() - self.last_failure_time
        return elapsed.total_seconds() >= self.reset_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

---

## Monitoring and Observability

### Comprehensive Logging

Structured logging for debugging:

```python
# logging_setup.py
import logging
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)

def setup_logging():
    """Configure production logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # JSON file handler
    fh = logging.FileHandler("agent_service.log")
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(JsonFormatter())
    logger.addHandler(ch)
    
    return logger

# Usage
logger = setup_logging()
logger.info("Agent service started", extra={"service": "agent-api"})
```

### Prometheus Metrics

Export metrics for monitoring:

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
agent_runs_total = Counter(
    'agent_runs_total',
    'Total agent runs',
    ['agent_name', 'status']
)

agent_run_duration = Histogram(
    'agent_run_duration_seconds',
    'Agent run duration',
    ['agent_name']
)

agent_errors_total = Counter(
    'agent_errors_total',
    'Total agent errors',
    ['agent_name', 'error_type']
)

active_runs = Gauge(
    'agent_active_runs',
    'Active agent runs'
)

tokens_used = Counter(
    'openai_tokens_used',
    'Total tokens used',
    ['model', 'token_type']
)

# Usage in agent execution
from agents import Agent, Runner
import asyncio

async def run_with_metrics(agent: Agent, query: str):
    start_time = time.time()
    active_runs.inc()
    
    try:
        result = await Runner.run(agent, query)
        agent_runs_total.labels(
            agent_name=agent.name,
            status="success"
        ).inc()
        
        tokens_used.labels(
            model=agent.model,
            token_type="total"
        ).inc(getattr(result, "usage", {}).get("total_tokens", 0))
        
        return result
    
    except Exception as e:
        agent_runs_total.labels(
            agent_name=agent.name,
            status="error"
        ).inc()
        agent_errors_total.labels(
            agent_name=agent.name,
            error_type=type(e).__name__
        ).inc()
        raise
    
    finally:
        duration = time.time() - start_time
        agent_run_duration.labels(agent_name=agent.name).observe(duration)
        active_runs.dec()
```

### OpenAI Tracing Integration

Trace agent execution:

```python
# tracing_integration.py
from agents import Agent, Runner, trace
import asyncio

async def main():
    agent = Agent(
        name="Production Agent",
        instructions="Process requests"
    )
    
    # Enable production tracing
    with trace(
        workflow_name="customer_service",
        group_id="session_123",
        metadata={
            "environment": "production",
            "region": "us-east-1",
            "customer_id": "cust_456"
        }
    ):
        result = await Runner.run(agent, "Customer query")
    
    print("Trace available at: https://platform.openai.com/traces")

asyncio.run(main())
```

---

## Security and Safety

### Input Validation and Sanitization

Prevent injection attacks:

```python
# security.py
from agents import (
    Agent, Runner, input_guardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem
)
import re
import asyncio

@input_guardrail
async def injection_prevention(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Prevent various injection attacks."""
    
    input_text = input_data if isinstance(input_data, str) else str(input_data)
    
    # SQL injection patterns
    sql_patterns = [
        r"(?i)(\b(DROP|DELETE|INSERT|UPDATE|UNION|SELECT)\b)",
        r"(-{2}|/\*|\*/)"  # Comments
    ]
    
    # Command injection patterns
    cmd_patterns = [
        r"[;&|`\$\(\)]",  # Shell metacharacters
        r"(?i)(bash|sh|cmd|powershell)"
    ]
    
    # XSS patterns
    xss_patterns = [
        r"<script",
        r"javascript:",
        r"on\w+\s*="
    ]
    
    all_patterns = sql_patterns + cmd_patterns + xss_patterns
    
    for pattern in all_patterns:
        if re.search(pattern, input_text):
            return GuardrailFunctionOutput(
                output_info={"threat": "injection detected"},
                tripwire_triggered=True
            )
    
    return GuardrailFunctionOutput(
        output_info={"threat": "none"},
        tripwire_triggered=False
    )

agent = Agent(
    name="Secure Agent",
    instructions="Process safely",
    input_guardrails=[injection_prevention]
)
```

### Rate Limiting

Prevent abuse and DOS:

```python
# rate_limiter.py
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit."""
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= self.requests_per_minute:
            return False
        
        self.requests[user_id].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=100)

async def check_rate_limit_middleware(user_id: str):
    """Middleware to check rate limits."""
    if not await rate_limiter.check_rate_limit(user_id):
        raise Exception("Rate limit exceeded")
```

---

## Cost Optimisation

### Model Selection Strategy

Choose appropriate models based on task:

```python
# cost_optimizer.py
from agents import Agent, Runner, ModelSettings
import asyncio

class CostOptimiser:
    # Model costs (per 1M tokens)
    MODEL_COSTS = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-5": {"input": 15.00, "output": 60.00}
    }
    
    @staticmethod
    def select_model(task_complexity: str) -> str:
        """Select model based on task complexity."""
        if task_complexity == "simple":
            return "gpt-4o-mini"  # Cheapest
        elif task_complexity == "medium":
            return "gpt-4o"
        else:
            return "gpt-5"
    
    @staticmethod
    def create_cost_optimised_agent(
        name: str,
        task_complexity: str
    ) -> Agent:
        """Create agent with cost-appropriate model."""
        
        model = CostOptimiser.select_model(task_complexity)
        
        return Agent(
            name=name,
            instructions="Complete task efficiently",
            model=model,
            model_settings=ModelSettings(
                temperature=0 if task_complexity == "simple" else 0.7,
                max_tokens=200 if task_complexity == "simple" else 1000
            )
        )

# Usage
async def main():
    # Simple query -> cheap model
    simple_agent = CostOptimiser.create_cost_optimised_agent(
        "FAQ Bot",
        "simple"
    )
    
    # Complex analysis -> expensive model
    analysis_agent = CostOptimiser.create_cost_optimised_agent(
        "Analyst",
        "complex"
    )

asyncio.run(main())
```

### Token Counting and Cost Estimation

Monitor and forecast costs:

```python
# cost_monitoring.py
from agents import Agent, Runner
import asyncio

class CostMonitor:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost for token usage."""
        costs = {
            "gpt-4o-mini": 0.00015 / 1000,  # $ per token
            "gpt-4o": 0.005 / 1000,
            "gpt-5": 0.015 / 1000
        }
        return tokens * costs.get(model, 0)
    
    def track_run(self, result, model: str):
        """Track usage and cost from run."""
        tokens_used = getattr(result, "usage", {}).get("total_tokens", 0)
        cost = self.estimate_cost(tokens_used, model)
        
        self.total_tokens += tokens_used
        self.total_cost += cost
        
        print(f"Cost: ${cost:.4f}, Total: ${self.total_cost:.2f}")

cost_monitor = CostMonitor()

async def main():
    agent = Agent(name="Analyst", model="gpt-4o-mini")
    
    result = await Runner.run(agent, "Analyse this")
    cost_monitor.track_run(result, agent.model)

asyncio.run(main())
```

---

## Testing Strategies

### Unit Testing Agents

Test agents in isolation:

```python
# test_agents.py
import pytest
from agents import Agent, Runner, function_tool
import asyncio

class TestAgentBasics:
    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """Test basic agent creation."""
        agent = Agent(
            name="Test Agent",
            instructions="Test instructions"
        )
        
        assert agent.name == "Test Agent"
        assert agent.instructions == "Test instructions"
    
    @pytest.mark.asyncio
    async def test_simple_agent_run(self):
        """Test agent execution."""
        agent = Agent(
            name="Test Agent",
            instructions="Answer briefly"
        )
        
        result = await Runner.run(agent, "2 + 2")
        
        assert result.final_output is not None
        assert len(result.final_output) > 0

class TestAgentTools:
    @pytest.mark.asyncio
    async def test_agent_with_tools(self):
        """Test agent with function tools."""
        
        @function_tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        agent = Agent(
            name="Calculator",
            tools=[add]
        )
        
        result = await Runner.run(agent, "What is 5 + 3?")
        
        assert result.final_output is not None
        assert "8" in result.final_output

class TestAgentErrors:
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test agent error handling."""
        from agents import MaxTurnsExceeded
        
        agent = Agent(name="Test Agent")
        
        # This might timeout or exceed max turns
        with pytest.raises((MaxTurnsExceeded, asyncio.TimeoutError)):
            await Runner.run(agent, "Test", max_turns=0)
```

### Integration Testing

Test multi-agent systems:

```python
# test_multi_agent.py
import pytest
from agents import Agent, Runner
import asyncio

class TestMultiAgentSystem:
    @pytest.mark.asyncio
    async def test_handoff_between_agents(self):
        """Test agent handoff mechanism."""
        
        agent_a = Agent(
            name="Agent A",
            handoff_description="Handles A tasks"
        )
        
        agent_b = Agent(
            name="Agent B",
            handoff_description="Handles B tasks"
        )
        
        router = Agent(
            name="Router",
            instructions="Route to appropriate agent",
            handoffs=[agent_a, agent_b]
        )
        
        result = await Runner.run(router, "Route this request")
        
        assert result.agent in [agent_a, agent_b, router]
    
    @pytest.mark.asyncio
    async def test_parallel_agents(self):
        """Test parallel agent execution."""
        
        agents = [
            Agent(name=f"Agent {i}")
            for i in range(3)
        ]
        
        tasks = [
            Runner.run(agent, f"Task {i}")
            for i, agent in enumerate(agents)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r.final_output for r in results)
```

---

## CI/CD Integration

### GitHub Actions Workflow

Automated testing and deployment:

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Lint with flake8
      run: |
        flake8 src/ tests/
    
    - name: Type checking
      run: mypy src/
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo "${{ secrets.GITHUB_TOKEN }}" | docker login ${{ env.REGISTRY }} -u $ -p -
        docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      env:
        DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
      run: |
        curl -X POST ${{ secrets.DEPLOY_WEBHOOK }} \
          -H "Authorization: Bearer $DEPLOY_TOKEN" \
          -d '{"image":"${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}"}'
```

---

## Database and Session Management

### SQLAlchemy Session Backend

Production database setup:

```python
# db_setup.py
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import asyncio

Base = declarative_base()

class ConversationItem(Base):
    __tablename__ = "conversation_items"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create engine with connection pooling
engine = create_engine(
    "postgresql://user:password@localhost/agents",
    pool_size=20,
    max_overflow=40,
    pool_recycle=3600
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

async def get_session():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## Rate Limiting and Quotas

### Token Quota Management

Enforce token limits:

```python
# quota_manager.py
from collections import defaultdict
from datetime import datetime, timedelta

class TokenQuotaManager:
    def __init__(self, monthly_limit: int = 1_000_000):
        self.monthly_limit = monthly_limit
        self.user_usage = defaultdict(lambda: {"tokens": 0, "month": None})
    
    def check_quota(self, user_id: str, tokens_needed: int) -> bool:
        """Check if user has quota for tokens."""
        current_month = datetime.now().strftime("%Y-%m")
        usage = self.user_usage[user_id]
        
        # Reset if new month
        if usage["month"] != current_month:
            usage["tokens"] = 0
            usage["month"] = current_month
        
        return (usage["tokens"] + tokens_needed) <= self.monthly_limit
    
    def consume_tokens(self, user_id: str, tokens: int):
        """Consume tokens from user's quota."""
        self.user_usage[user_id]["tokens"] += tokens

quota_manager = TokenQuotaManager(monthly_limit=10_000_000)
```

---

## Multi-Tenancy

### Tenant Isolation

Isolate data per tenant:

```python
# multi_tenancy.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TenantContext:
    tenant_id: str
    user_id: str
    api_key: str
    tier: str

class TenantManager:
    def __init__(self):
        self.tenants = {}
    
    def get_tenant_session(self, tenant_id: str) -> str:
        """Get isolated session for tenant."""
        return f"session_{tenant_id}_{datetime.now().timestamp()}"
    
    def get_tenant_database_url(self, tenant_id: str) -> str:
        """Get database URL for tenant."""
        return f"postgresql://user:pass@localhost/{tenant_id}_db"

tenant_manager = TenantManager()

from agents import Agent, Runner, SQLiteSession
import asyncio

async def run_tenant_isolated(
    tenant_context: TenantContext,
    query: str
):
    """Run agent with tenant isolation."""
    
    agent = Agent(
        name=f"Agent-{tenant_context.tenant_id}",
        instructions="Process request"
    )
    
    session_id = tenant_manager.get_tenant_session(tenant_context.tenant_id)
    session = SQLiteSession(session_id, f"{tenant_context.tenant_id}.db")
    
    result = await Runner.run(agent, query, session=session)
    
    return result.final_output
```

---

## Real-World Deployment Examples

### Customer Service Platform

Complete customer service system:

```python
# customer_service_platform.py
from fastapi import FastAPI, HTTPException, Depends
from agents import Agent, Runner, SQLiteSession, WebSearchTool, function_tool
from pydantic import BaseModel
import asyncio
import logging

app = FastAPI(title="Customer Service Platform")
logger = logging.getLogger(__name__)

class ServiceRequest(BaseModel):
    customer_id: str
    issue_type: str
    description: str

# Initialise specialised agents
billing_agent = Agent(
    name="Billing Support",
    handoff_description="Handles billing and payment issues",
    instructions="Help with invoices, refunds, and billing questions"
)

technical_agent = Agent(
    name="Technical Support",
    handoff_description="Handles technical issues and troubleshooting",
    instructions="Troubleshoot technical problems",
    tools=[WebSearchTool()]
)

account_agent = Agent(
    name="Account Support",
    handoff_description="Handles account management",
    instructions="Help with account settings and profile updates"
)

# Triage agent
triage_agent = Agent(
    name="Support Triage",
    instructions="""Determine issue type and route to appropriate specialist:
    - Billing issues -> Billing Support
    - Technical problems -> Technical Support
    - Account issues -> Account Support""",
    handoffs=[billing_agent, technical_agent, account_agent]
)

@app.post("/support/submit")
async def submit_support_request(request: ServiceRequest):
    try:
        session = SQLiteSession(request.customer_id, "support.db")
        
        result = await Runner.run(
            triage_agent,
            f"Issue: {request.issue_type}. Description: {request.description}",
            session=session
        )
        
        logger.info(f"Request handled by {result.agent.name}")
        
        return {
            "ticket_id": f"TKT-{request.customer_id}-{int(asyncio.time())}",
            "response": result.final_output,
            "agent": result.agent.name
        }
    
    except Exception as e:
        logger.error(f"Support request error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process request")

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

This comprehensive production guide provides enterprise-ready patterns for deploying and operating OpenAI Agents SDK at scale.


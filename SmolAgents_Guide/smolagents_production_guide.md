# SmolAgents: Production Deployment Guide

**Building Reliable, Scalable, and Cost-Effective Agent Systems for Production Environments**

---

## Table of Contents

1. [Production Readiness Checklist](#production-readiness-checklist)
2. [Performance Optimisation](#performance-optimisation)
3. [Cost Management](#cost-management)
4. [Monitoring & Observability](#monitoring--observability)
5. [Security Best Practices](#security-best-practices)
6. [Error Handling & Resilience](#error-handling--resilience)
7. [Deployment Options](#deployment-options)
8. [Scaling Strategies](#scaling-strategies)
9. [Testing & Quality Assurance](#testing--quality-assurance)
10. [Observability Integration (Weights & Biases Weave)](#observability-integration-weights--biases-weave)
11. [Incident Response](#incident-response)
12. [Migration Strategies](#migration-strategies)

---

## Production Readiness Checklist

Before deploying SmolAgents to production, complete this comprehensive checklist:

### Core Functionality

```python
# ✓ Agent implementation verified with unit tests
# ✓ All tools tested independently
# ✓ Integration tests passing
# ✓ Error cases handled gracefully
# ✓ Timeout values configured appropriately
# ✓ Model provider credentials validated
# ✓ Memory configuration tuned
# ✓ Output validation implemented

# Example implementation:
from smolagents import CodeAgent, InferenceClientModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ProductionAgent:
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        max_steps: int = 10,
        timeout: float = 60.0,
        enable_monitoring: bool = True
    ):
        self.model = InferenceClientModel(model_id=model_id)
        self.agent = CodeAgent(
            model=self.model,
            tools=[...],
            max_steps=max_steps,
            timeout=timeout,
            verbosity_level=0  # Minimal logging in production
        )
        self.enable_monitoring = enable_monitoring
    
    def run(self, task: str, return_full_result: bool = True) -> Optional[dict]:
        """Execute task with production safety checks"""
        
        try:
            # Validate input
            if not task or len(task) > 10000:
                logger.error(f"Invalid task length: {len(task)}")
                return None
            
            # Execute with monitoring
            result = self.agent.run(
                task,
                return_full_result=return_full_result
            )
            
            # Validate output
            if result and isinstance(result, dict):
                logger.info(f"Task completed: {result['success']}")
                if self.enable_monitoring:
                    self.log_metrics(result)
                return result
            else:
                logger.warning("Unexpected result type")
                return None
        
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return None
    
    def log_metrics(self, result: dict) -> None:
        """Log execution metrics for monitoring"""
        logger.info(f"Execution metrics: "
                   f"steps={len(result.get('steps', []))}, "
                   f"tokens_in={result.get('token_usage', {}).get('input_tokens')}, "
                   f"time={result.get('execution_time')}")
```

### Infrastructure Requirements

```python
# ✓ LLM provider account with sufficient quota
# ✓ Sandbox execution environment configured (Docker/E2B/Modal)
# ✓ Database connections configured and tested
# ✓ Cache layer deployed (Redis/Memcached)
# ✓ API rate limiting configured
# ✓ Load balancing setup
# ✓ Backup/redundancy implemented
# ✓ Monitoring infrastructure ready

# Infrastructure validation
import asyncio
from healthchecks import (
    check_llm_api,
    check_database,
    check_cache,
    check_sandbox
)

async def validate_infrastructure():
    """Ensure all production dependencies are ready"""
    checks = [
        check_llm_api(),
        check_database(),
        check_cache(),
        check_sandbox()
    ]
    
    results = await asyncio.gather(*checks)
    
    if all(results):
        logger.info("✓ All infrastructure checks passed")
        return True
    else:
        logger.error("✗ Infrastructure checks failed")
        return False
```

### Operational Requirements

```python
# ✓ Monitoring dashboards configured
# ✓ Alerting rules defined
# ✓ Logging infrastructure in place
# ✓ Backup procedures documented
# ✓ Runbooks created for common issues
# ✓ On-call escalation paths defined
# ✓ Performance baselines established
# ✓ Disaster recovery plan documented

# Example alerting configuration
alerting_rules = {
    'high_error_rate': {
        'threshold': 0.05,  # 5% error rate
        'window': 300,  # 5 minutes
        'severity': 'critical',
        'channels': ['pagerduty', 'slack']
    },
    'high_latency': {
        'threshold': 30.0,  # 30 seconds
        'window': 60,
        'severity': 'warning',
        'channels': ['slack']
    },
    'high_token_usage': {
        'threshold': 1000000,  # Daily token limit
        'window': 86400,
        'severity': 'warning',
        'channels': ['slack']
    },
    'quota_exceeded': {
        'pattern': 'quota',
        'severity': 'critical',
        'channels': ['pagerduty', 'email']
    }
}
```

---

## Performance Optimisation

### Model Selection for Performance

```python
from smolagents import CodeAgent, LiteLLMModel, TransformersModel

# For different performance requirements:

# 1. Speed-optimised (latency critical)
class FastAgent(CodeAgent):
    def __init__(self):
        model = LiteLLMModel(
            model_id="groq/llama-3.3-70b-versatile",
            temperature=0.1  # Lower temperature for consistency
        )
        super().__init__(
            model=model,
            max_steps=5,  # Limit iterations
            timeout=10.0  # Strict timeout
        )

# 2. Quality-optimised (accuracy critical)
class AccurateAgent(CodeAgent):
    def __init__(self):
        model = LiteLLMModel(
            model_id="gpt-4o",
            temperature=0.3  # Moderate temperature
        )
        super().__init__(
            model=model,
            max_steps=15,  # Allow more iterations
            timeout=60.0
        )

# 3. Cost-optimised (budget critical)
class EconomyAgent(CodeAgent):
    def __init__(self):
        model = LiteLLMModel(
            model_id="gpt-3.5-turbo",
            temperature=0.2
        )
        super().__init__(
            model=model,
            max_steps=8,
            timeout=30.0
        )

# 4. Local-optimised (no API costs)
class LocalAgent(CodeAgent):
    def __init__(self):
        model = TransformersModel(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            load_in_4bit=True,  # Quantisation for memory
            device_map="auto"
        )
        super().__init__(
            model=model,
            max_steps=10,
            executor_type="docker"  # Isolated execution
        )
```

### Caching & Memoisation

```python
from functools import lru_cache, wraps
from typing import Any, Callable
import hashlib
import json
import redis

# Redis connection pool for production
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True,
    health_check_interval=30
)

def cache_tool_result(ttl: int = 3600):
    """Decorator to cache tool results in Redis"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{json.dumps({
                'args': args,
                'kwargs': kwargs
            }, default=str, sort_keys=True)}"
            
            # Hash if too long
            if len(cache_key) > 1024:
                cache_key = f"{func.__name__}:{hashlib.md5(cache_key.encode()).hexdigest()}"
            
            # Try to get from cache
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            try:
                redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result, default=str)
                )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
            
            return result
        return wrapper
    return decorator

# Apply to expensive tools
@cache_tool_result(ttl=86400)  # 24 hour cache
def get_stock_price(ticker: str) -> float:
    """Get stock price with caching"""
    # Implementation
    pass

@cache_tool_result(ttl=3600)  # 1 hour cache
def web_search(query: str) -> list:
    """Web search with caching"""
    # Implementation
    pass
```

### Prompt Optimisation

```python
from smolagents import CodeAgent

class OptimisedAgent(CodeAgent):
    """Agent with optimised system prompt"""
    
    def _create_system_prompt(self) -> str:
        """Generate optimised system prompt"""
        
        # Minimal but effective prompt
        prompt = """You are a helpful AI agent that solves tasks efficiently.

You have access to these tools:
{tools}

IMPORTANT RULES:
1. Write Python code using only the provided tools
2. Use variables to store intermediate results
3. Stop after final_answer() call
4. Limit API calls - reuse results when possible
5. Handle errors gracefully with try/except

Generate code for the task below:
"""
        return prompt

    def _create_tool_descriptions(self) -> str:
        """Generate concise tool descriptions"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(
                f"def {tool.name}({self._format_inputs(tool.inputs)}) -> {tool.output_type}:\n"
                f"    \"{tool.description}\""
            )
        return "\n\n".join(descriptions)
    
    def _format_inputs(self, inputs: dict) -> str:
        """Format input parameters concisely"""
        params = []
        for name, spec in inputs.items():
            params.append(f"{name}: {spec.get('type', 'any')}")
        return ", ".join(params)
```

### Parallel Execution

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any

class ParallelExecutor:
    """Execute multiple agent tasks in parallel"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def run_parallel(self, tasks: List[str]) -> List[Any]:
        """Execute tasks in parallel"""
        futures = {
            self.executor.submit(self.agent.run, task): task
            for task in tasks
        }
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)
        
        return results
    
    def shutdown(self):
        """Clean up executor"""
        self.executor.shutdown(wait=True)

# Usage
executor = ParallelExecutor(max_workers=8)
tasks = [
    "Analyse market trends",
    "Calculate financial metrics",
    "Generate report summary",
    "Find competitor data"
]
results = executor.run_parallel(tasks)
executor.shutdown()
```

---

## Cost Management

### Token Usage Tracking

```python
from dataclasses import dataclass
from typing import Dict, Optional
import datetime

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    timestamp: datetime.datetime
    task_id: str
    model: str
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def cost_estimate(self, input_rate: float, output_rate: float) -> float:
        """Calculate estimated cost (per million tokens)"""
        input_cost = (self.input_tokens / 1_000_000) * input_rate
        output_cost = (self.output_tokens / 1_000_000) * output_rate
        return input_cost + output_cost

class TokenBudgetManager:
    """Track and enforce token budgets"""
    
    def __init__(self, daily_limit: int = 10_000_000):
        self.daily_limit = daily_limit
        self.usage_today = 0
        self.usages: List[TokenUsage] = []
    
    def can_execute(self, estimated_tokens: int) -> bool:
        """Check if execution would exceed budget"""
        return (self.usage_today + estimated_tokens) <= self.daily_limit
    
    def record_usage(self, usage: TokenUsage) -> None:
        """Record token usage"""
        self.usage_today += usage.total_tokens
        self.usages.append(usage)
        
        percentage = (self.usage_today / self.daily_limit) * 100
        logger.info(f"Token usage: {percentage:.1f}% of daily limit")
        
        if percentage > 80:
            logger.warning(f"Approaching token limit: {percentage:.1f}%")
    
    def reset_daily(self) -> None:
        """Reset daily counter (call once per day)"""
        logger.info(f"Daily reset: {self.usage_today} tokens used")
        self.usage_today = 0

# Production implementation
budget_manager = TokenBudgetManager(daily_limit=10_000_000)

def run_with_budget(agent, task):
    if not budget_manager.can_execute(5000):  # Estimated tokens
        logger.error("Token budget exceeded")
        return None
    
    result = agent.run(task, return_full_result=True)
    if result and 'token_usage' in result:
        usage = TokenUsage(
            input_tokens=result['token_usage']['input_tokens'],
            output_tokens=result['token_usage']['output_tokens'],
            timestamp=datetime.datetime.now(),
            task_id=result.get('task_id'),
            model=result.get('model_id')
        )
        budget_manager.record_usage(usage)
    
    return result
```

### Cost Optimisation Strategies

```python
from enum import Enum

class CostOptimisationStrategy(Enum):
    AGGRESSIVE = "use cheapest models"
    BALANCED = "balance cost and quality"
    CONSERVATIVE = "prioritise quality"

class AdaptiveModelSelector:
    """Select model based on task and budget"""
    
    def __init__(self, strategy: CostOptimisationStrategy = CostOptimisationStrategy.BALANCED):
        self.strategy = strategy
        self.model_costs = {
            'gpt-4o': {'input': 5.0, 'output': 15.0},  # per million tokens
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'claude-3-5-sonnet': {'input': 3.0, 'output': 15.0},
            'groq/llama-3.3-70b': {'input': 0.099, 'output': 0.30},  # Groq is cheap
            'local-model': {'input': 0.0, 'output': 0.0}  # Free
        }
    
    def select_model(
        self,
        task_complexity: str,
        quality_requirement: str,
        estimated_tokens: int
    ) -> str:
        
        if self.strategy == CostOptimisationStrategy.AGGRESSIVE:
            if estimated_tokens < 100_000:
                return "groq/llama-3.3-70b"  # Cheapest fast model
            else:
                return "gpt-3.5-turbo"  # Cheap for large tokens
        
        elif self.strategy == CostOptimisationStrategy.BALANCED:
            if task_complexity == "simple" and quality_requirement == "low":
                return "groq/llama-3.3-70b"
            elif task_complexity == "complex" and quality_requirement == "high":
                return "gpt-4o"
            else:
                return "gpt-3.5-turbo"
        
        else:  # CONSERVATIVE
            if quality_requirement == "critical":
                return "gpt-4o"
            else:
                return "gpt-3.5-turbo"
    
    def calculate_cost(self, model: str, tokens: TokenUsage) -> float:
        """Calculate execution cost"""
        rates = self.model_costs.get(model, {'input': 1.0, 'output': 1.0})
        input_cost = (tokens.input_tokens / 1_000_000) * rates['input']
        output_cost = (tokens.output_tokens / 1_000_000) * rates['output']
        return input_cost + output_cost
```

---

## Monitoring & Observability

### Key Metrics to Track

```python
from dataclasses import dataclass, field
from typing import Dict, Any
import time

@dataclass
class AgentMetrics:
    """Comprehensive metrics for agent execution"""
    
    # Execution metrics
    execution_time: float = 0.0
    total_steps: int = 0
    success_rate: float = 0.0
    
    # LLM metrics
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency: float = 0.0
    
    # Tool metrics
    tool_calls: int = 0
    tool_errors: int = 0
    tool_latency: Dict[str, float] = field(default_factory=dict)
    
    # Cost metrics
    estimated_cost: float = 0.0
    
    # Quality metrics
    output_quality_score: float = 0.0
    user_satisfaction: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'execution_time': self.execution_time,
            'total_steps': self.total_steps,
            'success_rate': self.success_rate,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'estimated_cost': f"${self.estimated_cost:.4f}",
            'quality_score': self.output_quality_score
        }

class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self):
        self.metrics_history = []
    
    def collect_metrics(
        self,
        result: dict,
        execution_time: float,
        cost: float
    ) -> AgentMetrics:
        """Extract and package metrics"""
        
        metrics = AgentMetrics(
            execution_time=execution_time,
            total_steps=len(result.get('steps', [])),
            success_rate=1.0 if result.get('success') else 0.0,
            input_tokens=result.get('token_usage', {}).get('input_tokens', 0),
            output_tokens=result.get('token_usage', {}).get('output_tokens', 0),
            estimated_cost=cost
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}
        
        total_exec = sum(m.execution_time for m in self.metrics_history)
        avg_exec = total_exec / len(self.metrics_history)
        total_tokens = sum(
            m.input_tokens + m.output_tokens
            for m in self.metrics_history
        )
        total_cost = sum(m.estimated_cost for m in self.metrics_history)
        success = sum(
            1 for m in self.metrics_history if m.success_rate > 0
        )
        
        return {
            'total_executions': len(self.metrics_history),
            'success_rate': success / len(self.metrics_history),
            'avg_execution_time': avg_exec,
            'total_tokens_used': total_tokens,
            'total_cost': f"${total_cost:.2f}"
        }
```

### Structured Logging

```python
import logging
import json
from pythonjsonlogger import jsonlogger
import sys

# Configure JSON logging for production
logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Log events with context
def log_agent_execution(
    task_id: str,
    task_description: str,
    model: str,
    result: dict
):
    """Log agent execution with full context"""
    
    logger.info("agent_execution", extra={
        'task_id': task_id,
        'task_description': task_description[:100],
        'model': model,
        'success': result.get('success'),
        'steps': len(result.get('steps', [])),
        'execution_time': result.get('execution_time'),
        'input_tokens': result.get('token_usage', {}).get('input_tokens'),
        'output_tokens': result.get('token_usage', {}).get('output_tokens'),
        'error': result.get('error') if not result.get('success') else None
    })

def log_tool_execution(
    tool_name: str,
    execution_time: float,
    success: bool,
    error: Optional[str] = None
):
    """Log individual tool execution"""
    
    logger.info("tool_execution", extra={
        'tool_name': tool_name,
        'execution_time': execution_time,
        'success': success,
        'error': error
    })
```

---

## Security Best Practices

### Input Validation

```python
from typing import Optional
import re

class InputValidator:
    """Validate user inputs for security"""
    
    MAX_TASK_LENGTH = 10000
    MAX_TASKS_PER_MINUTE = 100
    FORBIDDEN_PATTERNS = [
        r'__[a-zA-Z0-9_]+__',  # Dunder methods
        r'import\s+',  # Import statements
        r'exec\s*\(',  # Exec calls
        r'eval\s*\(',  # Eval calls
    ]
    
    def validate_task(self, task: str) -> Optional[str]:
        """Validate task description"""
        
        # Length check
        if not task or len(task) > self.MAX_TASK_LENGTH:
            raise ValueError(f"Task length must be 1-{self.MAX_TASK_LENGTH}")
        
        # Pattern checks
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, task):
                raise ValueError(f"Task contains forbidden pattern")
        
        # Sanitisation
        task = task.strip()
        
        return task

# Usage
validator = InputValidator()
try:
    clean_task = validator.validate_task(user_input)
except ValueError as e:
    logger.error(f"Invalid task: {e}")
    return None
```

### Code Execution Sandboxing

```python
from smolagents import CodeAgent

# Production sandbox configuration
class SecureAgent(CodeAgent):
    """Agent with enhanced security"""
    
    def __init__(self):
        super().__init__(
            # Use E2B for strongest isolation
            executor_type="e2b",
            
            # Strict timeout to prevent resource exhaustion
            timeout=30.0,
            
            # Limit iterations to prevent infinite loops
            max_steps=5,
            
            # Disable direct system access
            allow_shell_execution=False,
            allow_file_write=False
        )

# Alternative: Docker isolation
class DockerSecureAgent(CodeAgent):
    def __init__(self):
        super().__init__(
            executor_type="docker",
            timeout=30.0,
            max_steps=5,
            # Docker resource limits
            docker_memory_limit="512m",
            docker_cpu_limit="1"
        )
```

### Rate Limiting & Throttling

```python
import time
from collections import deque

class RateLimiter:
    """Prevent abuse through rate limiting"""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        max_tokens_per_minute: int = 1_000_000
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tokens = max_tokens_per_minute
        
        self.request_times = deque()
        self.token_counts = deque()
    
    def check_request_limit(self, user_id: str) -> bool:
        """Check if user can make request"""
        now = time.time()
        
        # Remove old timestamps outside window
        while self.request_times and self.request_times[0] < now - self.window_seconds:
            self.request_times.popleft()
        
        # Check limit
        if len(self.request_times) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False
        
        self.request_times.append(now)
        return True
    
    def check_token_limit(self, tokens: int) -> bool:
        """Check if token usage would exceed limit"""
        now = time.time()
        
        # Remove old counts
        while self.token_counts and self.token_counts[0][0] < now - 60:
            self.token_counts.popleft()
        
        total_tokens = sum(count for _, count in self.token_counts)
        
        if total_tokens + tokens > self.max_tokens:
            logger.warning("Token limit exceeded")
            return False
        
        self.token_counts.append((now, tokens))
        return True
```

---

## Error Handling & Resilience

### Comprehensive Error Handling

```python
from enum import Enum
from typing import Optional

class ErrorSeverity(Enum):
    RECOVERABLE = "recoverable"
    DEGRADED = "degraded"
    CRITICAL = "critical"

class AgentError(Exception):
    """Base agent error"""
    def __init__(self, message: str, severity: ErrorSeverity):
        self.message = message
        self.severity = severity
        super().__init__(message)

class ErrorHandler:
    """Handle errors gracefully"""
    
    def handle_llm_error(self, error: Exception) -> Optional[str]:
        """Handle LLM provider errors"""
        
        if "rate_limit" in str(error).lower():
            logger.warning("LLM rate limited, implementing backoff")
            # Implement exponential backoff
            time.sleep(5)
            return "retry"
        
        elif "quota" in str(error).lower():
            logger.error("LLM quota exceeded")
            return None  # Cannot retry
        
        elif "timeout" in str(error).lower():
            logger.warning("LLM timeout, retrying")
            return "retry"
        
        else:
            logger.error(f"Unknown LLM error: {error}")
            return None
    
    def handle_tool_error(self, tool_name: str, error: Exception) -> bool:
        """Handle tool execution errors"""
        
        # Tool-specific error handling
        if tool_name == "web_search":
            logger.warning(f"Web search failed: {error}")
            return False  # Cannot recover
        
        elif tool_name == "database_query":
            logger.error(f"Database query failed: {error}")
            # Attempt to retry with simplified query
            return True  # Can retry
        
        else:
            logger.error(f"Tool {tool_name} error: {error}")
            return False
    
    def handle_execution_error(self, error: Exception) -> Optional[str]:
        """Handle code execution errors"""
        
        if "timeout" in str(error).lower():
            logger.error("Code execution timeout")
            return None
        
        elif "memory" in str(error).lower():
            logger.error("Out of memory")
            return None
        
        elif "permission" in str(error).lower():
            logger.error("Permission denied")
            return None
        
        else:
            logger.warning(f"Execution error: {error}")
            return "retry"

# Usage
handler = ErrorHandler()

def run_with_error_handling(agent, task):
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            return agent.run(task)
        
        except Exception as e:
            action = handler.handle_llm_error(e)
            
            if action == "retry":
                retry_count += 1
                logger.info(f"Retrying ({retry_count}/{max_retries})")
                time.sleep(2 ** retry_count)  # Exponential backoff
            else:
                logger.error(f"Cannot recover from error: {e}")
                return None
    
    logger.error("Max retries exceeded")
    return None
```

---

## Deployment Options

### Containerised Deployment

```dockerfile
# Dockerfile for production deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml for production setup
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      HF_TOKEN: ${HF_TOKEN}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      REDIS_URL: redis://redis:6379
      LOG_LEVEL: INFO
    depends_on:
      - redis
      - postgres
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  redis:
    image: redis:7-alpine
    restart: always
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    restart: always
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml for production Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smolagent-api
  labels:
    app: smolagent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: smolagent
  template:
    metadata:
      labels:
        app: smolagent
    spec:
      containers:
      - name: agent-api
        image: myregistry/smolagent:latest
        ports:
        - containerPort: 8000
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: smolagent-secrets
              key: hf-token
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: smolagent-secrets
              key: openai-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: smolagent-service
spec:
  selector:
    app: smolagent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: smolagent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: smolagent-api
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

## Scaling Strategies

### Horizontal Scaling

```python
from typing import List

class LoadBalancedAgentPool:
    """Distribute requests across multiple agent instances"""
    
    def __init__(self, num_agents: int = 4):
        self.agents = [CodeAgent(...) for _ in range(num_agents)]
        self.request_counter = 0
    
    def get_next_agent(self) -> CodeAgent:
        """Round-robin agent selection"""
        agent = self.agents[self.request_counter % len(self.agents)]
        self.request_counter += 1
        return agent
    
    def process_batch(self, tasks: List[str]) -> List[str]:
        """Process multiple tasks in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = [
                executor.submit(self.get_next_agent().run, task)
                for task in tasks
            ]
            return [f.result() for f in futures]
```

### Vertical Scaling

```python
class OptimisedAgent(CodeAgent):
    """Agent optimised for high throughput"""
    
    def __init__(self):
        super().__init__(
            # Use faster models
            model=LiteLLMModel("groq/llama-3.3-70b-versatile"),
            
            # Reduce steps for efficiency
            max_steps=3,
            
            # Cache aggressively
            enable_cache=True,
            cache_ttl=3600,
            
            # Limit to simple tasks
            timeout=10.0
        )
```

---

## Testing & Quality Assurance

### Unit Testing Tools

```python
import pytest
from unittest.mock import Mock, patch

def test_agent_basic_execution():
    """Test basic agent execution"""
    agent = ProductionAgent()
    result = agent.run("What is 2+2?")
    assert result is not None
    assert "4" in str(result)

def test_error_handling():
    """Test error handling"""
    agent = ProductionAgent()
    
    # Test with invalid input
    with pytest.raises(ValueError):
        agent.run("")

def test_token_budget():
    """Test token budget enforcement"""
    budget = TokenBudgetManager(daily_limit=1000)
    
    assert budget.can_execute(500)
    budget.record_usage(TokenUsage(
        input_tokens=400,
        output_tokens=100,
        timestamp=datetime.datetime.now(),
        task_id="test",
        model="gpt-3.5"
    ))
    
    assert not budget.can_execute(600)  # Would exceed limit
```

---

## Observability Integration (Weights & Biases Weave)

### Weave Integration

```python
import weave
from smolagents import CodeAgent

# Initialise Weave tracking
weave.init("smolagent-production")

@weave.op()
def run_agent_with_tracking(task: str) -> dict:
    """Run agent with W&B Weave tracking"""
    
    agent = CodeAgent(...)
    
    # Weave automatically tracks execution
    result = agent.run(task, return_full_result=True)
    
    # Log custom metrics
    weave.log({
        'task': task,
        'success': result['success'],
        'steps': len(result['steps']),
        'tokens_used': result['token_usage']['input_tokens'] + 
                      result['token_usage']['output_tokens']
    })
    
    return result

# Create dataset for evaluation
@weave.op()
def evaluate_agent_quality(test_cases: list) -> float:
    """Evaluate agent on test cases"""
    
    results = []
    for test_case in test_cases:
        result = run_agent_with_tracking(test_case['prompt'])
        
        # Evaluate output quality
        score = evaluate_output(result['output'], test_case['expected'])
        results.append(score)
    
    average_score = sum(results) / len(results)
    weave.log({'average_score': average_score})
    
    return average_score
```

---

This production guide covers the essential aspects of deploying SmolAgents at scale. Follow these practices to ensure reliable, performant, and cost-effective agent systems in production.



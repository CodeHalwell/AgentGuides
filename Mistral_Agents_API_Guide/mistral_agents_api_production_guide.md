# Mistral Agents API: Production Deployment Guide

## 1. Infrastructure Setup

### Cloud Provider Configuration

**AWS Deployment (Recommended)**

```yaml
Region: us-east-1 (or closest to users)
Services:
  - ALB (Application Load Balancer)
  - ECS Fargate (Worker containers)
  - RDS PostgreSQL (Conversation storage)
  - ElastiCache Redis (Session/rate limit cache)
  - CloudWatch (Monitoring)
  - VPC with private subnets
```

**GCP Alternative**
```yaml
Services:
  - Cloud Load Balancing
  - Cloud Run (Serverless containers)
  - Cloud SQL (PostgreSQL)
  - Memorystore Redis
  - Cloud Logging/Monitoring
```

### Environment Variables

```bash
# Core Configuration
MISTRAL_API_KEY=your-api-key
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database
DB_HOST=prod-db.xxxxx.rds.amazonaws.com
DB_PORT=5432
DB_NAME=agents_prod
DB_USER=agentuser
DB_PASSWORD=${SECURE_PASSWORD}
DB_POOL_SIZE=20

# Cache
REDIS_HOST=prod-cache.xxxxx.cache.amazonaws.com
REDIS_PORT=6379
REDIS_PASSWORD=${SECURE_PASSWORD}

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Monitoring
DATADOG_API_KEY=${DATADOG_KEY}
DATADOG_SITE=datadoghq.com
```

---

## 2. Scaling Strategies

### Horizontal Scaling (Multiple Instances)

```python
from mistralai import Mistral
import os
from typing import Optional

class ScalableAgentManager:
    """Manages agents across multiple worker instances"""
    
    def __init__(self, worker_pool_size: int = 10):
        self.api_key = os.environ["MISTRAL_API_KEY"]
        self.pool_size = worker_pool_size
        self.clients = [
            Mistral(api_key=self.api_key) 
            for _ in range(worker_pool_size)
        ]
        self.current_idx = 0
    
    def get_next_client(self) -> Mistral:
        """Round-robin client selection"""
        client = self.clients[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.pool_size
        return client
    
    def process_conversation(self, agent_id: str, inputs: str) -> dict:
        """Process with load balancing"""
        client = self.get_next_client()
        response = client.beta.conversations.start(
            agent_id=agent_id,
            inputs=inputs
        )
        return response
```

### Vertical Scaling (Larger Instances)

```python
class PerformanceOptimizedManager:
    """Optimised for single powerful instance"""
    
    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    async def process_async(self, agent_id: str, inputs: str):
        """Process with concurrency control"""
        async with self.semaphore:
            response = await self.client.beta.conversations.start(
                agent_id=agent_id,
                inputs=inputs
            )
            return response
    
    async def batch_process(self, tasks: list) -> list:
        """Process multiple tasks concurrently"""
        results = await asyncio.gather(*[
            self.process_async(t['agent_id'], t['inputs'])
            for t in tasks
        ])
        return results
```

---

## 3. Monitoring & Observability

### Metrics Collection

```python
import time
from datadog import initialize, api
from datetime import datetime

class MetricsCollector:
    """Collect and send metrics to monitoring system"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    def track_conversation(self, agent_id: str, conversation_id: str):
        """Track conversation metrics"""
        start_time = time.time()
        
        try:
            history = self.client.beta.conversations.get_history(
                conversation_id=conversation_id
            )
            
            duration = time.time() - start_time
            entry_count = len(history.entries) if history.entries else 0
            
            # Send metrics
            metrics = {
                'agent_id': agent_id,
                'conversation_id': conversation_id,
                'entry_count': entry_count,
                'duration_seconds': duration,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'success'
            }
            
            self.send_to_monitoring(metrics)
            
        except Exception as e:
            self.send_to_monitoring({
                'agent_id': agent_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def send_to_monitoring(self, metrics: dict):
        """Send to Datadog/CloudWatch"""
        # Datadog
        api.Metric.send(
            metric='mistral.conversation',
            points=metrics['entry_count'],
            tags=[f"agent_id:{metrics['agent_id']}"]
        )
        
        # CloudWatch
        # cloudwatch.put_metric_data(...)
```

### Logging Setup

```python
import logging
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(timestamp)s %(level)s %(name)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/mistral_agents.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'formatter': 'json'
        }
    },
    'loggers': {
        'mistral_agents': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        },
        'mistralai': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('mistral_agents')
```

---

## 4. Error Handling & Recovery

### Retry Logic with Exponential Backoff

```python
import asyncio
from typing import Callable, Any

class RobustAgentClient:
    """Agent client with automatic retry logic"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Final attempt failed: {str(e)}")
                    raise
                
                delay = min(
                    self.base_delay * (2 ** attempt),
                    self.max_delay
                )
                
                logger.warning(
                    f"Attempt {attempt + 1} failed. "
                    f"Retrying in {delay}s: {str(e)}"
                )
                
                await asyncio.sleep(delay)
    
    async def start_conversation_resilient(
        self,
        agent_id: str,
        inputs: str
    ):
        """Start conversation with automatic retry"""
        async def _start():
            return self.client.beta.conversations.start(
                agent_id=agent_id,
                inputs=inputs
            )
        
        return await self.retry_with_backoff(_start)
```

### Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Prevent cascading failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60
    ):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker: attempting recovery")
            else:
                raise Exception("Circuit breaker OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error("Circuit breaker OPEN: too many failures")
    
    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return False
        
        timeout = timedelta(seconds=self.recovery_timeout)
        return datetime.now() - self.last_failure_time > timeout
```

---

## 5. Database Schema

### PostgreSQL Schema

```sql
-- Agents Table
CREATE TABLE agents (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model VARCHAR(50) NOT NULL,
    instructions TEXT,
    version INTEGER DEFAULT 1,
    tools JSONB,
    completion_args JSONB,
    owner_id VARCHAR(255),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(owner_id, name)
);

-- Conversations Table
CREATE TABLE conversations (
    id VARCHAR(255) PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    owner_id VARCHAR(255),
    name VARCHAR(255),
    description TEXT,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

-- Conversation Entries (Messages, Tool Calls, etc.)
CREATE TABLE conversation_entries (
    id VARCHAR(255) PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,  -- message.input, message.output, tool.execution
    role VARCHAR(20),           -- user, assistant, tool
    content TEXT,
    content_json JSONB,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    agent_id VARCHAR(255),
    model VARCHAR(50),
    tool_name VARCHAR(255),
    tool_call_id VARCHAR(255),
    parent_entry_id VARCHAR(255),
    metadata JSONB,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX idx_conv_agent ON conversations(agent_id);
CREATE INDEX idx_conv_owner ON conversations(owner_id);
CREATE INDEX idx_conv_created ON conversations(created_at DESC);
CREATE INDEX idx_entries_conv ON conversation_entries(conversation_id);
CREATE INDEX idx_entries_created ON conversation_entries(created_at DESC);
CREATE INDEX idx_entries_type ON conversation_entries(type);
CREATE INDEX idx_agent_owner ON agents(owner_id);
```

---

## 6. Rate Limiting

### Token Bucket Algorithm

```python
import time
from redis import Redis

class RateLimiter:
    """Token bucket rate limiting with Redis backend"""
    
    def __init__(
        self,
        redis_client: Redis,
        rate: float = 100,  # requests
        window: int = 60     # seconds
    ):
        self.redis = redis_client
        self.rate = rate
        self.window = window
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed"""
        key = f"rate_limit:{user_id}"
        now = time.time()
        
        # Get current token count
        data = self.redis.get(key)
        
        if not data:
            # First request
            self.redis.setex(
                key,
                self.window,
                self.rate - 1  # Use one token
            )
            return True
        
        tokens = float(data)
        
        if tokens > 0:
            # Token available
            self.redis.decr(key)
            return True
        
        return False
    
    def get_remaining(self, user_id: str) -> int:
        """Get remaining tokens"""
        key = f"rate_limit:{user_id}"
        tokens = self.redis.get(key)
        return int(tokens) if tokens else int(self.rate)
```

---

## 7. Performance Tuning

### Connection Pooling

```python
from mistralai import Mistral
import asyncio

class OptimisedClientPool:
    """Connection pool for efficient resource usage"""
    
    def __init__(self, pool_size: int = 10):
        self.pool = [
            Mistral(api_key=os.environ["MISTRAL_API_KEY"])
            for _ in range(pool_size)
        ]
        self.queue = asyncio.Queue(maxsize=pool_size)
        
        for client in self.pool:
            self.queue.put_nowait(client)
    
    async def execute(self, coro):
        """Execute with pooled client"""
        client = await self.queue.get()
        try:
            result = await coro
            return result
        finally:
            await self.queue.put(client)
```

### Batch Processing

```python
class BatchProcessor:
    """Process multiple conversations efficiently"""
    
    def __init__(self, batch_size: int = 50, client: Mistral = None):
        self.batch_size = batch_size
        self.client = client or Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.batch = []
    
    def add_task(self, agent_id: str, inputs: str) -> int:
        """Add task to batch"""
        self.batch.append((agent_id, inputs))
        
        if len(self.batch) >= self.batch_size:
            self.flush()
        
        return len(self.batch)
    
    def flush(self):
        """Process entire batch"""
        results = []
        for agent_id, inputs in self.batch:
            try:
                response = self.client.beta.conversations.start(
                    agent_id=agent_id,
                    inputs=inputs
                )
                results.append(response)
            except Exception as e:
                logger.error(f"Batch item failed: {e}")
        
        self.batch = []
        return results
```

---

## 8. Security Best Practices

### API Key Management

```python
import os
from typing import Optional

class SecureConfigManager:
    """Secure configuration and secrets management"""
    
    @staticmethod
    def get_api_key() -> str:
        """Get API key from environment (not hardcoded)"""
        key = os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY not set")
        return key
    
    @staticmethod
    def validate_request(request) -> bool:
        """Validate incoming requests"""
        # Check API key
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        
        # Check content-type
        if request.content_type != "application/json":
            return False
        
        return True
    
    @staticmethod
    def sanitise_inputs(inputs: str) -> str:
        """Sanitise user inputs"""
        # Remove potentially harmful content
        inputs = inputs.strip()
        
        # Prevent prompt injection
        forbidden_patterns = [
            "ignore previous instructions",
            "system override",
            "admin mode"
        ]
        
        for pattern in forbidden_patterns:
            if pattern.lower() in inputs.lower():
                logger.warning(f"Potential injection detected: {pattern}")
                inputs = inputs.replace(pattern, "[REDACTED]")
        
        return inputs
```

### Role-Based Access Control (RBAC)

```python
from enum import Enum

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    READ_ONLY = "read_only"

class AccessControl:
    """Manage access to agents and conversations"""
    
    PERMISSIONS = {
        Role.ADMIN: [
            'create_agent', 'update_agent', 'delete_agent',
            'list_conversations', 'read_conversation',
            'start_conversation', 'archive_conversation'
        ],
        Role.USER: [
            'list_conversations', 'read_conversation',
            'start_conversation', 'archive_conversation'
        ],
        Role.READ_ONLY: [
            'list_conversations', 'read_conversation'
        ]
    }
    
    def has_permission(self, role: Role, action: str) -> bool:
        """Check if role has permission"""
        return action in self.PERMISSIONS.get(role, [])
    
    def can_access_agent(self, user_id: str, agent_id: str) -> bool:
        """Check if user can access agent"""
        # Query database to verify ownership
        # or check if user is organisation member
        pass
```

---

## 9. CI/CD Integration

### GitHub Actions Example

{% raw %}
```yaml
name: Deploy Mistral Agents

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest --cov=mistral_agents tests/
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Deploy to ECS
        run: |
          ./scripts/deploy.sh
```
{% endraw %}

---

**This production guide provides everything needed to deploy and maintain Mistral Agents API at scale.**


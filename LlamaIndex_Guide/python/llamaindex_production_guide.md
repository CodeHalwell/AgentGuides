# LlamaIndex Production Guide

Deploying LlamaIndex Applications to Production: Best Practices, Deployment Strategies, and Optimization Techniques

---

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Deployment Strategies](#deployment-strategies)
3. [Performance Optimization](#performance-optimization)
4. [Monitoring and Observability](#monitoring-and-observability)
5. [Security and Access Control](#security-and-access-control)
6. [Error Handling and Recovery](#error-handling-and-recovery)
7. [Scaling Strategies](#scaling-strategies)
8. [Cost Optimization](#cost-optimization)
9. [Testing and Quality Assurance](#testing-and-quality-assurance)
10. [DevOps and CI/CD](#devops-and-cicd)

---

## Production Architecture

### Production-Ready LlamaIndex Stack

```python
# requirements-prod.txt
llama-index==0.14.6
llama-index-core==0.2.1
llama-index-llms-openai==0.3.1
llama-index-embeddings-openai==0.2.1
llama-index-vector-stores-chroma==0.4.1
llama-index-vector-stores-pinecone==0.4.1

# Web framework
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.0

# Monitoring and logging
prometheus-client==0.19.0
python-json-logger==2.0.7
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0

# Database and caching
redis==5.0.1
sqlalchemy==2.0.23
psycopg2-binary==2.9.9

# Security
python-jose==3.3.0
passlib==1.7.4
python-dotenv==1.0.0

# Testing
pytest==7.4.3
pytest-asyncio==0.23.1
pytest-cov==4.1.0

# Utilities
tenacity==8.2.3
structlog==23.3.0
```

### Multi-Tier Architecture

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging

# Configure structured logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# Global state for lifespan management
class AppState:
    llm = None
    embed_model = None
    indexes = {}
    vector_store = None
    query_engines = {}
    memory_store = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    logger.info("application_startup")
    
    # Initialize expensive resources
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    import pinecone
    
    # Setup vector store
    pinecone.init(
        api_key="your-key",
        environment="your-env"
    )
    pc_index = pinecone.Index("llamaindex-prod")
    app_state.vector_store = PineconeVectorStore(
        pinecone_index=pc_index,
        namespace="production",
    )
    
    # Setup LLM and embeddings
    app_state.llm = OpenAI(
        model="gpt-4",
        temperature=0.7,
    )
    app_state.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
    )
    
    logger.info("resources_initialized")
    
    yield
    
    # Cleanup
    logger.info("application_shutdown")
    pinecone.deinit()

app = FastAPI(lifespan=lifespan, title="LlamaIndex Production API")

# Middleware for request tracking
@app.middleware("http")
async def add_request_id(request, call_next):
    import uuid
    request.state.request_id = str(uuid.uuid4())
    
    logger.info(
        "request_start",
        request_id=request.state.request_id,
        method=request.method,
        path=request.url.path,
    )
    
    response = await call_next(request)
    
    logger.info(
        "request_end",
        request_id=request.state.request_id,
        status_code=response.status_code,
    )
    
    return response
```

---

## Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile for LlamaIndex application
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Local Development


```yaml
# docker-compose.yml
version: '3.8'

services:
  # Main API
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@postgres:5432/llamaindex
    depends_on:
      - redis
      - postgres
      - vector_store
    volumes:
      - ./:/app
    command: uvicorn main:app --host 0.0.0.0 --reload

  # Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # PostgreSQL for persistence
  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=llamaindex
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Chroma vector store
  vector_store:
    image: ghcr.io/chroma-core/chroma:latest
    ports:
      - "8001:8000"
    environment:
      - CHROMA_DB_IMPL=duckdb
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  redis_data:
  postgres_data:
  chroma_data:
```

### Kubernetes Deployment

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamaindex-api
  labels:
    app: llamaindex
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llamaindex
  template:
    metadata:
      labels:
        app: llamaindex
    spec:
      containers:
      - name: api
        image: llamaindex-api:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llamaindex-secrets
              key: openai-key
        - name: REDIS_URL
          value: redis://redis-service:6379
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: llamaindex-secrets
              key: database-url
        
        # Resource limits
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        
        # Liveness probe
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        
        # Readiness probe
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        
        # Volume mounts
        volumeMounts:
        - name: config
          mountPath: /app/config
      
      volumes:
      - name: config
        configMap:
          name: llamaindex-config

---
apiVersion: v1
kind: Service
metadata:
  name: llamaindex-service
spec:
  selector:
    app: llamaindex
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llamaindex-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llamaindex-api
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

## Performance Optimization

### Query Optimization

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import BM25Retriever
import time
from functools import wraps

# Performance monitoring decorator
def track_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(
                "operation_complete",
                operation=func.__name__,
                duration_ms=duration * 1000,
                status="success"
            )
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                "operation_failed",
                operation=func.__name__,
                duration_ms=duration * 1000,
                error=str(e),
            )
            raise
    return wrapper

# Optimized query engine with caching
from functools import lru_cache
import hashlib

class OptimizedQueryEngine:
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.cache = {}
        self.query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
        )
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    @track_performance
    async def query(self, query: str, use_cache: bool = True):
        """Query with optional caching."""
        cache_key = self._get_cache_key(query)
        
        # Check cache
        if use_cache and cache_key in self.cache:
            logger.info("cache_hit", query=query)
            return self.cache[cache_key]
        
        # Execute query
        response = await self.query_engine.aquery(query)
        
        # Cache result (with TTL in production)
        self.cache[cache_key] = response
        
        logger.info("cache_miss", query=query)
        return response
    
    def clear_cache(self):
        """Clear query cache."""
        self.cache.clear()
        logger.info("cache_cleared")
```

### Batch Processing

```python
from typing import List
from llama_index.core import Document

class BatchProcessor:
    """Process multiple queries in batches for efficiency."""
    
    def __init__(self, query_engine, batch_size: int = 10):
        self.query_engine = query_engine
        self.batch_size = batch_size
    
    async def process_batch(self, queries: List[str]) -> List[str]:
        """Process multiple queries efficiently."""
        results = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            
            logger.info(
                "processing_batch",
                batch_num=i // self.batch_size + 1,
                batch_size=len(batch)
            )
            
            # Process queries in parallel
            import asyncio
            batch_results = await asyncio.gather(
                *[self.query_engine.query(q) for q in batch]
            )
            
            results.extend(batch_results)
        
        return results
    
    async def stream_batch(self, queries: List[str]):
        """Stream results as they complete."""
        import asyncio
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            
            batch_results = await asyncio.gather(
                *[self.query_engine.query(q) for q in batch]
            )
            
            for result in batch_results:
                yield result
```

---

## Monitoring and Observability

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
query_counter = Counter(
    'llamaindex_queries_total',
    'Total number of queries',
    ['status', 'query_type']
)

query_duration = Histogram(
    'llamaindex_query_duration_seconds',
    'Query duration in seconds',
    ['query_type'],
    buckets=(0.1, 0.5, 1, 2, 5, 10)
)

active_queries = Gauge(
    'llamaindex_active_queries',
    'Number of active queries'
)

token_usage = Counter(
    'llamaindex_tokens_used',
    'Total tokens used',
    ['model', 'type']  # type: input, output
)

cache_hits = Counter(
    'llamaindex_cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

# Middleware to track metrics
from fastapi import Request

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    active_queries.inc()
    
    start_time = time.time()
    try:
        response = await call_next(request)
        status = "success"
    except Exception as e:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        query_type = request.url.path.split('/')[-1]
        
        query_counter.labels(
            status=status,
            query_type=query_type
        ).inc()
        
        query_duration.labels(
            query_type=query_type
        ).observe(duration)
        
        active_queries.dec()
    
    return response
```

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

# Use in operations
@app.post("/query")
async def query_endpoint(query: str):
    with tracer.start_as_current_span("process_query") as span:
        span.set_attribute("query", query)
        
        with tracer.start_as_current_span("retrieve_documents"):
            # Retrieval operation
            pass
        
        with tracer.start_as_current_span("generate_response"):
            # Generation operation
            pass
        
        return {"result": "response"}
```

### Log Aggregation

```python
import structlog
from pythonjsonlogger import jsonlogger
import logging

# Configure JSON logging for ELK stack
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

# Structured logging
logger = structlog.get_logger()

# Usage
logger.info(
    "query_processed",
    query_id="q123",
    duration_ms=150,
    tokens_used=245,
    cache_hit=False,
    user_id="user456",
)
```

---

## Security and Access Control

### Authentication and Authorization

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os

security = HTTPBearer()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthCredentials = Depends(security)):
    """Validate token and extract user."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return user_id

# Protected endpoint
@app.post("/query")
async def protected_query(query: str, user_id: str = Depends(get_current_user)):
    """Query endpoint protected by authentication."""
    logger.info("user_query", user_id=user_id, query=query)
    # Process query
    return {"result": "response"}
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    logger.warning("rate_limit_exceeded", client=get_remote_address(request))
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests"},
    )

# Apply rate limits
@app.post("/query")
@limiter.limit("10/minute")
async def query_with_rate_limit(request: Request, query: str):
    """Query endpoint with rate limiting."""
    return await query_endpoint(query)
```

### Data Encryption

```python
from cryptography.fernet import Fernet
import os

# Encryption key management
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
cipher_suite = Fernet(ENCRYPTION_KEY)

class EncryptedStorage:
    """Store sensitive data encrypted."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def store_query(self, user_id: str, query: str):
        """Store query encrypted."""
        encrypted_query = cipher_suite.encrypt(query.encode())
        self.db.execute(
            "INSERT INTO queries (user_id, encrypted_query) VALUES (?, ?)",
            (user_id, encrypted_query)
        )
    
    def retrieve_query(self, query_id: int) -> str:
        """Retrieve and decrypt query."""
        result = self.db.execute(
            "SELECT encrypted_query FROM queries WHERE id = ?",
            (query_id,)
        ).fetchone()
        
        if result:
            encrypted_query = result[0]
            decrypted_query = cipher_suite.decrypt(encrypted_query).decode()
            return decrypted_query
        
        return None
```

---

## Error Handling and Recovery

### Comprehensive Error Handling

```python
from typing import Optional
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error_id: str
    message: str
    status_code: int
    timestamp: str
    request_id: Optional[str] = None

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    import uuid
    from datetime import datetime
    
    error_id = str(uuid.uuid4())
    
    logger.error(
        "unhandled_exception",
        error_id=error_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        request_path=request.url.path,
        request_id=getattr(request.state, "request_id", None),
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_id=error_id,
            message="Internal server error",
            status_code=500,
            timestamp=datetime.utcnow().isoformat(),
            request_id=getattr(request.state, "request_id", None),
        ).dict(),
    )

# Specific exception handlers
class LLMException(Exception):
    pass

@app.exception_handler(LLMException)
async def llm_exception_handler(request: Request, exc: LLMException):
    logger.error("llm_error", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "LLM service unavailable"},
    )

# Retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def query_with_retry(query: str):
    """Query with automatic retry on failure."""
    try:
        return await query_engine.aquery(query)
    except Exception as e:
        logger.error("query_failed", attempt=query_with_retry.retry.statistics)
        raise
```

### Graceful Degradation

```python
class ResilientQueryEngine:
    """Query engine with fallback strategies."""
    
    def __init__(self, primary_engine, fallback_engines):
        self.primary_engine = primary_engine
        self.fallback_engines = fallback_engines
    
    async def query(self, query: str):
        """Query with fallback."""
        engines = [self.primary_engine] + self.fallback_engines
        
        for i, engine in enumerate(engines):
            try:
                logger.info("trying_engine", engine_num=i)
                result = await engine.aquery(query)
                
                if i > 0:
                    logger.warning("fallback_used", fallback_num=i)
                
                return result
            
            except Exception as e:
                logger.warning(
                    "engine_failed",
                    engine_num=i,
                    error=str(e),
                )
                
                if i == len(engines) - 1:
                    raise
        
        raise Exception("All query engines failed")
```

---

## Scaling Strategies

### Horizontal Scaling

```python
# Load balancer configuration (nginx)
upstream llamaindex_backend {
    least_conn;
    server api1.internal:8000 weight=1;
    server api2.internal:8000 weight=1;
    server api3.internal:8000 weight=1;
}

server {
    listen 80;
    server_name api.llamaindex.com;
    
    location / {
        proxy_pass http://llamaindex_backend;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Request-ID $request_id;
    }
}
```

### Database Sharding

```python
from sqlalchemy import create_engine

class ShardedRepository:
    """Database repository with sharding."""
    
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.engines = [
            create_engine(f"postgresql://user:pwd@shard{i}.db/llamaindex")
            for i in range(num_shards)
        ]
    
    def get_shard(self, user_id: str) -> int:
        """Determine shard for user."""
        return hash(user_id) % self.num_shards
    
    def store_query(self, user_id: str, query: str):
        """Store query in appropriate shard."""
        shard_idx = self.get_shard(user_id)
        engine = self.engines[shard_idx]
        
        with engine.connect() as conn:
            conn.execute(
                "INSERT INTO queries (user_id, query) VALUES (?, ?)",
                (user_id, query)
            )
```

### Caching Strategy

```python
import redis
import json
from datetime import timedelta

class DistributedCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=True)
    
    def get_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key."""
        return f"{prefix}:{identifier}"
    
    async def get(self, key: str):
        """Get from cache."""
        value = self.redis.get(key)
        if value:
            logger.info("cache_hit", key=key)
            return json.loads(value)
        logger.info("cache_miss", key=key)
        return None
    
    async def set(
        self,
        key: str,
        value: dict,
        ttl: timedelta = timedelta(hours=1)
    ):
        """Set cache with TTL."""
        self.redis.setex(
            key,
            int(ttl.total_seconds()),
            json.dumps(value)
        )
    
    async def delete(self, key: str):
        """Delete from cache."""
        self.redis.delete(key)
    
    async def clear_prefix(self, prefix: str):
        """Clear all keys with prefix."""
        pattern = f"{prefix}:*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)
```

---

## Cost Optimization

### Token Usage Tracking

```python
class TokenUsageTracker:
    """Track and optimize token usage."""
    
    def __init__(self):
        self.usage = {}
        self.limits = {
            "gpt-4": 100000,
            "gpt-3.5-turbo": 500000,
        }
    
    def log_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Log token usage."""
        if model not in self.usage:
            self.usage[model] = {"input": 0, "output": 0}
        
        self.usage[model]["input"] += input_tokens
        self.usage[model]["output"] += output_tokens
        
        # Alert if approaching limit
        total = self.usage[model]["input"] + self.usage[model]["output"]
        if total > self.limits.get(model, float('inf')) * 0.8:
            logger.warning(
                "token_limit_warning",
                model=model,
                total_tokens=total,
                limit=self.limits.get(model)
            )
    
    def get_cost_estimate(self) -> dict:
        """Estimate API costs."""
        # Pricing as of 2024
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
        
        total_cost = 0
        for model, tokens in self.usage.items():
            if model in pricing:
                input_cost = (tokens["input"] / 1000) * pricing[model]["input"]
                output_cost = (tokens["output"] / 1000) * pricing[model]["output"]
                total_cost += input_cost + output_cost
        
        return {
            "total_cost": total_cost,
            "usage": self.usage,
        }

# Global tracker
token_tracker = TokenUsageTracker()
```

### Query Optimization for Cost

```python
class CostOptimizedQueryEngine:
    """Query engine that optimizes for cost."""
    
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.cost_tracker = TokenUsageTracker()
    
    async def query(self, query: str, max_budget: float = 0.10):
        """Query within cost budget."""
        # Use cheaper model for simple queries
        if len(query) < 50 and not any(c in query for c in "[]{}()"):
            cheap_llm = OpenAI(model="gpt-3.5-turbo")
            engine = self.index.as_query_engine(llm=cheap_llm)
        else:
            engine = self.index.as_query_engine(llm=self.llm)
        
        response = await engine.aquery(query)
        
        # Track cost
        cost_estimate = self.cost_tracker.get_cost_estimate()
        if cost_estimate["total_cost"] > max_budget:
            logger.warning(
                "cost_exceeded",
                cost=cost_estimate["total_cost"],
                budget=max_budget
            )
        
        return response
```

---

## Testing and Quality Assurance

### Unit Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.complete = Mock(return_value="Test response")
    return llm

@pytest.fixture
def mock_embed():
    """Mock embeddings."""
    embed = Mock()
    embed.get_text_embedding = Mock(
        return_value=[0.1] * 1536
    )
    return embed

@pytest.mark.asyncio
async def test_query_engine(mock_llm, mock_embed):
    """Test query engine."""
    from llama_index.core import VectorStoreIndex, Document
    
    docs = [Document(text="Test content")]
    index = VectorStoreIndex.from_documents(
        docs,
        llm=mock_llm,
        embed_model=mock_embed,
    )
    
    query_engine = index.as_query_engine()
    response = await query_engine.aquery("Test question")
    
    assert response is not None
    mock_llm.complete.assert_called()

def test_authentication():
    """Test authentication."""
    token = create_access_token({"sub": "user123"})
    assert token is not None
    
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "user123"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_end_to_end_query():
    """Test complete query flow."""
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Setup
    response = client.post(
        "/documents",
        json={"text": "Test document"}
    )
    assert response.status_code == 200
    
    # Query
    query_response = client.post(
        "/query",
        json={"query": "What is this about?"}
    )
    assert query_response.status_code == 200
    assert "response" in query_response.json()
```

### Performance Tests

```python
import time
import statistics

def test_query_performance():
    """Test query performance."""
    times = []
    
    for _ in range(10):
        start = time.time()
        result = query_engine.query("Test query")
        duration = time.time() - start
        times.append(duration)
    
    avg_time = statistics.mean(times)
    p99_time = sorted(times)[9]
    
    assert avg_time < 0.5, f"Average query time {avg_time}s exceeds 0.5s"
    assert p99_time < 1.0, f"P99 query time {p99_time}s exceeds 1.0s"
```

---

## DevOps and CI/CD

### GitHub Actions Pipeline

{% raw %}
```yaml

# .github/workflows/deploy.yml
name: Deploy LlamaIndex

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-prod.txt
        pip install -r requirements-test.txt
    
    - name: Lint
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check .
    
    - name: Type check
      run: mypy . --ignore-missing-imports
    
    - name: Run tests
      run: pytest --cov=llamaindex_app --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost/test
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  deploy:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        push: true
        tags: |
          ghcr.io/${{ github.repository }}:latest
          ghcr.io/${{ github.repository }}:${{ github.sha }}
    
    - name: Deploy to K8s
      run: |
        kubectl set image deployment/llamaindex-api \
          api=ghcr.io/${{ github.repository }}:${{ github.sha }} \
          --record
        kubectl rollout status deployment/llamaindex-api
      env:
        KUBECONFIG: ${{ secrets.KUBE_CONFIG }}

```
{% endraw %}


---

This production guide provides comprehensive strategies for deploying and maintaining LlamaIndex applications at scale. Each section includes ready-to-use code examples and best practices for production environments.


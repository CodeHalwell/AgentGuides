# LangGraph: Production Deployment & Best Practices

---

## Pre-Deployment Checklist

### ✅ Code & Architecture

- [ ] Graph is compiled with proper checkpointer (not in-memory)
- [ ] All nodes have timeout handling
- [ ] Conditional edges have explicit exit conditions (no infinite loops)
- [ ] State schemas properly defined with TypedDict
- [ ] Reducers correctly handle all node outputs
- [ ] No hardcoded API keys or credentials (use env vars)
- [ ] Error handling in place for all external API calls
- [ ] Max recursion depth set appropriately
- [ ] Graph tested with edge cases and failure scenarios
- [ ] Tool definitions have proper error handling

### ✅ Memory & Persistence

- [ ] Checkpointer properly configured (PostgreSQL for production)
- [ ] Store configured for long-term memory needs
- [ ] Database connections pooled and configured for high concurrency
- [ ] Backup strategy for checkpoint database defined
- [ ] Data retention policy established
- [ ] Checkpoint cleanup/archival plan in place
- [ ] Thread isolation verified (no data leaks between users)

### ✅ Testing

- [ ] Unit tests for each node
- [ ] Integration tests for complete workflows
- [ ] Load testing with expected concurrent users
- [ ] Failure recovery tested (simulate crashes, database unavailability)
- [ ] Interrupt/human-approval workflows tested
- [ ] State persistence tested across restarts
- [ ] Tool API failure scenarios tested
- [ ] LLM timeout/rate limiting handled

### ✅ Observability

- [ ] LangSmith integration configured for tracing
- [ ] Structured logging in place (JSON format)
- [ ] Metrics collection (latency, errors, token usage)
- [ ] Alert rules defined for critical failures
- [ ] Dashboard created for monitoring
- [ ] Error reporting (Sentry/similar) configured
- [ ] Performance baselines established

### ✅ Security

- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] Input validation on all user-provided data
- [ ] SQL injection prevention (use parameterized queries)
- [ ] Secrets management (vault/secrets manager)
- [ ] HTTPS/TLS enabled
- [ ] CORS properly configured
- [ ] Role-based access control for different agents
- [ ] Audit logging of sensitive operations
- [ ] Data encryption at rest (database)

### ✅ Infrastructure

- [ ] Docker image created and tested
- [ ] Container registry configured
- [ ] Kubernetes manifests prepared (if using K8s)
- [ ] Load balancer configured
- [ ] Auto-scaling policies defined
- [ ] Disaster recovery plan documented
- [ ] Rollback procedure tested
- [ ] Resource limits/requests defined

### ✅ Documentation

- [ ] Architecture documented with diagrams
- [ ] API documentation generated
- [ ] Runbooks for common issues
- [ ] Deployment procedure documented
- [ ] Emergency response procedures defined
- [ ] README with setup instructions
- [ ] Example requests/responses documented

---

## Deployment Strategies

### Strategy 1: Docker + Kubernetes

```yaml
# docker-compose.yml (development/testing)
version: '3.8'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: langgraph
      POSTGRES_USER: lguser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lguser"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  agent:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://lguser:${DB_PASSWORD}@postgres:5432/langgraph
      REDIS_URL: redis://redis:6379
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      LOG_LEVEL: INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
```

```dockerfile
# Dockerfile (production)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check script
COPY healthcheck.py /app/healthcheck.py

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Start LangGraph server
CMD ["langgraph", "run", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--config", "langgraph.json"]
```

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-agent
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
      app: langgraph-agent
  template:
    metadata:
      labels:
        app: langgraph-agent
    spec:
      serviceAccountName: langgraph-agent
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: agent
        image: myregistry/langgraph-agent:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: database-url
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: langgraph-secrets
              key: anthropic-key
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: langgraph-config
---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-agent-service
  namespace: production
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: langgraph-agent
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langgraph-agent-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langgraph-agent
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

### Strategy 2: LangGraph Cloud (Managed)

```bash
# Login to LangGraph Cloud
langgraph auth login

# Deploy via CLI
langgraph deploy

# Or push to Git for CD pipeline
git push origin main  # Triggers webhook → auto-deployment
```

### Strategy 3: Self-Hosted with Docker Compose

```bash
# Build for production
docker build -t my-agent:v1.0.0 .

# Push to registry
docker tag my-agent:v1.0.0 myregistry.azurecr.io/my-agent:v1.0.0
docker push myregistry.azurecr.io/my-agent:v1.0.0

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Scale
docker-compose -f docker-compose.prod.yml up -d --scale agent=5

# Monitor
docker-compose logs -f agent
```

---

## Configuration Management

### Production Environment Variables

```bash
# .env.production
# Database
DATABASE_URL=postgresql://user:password@pg-prod.company.com:5432/langgraph_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-v0-...
OPENAI_API_KEY=sk-...

# External APIs
TAVILY_API_KEY=...
SERPAPI_KEY=...

# Monitoring
LANGSMITH_API_KEY=...
SENTRY_DSN=...
LOG_LEVEL=INFO
JSON_LOGS=true

# Security
ALLOWED_ORIGINS=https://example.com,https://app.example.com
API_KEY_HEADER=X-API-Key
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Performance
REQUEST_TIMEOUT=30
CHECKPOINT_CLEANUP_DAYS=90
MAX_CONCURRENT_GRAPHS=100
```

### langgraph.json Configuration

```json
{
  "dependencies": [
    "langchain_anthropic",
    "langchain_tavily",
    "langchain_openai",
    "psycopg2-binary",
    "./src"
  ],
  "graphs": {
    "main_agent": "./src/agents.py:main_graph",
    "research_agent": "./src/agents.py:research_graph",
    "support_agent": "./src/agents.py:support_graph"
  },
  "env": "./.env",
  "python_version": "3.11",
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y postgresql-client",
    "ENV LOG_LEVEL=INFO"
  ]
}
```

---

## Monitoring & Observability

### Logging Configuration

```python
# logging_config.py
import logging
import json
import sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for better parsing."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "thread_id"):
            log_data["thread_id"] = record.thread_id
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
            
        return json.dumps(log_data)

def setup_logging():
    """Configure JSON logging for production."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# Use in code
logger = logging.getLogger(__name__)

def process_request(user_id, thread_id):
    start = time.time()
    
    try:
        # Do work
        result = graph.invoke({"query": "..."}, config={...})
        
        elapsed = (time.time() - start) * 1000
        
        # Log with extra context
        extra = {"user_id": user_id, "thread_id": thread_id, "latency_ms": elapsed}
        logger.info(f"Request processed successfully", extra=extra)
        
        return result
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True, extra=extra)
        raise
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
graph_executions = Counter(
    'graph_executions_total',
    'Total graph executions',
    ['graph_name', 'status']
)

execution_latency = Histogram(
    'graph_execution_seconds',
    'Graph execution latency',
    ['graph_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 10]
)

active_executions = Gauge(
    'graph_executions_active',
    'Currently running executions',
    ['graph_name']
)

token_usage = Counter(
    'token_usage_total',
    'Total tokens used',
    ['model', 'token_type']  # input, output
)

# Use in code
def run_with_metrics(graph_name, graph, input_data, config):
    active_executions.labels(graph_name=graph_name).inc()
    start = time.time()
    
    try:
        result = graph.invoke(input_data, config=config)
        
        elapsed = time.time() - start
        execution_latency.labels(graph_name=graph_name).observe(elapsed)
        graph_executions.labels(graph_name=graph_name, status="success").inc()
        
        # Track tokens
        if hasattr(result, 'usage_metadata'):
            token_usage.labels(
                model="claude", 
                token_type="input"
            ).inc(result.usage_metadata.input_tokens)
            token_usage.labels(
                model="claude",
                token_type="output"
            ).inc(result.usage_metadata.output_tokens)
        
        return result
    except Exception as e:
        graph_executions.labels(graph_name=graph_name, status="failure").inc()
        raise
    finally:
        active_executions.labels(graph_name=graph_name).dec()
```

### Health Checks

{% raw %}
```python
# health_check.py
from fastapi import FastAPI, HTTPException
from sqlalchemy import text
import asyncio

app = FastAPI()

async def check_database():
    """Verify database connectivity."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False

async def check_llm():
    """Verify LLM API connectivity."""
    try:
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        result = model.invoke("ping")
        return len(result.content) > 0
    except Exception as e:
        logger.error(f"LLM check failed: {e}")
        return False

async def check_checkpoint_store():
    """Verify checkpoint storage is functional."""
    try:
        test_config = {"configurable": {"thread_id": "health-check"}}
        state = graph.get_state(test_config)
        return True
    except Exception as e:
        logger.error(f"Checkpoint store check failed: {e}")
        return False

@app.get("/health")
async def health_check():
    """Liveness probe - quick check."""
    return {"status": "alive"}

@app.get("/ready")
async def readiness_check():
    """Readiness probe - full dependency check."""
    checks = {
        "database": await check_database(),
        "llm": await check_llm(),
        "checkpoints": await check_checkpoint_store()
    }
    
    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks
    }, status_code
```
{% endraw %}

---

## Performance Optimization

### 1. Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Database connection pool
engine = create_engine(
    os.getenv("DATABASE_URL"),
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    connect_args={"timeout": 10}
)

# Use with checkpoint saver
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    os.getenv("DATABASE_URL"),
    pool_size=20
)
```

### 2. Batch Processing

{% raw %}
```python
# Process multiple requests efficiently
async def batch_process(requests: list[dict], batch_size: int = 10):
    """Process multiple graph invocations efficiently."""
    
    configs = [
        {"configurable": {"thread_id": f"batch-{uuid.uuid4()}"}}
        for _ in requests
    ]
    
    # Use async batch for parallel execution
    results = await graph.abatch(requests, configs=configs)
    
    return results

# Example
requests = [
    {"query": "Research topic A"},
    {"query": "Research topic B"},
    {"query": "Research topic C"},
    # ... 100 more
]

results = asyncio.run(batch_process(requests, batch_size=10))
```
{% endraw %}

### 3. Caching Strategy

```python
import redis.asyncio as redis
from functools import wraps

# Redis cache for expensive operations
redis_client = None

async def init_cache():
    global redis_client
    redis_client = await redis.from_url(os.getenv("REDIS_URL"))

async def cached_graph_call(cache_key: str, graph, input_data, config, ttl=3600):
    """Cache graph results in Redis."""
    
    # Check cache
    cached = await redis_client.get(cache_key)
    if cached:
        logger.info(f"Cache hit: {cache_key}")
        return json.loads(cached)
    
    # Execute graph
    result = graph.invoke(input_data, config=config)
    
    # Store in cache
    await redis_client.setex(
        cache_key,
        ttl,
        json.dumps(result)
    )
    
    return result

# Use with semantic caching for LLM outputs
async def semantic_cache_lookup(query: str, embeddings_model) -> Optional[str]:
    """Find similar cached results using vector search."""
    
    # Embed query
    query_embedding = await embeddings_model.aembed_query(query)
    
    # Search Redis with vector similarity
    results = await redis_client.ft("queries").search(
        f"@query_vector:[{query_embedding}]",
        {"return": ["result"]}
    )
    
    if results:
        return results[0]["result"]
    
    return None
```

### 4. Request Timeout Handling

{% raw %}
```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def timeout_context(seconds: int):
    """Ensure request completes within timeout."""
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {seconds}s")
        raise HTTPException(status_code=504, detail="Request timeout")

# Use in endpoint
@app.post("/invoke")
async def invoke_graph(request: GraphRequest):
    async with timeout_context(30):  # 30 second timeout
        result = graph.invoke(
            request.input,
            config={"configurable": {"thread_id": request.thread_id}}
        )
    return result
```
{% endraw %}

---

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh - Daily checkpoint backup

BACKUP_DIR="/backups/langgraph"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_URL="${DATABASE_URL}"

# Create backup directory
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

# Backup PostgreSQL
pg_dump "$DB_URL" | gzip > "$BACKUP_DIR/$TIMESTAMP/checkpoints.sql.gz"

# Upload to S3
aws s3 cp "$BACKUP_DIR/$TIMESTAMP/checkpoints.sql.gz" \
    "s3://backups/langgraph/checkpoints_$TIMESTAMP.sql.gz"

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed: $TIMESTAMP"
```

### Restore Procedure

```bash
#!/bin/bash
# restore.sh - Restore from backup

BACKUP_TIMESTAMP=$1
BACKUP_FILE="checkpoints_$BACKUP_TIMESTAMP.sql.gz"

if [ -z "$BACKUP_TIMESTAMP" ]; then
    echo "Usage: ./restore.sh TIMESTAMP"
    echo "Example: ./restore.sh 20240101_120000"
    exit 1
fi

# Download from S3
aws s3 cp "s3://backups/langgraph/$BACKUP_FILE" /tmp/

# Restore to database
gunzip -c "/tmp/$BACKUP_FILE" | psql "$DATABASE_URL"

echo "Restored from backup: $BACKUP_TIMESTAMP"
```

### Failover Procedure

```yaml
# Kubernetes failover configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: langgraph-failover-config
data:
  failover-procedure: |
    1. Detect primary database failure (connection timeout > 5s)
    2. Switch all connections to replica database
    3. Promote read-replica to primary
    4. Update DATABASE_URL env var in deployment
    5. Restart pods to pick up new connection string
    6. Verify checkpoints are accessible
    7. Resume in-flight graph executions

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: langgraph-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: langgraph-agent
```

---

## Performance Benchmarks

### Expected Metrics

```
Graph Execution Latency (P95):
- Simple chat:              100-200ms
- With tool call:           500-1000ms
- Multi-agent supervisor:   2-5s
- Tree-of-thoughts:         10-30s

Throughput:
- Single instance:          20-50 req/s
- With 5 replicas:          100-250 req/s
- With auto-scaling (10):   200-500 req/s

Database Performance:
- Checkpoint write:         10-50ms
- Checkpoint read:          5-20ms
- State history query:      100-500ms

Memory Usage:
- Per graph execution:      50-200MB
- Idle agent process:       200-300MB
- With 100 active threads:  1-2GB
```

### Load Testing Script

```python
# load_test.py
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from statistics import mean, stdev

async def load_test(
    url: str,
    num_requests: int = 1000,
    concurrency: int = 10
):
    """Run load test against graph endpoint."""
    
    results = []
    errors = []
    
    async def make_request(request_id):
        try:
            start = time.time()
            response = requests.post(
                f"{url}/invoke",
                json={
                    "input": f"Request {request_id}",
                    "thread_id": f"load-test-{request_id}"
                },
                timeout=60
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                results.append(elapsed)
            else:
                errors.append(f"Status {response.status_code}")
        except Exception as e:
            errors.append(str(e))
    
    # Run concurrent requests
    tasks = [
        make_request(i)
        for i in range(num_requests)
    ]
    
    start_time = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Compute statistics
    print(f"\n=== Load Test Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_requests/total_time:.2f} req/s")
    print(f"Mean latency: {mean(results):.2f}s")
    print(f"Median latency: {sorted(results)[len(results)//2]:.2f}s")
    print(f"P95 latency: {sorted(results)[int(len(results)*0.95)]:.2f}s")
    print(f"P99 latency: {sorted(results)[int(len(results)*0.99)]:.2f}s")
    
    if errors:
        print(f"\nErrors: {errors[:10]}")

# Run test
if __name__ == "__main__":
    asyncio.run(load_test("http://localhost:8000", num_requests=1000, concurrency=10))
```

---

## Rollback Procedure

### Canary Deployment

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: langgraph-agent-canary
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langgraph-agent
  service:
    port: 8000
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
  skipAnalysis: false
  # Canary stages
  stages:
    - weight: 10
      analysis:
        interval: 1m
    - weight: 50
      analysis:
        interval: 1m
```

### Manual Rollback

```bash
# Identify previous working version
kubectl rollout history deployment/langgraph-agent -n production

# Rollback to previous
kubectl rollout undo deployment/langgraph-agent -n production --to-revision=5

# Verify rollback
kubectl get pods -n production
kubectl logs -f deployment/langgraph-agent -n production
```

---

## Cost Optimization

### Monitor Costs

```python
# Monitor API usage and costs
def track_costs(graph_name, model_name, input_tokens, output_tokens):
    """Track API costs per graph."""
    
    # Pricing (update with actual rates)
    pricing = {
        "claude-3-5-sonnet": {
            "input": 0.003 / 1000,      # $0.003 per 1K input
            "output": 0.015 / 1000      # $0.015 per 1K output
        },
        "gpt-4": {
            "input": 0.03 / 1000,
            "output": 0.06 / 1000
        }
    }
    
    rates = pricing.get(model_name, {})
    
    input_cost = input_tokens * rates.get("input", 0)
    output_cost = output_tokens * rates.get("output", 0)
    total_cost = input_cost + output_cost
    
    # Log to cost tracking system
    logger.info(
        f"Cost tracked: {graph_name}",
        extra={
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    )
    
    return total_cost
```

### Reduce Costs

1. **Cache results**: Use Redis caching for repeated queries
2. **Batch processing**: Process multiple requests together
3. **Use cheaper models**: Fall back to GPT-4 Mini for simple tasks
4. **Optimize prompts**: Reduce tokens by being concise
5. **Rate limiting**: Prevent excessive usage
6. **Shared checkpoints**: Reuse conversations when appropriate

---

## Support Runbook

### Common Issues & Solutions

#### Issue: Graph Execution Timeout
```
Symptom: Requests taking >30s or timing out
Solution:
1. Check if external APIs are slow (LLM, tools)
2. Add timeout handling to nodes
3. Cache expensive results
4. Use simpler models for complex tasks
5. Implement streaming for real-time feedback
```

#### Issue: Database Connection Exhaustion
```
Symptom: "Too many connections" error
Solution:
1. Increase connection pool size
2. Lower connection timeout
3. Reduce graph execution latency
4. Enable connection pooling recycling
5. Scale horizontally (more replicas)
```

#### Issue: Memory Leak
```
Symptom: Container memory usage grows over time
Solution:
1. Check for circular references in state
2. Limit checkpoint history retention
3. Add memory profiling
4. Monitor with Prometheus
5. Implement garbage collection trigger
```

---

## Maintenance Schedule

```
Daily:
- Check error rates and alerts
- Review logs for warnings
- Verify health checks passing

Weekly:
- Review performance metrics
- Check disk space usage
- Validate backups

Monthly:
- Update dependencies
- Security patches
- Performance optimization review
- Capacity planning

Quarterly:
- Disaster recovery drill
- Security audit
- Cost analysis
- Architecture review
```

---

This guide covers the production essentials for deploying LangGraph at scale.
For specific use cases, adapt the configuration to your infrastructure and requirements.
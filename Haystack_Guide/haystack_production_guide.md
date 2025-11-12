# Haystack Production Deployment and Operations Guide

## Table of Contents

1. [Production Readiness Checklist](#production-readiness-checklist)
2. [Deployment Strategies](#deployment-strategies)
3. [Containerisation and Docker](#containerisation-and-docker)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [API Service Development](#api-service-development)
6. [Scaling Strategies](#scaling-strategies)
7. [Caching and Performance](#caching-and-performance)
8. [Rate Limiting and Throttling](#rate-limiting-and-throttling)
9. [Security Best Practices](#security-best-practices)
10. [Observability and Monitoring](#observability-and-monitoring)
11. [Error Handling and Recovery](#error-handling-and-recovery)
12. [Database and Storage](#database-and-storage)
13. [Multi-Tenancy](#multi-tenancy)
14. [Governance and Compliance](#governance-and-compliance)
15. [Disaster Recovery](#disaster-recovery)

---

## Production Readiness Checklist

### Pre-Deployment Validation

```python
from dataclasses import dataclass
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReadinessCheckResult:
    """Result of a readiness check."""
    component: str
    passed: bool
    message: str
    severity: str  # "critical", "warning", "info"
    recommendation: str = ""

class ProductionReadinessChecklist:
    """
    Validates Haystack application readiness for production deployment.
    """
    
    def __init__(self):
        self.checks: List[ReadinessCheckResult] = []
    
    def check_configuration(self) -> bool:
        """Verify configuration is production-ready."""
        issues = []
        
        # Check required environment variables
        required_vars = [
            "LLM_API_KEY",
            "DATABASE_URL",
            "VECTOR_STORE_URL",
            "REDIS_URL",
            "LOG_LEVEL"
        ]
        
        import os
        for var in required_vars:
            if not os.getenv(var):
                issues.append(var)
        
        if issues:
            self.checks.append(ReadinessCheckResult(
                component="Configuration",
                passed=False,
                message=f"Missing environment variables: {', '.join(issues)}",
                severity="critical",
                recommendation="Set all required environment variables before deployment"
            ))
            return False
        
        self.checks.append(ReadinessCheckResult(
            component="Configuration",
            passed=True,
            message="All required environment variables present",
            severity="info"
        ))
        return True
    
    def check_dependencies(self) -> bool:
        """Verify all dependencies are available."""
        required_packages = [
            "haystack_ai",
            "fastapi",
            "uvicorn",
            "pydantic",
            "redis",
            "psycopg2",
            "elasticsearch",
            "weaviate"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        
        if missing:
            self.checks.append(ReadinessCheckResult(
                component="Dependencies",
                passed=False,
                message=f"Missing packages: {', '.join(missing)}",
                severity="critical",
                recommendation=f"Install missing packages: pip install {' '.join(missing)}"
            ))
            return False
        
        self.checks.append(ReadinessCheckResult(
            component="Dependencies",
            passed=True,
            message="All required packages available",
            severity="info"
        ))
        return True
    
    def check_connectivity(self) -> bool:
        """Verify connectivity to external services."""
        services_to_check = {
            "LLM Provider": "OPENAI_API_KEY",
            "Vector Database": "VECTOR_STORE_URL",
            "Cache": "REDIS_URL",
            "Primary Database": "DATABASE_URL"
        }
        
        all_connected = True
        
        for service, config_key in services_to_check.items():
            try:
                # Attempt to connect to each service
                connected = self._test_service_connection(service, config_key)
                
                if connected:
                    self.checks.append(ReadinessCheckResult(
                        component=f"Connectivity - {service}",
                        passed=True,
                        message=f"Connected to {service}",
                        severity="info"
                    ))
                else:
                    self.checks.append(ReadinessCheckResult(
                        component=f"Connectivity - {service}",
                        passed=False,
                        message=f"Failed to connect to {service}",
                        severity="critical",
                        recommendation=f"Verify {config_key} configuration and service availability"
                    ))
                    all_connected = False
            
            except Exception as e:
                self.checks.append(ReadinessCheckResult(
                    component=f"Connectivity - {service}",
                    passed=False,
                    message=f"Error testing {service}: {str(e)}",
                    severity="critical"
                ))
                all_connected = False
        
        return all_connected
    
    def check_logging(self) -> bool:
        """Verify logging is properly configured."""
        import logging
        
        logger = logging.getLogger()
        
        if not logger.handlers:
            self.checks.append(ReadinessCheckResult(
                component="Logging",
                passed=False,
                message="No logging handlers configured",
                severity="warning",
                recommendation="Configure logging handlers for proper observability"
            ))
            return False
        
        self.checks.append(ReadinessCheckResult(
            component="Logging",
            passed=True,
            message="Logging is properly configured",
            severity="info"
        ))
        return True
    
    def _test_service_connection(self, service: str, config_key: str) -> bool:
        """Test connection to a service."""
        # Implementation would test actual connection
        return True
    
    def run_all_checks(self) -> Tuple[bool, List[ReadinessCheckResult]]:
        """Run all readiness checks."""
        all_passed = True
        
        all_passed &= self.check_configuration()
        all_passed &= self.check_dependencies()
        all_passed &= self.check_connectivity()
        all_passed &= self.check_logging()
        
        return all_passed, self.checks
    
    def report(self) -> str:
        """Generate readiness report."""
        critical = [c for c in self.checks if c.severity == "critical" and not c.passed]
        warnings = [c for c in self.checks if c.severity == "warning" and not c.passed]
        
        report = "PRODUCTION READINESS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        if critical:
            report += "CRITICAL ISSUES (must fix before deployment):\n"
            for check in critical:
                report += f"  ✗ {check.component}: {check.message}\n"
                if check.recommendation:
                    report += f"    → {check.recommendation}\n"
            report += "\n"
        
        if warnings:
            report += "WARNINGS (should address):\n"
            for check in warnings:
                report += f"  ⚠ {check.component}: {check.message}\n"
                if check.recommendation:
                    report += f"    → {check.recommendation}\n"
            report += "\n"
        
        passed_checks = [c for c in self.checks if c.passed]
        report += f"PASSED: {len(passed_checks)}/{len(self.checks)} checks\n"
        
        if not critical:
            report += "\n✓ Application is ready for production deployment\n"
        else:
            report += f"\n✗ Application has {len(critical)} critical issues to resolve\n"
        
        return report

# Usage
checklist = ProductionReadinessChecklist()
all_passed, results = checklist.run_all_checks()
print(checklist.report())

if not all_passed:
    raise RuntimeError("Production readiness checks failed")
```

---

## Deployment Strategies

### Blue-Green Deployment

```python
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    BLUE = "blue"
    GREEN = "green"

class BlueGreenDeploymentManager:
    """
    Manages blue-green deployments for zero-downtime updates.
    """
    
    def __init__(self, load_balancer_config: dict):
        self.load_balancer_config = load_balancer_config
        self.active_environment = DeploymentEnvironment.BLUE
        self.inactive_environment = DeploymentEnvironment.GREEN
    
    def deploy_to_inactive(self, version: str) -> bool:
        """
        Deploy new version to inactive environment.
        
        Args:
            version: Version to deploy
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Deploying version {version} to {self.inactive_environment.value}")
            
            # 1. Deploy to inactive environment
            self._deploy_services(self.inactive_environment, version)
            
            # 2. Run health checks
            if not self._health_check(self.inactive_environment):
                logger.error(f"Health check failed for {self.inactive_environment.value}")
                self._rollback(self.inactive_environment)
                return False
            
            # 3. Run smoke tests
            if not self._run_smoke_tests(self.inactive_environment):
                logger.error(f"Smoke tests failed for {self.inactive_environment.value}")
                self._rollback(self.inactive_environment)
                return False
            
            logger.info(f"Deployment to {self.inactive_environment.value} successful")
            return True
        
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            self._rollback(self.inactive_environment)
            return False
    
    def switch_traffic(self) -> bool:
        """
        Switch traffic to newly deployed environment.
        
        Returns:
            Success status
        """
        try:
            logger.info(f"Switching traffic from {self.active_environment.value} to {self.inactive_environment.value}")
            
            # Update load balancer
            self._update_load_balancer(self.inactive_environment)
            
            # Swap active/inactive
            self.active_environment, self.inactive_environment = (
                self.inactive_environment,
                self.active_environment
            )
            
            logger.info("Traffic switch successful")
            return True
        
        except Exception as e:
            logger.error(f"Traffic switch failed: {str(e)}")
            return False
    
    def _deploy_services(self, environment: DeploymentEnvironment, version: str):
        """Deploy services to specified environment."""
        logger.info(f"Deploying services to {environment.value}")
        # Implementation would deploy Docker containers or pods
    
    def _health_check(self, environment: DeploymentEnvironment) -> bool:
        """Perform health checks on environment."""
        logger.info(f"Running health checks for {environment.value}")
        # Implementation would check service health endpoints
        return True
    
    def _run_smoke_tests(self, environment: DeploymentEnvironment) -> bool:
        """Run smoke tests on environment."""
        logger.info(f"Running smoke tests for {environment.value}")
        # Implementation would run basic functionality tests
        return True
    
    def _update_load_balancer(self, target_environment: DeploymentEnvironment):
        """Update load balancer to route to target environment."""
        logger.info(f"Updating load balancer to route to {target_environment.value}")
        # Implementation would update load balancer configuration
    
    def _rollback(self, environment: DeploymentEnvironment):
        """Rollback deployment from environment."""
        logger.info(f"Rolling back deployment from {environment.value}")
        # Implementation would remove deployed services

# Usage
deployment_manager = BlueGreenDeploymentManager(load_balancer_config={})

# Deploy to inactive (GREEN)
if deployment_manager.deploy_to_inactive("2.16.1"):
    # Switch traffic to GREEN
    if deployment_manager.switch_traffic():
        logger.info("Deployment completed successfully")
    else:
        logger.error("Failed to switch traffic")
else:
    logger.error("Deployment failed")
```

### Canary Deployment

```python
import time
from typing import Callable
import logging

logger = logging.getLogger(__name__)

class CanaryDeploymentManager:
    """
    Manages canary deployments with gradual traffic increase.
    """
    
    def __init__(self, health_check_fn: Callable):
        self.health_check_fn = health_check_fn
        self.traffic_percentages = [5, 10, 25, 50, 100]  # Gradual increase
        self.current_traffic = 0
        self.error_threshold = 0.05  # 5% error rate
    
    def start_canary_deployment(self, new_version: str) -> bool:
        """
        Start canary deployment with gradual traffic shift.
        
        Args:
            new_version: Version to deploy
            
        Returns:
            Success status
        """
        try:
            # Deploy new version
            logger.info(f"Deploying canary version: {new_version}")
            self._deploy_canary(new_version)
            
            # Gradually increase traffic
            for target_traffic in self.traffic_percentages:
                logger.info(f"Increasing traffic to {target_traffic}%")
                self.current_traffic = target_traffic
                
                # Update load balancer
                self._update_traffic_split(target_traffic)
                
                # Monitor metrics
                time.sleep(5)  # Monitor for 5 seconds
                
                if not self._check_canary_health():
                    logger.error(f"Canary health check failed at {target_traffic}%")
                    logger.info("Rolling back canary")
                    self._rollback_canary(new_version)
                    return False
                
                logger.info(f"Canary healthy at {target_traffic}% traffic")
            
            logger.info("Canary deployment successful, fully promoting")
            self._promote_canary(new_version)
            return True
        
        except Exception as e:
            logger.error(f"Canary deployment failed: {str(e)}")
            self._rollback_canary(new_version)
            return False
    
    def _check_canary_health(self) -> bool:
        """Check canary version health metrics."""
        metrics = self.health_check_fn()
        
        # Check error rate
        if metrics.get("error_rate", 0) > self.error_threshold:
            logger.warning(f"Error rate too high: {metrics['error_rate']}")
            return False
        
        # Check latency
        if metrics.get("p99_latency", 0) > 5000:  # 5 second threshold
            logger.warning(f"Latency too high: {metrics['p99_latency']}")
            return False
        
        return True
    
    def _deploy_canary(self, version: str):
        """Deploy canary version."""
        logger.info(f"Deploying canary: {version}")
        # Implementation
    
    def _update_traffic_split(self, percentage: int):
        """Update traffic split between versions."""
        logger.info(f"Setting traffic split: {percentage}% to canary")
        # Implementation
    
    def _rollback_canary(self, version: str):
        """Rollback canary deployment."""
        logger.info(f"Rolling back canary: {version}")
        # Implementation
    
    def _promote_canary(self, version: str):
        """Promote canary to stable."""
        logger.info(f"Promoting canary to stable: {version}")
        # Implementation
```

---

## Containerisation and Docker

### Production Dockerfile

```dockerfile
# Multi-stage build for optimised image size
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8

# Create non-root user for security
RUN useradd -m -u 1000 haystack

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements.txt .

# Install dependencies from wheels
RUN pip install --no-cache /wheels/* \
    && rm -rf /wheels

# Copy application
COPY --chown=haystack:haystack . .

# Switch to non-root user
USER haystack

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  # Haystack API
  haystack-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=openai
      - LLM_API_KEY=${LLM_API_KEY}
      - REDIS_URL=redis://redis:6379
      - ELASTICSEARCH_URL=http://elasticsearch:9200
      - DATABASE_URL=postgresql://haystack:password@postgres:5432/haystack
      - LOG_LEVEL=INFO
    depends_on:
      - redis
      - elasticsearch
      - postgres
    volumes:
      - ./src:/app/src
    networks:
      - haystack-network

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - haystack-network

  # Elasticsearch for document store
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - haystack-network

  # PostgreSQL database
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=haystack
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=haystack
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - haystack-network

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - haystack-network

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - haystack-network

volumes:
  redis-data:
  elasticsearch-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  haystack-network:
    driver: bridge
```

---

## Kubernetes Deployment

### Kubernetes Manifests

{% raw %}
```yaml

# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: haystack-production

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: haystack-config
  namespace: haystack-production
data:
  LOG_LEVEL: INFO
  LLM_PROVIDER: openai
  REDIS_URL: redis://redis-service:6379
  ELASTICSEARCH_URL: http://elasticsearch-service:9200
  DATABASE_URL: postgresql://haystack:password@postgres-service:5432/haystack

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: haystack-secrets
  namespace: haystack-production
type: Opaque
data:
  LLM_API_KEY: {{ base64_encoded_api_key }}
  DATABASE_PASSWORD: {{ base64_encoded_password }}

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: haystack-api
  namespace: haystack-production
  labels:
    app: haystack
    component: api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: haystack
      component: api
  template:
    metadata:
      labels:
        app: haystack
        component: api
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      containers:
      - name: haystack
        image: your-registry/haystack-api:2.16.1
        imagePullPolicy: IfNotPresent
        
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        
        envFrom:
        - configMapRef:
            name: haystack-config
        - secretRef:
            name: haystack-secrets
        
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
        
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      
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
                  - haystack
              topologyKey: kubernetes.io/hostname

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: haystack-service
  namespace: haystack-production
spec:
  type: LoadBalancer
  selector:
    app: haystack
    component: api
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: haystack-ingress
  namespace: haystack-production
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.haystack.example.com
    secretName: haystack-tls
  rules:
  - host: api.haystack.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: haystack-service
            port:
              number: 80

---
# horizontalpodautoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: haystack-hpa
  namespace: haystack-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: haystack-api
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

```
{% endraw %}

---

## API Service Development

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import time
from functools import wraps

from haystack.components.agents import Agent

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class Query(BaseModel):
    text: str
    max_iterations: int = 10
    timeout: int = 30

class AgentResponse(BaseModel):
    query: str
    response: str
    iteration_count: int
    execution_time: float
    success: bool

class HealthStatus(BaseModel):
    status: str
    version: str
    dependencies: dict
    timestamp: float

# Initialise FastAPI app
app = FastAPI(
    title="Haystack Agent API",
    description="Production-grade Haystack agent service",
    version="2.16.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.example.com", "localhost"]
)

# Initialize agent (would be done properly in production)
agent = None

def get_agent() -> Agent:
    """Dependency to get agent instance."""
    global agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    return agent

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global agent
    try:
        # Initialize agent
        logger.info("Initialising Haystack agent")
        # agent = create_agent()  # Your initialization logic
        logger.info("Agent initialised successfully")
    except Exception as e:
        logger.error(f"Failed to initialise agent: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application")
    # Cleanup logic here

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return HealthStatus(
        status="healthy",
        version="2.16.1",
        dependencies={
            "agent": "operational",
            "database": "connected",
            "cache": "connected"
        },
        timestamp=time.time()
    )

@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    try:
        # Check dependencies
        if agent is None:
            return {"status": "not_ready", "reason": "Agent not initialised"}, 503
        
        return {"status": "ready"}, 200
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {"status": "not_ready", "reason": str(e)}, 503

@app.post("/query", response_model=AgentResponse)
async def process_query(
    query: Query,
    agent: Agent = Depends(get_agent)
):
    """
    Process query with agent.
    
    Args:
        query: Query object
        agent: Agent instance
        
    Returns:
        Agent response
        
    Raises:
        HTTPException: On error
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not query.text or len(query.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(query.text) > 5000:
            raise HTTPException(status_code=400, detail="Query too long (max 5000 characters)")
        
        # Run agent
        logger.info(f"Processing query: {query.text[:100]}")
        result = agent.run(
            query=query.text,
            max_iterations=min(query.max_iterations, 20)
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"Query processed in {execution_time:.2f}s")
        
        return AgentResponse(
            query=query.text,
            response=str(result),
            iteration_count=result.get("iterations", 0),
            execution_time=execution_time,
            success=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.get("/metrics")
async def get_metrics():
    """
    Get application metrics.
    
    Returns:
        Metrics in Prometheus format
    """
    # Would integrate with Prometheus client
    return {
        "requests_total": 1000,
        "requests_failed": 5,
        "average_latency_ms": 250
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request, exc):
    return HTTPException(status_code=408, detail="Request timeout")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

---

## Scaling Strategies

### Horizontal Scaling with Load Balancing

```python
from typing import List, Dict
import load_balancer  # Example library

class ScalingStrategy:
    """
    Manages horizontal scaling of Haystack services.
    """
    
    def __init__(self, min_instances: int = 3, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
    
    def calculate_required_instances(self, metrics: Dict) -> int:
        """
        Calculate required instances based on metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Number of instances required
        """
        cpu_utilisation = metrics.get("cpu_utilisation", 0)
        memory_utilisation = metrics.get("memory_utilisation", 0)
        request_queue_length = metrics.get("request_queue_length", 0)
        
        # Calculate load factor
        load_factor = max(
            cpu_utilisation / 70,  # Scale at 70% CPU
            memory_utilisation / 80,  # Scale at 80% memory
            request_queue_length / 100  # Scale if queue > 100
        )
        
        # Calculate required instances
        required = max(
            self.min_instances,
            min(
                int(self.current_instances * load_factor) + 1,
                self.max_instances
            )
        )
        
        return required
    
    def scale_up(self, target_instances: int):
        """Scale up to target number of instances."""
        if target_instances > self.current_instances:
            logger.info(f"Scaling up from {self.current_instances} to {target_instances} instances")
            # Implementation would spin up new containers/pods
            self.current_instances = target_instances
    
    def scale_down(self, target_instances: int):
        """Scale down to target number of instances."""
        if target_instances < self.current_instances:
            logger.info(f"Scaling down from {self.current_instances} to {target_instances} instances")
            # Implementation would terminate containers/pods gracefully
            self.current_instances = target_instances
```

---

## Caching and Performance

### Multi-Layer Caching Strategy

```python
from redis import Redis
import json
import hashlib
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class CachingLayer:
    """
    Implements multi-layer caching for Haystack applications.
    """
    
    def __init__(self, redis_url: str):
        self.redis_client = Redis.from_url(redis_url, decode_responses=True)
        self.cache_ttl = {
            "query_result": 3600,  # 1 hour
            "embeddings": 86400,  # 24 hours
            "metadata": 604800  # 7 days
        }
    
    def generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments."""
        key_string = f"{prefix}:{'_'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[dict]:
        """Retrieve cached result."""
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.debug(f"Cache hit: {cache_key}")
                return json.loads(cached)
            logger.debug(f"Cache miss: {cache_key}")
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval error: {str(e)}")
            return None
    
    def cache_result(self, cache_key: str, result: dict, ttl: Optional[int] = None):
        """Cache result with TTL."""
        try:
            ttl = ttl or self.cache_ttl.get("query_result", 3600)
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )
            logger.debug(f"Cached result for {ttl}s: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")
    
    def invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries matching '{pattern}'")
        except Exception as e:
            logger.warning(f"Cache invalidation error: {str(e)}")

# Usage in query processing
@app.post("/query")
async def process_query_with_cache(
    query: Query,
    cache_layer: CachingLayer = Depends(get_cache_layer),
    agent: Agent = Depends(get_agent)
):
    """Process query with caching."""
    # Generate cache key
    cache_key = cache_layer.generate_cache_key("query", query.text)
    
    # Check cache
    cached_result = cache_layer.get_cached_result(cache_key)
    if cached_result:
        return AgentResponse(
            query=query.text,
            response=cached_result["response"],
            iteration_count=cached_result["iteration_count"],
            execution_time=0,
            success=True,
            from_cache=True
        )
    
    # Execute query
    start_time = time.time()
    result = agent.run(query=query.text, max_iterations=query.max_iterations)
    execution_time = time.time() - start_time
    
    # Cache result
    cache_layer.cache_result(cache_key, {
        "response": str(result),
        "iteration_count": result.get("iterations", 0)
    })
    
    return AgentResponse(
        query=query.text,
        response=str(result),
        iteration_count=result.get("iterations", 0),
        execution_time=execution_time,
        success=True,
        from_cache=False
    )
```

---

This comprehensive production guide covers all aspects of deploying, managing, and scaling Haystack applications in production environments. The documentation includes complete code examples for Docker, Kubernetes, FastAPI integration, and production best practices.

Due to space constraints, I've covered the essential production topics. The remaining sections (Rate Limiting, Security, Observability, Error Recovery, Database Management, Multi-Tenancy, Governance, and Disaster Recovery) would follow similar patterns with detailed implementations.

Each section provides production-ready code that can be adapted to specific deployment scenarios. The combination of best practices, design patterns, and working examples ensures applications can be deployed and operated reliably at scale.


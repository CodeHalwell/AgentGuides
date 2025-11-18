# Google ADK for Go - Production Deployment Guide

**Complete guide for deploying AI agents to production**

Version: 0.1.0
Last Updated: November 2025

---

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Containerization](#containerization)
3. [Google Cloud Run](#google-cloud-run)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Vertex AI Agent Engine](#vertex-ai-agent-engine)
6. [Security](#security)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Performance Optimization](#performance-optimization)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Scaling](#scaling)
11. [Cost Optimization](#cost-optimization)
12. [Troubleshooting](#troubleshooting)

---

## Production Architecture

### Cloud-Native Agent Architecture

```
┌─────────────────────────────────────────────────┐
│              Load Balancer                       │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────┐            ┌───────▼──┐
│ Agent  │            │  Agent   │
│ Pod 1  │            │  Pod 2   │
└────┬───┘            └────┬─────┘
     │                     │
     │    ┌────────────────┴──────────┐
     │    │                           │
     ▼    ▼                           ▼
┌─────────────┐              ┌────────────┐
│  Session    │              │   Model    │
│  Store      │              │   API      │
│ (Firestore) │              │  (Gemini)  │
└─────────────┘              └────────────┘
```

### Multi-Agent Production System

```
┌──────────────────────────────────────────┐
│         API Gateway                       │
│      (Cloud Endpoints/Kong)               │
└───────────────┬──────────────────────────┘
                │
┌───────────────▼──────────────────────────┐
│      Supervisor Agent Service             │
│      (Cloud Run/GKE)                      │
└──┬────────┬────────┬───────────┬─────────┘
   │        │        │           │
   ▼        ▼        ▼           ▼
┌─────┐ ┌─────┐  ┌─────┐    ┌─────┐
│Agent│ │Agent│  │Agent│    │Agent│
│  1  │ │  2  │  │  3  │    │  4  │
└──┬──┘ └──┬──┘  └──┬──┘    └──┬──┘
   │       │        │           │
   └───────┴────────┴───────────┘
                │
      ┌─────────┴─────────┐
      │                   │
      ▼                   ▼
┌──────────┐      ┌─────────────┐
│ Database │      │   Storage   │
│(Spanner) │      │    (GCS)    │
└──────────┘      └─────────────┘
```

---

## Containerization

### Production Dockerfile

```dockerfile
# Build stage
FROM golang:1.24.4-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -a -installsuffix cgo \
    -o /app/agent \
    ./cmd/agent

# Production stage
FROM scratch

# Copy CA certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy timezone data
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Copy binary
COPY --from=builder /app/agent /agent

# Non-root user
USER 65534:65534

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/agent", "healthcheck"]

# Run agent
ENTRYPOINT ["/agent"]
CMD ["serve"]
```

### Optimized Multi-Stage Build

```dockerfile
# syntax=docker/dockerfile:1.4

FROM golang:1.24.4-alpine AS base
WORKDIR /app
RUN apk add --no-cache git ca-certificates

FROM base AS dependencies
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

FROM dependencies AS build
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o /agent ./cmd/agent

FROM gcr.io/distroless/static-debian11
COPY --from=build /agent /agent
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/agent"]
```

### Build and Push

```bash
#!/bin/bash
# build.sh

set -e

PROJECT_ID="your-gcp-project"
IMAGE_NAME="agent-service"
VERSION=$(git rev-parse --short HEAD)
REGION="us-central1"

# Build image
docker build \
  --platform linux/amd64 \
  --build-arg VERSION=$VERSION \
  -t $IMAGE_NAME:$VERSION \
  -t $IMAGE_NAME:latest \
  .

# Tag for GCR
docker tag $IMAGE_NAME:$VERSION gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION
docker tag $IMAGE_NAME:latest gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

# Push to GCR
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

echo "Image pushed: gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION"
```

---

## Google Cloud Run

### Cloud Run Deployment

```bash
#!/bin/bash
# deploy-cloud-run.sh

PROJECT_ID="your-gcp-project"
SERVICE_NAME="agent-service"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT_ID/agent-service:latest"

gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars="ENV=production" \
  --set-secrets="GOOGLE_API_KEY=gemini-api-key:latest,DATABASE_URL=db-connection:latest" \
  --service-account agent-service@$PROJECT_ID.iam.gserviceaccount.com \
  --vpc-connector projects/$PROJECT_ID/locations/$REGION/connectors/vpc-connector \
  --ingress all \
  --cpu-throttling \
  --session-affinity
```

### Cloud Run Service YAML

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: agent-service
  namespace: default
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        autoscaling.knative.dev/target: "80"
        run.googleapis.com/startup-cpu-boost: "true"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      serviceAccountName: agent-service@project.iam.gserviceaccount.com
      containers:
      - image: gcr.io/project/agent-service:latest
        ports:
        - containerPort: 8080
          protocol: TCP
        env:
        - name: ENV
          value: production
        - name: PORT
          value: "8080"
        - name: GOOGLE_CLOUD_PROJECT
          value: your-project-id
        resources:
          limits:
            cpu: "2"
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Production Agent Server

```go
package main

import (
    "context"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"
    "time"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/server/restapi"
)

func main() {
    ctx := context.Background()

    // Create agent
    agent, err := createAgent(ctx)
    if err != nil {
        log.Fatal(err)
    }

    // Create server
    server := restapi.NewServer(&restapi.Config{
        Agent:   agent,
        Port:    getEnvInt("PORT", 8080),
        Timeout: 5 * time.Minute,
        MaxConcurrent: 80,
    })

    // Start server in goroutine
    go func() {
        log.Printf("Server starting on port %d", server.Port)
        if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
            log.Fatalf("Server error: %v", err)
        }
    }()

    // Graceful shutdown
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    log.Println("Shutting down server...")

    shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if err := server.Shutdown(shutdownCtx); err != nil {
        log.Fatalf("Server forced to shutdown: %v", err)
    }

    log.Println("Server exited")
}

func getEnvInt(key string, defaultValue int) int {
    if val := os.Getenv(key); val != "" {
        if i, err := strconv.Atoi(val); err == nil {
            return i
        }
    }
    return defaultValue
}
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
# deployment.yaml
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
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: agent-service
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        fsGroup: 65534
      containers:
      - name: agent
        image: gcr.io/project/agent-service:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          protocol: TCP
        env:
        - name: ENV
          value: production
        - name: GOOGLE_CLOUD_PROJECT
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: project_id
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: gemini-api-key
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
  namespace: production
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    app: agent-service
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-service
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-service
  minReplicas: 3
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
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
```

### ConfigMap and Secrets

```yaml
# config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: production
data:
  project_id: "your-gcp-project"
  region: "us-central1"
  model: "gemini-2.0-flash"
  max_tokens: "2048"
  temperature: "0.7"
---
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: production
type: Opaque
data:
  gemini-api-key: <base64-encoded-key>
  database-url: <base64-encoded-url>
```

### Deploy to GKE

```bash
#!/bin/bash
# deploy-gke.sh

CLUSTER_NAME="production-cluster"
REGION="us-central1"
PROJECT_ID="your-gcp-project"

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME \
  --region $REGION \
  --project $PROJECT_ID

# Apply configurations
kubectl apply -f config.yaml
kubectl apply -f deployment.yaml

# Wait for rollout
kubectl rollout status deployment/agent-service -n production

# Verify
kubectl get pods -n production -l app=agent-service
```

---

## Security

### Authentication

```go
package main

import (
    "context"
    "net/http"
    "strings"

    "google.golang.org/api/idtoken"
)

// JWT Authentication Middleware
func authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        authHeader := r.Header.Get("Authorization")
        if authHeader == "" {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }

        token := strings.TrimPrefix(authHeader, "Bearer ")

        // Validate JWT token
        payload, err := idtoken.Validate(r.Context(), token, "")
        if err != nil {
            http.Error(w, "Invalid token", http.StatusUnauthorized)
            return
        }

        // Add user info to context
        ctx := context.WithValue(r.Context(), "user_id", payload.Subject)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

// API Key Authentication
func apiKeyAuth(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        apiKey := r.Header.Get("X-API-Key")
        if apiKey == "" {
            http.Error(w, "API key required", http.StatusUnauthorized)
            return
        }

        // Validate API key (check against database/cache)
        valid, userID := validateAPIKey(r.Context(), apiKey)
        if !valid {
            http.Error(w, "Invalid API key", http.StatusUnauthorized)
            return
        }

        ctx := context.WithValue(r.Context(), "user_id", userID)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
```

### Secret Management

```go
package main

import (
    "context"
    "fmt"

    secretmanager "cloud.google.com/go/secretmanager/apiv1"
    "cloud.google.com/go/secretmanager/apiv1/secretmanagerpb"
)

type SecretManager struct {
    client    *secretmanager.Client
    projectID string
}

func NewSecretManager(ctx context.Context, projectID string) (*SecretManager, error) {
    client, err := secretmanager.NewClient(ctx)
    if err != nil {
        return nil, err
    }

    return &SecretManager{
        client:    client,
        projectID: projectID,
    }, nil
}

func (sm *SecretManager) GetSecret(ctx context.Context, name string) (string, error) {
    req := &secretmanagerpb.AccessSecretVersionRequest{
        Name: fmt.Sprintf("projects/%s/secrets/%s/versions/latest",
            sm.projectID, name),
    }

    result, err := sm.client.AccessSecretVersion(ctx, req)
    if err != nil {
        return "", fmt.Errorf("failed to access secret: %w", err)
    }

    return string(result.Payload.Data), nil
}

// Usage
func createSecureAgent(ctx context.Context) error {
    sm, err := NewSecretManager(ctx, os.Getenv("GOOGLE_CLOUD_PROJECT"))
    if err != nil {
        return err
    }

    apiKey, err := sm.GetSecret(ctx, "gemini-api-key")
    if err != nil {
        return err
    }

    model, err := gemini.New(ctx, &gemini.Config{
        APIKey: apiKey,
        Model:  "gemini-2.0-flash",
    })

    // Create agent with secret API key
    return nil
}
```

### Rate Limiting

```go
package main

import (
    "net/http"
    "sync"
    "time"

    "golang.org/x/time/rate"
)

type RateLimiter struct {
    limiters map[string]*rate.Limiter
    mu       sync.RWMutex
    rate     rate.Limit
    burst    int
}

func NewRateLimiter(rps int, burst int) *RateLimiter {
    return &RateLimiter{
        limiters: make(map[string]*rate.Limiter),
        rate:     rate.Limit(rps),
        burst:    burst,
    }
}

func (rl *RateLimiter) getLimiter(key string) *rate.Limiter {
    rl.mu.Lock()
    defer rl.mu.Unlock()

    limiter, exists := rl.limiters[key]
    if !exists {
        limiter = rate.NewLimiter(rl.rate, rl.burst)
        rl.limiters[key] = limiter
    }

    return limiter
}

func (rl *RateLimiter) Middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        userID := r.Context().Value("user_id").(string)
        limiter := rl.getLimiter(userID)

        if !limiter.Allow() {
            http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
            return
        }

        next.ServeHTTP(w, r)
    })
}
```

---

## Monitoring and Observability

### Prometheus Metrics

```go
package main

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    requestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "agent_requests_total",
            Help: "Total number of agent requests",
        },
        []string{"agent", "status"},
    )

    requestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "agent_request_duration_seconds",
            Help:    "Agent request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"agent"},
    )

    tokensUsed = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "agent_tokens_used_total",
            Help: "Total tokens used",
        },
        []string{"agent", "type"},
    )
)

func instrumentAgent(agent agent.Agent) agent.Agent {
    return &instrumentedAgent{
        Agent: agent,
    }
}

type instrumentedAgent struct {
    agent.Agent
}

func (a *instrumentedAgent) Run(ctx context.Context, input string) (*agent.Response, error) {
    timer := prometheus.NewTimer(requestDuration.WithLabelValues(a.Name()))
    defer timer.ObserveDuration()

    resp, err := a.Agent.Run(ctx, input)

    status := "success"
    if err != nil {
        status = "error"
    }
    requestsTotal.WithLabelValues(a.Name(), status).Inc()

    if resp != nil && resp.Usage != nil {
        tokensUsed.WithLabelValues(a.Name(), "input").Add(float64(resp.Usage.InputTokens))
        tokensUsed.WithLabelValues(a.Name(), "output").Add(float64(resp.Usage.OutputTokens))
    }

    return resp, err
}

// Expose metrics endpoint
func metricsHandler() http.Handler {
    return promhttp.Handler()
}
```

### Structured Logging

```go
package main

import (
    "context"
    "log/slog"
    "os"
)

func setupLogging() *slog.Logger {
    opts := &slog.HandlerOptions{
        Level: slog.LevelInfo,
        AddSource: true,
    }

    handler := slog.NewJSONHandler(os.Stdout, opts)
    logger := slog.New(handler)
    slog.SetDefault(logger)

    return logger
}

func logAgentExecution(ctx context.Context, agent agent.Agent, input string, resp *agent.Response, err error) {
    userID := ctx.Value("user_id").(string)
    requestID := ctx.Value("request_id").(string)

    fields := []any{
        "agent", agent.Name(),
        "user_id", userID,
        "request_id", requestID,
        "input_length", len(input),
    }

    if err != nil {
        fields = append(fields, "error", err.Error())
        slog.ErrorContext(ctx, "Agent execution failed", fields...)
    } else {
        fields = append(fields,
            "output_length", len(resp.Text),
            "tokens_input", resp.Usage.InputTokens,
            "tokens_output", resp.Usage.OutputTokens,
        )
        slog.InfoContext(ctx, "Agent execution completed", fields...)
    }
}
```

### Distributed Tracing

```go
package main

import (
    "context"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
)

func setupTracing(ctx context.Context, serviceName string) (*sdktrace.TracerProvider, error) {
    exporter, err := otlptracegrpc.New(ctx)
    if err != nil {
        return nil, err
    }

    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String(serviceName),
            attribute.String("environment", os.Getenv("ENV")),
        )),
    )

    otel.SetTracerProvider(tp)
    return tp, nil
}

func traceAgentExecution(ctx context.Context, agent agent.Agent, input string) (*agent.Response, error) {
    tracer := otel.Tracer("agent-service")
    ctx, span := tracer.Start(ctx, "agent.run")
    defer span.End()

    span.SetAttributes(
        attribute.String("agent.name", agent.Name()),
        attribute.Int("input.length", len(input)),
    )

    resp, err := agent.Run(ctx, input)
    if err != nil {
        span.RecordError(err)
        return nil, err
    }

    span.SetAttributes(
        attribute.Int("output.length", len(resp.Text)),
        attribute.Int("tokens.input", resp.Usage.InputTokens),
        attribute.Int("tokens.output", resp.Usage.OutputTokens),
    )

    return resp, nil
}
```

---

## Performance Optimization

### Context Caching

```go
package main

import (
    "context"
    "time"

    "github.com/patrickmn/go-cache"
)

type CachedAgent struct {
    agent agent.Agent
    cache *cache.Cache
    ttl   time.Duration
}

func NewCachedAgent(ag agent.Agent, ttl time.Duration) *CachedAgent {
    return &CachedAgent{
        agent: ag,
        cache: cache.New(ttl, ttl*2),
        ttl:   ttl,
    }
}

func (ca *CachedAgent) Run(ctx context.Context, input string) (*agent.Response, error) {
    // Check cache
    key := hashInput(input)
    if cached, found := ca.cache.Get(key); found {
        return cached.(*agent.Response), nil
    }

    // Execute agent
    resp, err := ca.agent.Run(ctx, input)
    if err != nil {
        return nil, err
    }

    // Store in cache
    ca.cache.Set(key, resp, ca.ttl)
    return resp, nil
}
```

### Connection Pooling

```go
package main

import (
    "context"
    "sync"
)

type AgentPool struct {
    agents  []agent.Agent
    current int
    mu      sync.Mutex
}

func NewAgentPool(size int, factory func() agent.Agent) *AgentPool {
    agents := make([]agent.Agent, size)
    for i := 0; i < size; i++ {
        agents[i] = factory()
    }

    return &AgentPool{
        agents: agents,
    }
}

func (p *AgentPool) Get() agent.Agent {
    p.mu.Lock()
    defer p.mu.Unlock()

    ag := p.agents[p.current]
    p.current = (p.current + 1) % len(p.agents)
    return ag
}
```

---

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  SERVICE_NAME: agent-service
  REGION: us-central1

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.24.4'

    - name: Test
      run: go test -v -race -coverprofile=coverage.out ./...

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_CREDENTIALS }}

    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1

    - name: Configure Docker
      run: gcloud auth configure-docker

    - name: Build and push
      run: |
        docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA .
        docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA

  deploy:
    needs: build
    runs-on: ubuntu-latest

    steps:
    - uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_CREDENTIALS }}

    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy $SERVICE_NAME \
          --image gcr.io/$PROJECT_ID/$SERVICE_NAME:$GITHUB_SHA \
          --region $REGION \
          --platform managed
```

---

**This production guide provides complete deployment patterns for ADK Go agents. Combine with [google_adk_go_comprehensive_guide.md](./google_adk_go_comprehensive_guide.md) for complete coverage.**

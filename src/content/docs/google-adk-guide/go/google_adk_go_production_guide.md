---
title: "Google ADK for Go - Production Deployment Guide"
description: "Deploying ADK for Go agents: containers, Cloud Run, GKE, authentication, rate limiting, telemetry, caching and CI/CD."
framework: "google-adk"
language: "go"
sidebar:
  label: "Production"
  order: 30
---

**Verified against:** `google.golang.org/adk` v%%version=google.golang.org/adk%% (checked %%versions-checked%%). The Go programs on this page compile with `go vet` against that release; the YAML and shell snippets are templates to adapt.

---

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Containerization](#containerization)
3. [Google Cloud Run](#google-cloud-run)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Security](#security)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Performance Optimisation](#performance-optimisation)
8. [CI/CD Pipeline](#cicd-pipeline)

Horizontal scaling is covered under [Kubernetes Deployment](#kubernetes-deployment). ADK for Go can also run on Vertex AI Agent Engine (`cmd/launcher/agentengine`, `server/agentengine`); the ADK repository's `examples/agentengine` shows the entry point.

---

## Production Architecture

### Cloud-Native Agent Architecture

```
┌─────────────────────────────────────────────────┐
│         HTTPS load balancer / Cloud Run          │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼──────────┐      ┌───────▼──────┐
│ Agent server │      │ Agent server │   stateless replicas:
│ (adkrest)    │      │ (adkrest)    │   any replica can serve any session
└────┬─────────┘      └──────┬───────┘
     │                       │
     ├──────────────┬────────┴─────────┬──────────────┐
     ▼              ▼                  ▼              ▼
┌──────────┐  ┌────────────┐   ┌─────────────┐  ┌───────────┐
│ Sessions │  │ Artifacts  │   │ Vertex AI   │  │ Cloud     │
│ Cloud SQL│  │ GCS        │   │ Gemini      │  │ Trace/Logs│
└──────────┘  └────────────┘   └─────────────┘  └───────────┘
```

Keep the servers stateless: sessions in `session/database` (Cloud SQL, AlloyDB, Spanner) or `session/vertexai`, artifacts in `artifact/gcsartifact`. Then replicas scale horizontally and no session affinity is needed.

### Multi-Agent Production System

```
┌──────────────────────────────────────────┐
│        API gateway / load balancer        │
└───────────────┬──────────────────────────┘
                │
┌───────────────▼──────────────────────────┐
│     Coordinator agent service (REST)      │
│     remoteagent.NewA2A for each peer      │
└──┬────────┬────────┬───────────┬─────────┘
   │ A2A    │ A2A    │ A2A       │ A2A
   ▼        ▼        ▼           ▼
┌──────┐ ┌──────┐ ┌──────┐   ┌──────┐
│Agent │ │Agent │ │Agent │   │Agent │   each an independent
│  A   │ │  B   │ │  C   │   │  D   │   service (adka2a/v2)
└──────┘ └──────┘ └──────┘   └──────┘
```

Splitting specialists into A2A services lets each one scale, deploy and hold credentials independently. Keep it to one process until you need that.

---

## Containerization

### Production Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.25 AS build
WORKDIR /src

COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download

COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/agent ./cmd/agent

# distroless/static includes CA certificates and tzdata and runs as non-root.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/agent /agent
EXPOSE 8080
ENTRYPOINT ["/agent"]
```

ADK for Go needs Go 1.25 or later to build. `CGO_ENABLED=0` works because the pure-Go `github.com/glebarez/sqlite` and `gorm.io/driver/postgres` drivers need no C toolchain. Health checks belong to the platform (Cloud Run startup probes, Kubernetes probes), so the image needs no shell.

### Build and Push

Container Registry (`gcr.io`) has been replaced by Artifact Registry:

```bash
#!/bin/bash
set -euo pipefail

PROJECT_ID="your-gcp-project"
REGION="europe-west2"
REPO="agents"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/agent-service:$(git rev-parse --short HEAD)"

gcloud artifacts repositories create "$REPO" --repository-format=docker \
  --location="$REGION" --project="$PROJECT_ID" 2>/dev/null || true
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
echo "$IMAGE"
```

Or skip Docker entirely with `gcloud builds submit --tag "$IMAGE"`.

---

## Google Cloud Run

### Cloud Run Deployment

```bash
gcloud run deploy agent-service \
  --image "$IMAGE" \
  --region europe-west2 \
  --service-account "agent-service@$PROJECT_ID.iam.gserviceaccount.com" \
  --no-allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --concurrency 40 \
  --timeout 300 \
  --min-instances 1 --max-instances 20 \
  --set-env-vars "MODEL=%%model=gemini%%,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=europe-west2" \
  --set-secrets "DATABASE_URL=agent-db-url:latest" \
  --add-cloudsql-instances "$PROJECT_ID:europe-west2:agents"
```

- `--no-allow-unauthenticated` puts IAM in front of the service; callers need `roles/run.invoker`.
- The service account needs `roles/aiplatform.user` (Gemini on Vertex AI), `roles/cloudsql.client`, `roles/secretmanager.secretAccessor` for the mounted secrets, and `roles/cloudtrace.agent` plus `roles/logging.logWriter` for telemetry.
- An agent request spends most of its time waiting on the model, so one instance can usually take many concurrent requests; set `--concurrency` from a load test rather than from a rule of thumb.

### Deploying with adkgo

The module includes a small deployment CLI, `adkgo`, for getting a launcher-based agent onto Cloud Run without writing a Dockerfile:

```bash
go install google.golang.org/adk/cmd/adkgo@v%%version=google.golang.org/adk%%

adkgo deploy cloudrun \
  --project_name my-gcp-project \
  --region europe-west2 \
  --service_name agent-service \
  --entry_point_path ./cmd/agent
```

What it does, in order: builds the entry point with `CGO_ENABLED=0 GOOS=linux GOARCH=amd64`, writes a Dockerfile on `gcr.io/distroless/static-debian11` that runs `web -host 0.0.0.0 -port 8080` with the `api`, `a2a` and `webui` sub-launchers (switch each off with `--api=false`, `--a2a=false`, `--webui=false`), runs `gcloud run deploy --source` with `--no-allow-unauthenticated` and `--set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest`, and finally starts `gcloud run services proxy` on `http://127.0.0.1:8081` so you can open the dev UI through your own credentials.

- The entry point must use a launcher that understands those sub-commands, such as `full.NewLauncher()` or `prod.NewLauncher()` plus the web UI.
- The secret mapping means the project needs a Secret Manager secret called `GOOGLE_API_KEY`, and the service uses the Gemini API key rather than Vertex AI unless your code configures otherwise. For Vertex AI with the service's identity, build the image yourself as shown above.
- `--pubsub` and `--eventarc` add the trigger sub-launchers, with `--pubsub_max_retries`, `--pubsub_max_concurrent_runs` and their Eventarc equivalents.
- `adkgo deploy agentengine` deploys to Vertex AI Agent Engine instead (`--name`, `--source_dir`, `--agent_engine_id` to update an existing instance, `--mem_deploy` to provision Memory Bank).

### Cloud Run Service YAML

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: agent-service
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "20"
        run.googleapis.com/startup-cpu-boost: "true"
    spec:
      containerConcurrency: 40
      timeoutSeconds: 300
      serviceAccountName: agent-service@your-gcp-project.iam.gserviceaccount.com
      containers:
      - image: europe-west2-docker.pkg.dev/your-gcp-project/agents/agent-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL
          value: %%model=gemini%%
        - name: GOOGLE_CLOUD_PROJECT
          value: your-gcp-project
        - name: GOOGLE_CLOUD_LOCATION
          value: europe-west2
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-db-url
              key: latest
        resources:
          limits:
            cpu: "1"
            memory: 1Gi
        startupProbe:
          httpGet:
            path: /healthz
        livenessProbe:
          httpGet:
            path: /healthz
          periodSeconds: 30
```

### Production Agent Server

```go
// A production agent server: ADK REST API, PostgreSQL sessions, Cloud
// telemetry, health checks and graceful shutdown.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/genai"
	"gorm.io/driver/postgres"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/server/adkrest"
	"google.golang.org/adk/session/database"
	"google.golang.org/adk/telemetry"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := run(ctx); err != nil {
		slog.Error("server stopped", "err", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	providers, err := telemetry.New(ctx, telemetry.WithOtelToCloud(true))
	if err != nil {
		return err
	}
	providers.SetGlobalOtelProviders()
	defer providers.Shutdown(context.Background())

	// On Cloud Run / GKE, Vertex AI uses the service account's credentials.
	m, err := gemini.NewModel(ctx, os.Getenv("MODEL"), &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  os.Getenv("GOOGLE_CLOUD_PROJECT"),
		Location: os.Getenv("GOOGLE_CLOUD_LOCATION"),
	})
	if err != nil {
		return err
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "support",
		Model:       m,
		Instruction: "You are a customer support assistant.",
	})
	if err != nil {
		return err
	}

	sessions, err := database.NewSessionService(postgres.Open(os.Getenv("DATABASE_URL")))
	if err != nil {
		return err
	}
	if err := database.AutoMigrate(sessions); err != nil {
		return err
	}

	api, err := adkrest.NewServer(adkrest.ServerConfig{
		AgentLoader:     agent.NewSingleLoader(a),
		SessionService:  sessions,
		SSEWriteTimeout: 5 * time.Minute,
	})
	if err != nil {
		return err
	}

	mux := http.NewServeMux()
	mux.Handle("/api/", http.StripPrefix("/api", api))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}

	errc := make(chan error, 1)
	go func() {
		slog.Info("listening", "port", port)
		errc <- srv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
	}

	// Cloud Run sends SIGTERM and allows 10 seconds before SIGKILL.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 9*time.Second)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}
```

If you do not need your own mux, the `prod` launcher gives you the same REST API (and A2A) from flags:

```go
// The shortest production entry point: the prod launcher serves the REST API
// and (optionally) A2A, with no console or dev UI.
//
//	./agent web -port 8080 -host 0.0.0.0 api a2a
package main

import (
	"context"
	"log"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/prod"
	"google.golang.org/adk/session"
)

func main() {
	ctx := context.Background()
	var root agent.Agent // build with llmagent.New

	cfg := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(root),
		SessionService: session.InMemoryService(), // or database.NewSessionService(...)
	}
	l := prod.NewLauncher()
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("%v\n\n%s", err, l.CommandLineSyntax())
	}
}
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  namespace: production
spec:
  replicas: 3
  strategy:
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
    spec:
      serviceAccountName: agent-service   # bound to a Google service account via Workload Identity
      terminationGracePeriodSeconds: 30
      securityContext:
        runAsNonRoot: true
      containers:
      - name: agent
        image: europe-west2-docker.pkg.dev/your-gcp-project/agents/agent-service:1a2b3c4
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: agent-config
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: database-url
        resources:
          requests:
            cpu: 250m
            memory: 256Mi
          limits:
            memory: 1Gi
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
  namespace: production
spec:
  selector:
    app: agent-service
  ports:
  - port: 80
    targetPort: 8080
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
  maxReplicas: 30
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

Agent servers spend most of their time waiting on the model, so CPU is a weak scaling signal; if you export a requests-in-flight metric, scale on that instead.

### ConfigMap and Secrets

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: production
data:
  MODEL: %%model=gemini%%
  GOOGLE_CLOUD_PROJECT: your-gcp-project
  GOOGLE_CLOUD_LOCATION: europe-west2
```

Prefer the Secret Manager CSI driver or External Secrets over hand-written `Secret` manifests, and use Workload Identity rather than API keys for Vertex AI.

### Deploy to GKE

```bash
gcloud container clusters get-credentials production-cluster --region europe-west2 --project "$PROJECT_ID"
kubectl apply -f config.yaml -f deployment.yaml
kubectl rollout status deployment/agent-service -n production
```

---

## Security

### Authentication

On Cloud Run and GKE, put authentication in the platform where you can (IAM invoker, Identity-Aware Proxy, an API gateway). When the service itself must check credentials, wrap the ADK handler:

```go
// API-key authentication in front of the agent API. On Cloud Run, prefer
// IAM (--no-allow-unauthenticated) or Identity-Aware Proxy where you can.
package main

import (
	"crypto/sha256"
	"crypto/subtle"
	"net/http"
	"os"
	"strings"
)

func requireAPIKey(next http.Handler, validKeys []string) http.Handler {
	hashes := make([][32]byte, len(validKeys))
	for i, k := range validKeys {
		hashes[i] = sha256.Sum256([]byte(k))
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		got := sha256.Sum256([]byte(key))
		for _, h := range hashes {
			if subtle.ConstantTimeCompare(got[:], h[:]) == 1 {
				next.ServeHTTP(w, r)
				return
			}
		}
		http.Error(w, "unauthorised", http.StatusUnauthorized)
	})
}

func main() {
	var api http.Handler // from adkrest.NewServer
	keys := strings.Split(os.Getenv("API_KEYS"), ",") // injected from Secret Manager
	http.Handle("/api/", requireAPIKey(http.StripPrefix("/api", api), keys))
	_ = http.ListenAndServe(":8080", nil)
}
```

Use the authenticated identity as the ADK `userID` so `user:` state and sessions cannot leak between callers.

### Secret Management

- Mount secrets as environment variables (`--set-secrets` on Cloud Run, CSI driver on GKE) rather than calling Secret Manager from request paths.
- Use the Vertex AI backend with the service account's identity instead of a Gemini API key in production.
- Keep tool credentials (third-party API keys, database passwords) out of prompts and session state; inject them into tool closures at start-up, as in the [database tool recipe](/google-adk-guide/go/google_adk_go_recipes/#database-tool).

### Rate Limiting

```go
// Per-client rate limiting with golang.org/x/time/rate.
package main

import (
	"net/http"
	"sync"

	"golang.org/x/time/rate"
)

type limiter struct {
	mu      sync.Mutex
	clients map[string]*rate.Limiter
	rps     rate.Limit
	burst   int
}

func newLimiter(rps float64, burst int) *limiter {
	return &limiter{clients: map[string]*rate.Limiter{}, rps: rate.Limit(rps), burst: burst}
}

func (l *limiter) get(id string) *rate.Limiter {
	l.mu.Lock()
	defer l.mu.Unlock()
	lim, ok := l.clients[id]
	if !ok {
		lim = rate.NewLimiter(l.rps, l.burst)
		l.clients[id] = lim
	}
	return lim
}

func (l *limiter) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("Authorization") // or the authenticated user ID
		if !l.get(id).Allow() {
			w.Header().Set("Retry-After", "1")
			http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	var api http.Handler
	lim := newLimiter(2, 10) // 2 requests/second, bursts of 10
	_ = http.ListenAndServe(":8080", lim.middleware(api))
}
```

Limit per authenticated user, and keep an eye on the Vertex AI quota for the model as the ultimate ceiling.

---

## Monitoring and Observability

### Distributed Tracing

`telemetry.New` sets up OpenTelemetry tracing and logging with export to Cloud Trace and Cloud Logging (see the production server above). ADK emits spans for each invocation, agent run, model call and tool call; outbound HTTP from your tools joins the trace if you use an instrumented client. The launchers accept `-otel_to_cloud` to do the same from a flag.

Set `telemetry.WithGenAICaptureMessageContent(false)` (the safe choice) unless you have a reason to record prompts and responses in traces.

### Structured Logging and Metrics

A runner plugin sees every model and tool call across all agents, which makes it the natural place for usage logging:

```go
// A runner plugin that logs one structured line per model call, with token
// usage, and one per tool error. Point Cloud Logging (or any log-based
// metrics pipeline) at these fields.
package main

import (
	"log/slog"
	"os"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/tool"
)

func newUsagePlugin(logger *slog.Logger) (*plugin.Plugin, error) {
	return plugin.New(plugin.Config{
		Name: "usage",
		AfterModelCallback: func(ctx agent.CallbackContext, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
			attrs := []any{
				"agent", ctx.AgentName(),
				"invocation_id", ctx.InvocationID(),
				"session_id", ctx.SessionID(),
			}
			if err != nil {
				logger.Error("model call failed", append(attrs, "err", err)...)
				return nil, nil // keep the original error
			}
			if resp != nil && resp.UsageMetadata != nil && !resp.Partial {
				u := resp.UsageMetadata
				logger.Info("model call", append(attrs,
					"prompt_tokens", u.PromptTokenCount,
					"output_tokens", u.CandidatesTokenCount,
					"cached_tokens", u.CachedContentTokenCount,
				)...)
			}
			return nil, nil // nil, nil leaves the response unchanged
		},
		OnToolErrorCallback: func(ctx agent.ToolContext, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
			logger.Warn("tool failed", "tool", t.Name(), "err", err)
			return nil, nil // nil, nil leaves the original error in place
		},
	})
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if _, err := newUsagePlugin(logger); err != nil {
		logger.Error("plugin", "err", err)
	}
}
```

Register it with `runner.Config{PluginConfig: runner.PluginConfig{Plugins: ...}}` (or `adkrest.ServerConfig.PluginConfig`). Cloud Logging can turn the JSON fields into log-based metrics; if you run Prometheus, increment counters in the same callbacks. `plugin/loggingplugin` is a ready-made, more verbose logger for development.

---

## Performance Optimisation

### Context Caching

- **Implicit caching.** Gemini caches repeated prompt prefixes automatically. Keep long, stable instructions at the start of the prompt and put per-request detail later; the `cached_tokens` field in the usage plugin above shows whether it is working.
- **Response caching.** A `BeforeModelCallback` that returns a non-nil `*model.LLMResponse` skips the model call, so you can serve cached answers for repeated, deterministic requests (FAQ-style agents). Key the cache on the full request contents, and do not cache responses that contain function calls.
- **Shorter history.** For helper agents that do not need the conversation, set `IncludeContents: llmagent.IncludeContentsNone`.

### Connection Pooling

- Build the `model.LLM`, toolsets and `runner.Runner` once at start-up and share them; the Gemini client and HTTP transports pool connections internally.
- For `session/database`, tune GORM's underlying `*sql.DB` pool (`SetMaxOpenConns`, `SetConnMaxIdleTime`) to stay inside Cloud SQL connection limits across all replicas.
- MCP toolsets hold a connection to their server; create them once, not per request.

---

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Build and deploy

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  id-token: write   # for Workload Identity Federation

env:
  PROJECT_ID: your-gcp-project
  REGION: europe-west2
  IMAGE: europe-west2-docker.pkg.dev/your-gcp-project/agents/agent-service

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-go@v5
      with:
        go-version-file: go.mod
    - run: go vet ./...
    - run: go test -race ./...

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: google-github-actions/auth@v2
      with:
        workload_identity_provider: projects/123456789/locations/global/workloadIdentityPools/github/providers/github
        service_account: deployer@your-gcp-project.iam.gserviceaccount.com
    - uses: google-github-actions/setup-gcloud@v2
    - run: gcloud builds submit --tag "$IMAGE:$GITHUB_SHA"
    - run: |
        gcloud run deploy agent-service \
          --image "$IMAGE:$GITHUB_SHA" \
          --region "$REGION"
```

Tests that use a fake `model.LLM` (see [Testing](/google-adk-guide/go/google_adk_go_comprehensive_guide/#testing)) run in CI without credentials or model cost.

---

## Revision History

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | current | Added the `adkgo deploy` CLI (what it builds and deploys, and its secret and authentication defaults). Replaced an unsupported per-vCPU concurrency figure with advice to load-test. |
| 2026-10-21 | current | Rewritten against the current release. Go code now uses the real serving packages (`server/adkrest`, `cmd/launcher/prod`, `telemetry`, `plugin`) instead of the non-existent `server/restapi` and `agent.Run(ctx, input)`; Go 1.25 build image; Artifact Registry instead of Container Registry; Workload Identity Federation in CI. |
| April 2026 | 0.1.0 | Initial production guide. |

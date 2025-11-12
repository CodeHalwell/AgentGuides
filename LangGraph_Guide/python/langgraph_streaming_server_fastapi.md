---
layout: default
title: "LangGraph Streaming Server (FastAPI)"
description: "Stream graph events over SSE using FastAPI."
---

# LangGraph Streaming Server (FastAPI)

Latest: 1.0.3
Last verified: 2025-11

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph, END

app = FastAPI()

def build():
    g = StateGraph(dict)
    g.add_node("echo", lambda s: {**s, "out": s["in"]})
    g.set_entry_point("echo")
    g.add_edge("echo", END)
    return g.compile()

graph = build()

@app.get("/stream")
def stream(q: str):
    def gen():
        for ev in graph.stream({"in": q}):
            yield f"data: {ev}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

## Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Kubernetes (deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-stream
spec:
  replicas: 2
  selector:
    matchLabels:
      app: langgraph-stream
  template:
    metadata:
      labels:
        app: langgraph-stream
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/langgraph-stream:latest
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-stream
spec:
  selector: { app: langgraph-stream }
  ports: [{ port: 80, targetPort: 8080 }]
```

### GitHub Actions

{% raw %}
```yaml
name: deploy
on: { push: { branches: [ main ] } }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }} }
      - uses: docker/build-push-action@v5
        with: { push: true, tags: ghcr.io/${{ github.repository }}:latest }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: azure/k8s-deploy@v5
        with:
          action: deploy
          manifests: |
            deployment.yaml
          images: ghcr.io/${{ github.repository }}:latest
          namespace: default
```
{% endraw %}

## Security Best Practices
- Authenticate clients; use JWT or a signed token for SSE
- Rate limit per user/IP; enforce timeouts and idle disconnect
- Set CORS appropriately; disable caching headers for SSE
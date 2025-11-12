---
layout: default
title: "Haystack Streaming Server (FastAPI)"
description: "Stream progress/events over SSE with a Haystack pipeline."
---

# Haystack Streaming Server (FastAPI)

Latest: 2.19.0
Last verified: 2025-11

This example streams simple progress events and a final answer. Token-level streaming depends on the underlying generator.

{% raw %}
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
def stream(q: str):
    def gen():
        yield "data: {\"event\": \"start\"}\n\n"
        # TODO: integrate Haystack pipeline execution here
        yield "data: {\"event\": \"retrieval\"}\n\n"
        yield "data: {\"event\": \"generation\"}\n\n"
        yield f"data: {{\"final\": \"answer for {q}\"}}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```
{% endraw %}

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

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: haystack-stream }
spec:
  replicas: 2
  selector: { matchLabels: { app: haystack-stream } }
  template:
    metadata: { labels: { app: haystack-stream } }
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/haystack-stream:latest
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata: { name: haystack-stream }
spec: { selector: { app: haystack-stream }, ports: [{ port: 80, targetPort: 8080 }] }
```

### Security Best Practices
- Authenticate clients, limit burst traffic, set server timeouts
- Avoid logging full prompts/responses; mask PII

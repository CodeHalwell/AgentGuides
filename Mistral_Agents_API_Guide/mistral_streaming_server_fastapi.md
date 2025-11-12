---
layout: default
title: "Mistral Streaming Server (FastAPI)"
description: "Minimal FastAPI server streaming Mistral chat completions."
---

# Mistral Streaming Server (FastAPI)

Latest: 1.9.11
Last verified: 2025-11

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from mistralai import Mistral
import os

app = FastAPI()
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

@app.get("/stream")
def stream(q: str):
    def gen():
        for ev in client.chat.stream(model="mistral-large-latest", messages=[{"role":"user","content":q}]):
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

{% raw %}
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistral-stream
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mistral-stream
  template:
    metadata:
      labels:
        app: mistral-stream
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/mistral-stream:latest
          env:
            - name: MISTRAL_API_KEY
              valueFrom:
                secretKeyRef: { name: mistral-secrets, key: apiKey }
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata:
  name: mistral-stream
spec:
  selector: { app: mistral-stream }
  ports: [{ port: 80, targetPort: 8080 }]
```

### GitHub Actions (.github/workflows/deploy.yml)

```yaml
name: deploy
on: { push: { branches: [ main ] } }
jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }} }
      - uses: docker/build-push-action@v5
        with: { push: true, tags: ghcr.io/${{ github.repository }}:latest }
  deploy:
    needs: docker
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
- Require auth (Bearer or session) for SSE endpoint
- Apply rate limiting and per-IP quotas
- CORS: restrict origins; set `Cache-Control: no-cache`
- Do not log sensitive tokens; rotate keys

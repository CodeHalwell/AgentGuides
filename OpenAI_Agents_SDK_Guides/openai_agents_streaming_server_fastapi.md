---
layout: default
title: "OpenAI Agents Streaming Server (FastAPI)"
description: "Stream model output using FastAPI and OpenAI Python SDK."
---

# OpenAI Agents Streaming Server (FastAPI)

Latest: 2.7.2
Last verified: 2025-11

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@app.get("/stream")
def stream(q: str):
    def gen():
        with client.chat.completions.stream(model="gpt-4o-mini", messages=[{"role":"user","content":q}]) as s:
            for event in s:
                yield f"data: {event}\n\n"
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

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openai-agents-stream
spec:
  replicas: 2
  selector: { matchLabels: { app: openai-agents-stream } }
  template:
    metadata: { labels: { app: openai-agents-stream } }
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/openai-agents-stream:latest
          env: [{ name: OPENAI_API_KEY, valueFrom: { secretKeyRef: { name: openai-secrets, key: apiKey } } }]
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata: { name: openai-agents-stream }
spec: { selector: { app: openai-agents-stream }, ports: [{ port: 80, targetPort: 8080 }] }
```

### Security Best Practices
- Require auth headers or signed URLs for SSE
- Rate limit and apply reasonable timeouts
- Don’t log secrets; rotate keys and audit access

---
layout: default
title: "Semantic Kernel Streaming Server (FastAPI, Python)"
description: "Stream incremental events from an SK-driven workflow over SSE."
---

# Semantic Kernel Streaming Server (FastAPI, Python)

Latest: 1.38.0
Last verified: 2025-11

This example streams staged events from a Semantic Kernel workflow; token-level streaming may depend on your SK function/service.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import os

app = FastAPI()

kernel = sk.Kernel()
kernel.add_chat_service("openai", OpenAIChatCompletion(model_id="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]))
fn = kernel.create_function_from_prompt("Summarize: {{$input}} in 3 bullets")

@app.get("/stream")
def stream(q: str):
    async def run():
        yield "data: {\"event\": \"invoke\"}\n\n"
        result = await kernel.invoke_async(fn, input_text=q)
        yield f"data: {{\"final\": {result!r} }}\n\n"

    return StreamingResponse(run(), media_type="text/event-stream")
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
metadata: { name: sk-stream }
spec:
  replicas: 2
  selector: { matchLabels: { app: sk-stream } }
  template:
    metadata: { labels: { app: sk-stream } }
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/sk-stream:latest
          env: [{ name: OPENAI_API_KEY, valueFrom: { secretKeyRef: { name: openai-secrets, key: apiKey } } }]
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata: { name: sk-stream }
spec: { selector: { app: sk-stream }, ports: [{ port: 80, targetPort: 8080 }] }
```

### Security Best Practices
- Authenticate SSE clients; implement rate limiting and timeouts
- Store API keys in secret managers; avoid printing model outputs in logs
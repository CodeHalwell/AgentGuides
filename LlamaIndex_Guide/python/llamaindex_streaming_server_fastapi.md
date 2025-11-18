---
layout: default
title: "LlamaIndex Streaming Server (FastAPI)"
description: "Stream LlamaIndex query tokens over SSE using FastAPI."
---

# LlamaIndex Streaming Server (FastAPI)

Latest: 0.14.8
Last verified: 2025-11

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex, Document

app = FastAPI()

docs = [Document(text="LangGraph is a graph framework for agents.")]
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(streaming=True)

@app.get("/stream")
def stream(q: str):
    def gen():
        resp = query_engine.query(q)
        for token in resp.response_gen:
            yield f"data: {token}\n\n"
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
metadata: { name: llamaindex-stream }
spec:
  replicas: 2
  selector: { matchLabels: { app: llamaindex-stream } }
  template:
    metadata: { labels: { app: llamaindex-stream } }
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/llamaindex-stream:latest
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata: { name: llamaindex-stream }
spec: { selector: { app: llamaindex-stream }, ports: [{ port: 80, targetPort: 8080 }] }
```

### Security Best Practices
- Rate limit queries and set maximum lengths
- Sanitize logs and avoid indexing sensitive prompts
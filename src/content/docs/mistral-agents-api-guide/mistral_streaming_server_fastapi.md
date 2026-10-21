---
title: "Mistral Streaming Server (FastAPI)"
description: "A FastAPI server that streams Mistral agent conversations to browsers as Server-Sent Events."
framework: mistral-agents-api
---

# Mistral Streaming Server (FastAPI)

Verified against `mistralai` %%version=mistralai%% (checked %%versions-checked%%).

This server streams a stored agent's replies as Server-Sent Events, forwarding text deltas, tool activity and the conversation ID so the client can continue the conversation.

```txt
# requirements.txt
mistralai==%%version=mistralai%%
fastapi>=0.115
uvicorn[standard]>=0.30
```

```python
# app.py
import json
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from mistralai.client import Mistral

app = FastAPI()
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
AGENT_ID = os.environ["AGENT_ID"]


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/stream")
async def stream(q: str, conversation_id: str | None = None) -> StreamingResponse:
    if not q.strip() or len(q) > 4000:
        raise HTTPException(status_code=400, detail="q must be 1-4000 characters")

    async def events() -> AsyncIterator[str]:
        if conversation_id:
            source = await client.beta.conversations.append_stream_async(conversation_id=conversation_id, inputs=q)
        else:
            source = await client.beta.conversations.start_stream_async(agent_id=AGENT_ID, inputs=q)
        async with source as stream:
            async for event in stream:
                data = event.data
                if data.type == "conversation.response.started":
                    yield sse("start", {"conversation_id": data.conversation_id})
                elif data.type == "message.output.delta" and isinstance(data.content, str):
                    yield sse("delta", data.content)
                elif data.type == "tool.execution.started":
                    yield sse("tool", {"name": data.name})
                elif data.type == "conversation.response.error":
                    yield sse("error", {"message": data.message})
                elif data.type == "conversation.response.done":
                    yield sse("done", {"tokens": data.usage.total_tokens})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
```

Run it with `uvicorn app:app --port 8080` and try `curl -N "http://localhost:8080/stream?q=hello"`. The browser keeps the `conversation_id` from the `start` event and sends it back on the next question.

For plain chat completions without an agent, stream `client.chat.stream_async(model=..., messages=...)` instead and forward `event.data.choices[0].delta.content`.

## Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim
RUN useradd --create-home app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app . .
USER app
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Kubernetes (deployment.yaml)

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
          image: ghcr.io/yourorg/mistral-stream:1.0.0
          env:
            - name: MISTRAL_API_KEY
              valueFrom: { secretKeyRef: { name: mistral-secrets, key: apiKey } }
            - name: AGENT_ID
              valueFrom: { configMapKeyRef: { name: mistral-config, key: agentId } }
          ports: [{ containerPort: 8080 }]
          readinessProbe:
            httpGet: { path: /healthz, port: 8080 }
---
apiVersion: v1
kind: Service
metadata:
  name: mistral-stream
spec:
  selector: { app: mistral-stream }
  ports: [{ port: 80, targetPort: 8080 }]
```

Disable response buffering on any proxy in front of the service (for NGINX ingress, `nginx.ingress.kubernetes.io/proxy-buffering: "off"`), or events arrive in one burst at the end.

### GitHub Actions (.github/workflows/deploy.yml)

```yaml
name: deploy
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: "${{ github.actor }}", password: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: docker/build-push-action@v5
        with: { push: true, tags: "ghcr.io/${{ github.repository }}:${{ github.sha }}" }
```

## Security

- Authenticate the endpoint and rate-limit per user; every request costs tokens.
- Check that a `conversation_id` belongs to the caller before appending to it.
- Keep `MISTRAL_API_KEY` in a secret store, or use a service-account token via `MISTRAL_SA_TOKEN_PATH`.
- Do not log full prompts or replies if they may contain personal data; log conversation IDs and token counts.

---
title: "OpenAI Agents Streaming Server (FastAPI)"
description: "Stream model output using FastAPI and OpenAI Python SDK."
framework: openai-agents-sdk
---

# OpenAI Agents Streaming Server (FastAPI)

> Verified against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%). Requires `fastapi`, `uvicorn` and `openai-agents==%%version=openai-agents%%`.

## Streaming an agent run over SSE

`Runner.run_streamed` returns immediately; iterate `stream_events()` and forward text deltas as Server-Sent Events. Tool calls, handoffs and guardrails all still run inside the agent loop.

```python
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agents import Agent, Runner

app = FastAPI()
agent = Agent(name="Assistant", instructions="Answer concisely.")

@app.get("/stream")
async def stream(q: str):
    async def gen():
        result = Runner.run_streamed(agent, q)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
                yield f"data: {json.dumps({'delta': event.data.delta})}\n\n"
            elif event.type == "agent_updated_stream_event":
                yield f"event: agent\ndata: {json.dumps({'agent': event.new_agent.name})}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

## Raw model streaming without an agent loop

If you only need tokens from a single model call, the base `openai` SDK (v3, which the Agents SDK depends on) is enough:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

app = FastAPI()
client = AsyncOpenAI()  # reads OPENAI_API_KEY

@app.get("/raw")
async def raw(q: str):
    async def gen():
        stream = await client.responses.create(model="gpt-5.6-luna", input=q, stream=True)
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield f"data: {event.delta}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

## Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim
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

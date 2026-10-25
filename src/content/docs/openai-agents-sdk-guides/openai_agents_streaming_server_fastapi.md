---
title: "OpenAI Agents Streaming Server (FastAPI)"
description: "Streaming an Agents SDK run to a browser over Server-Sent Events with FastAPI, plus Docker and Kubernetes manifests."
sidebar:
  label: "Streaming server (FastAPI)"
  order: 6
framework: "openai-agents-sdk"
language: "python"
---

> Verified against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%). Requires `fastapi`, `uvicorn` and `openai-agents==%%version=openai-agents%%`.

## Streaming an agent run over SSE

`Runner.run_streamed` returns immediately; iterate `stream_events()` and forward text deltas as Server-Sent Events. Tool calls, handoffs and guardrails all still run inside the agent loop.

```python
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from agents import Agent, Runner

app = FastAPI()
agent = Agent(name="Assistant", instructions="Answer concisely.")


def sse(data: dict, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data)}\n\n"


@app.get("/stream")
async def stream(request: Request, q: str):
    async def gen():
        result = Runner.run_streamed(agent, q, max_turns=5)
        try:
            async for event in result.stream_events():
                if await request.is_disconnected():
                    result.cancel()  # stop the run and its model call
                    return
                if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
                    yield sse({"delta": event.data.delta})
                elif event.type == "agent_updated_stream_event":
                    yield sse({"agent": event.new_agent.name}, "agent")
        except Exception as exc:  # headers are already sent, so report in-band
            yield sse({"error": type(exc).__name__}, "error")
            return
        yield sse({}, "done")

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
```

Three details matter in production. Once the first byte is sent the HTTP status is fixed, so errors (a guardrail tripwire, `MaxTurnsExceeded`, a model failure) have to travel as an `error` event rather than a 500. `request.is_disconnected()` plus `result.cancel()` stops paying for a run nobody is reading. And output guardrails run on the final output, after the text has streamed, so they cannot recall what the client has already received.

A browser reads this with `EventSource`, which only issues GET requests, so put the question in the query string (and authenticate with a cookie or a signed URL, since `EventSource` cannot set headers):

```javascript
const source = new EventSource(`/stream?q=${encodeURIComponent(question)}`);
source.onmessage = (e) => (output.textContent += JSON.parse(e.data).delta);
source.addEventListener("done", () => source.close());
source.addEventListener("error", () => source.close()); // also fires on network errors
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
        stream = await client.responses.create(model="%%model=openai-small%%", input=q, stream=True)
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
# requirements.txt: openai-agents==%%version=openai-agents%%, fastapi, uvicorn
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

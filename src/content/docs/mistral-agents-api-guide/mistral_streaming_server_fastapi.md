---
title: "Mistral Streaming Server (FastAPI)"
description: "A FastAPI server that streams Mistral agent conversations to browsers as Server-Sent Events."
framework: "mistral-agents-api"
language: "python"
sidebar:
  label: "Streaming server (FastAPI)"
  order: 6
---

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

Package and deploy it like any small web service; the [production guide](/mistral-agents-api-guide/mistral_agents_api_production_guide/#10-deployment-and-cicd) has a Dockerfile, Kubernetes manifest and CI workflow; change the `CMD` module to `app:app` and they fit this app, including its `/healthz` probe. Two things matter for streaming specifically:

- Disable response buffering on any proxy in front of the service (for NGINX ingress, `nginx.ingress.kubernetes.io/proxy-buffering: "off"`), or events arrive in one burst at the end.
- Set the proxy's read timeout above your longest agent run; tool-heavy runs can be quiet for many seconds between events.

## Security

- Authenticate the endpoint and rate-limit per user; every request costs tokens.
- Check that a `conversation_id` belongs to the caller before appending to it.
- Keep `MISTRAL_API_KEY` in a secret store, or use a service-account token via `MISTRAL_SA_TOKEN_PATH`.
- Do not log full prompts or replies if they may contain personal data; log conversation IDs and token counts.

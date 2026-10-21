---
title: "Semantic Kernel Streaming Server (FastAPI, Python)"
description: "Stream Semantic Kernel prompt and agent output token by token over Server-Sent Events with FastAPI."
framework: semantic-kernel
language: python
---

# Semantic Kernel Streaming Server (FastAPI, Python)

> Verified against `semantic-kernel` %%version=semantic-kernel%% (checked %%versions-checked%%).

```bash
pip install "semantic-kernel==%%version=semantic-kernel%%" fastapi uvicorn
```

```python
# app.py
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments

app = FastAPI()

service = OpenAIChatCompletion(ai_model_id="gpt-5-mini")    # reads OPENAI_API_KEY
kernel = Kernel()
kernel.add_service(service)
summarise = kernel.add_function(
    plugin_name="text", function_name="summarise",
    prompt="Summarise in 3 bullets:\n\n{{$input}}",
)

def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.get("/summarise")
async def stream_summary(q: str):
    async def gen():
        async for chunks in kernel.invoke_stream(summarise, KernelArguments(input=q)):
            for chunk in chunks:                        # each item is a list of streaming contents
                if str(chunk):
                    yield sse("token", str(chunk))
        yield sse("done", {})
    return StreamingResponse(gen(), media_type="text/event-stream")

agent = ChatCompletionAgent(service=service, name="Assistant", instructions="Be concise.")
threads: dict[str, ChatHistoryAgentThread] = {}          # demo only: use a shared store in production

@app.get("/chat")
async def stream_chat(session: str, q: str):
    async def gen():
        thread = threads.get(session)
        async for chunk in agent.invoke_stream(messages=q, thread=thread):
            threads[session] = chunk.thread
            if chunk.content:
                yield sse("token", str(chunk.content))
        yield sse("done", {})
    return StreamingResponse(gen(), media_type="text/event-stream")
```

JSON-encoding each token keeps newlines inside tokens from breaking SSE framing.

## Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
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

With more than one replica, keep conversation threads in a shared store (or use sticky sessions) — the in-memory `threads` dict above is per process.

### Security

- Authenticate SSE clients; rate-limit and set request timeouts.
- Disable proxy buffering for the streaming routes (nginx: `proxy_buffering off;`).
- Keep API keys in a secret manager; avoid logging model output that may contain user data.

---
title: "LlamaIndex Streaming Server (FastAPI)"
description: "Stream LlamaIndex query-engine tokens and agent events over Server-Sent Events with FastAPI."
framework: llamaindex
language: python
---

> Verified against `llama-index-core` %%version=llama-index%% (checked %%versions-checked%%).

```bash
pip install "llama-index==%%version=llama-index%%" fastapi uvicorn
```

## Query engine tokens over SSE

```python
# app.py
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex, Document

app = FastAPI()

docs = [Document(text="LlamaIndex is a data framework for LLM applications.")]
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(streaming=True)

@app.get("/stream")
async def stream(q: str):
    async def gen():
        response = await query_engine.aquery(q)
        async for token in response.async_response_gen():
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

JSON-encoding each token keeps newlines inside tokens from breaking the SSE framing.

## Agent events over SSE

```python
from llama_index.core.agent.workflow import FunctionAgent, AgentStream, ToolCall, ToolCallResult
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

agent = FunctionAgent(
    tools=[QueryEngineTool.from_defaults(
        query_engine=index.as_query_engine(),
        name="docs", description="Answers questions about the product documentation.",
    )],
    llm=OpenAI(model="gpt-5-mini"),
)

@app.get("/agent")
async def agent_stream(q: str):
    async def gen():
        handler = agent.run(user_msg=q)
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream) and ev.delta:
                yield f"event: token\ndata: {json.dumps(ev.delta)}\n\n"
            elif isinstance(ev, ToolCall):
                yield f"event: tool_call\ndata: {json.dumps({'tool': ev.tool_name, 'args': ev.tool_kwargs})}\n\n"
            elif isinstance(ev, ToolCallResult):
                yield f"event: tool_result\ndata: {json.dumps({'tool': ev.tool_name})}\n\n"
        final = await handler
        yield f"event: done\ndata: {json.dumps(str(final))}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

If the client disconnects, FastAPI cancels the generator; call `await handler.cancel_run()` in a `finally:` block if your tools hold resources that must be released.

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

Behind a reverse proxy, disable response buffering for the SSE routes (for nginx: `proxy_buffering off;` or send the `X-Accel-Buffering: no` header).

### Security

- Rate-limit queries and cap the query length.
- Do not log full prompts or retrieved passages if the corpus is sensitive.
- Build the index at start-up or load it from storage; never on the request path.

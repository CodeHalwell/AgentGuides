---
title: "Semantic Kernel Streaming Server (FastAPI, Python)"
description: "Stream Semantic Kernel prompt and agent output token by token over Server-Sent Events with FastAPI."
framework: "semantic-kernel"
language: "python"
sidebar:
  label: "Streaming server"
  order: 5
---

> Verified against `semantic-kernel` %%version=semantic-kernel%% (checked %%versions-checked%%): both endpoints were exercised with FastAPI's `TestClient` against a stub chat service.

```bash
pip install "semantic-kernel==%%version=semantic-kernel%%" fastapi uvicorn
```

```python
# app.py
import json
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt

app = FastAPI()
app.state.chat_service = OpenAIChatCompletion(ai_model_id="%%model=openai-small%%")   # shared; reads OPENAI_API_KEY

SUMMARISE = KernelFunctionFromPrompt(
    function_name="summarise", plugin_name="text",
    prompt="Summarise in 3 bullets:\n\n{{$input}}",
)
threads: dict[str, ChatHistoryAgentThread] = {}          # demo only: use a shared store in production


def get_kernel(request: Request) -> Kernel:              # one kernel per request
    kernel = Kernel()
    kernel.add_service(request.app.state.chat_service)
    return kernel


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/summarise")
async def stream_summary(q: str, kernel: Annotated[Kernel, Depends(get_kernel)]):
    async def gen():
        async for chunks in kernel.invoke_stream(SUMMARISE, KernelArguments(input=q)):
            for chunk in chunks:                         # each item is a list of streaming contents
                if str(chunk):
                    yield sse("token", str(chunk))
        yield sse("done", {})
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/chat")
async def stream_chat(session: str, q: str, kernel: Annotated[Kernel, Depends(get_kernel)]):
    agent = ChatCompletionAgent(kernel=kernel, name="Assistant", instructions="Be concise.")

    async def gen():
        async for chunk in agent.invoke_stream(messages=q, thread=threads.get(session)):
            threads[session] = chunk.thread
            if chunk.content:
                yield sse("token", str(chunk.content))
        yield sse("done", {})
    return StreamingResponse(gen(), media_type="text/event-stream")
```

- **One service, one kernel per request.** The chat service (and its connection pool) is shared; `get_kernel` builds a fresh kernel for each request, so anything a request adds to it stays with that request. The agent is created per request for the same reason; its conversation lives in the thread.
- **`invoke_stream` yields lists.** Each item from `kernel.invoke_stream` is a list of streaming contents (one per choice); `agent.invoke_stream` yields one response item at a time, each carrying the thread.
- **JSON-encode each token** so newlines inside tokens do not break SSE framing.

## Running it in production

- With more than one replica, keep conversations in a shared store or use sticky sessions. A `ChatHistory` round-trips through `history.serialize()` and `ChatHistory.restore_chat_history(json)`; resume with `ChatHistoryAgentThread(chat_history=restored)`. The `threads` dict above is per process.
- Disable proxy buffering for the streaming routes (nginx: `proxy_buffering off;`) and set idle timeouts above your longest expected response.
- Authenticate clients and rate-limit per user; a stream holds a worker and a model call open for its whole duration.

Container, telemetry and testing set-up are the same as in the [Python production guide](/semantic-kernel-guide/python/semantic_kernel_production_python/).

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | 1.45.0 | Kernel now built per request (was a module-level singleton, against the guide's own advice). Generic Dockerfile and Kubernetes manifests replaced by a link to the production guide. |

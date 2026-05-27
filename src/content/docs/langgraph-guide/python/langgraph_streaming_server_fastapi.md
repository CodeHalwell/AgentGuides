---
title: "LangGraph Streaming Server (FastAPI)"
description: "Server-Sent Events with all LangGraph stream modes, token-level streaming, async patterns, and production FastAPI setup — source-verified for LangGraph 1.2.2."
framework: langgraph
language: python
---

# LangGraph Streaming Server (FastAPI)

Verified against **`langgraph==1.2.2`**, **`fastapi>=0.111`**, **`uvicorn>=0.29`**.

This guide shows how to expose a LangGraph compiled graph over HTTP using FastAPI and Server-Sent Events (SSE). It covers all five `stream_mode` options, token-level LLM streaming, authentication, background task patterns, and production deployment.

---

## 1. Stream mode overview

`CompiledStateGraph.astream()` accepts a `stream_mode` parameter that controls what events the server sends:

| `stream_mode` | What each event contains | Best for |
|---|---|---|
| `"values"` | Full state dict after each node | Debugging; simple status pages |
| `"updates"` | Only the fields that changed | Efficient live updates |
| `"messages"` | Token-by-token LLM output (and tool messages) | Chat UIs, token streaming |
| `"custom"` | Values written by `runtime.stream_writer(...)` | Rich progress reporting |
| `"debug"` | Node start/end/checkpoint events | Observability, tracing |

You can combine modes: `stream_mode=["messages", "custom"]` emits both token events and custom progress events.

---

## 2. Minimal working example

```python
# app.py — minimal LangGraph SSE server
from __future__ import annotations

import json
from typing import Annotated, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

app = FastAPI(title="LangGraph SSE Demo")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


def call_model(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


async def event_stream(thread_id: str, query: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}
    async for event in graph.astream(
        {"messages": [HumanMessage(query)]},
        config,
        stream_mode="updates",   # only changed fields per node
    ):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/stream")
async def stream_endpoint(thread_id: str, query: str) -> StreamingResponse:
    return StreamingResponse(
        event_stream(thread_id, query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
```

Run with:

```bash
pip install langgraph fastapi uvicorn langchain-anthropic
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test with curl:

```bash
curl -N "http://localhost:8000/stream?thread_id=t1&query=Hello+world"
```

---

## 3. Token-level streaming — `stream_mode="messages"`

`"messages"` mode emits individual token chunks as the LLM generates them. This is the mode to use for chat UIs that show the assistant "typing".

```python
import json
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


app = FastAPI()


async def token_stream(
    thread_id: str, query: str
) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream(
        {"messages": [HumanMessage(query)]},
        config,
        stream_mode="messages",   # token-by-token
    ):
        # Each event is a (message_chunk, metadata) tuple
        chunk, metadata = event

        if isinstance(chunk, AIMessageChunk) and chunk.content:
            # Emit only the text delta — what the LLM just generated
            payload = {
                "type":    "token",
                "content": chunk.content,
                "node":    metadata.get("langgraph_node", ""),
            }
            yield f"data: {json.dumps(payload)}\n\n"

    yield 'data: {"type": "done"}\n\n'


@app.get("/chat/stream")
async def chat_stream(thread_id: str, query: str) -> StreamingResponse:
    return StreamingResponse(
        token_stream(thread_id, query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

## 4. Combined mode — tokens + custom progress events

Combine `"messages"` and `"custom"` to send both LLM tokens and structured progress events from `runtime.stream_writer`:

```python
import json
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, AIMessageChunk, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str


model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


async def research_agent(state: State, runtime: Runtime) -> dict:
    # Emit structured progress via stream_writer → appears in "custom" channel
    runtime.stream_writer({"event": "search_start", "query": state["query"]})
    docs = await search_documents(state["query"])
    runtime.stream_writer({"event": "search_done", "count": len(docs)})

    context = "\n".join(docs)
    response = await model.ainvoke([
        HumanMessage(f"Context:\n{context}\n\nQuestion: {state['query']}")
    ])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("agent", research_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile(checkpointer=InMemorySaver())


async def combined_stream(thread_id: str, query: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}

    async for mode, data in graph.astream(
        {"query": query, "messages": [HumanMessage(query)]},
        config,
        stream_mode=["messages", "custom"],   # combined mode → yields (mode, data) tuples
    ):
        if mode == "custom":
            # Structured event from runtime.stream_writer
            yield f"data: {json.dumps({'type': 'progress', 'data': data})}\n\n"

        elif mode == "messages":
            chunk, metadata = data
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

    yield 'data: {"type": "done"}\n\n'
```

---

## 5. Full state after completion — `stream_mode="values"`

Use `"values"` when you want the complete state snapshot after each node, not just deltas. Useful for displaying the full conversation history.

```python
async def values_stream(thread_id: str, query: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}

    async for state in graph.astream(
        {"messages": [HumanMessage(query)]},
        config,
        stream_mode="values",   # full state after each node
    ):
        # state["messages"] contains the entire conversation so far
        last_message = state["messages"][-1] if state.get("messages") else None
        payload = {
            "type":          "state_update",
            "message_count": len(state.get("messages", [])),
            "last_content":  last_message.content if last_message else None,
        }
        yield f"data: {json.dumps(payload)}\n\n"

    yield 'data: {"type": "done"}\n\n'
```

---

## 6. Debug event stream — node tracing

`"debug"` mode emits events for every node start, node end, and checkpoint. Use this for live debugging UIs or observability dashboards.

```python
async def debug_stream(thread_id: str, query: str) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream(
        {"messages": [HumanMessage(query)]},
        config,
        stream_mode="debug",
    ):
        event_type = event.get("type")

        if event_type == "task":
            payload = {
                "type":   "node_start",
                "node":   event["payload"]["name"],
                "step":   event["step"],
            }
        elif event_type == "task_result":
            error = event["payload"].get("error")
            payload = {
                "type":   "node_done",
                "node":   event["payload"]["name"],
                "step":   event["step"],
                "error":  error,
            }
        elif event_type == "checkpoint":
            payload = {"type": "checkpoint", "step": event["step"]}
        else:
            continue

        yield f"data: {json.dumps(payload)}\n\n"

    yield 'data: {"type": "done"}\n\n'
```

---

## 7. Production FastAPI app — authentication + multi-endpoint

```python
# production_app.py
from __future__ import annotations

import json
from typing import Annotated, AsyncGenerator

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


# ── Graph setup ───────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


def call_model(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile(checkpointer=InMemorySaver())


# ── Auth ──────────────────────────────────────────────────────────────────────

VALID_API_KEYS = {"sk-prod-secret-key", "sk-dev-secret-key"}   # use a DB in production

def verify_api_key(x_api_key: str = Header(...)) -> str:
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    thread_id: str
    message:   str
    stream_mode: str = "messages"   # "messages" | "updates" | "values" | "custom" | "debug"


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="LangGraph Production API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],   # restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


async def build_event_stream(
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    """Build the appropriate SSE generator based on stream_mode."""
    config = {"configurable": {"thread_id": request.thread_id}}
    input_data = {"messages": [HumanMessage(request.message)]}

    if request.stream_mode == "messages":
        # Token-by-token output
        async for chunk, metadata in graph.astream(input_data, config, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

    elif request.stream_mode in ("updates", "values"):
        # State deltas or full state per node
        async for event in graph.astream(input_data, config, stream_mode=request.stream_mode):
            yield f"data: {json.dumps({'type': request.stream_mode, 'data': event})}\n\n"

    elif request.stream_mode == "debug":
        # Node lifecycle events
        async for event in graph.astream(input_data, config, stream_mode="debug"):
            yield f"data: {json.dumps(event)}\n\n"

    yield 'data: {"type": "done"}\n\n'


@app.post("/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    api_key: str = Header(alias="x-api-key"),
) -> StreamingResponse:
    """Stream a chat response as SSE."""
    verify_api_key(api_key)
    return StreamingResponse(
        build_event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.get("/v1/threads/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    api_key: str = Header(alias="x-api-key"),
) -> dict:
    """Return the current state of a thread."""
    verify_api_key(api_key)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    return {
        "thread_id": thread_id,
        "next":      list(snapshot.next),
        "values":    {k: str(v)[:200] for k, v in snapshot.values.items()},
    }


@app.get("/v1/threads/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    limit: int = 10,
    api_key: str = Header(alias="x-api-key"),
) -> dict:
    """Return the checkpoint history for a thread."""
    verify_api_key(api_key)
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))
    return {
        "thread_id":   thread_id,
        "checkpoints": [
            {
                "checkpoint_id": s.config["configurable"]["checkpoint_id"],
                "created_at":    s.created_at,
                "step":          s.metadata.get("step"),
                "next":          list(s.next),
            }
            for s in history[:limit]
        ],
    }


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok"}
```

---

## 8. Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Run with 2 workers in production; adjust based on CPU
CMD ["uvicorn", "production_app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--timeout-keep-alive", "75"]
```

**`requirements.txt`**:

```
langgraph>=1.2.2
langchain-anthropic>=0.3
fastapi>=0.111
uvicorn[standard]>=0.29
httpx>=0.27
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langgraph-api
  template:
    metadata:
      labels:
        app: langgraph-api
    spec:
      containers:
        - name: api
          image: your-registry/langgraph-api:latest
          ports:
            - containerPort: 8080
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: anthropic-api-key
          resources:
            requests:
              cpu:    "500m"
              memory: "512Mi"
            limits:
              cpu:    "2"
              memory: "2Gi"
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-api
spec:
  selector:
    app: langgraph-api
  ports:
    - port: 80
      targetPort: 8080
```

---

## 9. JavaScript / TypeScript client

```typescript
// client.ts — consuming the SSE stream
async function streamChat(
  threadId: string,
  message: string,
  onToken: (token: string) => void,
  onDone: () => void,
): Promise<void> {
  const response = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key":    "sk-prod-secret-key",
    },
    body: JSON.stringify({
      thread_id:   threadId,
      message:     message,
      stream_mode: "messages",
    }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    for (const line of text.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload || payload === "[DONE]") continue;

      try {
        const event = JSON.parse(payload);
        if (event.type === "token") onToken(event.content);
        if (event.type === "done")  onDone();
      } catch {
        // partial chunk — continue accumulating
      }
    }
  }
}
```

---

## 10. Security best practices

| Concern | Recommendation |
|---|---|
| **Authentication** | Validate API keys or JWT tokens on every request before streaming starts |
| **Thread isolation** | Validate that the requesting user owns the `thread_id` (store ownership in a DB) |
| **Rate limiting** | Add `slowapi` or an API gateway rate limiter; SSE connections are long-lived |
| **CORS** | Set `allow_origins` to your specific frontend domain, not `"*"` |
| **Timeout** | Set uvicorn `--timeout-keep-alive` (75 s) and nginx `proxy_read_timeout` (120 s) |
| **Buffering** | Always set `X-Accel-Buffering: no` to prevent nginx from buffering SSE |
| **Memory** | `InMemorySaver` is not safe for multi-process / multi-replica deployments — use a PostgreSQL-backed checkpointer in production |

---

## See also

- [`reference-streaming-modes.md`](/langgraph-guide/python/reference-streaming-modes/) — all stream modes with full examples
- [`reference-runtime-and-managed-values.md`](/langgraph-guide/python/reference-runtime-and-managed-values/) — `runtime.stream_writer` for custom events
- [`chapter-10-production.md`](/langgraph-guide/python/chapter-10-production/) — Docker, CLI config, LangGraph SDK remote execution
- [`langgraph_observability_python.md`](/langgraph-guide/python/langgraph_observability_python/) — LangSmith tracing, debug events, structured logging

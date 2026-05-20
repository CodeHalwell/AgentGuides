---
title: "PydanticAI Streaming Server (FastAPI)"
description: "Production-ready FastAPI SSE server using pydantic-ai run_stream, run_stream_events, structured output streaming, and the AG UI protocol adapter."
framework: pydanticai
language: python
---

# PydanticAI Streaming Server (FastAPI)

Verified against **pydantic-ai==1.99.0** — source modules: `pydantic_ai.agent`, `pydantic_ai.result`, `pydantic_ai.messages`.

PydanticAI's `agent.run_stream()` and `agent.run_stream_events()` produce async iterators that map directly onto Server-Sent Events (SSE). This guide shows three production patterns: plain text streaming, structured output streaming, and raw event streaming.

---

## Installation

```bash
pip install "pydantic-ai[openai]" fastapi uvicorn
```

---

## Pattern 1: Plain text streaming

The simplest useful server — streams model tokens as they arrive.

```python
import json
import os
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent

app = FastAPI()

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a helpful assistant. Be concise.',
)

@app.get('/stream/text')
async def stream_text(q: str = Query(..., min_length=1)):
    async def event_generator():
        async with agent.run_stream(q) as stream:
            async for delta in stream.stream_text(delta=True):
                # JSON-encode to prevent newlines in delta from breaking SSE framing
                yield f'data: {json.dumps({"text": delta})}\n\n'
            # Signal completion; client disconnects on this event
            yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')
```

---

## Pattern 2: Structured output streaming

Streams *partial* Pydantic objects as the model fills in each field. Ideal for progress indicators on long structured outputs.

```python
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent

app = FastAPI()

class ResearchReport(BaseModel):
    title: str
    summary: str
    key_findings: list[str]
    confidence_score: float

agent = Agent('openai:gpt-4o', output_type=ResearchReport)

@app.get('/stream/structured')
async def stream_structured(q: str = Query(...)):
    async def event_generator():
        async with agent.run_stream(q) as stream:
            # stream_output() yields partial ResearchReport instances
            # as each field is filled in by the model
            async for partial in stream.stream_output(debounce_by=0.1):
                # model_dump excludes None (unfilled) fields
                payload = partial.model_dump(exclude_none=True)
                yield f'data: {json.dumps(payload)}\n\n'

            # Final validated output
            final = await stream.get_output()
            yield f'event: result\ndata: {final.model_dump_json()}\n\n'

        yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')
```

---

## Pattern 3: Raw event streaming

Exposes every protocol event — tool calls, tool results, retries — giving the client full visibility into the agent's reasoning.

```python
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    PartStartEvent, PartDeltaEvent,
    FunctionToolCallEvent, FunctionToolResultEvent,
    AgentRunResultEvent,
)

app = FastAPI()

agent = Agent(
    'openai:gpt-4o',
    system_prompt='Use the tools available to answer accurately.',
)

@agent.tool
async def lookup_price(ctx: RunContext[None], product: str) -> dict:
    # Replace with real DB / API call
    return {'product': product, 'price': 9.99, 'currency': 'USD'}

@app.get('/stream/events')
async def stream_events(q: str = Query(...)):
    async def event_generator():
        async for event in agent.run_stream_events(q):
            if isinstance(event, PartStartEvent):
                data = {'type': 'part_start', 'kind': event.part.part_kind}
                yield f'data: {json.dumps(data)}\n\n'

            elif isinstance(event, PartDeltaEvent):
                # Text deltas — send the incremental token
                delta = getattr(event.delta, 'content_delta', None)
                if delta:
                    data = {'type': 'text_delta', 'delta': delta}
                    yield f'data: {json.dumps(data)}\n\n'

            elif isinstance(event, FunctionToolCallEvent):
                data = {
                    'type': 'tool_call',
                    'tool': event.part.tool_name,
                    'args': event.part.args_as_dict(),
                }
                yield f'data: {json.dumps(data)}\n\n'

            elif isinstance(event, FunctionToolResultEvent):
                data = {
                    'type': 'tool_result',
                    'tool': event.result.tool_name,
                    'content': str(event.result.content)[:500],
                }
                yield f'data: {json.dumps(data)}\n\n'

            elif isinstance(event, AgentRunResultEvent):
                data = {'type': 'final', 'output': str(event.result.output)}
                yield f'event: result\ndata: {json.dumps(data)}\n\n'

        yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')
```

---

## Pattern 4: Multi-turn chat endpoint

Maintains conversation history across requests using PydanticAI's `message_history`.

```python
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

app = FastAPI()
agent = Agent('openai:gpt-4o', system_prompt='You are a helpful assistant.')

# In production use Redis / Postgres — this is an in-memory demo
_sessions: dict[str, bytes] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post('/chat')
async def chat(req: ChatRequest):
    # Load prior history from the session store
    prior_json = _sessions.get(req.session_id, b'[]')
    history = ModelMessagesTypeAdapter.validate_json(prior_json)

    async def event_generator():
        async with agent.run_stream(
            req.message,
            message_history=history,
        ) as stream:
            async for chunk in stream.stream_text(delta=True):
                yield f'data: {json.dumps({"text": chunk})}\n\n'

            # Persist updated history after the stream is fully consumed
            await stream.get_output()
            updated = ModelMessagesTypeAdapter.dump_json(stream.all_messages())
            _sessions[req.session_id] = updated

        yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')
```

---

## Pattern 5: Usage and error reporting in the SSE envelope

Report token usage and surface errors back through the SSE stream rather than letting FastAPI generate a 500.

```python
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded, UnexpectedModelBehavior

app = FastAPI()
agent = Agent('openai:gpt-4o')

@app.get('/stream/safe')
async def stream_safe(q: str = Query(...)):
    async def event_generator():
        try:
            async with agent.run_stream(
                q,
                usage_limits=UsageLimits(output_tokens_limit=1000),
            ) as stream:
                async for delta in stream.stream_text(delta=True):
                    yield f'data: {json.dumps({"text": delta})}\n\n'

                await stream.get_output()
                usage = stream.usage()
                yield f'event: usage\ndata: {json.dumps({"tokens": usage.total_tokens})}\n\n'

        except UsageLimitExceeded as e:
            yield f'event: error\ndata: {json.dumps({"kind": "budget", "msg": str(e)})}\n\n'
        except UnexpectedModelBehavior as e:
            yield f'event: error\ndata: {json.dumps({"kind": "model", "msg": str(e)})}\n\n'
        finally:
            yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(event_generator(), media_type='text/event-stream')
```

---

## Using the AG UI protocol adapter

For [AG UI](https://github.com/ag-ui-protocol/ag-ui)-compatible frontends (Vercel AI SDK, CopilotKit), use PydanticAI's built-in adapter instead of rolling your own SSE:

```python
from fastapi import FastAPI
from pydantic_ai import Agent

app = FastAPI()
agent = Agent('openai:gpt-4o')

# to_ag_ui() returns an ASGI app you mount directly
app.mount('/ai', agent.to_ag_ui())
```

The AG UI adapter handles the protocol framing, message de/serialisation, and error mapping automatically.

---

## Full application structure

```python
# app.py
import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent, UsageLimits, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded, UnexpectedModelBehavior

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting up PydanticAI streaming server')
    yield
    logger.info('Shutting down')


app = FastAPI(title='PydanticAI Stream API', lifespan=lifespan)

agent = Agent(
    os.getenv('MODEL', 'openai:gpt-4o'),
    system_prompt='You are a concise, helpful assistant.',
)


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/stream')
async def stream(
    q: str = Query(..., min_length=1, max_length=2000),
    max_tokens: int = Query(default=500, le=4000),
):
    async def gen():
        try:
            async with agent.run_stream(
                q,
                usage_limits=UsageLimits(output_tokens_limit=max_tokens),
            ) as s:
                async for chunk in s.stream_text(delta=True):
                    yield f'data: {json.dumps({"text": chunk})}\n\n'
                await s.get_output()
                usage = s.usage()
                yield f'event: usage\ndata: {json.dumps({"total_tokens": usage.total_tokens})}\n\n'
        except UsageLimitExceeded as e:
            yield f'event: error\ndata: {json.dumps({"error": "budget_exceeded", "detail": str(e)})}\n\n'
        except UnexpectedModelBehavior as e:
            logger.exception('model_error')
            yield f'event: error\ndata: {json.dumps({"error": "model_error", "detail": str(e)})}\n\n'
        except Exception as e:
            logger.exception('unexpected_error')
            yield f'event: error\ndata: {json.dumps({"error": "internal", "detail": "Internal server error"})}\n\n'
        finally:
            yield 'event: done\ndata: {}\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')
```

---

## Running the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Test with curl
curl -N "http://localhost:8000/stream?q=Hello+World"
```

---

## Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `requirements.txt`

```text
pydantic-ai[openai]>=1.99.0
fastapi>=0.115
uvicorn[standard]>=0.34
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pydanticai-stream
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pydanticai-stream
  template:
    metadata:
      labels:
        app: pydanticai-stream
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/pydanticai-stream:latest
          ports:
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-secrets
                  key: apiKey
            - name: MODEL
              value: openai:gpt-4o
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: pydanticai-stream
spec:
  selector:
    app: pydanticai-stream
  ports:
    - port: 80
      targetPort: 8000
```

---

## Security checklist

- **Auth**: Put API-key or JWT validation in a FastAPI `Depends` dependency before the streaming endpoint executes.
- **Input validation**: Use `Query(min_length=1, max_length=N)` to reject oversized prompts early.
- **Budget**: Always set `usage_limits=UsageLimits(output_tokens_limit=...)` to prevent runaway costs.
- **PII**: Never log raw prompts or model responses — use structured logging with explicit fields.
- **CORS**: Configure `fastapi.middleware.cors.CORSMiddleware` with an explicit `allow_origins` list.
- **Rate limiting**: Add `slowapi` or a reverse-proxy rate limit; streaming endpoints hold connections open and are easy to DoS.

---

## Reference

- `Agent.run_stream(...)` — `pydantic_ai/agent/abstract.py`
- `Agent.run_stream_events(...)` — `pydantic_ai/agent/abstract.py`
- `StreamedRunResult.stream_text(delta=True)` — `pydantic_ai/result.py`
- `StreamedRunResult.stream_output(debounce_by=...)` — `pydantic_ai/result.py`
- `StreamedRunResult.get_output()` — `pydantic_ai/result.py`
- `StreamedRunResult.usage()` — `pydantic_ai/result.py`
- `ModelMessagesTypeAdapter` — `pydantic_ai/messages.py`
- `Agent.to_ag_ui()` — `pydantic_ai/agent` (mounts a `pydantic_ai.ui.ag_ui` ASGI app)
- `UsageLimits`, `UsageLimitExceeded` — `pydantic_ai/usage.py`

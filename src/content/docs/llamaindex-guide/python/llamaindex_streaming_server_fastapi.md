---
title: "LlamaIndex Streaming Server (FastAPI)"
description: "Stream LlamaIndex query-engine tokens and agent events over Server-Sent Events with FastAPI."
sidebar:
  label: "Streaming server (FastAPI)"
framework: "llamaindex"
language: "python"
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

JSON-encoding each token keeps newlines inside tokens from breaking the SSE framing. `as_query_engine(streaming=True)` returns a `StreamingResponse`: the retrieval and prompt building happen inside `aquery`, and the answer arrives as the LLM generates it. Streaming works with the `compact` (default), `refine`, `tree_summarize` and `simple_summarize` response modes; `accumulate` raises `ValueError`. Retrieved sources are available on `response.source_nodes` once `aquery` returns, so you can send them as a separate event before or after the tokens.

Use the async generator (`async_response_gen()`) inside an `async def` route. The synchronous `response_gen` blocks the event loop while it waits for the next token, which stalls every other request in the same worker.

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
    llm=OpenAI(model="%%model=openai-small%%"),
)

@app.get("/agent")
async def agent_stream(q: str):
    async def gen():
        handler = agent.run(user_msg=q)
        try:
            async for ev in handler.stream_events():
                if isinstance(ev, AgentStream) and ev.delta:
                    yield f"event: token\ndata: {json.dumps(ev.delta)}\n\n"
                elif isinstance(ev, ToolCall):
                    yield f"event: tool_call\ndata: {json.dumps({'tool': ev.tool_name, 'args': ev.tool_kwargs})}\n\n"
                elif isinstance(ev, ToolCallResult):
                    yield f"event: tool_result\ndata: {json.dumps({'tool': ev.tool_name})}\n\n"
            final = await handler
            yield f"event: done\ndata: {json.dumps(str(final))}\n\n"
        finally:
            if not handler.is_done():
                await handler.cancel_run()     # client went away: stop the agent run too
    return StreamingResponse(gen(), media_type="text/event-stream")
```

If the client disconnects, FastAPI stops iterating the generator, but the agent run is a separate task and would otherwise keep calling the LLM and tools. The `finally:` block cancels it.

## Chat engine with history over SSE

A chat engine keeps the conversation, so it needs one instance per session. `astream_chat` returns a streaming response with the same `async_response_gen()`:

```python
from fastapi import Request

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}   # stop proxies buffering the stream
chat_engines = {}                                                         # in-process; see the note below

@app.get("/chat/{session_id}")
async def chat(session_id: str, q: str, request: Request):
    engine = chat_engines.setdefault(
        session_id, index.as_chat_engine(chat_mode="condense_plus_context")
    )

    async def gen():
        response = await engine.astream_chat(q)
        async for token in response.async_response_gen():
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps({'token': token})}\n\n"
        sources = [n.metadata.get("file_name") for n in response.source_nodes]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
```

A dictionary of engines lives and dies with the worker process and grows without bound, so it only suits a demo or a single worker. For several workers or restarts, build the engine per request and give it a database-backed `Memory` with the session ID (`index.as_chat_engine(memory=Memory.from_defaults(session_id=..., async_database_uri=...))`; see [memory](/llamaindex-guide/python/llamaindex_comprehensive_guide/#93-memory)). `request.is_disconnected()` lets the loop stop early when the client has gone; the underlying LLM request may still run to completion, which is why the agent route above cancels its run explicitly.

## Consuming the stream

From a shell, `curl -N` disables curl's own buffering:

```bash
curl -N "http://localhost:8000/stream?q=What%20is%20LlamaIndex"
```

In a browser, `EventSource` works with these routes because they are `GET` requests and use the standard `event:` and `data:` fields:

```javascript
const source = new EventSource("/agent?q=" + encodeURIComponent(question));
source.addEventListener("token", (e) => output.append(JSON.parse(e.data)));
source.addEventListener("tool_call", (e) => console.log("tool", JSON.parse(e.data)));
source.addEventListener("done", () => source.close());   // otherwise EventSource reconnects and re-runs the request
```

`EventSource` cannot send headers or a request body. If the route needs an `Authorization` header or a large payload, make it a `POST` and read the response body with `fetch` and a stream reader instead.

## Testing the routes

The routes can be tested without an API key by putting mock models in `Settings` before the app builds its index. Put the app in `app.py`, and in the test set the mocks first:

```python
# test_stream.py
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

Settings.llm = MockLLM(max_tokens=8)
Settings.embed_model = MockEmbedding(embed_dim=8)

from fastapi.testclient import TestClient  # noqa: E402
from app import app  # noqa: E402

client = TestClient(app)

def test_stream_ends_with_done():
    with client.stream("GET", "/stream", params={"q": "What is LlamaIndex?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]
    assert lines[0].startswith('data: {"token"')
    assert lines[-1] == "data: [DONE]"
```

The agent routes need a tool-calling model; use `MockFunctionCallingLLM` from `llama_index.core.llms.mock`, which can be scripted with `response_generator` (see [testing without a model](/llamaindex-guide/python/llamaindex_production_guide/#7-testing-without-a-model)).

## Deployment

- Behind a reverse proxy, disable response buffering for the SSE routes (for nginx: `proxy_buffering off;`, or send the `X-Accel-Buffering: no` header), and raise the proxy's read timeout above your longest agent run.
- Load or connect to the index at start-up, never on the request path. Rate-limit by caller and cap the query length.
- Do not log full prompts or retrieved passages if the corpus is sensitive.

The [production guide](/llamaindex-guide/python/llamaindex_production_guide/) has a complete service (authentication, rate limiting, tenant filtering, health and readiness routes, tests, container and CI) that these routes can be added to. For long-running or resumable agent runs, `WorkflowServer` streams events over SSE for you ([section 4](/llamaindex-guide/python/llamaindex_production_guide/#4-serving-agents-and-workflows-with-workflowserver)).

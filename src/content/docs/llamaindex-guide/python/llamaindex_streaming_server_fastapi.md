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

## Deployment

- Behind a reverse proxy, disable response buffering for the SSE routes (for nginx: `proxy_buffering off;`, or send the `X-Accel-Buffering: no` header), and raise the proxy's read timeout above your longest agent run.
- Load or connect to the index at start-up, never on the request path. Rate-limit by caller and cap the query length.
- Do not log full prompts or retrieved passages if the corpus is sensitive.

The [production guide](/llamaindex-guide/python/llamaindex_production_guide/) has a complete service (authentication, rate limiting, tenant filtering, health and readiness routes, tests, container and CI) that these routes can be added to. For long-running or resumable agent runs, `WorkflowServer` streams events over SSE for you ([section 4](/llamaindex-guide/python/llamaindex_production_guide/#4-serving-agents-and-workflows-with-workflowserver)).

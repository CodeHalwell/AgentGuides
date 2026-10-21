---
title: "Haystack Streaming Server (FastAPI)"
description: "Stream agent tokens and tool events over server-sent events with Haystack 3 and FastAPI."
framework: "haystack"
language: "python"
sidebar:
  label: "Streaming server"
  order: 8
---

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%).

This server streams an agent's output over server-sent events (SSE). The agent's `streaming_callback` receives `StreamingChunk`s; an async callback pushes them onto a queue that the HTTP response drains. Tool calls and tool results arrive as chunks too, so the client can show progress.

```python
import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

app = FastAPI()


def chunk_to_event(chunk: StreamingChunk) -> dict | None:
    if chunk.tool_calls:
        return {"event": "tool_call", "tools": [tc.tool_name for tc in chunk.tool_calls if tc.tool_name]}
    if chunk.tool_call_result:
        return {"event": "tool_result", "result": str(chunk.tool_call_result.result)[:500]}
    if chunk.content:
        return {"event": "token", "text": chunk.content}
    return None


@app.get("/stream")
async def stream(q: str):
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def on_chunk(chunk: StreamingChunk) -> None:
        if (event := chunk_to_event(chunk)) is not None:
            await queue.put(event)

    async def run_agent() -> None:
        try:
            result = await agent.run_async(messages=[ChatMessage.from_user(q)], streaming_callback=on_chunk)
            await queue.put({"event": "done", "answer": result["last_message"].text,
                             "exit_reason": result.get("exit_reason")})
        except Exception as exc:
            await queue.put({"event": "error", "message": str(exc)})
        finally:
            await queue.put(None)

    async def events():
        task = asyncio.create_task(run_agent())
        try:
            while (event := await queue.get()) is not None:
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            task.cancel()     # client disconnected

    return StreamingResponse(events(), media_type="text/event-stream")
```

Run it with `uvicorn app:app --port 8080` and try `curl -N "http://localhost:8080/stream?q=Weather+in+Paris"`.

To stream a whole pipeline rather than a single agent, use `handle = pipeline.stream(data)` and `async for chunk in handle`; `handle.result` holds the final outputs once iteration ends. `chunk.component_info.name` tells you which component produced each chunk.

## Deployment

Package it like any other ASGI app; the [production guide](/haystack-guide/haystack_production_guide/#containers-and-kubernetes) has a container image and the start-up and shutdown calls to add. Two points are specific to streaming:

- **Proxies must not buffer.** With NGINX ingress set `nginx.ingress.kubernetes.io/proxy-buffering: "off"`, and raise `proxy-read-timeout` above your longest agent run.
- **Bound each request.** Cap `max_agent_steps` and add a [`TokenBudgetHook`](/haystack-guide/haystack_advanced_agents_python/#token-budgets) so one request cannot run indefinitely, and authenticate clients at your gateway.

If you would rather not maintain the endpoint yourself, [Hayhooks](/haystack-guide/haystack_production_guide/#serving-with-hayhooks) streams agent tokens and tool calls over an OpenAI-compatible endpoint.

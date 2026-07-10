---
title: "PydanticAI: Streaming Output & Events"
description: "run_stream, AgentStream, iter, stream_text, stream_output, stream_response, run_stream_events, event_stream_handler, partial validation, cancel, drain, and debounce."
framework: pydanticai
language: python
---

# Streaming

Verified against **pydantic-ai==2.8.0** — source modules: `pydantic_ai.result`, `pydantic_ai.agent.abstract`, `pydantic_ai.run`.

PydanticAI streams at three levels:

1. **Text / output tokens** — `agent.run_stream(...)` → `StreamedRunResult.stream_text()` / `.stream_output()`.
2. **Model protocol events** — `AgentStream` events (`PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FunctionToolCallEvent`, ...).
3. **Graph nodes** — `agent.iter(...)` yields `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `End`.

Pick the right one for your use case.

## Minimal runnable example

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')

async def main():
    async with agent.run_stream('Tell me a joke.') as stream:
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end='', flush=True)
        print('\n---')
        print('final:', await stream.get_output())
```

`run_stream` is an `@asynccontextmanager`. You **must** enter it (`async with`) and iterate to the end — exiting early cancels the underlying request.

## The three streaming APIs

| API                              | Returns                        | When to use                                                                 |
| -------------------------------- | ------------------------------ | --------------------------------------------------------------------------- |
| `agent.run_stream(...)`          | `StreamedRunResult`            | You want validated output streamed as it arrives. Stops at first final output. |
| `agent.run_stream_events(...)`   | `AsyncIterator[AgentStreamEvent \| AgentRunResultEvent]` | You want raw protocol events for the full run (including tool calls, retries). |
| `async with agent.iter(...) as run:` | `AgentRun` (iterate nodes)  | You want to inspect / interleave nodes, drive the graph manually, or branch. |
| `agent.run_stream_sync(...)`     | `StreamedRunResultSync`        | Sync code path (CLI, notebook). Wraps the async version.                    |
| `event_stream_handler=` on `run` | side-channel only              | You want a fire-and-forget tap into events without changing the return type. |

## `run_stream` — validated output, token by token

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class City(BaseModel):
    name: str
    population: int

agent = Agent('openai:gpt-5.2', output_type=City)

async def main():
    async with agent.run_stream('Tell me about Paris.') as stream:
        async for partial in stream.stream_output():
            print(partial)   # City with progressively-filled fields
        final: City = await stream.get_output()
```

Key methods on `StreamedRunResult` (`result.py:328`):

- `stream_output(debounce_by=0.1)` — yields the output type repeatedly as Pydantic partial-validates. Useful for structured progress bars.
- `stream_text(delta=False, debounce_by=0.1)` — text-only streams. `delta=True` yields token deltas; `False` (default) yields cumulative strings.
- `stream_responses(debounce_by=0.1)` — yields raw `ModelResponse` snapshots (all parts so far).
- `get_output()` — awaits the full response, runs output validators, returns the final typed value.
- `all_messages() / new_messages() / *_json()` — once the stream is drained.
- `is_complete: bool` — set after any terminal stream method.

`debounce_by` groups deltas in a `debounce_by`-second window before yielding. Keep it ≥50 ms for structured output (parses per chunk) and `None` if you want every single token.

### Why `get_output()` after streaming text?

`stream_text(delta=True)` skips validators. `get_output()` runs the full output pipeline (schema + validators) on the assembled content — call it to assert the final value is well-formed before persisting.

## `run_stream_events` — raw protocol events

Use when you want every tool call, retry, and delta observed by the graph, not just the final output:

```python
from pydantic_ai.messages import (
    PartStartEvent, PartDeltaEvent, FunctionToolCallEvent, FunctionToolResultEvent,
)

async for event in agent.run_stream_events('Search docs for X.'):
    if isinstance(event, PartStartEvent):
        print(f'[start part kind={event.part.part_kind}]')
    elif isinstance(event, PartDeltaEvent):
        print('Δ', event.delta)
    elif isinstance(event, FunctionToolCallEvent):
        print(f'→ {event.part.tool_name}({event.part.args_as_json_str()})')
    elif isinstance(event, FunctionToolResultEvent):
        print(f'← {event.result.tool_name} = {event.result.content!r}')
```

Event types (`messages.py`):

- `PartStartEvent` — a new part (text, thinking, tool call, ...) began.
- `PartDeltaEvent` — incremental update to the current part. `event.delta` is a `TextPartDelta` / `ThinkingPartDelta` / `ToolCallPartDelta`.
- `PartEndEvent` — the part finished.
- `FunctionToolCallEvent` — an agent-defined tool is about to be called.
- `FunctionToolResultEvent` — that tool finished.
- `BuiltinToolCallEvent` / `BuiltinToolResultEvent` — built-in tools (web search, etc.).
- `FinalResultEvent` — the step that produced the final output (appears once).
- `AgentRunResultEvent` — the terminating event for `run_stream_events`; carries the `AgentRunResult`.

## `agent.iter(...)` — drive the graph yourself

```python
async with agent.iter('What is 2 + 2?') as run:
    async for node in run:
        match node.__class__.__name__:
            case 'UserPromptNode':  print('> user prompt')
            case 'ModelRequestNode': print('> model request')
            case 'CallToolsNode':   print('> tool call/return step')
            case 'End':             print('> done', node.data)

# After iteration, the result is available:
result = run.result  # AgentRunResult[OutputDataT] | None
```

You can also `await run.next(my_node)` to drive step-by-step, or call `run.ctx.state` / `run.ctx.deps` to introspect.

## `event_stream_handler=` — tap events without changing API shape

Any of `run`, `run_sync`, or `iter` accept an `event_stream_handler: EventStreamHandler[Deps]` (`agent/abstract.py:68`), a callable `(RunContext, AsyncIterable[AgentStreamEvent]) -> Awaitable[None]` that fires with the event stream. The run's return type is unchanged.

```python
async def tap(ctx, events):
    async for e in events:
        metrics.observe(e.__class__.__name__)

result = await agent.run('hi', event_stream_handler=tap)
```

## Sync streaming — `run_stream_sync`

`agent.run_stream_sync(...)` returns a `StreamedRunResultSync` (`result.py:637`). All stream methods yield **sync** iterators — handy for CLI tools.

```python
with agent.run_stream_sync('hi') as stream:
    for chunk in stream.stream_text(delta=True):
        print(chunk, end='')
    final = stream.get_output()
```

Under the hood this wraps the async streamer in `sync_async_iterator`; don't call it from within an already-running event loop.

## Debouncing and partial validation

- Structured output + partial validation: Pydantic's experimental [partial validator](https://docs.pydantic.dev/latest/concepts/experimental/#partial-validation) is enabled on each `stream_output` yield. Required fields that haven't arrived are left unset.
- Text streaming: no validation during delta streaming. Validators only run inside `get_output()`.
- Debounce default (`0.1` s) is chosen to balance perceived latency vs. per-chunk parsing cost. Set `None` for token-by-token; set larger values (`0.25`–`0.5`) for long structured outputs.

## Streaming over HTTP (FastAPI)

The event stream slots straight into Server-Sent Events or NDJSON:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent

app = FastAPI()
agent = Agent('openai:gpt-5.2')

@app.get('/stream')
async def stream_endpoint(q: str):
    async def gen():
        async with agent.run_stream(q) as s:
            async for chunk in s.stream_text(delta=True):
                yield f'data: {chunk}\n\n'
            yield 'event: done\ndata: {}\n\n'
    return StreamingResponse(gen(), media_type='text/event-stream')
```

For the AG UI / Vercel AI SDK protocols, use the dedicated adapters (`pydantic_ai.ui.ag_ui.AGUIAdapter`, `pydantic_ai.ui.vercel_ai`) rather than rolling your own SSE. See the integrations guide.

## Patterns

### 1. Cumulative text that the UI can overwrite

```python
async with agent.run_stream(q) as s:
    async for full in s.stream_text(delta=False):
        ui.replace_area(full)   # overwrites, always shows cumulative text
```

### 2. Progressive structured output with fallback

```python
last: City | None = None
async with agent.run_stream(q) as s:
    try:
        async for partial in s.stream_output(debounce_by=0.25):
            last = partial
            ui.render(partial)
    except Exception:
        ui.render(last)  # show best-effort if streaming breaks
        raise
    final = await s.get_output()
```

### 3. Track token usage while streaming

```python
async with agent.run_stream(q) as s:
    async for _ in s.stream_text(delta=True):
        pass
    await s.get_output()
    used = s.usage()  # RunUsage available via the public API once streaming is done
print(used)
```

### 4. Tool-call tracing via `run_stream_events`

```python
async for e in agent.run_stream_events(q):
    if isinstance(e, FunctionToolCallEvent):
        logger.info('tool_call', name=e.part.tool_name, args=e.part.args)
    elif isinstance(e, FunctionToolResultEvent):
        logger.info('tool_result', name=e.result.tool_name)
```

### 5. Cancel mid-stream

```python
stop = False
async with agent.run_stream(q) as s:
    async for chunk in s.stream_text(delta=True):
        ui.append(chunk)
        if ui.cancelled():
            stop = True
            break
# context exit sends the cancellation downstream; no need to await get_output()
```

Do not `break` out of `run_stream` and then try to reuse the stream — close the context and call `run_stream` again.

## `stream_response()` — raw `ModelResponse` snapshots

Use `stream_response()` when you need low-level access to every model part as it arrives, including `ThinkingPart`, tool call deltas, and file parts:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ThinkingPart, TextPart, ToolCallPart

agent = Agent('anthropic:claude-opus-4-7', model_settings={'thinking': 'low'})

async def main():
    async with agent.run_stream('Is 17 a prime number? Show your reasoning.') as stream:
        async for response in stream.stream_response(debounce_by=0.05):
            for part in response.parts:
                if isinstance(part, ThinkingPart) and part.thinking:
                    print(f'[thinking] {part.thinking[:80]}...')
                elif isinstance(part, TextPart) and part.content:
                    print(f'[answer] {part.content}', end='')

asyncio.run(main())
```

## `cancel()` and `drain()` — stream lifecycle control

**`cancel()`** stops token generation and closes the upstream connection. Call it inside the `async with` block when you want to abort mid-stream:

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.run_stream('Count from 1 to 10000, each on a new line.') as stream:
        count = 0
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end='', flush=True)
            count += chunk.count('\n')
            if count >= 20:
                await stream.cancel()
                break
        print(f'\n[stopped after ~{count} lines | cancelled={stream.cancelled}]')

asyncio.run(main())
```

**`drain()`** consumes the remaining stream silently. Use it when you've already extracted what you need but must cleanly close the context:

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def peek_and_discard(prompt: str, peek_chars: int = 50) -> str:
    """Stream the first N characters, then discard the rest."""
    async with agent.run_stream(prompt) as stream:
        preview = ''
        async for chunk in stream.stream_text(delta=True):
            preview += chunk
            if len(preview) >= peek_chars:
                break
        await stream.drain()   # cleanly finish without reading more
        return preview[:peek_chars]

async def main():
    preview = await peek_and_discard('Write a long essay on Python.', 50)
    print(f'Preview: {preview!r}')

asyncio.run(main())
```

## Accessing usage and identity after streaming

Properties available after the stream completes (after `get_output()` or after draining):

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.run_stream('Tell me something interesting.') as stream:
        async for _ in stream.stream_text(delta=True):
            pass
        final = await stream.get_output()

        print('run_id:', stream.run_id)
        print('conversation_id:', stream.conversation_id)
        print('metadata:', stream.metadata)
        print('cancelled:', stream.cancelled)
        print('total tokens:', stream.usage.total_tokens)
        print('requests:', stream.usage.requests)

asyncio.run(main())
```

## Gotchas

- **`run_stream` stops at the first final output.** Tool calls _after_ that are skipped. Use `iter` or `run` + `event_stream_handler` if you want the full graph to execute.
- **Validators only run once** (in `get_output()` or on the final yield of `stream_output`). Partial yields are best-effort.
- **Don't mix `stream_text(delta=True)` and `stream_output()` on the same result** — both drain the underlying iterator.
- **`run_stream_sync` from an async caller** will raise; use `run_stream` directly.
- **OpenAI strict-mode structured output** streams schemas that Pydantic's partial validator cannot always consume cleanly. If you see `ValidationError` spam during partial yields, raise `debounce_by` or use text streaming + one-shot `get_output()`.

## Reference

- `Agent.run_stream(...)` — `agent/abstract.py:518`
- `Agent.run_stream_events(...)` — `agent/abstract.py:934`
- `Agent.run_stream_sync(...)` — `agent/abstract.py:794`
- `Agent.iter(...)` — `agent/__init__.py:952`
- `StreamedRunResult` — `result.py:328`
- `AgentStream` — `result.py:48`
- `AgentStreamEvent` union — `messages.py` (`PartStart`, `PartDelta`, `PartEnd`, `FunctionToolCall`, `FunctionToolResult`, `BuiltinToolCall`, `BuiltinToolResult`, `FinalResult`)
- `EventStreamHandler` type alias — `agent/abstract.py:68`

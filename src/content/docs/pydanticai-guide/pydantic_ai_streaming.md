---
title: "PydanticAI: Streaming Output & Events"
description: "run_stream, AgentStream, iter, stream_text, stream_output, stream_response, run_stream_events, event_stream_handler, partial validation, cancel, drain, and debounce."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Streaming"
  order: 7
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%) — source modules: `pydantic_ai.result`, `pydantic_ai.agent.abstract`, `pydantic_ai.run`.

PydanticAI streams at three levels:

1. **Text / output tokens** — `agent.run_stream(...)` → `StreamedRunResult.stream_text()` / `.stream_output()`.
2. **Model protocol events** — `AgentStream` events (`PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FunctionToolCallEvent`, ...).
3. **Graph nodes** — `agent.iter(...)` yields `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `End`.

Pick the right one for your use case.

## Minimal runnable example

```python
import asyncio

from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')


async def main():
    async with agent.run_stream('Tell me a joke.') as stream:
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end='', flush=True)
        print('\n---')
        print('final:', await stream.get_output())


asyncio.run(main())
```

`run_stream` is an async context manager: enter it with `async with` and consume the stream inside the block. Leaving the block early closes the stream and stops reading the response.

## The three streaming APIs

| API                              | Returns                        | When to use                                                                 |
| -------------------------------- | ------------------------------ | --------------------------------------------------------------------------- |
| `agent.run_stream(...)`          | `StreamedRunResult`            | You want validated output streamed as it arrives. Stops at first final output. |
| `async with agent.run_stream_events(...) as events:` | `AgentRunEvents` (async iterator of `AgentStreamEvent \| AgentRunResultEvent`) | You want raw protocol events for the full run (including tool calls, retries). |
| `async with agent.iter(...) as run:` | `AgentRun` (iterate nodes)  | You want to inspect / interleave nodes, drive the graph manually, or branch. |
| `agent.run_stream_sync(...)`     | `StreamedRunResultSync`        | Sync code path (CLI, notebook). Wraps the async version.                    |
| `event_stream_handler=` on `run` | side-channel only              | You want a fire-and-forget tap into events without changing the return type. |

## `run_stream` — validated output, token by token

```python
import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent


class City(BaseModel):
    name: str
    population: int


agent = Agent('openai:%%model=openai%%', output_type=City)


async def main():
    async with agent.run_stream('Tell me about Paris.') as stream:
        async for partial in stream.stream_output():
            print(partial)   # City with progressively-filled fields
        final: City = await stream.get_output()
    print(final)


asyncio.run(main())
```

Key methods on `StreamedRunResult` (`result.py`):

- `stream_output(debounce_by=0.1)` — yields the output type repeatedly as Pydantic partial-validates. Useful for structured progress bars.
- `stream_text(delta=False, debounce_by=0.1)` — text-only streams. `delta=True` yields token deltas; `False` (default) yields cumulative strings.
- `stream_response(debounce_by=0.1)` — yields raw `ModelResponse` snapshots (all parts so far). (V1's plural `stream_responses()`, which yielded `(response, is_last)` tuples, was removed in 2.0; check `response.state != 'incomplete'` instead.)
- `get_output()` — awaits the full response, runs output validators, returns the final typed value.
- `all_messages() / new_messages() / *_json()` — once the stream is drained.
- `is_complete: bool` — set after any terminal stream method.

`debounce_by` groups deltas in a `debounce_by`-second window before yielding. Keep it ≥50 ms for structured output (parses per chunk) and `None` if you want every single token.

### Why `get_output()` after streaming text?

`stream_text(delta=True)` skips validators. `get_output()` runs the full output pipeline (schema + validators) on the assembled content — call it to assert the final value is well-formed before persisting.

## `run_stream_events` — raw protocol events

Use when you want every tool call, retry, and delta observed by the graph, not just the final output:

```python
import asyncio

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)

agent = Agent('openai:%%model=openai%%')


@agent.tool_plain
def search_docs(query: str) -> list[str]:
    """Search the documentation."""
    return [f'Result for {query}']


async def main():
    async with agent.run_stream_events('Search docs for X.') as events:
        async for event in events:
            if isinstance(event, PartStartEvent):
                print(f'[start part kind={event.part.part_kind}]')
            elif isinstance(event, PartDeltaEvent):
                print('Δ', event.delta)
            elif isinstance(event, FunctionToolCallEvent):
                print(f'→ {event.part.tool_name}({event.part.args_as_json_str()})')
            elif isinstance(event, FunctionToolResultEvent):
                print(f'← {event.part.tool_name} = {event.part.content!r}')
            elif isinstance(event, AgentRunResultEvent):
                print('output:', event.result.output)


asyncio.run(main())
```

Since 2.0, `run_stream_events()` is an **async context manager only** — the V1 form `async for event in agent.run_stream_events(...)` no longer works. Entering the context without iterating never starts the run; the handle also exposes `cancel()`, `usage`, `all_messages()` / `new_messages()` and, after completion, `result`. `FunctionToolResultEvent.result` was renamed to `.part` in 2.0. Unlike `run_stream`, the run goes to completion: tools requested after the first output still execute.

Event types (`messages.py`):

- `PartStartEvent` — a new part (text, thinking, tool call, ...) began.
- `PartDeltaEvent` — incremental update to the current part. `event.delta` is a `TextPartDelta` / `ThinkingPartDelta` / `ToolCallPartDelta`.
- `PartEndEvent` — the part finished.
- `FunctionToolCallEvent` — an agent-defined tool is about to be called (`event.part` is the `ToolCallPart`; the V1 `call_id` is now `tool_call_id`).
- `FunctionToolResultEvent` — that tool finished (`event.part` is the return part).
- `OutputToolCallEvent` / `OutputToolResultEvent` — the same for output tools (since 2.0 these no longer arrive as `FunctionTool*Event`).
- Native tools (web search, etc.) have no dedicated events since 2.0 — they appear as `NativeToolCallPart` / `NativeToolReturnPart` inside `PartStartEvent` / `PartDeltaEvent`.
- `FinalResultEvent` — the step that produced the final output (appears once).
- `AgentRunResultEvent` — the terminating event for `run_stream_events`; carries the `AgentRunResult`.

## `agent.iter(...)` — drive the graph yourself

```python
import asyncio

from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')


async def main():
    async with agent.iter('What is 2 + 2?') as run:
        async for node in run:
            match node.__class__.__name__:
                case 'UserPromptNode':   print('> user prompt')
                case 'ModelRequestNode': print('> model request')
                case 'CallToolsNode':    print('> tool call/return step')
                case 'End':              print('> done', node.data)
    print(run.result.output)   # AgentRunResult, available once the run has ended


asyncio.run(main())
```

You can also `await run.next(node)` to drive step by step, read `run.ctx.state` / `run.ctx.deps` to introspect, and inside a node stream that step's events with `async with node.stream(run.ctx) as stream:` (for `ModelRequestNode` and `CallToolsNode`).

## `event_stream_handler=` — tap events without changing API shape

Any of `run`, `run_sync`, `run_stream` or `iter` accept an `event_stream_handler: EventStreamHandler[Deps]` (`agent/abstract.py`), a callable `(RunContext, AsyncIterable[AgentStreamEvent]) -> Awaitable[None]` that fires with the event stream. The run's return type is unchanged. (Only the per-run argument survives in 2.0; the agent-level `Agent(event_stream_handler=...)` became `capabilities=[ProcessEventStream(...)]`.)

```python
from collections import Counter
from collections.abc import AsyncIterable

from pydantic_ai import Agent, AgentStreamEvent, RunContext

agent = Agent('openai:%%model=openai%%')
event_counts: Counter[str] = Counter()


async def tap(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for e in events:
        event_counts[type(e).__name__] += 1


result = agent.run_sync('hi', event_stream_handler=tap)
print(result.output, dict(event_counts))
```

## Sync streaming — `run_stream_sync`

`agent.run_stream_sync(...)` returns a `StreamedRunResultSync` (`result.py`). All stream methods yield **sync** iterators — handy for CLI tools.

```python
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')

with agent.run_stream_sync('hi') as stream:
    for chunk in stream.stream_text(delta=True):
        print(chunk, end='')
    final = stream.get_output()
```

Under the hood a `SyncStreamBridge` keeps the async stream open in a long-lived task on the calling thread's event loop; don't call it from within an already-running event loop.

## Debouncing and partial validation

- Structured output + partial validation: Pydantic's experimental [partial validator](https://docs.pydantic.dev/latest/concepts/experimental/#partial-validation) runs on each `stream_output` yield. It completes truncated strings and lists, but it cannot invent a **required** field that has not arrived: a partial that is still missing one fails validation and is skipped. If every field is required, the first yield comes only once the last field has started. Give the fields that should appear progressively a default (`None`, `[]`), as below.
- Text streaming: no validation during delta streaming. Validators only run inside `get_output()`.
- Debounce default (`0.1` s) is chosen to balance perceived latency vs. per-chunk parsing cost. Set `None` for token-by-token; set larger values (`0.25`–`0.5`) for long structured outputs.

The effect is easy to see with a `FunctionModel` that streams an output-tool call twelve characters at a time:

```python
import asyncio
import json

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel


class Report(BaseModel):
    title: str                           # required: nothing is yielded until it starts
    summary: str | None = None           # optional: streams in progressively
    key_findings: list[str] = []
    confidence_score: float | None = None


FULL = json.dumps({'title': 'Solar', 'summary': 'Panels are cheap now', 'key_findings': ['cost fell', 'output rose']})


async def stream_fn(messages, info: AgentInfo):
    name = info.output_tools[0].name
    for i in range(0, len(FULL), 12):
        yield {0: DeltaToolCall(name=name if i == 0 else None, json_args=FULL[i : i + 12])}


agent = Agent(FunctionModel(stream_function=stream_fn), output_type=Report)


async def main():
    async with agent.run_stream('Write the report.') as stream:
        async for partial in stream.stream_output(debounce_by=None):
            print(partial.model_dump(exclude_none=True))


asyncio.run(main())
```

```
{'title': 'S', 'key_findings': []}
{'title': 'Solar', 'key_findings': []}
{'title': 'Solar', 'summary': 'Panel', 'key_findings': []}
...
{'title': 'Solar', 'summary': 'Panels are cheap now', 'key_findings': ['cost fell', 'output rose']}
```

## Streaming over HTTP (FastAPI)

The event stream slots straight into Server-Sent Events or NDJSON:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent

app = FastAPI()
agent = Agent('openai:%%model=openai%%')

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

The first, second and fifth patterns are sketches: `q` is the user's prompt and `ui` stands for your front end.

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
import asyncio

from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')


async def main():
    async with agent.run_stream('Write a limerick.') as s:
        async for _ in s.stream_text(delta=True):
            pass
        await s.get_output()
        used = s.usage  # RunUsage — a property since 2.0 (was s.usage() in V1)
    print(used)


asyncio.run(main())
```

### 4. Tool-call tracing via `run_stream_events`

```python
import asyncio
import logging

from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('agent.tools')
agent = Agent('openai:%%model=openai%%')


@agent.tool_plain
def lookup(sku: str) -> str:
    """Stock level for a SKU."""
    return f'{sku}: 12 in stock'


async def main():
    async with agent.run_stream_events('Is ABC in stock?') as events:
        async for e in events:
            if isinstance(e, FunctionToolCallEvent):
                logger.info('tool_call %s %s', e.part.tool_name, e.part.args)
            elif isinstance(e, FunctionToolResultEvent):
                logger.info('tool_result %s', e.part.tool_name)


asyncio.run(main())
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

agent = Agent('anthropic:%%model=anthropic-flagship%%', model_settings={'thinking': 'low'})

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

## `cancel()` — stream lifecycle control

**`cancel()`** stops local consumption of the current model response and asks the provider SDK to shut the stream down (whether remote generation actually stops depends on the provider). The interrupted response is recorded in the message history. It stops only the current response; to end a whole multi-step run use `AgentRun.cancel()` or `RunContext.cancel()`. Call it inside the `async with` block:

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')

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

There is no `drain()` on `StreamedRunResult`; leaving the `async with` block early is enough to close it cleanly. (`drain()` does exist on the lower-level `AgentStream` you get from `ModelRequestNode.stream(ctx)` inside `agent.iter()`, where it discards the remaining events of that one response.)

## Accessing usage and identity after streaming

Properties available after the stream completes (after `get_output()` or after draining):

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')

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
- **`TextOutput` and `stream_text()`**: `stream_text()` yields the raw text and does not apply a `TextOutput` function; use `stream_output()` to stream the function's result.

## Reference

- `Agent.run_stream(...)`, `Agent.run_stream_events(...)`, `Agent.run_stream_sync(...)`, `AgentRunEvents` — `agent/abstract.py`
- `Agent.iter(...)` — `agent/__init__.py`
- `StreamedRunResult`, `AgentStream` — `result.py`
- `AgentRunResultEvent` — `run.py`
- `AgentStreamEvent` union — `messages.py` (`PartStart`, `PartDelta`, `PartEnd`, `FinalResult`, `FunctionToolCall`, `FunctionToolResult`, `OutputToolCall`, `OutputToolResult`, …)
- `EventStreamHandler` type alias — `agent/abstract.py`

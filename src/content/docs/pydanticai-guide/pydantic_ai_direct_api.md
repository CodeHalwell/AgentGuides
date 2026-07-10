---
title: "PydanticAI: Direct API — Raw Model Calls"
description: "model_request, model_request_sync, model_request_stream, model_request_stream_sync, StreamedResponseSync — make bare model calls without an Agent when you need minimal abstraction."
framework: pydanticai
language: python
---

# Direct API — Raw Model Calls

Verified against **pydantic-ai==2.8.0** — source module: `pydantic_ai.direct`.

`pydantic_ai.direct` exposes four functions that talk to a language model with **no Agent overhead**: no tool execution, no output validation, no message-history management, no retry loop. The only abstraction is translating your `ModelMessage` list into the provider's wire format and back. This is the right layer when you're building custom pipelines, comparing model outputs, writing SDK-level unit tests, or calling a model as part of a custom capability.

## When to use the direct API vs Agent

| Use case | Use |
|----------|-----|
| Structured output validation, tool calls, retry loop | `Agent` |
| History management, multi-turn chat | `Agent` |
| Raw model comparison / benchmarking | `model_request` |
| Building a custom capability or wrapper | `model_request` |
| REPL / notebook one-shots | `model_request_sync` |
| Synchronous CLI tool | `model_request_stream_sync` |
| LLM-powered library helper (no event loop expected) | `model_request_sync` |

## Minimal runnable example

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request

async def main():
    response = await model_request(
        'openai:gpt-4o-mini',
        [ModelRequest.user_text_prompt('What is the capital of France?')],
    )
    print(response)
    # ModelResponse(
    #     parts=[TextPart(content='The capital of France is Paris.')],
    #     usage=RequestUsage(input_tokens=14, output_tokens=8),
    #     model_name='gpt-4o-mini',
    #     timestamp=datetime.datetime(...),
    # )

asyncio.run(main())
```

## `model_request` — async, non-streaming

```python
from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.direct import model_request
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

async def ask(prompt: str) -> ModelResponse:
    return await model_request(
        'anthropic:claude-haiku-4-5',
        [ModelRequest.user_text_prompt(prompt)],
        model_settings=ModelSettings(max_tokens=512, temperature=0.2),
    )
```

**Signature** (`direct.py`)

```python
async def model_request(
    model: Model | KnownModelName | str,
    messages: Sequence[ModelMessage],
    *,
    model_settings: ModelSettings | None = None,
    model_request_parameters: ModelRequestParameters | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> ModelResponse: ...
```

- `model` — any `KnownModelName` string, a `Model` instance, or a raw string.
- `messages` — a sequence of `ModelMessage` objects (see _Building messages_ below).
- `model_settings` — same `ModelSettings` TypedDict you'd pass to `Agent`.
- `model_request_parameters` — control tools, output schemas, sampling parameters.
- `instrument` — `True` / `InstrumentationSettings` to emit OTel spans. Defaults to the global `Agent._instrument_default`.

## `model_request_sync` — synchronous, non-streaming

Drop-in synchronous wrapper around `model_request`. Runs a new event loop internally — **do not call from within an already-running async loop**.

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

response = model_request_sync(
    'google-gla:gemini-2.0-flash',
    [ModelRequest.user_text_prompt('Explain backpropagation in two sentences.')],
)
print(response.parts[0].content)  # TextPart
print(response.usage.input_tokens, response.usage.output_tokens)
```

Useful in CLI scripts, Jupyter cells, or libraries that must expose a synchronous API.

## `model_request_stream` — async streaming

Returns an async context manager that yields a `StreamedResponse`. Iterate to get `ModelResponseStreamEvent` objects.

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta

async def stream_response(prompt: str) -> str:
    text = ''
    messages = [ModelRequest.user_text_prompt(prompt)]
    async with model_request_stream('openai:gpt-4o-mini', messages) as stream:
        async for event in stream:
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                text += event.delta.content_delta
                print(event.delta.content_delta, end='', flush=True)
    print()
    return text

asyncio.run(stream_response('Tell me a short joke.'))
```

Stream events (`messages.py`):

| Event | When emitted |
|-------|-------------|
| `PartStartEvent` | A new part (text, thinking, tool call) begins |
| `PartDeltaEvent` | Token delta arrives; `.delta` is `TextPartDelta`, `ThinkingPartDelta`, or `ToolCallPartDelta` |
| `PartEndEvent` | The part is complete |
| `FinalResultEvent` | The final result tool call was selected |

After the `async with` exits, `stream.get()` returns the assembled `ModelResponse`.

```python
async with model_request_stream('openai:gpt-4o-mini', messages) as stream:
    async for _ in stream:
        pass  # drain
    response = stream.get()
    print(response.usage)
```

## `model_request_stream_sync` — synchronous streaming

The synchronous peer of `model_request_stream`. Uses a background thread with its own event loop to run the async producer, exposing a standard context-manager + iterator interface.

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream_sync
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

messages = [ModelRequest.user_text_prompt('Count from 1 to 5.')]

with model_request_stream_sync('openai:gpt-4o-mini', messages) as stream:
    for event in stream:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            print(event.delta.content_delta, end='', flush=True)
print()
```

`model_request_stream_sync` **must be used as a context manager** — accessing it outside `with` raises `RuntimeError`.

## `StreamedResponseSync` — the sync stream wrapper

`model_request_stream_sync` returns a `StreamedResponseSync` instance, a dataclass that owns the background thread:

```python
from pydantic_ai.direct import StreamedResponseSync, model_request_stream_sync
from pydantic_ai import ModelRequest

messages = [ModelRequest.user_text_prompt('Who invented the telephone?')]

with model_request_stream_sync('openai:gpt-4o-mini', messages) as stream:
    # Iterate to consume
    chunks = list(stream)

    # After consuming the stream, inspect the assembled response
    print(stream.response)          # ModelResponse
    print(stream.model_name)        # e.g. 'gpt-4o-mini'
    print(stream.timestamp)         # datetime of the response
```

Key attributes (source: `direct.py`):

| Attribute | Type | Notes |
|-----------|------|-------|
| `response` | `ModelResponse` | Current / final assembled response |
| `model_name` | `str` | Model name from the provider |
| `timestamp` | `datetime` | Timestamp from the provider |

The background thread is joined on `__exit__`; it is safe to inspect the response immediately after the `with` block.

## Building messages

`ModelRequest` (`messages.py`) is the standard container. The convenience constructor `ModelRequest.user_text_prompt(...)` builds a request with a `UserPromptPart`:

```python
from pydantic_ai import ModelRequest
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
    SystemPromptPart,
    InstructionPart,
    BinaryImage,
    ImageUrl,
)

# Plain text turn
turn = ModelRequest.user_text_prompt('What is 2 + 2?')

# Text turn with a system prompt
multi = ModelRequest(parts=[
    SystemPromptPart(content='You are a concise assistant. Reply in one sentence.'),
    UserPromptPart(content='What is the capital of Spain?'),
])

# Add system-level instructions (preferred over SystemPromptPart for multi-turn)
with_instr = ModelRequest(
    instructions='Answer in bullet points.',
    parts=[UserPromptPart(content='List three advantages of Python.')],
)

# Multimodal: image URL
with_image = ModelRequest(parts=[
    UserPromptPart(content=[
        ImageUrl(url='https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/200px-PNG_transparency_demonstration_1.png'),
        'What do you see in this image?',
    ]),
])
```

## Multi-turn conversations

Chain turns by passing the prior history as additional messages:

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.messages import ModelMessages, ModelMessagesTypeAdapter
from pydantic_ai.direct import model_request

async def chat():
    history: list = [ModelRequest.user_text_prompt('My favourite colour is green.')]

    resp1 = await model_request('openai:gpt-4o-mini', history)
    history.append(resp1)  # ModelResponse also is a ModelMessage

    history.append(ModelRequest.user_text_prompt('What is my favourite colour?'))
    resp2 = await model_request('openai:gpt-4o-mini', history)
    print(resp2.parts[0].content)   # 'Your favourite colour is green.'

asyncio.run(chat())
```

## `ModelRequestParameters` — tools and output schemas

Pass `model_request_parameters` when you need to advertise tool schemas or set an output schema:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

class WeatherResult(BaseModel):
    temperature_c: float
    condition: str

# Advertise a tool schema (the model will emit a ToolCallPart if it wants to call it)
weather_tool = ToolDefinition(
    name='get_weather',
    description='Get current weather for a city.',
    parameters_json_schema={
        'type': 'object',
        'properties': {'city': {'type': 'string'}},
        'required': ['city'],
    },
)

async def main():
    resp = await model_request(
        'openai:gpt-4o-mini',
        [ModelRequest.user_text_prompt('What is the weather in Paris?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[weather_tool],
        ),
    )
    for part in resp.parts:
        if part.part_kind == 'tool-call':
            print(f'Model wants to call: {part.tool_name}({part.args_as_dict()})')

asyncio.run(main())
```

## OTel instrumentation on direct calls

Pass `instrument=True` (or a custom `InstrumentationSettings`) to emit spans:

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request
from pydantic_ai.models.instrumented import InstrumentationSettings

settings = InstrumentationSettings(include_content=True, version=4)

async def main():
    response = await model_request(
        'anthropic:claude-haiku-4-5',
        [ModelRequest.user_text_prompt('Hello!')],
        instrument=settings,
    )
    print(response.parts[0].content)

asyncio.run(main())
```

## Inspecting the `ModelResponse`

```python
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ThinkingPart

def inspect(response: ModelResponse) -> None:
    print(f'model: {response.model_name}')
    print(f'timestamp: {response.timestamp}')
    print(f'input tokens: {response.usage.input_tokens}')
    print(f'output tokens: {response.usage.output_tokens}')
    print(f'finish reason: {response.finish_reason}')  # 'stop' | 'tool_calls' | 'length' | None

    for part in response.parts:
        if isinstance(part, TextPart):
            print(f'text: {part.content[:80]}')
        elif isinstance(part, ThinkingPart):
            print(f'thinking: {part.thinking[:80]}')
        elif isinstance(part, ToolCallPart):
            print(f'tool call: {part.tool_name}({part.args_as_json_str()})')
```

## Patterns

### 1. Provider comparison harness

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request

MODELS = [
    'openai:gpt-4o-mini',
    'anthropic:claude-haiku-4-5',
    'google-gla:gemini-2.0-flash',
]

async def compare(prompt: str) -> None:
    messages = [ModelRequest.user_text_prompt(prompt)]
    results = await asyncio.gather(*[
        model_request(m, messages) for m in MODELS
    ])
    for model_name, resp in zip(MODELS, results):
        text = next((p.content for p in resp.parts if p.part_kind == 'text'), '')
        print(f'{model_name}: {text[:120]}')
        print(f'  tokens: in={resp.usage.input_tokens} out={resp.usage.output_tokens}')

asyncio.run(compare('Explain async/await in Python in one sentence.'))
```

### 2. Synchronous embedding-style call (library helper)

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

def classify(text: str, categories: list[str]) -> str:
    """Synchronous helper for non-async callers."""
    cats = ', '.join(categories)
    resp = model_request_sync(
        'openai:gpt-4o-mini',
        [ModelRequest(parts=[
            __import__('pydantic_ai.messages', fromlist=['SystemPromptPart']).SystemPromptPart(
                content=f'Classify the input into exactly one of: {cats}. Reply with the category name only.'
            ),
            __import__('pydantic_ai.messages', fromlist=['UserPromptPart']).UserPromptPart(content=text),
        ])],
    )
    return resp.parts[0].content.strip()

print(classify('I love the new Python 3.13 release', ['tech', 'sports', 'politics']))
```

### 3. Async stream with early exit and response inspection

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

async def stream_until_marker(prompt: str, stop_token: str = '[DONE]') -> str:
    collected = ''
    async with model_request_stream('openai:gpt-4o-mini',
                                    [ModelRequest.user_text_prompt(prompt)]) as stream:
        async for event in stream:
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                collected += event.delta.content_delta
                if stop_token in collected:
                    break
    return collected

asyncio.run(stream_until_marker('Say "Hello [DONE] World"'))
```

### 4. Mixing direct calls inside a custom capability

The direct API is ideal inside a `WrapModelRequestHandler` or a custom `AbstractCapability` where you want to call the model a second time (e.g. an inner-monologue step) without going through the full agent machinery:

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelResponse

async def self_critique(original_response: ModelResponse, model: str) -> str:
    """Ask the model to critique its own previous response."""
    original_text = next(
        (p.content for p in original_response.parts if p.part_kind == 'text'), ''
    )
    critique_resp = await model_request(
        model,
        [
            ModelRequest.user_text_prompt(
                f'Critique the following response and point out any inaccuracies:\n\n{original_text}'
            )
        ],
    )
    return next((p.content for p in critique_resp.parts if p.part_kind == 'text'), '')
```

### 5. Retrying with exponential backoff

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse

async def resilient_request(
    model: str, prompt: str, retries: int = 3
) -> ModelResponse:
    messages = [ModelRequest.user_text_prompt(prompt)]
    for attempt in range(retries):
        try:
            return await model_request(model, messages)
        except ModelAPIError as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f'Attempt {attempt + 1} failed ({exc}). Retrying in {wait}s…')
            await asyncio.sleep(wait)
    raise RuntimeError('unreachable')
```

## Gotchas

- **No tool execution.** If the model returns a `ToolCallPart`, it's your responsibility to run the function and continue the conversation (or ignore it). Use `Agent` if you want the loop managed for you.
- **No output validators.** The response is the raw `ModelResponse`; Pydantic validation runs only if you parse `parts` yourself.
- **`model_request_sync` / `model_request_stream_sync` block the thread.** Do not call them from inside `asyncio.run` or from an async function — they create a new event loop and will deadlock or raise.
- **`StreamedResponseSync` must be used as a context manager.** Calling `iter(stream)` outside a `with` block raises `RuntimeError`.
- **`instrument=None` inherits the global default.** Set `Agent.instrument_all(...)` once at startup to instrument all direct calls automatically.

## Reference

| Symbol | Module | Notes |
|--------|--------|-------|
| `model_request` | `pydantic_ai.direct` | Async, non-streamed |
| `model_request_sync` | `pydantic_ai.direct` | Sync wrapper around `model_request` |
| `model_request_stream` | `pydantic_ai.direct` | Async context manager → `StreamedResponse` |
| `model_request_stream_sync` | `pydantic_ai.direct` | Sync context manager → `StreamedResponseSync` |
| `StreamedResponseSync` | `pydantic_ai.direct` | Background-thread async producer; sync iterator |
| `ModelRequest.user_text_prompt` | `pydantic_ai.messages` | Convenience constructor for a plain text user turn |
| `ModelRequestParameters` | `pydantic_ai.models` | Tool schemas, output schema, sampling |
| `ModelResponse` | `pydantic_ai.messages` | Provider response with parts + usage |
| `InstrumentationSettings` | `pydantic_ai.models.instrumented` | OTel settings; pass to `instrument=` |

---
title: "PydanticAI Class Deep Dives Vol. 21"
description: "Source-verified deep dives into 10 pydantic-ai 1.107.0 classes: TestModel/TestStreamedResponse (deterministic test doubles), FunctionModel/AgentInfo (Python-function-driven model), DeltaToolCall/DeltaThinkingPart/stream delta types (streaming deltas for custom models), FunctionStreamedResponse (streaming companion for FunctionModel), ImageGenerationTool 1.107.0 complete (12 fields — new input_fidelity/partial_images/Google sizes/aspect_ratio), WebFetchTool 1.107.0 complete (new max_content_tokens/enable_citations), MCPServerTool 1.107.0 complete (new headers/connector_id), FileSearchTool multi-provider (OpenAI vector stores/Google Gemini Files API/xAI collections), ToolSearchStrategy types (ToolSearchNativeStrategy/ToolSearchLocalStrategy/ToolSearchFunc), and ToolSearchTool internal (framework-internal bm25/regex/custom dispatch)."
sidebar:
  order: 47
---

import { Aside } from '@astrojs/starlight/components';

# PydanticAI Class Deep Dives Vol. 21

Source-verified against **pydantic-ai 1.107.0** installed at `/tmp/pydantic_ai_install/pydantic_ai/`.

Each section covers one class group with key behaviours extracted directly from source, a quick-reference table, and three standalone runnable examples.

---

## 1 · `TestModel` + `TestStreamedResponse` — Deterministic test doubles for agents

**Module**: `pydantic_ai.models.test`  
**Exported as**: `pydantic_ai.models.test.TestModel`, `pydantic_ai.models.test.TestStreamedResponse`

`TestModel` is a fully deterministic fake model for unit testing. By default it calls **all** function tools, then either uses an output tool or returns a plain text response. `TestStreamedResponse` streams the same result word-by-word, simulating real token delivery and even raising `httpx.StreamClosed` when cancelled mid-stream.

<Aside type="note">
`TestModel.__test__ = False` prevents pytest from mistakenly treating it as a test class.
</Aside>

### Key behaviours (source-verified)

| Field / property | Default | Notes |
|---|---|---|
| `call_tools` | `'all'` | Pass `['tool_a', 'tool_b']` to call a specific subset |
| `custom_output_text` | `None` | Forces the model to return this string as its text output |
| `custom_output_args` | `None` | Forces the model to call the first output tool with these args |
| `seed` | `0` | Controls `_JsonSchemaTestData` — increments pick a different enum/string |
| `model_name` | `'test'` | Reported in `ModelResponse.model_name` |
| `last_model_request_parameters` | `None` | Set after the first call; inspect to see tool definitions seen by the model |
| `profile` | _(from parent)_ | Optional `ModelProfileSpec` override |
| `settings` | _(from parent)_ | Optional `ModelSettings` defaults |
| `supported_native_tools()` | All except `ToolSearchTool` | Class method — `ToolSearchTool` excluded because TestModel can't emulate provider search |

**Execution logic (source)**:
1. If tools exist and no `ModelResponse` yet in messages → call all tools first
2. If retry prompts present → re-call the failing tools
3. If `custom_output_text` set → return `TextPart(custom_output_text)`
4. If `custom_output_args` set → call `output_tools[0]` with those args
5. If `allow_text_output` → return JSON summary of all tool return values
6. If `output_tools` → call `output_tools[seed % len(output_tools)]` with generated args

**`_JsonSchemaTestData`** generates minimal valid data from a JSON schema. It respects `const`, `enum` (picks `enum[seed % len(enum)]`), `$ref`, `anyOf`, and all primitive types. String generation respects `minLength` and `format: date`.

```python
# 1 — Basic TestModel usage
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('test', system_prompt='You are a calculator.')

@agent.tool_plain
def add(a: int, b: int) -> int:
    return a + b

async def main():
    model = TestModel()
    result = await agent.run('What is 2+2?', model=model)
    # TestModel calls add() with generated args, then returns JSON of tool returns
    print(result.output)
    print(model.last_model_request_parameters.function_tools)

asyncio.run(main())
```

```python
# 2 — Control output with custom_output_text and seed
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('test', output_type=str)

async def main():
    model = TestModel(custom_output_text='The answer is 42', seed=3)
    result = await agent.run('Any question', model=model)
    print(result.output)  # 'The answer is 42'

asyncio.run(main())
```

```python
# 3 — Inspect last_model_request_parameters and stream TestStreamedResponse
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('test', system_prompt='Count words.')

@agent.tool_plain
def count_words(text: str) -> int:
    return len(text.split())

async def main():
    model = TestModel(call_tools=['count_words'], custom_output_text='Done')
    async with agent.run_stream('Hello world', model=model) as streamed:
        async for chunk in streamed.stream_text():
            print(repr(chunk), end=' ')
    print()
    # Inspect what tool definitions the model received
    params = model.last_model_request_parameters
    for td in params.function_tools:
        print(f'{td.name}: {td.parameters_json_schema}')

asyncio.run(main())
```

---

## 2 · `FunctionModel` + `AgentInfo` — Python-function-driven custom model

**Module**: `pydantic_ai.models.function`  
**Exported as**: `pydantic_ai.models.function.FunctionModel`, `pydantic_ai.models.function.AgentInfo`

`FunctionModel` replaces the LLM with a Python function you write. It gives you complete control over what the model responds — perfect for testing complex scenarios, replay, and deterministic output validation. The function receives `(messages, agent_info)` and returns a `ModelResponse`.

`AgentInfo` is a frozen dataclass passed as the second argument to your function. It exposes the full `ModelRequestParameters` including available tools, output schema, model settings, and extracted instructions.

### Key behaviours (source-verified)

| Parameter | Required | Notes |
|---|---|---|
| `function` | one of these | Sync or async `(messages, agent_info) -> ModelResponse` |
| `stream_function` | one of these | Async generator yielding `str \| DeltaToolCalls \| DeltaThinkingCalls \| BuiltinToolCallsReturns` |
| `model_name` | No | Auto-derives from `function.__name__` if omitted |
| `profile` | No | Defaults to `ModelProfile(supports_json_schema_output=True, supports_json_object_output=True)` |
| `settings` | No | `ModelSettings` defaults |

**Default profile**: `FunctionModel` injects a profile with `supports_json_schema_output=True` and `supports_json_object_output=True` automatically, so structured output works out of the box without provider setup.

**`AgentInfo` fields**:

| Field | Type | Notes |
|---|---|---|
| `function_tools` | `list[ToolDefinition]` | All registered `@agent.tool` / `@agent.tool_plain` tools |
| `allow_text_output` | `bool` | Whether the model may return plain text |
| `output_tools` | `list[ToolDefinition]` | Tools wrapping structured output types |
| `model_settings` | `ModelSettings \| None` | Run-level model settings |
| `model_request_parameters` | `ModelRequestParameters` | Full request parameters |
| `instructions` | `str \| None` | Extracted system prompt/instructions text |

```python
# 1 — Sync FunctionModel echoing the last user message
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart, ModelRequest, UserPromptPart

def echo_model(messages, agent_info: AgentInfo) -> ModelResponse:
    last_user = next(
        p.content for m in reversed(messages)
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart)
    )
    return ModelResponse(parts=[TextPart(f'Echo: {last_user}')])

agent = Agent(FunctionModel(echo_model))

async def main():
    result = await agent.run('Hello, world!')
    print(result.output)  # Echo: Hello, world!

asyncio.run(main())
```

```python
# 2 — Async FunctionModel that calls the first available tool
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

agent = Agent(None, system_prompt='You are a calculator.')  # model provided at run time

@agent.tool_plain
def multiply(a: int, b: int) -> int:
    return a * b

async def smart_model(messages, agent_info: AgentInfo) -> ModelResponse:
    # First turn: call multiply if not yet called
    already_called = any(
        isinstance(p, ToolCallPart)
        for m in messages
        for p in (m.parts if hasattr(m, 'parts') else [])
    )
    if not already_called and agent_info.function_tools:
        tool = agent_info.function_tools[0]
        args = {'a': 6, 'b': 7}
        return ModelResponse(parts=[ToolCallPart(tool.name, args)])
    # Second turn: report the result
    return ModelResponse(parts=[TextPart('Calculation complete.')])

async def main():
    result = await agent.run('Compute 6×7', model=FunctionModel(smart_model))
    print(result.output)

asyncio.run(main())
```

```python
# 3 — Inspect AgentInfo to assert tool schema in tests
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart

agent = Agent(None)  # model provided at run time via model=FunctionModel(...)

@agent.tool_plain
def search(query: str, limit: int = 10) -> list[str]:
    """Search for documents matching query."""
    return []

captured_info: AgentInfo | None = None

def capturing_model(messages, agent_info: AgentInfo) -> ModelResponse:
    global captured_info
    captured_info = agent_info
    return ModelResponse(parts=[TextPart('ok')])

async def test_tool_schema():
    await agent.run('test', model=FunctionModel(capturing_model))
    assert captured_info is not None
    tool_names = [t.name for t in captured_info.function_tools]
    assert 'search' in tool_names
    search_tool = next(t for t in captured_info.function_tools if t.name == 'search')
    props = search_tool.parameters_json_schema.get('properties', {})
    assert 'query' in props
    assert props.get('limit', {}).get('default') == 10

asyncio.run(test_tool_schema())
print('Schema assertions passed')
```

---

## 3 · `DeltaToolCall` + `DeltaThinkingPart` + delta type aliases — Streaming deltas

**Module**: `pydantic_ai.models.function`  
**Exported as**: `DeltaToolCall`, `DeltaThinkingPart`, `DeltaToolCalls`, `DeltaThinkingCalls`, `BuiltinToolCallsReturns`

These dataclasses and type aliases represent incremental chunks yielded by a `StreamFunctionDef`. Each chunk is one of: a plain `str` (text delta), a `DeltaToolCalls` dict (tool call delta), a `DeltaThinkingCalls` dict (thinking/reasoning delta), or a `BuiltinToolCallsReturns` dict (native tool call/return).

### Key behaviours (source-verified)

| Type | What to yield | Effect |
|---|---|---|
| `str` | Text content chunk | Appended to current `TextPart` |
| `DeltaToolCalls = dict[int, DeltaToolCall]` | `{index: DeltaToolCall(name=..., json_args=...)}` | Incrementally builds `ToolCallPart` |
| `DeltaThinkingCalls = dict[int, DeltaThinkingPart]` | `{index: DeltaThinkingPart(content=..., signature=...)}` | Builds `ThinkingPart` |
| `BuiltinToolCallsReturns = dict[int, NativeToolCallPart \| NativeToolReturnPart]` | `{index: part}` | Emits native tool parts |

**`DeltaToolCall` fields**:

| Field | Type | Notes |
|---|---|---|
| `name` | `str \| None` | Tool name delta (typically sent once at start) |
| `json_args` | `str \| None` | Partial JSON string accumulates across chunks |
| `tool_call_id` | `str \| None` | ID delta |

**`DeltaThinkingPart` fields**:

| Field | Type | Notes |
|---|---|---|
| `content` | `str \| None` | Thinking content delta |
| `signature` | `str \| None` | Thinking signature delta; providing `signature` marks the part as from `provider_name='function'` |

<Aside type="caution">
You must yield all strings, all `DeltaToolCalls`, or all `DeltaThinkingCalls` — mixing types within a single stream is not supported. The type alias documents this constraint: `AsyncIterator[str] | AsyncIterator[DeltaToolCalls] | ...`.
</Aside>

```python
# 1 — Streaming text word by word
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage

async def stream_text(messages: list[ModelMessage], info: AgentInfo):
    words = 'The answer is forty two'.split()
    for word in words:
        yield word + ' '

agent = Agent(FunctionModel(stream_function=stream_text))

async def main():
    async with agent.run_stream('What is 6x7?') as r:
        async for chunk in r.stream_text():
            print(chunk, end='', flush=True)
    print()

asyncio.run(main())
```

```python
# 2 — Streaming a tool call incrementally via DeltaToolCalls
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo, DeltaToolCall
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

agent = Agent(None)  # model provided at run time via model=FunctionModel(...)

@agent.tool_plain
def calculate(expression: str) -> float:
    return eval(expression)  # noqa: S307 (example only)

async def streaming_tool_model(messages: list[ModelMessage], info: AgentInfo):
    already_called = any(
        hasattr(m, 'parts') and any(hasattr(p, 'tool_name') for p in m.parts)
        for m in messages
    )
    if not already_called and info.function_tools:
        tool = info.function_tools[0]
        # Stream the tool call in two chunks
        yield {0: DeltaToolCall(name=tool.name, tool_call_id='call_01')}
        yield {0: DeltaToolCall(json_args='{"expression": "2**10"}')}
    else:
        yield 'The result is ready.'

async def main():
    async with agent.run_stream('2 to the power of 10', model=FunctionModel(stream_function=streaming_tool_model)) as r:
        async for chunk in r.stream_text():
            print(chunk, end='')
    print()

asyncio.run(main())
```

```python
# 3 — Streaming thinking content via DeltaThinkingCalls
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo, DeltaThinkingPart
from pydantic_ai.messages import ModelMessage, ThinkingPart

async def thinking_stream(messages: list[ModelMessage], info: AgentInfo):
    # Yield thinking deltas (index 0)
    yield {0: DeltaThinkingPart(content='Let me think... ')}
    yield {0: DeltaThinkingPart(content='The capital of France is Paris.')}
    # Signal end of thinking with signature
    yield {0: DeltaThinkingPart(signature='<END_THINKING>')}

# Must yield all thinking first, then switch to text in a new stream (separate async calls)
agent = Agent(FunctionModel(stream_function=thinking_stream))

async def main():
    async with agent.run_stream('What is the capital of France?') as r:
        async for event in r.stream_response():
            pass
        response = await r.get_output()
    # Check the full message history for thinking parts
    for msg in r.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if isinstance(part, ThinkingPart):
                    print(f'Thinking: {part.content}')

asyncio.run(main())
```

---

## 4 · `FunctionStreamedResponse` — Streaming companion for `FunctionModel`

**Module**: `pydantic_ai.models.function`  
**Exported as**: `pydantic_ai.models.function.FunctionStreamedResponse`

`FunctionStreamedResponse` is the `StreamedResponse` implementation returned by `FunctionModel.request_stream`. It wraps the async generator from your `stream_function` and dispatches each yielded chunk through the shared `ModelResponsePartsManager` so the standard streaming API (`stream_text`, `stream_response`, `stream_output`) works without modification.

### Key behaviours (source-verified)

| Property | Returns | Notes |
|---|---|---|
| `model_name` | `str` | Set from `FunctionModel._model_name` |
| `provider_name` | `None` | FunctionModel has no provider |
| `provider_url` | `None` | FunctionModel has no URL |
| `timestamp` | `datetime` | UTC at response creation time |
| `close_stream()` | coroutine | No-op (no underlying connection) |

**Token estimation**: `_estimate_usage` adds 50 overhead tokens and approximates input/output from raw string splitting on `[\s",.:]+`. This gives plausible test numbers rather than exact billing counts.

**`PeekableAsyncStream`**: `FunctionModel.request_stream` wraps the stream function in a `PeekableAsyncStream` and peeks the first item before yielding `FunctionStreamedResponse`. This ensures an empty generator raises `ValueError` immediately rather than on first iteration.

```python
# 1 — Verify streaming works end-to-end with FunctionStreamedResponse
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage

async def word_stream(messages: list[ModelMessage], info: AgentInfo):
    for word in ['Hello', ' ', 'from', ' ', 'FunctionStreamedResponse']:
        yield word

agent = Agent(FunctionModel(stream_function=word_stream))

async def main():
    chunks = []
    async with agent.run_stream('greet') as r:
        async for chunk in r.stream_text():
            chunks.append(chunk)
    print(''.join(chunks))
    # Check usage estimation
    usage = r.usage()
    print(f'input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}')

asyncio.run(main())
```

```python
# 2 — Empty stream raises ValueError
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage

async def empty_stream(messages: list[ModelMessage], info: AgentInfo):
    return
    yield  # unreachable — makes this an async generator so PeekableAsyncStream can peek it

agent = Agent(FunctionModel(stream_function=empty_stream))

async def main():
    try:
        async with agent.run_stream('test') as r:
            async for _ in r.stream_text():
                pass
    except ValueError as e:
        print(f'Caught: {e}')

asyncio.run(main())
```

```python
# 3 — Combining sync function with streaming function on the same FunctionModel
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

def sync_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('sync response')])

async def stream_fn(messages: list[ModelMessage], info: AgentInfo):
    yield 'streamed '
    yield 'response'

agent = Agent(FunctionModel(sync_fn, stream_function=stream_fn))

async def main():
    # Non-streamed uses sync_fn
    r1 = await agent.run('test')
    print(r1.output)  # sync response

    # Streamed uses stream_fn
    async with agent.run_stream('test') as r2:
        print(await r2.get_output())  # streamed response

asyncio.run(main())
```

---

## 5 · `ImageGenerationTool` 1.107.0 complete — All 12 fields

**Module**: `pydantic_ai.native_tools`  
**Exported as**: `pydantic_ai.ImageGenerationTool`, `pydantic_ai.native_tools.ImageAspectRatio`, `pydantic_ai.native_tools.ImageGenerationModelName`

`ImageGenerationTool` is a native tool that enables image generation by OpenAI Responses and Google (Gemini) models. Version 1.107.0 added four new fields: `input_fidelity`, `partial_images`, Google-specific `size` literals (`'512'`, `'1K'`, `'2K'`, `'4K'`), and `aspect_ratio`.

### Key behaviours (source-verified)

| Field | Default | Provider support | Notes |
|---|---|---|---|
| `action` | `'auto'` | OpenAI | `'generate'`, `'edit'`, `'auto'` |
| `background` | `'auto'` | OpenAI | `'transparent'` only with `png`/`webp` |
| `input_fidelity` | `None` | OpenAI | **New 1.107.0** — `'high'`/`'low'` controls how closely edits match input image style/features |
| `moderation` | `'auto'` | OpenAI | `'auto'` or `'low'` |
| `model` | `None` | OpenAI | `ImageGenerationModelName`: `'gpt-image-2'`, `'gpt-image-1.5'`, `'gpt-image-1'`, `'gpt-image-1-mini'` |
| `output_compression` | `None` | OpenAI, Google Vertex | JPEG/WebP compression level; sets `output_format` to `'jpeg'` on Vertex if unset |
| `output_format` | `None` | OpenAI, Google Vertex | `'png'` (default), `'webp'`, `'jpeg'` |
| `partial_images` | `0` | OpenAI | **New 1.107.0** — `0`–`3` intermediate images during streaming |
| `quality` | `'auto'` | OpenAI | `'low'`, `'medium'`, `'high'`, `'auto'` |
| `size` | `None` | OpenAI + Google | **Updated 1.107.0** — `'512'`/`'1K'`/`'2K'`/`'4K'` are Google Gemini sizes; `'1024x1024'`/`'1024x1536'`/`'1536x1024'` are OpenAI |
| `aspect_ratio` | `None` | Google, OpenAI | **New 1.107.0** — `ImageAspectRatio`: 10 values (`'21:9'`…`'4:5'`); OpenAI maps `'1:1'`, `'2:3'`, `'3:2'` |
| `optional` | `False` | All | Inherited from `AbstractNativeTool` — silently dropped on unsupported models |

**`ImageAspectRatio`** is a `Literal` of 10 ratios: `'21:9'`, `'16:9'`, `'4:3'`, `'3:2'`, `'1:1'`, `'9:16'`, `'3:4'`, `'2:3'`, `'5:4'`, `'4:5'`.

**`ImageGenerationModelName`**: OpenAI-specific model IDs as `Literal` plus `str` fallback for future models.

```python
# 1 — OpenAI high-quality PNG with streaming partial images
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import ImageGenerationTool

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        NativeTool(ImageGenerationTool(
            model='gpt-image-2',
            quality='high',
            output_format='png',
            background='transparent',
            partial_images=2,   # 2 previews during streaming
            input_fidelity='high',  # preserve uploaded face/style closely
            size='1024x1024',
        ))
    ],
)

async def main():
    result = await agent.run('Draw a minimalist logo: white circle on navy blue background')
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 2 — Google Gemini portrait image with aspect_ratio and large size
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import ImageGenerationTool

agent = Agent(
    'google-cloud:gemini-3-pro-image-preview',  # google-cloud prefix required for Vertex-only options
    capabilities=[
        NativeTool(ImageGenerationTool(
            size='2K',               # Google-only: 2048px
            aspect_ratio='3:4',      # Portrait orientation
            output_format='jpeg',    # Vertex AI only
            output_compression=85,   # Vertex AI only
        ))
    ],
)

async def main():
    result = await agent.run('A serene mountain lake at dawn, photorealistic')
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 3 — Cross-provider portable config using aspect_ratio (maps to closest size on OpenAI)
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import ImageGenerationTool, ImageAspectRatio

def make_image_agent(model_name: str, ratio: ImageAspectRatio = '1:1'):
    return Agent(
        model_name,
        capabilities=[
            NativeTool(ImageGenerationTool(
                aspect_ratio=ratio,
                quality='medium',
                optional=True,   # silently skip if provider doesn't support it
            ))
        ],
    )

# OpenAI maps '2:3' → '1024x1536'; Google uses '2:3' natively
openai_agent = make_image_agent('openai-responses:gpt-4o', ratio='2:3')
google_agent = make_image_agent('google-gla:gemini-3-pro-image-preview', ratio='2:3')

import asyncio
async def main():
    result = await openai_agent.run('A tall narrow portrait of a lighthouse')
    print(result.output)
asyncio.run(main())
```

---

## 6 · `WebFetchTool` 1.107.0 complete — URL fetching with citations and content limits

**Module**: `pydantic_ai.native_tools`  
**Exported as**: `pydantic_ai.WebFetchTool`

`WebFetchTool` is the native tool for fetching URL content directly within a model's context window. Version 1.107.0 added `max_content_tokens` (limit token consumption per fetch) and `enable_citations` (inline source citations). It supersedes the deprecated `UrlContextTool` alias.

<Aside type="note">
`UrlContextTool` is a deprecated alias for `WebFetchTool` kept for backward compatibility of serialised payloads. Its `kind` field stays `'url_context'` so old JSON deserialises correctly. New code should use `WebFetchTool` directly.
</Aside>

### Key behaviours (source-verified)

| Field | Default | Provider support | Notes |
|---|---|---|---|
| `max_uses` | `None` | Anthropic | Stop fetching after N URLs |
| `allowed_domains` | `None` | Anthropic | Allowlist — mutually exclusive with `blocked_domains` |
| `blocked_domains` | `None` | Anthropic | Blocklist — mutually exclusive with `allowed_domains` |
| `enable_citations` | `False` | Anthropic | **New 1.107.0** — Annotate response with inline source citations |
| `max_content_tokens` | `None` | Anthropic | **New 1.107.0** — Cap token consumption per fetch request |
| `optional` | `False` | All | Inherited from `AbstractNativeTool` |
| `kind` | `'web_fetch'` | — | Discriminator for deserialisation |

**Supported providers**: Anthropic, Google (Gemini). The `allowed_domains`/`blocked_domains` constraint is enforced server-side — not in PydanticAI's code.

```python
# 1 — Basic WebFetchTool with citation support
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebFetchTool

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[NativeTool(WebFetchTool(enable_citations=True))],
)

async def main():
    result = await agent.run('Summarise the content at https://example.com')
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 2 — Intranet-only fetcher with token budget
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebFetchTool

intranet_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        NativeTool(WebFetchTool(
            allowed_domains=['internal.corp.example.com', 'docs.corp.example.com'],
            max_content_tokens=4096,  # Cap each fetch at 4k tokens
            max_uses=3,               # Fetch at most 3 URLs per run
            enable_citations=True,
        ))
    ],
)

async def main():
    result = await intranet_agent.run(
        'What does our API documentation say about authentication?'
    )
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 3 — Domain blocklist with citations for research fetching
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebFetchTool

# Provider-side domain blocking — note: SSRF protection (safe_download) applies
# to PydanticAI's local fetch tools, not provider-native WebFetchTool.
research_agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        NativeTool(WebFetchTool(
            blocked_domains=['ads.example.com', 'tracker.example.org'],
            enable_citations=True,
            max_content_tokens=8192,
        ))
    ],
    system_prompt='You are a research assistant. Always cite your sources.',
)

async def main():
    result = await research_agent.run(
        'Compare the approaches described at https://peps.python.org/pep-0634/ and '
        'https://peps.python.org/pep-0636/'
    )
    print(result.output)

asyncio.run(main())
```

---

## 7 · `MCPServerTool` 1.107.0 complete — Native MCP server with headers

**Module**: `pydantic_ai.native_tools`  
**Exported as**: `pydantic_ai.MCPServerTool`

`MCPServerTool` is a native tool that lets the model call an **external MCP server** server-side without PydanticAI proxying the calls. Version 1.107.0 added the `headers` field for custom HTTP headers (authentication, tracing, etc.).

### Key behaviours (source-verified)

| Field | Required | Provider support | Notes |
|---|---|---|---|
| `id` | Yes | All | Unique identifier — used in `unique_id` property as `f'mcp_server:{id}'` |
| `url` | Yes | All | MCP server URL. For OpenAI: prefix `x-openai-connector:<connector_id>` to use a stored connector |
| `authorization_token` | No | OpenAI, Anthropic, xAI | Bearer token for the `Authorization` header |
| `description` | No | OpenAI, xAI | Human-readable description of this MCP server |
| `allowed_tools` | No | OpenAI, Anthropic, xAI | Allowlist of tools this server may expose |
| `headers` | No | OpenAI, xAI | **New 1.107.0** — `dict[str, str]` arbitrary HTTP headers |
| `optional` | `False` | All | Inherited from `AbstractNativeTool` |
| `kind` | `'mcp_server'` | — | Discriminator |

**`unique_id` property**: Returns `f'mcp_server:{self.id}'` — allows multiple `MCPServerTool` instances in one agent without ID collision.

**`label` property**: Returns `f'MCP: {self.id}'` — used in UI display.

**`connector_id` via URL**: For OpenAI, pass `url='x-openai-connector:conn_abc123'` to reference a stored connector instead of a live URL.

```python
# 1 — Basic MCPServerTool with authorization
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import MCPServerTool

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        NativeTool(MCPServerTool(
            id='github-mcp',
            url='https://mcp.github.com/',
            authorization_token='ghp_YOUR_TOKEN_HERE',
            allowed_tools=['list_repos', 'create_issue', 'search_code'],
            description='GitHub MCP server for repository management',
        ))
    ],
)

async def main():
    result = await agent.run('List my GitHub repositories and find any open issues')
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 2 — MCPServerTool with custom headers for tracing and auth
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import MCPServerTool
import os

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        NativeTool(MCPServerTool(
            id='internal-tools',
            url='https://tools.corp.example.com/mcp',
            headers={
                'X-API-Key': os.environ.get('INTERNAL_API_KEY', 'key'),
                'X-Trace-ID': 'session-001',
                'X-Team': 'platform-engineering',
            },
            allowed_tools=['deploy', 'rollback', 'get_metrics'],
        ))
    ],
)

async def main():
    result = await agent.run('What are the current error rates for the API service?')
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 3 — Multiple MCPServerTools with connector_id for stored OpenAI connectors
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import MCPServerTool

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        NativeTool(MCPServerTool(
            id='slack',
            url='x-openai-connector:conn_slack_abc123',  # stored connector
            allowed_tools=['send_message', 'list_channels'],
        )),
        NativeTool(MCPServerTool(
            id='jira',
            url='x-openai-connector:conn_jira_def456',
            allowed_tools=['create_ticket', 'search_issues'],
        )),
    ],
)

async def main():
    result = await agent.run(
        'Create a Jira ticket for the login bug and notify the team in #platform Slack'
    )
    print(result.output)

import asyncio
asyncio.run(main())
```

---

## 8 · `FileSearchTool` multi-provider — Vector search across OpenAI, Google, and xAI

**Module**: `pydantic_ai.native_tools`  
**Exported as**: `pydantic_ai.FileSearchTool`

`FileSearchTool` is a native RAG (Retrieval-Augmented Generation) tool. Providers manage file storage, chunking, embedding, and retrieval internally. The same `FileSearchTool` class maps `file_store_ids` differently per provider.

### Key behaviours (source-verified)

| Field | Required | Notes |
|---|---|---|
| `file_store_ids` | Yes | `Sequence[str]` — semantics depend on provider (see table) |
| `optional` | `False` | Inherited from `AbstractNativeTool` |
| `kind` | `'file_search'` | Discriminator |

**Provider mapping of `file_store_ids`**:

| Provider | What `file_store_ids` maps to | How to create |
|---|---|---|
| OpenAI Responses | Vector store IDs (e.g. `vs_abc123`) | `client.vector_stores.create(...)` |
| Google (Gemini) | File search store names created via the Gemini Files API | Gemini Files API — upload files to a named corpus |
| xAI | Collection IDs for xAI collections search | xAI Collections API |

**Note**: `FileSearchTool` is in `NATIVE_TOOLS_REQUIRING_CONFIG` — it cannot be used without providing `file_store_ids` (the store must exist before the agent runs).

```python
# 1 — OpenAI vector store search
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import FileSearchTool

# Assumes a vector store has been created and files uploaded via the OpenAI API
agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[NativeTool(FileSearchTool(file_store_ids=['vs_abc123xyz']))],
    system_prompt='You are a document Q&A assistant. Use the file search tool to find relevant information.',
)

async def main():
    result = await agent.run('What does our employee handbook say about remote work policies?')
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 2 — Multi-store search across different knowledge bases
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import FileSearchTool

# Search across multiple vector stores simultaneously
agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        NativeTool(FileSearchTool(
            file_store_ids=[
                'vs_technical_docs',
                'vs_customer_feedback',
                'vs_internal_wiki',
            ]
        ))
    ],
)

async def main():
    result = await agent.run(
        'Summarise all information we have about the payment integration issues '
        'reported in Q4 2024 — check documentation, feedback, and wiki.'
    )
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 3 — xAI collections search
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import FileSearchTool

agent = Agent(
    'xai:grok-3',
    capabilities=[
        NativeTool(FileSearchTool(
            file_store_ids=['col_research_papers_ml', 'col_internal_reports'],
        ))
    ],
    system_prompt='Search across our research paper collection and internal reports.',
)

async def main():
    result = await agent.run(
        'What are the latest findings on transformer attention efficiency from our research?'
    )
    print(result.output)

import asyncio
asyncio.run(main())
```

---

## 9 · `ToolSearchStrategy` types — Strategy selection for provider-native tool search

**Module**: `pydantic_ai.native_tools._tool_search` (re-exported from `pydantic_ai.capabilities`)  
**Exported as**: `ToolSearchNativeStrategy`, `ToolSearchLocalStrategy`, `ToolSearchFunc`, `ToolSearchStrategy`

These type aliases define the strategy options accepted by the `ToolSearch` capability's `strategy` field. They control how deferred tools (marked `defer_loading=True`) are discovered at runtime.

### Key behaviours (source-verified)

| Type | Value(s) | Behaviour |
|---|---|---|
| `ToolSearchNativeStrategy` | `Literal['bm25', 'regex']` | Provider-native strategy; both map to Anthropic server-side search algorithms. Rejected on OpenAI. |
| `ToolSearchLocalStrategy` | `Literal['keywords']` | Built-in local keyword-overlap algorithm. Use to pin to current local behaviour rather than letting Pydantic AI upgrade automatically. |
| `ToolSearchFunc` | `Callable[[RunContext, Sequence[str], Sequence[ToolDefinition]], Sequence[str] \| Awaitable[Sequence[str]]]` | Custom search function. Used locally AND as "client-executed" on Anthropic (tool-reference blocks) and OpenAI (`execution='client'`). |
| `ToolSearchStrategy` | Union of the above | Accepted by `ToolSearch.strategy`; `None` is also accepted (not in the union) to mean "provider default" |

**`TOOL_SEARCH_FUNCTION_TOOL_NAME`**: The string `'search_tools'` — the name of the local function tool that backs keyword-based discovery and is used as the wire name for client-executed callable modes.

**Provider strategy support matrix**:

| Provider | `None` (default) | `'bm25'` | `'regex'` | `'keywords'` | Custom callable |
|---|---|---|---|---|---|
| Anthropic Sonnet 4.5+ | server BM25 | server BM25 | server regex | local | local + `tool_reference` blocks |
| OpenAI GPT-5.4+ | server `tool_search` | Error | Error | local | local + `execution='client'` |
| All others | local `search_tools` | Error | Error | local | local |

```python
# 1 — ToolSearch capability with explicit 'bm25' strategy on Anthropic
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[ToolSearch(strategy='bm25', max_results=5)],
)

@agent.tool_plain(defer_loading=True)
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f'Sunny in {location}'

@agent.tool_plain(defer_loading=True)
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight between two cities."""
    return f'Booked {origin}→{destination} on {date}'

async def main():
    result = await agent.run('What is the weather like in Paris?')
    print(result.output)

import asyncio
asyncio.run(main())
```

```python
# 2 — Custom ToolSearchFunc for semantic search via embeddings
import asyncio
from collections.abc import Sequence
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import ToolDefinition

async def embedding_search(
    ctx: RunContext,
    queries: Sequence[str],
    tools: Sequence[ToolDefinition],
) -> Sequence[str]:
    """Custom semantic search: score each tool by keyword overlap."""
    scores: dict[str, float] = {}
    for tool in tools:
        desc = (tool.description or '').lower()
        score = sum(
            1.0 for q in queries for word in q.lower().split()
            if word in desc
        )
        if score > 0:
            scores[tool.name] = score
    return sorted(scores, key=scores.get, reverse=True)  # type: ignore

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[ToolSearch(strategy=embedding_search, max_results=3)],
)

@agent.tool_plain(defer_loading=True)
def search_database(table: str, query: str) -> list[dict]:
    """Search a database table for rows matching a SQL-like query."""
    return []

@agent.tool_plain(defer_loading=True)
def send_notification(user_id: str, message: str) -> bool:
    """Send a push notification to a user."""
    return True

async def main():
    result = await agent.run('Search the users table for active accounts')
    print(result.output)

asyncio.run(main())
```

```python
# 3 — Local 'keywords' strategy with forced local behaviour
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

# 'keywords' pins to the local algorithm even on Anthropic/OpenAI
# Useful for reproducible offline testing
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[ToolSearch(strategy='keywords', max_results=2)],
)

@agent.tool_plain(defer_loading=True)
def calculate_tax(income: float, country: str) -> float:
    """Calculate income tax for a given country."""
    return income * 0.2

@agent.tool_plain(defer_loading=True)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert an amount from one currency to another."""
    return amount * 1.1

async def main():
    result = await agent.run('Calculate my tax for $80000 income in the US')
    print(result.output)

asyncio.run(main())
```

---

## 10 · `ToolSearchTool` — Framework-internal native tool for provider-side tool search

**Module**: `pydantic_ai.native_tools._tool_search`  
**Exported as**: *internal — not re-exported from `pydantic_ai`*

`ToolSearchTool` is the `AbstractNativeTool` subclass that the `ToolSearch` capability injects into `ModelRequestParameters.native_tools`. It is **never constructed directly by user code** — users interact with it exclusively through the `ToolSearch` capability. Understanding its internals helps diagnose provider adapter behaviour.

<Aside type="caution">
`ToolSearchTool` is excluded from `TestModel.supported_native_tools()`. When using `TestModel`, the `ToolSearch` capability automatically falls back to its local `search_tools` function tool — so deferred tools still work in tests without needing provider-native tool search.
</Aside>

### Key behaviours (source-verified)

| Field | Type | Default | Notes |
|---|---|---|---|
| `strategy` | `Literal['bm25', 'regex', 'custom'] \| None` | `None` | Extended vs `ToolSearchStrategy`: adds `'custom'` for callable dispatch. Users don't pass `'custom'` directly. |
| `kind` | `str` | `'tool_search'` | Registered in `NATIVE_TOOL_TYPES` on class definition via `__init_subclass__` |
| `optional` | `bool` | `False` | Inherited from `AbstractNativeTool` |

**`strategy` values (internal)**:
- `None`: provider default (BM25 on Anthropic, `tool_search` on OpenAI)
- `'bm25'`: force Anthropic server-side BM25; error on OpenAI
- `'regex'`: force Anthropic server-side regex; error on OpenAI
- `'custom'`: set by `ToolSearch` when `strategy` is a callable; triggers "client-executed" native surface (Anthropic `tool_reference` blocks; OpenAI `execution='client'`)

**`NATIVE_TOOLS_REQUIRING_CONFIG`** contains `ToolSearchTool` — meaning it cannot appear in a model request without the `ToolSearch` capability to configure it.

**`ToolSearch` capability creates `ToolSearchTool`**:
- If provider supports native tool search AND `strategy != 'keywords'` → injects `ToolSearchTool` into native tools
- If `strategy` is a callable → sets `ToolSearchTool.strategy = 'custom'` and registers the callable
- If provider doesn't support native tool search → uses local `search_tools` function tool instead

```python
# 1 — Verify ToolSearchTool is injected by ToolSearch capability
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[ToolSearch(strategy='bm25')],
)

@agent.tool_plain(defer_loading=True)
def lookup(key: str) -> str:
    """Look up a value by key."""
    return f'value:{key}'

# TestModel excludes ToolSearchTool so it falls back to local search
async def test_with_test_model():
    model = TestModel(custom_output_text='Found it')
    result = await agent.run('Look up the value for "config"', model=model)
    # ToolSearchTool is NOT in native_tools for TestModel
    params = model.last_model_request_parameters
    for nt in params.native_tools:
        print(f'native tool: {nt.kind}')  # Should not include 'tool_search'
    print(result.output)

asyncio.run(test_with_test_model())
```

```python
# 2 — Inspect ToolSearchTool details via a custom FunctionModel
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.native_tools._tool_search import ToolSearchTool

captured_native_tools = []

def inspector_model(messages, info: AgentInfo) -> ModelResponse:
    for nt in info.model_request_parameters.native_tools:
        captured_native_tools.append(nt)
    return ModelResponse(parts=[TextPart('inspected')])

agent = Agent(
    FunctionModel(inspector_model),
    capabilities=[ToolSearch()],
)

@agent.tool_plain(defer_loading=True)
def do_thing(x: str) -> str:
    """Do something with x."""
    return x

async def main():
    await agent.run('test')
    for nt in captured_native_tools:
        print(f'kind={nt.kind}', end='')
        if isinstance(nt, ToolSearchTool):
            print(f', strategy={nt.strategy}', end='')
        print()

asyncio.run(main())
```

```python
# 3 — ToolSearch strategy=None vs explicit bm25 effect on ToolSearchTool
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.native_tools._tool_search import ToolSearchTool

def make_agent(strategy):
    captured = []

    def model(messages, info: AgentInfo) -> ModelResponse:
        for nt in info.model_request_parameters.native_tools:
            if isinstance(nt, ToolSearchTool):
                captured.append(nt.strategy)
        return ModelResponse(parts=[TextPart('ok')])

    ag = Agent(FunctionModel(model), capabilities=[ToolSearch(strategy=strategy)])

    @ag.tool_plain(defer_loading=True)
    def tool_a(x: str) -> str:
        """Tool A."""
        return x

    return ag, captured

async def main():
    for strategy in [None, 'bm25', 'regex']:
        ag, captured = make_agent(strategy)
        await ag.run('test')
        print(f'strategy={strategy!r} → ToolSearchTool.strategy={captured}')

asyncio.run(main())
```

---

<Aside type="tip">
**Testing workflow**: Use `TestModel` for fast deterministic unit tests, `FunctionModel` when you need precise control over model responses (simulating errors, specific tool call sequences, structured outputs). Both support all function tools without requiring API credentials.
</Aside>

## Quick navigation

| Class | Module | Prior coverage |
|---|---|---|
| `TestModel` | `pydantic_ai.models.test` | New in this volume |
| `TestStreamedResponse` | `pydantic_ai.models.test` | New in this volume |
| `FunctionModel` | `pydantic_ai.models.function` | New in this volume |
| `AgentInfo` | `pydantic_ai.models.function` | New in this volume |
| `DeltaToolCall` | `pydantic_ai.models.function` | New in this volume |
| `DeltaThinkingPart` | `pydantic_ai.models.function` | New in this volume |
| `FunctionStreamedResponse` | `pydantic_ai.models.function` | New in this volume |
| `ImageGenerationTool` | `pydantic_ai.native_tools` | New 1.107.0 fields: `input_fidelity`, `partial_images`, Google sizes, `aspect_ratio` |
| `WebFetchTool` | `pydantic_ai.native_tools` | New 1.107.0 fields: `max_content_tokens`, `enable_citations` |
| `MCPServerTool` | `pydantic_ai.native_tools` | New 1.107.0 field: `headers`; `connector_id` URL prefix |
| `FileSearchTool` | `pydantic_ai.native_tools` | New 1.107.0: xAI collections support |
| `ToolSearchNativeStrategy` | `pydantic_ai.capabilities` | New in this volume |
| `ToolSearchLocalStrategy` | `pydantic_ai.capabilities` | New in this volume |
| `ToolSearchFunc` | `pydantic_ai.capabilities` | New in this volume |
| `ToolSearchStrategy` | `pydantic_ai.capabilities` | New in this volume |
| `ToolSearchTool` | `pydantic_ai.native_tools._tool_search` | New in this volume (framework-internal) |

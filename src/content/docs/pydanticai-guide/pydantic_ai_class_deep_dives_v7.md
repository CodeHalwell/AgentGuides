---
title: "PydanticAI — Class Deep Dives Vol. 7"
description: "Source-verified deep dives into 10 PydanticAI classes: AgentEventStream/AgentRunResultEvent (event streaming), ThinkingPart/ThinkingPartDelta (extended thinking), AudioUrl/VideoUrl/DocumentUrl (multimodal URLs), OutputContext (output hooks), ModelRetry (retry mechanics), RequestUsage (per-call usage), WebSearchTool/WebSearchUserLocation (localized web search), MemoryTool, CodeExecutionTool, AbstractNativeTool (custom native tools)."
sidebar:
  label: "Class deep dives (Vol. 7)"
  order: 27
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.104.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.104.0`.
</Aside>

Ten class groups from the `pydantic_ai` 1.104.0 source covering the structured event-stream API,
extended thinking response parts, multimodal URL types for audio/video/documents, the output hook
context passed to lifecycle callbacks, the retry exception with Pydantic serialisation, per-request
usage tracking, the full web-search configuration including user location, the native memory and
code-execution tools, and the abstract native tool base class for building custom provider-native tools.

---

## 1. `AgentEventStream` + `AgentRunResultEvent` — Structured Event Streaming

**Module:** `pydantic_ai.result` / `pydantic_ai.run`  
**Import:** `from pydantic_ai import AgentEventStream`

`AgentEventStream` is the context-manager handle returned by `agent.run_stream_events()`. It wraps
an async generator of `AgentStreamEvent | AgentRunResultEvent` objects and guarantees cleanup via
`__aexit__` regardless of whether iteration completes normally or is interrupted.

### Class signature

```python
class AgentEventStream(Generic[OutputDataT]):
    def __init__(
        self,
        generator: AsyncGenerator[AgentStreamEvent | AgentRunResultEvent[Any], None],
    ) -> None: ...

    async def __aenter__(self) -> AgentEventStream[OutputDataT]: ...
    async def __aexit__(self, ...) -> bool: ...
    def __aiter__(self) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[OutputDataT]]: ...
    async def aclose(self) -> None: ...
```

### The event taxonomy

Every event emitted by the stream is discriminated by its `event_kind` field:

| Event class | `event_kind` | Meaning |
|---|---|---|
| `PartStartEvent` | `'part_start'` | A new `ModelResponsePart` has begun |
| `PartDeltaEvent` | `'part_delta'` | Incremental update to an in-progress part |
| `PartEndEvent` | `'part_end'` | A part is complete |
| `FunctionToolCallEvent` | `'function_tool_call'` | A function tool is being called |
| `FunctionToolResultEvent` | `'function_tool_result'` | A function tool returned a result |
| `BuiltinToolCallEvent` | `'builtin_tool_call'` | A native/built-in tool call |
| `BuiltinToolResultEvent` | `'builtin_tool_result'` | A native/built-in tool result |
| `OutputToolCallEvent` | `'output_tool_call'` | An output tool call (structured output via tool) |
| `OutputToolResultEvent` | `'output_tool_result'` | An output tool result |
| `FinalResultEvent` | `'final_result'` | Final agent output (streaming result) |
| `AgentRunResultEvent` | `'agent_run_result'` | Terminal event, carries the complete `AgentRunResult` |

### Basic usage — always use `async with`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, PartEndEvent

agent = Agent('openai:gpt-4o-mini', instructions='You are a concise assistant.')

async def stream_with_events() -> str:
    chunks: list[str] = []
    async with agent.run_stream_events('What is 2 + 2?') as stream:
        async for event in stream:
            if isinstance(event, PartDeltaEvent):
                # TextPartDelta has content_delta; only accumulate text deltas
                delta = event.delta
                if hasattr(delta, 'content_delta') and delta.content_delta:
                    chunks.append(delta.content_delta)
    return ''.join(chunks)

print(asyncio.run(stream_with_events()))
```

### `AgentRunResultEvent` — the terminal event

`AgentRunResultEvent` is always the **last** event emitted. It carries the completed
`AgentRunResult` so you can inspect usage, message history, and the final output without
a separate call.

```python
from pydantic_ai.run import AgentRunResultEvent

async def full_event_loop() -> None:
    async with agent.run_stream_events('Name the planets.') as stream:
        async for event in stream:
            match event.event_kind:
                case 'part_start':
                    print(f'[START] index={event.index} kind={event.part.part_kind}')
                case 'part_delta':
                    delta = event.delta
                    if hasattr(delta, 'content_delta') and delta.content_delta:
                        print(delta.content_delta, end='', flush=True)
                case 'part_end':
                    print(f'\n[END] index={event.index}')
                case 'function_tool_call':
                    print(f'[TOOL CALL] {event.part.tool_name}')
                case 'function_tool_result':
                    print(f'[TOOL RESULT] {event.part.content}')
                case 'agent_run_result':
                    result = event.result          # AgentRunResult
                    print(f'\nUsage: {result.usage()}')
                    print(f'Output: {result.output}')

asyncio.run(full_event_loop())
```

### `PartStartEvent` — UI grouping with `previous_part_kind`

`PartStartEvent` carries the `previous_part_kind` field so UI code knows whether to open
a new section or continue an existing one:

```python
from pydantic_ai.messages import PartStartEvent

async def ui_stream() -> None:
    in_thinking_block = False
    async with agent.run_stream_events('Think step-by-step: solve 5! + 3!') as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent):
                if event.part.part_kind == 'thinking':
                    in_thinking_block = True
                    print('<think>')
                elif event.part.part_kind == 'text' and in_thinking_block:
                    in_thinking_block = False
                    print('</think>')
```

### Iterating without `async with` — deprecated

Direct `async for event in stream:` iteration (without `async with`) is deprecated since
1.95.0 and will be removed in v2. The context-manager form guarantees `aclose()` runs on
every exit path including exceptions and `break`.

```python
# DEPRECATED — do not use in new code
async for event in agent.run_stream_events('Hello'):  # DeprecationWarning
    ...

# CORRECT
async with agent.run_stream_events('Hello') as stream:
    async for event in stream:
        ...
```

---

## 2. `ThinkingPart` + `ThinkingPartDelta` — Extended Thinking Response Parts

**Module:** `pydantic_ai.messages`  
**Import:** `from pydantic_ai.messages import ThinkingPart, ThinkingPartDelta`

`ThinkingPart` represents a chain-of-thought reasoning block returned by a model before (or
alongside) its final answer. It appears as a part inside `ModelResponse.parts` alongside
`TextPart` and `ToolCallPart`.

### Class signatures

```python
@dataclass(repr=False)
class ThinkingPart:
    content: str
    id: str | None = None              # Provider-issued ID (required when signature is set)
    signature: str | None = None       # Provider-specific opaque token for round-trips
    provider_name: str | None = None   # Required when id/signature/provider_details is set
    provider_details: dict[str, Any] | None = None
    part_kind: Literal['thinking'] = 'thinking'

    def has_content(self) -> bool: ...

@dataclass(repr=False, kw_only=True)
class ThinkingPartDelta:
    content_delta: str | None = None
    signature_delta: str | None = None
    provider_name: str | None = None
    provider_details: ProviderDetailsDelta = None
    part_delta_kind: Literal['thinking'] = 'thinking'

    def apply(self, part: ModelResponsePart | ThinkingPartDelta) -> ThinkingPart | ThinkingPartDelta: ...
```

### Provider signature semantics

The `signature` field (and its streaming counterpart `signature_delta`) is an **opaque token**
that some providers require you to return verbatim in subsequent turns. The `provider_name`
field tells PydanticAI which provider to route it back to.

| Provider | Field name | Must round-trip? |
|---|---|---|
| Anthropic | `signature` | Yes — required in `extended_thinking` |
| Bedrock | `signature` | Yes |
| Google | `thought_signature` → `signature` | Yes (Gemini 2.0+) |
| OpenAI | `encrypted_content` → `signature` | Yes (o3/o4 models) |

### Reading thinking parts from a completed run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ThinkingPart, TextPart

think_agent = Agent(
    'anthropic:claude-opus-4-5',
    instructions='Think carefully before answering.',
)

async def show_thinking() -> None:
    result = await think_agent.run('What are the first 5 Fibonacci numbers?')
    for msg in result.all_messages():
        for part in getattr(msg, 'parts', []):
            if isinstance(part, ThinkingPart):
                print(f'[THINKING] {part.content[:120]}...')
                if part.signature:
                    print(f'  signature present ({len(part.signature)} chars)')
            elif isinstance(part, TextPart):
                print(f'[ANSWER] {part.content}')

asyncio.run(show_thinking())
```

### Capturing thinking deltas in a streaming run

```python
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, ThinkingPartDelta

async def stream_thinking() -> None:
    thinking_buf: list[str] = []
    answer_buf: list[str] = []
    current_kind: str | None = None

    async with think_agent.run_stream_events('Solve: x² - 5x + 6 = 0') as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent):
                current_kind = event.part.part_kind
            elif isinstance(event, PartDeltaEvent):
                delta = event.delta
                if isinstance(delta, ThinkingPartDelta) and delta.content_delta:
                    thinking_buf.append(delta.content_delta)
                elif hasattr(delta, 'content_delta') and delta.content_delta:
                    answer_buf.append(delta.content_delta)

    print('THINKING:', ''.join(thinking_buf)[:200])
    print('ANSWER:', ''.join(answer_buf))

asyncio.run(stream_thinking())
```

### `ThinkingPartDelta.apply()` — building the full part incrementally

```python
from pydantic_ai.messages import ThinkingPart, ThinkingPartDelta

# Start with an initial partial part
part: ThinkingPart | ThinkingPartDelta = ThinkingPartDelta(
    content_delta='Let me think step by step. ',
)

deltas = [
    ThinkingPartDelta(content_delta='First, I consider the inputs. '),
    ThinkingPartDelta(content_delta='Then I derive the answer.', signature_delta='sig-abc'),
]

current: ThinkingPart | ThinkingPartDelta = part
for d in deltas:
    current = d.apply(current)     # Returns ThinkingPart once all content is accumulated

assert isinstance(current, ThinkingPart)
print(current.content)     # Full thinking text
print(current.signature)   # 'sig-abc'
```

---

## 3. `AudioUrl` + `VideoUrl` + `DocumentUrl` — Multimodal URL Types

**Module:** `pydantic_ai.messages`  
**Import:** `from pydantic_ai import AudioUrl, VideoUrl, DocumentUrl`  
*(also: `from pydantic_ai.messages import AudioUrl, VideoUrl, DocumentUrl`)*

These three classes extend the abstract `FileUrl` base and represent URLs pointing to
audio files, video files, and documents respectively. They appear inside `UserPromptPart`
as part of the `MultiModalContent` union.

### Class signatures (condensed from source)

```python
class FileUrl(ABC):
    url: str
    force_download: ForceDownloadMode = False   # False | True | 'allow-local'
    vendor_metadata: dict[str, Any] | None = None
    # Private, aliased:
    media_type: str | None = None   # override inferred MIME type
    identifier: str | None = None   # opaque provider file reference

class AudioUrl(FileUrl):
    kind: Literal['audio-url'] = 'audio-url'
    @property
    def format(self) -> AudioFormat: ...       # inferred from media_type

class VideoUrl(FileUrl):
    kind: Literal['video-url'] = 'video-url'
    @property
    def is_youtube(self) -> bool: ...          # True for youtu.be / youtube.com
    @property
    def format(self) -> VideoFormat: ...

class DocumentUrl(FileUrl):
    kind: Literal['document-url'] = 'document-url'
    @property
    def format(self) -> DocumentFormat: ...    # used by Bedrock Converse API
```

### `force_download` — SSRF protection levels

| Value | Behaviour |
|---|---|
| `False` (default) | Provider-native inline URL if supported; fallback downloads block private IPs + cloud metadata |
| `True` | Always download; blocks private IPs + cloud metadata |
| `'allow-local'` | Always download; allows private IPs but still blocks cloud metadata |

### `AudioUrl` — analysing a meeting recording

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import AudioUrl

audio_agent = Agent('google:gemini-2.0-flash', instructions='Summarise audio content.')

async def transcribe_meeting(url: str) -> str:
    result = await audio_agent.run(
        [
            'Summarise this meeting recording and extract action items.',
            AudioUrl(url=url, media_type='audio/mpeg'),
        ]
    )
    return result.output

# asyncio.run(transcribe_meeting('https://example.com/meeting.mp3'))
```

### `AudioUrl` — explicit media type vs inference

```python
from pydantic_ai.messages import AudioUrl

# Media type inferred from extension
wav = AudioUrl(url='https://example.com/speech.wav')
print(wav.media_type)   # 'audio/wav'  (inferred)

# Explicit override — useful when URL has no extension
stream = AudioUrl(
    url='https://api.example.com/audio/stream/12345',
    media_type='audio/ogg',
)
print(stream.media_type)   # 'audio/ogg'
```

### `VideoUrl` — YouTube and local video

```python
from pydantic_ai import Agent
from pydantic_ai.messages import VideoUrl

video_agent = Agent('google:gemini-2.0-flash', instructions='Analyse video content.')

# YouTube URLs are automatically detected
yt = VideoUrl(url='https://www.youtube.com/watch?v=dQw4w9WgXcQ')
print(yt.is_youtube)    # True
print(yt.media_type)    # 'video/mp4'  (always for YouTube)

# Google-specific vendor metadata for custom frame sampling
custom = VideoUrl(
    url='https://storage.googleapis.com/example/clip.mp4',
    vendor_metadata={
        'fps': 2,           # Google: video_metadata.fps
        'start_offset': 10, # Google: video_metadata.start_offset_sec
    },
)
```

### `DocumentUrl` — PDF analysis

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import DocumentUrl

doc_agent = Agent('anthropic:claude-opus-4-5', instructions='Extract key information.')

async def analyse_pdf(pdf_url: str) -> str:
    result = await doc_agent.run(
        [
            'What are the main findings in this report?',
            DocumentUrl(url=pdf_url, media_type='application/pdf'),
        ]
    )
    return result.output

# Multiple documents in one turn
async def compare_docs(url_a: str, url_b: str) -> str:
    result = await doc_agent.run(
        [
            'Compare the key differences between these two documents:',
            DocumentUrl(url=url_a, media_type='application/pdf'),
            DocumentUrl(url=url_b, media_type='application/pdf'),
        ]
    )
    return result.output
```

### `vendor_metadata` — provider-specific extensions

| Provider | Class | Field | Effect |
|---|---|---|---|
| Google | `VideoUrl` | `vendor_metadata` | Passed as `video_metadata` (fps, start/end offset) |
| OpenAI / xAI | `ImageUrl` | `vendor_metadata['detail']` | Image detail level (`'low'`/`'high'`/`'auto'`) |

### Combining media types in one prompt

```python
from pydantic_ai.messages import AudioUrl, VideoUrl, DocumentUrl

# All three in a single user turn (multi-modal)
prompt = [
    'Review this meeting: compare the slides with the recording and audio transcript.',
    VideoUrl(url='https://example.com/recording.mp4'),
    AudioUrl(url='https://example.com/transcript.mp3'),
    DocumentUrl(url='https://example.com/slides.pdf'),
]
```

---

## 4. `OutputContext` — Output Hook Context

**Module:** `pydantic_ai.output`  
**Import:** `from pydantic_ai.output import OutputContext`

`OutputContext` is the read-only context object passed to all four output lifecycle hooks
(`before_output_validate`, `after_output_validate`, `before_output_process`, and
`after_output_process`). It tells your hook *what kind of output* the agent is processing
in this step.

### Class signature

```python
@dataclass
class OutputContext:
    mode: OutputMode          # 'text'|'tool'|'native'|'prompted'|'tool_or_text'|'image'|'auto'
    output_type: type[Any] | None
    object_def: OutputObjectDefinition | None
    has_function: bool
    function_name: str | None = None
    tool_call: ToolCallPart | None = None
    tool_def: ToolDefinition | None = None
    allows_text: bool = False
    allows_image: bool = False
    allows_deferred_tools: bool = False
```

### Field semantics

| Field | What it tells you |
|---|---|
| `mode` | The *schema's* output mode — use it to branch between text and structured |
| `output_type` | The Python type the agent is expecting (e.g. `MyModel`, `str`) |
| `object_def` | Full schema + name + description for structured output types |
| `has_function` | Whether an output *function* will be called in the execute step |
| `function_name` | The function's name if known; `None` for union processors |
| `tool_call` | The raw `ToolCallPart` when output arrived via a tool call |
| `tool_def` | The tool's `ToolDefinition` when output arrived via a tool call |
| `allows_text` | `True` when the schema can also accept plain text |
| `allows_image` | `True` when the schema can accept image output |
| `allows_deferred_tools` | `True` when the schema accepts deferred tool requests |

### Registering output hooks via `Hooks`

```python
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import RunContext
from pydantic_ai.capabilities.hooks import Hooks

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]

@dataclass
class Deps:
    user_id: str

hooks = Hooks()

@hooks.before_output_validate
def log_output_mode(
    ctx: RunContext[Deps],
    *,
    output_context: OutputContext,
    output: object,
) -> object:
    print(f'[VALIDATE] mode={output_context.mode} type={output_context.output_type}')
    return output   # pass through unchanged

@hooks.after_output_process
def audit_output(
    ctx: RunContext[Deps],
    *,
    output_context: OutputContext,
    output: object,
) -> object:
    print(f'[AUDIT] user={ctx.deps.user_id} has_func={output_context.has_function}')
    return output

agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Recipe,
    deps_type=Deps,
    capabilities=[hooks],
)

async def main() -> None:
    result = await agent.run('Give me a simple pasta recipe.', deps=Deps(user_id='user-42'))
    print(result.output.name)

asyncio.run(main())
```

### Using `output_context.mode` to branch validation

```python
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import RunContext
from pydantic_ai.capabilities.hooks import Hooks

guard_hooks = Hooks()

@guard_hooks.before_output_validate
def enforce_schema(
    ctx: RunContext,
    *,
    output_context: OutputContext,
    output: object,
) -> object:
    if output_context.mode == 'text':
        # Plain text — apply basic length guard
        if isinstance(output, str) and len(output) > 10_000:
            raise ValueError('Response too long — truncating is not allowed.')
    elif output_context.mode in ('tool', 'native', 'prompted'):
        # Structured output — check the type name is allowed
        allowed = {'Recipe', 'Summary', 'Report'}
        type_name = output_context.output_type.__name__ if output_context.output_type else None
        if type_name not in allowed:
            raise ValueError(f'Output type {type_name!r} is not in the allowlist.')
    return output
```

### Inspecting `tool_call` for structured tool outputs

```python
@guard_hooks.after_output_validate
def inspect_tool_call(
    ctx: RunContext,
    *,
    output_context: OutputContext,
    output: object,
) -> object:
    if output_context.tool_call is not None:
        # The model produced a tool call to encode the structured output
        tc = output_context.tool_call
        print(f'  tool_name={tc.tool_name!r}  call_id={tc.tool_call_id!r}')
        td = output_context.tool_def
        if td:
            print(f'  schema keys: {list(td.parameters_json_schema.get("properties", {}).keys())}')
    return output
```

---

## 5. `ModelRetry` — Tool Retry Mechanics

**Module:** `pydantic_ai.tools` (re-exported from `pydantic_ai`)  
**Import:** `from pydantic_ai import ModelRetry`

`ModelRetry` is the single exception that signals "try again". Raising it from a tool, output
validator, or capability hook causes PydanticAI to return the `message` to the model as a
`RetryPromptPart`, prompting a corrected attempt up to `max_retries` times.

### Class signature (with Pydantic serialisation schema)

```python
class ModelRetry(Exception):
    message: str

    def __init__(self, message: str): ...
    def __eq__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...

    @classmethod
    def __get_pydantic_core_schema__(cls, ...) -> core_schema.CoreSchema:
        # Serialises to {'message': str, 'kind': 'model-retry'}
        # Deserialises back to ModelRetry(message=...)
```

`ModelRetry` is fully Pydantic-serialisable, which matters when you store message histories
containing `RetryPromptPart` in a database.

### Raising from a tool function

```python
import asyncio
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-4o-mini')

@agent.tool
def get_stock_price(ctx: RunContext, ticker: str) -> dict:
    """Get the current price for a stock ticker."""
    ticker = ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 5:
        raise ModelRetry(
            f'Invalid ticker {ticker!r}. Tickers are 1-5 alphabetic characters (e.g. AAPL, MSFT).'
        )
    # Simulate a lookup
    prices = {'AAPL': 189.50, 'MSFT': 415.20, 'GOOG': 178.00}
    if ticker not in prices:
        raise ModelRetry(
            f'Ticker {ticker!r} not found. Known tickers: {", ".join(prices)}. '
            f'Please try one of those.'
        )
    return {'ticker': ticker, 'price': prices[ticker], 'currency': 'USD'}

result = agent.run_sync('What is the price of apple?')
print(result.output)
```

### `ModelRetry` in output validators

Output validators (`output_validator`) can also raise `ModelRetry` to force a re-generation:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

class SafeEmail(BaseModel):
    subject: str
    body: str

agent = Agent('openai:gpt-4o-mini', output_type=SafeEmail)

@agent.output_validator
def check_length(ctx: RunContext, output: SafeEmail) -> SafeEmail:
    if len(output.body) < 50:
        raise ModelRetry(
            f'Email body is too short ({len(output.body)} chars). '
            'Please write at least 50 characters.'
        )
    return output
```

### `ModelRetry` in capability hooks

```python
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.output import OutputContext
from pydantic_ai import ModelRetry

safety_hooks = Hooks()

@safety_hooks.after_output_validate
def require_complete_sentences(
    ctx: RunContext,
    *,
    output_context: OutputContext,
    output: object,
) -> object:
    if output_context.mode == 'text' and isinstance(output, str):
        if not output.strip().endswith(('.', '!', '?')):
            raise ModelRetry(
                'Your response must end with a complete sentence (period, exclamation, or question mark).'
            )
    return output
```

### Pydantic serialisation round-trip

```python
import json
from pydantic_ai import ModelRetry
import pydantic

# ModelRetry integrates with Pydantic's type system
class RetryWrapper(pydantic.BaseModel):
    error: ModelRetry

w = RetryWrapper(error=ModelRetry('Bad input'))
serialised = w.model_dump()
print(serialised)
# {'error': {'message': 'Bad input', 'kind': 'model-retry'}}

restored = RetryWrapper.model_validate(serialised)
assert restored.error.message == 'Bad input'
```

### Retry count interaction

The agent's `max_retries` parameter (default 1) controls how many `ModelRetry` raises are
tolerated per output or per tool call. After `max_retries` is exhausted, the agent raises
`UnexpectedModelBehavior`.

```python
agent = Agent(
    'openai:gpt-4o-mini',
    max_retries=3,   # allow up to 3 ModelRetry cycles per generation
)
```

---

## 6. `RequestUsage` — Per-Request Usage Tracking

**Module:** `pydantic_ai.usage`  
**Import:** `from pydantic_ai.usage import RequestUsage`

`RequestUsage` captures token usage for a **single model API call** — one HTTP request,
one `ModelResponse`. It differs from `RunUsage` (which aggregates across the entire run)
in that it exposes pricing-integration hooks and cannot safely be summed across multiple
requests for pricing purposes.

### Class signature (from `UsageBase`)

```python
@dataclass(repr=False, kw_only=True)
class UsageBase:
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    input_audio_tokens: int = 0
    cache_audio_read_tokens: int = 0
    output_audio_tokens: int = 0
    details: dict[str, int] = field(default_factory=dict)

@dataclass(repr=False, kw_only=True)
class RequestUsage(UsageBase):
    @property
    def requests(self) -> int: ...   # always 1

    def incr(self, incr_usage: RequestUsage) -> None: ...
    def __add__(self, other: RequestUsage) -> RequestUsage: ...

    @classmethod
    def extract(
        cls,
        data: Any,
        *,
        provider: str,
        provider_url: str,
        provider_fallback: str,
        api_flavor: str = 'default',
        details: dict[str, Any] | None = None,
    ) -> RequestUsage: ...
```

### Token field reference

| Field | What it counts |
|---|---|
| `input_tokens` | Prompt / input tokens sent to the model |
| `cache_write_tokens` | Tokens written to the provider's cache (Anthropic, Google) |
| `cache_read_tokens` | Tokens served from the provider's cache (cache hit) |
| `output_tokens` | Tokens generated by the model |
| `input_audio_tokens` | Audio input tokens (OpenAI audio models) |
| `cache_audio_read_tokens` | Audio tokens served from cache |
| `output_audio_tokens` | Audio tokens generated |
| `details` | Provider-specific extras (e.g. `{'reasoning_tokens': 1024}`) |

### Reading usage from a completed run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.usage import RequestUsage

agent = Agent('openai:gpt-4o-mini')

async def track_per_request_usage() -> None:
    result = await agent.run('List 5 Python tips.')

    # RunUsage — aggregated across the whole run
    run_usage = result.usage()
    print(f'Run: input={run_usage.input_tokens} output={run_usage.output_tokens} '
          f'requests={run_usage.requests}')

    # Per-request usage from message history
    for msg in result.all_messages():
        req_usage = getattr(msg, 'usage', None)
        if isinstance(req_usage, RequestUsage):
            print(f'  Request: input={req_usage.input_tokens} output={req_usage.output_tokens} '
                  f'cache_read={req_usage.cache_read_tokens}')

asyncio.run(track_per_request_usage())
```

### Cache hit rate monitoring

```python
from pydantic_ai.usage import RequestUsage

def cache_hit_rate(usage: RequestUsage) -> float:
    """Fraction of input tokens served from cache."""
    total_input = usage.input_tokens + usage.cache_read_tokens
    return usage.cache_read_tokens / total_input if total_input > 0 else 0.0

# Monitor across a conversation
async def monitor_caching(agent: Agent, turns: list[str]) -> None:
    messages = None
    for turn in turns:
        result = await agent.run(turn, message_history=messages)
        messages = result.all_messages()
        for msg in messages:
            usage = getattr(msg, 'usage', None)
            if isinstance(usage, RequestUsage):
                rate = cache_hit_rate(usage)
                print(f'  Turn {turn[:30]!r}: cache_hit={rate:.1%}')
```

### `RequestUsage.__add__` — safe within a single response

The `__add__` operator is provided for summing multiple parts of the **same** response
(e.g. combining streaming chunks). It must **not** be used to aggregate across requests
for pricing calculations — use `RunUsage.incr()` for that:

```python
from pydantic_ai.usage import RequestUsage, RunUsage

# Safe — combining two parts of the same API call
part_a = RequestUsage(input_tokens=100, output_tokens=50)
part_b = RequestUsage(cache_read_tokens=200, output_tokens=30)
combined = part_a + part_b
print(combined.input_tokens, combined.cache_read_tokens, combined.output_tokens)
# 100  200  80

# Aggregating across multiple calls — use RunUsage
run = RunUsage()
for req in [part_a, part_b]:
    run.incr(req)
print(run.requests)   # 2
```

### `details` — provider-specific extras

OpenAI's `o`-series models include reasoning token counts in `details`:

```python
from pydantic_ai.usage import RequestUsage

# Reading OpenAI reasoning token details
usage = RequestUsage(output_tokens=500, details={'reasoning_tokens': 300})
reasoning = usage.details.get('reasoning_tokens', 0)
visible_output = usage.output_tokens - reasoning
print(f'Visible: {visible_output}, Reasoning: {reasoning}')
```

---

## 7. `WebSearchTool` + `WebSearchUserLocation` — Localized Native Web Search

**Module:** `pydantic_ai.native_tools`  
**Import:** `from pydantic_ai import WebSearchTool, WebSearchUserLocation`

`WebSearchTool` is a native tool that delegates web searches to the model's built-in search
capability (Anthropic, OpenAI Responses, Groq, Google, xAI, OpenRouter). `WebSearchUserLocation`
is a `TypedDict` that localizes results by geography.

### Class signatures

```python
@dataclass(kw_only=True)
class WebSearchTool(AbstractNativeTool):
    search_context_size: Literal['low', 'medium', 'high'] = 'medium'
    user_location: WebSearchUserLocation | None = None
    blocked_domains: list[str] | None = None
    allowed_domains: list[str] | None = None

class WebSearchUserLocation(TypedDict, total=False):
    city: str
    country: str    # 2-letter ISO for OpenAI (e.g. 'US', 'GB')
    region: str
    timezone: str
```

### Provider compatibility matrix

| Parameter | Anthropic | OpenAI Responses | Groq | Google | xAI | OpenRouter |
|---|---|---|---|---|---|---|
| `search_context_size` | — | ✓ | — | — | — | ✓ |
| `user_location` | ✓ | ✓ | — | — | — | — |
| `blocked_domains` | ✓ | ✓ | ✓ | — | ✓ | — |
| `allowed_domains` | ✓ | ✓ | — | — | ✓ | — |
| Domain mutual exclusion | ✓ | — | — | — | ✓ | — |

> **Note:** Anthropic and xAI only allow `blocked_domains` OR `allowed_domains`, not both.

### Minimal usage

```python
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool

search_agent = Agent(
    'openai:gpt-4o-mini',
    instructions='Answer questions using up-to-date web search results.',
    capabilities=[NativeTool(WebSearchTool())],
)

result = search_agent.run_sync('What is the latest version of Python?')
print(result.output)
```

### Localized search with `WebSearchUserLocation`

```python
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation
from pydantic_ai.capabilities import NativeTool

london_search = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[
        NativeTool(
            WebSearchTool(
                user_location=WebSearchUserLocation(
                    city='London',
                    country='GB',
                    region='England',
                    timezone='Europe/London',
                ),
                search_context_size='high',   # more web context retrieved (OpenAI only)
            )
        )
    ],
)

result = london_search.run_sync('What restaurants are highly rated near me right now?')
print(result.output)
```

### Domain filtering — news research agent

```python
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool

news_agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[
        NativeTool(
            WebSearchTool(
                allowed_domains=['bbc.com', 'reuters.com', 'apnews.com', 'theguardian.com'],
            )
        )
    ],
)
```

### Blocking unreliable sources

```python
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool

reliable_agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[
        NativeTool(
            WebSearchTool(
                blocked_domains=['reddit.com', 'twitter.com', 'facebook.com'],
            )
        )
    ],
)
```

### Combining `WebSearchTool` with a local fallback

For models that lack native search support, use `NativeOrLocalTool` to fall back to a
local DuckDuckGo search automatically:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.common_tools import duckduckgo_search_tool

hybrid_agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[
        NativeOrLocalTool(
            native=WebSearchTool(search_context_size='high'),
            local=duckduckgo_search_tool(),
        )
    ],
)
```

---

## 8. `MemoryTool` — Native Persistent Memory

**Module:** `pydantic_ai.native_tools`  
**Import:** `from pydantic_ai import MemoryTool`

`MemoryTool` exposes the model's built-in memory capability. When equipped, the model can
store and recall facts across conversations without explicit application-layer persistence.

### Class signature

```python
@dataclass(kw_only=True)
class MemoryTool(AbstractNativeTool):
    kind: str = 'memory'
    optional: bool = False   # inherited
```

**Currently supported by:** Anthropic only.

### Why use the native memory tool?

- Zero application-side schema: the model decides what to remember, when to store, and when to recall
- Automatic association: the model links memories to conversation context
- No separate memory API calls: reads and writes happen within the model's context window

### Basic usage

```python
from pydantic_ai import Agent, MemoryTool
from pydantic_ai.capabilities import NativeTool

memory_agent = Agent(
    'anthropic:claude-opus-4-5',
    instructions=(
        'You are a personal assistant. '
        'Remember important facts about the user and their preferences.'
    ),
    capabilities=[NativeTool(MemoryTool())],
)

# First session — introduce user preferences
result = memory_agent.run_sync(
    "I'm John, I prefer bullet-point summaries and I'm vegetarian."
)
print(result.output)

# Later session — model recalls stored facts
result2 = memory_agent.run_sync(
    "Summarise last week's AI news for me.",
    message_history=result.all_messages(),   # pass history for continuity
)
print(result2.output)  # Should use bullet points and avoid meat-related content
```

### Making `MemoryTool` optional for cross-provider agents

Set `optional=True` to silently drop the tool on models that do not support it:

```python
from pydantic_ai import Agent, MemoryTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai import FallbackModel

# MemoryTool is silently ignored on gpt-4o-mini (no native memory)
# but activates on Claude
fallback_agent = Agent(
    FallbackModel('openai:gpt-4o-mini', 'anthropic:claude-opus-4-5'),
    capabilities=[NativeTool(MemoryTool(optional=True))],
)
```

### Inspecting memory-related events in the event stream

When the model uses its memory tool, events appear as `BuiltinToolCallEvent` /
`BuiltinToolResultEvent` in the stream:

```python
from pydantic_ai.messages import BuiltinToolCallEvent, BuiltinToolResultEvent

async def watch_memory_ops() -> None:
    async with memory_agent.run_stream_events('Remember: I prefer dark mode.') as stream:
        async for event in stream:
            if isinstance(event, BuiltinToolCallEvent):
                print(f'[MEMORY OP] {event.part.tool_name}')
            elif isinstance(event, BuiltinToolResultEvent):
                print(f'[MEMORY RESULT] stored/recalled')
```

---

## 9. `CodeExecutionTool` — Native Code Execution

**Module:** `pydantic_ai.native_tools`  
**Import:** `from pydantic_ai import CodeExecutionTool`

`CodeExecutionTool` gives the model access to a sandboxed code interpreter. The model can
write and run Python (or other languages) to solve computations, transform data, draw charts,
and verify answers.

### Class signature

```python
@dataclass(kw_only=True)
class CodeExecutionTool(AbstractNativeTool):
    kind: str = 'code_execution'
    optional: bool = False   # inherited
```

**Supported by:** Anthropic, OpenAI Responses, Google, Bedrock (Nova 2.0), xAI.

### Basic data analysis agent

```python
import asyncio
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

code_agent = Agent(
    'openai:gpt-4o-mini',
    instructions=(
        'You are a data analyst. Use code execution to run calculations, '
        'generate statistics, and produce accurate results.'
    ),
    capabilities=[NativeTool(CodeExecutionTool())],
)

async def analyse_data() -> str:
    result = await code_agent.run(
        'Calculate the compound annual growth rate (CAGR) for an investment '
        'that grew from $10,000 to $18,500 over 7 years. Show your working.'
    )
    return result.output

print(asyncio.run(analyse_data()))
```

### Mathematical verification

```python
result = code_agent.run_sync(
    'Verify: is 982,451,653 a prime number? Then find the next prime after it.'
)
print(result.output)
# Model will write and execute primality-testing code to give a definitive answer
```

### Combining code execution with web search

```python
from pydantic_ai import Agent, WebSearchTool, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

analyst = Agent(
    'anthropic:claude-opus-4-5',
    instructions='Research topics online then use code to analyse the data you find.',
    capabilities=[
        NativeTool(WebSearchTool()),
        NativeTool(CodeExecutionTool()),
    ],
)

result = analyst.run_sync(
    'Find the current population of the 5 largest cities in France '
    'and calculate their combined total and average.'
)
print(result.output)
```

### File generation (OpenAI's code interpreter)

On OpenAI Responses, `CodeExecutionTool` can produce downloadable files. The model
includes file data in the response and PydanticAI surfaces it via `FilePart` in the
message parts:

```python
from pydantic_ai.messages import FilePart

async def generate_csv_report() -> None:
    result = await code_agent.run(
        'Create a CSV report of the multiplication table from 1×1 to 10×10. '
        'Return it as a downloadable file.'
    )
    for msg in result.all_messages():
        for part in getattr(msg, 'parts', []):
            if isinstance(part, FilePart):
                print(f'File: {part.id} via {part.provider_name}')
                # part.provider_details holds the raw file data
```

### Making `CodeExecutionTool` optional for multi-model setups

```python
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

multi_model_agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[
        NativeTool(CodeExecutionTool(optional=True))  # silently skipped if unsupported
    ],
)
```

---

## 10. `AbstractNativeTool` — The Native Tool Base Class

**Module:** `pydantic_ai.native_tools`  
**Import:** `from pydantic_ai.native_tools import AbstractNativeTool`

`AbstractNativeTool` is the abstract dataclass from which all eight built-in native tools
(`WebSearchTool`, `WebFetchTool`, `CodeExecutionTool`, `MemoryTool`, `ImageGenerationTool`,
`FileSearchTool`, `MCPServerTool`, `XSearchTool`) inherit. Understanding it is the key to
building **custom** provider-native tools.

### Class signature

```python
@dataclass(kw_only=True)
class AbstractNativeTool(ABC):
    kind: str = 'unknown_native_tool'   # discriminator field; override in subclass
    optional: bool = False              # when True: silently dropped on unsupported models

    @property
    def unique_id(self) -> str: ...     # default: self.kind; override when multiple instances needed
    @property
    def label(self) -> str: ...         # human-readable UI label

    def __init_subclass__(cls, **kwargs) -> None:
        # Auto-registers the subclass into NATIVE_TOOL_TYPES[cls.kind]
        NATIVE_TOOL_TYPES[cls.kind] = cls
```

### `NATIVE_TOOL_TYPES` registry

Every concrete `AbstractNativeTool` subclass is auto-registered in the module-level
`NATIVE_TOOL_TYPES` dictionary keyed by its `kind` string:

```python
from pydantic_ai.native_tools import NATIVE_TOOL_TYPES

print(list(NATIVE_TOOL_TYPES.keys()))
# ['web_search', 'x_search', 'code_execution', 'web_fetch',
#  'url_context', 'image_generation', 'memory', 'mcp_server',
#  'file_search', ...]
```

### The `optional` flag — graceful degradation

`optional=True` silently drops a native tool when the model doesn't support it, instead
of raising an error. Use it when you have a fallback path:

```python
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool

# Search is a best-effort enhancement — no error if model lacks it
agent = Agent(
    'openai:gpt-4o-mini',
    capabilities=[NativeTool(WebSearchTool(optional=True))],
)
```

### `unique_id` — multiple instances of the same tool

Some native tools may need to be registered with different configurations. Override
`unique_id` to distinguish them:

```python
from pydantic_ai.native_tools import AbstractNativeTool
from dataclasses import dataclass
from typing import Literal

@dataclass(kw_only=True)
class NamedSearchTool(AbstractNativeTool):
    """A web search tool tagged with a named corpus."""
    kind: str = 'web_search'
    corpus_name: str = 'default'

    @property
    def unique_id(self) -> str:
        return f'web_search:{self.corpus_name}'   # e.g. 'web_search:news'

    @property
    def label(self) -> str:
        return f'Web Search ({self.corpus_name.title()})'
```

### Building a custom native tool

Subclassing `AbstractNativeTool` and wiring it up via a custom `Model` implementation:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic_ai.native_tools import AbstractNativeTool

@dataclass(kw_only=True)
class CompanyKnowledgeBaseTool(AbstractNativeTool):
    """Search the company's internal knowledge base (custom provider feature)."""

    kind: str = 'company_kb'
    index_name: str = 'main'
    max_results: int = 5

    @property
    def label(self) -> str:
        return f'Company KB ({self.index_name})'

    def to_provider_payload(self) -> dict[str, Any]:
        """Convert to the custom API's tool spec (provider-specific)."""
        return {
            'type': 'company_knowledge_base',
            'index': self.index_name,
            'top_k': self.max_results,
        }
```

### Summary table — all built-in `AbstractNativeTool` subclasses

| Class | `kind` | Provider(s) |
|---|---|---|
| `WebSearchTool` | `'web_search'` | Anthropic, OpenAI, Groq, Google, xAI, OpenRouter |
| `WebFetchTool` | `'web_fetch'` | Anthropic, Google |
| `UrlContextTool` | `'url_context'` | Deprecated alias for `WebFetchTool` |
| `CodeExecutionTool` | `'code_execution'` | Anthropic, OpenAI, Google, Bedrock Nova 2.0, xAI |
| `MemoryTool` | `'memory'` | Anthropic |
| `ImageGenerationTool` | `'image_generation'` | OpenAI |
| `FileSearchTool` | `'file_search'` | OpenAI |
| `MCPServerTool` | `'mcp_server'` | OpenAI Responses |
| `XSearchTool` | `'x_search'` | xAI only |

---

## Capstone — Event Stream + Thinking + Usage Monitoring

The following example combines `AgentEventStream`, `ThinkingPart`/`ThinkingPartDelta`,
`RequestUsage`, and `CodeExecutionTool` to build a transparent reasoning agent that surfaces
its thinking chain, tool usage, and per-request costs:

```python
import asyncio
from dataclasses import dataclass, field
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import (
    PartStartEvent,
    PartDeltaEvent,
    ThinkingPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.run import AgentRunResultEvent

@dataclass
class RunStats:
    thinking_chars: int = 0
    tool_calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

async def transparent_agent_run(question: str) -> tuple[str, RunStats]:
    agent = Agent(
        'anthropic:claude-opus-4-5',
        capabilities=[NativeTool(CodeExecutionTool())],
    )
    stats = RunStats()
    answer_chunks: list[str] = []

    async with agent.run_stream_events(question) as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent):
                kind = event.part.part_kind
                if kind == 'thinking':
                    print('[THINKING…]', end='', flush=True)
                elif kind == 'text' and stats.thinking_chars:
                    print()  # newline after thinking block

            elif isinstance(event, PartDeltaEvent):
                d = event.delta
                if isinstance(d, ThinkingPartDelta) and d.content_delta:
                    stats.thinking_chars += len(d.content_delta)
                elif hasattr(d, 'content_delta') and d.content_delta:
                    answer_chunks.append(d.content_delta)
                    print(d.content_delta, end='', flush=True)

            elif isinstance(event, FunctionToolCallEvent):
                stats.tool_calls.append(event.part.tool_name)
                print(f'\n[TOOL] calling {event.part.tool_name}')

            elif isinstance(event, AgentRunResultEvent):
                usage = event.result.usage()
                stats.input_tokens = usage.input_tokens
                stats.output_tokens = usage.output_tokens
                stats.cache_read_tokens = usage.cache_read_tokens

    print('\n')
    return ''.join(answer_chunks), stats

async def main() -> None:
    answer, stats = await transparent_agent_run(
        'What is the 10,000th Fibonacci number modulo 1,000,000,007?'
    )
    print(f'\n--- Stats ---')
    print(f'Thinking: {stats.thinking_chars:,} chars')
    print(f'Tool calls: {stats.tool_calls}')
    print(f'Tokens: in={stats.input_tokens} out={stats.output_tokens} '
          f'cache_read={stats.cache_read_tokens}')

asyncio.run(main())
```

---

## Cross-references

| Class | Covered in | Notes |
|---|---|---|
| `AgentEventStream` | **This volume** | v1 Ch. 1 has a brief mention only |
| `ThinkingPart` + `ThinkingPartDelta` | **This volume** | `Thinking` capability in source_code_deep_dive Ch. 5 |
| `AudioUrl` / `VideoUrl` / `DocumentUrl` | **This volume** (deep) + v3 Ch. 5 (brief) | v3 covers the full `FileUrl` family |
| `OutputContext` | **This volume** | Output hooks covered in advanced_classes_part2 Ch. 2 |
| `ModelRetry` | **This volume** | Used throughout; source_code_deep_dive capstone |
| `RequestUsage` | **This volume** | `RunUsage` + `UsageLimits` in source_code_deep_dive Ch. 10 |
| `WebSearchTool` + `WebSearchUserLocation` | **This volume** (full params) + source_code_deep_dive Ch. 3 | source_code_deep_dive focuses on the `WebSearch` capability |
| `MemoryTool` | **This volume** | Brief mention in builtin_tools.md |
| `CodeExecutionTool` | **This volume** | Brief mention in builtin_tools.md |
| `AbstractNativeTool` | **This volume** | `NativeTool` / `NativeOrLocalTool` capabilities in v3 |

---

**Continue to [Class Deep Dives Vol. 8 →](./pydantic_ai_class_deep_dives_v8/)** —
`ToolOutput`/`NativeOutput`/`PromptedOutput`/`TextOutput`/`StructuredDict`, `ApprovalRequiredToolset`,
`DeferredLoadingToolset`, `Embedder`/`EmbeddingModel`/`EmbeddingResult`, `web_fetch_tool`,
`PrefectAgent`/`TaskConfig`, `ImageGenerationSubagentTool`, `ConcurrencyLimitedModel`,
`InstructionPart`/`AgentInstructions`.

---
title: "PydanticAI Class Deep Dives Vol. 36"
description: "Source-verified deep dives into 10 pydantic-ai class groups: streaming event protocol, delta types, DynamicToolset, capability toolset injection, search result DTOs, AnthropicModelSettings, Vercel AI SDK types, ModelResponsePartsManager, AG-UI HITL interrupt translation, and IncludeToolReturnSchemas."
sidebar:
  label: "Class deep dives (Vol. 36)"
  order: 62
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.9.1** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.9.x API. Three runnable examples per class group; all code blocks pass `ast.parse()` syntax validation.
</Aside>

Ten class groups covering the streaming event protocol, delta types, DynamicToolset factory patterns, capability toolset injection, search result DTOs, AnthropicModelSettings cache controls, Vercel AI SDK inbound request types, ModelResponsePartsManager stream management, AG-UI HITL interrupt translation, and IncludeToolReturnSchemas schema injection.

---

## 1 · `PartStartEvent` + `PartDeltaEvent` + `PartEndEvent` + `FinalResultEvent`

**Source:** `pydantic_ai/messages.py`

These four dataclasses make up `ModelResponseStreamEvent`, the discriminated union
that flows out of `StreamedResponse._get_event_iterator()`. Each carries an `index`
that identifies which part in the running `parts` list is being updated, plus an
`event_kind` literal used by Pydantic for efficient discriminated-union parsing.

```python
# Example 1 — Consume a raw event stream from agent.run_stream()
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartStartEvent,
    PartDeltaEvent,
    PartEndEvent,
    FinalResultEvent,
    TextPart,
)

agent = Agent('openai:gpt-4o-mini', system_prompt='Be concise.')


async def main() -> None:
    async with agent.run_stream('Name three planets.') as stream:
        async for event in stream.stream_response():
            if isinstance(event, PartStartEvent):
                print(f'[start  idx={event.index}] kind={event.part.part_kind}')
            elif isinstance(event, PartDeltaEvent):
                pass  # deltas are high-frequency; skip for demo
            elif isinstance(event, PartEndEvent):
                if isinstance(event.part, TextPart):
                    print(f'[end    idx={event.index}] text={event.part.content[:40]!r}')
            elif isinstance(event, FinalResultEvent):
                print(f'[result] tool_name={event.tool_name!r}')


asyncio.run(main())
```

```python
# Example 2 — Multiplex events to separate text and tool-call collectors
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ToolCallPartDelta,
    TextPart,
    ToolCallPart,
)

agent = Agent('openai:gpt-4o-mini', system_prompt='Use the get_time tool.')


@agent.tool_plain
def get_time() -> str:
    return '12:00 UTC'


async def main() -> None:
    text_buf: dict[int, str] = {}
    tool_calls: dict[int, str] = {}

    async with agent.run_stream('What time is it?') as stream:
        async for event in stream.stream_response():
            if isinstance(event, PartStartEvent):
                if isinstance(event.part, TextPart):
                    text_buf[event.index] = event.part.content
                elif isinstance(event.part, ToolCallPart):
                    tool_calls[event.index] = event.part.tool_name
            elif isinstance(event, PartDeltaEvent):
                delta = event.delta
                if isinstance(delta, TextPartDelta):
                    text_buf[event.index] = text_buf.get(event.index, '') + delta.content_delta
                elif isinstance(delta, ToolCallPartDelta) and delta.args_delta:
                    pass  # accumulate args if needed

    print('text chunks:', text_buf)
    print('tool calls:', tool_calls)


asyncio.run(main())
```

```python
# Example 3 — Build a latency monitor that tracks time-to-first-token per part
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.messages import PartStartEvent, PartEndEvent

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    timings: list[dict] = []
    t0 = time.perf_counter()

    async with agent.run_stream('Tell me a joke.') as stream:
        async for event in stream.stream_response():
            ts = time.perf_counter() - t0
            if isinstance(event, PartStartEvent):
                timings.append({'idx': event.index, 'kind': event.part.part_kind, 'start': ts})
            elif isinstance(event, PartEndEvent):
                for t in timings:
                    if t['idx'] == event.index:
                        t['end'] = ts
                        t['duration_ms'] = round((ts - t['start']) * 1000)

    for t in timings:
        print(f"part {t['idx']} ({t['kind']}): {t.get('duration_ms', '?')} ms")


asyncio.run(main())
```

---

## 2 · `TextPartDelta` + `ThinkingPartDelta` + `ToolCallPartDelta`

**Source:** `pydantic_ai/messages.py`

Delta dataclasses that ride inside `PartDeltaEvent.delta`. Each models *incremental*
data:

| Class | Key fields | Notes |
|---|---|---|
| `TextPartDelta` | `content_delta: str` | Append to `TextPart.content` |
| `ThinkingPartDelta` | `content_delta: str \| None`, `signature_delta: str \| None` | Signature replaces (never appends) |
| `ToolCallPartDelta` | `tool_name_delta: str \| None`, `args_delta: str \| dict \| None`, `tool_call_id: str \| None` | String args append; dict args merge |

```python
# Example 1 — Reconstruct full text by accumulating TextPartDelta
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta, TextPart

agent = Agent('openai:gpt-4o-mini')


async def collect_text() -> str:
    buf: dict[int, list[str]] = {}
    async with agent.run_stream('Write one sentence about Python.') as s:
        async for event in s.stream_response():
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                buf[event.index] = [event.part.content]
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                buf.setdefault(event.index, []).append(event.delta.content_delta)
    return ''.join(buf.get(0, []))


result = asyncio.run(collect_text())
print(repr(result))
```

```python
# Example 2 — Strip thinking content; only keep final text (extended-thinking model)
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import (
    PartStartEvent,
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
    TextPart,
    ThinkingPart,
)

agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[Thinking(effort='low')],
)


async def text_only() -> str:
    text_parts: dict[int, list[str]] = {}
    async with agent.run_stream('Is 17 prime?') as s:
        async for event in s.stream_response():
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                text_parts[event.index] = [event.part.content]
            elif isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, TextPartDelta):
                    text_parts.setdefault(event.index, []).append(event.delta.content_delta)
                elif isinstance(event.delta, ThinkingPartDelta):
                    pass  # discard thinking deltas
    return ' '.join(''.join(v) for v in text_parts.values())


# asyncio.run(text_only())
```

```python
# Example 3 — Accumulate streamed tool call args from ToolCallPartDelta
import asyncio
import json
from pydantic_ai import Agent
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, ToolCallPart, ToolCallPartDelta

agent = Agent('openai:gpt-4o-mini', system_prompt='Always call search_web.')


@agent.tool_plain
def search_web(query: str) -> list[str]:
    return [f'Result for {query}']


async def watch_args() -> None:
    args_buf: dict[int, str] = {}
    async with agent.run_stream('Search for "pydantic-ai streaming"') as s:
        async for event in s.stream_response():
            if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
                args_buf[event.index] = event.part.args_as_json_str()
                print(f'tool call started: {event.part.tool_name}')
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta):
                if isinstance(event.delta.args_delta, str):
                    args_buf[event.index] = args_buf.get(event.index, '') + event.delta.args_delta

    for idx, raw in args_buf.items():
        try:
            print(f'part {idx} args:', json.loads(raw))
        except json.JSONDecodeError:
            print(f'part {idx} incomplete args:', raw)


asyncio.run(watch_args())
```

---

## 3 · `DynamicToolset`

**Source:** `pydantic_ai/toolsets/_dynamic.py`

`DynamicToolset` wraps a `ToolsetFunc` — a callable that takes `RunContext` and
returns a (possibly `None`) `AbstractToolset`. When `per_run_step=True` (default),
the factory is re-called at every run step, enabling runtime decisions like:

- Role-based tool exposure
- Feature-flag gating
- One-time initialisation followed by a different toolset

```python
# Example 1 — Role-based tool exposure: admin tools only for privileged users
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import DynamicToolset, FunctionToolset


@dataclass
class AppDeps:
    role: str  # 'admin' | 'user'


user_tools = FunctionToolset[AppDeps]()
admin_tools = FunctionToolset[AppDeps]()


@user_tools.tool
def get_profile(ctx: RunContext[AppDeps]) -> str:
    return f'Profile for role={ctx.deps.role}'


@admin_tools.tool
def delete_user(ctx: RunContext[AppDeps], user_id: str) -> str:
    return f'Deleted user {user_id}'


def role_based_toolset(ctx: RunContext[AppDeps]):
    if ctx.deps.role == 'admin':
        return admin_tools
    return user_tools


agent: Agent[AppDeps, str] = Agent(
    'openai:gpt-4o-mini',
    deps_type=AppDeps,
    toolsets=[DynamicToolset(role_based_toolset)],
)

# asyncio.run(agent.run('What tools do I have?', deps=AppDeps(role='admin')))
```

```python
# Example 2 — Feature-flag toolset: serve different tools based on a config flag
import asyncio
import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import DynamicToolset, FunctionToolset

ENABLE_BETA = os.getenv('ENABLE_BETA', 'false').lower() == 'true'

stable_tools = FunctionToolset()
beta_tools = FunctionToolset()


@stable_tools.tool_plain
def stable_search(query: str) -> str:
    return f'Stable search: {query}'


@beta_tools.tool_plain
def vector_search(query: str, top_k: int = 5) -> list[str]:
    return [f'Vector result {i} for {query}' for i in range(top_k)]


def feature_gated(ctx: RunContext) -> FunctionToolset:
    return beta_tools if ENABLE_BETA else stable_tools


agent = Agent('openai:gpt-4o-mini', toolsets=[DynamicToolset(feature_gated)])

# asyncio.run(agent.run('Search for "embeddings"'))
```

```python
# Example 3 — per_run_step=False: expensive one-time setup, same toolset for all steps
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import DynamicToolset, FunctionToolset


async def build_db_toolset(ctx: RunContext) -> FunctionToolset:
    """Async factory: simulate opening a DB connection pool once per run."""
    print('Connecting to DB...')
    tools = FunctionToolset()

    @tools.tool_plain
    def query_db(sql: str) -> list[dict]:
        return [{'row': 1, 'sql': sql}]

    return tools


# per_run_step=False: async factory awaited once at run start;
# DynamicToolset handles async factories directly — no event-loop bridging needed.
agent = Agent('openai:gpt-4o-mini', toolsets=[DynamicToolset(build_db_toolset, per_run_step=False)])

# asyncio.run(agent.run('Query the users table'))
```

---

## 4 · `Toolset` capability + `IncludeToolReturnSchemas`

**Sources:** `pydantic_ai/capabilities/toolset.py`, `pydantic_ai/capabilities/include_return_schemas.py`

**`Toolset`** is a thin `AbstractCapability` wrapper that injects any `AgentToolset`
via the capabilities list instead of the `toolsets=` constructor arg — useful when
composing specs from YAML or when the toolset must live inside a capability chain.

**`IncludeToolReturnSchemas`** sets `include_return_schema=True` on matching
`ToolDefinition`s so the model sees the tool's return JSON Schema. Supports all
four `ToolSelector` shapes: `'all'`, `list[str]`, `dict[str, Any]` (metadata match),
and a sync/async callable predicate.

```python
# Example 1 — Inject a toolset via the Toolset capability (useful for AgentSpec)
from pydantic_ai import Agent
from pydantic_ai.capabilities import Toolset
from pydantic_ai.toolsets import FunctionToolset

tools = FunctionToolset()


@tools.tool_plain
def add(a: int, b: int) -> int:
    return a + b


@tools.tool_plain
def multiply(a: int, b: int) -> int:
    return a * b


# Equivalent to Agent(..., toolsets=[tools]) but composable via capabilities
agent = Agent('openai:gpt-4o-mini', capabilities=[Toolset(tools)])

result = agent.run_sync('What is 7 * 8?')
print(result.output)  # 56
```

```python
# Example 2 — IncludeToolReturnSchemas for all tools so the model sees return types
from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.toolsets import FunctionToolset
from pydantic import BaseModel

tools = FunctionToolset()


class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    condition: str


@tools.tool_plain
def get_weather(city: str) -> WeatherReport:
    return WeatherReport(city=city, temperature_c=22.5, condition='sunny')


# The model now receives the WeatherReport JSON Schema in the tool description
agent = Agent(
    'openai:gpt-4o-mini',
    toolsets=[tools],
    capabilities=[IncludeToolReturnSchemas()],  # 'all' by default
)

result = agent.run_sync('What is the weather in Paris?')
print(result.output)
```

```python
# Example 3 — IncludeToolReturnSchemas with a selective name list + metadata predicate
from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.toolsets import FunctionToolset

tools = FunctionToolset()


@tools.tool_plain
def search(query: str) -> list[str]:
    return [f'Result: {query}']


@tools.tool_plain
def lookup(key: str) -> dict:
    return {'key': key, 'value': 'some_value'}


# Only expose the return schema for 'search'; 'lookup' stays schema-free
agent = Agent(
    'openai:gpt-4o-mini',
    toolsets=[tools],
    capabilities=[IncludeToolReturnSchemas(tools=['search'])],
)

result = agent.run_sync('Search for "pydantic"')
print(result.output)
```

---

## 5 · `DuckDuckGoResult` + `TavilySearchResult` + `ExaSearchResult` + `ExaAnswerResult` + `WebFetchResult`

**Sources:** `pydantic_ai/common_tools/duckduckgo.py`, `common_tools/tavily.py`,
`common_tools/exa.py`, `common_tools/web_fetch.py`

These `TypedDict` classes are the structured payloads returned by common tools to
the LLM. Knowing their shapes lets you post-process results in your own tools, write
validators, or build RAG pipelines on top of the built-in search tools.

| Class | Key fields |
|---|---|
| `DuckDuckGoResult` | `title`, `href`, `body` |
| `TavilySearchResult` | `title`, `url`, `content`, `score` |
| `ExaSearchResult` | `title`, `url`, `published_date`, `author`, `text` |
| `ExaAnswerResult` | `answer`, `citations` |
| `WebFetchResult` | `url`, `title`, `content` (markdown) |

```python
# Example 1 — Post-process DuckDuckGo results: deduplicate by domain
from pydantic_ai.common_tools.duckduckgo import DuckDuckGoResult, duckduckgo_search_tool
from pydantic_ai import Agent
from urllib.parse import urlparse


def deduplicate_by_domain(results: list[DuckDuckGoResult]) -> list[DuckDuckGoResult]:
    """Return at most one result per domain."""
    seen: set[str] = set()
    unique: list[DuckDuckGoResult] = []
    for r in results:
        domain = urlparse(r['href']).netloc
        if domain not in seen:
            seen.add(domain)
            unique.append(r)
    return unique


# The tool returns list[DuckDuckGoResult]; a wrapper tool post-processes it
agent = Agent('openai:gpt-4o-mini', tools=[duckduckgo_search_tool(max_results=10)])

# In a real run the LLM calls duckduckgo_search; your code can use the typed dicts
sample: list[DuckDuckGoResult] = [
    {'title': 'Pydantic Docs', 'href': 'https://docs.pydantic.dev/a', 'body': 'Pydantic is...'},
    {'title': 'Pydantic Blog', 'href': 'https://docs.pydantic.dev/b', 'body': 'Blog post...'},
    {'title': 'FastAPI', 'href': 'https://fastapi.tiangolo.com/', 'body': 'FastAPI...'},
]
deduped = deduplicate_by_domain(sample)
print(len(deduped))  # 2
```

```python
# Example 2 — Filter TavilySearchResult by relevance score, then pass to agent
from pydantic_ai.common_tools.tavily import TavilySearchResult


def high_confidence(results: list[TavilySearchResult], min_score: float = 0.7) -> list[TavilySearchResult]:
    return [r for r in results if r['score'] >= min_score]


sample_results: list[TavilySearchResult] = [
    {'title': 'Pydantic AI', 'url': 'https://ai.pydantic.dev', 'content': 'Type-safe agents.', 'score': 0.95},
    {'title': 'Random Blog', 'url': 'https://blog.example.com', 'content': 'An old post.', 'score': 0.42},
]

trusted = high_confidence(sample_results)
print([r['title'] for r in trusted])  # ['Pydantic AI']
```

```python
# Example 3 — Consume ExaSearchResult and ExaAnswerResult for a citations-aware agent
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaSearchResult, ExaAnswerResult, ExaToolset

# ExaToolset bundles 4 Exa tools: search, find_similar, get_contents, answer
# exa_toolset = ExaToolset(api_key='...', num_results=5, max_characters=2000)
# agent = Agent('openai:gpt-4o-mini', toolsets=[exa_toolset])

# When the agent calls exa_answer, results are list[ExaAnswerResult]
sample_answer: ExaAnswerResult = {
    'answer': 'pydantic-ai is a Python agent framework.',
    'citations': [{'url': 'https://ai.pydantic.dev', 'title': 'PydanticAI'}],
}

citation_urls = [c.get('url', '') for c in sample_answer['citations']]
print('Answer:', sample_answer['answer'])
print('Cited:', citation_urls)
```

---

## 6 · `AnthropicModelSettings`

**Source:** `pydantic_ai/models/anthropic.py`

`AnthropicModelSettings` extends `ModelSettings` with `anthropic_`-prefixed fields
for Anthropic-specific request parameters. All fields are optional (`TypedDict,
total=False`) and ignored by other providers, so they can be safely merged with a
cross-provider `ModelSettings` dict.

Key fields:

| Field | Type | Purpose |
|---|---|---|
| `anthropic_cache` | `bool \| Literal['5m','1h']` | Auto prompt-caching (multi-turn) |
| `anthropic_cache_tool_definitions` | `bool \| Literal['5m','1h']` | Cache the tool schema block |
| `anthropic_cache_instructions` | `bool \| Literal['5m','1h']` | Cache the last system prompt block |
| `anthropic_thinking` | `BetaThinkingConfigParam` | Low-level thinking budget config |
| `anthropic_metadata` | `BetaMetadataParam` | Pass `user_id` for abuse detection |
| `anthropic_service_tier` | `Literal['auto','standard_only']` | Billing tier selection |

```python
# Example 1 — Prompt caching: cache instructions and tool definitions
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-opus-4-5',
    system_prompt='You are a helpful coding assistant with extensive Python knowledge.',
)

settings: AnthropicModelSettings = {
    'anthropic_cache_instructions': '1h',  # cache the system prompt for 1 hour
    'anthropic_cache_tool_definitions': True,  # cache tool schema (5 min TTL)
}

# asyncio.run(agent.run('Explain generators in Python.', model_settings=settings))
```

```python
# Example 2 — Attach user_id metadata for Anthropic's trust-and-safety pipeline
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings


async def run_for_user(user_id: str, prompt: str) -> str:
    agent = Agent('anthropic:claude-haiku-4-5')
    settings: AnthropicModelSettings = {
        'anthropic_metadata': {'user_id': user_id},
    }
    result = await agent.run(prompt, model_settings=settings)
    return result.output


# asyncio.run(run_for_user('user-abc-123', 'What is 2+2?'))
```

```python
# Example 3 — Standard-only tier + explicit thinking budget for cost control
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent('anthropic:claude-opus-4-5', system_prompt='Solve step-by-step.')

settings: AnthropicModelSettings = {
    'anthropic_service_tier': 'standard_only',  # avoid priority-tier pricing
    'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 2000},
    'max_tokens': 4096,
}

# asyncio.run(agent.run('Is 1_048_573 prime?', model_settings=settings))
```

---

## 7 · Vercel AI SDK request types: `TextUIPart` + `ReasoningUIPart` + `FileUIPart` + `ToolApprovalRespondedPart`

**Source:** `pydantic_ai/ui/vercel_ai/request_types.py`

These `CamelBaseModel` subclasses represent the *inbound* parts of a `UIMessage`
(what the Vercel AI SDK sends to the server). They use Pydantic's `alias_generator`
so JSON keys are `camelCase` on the wire but `snake_case` in Python.

```python
# Example 1 — Parse a full UIMessage payload from a Vercel AI SDK frontend request
from pydantic_ai.ui.vercel_ai.request_types import (
    TextUIPart,
    FileUIPart,
    UIMessage,
    SubmitMessage,
)

# UIMessage uses `parts` (not `content`); SubmitMessage wraps messages in `messages`.
user_msg = UIMessage(
    id='msg-1',
    role='user',
    parts=[
        TextUIPart(text='Explain this image.'),
        FileUIPart(media_type='image/png', url='data:image/png;base64,iVBORw0KGgo='),
    ],
)

# SubmitMessage: required fields are `id` and `messages`.
submit = SubmitMessage(id='req-1', messages=[user_msg])

for part in submit.messages[0].parts:
    if isinstance(part, TextUIPart):
        print('Text:', part.text)
    elif isinstance(part, FileUIPart):
        print('File:', part.media_type, part.url[:30])
```

```python
# Example 2 — Check for a HITL approval response in an incoming UIMessage
from pydantic_ai.ui.vercel_ai.request_types import (
    TextUIPart,
    ToolApprovalRespondedPart,
    UIMessage,
)

# Parts use the *Part suffix classes; UIMessage.parts holds them.
user_msg = UIMessage(
    id='msg-2',
    role='user',
    parts=[
        TextUIPart(text='Yes, proceed.'),
        ToolApprovalRespondedPart(
            type='tool-approval-responded',
            tool_call_id='tc-001',
        ),
    ],
)

approvals = [p for p in user_msg.parts if isinstance(p, ToolApprovalRespondedPart)]
print('Approval tool_call_ids:', [a.tool_call_id for a in approvals])
```

```python
# Example 3 — Build a SubmitMessage programmatically for testing an endpoint
from pydantic_ai.ui.vercel_ai.request_types import TextUIPart, UIMessage, SubmitMessage

user_msg = UIMessage(
    id='msg-3',
    role='user',
    parts=[TextUIPart(text='Hello from the test suite!', state='done')],
)

submit = SubmitMessage(id='req-3', messages=[user_msg])

# Serialise to camelCase JSON for wire transmission
payload = submit.model_dump(by_alias=True, exclude_none=True)
print(payload)
# {'trigger': 'submit-message', 'id': 'req-3', 'messages': [{'id': 'msg-3', 'role': 'user', ...}]}
```

---

## 8 · Vercel AI SDK response types: `TextStartChunk` + `TextDeltaChunk` + `ToolInputStartChunk` + `ToolApprovalRequestChunk`

**Source:** `pydantic_ai/ui/vercel_ai/response_types.py`

These classes represent the *outbound* SSE stream chunks sent back to a Vercel AI
SDK frontend. Each has an `encode(sdk_version: int) -> str` method that serialises
to a camelCase JSON line — the adapter strips `sdk_version`-unavailable fields
automatically.

```python
# Example 1 — Build a minimal SSE stream for a text response
from pydantic_ai.ui.vercel_ai.response_types import (
    StartChunk,
    TextStartChunk,
    TextDeltaChunk,
    TextEndChunk,
    FinishChunk,
)

SDK_VERSION = 6

chunks = [
    StartChunk(message_id='msg-001'),
    TextStartChunk(id='txt-001'),
    TextDeltaChunk(id='txt-001', delta='Hello, '),
    TextDeltaChunk(id='txt-001', delta='world!'),
    TextEndChunk(id='txt-001'),
    FinishChunk(finish_reason='stop'),
]

# Each line is sent as `data: <json>\n\n` in an SSE endpoint
for chunk in chunks:
    print(f'data: {chunk.encode(SDK_VERSION)}')
```

```python
# Example 2 — Emit reasoning chunks alongside text (for extended-thinking models)
from pydantic_ai.ui.vercel_ai.response_types import (
    StartChunk,
    ReasoningStartChunk,
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    TextStartChunk,
    TextDeltaChunk,
    TextEndChunk,
    FinishChunk,
)

SDK_VERSION = 6

stream = [
    StartChunk(message_id='msg-002'),
    ReasoningStartChunk(id='think-001'),
    ReasoningDeltaChunk(id='think-001', delta='Let me think about this...'),
    ReasoningEndChunk(id='think-001'),
    TextStartChunk(id='txt-002'),
    TextDeltaChunk(id='txt-002', delta='The answer is 42.'),
    TextEndChunk(id='txt-002'),
    FinishChunk(finish_reason='stop'),
]

for c in stream:
    line = f'data: {c.encode(SDK_VERSION)}'
    print(line)
```

```python
# Example 3 — Emit a tool-approval-request chunk for HITL (SDK v6+)
from pydantic_ai.ui.vercel_ai.response_types import (
    StartChunk,
    TextStartChunk,
    TextDeltaChunk,
    TextEndChunk,
    ToolInputStartChunk,
    ToolInputDeltaChunk,
    ToolInputAvailableChunk,
    ToolApprovalRequestChunk,
    FinishChunk,
)

SDK_VERSION = 6

stream = [
    StartChunk(message_id='msg-003'),
    TextStartChunk(id='txt-003'),
    TextDeltaChunk(id='txt-003', delta='I need to delete a file.'),
    TextEndChunk(id='txt-003'),
    ToolInputStartChunk(tool_call_id='tc-001', tool_name='delete_file'),
    ToolInputDeltaChunk(tool_call_id='tc-001', input_text_delta='{"path": "/etc/hosts"}'),
    ToolInputAvailableChunk(tool_call_id='tc-001', tool_name='delete_file', input='{"path": "/etc/hosts"}'),
    ToolApprovalRequestChunk(approval_id='apr-001', tool_call_id='tc-001'),
    FinishChunk(finish_reason='tool-calls'),
]

for c in stream:
    print(f'data: {c.encode(SDK_VERSION)}')
```

---

## 9 · `ModelResponsePartsManager`

**Source:** `pydantic_ai/_parts_manager.py`

`ModelResponsePartsManager` is the streaming aggregator used internally by every
`StreamedResponse` subclass (OpenAI, Anthropic, Google, etc.). When implementing a
custom model provider, instantiate it in your `_get_event_iterator()` and call
`handle_text_delta()`, `handle_tool_call_delta()`, etc. to get correctly typed
`PartStartEvent` / `PartDeltaEvent` / `PartEndEvent` events with automatic
deduplication and vendor-ID tracking.

```python
# Example 1 — Custom streamed response using ModelResponsePartsManager
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    ModelResponseStreamEvent,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.usage import RequestUsage


class EchoStreamedResponse(StreamedResponse):
    """Streams a fixed reply word-by-word using ModelResponsePartsManager."""

    def __init__(self, params: ModelRequestParameters, words: list[str]) -> None:
        super().__init__(params)
        self._words = words

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for word in self._words:
            # Use self._parts_manager so __aiter__'s PartEndEvent logic stays in sync
            for event in self._parts_manager.handle_text_delta(vendor_part_id='text', content=word + ' '):
                yield event

    async def close_stream(self) -> None:
        pass

    def get(self) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=' '.join(self._words))])

    @property
    def usage(self) -> RequestUsage:
        return RequestUsage()

    @property
    def model_name(self) -> str:
        return 'echo'

    @property
    def provider_name(self) -> str:
        return 'custom'

    @property
    def provider_url(self) -> str | None:
        return None

    @property
    def timestamp(self) -> datetime:
        return datetime.now(timezone.utc)
```

```python
# Example 2 — Emit mixed text + tool call events from a single manager
import asyncio
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent


async def demo_mixed_stream(manager: ModelResponsePartsManager):
    """Show how the manager emits PartStartEvent on first delta per vendor_part_id."""
    events: list[ModelResponseStreamEvent] = []

    # Text part — handle_text_delta is a sync Iterator; content= (not content_delta=)
    for e in manager.handle_text_delta(vendor_part_id='t0', content='Hello'):
        events.append(e)
    for e in manager.handle_text_delta(vendor_part_id='t0', content=' world'):
        events.append(e)

    # Tool call part — handle_tool_call_delta returns a single event or None
    if e := manager.handle_tool_call_delta(
        vendor_part_id='tc0', tool_name='get_weather', args='{"city":', tool_call_id='id-1'
    ):
        events.append(e)
    if e := manager.handle_tool_call_delta(vendor_part_id='tc0', tool_name=None, args='"Paris"}'):
        events.append(e)

    for ev in events:
        print(ev.event_kind, getattr(ev, 'index', ''))
```

```python
# Example 3 — Inspect manager internals: current parts and last_vendor_part_id
import asyncio
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import TextPart


async def inspect_manager() -> None:
    # ModelResponsePartsManager requires ModelRequestParameters but accepts None for testing
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.tools import ToolDefinition

    params = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
    )
    manager = ModelResponsePartsManager(params)

    # First delta creates a new TextPart via PartStartEvent
    for _ in manager.handle_text_delta(vendor_part_id='p0', content='Hi!'):
        pass

    # The manager tracks current parts
    current = manager.get_parts()
    assert len(current) == 1
    assert isinstance(current[0], TextPart)
    print('Content so far:', current[0].content)  # 'Hi!'


asyncio.run(inspect_manager())
```

---

## 10 · AG-UI HITL: `approval_to_interrupt` + `resume_entry_to_approval`

**Source:** `pydantic_ai/ui/ag_ui/_interrupt.py`

Pydantic AI translates between its own `DeferredToolApprovalResult` model and the
AG-UI protocol's `Interrupt` / `ResumeEntry` types. Two functions do the
bidirectional translation:

- **`approval_to_interrupt(call, metadata)`** — outbound: converts a pending
  `ToolCallPart` into an `Interrupt` the AG-UI frontend can display as a HITL
  confirmation dialog.
- **`resume_entry_to_approval(entry)`** — inbound: converts the user's
  `ResumeEntry` (containing `{approved: true, editedArgs?: {...}}`) back into
  `ToolApproved` or `ToolDenied`.

The module also exports `HAS_INTERRUPTS: bool` which is `True` only when
`ag-ui-protocol >= 0.1.19` is installed.

```python
# Example 1 — Check feature availability and build an Interrupt for a tool call
# Requires: pip install "pydantic-ai[ag-ui]"
from pydantic_ai.ui.ag_ui._interrupt import HAS_INTERRUPTS, approval_to_interrupt
from pydantic_ai.messages import ToolCallPart

print('AG-UI interrupt support:', HAS_INTERRUPTS)

if HAS_INTERRUPTS:
    pending_call = ToolCallPart(
        tool_name='delete_file',
        args='{"path": "/var/log/app.log"}',
        tool_call_id='tc-abc-001',
    )
    interrupt = approval_to_interrupt(pending_call, metadata={})
    print('Interrupt ID:', interrupt.id)       # 'pai_tc-abc-001'
    print('Message:', interrupt.message)       # 'Approve delete_file(...)? '
    print('Schema:', interrupt.response_schema['properties'].keys())
```

```python
# Example 2 — Resume: translate a user's approval back into ToolApproved
# Requires: pip install "pydantic-ai[ag-ui]"
from pydantic_ai.ui.ag_ui._interrupt import HAS_INTERRUPTS, resume_entry_to_approval
from pydantic_ai.tools import ToolApproved, ToolDenied

if HAS_INTERRUPTS:
    from ag_ui.core import ResumeEntry

    # Approved with edited args
    approved_entry = ResumeEntry(
        interrupt_id='pai_tc-abc-001',
        status='completed',
        payload={'approved': True, 'editedArgs': {'path': '/tmp/safe.log'}},
    )
    result = resume_entry_to_approval(approved_entry)
    assert isinstance(result, ToolApproved)
    print('Override args:', result.override_args)  # {'path': '/tmp/safe.log'}

    # Denied with reason
    denied_entry = ResumeEntry(
        interrupt_id='pai_tc-abc-001',
        status='completed',
        payload={'approved': False, 'reason': 'Too dangerous'},
    )
    result2 = resume_entry_to_approval(denied_entry)
    assert isinstance(result2, ToolDenied)
    print('Denial message:', result2.message)  # 'Too dangerous'
```

```python
# Example 3 — Full HITL round-trip guard: deny-by-default on ambiguous payloads
# Requires: pip install "pydantic-ai[ag-ui]"
from pydantic_ai.ui.ag_ui._interrupt import HAS_INTERRUPTS, resume_entry_to_approval
from pydantic_ai.tools import ToolApproved, ToolDenied

if HAS_INTERRUPTS:
    from ag_ui.core import ResumeEntry

    ambiguous_cases = [
        ResumeEntry(interrupt_id='pai_x', status='cancelled', payload=None),
        ResumeEntry(interrupt_id='pai_x', status='completed', payload=None),
        ResumeEntry(interrupt_id='pai_x', status='completed', payload={'approved': False}),
        ResumeEntry(interrupt_id='pai_x', status='completed', payload={'approved': 'yes'}),
    ]

    for entry in ambiguous_cases:
        result = resume_entry_to_approval(entry)
        assert isinstance(result, ToolDenied), f'Expected denial, got {result}'
        print(f'status={entry.status!r} payload={entry.payload!r} -> ToolDenied ✓')
```

---

## Revision history

| Version | Date | Notes |
|---|---|---|
| v2.9.1 | 2026-07-14 | Initial publication — 10 class groups, 30 examples, source-verified |

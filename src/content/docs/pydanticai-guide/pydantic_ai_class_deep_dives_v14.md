---
title: "PydanticAI — Class Deep Dives Vol. 14"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.107.0: UIAdapter/UIEventStream (unified UI protocol base), AGUIAdapter (new dispatch_request pattern + ag_ui_version thresholds), VercelAIAdapter (sdk_version=6 HITL), Provider ABC (custom providers + infer_provider), ModelProfile complete field reference (all 17 fields + StructuredOutputMode), AnthropicModelProfile + OpenAIModelProfile (provider-specific profile extensions), WrapperEmbeddingModel + InstrumentedEmbeddingModel (custom embedding wrappers), additional embedding providers (Google/Bedrock/Cohere/VoyageAI), BuilderCheckpoint + MessagesBuilder advanced patterns, OutlinesModel (deprecated constrained generation + migration guide). All verified against pydantic-ai 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 14)"
  order: 40
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups spanning the UI protocol layer, provider abstraction, model capability profiles, and extended embeddings ecosystem: `UIAdapter` + `UIEventStream` (the unified streaming adapter ABC that backs every frontend integration); `AGUIAdapter` (updated AG-UI specific adapter with version-gated protocol thresholds); `VercelAIAdapter` (updated Vercel AI SDK adapter with SDK v6 HITL streaming); `Provider` abstract base (the authenticated-client ABC that all 30+ providers implement); `ModelProfile` complete field reference (all 17 fields including every `supports_*` flag added since Vol. 2); `AnthropicModelProfile` + `OpenAIModelProfile` (provider-specific profile extensions for adaptive thinking and custom reasoning fields); `WrapperEmbeddingModel` + `InstrumentedEmbeddingModel` (custom embedding wrapper base and OTel instrumentation); additional embedding providers (`GoogleEmbeddingModel`, `BedrockEmbeddingModel`, `CohereEmbeddingModel`, `VoyageAIEmbeddingModel`); `BuilderCheckpoint` + `MessagesBuilder` advanced patterns (message attribution and snapshot/diff for custom `UIEventStream` implementations); and `OutlinesModel` (deprecated constrained generation model with migration guide).

---

## 1. `UIAdapter` + `UIEventStream` + `StateDeps` + `StateHandler` + `OnCompleteFunc` + `NativeEvent`

**Module:** `pydantic_ai.ui`  
**Imports:**
```python
from pydantic_ai.ui import (
    UIAdapter, UIEventStream, StateDeps, StateHandler,
    OnCompleteFunc, NativeEvent, SSE_CONTENT_TYPE,
    MessagesBuilder, BuilderCheckpoint,
)
```

`UIAdapter` is the abstract dataclass base that both `AGUIAdapter` and `VercelAIAdapter` extend. It owns the security policy for every incoming frontend request and provides the three-method lifecycle that custom adapters implement.

### `UIAdapter` constructor fields

```python
@dataclass
class UIAdapter(ABC, Generic[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]):
    agent: AbstractAgent[AgentDepsT, OutputDataT]
    run_input: RunInputT

    # Security policy
    manage_system_prompt: Literal['server', 'client'] = 'server'
    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'})
    allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset()
    preserve_file_data: bool = False
    accept: str | None = None
```

| Field | Purpose |
|---|---|
| `manage_system_prompt='server'` | Strips any `SystemPromptPart` the client sends and reinjects the agent's own prompt via `ReinjectSystemPrompt` capability — prevents prompt injection. |
| `manage_system_prompt='client'` | Passes client `SystemPromptPart` through unchanged; agent's configured prompt is **not** injected. |
| `allowed_file_url_schemes` | URL parts (`ImageUrl`, `AudioUrl`, etc.) whose scheme is not in this set are dropped with a warning. Default `{'http', 'https'}` only. Add `'s3'`/`'gs'` after auditing IAM exposure. |
| `allowed_file_url_force_download` | Extra `force_download` values accepted from clients. `False` is always safe; `True` makes the server download the file; `'allow-local'` disables SSRF protection. |
| `preserve_file_data` | When `True`, `UploadedFile` items are kept in client messages. Default `False`. |

### The three abstract methods

```python
@classmethod
@abstractmethod
def build_run_input(cls, body: bytes) -> RunInputT: ...

@abstractmethod
def build_event_stream(
    self,
) -> UIEventStream[RunInputT, EventT, AgentDepsT, OutputDataT]: ...

@classmethod
@abstractmethod
async def from_request(
    cls,
    request: Request,
    *,
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    **kwargs: Any,
) -> Self: ...
```

### `dispatch_request` — the entry point

Every adapter ships a `dispatch_request` class method that does the full parse → security sanitise → run → stream cycle:

```python
from starlette.applications import Starlette
from starlette.routing import Route
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter  # concrete subclass

agent = Agent('anthropic:claude-sonnet-4-6', system_prompt='You are helpful.')

async def handle(request):
    return await AGUIAdapter.dispatch_request(request, agent=agent)

app = Starlette(routes=[Route('/', handle, methods=['POST'])])
```

### `StateDeps` + `StateHandler`

`StateDeps` is a protocol for dependency objects that carry frontend state:

```python
from pydantic_ai.ui import StateDeps, StateHandler
from dataclasses import dataclass
from typing import Any

@dataclass
class MyDeps:
    user_id: str
    state: dict[str, Any] | None = None  # injected per-request

# Implements StateDeps protocol automatically — no explicit inheritance needed
```

`StateHandler` is a narrower protocol for objects that implement a `handle_state` method, used when the adapter needs to pass arbitrary AG-UI/Vercel AI state from the frontend to the backend:

```python
class MyDepsWithHandler:
    def handle_state(self, state: dict[str, Any]) -> None:
        self.frontend_state = state
```

### `OnCompleteFunc`

Called once after the agent run completes successfully. Receives the full `AgentRunResult`:

```python
from pydantic_ai.ui import OnCompleteFunc
from pydantic_ai import AgentRunResult

async def on_done(result: AgentRunResult[str]) -> None:
    print('Run complete, tokens used:', result.usage().total_tokens)
    # Save messages to DB, emit analytics, etc.

# Pass to dispatch_request:
await AGUIAdapter.dispatch_request(request, agent=agent, on_complete=on_done)
```

### `NativeEvent`

A union alias for the raw SSE event objects emitted by a `UIEventStream`. For AG-UI this is `ag_ui.core.BaseEvent`; for Vercel AI it is the `BaseChunk` union. You rarely need to import it directly unless writing a custom event stream.

### `UIEventStream`

The abstract streaming transformer that converts `AgentStreamEvent` objects into protocol-specific events:

```python
from pydantic_ai.ui import UIEventStream

class MyEventStream(UIEventStream[MyRunInput, MyEvent, MyDeps, str]):
    def encode_event(self, event: MyEvent) -> str:
        return f'data: {event.model_dump_json()}\n\n'

    async def on_agent_stream_event(
        self, event: AgentStreamEvent
    ) -> AsyncIterator[MyEvent]:
        # transform pydantic_ai events into your protocol events
        ...
```

---

## 2. `AGUIAdapter` — Updated AG-UI Adapter

**Module:** `pydantic_ai.ui.ag_ui`  
**Import:** `from pydantic_ai.ui.ag_ui import AGUIAdapter`

`AGUIAdapter` extends `UIAdapter` with AG-UI protocol-specific behaviour including version-gated event formats.

### Constructor extras

```python
@dataclass
class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    ag_ui_version: str = DEFAULT_AG_UI_VERSION  # detected from installed ag-ui-protocol
```

### `ag_ui_version` thresholds

| Version threshold | Behaviour |
|---|---|
| `< 0.1.13` | Emits `THINKING_*` events; drops `ThinkingPart` from `dump_messages` |
| `≥ 0.1.13` | Emits `REASONING_*` events with encrypted provider metadata; includes `ThinkingPart` as `ReasoningMessage` for round-trip fidelity |
| `≥ 0.1.15` | Emits typed multimodal input content (`ImageInputContent`, `AudioInputContent`, `VideoInputContent`, `DocumentInputContent`) instead of generic `BinaryInputContent` |

```python
from pydantic_ai.ui.ag_ui import AGUIAdapter

# Force a specific protocol version (e.g. for clients that haven't upgraded yet):
async def handle(request):
    return await AGUIAdapter.dispatch_request(
        request,
        agent=agent,
        ag_ui_version='0.1.12',  # emit THINKING_* instead of REASONING_*
    )
```

### `dispatch_request` — replacing deprecated `handle_ag_ui_request`

```python
# Before (deprecated):
from pydantic_ai.ag_ui import handle_ag_ui_request

async def old_handle(request):
    return await handle_ag_ui_request(agent, request)

# After (current):
from pydantic_ai.ui.ag_ui import AGUIAdapter

async def new_handle(request):
    return await AGUIAdapter.dispatch_request(request, agent=agent)
```

### `preserve_file_data` round-trip

```python
async def handle_with_files(request):
    return await AGUIAdapter.dispatch_request(
        request,
        agent=agent,
        preserve_file_data=True,           # keep UploadedFile items across turns
        allowed_file_url_schemes=frozenset({'http', 'https', 's3'}),  # allow S3 URLs
    )
```

### Migration from deprecated `AGUIApp`

```python
# Before:
from pydantic_ai.ui.ag_ui.app import AGUIApp  # deprecated, issues DeprecationWarning
app = AGUIApp(agent)

# After (bare Starlette):
from starlette.applications import Starlette
from starlette.routing import Route
from pydantic_ai.ui.ag_ui import AGUIAdapter

async def run_agent(request):
    return await AGUIAdapter.dispatch_request(request, agent=agent)

app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
```

### Per-request deps with `from_request`

When you need per-request dependencies (e.g. authenticated user from a JWT), use `from_request()` then call `run_stream()`:

```python
from starlette.requests import Request
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai import Agent
from dataclasses import dataclass

@dataclass
class RequestDeps:
    user_id: str

agent: Agent[RequestDeps, str] = Agent('openai:gpt-4o')

async def handle(request: Request):
    user_id = request.headers.get('X-User-Id', 'anon')
    adapter = await AGUIAdapter.from_request(
        request, agent=agent, manage_system_prompt='server'
    )
    return await adapter.run_stream(deps=RequestDeps(user_id=user_id))
```

---

## 3. `VercelAIAdapter` — Updated Vercel AI Adapter

**Module:** `pydantic_ai.ui.vercel_ai`  
**Import:** `from pydantic_ai.ui.vercel_ai import VercelAIAdapter`

### Constructor extras

```python
@dataclass
class VercelAIAdapter(UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    sdk_version: Literal[5, 6] = 5
    server_message_id: str | None = None
```

### `sdk_version=6` — tool approval streaming (HITL)

SDK v6 enables human-in-the-loop tool approval via streaming. When the agent's `ApprovalRequiredToolset` defers a tool call, the adapter emits an approval-request chunk that the Vercel AI SDK v6 frontend can render as a UI prompt:

```python
from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset

toolset = FunctionToolset()

@toolset.tool
async def delete_record(record_id: str) -> str:
    """Delete a database record."""
    return f'Deleted {record_id}'

approval_ts = ApprovalRequiredToolset(toolset)
agent = Agent('openai:gpt-4o', toolsets=[approval_ts])

async def handle(request):
    return await VercelAIAdapter.dispatch_request(
        request, agent=agent, sdk_version=6  # enables approval streaming
    )
```

### `server_message_id` — stable message IDs

Assign a server-generated message ID included in the `StartChunk` so the frontend can correlate streaming responses:

```python
import uuid

async def handle(request):
    return await VercelAIAdapter.dispatch_request(
        request,
        agent=agent,
        server_message_id=str(uuid.uuid4()),
    )
```

### `dispatch_request` replacing old route pattern

```python
# Old pattern (from Vol. 4):
from pydantic_ai.ui.vercel_ai._adapter import VercelAIAdapter
adapter = VercelAIAdapter(agent=agent, run_input=run_input)
return adapter.streaming_response(adapter.run_stream(deps=deps))

# New pattern (current):
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
return await VercelAIAdapter.dispatch_request(request, agent=agent, deps=deps)
```

### SDK version comparison

| Feature | `sdk_version=5` | `sdk_version=6` |
|---|---|---|
| Tool calls | Standard function call chunks | Standard function call chunks |
| HITL tool approval | Not supported | `ToolApprovalChunk` emitted |
| Default | ✓ | Must opt in |

---

## 4. `Provider` ABC + `infer_provider` / `infer_provider_class`

**Module:** `pydantic_ai.providers`  
**Import:** `from pydantic_ai.providers import Provider, infer_provider, infer_provider_class`

`Provider[InterfaceClient]` is the abstract base that every provider (OpenAI, Anthropic, Google, Bedrock, etc.) implements. It separates authentication and HTTP lifecycle from the model's inference logic.

### Abstract properties every provider must implement

```python
from abc import abstractmethod
from pydantic_ai.providers import Provider

class MyProvider(Provider[MySDKClient]):
    @property
    def name(self) -> str:
        return 'my-provider'      # appears in ModelMessage.provider_name

    @property
    def base_url(self) -> str:
        return 'https://api.myprovider.com/v1'

    @property
    def client(self) -> MySDKClient:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        # Return None to use the default profile, or a custom one:
        if model_name.startswith('my-thinking-'):
            return ModelProfile(supports_thinking=True, thinking_always_enabled=False)
        return None
```

### Async context manager lifecycle

Providers that own their own HTTP client implement `__aenter__`/`__aexit__` to open/close it:

```python
class MyProvider(Provider[MySDKClient]):
    def __init__(self, api_key: str):
        import httpx
        http = httpx.AsyncClient(headers={'Authorization': f'Bearer {api_key}'})
        self._own_http_client = http
        self._http_client_factory = lambda: httpx.AsyncClient(
            headers={'Authorization': f'Bearer {api_key}'}
        )
        self._client = MySDKClient(http_client=http)
        self._entered_count = 0

# Use as context manager to ensure clean HTTP shutdown:
async with MyProvider('sk-...') as provider:
    model = MyChatModel(model_name='my-model', provider=provider)
    agent = Agent(model)
    result = await agent.run('Hello')
```

The base class handles the `_entered_count` counter and `_enter_lock` (an `anyio.Lock` created lazily on first access to bind correctly to the running event loop).

### `infer_provider` — string to provider instance

```python
from pydantic_ai.providers import infer_provider

# Returns a concrete Provider instance from its string prefix:
openai_provider = infer_provider('openai')      # OpenAIProvider()
anthropic = infer_provider('anthropic')          # AnthropicProvider()
groq = infer_provider('groq')                   # GroqProvider()
gw = infer_provider('gateway/anthropic')         # GatewayProvider wrapping anthropic
```

### `infer_provider_class` — string to provider class

```python
from pydantic_ai.providers import infer_provider_class

ProviderCls = infer_provider_class('openai')
# ProviderCls is OpenAIProvider (uninstantiated)
custom = ProviderCls(base_url='https://my-openai-compatible.api/v1', api_key='...')
```

### All provider string keys (selected)

| String | Provider class |
|---|---|
| `'openai'` | `OpenAIProvider` |
| `'anthropic'` | `AnthropicProvider` |
| `'google'` | `GoogleProvider` (Gemini API / AI Studio) |
| `'google-cloud'` | `GoogleCloudProvider` (Vertex AI) |
| `'groq'` | `GroqProvider` |
| `'mistral'` | `MistralProvider` |
| `'xai'` | `XAIProvider` |
| `'bedrock'` | `BedrockProvider` |
| `'cohere'` | `CohereProvider` |
| `'ollama'` | `OllamaProvider` |
| `'openrouter'` | `OpenRouterProvider` |
| `'azure'` | `AzureProvider` |
| `'deepseek'` | `DeepSeekProvider` |
| `'gateway/<name>'` | `GatewayProvider(name)` |

---

## 5. `ModelProfile` — Complete Field Reference

**Module:** `pydantic_ai.profiles`  
**Import:** `from pydantic_ai.profiles import ModelProfile, ModelProfileSpec, DEFAULT_PROFILE`

`ModelProfile` is a `@dataclass(kw_only=True)` with 17 fields that govern how a model/provider combination handles requests. Every field has a conservative default; provider-specific subclasses override the relevant ones.

### Complete field reference

```python
from pydantic_ai.profiles import ModelProfile
from pydantic_ai._json_schema import JsonSchemaTransformer

profile = ModelProfile(
    # Tool support
    supports_tools=True,
    supports_tool_return_schema=False,     # True → send return schema alongside tool def
                                           # False → inject schema as JSON in tool description

    # Structured output modes
    supports_json_schema_output=False,     # True → NativeOutput mode works natively
    supports_json_object_output=False,     # True → PromptedOutput/JSON-mode works
    supports_image_output=False,           # True → model can emit image content

    # Prompt format
    supports_inline_system_prompts=False,  # True → SystemPromptPart accepted mid-turn
                                           # False → non-leading system prompts wrapped as UserPromptPart

    # Structured output default
    default_structured_output_mode='tool', # 'tool' | 'json_schema' | 'json_object' | 'prompted'

    # Template for PromptedOutput / NativeOutput fallback
    prompted_output_template='Always respond with a JSON object ...',
    native_output_requires_schema_in_instructions=False,

    # JSON schema compatibility
    json_schema_transformer=None,           # JsonSchemaTransformer subclass or None

    # Thinking/reasoning
    supports_thinking=False,
    thinking_always_enabled=False,         # True → thinking can't be disabled (o-series, R1)
    thinking_tags=('<think>', '</think>'),  # Delimiter pair for inline think blocks

    # Streaming quirks
    ignore_streamed_leading_whitespace=False,  # Workaround for Ollama + Qwen3 empty parts

    # Native tool type set
    supported_native_tools=frozenset(...), # defaults to ALL AbstractNativeTool subclasses
)
```

### `StructuredOutputMode` values

| Value | What the model does |
|---|---|
| `'tool'` | Output via a special tool call (most compatible) |
| `'json_schema'` | Native JSON schema output (`NativeOutput`) |
| `'json_object'` | JSON-mode without a schema (`PromptedOutput`) |
| `'prompted'` | Inject schema in system prompt as text (`PromptedOutput`) |

### `from_profile()` + `update()` — merging profiles

```python
from pydantic_ai.profiles.anthropic import AnthropicModelProfile

# Build an Anthropic profile that overrides defaults from a base profile:
base = ModelProfile(supports_thinking=True)
anthropic_profile = AnthropicModelProfile.from_profile(base)
# AnthropicModelProfile inherits supports_thinking=True from base

# update() applies non-default fields from a partial profile:
partial = ModelProfile(supports_image_output=True)
merged = anthropic_profile.update(partial)
```

### Deprecated field alias

```python
# 'supported_builtin_tools' was renamed to 'supported_native_tools' in 1.104.0
profile.supported_builtin_tools   # DeprecationWarning; reads supported_native_tools
```

### `DEFAULT_PROFILE`

A module-level singleton with all conservative defaults. Use it as a baseline:

```python
from pydantic_ai.profiles import DEFAULT_PROFILE
print(DEFAULT_PROFILE.supports_thinking)          # False
print(DEFAULT_PROFILE.default_structured_output_mode)  # 'tool'
```

### Custom profile for a private model

```python
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers import Provider

class MyOllamaProfile(ModelProfile):
    pass  # or add Ollama-specific fields

class MyOllamaProvider(Provider[...]):
    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        if 'qwen3' in model_name:
            return MyOllamaProfile(
                supports_thinking=True,
                thinking_tags=('<think>', '</think>'),
                ignore_streamed_leading_whitespace=True,
            )
        return None
```

---

## 6. `AnthropicModelProfile` + `OpenAIModelProfile`

**Modules:** `pydantic_ai.profiles.anthropic`, `pydantic_ai.profiles.openai`  
**Imports:**
```python
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile
```

These are `@dataclass(kw_only=True)` subclasses of `ModelProfile` that add provider-specific fields. All fields are prefixed so they can be safely merged with base profiles from other providers.

### `AnthropicModelProfile` fields

```python
@dataclass(kw_only=True)
class AnthropicModelProfile(ModelProfile):
    anthropic_supports_fast_speed: bool = False
    # True for Claude Opus 4.6, 4.7, 4.8 — enables anthropic_speed='fast'

    anthropic_supports_adaptive_thinking: bool = False
    # True for Sonnet 4.6+, Opus 4.6+
    # When True: thinking → {'type': 'adaptive'}
    # When False: thinking → {'type': 'enabled', 'budget_tokens': N}

    anthropic_supports_effort: bool = False
    # True for Opus 4.5+, Sonnet 4.6+ — maps unified thinking level to output_config.effort

    anthropic_supports_xhigh_effort: bool = False
    # True for Opus 4.7 and 4.8 — 'xhigh' effort value accepted

    anthropic_disallows_budget_thinking: bool = False
    # True for Opus 4.7 and 4.8 — {'type': 'enabled', 'budget_tokens': ...} returns 400

    anthropic_disallows_sampling_settings: bool = False
    # True for Opus 4.7 and 4.8 — temperature/top_p must be omitted

    anthropic_default_code_execution_tool_version: str = '20250825'
    # Used when code_execution_tool_version='auto'
```

Example — reading the profile for a specific model:

```python
from pydantic_ai.providers.anthropic import AnthropicProvider

provider = AnthropicProvider()
profile = provider.model_profile('claude-opus-4-8')
if isinstance(profile, AnthropicModelProfile):
    print(profile.anthropic_supports_adaptive_thinking)   # True
    print(profile.anthropic_disallows_budget_thinking)    # True
    print(profile.anthropic_disallows_sampling_settings)  # True
```

### `OpenAIModelProfile` fields (selected)

```python
@dataclass(kw_only=True)
class OpenAIModelProfile(ModelProfile):
    openai_chat_thinking_field: str | None = None
    # Non-standard field name for reasoning content in Chat Completions API responses.
    # Ollama/newer vLLM use 'reasoning'; DeepSeek/older vLLM use 'reasoning_content'.
    # Must be set when openai_chat_send_back_thinking_parts='field'.

    openai_chat_send_back_thinking_parts: Literal['auto', 'tags', 'field', False] = 'auto'
    # How to include thinking content in requests:
    # 'auto':  auto-detect from ThinkingPart.id / ThinkingPart.provider_name
    # 'tags':  embed in <think>...</think> tags
    # 'field': send in the field named by openai_chat_thinking_field
    # False:   strip all thinking parts from request messages
```

### Custom OpenAI-compatible provider with thinking

```python
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

class MyVLLMProvider(OpenAIProvider):
    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        if 'qwen3' in model_name.lower():
            return OpenAIModelProfile(
                supports_thinking=True,
                openai_chat_thinking_field='reasoning',           # vLLM field name
                openai_chat_send_back_thinking_parts='field',     # roundtrip via field
                ignore_streamed_leading_whitespace=True,
            )
        return None

provider = MyVLLMProvider(base_url='http://localhost:8000/v1', api_key='not-used')
```

---

## 7. `WrapperEmbeddingModel` + `InstrumentedEmbeddingModel`

**Module:** `pydantic_ai.embeddings.wrapper`, `pydantic_ai.embeddings.instrumented`  
**Imports:**
```python
from pydantic_ai.embeddings.wrapper import WrapperEmbeddingModel
from pydantic_ai.embeddings.instrumented import InstrumentedEmbeddingModel
```

### `WrapperEmbeddingModel` — base for custom embedding wrappers

`WrapperEmbeddingModel` delegates all methods to a wrapped `EmbeddingModel` via `__getattr__`. Override specific methods to add caching, logging, or any other cross-cutting concern:

```python
from pydantic_ai.embeddings.wrapper import WrapperEmbeddingModel
from pydantic_ai.embeddings import EmbeddingResult, EmbeddingSettings
from collections.abc import Sequence
import hashlib, json

class CachedEmbeddingModel(WrapperEmbeddingModel):
    """In-memory cache for embedding results to avoid redundant API calls."""

    def __init__(self, wrapped: str):
        super().__init__(wrapped)          # accepts model name string or EmbeddingModel
        self._cache: dict[str, EmbeddingResult] = {}

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        # Stable cache key from inputs + settings
        key = hashlib.sha256(
            json.dumps({'inputs': list([inputs] if isinstance(inputs, str) else inputs),
                        'settings': str(settings)}).encode()
        ).hexdigest()
        if key not in self._cache:
            self._cache[key] = await super().embed(inputs, input_type=input_type, settings=settings)
        return self._cache[key]

# Usage:
cached = CachedEmbeddingModel('openai:text-embedding-3-small')
result = await cached.embed('hello world', input_type='query')
```

### `infer_embedding_model` — string-to-model factory

```python
from pydantic_ai.embeddings import infer_embedding_model

model = infer_embedding_model('openai:text-embedding-3-small')
model = infer_embedding_model('google:gemini-embedding-001')
model = infer_embedding_model('cohere:embed-v4.0')
model = infer_embedding_model('bedrock:amazon.titan-embed-text-v2:0')
```

### `InstrumentedEmbeddingModel` — OTel tracing for embeddings

Wraps any embedding model and emits OpenTelemetry spans for every `embed()` call:

```python
from pydantic_ai.embeddings.instrumented import InstrumentedEmbeddingModel
from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai._instrumentation import InstrumentationSettings
import logfire

logfire.configure()

base = infer_embedding_model('openai:text-embedding-3-small')
instrumented = InstrumentedEmbeddingModel(
    base,
    options=InstrumentationSettings(include_content=True),  # log input texts in span
)

result = await instrumented.embed(['doc1', 'doc2'], input_type='document')
# → Logfire span: "embeddings text-embedding-3-small" with inputs, count, settings
```

The span carries:
- `gen_ai.operation.name = 'embeddings'`
- `gen_ai.request.model` / `gen_ai.response.model`
- `inputs_count` (number of texts)
- `input_type` ('query' or 'document')
- `inputs` (only when `include_content=True`)
- Token usage + cost via `genai-prices`

### Composing wrappers

```python
from pydantic_ai.embeddings.wrapper import WrapperEmbeddingModel
from pydantic_ai.embeddings.instrumented import InstrumentedEmbeddingModel

base = infer_embedding_model('openai:text-embedding-3-small')
cached = CachedEmbeddingModel(base)
traced = InstrumentedEmbeddingModel(cached)  # outermost = first span
```

---

## 8. Additional Embedding Providers

**Module:** `pydantic_ai.embeddings`

Since Vol. 8 covered the base `Embedder` / `EmbeddingModel` / `EmbeddingResult` API, this section documents the four additional embedding providers not covered there.

### Provider comparison

| Provider | Model class | Provider string | Typical models |
|---|---|---|---|
| Google Gemini/Vertex | `GoogleEmbeddingModel` | `'google'` / `'google-cloud'` | `gemini-embedding-001` |
| AWS Bedrock | `BedrockEmbeddingModel` | `'bedrock'` | `amazon.titan-embed-text-v2:0`, `cohere.embed-english-v3` |
| Cohere | `CohereEmbeddingModel` | `'cohere'` | `embed-v4.0`, `embed-multilingual-v3.0` |
| Voyage AI | `VoyageAIEmbeddingModel` | `'voyageai'` | `voyage-3`, `voyage-3-large`, `voyage-code-3` |

### `GoogleEmbeddingModel`

```python
from pydantic_ai.embeddings.google import GoogleEmbeddingModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

# Gemini API (GOOGLE_API_KEY env var)
model = GoogleEmbeddingModel('gemini-embedding-001', provider='google')

# Google Cloud / Vertex AI
model = GoogleEmbeddingModel(
    'gemini-embedding-001',
    provider=GoogleCloudProvider(project='my-project', location='us-central1'),
)

result = await model.embed(['hello', 'world'], input_type='document')
print(result.model_name)   # 'gemini-embedding-001'
```

### `BedrockEmbeddingModel`

Supports both Amazon Titan Embeddings and Cohere Embed on Bedrock:

```python
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Default AWS credential chain
titan = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0')

# Explicit credentials + region
cohere_bedrock = BedrockEmbeddingModel(
    'cohere.embed-english-v3',
    provider=BedrockProvider(
        region_name='us-east-1',
        aws_access_key_id='AKIA...',
        aws_secret_access_key='...',
    ),
)

result = await titan.embed('search query', input_type='query')
```

### `CohereEmbeddingModel`

```python
from pydantic_ai.embeddings.cohere import CohereEmbeddingModel
from pydantic_ai.embeddings import EmbeddingSettings

model = CohereEmbeddingModel('embed-v4.0')   # COHERE_API_KEY env var
result = await model.embed(
    ['Document about machine learning', 'Another document'],
    input_type='document',
    settings=EmbeddingSettings(dimensions=1024),  # truncate to 1024 dims
)
```

### `VoyageAIEmbeddingModel`

Voyage AI specialises in code and domain-specific embeddings:

```python
# Requires: pip install "pydantic-ai-slim[voyageai]"
from pydantic_ai.embeddings.voyageai import VoyageAIEmbeddingModel
from pydantic_ai.embeddings import EmbeddingSettings

code_model = VoyageAIEmbeddingModel('voyage-code-3')   # VOYAGE_API_KEY env var

query_vec = await code_model.embed(
    'how to sort a list in Python', input_type='query'
)
doc_vecs = await code_model.embed(
    ['def sort_list(lst): return sorted(lst)', 'import heapq'],
    input_type='document',
)

# Cosine similarity:
import numpy as np
scores = np.array(doc_vecs.embeddings) @ np.array(query_vec.embeddings[0])
print('Best match index:', scores.argmax())
```

### RAG pipeline with `EmbeddingResult.cost()`

```python
from pydantic_ai.embeddings import infer_embedding_model

model = infer_embedding_model('openai:text-embedding-3-small')

docs = ['Quantum entanglement is ...', 'The French Revolution began ...']
result = await model.embed(docs, input_type='document')
print(f'Embedded {len(docs)} docs, cost: ${result.cost():.6f}')

# Look up a single document by index:
vec = result[0]    # __getitem__ → single embedding vector
```

---

## 9. `BuilderCheckpoint` + `MessagesBuilder` Advanced Patterns

**Module:** `pydantic_ai.ui`  
**Import:** `from pydantic_ai.ui import MessagesBuilder, BuilderCheckpoint`

`MessagesBuilder` constructs a `list[ModelMessage]` incrementally by appending `ModelRequestPart` or `ModelResponsePart` objects. It automatically coalesces consecutive same-type parts into the same message. `BuilderCheckpoint` snapshots the builder state so you can find *which* message was created or extended by a batch of `add()` calls.

### `MessagesBuilder.add()` — auto-coalescing parts

```python
from pydantic_ai.ui import MessagesBuilder
from pydantic_ai.messages import (
    UserPromptPart, TextPart, ToolCallPart, ToolReturnPart
)

builder = MessagesBuilder()
builder.add(UserPromptPart(content='What is the weather?'))
builder.add(TextPart(content='The weather is sunny.'))   # starts a ModelResponse
builder.add(ToolCallPart(tool_name='get_weather', args='{}', tool_call_id='1'))
# ToolCallPart is a ModelResponsePart → appended to same ModelResponse
builder.add(ToolReturnPart(tool_name='get_weather', content='Sunny', tool_call_id='1'))
# ToolReturnPart is a ModelRequestPart → starts new ModelRequest

print(len(builder.messages))   # 3 (request, response, request)
```

### `checkpoint()` + `last_modified()` — message attribution

The pattern is: take a checkpoint before a batch of `add()` calls, then call `last_modified()` to find the concrete message you just built or extended:

```python
from pydantic_ai.messages import ModelResponse

checkpoint = builder.checkpoint()
builder.add(TextPart(content='My conclusion.'))

response = builder.last_modified(checkpoint, of_type=ModelResponse)
# response is the ModelResponse that received the TextPart — either the
# pre-existing tail (if it was already a ModelResponse) or a newly appended one.
if response is not None:
    response.timestamp = datetime.utcnow()   # annotate after building
```

### Custom `UIEventStream` using `MessagesBuilder`

```python
from pydantic_ai.ui import UIEventStream, MessagesBuilder
from pydantic_ai.messages import ModelResponse

class MyEventStream(UIEventStream[...]):
    def __init__(self, run_input, **kwargs):
        super().__init__(run_input, **kwargs)
        self._builder = MessagesBuilder()

    async def on_agent_stream_event(self, event):
        cp = self._builder.checkpoint()
        for part in self._extract_parts(event):
            self._builder.add(part)
        # Find the latest response message to annotate with a run_id:
        resp = self._builder.last_modified(cp, of_type=ModelResponse)
        if resp is not None:
            resp.run_id = self._current_run_id
        async for protocol_event in self._convert(event):
            yield protocol_event
```

### `BuilderCheckpoint` fields

```python
@dataclass
class BuilderCheckpoint:
    message_count: int            # len(builder.messages) at snapshot time
    last_message: ModelMessage | None   # tail message at snapshot time
    last_message_part_count: int  # len(tail.parts) at snapshot time
```

`last_modified()` returns a candidate from *either* new messages (`messages[message_count:]`) *or* the pre-existing tail if its parts list grew. This handles both the "new message created" and "existing message extended" cases in a single call.

---

## 10. `OutlinesModel` — Deprecated Constrained Generation

**Module:** `pydantic_ai.models.outlines`  
**Import:** `from pydantic_ai.models.outlines import OutlinesModel`  
**Status:** `@deprecated` — will be removed in v2.0

<Aside type="caution">
`OutlinesModel` is deprecated in pydantic-ai 1.107.0 and will be removed in v2. If you need constrained generation, file an issue at https://github.com/dottxt-ai/outlines/issues. For production use prefer `NativeOutput` or `PromptedOutput` with API-based models.
</Aside>

`OutlinesModel` wraps a local Outlines model (Transformers, LlamaCpp, SGLang, vLLM Offline, MLX-LM) and applies grammar-constrained decoding so the model is forced to emit valid JSON or regex-matching text. Unlike API models, this constraint is enforced at the token-generation level, not via post-processing or retries.

### Constructor

```python
# Requires: pip install "pydantic-ai-slim[outlines]"
from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel(
    model=outlines_model_instance,    # any Outlines BaseModel / AsyncModel
    provider='outlines',              # or a custom Provider[OutlinesBaseModel]
    profile=None,                     # ModelProfileSpec | None
    settings=None,                    # ModelSettings | None
)
```

### Factory classmethods

| Method | Backend |
|---|---|
| `OutlinesModel.from_transformers(hf_model, tokenizer)` | HuggingFace Transformers |
| `OutlinesModel.from_llamacpp(llama_model)` | llama.cpp |
| `OutlinesModel.from_sglang(sg_model)` | SGLang |
| `OutlinesModel.from_mlxlm(mlx_model, tokenizer)` | Apple MLX (macOS only) |
| `OutlinesModel.from_vllm_offline(vllm_model)` | vLLM offline mode |

### Example: constrained JSON output with a local model

```python
# pip install "pydantic-ai-slim[outlines]" transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai import Agent
from pydantic import BaseModel

class Movie(BaseModel):
    title: str
    year: int
    genre: str

hf_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

outlines_model = OutlinesModel.from_transformers(hf_model, tokenizer)
agent: Agent[None, Movie] = Agent(outlines_model, output_type=Movie)

result = agent.run_sync('Recommend a movie about space travel.')
print(result.output.title)   # Always valid JSON → Movie, no retries needed
```

### Why it was deprecated

1. **API models are better**: `NativeOutput` / `PromptedOutput` on API models (OpenAI, Anthropic, Google) give reliable structured output without local GPU requirements.
2. **Outlines integration complexity**: Grammar-constrained decoding requires tight coupling with the tokenizer and model internals; each Outlines backend update risked breaking pydantic-ai.
3. **Maintenance burden**: Supporting 5 backends (Transformers, LlamaCpp, SGLang, vLLM, MLX) multiplies the test matrix significantly.

### Migration options

| Use case | Recommended alternative |
|---|---|
| Simple structured output | `Agent(model, output_type=MyModel)` with `NativeOutput` or default tool-mode |
| Forced JSON-schema output | `Agent(model, output_type=NativeOutput(MyModel))` with OpenAI/Gemini |
| Local models + structured output | vLLM with structured-output API + `OpenAIProvider(base_url=...)` |
| Token-level constraint (grammar) | Use Outlines directly, pass output to pydantic `model_validate()` |

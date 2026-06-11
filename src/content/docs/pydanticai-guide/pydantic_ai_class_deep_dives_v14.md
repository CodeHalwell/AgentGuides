---
title: "PydanticAI — Class Deep Dives Vol. 14"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.107.0: FallbackModel + ResponseRejected + ExceptionHandler/ResponseHandler/FallbackOn type aliases (multi-model resilience chains with exception and response routing), InstrumentationSettings + InstrumentedModel + instrument_model (OTel/Logfire tracing v1–v5 protocol, event_mode, token histogram, cost histogram), ModelProfile + ModelProfileSpec + AnthropicModelProfile + OpenAIModelProfile (per-model capability flags, .update() merge semantics, profile spec callable), EmbeddingModel ABC + EmbeddingResult + EmbedInputType + EmbeddingSettings (embedding foundation — prepare_embed, token counting, cost tracking, indexing by input text), WrapperEmbeddingModel + InstrumentedEmbeddingModel + TestEmbeddingModel (embedding extension, OTel spans, deterministic testing), CombinedToolset + _CombinedToolsetTool (multi-toolset combination, for_run lifecycle, conflict detection), ApprovalRequiredToolset + DeferredLoadingToolset (HITL approval gate + lazy tool discovery), WrapperToolset + FilteredToolset + RenamedToolset (toolset transformation pipeline — delegation, async filter, name remapping), WrapperCapability + CombinedCapability (capability composition — transparent id delegation, flat-splatting, ordering), AGUIAdapter + AGUIEventStream + DEFAULT_AG_UI_VERSION (AG-UI protocol adapter — dispatch_request, preserve_file_data, multimodal version gating). All verified against pydantic-ai 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 14)"
  order: 40
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.107.0`.
</Aside>

Ten class groups covering the resilience, observability, model profiles, embeddings, toolset composition, and AG-UI adapter surface of `pydantic-ai 1.107.0`: `FallbackModel` (automatic multi-model failover with exception-type and response-handler routing); `InstrumentationSettings` + `InstrumentedModel` (OTel/Logfire v1-v5 protocol, event modes, cost histograms); `ModelProfile` + provider subclasses (capability flags driving request shaping); `EmbeddingModel` + `EmbeddingResult` + `EmbeddingSettings` (embedding foundation); `WrapperEmbeddingModel` + `InstrumentedEmbeddingModel` + `TestEmbeddingModel` (embedding extension patterns); `CombinedToolset` (parallel toolset combination with conflict detection); `ApprovalRequiredToolset` + `DeferredLoadingToolset` (HITL approval gating + lazy discovery); `WrapperToolset` + `FilteredToolset` + `RenamedToolset` (toolset transformation pipeline); `WrapperCapability` + `CombinedCapability` (capability composition internals); `AGUIAdapter` + `AGUIEventStream` (AG-UI streaming adapter with multimodal version gating).

[← Vol. 13](./pydantic_ai_class_deep_dives_v13/)

---

## 1. `FallbackModel` + `ResponseRejected` + type aliases — Multi-Model Resilience

**Module:** `pydantic_ai.models.fallback`  
**Import:** `from pydantic_ai.models.fallback import FallbackModel, ResponseRejected`

`FallbackModel` wraps two or more models and automatically falls back to the next when the current one raises an exception or its response is rejected by a handler. It is the standard way to add fault-tolerance, provider diversity, and cost-tier routing without changing agent code.

### Class signature

```python
@dataclass(init=False)
class FallbackModel(Model):
    models: list[Model]               # All wrapped models (default first)
    _model_name: str                  # repr=False
    _exception_handlers: list[ExceptionHandler]   # repr=False
    _response_handlers: list[ResponseHandler]     # repr=False

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: FallbackOn = (ModelAPIError,),
    ) -> None: ...
```

### Type aliases

```python
ExceptionHandler = Callable[[Exception], Awaitable[bool]] | Callable[[Exception], bool]
ResponseHandler  = Callable[[ModelResponse], Awaitable[bool]] | Callable[[ModelResponse], bool]

FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]
)
```

**Auto-detection rule:** if the first parameter of a callable is type-hinted as `ModelResponse`, it is treated as a `ResponseHandler`; otherwise (including untyped lambdas) it is an `ExceptionHandler`.

### How `_should_fallback` works

```
_should_fallback(value: Exception | ModelResponse) → bool

1. Check isinstance(value, Exception)
   → run _exception_handlers; return True on first truthy result
2. Otherwise (value is ModelResponse)
   → run _response_handlers; return True on first truthy result
3. Return False if no handler fires
```

### Example 1 — Simple exception-type fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

# Try claude-haiku first; fall back to gpt-4o-mini on any ModelAPIError
model = FallbackModel(
    'anthropic:claude-haiku-4-5',
    'openai:gpt-4o-mini',
    fallback_on=(ModelAPIError,),  # default
)

agent = Agent(model)

async def main():
    result = await agent.run('Summarise this article in one sentence.')
    print(result.output)

asyncio.run(main())
```

### Example 2 — Response-handler fallback (reject empty / refusal)

```python
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel

def reject_short_response(response: ModelResponse) -> bool:
    """Reject responses under 10 characters — likely a refusal or error message."""
    text = ''.join(p.content for p in response.parts if isinstance(p, TextPart))
    return len(text.strip()) < 10

model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-opus-4-5',
    fallback_on=reject_short_response,
)
```

### Example 3 — Mixed sequence handler

```python
import httpx
from pydantic_ai.exceptions import ModelAPIError

async def is_rate_limited(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return False

model = FallbackModel(
    'openai:gpt-4o',
    'openai:gpt-4o-mini',
    'anthropic:claude-haiku-4-5',
    fallback_on=[ModelAPIError, is_rate_limited],
)
```

### Example 4 — Three-tier cost routing

```python
from pydantic_ai.models.fallback import FallbackModel

# Premium → Standard → Economy — all tried in order on any API error
model = FallbackModel(
    'anthropic:claude-opus-4-8',   # Premium
    'anthropic:claude-sonnet-4-6', # Standard
    'anthropic:claude-haiku-4-5',  # Economy
)
```

### `ResponseRejected` exception

```python
class ResponseRejected(Exception):
    def __init__(self, rejected_count: int):
        super().__init__(f'{rejected_count} model response(s) rejected by fallback_on handler')
```

When all models' responses are rejected by a response handler, a `FallbackExceptionGroup` is raised that includes a `ResponseRejected` instance. It is **not** raised directly — always surfaces inside the group.

### Key properties

| Property | Description |
|---|---|
| `model_name` | `'fallback:<m1>,<m2>,...'` — composite string |
| `model_id` | `'fallback:<id1>,<id2>,...'` |
| `system` | `'fallback:<sys1>,<sys2>,...'` |
| `base_url` | Returns `self.models[0].base_url` |
| `profile` | Raises `NotImplementedError` — no own profile |

`FallbackModel` defers `prepare_messages` to each inner model's own call so per-model profiles apply correctly.

### `__aenter__` / `__aexit__` — shared lifecycle

All inner models share a single `AsyncExitStack` behind an `anyio.Lock`. The lock prevents double-entry on concurrent first uses. HTTP client lifecycles are properly managed even in high-concurrency scenarios.

---

## 2. `InstrumentationSettings` + `InstrumentedModel` + `instrument_model` — OTel / Logfire

**Module:** `pydantic_ai.models.instrumented`  
**Import:** `from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel, instrument_model`

`InstrumentedModel` wraps any `Model` to emit OpenTelemetry spans and metrics for every request. `InstrumentationSettings` configures what is emitted and which OTel protocol version to use.

### `InstrumentationSettings` signature

```python
@dataclass(init=False)
class InstrumentationSettings:
    tracer: Tracer                                    # repr=False
    logger: Logger                                    # repr=False
    event_mode: Literal['attributes', 'logs'] = 'attributes'
    include_binary_content: bool = True
    include_content: bool = True
    version: Literal[1, 2, 3, 4, 5] = DEFAULT_INSTRUMENTATION_VERSION
    use_aggregated_usage_attribute_names: bool = False

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        include_binary_content: bool = True,
        include_content: bool = True,
        version: Literal[1, 2, 3, 4, 5] = DEFAULT_INSTRUMENTATION_VERSION,
        event_mode: Literal['attributes', 'logs'] = 'attributes',
        logger_provider: LoggerProvider | None = None,
        use_aggregated_usage_attribute_names: bool = False,
    ) -> None: ...
```

### Protocol version reference table

| Version | Description |
|---|---|
| 1 | Legacy event-based OTel GenAI spec. Deprecated. `event_mode` and `logger_provider` only apply here. |
| 2 | Newer OTel GenAI spec. Instructions in `gen_ai.system_instructions`, I/O in `gen_ai.input.messages` / `gen_ai.output.messages`, all messages in `pydantic_ai.all_messages`. |
| 3 | v2 + thinking token support. |
| 4 | v3 + GenAI semantic conventions for multimodal content (`type='uri'` / `type='blob'` fields). |
| 5 | v4 + `CallDeferred` / `ApprovalRequired` exceptions no longer set span status to ERROR (they are control flow, not errors). |

### Metrics created by `InstrumentationSettings.__init__`

```python
# Token usage histogram (OTel GenAI spec)
tokens_histogram = meter.create_histogram(
    name='gen_ai.client.token.usage',
    unit='{token}',
    description='Measures number of input and output tokens used',
    explicit_bucket_boundaries_advisory=TOKEN_HISTOGRAM_BOUNDARIES,
)

# Cost histogram (custom pydantic-ai metric)
cost_histogram = meter.create_histogram(
    'operation.cost',
    unit='{USD}',
    description='Monetary cost',
)
```

### `InstrumentedModel` class

```python
@dataclass(init=False)
class InstrumentedModel(WrapperModel):
    instrumentation_settings: InstrumentationSettings

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        options: InstrumentationSettings | None = None,
    ) -> None: ...
```

### Example 1 — Global instrumentation via Logfire

```python
import logfire
from pydantic_ai import Agent

logfire.configure()  # sets global OTel providers

# Agent.instrument_all() wraps all models globally
Agent.instrument_all()

agent = Agent('openai:gpt-4o')

async def main():
    # Every agent.run() now emits OTel spans to Logfire
    result = await agent.run('What is 2 + 2?')
    print(result.output)  # '4'
```

### Example 2 — Per-agent instrumentation with custom settings

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

settings = InstrumentationSettings(
    include_content=False,          # redact prompts and completions
    include_binary_content=False,   # redact images, audio
    version=5,                      # latest protocol
    use_aggregated_usage_attribute_names=True,  # avoids double-counting
)

agent = Agent('anthropic:claude-haiku-4-5', instrument=settings)
```

### Example 3 — Wrapping a model directly

```python
from pydantic_ai.models.instrumented import instrument_model, InstrumentationSettings
from pydantic_ai.models.openai import OpenAIChatModel

base = OpenAIChatModel('gpt-4o-mini')
instrumented = instrument_model(base, instrument=InstrumentationSettings(version=4))

# instrumented is an InstrumentedModel wrapping base
assert isinstance(instrumented.wrapped, OpenAIChatModel)
```

### Example 4 — Custom OTLP exporter (non-Logfire)

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pydantic_ai.models.instrumented import InstrumentationSettings

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint='http://jaeger:4318/v1/traces')))

settings = InstrumentationSettings(tracer_provider=provider, version=4)
```

### `messages_to_otel_events` helper

```python
settings = InstrumentationSettings()
events = settings.messages_to_otel_events(messages, parameters=model_request_parameters)
# Returns list[LogRecord] — useful for version=1 log-mode or custom exporters
```

---

## 3. `ModelProfile` + `AnthropicModelProfile` + `OpenAIModelProfile` — Per-Model Capability Flags

**Module:** `pydantic_ai.profiles`  
**Import:** `from pydantic_ai.profiles import ModelProfile, ModelProfileSpec`

`ModelProfile` is a dataclass that encodes which capabilities a specific model supports — JSON schema structured output, native tools, thinking, inline system prompts — so the framework can adapt requests and responses without provider-specific conditionals scattered through agent code.

### `ModelProfile` full signature

```python
@dataclass(kw_only=True)
class ModelProfile:
    supports_tools: bool = True
    supports_tool_return_schema: bool = False
    supports_json_schema_output: bool = False
    supports_json_object_output: bool = False
    supports_image_output: bool = False
    supports_inline_system_prompts: bool = False
    default_structured_output_mode: StructuredOutputMode = 'tool'
    prompted_output_template: str = '...'     # contains '{schema}' placeholder
    native_output_requires_schema_in_instructions: bool = False
    json_schema_transformer: type[JsonSchemaTransformer] | None = None
    supports_thinking: bool = False
    thinking_always_enabled: bool = False
    thinking_tags: tuple[str, str] = ('<think>', '</think>')
    ignore_streamed_leading_whitespace: bool = False
    supported_native_tools: frozenset[type[AbstractNativeTool]] = field(
        default_factory=lambda: SUPPORTED_NATIVE_TOOLS
    )
```

### `ModelProfileSpec` type alias

```python
ModelProfileSpec = ModelProfile | Callable[[str], ModelProfile | None]
```

A `ModelProfileSpec` is either a direct profile instance or a factory function that takes the model name string and returns a profile (or `None` to fall back to the model's default).

### `.update()` merge semantics

```python
def update(self, profile: ModelProfile | None) -> Self:
    """Overlay non-default values from another profile onto this one."""
    ...
```

`update()` only copies fields whose value differs from the base `ModelProfile` default. This allows provider profiles to be stacked:

```python
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.anthropic import AnthropicModelProfile

# Build a custom Anthropic profile that forces no thinking
base = AnthropicModelProfile()
overridden = base.update(ModelProfile(supports_thinking=False))
```

### `AnthropicModelProfile`

```python
from pydantic_ai.profiles.anthropic import AnthropicModelProfile

@dataclass(kw_only=True)
class AnthropicModelProfile(ModelProfile):
    anthropic_supports_fast_speed: bool = False
    anthropic_supports_adaptive_thinking: bool = False   # Sonnet 4.6+, Opus 4.6+
    anthropic_supports_effort: bool = False              # Opus 4.5+, Sonnet 4.6+
    anthropic_supports_xhigh_effort: bool = False        # Opus 4.7+
    anthropic_disallows_budget_thinking: bool = False    # Opus 4.7+
    anthropic_disallows_sampling_settings: bool = False  # Opus 4.7+
    anthropic_code_execution_tool_version: AnthropicCodeExecutionToolVersion = '20250825'
    # ...
```

### `OpenAIModelProfile`

```python
from pydantic_ai.profiles.openai import OpenAIModelProfile

@dataclass(kw_only=True)
class OpenAIModelProfile(ModelProfile):
    openai_chat_thinking_field: str | None = None
    openai_chat_send_back_thinking_parts: Literal['auto', 'tags', 'field', False] = 'auto'
    openai_supports_strict_tool_definition: bool = False
    openai_json_schema_transformer: type[JsonSchemaTransformer] | None = None
    openai_reasoning_effort: str | None = None
    openai_system_prompt_role: OpenAISystemPromptRole = 'system'
    openai_image_detail: Literal['auto', 'low', 'high'] | None = None
    openai_supports_audio: bool = False
    openai_disable_parallel_tool_calls: bool = False
    openai_max_output_tokens: int | None = None
    # ...
```

### Example 1 — Override structured output mode per-model

```python
from pydantic_ai import Agent
from pydantic_ai.profiles import ModelProfile

# Force JSON-mode output for a model that doesn't support native schema
profile = ModelProfile(
    supports_json_schema_output=False,
    supports_json_object_output=True,
    default_structured_output_mode='json',
)

agent = Agent('openai:gpt-4o-mini', model_profile=profile)
```

### Example 2 — Profile spec callable for dynamic profiles

```python
from pydantic_ai.profiles import ModelProfile, ModelProfileSpec
from pydantic_ai.profiles.openai import OpenAIModelProfile

def my_profile_spec(model_name: str) -> ModelProfile | None:
    """Apply stricter profile to o-series models."""
    if model_name.startswith('o') or '-o3' in model_name:
        return OpenAIModelProfile(
            openai_reasoning_effort='high',
            supports_thinking=True,
            thinking_always_enabled=True,
        )
    return None  # use model default

agent = Agent('openai:o3', model_profile=my_profile_spec)
```

### Example 3 — Custom provider profile for a third-party OpenAI-compatible API

```python
from pydantic_ai.profiles.openai import OpenAIModelProfile

# Profile for a Qwen3 model served via Ollama
qwen3_profile = OpenAIModelProfile(
    supports_thinking=True,
    thinking_tags=('<think>', '</think>'),
    ignore_streamed_leading_whitespace=True,
    openai_chat_thinking_field='reasoning',
    openai_chat_send_back_thinking_parts='field',
)
```

---

## 4. `EmbeddingModel` + `EmbeddingResult` + `EmbeddingSettings` — Embedding Foundation

**Module:** `pydantic_ai.embeddings`  
**Import:** `from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, EmbeddingSettings, EmbedInputType`

### `EmbeddingModel` ABC

```python
class EmbeddingModel(ABC):
    _settings: EmbeddingSettings | None = None

    def __init__(self, *, settings: EmbeddingSettings | None = None) -> None: ...

    @property
    def base_url(self) -> str | None: ...  # optional

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def system(self) -> str: ...

    @abstractmethod
    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult: ...

    def prepare_embed(
        self, inputs: str | Sequence[str], settings: EmbeddingSettings | None = None
    ) -> tuple[list[str], EmbeddingSettings]: ...

    async def max_input_tokens(self) -> int | None: ...
    async def count_tokens(self, text: str) -> int: ...
```

### `EmbedInputType`

```python
EmbedInputType = Literal['query', 'document']
```

Some models (e.g. Cohere) optimize differently for queries vs. documents. Always pass `'query'` for search queries and `'document'` for texts being stored and searched against.

### `EmbeddingResult` fields

```python
@dataclass
class EmbeddingResult:
    embeddings: Sequence[Sequence[float]]  # one vector per input
    inputs: list[str]                      # original input texts (populated by prepare_embed)
    model: str                             # model name reported by provider
    usage: RequestUsage                    # input_tokens, etc.
    provider_name: str
    provider_url: str | None = None
    timestamp: datetime = field(default_factory=now_utc)

    def __getitem__(self, key: int | str) -> Sequence[float]:
        """Access embedding by index (int) or by original input text (str)."""
        ...
```

### `EmbeddingSettings`

```python
@dataclass
class EmbeddingSettings:
    dimensions: int | None = None
    truncation: bool | None = None
    timeout: float | None = None
    concurrent_requests: int | None = None
```

### Example 1 — Basic embedding via `Embedder` facade

```python
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

async def main():
    result = await embedder.embed_query('What is pydantic-ai?')
    vec = result['What is pydantic-ai?']  # index by input text
    print(len(vec))     # 1536
    print(result.usage.input_tokens)

asyncio.run(main())
```

### Example 2 — Batch document embedding

```python
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

async def main():
    docs = [
        'PydanticAI is an agent framework.',
        'LangGraph is a workflow framework.',
        'CrewAI is a multi-agent framework.',
    ]
    result = await embedder.embed_documents(docs)
    for doc, vec in zip(docs, result.embeddings):
        print(f'{doc[:30]:32s} → {len(vec)}-dim vector')

asyncio.run(main())
```

### Example 3 — Custom dimensions and truncation

```python
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

embedder = Embedder('openai:text-embedding-3-large')

async def main():
    result = await embedder.embed_query(
        'Semantic search query',
        settings=EmbeddingSettings(dimensions=512, truncation=True),
    )
    print(len(result.embeddings[0]))  # 512
```

### Example 4 — Custom `EmbeddingModel` implementation

```python
import asyncio
from collections.abc import Sequence
from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, EmbedInputType, EmbeddingSettings
from pydantic_ai.usage import RequestUsage

class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Local embedding model using a Sentence Transformer."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__()
        self._model_name = model_name
        # self._model = SentenceTransformer(model_name)  # lazy-load in production

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return 'sentence-transformers'

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        prepared_inputs, merged_settings = self.prepare_embed(inputs, settings)
        # vecs = self._model.encode(prepared_inputs).tolist()
        vecs = [[0.1, 0.2, 0.3] for _ in prepared_inputs]  # placeholder
        return EmbeddingResult(
            embeddings=vecs,
            inputs=prepared_inputs,
            model=self._model_name,
            usage=RequestUsage(input_tokens=sum(len(t.split()) for t in prepared_inputs)),
            provider_name='sentence-transformers',
        )
```

---

## 5. `WrapperEmbeddingModel` + `InstrumentedEmbeddingModel` + `TestEmbeddingModel` — Embedding Extension Patterns

**Module:** `pydantic_ai.embeddings`  
**Import:** `from pydantic_ai.embeddings import WrapperEmbeddingModel, InstrumentedEmbeddingModel, TestEmbeddingModel`

### `WrapperEmbeddingModel`

```python
@dataclass(init=False)
class WrapperEmbeddingModel(EmbeddingModel):
    wrapped: EmbeddingModel

    def __init__(self, wrapped: EmbeddingModel | str) -> None:
        # accepts model name string for convenience
        ...
```

Delegates all methods to `wrapped` by default. Override specific methods to add caching, rate-limiting, or logging.

### Example 1 — Caching wrapper

```python
import hashlib
from pydantic_ai.embeddings import WrapperEmbeddingModel, EmbeddingResult, EmbedInputType, EmbeddingSettings

class CachingEmbeddingModel(WrapperEmbeddingModel):
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self._cache: dict[str, EmbeddingResult] = {}

    async def embed(
        self,
        inputs,
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        key = hashlib.sha256(str((inputs, input_type)).encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = await super().embed(inputs, input_type=input_type, settings=settings)
        return self._cache[key]
```

### `InstrumentedEmbeddingModel`

```python
@dataclass(init=False)
class InstrumentedEmbeddingModel(WrapperEmbeddingModel):
    instrumentation_settings: InstrumentationSettings

    def __init__(
        self,
        wrapped: EmbeddingModel | str,
        options: InstrumentationSettings | None = None,
    ) -> None: ...
```

When `embed()` is called, `_instrument()` opens an OTel span named `'embeddings <model_name>'` with attributes:

| Attribute | Value |
|---|---|
| `gen_ai.operation.name` | `'embeddings'` |
| `gen_ai.request.model` | model name |
| `input_type` | `'query'` or `'document'` |
| `inputs_count` | number of inputs |
| `inputs` | JSON array of input strings (if `include_content=True`) |

### Example 2 — OTel-instrumented embedder with custom provider

```python
import logfire
from pydantic_ai import Embedder
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.embeddings.instrumented import instrument_embedding_model

logfire.configure()

embedder = Embedder('cohere:embed-v4.0', instrument=True)

async def main():
    result = await embedder.embed_query('semantic search')
    print(result.model)  # 'embed-v4.0'
```

### `TestEmbeddingModel`

```python
@dataclass(init=False)
class TestEmbeddingModel(EmbeddingModel):
    __test__ = False   # prevents pytest collection

    _model_name: str
    _provider_name: str
    _dimensions: int
    last_settings: EmbeddingSettings | None
    # ...

    def __init__(
        self,
        model_name: str = 'test',
        *,
        provider_name: str = 'test',
        dimensions: int = 3,
    ) -> None: ...
```

Returns `[1.0, 1.0, ..., 1.0]` vectors of `dimensions` length for every input. The `last_settings` attribute captures the merged settings from the most recent call for assertion in tests.

### Example 3 — Unit-testing an embedding pipeline

```python
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

async def build_index(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    result = await embedder.embed_documents(texts)
    return [list(v) for v in result.embeddings]

async def test_build_index():
    test_model = TestEmbeddingModel(dimensions=4)
    embedder = Embedder('openai:text-embedding-3-small')

    with embedder.override(model=test_model):
        index = await build_index(embedder, ['doc1', 'doc2'])

    assert len(index) == 2
    assert index[0] == [1.0, 1.0, 1.0, 1.0]
    assert test_model.last_settings is not None

asyncio.run(test_build_index())
```

---

## 6. `CombinedToolset` + `_CombinedToolsetTool` — Multi-Toolset Combination

**Module:** `pydantic_ai.toolsets.combined`  
**Import:** `from pydantic_ai.toolsets import CombinedToolset`

`CombinedToolset` merges multiple toolsets into one, making all their tools available to the agent simultaneously. Internally it tracks each tool's source via `_CombinedToolsetTool`.

### Class signature

```python
@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    toolsets: Sequence[AbstractToolset[AgentDepsT]]
    _exit_stack: AsyncExitStack | None = field(init=False, default=None)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]: ...
    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]: ...
    async def get_instructions(...): ...
    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]: ...
    async def call_tool(...): ...
```

### `_CombinedToolsetTool` internals

```python
@dataclass(kw_only=True)
class _CombinedToolsetTool(ToolsetTool[AgentDepsT]):
    source_toolset: AbstractToolset[AgentDepsT]
    source_tool: ToolsetTool[AgentDepsT]
```

Each tool is wrapped with its source toolset reference so `call_tool` routes dispatches correctly regardless of how many toolsets share the same tool name.

### Lifecycle

```
__aenter__  → enters each inner toolset's async context in sequence
for_run     → calls for_run on each toolset and returns new CombinedToolset if any changed
for_run_step → calls for_run_step on each toolset; returns self if none changed (identity check)
get_instructions → concatenates instructions from all toolsets as a flat list
get_tools   → iterates toolsets in order; later tools silently shadow same-named earlier ones
call_tool   → dispatches to the source_toolset.call_tool of the matched _CombinedToolsetTool
__aexit__   → closes the shared exit_stack, which exits all toolsets in LIFO order
```

### Example 1 — Combining two toolsets

```python
from pydantic_ai import Agent, FunctionToolset

search_tools = FunctionToolset()
math_tools = FunctionToolset()

@search_tools.tool_plain
def web_search(query: str) -> str:
    return f'results for: {query}'

@math_tools.tool_plain
def calculate(expression: str) -> float:
    return eval(expression)  # noqa: S307 — demo only

combined = search_tools + math_tools  # uses CombinedToolset under the hood

agent = Agent('openai:gpt-4o', toolsets=[combined])
```

### Example 2 — `+` operator shorthand

```python
from pydantic_ai.toolsets import FunctionToolset, CombinedToolset

# Explicit
combined = CombinedToolset([toolset_a, toolset_b, toolset_c])

# Using + operator (same result)
combined = toolset_a + toolset_b + toolset_c
```

### Example 3 — Inspecting tool sources at runtime

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.toolsets.combined import CombinedToolset

ts_a = FunctionToolset()
ts_b = FunctionToolset()

@ts_a.tool_plain
def greet(name: str) -> str:
    return f'Hello, {name}!'

@ts_b.tool_plain
def farewell(name: str) -> str:
    return f'Goodbye, {name}!'

combined = ts_a + ts_b
assert isinstance(combined, CombinedToolset)
print(combined.label)  # 'CombinedToolset(FunctionToolset, FunctionToolset)'
```

---

## 7. `ApprovalRequiredToolset` + `DeferredLoadingToolset` — HITL Approval Gate + Lazy Discovery

**Module:** `pydantic_ai.toolsets.approval_required`, `pydantic_ai.toolsets.deferred_loading`

### `ApprovalRequiredToolset`

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if not ctx.tool_call_approved and self.approval_required_func(ctx, tool.tool_def, tool_args):
            raise ApprovalRequired
        return await super().call_tool(name, tool_args, ctx, tool)
```

When `approval_required_func` returns `True` and the call has not been approved (`ctx.tool_call_approved == False`), the tool call raises `ApprovalRequired` which pauses the run and surfaces the pending call to external code. Resume by calling `agent.run()` again with `deferred_tool_results` containing the approved result.

### `ApprovalRequired` exception flow

```
LLM calls tool → ApprovalRequiredToolset.call_tool()
                 → approval_required_func(ctx, tool_def, tool_args) → True
                 → raise ApprovalRequired
                 → agent run pauses; DeferredToolRequests captured
                 
Human reviews args → approves
                 → re-run agent with deferred_tool_results containing approval
                 → ctx.tool_call_approved == True
                 → call_tool proceeds normally
```

### Example 1 — Gate all tool calls

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import ApprovalRequiredToolset

tools = FunctionToolset()

@tools.tool_plain
def delete_record(record_id: int) -> str:
    return f'Deleted record {record_id}'

# Wrap with approval required for all calls
gated = ApprovalRequiredToolset(tools)

agent = Agent('openai:gpt-4o', toolsets=[gated])
```

### Example 2 — Selective approval by tool name

```python
from pydantic_ai.toolsets import ApprovalRequiredToolset
from pydantic_ai.tools import ToolDefinition, RunContext

HIGH_RISK_TOOLS = {'delete_record', 'send_email', 'transfer_funds'}

def requires_approval(ctx: RunContext, tool_def: ToolDefinition, args: dict) -> bool:
    return tool_def.name in HIGH_RISK_TOOLS

gated = ApprovalRequiredToolset(tools, approval_required_func=requires_approval)
```

### `DeferredLoadingToolset`

```python
@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)
    tool_names: frozenset[str] | None = None

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_names: frozenset[str] | None = None,
    ) -> None:
        ...
        async def _mark_deferred(ctx, tool_defs):
            return [
                replace(td, defer_loading=True)
                if (tool_names is None or td.name in tool_names)
                else td
                for td in tool_defs
            ]
```

`DeferredLoadingToolset` marks tools with `defer_loading=True`, hiding them from the initial tool list. They are discovered lazily via tool search (e.g. when the agent uses `ToolSearch`). `tool_names=None` defers all tools; pass a frozenset to defer only specific tools.

### Example 3 — Deferred large toolset with tool search

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset, DeferredLoadingToolset
from pydantic_ai.capabilities import ToolSearch

# 100+ tool library — don't send all 100 definitions in every request
big_library = FunctionToolset()  # populated with many tools

deferred = DeferredLoadingToolset(
    big_library,
    tool_names=frozenset({'rare_tool_1', 'rare_tool_2', 'rare_tool_3'}),
)

# ToolSearch capability discovers deferred tools by keyword
agent = Agent('openai:gpt-4o', capabilities=[ToolSearch()], toolsets=[deferred])
```

### Example 4 — Combining approval gate and deferred loading

```python
from pydantic_ai.toolsets import DeferredLoadingToolset, ApprovalRequiredToolset, FunctionToolset

base_tools = FunctionToolset()
# ... register tools ...

# Stack: base → deferred (hide expensive tools) → approval (gate all calls)
pipeline = ApprovalRequiredToolset(DeferredLoadingToolset(base_tools))
```

---

## 8. `WrapperToolset` + `FilteredToolset` + `RenamedToolset` — Toolset Transformation Pipeline

**Module:** `pydantic_ai.toolsets`  
**Import:** `from pydantic_ai.toolsets import WrapperToolset, FilteredToolset, RenamedToolset`

### `WrapperToolset`

The delegation base class for all single-toolset wrappers:

```python
@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    wrapped: AbstractToolset[AgentDepsT]

    # Delegates for_run, for_run_step, __aenter__, __aexit__,
    # get_instructions, get_tools, call_tool, apply, visit_and_replace
    # to self.wrapped. Override any method to change behaviour.
```

`for_run` and `for_run_step` return `self` unchanged when `wrapped.for_run` returns the same object (identity check), avoiding unnecessary allocation.

`apply(visitor)` and `visit_and_replace(visitor)` walk the toolset tree recursively — used by `Agent` internals for schema transformations.

### `FilteredToolset`

```python
@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    filter_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition],
        bool | Awaitable[bool]
    ]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        result = {}
        for name, tool in (await super().get_tools(ctx)).items():
            match = self.filter_func(ctx, tool.tool_def)
            if inspect.isawaitable(match):
                match = await match
            if match:
                result[name] = tool
        return result
```

Both sync and async filter functions are accepted. The filter function receives the `RunContext` (with `deps`) and the `ToolDefinition`.

### Example 1 — Role-based tool visibility

```python
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import FilteredToolset
from pydantic_ai.tools import RunContext, ToolDefinition

@dataclass
class UserDeps:
    role: str  # 'admin' | 'user' | 'guest'

ADMIN_ONLY_TOOLS = {'delete_user', 'impersonate', 'view_audit_log'}

tools = FunctionToolset[UserDeps]()

@tools.tool_plain
def delete_user(user_id: int) -> str: return f'Deleted {user_id}'

@tools.tool_plain
def view_profile(user_id: int) -> str: return f'Profile of {user_id}'

def role_filter(ctx: RunContext[UserDeps], tool_def: ToolDefinition) -> bool:
    if tool_def.name in ADMIN_ONLY_TOOLS:
        return ctx.deps.role == 'admin'
    return True

filtered = FilteredToolset(tools, filter_func=role_filter)

agent = Agent('openai:gpt-4o', deps_type=UserDeps, toolsets=[filtered])
```

### `RenamedToolset`

```python
@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    name_map: dict[str, str]  # new_name → original_name

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        # Builds reverse map (original → new), renames matching tools
        ...

    async def call_tool(self, name: str, tool_args: dict, ctx, tool) -> Any:
        original_name = self.name_map.get(name, name)
        ctx = replace(ctx, tool_name=original_name)
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)
```

Tools absent from `name_map` keep their original names. `call_tool` restores the original name before dispatching so internal tool handlers receive the name they were registered with.

### Example 2 — Namespace collision prevention

```python
from pydantic_ai.toolsets import RenamedToolset

# Two toolsets both have a 'search' tool
web_tools = FunctionToolset()
db_tools = FunctionToolset()

@web_tools.tool_plain
def search(query: str) -> str: return f'web: {query}'

@db_tools.tool_plain
def search(query: str) -> str: return f'db: {query}'  # noqa: F811

# Rename before combining
renamed_web = RenamedToolset(web_tools, name_map={'web_search': 'search'})
renamed_db  = RenamedToolset(db_tools,  name_map={'db_search': 'search'})

combined = renamed_web + renamed_db
```

### Example 3 — Transformation pipeline: filter → rename → defer

```python
from pydantic_ai.toolsets import FilteredToolset, RenamedToolset, DeferredLoadingToolset

base = FunctionToolset()
# ... register many tools ...

pipeline = (
    DeferredLoadingToolset(           # hide slow tools
        RenamedToolset(               # add namespace prefix
            FilteredToolset(base, filter_func=lambda ctx, td: True),
            name_map={'ns_slow_tool': 'slow_tool'},
        ),
        tool_names=frozenset({'ns_slow_tool'}),
    )
)
```

---

## 9. `WrapperCapability` + `CombinedCapability` — Capability Composition Internals

**Module:** `pydantic_ai.capabilities.wrapper`, `pydantic_ai.capabilities.combined`  
**Import:** `from pydantic_ai.capabilities import WrapperCapability, CombinedCapability`

### `WrapperCapability`

```python
@dataclass
class WrapperCapability(AbstractCapability[AgentDepsT]):
    wrapped: AbstractCapability[AgentDepsT]

    def __post_init__(self) -> None:
        # Transparently adopts wrapped.id and wrapped.defer_loading
        # unless the subclass sets its own explicit id.
        ...
```

`WrapperCapability` delegates all hook methods to `wrapped`. Override only what needs to change. The transparent `id` and `defer_loading` adoption means a wrapper over a deferred capability preserves its deferral state.

### Example 1 — Audit logging wrapper

```python
import time
from pydantic_ai.capabilities import WrapperCapability
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapModelRequestHandler

class AuditCapability(WrapperCapability):
    async def wrap_model_request(
        self,
        ctx,
        request,
        handler: WrapModelRequestHandler,
    ):
        start = time.monotonic()
        response = await handler(ctx, request)
        elapsed = time.monotonic() - start
        print(f'[audit] {ctx.run_id}: {elapsed:.3f}s')
        return response
```

### `CombinedCapability` internals

```python
@dataclass
class CombinedCapability(AbstractCapability[AgentDepsT]):
    capabilities: Sequence[AbstractCapability[AgentDepsT]]

    def __post_init__(self) -> None:
        # Flat-splatting: nested CombinedCapability contents are inlined
        # so all leaves participate as siblings in the ordering pass.
        flat: list[AbstractCapability[AgentDepsT]] = []
        for cap in self.capabilities:
            if isinstance(cap, CombinedCapability):
                flat.extend(cap.capabilities)
            else:
                flat.append(cap)
        object.__setattr__(self, 'capabilities', flat)
```

**Flat-splatting** is critical for ordering: a `CombinedCapability` whose leaves span `outermost` and `innermost` tiers would cause ordering conflicts if left nested. By promoting all leaves to siblings, `_effective_ordering` can sort them correctly.

### `+` operator on capabilities

```python
from pydantic_ai.capabilities import Thinking, Hooks, WebSearch

# All return CombinedCapability
combined = Thinking() + Hooks() + WebSearch()

# Equivalent to:
from pydantic_ai.capabilities import CombinedCapability
combined = CombinedCapability([Thinking(), Hooks(), WebSearch()])
```

### Example 2 — Ordering-aware capability stack

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, Hooks, WebSearch
from pydantic_ai.capabilities.thinking import Thinking

class LatencyHooks(Hooks): ...   # custom hooks capability

# Capabilities are ordered by their internal CapabilityOrdering position.
# Combining them lets the framework pick the right injection order.
agent = Agent(
    'anthropic:claude-opus-4-8',
    capabilities=[Thinking(effort='high') + LatencyHooks() + WebSearch()],
)
```

### Example 3 — Wrapping a combined capability

```python
from pydantic_ai.capabilities import WrapperCapability, CombinedCapability, WebSearch, WebFetch

class RedactingCapability(WrapperCapability):
    async def wrap_tool_execute(self, ctx, tool_args, handler):
        if 'api_key' in tool_args:
            tool_args = {**tool_args, 'api_key': '***'}
        return await handler(ctx, tool_args)

web_cap = WebSearch() + WebFetch()  # CombinedCapability
redacted = RedactingCapability(wrapped=web_cap)  # wraps the combination
```

---

## 10. `AGUIAdapter` + `AGUIEventStream` — AG-UI Protocol Adapter

**Module:** `pydantic_ai.ui.ag_ui`  
**Import:** `from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream, DEFAULT_AG_UI_VERSION`

`AGUIAdapter` bridges pydantic-ai agents to the [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) — an open, event-based streaming standard for generative UI frontends. It converts AG-UI `RunAgentInput` to agent invocations and emits AG-UI events as Server-Sent Events.

<Aside type="note">
`pydantic_ai.ag_ui` (no `ui.`) was deprecated in 1.98.0 and will be removed in 2.0. Use `pydantic_ai.ui.ag_ui` instead.
</Aside>

### Class overview

```python
@dataclass(kw_only=True)
class AGUIAdapter(UIAdapter[AgentDepsT]):
    agent: AbstractAgent[AgentDepsT, Any]
    run_input: RunAgentInput
    accept: str = SSE_CONTENT_TYPE
    ag_ui_version: str = DEFAULT_AG_UI_VERSION
    preserve_file_data: bool = False
    manage_system_prompt: Literal['server', 'client'] = 'server'
    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'})
    allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset()

    @classmethod
    async def dispatch_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, Any],
        deps: AgentDepsT = None,
        output_type: OutputSpec | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        model: Model | KnownModelName | str | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        toolsets: Sequence[AbstractToolset] | None = None,
        on_complete: OnCompleteFunc | None = None,
        # ... many more kwargs
    ) -> Response: ...
```

### `DEFAULT_AG_UI_VERSION`

```python
DEFAULT_AG_UI_VERSION = '0.0.7'
```

The version string controls which AG-UI protocol features are emitted:

| Version threshold | Feature |
|---|---|
| >= `REASONING_VERSION` (0.0.5) | ThinkingPart → `ReasoningMessage` events |
| >= `MULTIMODAL_VERSION` (0.1.15) | ImageUrl/VideoUrl/AudioUrl/DocumentUrl → typed multimodal input content |

### Example 1 — Minimal FastAPI AG-UI endpoint

```python
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

app = FastAPI()
agent = Agent('openai:gpt-4o', system_prompt='You are a helpful assistant.')

@app.post('/agent')
async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent)
```

### Example 2 — AG-UI with typed dependencies

```python
from dataclasses import dataclass
from fastapi import FastAPI, Depends
from starlette.requests import Request
from starlette.responses import Response
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

@dataclass
class AppDeps:
    user_id: str
    db_session: object  # your DB type

agent = Agent('anthropic:claude-haiku-4-5', deps_type=AppDeps)

@agent.system_prompt
def get_system_prompt(ctx) -> str:
    return f'You are helping user {ctx.deps.user_id}.'

@app.post('/chat')
async def chat(request: Request, user: str = Depends(get_current_user)) -> Response:
    return await AGUIAdapter.dispatch_request(
        request,
        agent=agent,
        deps=AppDeps(user_id=user, db_session=db),
    )
```

### Example 3 — Streaming via `run_stream` + `encode_stream`

```python
from ag_ui.core.types import RunAgentInput
from pydantic_ai.ui.ag_ui import AGUIAdapter

adapter = AGUIAdapter(
    agent=agent,
    run_input=run_input,  # RunAgentInput from the HTTP body
    ag_ui_version='0.1.15',  # enable multimodal content
    preserve_file_data=True,
)

# Returns an async iterator of SSE-encoded strings
async def stream_events():
    async for chunk in adapter.encode_stream(
        adapter.run_stream(
            output_type=None,
            message_history=None,
            model=None,
        )
    ):
        yield chunk
```

### `AGUIEventStream` overview

`AGUIEventStream` extends `UIEventStream` and converts pydantic-ai streaming events to AG-UI protocol events:

| pydantic-ai event | AG-UI protocol event |
|---|---|
| `PartStartEvent(TextPart)` | `TextMessageStartEvent` |
| `PartDeltaEvent(TextPartDelta)` | `TextMessageContentEvent` |
| `PartEndEvent(TextPart)` | `TextMessageEndEvent` |
| `PartStartEvent(ThinkingPart)` | `ReasoningMessageStartEvent` (v >= REASONING_VERSION) |
| `ToolCallPart` start | `ToolCallStartEvent` |
| `ToolReturnPart` | `ToolCallResultEvent` |

### Example 4 — Custom `on_complete` callback

```python
from pydantic_ai.ui.ag_ui import AGUIAdapter

async def save_conversation(result):
    messages = result.all_messages()
    await db.save(messages)

@app.post('/chat')
async def chat(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(
        request,
        agent=agent,
        on_complete=save_conversation,
    )
```

---

## Vol. index

| Vol. | Classes | Version |
|---|---|---|
| [Vol. 1](./pydantic_ai_class_deep_dives/) | Core Agent, RunContext, ModelRetry, FunctionModel | 1.100.0 |
| [Vol. 2](./pydantic_ai_class_deep_dives_v2/) | Tool, ToolDefinition, SystemPromptRunner, OutputValidator | 1.101.0 |
| [Vol. 3](./pydantic_ai_class_deep_dives_v3/) | UIAdapter, CachePoint, format_as_xml, DeferredToolRequests, ToolSearch | 1.103.0 |
| [Vol. 4](./pydantic_ai_class_deep_dives_v4/) | LangChainToolset, VercelAIAdapter, ToolManager, ImageGeneration, XSearch | 1.104.0 |
| [Vol. 5](./pydantic_ai_class_deep_dives_v5/) | Model ABC, OpenAIChatModel, AnthropicModel, StreamedResponse | 1.104.0 |
| [Vol. 6](./pydantic_ai_class_deep_dives_v6/) | ModelSettings, UsageLimits, RunUsage, capture_run_messages | 1.104.0 |
| [Vol. 7](./pydantic_ai_class_deep_dives_v7/) | AgentEventStream, ThinkingPart, AudioUrl, WebSearchTool, MemoryTool | 1.104.0 |
| [Vol. 8](./pydantic_ai_class_deep_dives_v8/) | MCPToolset, FunctionToolset, AbstractToolset, NativeTool, DynamicCapability | 1.105.0 |
| [Vol. 9](./pydantic_ai_class_deep_dives_v9/) | Agent run iteration, AgentRun, AgentRunResult, output validators | 1.105.0 |
| [Vol. 10](./pydantic_ai_class_deep_dives_v10/) | AgentStream, WrapperCapability, FunctionToolset full, AbstractCapability | 1.105.0 |
| [Vol. 11](./pydantic_ai_class_deep_dives_v11/) | UserPromptNode, ModelRequestNode, CallToolsNode, AgentCapability, ToolSearchCallPart | 1.106.0 |
| [Vol. 12](./pydantic_ai_class_deep_dives_v12/) | Dataset, Evaluator, LLMJudge, SpanTree, MCPSamplingModel, ExternalToolset | 1.106.0 |
| [Vol. 13](./pydantic_ai_class_deep_dives_v13/) | TemplateStr, Hooks, WebSearch+WebFetch, Thinking, ExaToolset, AgentWorker | 1.106.0 |
| **Vol. 14 (this page)** | FallbackModel, InstrumentedModel, ModelProfile, EmbeddingModel, AGUIAdapter | **1.107.0** |

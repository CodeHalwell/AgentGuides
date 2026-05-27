---
title: "PydanticAI: Model Providers, FallbackModel & InstrumentationSettings"
description: "Every supported provider, provider prefixes, FallbackModel with exception and response handlers, FallbackExceptionGroup, InstrumentedModel, InstrumentationSettings versions 1-5, OTel token/cost histograms, gateway routing, and local models."
framework: pydanticai
language: python
---

# Model Providers, FallbackModel & InstrumentationSettings

Verified against **pydantic-ai==1.103.0** — source modules: `pydantic_ai.providers`, `pydantic_ai.models`, `pydantic_ai.models.fallback`, `pydantic_ai.models.instrumented`.

A PydanticAI `Agent` talks to an `LLM` through a `Model` backed by a `Provider`. The quickest way to wire one up is the `'provider:model-name'` string; the full way is constructing `SpecificModel(..., provider=SpecificProvider(...))` yourself. This page lists every prefix the installed source recognises and how to compose, gateway, or fall back between them.

## Minimal runnable example

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')
print(agent.run_sync('Hello!').output)

# Same thing, constructed explicitly:
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

agent = Agent(OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(api_key='sk-...')))
```

The string `'openai:gpt-5.2'` is parsed by `infer_provider_class` + `infer_provider` (`providers/__init__.py:100`, `:234`). The first token before `:` selects the provider; the remainder is the model name, verbatim.

## Provider prefixes

Verified from `providers/__init__.py:100`:

| Prefix                  | Model class                         | Notes                                                                   |
| ----------------------- | ----------------------------------- | ----------------------------------------------------------------------- |
| `openai`                | `OpenAIChatModel`                   | Default. `OPENAI_API_KEY` env var.                                      |
| `openai-chat`           | `OpenAIChatModel`                   | Forces the Chat Completions API.                                        |
| `openai-responses`      | `OpenAIResponsesModel`              | Forces the Responses API (reasoning, built-in tools).                   |
| `anthropic`             | `AnthropicModel`                    | `ANTHROPIC_API_KEY`.                                                    |
| `google-gla`            | `GoogleModel` (Gemini API)          | `GEMINI_API_KEY`. Formerly `google`.                                    |
| `google-vertex` / `vertexai` | `GoogleModel` (Vertex AI)      | Uses ADC / service-account credentials.                                 |
| `bedrock`               | `BedrockConverseModel`              | AWS credentials resolution.                                             |
| `groq`                  | `GroqModel`                         | `GROQ_API_KEY`.                                                         |
| `mistral`               | `MistralModel`                      | `MISTRAL_API_KEY`.                                                      |
| `cohere`                | `CohereModel`                       | `COHERE_API_KEY`.                                                       |
| `xai`                   | `OpenAI*`-compatible xAI model      | `XAI_API_KEY`. Supports `XSearchTool`.                                  |
| `grok`                  | deprecated alias of `xai`           | Prefer `xai:`.                                                          |
| `deepseek`              | OpenAI-compatible DeepSeek          |                                                                         |
| `openrouter`            | OpenAI-compatible OpenRouter        | Route to any OR model.                                                  |
| `vercel`                | Vercel AI Gateway                   |                                                                         |
| `azure`                 | Azure OpenAI                        | `AzureProvider(endpoint=..., api_key=...)`.                             |
| `cerebras`              | Cerebras                            |                                                                         |
| `moonshotai`            | Moonshot / Kimi                     |                                                                         |
| `fireworks`             | Fireworks AI                        |                                                                         |
| `together`              | Together AI                         |                                                                         |
| `heroku`                | Heroku Inference                    |                                                                         |
| `huggingface`           | HF Inference API                    |                                                                         |
| `ollama`                | `OllamaModel` (local)               | OpenAI-chat-compatible, no API key.                                     |
| `github`                | GitHub Models                       |                                                                         |
| `litellm`               | LiteLLM gateway                     |                                                                         |
| `nebius`, `ovhcloud`, `alibaba`, `sambanova` | OpenAI-compatible  | regional/cloud providers                                                |
| `outlines`              | Outlines (Transformers, vLLM, ...)  | Local constrained decoding.                                             |
| `sentence-transformers` | Embeddings only                     | `pydantic_ai.embeddings`.                                               |
| `voyageai`              | Embeddings only                     | `pydantic_ai.embeddings`.                                               |
| `gateway/<upstream>`    | Any upstream via Pydantic AI Gateway| e.g. `'gateway/openai:gpt-5.2'`.                                        |

The full list of `KnownModelName` literals (200+ entries) is in `models/__init__.py`. An unknown string with a known prefix still works — it's passed through to the provider.

## Explicit provider construction

Each provider accepts an `api_key`, a pre-built SDK client, or env-var fallback. Typical pattern:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    'claude-sonnet-4-6',
    provider=AnthropicProvider(api_key='...'),
)
agent = Agent(model)
```

Useful when you need to:

- Configure a custom `httpx.AsyncClient` (timeouts, proxies, retries).
- Share a single SDK client across many agents.
- Point at a self-hosted OpenAI-compatible endpoint (pass `base_url=` to `OpenAIProvider`).

OpenAI-compatible providers (`OpenAIProvider(base_url='http://localhost:8000/v1', api_key='...')`) unlock vLLM, LM Studio, oobabooga, or any homegrown server.

## `ModelSettings` — provider-agnostic knobs

`pydantic_ai.settings.ModelSettings` is a `TypedDict`. Common fields verified in `settings.py`:

`max_tokens`, `temperature`, `top_p`, `timeout`, `parallel_tool_calls`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `stop_sequences`, `extra_headers`, `extra_body`, `thinking` (`True` / `False` / `'minimal' | 'low' | 'medium' | 'high' | 'xhigh'`).

```python
agent = Agent(
    'openai:gpt-5.2',
    model_settings=ModelSettings(temperature=0.1, max_tokens=1024),
)
```

Provider-specific extensions (`OpenAIChatModelSettings`, `AnthropicModelSettings`, `GoogleModelSettings`) subclass it and add provider keys (e.g. `openai_reasoning_effort`, `anthropic_thinking`).

## `FallbackModel` — wrap primaries with backups

`pydantic_ai.models.fallback.FallbackModel` (`models/fallback.py`) accepts a default + one or more fallbacks. When a request fails (or a response predicate fires), it retries the next model in the list until one succeeds or all are exhausted.

```python
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

model = FallbackModel(
    'openai:gpt-4o',                    # primary
    'anthropic:claude-sonnet-4-6',      # first fallback
    'google-gla:gemini-2.0-flash',      # second fallback
    fallback_on=(ModelAPIError,),
)
agent = Agent(model)
result = agent.run_sync('What is 2 + 2?')
print(result.output)
```

### Constructor signature

```python
FallbackModel(
    default_model: Model | KnownModelName | str,
    *fallback_models: Model | KnownModelName | str,
    fallback_on: FallbackOn = (ModelAPIError,),
)
```

`fallback_on` is the core parameter. It can be any of:

| Form | Example | Behaviour |
|------|---------|-----------|
| Tuple of exception types | `(ModelAPIError,)` | Triggers when any listed exception is raised |
| Single exception type | `ModelAPIError` | Shorthand for a one-element tuple |
| Sync exception handler | `def fn(exc) -> bool` | `True` → fallback, `False` → re-raise |
| Async exception handler | `async def fn(exc) -> bool` | Same, awaited |
| Sync **response** handler | `def fn(resp: ModelResponse) -> bool` | Called after each successful request; `True` → fallback |
| Async response handler | `async def fn(resp: ModelResponse) -> bool` | Same, awaited |
| Sequence mixing any of the above | `[ModelAPIError, fn1, fn2]` | Evaluated in order; first `True` wins |

**Handler type auto-detection**: the source (`models/fallback.py:_add_handler`) inspects the first parameter's type hint. If it's annotated as `ModelResponse`, it's a response handler; otherwise (including bare `def fn(exc)` lambdas) it's an exception handler.

### Exception handlers — sync and async

```python
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

# Sync exception handler
def rate_limit_handler(exc: Exception) -> bool:
    """Fall back on 429 rate-limit errors only."""
    return isinstance(exc, ModelAPIError) and '429' in str(exc)

# Async exception handler — can call your own metrics service
async def async_exc_handler(exc: Exception) -> bool:
    # Could log to observability platform here
    return isinstance(exc, ModelAPIError)

model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-haiku-4-5',
    fallback_on=async_exc_handler,
)
```

### Response handlers — fall back on weak or malformed replies

```python
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

def too_short(resp: ModelResponse) -> bool:
    """Fall back if the model returned almost nothing."""
    texts = [p.content for p in resp.parts if p.part_kind == 'text']
    total_chars = sum(len(t) for t in texts)
    return total_chars < 10

async def async_quality_check(resp: ModelResponse) -> bool:
    """Async response handler — could POST to a scoring endpoint."""
    texts = [p.content for p in resp.parts if p.part_kind == 'text']
    return not any(texts)  # fall back on empty response

model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-sonnet-4-6',
    fallback_on=[too_short, ModelAPIError],
)
```

### Mixed sequence — all four forms together

```python
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

def is_rate_limited(exc: Exception) -> bool:
    return '429' in str(exc)

def is_server_error(exc: Exception) -> bool:
    return '5' in str(type(exc).__name__)

def empty_reply(resp: ModelResponse) -> bool:
    return not any(p.part_kind == 'text' for p in resp.parts)

model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-sonnet-4-6',
    'google-gla:gemini-2.0-flash',
    fallback_on=[
        ModelAPIError,        # exception type
        is_rate_limited,      # exception handler
        is_server_error,      # exception handler
        empty_reply,          # response handler (auto-detected from type hint)
    ],
)
```

### `FallbackExceptionGroup` — when all models fail

If every model in the chain fails, `FallbackModel` raises a `FallbackExceptionGroup` — a `BaseExceptionGroup` subclass that bundles all the individual exceptions:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel, FallbackExceptionGroup
from pydantic_ai.exceptions import ModelAPIError

model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-haiku-4-5',
    fallback_on=(ModelAPIError,),
)
agent = Agent(model)

async def main():
    try:
        await agent.run('Hello')
    except* ModelAPIError as eg:
        for exc in eg.exceptions:
            print(f'Provider failed: {exc}')
    except FallbackExceptionGroup as eg:
        print(f'All {len(eg.exceptions)} models failed.')

asyncio.run(main())
```

`except*` is Python 3.11+ ExceptionGroup syntax. For 3.10, use `except FallbackExceptionGroup` and inspect `.exceptions` manually.

### `FallbackModel.models` — inspecting the chain

```python
model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-haiku-4-5',
    fallback_on=(ModelAPIError,),
)
print([m.model_name for m in model.models])
# ['gpt-4o', 'claude-haiku-4-5']
```

### Gotchas

- **`fallback_on=()` (empty tuple) raises `UserError`** at construction time with the message "All exceptions will propagate". Always supply at least one condition.
- **Usage reflects the winning model only.** If the primary fails and the fallback succeeds, `result.usage` only includes the fallback's token counts. Use `InstrumentedModel` on each member of the chain to track per-model cost.
- **Fallbacks are tried serially**, not concurrently. For true parallel hedging, run multiple `Agent.run()` calls yourself and take the first result.
- **Response handlers run after every successful request**, including intermediate ones that aren't the final step (tool-calling loops). Keep them cheap.

---

## `InstrumentedModel` and `InstrumentationSettings`

`InstrumentedModel` (`models/instrumented.py`) wraps any `Model` to emit OpenTelemetry spans, log events, and metrics around every model request. `InstrumentationSettings` controls what goes on the wire.

### Quick start

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings

# Wrap a single agent
agent = Agent(
    InstrumentedModel(
        'openai:gpt-4o',
        options=InstrumentationSettings(version=4, include_content=True),
    )
)

# Or apply globally to every agent in the process
Agent.instrument_all(InstrumentationSettings(version=4))
```

### `InstrumentationSettings` constructor (source-verified)

```python
InstrumentationSettings(
    *,
    tracer_provider: TracerProvider | None = None,        # default: global provider
    meter_provider: MeterProvider | None = None,          # default: global provider
    logger_provider: LoggerProvider | None = None,        # default: global provider
    include_binary_content: bool = True,
    include_content: bool = True,
    version: Literal[1, 2, 3, 4, 5] = 5,   # default changed in 1.103.0
    event_mode: Literal['attributes', 'logs'] = 'attributes',   # only for version=1
    use_aggregated_usage_attribute_names: bool = False,
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tracer_provider` | global | Pass to scope spans to a custom provider |
| `meter_provider` | global | Token + cost histograms |
| `logger_provider` | global | OTel log events (version 1 only) |
| `include_binary_content` | `True` | Whether to encode base64 images/files in spans |
| `include_content` | `True` | Set `False` to strip prompts & completions from spans (PII) |
| `version` | `5` | Wire format version (see table below) |
| `event_mode` | `'attributes'` | `'logs'` emits OTel log-based events (version 1 only) |
| `use_aggregated_usage_attribute_names` | `False` | Use `gen_ai.aggregated_usage.*` to avoid double-counting in backends that aggregate child+parent spans |

### Data format versions

| Version | Notes |
|---------|-------|
| `1` | Legacy event-based GenAI spec — **deprecated**, will be removed. Use `event_mode` / `logger_provider` with this version. |
| `2` | Newer OTel GenAI spec. Messages in `gen_ai.input.messages` / `gen_ai.output.messages` span attributes; system instructions in `gen_ai.system_instructions`. |
| `3` | Same as v2 + thinking tokens (e.g. Claude extended thinking, o-series reasoning tokens). |
| `4` | Same as v3 + GenAI semantic conventions for multimodal content. URL media uses `type='uri'`; inline binaries use `type='blob'` with `modality`. |
| `5` | Same as v4 + `CallDeferred` and `ApprovalRequired` control-flow exceptions no longer set the span to `ERROR` — spans are left `UNSET` since deferrals are not failures. **Default from 1.103.0.** |

```python
from pydantic_ai.models.instrumented import InstrumentationSettings

# Production — PII scrubbed, version 5 (default)
prod_settings = InstrumentationSettings(
    include_content=False,
    include_binary_content=False,
    version=5,
)

# Development — full content for debugging
dev_settings = InstrumentationSettings(
    include_content=True,
    version=5,
)

# Legacy Logfire setup still on version 1
legacy = InstrumentationSettings(
    version=1,
    event_mode='logs',
)
```

### OTel metrics emitted

`InstrumentationSettings` creates two meters on init (source: `models/instrumented.py`):

| Metric | Unit | Description |
|--------|------|-------------|
| `gen_ai.client.token.usage` | `{token}` | Input and output token counts per request (type=`input`/`output`) |
| `operation.cost` | `{USD}` | Monetary cost per request (via `genai-prices`) |

```python
# Both meters are histogram-type; bucket boundaries follow the OTel GenAI spec.
# They are automatically populated on every InstrumentedModel request — no extra code needed.
```

To avoid double-counting on observability backends that sum child+parent spans:

```python
settings = InstrumentationSettings(
    use_aggregated_usage_attribute_names=True,
    # => uses gen_ai.aggregated_usage.input_tokens / gen_ai.aggregated_usage.output_tokens
    # on agent-run spans instead of gen_ai.usage.*
)
```

### `include_content=False` — PII scrubbing

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings

pii_safe = InstrumentationSettings(
    include_content=False,          # no prompts or completions in spans
    include_binary_content=False,   # no base64-encoded images/files
)

agent = Agent(InstrumentedModel('openai:gpt-4o', options=pii_safe))
# Spans will contain timing, token counts, and model name — but NO text content.
```

### `messages_to_otel_events` — inspect span events manually

```python
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.messages import ModelRequest, UserPromptPart

settings = InstrumentationSettings()
messages = [ModelRequest(parts=[UserPromptPart(content='Hello')])]
events = settings.messages_to_otel_events(messages)
for event in events:
    print(event.attributes.get('event.name'), event.body)
```

### `instrument_model` helper function

`instrument_model` (`models/instrumented.py:instrument_model`) is the low-level primitive. It only wraps if the model isn't already instrumented:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.instrumented import instrument_model, InstrumentationSettings

raw_model = OpenAIChatModel('gpt-4o')
model = instrument_model(raw_model, instrument=InstrumentationSettings(version=5))
# If raw_model were already an InstrumentedModel, it's returned unchanged.
```

### Global instrumentation with `Agent.instrument_all`

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

# Logfire configures the global OTel providers — instrument_all picks them up
logfire.configure()
Agent.instrument_all(InstrumentationSettings(
    include_content=True,
    version=5,
    use_aggregated_usage_attribute_names=True,
))

# Every Agent created after this call will have spans emitted
agent1 = Agent('openai:gpt-4o')
agent2 = Agent('anthropic:claude-sonnet-4-6')
```

### Per-provider tracing with custom TracerProvider

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings

# Route only model spans to a dedicated OTLP endpoint
model_provider = TracerProvider()
model_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint='http://otel-collector:4318/v1/traces'))
)

settings = InstrumentationSettings(tracer_provider=model_provider, include_content=True)
model = InstrumentedModel('openai:gpt-4o', options=settings)
```

## Gateway routing

Prefix a known provider with `gateway/` to route it through the [Pydantic AI Gateway](https://ai.pydantic.dev/gateway/):

```python
agent = Agent('gateway/openai:gpt-5.2')
# => uses the gateway provider, normalising to the upstream OpenAI profile
```

`normalize_gateway_provider` (`providers/gateway.py`) strips the prefix so model-profile lookups still resolve correctly.

## Local models

```python
# Ollama (no key, OpenAI-chat compatible)
agent = Agent('ollama:llama3.1')

# Any OpenAI-compatible local server
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'qwen2.5-coder:32b',
    provider=OpenAIProvider(base_url='http://localhost:8000/v1', api_key='x'),
)
```

`OpenAIProvider` injects `api_key='api-key-not-set'` when you pass `base_url` without a key, which keeps the OpenAI SDK happy against local servers that don't require auth.

## Patterns

### 1. Provider-level concurrency limit

```python
from pydantic_ai import limit_model_concurrency

model = limit_model_concurrency(OpenAIChatModel('gpt-5.2'), limit=8)
```

Enforces a max of 8 concurrent in-flight requests at the model layer.

### 2. Region-aware fallback

```python
model = FallbackModel(
    'bedrock:us.anthropic.claude-sonnet-4-6',
    'bedrock:eu.anthropic.claude-sonnet-4-6',
    fallback_on=(ModelAPIError,),
)
```

### 3. Rate-limit-aware fallback with response sniff

```python
def empty_or_short(resp: ModelResponse) -> bool:
    for p in resp.parts:
        if getattr(p, 'part_kind', None) == 'text' and len(p.content) >= 20:
            return False
    return True

model = FallbackModel('openai:gpt-5.2', 'anthropic:claude-sonnet-4-6',
                     fallback_on=[empty_or_short, ModelAPIError])
```

### 4. Self-hosted vLLM with a shared `httpx` client

```python
import httpx

shared = httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=50))
provider = OpenAIProvider(base_url='http://vllm:8000/v1', api_key='x', http_client=shared)
model = OpenAIChatModel('meta-llama/Llama-3.1-8B-Instruct', provider=provider)
```

### 5. Swap model per environment with `agent.override`

```python
if env == 'production':
    ctx = agent.override(model='openai:gpt-5.2')
elif env == 'canary':
    ctx = agent.override(model=FallbackModel('openai:gpt-5.2', 'anthropic:claude-sonnet-4-6'))
else:
    from pydantic_ai.models.test import TestModel
    ctx = agent.override(model=TestModel())
with ctx:
    result = agent.run_sync(prompt)
```

### 6. `FallbackModel` with async response quality check

```python
import asyncio
import httpx
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

async def quality_check(resp: ModelResponse) -> bool:
    """POST response to an internal scoring endpoint; fall back if score < 0.7."""
    texts = [p.content for p in resp.parts if p.part_kind == 'text']
    text = ' '.join(texts)
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            r = await client.post('http://scorer/score', json={'text': text})
            return r.json()['score'] < 0.7
    except Exception:
        return False  # don't fall back if scorer is down

model = FallbackModel(
    'openai:gpt-4o-mini',
    'anthropic:claude-haiku-4-5',
    fallback_on=[quality_check, ModelAPIError],
)
agent = Agent(model)
```

## Reference

- `infer_provider`, `infer_provider_class` — `providers/__init__.py:100`, `:234`
- `KnownModelName` — `models/__init__.py` (near the top)
- `FallbackModel` — `models/fallback.py`
- `FallbackExceptionGroup` — `models/fallback.py`
- `InstrumentedModel` — `models/instrumented.py`
- `InstrumentationSettings` — `models/instrumented.py`
- `instrument_model` — `models/instrumented.py`
- `ModelSettings` base — `settings.py:24`
- `limit_model_concurrency` — `models/concurrency.py`

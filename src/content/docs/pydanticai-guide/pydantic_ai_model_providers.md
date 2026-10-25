---
title: "PydanticAI: Model Providers, FallbackModel & Instrumentation"
description: "Every supported provider, provider prefixes, FallbackModel with exception and response handlers, FallbackExceptionGroup, InstrumentedModel, InstrumentationSettings versions 2-6, OTel token/cost histograms, gateway routing, and local models."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Model providers"
  order: 5
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%) — source modules: `pydantic_ai.providers`, `pydantic_ai.models`, `pydantic_ai.models.fallback`, `pydantic_ai.models.instrumented`.

A PydanticAI `Agent` talks to an `LLM` through a `Model` backed by a `Provider`. The quickest way to wire one up is the `'provider:model-name'` string; the full way is constructing `SpecificModel(..., provider=SpecificProvider(...))` yourself. This page lists every prefix the installed source recognises and how to compose, gateway, or fall back between them.

## Minimal runnable example

```python
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')
print(agent.run_sync('Hello!').output)

# Same thing, constructed explicitly (`openai:` means the Responses API):
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

agent = Agent(OpenAIResponsesModel('%%model=openai%%', provider=OpenAIProvider(api_key='sk-...')))
```

A string such as `'openai:%%model=openai%%'` is parsed by `infer_provider_class` + `infer_provider` (`providers/__init__.py`). The first token before `:` selects the provider; the remainder is the model name, verbatim.

## Provider prefixes

Verified from `infer_provider_class` in `providers/__init__.py`:

| Prefix                  | Model class                         | Notes                                                                   |
| ----------------------- | ----------------------------------- | ----------------------------------------------------------------------- |
| `openai`                | `OpenAIResponsesModel`              | `OPENAI_API_KEY`. **Since 2.0 the bare prefix means the Responses API** (it meant Chat Completions in V1). |
| `openai-chat`           | `OpenAIChatModel`                   | Chat Completions API — use this to keep the V1 behaviour.               |
| `openai-responses`      | `OpenAIResponsesModel`              | Explicit form of the default.                                           |
| `azure` / `azure-responses` | Azure OpenAI                    | `AzureProvider(endpoint=..., api_key=...)`.                             |
| `anthropic`             | `AnthropicModel`                    | `ANTHROPIC_API_KEY`.                                                    |
| `google`                | `GoogleModel` (Gemini API)          | `GOOGLE_API_KEY` / `GEMINI_API_KEY`. Replaces V1's `google-gla:`.       |
| `google-cloud`          | `GoogleModel` (Vertex AI)           | ADC / service-account credentials. Replaces V1's `google-vertex:` / `vertexai:`. |
| `bedrock`               | `BedrockConverseModel`              | AWS credentials resolution. Needs the `bedrock` extra.                  |
| `bedrock-mantle`        | Bedrock Mantle                      | Needs the `bedrock-mantle` extra.                                       |
| `groq`                  | `GroqModel`                         | `GROQ_API_KEY`. Needs the `groq` extra.                                 |
| `mistral`               | `MistralModel`                      | `MISTRAL_API_KEY`. Needs the `mistral` extra.                           |
| `cohere`                | `CohereModel`                       | `CO_API_KEY`. Needs the `cohere` extra.                                 |
| `xai`                   | `XaiModel`                          | `XAI_API_KEY`. Supports `XSearchTool`. Replaces V1's `grok:`. Needs the `xai` extra. |
| `deepseek`              | OpenAI-compatible DeepSeek          |                                                                         |
| `openrouter`            | `OpenRouterModel`                   | Route to any OpenRouter model.                                          |
| `vercel`                | Vercel AI Gateway                   |                                                                         |
| `cerebras`              | Cerebras                            |                                                                         |
| `moonshotai`            | Moonshot / Kimi                     |                                                                         |
| `zai`                   | Z.AI                                |                                                                         |
| `fireworks`, `together`, `heroku`, `crusoe`, `nebius`, `ovhcloud`, `alibaba`, `sambanova`, `meta`, `snowflake` | OpenAI-compatible | hosted / regional providers |
| `huggingface`           | HF Inference Providers              | Needs the `huggingface` extra.                                          |
| `ollama`                | `OllamaModel` (local)               | Set `OLLAMA_BASE_URL` (e.g. `http://localhost:11434/v1`); no API key.   |
| `vllm`, `litellm`       | OpenAI-compatible                   | Self-hosted server / LiteLLM proxy (`vllm:` needs `VLLM_BASE_URL`).     |
| `github`, `github-copilot` | GitHub Models / Copilot          |                                                                         |
| `sentence-transformers`, `voyageai` | Embeddings only         | `pydantic_ai.embeddings`.                                               |
| `gateway/<upstream>`    | Any upstream via Pydantic AI Gateway| e.g. `'gateway/openai:<model>'`.                                       |
| `typesafe`, `system-one`, `openai-codex`, `openai-chatgpt`, `openai-decisions` | provider-specific model classes | Newer integrations; see the docstrings in `pydantic_ai.models`.        |
| `test`                  | `TestModel`                         | No network; `Agent('test')`.                                            |

Removed in 2.0: the `grok:`, `google-gla:`, `google-vertex:` / `vertexai:` prefixes and the Outlines integration. A bare model name with no prefix (`Agent('gpt-5')`) now raises `UserError`.

Since 2.0 a bare `pip install pydantic-ai` installs only the OpenAI, Anthropic and Google providers (plus CLI, MCP, evals, web and Logfire); the `retries` extra is separate. Add the extras you use, e.g. `pip install "pydantic-ai[bedrock,groq]"`.

The `KnownModelName` literal in `pydantic_ai.models` lists several hundred model names. An unknown string with a known prefix still works — it's passed through to the provider.

## Explicit provider construction

Each provider accepts an `api_key`, a pre-built SDK client, or env-var fallback. Typical pattern:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    '%%model=anthropic%%',
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

`pydantic_ai.settings.ModelSettings` is a `TypedDict`; every key is optional. Most keys a model does not support are ignored silently; the few whose absence changes what the request guarantees (`timeout`, `max_retries`, `stop_sequences`, `parallel_tool_calls=False`, `extra_headers`, `extra_body`) emit an `UnsupportedSettingWarning`. Settings are layered key by key — the model's own `settings`, then the agent's `model_settings`, then capabilities, then the run's `model_settings=` — and the later layer wins.

| Key | Notes |
|---|---|
| `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop_sequences`, `presence_penalty`, `frequency_penalty`, `logit_bias` | The usual sampling controls. |
| `thinking` | `ThinkingLevel = bool \| Literal['minimal', 'low', 'medium', 'high', 'xhigh', 'max']`. `True` = provider default effort; `False` disables thinking where the model allows it (on models that cannot switch it off, 2.61 sends the lowest level they accept); unmapped levels round to the nearest the provider has. |
| `service_tier` | `ServiceTier = Literal['auto', 'default', 'flex', 'priority']`. Provider-specific keys (`openai_service_tier`, …) take precedence. |
| `cache` | Library-managed prompt caching: `True`, `False`, a retention (`'5m'`, `'30m'`, `'1h'`) or `CacheConfig(retention=..., messages=False)`. Provider-specific cache keys win if set. See also `CachePoint` in the [message history guide](/pydanticai-guide/pydantic_ai_message_history/). |
| `tool_choice` | `None`/`'auto'`, `'none'`, `'required'`, a list of tool names, or `ToolOrOutput(function_tools=[...])`. |
| `parallel_tool_calls` | `False` asks the model for one tool call per response. |
| `timeout` | Per-attempt timeout applied by the provider client. |
| `request_timeout` | One deadline, enforced by PydanticAI, covering every SDK retry and the whole streamed response. |
| `max_retries` | Override the provider SDK's retry count for one request (`0` = single attempt). |
| `extra_headers`, `extra_body` | Escape hatches for provider features without a first-class key. |

```python
from pydantic_ai import Agent, ModelSettings

agent = Agent(
    'anthropic:%%model=anthropic-flagship%%',
    model_settings=ModelSettings(temperature=0.1, max_tokens=4096, thinking='high', service_tier='priority'),
)
```

Provider-specific extensions (`OpenAIChatModelSettings`, `OpenAIResponsesModelSettings`, `AnthropicModelSettings`, `GoogleModelSettings`, …) subclass it and add prefixed keys (e.g. `openai_reasoning_effort`, `anthropic_thinking`). V1's `OpenAIModelSettings` is now `OpenAIChatModelSettings`.

### `tool_choice` and `ToolOrOutput`

A static `tool_choice=['search']` would also hide the output tools, so the run could never finish (PydanticAI raises `UserError` for that, and for a static `'required'`). `ToolOrOutput` restricts the *function* tools while keeping output available:

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ToolOrOutput

agent = Agent('openai:%%model=openai%%')


@agent.tool_plain
def search_web(query: str) -> str:
    """Search the web."""
    return f'Results for {query!r}'


@agent.tool_plain
def delete_cache() -> str:
    """Clear the cache."""
    return 'cleared'


result = agent.run_sync(
    'What is the population of France?',
    model_settings={'tool_choice': ToolOrOutput(function_tools=['search_web'])},  # delete_cache hidden
)
```

To force a particular tool on one step only (say, the first), return the setting from a capability's `get_model_settings` callable rather than setting it statically.

## Model-level capabilities

These capabilities change *which* model is used or *how* it is called, without touching your agent code.

| Capability | What it does |
|---|---|
| `Thinking(effort=True)` | Sets `thinking` portably; provider-specific thinking settings still win. |
| `Caching(retention=True, messages=True)` | Sets `cache` (library-managed prompt caching). |
| `SelectModel(selector)` | Picks the model before each model-request step from a `ModelSelectionContext` (`deps`, `prompt`, `messages`, `usage`, `run_step`, `usage_limits`, `deadline`, `run_id`, `conversation_id`, the lower-precedence `model`). |
| `ResolveModelId(resolver)` | Turns a model-ID string into a `Model` (custom names, per-tenant credentials); return `None` to fall through to the normal lookup. Replaces Temporal's old `provider_factory=`. |
| `RaiseContentFilterError()` | Raises `ContentFilterError` (an `UnexpectedModelBehavior`) when a response finishes with `finish_reason='content_filter'`, instead of returning an empty or partial output. |

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching, RaiseContentFilterError, ResolveModelId, SelectModel, Thinking
from pydantic_ai.models import ModelResolutionContext, ModelSelectionContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider


def route(ctx: ModelSelectionContext) -> str:
    # cheap model for the first step, stronger one once tools have produced results
    return 'anthropic:%%model=anthropic-small%%' if ctx.run_step == 1 else 'anthropic:%%model=anthropic%%'


def resolve(ctx: ModelResolutionContext, model_id: str):
    if model_id == 'tenant-default':
        return AnthropicModel('%%model=anthropic%%', provider=AnthropicProvider(api_key='per-tenant-key'))
    return None  # fall through to the standard 'provider:model' lookup


agent = Agent(
    'tenant-default',
    capabilities=[
        ResolveModelId(resolve),
        SelectModel(route),
        Thinking(effort='low'),
        Caching('1h'),
        RaiseContentFilterError(),
    ],
)
```

Capabilities can also be passed per run (`agent.run(..., capabilities=[Thinking(effort='xhigh')])`) for a one-off change.

## `FallbackModel` — wrap primaries with backups

`pydantic_ai.models.fallback.FallbackModel` (`models/fallback.py`) accepts a default + one or more fallbacks. When a request fails (or a response predicate fires), it retries the next model in the list until one succeeds or all are exhausted.

```python
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

model = FallbackModel(
    'openai:%%model=openai%%',                    # primary
    'anthropic:%%model=anthropic%%',      # first fallback
    'google:%%model=gemini%%',      # second fallback
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
    'openai:%%model=openai%%',
    'anthropic:%%model=anthropic-small%%',
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
    'openai:%%model=openai%%',
    'anthropic:%%model=anthropic%%',
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
    'openai:%%model=openai%%',
    'anthropic:%%model=anthropic%%',
    'google:%%model=gemini%%',
    fallback_on=[
        ModelAPIError,        # exception type
        is_rate_limited,      # exception handler
        is_server_error,      # exception handler
        empty_reply,          # response handler (auto-detected from type hint)
    ],
)
```

### `FallbackExceptionGroup` — when all models fail

If every model in the chain fails, `FallbackModel` raises a `FallbackExceptionGroup` — an `ExceptionGroup` subclass that bundles all the individual exceptions:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel, FallbackExceptionGroup
from pydantic_ai.exceptions import ModelAPIError

model = FallbackModel(
    'openai:%%model=openai%%',
    'anthropic:%%model=anthropic-small%%',
    fallback_on=(ModelAPIError,),
)
agent = Agent(model)

async def main():
    try:
        await agent.run('Hello')
    except FallbackExceptionGroup as eg:
        print(f'All {len(eg.exceptions)} models failed:')
        for exc in eg.exceptions:
            print(f'  {type(exc).__name__}: {exc}')

asyncio.run(main())
```

`FallbackExceptionGroup` is an `ExceptionGroup`, so `except* ModelAPIError` also works (Python 3.11+, which PydanticAI requires) — but a single `try` cannot mix `except` and `except*` clauses; pick one style.

### `FallbackModel.models` — inspecting the chain

```python
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel(
    'openai:%%model=openai%%',
    'anthropic:%%model=anthropic-small%%',
    fallback_on=(ModelAPIError,),
)
print([m.model_name for m in model.models])
# ['%%model=openai%%', '%%model=anthropic-small%%']
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
        'openai:%%model=openai%%',
        options=InstrumentationSettings(include_content=True),
    )
)

# Or apply globally to every agent in the process
Agent.instrument_all(InstrumentationSettings())
```

### `InstrumentationSettings` constructor

```python
InstrumentationSettings(
    *,
    tracer_provider: TracerProvider | None = None,        # default: global provider
    meter_provider: MeterProvider | None = None,          # default: global provider
    include_binary_content: bool = True,
    include_content: bool = True,
    include_model_request_parameters: bool = True,
    version: Literal[2, 3, 4, 5, 6] = 5,
    use_aggregated_usage_attribute_names: bool = True,
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tracer_provider` | global | Pass to scope spans to a custom provider |
| `meter_provider` | global | Token + cost histograms |
| `include_binary_content` | `True` | Whether to encode base64 images/files in spans |
| `include_content` | `True` | Set `False` to strip prompts & completions from spans (PII) |
| `include_model_request_parameters` | `True` | Record the full request parameters (tool definitions etc.) on spans; set `False` if large tool schemas bloat spans |
| `version` | `5` | Wire format version (see table below) |
| `use_aggregated_usage_attribute_names` | `True` | Agent-run spans report `gen_ai.aggregated_usage.*` to avoid double-counting in backends that aggregate child+parent spans. Default flipped to `True` in 2.0 |

Version 1 and its `event_mode=` / `logger_provider=` arguments were removed in 2.0.

### Data format versions

| Version | Notes |
|---------|-------|
| `1` | Legacy event-based spec — **removed in 2.0**. |
| `2` | Newer OTel GenAI spec (deprecated since 2.0 — emits `PydanticAIDeprecationWarning`, as do 3 and 4). Messages in `gen_ai.input.messages` / `gen_ai.output.messages` span attributes; system instructions in `gen_ai.system_instructions`. |
| `3` | Same as v2 + thinking tokens (e.g. Claude extended thinking, o-series reasoning tokens). |
| `4` | Same as v3 + GenAI semantic conventions for multimodal content. URL media uses `type='uri'`; inline binaries use `type='blob'` with `modality`. |
| `5` | Same as v4 + `CallDeferred` and `ApprovalRequired` control-flow exceptions no longer set the span to `ERROR` — spans are left `UNSET` since deferrals are not failures. **Default** (since 1.103.0). |
| `6` | Same as v5, but tool results are emitted in a message with `role='tool'` instead of `role='user'`. Opt-in. |

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
from pydantic_ai.models.instrumented import InstrumentationSettings

# Already the default since 2.0: agent-run spans use
# gen_ai.aggregated_usage.input_tokens / output_tokens, model-request spans keep gen_ai.usage.*.
# To keep the V1 attribute names on run spans (e.g. for existing dashboards):
settings = InstrumentationSettings(use_aggregated_usage_attribute_names=False)
```

### `include_content=False` — PII scrubbing

```python
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings

pii_safe = InstrumentationSettings(
    include_content=False,          # no prompts or completions in spans
    include_binary_content=False,   # no base64-encoded images/files
)

agent = Agent(InstrumentedModel('openai:%%model=openai%%', options=pii_safe))
# Spans will contain timing, token counts, and model name — but NO text content.
```

### `messages_to_otel_messages` — see what lands on the span

The V1 `messages_to_otel_events()` helper went with version 1. To see the GenAI-convention message list that version 2+ writes to `gen_ai.input.messages`, call `messages_to_otel_messages()`:

```python
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.messages import ModelRequest, UserPromptPart

settings = InstrumentationSettings()
messages = [ModelRequest(parts=[UserPromptPart(content='Hello')])]
print(settings.messages_to_otel_messages(messages))
```

### `instrument_model` helper function

`instrument_model` (`models/instrumented.py:instrument_model`) is the low-level primitive. It only wraps if the model isn't already instrumented:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.instrumented import instrument_model, InstrumentationSettings

raw_model = OpenAIChatModel('%%model=openai%%')
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
agent1 = Agent('openai:%%model=openai%%')
agent2 = Agent('anthropic:%%model=anthropic%%')
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
model = InstrumentedModel('openai:%%model=openai%%', options=settings)
```

## Gateway routing

Prefix a known provider with `gateway/` to route it through the [Pydantic AI Gateway](https://pydantic.dev/docs/ai/overview/gateway/):

```python
from pydantic_ai import Agent

agent = Agent('gateway/openai:%%model=openai%%')   # needs PYDANTIC_AI_GATEWAY_API_KEY
# => uses the gateway provider, normalising to the upstream OpenAI profile
```

`normalize_gateway_provider` (`providers/gateway.py`) strips the prefix so model-profile lookups still resolve correctly.

## Local models

```python
from pydantic_ai import Agent

# Ollama (no key; set OLLAMA_BASE_URL=http://localhost:11434/v1)
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
from pydantic_ai.models.openai import OpenAIChatModel

model = limit_model_concurrency(OpenAIChatModel('%%model=openai%%'), 8)   # or ConcurrencyLimit(max_running=8, max_queued=100)
```

Enforces a max of 8 concurrent in-flight requests at the model layer.

### 2. Region-aware fallback

```python
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel(
    'bedrock:us.anthropic.%%model=anthropic%%',
    'bedrock:eu.anthropic.%%model=anthropic%%',
    fallback_on=(ModelAPIError,),
)
```

### 3. Rate-limit-aware fallback with response sniff

```python
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel


def empty_or_short(resp: ModelResponse) -> bool:
    for p in resp.parts:
        if getattr(p, 'part_kind', None) == 'text' and len(p.content) >= 20:
            return False
    return True

model = FallbackModel('openai:%%model=openai%%', 'anthropic:%%model=anthropic%%',
                     fallback_on=[empty_or_short, ModelAPIError])
```

### 4. Self-hosted vLLM with a shared `httpx` client

```python
import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

shared = httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=50))
provider = OpenAIProvider(base_url='http://vllm:8000/v1', api_key='x', http_client=shared)
model = OpenAIChatModel('meta-llama/Llama-3.1-8B-Instruct', provider=provider)
```

### 5. Swap model per environment with `agent.override`

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

agent = Agent('openai:%%model=openai%%', defer_model_check=True)
env = os.getenv('APP_ENV', 'test')
prompt = 'Hello'

if env == 'production':
    ctx = agent.override(model='openai:%%model=openai%%')
elif env == 'canary':
    ctx = agent.override(model=FallbackModel('openai:%%model=openai%%', 'anthropic:%%model=anthropic%%'))
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
    'openai:%%model=openai-small%%',
    'anthropic:%%model=anthropic-small%%',
    fallback_on=[quality_check, ModelAPIError],
)
agent = Agent(model)
```

## Reference

- `infer_provider`, `infer_provider_class` — `providers/__init__.py`
- `KnownModelName` — `models/__init__.py` (near the top)
- `FallbackModel` — `models/fallback.py`
- `FallbackExceptionGroup` — `models/fallback.py`
- `InstrumentedModel` — `models/instrumented.py`
- `InstrumentationSettings` — `models/instrumented.py`
- `instrument_model` — `models/instrumented.py`
- `ModelSettings` base — `settings.py`
- `limit_model_concurrency` — `models/concurrency.py`

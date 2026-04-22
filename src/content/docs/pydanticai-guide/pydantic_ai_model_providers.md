---
title: "PydanticAI: Model Providers & FallbackModel"
description: "Every supported provider, provider prefixes, the model string format, custom Provider wiring, FallbackModel with custom handlers, gateway routing, and local models."
framework: pydanticai
language: python
---

# Model Providers & FallbackModel

Verified against **pydantic-ai==1.85.1** — source: `/tmp/pydantic-ai-install/pydantic_ai/providers/__init__.py`, `/tmp/pydantic-ai-install/pydantic_ai/models/__init__.py`, `/tmp/pydantic-ai-install/pydantic_ai/models/fallback.py`.

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

`pydantic_ai.models.fallback.FallbackModel` accepts a default + one or more fallbacks and decides when to switch.

```python
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

model = FallbackModel(
    'openai:gpt-5.2',
    'anthropic:claude-sonnet-4-6',
    'google-gla:gemini-embedding-001',  # won't ever run in practice, just showing multi-fallback
    fallback_on=(ModelAPIError,),
)
agent = Agent(model)
```

### `fallback_on` — exception types _and_ response predicates

`FallbackModel.__init__` (`models/fallback.py:81`) accepts any of:

- A tuple of exception types: `(ModelAPIError, RateLimitError)`
- A single exception type: `ModelAPIError`
- A sync/async **exception handler**: `def(exc) -> bool`
- A sync/async **response handler**: `def(resp: ModelResponse) -> bool`
- A sequence mixing any of the above.

Handler type is auto-detected from the first parameter's type hint. `ModelResponse` → response handler; anything else (or untyped) → exception handler.

```python
from pydantic_ai.messages import ModelResponse

def weak_response(resp: ModelResponse) -> bool:
    # treat empty text as a failure worth switching on
    texts = [p.content for p in resp.parts if getattr(p, 'part_kind', None) == 'text']
    return not texts or all(not t.strip() for t in texts)

def is_rate_limit(exc) -> bool:
    return 'rate' in str(exc).lower()

model = FallbackModel(
    'openai:gpt-5.2',
    'anthropic:claude-sonnet-4-6',
    fallback_on=[weak_response, is_rate_limit, ModelAPIError],
)
```

### Gotchas

- `fallback_on=()` (empty tuple) raises `UserError` — "All exceptions will propagate". Always supply at least one condition.
- Fallbacks do **not** stack usage costs; `result.usage` reflects whichever model finally succeeded. Track per-model cost via OpenTelemetry (see `InstrumentationSettings`).
- Exceptions from the last model propagate wrapped in `FallbackExceptionGroup`.

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

## Instrumenting model calls

```python
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings

instrumented = InstrumentedModel(
    OpenAIChatModel('gpt-5.2'),
    options=InstrumentationSettings(event_mode='attributes'),
)
agent = Agent(instrumented)
```

Or globally: `Agent.instrument_all(InstrumentationSettings(...))` (`agent/__init__.py:844`). See the production guide for Logfire / OTel wiring.

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

## Reference

- `infer_provider`, `infer_provider_class` — `providers/__init__.py:100`, `:234`
- `KnownModelName` — `models/__init__.py` (near the top)
- `FallbackModel` — `models/fallback.py:69`
- `InstrumentedModel`, `InstrumentationSettings` — `models/instrumented.py:78`, `:388`
- `ModelSettings` base — `settings.py:24`
- `limit_model_concurrency` — `models/concurrency.py`

---
title: "PydanticAI — Class Deep Dives Vol. 18"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.107.0: AlibabaProvider (DashScope + Qwen omni audio), OVHcloudProvider (multi-family profile routing), HerokuProvider (Heroku Inference API), gateway_provider (Pydantic AI Gateway multi-upstream routing + region-encoded keys), GoogleProvider + GoogleCloudProvider (Gemini API vs Google Cloud/Vertex AI), GoogleModel + GoogleModelSettings (all 9 settings including cached_content caveat + GoogleCloudServiceTier), CohereModel + CohereProvider + CohereModelSettings (Command-R family via AsyncClientV2), XaiProvider + XaiModel (gRPC SDK + LazyAsyncClient loop affinity + GrokModelProfile/GrokReasoningEffort), FileStatePersistence (atomic file-lock persistence for pydantic_graph), SimpleStatePersistence + FullStatePersistence (in-memory simple vs full-history with dump_json/load_json). All verified against pydantic-ai 1.107.0 / pydantic-graph 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 18)"
  order: 44
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** / **pydantic-graph 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups spanning new providers, the Google ecosystem, and graph persistence: `AlibabaProvider` (Alibaba Cloud DashScope — Qwen profiles, omni audio URI encoding, dual env-var support); `OVHcloudProvider` (OVHcloud AI Endpoints — prefix-based multi-family profile routing for six model families); `HerokuProvider` (Heroku Inference API — `HEROKU_INFERENCE_KEY`, configurable `HEROKU_INFERENCE_URL`); `gateway_provider` (Pydantic AI Gateway — multi-upstream dispatch, `route` override, region-encoded API key with automatic base-URL inference); `GoogleProvider` + `GoogleCloudProvider` (Gemini API vs Google Cloud/Vertex AI — `GOOGLE_API_KEY` vs ADC credentials, `BaseGoogleProvider` shared base, migration guidance); `GoogleModel` + `GoogleModelSettings` (all nine provider-prefixed settings — `google_cached_content` caveat on tool/system-prompt stripping, `google_safety_settings`, `MediaResolution`, `GoogleCloudServiceTier`); `CohereModel` + `CohereProvider` + `CohereModelSettings` (Command-R family — `CO_API_KEY`, `AsyncClientV2`, thinking content mapping, `LatestCohereModelNames`); `XaiProvider` + `XaiModel` (xAI gRPC SDK — `_LazyAsyncClient` per-loop affinity, `GrokModelProfile`, `GrokReasoningEffort`, builtin-tool gate, gRPC error mapping); `FileStatePersistence` (file-based graph persistence — atomic `.pydantic-graph-persistence-lock` lock file, snapshot lifecycle, `anyio`-based async I/O); `SimpleStatePersistence` + `FullStatePersistence` (in-memory graph persistence — lightweight latest-only vs full history with `dump_json`/`load_json` round-trip).

<Aside type="note" title="Provider pattern">
Providers are passed to the **model constructor**, not directly to `Agent`. Always write `Agent(SomeModel('model-name', provider=provider))`, never `Agent(model='model-name', provider=provider)`. All examples below follow this pattern.
</Aside>

---

## 1. `AlibabaProvider` — Alibaba Cloud DashScope

**Module:** `pydantic_ai.providers.alibaba`  
**Import:**
```python
from pydantic_ai.providers.alibaba import AlibabaProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`AlibabaProvider` wraps the [Alibaba Cloud Model Studio (DashScope)](https://www.alibabacloud.com/en/product/modelstudio) OpenAI-compatible endpoint, primarily for **Qwen** models. It automatically applies the correct `qwen_model_profile` and, for multimodal "Omni" models, forces URI-based audio input encoding.

### Constructor

```python
AlibabaProvider(
    *,
    api_key: str | None = None,          # ALIBABA_API_KEY or DASHSCOPE_API_KEY env var
    base_url: str | None = None,         # defaults to dashscope-intl.aliyuncs.com
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

**Key behaviour:** The provider accepts `DASHSCOPE_API_KEY` as a fallback for `ALIBABA_API_KEY` — matching Alibaba's own documentation convention. The default `base_url` points at the international endpoint `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`; for mainland China workloads, override to `https://dashscope.aliyuncs.com/compatible-mode/v1`.

### Automatic model profile

`AlibabaProvider.model_profile()` calls `qwen_model_profile(model_name)` and then wraps the result in `OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)` via `.update()`. For models whose name contains `'omni'`, an additional `openai_chat_audio_input_encoding='uri'` setting is applied — this forces the audio data to be sent as a URI reference rather than as a raw base-64 blob, which is required by the DashScope API for audio inputs.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.alibaba import AlibabaProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = AlibabaProvider(api_key='sk-your-dashscope-key')

# Standard Qwen text model
agent = Agent(OpenAIChatModel('qwen-plus', provider=provider))

async def main():
    result = await agent.run('Explain gradient descent in two sentences.')
    print(result.output)

asyncio.run(main())
```

### Using the DASHSCOPE_API_KEY environment variable

```python
import os
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.alibaba import AlibabaProvider
from pydantic_ai.models.openai import OpenAIChatModel

# AlibabaProvider reads DASHSCOPE_API_KEY automatically — no explicit api_key needed
os.environ['DASHSCOPE_API_KEY'] = 'sk-your-key'

provider = AlibabaProvider()  # picks up the env var
agent = Agent(OpenAIChatModel('qwen-turbo', provider=provider))

async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)

asyncio.run(main())
```

### Qwen Omni multimodal with URI audio encoding

```python
import asyncio
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.providers.alibaba import AlibabaProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = AlibabaProvider(api_key='sk-your-key')

# 'qwen-omni-turbo' matches 'omni' → model_profile forces audio_input_encoding='uri'
agent = Agent(OpenAIChatModel('qwen-omni-turbo', provider=provider))

async def main():
    with open('audio_sample.wav', 'rb') as f:
        audio_bytes = f.read()

    result = await agent.run([
        'Transcribe and summarise the following audio:',
        BinaryContent(data=audio_bytes, media_type='audio/wav'),
    ])
    print(result.output)

asyncio.run(main())
```

---

## 2. `OVHcloudProvider` — European Cloud AI Endpoints

**Module:** `pydantic_ai.providers.ovhcloud`  
**Import:**
```python
from pydantic_ai.providers.ovhcloud import OVHcloudProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`OVHcloudProvider` targets OVHcloud AI Endpoints (`oai.endpoints.kepler.ai.cloud.ovh.net/v1`) and applies prefix-based model-family routing so that the correct schema transformer and feature flags are set automatically for each hosted model family.

### Model profile routing

```python
# Source-verified prefix mapping from pydantic_ai.providers.ovhcloud:
prefix_to_profile = {
    'llama':    meta_model_profile,
    'meta-':    meta_model_profile,
    'deepseek': deepseek_model_profile,
    'mistral':  mistral_model_profile,
    'gpt':      harmony_model_profile,
    'qwen':     qwen_model_profile,
}
# All returned profiles are further wrapped in OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)
```

Models whose name does not match any prefix still receive `OpenAIModelProfile` with `OpenAIJsonSchemaTransformer` as a safe default.

### Constructor

```python
OVHcloudProvider(
    *,
    api_key: str | None = None,          # OVHCLOUD_API_KEY env var
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

### Routing to different model families

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.ovhcloud import OVHcloudProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = OVHcloudProvider(api_key='your-ovhcloud-token')

# Meta Llama — meta_model_profile applied
llama_agent = Agent(OpenAIChatModel('Meta-Llama-3.3-70B-Instruct', provider=provider))

# Mistral — mistral_model_profile applied
mistral_agent = Agent(OpenAIChatModel('Mistral-Nemo-Instruct-2407', provider=provider))

# DeepSeek — deepseek_model_profile applied (reasoning field support)
deepseek_agent = Agent(OpenAIChatModel('DeepSeek-R1', provider=provider))

async def main():
    result = await llama_agent.run('Summarise the benefits of open-source AI.')
    print(result.output)

asyncio.run(main())
```

### Structured output with Qwen on OVHcloud

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.providers.ovhcloud import OVHcloudProvider
from pydantic_ai.models.openai import OpenAIChatModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

provider = OVHcloudProvider(api_key='your-token')
agent = Agent(
    OpenAIChatModel('Qwen2.5-72B-Instruct', provider=provider),
    output_type=Summary,
)

async def main():
    result = await agent.run('Summarise the Python programming language.')
    print(result.output.title)
    print(result.output.key_points)

asyncio.run(main())
```

### Using a pre-built OpenAI client

```python
import asyncio
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.providers.ovhcloud import OVHcloudProvider
from pydantic_ai.models.openai import OpenAIChatModel

custom_client = AsyncOpenAI(
    api_key='your-token',
    base_url='https://oai.endpoints.kepler.ai.cloud.ovh.net/v1',
    timeout=120.0,
)
provider = OVHcloudProvider(openai_client=custom_client)
agent = Agent(OpenAIChatModel('Mixtral-8x22B-Instruct-v0.1', provider=provider))

async def main():
    result = await agent.run('What are mixture-of-experts models?')
    print(result.output)

asyncio.run(main())
```

---

## 3. `HerokuProvider` — Heroku Inference API

**Module:** `pydantic_ai.providers.heroku`  
**Import:**
```python
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`HerokuProvider` connects to the [Heroku Inference API](https://devcenter.heroku.com/articles/ai-inference-api) — an OpenAI-compatible endpoint bundled with Heroku's cloud platform. It always uses `OpenAIModelProfile` with `OpenAIJsonSchemaTransformer`.

### Constructor

```python
HerokuProvider(
    *,
    api_key: str | None = None,         # HEROKU_INFERENCE_KEY env var
    base_url: str | None = None,        # HEROKU_INFERENCE_URL env var; default us.inference.heroku.com
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

The `base_url` resolves as `(base_url or HEROKU_INFERENCE_URL or 'https://us.inference.heroku.com').rstrip('/') + '/v1'`.

### Basic usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = HerokuProvider(api_key='your-heroku-inference-key')
agent = Agent(OpenAIChatModel('claude-3-5-haiku', provider=provider))

async def main():
    result = await agent.run('Write a haiku about distributed systems.')
    print(result.output)

asyncio.run(main())
```

### Using environment variables (recommended for Heroku dynos)

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel

# In a Heroku dyno these are set automatically via the add-on config vars.
# os.environ['HEROKU_INFERENCE_KEY'] = '...'
# os.environ['HEROKU_INFERENCE_URL'] = 'https://us.inference.heroku.com'

provider = HerokuProvider()   # reads env vars automatically
agent = Agent(OpenAIChatModel('claude-3-5-sonnet', provider=provider))

async def main():
    result = await agent.run('Explain twelve-factor app methodology.')
    print(result.output)

asyncio.run(main())
```

### Structured output on Heroku Inference

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel

class AppReview(BaseModel):
    rating: int         # 1-5
    strengths: list[str]
    improvements: list[str]

provider = HerokuProvider(api_key='your-key')
agent = Agent(
    OpenAIChatModel('claude-3-5-haiku', provider=provider),
    output_type=AppReview,
    system_prompt='You are a senior developer reviewing apps.',
)

async def main():
    result = await agent.run('Review a todo-list app with offline sync.')
    print(result.output.rating)

asyncio.run(main())
```

---

## 4. `gateway_provider` — Pydantic AI Gateway

**Module:** `pydantic_ai.providers.gateway`  
**Import:**
```python
from pydantic_ai.providers.gateway import gateway_provider
```

`gateway_provider` is a **factory function** (not a class) that creates a provider pointing at the [Pydantic AI Gateway](https://gateway.pydantic.dev) — a managed proxy that routes requests to upstream cloud providers. The correct provider sub-type (OpenAI, Anthropic, Groq, Bedrock, Google Cloud) is selected from the `upstream_provider` argument.

### Signature

```python
gateway_provider(
    upstream_provider: str,   # 'openai' | 'anthropic' | 'groq' | 'bedrock' | 'google-cloud' | ...
    /,
    *,
    route: str | None = None,       # override the routing group; defaults to normalized upstream_provider
    api_key: str | None = None,     # PYDANTIC_AI_GATEWAY_API_KEY or PAIG_API_KEY env var
    base_url: str | None = None,    # PYDANTIC_AI_GATEWAY_BASE_URL; inferred from key region if absent
    http_client: httpx.AsyncClient | None = None,
) -> Provider[Any]
```

The returned type varies: `Provider[AsyncOpenAI]` for OpenAI/Groq, `Provider[AsyncAnthropicClient]` for Anthropic, `Provider[BaseClient]` for Bedrock, `Provider[GoogleClient]` for Google Cloud.

### Region-encoded API key and automatic base-URL inference

API keys issued by the Pydantic AI Gateway follow the pattern `pylf_v<version>_<region>_<token>`. The `_infer_base_url()` helper parses this pattern to derive the correct regional endpoint automatically:

- `pylf_v1_us_…` → `https://gateway-us.pydantic.dev/proxy`
- `pylf_v1_eu_…` → `https://gateway-eu.pydantic.dev/proxy`
- `pylf_v1_staging_…` → `https://gateway.pydantic.info/proxy`

If the key does not encode a region and `PYDANTIC_AI_GATEWAY_BASE_URL` is not set, a `UserError` is raised.

### Routing to OpenAI via the Gateway

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.openai import OpenAIChatModel

provider = gateway_provider('openai', api_key='pylf_v1_us_your-key')
agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))

async def main():
    result = await agent.run('Explain the Pydantic AI Gateway in one paragraph.')
    print(result.output)

asyncio.run(main())
```

### Routing to Anthropic via the Gateway

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.anthropic import AnthropicModel

provider = gateway_provider('anthropic', api_key='pylf_v1_eu_your-key')
agent = Agent(AnthropicModel('claude-opus-4-8', provider=provider))

async def main():
    result = await agent.run('What is a good system design for a chat application?')
    print(result.output)

asyncio.run(main())
```

### Routing to Bedrock via the Gateway

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.bedrock import BedrockConverseModel

# Bedrock uses the AWS SDK (botocore), so no http_client is passed.
provider = gateway_provider('bedrock', api_key='pylf_v1_us_your-key')
agent = Agent(BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20251001-v1:0', provider=provider))

async def main():
    result = await agent.run('Summarise quantum computing in three bullet points.')
    print(result.output)

asyncio.run(main())
```

### Using a custom routing group

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.openai import OpenAIChatModel

# route='my-team-budget' sends requests to a custom routing group configured
# in the Gateway dashboard; upstream_provider still determines the client type.
provider = gateway_provider(
    'openai',
    api_key='pylf_v1_us_your-key',
    route='my-team-budget',
)
agent = Agent(OpenAIChatModel('gpt-4o-mini', provider=provider))

async def main():
    result = await agent.run('Draft a meeting agenda for a sprint retrospective.')
    print(result.output)

asyncio.run(main())
```

---

## 5. `GoogleProvider` + `GoogleCloudProvider` — Google Model Providers

**Module:** `pydantic_ai.providers.google`, `pydantic_ai.providers.google_cloud`  
**Import:**
```python
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
```

Both providers extend `BaseGoogleProvider[Client]` (from `google.genai`). The shared base handles `base_url` derivation, user-agent injection, and `HttpOptions` wiring. `GoogleProvider` targets the **Gemini API** (formerly Google AI Studio / GLA); `GoogleCloudProvider` targets **Google Cloud / Vertex AI**.

### `GoogleProvider` — Gemini API

```python
GoogleProvider(
    *,
    api_key: str | None = None,          # GOOGLE_API_KEY env var
    client: Client | None = None,        # pre-built google.genai.Client
    http_client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
)
```

Passing `vertexai=True`, `location=`, `project=`, or `credentials=` on `GoogleProvider` is **deprecated** in 1.x — use `GoogleCloudProvider` instead.

### `GoogleCloudProvider` — Vertex AI / Google Cloud

```python
GoogleCloudProvider(
    *,
    api_key: str | None = None,          # Express Mode key; GOOGLE_API_KEY env var
    credentials: Credentials | None = None,   # ADC credentials
    project: str | None = None,          # GOOGLE_CLOUD_PROJECT env var
    location: str | None = None,         # GOOGLE_CLOUD_LOCATION env var (e.g. 'us-central1')
    client: Client | None = None,
    http_client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
)
```

When `credentials`/`project`/`location` are used (Application Default Credentials path), `api_key` must be `None`. When `api_key` is set (Express Mode), `credentials`/`project`/`location` must be `None`.

### Gemini API with `GoogleProvider`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel

provider = GoogleProvider(api_key='AIza-your-gemini-key')
agent = Agent(GoogleModel('gemini-2.5-flash', provider=provider))

async def main():
    result = await agent.run('Explain transformer self-attention in two paragraphs.')
    print(result.output)

asyncio.run(main())
```

### Vertex AI with ADC credentials

```python
import asyncio
from google.auth import default as google_auth_default
from pydantic_ai import Agent
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.models.google import GoogleModel

credentials, project = google_auth_default()
provider = GoogleCloudProvider(
    credentials=credentials,
    project=project,
    location='us-central1',
)
agent = Agent(GoogleModel('gemini-2.5-pro', provider=provider))

async def main():
    result = await agent.run('What is federated learning?')
    print(result.output)

asyncio.run(main())
```

### Google Cloud Express Mode (API key on Vertex)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.models.google import GoogleModel

# Express Mode: API key on Google Cloud (no service account needed)
provider = GoogleCloudProvider(api_key='your-vertex-express-key')
agent = Agent(GoogleModel('gemini-3.5-flash', provider=provider))

async def main():
    result = await agent.run('List five real-world uses of LLMs in healthcare.')
    print(result.output)

asyncio.run(main())
```

---

## 6. `GoogleModel` + `GoogleModelSettings` — Google Gemini Model

**Module:** `pydantic_ai.models.google`  
**Import:**
```python
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
```

`GoogleModel` uses the `google-genai` SDK (`google.genai.Client`) for both streaming and non-streaming requests. It supports all Gemini native tools (WebSearch, WebFetch, FileSearch, CodeExecution, ImageGeneration) and the full range of provider-specific settings.

### `GoogleModelSettings` fields

All nine fields carry the `google_` prefix so they merge cleanly with other model settings:

| Field | Type | Purpose |
|---|---|---|
| `google_safety_settings` | `list[SafetySettingDict]` | Per-category harm thresholds |
| `google_thinking_config` | `ThinkingConfigDict` | Extended-thinking budget/mode |
| `google_labels` | `dict[str, str]` | Billing breakdown labels (Vertex only) |
| `google_video_resolution` | `MediaResolution` | Video frame resolution |
| `google_cached_content` | `str` | Cached content resource name (see caveat) |
| `google_logprobs` | `bool` | Include log-probabilities (non-streaming + Vertex only) |
| `google_top_logprobs` | `int` | Number of top-token log-probs |
| `google_cloud_service_tier` | `GoogleCloudServiceTier` | PT/Flex/Priority routing (Cloud only) |
| `google_service_tier` | `GoogleServiceTier` | **Deprecated** — use `service_tier` or `google_cloud_service_tier` |

### `google_cached_content` caveat

When `google_cached_content` is set, **the model strips `system_instruction`, `tools`, and `tool_config` from the outgoing request** — both the Gemini API and Vertex AI reject requests that include those fields alongside a cached-content reference. Any tools registered on the agent and any system prompt are dropped from the request; a `UserWarning` is emitted whenever stripping actually removes a field.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='AIza-your-key')

# google_cached_content: use a pre-warmed context cache
# NOTE: system prompts and tools are stripped from requests that use the cache.
agent = Agent(
    GoogleModel(
        'gemini-2.5-flash',
        provider=provider,
        settings=GoogleModelSettings(
            google_cached_content='cachedContents/abc123',
        ),
    )
)

async def main():
    result = await agent.run('Based on the cached document, what are the key findings?')
    print(result.output)

asyncio.run(main())
```

### Safety settings

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from google.genai.types import SafetySettingDict

provider = GoogleProvider(api_key='AIza-your-key')

agent = Agent(
    GoogleModel(
        'gemini-2.5-flash',
        provider=provider,
        settings=GoogleModelSettings(
            google_safety_settings=[
                SafetySettingDict(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_ONLY_HIGH'),
                SafetySettingDict(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_ONLY_HIGH'),
            ],
        ),
    ),
    system_prompt='You are a research assistant.',
)

async def main():
    result = await agent.run('Explain the history of chemical weapons treaties.')
    print(result.output)

asyncio.run(main())
```

### Extended thinking with `google_thinking_config`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='AIza-your-key')

agent = Agent(
    GoogleModel(
        'gemini-2.5-pro',
        provider=provider,
        settings=GoogleModelSettings(
            google_thinking_config={'thinking_budget': 8192},
        ),
    )
)

async def main():
    result = await agent.run('Solve: find all integer pairs (x, y) where x² + y² = 100.')
    print(result.output)

asyncio.run(main())
```

---

## 7. `CohereModel` + `CohereProvider` + `CohereModelSettings` — Cohere Command Models

**Module:** `pydantic_ai.models.cohere`, `pydantic_ai.providers.cohere`  
**Import:**
```python
from pydantic_ai.models.cohere import CohereModel, CohereModelSettings
from pydantic_ai.providers.cohere import CohereProvider
```

`CohereModel` drives the Cohere v2 chat API via `cohere.AsyncClientV2`. It supports tool calling, thinking content (mapped from `ThinkingAssistantMessageV2ContentOneItem`), and all standard `ModelSettings` fields that Cohere accepts.

### `CohereProvider` constructor

```python
CohereProvider(
    *,
    api_key: str | None = None,           # CO_API_KEY env var
    cohere_client: AsyncClientV2 | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

`CO_BASE_URL` env var overrides the API endpoint. `CohereProvider` also exposes a `.v1_client` (the deprecated v1 `AsyncClient`) for backward compatibility.

### `CohereModel` constructor

```python
CohereModel(
    model_name: CohereModelName,            # e.g. 'command-r-plus-08-2024'
    *,
    provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
    profile: ModelProfileSpec | None = None,
    settings: ModelSettings | None = None,
)
```

`LatestCohereModelNames` includes: `'command-r-plus-08-2024'`, `'command-r-08-2024'`, `'command-r7b-12-2024'`, `'command-nightly'`, `'c4ai-aya-expanse-32b'`, `'c4ai-aya-expanse-8b'`.

### Settings passed to the Cohere chat endpoint

`CohereModel._chat()` maps `ModelSettings` keys to Cohere's `AsyncClientV2.chat()` parameters:
- `max_tokens` → `max_tokens`
- `stop_sequences` → `stop_sequences`
- `temperature` → `temperature`
- `top_p` → `p`
- `top_k` → `k`
- `seed` → `seed`
- `presence_penalty` → `presence_penalty`
- `frequency_penalty` → `frequency_penalty`

### Basic usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

provider = CohereProvider(api_key='your-co-api-key')
agent = Agent(CohereModel('command-r-plus-08-2024', provider=provider))

async def main():
    result = await agent.run('Explain Retrieval-Augmented Generation in three sentences.')
    print(result.output)

asyncio.run(main())
```

### Structured output with Cohere

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

class TechSummary(BaseModel):
    topic: str
    summary: str
    complexity: str   # 'beginner' | 'intermediate' | 'advanced'

provider = CohereProvider(api_key='your-key')
agent = Agent(
    CohereModel('command-r-plus-08-2024', provider=provider),
    output_type=TechSummary,
)

async def main():
    result = await agent.run('Summarise vector databases for an AI engineer.')
    print(result.output.topic, '-', result.output.complexity)

asyncio.run(main())
```

### Tool use with Cohere Command

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

provider = CohereProvider(api_key='your-key')
agent = Agent(
    CohereModel('command-r-plus-08-2024', provider=provider),
    system_prompt='You are a helpful assistant with access to a product catalogue.',
)

@agent.tool_plain
def get_product_price(product_name: str) -> str:
    """Return the price of a product."""
    prices = {'widget': '$9.99', 'gadget': '$24.99', 'doohickey': '$4.49'}
    return prices.get(product_name.lower(), 'Product not found')

async def main():
    result = await agent.run('How much does a gadget cost?')
    print(result.output)

asyncio.run(main())
```

---

## 8. `XaiProvider` + `XaiModel` — xAI Grok via gRPC SDK

**Module:** `pydantic_ai.providers.xai`, `pydantic_ai.models.xai`  
**Import:**
```python
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.models.xai import XaiModel, XaiModelSettings
```

`XaiModel` is the only model in pydantic-ai that uses a **gRPC transport** (`xai-sdk`) rather than an HTTP client. This shapes both the provider design (a `_LazyAsyncClient` that binds per event-loop) and the error handling (gRPC status codes mapped to HTTP equivalents).

### `_LazyAsyncClient` — per-event-loop gRPC channel

gRPC async channels bind to the asyncio event loop at creation time. If the `AsyncClient` is created outside an async context (e.g. at module level) and later used inside `asyncio.run()`, the loop will differ and cause a `RuntimeError`. `XaiProvider` wraps its client in `_LazyAsyncClient`, which defers construction and re-creates the `AsyncClient` whenever the running loop changes.

### `XaiProvider` constructor

```python
XaiProvider(
    *,
    api_key: str | None = None,       # XAI_API_KEY env var
    api_host: str | None = None,      # gRPC host override
    timeout: float | None = None,
    xai_client: AsyncClient | None = None,  # pre-built xai_sdk.AsyncClient
)
```

The provider's `base_url` is always `'https://api.x.ai/v1'` — this is a **canonical label** for pricing/telemetry, not the actual gRPC channel target.

### `GrokModelProfile` and `GrokReasoningEffort`

```python
from pydantic_ai.profiles.grok import GrokModelProfile, GrokReasoningEffort

# GrokModelProfile fields:
# grok_supports_builtin_tools: bool    — web_search, x_search, code_execution, mcp supported?
# grok_supports_tool_choice_required: bool — does the provider accept tool_choice='required'?
# grok_reasoning_efforts: frozenset[GrokReasoningEffort]  — valid effort levels

# GrokReasoningEffort: Literal['none', 'low', 'medium', 'high']
# - grok-4.3 / grok-4-family: frozenset({'none', 'low', 'medium', 'high'})
# - older grok models: frozenset({'low', 'high'})
# - non-reasoning models: frozenset() (empty)
```

Models that match `'grok-4'`, `'code'`, `'build'`, or are in the `_GROK_43_REASONING_MODELS` set get `grok_supports_builtin_tools=True`.

### Basic usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.models.xai import XaiModel

# XAI_API_KEY env var is read automatically if api_key is omitted
provider = XaiProvider(api_key='xai-your-key')
agent = Agent(XaiModel('grok-3', provider=provider))

async def main():
    result = await agent.run('What are the design principles of the Rust programming language?')
    print(result.output)

asyncio.run(main())
```

### Using Grok 4.3 with reasoning effort

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.settings import ModelSettings

provider = XaiProvider(api_key='xai-your-key')
agent = Agent(
    XaiModel('grok-4.3', provider=provider),
    model_settings=ModelSettings(thinking='high'),  # maps to GrokModelProfile reasoning_effort via unified thinking
)

async def main():
    result = await agent.run('Prove that there are infinitely many prime numbers.')
    print(result.output)

asyncio.run(main())
```

### Web search with Grok builtin tools

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.capabilities.x_search import XSearch

provider = XaiProvider(api_key='xai-your-key')

# grok-4.3 has grok_supports_builtin_tools=True
agent = Agent(
    XaiModel('grok-4.3', provider=provider),
    capabilities=[XSearch()],   # activates xAI native X/web search
)

async def main():
    result = await agent.run('What are the latest AI model releases from this week?')
    print(result.output)

asyncio.run(main())
```

---

## 9. `FileStatePersistence` — File-based Graph Persistence

**Module:** `pydantic_graph.persistence.file`  
**Import:**
```python
from pydantic_graph.persistence.file import FileStatePersistence
```

`FileStatePersistence` stores `pydantic_graph` run snapshots in a JSON file, enabling graph runs to **survive process restarts**. It uses a lightweight `.pydantic-graph-persistence-lock` advisory lock file to coordinate concurrent writers.

### Constructor

```python
@dataclass
class FileStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    json_file: Path   # one file per graph run; reused across steps of the same run
```

Types are registered before the first write via `set_graph_types(graph)` (higher-level helper that extracts the type args automatically) or the lower-level `set_types(state_type, run_end_type)` — the `Graph.run()` machinery calls this automatically so manual registration is only needed when deserialising snapshots outside of a normal run. The internal `pydantic.TypeAdapter` handles polymorphic `Snapshot[StateT, RunEndT]` serialisation.

### File locking protocol

`_lock()` creates a `.pydantic-graph-persistence-lock` sibling file using `anyio.open_file(mode='ab')` (append-binary). Because file-level `append + check` is atomic on most Unix systems, the first writer wins; subsequent writers spin with a `10 ms` sleep until the lock is released or the 1 s `timeout` is exceeded. The lock file is deleted in a `finally` block.

### Snapshot lifecycle

```
snapshot_node()          → appends a NodeSnapshot(status='created')
record_run(snapshot_id)  → transitions status: 'created' → 'running'
                           records start_ts
[node runs]
record_run exits         → transitions status: 'running' → 'success'/'error'
                           records duration
snapshot_end()           → appends EndSnapshot
```

### Basic file persistence

```python
import asyncio
from pathlib import Path
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End
from pydantic_graph.persistence.file import FileStatePersistence

@dataclass
class CountState:
    count: int = 0

@dataclass
class Increment(BaseNode[CountState, None, int]):
    async def run(self, ctx) -> 'Increment | End[int]':
        ctx.state.count += 1
        if ctx.state.count >= 5:
            return End(ctx.state.count)
        return Increment()

graph = Graph(nodes=[Increment])

async def main():
    run_id = 'demo-run-001'
    persistence = FileStatePersistence(Path('runs') / f'{run_id}.json')
    Path('runs').mkdir(exist_ok=True)

    state = CountState()
    run_result = await graph.run(Increment(), state=state, persistence=persistence)
    print(f'Final count: {run_result.output}')   # 5
    print(f'Snapshots: {persistence.json_file.read_text()[:80]}...')

asyncio.run(main())
```

### Resuming an interrupted run

```python
import asyncio
from pathlib import Path
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End, GraphRuntimeError
from pydantic_graph.persistence.file import FileStatePersistence

@dataclass
class WorkState:
    items_processed: int = 0

@dataclass
class ProcessItem(BaseNode[WorkState, None, str]):
    async def run(self, ctx) -> 'ProcessItem | End[str]':
        ctx.state.items_processed += 1
        if ctx.state.items_processed >= 3:
            return End(f'Processed {ctx.state.items_processed} items')
        return ProcessItem()

graph = Graph(nodes=[ProcessItem])

async def resume_or_start(run_id: str) -> str:
    persistence = FileStatePersistence(Path(f'{run_id}.json'))

    try:
        # iter_from_persistence calls load_next() internally; raises
        # GraphRuntimeError when no 'created' snapshot exists — which covers
        # both a brand-new run and a run whose file already contains only
        # completed snapshots.
        print('Attempting to resume...')
        async with graph.iter_from_persistence(persistence) as run:
            async for _ in run:
                pass
        return run.result.output
    except GraphRuntimeError:
        print('No pending snapshot — starting fresh run')
        run_result = await graph.run(ProcessItem(), state=WorkState(), persistence=persistence)
        return run_result.output

asyncio.run(resume_or_start('work-run-001'))
```

### Inspecting raw snapshot JSON

```python
import asyncio
from pathlib import Path
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End
from pydantic_graph.persistence.file import FileStatePersistence
import json

@dataclass
class SimpleState:
    step: int = 0

@dataclass
class Step(BaseNode[SimpleState, None, str]):
    async def run(self, ctx) -> 'Step | End[str]':
        ctx.state.step += 1
        return End('done') if ctx.state.step >= 2 else Step()

graph = Graph(nodes=[Step])

async def main():
    p = FileStatePersistence(Path('inspect-run.json'))
    await graph.run(Step(), state=SimpleState(), persistence=p)
    data = json.loads(p.json_file.read_bytes())
    for snap in data:
        if snap.get('kind') == 'node':
            print(snap['status'], snap.get('node', {}).get('__class__', '?'))
        else:   # EndSnapshot: no 'status' field
            print('end', snap.get('result'))

asyncio.run(main())
```

---

## 10. `SimpleStatePersistence` + `FullStatePersistence` — In-Memory Graph Persistence

**Module:** `pydantic_graph.persistence.in_mem`  
**Import:**
```python
from pydantic_graph.persistence.in_mem import SimpleStatePersistence, FullStatePersistence
```

Two in-memory persistence implementations that trade history for simplicity:

| | `SimpleStatePersistence` | `FullStatePersistence` |
|---|---|---|
| Storage | Single `last_snapshot` | `list[Snapshot]` history |
| `load_all()` | `NotImplementedError` | Returns entire history |
| `dump_json()` | Not available | Serialises full history |
| `load_json()` | Not available | Deserialises full history |
| Default? | **Yes** (used when no persistence is provided) | No |
| Deep copy | No | Yes (`deep_copy=True` by default) |

### `SimpleStatePersistence` — lightweight default

When `Graph.run()` is called without a `persistence` argument, pydantic_graph uses `SimpleStatePersistence` internally. It stores only the most recent snapshot, so you cannot replay or inspect history. `load_all()` raises `NotImplementedError`.

```python
import asyncio
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End
from pydantic_graph.persistence.in_mem import SimpleStatePersistence

@dataclass
class RunState:
    value: int = 0

@dataclass
class DoubleIt(BaseNode[RunState, None, int]):
    async def run(self, ctx) -> 'DoubleIt | End[int]':
        ctx.state.value = ctx.state.value * 2 or 1
        if ctx.state.value >= 8:
            return End(ctx.state.value)
        return DoubleIt()

graph = Graph(nodes=[DoubleIt])

async def main():
    persistence = SimpleStatePersistence()
    run_result = await graph.run(DoubleIt(), state=RunState(value=1), persistence=persistence)
    print(run_result.output)               # 8
    print(persistence.last_snapshot)       # EndSnapshot with state.value=8

asyncio.run(main())
```

### `FullStatePersistence` — complete history

`FullStatePersistence` keeps all snapshots in its `.history` list. `deep_copy=True` (default) ensures that mutations to state after a snapshot is taken do not retroactively alter the recorded history.

```python
import asyncio
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End
from pydantic_graph.persistence.in_mem import FullStatePersistence
from pydantic_graph.persistence import NodeSnapshot

@dataclass
class CountState:
    n: int = 0

@dataclass
class Count(BaseNode[CountState, None, int]):
    async def run(self, ctx) -> 'Count | End[int]':
        ctx.state.n += 1
        return End(ctx.state.n) if ctx.state.n >= 3 else Count()

graph = Graph(nodes=[Count])

async def main():
    persistence = FullStatePersistence()
    run_result = await graph.run(Count(), state=CountState(), persistence=persistence)

    print(f'Result: {run_result.output}')   # 3
    print(f'Snapshots taken: {len(persistence.history)}')
    for snap in persistence.history:
        if isinstance(snap, NodeSnapshot):
            print(type(snap).__name__, snap.status, snap.state.n)
        else:   # EndSnapshot has no 'status' field
            print(type(snap).__name__, snap.state.n)

asyncio.run(main())
```

### JSON round-trip with `FullStatePersistence`

```python
import asyncio
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End
from pydantic_graph.persistence.in_mem import FullStatePersistence

@dataclass
class PingState:
    pings: int = 0

@dataclass
class Ping(BaseNode[PingState, None, str]):
    async def run(self, ctx) -> 'Ping | End[str]':
        ctx.state.pings += 1
        return End('pong') if ctx.state.pings >= 2 else Ping()

graph = Graph(nodes=[Ping])

async def main():
    p1 = FullStatePersistence()
    await graph.run(Ping(), state=PingState(), persistence=p1)

    # Serialise history to bytes
    raw = p1.dump_json(indent=2)
    print(raw.decode()[:120], '...')

    # Deserialise into a new instance
    p2 = FullStatePersistence()
    p2.set_graph_types(graph)   # must set types before load_json
    p2.load_json(raw)
    print(f'Restored {len(p2.history)} snapshots')

asyncio.run(main())
```

### `deep_copy=False` for performance

`deep_copy=True` (default) deep-copies the state object before recording each snapshot so that later in-place mutations don't retroactively alter historical records. Set `deep_copy=False` to skip that copy when you know you won't read snapshot state values after the run.

```python
import asyncio
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End
from pydantic_graph.persistence.in_mem import FullStatePersistence

@dataclass
class CountState:
    n: int = 0

@dataclass
class Node(BaseNode[CountState, None, int]):
    async def run(self, ctx) -> 'Node | End[int]':
        ctx.state.n += 1
        return End(ctx.state.n) if ctx.state.n >= 3 else Node()

graph = Graph(nodes=[Node])

async def main():
    # deep_copy=False avoids the per-snapshot deep copy.  All NodeSnapshot.state
    # entries then share the same object, so they all reflect the final state value.
    persistence = FullStatePersistence(deep_copy=False)
    run_result = await graph.run(Node(), state=CountState(), persistence=persistence)
    print(run_result.output)   # 3

    node_snaps = [s for s in persistence.history if hasattr(s, 'node')]
    print([s.state.n for s in node_snaps])   # [3, 3, 3] — all share the final state

asyncio.run(main())
```

---

## Revision history

| Version | Date | Changes |
|---|---|---|
| 1.107.0 | 2026-06-17 | Initial publication — `AlibabaProvider`, `OVHcloudProvider`, `HerokuProvider`, `gateway_provider`, `GoogleProvider`/`GoogleCloudProvider`, `GoogleModel`/`GoogleModelSettings`, `CohereModel`/`CohereProvider`, `XaiProvider`/`XaiModel`, `FileStatePersistence`, `SimpleStatePersistence`/`FullStatePersistence` |

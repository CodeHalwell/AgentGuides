---
title: "PydanticAI — Class Deep Dives Vol. 17"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.107.0: LiteLLMProvider (universal proxy routing + automatic model profiles), AzureProvider (Azure AI Foundry Express Mode + ADC), DeepSeekProvider (reasoning_content field + send_back_thinking_parts + v4 thinking effort), CerebrasProvider (ultra-fast inference + zai/gpt-oss reasoning + disabled settings), GitHubProvider (provider/model:tag naming + multi-family support), FireworksProvider + TogetherProvider + NebiusProvider + SambaNovaProvider (OpenAI-compatible GPU provider quartet), GraphBuilder (pydantic_graph fluent builder — step/stream decorators + Fork/Join wiring + build()), Fork + Join + ReducerContext (parallel fan-out/fan-in), Decision + DecisionBranch + Edge + TypeExpression (conditional routing), Step + StepContext + StepNode (step execution primitives). All verified against pydantic-ai 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 17)"
  order: 43
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups spanning the complete provider ecosystem and the new `pydantic_graph` builder API: `LiteLLMProvider` (the universal proxy that auto-routes to any backend with correct model profiles); `AzureProvider` (Azure AI Foundry — Express Mode API keys, ADC credentials, and the `api_version` suppression rule for `/v1`-suffix endpoints); `DeepSeekProvider` (the `reasoning_content` wire-field, `send_back_thinking_parts='field'`, per-model `tool_choice=required` restriction); `CerebrasProvider` (ultra-fast inference with `X-Cerebras-3rd-Party-Integration` header, `zai`/`gpt-oss` reasoning models, six disabled settings); `GitHubProvider` (GitHub Models — `provider/model:tag` naming, six provider families, `GITHUB_API_KEY` env var); `FireworksProvider` + `TogetherProvider` + `NebiusProvider` + `SambaNovaProvider` (the OpenAI-compatible community GPU provider quartet — endpoint URLs, `accounts/fireworks/models/` prefix stripping, `SAMBANOVA_BASE_URL` custom endpoint); `GraphBuilder` (`pydantic_graph`'s fluent builder — `step()` / `stream()` decorators, `join()` / `decision()` helpers, `add()` / `add_edge()` / `add_mapping_edge()` wiring, `build()`); `Fork` + `Join` + `ReducerContext` (parallel fan-out with `is_map`, `cancel_sibling_tasks()` for early stopping, `preferred_parent_fork`); `Decision` + `DecisionBranch` + `Edge` + `TypeExpression` (conditional routing — type-based dispatch with `Literal` types, custom `matches` predicate, `Edge` labels for Mermaid, `TypeExpression` workaround for complex union types); `Step` + `StepContext` + `StepNode` (step execution primitives — `StepContext.state`/`deps`/`inputs`, `Step.as_node()`, streaming steps).

<Aside type="note" title="Provider pattern">
Providers are **not** passed directly to `Agent`. Pass the provider to the model constructor (e.g. `OpenAIChatModel('name', provider=provider)`) and hand the resulting model object to `Agent`. All examples below follow this pattern.
</Aside>

---

## 1. `LiteLLMProvider` — Universal Proxy with Automatic Model Profiles

**Module:** `pydantic_ai.providers.litellm`  
**Import:**
```python
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`LiteLLMProvider` is the universal gateway that lets you point a single `OpenAIChatModel` at [LiteLLM Proxy](https://docs.litellm.ai/docs/proxy/quick_start) and route calls to any backend (Anthropic, Gemini, Bedrock, Azure, local Ollama) without changing your agent code. It wraps an `AsyncOpenAI` client pointed at the LiteLLM endpoint and implements `model_profile()` to return the correct schema transformer for whatever backend the model name prefix implies.

### Constructor

```python
LiteLLMProvider(
    *,
    api_key: str | None = None,       # passed to AsyncOpenAI; None → 'litellm-placeholder'
    api_base: str | None = None,      # LiteLLM proxy URL, e.g. 'http://localhost:4000'
    openai_client: AsyncOpenAI | None = None,  # pre-built client; ignores other params
    http_client: httpx.AsyncClient | None = None,
)
```

When `openai_client` is `None`, the provider creates an `AsyncOpenAI` client pointing at `api_base`. If no `api_key` is provided it uses `'litellm-placeholder'` — LiteLLM Proxy typically reads its own API keys from its config, so the placeholder prevents the `openai` SDK from raising a missing-key error.

### Automatic model profile dispatch

`LiteLLMProvider.model_profile()` parses the `provider/model` prefix from the model name and delegates to the matching family profile:

```python
from pydantic_ai import Agent
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = LiteLLMProvider(api_base='http://localhost:4000', api_key='sk-litellm-key')

# Route to Anthropic Claude via LiteLLM — provider sets AnthropicModelProfile automatically
agent = Agent(OpenAIChatModel('anthropic/claude-opus-4-8', provider=provider))

# Route to Google Gemini — GoogleModelProfile applied
agent_gemini = Agent(OpenAIChatModel('google/gemini-2.0-flash', provider=provider))

# Route to Bedrock — AmazonModelProfile applied
agent_bedrock = Agent(OpenAIChatModel('bedrock/us.amazon.nova-pro-v1:0', provider=provider))

# Route to local Ollama — falls back to OpenAIModelProfile (no prefix match)
agent_ollama = Agent(OpenAIChatModel('ollama/llama3.2', provider=provider))
```

The prefix-to-profile mapping:

| Prefix | Profile function |
|--------|-----------------|
| `anthropic/` | `anthropic_model_profile` |
| `openai/` | `openai_model_profile` |
| `google/` | `google_model_profile` |
| `mistralai/`, `mistral/` | `mistral_model_profile` |
| `cohere/` | `cohere_model_profile` |
| `amazon/`, `bedrock/` | `amazon_model_profile` |
| `meta-llama/`, `meta/` | `meta_model_profile` |
| `groq/` | `groq_model_profile` |
| `deepseek/` | `deepseek_model_profile` |
| `moonshotai/` | `moonshotai_model_profile` |
| `x-ai/` | `grok_model_profile` |
| `qwen/` | `qwen_model_profile` |
| *(no prefix match)* | `openai_model_profile` |

### Pointing at a local LiteLLM proxy

```python
import os
import asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.models.openai import OpenAIChatModel

# LiteLLM Proxy running locally with master key in config.yaml
provider = LiteLLMProvider(
    api_base=os.getenv('LITELLM_PROXY_URL', 'http://localhost:4000'),
    api_key=os.getenv('LITELLM_API_KEY', 'sk-my-master-key'),
)

agent = Agent(
    OpenAIChatModel('anthropic/claude-sonnet-4-6', provider=provider),
    system_prompt='You are a helpful assistant.',
)

async def main():
    result = await agent.run('Summarise the benefits of type safety in Python.')
    print(result.output)

asyncio.run(main())
```

### Using a pre-built `AsyncOpenAI` client

When you need fine-grained control over connection pooling or headers, pass in a pre-configured client:

```python
import httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.models.openai import OpenAIChatModel

custom_http = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    headers={'X-Custom-Header': 'my-value'},
)
openai_client = AsyncOpenAI(
    base_url='http://localhost:4000',
    api_key='sk-litellm',
    http_client=custom_http,
)
provider = LiteLLMProvider(openai_client=openai_client)

agent = Agent(OpenAIChatModel('mistralai/mistral-large-latest', provider=provider))
```

---

## 2. `AzureProvider` — Azure AI Foundry and Azure OpenAI

**Module:** `pydantic_ai.providers.azure`  
**Import:**
```python
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`AzureProvider` wraps `AsyncAzureOpenAI` and adds PydanticAI model profiles for non-OpenAI models deployed on Azure (Llama, DeepSeek, Mistral, Cohere, Grok). It supports two authentication paths: API key and Azure AD / Entra ID via `openai_client`.

### Constructor

```python
AzureProvider(
    *,
    azure_endpoint: str | None = None,   # reads AZURE_OPENAI_ENDPOINT if None
    api_version: str | None = None,       # reads OPENAI_API_VERSION if None
    api_key: str | None = None,           # reads AZURE_OPENAI_API_KEY if None
    openai_client: AsyncAzureOpenAI | None = None,  # bypasses all other params
    http_client: httpx.AsyncClient | None = None,
)
```

### API-key authentication (simplest case)

```python
import os
from pydantic_ai import Agent
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = AzureProvider(
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],  # e.g. https://<resource>.openai.azure.com/
    api_version='2025-04-01-preview',
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
)

agent = Agent(OpenAIChatModel('gpt-4.5', provider=provider))
result = agent.run_sync('List the Azure regions with lowest latency.')
print(result.output)
```

### Express Mode (`/v1`-suffix endpoint — no `api_version` sent)

Azure's GA API (`/openai/v1/`) and Azure AI Foundry serverless model endpoints (`*.models.ai.azure.com`) reject the `api-version` query parameter. AzureProvider detects the `/v1` suffix and suppresses `api_version` automatically:

```python
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

# Azure OpenAI GA API (v1 suffix) — api_version is NOT sent in requests
provider = AzureProvider(
    azure_endpoint='https://my-resource.openai.azure.com/openai/v1/',
    api_key='my-azure-key',
    # api_version MUST be omitted for /v1 endpoints — the constructor raises
    # UserError if you pass it, because these endpoints reject api-version params.
    # OPENAI_API_VERSION env var is also ignored on this code path.
)
agent = Agent(OpenAIChatModel('gpt-4.1', provider=provider))
```

### Azure AI Foundry serverless model endpoint

```python
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

# Serverless endpoint — no api_version, model name is just the deployment
provider = AzureProvider(
    azure_endpoint='https://my-deployment.models.ai.azure.com',
    api_key='my-serverless-key',
)
# DeepSeek-V3-0324 on Azure — profile sets reasoning_content field automatically
agent = Agent(OpenAIChatModel('DeepSeek-V3-0324', provider=provider))
```

### Azure AD / Entra ID authentication

Pass a pre-built `AsyncAzureOpenAI` client configured with an `azure.identity` credential:

```python
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    'https://cognitiveservices.azure.com/.default',
)
client = AsyncAzureOpenAI(
    azure_endpoint='https://my-resource.openai.azure.com/',
    azure_ad_token_provider=token_provider,
    api_version='2025-04-01-preview',
)
provider = AzureProvider(openai_client=client)
agent = Agent(OpenAIChatModel('gpt-4.5-turbo', provider=provider))
```

### Non-OpenAI models and model profiles

`AzureProvider.model_profile()` recognises these family prefixes on Azure deployments:

| Deployment prefix | Profile applied |
|-------------------|----------------|
| `llama`, `meta-` | `meta_model_profile` |
| `deepseek` | `deepseek_model_profile` |
| `mistralai-`, `mistral` | `mistral_model_profile` |
| `cohere-` | `cohere_model_profile` |
| `grok` | `grok_model_profile` |
| *(any other)* | `openai_model_profile` |

Note: all Azure profiles disable `openai_chat_supports_document_input` since the Azure Chat Completions API does not support document inputs.

```python
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

provider = AzureProvider(
    azure_endpoint='https://my-resource.openai.azure.com/',
    api_version='2025-04-01-preview',
    api_key='my-key',
)
# Llama on Azure — meta_model_profile applied (no JSON schema transformer quirks)
llama_agent = Agent(OpenAIChatModel('Llama-3.1-8B-Instruct', provider=provider))
# Mistral on Azure — mistral_model_profile, PromptedOutput workaround enabled
mistral_agent = Agent(OpenAIChatModel('mistral-large-2407', provider=provider))
```

---

## 3. `DeepSeekProvider` — Reasoning Content and Thinking Parts

**Module:** `pydantic_ai.providers.deepseek`  
**Import:**
```python
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`DeepSeekProvider` is the provider for the [DeepSeek API](https://api.deepseek.com). Its model profile has three DeepSeek-specific quirks not present in any other provider: the `reasoning_content` wire field for thinking tokens, the `send_back_thinking_parts='field'` flag that tells PydanticAI to echo thinking tokens back on subsequent turns, and a per-model restriction on `tool_choice=required` for reasoning models.

### Constructor

```python
DeepSeekProvider(
    *,
    api_key: str | None = None,   # reads DEEPSEEK_API_KEY if None
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

If neither `api_key` nor `openai_client` is provided the constructor raises `UserError` immediately.

### Basic usage

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = DeepSeekProvider(api_key=os.environ['DEEPSEEK_API_KEY'])

# deepseek-chat is the V3 chat model (non-reasoning)
agent = Agent(OpenAIChatModel('deepseek-chat', provider=provider))

async def main():
    result = await agent.run('Explain beam search in two sentences.')
    print(result.output)

asyncio.run(main())
```

### R1 reasoning model — thinking tokens and `reasoning_content`

The `deepseek-reasoner` model (DeepSeek-R1) exposes its chain-of-thought in the `reasoning_content` field of each response delta, not inside `<think>` tags. PydanticAI surfaces these as `ThinkingPart` objects on each `ModelResponse`.

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.messages import ThinkingPart, TextPart

provider = DeepSeekProvider(api_key=os.environ['DEEPSEEK_API_KEY'])

# deepseek-reasoner → profile sets:
#   openai_chat_thinking_field='reasoning_content'
#   openai_chat_send_back_thinking_parts='field'
#   thinking_always_enabled=True
#   ignore_streamed_leading_whitespace=True
reasoning_agent = Agent(OpenAIChatModel('deepseek-reasoner', provider=provider))

async def main():
    result = await reasoning_agent.run('What is 17 × 23?')
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ThinkingPart):
                print(f'[Thinking] {part.content[:120]}...')
            elif isinstance(part, TextPart):
                print(f'[Answer]   {part.content}')

asyncio.run(main())
```

### V4 models — optional thinking via `reasoning_effort`

DeepSeek V4 models (`deepseek-v4-flash`, `deepseek-v4-pro`, etc.) support thinking but do not always enable it. Use `ModelSettings` to control effort:

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

provider = DeepSeekProvider(api_key=os.environ['DEEPSEEK_API_KEY'])

# deepseek-v4-flash: supports_thinking=True, thinking_always_enabled=False
agent = Agent(
    OpenAIChatModel('deepseek-v4-flash', provider=provider),
    model_settings=ModelSettings(thinking='high'),  # unified cross-provider field
)

async def main():
    result = await agent.run('Prove that √2 is irrational.')
    print(result.output)

asyncio.run(main())
```

### `tool_choice=required` restriction

Reasoning models (`deepseek-reasoner` and all `deepseek-v4-*` SKUs) do not support `tool_choice='required'`. PydanticAI's profile sets `openai_supports_tool_choice_required=False` for these models, so forced tool calls silently degrade to `tool_choice='auto'`:

```python
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

provider = DeepSeekProvider(api_key='...')

# tool_choice=required is silently dropped for reasoning models
agent = Agent(OpenAIChatModel('deepseek-reasoner', provider=provider))
# For deepseek-chat (non-reasoning) tool_choice=required IS supported
chat_agent = Agent(OpenAIChatModel('deepseek-chat', provider=provider))
```

---

## 4. `CerebrasProvider` — Ultra-Fast Inference with Reasoning Support

**Module:** `pydantic_ai.providers.cerebras`  
**Import:**
```python
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`CerebrasProvider` targets the [Cerebras Cloud API](https://inference.cerebras.ai/), known for extremely low latency on Llama and reasoning models. The provider adds a `X-Cerebras-3rd-Party-Integration: pydantic-ai` header on every request and disables six OpenAI settings that Cerebras does not support.

### Constructor

```python
CerebrasProvider(
    *,
    api_key: str | None = None,   # reads CEREBRAS_API_KEY if None
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

### Basic usage

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = CerebrasProvider(api_key=os.environ['CEREBRAS_API_KEY'])

# Llama 3.3 70B on Cerebras hardware — extremely low latency
agent = Agent(
    OpenAIChatModel('llama-3.3-70b', provider=provider),
    system_prompt='Answer concisely.',
)

async def main():
    result = await agent.run('What is the capital of Japan?')
    print(result.output)  # Tokyo

asyncio.run(main())
```

### Disabled model settings

Cerebras does not support: `frequency_penalty`, `logit_bias`, `presence_penalty`, `parallel_tool_calls`, `service_tier`, `openai_service_tier`. The profile marks these as `openai_unsupported_model_settings` so PydanticAI silently drops them rather than sending an unsupported field:

```python
from pydantic_ai import Agent
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

provider = CerebrasProvider(api_key='...')

# frequency_penalty is silently ignored — Cerebras doesn't support it
agent = Agent(
    OpenAIChatModel('llama-3.3-70b', provider=provider),
    model_settings=ModelSettings(frequency_penalty=0.5),  # dropped at wire level
)
```

### `zai` and `gpt-oss` reasoning models

Models prefixed `zai-` or `gpt-oss` on Cerebras support extended thinking. These map to `zai_model_profile` and `harmony_model_profile` respectively, and the provider sets `supports_thinking=True`:

```python
import os
from pydantic_ai import Agent
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

provider = CerebrasProvider(api_key=os.environ['CEREBRAS_API_KEY'])

# zai-r1-mini — fast reasoning model on Cerebras
reasoning_agent = Agent(
    OpenAIChatModel('zai-r1-mini', provider=provider),
    model_settings=ModelSettings(reasoning_effort='low'),  # supported by zai models
)
result = reasoning_agent.run_sync('How many prime numbers are below 100?')
print(result.output)
```

### FallbackModel: Cerebras → OpenAI

Because Cerebras is so fast for simple queries, a common pattern is to try Cerebras first and fall back to OpenAI for complex requests:

```python
import os
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.models.openai import OpenAIChatModel

cerebras_provider = CerebrasProvider(api_key=os.environ['CEREBRAS_API_KEY'])

agent = Agent(
    FallbackModel(
        OpenAIChatModel('llama-3.3-70b', provider=cerebras_provider),  # tries Cerebras first
        OpenAIChatModel('gpt-4.1'),  # falls back to OpenAI on errors
    ),
    system_prompt='You are a coding assistant.',
)
```

---

## 5. `GitHubProvider` — GitHub Models API

**Module:** `pydantic_ai.providers.github`  
**Import:**
```python
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.models.openai import OpenAIChatModel
```

`GitHubProvider` connects to [GitHub Models](https://docs.github.com/en/github-models) at `https://models.github.ai/inference`. GitHub Models provides free-tier access to dozens of models from multiple vendors. Model names follow the `provider/model:tag` convention (e.g. `meta/llama-3.3-70b-instruct:latest`).

### Constructor

```python
GitHubProvider(
    *,
    api_key: str | None = None,   # reads GITHUB_API_KEY if None (a GitHub PAT)
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
)
```

### Basic usage

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.models.openai import OpenAIChatModel

# GITHUB_API_KEY should be a GitHub Personal Access Token with models:read scope
provider = GitHubProvider(api_key=os.environ['GITHUB_API_KEY'])

agent = Agent(OpenAIChatModel('gpt-4o', provider=provider))

async def main():
    result = await agent.run('Explain the difference between asyncio and threading.')
    print(result.output)

asyncio.run(main())
```

### Provider-prefixed model names

GitHub Models uses `provider/model` or `provider/model:tag` names. The provider strips the `:tag` suffix before profile matching:

```python
import os
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

provider = GitHubProvider(api_key=os.environ['GITHUB_API_KEY'])

# Meta Llama — meta_model_profile applied
llama = Agent(OpenAIChatModel('meta/llama-3.3-70b-instruct:latest', provider=provider))

# xAI Grok — grok_model_profile applied
grok = Agent(OpenAIChatModel('xai/grok-2', provider=provider))

# Mistral — mistral_model_profile applied
mistral = Agent(OpenAIChatModel('mistral-ai/mistral-large-24.11', provider=provider))

# DeepSeek — deepseek_model_profile applied (reasoning_content field etc.)
deepseek = Agent(OpenAIChatModel('deepseek/deepseek-r1', provider=provider))

# OpenAI models (no prefix) — openai_model_profile
gpt41 = Agent(OpenAIChatModel('gpt-4.1', provider=provider))
```

Provider prefix → profile mapping:

| Prefix | Profile |
|--------|---------|
| `xai` | `grok_model_profile` |
| `meta` | `meta_model_profile` |
| `microsoft` | `openai_model_profile` |
| `mistral-ai` | `mistral_model_profile` |
| `cohere` | `cohere_model_profile` |
| `deepseek` | `deepseek_model_profile` |
| *(no prefix)* | `openai_model_profile` |

### Multi-model comparison in a single script

```python
import os, asyncio
from pydantic_ai.providers.github import GitHubProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai import Agent

provider = GitHubProvider(api_key=os.environ['GITHUB_API_KEY'])

models = [
    'gpt-4.1',
    'meta/llama-3.3-70b-instruct',
    'mistral-ai/mistral-large-24.11',
]
question = 'What is the time complexity of quicksort?'

async def main():
    for model_name in models:
        agent = Agent(OpenAIChatModel(model_name, provider=provider))
        result = await agent.run(question)
        print(f'--- {model_name} ---')
        print(result.output[:200])

asyncio.run(main())
```

---

## 6. `FireworksProvider` + `TogetherProvider` + `NebiusProvider` + `SambaNovaProvider` — OpenAI-Compatible GPU Providers

Four community GPU providers share the same `Provider[AsyncOpenAI]` pattern but differ in their endpoint URLs, model naming conventions, and supported model families. All are used by passing the provider to `OpenAIChatModel`.

### `FireworksProvider` — Fireworks AI

**Module:** `pydantic_ai.providers.fireworks`

Models on Fireworks use an `accounts/fireworks/models/<model-name>` path. The provider strips this prefix before matching profiles:

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.fireworks import FireworksProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = FireworksProvider(api_key=os.environ['FIREWORKS_API_KEY'])

# Full model path required — prefix stripped before profile lookup
agent = Agent(
    OpenAIChatModel('accounts/fireworks/models/llama-v3p3-70b-instruct', provider=provider)
)
# Qwen model — qwen_model_profile applied automatically
qwen_agent = Agent(
    OpenAIChatModel('accounts/fireworks/models/qwen2p5-72b-instruct', provider=provider)
)

async def main():
    result = await agent.run('Translate "hello world" into French.')
    print(result.output)

asyncio.run(main())
```

Supported prefixes after stripping `accounts/fireworks/models/`: `llama` → `meta_model_profile`, `qwen` → `qwen_model_profile`, `deepseek` → `deepseek_model_profile`, `mistral` → `mistral_model_profile`, `gemma` → `google_model_profile`.

**Endpoint:** `https://api.fireworks.ai/inference/v1`

### `TogetherProvider` — Together AI

**Module:** `pydantic_ai.providers.together`

Together AI uses `org/model` naming (e.g. `meta-llama/Llama-3-8b-hf`):

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.providers.together import TogetherProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = TogetherProvider(api_key=os.environ['TOGETHER_API_KEY'])

# meta-llama prefix → meta_model_profile
agent = Agent(
    OpenAIChatModel('meta-llama/Llama-3.3-70B-Instruct-Turbo', provider=provider)
)
# DeepSeek on Together
deepseek_agent = Agent(
    OpenAIChatModel('deepseek-ai/DeepSeek-R1-Distill-Llama-70B', provider=provider)
)

async def main():
    result = await agent.run('List 3 benefits of the Rust programming language.')
    print(result.output)

asyncio.run(main())
```

**Endpoint:** `https://api.together.xyz/v1`  
Supported prefixes: `deepseek-ai` → `deepseek_model_profile`, `google` → `google_model_profile`, `qwen` → `qwen_model_profile`, `meta-llama` → `meta_model_profile`, `mistralai` → `mistral_model_profile`.

### `NebiusProvider` — Nebius AI Studio

**Module:** `pydantic_ai.providers.nebius`

Nebius also uses `org/model` naming. Unlike Fireworks, models without a `/` separator fall back to `OpenAIModelProfile` immediately:

```python
import os
from pydantic_ai import Agent
from pydantic_ai.providers.nebius import NebiusProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = NebiusProvider(api_key=os.environ['NEBIUS_API_KEY'])

# Qwen on Nebius
agent = Agent(OpenAIChatModel('Qwen/Qwen3-30B-A3B', provider=provider))
# DeepSeek on Nebius — deepseek_model_profile (reasoning_content etc.)
deepseek_agent = Agent(OpenAIChatModel('deepseek-ai/DeepSeek-R1', provider=provider))
# OpenAI-compatible gpt-oss models (Harmony format)
gpt_oss_agent = Agent(OpenAIChatModel('openai/gpt-oss-mini', provider=provider))
```

**Endpoint:** `https://api.studio.nebius.com/v1`  
Supported prefixes: `meta-llama/` → `meta_model_profile`, `deepseek-ai/` → `deepseek_model_profile`, `qwen/` → `qwen_model_profile`, `google/` → `google_model_profile`, `openai/` → `harmony_model_profile`, `mistralai/` → `mistral_model_profile`, `moonshotai/` → `moonshotai_model_profile`.

### `SambaNovaProvider` — SambaNova Cloud

**Module:** `pydantic_ai.providers.sambanova`

SambaNova is unique in supporting a `SAMBANOVA_BASE_URL` env var for on-prem deployments, and it explicitly validates that an API key is present:

```python
import os
from pydantic_ai import Agent
from pydantic_ai.providers.sambanova import SambaNovaProvider
from pydantic_ai.models.openai import OpenAIChatModel

# Cloud endpoint (default: https://api.sambanova.ai/v1)
provider = SambaNovaProvider(api_key=os.environ['SAMBANOVA_API_KEY'])

# On-premise SambaNova deployment
on_prem_provider = SambaNovaProvider(
    api_key='my-on-prem-key',
    base_url='https://sambanova.mycompany.internal/v1',
)
# Also readable from env: SAMBANOVA_BASE_URL

agent = Agent(OpenAIChatModel('Meta-Llama-3.3-70B-Instruct', provider=provider))
qwen_agent = Agent(OpenAIChatModel('Qwen2.5-72B-Instruct', provider=provider))
```

**Endpoint:** `https://api.sambanova.ai/v1` (or `SAMBANOVA_BASE_URL`)  
Supported prefixes: `deepseek-` → `deepseek_model_profile`, `meta-llama-`, `llama-` → `meta_model_profile`, `qwen` → `qwen_model_profile`, `mistral` → `mistral_model_profile`.

### Provider quick-reference table

| Provider | Base URL | Env var | Naming convention | Unique feature |
|----------|----------|---------|-------------------|----------------|
| `FireworksProvider` | `api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | `accounts/fireworks/models/<name>` | Path prefix stripping |
| `TogetherProvider` | `api.together.xyz/v1` | `TOGETHER_API_KEY` | `org/model` | — |
| `NebiusProvider` | `api.studio.nebius.com/v1` | `NEBIUS_API_KEY` | `org/model` | No-slash → OpenAI fallback |
| `SambaNovaProvider` | `api.sambanova.ai/v1` | `SAMBANOVA_API_KEY` | Flat names | `base_url` / `SAMBANOVA_BASE_URL` for on-prem |

---

## 7. `GraphBuilder` — pydantic_graph Fluent Builder API

**Module:** `pydantic_graph.graph_builder`  
**Import:**
```python
from pydantic_graph import GraphBuilder
```

`GraphBuilder` is the **primary API for building graphs** in `pydantic_graph`. It replaces the deprecated `BaseNode`-based class hierarchy with a fluent, type-safe builder that compiles to an executable `Graph`. You define step functions, wire them with edges, and call `.build()` once.

<Aside type="caution">
The old `BaseNode` subclass API is deprecated as of `pydantic_graph` 1.107.0. Use `GraphBuilder` for all new graphs.
</Aside>

### Constructor

```python
GraphBuilder(
    *,
    name: str | None = None,
    state_type: type[StateT] = NoneType,
    deps_type: type[DepsT] = NoneType,
    input_type: type[GraphInputT] = NoneType,
    output_type: type[GraphOutputT] = NoneType,
    auto_instrument: bool = True,
)
```

### Minimal linear graph

```python
import asyncio
from dataclasses import dataclass
from pydantic_graph import GraphBuilder

@dataclass
class State:
    value: int

builder = GraphBuilder(state_type=State)

@builder.step
async def increment(ctx):
    ctx.state.value += 1
    return ctx.state.value

@builder.step
async def double(ctx):
    ctx.state.value *= 2
    return ctx.state.value

builder.add(
    builder.edge_from(builder.start_node)
        .to(increment)
)
builder.add(
    builder.edge_from(increment)
        .to(double)
)
builder.add(
    builder.edge_from(double)
        .to(builder.end_node)
)

graph = builder.build()

async def main():
    state = State(value=3)
    output = await graph.run(state=state, inputs=None)
    print(state.value)  # (3 + 1) * 2 = 8

asyncio.run(main())
```

### `@builder.step` decorator — two forms

```python
from pydantic_graph import GraphBuilder

builder = GraphBuilder()

# Form 1: bare decorator — node_id inferred from function name
@builder.step
async def my_step(ctx):
    return 'result'

# Form 2: factory with explicit node_id / label
@builder.step(node_id='compute', label='Compute Result')
async def compute_result(ctx):
    return 42

# Direct call (no decorator)
async def helper_step(ctx):
    return ctx.state

helper = builder.step(helper_step, node_id='helper')
```

### `@builder.stream` — async-generator steps

A "stream" step is an async generator that yields values one at a time. The builder wraps it into a `Step` that returns an `AsyncIterable`:

```python
import asyncio
from pydantic_graph import GraphBuilder

builder = GraphBuilder()

@builder.stream
async def token_stream(ctx):
    for word in ['The', 'quick', 'brown', 'fox']:
        yield word

builder.add(builder.edge_from(builder.start_node).to(token_stream))
builder.add(builder.edge_from(token_stream).to(builder.end_node))

graph = builder.build()

async def main():
    output = await graph.run(inputs=None)
    # output is an AsyncIterable of tokens
    async for token in output:
        print(token, end=' ')

asyncio.run(main())
```

### `add_edge` — simple directed edge

```python
builder.add_edge(increment, double)                    # auto-typed
builder.add_edge(increment, double, label='×2 path')  # with Mermaid label
```

### `add_mapping_edge` — fan-out map over iterable

```python
from pydantic_graph import GraphBuilder

builder = GraphBuilder()

@builder.step
async def produce_items(ctx):
    return [1, 2, 3, 4, 5]

@builder.step
async def process_item(ctx):
    return ctx.inputs * 2   # processes a single int

@builder.step
async def collect(ctx):
    return sum(ctx.inputs)  # receives list of results from join

# produce_items emits a list; each element goes to process_item in parallel
builder.add_mapping_edge(produce_items, process_item)
```

### `edge_from().broadcast()` — send same data to multiple steps in parallel

`broadcast()` takes a **single callback** that receives the builder and returns a list of edge paths — not multiple path arguments:

```python
builder.add(
    builder.edge_from(source_step)
        .broadcast(lambda b: [
            b.to(branch_a),
            b.to(branch_b),
        ])
)
```

For simple fan-out to a fixed set of steps you can also call `.to()` with multiple destinations, which creates an implicit broadcast fork:

```python
builder.add(builder.edge_from(source_step).to(branch_a, branch_b))
```

### `build(validate_graph_structure=True)` — compile to executable Graph

```python
graph = builder.build()
# Returns Graph[StateT, DepsT, GraphInputT, GraphOutputT]

# Run to completion
output = await graph.run(state=my_state, inputs=my_input)

# Iterate node-by-node
async with graph.iter(state=my_state, inputs=my_input) as graph_run:
    async for event in graph_run:
        print(event)  # EndMarker or list[GraphTask]
```

---

## 8. `Fork` + `Join` + `ReducerContext` — Parallel Fan-Out and Fan-In

**Module:** `pydantic_graph` (re-exported from `pydantic_graph.node` and `pydantic_graph.join`)  
**Import:**
```python
from pydantic_graph import Fork, Join
from pydantic_graph.join import ReducerContext
```

`Fork` and `Join` implement the parallel execution pattern in `pydantic_graph`. A `Fork` splits one execution path into multiple concurrent branches; a `Join` aggregates their outputs using a reducer function.

### `Fork` — split into parallel branches

```python
@dataclass
class Fork(Generic[InputT, OutputT]):
    id: ForkID
    is_map: bool            # True → map over Sequence[OutputT]; False → broadcast same data
    downstream_join_id: JoinID | None
```

You never construct `Fork` directly — `GraphBuilder` creates it internally when you call `.map()` or `.broadcast()` on a path builder. The `is_map=True` mode takes `Sequence[T]` input and fans out one element per branch; `is_map=False` duplicates the same value to all branches.

### `Join` — aggregate parallel results

```python
# Created via builder.join()
my_join = builder.join(
    reducer=lambda current, item: current + [item],  # append each result
    initial=[],
    node_id='collect_results',
)
```

The reducer has two forms:
- **Plain:** `(current: OutputT, item: InputT) -> OutputT` — no context
- **Context-aware:** `(ctx: ReducerContext[StateT, DepsT], current: OutputT, item: InputT) -> OutputT`

### `ReducerContext` — context passed to reducers

```python
@dataclass
class ReducerContext(Generic[StateT, DepsT]):
    state: StateT    # the graph state (mutable)
    deps: DepsT      # the graph dependencies

    def cancel_sibling_tasks(self) -> None:
        """Cancel all other branches in the same fork — early stopping."""
```

`cancel_sibling_tasks()` is the key feature: call it in your reducer to stop remaining parallel branches once a result satisfies your condition (e.g. first-match wins).

### Complete fan-out / fan-in example

```python
import asyncio
from dataclasses import dataclass, field
from pydantic_graph import GraphBuilder
from pydantic_graph.join import ReducerContext

@dataclass
class State:
    results: list[int] = field(default_factory=list)

builder = GraphBuilder(state_type=State, input_type=list[int], output_type=list[int])

@builder.step
async def produce(ctx):
    # Returns a list — will be mapped over by the fork
    return ctx.inputs   # e.g. [10, 20, 30]

@builder.step
async def square(ctx):
    return ctx.inputs ** 2  # process one int at a time

def collect_reducer(
    ctx: ReducerContext[State, None],
    current: list[int],
    item: int,
) -> list[int]:
    ctx.state.results.append(item)
    return current + [item]

collect_join = builder.join(collect_reducer, initial=[])

@builder.step
async def output_step(ctx):
    return ctx.inputs   # pass through the joined list

builder.add_mapping_edge(produce, square)
builder.add_edge(square, collect_join)      # Join is a MiddleNode — wire it directly
builder.add_edge(collect_join, output_step)
builder.add_edge(builder.start_node, produce)
builder.add_edge(output_step, builder.end_node)

graph = builder.build()

async def main():
    state = State()
    result = await graph.run(state=state, inputs=[3, 4, 5])
    print(result)       # [9, 16, 25] (order may vary)
    print(state.results)

asyncio.run(main())
```

### `cancel_sibling_tasks()` — first-match early stopping

```python
def first_success_reducer(
    ctx: ReducerContext,
    current: str | None,
    item: str | None,
) -> str | None:
    if item is not None and current is None:
        ctx.cancel_sibling_tasks()  # stop other branches immediately
        return item
    return current

winner_join = builder.join(first_success_reducer, initial=None)
```

### `preferred_parent_fork` — join topology when nested

When a join is downstream of multiple forks (nested parallel execution), `preferred_parent_fork` controls which fork the join waits for:
- `'farthest'` (default) — waits for the outermost (earliest) enclosing fork
- `'closest'` — waits only for the innermost enclosing fork

```python
inner_join = builder.join(reducer, initial=0, preferred_parent_fork='closest')
outer_join = builder.join(reducer, initial=0, preferred_parent_fork='farthest')
```

---

## 9. `Decision` + `DecisionBranch` + `Edge` + `TypeExpression` — Conditional Routing

**Module:** `pydantic_graph` (re-exported from `pydantic_graph.decision`)  
**Import:**
```python
from pydantic_graph import Decision, Edge, TypeExpression
from typing import Literal
```

`Decision` is the conditional branching node in `pydantic_graph`. It inspects the **type** of the output (using `isinstance` or `Literal` matching) to route execution to different downstream steps. The `source` argument to `builder.match()` must be a type or `Literal` — not a raw value. `Edge` annotates graph edges with labels for Mermaid diagram generation. `TypeExpression` is a workaround for Python type-checker limitations when using complex union types in generic parameters.

<Aside type="note" title="match() requires types, not values">
`builder.match(Literal['urgent'])` works — `Literal` is a type. `builder.match('urgent')` does **not** work — a raw string is a value, not a type, and causes a `TypeError` at runtime when the runner tries `isinstance(value, 'urgent')`. Always use `Literal[...]` to match specific string/int values.
</Aside>

### `Decision` + `DecisionBranch`

Decisions are built through `GraphBuilder.decision()` and the `.match()` / `.match_node()` helpers:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Literal
from pydantic_graph import GraphBuilder

@dataclass
class State:
    ticket: str = field(default='')  # store original text for downstream steps

builder = GraphBuilder(state_type=State, input_type=str, output_type=str)

@builder.step
async def classify(ctx) -> Literal['urgent', 'billing', 'general']:
    ctx.state.ticket = ctx.inputs  # save original text before routing
    if 'urgent' in ctx.inputs.lower():
        return 'urgent'
    elif 'billing' in ctx.inputs.lower():
        return 'billing'
    else:
        return 'general'

# NOTE: ctx.inputs here is the *category string* ('urgent'/'billing'/'general'),
# not the original ticket text. Use ctx.state.ticket to access the original.
@builder.step
async def handle_urgent(ctx):
    return f'URGENT: {ctx.state.ticket}'

@builder.step
async def handle_billing(ctx):
    return f'BILLING: {ctx.state.ticket}'

@builder.step
async def handle_general(ctx):
    return f'GENERAL: {ctx.state.ticket}'

# Use Literal types — not raw strings — for value-based routing
ticket_router = (
    builder.decision(note='Route ticket by category')
    .branch(builder.match(Literal['urgent']).to(handle_urgent))
    .branch(builder.match(Literal['billing']).to(handle_billing))
    .branch(builder.match(Literal['general']).to(handle_general))
)

builder.add(builder.edge_from(builder.start_node).to(classify))
builder.add(builder.edge_from(classify).to(ticket_router))
builder.add(builder.edge_from(handle_urgent).to(builder.end_node))
builder.add(builder.edge_from(handle_billing).to(builder.end_node))
builder.add(builder.edge_from(handle_general).to(builder.end_node))

graph = builder.build()

async def main():
    state = State()
    out = await graph.run(state=state, inputs='This is urgent!')
    print(out)  # URGENT: This is urgent!

asyncio.run(main())
```

### Custom `matches` predicate

When you need logic beyond `isinstance`/`Literal` dispatch, supply a `matches` callable. The `source` is still a type (for exhaustiveness checking), but the runtime decision uses your predicate:

```python
import re
from typing import Literal

# Route based on a regex pattern
spam_decision = (
    builder.decision(note='Spam filter')
    .branch(
        builder.match(
            str,
            matches=lambda s: bool(re.search(r'(buy now|click here|free offer)', s, re.I)),
        ).to(handle_spam)
    )
    .branch(builder.match(str).to(handle_legitimate))
)
```

### Union type routing

Decisions shine when a step returns a union type — each branch handles one variant:

```python
from dataclasses import dataclass
from pydantic_graph import GraphBuilder

@dataclass
class SuccessResult:
    data: str

@dataclass
class ErrorResult:
    message: str

builder = GraphBuilder()

@builder.step
async def fetch_data(ctx) -> SuccessResult | ErrorResult:
    # Simulated: might return success or error
    return SuccessResult(data='some data')

@builder.step
async def process_success(ctx):
    return f'Processed: {ctx.inputs.data}'

@builder.step
async def log_error(ctx):
    return f'Error logged: {ctx.inputs.message}'

# SuccessResult and ErrorResult are concrete types — isinstance dispatch works correctly
result_router = (
    builder.decision()
    .branch(builder.match(SuccessResult).to(process_success))
    .branch(builder.match(ErrorResult).to(log_error))
)

builder.add(builder.edge_from(builder.start_node).to(fetch_data))
builder.add(builder.edge_from(fetch_data).to(result_router))
builder.add(builder.edge_from(process_success).to(builder.end_node))
builder.add(builder.edge_from(log_error).to(builder.end_node))
```

### `Edge` — label graph edges for Mermaid diagrams

`Edge` is a frozen dataclass with a single field. In **legacy `BaseNode`-based graphs**, `Annotated[T, Edge(label='...')]` return-type annotations are parsed to label edges. In **builder-based graphs**, labels are added via `.label(text)` on the `EdgePathBuilder` chain — `Edge` annotations on step return types are not parsed by the builder runner:

```python
from pydantic_graph import GraphBuilder

builder = GraphBuilder(input_type=str, output_type=str)

@builder.step
async def evaluate(ctx) -> str | None:
    return ctx.inputs if ctx.inputs else None

@builder.step
async def skip_step(ctx) -> str:
    return 'skipped'

@builder.step
async def result_step(ctx) -> str:
    return f'processed: {ctx.inputs}'

# Use .label() on edge_from() chains to annotate Mermaid diagrams
builder.add(builder.edge_from(builder.start_node).to(evaluate))
builder.add(builder.edge_from(evaluate).label('result').to(result_step))
builder.add(builder.edge_from(evaluate).label('skip').to(skip_step))
builder.add(builder.edge_from(result_step).to(builder.end_node))
builder.add(builder.edge_from(skip_step).to(builder.end_node))

graph = builder.build()
print(graph.render())  # Mermaid diagram with 'result' and 'skip' edge labels
```

### `TypeExpression` — complex union types in generic positions

Python's type checker sometimes rejects complex type expressions (like `Union[str, int]`) in positions that expect `type[T]`. `TypeExpression[T]` is a workaround:

```python
from typing import Union, Literal, Any
from pydantic_graph import GraphBuilder, TypeExpression

# Without TypeExpression — may cause type checker error:
# builder = GraphBuilder(output_type=Union[str, int])  # ← type error in some checkers

# With TypeExpression — always safe:
builder = GraphBuilder(output_type=TypeExpression[Union[str, int]])

# Similarly for Literal types, Any, or complex generics:
builder2 = GraphBuilder(input_type=TypeExpression[Literal['start', 'resume']])
builder3 = GraphBuilder(state_type=TypeExpression[Any])
```

---

## 10. `Step` + `StepContext` + `StepNode` — Step Execution Primitives

**Module:** `pydantic_graph` (re-exported from `pydantic_graph.step`)  
**Import:**
```python
from pydantic_graph.step import Step, StepContext, StepNode
```

These three classes are the lowest-level execution primitives that `GraphBuilder` creates when you use `@builder.step`. Understanding them is useful for introspection, dynamic graph modification, and bridging the new builder API with legacy `BaseNode` code.

### `StepContext` — context object passed to every step function

```python
@dataclass
class StepContext(Generic[StateT, DepsT, InputT]):
    state: StateT   # the shared mutable graph state
    deps: DepsT     # graph-level dependencies (read-only by convention)
    inputs: InputT  # the data passed to this step from the upstream edge
```

Every step function receives a `StepContext` as its only argument:

```python
import asyncio
from dataclasses import dataclass
from pydantic_graph import GraphBuilder

@dataclass
class AppState:
    counter: int = 0

@dataclass
class AppDeps:
    multiplier: int

builder = GraphBuilder(state_type=AppState, deps_type=AppDeps, input_type=int, output_type=int)

@builder.step
async def compute(ctx: builder.Source[int]) -> int:
    # Access all three context fields
    value = ctx.inputs                  # the input data (int)
    ctx.state.counter += 1             # mutate shared state
    return value * ctx.deps.multiplier  # use deps

builder.add_edge(builder.start_node, compute)
builder.add_edge(compute, builder.end_node)

graph = builder.build()

async def main():
    state = AppState()
    deps = AppDeps(multiplier=5)
    result = await graph.run(state=state, deps=deps, inputs=7)
    print(result)          # 35
    print(state.counter)   # 1

asyncio.run(main())
```

### `Step` — a wrapped step function with metadata

`Step` is the node object that holds a step function along with its `id` and optional `label`. You obtain it from the `@builder.step` decorator and can use it to inspect or rewire the graph:

```python
from pydantic_graph import GraphBuilder

builder = GraphBuilder()

@builder.step
async def my_step(ctx):
    return ctx.inputs + 1

# Step metadata
print(my_step.id)       # NodeID('my_step')
print(my_step.label)    # None (no label provided)
print(my_step.call)     # the underlying async function

# Step with explicit metadata
@builder.step(node_id='transform', label='Transform Data')
async def transform(ctx):
    return str(ctx.inputs)

print(transform.id)     # NodeID('transform')
print(transform.label)  # 'Transform Data'
```

### `Step.as_node()` — produce a `StepNode` for legacy bridge usage

`as_node(inputs=None)` binds an optional `InputT` value to the step and returns a `StepNode` — a `BaseNode` subclass. Its primary purpose is to produce a v1-compatible node so that a builder-defined step can be passed to a **legacy `BaseNode`-based graph runner**. For the builder-based `Graph`, pass inputs via the keyword argument `graph.run(inputs=...)` — the `Graph.run()` signature is keyword-only and does not accept a positional node argument.

```python
import asyncio
from pydantic_graph import GraphBuilder

builder = GraphBuilder(input_type=int, output_type=int)

@builder.step
async def process(ctx):
    return ctx.inputs * 2

builder.add_edge(builder.start_node, process)
builder.add_edge(process, builder.end_node)
graph = builder.build()

# Graph.run() is keyword-only; pass inputs= directly
result = asyncio.run(graph.run(inputs=21))
print(result)  # 42
```

<Aside type="caution">
`Step.as_node()` is **not** a "dynamic jump" mechanism inside a builder step body. Returning a `StepNode` from within `@builder.step` is treated as regular data, not as a routing instruction. To branch dynamically, wire edges through a `builder.decision()` node instead.
</Aside>

### `StepNode` — bridge from BaseNode to Step

`StepNode` is a `BaseNode` subclass that wraps a `Step` with bound inputs. It is produced by `Step.as_node()` and is not meant to be run directly — the v2 graph runner detects `StepNode` and executes the wrapped `Step` via `StepContext`:

```python
from pydantic_graph.step import StepNode

# StepNode is produced automatically — you rarely construct it manually
node = my_step.as_node(inputs=42)
print(node.step)    # the Step object
print(node.inputs)  # 42
```

### Streaming step with `@builder.stream`

A streaming step is a step function that returns an `AsyncIterable` of values. The `@builder.stream` decorator wraps an async generator into a standard `Step`:

```python
import asyncio
from pydantic_graph import GraphBuilder

builder = GraphBuilder(input_type=str, output_type=list[str])

@builder.stream
async def tokenize(ctx):
    for word in ctx.inputs.split():
        yield word

@builder.step
async def collect(ctx):
    tokens = []
    async for token in ctx.inputs:   # ctx.inputs is the AsyncIterable from tokenize
        tokens.append(token)
    return tokens

builder.add_edge(builder.start_node, tokenize)
builder.add_edge(tokenize, collect)
builder.add_edge(collect, builder.end_node)

graph = builder.build()

async def main():
    result = await graph.run(inputs='the quick brown fox')
    print(result)  # ['the', 'quick', 'brown', 'fox']

asyncio.run(main())
```

### Step type annotation for full IDE support

`GraphBuilder` exposes `Source[T]` and `Destination[T]` type aliases to give step functions proper type-safe signatures:

```python
from pydantic_graph import GraphBuilder
from dataclasses import dataclass

@dataclass
class MyState:
    total: float = 0.0

builder = GraphBuilder(state_type=MyState, input_type=float, output_type=float)

# Fully typed step — IDE knows ctx.state is MyState, ctx.inputs is float
@builder.step
async def accumulate(ctx: builder.Source[float]) -> float:
    ctx.state.total += ctx.inputs
    return ctx.state.total
```

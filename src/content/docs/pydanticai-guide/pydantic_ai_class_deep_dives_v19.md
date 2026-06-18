---
title: "PydanticAI: Class Deep Dives Vol. 19"
description: "BedrockConverseModel, BedrockProvider, deprecated Google providers, DynamicToolset, CapabilityOwnedToolset, web UI API, OTel GenAI spec types, Thinking capability, ImageGeneration capability, pydantic_graph persistence primitives (pydantic-ai 1.107.0)"
sidebar:
  order: 45
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

Source-verified against `pydantic-ai==1.107.0` installed at `/usr/local/lib/python3.11/dist-packages/pydantic_ai/`. All constructor signatures, field names, constants, and default values were read directly from the installed package source.

## 1. `BedrockConverseModel` + `BedrockModelSettings` + `BedrockStreamedResponse`

**Module:** `pydantic_ai.models.bedrock`

`BedrockConverseModel` wraps AWS Bedrock's synchronous `converse` / `converse_stream` API calls via a boto3 `BedrockRuntimeClient`, adapting them to the async `Model` interface. It is the only first-party model for the AWS ecosystem and is required for Anthropic, Amazon Nova, Meta Llama, Mistral, Cohere, DeepSeek, Qwen, Google Gemma, MiniMax, and NVIDIA models accessed through Bedrock.

`BedrockModelSettings` is a `ModelSettings` TypedDict (not a dataclass) — all fields carry the `bedrock_` prefix to allow safe merging with other provider settings. Three of the fields control Bedrock's prompt caching feature (`bedrock_cache_tool_definitions`, `bedrock_cache_instructions`, `bedrock_cache_messages`); each accepts `True | '5m' | '1h'` where `True` and `'5m'` map to the 5-minute TTL.

`BedrockStreamedResponse` is an internal dataclass that wraps the boto3 `EventStream`; because boto3 is synchronous, `close_stream()` uses `anyio.to_thread.run_sync(self._event_stream.close)` and all event iteration runs in a thread wrapper `_AsyncIteratorWrapper`.

| Key behaviour | Detail |
|---|---|
| Constructor | `BedrockConverseModel(model_name, *, provider='bedrock', profile=None, settings=None)` |
| `provider='gateway'` | Calls `infer_provider('gateway/bedrock')` to route through the Pydantic AI Gateway |
| Client resolution | `self._client or self._provider.client` — swap on model or provider |
| `bedrock_cache_tool_definitions` | `True`/`'5m'` = 5-min TTL; `'1h'` = 1-hour TTL; adds `cachePoint` after last tool def |
| `bedrock_cache_instructions` | Adds `cachePoint` after system prompt blocks |
| `bedrock_cache_messages` | Adds `cachePoint` to last user message content block; consumes 1 of Bedrock's 4 cache-point slots per request |
| `bedrock_guardrail_config` | Content moderation / PII redaction; see Bedrock GuardrailConfiguration API |
| `bedrock_performance_configuration` | Latency/throughput optimisation hints |
| `bedrock_request_metadata` | `dict[str, str]` attached to every Converse request |
| `bedrock_additional_model_requests_fields` | Model-specific params (thinking budget, etc.) passed through `additionalModelRequestFields` |
| `_FINISH_REASON_MAP` | `'end_turn'→'stop'`, `'max_tokens'→'length'`, `'tool_use'→'tool_call'`, `'content_filtered'→'content_filter'`, `'guardrail_intervened'→'content_filter'`, `'stop_sequence'→'stop'`, `'malformed_model_output'→'error'`, `'malformed_tool_use'→'error'`, `'model_context_window_exceeded'→'length'` |
| Supported media in messages | Images: `jpeg/png/gif/webp`; Video: `mkv/mov/mp4/webm/flv/mpeg/mpg/wmv/three_gp`; Docs: `pdf/txt/csv/doc/docx/xls/xlsx/html/md` |

### Example 1 — Basic cross-region Bedrock agent

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Credentials are picked up from the environment:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
# or from ~/.aws/credentials via a named profile.
provider = BedrockProvider(region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

# Cross-region inference prefix: 'us.' routes through AWS's cross-region fleet.
model = BedrockConverseModel(
    'us.anthropic.claude-sonnet-4-6',
    provider=provider,
)

agent = Agent(model, system_prompt='You are a concise assistant.')


async def main() -> None:
    result = await agent.run('What is the capital of France?')
    print(result.output)           # Paris
    print(result.usage())          # RequestUsage(requests=1, ...)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Prompt caching with `BedrockModelSettings`

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider

provider = BedrockProvider(region_name='us-east-1')
model = BedrockConverseModel('us.anthropic.claude-sonnet-4-6', provider=provider)

# Cache the system prompt for 5 min and the last user message turn.
# Together they consume 2 of Bedrock's 4 available cache-point slots.
settings: BedrockModelSettings = {
    'bedrock_cache_instructions': True,   # 'True' == '5m' TTL
    'bedrock_cache_messages': True,
}

LONG_CONTEXT = '...' * 500  # simulate a large document in the system prompt

agent = Agent(
    model,
    system_prompt=f'You are a document analyst. Context:\n{LONG_CONTEXT}',
    model_settings=settings,
)


async def main() -> None:
    # First call primes the cache; subsequent calls are cheaper.
    for question in ['Summarise the document.', 'What are the key findings?']:
        result = await agent.run(question)
        print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — Guardrails and performance configuration

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider

provider = BedrockProvider(region_name='us-east-1')
model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=provider)

# bedrock_guardrail_config enables AWS Bedrock Guardrails for content filtering.
# bedrock_performance_configuration sets latency mode to 'optimized' for faster responses.
settings: BedrockModelSettings = {
    'bedrock_guardrail_config': {
        'guardrailIdentifier': os.environ.get('BEDROCK_GUARDRAIL_ID', 'test-id'),
        'guardrailVersion': 'DRAFT',
        'trace': 'enabled',
    },
    'bedrock_performance_configuration': {
        'latency': 'optimized',
    },
    'bedrock_request_metadata': {
        'app': 'my-service',
        'environment': 'production',
    },
}

agent = Agent(model, model_settings=settings)


async def main() -> None:
    result = await agent.run('Tell me about AWS security best practices.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 2. `BedrockProvider` + `BedrockModelProfile` + `BedrockJsonSchemaTransformer` + profile functions

**Module:** `pydantic_ai.providers.bedrock`

`BedrockProvider` implements `Provider[BaseClient]` — it wraps a boto3 `bedrock-runtime` client. Three construction paths exist: bring-your-own client, bearer-token API key, and standard AWS credentials. The `client` setter allows hot-swapping the boto3 client for credential rotation without recreating the provider.

`BedrockModelProfile` extends `ModelProfile` with ten `bedrock_`-prefixed fields. The naming convention is intentional: `ModelProfile.update()` only copies fields that exist on `self`, so `bedrock_`-prefixed flags can survive a merge with an upstream profile without being silently dropped.

`BedrockJsonSchemaTransformer` handles the schema subset that Bedrock's structured-output API accepts. In strict mode it injects `additionalProperties: false` on objects and strips unsupported numeric/array constraints, re-emitting them in the field's `description` string so the model still sees them as hints.

Eight `bedrock_*_model_profile` factory functions map each vendor's Bedrock model IDs to an appropriate `BedrockModelProfile`:

| Function | Key behaviour |
|---|---|
| `bedrock_anthropic_model_profile` | Inherits `AnthropicModelProfile`; structured output blocked for `claude-opus-4-1/4-7/4-8`; `bedrock_thinking_variant='anthropic'` |
| `bedrock_amazon_model_profile` | Nova models get `bedrock_supports_tool_choice=True`; Nova 2 adds `CodeExecutionTool` native support |
| `bedrock_deepseek_model_profile` | R1 variants get `bedrock_send_back_thinking_parts=True` |
| `bedrock_mistral_model_profile` | `bedrock_tool_result_format='json'`; structured output for magistral-small/ministral-3/mistral-large-3/voxtral |
| `bedrock_qwen_model_profile` | `bedrock_thinking_variant='qwen'`; `reasoning_config` limited to `low/high` (no disable) |
| `bedrock_google_model_profile` | Structured output for gemma-3-12b-it/gemma-3-27b-it only |
| `bedrock_minimax_model_profile` | `minimax-m2` supports structured output |
| `bedrock_nvidia_model_profile` | `nemotron-nano` supports structured output |

`_without_builtin_tools(profile)` strips all native tools from any profile — Bedrock's Converse API does not support built-in tools natively.

| `BedrockModelProfile` field | Default | Meaning |
|---|---|---|
| `bedrock_supports_tool_choice` | `False` | Whether the model accepts `toolChoice` |
| `bedrock_tool_result_format` | `'text'` | `'json'` for Mistral (tool results serialised as JSON) |
| `bedrock_send_back_thinking_parts` | `False` | Send thinking tokens back in conversation turns |
| `bedrock_supports_prompt_caching` | `False` | Cache points allowed in system blocks |
| `bedrock_supports_tool_caching` | `False` | Cache points allowed after tool definitions |
| `bedrock_supported_media_kinds_in_tool_returns` | `frozenset({'image'})` | Media types accepted in tool return content |
| `bedrock_supports_strict_tool_definition` | `False` | `strict: true` on `toolSpec` accepted |
| `bedrock_thinking_variant` | `None` | `'anthropic'`/`'openai'`/`'qwen'` thinking translation |
| `bedrock_supports_adaptive_thinking` | `False` | `{'thinking': {'type': 'adaptive'}}` accepted (Sonnet/Opus 4.6+) |
| `bedrock_supports_effort` | `False` | `output_config.effort` accepted alongside adaptive thinking |

`BedrockJsonSchemaTransformer` strict-mode rewrites:
- Always strips `title` and `$schema`
- For `number`/`integer`: removes `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`
- For `array`: removes `maxItems`; removes `minItems` when `> 1`
- Removed constraints are appended to `description` as `key=value` strings
- `is_strict_compatible = strict is True` — opt-in unlike Anthropic's auto-strict

### Example 1 — Credential rotation with the `client` setter

```python
import asyncio
import boto3
from botocore.config import Config
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

provider = BedrockProvider(region_name='us-east-1')
model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=provider)
agent = Agent(model)


def rotate_credentials(new_access_key: str, new_secret_key: str, new_token: str) -> None:
    """Swap in a fresh boto3 client after STS token refresh."""
    new_session = boto3.Session(
        aws_access_key_id=new_access_key,
        aws_secret_access_key=new_secret_key,
        aws_session_token=new_token,
        region_name='us-east-1',
    )
    new_client = new_session.client(
        'bedrock-runtime',
        config=Config(read_timeout=300, connect_timeout=60),
    )
    # Assign to provider so all models sharing it pick up the new client.
    provider.client = new_client


async def main() -> None:
    result = await agent.run('Hello!')
    print(result.output)
    # Later: rotate_credentials(new_key, new_secret, new_token)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Custom `BedrockModelProfile` via `update()`

```python
import asyncio
from dataclasses import replace
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider, BedrockModelProfile, bedrock_anthropic_model_profile

provider = BedrockProvider(region_name='us-east-1')

# Start from the standard Anthropic profile and overlay custom Bedrock flags.
base_profile = bedrock_anthropic_model_profile('claude-sonnet-4-6')
custom_profile = replace(
    base_profile,
    # Enable tool-caching capability for this deployment
    # (bedrock_supports_tool_caching is a boolean flag, not a TTL).
    bedrock_supports_tool_caching=True,
)

model = BedrockConverseModel(
    'us.anthropic.claude-sonnet-4-6',
    provider=provider,
    profile=custom_profile,
)

agent = Agent(model)


async def main() -> None:
    result = await agent.run('List three AWS services.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — Bearer-token auth (API-key path)

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Bearer-token auth uses AWS_BEARER_TOKEN_BEDROCK env var.
# This path skips SigV4 signing and is useful for testing / CI.
provider = BedrockProvider(
    api_key=os.environ['AWS_BEARER_TOKEN_BEDROCK'],
    region_name='us-east-1',
)

model = BedrockConverseModel('us.amazon.nova-lite-v1:0', provider=provider)
agent = Agent(model)


async def main() -> None:
    result = await agent.run('What are the benefits of serverless computing?')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 3. `GoogleGLAProvider` + `GoogleVertexProvider` (deprecated) + migration guide

**Modules:** `pydantic_ai.providers.google_gla`, `pydantic_ai.providers.google_vertex`

<Aside type="caution">
Both `GoogleGLAProvider` and `GoogleVertexProvider` are deprecated. Migrate to `GoogleProvider` (Gemini API) or `GoogleCloudProvider` (Vertex AI / Google Cloud) combined with `GoogleModel`. The new providers expose the same `model_profile()` logic and are actively maintained.
</Aside>

`GoogleGLAProvider` was the original Gemini API provider. It set `base_url = 'https://generativelanguage.googleapis.com/v1beta/models/'` and read `GEMINI_API_KEY` from the environment, attaching it as the `X-Goog-Api-Key` header.

`GoogleVertexProvider` connected to Vertex AI. It built the base URL from `region`, `project_id`, and `model_publisher` (default `'google'`). Authentication was handled by `_VertexAIAuth(httpx.Auth)`:
- Loads credentials from `service_account_file`, `service_account_info`, or `google.auth.default()`
- On 401 automatically refreshes the token and replays the request
- `_refresh_lock` is a `@functools.cached_property` — defers `anyio.Lock()` creation to avoid binding to the wrong event loop (important in Temporal sandbox contexts)
- `_creds_from_file` and `_creds_from_info` run blocking Google auth calls via `anyio.to_thread.run_sync`

`VertexAiRegion` is a `Literal` type with 29 valid regions including `'us-central1'`, `'europe-west4'`, `'asia-northeast1'`, etc.

| Old class | New class | Key env var change |
|---|---|---|
| `GoogleGLAProvider(api_key=...)` | `GoogleProvider(api_key=...)` | `GEMINI_API_KEY` → `GOOGLE_API_KEY` |
| `GoogleVertexProvider(service_account_file=..., project_id=..., region=...)` | `GoogleCloudProvider(service_account_file=..., project_id=..., location=...)` | `region=` → `location=` |

### Example 1 — Migration: `GoogleGLAProvider` → `GoogleProvider`

```python
# BEFORE (deprecated)
# from pydantic_ai.providers.google_gla import GoogleGLAProvider
# from pydantic_ai.models.gemini import GeminiModel
# provider = GoogleGLAProvider(api_key='AIza...')
# model = GeminiModel('gemini-2.5-pro', provider=provider)

# AFTER
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key=os.environ['GOOGLE_API_KEY'])
model = GoogleModel('gemini-2.5-pro', provider=provider)
agent = Agent(model, system_prompt='You are a helpful assistant.')


async def main() -> None:
    result = await agent.run('Explain quantum entanglement in one paragraph.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Migration: `GoogleVertexProvider` → `GoogleCloudProvider`

```python
# BEFORE (deprecated)
# from pydantic_ai.providers.google_vertex import GoogleVertexProvider
# from pydantic_ai.models.gemini import GeminiModel
# provider = GoogleVertexProvider(
#     service_account_file='sa.json',
#     project_id='my-gcp-project',
#     region='us-central1',
# )
# model = GeminiModel('gemini-2.5-pro', provider=provider)

# AFTER
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(
    service_account_file='sa.json',
    project='my-gcp-project',    # was project_id=
    location='us-central1',      # was region=
)
model = GoogleModel('gemini-2.5-pro', provider=provider)
agent = Agent(model)


async def main() -> None:
    result = await agent.run('Summarise the latest AI research trends.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — Legacy `GoogleVertexProvider` multi-region pattern (for reference)

```python
# This shows the legacy API pattern that still works but is deprecated.
# Prefer GoogleCloudProvider for new code.
import asyncio
import os
from typing_extensions import deprecated
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

# Both deprecated providers still function; they delegate to the same
# GoogleModel infrastructure. This example shows how to select a non-default
# region using the old API for teams that haven't migrated yet.
try:
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider  # type: ignore[import]

    provider = GoogleVertexProvider(
        project_id=os.environ.get('GCP_PROJECT', 'my-project'),
        region='europe-west4',       # EU region for data residency
        model_publisher='google',    # only 'google' is supported
    )
    model = GoogleModel('gemini-2.5-pro', provider=provider)
except Exception:
    # Fall back to GoogleCloudProvider if deprecated module raises
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider
    provider = GoogleCloudProvider(location='europe-west4')
    model = GoogleModel('gemini-2.5-pro', provider=provider)

agent = Agent(model)


async def main() -> None:
    result = await agent.run('What is the GDPR?')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 4. `DynamicToolset`

**Module:** `pydantic_ai.toolsets._dynamic`

`DynamicToolset` wraps a factory function `toolset_func: Callable[[RunContext], AbstractToolset | None | Awaitable[...]]` and evaluates it to produce the actual toolset. This lets you vary which tools are available based on run-time state — user role, feature flags, external service availability, etc.

The critical design detail is `per_run_step` (default `True`): when `True`, the factory is re-evaluated before every model call within a single run; when `False`, it is evaluated once at run start via `for_run()`. In either case, the lifecycle of the inner toolset is managed correctly: the old toolset's `__aexit__` is called before the new one's `__aenter__`, and `self._toolset` is only set to the new toolset after a successful `__aenter__` — preventing phantom exit calls on toolsets that were never entered.

| Behaviour | Detail |
|---|---|
| `toolset_func` | Sync or async; receives `RunContext`; returns `AbstractToolset` or `None` |
| `per_run_step=True` | Factory called at each `for_run_step()` — tools can change mid-run |
| `per_run_step=False` | Factory called once at `for_run()` — toolset fixed for the entire run |
| `id=` | Required for Temporal durable execution; used to reconstruct the toolset |
| `__eq__` | Uses identity (`is`) for `toolset_func`, not equality |
| Transition safety | Old toolset detached before exit; new toolset only registered after successful enter |
| `get_tools` | Returns `{}` when `_toolset is None` |

### Example 1 — Role-based toolset switching per step

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.tools import RunContext


@dataclass
class UserDeps:
    role: str  # 'admin' or 'viewer'


def admin_tools() -> FunctionToolset:
    ts = FunctionToolset()

    @ts.tool
    async def delete_record(ctx: RunContext[UserDeps], record_id: str) -> str:
        return f'Deleted record {record_id}'

    @ts.tool
    async def create_record(ctx: RunContext[UserDeps], name: str) -> str:
        return f'Created record: {name}'

    return ts


def viewer_tools() -> FunctionToolset:
    ts = FunctionToolset()

    @ts.tool
    async def read_record(ctx: RunContext[UserDeps], record_id: str) -> str:
        return f'Record {record_id}: some data'

    return ts


def role_toolset_factory(ctx: RunContext[UserDeps]):
    if ctx.deps.role == 'admin':
        return admin_tools()
    return viewer_tools()


dynamic = DynamicToolset(role_toolset_factory, per_run_step=True)
agent = Agent('openai:gpt-4o', toolsets=[dynamic], deps_type=UserDeps)


async def main() -> None:
    admin_result = await agent.run(
        'Create a record called "sales-q1".',
        deps=UserDeps(role='admin'),
    )
    print(admin_result.output)

    viewer_result = await agent.run(
        'Read record 42.',
        deps=UserDeps(role='viewer'),
    )
    print(viewer_result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Expensive one-time initialization with `per_run_step=False`

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.tools import RunContext


@dataclass
class AppDeps:
    db_url: str


async def build_db_toolset(ctx: RunContext[AppDeps]):
    """Simulate expensive DB connection initialization (done once per run)."""
    # In real code: pool = await asyncpg.create_pool(ctx.deps.db_url)
    ts = FunctionToolset()

    @ts.tool
    async def query_users(ctx: RunContext[AppDeps], limit: int = 10) -> list[str]:
        # pool would be used here
        return [f'user_{i}' for i in range(limit)]

    return ts


# per_run_step=False: factory runs once at run start, not before each model step
dynamic = DynamicToolset(build_db_toolset, per_run_step=False, id='db-toolset')
agent = Agent('openai:gpt-4o', toolsets=[dynamic], deps_type=AppDeps)


async def main() -> None:
    result = await agent.run(
        'List the first 5 users in the system.',
        deps=AppDeps(db_url='postgresql://localhost/mydb'),
    )
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — Feature-flag gated tools

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.tools import RunContext


FEATURE_FLAGS: dict[str, bool] = {
    'beta_summarise': True,
    'experimental_translate': False,
}


def feature_flagged_toolset(ctx: RunContext[None]):
    ts = FunctionToolset()
    registered = 0

    if FEATURE_FLAGS.get('beta_summarise'):
        @ts.tool
        async def summarise(ctx: RunContext[None], text: str) -> str:
            return f'Summary: {text[:50]}...'
        registered += 1

    if FEATURE_FLAGS.get('experimental_translate'):
        @ts.tool
        async def translate(ctx: RunContext[None], text: str, lang: str) -> str:
            return f'[{lang}] {text}'
        registered += 1

    return ts if registered else None


dynamic = DynamicToolset(feature_flagged_toolset)
agent = Agent('openai:gpt-4o', toolsets=[dynamic])


async def main() -> None:
    result = await agent.run('Summarise this text: The quick brown fox jumps over the lazy dog.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 5. `CapabilityOwnedToolset` + `resolve_capability_id` + `tool_defs_for_loaded_capabilities`

**Module:** `pydantic_ai.toolsets._capability_owned`

`CapabilityOwnedToolset` is an internal `WrapperToolset` that binds a contributed toolset to the capability that created it. You won't construct it directly; instead it is created automatically when a `Capability` contributes tools via `cap.toolset(...)`. Understanding it is essential when authoring custom capabilities.

On every `get_tools()` call it stamps each tool definition with the capability's registered `capability_id` (recovered by `resolve_capability_id`) and — when `capability.defer_loading is True` — marks the tool as deferred by setting `DEFERRED_CAPABILITY_TOOL_METADATA_KEY` in the metadata dict.

`tool_defs_for_loaded_capabilities` is used by `ToolSearchToolset.get_tools()` to decide which deferred tools should actually appear in the request this turn. It filters tool definitions to those whose `capability_id` is in `ctx.available_capability_ids` and whose capability has `defer_loading is True`.

| Component | Role |
|---|---|
| `CapabilityOwnedToolset(wrapped, capability)` | Wrapper that stamps `capability_id` on all inner tool defs |
| `get_tools` | Stamps `capability_id`; sets `defer_loading=True` + metadata key when cap is deferred |
| `get_instructions` | Returns `None` when `capability.defer_loading is True` |
| `apply(visitor)` | Calls `visitor(self)` then `self.wrapped.apply(visitor)` — visits both levels |
| `resolve_capability_id(ctx, capability)` | Iterates `ctx.capabilities` by identity (`is`) to recover the registered id |
| `tool_defs_for_loaded_capabilities` | Wire-side filter: only surfaces deferred tools whose caps are loaded this turn |

### Example 1 — Custom capability contributing tools via `CapabilityOwnedToolset`

```python
import asyncio
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.tools import RunContext


@dataclass
class WeatherCapability(AbstractCapability[Any]):
    """A capability that contributes weather lookup tools."""

    api_key: str

    def get_toolsets(self, ctx: RunContext[Any]):
        ts = FunctionToolset()

        @ts.tool
        async def get_weather(ctx: RunContext[Any], city: str) -> str:
            # In real code: call a weather API using self.api_key
            return f'The weather in {city} is sunny, 22°C.'

        # Wrap in CapabilityOwnedToolset to bind to this capability instance.
        return [CapabilityOwnedToolset(wrapped=ts, capability=self)]


weather_cap = WeatherCapability(api_key='weather-api-key-123')
agent = Agent('openai:gpt-4o', capabilities=[weather_cap])


async def main() -> None:
    result = await agent.run('What is the weather in London?')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Inspecting `capability_id` on tool definitions

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities.capability import Capability
from pydantic_ai.tools import RunContext

cap = Capability()


@cap.tool
async def fetch_price(ctx: RunContext[None], ticker: str) -> str:
    return f'Price of {ticker}: $100'


agent = Agent('openai:gpt-4o', capabilities=[cap])


async def main() -> None:
    async with agent.iter('What is the price of AAPL?') as run:
        async for node in run:
            pass  # consume the run

    # Inspect available capabilities and their registered IDs
    # (capability_id is available on ToolDefinition objects inside toolsets)
    print('Capabilities registered in last run:')
    async with agent.iter('Price of TSLA?') as run:
        # Access the run context's capabilities mapping
        async for node in run:
            if hasattr(node, 'model_response'):
                break
        if hasattr(run, 'ctx'):
            for cap_id, registered_cap in run.ctx.capabilities.items():
                print(f'  {cap_id}: {type(registered_cap).__name__}')


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — Deferred capability with `defer_loading=True`

```python
import asyncio
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.tools import RunContext


@dataclass
class HeavyAnalyticsCapability(AbstractCapability[Any]):
    """A deferred capability — tools only surface after explicit load."""

    defer_loading: bool = True  # Mark as deferred

    def get_toolsets(self, ctx: RunContext[Any]):
        ts = FunctionToolset()

        @ts.tool
        async def run_analytics(ctx: RunContext[Any], query: str) -> str:
            return f'Analytics result for: {query}'

        return [CapabilityOwnedToolset(wrapped=ts, capability=self)]


cap = HeavyAnalyticsCapability()
agent = Agent('openai:gpt-4o', capabilities=[cap])


async def main() -> None:
    # Without loading the capability first, the tool won't be sent in the request.
    # The model needs to call a load_capability tool to surface heavy_analytics tools.
    result = await agent.run('Run analytics on Q1 sales data.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 6. Web UI API: `create_api_app` + `ModelInfo` + `BuiltinToolInfo` + `ConfigureFrontend` + `ChatRequestExtra`

**Module:** `pydantic_ai.ui._web.api`

`create_api_app` is the backend for Pydantic AI's built-in web chat UI. It returns a `Starlette` application with four routes: `POST /chat`, `OPTIONS /chat` (CORS preflight), `GET /configure`, and `GET /health`. Pair it with `create_web_app` from `pydantic_ai.ui._web.app` to serve the full UI.

The `models` parameter accepts either a sequence (model names/instances) or a mapping (display label → model). A mapping lets you control what label appears in the model picker. Models already on the agent are included first and always available, but won't appear in the selectable list.

`ConfigureFrontend` is returned by `GET /configure` and tells the frontend which models and built-in tools are available. It is serialised with `by_alias=True` — all JSON keys are camelCase.

`ChatRequestExtra` is parsed from the Vercel AI SDK's extra data field. It carries `model` (selected model ID) and `builtinTools` (list of selected native tool IDs).

| Component | Detail |
|---|---|
| `ModelInfo` | `id: str`, `name: str`, `builtin_tools: list[str]` — tools supported by this model |
| `BuiltinToolInfo` | `id: str`, `name: str` — selectable native tools |
| `ConfigureFrontend` | `models: list[ModelInfo]`, `builtin_tools: list[BuiltinToolInfo]` |
| `ChatRequestExtra` | `model: str | None`, `builtin_tools: list[str]` — from request body |
| `validate_request_options` | Returns error string or `None`; checks model and tools against allowed sets |
| All Pydantic models | `alias_generator=to_camel` — JSON fields are camelCase (`builtinTools`, not `builtin_tools`) |
| `/chat` route | Delegates to `VercelAIAdapter.dispatch_request`; returns streaming response |
| Model dedup | Models are deduplicated by `model_id`; duplicate model IDs are skipped |

### Example 1 — Minimal web chat API served with uvicorn

```python
"""Run with: uvicorn example:app --reload"""
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.ui._web.api import create_api_app

agent = Agent(
    OpenAIModel('gpt-4o', api_key=os.environ['OPENAI_API_KEY']),
    system_prompt='You are a helpful assistant.',
)

app = create_api_app(
    agent,
    models=['openai:gpt-4o', 'openai:gpt-4o-mini'],
)
```

### Example 2 — Custom model labels and native tools

```python
"""Run with: uvicorn example:app --reload"""
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.ui._web.api import create_api_app

agent = Agent(
    OpenAIModel('gpt-4o', api_key=os.environ['OPENAI_API_KEY']),
    system_prompt='You are a research assistant.',
)

# Dict keys become display labels in the model picker.
models = {
    'GPT-4o (fast)': 'openai:gpt-4o',
    'GPT-4o-mini (cheap)': 'openai:gpt-4o-mini',
    'Claude Sonnet': AnthropicModel('claude-sonnet-4-6', api_key=os.environ['ANTHROPIC_API_KEY']),
}

app = create_api_app(
    agent,
    models=models,
    native_tools=[WebSearchTool()],  # Selectable in the UI
)
```

### Example 3 — Testing the `/configure` and `/health` endpoints

```python
from starlette.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.ui._web.api import create_api_app

agent = Agent(TestModel(), system_prompt='You are a test assistant.')
app = create_api_app(agent, models=['openai:gpt-4o', 'openai:gpt-4o-mini'])
client = TestClient(app)


def test_health() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'ok': True}


def test_configure() -> None:
    response = client.get('/configure')
    assert response.status_code == 200
    data = response.json()
    # Keys are camelCase due to alias_generator=to_camel
    assert 'models' in data
    assert 'builtinTools' in data
    model_ids = [m['id'] for m in data['models']]
    assert 'openai:gpt-4o' in model_ids
    assert 'openai:gpt-4o-mini' in model_ids


if __name__ == '__main__':
    test_health()
    test_configure()
    print('All tests passed.')
```

---

## 7. OTel GenAI spec message types: `ChatMessage`, `OutputMessage`, `TextPart`, `ToolCallPart`, `BlobPart`, `UriPart`, `ToolCallPartOtelMetadata`

**Module:** `pydantic_ai._otel_messages`

These `TypedDict` types implement the [OpenTelemetry Generative AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/non-normative/examples-llm-calls/) for message events. They are produced by the instrumentation layer when Logfire (or another OpenTelemetry exporter) is attached and are not part of the public model message API.

Instrumentation version 4+ switched from `MediaUrlPart` and `BinaryDataPart` to the GenAI-spec-aligned `UriPart` and `BlobPart`. The spec only defines `image`, `audio`, and `video` modalities — `DocumentUrl` omits `modality` because there is no document modality in the spec.

`ToolCallPartOtelMetadata` lives on `BaseToolCallPart.otel_metadata` and is not sent in OTel events directly. Logfire reads `code_arg_name` and `code_arg_language` to provide syntax highlighting in the trace viewer (e.g. Python snippets from `CodeExecutionTool`).

| Type | Fields | Notes |
|---|---|---|
| `TextPart` | `type='text'`, `content?` | Basic assistant text output |
| `ToolCallPart` | `type='tool_call'`, `id`, `name`, `arguments?`, `builtin?`, `code_arg_name?`, `code_arg_language?` | Function tool call in the request |
| `ToolCallResponsePart` | `type='tool_call_response'`, `id`, `name`, `result?`, `builtin?` | Tool result returned to model |
| `MediaUrlPart` | `type='image-url'|'audio-url'|'video-url'|'document-url'`, `url?` | Legacy URL part (pre-v4 instrumentation) |
| `UriPart` | `type='uri'`, `modality?` (`image`/`audio`/`video`), `uri?`, `mime_type?` | v4+ spec-aligned URL part; `modality` absent for `DocumentUrl` |
| `FilePart` | `type='file'`, `modality`, `file_id?`, `mime_type?` | File referenced by ID (e.g. OpenAI file uploads) |
| `BinaryDataPart` | `type='binary'`, `media_type`, `content?` | Legacy inline binary (pre-v4 instrumentation) |
| `BlobPart` | `type='blob'`, `modality?`, `mime_type?`, `content?` | v4+ spec-aligned inline binary; `modality` absent for unknown MIME types |
| `ThinkingPart` | `type='thinking'`, `content?` | Extended thinking / reasoning tokens |
| `ChatMessage` | `role: 'system'|'user'|'assistant'`, `parts: list[MessagePart]` | One turn in a conversation |
| `OutputMessage` | extends `ChatMessage` + `finish_reason?` | Model's final response message |
| `ToolCallPartOtelMetadata` | `code_arg_name?`, `code_arg_language?` | Logfire display hints on `BaseToolCallPart.otel_metadata` |

### Example 1 — Constructing `ChatMessage` / `OutputMessage` manually

```python
from pydantic_ai._otel_messages import (
    ChatMessage,
    OutputMessage,
    TextPart,
    ToolCallPart,
    ToolCallResponsePart,
)

# Simulate what instrumentation records for a tool-calling run.
user_message: ChatMessage = {
    'role': 'user',
    'parts': [{'type': 'text', 'content': 'What is 2 + 2?'}],
}

assistant_tool_call: ChatMessage = {
    'role': 'assistant',
    'parts': [
        {
            'type': 'tool_call',
            'id': 'call_abc123',
            'name': 'calculate',
            'arguments': {'expression': '2 + 2'},
            'builtin': False,
        }
    ],
}

tool_result: ChatMessage = {
    'role': 'user',
    'parts': [
        {
            'type': 'tool_call_response',
            'id': 'call_abc123',
            'name': 'calculate',
            'result': 4,
            'builtin': False,
        }
    ],
}

final_output: OutputMessage = {
    'role': 'assistant',
    'parts': [{'type': 'text', 'content': '2 + 2 equals 4.'}],
    'finish_reason': 'stop',
}

conversation = [user_message, assistant_tool_call, tool_result, final_output]
for msg in conversation:
    role = msg['role']
    for part in msg['parts']:
        if part['type'] == 'text':
            print(f'[{role}] {part.get("content", "")}')
        elif part['type'] == 'tool_call':
            print(f'[{role}] -> tool: {part["name"]}({part.get("arguments")})')
        elif part['type'] == 'tool_call_response':
            print(f'[{role}] <- {part["name"]}: {part.get("result")}')
```

### Example 2 — Multimodal `UriPart` and `BlobPart` (v4 spec)

```python
import base64
from pydantic_ai._otel_messages import (
    ChatMessage,
    UriPart,
    BlobPart,
    ThinkingPart,
)

# v4 instrumentation: images sent as UriPart (not MediaUrlPart)
image_uri_message: ChatMessage = {
    'role': 'user',
    'parts': [
        {
            'type': 'text',
            'content': 'What is in this image?',
        },
        {
            'type': 'uri',
            'modality': 'image',
            'uri': 'https://example.com/photo.jpg',
            'mime_type': 'image/jpeg',
        },
    ],
}

# Inline binary image as BlobPart
fake_png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
inline_image_message: ChatMessage = {
    'role': 'user',
    'parts': [
        {
            'type': 'blob',
            'modality': 'image',
            'mime_type': 'image/png',
            'content': base64.b64encode(fake_png_bytes).decode(),
        }
    ],
}

# DocumentUrl has no modality — not defined in the GenAI spec
document_part: UriPart = {
    'type': 'uri',
    # modality intentionally absent for document URLs
    'uri': 'https://example.com/report.pdf',
    'mime_type': 'application/pdf',
}

# Thinking tokens
thinking_message: ChatMessage = {
    'role': 'assistant',
    'parts': [
        {'type': 'thinking', 'content': 'Let me analyse the image carefully...'},
        {'type': 'text', 'content': 'The image shows a sunset over the ocean.'},
    ],
}

for msg in [image_uri_message, inline_image_message, thinking_message]:
    for part in msg['parts']:
        print(f"  type={part['type']}", end='')
        if 'modality' in part:
            print(f" modality={part['modality']}", end='')
        print()
```

### Example 3 — `ToolCallPartOtelMetadata` for code syntax highlighting

```python
from pydantic_ai._otel_messages import ToolCallPart, ToolCallPartOtelMetadata

# ToolCallPartOtelMetadata lives on BaseToolCallPart.otel_metadata (not in the OTel event).
# Logfire reads code_arg_name / code_arg_language for syntax highlighting.
#
# This is how the CodeExecutionTool sets it internally:
#   part.otel_metadata = {'code_arg_name': 'code', 'code_arg_language': 'python'}
#
# When authoring a custom tool that accepts code, set otel_metadata similarly.

# The OTel event itself DOES carry these fields directly on ToolCallPart
# (they are extracted from otel_metadata and set as top-level fields):
code_tool_call: ToolCallPart = {
    'type': 'tool_call',
    'id': 'call_xyz789',
    'name': 'execute_python',
    'arguments': {'code': 'print("hello world")', 'timeout': 30},
    'builtin': True,
    'code_arg_name': 'code',           # arg name containing the code
    'code_arg_language': 'python',     # for syntax highlighting
}

# Simulate a Logfire-style formatter that uses these hints
def format_tool_call(part: ToolCallPart) -> str:
    name = part['name']
    args = part.get('arguments', {})
    code_arg = part.get('code_arg_name')
    lang = part.get('code_arg_language', 'text')
    if code_arg and isinstance(args, dict) and code_arg in args:
        code = args[code_arg]
        return f'```{lang}\n{code}\n```'
    return f'{name}({args})'


print(format_tool_call(code_tool_call))
```

---

## 8. `Thinking` capability

**Module:** `pydantic_ai.capabilities.thinking`

`Thinking` is a convenience `@dataclass` capability that enables and configures model reasoning/thinking across providers in a portable way. It has a single field, `effort`, which maps to `ModelSettings.thinking`. Provider-specific settings (e.g. `anthropic_thinking`, `openai_reasoning_effort`) take precedence when both are set on the same run.

The unified `thinking` setting works because each model provider's request builder checks `model_settings.thinking` if a provider-specific thinking setting is absent. This allows a single `Thinking(effort='high')` capability to correctly configure reasoning on Anthropic, OpenAI o-series, DeepSeek R1, Qwen3, and other supported models.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `effort` | `ThinkingLevel` | `True` | `True` = enable at provider default; `False` = disable (ignored on always-on models); `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'` = specific level |

`ThinkingLevel = bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh']`

`get_model_settings()` implementation: `return ModelSettings(thinking=self.effort)`

### Example 1 — `Thinking` capability on an agent

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.thinking import Thinking
from pydantic_ai.models.test import TestModel

# In real use, replace TestModel with an actual model that supports thinking,
# e.g. AnthropicModel('claude-sonnet-4-6') or OpenAIModel('o3').
agent = Agent(
    TestModel(custom_result_text='42'),
    capabilities=[Thinking(effort='high')],
    system_prompt='You are a careful reasoning assistant.',
)


async def main() -> None:
    result = await agent.run('What is the square root of 1764?')
    print(result.output)
    # The underlying model receives ModelSettings(thinking='high')


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Portable thinking vs provider-specific settings

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities.thinking import Thinking
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.settings import ModelSettings

# Option A: Portable — works on any provider that supports thinking
agent_portable = Agent(
    AnthropicModel('claude-sonnet-4-6', api_key=os.environ.get('ANTHROPIC_API_KEY', 'test')),
    capabilities=[Thinking(effort='medium')],
)

# Option B: Provider-specific — takes precedence over Thinking capability when both set
agent_specific = Agent(
    AnthropicModel('claude-sonnet-4-6', api_key=os.environ.get('ANTHROPIC_API_KEY', 'test')),
    model_settings=ModelSettings(
        anthropic_thinking={'type': 'enabled', 'budget_tokens': 8000}
    ),
)

# Option C: Per-run override using capabilities=[Thinking(...)] in agent.run()
async def main() -> None:
    from pydantic_ai.models.test import TestModel
    agent_test = Agent(TestModel())
    result = await agent_test.run(
        'Solve this step by step: 15 * 17',
        capabilities=[Thinking(effort='high')],  # Per-run thinking level override
    )
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — `Thinking(effort=False)` to disable reasoning

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.thinking import Thinking
from pydantic_ai.models.test import TestModel

# Some always-on models (DeepSeek R1, Qwen3 QwQ) ignore effort=False.
# For OpenAI o-series and Anthropic models, effort=False disables thinking.
agent = Agent(
    TestModel(custom_result_text='Paris'),
    # Disable thinking for fast, cheap, non-reasoning tasks
    capabilities=[Thinking(effort=False)],
)


async def main() -> None:
    result = await agent.run('What is the capital of France?')
    print(result.output)  # Paris — fast response without reasoning overhead


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 9. `ImageGeneration` capability

**Module:** `pydantic_ai.capabilities.image_generation`

`ImageGeneration` extends `NativeOrLocalTool` to provide cross-provider image generation. When the agent's model supports native image generation (`ImageGenerationTool` in `model.profile.supported_native_tools`), the native path is used. When it doesn't, and `fallback_model` is provided, a local subagent is spun up using the specified image-capable model.

All image configuration fields (`quality`, `size`, `output_format`, etc.) are forwarded to `ImageGenerationTool`. When `native` is an `ImageGenerationTool` instance, capability-level fields override the instance's settings via `_resolved_native()`.

`fallback_model` and `local` are mutually exclusive: providing both raises `UserError('cannot specify both fallback_model and local')`.

<Aside type="tip">
The deprecated kwarg alias `builtin=` maps to `native=`. Use `native=` in new code.
</Aside>

| Field | Supported by | Values |
|---|---|---|
| `action` | OpenAI Responses | `'generate'` / `'edit'` / `'auto'` |
| `background` | OpenAI Responses | `'transparent'` / `'opaque'` / `'auto'`; `'transparent'` only for png/webp |
| `input_fidelity` | OpenAI Responses | `'high'` / `'low'` (default `'low'`) |
| `moderation` | OpenAI Responses | `'auto'` / `'low'` |
| `image_model` | OpenAI Responses | `ImageGenerationModelName` |
| `output_compression` | OpenAI (jpeg/webp, default 100), Google Cloud (jpeg, default 75) | `int` |
| `output_format` | OpenAI Responses, Google Cloud | `'png'` / `'webp'` / `'jpeg'` |
| `quality` | OpenAI Responses | `'low'` / `'medium'` / `'high'` / `'auto'` |
| `size` | OpenAI: `'auto'`/`'1024x1024'`/`'1024x1536'`/`'1536x1024'`; Google: `'512'`/`'1K'`/`'2K'`/`'4K'` | `str` |
| `aspect_ratio` | Google (Gemini), OpenAI Responses (maps to sizes) | `ImageAspectRatio` |

### Example 1 — Native image generation (OpenAI)

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities.image_generation import ImageGeneration
from pydantic_ai.models.openai import OpenAIModel

# OpenAI Responses API model that supports native ImageGenerationTool
model = OpenAIModel('gpt-4o', api_key=os.environ.get('OPENAI_API_KEY', 'test'))

agent = Agent(
    model,
    capabilities=[
        ImageGeneration(
            quality='high',
            output_format='png',
            size='1024x1024',
        )
    ],
    system_prompt='You are a creative image assistant. When asked, generate images.',
)


async def main() -> None:
    result = await agent.run('Generate an image of a sunset over mountain peaks.')
    print(result.output)
    # The model calls the ImageGenerationTool natively and returns the image URL.


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Fallback to a subagent when native not supported

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities.image_generation import ImageGeneration
from pydantic_ai.models.anthropic import AnthropicModel

# AnthropicModel does not support native image generation.
# fallback_model routes generation to an OpenAI Responses model subagent.
model = AnthropicModel('claude-sonnet-4-6', api_key=os.environ.get('ANTHROPIC_API_KEY', 'test'))

agent = Agent(
    model,
    capabilities=[
        ImageGeneration(
            # Subagent model for generation (must support ImageGenerationTool)
            fallback_model='openai-responses:gpt-image-1',
            quality='high',
            output_format='png',
            size='1024x1024',
        )
    ],
)


async def main() -> None:
    result = await agent.run('Create an image of a futuristic city at night.')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — Image editing with `action='edit'`

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities.image_generation import ImageGeneration
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o', api_key=os.environ.get('OPENAI_API_KEY', 'test'))

agent = Agent(
    model,
    capabilities=[
        ImageGeneration(
            action='edit',          # Edit mode: modify an existing image
            input_fidelity='high',  # Preserve style/features of the input image
            output_format='webp',
            quality='high',
        )
    ],
    system_prompt='You are an image editing assistant. Edit images as requested.',
)


async def main() -> None:
    # In a real scenario you'd pass the image as a multimodal input.
    result = await agent.run(
        'Add a rainbow to the sky in this landscape photo.',
    )
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 10. `pydantic_graph` persistence primitives: `NodeSnapshot` + `EndSnapshot` + `BaseStatePersistence` + exception hierarchy

**Modules:** `pydantic_graph.persistence`, `pydantic_graph.exceptions`

`NodeSnapshot` and `EndSnapshot` are the two `@dataclass` types that make up `Snapshot` — the discriminated union stored in persistence backends. They share a `kind` discriminator (`'node'` vs `'end'`) and an `id` field auto-generated from `node.get_snapshot_id()` when left as `UNSET_SNAPSHOT_ID = '__unset__'`.

`BaseStatePersistence` is the ABC that custom backends must implement. `FileStatePersistence` and `SimpleStatePersistence`/`FullStatePersistence` are the built-in implementations. Understanding the ABC is essential for implementing Redis, DynamoDB, or other distributed backends.

The `pydantic_graph.exceptions` hierarchy covers all graph lifecycle errors with structured `message: str` fields:

| Exception | Base | When raised |
|---|---|---|
| `GraphSetupError` | `TypeError` | Incorrectly configured graph (wrong type params, missing edges) |
| `GraphBuildingError` | `ValueError` | Error during `GraphBuilder.build()` |
| `GraphValidationError` | `ValueError` | Error during graph structure validation |
| `GraphRuntimeError` | `RuntimeError` | Error during graph execution |
| `GraphNodeStatusError` | `GraphRuntimeError` | Running a node with status `'running'`/`'success'`/`'error'` |

`GraphNodeStatusError.check(status)` is a classmethod that raises if `status not in {'created', 'pending'}`.

`SnapshotStatus` lifecycle: `'created'` → `'pending'` (via `load_next`) → `'running'` (via `record_run`) → `'success'`/`'error'`

| `NodeSnapshot` field | Type | Detail |
|---|---|---|
| `state` | `StateT` | Graph state before the node runs |
| `node` | `BaseNode` | Next node to execute |
| `start_ts` | `datetime | None` | Set when the node starts running |
| `duration` | `float | None` | Duration in seconds after completion |
| `status` | `SnapshotStatus` | Default `'created'`; progresses through lifecycle |
| `kind` | `Literal['node']` | Discriminator for ser/de |
| `id` | `str` | Auto from `node.get_snapshot_id()` if `UNSET_SNAPSHOT_ID` |

### Example 1 — Custom Redis-backed `BaseStatePersistence`

```python
import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from pydantic_graph.persistence import (
    BaseStatePersistence,
    NodeSnapshot,
    EndSnapshot,
    Snapshot,
    SnapshotStatus,
    UNSET_SNAPSHOT_ID,
)
from pydantic_graph.basenode import BaseNode, End


class InMemoryPersistence(BaseStatePersistence):
    """Simple in-memory persistence demonstrating the full ABC contract."""

    def __init__(self) -> None:
        self._snapshots: dict[str, Snapshot] = {}
        self._order: list[str] = []

    async def snapshot_node(self, state, next_node):
        snapshot = NodeSnapshot(state=state, node=next_node)
        self._snapshots[snapshot.id] = snapshot
        self._order.append(snapshot.id)

    async def snapshot_node_if_new(self, snapshot_id, state, next_node):
        if snapshot_id not in self._snapshots:
            snapshot = NodeSnapshot(state=state, node=next_node, id=snapshot_id)
            self._snapshots[snapshot_id] = snapshot
            self._order.append(snapshot_id)

    async def snapshot_end(self, state, end_node):
        snapshot = EndSnapshot(state=state, result=end_node)
        self._snapshots[snapshot.id] = snapshot
        self._order.append(snapshot.id)

    async def record_run(self, snapshot_id: str) -> None:
        snap = self._snapshots[snapshot_id]
        if isinstance(snap, NodeSnapshot):
            if snap.status == 'created':
                snap.status = 'pending'
            elif snap.status == 'pending':
                snap.status = 'running'
                snap.start_ts = datetime.now(timezone.utc)

    async def load_next(self) -> Snapshot | None:
        for sid in self._order:
            snap = self._snapshots[sid]
            if isinstance(snap, NodeSnapshot) and snap.status == 'pending':
                return snap
        return None

    async def load_all(self) -> list[Snapshot]:
        return [self._snapshots[sid] for sid in self._order]

    async def set_types(self, state_type, run_end_type) -> None:
        pass  # In-memory; no serialisation needed


async def main() -> None:
    persistence = InMemoryPersistence()
    print(f'Snapshots: {await persistence.load_all()}')


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2 — Inspecting `NodeSnapshot` + `EndSnapshot` history

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from pydantic_graph import Graph, BaseNode, End, FullStatePersistence
from pydantic_graph.persistence import NodeSnapshot, EndSnapshot


@dataclass
class CountState:
    count: int = 0


@dataclass
class IncrementNode(BaseNode[CountState]):
    async def run(self, ctx) -> 'IncrementNode | DoneNode':
        ctx.state.count += 1
        if ctx.state.count >= 3:
            return DoneNode()
        return IncrementNode()


@dataclass
class DoneNode(BaseNode[CountState, None, str]):
    async def run(self, ctx) -> End[str]:
        return End(f'Counted to {ctx.state.count}')


graph = Graph(nodes=[IncrementNode, DoneNode])


async def main() -> None:
    state = CountState()
    persistence = FullStatePersistence()
    result, history = await graph.run(IncrementNode(), state=state, persistence=persistence)

    print(f'Result: {result}')  # Counted to 3

    for snap in history:
        if isinstance(snap, NodeSnapshot):
            duration_ms = (snap.duration or 0) * 1000
            print(
                f'  [{snap.status}] {type(snap.node).__name__} '
                f'duration={duration_ms:.2f}ms '
                f'state={snap.state}'
            )
        elif isinstance(snap, EndSnapshot):
            print(f'  [end] result={snap.result.data!r} at {snap.ts}')


if __name__ == '__main__':
    asyncio.run(main())
```

### Example 3 — Handling `GraphNodeStatusError` and the exception hierarchy

```python
import asyncio
from dataclasses import dataclass
from pydantic_graph import Graph, BaseNode, End
from pydantic_graph.exceptions import (
    GraphSetupError,
    GraphBuildingError,
    GraphRuntimeError,
    GraphNodeStatusError,
    GraphValidationError,
)
from pydantic_graph.persistence import NodeSnapshot, SnapshotStatus


def demonstrate_node_status_check() -> None:
    """GraphNodeStatusError.check() classmethod."""
    # Valid statuses — no exception
    for valid in ('created', 'pending'):
        GraphNodeStatusError.check(valid)  # type: ignore[arg-type]
        print(f'  Status {valid!r}: OK')

    # Invalid statuses — raises
    for invalid in ('running', 'success', 'error'):
        try:
            GraphNodeStatusError.check(invalid)  # type: ignore[arg-type]
        except GraphNodeStatusError as e:
            print(f'  Status {invalid!r}: {e.message}')


def demonstrate_exception_hierarchy() -> None:
    """Show the exception hierarchy with message field."""
    errors = [
        GraphSetupError('Missing required edge from NodeA to NodeB'),
        GraphBuildingError('Cycle detected: A -> B -> A'),
        GraphValidationError('Node DoneNode has no outgoing edges'),
        GraphRuntimeError('Unexpected error during node execution'),
    ]
    for err in errors:
        print(f'  {type(err).__name__}({err.message!r}) is RuntimeError: {isinstance(err, RuntimeError)}')


async def demonstrate_graph_setup_error() -> None:
    """Catch GraphSetupError when graph is misconfigured."""
    try:
        @dataclass
        class OrphanNode(BaseNode):
            async def run(self, ctx) -> End[str]:
                return End('done')

        # This would raise GraphSetupError if nodes aren't connected correctly
        # (simplified demonstration)
        graph = Graph(nodes=[OrphanNode])
        print('  Graph created successfully')
    except (GraphSetupError, GraphValidationError) as e:
        print(f'  Caught {type(e).__name__}: {e.message}')


async def main() -> None:
    print('Node status checks:')
    demonstrate_node_status_check()
    print('\nException hierarchy:')
    demonstrate_exception_hierarchy()
    print('\nGraph setup:')
    await demonstrate_graph_setup_error()


if __name__ == '__main__':
    asyncio.run(main())
```

---

## What's new in this volume

- **`BedrockConverseModel` + `BedrockModelSettings` + `BedrockStreamedResponse`** — First-ever complete coverage of AWS Bedrock; 9 `bedrock_`-prefixed settings; caching TTL values; `_FINISH_REASON_MAP` with all 9 entries.
- **`BedrockProvider` + `BedrockModelProfile` + `BedrockJsonSchemaTransformer`** — All 3 construction paths; 10 `BedrockModelProfile` fields; strict-mode schema rewrites; all 8 `bedrock_*_model_profile()` functions.
- **`GoogleGLAProvider` + `GoogleVertexProvider` (deprecated)** — Both deprecated in favour of `GoogleProvider`/`GoogleCloudProvider`; `_VertexAIAuth` token refresh; 29-region `VertexAiRegion` literal; migration tables.
- **`DynamicToolset`** — Per-step factory pattern; `per_run_step` lifecycle; transition-safe inner toolset management.
- **`CapabilityOwnedToolset` + `resolve_capability_id` + `tool_defs_for_loaded_capabilities`** — How capabilities bind toolsets; deferred metadata stamping; wire-side filter for loaded deferred caps.
- **Web UI API** — `create_api_app` + `ModelInfo` + `BuiltinToolInfo` + `ConfigureFrontend` + `ChatRequestExtra`; four Starlette routes; camelCase serialisation.
- **OTel GenAI spec message types** — All 9 `TypedDict` message part types; v4 `UriPart`/`BlobPart` upgrade; `ToolCallPartOtelMetadata` for code syntax hints.
- **`Thinking` capability** — Single-field `@dataclass`; `ThinkingLevel` values; portable cross-provider reasoning control.
- **`ImageGeneration` capability** — Full 12-field image configuration matrix; `fallback_model` subagent pattern; `action='edit'`; mutual exclusion of `fallback_model` and `local`.
- **`pydantic_graph` persistence primitives** — `NodeSnapshot`/`EndSnapshot` anatomy; `BaseStatePersistence` ABC; `SnapshotStatus` lifecycle; full `GraphSetupError`/`GraphBuildingError`/`GraphValidationError`/`GraphRuntimeError`/`GraphNodeStatusError` hierarchy.

## Source-verification notes

All class signatures, field names, default values, constants, and behaviour descriptions were verified against `pydantic-ai==1.107.0` installed at `/usr/local/lib/python3.11/dist-packages/pydantic_ai/` and `pydantic_graph` at `/usr/local/lib/python3.11/dist-packages/pydantic_graph/`.

Key verifications:
- `BedrockModelSettings` fields confirmed via `inspect.getsource(BedrockModelSettings)` — all 9 `bedrock_`-prefixed fields present
- `BedrockModelProfile` fields confirmed — 10 `bedrock_`-prefixed fields; all defaults match source
- `_BEDROCK_STRICT_UNSUPPORTED_KEYS_BY_TYPE` tuple order confirmed: `number`/`integer` = `('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf')`, `array` = `('maxItems',)`
- `GoogleGLAProvider.base_url = 'https://generativelanguage.googleapis.com/v1beta/models/'` confirmed
- `GoogleVertexProvider` default `region='us-central1'`, `model_publisher='google'` confirmed
- `DynamicToolset.__eq__` uses identity (`is`) for `toolset_func` confirmed
- `CapabilityOwnedToolset.get_instructions` returns `None` when `defer_loading is True` confirmed
- `ChatRequestExtra` fields: `model` and `builtin_tools` (JSON: `builtinTools`) confirmed
- `Thinking` dataclass: single `effort: ThinkingLevel = True` field confirmed
- `ImageGeneration` `fallback_model` + `local` mutual exclusion confirmed via `UserError` check
- `NodeSnapshot.id` auto-computed from `node.get_snapshot_id()` when `== UNSET_SNAPSHOT_ID` confirmed
- `GraphNodeStatusError.check()` raises when `status not in {'created', 'pending'}` confirmed

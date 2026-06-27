---
title: "PydanticAI Class Deep Dives Vol. 26"
description: "Source-verified deep dives into 10 pydantic-ai 2.0.0 class groups: NativeOrLocalTool base class (native=True/False/instance, local strategy resolution, PreparedToolset + unless_native injection, _requires_native guard), WebSearch capability (DuckDuckGo fallback via local='duckduckgo'/local=True, blocked_domains/allowed_domains/max_uses native-only constraint fields), WebFetch capability (markdownify-based local fallback, enable_citations/max_content_tokens, SSRF-safe local, per-hop domain validation), ImageGeneration capability (fallback_model subagent pattern, 12 config fields, _image_gen_kwargs override bridge, action/background/input_fidelity/quality/size/aspect_ratio), Thinking capability (ThinkingLevel Literal — True/False/minimal/low/medium/high/xhigh, unified cross-provider thinking, provider-specific override precedence), ThreadExecutor capability (concurrent.futures.Executor wrapping, anyio.to_thread.run_sync replacement, Agent.using_thread_executor() class-level context manager), TenacityTransport + AsyncTenacityTransport + RetryConfig + wait_retry_after (tenacity-backed HTTP retry transport, Retry-After header honoring, validate_response callback), DeferredLoadingToolset (PreparedToolset subclass, defer_loading=True injection, tool_names= partial marking, deferred tool discovery via ToolSearch), PrefectAgent + TaskConfig (Prefect durable workflow wrapper, per-tool task config, prefectify_toolset dispatch, default_task_config cache policy), split_content_into_text_and_thinking + HistoryProcessor (tag-splitting TextPart/ThinkingPart parser, four-variant HistoryProcessor TypeAlias — sync/async × with/without RunContext). All verified against pydantic-ai 2.0.0 source."
sidebar:
  label: "Class deep dives (Vol. 26)"
  order: 52
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.0.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.x API.
</Aside>

Ten class groups covering the new `NativeOrLocalTool` architectural pattern and its three built-in capability subclasses (`WebSearch`, `WebFetch`, `ImageGeneration`), the unified `Thinking` capability, production-safe `ThreadExecutor`, tenacity-backed HTTP retry transports, the `DeferredLoadingToolset` for lazy tool discovery, Prefect durable workflow integration, and the utility functions for thinking-tag parsing and history processing.

---

## 1. `NativeOrLocalTool` — Capability Base for Native/Local Fallback Pairs

**Source**: `pydantic_ai/capabilities/native_or_local.py`

`NativeOrLocalTool` is the v2.0.0 architectural base class that implements the *provider-native tool with local fallback* pattern used by `WebSearch`, `WebFetch`, and `ImageGeneration`. It handles the resolution logic, the `unless_native` injection into local tool definitions, and the constraint-field guard that prevents silent failures when native-only features are enabled.

```python
@dataclass(init=False)
class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    native: AgentNativeTool[AgentDepsT] | bool = True
    # True  → _default_native() (subclass hook)
    # False → disable native, always use local
    # AbstractNativeTool instance → use this config
    # NativeToolFunc → dynamically create native tool per-run

    local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = None
    # None  → _default_local() (subclass hook)
    # True  → _resolve_local_strategy(True)
    # str   → _resolve_local_strategy(name)
    # False → disable local, only use native
    # Tool / AbstractToolset / callable → use directly

    # Subclass hooks
    def _default_native(self) -> AbstractNativeTool | None: ...
    def _native_unique_id(self) -> str: ...
    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None: ...
    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]: ...
    def _requires_native(self) -> bool: ...   # True → local suppressed; native=False raises UserError

    # AbstractCapability protocol
    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]: ...
    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None: ...
    # get_toolset() wraps local in PreparedToolset that adds unless_native=uid to every ToolDefinition
    # Models that support the native tool see native only; models that don't see only local
```

### Key behaviour: `unless_native` injection

When both `native` and `local` are enabled, `get_toolset()` wraps the local toolset in a `PreparedToolset` that calls `_add_unless_native`, which sets `tool_def.unless_native = uid` on every local tool definition. The model infrastructure removes these tools from the prompt when the model supports the corresponding native tool — the fallback is completely invisible to capable models.

### 1.1 Direct Use Without Subclassing

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.tools import Tool


async def my_search(query: str) -> str:
    """Perform a DuckDuckGo search."""
    return f"Results for: {query}"


async def main() -> None:
    # Build a NativeOrLocalTool without subclassing
    cap = NativeOrLocalTool(
        native=WebSearchTool(),
        local=Tool(my_search),
    )

    agent = Agent('openai:gpt-4o-mini', capabilities=[cap])
    result = await agent.run("Search for pydantic-ai 2.0.0 release notes")
    print(result.output)


asyncio.run(main())
```

### 1.2 Building a Custom NativeOrLocalTool Subclass

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.tools import AgentDepsT, Tool


@dataclass(init=False)
class MySearch(NativeOrLocalTool[AgentDepsT]):
    """Custom web-search capability that defaults to DuckDuckGo locally."""

    def __init__(self, *, local_first: bool = False) -> None:
        # native=True uses _default_native; local=True uses _resolve_local_strategy(True)
        super().__init__(native=not local_first, local=True)

    def _default_native(self) -> WebSearchTool:
        return WebSearchTool(search_context_size='medium')

    def _native_unique_id(self) -> str:
        return WebSearchTool.kind

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT]:
        # This local strategy requires the duckduckgo extra:
        # pip install "pydantic-ai-slim[duckduckgo]"
        async def duck_search(query: str) -> str:
            from duckduckgo_search import DDGS
            results = DDGS().text(query, max_results=3)
            return "\n".join(r["body"] for r in results)

        return Tool(duck_search)


async def main() -> None:
    agent = Agent('openai:gpt-4o-mini', capabilities=[MySearch()])
    result = await agent.run("What is new in pydantic-ai 2.0?")
    print(result.output)


asyncio.run(main())
```

### 1.3 Forcing Native-Only with `_requires_native`

```python
import asyncio
from dataclasses import dataclass
from typing import Literal
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.tools import AgentDepsT


@dataclass(init=False)
class ConstrainedSearch(NativeOrLocalTool[AgentDepsT]):
    """Search that requires native if domain constraints are set."""

    blocked_domains: list[str] | None

    def __init__(self, *, blocked_domains: list[str] | None = None) -> None:
        self.blocked_domains = blocked_domains
        # local=False because this subclass has no _resolve_local_strategy() override;
        # passing local=True without one raises UserError in __post_init__.
        super().__init__(native=True, local=False)

    def _default_native(self) -> WebSearchTool:
        kwargs = {}
        if self.blocked_domains:
            kwargs['blocked_domains'] = self.blocked_domains
        return WebSearchTool(**kwargs)

    def _native_unique_id(self) -> str:
        return WebSearchTool.kind

    def _requires_native(self) -> bool:
        # Returning True suppresses local and enforces native-only
        return self.blocked_domains is not None


async def main() -> None:
    cap = ConstrainedSearch(blocked_domains=['spam.example.com'])
    # openai-responses: provider is required for native WebSearch on OpenAI;
    # openai: (Chat Completions) does not support WebSearchTool.
    agent = Agent('openai-responses:gpt-4o-mini', capabilities=[cap])
    result = await agent.run("Search for safe content")
    print(result.output)


asyncio.run(main())
```

---

## 2. `WebSearch` — Native Search with DuckDuckGo Fallback

**Source**: `pydantic_ai/capabilities/web_search.py`

`WebSearch` extends `NativeOrLocalTool` to provide a ready-made web-search capability. `local='duckduckgo'` (or the shorthand `local=True`) wires up the built-in DuckDuckGo fallback. Constraint fields (`blocked_domains`, `allowed_domains`, `max_uses`) are native-only: setting any of them causes `_requires_native()` to return `True`, suppressing the local tool.

```python
@dataclass(init=False)
class WebSearch(NativeOrLocalTool[AgentDepsT]):
    search_context_size: Literal['low', 'medium', 'high'] | None
    # Controls how much web context is retrieved — native only
    user_location: WebSearchUserLocation | None
    # Localise results — native only
    blocked_domains: list[str] | None     # native-only constraint
    allowed_domains: list[str] | None     # native-only constraint
    max_uses: int | None                  # native-only constraint

    def __init__(
        self,
        *,
        native: WebSearchTool | ... | bool = True,
        local: WebSearchLocalStrategy | Tool | Callable | bool | None = None,
        # local='duckduckgo' or local=True → duckduckgo_search_tool()
        # Requires: pip install "pydantic-ai-slim[duckduckgo]"
        search_context_size: ...,
        user_location: ...,
        blocked_domains: ...,
        allowed_domains: ...,
        max_uses: ...,
        ...
    ) -> None: ...
```

### 2.1 Basic Web Search with DuckDuckGo Fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch


async def main() -> None:
    # local=True falls back to DuckDuckGo when the model doesn't support native search.
    # Requires: pip install "pydantic-ai-slim[duckduckgo]"
    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[WebSearch(local=True)],
    )
    result = await agent.run("What happened in AI news this week?")
    print(result.output)


asyncio.run(main())
```

### 2.2 Localised Search with Context Size Control

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.native_tools import WebSearchUserLocation


async def main() -> None:
    cap = WebSearch(
        search_context_size='high',       # more retrieved context per search
        user_location=WebSearchUserLocation(
            city='London',
            country='GB',
        ),
    )

    agent = Agent('openai:gpt-4o-mini', capabilities=[cap])
    result = await agent.run("What's the weather forecast near me?")
    print(result.output)


asyncio.run(main())
```

### 2.3 Domain Filtering (Native-Only — Forces Native)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch


async def main() -> None:
    # blocked_domains triggers _requires_native() → True,
    # so the local DuckDuckGo fallback is suppressed entirely.
    # Use blocked_domains OR allowed_domains — providers accept only one, not both.
    # blocked_domains and max_uses are Anthropic-only; OpenAI supports allowlists only.
    cap = WebSearch(
        blocked_domains=['reddit.com', 'twitter.com'],
        max_uses=5,
    )

    agent = Agent('anthropic:claude-opus-4-8', capabilities=[cap])
    result = await agent.run("Find recent transformer architecture papers")
    print(result.output)


asyncio.run(main())
```

---

## 3. `WebFetch` — URL Fetching with SSRF-Safe Local Fallback

**Source**: `pydantic_ai/capabilities/web_fetch.py`

`WebFetch` extends `NativeOrLocalTool` for URL fetching. `local=True` activates the built-in markdownify-based local tool from `pydantic_ai.common_tools.web_fetch`, which applies the full SSRF protection from `_ssrf.safe_download` before fetching. The `allowed_domains` and `blocked_domains` constraint fields are enforced by the local tool on every hop; `max_uses` and `enable_citations` are native-only.

```python
@dataclass(init=False)
class WebFetch(NativeOrLocalTool[AgentDepsT]):
    allowed_domains: list[str] | None   # enforced locally too (via safe_download)
    blocked_domains: list[str] | None   # enforced locally too
    max_uses: int | None                # native-only
    enable_citations: bool | None       # native-only
    max_content_tokens: int | None      # native-only

    def _requires_native(self) -> bool:
        # Only max_uses forces native — domain lists are implemented locally too
        return self.max_uses is not None
```

### 3.1 Local Web Fetch with SSRF Protection

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch


async def main() -> None:
    # local=True activates markdownify-based fetch with full SSRF protection.
    # Requires: pip install "pydantic-ai-slim[web-fetch]"
    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[WebFetch(local=True)],
    )
    result = await agent.run(
        "Fetch https://docs.pydantic.dev and summarise the quick-start section"
    )
    print(result.output)


asyncio.run(main())
```

### 3.2 Restricting Fetch to Allowed Domains

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch


async def main() -> None:
    # allowed_domains is forwarded to web_fetch_tool(), which enforces it via safe_download.
    # The local fallback respects the allowlist even without native support.
    cap = WebFetch(
        local=True,
        allowed_domains=['docs.python.org', 'docs.pydantic.dev', 'pydantic.dev'],
    )

    agent = Agent('openai:gpt-4o-mini', capabilities=[cap])
    result = await agent.run(
        "Fetch https://docs.pydantic.dev/latest/ and tell me what's new in v2"
    )
    print(result.output)


asyncio.run(main())
```

### 3.3 Native Fetch with Citations and Token Budget

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch


async def main() -> None:
    # max_uses and enable_citations are native-only.
    # _requires_native() returns True only when max_uses is set.
    cap = WebFetch(
        enable_citations=True,      # native: embed source citations in response
        max_content_tokens=4096,    # native: truncate fetched content
        max_uses=3,                 # native-only: limits total fetch calls per run
    )

    # Native WebFetchTool is supported by Anthropic and Google only, not OpenAI.
    agent = Agent('anthropic:claude-opus-4-8', capabilities=[cap])
    result = await agent.run(
        "Fetch https://arxiv.org/abs/2502.11157 and summarise the key contributions"
    )
    print(result.output)


asyncio.run(main())
```

---

## 4. `ImageGeneration` — Native Image Generation with Subagent Fallback

**Source**: `pydantic_ai/capabilities/image_generation.py`

`ImageGeneration` pairs the model's native `ImageGenerationTool` with an optional `fallback_model` that spawns a subagent running a dedicated image-capable model. It carries 12 config fields mirroring `ImageGenerationTool` (`action`, `background`, `input_fidelity`, `moderation`, `image_model`, `output_compression`, `output_format`, `quality`, `size`, `aspect_ratio`, plus `id`, `description`). The `_image_gen_kwargs()` bridge applies non-`None` fields to both the native tool and the subagent fallback.

```python
@dataclass(init=False)
class ImageGeneration(NativeOrLocalTool[AgentDepsT]):
    fallback_model: ImageGenerationFallbackModel
    # Model to run as fallback subagent when native image gen is unavailable.
    # Accepts: model name str, Model instance, or Callable[[RunContext], Model|str]

    action: Literal['generate', 'edit', 'auto'] | None
    background: Literal['transparent', 'opaque', 'auto'] | None
    input_fidelity: Literal['high', 'low'] | None
    moderation: Literal['auto', 'low'] | None
    image_model: ImageGenerationModelName | None
    output_compression: int | None
    output_format: Literal['png', 'webp', 'jpeg'] | None
    quality: Literal['low', 'medium', 'high', 'auto'] | None
    size: Literal['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'] | None
    aspect_ratio: ImageAspectRatio | None
```

### 4.1 Basic Image Generation with Fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration


async def main() -> None:
    # If the main model supports native image generation, it is used directly.
    # Otherwise, a subagent running 'openai-responses:gpt-5.4' generates the image.
    cap = ImageGeneration(
        fallback_model='openai-responses:gpt-5.4',
        quality='high',
        output_format='png',
    )

    agent = Agent('openai:gpt-4o-mini', capabilities=[cap])
    result = await agent.run("Generate a futuristic city skyline at dusk")
    print(result.output)


asyncio.run(main())
```

### 4.2 Transparent PNG with Background Control

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration


async def main() -> None:
    # 'transparent' background is only supported with png/webp output formats.
    cap = ImageGeneration(
        action='generate',
        background='transparent',
        output_format='png',
        size='1024x1024',
        quality='high',
    )

    agent = Agent(
        'openai-responses:gpt-5.4',   # model with native image generation
        capabilities=[cap],
    )
    result = await agent.run("Draw a pydantic snake mascot with no background")
    print(result.output)


asyncio.run(main())
```

### 4.3 Image Editing with Input Fidelity Control

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.messages import BinaryContent


async def main() -> None:
    # action='edit' tells the model to modify an existing image.
    # input_fidelity='high' preserves more style/features from the input.
    cap = ImageGeneration(
        action='edit',
        input_fidelity='high',
        output_format='webp',
        quality='medium',
    )

    with open('original.png', 'rb') as f:
        image_bytes = f.read()

    agent = Agent('openai-responses:gpt-5.4', capabilities=[cap])
    result = await agent.run([
        'Add a sunset gradient to the sky in this image',
        BinaryContent(data=image_bytes, media_type='image/png'),
    ])
    print(result.output)


asyncio.run(main())
```

---

## 5. `Thinking` — Unified Cross-Provider Reasoning Control

**Source**: `pydantic_ai/capabilities/thinking.py`

`Thinking` is a thin capability that injects a `ModelSettings(thinking=...)` into every run. The `ThinkingLevel` type is `bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh']`. Provider-specific thinking settings (`anthropic_thinking`, `openai_reasoning_effort`) take precedence when both are set, so `Thinking` is a safe portable default that can be overridden per-provider without conflict.

```python
@dataclass
class Thinking(AbstractCapability[Any]):
    effort: ThinkingLevel = True
    # True   → enable with provider default
    # False  → disable (silently ignored on always-on models like o3)
    # 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' → specific effort tier

    def get_model_settings(self) -> ModelSettings | None:
        return ModelSettings(thinking=self.effort)
```

### 5.1 Enable Thinking with Provider Default

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking


async def main() -> None:
    # Thinking(effort=True) tells pydantic-ai to enable thinking at the provider's default.
    # On Anthropic: uses budget_tokens heuristic. On OpenAI: uses 'medium' reasoning effort.
    agent = Agent(
        'anthropic:claude-opus-4-8',
        capabilities=[Thinking()],   # effort=True by default
    )
    result = await agent.run("Prove that √2 is irrational using a proof by contradiction")
    print(result.output)


asyncio.run(main())
```

### 5.2 Effort Tiers for Cost/Quality Trade-Off

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking


async def main() -> None:
    # Use 'xhigh' for complex multi-step reasoning tasks where quality matters most.
    agent = Agent(
        'openai:o3',
        capabilities=[Thinking(effort='xhigh')],
    )
    result = await agent.run(
        "Design an algorithm that finds all prime pairs (p, p+2) up to 10^9 in under 1 second"
    )
    print(result.output)


asyncio.run(main())
```

### 5.3 Disabling Thinking on Always-Off Paths

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking


async def main() -> None:
    # effort=False is a signal to disable thinking.
    # On models that always think (o3), this is silently ignored.
    # On models where it can be toggled (claude-sonnet-4-6), it turns it off.
    fast_agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[Thinking(effort=False)],
    )
    result = await fast_agent.run("What is 42 + 17?")
    print(result.output)  # Fast, no extended thinking budget


asyncio.run(main())
```

---

## 6. `ThreadExecutor` — Bounded Thread Pool for Production Servers

**Source**: `pydantic_ai/capabilities/thread_executor.py`

By default, pydantic-ai runs sync tools and callbacks in ephemeral threads via `anyio.to_thread.run_sync`. In high-throughput servers this can create unbounded thread growth under load. `ThreadExecutor` accepts any `concurrent.futures.Executor` (typically a `ThreadPoolExecutor`) and scopes it to agent runs, replacing the default anyio ephemeral threads. A class-level alternative is `Agent.using_thread_executor()`.

```python
@dataclass
class ThreadExecutor(AbstractCapability[Any]):
    executor: Executor
    # The executor to use for running sync functions inside agent runs.
    # Wraps the run via _utils.using_thread_executor(self.executor).

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None   # not serialisable — executor holds OS resources

    async def wrap_run(self, ctx, *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        with _utils.using_thread_executor(self.executor):
            return await handler()
```

### 6.1 Bounded Thread Pool for FastAPI

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

# Create once at startup — shared across all requests
executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='agent-worker')
agent = Agent('openai:gpt-4o-mini', capabilities=[ThreadExecutor(executor)])


async def handle_request(prompt: str) -> str:
    result = await agent.run(prompt)
    return result.output


# In practice, mount inside a FastAPI app:
# app = FastAPI()
# @app.on_event('startup')
# def startup():
#     executor.__enter__()
#
# @app.on_event('shutdown')
# def shutdown():
#     executor.shutdown(wait=True)
```

### 6.2 Class-Level `using_thread_executor` Context Manager

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pydantic_ai import Agent

executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix='batch')
agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    # using_thread_executor sets the executor for ALL runs inside the context,
    # including runs on agents that don't have ThreadExecutor in their capabilities.
    with Agent.using_thread_executor(executor):
        results = await asyncio.gather(
            agent.run("Task A"),
            agent.run("Task B"),
            agent.run("Task C"),
        )
    for r in results:
        print(r.output)


asyncio.run(main())
```

### 6.3 Bounded ThreadPoolExecutor for Sync Tools

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor


def heavy_io_work(n: int) -> int:
    """I/O-bound work that benefits from thread parallelism."""
    return sum(i * i for i in range(n))


async def main() -> None:
    # ThreadExecutor wraps a concurrent.futures.Executor.
    # Use ThreadPoolExecutor here — ProcessPoolExecutor is incompatible because
    # pydantic-ai passes a contextvars.Context via run_in_executor, which cannot
    # be pickled across process boundaries (TypeError at runtime).
    thread_pool = ThreadPoolExecutor(max_workers=4)
    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[ThreadExecutor(thread_pool)],
    )

    @agent.tool_plain
    def compute_squares(n: int) -> int:
        """Sum the squares of integers from 0 to n."""
        return heavy_io_work(n)

    result = await agent.run("What is the sum of squares from 0 to 10000?")
    print(result.output)
    thread_pool.shutdown(wait=True)


asyncio.run(main())
```

---

## 7. `TenacityTransport` + `AsyncTenacityTransport` + `RetryConfig` + `wait_retry_after`

**Source**: `pydantic_ai/retries.py`

The `retries` module (new in v2.0.0; requires `pip install "pydantic-ai-slim[retries]"`) provides tenacity-backed `httpx` transport wrappers that add declarative retry logic to HTTP clients used inside pydantic-ai. `RetryConfig` is a `TypedDict` of all tenacity `@retry` decorator kwargs. `wait_retry_after` is a tenacity-compatible wait function that reads the HTTP `Retry-After` header before falling back to an exponential strategy.

```python
class RetryConfig(TypedDict, total=False):
    sleep: Callable[[int | float], None | Awaitable[None]]
    stop: StopBaseT               # e.g. stop_after_attempt(5)
    wait: WaitBaseT               # e.g. wait_exponential(multiplier=1, max=60)
    retry: SyncRetryBaseT | RetryBaseT  # e.g. retry_if_exception_type(HTTPStatusError)
    before: Callable[[RetryCallState], None | Awaitable[None]]
    after: Callable[[RetryCallState], None | Awaitable[None]]
    before_sleep: Callable[[RetryCallState], None | Awaitable[None]] | None
    reraise: bool                 # reraise last exception vs RetryError
    retry_error_cls: type[RetryError]
    retry_error_callback: Callable[[RetryCallState], Any | Awaitable[Any]] | None

class TenacityTransport(BaseTransport):
    config: RetryConfig
    wrapped: BaseTransport        # default: HTTPTransport()
    validate_response: Callable[[Response], Any] | None

class AsyncTenacityTransport(AsyncBaseTransport):
    config: RetryConfig
    wrapped: AsyncBaseTransport   # default: AsyncHTTPTransport()
    validate_response: Callable[[Response], Any] | None

def wait_retry_after(
    fallback_strategy: Callable[[RetryCallState], float] | None = None,
    max_wait: float = 300,
) -> Callable[[RetryCallState], float]: ...
# Reads Retry-After header (seconds or HTTP date) before falling back.
```

### 7.1 Async Transport with Exponential Backoff

```python
import asyncio
import httpx
from tenacity import retry_if_exception_type, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


async def main() -> None:
    transport = AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_retry_after(max_wait=120),   # honour Retry-After up to 2 min
            stop=stop_after_attempt(5),
            reraise=True,
        ),
        validate_response=lambda r: r.raise_for_status(),
    )

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://api.example.com/data")
        print(response.json())


asyncio.run(main())
```

### 7.2 Sync Transport for Custom Model Clients

```python
import httpx
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential
from pydantic_ai.retries import TenacityTransport, RetryConfig


def build_resilient_client() -> httpx.Client:
    """Build an httpx.Client with automatic retries for 5xx errors."""
    transport = TenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(4),
            reraise=True,
        ),
        validate_response=lambda r: r.raise_for_status(),
    )
    return httpx.Client(transport=transport, timeout=30.0)


client = build_resilient_client()
response = client.get("https://api.openai.com/v1/models")
print(response.status_code)
```

### 7.3 `wait_retry_after` with Custom Fallback Strategy

```python
import asyncio
import httpx
from tenacity import retry_if_exception_type, stop_after_attempt, wait_fixed
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


async def main() -> None:
    # Use wait_retry_after with a fixed 2s fallback when no Retry-After header exists.
    transport = AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type(httpx.HTTPStatusError),
            wait=wait_retry_after(
                fallback_strategy=wait_fixed(2),   # 2s when header absent
                max_wait=60,                       # cap at 60s even if header says longer
            ),
            stop=stop_after_attempt(3),
            reraise=True,
        ),
        validate_response=lambda r: r.raise_for_status() if r.status_code in (429, 503) else None,
    )

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://rate-limited-api.example.com/endpoint")
        print(response.text)


asyncio.run(main())
```

---

## 8. `DeferredLoadingToolset` — Lazy Tool Discovery via ToolSearch

**Source**: `pydantic_ai/toolsets/deferred_loading.py`

`DeferredLoadingToolset` wraps any toolset and marks its tools with `defer_loading=True` in their `ToolDefinition`. Tools so marked are hidden from the model's initial prompt; they are only revealed when the model invokes the built-in `ToolSearch` capability (itself a `DeferredCapabilityLoader`). This keeps the system prompt short for toolsets with many functions while still making all tools discoverable on demand.

```python
@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]  # auto-built in __init__
    tool_names: frozenset[str] | None = None
    # None  → mark ALL wrapped tools as deferred
    # frozenset → mark only the named subset; others stay visible immediately

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_names: frozenset[str] | None = None,
    ): ...
    # Internally sets prepare_func = _mark_deferred that does:
    # replace(td, defer_loading=True) for each tool in tool_names (or all)
```

### 8.1 Hide All Tools Until Discovered

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.capabilities import ToolSearch


async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email message."""
    return f"Email sent to {to}"


async def create_calendar_event(title: str, date: str) -> str:
    """Create a calendar event."""
    return f"Event '{title}' created for {date}"


async def main() -> None:
    # None tool_names → all tools are deferred; hidden until ToolSearch reveals them
    toolset = DeferredLoadingToolset(
        FunctionToolset([send_email, create_calendar_event]),
    )

    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[toolset],
        capabilities=[ToolSearch()],   # enables the deferred discovery mechanism
    )
    result = await agent.run(
        "Send an email to alice@example.com with subject 'Hello' and body 'Hi Alice!'"
    )
    print(result.output)


asyncio.run(main())
```

### 8.2 Partial Deferral — Keep Core Tools Visible

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.capabilities import ToolSearch


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}"


def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights."""
    return f"3 flights from {origin} to {destination} on {date}"


def book_flight(flight_id: str) -> str:
    """Book a specific flight by ID."""
    return f"Booked flight {flight_id}"


async def main() -> None:
    # Only defer the booking tool — weather and search are immediately visible
    toolset = DeferredLoadingToolset(
        FunctionToolset([get_weather, search_flights, book_flight]),
        tool_names=frozenset({'book_flight'}),   # only 'book_flight' is deferred
    )

    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[toolset],
        capabilities=[ToolSearch()],
    )
    result = await agent.run("What's the weather in Paris, and find me flights there from London on 2026-12-01")
    print(result.output)


asyncio.run(main())
```

### 8.3 Combining with ExternalToolset for Human-in-the-Loop

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.capabilities import ToolSearch


async def main() -> None:
    # External tools have no Python implementation — results come from outside the run.
    # Deferring them hides them until explicitly needed.
    external = ExternalToolset([
        ToolDefinition(
            name='approve_payment',
            description='Approve a payment after human review',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'amount': {'type': 'number'},
                    'currency': {'type': 'string'},
                },
                'required': ['amount', 'currency'],
            },
        ),
    ])

    deferred = DeferredLoadingToolset(external)

    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[deferred],
        capabilities=[ToolSearch()],
    )

    result = await agent.run("Process a payment of 500 USD")
    print(result.output)


asyncio.run(main())
```

---

## 9. `PrefectAgent` + `TaskConfig` — Prefect Durable Workflow Integration

**Source**: `pydantic_ai/durable_exec/prefect/_agent.py`, `pydantic_ai/durable_exec/prefect/_types.py`

`PrefectAgent` wraps any `AbstractAgent` to run model requests, tool calls, and MCP server interactions as Prefect tasks inside a Prefect flow. This makes the entire agent run durable — if a Prefect worker crashes mid-run, the flow resumes from the last completed task. `TaskConfig` is a `TypedDict` of Prefect `@task` decorator kwargs; `default_task_config` sets sensible defaults (cache policy, `persist_result=True`, `log_prints=False`).

```python
class PrefectAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,          # required; used as Prefect flow name prefix
        event_stream_handler: ...,
        mcp_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
        tool_task_config_by_name: dict[str, TaskConfig | None] | None = None,
        # None value for a tool name → disable task wrapping for that tool
        event_stream_handler_task_config: TaskConfig | None = None,
        prefectify_toolset_func: Callable[...] = prefectify_toolset,
        # prefectify_toolset wraps FunctionToolset → PrefectFunctionToolset
        # and MCPToolset → PrefectMCPToolset; others are passed through
    ): ...

class TaskConfig(TypedDict, total=False):
    retries: int
    retry_delay_seconds: float | list[float]
    timeout_seconds: float
    cache_policy: CachePolicy       # default: DEFAULT_PYDANTIC_AI_CACHE_POLICY
    persist_result: bool            # default: True
    result_storage: ResultStorage
    log_prints: bool                # default: False

default_task_config = TaskConfig(
    retries=0,
    retry_delay_seconds=1.0,
    persist_result=True,
    log_prints=False,
    cache_policy=DEFAULT_PYDANTIC_AI_CACHE_POLICY,
)
```

### 9.1 Basic Durable Agent Run Inside a Prefect Flow

```python
import asyncio
from prefect import flow
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent


base_agent = Agent('openai:gpt-4o-mini', name='research-agent')
durable_agent = PrefectAgent(base_agent)


@flow(name='research-flow')
async def research_flow(topic: str) -> str:
    # If this flow is interrupted (worker crash, timeout), Prefect resumes
    # from the last completed task checkpoint automatically.
    result = await durable_agent.run(f"Research the topic: {topic}")
    return result.output


async def main() -> None:
    output = await research_flow("pydantic-ai v2.0.0 durable execution")
    print(output)


asyncio.run(main())
```

### 9.2 Per-Tool Task Configuration

```python
import asyncio
from prefect import flow
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.durable_exec.prefect._types import TaskConfig
from pydantic_ai.tools import RunContext


base_agent = Agent('openai:gpt-4o-mini', name='data-pipeline-agent')


@base_agent.tool_plain
async def fetch_data(query: str) -> str:
    """Fetch data from the data warehouse."""
    return f"Data for: {query}"


@base_agent.tool_plain
async def write_report(content: str) -> str:
    """Write the final report to storage."""
    return "Report written"


durable_agent = PrefectAgent(
    base_agent,
    tool_task_config=TaskConfig(retries=3, retry_delay_seconds=5.0),
    # Override write_report to not retry (it's idempotent with cache)
    tool_task_config_by_name={
        'write_report': TaskConfig(retries=0, timeout_seconds=30.0),
    },
    model_task_config=TaskConfig(retries=2, retry_delay_seconds=2.0),
)


@flow(name='data-pipeline')
async def run_pipeline(query: str) -> str:
    result = await durable_agent.run(f"Fetch and report on: {query}")
    return result.output


asyncio.run(run_pipeline("Q1 2026 sales metrics"))
```

### 9.3 Disabling Task Wrapping for a Specific Tool

```python
import asyncio
from prefect import flow
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.durable_exec.prefect._types import TaskConfig
from pydantic_ai.tools import RunContext


base_agent = Agent('openai:gpt-4o-mini', name='mixed-agent')


@base_agent.tool_plain
def fast_lookup(key: str) -> str:
    """Look up a value in an in-memory cache (microsecond call — no task overhead needed)."""
    cache = {'greeting': 'Hello, World!', 'version': '2.0.0'}
    return cache.get(key, 'not found')


@base_agent.tool_plain
async def slow_external_call(endpoint: str) -> str:
    """Call an external API that may be slow or flaky."""
    import asyncio
    await asyncio.sleep(0.1)   # simulate latency
    return f"Response from {endpoint}"


durable_agent = PrefectAgent(
    base_agent,
    tool_task_config=TaskConfig(retries=2, persist_result=True),
    # Passing None disables Prefect task wrapping for fast_lookup
    tool_task_config_by_name={'fast_lookup': None},
)


@flow(name='mixed-flow')
async def mixed_flow(prompt: str) -> str:
    result = await durable_agent.run(prompt)
    return result.output


asyncio.run(mixed_flow("Look up 'greeting' and also call /api/status"))
```

---

## 10. `split_content_into_text_and_thinking` + `HistoryProcessor`

**Source**: `pydantic_ai/_thinking_part.py`, `pydantic_ai/_history_processor.py`

### `split_content_into_text_and_thinking`

Some LLM providers (DeepSeek, Qwen, older Ollama builds) return thinking content inside `<think>...</think>` tags embedded in the text stream rather than as a separate `ThinkingPart`. `split_content_into_text_and_thinking` parses this convention, splitting a raw string into an alternating list of `TextPart` and `ThinkingPart` objects.

```python
def split_content_into_text_and_thinking(
    content: str,
    thinking_tags: tuple[str, str],     # e.g. ('<think>', '</think>')
) -> list[ThinkingPart | TextPart]: ...

# Algorithm:
# 1. Find start_tag in content.
# 2. Everything before start_tag → TextPart.
# 3. Everything between start_tag and end_tag → ThinkingPart.
# 4. If end_tag is missing, the remaining content becomes a TextPart (start_tag dropped).
# 5. Repeat until no more start_tags remain.
# 6. Any trailing text after the last end_tag → TextPart.
```

### `HistoryProcessor`

`HistoryProcessor` is a TypeAlias for four callable forms that preprocess the message history before each model request. All four accept `list[ModelMessage]` and return a (possibly modified) `list[ModelMessage]`. The two `WithCtx` variants additionally receive the `RunContext` as first argument.

```python
HistoryProcessor = (
    _HistoryProcessorSync            # (messages) -> list[ModelMessage]
    | _HistoryProcessorAsync         # (messages) -> Awaitable[list[ModelMessage]]
    | _HistoryProcessorSyncWithCtx   # (ctx, messages) -> list[ModelMessage]
    | _HistoryProcessorAsyncWithCtx  # (ctx, messages) -> Awaitable[list[ModelMessage]]
)
```

Attach a `HistoryProcessor` to an agent via the `ProcessHistory` capability: `Agent(capabilities=[ProcessHistory(my_processor)])`. There is no `history_processors` constructor kwarg in v2.0.0.

### 10.1 Parsing Embedded Thinking Tags from DeepSeek

```python
from pydantic_ai import TextPart, ThinkingPart
from pydantic_ai._thinking_part import split_content_into_text_and_thinking


def process_deepseek_response(raw: str) -> list[TextPart | ThinkingPart]:
    """Split DeepSeek-style embedded thinking into structured parts."""
    return split_content_into_text_and_thinking(raw, thinking_tags=('<think>', '</think>'))


raw = "<think>Let me work through this step by step...</think>The answer is 42."
parts = process_deepseek_response(raw)
for part in parts:
    if isinstance(part, ThinkingPart):
        print(f"[THINKING] {part.content}")
    else:
        print(f"[TEXT] {part.content}")

# Output:
# [THINKING] Let me work through this step by step...
# [TEXT] The answer is 42.
```

### 10.2 History Processor — Truncating Long Conversations

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.process_history import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse


def keep_last_n_exchanges(n: int = 5):
    """Return a HistoryProcessor that keeps only the last n request/response pairs."""

    def processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # The processor always receives the current ModelRequest as the final item.
        # Separate it from the completed request/response pairs so it is never dropped.
        pending: ModelRequest | None = None
        history = list(messages)
        if history and isinstance(history[-1], ModelRequest):
            pending = history[-1]
            history = history[:-1]

        pairs: list[tuple[ModelRequest, ModelResponse]] = []
        for i in range(0, len(history) - 1, 2):
            if isinstance(history[i], ModelRequest) and isinstance(history[i + 1], ModelResponse):
                pairs.append((history[i], history[i + 1]))

        kept = pairs[-n:]
        result = [msg for pair in kept for msg in pair]
        if pending is not None:
            result.append(pending)
        return result

    return processor


async def main() -> None:
    # Attach via ProcessHistory capability — Agent has no history_processors kwarg.
    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[ProcessHistory(keep_last_n_exchanges(3))],
    )

    # Simulate a long conversation — only the last 3 exchanges reach each call
    result = await agent.run(
        "Remember: my name is Alice",
        message_history=[],
    )
    print(result.output)


asyncio.run(main())
```

### 10.3 Async History Processor with RunContext

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.process_history import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, TextPart
from pydantic_ai.tools import RunContext


async def inject_user_profile(
    ctx: RunContext[dict],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Inject a personalised system prompt into each request based on RunContext deps."""
    user = ctx.deps.get('user', {})
    name = user.get('name', 'user')
    role = user.get('role', 'standard')

    # Find and enrich the first system prompt part in the first request message
    enriched = list(messages)
    if enriched and isinstance(enriched[0], ModelRequest):
        prefix = SystemPromptPart(content=f"User: {name} | Role: {role}\n")
        original_parts = list(enriched[0].parts)
        enriched[0] = ModelRequest(parts=[prefix] + original_parts)
    return enriched


async def main() -> None:
    # Attach via ProcessHistory capability — Agent has no history_processors kwarg.
    agent = Agent(
        'openai:gpt-4o-mini',
        capabilities=[ProcessHistory(inject_user_profile)],
        deps_type=dict,
    )

    result = await agent.run(
        "What can I do on this platform?",
        deps={'user': {'name': 'Bob', 'role': 'admin'}},
    )
    print(result.output)


asyncio.run(main())
```

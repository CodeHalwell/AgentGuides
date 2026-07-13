---
title: "PydanticAI Class Deep Dives Vol. 35"
description: "Source-verified deep dives into 10 pydantic-ai 2.9.0 class groups: Thinking capability (portable reasoning control — ThinkingLevel union, ModelSettings.thinking bridge, effort='minimal'/'low'/'medium'/'high'/'xhigh'/True/False, always-on model silencing), WebSearch capability (native/local web search — WebSearchLocalStrategy='duckduckgo', search_context_size, user_location, blocked/allowed_domains, max_uses, _requires_native guard), WebFetch capability (native/local URL fetching — WebFetchTool bridge, enable_citations, max_content_tokens, local=True markdownify fallback, allowed/blocked_domains local enforcement), ImageGeneration capability (native/subagent image gen — NativeOrLocalTool subclass, fallback_model subagent bridge, action/background/input_fidelity/moderation/quality/size/aspect_ratio settings, _resolved_native() override merging), XSearch capability (X/Twitter native search — xAI-native, fallback_model required for non-xAI, allowed/excluded_x_handles, from/to_date, enable_image/video_understanding, XSearchSubagentTool bridge), ThreadExecutor capability (bounded thread pool — concurrent.futures.Executor, using_thread_executor context, Agent.using_thread_executor() global setter, FastAPI/long-running server pattern), WebFetchLocalTool+web_fetch_tool (SSRF-safe markdown fetcher — safe_download, markdownify HTML→MD, Accept: text/markdown negotiation, BinaryContent passthrough, max_content_length, allow_local_urls, headers override), DuckDuckGoSearchTool+duckduckgo_search_tool (local DuckDuckGo search — DDGS async-in-thread via anyio, DuckDuckGoResult TypedDict, max_results, ddgs/duckduckgo_search package support), TavilySearchTool+tavily_search_tool (Tavily search — AsyncTavilyClient, partial signature freeze for LLM schema, search_depth/topic/time_range/include_domains/exclude_domains, max_results developer-only), ExaToolset+ExaSearchTool+ExaFindSimilarTool+ExaGetContentsTool+ExaAnswerTool (Exa neural search — FunctionToolset subclass, shared AsyncExa client, search_type='auto'/'keyword'/'neural'/'fast'/'deep', max_characters token budget, find_similar/get_contents/answer capabilities). All verified against pydantic-ai 2.9.0 source installed from PyPI."
sidebar:
  label: "Class deep dives (Vol. 35)"
  order: 61
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.9.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.9.x API.
</Aside>

Ten class groups covering the real-world tool-integration layer of pydantic-ai 2.9.0: the `Thinking` reasoning-control capability, three native-or-local grounded-data capabilities (`WebSearch`, `WebFetch`, `ImageGeneration`), the `XSearch` X/Twitter capability, the `ThreadExecutor` threading control capability, and four `common_tools` wrappers for local search and fetch — DuckDuckGo, Tavily, Exa, and the built-in SSRF-safe `web_fetch_tool`.

---

## 1. `Thinking` — Portable Reasoning-Effort Control

**Source**: `pydantic_ai/capabilities/thinking.py`  
**Export**: `from pydantic_ai.capabilities import Thinking`

`Thinking` is a thin but powerful capability that injects a `thinking` key into the `ModelSettings` passed to every model request in the run. It is deliberately provider-agnostic: the unified `thinking` field in `ModelSettings` propagates to Anthropic's extended thinking, OpenAI's `reasoning_effort`, Google's `thinking_budget`, and any other provider that maps the field. Provider-specific settings (e.g. `anthropic_thinking`, `openai_reasoning_effort`) override the unified setting when both are present.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

ThinkingLevel = Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | bool

@dataclass
class Thinking(AbstractCapability[Any]):
    effort: ThinkingLevel = True
    # get_model_settings() → ModelSettings(thinking=self.effort)
```

`effort=True` enables thinking at the provider's default effort. `effort=False` disables it (silently ignored on always-on models like o3). String levels provide portable cross-provider effort laddering.

### 1.1 Enable Thinking on a Per-Agent Basis

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Attach to agent — applies to every run, all turns
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[Thinking(effort='high')],
    system_prompt='You are a careful mathematical reasoner.',
)

async def main() -> None:
    result = await agent.run('Prove that sqrt(2) is irrational.')
    print(result.output)

asyncio.run(main())
```

### 1.2 Override Effort per Run with `model_settings`

Provider-specific settings always win over `Thinking`. Use this to suppress thinking on a cheap lookup even when the agent has `Thinking` attached.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.settings import ModelSettings

reasoning_agent = Agent(
    'openai:o3',
    capabilities=[Thinking(effort='medium')],
)

async def main() -> None:
    # This run: override to 'high' via openai_reasoning_effort (takes precedence)
    deep = await reasoning_agent.run(
        'Analyse the trade-offs between B-trees and LSM-trees.',
        model_settings=ModelSettings(openai_reasoning_effort='high'),
    )
    # This run: disable thinking entirely for a simple factual lookup
    quick = await reasoning_agent.run(
        'What is the capital of France?',
        model_settings=ModelSettings(thinking=False),
    )
    print(deep.output[:200])
    print(quick.output)

asyncio.run(main())
```

### 1.3 Disable Thinking on an Always-On Model

On always-on reasoning models `effort=False` is silently ignored — it does not raise an error, so the same agent definition works across model families.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Same capability works on both always-on and opt-in reasoning models
def make_agent(model: str) -> Agent:
    return Agent(model, capabilities=[Thinking(effort=False)])

async def main() -> None:
    # gpt-5.2 respects effort=False; o3 silently ignores it
    for model in ('openai:gpt-5.2', 'openai:o3'):
        agent = make_agent(model)
        result = await agent.run('Name three sorting algorithms.')
        print(f'{model}: {result.output[:80]}')

asyncio.run(main())
```

---

## 2. `WebSearch` — Native/Local Web Search Capability

**Source**: `pydantic_ai/capabilities/web_search.py`  
**Export**: `from pydantic_ai.capabilities import WebSearch`

`WebSearch` extends `NativeOrLocalTool` to provide portable web search. When `local='duckduckgo'` (or `local=True`) is set it resolves to the `duckduckgo_search_tool()` fallback — requiring the `duckduckgo` optional group. Fields `blocked_domains`, `allowed_domains`, and `max_uses` require native support and are gated by `_requires_native()`.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

WebSearchLocalStrategy = Literal['duckduckgo']

@dataclass(init=False)
class WebSearch(NativeOrLocalTool[AgentDepsT]):
    search_context_size: Literal['low', 'medium', 'high'] | None
    user_location: WebSearchUserLocation | None
    blocked_domains: list[str] | None
    allowed_domains: list[str] | None
    max_uses: int | None

    def __init__(
        self,
        *,
        native: WebSearchTool | Callable[...] | bool = True,
        local: WebSearchLocalStrategy | Tool | Callable | bool | None = None,
        search_context_size: Literal['low', 'medium', 'high'] | None = None,
        user_location: WebSearchUserLocation | None = None,
        blocked_domains: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_uses: int | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> None: ...
```

### 2.1 Native Web Search with Context Size

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        WebSearch(
            search_context_size='high',  # more web context per search
        )
    ],
    system_prompt='Answer using up-to-date web information.',
)

async def main() -> None:
    result = await agent.run(
        'What are the latest developments in quantum computing as of 2025?'
    )
    print(result.output)

asyncio.run(main())
```

### 2.2 DuckDuckGo Fallback for Non-Native Models

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# local=True → resolves to 'duckduckgo' strategy automatically
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        WebSearch(native=False, local=True)
    ],
)

async def main() -> None:
    result = await agent.run('Find the current Python version.')
    print(result.output)

asyncio.run(main())
```

### 2.3 Domain-Gated Native Search

`blocked_domains` and `allowed_domains` require native support. Setting them auto-raises `UserError` when the model doesn't support the native search tool.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# Restrict searches to known high-quality domains
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        WebSearch(
            allowed_domains=['arxiv.org', 'nature.com', 'science.org'],
            max_uses=3,
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Summarise recent NLP research on chain-of-thought prompting.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 3. `WebFetch` — Native/Local URL Fetching Capability

**Source**: `pydantic_ai/capabilities/web_fetch.py`  
**Export**: `from pydantic_ai.capabilities import WebFetch`

`WebFetch` follows the same `NativeOrLocalTool` pattern as `WebSearch`. `local=True` resolves to `web_fetch_tool()` backed by SSRF-protected `httpx` + `markdownify`. `allowed_domains` and `blocked_domains` are enforced locally when the native tool is not used. `max_uses`, `enable_citations`, and `max_content_tokens` require native support.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

@dataclass(init=False)
class WebFetch(NativeOrLocalTool[AgentDepsT]):
    allowed_domains: list[str] | None
    blocked_domains: list[str] | None
    max_uses: int | None
    enable_citations: bool | None
    max_content_tokens: int | None

    def __init__(
        self,
        *,
        native: WebFetchTool | Callable[...] | bool = True,
        local: Tool | Callable | bool | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_uses: int | None = None,
        enable_citations: bool | None = None,
        max_content_tokens: int | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> None: ...
```

### 3.1 Native Web Fetch with Citations

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        WebFetch(enable_citations=True, max_content_tokens=8192)
    ],
)

async def main() -> None:
    result = await agent.run(
        'Fetch https://peps.python.org/pep-0695/ and summarise the new type alias syntax.'
    )
    print(result.output)

asyncio.run(main())
```

### 3.2 Local Markdownify Fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

# pip install "pydantic-ai-slim[web-fetch]" for markdownify
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        WebFetch(native=False, local=True)
    ],
)

async def main() -> None:
    result = await agent.run(
        'Fetch https://docs.pydantic.dev/latest/ and list the main sections.'
    )
    print(result.output)

asyncio.run(main())
```

### 3.3 Domain-Locked Fetch (Local Enforcement)

`allowed_domains` is enforced by the local tool even when native is unavailable, making it safe to use without a model that supports the native `WebFetchTool`.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        WebFetch(
            native=False,
            local=True,
            allowed_domains=['github.com', 'raw.githubusercontent.com'],
        )
    ],
)

async def main() -> None:
    # Will succeed — domain is allowed
    result = await agent.run(
        'Fetch https://github.com/pydantic/pydantic-ai and get the star count from the README.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 4. `ImageGeneration` — Native/Subagent Image Generation Capability

**Source**: `pydantic_ai/capabilities/image_generation.py`  
**Export**: `from pydantic_ai.capabilities import ImageGeneration`

`ImageGeneration` extends `NativeOrLocalTool` for image generation. When the current model supports image generation natively it uses `ImageGenerationTool`; when it does not and `fallback_model` is set, it delegates to an `ImageGenerationSubagentTool` that spins up a sub-`Agent` on the fallback model. Capability-level fields (`quality`, `size`, `output_format`, etc.) override any settings on a custom `native` instance via `_resolved_native()`.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

@dataclass(init=False)
class ImageGeneration(NativeOrLocalTool[AgentDepsT]):
    fallback_model: ImageGenerationFallbackModel

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

    def __init__(
        self, *, native=True, local=None, fallback_model=None,
        action=None, background=None, input_fidelity=None,
        moderation=None, image_model=None, output_compression=None,
        output_format=None, quality=None, size=None, aspect_ratio=None,
        id=None, defer_loading=False, description=None,
    ) -> None: ...
```

### 4.1 Native Image Generation on OpenAI Responses

```python
import asyncio
import base64
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'openai-responses:gpt-5.4',
    capabilities=[
        ImageGeneration(
            quality='high',
            size='1024x1024',
            output_format='png',
        )
    ],
    output_type=bytes,
)

async def main() -> None:
    result = await agent.run(
        'Generate a photorealistic image of a red fox sitting in a snowy forest.'
    )
    Path('fox.png').write_bytes(result.output)
    print('Image saved to fox.png')

asyncio.run(main())
```

### 4.2 Transparent Background for UI Assets

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'openai-responses:gpt-5.4',
    capabilities=[
        ImageGeneration(
            output_format='png',    # transparent requires png or webp
            background='transparent',
            size='1024x1024',
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Create a transparent-background logo for a coffee shop called "Morning Brew".'
    )
    print(type(result.output))

asyncio.run(main())
```

### 4.3 Cross-Provider Fallback

On a model that does not support native image generation, `fallback_model` activates an `ImageGenerationSubagentTool` that runs the specified model in a sub-agent.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

# Outer agent doesn't generate images; subagent does
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        ImageGeneration(
            native=False,
            fallback_model='openai-responses:gpt-5.4',
            quality='medium',
            size='1024x1024',
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Draw a minimalist line-art cat wearing a wizard hat.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 5. `XSearch` — X/Twitter Search Capability

**Source**: `pydantic_ai/capabilities/x_search.py`  
**Export**: `from pydantic_ai.capabilities import XSearch`

`XSearch` extends `NativeOrLocalTool` for X (Twitter) search. On xAI models it maps directly to the `XSearchTool` native tool. On any other model `fallback_model` (which must be an xAI model) is **required** — without it, using `XSearch` on a non-xAI model raises a `UserError`. Handle constraints (`allowed_x_handles`, `excluded_x_handles`) require native support unless `fallback_model` is set (the subagent also runs the native tool).

```python
# Key signature verified from source (pydantic-ai 2.9.0):

@dataclass(init=False)
class XSearch(NativeOrLocalTool[AgentDepsT]):
    fallback_model: XSearchFallbackModel        # required for non-xAI models
    allowed_x_handles: list[str] | None         # max 20
    excluded_x_handles: list[str] | None        # max 20
    from_date: datetime | None
    to_date: datetime | None
    enable_image_understanding: bool | None
    enable_video_understanding: bool | None
    include_output: bool | None

    def __init__(
        self,
        *,
        native=True, local=None, fallback_model=None,
        allowed_x_handles=None, excluded_x_handles=None,
        from_date=None, to_date=None,
        enable_image_understanding=None, enable_video_understanding=None,
        include_output=None,
        id=None, description=None, defer_loading=False,
    ) -> None: ...
```

### 5.1 Native X Search on an xAI Model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'xai:grok-4.3',
    capabilities=[XSearch()],  # native=True is default
    system_prompt='Use real-time X/Twitter data to answer questions.',
)

async def main() -> None:
    result = await agent.run(
        'What are people saying about the latest Python release on X?'
    )
    print(result.output)

asyncio.run(main())
```

### 5.2 Filtered Search by Handle and Date Range

```python
import asyncio
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'xai:grok-4.3',
    capabilities=[
        XSearch(
            allowed_x_handles=['pydantic', 'gvanrossum', 'tiangolo'],
            from_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            to_date=datetime(2025, 6, 30, tzinfo=timezone.utc),
            enable_image_understanding=True,
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'What did the Python and Pydantic teams share about new features in H1 2025?'
    )
    print(result.output)

asyncio.run(main())
```

### 5.3 Fallback to xAI Subagent from Non-xAI Model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

# Outer agent is GPT; XSearch delegates to a Grok subagent
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        XSearch(
            native=False,
            fallback_model='xai:grok-4.3',
            enable_video_understanding=True,
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Find viral posts about AI coding assistants this week.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 6. `ThreadExecutor` — Bounded Thread Pool for Sync Tools

**Source**: `pydantic_ai/capabilities/thread_executor.py`  
**Export**: `from pydantic_ai.capabilities import ThreadExecutor`

By default pydantic-ai runs sync tool functions via `anyio.to_thread.run_sync`, which spawns ephemeral threads. In long-lived servers (FastAPI, Starlette) this can exhaust the OS thread limit under load. `ThreadExecutor` plugs in a `concurrent.futures.Executor` — typically a `ThreadPoolExecutor` — that bounds the thread count and reuses threads. For global application-level control, `Agent.using_thread_executor()` is a context manager that sets the executor for all agents in that scope.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

@dataclass
class ThreadExecutor(AbstractCapability[Any]):
    executor: Executor

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None   # not serializable — Executor holds OS resources

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        with _utils.using_thread_executor(self.executor):
            return await handler()
```

### 6.1 Per-Agent Bounded Thread Pool

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

# Create once and share across requests
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='pai-worker')

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[ThreadExecutor(_executor)],
)

def sync_tool(query: str) -> str:
    import time; time.sleep(0.1)  # simulate blocking I/O
    return f'Result for: {query}'

agent.tool_plain(sync_tool)

async def main() -> None:
    result = await agent.run('Call the sync tool with query "hello".')
    print(result.output)

asyncio.run(main())
```

### 6.2 Global Executor via `Agent.using_thread_executor()`

`using_thread_executor()` is the class method variant — it sets the executor for all agents in the `with` block, useful for test isolation or request-scoped overrides.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai import Agent

agent_a = Agent('openai:gpt-5.2')
agent_b = Agent('anthropic:claude-haiku-4-5')

async def handle_request() -> None:
    # All sync callbacks in this scope share one bounded pool
    with Agent.using_thread_executor(
        ThreadPoolExecutor(max_workers=8, thread_name_prefix='req-worker')
    ):
        result_a = await agent_a.run('List 3 animals.')
        result_b = await agent_b.run('List 3 fruits.')
    print(result_a.output, result_b.output)

asyncio.run(handle_request())
```

### 6.3 FastAPI Lifespan Integration

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

_executor: ThreadPoolExecutor

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _executor
    _executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix='api-agent')
    yield
    _executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

@app.get('/ask')
async def ask(q: str) -> dict:
    agent = Agent(
        'openai:gpt-5.2',
        capabilities=[ThreadExecutor(_executor)],
    )
    result = await agent.run(q)
    return {'answer': result.output}
```

---

## 7. `WebFetchLocalTool` + `web_fetch_tool` — SSRF-Safe Local Markdown Fetcher

**Source**: `pydantic_ai/common_tools/web_fetch.py`  
**Export**: `from pydantic_ai.common_tools.web_fetch import web_fetch_tool, WebFetchResult`

`web_fetch_tool()` creates a `Tool` that fetches any URL and returns either a `WebFetchResult` (text/HTML/JSON → markdown) or a `BinaryContent` object (PDF, images). It uses `pydantic_ai._ssrf.safe_download` internally for SSRF protection, `markdownify` for HTML→markdown conversion, and proactively sends `Accept: text/markdown` to let Markdown-aware servers (Cloudflare, Vercel, Mintlify) skip the HTML serialisation step. Requires `pip install "pydantic-ai-slim[web-fetch]"`.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

@dataclass
class WebFetchLocalTool:
    max_content_length: int | None        # default 50_000 chars (~12 500 tokens)
    allow_local_urls: bool                 # default False (SSRF protection)
    timeout: int                           # default 30 s
    allowed_domains: list[str] | None
    blocked_domains: list[str] | None
    headers: dict[str, str] | None

    async def __call__(self, url: str) -> WebFetchResult | BinaryContent: ...

def web_fetch_tool(
    *,
    max_content_length: int | None = 50_000,
    allow_local_urls: bool = False,
    timeout: int = 30,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> Tool[Any]: ...
```

### 7.1 Basic Local Web Fetch

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'openai:gpt-5.2',
    tools=[web_fetch_tool()],
)

async def main() -> None:
    result = await agent.run(
        'Fetch https://httpbin.org/json and explain the JSON structure.'
    )
    print(result.output)

asyncio.run(main())
```

### 7.2 Custom Headers and Token Budget

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'openai:gpt-5.2',
    tools=[
        web_fetch_tool(
            max_content_length=20_000,   # ~5 000 tokens
            timeout=15,
            headers={
                'User-Agent': 'MyAgent/1.0',
                'Accept-Language': 'en-US,en;q=0.9',
            },
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Summarise the README at https://raw.githubusercontent.com/pydantic/pydantic-ai/main/README.md'
    )
    print(result.output)

asyncio.run(main())
```

### 7.3 Domain-Locked Fetch with Binary Passthrough

When a URL returns binary content (PDF, image), the tool returns a `BinaryContent` object that the model can inspect natively.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool, WebFetchResult
from pydantic_ai.messages import BinaryContent

agent = Agent(
    'openai:gpt-5.2',
    tools=[
        web_fetch_tool(
            allowed_domains=['arxiv.org'],
            max_content_length=None,  # no limit for research papers
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Fetch https://arxiv.org/abs/2307.09288 and list the main contributions.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 8. `DuckDuckGoSearchTool` + `duckduckgo_search_tool` — Local DuckDuckGo Search

**Source**: `pydantic_ai/common_tools/duckduckgo.py`  
**Export**: `from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool`

`duckduckgo_search_tool()` wraps the `DDGS` client (from the `ddgs` package, with `duckduckgo_search` as legacy fallback) in a `Tool` that runs DuckDuckGo text searches asynchronously via `anyio.to_thread.run_sync`. Results are validated as `list[DuckDuckGoResult]` (TypedDict with `title`, `href`, `body`). Requires `pip install "pydantic-ai-slim[duckduckgo]"`.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

class DuckDuckGoResult(TypedDict):
    title: str
    href: str
    body: str

@dataclass
class DuckDuckGoSearchTool:
    client: DDGS
    max_results: int | None

    async def __call__(self, query: str) -> list[DuckDuckGoResult]: ...

def duckduckgo_search_tool(
    duckduckgo_client: DDGS | None = None,
    max_results: int | None = None,
) -> Tool[Any]: ...
```

### 8.1 Basic DuckDuckGo Search Agent

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-5.2',
    tools=[duckduckgo_search_tool(max_results=5)],
    system_prompt='Search the web and provide accurate, cited answers.',
)

async def main() -> None:
    result = await agent.run('What is the latest stable release of FastAPI?')
    print(result.output)

asyncio.run(main())
```

### 8.2 Shared DDGS Client Across Multiple Tool Instances

```python
import asyncio
from ddgs.ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# Share one client to reuse the HTTP session
shared_client = DDGS()

research_agent = Agent(
    'openai:gpt-5.2',
    tools=[duckduckgo_search_tool(duckduckgo_client=shared_client, max_results=10)],
)

summary_agent = Agent(
    'anthropic:claude-haiku-4-5',
    tools=[duckduckgo_search_tool(duckduckgo_client=shared_client, max_results=3)],
)

async def main() -> None:
    result = await research_agent.run(
        'Research pydantic-ai architectural decisions and key design patterns.'
    )
    print(result.output[:500])

asyncio.run(main())
```

### 8.3 WebSearch Capability with DuckDuckGo as the Fallback

The `WebSearch` capability's `local='duckduckgo'` path calls `duckduckgo_search_tool()` internally, providing a higher-level API that transparently upgrades to native search on capable models.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# Upgrade path: use native search if available, otherwise DuckDuckGo
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[WebSearch(local='duckduckgo')],
)

async def main() -> None:
    # On gpt-5.2: native search; swap to claude-haiku-4-5 → DuckDuckGo fallback
    result = await agent.run('Find the homepage for the pydantic library.')
    print(result.output)

asyncio.run(main())
```

---

## 9. `TavilySearchTool` + `tavily_search_tool` — Structured AI-Powered Search

**Source**: `pydantic_ai/common_tools/tavily.py`  
**Export**: `from pydantic_ai.common_tools.tavily import tavily_search_tool`

`tavily_search_tool()` wraps Tavily's `AsyncTavilyClient` in a `Tool`. Its key design: any of `search_depth`, `topic`, `time_range`, `include_domains`, `exclude_domains` can be **fixed** at factory time — when fixed they are stripped from the LLM's tool schema via an explicit `__signature__` override using `functools.partial`. `max_results` is always developer-controlled and never exposed in the schema. Requires `pip install "pydantic-ai-slim[tavily]"`.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

class TavilySearchResult(TypedDict):
    title: str; url: str; content: str; score: float

@dataclass
class TavilySearchTool:
    client: AsyncTavilyClient
    max_results: int | None = None

    async def __call__(
        self,
        query: str,
        search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = 'basic',
        topic: Literal['general', 'news', 'finance'] = 'general',
        time_range: Literal['day', 'week', 'month', 'year'] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[TavilySearchResult]: ...

def tavily_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncTavilyClient | None = None,
    max_results: int | None = None,
    # Any of the following, when provided, are fixed and hidden from the LLM schema:
    search_depth=_UNSET, topic=_UNSET,
    time_range=_UNSET, include_domains=_UNSET, exclude_domains=_UNSET,
) -> Tool[Any]: ...
```

### 9.1 Basic Tavily Search with LLM-Controlled Depth

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

agent = Agent(
    'openai:gpt-5.2',
    tools=[tavily_search_tool(api_key=os.environ['TAVILY_API_KEY'], max_results=5)],
)

async def main() -> None:
    # LLM can choose search_depth and topic per call
    result = await agent.run(
        'Find recent news about AI safety regulations in the EU.'
    )
    print(result.output)

asyncio.run(main())
```

### 9.2 Fixed Developer-Controlled Parameters

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

# Fix search depth to 'advanced' and topic to 'finance' — hidden from LLM
financial_agent = Agent(
    'openai:gpt-5.2',
    tools=[
        tavily_search_tool(
            api_key=os.environ['TAVILY_API_KEY'],
            max_results=8,
            search_depth='advanced',
            topic='finance',
        )
    ],
    system_prompt='You are a financial research assistant.',
)

async def main() -> None:
    result = await financial_agent.run(
        'What are analysts saying about NVIDIA stock this week?'
    )
    print(result.output)

asyncio.run(main())
```

### 9.3 Domain-Filtered News Search

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

agent = Agent(
    'openai:gpt-5.2',
    tools=[
        tavily_search_tool(
            api_key=os.environ['TAVILY_API_KEY'],
            max_results=10,
            time_range='week',
            include_domains=['techcrunch.com', 'wired.com', 'theverge.com'],
            topic='news',
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Summarise the top AI stories from tech publications this week.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 10. `ExaToolset` + Exa Tool Family — Neural Web Search Suite

**Source**: `pydantic_ai/common_tools/exa.py`  
**Export**: `from pydantic_ai.common_tools.exa import ExaToolset, exa_search_tool, exa_find_similar_tool, exa_get_contents_tool, exa_answer_tool`

The Exa integration provides four individual tools and one `FunctionToolset` subclass that shares a single `AsyncExa` client. `ExaSearchTool` supports five search types (`auto`/`keyword`/`neural`/`fast`/`deep`), `ExaFindSimilarTool` finds similar pages by URL, `ExaGetContentsTool` batch-fetches full content, and `ExaAnswerTool` returns an AI-generated answer with citations. `ExaToolset` composes all four, gating each with `include_*` flags. Requires `pip install "pydantic-ai-slim[exa]"`.

```python
# Key signature verified from source (pydantic-ai 2.9.0):

class ExaSearchResult(TypedDict):
    title: str; url: str; published_date: str | None
    author: str | None; text: str

class ExaAnswerResult(TypedDict):
    answer: str; citations: list[dict[str, Any]]

@dataclass
class ExaSearchTool:
    client: AsyncExa; num_results: int; max_characters: int | None
    async def __call__(
        self, query: str,
        search_type: Literal['auto','keyword','neural','fast','deep'] = 'auto',
    ) -> list[ExaSearchResult]: ...

class ExaToolset(FunctionToolset):
    def __init__(
        self, api_key: str, *,
        num_results: int = 5,
        max_characters: int | None = None,
        include_search: bool = True,
        include_find_similar: bool = True,
        include_get_contents: bool = True,
        include_answer: bool = True,
        id: str | None = None,
    ): ...
```

### 10.1 Full Exa Toolset with Shared Client

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset

agent = Agent(
    'openai:gpt-5.2',
    toolsets=[
        ExaToolset(
            api_key=os.environ['EXA_API_KEY'],
            num_results=8,
            max_characters=2000,
        )
    ],
    system_prompt='Use Exa to research questions thoroughly, citing your sources.',
)

async def main() -> None:
    result = await agent.run(
        'What are the most authoritative sources on retrieval-augmented generation?'
    )
    print(result.output)

asyncio.run(main())
```

### 10.2 Search + Find Similar Workflow

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset

agent = Agent(
    'openai:gpt-5.2',
    toolsets=[
        ExaToolset(
            api_key=os.environ['EXA_API_KEY'],
            include_answer=False,   # skip answer tool for this agent
            num_results=5,
        )
    ],
)

async def main() -> None:
    result = await agent.run(
        'Find the pydantic-ai documentation homepage, '
        'then find 5 similar sites to it.'
    )
    print(result.output)

asyncio.run(main())
```

### 10.3 AI-Powered Answer with Citations via Individual Tools

Use `exa_answer_tool()` on its own when you only need the answer capability and want to keep the other tools out of the LLM's schema.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import exa_answer_tool

answer_agent = Agent(
    'openai:gpt-5.2',
    tools=[exa_answer_tool(api_key=os.environ['EXA_API_KEY'])],
    system_prompt=(
        'You answer questions using Exa. '
        'Always include at least one citation URL in your response.'
    ),
)

async def main() -> None:
    result = await answer_agent.run(
        'What is the difference between RLHF and DPO for LLM alignment?'
    )
    print(result.output)

asyncio.run(main())
```

---

## Summary

| Class / Factory | Source module | Key pattern |
|---|---|---|
| `Thinking` | `capabilities/thinking.py` | `ModelSettings(thinking=effort)` — portable reasoning effort |
| `WebSearch` | `capabilities/web_search.py` | `NativeOrLocalTool` — DuckDuckGo fallback, domain gating |
| `WebFetch` | `capabilities/web_fetch.py` | `NativeOrLocalTool` — markdownify fallback, citation control |
| `ImageGeneration` | `capabilities/image_generation.py` | `NativeOrLocalTool` — subagent fallback, 10 image settings |
| `XSearch` | `capabilities/x_search.py` | `NativeOrLocalTool` — xAI native, fallback model required on others |
| `ThreadExecutor` | `capabilities/thread_executor.py` | Bounded `Executor` for sync tools in long-running servers |
| `WebFetchLocalTool` / `web_fetch_tool` | `common_tools/web_fetch.py` | SSRF-safe httpx + markdownify, `BinaryContent` passthrough |
| `DuckDuckGoSearchTool` / `duckduckgo_search_tool` | `common_tools/duckduckgo.py` | DDGS async-in-thread, `DuckDuckGoResult` TypedDict |
| `TavilySearchTool` / `tavily_search_tool` | `common_tools/tavily.py` | Partial-signature schema freeze, `AsyncTavilyClient` |
| `ExaToolset` + family | `common_tools/exa.py` | `FunctionToolset` subclass, shared `AsyncExa`, 4 tools |

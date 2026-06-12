---
title: "PydanticAI — Class Deep Dives Vol. 13"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.107.0: Capability (convenience bundle), MCP capability (auto native/local), WebSearch capability (NativeOrLocalTool), WebFetch capability (NativeOrLocalTool), XSearch capability (fallback model), Instrumentation capability (OTel/Logfire), HandleDeferredToolCalls (inline deferred resolution), ProcessEventStream (observer/processor forms), WebFetchTool+XSearchTool+ImageGenerationTool (native tool dataclasses), ToolSearch capability (strategy options + cache-compatible discovery). All verified against pydantic-ai 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 13)"
  order: 39
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups spanning the capabilities subsystem added in 1.107.0: `Capability` (the general-purpose convenience bundle for tools + instructions); `MCP` (auto-selects native or local MCP toolset by URL); `WebSearch` (NativeOrLocalTool abstraction for web search); `WebFetch` (NativeOrLocalTool abstraction for URL fetching); `XSearch` (Twitter/X search with fallback model support); `Instrumentation` (OpenTelemetry / Logfire tracing as a capability); `HandleDeferredToolCalls` (inline deferred-tool resolution without breaking the run); `ProcessEventStream` (observer and processor wrappers for `AgentStreamEvent`); the three native tool dataclasses `WebFetchTool`, `XSearchTool`, and `ImageGenerationTool`; and `ToolSearch` (vector-or-keyword tool discovery for large toolsets).

---

## 1. `Capability` — Convenience Bundle

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import Capability`

`Capability` is a reusable bundle that combines static or dynamic instructions, a set of tools, and optionally one or more nested toolsets. It implements `AbstractCapability` so it can be passed directly to `Agent(capabilities=[...])` or injected at run time via `agent.run(..., capabilities=[...])`.

Unlike registering tools directly on an agent, a `Capability` instance can be shared across multiple agents, lazily loaded with `defer_loading=True`, and given a callable `description` that lets a deferred-tool discovery system decide whether to activate it.

### Constructor

```python
def __init__(
    self,
    *,
    instructions: AgentInstructions[AgentDepsT] | None = None,
    toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
    id: str | None = None,
    description: CapabilityDescription[AgentDepsT] | None = None,
    defer_loading: bool = False,
) -> None: ...
```

Key points:
- `instructions` — a string, a callable returning a string, or a full `AgentInstructions` object. Appended to the system prompt when the capability is active.
- `tools` — plain functions or `Tool` objects; identical to registering tools on the agent directly.
- `toolsets` — nested `AgentToolset` instances (e.g. an `MCPToolset`) merged into the run.
- `id` — stable identifier used by deferred-loading and `ToolSearch`.
- `description` — static string or `Callable[[RunContext], str]` shown to the discovery LLM.
- `defer_loading` — when `True` the capability is not included in the run automatically; a `ToolSearch` capability must select it first.

### Decorator syntax

`Capability` exposes three class decorators that create and attach helpers inline:

```python
cap = Capability(id='my-cap')

@cap.tool
def my_tool(ctx: RunContext[None], x: int) -> int: ...

@cap.tool_plain
def my_plain_tool(x: int) -> str: ...

@cap.instructions
def my_instructions(ctx: RunContext[None]) -> str: ...
```

### Example 1: minimal bundle — 2 tools + static instructions

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


def word_count(ctx: RunContext[None], text: str) -> int:
    """Return the number of words in text."""
    return len(text.split())


def reverse_text(ctx: RunContext[None], text: str) -> str:
    """Reverse the characters in text."""
    return text[::-1]


text_cap = Capability(
    instructions='You are a text analysis assistant. Use the available tools to process text.',
    tools=[word_count, reverse_text],
    id='text-tools',
)

agent = Agent(TestModel(), capabilities=[text_cap])


async def main() -> None:
    result = await agent.run('How many words are in "hello world"?')
    print(result.output)


asyncio.run(main())
```

### Example 2: decorator pattern — `@cap.tool`, `@cap.tool_plain`, `@cap.instructions`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

math_cap = Capability(id='math-tools')


@math_cap.instructions
def math_instructions(ctx: RunContext[None]) -> str:
    return 'You are a math assistant. Use tools to compute results precisely.'


@math_cap.tool
def add(ctx: RunContext[None], a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@math_cap.tool_plain
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = Agent(TestModel(), capabilities=[math_cap])


async def main() -> None:
    result = await agent.run('What is 3.5 plus 1.5?')
    print(result.output)


asyncio.run(main())
```

### Example 3: callable description for deferred capability routing

When `defer_loading=True`, a `ToolSearch` capability needs to decide whether to activate this capability. The `description` callable receives the current `RunContext` and can return context-sensitive text.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


def _finance_description(ctx: RunContext[None]) -> str:
    return (
        'Finance tools: look up stock prices, compute portfolio returns, '
        'and fetch earnings calendar data. Use when the user asks about stocks, '
        'investments, or financial markets.'
    )


def get_stock_price(ctx: RunContext[None], ticker: str) -> float:
    """Return the latest closing price for a ticker symbol."""
    # Placeholder — real impl would call a market data API
    return 150.0


finance_cap = Capability(
    description=_finance_description,
    tools=[get_stock_price],
    id='finance-tools',
    defer_loading=True,
)

agent = Agent(
    TestModel(),
    capabilities=[
        ToolSearch(),      # discovers deferred capabilities
        finance_cap,
    ],
)


async def main() -> None:
    result = await agent.run('What is the price of AAPL?')
    print(result.output)


asyncio.run(main())
```

### Example 4: `defer_loading=True` — lazy capability loading

Deferred capabilities are not wired into the agent until explicitly selected. This reduces prompt length for agents with many optional capabilities.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


def translate_text(ctx: RunContext[None], text: str, target_lang: str) -> str:
    """Translate text into the specified language."""
    # Placeholder translation
    return f'[{target_lang}] {text}'


translation_cap = Capability(
    description='Translation tools for converting text between languages.',
    tools=[translate_text],
    id='translation',
    defer_loading=True,
)

summarisation_cap = Capability(
    description='Summarisation tools for condensing long documents.',
    tools=[],   # tools added via @cap.tool decorators elsewhere
    id='summarisation',
    defer_loading=True,
)

agent = Agent(
    TestModel(),
    capabilities=[
        ToolSearch(max_results=3),   # picks the best-matching deferred cap
        translation_cap,
        summarisation_cap,
    ],
)


async def main() -> None:
    result = await agent.run('Please translate "Good morning" into Spanish.')
    print(result.output)


asyncio.run(main())
```

### Example 5: shared `Capability` instance across multiple agents

A single `Capability` object is stateless and safe to share between agents, enabling DRY capability definitions.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


def lookup_user(ctx: RunContext[None], user_id: str) -> dict:
    """Return basic user profile."""
    return {'id': user_id, 'name': 'Alice', 'plan': 'pro'}


def list_users(ctx: RunContext[None]) -> list[str]:
    """Return all user IDs."""
    return ['u1', 'u2', 'u3']


# Shared — instantiated once, used by multiple agents
user_tools_cap = Capability(
    instructions='You have access to user management tools.',
    tools=[lookup_user, list_users],
    id='user-tools',
)

support_agent = Agent(TestModel(), capabilities=[user_tools_cap])
admin_agent = Agent(TestModel(), capabilities=[user_tools_cap])


async def main() -> None:
    r1 = await support_agent.run('Look up user u1.')
    r2 = await admin_agent.run('List all users.')
    print(r1.output)
    print(r2.output)


asyncio.run(main())
```

---

## 2. `MCP` Capability — Auto Native/Local MCP

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import MCP`

`MCP` is a convenience capability that wraps an MCP server URL and automatically selects either a native tool (`MCPServerTool`) or a local toolset (`MCPToolset`, `FastMCPToolset`, etc.) depending on which the current model supports. This eliminates the boilerplate of manually wiring `MCPToolset` when most models default to the local streaming-HTTP approach.

### Constructor

```python
def __init__(
    self,
    url: str,
    *,
    native: MCPServerTool | Callable | bool | None = None,
    local: MCPToolsetClient | MCPToolset | MCPServer | FastMCPToolset | Callable | bool | None = None,
    id: str | None = None,
    authorization_token: str | None = None,
    headers: dict[str, str] | None = None,
    allowed_tools: list[str] | None = None,
    description: str | None = None,
    defer_loading: bool = False,
) -> None: ...
```

Key points:
- `url` — the base URL of the MCP server (e.g. `https://mcp.example.com`).
- `native` — pass `True` to force native MCP (requires a model that supports it), `False` to disable native, or a pre-built `MCPServerTool` instance.
- `local` — pass `True` to force local MCPToolset, `False` to disable, or a pre-built toolset instance.
- When both are `None`, the capability auto-selects based on model support at run time.
- `authorization_token` — passed as the `Authorization: Bearer <token>` header.
- `allowed_tools` — whitelist of tool names exposed to the agent.

### Example 1: minimal with `native=True`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel

mcp_cap = MCP('https://mcp.example.com', native=True)

agent = Agent(TestModel(), capabilities=[mcp_cap])


async def main() -> None:
    result = await agent.run('List available resources from the MCP server.')
    print(result.output)


asyncio.run(main())
```

### Example 2: `native=False, local=True` — force local MCPToolset

Useful when your deployment environment cannot reach the MCP server via native tool APIs, or when you need to test locally with a streaming-HTTP connection.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel

# Force local MCPToolset — creates an MCPToolset(url=...) internally
local_mcp = MCP(
    'http://localhost:8080',
    native=False,
    local=True,
    id='local-mcp',
)

agent = Agent(TestModel(), capabilities=[local_mcp])


async def main() -> None:
    result = await agent.run('Fetch the status from the local MCP server.')
    print(result.output)


asyncio.run(main())
```

### Example 3: `authorization_token` + `allowed_tools` filtering

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel

secure_mcp = MCP(
    'https://api.example.com/mcp',
    authorization_token=os.environ.get('MCP_TOKEN', 'test-token'),
    # Only expose read-only tools to this agent
    allowed_tools=['get_document', 'search_documents', 'list_collections'],
    id='docs-mcp',
)

agent = Agent(TestModel(), capabilities=[secure_mcp])


async def main() -> None:
    result = await agent.run('Search for documents about PydanticAI.')
    print(result.output)


asyncio.run(main())
```

### Example 4: deferred MCP capability with description

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, ToolSearch
from pydantic_ai.models.test import TestModel

analytics_mcp = MCP(
    'https://analytics.internal/mcp',
    description=(
        'Analytics MCP server: run SQL queries, fetch dashboards, and '
        'export reports. Use when the user asks about metrics, KPIs, or data analysis.'
    ),
    defer_loading=True,
    id='analytics-mcp',
)

agent = Agent(
    TestModel(),
    capabilities=[
        ToolSearch(),
        analytics_mcp,
    ],
)


async def main() -> None:
    result = await agent.run("Show me last week's active user count.")
    print(result.output)


asyncio.run(main())
```

### Example 5: migration from explicit `MCPToolset` to `MCP()` capability

Before 1.107.0, you would wire an MCPToolset explicitly. The `MCP` capability is a drop-in replacement that additionally supports native tools.

```python
# Before (still valid but more verbose):
# from pydantic_ai.mcp import MCPToolset
# toolset = MCPToolset(url='https://mcp.example.com', headers={'Authorization': 'Bearer token'})
# agent = Agent('openai:gpt-4o', toolsets=[toolset])

# After — equivalent, plus auto native/local selection:
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        MCP(
            'https://mcp.example.com',
            headers={'X-Tenant-ID': 'acme'},
            allowed_tools=['read_file', 'write_file', 'list_files'],
            id='file-mcp',
        ),
    ],
)


async def main() -> None:
    result = await agent.run('List files in the /docs folder.')
    print(result.output)


asyncio.run(main())
```

---

## 3. `WebSearch` Capability

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import WebSearch`

`WebSearch` wraps web-search functionality as a `NativeOrLocalTool` capability. When the running model supports a native web-search tool (e.g. OpenAI's `web_search_preview` or Anthropic's search), it configures the native tool. Otherwise it falls back to a local tool — DuckDuckGo, a custom callable, or any `Tool` object.

### Constructor

```python
def __init__(
    self,
    *,
    native: WebSearchTool | Callable | bool = True,
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

Key points:
- `native=True` (default) — prefer the model's built-in web-search if available.
- `local` — fallback when native is unavailable. Use `'duckduckgo'`, `'tavily'`, `'exa'`, or any callable.
- `search_context_size` — hint to the model for how much search context to retrieve.
- `user_location` — `WebSearchUserLocation` dataclass with `country`, `city`, `region`, `timezone` fields.
- `blocked_domains` / `allowed_domains` — domain filter lists.
- `max_uses` — cap total search calls per agent run.

### Example 1: simple web search with `search_context_size='high'`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        WebSearch(search_context_size='high'),
    ],
)


async def main() -> None:
    result = await agent.run('What are the latest developments in quantum computing?')
    print(result.output)


asyncio.run(main())
```

### Example 2: geo-targeted search with `WebSearchUserLocation`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.capabilities.web_search import WebSearchUserLocation
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        WebSearch(
            search_context_size='medium',
            user_location=WebSearchUserLocation(
                country='GB',
                city='London',
                region='England',
                timezone='Europe/London',
            ),
        ),
    ],
)


async def main() -> None:
    result = await agent.run('What are the current weather conditions in my city?')
    print(result.output)


asyncio.run(main())
```

### Example 3: domain allow-list for restricted search

Useful for enterprise deployments where the agent should only retrieve information from approved sources.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel

internal_search = WebSearch(
    allowed_domains=[
        'docs.internal.example.com',
        'wiki.internal.example.com',
        'confluence.example.com',
    ],
    max_uses=5,
    id='internal-search',
)

agent = Agent(TestModel(), capabilities=[internal_search])


async def main() -> None:
    result = await agent.run('Find the deployment runbook for the payments service.')
    print(result.output)


asyncio.run(main())
```

### Example 4: force local DuckDuckGo with `native=False, local='duckduckgo'`

Forces the local DuckDuckGo implementation regardless of model support. Useful for testing without model API calls or in environments where native search is unavailable.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        WebSearch(
            native=False,
            local='duckduckgo',
            search_context_size='low',
        ),
    ],
)


async def main() -> None:
    result = await agent.run('Who won the 2024 Formula 1 championship?')
    print(result.output)


asyncio.run(main())
```

### Example 5: custom local search function as callable fallback

```python
import asyncio
import json
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


async def my_search(ctx: RunContext[None], query: str) -> str:
    """Custom search implementation using an internal index."""
    # Placeholder — real impl would call Elasticsearch, Typesense, etc.
    return json.dumps([
        {
            'title': f'Result for {query}',
            'url': 'https://internal.example.com/1',
            'snippet': 'Relevant content...',
        },
    ])


agent = Agent(
    TestModel(),
    capabilities=[
        WebSearch(
            native=False,
            local=my_search,
            id='custom-search',
        ),
    ],
)


async def main() -> None:
    result = await agent.run('Search for PydanticAI documentation.')
    print(result.output)


asyncio.run(main())
```

---

## 4. `WebFetch` Capability

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import WebFetch`

`WebFetch` adds URL-fetching capability to an agent. Like `WebSearch`, it auto-selects native (model-level) or local (in-process httpx) fetch depending on model support.

### Constructor

```python
def __init__(
    self,
    *,
    native: WebFetchTool | Callable | bool = True,
    local: Tool | Callable | bool | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    max_uses: int | None = None,
    enable_citations: bool | None = None,
    max_content_tokens: int | None = None,
    id: str | None = None,
    defer_loading: bool = False,
) -> None: ...
```

Key points:
- `enable_citations` — when `True`, responses include inline citation markers pointing back to source URLs. Supported by OpenAI's native fetch.
- `max_content_tokens` — cap the number of tokens extracted from fetched pages to control context window usage.
- `allowed_domains` / `blocked_domains` — restrict which URLs the agent can fetch.
- `max_uses` — limit total fetch calls per run.

### Example 1: minimal web fetch with citations

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        WebFetch(enable_citations=True),
    ],
)


async def main() -> None:
    result = await agent.run('Fetch https://pydantic.dev and summarise the main features.')
    print(result.output)


asyncio.run(main())
```

### Example 2: domain allow-list (internal wiki only)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

wiki_fetch = WebFetch(
    allowed_domains=['wiki.example.com', 'docs.example.com'],
    max_content_tokens=4096,
    max_uses=3,
    id='wiki-fetch',
)

agent = Agent(TestModel(), capabilities=[wiki_fetch])


async def main() -> None:
    result = await agent.run(
        'Fetch https://wiki.example.com/deployments and extract the rollback steps.'
    )
    print(result.output)


asyncio.run(main())
```

### Example 3: `native=False` — force local fetch for debugging

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

# Forces the local httpx-based fetch; good for offline tests or intercepting traffic
local_fetch = WebFetch(
    native=False,
    local=True,
    max_content_tokens=2048,
)

agent = Agent(TestModel(), capabilities=[local_fetch])


async def main() -> None:
    result = await agent.run('Fetch https://httpbin.org/json and show me the data.')
    print(result.output)


asyncio.run(main())
```

### Example 4: combining `WebFetch` + `WebSearch` in one agent

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        WebSearch(search_context_size='medium'),
        WebFetch(enable_citations=True, max_content_tokens=8192),
    ],
)


async def main() -> None:
    result = await agent.run(
        'Search for the PydanticAI changelog, then fetch the page and list recent breaking changes.'
    )
    print(result.output)


asyncio.run(main())
```

### Example 5: custom local fetch tool as `local=my_fetch_func`

```python
import asyncio
import httpx
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


async def cached_fetch(ctx: RunContext[None], url: str) -> str:
    """Fetch a URL, returning from a local cache if available."""
    # Simplified — real impl would check a Redis/disk cache first
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=10)
        response.raise_for_status()
        return response.text[:4000]   # truncate to avoid context overflow


agent = Agent(
    TestModel(),
    capabilities=[
        WebFetch(
            native=False,
            local=cached_fetch,
            id='cached-fetch',
        ),
    ],
)


async def main() -> None:
    result = await agent.run('Fetch https://example.com and summarise it.')
    print(result.output)


asyncio.run(main())
```

---

## 5. `XSearch` Capability

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import XSearch`

`XSearch` exposes X (Twitter) search as a capability. When using xAI's Grok models, the native `x_search` tool is used directly. For other providers a `fallback_model` routes X-search calls through an xAI model, then returns results to the primary model.

### Constructor

```python
def __init__(
    self,
    *,
    fallback_model: XSearchFallbackModel = None,
    allowed_x_handles: list[str] | None = None,
    excluded_x_handles: list[str] | None = None,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    enable_image_understanding: bool | None = None,
    enable_video_understanding: bool | None = None,
    native: XSearchTool | Callable | bool = True,
    local: ... | bool | None = None,
    id: str | None = None,
    defer_loading: bool = False,
    description: str | None = None,
) -> None: ...
```

Key points:
- `fallback_model` — an xAI model name string (e.g. `'xai:grok-4.3'`) or a fully constructed `Model` instance. Required when the primary model does not support native X search.
- `allowed_x_handles` — whitelist of `@handles` whose posts can be returned.
- `excluded_x_handles` — blacklist of handles.
- `from_date` / `to_date` — restrict search to a date range.
- `enable_image_understanding` / `enable_video_understanding` — enable multimodal analysis of media in tweets.

### Example 1: xAI model native X search

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

# xAI's Grok supports native x_search — no fallback_model needed
# agent = Agent('xai:grok-4.3', capabilities=[XSearch()])
# Shown with TestModel to be runnable without an API key:
from pydantic_ai.models.test import TestModel

agent = Agent(TestModel(), capabilities=[XSearch()])


async def main() -> None:
    result = await agent.run('What is @sama saying about AI safety this week?')
    print(result.output)


asyncio.run(main())
```

### Example 2: non-xAI model with `fallback_model='xai:grok-4.3'`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

# Primary model is OpenAI/Anthropic/etc.; X search is handled by Grok
agent = Agent(
    TestModel(),
    capabilities=[
        XSearch(fallback_model='xai:grok-4.3'),
    ],
)


async def main() -> None:
    result = await agent.run('Summarise the latest posts from @PydanticDev.')
    print(result.output)


asyncio.run(main())
```

### Example 3: date-ranged X search for specific events

```python
import asyncio
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        XSearch(
            from_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            to_date=datetime(2025, 3, 31, tzinfo=timezone.utc),
            fallback_model='xai:grok-4.3',
            id='q1-2025-search',
        ),
    ],
)


async def main() -> None:
    result = await agent.run(
        'What were the most discussed AI topics on X in Q1 2025?'
    )
    print(result.output)


asyncio.run(main())
```

### Example 4: handle-filtered search for competitor monitoring

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

competitor_search = XSearch(
    allowed_x_handles=['OpenAI', 'AnthropicAI', 'GoogleDeepMind', 'MistralAI'],
    fallback_model='xai:grok-4.3',
    description='Monitor competitor AI lab announcements on X.',
    id='competitor-monitor',
)

agent = Agent(TestModel(), capabilities=[competitor_search])


async def main() -> None:
    result = await agent.run(
        'What new model releases did AI labs announce this month?'
    )
    print(result.output)


asyncio.run(main())
```

### Example 5: combined text + image/video understanding

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

multimodal_search = XSearch(
    enable_image_understanding=True,
    enable_video_understanding=True,
    fallback_model='xai:grok-4.3',
    id='multimodal-x-search',
)

agent = Agent(TestModel(), capabilities=[multimodal_search])


async def main() -> None:
    result = await agent.run(
        'Find recent posts about robotics breakthroughs that include videos or images.'
    )
    print(result.output)


asyncio.run(main())
```

---

## 6. `Instrumentation` Capability — OTel / Logfire

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import Instrumentation`

`Instrumentation` wraps `InstrumentationSettings` as a capability so you can inject OpenTelemetry or Logfire tracing into an agent without importing `logfire` at the agent definition site. It has `position='outermost'`, meaning it wraps the entire run and therefore captures all sub-spans.

### Class definition

```python
@dataclass
class Instrumentation(AbstractCapability[Any]):
    settings: InstrumentationSettings = field(default_factory=lambda: InstrumentationSettings())

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def from_spec(cls, **kwargs: Any) -> Instrumentation: ...
```

`InstrumentationSettings` fields:

```python
@dataclass
class InstrumentationSettings:
    tracer: Tracer | None = None
    event_mode: Literal['logs', 'attributes'] = 'attributes'
    include_content: bool = True
    version: Literal[1, 2] = 1
```

### Example 1: basic Logfire instrumentation

```python
import asyncio
import logfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.test import TestModel

logfire.configure()

agent = Agent(
    TestModel(),
    capabilities=[Instrumentation()],
)


async def main() -> None:
    with logfire.span('main'):
        result = await agent.run('Explain what PydanticAI is.')
        print(result.output)


asyncio.run(main())
```

### Example 2: privacy mode — `include_content=False`

Use when prompt and response content must not appear in traces (e.g. healthcare, finance, legal).

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.settings import InstrumentationSettings
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    capabilities=[
        Instrumentation(
            settings=InstrumentationSettings(
                include_content=False,   # omit prompt/response from spans
                event_mode='attributes',
            )
        ),
    ],
)


async def main() -> None:
    result = await agent.run('Sensitive user query here.')
    print(result.output)


asyncio.run(main())
```

### Example 3: OTel v2 schema with `version=2`

```python
import asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.settings import InstrumentationSettings
from pydantic_ai.models.test import TestModel

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer('my-agent')

agent = Agent(
    TestModel(),
    capabilities=[
        Instrumentation(
            settings=InstrumentationSettings(
                tracer=tracer,
                version=2,              # OTel GenAI semantic conventions v2
                event_mode='logs',
                include_content=True,
            )
        ),
    ],
)


async def main() -> None:
    result = await agent.run('What is the capital of France?')
    spans = exporter.get_finished_spans()
    print(f'Captured {len(spans)} spans')
    print(result.output)


asyncio.run(main())
```

### Example 4: `from_spec()` for config-driven setup

`from_spec` accepts the same keyword arguments as `InstrumentationSettings` and is the idiomatic way to build `Instrumentation` from YAML/environment config.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.test import TestModel

# Config values might come from environment or YAML
spec = {
    'include_content': os.environ.get('TRACING_INCLUDE_CONTENT', 'true').lower() == 'true',
    'event_mode': os.environ.get('TRACING_EVENT_MODE', 'attributes'),
    'version': int(os.environ.get('TRACING_VERSION', '1')),
}

instrumentation = Instrumentation.from_spec(**spec)

agent = Agent(TestModel(), capabilities=[instrumentation])


async def main() -> None:
    result = await agent.run('Hello.')
    print(result.output)


asyncio.run(main())
```

### Example 5: custom span attributes alongside `Instrumentation`

Pair `Instrumentation` with a custom `AbstractCapability` that injects request metadata into the active OTel span.

```python
import asyncio
from dataclasses import dataclass
from opentelemetry import trace
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.capabilities.base import AbstractCapability, CapabilityOrdering
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


@dataclass
class TenantSpanAttributes(AbstractCapability[None]):
    tenant_id: str

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    async def prepare_run(self, ctx: RunContext[None]) -> None:
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute('tenant.id', self.tenant_id)
            span.set_attribute('tenant.plan', 'enterprise')


agent = Agent(
    TestModel(),
    capabilities=[
        Instrumentation(),
        TenantSpanAttributes(tenant_id='acme-corp'),
    ],
)


async def main() -> None:
    result = await agent.run('Generate a report.')
    print(result.output)


asyncio.run(main())
```

---

## 7. `HandleDeferredToolCalls`

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import HandleDeferredToolCalls`

`HandleDeferredToolCalls` is a capability that intercepts deferred tool calls — tool invocations registered via `ExternalToolset` where execution happens outside the agent — and resolves them inline during the run. This avoids manually resuming the agent after collecting external results.

### Class definition

```python
@dataclass
class HandleDeferredToolCalls(AbstractCapability[AgentDepsT]):
    handler: Callable[
        [RunContext[AgentDepsT], DeferredToolRequests],
        DeferredToolResults | None | Awaitable[DeferredToolResults | None],
    ]

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None: ...
```

The `handler` receives a `DeferredToolRequests` (a list of pending deferred tool calls) and must return a `DeferredToolResults` mapping each call to its result, or `None` to pass through without resolving.

### Example 1: auto-approve all deferred tool calls

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai import ToolDefinition, DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


def auto_approve_handler(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    """Approve every deferred call with a stub result."""
    return DeferredToolResults(
        results={req.tool_call_id: 'approved: done' for req in requests}
    )


external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='send_notification',
            description='Send a push notification to a user.',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'string'},
                    'message': {'type': 'string'},
                },
                'required': ['user_id', 'message'],
            },
        ),
    ]
)

agent = Agent(
    TestModel(),
    capabilities=[HandleDeferredToolCalls(handler=auto_approve_handler)],
    toolsets=[external],
)


async def main() -> None:
    result = await agent.run('Send a notification to user u1 saying their order shipped.')
    print(result.output)


asyncio.run(main())
```

### Example 2: filter — approve low-risk tools, deny high-risk ones

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai import ToolDefinition, DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

LOW_RISK_TOOLS = {'lookup_record', 'get_status', 'list_items'}
HIGH_RISK_TOOLS = {'delete_record', 'send_email', 'transfer_funds'}


def risk_filter_handler(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    results: dict[str, str] = {}
    for req in requests:
        if req.tool_name in LOW_RISK_TOOLS:
            results[req.tool_call_id] = f'auto-approved result for {req.tool_name}'
        elif req.tool_name in HIGH_RISK_TOOLS:
            results[req.tool_call_id] = f'DENIED: {req.tool_name} requires manual approval'
        else:
            results[req.tool_call_id] = 'unknown tool — denied by default'
    return DeferredToolResults(results=results)


tool_defs = [
    ToolDefinition(
        name=name,
        description=f'Perform {name}.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'id': {'type': 'string'}},
            'required': ['id'],
        },
    )
    for name in LOW_RISK_TOOLS | HIGH_RISK_TOOLS
]

agent = Agent(
    TestModel(),
    capabilities=[HandleDeferredToolCalls(handler=risk_filter_handler)],
    toolsets=[ExternalToolset(tool_defs=tool_defs)],
)


async def main() -> None:
    result = await agent.run('Look up record r1 and then delete it.')
    print(result.output)


asyncio.run(main())
```

### Example 3: human approval via async wait (webhook-style)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai import ToolDefinition, DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

# Simulated approval queue
_approval_futures: dict[str, asyncio.Future[str]] = {}


async def request_human_approval(tool_name: str, args: dict) -> str:
    """Simulate waiting for a human to approve via a webhook."""
    future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    request_id = f'{tool_name}-{id(future)}'
    _approval_futures[request_id] = future
    print(f'[APPROVAL NEEDED] {tool_name}({args}) — request id: {request_id}')
    # In production, this would block until a webhook calls back.
    # For demo purposes, auto-approve after a short delay.
    await asyncio.sleep(0.01)
    future.set_result(f'approved by operator for {tool_name}')
    return await future


async def human_approval_handler(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    results: dict[str, str] = {}
    for req in requests:
        approval = await request_human_approval(req.tool_name, req.args)
        results[req.tool_call_id] = approval
    return DeferredToolResults(results=results)


external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='publish_post',
            description='Publish a post to the company blog.',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'body': {'type': 'string'},
                },
                'required': ['title', 'body'],
            },
        ),
    ]
)

agent = Agent(
    TestModel(),
    capabilities=[HandleDeferredToolCalls(handler=human_approval_handler)],
    toolsets=[external],
)


async def main() -> None:
    result = await agent.run('Publish a post titled "Hello World" with body "First post!".')
    print(result.output)


asyncio.run(main())
```

### Example 4: chaining two handlers — first for low-risk, second for escalation

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai import ToolDefinition, DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

SAFE = {'read_file', 'list_dir'}


def first_handler(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults | None:
    """Handle safe tools; return None for anything requiring escalation."""
    safe_results: dict[str, str] = {}
    needs_escalation = False
    for req in requests:
        if req.tool_name in SAFE:
            safe_results[req.tool_call_id] = (
                f'file contents for {req.args.get("path", "unknown")}'
            )
        else:
            needs_escalation = True
    if needs_escalation:
        return None   # escalate to second handler
    return DeferredToolResults(results=safe_results)


def escalation_handler(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    return DeferredToolResults(
        results={
            req.tool_call_id: f'escalated: {req.tool_name} denied'
            for req in requests
        }
    )


async def chained_handler(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> DeferredToolResults:
    result = first_handler(ctx, requests)
    if result is None:
        result = escalation_handler(ctx, requests)
    return result


external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='read_file',
            description='Read a file.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
            },
        ),
        ToolDefinition(
            name='execute_script',
            description='Execute a shell script.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'script': {'type': 'string'}},
                'required': ['script'],
            },
        ),
    ]
)

agent = Agent(
    TestModel(),
    capabilities=[HandleDeferredToolCalls(handler=chained_handler)],
    toolsets=[external],
)


async def main() -> None:
    result = await agent.run('Read /etc/hosts and then run setup.sh.')
    print(result.output)


asyncio.run(main())
```

### Example 5: logging deferred calls without resolving (return None)

Returning `None` leaves the deferred calls unresolved, allowing the agent run to continue to its next iteration or to surface the calls to the caller.

```python
import asyncio
import logging
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai import ToolDefinition, DeferredToolRequests
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

logger = logging.getLogger('deferred-audit')
logging.basicConfig(level=logging.INFO)


def audit_logger(
    ctx: RunContext[None], requests: DeferredToolRequests
) -> None:
    """Log deferred tool calls for auditing without resolving them."""
    for req in requests:
        logger.info(
            'Deferred tool requested: tool=%s args=%s call_id=%s',
            req.tool_name,
            req.args,
            req.tool_call_id,
        )
    return None   # Explicit pass — do not resolve


external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='archive_document',
            description='Archive a document to cold storage.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'doc_id': {'type': 'string'}},
                'required': ['doc_id'],
            },
        ),
    ]
)

agent = Agent(
    TestModel(),
    capabilities=[HandleDeferredToolCalls(handler=audit_logger)],
    toolsets=[external],
)


async def main() -> None:
    result = await agent.run('Archive document doc-42.')
    print(result.output)


asyncio.run(main())
```

---

## 8. `ProcessEventStream`

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import ProcessEventStream`

`ProcessEventStream` wraps the `AgentStreamEvent` sequence emitted during a streaming run. It supports two handler forms:

- **Observer** (`EventStreamHandlerFunc`) — `async def(ctx, stream) -> None`. Consumes events for side-effects (logging, metrics) without modifying them.
- **Processor** (`EventStreamProcessorFunc`) — `async def(ctx, stream) -> AsyncIterator[AgentStreamEvent]`. Transforms or filters the event stream.

The distinction is inferred from the handler's return annotation: a function that returns `None` is an observer; one that returns an `AsyncIterator` is a processor.

### Class definition

```python
@dataclass
class ProcessEventStream(AbstractCapability[AgentDepsT]):
    handler: EventStreamHandlerFunc[AgentDepsT] | EventStreamProcessorFunc[AgentDepsT]

    async def wrap_run_event_stream(
        self, ctx: RunContext[AgentDepsT], *, stream: AsyncIterable[AgentStreamEvent]
    ) -> AsyncIterable[AgentStreamEvent]: ...
```

### Example 1: observer — log all events to a list

```python
import asyncio
from typing import AsyncIterable
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.events import AgentStreamEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

captured_events: list[AgentStreamEvent] = []


async def event_logger(
    ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]
) -> None:
    async for event in stream:
        captured_events.append(event)


agent = Agent(
    TestModel(),
    capabilities=[ProcessEventStream(handler=event_logger)],
)


async def main() -> None:
    async with agent.run_stream('Tell me a short joke.') as response:
        async for chunk in response.stream_text():
            print(chunk, end='', flush=True)
    print(f'\nCaptured {len(captured_events)} events')


asyncio.run(main())
```

### Example 2: observer — compute first-token latency

```python
import asyncio
import time
from typing import AsyncIterable
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.events import AgentStreamEvent, TextDeltaEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

first_token_latency: float | None = None


async def latency_observer(
    ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]
) -> None:
    global first_token_latency
    start = time.monotonic()
    first_seen = False
    async for event in stream:
        if not first_seen and isinstance(event, TextDeltaEvent):
            first_token_latency = time.monotonic() - start
            first_seen = True


agent = Agent(
    TestModel(),
    capabilities=[ProcessEventStream(handler=latency_observer)],
)


async def main() -> None:
    async with agent.run_stream('What is 2 + 2?') as response:
        async for _ in response.stream_text():
            pass
    print(f'First-token latency: {first_token_latency:.4f}s')


asyncio.run(main())
```

### Example 3: processor — filter out `ThinkingPart` events

```python
import asyncio
from typing import AsyncIterable, AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.events import (
    AgentStreamEvent,
    ThinkingPartStartEvent,
    ThinkingPartDeltaEvent,
)
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

THINKING_EVENT_TYPES = (ThinkingPartStartEvent, ThinkingPartDeltaEvent)


async def strip_thinking(
    ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]
) -> AsyncIterator[AgentStreamEvent]:
    async for event in stream:
        if not isinstance(event, THINKING_EVENT_TYPES):
            yield event


agent = Agent(
    TestModel(),
    capabilities=[ProcessEventStream(handler=strip_thinking)],
)


async def main() -> None:
    async with agent.run_stream('Reason about whether 17 is prime.') as response:
        async for chunk in response.stream_text():
            print(chunk, end='', flush=True)
    print()


asyncio.run(main())
```

### Example 4: processor — inject custom metadata events

```python
import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.events import AgentStreamEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


@dataclass
class RunStartMarker:
    """Custom sentinel injected at the start of each run's event stream."""
    run_id: str


@dataclass
class RunEndMarker:
    """Custom sentinel injected at the end of each run's event stream."""
    run_id: str
    total_events: int


async def wrap_with_markers(
    ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]
) -> AsyncIterator[AgentStreamEvent]:
    run_id = 'run-001'
    yield RunStartMarker(run_id=run_id)     # type: ignore[misc]
    count = 0
    async for event in stream:
        count += 1
        yield event
    yield RunEndMarker(run_id=run_id, total_events=count)   # type: ignore[misc]


agent = Agent(
    TestModel(),
    capabilities=[ProcessEventStream(handler=wrap_with_markers)],
)


async def main() -> None:
    async with agent.run_stream('Hello.') as response:
        async for event in response.stream():
            print(type(event).__name__)


asyncio.run(main())
```

### Example 5: observer + processor combined via `CombinedCapability`

```python
import asyncio
from typing import AsyncIterable, AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.capabilities.base import CombinedCapability
from pydantic_ai.events import AgentStreamEvent, TextDeltaEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

text_chunks: list[str] = []


async def collect_text(
    ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]
) -> None:
    """Observer: collect text deltas for post-run analysis."""
    async for event in stream:
        if isinstance(event, TextDeltaEvent):
            text_chunks.append(event.delta)


async def uppercase_text(
    ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]
) -> AsyncIterator[AgentStreamEvent]:
    """Processor: capitalise text delta content."""
    async for event in stream:
        if isinstance(event, TextDeltaEvent):
            yield TextDeltaEvent(delta=event.delta.upper())
        else:
            yield event


combined = CombinedCapability([
    ProcessEventStream(handler=collect_text),
    ProcessEventStream(handler=uppercase_text),
])

agent = Agent(TestModel(), capabilities=[combined])


async def main() -> None:
    async with agent.run_stream('Say hello.') as response:
        async for chunk in response.stream_text():
            print(chunk, end='', flush=True)
    print(f'\nCollected {len(text_chunks)} text chunk(s)')


asyncio.run(main())
```

---

## 9. `WebFetchTool` + `XSearchTool` + `ImageGenerationTool`

**Module:** `pydantic_ai.native_tools`  
**Imports:** `from pydantic_ai import WebFetchTool, XSearchTool, ImageGenerationTool`

These three dataclasses are the native tool descriptors used internally by their corresponding capabilities. You can also construct and pass them directly as the `native=` argument to `WebFetch()`, `XSearch()`, or other capability constructors, or include them directly in a model's `native_tools` list.

### Class signatures

```python
@dataclass(kw_only=True)
class WebFetchTool(AbstractNativeTool):
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    enable_citations: bool = False
    max_content_tokens: int | None = None
    kind: str = 'web_fetch'


@dataclass(kw_only=True)
class XSearchTool(AbstractNativeTool):
    allowed_x_handles: list[str] | None = None
    excluded_x_handles: list[str] | None = None
    from_date: datetime | None = None
    to_date: datetime | None = None
    enable_image_understanding: bool = False
    enable_video_understanding: bool = False
    kind: str = 'x_search'


@dataclass(kw_only=True)
class ImageGenerationTool(AbstractNativeTool):
    action: Literal['generate', 'edit', 'auto'] = 'auto'
    background: Literal['transparent', 'opaque', 'auto'] = 'auto'
    input_fidelity: Literal['high', 'low'] | None = None
    moderation: Literal['auto', 'low'] = 'auto'
    model: ImageGenerationModelName | None = None
    kind: str = 'image_generation'
```

### Example 1: `WebFetchTool` with domain allow-list and citations

```python
import asyncio
from pydantic_ai import Agent, WebFetchTool
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

# Construct the native tool directly for fine-grained control
fetch_tool = WebFetchTool(
    allowed_domains=['docs.pydantic.dev', 'ai.pydantic.dev'],
    enable_citations=True,
    max_content_tokens=6000,
    max_uses=4,
)

# Pass as the native= override to the WebFetch capability
agent = Agent(
    TestModel(),
    capabilities=[WebFetch(native=fetch_tool)],
)


async def main() -> None:
    result = await agent.run(
        'Fetch https://ai.pydantic.dev and list all top-level sections.'
    )
    print(result.output)


asyncio.run(main())
```

### Example 2: `XSearchTool` for date-ranged political news

```python
import asyncio
from datetime import datetime, timezone
from pydantic_ai import Agent, XSearchTool
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

x_tool = XSearchTool(
    from_date=datetime(2025, 11, 1, tzinfo=timezone.utc),
    to_date=datetime(2025, 11, 30, tzinfo=timezone.utc),
    excluded_x_handles=['spam_account', 'bot_news'],
    enable_image_understanding=False,
    enable_video_understanding=False,
)

agent = Agent(
    TestModel(),
    capabilities=[XSearch(native=x_tool, fallback_model='xai:grok-4.3')],
)


async def main() -> None:
    result = await agent.run(
        'What were the biggest political stories on X in November 2025?'
    )
    print(result.output)


asyncio.run(main())
```

### Example 3: `ImageGenerationTool` with transparent background edit

```python
import asyncio
from pydantic_ai import Agent, ImageGenerationTool
from pydantic_ai.models.test import TestModel

# Configure for product image editing with transparent background
image_tool = ImageGenerationTool(
    action='edit',
    background='transparent',
    input_fidelity='high',
    moderation='auto',
    model='dall-e-3',   # or 'gpt-image-1' depending on provider support
)

# With a real OpenAI model:
#   agent = Agent('openai:gpt-4o', native_tools=[image_tool])
# TestModel shown for a runnable example:
agent = Agent(TestModel())


async def main() -> None:
    print(
        f'ImageGenerationTool: action={image_tool.action}, '
        f'background={image_tool.background}, '
        f'input_fidelity={image_tool.input_fidelity}'
    )
    # result = await agent.run('Remove the background from this product photo.')
    # print(result.output)


asyncio.run(main())
```

### Example 4: using `WebFetchTool` as `native=` override in `WebFetch` capability

This pattern pre-configures a `WebFetchTool` instance with specific options and injects it into the capability at definition time.

```python
import asyncio
from pydantic_ai import Agent, WebFetchTool
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

# Pre-configured tool — reusable across multiple agents
docs_fetch_tool = WebFetchTool(
    allowed_domains=['docs.python.org', 'peps.python.org', 'packaging.python.org'],
    enable_citations=True,
    max_content_tokens=8000,
)

python_docs_agent = Agent(
    TestModel(),
    capabilities=[WebFetch(native=docs_fetch_tool)],
    system_prompt='You help developers understand Python documentation.',
)


async def main() -> None:
    result = await python_docs_agent.run(
        'Fetch https://docs.python.org/3/library/asyncio.html and explain the main classes.'
    )
    print(result.output)


asyncio.run(main())
```

### Example 5: provider support matrix

Not all native tools are available on all model providers. The table below summarises support as of pydantic-ai 1.107.0:

| Native Tool | OpenAI (`gpt-4o` family) | Anthropic (`claude-3.x`) | xAI (`grok-4.x`) | Google (`gemini-2.x`) |
|---|---|---|---|---|
| `WebFetchTool` | Yes (`web_fetch`) | Yes (via beta header) | No | No |
| `WebSearchTool` | Yes (`web_search_preview`) | Yes (`web_search`) | Yes (`search`) | Yes |
| `XSearchTool` | No (use `fallback_model`) | No (use `fallback_model`) | Yes (native) | No |
| `ImageGenerationTool` | Yes (`dall-e-3`, `gpt-image-1`) | No | No | No |

When a native tool is unsupported by the active model, the `NativeOrLocalTool` pattern in `WebFetch`, `WebSearch`, and `XSearch` capabilities automatically falls back to the configured `local` strategy. If no `local` strategy is provided, the capability is skipped silently rather than raising an error.

---

## 10. `ToolSearch` Capability

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import ToolSearch`

`ToolSearch` enables dynamic tool discovery for agents that have a large number of deferred-loading capabilities. Rather than including every tool definition in the system prompt, the agent first calls a special discovery tool to select which capabilities are relevant to the current request.

### Class definition

```python
@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    strategy: ToolSearchStrategy[AgentDepsT] | None = None
    max_results: int = 10
    tool_description: str | None = None
    parameter_description: str | None = None
```

`ToolSearchStrategy` can be:
- `None` — auto-selects based on model support (embedding-based if available, keyword otherwise).
- `'keyword'` — BM25-style keyword matching against tool descriptions; deterministic across providers.
- `'embedding'` — vector similarity using the model's embedding endpoint.
- A `Callable[[RunContext, str], Awaitable[list[str]]]` — fully custom search returning a list of capability IDs ranked by relevance.

`tool_description` and `parameter_description` override the default prompt shown to the model when it calls the discovery tool.

### Example 1: basic deferred tools with `ToolSearch()` default strategy

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


def get_weather(ctx: RunContext[None], city: str) -> str:
    """Return the current weather for a city."""
    return f'Sunny, 22 degrees in {city}'


def get_forecast(ctx: RunContext[None], city: str, days: int) -> str:
    """Return a weather forecast for the next N days."""
    return f'{days}-day forecast for {city}: mostly sunny'


weather_cap = Capability(
    description='Weather tools: current conditions and multi-day forecasts for any city.',
    tools=[get_weather, get_forecast],
    id='weather',
    defer_loading=True,
)


def convert_currency(ctx: RunContext[None], amount: float, from_: str, to: str) -> float:
    """Convert an amount between currencies using the latest exchange rates."""
    return round(amount * 1.08, 2)  # placeholder rate


finance_cap = Capability(
    description='Finance tools: currency conversion and live exchange rates.',
    tools=[convert_currency],
    id='finance',
    defer_loading=True,
)

agent = Agent(
    TestModel(),
    capabilities=[
        ToolSearch(),        # default strategy
        weather_cap,
        finance_cap,
    ],
)


async def main() -> None:
    result = await agent.run('What is the weather in Tokyo?')
    print(result.output)


asyncio.run(main())
```

### Example 2: forcing keyword strategy for cross-provider consistency

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


def search_kb(ctx: RunContext[None], query: str) -> str:
    """Search the internal knowledge base for relevant articles."""
    return f'KB results for: {query}'


kb_cap = Capability(
    description='Knowledge base search for internal company documentation and policies.',
    tools=[search_kb],
    id='kb-search',
    defer_loading=True,
)

agent = Agent(
    TestModel(),
    capabilities=[
        ToolSearch(
            strategy='keyword',   # deterministic; same behaviour on any model provider
            max_results=5,
        ),
        kb_cap,
    ],
)


async def main() -> None:
    result = await agent.run('Find the documentation about the onboarding process.')
    print(result.output)


asyncio.run(main())
```

### Example 3: custom callable search function for semantic matching

```python
import asyncio
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

# Simulated capability registry
_registry: dict[str, dict[str, Any]] = {}


def register_cap(cap: Capability) -> None:
    if cap.id:
        desc = cap.description if isinstance(cap.description, str) else ''
        _registry[cap.id] = {'description': desc}


async def semantic_search(
    ctx: RunContext[None], query: str
) -> list[str]:
    """Return capability IDs ranked by keyword overlap (production: use embeddings)."""
    query_words = set(query.lower().split())
    scored = []
    for cap_id, info in _registry.items():
        desc_words = set(info['description'].lower().split())
        overlap = len(query_words & desc_words)
        if overlap > 0:
            scored.append((cap_id, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cap_id for cap_id, _ in scored[:5]]


def compute_stats(ctx: RunContext[None], data: list[float]) -> dict:
    """Compute mean and count of a list of numbers."""
    mean = sum(data) / len(data) if data else 0.0
    return {'mean': mean, 'count': len(data)}


stats_cap = Capability(
    description='Statistics and data analysis tools for numerical datasets.',
    tools=[compute_stats],
    id='stats',
    defer_loading=True,
)
register_cap(stats_cap)

agent = Agent(
    TestModel(),
    capabilities=[
        ToolSearch(strategy=semantic_search, max_results=3),
        stats_cap,
    ],
)


async def main() -> None:
    result = await agent.run('Compute the mean of [1.0, 2.0, 3.0, 4.0].')
    print(result.output)


asyncio.run(main())
```

### Example 4: `max_results` tuning for large toolsets

When an agent has many deferred capabilities, a low `max_results` keeps the activated tool set focused and prevents context bloat.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel

DEPARTMENTS = ['hr', 'finance', 'legal', 'it', 'marketing', 'sales', 'ops', 'r_and_d']

caps = []
for dept in DEPARTMENTS:
    def _make_lookup(d: str):
        def lookup(ctx: RunContext[None], query: str) -> str:
            return f'{d.upper()} lookup: {query}'
        lookup.__name__ = f'lookup_{d}'
        lookup.__doc__ = f'Look up {d} department records, policies, and reports.'
        return lookup

    cap = Capability(
        description=f'{dept.upper()} department tools: access records, policies, and reports.',
        tools=[_make_lookup(dept)],
        id=f'{dept}-tools',
        defer_loading=True,
    )
    caps.append(cap)

agent = Agent(
    TestModel(),
    capabilities=[
        # Only surface the 2 most relevant capabilities — prevents context overload
        ToolSearch(strategy='keyword', max_results=2),
        *caps,
    ],
)


async def main() -> None:
    result = await agent.run('I need to check the HR leave policy.')
    print(result.output)


asyncio.run(main())
```

### Example 5: combining `ToolSearch` + `Capability(defer_loading=True)` for full lazy loading

This pattern is the recommended architecture for agents that serve a wide domain. Register all capabilities with `defer_loading=True`, add a single `ToolSearch`, and let the model decide at runtime which capabilities to activate per request.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


# --- Document tools ---
def fetch_document(ctx: RunContext[None], doc_id: str) -> str:
    """Fetch a document by ID from the document store."""
    return f'Document {doc_id}: Lorem ipsum content...'


def list_documents(ctx: RunContext[None], folder: str) -> list[str]:
    """List all document IDs in a folder."""
    return [f'{folder}/doc-1', f'{folder}/doc-2']


docs_cap = Capability(
    description='Document management tools: fetch, list, and search documents in the store.',
    tools=[fetch_document, list_documents],
    id='docs',
    defer_loading=True,
)


# --- Calendar tools ---
def get_calendar(ctx: RunContext[None], date: str) -> list[str]:
    """Return scheduled events for a given date (YYYY-MM-DD)."""
    return [f'Meeting at 10:00 on {date}', f'Standup at 09:00 on {date}']


def create_event(ctx: RunContext[None], date: str, title: str) -> str:
    """Create a new calendar event on the specified date."""
    return f'Created: {title} on {date}'


calendar_cap = Capability(
    description='Calendar tools: view and create calendar events and meetings.',
    tools=[get_calendar, create_event],
    id='calendar',
    defer_loading=True,
)


# --- Billing tools ---
def get_invoice(ctx: RunContext[None], invoice_id: str) -> dict:
    """Fetch invoice details by ID from the billing system."""
    return {'id': invoice_id, 'amount': 1200.00, 'status': 'paid'}


billing_cap = Capability(
    description='Billing tools: fetch invoices, check payment status, and issue refunds.',
    tools=[get_invoice],
    id='billing',
    defer_loading=True,
)


agent = Agent(
    TestModel(),
    capabilities=[
        ToolSearch(
            strategy='keyword',
            max_results=2,
            tool_description=(
                'Select the most relevant capability for the user request. '
                'Return only capability IDs that are directly needed.'
            ),
        ),
        docs_cap,
        calendar_cap,
        billing_cap,
    ],
)


async def main() -> None:
    # Only calendar_cap will be activated for this request
    result = await agent.run('What meetings do I have on 2026-06-15?')
    print(result.output)

    # Only billing_cap will be activated for this request
    result2 = await agent.run('Show me invoice INV-2026-001.')
    print(result2.output)


asyncio.run(main())
```

---

## Cross-reference with previous volumes

| Topic | Volume |
|---|---|
| `Agent` constructor (all params) | Vol. 1 |
| `FallbackModel` | Vol. 2 |
| `ApprovalRequiredToolset` | Vol. 2 |
| `PreparedToolset` | Vol. 2 |
| `DeferredToolResults` + `CallDeferred` | Vol. 3 |
| `MCPToolset` + MCP server integration | Vol. 3 |
| `RunContext` + `Tool` + `ToolDefinition` | Vol. 4 |
| `StreamedRunResult` + `AgentStreamEvent` | Vol. 5 |
| `ModelSettings` (all providers) | Vol. 6 |
| `ConcurrencyLimitedModel` + rate limiting | Vol. 8 |
| `common_tools` (DuckDuckGo, Tavily, Exa) | Vol. 9 |
| `FunctionToolset` (all params) | Vol. 10 |
| `AbstractToolset` (ABC) | Vol. 10 |
| `WrapperCapability` | Vol. 10 |
| `AgentInstructions` + `AgentMetadata` | Vol. 11 |
| `Dataset` + `Case` (pydantic-evals) | Vol. 12 |
| `Evaluator` + `EvaluatorContext` | Vol. 12 |
| Built-in evaluators (`Equals`, `MaxDuration`, `HasMatchingSpan`, etc.) | Vol. 12 |
| `LLMJudge` + `GradingOutput` | Vol. 12 |
| `generate_dataset` | Vol. 12 |
| Online evaluation (`@evaluate`, `OnlineEvalConfig`) | Vol. 12 |
| `SpanTree` + `SpanNode` + `SpanQuery` | Vol. 12 |
| `MCPSamplingModel` + `MCPSamplingModelSettings` | Vol. 12 |
| `RetryConfig` + `TenacityTransport` | Vol. 12 |
| `ExternalToolset` | Vol. 12 |
| `Capability` (convenience bundle) | Vol. 13 (this volume) |
| `MCP` capability (auto native/local) | Vol. 13 (this volume) |
| `WebSearch` capability | Vol. 13 (this volume) |
| `WebFetch` capability | Vol. 13 (this volume) |
| `XSearch` capability | Vol. 13 (this volume) |
| `Instrumentation` capability (OTel/Logfire) | Vol. 13 (this volume) |
| `HandleDeferredToolCalls` | Vol. 13 (this volume) |
| `ProcessEventStream` | Vol. 13 (this volume) |
| `WebFetchTool` + `XSearchTool` + `ImageGenerationTool` | Vol. 13 (this volume) |
| `ToolSearch` capability | Vol. 13 (this volume) |

---

## Revision history

| Date | Package version | Notes |
|---|---|---|
| 2026-06-12 | pydantic-ai 1.107.0 | Initial Vol. 13. Ten class groups deep-dived: `Capability` (convenience bundle with instructions/tools/toolsets; `defer_loading`; decorator syntax `@cap.tool`/`@cap.tool_plain`/`@cap.instructions`; shared instance pattern; callable description for deferred routing); `MCP` capability (auto native/local selection by URL; `native=True/False`; `local=True/False`; `authorization_token`; `allowed_tools` filtering; deferred with description; migration from `MCPToolset`); `WebSearch` capability (`search_context_size`; `WebSearchUserLocation` geo-targeting; domain allow-lists; force local DuckDuckGo with `native=False, local='duckduckgo'`; custom callable fallback); `WebFetch` capability (`enable_citations`; `max_content_tokens`; domain allow-list; force local; combined with `WebSearch`; custom cached fetch callable); `XSearch` capability (`fallback_model` for non-xAI models; date-ranged search with `from_date`/`to_date`; handle allow/block lists for competitor monitoring; `enable_image_understanding`/`enable_video_understanding`); `Instrumentation` capability (`InstrumentationSettings` with `include_content=False` privacy mode; OTel v2 schema with `version=2`; `from_spec()` config-driven setup; custom span attribute capability alongside); `HandleDeferredToolCalls` (auto-approve all; risk-filter approve/deny; async webhook-style human approval; chained two-handler pattern; audit-log-only with `return None`); `ProcessEventStream` (observer for event logging and latency measurement; processor for filtering `ThinkingPart` events and injecting custom markers; combined observer+processor via `CombinedCapability`); `WebFetchTool`/`XSearchTool`/`ImageGenerationTool` native tool dataclasses (all constructor fields; direct construction + `native=` injection; provider support matrix); `ToolSearch` capability (default auto strategy; `'keyword'` for cross-provider determinism; custom callable semantic search; `max_results` tuning for large catalogues; full lazy-loading composition with `Capability(defer_loading=True)` across docs/calendar/billing). |

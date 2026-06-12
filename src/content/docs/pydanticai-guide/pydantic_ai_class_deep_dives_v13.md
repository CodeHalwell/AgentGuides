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

Ten class groups drawn from the capabilities subsystem and native-tool layer: `Capability` (the zero-subclassing bundle for tools + instructions); `MCP` (auto native-vs-local MCP capability); `WebSearch` + `WebFetch` (auto native-vs-local search/fetch capabilities); `XSearch` (X/Twitter search with optional fallback model); `Instrumentation` (OTel/Logfire tracing capability, always outermost); `HandleDeferredToolCalls` (inline deferred-tool resolution); `ProcessEventStream` (observer and processor forms for the event stream); `WebFetchTool` + `XSearchTool` + `ImageGenerationTool` (the three uncovered native-tool dataclasses); and `ToolSearch` (tool discovery capability with all strategy options).

---

## 1. `Capability` — Convenience Bundle (no subclassing required)

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import Capability`

`Capability` lets you bundle static instructions, function tools, and toolsets under a single named identity — without subclassing `AbstractCapability`. Use it when you want role-scoped tool sets, feature flags, or per-request capability injection without writing a class.

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

# Decorator API (mirrors Agent):
cap.tool(func)          # tool with RunContext
cap.tool_plain(func)    # tool without RunContext
cap.instructions(func)  # instruction function
```

Key points:
- Returns a live `FunctionToolset` reference — tools registered **after** construction via `@cap.tool` still surface
- `defer_loading=True` + `id=` hides the capability until the model calls `load_capability`
- `description` can be a callable `(RunContext) -> str | None` for dynamic routing hints
- NOT spec-serializable (`get_serialization_name()` returns `None`)

### 1a — Minimal bundle with static instructions

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

analytics_cap = Capability(
    id='analytics',
    instructions='You are a data analyst. Report numbers clearly.',
)


@analytics_cap.tool_plain
def total_sales(month: str) -> float:
    """Return total sales for the given month."""
    return {'jan': 12_500.0, 'feb': 9_800.0}.get(month.lower(), 0.0)


@analytics_cap.tool_plain
def top_products(month: str) -> list[str]:
    """Return the top 3 products for the given month."""
    return ['Widget A', 'Widget B', 'Gadget X']


agent = Agent(TestModel(custom_result_text='Sales report ready.'))


async def main() -> None:
    result = await agent.run('What were January sales?', capabilities=[analytics_cap])
    print(result.output)


asyncio.run(main())
```

### 1b — `@cap.tool` (with RunContext) + `@cap.instructions` decorator

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel


@dataclass
class UserDeps:
    user_id: str
    locale: str


user_cap: Capability[UserDeps] = Capability(id='user-tools')


@user_cap.tool
def get_profile(ctx: RunContext[UserDeps]) -> dict:
    """Fetch the current user's profile."""
    return {'user_id': ctx.deps.user_id, 'locale': ctx.deps.locale}


@user_cap.instructions
def locale_instructions(ctx: RunContext[UserDeps]) -> str:
    return f'Always reply in {ctx.deps.locale}. Current user: {ctx.deps.user_id}.'


agent = Agent(TestModel(custom_result_text='Profile loaded.'), deps_type=UserDeps)


async def main() -> None:
    result = await agent.run(
        'Show my profile',
        deps=UserDeps(user_id='u42', locale='fr-FR'),
        capabilities=[user_cap],
    )
    print(result.output)


asyncio.run(main())
```

### 1c — Callable description for deferred capability routing

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel


def billing_description(ctx: RunContext[None]) -> str:
    return 'Billing tools: create invoices, apply discounts, check payment status.'


billing_cap = Capability(
    id='billing',
    description=billing_description,   # shown to model when defer_loading=True
    defer_loading=True,                 # hidden until the model requests it
    instructions='Handle billing queries professionally and precisely.',
)


@billing_cap.tool_plain
def create_invoice(amount: float, customer_id: str) -> str:
    """Create a new invoice."""
    return f'Invoice #INV-{customer_id[-4:]} for ${amount:.2f} created.'


@billing_cap.tool_plain
def check_payment(invoice_id: str) -> str:
    """Check the payment status of an invoice."""
    return f'Invoice {invoice_id}: PAID'


agent = Agent(TestModel(custom_result_text='Billing handled.'))


async def main() -> None:
    result = await agent.run(
        'Create an invoice for $250 for customer C1234',
        capabilities=[billing_cap],
    )
    print(result.output)


asyncio.run(main())
```

### 1d — Role-scoped capabilities per run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

read_cap = Capability(id='read', instructions='Read-only access. Never modify data.')
write_cap = Capability(id='write', instructions='Full read/write access.')


@read_cap.tool_plain
def list_records() -> list[str]:
    """List all records."""
    return ['rec-1', 'rec-2', 'rec-3']


@write_cap.tool_plain
def delete_record(record_id: str) -> str:
    """Delete a record by ID."""
    return f'{record_id} deleted.'


# write_cap also gets read tools
@write_cap.tool_plain
def list_records_full() -> list[str]:
    """List all records including archived."""
    return ['rec-1', 'rec-2', 'rec-3', 'arc-99']


agent = Agent(TestModel(custom_result_text='Done.'))


async def run_with_role(role: str, query: str) -> str:
    cap = write_cap if role == 'admin' else read_cap
    result = await agent.run(query, capabilities=[cap])
    return result.output


async def main() -> None:
    print(await run_with_role('viewer', 'List records'))
    print(await run_with_role('admin', 'Delete rec-2'))


asyncio.run(main())
```

### 1e — Sharing a `Capability` instance across multiple agents

```python
# A single Capability instance can be attached to multiple agents.
# Late-registered tools (via @cap.tool_plain after construction) appear on all agents.

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

shared = Capability(id='shared-utils', instructions='Use shared utilities as needed.')


@shared.tool_plain
def current_time() -> str:
    """Return the current UTC time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


agent_a = Agent(TestModel(custom_result_text='A done.'))
agent_b = Agent(TestModel(custom_result_text='B done.'))


# Both agents get `current_time` via the shared capability
async def use_both() -> None:
    import asyncio
    a, b = await asyncio.gather(
        agent_a.run('What time is it?', capabilities=[shared]),
        agent_b.run('Tell me the time', capabilities=[shared]),
    )
    print(a.output, b.output)
```

---

## 2. `MCP` Capability — Auto Native/Local MCP

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import MCP`

`MCP` is a `NativeOrLocalTool` subclass that connects to a remote MCP server. It prefers the model's native MCP support and falls back to a local `MCPToolset` when native is unavailable. A deprecation warning fires when `native=None` (the current default) — always pass `native=True` or `native=False` explicitly.

### Constructor

```python
def __init__(
    self,
    url: str,
    *,
    native: MCPServerTool | Callable | bool | None = None,  # deprecated default → True in v2 it will be False
    local: MCPToolsetClient | MCPToolset | MCPServer | FastMCPToolset | Callable | bool | None = None,
    id: str | None = None,                    # defaults to slug from url
    authorization_token: str | None = None,   # auth header for both native and local
    headers: dict[str, str] | None = None,    # extra headers for both
    allowed_tools: list[str] | None = None,   # filter applied to both native and local
    description: str | None = None,           # shown when defer_loading=True
    defer_loading: bool = False,
) -> None: ...
```

### 2a — Basic usage (explicit `native=True`)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='MCP response.'),
    capabilities=[
        MCP(
            'https://mcp.example.com',
            native=True,   # use native MCP on supporting providers; MCPToolset elsewhere
        )
    ],
)


async def main() -> None:
    result = await agent.run('Call the MCP server')
    print(result.output)


asyncio.run(main())
```

### 2b — Native-only (no local fallback)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel

# native=True, local=False → errors on providers without native MCP support
agent = Agent(
    TestModel(custom_result_text='Native MCP only.'),
    capabilities=[
        MCP('https://mcp.example.com', native=True, local=False)
    ],
)
```

### 2c — Local-only with authorization

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel
import os

agent = Agent(
    TestModel(custom_result_text='Local MCP response.'),
    capabilities=[
        MCP(
            'https://internal-mcp.corp.example.com',
            native=False,                                      # force local MCPToolset
            local=True,                                        # use default MCPToolset
            authorization_token=os.getenv('MCP_TOKEN', ''),
            headers={'X-Tenant': 'acme'},
            allowed_tools=['search', 'summarise'],             # restrict to two tools
        )
    ],
)
```

### 2d — Deferred MCP capability

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.models.test import TestModel

# The MCP tools stay hidden until the model explicitly requests loading
heavy_mcp = MCP(
    'https://large-corpus-mcp.example.com',
    native=True,
    local=False,
    id='large-corpus',
    description='Large knowledge corpus — load only when deep research is needed.',
    defer_loading=True,
)

agent = Agent(
    TestModel(custom_result_text='Research complete.'),
    capabilities=[heavy_mcp],
)
```

### 2e — Migration: explicit `MCPToolset` → `MCP` capability

```python
# Before (explicit MCPToolset):
from pydantic_ai.toolsets import MCPToolset  # type: ignore

# After (MCP capability — simpler, native-first):
from pydantic_ai.capabilities import MCP
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Migrated.'),
    capabilities=[
        MCP(
            'https://mcp.example.com',
            native=True,              # native when supported
            local=True,               # MCPToolset fallback otherwise
            authorization_token='tok_xyz',
        )
    ],
)
```

---

## 3. `WebSearch` Capability — Auto Native/Local Web Search

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import WebSearch`

`WebSearch` is a `NativeOrLocalTool` subclass. With `native=True` (default), it uses the provider's built-in web search (Anthropic, OpenAI). On providers without native support it falls back to a local tool — DuckDuckGo by default when `local=None`.

### Constructor

```python
def __init__(
    self,
    *,
    native: WebSearchTool | Callable | bool = True,
    local: WebSearchLocalStrategy | Tool | Callable | bool | None = None,
    search_context_size: Literal['low', 'medium', 'high'] | None = None,  # native only
    user_location: WebSearchUserLocation | None = None,  # native only
    blocked_domains: list[str] | None = None,   # native; enforced locally when possible
    allowed_domains: list[str] | None = None,   # native; mutually exclusive with blocked_domains
    max_uses: int | None = None,                # native only
    id: str | None = None,
    defer_loading: bool = False,
    description: str | None = None,
) -> None: ...
```

### 3a — Default native search with higher context

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Search results processed.'),
    capabilities=[
        WebSearch(search_context_size='high')  # more context per result on native
    ],
)


async def main() -> None:
    result = await agent.run('What happened in AI research this week?')
    print(result.output)


asyncio.run(main())
```

### 3b — Geo-targeted search

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models import WebSearchUserLocation
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Local results.'),
    capabilities=[
        WebSearch(
            search_context_size='medium',
            user_location=WebSearchUserLocation(
                city='London',
                country='GB',
                timezone='Europe/London',
            ),
        )
    ],
)
```

### 3c — Domain allow-list for constrained search

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel

# Only search trusted medical sources
agent = Agent(
    TestModel(custom_result_text='Medical info.'),
    capabilities=[
        WebSearch(
            allowed_domains=['nih.gov', 'pubmed.ncbi.nlm.nih.gov', 'who.int'],
        )
    ],
)
```

### 3d — Force local DuckDuckGo (for testing / non-native providers)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='DuckDuckGo results.'),
    capabilities=[
        WebSearch(
            native=False,        # disable native search
            local='duckduckgo',  # always use DuckDuckGo function tool
        )
    ],
)
```

### 3e — Custom local search function

```python
from collections.abc import Callable
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.test import TestModel


async def my_search(query: str) -> str:
    """Custom search backed by an internal search index."""
    # In production: call your own search API
    return f'Internal results for: {query}'


agent = Agent(
    TestModel(custom_result_text='Custom search done.'),
    capabilities=[
        WebSearch(
            native=True,           # still try native first
            local=my_search,       # fall back to this async function
        )
    ],
)
```

---

## 4. `WebFetch` Capability — Auto Native/Local URL Fetch

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import WebFetch`

`WebFetch` is a `NativeOrLocalTool` subclass for fetching URLs. `native=True` uses the model's built-in URL fetching (Anthropic, Google); the local fallback converts HTML to Markdown via `markdownify` and requires `pip install "pydantic-ai-slim[web-fetch]"`.

### Constructor

```python
def __init__(
    self,
    *,
    native: WebFetchTool | Callable | bool = True,
    local: Tool | Callable | bool | None = None,
    allowed_domains: list[str] | None = None,   # enforced locally when native unavailable
    blocked_domains: list[str] | None = None,   # enforced locally when native unavailable
    max_uses: int | None = None,                # native only
    enable_citations: bool | None = None,       # native only
    max_content_tokens: int | None = None,      # native only
    id: str | None = None,
    defer_loading: bool = False,
) -> None: ...
```

### 4a — Basic fetch with citations

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Page summarised.'),
    capabilities=[
        WebFetch(
            enable_citations=True,   # cite the source in the response (native)
            max_content_tokens=2048, # cap large pages (native)
        )
    ],
)


async def main() -> None:
    result = await agent.run('Summarise https://docs.pydantic.dev/latest/')
    print(result.output)


asyncio.run(main())
```

### 4b — Domain allow-list for intranet-only access

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

# Only allow fetching internal documentation
agent = Agent(
    TestModel(custom_result_text='Internal doc fetched.'),
    capabilities=[
        WebFetch(
            allowed_domains=['wiki.corp.example.com', 'docs.internal.example.com'],
        )
    ],
)
```

### 4c — Force local fetch for debugging

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel

# native=False → always use the markdownify-based local tool
# Useful when testing on providers without native web fetch
agent = Agent(
    TestModel(custom_result_text='Local fetch done.'),
    capabilities=[
        WebFetch(
            native=False,
            local=True,   # requires: pip install "pydantic-ai-slim[web-fetch]"
        )
    ],
)
```

### 4d — Combining `WebFetch` + `WebSearch` on one agent

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch, WebFetch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Research complete.'),
    system_prompt=(
        'First use web_search to find relevant URLs, '
        'then use web_fetch to read each one in full.'
    ),
    capabilities=[
        WebSearch(search_context_size='medium'),
        WebFetch(enable_citations=True, max_content_tokens=4096),
    ],
)
```

### 4e — Custom local fetch function

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.models.test import TestModel


async def cached_fetch(url: str) -> str:
    """Fetch via an internal caching proxy."""
    return f'[cached content of {url}]'


agent = Agent(
    TestModel(custom_result_text='Proxy fetch done.'),
    capabilities=[
        WebFetch(
            native=True,           # native when available
            local=cached_fetch,    # cached proxy fallback
        )
    ],
)
```

---

## 5. `XSearch` Capability — X/Twitter Search

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import XSearch`

`XSearch` is a `NativeOrLocalTool` subclass for X (Twitter) content. On xAI models the native `XSearchTool` is used directly. On non-xAI models, `fallback_model` must point to an xAI model — there is **no default fallback**.

### Constructor (key params)

```python
def __init__(
    self,
    *,
    fallback_model: str | Model | Callable[[RunContext], Model] | None = None,
    allowed_x_handles: list[str] | None = None,   # max 10
    excluded_x_handles: list[str] | None = None,  # max 10
    from_date: datetime | None = None,             # naive → UTC
    to_date: datetime | None = None,               # naive → UTC
    enable_image_understanding: bool | None = None,
    enable_video_understanding: bool | None = None,
    native: XSearchTool | Callable | bool = True,
    local: ... | bool | None = None,
    id: str | None = None,
    defer_loading: bool = False,
    description: str | None = None,
) -> None: ...
```

### 5a — Native X search on an xAI model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

# On a real xAI agent: Agent('xai:grok-4.3', ...)
agent = Agent(
    TestModel(custom_result_text='X search results.'),
    capabilities=[XSearch()],
)


async def main() -> None:
    result = await agent.run('What are people saying about pydantic-ai today?')
    print(result.output)


asyncio.run(main())
```

### 5b — Non-xAI model with `fallback_model`

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

# GPT-4o agent but X search runs on a grok subagent
agent = Agent(
    TestModel(custom_result_text='X search via fallback.'),
    capabilities=[
        XSearch(
            fallback_model='xai:grok-4.3',  # required for non-xAI models
        )
    ],
)
```

### 5c — Date-ranged event monitoring

```python
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Event coverage found.'),
    capabilities=[
        XSearch(
            from_date=datetime(2026, 6, 1),   # naive → UTC
            to_date=datetime(2026, 6, 7),
        )
    ],
)
```

### 5d — Handle-filtered competitor monitoring

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Competitor mentions.'),
    capabilities=[
        XSearch(
            allowed_x_handles=['pydantic', 'langchain', 'crewai'],  # max 10
            enable_image_understanding=True,
        )
    ],
)
```

### 5e — Combined image + video understanding

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Multimodal X analysis.'),
    capabilities=[
        XSearch(
            enable_image_understanding=True,
            enable_video_understanding=True,
        )
    ],
)
```

---

## 6. `Instrumentation` Capability — OTel / Logfire Tracing

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import Instrumentation`

`Instrumentation` wraps the full agent run in an OpenTelemetry span. It always positions itself `'outermost'` so it captures the entire run regardless of other capabilities. It uses the global `TracerProvider` by default (set by `logfire.configure()`).

### Class

```python
@dataclass
class Instrumentation(AbstractCapability[Any]):
    settings: InstrumentationSettings = field(
        default_factory=lambda: InstrumentationSettings()
    )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def from_spec(cls, **kwargs: Any) -> Instrumentation: ...
    # Accepted kwargs: include_content, include_binary_content, version,
    #                  event_mode, use_aggregated_usage_attribute_names
```

### 6a — Basic Logfire instrumentation

```python
import asyncio
import logfire  # pip install logfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.test import TestModel

logfire.configure(send_to_logfire=False)  # local-only for demo

agent = Agent(
    TestModel(custom_result_text='Traced response.'),
    capabilities=[Instrumentation()],   # uses global TracerProvider from logfire.configure()
)


async def main() -> None:
    result = await agent.run('Hello, world!')
    print(result.output)


asyncio.run(main())
```

### 6b — Privacy mode (omit content from spans)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Private response.'),
    capabilities=[
        Instrumentation(
            settings=InstrumentationSettings(
                include_content=False,         # don't log prompts / completions
                include_binary_content=False,  # don't log images / audio
            )
        )
    ],
)
```

### 6c — OTel schema version 2 with aggregated usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='v2 schema.'),
    capabilities=[
        Instrumentation(
            settings=InstrumentationSettings(
                version=2,
                use_aggregated_usage_attribute_names=True,
            )
        )
    ],
)
```

### 6d — `from_spec()` for YAML / JSON config

```python
# Build Instrumentation from a plain dict (e.g. loaded from YAML config)
from pydantic_ai.capabilities import Instrumentation

spec = {
    'version': 2,
    'include_content': False,
    'event_mode': 'logs',
}

instrumentation = Instrumentation.from_spec(**spec)
# Equivalent to:
# Instrumentation(settings=InstrumentationSettings(version=2, include_content=False, event_mode='logs'))
```

### 6e — Adding custom span attributes alongside `Instrumentation`

```python
import asyncio
from opentelemetry import trace
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation, Capability
from pydantic_ai.models.test import TestModel

tag_cap = Capability(id='tagger', instructions='')


@tag_cap.tool
def get_data(ctx: RunContext[None]) -> dict:
    """Fetch data and tag the span with a business metric."""
    span = trace.get_current_span()
    span.set_attribute('app.data_source', 'warehouse')
    span.set_attribute('app.rows_fetched', 42)
    return {'rows': 42}


agent = Agent(
    TestModel(custom_result_text='Tagged.'),
    capabilities=[
        Instrumentation(),  # outermost — creates the run span
        tag_cap,
    ],
)


async def main() -> None:
    result = await agent.run('Fetch data')
    print(result.output)


asyncio.run(main())
```

---

## 7. `HandleDeferredToolCalls` — Inline Deferred Tool Resolution

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import HandleDeferredToolCalls`

Without this capability, tools marked `requires_approval=True` (or in an `ExternalToolset`) cause the agent run to pause and return `DeferredToolRequests` as output. `HandleDeferredToolCalls` intercepts those calls, runs the handler inline, and the agent continues automatically.

### Class

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
    # Returns None → pass to next capability in chain
```

### 7a — Auto-approve all deferred calls

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults


async def approve_all(
    ctx: RunContext[None],
    requests: DeferredToolRequests,
) -> DeferredToolResults:
    return requests.build_results(approve_all=True)


toolset = FunctionToolset()


@toolset.tool_plain(requires_approval=True)
def send_notification(message: str) -> str:
    """Send a system notification."""
    return f'Sent: {message}'


agent = Agent(
    TestModel(custom_result_text='Notification sent.'),
    toolsets=[toolset],
    capabilities=[HandleDeferredToolCalls(handler=approve_all)],
)


async def main() -> None:
    result = await agent.run('Send a notification: Deploy complete')
    print(result.output)


asyncio.run(main())
```

### 7b — Selective approval based on risk

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved

LOW_RISK = {'send_notification', 'log_event', 'read_file'}
HIGH_RISK = {'delete_record', 'send_email_blast', 'write_file'}


async def selective_handler(
    ctx: RunContext[None],
    requests: DeferredToolRequests,
) -> DeferredToolResults | None:
    results: dict[str, ToolApproved | str] = {}
    for call in requests.tool_calls:
        if call.tool_name in LOW_RISK:
            results[call.tool_call_id] = ToolApproved()
        elif call.tool_name in HIGH_RISK:
            results[call.tool_call_id] = (
                f'Denied: {call.tool_name!r} requires human approval'
            )
    return requests.build_results(results) if results else None


toolset = FunctionToolset()


@toolset.tool_plain(requires_approval=True)
def send_notification(message: str) -> str:
    return f'Notified: {message}'


@toolset.tool_plain(requires_approval=True)
def delete_record(record_id: str) -> str:
    return f'Deleted: {record_id}'


agent = Agent(
    TestModel(custom_result_text='Selective approval done.'),
    toolsets=[toolset],
    capabilities=[HandleDeferredToolCalls(handler=selective_handler)],
)


async def main() -> None:
    result = await agent.run('Notify team AND delete record #99.')
    print(result.output)


asyncio.run(main())
```

### 7c — Chaining two handlers (escalation pattern)

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved

# First handler: auto-approve low-risk
async def auto_handler(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults | None:
    auto = {
        call.tool_call_id: ToolApproved()
        for call in requests.tool_calls
        if call.tool_name in {'read', 'log'}
    }
    return requests.build_results(auto) if auto else None


# Second handler (fallback): deny everything else with an explanation
async def deny_handler(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
    return requests.build_results({
        call.tool_call_id: 'Requires escalation review'
        for call in requests.tool_calls
    })


toolset = FunctionToolset()


@toolset.tool_plain(requires_approval=True)
def read(path: str) -> str:
    return f'content of {path}'


@toolset.tool_plain(requires_approval=True)
def deploy(target: str) -> str:
    return f'deployed to {target}'


agent = Agent(
    TestModel(custom_result_text='Chained approval.'),
    toolsets=[toolset],
    capabilities=[
        HandleDeferredToolCalls(handler=auto_handler),   # first
        HandleDeferredToolCalls(handler=deny_handler),   # fallback
    ],
)
```

### 7d — Async handler with external approval queue simulation

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved


async def webhook_approval(
    ctx: RunContext[None],
    requests: DeferredToolRequests,
) -> DeferredToolResults:
    # Simulate an async approval check (e.g. poll a webhook)
    await asyncio.sleep(0)   # stand-in for real I/O
    return requests.build_results({
        call.tool_call_id: ToolApproved()
        for call in requests.tool_calls
    })


toolset = FunctionToolset()


@toolset.tool_plain(requires_approval=True)
def send_email(to: str, subject: str) -> str:
    return f'Email sent to {to}: {subject}'


agent = Agent(
    TestModel(custom_result_text='Email queued.'),
    toolsets=[toolset],
    capabilities=[HandleDeferredToolCalls(handler=webhook_approval)],
)


async def main() -> None:
    result = await agent.run('Send a welcome email to alice@example.com')
    print(result.output)


asyncio.run(main())
```

---

## 8. `ProcessEventStream` — Observer and Processor Forms

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import ProcessEventStream`

`ProcessEventStream` exposes the agent's event stream to user code. Two forms:
- **Observer** — `async def(ctx, stream) -> None`: sees all events, passes them through unchanged; a slow observer back-pressures the stream
- **Processor** — `async def(ctx, stream) -> AsyncIterator[AgentStreamEvent]` (async generator): its yielded events *replace* the stream for downstream consumers

When this capability is registered, `agent.run()` automatically enables streaming — no explicit `event_stream_handler` argument needed.

### Class

```python
@dataclass
class ProcessEventStream(AbstractCapability[AgentDepsT]):
    handler: EventStreamHandlerFunc[AgentDepsT] | EventStreamProcessorFunc[AgentDepsT]
    # EventStreamHandlerFunc  = async def(ctx, stream: AsyncIterable) -> None
    # EventStreamProcessorFunc = async def(ctx, stream: AsyncIterable) -> AsyncIterator
```

### 8a — Observer: log all events

```python
import asyncio
from collections.abc import AsyncIterable
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models.test import TestModel

received_events: list[str] = []


async def log_observer(
    ctx: RunContext[None],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        received_events.append(type(event).__name__)


agent = Agent(
    TestModel(custom_result_text='Observed.'),
    capabilities=[ProcessEventStream(handler=log_observer)],
)


async def main() -> None:
    result = await agent.run('Hello')
    print(result.output)
    print('Events seen:', received_events)


asyncio.run(main())
```

### 8b — Observer: measure first-token latency

```python
import asyncio
import time
from collections.abc import AsyncIterable
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartStartEvent
from pydantic_ai.models.test import TestModel


class LatencyTracker:
    first_token_ms: float | None = None
    _start: float = 0.0

    def reset(self) -> None:
        self.first_token_ms = None
        self._start = time.monotonic()


tracker = LatencyTracker()


async def latency_observer(
    ctx: RunContext[None],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    tracker.reset()
    async for event in stream:
        if tracker.first_token_ms is None and isinstance(event, PartStartEvent):
            tracker.first_token_ms = (time.monotonic() - tracker._start) * 1000


agent = Agent(
    TestModel(custom_result_text='Timed.'),
    capabilities=[ProcessEventStream(handler=latency_observer)],
)


async def main() -> None:
    result = await agent.run('Quick question')
    print(result.output)
    if tracker.first_token_ms is not None:
        print(f'First-token latency: {tracker.first_token_ms:.1f} ms')


asyncio.run(main())
```

### 8c — Processor: strip ThinkingPart events

```python
import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartStartEvent, ThinkingPart
from pydantic_ai.models.test import TestModel


async def strip_thinking(
    ctx: RunContext[None],
    stream: AsyncIterable[AgentStreamEvent],
) -> AsyncIterator[AgentStreamEvent]:
    async for event in stream:
        # Drop PartStartEvent that carries a ThinkingPart
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            continue
        yield event


agent = Agent(
    TestModel(custom_result_text='No thinking exposed.'),
    capabilities=[ProcessEventStream(handler=strip_thinking)],
)


async def main() -> None:
    result = await agent.run('Reason carefully')
    print(result.output)


asyncio.run(main())
```

### 8d — Processor: inject audit metadata event

```python
import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, FinalResultEvent
from pydantic_ai.models.test import TestModel
import time


async def audit_injector(
    ctx: RunContext[None],
    stream: AsyncIterable[AgentStreamEvent],
) -> AsyncIterator[AgentStreamEvent]:
    run_start = time.monotonic()
    async for event in stream:
        yield event
        if isinstance(event, FinalResultEvent):
            # After the final result, record timing in run metadata
            # (Real usage: write to audit log, emit OTel event, etc.)
            elapsed = (time.monotonic() - run_start) * 1000
            # ctx.metadata is available for attaching run-level data
            pass


agent = Agent(
    TestModel(custom_result_text='Audited.'),
    capabilities=[ProcessEventStream(handler=audit_injector)],
)


async def main() -> None:
    result = await agent.run('Process order #42')
    print(result.output)


asyncio.run(main())
```

### 8e — Combining observer + processor via multiple capabilities

```python
from collections.abc import AsyncIterable, AsyncIterator
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartStartEvent, ThinkingPart
from pydantic_ai.models.test import TestModel

seen: list[str] = []


async def count_observer(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        seen.append(type(event).__name__)


async def strip_thinking(
    ctx: RunContext[None],
    stream: AsyncIterable[AgentStreamEvent],
) -> AsyncIterator[AgentStreamEvent]:
    async for event in stream:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            continue
        yield event


# Processor runs first (strips events), then observer sees the stripped stream
agent = Agent(
    TestModel(custom_result_text='Combined.'),
    capabilities=[
        ProcessEventStream(handler=strip_thinking),  # processor
        ProcessEventStream(handler=count_observer),  # observer
    ],
)
```

---

## 9. `WebFetchTool` + `XSearchTool` + `ImageGenerationTool` — Native Tool Dataclasses

**Module:** `pydantic_ai.native_tools`  
**Imports:**
```python
from pydantic_ai import WebFetchTool, XSearchTool, ImageGenerationTool
```

These are the low-level `AbstractNativeTool` dataclasses. Prefer the capability wrappers (`WebFetch`, `XSearch`, `ImageGeneration`) for most use — they auto-handle native vs. local fallback. Use the raw dataclasses when you need precise parameter control or to pass a specific configuration to a capability's `native=` parameter.

### `WebFetchTool` — Anthropic + Google

```python
@dataclass(kw_only=True)
class WebFetchTool(AbstractNativeTool):
    max_uses: int | None = None
    # Anthropic: stop fetching after N URLs
    allowed_domains: list[str] | None = None
    # Anthropic: whitelist (mutually exclusive with blocked_domains)
    blocked_domains: list[str] | None = None
    # Anthropic: blacklist (mutually exclusive with allowed_domains)
    enable_citations: bool = False
    # Anthropic: include source citations in responses
    max_content_tokens: int | None = None
    # Anthropic: truncate large pages to N tokens
    kind: str = 'web_fetch'
```

### `XSearchTool` — xAI only

```python
@dataclass(kw_only=True)
class XSearchTool(AbstractNativeTool):
    allowed_x_handles: list[str] | None = None   # max 10
    excluded_x_handles: list[str] | None = None  # max 10
    from_date: datetime | None = None            # naive → UTC
    to_date: datetime | None = None              # naive → UTC
    enable_image_understanding: bool = False
    enable_video_understanding: bool = False
    kind: str = 'x_search'
```

### `ImageGenerationTool` — OpenAI Responses + Google

```python
@dataclass(kw_only=True)
class ImageGenerationTool(AbstractNativeTool):
    action: Literal['generate', 'edit', 'auto'] = 'auto'
    # OpenAI: 'edit' crops/modifies existing images; 'auto' picks based on input
    background: Literal['transparent', 'opaque', 'auto'] = 'auto'
    # OpenAI: 'transparent' only for PNG/WebP output
    input_fidelity: Literal['high', 'low'] | None = None
    # OpenAI: how closely to match input image features; default 'low'
    moderation: Literal['auto', 'low'] = 'auto'
    # OpenAI: content moderation strictness
    model: ImageGenerationModelName | None = None
    # OpenAI: 'gpt-image-2', 'gpt-image-1.5', etc.; None → provider default
    kind: str = 'image_generation'
```

### 9a — `WebFetchTool` with citations and domain filtering

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, NativeOrLocalTool
from pydantic_ai import WebFetchTool
from pydantic_ai.models.test import TestModel

# Pass a fully-configured WebFetchTool as the native= override
agent = Agent(
    TestModel(custom_result_text='Fetched with citations.'),
    capabilities=[
        WebFetch(
            native=WebFetchTool(
                enable_citations=True,
                max_content_tokens=2048,
                allowed_domains=['docs.pydantic.dev', 'ai.pydantic.dev'],
                max_uses=5,
            ),
        )
    ],
)
```

### 9b — `XSearchTool` for date-ranged news monitoring

```python
from datetime import datetime
from pydantic_ai import Agent, XSearchTool
from pydantic_ai.capabilities import XSearch
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='X news found.'),
    capabilities=[
        XSearch(
            native=XSearchTool(
                from_date=datetime(2026, 6, 1),
                to_date=datetime(2026, 6, 7),
                allowed_x_handles=['openai', 'anthropic', 'pydantic'],
                enable_image_understanding=True,
            ),
        )
    ],
)
```

### 9c — `ImageGenerationTool` with transparent background editing

```python
from pydantic_ai import Agent, ImageGenerationTool
from pydantic_ai.capabilities.image_generation import ImageGeneration
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(custom_result_text='Image edited.'),
    capabilities=[
        ImageGeneration(
            native=ImageGenerationTool(
                action='edit',
                background='transparent',
                moderation='low',
                model='gpt-image-2',
            ),
        )
    ],
)
```

### 9d — Provider support matrix

| Native Tool | Anthropic | OpenAI | Google | xAI | Bedrock |
|---|---|---|---|---|---|
| `WebFetchTool` | ✅ all params | ❌ | ✅ basic | ❌ | ❌ |
| `WebSearchTool` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `XSearchTool` | ❌ | ❌ | ❌ | ✅ | ❌ |
| `ImageGenerationTool` | ❌ | ✅ full params | ✅ basic | ❌ | ❌ |
| `CodeExecutionTool` | ✅ | ✅ | ✅ | ✅ | ✅ (Nova 2) |
| `MemoryTool` | ✅ | ❌ | ❌ | ❌ | ❌ |

`UrlContextTool` is a **deprecated alias** for `WebFetchTool` — it differs only in `kind='url_context'` for backward-compatible deserialization. Migrate any existing code to `WebFetchTool`.

### 9e — Using native tools directly via `NativeTool` capability

```python
from pydantic_ai import Agent, ImageGenerationTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.models.test import TestModel

# Use AbstractNativeTool subclasses directly without a wrapper capability
agent = Agent(
    TestModel(custom_result_text='Generated.'),
    capabilities=[
        NativeTool(
            ImageGenerationTool(
                action='generate',
                model='gpt-image-2',
                background='opaque',
            )
        )
    ],
)
```

---

## 10. `ToolSearch` Capability — Tool Discovery (1.107.0)

**Module:** `pydantic_ai.capabilities`  
**Import:** `from pydantic_ai.capabilities import ToolSearch`

`ToolSearch` is auto-injected into every agent at zero overhead when no deferred tools exist. When tools have `defer_loading=True`, the capability manages their discovery. On Anthropic (BM25/regex) and OpenAI Responses (server search), discovery is provider-native and **prompt-cache compatible** — the tool list stays stable across discovery turns.

### Class

```python
@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    strategy: ToolSearchStrategy[AgentDepsT] | None = None
    # None        → provider default (Anthropic BM25, OpenAI server, local keywords elsewhere)
    # 'keywords'  → local keyword-overlap (cache-compatible on Anthropic/OpenAI too)
    # 'bm25'      → Anthropic BM25 native (errors on other providers)
    # 'regex'     → Anthropic regex native (errors on other providers)
    # callable    → custom (ctx, queries, tools) -> Sequence[str]

    max_results: int = 10
    tool_description: str | None = None       # description for the search tool the model sees
    parameter_description: str | None = None  # description for the queries parameter
```

### 10a — Basic deferred tools with default strategy

```python
import asyncio
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.models.test import TestModel


def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f'Sunny in {city}'


def get_forecast(city: str, days: int) -> str:
    """Return a weather forecast."""
    return f'{days}-day forecast for {city}: mostly sunny'


weather_tool = Tool(get_weather, defer_loading=True)
forecast_tool = Tool(get_forecast, defer_loading=True)

agent = Agent(
    TestModel(custom_result_text='Weather fetched.'),
    tools=[weather_tool, forecast_tool],
    capabilities=[ToolSearch()],  # provider default strategy
)


async def main() -> None:
    result = await agent.run('What is the weather in London?')
    print(result.output)


asyncio.run(main())
```

### 10b — Force `'keywords'` strategy for cross-provider consistency

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.models.test import TestModel


def search_docs(query: str) -> str:
    return f'Docs results: {query}'


def run_query(sql: str) -> list[dict]:
    return [{'result': 'data'}]


agent = Agent(
    TestModel(custom_result_text='Search done.'),
    tools=[
        Tool(search_docs, defer_loading=True),
        Tool(run_query, defer_loading=True),
    ],
    capabilities=[
        ToolSearch(
            strategy='keywords',  # consistent behavior on any provider
            max_results=5,
        )
    ],
)
```

### 10c — Custom callable search function

```python
import asyncio
from collections.abc import Sequence
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition


def semantic_search(
    ctx: RunContext[None],
    queries: Sequence[str],
    tools: Sequence[ToolDefinition],
) -> list[str]:
    """Return tool names matching any query keyword (simplified semantic search)."""
    keywords = {word.lower() for q in queries for word in q.split()}
    return [
        tool.name
        for tool in tools
        if keywords & set((tool.description or '').lower().split())
    ]


def send_email(to: str, subject: str, body: str) -> str:
    return f'Email sent to {to}'


def create_ticket(title: str, priority: str) -> str:
    return f'Ticket created: {title}'


def fetch_report(report_id: str) -> dict:
    return {'id': report_id, 'data': []}


agent = Agent(
    TestModel(custom_result_text='Custom search result.'),
    tools=[
        Tool(send_email, defer_loading=True),
        Tool(create_ticket, defer_loading=True),
        Tool(fetch_report, defer_loading=True),
    ],
    capabilities=[
        ToolSearch(
            strategy=semantic_search,
            max_results=3,
        )
    ],
)
```

### 10d — `max_results` tuning for large toolsets

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.models.test import TestModel

# Large toolset: 50+ deferred tools; only reveal 3 per discovery turn
# so the model's context stays small

tools = [
    Tool(lambda x: x, name=f'tool_{i}', description=f'Tool number {i}', defer_loading=True)
    for i in range(50)
]

agent = Agent(
    TestModel(custom_result_text='Narrow discovery done.'),
    tools=tools,
    capabilities=[
        ToolSearch(
            strategy='keywords',
            max_results=3,   # tight budget — reveal at most 3 tools per search
        )
    ],
)
```

### 10e — Combining `ToolSearch` with a deferred `Capability`

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, ToolSearch
from pydantic_ai.models.test import TestModel

# A capability that stays hidden until the model decides to load it
heavy_cap = Capability(
    id='data-analysis',
    description='Advanced statistical analysis and visualisation tools.',
    defer_loading=True,
    instructions='Use these tools only when the user explicitly requests data analysis.',
)


@heavy_cap.tool_plain
def run_regression(x: list[float], y: list[float]) -> dict:
    """Run a linear regression on paired data."""
    return {'slope': 1.0, 'intercept': 0.0, 'r2': 0.95}


@heavy_cap.tool_plain
def summarise_stats(values: list[float]) -> dict:
    """Return descriptive statistics for a list of values."""
    import statistics
    return {
        'mean': statistics.mean(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
    }


agent = Agent(
    TestModel(custom_result_text='Analysis complete.'),
    capabilities=[
        heavy_cap,
        ToolSearch(),   # discovers tools from heavy_cap when model requests them
    ],
)
```

---

## Cross-reference with previous volumes

| Topic | Volume |
|---|---|
| `AbstractCapability` (subclassing, `get_ordering`, `wrap_*` hooks) | Vol. 10 |
| `NativeOrLocalTool` (base class for WebSearch/WebFetch/XSearch/MCP) | Vol. 3 |
| `NativeTool` capability | Vol. 3 |
| `DynamicCapability` | Vol. 3 |
| `WebSearch` (capability, source deep-dive) | Source Deep Dive |
| `WebFetch` (capability, source deep-dive) | Source Deep Dive |
| `MCPToolset` + `load_mcp_toolsets` | Vol. 3 |
| `MCPServerTool` (native MCP wire format) | Vol. 9 |
| `InstrumentationSettings` + `InstrumentedModel` | Vol. 2 |
| `ToolSearch` (first introduction) | Vol. 3 |
| `DeferredToolRequests` + `DeferredToolResults` + `CallDeferred` | Vol. 3 |
| `HandleDeferredToolCalls` (first mention) | Advanced Classes Part 2 |
| `ProcessEventStream` (first mention) | Advanced Classes Part 2 |
| `ExternalToolset` (deferred / HITL patterns) | Vol. 12 |
| `ApprovalRequiredToolset` | Vol. 8 |
| `ToolDefinition` | Vol. 1 |
| `AgentStream` (streaming API) | Vol. 10 |
| `ImageGeneration` capability | Vol. 4 |
| `XSearch` capability (first deep-dive) | Vol. 4 |
| `CodeExecutionTool` | Vol. 7 |
| `MemoryTool` | Vol. 7 |
| `AbstractNativeTool` + `NATIVE_TOOL_TYPES` registry | Vol. 7 |
| `WrapperCapability` | Vol. 10 |
| `CombinedCapability` | Vol. 5 |

---

## Revision history

| Date | Package version | Notes |
|---|---|---|
| 2026-06-12 | pydantic-ai 1.107.0 | Initial Vol. 13. Library installed (1.107.0) and source inspected directly from PyPI. Ten class groups deep-dived: `Capability` (convenience bundle without subclassing; `@cap.tool`/`@cap.tool_plain`/`@cap.instructions` decorators; `defer_loading=True` with `id=`; callable description for deferred routing; shared instance across multiple agents; live `FunctionToolset` ref for late-registered tools; `get_serialization_name()` returns `None`); `MCP` capability (`NativeOrLocalTool` subclass; `native=None` deprecation warning — v2 will default to `local`; `authorization_token`/`headers` applied to both paths; `allowed_tools` filter; `id` from URL slug; deferred MCP capability pattern; migration from explicit `MCPToolset`); `WebSearch` capability (`native=True` default; `local='duckduckgo'`/`'tavily'` named strategies; `search_context_size` `'low'`/`'medium'`/`'high'` native only; `WebSearchUserLocation` for geo-targeted results; `blocked_domains`/`allowed_domains` mutual exclusion on Anthropic; custom async local callable); `WebFetch` capability (local fallback requires `pydantic-ai-slim[web-fetch]`; `enable_citations` native only; `max_content_tokens` native only; `allowed_domains`/`blocked_domains` enforced locally when native unavailable; combined `WebSearch`+`WebFetch` pattern; custom cached-proxy local function); `XSearch` capability (`fallback_model` required for non-xAI models — no default; `allowed_x_handles`/`excluded_x_handles` max 10 each; naive datetime → UTC; `enable_image_understanding`/`enable_video_understanding`); `Instrumentation` capability (always `position='outermost'`; `from_spec(**kwargs)` for YAML/JSON deserialization; `include_content=False` privacy mode; OTel schema `version=2`; `use_aggregated_usage_attribute_names`; per-run state isolation via `dataclasses.replace(self)`; adding custom span attributes from sibling capability via `get_current_span().set_attribute()`); `HandleDeferredToolCalls` (`handler` sync or async; `build_results(approve_all=True)` shortcut; selective approval by tool name; chaining two handlers — first non-None wins; async external approval queue pattern); `ProcessEventStream` (observer form `async def → None` — teed, pass-through, back-pressures; processor form async generator — replaces stream for downstream; `agent.run()` auto-enables streaming; multiple `ProcessEventStream` capabilities chain; durable-execution caveat — model-response events not seen live inside activities); `WebFetchTool`/`XSearchTool`/`ImageGenerationTool` (all fields; provider support matrix table; `UrlContextTool` deprecated alias for `WebFetchTool` with `kind='url_context'`; using as `native=` override in wrapper capabilities; using via `NativeTool` capability directly); `ToolSearch` capability (auto-injected at zero overhead; `strategy=None` provider default; `'keywords'` local cache-compatible; `'bm25'`/`'regex'` Anthropic-native; custom callable sync or async; `max_results` budget for large toolsets; `tool_description`/`parameter_description` for custom model prompts; combining with deferred `Capability`). |

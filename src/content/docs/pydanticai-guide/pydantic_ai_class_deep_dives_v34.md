---
title: "PydanticAI Class Deep Dives Vol. 34"
description: "Source-verified deep dives into 10 pydantic-ai 2.7.0 class groups: MCP capability (primary MCP entry point — url/native/local/authorization_token/allowed_tools/defer_loading, native=True requires url, MCPToolset passthrough), OpenRouterModel+OpenRouterModelSettings+OpenRouterReasoning+OpenRouterProviderConfig+OpenRouterUsageConfig (meta-provider — fallback models list, provider ordering, reasoning effort/max_tokens/exclude, usage tracking, data_collection policy), NativeOrLocalTool+NativeTool base classes (hybrid native/local capability pattern — _default_native/_default_local/_requires_native override points, local= accepts callable/Tool/AbstractToolset/bool), ProcessEventStream (event stream forwarding and transformation — handler vs processor form, tee'd fan-out, drop/inject events, ThinkingPart strip), ToolSearch+ToolSearchToolset (deferred tool discovery — strategy='bm25'/'regex'/'keywords'/ToolSearchFunc, max_results, enable_fallback=False native-only, custom search_fn), FunctionSignature+FunctionParam+TypeSignature+TypeExpr types (function-to-schema rendering — SimpleTypeExpr/UnionTypeExpr/GenericTypeExpr/LiteralTypeExpr, TypeFieldSignature, referenced_types, render()), WrapperAgent+AbstractAgent (agent delegation base — transparent property forwarding, override run/run_sync hooks, auth middleware, multi-agent routing), PendingMessage+PendingMessageDrainCapability (message queue internals — 'asap'/'when_idle' priority, from_content() coalescing, outermost capability ordering, end-of-run redirect), LangChainToolset+LangChainTool (LangChain bridge — Protocol-based tool wrapping, tool_from_langchain adapter, FunctionToolset subclass, id= for capability registry), AgentSpec YAML composition (complete YAML-driven agents — capabilities list, deps_schema, output_schema, CapabilitySpec short-forms, from_file/to_file, registry injection). All verified against pydantic-ai 2.7.0 source installed from PyPI."
sidebar:
  label: "Class deep dives (Vol. 34)"
  order: 60
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.7.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.7.x API.
</Aside>

Ten class groups covering the primary MCP capability entry point, the full OpenRouter meta-provider API, the `NativeOrLocalTool` hybrid pattern, event stream processing, deferred tool discovery, the function signature rendering system, agent wrapping, the pending message queue, the LangChain bridge, and YAML-driven agent composition.

---

## 1. `MCP` Capability — Primary MCP Integration Entry Point

**Source**: `pydantic_ai/capabilities/mcp.py`  
**Export**: `from pydantic_ai.capabilities import MCP`

`MCP` is the recommended capability-first way to attach an MCP server to a PydanticAI agent. It extends `NativeOrLocalTool` and accepts any MCPToolset input — a URL string, a `fastmcp.Client`, a transport object, or an in-process `FastMCP` server — directly via `local=`. The `native=True` flag advertises the server to providers that support native (server-side) MCP execution; combining both gives a transparent fallback: native when the provider supports it, local otherwise.

```python
# Key signature verified from source (pydantic-ai 2.7.0):

@dataclass(init=False)
class MCP(NativeOrLocalTool[AgentDepsT]):
    """MCP server capability."""

    url: str | None
    authorization_token: str | None
    headers: dict[str, str] | None
    allowed_tools: list[str] | None
    description: str | None

    def __init__(
        self,
        url: str | None = None,
        *,
        native: MCPServerTool
            | Callable[[RunContext[AgentDepsT]], Awaitable[MCPServerTool | None] | MCPServerTool | None]
            | bool = False,
        local: MCPToolsetClient | MCPToolset[AgentDepsT] | Callable[..., Any] | bool | None = None,
        id: str | None = None,
        authorization_token: str | None = None,
        headers: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None: ...
```

Key constraint: `MCP(native=True)` **requires** `url=` because the URL is what the provider receives to connect to the server. An explicit `MCPServerTool(...)` or a per-run callable carries its own URL so the capability's `url=` is not needed in those cases.

### 1.1 HTTP MCP Server with Local Fallback

Connect to a remote MCP server over HTTP; fall back to the local client when the provider does not support native MCP.

```python  {test="skip"}
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

# local=True: let MCP auto-create an MCPToolset from the URL
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        MCP(
            url='https://my-mcp-server.example.com/mcp',
            local=True,
            authorization_token=os.environ['MCP_AUTH_TOKEN'],
        )
    ],
)

async def main() -> None:
    async with agent:
        result = await agent.run('List available resources from the MCP server.')
        print(result.output)

asyncio.run(main())
```

### 1.2 Native-Only MCP (Provider-Side Execution)

Advertise the server to the provider; skip the local client entirely. Useful when the provider's native MCP integration handles credentials and rate limiting.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.native_tools import MCPServerTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        MCP(
            url='https://my-mcp-server.example.com/mcp',
            native=True,       # provider executes server-side
            local=False,       # no local fallback
            allowed_tools=['read_file', 'write_file', 'list_directory'],
            description='File system access MCP server',
        )
    ],
)

async def main() -> None:
    async with agent:
        result = await agent.run('Read the contents of /etc/hosts.')
        print(result.output)

asyncio.run(main())
```

### 1.3 In-Process FastMCP Server (Local Only)

Pass a `FastMCP` server object or a `fastmcp.Client` as `local=` — no URL needed when running entirely in-process.

```python  {test="skip"}
import asyncio
import fastmcp
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

# Define an in-process MCP server
mcp_server = fastmcp.FastMCP('MyServer')

@mcp_server.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp_server.tool()
def echo(message: str) -> str:
    """Echo a message back."""
    return f'Echo: {message}'

# Pass the FastMCP server directly via local=
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[MCP(local=mcp_server)],
)

async def main() -> None:
    async with agent:
        result = await agent.run('Add 42 and 58 using the MCP tool.')
        print(result.output)  # > 100

asyncio.run(main())
```

---

## 2. `OpenRouterModel` + `OpenRouterModelSettings` + `OpenRouterReasoning` + `OpenRouterProviderConfig` + `OpenRouterUsageConfig`

**Source**: `pydantic_ai/models/openrouter.py`  
**Export**: `from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings`

`OpenRouterModel` extends `OpenAIChatModel` with OpenRouter-specific metadata capture. The companion `OpenRouterModelSettings` TypedDict adds five `openrouter_`-prefixed fields: `openrouter_models` (fallback model list), `openrouter_provider` (provider routing config), `openrouter_reasoning` (reasoning token control), `openrouter_transforms` (middleware transforms), and `openrouter_usage` (usage tracking).

```python
# Key signatures verified from source:

class OpenRouterModel(OpenAIChatModel):
    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['openrouter'] | Provider[AsyncOpenAI] = 'openrouter',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ): ...

class OpenRouterModelSettings(ModelSettings, total=False):
    openrouter_models: list[str]          # fallback model chain
    openrouter_provider: OpenRouterProviderConfig  # routing constraints
    openrouter_reasoning: OpenRouterReasoning       # thinking config
    openrouter_transforms: list[str]      # middleware (e.g. ['middle-out'])
    openrouter_usage: OpenRouterUsageConfig         # usage tracking

class OpenRouterReasoning(TypedDict, total=False):
    effort: Literal['xhigh', 'high', 'medium', 'low', 'minimal', 'none']
    max_tokens: int       # cannot combine with effort
    exclude: bool         # strip reasoning from response
    enabled: bool         # enable with default params

class OpenRouterProviderConfig(TypedDict, total=False):
    order: list[OpenRouterProviderName]    # preferred provider order
    allow_fallbacks: bool
    require_parameters: bool
    data_collection: Literal['allow', 'deny']
    only: list[OpenRouterProviderName]     # restrict to subset
    quantizations: list[str]

class OpenRouterUsageConfig(TypedDict, total=False):
    include: bool
```

### 2.1 Basic OpenRouter Agent with Model Fallback Chain

Use OpenRouter's built-in fallback routing: if the primary model is unavailable or rate-limited, the next model in `openrouter_models` is tried automatically.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings, OpenRouterUsageConfig

agent = Agent(
    OpenRouterModel('anthropic/claude-sonnet-4-6'),
    model_settings=OpenRouterModelSettings(
        # Provider-side fallback chain — handled by OpenRouter, not PydanticAI
        openrouter_models=[
            'anthropic/claude-sonnet-4-6',
            'openai/gpt-5.2',
            'google/gemini-2.5-pro',
        ],
        openrouter_usage=OpenRouterUsageConfig(include=True),
    ),
    instructions='You are a helpful assistant.',
)

async def main() -> None:
    result = await agent.run('Explain the difference between TCP and UDP.')
    print(result.output)
    print(f'Tokens used: {result.usage().total_tokens}')

asyncio.run(main())
```

### 2.2 Provider Routing with Data-Privacy Constraints

Route requests to specific providers, deny data collection, and require that all request parameters are supported.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterProviderConfig,
)

agent = Agent(
    OpenRouterModel('meta-llama/llama-4-70b-instruct'),
    model_settings=OpenRouterModelSettings(
        openrouter_provider=OpenRouterProviderConfig(
            order=['together', 'fireworks'],   # provider slugs: try Together first, then Fireworks
            allow_fallbacks=True,
            require_parameters=True,           # don't route to providers missing params
            data_collection='deny',            # GDPR-friendly: no data stored
        ),
    ),
)

async def main() -> None:
    result = await agent.run('Summarise the key points of the Kyoto Protocol.')
    print(result.output)

asyncio.run(main())
```

### 2.3 Reasoning Models via OpenRouter

Configure chain-of-thought reasoning tokens. Use `effort` for OpenAI-style levels or `max_tokens` for Anthropic-style budgets — not both.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterReasoning,
)

# Effort-based reasoning (OpenAI-style)
reasoning_agent = Agent(
    OpenRouterModel('openai/o4-mini'),
    model_settings=OpenRouterModelSettings(
        openrouter_reasoning=OpenRouterReasoning(
            effort='high',
            exclude=False,   # include reasoning in response for debugging
        ),
    ),
)

# Token-budget reasoning (Anthropic-style)
budget_agent = Agent(
    OpenRouterModel('anthropic/claude-opus-4-8'),
    model_settings=OpenRouterModelSettings(
        openrouter_reasoning=OpenRouterReasoning(
            max_tokens=8000,
            exclude=True,    # strip thinking tokens from final response
        ),
    ),
)

async def main() -> None:
    problem = 'A bat and ball cost £1.10 in total. The bat costs £1 more than the ball. How much does the ball cost?'
    
    result = await reasoning_agent.run(problem)
    print('Reasoning agent:', result.output)
    
    result2 = await budget_agent.run(problem)
    print('Budget agent:', result2.output)

asyncio.run(main())
```

---

## 3. `NativeOrLocalTool` + `NativeTool` — Hybrid Native/Local Capability Base

**Source**: `pydantic_ai/capabilities/native_or_local.py`, `pydantic_ai/capabilities/native_tool.py`  
**Export**: `from pydantic_ai.capabilities import NativeOrLocalTool, NativeTool`

`NativeOrLocalTool` is the abstract base for capabilities that pair a provider-native tool with a local Python fallback. When the model supports the native tool, the local version is suppressed; otherwise the native tool is removed and the local version runs. Subclass it to define `_default_native`, `_default_local`, and `_requires_native` — then callers override with their own `native=` / `local=` arguments.

`NativeTool` is a simpler capability that wraps a single `AbstractNativeTool` (or a per-run callable producing one) and registers it with the agent without any local fallback logic.

```python
# Key signatures verified from source:

@dataclass(init=False)
class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    # Overrideable class-level defaults (set in subclasses):
    def _default_native(self) -> AgentNativeTool[AgentDepsT] | bool | None: return False
    def _default_local(self) -> AgentToolset[AgentDepsT] | bool | None: return None
    def _requires_native(self) -> bool: return False   # raise UserError if native unavailable

    # Runtime resolved fields:
    _native: AgentNativeTool[AgentDepsT] | bool | None  # resolved from native= or _default_native()
    _local: AgentToolset[AgentDepsT] | bool | None      # resolved from local= or _default_local()

@dataclass
class NativeTool(AbstractCapability[AgentDepsT]):
    tool: AgentNativeTool[AgentDepsT]  # AbstractNativeTool or per-run callable
```

### 3.1 Custom Hybrid Capability — Native + DuckDuckGo Fallback

Create a web search capability that uses the model's native search when available and falls back to a DuckDuckGo function tool otherwise.

```python  {test="skip"}
from __future__ import annotations

import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.toolsets import FunctionToolset


def _make_ddg_fallback() -> FunctionToolset:
    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain
    def web_search(query: str) -> str:
        """Search the web using DuckDuckGo."""
        # Minimal local implementation — replace with real DuckDuckGo call
        return f'[DuckDuckGo fallback] Results for: {query}'

    return toolset


class HybridWebSearch(NativeOrLocalTool):
    """Web search — native on supporting providers, DuckDuckGo locally."""

    def __init__(self, *, search_context_size: str = 'medium') -> None:
        super().__init__(
            native=WebSearchTool(search_context_size=search_context_size),  # type: ignore[arg-type]
            local=_make_ddg_fallback(),
        )

    def _requires_native(self) -> bool:
        return False  # local fallback is always acceptable


# Agent gets native search on Anthropic/OpenAI, local on Ollama
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[HybridWebSearch(search_context_size='high')],
)

async def main() -> None:
    result = await agent.run('What is the latest version of Python?')
    print(result.output)

asyncio.run(main())
```

### 3.2 `NativeTool` — Register a Single Native Tool

Register a static `AbstractNativeTool` directly without any local fallback consideration.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool, CodeExecutionTool

# Static tool
search_cap = NativeTool(tool=WebSearchTool(search_context_size='low'))

# Per-run callable — decide the tool dynamically
def choose_code_tool(ctx: RunContext) -> CodeExecutionTool | None:
    # Disable code execution if the run is in safe mode
    metadata = ctx.metadata or {}
    if metadata.get('safe_mode'):
        return None
    return CodeExecutionTool()

code_cap = NativeTool(tool=choose_code_tool)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[search_cap, code_cap],
)

async def main() -> None:
    result = await agent.run(
        'Search for "Python asyncio" and summarise the top result.',
        deps=None,
    )
    print(result.output)

asyncio.run(main())
```

---

## 4. `ProcessEventStream` — Event Stream Forwarding and Transformation

**Source**: `pydantic_ai/capabilities/process_event_stream.py`  
**Export**: `from pydantic_ai.capabilities import ProcessEventStream`

`ProcessEventStream` wraps the agent's event stream so a user-provided handler or async-generator processor sees every `AgentStreamEvent` emitted during model streaming and tool execution. Two forms exist: a **handler** (`async def` returning `None`) that observes events without modifying them; and a **processor** (async generator) that can add, drop, or transform events for downstream consumers.

```python
# Key signature verified from source:

@dataclass
class ProcessEventStream(AbstractCapability[AgentDepsT]):
    handler: EventStreamHandlerFunc[AgentDepsT] | EventStreamProcessorFunc[AgentDepsT]
    # EventStreamHandlerFunc  = async def(ctx, stream) -> None
    # EventStreamProcessorFunc = async def(ctx, stream) -> AsyncIterator[AgentStreamEvent]
```

### 4.1 Observer Handler — Log All Events

Observe events without affecting the stream. Multiple `ProcessEventStream` capabilities tee the stream so each receives every event independently.

```python  {test="skip"}
import asyncio
import json
from datetime import datetime
from collections.abc import AsyncIterator

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent


async def audit_handler(ctx: RunContext, stream: AsyncIterator[AgentStreamEvent]) -> None:
    """Write every event to an audit log."""
    log_lines: list[str] = []
    async for event in stream:
        log_lines.append(json.dumps({
            'ts': datetime.utcnow().isoformat(),
            'event': type(event).__name__,
        }))
    # In production: write to a file, database, or observability backend
    for line in log_lines:
        print(f'[AUDIT] {line}')


agent = Agent(
    'openai:gpt-5.2',
    capabilities=[ProcessEventStream(audit_handler)],
    instructions='You are a helpful assistant.',
)

async def main() -> None:
    result = await agent.run('What is 2 + 2?')
    print(result.output)

asyncio.run(main())
```

### 4.2 Processor — Strip `ThinkingPart` from the Stream

Use the processor form (async generator) to remove thinking events before downstream consumers see them — useful when thinking tokens should not be forwarded to a frontend.

```python  {test="skip"}
import asyncio
from collections.abc import AsyncIterator

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, PartEndEvent, PartStartEvent, ThinkingPart


async def strip_thinking(
    ctx: RunContext, stream: AsyncIterator[AgentStreamEvent]
) -> AsyncIterator[AgentStreamEvent]:
    """Processor: drop all ThinkingPart start/delta/end events."""
    suppressed: set[int] = set()
    async for event in stream:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            suppressed.add(event.index)  # remember this index
            continue
        if isinstance(event, (PartDeltaEvent, PartEndEvent)) and event.index in suppressed:
            continue   # drop deltas and end-events for suppressed parts
        yield event


agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ProcessEventStream(strip_thinking)],
)

async def main() -> None:
    async with agent.run_stream('Walk me through long division of 847 by 7.') as streamed:
        async for text in streamed.stream_text(delta=True):
            print(text, end='', flush=True)
    print()

asyncio.run(main())
```

### 4.3 Tee'd Fan-Out — Latency Monitor + Audit in Parallel

Stack two `ProcessEventStream` capabilities; both handlers receive every event independently.

```python  {test="skip"}
import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, FinalResultEvent


async def latency_monitor(ctx: RunContext, stream: AsyncIterator[AgentStreamEvent]) -> None:
    """Measure time-to-first-token and total streaming duration."""
    first_token_time: float | None = None
    start = time.monotonic()
    async for event in stream:
        if first_token_time is None:
            first_token_time = time.monotonic() - start
    total = time.monotonic() - start
    ttft = f'{first_token_time:.3f}s' if first_token_time is not None else 'n/a'
    print(f'[LATENCY] TTFT={ttft}  total={total:.3f}s')


async def result_logger(ctx: RunContext, stream: AsyncIterator[AgentStreamEvent]) -> None:
    """Log when the final result arrives."""
    async for event in stream:
        if isinstance(event, FinalResultEvent):
            print(f'[RESULT] final result arrived for run {ctx.run_id}')


agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        ProcessEventStream(latency_monitor),
        ProcessEventStream(result_logger),
    ],
)

async def main() -> None:
    result = await agent.run('Tell me a short joke.')
    print(result.output)

asyncio.run(main())
```

---

## 5. `ToolSearch` + `ToolSearchToolset` — Deferred Tool Discovery

**Source**: `pydantic_ai/capabilities/_tool_search.py`, `pydantic_ai/toolsets/_tool_search.py`  
**Export**: `from pydantic_ai.capabilities import ToolSearch`

`ToolSearch` is auto-injected into every agent (zero overhead when no deferred tools exist). On providers that support native tool search (Anthropic BM25/regex, OpenAI Responses), the deferred tools are sent on the wire and the provider exposes them after discovery; on other providers, a local `search_tools` function is exposed as a regular tool. Add `ToolSearch()` explicitly only when you need to configure `strategy`, `max_results`, `tool_description`, or `parameter_description`.

`ToolSearchToolset` is the companion toolset wrapper that hides deferred tools and exposes the `search_tools` function; it is used internally by `ToolSearch` when building the local-fallback path. Its `enable_fallback` flag (default `True`) controls whether a local `search_tools` fallback is emitted when native-only strategies (`'bm25'`/`'regex'`) are in use.

```python
# Key signatures verified from source:

@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    strategy: ToolSearchStrategy | None = None
    # ToolSearchStrategy = 'bm25' | 'regex' | 'keywords' | ToolSearchFunc
    # None = auto: native on supporting providers, keywords elsewhere
    max_results: int = 10
    tool_description: str | None = None      # description for the model-facing search tool
    parameter_description: str | None = None # description for the 'queries' parameter

@dataclass
class ToolSearchToolset(WrapperToolset[AgentDepsT]):
    search_fn: ToolSearchFunc[AgentDepsT] | None = None
    max_results: int = 10
    tool_description: str | None = None
    parameter_description: str | None = None
    enable_fallback: bool = True    # False = named-native only, no local search_tools
```

### 5.1 Large Toolset with Auto Strategy

Mark rarely-needed tools as `defer_loading=True` so they stay hidden until the model discovers them via search — reducing the initial tool-list token cost.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f'Weather in {city}: 22°C, partly cloudy'


def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight between two cities."""
    return f'Flight booked: {origin} → {destination} on {date}'


def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking by ID."""
    return f'Booking {booking_id} cancelled'


# Frequently used: always visible
weather_tool = Tool(get_weather)

# Rarely used: hidden until discovered
flight_tool = Tool(book_flight, defer_loading=True)
cancel_tool = Tool(cancel_booking, defer_loading=True)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    tools=[weather_tool, flight_tool, cancel_tool],
    capabilities=[ToolSearch()],   # auto strategy: BM25 on Anthropic
    instructions='You are a travel assistant.',
)

async def main() -> None:
    result = await agent.run('What is the weather in Paris?')
    print(result.output)   # uses get_weather directly — no discovery needed

    result2 = await agent.run('Book me a flight from London to Tokyo for 2025-09-15.')
    print(result2.output)  # discovers book_flight via tool search first

asyncio.run(main())
```

### 5.2 Custom Search Function

Override the default keyword-overlap algorithm with a domain-specific ranking function.

```python  {test="skip"}
import asyncio
from collections.abc import Sequence

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.tools import ToolDefinition


TOOL_CATEGORIES = {
    'send_email': ['email', 'message', 'notify', 'contact'],
    'create_ticket': ['ticket', 'bug', 'issue', 'jira', 'task'],
    'query_database': ['sql', 'query', 'database', 'db', 'records'],
    'generate_report': ['report', 'chart', 'summary', 'analytics'],
    'deploy_service': ['deploy', 'release', 'ci', 'production', 'k8s'],
}


def category_search(
    ctx: RunContext,
    queries: Sequence[str],
    tools: Sequence[ToolDefinition],
) -> list[str]:
    """Rank tools by how many category keywords match the query."""
    query_words = {w.lower() for q in queries for w in q.split()}
    scores: list[tuple[int, str]] = []
    for tool in tools:
        keywords = TOOL_CATEGORIES.get(tool.name, [])
        score = sum(1 for kw in keywords if kw in query_words)
        if score > 0:
            scores.append((score, tool.name))
    scores.sort(reverse=True)
    return [name for _, name in scores]


# Build deferred tools
def send_email(to: str, subject: str, body: str) -> str: ...
def create_ticket(title: str, description: str) -> str: ...
def query_database(sql: str) -> str: ...
def generate_report(metric: str, period: str) -> str: ...
def deploy_service(service: str, environment: str) -> str: ...


agent = Agent(
    'openai:gpt-5.2',
    tools=[
        Tool(fn, defer_loading=True)
        for fn in [send_email, create_ticket, query_database, generate_report, deploy_service]
    ],
    capabilities=[ToolSearch(strategy=category_search)],
)

async def main() -> None:
    result = await agent.run('I need to file a bug report for the login page.')
    print(result.output)

asyncio.run(main())
```

### 5.3 Force Native-Only BM25 (Anthropic)

Use Anthropic's server-side BM25 search exclusively; disable the local keyword fallback.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch


def tool_a(x: str) -> str: ...
def tool_b(x: str) -> str: ...
def tool_c(x: str) -> str: ...


agent = Agent(
    'anthropic:claude-sonnet-4-6',
    tools=[Tool(fn, defer_loading=True) for fn in [tool_a, tool_b, tool_c]],
    capabilities=[
        ToolSearch(strategy='bm25'),   # native BM25 only — errors on non-Anthropic providers
    ],
)

async def main() -> None:
    result = await agent.run('Use tool_b to process "hello world".')
    print(result.output)

asyncio.run(main())
```

---

## 6. `FunctionSignature` + `FunctionParam` + `TypeSignature` + `TypeExpr` types

**Source**: `pydantic_ai/function_signature.py`  
**Export**: `from pydantic_ai.function_signature import FunctionSignature, FunctionParam, TypeSignature`

These classes power PydanticAI's CLI tool rendering. `FunctionSignature` holds a parsed function's name, parameters (as `FunctionParam`), return type, and referenced TypedDict definitions (as `TypeSignature`). The type expression hierarchy — `SimpleTypeExpr`, `UnionTypeExpr`, `GenericTypeExpr`, `LiteralTypeExpr` — covers all Python type annotation shapes. `TypeFieldSignature` holds a single TypedDict field with its type, required flag, and description.

```python
# Key signatures verified from source:

@dataclass(kw_only=True)
class FunctionSignature:
    name: str
    description: str | None = None
    params: dict[str, FunctionParam] = field(default_factory=dict)
    return_type: TypeExpr          # one of Simple/Union/Generic/Literal
    referenced_types: list[TypeSignature] = field(default_factory=list)
    is_async: bool = False
    kind: Literal['function'] = 'function'

    def render(self, body: str, *, name: str | None = None) -> str: ...

@dataclass(kw_only=True)
class FunctionParam:
    name: str
    type: TypeExpr
    default: str | None = None
    kind: Literal['param'] = 'param'

@dataclass(kw_only=True)
class TypeSignature:
    name: str
    description: str | None = None
    fields: dict[str, TypeFieldSignature] = field(default_factory=dict)
    kind: Literal['type'] = 'type'

# TypeExpr union:
TypeExpr = SimpleTypeExpr | UnionTypeExpr | GenericTypeExpr | LiteralTypeExpr
```

### 6.1 Build a `FunctionSignature` Programmatically

Construct a signature describing `async def search(query: str, max_results: int = 10) -> list[str]`.

```python
from pydantic_ai.function_signature import (
    FunctionParam,
    FunctionSignature,
    GenericTypeExpr,
    SimpleTypeExpr,
)

sig = FunctionSignature(
    name='search',
    description='Search the knowledge base and return matching document titles.',
    params={
        'query': FunctionParam(
            name='query',
            type=SimpleTypeExpr(name='str'),
        ),
        'max_results': FunctionParam(
            name='max_results',
            type=SimpleTypeExpr(name='int'),
            default='10',
        ),
    },
    return_type=GenericTypeExpr(
        base='list',
        args=[SimpleTypeExpr(name='str')],
    ),
    is_async=True,
)

rendered = sig.render(body='    ...')
print(rendered)
# async def search(query: str, max_results: int = 10) -> list[str]:
#     ...
```

### 6.2 Signature with Nested TypedDict Reference

Represent `def create_user(data: UserData) -> User` where both types are TypedDict definitions.

```python
from pydantic_ai.function_signature import (
    FunctionParam,
    FunctionSignature,
    SimpleTypeExpr,
    TypeFieldSignature,
    TypeSignature,
)

user_data_type = TypeSignature(
    name='UserData',
    description='Input data for creating a new user.',
    fields={
        'name': TypeFieldSignature(
            name='name',
            type=SimpleTypeExpr(name='str'),
            required=True,
            description='Full name of the user.',
        ),
        'email': TypeFieldSignature(
            name='email',
            type=SimpleTypeExpr(name='str'),
            required=True,
            description='Email address (must be unique).',
        ),
        'role': TypeFieldSignature(
            name='role',
            type=SimpleTypeExpr(name='str'),
            required=False,
            description='User role. Defaults to "viewer".',
        ),
    },
)

user_type = TypeSignature(
    name='User',
    description='A created user record.',
    fields={
        'id': TypeFieldSignature(name='id', type=SimpleTypeExpr(name='str'), required=True),
        'name': TypeFieldSignature(name='name', type=SimpleTypeExpr(name='str'), required=True),
        'email': TypeFieldSignature(name='email', type=SimpleTypeExpr(name='str'), required=True),
    },
)

sig = FunctionSignature(
    name='create_user',
    description='Create a new user account.',
    params={
        'data': FunctionParam(name='data', type=SimpleTypeExpr(name='UserData')),
    },
    return_type=SimpleTypeExpr(name='User'),
    referenced_types=[user_data_type, user_type],
)

print(sig.render(body='    ...'))
# def create_user(data: UserData) -> User:
#     ...
```

### 6.3 Union and Literal Type Expressions

Represent `def classify(text: str, mode: Literal['fast', 'accurate']) -> str | None`.

```python
from pydantic_ai.function_signature import (
    FunctionParam,
    FunctionSignature,
    LiteralTypeExpr,
    SimpleTypeExpr,
    UnionTypeExpr,
)

sig = FunctionSignature(
    name='classify',
    description='Classify text as positive or negative sentiment.',
    params={
        'text': FunctionParam(
            name='text',
            type=SimpleTypeExpr(name='str'),
        ),
        'mode': FunctionParam(
            name='mode',
            type=LiteralTypeExpr(values=['fast', 'accurate']),
            default="'fast'",
        ),
    },
    return_type=UnionTypeExpr(
        members=[SimpleTypeExpr(name='str'), SimpleTypeExpr(name='None')]
    ),
)

print(sig.render(body='    ...'))
# def classify(text: str, mode: Literal['fast', 'accurate'] = 'fast') -> str | None:
#     ...
```

---

## 7. `WrapperAgent` + `AbstractAgent` — Agent Delegation and Middleware

**Source**: `pydantic_ai/agent/wrapper.py`, `pydantic_ai/agent/abstract.py`  
**Export**: `from pydantic_ai.agent import WrapperAgent, AbstractAgent`

`WrapperAgent` is the transparent delegation base. Every `AbstractAgent` property and method delegates to `self.wrapped`. Override specific methods to inject middleware: authentication, rate limiting, audit logging, routing between specialised agents.

```python
# Key signature verified from source:

class WrapperAgent(AbstractAgent[AgentDepsT, OutputDataT]):
    def __init__(self, wrapped: AbstractAgent[AgentDepsT, OutputDataT]): ...

    # All properties delegate to self.wrapped:
    model, name, description, deps_type, output_type,
    event_stream_handler, root_capability, toolsets

    # All run methods also delegate:
    async def run(self, ...) -> AgentRunResult: ...
    def run_sync(self, ...) -> AgentRunResult: ...
    async def run_stream(self, ...) -> StreamedRunResult: ...
```

### 7.1 Audit-Logging Wrapper

Intercept every run, log metadata, and forward to the inner agent.

```python  {test="skip"}
import asyncio
import time
from typing import Any, Sequence

from pydantic_ai import Agent
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.result import AgentRunResult
from pydantic_ai.settings import ModelSettings


class AuditedAgent(WrapperAgent):
    """Wraps any agent to log run metadata."""

    def __init__(self, wrapped, audit_log: list[dict[str, Any]]) -> None:
        super().__init__(wrapped)
        self.audit_log = audit_log

    async def run(
        self,
        user_prompt: str | Sequence[Any],
        *,
        model_settings: ModelSettings | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        start = time.monotonic()
        try:
            result = await self.wrapped.run(
                user_prompt,
                model_settings=model_settings,
                message_history=message_history,
                **kwargs,
            )
            self.audit_log.append({
                'prompt': str(user_prompt)[:100],
                'duration_s': round(time.monotonic() - start, 3),
                'tokens': result.usage().total_tokens,
                'ok': True,
            })
            return result
        except Exception as exc:
            self.audit_log.append({
                'prompt': str(user_prompt)[:100],
                'duration_s': round(time.monotonic() - start, 3),
                'ok': False,
                'error': str(exc),
            })
            raise


inner = Agent('openai:gpt-5.2', instructions='You are a helpful assistant.')
log: list[dict[str, Any]] = []
agent = AuditedAgent(wrapped=inner, audit_log=log)

async def main() -> None:
    result = await agent.run('What is the capital of France?')
    print(result.output)
    print('Audit log:', log)

asyncio.run(main())
```

### 7.2 Router — Dispatch to Specialised Agents

Route each request to a specialised agent based on keyword analysis.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.result import AgentRunResult


coding_agent = Agent('anthropic:claude-sonnet-4-6', instructions='You are an expert Python programmer.')
writing_agent = Agent('openai:gpt-5.2', instructions='You are a professional copywriter.')
math_agent = Agent('openai:o4-mini', instructions='You are a mathematics tutor.')

CODE_KEYWORDS = {'code', 'function', 'class', 'python', 'bug', 'debug', 'implement'}
MATH_KEYWORDS = {'solve', 'calculate', 'equation', 'integral', 'derivative', 'matrix'}


class RouterAgent(WrapperAgent):
    """Routes requests to specialised agents based on keyword matching.

    Note: only plain string prompts are supported; non-string inputs are
    forwarded directly to the default (writing) agent without routing.
    """

    def __init__(self, wrapped) -> None:
        super().__init__(wrapped)

    async def run(self, user_prompt, **kwargs) -> AgentRunResult:
        if isinstance(user_prompt, str):
            words = set(user_prompt.lower().split())
            if words & CODE_KEYWORDS:
                return await coding_agent.run(user_prompt, **kwargs)
            if words & MATH_KEYWORDS:
                return await math_agent.run(user_prompt, **kwargs)
        return await writing_agent.run(user_prompt, **kwargs)


router = RouterAgent(wrapped=writing_agent)

async def main() -> None:
    result1 = await router.run('Write a Python function that sorts a list of dicts by key.')
    print('Coding:', result1.output[:80])

    result2 = await router.run('Solve the quadratic equation x² - 5x + 6 = 0.')
    print('Math:', result2.output[:80])

    result3 = await router.run('Write a product description for noise-cancelling headphones.')
    print('Writing:', result3.output[:80])

asyncio.run(main())
```

---

## 8. `PendingMessage` + `PendingMessageDrainCapability` — Message Queue Internals

**Source**: `pydantic_ai/_enqueue.py`, `pydantic_ai/capabilities/_pending_messages.py`  
**Export**: `from pydantic_ai._enqueue import PendingMessage` (internal; use via `RunContext.enqueue`)

`PendingMessage` wraps one or more `ModelMessage`s queued for injection into the conversation. The `priority` field controls delivery timing: `'asap'` injects into the very next model request (or forces a new request if the agent would otherwise terminate), while `'when_idle'` injects only when the agent is about to terminate and all `'asap'` messages have been drained.

`PendingMessageDrainCapability` is auto-injected into every agent at the `'outermost'` position. It runs `before_model_request` to drain `'asap'` messages and `after_node_run` to redirect idle termination when messages remain.

```python
# Key signatures verified from source:

@dataclass
class PendingMessage:
    messages: list[ModelMessage]
    priority: Literal['asap', 'when_idle'] = 'asap'

    @classmethod
    def from_content(
        cls,
        *content: EnqueueContent,
        priority: Literal['asap', 'when_idle'] = 'asap',
    ) -> PendingMessage | None: ...
    # Returns None for empty call (no-op rather than error)
    # Raises UserError if assembled messages don't end in a ModelRequest


class PendingMessageDrainCapability(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')  # always runs last/first
```

### 8.1 Tool Enqueues a Follow-Up Message (`'asap'`)

A tool injects additional context into the conversation immediately after it runs — the model receives it on the very next request without returning to the caller first.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, RunContext


async def fetch_profile(ctx: RunContext, user_id: str) -> str:
    """Fetch a user profile and immediately inject enrichment context."""
    profile = f'User {user_id}: Alice Smith, role=admin, tier=enterprise'

    # Inject a system-level context message that arrives before the model's next turn
    ctx.enqueue(
        f'[CONTEXT] Retrieved profile: {profile}',
        priority='asap',
    )
    return f'Profile fetched for {user_id}'


agent = Agent(
    'openai:gpt-5.2',
    tools=[fetch_profile],
    instructions='Assist with user profile queries.',
)

async def main() -> None:
    result = await agent.run('Fetch profile for user U-42 and tell me their tier.')
    print(result.output)

asyncio.run(main())
```

### 8.2 Same-Run Idle Injection (`'when_idle'`)

Queue a message to be processed when the agent reaches idle state within the same `agent.run()` call — the run continues with another model request before returning to the caller.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent, RunContext


def analyse_sentiment(ctx: RunContext, text: str) -> str:
    """Analyse text sentiment and queue a follow-up for the idle phase."""
    sentiment = 'positive' if 'great' in text.lower() else 'neutral'

    # Injected when the run reaches idle — same agent.run(), next model request
    ctx.enqueue(
        f'[PREV_ANALYSIS] Sentiment was {sentiment}.',
        priority='when_idle',
    )
    return f'Sentiment: {sentiment}'


agent = Agent(
    'openai:gpt-5.2',
    tools=[analyse_sentiment],
    instructions='Analyse user messages and track sentiment.',
)

async def main() -> None:
    result = await agent.run('This product is great, I love it!')
    print(result.output)
    print('All messages:', len(result.all_messages()))

asyncio.run(main())
```

### 8.3 `AgentRun.enqueue` — Inject from Outside the Tool

Enqueue a message from the orchestration layer (outside a tool call) using `AgentRun.enqueue`.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent


agent = Agent(
    'openai:gpt-5.2',
    instructions='You are a research assistant that refines queries iteratively.',
)

async def main() -> None:
    enqueued = False
    async with agent.iter('Summarise the main themes in Moby Dick.') as agent_run:
        async for node in agent_run:
            # After the first model response, inject additional instructions
            if hasattr(node, 'model_response') and not enqueued:
                agent_run.enqueue(
                    'Also note the significance of the white whale as a symbol.',
                    priority='asap',
                )
                enqueued = True  # only enqueue once; loop continues to process it
    result = agent_run.result
    print(result.output)

asyncio.run(main())
```

---

## 9. `LangChainToolset` + `LangChainTool` — LangChain Bridge

**Source**: `pydantic_ai/ext/langchain.py`  
**Export**: `from pydantic_ai.ext.langchain import LangChainToolset, LangChainTool`

`LangChainTool` is a structural Protocol that any LangChain tool satisfies via duck-typing: it must expose `.args`, `.get_input_jsonschema()`, `.name`, `.description`, and `.run()`. `LangChainToolset` extends `FunctionToolset` and wraps each LangChain tool using the `tool_from_langchain` adapter, which bridges LangChain's schema format to PydanticAI's `ToolDefinition`.

```python
# Key signatures verified from source:

class LangChainTool(Protocol):
    @property
    def args(self) -> dict[str, JsonSchemaValue]: ...
    def get_input_jsonschema(self) -> JsonSchemaValue: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def run(self, *args: Any, **kwargs: Any) -> str: ...

class LangChainToolset(FunctionToolset):
    def __init__(
        self,
        tools: list[LangChainTool],
        *,
        id: str | None = None,
    ): ...
```

### 9.1 Migrate LangChain Tools to a PydanticAI Agent

Wrap existing LangChain community tools in a `LangChainToolset` and use them in a typed PydanticAI agent.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset

# Import LangChain tools (requires langchain-community)
# from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper

# For illustration, we mock LangChain tools using the Protocol shape:
class MockSearchTool:
    name = 'duckduckgo_search'
    description = 'Search the web using DuckDuckGo. Input should be a search query.'

    @property
    def args(self):
        return {'query': {'title': 'Query', 'type': 'string', 'description': 'Search query.'}}

    def get_input_jsonschema(self):
        return {
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': 'Search query.'}},
            'required': ['query'],
        }

    def run(self, query: str = '', **kwargs) -> str:
        return f'[DuckDuckGo] Top results for "{query}": ...'


class MockWikiTool:
    name = 'wikipedia'
    description = 'Search Wikipedia. Input should be a search query.'

    @property
    def args(self):
        return {'query': {'title': 'Query', 'type': 'string', 'description': 'Wikipedia query.'}}

    def get_input_jsonschema(self):
        return {
            'type': 'object',
            'properties': {'query': {'type': 'string', 'description': 'Wikipedia query.'}},
            'required': ['query'],
        }

    def run(self, query: str = '', **kwargs) -> str:
        return f'[Wikipedia] Summary for "{query}": ...'


lc_toolset = LangChainToolset([MockSearchTool(), MockWikiTool()])

agent = Agent(
    'openai:gpt-5.2',
    toolsets=[lc_toolset],
    instructions='You are a research assistant. Use available tools to answer questions.',
)

async def main() -> None:
    result = await agent.run('Who invented the telephone and what does Wikipedia say about them?')
    print(result.output)

asyncio.run(main())
```

### 9.2 Mixed PydanticAI + LangChain Toolsets

Combine LangChain tools with native PydanticAI function tools in the same agent.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset
from pydantic_ai.toolsets import FunctionToolset


# Native PydanticAI tool
native_toolset = FunctionToolset()

@native_toolset.tool_plain
def get_current_date() -> str:
    """Return the current date in ISO format."""
    from datetime import date
    return date.today().isoformat()


# Reuse the mock LangChain tools from example 9.1
class MockSearchTool:
    name = 'web_search'
    description = 'Search the web for current information.'

    @property
    def args(self): return {'query': {'type': 'string', 'description': 'Search query.'}}
    def get_input_jsonschema(self):
        return {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}
    def run(self, query: str = '', **kwargs) -> str:
        return f'Web results for "{query}": ...'


lc_toolset = LangChainToolset([MockSearchTool()])

agent = Agent(
    'openai:gpt-5.2',
    toolsets=[native_toolset, lc_toolset],   # mix freely
    instructions='Answer questions using tools. Always include the current date.',
)

async def main() -> None:
    result = await agent.run('What is happening in AI research today?')
    print(result.output)

asyncio.run(main())
```

### 9.3 LangChain Tools with Semantic Search

Use `ToolSearch` to let the model discover relevant LangChain tools from a large toolset rather than seeing all of them at once. `ToolSearch` activates automatically when tools are registered with `defer_loading=True`; add it explicitly here to set the search strategy.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.ext.langchain import LangChainToolset


class HeavyDataTool:
    """A LangChain-compatible tool that queries a large dataset."""
    name = 'query_data_warehouse'
    description = 'Query the corporate data warehouse for business metrics.'

    @property
    def args(self): return {'query': {'type': 'string', 'description': 'SQL-like query.'}}
    def get_input_jsonschema(self):
        return {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}
    def run(self, query: str = '', **kwargs) -> str:
        return f'DWH results for "{query}": revenue=£1.2M, units=4500'


lc_toolset = LangChainToolset([HeavyDataTool()])

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[lc_toolset],
    # Explicit ToolSearch configures BM25 strategy; auto-activated for deferred tools
    capabilities=[ToolSearch(strategy='bm25', max_results=5)],
    instructions='You are a business intelligence assistant.',
)

async def main() -> None:
    result = await agent.run('What were our revenue figures last quarter?')
    print(result.output)

asyncio.run(main())
```

---

## 10. `AgentSpec` YAML Composition — Full Declarative Agent Configuration

**Source**: `pydantic_ai/agent/spec.py`  
**Export**: `from pydantic_ai.agent import AgentSpec`

`AgentSpec` is a Pydantic `BaseModel` that deserialises an agent from a YAML/JSON file or dict. Fields mirror the `Agent(...)` constructor: `model`, `name`, `description`, `instructions`, `deps_schema`, `output_schema`, `model_settings`, `retries`, `end_strategy`, `tool_timeout`, `metadata`, and `capabilities`. The `capabilities` list accepts `CapabilitySpec` short-forms — bare strings, single-arg dicts, or kwarg dicts — resolved against a type registry.

```python
# Key signature verified from source:

class AgentSpec(BaseModel):
    model: str | None = None
    name: str | None = None
    description: TemplateStr[Any] | str | None = None
    instructions: TemplateStr[Any] | str | list[...] | None = None
    deps_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    model_settings: dict[str, Any] | None = None
    retries: int | AgentRetries | None = None
    end_strategy: EndStrategy = 'graceful'
    tool_timeout: float | None = None
    metadata: dict[str, Any] | None = None
    capabilities: list[CapabilitySpec] = []

    @classmethod
    def from_file(cls, path: Path | str, fmt: Literal['yaml', 'json'] | None = None) -> AgentSpec: ...
    def to_file(self, path: Path | str, fmt: Literal['yaml', 'json'] | None = None) -> None: ...
    # Note: AgentSpec has no build() method.
    # Turn a spec into an Agent via Agent.from_file(path) or Agent.from_spec(spec, custom_capability_types=[...])
```

### 10.1 Load an Agent from YAML

Define an agent entirely in YAML and load it at runtime — good for per-environment configuration.

```python  {test="skip"}
# agent_config.yaml
# ---
# model: openai:gpt-5.2
# name: support-bot
# instructions: |
#   You are a customer support agent for AcmeCorp.
#   Always be polite and escalate complex issues.
# retries: 3
# end_strategy: graceful
# capabilities:
#   - WebSearch:
#       search_context_size: low
#   - Thinking:
#       effort: low

import asyncio
from pydantic_ai import Agent

# Agent.from_file() reads the YAML and constructs the Agent in one step
agent = Agent.from_file('agent_config.yaml')

async def main() -> None:
    result = await agent.run('My order has not arrived in 14 days, what should I do?')
    print(result.output)

asyncio.run(main())
```

### 10.2 Build an `AgentSpec` Programmatically and Serialise

Create an `AgentSpec` in code, then save it as YAML for version control.

```python  {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.agent import AgentSpec

spec = AgentSpec(
    model='anthropic:claude-sonnet-4-6',
    name='research-agent',
    instructions='You are a research assistant. Cite sources when possible.',
    model_settings={'max_tokens': 2048, 'temperature': 0.3},
    retries=2,
    capabilities=[
        {'WebSearch': {'search_context_size': 'high'}},  # kwarg short-form
        {'WebFetch': {'local': True}},
        'Thinking',                                       # bare string short-form
    ],
)

# Save to YAML for version control
spec.to_file('research_agent.yaml')

# Build an Agent from the spec
agent = Agent.from_spec(spec)

async def main() -> None:
    result = await agent.run('What are the latest breakthroughs in protein folding?')
    print(result.output)

asyncio.run(main())
```

### 10.3 Custom Capability Registry for Private Capabilities

Register your own capabilities so they can be referenced by name in YAML configuration files.

```python  {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.agent import AgentSpec
from pydantic_ai.capabilities.abstract import AbstractCapability


@dataclass
class CompanyStyleGuide(AbstractCapability):
    """Injects company-specific style instructions into every request."""

    brand_name: str = 'AcmeCorp'
    tone: str = 'professional'

    def get_instructions(self) -> str:
        return (
            f'Always refer to us as {self.brand_name}. '
            f'Maintain a {self.tone} tone in all responses.'
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'CompanyStyleGuide'


# YAML would look like:
# capabilities:
#   - CompanyStyleGuide:
#       brand_name: AcmeCorp
#       tone: friendly

spec = AgentSpec(
    model='openai:gpt-5.2',
    name='branded-assistant',
    instructions='Help customers with their enquiries.',
    capabilities=[
        {'CompanyStyleGuide': {'brand_name': 'AcmeCorp', 'tone': 'friendly'}},
    ],
)

# Pass custom capability classes so the spec resolver can find them by name
agent = Agent.from_spec(spec, custom_capability_types=[CompanyStyleGuide])

async def main() -> None:
    result = await agent.run('What services does your company offer?')
    print(result.output)

asyncio.run(main())
```

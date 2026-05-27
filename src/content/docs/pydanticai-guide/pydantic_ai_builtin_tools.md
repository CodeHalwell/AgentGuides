---
title: "PydanticAI: Built-in / Native Tools & Common Tools"
description: "Provider-native tools (WebSearchTool, CodeExecutionTool, ImageGenerationTool, …) and local common tools (web_fetch_tool, duckduckgo_search_tool) — source-verified coverage of both families."
framework: pydanticai
language: python
---

# Built-in / Native Tools & Common Tools

Verified against **pydantic-ai==1.103.0** — source modules: `pydantic_ai.native_tools`, `pydantic_ai.common_tools`.

Native tools execute inside the LLM provider's infrastructure, not in your Python process. PydanticAI forwards a typed config to the provider and streams the results back as `NativeToolCallPart` / `NativeToolReturnPart`.

## Migration note — `builtin_tools` → `capabilities`

`Agent(builtin_tools=[...])` is **deprecated** in 1.101.0. The new API uses `capabilities=[...]` with provider-adaptive capability classes that fall back gracefully when a provider doesn't support the native feature:

```python
# OLD (deprecated — still works, emits PydanticAIDeprecationWarning)
from pydantic_ai import Agent, WebSearchTool
agent = Agent('anthropic:claude-sonnet-4-6', builtin_tools=[WebSearchTool(max_uses=3)])

# NEW — preferred
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[WebSearch(max_uses=3)])
```

For raw, provider-specific tool objects (no fallback logic), use `NativeTool`:

```python
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[NativeTool(WebSearchTool(max_uses=3))],
)
```

## Minimal runnable example

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[WebSearch(max_uses=3)],
)
result = agent.run_sync('What happened in AI news this week? Cite sources.')
print(result.output)
```

## The catalogue

All are exported from `pydantic_ai` directly (also at `pydantic_ai.builtin_tools`):

| Tool                  | Providers that support it                            | Requires config | Status              |
| --------------------- | ---------------------------------------------------- | --------------- | ------------------- |
| `WebSearchTool`       | Anthropic, OpenAI Responses, Groq, Google, xAI, OpenRouter | no        | GA                  |
| `WebFetchTool`        | Anthropic, Google                                    | no              | GA (replaces `UrlContextTool`) |
| `CodeExecutionTool`   | Anthropic, OpenAI Responses, Google                  | no              | GA                  |
| `ImageGenerationTool` | OpenAI Responses, Google                             | no              | GA                  |
| `FileSearchTool`      | OpenAI Responses, Google (Gemini Files), xAI         | yes (`file_store_ids`) | GA              |
| `MemoryTool`          | Anthropic                                            | yes (provider account) | GA              |
| `MCPServerTool`       | OpenAI Responses, Anthropic, xAI                     | yes (`id`, `url`)| GA                 |
| `XSearchTool`         | xAI only                                             | no              | GA                  |
| `UrlContextTool`      | Anthropic, Google (deprecated alias of `WebFetchTool`) | no            | **deprecated — use `WebFetchTool`** |

A tool listed as GA may still have provider-specific constraints (e.g. `blocked_domains` is Anthropic-only on `WebSearchTool`). Each class's docstring catalogs the per-provider support for every argument.

## `WebSearchTool` / `WebSearch` capability

Use the `WebSearch` capability for provider-adaptive search. Pass `WebSearchTool` args directly as keyword arguments:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.native_tools import WebSearchUserLocation

agent = Agent(
    'openai-responses:gpt-5.2',   # OpenAI Responses API
    capabilities=[WebSearch(
        search_context_size='high',
        max_uses=5,
        allowed_domains=['docs.python.org', 'peps.python.org'],
        user_location=WebSearchUserLocation(country='US', city='San Francisco'),
    )],
)
```

Or use `NativeTool(WebSearchTool(...))` directly to skip the fallback logic:

```python
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchUserLocation

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[NativeTool(WebSearchTool(
        search_context_size='high',
        max_uses=5,
        allowed_domains=['docs.python.org', 'peps.python.org'],
        user_location=WebSearchUserLocation(country='US', city='San Francisco'),
    ))],
)
```

Fields (`builtin_tools/__init__.py:90`):

| Field                 | Default    | Supported by                                |
| --------------------- | ---------- | ------------------------------------------- |
| `search_context_size` | `'medium'` | OpenAI Responses, OpenRouter                |
| `user_location`       | `None`     | Anthropic, OpenAI Responses                 |
| `blocked_domains`     | `None`     | Anthropic, Groq, xAI                        |
| `allowed_domains`     | `None`     | Anthropic, Groq, OpenAI Responses, xAI      |
| `max_uses`            | `None`     | Anthropic                                   |

Anthropic forbids both `blocked_domains` and `allowed_domains` at the same time.

## `WebFetchTool` (replaces `UrlContextTool`)

```python
from pydantic_ai import Agent, WebFetchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[NativeTool(WebFetchTool(max_uses=3, enable_citations=True))],
)
```

Fields:

- `max_uses` — Anthropic only.
- `allowed_domains` / `blocked_domains` — Anthropic only (mutually exclusive).
- `enable_citations` — Anthropic, adds citation metadata to the returned text.
- `max_content_tokens` — Anthropic, caps the fetched payload size.

`UrlContextTool` is a deprecated subclass kept for backward compatibility; its `kind='url_context'` lets old persisted sessions deserialize. Update to `WebFetchTool` in new code.

## `CodeExecutionTool`

```python
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[NativeTool(CodeExecutionTool())],
)
result = agent.run_sync('Compute the 50th Fibonacci number.')
```

Runs in the provider's sandbox (Anthropic's container, OpenAI's Python interpreter, Google's Gemini code execution). The tool has **no configurable fields** in 1.85 — the only knob is model-level settings.

## `ImageGenerationTool`

```python
from pydantic_ai import Agent, ImageGenerationTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[NativeTool(ImageGenerationTool(
        size='1024x1024',
        quality='high',
        output_format='png',
    ))],
)
result = agent.run_sync('Generate a cover image for a book about Paris.')
```

Fields (`builtin_tools/__init__.py:364`):

| Field               | Type / Default                        | Notes                                                |
| ------------------- | ------------------------------------- | ---------------------------------------------------- |
| `background`        | `'transparent' | 'opaque' | 'auto'`   | OpenAI Responses; transparent only for png/webp      |
| `input_fidelity`    | `'high' | 'low' | None`               | OpenAI Responses (facial features etc.)              |
| `moderation`        | `'auto' | 'low'`                      | OpenAI Responses                                     |
| `output_compression`| `int | None`                          | OpenAI jpeg/webp; Google jpeg                        |
| `output_format`     | `'png' | 'webp' | 'jpeg' | None`      | —                                                    |
| `partial_images`    | `0-3`                                 | OpenAI streaming mode                                |
| `quality`           | `'low' | 'medium' | 'high' | 'auto'`  | OpenAI Responses                                     |
| `size`              | `'1024x1024'...'4K'` literal          | See class for the accepted set per provider          |
| `aspect_ratio`      | `ImageAspectRatio | None`             | Google, OpenAI (mapped to supported sizes)           |

Generated images arrive as `FilePart`s in the response, then as `BinaryImage` in the final output when `output_type` supports images.

## `FileSearchTool`

```python
from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[NativeTool(FileSearchTool(file_store_ids=['vs_abc123', 'vs_xyz789']))],
)
```

- OpenAI: `file_store_ids` are vector-store IDs.
- Google (Gemini): file search store names (uploaded via Files API).
- xAI: collection IDs.

The provider handles embedding, chunking, and retrieval — you never touch vectors. For bring-your-own-retrieval, use `Embedder` + your own vector DB.

## `MemoryTool`

Anthropic-only. Enables persistent memory across runs managed by Anthropic's infrastructure:

```python
from pydantic_ai import Agent, MemoryTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[NativeTool(MemoryTool())],
)
```

## `MCPServerTool`

Built-in tool that asks the provider to call out to a remote MCP server. Different from `pydantic_ai.mcp.MCPServerStdio` / `MCPServerSSE` / `MCPServerStreamableHTTP`, which run the client locally.

```python
from pydantic_ai import Agent, MCPServerTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[NativeTool(MCPServerTool(
        id='docs-mcp',
        url='https://mcp.example.com/docs',
        authorization_token='Bearer ...',
        allowed_tools=['search', 'fetch'],
        headers={'x-org': 'acme'},
    ))],
)
```

Fields:

- `id` (required), `url` (required)
- `authorization_token`, `description`, `allowed_tools`, `headers`

Use this when you want the provider to handle the MCP round-trip (lower latency, no local subprocess). Use the `pydantic_ai.mcp` classes when you want to run the MCP client yourself (full control, supports `stdio`).

## `XSearchTool` (xAI only)

```python
from datetime import datetime
from pydantic_ai import Agent, XSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'xai:grok-3-latest',
    capabilities=[NativeTool(XSearchTool(
        allowed_x_handles=['pydantic'],
        from_date=datetime(2026, 1, 1),
        include_output=True,
    ))],
)
```

Fields (with validation in `__post_init__`):

- `allowed_x_handles` / `excluded_x_handles` (max 10, mutually exclusive)
- `from_date` / `to_date` (naive datetimes = UTC)
- `enable_image_understanding`, `enable_video_understanding`
- `include_output` — emit raw search results as `BuiltinToolReturnPart` (otherwise the model only uses them internally).

## Inspecting native tool traffic

Native tool calls appear as `NativeToolCallPart` and `NativeToolReturnPart` in the message history, distinct from `ToolCallPart` / `ToolReturnPart` (which are for your function tools). Streaming events fire as `NativeToolCallEvent` / `NativeToolResultEvent`.

```python
from pydantic_ai.messages import NativeToolCallPart

for m in result.all_messages():
    for p in m.parts:
        if isinstance(p, NativeToolCallPart):
            print(p.tool_name, p.args_as_json_str())
```

## Mixing native and function tools

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import WebSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[WebSearch(max_uses=2)],
)

@agent.tool
def internal_lookup(ctx: RunContext[None], sku: str) -> dict:
    return db.get(sku)
```

The model sees both flavours. Native tool calls don't consume your function-tool retry budget.

## Gotchas

- **`builtin_tools=` is deprecated.** Migrate to `capabilities=[NativeTool(...)]` or the provider-adaptive wrappers (`WebSearch`, `WebFetch`, `ImageGeneration`, `MCP`).
- **`UrlContextTool` is deprecated.** Swap to `WebFetchTool`; the serialised `kind='url_context'` still deserialises via the deprecated subclass for backward-compat.
- **`defer_model_check=False`** (default) validates the native tool list against the model at construction. If you swap models via `agent.override(model=...)` to one that doesn't support the tools, you'll hit `UserError` at run time.
- **Field support varies by provider** even on "supported" tools. The docstrings list per-field support; unsupported fields are silently ignored by some providers and raise on others.
- **Native tool usage is counted against provider quotas, not PydanticAI's `tool_calls_limit`**. Use `max_uses` where available.
- **`MCPServerTool` vs. `pydantic_ai.mcp`**: the former is the provider's remote MCP client; the latter is a local MCP client. Don't register the same server twice under both — you'd get duplicate tools.

## Patterns

### 1. Research agent with web search + fetch

```python
from pydantic_ai import Agent, WebFetchTool
from pydantic_ai.capabilities import WebSearch, NativeTool

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[
    WebSearch(max_uses=5),
    NativeTool(WebFetchTool(max_uses=10, enable_citations=True)),
])
```

### 2. Constrain searches to a domain

```python
from pydantic_ai.capabilities import WebSearch

WebSearch(allowed_domains=['docs.company.com'])
```

### 3. Code-runner with a structured output contract

```python
from pydantic import BaseModel
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

class CalcResult(BaseModel):
    answer: float
    reasoning: str

agent = Agent('openai-responses:gpt-5.2',
              output_type=CalcResult,
              capabilities=[NativeTool(CodeExecutionTool())])
```

### 4. RAG via `FileSearchTool` + fallback to local embeddings

```python
from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('openai-responses:gpt-5.2',
              capabilities=[NativeTool(FileSearchTool(file_store_ids=['vs_prod']))])

@agent.tool
async def fallback_search(ctx, query: str) -> list[str]:
    # Only called if the built-in search misses
    return await embedder_search(query)
```

### 5. Inspect what the model actually searched

```python
async for event in agent.run_stream_events(prompt):
    from pydantic_ai.messages import NativeToolCallEvent, NativeToolResultEvent
    if isinstance(event, NativeToolCallEvent):
        print(f'[native call] {event.part.tool_name}({event.part.args_as_json_str()})')
    if isinstance(event, NativeToolResultEvent):
        print(f'[native result] {event.result.content[:200]}')
```

---

## Common Tools — local Python tools (`pydantic_ai.common_tools`)

**Common tools** are regular PydanticAI `Tool` objects that run in your Python process — not on the provider's infrastructure. They ship with the `pydantic-ai` package and are ready to use without API keys or provider configuration.

> **Native vs. common tools — the key distinction**
>
> | | Native tools | Common tools |
> |--|--|--|
> | Runs in | Provider's infrastructure | Your Python process |
> | Requires provider support | Yes (varies by provider) | No — works with any model |
> | Configured via | `capabilities=[NativeTool(...)]` | `toolsets=[FunctionToolset([tool])]` or `tools=[tool]` |
> | Examples | `WebSearchTool`, `CodeExecutionTool` | `web_fetch_tool`, `duckduckgo_search_tool` |

---

### `web_fetch_tool` — SSRF-protected URL fetching

**Source**: `common_tools/web_fetch.py`  
**Install**: `pip install "pydantic-ai-slim[web-fetch]"` (installs `markdownify` + `httpx`)

`web_fetch_tool` fetches a URL, converts HTML to Markdown, and returns a `WebFetchResult` dict (or `BinaryContent` for PDFs/images). It uses `pydantic_ai._ssrf.safe_download` to block requests to private IP ranges by default.

#### Minimal example

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[web_fetch_tool()],
)
result = agent.run_sync('Summarise the content of https://docs.pydantic.dev/latest/')
print(result.output)
```

#### Constructor (source: `common_tools/web_fetch.py:web_fetch_tool`)

```python
web_fetch_tool(
    *,
    max_content_length: int | None = 50_000,
    allow_local_urls: bool = False,
    timeout: int = 30,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> Tool
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_content_length` | `50_000` | Max characters returned (~12,500 tokens). `None` = no limit |
| `allow_local_urls` | `False` | Set `True` in dev to allow `localhost`. **Never** in production |
| `timeout` | `30` | HTTP request timeout in seconds |
| `allowed_domains` | `None` | Whitelist — raises `ModelRetry` for other domains |
| `blocked_domains` | `None` | Blacklist — raises `ModelRetry` for these domains |
| `headers` | `None` | Extra HTTP headers. Overrides the default `Accept: text/markdown` header if `Accept` is provided |

#### Default `Accept` header — markdown shortcut

`web_fetch_tool` sends `Accept: text/markdown` by default. Servers that support it (Cloudflare Workers, Vercel, Mintlify) return Markdown directly, reducing token usage and improving quality. HTML is converted via `markdownify` when the server doesn't serve Markdown.

#### Domain filtering — allowlist and blocklist

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

# Only allow fetching from trusted documentation sites
safe_fetcher = web_fetch_tool(
    allowed_domains=['docs.pydantic.dev', 'docs.python.org', 'ai.pydantic.dev'],
    max_content_length=30_000,
    timeout=15,
)

agent = Agent('openai:gpt-4o', tools=[safe_fetcher])
# Model can only fetch from the listed domains; any other URL raises ModelRetry
```

```python
# Block specific domains (social media, tracking sites, etc.)
filtered_fetcher = web_fetch_tool(
    blocked_domains=['facebook.com', 'twitter.com', 'x.com', 'linkedin.com'],
)
```

Note: `allowed_domains` and `blocked_domains` are **mutually exclusive per call** — the SSRF protection layer enforces exact hostname matching.

#### Binary content handling

For non-text URLs (PDFs, images, audio), `web_fetch_tool` returns a `BinaryContent` object rather than `WebFetchResult`. The model receives the binary directly (if the model supports multimodal input):

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool, WebFetchResult
from pydantic_ai.messages import BinaryContent

agent = Agent(
    'anthropic:claude-sonnet-4-6',   # Supports PDF reading
    tools=[web_fetch_tool(max_content_length=None)],  # no truncation for PDFs
)

async def main():
    result = await agent.run(
        'Read https://example.com/report.pdf and summarise the key findings.'
    )
    print(result.output)

asyncio.run(main())
```

#### Full configuration example — research agent

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

research_fetcher = web_fetch_tool(
    max_content_length=80_000,    # allow larger pages for research
    allow_local_urls=False,        # keep SSRF protection on
    timeout=60,                    # longer timeout for slow sites
    headers={
        'User-Agent': 'ResearchBot/1.0 (pydantic-ai)',
        'Accept-Language': 'en-US,en;q=0.9',
    },
    blocked_domains=['ads.google.com', 'doubleclick.net'],
)

research_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    tools=[research_fetcher],
    system_prompt=(
        'You are a research assistant. When you fetch a URL, extract the most relevant '
        'information, cite the source URL, and keep your summary concise.'
    ),
)

result = research_agent.run_sync(
    'Fetch https://peps.python.org/pep-0703/ and explain the main proposal.'
)
print(result.output)
```

#### `WebFetchResult` TypedDict (source-verified)

```python
from pydantic_ai.common_tools.web_fetch import WebFetchResult

# TypedDict fields:
# url: str       — the URL that was fetched
# title: str     — page <title>, or '' if not found
# content: str   — page content converted to Markdown (or raw text for JSON/plain-text)
```

---

### `duckduckgo_search_tool` — local DuckDuckGo search

**Source**: `common_tools/duckduckgo.py`  
**Install**: `pip install "pydantic-ai-slim[duckduckgo]"` (installs `ddgs`)

`duckduckgo_search_tool` wraps the `ddgs` (DuckDuckGo Search) library as a PydanticAI tool. It runs in a thread pool (`anyio.to_thread.run_sync`) to keep the async event loop unblocked.

#### Minimal example

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[duckduckgo_search_tool(max_results=5)],
)
result = agent.run_sync('What are the latest Python 3.13 features?')
print(result.output)
```

#### Constructor

```python
duckduckgo_search_tool(
    duckduckgo_client: DDGS | None = None,
    max_results: int | None = None,
) -> Tool
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `duckduckgo_client` | `None` | Pass a pre-configured `DDGS()` instance to share across calls |
| `max_results` | `None` | Max results to return. `None` = results from the first response only |

#### `DuckDuckGoResult` TypedDict (source-verified)

Each result is a dict with these fields:

```python
from pydantic_ai.common_tools.duckduckgo import DuckDuckGoResult

# TypedDict fields:
# title: str   — result title
# href: str    — result URL
# body: str    — result snippet / body text
```

#### Shared DDGS client (reuse session)

```python
from ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# Shared client — connection pool is reused across calls
ddgs_client = DDGS()

agent = Agent(
    'openai:gpt-4o',
    tools=[duckduckgo_search_tool(duckduckgo_client=ddgs_client, max_results=8)],
)
```

#### Combining `duckduckgo_search_tool` with `web_fetch_tool`

A common pattern: search for relevant URLs, then fetch and summarise each one.

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    tools=[
        duckduckgo_search_tool(max_results=5),
        web_fetch_tool(max_content_length=20_000, timeout=20),
    ],
    system_prompt=(
        'You are a research assistant. First search DuckDuckGo to find relevant pages, '
        'then fetch the most promising URLs to read the full content. '
        'Always cite your sources.'
    ),
)

result = agent.run_sync('Research the current state of async Python frameworks in 2026.')
print(result.output)
```

#### Note on rate limiting

DuckDuckGo's unofficial API has no authentication but may rate-limit aggressive usage. For production workloads consider:
- Caching results with a short TTL.
- Using the `max_results` parameter to limit request size.
- Using the native `WebSearchTool` capability on providers that support it (Anthropic, OpenAI, Groq) for higher rate limits.

---

## Reference

- `AbstractBuiltinTool` — `builtin_tools/__init__.py:41`
- `WebSearchTool`, `WebSearchUserLocation` — `:90`, `:160`
- `XSearchTool` — `:183`
- `CodeExecutionTool` — `:274`
- `WebFetchTool` / `UrlContextTool` — `:291`, `:352` (deprecated native tool)
- `ImageGenerationTool` — `:364`
- `MemoryTool` — `:456`
- `MCPServerTool` — `:469`
- `FileSearchTool` — `:540`
- `BUILTIN_TOOLS_REQUIRING_CONFIG`, `SUPPORTED_BUILTIN_TOOLS`, `DEPRECATED_BUILTIN_TOOLS` — tail of `builtin_tools/__init__.py`
- `web_fetch_tool`, `WebFetchResult` — `common_tools/web_fetch.py`
- `duckduckgo_search_tool`, `DuckDuckGoResult` — `common_tools/duckduckgo.py`

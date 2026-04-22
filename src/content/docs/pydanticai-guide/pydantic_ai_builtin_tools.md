---
title: "PydanticAI: Built-in Tools"
description: "Provider-native tools — WebSearchTool, WebFetchTool, CodeExecutionTool, ImageGenerationTool, FileSearchTool, MemoryTool, MCPServerTool, XSearchTool — and which providers support each."
framework: pydanticai
language: python
---

# Built-in Tools

Verified against **pydantic-ai==1.85.1** — source module: `pydantic_ai.builtin_tools`.

Built-in tools are provider-native — they execute inside the LLM provider's infrastructure, not in your Python process. You opt in via `builtin_tools=[...]` on `Agent`; PydanticAI forwards a typed config to the provider and streams the results back as `BuiltinToolCallPart` / `BuiltinToolReturnPart`.

## Minimal runnable example

```python
from pydantic_ai import Agent, WebSearchTool

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    builtin_tools=[WebSearchTool(max_uses=3)],
)
result = agent.run_sync('What happened in AI news this week? Cite sources.')
print(result.output)
```

The `builtin_tools=` list is validated at agent construction against each model's `Model.supported_builtin_tools()`, so unsupported combinations fail fast.

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

## `WebSearchTool`

```python
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation

agent = Agent(
    'openai-responses:gpt-5.2',   # OpenAI Responses API required
    builtin_tools=[
        WebSearchTool(
            search_context_size='high',
            max_uses=5,
            allowed_domains=['docs.python.org', 'peps.python.org'],
            user_location=WebSearchUserLocation(country='US', city='San Francisco'),
        ),
    ],
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

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    builtin_tools=[WebFetchTool(max_uses=3, enable_citations=True)],
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

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    builtin_tools=[CodeExecutionTool()],
)
result = agent.run_sync('Compute the 50th Fibonacci number.')
```

Runs in the provider's sandbox (Anthropic's container, OpenAI's Python interpreter, Google's Gemini code execution). The tool has **no configurable fields** in 1.85 — the only knob is model-level settings.

## `ImageGenerationTool`

```python
from pydantic_ai import Agent, ImageGenerationTool

agent = Agent(
    'openai-responses:gpt-5.2',
    builtin_tools=[ImageGenerationTool(
        size='1024x1024',
        quality='high',
        output_format='png',
    )],
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

agent = Agent(
    'openai-responses:gpt-5.2',
    builtin_tools=[FileSearchTool(file_store_ids=['vs_abc123', 'vs_xyz789'])],
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

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    builtin_tools=[MemoryTool()],
)
```

## `MCPServerTool`

Built-in tool that asks the provider to call out to a remote MCP server. Different from `pydantic_ai.mcp.MCPServerStdio` / `MCPServerSSE` / `MCPServerStreamableHTTP`, which run the client locally.

```python
from pydantic_ai import Agent, MCPServerTool

agent = Agent(
    'openai-responses:gpt-5.2',
    builtin_tools=[MCPServerTool(
        id='docs-mcp',
        url='https://mcp.example.com/docs',
        authorization_token='Bearer ...',
        allowed_tools=['search', 'fetch'],
        headers={'x-org': 'acme'},
    )],
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

agent = Agent(
    'xai:grok-3-latest',
    builtin_tools=[XSearchTool(
        allowed_x_handles=['pydantic'],
        from_date=datetime(2026, 1, 1),
        include_output=True,
    )],
)
```

Fields (with validation in `__post_init__`):

- `allowed_x_handles` / `excluded_x_handles` (max 10, mutually exclusive)
- `from_date` / `to_date` (naive datetimes = UTC)
- `enable_image_understanding`, `enable_video_understanding`
- `include_output` — emit raw search results as `BuiltinToolReturnPart` (otherwise the model only uses them internally).

## Inspecting built-in tool traffic

Built-in tool calls appear as `BuiltinToolCallPart` and `BuiltinToolReturnPart` in the message history, distinct from `ToolCallPart` / `ToolReturnPart` (which are for your function tools). Streaming events fire as `BuiltinToolCallEvent` / `BuiltinToolResultEvent`.

```python
from pydantic_ai.messages import BuiltinToolCallPart

for m in result.all_messages():
    for p in m.parts:
        if isinstance(p, BuiltinToolCallPart):
            print(p.tool_name, p.args_as_json_str())
```

## Mixing built-in and function tools

```python
from pydantic_ai import Agent, WebSearchTool, RunContext

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    builtin_tools=[WebSearchTool(max_uses=2)],
)

@agent.tool
def internal_lookup(ctx: RunContext[None], sku: str) -> dict:
    return db.get(sku)
```

The model sees both flavours. Built-in tool calls don't consume your function-tool retry budget.

## Gotchas

- **`UrlContextTool` is deprecated.** Swap to `WebFetchTool`; the serialised `kind='url_context'` still deserialises via the deprecated subclass for backward-compat.
- **`defer_model_check=False`** (default) validates the built-in tool list against the model at construction. If you swap models via `agent.override(model=...)` to one that doesn't support the tools, you'll hit `UserError` at run time — overriding isn't auto-filtered.
- **Field support varies by provider** even on "supported" tools. The docstrings list per-field support; unsupported fields are silently ignored by some providers and raise on others.
- **Built-in tool usage is counted against provider quotas, not PydanticAI's `tool_calls_limit`**. Use `max_uses` where available.
- **`MCPServerTool` vs. `pydantic_ai.mcp`**: the former is the provider's remote MCP client; the latter is a local MCP client. Don't register the same server twice under both — you'd get duplicate tools.

## Patterns

### 1. Research agent with web search + fetch

```python
agent = Agent('anthropic:claude-sonnet-4-6', builtin_tools=[
    WebSearchTool(max_uses=5),
    WebFetchTool(max_uses=10, enable_citations=True),
])
```

### 2. Constrain searches to a domain

```python
WebSearchTool(allowed_domains=['docs.company.com'])
```

### 3. Code-runner with a structured output contract

```python
from pydantic import BaseModel
class CalcResult(BaseModel):
    answer: float
    reasoning: str

agent = Agent('openai-responses:gpt-5.2',
              output_type=CalcResult,
              builtin_tools=[CodeExecutionTool()])
```

### 4. RAG via `FileSearchTool` + fallback to local embeddings

```python
agent = Agent('openai-responses:gpt-5.2',
              builtin_tools=[FileSearchTool(file_store_ids=['vs_prod'])])

@agent.tool
async def fallback_search(ctx, query: str) -> list[str]:
    # Only called if the built-in search misses
    return await embedder_search(query)
```

### 5. Inspect what the model actually searched

```python
async for event in agent.run_stream_events(prompt):
    from pydantic_ai.messages import BuiltinToolCallEvent, BuiltinToolResultEvent
    if isinstance(event, BuiltinToolCallEvent):
        print(f'[builtin call] {event.part.tool_name}({event.part.args_as_json_str()})')
    if isinstance(event, BuiltinToolResultEvent):
        print(f'[builtin result] {event.result.content[:200]}')
```

## Reference

- `AbstractBuiltinTool` — `builtin_tools/__init__.py:41`
- `WebSearchTool`, `WebSearchUserLocation` — `:90`, `:160`
- `XSearchTool` — `:183`
- `CodeExecutionTool` — `:274`
- `WebFetchTool` / `UrlContextTool` — `:291`, `:352` (deprecated)
- `ImageGenerationTool` — `:364`
- `MemoryTool` — `:456`
- `MCPServerTool` — `:469`
- `FileSearchTool` — `:540`
- `BUILTIN_TOOLS_REQUIRING_CONFIG`, `SUPPORTED_BUILTIN_TOOLS`, `DEPRECATED_BUILTIN_TOOLS` — tail of the module

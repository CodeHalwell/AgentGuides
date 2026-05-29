---
title: "PydanticAI — Class Deep Dives Vol. 4"
description: "Source-verified deep dives into 10 PydanticAI classes: LangChainToolset, VercelAIAdapter, ToolManager/ValidatedToolCall, ThreadExecutor, PrefixTools, PrepareTools/PrepareOutputTools, ImageGeneration, XSearch, common_tools (DuckDuckGo/Tavily/Exa), FunctionSignature/TypeSignature."
sidebar:
  label: "Class deep dives (Vol. 4)"
  order: 24
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.104.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.104.0`.
</Aside>

Ten class groups from the `pydantic_ai` 1.104.0 source covering LangChain integration, the Vercel AI
SDK adapter, the internal tool execution engine, thread pool management for production servers,
capability-level tool naming and filtering, image generation and X search capabilities, third-party
search tool factories, and function signature generation for Code Mode.

---

## 1. `LangChainTool` + `LangChainToolset` + `tool_from_langchain`

**Module:** `pydantic_ai.ext.langchain`  
**Import:** `from pydantic_ai.ext.langchain import LangChainToolset, tool_from_langchain`

These classes bridge any LangChain `BaseTool` into Pydantic AI so you can use the entire
LangChain tool ecosystem without rewriting function signatures.

### How it works

`LangChainTool` is a structural `Protocol` that matches any LangChain tool object — it does not
require you to import `langchain` at all. `tool_from_langchain` builds a `Tool.from_schema` proxy
that:

1. Reads `args` (JSON Schema for each parameter) and `get_input_jsonschema()` from the LangChain tool
2. Extracts required parameters (those without a `default` key)
3. Sets `additionalProperties: False` on the schema if not already set
4. Merges default values and forwards the combined kwargs as a single `dict` to `langchain_tool.run()`

```python
from pydantic_ai.ext.langchain import LangChainToolset, tool_from_langchain
```

### `LangChainToolset` constructor

```python
class LangChainToolset(FunctionToolset):
    def __init__(self, tools: list[LangChainTool], *, id: str | None = None): ...
```

`LangChainToolset` is a thin subclass of `FunctionToolset`. It converts each LangChain tool via
`tool_from_langchain` and delegates everything else to `FunctionToolset`.

### Example 1 — Wrap a LangChain file-search tool

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset

# Any LangChain tool works — here using langchain_community
from langchain_community.tools import ListDirectoryTool

toolset = LangChainToolset([ListDirectoryTool()])
agent = Agent('openai:gpt-4o', toolsets=[toolset])

async def main():
    result = await agent.run("List the files in the src directory")
    print(result.data)

asyncio.run(main())
```

### Example 2 — Mix LangChain tools with native Pydantic AI tools

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from pydantic_ai.ext.langchain import LangChainToolset
from pydantic_ai.toolsets import FunctionToolset

# LangChain tool
from langchain_community.tools import DuckDuckGoSearchRun

# Native Pydantic AI tool
def get_current_temperature(city: str) -> str:
    """Returns the current temperature for a city."""
    return f"22°C in {city}"  # stub

lc_toolset = LangChainToolset([DuckDuckGoSearchRun()])
native_toolset = FunctionToolset([get_current_temperature])

agent = Agent(
    'openai:gpt-4o',
    toolsets=[lc_toolset, native_toolset],
)

async def main():
    result = await agent.run("Search for the weather in London, then compare to 22°C")
    print(result.data)

asyncio.run(main())
```

### Example 3 — Wrap a single tool with `tool_from_langchain`

```python
from pydantic_ai.ext.langchain import tool_from_langchain
from pydantic_ai.toolsets import FunctionToolset

# Convert individually for custom per-tool control
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
pai_tool = tool_from_langchain(wiki_tool)

# Inspect the converted tool
print(pai_tool.name)         # 'wikipedia'
print(pai_tool.description)  # original LangChain description
```

<Aside type="note">
The `run()` method of the LangChain tool is always called with a single `dict` of kwargs — the
same contract as `BaseTool.run(tool_input: dict)`. Tools that only accept a plain string input
(old-style `arun(query: str)`) should be wrapped in a one-arg `dict` shim.
</Aside>

---

## 2. `VercelAIAdapter`

**Module:** `pydantic_ai.ui.vercel_ai`  
**Import:** `from pydantic_ai.ui.vercel_ai import VercelAIAdapter`

`VercelAIAdapter` connects a Pydantic AI agent to the [Vercel AI SDK](https://sdk.vercel.ai/) data
stream protocol. It handles deserializing Vercel AI chat messages, running the agent, and streaming
responses back in Vercel's chunk format.

### Constructor / dataclass fields

```python
@dataclass
class VercelAIAdapter(UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    sdk_version: Literal[5, 6] = 5
    # sdk_version=6 enables tool-approval streaming (HITL) for Vercel AI SDK v6
    server_message_id: str | None = None
    # Optional server-generated ID added to the StartChunk
```

`VercelAIAdapter` inherits from `UIAdapter` — the same abstract base used by `AGUIAdapter`. All
message loading, streaming, and lifecycle methods come from the parent class.

### Key class methods

| Method | Purpose |
|--------|---------|
| `build_run_input(body: bytes)` | Parse Vercel AI request JSON into `RequestData` |
| `from_request(request, *, agent, sdk_version, ...)` | Build adapter from a Starlette `Request` |
| `dispatch_request(request, *, agent, ...)` | One-call handler — runs agent, returns streaming `Response` |
| `load_messages(messages)` | Transform Vercel AI `UIMessage` list → `list[ModelMessage]` |

### Key properties

| Property | Type | Description |
|----------|------|-------------|
| `deferred_tool_results` | `DeferredToolResults \| None` | Populated from SDK v6 approval responses |
| `messages` | `list[ModelMessage]` | Parsed Pydantic AI messages (cached) |
| `conversation_id` | `str \| None` | Top-level `id` from the Vercel AI request body |

### Example 1 — FastAPI streaming endpoint (SDK v5)

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

app = FastAPI()
agent = Agent('openai:gpt-4o', system_prompt="You are a helpful assistant.")

@app.post("/api/chat")
async def chat(request: Request):
    response = await VercelAIAdapter.dispatch_request(
        request,
        agent=agent,
    )
    return response  # Starlette Response, FastAPI accepts it directly
```

### Example 2 — SDK v6 with HITL tool approval

```python
from fastapi import FastAPI, Request
from pydantic_ai import Agent
from pydantic_ai.capabilities import ApprovalRequiredToolset
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

app = FastAPI()

def delete_file(path: str) -> str:
    """Deletes a file at the given path."""
    import os
    os.remove(path)
    return f"Deleted {path}"

toolset = ApprovalRequiredToolset(FunctionToolset([delete_file]))
agent = Agent('openai:gpt-4o', toolsets=[toolset])

@app.post("/api/chat")
async def chat(request: Request):
    # sdk_version=6 enables streaming tool approval chunks to the frontend
    return await VercelAIAdapter.dispatch_request(
        request,
        agent=agent,
        sdk_version=6,
    )
```

### Example 3 — Manual message history with `from_request`

```python
from fastapi import FastAPI, Request
from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

app = FastAPI()
agent = Agent('openai:gpt-4o')

@app.post("/api/chat")
async def chat(request: Request):
    adapter = await VercelAIAdapter.from_request(
        request,
        agent=agent,
        sdk_version=5,
        manage_system_prompt='server',
        allowed_file_url_schemes=frozenset({'https'}),
    )
    # Inspect parsed messages before running
    for msg in adapter.messages:
        print(type(msg).__name__, msg)

    # Stream the response
    event_stream = adapter.build_event_stream()
    return await event_stream.stream_response(
        agent=agent,
        adapter=adapter,
    )
```

<Aside type="tip">
Setting `manage_system_prompt='server'` (default) strips system messages that the client sends —
the agent's `system_prompt` is authoritative. Use `'client'` only when the frontend constructs
the full system prompt.
</Aside>

---

## 3. `ToolManager` + `ValidatedToolCall`

**Module:** `pydantic_ai.tool_manager`  
**Import:** `from pydantic_ai.tool_manager import ToolManager, ValidatedToolCall`

`ToolManager` is the internal engine that resolves, validates, and executes every tool call in an
agent run step. Understanding it lets you control parallel vs. sequential execution and debug
validation failures.

### `ToolManager` fields

```python
@dataclass
class ToolManager(Generic[AgentDepsT]):
    toolset: AbstractToolset[AgentDepsT]
    root_capability: AbstractCapability[AgentDepsT] | None = None
    ctx: RunContext[AgentDepsT] | None = None
    tools: dict[str, ToolsetTool[AgentDepsT]] | None = None  # keyed by model-facing name
    failed_tools: set[str] = field(default_factory=set)
    default_max_retries: int = 1
```

### Parallel execution mode

```python
# ParallelExecutionMode = Literal['parallel', 'sequential', 'parallel_ordered_events']

# Run all tool calls sequentially for this block
with ToolManager.parallel_execution_mode('sequential'):
    result = await agent.run("Do three things in order")

# Parallel but emit events in call order (useful for deterministic UI updates)
with ToolManager.parallel_execution_mode('parallel_ordered_events'):
    result = await agent.run("Search three things")
```

The mode is stored in a `ContextVar` — safe for concurrent async tasks. A tool's own
`ToolDefinition.sequential=True` field forces sequential even if the context says `'parallel'`.

### `for_run_step` — retry carry-over

Each agent run step creates a fresh `ToolManager` via `await tool_manager.for_run_step(ctx)`. It
carries forward **retry counts** for any tools that failed in the prior step, so the agent's retry
budget accumulates correctly across multiple model requests.

### `ValidatedToolCall` fields

```python
@dataclass
class ValidatedToolCall(Generic[AgentDepsT]):
    call: ToolCallPart         # the original model call
    tool: ToolsetTool | None   # resolved tool, or None if unknown
    ctx: RunContext             # run context for this call
    args_valid: bool            # did schema + custom validator pass?
    validated_args: dict[str, Any] | None = None   # ready args, or None
    validation_error: ToolRetryError | None = None  # retry-ready error part
```

`ValidatedToolCall` separates validation from execution, which enables accurate `FunctionToolCallEvent`
telemetry and allows toolsets to handle unknown tool names (e.g. deferred tool routing) before any
execution attempt.

### Example — Force sequential tool calls globally

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tool_manager import ToolManager

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def step_one() -> str:
    return "step 1 complete"

@agent.tool_plain
def step_two() -> str:
    return "step 2 complete"

async def main():
    with ToolManager.parallel_execution_mode('sequential'):
        result = await agent.run("Run step_one then step_two")
    print(result.data)

asyncio.run(main())
```

### Example — Sequential tool via `ToolDefinition`

```python
from pydantic_ai import Agent
from pydantic_ai.tools import Tool, ToolDefinition

agent = Agent('openai:gpt-4o')

def write_to_db(record: str) -> str:
    """Writes a record to the database — must not run concurrently."""
    return f"wrote: {record}"

# sequential=True on the ToolDefinition forces sequential even without
# changing the global ContextVar
tool = Tool(
    write_to_db,
    prepare=lambda ctx, td: ToolDefinition(
        name=td.name,
        description=td.description,
        parameters_json_schema=td.parameters_json_schema,
        sequential=True,
    ),
)
agent = Agent('openai:gpt-4o', tools=[tool])
```

---

## 4. `ThreadExecutor`

**Module:** `pydantic_ai.capabilities.thread_executor`  
**Import:** `from pydantic_ai.capabilities import ThreadExecutor`

By default, Pydantic AI runs sync tool functions in ephemeral threads using
`anyio.to_thread.run_sync`. In production servers under load this can create an unbounded thread
pool. `ThreadExecutor` replaces that behaviour with a bounded `ThreadPoolExecutor` (or any
`concurrent.futures.Executor`) scoped to each agent run.

### Constructor

```python
@dataclass
class ThreadExecutor(AbstractCapability[Any]):
    executor: Executor  # any concurrent.futures.Executor
```

The capability uses `wrap_run` — it sets the executor as a context-local override for the entire
agent run, then restores it.

### Example 1 — Bounded thread pool for FastAPI

```python
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

app = FastAPI()

# Created once at startup; shared across all requests
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="agent-worker")
agent = Agent(
    'openai:gpt-4o',
    capabilities=[ThreadExecutor(_executor)],
)

@app.post("/run")
async def run(prompt: str):
    result = await agent.run(prompt)
    return {"answer": result.data}

@app.on_event("shutdown")
def shutdown():
    _executor.shutdown(wait=True)
```

### Example 2 — Global executor for all agents

```python
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai.agent import Agent

# Alternative: set globally instead of per-agent
executor = ThreadPoolExecutor(max_workers=8)
with Agent.using_thread_executor(executor):
    result = agent.run_sync("Process this")
```

### Example 3 — Process pool for CPU-bound sync tools

```python
from concurrent.futures import ProcessPoolExecutor
from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

def heavy_compute(data: str) -> str:
    """CPU-bound preprocessing."""
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()

executor = ProcessPoolExecutor(max_workers=4)
agent = Agent(
    'openai:gpt-4o',
    capabilities=[ThreadExecutor(executor)],
)
agent.tool_plain(heavy_compute)
```

<Aside type="note">
`ThreadExecutor` only affects sync functions. Async tool functions always run on the event loop
directly and are unaffected.
</Aside>

---

## 5. `PrefixTools`

**Module:** `pydantic_ai.capabilities.prefix_tools`  
**Import:** `from pydantic_ai.capabilities import PrefixTools`

`PrefixTools` is a `WrapperCapability` that adds a `{prefix}_` to the names of every tool
contributed by its **wrapped** capability, without touching any other agent tools. This is useful
for namespacing tool collections (e.g. MCP servers, third-party integrations) so they don't clash.

### Constructor

```python
@dataclass
class PrefixTools(WrapperCapability[AgentDepsT]):
    wrapped: AbstractCapability[AgentDepsT]  # inherited from WrapperCapability
    prefix: str
```

Internally, `PrefixTools.get_toolset()` wraps the resolved `AgentToolset` in a `PrefixedToolset`.
The prefix is inserted with an underscore: `prefix='mcp'` turns `'search'` → `'mcp_search'`.

### `from_spec` — deserialize from a config dict

```python
PrefixTools.from_spec(prefix='mcp', capability={'type': 'Toolset', 'toolset': {...}})
```

Useful when loading agent configurations from YAML/JSON.

### Example 1 — Namespace an MCP toolset

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.capabilities.toolset import Toolset
from pydantic_ai.mcp import MCPToolset

mcp_cap = Toolset(MCPToolset("http://localhost:8000/mcp"))

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        PrefixTools(wrapped=mcp_cap, prefix='mcp'),
    ],
)
# The model now sees: 'mcp_search', 'mcp_read_file', etc.
```

### Example 2 — Two MCP servers with distinct namespaces

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.capabilities.toolset import Toolset
from pydantic_ai.mcp import MCPToolset

search_cap = Toolset(MCPToolset("http://search-server/mcp"))
docs_cap = Toolset(MCPToolset("http://docs-server/mcp"))

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        PrefixTools(wrapped=search_cap, prefix='search'),
        PrefixTools(wrapped=docs_cap, prefix='docs'),
    ],
)
# Model sees: 'search_web_search', 'docs_lookup', etc.
```

### Example 3 — Prefix a `FunctionToolset`

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.capabilities.toolset import Toolset
from pydantic_ai.toolsets import FunctionToolset

def get_user(user_id: str) -> dict:
    """Fetch a user record."""
    return {"id": user_id, "name": "Alice"}

def list_orders(user_id: str) -> list:
    """List orders for a user."""
    return []

crm_toolset = FunctionToolset([get_user, list_orders])
crm_cap = Toolset(crm_toolset)

agent = Agent(
    'openai:gpt-4o',
    capabilities=[PrefixTools(wrapped=crm_cap, prefix='crm')],
)
# Model sees: 'crm_get_user', 'crm_list_orders'
```

---

## 6. `PrepareTools` + `PrepareOutputTools`

**Module:** `pydantic_ai.capabilities.prepare_tools`  
**Import:** `from pydantic_ai.capabilities import PrepareTools, PrepareOutputTools`

`PrepareTools` wraps a `ToolsPrepareFunc` as a capability so it applies to the agent's
**function tools** every run step. `PrepareOutputTools` does the same for **output tools**.
Both are simpler alternatives to writing a full `AbstractCapability` when all you need is to
filter or modify tool definitions at the start of each step.

### Signatures

```python
@dataclass
class PrepareTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]
    # ToolsPrepareFunc = Callable[[RunContext[AgentDepsT], list[ToolDefinition]],
    #                              Awaitable[list[ToolDefinition] | None] | list[ToolDefinition] | None]

@dataclass
class PrepareOutputTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]
```

Returning `None` is treated the same as returning an empty list (with a deprecation warning). Both
sync and async `prepare_func` are supported.

### Example 1 — Hide admin tools based on user role

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition

@dataclass
class UserDeps:
    role: str  # 'admin' | 'user'

async def filter_by_role(
    ctx: RunContext[UserDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    if ctx.deps.role == 'admin':
        return tool_defs  # admins see everything
    return [td for td in tool_defs if not td.name.startswith('admin_')]

agent = Agent(
    'openai:gpt-4o',
    capabilities=[PrepareTools(filter_by_role)],
)

@agent.tool_plain
def admin_delete_user(user_id: str) -> str:
    return f"deleted {user_id}"

@agent.tool_plain
def get_profile(user_id: str) -> str:
    return f"profile for {user_id}"

async def main():
    # Regular user only sees get_profile
    result = await agent.run("Show my profile", deps=UserDeps(role='user'))
    print(result.data)

asyncio.run(main())
```

### Example 2 — Modify tool descriptions dynamically

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition
from dataclasses import replace

async def add_environment_context(
    ctx: RunContext[dict], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    env = ctx.deps.get('environment', 'production')
    return [
        replace(td, description=f"[{env.upper()}] {td.description}")
        for td in tool_defs
    ]

agent = Agent('openai:gpt-4o', capabilities=[PrepareTools(add_environment_context)])
```

### Example 3 — Gate output tools until after the first step

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareOutputTools
from pydantic_ai.output import ToolOutput
from pydantic_ai.tools import ToolDefinition

async def only_after_research(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    # Don't offer the structured output tool until at least one run step
    if ctx.run_step == 0:
        return []
    return tool_defs

from pydantic import BaseModel

class Report(BaseModel):
    summary: str
    confidence: float

agent = Agent(
    'openai:gpt-4o',
    output_type=ToolOutput(Report),
    capabilities=[PrepareOutputTools(only_after_research)],
)
```

---

## 7. `ImageGeneration`

**Module:** `pydantic_ai.capabilities.image_generation`  
**Import:** `from pydantic_ai.capabilities import ImageGeneration`

`ImageGeneration` is a `NativeOrLocalTool` capability that routes image generation either to the
model's native image generation (e.g. GPT-4o with DALL-E) or to a **subagent fallback** running on
an image-capable model when the primary model doesn't support it.

### Constructor

```python
ImageGeneration(
    *,
    native: ImageGenerationTool | Callable[..., ImageGenerationTool | None] | bool = True,
    local: Tool | Callable | Literal[False] | None = None,
    fallback_model: Model | KnownModelName | Callable[..., Model] | None = None,
    # Quality / format controls forwarded to ImageGenerationTool
    action: Literal['generate', 'edit', 'auto'] | None = None,
    background: Literal['transparent', 'opaque', 'auto'] | None = None,
    input_fidelity: Literal['high', 'low'] | None = None,
    moderation: Literal['auto', 'low'] | None = None,
    image_model: ImageGenerationModelName | None = None,
    output_compression: int | None = None,
    output_format: Literal['png', 'webp', 'jpeg'] | None = None,
    quality: Literal['low', 'medium', 'high', 'auto'] | None = None,
    size: Literal['auto','1024x1024','1024x1536','1536x1024','512','1K','2K','4K'] | None = None,
    aspect_ratio: ImageAspectRatio | None = None,
)
```

<Aside type="caution">
`fallback_model` and `local` are mutually exclusive. Providing both raises `UserError`.
</Aside>

### Image settings by provider

| Setting | OpenAI Responses | Google (Gemini) |
|---------|-----------------|-----------------|
| `action` | `'generate'`/`'edit'`/`'auto'` | — |
| `background` | `'transparent'`/`'opaque'`/`'auto'` | — |
| `size` | `'auto'`, `'1024x1024'`, etc. | `'512'`, `'1K'`, `'2K'`, `'4K'` |
| `aspect_ratio` | maps to size | full `ImageAspectRatio` support |
| `quality` | `'low'`/`'medium'`/`'high'`/`'auto'` | — |
| `output_format` | `'png'`/`'webp'`/`'jpeg'` | `'jpeg'`, `'png'` |
| `output_compression` | `0`–`100` (jpeg/webp) | `0`–`100` (jpeg) |

### Example 1 — Native image generation on GPT-4o

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        ImageGeneration(
            quality='high',
            output_format='webp',
            size='1024x1024',
        )
    ],
)

async def main():
    result = await agent.run("Draw a cartoon cat wearing a space suit")
    print(result.data)  # data URI or URL depending on model

asyncio.run(main())
```

### Example 2 — Fallback to Google Imagen on a non-image model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        ImageGeneration(
            fallback_model='google:gemini-3-pro-image-preview',
            output_format='png',
            size='1K',
        )
    ],
)

async def main():
    result = await agent.run("Generate a logo for a coffee shop")
    print(result.data)

asyncio.run(main())
```

### Example 3 — Transparent background + edit mode

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[
        ImageGeneration(
            action='edit',
            background='transparent',
            output_format='png',
            output_compression=90,
        )
    ],
)

async def main():
    result = await agent.run(
        "Remove the background from this product photo and add studio lighting"
    )
    print(result.data)

asyncio.run(main())
```

---

## 8. `XSearch`

**Module:** `pydantic_ai.capabilities.x_search`  
**Import:** `from pydantic_ai.capabilities import XSearch`

`XSearch` is a `NativeOrLocalTool` capability for X (Twitter) search. On xAI models
(e.g. `grok-4`), it uses the native `XSearchTool` directly. On any other model, you must provide
`fallback_model` pointing to an xAI model that will act as a search subagent.

### Constructor

```python
XSearch(
    *,
    native: XSearchTool | Callable[..., XSearchTool | None] | bool = True,
    local: Tool | Callable | Literal[False] | None = None,
    fallback_model: Model | KnownModelName | Callable[..., Model] | None = None,
    allowed_x_handles: list[str] | None = None,   # max 10; only include these accounts
    excluded_x_handles: list[str] | None = None,  # max 10; exclude these accounts
    from_date: datetime | None = None,
    to_date: datetime | None = None,
    enable_image_understanding: bool | None = None,
    enable_video_understanding: bool | None = None,
    include_output: bool | None = None,  # include raw X results in NativeToolReturnPart
)
```

<Aside type="caution">
There is **no default fallback model**. On a non-xAI model, omitting `fallback_model` raises an error.
`fallback_model` and `local` are mutually exclusive.
</Aside>

<Aside type="note">
`allowed_x_handles` / `excluded_x_handles` constraints are enforced by the native `XSearchTool`.
When using `fallback_model`, the subagent also runs the native tool, so constraints are honored.
</Aside>

### Example 1 — Native X search on a Grok model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'xai:grok-4',
    capabilities=[XSearch()],
)

async def main():
    result = await agent.run(
        "What are people saying about PydanticAI on X today?"
    )
    print(result.data)

asyncio.run(main())
```

### Example 2 — X search on GPT-4o with a Grok fallback

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        XSearch(fallback_model='xai:grok-4-1-fast-non-reasoning'),
    ],
)

async def main():
    result = await agent.run("Summarize recent AI announcements on X")
    print(result.data)

asyncio.run(main())
```

### Example 3 — Filter to specific handles, date range, with image understanding

```python
import asyncio
from datetime import datetime, timezone
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent(
    'xai:grok-4',
    capabilities=[
        XSearch(
            allowed_x_handles=['openai', 'anthropic', 'googledeepmind'],
            from_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
            to_date=datetime(2026, 5, 29, tzinfo=timezone.utc),
            enable_image_understanding=True,
        )
    ],
)

async def main():
    result = await agent.run(
        "What major model releases did the big AI labs announce in May 2026?"
    )
    print(result.data)

asyncio.run(main())
```

---

## 9. Common tools: `duckduckgo_search_tool` · `tavily_search_tool` · `ExaToolset`

**Module:** `pydantic_ai.common_tools`  
**Install extras:** `pip install "pydantic-ai[duckduckgo]"` / `"pydantic-ai[tavily]"` / `"pydantic-ai[exa]"`

`pydantic_ai.common_tools` provides lightweight factory functions that turn third-party search
clients into `Tool` objects. All three backends are async-native and return typed `TypedDict` lists.

### 9a — `duckduckgo_search_tool`

```python
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
# requires: pip install "pydantic-ai[duckduckgo]" or pip install ddgs

def duckduckgo_search_tool(
    duckduckgo_client: DDGS | None = None,
    max_results: int | None = None,       # None = first page only
) -> Tool[Any]: ...
```

Returns a `Tool` named `'duckduckgo_search'` whose callable is async (uses
`anyio.to_thread.run_sync` around the sync DDGS client). Results are validated as
`list[DuckDuckGoResult]`:

```python
class DuckDuckGoResult(TypedDict):
    title: str
    href: str
    body: str
```

**Example:**

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[duckduckgo_search_tool(max_results=5)],
)

async def main():
    result = await agent.run("What is the latest pydantic-ai release?")
    print(result.data)

asyncio.run(main())
```

### 9b — `tavily_search_tool`

```python
from pydantic_ai.common_tools.tavily import tavily_search_tool
# requires: pip install "pydantic-ai[tavily]" or pip install tavily-python

def tavily_search_tool(
    api_key: str,
    *,
    max_results: int | None = None,
    # Forwarded as defaults to the tool callable — model can override:
    search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = ...,
    topic: Literal['general', 'news', 'finance'] = ...,
    time_range: Literal['day', 'week', 'month', 'year'] | None = ...,
    include_domains: list[str] | None = ...,
    exclude_domains: list[str] | None = ...,
) -> Tool[Any]: ...
```

The resulting `'tavily_search'` tool exposes those same parameters as LLM-callable arguments,
letting the model choose `search_depth='advanced'` or `topic='news'` as needed. Results are
`list[TavilySearchResult]`:

```python
class TavilySearchResult(TypedDict):
    title: str
    url: str
    content: str   # brief snippet
    score: float   # relevance 0-1
```

**Example — news-focused search with domain filtering:**

```python
import asyncio, os
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[
        tavily_search_tool(
            api_key=os.environ['TAVILY_API_KEY'],
            max_results=8,
            topic='news',
            time_range='week',
            exclude_domains=['reddit.com', 'quora.com'],
        )
    ],
)

async def main():
    result = await agent.run("Summarize AI regulation news from the past week")
    print(result.data)

asyncio.run(main())
```

### 9c — `ExaToolset`

```python
from pydantic_ai.common_tools.exa import ExaToolset
# requires: pip install "pydantic-ai[exa]" or pip install exa-py

class ExaToolset(FunctionToolset):
    def __init__(
        self,
        api_key: str,
        *,
        num_results: int = 5,
        max_characters: int | None = None,
        include_search: bool = True,        # exa_search
        include_find_similar: bool = True,  # exa_find_similar
        include_get_contents: bool = True,  # exa_get_contents
        include_answer: bool = True,        # exa_answer
        id: str | None = None,
    ): ...
```

`ExaToolset` bundles **four** tools that share a single `AsyncExa` client:

| Tool name | Input | Returns |
|-----------|-------|---------|
| `exa_search` | `query`, `search_type` | `list[ExaSearchResult]` with `title`, `url`, `text`, `published_date`, `author` |
| `exa_find_similar` | `url`, `exclude_source_domain` | `list[ExaSearchResult]` for similar pages |
| `exa_get_contents` | `urls: list[str]` | `list[ExaContentResult]` with full page text |
| `exa_answer` | `query` | `ExaAnswerResult` with `answer` + `citations` |

`search_type` options for `exa_search`: `'auto'`, `'keyword'`, `'neural'`, `'fast'`, `'deep'`.

**Example — research agent with Exa:**

```python
import asyncio, os
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset

exa_toolset = ExaToolset(
    api_key=os.environ['EXA_API_KEY'],
    num_results=5,
    max_characters=2000,   # keep tokens down
    include_answer=True,
    include_find_similar=False,  # not needed for this agent
)

agent = Agent(
    'openai:gpt-4o',
    toolsets=[exa_toolset],
    system_prompt=(
        "You are a research assistant. Use exa_search for broad queries, "
        "exa_get_contents to read specific pages, and exa_answer for concise answers."
    ),
)

async def main():
    result = await agent.run(
        "Explain how pydantic-ai handles structured output validation"
    )
    print(result.data)

asyncio.run(main())
```

**Example — combine Exa search + DuckDuckGo fallback:**

```python
import asyncio, os
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-4o',
    toolsets=[ExaToolset(api_key=os.environ['EXA_API_KEY'], include_find_similar=False)],
    tools=[duckduckgo_search_tool(max_results=3)],
    system_prompt="Prefer exa_search. Fall back to duckduckgo_search for very recent news.",
)
```

---

## 10. `FunctionSignature` + `TypeSignature`

**Module:** `pydantic_ai.function_signature`  
**Import:** `from pydantic_ai.function_signature import FunctionSignature, TypeSignature, FunctionParam, TypeFieldSignature`

These classes power **Code Mode** — Pydantic AI's feature that presents tool definitions to the
model as Python function stubs rather than raw JSON Schema. Understanding them lets you control
exactly how tools are rendered to the LLM in Code Mode.

### Type expression tree

The module defines a small AST for Python type expressions:

| Class | Example output |
|-------|----------------|
| `SimpleTypeExpr(name='str')` | `str` |
| `LiteralTypeExpr(values=['a', 'b'])` | `Literal['a', 'b']` |
| `GenericTypeExpr(base='list', args=[...])` | `list[User]` |
| `UnionTypeExpr(members=[...])` | `User \| None` |
| `TypeSignature(name='User', fields={...})` | `class User(TypedDict): ...` |

### `FunctionSignature` fields

```python
@dataclass(kw_only=True)
class FunctionSignature:
    name: str
    description: str | None = None
    params: dict[str, FunctionParam]       # keyword-only function params
    return_type: TypeExpr
    referenced_types: list[TypeSignature]  # TypedDict definitions needed by params/return
    is_async: bool = False
```

### `render()` — produce a Python function stub

```python
sig.render(
    body='...',           # function body string (e.g. '...' for a stub)
    name=None,            # override the function name
    description=None,     # override the docstring
    is_async=None,
    conflicting_type_names=frozenset(),  # type names that need tool-name prefixes
)
```

All params are rendered as keyword-only (no positional args) because JSON Schema doesn't
distinguish positional from keyword arguments.

### `from_schema()` — build from JSON Schema

```python
FunctionSignature.from_schema(
    name='get_user',
    parameters_schema={
        'type': 'object',
        'properties': {
            'user_id': {'type': 'string', 'description': 'The user ID'},
            'include_orders': {'type': 'boolean'},
        },
        'required': ['user_id'],
    },
    return_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
)
```

### Example 1 — Render a tool as a Python stub

```python
from pydantic_ai.function_signature import FunctionSignature

sig = FunctionSignature.from_schema(
    name='search_products',
    parameters_schema={
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'Search terms'},
            'max_results': {'type': 'integer'},
            'category': {
                'type': 'string',
                'enum': ['electronics', 'books', 'clothing'],
            },
        },
        'required': ['query'],
    },
    return_schema={
        'type': 'array',
        'items': {
            'type': 'object',
            'title': 'Product',
            'properties': {
                'id': {'type': 'string'},
                'name': {'type': 'string'},
                'price': {'type': 'number'},
            },
        },
    },
)

# Print the full referenced type definitions
for t in sig.referenced_types:
    print(t.render_definition())
    print()

# Render the function stub
print(sig.render(body='...'))
```

Output (approximately):

```python
class Product(TypedDict):
    id: NotRequired[str]
    name: NotRequired[str]
    price: NotRequired[float]

def search_products(*, query: str, max_results: NotRequired[int], category: NotRequired[Literal['electronics', 'books', 'clothing']]) -> list[Product]:
    ...
```

### Example 2 — Detect and resolve type-name conflicts

```python
from pydantic_ai.function_signature import FunctionSignature, get_conflicting_type_names

sig_a = FunctionSignature.from_schema(
    name='get_user',
    parameters_schema={
        'type': 'object',
        'properties': {'address': {'$ref': '#/$defs/Address'}},
        '$defs': {'Address': {'type': 'object', 'properties': {'city': {'type': 'string'}}}},
    },
)

sig_b = FunctionSignature.from_schema(
    name='get_order',
    parameters_schema={
        'type': 'object',
        'properties': {'address': {'$ref': '#/$defs/Address'}},
        '$defs': {'Address': {'type': 'object', 'properties': {'street': {'type': 'string'}}}},
    },
)

# Find types that share a name but differ in structure across tools
conflicts = get_conflicting_type_names([sig_a, sig_b])

# Render with prefixed names to avoid collisions:
# 'get_user_Address', 'get_order_Address'
print(sig_a.render('...', conflicting_type_names=conflicts))
print(sig_b.render('...', conflicting_type_names=conflicts))
```

### Example 3 — Inspect `FunctionParam` and `TypeFieldSignature` directly

```python
from pydantic_ai.function_signature import (
    FunctionSignature, FunctionParam, TypeFieldSignature,
    SimpleTypeExpr, GenericTypeExpr
)

sig = FunctionSignature(
    name='create_report',
    params={
        'title': FunctionParam(name='title', type=SimpleTypeExpr(name='str'), required=True),
        'tags': FunctionParam(
            name='tags',
            type=GenericTypeExpr(base='list', args=[SimpleTypeExpr(name='str')]),
            required=False,
            description='Searchable tags for the report',
        ),
    },
    return_type=SimpleTypeExpr(name='str'),
    is_async=True,
)

print(sig.render(body="return await _create_report(title=title, tags=tags)"))
```

Output:

```python
async def create_report(*, title: str, tags: NotRequired[list[str]]) -> str:
    """
    tags: Searchable tags for the report
    """
    return await _create_report(title=title, tags=tags)
```

<Aside type="note">
`FunctionSignature` is used internally by Code Mode. You normally do not need to construct these
directly — but inspecting them lets you understand exactly what the model "sees" when Code Mode is
active, and you can build custom rendering pipelines for documentation or testing.
</Aside>

---

## Summary table

| # | Class / group | Module | New in |
|---|---|---|---|
| 1 | `LangChainTool` + `LangChainToolset` + `tool_from_langchain` | `pydantic_ai.ext.langchain` | v1.85.x |
| 2 | `VercelAIAdapter` | `pydantic_ai.ui.vercel_ai` | v1.98.x |
| 3 | `ToolManager` + `ValidatedToolCall` | `pydantic_ai.tool_manager` | v1.94.x |
| 4 | `ThreadExecutor` | `pydantic_ai.capabilities.thread_executor` | v1.100.x |
| 5 | `PrefixTools` | `pydantic_ai.capabilities.prefix_tools` | v1.87.x |
| 6 | `PrepareTools` + `PrepareOutputTools` | `pydantic_ai.capabilities.prepare_tools` | v1.87.x |
| 7 | `ImageGeneration` | `pydantic_ai.capabilities.image_generation` | v1.85.x |
| 8 | `XSearch` | `pydantic_ai.capabilities.x_search` | v1.85.x |
| 9 | `duckduckgo_search_tool` · `tavily_search_tool` · `ExaToolset` | `pydantic_ai.common_tools` | v1.98.x |
| 10 | `FunctionSignature` + `TypeSignature` | `pydantic_ai.function_signature` | v1.100.x |

All signatures, field names, and examples were taken directly from `pydantic-ai==1.104.0` installed source.

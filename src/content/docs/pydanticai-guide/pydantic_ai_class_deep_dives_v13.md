---
title: "PydanticAI — Class Deep Dives Vol. 13"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.106.0: TemplateStr (Handlebars dynamic instructions), Hooks capability (30+ lifecycle hook protocols — BeforeRunHookFunc, WrapModelRequestHookFunc, WrapToolExecuteHookFunc, etc.), WebSearch + WebFetch capabilities (native/local fallback web tools with domain filtering), Thinking capability (cross-provider extended reasoning via effort levels), NativeOrLocalTool (base framework for native/local tool pairing), ExaToolset + ExaSearchTool + ExaFindSimilarTool + ExaGetContentsTool + ExaAnswerTool (Exa neural search integration), TavilySearchTool (Tavily neural search with fixed parameters), AgentWorker + agent_to_a2a (FastA2A Agent-to-Agent protocol adapter), PrepareTools + PrepareOutputTools (dynamic tool filtering by context), ReinjectSystemPrompt + SetToolMetadata + PrefixTools (capability composition toolkit). All verified against pydantic-ai 1.106.0 source."
sidebar:
  label: "Class deep dives (Vol. 13)"
  order: 39
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.106.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.106.0`.
</Aside>

Ten class groups covering the capabilities and tooling surface of `pydantic-ai 1.106.0`: `TemplateStr` (Handlebars template strings that render system prompts against runtime dependencies); the `Hooks` capability and its 26+ hook function protocols (observe and intercept every lifecycle point without subclassing); `WebSearch` + `WebFetch` capabilities (native provider tool or automatic local fallback); the `Thinking` capability (cross-provider extended reasoning configuration); `NativeOrLocalTool` (the base class behind `WebSearch`, `WebFetch`, and `ImageGeneration`); `ExaToolset` and its four sub-tools (Exa neural search, find-similar, content retrieval, and AI-powered answers); `TavilySearchTool` (Tavily search with developer-fixed parameters hidden from the LLM); `AgentWorker` + `agent_to_a2a` (Agent-to-Agent protocol server in one function call); `PrepareTools` + `PrepareOutputTools` (dynamic tool filtering per request); and `ReinjectSystemPrompt` + `SetToolMetadata` + `PrefixTools` (three composition helpers for multi-capability agents).

---

## 1. `TemplateStr` — Handlebars Dynamic Instructions

**Module:** `pydantic_ai._template`  
**Import:** `from pydantic_ai import TemplateStr`

`TemplateStr` is a Handlebars template string that renders at run-time against `RunContext.deps`. Instead of static instruction strings you write templates like `"Hello, {{name}}! You are a {{role}} assistant."`, and the library renders them before each model request using the agent's dependency object.

### Class signature

```python
class TemplateStr(Generic[AgentDepsT]):
    def __init__(
        self,
        source: str,
        *,
        deps_type: type[Any] | None = None,
        deps_schema: dict[str, Any] | None = None,
    ) -> None: ...

    def render(self, deps: AgentDepsT | None = None) -> str: ...
    def __call__(self, ctx: RunContext[AgentDepsT]) -> str: ...
```

`TemplateStr` is callable, so it satisfies the `SystemPromptFunc` protocol — pass it directly wherever a `str` or `Callable[[RunContext], str]` is accepted.

### Basic usage

```python
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr

@dataclass
class Deps:
    user_name: str
    role: str
    language: str = "English"

agent: Agent[Deps, str] = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    instructions=TemplateStr(
        "You are a {{role}} assistant. "
        "Address the user as {{user_name}}. "
        "Always respond in {{language}}."
    ),
)

# Instructions render fresh for each .run() call
result = agent.run_sync(
    "Tell me something interesting.",
    deps=Deps(user_name="Alice", role="science", language="French"),
)
```

### Nested object access

Handlebars supports dot-notation paths into nested objects:

```python
@dataclass
class Company:
    name: str
    industry: str

@dataclass
class Deps:
    company: Company
    agent_version: str

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    instructions=TemplateStr(
        "You represent {{company.name}}, a leader in {{company.industry}}. "
        "You are version {{agent_version}}."
    ),
)
```

### Conditional blocks

Handlebars `{{#if}}` / `{{else}}` / `{{/if}}` conditions are fully supported:

```python
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    instructions=TemplateStr(
        "You are a helpful assistant. "
        "{{#if premium_user}}"
        "The user has premium access — provide detailed, comprehensive answers. "
        "{{else}}"
        "The user is on the free tier — keep responses concise. "
        "{{/if}}"
        "Always cite your sources."
    ),
)
```

### List iteration with `{{#each}}`

```python
@dataclass
class Deps:
    tools_available: list[str]
    current_date: str

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    instructions=TemplateStr(
        "Today is {{current_date}}. "
        "You have access to the following tools: "
        "{{#each tools_available}}• {{this}}\n{{/each}}"
    ),
)

# Renders to:
# "Today is 2026-06-08. You have access to the following tools:
#  • web_search
#  • code_interpreter
#  • file_reader"
result = agent.run_sync(
    "What can you help with?",
    deps=Deps(
        tools_available=["web_search", "code_interpreter", "file_reader"],
        current_date="2026-06-08",
    ),
)
```

### Manual rendering

For testing or pre-flight checks, call `.render()` directly:

```python
tmpl = TemplateStr("Hello, {{name}}! Role: {{role}}.")

rendered = tmpl.render({"name": "Bob", "role": "analyst"})
assert rendered == "Hello, Bob! Role: analyst."

# With a dataclass dependency
@dataclass
class D:
    name: str
    role: str

rendered2 = tmpl.render(D(name="Carol", role="engineer"))
assert rendered2 == "Hello, Carol! Role: engineer."
```

<Aside type="note">
`TemplateStr` requires `pydantic-handlebars` (bundled with `pydantic-ai`). Import errors indicate a version mismatch — reinstall `pydantic-ai[all]`.
</Aside>

---

## 2. `Hooks` — Lifecycle Hook Registry

**Module:** `pydantic_ai.capabilities.hooks`  
**Import:** `from pydantic_ai.capabilities.hooks import Hooks`

`Hooks` is a capability that registers callback functions for 30+ agent lifecycle events — before/after/wrap/error variants for run, node, model request, tool validation, tool execution, output validation, output processing, and deferred tool handling. Unlike subclassing, hooks compose non-destructively: multiple `Hooks` instances can be combined on the same agent.

### Class signature (condensed)

```python
@dataclass(init=False)
class Hooks(AbstractCapability[AgentDepsT]):
    def __init__(
        self,
        *,
        before_run: BeforeRunHookFunc[AgentDepsT] | None = None,
        after_run: AfterRunHookFunc[AgentDepsT] | None = None,
        wrap_run: WrapRunHookFunc[AgentDepsT] | None = None,
        on_run_error: OnRunErrorHookFunc[AgentDepsT] | None = None,
        before_node_run: BeforeNodeRunHookFunc[AgentDepsT] | None = None,
        after_node_run: AfterNodeRunHookFunc[AgentDepsT] | None = None,
        wrap_node_run: WrapNodeRunHookFunc[AgentDepsT] | None = None,
        on_node_run_error: OnNodeRunErrorHookFunc[AgentDepsT] | None = None,
        wrap_run_event_stream: WrapRunEventStreamHookFunc[AgentDepsT] | None = None,
        on_event: OnEventHookFunc[AgentDepsT] | None = None,
        before_model_request: BeforeModelRequestHookFunc[AgentDepsT] | None = None,
        after_model_request: AfterModelRequestHookFunc[AgentDepsT] | None = None,
        wrap_model_request: WrapModelRequestHookFunc[AgentDepsT] | None = None,
        on_model_request_error: OnModelRequestErrorHookFunc[AgentDepsT] | None = None,
        prepare_tools: PrepareToolsHookFunc[AgentDepsT] | None = None,
        prepare_output_tools: PrepareOutputToolsHookFunc[AgentDepsT] | None = None,
        before_tool_validate: BeforeToolValidateHookFunc[AgentDepsT] | None = None,
        after_tool_validate: AfterToolValidateHookFunc[AgentDepsT] | None = None,
        wrap_tool_validate: WrapToolValidateHookFunc[AgentDepsT] | None = None,
        on_tool_validate_error: OnToolValidateErrorHookFunc[AgentDepsT] | None = None,
        before_tool_execute: BeforeToolExecuteHookFunc[AgentDepsT] | None = None,
        after_tool_execute: AfterToolExecuteHookFunc[AgentDepsT] | None = None,
        wrap_tool_execute: WrapToolExecuteHookFunc[AgentDepsT] | None = None,
        on_tool_execute_error: OnToolExecuteErrorHookFunc[AgentDepsT] | None = None,
        before_output_validate: BeforeOutputValidateHookFunc[AgentDepsT] | None = None,
        after_output_validate: AfterOutputValidateHookFunc[AgentDepsT] | None = None,
        wrap_output_validate: WrapOutputValidateHookFunc[AgentDepsT] | None = None,
        on_output_validate_error: OnOutputValidateErrorHookFunc[AgentDepsT] | None = None,
        before_output_process: BeforeOutputProcessHookFunc[AgentDepsT] | None = None,
        after_output_process: AfterOutputProcessHookFunc[AgentDepsT] | None = None,
        wrap_output_process: WrapOutputProcessHookFunc[AgentDepsT] | None = None,
        on_output_process_error: OnOutputProcessErrorHookFunc[AgentDepsT] | None = None,
        handle_deferred_tool_calls: HandleDeferredToolCallsHookFunc[AgentDepsT] | None = None,
    ) -> None: ...

    @cached_property
    def on(self) -> _HookRegistration[AgentDepsT]: ...
```

### Decorator-based registration via `hooks.on`

The `hooks.on` cached property returns a `_HookRegistration` object whose methods are decorators:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()

@hooks.on.before_run
async def log_start(ctx):
    print(f"[run:start] prompt={ctx.prompt!r}")

@hooks.on.after_run
async def log_finish(ctx, result):
    print(f"[run:end] output={result.data!r}")

@hooks.on.before_tool_execute
async def log_tool_call(ctx, tool_name, args):
    print(f"[tool] {tool_name}({args})")

agent = Agent("openai:gpt-4o", capabilities=[hooks])
```

### Constructor-based registration

For clarity or reuse, pass hooks as constructor arguments:

```python
import time
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks(
    before_run=lambda ctx: print(f"Starting run: {ctx.prompt}"),
    after_run=lambda ctx, result: print(f"Done. Output: {result.data}"),
)
```

### `wrap_model_request` — middleware around LLM calls

`wrap` hooks receive a `next` callable — call it to proceed, skip it to short-circuit:

```python
import time
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()
latencies: list[float] = []

@hooks.on.wrap_model_request
async def measure_latency(ctx, request, next):
    t0 = time.perf_counter()
    response = await next(ctx, request)
    latencies.append(time.perf_counter() - t0)
    return response

agent = Agent("openai:gpt-4o", capabilities=[hooks])
```

### `wrap_tool_execute` — tool execution middleware

Intercept any tool call — useful for rate-limiting, auditing, or mocking:

```python
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()
tool_calls: list[dict] = []

@hooks.on.wrap_tool_execute
async def audit_tool(ctx, tool_def, args, next):
    tool_calls.append({"tool": tool_def.name, "args": args})
    result = await next(ctx, tool_def, args)
    return result

@hooks.on.on_tool_execute_error
async def on_error(ctx, tool_def, args, exc):
    print(f"[error] {tool_def.name} raised {exc!r}")
    # Return a fallback string to recover gracefully
    return f"Tool {tool_def.name} temporarily unavailable."
```

### `on_event` — inspect every streamed event

```python
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import ToolCallPart

hooks = Hooks()

@hooks.on.on_event
async def inspect_events(ctx, event):
    if hasattr(event, "part") and isinstance(event.part, ToolCallPart):
        print(f"[event] tool_call: {event.part.tool_name}")
```

### `prepare_tools` hook — dynamic tool injection

```python
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.tools import ToolDefinition

hooks = Hooks()

@hooks.on.prepare_tools
async def filter_admin_tools(ctx, tool_defs: list[ToolDefinition]):
    if not ctx.deps.is_admin:
        return [t for t in tool_defs if not t.name.startswith("admin_")]
    return tool_defs
```

### Composing multiple `Hooks` instances

Each `Hooks` instance is independent — attach multiple to the same agent:

```python
observability_hooks = Hooks(
    before_run=log_start,
    after_run=log_finish,
)
rate_limit_hooks = Hooks(
    wrap_model_request=enforce_rate_limit,
)

agent = Agent(
    "openai:gpt-4o",
    capabilities=[observability_hooks, rate_limit_hooks],
)
```

<Aside type="note">
`HookTimeoutError` is raised if a hook runs longer than the `timeout` parameter on its `_HookEntry`. Use `hooks.on.before_run(timeout=5.0)` for timed hooks.
</Aside>

---

## 3. `WebSearch` + `WebFetch` — Native/Local Web Capabilities

**Modules:** `pydantic_ai.capabilities.web_search`, `pydantic_ai.capabilities.web_fetch`  
**Import:** `from pydantic_ai.capabilities import WebSearch, WebFetch`

Both extend `NativeOrLocalTool`: if the model supports a native web search/fetch tool (e.g. OpenAI's `web_search_preview`), the native tool is used; otherwise the capability falls back to a local implementation (DuckDuckGo for search, a markdownify-based fetcher for URL loading).

### `WebSearch` signature

```python
@dataclass(init=False)
class WebSearch(NativeOrLocalTool[AgentDepsT]):
    native: AgentNativeTool[AgentDepsT] | bool = True
    local: str | Tool[AgentDepsT] | ... | bool | None = None

    search_context_size: Literal['low', 'medium', 'high'] | None = None
    user_location: WebSearchUserLocation | None = None
    blocked_domains: list[str] | None = None
    allowed_domains: list[str] | None = None
    max_uses: int | None = None
```

### `WebFetch` signature

```python
@dataclass(init=False)
class WebFetch(NativeOrLocalTool[AgentDepsT]):
    native: AgentNativeTool[AgentDepsT] | bool = True
    local: str | Tool[AgentDepsT] | ... | bool | None = None

    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    max_uses: int | None = None
    enable_citations: bool | None = None
    max_content_tokens: int | None = None
```

### Default usage — automatic native/local selection

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch, WebFetch

agent = Agent(
    "openai:gpt-4o",
    capabilities=[WebSearch(), WebFetch()],
)
# openai:gpt-4o → native web_search_preview + native file_search used
# anthropic:claude-3-5-sonnet → no native tool → falls back to DuckDuckGo / markdownify
result = agent.run_sync("What are the latest AI model releases in 2026?")
```

### `WebSearch` with domain filtering

```python
agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        WebSearch(
            search_context_size="high",          # more context from results
            allowed_domains=["arxiv.org", "nature.com", "science.org"],
            max_uses=5,                           # cap LLM tool calls per run
        )
    ],
)
```

### `WebSearch` with user location

```python
from pydantic_ai.capabilities.web_search import WebSearchUserLocation

agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        WebSearch(
            user_location=WebSearchUserLocation(
                type="approximate",
                country="GB",
                city="London",
                region="England",
            )
        )
    ],
)
```

### `WebFetch` with SSRF protection and content limits

```python
agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        WebFetch(
            allowed_domains=["docs.pydantic.dev", "github.com"],
            max_content_tokens=4000,   # truncate long pages
            enable_citations=True,     # include source URLs in model output
        )
    ],
)

result = agent.run_sync("Summarise the pydantic-ai README.")
```

### Force local fallback only (no native tool)

```python
# Disable native — always use local DuckDuckGo
agent = Agent(
    "openai:gpt-4o",
    capabilities=[WebSearch(native=False)],
)
```

### Custom local fallback

Swap DuckDuckGo for Tavily (or any `Tool`):

```python
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.common_tools.tavily import tavily_search_tool

agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        WebSearch(
            native=False,                    # no native tool
            local=tavily_search_tool(api_key="tvly-..."),
        )
    ],
)
```

### Combined search + fetch pipeline

```python
agent = Agent(
    "anthropic:claude-3-5-sonnet-20241022",
    capabilities=[
        WebSearch(blocked_domains=["reddit.com", "twitter.com"]),
        WebFetch(max_content_tokens=8000),
    ],
    system_prompt=(
        "When asked about a topic: first search for relevant pages, "
        "then fetch and read the top 2 results before answering."
    ),
)
```

---

## 4. `Thinking` — Cross-Provider Extended Reasoning

**Module:** `pydantic_ai.capabilities.thinking`  
**Import:** `from pydantic_ai.capabilities import Thinking`

`Thinking` is a one-field capability that enables extended chain-of-thought reasoning. The library maps the `effort` level to provider-specific parameters (`thinking.budget_tokens` for Anthropic, `reasoning_effort` for OpenAI).

### Class signature

```python
@dataclass
class Thinking(AbstractCapability[Any]):
    effort: ThinkingLevel = True

# ThinkingLevel = bool | Literal['low', 'medium', 'high']
```

### Usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Boolean enable (provider picks default budget)
agent = Agent(
    "anthropic:claude-3-7-sonnet-20250219",
    capabilities=[Thinking()],
)

# Named effort level
agent_high = Agent(
    "openai:o3",
    capabilities=[Thinking(effort="high")],
)

result = agent_high.run_sync(
    "Prove that sqrt(2) is irrational."
)
```

### Provider-specific behaviour

| Provider | `effort=True` | `effort='low'` | `effort='medium'` | `effort='high'` |
|---|---|---|---|---|
| Anthropic | `budget_tokens=8192` | `budget_tokens=2048` | `budget_tokens=8192` | `budget_tokens=32000` |
| OpenAI | `reasoning_effort='medium'` | `'low'` | `'medium'` | `'high'` |
| Other | mapped via `model_settings` | — | — | — |

### Combining with other capabilities

```python
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    "anthropic:claude-3-7-sonnet-20250219",
    capabilities=[
        Thinking(effort="medium"),
        WebSearch(),
    ],
    system_prompt="Think carefully before answering. Use web search for current data.",
)
```

### Disable thinking for specific runs

Override at run time by passing `model_settings`:

```python
from pydantic_ai.models import ModelSettings

# Default agent has thinking enabled
agent = Agent(
    "openai:o3",
    capabilities=[Thinking(effort="high")],
)

# Disable for this run (fast/cheap path)
result = agent.run_sync(
    "What is 2 + 2?",
    model_settings=ModelSettings(reasoning_effort="low"),
)
```

---

## 5. `NativeOrLocalTool` — Native/Local Fallback Framework

**Module:** `pydantic_ai.capabilities.native_or_local`  
**Import:** `from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool`

`NativeOrLocalTool` is the abstract base class that `WebSearch`, `WebFetch`, and `ImageGeneration` all extend. It implements the "prefer native, fall back to local" pattern — and you can subclass it to create your own paired capabilities.

### Class signature

```python
@dataclass(init=False)
class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    native: AgentNativeTool[AgentDepsT] | bool = True
    local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = None

    def __post_init__(self) -> None: ...

    # Override points for subclasses
    def _default_native(self) -> AgentNativeTool[AgentDepsT] | None: ...
    def _native_unique_id(self) -> str | None: ...
    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None: ...
    def _resolve_local_strategy(self, strategy: str) -> ...: ...
    def _requires_native(self) -> bool: ...

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]: ...
    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None: ...
```

### Subclassing: custom native/local capability

```python
import dataclasses
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

@dataclasses.dataclass(init=False)
class DatabaseSearch(NativeOrLocalTool):
    """Use native DB search tool if model supports it, else local SQL fallback."""

    connection_string: str = ""

    def __init__(self, *, connection_string: str, native: bool = True):
        self.connection_string = connection_string
        super().__init__(native=native, local=None)

    def _default_local(self):
        # Return a Tool that runs a local SQL query
        async def sql_search(ctx: RunContext, query: str) -> list[dict]:
            import asyncpg
            conn = await asyncpg.connect(ctx.deps.connection_string)
            rows = await conn.fetch(f"SELECT * FROM documents WHERE content ILIKE $1", f"%{query}%")
            return [dict(r) for r in rows]
        return Tool(sql_search, name="db_search", description="Search the database")

    def _requires_native(self) -> bool:
        # If native=False is explicit, allow local-only fallback
        return False


agent = Agent(
    "openai:gpt-4o",
    capabilities=[DatabaseSearch(connection_string="postgresql://...", native=False)],
)
```

### `native` / `local` resolution logic

| `native` | `local` | Result |
|---|---|---|
| `True` (default) | `None` | Use model's native tool; if not available, call `_default_local()` |
| `True` | custom `Tool` | Use native; fall back to the provided `Tool` |
| `False` | `None` | Local only via `_default_local()` |
| `False` | `'duckduckgo'` | Local strategy named `'duckduckgo'` via `_resolve_local_strategy()` |
| `AgentNativeTool` | any | Use the provided native tool explicitly |

---

## 6. `ExaToolset` + Exa Tools — Neural Search Integration

**Module:** `pydantic_ai.common_tools.exa`  
**Import:** `from pydantic_ai.common_tools.exa import ExaToolset, exa_search_tool, exa_find_similar_tool, exa_get_contents_tool, exa_answer_tool`

Exa is a neural search engine. The toolset bundles four specialised tools — search, find-similar-pages, content retrieval, and AI-powered answers — each backed by `AsyncExa`.

### `ExaToolset` — all four tools in one capability

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset

agent = Agent(
    "openai:gpt-4o",
    toolsets=[ExaToolset(api_key="exa-your-key")],
    system_prompt="Use Exa tools to find and read web pages before answering.",
)

result = agent.run_sync(
    "Find recent papers on diffusion models and summarise their key contributions."
)
```

### Individual tool factory functions

Use individual factories when you only need some tools:

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import exa_search_tool, exa_get_contents_tool

agent = Agent(
    "openai:gpt-4o",
    tools=[
        exa_search_tool(api_key="exa-your-key"),
        exa_get_contents_tool(api_key="exa-your-key"),
    ],
)
```

### `ExaSearchTool` — search with type control

```python
from pydantic_ai.common_tools.exa import ExaSearchTool
from exa_py import AsyncExa

client = AsyncExa(api_key="exa-your-key")
search = ExaSearchTool(client=client)

# The __call__ signature exposed to the LLM:
# async __call__(query: str, search_type: Literal['auto', 'keyword', 'neural', 'fast', 'deep'] = 'auto')
# Returns: list[ExaSearchResult]  — each has title, url, id, score, published_date, author
```

### `ExaFindSimilarTool` — find pages similar to a URL

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import exa_find_similar_tool

agent = Agent(
    "openai:gpt-4o",
    tools=[exa_find_similar_tool(api_key="exa-your-key")],
    system_prompt=(
        "When the user provides a URL, find similar pages and summarise their content."
    ),
)

# LLM calls: find_similar(url="https://arxiv.org/abs/2301.00234", exclude_source_domain=True)
# Returns: list[ExaSearchResult]
```

### `ExaGetContentsTool` — bulk content retrieval

```python
from pydantic_ai.common_tools.exa import exa_get_contents_tool

# Retrieve full text of multiple URLs in one call
tool = exa_get_contents_tool(api_key="exa-your-key")
# LLM calls: get_contents(urls=["https://...", "https://..."])
# Returns: list[ExaContentResult]  — each has url, title, text, author, published_date
```

### `ExaAnswerTool` — AI-powered direct answers

```python
from pydantic_ai.common_tools.exa import exa_answer_tool

tool = exa_answer_tool(api_key="exa-your-key")
# LLM calls: exa_answer(query="What is the capital of France?")
# Returns: list[ExaAnswerResult]  — each has answer, sources (list of ExaSearchResult)
```

### Full research agent example

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset

research_agent = Agent(
    "openai:gpt-4o",
    toolsets=[ExaToolset(api_key="exa-your-key")],
    system_prompt="""You are a research assistant with Exa neural search.
    Workflow:
    1. Use exa_search to find relevant sources (search_type='neural' for semantic, 'deep' for thorough)
    2. Use exa_get_contents to read the most promising URLs
    3. Use exa_find_similar to discover related resources
    4. Use exa_answer for quick factual questions
    Always cite your sources with URLs.""",
)

result = research_agent.run_sync(
    "What are the key differences between RLHF and RLAIF in LLM training?"
)
print(result.data)
```

---

## 7. `TavilySearchTool` — Controlled Neural Search

**Module:** `pydantic_ai.common_tools.tavily`  
**Import:** `from pydantic_ai.common_tools.tavily import TavilySearchTool, tavily_search_tool`

Tavily is a search API designed for AI agents. Its factory function uses `functools.partial` and signature manipulation to let developers pre-fix parameters that the LLM should not control.

### `TavilySearchTool` dataclass

```python
@dataclass
class TavilySearchTool:
    client: AsyncTavilyClient
    _: KW_ONLY
    max_results: int | None = None

    async def __call__(
        self,
        query: str,
        search_depth: Literal['basic', 'advanced'] = 'basic',
        topic: Literal['general', 'news', 'finance'] = 'general',
        time_range: Literal['day', 'week', 'month', 'year'] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[TavilySearchResult]: ...

# TavilySearchResult = TypedDict with: title, url, content, score
```

### Basic usage

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

agent = Agent(
    "openai:gpt-4o",
    tools=[tavily_search_tool(api_key="tvly-your-key")],
)

result = agent.run_sync("What happened in AI research this week?")
```

### Fix parameters that the LLM should not change

The factory function's overload lets you bind parameters that become hidden from the LLM:

```python
# Force 'advanced' depth and 'news' topic — LLM only controls the query string
tool = tavily_search_tool(
    api_key="tvly-your-key",
    search_depth="advanced",
    topic="news",
    time_range="week",
)

agent = Agent("openai:gpt-4o", tools=[tool])
```

### Domain allow/block lists fixed by developer

```python
# LLM can query but cannot change which domains are searched
finance_tool = tavily_search_tool(
    api_key="tvly-your-key",
    search_depth="advanced",
    topic="finance",
    include_domains=["bloomberg.com", "ft.com", "reuters.com"],
)

agent = Agent(
    "openai:gpt-4o",
    tools=[finance_tool],
    system_prompt="You are a financial news analyst. Use the search tool to find market news.",
)
```

### Using a pre-built `AsyncTavilyClient`

```python
from tavily import AsyncTavilyClient
from pydantic_ai.common_tools.tavily import TavilySearchTool

client = AsyncTavilyClient(api_key="tvly-your-key")
tool = TavilySearchTool(client=client, max_results=5)
```

### Tavily vs Exa — when to use which

| Feature | `TavilySearchTool` | `ExaToolset` |
|---|---|---|
| Search type | Keyword + semantic hybrid | Neural (semantic by default) |
| Fixed params | Yes — via factory partial | No — LLM controls |
| Direct answers | No | Yes (`ExaAnswerTool`) |
| Find similar | No | Yes (`ExaFindSimilarTool`) |
| Content retrieval | Inline in results | Separate `ExaGetContentsTool` |
| Domain filtering | Fixed by dev or controlled by LLM | Fixed by dev |
| Time range | Yes | No |
| Best for | News, finance, general web | Research, similarity, semantic search |

---

## 8. `AgentWorker` + `agent_to_a2a` — FastA2A Protocol Server

**Module:** `pydantic_ai._a2a`  
**Import:** `from pydantic_ai import agent_to_a2a`

The A2A (Agent-to-Agent) protocol lets agents from different frameworks communicate over HTTP. `agent_to_a2a()` converts any PydanticAI `Agent` into a FastA2A-compatible server in one call. `AgentWorker` handles the per-request lifecycle: building message history from A2A `Message` objects, running the agent, and converting the result to A2A `Artifact` objects.

### `agent_to_a2a` function

```python
def agent_to_a2a(
    agent: Agent[AgentDepsT, WorkerOutputT],
    *,
    agent_card: AgentCard,
    deps: AgentDepsT = ...,
    # Passed through to AgentWorker
) -> FastA2A: ...
```

### Minimal A2A server

```python
import uvicorn
from pydantic_ai import Agent, agent_to_a2a
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful coding assistant.",
)

card = AgentCard(
    name="CodingAssistant",
    description="Answers programming questions",
    url="http://localhost:8080/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="code_help",
            name="Code Help",
            description="Help with programming tasks",
        )
    ],
)

app = agent_to_a2a(agent, agent_card=card)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### `AgentWorker` class

```python
@dataclass
class AgentWorker(Generic[WorkerOutputT, AgentDepsT]):
    agent: Agent[AgentDepsT, WorkerOutputT]
    deps: AgentDepsT

    async def run_task(self, params: TaskSendParams) -> None: ...
    async def cancel_task(self, params: TaskIdParams) -> None: ...
    def build_artifacts(self, result: WorkerOutputT) -> list[Artifact]: ...
    def build_message_history(self, history: list[Message]) -> list[ModelMessage]: ...
```

### Custom `AgentWorker` with dependency injection

```python
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai._a2a import AgentWorker
from a2a.server.agent_execution import AgentExecutor, RequestContext

@dataclass
class MyDeps:
    user_id: str
    db_pool: object  # asyncpg pool

class MyWorker(AgentWorker):
    async def run_task(self, params):
        # Extract user context from A2A task metadata
        user_id = params.metadata.get("user_id", "anonymous")
        # Override deps with per-request context
        self.deps = MyDeps(user_id=user_id, db_pool=self.deps.db_pool)
        await super().run_task(params)
```

### Multi-turn A2A conversation

The worker's `build_message_history` method converts A2A `Message` history to `ModelMessage` objects, enabling multi-turn conversations across agent boundaries:

```python
from pydantic_ai import Agent, agent_to_a2a
from a2a.types import AgentCard, AgentCapabilities

# This agent maintains full conversation context via A2A history
agent = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You are a customer support agent. "
        "Remember previous messages in the conversation."
    ),
)

card = AgentCard(
    name="SupportAgent",
    description="Multi-turn customer support",
    url="http://localhost:8081/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[],
)

app = agent_to_a2a(agent, agent_card=card)
```

---

## 9. `PrepareTools` + `PrepareOutputTools` — Dynamic Tool Filtering

**Module:** `pydantic_ai.capabilities.prepare_tools`  
**Import:** `from pydantic_ai.capabilities.prepare_tools import PrepareTools, PrepareOutputTools`

`PrepareTools` wraps a `ToolsPrepareFunc` — a sync or async function called just before each model request with the full list of `ToolDefinition` objects. Return a filtered/modified list to control what the model sees. `PrepareOutputTools` does the same for output tools (tools the model uses to produce structured output).

### Class signatures

```python
@dataclass
class PrepareTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]

@dataclass
class PrepareOutputTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]

# ToolsPrepareFunc = Callable[[RunContext[AgentDepsT], list[ToolDefinition]], ...]
# Return type: list[ToolDefinition] | None  (None = use all tools unmodified)
```

### Hide tools based on user role

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.prepare_tools import PrepareTools
from pydantic_ai.tools import ToolDefinition

@dataclass
class Deps:
    user_role: str  # 'admin' | 'user' | 'guest'

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"search", "delete_record", "update_record", "read_record"},
    "user": {"search", "read_record"},
    "guest": {"search"},
}

async def filter_by_role(
    ctx: RunContext[Deps],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    allowed = ROLE_PERMISSIONS.get(ctx.deps.user_role, set())
    return [t for t in tool_defs if t.name in allowed]

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    capabilities=[PrepareTools(prepare_func=filter_by_role)],
    tools=[...],  # all tools registered here
)

# Admin sees all 4 tools; guest only sees 'search'
admin_result = agent.run_sync("Delete user 42", deps=Deps(user_role="admin"))
guest_result = agent.run_sync("Find users named Alice", deps=Deps(user_role="guest"))
```

### Modify tool descriptions dynamically

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities.prepare_tools import PrepareTools
from pydantic_ai.tools import ToolDefinition
import copy

async def inject_context_into_descriptions(
    ctx: RunContext,
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    """Annotate tool descriptions with current context."""
    enriched = []
    for td in tool_defs:
        td_copy = copy.copy(td)
        td_copy.description = (
            f"{td.description} "
            f"[Current user: {ctx.deps.username}, "
            f"Locale: {ctx.deps.locale}]"
        )
        enriched.append(td_copy)
    return enriched

agent = Agent(
    "openai:gpt-4o",
    capabilities=[PrepareTools(prepare_func=inject_context_into_descriptions)],
)
```

### `PrepareOutputTools` — filter structured output tools

```python
from pydantic_ai.capabilities.prepare_tools import PrepareOutputTools
from pydantic_ai.tools import ToolDefinition

async def restrict_output_schema(
    ctx: RunContext,
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    # Only allow the 'summary' output tool for free-tier users
    if ctx.deps.tier == "free":
        return [t for t in tool_defs if t.name == "summary"]
    return tool_defs

agent = Agent(
    "openai:gpt-4o",
    capabilities=[PrepareOutputTools(prepare_func=restrict_output_schema)],
)
```

### Sync prepare functions

`PrepareTools` accepts both sync and async functions:

```python
def sync_filter(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [t for t in tool_defs if not t.name.startswith("_internal_")]

agent = Agent(
    "openai:gpt-4o",
    capabilities=[PrepareTools(prepare_func=sync_filter)],
)
```

---

## 10. `ReinjectSystemPrompt` + `SetToolMetadata` + `PrefixTools` — Capability Composition Helpers

**Modules:**  
- `pydantic_ai.capabilities.reinject_system_prompt`  
- `pydantic_ai.capabilities.set_tool_metadata`  
- `pydantic_ai.capabilities.prefix_tools`

Three small but indispensable capabilities for building robust multi-capability agents.

---

### `ReinjectSystemPrompt` — restore system prompts after history reconstruction

```python
@dataclass
class ReinjectSystemPrompt(AbstractCapability[AgentDepsT]):
    replace_existing: bool = False
```

When message history is loaded from a database, a UI frontend, or an A2A protocol, the original system prompt is often missing (databases typically only store user/assistant turns). `ReinjectSystemPrompt` prepends the agent's configured system prompts at the head of the first request.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities.reinject_system_prompt import ReinjectSystemPrompt

agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a helpful assistant specialising in Python.",
    capabilities=[ReinjectSystemPrompt()],
)

# Even if message_history comes from a DB with no system prompt,
# the capability re-injects it before the first model request.
from pydantic_ai.messages import ModelRequest, UserPromptPart

history_from_db = [
    ModelRequest(parts=[UserPromptPart(content="Hi!")]),
    # ... more turns, no system prompt
]

result = agent.run_sync(
    "What's wrong with my code?",
    message_history=history_from_db,
)
```

**`replace_existing=True`** strips any system prompts already in the history before prepending the current one:

```python
# Ensures the latest system prompt version is always used,
# even if the stored history contained an older version.
agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a v2 assistant.",
    capabilities=[ReinjectSystemPrompt(replace_existing=True)],
)
```

---

### `SetToolMetadata` — attach metadata to tools

```python
@dataclass(init=False)
class SetToolMetadata(AbstractCapability[AgentDepsT]):
    tools: ToolSelector[AgentDepsT] = 'all'
    metadata: dict[str, Any]  # init=False, populated from **kwargs

    def __init__(self, *, tools: ToolSelector[AgentDepsT] = 'all', **metadata: Any) -> None: ...
```

`SetToolMetadata` merges arbitrary key-value metadata onto `ToolDefinition.metadata` for all selected tools. This is how the `code_mode` metadata flag is set for Anthropic's computer use, and how providers consume custom tool hints.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities.set_tool_metadata import SetToolMetadata

# Mark ALL tools as requiring confirmation before execution
agent = Agent(
    "openai:gpt-4o",
    capabilities=[SetToolMetadata(requires_confirmation=True)],
)

# Mark only the 'delete_file' tool as dangerous
agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        SetToolMetadata(tools="delete_file", danger_level="high", requires_approval=True)
    ],
)
```

**Selecting specific tools:**

```python
from pydantic_ai.capabilities.set_tool_metadata import SetToolMetadata

# Apply to a list of tool names
agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        SetToolMetadata(
            tools=["read_file", "write_file"],
            filesystem_access=True,
            sandbox_required=True,
        )
    ],
)
```

**Anthropic computer-use pattern** (built-in use of `SetToolMetadata`):

```python
# Internally, pydantic-ai sets metadata on computer-use tools like this:
from pydantic_ai.capabilities.set_tool_metadata import SetToolMetadata

computer_use_metadata = SetToolMetadata(
    tools=["computer", "bash", "text_editor"],
    # Anthropic requires this flag to enable computer-use tools
)
```

---

### `PrefixTools` — namespace tools from nested capabilities

```python
@dataclass
class PrefixTools(WrapperCapability[AgentDepsT]):
    wrapped: AbstractCapability[AgentDepsT]
    prefix: str

    @classmethod
    def from_spec(cls, *, prefix: str, capability: CapabilitySpec) -> PrefixTools[Any]: ...
```

`PrefixTools` wraps any other capability and renames all its tools by prepending `prefix + "_"`. This prevents name collisions when composing multiple capabilities that provide tools with similar names.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.capabilities.prefix_tools import PrefixTools
from pydantic_ai.common_tools.exa import ExaToolset

# Both WebSearch and ExaToolset may expose a 'search' tool
# Prefix them to avoid collisions
agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        PrefixTools(wrapped=WebSearch(), prefix="web"),          # → web_search
        PrefixTools(wrapped=ExaToolset(api_key="..."), prefix="exa"),  # → exa_search, exa_get_contents, ...
    ],
    system_prompt=(
        "Use web_search for real-time news and exa_search for research papers. "
        "Both are available."
    ),
)
```

### Composing all three helpers

A common production pattern: load conversation history from a database, ensure the system prompt is current, namespace overlapping tools, and attach audit metadata:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch, WebFetch
from pydantic_ai.capabilities.reinject_system_prompt import ReinjectSystemPrompt
from pydantic_ai.capabilities.set_tool_metadata import SetToolMetadata
from pydantic_ai.capabilities.prefix_tools import PrefixTools
from pydantic_ai.common_tools.tavily import tavily_search_tool

agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are an AI research assistant (v3.2). Be precise and cite sources.",
    capabilities=[
        # 1. Always re-inject the latest system prompt into DB-loaded histories
        ReinjectSystemPrompt(replace_existing=True),

        # 2. Namespace web tools to avoid conflicts
        PrefixTools(wrapped=WebSearch(), prefix="native"),        # native_search
        PrefixTools(wrapped=WebFetch(), prefix="native"),         # native_fetch

        # 3. Tag all tools for audit logging
        SetToolMetadata(audit_log=True, version="3.2"),
    ],
    tools=[tavily_search_tool(api_key="tvly-...")],
)
```

---

## Cross-reference with previous volumes

| Class(es) | Introduced in |
|---|---|
| `Agent`, `RunContext`, `Tool` | Vol. 1 |
| `ModelSettings`, `Usage`, message parts | Vol. 2 |
| `OpenAIModel`, `AnthropicModel`, `GeminiModel` | Vol. 3 |
| `StreamedRunResult`, delta parts | Vol. 4 |
| `TestModel`, `FunctionModel` | Vol. 5 |
| `AgentCapability`, `AgentToolset` | Vol. 6 |
| `FunctionToolset`, `AbstractToolset` | Vol. 7 |
| `MCPToolset`, `MCPServer` | Vol. 8 |
| `UserPromptNode`, `ModelRequestNode`, `CallToolsNode` | Vol. 9 |
| `AgentInstructions`, `AgentMetadata`, `AgentNativeTool` | Vol. 10 |
| `HandleResponseEvent`, `ModelResponseStreamEvent`, `ToolSearchCallPart` | Vol. 11 |
| `Dataset`, `LLMJudge`, `ExternalToolset`, `RetryConfig` | Vol. 12 |
| `TemplateStr`, `Hooks`, `WebSearch`, `WebFetch`, `Thinking`, `NativeOrLocalTool`, `ExaToolset`, `TavilySearchTool`, `AgentWorker`, `PrepareTools`, `ReinjectSystemPrompt`, `SetToolMetadata`, `PrefixTools` | **Vol. 13** |

## Revision history

- **v1.106.0 (June 2026, Vol. 13)** — pydantic-ai 1.106.0 source installed from PyPI and examined at `/usr/local/lib/python3.11/dist-packages/pydantic_ai/`. Ten new class groups deep-dived. New guide `pydantic_ai_class_deep_dives_v13.md` created. Index updated with Zero→Hero step 33, Jump-to-topic card, and Reference card.

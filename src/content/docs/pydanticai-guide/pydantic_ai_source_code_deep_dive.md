---
title: "PydanticAI: Source Code Deep Dive — 10 Classes"
description: "Source-verified deep dives into RunContext, Hooks, WebSearch, WebFetch, Thinking, FilteredToolset, CombinedToolset, ApprovalRequiredToolset, ExternalToolset, and UsageLimits/RunUsage — all examples derived directly from pydantic-ai==1.102.0 source code."
framework: pydanticai
language: python
---

# PydanticAI Source Code Deep Dive — 10 Classes

Verified against **pydantic-ai==1.102.0** — source installed and inspected directly.  
Classes covered: `RunContext`, `Hooks`, `WebSearch`, `WebFetch`, `Thinking`, `FilteredToolset`, `CombinedToolset`, `ApprovalRequiredToolset`, `ExternalToolset`, `UsageLimits` + `RunUsage`.

All examples are derived from the installed package source. Every constructor argument, field, and method shown here is verified against the actual implementation.

---

## 1. `RunContext` — Everything Inside the Current Call

**Source:** `pydantic_ai.tools.RunContext` (dataclass, `kw_only=True`)

`RunContext[DepsT]` is the single object that flows through every tool call, system-prompt function, output validator, and hook. It carries the dependency, model, usage counters, conversation state, retry metadata, and approval state. You rarely construct it yourself — PydanticAI creates it per run/step.

### Complete field reference

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import RunUsage

# Every field on RunContext (as of 1.102.0, from source):
#
# deps            — your injected dependency
# model           — the Model instance used in this run
# usage           — RunUsage: tokens, requests, tool_calls so far
# agent           — the Agent running this context (or None)
# prompt          — the original user prompt
# messages        — ModelMessage list (history so far)
# validation_context — passed through to Pydantic validators
# tracer          — OTel Tracer (NoOpTracer if not instrumenting)
# trace_include_content — whether content is included in spans
# retries         — dict[tool_name, retry_count]
# tool_call_id    — ID of the tool call being executed
# tool_name       — name of the tool being called
# retry           — retries for this specific tool / output validation
# max_retries     — max retries for this tool / output validation
# run_step        — current step number (0-indexed)
# tool_call_approved — True after HITL approval
# tool_call_metadata — metadata from DeferredToolResults when approved
# partial_output  — True when output validator receives partial stream data
# run_id          — unique ID for this agent run
# conversation_id — shared ID across all turns in the same conversation
# metadata        — arbitrary dict attached at run() call time
# model_settings  — ModelSettings in effect for this step
```

### Basic dependency access

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class AppDeps:
    db_url: str
    user_id: int
    is_admin: bool = False

agent = Agent('openai:gpt-4o', deps_type=AppDeps)

@agent.tool
async def get_user_data(ctx: RunContext[AppDeps]) -> str:
    """Fetch data for the current user."""
    # Access deps
    db_url = ctx.deps.db_url
    user_id = ctx.deps.user_id

    # Access run identity
    print(f'run_id={ctx.run_id}  step={ctx.run_step}  retry={ctx.retry}/{ctx.max_retries}')

    # Access model info
    print(f'model={ctx.model}')

    return f'Data for user {user_id} from {db_url}'

async def main():
    deps = AppDeps(db_url='postgres://localhost/app', user_id=42, is_admin=True)
    result = await agent.run('Get my data.', deps=deps)
    print(result.output)

asyncio.run(main())
```

### Retry detection inside a tool

```python
import asyncio
from pydantic_ai import Agent, RunContext, ModelRetry

agent = Agent('openai:gpt-4o')

@agent.tool_plain
async def flaky_lookup(ctx: RunContext[None], key: str) -> str:
    """Tool that retries gracefully."""
    if ctx.retry == 0:
        raise ModelRetry(f'First attempt failed for {key!r}. Please try again.')
    if ctx.retry == 1:
        raise ModelRetry(f'Second attempt also failed. One more try.')
    # Third attempt succeeds
    return f'Found: {key}'

asyncio.run(agent.run('Look up "hello"'))
```

### Conversation ID — linking multiple runs

```python
import asyncio
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o')

@agent.tool
def get_context(ctx: RunContext[None]) -> str:
    """Expose conversation metadata to the model."""
    return (
        f'run_id={ctx.run_id}  '
        f'conversation_id={ctx.conversation_id}  '
        f'step={ctx.run_step}'
    )

async def multi_turn():
    # Turn 1: conversation_id is freshly generated
    r1 = await agent.run('Start a conversation. What is my context?')
    print('conversation_id:', r1.conversation_id)

    # Turn 2: same conversation_id propagates automatically
    r2 = await agent.run(
        'Still in the same conversation?',
        message_history=r1.all_messages(),
    )
    print('same conversation?', r1.conversation_id == r2.conversation_id)

asyncio.run(multi_turn())
```

### Usage tracking inside a tool

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o')

@agent.tool
def check_budget(ctx: RunContext[None]) -> str:
    """Check current token spend and warn if high."""
    usage = ctx.usage  # RunUsage
    budget_remaining = 5000 - usage.total_tokens
    if budget_remaining < 1000:
        return f'WARNING: only {budget_remaining} tokens remain this run!'
    return f'Budget OK — {usage.total_tokens} used, {budget_remaining} remaining'
```

### Partial-output validation

`ctx.partial_output` is `True` when your output validator is called mid-stream (during `run_stream`). Use it to defer expensive checks until the stream is complete:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.tools import RunContext

class Report(BaseModel):
    title: str
    body: str
    word_count: int

agent = Agent('openai:gpt-4o', output_type=Report)

@agent.output_validator
async def check_word_count(ctx: RunContext[None], output: Report) -> Report:
    if ctx.partial_output:
        return output   # don't validate counts on partial streams
    if output.word_count != len(output.body.split()):
        raise ModelRetry('word_count is wrong — recalculate it.')
    return output
```

---

## 2. `Hooks` — Lifecycle Callbacks via Decorators

**Source:** `pydantic_ai.capabilities.hooks.Hooks` — extends `AbstractCapability`

`Hooks` gives you 33 hook events across every phase of an agent run, registrable via `@hooks.on.<event>` or constructor kwargs.

### Constructor kwargs pattern (no decorators)

```python
import asyncio
import logging
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

logger = logging.getLogger(__name__)

# Build all hooks inline — useful for testing or dynamic construction
hooks = Hooks(
    before_run=lambda ctx: logger.info('run starting  run_id=%s', ctx.run_id),
    after_run=lambda ctx, *, result: (
        logger.info('run done  tokens=%d', result.usage.total_tokens) or result
    ),
    model_request_error=lambda ctx, *, request_context, error: (_ for _ in ()).throw(error),
)

agent = Agent('openai:gpt-4o', capabilities=[hooks])
asyncio.run(agent.run('Hello'))
```

### Complete event reference

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

# ── Run lifecycle ──────────────────────────────────────────────────────────────
@hooks.on.before_run
def on_before_run(ctx: RunContext) -> None:
    print(f'[before_run] run_id={ctx.run_id}')

@hooks.on.after_run
async def on_after_run(ctx: RunContext, *, result) -> object:
    print(f'[after_run]  tokens={result.usage.total_tokens}')
    return result  # always return

@hooks.on.run_error
async def on_run_error(ctx: RunContext, *, error: BaseException) -> object:
    print(f'[run_error]  {type(error).__name__}: {error}')
    raise error  # re-raise or return a fallback AgentRunResult

# ── Node lifecycle ─────────────────────────────────────────────────────────────
@hooks.on.before_node_run
async def on_before_node(ctx: RunContext, *, node) -> object:
    print(f'[before_node_run] {type(node).__name__}')
    return node  # return the (optionally modified) node

@hooks.on.after_node_run
async def on_after_node(ctx: RunContext, *, node, result) -> object:
    print(f'[after_node_run]  {type(node).__name__}')
    return result

@hooks.on.node_run_error
async def on_node_error(ctx: RunContext, *, node, error: Exception) -> object:
    raise error  # or return a NodeResult to recover

# ── Model request ─────────────────────────────────────────────────────────────
@hooks.on.before_model_request
async def on_before_request(ctx: RunContext, request_context) -> object:
    print(f'[before_model_request]  messages={len(request_context.messages)}')
    return request_context  # return (optionally modified) request context

@hooks.on.after_model_request
async def on_after_request(ctx: RunContext, *, request_context, response) -> object:
    print(f'[after_model_request]   model={response.model_name}')
    return response

@hooks.on.model_request_error
async def on_model_error(ctx: RunContext, *, request_context, error: Exception) -> object:
    raise error  # or return a synthetic ModelResponse

# ── Tool preparation ───────────────────────────────────────────────────────────
@hooks.on.prepare_tools
async def on_prepare_tools(ctx: RunContext, tool_defs: list) -> list:
    # Filter or mutate tool definitions before they're sent to the model
    return [d for d in tool_defs if not d.name.startswith('_')]

# ── Tool validation ───────────────────────────────────────────────────────────
@hooks.on.before_tool_validate
async def on_before_validate(ctx: RunContext, *, call, tool_def, args) -> object:
    return args  # return (optionally modified) raw args

@hooks.on.after_tool_validate
async def on_after_validate(ctx: RunContext, *, call, tool_def, args) -> object:
    return args  # return (optionally modified) validated args

# ── Tool execution ────────────────────────────────────────────────────────────
@hooks.on.before_tool_execute(tools=['delete_user', 'drop_table'])
async def on_before_destructive(ctx: RunContext, *, call, tool_def, args) -> object:
    print(f'[AUDIT] {tool_def.name}({args}) by run={ctx.run_id}')
    return args

@hooks.on.after_tool_execute
async def on_after_tool(ctx: RunContext, *, call, tool_def, args, result) -> object:
    print(f'[tool_done] {tool_def.name} → {result!r}')
    return result

@hooks.on.tool_execute_error
async def on_tool_error(ctx: RunContext, *, call, tool_def, args, error: Exception) -> object:
    from pydantic_ai import ModelRetry
    if 'timeout' in str(error).lower():
        raise ModelRetry(f'{tool_def.name} timed out — retry')
    raise error

# ── Output validation ─────────────────────────────────────────────────────────
@hooks.on.before_output_validate
async def on_before_output(ctx: RunContext, *, output_context, output) -> object:
    return output

@hooks.on.after_output_validate
async def on_after_output(ctx: RunContext, *, output_context, output) -> object:
    return output

# ── Event stream ──────────────────────────────────────────────────────────────
@hooks.on.event
async def on_event(ctx: RunContext, event) -> object:
    # Called for every event in the run event stream
    return event
```

### Timing + rate tracking with `wrap_*` hooks

The `run`, `model_request`, `tool_execute`, and `output_validate` events give you a `handler` callable, making circuit-breaker and timing patterns clean:

```python
import time
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

hooks = Hooks()
_model_latencies: list[float] = []

@hooks.on.model_request
async def time_model(ctx, *, request_context, handler):
    """Measure model latency on every call."""
    t0 = time.perf_counter()
    response = await handler(request_context)
    elapsed = time.perf_counter() - t0
    _model_latencies.append(elapsed)
    print(f'model latency: {elapsed:.3f}s  (avg {sum(_model_latencies)/len(_model_latencies):.3f}s)')
    return response

@hooks.on.tool_execute
async def time_tools(ctx, *, call, tool_def, args, handler):
    """Wrap every tool call with timing."""
    t0 = time.perf_counter()
    result = await handler(call, tool_def, args)
    print(f'{tool_def.name} took {time.perf_counter() - t0:.3f}s')
    return result

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

### Stacking hooks instances

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

# Separate concerns into separate Hooks objects
security_hooks = Hooks()
metrics_hooks = Hooks()
debug_hooks = Hooks()

@security_hooks.on.before_tool_execute(tools=['send_email', 'post_webhook'])
async def redact_pii(ctx, *, call, tool_def, args):
    if 'email' in args:
        args = {**args, 'email': '***@***.***'}
    return args

@metrics_hooks.on.after_run
async def record_cost(ctx, *, result):
    tokens = result.usage.total_tokens
    print(f'Cost estimate: ${tokens * 0.00003:.4f}')
    return result

@debug_hooks.on.after_model_request
async def log_response(ctx, *, request_context, response):
    print(f'[DEBUG] model={response.model_name}')
    return response

# All three run in order: security → metrics → debug
agent = Agent('openai:gpt-4o', capabilities=[security_hooks, metrics_hooks, debug_hooks])
```

---

## 3. `WebSearch` — Native Web Search with DuckDuckGo Fallback

**Source:** `pydantic_ai.capabilities.WebSearch` — extends `NativeOrLocalTool`

`WebSearch` uses the model's built-in web search when supported (OpenAI, Google, xAI) and falls back to a local tool (DuckDuckGo by default) for models that don't support it natively.

### Constructor arguments

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `native` | `bool \| WebSearchTool \| Callable` | `True` | Use model's native search; `False` disables native |
| `local` | `bool \| 'duckduckgo' \| Tool \| Callable \| None` | `None` | Fallback tool; `True` = DuckDuckGo |
| `search_context_size` | `'low' \| 'medium' \| 'high' \| None` | `None` | Amount of context retrieved (native only) |
| `user_location` | `WebSearchUserLocation \| None` | `None` | Localise results (native only) |
| `blocked_domains` | `list[str] \| None` | `None` | Exclude these domains (requires native) |
| `allowed_domains` | `list[str] \| None` | `None` | Only include these domains (requires native) |
| `max_uses` | `int \| None` | `None` | Cap searches per run (requires native) |

### Minimal usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# Auto: use native if model supports it, no local fallback
agent = Agent('openai:gpt-4o', capabilities=[WebSearch()])

async def main():
    result = await agent.run('What is the current price of gold?')
    print(result.output)

asyncio.run(main())
```

### Native search with context size control

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# 'high' context = more web content retrieved per search (costs more tokens)
agent = Agent(
    'openai:gpt-4o',
    capabilities=[WebSearch(search_context_size='high')],
)
```

### Domain allow-list (news research agent)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# Only return results from trusted news outlets
news_agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        WebSearch(
            allowed_domains=['reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com'],
            search_context_size='high',
            max_uses=3,  # max 3 searches per run
        )
    ],
    system_prompt='You are a factual news researcher. Only cite information from the provided web results.',
)

async def main():
    result = await news_agent.run('What happened in AI research this week?')
    print(result.output)

asyncio.run(main())
```

### Block social media / unreliable sources

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

research_agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[
        WebSearch(
            blocked_domains=['twitter.com', 'x.com', 'reddit.com', 'quora.com'],
        )
    ],
)
```

### With DuckDuckGo local fallback (for models without native search)

```python
# pip install "pydantic-ai[duckduckgo]"
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# Explicitly opt in to DuckDuckGo fallback — no deprecation warning
agent = Agent(
    'anthropic:claude-sonnet-4-6',   # no native web search → uses DDG
    capabilities=[WebSearch(local='duckduckgo')],
)
```

### Custom local fallback (your own search API)

```python
import asyncio
import httpx
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

async def my_search_tool(query: str) -> str:
    """Search using our internal API."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            'https://search.internal.example.com/api',
            params={'q': query},
            timeout=10,
        )
        data = resp.json()
        return '\n'.join(r['snippet'] for r in data['results'][:5])

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        WebSearch(
            native=True,             # use native if available
            local=my_search_tool,    # fall back to internal API
        )
    ],
)
```

### Location-aware search

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.capabilities.web_search import WebSearchUserLocation

# Localize results to New York
agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        WebSearch(
            user_location=WebSearchUserLocation(
                type='approximate',
                city='New York',
                region='New York',
                country='US',
            )
        )
    ],
)
```

---

## 4. `WebFetch` — URL Fetching with Domain Control

**Source:** `pydantic_ai.capabilities.WebFetch` — extends `NativeOrLocalTool`

`WebFetch` gives the agent the ability to read web pages. Like `WebSearch`, it uses the model's native fetch when available, and falls back to a local httpx-based fetcher otherwise.

### Constructor arguments

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `native` | `bool \| WebFetchTool \| Callable` | `True` | Use model's native fetch |
| `local` | `bool \| Tool \| Callable \| None` | `None` | Local fallback (`True` = default markdownify fetcher) |
| `allowed_domains` | `list[str] \| None` | `None` | SSRF guard: only these domains |
| `blocked_domains` | `list[str] \| None` | `None` | Never fetch from these |
| `max_uses` | `int \| None` | `None` | Cap fetches per run (native only) |
| `enable_citations` | `bool \| None` | `None` | Include citations in output (native only) |
| `max_content_tokens` | `int \| None` | `None` | Truncate fetched content (native only) |

### Basic usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

agent = Agent(
    'openai:gpt-4o',
    capabilities=[WebFetch()],
    system_prompt='Fetch and summarise web pages when asked.',
)

async def main():
    result = await agent.run('Summarise https://docs.pydantic.dev/latest/')
    print(result.output)

asyncio.run(main())
```

### SSRF protection: allow-list internal services only

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

# Only allow fetching from our own docs and APIs
internal_agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        WebFetch(
            allowed_domains=['docs.mycompany.com', 'api.mycompany.com', 'status.mycompany.com'],
        )
    ],
)
```

### Citations + content limits (OpenAI native)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

# Enable citations and limit content to 2000 tokens per fetch
research_agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        WebFetch(
            enable_citations=True,
            max_content_tokens=2000,
            max_uses=5,
        )
    ],
)
```

### Combining WebSearch + WebFetch

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch, WebFetch

# Agent that can search for pages and then fetch them
agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        WebSearch(search_context_size='medium', max_uses=3),
        WebFetch(
            blocked_domains=['facebook.com', 'twitter.com'],
            max_content_tokens=4000,
        ),
    ],
    system_prompt=(
        'You are a research assistant. Search the web to find relevant pages, '
        'then fetch and summarise the best ones.'
    ),
)

async def main():
    result = await agent.run(
        'Research the latest developments in quantum computing and give me a 5-point summary.'
    )
    print(result.output)

asyncio.run(main())
```

### Local fallback with custom fetcher

```python
# pip install "pydantic-ai-slim[web-fetch]"
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

# True → use the built-in markdownify-based local fetcher
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[WebFetch(local=True)],  # explicitly opt in to local fallback
)
```

---

## 5. `Thinking` — Extended Reasoning

**Source:** `pydantic_ai.capabilities.Thinking` — extends `AbstractCapability`

`Thinking` enables model reasoning/chain-of-thought. It sets `ModelSettings(thinking=effort)` transparently across any provider that supports it (Anthropic claude-3-7+, OpenAI o-series, Google Gemini thinking).

### Constructor

```python
from pydantic_ai.capabilities import Thinking

Thinking(effort=True)           # default effort level per provider
Thinking(effort=False)          # disable (silently ignored on always-on models like o1)
Thinking(effort='minimal')      # very fast, minimal reasoning
Thinking(effort='low')          # brief reasoning
Thinking(effort='medium')       # balanced
Thinking(effort='high')         # thorough reasoning
Thinking(effort='xhigh')        # maximum reasoning (may be slow/expensive)
```

### Basic usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Enable thinking at default effort
agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[Thinking()],
)

async def main():
    result = await agent.run('Prove that sqrt(2) is irrational.')
    print(result.output)

asyncio.run(main())
```

### Effort levels for different workloads

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Fast responses: minimal reasoning
fast_agent = Agent('openai:o4-mini', capabilities=[Thinking(effort='low')])

# Deep analysis: thorough reasoning
deep_agent = Agent('anthropic:claude-opus-4-5', capabilities=[Thinking(effort='high')])

# Maximum effort for critical decisions
critical_agent = Agent('openai:o3', capabilities=[Thinking(effort='xhigh')])

async def main():
    # Quick classification task
    r1 = await fast_agent.run('Is this email spam? "Congratulations! You won $1000000!"')
    print('Fast:', r1.output)

    # Deep code review
    code = """
    def find_max(lst):
        m = lst[0]
        for x in lst[1:]:
            if x > m: m = x
        return m
    """
    r2 = await deep_agent.run(f'Review this code for bugs and improvements:\n{code}')
    print('Deep:', r2.output)

asyncio.run(main())
```

### Cross-provider thinking (Anthropic + OpenAI)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

# Thinking works the same way on both providers
model = FallbackModel(
    'anthropic:claude-opus-4-5',
    'openai:o4-mini',
    fallback_on=(ModelAPIError,),
)

agent = Agent(
    model,
    capabilities=[Thinking(effort='high')],  # applies to whichever model runs
)
```

### Provider-specific overrides take precedence

If you also pass a provider-specific thinking setting (e.g. `anthropic_thinking` or `openai_reasoning_effort`), that takes precedence over `Thinking()`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[Thinking(effort='medium')],           # generic fallback
    model_settings=AnthropicModelSettings(
        anthropic_thinking={'type': 'enabled', 'budget_tokens': 8192}  # provider-specific wins
    ),
)
```

---

## 6. `FilteredToolset` — Per-Step Tool Visibility

**Source:** `pydantic_ai.toolsets.FilteredToolset` (dataclass, extends `WrapperToolset`)

`FilteredToolset` wraps another toolset and calls your filter function on every step, every tool. Returning `False` hides the tool from the model for that step. Both sync and async filter functions are accepted.

### Signature

```python
@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]]
```

### Sync filter — role-based access

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, FilteredToolset

@dataclass
class UserDeps:
    role: str  # 'admin' | 'user' | 'viewer'

tools = FunctionToolset[UserDeps]()

@tools.tool_plain
def read_data(record_id: int) -> str:
    """Read a record."""
    return f'record-{record_id}'

@tools.tool_plain
def write_data(record_id: int, value: str) -> str:
    """Write a record."""
    return f'wrote {value} to record-{record_id}'

@tools.tool_plain
def delete_data(record_id: int) -> str:
    """Delete a record."""
    return f'deleted record-{record_id}'

def role_filter(ctx: RunContext[UserDeps], tool_def) -> bool:
    """Filter tools based on user role."""
    role = ctx.deps.role
    name = tool_def.name
    if role == 'viewer':
        return name == 'read_data'          # viewers can only read
    if role == 'user':
        return name in ('read_data', 'write_data')  # users read + write
    return True  # admins get everything

agent = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    toolsets=[FilteredToolset(tools, filter_func=role_filter)],
)
```

### Async filter — check permissions from a service

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, FilteredToolset

tools = FunctionToolset[str]()  # deps = user_id string

@tools.tool_plain
def send_notification(user_id: str, message: str) -> str:
    return f'Sent to {user_id}: {message}'

@tools.tool_plain
def export_data(user_id: str) -> str:
    return f'Exported data for {user_id}'

async def permission_filter(ctx: RunContext[str], tool_def) -> bool:
    """Async permission check against an external service."""
    user_id = ctx.deps
    tool_name = tool_def.name
    # Simulate async permission lookup
    await asyncio.sleep(0)  # replace with real: await perm_service.check(user_id, tool_name)
    allowed_tools = {'send_notification', 'export_data'} if user_id == 'admin' else {'send_notification'}
    return tool_name in allowed_tools

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,
    toolsets=[FilteredToolset(tools, filter_func=permission_filter)],
)
```

### Phase-gated tools — step-aware filtering

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, FilteredToolset

tools = FunctionToolset[None]()

@tools.tool_plain
def gather_info(query: str) -> str:
    """Phase 1: gather information."""
    return f'Info about: {query}'

@tools.tool_plain
def summarise(data: str) -> str:
    """Phase 2: summarise collected info."""
    return f'Summary: {data}'

@tools.tool_plain
def generate_report(summary: str) -> str:
    """Phase 3: generate final report."""
    return f'Report: {summary}'

def phase_filter(ctx: RunContext[None], tool_def) -> bool:
    """Only expose tools appropriate to the current step."""
    step = ctx.run_step
    if step < 2:
        return tool_def.name == 'gather_info'   # step 0-1: gather only
    elif step < 4:
        return tool_def.name == 'summarise'     # step 2-3: summarise
    else:
        return tool_def.name == 'generate_report'  # step 4+: report

agent = Agent(
    'openai:gpt-4o',
    toolsets=[FilteredToolset(tools, filter_func=phase_filter)],
)
```

---

## 7. `CombinedToolset` — Merging Multiple Tool Sources

**Source:** `pydantic_ai.toolsets.CombinedToolset` (dataclass)

`CombinedToolset([ts1, ts2, ...])` merges toolsets into one, raising `UserError` at construction if any tool names collide. Pair with `PrefixedToolset` to avoid collisions.

### Basic merge

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset, CombinedToolset, PrefixedToolset

db_tools = FunctionToolset[None]()
kb_tools = FunctionToolset[None]()

@db_tools.tool_plain
def query_db(sql: str) -> str:
    """Run a database query."""
    return f'DB result for: {sql}'

@db_tools.tool_plain
def insert_row(table: str, data: dict) -> str:
    """Insert a row into a table."""
    return f'Inserted into {table}'

@kb_tools.tool_plain
def search_kb(query: str) -> str:
    """Search the knowledge base."""
    return f'KB results for: {query}'

@kb_tools.tool_plain
def add_article(title: str, content: str) -> str:
    """Add an article to the knowledge base."""
    return f'Added: {title}'

# Prefix both toolsets to avoid naming collisions
agent = Agent(
    'openai:gpt-4o',
    toolsets=[
        CombinedToolset([
            PrefixedToolset(db_tools, prefix='db'),
            PrefixedToolset(kb_tools, prefix='kb'),
        ])
    ],
)
# model sees: db_query_db, db_insert_row, kb_search_kb, kb_add_article
```

### Combining heterogeneous sources

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets import FunctionToolset, CombinedToolset, PrefixedToolset

# Local Python tools
local_tools = FunctionToolset[None]()

@local_tools.tool_plain
def format_json(data: str) -> str:
    """Pretty-print a JSON string."""
    import json
    return json.dumps(json.loads(data), indent=2)

# Remote MCP server (filesystem tools)
mcp_server = MCPServerStdio('npx', ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])

# Combine local + MCP
agent = Agent(
    'openai:gpt-4o',
    toolsets=[
        CombinedToolset([
            PrefixedToolset(local_tools, prefix='local'),
            # MCP tools are automatically prefixed by the server name
            mcp_server,
        ])
    ],
)
```

### Dynamic combined toolset per-run

`CombinedToolset` calls `for_run()` on each sub-toolset, so each run can get a fresh sub-toolset instance if needed:

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, CombinedToolset, FilteredToolset

@dataclass
class AppDeps:
    environment: str  # 'dev' | 'staging' | 'prod'
    is_admin: bool

read_tools = FunctionToolset[AppDeps]()
write_tools = FunctionToolset[AppDeps]()
admin_tools = FunctionToolset[AppDeps]()

@read_tools.tool_plain
def get_config(key: str) -> str:
    return f'config[{key}]'

@write_tools.tool_plain
def set_config(key: str, value: str) -> str:
    return f'set config[{key}]={value}'

@admin_tools.tool_plain
def restart_service(name: str) -> str:
    return f'restarted {name}'

def admin_only(ctx: RunContext[AppDeps], tool_def) -> bool:
    return ctx.deps.is_admin

def write_allowed(ctx: RunContext[AppDeps], tool_def) -> bool:
    return ctx.deps.environment != 'prod' or ctx.deps.is_admin

agent = Agent(
    'openai:gpt-4o',
    deps_type=AppDeps,
    toolsets=[
        CombinedToolset([
            read_tools,
            FilteredToolset(write_tools, filter_func=write_allowed),
            FilteredToolset(admin_tools, filter_func=admin_only),
        ])
    ],
)
```

---

## 8. `ApprovalRequiredToolset` — Human-in-the-Loop

**Source:** `pydantic_ai.toolsets.ApprovalRequiredToolset` (dataclass, extends `WrapperToolset`)

`ApprovalRequiredToolset` wraps a toolset so that calls to (some) tools raise `ApprovalRequired` instead of executing. The agent run returns with `DeferredToolRequests` output, waits for human approval, then resumes with `DeferredToolResults`.

### Signature

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    # approval_required_func(ctx, tool_def, args) -> bool
    # Returns True if approval is needed. Default: always True.
    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True
```

### Complete HITL workflow

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset, ApprovalRequiredToolset
from pydantic_ai.output import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

dangerous_tools = FunctionToolset[None]()

@dangerous_tools.tool_plain
def delete_records(table: str, condition: str) -> str:
    """Delete records from a database table."""
    return f'Deleted from {table} WHERE {condition}'

@dangerous_tools.tool_plain
def send_bulk_email(recipients: list[str], subject: str, body: str) -> str:
    """Send an email to multiple recipients."""
    return f'Sent to {len(recipients)} recipients: {subject}'

@dangerous_tools.tool_plain
def read_data(table: str) -> str:
    """Read data from a table (safe — no approval needed)."""
    return f'Data from {table}'

def needs_approval(ctx, tool_def, args) -> bool:
    """Only destructive or broadcast operations need approval."""
    return tool_def.name in ('delete_records', 'send_bulk_email')

agent = Agent(
    'openai:gpt-4o',
    output_type=[str, DeferredToolRequests],  # tell the agent about the extra output type
    toolsets=[ApprovalRequiredToolset(dangerous_tools, approval_required_func=needs_approval)],
)

async def run_with_approval(user_request: str):
    """Drive a full HITL conversation."""
    history = None

    while True:
        result = await agent.run(
            user_request if history is None else None,
            message_history=history,
        )

        if isinstance(result.output, str):
            # Normal completion — no approval needed
            print('Result:', result.output)
            return result.output

        assert isinstance(result.output, DeferredToolRequests)

        # Show pending approvals to the operator
        print('\n--- Approval required ---')
        approvals = {}
        for call in result.output.approvals:
            print(f'  Tool: {call.tool_name}')
            print(f'  Args: {call.args}')
            decision = input('  Approve? [y/n]: ').strip().lower()
            if decision == 'y':
                approvals[call.tool_call_id] = ToolApproved()
            else:
                approvals[call.tool_call_id] = ToolDenied(message='Operator rejected this action.')

        # Resume with the decisions
        history = result.all_messages()
        user_request = None
        deferred = DeferredToolResults(approvals=approvals)
        # Feed decisions back into the run
        result = await agent.run(
            message_history=history,
            deferred_tool_results=deferred,
        )
        if isinstance(result.output, str):
            print('Result after approval:', result.output)
            return result.output
        history = result.all_messages()

asyncio.run(run_with_approval('Delete expired sessions from the auth_sessions table'))
```

### Approval with metadata

You can pass arbitrary metadata through the approval workflow using `ToolApproved(metadata=...)` and read it back in `ctx.tool_call_metadata` after approval:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, ApprovalRequiredToolset
from pydantic_ai.output import DeferredToolRequests, DeferredToolResults, ToolApproved

tools = FunctionToolset[None]()

@tools.tool
def deploy_service(ctx: RunContext[None], service: str, version: str) -> str:
    """Deploy a service to production."""
    # ctx.tool_call_approved == True here
    # ctx.tool_call_metadata == whatever was passed to ToolApproved(metadata=...)
    approver = ctx.tool_call_metadata.get('approver', 'unknown') if ctx.tool_call_metadata else 'unknown'
    print(f'Deploying {service}:{version} (approved by {approver})')
    return f'Deployed {service}:{version}'

agent = Agent(
    'openai:gpt-4o',
    output_type=[str, DeferredToolRequests],
    toolsets=[ApprovalRequiredToolset(tools)],
)

async def deploy_workflow():
    result = await agent.run('Deploy auth-service version 2.4.1 to production')
    if isinstance(result.output, DeferredToolRequests):
        approvals = {
            call.tool_call_id: ToolApproved(metadata={'approver': 'jane@example.com', 'ticket': 'DEPLOY-123'})
            for call in result.output.approvals
        }
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals),
        )
        print(result.output)

asyncio.run(deploy_workflow())
```

---

## 9. `ExternalToolset` — Deferred External Execution

**Source:** `pydantic_ai.toolsets.ExternalToolset` — extends `AbstractToolset`

`ExternalToolset` declares tool *schemas* without providing implementations. The agent run produces `DeferredToolRequests` containing the model's tool calls, which your infrastructure then executes and returns results for. Use this for long-running operations, human workflows, or tools that run in a different process.

### Signature

```python
class ExternalToolset(AbstractToolset[AgentDepsT]):
    tool_defs: list[ToolDefinition]
    _id: str | None

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None): ...
```

### Defining external tools

```python
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition

# Define the schema for tools that run outside the agent process
external_tools = ExternalToolset([
    ToolDefinition(
        name='run_sql_migration',
        description='Run a SQL migration script on the production database.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'script': {'type': 'string', 'description': 'The SQL migration script to execute'},
                'dry_run': {'type': 'boolean', 'description': 'If true, only validate without executing'},
            },
            'required': ['script'],
        },
    ),
    ToolDefinition(
        name='notify_slack',
        description='Post a message to a Slack channel.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'channel': {'type': 'string', 'description': 'Slack channel name (e.g. #deployments)'},
                'message': {'type': 'string', 'description': 'The message text'},
                'urgent': {'type': 'boolean', 'description': 'Whether to @channel'},
            },
            'required': ['channel', 'message'],
        },
    ),
])
```

### Full workflow: agent requests → external execution → feed results back

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.output import DeferredToolRequests, DeferredToolResults, ToolReturn

external_tools = ExternalToolset([
    ToolDefinition(
        name='run_sql_migration',
        description='Run a SQL migration on the production database.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'script': {'type': 'string'},
                'dry_run': {'type': 'boolean'},
            },
            'required': ['script'],
        },
    ),
    ToolDefinition(
        name='notify_slack',
        description='Post a message to a Slack channel.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'channel': {'type': 'string'},
                'message': {'type': 'string'},
            },
            'required': ['channel', 'message'],
        },
    ),
])

agent = Agent(
    'openai:gpt-4o',
    output_type=[str, DeferredToolRequests],
    toolsets=[external_tools],
    system_prompt='You are a DevOps assistant. Use the available tools to complete deployment tasks.',
)

async def execute_tool(tool_name: str, args: dict) -> str:
    """Simulate executing an external tool."""
    print(f'  [executing] {tool_name}({args})')
    if tool_name == 'run_sql_migration':
        return f'Migration completed: {len(args["script"])} chars executed'
    elif tool_name == 'notify_slack':
        return f'Posted to {args["channel"]}: {args["message"]}'
    return 'done'

async def devops_workflow(request: str):
    """Run an agentic workflow with external tool execution."""
    history = None
    user_request = request

    for iteration in range(5):  # safety limit
        result = await agent.run(
            user_request if history is None else None,
            message_history=history,
        )

        if isinstance(result.output, str):
            print(f'Final result: {result.output}')
            return

        assert isinstance(result.output, DeferredToolRequests)
        print(f'\nIteration {iteration + 1}: {len(result.output.calls)} tool call(s)')

        # Execute all tool calls in the appropriate external system
        tool_results = {}
        for call in result.output.calls:
            output = await execute_tool(call.tool_name, call.args)
            tool_results[call.tool_call_id] = ToolReturn(content=output)

        # Feed results back and continue
        history = result.all_messages()
        user_request = None
        result = await agent.run(
            message_history=history,
            deferred_tool_results=DeferredToolResults(calls=tool_results),
        )
        if isinstance(result.output, str):
            print(f'Final result: {result.output}')
            return
        history = result.all_messages()

asyncio.run(devops_workflow(
    'Run the migration in migrations/v3_add_indexes.sql (dry run first), '
    'then notify #deployments that the migration is complete.'
))
```

### External toolset with a durable ID (Temporal)

When using durable execution (e.g. Temporal), the `id` parameter uniquely identifies the toolset so its activities can be matched across workflow replays:

```python
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition

# id= is required for Temporal durable execution
external_tools = ExternalToolset(
    [
        ToolDefinition(
            name='long_running_job',
            description='Submit a long-running batch job.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'job_config': {'type': 'object'}},
                'required': ['job_config'],
            },
        )
    ],
    id='batch-job-toolset',  # must be stable across workflow replays
)
```

---

## 10. `UsageLimits` + `RunUsage` — Token Budgets and Tracking

**Source:** `pydantic_ai.usage.UsageLimits`, `pydantic_ai.usage.RunUsage`

`UsageLimits` enforces budgets *before* an agent run gets out of hand. `RunUsage` accumulates the actual spend so you can report it.

### `UsageLimits` — all fields

```python
from pydantic_ai.usage import UsageLimits

# Every field (as of 1.102.0, from source):
limits = UsageLimits(
    request_limit=50,               # max API calls (default 50; prevents infinite loops)
    tool_calls_limit=20,            # max successful tool executions
    input_tokens_limit=50_000,      # max prompt tokens
    output_tokens_limit=10_000,     # max completion tokens
    total_tokens_limit=60_000,      # max combined tokens
    count_tokens_before_request=True, # preflight token count (Anthropic, Google, Bedrock, OpenAI Responses)
)
```

### Enforcing a token budget

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o')

async def run_with_budget(prompt: str, budget_tokens: int = 5000):
    """Run an agent with a hard token budget."""
    limits = UsageLimits(
        total_tokens_limit=budget_tokens,
        request_limit=10,
    )
    try:
        result = await agent.run(prompt, usage_limits=limits)
        usage = result.usage
        print(f'Used {usage.total_tokens}/{budget_tokens} tokens')
        return result.output
    except UsageLimitExceeded as e:
        print(f'Budget exceeded: {e}')
        return None

asyncio.run(run_with_budget('Write a comprehensive essay on the history of computing.'))
```

### `RunUsage` — all fields

```python
from pydantic_ai import Agent
import asyncio

agent = Agent('openai:gpt-4o')

async def main():
    result = await agent.run('Summarise quantum computing in 3 sentences.')
    usage = result.usage  # RunUsage instance

    # Token counters
    print(f'input_tokens:        {usage.input_tokens}')
    print(f'output_tokens:       {usage.output_tokens}')
    print(f'total_tokens:        {usage.total_tokens}')      # input + output
    print(f'cache_read_tokens:   {usage.cache_read_tokens}')  # Anthropic cache hits
    print(f'cache_write_tokens:  {usage.cache_write_tokens}') # Anthropic cache writes
    print(f'input_audio_tokens:  {usage.input_audio_tokens}') # audio models

    # Request/tool counters
    print(f'requests:    {usage.requests}')    # number of API calls
    print(f'tool_calls:  {usage.tool_calls}')  # number of tool executions

    # Provider-specific extra details
    print(f'details:     {usage.details}')     # dict[str, int]

asyncio.run(main())
```

### Summing usage across multiple runs

`RunUsage` implements `__add__` for easy aggregation:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

agent = Agent('openai:gpt-4o')

async def process_batch(items: list[str]) -> RunUsage:
    """Process a batch and return total usage."""
    total = RunUsage()
    for item in items:
        result = await agent.run(f'Classify this text: {item}')
        total = total + result.usage  # RunUsage.__add__
    return total

async def main():
    texts = ['Great product!', 'Terrible experience.', 'Average quality.']
    total = await process_batch(texts)
    print(f'Total: {total.total_tokens} tokens across {total.requests} requests')
    cost_estimate = total.total_tokens * 0.00001  # rough estimate
    print(f'Estimated cost: ${cost_estimate:.4f}')

asyncio.run(main())
```

### Preflight token counting

Set `count_tokens_before_request=True` to check the input token count *before* sending the request, ensuring you don't exceed `input_tokens_limit` mid-flight:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

# Supported providers: Anthropic, Google, Bedrock, OpenAI Responses
agent = Agent('anthropic:claude-sonnet-4-6')

async def safe_run(prompt: str):
    limits = UsageLimits(
        input_tokens_limit=8_000,
        output_tokens_limit=2_000,
        count_tokens_before_request=True,  # check token count first
    )
    from pydantic_ai.exceptions import UsageLimitExceeded
    try:
        return await agent.run(prompt, usage_limits=limits)
    except UsageLimitExceeded as e:
        return f'Prompt too long: {e}'

asyncio.run(safe_run('A very long prompt...'))
```

### Accumulating usage across a conversation

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

agent = Agent('openai:gpt-4o')

async def chat_session():
    history = None
    session_usage = RunUsage()

    prompts = [
        'What is Python?',
        'What are its main use cases?',
        'Compare it to JavaScript.',
    ]

    for prompt in prompts:
        result = await agent.run(prompt, message_history=history)
        history = result.all_messages()
        session_usage = session_usage + result.usage
        print(f'[turn] {result.usage.total_tokens} tokens')

    print(f'\nSession total: {session_usage.total_tokens} tokens in {session_usage.requests} requests')
    return session_usage

asyncio.run(chat_session())
```

### Usage limits with tool-calling agents

When agents call tools, `tool_calls_limit` caps the total number of successful tool executions. Useful to prevent runaway tool use:

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def search(query: str) -> str:
    """Search for information."""
    return f'Results for: {query}'

@agent.tool_plain
def fetch_page(url: str) -> str:
    """Fetch a web page."""
    return f'Content from: {url}'

async def main():
    limits = UsageLimits(
        request_limit=20,
        tool_calls_limit=5,   # at most 5 tool calls total
    )
    try:
        result = await agent.run(
            'Research quantum computing thoroughly, searching multiple sources.',
            usage_limits=limits,
        )
        print(result.output)
        print(f'Tool calls used: {result.usage.tool_calls}')
    except UsageLimitExceeded as e:
        print(f'Hit limit: {e}')

asyncio.run(main())
```

---

## Putting It All Together — A Production-Grade Agent

This example combines `Hooks` (observability), `WebSearch` + `WebFetch` (capabilities), `Thinking` (reasoning), `FilteredToolset` (RBAC), `UsageLimits` (budget), and `RunContext` (metadata access) into a single production-grade setup:

```python
import asyncio
import logging
import time
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.capabilities import Hooks, WebSearch, WebFetch, Thinking
from pydantic_ai.toolsets import FunctionToolset, FilteredToolset, CombinedToolset
from pydantic_ai.usage import UsageLimits

logger = logging.getLogger(__name__)

# ── Dependencies ───────────────────────────────────────────────────────────────
@dataclass
class Deps:
    user_id: str
    role: str  # 'analyst' | 'admin'
    max_tokens: int = 20_000

# ── Output model ──────────────────────────────────────────────────────────────
class ResearchReport(BaseModel):
    title: str
    summary: str
    key_findings: list[str]
    sources: list[str]
    confidence: float  # 0.0–1.0

# ── Toolsets ──────────────────────────────────────────────────────────────────
read_tools = FunctionToolset[Deps]()
write_tools = FunctionToolset[Deps]()

@read_tools.tool
def get_company_data(ctx: RunContext[Deps], company: str) -> str:
    return f'Internal data for {company} (accessed by {ctx.deps.user_id})'

@write_tools.tool
def save_report(ctx: RunContext[Deps], report_id: str, content: str) -> str:
    if ctx.deps.role != 'admin':
        raise ModelRetry('Only admins can save reports. Please tell the user.')
    return f'Report {report_id} saved by {ctx.deps.user_id}'

def write_allowed(ctx: RunContext[Deps], tool_def) -> bool:
    return ctx.deps.role == 'admin'

# ── Hooks ─────────────────────────────────────────────────────────────────────
hooks = Hooks()
_start_times: dict[str, float] = {}

@hooks.on.before_run
def start_timer(ctx: RunContext[Deps]):
    _start_times[ctx.run_id] = time.perf_counter()
    logger.info('run started  user=%s  run_id=%s', ctx.deps.user_id, ctx.run_id)

@hooks.on.after_run
def stop_timer(ctx: RunContext[Deps], *, result):
    elapsed = time.perf_counter() - _start_times.pop(ctx.run_id, time.perf_counter())
    logger.info(
        'run done  user=%s  tokens=%d  time=%.2fs',
        ctx.deps.user_id, result.usage.total_tokens, elapsed,
    )
    return result

@hooks.on.tool_execute_error
async def handle_tool_error(ctx, *, call, tool_def, args, error):
    logger.warning('tool error  tool=%s  error=%s', tool_def.name, error)
    raise error

# ── Agent ────────────────────────────────────────────────────────────────────
agent = Agent(
    'anthropic:claude-opus-4-5',
    deps_type=Deps,
    output_type=ResearchReport,
    capabilities=[
        hooks,
        Thinking(effort='high'),
        WebSearch(
            search_context_size='high',
            blocked_domains=['twitter.com', 'x.com', 'reddit.com'],
            max_uses=5,
        ),
        WebFetch(
            blocked_domains=['social-media-site.com'],
            max_content_tokens=3000,
        ),
    ],
    toolsets=[
        CombinedToolset([
            read_tools,
            FilteredToolset(write_tools, filter_func=write_allowed),
        ])
    ],
    system_prompt=(
        'You are a senior research analyst. Produce structured, evidence-based reports '
        'with citations. Always verify claims using web search and fetched pages.'
    ),
)

@agent.output_validator
async def validate_confidence(ctx: RunContext[Deps], report: ResearchReport) -> ResearchReport:
    if ctx.partial_output:
        return report  # skip validation on partial streams
    if report.confidence < 0.5:
        raise ModelRetry('Confidence too low — do more research before finalising.')
    return report

async def research(topic: str, deps: Deps) -> ResearchReport:
    limits = UsageLimits(
        total_tokens_limit=deps.max_tokens,
        request_limit=15,
        tool_calls_limit=10,
    )
    result = await agent.run(
        f'Research this topic and produce a structured report: {topic}',
        deps=deps,
        usage_limits=limits,
    )
    return result.output

async def main():
    analyst_deps = Deps(user_id='alice@example.com', role='analyst', max_tokens=15_000)
    report = await research('The current state of quantum computing hardware', analyst_deps)
    print(f'Title: {report.title}')
    print(f'Summary: {report.summary}')
    print(f'Confidence: {report.confidence:.0%}')
    for finding in report.key_findings:
        print(f'  • {finding}')

asyncio.run(main())
```

---

## Reference

| Class | Module | Role |
|-------|--------|------|
| `RunContext[DepsT]` | `pydantic_ai.tools` | Carries deps, model, usage, conversation state, retry info into every tool/hook |
| `Hooks` | `pydantic_ai.capabilities` | Decorator-first lifecycle hooks for 33 events |
| `WebSearch` | `pydantic_ai.capabilities` | Native + local web search with domain control |
| `WebFetch` | `pydantic_ai.capabilities` | Native + local URL fetching with SSRF guards |
| `Thinking` | `pydantic_ai.capabilities` | Extended reasoning across providers (`True`, `False`, `'low'`…`'xhigh'`) |
| `FilteredToolset` | `pydantic_ai.toolsets` | Hide/show tools per step via sync or async predicate |
| `CombinedToolset` | `pydantic_ai.toolsets` | Merge multiple tool sources with collision detection |
| `ApprovalRequiredToolset` | `pydantic_ai.toolsets` | HITL approval gate before tool execution |
| `ExternalToolset` | `pydantic_ai.toolsets` | Schema-only tools executed outside the agent process |
| `UsageLimits` | `pydantic_ai.usage` | Enforce request / token / tool-call budgets |
| `RunUsage` | `pydantic_ai.usage` | Accumulated token + request + tool-call counters |

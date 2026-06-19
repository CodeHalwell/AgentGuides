---
title: "PydanticAI Class Deep Dives Vol. 20"
description: "Source-verified deep dives into 10 pydantic-ai 1.107.0 classes: Hooks capability, Instrumentation, ConcurrencyLimiter, toolset composition (CombinedToolset / PrefixedToolset / RenamedToolset / FilteredToolset / PreparedToolset), ApprovalRequiredToolset + ExternalToolset, TemplateStr, FunctionSchema, safe_download SSRF protection, deferred capability loading primitives (LoadCapabilityCallPart / LoadCapabilityReturnPart), and ModelProfile"
sidebar:
  order: 46
---

import { Aside } from '@astrojs/starlight/components';

# PydanticAI Class Deep Dives Vol. 20

Source-verified against **pydantic-ai 1.107.0** installed at `/usr/local/lib/python3.11/dist-packages/pydantic_ai/`.

Each section covers one class group with key behaviours extracted directly from source, a quick-reference table, and three standalone runnable examples.

---

## 1 · `Hooks` + `HookTimeoutError` — decorator-based capability middleware

**Module**: `pydantic_ai.capabilities.hooks`  
**Exported as**: `pydantic_ai.capabilities.Hooks`

`Hooks` is a concrete `AbstractCapability` subclass that lets you register lifecycle hook functions via a decorator namespace (`hooks.on.<hook_name>`) instead of subclassing `AbstractCapability` directly. Every hook the abstract capability system exposes is available as a decorator on `hooks.on`.

### Key behaviours (source-verified)

| Detail | Value |
|---|---|
| Registration namespace | `hooks.on` — a `_HookRegistration` proxy object |
| Decorator forms | `@hooks.on.before_model_request`, `@hooks.on.tool_execute`, etc. |
| `tools=` filter | Pass `tools=['tool_a']` to a hook decorator to restrict it to specific tool names |
| `timeout=` per hook | Pass `timeout=5.0` (seconds) to any hook; `HookTimeoutError` is raised on expiry |
| Sync/async parity | Both sync and async hook functions are accepted; sync ones are auto-wrapped |
| `HookTimeoutError` base | Inherits `TimeoutError` — catch with `except TimeoutError` |
| `get_ordering()` | Returns `CapabilityOrdering(position='middle')` (default middleware position) |

### Available hooks

`before_run` · `after_run` · `run` (wraps full run) · `run_error`  
`before_model_request` · `after_model_request` · `model_request` (wraps model request) · `model_request_error`  
`prepare_tools` · `prepare_output_tools`  
`before_tool_validate` · `after_tool_validate` · `tool_validate` · `tool_validate_error`  
`before_tool_execute` · `after_tool_execute` · `tool_execute` · `tool_execute_error`  
`before_output_validate` · `after_output_validate` · `output_validate` · `output_validate_error`  
`before_output_process` · `after_output_process` · `output_process` · `output_process_error`  
`deferred_tool_calls`

### Example 1 — logging every model request

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx, request_context):
    print(f"[before_model_request] run_step={ctx.run_step}")
    return request_context  # must return the (potentially modified) request_context

@hooks.on.after_model_request
async def log_response(ctx, response, request_context):
    print(f"[after_model_request] parts={len(response.parts)}")

agent = Agent('openai:gpt-4o', capabilities=[hooks])

async def main():
    result = await agent.run('Say hello', api_key='test')
    print(result.output)

# asyncio.run(main())
```

### Example 2 — tool-specific hook with timeout

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.hooks import Hooks, HookTimeoutError

hooks = Hooks()

@hooks.on.before_tool_execute(tools=['search'], timeout=2.0)
async def audit_search(ctx: RunContext, tool_def, validated_args):
    """Runs only for the 'search' tool; raises HookTimeoutError after 2 s."""
    import asyncio
    await asyncio.sleep(0)  # simulate a quick async check
    print(f"Auditing search call with args: {validated_args}")
    return validated_args  # must return validated_args

@hooks.on.after_tool_execute(tools=['search'])
async def log_search_result(ctx, result, tool_def, validated_args):
    print(f"Search returned: {result!r}")

async def search(ctx: RunContext, query: str) -> str:
    return f"Results for: {query}"

agent = Agent('openai:gpt-4o', tools=[search], capabilities=[hooks])
```

### Example 3 — wrapping a full run for timing

```python
import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pydantic_ai import Agent
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()

@hooks.on.run
@asynccontextmanager
async def time_run(ctx, handler):
    """'run' hook is a context manager that wraps the full agent run."""
    start = time.perf_counter()
    try:
        result = await handler(ctx)
        elapsed = time.perf_counter() - start
        print(f"Run completed in {elapsed:.3f}s, steps={ctx.run_step}")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"Run failed after {elapsed:.3f}s: {exc}")
        raise

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

---

## 2 · `Instrumentation` — OpenTelemetry/Logfire capability

**Module**: `pydantic_ai.capabilities.instrumentation`  
**Exported as**: `pydantic_ai.capabilities.Instrumentation`

`Instrumentation` is a concrete `AbstractCapability` that creates OpenTelemetry spans for the agent run, each model request, each tool execution, and each output-processing step. It is the built-in replacement for passing `instrument=True` / `instrument=logfire` to `Agent()`.

### Key behaviours (source-verified)

| Detail | Value |
|---|---|
| `get_ordering()` | Returns `CapabilityOrdering(position='outermost')` — always outermost capability |
| Default settings | `InstrumentationSettings()` — uses the global `TracerProvider` |
| `_variable_instructions` flag | Set to `True` when agent-level instructions differ across requests in one run |
| Per-run isolation | `for_run()` calls `dataclasses.replace(self)` — fresh state per run |
| Distinguishes errors | `ToolRetryError` → span status OK (expected); `ApprovalRequired`/`CallDeferred` → OK too |
| `InstrumentationSettings.version` | Controls OTel GenAI spec version (1–5); default version resolved via `InstrumentationNames.for_version()` |
| `from_spec(**kwargs)` | Class method for building from serialisable kwargs (YAML/JSON config) |

### Example 1 — basic Logfire tracing

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation

logfire.configure()  # sets up TracerProvider

agent = Agent(
    'openai:gpt-4o',
    capabilities=[Instrumentation()],
)

# Every agent.run() will now emit spans:
#   "agent run" (outermost)
#   ├── "model request" (per LLM call)
#   └── "tool execute: my_tool" (per tool call)
```

### Example 2 — custom settings (content redaction)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

# Disable content capture for PII-sensitive agents
settings = InstrumentationSettings(
    include_content=False,          # redacts prompt/response text from spans
    event_mode='logs',              # emit OTel log events instead of span events
    use_aggregated_usage_attribute_names=True,
)

agent = Agent('openai:gpt-4o', capabilities=[Instrumentation(settings=settings)])
```

### Example 3 — adding custom attributes inside a hook

```python
from opentelemetry.trace import get_current_span
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.capabilities.hooks import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def tag_tenant(ctx, request_context):
    # Instrumentation runs outermost, so its spans are already open here
    span = get_current_span()
    span.set_attribute('app.tenant_id', getattr(ctx.deps, 'tenant_id', 'unknown'))
    return request_context

agent = Agent(
    'openai:gpt-4o',
    # Order matters: Instrumentation first so its spans are outermost
    capabilities=[Instrumentation(), hooks],
)
```

---

## 3 · `ConcurrencyLimiter` + `AbstractConcurrencyLimiter` + `ConcurrencyLimit`

**Module**: `pydantic_ai.concurrency`

These three exports provide a layered concurrency-limiting API for controlling how many simultaneous model requests an agent issues.

### Key behaviours (source-verified)

| Symbol | Role |
|---|---|
| `AnyConcurrencyLimit` | Type alias: `int \| ConcurrencyLimit \| AbstractConcurrencyLimiter \| None` |
| `ConcurrencyLimit(max_running, max_queued=None)` | Config dataclass; `max_queued=None` means unlimited queue |
| `ConcurrencyLimiter(max_running, *, max_queued=None, name=None, tracer=None)` | Concrete limiter; wraps `anyio.CapacityLimiter` |
| `ConcurrencyLimiter.from_limit(limit)` | Class method: accepts `int` or `ConcurrencyLimit` |
| `acquire(source)` | Fast-path `acquire_nowait()` first; only creates OTel span when it must wait |
| `_waiting_count` | Atomic counter guarded by `anyio.Lock()` to prevent race on `max_queued` check |
| `ConcurrencyLimitExceeded` | Raised when `waiting_count >= max_queued` at acquire time |
| `normalize_to_limiter(limit, *, name)` | Module-level helper; returns `None` if `limit is None` |
| `get_concurrency_context(limiter, source)` | Returns a no-op ctx if `limiter is None`; otherwise `_limiter_context` |

### Example 1 — limit concurrent model calls on an agent

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter

# At most 3 simultaneous model requests; up to 10 queued
limiter = ConcurrencyLimiter(max_running=3, max_queued=10, name='my-agent')

agent = Agent('openai:gpt-4o')

async def run_batch(prompts: list[str]):
    async def _run(prompt: str):
        async with limiter._limiter:  # low-level usage; agent integration via model settings
            return await agent.run(prompt)

    results = await asyncio.gather(*(_run(p) for p in prompts))
    return results
```

### Example 2 — per-model concurrency via `ModelSettings`

```python
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimit
from pydantic_ai.settings import ModelSettings

# Pass via ModelSettings — agent uses this when constructing the model
agent = Agent(
    'openai:gpt-4o',
    model_settings=ModelSettings(concurrency_limit=ConcurrencyLimit(max_running=5)),
)
```

### Example 3 — custom Redis-backed distributed limiter

```python
from pydantic_ai.concurrency import AbstractConcurrencyLimiter


class RedisConcurrencyLimiter(AbstractConcurrencyLimiter):
    """Distributed limiter backed by Redis INCR/DECR."""

    def __init__(self, redis_client, key: str, max_running: int):
        self._redis = redis_client
        self._key = key
        self._max = max_running

    async def acquire(self, source: str) -> None:
        while True:
            count = await self._redis.incr(self._key)
            if count <= self._max:
                return
            await self._redis.decr(self._key)
            import asyncio
            await asyncio.sleep(0.1)

    def release(self) -> None:
        import asyncio
        asyncio.get_event_loop().create_task(self._redis.decr(self._key))
```

---

## 4 · Toolset composition — `CombinedToolset` · `PrefixedToolset` · `RenamedToolset` · `FilteredToolset` · `PreparedToolset`

**Module**: `pydantic_ai.toolsets`

These five wrapper classes let you compose and modify existing toolsets without subclassing, all following the `AbstractToolset` interface.

### Key behaviours (source-verified)

| Class | Constructor | What it does |
|---|---|---|
| `CombinedToolset(toolsets)` | `Sequence[AbstractToolset]` | Merges tools from multiple toolsets; raises `UserError` on name conflict |
| `PrefixedToolset(wrapped, prefix)` | `AbstractToolset, str` | Renames every tool to `{prefix}_{original_name}` |
| `RenamedToolset(wrapped, name_map)` | `AbstractToolset, dict[str, str]` | Renames specific tools; `name_map = {new_name: original_name}` |
| `FilteredToolset(wrapped, filter_func)` | `AbstractToolset, Callable[[RunContext, ToolDefinition], bool \| Awaitable[bool]]` | Removes tools where `filter_func` returns `False`; async filter supported |
| `PreparedToolset(wrapped, prepare_func)` | `AbstractToolset, ToolsPrepareFunc` | Mutates tool definitions (description, parameters) via a prepare function |

#### `CombinedToolset` conflict detection

When two toolsets expose a tool with the same name, `get_tools()` raises:

```
UserError: <ToolsetLabel> defines a tool whose name conflicts with existing tool from <Other>: 'tool_name'. <tool_name_conflict_hint>
```

#### `PreparedToolset` constraint

The `prepare_func` **cannot add or rename tools** — only modify existing `ToolDefinition` fields (e.g. `description`, `parameters_json_schema`, `max_retries`). Adding/renaming raises `UserError`.

### Example 1 — combine two toolsets, avoid name collision via prefix

```python
from pydantic_ai.toolsets import FunctionToolset, CombinedToolset, PrefixedToolset

db_toolset = FunctionToolset()
web_toolset = FunctionToolset()

@db_toolset.tool
async def search(query: str) -> str:
    return f"DB: {query}"

@web_toolset.tool
async def search(query: str) -> str:  # would conflict in CombinedToolset without prefix
    return f"Web: {query}"

agent_toolset = CombinedToolset([
    PrefixedToolset(db_toolset, prefix='db'),   # → db_search
    PrefixedToolset(web_toolset, prefix='web'), # → web_search
])
```

### Example 2 — role-based tool filtering

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, FilteredToolset

all_tools = FunctionToolset()

@all_tools.tool
async def delete_record(ctx: RunContext, id: int) -> str:
    return f"Deleted {id}"

@all_tools.tool
async def read_record(ctx: RunContext, id: int) -> str:
    return f"Read {id}"

def only_read_tools(ctx: RunContext, tool_def) -> bool:
    return not tool_def.name.startswith('delete_')

read_only_toolset = FilteredToolset(all_tools, filter_func=only_read_tools)

agent = Agent('openai:gpt-4o', toolsets=[read_only_toolset])
```

### Example 3 — inject richer descriptions at runtime

```python
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset, PreparedToolset
from pydantic_ai.tools import ToolDefinition

schema_toolset = FunctionToolset()

@schema_toolset.tool
async def list_tables(ctx: RunContext) -> list[str]:
    return ['users', 'orders']

def enrich_descriptions(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    enriched = []
    for td in tool_defs:
        enriched.append(ToolDefinition(
            name=td.name,
            description=f"{td.description} [DB: {getattr(ctx.deps, 'db_name', 'unknown')}]",
            parameters_json_schema=td.parameters_json_schema,
        ))
    return enriched

agent_toolset = PreparedToolset(schema_toolset, prepare_func=enrich_descriptions)
```

---

## 5 · `ApprovalRequiredToolset` + `ExternalToolset`

**Module**: `pydantic_ai.toolsets`

Two toolset wrappers for human-in-the-loop and external-handler patterns.

### `ApprovalRequiredToolset`

Wraps any toolset and intercepts each `call_tool` to check whether the call needs human approval. If approval is required and `ctx.tool_call_approved` is not set, it raises `ApprovalRequired`.

```python
# Source signature (approval_required.py line 22-23)
approval_required_func: Callable[[RunContext, ToolDefinition, dict[str, Any]], bool] = (
    lambda ctx, tool_def, tool_args: True
)
```

**Default**: every call requires approval (`lambda ... True`).

### `ExternalToolset`

Holds tool definitions whose results are produced *outside* the agent run — e.g. by an external system that feeds results back via `run_context.tool_results`. `call_tool()` raises `NotImplementedError`; the `tool_kind='external'` marker tells the model the tools are provided but their execution is handled elsewhere.

`DeferredToolset` is a deprecated alias for `ExternalToolset`.

```python
# Source (external.py line 13-14)
TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())
# All args pass validation — type checking is deferred to the external handler
```

### Key behaviours (source-verified)

| Detail | Value |
|---|---|
| `ApprovalRequiredToolset` default | All calls require approval |
| Approval check order | `not ctx.tool_call_approved` **AND** `approval_required_func(ctx, tool_def, tool_args)` |
| `ExternalToolset.call_tool()` | Raises `NotImplementedError('External tools cannot be called directly')` |
| `ExternalToolset` `tool_kind` | Forces `kind='external'` on every `ToolDefinition` via `dataclasses.replace` |
| `ExternalToolset` args validation | `TOOL_SCHEMA_VALIDATOR = SchemaValidator(core_schema.any_schema())` — passes everything |
| `DeferredToolset` | `@deprecated` alias for `ExternalToolset` |

### Example 1 — require approval only for destructive tools

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, ApprovalRequiredToolset
from pydantic_ai.tools import ToolDefinition

dangerous_tools = FunctionToolset()

@dangerous_tools.tool
async def delete_file(ctx: RunContext, path: str) -> str:
    import os
    os.remove(path)
    return f"Deleted {path}"

@dangerous_tools.tool
async def list_files(ctx: RunContext) -> list[str]:
    import os
    return os.listdir('.')

def needs_approval(ctx: RunContext, tool_def: ToolDefinition, args: dict) -> bool:
    return tool_def.name.startswith('delete_')

safe_toolset = ApprovalRequiredToolset(dangerous_tools, approval_required_func=needs_approval)
agent = Agent('openai:gpt-4o', toolsets=[safe_toolset])
```

### Example 2 — external tool results from a webhook handler

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition

# Declare tools the LLM can call, but results come from an external service
payment_tool = ToolDefinition(
    name='charge_card',
    description='Charges a credit card. Returns transaction ID.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'amount': {'type': 'number'},
            'card_token': {'type': 'string'},
        },
        'required': ['amount', 'card_token'],
    },
)

external = ExternalToolset(tool_defs=[payment_tool], id='payment-service')
agent = Agent('openai:gpt-4o', toolsets=[external])
# Results injected later via agent.run(..., tool_results=[...])
```

### Example 3 — combining approval and external toolsets

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import (
    FunctionToolset, ApprovalRequiredToolset,
    ExternalToolset, CombinedToolset
)
from pydantic_ai.tools import ToolDefinition

local_tools = FunctionToolset()

@local_tools.tool
async def summarise(text: str) -> str:
    return text[:200]

external_tools = ExternalToolset([
    ToolDefinition(
        name='send_email',
        description='Send an email via external mailer service.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'to': {'type': 'string'}, 'body': {'type': 'string'}},
            'required': ['to', 'body'],
        },
    )
])

# All tools require approval
combined = CombinedToolset([
    ApprovalRequiredToolset(local_tools),
    ApprovalRequiredToolset(external_tools),
])

agent = Agent('openai:gpt-4o', toolsets=[combined])
```

---

## 6 · `TemplateStr` — Handlebars template instructions

**Module**: `pydantic_ai._template`  
**Exported as**: `pydantic_ai.TemplateStr`

`TemplateStr` is a `Generic[AgentDepsT]` string subclass that compiles a [Handlebars](https://handlebarsjs.com/) template at construction time and renders it against `RunContext.deps` when invoked. Used directly as the `instructions=` value on `Agent`.

### Key behaviours (source-verified)

| Detail | Value |
|---|---|
| Template trigger | String must contain `{{` to be treated as a template; plain strings fall through in `Union[TemplateStr, str]` |
| `deps_type` known | Compiles with `hbs.compile(source, deps_type)` — type-checked Handlebars |
| `deps_type` unknown | Compiles with `hbs.compile(source)` then uses `TypeAdapter(type(deps)).dump_python(deps)` at render time |
| `__call__(ctx)` | Calls `self.render(ctx.deps)` — satisfies the `InstructionCallable` protocol |
| Pydantic validation | `'{{' not in value` → `ValueError('Not a template string')` — used to fall through to `str` in Union types |
| `deps_schema` keyword | Checked at construction via `hbs.check_template_compatibility(source, schema)` without full compile |
| Serialisation | `TemplateStr.__get_pydantic_core_schema__` serialises back to the raw `_source` string |

### Example 1 — parameterised system prompt from deps

```python
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr


@dataclass
class UserContext:
    username: str
    role: str
    allowed_tools: list[str]


agent = Agent(
    'openai:gpt-4o',
    deps_type=UserContext,
    instructions=TemplateStr(
        'You are helping {{username}} who has role {{role}}. '
        'Available tools: {{#each allowed_tools}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}.'
    ),
)

# When agent.run(prompt, deps=UserContext('Alice', 'admin', ['search', 'delete'])) is called,
# the system prompt becomes:
# "You are helping Alice who has role admin. Available tools: search, delete."
```

### Example 2 — standalone rendering outside an agent

```python
from dataclasses import dataclass
from pydantic_ai import TemplateStr


@dataclass
class ReportDeps:
    report_date: str
    author: str


template = TemplateStr(
    'Report for {{report_date}} by {{author}}.',
    deps_type=ReportDeps,
)

rendered = template.render(ReportDeps(report_date='2026-06-19', author='Alice'))
print(rendered)  # "Report for 2026-06-19 by Alice."
```

### Example 3 — Union[TemplateStr, str] in a Pydantic model

```python
from typing import Union
from pydantic import BaseModel
from pydantic_ai import TemplateStr


class AgentSpec(BaseModel):
    instructions: Union[TemplateStr, str]


# A plain string falls through to str branch (no {{ found)
spec_plain = AgentSpec(instructions='Hello world')
assert isinstance(spec_plain.instructions, str)

# A template string is compiled into TemplateStr
spec_template = AgentSpec(instructions='Hello {{name}}')
assert isinstance(spec_template.instructions, TemplateStr)
```

---

## 7 · `FunctionSchema` — Python function → LLM tool parameter schema

**Module**: `pydantic_ai._function_schema`

`FunctionSchema` is a frozen dataclass that captures the Pydantic schema, async/sync status, and calling convention for a Python function that will be exposed as an LLM tool. Created by the `function_schema()` factory function.

### Key behaviours (source-verified)

```python
@dataclass
class FunctionSchema:
    function: Callable[..., Any]
    description: str | None
    validator: SchemaValidator           # Pydantic core validator
    json_schema: ObjectJsonSchema        # Tool parameter JSON schema
    single_arg_name: str | None = None  # set for model-like or primitive single args
    positional_fields: list[str]
    var_positional_field: str | None
    takes_ctx: bool                      # first arg is RunContext?
    is_async: bool
    return_schema: ObjectJsonSchema      # JSON schema of return type (may be {})
```

| Detail | Value |
|---|---|
| `single_field_name` property | Returns `single_arg_name` if set, else the sole property name from `json_schema` |
| Async execution | If `is_async`, called directly; if sync, wrapped with `run_in_executor` |
| `takes_ctx` detection | Auto-detected: first param is `RunContext` if `takes_ctx=None`; override with `takes_ctx=True/False` |
| `docstring_format` | `'auto'` (default), `'google'`, `'numpy'`, `'restructuredtext'` |
| `require_parameter_descriptions` | If `True`, raises `UserError` for any parameter missing a docstring description |
| Partial function support | `functools.partial` is unwrapped; original function preserved in `.function` |
| Return schema | Extracted from return annotation via `TypeAdapter(return_annotation).json_schema()` |

### Example 1 — inspect a tool's generated schema

```python
from pydantic_ai._function_schema import function_schema
from pydantic_ai._json_schema import GenerateJsonSchema


def get_weather(city: str, unit: str = 'celsius') -> dict:
    """Get current weather for a city.

    Args:
        city: The city to get weather for.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'.
    """
    return {'city': city, 'temp': 22, 'unit': unit}


schema = function_schema(
    get_weather,
    schema_generator=GenerateJsonSchema,
    docstring_format='google',
)

print(schema.description)    # "Get current weather for a city."
print(schema.json_schema)    # {'type': 'object', 'properties': {'city': ..., 'unit': ...}, ...}
print(schema.takes_ctx)      # False
print(schema.is_async)       # False
```

### Example 2 — context-taking async tool schema

```python
from pydantic_ai import RunContext
from pydantic_ai._function_schema import function_schema
from pydantic_ai._json_schema import GenerateJsonSchema


async def search_db(ctx: RunContext, query: str, limit: int = 10) -> list[str]:
    """Search the database.

    Args:
        query: Search query string.
        limit: Maximum number of results.
    """
    ...


schema = function_schema(search_db, schema_generator=GenerateJsonSchema)
print(schema.takes_ctx)   # True — RunContext is detected and excluded from json_schema
print(schema.is_async)    # True
print(list(schema.json_schema['properties'].keys()))  # ['query', 'limit']
```

### Example 3 — single-argument model-like tool

```python
from pydantic import BaseModel
from pydantic_ai._function_schema import function_schema
from pydantic_ai._json_schema import GenerateJsonSchema


class SearchQuery(BaseModel):
    """A structured search query."""
    terms: list[str]
    filters: dict[str, str] = {}


def run_search(query: SearchQuery) -> list[str]:
    """Execute a structured search."""
    return [f"result for {t}" for t in query.terms]


schema = function_schema(run_search, schema_generator=GenerateJsonSchema)
# single_arg_name is set because the only parameter is a model-like type
print(schema.single_arg_name)   # 'query'
print(schema.single_field_name) # 'query'
```

---

## 8 · `safe_download` + `ResolvedUrl` — SSRF-protected URL download

**Module**: `pydantic_ai._ssrf`  
**Public export**: `safe_download` only

`safe_download` is an async function that downloads content from a URL with comprehensive SSRF (Server-Side Request Forgery) protection. Used internally by the `web_fetch` built-in tool.

### Key behaviours (source-verified)

| Detail | Value |
|---|---|
| Allowed protocols | `http` and `https` only — all others raise `ValueError` |
| DNS resolution | `socket.getaddrinfo` in thread executor; returns `list[str]` of IPs |
| Cloud metadata IPs | **Always blocked** even with `allow_local=True` |
| Private IP ranges | 14 IPv4 + 7 IPv6 CIDR blocks; blocked unless `allow_local=True` |
| Max redirects | Default 10 (`_MAX_REDIRECTS = 10`) |
| Default timeout | 30 seconds (`_DEFAULT_TIMEOUT = 30`) |
| Redirect SSRF | Each redirect hop is re-validated — no DNS rebinding bypass |
| Sensitive headers stripped | `authorization`, `cookie`, `proxy-authorization` removed on cross-origin redirect |
| Trailing dot removal | `hostname.rstrip('.')` — prevents FQDN bypass of domain lists |
| IPv6 transition decoding | NAT64 (RFC 6052), 6to4 (RFC 3056), ISATAP (RFC 5214), Teredo (RFC 4380) all decoded |

#### Cloud metadata IPs always blocked

```python
# From _ssrf.py lines 96-106
_CLOUD_METADATA_IPV4 = frozenset({
    '169.254.169.254',  # AWS IMDS, GCP, Azure, OCI, DigitalOcean, Hetzner, IBM, OpenStack
    '169.254.170.2',    # AWS ECS task IAM role credentials
    '169.254.170.23',   # AWS EKS Pod Identity Agent
    '168.63.129.16',    # Azure WireServer (public IP — metadata guard is the only block)
    '100.100.100.200',  # Alibaba Cloud
    '192.0.0.192',      # Oracle Cloud (Classic)
    '169.254.42.42',    # Scaleway
})
```

### Example 1 — basic SSRF-safe fetch

```python
import asyncio
from pydantic_ai._ssrf import safe_download


async def fetch_content(url: str) -> str:
    """Download URL content with SSRF protection."""
    try:
        response = await safe_download(url)
        return response.text
    except ValueError as e:
        # Catches: protocol errors, private IP, cloud metadata, too many redirects
        print(f"SSRF validation failed: {e}")
        return ""


async def main():
    content = await fetch_content('https://example.com')
    print(content[:200])

# asyncio.run(main())
```

### Example 2 — domain allowlist + timeout

```python
import asyncio
from pydantic_ai._ssrf import safe_download


async def fetch_from_allowlist(url: str) -> bytes:
    """Only fetch from approved domains, with a short timeout."""
    ALLOWED = ['api.example.com', 'data.example.com']
    response = await safe_download(
        url,
        timeout=10,
        allowed_domains=ALLOWED,
        max_redirects=3,
    )
    return response.content


# ValueError raised for any domain not in ALLOWED, including via redirect
```

### Example 3 — helper functions for custom validation

```python
from pydantic_ai._ssrf import is_cloud_metadata_ip, is_private_ip, validate_url_protocol


def validate_webhook_url(url: str) -> tuple[bool, str]:
    """Custom webhook URL validator using safe_download primitives."""
    try:
        scheme, is_https = validate_url_protocol(url)
    except ValueError as e:
        return False, str(e)

    if not is_https:
        return False, "Webhooks must use HTTPS"

    import ipaddress
    from urllib.parse import urlparse
    hostname = urlparse(url).hostname or ''
    try:
        ip = str(ipaddress.ip_address(hostname))
        if is_cloud_metadata_ip(ip):
            return False, "Cloud metadata endpoint blocked"
        if is_private_ip(ip):
            return False, "Private IP blocked"
    except ValueError:
        pass  # hostname, not IP — DNS resolution deferred to safe_download

    return True, "OK"
```

---

## 9 · `LoadCapabilityCallPart` + `LoadCapabilityReturnPart` + `parse_loaded_capabilities`

**Module**: `pydantic_ai._deferred_capabilities`

These typed message-part subclasses and the `parse_loaded_capabilities` helper form the wire protocol for deferred (lazy-loaded) capabilities. When a capability has `defer_loading=True`, the agent first calls the `load_capability` built-in tool before the capability's tools become available in the context.

### Key behaviours (source-verified)

| Symbol | Role |
|---|---|
| `DEFERRED_CAPABILITY_TOOL_METADATA_KEY` | `'pydantic_ai_deferred_capability_tool'` — metadata key marking deferred function tools |
| `LoadCapabilityArgs` | `TypedDict` with field `id: str` (the capability ID to load) |
| `LoadCapabilityReturn` | `TypedDict` with optional field `instructions: str` (loaded capability instructions) |
| `LoadCapabilityCallPart` | Subclass of `ToolCallPart`; `tool_name='load_capability'`, `tool_kind='capability-load'` |
| `LoadCapabilityReturnPart` | Subclass of `ToolReturnPart`; `content: LoadCapabilityReturn`, `tool_kind='capability-load'` |
| `tool_kind='capability-load'` discriminator | Prevents user tools named `load_capability` from being promoted to the typed subclass |
| `capability_id` property | `LoadCapabilityCallPart.capability_id` → `str \| None` from parsed args |
| `instructions` property | `LoadCapabilityReturnPart.instructions` → `str \| None` from `content.get('instructions')` |
| `parse_loaded_capabilities(messages)` | Iterates messages to collect IDs of successfully loaded capabilities |

#### Wire protocol tags

```python
# From _deferred_capabilities.py lines 134-138
_TYPED_PART_TAGS[('tool-call', 'capability-load')]   = 'capability-load-call'
_TYPED_PART_TAGS[('tool-return', 'capability-load')] = 'capability-load-return'
```

### Example 1 — inspect deferred capability loading in message history

```python
from pydantic_ai._deferred_capabilities import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    parse_loaded_capabilities,
)
from pydantic_ai.messages import ModelRequest, ModelResponse


def describe_capability_loads(messages):
    """Print a summary of capability loading activity in a message history."""
    for i, msg in enumerate(messages):
        for part in msg.parts:
            if isinstance(part, LoadCapabilityCallPart):
                print(f"[msg {i}] Loading capability: {part.capability_id!r}")
            elif isinstance(part, LoadCapabilityReturnPart):
                instr = part.instructions
                print(f"[msg {i}] Loaded — instructions: {instr[:60]!r}" if instr else f"[msg {i}] Loaded (no instructions)")

    loaded_ids = parse_loaded_capabilities(messages)
    print(f"Total capabilities loaded: {loaded_ids}")
```

### Example 2 — check which capabilities were loaded before continuing

```python
from pydantic_ai import Agent
from pydantic_ai._deferred_capabilities import parse_loaded_capabilities


async def run_with_capability_check(agent: Agent, prompt: str) -> str:
    """Run an agent and report which capabilities were lazily loaded."""
    result = await agent.run(prompt)

    loaded = parse_loaded_capabilities(result.all_messages())
    if loaded:
        print(f"Capabilities loaded during run: {loaded}")
    else:
        print("No deferred capabilities were loaded.")

    return result.output
```

### Example 3 — build a deferred capability (lazy native tool)

```python
from pydantic_ai.capabilities.capability import Capability
from pydantic_ai.toolsets import FunctionToolset


# A capability marked defer_loading=True registers its tool only after
# the model explicitly calls load_capability('heavy-nlp').
heavy_toolset = FunctionToolset(
    id='heavy-nlp',
    defer_loading=True,
)


@heavy_toolset.tool
async def nlp_analyse(text: str) -> dict:
    """Analyse text using heavy NLP models (loaded on demand)."""
    # Imagine importing spaCy, loading a large model here
    return {'sentiment': 'positive', 'entities': []}


# The agent will only include nlp_analyse in the context after the model
# calls load_capability({'id': 'heavy-nlp'}).
```

---

## 10 · `ModelProfile` + `DEFAULT_PROFILE` + `ModelProfileSpec`

**Module**: `pydantic_ai.profiles`  
**Exported as**: `pydantic_ai.profiles.ModelProfile`

`ModelProfile` is a `@dataclass(kw_only=True)` that declares the capabilities of a model family, independent of the provider. Every model type holds a `ModelProfile` (or a provider-specific subclass) that controls structured output mode, tool support, thinking tags, and native tool availability.

### Key fields (source-verified)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `supports_tools` | `bool` | `True` | Whether the model supports tool calls |
| `supports_tool_return_schema` | `bool` | `False` | Native structured return schemas; if `False`, schema injected into description as JSON text |
| `supports_json_schema_output` | `bool` | `False` | Native JSON schema structured output (`NativeOutput`) |
| `supports_json_object_output` | `bool` | `False` | JSON mode without schema (`PromptedOutput` with JSON mode) |
| `supports_image_output` | `bool` | `False` | Model can generate images |
| `supports_inline_system_prompts` | `bool` | `False` | API accepts `SystemPromptPart` at any position; otherwise non-leading system prompts are wrapped as user content |
| `default_structured_output_mode` | `StructuredOutputMode` | `'tool'` | Default mode for structured output |
| `prompted_output_template` | `str` | See source | Template injected as instructions for prompted structured output; `{schema}` placeholder |
| `supports_thinking` | `bool` | `False` | Whether thinking/reasoning configuration is accepted |
| `thinking_always_enabled` | `bool` | `False` | Model always thinks (e.g. o-series, DeepSeek R1); `thinking=False` is silently ignored |
| `thinking_tags` | `tuple[str, str]` | `('<think>', '</think>')` | Tags wrapping thinking content in model output |
| `ignore_streamed_leading_whitespace` | `bool` | `False` | Workaround for Ollama + Qwen3 empty text parts before tool calls |
| `supported_native_tools` | `frozenset[type[AbstractNativeTool]]` | All native tools | Set of native tool types this model supports |
| `json_schema_transformer` | `type[JsonSchemaTransformer] \| None` | `None` | Transformer to adapt JSON schemas for model-specific constraints |

### Key methods (source-verified)

| Method | Behaviour |
|---|---|
| `from_profile(profile)` | Class method; if `profile` is already a subclass instance, returns it; otherwise creates a new subclass instance and calls `update(profile)` |
| `update(profile)` | Returns a `replace(self, **non_default_attrs)` merging non-default values from `profile` into `self` |
| `__new__` | Lazy-installs deprecated kwarg aliases (e.g. `supported_builtin_tools` → `supported_native_tools`) once per subclass via MRO walk |
| `__getattr__` | Supports deprecated attribute names; raises `AttributeError` for genuine typos |

```python
# Deprecated alias (installed lazily via __new__)
_MODEL_PROFILE_DEPRECATED_FIELD_ALIASES = {
    'supported_builtin_tools': 'supported_native_tools',
}
DEFAULT_PROFILE = ModelProfile()  # all defaults
```

`ModelProfileSpec = ModelProfile | Callable[[str], ModelProfile | None]` — a profile can also be a factory function that takes a model name string.

### Example 1 — inspect the default profile

```python
from pydantic_ai.profiles import ModelProfile, DEFAULT_PROFILE

print(DEFAULT_PROFILE.supports_tools)               # True
print(DEFAULT_PROFILE.supports_json_schema_output)  # False
print(DEFAULT_PROFILE.default_structured_output_mode)  # 'tool'
print(DEFAULT_PROFILE.thinking_tags)                # ('<think>', '</think>')
print(DEFAULT_PROFILE.supports_thinking)            # False
```

### Example 2 — custom profile for a constrained model

```python
from pydantic_ai.profiles import ModelProfile


class RestrictedModelProfile(ModelProfile):
    """Profile for a model that only supports text and JSON mode (no tools)."""
    pass


restricted = RestrictedModelProfile(
    supports_tools=False,
    supports_json_object_output=True,
    default_structured_output_mode='json',
    thinking_always_enabled=False,
)

print(restricted.supports_tools)               # False
print(restricted.default_structured_output_mode)  # 'json'
```

### Example 3 — profile factory function (`ModelProfileSpec`)

```python
from pydantic_ai.profiles import ModelProfile, ModelProfileSpec


def my_profile_factory(model_name: str) -> ModelProfile | None:
    """Return a custom profile based on model name prefix."""
    if model_name.startswith('thinking-'):
        return ModelProfile(
            supports_thinking=True,
            thinking_tags=('<thinking>', '</thinking>'),
            default_structured_output_mode='tool',
        )
    if model_name.startswith('json-only-'):
        return ModelProfile(
            supports_tools=False,
            supports_json_schema_output=True,
            default_structured_output_mode='json_schema',
        )
    return None  # use default profile


spec: ModelProfileSpec = my_profile_factory
```

---

## What's new in 1.107.0 relevant to Vol. 20

- **`Hooks` capability** (`pydantic_ai.capabilities.Hooks`) — ergonomic decorator-based hook registration; all 25+ hook types; `tools=` filter; `timeout=` per hook
- **`Instrumentation` capability** (`pydantic_ai.capabilities.Instrumentation`) — replaces `instrument=True`; `position='outermost'`; per-run state isolation; `ToolRetryError` classified as OK
- **`ConcurrencyLimiter`** (`pydantic_ai.concurrency`) — anyio-backed; OTel span per wait; `max_queued` backpressure; `ConcurrencyLimitExceeded`
- **Toolset composition** (`CombinedToolset`, `PrefixedToolset`, `RenamedToolset`, `FilteredToolset`, `PreparedToolset`) — composable toolset middleware without subclassing
- **`ApprovalRequiredToolset` / `ExternalToolset`** — HITL approval and external result injection; `DeferredToolset` deprecated alias
- **`TemplateStr`** (`pydantic_ai.TemplateStr`) — Handlebars templates for dynamic instructions; auto-detected in `Union[TemplateStr, str]`
- **`FunctionSchema`** (`pydantic_ai._function_schema`) — function → LLM schema conversion; `takes_ctx` auto-detection; `return_schema` extraction
- **`safe_download`** (`pydantic_ai._ssrf`) — SSRF-protected fetch; 14 IPv4 + 7 IPv6 private ranges; 7 cloud metadata IPs; IPv6 transition decoding; per-hop redirect validation
- **`LoadCapabilityCallPart` / `LoadCapabilityReturnPart`** (`pydantic_ai._deferred_capabilities`) — deferred capability wire protocol; `tool_kind='capability-load'` discriminator; `parse_loaded_capabilities(messages)`
- **`ModelProfile`** (`pydantic_ai.profiles.ModelProfile`) — model capability declaration; 14 fields; `from_profile` / `update` merge; `ModelProfileSpec` factory pattern; lazy deprecated kwarg alias installation

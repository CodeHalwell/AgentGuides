---
title: "PydanticAI: Toolsets"
description: "FunctionToolset, CombinedToolset, FilteredToolset, PrefixedToolset, RenamedToolset, PreparedToolset, ApprovalRequiredToolset, DeferredLoadingToolset, ExternalToolset — compose and reshape tool collections."
framework: pydanticai
language: python
---

# Toolsets

Verified against **pydantic-ai==1.102.0** — source module: `pydantic_ai.toolsets`.

A *toolset* is a reusable, named collection of tools with a shared policy (retries, timeout, metadata, instructions). PydanticAI ships 10+ toolset wrappers that let you filter, rename, combine, gate, or lazy-load tools without rewriting the functions. They're the supported way to attach non-code tool sources — MCP servers, remote APIs, human approval — to an agent.

## Minimal runnable example

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

tools = FunctionToolset[int]()  # generic deps type

@tools.tool
def multiply(ctx: RunContext[int], x: int) -> int:
    return ctx.deps * x

agent = Agent('openai:gpt-5.2', deps_type=int, toolsets=[tools])
print(agent.run_sync('Multiply my deps by 3', deps=7).output)
#> 21
```

`toolsets=[...]` lives next to `tools=[...]`. Tools registered via `@agent.tool` / `@agent.tool_plain` are included automatically; `toolsets=[...]` adds extra toolsets on top of those.

## The toolset catalogue

All of these live in `pydantic_ai.toolsets` and are exported from `pydantic_ai` directly.

| Toolset                       | Role                                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| `FunctionToolset`             | Wraps Python callables as tools. The primitive building block.                                |
| `CombinedToolset`             | Merges several toolsets into one (preserves ordering).                                        |
| `PrefixedToolset`             | Prepends a string to every tool name. Avoids collisions when combining.                       |
| `RenamedToolset`              | Per-tool rename map.                                                                          |
| `FilteredToolset`             | Drops tools via a `(ctx, tool_def) -> bool` predicate, evaluated per run step.                |
| `PreparedToolset`             | Runs a `(ctx, defs) -> defs` hook per step to mutate tool definitions.                        |
| `ApprovalRequiredToolset`     | Wraps a toolset so some/all calls raise `ApprovalRequired` until approved.                    |
| `DeferredLoadingToolset`      | Hides tools until discovered via tool search.                                                 |
| `ExternalToolset`             | Declares tool _schemas_ whose execution happens outside the agent (deferred).                 |
| `IncludeReturnSchemasToolset` | Sets `include_return_schema=True` on every wrapped tool.                                      |
| `SetMetadataToolset`          | Merges metadata onto every wrapped tool.                                                      |
| `WrapperToolset` / `AbstractToolset` | Base classes for custom toolsets.                                                      |
| `MCPServer*` (in `pydantic_ai.mcp`) | Toolsets backed by MCP stdio/SSE/HTTP.                                                  |

## `FunctionToolset` — the primitive

`toolsets/function.py:44`. Constructor args (verified at `:60`):

| Arg                           | Default                   | Notes                                                              |
| ----------------------------- | ------------------------- | ------------------------------------------------------------------ |
| `tools`                       | `[]`                      | `Sequence[Tool | ToolFunc]` — seed tools.                          |
| `max_retries`                 | `1`                       | Per-tool retry budget.                                             |
| `timeout`                     | `None`                    | Seconds per tool call (per-tool override available).               |
| `docstring_format`            | `'auto'`                  | `'google' | 'numpy' | 'sphinx' | 'auto'`.                         |
| `require_parameter_descriptions` | `False`                | If `True`, missing param doc raises at registration.               |
| `schema_generator`            | `GenerateToolJsonSchema`  | Override Pydantic JSON-schema generator.                           |
| `strict`                      | `None`                    | Forward `strict` hint to OpenAI.                                   |
| `sequential`                  | `False`                   | Tools in this set must run serially.                               |
| `requires_approval`           | `False`                   | All tools require HITL approval.                                   |
| `metadata`                    | `None`                    | Merged into each tool's metadata.                                  |
| `defer_loading`               | `False`                   | Hide from model until tool search surfaces them.                   |
| `include_return_schema`       | `None`                    | Include tool return schemas in definitions.                        |
| `id`                          | `None`                    | Required when using under durable execution (Temporal).            |
| `instructions`                | `None`                    | Auto-injected instruction string(s) when any tool is active.       |

Register tools three ways:

```python
tools = FunctionToolset[None]()

@tools.tool
def ping(ctx: RunContext[None]) -> str:   # decorator with ctx
    return 'pong'

@tools.tool_plain                          # no RunContext needed
def square(x: int) -> int:
    return x * x

tools.add_function(lambda x: x + 1, name='inc')   # programmatic add
```

## Composition examples

### `CombinedToolset` — layering

```python
from pydantic_ai import CombinedToolset, FunctionToolset

core = FunctionToolset([...])
extras = FunctionToolset([...])
combined = CombinedToolset([core, extras])
agent = Agent('openai:gpt-5.2', toolsets=[combined])
```

Tool-name collisions raise at construction time; `PrefixedToolset` solves that.

### `PrefixedToolset` — namespaces

```python
from pydantic_ai import PrefixedToolset

agent = Agent('openai:gpt-5.2', toolsets=[
    PrefixedToolset(db_tools, prefix='db_'),
    PrefixedToolset(kb_tools, prefix='kb_'),
])
# model sees: db_search, db_write, kb_search, ...
```

### `RenamedToolset` — per-tool rename

```python
from pydantic_ai import RenamedToolset

renamed = RenamedToolset(tools, name_map={'lookup': 'find_customer'})
```

### `FilteredToolset` — conditional visibility

```python
from pydantic_ai import FilteredToolset

def visible(ctx, tool_def):
    # only expose write tools to admins
    return tool_def.metadata.get('scope') != 'write' or ctx.deps.user.is_admin

agent = Agent('openai:gpt-5.2', deps_type=Deps,
              toolsets=[FilteredToolset(tools, filter_func=visible)])
```

Evaluated every step — you can hide a tool once a certain state is reached.

### `PreparedToolset` — mutate definitions on the fly

```python
from pydantic_ai import PreparedToolset
from pydantic_ai.tools import ToolDefinition

async def strict_openai(ctx, defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [d._replace(strict=True) for d in defs]

prep = PreparedToolset(tools, prepare_func=strict_openai)
```

Use cases: toggling `strict`, swapping descriptions per locale, overriding schemas in a migration.

### `ApprovalRequiredToolset` — human-in-the-loop

```python
from pydantic_ai import ApprovalRequiredToolset, DeferredToolRequests, DeferredToolResults, ToolApproved

def needs_approval(ctx, tool_def, args) -> bool:
    return tool_def.name.startswith('delete_')

agent = Agent(
    'openai:gpt-5.2',
    output_type=[str, DeferredToolRequests],
    toolsets=[ApprovalRequiredToolset(write_tools, approval_required_func=needs_approval)],
)

result1 = agent.run_sync('Delete old records.')
if isinstance(result1.output, DeferredToolRequests):
    # Show result1.output.approvals to the user ...
    approvals = {call.tool_call_id: ToolApproved() for call in result1.output.approvals}
    result2 = agent.run_sync(
        message_history=result1.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals=approvals),
    )
```

`approval_required_func` defaults to `lambda ctx, tool_def, args: True` — every call requires approval. Return `False` to skip approval. On approval, the original tool runs; rejection sends `ToolDenied(message=...)` back to the model.

### `DeferredLoadingToolset` — tool search integration

```python
from pydantic_ai import DeferredLoadingToolset

big_library = FunctionToolset([...])
hidden = DeferredLoadingToolset(big_library)   # all tools hidden
agent = Agent('openai:gpt-5.2', toolsets=[hidden])
```

Combined with the built-in tool search capability (`pydantic_ai.capabilities.ToolSearch`), only tools the model asks for via search get surfaced — saves tokens on large libraries.

### `ExternalToolset` — execute outside the agent

```python
from pydantic_ai import ExternalToolset
from pydantic_ai.tools import ToolDefinition

external = ExternalToolset([
    ToolDefinition(
        name='slack_post',
        description='Post to a Slack channel.',
        parameters_json_schema={'type': 'object', 'properties': {'channel': {'type': 'string'}, 'text': {'type': 'string'}}, 'required': ['channel', 'text']},
    ),
])

agent = Agent('openai:gpt-5.2',
              output_type=[str, DeferredToolRequests],
              toolsets=[external])

result = agent.run_sync('Announce the release to #eng.')
if isinstance(result.output, DeferredToolRequests):
    for call in result.output.calls:
        # hand to your backend worker
        worker.enqueue(call.tool_name, call.args)
```

When all external calls complete you feed results back with `DeferredToolResults(calls={tool_call_id: ToolReturn(...)})`.

### `IncludeReturnSchemasToolset` — inject return schemas

Forces every tool's return schema into the definition sent to the model. Useful for providers that use return type hints to guide structured tool usage:

```python
from pydantic_ai import Agent, IncludeReturnSchemasToolset, FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic import BaseModel

class Product(BaseModel):
    id: int
    name: str
    price: float

tools = FunctionToolset[None]()

@tools.tool_plain
def get_product(product_id: int) -> Product:
    """Retrieve a product by ID."""
    return Product(id=product_id, name='Widget', price=9.99)

# OpenAI and Google models can use the Product schema as a hint
agent = Agent('openai:gpt-4o', toolsets=[IncludeReturnSchemasToolset(tools)])
```

### `SetMetadataToolset` — bulk-tag tools

Merges a metadata dictionary onto every tool in the wrapped toolset. Combine with `FilteredToolset` to create dynamic access control:

```python
from pydantic_ai import (
    Agent, FunctionToolset, SetMetadataToolset, FilteredToolset, CombinedToolset
)
from dataclasses import dataclass

@dataclass
class UserDeps:
    role: str   # 'admin' | 'reader'

# Two toolsets, tagged with their access level
read_tools  = FunctionToolset[UserDeps]()
write_tools = FunctionToolset[UserDeps]()

@read_tools.tool_plain
def list_records() -> list[str]:
    return ['record_1', 'record_2']

@write_tools.tool_plain
def delete_record(record_id: str) -> str:
    return f'Deleted {record_id}'

# Tag all write tools as requiring admin access
tagged_write = SetMetadataToolset(write_tools, metadata={'requires_role': 'admin'})

# Filter based on user's role at runtime
def role_filter(ctx, tool_def) -> bool:
    required = tool_def.metadata and tool_def.metadata.get('requires_role')
    if required is None:
        return True
    return ctx.deps.role == required

agent = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    toolsets=[
        read_tools,
        FilteredToolset(tagged_write, filter_func=role_filter),
    ],
)

# Admin sees all tools; reader only sees read tools
result_admin  = agent.run_sync('Delete record_1', deps=UserDeps(role='admin'))
result_reader = agent.run_sync('Delete record_1', deps=UserDeps(role='reader'))
```

## Instructions that follow a toolset

```python
tools = FunctionToolset(
    [...],
    instructions='When using DB tools, prefer read-only unless the user explicitly asks to write.',
)
```

The string is automatically appended to the model's instructions when any tool in this set is active. You can also pass a callable `(ctx) -> str` or an async one.

## Using an agent _as_ a toolset

```python
from pydantic_ai import Agent

sub = Agent('openai:gpt-5.2-mini', name='citations')

@sub.tool_plain
def lookup_citation(key: str) -> str: ...

parent = Agent('openai:gpt-5.2', toolsets=[sub.toolset])
```

Every `Agent` exposes a `.toolset` (an internal `FunctionToolset`) for reuse.

## Building custom toolsets with `WrapperToolset` and `AbstractToolset`

### `WrapperToolset` — decorate an existing toolset

`WrapperToolset` wraps another toolset and delegates all calls. Override `get_tools` or `call_tool` to add cross-cutting behaviour without rebuilding from scratch:

```python
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
import time

@dataclass
class TimedToolset(WrapperToolset):
    """A toolset that logs execution time for every tool call."""

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        t0 = time.perf_counter()
        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
            elapsed = time.perf_counter() - t0
            print(f'[{name}] completed in {elapsed:.3f}s → {result!r}')
            return result
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f'[{name}] failed in {elapsed:.3f}s: {e}')
            raise

# Wrap any existing toolset
base_tools = FunctionToolset[None]()

@base_tools.tool_plain
def slow_operation(n: int) -> int:
    import time; time.sleep(0.1)
    return n * 2

agent = Agent('openai:gpt-4o', toolsets=[TimedToolset(wrapped=base_tools)])
```

### `AbstractToolset` — build from scratch

Implement `AbstractToolset` when you need full control over tool definitions and execution — for example, wrapping a database schema or a remote API registry:

```python
from abc import ABC
from dataclasses import dataclass
from typing import Any
import json
from pydantic_core import SchemaValidator, core_schema
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.tools import RunContext, ToolDefinition

@dataclass
class DatabaseToolset(AbstractToolset):
    """Dynamically exposes SQL tables as tools at runtime."""

    db_url: str
    _tables: dict[str, dict] | None = None

    @property
    def id(self) -> str | None:
        return f'db:{self.db_url}'

    async def __aenter__(self):
        # Connect to DB and introspect schema
        self._tables = await self._introspect_schema()
        return self

    async def __aexit__(self, *args):
        self._tables = None

    async def _introspect_schema(self) -> dict[str, dict]:
        # Returns {'users': {'id': 'int', 'name': 'str'}, ...}
        return {'users': {'id': 'integer', 'name': 'text'}}

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        tables = self._tables or {}
        result = {}
        for table, columns in tables.items():
            props = {col: {'type': 'string', 'description': f'{dtype} column'} for col, dtype in columns.items()}
            tool_def = ToolDefinition(
                name=f'query_{table}',
                description=f'Query the {table} table.',
                parameters_json_schema={
                    'type': 'object',
                    'properties': {'filter': {'type': 'string', 'description': 'SQL WHERE clause'}},
                    'required': [],
                },
            )
            validator = SchemaValidator(core_schema.dict_schema())
            result[tool_def.name] = ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=1,
                args_validator=validator,
            )
        return result

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        table = name.removeprefix('query_')
        where = tool_args.get('filter', '1=1')
        # Execute: SELECT * FROM {table} WHERE {where}
        return [{'id': 1, 'name': 'Alice'}]   # placeholder

agent = Agent('openai:gpt-4o', toolsets=[DatabaseToolset(db_url='postgresql://...')])
async with agent:
    result = await agent.run('List all users')
```

## Dynamic toolsets — `@agent.toolset`

`agent/__init__.py:2237`. Register a factory that builds a toolset per run based on `RunContext`:

```python
@agent.toolset
async def per_tenant(ctx: RunContext[TenantDeps]) -> AbstractToolset[TenantDeps]:
    return FunctionToolset([load_tools_for(ctx.deps.tenant_id)])
```

## `PreparedToolset` — advanced patterns

### Internationalization: per-locale tool descriptions

```python
import asyncio
import dataclasses
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, PreparedToolset, RunContext
from pydantic_ai.tools import ToolDefinition

DESCRIPTIONS = {
    'en': {
        'search_products': 'Search the product catalogue.',
        'get_order': 'Retrieve an order by ID.',
    },
    'es': {
        'search_products': 'Buscar en el catálogo de productos.',
        'get_order': 'Recuperar un pedido por ID.',
    },
    'ja': {
        'search_products': '商品カタログを検索します。',
        'get_order': 'IDで注文を取得します。',
    },
}

@dataclass
class UserDeps:
    locale: str = 'en'

tools = FunctionToolset[UserDeps]()

@tools.tool_plain
def search_products(query: str) -> list[str]:
    return [f'Product: {query}']

@tools.tool_plain
def get_order(order_id: str) -> dict:
    return {'id': order_id, 'status': 'shipped'}

def localise_descriptions(ctx: RunContext[UserDeps], defs: list[ToolDefinition]) -> list[ToolDefinition]:
    locale_map = DESCRIPTIONS.get(ctx.deps.locale, DESCRIPTIONS['en'])
    return [
        dataclasses.replace(d, description=locale_map.get(d.name, d.description))
        for d in defs
    ]

agent = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    toolsets=[PreparedToolset(tools, prepare_func=localise_descriptions)],
)

async def main():
    result = await agent.run('¿Puedes buscar laptops?', deps=UserDeps(locale='es'))
    print(result.output)

asyncio.run(main())
```

### Progressively restrict tools as workflow advances

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, PreparedToolset, RunContext
from pydantic_ai.tools import ToolDefinition

@dataclass
class WorkflowDeps:
    phase: str = 'init'  # 'init' → 'validated' → 'committed'

tools = FunctionToolset[WorkflowDeps]()

@tools.tool
def validate_data(ctx: RunContext[WorkflowDeps], data: str) -> str:
    ctx.deps.phase = 'validated'
    return f'Validated: {data}'

@tools.tool
def commit_transaction(ctx: RunContext[WorkflowDeps], transaction_id: str) -> str:
    ctx.deps.phase = 'committed'
    return f'Committed: {transaction_id}'

@tools.tool
def rollback(ctx: RunContext[WorkflowDeps], reason: str) -> str:
    ctx.deps.phase = 'init'
    return f'Rolled back: {reason}'

# Phase-gated tool visibility
PHASE_TOOLS: dict[str, set[str]] = {
    'init': {'validate_data'},
    'validated': {'commit_transaction', 'rollback'},
    'committed': set(),  # no tools after commit
}

def phase_filter(ctx: RunContext[WorkflowDeps], defs: list[ToolDefinition]) -> list[ToolDefinition]:
    allowed = PHASE_TOOLS.get(ctx.deps.phase, set())
    return [d for d in defs if d.name in allowed]

agent = Agent(
    'openai:gpt-4o',
    deps_type=WorkflowDeps,
    toolsets=[PreparedToolset(tools, prepare_func=phase_filter)],
)

async def main():
    deps = WorkflowDeps(phase='init')
    result = await agent.run('Process data "order_42" and commit if valid.', deps=deps)
    print(f'Final phase: {deps.phase}')
    print(result.output)

asyncio.run(main())
```

## `DeferredLoadingToolset` — advanced patterns

### Gradual tool discovery with tool search

When using `DeferredLoadingToolset` with the `ToolSearch` capability, the model discovers tools through search rather than seeing them all upfront. This is especially powerful for agents with 50+ tools:

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset, DeferredLoadingToolset

# Large library with many specialised tools
analytics_tools = FunctionToolset[None]()

@analytics_tools.tool_plain
def cohort_analysis(cohort_id: str, metric: str) -> dict:
    """Run a cohort analysis for the given metric."""
    return {'cohort': cohort_id, 'metric': metric, 'value': 42.5}

@analytics_tools.tool_plain
def funnel_report(funnel_name: str, date_range: str) -> dict:
    """Generate a conversion funnel report."""
    return {'funnel': funnel_name, 'conversion_rate': 0.23}

@analytics_tools.tool_plain
def retention_curve(product_id: str, cohort_weeks: int) -> list[float]:
    """Compute a retention curve for a product cohort."""
    return [1.0, 0.8, 0.65, 0.55, 0.48]

@analytics_tools.tool_plain
def ab_test_significance(test_id: str) -> dict:
    """Calculate statistical significance for an A/B test."""
    return {'test_id': test_id, 'p_value': 0.03, 'significant': True}

# Defer ALL analytics tools — the model must search for them
deferred = DeferredLoadingToolset(analytics_tools)
agent = Agent('openai:gpt-4o', toolsets=[deferred])

async def main():
    # Without deferred loading, all 4 tools appear in every prompt.
    # With deferred loading, only tools the model searches for are loaded.
    result = await agent.run('Is A/B test "checkout_v2" statistically significant?')
    print(result.output)

asyncio.run(main())
```

### Mixing deferred and always-visible tools

Expose lightweight utility tools immediately; defer heavy/specialised ones:

```python
from pydantic_ai import Agent, FunctionToolset, DeferredLoadingToolset

# Always-visible: fast, cheap, universally needed
quick_tools = FunctionToolset[None]()

@quick_tools.tool_plain
def get_current_date() -> str:
    from datetime import date
    return date.today().isoformat()

@quick_tools.tool_plain
def format_number(n: float, decimals: int = 2) -> str:
    return f'{n:,.{decimals}f}'

# Deferred: expensive or rarely needed
heavy_tools = FunctionToolset[None]()

@heavy_tools.tool_plain
def run_ml_model(model_name: str, input_data: dict) -> dict:
    """Run inference on a large ML model."""
    return {'prediction': 0.87}

@heavy_tools.tool_plain
def generate_report(report_type: str, parameters: dict) -> str:
    """Generate a complex analytical report."""
    return f'{report_type} report generated'

agent = Agent(
    'openai:gpt-4o',
    toolsets=[
        quick_tools,                                    # always visible
        DeferredLoadingToolset(heavy_tools),            # discovered on demand
    ],
)
```

## Gotchas

- **Enter before use**: toolsets may hold resources (processes, HTTP clients, MCP sessions). Using an agent as an async context manager (`async with agent: ...`) enters every toolset.
- **Naming collisions**: `CombinedToolset` raises if two toolsets expose the same tool name. Wrap with `PrefixedToolset` or `RenamedToolset` to disambiguate.
- **`requires_approval=True` without `DeferredToolRequests`** in `output_type` raises at runtime. Always add `DeferredToolRequests` to the output union.
- **`ExternalToolset` + streaming**: external deferrals terminate the stream early. Handle `DeferredToolRequests` as a normal output value.
- **Durable execution**: every toolset must have an `id` when running under Temporal/Prefect/DBOS so activities can be routed.
- **`PreparedToolset` constraint**: the prepare function cannot add or rename tools. Reducing or modifying definitions is fine; use `RenamedToolset` for renaming and `FunctionToolset.add_function()` for additions.
- **`DeferredLoadingToolset` + non-search agent**: if the agent doesn't have a `ToolSearch` capability, deferred tools are simply never offered to the model. Make sure `ToolSearch` or the built-in tool search is active.

## Patterns

### 1. Tenant-scoped toolset with filtering

```python
def own_tenant(ctx, tool_def):
    return tool_def.metadata.get('tenant') == ctx.deps.tenant_id
agent = Agent(..., toolsets=[FilteredToolset(all_tools, filter_func=own_tenant)])
```

### 2. Write-operations behind HITL

```python
ApprovalRequiredToolset(write_tools,
    approval_required_func=lambda ctx, d, a: d.metadata.get('destructive', False))
```

### 3. MCP server alongside local tools

```python
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio('uv', args=['run', 'mcp-run-python', 'stdio'])
agent = Agent('openai:gpt-5.2',
              toolsets=[local_tools, PrefixedToolset(server, prefix='mcp_')])
async with agent:
    result = await agent.run('run this python snippet safely')
```

### 4. Progressive disclosure with `DeferredLoadingToolset`

```python
deep_library = FunctionToolset([...])  # 120 tools
agent = Agent('openai:gpt-5.2',
              toolsets=[DeferredLoadingToolset(deep_library)])
```

Combined with `ToolSearch` capability, only searched tools appear in the step.

### 5. External tool execution dispatched to a queue

```python
external = ExternalToolset([ToolDefinition(...)])
agent = Agent(..., output_type=[str, DeferredToolRequests], toolsets=[external])
result = agent.run_sync(prompt)
if isinstance(result.output, DeferredToolRequests):
    for call in result.output.calls:
        queue.push({'id': call.tool_call_id, 'name': call.tool_name, 'args': call.args})
```

### 6. Full-featured multi-source toolset with approval and scoping

This end-to-end example shows `CombinedToolset`, `PrefixedToolset`, `FilteredToolset`, and `ApprovalRequiredToolset` working together for a multi-tenant CRUD agent.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai import FunctionToolset, CombinedToolset, PrefixedToolset, FilteredToolset, ApprovalRequiredToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied, RunContext

@dataclass
class UserDeps:
    user_id: str
    is_admin: bool
    tenant_id: str

# --- Read tools ---
read_tools = FunctionToolset[UserDeps](metadata={'scope': 'read'})

@read_tools.tool
def list_records(ctx: RunContext[UserDeps], limit: int = 10) -> list[str]:
    """List records for the current tenant."""
    return [f'record-{ctx.deps.tenant_id}-{i}' for i in range(limit)]

@read_tools.tool
def get_record(ctx: RunContext[UserDeps], record_id: str) -> dict:
    """Fetch a single record."""
    return {'id': record_id, 'tenant': ctx.deps.tenant_id}

# --- Write tools (require admin approval) ---
write_tools = FunctionToolset[UserDeps](metadata={'scope': 'write'})

@write_tools.tool
def delete_record(ctx: RunContext[UserDeps], record_id: str) -> str:
    """Permanently delete a record."""
    return f'Deleted {record_id}'

@write_tools.tool
def bulk_update(ctx: RunContext[UserDeps], field: str, value: str) -> str:
    """Update a field on all records for this tenant."""
    return f'Updated {field}={value} on all records'

# Gate write operations behind human approval
gated_write_tools = ApprovalRequiredToolset(write_tools)

# Prefix both toolsets to avoid name collisions
combined = CombinedToolset([
    PrefixedToolset(read_tools, prefix='read_'),
    PrefixedToolset(gated_write_tools, prefix='write_'),
])

# Filter: non-admins only see read tools
def admin_filter(ctx: RunContext[UserDeps], tool_def) -> bool:
    if tool_def.name.startswith('write_') and not ctx.deps.is_admin:
        return False
    return True

agent = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    output_type=[str, DeferredToolRequests],
    toolsets=[FilteredToolset(combined, filter_func=admin_filter)],
)

async def main():
    admin = UserDeps(user_id='u1', is_admin=True, tenant_id='acme')
    result = await agent.run('Delete record r-42 and list remaining records.', deps=admin)

    if isinstance(result.output, DeferredToolRequests):
        print('Awaiting approval for:')
        for call in result.output.approvals:
            print(f'  {call.tool_name}({call.args_as_dict()})')

        # Admin approves deletions
        approvals = {c.tool_call_id: ToolApproved() for c in result.output.approvals}
        final = await agent.run(
            'continue',
            deps=admin,
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals),
        )
        print(final.output)
    else:
        print(result.output)

asyncio.run(main())
```

### 7. `FunctionToolset` with `instructions` and per-toolset timeout

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset, RunContext

db_tools = FunctionToolset[None](
    timeout=5.0,   # Any tool taking >5s gets a ModelRetry prompt
    instructions=(
        'When querying the database, always filter by active=True unless '
        'the user explicitly asks for inactive records.'
    ),
)

@db_tools.tool_plain
def query_users(filter_active: bool = True) -> list[str]:
    """Query users from the database."""
    import time; time.sleep(0.1)  # simulate DB latency
    return ['alice', 'bob'] if filter_active else ['alice', 'bob', 'charlie_inactive']

@db_tools.tool_plain
def count_records(table: str) -> int:
    """Count rows in a database table."""
    return {'users': 2, 'orders': 15}.get(table, 0)

agent = Agent('openai:gpt-4o', toolsets=[db_tools])
result = agent.run_sync('How many users are there and who are they?')
print(result.output)
```

### 8. `FilteredToolset` with async predicate

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, FilteredToolset, RunContext

@dataclass
class RequestDeps:
    user_token: str

def search_web(ctx: RunContext[RequestDeps], query: str) -> str:
    return f'Results for: {query}'

def send_notification(ctx: RunContext[RequestDeps], message: str) -> str:
    return f'Notification sent: {message}'

all_tools = FunctionToolset[RequestDeps]([search_web, send_notification])

async def permission_check(ctx: RunContext[RequestDeps], tool_def) -> bool:
    """Async filter — could call an auth service."""
    if tool_def.name == 'send_notification':
        # Simulate checking permissions via an API
        await asyncio.sleep(0.01)
        return ctx.deps.user_token.startswith('premium-')
    return True

agent = Agent(
    'openai:gpt-4o',
    deps_type=RequestDeps,
    toolsets=[FilteredToolset(all_tools, filter_func=permission_check)],
)

async def main():
    free_user = RequestDeps(user_token='free-abc')
    result = await agent.run('Search for Python news and notify the team.', deps=free_user)
    print(result.output)  # Can search but not notify

asyncio.run(main())
```

## Reference

- `AbstractToolset` — `toolsets/abstract.py`
- `FunctionToolset` — `toolsets/function.py:44`
- `CombinedToolset` — `toolsets/combined.py:26`
- `PrefixedToolset` / `RenamedToolset` / `FilteredToolset` / `PreparedToolset` — `toolsets/*.py`
- `ApprovalRequiredToolset` — `toolsets/approval_required.py:16`
- `DeferredLoadingToolset` — `toolsets/deferred_loading.py:12`
- `ExternalToolset` — `toolsets/external.py:17`
- `IncludeReturnSchemasToolset` — `toolsets/include_return_schemas.py:12`
- `SetMetadataToolset` — `toolsets/set_metadata.py`

---
title: "PydanticAI: Toolsets"
description: "FunctionToolset, CombinedToolset, FilteredToolset, PrefixedToolset, RenamedToolset, PreparedToolset, ApprovalRequiredToolset, DeferredLoadingToolset, ExternalToolset — compose and reshape tool collections."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Toolsets"
  order: 3
---

Checked against **pydantic-ai %%version=pydantic-ai%%** (%%versions-checked%%). Source package: `pydantic_ai.toolsets`.

A *toolset* is a reusable, named collection of tools with a shared policy (retries, timeout, metadata, instructions). PydanticAI ships about a dozen toolset classes that let you filter, rename, combine, gate, or lazy-load tools without rewriting the functions. They're the supported way to attach non-code tool sources — MCP servers, remote APIs, human approval — to an agent.

## Minimal runnable example

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

tools = FunctionToolset[int]()  # generic deps type

@tools.tool
def multiply(ctx: RunContext[int], x: int) -> int:
    return ctx.deps * x

agent = Agent('openai:%%model=openai%%', deps_type=int, toolsets=[tools])
print(agent.run_sync('Multiply my deps by 3', deps=7).output)
#> 21
```

`toolsets=[...]` lives next to `tools=[...]`. Tools registered via `@agent.tool` / `@agent.tool_plain` are included automatically; `toolsets=[...]` adds extra toolsets on top of those.

## The toolset catalogue

All of these live in `pydantic_ai.toolsets`; all except `DynamicToolset` are also exported from `pydantic_ai`.

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
| `MCPToolset` (in `pydantic_ai.mcp`) | Toolset backed by an MCP server over stdio, HTTP or in-process FastMCP. The V1 `MCPServer*` classes were removed in 2.0. |

## `FunctionToolset` — the primitive

`pydantic_ai.toolsets.FunctionToolset` (also exported from `pydantic_ai`). Constructor arguments:

| Arg                           | Default                   | Notes                                                              |
| ----------------------------- | ------------------------- | ------------------------------------------------------------------ |
| `tools`                       | `[]`                      | `Sequence[Tool | ToolFunc]` — seed tools.                          |
| `max_retries`                 | `None`                    | Per-tool retry budget; `None` inherits the agent's `retries`.     |
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
from pydantic_ai import FunctionToolset, RunContext

tools = FunctionToolset[None]()

@tools.tool
def ping(ctx: RunContext[object]) -> str:   # decorator with ctx
    return 'pong'

@tools.tool_plain                          # no RunContext needed
def square(x: int) -> int:
    return x * x

tools.add_function(lambda x: x + 1, name='inc')   # programmatic add
```

### Instructions that follow a toolset

`FunctionToolset(instructions=...)` injects guidance into the model request whenever any tool from the set is active. Four forms are accepted:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, RunContext

# 1. Plain string — always injected
db_tools = FunctionToolset(
    instructions='When using DB tools, prefer read-only unless the user explicitly asks to write.',
)

# 2. Sync callable — computed at run time from RunContext
@dataclass
class Locale:
    lang: str

def locale_hint(ctx: RunContext[Locale]) -> str:
    return f'Always respond in {ctx.deps.lang}.'

kb_tools = FunctionToolset[Locale](instructions=locale_hint)

# 3. Async callable — same shape, awaited before the request
async def async_locale_hint(ctx: RunContext[Locale]) -> str:
    # fetch from a config service, etc.
    return f'Preferred language: {ctx.deps.lang}.'

kb_tools_async = FunctionToolset[Locale](instructions=async_locale_hint)

# 4. Sequence — multiple instructions combined in order
mixed_tools = FunctionToolset[Locale](
    instructions=[
        'Be concise.',
        locale_hint,          # sync callable
        async_locale_hint,    # async callable
    ]
)
```

### Per-toolset timeout

`timeout=` applies to every tool in the set (a tool's own `timeout` wins). A call that runs over is treated as a failed attempt: the model gets a retry prompt, which counts towards the retry budget.

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

agent = Agent('openai:%%model=openai%%', toolsets=[db_tools])
result = agent.run_sync('How many users are there and who are they?')
print(result.output)
```

## Combining and reshaping toolsets

### `CombinedToolset` and `PrefixedToolset`

`CombinedToolset([a, b])` merges toolsets. It constructs even when two children expose the same tool name; the `UserError` comes when the run lists the tools. `PrefixedToolset(toolset, prefix='web')` renames every tool to `web_<name>` (the underscore is added for you) and strips the prefix again when it dispatches the call, so the underlying function keeps its own name in `RunContext.tool_name`.

```python
from pydantic_ai import Agent, FunctionToolset, PrefixedToolset, CombinedToolset, RunContext
from dataclasses import dataclass

@dataclass
class Deps:
    user: str

# Two toolsets with a name collision: both have a "search" tool
web_tools = FunctionToolset[Deps]()
db_tools = FunctionToolset[Deps]()

@web_tools.tool_plain
def search(query: str) -> str:
    """Search the web for information."""
    return f'Web results for: {query}'

@db_tools.tool_plain
def search(query: str) -> str:  # noqa: F811 — same name, different toolset
    """Search the internal database."""
    return f'DB results for: {query}'

# Without prefixing, the run would raise UserError on the name collision.
agent = Agent(
    'openai:%%model=openai%%',
    deps_type=Deps,
    toolsets=[
        CombinedToolset([
            PrefixedToolset(web_tools, prefix='web'),
            PrefixedToolset(db_tools, prefix='db'),
        ])
    ],
)
# Model now sees: web_search, db_search — no collision.
result = agent.run_sync('Search the web for Python 3.13 news.', deps=Deps(user='alice'))
print(result.output)
```

**`tool_name_conflict_hint`** — if a collision still occurs after prefixing, the error message says _"Change the `prefix` attribute to avoid name conflicts."_ You can customise this hint on a subclass:

```python
from pydantic_ai import PrefixedToolset

class MyPrefixed(PrefixedToolset):
    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Rename the conflicting tool in the underlying FunctionToolset.'
```

### `RenamedToolset`

`name_map` maps **new name → original name**; unmapped tools pass through. Renaming onto a name that already exists raises `UserError`.

```python
from pydantic_ai import Agent, FunctionToolset, RenamedToolset

crm = FunctionToolset()

@crm.tool_plain
def lookup(email: str) -> str:
    """Find a customer by email."""
    return f'customer for {email}'

renamed = RenamedToolset(crm, name_map={'find_customer': 'lookup'})
agent = Agent('openai:%%model=openai%%', toolsets=[renamed])   # the model sees find_customer
```

### `FilteredToolset` — per-step conditional visibility

`FilteredToolset` calls `filter_func(ctx, tool_def) -> bool` **on every agent step** before handing the tool list to the model. The filter is re-evaluated at each step, so you can dynamically hide or reveal tools based on conversation state.

Both sync and async filter functions are accepted.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, FilteredToolset, RunContext

@dataclass
class UserContext:
    role: str          # 'admin' | 'viewer'
    subscription: str  # 'free' | 'pro'

tools = FunctionToolset[UserContext]()

@tools.tool_plain
def list_reports() -> list[str]:
    """List available reports."""
    return ['q1_report', 'q2_report']

@tools.tool_plain
def delete_report(report_id: str) -> str:
    """Delete a report permanently. Admin only."""
    return f'Deleted {report_id}'

@tools.tool_plain
def export_csv(report_id: str) -> str:
    """Export a report as CSV. Pro subscribers only."""
    return f'Exported {report_id} as CSV'

# Sync filter — called per step, has access to ctx.deps
def rbac_filter(ctx: RunContext[UserContext], tool_def) -> bool:
    if tool_def.name == 'delete_report' and ctx.deps.role != 'admin':
        return False
    if tool_def.name == 'export_csv' and ctx.deps.subscription != 'pro':
        return False
    return True

agent = Agent('openai:%%model=openai%%', deps_type=UserContext,
              toolsets=[FilteredToolset(tools, filter_func=rbac_filter)])

viewer = UserContext(role='viewer', subscription='free')
result = agent.run_sync('List reports and delete report q1.', deps=viewer)
# viewer only sees list_reports — the model can't call delete_report or export_csv
print(result.output)
```

**Async filter** — useful for fetching permissions from an external service:

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, FilteredToolset, RunContext
from pydantic_ai.tools import ToolDefinition

@dataclass
class UserSession:
    user_id: str
    scopes: list[str]   # e.g. ['read', 'write', 'admin']

write_ops = FunctionToolset[UserSession]()

@write_ops.tool
async def delete_record(ctx: RunContext[UserSession], record_id: str) -> str:
    return f'Deleted {record_id}'

@write_ops.tool
async def bulk_export(ctx: RunContext[UserSession], table: str) -> str:
    return f'Exporting {table}'

# Async predicate — can hit a permissions service
async def scope_check(ctx: RunContext[UserSession], tool_def: ToolDefinition) -> bool:
    metadata = tool_def.metadata or {}   # metadata is None when not set at registration
    required = metadata.get('required_scope', 'read')
    return required in ctx.deps.scopes

# Tag tools at registration with metadata
@write_ops.tool(metadata={'required_scope': 'admin'})
async def drop_table(ctx: RunContext[UserSession], table: str) -> str:
    return f'Dropped {table}'

agent = Agent(
    'openai:%%model=openai%%',
    deps_type=UserSession,
    toolsets=[FilteredToolset(write_ops, filter_func=scope_check)],
)

# Reader only sees tools where 'read' is the required scope (or no scope set)
reader = UserSession(user_id='u1', scopes=['read'])
# Admin sees every tool including drop_table
admin  = UserSession(user_id='u2', scopes=['read', 'write', 'admin'])

# The filter runs before every model request — tool visibility can change mid-run
async def demo():
    r1 = await agent.run('List options', deps=reader)
    r2 = await agent.run('Drop the logs table', deps=admin)
    return r1.output, r2.output
```

**State-dependent filtering** — use the filter to hide a tool once a certain condition is reached:

```python
from pydantic_ai.messages import ModelRequest
from pydantic_ai import Agent, FunctionToolset, FilteredToolset, RunContext

tools = FunctionToolset[None]()

@tools.tool_plain
def confirm_purchase() -> str:
    """Confirm the purchase. Only available once the cart is non-empty."""
    return 'Purchase confirmed!'

@tools.tool_plain
def add_to_cart(item: str) -> str:
    """Add an item to the cart."""
    return f'{item} added to cart.'

# Count how many items were added based on tool history
def cart_filter(ctx: RunContext[object], tool_def) -> bool:
    if tool_def.name != 'confirm_purchase':
        return True
    # Only show confirm_purchase if add_to_cart was called at least once
    cart_calls = sum(
        1 for msg in ctx.messages
        for part in msg.parts
        if hasattr(part, 'tool_name') and part.tool_name == 'add_to_cart'
    )
    return cart_calls > 0

agent = Agent('openai:%%model=openai%%', toolsets=[FilteredToolset(tools, filter_func=cart_filter)])
```

### `PreparedToolset` — rewrite definitions per step

`prepare_func(ctx, tool_defs) -> tool_defs` runs before every model request. It may drop tools or change their definitions (description, `strict`, metadata) but cannot add or rename them; that raises `UserError`. `ToolDefinition` is a dataclass, so use `dataclasses.replace`.

```python
from dataclasses import replace

from pydantic_ai import Agent, FunctionToolset, PreparedToolset, RunContext
from pydantic_ai.tools import ToolDefinition

tools = FunctionToolset()

@tools.tool_plain
def get_price(sku: str) -> float:
    """Look up a price."""
    return 9.99

async def make_strict(ctx: RunContext, defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [replace(d, strict=True) for d in defs]

agent = Agent('openai:%%model=openai%%', toolsets=[PreparedToolset(tools, prepare_func=make_strict)])
```

Two common uses: per-locale descriptions, and tools that appear only at a given stage of a workflow.

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
    'openai:%%model=openai%%',
    deps_type=UserDeps,
    toolsets=[PreparedToolset(tools, prepare_func=localise_descriptions)],
)

async def main():
    result = await agent.run('¿Puedes buscar laptops?', deps=UserDeps(locale='es'))
    print(result.output)

asyncio.run(main())
```

Phase-gated visibility, where a tool's own side effect decides what the model may call next:

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
    'openai:%%model=openai%%',
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

### `SetMetadataToolset` — tag every tool in a set

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
    'openai:%%model=openai%%',
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
agent = Agent('openai:%%model=openai%%', toolsets=[IncludeReturnSchemasToolset(tools)])
```

## Deferred tools: approval and external execution

A deferred tool call ends the run with a `DeferredToolRequests` output instead of a tool result. Add `DeferredToolRequests` to `output_type`, then resume with `message_history=` and `deferred_tool_results=`. Approval-type calls are in `.approvals`, external ones in `.calls`.

### `ApprovalRequiredToolset` — human-in-the-loop

`ApprovalRequiredToolset` wraps any toolset so that calling one of its tools raises `ApprovalRequired`. The agent catches this and — if the output type includes `DeferredToolRequests` — surfaces it as a structured value that your application can use to ask a human for approval.

#### Full HITL workflow

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, ApprovalRequiredToolset, RunContext
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

@dataclass
class AdminDeps:
    admin_email: str

# Tools that always need approval
dangerous_tools = FunctionToolset[AdminDeps]()

@dangerous_tools.tool
def delete_user(ctx: RunContext[AdminDeps], user_id: str) -> str:
    """Permanently delete a user account."""
    return f'User {user_id} deleted by {ctx.deps.admin_email}'

@dangerous_tools.tool
def bulk_export(ctx: RunContext[AdminDeps], table: str) -> str:
    """Export an entire database table to CSV."""
    return f'Exported table {table}'

# Wrap with approval gate
gated = ApprovalRequiredToolset(dangerous_tools)

agent = Agent(
    'openai:%%model=openai%%',
    deps_type=AdminDeps,
    output_type=[str, DeferredToolRequests],  # <-- critical: allows DeferredToolRequests output
    toolsets=[gated],
)

async def main():
    deps = AdminDeps(admin_email='ops@example.com')
    result1 = await agent.run('Delete user u-42 and export the audit_log table.', deps=deps)

    if isinstance(result1.output, DeferredToolRequests):
        print('Model wants to call these tools (awaiting approval):')
        for call in result1.output.approvals:
            print(f'  [{call.tool_call_id}] {call.tool_name}({call.args_as_dict()})')

        # Human reviews and decides per call
        human_decisions: dict[str, bool | ToolApproved | ToolDenied] = {}
        for call in result1.output.approvals:
            answer = input(f'Approve {call.tool_name}({call.args_as_dict()})? [y/N] ')
            if answer.lower() == 'y':
                human_decisions[call.tool_call_id] = ToolApproved()
            else:
                human_decisions[call.tool_call_id] = ToolDenied(message='Operation not approved by operator.')

        # Resume the run with the decisions
        result2 = await agent.run(
            '',  # no new user message needed
            deps=deps,
            message_history=result1.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=human_decisions),
        )
        print(result2.output)
    else:
        print(result1.output)

asyncio.run(main())
```

#### Selective approval — `approval_required_func`

By default, `ApprovalRequiredToolset` requires approval for **every** call. Pass `approval_required_func` to gate only specific tools (continuing the example above):

```python
from pydantic_ai import ApprovalRequiredToolset, RunContext
from pydantic_ai.tools import ToolDefinition

def only_destructive(ctx: RunContext, tool_def: ToolDefinition, tool_args: dict) -> bool:
    """Require approval only for destructive operations."""
    return tool_def.name.startswith('delete_') or tool_def.name.startswith('bulk_')

gated_selective = ApprovalRequiredToolset(
    dangerous_tools,
    approval_required_func=only_destructive,
)
```

#### `ToolApproved` — override args

`ToolApproved` accepts an optional `override_args` to substitute different arguments before the tool actually runs. This lets an operator correct or sanitise the model's arguments (`result1` is the first run from the workflow above):

```python
from pydantic_ai.tools import ToolApproved

# Model wanted to delete u-99, operator redirects to a safer test user
decisions = {
    call.tool_call_id: ToolApproved(override_args={'user_id': 'test-user-sandbox'})
    for call in result1.output.approvals
    if call.tool_name == 'delete_user'
}
```

#### `DeferredToolRequests.build_results` convenience method

```python
# Approve all pending requests at once
deferred_results = result1.output.build_results(approve_all=True)

# Or approve some, deny others
deferred_results = result1.output.build_results(
    approvals={
        call.tool_call_id: ToolApproved()
        for call in result1.output.approvals
        if call.tool_name != 'bulk_export'
    },
)
```

### `ExternalToolset` — tools executed outside the agent run

`ExternalToolset` advertises tool schemas to the model but does **not** execute them. The agent pauses on a `DeferredToolRequests` output containing the model's calls; your application routes those calls to an external system (a queue, another process, a UI) and resumes the agent with the results.

This pattern is useful for:
- Long-running operations (file processing, slow APIs) that shouldn't block the agent.
- Operations that require UI interaction (file upload dialogs, OAuth flows).
- Durable execution contexts where the agent must survive a process restart.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition

# Define tool schemas — no Python implementation needed
external = ExternalToolset([
    ToolDefinition(
        name='upload_file',
        description='Upload a file to the document store. Returns the file ID.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'filename': {'type': 'string'},
                'content_type': {'type': 'string'},
            },
            'required': ['filename'],
        },
    ),
    ToolDefinition(
        name='run_etl_job',
        description='Trigger a long-running ETL job. Returns job ID.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'source_table': {'type': 'string'},
                'target_table': {'type': 'string'},
            },
            'required': ['source_table', 'target_table'],
        },
    ),
])

agent = Agent(
    'openai:%%model=openai%%',
    output_type=[str, DeferredToolRequests],
    toolsets=[external],
)

async def dispatch_to_queue(calls) -> dict[str, str]:
    """Simulate dispatching to an external job queue."""
    results = {}
    for call in calls:
        if call.tool_name == 'upload_file':
            args = call.args_as_dict()
            results[call.tool_call_id] = f'file_id_{args["filename"].replace(".", "_")}'
        elif call.tool_name == 'run_etl_job':
            args = call.args_as_dict()
            results[call.tool_call_id] = f'job_{args["source_table"]}_to_{args["target_table"]}'
    return results

async def main():
    result1 = await agent.run('Upload report.csv and run an ETL from raw_data to warehouse.')

    if isinstance(result1.output, DeferredToolRequests):
        print('External calls requested:')
        for call in result1.output.calls:
            print(f'  {call.tool_name}({call.args_as_dict()})')

        # Execute externally and collect results
        external_results = await dispatch_to_queue(result1.output.calls)

        # Resume agent with the external results
        result2 = await agent.run(
            '',
            message_history=result1.all_messages(),
            deferred_tool_results=DeferredToolResults(calls=external_results),
        )
        print(result2.output)
    else:
        print(result1.output)

asyncio.run(main())
```

**`DeferredToolset` was removed in 2.0** — `ExternalToolset` is the replacement (it was deprecated in V1). Likewise V1's `DeferredToolCalls` output type is now `DeferredToolRequests`, whose `.calls` replaced `.tool_calls`.

### Deferring from inside a tool — `ApprovalRequired` and `CallDeferred`

The wrappers above defer whole toolsets. For a per-call decision, raise from the tool itself:

- `ApprovalRequired(metadata=...)` pauses the call until a human approves it. On the resumed run `ctx.tool_call_approved` is `True`, so the same code path proceeds.
- `CallDeferred(metadata=...)` hands the call to something outside the run (a job queue, a browser); you supply its result later.

Either way the run ends with a `DeferredToolRequests` output (it must be in `output_type`): approval-type calls are in `.approvals`, external ones in `.calls`, and per-call metadata in `.metadata`. Resume with `deferred_tool_results=requests.build_results(approvals=..., calls=...)` (or `approve_all=True`).

```python
from pydantic_ai import Agent, ApprovalRequired, CallDeferred, DeferredToolRequests, RunContext

agent = Agent('openai:%%model=openai%%', output_type=[str, DeferredToolRequests])

@agent.tool
def refund(ctx: RunContext, order_id: str, amount: float) -> str:
    """Refund an order."""
    if amount > 100 and not ctx.tool_call_approved:
        raise ApprovalRequired(metadata={'reason': 'refund over 100'})
    return f'Refunded {amount} on {order_id}'

@agent.tool_plain
def render_report(name: str) -> str:
    """Render a report in the background."""
    raise CallDeferred(metadata={'queue': 'reports'})

result = agent.run_sync('Refund 250 on order A-1 and render the Q3 report.')
if isinstance(result.output, DeferredToolRequests):
    requests = result.output
    results = requests.build_results(
        approvals={call.tool_call_id: True for call in requests.approvals},
        calls={call.tool_call_id: 'report-123.pdf' for call in requests.calls},
    )
    final = agent.run_sync(message_history=result.all_messages(), deferred_tool_results=results)
    print(final.output)
```

To resolve deferrals *inside* the run instead of returning them — auto-approve low-risk calls, deny others, call a policy service — add the `HandleDeferredToolCalls(handler)` capability (or a `@hooks.on.deferred_tool_calls` hook). The handler receives the pending `DeferredToolRequests` and returns `DeferredToolResults` for the calls it decides, or `None`; whatever is left unresolved still ends the run as `DeferredToolRequests`. See [recipe 13](/pydanticai-guide/pydantic_ai_recipes/#recipe-13-inline-hitl-with-handledeferredtoolcalls) for a full example.

## `DeferredLoadingToolset` — load tools on demand

`DeferredLoadingToolset` marks tools with `defer_loading=True` on their `ToolDefinition`, hiding them from the model until the `search_tools` function (or a native provider search) discovers them. This is the recommended way to work with large tool libraries (100+ tools) without overwhelming the context window.

```python
from pydantic_ai import Agent, FunctionToolset, DeferredLoadingToolset
from pydantic_ai.capabilities import ToolSearch

# A large library of 50+ tools
big_library = FunctionToolset[None]()

for i in range(20):
    name = f'operation_{i}'
    desc = f'Performs operation {i} on the dataset.'
    # Register dynamically for this example
    big_library.add_function(
        lambda ctx, i=i: f'result of operation {i}',
        name=name,
        description=desc,
    )

# Hide all tools until tool search surfaces them
hidden = DeferredLoadingToolset(big_library)

# ToolSearch capability adds a search_tools function tool that discovers deferred tools
agent = Agent(
    'openai:%%model=openai%%',
    toolsets=[hidden],
    capabilities=[ToolSearch()],  # enables the search_tools built-in
)
result = agent.run_sync('Run operation 5 on the dataset.')
print(result.output)
```

### Selectively hide only some tools

Pass `tool_names` to `DeferredLoadingToolset` to hide only specific tools; others remain visible:

```python
from pydantic_ai import FunctionToolset, DeferredLoadingToolset

tools = FunctionToolset[None]()

@tools.tool_plain
def get_weather(city: str) -> str:
    """Get current weather."""
    return f'Sunny in {city}'

@tools.tool_plain
def send_alert(message: str) -> str:
    """Send an emergency alert. Rarely needed."""
    return f'Alert sent: {message}'

@tools.tool_plain
def list_sensors() -> list[str]:
    """List active sensors."""
    return ['sensor-1', 'sensor-2']

# Only hide the rarely-used send_alert; get_weather and list_sensors stay visible
partial_deferred = DeferredLoadingToolset(
    tools,
    tool_names=frozenset({'send_alert'}),
)
```

### How deferral works under the hood

`DeferredLoadingToolset` installs a `prepare_func` that calls `ToolDefinition.replace(defer_loading=True)` on the matching tools. The framework then:

1. **Native path** (providers that support it like Anthropic/OpenAI): keeps all deferred tools in the wire payload with a provider-specific `defer_loading=True` flag, so the provider handles server-side discovery.
2. **Local path**: drops deferred tools from the wire until the model calls `search_tools`, which returns the matching tool names. Discovered tools are promoted (set `defer_loading=False`) for subsequent steps.

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
    'openai:%%model=openai%%',
    toolsets=[
        quick_tools,                                    # always visible
        DeferredLoadingToolset(heavy_tools),            # discovered on demand
    ],
)
```

## Toolsets from other sources

### An agent's own toolset

```python
from pydantic_ai import Agent

sub = Agent('openai:%%model=openai-small%%', name='citations')

@sub.tool_plain
def lookup_citation(key: str) -> str: ...

parent = Agent('openai:%%model=openai%%', toolsets=[sub.toolset])
```

Every `Agent` exposes a `.toolset` (an internal `FunctionToolset`) for reuse.

### Dynamic toolsets — `@agent.toolset`

Register a factory that builds a toolset for each run (or, with the default `per_run_step=True`, each step) from the `RunContext`. Returning `None` contributes no tools.

```python
from dataclasses import dataclass

from pydantic_ai import Agent, FunctionToolset, RunContext

@dataclass
class TenantDeps:
    tenant_id: str
    premium: bool

agent = Agent('openai:%%model=openai%%', deps_type=TenantDeps)

premium_tools = FunctionToolset()

@premium_tools.tool_plain
def forecast(city: str) -> str:
    """Seven-day forecast."""
    return f'7-day forecast for {city}'

@agent.toolset
def per_tenant(ctx: RunContext[TenantDeps]) -> FunctionToolset | None:
    return premium_tools if ctx.deps.premium else None

print(agent.run_sync('Forecast for Leeds?', deps=TenantDeps('acme', premium=True)).output)
```

### MCP servers alongside local tools

```python
import asyncio

from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent, FunctionToolset, PrefixedToolset
from pydantic_ai.mcp import MCPToolset

local_tools = FunctionToolset()

@local_tools.tool_plain
def today() -> str:
    """Today's date."""
    return '2026-10-21'

# MCPToolset replaced V1's MCPServerStdio / MCPServerSSE / MCPServerStreamableHTTP.
# It accepts a URL, a script path, a FastMCP server or client, or a transport.
server = MCPToolset(StdioTransport('uv', args=['run', 'mcp-run-python', 'stdio']))
agent = Agent('openai:%%model=openai%%',
              toolsets=[local_tools, PrefixedToolset(server, prefix='mcp')])

async def main():
    async with agent:   # starts the MCP server process, stops it on exit
        result = await agent.run('Run a Python snippet that prints the date.')
        print(result.output)

asyncio.run(main())
```

MCP is covered fully on the [integrations page](/pydanticai-guide/pydantic_ai_integrations/#model-context-protocol-mcp).

## Custom toolsets with `WrapperToolset` and `AbstractToolset`

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

agent = Agent('openai:%%model=openai%%', toolsets=[TimedToolset(wrapped=base_tools)])
```

### `AbstractToolset` — build from scratch

Implement `AbstractToolset` when you need full control over tool definitions and execution — for example, wrapping a database schema or a remote API registry:

```python
import asyncio
from dataclasses import dataclass
from typing import Any
from pydantic_core import SchemaValidator, core_schema
from pydantic_ai import Agent
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

agent = Agent('openai:%%model=openai%%', toolsets=[DatabaseToolset(db_url='postgresql://...')])

async def main():
    async with agent:   # enters the toolset, which introspects the schema
        result = await agent.run('List all users')
        print(result.output)

asyncio.run(main())
```

## A complete example: scoping and approval together

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
    PrefixedToolset(read_tools, prefix='read'),
    PrefixedToolset(gated_write_tools, prefix='write'),
])

# Filter: non-admins only see read tools
def admin_filter(ctx: RunContext[UserDeps], tool_def) -> bool:
    if tool_def.name.startswith('write_') and not ctx.deps.is_admin:
        return False
    return True

agent = Agent(
    'openai:%%model=openai%%',
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

## Gotchas

- **Enter before use**: toolsets may hold resources (processes, HTTP clients, MCP sessions). Using an agent as an async context manager (`async with agent: ...`) enters every toolset.
- **Naming collisions**: two toolsets exposing the same tool name raise `UserError` when the run lists tools. Wrap with `PrefixedToolset` or `RenamedToolset` to disambiguate.
- **`requires_approval=True` without `DeferredToolRequests`** in `output_type` raises at runtime. Always add `DeferredToolRequests` to the output union.
- **`ExternalToolset` + streaming**: external deferrals terminate the stream early. Handle `DeferredToolRequests` as a normal output value.
- **Durable execution**: every toolset must have an `id` when running under Temporal/Prefect/DBOS so activities can be routed.
- **`PreparedToolset` constraint**: the prepare function cannot add or rename tools. Reducing or modifying definitions is fine; use `RenamedToolset` for renaming and `FunctionToolset.add_function()` for additions.
- **`DeferredLoadingToolset` and `ToolSearch`**: when deferred tools exist the agent adds tool search automatically (a local `search_tools` tool, or the provider's native search). Add `ToolSearch(...)` yourself only to choose a strategy or result count.

## Reference

- `AbstractToolset`, `WrapperToolset`, `ToolsetTool` — `pydantic_ai.toolsets.abstract` / `pydantic_ai.toolsets.wrapper`
- `FunctionToolset`, `CombinedToolset`, `PrefixedToolset`, `RenamedToolset`, `FilteredToolset`, `PreparedToolset`, `SetMetadataToolset`, `IncludeReturnSchemasToolset` — `pydantic_ai.toolsets`
- `ApprovalRequiredToolset`, `ExternalToolset` (replaced `DeferredToolset`, removed in 2.0), `DeferredLoadingToolset`, `DynamicToolset` — `pydantic_ai.toolsets`
- `DeferredToolRequests`, `DeferredToolResults`, `ToolApproved`, `ToolDenied` — `pydantic_ai.tools` (also exported from `pydantic_ai`)
- `ApprovalRequired`, `CallDeferred` — `pydantic_ai.exceptions` (also exported from `pydantic_ai`)
- `ToolSearch` capability — `pydantic_ai.capabilities`
- `MCPToolset` — `pydantic_ai.mcp`

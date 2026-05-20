---
title: "PydanticAI: Toolsets"
description: "FunctionToolset, CombinedToolset, FilteredToolset, PrefixedToolset, RenamedToolset, PreparedToolset, ApprovalRequiredToolset, DeferredLoadingToolset, ExternalToolset — compose and reshape tool collections."
framework: pydanticai
language: python
---

# Toolsets

Verified against **pydantic-ai==1.99.0** — source module: `pydantic_ai.toolsets`.

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

### `IncludeReturnSchemasToolset` & `SetMetadataToolset`

- `IncludeReturnSchemasToolset(wrapped)` — forces every tool's return schema into the sent definition. Useful for models that infer structure from return types (some OpenAI / Google configurations).
- `SetMetadataToolset(wrapped, metadata={'team': 'search'})` — bulk-tags tools for filtering later.

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

## Dynamic toolsets — `@agent.toolset`

`agent/__init__.py:2237`. Register a factory that builds a toolset per run based on `RunContext`:

```python
@agent.toolset
async def per_tenant(ctx: RunContext[TenantDeps]) -> AbstractToolset[TenantDeps]:
    return FunctionToolset([load_tools_for(ctx.deps.tenant_id)])
```

## Gotchas

- **Enter before use**: toolsets may hold resources (processes, HTTP clients, MCP sessions). Using an agent as an async context manager (`async with agent: ...`) enters every toolset.
- **Naming collisions**: `CombinedToolset` raises if two toolsets expose the same tool name. Wrap with `PrefixedToolset` or `RenamedToolset` to disambiguate.
- **`requires_approval=True` without `DeferredToolRequests`** in `output_type` raises at runtime. Always add `DeferredToolRequests` to the output union.
- **`ExternalToolset` + streaming**: external deferrals terminate the stream early. Handle `DeferredToolRequests` as a normal output value.
- **Durable execution**: every toolset must have an `id` when running under Temporal/Prefect/DBOS so activities can be routed.

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

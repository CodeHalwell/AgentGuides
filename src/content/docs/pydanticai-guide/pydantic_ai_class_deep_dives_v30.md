---
title: "PydanticAI Class Deep Dives Vol. 30"
description: "Source-verified deep dives into 10 pydantic-ai 2.1.0 class groups: ApprovalRequiredToolset (human-in-the-loop approval gate — approval_required_func default all-true, ctx.tool_call_approved bypass, ApprovalRequired exception), FilteredToolset + RenamedToolset (sync/async filter_func dispatch, name_map NEW→ORIGINAL reverse-map, ctx+tool patching on call), CombinedToolset + _CombinedToolsetTool (parallel gather for get_tools/for_run, name-conflict UserError with toolset labels, source_toolset+source_tool dispatch chain), ExternalToolset (kind='external' ToolDefinition injection, any-schema validator, call_tool NotImplementedError, id= optional discriminator), AbstractToolset + ToolsetTool + WrapperToolset (base protocol fields: toolset/tool_def/max_retries/args_validator, identity-optimized for_run return-self pattern, WrapperToolset delegation with get_instructions propagation), FallbackModel (FallbackOn union — exception types/ExceptionHandler/ResponseHandler/sequence, _is_response_handler first-param ModelResponse detection, FallbackExceptionGroup + ResponseRejected, suppress-on-close asynccontextmanager), Hooks + HookTimeoutError + _HookRegistration (decorator namespace via cached_property on, _registry dict dispatch, bare-vs-parameterized overload pattern, tool-hook frozenset filter, wrap-chain reversed() link building, anyio.fail_after timeout), WebFetchLocalTool + web_fetch_tool (SSRF-safe safe_download, markdownify HTML→markdown, max_content_length char cap, allowed/blocked exact hostname match, WebFetchResult TypedDict), ImageGenerationSubagentTool + image_generation_tool + _IMAGE_ONLY_MODELS (BinaryImage subagent — _check_image_only_model guard maps gpt-image-* / dall-e-* to conversational alternatives, static vs dynamic model resolution, NativeTool wrapping), XSearchSubagentTool + x_search_tool + XSearchFallbackModel (X/Twitter subagent — XSearchFallbackModelFunc RunContext callable, dynamic model resolution, default instructions, fallback_model=None native-only path). All verified against pydantic-ai 2.1.0 source."
sidebar:
  label: "Class deep dives (Vol. 30)"
  order: 56
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.1.0** source installed directly from PyPI. Every class signature, field name, and method reflects the 2.1.x API.
</Aside>

Ten class groups covering the toolset composition layer (approval gating, filtering, renaming, combining, external execution), the toolset base protocol, the `FallbackModel` multi-model router, the `Hooks` decorator capability, and three subagent-backed common tools (web fetch, image generation, X search). These classes sit one layer above or alongside `FunctionToolset` and unlock patterns that are hard to achieve through basic `@agent.tool` registration alone.

---

## 1. `ApprovalRequiredToolset` — Human-in-the-Loop Approval Gate

**Source**: `pydantic_ai/toolsets/approval_required.py`

`ApprovalRequiredToolset` wraps any `AbstractToolset` and pauses execution before each tool call, giving a human (or programmatic guard) the opportunity to approve or block it. The approval decision is made by `approval_required_func`; the default requires approval for every call (the predicate always returns `True`). Approval is communicated back to the framework via `ctx.tool_call_approved` on subsequent retries.

```python
# Key signatures verified from source:

@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):

    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True
    # Default: every call requires approval (all-true predicate).

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        # ctx.tool_call_approved is set by the framework on re-entry after approval.
        if not ctx.tool_call_approved and self.approval_required_func(ctx, tool.tool_def, tool_args):
            raise ApprovalRequired
        return await super().call_tool(name, tool_args, ctx, tool)
```

Key implementation facts:
- `ApprovalRequired` (from `pydantic_ai.exceptions`) is a sentinel exception that adds the pending call to a `DeferredToolRequests` result. The caller approves it by constructing `DeferredToolResults(approvals={'<tool_call_id>': True})` (or `ToolApproved()`/`ToolDenied()` for fine-grained control) and passing it as `deferred_tool_results=` on the next `agent.run(message_history=result.all_messages(), ...)` call, which sets `ctx.tool_call_approved` on re-entry.
- When `approval_required_func` returns `False` the call proceeds immediately — no approval needed for that invocation.
- `ctx.tool_call_approved` is checked *first*; if the call has already been approved, `approval_required_func` is never invoked, ensuring a single approve-then-execute cycle.

### 1.1 Require Approval for All Tools

```python
import asyncio
from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset
from pydantic_ai.toolsets import ApprovalRequiredToolset

async def delete_record(record_id: str) -> str:
    """Delete a record from the database."""
    return f'Deleted record {record_id}'

async def fetch_record(record_id: str) -> dict:
    """Fetch a record from the database."""
    return {'id': record_id, 'status': 'active'}

base_toolset = FunctionToolset([delete_record, fetch_record])

# Wrap with ApprovalRequiredToolset — default requires approval for every call.
# In production, the framework pauses and calls your UI/approval endpoint.
guarded = ApprovalRequiredToolset(wrapped=base_toolset)

# output_type must include DeferredToolRequests: ApprovalRequired raises the same
# deferred-call signal as ExternalToolset, and the runtime raises UserError if
# DeferredToolRequests is not in the output schema when an approval is triggered.
agent = Agent('openai:gpt-4.1', toolsets=[guarded], output_type=DeferredToolRequests | str)
```

### 1.2 Selective Approval via `approval_required_func`

```python
from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset
from pydantic_ai.toolsets import ApprovalRequiredToolset
from pydantic_ai.tools import RunContext, ToolDefinition

# Only destructive tools (name contains 'delete' or 'drop') need approval.
DESTRUCTIVE_KEYWORDS = {'delete', 'drop', 'remove', 'destroy', 'purge'}

def needs_approval(
    ctx: RunContext,
    tool_def: ToolDefinition,
    tool_args: dict,
) -> bool:
    return any(kw in tool_def.name.lower() for kw in DESTRUCTIVE_KEYWORDS)

async def delete_item(item_id: str) -> str:
    return f'Deleted {item_id}'

async def list_items() -> list[str]:
    return ['item-1', 'item-2']

toolset = ApprovalRequiredToolset(
    wrapped=FunctionToolset([delete_item, list_items]),
    approval_required_func=needs_approval,
)
agent = Agent('openai:gpt-4.1', toolsets=[toolset], output_type=DeferredToolRequests | str)
# list_items → no approval needed; delete_item → pauses for approval (DeferredToolRequests result).
```

### 1.3 Context-Aware Approval (Check User Role)

```python
from dataclasses import dataclass
from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset
from pydantic_ai.toolsets import ApprovalRequiredToolset
from pydantic_ai.tools import RunContext, ToolDefinition

@dataclass
class AppDeps:
    user_role: str  # 'admin' | 'viewer' | 'editor'

def role_aware_approval(
    ctx: RunContext[AppDeps],
    tool_def: ToolDefinition,
    tool_args: dict,
) -> bool:
    # Admins can execute anything without extra approval.
    if ctx.deps.user_role == 'admin':
        return False
    # Editors need approval only for write operations.
    if ctx.deps.user_role == 'editor':
        return tool_def.name.startswith(('write_', 'update_', 'delete_'))
    # Viewers always need approval for everything.
    return True

async def write_document(content: str) -> str:
    return f'Written: {content[:40]}...'

async def read_document(doc_id: str) -> str:
    return f'Content of {doc_id}'

toolset = ApprovalRequiredToolset(
    wrapped=FunctionToolset([write_document, read_document]),
    approval_required_func=role_aware_approval,
)
agent = Agent('openai:gpt-4.1', toolsets=[toolset], output_type=DeferredToolRequests | str)
result = agent.run_sync(
    'Read document and update it', deps=AppDeps(user_role='editor')
)
# When editor role triggers write_document, result.output is DeferredToolRequests.
```

---

## 2. `FilteredToolset` + `RenamedToolset` — Dynamic Filter and Name Remapping

**Source**: `pydantic_ai/toolsets/filtered.py` · `pydantic_ai/toolsets/renamed.py`

`FilteredToolset` wraps a toolset and removes tools from the model's view using a predicate evaluated *per run step* — enabling context-dependent tool visibility. `RenamedToolset` remaps tool names using a `dict[new_name, original_name]` mapping, patching the `ToolDefinition` sent to the model and translating back before dispatch.

```python
# FilteredToolset (verified from source):
@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    filter_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition],
        bool | Awaitable[bool],        # sync AND async accepted
    ]

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[AgentDepsT]]:
        result = {}
        for name, tool in (await super().get_tools(ctx)).items():
            match = self.filter_func(ctx, tool.tool_def)
            if inspect.isawaitable(match):   # async filter awaited here
                match = await match
            if match:
                result[name] = tool
        return result

# RenamedToolset (verified from source):
@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    name_map: dict[str, str]   # new_name → original_name

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[AgentDepsT]]:
        # Build reverse map: original_name → new_name
        original_to_new = {v: k for k, v in self.name_map.items()}
        original_tools = await super().get_tools(ctx)
        tools = {}
        for original_name, tool in original_tools.items():
            new_name = original_to_new.get(original_name)
            if new_name:
                tools[new_name] = replace(
                    tool, toolset=self,
                    tool_def=replace(tool.tool_def, name=new_name),
                )
            else:
                tools[original_name] = tool   # unmapped tools pass through unchanged
        return tools

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        original_name = self.name_map.get(name, name)  # fall back to name if not in map
        ctx = replace(ctx, tool_name=original_name)
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)
```

Key implementation facts:
- `FilteredToolset.filter_func` accepts both sync and async predicates; `inspect.isawaitable` determines which path runs.
- Filtering happens in `get_tools()` (called before each model request), so the visible tool set can change mid-conversation.
- `RenamedToolset.name_map` maps *new → original*; the reverse map is rebuilt on each `get_tools()` call — cheap and always consistent.
- Unmapped tools pass through `RenamedToolset` unchanged (neither filtered nor broken).
- Both classes extend `WrapperToolset` so they inherit `for_run`, `for_run_step`, and lifecycle management automatically.

### 2.1 Feature-Flag Tool Filtering

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import FilteredToolset
from pydantic_ai.tools import RunContext, ToolDefinition

@dataclass
class AppDeps:
    enabled_tools: set[str]   # tool names enabled for this user/tenant

def feature_gate(ctx: RunContext[AppDeps], tool_def: ToolDefinition) -> bool:
    return tool_def.name in ctx.deps.enabled_tools

async def advanced_analytics(query: str) -> dict:
    return {'result': f'Analytics for: {query}'}

async def basic_search(query: str) -> list[str]:
    return [f'Result for {query}']

toolset = FilteredToolset(
    wrapped=FunctionToolset([advanced_analytics, basic_search]),
    filter_func=feature_gate,
)
agent = Agent('openai:gpt-4.1', toolsets=[toolset])

result = await agent.run(
    'Search and analyse the data',
    deps=AppDeps(enabled_tools={'basic_search'}),   # advanced_analytics hidden
)
```

### 2.2 Async Filter with Database Lookup

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import FilteredToolset
from pydantic_ai.tools import RunContext, ToolDefinition

async def check_permission(ctx: RunContext, tool_def: ToolDefinition) -> bool:
    """Async predicate — checks permission from a remote store."""
    # e.g. await redis_client.sismember(f'permissions:{ctx.deps.user_id}', tool_def.name)
    allowed = {'read_file', 'list_files'}   # simulate async permission check
    return tool_def.name in allowed

async def read_file(path: str) -> str:
    return f'Contents of {path}'

async def write_file(path: str, content: str) -> str:
    return f'Written to {path}'

async def list_files(directory: str) -> list[str]:
    return ['file1.txt', 'file2.txt']

toolset = FilteredToolset(
    wrapped=FunctionToolset([read_file, write_file, list_files]),
    filter_func=check_permission,   # async filter accepted natively
)
agent = Agent('openai:gpt-4.1', toolsets=[toolset])
```

### 2.3 Renaming Tools for Model Compatibility

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import RenamedToolset

async def python_search__v2(query: str) -> list[str]:
    """Internal snake_case name with version suffix."""
    return [f'Result: {query}']

async def database__query_records(sql: str) -> list[dict]:
    """Internal qualified name."""
    return [{'id': 1, 'value': sql}]

# Models like gpt-4.1 work better with clean, unambiguous tool names.
toolset = RenamedToolset(
    wrapped=FunctionToolset([python_search__v2, database__query_records]),
    name_map={
        'search':          'python_search__v2',     # new_name → original_name
        'query_database':  'database__query_records',
    },
)
agent = Agent('openai:gpt-4.1', toolsets=[toolset])
# Model sees 'search' and 'query_database'; calls are routed to original functions.
```

---

## 3. `CombinedToolset` + `_CombinedToolsetTool` — Merging Multiple Toolsets

**Source**: `pydantic_ai/toolsets/combined.py`

`CombinedToolset` presents multiple toolsets as a single unified toolset. It gathers tools from all wrapped toolsets in parallel and raises `UserError` on name conflicts. Dispatch routes each call back to its original source toolset via the `_CombinedToolsetTool` bridge.

```python
# Key signatures verified from source:

@dataclass(kw_only=True)
class _CombinedToolsetTool(ToolsetTool[AgentDepsT]):
    """Bridge that carries both the tool's owning toolset and the original ToolsetTool."""
    source_toolset: AbstractToolset[AgentDepsT]
    source_tool: ToolsetTool[AgentDepsT]

@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    toolsets: Sequence[AbstractToolset[AgentDepsT]]
    _exit_stack: AsyncExitStack | None = field(init=False, default=None)

    async def for_run(self, ctx):
        new_toolsets = await gather(*(t.for_run(ctx) for t in self.toolsets))
        return replace(self, toolsets=new_toolsets)        # parallel gather

    async def for_run_step(self, ctx):
        new_toolsets = await gather(*(t.for_run_step(ctx) for t in self.toolsets))
        if all(new is old for new, old in zip(new_toolsets, self.toolsets)):
            return self                    # identity optimisation — no allocation if unchanged
        return replace(self, toolsets=new_toolsets)

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[AgentDepsT]]:
        toolsets_tools = await gather(*(toolset.get_tools(ctx) for toolset in self.toolsets))
        all_tools = {}
        for toolset, tools in zip(self.toolsets, toolsets_tools):
            for name, tool in tools.items():
                if existing_tool := all_tools.get(name):
                    # Error message includes both conflicting toolset labels.
                    raise UserError(
                        f'... conflicts with existing tool from {existing_tool.toolset.label}: {name!r}.'
                        f' {toolset.tool_name_conflict_hint}'
                    )
                all_tools[name] = _CombinedToolsetTool(
                    toolset=tool.toolset, tool_def=tool.tool_def,
                    max_retries=tool.max_retries, args_validator=tool.args_validator,
                    args_validator_func=tool.args_validator_func,
                    source_toolset=toolset, source_tool=tool,
                )
        return all_tools

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        assert isinstance(tool, _CombinedToolsetTool)
        return await tool.source_toolset.call_tool(name, tool_args, ctx, tool.source_tool)

    async def get_instructions(self, ctx):
        results = await gather(*(ts.get_instructions(ctx) for ts in self.toolsets))
        # Flattens and merges instruction parts from all wrapped toolsets.
        parts = [p for r in results if r is not None for p in (r if isinstance(r, list) else [r])]
        return parts or None
```

Key implementation facts:
- `gather()` is pydantic-ai's internal parallel async utility — all toolset operations run concurrently, not sequentially.
- Name conflict detection is O(n) per tool; the error message includes the label of the conflicting toolset for easy debugging.
- `_CombinedToolsetTool.source_toolset` carries the *wrapping* toolset (the `CombinedToolset` child), while `source_tool` carries the *original* `ToolsetTool` — `call_tool()` uses both to dispatch correctly.
- `get_instructions()` merges instructions from all children; `None` children are skipped; the result is `None` rather than `[]` when nothing is collected (avoids empty system prompt additions).
- `__aenter__`/`__aexit__` use `AsyncExitStack` to enter and exit all wrapped toolsets in order.

### 3.1 Combine Function and MCP Toolsets

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import CombinedToolset

async def local_search(query: str) -> list[str]:
    """Local in-memory search."""
    return [f'local: {query}']

async def calculate(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

local_tools = FunctionToolset([local_search, calculate])
# MCP is not a toolset class in pydantic-ai 2.x; use the pydantic-ai-mcp package
# or construct an AbstractToolset subclass backed by an MCP session.

combined = CombinedToolset(toolsets=[local_tools])
agent = Agent('openai:gpt-4.1', toolsets=[combined])
```

### 3.2 Stack Toolset Wrappers Around a Combined Set

```python
from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset
from pydantic_ai.toolsets import CombinedToolset, ApprovalRequiredToolset, FilteredToolset

async def send_email(to: str, subject: str, body: str) -> str:
    return f'Email sent to {to}'

async def get_weather(city: str) -> str:
    return f'Weather in {city}: sunny, 22°C'

async def read_calendar() -> list[str]:
    return ['Meeting at 10am', 'Lunch at 12pm']

# Build composable layers:
# 1. Combine unrelated toolsets
combined = CombinedToolset(toolsets=[
    FunctionToolset([send_email]),
    FunctionToolset([get_weather, read_calendar]),
])
# 2. Require approval only for email sends
guarded = ApprovalRequiredToolset(
    wrapped=combined,
    approval_required_func=lambda ctx, td, args: td.name == 'send_email',
)
agent = Agent('openai:gpt-4.1', toolsets=[guarded], output_type=DeferredToolRequests | str)
```

### 3.3 Detect Name Conflicts Early at Startup

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import CombinedToolset
from pydantic_ai.exceptions import UserError

async def search(query: str) -> list[str]:   # name collision!
    return [f'toolset-A: {query}']

async def search_alt(query: str) -> list[str]:
    return [f'toolset-B: {query}']

toolset_a = FunctionToolset([search])
toolset_b = FunctionToolset([search_alt])

# Rename one to avoid conflict before combining.
from pydantic_ai.toolsets import RenamedToolset
toolset_b_renamed = RenamedToolset(
    wrapped=toolset_b,
    name_map={'web_search': 'search_alt'},
)
combined = CombinedToolset(toolsets=[toolset_a, toolset_b_renamed])
agent = Agent('openai:gpt-4.1', toolsets=[combined])
```

---

## 4. `ExternalToolset` — Tools Executed Outside the Agent Run

**Source**: `pydantic_ai/toolsets/external.py`

`ExternalToolset` advertises tools to the model but marks them `kind='external'` in their `ToolDefinition`. The framework yields back a `ToolCallPart` to the caller without trying to execute it — the caller resolves the result externally and feeds it back. This pattern is essential for human-delegated tasks, RPC calls, and browser automation.

```python
# Key signatures verified from source:

TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())
# Single shared any-schema validator — all tool arguments accepted without Pydantic validation.

class ExternalToolset(AbstractToolset[AgentDepsT]):
    tool_defs: list[ToolDefinition]
    _id: str | None

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None):
        self.tool_defs = tool_defs
        self._id = id

    async def get_tools(self, ctx) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=replace(tool_def, kind='external'),  # marks as external
                max_retries=0,
                args_validator=TOOL_SCHEMA_VALIDATOR,          # any-schema
            )
            for tool_def in self.tool_defs
        }

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        raise NotImplementedError('External tools cannot be called directly')
```

Key implementation facts:
- `kind='external'` is set on every `ToolDefinition` at `get_tools()` time. This signals the agent runtime to yield a `ToolCallPart` without dispatching.
- `max_retries=0` — external tools don't retry; the human/remote caller decides whether to re-invoke.
- `TOOL_SCHEMA_VALIDATOR` uses `core_schema.any_schema()` — argument shapes are not validated by pydantic-ai, since the external executor owns validation.
- `id=` is optional; when set it acts as a stable discriminator for serialisation/spec round-trips.

### 4.1 Human-Delegated Tool in a Web App

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition

# Define tool schemas without Python implementations.
approve_payment = ToolDefinition(
    name='approve_payment',
    description='Ask the human user to approve a payment.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'amount': {'type': 'number', 'description': 'Amount in USD'},
            'recipient': {'type': 'string', 'description': 'Payment recipient'},
        },
        'required': ['amount', 'recipient'],
    },
)

external_toolset = ExternalToolset(tool_defs=[approve_payment])
# Include DeferredToolRequests in output_type so the run surfaces external calls
# to the caller rather than stub-skipping them.
from pydantic_ai import DeferredToolRequests
agent = Agent('openai:gpt-4.1', toolsets=[external_toolset],
              output_type=DeferredToolRequests | str)

# When the model calls 'approve_payment', agent.run() returns a DeferredToolRequests.
# The web-app layer shows the user a confirmation dialog, then feeds the result back
# via deferred_tool_results=DeferredToolResults(...) on the next agent.run() call.
```

### 4.2 Fire-and-Forget RPC via External Toolset

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.tools import ToolDefinition

# RPC call dispatched to a remote worker; result injected later.
run_etl = ToolDefinition(
    name='run_etl_pipeline',
    description='Trigger an ETL pipeline job.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'pipeline_id': {'type': 'string'},
            'start_date': {'type': 'string', 'format': 'date'},
        },
        'required': ['pipeline_id'],
    },
)

external_toolset = ExternalToolset(tool_defs=[run_etl], id='etl-external')
from pydantic_ai import DeferredToolRequests
agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[external_toolset],
              output_type=DeferredToolRequests | str)
# When the model calls 'run_etl_pipeline', agent.run() returns DeferredToolRequests.
# The caller dispatches the RPC, collects the result, then resumes via
# deferred_tool_results=DeferredToolResults(...) on the next agent.run() call.
```

### 4.3 Multiple External Tools with Stable ID

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset, CombinedToolset
from pydantic_ai.tools import ToolDefinition

# Local tools alongside externally-resolved tools.
async def search_internal_docs(query: str) -> list[str]:
    return [f'Doc: {query}']

browse_page = ToolDefinition(
    name='browse_web_page',
    description='Use a browser to visit a URL and extract content.',
    parameters_json_schema={
        'type': 'object',
        'properties': {'url': {'type': 'string'}},
        'required': ['url'],
    },
)
click_element = ToolDefinition(
    name='click_element',
    description='Click a DOM element identified by CSS selector.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'selector': {'type': 'string'},
            'frame': {'type': 'integer', 'default': 0},
        },
        'required': ['selector'],
    },
)

browser_toolset = ExternalToolset(
    tool_defs=[browse_page, click_element],
    id='browser-external',  # stable id for spec serialization
)
from pydantic_ai import DeferredToolRequests
agent = Agent(
    'openai:gpt-4.1',
    output_type=DeferredToolRequests | str,
    toolsets=[
        CombinedToolset(toolsets=[
            FunctionToolset([search_internal_docs]),
            browser_toolset,
        ])
    ],
)
```

---

## 5. `AbstractToolset` + `ToolsetTool` + `WrapperToolset` — Toolset Base Protocol

**Source**: `pydantic_ai/toolsets/abstract.py` · `pydantic_ai/toolsets/wrapper.py`

`ToolsetTool` is the per-tool wrapper that carries provenance (`toolset`), the wire schema (`tool_def`), the retry budget (`max_retries`), and the argument validator. `AbstractToolset` is the ABC that all toolsets implement. `WrapperToolset` provides default delegation to a `wrapped` inner toolset, making single-responsibility wrappers (filter, rename, approve) trivially composable.

```python
# ToolsetTool (verified from source):
@dataclass(kw_only=True)
class ToolsetTool(Generic[AgentDepsT]):
    toolset: AbstractToolset[AgentDepsT]
    """The toolset that provided this tool — used in error messages."""
    tool_def: ToolDefinition
    """Wire schema: name, description, JSON Schema parameters."""
    max_retries: int
    """Max retries if the tool call fails validation."""
    args_validator: SchemaValidator | SchemaValidatorProt
    """Pydantic Core JSON validator for tool arguments."""
    args_validator_func: ... | None    # optional functional validator override

# AbstractToolset mandatory interface (verified from source):
class AbstractToolset(ABC, Generic[AgentDepsT]):
    @property
    @abstractmethod
    def id(self) -> str | None: ...                       # stable discriminator for specs

    @abstractmethod
    async def get_tools(self, ctx) -> dict[str, ToolsetTool]: ...    # called before each model request
    @abstractmethod
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...  # called on model tool request

    # Lifecycle hooks (default no-ops in base):
    async def for_run(self, ctx) -> AbstractToolset: return self
    async def for_run_step(self, ctx) -> AbstractToolset: return self
    async def __aenter__(self) -> Self: return self
    async def __aexit__(self, *args) -> bool | None: return None
    async def get_instructions(self, ctx) -> ...: return None

# WrapperToolset (verified from source):
@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    wrapped: AbstractToolset[AgentDepsT]

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({self.wrapped.label})'

    async def for_run(self, ctx):
        new_wrapped = await self.wrapped.for_run(ctx)
        if new_wrapped is self.wrapped:
            return self                # identity optimisation
        return replace(self, wrapped=new_wrapped)

    async def get_instructions(self, ctx):
        return await self.wrapped.get_instructions(ctx)   # explicit delegation

    async def get_tools(self, ctx): return await self.wrapped.get_tools(ctx)
    async def call_tool(self, name, tool_args, ctx, tool): return await self.wrapped.call_tool(...)
```

Key implementation facts:
- `WrapperToolset.for_run()` and `for_run_step()` use identity (`is`) comparison before `replace()` — no allocation when the wrapped toolset doesn't change per-run.
- `WrapperToolset.get_instructions()` explicitly delegates to `wrapped.get_instructions(ctx)` — subclasses override only when they need to inject additional instruction parts.
- `WrapperToolset.label` uses `f'{ClassName}({wrapped.label})'` recursively, producing readable labels like `ApprovalRequiredToolset(FilteredToolset(FunctionToolset))` for error messages.
- `AbstractToolset.__aenter__`/`__aexit__` default to no-ops; `WrapperToolset` delegates to `self.wrapped`.
- `SchemaValidatorProt` is a protocol for `SchemaValidator` or `PluggableSchemaValidator`, accepting both `.validate_json()` and `.validate_python()`.

### 5.1 Custom Toolset from Scratch

```python
import asyncio
from typing import Any
from dataclasses import dataclass
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_core import SchemaValidator, core_schema

@dataclass
class InMemoryToolset(AbstractToolset):
    """Toolset backed by a mutable dict of callables."""
    _tools: dict[str, tuple[ToolDefinition, Any]]

    @property
    def id(self) -> str | None:
        return 'in-memory'

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        validator = SchemaValidator(schema=core_schema.any_schema())
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=3,
                args_validator=validator,
                args_validator_func=None,
            )
            for name, (tool_def, _) in self._tools.items()
        }

    async def call_tool(self, name: str, tool_args: dict, ctx: RunContext, tool: ToolsetTool) -> Any:
        _, func = self._tools[name]
        return await func(**tool_args) if asyncio.iscoroutinefunction(func) else func(**tool_args)
```

### 5.2 Thin WrapperToolset: Add Instructions to Any Toolset

```python
from dataclasses import dataclass
from typing import Any
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.tools import RunContext

@dataclass
class InstructedToolset(WrapperToolset):
    """Injects a fixed instruction string into the agent system prompt."""
    instructions: str

    async def get_instructions(self, ctx: RunContext):
        return self.instructions   # prepended to system prompt

from pydantic_ai import Agent, FunctionToolset

async def get_stock_price(ticker: str) -> float:
    return 189.45

agent = Agent(
    'openai:gpt-4.1',
    toolsets=[
        InstructedToolset(
            wrapped=FunctionToolset([get_stock_price]),
            instructions='Always check the stock price before making recommendations.',
        )
    ],
)
```

### 5.3 Inspect `ToolsetTool` Metadata During Validation

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.tools import RunContext, ToolDefinition
from dataclasses import dataclass, replace
from typing import Any

@dataclass
class LoggingToolset(WrapperToolset):
    """Logs tool name, args, and max_retries before every call."""

    async def call_tool(self, name: str, tool_args: dict, ctx: RunContext, tool: ToolsetTool) -> Any:
        print(f'[{tool.toolset.label}] calling {name!r} | max_retries={tool.max_retries}')
        print(f'  args: {tool_args}')
        result = await super().call_tool(name, tool_args, ctx, tool)
        print(f'  result: {result!r}')
        return result

async def multiply(a: float, b: float) -> float:
    return a * b

agent = Agent(
    'openai:gpt-4.1',
    toolsets=[LoggingToolset(wrapped=FunctionToolset([multiply]))],
)
```

---

## 6. `FallbackModel` — Multi-Model Fallback with Custom Conditions

**Source**: `pydantic_ai/models/fallback.py`

`FallbackModel` sequences a list of models, trying each in turn when the previous one fails or produces an unacceptable response. Fallback conditions are extremely flexible: exception types, callables inspecting the exception, or callables inspecting the `ModelResponse` can all be combined.

```python
# Key type aliases (verified from source):

ExceptionHandler = Callable[[Exception], Awaitable[bool]] | Callable[[Exception], bool]
"""Sync or async callable: receives an exception, returns True to trigger fallback."""

ResponseHandler = Callable[[ModelResponse], Awaitable[bool]] | Callable[[ModelResponse], bool]
"""Sync or async callable: receives the ModelResponse, returns True to trigger fallback."""

FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]
)

# ResponseRejected (verified from source):
class ResponseRejected(Exception):
    """Raised in FallbackExceptionGroup when a response handler rejects the response."""
    def __init__(self, rejected_count: int):
        super().__init__(f'{rejected_count} model response(s) rejected by fallback_on handler')

# _is_response_handler() determines handler type from first-parameter type hint:
def _is_response_handler(handler: Callable) -> bool:
    first_param_type = get_first_param_type(handler)
    if first_param_type is None:
        return False
    return first_param_type is ModelResponse   # exact ModelResponse match only
```

Key implementation facts:
- `FallbackModel` distinguishes `ExceptionHandler` from `ResponseHandler` via the type annotation of the callable's *first parameter*. A handler annotated `(exc: Exception)` triggers on errors; one annotated `(response: ModelResponse)` triggers on content inspection.
- All attempted exceptions are collected into a `FallbackExceptionGroup` so nothing is silently swallowed.
- `ResponseRejected` is added to the group when a response handler returns `True`.
- `asynccontextmanager` + `suppress` is used internally for clean model lifecycle management.
- `FallbackModel` itself implements the full `Model` interface — it can be passed wherever any other model is accepted.

### 6.1 Basic Fallback on Rate-Limit Errors

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

primary = OpenAIModel('gpt-4.1')
backup_a = OpenAIModel('gpt-4.1-mini')
# backup_b could be from a different provider entirely

fallback = FallbackModel(primary, backup_a)
agent = Agent(fallback)

result = await agent.run('Summarise the latest AI research trends.')
print(result.output)
```

### 6.2 Custom Exception Handler — Retry on 5xx Only

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

def is_server_error(exc: Exception) -> bool:
    """Trigger fallback only for 5xx API errors, not 4xx client errors."""
    if isinstance(exc, ModelAPIError):
        status = getattr(exc, 'status_code', None)
        return status is not None and status >= 500
    return False

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

fallback = FallbackModel(
    OpenAIModel('gpt-4.1'),
    AnthropicModel('claude-sonnet-4-6'),
    fallback_on=is_server_error,
)
agent = Agent(fallback)
result = await agent.run('Draft a short report on quantum computing.')
```

### 6.3 Response Handler — Skip Low-Confidence Outputs

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.messages import ModelResponse

def is_low_quality_response(response: ModelResponse) -> bool:
    """Fallback when the model produces an extremely short or empty response."""
    text = ''.join(
        part.content
        for part in response.parts
        if hasattr(part, 'content') and isinstance(part.content, str)
    )
    return len(text.strip()) < 20   # trigger fallback for suspiciously short responses

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

# Chain exception types AND a response handler together in a list.
fallback = FallbackModel(
    OpenAIModel('gpt-4.1-mini'),
    AnthropicModel('claude-haiku-4-5-20251001'),
    fallback_on=[Exception, is_low_quality_response],
)
agent = Agent(fallback)
result = await agent.run('Explain gradient descent in simple terms.')
print(result.output)
```

---

## 7. `Hooks` + `HookTimeoutError` + `_HookRegistration` — Decorator-Based Hook Registration

**Source**: `pydantic_ai/capabilities/hooks.py`

`Hooks` is the ergonomic entry-point for registering lifecycle callbacks without subclassing `AbstractCapability`. Every lifecycle hook exposed by pydantic-ai is accessible via `hooks.on.<hook_name>` (the `_HookRegistration` decorator namespace) or as a constructor kwarg. Timeouts, tool name filters, and multiple handlers per hook are all supported.

```python
# Key signatures (verified from source):

@dataclass(init=False)
class Hooks(AbstractCapability[AgentDepsT]):
    _registry: dict[str, list[_HookEntry[Any]]]

    def __init__(
        self, *,
        # Run lifecycle
        before_run: BeforeRunHookFunc | None = None,
        after_run: AfterRunHookFunc | None = None,
        run: WrapRunHookFunc | None = None,
        run_error: OnRunErrorHookFunc | None = None,
        # Node lifecycle
        before_node_run, after_node_run, node_run, node_run_error,
        # Event stream
        run_event_stream: WrapRunEventStreamHookFunc | None = None,
        event: OnEventHookFunc | None = None,
        # Model request
        before_model_request, after_model_request, model_request, model_request_error,
        # Tool preparation
        prepare_tools, prepare_output_tools,
        # Tool validation
        before_tool_validate, after_tool_validate, tool_validate, tool_validate_error,
        # Tool execution
        before_tool_execute, after_tool_execute, tool_execute, tool_execute_error,
        # Output validation & processing (similar pattern)
        ...
        ordering: CapabilityOrdering | None = None,
        id: str | None = None,
        defer_loading: bool = False,
    ):
        # Constructor kwarg names differ from internal registry keys:
        # 'run' → 'wrap_run',  'run_error' → 'on_run_error',  'event' → '_on_event', etc.
        ...

    @cached_property
    def on(self) -> _HookRegistration[AgentDepsT]:
        return _HookRegistration(self)   # lazy decorator namespace

class HookTimeoutError(TimeoutError):
    def __init__(self, hook_name: str, func_name: str, timeout: float):
        self.hook_name = hook_name
        self.func_name = func_name
        self.timeout = timeout
        super().__init__(f'Hook {hook_name!r} function {func_name!r} timed out after {timeout}s')
```

Key implementation facts:
- Constructor kwarg `run=` maps to internal key `'wrap_run'`; `run_error=` maps to `'on_run_error'`; `event=` maps to `'_on_event'`. The decorator names (`hooks.on.run`) match constructor names, not internal keys.
- Multiple handlers registered for the same hook run sequentially; `before_*` and `after_*` handlers pass results through the chain; `wrap_*` handlers are chained via `reversed(entries)` — the first registered wrapper becomes outermost (middleware-stack ordering).
- Tool hooks (`before_tool_execute`, etc.) accept an optional `tools: Sequence[str]` filter; only calls to listed tools trigger that handler.
- `anyio.fail_after(entry.timeout)` enforces per-handler timeouts; `HookTimeoutError` is raised (a `TimeoutError` subclass) with the hook name and handler function name for clear diagnostics.
- Sync functions are auto-wrapped: `inspect.isawaitable(result)` after calling the function routes through `await` or returns directly.
- `hooks.on` is a `cached_property` — the `_HookRegistration` namespace is created once and shared.

### 7.1 Request/Response Logging via Decorators

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.tools import RunContext
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx: RunContext, request_context: ModelRequestContext):
    msg_count = len(request_context.messages)
    print(f'[request] run_id={ctx.run_id} | {msg_count} messages')
    return request_context  # must return request_context

@hooks.on.after_model_request
async def log_response(ctx: RunContext, *, request_context: ModelRequestContext, response: ModelResponse):
    parts = len(response.parts)
    print(f'[response] {parts} parts received')
    return response  # must return response

agent = Agent('openai:gpt-4.1', capabilities=[hooks])
result = await agent.run('What is 2 + 2?')
```

### 7.2 Per-Tool Execution Guard with Timeout

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import Hooks
from pydantic_ai.tools import RunContext, ToolDefinition

hooks = Hooks()

# Only guard the 'dangerous_operation' tool, with a 5-second timeout.
@hooks.on.before_tool_execute(tools=['dangerous_operation'], timeout=5.0)
async def audit_dangerous_tool(ctx: RunContext, *, call, tool_def: ToolDefinition, args):
    print(f'AUDIT: dangerous_operation called with {args}')
    # Log to audit trail, check rate limits, etc.
    return args  # return (possibly modified) args

@hooks.on.after_tool_execute
async def log_all_tools(ctx: RunContext, *, call, tool_def: ToolDefinition, args, result):
    print(f'Tool {tool_def.name!r} returned: {result!r}')
    return result

async def dangerous_operation(target: str) -> str:
    return f'Executed on {target}'

async def safe_query(question: str) -> str:
    return f'Answer to {question}'

agent = Agent(
    'openai:gpt-4.1',
    toolsets=[FunctionToolset([dangerous_operation, safe_query])],
    capabilities=[hooks],
)
```

### 7.3 Constructor-Style Hooks for Inline Configuration

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.tools import RunContext
from pydantic_ai.run import AgentRunResult

async def track_run_start(ctx: RunContext) -> None:
    print(f'Run started: {ctx.run_id}')

async def track_run_end(ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult:
    print(f'Run ended: output_type={type(result.output).__name__}')
    return result

async def handle_run_error(ctx: RunContext, *, error: BaseException) -> AgentRunResult:
    print(f'Run error: {error!r}')
    raise error  # re-raise; framework records the error

# Constructor kwargs — same names as decorator methods.
hooks = Hooks(
    before_run=track_run_start,
    after_run=track_run_end,
    run_error=handle_run_error,
)
agent = Agent('openai:gpt-4.1', capabilities=[hooks])
result = await agent.run('Tell me a joke.')
```

---

## 8. `WebFetchLocalTool` + `web_fetch_tool` — Local SSRF-Safe URL Fetcher

**Source**: `pydantic_ai/common_tools/web_fetch.py`

`WebFetchLocalTool` is the local (non-native) fallback for URL fetching. It uses `safe_download()` for SSRF protection, `markdownify` for HTML → Markdown conversion, and returns a `WebFetchResult` TypedDict. `web_fetch_tool()` is the factory that wraps it as a `Tool` for use with `WebFetch(local=True)` or standalone.

```python
# Key signatures (verified from source):

class WebFetchResult(TypedDict):
    url: str
    title: str       # empty string if no <title> found
    content: str     # HTML converted to Markdown

@dataclass
class WebFetchLocalTool:
    _: KW_ONLY
    max_content_length: int | None
    """Max character length of returned content. None = no limit."""
    allow_local_urls: bool
    """Whether to allow private/local IP addresses."""
    timeout: int
    """HTTP request timeout in seconds."""
    allowed_domains: list[str] | None = None
    """Exact hostname match allowlist. Raises ModelRetry on violation."""
    blocked_domains: list[str] | None = None
    """Exact hostname match blocklist. Raises ModelRetry on violation."""
    headers: dict[str, str] | None = None

_EXCESSIVE_NEWLINES_RE = re.compile(r'\n{3,}')
# HTML → markdown → collapse 3+ consecutive newlines to 2.
```

Key implementation facts:
- `safe_download()` (from `pydantic_ai._ssrf`) blocks cloud metadata endpoints (AWS/Azure/GCP), private IP ranges, and link-local addresses. SSRF protection is always on unless `allow_local_urls=True`.
- Domain checks use *exact hostname match* — `blocked_domains=['evil.com']` blocks `evil.com` but not `not-evil.com`.
- `ModelRetry` is raised (not `UserError`) for domain violations and network errors — allows the agent to handle or retry gracefully.
- `markdownify(html)` converts the full HTML body; `_EXCESSIVE_NEWLINES_RE.sub('\n\n', content)` collapses excess whitespace.
- Content from binary responses (`BinaryContent`) detected via `_utils.is_text_like_media_type()` — PDFs and images are not processed.
- The factory `web_fetch_tool()` returns a `Tool` wrapping a `WebFetchLocalTool` instance.

### 8.1 Basic Local Web Fetch Tool

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

# WebFetch(local=True) creates a WebFetchLocalTool with defaults under the hood.
agent = Agent(
    'openai:gpt-4.1',
    capabilities=[WebFetch(local=True)],    # requires pydantic-ai-slim[web-fetch]
)
result = await agent.run('Summarise the content at https://pydantic.dev/')
print(result.output)
```

### 8.2 Standalone `web_fetch_tool` with Domain Restrictions

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

# Create a restricted fetcher: only pydantic.dev and docs.pydantic.dev, 30s timeout.
fetch_tool = web_fetch_tool(
    max_content_length=50_000,  # 50k chars max
    allow_local_urls=False,
    timeout=30,
    allowed_domains=['pydantic.dev', 'docs.pydantic.dev'],
)

agent = Agent(
    'openai:gpt-4.1',
    toolsets=[FunctionToolset([fetch_tool])],
)
result = await agent.run('Get the latest release notes from docs.pydantic.dev')
```

### 8.3 Custom Headers for Authenticated Endpoints

```python
import asyncio
import os
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.common_tools.web_fetch import WebFetchLocalTool, web_fetch_tool

# Inject bearer token for internal API endpoints.
auth_fetcher = WebFetchLocalTool(
    max_content_length=100_000,
    allow_local_urls=True,    # allow internal services
    timeout=15,
    allowed_domains=['internal-api.company.com'],
    headers={'Authorization': f'Bearer {os.environ.get("API_TOKEN", "")}'},
)

# Wrap it as a Tool manually.
from pydantic_ai.tools import Tool

auth_tool = Tool(
    auth_fetcher,
    name='fetch_internal_api',
    description='Fetch data from internal company APIs.',
)

agent = Agent(
    'openai:gpt-4.1',
    toolsets=[FunctionToolset([auth_tool])],
)
```

---

## 9. `ImageGenerationSubagentTool` + `image_generation_tool` + `_IMAGE_ONLY_MODELS` — Subagent-Based Image Generation

**Source**: `pydantic_ai/common_tools/image_generation.py`

`ImageGenerationSubagentTool` implements the local fallback for `ImageGeneration(fallback_model=...)`. It spawns a nested `Agent` configured with `NativeTool(ImageGenerationTool(...))` and `output_type=BinaryImage`, then returns the image for the outer agent to use. The `_IMAGE_ONLY_MODELS` guard prevents confusing errors when a dedicated image model is accidentally passed as `fallback_model`.

```python
# Key signatures (verified from source):

ImageGenerationFallbackModelFunc = Callable[
    [RunContext[Any]],
    Awaitable[Model | KnownModelName | str] | Model | KnownModelName | str,
]
ImageGenerationFallbackModel = Model | KnownModelName | str | ImageGenerationFallbackModelFunc | None

_IMAGE_ONLY_MODELS: dict[str, str] = {
    'gpt-image-2': 'openai-responses:gpt-5.5',
    'gpt-image-1.5': 'openai-responses:gpt-5.5',
    'gpt-image-1': 'openai-responses:gpt-5.4',
    'gpt-image-1-mini': 'openai-responses:gpt-5.4',
    'dall-e-3': 'openai-responses:gpt-5.4',
    'dall-e-2': 'openai-responses:gpt-5.4',
    'imagen-3.0-generate-002': 'google:gemini-3-pro-image-preview',
    'imagen-3.0-fast-generate-001': 'google:gemini-3-pro-image-preview',
}

@dataclass(kw_only=True)
class ImageGenerationSubagentTool:
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc
    native_tool: ImageGenerationTool
    instructions: str = 'Generate an image based on the user prompt. Do not ask clarifying questions.'

    async def __call__(self, ctx: RunContext[Any], prompt: str) -> BinaryImage:
        model = self.model
        if callable(model):                  # resolve dynamic model per-run
            result = model(ctx)
            if inspect.isawaitable(result):
                result = await result
            model = result

        if isinstance(model, str) and callable(self.model):
            # Static strings are guarded at factory time; callable-resolved
            # strings must be checked here (at call time).
            _check_image_only_model(model)

        agent = Agent(
            model,
            output_type=BinaryImage,
            capabilities=[NativeTool(self.native_tool)],
            instructions=self.instructions,
        )
        try:
            result = await agent.run(prompt)
        except UnexpectedModelBehavior as e:
            raise ModelRetry(str(e)) from e   # surface as retryable tool error
        return result.output

def image_generation_tool(
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc,
    native_tool: ImageGenerationTool,
    *,
    instructions: str = ...,
) -> Tool[Any]:
    if isinstance(model, str):
        _check_image_only_model(model)         # guard at factory time for static strings
    return Tool[Any](
        ImageGenerationSubagentTool(model=model, native_tool=native_tool, instructions=instructions).__call__,
        name='generate_image',
        description='Generate an image based on the given prompt.',
    )
```

Key implementation facts:
- `_check_image_only_model()` raises `UserError` for known image-only models (gpt-image-*, dall-e-*, imagen-*), directing users to conversational alternatives. Static string models are validated *at factory time*; `ImageGenerationFallbackModelFunc` callables are validated at *call time*.
- `UnexpectedModelBehavior` from the subagent is re-raised as `ModelRetry` — the outer agent can retry the image generation request.
- The subagent uses `NativeTool(self.native_tool)` wrapped in `capabilities` — not a toolset — ensuring the native tool is properly configured for the subagent's model.
- Default tool name is always `'generate_image'`; description is `'Generate an image based on the given prompt.'`.
- `ImageGeneration(fallback_model=...)` internally calls `image_generation_tool(model=..., native_tool=self._resolved_native())`.

### 9.1 Image Generation via `ImageGeneration` Capability (High-Level)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.messages import BinaryImage

# ImageGeneration selects native generation if the model supports it;
# falls back to a subagent if not. 'openai-responses:gpt-5.4' is a
# conversational model that can generate images natively.
# output_type=BinaryImage is required so the agent returns the image
# directly rather than a text description.
agent = Agent(
    'openai:gpt-4.1',
    output_type=BinaryImage,
    capabilities=[
        ImageGeneration(
            fallback_model='openai-responses:gpt-5.4',
            quality='high',
            size='1024x1024',
        )
    ],
)
result = await agent.run('Generate a photorealistic image of a golden retriever on a beach at sunset.')
print(type(result.output))  # <class 'pydantic_ai.messages.BinaryImage'>
```

### 9.2 Direct `image_generation_tool` Usage

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai.native_tools import ImageGenerationTool

# Use image_generation_tool directly when you want to control toolset composition.
img_tool = image_generation_tool(
    model='openai-responses:gpt-5.4',
    native_tool=ImageGenerationTool(
        quality='medium',
        size='1024x1024',
        output_format='webp',
    ),
    instructions='Create a detailed image matching the prompt. No text in the image.',
)

agent = Agent(
    'anthropic:claude-sonnet-4-6',   # this model can't generate images natively
    toolsets=[FunctionToolset([img_tool])],
)
result = await agent.run('Draw a minimalist logo for a tech startup called NeuralPath.')
```

### 9.3 Dynamic Fallback Model Selected Per-Run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.messages import BinaryImage
from pydantic_ai.tools import RunContext
from dataclasses import dataclass

@dataclass
class Deps:
    tier: str  # 'free' | 'pro'

def select_image_model(ctx: RunContext[Deps]) -> str:
    if ctx.deps.tier == 'pro':
        return 'openai-responses:gpt-5.5'   # highest quality
    return 'openai-responses:gpt-5.4'       # standard quality

agent = Agent(
    'openai:gpt-4.1',
    output_type=BinaryImage,               # required to get BinaryImage output
    capabilities=[
        ImageGeneration(
            fallback_model=select_image_model,  # callable: RunContext → model
            quality='auto',
        )
    ],
)

result_pro = await agent.run('A futuristic cityscape', deps=Deps(tier='pro'))
result_free = await agent.run('A futuristic cityscape', deps=Deps(tier='free'))
```

---

## 10. `XSearchSubagentTool` + `x_search_tool` + `XSearchFallbackModel` — X/Twitter Search via Subagent

**Source**: `pydantic_ai/common_tools/x_search.py`

`XSearchSubagentTool` implements the local fallback for X/Twitter search when the outer agent's model doesn't support `XSearchTool` natively. It spawns a subagent configured with `NativeTool(XSearchTool(...))` and returns a text summary. The `XSearchFallbackModel` type alias supports static strings, model instances, and per-run factory callables.

```python
# Key signatures (verified from source):

XSearchFallbackModelFunc = Callable[
    [RunContext[Any]],
    Awaitable[Model | KnownModelName | str] | Model | KnownModelName | str,
]
"""Callable that resolves a fallback model dynamically per-run."""

XSearchFallbackModel = Model | KnownModelName | str | XSearchFallbackModelFunc | None

@dataclass(kw_only=True)
class XSearchSubagentTool:
    model: Model | KnownModelName | str | XSearchFallbackModelFunc
    native_tool: XSearchTool
    instructions: str = (
        'Search X/Twitter based on the user query. '
        'Return a comprehensive summary of the results.'
    )

    async def __call__(self, ctx: RunContext[Any], query: str) -> str:
        model = self.model
        if callable(model):             # resolve dynamic model per-run
            result = model(ctx)
            if inspect.isawaitable(result):
                result = await result
            model = result

        agent = Agent(
            model,
            capabilities=[NativeTool(self.native_tool)],
            instructions=self.instructions,
        )
        try:
            result = await agent.run(query)
        except UnexpectedModelBehavior as e:
            raise ModelRetry(str(e)) from e
        return result.output

def x_search_tool(
    model: Model | KnownModelName | str | XSearchFallbackModelFunc,
    native_tool: XSearchTool,          # required — no default
    *,
    instructions: str = ...,
) -> Tool[Any]:
    ...
```

Key implementation facts:
- `XSearchSubagentTool` uses `NativeTool(self.native_tool)` in `capabilities=` — not `toolsets=` — just like `ImageGenerationSubagentTool`.
- `UnexpectedModelBehavior` from the subagent is re-raised as `ModelRetry`, allowing the outer agent to handle X search failures gracefully.
- Both sync and async `XSearchFallbackModelFunc` are supported via `inspect.isawaitable`.
- Unlike `ImageGenerationSubagentTool`, there is no `_XSEARCH_ONLY_MODELS` guard — X search is always performed via conversational models.
- `x_search_tool()` requires both `model` and `native_tool` — neither has a default. Accepting `None` for the fallback model is a feature of the higher-level `XSearch` capability, not the factory function.

### 10.1 X Search via the `XSearch` Capability (High-Level)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

# XSearch capability uses native X search when the model supports it
# (e.g. xai:grok-*), otherwise falls back to x_search_tool with a subagent.
agent = Agent(
    'openai:gpt-4.1',
    capabilities=[
        XSearch(
            fallback_model='xai:grok-4-1-fast-non-reasoning',
        )
    ],
)

result = await agent.run(
    'Find the latest discussions about pydantic-ai on X/Twitter from the past 24 hours.'
)
print(result.output)
```

### 10.2 Direct `XSearchSubagentTool` with Custom Instructions

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.common_tools.x_search import XSearchSubagentTool, x_search_tool
from pydantic_ai.native_tools import XSearchTool
from pydantic_ai.tools import Tool

# Fine-tune the subagent's instructions for domain-specific searches.
subagent_tool = XSearchSubagentTool(
    model='xai:grok-4-1-fast-non-reasoning',
    native_tool=XSearchTool(allowed_x_handles=['pydantic_ai', 'tiangolo', 'samuel_colvin']),
    instructions=(
        'Search X/Twitter for the given query. '
        'Focus on posts from verified accounts and developers. '
        'Return a structured summary with top insights and links.'
    ),
)

tool = Tool(subagent_tool, name='search_twitter', description='Search X/Twitter for developer discussions.')
agent = Agent('openai:gpt-4.1', toolsets=[FunctionToolset([tool])])

result = await agent.run('What are developers saying about LangChain vs pydantic-ai?')
print(result.output)
```

### 10.3 Dynamic Model Selection for X Search

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch
from pydantic_ai.tools import RunContext

@dataclass
class RequestDeps:
    use_fast_model: bool

def choose_xsearch_model(ctx: RunContext[RequestDeps]) -> str:
    if ctx.deps.use_fast_model:
        return 'xai:grok-4-1-fast-non-reasoning'
    return 'xai:grok-4-1'

agent = Agent(
    'openai:gpt-4.1',
    capabilities=[
        XSearch(fallback_model=choose_xsearch_model)
    ],
)

fast_result = await agent.run(
    'Trending AI topics on X right now',
    deps=RequestDeps(use_fast_model=True),
)
detailed_result = await agent.run(
    'In-depth opinions on agent frameworks on X this week',
    deps=RequestDeps(use_fast_model=False),
)
```

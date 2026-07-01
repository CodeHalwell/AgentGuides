---
title: "PydanticAI Class Deep Dives Vol. 28"
description: "Source-verified deep dives into 10 pydantic-ai 2.2.0 class groups: FallbackModel (multi-provider failover — exception/response handler auto-detection, ResponseRejected, FallbackExceptionGroup, anyio.Lock cached_property, _exception_handlers + _response_handlers split, per-model prepare_messages defer), FilteredToolset (per-request sync/async predicate filtering — filter_func signature, get_tools override, inspect.isawaitable), ApprovalRequiredToolset (human-in-the-loop gating — approval_required_func triple signature, ctx.tool_call_approved flag, ApprovalRequired exception), RenamedToolset (map-based renaming — name_map new→original, dataclass.replace() immutability, ctx.tool_name restoration in call_tool), PrefixedToolset (namespace prefix — {prefix}_{name} pattern, tool_name_conflict_hint, removeprefix() stripping), PreparedToolset (per-request ToolDefinition transformation — ToolsPrepareFunc, add/rename guard, check_tools_prepare_func_result), CombinedToolset (multi-toolset merge — _CombinedToolsetTool source tracking, UserError on name collision, for_run_step short-circuit, get_instructions() fan-out), ExternalToolset (out-of-band tool results — kind='external', id for deferred capability, call_tool raises NotImplementedError), Embedder + EmbeddingModel + EmbeddingResult (full embedding pipeline — embed_query/embed_documents/embed, EmbedInputType literal, EmbeddingResult.__getitem__ by string, cost() via genai-prices, EmbeddingSettings: dimensions/truncate/extra_headers, instrument_all class method), FunctionToolset (rich function toolset — timeout float retry prompt, sequential barrier flag, requires_approval, defer_loading, include_return_schema, instructions/system-prompt support, id for capability registry). All verified against pydantic-ai 2.2.0 source."
sidebar:
  label: "Class deep dives (Vol. 28)"
  order: 54
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.2.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.2.x API.
</Aside>

Ten class groups covering the toolset composition layer (five wrapper toolset classes), multi-provider failover, out-of-band tool execution, the full text-embedding pipeline, and the richly configurable `FunctionToolset`. These classes unlock patterns — graceful degradation, dynamic tool filtering, namespaced MCP bridges, semantic search — that the higher-level `Agent` API doesn't expose directly.

---

## 1. `FallbackModel` — Multi-Provider Failover with Handler Auto-Detection

**Source**: `pydantic_ai/models/fallback.py`

`FallbackModel` wraps two or more `Model` instances and tries them in sequence until one succeeds. New in 2.2.0, it gains a **response handler** pathway: alongside catching exceptions you can now inspect the `ModelResponse` itself and decide to fall through. Handler type is auto-detected by inspecting the first parameter's type annotation — if it is `ModelResponse` the callable is a response handler; otherwise it is an exception handler.

```python
# Key signatures verified from source (pydantic-ai 2.2.0):

ExceptionHandler = Callable[[Exception], Awaitable[bool]] | Callable[[Exception], bool]
ResponseHandler  = Callable[[ModelResponse], Awaitable[bool]] | Callable[[ModelResponse], bool]

FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]
)

class ResponseRejected(Exception):
    """Raised inside a FallbackExceptionGroup when a response handler rejects a response."""
    def __init__(self, rejected_count: int): ...

@dataclass(init=False)
class FallbackModel(Model):
    models: list[Model]
    _exception_handlers: list[ExceptionHandler]   # populated from fallback_on
    _response_handlers: list[ResponseHandler]      # populated from fallback_on

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: FallbackOn = (ModelAPIError,),
    ): ...

    @cached_property
    def _enter_lock(self) -> anyio.Lock:
        # Deferred to bind to the running event loop, not the import-time loop.
        # Avoids issues with Temporal's workflow sandbox.
        ...

    @property
    def model_name(self) -> str:
        return f'fallback:{",".join(model.model_name for model in self.models)}'

    def prepare_messages(self, messages):
        # Returns messages unchanged — per-model prepare_messages is deferred
        # to each inner model's request() call so the right profile gates transformations.
        return messages
```

### 1.1 Basic Three-Provider Chain

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

# Try GPT-4o first, fall back to Claude Sonnet, then Gemini Flash.
# Default fallback_on=(ModelAPIError,) catches network errors, 5xx, rate limits.
model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-sonnet-4-5',
    'google-gla:gemini-2.0-flash',
)

agent = Agent(model, system_prompt='You are a concise assistant.')


async def main():
    result = await agent.run('Summarise quantum entanglement in one sentence.')
    print(result.output)
    # The model_name reveals which providers were configured:
    print(result.model_name)
    #> fallback:openai:gpt-4o,anthropic:claude-sonnet-4-5,...


asyncio.run(main())
```

### 1.2 Custom Exception + Response Handler Mix

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError


def on_rate_limit(exc: Exception) -> bool:
    """Fall through on rate-limit (429) or server errors (5xx)."""
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, ModelAPIError)


def response_too_short(response: ModelResponse) -> bool:
    """Reject responses that contain fewer than 10 characters — likely a refusal."""
    text = response.text or ''
    return len(text.strip()) < 10


model = FallbackModel(
    'openai:gpt-4o-mini',
    'anthropic:claude-haiku-4-5',
    # Pass a sequence to mix exception handlers and response handlers:
    fallback_on=[on_rate_limit, response_too_short],
)

agent = Agent(model)


async def main():
    result = await agent.run('What is 2 + 2?')
    print(result.output)  #> 4


asyncio.run(main())
```

### 1.3 Async Response Handler + FallbackExceptionGroup Inspection

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError


async def content_policy_rejected(response: ModelResponse) -> bool:
    """Async handler — falls through if the model returned an empty response."""
    return not (response.text or '').strip()


model = FallbackModel(
    'openai:gpt-4o',
    'anthropic:claude-sonnet-4-5',
    fallback_on=[ModelAPIError, content_policy_rejected],
)

agent = Agent(model)


async def main():
    try:
        result = await agent.run('Generate something creative.')
        print(result.output)
    except FallbackExceptionGroup as eg:
        print(f'All {len(eg.exceptions)} models failed:')
        for exc in eg.exceptions:
            print(f'  {type(exc).__name__}: {exc}')


asyncio.run(main())
```

---

## 2. `FilteredToolset` — Per-Request Sync/Async Tool Predicate

**Source**: `pydantic_ai/toolsets/filtered.py`

`FilteredToolset` wraps any `AbstractToolset` and calls a predicate on every tool before each agent step. The predicate receives the live `RunContext` (so it can inspect dependencies, usage, run metadata) and the `ToolDefinition`. Both synchronous and asynchronous predicates are accepted — the class uses `inspect.isawaitable()` to decide whether to `await` the result.

```python
# Key signatures verified from source:

@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool | Awaitable[bool]]

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in (await super().get_tools(ctx)).items():
            match = self.filter_func(ctx, tool.tool_def)
            if inspect.isawaitable(match):
                match = await match
            if match:
                result[name] = tool
        return result
```

### 2.1 Role-Based Tool Visibility

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import RunContext


@dataclass
class UserDeps:
    role: str  # 'admin' | 'viewer'


toolset = FunctionToolset[UserDeps]()


@toolset.tool_plain
def list_users() -> list[str]:
    """List all registered users."""
    return ['alice', 'bob', 'carol']


@toolset.tool_plain
def delete_user(username: str) -> str:
    """Permanently delete a user account."""
    return f'Deleted {username}'


# Only expose destructive tools to admins.
def role_filter(ctx: RunContext[UserDeps], tool_def: ToolDefinition) -> bool:
    if tool_def.name == 'delete_user':
        return ctx.deps.role == 'admin'
    return True


filtered = FilteredToolset(toolset=toolset, filter_func=role_filter)

agent = Agent('test', toolsets=[filtered], system_prompt='Help the user.')


async def main():
    # Viewer sees only list_users
    result = await agent.run('What can you do?', deps=UserDeps(role='viewer'))
    print(result.output)

    # Admin sees both tools
    result = await agent.run('Delete bob', deps=UserDeps(role='admin'))
    print(result.output)


asyncio.run(main())
```

### 2.2 Async Predicate Checking a Database

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import RunContext


@dataclass
class AppDeps:
    user_id: str
    db: object  # your DB connection


async def permission_filter(ctx: RunContext[AppDeps], tool_def: ToolDefinition) -> bool:
    """Async filter — checks a permissions table in the database."""
    # Replace with a real DB query, e.g. ctx.deps.db.fetch(...)
    allowed_tools = {'search', 'summarise'}  # simplified
    return tool_def.name in allowed_tools


toolset = FunctionToolset[AppDeps]()


@toolset.tool_plain
async def search(query: str) -> str:
    """Search the knowledge base."""
    return f'Results for: {query}'


@toolset.tool_plain
async def delete_record(record_id: str) -> str:
    """Delete a database record."""
    return f'Deleted {record_id}'


filtered = FilteredToolset(toolset=toolset, filter_func=permission_filter)
agent = Agent('test', toolsets=[filtered])
```

### 2.3 Usage-Based Dynamic Filtering

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import RunContext

toolset = FunctionToolset()


@toolset.tool_plain
def cheap_lookup(q: str) -> str:
    """Fast, cheap knowledge lookup."""
    return f'Quick answer: {q}'


@toolset.tool_plain
def expensive_api_call(q: str) -> str:
    """Calls a paid third-party API."""
    return f'Expensive answer: {q}'


def budget_filter(ctx: RunContext[None], tool_def: ToolDefinition) -> bool:
    """Block expensive tools once the run has consumed >10k tokens."""
    total_tokens = ctx.usage.total_tokens or 0
    if tool_def.name == 'expensive_api_call' and total_tokens > 10_000:
        return False
    return True


filtered = FilteredToolset(toolset=toolset, filter_func=budget_filter)
agent = Agent('test', toolsets=[filtered])
```

---

## 3. `ApprovalRequiredToolset` — Human-in-the-Loop Tool Gating

**Source**: `pydantic_ai/toolsets/approval_required.py`

`ApprovalRequiredToolset` intercepts `call_tool()` and raises `ApprovalRequired` when the `approval_required_func` returns `True` and `ctx.tool_call_approved` is `False`. This suspends the agent run; the calling code can then show the user the pending call, collect approval, and resume with `tool_call_approved=True`.

```python
# Key signatures verified from source:

@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    # Default: every call requires approval
    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = (lambda ctx, tool_def, tool_args: True)

    async def call_tool(self, name, tool_args, ctx, tool):
        if not ctx.tool_call_approved and self.approval_required_func(ctx, tool.tool_def, tool_args):
            raise ApprovalRequired   # <-- suspends the run
        return await super().call_tool(name, tool_args, ctx, tool)
```

### 3.1 Require Approval for All Tool Calls

```python
import asyncio
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

toolset = FunctionToolset()


@toolset.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f'Email sent to {to}'


@toolset.tool_plain
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path) as f:
        return f.read()


# All tools require human approval before execution.
# output_type=[str, DeferredToolRequests] tells the agent to surface pending
# approval requests as structured output rather than raising an exception.
gated = ApprovalRequiredToolset(toolset=toolset)
agent = Agent('test', toolsets=[gated], output_type=[str, DeferredToolRequests])


async def main():
    result = await agent.run('Send a welcome email to alice@example.com')
    if isinstance(result.output, DeferredToolRequests):
        # Show the pending approval requests to the user
        for call in result.output.approvals:
            print(f'Approval required: {call.tool_name}({call.args_as_dict()})')
        # User approves — build results and resume the run
        tool_results = result.output.build_results(approve_all=True)
        final = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=tool_results,
        )
        print(final.output)
    else:
        print(result.output)


asyncio.run(main())
```

### 3.2 Selective Approval — Only Destructive Tools

```python
import asyncio
from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.tools import ToolDefinition

DESTRUCTIVE_TOOLS = {'delete_file', 'send_email', 'execute_sql'}

toolset = FunctionToolset()


@toolset.tool_plain
def search_docs(query: str) -> list[str]:
    """Search documentation."""
    return [f'Doc about {query}']


@toolset.tool_plain
def delete_file(path: str) -> str:
    """Delete a file from disk."""
    return f'Deleted {path}'


def requires_approval(
    ctx: RunContext[None], tool_def: ToolDefinition, tool_args: dict
) -> bool:
    return tool_def.name in DESTRUCTIVE_TOOLS


gated = ApprovalRequiredToolset(toolset=toolset, approval_required_func=requires_approval)
# output_type=[str, DeferredToolRequests]: non-destructive tools run immediately;
# destructive tools are surfaced as pending approvals instead of executing.
agent = Agent('test', toolsets=[gated], output_type=[str, DeferredToolRequests])
```

### 3.3 AG-UI Human-in-the-Loop Pattern

```python
# When using the AG-UI adapter with output_type=[str, DeferredToolRequests], approval
# requests are surfaced to the frontend as interrupt events; the user clicks
# Approve/Deny in the UI which re-submits the run with deferred_tool_results.
import asyncio
from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.tools import ToolDefinition


def high_cost_guard(ctx: RunContext[None], tool_def: ToolDefinition, args: dict) -> bool:
    """Only gate tools that are expensive or irreversible."""
    costly = {'charge_card', 'send_bulk_email', 'run_migration'}
    return tool_def.name in costly


toolset = FunctionToolset()


@toolset.tool_plain
async def charge_card(amount_usd: float, card_token: str) -> str:
    """Charge a payment card."""
    return f'Charged ${amount_usd}'


gated = ApprovalRequiredToolset(toolset=toolset, approval_required_func=high_cost_guard)

# output_type=[str, DeferredToolRequests] enables the AG-UI adapter to forward
# approval requests to the frontend as structured interrupt events.
agent = Agent('openai:gpt-4o', toolsets=[gated], output_type=[str, DeferredToolRequests])
```

---

## 4. `RenamedToolset` — Map-Based Tool Renaming

**Source**: `pydantic_ai/toolsets/renamed.py`

`RenamedToolset` accepts a `name_map` dict of `{new_name: original_name}` pairs. Tools listed in the map are exposed under their new names; unlisted tools pass through unchanged. During `call_tool()`, the class inverts the map, restores `ctx.tool_name` to the original, and delegates to the wrapped toolset — so decorators, retry logic, and telemetry all see the original names.

```python
# Key signatures verified from source:

@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    name_map: dict[str, str]   # {new_name: original_name}

    async def get_tools(self, ctx):
        original_to_new = {v: k for k, v in self.name_map.items()}
        tools: dict[str, ToolsetTool] = {}
        for original_name, tool in (await super().get_tools(ctx)).items():
            if new_name := original_to_new.get(original_name):
                # Uses dataclass.replace() — immutable update pattern
                tools[new_name] = replace(tool, tool_def=replace(tool.tool_def, name=new_name))
            else:
                tools[original_name] = tool
        return tools

    async def call_tool(self, name, tool_args, ctx, tool):
        original_name = self.name_map.get(name, name)   # look up original
        ctx = replace(ctx, tool_name=original_name)      # restore for callbacks
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)
```

### 4.1 Rename MCP Server Tools for a Specific Model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets.renamed import RenamedToolset

# Some MCP servers expose tools with names that clash with model restrictions.
# Rename them before they reach the model.
mcp_server = MCPServerStdio('python', ['-m', 'my_mcp_server'])

# The MCP server exposes 'get' and 'set' — too short for some models.
renamed = RenamedToolset(
    toolset=mcp_server,
    name_map={
        'kv_get': 'get',   # new_name: original_name
        'kv_set': 'set',
    },
)

agent = Agent('openai:gpt-4o', toolsets=[renamed])
```

### 4.2 Adapting a Third-Party Toolset to Your Naming Convention

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.renamed import RenamedToolset

# Third-party toolset uses camelCase names
vendor_toolset = FunctionToolset()


@vendor_toolset.tool_plain
def getUserProfile(user_id: str) -> dict:
    """Get a user profile by ID."""
    return {'id': user_id, 'name': 'Alice'}


@vendor_toolset.tool_plain
def updateUserEmail(user_id: str, email: str) -> bool:
    """Update a user's email address."""
    return True


# Expose snake_case names to the model — original names preserved internally
renamed = RenamedToolset(
    toolset=vendor_toolset,
    name_map={
        'get_user_profile': 'getUserProfile',
        'update_user_email': 'updateUserEmail',
    },
)

agent = Agent('openai:gpt-4o', toolsets=[renamed])
```

### 4.3 Partial Rename — Only Specific Tools

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.renamed import RenamedToolset

toolset = FunctionToolset()


@toolset.tool_plain
def search(query: str) -> list[str]:
    """Search the knowledge base."""
    return [f'Result for {query}']


@toolset.tool_plain
def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    return 'content...'


@toolset.tool_plain
def summarise(text: str) -> str:
    """Summarise a piece of text."""
    return 'summary...'


# Only rename 'fetch_url'; 'search' and 'summarise' pass through unchanged.
renamed = RenamedToolset(
    toolset=toolset,
    name_map={'web_fetch': 'fetch_url'},
)

agent = Agent('openai:gpt-4o', toolsets=[renamed])
```

---

## 5. `PrefixedToolset` — Namespace Tools with a Prefix

**Source**: `pydantic_ai/toolsets/prefixed.py`

`PrefixedToolset` prepends `{prefix}_` to every tool name. This is the recommended pattern for combining multiple toolsets that might have overlapping names — e.g., two different search providers both exposing a `search` tool. The `tool_name_conflict_hint` property provides a user-friendly error message pointing at the `prefix` attribute when a conflict still occurs.

```python
# Key signatures verified from source:

@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    prefix: str

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Change the `prefix` attribute to avoid name conflicts.'

    async def get_tools(self, ctx):
        return {
            f'{self.prefix}_{name}': replace(tool, tool_def=replace(tool.tool_def, name=f'{self.prefix}_{name}'))
            for name, tool in (await super().get_tools(ctx)).items()
        }

    async def call_tool(self, name, tool_args, ctx, tool):
        original_name = name.removeprefix(self.prefix + '_')  # strips prefix
        ctx = replace(ctx, tool_name=original_name)
        tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
        return await super().call_tool(original_name, tool_args, ctx, tool)
```

### 5.1 Namespace Two Search Providers

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.toolsets.combined import CombinedToolset

# Provider A
toolset_a = FunctionToolset()


@toolset_a.tool_plain
def search(query: str) -> list[str]:
    """Search using provider A."""
    return [f'A: {query}']


# Provider B
toolset_b = FunctionToolset()


@toolset_b.tool_plain
def search(query: str) -> list[str]:  # same name!
    """Search using provider B."""
    return [f'B: {query}']


# Give each its own namespace to avoid the collision
prefixed_a = PrefixedToolset(toolset=toolset_a, prefix='google')
prefixed_b = PrefixedToolset(toolset=toolset_b, prefix='bing')

# Combine — tools are now 'google_search' and 'bing_search'
combined = CombinedToolset(toolsets=[prefixed_a, prefixed_b])
agent = Agent('openai:gpt-4o', toolsets=[combined])
```

### 5.2 Prefix-Isolating Multiple MCP Servers

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.toolsets.combined import CombinedToolset

# Two MCP servers may expose tools with the same names.
filesystem = MCPServerStdio('python', ['-m', 'mcp_filesystem'])
database = MCPServerStdio('python', ['-m', 'mcp_database'])

agent = Agent(
    'openai:gpt-4o',
    toolsets=[
        CombinedToolset(toolsets=[
            PrefixedToolset(toolset=filesystem, prefix='fs'),
            PrefixedToolset(toolset=database, prefix='db'),
        ])
    ],
)
# Model now sees 'fs_read', 'fs_write', 'db_query', 'db_insert', etc.
```

### 5.3 Versioned Tool Rollout

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.toolsets.combined import CombinedToolset

v1_tools = FunctionToolset()
v2_tools = FunctionToolset()


@v1_tools.tool_plain
def analyse(text: str) -> str:
    """Analyse text (v1 — keyword based)."""
    return f'v1 analysis of: {text}'


@v2_tools.tool_plain
def analyse(text: str) -> str:
    """Analyse text (v2 — ML powered)."""
    return f'v2 analysis of: {text}'


combined = CombinedToolset(toolsets=[
    PrefixedToolset(toolset=v1_tools, prefix='v1'),
    PrefixedToolset(toolset=v2_tools, prefix='v2'),
])

agent = Agent('openai:gpt-4o', toolsets=[combined],
              system_prompt='Prefer v2 tools unless asked otherwise.')
```

---

## 6. `PreparedToolset` — Per-Request Tool Definition Transformation

**Source**: `pydantic_ai/toolsets/prepared.py`

`PreparedToolset` calls a `ToolsPrepareFunc` on each agent step, passing the live `RunContext` and the full list of `ToolDefinition` objects. The function can return a filtered or modified subset. It **cannot** add new tools or change tool names — attempting to do so raises `UserError`. Both sync and async prepare functions are supported via `inspect.isawaitable()`.

```python
# Key signatures verified from source:

@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]
    # ToolsPrepareFunc = Callable[
    #     [RunContext[AgentDepsT], list[ToolDefinition]],
    #     list[ToolDefinition] | Awaitable[list[ToolDefinition]] | None
    # ]

    async def get_tools(self, ctx):
        original_tools = await super().get_tools(ctx)
        original_tool_defs = [tool.tool_def for tool in original_tools.values()]
        result = self.prepare_func(ctx, original_tool_defs)
        if inspect.isawaitable(result):
            result = await result
        # Validate: cannot add or rename (raises UserError if new names found)
        prepared_by_name = {d.name: d for d in check_tools_prepare_func_result(result, self.prepare_func)}
        if len(prepared_by_name.keys() - original_tools.keys()) > 0:
            raise UserError('Prepare function cannot add or rename tools ...')
        return {
            name: replace(original_tools[name], tool_def=tool_def)
            for name, tool_def in prepared_by_name.items()
        }
```

### 6.1 Hide Low-Priority Tools to Save Tokens

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.prepared import PreparedToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import RunContext

toolset = FunctionToolset()


@toolset.tool_plain
def quick_answer(q: str) -> str:
    """Fast local lookup."""
    return f'quick: {q}'


@toolset.tool_plain
def deep_research(topic: str) -> str:
    """Thorough research — slow and expensive."""
    return f'deep: {topic}'


@toolset.tool_plain
def translate(text: str, lang: str) -> str:
    """Translate text to another language."""
    return f'{text} in {lang}'


def hide_expensive_on_first_step(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Only expose deep_research after at least one prior tool call."""
    if not ctx.messages:
        # First step — keep only cheap tools
        return [d for d in tool_defs if d.name != 'deep_research']
    return tool_defs


prepared = PreparedToolset(toolset=toolset, prepare_func=hide_expensive_on_first_step)
agent = Agent('test', toolsets=[prepared])
```

### 6.2 Inject Runtime Context into Tool Descriptions

```python
import asyncio
from dataclasses import dataclass, replace as dc_replace
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.prepared import PreparedToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import RunContext


@dataclass
class SessionDeps:
    language: str
    region: str


toolset = FunctionToolset[SessionDeps]()


@toolset.tool_plain
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f'Weather in {city}: sunny'


def localise_descriptions(
    ctx: RunContext[SessionDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Append locale info to every tool description."""
    suffix = f' [Respond in {ctx.deps.language} for {ctx.deps.region}]'
    return [
        dc_replace(d, description=(d.description or '') + suffix)
        for d in tool_defs
    ]


prepared = PreparedToolset(toolset=toolset, prepare_func=localise_descriptions)
agent = Agent('openai:gpt-4o', toolsets=[prepared])
```

### 6.3 Async Preparation with Feature Flag Check

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.prepared import PreparedToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import RunContext

toolset = FunctionToolset()


@toolset.tool_plain
def beta_feature(input: str) -> str:
    """Access a beta feature."""
    return f'beta: {input}'


@toolset.tool_plain
def stable_feature(input: str) -> str:
    """Access a stable feature."""
    return f'stable: {input}'


async def feature_flag_filter(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Async — checks a feature flag service before each step."""
    # In production: await feature_flags.is_enabled('beta_feature', user_id=...)
    beta_enabled = False  # simplified
    if not beta_enabled:
        return [d for d in tool_defs if not d.name.startswith('beta_')]
    return tool_defs


prepared = PreparedToolset(toolset=toolset, prepare_func=feature_flag_filter)
agent = Agent('openai:gpt-4o', toolsets=[prepared])
```

---

## 7. `CombinedToolset` — Merge Multiple Toolsets with Conflict Detection

**Source**: `pydantic_ai/toolsets/combined.py`

`CombinedToolset` fans out `get_tools()`, `get_instructions()`, `for_run()`, and `for_run_step()` across all child toolsets in parallel (via `gather()`). It detects name collisions eagerly, raising a `UserError` that names both conflicting toolsets and suggests using `PrefixedToolset`. It manages child lifecycle via `AsyncExitStack`.

```python
# Key signatures verified from source:

@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    toolsets: Sequence[AbstractToolset[AgentDepsT]]
    _exit_stack: AsyncExitStack | None = field(init=False, default=None)

    async def get_tools(self, ctx):
        toolsets_tools = await gather(*(toolset.get_tools(ctx) for toolset in self.toolsets))
        all_tools: dict[str, ToolsetTool] = {}
        for toolset, tools in zip(self.toolsets, toolsets_tools):
            for name, tool in tools.items():
                if existing := all_tools.get(name):
                    raise UserError(
                        f'... conflicts with existing tool from {existing.toolset.label}: {name!r}. '
                        f'{toolset.tool_name_conflict_hint}'
                    )
                all_tools[name] = _CombinedToolsetTool(
                    source_toolset=toolset, source_tool=tool, ...
                )
        return all_tools

    async def for_run_step(self, ctx):
        new_toolsets = await gather(*(t.for_run_step(ctx) for t in self.toolsets))
        if all(new is old for new, old in zip(new_toolsets, self.toolsets)):
            return self   # short-circuit: avoid allocation if nothing changed
        return replace(self, toolsets=new_toolsets)

    async def get_instructions(self, ctx):
        """Collects instructions from ALL child toolsets — not just the first."""
        results = await gather(*(ts.get_instructions(ctx) for ts in self.toolsets))
        return [part for r in results if r for part in (r if isinstance(r, list) else [r])] or None
```

### 7.1 Combine Agent Tools, MCP Tools, and Custom Toolset

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset

# In-process Python tools
local_tools = FunctionToolset()


@local_tools.tool_plain
def get_current_time() -> str:
    """Return the current UTC time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# External MCP server tools — prefix to avoid conflicts
mcp = MCPServerStdio('uvx', ['mcp-server-fetch'])
mcp_prefixed = PrefixedToolset(toolset=mcp, prefix='web')

agent = Agent(
    'openai:gpt-4o',
    toolsets=[
        CombinedToolset(toolsets=[local_tools, mcp_prefixed])
    ],
)
```

### 7.2 Nested CombinedToolsets for Layered Organisation

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset

read_tools = FunctionToolset()
write_tools = FunctionToolset()
admin_tools = FunctionToolset()


@read_tools.tool_plain
def read_doc(doc_id: str) -> str:
    """Fetch a document by ID."""
    return f'content of {doc_id}'


@write_tools.tool_plain
def create_doc(title: str, body: str) -> str:
    """Create a new document."""
    return f'created: {title}'


@admin_tools.tool_plain
def delete_doc(doc_id: str) -> str:
    """Delete a document permanently."""
    return f'deleted {doc_id}'


# Compose hierarchically
doc_tools = CombinedToolset(toolsets=[
    PrefixedToolset(toolset=read_tools, prefix='doc_read'),
    PrefixedToolset(toolset=write_tools, prefix='doc_write'),
])

all_tools = CombinedToolset(toolsets=[
    doc_tools,
    PrefixedToolset(toolset=admin_tools, prefix='admin'),
])

agent = Agent('openai:gpt-4o', toolsets=[all_tools])
```

### 7.3 Collecting Instructions from Multiple Toolsets

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.combined import CombinedToolset

# FunctionToolset can carry instructions that get injected as system prompt segments
search_tools = FunctionToolset(
    instructions='Always cite the source when returning search results.'
)
code_tools = FunctionToolset(
    instructions='Return code examples in fenced code blocks.'
)


@search_tools.tool_plain
def web_search(query: str) -> list[str]:
    """Search the web."""
    return [f'Result: {query}']


@code_tools.tool_plain
def run_code(snippet: str) -> str:
    """Execute a Python snippet."""
    return eval(snippet)  # noqa: S307 (demo only)


# CombinedToolset.get_instructions() fans out and merges both instruction sets
combined = CombinedToolset(toolsets=[search_tools, code_tools])
agent = Agent('openai:gpt-4o', toolsets=[combined])
```

---

## 8. `ExternalToolset` — Out-of-Band Tool Execution

**Source**: `pydantic_ai/toolsets/external.py`

`ExternalToolset` registers tools whose results are produced **outside** the pydantic-ai run — for example, by a human operator, a separate process, or a frontend form. Every tool is stamped with `kind='external'`, which tells the agent graph to pause and wait for an external result rather than calling any Python function. `call_tool()` raises `NotImplementedError` by design.

```python
# Key signatures verified from source:

TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())

class ExternalToolset(AbstractToolset[AgentDepsT]):
    tool_defs: list[ToolDefinition]
    _id: str | None

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None):
        self.tool_defs = tool_defs
        self._id = id

    async def get_tools(self, ctx):
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=replace(tool_def, kind='external'),  # <-- forces external kind
                max_retries=0,
                args_validator=TOOL_SCHEMA_VALIDATOR,         # accepts any args shape
            )
            for tool_def in self.tool_defs
        }

    async def call_tool(self, name, tool_args, ctx, tool):
        raise NotImplementedError('External tools cannot be called directly')
```

### 8.1 Pause for Human Approval of a Structured Action

```python
import asyncio
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.external import ExternalToolset

# Declare the tool schema — the model will fill in the args, then pause.
approve_transfer = ToolDefinition(
    name='approve_bank_transfer',
    description='Request approval for a bank transfer. A human must confirm.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'recipient_iban': {'type': 'string'},
            'amount_gbp': {'type': 'number'},
            'reference': {'type': 'string'},
        },
        'required': ['recipient_iban', 'amount_gbp'],
    },
)

ext_toolset = ExternalToolset(tool_defs=[approve_transfer])
# output_type=[str, DeferredToolRequests]: the agent pauses when the model calls an
# external tool and surfaces the pending call so you can supply the result.
agent = Agent('openai:gpt-4o', toolsets=[ext_toolset], output_type=[str, DeferredToolRequests],
              system_prompt='You help users with banking. Use approve_bank_transfer to initiate.')


async def main():
    result = await agent.run('Transfer £500 to GB33BUKB20201555555555')
    if isinstance(result.output, DeferredToolRequests):
        # result.output.calls contains the external tool requests
        for call in result.output.calls:
            print(f'External call requested: {call.tool_name}({call.args_as_dict()})')
        # Supply results from your external system and resume
        tool_results = result.output.build_results(
            calls={call.tool_call_id: {'status': 'approved'} for call in result.output.calls}
        )
        final = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=tool_results,
        )
        print(final.output)


asyncio.run(main())
```

### 8.2 Frontend Form Integration

```python
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.external import ExternalToolset

# The frontend renders a form when the model requests this tool.
# The user fills in the form; the frontend submits the result back via
# deferred_tool_results on the next agent.run() call.
collect_address = ToolDefinition(
    name='collect_shipping_address',
    description='Display a form for the user to enter their shipping address.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'reason': {'type': 'string', 'description': 'Why the address is needed'},
        },
        'required': ['reason'],
    },
)

agent = Agent(
    'openai:gpt-4o',
    toolsets=[ExternalToolset(tool_defs=[collect_address])],
    output_type=[str, DeferredToolRequests],
    system_prompt='Guide the user through checkout. Collect their address first.',
)
```

### 8.3 Using `id` for Deferred Capability Loading

```python
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.external import ExternalToolset

# The 'id' links this toolset to a capability registered in the agent spec,
# enabling deferred loading — the capability is only materialised when needed.
TOOLSET_ID = 'human-approval-v1'

approval_tool = ToolDefinition(
    name='request_manager_sign_off',
    description='Escalate a decision to a human manager for sign-off.',
    parameters_json_schema={
        'type': 'object',
        'properties': {'decision': {'type': 'string'}, 'urgency': {'type': 'string'}},
        'required': ['decision'],
    },
)

ext = ExternalToolset(tool_defs=[approval_tool], id=TOOLSET_ID)
# output_type=[str, DeferredToolRequests] is required to receive the external tool
# call so you can route it to the deferred capability system.
agent = Agent('openai:gpt-4o', toolsets=[ext], output_type=[str, DeferredToolRequests])
```

---

## 9. `Embedder` + `EmbeddingModel` + `EmbeddingResult` — Full Embedding Pipeline

**Sources**: `pydantic_ai/embeddings/base.py`, `pydantic_ai/embeddings/result.py`, and `pydantic_ai/__init__.py` (`Embedder`)

The embedding stack has three layers. `EmbeddingModel` is the abstract provider-specific backend. `Embedder` is the high-level façade — mirrors `Agent`'s API surface with provider-prefix string inference, OTel instrumentation, and sync wrappers. `EmbeddingResult` is the structured response with dict-like access by input string, cost calculation via `genai-prices`, and usage tracking.

```python
# Key signatures verified from source:

class EmbeddingModel(ABC):
    async def embed(self, inputs: str | Sequence[str], *, input_type: EmbedInputType,
                    settings: EmbeddingSettings | None = None) -> EmbeddingResult: ...
    def prepare_embed(self, inputs, settings=None) -> tuple[list[str], EmbeddingSettings]:
        """Normalise inputs to list + merge settings. Call at start of embed()."""
        ...
    async def max_input_tokens(self) -> int | None: ...
    async def count_tokens(self, text: str) -> int: ...

EmbedInputType = Literal['query', 'document']

class EmbeddingSettings(TypedDict, total=False):
    dimensions: int        # output vector size (OpenAI, Cohere, Google, Bedrock, VoyageAI)
    truncate: bool         # truncate overlength inputs (Cohere, Bedrock, VoyageAI)
    extra_headers: dict[str, str]
    extra_body: object

@dataclass
class EmbeddingResult:
    embeddings: Sequence[Sequence[float]]
    inputs: Sequence[str]
    input_type: EmbedInputType
    model_name: str
    provider_name: str
    timestamp: datetime
    usage: RequestUsage
    provider_details: dict[str, Any] | None
    provider_response_id: str | None

    def __getitem__(self, item: int | str) -> Sequence[float]:
        """Access embedding by index OR by the original input string."""
        if isinstance(item, str):
            item = self.inputs.index(item)
        return self.embeddings[item]

    def cost(self) -> genai_types.PriceCalculation:
        """Uses genai-prices for cost estimation."""
        ...

@dataclass(init=False)
class Embedder:
    async def embed_query(self, query: str | Sequence[str], *,
                          settings: EmbeddingSettings | None = None) -> EmbeddingResult: ...
    async def embed_documents(self, documents: str | Sequence[str], *,
                              settings: EmbeddingSettings | None = None) -> EmbeddingResult: ...
    async def embed(self, inputs: str | Sequence[str], *, input_type: EmbedInputType,
                    settings: EmbeddingSettings | None = None) -> EmbeddingResult: ...
    # Sync variants: embed_query_sync, embed_documents_sync, embed_sync
    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None: ...
```

### 9.1 Query vs Document Embedding for Semantic Search

```python
import asyncio
import numpy as np
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

DOCUMENTS = [
    'PydanticAI is a Python agent framework.',
    'Embeddings map text to dense vector space.',
    'The Eiffel Tower is in Paris.',
]


async def semantic_search(query: str) -> str:
    # Embed documents — optimised for storage/retrieval
    doc_result = await embedder.embed_documents(DOCUMENTS)

    # Embed query — optimised for searching
    q_result = await embedder.embed_query(query)

    # Cosine similarity
    q_vec = np.array(q_result.embeddings[0])
    scores = [
        float(np.dot(q_vec, np.array(d)) / (np.linalg.norm(q_vec) * np.linalg.norm(d)))
        for d in doc_result.embeddings
    ]
    best_idx = int(np.argmax(scores))

    # EmbeddingResult supports __getitem__ by string:
    assert list(doc_result[DOCUMENTS[best_idx]]) == list(doc_result.embeddings[best_idx])

    print(f'Best match ({scores[best_idx]:.3f}): {DOCUMENTS[best_idx]}')
    return DOCUMENTS[best_idx]


asyncio.run(semantic_search('What is an AI agent framework?'))
```

### 9.2 Batch Embedding with Dimensionality Reduction and Cost Tracking

```python
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

# Request 256-dimension vectors (saves tokens vs default 1536)
settings: EmbeddingSettings = {'dimensions': 256}

embedder = Embedder('openai:text-embedding-3-small', settings=settings)

CORPUS = [f'Document number {i} about topic {i % 5}' for i in range(50)]


async def main():
    result = await embedder.embed_documents(CORPUS)

    print(f'Vectors: {len(result.embeddings)} × {len(result.embeddings[0])}')
    #> Vectors: 50 × 256

    print(f'Input tokens: {result.usage.input_tokens}')

    try:
        price = result.cost()
        print(f'Estimated cost: ${price.total_price:.6f}')
    except LookupError:
        print('Pricing not available for this model/provider combination.')

    # Access by index or by original string
    vec_by_idx = result[0]
    vec_by_str = result[CORPUS[0]]
    assert vec_by_idx == vec_by_str


asyncio.run(main())
```

### 9.3 Custom EmbeddingModel + Override in Tests

```python
import asyncio
from collections.abc import Sequence
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingModel, EmbeddingSettings
from pydantic_ai.embeddings.result import EmbeddingResult, EmbedInputType
from pydantic_ai.usage import RequestUsage


class DeterministicEmbeddingModel(EmbeddingModel):
    """Test double — returns a constant vector for each input."""

    @property
    def model_name(self) -> str:
        return 'deterministic-test'

    @property
    def system(self) -> str:
        return 'test'

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        normalised, _ = self.prepare_embed(inputs, settings)
        # One unique dimension per input so comparisons are meaningful
        vecs = [[float(i)] * 4 for i in range(len(normalised))]
        return EmbeddingResult(
            embeddings=vecs,
            inputs=normalised,
            input_type=input_type,
            model_name=self.model_name,
            provider_name=self.system,
            usage=RequestUsage(input_tokens=sum(len(t) for t in normalised)),
        )


# Use directly
model = DeterministicEmbeddingModel()
embedder = Embedder(model)


async def main():
    result = await embedder.embed_query(['hello', 'world'])
    assert result['hello'] == [0.0, 0.0, 0.0, 0.0]
    assert result['world'] == [1.0, 1.0, 1.0, 1.0]

    # Override in tests using Embedder.override()
    production_embedder = Embedder('openai:text-embedding-3-small')
    with production_embedder.override(model=model):
        r = production_embedder.embed_query_sync('test')
        assert r.model_name == 'deterministic-test'


asyncio.run(main())
```

---

## 10. `FunctionToolset` — Richly Configurable Python-Function Toolset

**Source**: `pydantic_ai/toolsets/function.py`

`FunctionToolset` is the most feature-complete toolset — it powers the `@agent.tool` decorator but also works standalone. Beyond basic function registration it adds: per-call **timeout** (returns a retry prompt rather than crashing), **sequential** barrier mode (tool runs alone, no overlap), `requires_approval` (shortcuts to `ApprovalRequiredToolset` semantics), `defer_loading` for lazy discovery, `include_return_schema` for type-aware output, and `instructions` to inject system-prompt segments alongside the tools.

```python
# Key constructor signature verified from source:

class FunctionToolset(AbstractToolset[AgentDepsT]):
    def __init__(
        self,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [],
        *,
        max_retries: int | None = None,   # None → inherits agent default
        timeout: float | None = None,     # seconds; triggers retry prompt on timeout
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,       # OpenAI-only JSON schema strictness
        sequential: bool = False,         # tool acts as a barrier — runs alone
        requires_approval: bool = False,  # all tools need human approval
        metadata: dict[str, Any] | None = None,
        defer_loading: bool = False,      # lazy tool discovery
        include_return_schema: bool | None = None,
        id: str | None = None,            # deferred capability registry ID
        instructions: str | SystemPromptFunc[AgentDepsT] | Sequence[...] | None = None,
    ): ...

    @overload
    def tool(self, func: ToolFuncEither[AgentDepsT, ...]) -> ToolFuncEither[AgentDepsT, ...]: ...
    @overload
    def tool(self, *, name: str | None = None, ...) -> Callable: ...
```

### 10.1 Timeout — Return a Retry Prompt on Slow Tools

```python
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

# Any tool that takes longer than 2 s will cause the model to receive
# a retry prompt rather than a Python exception propagating.
slow_toolset = FunctionToolset(timeout=2.0)


@slow_toolset.tool_plain
async def fetch_report(report_id: str) -> str:
    """Fetch a report from the data warehouse."""
    await asyncio.sleep(0.5)  # fast enough
    return f'Report {report_id}: revenue = £12,000'


@slow_toolset.tool_plain
async def run_heavy_query(sql: str) -> str:
    """Run an arbitrary SQL query — may be slow."""
    await asyncio.sleep(5)  # will time out
    return 'never reached'


agent = Agent('openai:gpt-4o', toolsets=[slow_toolset])


async def main():
    # The agent will receive a timeout retry prompt for run_heavy_query
    # and can choose to retry or give up gracefully.
    result = await agent.run('Run the daily summary query and fetch report 42.')
    print(result.output)


asyncio.run(main())
```

### 10.2 Sequential Barrier + Instructions Injection

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

# sequential=True: this toolset's tools are never called in parallel with others.
# The model must finish all parallel calls before invoking a sequential tool.
commit_toolset = FunctionToolset(
    sequential=True,
    instructions=(
        'Use commit_changes only after all validation tools have confirmed success. '
        'Never call commit_changes in the same step as any other tool.'
    ),
)

validate_toolset = FunctionToolset()


@validate_toolset.tool_plain
def run_tests() -> str:
    """Run the test suite."""
    return 'All 142 tests passed.'


@validate_toolset.tool_plain
def lint_code() -> str:
    """Run the linter."""
    return 'No lint errors.'


@commit_toolset.tool_plain
def commit_changes(message: str) -> str:
    """Commit the current changes to version control."""
    return f'Committed: {message}'


# CombinedToolset aggregates instructions from both child toolsets
from pydantic_ai.toolsets.combined import CombinedToolset

all_tools = CombinedToolset(toolsets=[validate_toolset, commit_toolset])
agent = Agent('openai:gpt-4o', toolsets=[all_tools])


async def main():
    result = await agent.run('Validate and commit the current changes.')
    print(result.output)


asyncio.run(main())
```

### 10.3 `include_return_schema` + `require_parameter_descriptions`

```python
import asyncio
from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

# include_return_schema=True: the JSON schema for the return type is included
# in the tool definition so the model can structure its follow-up reasoning.
# require_parameter_descriptions=True: raises an error at registration time
# if any parameter lacks a description in the docstring.
strict_toolset = FunctionToolset(
    include_return_schema=True,
    require_parameter_descriptions=True,
    strict=True,  # OpenAI strict mode — all params required, no additionalProperties
)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float


@strict_toolset.tool_plain
def structured_search(
    query: Annotated[str, Field(description='The search query string')],
    max_results: Annotated[int, Field(description='Maximum number of results to return', ge=1, le=20)] = 5,
) -> list[SearchResult]:
    """Search the knowledge base and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    return [
        SearchResult(
            title=f'Result {i}',
            url=f'https://example.com/{i}',
            snippet=f'Relevant content about {query}',
            relevance_score=1.0 - i * 0.1,
        )
        for i in range(min(max_results, 3))
    ]


agent = Agent('openai:gpt-4o', toolsets=[strict_toolset])


async def main():
    result = await agent.run('Find information about pydantic-ai toolsets.')
    print(result.output)


asyncio.run(main())
```

---

## Summary Table

| Class | Module | Key Innovation (2.2.0) |
|---|---|---|
| `FallbackModel` | `models/fallback.py` | Auto-detects exception vs response handlers via type-hint inspection; `anyio.Lock` deferred via `cached_property` |
| `FilteredToolset` | `toolsets/filtered.py` | Sync/async predicate; `inspect.isawaitable()` dispatch |
| `ApprovalRequiredToolset` | `toolsets/approval_required.py` | `ctx.tool_call_approved` flag; `ApprovalRequired` exception; triple-param selector |
| `RenamedToolset` | `toolsets/renamed.py` | `name_map {new: original}`; immutable `replace()`; `ctx.tool_name` restoration |
| `PrefixedToolset` | `toolsets/prefixed.py` | `{prefix}_{name}` pattern; `removeprefix()` strip; `tool_name_conflict_hint` |
| `PreparedToolset` | `toolsets/prepared.py` | Per-step `ToolDefinition` mutation; add/rename guard raises `UserError` |
| `CombinedToolset` | `toolsets/combined.py` | Parallel `gather()` fan-out; `_CombinedToolsetTool` source tracking; `for_run_step` short-circuit |
| `ExternalToolset` | `toolsets/external.py` | `kind='external'` stamp; `call_tool` raises; `id` for deferred capability |
| `Embedder` / `EmbeddingModel` / `EmbeddingResult` | `embeddings/` | `embed_query` vs `embed_documents`; `result['text']` dict access; `cost()` via genai-prices; `instrument_all()` |
| `FunctionToolset` | `toolsets/function.py` | `timeout` retry prompt; `sequential` barrier; `requires_approval`; `instructions` system-prompt injection |

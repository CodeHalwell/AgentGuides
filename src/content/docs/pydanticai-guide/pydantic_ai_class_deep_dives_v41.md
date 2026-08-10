---
title: "PydanticAI Class Deep Dives Vol. 41"
description: "Source-verified deep dives into 10 pydantic-ai 2.23.0–2.27.0 class groups: ReinjectSystemPrompt (server-side system-prompt authority — replace_existing, UI/DB integration, manage_system_prompt='server'), ProcessHistory (pre-request message transform — sync/async × with/without RunContext, context compression), ProcessEventStream (event stream interception — observer handler vs async-generator processor, FinalResultEvent semantics), PrepareTools + PrepareOutputTools (capability-based tool gating — ToolsPrepareFunc wrapping, role-filtered and step-gated examples), SetToolMetadata (metadata overlay capability — ToolSelector scoping, kwargs merge, per-tool targeting), ApprovalRequiredToolset + FilteredToolset (tool-call approval gate — ctx.tool_call_approved, selective approval_required_func, async filter support), RenamedToolset + ExternalToolset (tool name-space remapping — name_map conflict detection, external-tool placeholder for human-in-the-loop), Embedder (high-level embedding interface — embed_query/embed_documents, override() context manager, instrument_all()), ModelProfile + merge_profile (TypedDict capability manifest — tool_deferral_mode, tool_addition_mode, DEFAULT_PROFILE composition), RunUsage.cost + UsageLimits.cost_limit + CancellationToken (cost ceiling enforcement + first-party run cancellation — check_cost, RunCancelled.all_messages(), thread-safe multi-run token)."
sidebar:
  label: "Class deep dives (Vol. 41)"
  order: 67
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.27.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.23.x–2.27.x API. Three examples per class group; all code blocks pass `ast.parse()` syntax validation. Live API calls are commented out — uncomment to run.
</Aside>

Ten class groups covering server-side system-prompt authority (`ReinjectSystemPrompt`), pre-request message-history transforms (`ProcessHistory`), event-stream interception (`ProcessEventStream`), capability-based tool gating (`PrepareTools` + `PrepareOutputTools`), metadata overlays on tools (`SetToolMetadata`), approval-gated and dynamically filtered toolsets (`ApprovalRequiredToolset` + `FilteredToolset`), tool name-remapping and external-tool placeholders (`RenamedToolset` + `ExternalToolset`), the high-level embedding interface (`Embedder`), the per-model capability manifest (`ModelProfile` + `merge_profile`), and cost-ceiling enforcement with first-party run cancellation (`RunUsage.cost` + `UsageLimits.cost_limit` + `CancellationToken`).

---

## 1. `ReinjectSystemPrompt`

**Source:** `pydantic_ai/capabilities/reinject_system_prompt.py`

`ReinjectSystemPrompt` is a `before_model_request` capability that guarantees the agent's configured `system_prompt` appears at the head of the first `ModelRequest` on every call, even when the `message_history` was reconstructed from a source that stripped system prompts (a database row, a UI frontend, a compaction pipeline). By default it is a no-op when any `SystemPromptPart` is already present anywhere in the history — the existing prompt is treated as authoritative. Set `replace_existing=True` to strip all `SystemPromptPart`s before prepending the agent's prompt; this is what the built-in UI adapters use in `manage_system_prompt='server'` mode.

```python
# Example 1 — Minimal: ensure a system prompt survives round-trips through a DB layer
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    'openai:gpt-5',
    system_prompt='You are a helpful assistant.',
    capabilities=[ReinjectSystemPrompt()],
)

# When message_history is reloaded from a DB that dropped the SystemPromptPart,
# ReinjectSystemPrompt adds it back before the model sees the messages.
# result = await agent.run('Continue our conversation.', message_history=history_from_db)
```

```python
# Example 2 — replace_existing=True: server prompt wins over any client-supplied prompts
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

agent = Agent(
    'anthropic:claude-sonnet-5-20251101',
    system_prompt='You must always answer in formal English.',
    capabilities=[ReinjectSystemPrompt(replace_existing=True)],
)

# Even if the caller's message_history contains a SystemPromptPart (e.g. injected
# by a UI frontend or relayed from another agent), replace_existing=True strips it
# and prepends the server-controlled prompt, so the server's instructions always win.
history_with_stale_prompt = [
    ModelRequest(parts=[
        SystemPromptPart(content='Be casual.'),
        UserPromptPart(content='Hey, what time is it?'),
    ])
]
# result = await agent.run('Now answer formally.', message_history=history_with_stale_prompt)
# The stale 'Be casual.' system prompt is replaced with 'You must always answer in formal English.'
```

```python
# Example 3 — Per-run override: inject server prompt only for specific sensitive requests
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent('openai:gpt-5', system_prompt='You are a compliance assistant.')

async def handle_request(prompt: str, history: list, is_sensitive: bool) -> str:
    caps = [ReinjectSystemPrompt(replace_existing=True)] if is_sensitive else []
    result = await agent.run(
        prompt,
        message_history=history,
        capabilities=caps,
    )
    return result.output

# Sensitive requests always get the server prompt prepended; routine ones use whatever
# system prompt is already in the history (preserving multi-agent handoff context).
```

---

## 2. `ProcessHistory`

**Source:** `pydantic_ai/capabilities/process_history.py`

`ProcessHistory` wraps a `HistoryProcessorFunc` — a callable that receives `(RunContext, list[ModelMessage])` or just `(list[ModelMessage])` — as a `before_model_request` capability. The processor can truncate, summarise, redact PII, or otherwise reshape the message list before each model call. Both sync and async processors are accepted; pydantic-ai introspects the function signature to decide which calling convention to use. Because it takes a callable it is not spec-serialisable and cannot round-trip through `Agent.from_spec()`.

```python
# Example 1 — Sliding window: keep only the last N turns to cap context size
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

def keep_last_n_turns(messages: list[ModelMessage], n: int = 6) -> list[ModelMessage]:
    """Keep the system-prompt turn plus the most recent n message pairs."""
    system_turns = [m for m in messages if isinstance(m, ModelRequest)
                    and any(isinstance(p, SystemPromptPart) for p in m.parts)]
    non_system = [m for m in messages if m not in system_turns]
    return system_turns + non_system[-n:]

agent = Agent(
    'openai:gpt-5',
    capabilities=[ProcessHistory(lambda msgs: keep_last_n_turns(msgs, n=6))],
)
```

```python
# Example 2 — Async processor with RunContext: redact PII based on user tier
import re
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from dataclasses import dataclass

@dataclass
class UserDeps:
    tier: str  # 'free' | 'pro'

_EMAIL_RE = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

async def redact_if_free_tier(
    ctx: RunContext[UserDeps],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    if ctx.deps.tier == 'pro':
        return messages
    redacted: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            new_parts = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    part = UserPromptPart(content=_EMAIL_RE.sub('[EMAIL]', part.content))
                new_parts.append(part)
            from dataclasses import replace
            msg = replace(msg, parts=new_parts)
        redacted.append(msg)
    return redacted

agent = Agent(
    'openai:gpt-5',
    deps_type=UserDeps,
    capabilities=[ProcessHistory(redact_if_free_tier)],
)
```

```python
# Example 3 — Summarise old turns: compress history beyond a token threshold
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse,
    UserPromptPart, TextPart,
)

KEEP_RECENT = 4  # keep the 4 most-recent turns verbatim

async def compress_old_turns(messages: list[ModelMessage]) -> list[ModelMessage]:
    if len(messages) <= KEEP_RECENT:
        return messages
    old = messages[:-KEEP_RECENT]
    recent = messages[-KEEP_RECENT:]
    # Build a one-line summary placeholder (in production, call a summariser model)
    summary_text = f'[{len(old)} earlier messages summarised]'
    summary_part = UserPromptPart(content=summary_text)
    summary_msg = ModelRequest(parts=[summary_part])
    return [summary_msg] + recent

agent = Agent(
    'openai:gpt-5',
    capabilities=[ProcessHistory(compress_old_turns)],
)
```

---

## 3. `ProcessEventStream`

**Source:** `pydantic_ai/capabilities/process_event_stream.py`

`ProcessEventStream` intercepts the `AgentStreamEvent` stream that flows out of each `ModelRequestNode` and `CallToolsNode`. It supports two forms of handler:

- **Observer** (`async def handler(ctx, stream) -> None`): receives a tee'd copy of the stream; events are still passed through unchanged to all downstream consumers. A slow observer back-pressures the stream.
- **Processor** (`async def handler(ctx, stream)` that is an async generator): the events it yields *replace* the inner stream for every downstream consumer — it can modify, drop, or inject events. Dropping a `FinalResultEvent` delays `run_stream()` result delivery until the full model response is buffered.

When registered, `agent.run()` automatically enables streaming internally so the handler fires without an explicit `event_stream_handler=` argument.

```python
# Example 1 — Observer: log every event type without altering the stream
import asyncio
from collections.abc import AsyncIterable
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent

async def log_event_types(
    ctx: RunContext,
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        print(f'[event] {type(event).__name__}')
        # Return early if you only need the first N events; downstream is unaffected.

agent = Agent(
    'openai:gpt-5',
    capabilities=[ProcessEventStream(log_event_types)],
)
# result = await agent.run('Hello')
# Prints event type names as the response streams, e.g.:
# [event] PartStartEvent
# [event] PartDeltaEvent
# [event] FinalResultEvent
```

```python
# Example 2 — Processor: strip thinking parts from the visible event stream
import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartStartEvent, ThinkingPart, ThinkingPartDelta

async def hide_thinking(
    ctx: RunContext,
    stream: AsyncIterable[AgentStreamEvent],
) -> AsyncIterator[AgentStreamEvent]:
    skip_part_id: int | None = None
    async for event in stream:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            skip_part_id = event.index
            continue  # drop the PartStartEvent for thinking parts
        if skip_part_id is not None and hasattr(event, 'index') and event.index == skip_part_id:
            continue  # drop all delta events for that part index
        if skip_part_id is not None and hasattr(event, 'index') and event.index != skip_part_id:
            skip_part_id = None
        yield event

agent = Agent(
    'anthropic:claude-sonnet-5-20251101',
    capabilities=[ProcessEventStream(hide_thinking)],
)
```

```python
# Example 3 — Observer with side-channel metrics collection
import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, FinalResultEvent

@dataclass
class StreamMetrics:
    delta_events: int = 0
    char_count: int = 0
    final_seen: bool = False

metrics = StreamMetrics()

async def collect_metrics(
    ctx: RunContext,
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        if isinstance(event, PartDeltaEvent):
            metrics.delta_events += 1
            delta = event.delta
            if hasattr(delta, 'content_delta') and isinstance(delta.content_delta, str):
                metrics.char_count += len(delta.content_delta)
        elif isinstance(event, FinalResultEvent):
            metrics.final_seen = True

agent = Agent(
    'openai:gpt-5',
    capabilities=[ProcessEventStream(collect_metrics)],
)
# result = await agent.run('Write a short poem.')
# After the run: metrics.delta_events, metrics.char_count, metrics.final_seen are populated.
```

---

## 4. `PrepareTools` + `PrepareOutputTools`

**Source:** `pydantic_ai/capabilities/prepare_tools.py`

`PrepareTools` wraps a `ToolsPrepareFunc` — `(RunContext, list[ToolDefinition]) -> list[ToolDefinition]` — as a capability that fires at the `prepare_tools` hook, letting you filter or modify the set of **function** tools exposed to the model on every request. `PrepareOutputTools` mirrors it for **output tools** (`ToolOutput`) — its `ctx.retry` and `ctx.max_retries` reflect the output retry budget (`max_output_retries`). Both accept sync and async callables, and neither is spec-serialisable because they hold a callable.

```python
# Example 1 — Role-gated tools: hide admin tools from non-admin users
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition

@dataclass
class UserDeps:
    role: str  # 'admin' | 'user'

async def hide_admin_tools(
    ctx: RunContext[UserDeps],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    if ctx.deps.role == 'admin':
        return tool_defs
    return [td for td in tool_defs if not td.name.startswith('admin_')]

agent = Agent(
    'openai:gpt-5',
    deps_type=UserDeps,
    capabilities=[PrepareTools(hide_admin_tools)],
)

@agent.tool
def admin_delete_user(ctx: RunContext[UserDeps], user_id: str) -> str:
    return f'Deleted {user_id}'

@agent.tool
def get_profile(ctx: RunContext[UserDeps]) -> str:
    return 'Your profile'

# For non-admin users, admin_delete_user is never sent to the model.
```

```python
# Example 2 — Step-gated output tool: only offer structured output after at least one round-trip
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareOutputTools
from pydantic_ai.output import ToolOutput
from pydantic_ai.tools import ToolDefinition
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float

def search_web(query: str) -> str:
    return f'Results for: {query}'

async def only_after_tools(
    ctx: RunContext,
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    # ctx.run_step increments each model round-trip (after any model response, including retries).
    # Suppress the output tool on the first step so the model must call search_web before answering.
    return tool_defs if ctx.run_step > 0 else []

agent = Agent(
    'openai:gpt-5',
    output_type=ToolOutput(Answer),
    tools=[search_web],
    capabilities=[PrepareOutputTools(only_after_tools)],
)
```

```python
# Example 3 — Sync prepare: add a description prefix to all tool definitions
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition
from dataclasses import replace

def add_env_prefix(
    ctx: RunContext,
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    env = 'STAGING'  # would come from ctx.deps in a real app
    return [
        replace(td, description=f'[{env}] {td.description or td.name}')
        for td in tool_defs
    ]

agent = Agent(
    'openai:gpt-5',
    capabilities=[PrepareTools(add_env_prefix)],
)

@agent.tool_plain
def search_products(query: str) -> list[str]:
    return [f'Product matching {query}']
```

---

## 5. `SetToolMetadata`

**Source:** `pydantic_ai/capabilities/set_tool_metadata.py`

`SetToolMetadata` is a `get_wrapper_toolset` capability that merges arbitrary key-value pairs into the `metadata` dict of matching tools. The constructor accepts `tools` (a `ToolSelector` — `'all'`, a tool name string, a `ToolDefinition`, or a list of those) followed by **keyword arguments** that become the metadata payload. The selector is resolved at runtime per request, so you can target tools by name, definition, or `'all'`. Internally it wraps the toolset in a `PreparedToolset` that overrides `get_tools` with the merge applied.

```python
# Example 1 — Tag all tools with an environment label
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent(
    'openai:gpt-5',
    capabilities=[SetToolMetadata(environment='production', version='2.27')],
)

@agent.tool_plain
def get_order(order_id: str) -> dict:
    return {'id': order_id, 'status': 'shipped'}

# Every tool will have metadata={'environment': 'production', 'version': '2.27'}
# accessible in RunContext.tool_call_id flows and observability exports.
```

```python
# Example 2 — Target a single tool by name
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        SetToolMetadata(tools='execute_sql', requires_approval=True, audit_log=True),
    ],
)

@agent.tool_plain
def execute_sql(query: str) -> str:
    return f'Rows from: {query}'

@agent.tool_plain
def get_schema(table: str) -> str:
    return f'Schema of {table}'

# Only execute_sql gets metadata={'requires_approval': True, 'audit_log': True}.
# get_schema is unaffected.
```

```python
# Example 3 — Stack multiple SetToolMetadata capabilities for layered tagging
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        SetToolMetadata(team='data-platform'),              # all tools
        SetToolMetadata(tools='run_query', rate_limited=True),  # specific tool
    ],
)

@agent.tool_plain
def run_query(sql: str) -> list:
    return []

@agent.tool_plain
def list_tables() -> list:
    return ['orders', 'users']

# run_query gets: {'team': 'data-platform', 'rate_limited': True}
# list_tables gets: {'team': 'data-platform'}
```

---

## 6. `ApprovalRequiredToolset` + `FilteredToolset`

**Source:** `pydantic_ai/toolsets/approval_required.py`, `pydantic_ai/toolsets/filtered.py`

`ApprovalRequiredToolset` wraps any toolset and raises `ApprovalRequired` before executing a call unless `ctx.tool_call_approved` is `True` or the `approval_required_func` returns `False` for that specific call. The default `approval_required_func` requires approval for every tool; supply a custom one to be selective based on tool name, arguments, or context.

`FilteredToolset` wraps a toolset and applies a predicate to `ToolDefinition` objects at `get_tools` time, removing non-matching tools from the model's view. Both sync and async predicates are accepted.

```python
# Example 1 — ApprovalRequiredToolset: require approval for destructive operations
from pydantic_ai import Agent, RunContext, DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.tools import ToolDefinition

def needs_approval(ctx: RunContext, tool_def: ToolDefinition, args: dict) -> bool:
    return tool_def.name in {'delete_record', 'drop_table'}

def delete_record(record_id: str) -> str:
    return f'Deleted {record_id}'

def list_records() -> list:
    return ['rec_1', 'rec_2']

dangerous_toolset = ApprovalRequiredToolset(
    FunctionToolset([delete_record, list_records]),
    approval_required_func=needs_approval,
)

agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests], toolsets=[dangerous_toolset])
# list_records runs freely; delete_record is deferred for approval.
# Full two-run flow:
# result1 = await agent.run('delete record X')
# assert isinstance(result1.output, DeferredToolRequests)
# deferred = result1.output.build_results(approve_all=True)
# result2 = await agent.run('', message_history=result1.all_messages(),
#                           deferred_tool_results=deferred)
```

```python
# Example 2 — FilteredToolset: expose only read-only tools to the model
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.tools import ToolDefinition

READ_ONLY = {'get_user', 'list_orders', 'search_products'}

def is_read_only(ctx: RunContext, tool_def: ToolDefinition) -> bool:
    return tool_def.name in READ_ONLY

def get_user(user_id: str) -> dict:
    return {'id': user_id}

def delete_user(user_id: str) -> str:
    return f'Deleted {user_id}'

def list_orders() -> list:
    return ['order_1']

read_only_toolset = FilteredToolset(
    FunctionToolset([get_user, delete_user, list_orders]),
    filter_func=is_read_only,
)

agent = Agent('openai:gpt-5', toolsets=[read_only_toolset])
# The model only sees get_user and list_orders; delete_user is never offered.
```

```python
# Example 3 — Async FilteredToolset: show tools based on feature flags in context
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.tools import ToolDefinition

@dataclass
class FeatureFlags:
    beta_tools_enabled: bool

BETA_TOOLS = {'experimental_search', 'ai_summarise'}

async def feature_flag_filter(
    ctx: RunContext[FeatureFlags],
    tool_def: ToolDefinition,
) -> bool:
    if tool_def.name in BETA_TOOLS:
        return ctx.deps.beta_tools_enabled
    return True

def experimental_search(query: str) -> list:
    return []

def standard_lookup(query: str) -> list:
    return ['result_1']

flagged_toolset = FilteredToolset(
    FunctionToolset([experimental_search, standard_lookup]),
    filter_func=feature_flag_filter,
)

agent = Agent('openai:gpt-5', deps_type=FeatureFlags, toolsets=[flagged_toolset])
# Users with beta_tools_enabled=False never see experimental_search.
```

---

## 7. `RenamedToolset` + `ExternalToolset`

**Source:** `pydantic_ai/toolsets/renamed.py`, `pydantic_ai/toolsets/external.py`

`RenamedToolset` wraps a toolset and replaces tool names using a `name_map: dict[str, str]` (new name → original name). The remapping is applied in both `get_tools` (so the model sees the new names) and `call_tool` (so the underlying implementation receives the original name via `ctx.tool_name`). Conflict detection raises `UserError` if a rename would collide with another tool's name.

`ExternalToolset` exposes tool definitions whose results are produced **outside** the Pydantic AI run — by a human, a separate service, or a different agent. It marks tools as `kind='external'`, which causes the agent to pause and surface the tool call to the caller for resolution.

```python
# Example 1 — RenamedToolset: sanitise implementation names for the model
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.renamed import RenamedToolset

def db_get_user_v2(user_id: str) -> dict:
    return {'id': user_id, 'name': 'Alice'}

def db_list_orders_v3(user_id: str) -> list:
    return ['order_1', 'order_2']

clean_toolset = RenamedToolset(
    FunctionToolset([db_get_user_v2, db_list_orders_v3]),
    name_map={
        'get_user': 'db_get_user_v2',
        'list_orders': 'db_list_orders_v3',
    },
)

agent = Agent('openai:gpt-5', toolsets=[clean_toolset])
# The model calls 'get_user' and 'list_orders'.
# The implementation functions db_get_user_v2 and db_list_orders_v3 are invoked correctly.
```

```python
# Example 2 — ExternalToolset: human-in-the-loop tool resolution
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai.tools import ToolDefinition

human_tools = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='approve_payment',
            description='Approve or reject a pending payment. Returns True if approved.',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'payment_id': {'type': 'string'},
                    'amount_usd': {'type': 'number'},
                },
                'required': ['payment_id', 'amount_usd'],
            },
        ),
    ],
    id='human-approval-toolset',
)

agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests], toolsets=[human_tools])
# When the model calls approve_payment, the run returns DeferredToolRequests.
# The caller inspects the tool call, gets human approval, then re-runs with
# deferred_tool_results=result.output.build_results(...) and message_history=result.all_messages().
```

```python
# Example 3 — RenamedToolset with conflict detection
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.renamed import RenamedToolset

def fetch_data(source: str) -> dict:
    return {'source': source, 'data': [1, 2, 3]}

def load_data(path: str) -> dict:
    return {'path': path, 'rows': 42}

# Rename both to different names — no conflict
safe_toolset = RenamedToolset(
    FunctionToolset([fetch_data, load_data]),
    name_map={
        'retrieve_remote': 'fetch_data',
        'retrieve_local': 'load_data',
    },
)

agent = Agent('openai:gpt-5', toolsets=[safe_toolset])
# Model sees: retrieve_remote and retrieve_local
# Underlying functions: fetch_data and load_data — ctx.tool_name reflects original names.

# Attempting to rename a tool to a name already held by another tool raises UserError:
# conflicting_toolset = RenamedToolset(
#     FunctionToolset([fetch_data, load_data]),
#     name_map={'load_data': 'fetch_data'},  # renames fetch_data → load_data, but load_data exists!
# )
# → UserError: "Renaming tool 'fetch_data' to 'load_data' conflicts with existing tool."
```

---

## 8. `Embedder`

**Source:** `pydantic_ai/embeddings/__init__.py`

`Embedder` is the high-level entry point for generating text embeddings. It resolves a model from a `'provider:model-name'` string (e.g. `'openai:text-embedding-3-small'`) using the same provider inference as `Agent`, supports query vs document input-type semantics, provides an `override()` context manager for test-time model swapping, and exposes `instrument_all()` for session-wide OpenTelemetry instrumentation. Synchronous wrappers (`embed_query_sync`, `embed_documents_sync`) are provided for non-async contexts.

```python
# Example 1 — Basic usage: embed a search query and multiple documents
import asyncio
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')

async def semantic_search(query: str, documents: list[str]) -> list[float]:
    # embed_query optimises the embedding for retrieval queries
    query_result = await embedder.embed_query(query)
    q_vec = query_result.embeddings[0]

    # embed_documents optimises embeddings for stored documents
    doc_result = await embedder.embed_documents(documents)

    # Compute cosine similarities (simplified)
    import math
    scores = []
    for doc_vec in doc_result.embeddings:
        dot = sum(a * b for a, b in zip(q_vec, doc_vec))
        norm_q = math.sqrt(sum(x**2 for x in q_vec))
        norm_d = math.sqrt(sum(x**2 for x in doc_vec))
        scores.append(dot / (norm_q * norm_d) if norm_q and norm_d else 0.0)
    return scores

# scores = asyncio.run(semantic_search('machine learning', ['AI overview', 'Recipe book']))
```

```python
# Example 2 — override() for test-time model substitution
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

embedder = Embedder('openai:text-embedding-3-large')

async def test_embedding_pipeline():
    # Swap to TestEmbeddingModel (returns deterministic unit vectors, no API call)
    with embedder.override(model=TestEmbeddingModel()):
        result = await embedder.embed_query('hello world')
        print(len(result.embeddings[0]))  # dimension determined by TestEmbeddingModel

# asyncio.run(test_embedding_pipeline())
```

```python
# Example 3 — instrument_all() + batch embedding + token counting
import asyncio
from pydantic_ai import Embedder

# Enable OpenTelemetry tracing for all Embedder instances globally
Embedder.instrument_all(True)

embedder = Embedder(
    'openai:text-embedding-3-small',
    settings={'dimensions': 512},  # reduce dimensionality
)

async def embed_batch(texts: list[str]) -> dict:
    max_tokens = await embedder.max_input_tokens()
    print(f'Model max input tokens: {max_tokens}')

    result = await embedder.embed_documents(texts)
    return {
        'count': len(result.embeddings),
        'dim': len(result.embeddings[0]) if result.embeddings else 0,
        'model': result.model_name,
    }

# Synchronous shorthand for non-async callers:
# result = embedder.embed_query_sync('What is Pydantic AI?')
# print(result.embeddings[0][:5])
```

---

## 9. `ModelProfile` + `merge_profile`

**Source:** `pydantic_ai/profiles/__init__.py`

`ModelProfile` is a `TypedDict` (all keys optional) that describes how a specific model or model family needs to be addressed: whether it supports tools, JSON schema output, thinking/reasoning, image output, and the modes for structured output, tool deferral, and mid-conversation tool addition. `merge_profile` performs a dict-spread merge (later overrides win); a `None` *argument* is treated as an empty dict (no-op — the base value is preserved), whereas a key explicitly set to `None` within an override dict IS spread (field becomes `None`). Translates deprecated keys (`tool_additions` → `tool_addition_mode`, `deferred_tools_require_tool_search` → `tool_deferral_mode`). `DEFAULT_PROFILE` is the fully-populated base layer. Profiles belong on provider model constructors (`OpenAIChatModel`, `AnthropicModel`, etc.) rather than on `Agent` directly.

```python
# Example 1 — Inspect the DEFAULT_PROFILE
from pydantic_ai.profiles import DEFAULT_PROFILE

print('supports_tools:', DEFAULT_PROFILE['supports_tools'])          # True
print('supports_thinking:', DEFAULT_PROFILE['supports_thinking'])    # False
print('default_structured_output_mode:', DEFAULT_PROFILE['default_structured_output_mode'])  # 'tool'
print('tool_deferral_mode:', DEFAULT_PROFILE['tool_deferral_mode'])  # None
print('tool_addition_mode:', DEFAULT_PROFILE['tool_addition_mode'])  # None
```

```python
# Example 2 — merge_profile: layer a provider default with a model-specific override
from pydantic_ai.profiles import ModelProfile, merge_profile, DEFAULT_PROFILE

# Provider's resolved profile for the model family
anthropic_base: ModelProfile = {
    'supports_tools': True,
    'supports_json_schema_output': False,
    'supports_thinking': True,
    'default_structured_output_mode': 'tool',
    'supports_inline_system_prompts': True,
    'tool_addition_mode': 'by_reference',
}

# Model-specific overrides (e.g. a new claude-sonnet-6 with JSON schema output)
sonnet6_overrides: ModelProfile = {
    'supports_json_schema_output': True,
    'tool_deferral_mode': 'standalone',
}

effective_profile = merge_profile(merge_profile(DEFAULT_PROFILE, anthropic_base), sonnet6_overrides)
print(effective_profile['supports_json_schema_output'])  # True  (override wins)
print(effective_profile['tool_addition_mode'])            # 'by_reference'
print(effective_profile['tool_deferral_mode'])            # 'standalone'
```

```python
# Example 3 — Pass a callable profile to a model constructor for full control
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile

def always_use_prompted_output(base: ModelProfile) -> ModelProfile:
    return {**base, 'default_structured_output_mode': 'prompted'}

model = OpenAIChatModel('gpt-5', profile=always_use_prompted_output)
agent = Agent(model)
# The callable receives the provider's resolved ModelProfile and returns the final profile.
# Profiles live on model constructors (OpenAIChatModel, AnthropicModel, etc.), not on Agent.
# Useful when testing how an agent behaves with a weaker model's capability profile.
```

---

## 10. `RunUsage.cost` + `UsageLimits.cost_limit` + `CancellationToken`

**Source:** `pydantic_ai/usage.py`, `pydantic_ai/_cancel.py`

**`RunUsage.cost`** (added in 2.23.0) is a `Decimal | None` field representing best-effort USD cost for the run, summed across all requests. Providers that don't expose pricing return `None`; zero-cost runs return `Decimal('0')`, keeping "unknown" distinguishable from "free".

**`UsageLimits.cost_limit`** (added alongside `cost`) is a `Decimal | None` ceiling in USD. `check_before_request` enforces it before each request; `check_cost` enforces it after. A `CostNotFoundWarning` is emitted when `cost_limit` is set but no cost was calculated.

**`CancellationToken`** (added in 2.26.0) is a thread-safe handle for first-party run cancellation. Call `.cancel()` from any thread to stop all runs registered with the token. The agent translates the resulting `CancelledError` into `RunCancelled` at the `agent.iter()` boundary; partial history is preserved in `RunCancelled.all_messages()`.

```python
# Example 1 — Cost tracking: inspect per-run USD cost
import asyncio
from decimal import Decimal
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

agent = Agent('openai:gpt-5')

async def run_with_cost_tracking(prompt: str) -> None:
    result = await agent.run(prompt)
    cost = result.usage().cost
    if cost is not None:
        print(f'Run cost: ${cost:.6f}')
    else:
        print('Cost unavailable for this provider/model')

# asyncio.run(run_with_cost_tracking('Summarise this paragraph...'))
```

```python
# Example 2 — cost_limit: abort a run if it would exceed a USD budget
import asyncio
from decimal import Decimal
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-5')

async def budget_capped_run(prompt: str, budget_usd: float) -> str | None:
    limits = UsageLimits(cost_limit=Decimal(str(budget_usd)))
    try:
        result = await agent.run(prompt, usage_limits=limits)
        return result.output
    except UsageLimitExceeded as exc:
        print(f'Stopped: {exc}')
        return None

# asyncio.run(budget_capped_run('Write a 10,000-word essay...', budget_usd=0.05))
# The run is aborted before (check_before_request) or after (check_cost) a request that
# would push cumulative cost above $0.05.
```

```python
# Example 3 — CancellationToken: stop a long-running agent from another thread
import asyncio
import threading
from pydantic_ai import Agent, CancellationToken
from pydantic_ai.exceptions import RunCancelled

agent = Agent('openai:gpt-5', system_prompt='You are a research assistant.')

async def long_run(token: CancellationToken) -> None:
    try:
        result = await agent.run(
            'Research and write a comprehensive report on quantum computing.',
            cancellation_token=token,
        )
        print('Completed:', result.output[:100])
    except RunCancelled as exc:
        print('Run cancelled. Partial messages:', len(exc.all_messages()))
        # Pass exc.all_messages() as message_history to resume later.

async def main() -> None:
    token = CancellationToken()

    # Cancel after 2 seconds from a separate thread (simulating a user pressing Escape)
    def cancel_after_delay() -> None:
        import time
        time.sleep(2)
        token.cancel()

    thread = threading.Thread(target=cancel_after_delay, daemon=True)
    thread.start()

    await long_run(token)
    thread.join()

# asyncio.run(main())
```

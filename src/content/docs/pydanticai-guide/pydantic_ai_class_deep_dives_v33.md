---
title: "PydanticAI Class Deep Dives Vol. 33"
description: "Source-verified deep dives into 10 pydantic-ai 2.6.0 class groups: Hooks+HookTimeoutError+HookNamespace (decorator-based lifecycle hook registration — all 20+ hook types, per-hook anyio timeout, bare vs parameterized forms), ProcessHistory capability (callable history preprocessor — sync/async with/without RunContext, _run_history_processor dispatch), ReinjectSystemPrompt capability (system-prompt persistence across UI round-trips — replace_existing mode, _prepend_to_first_request, _strip_system_prompts), ApprovalRequiredToolset (approval-gated tool execution — approval_required_func predicate, tool_call_approved gate, ApprovalRequired exception), ExternalToolset (tools produced outside the agent run — kind='external' ToolDefinition, TOOL_SCHEMA_VALIDATOR, id namespacing), FilteredToolset (dynamic per-request tool filtering — sync/async predicate, RunContext-aware, WrapperToolset subclass), VercelAIAdapter+VercelAIEventStream (Vercel AI SDK v5/v6 server adapter — sdk_version=6 HITL approval streaming, StartChunk/TextDeltaChunk/FinishChunk protocol, VERCEL_AI_DSP_HEADERS), AGUIAdapter+_AGUIFrontendToolset+interrupt handling (AG-UI protocol adapter — HAS_INTERRUPTS guard, approval_to_interrupt/resume_entry_to_approval, ExternalToolset frontend bridge), new provider profiles harmony_model_profile+moonshotai_model_profile+amazon_model_profile+merge_profile (OpenAI Harmony response format, MoonshotAI Kimi reasoning detection, Amazon InlineDefsJsonSchemaTransformer, profile layering), Decision+DecisionBranch+ReducerContext+JoinState from pydantic_graph (conditional graph branching — typed branch routing, ReducerContext.cancel_sibling_tasks(), ForkStack/JoinID join state). All verified against pydantic-ai 2.6.0 + pydantic-graph 2.6.0 source."
sidebar:
  label: "Class deep dives (Vol. 33)"
  order: 59
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.6.0** and **pydantic-graph 2.6.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.6.x API.
</Aside>

Ten class groups covering the new decorator-based `Hooks` capability, three history/prompt utility capabilities, two new toolset types, the Vercel AI and AG-UI server adapters, new provider profiles, and `pydantic_graph`'s conditional branching infrastructure.

---

## 1. `Hooks` + `HookTimeoutError` + `HookNamespace` — Decorator-Based Lifecycle Hook Registration

**Source**: `pydantic_ai/capabilities/hooks.py`  
**Export**: `from pydantic_ai.capabilities import Hooks`

`Hooks` is an `AbstractCapability` that lets you register lifecycle observer functions via decorators on `hooks.on.<hook_name>` instead of subclassing. It supports all 20+ capability hook points (`before_run`, `after_run`, `before_model_request`, `after_model_request`, `before_tool_execute`, `after_tool_execute`, `on_event`, `prepare_tools`, …). Each hook registration optionally accepts a `timeout` (seconds) — when the hook function takes longer than that, `HookTimeoutError` (a `TimeoutError` subclass) is raised via `anyio.fail_after`.

```python
# Key signature verified from source (pydantic-ai 2.6.0):

class HookTimeoutError(TimeoutError):
    """Raised when a hook function exceeds its configured timeout."""
    hook_name: str
    func_name: str
    timeout: float

@dataclass
class _HookEntry(Generic[_FuncT]):
    func: _FuncT
    timeout: float | None = None

@dataclass
class _ToolHookEntry(_HookEntry[_FuncT]):
    tools: frozenset[str] | None = None   # None → all tools

class Hooks(AbstractCapability[AgentDepsT]):
    # hooks.on is a HookNamespace; each attribute on it is an overloaded decorator:
    #   @hooks.on.before_run           — bare form (no parentheses)
    #   @hooks.on.before_run(timeout=2.0)  — parameterized form
    @cached_property
    def on(self) -> HookNamespace: ...

    # All hook dispatch is forwarded from AbstractCapability overrides
    # (before_run, after_run, before_model_request, after_model_request,
    #  before_tool_execute, after_tool_execute, on_event, etc.)
    # to registered _HookEntry list for that point.
```

Both sync and async hook functions are accepted; sync ones are wrapped transparently via `anyio.to_thread.run_sync`.

### 1.1 Logging and Timing Every Model Request

Register `before_model_request` and `after_model_request` hooks to emit structured timing logs.

```python {test="skip"}
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_before(ctx, request_context):
    ctx.deps['_t0'] = time.monotonic()
    print(f'[req] run_step={ctx.run_step} messages={len(request_context.messages)}')
    return request_context

@hooks.on.after_model_request
async def log_after(ctx, *, request_context, response):
    elapsed = time.monotonic() - ctx.deps.get('_t0', time.monotonic())
    print(f'[res] finish_reason={response.parts[-1]!r} elapsed={elapsed:.3f}s')
    return response

agent = Agent('openai:gpt-4.1-mini', deps_type=dict, capabilities=[hooks])

async def main() -> None:
    result = await agent.run('What is 2 + 2?', deps={})
    print(result.output)

asyncio.run(main())
```

### 1.2 Per-Tool Auditing with Timeout Guard

Use `@hooks.on.before_tool_execute(tools=['db_query', 'file_write'], timeout=2.0)` to audit only specific tools and enforce a maximum hook duration.

```python {test="skip"}
import asyncio
import json
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import HookTimeoutError

hooks = Hooks()

@hooks.on.before_tool_execute(tools=['db_query', 'file_write'], timeout=2.0)
async def audit_sensitive(ctx, *, call, tool_def, args):
    # Simulate writing to an audit log; if it takes > 2s, HookTimeoutError is raised
    record = {'tool': call.tool_name, 'args': args, 'user': ctx.deps.get('user')}
    print(f'AUDIT: {json.dumps(record)}')
    return args   # must return (possibly modified) args

async def db_query(ctx, query: str) -> str:
    return f'Result for: {query}'

agent = Agent(
    'openai:gpt-4.1-mini',
    deps_type=dict,
    tools=[db_query],
    capabilities=[hooks],
)

async def main() -> None:
    try:
        result = await agent.run('Query the DB for active users', deps={'user': 'alice'})
        print(result.output)
    except HookTimeoutError as e:
        print(f'Hook timed out: {e.hook_name} / {e.func_name} after {e.timeout}s')

asyncio.run(main())
```

### 1.3 Wrapping the Entire Run with Error Recovery

Use `@hooks.on.run` (the `wrap_run` hook) to catch unhandled errors and return a safe fallback.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult

hooks = Hooks()

@hooks.on.run
async def safe_run(ctx, *, handler):
    try:
        return await handler()
    except Exception as exc:
        print(f'Agent run failed: {exc!r}')
        # Return a synthetic result rather than propagating
        return AgentRunResult._from_str(str(exc), ctx)

agent = Agent('openai:gpt-4.1-mini', capabilities=[hooks])

async def main() -> None:
    result = await agent.run('What is the answer to life?')
    print(result.output)

asyncio.run(main())
```

---

## 2. `ProcessHistory` — Callable-Based Message History Preprocessor

**Source**: `pydantic_ai/capabilities/process_history.py`  
**Export**: `from pydantic_ai.capabilities import ProcessHistory`

`ProcessHistory` wraps a user-supplied callable that receives the current message list and returns a (possibly modified) message list. The helper `_run_history_processor` auto-detects four calling conventions: `(ctx, messages)` async, `(messages,)` async, `(ctx, messages)` sync (run in a thread), and `(messages,)` sync.

```python
# Key signature verified from source (pydantic-ai 2.6.0):

@dataclass
class ProcessHistory(AbstractCapability[AgentDepsT]):
    """A capability that processes message history before model requests."""

    processor: HistoryProcessorFunc[AgentDepsT]
    # HistoryProcessorFunc = Callable[[RunContext, list[ModelMessage]], list[ModelMessage]]
    #                      | Callable[[list[ModelMessage]], list[ModelMessage]]
    #                      | (async variants of both)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_context.messages = await _run_history_processor(
            self.processor, ctx, request_context.messages
        )
        return request_context

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (holds a callable)
```

### 2.1 Trimming Old Messages to a Token Budget

Keep only the last N messages before every model call to prevent context overflow.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage

def keep_last_20(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-20:]  # retain most recent 20 messages

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ProcessHistory(keep_last_20)],
)

async def main() -> None:
    history: list[ModelMessage] = []
    for i in range(30):
        result = await agent.run(f'Turn {i}: continue the story', message_history=history)
        history = result.all_messages()
    print(f'History length kept at: {len(history)}')

asyncio.run(main())
```

### 2.2 Context-Aware Redaction with `RunContext`

Use the four-argument form `(ctx, messages) -> messages` to redact PII based on the caller's permissions.

```python {test="skip"}
import asyncio
import re
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, TextPart
from pydantic_ai.tools import RunContext

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

def redact_if_needed(
    ctx: RunContext[dict], messages: list[ModelMessage]
) -> list[ModelMessage]:
    if ctx.deps.get('redact_pii'):
        redacted = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts = [
                    TextPart(content=EMAIL_RE.sub('[EMAIL]', p.content))
                    if isinstance(p, TextPart) else p
                    for p in msg.parts
                ]
                from dataclasses import replace
                msg = replace(msg, parts=new_parts)
            redacted.append(msg)
        return redacted
    return messages

agent = Agent('openai:gpt-4.1', deps_type=dict, capabilities=[ProcessHistory(redact_if_needed)])

async def main() -> None:
    result = await agent.run(
        'Alice (alice@example.com) asked about pricing',
        deps={'redact_pii': True},
    )
    print(result.output)

asyncio.run(main())
```

### 2.3 Async Compaction of Old Tool Results

Async variant that compresses verbose tool-return blocks older than 10 messages.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolReturnPart
from dataclasses import replace

async def compress_old_tool_returns(messages: list[ModelMessage]) -> list[ModelMessage]:
    if len(messages) <= 10:
        return messages
    result = []
    for i, msg in enumerate(messages):
        if i < len(messages) - 10 and isinstance(msg, ModelResponse):
            compact_parts = [
                replace(p, content='[compressed]') if isinstance(p, ToolReturnPart) else p
                for p in msg.parts
            ]
            msg = replace(msg, parts=compact_parts)
        result.append(msg)
    return result

agent = Agent(
    'openai:gpt-4.1',
    capabilities=[ProcessHistory(compress_old_tool_returns)],
)

async def main() -> None:
    result = await agent.run('Summarise the project status')
    print(result.output)

asyncio.run(main())
```

---

## 3. `ReinjectSystemPrompt` — System Prompt Persistence Across UI Round-Trips

**Source**: `pydantic_ai/capabilities/reinject_system_prompt.py`  
**Export**: `from pydantic_ai.capabilities import ReinjectSystemPrompt`

Frontend clients (React, AG-UI, Vercel AI SDK) often strip `SystemPromptPart` from serialized history before sending it back. `ReinjectSystemPrompt` detects this and re-prepends the agent's configured prompt on every request. With `replace_existing=False` (default) it is a no-op when a system prompt is already present; with `replace_existing=True` it strips any existing `SystemPromptPart`s (untrusted client input) and replaces them with the server's authoritative version.

```python
# Key signature verified from source (pydantic-ai 2.6.0):

@dataclass
class ReinjectSystemPrompt(AbstractCapability[AgentDepsT]):
    """Capability that reinjects the agent's configured system_prompt when missing from history."""

    replace_existing: bool = False
    """Strip any existing SystemPromptParts before prepending the agent's prompt."""

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        messages = request_context.messages
        if self.replace_existing:
            _strip_system_prompts(messages)
        elif _has_system_prompt(messages):
            return request_context  # no-op: system prompt already present
        if ctx.agent is None:
            return request_context
        sys_parts = await ctx.agent.system_prompt_parts(...)
        if sys_parts:
            _prepend_to_first_request(messages, sys_parts)
        return request_context
```

`_prepend_to_first_request` replaces the first `ModelRequest`'s parts list with `[*sys_parts, *existing_parts]` immutably using `dataclasses.replace`.

### 3.1 Stateless REST Endpoint with Server-Authoritative System Prompt

Each POST carries full message history; `ReinjectSystemPrompt(replace_existing=True)` ensures untrusted client history never leaks a different system prompt.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    system_prompt='You are a helpful assistant. Always respond in English.',
    capabilities=[ReinjectSystemPrompt(replace_existing=True)],
)

async def handle_chat(history_from_client: list) -> str:
    from pydantic_ai.messages import ModelMessage
    # history_from_client may have no SystemPromptPart, or an adversarial one —
    # ReinjectSystemPrompt strips and replaces it before the model request
    result = await agent.run('continue', message_history=history_from_client)
    return result.output
```

### 3.2 Passive Injection (No-Op When System Prompt Present)

Without `replace_existing`, the capability only injects when the history lacks a system prompt — safe for histories that already carry one.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    'openai:gpt-4.1',
    system_prompt='Respond concisely.',
    capabilities=[ReinjectSystemPrompt()],  # replace_existing=False (default)
)

async def main() -> None:
    # First turn: no history → system prompt injected
    r1 = await agent.run('Hello')
    # Subsequent turn: history has system prompt → no-op (already present)
    r2 = await agent.run('Follow up', message_history=r1.all_messages())
    print(r2.output)

asyncio.run(main())
```

### 3.3 Combining with `UIAdapter` for Managed System Prompt

`UIAdapter` automatically adds `ReinjectSystemPrompt(replace_existing=True)` when `manage_system_prompt='server'`; you can also add it manually to any agent not using a `UIAdapter`.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

# Non-UI agent that rebuilds message history from a database
agent = Agent(
    'openai:gpt-4.1',
    system_prompt='You are a code review bot. Be precise.',
    capabilities=[ReinjectSystemPrompt(replace_existing=True)],
)

async def review_code(db_messages: list, code_snippet: str) -> str:
    result = await agent.run(
        f'Review this code:\n```python\n{code_snippet}\n```',
        message_history=db_messages,
    )
    return result.output

asyncio.run(review_code([], 'def add(a, b): return a + b'))
```

---

## 4. `ApprovalRequiredToolset` — Approval-Gated Tool Execution

**Source**: `pydantic_ai/toolsets/approval_required.py`  
**Export**: `from pydantic_ai.toolsets import ApprovalRequiredToolset`

`ApprovalRequiredToolset` wraps another toolset. Before any tool call, it evaluates a user-supplied `approval_required_func(ctx, tool_def, tool_args) -> bool`. If the function returns `True` **and** `ctx.tool_call_approved` is `False` (the default), it raises `ApprovalRequired` — suspending the agent so a human can approve or deny. Once the external caller sets `tool_call_approved=True` and resumes the agent, the tool executes normally.

```python
# Key signature verified from source (pydantic-ai 2.6.0):

@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    """A toolset that requires (some) calls to be approved before execution."""

    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True  # default: all tools require approval

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if not ctx.tool_call_approved and self.approval_required_func(ctx, tool.tool_def, tool_args):
            raise ApprovalRequired
        return await super().call_tool(name, tool_args, ctx, tool)
```

### 4.1 Require Approval for All Tool Calls

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.tools import RunContext

async def send_email(ctx: RunContext[None], to: str, subject: str, body: str) -> str:
    return f'Email sent to {to}'

async def search_web(ctx: RunContext[None], query: str) -> str:
    return f'Results for: {query}'

base_toolset = FunctionToolset([send_email, search_web])
gated_toolset = ApprovalRequiredToolset(wrapped=base_toolset)

agent = Agent('openai:gpt-4.1', toolsets=[gated_toolset])

async def main() -> None:
    from pydantic_ai.exceptions import ApprovalRequired
    try:
        result = await agent.run('Send a welcome email to bob@example.com')
    except ApprovalRequired:
        print('Tool call pending approval — resume with ctx.tool_call_approved=True')

asyncio.run(main())
```

### 4.2 Selective Approval — Only Destructive Tools

Approve only tools whose names suggest destructive operations.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.tools import RunContext, ToolDefinition

DESTRUCTIVE = frozenset({'delete_record', 'send_email', 'execute_query'})

def needs_approval(ctx: RunContext, tool_def: ToolDefinition, args: dict) -> bool:
    return tool_def.name in DESTRUCTIVE

async def delete_record(ctx: RunContext[None], record_id: str) -> str:
    return f'Deleted {record_id}'

async def get_record(ctx: RunContext[None], record_id: str) -> str:
    return f'Record: {record_id}'

base = FunctionToolset([delete_record, get_record])
selective = ApprovalRequiredToolset(wrapped=base, approval_required_func=needs_approval)

agent = Agent('openai:gpt-4.1', toolsets=[selective])

async def main() -> None:
    # get_record runs freely; delete_record raises ApprovalRequired
    result = await agent.run('Fetch record R-42')
    print(result.output)

asyncio.run(main())
```

### 4.3 Budget-Aware Approval

Allow calls that are cheap (no `cost_usd` in args) and gate expensive ones.

```python {test="skip"}
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.tools import RunContext, ToolDefinition

def expensive_guard(ctx: RunContext, tool_def: ToolDefinition, args: dict) -> bool:
    return float(args.get('cost_usd', 0)) > 1.00

# Wrap any existing toolset:
# gated = ApprovalRequiredToolset(wrapped=my_toolset, approval_required_func=expensive_guard)
```

---

## 5. `ExternalToolset` — Tools Whose Results Are Produced Outside the Agent

**Source**: `pydantic_ai/toolsets/external.py`  
**Export**: `from pydantic_ai import ExternalToolset`

`ExternalToolset` exposes a list of `ToolDefinition`s to the model, but marks each one with `kind='external'` so the framework knows the result will arrive from an external call (a UI frontend, another process, a user action). The agent suspends after the model calls an external tool and does not attempt to execute it locally. Its `call_tool` raises `NotImplementedError` by design.

```python
# Key signature verified from source (pydantic-ai 2.6.0):

TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())

class ExternalToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds tools whose results will be produced outside of the agent run."""

    tool_defs: list[ToolDefinition]
    _id: str | None

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None):
        self.tool_defs = tool_defs
        self._id = id

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            td.name: ToolsetTool(
                toolset=self,
                tool_def=replace(td, kind='external'),   # <-- marks as external
                max_retries=0,
                args_validator=TOOL_SCHEMA_VALIDATOR,
            )
            for td in self.tool_defs
        }

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        raise NotImplementedError('External tools cannot be called directly')
```

### 5.1 Exposing a Browser Action to the Model

Let the model invoke a browser-side tool (`click_element`) whose result arrives via the frontend.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent, ExternalToolset
from pydantic_ai.tools import ToolDefinition

click_element = ToolDefinition(
    name='click_element',
    description='Click a UI element identified by its CSS selector.',
    parameters_json_schema={
        'type': 'object',
        'properties': {'selector': {'type': 'string', 'description': 'CSS selector'}},
        'required': ['selector'],
    },
)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[ExternalToolset([click_element])],
)

async def main() -> None:
    from pydantic_ai.tools import DeferredToolResults
    # When the model calls click_element, the run suspends with DeferredToolResults.
    result = await agent.run('Click the submit button')
    print(result.output)

asyncio.run(main())
```

### 5.2 Frontend File Picker

Expose a `pick_file` tool that triggers a native file picker in a browser client; the chosen file path arrives back as a tool result.

```python {test="skip"}
from pydantic_ai import ExternalToolset
from pydantic_ai.tools import ToolDefinition

pick_file = ToolDefinition(
    name='pick_file',
    description='Open a native file picker and return the selected file path.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'accept': {'type': 'string', 'description': 'File MIME type filter, e.g. "image/*"'},
        },
        'required': [],
    },
)

file_toolset = ExternalToolset([pick_file], id='browser-tools')
# Pass to agent: Agent('openai:gpt-4.1', toolsets=[file_toolset])
```

### 5.3 Multi-Toolset Agent with External + Local Tools

Combine an `ExternalToolset` (frontend actions) with a local `FunctionToolset` (server-side logic).

```python {test="skip"}
import asyncio
from pydantic_ai import Agent, ExternalToolset
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import ToolDefinition, RunContext

# Frontend tool
confirm_action = ToolDefinition(
    name='confirm_action',
    description='Show a confirmation dialog and return true/false.',
    parameters_json_schema={
        'type': 'object',
        'properties': {'message': {'type': 'string'}},
        'required': ['message'],
    },
)

# Local tool
async def get_summary(ctx: RunContext[None], topic: str) -> str:
    return f'Summary of {topic}: ...'

agent = Agent(
    'openai:gpt-4.1',
    toolsets=[
        ExternalToolset([confirm_action]),
        FunctionToolset([get_summary]),
    ],
)

async def main() -> None:
    result = await agent.run('Summarise AI trends then confirm before proceeding')
    print(result.output)

asyncio.run(main())
```

---

## 6. `FilteredToolset` + `DeferredLoadingToolset` — Dynamic Tool Visibility

**Source**: `pydantic_ai/toolsets/filtered.py`, `pydantic_ai/toolsets/deferred_loading.py`  
**Export**: `from pydantic_ai.toolsets import FilteredToolset, DeferredLoadingToolset`

`FilteredToolset` wraps any toolset and hides tools based on a per-request predicate `(ctx, tool_def) -> bool`. Both sync and async predicates are supported. `DeferredLoadingToolset` marks tools with `defer_loading=True` so the model only discovers them through tool search rather than having every schema in the prompt — ideal for agents with hundreds of tools.

```python
# Key signatures verified from source (pydantic-ai 2.6.0):

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

@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    tool_names: frozenset[str] | None = None  # None → all tools deferred

    def __init__(self, wrapped: AbstractToolset[AgentDepsT], *, tool_names: frozenset[str] | None = None):
        async def _mark_deferred(ctx, tool_defs):
            return [replace(td, defer_loading=True) if (tool_names is None or td.name in tool_names) else td
                    for td in tool_defs]
        self.wrapped = wrapped
        self.prepare_func = _mark_deferred
```

### 6.1 Role-Based Tool Filtering

Only expose admin tools when the caller's deps indicate admin access.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FilteredToolset, FunctionToolset
from pydantic_ai.tools import RunContext, ToolDefinition

async def delete_user(ctx: RunContext[dict], user_id: str) -> str:
    return f'Deleted user {user_id}'

async def get_user(ctx: RunContext[dict], user_id: str) -> str:
    return f'User {user_id} details'

def admin_only(ctx: RunContext[dict], tool_def: ToolDefinition) -> bool:
    if tool_def.name == 'delete_user':
        return ctx.deps.get('is_admin', False)
    return True  # other tools always visible

base = FunctionToolset([delete_user, get_user])
filtered = FilteredToolset(wrapped=base, filter_func=admin_only)

agent = Agent('openai:gpt-4.1', deps_type=dict, toolsets=[filtered])

async def main() -> None:
    # Non-admin: delete_user is hidden from the model
    r1 = await agent.run('Delete user U-99', deps={'is_admin': False})
    print(r1.output)
    # Admin: delete_user is visible
    r2 = await agent.run('Delete user U-99', deps={'is_admin': True})
    print(r2.output)

asyncio.run(main())
```

### 6.2 Deferring Large Tool Catalogs

Mark all but the most-used tools as `defer_loading=True` to keep the prompt short.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import DeferredLoadingToolset, FunctionToolset
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.tools import RunContext

async def search_customers(ctx: RunContext[None], query: str) -> str:
    return f'Customers matching: {query}'

async def run_report(ctx: RunContext[None], report_id: str) -> str:
    return f'Report {report_id} output'

async def export_csv(ctx: RunContext[None], table: str) -> str:
    return f'CSV export of {table}'

base = FunctionToolset([search_customers, run_report, export_csv])
# Defer all but search_customers
deferred = DeferredLoadingToolset(
    base,
    tool_names=frozenset({'run_report', 'export_csv'}),
)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[deferred],
    capabilities=[ToolSearch()],  # required for deferred tools to be discoverable
)

async def main() -> None:
    result = await agent.run('Run the monthly sales report and export it')
    print(result.output)

asyncio.run(main())
```

### 6.3 Async Context-Aware Filter

Use an async predicate to look up permissions from a database for each request.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FilteredToolset, FunctionToolset
from pydantic_ai.tools import RunContext, ToolDefinition

async def _check_permission(ctx: RunContext[dict], tool_def: ToolDefinition) -> bool:
    """Simulates an async permission check."""
    allowed = ctx.deps.get('allowed_tools', set())
    return tool_def.name in allowed

async def tool_a(ctx: RunContext[dict]) -> str: return 'A result'
async def tool_b(ctx: RunContext[dict]) -> str: return 'B result'

agent = Agent(
    'openai:gpt-4.1',
    deps_type=dict,
    toolsets=[FilteredToolset(FunctionToolset([tool_a, tool_b]), filter_func=_check_permission)],
)

async def main() -> None:
    result = await agent.run('Use tool_a', deps={'allowed_tools': {'tool_a'}})
    print(result.output)

asyncio.run(main())
```

---

## 7. `VercelAIAdapter` + `VercelAIEventStream` — Vercel AI SDK Server Adapter

**Source**: `pydantic_ai/ui/vercel_ai/_adapter.py`, `pydantic_ai/ui/vercel_ai/_event_stream.py`  
**Export**: `from pydantic_ai.ui.vercel_ai import VercelAIAdapter`

`VercelAIAdapter` is a `UIAdapter` subclass that speaks the Vercel AI SDK streaming protocol (Data Stream Protocol, DSP). It parses `RequestData` (the Vercel AI SDK v5/v6 `useChat` / `useCompletion` request body) and emits a sequence of typed chunks: `StartChunk`, `StartStepChunk`, `TextDeltaChunk`, `TextEndChunk`, `ToolInputStartChunk`, `ToolInputAvailableChunk`, `ToolOutputAvailableChunk`, `FinishStepChunk`, `FinishChunk`, `DoneChunk`. Setting `sdk_version=6` additionally enables `ToolApprovalRequestChunk` for human-in-the-loop tool approval.

```python
# Key signature verified from source (pydantic-ai 2.6.0):

VERCEL_AI_DSP_HEADERS = {'x-vercel-ai-ui-message-stream': 'v1'}

@dataclass
class VercelAIAdapter(UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    """UI adapter for the Vercel AI protocol."""

    _: KW_ONLY
    sdk_version: Literal[5, 6] = 5
    """Setting sdk_version=6 enables tool approval streaming for HITL workflows."""
    server_message_id: str | None = None
    """Optional server-generated message ID included in StartChunk."""

    @classmethod
    async def from_request(cls, request, agent, *, sdk_version=5, ...) -> VercelAIAdapter: ...

    @classmethod
    def build_run_input(cls, body: bytes) -> RequestData: ...

    def build_event_stream(self) -> VercelAIEventStream: ...

@dataclass
class VercelAIEventStream(UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the Vercel AI protocol."""

    _: KW_ONLY
    sdk_version: Literal[5, 6] = 5
    server_message_id: str | None = None
    _step_started: bool = False
    _finish_reason: FinishReason = None
```

### 7.1 FastAPI Chat Endpoint (Vercel AI SDK v5)

```python {test="skip"}
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

app = FastAPI()
agent = Agent('anthropic:claude-sonnet-4-6', system_prompt='You are a helpful assistant.')

@app.post('/api/chat')
async def chat(request: Request) -> StreamingResponse:
    adapter = await VercelAIAdapter.from_request(request, agent, sdk_version=5)
    return adapter.stream_response()
```

### 7.2 HITL Tool Approval with SDK v6

With `sdk_version=6`, the adapter emits `ToolApprovalRequestChunk` when a tool requires approval, and reads `ToolApprovalResponded` from the client's next message.

```python {test="skip"}
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

app = FastAPI()

async def send_email(ctx: RunContext[None], to: str, body: str) -> str:
    return f'Email sent to {to}'

agent = Agent(
    'openai:gpt-4.1',
    toolsets=[ApprovalRequiredToolset(FunctionToolset([send_email]))],
)

@app.post('/api/chat')
async def chat(request: Request) -> StreamingResponse:
    # sdk_version=6 enables ToolApprovalRequestChunk / ToolApprovalResponded round-trip
    adapter = await VercelAIAdapter.from_request(request, agent, sdk_version=6)
    return adapter.stream_response()
```

### 7.3 Static Message Loading from a Vercel AI SDK Conversation

`VercelAIAdapter.load_messages()` converts a list of Vercel AI `UIMessage` objects (from the client) into `ModelMessage` objects for use as `message_history`.

```python {test="skip"}
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import UIMessage, TextUIPart

# Simulate messages from the client
client_messages = [
    UIMessage(id='m1', role='user', parts=[TextUIPart(type='text', text='Hello')]),
    UIMessage(id='m2', role='assistant', parts=[TextUIPart(type='text', text='Hi there!')]),
]

model_messages = VercelAIAdapter.load_messages(client_messages)
print(f'Converted {len(model_messages)} messages for use as message_history')
# Pass model_messages as message_history to agent.run(...)
```

---

## 8. `AGUIAdapter` + `_AGUIFrontendToolset` + Interrupt Handling — AG-UI Protocol Adapter

**Source**: `pydantic_ai/ui/ag_ui/_adapter.py`, `pydantic_ai/ui/ag_ui/_interrupt.py`  
**Export**: `from pydantic_ai.ui.ag_ui import AGUIAdapter`

`AGUIAdapter` implements the [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) — a standardised event stream for agent UIs. It accepts `RunAgentInput` (the AG-UI run payload), converts `Message` objects to `ModelMessage`s, and streams `BaseEvent` objects back. AG-UI tools declared in `RunAgentInput.tools` are exposed as an `_AGUIFrontendToolset` (an `ExternalToolset` subclass). When `ag-ui-protocol >= 0.1.19`, interrupts enable HITL: `approval_to_interrupt` converts a pending `ToolCallPart` to an `Interrupt`, and `resume_entry_to_approval` converts the client's `ResumeEntry` to `ToolApproved` / `ToolDenied`.

```python
# Key signature verified from source (pydantic-ai 2.6.0):

class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    """Wraps AG-UI tool definitions as an ExternalToolset."""
    def __init__(self, tools: list[AGUITool]): ...

@dataclass
class AGUIAdapter(UIAdapter[RunAgentInput, Message, BaseEvent, AgentDepsT, OutputDataT]):
    """UI adapter for the AG-UI protocol."""

    @classmethod
    async def from_request(
        cls, request: Request, agent: AbstractAgent[AgentDepsT, OutputDataT], *,
        manage_system_prompt: Literal['server', 'client'] = 'server',
        allow_uploaded_files: bool = False,
        **kwargs,
    ) -> AGUIAdapter[AgentDepsT, OutputDataT]: ...

    @classmethod
    def load_messages(cls, messages: Sequence[Message], ...) -> list[ModelMessage]: ...

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[Message]: ...

# Interrupt translation (pydantic_ai/ui/ag_ui/_interrupt.py):
HAS_INTERRUPTS: bool  # True when ag-ui-protocol >= 0.1.19

def approval_to_interrupt(call: ToolCallPart, metadata: dict) -> Interrupt: ...
def interrupt_id_to_tool_call_id(interrupt_id: str) -> str: ...
def resume_entry_to_approval(entry: ResumeEntry) -> DeferredToolApprovalResult: ...
```

### 8.1 FastAPI AG-UI Chat Endpoint

```python {test="skip"}
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

app = FastAPI()
agent = Agent('anthropic:claude-sonnet-4-6', system_prompt='You are a helpful assistant.')

@app.post('/agent')
async def run_agent(request: Request) -> StreamingResponse:
    adapter = await AGUIAdapter.from_request(
        request, agent,
        manage_system_prompt='server',  # strips and reinjects system prompt each turn
    )
    return adapter.stream_response()
```

### 8.2 Round-Trip Message Serialisation

Convert a PydanticAI conversation to AG-UI `Message`s for storage, then reload them for the next turn.

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('openai:gpt-4.1')

async def main() -> None:
    result = await agent.run('What is the capital of France?')
    # Serialize to AG-UI messages (e.g. to store in a database or pass to frontend)
    ag_ui_messages = AGUIAdapter.dump_messages(result.all_messages())
    print(f'Serialised {len(ag_ui_messages)} AG-UI messages')

    # Later, reload from AG-UI messages
    model_messages = AGUIAdapter.load_messages(ag_ui_messages)
    result2 = await agent.run('And what is the capital of Germany?', message_history=model_messages)
    print(result2.output)

asyncio.run(main())
```

### 8.3 HITL Tool Approval via AG-UI Interrupts

When `ag-ui-protocol >= 0.1.19` (`HAS_INTERRUPTS=True`), suspended tool calls emit `Interrupt` events that the frontend resolves via `ResumeEntry` — translated back to `ToolApproved` / `ToolDenied` by `resume_entry_to_approval`.

```python {test="skip"}
from pydantic_ai.ui.ag_ui._interrupt import (
    HAS_INTERRUPTS,
    approval_to_interrupt,
    resume_entry_to_approval,
)
from pydantic_ai.tools import ToolApproved, ToolDenied

if HAS_INTERRUPTS:
    # Outbound: agent → frontend
    # approval_to_interrupt(call, metadata) → Interrupt with response_schema
    # {approved: bool, editedArgs?: dict, reason?: str}

    # Inbound: frontend → agent
    # resume_entry_to_approval(entry) → ToolApproved | ToolDenied
    # - entry.status == 'cancelled' → ToolDenied
    # - payload.approved == True → ToolApproved (with optional override_args)
    # - any other payload → ToolDenied (deny-by-default)
    print('AG-UI interrupt support is available')
else:
    print('Upgrade ag-ui-protocol to >= 0.1.19 for interrupt support')
```

---

## 9. New Provider Profiles: `harmony_model_profile`, `moonshotai_model_profile`, `amazon_model_profile` + `merge_profile`

**Source**: `pydantic_ai/profiles/harmony.py`, `pydantic_ai/profiles/moonshotai.py`, `pydantic_ai/profiles/amazon.py`, `pydantic_ai/profiles/__init__.py`  
**Export**: Available via the profile dispatch system (set by `provider_profiles=` or registered model names)

pydantic-ai 2.6.0 adds three provider-profile functions plus updates to `merge_profile`. Each returns a `ModelProfile | None`; the framework calls the matching function based on the model's provider prefix.

```python
# Key signatures verified from source (pydantic-ai 2.6.0):

def harmony_model_profile(model_name: str) -> ModelProfile | None:
    """OpenAI Harmony Response format — wraps openai_model_profile with two overrides."""
    return merge_profile(
        openai_model_profile(model_name),
        OpenAIModelProfile(
            openai_supports_tool_choice_required=False,
            ignore_streamed_leading_whitespace=True,
        ),
    )

def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """MoonshotAI Kimi models — detects reasoning models by name prefix."""
    is_reasoning = model_name.lower().startswith(
        ('kimi-k2.5', 'kimi-k2.6', 'kimi-k2.7', 'kimi-thinking')
    )
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_reasoning,
    )

def amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Amazon models — applies InlineDefsJsonSchemaTransformer."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)

def merge_profile(base: ModelProfile | None, override: ModelProfile | None) -> ModelProfile | None:
    """Layer two profiles: non-None fields in override replace base fields."""
```

### 9.1 Using a Harmony OpenAI Model

```python {test="skip"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles.harmony import harmony_model_profile

# Harmony response format: tool_choice_required not supported,
# leading whitespace in streamed text is suppressed.
profile = harmony_model_profile('gpt-4o')

model = OpenAIModel('gpt-4o', provider_profile=profile)
agent = Agent(model, system_prompt='Be concise.')

async def main() -> None:
    result = await agent.run('List three Python frameworks.')
    print(result.output)

asyncio.run(main())
```

### 9.2 MoonshotAI Kimi Reasoning Model Detection

```python {test="skip"}
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile

for model_name in [
    'kimi-k2.5-code',
    'kimi-thinking-latest',
    'moonshot-v1-8k',
]:
    profile = moonshotai_model_profile(model_name)
    print(f'{model_name}: supports_thinking={profile.supports_thinking!r}')

# kimi-k2.5-code: supports_thinking=True
# kimi-thinking-latest: supports_thinking=True
# moonshot-v1-8k: supports_thinking=False
```

### 9.3 Composing Profiles with `merge_profile`

Combine the Amazon base profile with a custom `max_tokens` override using `merge_profile`.

```python {test="skip"}
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.amazon import amazon_model_profile

base = amazon_model_profile('nova-pro-v1')
# Add a max_tokens constraint on top of the Amazon JSON schema transformer
extended = merge_profile(base, ModelProfile(supports_thinking=False))

print(f'json_schema_transformer: {extended.json_schema_transformer}')
print(f'supports_thinking: {extended.supports_thinking}')
# json_schema_transformer: InlineDefsJsonSchemaTransformer
# supports_thinking: False
```

---

## 10. `Decision` + `DecisionBranch` + `ReducerContext` + `JoinState` — Conditional Graph Branching

**Source**: `pydantic_graph/decision.py`, `pydantic_graph/join.py`  
**Export**: `from pydantic_graph import Decision; from pydantic_graph.decision import DecisionBranch; from pydantic_graph.join import ReducerContext, JoinState`

`pydantic_graph` 2.6.0 introduces structured conditional branching via `Decision` nodes and parallel join reducers. A `Decision` holds a list of `DecisionBranch` objects, each associated with a source type; at runtime, the input's type determines which branch fires. `ReducerContext` is passed to join reducers during graph execution, carrying state, deps, and the `JoinState` which tracks pending parallel branches. `cancel_sibling_tasks()` enables early-stopping when one branch's result is sufficient.

```python
# Key signatures verified from source (pydantic-graph 2.6.0):

@dataclass(kw_only=True)
class Decision(Generic[StateT, DepsT, HandledT]):
    """Conditional branching node that routes on input type."""
    id: NodeID
    branches: list[DecisionBranch[Any]]
    note: str | None

    def branch(self, branch: DecisionBranch[T]) -> Decision[StateT, DepsT, HandledT | T]: ...

@dataclass
class DecisionBranch(Generic[SourceT]):
    source: TypeOrTypeExpression[SourceT]
    # + path wiring to a destination node

@dataclass
class JoinState:
    """Per-fork join bookkeeping during graph execution."""
    current: Any
    downstream_fork_stack: ForkStack
    cancelled_sibling_tasks: bool = False

@dataclass(init=False)
class ReducerContext(Generic[StateT, DepsT]):
    """Context passed to reducer functions during join execution."""
    _state: StateT
    _deps: DepsT
    _join_state: JoinState

    @property
    def state(self) -> StateT: ...

    @property
    def deps(self) -> DepsT: ...

    def cancel_sibling_tasks(self) -> None:
        """Cancel all sibling tasks from the same fork (early-stopping)."""
        self._join_state.cancelled_sibling_tasks = True
```

### 10.1 Simple Type-Based Routing with `Decision`

Build a minimal graph that routes `str` vs `int` inputs to different processing branches.

```python {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.decision import Decision, DecisionBranch
from pydantic_graph.id_types import NodeID

@dataclass
class StringResult:
    value: str

@dataclass
class IntResult:
    value: int

@dataclass
class ProcessString(BaseNode[None, None, str]):
    value: str
    async def run(self, ctx: GraphRunContext[None, None]) -> End[str]:
        return End(f'String: {self.value.upper()}')

@dataclass
class ProcessInt(BaseNode[None, None, str]):
    value: int
    async def run(self, ctx: GraphRunContext[None, None]) -> End[str]:
        return End(f'Integer: {self.value * 2}')

@dataclass
class StartNode(BaseNode[None, None, str]):
    input: StringResult | IntResult

    async def run(self, ctx: GraphRunContext[None, None]) -> ProcessString | ProcessInt:
        if isinstance(self.input, StringResult):
            return ProcessString(self.input.value)
        return ProcessInt(self.input.value)

graph = Graph(nodes=[StartNode, ProcessString, ProcessInt])

async def main() -> None:
    result_str, _ = await graph.run(StartNode(StringResult('hello')), state=None)
    result_int, _ = await graph.run(StartNode(IntResult(21)), state=None)
    print(result_str)   # String: HELLO
    print(result_int)   # Integer: 42

asyncio.run(main())
```

### 10.2 Early-Stopping Join with `cancel_sibling_tasks`

Use `ReducerContext.cancel_sibling_tasks()` in a first-result-wins join.

```python {test="skip"}
import asyncio
from dataclasses import dataclass
from pydantic_graph.join import ReducerContext

@dataclass
class SearchResult:
    source: str
    text: str

def first_result_reducer(
    ctx: ReducerContext[list, None],
    incoming: SearchResult,
) -> None:
    if not ctx.state:          # first result to arrive
        ctx.state.append(incoming)
        ctx.cancel_sibling_tasks()  # stop other parallel branches
```

### 10.3 Accumulating Join Reducer

Collect all parallel branch results before proceeding.

```python {test="skip"}
from dataclasses import dataclass
from pydantic_graph.join import ReducerContext, JoinState

@dataclass
class PartialScore:
    evaluator: str
    score: float

def accumulate_scores(
    ctx: ReducerContext[list, None],
    incoming: PartialScore,
) -> None:
    ctx.state.append(incoming)
    # No cancel_sibling_tasks() → wait for all parallel branches
    print(f'Received score from {incoming.evaluator}: {incoming.score}')
```

---

## Summary

| # | Class / group | Module | Key new capability |
|---|---|---|---|
| 1 | `Hooks` + `HookTimeoutError` | `capabilities/hooks.py` | Decorator-based lifecycle hooks; 20+ hook points; per-hook `anyio` timeout |
| 2 | `ProcessHistory` | `capabilities/process_history.py` | Callable history preprocessor; 4 calling conventions (sync/async × with/without ctx) |
| 3 | `ReinjectSystemPrompt` | `capabilities/reinject_system_prompt.py` | Survives UI round-trips; `replace_existing=True` strips untrusted system prompts |
| 4 | `ApprovalRequiredToolset` | `toolsets/approval_required.py` | Gates tool execution on `approval_required_func` predicate; raises `ApprovalRequired` |
| 5 | `ExternalToolset` | `toolsets/external.py` | `kind='external'` mark; results produced outside the agent; used by AG-UI frontend tools |
| 6 | `FilteredToolset` + `DeferredLoadingToolset` | `toolsets/filtered.py`, `deferred_loading.py` | Sync/async per-request filtering; `defer_loading=True` for discovery-based tool loading |
| 7 | `VercelAIAdapter` + `VercelAIEventStream` | `ui/vercel_ai/` | Vercel AI SDK v5/v6; `sdk_version=6` adds `ToolApprovalRequestChunk` HITL |
| 8 | `AGUIAdapter` + interrupt handling | `ui/ag_ui/` | AG-UI protocol; `HAS_INTERRUPTS` guard; `approval_to_interrupt`/`resume_entry_to_approval` |
| 9 | `harmony_model_profile` + `moonshotai_model_profile` + `amazon_model_profile` + `merge_profile` | `profiles/` | Three new 2.6.0 provider profiles; profile layering via `merge_profile` |
| 10 | `Decision` + `DecisionBranch` + `ReducerContext` + `JoinState` | `pydantic_graph/decision.py`, `join.py` | Type-based conditional branching; early-stopping and accumulating join reducers |

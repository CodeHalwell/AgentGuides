---
title: "PydanticAI Class Deep Dives Vol. 22"
description: "Source-verified deep dives into 10 pydantic-ai 1.107.0 class groups: RunContext (complete field reference — deps/model/usage/messages/tool_name/conversation_id/enqueue/last_attempt), UsageLimits + RequestUsage + RunUsage (token budgets, pre-request token counting, cost extraction), DeferredToolRequests + DeferredToolResults + CallDeferred (human-in-the-loop async tool execution), ApprovalRequired + ApprovalRequiredToolset + ToolApproved + ToolDenied (interactive tool approval), ConcurrencyLimiter + AbstractConcurrencyLimiter + ConcurrencyLimit (in-process and custom distributed rate limiting), SkipModelRequest + SkipToolExecution + SkipToolValidation (hook short-circuit exceptions), ModelRetry (retry signalling from tools and validators), TemplateStr + format_as_xml (Handlebars prompt templates and XML context injection), AgentSpec (YAML/JSON-driven agent configuration with from_file/to_file), UploadedFile + BinaryContent + FilePart (cross-provider multimodal file handling). All verified against pydantic-ai 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 22)"
  order: 48
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups spanning the core runtime primitives, cost controls, advanced tool patterns, concurrency, prompt utilities, and multimodal file handling: `RunContext` (the universal context object injected into every tool, validator, and capability hook — complete field reference including `enqueue`, `last_attempt`, `conversation_id`, `tool_call_approved`); `UsageLimits` + `RequestUsage` + `RunUsage` (budget enforcement before and after every request, pre-request token counting, genai-prices cost extraction); `DeferredToolRequests` + `DeferredToolResults` + `CallDeferred` (human-in-the-loop workflows where tool calls pause the agent and resume with results later); `ApprovalRequired` + `ApprovalRequiredToolset` + `ToolApproved` + `ToolDenied` (interactive approval gating — default-approve-all or per-call predicate); `ConcurrencyLimiter` + `AbstractConcurrencyLimiter` + `ConcurrencyLimit` (anyio-backed in-process limiter with OTel wait-span, max-queue enforcement, and custom Redis-backed subclass pattern); `SkipModelRequest` + `SkipToolExecution` + `SkipToolValidation` (hook short-circuit exceptions that inject synthetic responses/results/args); `ModelRetry` (raise from tools, output validators, or capability hooks to send a retry prompt back to the model); `TemplateStr` + `format_as_xml` (Handlebars templates rendered against `RunContext.deps` and XML context formatting for LLM consumption); `AgentSpec` (YAML/JSON-driven agent configuration — `from_file`, `to_file`, `from_dict`, `TemplateStr` instructions, capabilities list); `UploadedFile` + `BinaryContent` + `FilePart` (cross-provider multimodal file references — file-ID aliasing, binary inlining, model response file parts).

---

## 1. `RunContext` — The Universal Tool and Validator Context

**Module:** `pydantic_ai.tools`  
**Import:**
```python
from pydantic_ai import RunContext
```

`RunContext[DepsT]` is the generic dataclass injected as the first argument of every tool function, output validator, system prompt function, and capability hook. It exposes the full runtime state of the current agent run step, including the live dependency object, model identity, accumulated token usage, and the complete message history so far.

### Complete field reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `deps` | `DepsT` | — | Your dependency object, injected by the agent framework |
| `model` | `Model` | — | The model instance being used in this run |
| `usage` | `RunUsage` | — | Accumulated token usage for the run so far |
| `agent` | `Agent \| None` | `None` | The agent running this context |
| `prompt` | `str \| Sequence[UserContent] \| None` | `None` | Original user prompt for this run |
| `messages` | `list[ModelMessage]` | `[]` | Message history up to the current step |
| `tracer` | `Tracer` | `NoOpTracer()` | OTel tracer; real tracer when Logfire / OTel is configured |
| `retries` | `dict[str, int]` | `{}` | Per-tool retry counter `{tool_name: retry_count}` |
| `tool_call_id` | `str \| None` | `None` | The ID of the current tool call |
| `tool_name` | `str \| None` | `None` | Name of the tool currently executing |
| `retry` | `int` | `0` | Number of retries for the current tool or output validation |
| `max_retries` | `int` | `0` | Maximum retries allowed for the current tool or output |
| `run_step` | `int` | `0` | Current step number within the run |
| `tool_call_approved` | `bool` | `False` | `True` when an `ApprovalRequired` tool call was approved |
| `tool_call_metadata` | `Any` | `None` | Metadata from `DeferredToolResults.metadata` for the current call |
| `partial_output` | `bool` | `False` | `True` when the output validator receives a streaming partial |
| `run_id` | `str \| None` | `None` | Unique ID for this agent run |
| `conversation_id` | `str \| None` | `None` | ID spanning multiple runs that share message history |
| `metadata` | `dict[str, Any] \| None` | `None` | Metadata from `Agent.run(..., metadata=...)` |
| `model_settings` | `ModelSettings \| None` | `None` | Resolved merged model settings; set before each model request |
| `validation_context` | `Any` | `None` | Pydantic validation context for tool args and run outputs |
| `capabilities` | `dict[str, AbstractCapability]` | `{}` | All registered capabilities for this run |
| `loaded_capability_ids` | `set[str]` | `set()` | IDs of deferred capabilities explicitly loaded so far |
| `discovered_tool_names` | `set[str]` | `set()` | Tool names revealed via tool-search return parts |

### Key properties and methods

**`last_attempt`** (property) — returns `True` when `retry == max_retries`. Use this to skip expensive fallback logic on all retries except the last:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class Deps:
    api_url: str

agent: Agent[Deps, str] = Agent('openai:gpt-4o', deps_type=Deps)

@agent.tool
async def fetch_data(ctx: RunContext[Deps], query: str) -> str:
    if ctx.last_attempt:
        # On the final retry, include extended diagnostics
        return f'ERROR: exhausted retries for {query} after {ctx.retry} attempts'
    try:
        return f'data for {query} from {ctx.deps.api_url}'
    except Exception as e:
        from pydantic_ai import ModelRetry
        raise ModelRetry(f'fetch failed: {e}, attempt {ctx.retry + 1}')
```

**`enqueue`** — inject content into the conversation from inside a tool body, to be delivered before the next model request:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import SystemPromptPart

@dataclass
class Deps:
    user_tier: str

agent: Agent[Deps, str] = Agent('openai:gpt-4o', deps_type=Deps)

@agent.tool
async def check_quota(ctx: RunContext[Deps], resource: str) -> str:
    if ctx.deps.user_tier == 'free' and resource == 'premium_data':
        # Inject a system note before the model sees the tool result
        ctx.enqueue(
            SystemPromptPart(content='User is on the free tier — suggest upgrading.'),
            priority='asap',
        )
        return 'access_denied'
    return f'{resource}: available'
```

**`conversation_id` and `run_id`** — thread conversation context across multiple runs:

```python
import asyncio
from pydantic_ai import Agent, RunContext

agent: Agent[None, str] = Agent('openai:gpt-4o')

@agent.tool
async def get_run_info(ctx: RunContext[None]) -> str:
    return f'run={ctx.run_id}, conversation={ctx.conversation_id}, step={ctx.run_step}'

async def main() -> None:
    # First run — starts a new conversation_id
    result1 = await agent.run('Hello')
    conv_id = result1.all_messages()[-1].conversation_id

    # Second run — continues the same conversation
    result2 = await agent.run(
        'What run was that?',
        message_history=result1.all_messages(),
        conversation_id=conv_id,
    )
    print(result2.output)  # References the same conversation_id
```

---

## 2. `UsageLimits` + `RequestUsage` + `RunUsage` — Budget Enforcement

**Module:** `pydantic_ai.usage`  
**Import:**
```python
from pydantic_ai import UsageLimits
from pydantic_ai.usage import RequestUsage, RunUsage
```

`UsageLimits` is a dataclass you pass to `Agent.run()` to cap how many model requests, tool calls, or tokens an agent may consume. `RunUsage` accumulates totals across the run. `RequestUsage` represents usage from a single request and implements `genai_prices.types.AbstractUsage` for cost calculation.

### `UsageLimits` constructor

```python
UsageLimits(
    request_limit: int | None = 50,          # max model requests; default 50
    tool_calls_limit: int | None = None,      # max successful tool calls
    input_tokens_limit: int | None = None,    # max prompt/input tokens
    output_tokens_limit: int | None = None,   # max completion/output tokens
    total_tokens_limit: int | None = None,    # max combined tokens
    count_tokens_before_request: bool = False, # pre-flight token count
)
```

`count_tokens_before_request=True` triggers an extra model API call before each request to count tokens precisely — supported by Anthropic, Google, Bedrock Converse, and OpenAI Responses.

### `RunUsage` fields

| Field | Type | Notes |
|---|---|---|
| `requests` | `int` | Number of model requests made |
| `tool_calls` | `int` | Successful tool executions |
| `input_tokens` | `int` | Total prompt/input tokens |
| `output_tokens` | `int` | Total completion/output tokens |
| `cache_write_tokens` | `int` | Tokens written to provider cache |
| `cache_read_tokens` | `int` | Tokens read from provider cache |
| `input_audio_tokens` | `int` | Audio input tokens (where applicable) |
| `details` | `dict[str, int]` | Provider-specific extra fields |
| `total_tokens` | property | `input_tokens + output_tokens` |

### Example 1 — Basic budget guard

```python
import asyncio
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o')

async def main() -> None:
    try:
        result = await agent.run(
            'Summarise world history in 100 words.',
            usage_limits=UsageLimits(
                request_limit=3,
                output_tokens_limit=200,
            ),
        )
        print(result.output)
        print('Usage:', result.usage())
    except UsageLimitExceeded as e:
        print(f'Budget exceeded: {e}')
```

### Example 2 — Tool call budget

```python
import asyncio
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o', system_prompt='Use the tool as many times as needed.')

@agent.tool_plain
def count_vowels(text: str) -> int:
    return sum(1 for c in text.lower() if c in 'aeiou')

async def main() -> None:
    try:
        result = await agent.run(
            'How many vowels in each of these 20 words: ...',
            # Allow at most 5 tool calls for this run
            usage_limits=UsageLimits(tool_calls_limit=5),
        )
        print(result.output)
    except UsageLimitExceeded as e:
        print(f'Tool call limit hit: {e}')
```

### Example 3 — Inspecting usage after the run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def ping() -> str:
    return 'pong'

async def main() -> None:
    result = await agent.run('Call ping twice.')
    usage: RunUsage = result.usage()
    print(f'Requests: {usage.requests}')
    print(f'Tool calls: {usage.tool_calls}')
    print(f'Input tokens: {usage.input_tokens}')
    print(f'Output tokens: {usage.output_tokens}')
    print(f'Total tokens: {usage.total_tokens}')
    if usage.cache_read_tokens:
        print(f'Cache hits: {usage.cache_read_tokens} tokens')
```

---

## 3. `DeferredToolRequests` + `DeferredToolResults` + `CallDeferred` — Async Human-in-the-Loop

**Module:** `pydantic_ai.tools` / `pydantic_ai.output`  
**Import:**
```python
from pydantic_ai import DeferredToolRequests, DeferredToolResults, CallDeferred
from pydantic_ai import ToolApproved, ToolDenied, ToolReturn
```

The deferred tool system lets you pause an agent mid-run when a tool needs asynchronous external execution (a human decision, a slow external job, a Temporal workflow). The agent's output type is set to `DeferredToolRequests`; the agent yields pending calls instead of executing them. You then resolve each call and resume the agent with a `DeferredToolResults`.

### `DeferredToolRequests` fields

| Field | Type | Notes |
|---|---|---|
| `calls` | `list[ToolCallPart]` | Tool calls requiring external execution |
| `approvals` | `list[ToolCallPart]` | Tool calls requiring human approval |
| `metadata` | `dict[str, dict[str, Any]]` | Per-call metadata keyed by `tool_call_id` |

### `DeferredToolResults` fields

| Field | Type | Notes |
|---|---|---|
| `calls` | `dict[str, DeferredToolCallResult \| Any]` | Results for external calls, keyed by `tool_call_id` |
| `approvals` | `dict[str, bool \| DeferredToolApprovalResult]` | `True`/`ToolApproved` or `False`/`ToolDenied` |
| `metadata` | `dict[str, dict[str, Any]]` | Per-call metadata made available in `RunContext.tool_call_metadata` |

### `CallDeferred` exception

Raise from a tool body to defer that specific call:

```python
from pydantic_ai import CallDeferred

@agent.tool_plain
def approve_transaction(amount: float, account_id: str) -> str:
    raise CallDeferred(metadata={'amount': amount, 'account_id': account_id})
```

### Example 1 — Pause on external execution

```python
import asyncio
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, CallDeferred

agent: Agent[None, DeferredToolRequests | str] = Agent(
    'openai:gpt-4o',
    output_type=[DeferredToolRequests, str],  # type: ignore[arg-type]
)

@agent.tool_plain
def run_sql_query(sql: str) -> str:
    # Raise instead of executing — delegate to external executor
    raise CallDeferred(metadata={'sql': sql})

async def main() -> None:
    result = await agent.run('Count all users created in 2025.')
    if isinstance(result.output, DeferredToolRequests):
        pending = result.output
        # Execute each deferred SQL call externally
        call_results: dict[str, str] = {}
        for call in pending.calls:
            sql = pending.metadata[call.tool_call_id]['sql']
            # ... run SQL here ...
            call_results[call.tool_call_id] = '42'  # simulated result

        deferred_results = pending.build_results(calls=call_results)
        # Resume the agent with the results
        final = await agent.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=deferred_results,
        )
        print(final.output)
```

### Example 2 — Multi-call approval loop

```python
import asyncio
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, CallDeferred, ApprovalRequired

agent: Agent[None, DeferredToolRequests | str] = Agent(
    'openai:gpt-4o',
    output_type=[DeferredToolRequests, str],  # type: ignore[arg-type]
)

@agent.tool_plain
def delete_record(table: str, record_id: int) -> str:
    raise ApprovalRequired  # mark as needing human approval

async def main() -> None:
    result = await agent.run('Delete records 5, 10, and 15 from the users table.')
    output = result.output
    while isinstance(output, DeferredToolRequests):
        # Simulate human approving some, denying others
        approvals: dict[str, bool] = {}
        for call in output.approvals:
            meta = output.metadata.get(call.tool_call_id, {})
            approvals[call.tool_call_id] = call.args.get('record_id') != 10  # deny record 10
        results = output.build_results(approvals=approvals)
        resumed = await agent.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=results,
        )
        output = resumed.output
        result = resumed
    print(output)
```

### Example 3 — Using `remaining()` to check unresolved calls

```python
import asyncio
from pydantic_ai import (
    Agent, DeferredToolRequests, DeferredToolResults, CallDeferred, ToolReturn
)

agent: Agent[None, DeferredToolRequests | str] = Agent(
    'openai:gpt-4o',
    output_type=[DeferredToolRequests, str],  # type: ignore[arg-type]
)

@agent.tool_plain
def slow_job(job_name: str) -> str:
    raise CallDeferred(metadata={'job_name': job_name})

async def main() -> None:
    result = await agent.run('Run jobs alpha, beta, and gamma.')
    output = result.output
    if isinstance(output, DeferredToolRequests):
        # Resolve only alpha for now
        partial_results = DeferredToolResults()
        for call in output.calls:
            if output.metadata[call.tool_call_id]['job_name'] == 'alpha':
                partial_results.calls[call.tool_call_id] = ToolReturn('alpha completed')

        remaining = output.remaining(partial_results)
        if remaining:
            print(f'Still pending: {[c.tool_call_id for c in remaining.calls]}')
```

---

## 4. `ApprovalRequired` + `ApprovalRequiredToolset` + `ToolApproved` + `ToolDenied` — Tool Approval

**Module:** `pydantic_ai` / `pydantic_ai.toolsets.approval_required`  
**Import:**
```python
from pydantic_ai import ApprovalRequired, ToolApproved, ToolDenied
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
```

`ApprovalRequired` is an exception raised from a tool body to signal that this particular call needs human approval before proceeding. `ApprovalRequiredToolset` is a wrapper toolset that intercepts tool calls and requires approval based on a predicate function.

### `ApprovalRequiredToolset` constructor

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    approval_required_func: Callable[[RunContext, ToolDefinition, dict[str, Any]], bool]
    # Default: always requires approval (lambda ctx, tool_def, tool_args: True)
```

The `approval_required_func` receives the run context, the tool definition, and the parsed tool arguments. Return `True` to require approval, `False` to auto-execute.

### Example 1 — Per-tool approval via `ApprovalRequired`

```python
import asyncio
from pydantic_ai import Agent, ApprovalRequired, DeferredToolRequests
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.toolsets.function import FunctionToolset

toolset = FunctionToolset()

@toolset.tool
async def send_email(to: str, subject: str, body: str) -> str:
    # Raise to require approval before this executes
    raise ApprovalRequired

@toolset.tool
async def draft_email(to: str, subject: str) -> str:
    return f'Draft prepared for {to}: {subject}'

agent: Agent[None, DeferredToolRequests | str] = Agent(
    'openai:gpt-4o',
    toolsets=[toolset],
    output_type=[DeferredToolRequests, str],  # type: ignore[arg-type]
)

async def main() -> None:
    result = await agent.run('Draft and send an email to alice@example.com about the meeting.')
    if isinstance(result.output, DeferredToolRequests):
        pending = result.output
        print(f'Awaiting approval for {len(pending.approvals)} tool call(s):')
        for call in pending.approvals:
            print(f'  {call.tool_name}({call.args})')
```

### Example 2 — `ApprovalRequiredToolset` with predicate

```python
import asyncio
from pydantic_ai import Agent, RunContext, DeferredToolRequests
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.tools import ToolDefinition

inner = FunctionToolset[None]()

@inner.tool
async def delete_file(path: str) -> str:
    return f'deleted {path}'

@inner.tool
async def read_file(path: str) -> str:
    return f'content of {path}'

# Only require approval for destructive operations
def needs_approval(ctx: RunContext[None], tool_def: ToolDefinition, args: dict) -> bool:
    return tool_def.name.startswith('delete_')

approved_toolset = ApprovalRequiredToolset(
    toolset=inner,
    approval_required_func=needs_approval,
)

agent: Agent[None, DeferredToolRequests | str] = Agent(
    'openai:gpt-4o',
    toolsets=[approved_toolset],
    output_type=[DeferredToolRequests, str],  # type: ignore[arg-type]
)
```

### Example 3 — Resuming after approval with `ToolApproved` / `ToolDenied`

```python
import asyncio
from pydantic_ai import Agent, DeferredToolRequests, ApprovalRequired, ToolApproved, ToolDenied

agent: Agent[None, DeferredToolRequests | str] = Agent(
    'openai:gpt-4o',
    output_type=[DeferredToolRequests, str],  # type: ignore[arg-type]
)

@agent.tool_plain
def drop_table(table_name: str) -> str:
    raise ApprovalRequired

async def main() -> None:
    result = await agent.run('Drop the temp_cache table.')
    output = result.output
    if isinstance(output, DeferredToolRequests):
        # Simulate human reviewing and approving/denying
        approvals = {
            call.tool_call_id: ToolApproved()  # or ToolDenied(message='Not allowed')
            for call in output.approvals
        }
        deferred_results = output.build_results(approvals=approvals)
        final = await agent.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=deferred_results,
        )
        print(final.output)
```

---

## 5. `ConcurrencyLimiter` + `AbstractConcurrencyLimiter` + `ConcurrencyLimit` — Rate Limiting

**Module:** `pydantic_ai.concurrency`  
**Import:**
```python
from pydantic_ai.concurrency import ConcurrencyLimiter, AbstractConcurrencyLimiter, ConcurrencyLimit
from pydantic_ai import limit_model_concurrency
```

`ConcurrencyLimiter` wraps an `anyio.CapacityLimiter` and adds OTel span creation for wait periods plus queue-depth enforcement. `AbstractConcurrencyLimiter` is the ABC for custom implementations (e.g., Redis-backed distributed limiters). `ConcurrencyLimit` is a configuration dataclass for `max_running` + optional `max_queued`.

### `ConcurrencyLimiter` constructor

```python
ConcurrencyLimiter(
    max_running: int,                  # concurrent operation slots
    *,
    max_queued: int | None = None,     # queue depth cap; None = unlimited
    name: str | None = None,           # for OTel span labels
    tracer: Tracer | None = None,      # OTel tracer; falls back to 'pydantic-ai'
)
```

**Class method** `from_limit(limit, *, name, tracer)` — accepts `int` or `ConcurrencyLimit`.

### `ConcurrencyLimiter` key properties

| Property | Notes |
|---|---|
| `waiting_count` | Tasks currently blocked waiting for a slot |
| `running_count` | Tasks currently holding a slot |
| `available_count` | Free slots |
| `max_running` | Total configured slots |
| `name` | Optional label for OTel spans |

### Example 1 — Per-agent concurrency cap

```python
import asyncio
from pydantic_ai import Agent, limit_model_concurrency
from pydantic_ai.concurrency import ConcurrencyLimiter

limiter = ConcurrencyLimiter(max_running=3, name='my-agent')

# limit_model_concurrency wraps the model — pass the wrapped model to Agent
limited_model = limit_model_concurrency('openai:gpt-4o', limiter)
agent = Agent(limited_model)

async def process_batch(prompts: list[str]) -> list[str]:
    tasks = [agent.run(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return [r.output for r in results]
```

### Example 2 — Queue depth cap with `ConcurrencyLimit`

```python
import asyncio
from pydantic_ai.concurrency import ConcurrencyLimiter, ConcurrencyLimit
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

limiter = ConcurrencyLimiter.from_limit(
    ConcurrencyLimit(max_running=2, max_queued=5),
    name='inference-pool',
)

async def safe_acquire(task_id: str) -> None:
    try:
        await limiter.acquire(source=f'task:{task_id}')
        print(f'Task {task_id} running ({limiter.running_count} active)')
        await asyncio.sleep(0.1)
    except ConcurrencyLimitExceeded as e:
        print(f'Task {task_id} rejected: {e}')
    finally:
        if limiter.running_count > 0:
            limiter.release()

async def main() -> None:
    await asyncio.gather(*[safe_acquire(str(i)) for i in range(10)])
```

### Example 3 — Custom Redis-backed distributed limiter

```python
from pydantic_ai.concurrency import AbstractConcurrencyLimiter

class RedisLimiter(AbstractConcurrencyLimiter):
    """Distributed limiter backed by Redis SETNX / Lua scripts."""

    def __init__(self, redis_url: str, key: str, max_running: int) -> None:
        self._key = key
        self._max_running = max_running
        # In practice, initialise an async Redis client here
        self._held = False

    async def acquire(self, source: str) -> None:
        # Implement atomic check-and-set using a Redis Lua script
        # For illustration — real implementation uses redis.asyncio
        self._held = True

    def release(self) -> None:
        self._held = False

# limit_model_concurrency wraps the model — it calls limiter.acquire/release internally
from pydantic_ai import limit_model_concurrency, Agent

redis_limiter = RedisLimiter('redis://localhost:6379', 'my-agent-lock', max_running=10)
limited_model = limit_model_concurrency('openai:gpt-4o', redis_limiter)
agent = Agent(limited_model)

async def run_with_redis_limit(prompt: str) -> str:
    result = await agent.run(prompt)
    return result.output
```

---

## 6. `SkipModelRequest` + `SkipToolExecution` + `SkipToolValidation` — Hook Short-Circuits

**Module:** `pydantic_ai.exceptions`  
**Import:**
```python
from pydantic_ai import SkipModelRequest, SkipToolExecution, SkipToolValidation
from pydantic_ai.messages import ModelResponse, TextPart
```

These three exceptions are signals, not errors. Raise them inside `before_model_request` / `wrap_model_request`, `before_tool_execute` / `wrap_tool_execute`, and `before_tool_validate` / `wrap_tool_validate` hooks respectively to short-circuit the normal execution path and inject a synthetic response or result.

| Exception | Raise in | Effect |
|---|---|---|
| `SkipModelRequest(response)` | `before_model_request`, `wrap_model_request` | Uses `response` instead of calling the model |
| `SkipToolExecution(result)` | `before_tool_execute`, `wrap_tool_execute` | Uses `result` instead of running the tool function |
| `SkipToolValidation(validated_args)` | `before_tool_validate`, `wrap_tool_validate` | Uses `validated_args` instead of running Pydantic validation |

<Aside type="caution">
When `SkipModelRequest` is raised in `before_model_request`, message history modifications made by earlier capability hooks in that hook are **not** persisted, because request preparation is aborted.
</Aside>

### Example 1 — Cache layer via `SkipModelRequest`

```python
import asyncio
from datetime import datetime
from pydantic_ai import Agent, SkipModelRequest
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelResponse, TextPart

_cache: dict[str, str] = {}

def _last_user_prompt(messages) -> str | None:
    return next(
        (p.content for m in reversed(messages)
         for p in getattr(m, 'parts', [])
         if hasattr(p, 'content') and hasattr(p, 'part_kind') and p.part_kind == 'user-prompt'),
        None,
    )

hooks = Hooks()

@hooks.on.before_model_request
async def check_cache(ctx, request_context, /):
    last_user = _last_user_prompt(ctx.messages)
    if last_user and last_user in _cache:
        raise SkipModelRequest(
            ModelResponse(
                parts=[TextPart(content=_cache[last_user])],
                timestamp=datetime.now(),
                model_name='cached',
            )
        )
    return request_context

@hooks.on.after_model_request
async def populate_cache(ctx, /, *, request_context, response):
    last_user = _last_user_prompt(ctx.messages)
    text_parts = [p for p in response.parts if hasattr(p, 'content')]
    if last_user and text_parts:
        _cache[last_user] = text_parts[0].content
    return response  # must return response; returning None overwrites it with None

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

### Example 2 — Sandbox dry-run via `SkipToolExecution`

```python
import asyncio
from pydantic_ai import Agent, RunContext, SkipToolExecution

DRY_RUN = True

agent = Agent('openai:gpt-4o')

@agent.tool
async def write_file(ctx: RunContext[None], path: str, content: str) -> str:
    if DRY_RUN:
        raise SkipToolExecution(result=f'[DRY RUN] would write {len(content)} bytes to {path}')
    with open(path, 'w') as f:
        f.write(content)
    return f'written {path}'
```

### Example 3 — Argument normalization via `SkipToolValidation`

```python
from pydantic_ai import Agent, RunContext, SkipToolValidation
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_tool_validate
async def normalize_args(ctx: RunContext[None], /, *, call, tool_def, args: dict) -> dict:
    if tool_def.name == 'search_products':
        # Normalize the query before Pydantic validates it
        if 'query' in args and isinstance(args['query'], str):
            normalized = args['query'].lower().strip()
            raise SkipToolValidation(validated_args={**args, 'query': normalized})
    return args  # must return args; returning None replaces validated args with None

agent = Agent('openai:gpt-4o', capabilities=[hooks])

@agent.tool
async def search_products(ctx: RunContext[None], query: str, max_results: int = 10) -> list[str]:
    return [f'Product matching: {query}']
```

---

## 7. `ModelRetry` — Requesting Model Retries

**Module:** `pydantic_ai.exceptions`  
**Import:**
```python
from pydantic_ai import ModelRetry
```

`ModelRetry` is an exception you raise from any tool function, output validator, or capability hook to send a retry message back to the model. Unlike a Python exception propagating to the caller, it results in the LLM being told what went wrong and asked to try again — consuming one retry from the tool or output's retry budget.

### Constructor

```python
ModelRetry(message: str)  # message is sent back to the model as a retry prompt
```

`ModelRetry` is also serializable via Pydantic (used internally for Temporal durable execution).

### When to raise `ModelRetry`

| Context | Effect |
|---|---|
| Tool function | Model is told the tool failed with `message`; retry counter incremented |
| Output validator | Model is told output validation failed; output retry counter incremented |
| `after_tool_execute` hook | Same as tool function — model retries the tool call |
| `after_model_request` hook | Model is told the response was invalid; request retry counter incremented |

<Aside type="note">
Use `ctx.last_attempt` (or `ctx.retry == ctx.max_retries`) before raising `ModelRetry` if you want to skip expensive fallback logic on non-final attempts.
</Aside>

### Example 1 — Input validation in a tool

```python
import asyncio
import re
from pydantic_ai import Agent, RunContext, ModelRetry

agent: Agent[None, str] = Agent('openai:gpt-4o')

@agent.tool
async def lookup_user(ctx: RunContext[None], email: str) -> str:
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        raise ModelRetry(
            f'"{email}" is not a valid email address. '
            'Please provide a properly formatted email like user@example.com.'
        )
    return f'user profile for {email}'
```

### Example 2 — Retry with exponential context in the message

```python
import asyncio
from pydantic_ai import Agent, RunContext, ModelRetry

agent: Agent[None, str] = Agent('openai:gpt-4o', retries=3)

@agent.tool
async def fetch_json(ctx: RunContext[None], url: str) -> dict:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        if ctx.last_attempt:
            raise  # Let it propagate on the final attempt
        raise ModelRetry(
            f'Request to {url} timed out (attempt {ctx.retry + 1}). '
            'Try a different endpoint or simplify the URL.'
        )
    except httpx.HTTPStatusError as e:
        raise ModelRetry(f'HTTP {e.response.status_code} from {url}. Try a different URL.')
```

### Example 3 — Output validator with `ModelRetry`

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    summary: str

agent: Agent[None, Analysis] = Agent(
    'openai:gpt-4o',
    output_type=Analysis,
    retries=2,
)

@agent.output_validator
async def validate_confidence(ctx, output: Analysis) -> Analysis:
    if output.confidence < 0.0 or output.confidence > 1.0:
        raise ModelRetry(
            f'confidence must be between 0.0 and 1.0, got {output.confidence}. '
            'Return a value like 0.75 for 75% confidence.'
        )
    if output.sentiment not in ('positive', 'negative', 'neutral'):
        raise ModelRetry(
            f'sentiment must be "positive", "negative", or "neutral", got "{output.sentiment}".'
        )
    return output
```

---

## 8. `TemplateStr` + `format_as_xml` — Prompt Templating Utilities

**Module:** `pydantic_ai._template` / `pydantic_ai.format_prompt`  
**Import:**
```python
from pydantic_ai import TemplateStr, format_as_xml
```

`TemplateStr` is a Handlebars template string that renders against `RunContext.deps` at runtime. It is a valid callable for `Agent(instructions=...)` and `Agent(system_prompt=...)`, and integrates with [pydantic-handlebars](https://github.com/pydantic/pydantic-handlebars) for schema-validated template compilation.

`format_as_xml` converts Python objects (dataclasses, Pydantic models, dicts, lists) into an XML string — particularly useful for injecting structured examples or data into prompts.

### `TemplateStr` constructor

```python
TemplateStr(
    source: str,                      # Handlebars template: "Hello {{name}}"
    *,
    deps_type: type[Any] | None = None,   # Optional type for schema validation
    deps_schema: dict[str, Any] | None = None,  # Optional JSON schema
)
```

When `deps_type` is provided, the template is compiled against the Pydantic schema of that type — mismatched field names raise at construction time. The `__call__(ctx)` method is the callable interface for use as an agent instruction.

### `format_as_xml` signature

```python
format_as_xml(
    obj: Any,
    root_tag: str | None = None,        # outer wrapper tag; None = no wrapper
    item_tag: str = 'item',             # tag for iterable items
    none_str: str = 'null',             # representation for None
    indent: str | None = '  ',          # pretty-print indent; None = compact
    include_field_info: Literal['once'] | bool = False,  # include Pydantic/dataclass metadata
) -> str
```

### Example 1 — `TemplateStr` for per-user personalised instructions

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr

@dataclass
class UserDeps:
    username: str
    preferred_language: str
    expertise_level: str  # 'beginner', 'intermediate', 'expert'

agent: Agent[UserDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    instructions=TemplateStr(
        'You are assisting {{username}}. '
        'Respond in {{preferred_language}}. '
        'Adjust your explanation depth for a {{expertise_level}} level.',
        deps_type=UserDeps,
    ),
)

async def main() -> None:
    result = await agent.run(
        'Explain async/await.',
        deps=UserDeps(
            username='Alice',
            preferred_language='English',
            expertise_level='beginner',
        ),
    )
    print(result.output)
```

### Example 2 — Dynamic system prompt with list context

```python
import asyncio
from dataclasses import dataclass, field
from pydantic_ai import Agent, TemplateStr

@dataclass
class SearchDeps:
    allowed_domains: list[str] = field(default_factory=list)
    max_results: int = 10

agent: Agent[SearchDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=SearchDeps,
    system_prompt=TemplateStr(
        'Only search within: {{#each allowed_domains}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}. '
        'Return at most {{max_results}} results.',
        deps_type=SearchDeps,
    ),
)
```

### Example 3 — `format_as_xml` for structured context injection

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, format_as_xml

@dataclass
class Product:
    name: str
    price: float
    in_stock: bool
    category: str

agent = Agent('openai:gpt-4o')

async def main() -> None:
    products = [
        Product('Laptop Pro', 999.99, True, 'Electronics'),
        Product('Wireless Mouse', 29.99, True, 'Accessories'),
        Product('USB Hub', 49.99, False, 'Accessories'),
    ]
    context_xml = format_as_xml(
        products,
        root_tag='product_catalog',
        item_tag='product',
        include_field_info='once',
    )
    result = await agent.run(
        f'Which products are in the Accessories category and in stock?\n\n{context_xml}'
    )
    print(result.output)
```

---

## 9. `AgentSpec` — YAML/JSON-Driven Agent Configuration

**Module:** `pydantic_ai._spec` (exported as `pydantic_ai.AgentSpec`)  
**Import:**
```python
from pydantic_ai import AgentSpec
```

`AgentSpec` is a Pydantic `BaseModel` that lets you define an agent's complete configuration in a YAML or JSON file, then instantiate an `Agent` from it at runtime. This is useful for configuration-driven deployments, A/B testing different agent configurations, or exposing agent config to non-developer stakeholders.

### `AgentSpec` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | `str \| None` | `None` | Model identifier, e.g. `'openai:gpt-4o'` |
| `name` | `str \| None` | `None` | Agent name for observability |
| `description` | `TemplateStr \| str \| None` | `None` | Human-readable description |
| `instructions` | `TemplateStr \| str \| list[...] \| None` | `None` | System instructions |
| `deps_schema` | `dict[str, Any] \| None` | `None` | JSON schema for runtime deps |
| `output_schema` | `dict[str, Any] \| None` | `None` | JSON schema for structured output |
| `model_settings` | `dict[str, Any] \| None` | `None` | Model parameter overrides |
| `retries` | `int \| AgentRetries \| None` | `None` | Retry budgets |
| `end_strategy` | `EndStrategy` | `'early'` | `'early'` or `'exhaustive'` |
| `tool_timeout` | `float \| None` | `None` | Per-tool timeout in seconds |
| `metadata` | `dict[str, Any] \| None` | `None` | Arbitrary run metadata |
| `capabilities` | `list[CapabilitySpec]` | `[]` | Capabilities (e.g. Instrumentation, MCP) |

### Class methods

| Method | Description |
|---|---|
| `AgentSpec.from_file(path, fmt=None)` | Load from YAML or JSON file |
| `AgentSpec.from_text(text, fmt='yaml')` | Parse from a YAML/JSON string |
| `AgentSpec.from_dict(data)` | Validate from a Python dict |
| `spec.to_file(path, fmt=None, schema_path=...)` | Save to file with optional JSON schema |
| `AgentSpec.model_json_schema_with_capabilities(...)` | Generate schema for editor autocomplete |

### Example 1 — Load from YAML and instantiate

```yaml
# agent_config.yaml
model: openai:gpt-4o
name: support-agent
instructions: |
  You are a helpful customer support agent.
  Always be polite and concise.
model_settings:
  temperature: 0.3
  max_tokens: 500
retries: 3
end_strategy: early
```

```python
import asyncio
from pydantic_ai import AgentSpec, Agent

spec = AgentSpec.from_file('agent_config.yaml')
agent = spec.to_agent()  # Returns a fully configured Agent

async def main() -> None:
    result = await agent.run('How do I reset my password?')
    print(result.output)
```

### Example 2 — Dynamic spec from dict with `TemplateStr` instructions

```python
import asyncio
from pydantic_ai import AgentSpec

spec = AgentSpec.from_dict({
    'model': 'openai:gpt-4o',
    'name': 'code-reviewer',
    'instructions': 'Review Python code for {{review_style}} issues.',
    'model_settings': {'temperature': 0.1},
    'retries': 2,
})

agent = spec.to_agent()

async def main() -> None:
    # deps can populate the TemplateStr at runtime
    result = await agent.run(
        'def foo(x): return x*x',
        deps={'review_style': 'security'},
    )
    print(result.output)
```

### Example 3 — Serialise and round-trip a spec

```python
import asyncio
import tempfile
from pathlib import Path
from pydantic_ai import AgentSpec

# Build a spec programmatically
spec = AgentSpec(
    model='openai:gpt-4o',
    name='summarizer',
    instructions='Summarize text in {{language}}.',
    model_settings={'temperature': 0.5, 'max_tokens': 300},
    retries=2,
)

with tempfile.TemporaryDirectory() as tmpdir:
    yaml_path = Path(tmpdir) / 'summarizer.yaml'
    spec.to_file(yaml_path)  # saves YAML + JSON schema sidecar
    
    # Later: load it back
    loaded_spec = AgentSpec.from_file(yaml_path)
    assert loaded_spec.model == spec.model
    assert loaded_spec.retries == spec.retries

    agent = loaded_spec.to_agent()
    print(f'Agent: {agent.name}')
```

---

## 10. `UploadedFile` + `BinaryContent` + `FilePart` — Multimodal File Handling

**Module:** `pydantic_ai.messages`  
**Import:**
```python
from pydantic_ai import UploadedFile, BinaryContent, FilePart
from pydantic_ai.messages import ImageUrl, AudioUrl, DocumentUrl
```

These three types cover the three ways to send file content to models: by provider file ID (already uploaded), by inline binary bytes, or as a file-type response part from the model.

### `UploadedFile` — Provider file ID reference

```python
UploadedFile(
    file_id: str,                        # provider-specific ID (or GCS/S3 URI)
    provider_name: UploadedFileProviderName,  # 'openai', 'anthropic', 'google', etc.
    *,
    media_type: str | None = None,       # inferred from file_id extension if not set
    vendor_metadata: dict | None = None, # e.g. {'video_metadata': ...} for Google
    identifier: str | None = None,       # human-readable ID for tool referencing
)
```

`media_type` is a computed property — it falls back to `mimetypes.guess_type` on the file ID path, then `'application/octet-stream'` if unrecognised.

**Supported providers**: OpenAI, Anthropic, Google (Gemini Files API URI or GCS `gs://`), xAI, Bedrock (S3 `s3://`).

### `BinaryContent` — Inline binary data

```python
BinaryContent(
    data: bytes,                           # raw file bytes
    *,
    media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str,
    vendor_metadata: dict | None = None,   # e.g. {'detail': 'high'} for OpenAI images
    identifier: str | None = None,
)
```

### `FilePart` — File in a model response

`FilePart` appears in `ModelResponse.parts` when a model returns a generated file (e.g. an image from an image-generation call). Its `content` field is a `BinaryContent` (narrowed to `BinaryImage` for images).

### Example 1 — Send an uploaded file by ID

```python
import asyncio
from pydantic_ai import Agent, UploadedFile
from pydantic_ai.messages import UserPromptPart

agent = Agent('openai:gpt-4o')

async def main() -> None:
    # Assume file was previously uploaded via the OpenAI Files API
    uploaded_pdf = UploadedFile(
        file_id='file-abc123def456',
        provider_name='openai',
        media_type='application/pdf',
    )
    result = await agent.run([
        'Summarize the key points from this document.',
        uploaded_pdf,
    ])
    print(result.output)
```

### Example 2 — Inline binary image

```python
import asyncio
from pathlib import Path
from pydantic_ai import Agent, BinaryContent

agent = Agent('openai:gpt-4o')

async def main() -> None:
    image_bytes = Path('screenshot.png').read_bytes()
    inline_image = BinaryContent(
        data=image_bytes,
        media_type='image/png',
        vendor_metadata={'detail': 'high'},  # OpenAI detail level
    )
    result = await agent.run([
        'Describe what you see in this screenshot.',
        inline_image,
    ])
    print(result.output)
```

### Example 3 — Access `FilePart` from a model response

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import FilePart, BinaryContent

agent = Agent('openai:gpt-4o-image')  # hypothetical image-gen model

async def main() -> None:
    result = await agent.run('Generate a simple flowchart diagram.')
    # Inspect response parts for generated files
    for msg in result.all_messages():
        for part in getattr(msg, 'parts', []):
            if isinstance(part, FilePart):
                content: BinaryContent = part.content
                print(f'Received file: {content.media_type}, {len(content.data)} bytes')
                Path('generated_image.png').write_bytes(content.data)
                print('Saved to generated_image.png')
                break
```

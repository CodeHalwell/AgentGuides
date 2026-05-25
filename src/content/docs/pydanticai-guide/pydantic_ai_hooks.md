---
title: "PydanticAI: Hooks — Lifecycle Callbacks"
description: "Decorator-based hook registration for agent run lifecycle, model requests, tool calls, and output validation using the Hooks capability."
framework: pydanticai
language: python
---

# Hooks — Lifecycle Callbacks

Verified against **pydantic-ai==1.102.0** — source module: `pydantic_ai.capabilities.hooks`.

The `Hooks` class gives you a decorator-first API for intercepting every phase of an agent run without subclassing `AbstractCapability`. Register a function once with `@hooks.on.<event>` and it fires automatically during runs that include this capability.

## Minimal runnable example

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_run
async def log_start(ctx):
    print(f'Run starting — prompt: {ctx.prompt!r}')

@hooks.on.after_run
async def log_finish(ctx, *, result):
    print(f'Run finished — output: {result.output!r}')
    return result

agent = Agent('openai:gpt-4o', capabilities=[hooks])

async def main():
    result = await agent.run('What is 2 + 2?')
    print(result.output)

asyncio.run(main())
```

Both sync and async hook functions are accepted — sync ones are wrapped automatically.

## Hook categories

All hooks live on `hooks.on.*`:

| Category | Events |
|---|---|
| **Run lifecycle** | `before_run`, `after_run`, `run` (wrap), `run_error` |
| **Node lifecycle** | `before_node_run`, `after_node_run`, `node_run` (wrap), `node_run_error` |
| **Model request** | `before_model_request`, `after_model_request`, `model_request` (wrap), `model_request_error` |
| **Tool preparation** | `prepare_tools`, `prepare_output_tools` |
| **Tool validation** | `before_tool_validate`, `after_tool_validate`, `tool_validate` (wrap), `tool_validate_error` |
| **Tool execution** | `before_tool_execute`, `after_tool_execute`, `tool_execute` (wrap), `tool_execute_error` |
| **Output validation** | `before_output_validate`, `after_output_validate`, `output_validate` (wrap), `output_validate_error` |
| **Output processing** | `before_output_process`, `after_output_process`, `output_process` (wrap), `output_process_error` |
| **Deferred tools** | `handle_deferred_tool_calls` |

## Run lifecycle hooks

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult

hooks = Hooks()

@hooks.on.before_run
def before(ctx: RunContext) -> None:
    print(f'[before_run] conversation_id={ctx.run_id}')

@hooks.on.after_run
async def after(ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult:
    print(f'[after_run] tokens used: {result.usage.total_tokens}')  # .usage is a property in 1.102.0
    return result  # must return the result

@hooks.on.run_error
async def on_error(ctx: RunContext, *, error: BaseException) -> AgentRunResult:
    print(f'[run_error] {type(error).__name__}: {error}')
    raise error   # re-raise or return a fallback result

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

## Model request hooks — observability

```python
from typing import TYPE_CHECKING
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelResponse

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

hooks = Hooks()
_request_log: list[dict] = []

@hooks.on.before_model_request
async def record_request(ctx, request_context: 'ModelRequestContext'):
    _request_log.append({'run_id': ctx.run_id, 'messages': len(request_context.messages)})
    return request_context  # must return (optionally modified) context

@hooks.on.after_model_request
async def record_response(ctx, *, request_context, response: ModelResponse) -> ModelResponse:
    print(f'  model={response.model_name}  tokens={response.usage.total_tokens}')
    return response  # must return (optionally modified) response

@hooks.on.model_request_error
async def handle_request_error(ctx, *, request_context, error: Exception) -> ModelResponse:
    print(f'[model error] {error}')
    raise error
```

## Tool hooks — guarding execution

Filter which tools a hook applies to with the `tools=` parameter:

```python
from pydantic_ai import ModelRetry
from pydantic_ai.capabilities import Hooks
from pydantic_ai.tools import ToolDefinition

hooks = Hooks()

@hooks.on.before_tool_execute(tools=['delete_record', 'drop_table'])
async def audit_destructive(ctx, *, call, tool_def: ToolDefinition, args):
    print(f'[AUDIT] {tool_def.name} called with {args}')
    return args  # must return (optionally modified) args

@hooks.on.after_tool_execute
async def cache_tool_result(ctx, *, call, tool_def: ToolDefinition, args, result):
    # Could persist result to a cache keyed by (tool_def.name, frozenset(args.items()))
    print(f'[cache] {tool_def.name} → {result!r}')
    return result  # must return (optionally modified) result

@hooks.on.tool_execute_error
async def retry_on_network_error(ctx, *, call, tool_def, args, error: Exception):
    if 'timeout' in str(error).lower():
        raise ModelRetry(f'Tool {tool_def.name} timed out — please try again.')
    raise error
```

The `tools=` filter is only available on `before_tool_execute`, `after_tool_execute`, `tool_execute`, `tool_execute_error`, `before_tool_validate`, `after_tool_validate`, `tool_validate`, `tool_validate_error`.

## Output validation hooks

```python
from pydantic_ai import ModelRetry
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_output_validate
async def normalise_text(ctx, *, output_context, output):
    if isinstance(output, str):
        return output.strip()
    return output

@hooks.on.after_output_validate
async def enforce_policy(ctx, *, output_context, output):
    if hasattr(output, 'confidence') and output.confidence < 0.5:
        raise ModelRetry('Confidence too low — provide a more confident answer.')
    return output
```

## Using timeouts

Pass `timeout=` (seconds) to any hook decorator. If the hook function takes longer, `HookTimeoutError` is raised:

```python
from pydantic_ai.capabilities import Hooks, HookTimeoutError

hooks = Hooks()

@hooks.on.before_tool_execute(timeout=2.0)
async def slow_guard(ctx, *, call, tool_def, args):
    await some_external_check()  # must finish in < 2 s
    return args
```

## Wrapping hooks — full control of the call chain

`wrap_*` variants give you a `handler` callable to invoke the original operation, enabling retry decorators, circuit-breakers, and tracing:

```python
import time
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.model_request
async def time_model_request(ctx, *, request_context, handler):
    t0 = time.perf_counter()
    response = await handler(request_context)   # call the model
    elapsed = time.perf_counter() - t0
    print(f'model request took {elapsed:.3f}s')
    return response

@hooks.on.tool_execute
async def wrap_tool(ctx, *, call, tool_def, args, handler):
    print(f'→ executing {tool_def.name}')
    result = await handler(call, tool_def, args)
    print(f'← {tool_def.name} returned')
    return result
```

## Composing multiple Hooks instances

Stack several `Hooks` objects in the `capabilities=` list — they run in order:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

audit_hooks = Hooks()
metrics_hooks = Hooks()

@audit_hooks.on.before_tool_execute(tools=['write_db'])
async def audit_write(ctx, *, call, tool_def, args):
    print(f'AUDIT: {ctx.run_id} — {tool_def.name}')
    return args

@metrics_hooks.on.after_run
async def record_metric(ctx, *, result):
    metrics.increment('agent.run.completed')
    return result

agent = Agent('openai:gpt-4o', capabilities=[audit_hooks, metrics_hooks])
```

## Per-run hooks

Hooks can also be passed at `run()` time for one-shot observability:

```python
one_time = Hooks()

@one_time.on.before_model_request
async def debug_this_request(ctx, request_context):
    print('DEBUG:', request_context.messages)
    return request_context

result = await agent.run('test prompt', capabilities=[one_time])
```

## Event streaming hook

`wrap_run_event_stream` intercepts the raw event stream for custom event processing:

```python
from collections.abc import AsyncIterable
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import AgentStreamEvent

hooks = Hooks()

@hooks.on.event
async def forward_events(ctx, event: AgentStreamEvent) -> AgentStreamEvent:
    # Forward every event to a websocket, message queue, etc.
    await ws.send_json({'type': event.__class__.__name__})
    return event

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

## Complete observability pattern

```python
import time
import logging
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

logger = logging.getLogger(__name__)

@dataclass
class RunMetrics:
    model_calls: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    errors: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0

def make_observability_hooks() -> tuple[Hooks, RunMetrics]:
    hooks = Hooks()
    m = RunMetrics()
    _t0: list[float] = []

    @hooks.on.before_run
    def start_timer(ctx):
        _t0.append(time.perf_counter())

    @hooks.on.after_run
    def stop_timer(ctx, *, result):
        m.wall_time_s = time.perf_counter() - _t0[0]
        logger.info('run finished in %.3fs  tokens=%d', m.wall_time_s, m.total_tokens)
        return result

    @hooks.on.after_model_request
    def count_tokens(ctx, *, request_context, response):
        m.model_calls += 1
        m.total_tokens += response.usage.total_tokens if response.usage else 0
        return response

    @hooks.on.after_tool_execute
    def count_tool(ctx, *, call, tool_def, args, result):
        m.tool_calls[tool_def.name] = m.tool_calls.get(tool_def.name, 0) + 1
        return result

    @hooks.on.run_error
    def capture_error(ctx, *, error):
        m.errors.append(f'{type(error).__name__}: {error}')
        raise error

    return hooks, m

obs_hooks, metrics = make_observability_hooks()
agent = Agent('openai:gpt-4o', capabilities=[obs_hooks])
```

## `handle_deferred_tool_calls` — intercept deferred tool batches

Fires whenever the agent produces a `DeferredToolRequests` (i.e. some tool called `CallDeferred` or `ApprovalRequired`). Use it for logging, queuing, or automatically resolving deferrals in tests:

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ToolReturn

hooks = Hooks()

@hooks.on.handle_deferred_tool_calls
async def log_and_auto_resolve(ctx, *, deferred: DeferredToolRequests):
    """Log every deferred call and automatically resolve them with a placeholder."""
    print(f'[deferred] {len(deferred.calls)} call(s) deferred this step')
    for call in deferred.calls:
        print(f'  → {call.tool_name}({call.args})')
    # Return resolved results to continue the run automatically
    return DeferredToolResults(calls={
        call.tool_call_id: ToolReturn(content=f'Auto-resolved: {call.tool_name}')
        for call in deferred.calls
    })

agent = Agent(
    'openai:gpt-4o',
    output_type=[str, DeferredToolRequests],
    capabilities=[hooks],
)

@agent.tool_plain
def slow_job(task: str) -> str:
    raise CallDeferred(metadata={'task': task})

async def main():
    # With the hook, deferred calls are auto-resolved — no manual loop needed
    result = await agent.run('Run slow_job for task "data-export".')
    print(result.output)

asyncio.run(main())
```

## `on_run_error` — recover from run failures

`on_run_error` fires when an exception escapes the run. It can either re-raise the original error or return a fallback `AgentRunResult`:

```python
import asyncio
import logging
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.run import AgentRunResult
from pydantic_ai.exceptions import UsageLimitExceeded

log = logging.getLogger(__name__)
hooks = Hooks()

@hooks.on.on_run_error
async def handle_error(ctx: RunContext, *, error: BaseException) -> AgentRunResult:
    if isinstance(error, UsageLimitExceeded):
        # Return a graceful fallback instead of crashing
        log.warning('Usage limit exceeded for run %s', ctx.run_id)
        raise error  # re-raise so caller can handle it
    log.error('Unexpected run failure', exc_info=error)
    raise error  # always re-raise unknown errors

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

## Complete hook lifecycle diagram

```
agent.run(prompt)
│
├─ before_run
│   │
│   └─ [loop per model step]
│       ├─ before_node_run
│       ├─ wrap_node_run(handler)
│       │   ├─ before_model_request  ← modify messages before sending
│       │   ├─ wrap_model_request(handler)
│       │   │   └─ [model API call]
│       │   ├─ after_model_request   ← inspect/modify response
│       │   │
│       │   └─ [for each tool call in response]
│       │       ├─ before_tool_validate  ← inspect raw args JSON
│       │       ├─ wrap_tool_validate(handler)
│       │       ├─ after_tool_validate   ← inspect validated typed args
│       │       │
│       │       ├─ before_tool_execute  ← last chance to skip/modify
│       │       ├─ wrap_tool_execute(handler)
│       │       │   └─ [tool function runs]
│       │       └─ after_tool_execute   ← inspect/modify result
│       │
│       └─ after_node_run
│
├─ after_run (or on_run_error on failure)
└─ AgentRunResult returned
```

## Skip exceptions — short-circuit the pipeline

Three exception classes let hook functions completely bypass the normal execution step and inject a synthetic result. Raise them inside `before_*` or `wrap_*` hooks.

### `SkipModelRequest` — inject a cached response

```python
import asyncio
import hashlib
import json
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart, RequestUsage
import datetime

_cache: dict[str, ModelResponse] = {}

def _cache_key(messages) -> str:
    # Use model_dump for stable, field-aware serialisation instead of str()
    content = json.dumps([m.model_dump(mode='json') for m in messages], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

def _make_response(text: str) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=text)],
        model_name='cache',
        usage=RequestUsage(request_tokens=0, response_tokens=0),
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )

hooks = Hooks()

@hooks.on.before_model_request
async def serve_from_cache(ctx, request_context):
    key = _cache_key(request_context.messages)
    if key in _cache:
        raise SkipModelRequest(_cache[key])   # ← bypass the model API call
    return request_context

@hooks.on.after_model_request
async def populate_cache(ctx, *, request_context, response):
    _cache[_cache_key(request_context.messages)] = response
    return response

agent = Agent('openai:gpt-4o', capabilities=[hooks])

async def main():
    r1 = await agent.run('What is 2 + 2?')   # API call
    r2 = await agent.run('What is 2 + 2?')   # cache hit — no API call
    print(r1.output, r2.output)

asyncio.run(main())
```

`SkipModelRequest` takes a single `ModelResponse` argument. Any message history modifications made by earlier capabilities in the same `before_model_request` call are **not** persisted when this exception is raised.

### `SkipToolValidation` — bypass Pydantic validation

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipToolValidation

hooks = Hooks()

@hooks.on.before_tool_validate(tools=['search'])
async def normalise_query(ctx, *, call, tool_def, raw_args):
    """Lowercase the query before validation so it doesn't fail enum checks."""
    if isinstance(raw_args, dict) and 'query' in raw_args:
        raise SkipToolValidation({**raw_args, 'query': raw_args['query'].lower()})
    return raw_args

agent = Agent('openai:gpt-4o', capabilities=[hooks])

@agent.tool_plain
def search(query: str) -> list[str]:
    return [f'Result: {query}']
```

### `SkipToolExecution` — mock tool results

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipToolExecution

DRY_RUN = True

hooks = Hooks()

@hooks.on.before_tool_execute(tools=['delete_record', 'send_email'])
async def dry_run_guard(ctx, *, call, tool_def, args):
    if DRY_RUN:
        raise SkipToolExecution(f'[DRY RUN] {tool_def.name}({args})')
    return args

agent = Agent('openai:gpt-4o', capabilities=[hooks])

@agent.tool_plain
def delete_record(record_id: str) -> bool:
    return True   # never runs in dry-run mode
```

`SkipToolExecution` takes the result to return to the model as its sole argument. The tool function is never called.

## Reference

- `Hooks` class — `capabilities/hooks.py`
- `HookTimeoutError` — raised when `timeout=` is exceeded
- `AbstractCapability` — base class for custom capabilities (`capabilities/abstract.py`)
- Hook function protocols (type annotations) — all `*HookFunc` classes in `capabilities/hooks.py`
- `handle_deferred_tool_calls` hook — `capabilities/deferred_tool_handler.py`
- `SkipModelRequest`, `SkipToolValidation`, `SkipToolExecution` — `exceptions.py`

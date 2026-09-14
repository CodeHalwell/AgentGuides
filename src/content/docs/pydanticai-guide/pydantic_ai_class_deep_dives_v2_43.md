---
title: "PydanticAI: 10 Source-Verified Class Deep Dives (2.43.0)"
description: "Runnable, source-verified code examples for ToolFailed, RunCancelled, ToolSelector, ToolOrOutput, ServiceTier/ThinkingLevel in ModelSettings, AgentStream, SkipModelRequest/SkipToolValidation/SkipToolExecution, ToolDefinition.sequential, OutputContext, and ApprovalRequired — verified against pydantic-ai 2.43.0."
framework: pydanticai
language: python
sidebar:
  order: 127
---

# 10 Source-Verified Class Deep Dives — v2.43.0

Verified against **pydantic-ai 2.43.0** (installed package, sources read directly).
Modules consulted: `pydantic_ai/exceptions.py`, `pydantic_ai/tools.py`,
`pydantic_ai/settings.py`, `pydantic_ai/result.py`, `pydantic_ai/output.py`,
`pydantic_ai/tool_manager.py`.

This page covers classes that were **absent or thin** in the three earlier deep-dive pages
(v2.33.0, v2.36.0, v2.40.0). All 10 items here are distinct from those sets.

```bash
pip install "pydantic-ai==2.43.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.43.0
```

---

## 1. `ToolFailed` — terminal tool failure without retry

**Module:** `pydantic_ai.exceptions`

`ToolFailed` is the counterpart to `ModelRetry`. Raise it from a tool when the call has
*definitively* failed — a missing resource, an unsupported operation, a confirmed upstream
error — and you want the model to see the failure and adapt rather than retry the same call.

Unlike `ModelRetry` it does **not** prepend "please try again" instructions and does **not**
consume the tool's retry budget. Use `UsageLimits` at the run level to bound repeated failures.

### Constructor

```python
raise ToolFailed('The requested resource was not found.')
```

`message` is returned to the model as a failed tool result, verbatim.
Constructor signature: `ToolFailed(message: str)`.

### `ToolFailed` vs `ModelRetry` comparison

| | `ModelRetry` | `ToolFailed` |
|---|---|---|
| Model sees | Retry prompt + your message | Your message only |
| Consumes retry budget | Yes | No |
| Model behaviour | Likely retries with corrections | Adapts, moves on |
| When to use | Transient / fixable failures | Definitive failures |

### Example 1 — raising from a tool

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.exceptions import ToolFailed

agent = Agent('openai:gpt-4o-mini')


@agent.tool_plain
def fetch_user(user_id: int) -> dict:
    """Fetch a user record."""
    # Simulate a definitive "not found" — no point retrying.
    if user_id < 0:
        raise ToolFailed(f'User {user_id} does not exist.')
    return {'id': user_id, 'name': 'Alice'}


async def main() -> None:
    result = await agent.run('Look up user -5 and summarise what you find.')
    print(result.output)
    # Model receives the failure and explains it cannot find the user.


asyncio.run(main())
```

### Example 2 — distinguishing ToolFailed from unexpected exceptions

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, ToolFailed

agent = Agent('openai:gpt-4o-mini')


@agent.tool_plain
def call_external_api(endpoint: str) -> str:
    """Call an external service."""
    if endpoint == '/gone':
        raise ToolFailed('Endpoint permanently removed (410 Gone).')
    if endpoint == '/busy':
        # Transient — ask model to retry with a different endpoint or later.
        raise ModelRetry('Service temporarily unavailable. Try again shortly.')
    return f'Response from {endpoint}'


async def main() -> None:
    result = await agent.run(
        'Call /gone and tell me what happened.',
    )
    print(result.output)


asyncio.run(main())
```

---

## 2. `RunCancelled` — first-party cancellation with history recovery

**Module:** `pydantic_ai.exceptions`

`RunCancelled` is raised when *your own code* cancels a run via
`AgentRun.cancel()` or `RunContext.cancel()`. It is a normal, catchable
application-level outcome — not an infrastructure crash.

Everything the run completed before cancellation is preserved: `all_messages()` returns the
full resumable history ready to pass as `message_history` to a new run. Any tool calls that
never produced a result are closed out with synthetic `outcome='interrupted'` returns.

### Key API

| Member | Type | Purpose |
|---|---|---|
| `all_messages()` | `list[ModelMessage]` | Full resumable history including seeded history |
| `new_messages()` | `list[ModelMessage]` | Only messages produced in this run |
| `usage` | `RunUsage` | Token / cost usage up to cancellation |
| `run_id` | `str \| None` | Run identifier, or `None` if cancelled before start |
| `conversation_id` | `str \| None` | Conversation identifier |
| `from_cancellation(exc)` | classmethod | Recover run state from an external `CancelledError` |

### `from_cancellation` — external timeout / task cancellation

When an external `asyncio.timeout()` or `asyncio.Task.cancel()` fires, Pydantic AI attaches
the partial run state to the raised exception. `from_cancellation()` traverses the exception
chain to find it:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.exceptions import RunCancelled

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    partial_history = None
    try:
        async with asyncio.timeout(0.5):   # very short timeout for demo
            result = await agent.run('Write a 10-paragraph essay on AI.')
    except asyncio.TimeoutError as exc:
        run_cancelled = RunCancelled.from_cancellation(exc)
        if run_cancelled:
            partial_history = run_cancelled.all_messages()
            print(f'Cancelled after {run_cancelled.usage.requests} request(s)')
            # Resume from where we left off.
            result = await agent.run(
                'Continue from where you left off.',
                message_history=partial_history,
            )
            print(result.output)


asyncio.run(main())
```

### Example — first-party cancellation via AgentRun

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.exceptions import RunCancelled

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    try:
        async with agent.iter('Write a very long story.') as run:
            async for node in run:
                node_name = type(node).__name__
                print(f'Node: {node_name}')
                if node_name == 'CallToolsNode':
                    # Cancel after the first model response.
                    run.cancel()
    except RunCancelled as exc:
        msgs = exc.all_messages()
        print(f'Stopped early, captured {len(msgs)} message(s).')
        print(f'Usage: {exc.usage}')


asyncio.run(main())
```

---

## 3. `ToolSelector` — unified tool selection

**Module:** `pydantic_ai.tools`

`ToolSelector` is the unified type alias for specifying which tools a capability or toolset
wrapper should apply to. It replaces ad-hoc string lists in older patterns.

```python
ToolSelector = Literal['all'] | Sequence[str] | dict[str, Any] | ToolSelectorFunc
```

| Form | Matches |
|---|---|
| `'all'` | Every tool (default for most capabilities) |
| `['search', 'calc']` | Tools whose names are in the list |
| `{'category': 'read_only'}` | Tools whose `metadata` deeply includes all key-value pairs |
| `Callable[[RunContext, ToolDefinition], bool]` | Custom sync / async predicate |

The first three forms are serializable for use in agent specs (YAML/JSON).

### Example — metadata-based selection applied via a hook

The dict form of `ToolSelector` is accepted by hook decorators' `tools=` argument,
so the hook fires only for tools whose metadata matches:

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

# This hook fires only for tools whose metadata contains category='read_only'.
@hooks.on.before_tool_execute(tools={'category': 'read_only'})
async def log_read_tool(ctx, *, call, tool_def, args):
    print(f'Read-only tool called: {tool_def.name}')
    return args  # must return args unchanged (or modified)


agent = Agent('openai:gpt-4o-mini', capabilities=[hooks])


@agent.tool(metadata={'category': 'read_only'})
def list_files(ctx: RunContext[None]) -> list[str]:
    """List files in the project."""
    return ['README.md', 'main.py']


@agent.tool(metadata={'category': 'write'})
def create_file(ctx: RunContext[None], name: str) -> str:
    """Create a new file."""
    return f'Created {name}'


async def main() -> None:
    result = await agent.run('What files are in the project?')
    print(result.output)


asyncio.run(main())
```

### Example — callable predicate selector

```python
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition, ToolSelectorFunc


def long_description_only(ctx: RunContext, tool_def: ToolDefinition) -> bool:
    """Only expose tools with more than 30 chars in their description."""
    return bool(tool_def.description and len(tool_def.description) > 30)


# Pass long_description_only wherever a ToolSelector is accepted.
```

---

## 4. `ToolOrOutput` — restrict function tools while keeping output available

**Module:** `pydantic_ai.settings`

`ToolOrOutput` lets you control which *function* tools the model can invoke while still
allowing the agent to complete with structured output, plain text, or images.

This solves a real problem: setting `tool_choice = ['my_tool']` also excludes output tools,
preventing the agent from ever finishing. `ToolOrOutput` threads the needle.

```python
@dataclass
class ToolOrOutput:
    function_tools: list[str]
```

### Constructor

```python
from pydantic_ai.settings import ToolOrOutput

# Allow only 'search' and 'calculator' function tools.
# Output tools (structured output, text, images) are always available.
settings = {'tool_choice': ToolOrOutput(function_tools=['search', 'calculator'])}
```

### Example — restricting function tools while keeping output available

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.settings import ToolOrOutput

agent = Agent('openai:gpt-4o-mini')


@agent.tool_plain
def search_web(query: str) -> str:
    """Search the web for information."""
    return f'Results for "{query}": some relevant data.'


@agent.tool_plain
def calculate(expression: str) -> float:
    """Evaluate a simple arithmetic expression (addition only)."""
    # Restricted to addition to avoid arbitrary code execution.
    parts = expression.split('+')
    return sum(float(p.strip()) for p in parts)


async def main() -> None:
    # Allow only search_web as a function tool; the output tool (text / structured
    # output) remains available so the model can still finish the run.
    # `calculate` is excluded from the function tools but the model can still
    # complete — ToolOrOutput does not force a specific tool call.
    result = await agent.run(
        'What is the population of France? Search then summarise.',
        model_settings={'tool_choice': ToolOrOutput(function_tools=['search_web'])},
    )
    print(result.output)


asyncio.run(main())
```

---

## 5. `ServiceTier` and `ThinkingLevel` in `ModelSettings`

**Module:** `pydantic_ai.settings`

Two new cross-provider settings landed in recent versions: `service_tier` and `thinking`.
Both are `TypeAlias` values and both are fields on `ModelSettings` (a `TypedDict`).

### `ServiceTier`

```python
ServiceTier = Literal['auto', 'default', 'flex', 'priority']
```

| Value | Meaning |
|---|---|
| `'auto'` | Provider decides — typically "use a higher tier when available". |
| `'default'` | Standard tier. Opts out of any server-side auto-promotion. |
| `'flex'` | Lower-cost, latency-tolerant. Provider-specific; silently ignored where unsupported. |
| `'priority'` | Higher-priority / lower-latency. Silently ignored where unsupported. |

Provider-specific settings (`openai_service_tier`, `anthropic_service_tier`, etc.) take
precedence over this unified field when set.

### `ThinkingLevel`

```python
ThinkingLevel = bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh']
```

| Value | Effect |
|---|---|
| `True` | Enable thinking at the provider's default effort. |
| `False` | Disable thinking (silently ignored on always-on reasoning models). |
| `'minimal'` … `'xhigh'` | Enable at a specific effort level. Unmapped levels round to nearest. |

### Example — combining both

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('anthropic:claude-opus-4-8')


async def main() -> None:
    result = await agent.run(
        'Solve this step-by-step: a train leaves at 09:00 at 120 km/h...',
        model_settings={
            'thinking': 'high',           # deep reasoning pass
            'service_tier': 'priority',   # low-latency capacity
            'max_tokens': 4096,
        },
    )
    print(result.output)
    print(f'Tokens: {result.usage.total_tokens}')


asyncio.run(main())
```

### Example — disabling thinking for cost savings

```python
import asyncio
from pydantic_ai import Agent

# On providers that support disabling thinking (e.g. Anthropic extended thinking
# models), thinking=False reduces cost. On always-on reasoning models like o4-mini
# it is silently ignored — the model still reasons. Use a non-reasoning model for
# tasks where reasoning overhead is undesirable.
agent = Agent('openai:o4-mini', model_settings={'thinking': False})


async def main() -> None:
    result = await agent.run('Is "Python" a programming language? Answer yes or no.')
    print(result.output)


asyncio.run(main())
```

---

## 6. `AgentStream` and `StreamedRunResult` — the streaming result objects

**Module:** `pydantic_ai.result`

`agent.run_stream()` returns a `StreamedRunResult` context manager. Inside the `async with`
block the `stream` variable is a `StreamedRunResult`, which wraps an `AgentStream` and
delegates its streaming methods to it.

`StreamedRunResult` provides three independent axes for consuming a streamed model response:

| Method | What you get |
|---|---|
| `stream_text(delta=False)` | Full cumulative text after each chunk |
| `stream_text(delta=True)` | Individual text deltas (chunks) as they arrive |
| `stream_output()` | Validated `OutputDataT` snapshots (partial → final) |
| `stream_response()` | Raw `ModelResponse` snapshots (unvalidated) |
| `get_output()` | Await the complete, validated final output |
| `cancel()` | Stop the stream and signal provider shutdown |
| `drain()` | Consume all remaining events (discard) |

Key properties: `run_id`, `conversation_id`, `metadata`, `usage`, `timestamp`, `cancelled`.

### Example 1 — streaming text with delta mode

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    async with agent.run_stream('Tell me a short story about a robot.') as stream:
        async for delta in stream.stream_text(delta=True):
            print(delta, end='', flush=True)
        print()  # newline after streaming
        print(f'\nTotal tokens: {stream.usage.total_tokens}')


asyncio.run(main())
```

### Example 2 — streaming structured output with partial validation

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent


class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    steps: list[str]


agent = Agent('openai:gpt-4o-mini', output_type=Recipe)


async def main() -> None:
    async with agent.run_stream('Give me a recipe for chocolate cake.') as stream:
        async for partial in stream.stream_output():
            # Yields partial Recipe objects as JSON arrives.
            print(f'Partial: title={partial.title!r}, steps_so_far={len(partial.steps)}')

    # After the context manager, stream.usage contains final token counts.
    print(f'run_id: {stream.run_id}')


asyncio.run(main())
```

### Example 3 — cancelling mid-stream

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    async with agent.run_stream('Count from 1 to 1000.') as stream:
        count = 0
        async for text in stream.stream_text(delta=True):
            print(text, end='', flush=True)
            count += 1
            if count > 10:
                await stream.cancel()
                break

    print(f'\nCancelled: {stream.cancelled}')


asyncio.run(main())
```

---

## 7. `SkipModelRequest`, `SkipToolValidation`, `SkipToolExecution`

**Module:** `pydantic_ai.exceptions`

These three exceptions are escape hatches for capability hooks. Raise them inside the
appropriate hook to short-circuit normal processing:

### `SkipModelRequest`

Raise in `before_model_request` / `wrap_model_request` to skip the actual LLM call and
substitute your own response. The provided `ModelResponse` is used instead.

```python
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart
from datetime import datetime

raise SkipModelRequest(
    ModelResponse(
        parts=[TextPart(content='Cached answer: Paris.')],
        model_name='cache',
        timestamp=datetime.utcnow(),
    )
)
```

### `SkipToolValidation`

Raise in `before_tool_validate` to bypass Pydantic schema validation and use pre-validated
args directly. `validated_args` is a `dict[str, Any]` that the tool function receives.

```python
from pydantic_ai.exceptions import SkipToolValidation

raise SkipToolValidation(validated_args={'query': 'already cleaned', 'limit': 10})
```

### `SkipToolExecution`

Raise in `before_tool_execute` to skip actual execution and return a pre-computed result
without running the tool function at all.

```python
from pydantic_ai.exceptions import SkipToolExecution

raise SkipToolExecution(result='Cached: 42.0 USD')
```

### Full example — a caching capability using SkipModelRequest and SkipToolExecution

Note: `SkipToolValidation` (for `before_tool_validate`) is a third escape hatch in the same
family. It is shown individually above; the caching pattern below combines only model-level
and tool-level caching to keep the example focused.

```python
import asyncio
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipModelRequest, SkipToolExecution
from pydantic_ai.messages import ModelResponse, TextPart

_model_cache: dict[str, str] = {}
_tool_cache: dict[str, str] = {}

hooks = Hooks()


@hooks.on.before_model_request
async def maybe_skip_model(ctx, request_context):
    key = str(ctx.messages)
    if key in _model_cache:
        raise SkipModelRequest(
            ModelResponse(
                parts=[TextPart(content=_model_cache[key])],
                model_name='cache',
                timestamp=datetime.utcnow(),
            )
        )
    return request_context  # must return request_context on cache miss


@hooks.on.before_tool_execute
async def maybe_skip_tool(ctx, *, call, tool_def, args):
    import json
    key = f'{tool_def.name}:{json.dumps(args, sort_keys=True)}'
    if key in _tool_cache:
        raise SkipToolExecution(result=_tool_cache[key])
    return args  # must return args (unchanged or modified)


agent = Agent('openai:gpt-4o-mini', capabilities=[hooks])


@agent.tool_plain
def weather(city: str) -> str:
    return f'Sunny in {city}'


async def main() -> None:
    import json
    _tool_cache[f'weather:{json.dumps({"city": "Paris"}, sort_keys=True)}'] = 'Cloudy in Paris (cached)'
    result = await agent.run("What's the weather in Paris?")
    print(result.output)


asyncio.run(main())
```

---

## 8. `ToolDefinition.sequential` and `ToolManager.parallel_execution_mode`

**Module:** `pydantic_ai.tools`, `pydantic_ai.tool_manager`

By default, all tool calls the model emits in a single turn run in parallel. Two
mechanisms give you serial control:

### `ToolDefinition.sequential` — per-tool barrier

Setting `sequential=True` on a tool definition makes that tool act as a *barrier*:
tools emitted before it complete first, then it runs alone, then tools emitted after
it start. Other tools still run in parallel around it.

The `sequential` parameter is available on `Tool(...)` and exposed via `ToolOutput.sequential`.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import Tool

agent = Agent('openai:gpt-4o-mini')


async def write_to_db(table: str, data: dict) -> str:
    """Write a record — must not overlap with other DB writes."""
    await asyncio.sleep(0.1)  # simulate DB write
    return f'Written {data} to {table}'


# sequential=True makes this tool a per-turn barrier: other tools in the same
# turn complete first, then this one runs alone, then any remaining tools start.
# This prevents overlapping DB writes within a single model turn.
# For cross-run exclusion, use an application-level lock or transaction.
agent.add_tool(Tool(write_to_db, sequential=True))


@agent.tool_plain
def read_from_db(table: str) -> list[dict]:
    """Read records from a table."""
    return [{'id': 1, 'value': 'x'}]


async def main() -> None:
    result = await agent.run('Read the users table then write a new entry.')
    print(result.output)


asyncio.run(main())
```

### `ToolManager.parallel_execution_mode` — run-level control

Use `ToolManager.parallel_execution_mode` as a context manager to switch all tools
in a run to sequential execution without touching individual tool definitions:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tool_manager import ToolManager

agent = Agent('openai:gpt-4o-mini')


@agent.tool_plain
def step_one(x: int) -> int:
    return x + 1


@agent.tool_plain
def step_two(x: int) -> int:
    return x * 2


async def main() -> None:
    # Force all tool calls to run one at a time, in order.
    with ToolManager.parallel_execution_mode('sequential'):
        result = await agent.run('Apply step_one then step_two to 5.')
    print(result.output)


asyncio.run(main())
```

Available modes: `'parallel'` (default), `'sequential'` (one at a time), 
`'parallel_ordered_events'` (parallel execution but events delivered in submission order).

---

## 9. `OutputContext` — inspect the output mechanism in hooks

**Module:** `pydantic_ai.output`

`OutputContext` is passed to output hooks and validators and describes *how* the model
produced the current output: which mode was used, what tool call (if any) triggered it,
and whether text or structured data was expected.

### Key fields

| Field | Type | Purpose |
|---|---|---|
| `mode` | `OutputMode` | `'text'`, `'tool'`, `'native'`, `'prompted'`, `'image'`, `'auto'` |
| `output_type` | `type \| None` | Resolved Python type (e.g. `MyModel`, `str`) |
| `object_def` | `OutputObjectDefinition \| None` | JSON schema + name + description for structured output |
| `has_function` | `bool` | Whether an output function will be called |
| `function_name` | `str \| None` | Name of the output function |
| `tool_call` | `ToolCallPart \| None` | The tool call part for tool-based output |
| `tool_def` | `ToolDefinition \| None` | The tool definition for tool-based output |
| `allows_text` | `bool` | Whether the schema accepts plain text |
| `allows_image` | `bool` | Whether the schema accepts image output |

### Example — inspecting output mode via an output lifecycle hook

`@agent.output_validator` accepts only `(output)` or `(RunContext, output)` — it does not
receive an `OutputContext`. To inspect the output mechanism, use the `after_output_validate`
capability hook, which is passed `output_context` as a keyword argument:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry


class Summary(BaseModel):
    headline: str
    detail: str


hooks = Hooks()


@hooks.on.after_output_validate
async def inspect_output(ctx, *, output_context, output):
    print(f'Output mode: {output_context.mode}')
    print(f'Has function: {output_context.has_function}')
    return output


agent = Agent('openai:gpt-4o-mini', output_type=Summary, capabilities=[hooks])


@agent.output_validator
async def validate_summary(ctx: RunContext[None], output: Summary) -> Summary:
    if not output.headline:
        raise ModelRetry('Headline cannot be empty.')
    return output


async def main() -> None:
    result = await agent.run('Summarise the history of Python in one sentence.')
    print(result.output)


asyncio.run(main())
```

---

## 10. `ApprovalRequired` — in-tool human-in-the-loop approval

**Module:** `pydantic_ai.exceptions`

`ApprovalRequired` is raised from a tool to signal that human approval is needed before
the action proceeds. It defers the tool call and surfaces a `DeferredToolRequests` object
as the run's output, with optional metadata the UI can display.

This is distinct from `ApprovalRequiredToolset` (which wraps a whole toolset) — raising
`ApprovalRequired` gives you fine-grained, conditional approval on a per-call basis.

```python
class ApprovalRequired(Exception):
    def __init__(self, metadata: dict[str, Any] | None = None): ...
```

The tool call is paused; resume it by passing `DeferredToolResults` (with `ToolApproved` or
`ToolDenied` per call) as `deferred_tool_results` to the next `agent.run()`.

### Example — tool that requires approval for destructive operations

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

# DeferredToolRequests must be included in output_type so the agent can surface
# the approval pause as structured output rather than raising at run time.
agent = Agent('openai:gpt-4o-mini', output_type=[str, DeferredToolRequests])


@agent.tool
def delete_record(ctx: RunContext[None], record_id: int) -> str:
    """Delete a database record. Requires human approval for IDs > 1000."""
    if record_id > 1000 and not ctx.tool_call_approved:
        # Raise only when the call has NOT yet been approved.
        # On the resume pass ctx.tool_call_approved is True, so we proceed.
        raise ApprovalRequired(
            metadata={
                'action': 'delete',
                'record_id': record_id,
                'risk': 'high',
                'message': f'About to delete record {record_id} — are you sure?',
            }
        )
    return f'Deleted record {record_id}'


async def main() -> None:
    # Step 1: run until we hit the approval gate.
    result = await agent.run('Delete record 1500.')

    if isinstance(result.output, DeferredToolRequests):
        deferred = result.output
        print('Approval required:')
        for call in deferred.approvals:
            meta = deferred.metadata.get(call.tool_call_id, {})
            print(f'  {call.tool_call_id}: {meta}')

        # Step 2 (in a real app, show the metadata to a human):
        approved = True  # human says yes

        # Build results using approvals= kwarg keyed by tool_call_id.
        tool_results = DeferredToolResults(
            approvals={
                call.tool_call_id: ToolApproved() if approved else ToolDenied('User declined.')
                for call in deferred.approvals
            }
        )

        # Step 3: resume the run.
        final = await agent.run(
            '',
            message_history=result.all_messages(),
            deferred_tool_results=tool_results,
        )
        print(final.output)


asyncio.run(main())
```

### Metadata access pattern

```python
# When DeferredToolRequests arrives, inspect per-call metadata:
if isinstance(result.output, DeferredToolRequests):
    for call_id, meta in result.output.metadata.items():
        risk = meta.get('risk', 'unknown')
        print(f'Call {call_id}: risk={risk}')
```

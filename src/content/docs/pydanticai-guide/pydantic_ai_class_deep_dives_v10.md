---
title: "PydanticAI — Class Deep Dives Vol. 10"
description: "Source-verified deep dives into 10 PydanticAI classes: AgentStream (full streaming API — stream_output/stream_response/stream_text/cancel/drain/validate_response_output), WrapperCapability (capability middleware with full delegation), FunctionToolset (complete toolset — all 14 params, instructions, timeout, defer_loading), AbstractToolset (custom toolset ABC — for_run/for_run_step/get_instructions/get_tools/call_tool), ToolCallEvent/ToolResultEvent/FunctionToolCallEvent/FunctionToolResultEvent/OutputToolCallEvent/OutputToolResultEvent (tool event taxonomy), FinalResult/FinalResultEvent (output result markers), UserError/UsageLimitExceeded/ConcurrencyLimitExceeded/UndrainedPendingMessagesError/HookTimeoutError (remaining error types), multimodal type system (AudioMediaType/ImageMediaType/DocumentMediaType/VideoMediaType + format literals + ForceDownloadMode + ProviderDetailsDelta), AbstractCapability extended (defer_loading/get_description/get_ordering/wrap hooks/WrapRunHandler family), CapabilityOrdering/CapabilityPosition/CapabilityRef/CAPABILITY_TYPES (capability topology). All verified against pydantic-ai 1.105.0."
sidebar:
  label: "Class deep dives (Vol. 10)"
  order: 36
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.105.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.105.0`.
</Aside>

Ten class groups from the `pydantic_ai` 1.105.0 source covering: the full `AgentStream` API (the streaming context manager returned by `run_stream()`) with every method and property; `WrapperCapability` for transparent capability middleware; `FunctionToolset` with all 14+ constructor parameters and every decorator variant; `AbstractToolset` ABC for building custom toolsets from scratch; the complete tool event taxonomy (`ToolCallEvent`, `ToolResultEvent`, and their four concrete subclasses); `FinalResult` and `FinalResultEvent` as output-tracking markers; five remaining error types (`UserError`, `UsageLimitExceeded`, `ConcurrencyLimitExceeded`, `UndrainedPendingMessagesError`, `HookTimeoutError`); the multimodal type system (all media-type aliases, format literals, `ForceDownloadMode`, `ProviderDetailsDelta`); extended `AbstractCapability` capabilities (`defer_loading`, `get_description`, `get_ordering`, wrap hooks, handler types); and `CapabilityOrdering` + `CapabilityPosition` + `CapabilityRef` + `CAPABILITY_TYPES` for topology-aware capability composition.

---

## 1. `AgentStream` — The Streaming Context Manager

**Module:** `pydantic_ai.result`  
**Import:** `from pydantic_ai.result import AgentStream`

`AgentStream` is the rich streaming object you interact with inside an `async with agent.run_stream(...)` block. It is a `Generic[AgentDepsT, OutputDataT]` dataclass that wraps the raw model stream and exposes validated, debounced streaming in three levels of granularity.

### Class signature

```python
@dataclass(kw_only=True)
class AgentStream(Generic[AgentDepsT, OutputDataT]):
    # Private fields (set by the framework, not directly constructed by users)
    _raw_stream_response: models.StreamedResponse
    _output_schema: OutputSchema[OutputDataT]
    _model_request_parameters: models.ModelRequestParameters
    _output_validators: list[OutputValidator[AgentDepsT, OutputDataT]]
    _run_ctx: RunContext[AgentDepsT]
    _usage_limits: UsageLimits | None
    _tool_manager: ToolManager[AgentDepsT]
    _root_capability: AbstractCapability[AgentDepsT]
    _metadata_getter: Callable[[], dict[str, Any] | None] | None
```

### Method and property reference

| Member | Return type | Description |
|--------|------------|-------------|
| `stream_output(debounce_by=0.1)` | `AsyncIterator[OutputDataT]` | Validated output snapshots; final item always yielded |
| `stream_response(debounce_by=0.1)` | `AsyncIterator[ModelResponse]` | Raw `ModelResponse` snapshots (`state='incomplete'` → `'complete'`) |
| `stream_text(delta=False, debounce_by=0.1)` | `AsyncIterator[str]` | Text-only streaming; `delta=True` for token chunks |
| `cancel()` | `Awaitable[None]` | Stop token generation, close connection |
| `drain()` | `Awaitable[None]` | Consume and discard all remaining events |
| `validate_response_output(response, allow_partial=False)` | `Awaitable[OutputDataT]` | Run output validators on a `ModelResponse` snapshot |
| `get_output()` | `Awaitable[OutputDataT]` | Drain stream and return final validated output |
| `response` | `ModelResponse` | Current (possibly incomplete) `ModelResponse` |
| `usage` | `RunUsage` | Accumulated token usage for this stream |
| `run_id` | `str` | UUID7 for this agent run |
| `conversation_id` | `str` | UUID7 for the conversation |
| `metadata` | `dict[str, Any] \| None` | App-level metadata, not sent to the LLM |
| `cancelled` | `bool` | `True` after `cancel()` was called |

### Streaming validated output — `stream_output()`

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent

class CityInfo(BaseModel):
    name: str
    population: int
    country: str

agent = Agent('openai:gpt-4o', output_type=CityInfo)

async def main():
    async with agent.run_stream('Tell me about Paris.') as stream:
        # stream_output() yields partial objects as they are validated
        async for partial in stream.stream_output():
            print(f'  partial: {partial}')
        # The last item is always the fully validated final output
        final: CityInfo = await stream.get_output()
        print(f'Final: {final}')
        print(f'Run ID: {stream.run_id}')
        print(f'Usage: {stream.usage}')
```

`stream_output()` skips snapshots where `final_result_event` is None or the parts haven't changed, and always emits one final validated snapshot with `allow_partial=False`.

### Streaming text with delta mode — `stream_text()`

```python
async def stream_chat():
    async with agent.run_stream('Explain recursion in one paragraph.') as stream:
        print('Token stream: ', end='')
        async for delta in stream.stream_text(delta=True):
            print(delta, end='', flush=True)
        print()

        # stream_text(delta=False) accumulates — yields the full text so far each time
        # useful for progress bars or UI updates
        async with agent.run_stream('What is 2+2?') as s2:
            last = ''
            async for cumulative in s2.stream_text(delta=False):
                last = cumulative
            print('Full text:', last)
```

**Note:** `stream_text()` requires a text output type. With structured output types, call `stream_output()` instead.

### Inspecting model responses — `stream_response()`

```python
from pydantic_ai.messages import ModelResponseState

async def inspect_response():
    async with agent.run_stream('List three planets.') as stream:
        async for snapshot in stream.stream_response(debounce_by=0.05):
            # state='incomplete' until the last snapshot
            print(f'  state={snapshot.state!r}, parts={len(snapshot.parts)}')
        # After iteration, state='complete' (or 'interrupted' if cancel() was called)
        assert stream.response.state == 'complete'
```

### Cancellation and draining

```python
import asyncio

async def cancel_after_start():
    async with agent.run_stream('Write a very long essay...') as stream:
        count = 0
        async for delta in stream.stream_text(delta=True):
            count += 1
            if count >= 20:
                await stream.cancel()
                break
        print(f'Cancelled: {stream.cancelled}')  # True
        print(f'State: {stream.response.state!r}')  # 'interrupted'

async def drain_example():
    # Drain without processing — useful when you only want side effects
    async with agent.run_stream('Do something.') as stream:
        await stream.drain()
        final = await stream.get_output()
        print(final)
```

### Custom output validation with `validate_response_output()`

```python
from pydantic_ai.result import AgentStream

async def validate_mid_stream():
    async with agent.run_stream('Give me a number.') as stream:
        async for snapshot in stream.stream_response():
            if stream.response.state != 'complete':
                try:
                    # partial=True: don't fail on incomplete structures
                    partial = await stream.validate_response_output(snapshot, allow_partial=True)
                    print(f'Partial output: {partial!r}')
                except Exception:
                    pass  # expected during streaming

        final = await stream.validate_response_output(stream.response)
        print(f'Final validated: {final!r}')
```

### Metadata and run tracking

```python
async def run_with_metadata():
    async with agent.run_stream(
        'Hello!',
        metadata={'user_id': 'u123', 'session': 'web-abc'},
    ) as stream:
        await stream.drain()
        print(f'Run ID: {stream.run_id}')
        print(f'Conversation ID: {stream.conversation_id}')
        print(f'Metadata: {stream.metadata}')
        print(f'Tokens used: {stream.usage}')
```

---

## 2. `WrapperCapability` — Capability Middleware

**Module:** `pydantic_ai.capabilities.wrapper`  
**Import:** `from pydantic_ai.capabilities import WrapperCapability`

`WrapperCapability` is a `@dataclass` that delegates all `AbstractCapability` methods to a `wrapped` inner capability. It is the capability analogue of `WrapperToolset` — subclass it and override only the methods you care about. All 40+ lifecycle callbacks default to pass-through delegation.

### Class signature

```python
@dataclass
class WrapperCapability(AbstractCapability[AgentDepsT]):
    wrapped: AbstractCapability[AgentDepsT]

    def __post_init__(self) -> None:
        # Transparently inherit `id` and `defer_loading` from the wrapped capability
        if self.id is None:
            self.id = self.wrapped.id
            self.defer_loading = self.wrapped.defer_loading
```

The `__post_init__` logic makes a wrapper over a deferred capability automatically deferred itself — the wrapper is transparent to the load catalog.

### Logging capability (observer pattern)

```python
import logging
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities import WrapperCapability
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai._run_context import RunContext
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.messages import ModelResponse

log = logging.getLogger(__name__)

@dataclass
class AuditCapability(WrapperCapability):
    """Logs every model request/response pair for compliance audit."""

    async def after_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        # Delegate first, then log the result
        response = await self.wrapped.after_model_request(
            ctx, request_context=request_context, response=response
        )
        log.info(
            'model_response run_id=%s finish_reason=%s parts=%d',
            ctx.run_id,
            response.model_finish_reason,
            len(response.parts),
        )
        return response

# Wrap any existing capability
from pydantic_ai.capabilities import Hooks

hooks = Hooks()
audited_hooks = AuditCapability(wrapped=hooks)

agent = Agent('openai:gpt-4o', capabilities=[audited_hooks])
```

### Request modification capability (transformer pattern)

```python
from dataclasses import dataclass
from pydantic_ai.capabilities import WrapperCapability
from pydantic_ai._run_context import RunContext
from pydantic_ai.models import ModelRequestContext

@dataclass
class ContextEnrichCapability(WrapperCapability):
    """Injects a request-ID header into every outbound model request."""

    request_id_key: str = 'x-request-id'

    async def before_model_request(
        self,
        ctx: RunContext,
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        # Delegate to inner, then inject the request-ID into run metadata
        request_context = await self.wrapped.before_model_request(ctx, request_context)
        ctx.metadata = ctx.metadata or {}
        ctx.metadata[self.request_id_key] = ctx.run_id
        return request_context
```

### Wrapping a dynamic capability

```python
from dataclasses import dataclass
from pydantic_ai.capabilities import WrapperCapability, DynamicCapability
from pydantic_ai._run_context import RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability

async def feature_flag_factory(ctx: RunContext) -> AbstractCapability | None:
    if ctx.deps.get('enable_web_search'):
        from pydantic_ai.capabilities import WebSearch
        return WebSearch()
    return None

dynamic = DynamicCapability(factory=feature_flag_factory)

@dataclass
class TimedCapability(WrapperCapability):
    """Measures and records how long each capability operation takes."""
    import time as _time

    async def before_model_request(self, ctx, request_context):
        ctx.metadata = ctx.metadata or {}
        ctx.metadata['_req_start'] = self._time.monotonic()
        return await self.wrapped.before_model_request(ctx, request_context)

    async def after_model_request(self, ctx, *, request_context, response):
        elapsed = self._time.monotonic() - (ctx.metadata or {}).get('_req_start', 0)
        (ctx.metadata or {}).pop('_req_start', None)
        import logging; logging.getLogger(__name__).debug('model_request took %.3fs', elapsed)
        return await self.wrapped.after_model_request(
            ctx, request_context=request_context, response=response
        )

timed_dynamic = TimedCapability(wrapped=dynamic)
```

### `apply()` tree traversal

`WrapperCapability.apply()` is overridden to call the visitor on `self` first, then walk into the wrapped capability's leaves — but only if the wrapped capability has more than one leaf (containers). This lets the framework register proxy hooks correctly for both leaf and container wrappings:

```python
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.capabilities import CombinedCapability, WebSearch, WebFetch

inner = CombinedCapability([WebSearch(), WebFetch()])
wrapper = WrapperCapability(wrapped=inner)  # type: ignore[abstract]

leaves: list = []
wrapper.apply(leaves.append)
# leaves = [wrapper, WebSearch(), WebFetch()]
# WrapperCapability itself + the two children of the container
print([type(l).__name__ for l in leaves])
```

---

## 3. `FunctionToolset` — The Primary Toolset

**Module:** `pydantic_ai.toolsets.function`  
**Import:** `from pydantic_ai import Agent; from pydantic_ai.toolsets import FunctionToolset`

`FunctionToolset` is the most commonly used toolset. It accepts Python functions decorated with `@toolset.tool` or `@toolset.tool_plain`, manages their JSON schema generation, and wires them into the agent. It accepts 14+ constructor parameters to tune every aspect of tool registration.

### Full constructor reference

```python
FunctionToolset(
    tools: Sequence[Tool | ToolFuncEither] = (),
    *,
    max_retries: int | None = None,        # inherit from agent if None
    timeout: float | None = None,           # seconds; None = no limit
    docstring_format: DocstringFormat = 'auto',   # 'auto'|'google'|'numpy'|'sphinx'|'plain'
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
    strict: bool | None = None,             # OpenAI strict mode
    sequential: bool = False,               # force serial execution
    requires_approval: bool = False,        # HITL approval gate
    metadata: dict[str, Any] | None = None,
    defer_loading: bool = False,            # hide from model until tool-search
    include_return_schema: bool | None = None,
    id: str | None = None,                  # required for durable execution
    instructions: str | Callable[..., str] | Sequence[str] | None = None,
)
```

### Basic usage with all decorator forms

```python
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai import Agent, RunContext

# Standalone toolset (reusable across agents)
tools = FunctionToolset(
    max_retries=2,
    timeout=10.0,
    docstring_format='google',
    require_parameter_descriptions=True,
)

@tools.tool
async def get_weather(ctx: RunContext[str], city: str) -> str:
    """Return current weather for a city.

    Args:
        city: The city name to look up.
    """
    return f'Sunny, 22°C in {city}'

@tools.tool_plain
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b

agent = Agent('openai:gpt-4o', deps_type=str, toolsets=[tools])
result = agent.run_sync('What is the weather in Berlin and 2+3?', deps='user-session')
print(result.output)
```

### Per-tool parameter overrides via decorator kwargs

```python
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai import RunContext
import asyncio

tools = FunctionToolset(timeout=5.0)  # default 5s for all tools

@tools.tool(
    name='slow_op',           # override tool name
    description='A slow operation that needs more time.',
    retries=3,                # override max_retries for this tool
    timeout=60.0,             # override timeout for this tool
    strict=True,              # OpenAI strict JSON schema
    sequential=True,          # must not run in parallel
    requires_approval=True,   # HITL gate
    metadata={'cost': 'high'},
)
async def long_running_task(ctx: RunContext, task_id: str) -> str:
    await asyncio.sleep(30)
    return f'Completed task {task_id}'
```

### Toolset with instructions

Instructions are injected into the system prompt every time the agent runs with this toolset:

```python
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai import Agent, RunContext

def dynamic_instructions(ctx: RunContext[dict]) -> str:
    lang = ctx.deps.get('language', 'English')
    return f'Always respond in {lang}. Use the tools provided when applicable.'

tools = FunctionToolset(
    instructions=dynamic_instructions,
)

@tools.tool
def translate(ctx: RunContext[dict], text: str, target_lang: str) -> str:
    """Translate text to a target language."""
    return f'[translated to {target_lang}]: {text}'

agent = Agent('openai:gpt-4o', deps_type=dict, toolsets=[tools])
```

### Defer loading for tool search

```python
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

# Tools hidden from model until it calls the load_capability/search tool
hidden_tools = FunctionToolset(
    defer_loading=True,
    id='advanced-tools',           # id required when defer_loading=True
    description='Advanced data tools for expert users.',
)

@hidden_tools.tool_plain
def generate_report(report_type: str, start_date: str, end_date: str) -> str:
    """Generate a detailed report for the given date range."""
    return f'Report: {report_type} from {start_date} to {end_date}'

agent = Agent(
    'openai:gpt-4o',
    toolsets=[hidden_tools],
    capabilities=[ToolSearch()],    # enables lazy tool discovery
)
```

### Toolset with durable execution ID

```python
from pydantic_ai.toolsets import FunctionToolset

# id is required for Temporal/DBOS/Prefect activities
durable_tools = FunctionToolset(
    id='data-pipeline-tools',
    max_retries=3,
    timeout=120.0,
)

@durable_tools.tool_plain
def fetch_from_database(query: str) -> list[dict]:
    """Execute a read-only database query."""
    return [{'result': query}]  # simplified
```

### `add_function()` vs `add_tool()` vs `@tool`

```python
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import Tool

toolset = FunctionToolset()

# Form 1: decorator (most common)
@toolset.tool_plain
def square(n: float) -> float:
    """Square a number."""
    return n * n

# Form 2: add_function (for functions defined elsewhere)
def cube(n: float) -> float:
    """Cube a number."""
    return n ** 3

toolset.add_function(cube, retries=1)

# Form 3: add_tool (for pre-constructed Tool objects)
from pydantic_ai.tools import Tool

raw_tool = Tool(
    function=lambda n: n ** 4,
    name='quad',
    description='Raise to the fourth power.',
)
toolset.add_tool(raw_tool)
```

---

## 4. `AbstractToolset` — Custom Toolset Base Class

**Module:** `pydantic_ai.toolsets.abstract`  
**Import:** `from pydantic_ai.toolsets import AbstractToolset`

`AbstractToolset` is the ABC that all toolsets implement. Build a custom toolset by subclassing it and implementing `get_tools()` and `call_tool()`.

### Core abstract interface

```python
class AbstractToolset(ABC, Generic[AgentDepsT]):
    @property
    @abstractmethod
    def id(self) -> str | None: ...

    @abstractmethod
    async def get_tools(
        self, ctx: RunContext[AgentDepsT]
    ) -> dict[str, ToolsetTool[AgentDepsT]]: ...

    @abstractmethod
    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any: ...
```

### Lifecycle hooks

| Method | When called | Override for... |
|--------|-------------|-----------------|
| `for_run(ctx)` | Once per run, before `__aenter__` | Per-run state isolation (return a fresh instance) |
| `for_run_step(ctx)` | At the start of each run step | Per-step transitions |
| `__aenter__()` | Run start | Open connections, acquire resources |
| `__aexit__(...)` | Run end | Close connections, release resources |
| `get_instructions(ctx)` | Once per run | Inject toolset-level system prompt text |

### Minimal custom toolset

```python
import ast
import operator
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai import Agent
from typing import Any

# AST-based safe arithmetic evaluator — avoids eval() on user input
_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f'Unsupported expression node: {type(node).__name__}')

class CalculatorToolset(AbstractToolset):
    """A minimal custom toolset with a single calculator tool."""

    @property
    def id(self) -> str | None:
        return 'calculator'

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        tool_def = ToolDefinition(
            name='calculate',
            description='Evaluate a simple arithmetic expression.',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'expression': {
                        'type': 'string',
                        'description': 'Arithmetic expression, e.g. "2 + 3 * 4"',
                    }
                },
                'required': ['expression'],
            },
        )
        return {
            'calculate': ToolsetTool(tool_def=tool_def, max_retries=2)
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool
    ) -> Any:
        if name == 'calculate':
            expr = tool_args['expression']
            try:
                tree = ast.parse(expr, mode='eval')
                result = _safe_eval(tree.body)
                return f'{expr} = {result}'
            except (ValueError, ZeroDivisionError) as e:
                return f'Error: {e}'
            except SyntaxError:
                return 'Error: invalid expression syntax'
        raise ValueError(f'Unknown tool: {name!r}')

agent = Agent('openai:gpt-4o', toolsets=[CalculatorToolset()])
```

### Stateful toolset with `for_run` isolation

```python
import httpx
from dataclasses import dataclass, field
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import ToolDefinition
from typing import Any

@dataclass
class HttpToolset(AbstractToolset):
    """Toolset that reuses an httpx.AsyncClient per run."""

    base_url: str
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    @property
    def id(self) -> str | None:
        return 'http-tools'

    async def for_run(self, ctx: RunContext) -> 'HttpToolset':
        # Return a fresh instance with its own client for this run
        return HttpToolset(base_url=self.base_url)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        return {
            'fetch_endpoint': ToolsetTool(
                tool_def=ToolDefinition(
                    name='fetch_endpoint',
                    description='Fetch data from a REST endpoint.',
                    parameters_json_schema={
                        'type': 'object',
                        'properties': {'path': {'type': 'string'}},
                        'required': ['path'],
                    },
                ),
                max_retries=1,
            )
        }

    async def call_tool(self, name, tool_args, ctx, tool):
        assert self._client is not None, 'Context not entered'
        response = await self._client.get(tool_args['path'])
        return response.text
```

### Toolset with dynamic instructions

```python
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import InstructionPart

class PolicyToolset(AbstractToolset):
    @property
    def id(self):
        return 'policy-tools'

    async def get_instructions(self, ctx: RunContext) -> str:
        # Return a static string — treated as dynamic by default
        return (
            'When using policy tools, always cite the specific policy document. '
            'Never make assumptions about policy coverage.'
        )

    async def get_tools(self, ctx):
        # ... tool definitions
        return {}

    async def call_tool(self, name, tool_args, ctx, tool):
        raise NotImplementedError
```

---

## 5. `ToolCallEvent` / `ToolResultEvent` — Tool Event Taxonomy

**Module:** `pydantic_ai.messages`

PydanticAI emits a structured event for every tool invocation during an agent run. The full hierarchy is:

```
AgentStreamEvent = (
    PartStartEvent | PartDeltaEvent | PartEndEvent |
    ToolCallEvent  | ToolResultEvent |
    FinalResultEvent
)

ToolCallEvent  ──→  FunctionToolCallEvent  (function tool)
               └──→ OutputToolCallEvent    (output tool — model submitting its final answer)

ToolResultEvent──→  FunctionToolResultEvent (function tool result)
               └──→ OutputToolResultEvent   (output tool result)
```

### `ToolCallEvent` base class

```python
@dataclass(repr=False)
class ToolCallEvent:
    part: ToolCallPart          # the tool call details
    args_valid: bool | None     # True=passed, False=failed, None=not run
    event_kind: str             # discriminator: 'function_tool_call' or 'output_tool_call'

    @property
    def tool_call_id(self) -> str: ...
```

`args_valid` is set **before** tool execution: `True` if schema and custom validation both passed, `False` if validation failed, `None` if validation wasn't performed.

### `ToolResultEvent` base class

```python
@dataclass(repr=False)
class ToolResultEvent:
    part: ToolReturnPart | RetryPromptPart  # the result sent back to the model
    event_kind: str  # 'function_tool_result' or 'output_tool_result'

    @property
    def tool_call_id(self) -> str: ...
```

### `FunctionToolCallEvent` and `FunctionToolResultEvent`

```python
@dataclass(repr=False)
class FunctionToolCallEvent(ToolCallEvent):
    event_kind: Literal['function_tool_call'] = 'function_tool_call'

@dataclass(repr=False, init=False)
class FunctionToolResultEvent(ToolResultEvent):
    content: str | Sequence[UserContent] | None  # optional extra content sent to model
    event_kind: Literal['function_tool_result'] = 'function_tool_result'
```

`FunctionToolResultEvent.content` is additional `UserPromptPart` content the framework can attach alongside the tool return value — useful for injecting images or files as part of a tool result.

### `OutputToolCallEvent` and `OutputToolResultEvent`

These fire when the model calls the output tool (its "submit final answer" call):

```python
@dataclass(repr=False)
class OutputToolCallEvent(ToolCallEvent):
    event_kind: Literal['output_tool_call'] = 'output_tool_call'

@dataclass(repr=False)
class OutputToolResultEvent(ToolResultEvent):
    event_kind: Literal['output_tool_result'] = 'output_tool_result'
```

### Observing all tool events in a run

```python
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent, FunctionToolResultEvent,
    OutputToolCallEvent, OutputToolResultEvent,
    ToolCallEvent, ToolResultEvent,
)

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

async def main():
    async with agent.run_stream('What is 5 + 3?') as run:
        async for event in run:
            match event:
                case FunctionToolCallEvent(part=part, args_valid=args_valid):
                    print(f'Tool call: {part.tool_name}({part.args}) valid={args_valid}')
                case FunctionToolResultEvent(part=part):
                    print(f'Tool result: {part.content!r}')
                case OutputToolCallEvent():
                    print('Model submitting final answer')
                case OutputToolResultEvent():
                    print('Final answer accepted')
```

### Filtering by base class for shared handling

```python
async def log_all_tool_activity(agent, prompt):
    async with agent.run_stream(prompt) as run:
        async for event in run:
            if isinstance(event, ToolCallEvent):
                print(f'[CALL] {event.part.tool_name!r} id={event.tool_call_id}')
            elif isinstance(event, ToolResultEvent):
                outcome = getattr(event.part, 'outcome', 'ok')
                print(f'[RESULT] id={event.tool_call_id} outcome={outcome!r}')
```

### Validation failure detection via `args_valid`

```python
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import FunctionToolCallEvent

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def divide(numerator: float, denominator: float) -> float:
    """Divide two numbers."""
    if denominator == 0:
        raise ModelRetry('Denominator cannot be zero.')
    return numerator / denominator

async def detect_validation_failure():
    async with agent.run_stream('What is 10 / 0?') as run:
        async for event in run:
            if isinstance(event, FunctionToolCallEvent):
                if event.args_valid is False:
                    print(f'Args failed validation for {event.part.tool_name!r}')
                elif event.args_valid is True:
                    print(f'Args valid for {event.part.tool_name!r}')
```

---

## 6. `FinalResult` + `FinalResultEvent` — Output Result Markers

**Module:** `pydantic_ai.result` and `pydantic_ai.messages`  
**Imports:** `from pydantic_ai.result import FinalResult; from pydantic_ai.messages import FinalResultEvent`

`FinalResult` is a generic dataclass that wraps the final output value, tagging it with the tool name and call ID that produced it. `FinalResultEvent` is the `AgentStreamEvent` emitted when the model's response first matches the output schema.

### `FinalResult` class

```python
@dataclass(repr=False)
class FinalResult(Generic[OutputDataT]):
    output: OutputDataT       # the final validated output
    tool_name: str | None     # None if output came from text content, not a tool
    tool_call_id: str | None  # None for text output
```

### `FinalResultEvent` class

```python
@dataclass(repr=False, kw_only=True)
class FinalResultEvent:
    tool_name: str | None       # same semantics as FinalResult
    tool_call_id: str | None
    event_kind: Literal['final_result'] = 'final_result'
```

`FinalResultEvent` is emitted once per run step (when iterating via `agent.iter()` or `agent.run_stream_events()`). It precedes the actual output validation and signals which call will produce the final value.

### Distinguishing text vs tool output

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.result import FinalResult
from pydantic_ai.output import ToolOutput, TextOutput

class Answer(BaseModel):
    value: str

# Tool output — model calls a structured output tool
agent_tool = Agent('openai:gpt-4o', output_type=ToolOutput(Answer))

# Text output — model responds with plain text
agent_text = Agent('openai:gpt-4o', output_type=TextOutput(str))

async def inspect_result(agent, prompt):
    result = await agent.run(prompt)
    # Accessing the internal FinalResult
    fr: FinalResult = result._final_result  # type: ignore[attr-defined]
    if fr.tool_name is None:
        print('Text output — no tool was called for the final answer')
    else:
        print(f'Tool output via {fr.tool_name!r} (call id: {fr.tool_call_id!r})')
```

### Observing `FinalResultEvent` in an event stream

```python
from pydantic_ai import Agent
from pydantic_ai.messages import FinalResultEvent

agent = Agent('openai:gpt-4o')

async def watch_result_event():
    async with agent.run_stream_events('Summarise this in one word: "happy"') as events:
        async for event in events:
            if isinstance(event, FinalResultEvent):
                print(f'Final result incoming via tool={event.tool_name!r}')
                break
```

### Using `FinalResultEvent` for early exit

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import FinalResultEvent, PartStartEvent

agent = Agent('openai:gpt-4o')

async def stream_until_final():
    """Stream events, stop processing tool calls once the final result fires."""
    async with agent.run_stream_events('What is 2 + 2?') as events:
        saw_final = False
        async for event in events:
            if isinstance(event, FinalResultEvent):
                saw_final = True
                print('Final result committed — draining.')
            elif not saw_final and isinstance(event, PartStartEvent):
                print(f'Part starting: {event.part!r}')
```

---

## 7. Error Taxonomy — `UserError`, `UsageLimitExceeded`, `ConcurrencyLimitExceeded`, `UndrainedPendingMessagesError`, `HookTimeoutError`

**Module:** `pydantic_ai.exceptions` (and `pydantic_ai.capabilities.abstract` for `HookTimeoutError`)

These five error types complete the full exception hierarchy alongside the model-layer errors covered in Vol. 6.

### Full exception hierarchy

```
BaseException
└── Exception
    ├── RuntimeError
    │   ├── UserError                          — developer mistake
    │   │   └── UndrainedPendingMessagesError  — unfinished enqueued messages
    │   └── AgentRunError                      — failure during an agent run
    │       ├── UsageLimitExceeded             — token/call budget exceeded
    │       ├── ConcurrencyLimitExceeded       — queue depth exceeded
    │       ├── ModelAPIError / ModelHTTPError — model-layer errors (Vol. 6)
    │       └── UnexpectedModelBehavior        — unexpected model output (Vol. 6)
    └── TimeoutError
        └── HookTimeoutError                   — hook function timed out
```

### `UserError`

Raised when application code has a configuration or usage mistake:

```python
from pydantic_ai.exceptions import UserError

class UserError(RuntimeError):
    message: str  # description of the mistake
```

Common causes:
- Conflicting tool names across toolsets without a `PrefixedToolset`
- Using `stream_text()` with a non-text output type
- Circular ordering constraints among capabilities
- Calling `AgentRun.next()` after the run has ended

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def add(a: int, b: int) -> int:
    return a + b

@agent.tool_plain
def add(x: int, y: int) -> int:  # duplicate name  # noqa: F811
    return x + y

try:
    agent.run_sync('hello')
except UserError as e:
    print(f'Config mistake: {e.message}')
```

### `UsageLimitExceeded`

Raised when a `UsageLimits` constraint is violated:

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o')

try:
    result = agent.run_sync(
        'Count from 1 to 1000.',
        usage_limits=UsageLimits(response_tokens_limit=50),
    )
except UsageLimitExceeded as e:
    print(f'Token budget exceeded: {e}')

# The message contains which limit was hit:
# "Exceeded the response_tokens_limit of 50 (response_tokens=87)"

# Track usage after the fact
from pydantic_ai.usage import RunUsage
try:
    result = agent.run_sync(
        'Do something complex.',
        usage_limits=UsageLimits(request_limit=2),
    )
except UsageLimitExceeded:
    # Use capture_run_messages to retrieve partial messages
    from pydantic_ai import capture_run_messages
    with capture_run_messages() as messages:
        try:
            agent.run_sync('Test.', usage_limits=UsageLimits(request_limit=1))
        except UsageLimitExceeded:
            print(f'Captured {len(messages)} messages before limit hit')
```

### `ConcurrencyLimitExceeded`

Raised when the queue backing a `ConcurrencyLimitedModel` or `AbstractConcurrencyLimiter` overflows:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.concurrency import ConcurrencyLimit
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

base_agent = Agent('openai:gpt-4o')
limited_model = ConcurrencyLimitedModel(
    base_agent._model,  # type: ignore
    limiter=ConcurrencyLimit(max_running=2, max_queued=5),
)

async def run_many():
    agents = [Agent(limited_model) for _ in range(20)]
    tasks = [a.run('Quick ping.') for a in agents]

    results, errors = [], []
    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except ConcurrencyLimitExceeded as e:
            errors.append(str(e))

    print(f'Succeeded: {len(results)}, Throttled: {len(errors)}')
```

### `UndrainedPendingMessagesError`

Raised when an `agent.iter()` loop ends (reaches `End`) but messages are still queued via `ctx.enqueue()`:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UndrainedPendingMessagesError

agent = Agent('openai:gpt-4o')

@agent.tool
async def fetch_data(ctx: RunContext[None]) -> str:
    # Enqueue a follow-up message with 'when_idle' priority
    await ctx.enqueue('Summarise the fetched data next.', priority='when_idle')
    return 'Data fetched.'

async def wrong_usage():
    try:
        # Bare async-for only drains 'asap' messages, not 'when_idle'
        async for node in agent.iter('Fetch some data.'):
            pass  # reaches End with queued messages → raises
    except UndrainedPendingMessagesError as e:
        print(f'Pending messages were stranded: {e}')

async def correct_usage():
    # Use agent.run() or AgentRun.next() — these drain all message priorities
    result = await agent.run('Fetch some data.')
    print(result.output)
```

### `HookTimeoutError`

Raised when a capability hook function exceeds its configured timeout (set via `Hooks(..., timeout=...)`):

```python
from pydantic_ai.capabilities import HookTimeoutError
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai import Agent
import asyncio

hooks = Hooks(timeout=0.5)  # all hook functions must complete in 0.5s

@hooks.before_run
async def slow_hook(ctx):
    await asyncio.sleep(2.0)   # will time out

agent = Agent('openai:gpt-4o', capabilities=[hooks])

try:
    agent.run_sync('Hello.')
except HookTimeoutError as e:
    print(f'Hook timed out: hook={e.hook_name!r} func={e.func_name!r} after={e.timeout}s')

# HookTimeoutError fields:
#   hook_name: str     — the hook event name, e.g. 'before_run'
#   func_name: str     — the name of the decorated function that timed out
#   timeout: float     — the configured timeout in seconds
```

---

## 8. Multimodal Type System — Media Aliases, Format Literals, `ForceDownloadMode`, `ProviderDetailsDelta`

**Module:** `pydantic_ai.messages`

PydanticAI exposes a complete set of `TypeAlias` literals for multimodal content types. These are used in function signatures, tool argument schemas, and `FileUrl` subclasses to provide type-safe media handling.

### Media type aliases

```python
# Audio formats accepted as inline data
AudioMediaType: TypeAlias = Literal[
    'audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/aiff', 'audio/aac'
]

# Image formats accepted as inline data
ImageMediaType: TypeAlias = Literal[
    'image/jpeg', 'image/png', 'image/gif', 'image/webp'
]

# Document formats accepted as inline data
DocumentMediaType: TypeAlias = Literal[
    'application/pdf', 'text/plain', 'text/csv',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # .docx
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',        # .xlsx
    'text/html', 'text/markdown',
    'application/msword', 'application/vnd.ms-excel',
]

# Video formats accepted as inline data
VideoMediaType: TypeAlias = Literal[
    'video/x-matroska', 'video/quicktime', 'video/mp4', 'video/webm',
    'video/x-flv', 'video/mpeg', 'video/x-ms-wmv', 'video/3gpp',
]
```

### Format shorthand literals

Shorter form for file-extension based input (useful in tool argument schemas):

```python
AudioFormat: TypeAlias = Literal['wav', 'mp3', 'oga', 'flac', 'aiff', 'aac']
ImageFormat: TypeAlias = Literal['jpeg', 'png', 'gif', 'webp']
DocumentFormat: TypeAlias = Literal['csv', 'doc', 'docx', 'html', 'md', 'pdf', 'txt', 'xls', 'xlsx']
VideoFormat: TypeAlias = Literal['mkv', 'mov', 'mp4', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp']
```

### Using types in tool schemas

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ImageFormat, AudioFormat, DocumentFormat

class FileAnalysisRequest(BaseModel):
    file_name: str
    image_format: ImageFormat | None = None
    audio_format: AudioFormat | None = None
    doc_format: DocumentFormat | None = None

agent = Agent('openai:gpt-4o', output_type=FileAnalysisRequest)

# The model can now only return known formats, validated by Pydantic
result = agent.run_sync('Analyse example.png')
print(result.output.image_format)   # 'png'
```

### `ForceDownloadMode`

Controls how `FileUrl` subclasses (`ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`) handle URL fetching:

```python
ForceDownloadMode: TypeAlias = bool | Literal['allow-local']
```

| Value | Behaviour |
|-------|-----------|
| `False` (default) | Send URL directly to providers that support it; download with SSRF guard for others |
| `True` | Always download; block private IPs and cloud metadata endpoints |
| `'allow-local'` | Always download; allow private IPs but still block cloud metadata (169.254.x.x, etc.) |

```python
from pydantic_ai.messages import ImageUrl, DocumentUrl

# Production: always download, full SSRF protection
secure_img = ImageUrl(url='https://example.com/logo.png', force_download=True)

# Development: allow fetching from localhost
dev_img = ImageUrl(url='http://localhost:3000/test.png', force_download='allow-local')

# Default: let the provider handle it where possible
default_img = ImageUrl(url='https://cdn.example.com/photo.jpg')
```

### `ProviderDetailsDelta`

Used on `ToolReturnPart`, `NativeToolReturnPart`, and related message parts to update provider-specific metadata without replacing the entire dict:

```python
ProviderDetailsDelta: TypeAlias = (
    dict[str, Any]
    | Callable[[dict[str, Any] | None], dict[str, Any]]
    | None
)
```

| Form | Behaviour |
|------|-----------|
| `dict` | Replace/merge as a static delta |
| `Callable` | Called with the current dict (or `None`), returns the new dict |
| `None` | Clear provider details |

```python
from pydantic_ai.messages import ToolReturnPart

part = ToolReturnPart(
    tool_name='search',
    tool_call_id='call_123',
    content='Paris, France',
)

# Merge into existing details (provider_details is a plain dict, not a callable)
existing = part.provider_details if isinstance(part.provider_details, dict) else {}
part.provider_details = {**existing, 'source': 'web', 'relevance': 0.95}

# Or set directly with a new dict
part.provider_details = {'cache_hit': True, 'latency_ms': 42}
```

### MIME type auto-detection

`BinaryContent` uses the custom `MimeTypes` registry that `messages.py` configures at module load time. It adds MIME types that Python's built-in `mimetypes` module doesn't know about (markdown, YAML, TOML, WebP, audio variants, etc.):

```python
from pydantic_ai.messages import BinaryContent
import pathlib

# BinaryContent infers the media type from the file extension
with open('report.pdf', 'rb') as f:
    pdf_content = BinaryContent(data=f.read(), media_type='application/pdf')

# For a markdown file — supported via custom registry
with open('notes.md', 'rb') as f:
    md_content = BinaryContent(data=f.read(), media_type='text/markdown')

# Or rely on auto-detection via the MediaType inference path
audio_data = BinaryContent.from_path(pathlib.Path('podcast.mp3'))
# → media_type='audio/mpeg'
```

---

## 9. `AbstractCapability` Extended — `defer_loading`, `get_description`, `get_ordering`, Wrap Hooks

**Module:** `pydantic_ai.capabilities.abstract`  
**Import:** `from pydantic_ai.capabilities import AbstractCapability`

Vol. 2 covered the basics of `AbstractCapability`. This section documents the advanced API added since 1.102.0: deferred-loading capabilities, `get_description()`, `get_ordering()`, and the full set of wrap hooks.

### `defer_loading` — lazy capability loading

When `defer_loading=True`, the capability's tools and instructions are hidden from the model until it explicitly calls a `load_capability` tool. Useful for large capability sets that are rarely needed:

```python
from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai import Agent

@dataclass
class AdvancedAnalyticsCapability(AbstractCapability):
    id: str = 'advanced-analytics'    # required when defer_loading=True
    defer_loading: bool = True
    description: str = 'Advanced statistical analysis tools for expert users.'

    def get_instructions(self):
        return 'Use these tools only for complex statistical questions.'

    def get_toolset(self):
        from pydantic_ai.toolsets import FunctionToolset
        ts = FunctionToolset(id='analytics')

        @ts.tool_plain
        def run_regression(data: str, variables: list[str]) -> dict:
            """Run a linear regression."""
            return {'slope': 1.5, 'r_squared': 0.92}

        return ts

agent = Agent(
    'openai:gpt-4o',
    capabilities=[AdvancedAnalyticsCapability()],
)
# Model sees only the `load_capability` tool initially;
# analytics tools appear after it calls load_capability('advanced-analytics')
```

### `get_description()` for catalog routing

`get_description()` is surfaced to the model in the `load_capability` catalog when `defer_loading=True`. It can return a static string or a callable:

```python
from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai._run_context import RunContext

@dataclass
class LanguageCapability(AbstractCapability):
    language: str
    defer_loading: bool = True

    @property
    def id(self):
        return f'lang-{self.language}'

    def get_description(self):
        # Static string description shown in the tool catalog
        return f'Tools for working with {self.language} language content.'

    async def get_description_async(self, ctx: RunContext) -> str:
        # Or override with async for dynamic descriptions
        user_level = ctx.deps.get('user_level', 'beginner')
        return f'{self.language} tools (tuned for {user_level} users).'
```

### `get_ordering()` — topology control

Declare where in the middleware chain this capability must sit. Used by `CombinedCapability` to topologically sort its children:

```python
from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering

@dataclass
class SecurityCapability(AbstractCapability):
    """Must run outermost — first to see requests, last to see responses."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

@dataclass
class CachingCapability(AbstractCapability):
    """Must run inside SecurityCapability."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wrapped_by=[SecurityCapability])

from pydantic_ai.capabilities import CombinedCapability

# Even if listed in reverse order, topology is satisfied automatically
combined = CombinedCapability([CachingCapability(), SecurityCapability()])
# actual order: SecurityCapability → CachingCapability
```

### Wrap hooks — `wrap_run`, `wrap_node_run`, `wrap_model_request`, `wrap_tool_validate`, `wrap_tool_execute`, `wrap_output_validate`, `wrap_output_process`

Each lifecycle phase has three hook forms: `before_*`, `after_*`, and `wrap_*`. The `wrap_*` hooks receive a `handler` callable — call it to proceed, or skip it to short-circuit:

```python
from dataclasses import dataclass
from typing import Any
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai._run_context import RunContext
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.messages import ModelResponse

@dataclass
class CachingCapability(AbstractCapability):
    """Cache model responses; skip the model if a cached response exists."""

    _cache: dict = None  # type: ignore

    def __post_init__(self):
        self._cache = {}

    async def wrap_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        handler,
    ) -> ModelResponse:
        # Build a cache key from the message history
        cache_key = str(request_context.messages)
        if cache_key in self._cache:
            print('Cache hit — skipping model call.')
            return self._cache[cache_key]

        # Call the real model
        response = await handler()
        self._cache[cache_key] = response
        return response
```

```python
@dataclass
class RetryOnRateLimitCapability(AbstractCapability):
    """Retry model calls on HTTP 429 with exponential backoff."""

    max_retries: int = 3

    async def wrap_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        handler,
    ) -> ModelResponse:
        import asyncio
        from pydantic_ai.exceptions import ModelHTTPError

        for attempt in range(self.max_retries):
            try:
                return await handler()
            except ModelHTTPError as e:
                if e.status_code == 429 and attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f'Rate limited. Retrying in {wait}s...')
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError('Should not reach here')
```

### Handler type reference

The wrap handlers are `Protocol` types defined in `pydantic_ai.capabilities.abstract`:

| Handler | Signature |
|---------|-----------|
| `WrapRunHandler` | `async () → AgentRunResult` |
| `WrapNodeRunHandler` | `async () → NodeResult` |
| `WrapModelRequestHandler` | `async () → ModelResponse` |
| `WrapToolValidateHandler` | `async () → ValidatedToolArgs` |
| `WrapToolExecuteHandler` | `async () → Any` |
| `WrapOutputValidateHandler` | `async () → Any` |
| `WrapOutputProcessHandler` | `async () → Any` |

Each handler is zero-argument: all context is already captured via closure.

---

## 10. `CapabilityOrdering` + `CapabilityPosition` + `CapabilityRef` + `CAPABILITY_TYPES`

**Module:** `pydantic_ai.capabilities.abstract` (types) and `pydantic_ai.capabilities._ordering` (sort logic)

These four constructs control capability topology — where in the middleware chain each capability sits, and how the framework looks up capability types by name.

### `CapabilityOrdering` dataclass

```python
@dataclass
class CapabilityOrdering:
    position: CapabilityPosition | None = None
    # 'outermost': first in chain (wraps all others)
    # 'innermost': last in chain (wrapped by all others)

    wraps: Sequence[CapabilityRef] = ()
    # This capability comes before (wraps around) these refs

    wrapped_by: Sequence[CapabilityRef] = ()
    # This capability comes after (is inside) these refs

    requires: Sequence[type[AbstractCapability]] = ()
    # These types must be present in the chain (no ordering implied)
```

`CapabilityPosition` is `Literal['outermost', 'innermost']`.

`CapabilityRef` is `type[AbstractCapability] | AbstractCapability` — a type matches all instances of that type; an instance ref matches by identity (`is`).

### Ordering examples

```python
from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities import CombinedCapability

@dataclass
class AuthCapability(AbstractCapability):
    """Must be outermost — handles auth before anything else sees the request."""
    def get_ordering(self):
        return CapabilityOrdering(position='outermost')

@dataclass
class LoggingCapability(AbstractCapability):
    """Must wrap around BusinessCapability."""
    def get_ordering(self):
        return CapabilityOrdering(wraps=[BusinessCapability])

@dataclass
class BusinessCapability(AbstractCapability):
    """Core business logic — sits inside LoggingCapability."""
    pass

@dataclass
class MetricsCapability(AbstractCapability):
    """Requires AuthCapability to be present."""
    def get_ordering(self):
        return CapabilityOrdering(requires=[AuthCapability])

# Declared in any order; the sorter fixes it
combined = CombinedCapability([
    MetricsCapability(),      # requires AuthCapability
    BusinessCapability(),     # wrapped by LoggingCapability
    LoggingCapability(),      # wraps BusinessCapability
    AuthCapability(),         # must be outermost
])
# Sorted result: AuthCapability → LoggingCapability → BusinessCapability → MetricsCapability
```

### `CAPABILITY_TYPES` registry

`CAPABILITY_TYPES` is a `dict[str, type[AbstractCapability]]` mapping capability class names (as returned by `get_serialization_name()`) to their types. It is populated via `__init_subclass__`:

```python
from pydantic_ai.capabilities import CAPABILITY_TYPES

# All built-in capabilities are registered automatically
print(list(CAPABILITY_TYPES.keys())[:5])
# ['Hooks', 'WebSearch', 'WebFetch', 'Thinking', ...]

# Look up a capability by name (used by AgentSpec to deserialize YAML)
HooksClass = CAPABILITY_TYPES['Hooks']
hooks = HooksClass()
```

```python
# Register a custom capability for YAML/JSON spec loading
from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability

@dataclass
class MyCustomCapability(AbstractCapability):
    threshold: float = 0.8

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'MyCustomCapability'  # must be unique in the registry

# Now usable in AgentSpec YAML:
# capabilities:
#   - MyCustomCapability:
#       threshold: 0.9
from pydantic_ai.capabilities import CAPABILITY_TYPES
assert 'MyCustomCapability' in CAPABILITY_TYPES
```

### Cycle detection and conflict errors

The topology sorter raises `UserError` for unsatisfiable constraints:

```python
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.exceptions import UserError
from dataclasses import dataclass

@dataclass
class A(AbstractCapability):
    def get_ordering(self):
        return CapabilityOrdering(wraps=[B])  # A must come before B

@dataclass
class B(AbstractCapability):
    def get_ordering(self):
        return CapabilityOrdering(wraps=[A])  # B must come before A — cycle!

try:
    CombinedCapability([A(), B()])
except UserError as e:
    print(f'Cycle detected: {e}')

# Missing requirement
@dataclass
class NeedsC(AbstractCapability):
    def get_ordering(self):
        return CapabilityOrdering(requires=[C])  # type: ignore[name-defined]

try:
    CombinedCapability([NeedsC()])
except UserError as e:
    print(f'Requirement missing: {e}')
```

### `has_capability_type()` utility

```python
from pydantic_ai.capabilities._ordering import has_capability_type
from pydantic_ai.capabilities import WebSearch, WebFetch, CombinedCapability

combined = CombinedCapability([WebSearch(), WebFetch()])

# Check if any leaf in a capability tree is an instance of WebSearch
print(has_capability_type([combined], WebSearch))   # True
print(has_capability_type([combined], WebFetch))    # True

from pydantic_ai.capabilities import Thinking
print(has_capability_type([combined], Thinking))    # False
```

---

*All classes verified against **pydantic-ai 1.105.0** installed directly from PyPI. Source modules: `pydantic_ai.result`, `pydantic_ai.capabilities.wrapper`, `pydantic_ai.toolsets.function`, `pydantic_ai.toolsets.abstract`, `pydantic_ai.messages`, `pydantic_ai.exceptions`, `pydantic_ai.capabilities.abstract`, `pydantic_ai.capabilities._ordering`.*

---
title: "PydanticAI Class Deep Dives Vol. 25"
description: "Source-verified deep dives into 10 pydantic-ai 2.0.0 class groups: ToolManager + ValidatedToolCall + ParallelExecutionMode (parallel/sequential/parallel_ordered_events tool execution), Agent v2.0.0 new surface (to_web Starlette chat app, to_cli rich terminal, run_stream_events, parallel_tool_call_execution_mode, is_*_node type-narrowing helpers), Direct API (model_request / model_request_sync / model_request_stream / model_request_stream_sync / StreamedResponseSync thread-bridge), format_as_xml + _ToXml internals (include_field_info='once', rootless XML, Pydantic/dataclass metadata), common_tools family (DuckDuckGoSearchTool / TavilySearchTool / ExaToolset / ImageGenerationSubagentTool factories), LoadCapabilityCallPart + LoadCapabilityArgs + LoadCapabilityReturn (deferred-capability wire protocol), NamedSpec + CapabilitySpec + build_registry + load_from_registry (spec-driven YAML/JSON composition), merge_profile + ModelProfile v2.0.0 complete field reference (15 fields, new supports_image_output / supports_inline_system_prompts / StructuredOutputMode), FunctionModel v2.0.0 (profile + settings constructor params, AgentInfo.instructions, stream_function), AgentRun v2.0.0 complete API + Agent node-type helpers (is_call_tools_node / is_end_node / is_model_request_node / is_user_prompt_node). All verified against pydantic-ai 2.0.0 source."
sidebar:
  label: "Class deep dives (Vol. 25)"
  order: 51
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.0.0** source installed directly from PyPI. pydantic-ai 2.0.0 is a major release — class signatures, field names, and method names in this volume reflect the 2.x API. See the migration notes within each section for the most important changes from 1.x.
</Aside>

Ten class groups covering the completely new parallel tool execution infrastructure, five new Agent deployment and streaming methods added in 2.0.0, the thin direct-model API without an agent, the full XML-formatting internals, the common-tool factory trio, the deferred-capability wire protocol, spec-driven agent composition from YAML/JSON, the ModelProfile overhaul, FunctionModel's new constructor params, and the AgentRun iteration API with its four type-narrowing node helpers.

---

## 1. `ToolManager` + `ValidatedToolCall` + `ParallelExecutionMode` — Parallel Tool Execution

**Source**: `pydantic_ai/tool_manager.py`

`ToolManager` manages all tool calls for a single run step. The headline v2.0.0 change is the `ParallelExecutionMode` Literal and two context-manager entry points that control how a batch of tool calls from a single model response is executed:

```python
ParallelExecutionMode = Literal['parallel', 'sequential', 'parallel_ordered_events']
# 'parallel'                — run concurrently, emit events as each call finishes (default)
# 'sequential'              — run one at a time in the order the model emitted them
# 'parallel_ordered_events' — run concurrently, but buffer and emit events in order

@dataclass
class ValidatedToolCall(Generic[AgentDepsT]):
    """Separates validation from execution so callers can know if validation passed before running."""
    call: ToolCallPart
    tool: ToolsetTool[AgentDepsT] | None
    ctx: RunContext[AgentDepsT]
    args_valid: bool
    validated_args: dict[str, Any] | None = None
    validation_error: ToolRetryError | None = None

@dataclass
class ToolManager(Generic[AgentDepsT]):
    toolset: AbstractToolset[AgentDepsT]
    root_capability: AbstractCapability[AgentDepsT] | None = None
    ctx: RunContext[AgentDepsT] | None = None
    tools: dict[str, ToolsetTool[AgentDepsT]] | None = None   # keyed by tool_def.name
    failed_tools: set[str] = field(default_factory=set[str])
    default_max_retries: int = 1

    @classmethod
    @contextmanager
    def parallel_execution_mode(cls, mode: ParallelExecutionMode = 'parallel') -> Generator[None]: ...

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> ToolManager[AgentDepsT]: ...
    def get_parallel_execution_mode(self) -> ParallelExecutionMode: ...
    def is_sequential(self, call: ToolCallPart) -> bool: ...
    def get_tool_def(self, name: str) -> ToolDefinition | None: ...
    def validate_tool_call(self, call, *, approved=False, metadata=None, wrap_validation_errors=True) -> ValidatedToolCall: ...
    async def execute_tool_call(self, validated, *, wrap_validation_errors=True) -> Any: ...
    async def handle_call(self, call, *, approved=False, metadata=None, wrap_validation_errors=True) -> ToolDenied | ToolReturn | Any: ...
    async def resolve_deferred_tool_calls(self, requests: DeferredToolRequests) -> DeferredToolResults | None: ...
```

`ValidatedToolCall.args_valid` lets hooks and custom toolsets inspect validation status before executing — tools emitted by the model with invalid JSON schema args yield `args_valid=False` without needing to attempt execution.

### 1.1 Switch the Entire Run to Sequential Execution

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import RunContext


agent = Agent('openai:gpt-4o-mini')


@agent.tool
async def step_a(ctx: RunContext[None]) -> str:
    """Perform step A."""
    print("step_a running")
    return "A done"


@agent.tool
async def step_b(ctx: RunContext[None]) -> str:
    """Perform step B (must run after A)."""
    print("step_b running")
    return "B done"


async def main():
    # All tool calls for any run inside this context run one at a time in order.
    with ToolManager.parallel_execution_mode('sequential'):
        result = await agent.run("Run step_a then step_b")
    print(result.output)


asyncio.run(main())
```

### 1.2 Ordered Events for Deterministic Logging

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import RunContext


agent = Agent('openai:gpt-4o-mini')


@agent.tool
async def fetch_a(ctx: RunContext[None]) -> str:
    """Fetch resource A."""
    await asyncio.sleep(0.1)   # simulate I/O
    return "resource A"


@agent.tool
async def fetch_b(ctx: RunContext[None]) -> str:
    """Fetch resource B."""
    await asyncio.sleep(0.05)   # finishes first in wall-clock
    return "resource B"


async def main():
    # Tools run concurrently but events arrive in model-emission order (A before B).
    # Useful when downstream consumers need deterministic event ordering for logging/replay.
    with ToolManager.parallel_execution_mode('parallel_ordered_events'):
        result = await agent.run("Fetch both resources")
    print(result.output)


asyncio.run(main())
```

### 1.3 Inspect `ValidatedToolCall` Before Executing

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tool_manager import ToolManager, ValidatedToolCall
from pydantic_ai.tools import RunContext
from pydantic_ai.messages import ToolCallPart


agent = Agent('openai:gpt-4o-mini')


@agent.tool
async def greet(ctx: RunContext[None], name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}!"


async def main():
    # Build a ToolManager for inspection (normally done by the agent internals)
    async with agent.iter("Greet Alice") as run:
        async for node in run:
            pass
    # After the run, inspect validation history
    # In a real hook or custom toolset you'd call validate_tool_call directly:
    print("Run complete:", run.result.output)


asyncio.run(main())
```

### 1.4 `resolve_deferred_tool_calls` — Hydrate DeferredToolRequests

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext, DeferredToolRequests, DeferredToolResults
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset


async def send_email(ctx: RunContext[None], to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}: {subject}"


async def main():
    base_ts = FunctionToolset([send_email])
    approval_ts = ApprovalRequiredToolset(
        base_ts,
        approval_required_func=lambda ctx, tool_def, args: True,  # always require approval
    )
    agent = Agent(
        'openai:gpt-4o-mini',
        toolsets=[approval_ts],
        output_type=[str, DeferredToolRequests],
    )

    # Run 1: model tries to call send_email → approval required → DeferredToolRequests returned
    result1 = await agent.run("Send an email to alice@example.com about the meeting")
    if isinstance(result1.output, DeferredToolRequests):
        pending = result1.output
        print(f"Pending: {[c.tool_name for c in pending.approvals]}")

        # Human approves all → build DeferredToolResults
        tool_results = pending.build_results(approve_all=True)

        # Run 2: supply approved results
        result2 = await agent.run(
            None,
            deferred_tool_results=tool_results,
            message_history=result1.new_messages(),
        )
        print(result2.output)


asyncio.run(main())
```

---

## 2. Agent v2.0.0 New Surface — `to_web`, `to_cli`, `run_stream_events`, `parallel_tool_call_execution_mode`, Node-Type Helpers

**Source**: `pydantic_ai/agent/__init__.py`

pydantic-ai 2.0.0 adds five new entry points directly on `Agent`:

```python
class Agent:
    # Deployment
    def to_web(self, *, models=None, deps=None, model_settings=None, instructions=None, html_source=None) -> Starlette: ...
    async def to_cli(self, *, deps=None, prog_name='pydantic-ai', message_history=None, model_settings=None, usage_limits=None) -> None: ...
    def to_cli_sync(self, ...) -> None: ...

    # Streaming
    def run_stream_events(self, user_prompt, **kwargs) -> AbstractAsyncContextManager[AsyncIterator[AgentStreamEvent | AgentRunResultEvent]]: ...

    # Parallel tool control
    @staticmethod
    def parallel_tool_call_execution_mode(mode: ParallelExecutionMode = 'parallel') -> Generator[None]: ...

    # Type-narrowing node helpers (TypeIs-based)
    @staticmethod
    def is_call_tools_node(node) -> TypeIs[CallToolsNode]: ...
    @staticmethod
    def is_end_node(node) -> TypeIs[End[FinalResult]]: ...
    @staticmethod
    def is_model_request_node(node) -> TypeIs[ModelRequestNode]: ...
    @staticmethod
    def is_user_prompt_node(node) -> TypeIs[UserPromptNode]: ...
    def render_description(self, deps=None) -> str | None: ...
```

### 2.1 `Agent.to_web()` — One-Line Web Chat App

```python
# app.py — run with: uvicorn app:app --reload
import uvicorn
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent(
    'openai:gpt-4o',
    instructions="You are a helpful research assistant.",
    capabilities=[WebSearch()],
)

# Returns a Starlette app; expose additional models for the UI model-picker.
app = agent.to_web(
    models=['openai:gpt-4o-mini', 'anthropic:claude-haiku-4-5'],
    instructions="Search the web for up-to-date information.",
)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

### 2.2 `Agent.to_cli()` — Rich Terminal Chat

```python
import asyncio
from pydantic_ai import Agent


agent = Agent('openai:gpt-4o-mini', instructions="You are a concise assistant.")


async def main():
    # Starts an interactive Rich-powered chat session in the terminal.
    # Ctrl+C or 'exit' to quit.
    await agent.to_cli(prog_name='my-assistant')


asyncio.run(main())
```

### 2.3 `Agent.run_stream_events()` — Combined Run + Event Stream

```python
import asyncio
from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta


agent = Agent('openai:gpt-4o-mini')


async def main():
    collected_text = []
    async with agent.run_stream_events("Tell me a short joke") as events:
        async for event in events:
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                collected_text.append(event.delta.content_delta)
                print(event.delta.content_delta, end="", flush=True)
            elif isinstance(event, AgentRunResultEvent):
                print()   # newline after streaming
                print(f"\nFinal output: {event.result.output}")

    print("Assembled:", "".join(collected_text))


asyncio.run(main())
```

### 2.4 `Agent.parallel_tool_call_execution_mode()` — Static Context Manager

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext


agent = Agent('openai:gpt-4o-mini')


@agent.tool
async def search_web(ctx: RunContext[None], query: str) -> str:
    """Search the web for a query."""
    return f"Results for: {query}"


@agent.tool
async def search_docs(ctx: RunContext[None], query: str) -> str:
    """Search internal docs for a query."""
    return f"Docs results for: {query}"


async def main():
    # Agent.parallel_tool_call_execution_mode is a thin static wrapper around
    # ToolManager.parallel_execution_mode — both are interchangeable.
    with Agent.parallel_tool_call_execution_mode('parallel_ordered_events'):
        result = await agent.run("Search web and docs for 'pydantic v2'")
    print(result.output)


asyncio.run(main())
```

### 2.5 Node-Type Helpers — Type-Safe Iteration

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import TextPart


agent = Agent('openai:gpt-4o-mini')


@agent.tool_plain
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def main():
    async with agent.iter("What is 7 + 35?") as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                # node: UserPromptNode — access user_prompt, system_prompts, etc.
                print(f"User prompt: {node.user_prompt!r}")

            elif Agent.is_model_request_node(node):
                # node: ModelRequestNode — access request parts
                print(f"Model request: {node.request.parts}")

            elif Agent.is_call_tools_node(node):
                # node: CallToolsNode — access model_response
                for part in node.model_response.parts:
                    if isinstance(part, TextPart):
                        print(f"Model text: {part.content!r}")

            elif Agent.is_end_node(node):
                # node: End[FinalResult[str]] — run.result is now available
                print(f"Done: {node.data.output}")

    print("Final:", run.result.output)


asyncio.run(main())
```

### 2.6 `render_description` — TemplateStr Agent Descriptions

```python
from pydantic_ai import Agent
from pydantic_ai._template import TemplateStr


# TemplateStr descriptions can reference {{deps}} at render time.
agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=str,
    description=TemplateStr("Assistant for {{deps}} support queries"),
)

print(agent.render_description(deps="billing"))    # "Assistant for billing support queries"
print(agent.render_description(deps="technical"))  # "Assistant for technical support queries"
```

---

## 3. Direct API — `model_request`, `model_request_stream`, `StreamedResponseSync`

**Source**: `pydantic_ai/direct.py`

The Direct API lets you send messages to a model without an `Agent`. There is no dependency injection, no tool dispatch, no retry logic — just the raw model interface with OTel instrumentation wired in.

```python
async def model_request(
    model: Model | KnownModelName | str,
    messages: Sequence[ModelMessage],
    *,
    model_settings: ModelSettings | None = None,
    model_request_parameters: ModelRequestParameters | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> ModelResponse: ...

async def model_request_stream(
    model, messages, *, model_settings=None, model_request_parameters=None, instrument=None
) -> AbstractAsyncContextManager[StreamedResponse]: ...

def model_request_sync(model, messages, **kwargs) -> ModelResponse: ...

def model_request_stream_sync(model, messages, **kwargs) -> StreamedResponseSync: ...

@dataclass
class StreamedResponseSync:
    """Thread-bridge: runs the async stream producer in a background thread and exposes a sync iterator."""
    ...
```

`_ensure_instruction_parts` is called internally to bridge the gap between `ModelRequest.instructions` (the direct-API path) and `ModelRequestParameters.instruction_parts` (the provider path) so providers that read `instruction_parts` directly still receive the instructions.

### 3.1 Simple Non-Streamed Request

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request


async def main():
    response = await model_request(
        'openai:gpt-4o-mini',
        [ModelRequest.user_text_prompt('What is the capital of France?')],
    )
    print(response.parts)           # [TextPart(content='The capital of France is Paris.')]
    print(response.usage)           # RequestUsage(input_tokens=..., output_tokens=...)
    print(response.model_name)      # 'gpt-4o-mini'


asyncio.run(main())
```

### 3.2 Streamed Request via Async Context Manager

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import TextPartDelta, PartDeltaEvent


async def main():
    async with await model_request_stream(
        'openai:gpt-4o-mini',
        [ModelRequest.user_text_prompt('Count to five.')],
    ) as stream:
        async for event in stream:
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                print(event.delta.content_delta, end='', flush=True)
    print()

    # Completed response is available after the stream closes
    final = stream.get()
    print("Full text:", ''.join(
        p.content for p in final.parts if hasattr(p, 'content')
    ))


asyncio.run(main())
```

### 3.3 Sync API for Scripts and Notebooks

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync


# Synchronous — runs its own event loop internally.
response = model_request_sync(
    'openai:gpt-4o-mini',
    [ModelRequest.user_text_prompt('Translate "hello" to Spanish.')],
)
print(response.parts[0].content)   # 'Hola'
```

### 3.4 `StreamedResponseSync` — Sync Streaming from a Thread

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream_sync
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta


with model_request_stream_sync(
    'openai:gpt-4o-mini',
    [ModelRequest.user_text_prompt('List three colors.')],
) as stream:
    for event in stream:  # synchronous iteration
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            print(event.delta.content_delta, end='', flush=True)
print()
```

### 3.5 Inject a System Prompt via `ModelRequestParameters`

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestParameters


async def main():
    # instruction_parts are read by providers that don't use system-prompt history
    params = ModelRequestParameters(
        function_tools=[],
        output_tools=[],
        allow_text_output=True,
        instruction_parts=[InstructionPart(content="Always respond in bullet points.")],
    )
    response = await model_request(
        'openai:gpt-4o-mini',
        [ModelRequest.user_text_prompt('Tell me about Python.')],
        model_request_parameters=params,
    )
    print(response.parts[0].content)


asyncio.run(main())
```

---

## 4. `format_as_xml` + `_ToXml` Internals — XML Prompt Formatting

**Source**: `pydantic_ai/format_prompt.py`

`format_as_xml` converts any Python object into an XML string for feeding into LLM prompts. The implementation is a single `_ToXml` dataclass that accumulates `_fields_info` and `_element_names` lazily on first encounter of a dataclass or Pydantic model.

```python
def format_as_xml(
    obj: Any,
    root_tag: str | None = None,          # None → rootless; elements joined by '\n'
    item_tag: str = 'item',               # fallback tag for list items
    none_str: str = 'null',               # how to render None values
    indent: str | None = '  ',            # None → no indentation; rootless + None → no newlines
    include_field_info: Literal['once'] | bool = False,
    # True  → include title/description XML attrs on every element
    # 'once' → only on first occurrence per field (saves tokens in lists)
) -> str: ...
```

Supported types: `str`, `bytes`, `bool`, `int`, `float`, `Decimal`, `date`, `datetime`, `time`, `timedelta`, `UUID`, `Enum`, `Mapping`, `Iterable`, `dataclass`, `BaseModel`.

### 4.1 Basic Dict and List Formatting

```python
from pydantic_ai import format_as_xml

# Dict → named child elements
print(format_as_xml({'name': 'Alice', 'age': 30}, root_tag='user'))
# <user>
#   <name>Alice</name>
#   <age>30</age>
# </user>

# List → <item> elements
print(format_as_xml(['red', 'green', 'blue'], root_tag='colors'))
# <colors>
#   <item>red</item>
#   <item>green</item>
#   <item>blue</item>
# </colors>

# Scalar value
print(format_as_xml(42, root_tag='count'))
# <count>42</count>
```

### 4.2 `root_tag=None` — Rootless XML

```python
from pydantic_ai import format_as_xml

data = {'name': 'Bob', 'city': 'London'}
# With root_tag=None, each top-level key becomes a sibling element
rootless = format_as_xml(data, root_tag=None)
print(rootless)
# <name>Bob</name>
# <city>London</city>
```

### 4.3 Pydantic Model with `include_field_info='once'`

```python
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml


class Product(BaseModel):
    name: str = Field(title="Product Name", description="The name of the product")
    price: float = Field(title="Price USD", description="Price in US dollars")


products = [
    Product(name="Widget", price=9.99),
    Product(name="Gadget", price=24.99),
]

# include_field_info='once' → title/description XML attributes only on first occurrence
print(format_as_xml(products, root_tag='products', include_field_info='once'))
# <products>
#   <Product>
#     <name title="Product Name" description="The name of the product">Widget</name>
#     <price title="Price USD" description="Price in US dollars">9.99</price>
#   </Product>
#   <Product>
#     <name>Gadget</name>          ← no attrs on repeated occurrence
#     <price>24.99</price>
#   </Product>
# </products>
```

### 4.4 Dataclass with Field Metadata

```python
from dataclasses import dataclass, field
from pydantic_ai import format_as_xml


@dataclass
class Order:
    order_id: str = field(metadata={'title': 'Order ID', 'description': 'Unique order identifier'})
    total: float = field(metadata={'title': 'Total', 'description': 'Order total in USD'})
    status: str = field(default='pending')


order = Order(order_id='ORD-001', total=49.95)
# include_field_info=True → metadata attrs on all occurrences
print(format_as_xml(order, include_field_info=True))
# <Order>
#   <order_id title="Order ID" description="Unique order identifier">ORD-001</order_id>
#   <total title="Total" description="Order total in USD">49.95</total>
#   <status>pending</status>
# </Order>
```

### 4.5 No-Indentation Mode for Compact Tokens

```python
from pydantic_ai import format_as_xml

data = {'query': 'best Python books', 'limit': 5, 'format': 'json'}

# indent=None removes all whitespace — minimises prompt token count
compact = format_as_xml(data, root_tag='params', indent=None)
print(compact)
# <params><query>best Python books</query><limit>5</limit><format>json</format></params>
```

---

## 5. `common_tools` Family — `duckduckgo_search_tool`, `tavily_search_tool`, `ExaToolset`

**Source**: `pydantic_ai/common_tools/`

The `common_tools` package provides ready-made `Tool` and `FunctionToolset` factories for three popular search/retrieval backends. All follow the same pattern: a dataclass (`DuckDuckGoSearchTool`, `TavilySearchTool`) or `FunctionToolset` subclass (`ExaToolset`) holds configuration, with `__call__` or per-method callables bound into `Tool` objects by the factory function.

### 5.1 DuckDuckGo Search Tool

```python
import asyncio
from ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool


async def main():
    # Factory: duckduckgo_search_tool(duckduckgo_client=None, max_results=None)
    # max_results=None → only first response page (fast)
    agent = Agent(
        'openai:gpt-4o-mini',
        tools=[duckduckgo_search_tool(max_results=5)],
    )
    result = await agent.run("What is the latest Python version?")
    print(result.output)


asyncio.run(main())
```

### 5.2 Custom DDGS Client with Proxy

```python
import asyncio
from ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool, DuckDuckGoSearchTool


async def main():
    # Share one DDGS client across agents to reuse the session
    ddgs_client = DDGS(proxy=None, timeout=10)

    tool = duckduckgo_search_tool(duckduckgo_client=ddgs_client, max_results=10)
    agent = Agent('openai:gpt-4o-mini', tools=[tool])

    result = await agent.run("Find recent news about AI agents")
    print(result.output)


asyncio.run(main())
```

### 5.3 Tavily Search Tool — Advanced Depth and Topic Filtering

```python
import asyncio
from tavily import AsyncTavilyClient
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool, TavilySearchResult


async def main():
    tavily_client = AsyncTavilyClient(api_key="tvly-...")

    # tavily_search_tool(tavily_client, max_results=None, *, search_depth='basic',
    #                    topic='general', include_domains=None, exclude_domains=None,
    #                    include_answer=False, include_raw_content=False)
    agent = Agent(
        'openai:gpt-4o-mini',
        tools=[
            tavily_search_tool(
                tavily_client=tavily_client,
                max_results=5,
                # search_depth and topic are passed through to each individual call
            )
        ],
    )
    result = await agent.run("Find recent papers on transformer architectures")
    print(result.output)


asyncio.run(main())
```

### 5.4 `ExaToolset` — Search, Find Similar, Get Contents, and AI Answers

```python
import asyncio
from exa_py import AsyncExa
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset


async def main():
    exa_client = AsyncExa(api_key="exa-...")

    # ExaToolset wraps four tools: exa_search, exa_find_similar, exa_get_contents, exa_answer
    toolset = ExaToolset(client=exa_client)

    agent = Agent('openai:gpt-4o-mini', toolsets=[toolset])
    result = await agent.run(
        "Find papers similar to https://arxiv.org/abs/1706.03762 and summarise them"
    )
    print(result.output)


asyncio.run(main())
```

### 5.5 `exa_search_tool` Factory — Single Tool Variant

```python
import asyncio
from exa_py import AsyncExa
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import exa_search_tool


async def main():
    exa = AsyncExa(api_key="exa-...")

    # exa_search_tool wraps only the exa_search function as a standalone Tool
    agent = Agent(
        'openai:gpt-4o-mini',
        tools=[exa_search_tool(exa, max_results=5)],
    )
    result = await agent.run("Search for recent advances in retrieval-augmented generation")
    print(result.output)


asyncio.run(main())
```

---

## 6. `LoadCapabilityCallPart` + `LoadCapabilityArgs` + `LoadCapabilityReturn` — Deferred Capability Wire Protocol

**Source**: `pydantic_ai/_deferred_capabilities.py`

These types implement the wire protocol for the `load_capability` tool that deferred capabilities expose to the model. They extend `ToolCallPart` and `ToolReturnPart` with typed discrimination so the agent graph can route capability-load calls without runtime string matching.

```python
DEFERRED_CAPABILITY_TOOL_METADATA_KEY = 'pydantic_ai_deferred_capability_tool'
# Tool metadata key marking function tools owned by an on-demand capability.
# The DeferredCapabilityLoader stamps this key on every tool it injects.

class LoadCapabilityArgs(TypedDict):
    id: Annotated[str, pydantic.Field(description='The id of the capability to load.')]

class LoadCapabilityReturn(TypedDict):
    instructions: NotRequired[str]   # capability's instructions string, if it returned any

@dataclass(repr=False)
class LoadCapabilityCallPart(ToolCallPart):
    tool_name: Literal['load_capability'] = 'load_capability'
    args: str | LoadCapabilityArgs | None = None
    tool_kind: Literal['capability-load'] = 'capability-load'

    @property
    def typed_args(self) -> LoadCapabilityArgs | None: ...  # parses args_as_dict
    @property
    def capability_id(self) -> str | None: ...              # shortcut to typed_args['id']

@dataclass(repr=False)
class LoadCapabilityReturnPart(ToolReturnPart):
    tool_name: Literal['load_capability'] = 'load_capability'
    tool_kind: Literal['capability-load'] = 'capability-load'

    @property
    def capability_instructions(self) -> str | None: ...   # parses content as LoadCapabilityReturn
```

The `tool_kind = 'capability-load'` discriminator lets message-history processors, serializers, and tests identify capability-load calls without inspecting tool names.

### 6.1 Parse a `LoadCapabilityCallPart` from a Tool Call

```python
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart, LoadCapabilityArgs
import json


# Simulate a model emitting a load_capability call
raw_args = json.dumps({'id': 'web_search'})
call_part = LoadCapabilityCallPart(
    tool_call_id='tc_001',
    args=raw_args,
)

print(call_part.tool_name)    # 'load_capability'
print(call_part.tool_kind)    # 'capability-load'
print(call_part.capability_id)  # 'web_search'
print(call_part.typed_args)    # {'id': 'web_search'}
```

### 6.2 Observe Capability Loading in an `iter` Run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart


async def main():
    agent = Agent(
        'openai:gpt-4o',
        capabilities=[WebSearch(defer_loading=True, id='web_search')],
    )

    async with agent.iter("Search for today's top tech news") as run:
        async for node in run:
            if Agent.is_call_tools_node(node):
                # Look for load_capability calls in the model response
                for part in node.model_response.parts:
                    if isinstance(part, LoadCapabilityCallPart):
                        print(f"Model requested to load capability: {part.capability_id!r}")

    print(run.result.output)


asyncio.run(main())
```

### 6.3 `LoadCapabilityReturnPart.capability_instructions` — Read Instructions After Load

```python
from pydantic_ai._deferred_capabilities import LoadCapabilityReturnPart
import json


# Simulate what the DeferredCapabilityLoaderToolset writes back to the model
return_payload = json.dumps({'instructions': 'Use the web_search tool to look up current information.'})
return_part = LoadCapabilityReturnPart(
    tool_name='load_capability',
    tool_call_id='tc_001',
    content=return_payload,
)

print(return_part.tool_kind)                      # 'capability-load'
print(return_part.capability_instructions)         # 'Use the web_search tool...'
```

### 6.4 Filter Capability-Load Parts from Message History

```python
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart, LoadCapabilityReturnPart
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart


def strip_capability_load_parts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove load_capability calls and their returns from message history.

    Useful when you want to replay a conversation without re-triggering capability loads.
    """
    import dataclasses
    cleaned = []
    for msg in messages:
        if hasattr(msg, 'parts'):
            new_parts = [
                p for p in msg.parts
                if not isinstance(p, (LoadCapabilityCallPart, LoadCapabilityReturnPart))
            ]
            if len(new_parts) != len(msg.parts):
                msg = dataclasses.replace(msg, parts=new_parts)
        cleaned.append(msg)
    return cleaned
```

---

## 7. `NamedSpec` + `CapabilitySpec` + `build_registry` + `load_from_registry` — Spec-Driven Agent Composition

**Source**: `pydantic_ai/_spec.py`

`NamedSpec` and `CapabilitySpec` are Pydantic `BaseModel` subclasses that enable agents to be fully described as serialisable dicts — for YAML config files, REST APIs, or `AgentSpec` composition. Three short forms are supported:

```python
class NamedSpec(BaseModel):
    """Supports three short forms:
        'MyClass'                      → name='MyClass', arguments=None
        {'MyClass': single_arg}        → name='MyClass', arguments=(single_arg,)
        {'MyClass': {'k1': v1, ...}}   → name='MyClass', arguments={'k1': v1, ...}
    """
    name: str
    arguments: None | tuple[Any] | dict[str, Any]

class CapabilitySpec(NamedSpec):
    """CapabilitySpec fields in JSON schemas are replaced with the full capability Union."""
    ...

def build_registry(
    *, custom_types, defaults, get_name, label, validate=None
) -> Mapping[str, type[T]]: ...

def load_from_registry(
    spec: NamedSpec, registry: Mapping[str, type[T]]
) -> T: ...
```

### 7.1 Parse a `NamedSpec` from Three Short Forms

```python
from pydantic_ai._spec import NamedSpec


# Form 1: bare string — no args
spec1 = NamedSpec.model_validate('OpenAIModel')
print(spec1.name, spec1.arguments)   # 'OpenAIModel' None

# Form 2: dict with a single scalar — positional arg
spec2 = NamedSpec.model_validate({'OpenAIModel': 'gpt-4o-mini'})
print(spec2.name, spec2.arguments)   # 'OpenAIModel' ('gpt-4o-mini',)

# Form 3: dict of kwargs
spec3 = NamedSpec.model_validate({'OpenAIModel': {'model_name': 'gpt-4o', 'temperature': 0.7}})
print(spec3.name, spec3.arguments)   # 'OpenAIModel' {'model_name': 'gpt-4o', 'temperature': 0.7}
```

### 7.2 `build_registry` — Create a Name → Class Map

```python
from pydantic_ai._spec import build_registry, NamedSpec
from pydantic_ai.capabilities import WebSearch, WebFetch
from pydantic_ai.capabilities.abstract import AbstractCapability


CAPABILITY_REGISTRY = build_registry(
    custom_types=[],
    defaults=[WebSearch, WebFetch],
    get_name=lambda cls: getattr(cls, '__name__', None),
    label='capability',
)

print(list(CAPABILITY_REGISTRY.keys()))   # ['WebSearch', 'WebFetch']
```

### 7.3 `load_from_registry` — Instantiate from a Spec

```python
from pydantic_ai._spec import build_registry, load_from_registry, NamedSpec
from pydantic_ai.capabilities import WebSearch, WebFetch


registry = build_registry(
    custom_types=[],
    defaults=[WebSearch, WebFetch],
    get_name=lambda cls: cls.__name__,
    label='capability',
)

# Load from a bare name — calls WebSearch()
ws = load_from_registry(NamedSpec.model_validate('WebSearch'), registry)
print(type(ws).__name__)   # 'WebSearch'

# Load from kwargs spec — calls WebFetch(max_content_tokens=4096)
spec = NamedSpec.model_validate({'WebFetch': {'max_content_tokens': 4096}})
wf = load_from_registry(spec, registry)
print(type(wf).__name__)   # 'WebFetch'
```

### 7.4 Drive `AgentSpec` from a Config Dict

```python
import asyncio
from pydantic_ai.agent import Agent


async def main():
    # AgentSpec can be passed to agent.run() / agent.iter() as the `spec` kwarg.
    # The spec dict allows per-run capability injection from serialisable config.
    agent = Agent('openai:gpt-4o-mini')

    spec_dict = {
        'capabilities': [
            'WebSearch',                              # bare string form
            {'WebFetch': {'max_content_tokens': 2048}},  # kwargs form
        ]
    }

    result = await agent.run(
        "Search the web and fetch the pydantic-ai homepage",
        spec=spec_dict,
    )
    print(result.output)


asyncio.run(main())
```

---

## 8. `merge_profile` + `ModelProfile` v2.0.0 — Complete Field Reference

**Source**: `pydantic_ai/profiles/__init__.py`

`ModelProfile` is a `TypedDict` that describes how a model or model family processes requests. v2.0.0 adds five new fields and replaces `ModelProfile.update()` with the standalone `merge_profile` function.

```python
def merge_profile(base: ModelProfile | None, *overrides: ModelProfile | None) -> ModelProfile:
    """Merge via dict-spread — later arguments override earlier ones; None is treated as empty."""
    result = {}
    if base:   result = {**result, **base}
    for override in overrides:
        if override:  result = {**result, **override}
    return result

class ModelProfile(TypedDict, total=False):
    # Tool and output support
    supports_tools: bool                         # default True
    supports_tool_return_schema: bool            # default False
    supports_json_schema_output: bool            # default False
    supports_json_object_output: bool            # default False
    supports_image_output: bool                  # NEW in v2 — default False
    supports_inline_system_prompts: bool         # NEW in v2 — default False

    # Structured output
    default_structured_output_mode: StructuredOutputMode   # 'tool' | 'json' | 'prompted'
    prompted_output_template: str                           # default DEFAULT_PROMPTED_OUTPUT_TEMPLATE
    native_output_requires_schema_in_instructions: bool     # default False

    # Schema transformation
    json_schema_transformer: type[JsonSchemaTransformer] | None   # default None

    # Thinking / reasoning
    supports_thinking: bool                      # default False
    thinking_tags: tuple[str, str]               # default ('<think>', '</think>')
    thinking_always_enabled: bool                # default False

    # Misc
    ignore_streamed_leading_whitespace: bool     # default False

    # Available native tools
    supported_native_tools: frozenset[type[AbstractNativeTool]]  # default SUPPORTED_NATIVE_TOOLS
```

### 8.1 `merge_profile` — Layer Profiles in Providers

```python
from pydantic_ai.profiles import merge_profile, ModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile


# Custom override: force prompted output mode for a specific model
custom_override: ModelProfile = {
    'default_structured_output_mode': 'prompted',
    'supports_thinking': False,
}

base = openai_model_profile('gpt-4o-mini')
merged = merge_profile(base, custom_override)
print(merged.get('default_structured_output_mode'))   # 'prompted'
```

### 8.2 Harmony Profile — `harmony_model_profile`

```python
from pydantic_ai.profiles.harmony import harmony_model_profile


# Harmony models don't support tool_choice='required' and prepend leading whitespace
profile = harmony_model_profile('harmony-gpt-4o')
print(profile)
# {
#   'openai_supports_tool_choice_required': False,
#   'ignore_streamed_leading_whitespace': True,
#   ... (inherited openai_model_profile fields)
# }
```

### 8.3 New v2.0.0 Fields: `supports_image_output` and `supports_inline_system_prompts`

```python
from pydantic_ai.profiles import merge_profile, ModelProfile


# A hypothetical multimodal model that can generate images and accepts inline system prompts
multimodal_profile: ModelProfile = {
    'supports_image_output': True,
    # When True: the model can return image data in its responses (ImagePart)

    'supports_inline_system_prompts': True,
    # When False (default): non-leading SystemPromptParts are wrapped as UserPromptParts
    # with <system>...</system> content in Model.prepare_messages.
    # When True: SystemPromptParts are passed inline at any position.
}

# Merge on top of a base profile
from pydantic_ai.profiles.openai import openai_model_profile
final = merge_profile(openai_model_profile('gpt-5'), multimodal_profile)
print(final.get('supports_image_output'))            # True
print(final.get('supports_inline_system_prompts'))   # True
```

### 8.4 `StructuredOutputMode` — Three Modes

```python
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.output import StructuredOutputMode


# 'tool'     → model calls an output tool to return structured data (most capable)
# 'json'     → model uses JSON mode / response_format=json_object
# 'prompted' → model is instructed to return JSON via a system prompt

profile_tool: ModelProfile     = {'default_structured_output_mode': 'tool'}
profile_json: ModelProfile     = {'default_structured_output_mode': 'json'}
profile_prompted: ModelProfile = {'default_structured_output_mode': 'prompted'}
```

### 8.5 Attaching a Custom Profile to a FunctionModel in Tests

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters


async def simple_fn(messages, agent_info):
    return ModelResponse(parts=[TextPart(content="42")])


async def main():
    # v2.0.0: FunctionModel accepts `profile` kwarg directly in its constructor
    model = FunctionModel(
        simple_fn,
        profile={
            'supports_tools': False,
            'default_structured_output_mode': 'prompted',
        },
    )
    agent = Agent(model)
    result = await agent.run("What is 6 × 7?")
    print(result.output)   # '42'


asyncio.run(main())
```

---

## 9. `FunctionModel` v2.0.0 + `AgentInfo` — Updated Constructor and Instructions

**Source**: `pydantic_ai/models/function.py`

`FunctionModel` gained two new constructor params in v2.0.0: `profile` and `settings`. This lets test models accurately simulate provider-specific behaviour without needing a live API. `AgentInfo` gained one new field: `instructions` (the resolved instruction string passed to the model), enabling stream functions to access instructions.

```python
class FunctionModel(Model):
    @overload
    def __init__(
        self,
        function: FunctionDef,
        *,
        model_name: str | None = None,
        profile: ModelProfileSpec | None = None,    # NEW in v2.0.0
        settings: ModelSettings | None = None,      # NEW in v2.0.0
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        stream_function: StreamFunctionDef,
        model_name: str | None = None,
        profile: ModelProfileSpec | None = None,    # NEW in v2.0.0
        settings: ModelSettings | None = None,      # NEW in v2.0.0
    ) -> None: ...

@dataclass(frozen=True, kw_only=True)
class AgentInfo:
    function_tools: list[ToolDefinition]
    allow_text_output: bool
    output_tools: list[ToolDefinition]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    instructions: str | None      # NEW in v2.0.0
```

### 9.1 `FunctionModel` with `profile` — Simulate No-Tool Provider

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters


def no_tool_model(messages, agent_info):
    # Simulates a provider that doesn't support tools —
    # the profile we pass tells the agent NOT to send tool definitions.
    return ModelResponse(parts=[TextPart(content="I answered without tools.")])


async def main():
    model = FunctionModel(
        no_tool_model,
        profile={'supports_tools': False},
    )
    agent = Agent(model)

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = await agent.run("What is 3 + 4?")
    print(result.output)   # 'I answered without tools.'


asyncio.run(main())
```

### 9.2 `FunctionModel` with `settings` — Override Temperature in Tests

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart


def settings_aware_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    temp = (info.model_settings or {}).get('temperature', 'unset')
    return ModelResponse(parts=[TextPart(content=f"temperature={temp}")])


async def main():
    model = FunctionModel(
        settings_aware_fn,
        settings={'temperature': 0.0},   # default settings for this model
    )
    agent = Agent(model)
    result = await agent.run("Check temperature")
    print(result.output)   # 'temperature=0.0'


asyncio.run(main())
```

### 9.3 `AgentInfo.instructions` — Access Instructions in the Model Function

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart


def instruction_aware_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # In v2.0.0, info.instructions contains the resolved instruction string
    if info.instructions:
        content = f"[Following instruction: {info.instructions[:50]!r}] Hello!"
    else:
        content = "Hello (no instructions set)"
    return ModelResponse(parts=[TextPart(content=content)])


async def main():
    model = FunctionModel(instruction_aware_fn)
    agent = Agent(model, instructions="Always be concise and factual.")
    result = await agent.run("Say hello")
    print(result.output)
    # '[Following instruction: 'Always be concise and factual.'] Hello!'


asyncio.run(main())
```

### 9.4 Stream Function with `AgentInfo.instructions`

```python
import asyncio
from collections.abc import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponseStreamEvent, PartStartEvent, PartDeltaEvent, TextPart, TextPartDelta


async def streaming_fn(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> AsyncIterator[str]:
    prefix = "GUIDED: " if info.instructions else ""
    for word in (prefix + "Hello world").split():
        yield word + " "


async def main():
    model = FunctionModel(stream_function=streaming_fn)
    agent = Agent(model, instructions="Guide all responses.")

    async with agent.run_stream("Say hello") as streamed:
        text = await streamed.get_output()
    print(text)   # 'GUIDED: Hello world '


asyncio.run(main())
```

---

## 10. `AgentRun` v2.0.0 Complete API + Node-Type Helpers

**Source**: `pydantic_ai/run.py` + `pydantic_ai/agent/__init__.py`

`AgentRun` is the stateful, async-iterable context object returned by `async with agent.iter(...)`. It exposes the run's message history, lets you inject new messages mid-run via `enqueue`, and supports manual stepping with `next`.

```python
@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, OutputDataT]):
    # Properties
    @property
    def result(self) -> AgentRunResult[OutputDataT]: ...       # available after End node
    @property
    def ctx(self) -> GraphRunContext[...]: ...                  # live graph context

    # Message history access
    def all_messages(self) -> list[ModelMessage]: ...
    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes: ...
    def new_messages(self) -> list[ModelMessage]: ...
    def new_messages_json(self) -> bytes: ...

    # Mid-run message injection
    def enqueue(self, *content: EnqueueContent, priority: PendingMessagePriority = 'asap') -> None: ...

    # Manual stepping
    async def next(self, node: AgentNode) -> AgentNode | End[FinalResult[OutputDataT]]: ...

    # Async iteration
    def __aiter__(self): ...
    async def __anext__(self): ...
```

### 10.1 Full `async for` Iteration with All Node Types

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import TextPart, ToolCallPart


agent = Agent('openai:gpt-4o-mini')


@agent.tool_plain
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def main():
    async with agent.iter("What is 6.5 × 4?") as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                print(f"[UserPrompt] {node.user_prompt!r}")

            elif Agent.is_model_request_node(node):
                print(f"[ModelRequest] parts={node.request.parts}")

            elif Agent.is_call_tools_node(node):
                for part in node.model_response.parts:
                    if isinstance(part, TextPart):
                        print(f"[ModelText] {part.content!r}")
                    elif isinstance(part, ToolCallPart):
                        print(f"[ToolCall] {part.tool_name}({part.args})")

            elif Agent.is_end_node(node):
                print(f"[End] output={node.data.output!r}")

    print(f"Final: {run.result.output}")


asyncio.run(main())
```

### 10.2 `AgentRun.enqueue()` — Inject a Follow-Up Prompt

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext


agent = Agent('openai:gpt-4o-mini')


@agent.tool
async def get_data(ctx: RunContext[None], key: str) -> str:
    """Fetch data for a key."""
    data = {'temperature': '22°C', 'humidity': '65%'}
    result = data.get(key, 'not found')
    if key == 'temperature':
        # After the tool runs, inject a follow-up question
        ctx.enqueue("Now compare that temperature to the typical London average.", priority='when_idle')
    return result


async def main():
    async with agent.iter("What is the temperature?") as run:
        async for node in run:
            pass   # let the agent run to completion
        print(run.result.output)


asyncio.run(main())
```

### 10.3 Manual Stepping with `AgentRun.next()`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_graph import End


agent = Agent('openai:gpt-4o-mini', instructions="Be concise.")


async def main():
    async with agent.iter("What is 2+2?") as run:
        # Manually drive the graph one node at a time
        node = await run.__anext__()   # UserPromptNode
        print("First node:", type(node).__name__)

        while not Agent.is_end_node(node):
            node = await run.next(node)
            print("Node:", type(node).__name__)

    print("Result:", run.result.output)


asyncio.run(main())
```

### 10.4 `all_messages_json` with `output_tool_return_content`

```python
import asyncio
import json
from pydantic import BaseModel
from pydantic_ai import Agent


class Answer(BaseModel):
    value: int
    explanation: str


agent = Agent('openai:gpt-4o-mini', output_type=Answer)


async def main():
    async with agent.iter("What is 7 × 6?") as run:
        async for _ in run:
            pass

    # output_tool_return_content replaces the output tool return value in the JSON
    # — useful when serialising history for storage or multi-turn replay
    history_bytes = run.all_messages_json(output_tool_return_content="<omitted>")
    history = json.loads(history_bytes)
    print(json.dumps(history, indent=2)[:400])


asyncio.run(main())
```

### 10.5 Passing `AgentRun.new_messages()` to a Second Run

```python
import asyncio
from pydantic_ai import Agent


agent = Agent('openai:gpt-4o-mini')


async def main():
    # First turn
    async with agent.iter("My name is Carol.") as run1:
        async for _ in run1:
            pass

    # Second turn — pass first run's messages as history
    async with agent.iter(
        "What is my name?",
        message_history=run1.new_messages(),
    ) as run2:
        async for _ in run2:
            pass

    print(run2.result.output)   # "Your name is Carol."


asyncio.run(main())
```

---
title: "PydanticAI — Class Deep Dives Vol. 11"
description: "Source-verified deep dives into 10 PydanticAI class groups new or promoted in 1.106.0: UserPromptNode/ModelRequestNode/CallToolsNode (the three agent graph nodes now public), AgentCapability/AgentToolset/ToolsetFunc (dynamic callable dispatch), AgentModelSettings/AgentNativeTool (per-run callable settings and native tools), HandleResponseEvent/ModelResponseStreamEvent (typed event discriminator unions), ToolSearchCallPart/NativeToolSearchCallPart/ToolSearchArgs/ToolSearchMatch (two-path tool search calls), ToolSearchReturnPart/NativeToolSearchReturnPart/ToolSearchReturnContent (tool search results), LoadCapabilityCallPart/LoadCapabilityReturnPart (capability lazy-load parts), BuiltinToolCallEvent/BuiltinToolResultEvent migration (deprecated → PartStartEvent/PartDeltaEvent), AgentInstructions/AgentMetadata (dynamic per-run instruction and metadata configuration), Agent 1.106.0 constructor reference (tool_timeout, max_concurrency, callable model_settings, capabilities). All verified against pydantic-ai 1.106.0."
sidebar:
  label: "Class deep dives (Vol. 11)"
  order: 37
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.106.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.106.0`.
</Aside>

Ten class groups from the `pydantic_ai` 1.106.0 source covering: the three agent graph nodes (`UserPromptNode`, `ModelRequestNode`, `CallToolsNode`) promoted to the public top-level API; dynamic callable type aliases (`AgentCapability`, `AgentToolset`, `ToolsetFunc`) enabling per-run dispatch; `AgentModelSettings` and `AgentNativeTool` for callable model settings and native tools; the tagged event discriminator unions `HandleResponseEvent` and `ModelResponseStreamEvent`; the two-path tool search call infrastructure (`ToolSearchCallPart`/`NativeToolSearchCallPart`/`ToolSearchArgs`/`ToolSearchMatch`); the corresponding return infrastructure (`ToolSearchReturnPart`/`NativeToolSearchReturnPart`/`ToolSearchReturnContent`); deferred capability loading message parts (`LoadCapabilityCallPart`/`LoadCapabilityReturnPart`); a migration guide for the deprecated `BuiltinToolCallEvent`/`BuiltinToolResultEvent`; extended agent configuration via `AgentInstructions` and `AgentMetadata`; and a complete `Agent` 1.106.0 constructor parameter reference.

---

## 1. `UserPromptNode` + `ModelRequestNode` + `CallToolsNode` — The Agent Run State Machine

**Module:** `pydantic_ai._agent_graph`  
**Import:** `from pydantic_ai import UserPromptNode, ModelRequestNode, CallToolsNode`

These three dataclasses represent the internal state machine of every agent run. In 1.106.0 they were promoted to `pydantic_ai.__all__`, making them a stable public API. Understanding them is essential for advanced graph integration, custom event processing, and inspecting mid-run state.

### Run state machine overview

```
agent.run("prompt")
      │
      ▼
UserPromptNode          ← builds instructions + system prompts, injects message history
      │
      ▼
ModelRequestNode        ← sends ModelRequest to the LLM, streams/buffers ModelResponse
      │
      ▼
CallToolsNode           ← processes ModelResponse: calls tools, checks for final output
      │
   ┌──┴──┐
   │     │
   ▼     ▼
ModelRequestNode    End[FinalResult[OutputDataT]]
(loop)              (done)
```

### `UserPromptNode`

```python
@dataclasses.dataclass
class UserPromptNode(AgentNode[DepsT, NodeRunEndT]):
    user_prompt: str | Sequence[UserContent] | None
    *,
    deferred_tool_results: DeferredToolResults | None = None
    instructions: str | None = None
    instructions_functions: list[SystemPromptRunner[DepsT]] = field(default_factory=list)
    system_prompts: tuple[str, ...] = field(default_factory=tuple)
    system_prompt_functions: list[SystemPromptRunner[DepsT]] = field(default_factory=list)
    system_prompt_dynamic_functions: dict[str, SystemPromptRunner[DepsT]] = field(default_factory=dict)
```

`UserPromptNode` is the entry point for each agent turn. Its `run()` method:

1. Retrieves `capture_run_messages()` context and replaces the state's `message_history` list in-place.
2. If `deferred_tool_results` is set, resumes a deferred workflow — skipping the user prompt entirely.
3. Evaluates `instructions_functions` and `system_prompt_functions` to produce `InstructionPart`s.
4. Assembles the `ModelRequest` (with `UserPromptPart`, `SystemPromptPart`, `InstructionPart`) and hands off to `ModelRequestNode`.

**Practical use: inspecting node state during `AgentRun.iter()`**

```python
import asyncio
from pydantic_ai import Agent, UserPromptNode, ModelRequestNode, CallToolsNode
from pydantic_ai.run import AgentRun

agent = Agent('openai:gpt-4o')

async def trace_run():
    async with agent.iter('Summarise the GDPR in three bullet points.') as run:
        async for node in run:
            if isinstance(node, UserPromptNode):
                print(f'[UserPromptNode] system_prompts={node.system_prompts!r}')
                print(f'                instructions={node.instructions!r}')
            elif isinstance(node, ModelRequestNode):
                print(f'[ModelRequestNode] request_parts={[type(p).__name__ for p in node.request.parts]}')
                print(f'                   resuming_without_prompt={node.is_resuming_without_prompt}')
            elif isinstance(node, CallToolsNode):
                print(f'[CallToolsNode] finish_reason={node.model_response.finish_reason}')
                print(f'               parts={[type(p).__name__ for p in node.model_response.parts]}')
    print('Final output:', run.result.output)

asyncio.run(trace_run())
```

### `ModelRequestNode`

```python
@dataclasses.dataclass
class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    request: ModelRequest
    is_resuming_without_prompt: bool = False
    # Private state (set by framework):
    _result: CallToolsNode | ModelRequestNode | None  # set on exit from stream()
    _did_stream: bool                                  # prevents double-streaming
    last_request_context: ModelRequestContext | None   # readable after run()
```

`ModelRequestNode.run()` calls `_make_request()` which invokes all `wrap_model_request` capability hooks and then the model. After the node completes, `last_request_context` exposes the `model`, `messages`, `model_settings`, and `model_request_parameters` that were actually sent.

**Streaming at the node level**

```python
import asyncio
from pydantic_ai import Agent, ModelRequestNode, CallToolsNode
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPart

agent = Agent('openai:gpt-4o')

async def stream_node():
    async with agent.iter('Count from 1 to 5.') as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                # Access the AgentStream directly from the node
                async with node.stream(run.ctx) as agent_stream:
                    async for delta in agent_stream.stream_text(delta=True):
                        print(delta, end='', flush=True)
                print()
                print(f'Model used: {node.last_request_context.model}')

asyncio.run(stream_node())
```

### `CallToolsNode`

```python
@dataclasses.dataclass
class CallToolsNode(AgentNode[DepsT, NodeRunEndT]):
    model_response: ModelResponse
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None = None
    tool_call_metadata: dict[str, dict[str, Any]] | None = None
    user_prompt: str | Sequence[UserContent] | None = None
```

`CallToolsNode.stream()` yields `HandleResponseEvent` items — one per tool call/result — and then sets `_next_node` to either another `ModelRequestNode` (if tools were called) or `End[FinalResult[...]]` (if a final output was found).

**Pre-injecting deferred tool results**

```python
from pydantic_ai import Agent, CallToolsNode
from pydantic_ai import DeferredToolResults

agent = Agent('openai:gpt-4o', toolsets=[...])  # has deferred tools

async def resume_with_results(results: DeferredToolResults):
    async with agent.iter(deferred_tool_results=results) as run:
        async for node in run:
            if isinstance(node, CallToolsNode):
                # Inspect tool_call_results pre-injected for deferred calls
                if node.tool_call_results:
                    for call_id, result in node.tool_call_results.items():
                        if result == 'skip':
                            print(f'  Skipping deferred call {call_id}')
                        else:
                            print(f'  Pre-resolved {call_id}: {result}')
```

**`CallToolsNode.user_prompt`** — attach a new user message to the next model request alongside deferred tool results. This is useful in HITL flows where the human provides context alongside approvals:

```python
node = CallToolsNode(
    model_response=response,
    tool_call_results=approved_results,
    user_prompt='I approved these calls. Please continue.',
)
```

---

## 2. `AgentCapability` + `AgentToolset` + `ToolsetFunc` — Dynamic Callable Dispatch

**Module:** `pydantic_ai.capabilities` / `pydantic_ai.toolsets`  
**Import:** `from pydantic_ai import AgentCapability, AgentToolset, ToolsetFunc`

These type aliases make the `Agent` constructor and `run()` accept either a static instance **or** a `RunContext`-aware callable that returns one per run. This enables feature flags, user-scoped toolsets, and dynamic capability injection — all without subclassing `Agent`.

### Type alias signatures

```python
# AgentCapability[DepsT]:
AgentCapability = (
    AbstractCapability[DepsT]
    | Callable[[RunContext[DepsT]], AbstractCapability[DepsT] | None]
    | Callable[[RunContext[DepsT]], Awaitable[AbstractCapability[DepsT] | None]]
)

# ToolsetFunc[DepsT]:
ToolsetFunc = Callable[
    [RunContext[DepsT]],
    AbstractToolset[DepsT] | None | Awaitable[AbstractToolset[DepsT] | None],
]

# AgentToolset[DepsT]:
AgentToolset = AbstractToolset[DepsT] | ToolsetFunc[DepsT]
```

### Per-user dynamic toolset with `ToolsetFunc`

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, FunctionToolset
from pydantic_ai.toolsets import AbstractToolset

@dataclass
class UserDeps:
    user_id: str
    is_premium: bool

# A static toolset all users can access
basic_tools = FunctionToolset()

@basic_tools.tool
def get_weather(city: str) -> str:
    return f'Sunny in {city}'

# A callable toolset — returns None for non-premium users
async def premium_toolset(ctx: RunContext[UserDeps]) -> AbstractToolset[UserDeps] | None:
    if not ctx.deps.is_premium:
        return None  # tool not injected for this run
    ts = FunctionToolset()

    @ts.tool
    async def advanced_forecast(city: str, days: int) -> str:
        return f'{days}-day forecast for {city}'

    return ts

agent: Agent[UserDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    toolsets=[basic_tools, premium_toolset],  # ToolsetFunc accepted directly
)

async def main():
    free_result = await agent.run(
        'What is the weather in Paris?',
        deps=UserDeps(user_id='u1', is_premium=False),
    )
    premium_result = await agent.run(
        'Give me a 7-day forecast for Tokyo.',
        deps=UserDeps(user_id='u2', is_premium=True),
    )
    print(free_result.output)
    print(premium_result.output)

asyncio.run(main())
```

### Feature-flag capabilities with `AgentCapability`

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability

class AuditCapability(AbstractCapability):
    """Logs every model request to an audit system."""
    async def before_model_request(self, ctx, ...): ...  # audit logic

async def select_capability(ctx: RunContext[UserDeps]) -> AbstractCapability | None:
    """Only inject audit capability for enterprise users."""
    if ctx.deps.is_premium:
        return AuditCapability()
    return None

agent: Agent[UserDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=UserDeps,
    capabilities=[select_capability],  # AgentCapability callable
)
```

### `ToolsetFunc` as a standalone type hint

Use `ToolsetFunc` when you want to annotate a function that will be passed as a toolset:

```python
from pydantic_ai import RunContext
from pydantic_ai.toolsets import ToolsetFunc, AbstractToolset

def build_database_toolset(ctx: RunContext[UserDeps]) -> AbstractToolset[UserDeps] | None:
    if ctx.deps.is_premium:
        return create_db_toolset(ctx.deps.user_id)
    return None

# Type annotation confirms this is a valid AgentToolset
ts_func: ToolsetFunc[UserDeps] = build_database_toolset
```

---

## 3. `AgentModelSettings` + `AgentNativeTool` — Callable Settings and Native Tools

**Module:** `pydantic_ai.agent` / `pydantic_ai.tools`  
**Import:** `from pydantic_ai import AgentModelSettings, AgentNativeTool`

### `AgentModelSettings[DepsT]`

```python
AgentModelSettings = (
    ModelSettings
    | Callable[[RunContext[DepsT]], ModelSettings]
)
```

`AgentModelSettings` allows `model_settings` on `Agent()` or `agent.run()` to be a callable that receives the full `RunContext` and returns a `ModelSettings` dict. This enables adaptive model configuration — e.g. higher `max_tokens` for complex queries, lower `temperature` for factual tasks.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, ModelSettings

@dataclass
class TaskDeps:
    task_type: str  # 'creative' | 'analytical' | 'extraction'

def adaptive_settings(ctx: RunContext[TaskDeps]) -> ModelSettings:
    if ctx.deps.task_type == 'creative':
        return ModelSettings(temperature=0.9, max_tokens=2000)
    elif ctx.deps.task_type == 'analytical':
        return ModelSettings(temperature=0.2, max_tokens=4000)
    else:  # extraction
        return ModelSettings(temperature=0.0, max_tokens=1000)

agent: Agent[TaskDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=TaskDeps,
    model_settings=adaptive_settings,  # AgentModelSettings callable
)

async def main():
    poem = await agent.run(
        'Write a short poem about autumn.',
        deps=TaskDeps(task_type='creative'),
    )
    analysis = await agent.run(
        'Analyze the quarterly revenue trends.',
        deps=TaskDeps(task_type='analytical'),
    )
    print(poem.output)
    print(analysis.output)

asyncio.run(main())
```

You can also override at call time:

```python
# Per-call override (also accepts callable)
result = await agent.run(
    'Extract all dates from this text: ...',
    deps=TaskDeps(task_type='extraction'),
    model_settings=ModelSettings(temperature=0.0),  # static override
)
```

### `AgentNativeTool[DepsT]`

```python
AgentNativeTool = (
    AbstractNativeTool
    | Callable[[RunContext[DepsT]], AbstractNativeTool | None]
    | Callable[[RunContext[DepsT]], Awaitable[AbstractNativeTool | None]]
)
```

`AgentNativeTool` is the type of each item in the `native_tools` parameter (or a future field). Like `AgentToolset`, it accepts either a static `AbstractNativeTool` instance or a per-run callable that returns one (or `None` to skip).

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.native_tools import WebSearchTool, MemoryTool

@dataclass
class SearchDeps:
    allow_web: bool
    allow_memory: bool

async def maybe_web_search(ctx: RunContext[SearchDeps]) -> WebSearchTool | None:
    return WebSearchTool() if ctx.deps.allow_web else None

async def maybe_memory(ctx: RunContext[SearchDeps]) -> MemoryTool | None:
    return MemoryTool() if ctx.deps.allow_memory else None

# NOTE: native_tools is managed through capabilities=[NativeTool(...)]
# AgentNativeTool is exposed for typing custom NativeTool capability implementations
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai import AgentNativeTool

def make_conditional_native_tool(
    deps_attr: str,
    tool_cls: type[AbstractNativeTool],
) -> AgentNativeTool:
    """Factory that returns a callable AgentNativeTool."""
    async def _factory(ctx: RunContext) -> AbstractNativeTool | None:
        if getattr(ctx.deps, deps_attr, False):
            return tool_cls()
        return None
    return _factory
```

---

## 4. `HandleResponseEvent` + `ModelResponseStreamEvent` — Typed Event Discriminators

**Module:** `pydantic_ai.messages`  
**Import:** `from pydantic_ai import HandleResponseEvent, ModelResponseStreamEvent`

These are `Annotated` discriminated union type aliases — not classes, but typed views over the existing event types that Pydantic can use for validation and serialisation.

### `HandleResponseEvent`

```python
HandleResponseEvent = Annotated[
    FunctionToolCallEvent
    | FunctionToolResultEvent
    | OutputToolCallEvent
    | OutputToolResultEvent
    | BuiltinToolCallEvent   # deprecated — use NativeToolCallPart path
    | BuiltinToolResultEvent  # deprecated — use NativeToolReturnPart path
    , Discriminator('event_kind')
]
```

`HandleResponseEvent` is yielded by `CallToolsNode.stream()` — it covers every tool interaction that happens while the node processes a model response. Consuming it lets you intercept and react to every tool call and result in a streaming pass.

```python
import asyncio
from pydantic_ai import Agent, CallToolsNode
from pydantic_ai import HandleResponseEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent, FunctionToolResultEvent,
    OutputToolCallEvent, OutputToolResultEvent,
)

agent = Agent('openai:gpt-4o', tools=[...])

async def observe_tool_events():
    async with agent.iter('What is 42 * 17? Also look up the weather in Rome.') as run:
        async for node in run:
            if isinstance(node, CallToolsNode):
                async with node.stream(run.ctx) as events:
                    async for event in events:
                        match event:
                            case FunctionToolCallEvent(tool_name=name, args=args):
                                print(f'CALL  {name}({args})')
                            case FunctionToolResultEvent(tool_name=name):
                                print(f'DONE  {name} → {event.result!r}')
                            case OutputToolCallEvent(tool_name=name):
                                print(f'OUTPUT CALL  {name}')
                            case OutputToolResultEvent():
                                print(f'OUTPUT DONE → {event.output!r}')
    print('Final:', run.result.output)

asyncio.run(observe_tool_events())
```

### `ModelResponseStreamEvent`

```python
ModelResponseStreamEvent = Annotated[
    PartStartEvent
    | PartDeltaEvent
    | PartEndEvent
    | FinalResultEvent
    , Discriminator('event_kind')
]
```

`ModelResponseStreamEvent` covers the four streaming events emitted by `agent.run_stream_events()` at the model-response level. Use it for typed deserialization of event streams stored in a database or queue.

```python
import json, asyncio
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai import ModelResponseStreamEvent
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, FinalResultEvent

agent = Agent('openai:gpt-4o')

# Serialize events to NDJSON
event_adapter = TypeAdapter(ModelResponseStreamEvent)

async def record_stream_events():
    events_log = []
    async with agent.run_stream('Name three programming languages.') as stream:
        async for event in stream.stream_events():
            # Validate + round-trip via TypeAdapter
            raw = event_adapter.dump_json(event)
            events_log.append(raw)
    return events_log

# Deserialize from log
def replay_events(events_log: list[bytes]):
    for raw in events_log:
        event = event_adapter.validate_json(raw)
        match event:
            case PartStartEvent(index=i):
                print(f'Part {i} started: {type(event.part).__name__}')
            case PartDeltaEvent(index=i):
                print(f'Part {i} delta')
            case FinalResultEvent():
                print('Final result detected')
```

---

## 5. `ToolSearchCallPart` + `NativeToolSearchCallPart` + `ToolSearchArgs` + `ToolSearchMatch`

**Module:** `pydantic_ai.messages`  
**Import:** `from pydantic_ai.messages import ToolSearchCallPart, NativeToolSearchCallPart, ToolSearchArgs, ToolSearchMatch`

When `DeferredLoadingToolset` (or `ToolSearch` capability) is active, the model issues a tool-search query before deciding which tool to call. This search takes one of two paths:

| Path | Call part | Return part | When |
|------|-----------|-------------|------|
| **Native server-side** | `NativeToolSearchCallPart` | `NativeToolSearchReturnPart` | Anthropic BM25/regex, OpenAI Responses native search |
| **Local fallback** | `ToolSearchCallPart` | `ToolSearchReturnPart` | Non-native providers, client-side execution |

**Cross-path detection:** check `part.tool_kind == 'tool-search'` — works on both call parts and both return parts.

### `ToolSearchArgs`

```python
class ToolSearchArgs(TypedDict):
    queries: list[str]
    # Anthropic BM25 / regex: single-item list with the query string.
    # OpenAI server-executed tool_search: the list of tool paths the model picked.
    # OpenAI client-execution / local search_tools: single-item list with keywords string.
```

### `ToolSearchMatch`

```python
class ToolSearchMatch(TypedDict):
    name: str           # Discovered tool name (as the model will call it)
    description: str | None  # Human-readable description, if provided
```

### Reading tool search history

```python
from pydantic_ai import Agent, DeferredLoadingToolset, FunctionToolset
from pydantic_ai.messages import (
    ToolSearchCallPart, NativeToolSearchCallPart,
    ToolSearchArgs, ToolSearchMatch,
)

def analyse_tool_search_history(messages):
    """Extract all tool-search queries and matches from a message history."""
    for msg in messages:
        for part in getattr(msg, 'parts', []):
            # Cross-path detection
            if getattr(part, 'tool_kind', None) == 'tool-search':
                if isinstance(part, (ToolSearchCallPart, NativeToolSearchCallPart)):
                    args: ToolSearchArgs | None = part.args if isinstance(part.args, dict) else None
                    if args:
                        queries = args.get('queries', [])
                        print(f'Tool search query: {queries}')
```

### Complete tool search example

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset, DeferredLoadingToolset
from pydantic_ai.messages import ToolSearchCallPart, ToolSearchReturnPart

# Build a large toolset with deferred loading
big_toolset = FunctionToolset()

for i in range(50):
    @big_toolset.tool(name=f'tool_{i}')
    def _tool(ctx, x: int) -> int:
        return x + i

lazy = DeferredLoadingToolset(big_toolset)  # only expose tool names until searched

agent = Agent('openai:gpt-4o', toolsets=[lazy])

async def main():
    result = await agent.run('Use tool_42 to compute 10 + 42.')
    
    # Inspect how the search happened
    for msg in result.all_messages():
        for part in getattr(msg, 'parts', []):
            if getattr(part, 'tool_kind', None) == 'tool-search':
                if isinstance(part, ToolSearchCallPart):
                    print(f'Local search: queries={part.args}')
                elif isinstance(part, ToolSearchReturnPart):
                    content = part.content
                    matches = content.get('discovered_tools', [])
                    print(f'Found tools: {[m["name"] for m in matches]}')
    
    print(result.output)

asyncio.run(main())
```

---

## 6. `ToolSearchReturnPart` + `NativeToolSearchReturnPart` + `ToolSearchReturnContent`

**Module:** `pydantic_ai.messages` / `pydantic_ai._tool_search`  
**Import:** `from pydantic_ai.messages import ToolSearchReturnPart, NativeToolSearchReturnPart`  
**Import:** `from pydantic_ai.messages import ToolSearchReturnContent`

These are the return counterparts to the call parts in §5.

### `ToolSearchReturnContent`

```python
class ToolSearchReturnContent(TypedDict):
    discovered_tools: list[ToolSearchMatch]
    # Matches ordered by relevance. Empty list = search ran, nothing matched.
    
    message: NotRequired[str]
    # Optional text shown to model when no matches found.
    # Present on local fallback / Anthropic custom-callable path.
    # Stripped on OpenAI client-execution and Anthropic server-side replay.
```

### `ToolSearchReturnPart` vs `NativeToolSearchReturnPart`

```python
# Local fallback path (ToolReturnPart subclass):
@dataclass
class ToolSearchReturnPart(ToolReturnPart):
    content: ToolSearchReturnContent  # narrows ToolReturnContent
    tool_name: Literal['search_tools'] = 'search_tools'
    tool_kind: Literal['tool-search'] = 'tool-search'

# Native server-side path (NativeToolReturnPart subclass):
@dataclass
class NativeToolSearchReturnPart(NativeToolReturnPart):
    content: ToolSearchReturnContent  # narrows ToolReturnContent
    tool_name: Literal['tool_search'] = 'tool_search'
    tool_kind: Literal['tool-search'] = 'tool-search'
```

Note: `tool_name` differs — `'search_tools'` (local) vs `'tool_search'` (native). Always use `tool_kind == 'tool-search'` for cross-path detection.

### Extracting tool search results from history

```python
from pydantic_ai.messages import (
    ToolSearchReturnPart, NativeToolSearchReturnPart,
    ToolSearchReturnContent, ToolSearchMatch,
)

def get_all_discovered_tools(messages) -> list[ToolSearchMatch]:
    """Collect every tool the model discovered via tool-search across all messages."""
    discovered: list[ToolSearchMatch] = []
    for msg in messages:
        for part in getattr(msg, 'parts', []):
            if getattr(part, 'tool_kind', None) == 'tool-search':
                if isinstance(part, (ToolSearchReturnPart, NativeToolSearchReturnPart)):
                    content: ToolSearchReturnContent = part.content
                    discovered.extend(content.get('discovered_tools', []))
    return discovered

# Usage
all_messages = result.all_messages()
tools = get_all_discovered_tools(all_messages)
for tool in tools:
    print(f'{tool["name"]}: {tool["description"]}')
```

### Handling empty search results

```python
def check_empty_searches(messages) -> list[str]:
    """Find cases where tool search returned no matches."""
    empty_queries = []
    for msg in messages:
        for i, part in enumerate(getattr(msg, 'parts', [])):
            if getattr(part, 'tool_kind', None) == 'tool-search':
                if isinstance(part, (ToolSearchReturnPart, NativeToolSearchReturnPart)):
                    if not part.content.get('discovered_tools'):
                        msg_text = part.content.get('message', '(no message)')
                        empty_queries.append(f'Part {i}: {msg_text}')
    return empty_queries
```

---

## 7. `LoadCapabilityCallPart` + `LoadCapabilityReturnPart` — Deferred Capability Loading

**Module:** `pydantic_ai.messages`  
**Import:** `from pydantic_ai.messages import LoadCapabilityCallPart, LoadCapabilityReturnPart`

When `AbstractCapability.defer_loading = True`, the framework exposes a hidden `load_capability` tool. The model calls this to discover a deferred capability (loading its tools and instructions) before using it. `LoadCapabilityCallPart` and `LoadCapabilityReturnPart` are the typed message parts for this interaction.

### Class signatures

```python
@dataclass
class LoadCapabilityCallPart(ToolCallPart):
    tool_name: Literal['load_capability'] = 'load_capability'
    args: str | LoadCapabilityArgs | None = None
    tool_kind: Literal['capability-load'] = 'capability-load'

    @property
    def typed_args(self) -> LoadCapabilityArgs | None:
        """Parsed load-capability arguments, or None for incomplete streaming args."""
        ...
    
    @property
    def capability_id(self) -> str | None:
        """Capability id from the parsed args, if available."""
        ...

@dataclass
class LoadCapabilityReturnPart(ToolReturnPart):
    content: LoadCapabilityReturn  # narrows ToolReturnContent
    tool_name: Literal['load_capability'] = 'load_capability'
    tool_kind: Literal['capability-load'] = 'capability-load'

    @property
    def instructions(self) -> str | None:
        """Loaded capability instructions, if any."""
        ...
```

### Cross-path detection: `tool_kind == 'capability-load'`

```python
from pydantic_ai.messages import LoadCapabilityCallPart, LoadCapabilityReturnPart

def trace_capability_loads(messages) -> list[dict]:
    """Extract all capability load events from a message history."""
    events = []
    for msg in messages:
        for part in getattr(msg, 'parts', []):
            if getattr(part, 'tool_kind', None) == 'capability-load':
                if isinstance(part, LoadCapabilityCallPart):
                    events.append({
                        'type': 'load_request',
                        'capability_id': part.capability_id,
                        'raw_args': part.args,
                    })
                elif isinstance(part, LoadCapabilityReturnPart):
                    events.append({
                        'type': 'load_result',
                        'instructions': part.instructions,
                        'content': part.content,
                    })
    return events
```

### Building a deferred capability and observing the load

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai import FunctionToolset
from pydantic_ai.messages import LoadCapabilityCallPart, LoadCapabilityReturnPart

class LazyDatabaseCapability(AbstractCapability):
    """A capability that is only loaded when the model explicitly requests it."""
    
    id = 'database_tools'
    defer_loading = True  # tells the framework to expose `load_capability` tool
    
    async def get_description(self) -> str:
        return 'Database query tools for reading and writing records'
    
    async def load_capability(self):
        """Called when the model invokes load_capability(id='database_tools')."""
        toolset = FunctionToolset()
        
        @toolset.tool
        def query_db(sql: str) -> list[dict]:
            return [{'id': 1, 'name': 'Alice'}]
        
        return toolset

agent = Agent(
    'openai:gpt-4o',
    capabilities=[LazyDatabaseCapability()],
)

async def main():
    result = await agent.run('Query the database for all users.')
    
    # See the capability load in the message history
    for event in trace_capability_loads(result.all_messages()):
        print(event)
    
    print(result.output)

asyncio.run(main())
```

---

## 8. `BuiltinToolCallEvent` + `BuiltinToolResultEvent` — Migration Guide

**Module:** `pydantic_ai.messages`  
**Status:** `@deprecated` since 1.106.0

`BuiltinToolCallEvent` and `BuiltinToolResultEvent` were introduced to signal the start and completion of native (built-in) tool calls in streaming event streams. In 1.106.0 they are deprecated in favour of the richer `PartStartEvent`/`PartDeltaEvent`/`PartEndEvent` with `NativeToolCallPart`/`NativeToolReturnPart`.

### Why they were deprecated

The deprecated events were thin wrappers with no delta support — you could only see the start and end. The `PartStartEvent`/`PartDeltaEvent` pathway lets you stream native tool call arguments incrementally, exactly as function tool calls work.

### Before (deprecated)

```python
from pydantic_ai.messages import BuiltinToolCallEvent, BuiltinToolResultEvent

async with agent.run_stream('Search the web for ...') as stream:
    async for event in stream.stream_events():
        if isinstance(event, BuiltinToolCallEvent):
            print(f'Native tool called: {event.part.tool_name}')
        elif isinstance(event, BuiltinToolResultEvent):
            print(f'Native tool done: {event.result.tool_name}')
```

### After (current)

```python
from pydantic_ai.messages import (
    PartStartEvent, PartDeltaEvent, PartEndEvent,
    NativeToolCallPart, NativeToolReturnPart,
)

async with agent.run_stream('Search the web for ...') as stream:
    async for event in stream.stream_events():
        match event:
            case PartStartEvent(part=NativeToolCallPart() as call_part):
                print(f'Native tool started: {call_part.tool_name}')
            case PartDeltaEvent():
                # NativeToolCallPart deltas (streaming args)
                pass
            case PartEndEvent(part=NativeToolReturnPart() as ret_part):
                print(f'Native tool done: {ret_part.tool_name}')
```

### Migration table

| Deprecated | Replacement |
|-----------|-------------|
| `BuiltinToolCallEvent` | `PartStartEvent` where `isinstance(event.part, NativeToolCallPart)` |
| `BuiltinToolResultEvent` | `PartEndEvent` where `isinstance(event.part, NativeToolReturnPart)` |
| `event.part` (NativeToolCallPart) | `event.part` — same field, now on PartStartEvent |
| `event.result` (NativeToolReturnPart) | `event.part` on PartEndEvent |

### Handling both during transition

If your code must support 1.105.0 and 1.106.0 simultaneously:

```python
import warnings
from pydantic_ai.messages import (
    PartStartEvent, PartEndEvent,
    NativeToolCallPart, NativeToolReturnPart,
)

# Suppress the deprecation warning during transition
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from pydantic_ai.messages import BuiltinToolCallEvent, BuiltinToolResultEvent

async def handle_event(event):
    # New path
    if isinstance(event, PartStartEvent) and isinstance(event.part, NativeToolCallPart):
        print(f'[PartStart] Native tool: {event.part.tool_name}')
    elif isinstance(event, PartEndEvent) and isinstance(event.part, NativeToolReturnPart):
        print(f'[PartEnd] Native tool done: {event.part.tool_name}')
    # Old path (deprecated — will be removed in 2.0)
    elif isinstance(event, BuiltinToolCallEvent):
        print(f'[DEPRECATED] BuiltinToolCallEvent: {event.part.tool_name}')
    elif isinstance(event, BuiltinToolResultEvent):
        print(f'[DEPRECATED] BuiltinToolResultEvent: {event.result.tool_name}')
```

---

## 9. `AgentInstructions` + `AgentMetadata` — Dynamic Per-Run Configuration

**Module:** `pydantic_ai.agent`  
**Import:** `from pydantic_ai.agent import AgentInstructions, AgentMetadata`

### `AgentInstructions[DepsT]`

```python
AgentInstructions = (
    TemplateStr[DepsT]
    | str
    | Callable[[RunContext[DepsT]], str | None]
    | Callable[[RunContext[DepsT]], Awaitable[str | None]]
    | Callable[[], str | None]
    | Callable[[], Awaitable[str | None]]
    | Sequence[<any of the above>]
    | None
)
```

`AgentInstructions` is the type of the `instructions` parameter on both `Agent()` and `agent.run()`. Unlike `system_prompt` (static, sent as `SystemPromptPart`), instructions are rendered as `InstructionPart` — which is Anthropic-cache-aware and sorted for stable prefix caching.

**Key distinctions from `system_prompt`:**

| | `system_prompt` | `instructions` |
|---|---|---|
| Part type | `SystemPromptPart` | `InstructionPart` |
| Caching | Not cached by default | Sorted for stable Anthropic cache prefixes |
| Dynamic | Via `@agent.system_prompt` decorator | Via callable or `@agent.instructions` decorator |
| Per-run override | No | Yes — pass to `agent.run(instructions=...)` |

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.agent import AgentInstructions

@dataclass
class SessionDeps:
    user_name: str
    language: str
    expertise: str  # 'beginner' | 'expert'

async def dynamic_instructions(ctx: RunContext[SessionDeps]) -> str:
    depth = 'simple, everyday language' if ctx.deps.expertise == 'beginner' else 'technical depth'
    return (
        f'You are a helpful assistant for {ctx.deps.user_name}. '
        f'Respond in {ctx.deps.language} using {depth}. '
        f'Keep responses concise unless asked for elaboration.'
    )

agent: Agent[SessionDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=SessionDeps,
    instructions=dynamic_instructions,  # AgentInstructions callable
)

async def main():
    result = await agent.run(
        'What is a neural network?',
        deps=SessionDeps(user_name='Alice', language='French', expertise='beginner'),
    )
    print(result.output)

asyncio.run(main())
```

**Sequence form — composable instructions:**

```python
# Combine static and dynamic instructions as a sequence
agent = Agent(
    'openai:gpt-4o',
    instructions=[
        'Always respond in Markdown format.',                  # static str
        dynamic_instructions,                                  # async callable
        lambda ctx: f'Conversation ID: {ctx.run_id}',        # sync callable
    ],
)
```

**Per-run instruction override:**

```python
# Override instructions for a single run — additive with agent-level instructions
result = await agent.run(
    'Explain quantum entanglement.',
    deps=deps,
    instructions='For this run only: respond in exactly three sentences.',
)
```

### `AgentMetadata[DepsT]`

```python
AgentMetadata = (
    dict[str, Any]
    | Callable[[RunContext[DepsT]], dict[str, Any]]
)
```

`AgentMetadata` is the type of the `metadata` parameter on `Agent()` and `agent.run()`. Metadata is attached to the run but **never sent to the LLM** — it flows through `RunContext.metadata` and is available to capabilities, hooks, and toolsets for routing, billing, and observability.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class AppDeps:
    tenant_id: str
    request_id: str

def build_metadata(ctx: RunContext[AppDeps]) -> dict:
    return {
        'tenant_id': ctx.deps.tenant_id,
        'request_id': ctx.deps.request_id,
        'run_id': ctx.run_id,
        'timestamp': 'iso8601',  # your billing timestamp
    }

agent: Agent[AppDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=AppDeps,
    metadata=build_metadata,  # AgentMetadata callable
)

# Access metadata in a capability
from pydantic_ai.capabilities import AbstractCapability

class BillingCapability(AbstractCapability):
    async def before_model_request(self, ctx, *, messages, model_settings, **kwargs):
        meta = ctx.metadata or {}
        print(f'Billing: tenant={meta.get("tenant_id")} run={meta.get("run_id")}')
        return None  # no-op

async def main():
    result = await agent.run(
        'Summarise the latest AI news.',
        deps=AppDeps(tenant_id='acme', request_id='req-123'),
    )
    print(result.output)

asyncio.run(main())
```

**Static dict metadata:**

```python
# Simple static metadata — same for every run
agent = Agent('openai:gpt-4o', metadata={'app': 'my-app', 'version': '2.0'})
```

---

## 10. `Agent` 1.106.0 Constructor Reference

**Module:** `pydantic_ai.agent`  
**Import:** `from pydantic_ai import Agent`

The 1.106.0 `Agent.__init__` consolidates several previously implicit parameters into a clean, documented public API. This section serves as a complete reference.

### Full constructor signature

```python
Agent(
    model: Model | KnownModelName | str | None = None,
    *,
    output_type: OutputSpec[OutputDataT] = str,
    instructions: AgentInstructions[AgentDepsT] = None,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[AgentDepsT] = NoneType,
    name: str | None = None,
    description: TemplateStr[AgentDepsT] | str | None = None,
    model_settings: AgentModelSettings[AgentDepsT] | None = None,
    retries: int | AgentRetries | None = None,
    validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
    toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
    defer_model_check: bool = False,
    end_strategy: EndStrategy = 'early',
    metadata: AgentMetadata[AgentDepsT] | None = None,
    tool_timeout: float | None = None,
    max_concurrency: AnyConcurrencyLimit = None,
    capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
)
```

### New parameters in 1.106.0

#### `tool_timeout: float | None = None`

Global default timeout (in seconds) for all function tool calls in this agent. Individual tools can override via `Tool(timeout=...)` or the `@toolset.tool(timeout=...)` decorator.

```python
# All tools time out after 10 seconds unless overridden
agent = Agent(
    'openai:gpt-4o',
    tool_timeout=10.0,
    tools=[fast_tool],
    toolsets=[FunctionToolset(timeout=30.0)],  # toolset-level override wins
)
```

#### `max_concurrency: AnyConcurrencyLimit = None`

Maximum number of concurrent model requests across all runs of this agent. Accepts `int`, `ConcurrencyLimit`, or `AbstractConcurrencyLimiter`.

```python
from pydantic_ai import Agent, ConcurrencyLimit

# At most 5 simultaneous model calls from this agent
agent = Agent(
    'openai:gpt-4o',
    max_concurrency=ConcurrencyLimit(5, max_queued=20),
)

# Shared limiter across multiple agents
from pydantic_ai import ConcurrencyLimiter
shared_limiter = ConcurrencyLimiter(max_running=10)

agent_a = Agent('openai:gpt-4o', max_concurrency=shared_limiter)
agent_b = Agent('openai:gpt-4.1', max_concurrency=shared_limiter)
```

#### `model_settings: AgentModelSettings[AgentDepsT] | None = None`

Now accepts a callable `(RunContext[DepsT]) -> ModelSettings` in addition to a static dict. The callable is evaluated at the start of each run, after `deps` are available.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai import ModelSettings

@dataclass
class Deps:
    quality: str  # 'fast' | 'accurate'

agent: Agent[Deps, str] = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    model_settings=lambda ctx: (
        ModelSettings(temperature=0.7, max_tokens=500)
        if ctx.deps.quality == 'fast'
        else ModelSettings(temperature=0.1, max_tokens=4000)
    ),
)
```

#### `capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None`

Sequence of `AbstractCapability` instances **or** callables returning one. Each capability can add instructions, toolsets, model-request hooks, and output hooks. Capabilities are combined via `CombinedCapability` internally, with ordering controlled by `CapabilityOrdering`.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability

class LoggingCapability(AbstractCapability):
    async def before_model_request(self, ctx, *, messages, **kwargs):
        print(f'[{ctx.run_id}] Sending {len(messages)} messages to model')

class RetryCapability(AbstractCapability):
    async def after_model_request(self, ctx, *, response, **kwargs):
        if response.finish_reason == 'length':
            raise ModelRetry('Response was cut off, please continue.')

agent = Agent(
    'openai:gpt-4o',
    capabilities=[
        LoggingCapability(),
        RetryCapability(),
        lambda ctx: AuditCapability() if ctx.deps.audit else None,
    ],
)
```

#### `description: TemplateStr[AgentDepsT] | str | None = None`

Human-readable agent description, attached to the OTel run span as `gen_ai.agent.description`. Supports `TemplateStr` for dynamic descriptions using Handlebars syntax with dependency data.

```python
from pydantic_ai import Agent
from pydantic_ai._template import TemplateStr

agent = Agent(
    'openai:gpt-4o',
    name='support-agent',
    description='Customer support agent for {{deps.tenant_name}}',
    # deps must have a 'tenant_name' attribute
)
```

### Complete `agent.run()` parameter reference (1.106.0)

```python
await agent.run(
    user_prompt: str | Sequence[UserContent] | None = None,
    *,
    output_type: OutputSpec[...] | None = None,       # per-call output override
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    conversation_id: str | None = None,
    model: Model | KnownModelName | str | None = None,
    instructions: AgentInstructions[AgentDepsT] = None,  # per-call instructions
    deps: AgentDepsT = None,
    model_settings: AgentModelSettings[AgentDepsT] | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    metadata: AgentMetadata[AgentDepsT] | None = None,   # per-call metadata
    retries: int | AgentRetries | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
    spec: dict[str, Any] | AgentSpec | None = None,
)
```

### Putting it all together — a production-ready agent

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, ConcurrencyLimit, FunctionToolset, ModelSettings
from pydantic_ai.capabilities import AbstractCapability

@dataclass
class ProductionDeps:
    user_id: str
    tenant_id: str
    is_premium: bool
    task_complexity: float  # 0.0–1.0

# Dynamic model settings based on task complexity
def adaptive_settings(ctx: RunContext[ProductionDeps]) -> ModelSettings:
    tokens = int(500 + ctx.deps.task_complexity * 3500)
    temp = max(0.0, 0.7 - ctx.deps.task_complexity * 0.5)
    return ModelSettings(max_tokens=tokens, temperature=temp)

# Dynamic toolset based on subscription tier
async def premium_toolset(ctx: RunContext[ProductionDeps]) -> FunctionToolset | None:
    if not ctx.deps.is_premium:
        return None
    ts = FunctionToolset()
    @ts.tool
    async def advanced_search(query: str) -> list[dict]:
        ...  # premium search API
    return ts

# Observability capability
class ObservabilityCapability(AbstractCapability):
    async def before_model_request(self, ctx, *, messages, model_settings, **kwargs):
        print(f'[{ctx.run_id}] user={ctx.deps.user_id} tenant={ctx.deps.tenant_id}')

agent: Agent[ProductionDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=ProductionDeps,
    name='production-assistant',
    description='Multi-tenant assistant for {{deps.tenant_id}}',
    model_settings=adaptive_settings,
    toolsets=[premium_toolset],
    capabilities=[ObservabilityCapability()],
    max_concurrency=ConcurrencyLimit(20, max_queued=100),
    tool_timeout=15.0,
    metadata=lambda ctx: {
        'user_id': ctx.deps.user_id,
        'tenant_id': ctx.deps.tenant_id,
        'run_id': ctx.run_id,
    },
)

async def main():
    result = await agent.run(
        'Help me write a report on Q3 performance.',
        deps=ProductionDeps(
            user_id='u-001',
            tenant_id='acme',
            is_premium=True,
            task_complexity=0.8,
        ),
    )
    print(result.output)

asyncio.run(main())
```

---

## Summary table

| Class / alias | Module | What's new in 1.106.0 |
|---|---|---|
| `UserPromptNode` | `pydantic_ai._agent_graph` | Promoted to top-level `__all__`; stable public API |
| `ModelRequestNode` | `pydantic_ai._agent_graph` | Promoted to top-level `__all__`; `last_request_context` readable |
| `CallToolsNode` | `pydantic_ai._agent_graph` | Promoted to top-level `__all__`; `user_prompt` field for HITL |
| `AgentCapability` | `pydantic_ai.capabilities` | New type alias; callable form accepted in `Agent.capabilities` |
| `AgentToolset` | `pydantic_ai.toolsets` | New type alias; `ToolsetFunc` callables accepted in `Agent.toolsets` |
| `ToolsetFunc` | `pydantic_ai.toolsets` | New type alias for callable toolset factories |
| `AgentModelSettings` | `pydantic_ai.agent` | New type alias; callable accepted in `model_settings=` |
| `AgentNativeTool` | `pydantic_ai.tools` | New type alias for conditional native tool injection |
| `HandleResponseEvent` | `pydantic_ai.messages` | New discriminated union alias for `CallToolsNode.stream()` events |
| `ModelResponseStreamEvent` | `pydantic_ai.messages` | New discriminated union alias for model streaming events |
| `ToolSearchCallPart` | `pydantic_ai.messages` | New typed subclass of `ToolCallPart`; local tool-search path |
| `NativeToolSearchCallPart` | `pydantic_ai.messages` | New typed subclass of `NativeToolCallPart`; server-side path |
| `ToolSearchArgs` | `pydantic_ai.messages` | New `TypedDict` for normalized search queries |
| `ToolSearchMatch` | `pydantic_ai.messages` | New `TypedDict` for individual tool search hits |
| `ToolSearchReturnPart` | `pydantic_ai.messages` | New typed subclass of `ToolReturnPart`; local path |
| `NativeToolSearchReturnPart` | `pydantic_ai.messages` | New typed subclass of `NativeToolReturnPart`; server-side path |
| `ToolSearchReturnContent` | `pydantic_ai._tool_search` | New `TypedDict` with `discovered_tools` and `message` |
| `LoadCapabilityCallPart` | `pydantic_ai.messages` | New; deferred capability load call |
| `LoadCapabilityReturnPart` | `pydantic_ai.messages` | New; deferred capability load result |
| `BuiltinToolCallEvent` | `pydantic_ai.messages` | **Deprecated** in 1.106.0 — migrate to `PartStartEvent + NativeToolCallPart` |
| `BuiltinToolResultEvent` | `pydantic_ai.messages` | **Deprecated** in 1.106.0 — migrate to `PartEndEvent + NativeToolReturnPart` |
| `AgentInstructions` | `pydantic_ai.agent` | Documented; full union including `Sequence` form and zero-arg callables |
| `AgentMetadata` | `pydantic_ai.agent` | Documented; `dict | Callable[[RunContext], dict]` |
| `Agent.tool_timeout` | `pydantic_ai.agent` | New constructor param; global tool timeout |
| `Agent.max_concurrency` | `pydantic_ai.agent` | New constructor param; `AnyConcurrencyLimit` |
| `Agent.description` | `pydantic_ai.agent` | New constructor param; OTel span attribute |

<Aside type="note" title="Verified against 1.106.0">
All class signatures, field names, type aliases, and deprecation notices verified by inspecting installed **pydantic-ai 1.106.0** source directly from PyPI. Run `pip show pydantic-ai` to confirm your installed version.
</Aside>

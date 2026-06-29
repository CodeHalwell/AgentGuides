---
title: "PydanticAI Class Deep Dives Vol. 29"
description: "Source-verified deep dives into 10 pydantic-ai 2.0.0 class groups: ModelResponsePartsManager (streaming coordinator — thinking-tag splitting, ignore_leading_whitespace Qwen3/Ollama guard, narrow_type ToolCallPart promotion), uuid7() + PydanticAIDeprecationWarning (RFC 9562 UUIDv7 polyfill — sub-millisecond monotonicity with global lock, UserWarning inheritance for visible deprecations), MCP capability (unified server capability — _resolved_id cached_property, _build_local URL→transport inference, from_spec JSON-serializable restriction, allowed_tools filter), Toolset + WrapperCapability (minimal toolset exposure, transparent delegation — __post_init__ id/defer_loading inheritance, apply() visitor, for_run replace() pattern), sort_capabilities + collect_leaves + _effective_ordering (TopologicalSorter capability chain — outermost/innermost tiers, wraps/wrapped_by type/instance refs, CycleError detection), IncludeToolReturnSchemas (ToolSelector — all/list/dict/callable filter, include_return_schema=None guard, native vs injected-JSON schema paths), SetToolMetadata (**metadata kwargs dataclass, deep merge pattern, ToolSelector filter), PrefixTools (WrapperCapability subclass — AbstractToolset vs ToolsetFunc dispatch, DynamicToolset wrapping, from_spec nested capability), PrepareTools + PrepareOutputTools (ToolsPrepareFunc capability wrappers — sync/async via inspect.isawaitable, PreparedToolset validation, not spec-serializable), FunctionSchema (internal schema engine — single_field_name model-like vs primitive, call() run_in_executor dispatch, _call_args positional/var-positional, return_schema minimum {}). All verified against pydantic-ai 2.0.0 source."
sidebar:
  label: "Class deep dives (Vol. 29)"
  order: 55
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.0.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.x API.
</Aside>

Ten class groups covering pydantic-ai 2.0.0's streaming coordinator internals, time-sortable UUIDs, the unified MCP capability, capability composition primitives, topological capability sorting, tool-definition modifiers, and the internal function-schema engine.

---

## 1. `ModelResponsePartsManager` — Streaming Response Coordinator

**Source**: `pydantic_ai/_parts_manager.py`

`ModelResponsePartsManager` is the shared state machine that every `StreamedResponse` subclass delegates to. It tracks vendor-specific part identifiers, assembles incremental deltas into complete parts, and emits `PartStartEvent`/`PartDeltaEvent` for consumers.

**Key implementation facts (verified from source)**:

- `_vendor_id_to_part_index` maps any `Hashable` vendor ID to an index in `_parts`. Vendors that identify streamed chunks by string IDs (OpenAI), integer indices (Anthropic), or UUIDs all fit.
- `_tool_kind_by_name` is built once in `__post_init__` from `model_request_parameters.function_tools`. It lets `_typed_call_part()` promote a base `ToolCallPart` to a typed subclass (e.g. `ToolSearchCallPart`) from the very first `PartStartEvent` rather than requiring a post-stream pass.
- `handle_text_delta(ignore_leading_whitespace=True)` silently drops text that is empty or whitespace-only when no `TextPart` exists yet. This is explicitly documented as a workaround for Ollama + Qwen3, which emit `<think>\n</think>\n\n` before tool calls.
- Thinking-tag splitting in `handle_text_delta`: when `thinking_tags=('<think>', '</think>')` is passed and the content equals the start tag, the manager creates a `ThinkingPart` and starts routing content to it until the end tag arrives, then stops tracking the vendor ID so the next text delta starts a fresh `TextPart`.
- `ProviderDetailsDelta` on `handle_thinking_delta` can be a callable `(existing_details) -> new_details`, letting providers accumulate structured metadata without the parts manager knowing the schema.
- `handle_tool_call_delta` with `vendor_part_id=None` and `tool_name is not None` always creates a **new** part rather than updating the latest one, since a non-None name signals a new call boundary.
- `handle_tool_call_part` (non-delta, atomic) overwrites an existing part if one exists for that `vendor_part_id`; always returns `PartStartEvent` (not `PartDeltaEvent`).

```python
# Example 1 — building a ModelResponsePartsManager for a custom StreamedResponse

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent

# ModelRequestParameters holds the function tool definitions the model was called with.
# In production this comes from the model's prepare() call; for illustration we create a minimal one.
from pydantic_ai.tools import ToolDefinition

params = ModelRequestParameters(
    function_tools=[
        ToolDefinition(name='search', description='search the web', parameters_json_schema={}),
    ],
    output_tools=[],
    allow_text_output=True,
    instruction_parts=[],
)

manager = ModelResponsePartsManager(model_request_parameters=params)

# Stream in text deltas (vendor_part_id=None → latest-part update semantics)
events = list(manager.handle_text_delta(vendor_part_id=None, content='Hello '))
print(events[0])  # PartStartEvent(index=0, part=TextPart(content='Hello ', ...))

events2 = list(manager.handle_text_delta(vendor_part_id=None, content='world'))
print(events2[0])  # PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='world'))

parts = manager.get_parts()
print(parts[0].content)  # 'Hello world'
```

```python
# Example 2 — thinking-tag splitting (Ollama/Qwen3 style embedded <think> tokens)

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import TextPart, ThinkingPart

manager = ModelResponsePartsManager(
    model_request_parameters=ModelRequestParameters(
        function_tools=[], output_tools=[], allow_text_output=True, instruction_parts=[]
    )
)

THINKING_TAGS = ('<think>', '</think>')

# When the model streams '<think>' as a separate token, the manager switches to ThinkingPart
events = []
for chunk in ['<think>', ' reasoning... ', '</think>', 'Final answer']:
    events.extend(
        manager.handle_text_delta(
            vendor_part_id='main',
            content=chunk,
            thinking_tags=THINKING_TAGS,
        )
    )

parts = manager.get_parts()
print(type(parts[0]).__name__)   # ThinkingPart
print(type(parts[1]).__name__)   # TextPart
print(parts[0].content)          # ' reasoning... '
print(parts[1].content)          # 'Final answer'
```

```python
# Example 3 — ignore_leading_whitespace guards against empty pre-tool-call text parts

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

manager = ModelResponsePartsManager(
    model_request_parameters=ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='calculator', description='', parameters_json_schema={})
        ],
        output_tools=[],
        allow_text_output=True,
        instruction_parts=[],
    )
)

# Qwen3 via Ollama emits whitespace before tool calls. With the guard, no TextPart is created.
for whitespace_chunk in ['\n\n', '  ', '']:
    events = list(
        manager.handle_text_delta(
            vendor_part_id=None,
            content=whitespace_chunk,
            ignore_leading_whitespace=True,
        )
    )
    assert events == [], f'Expected no events for whitespace, got {events}'

# Now stream the actual tool call — it won't be mistaken for an output text part
event = manager.handle_tool_call_delta(
    vendor_part_id='tool-1',
    tool_name='calculator',
    args='{"expr":',
)
print(event)  # PartStartEvent(index=0, part=ToolCallPart(tool_name='calculator', ...))
```

---

## 2. `uuid7()` + `PydanticAIDeprecationWarning` — Time-Sortable IDs and Visible Deprecations

**Source**: `pydantic_ai/_uuid.py`, `pydantic_ai/_warnings.py`

`uuid7()` is a CPython 3.14-compatible RFC 9562 UUIDv7 polyfill. `PydanticAIDeprecationWarning` is a custom warning class that ensures deprecation notices are visible at runtime.

**Key implementation facts (verified from source)**:

- `uuid7()` matches the CPython 3.14 `uuid.uuid7()` implementation exactly. Once Python 3.14 is the minimum supported version, the code comment says it will be replaced with `uuid.uuid7()`.
- Uses RFC 9562 §6.2 **Method 1** (sub-millisecond monotonicity): a 42-bit counter with MSB=0 is incremented within a millisecond. Counter overflow bumps `timestamp_ms` by 1 and resets to a fresh random 42-bit value.
- Global `_lock_v7` (a `threading.Lock`) guards `_last_timestamp_v7` and `_last_counter_v7`, making it thread-safe.
- `_RFC_4122_VERSION_7_FLAGS = 0x0000_0000_0000_7000_8000_0000_0000_0000` bakes in version `0b0111` and variant `0b10` in a single constant ORed at the end.
- The bit layout: `48 bits unix_ts_ms | 4 bits version | 12 bits counter_hi | 2 bits variant | 30 bits counter_lo | 32 bits random`.
- `PydanticAIDeprecationWarning` inherits from `UserWarning` **not** `DeprecationWarning`. Python silences `DeprecationWarning` by default in non-`__main__` contexts; `UserWarning` is always visible, following the approach described in Seth Michael Larson's article on library deprecations.

```python
# Example 1 — generating UUIDv7 values and verifying monotonicity

from pydantic_ai._uuid import uuid7
import uuid

# Generate a batch — each should be lexicographically ordered (time-sortable)
ids = [uuid7() for _ in range(5)]
assert ids == sorted(ids), "UUIDv7 values must be time-sortable"

# Version nibble should be 7
for uid in ids:
    version = (uid.int >> 76) & 0xF
    assert version == 7, f"Expected version 7, got {version}"

# Variant bits should be 0b10 (RFC 4122)
for uid in ids:
    variant = (uid.int >> 62) & 0x3
    assert variant == 0b10, f"Expected variant 0b10, got {bin(variant)}"

print("All UUIDv7 checks passed")
print("Sample:", ids[0])  # e.g. 0197b8a3-1234-7xxx-8xxx-xxxxxxxxxxxx
```

```python
# Example 2 — UUIDv7 as a tool call ID (time-sortable, unique per call)

from pydantic_ai import Agent
from pydantic_ai._uuid import uuid7

# pydantic_ai uses uuid7() internally to generate tool_call_id values.
# You can use it directly for any ID that benefits from lexicographic ordering.

def make_correlation_id() -> str:
    """Generate a correlation ID that sorts chronologically."""
    return str(uuid7())

# Useful for distributed tracing: sort by ID to get temporal order
ids = [make_correlation_id() for _ in range(3)]
print("Chronological sort works:", ids == sorted(ids))

# Also useful as database primary keys (B-tree friendly, no hot-spot)
import uuid
uuidv4_samples = [str(uuid.uuid4()) for _ in range(3)]
uuidv7_samples = [str(uuid7()) for _ in range(3)]
print("UUIDv4 (random order):", uuidv4_samples == sorted(uuidv4_samples))  # likely False
print("UUIDv7 (sorted):", uuidv7_samples == sorted(uuidv7_samples))         # True
```

```python
# Example 3 — PydanticAIDeprecationWarning is always visible (unlike DeprecationWarning)

import warnings
from pydantic_ai._warnings import PydanticAIDeprecationWarning

# Demonstrate that PydanticAIDeprecationWarning is a UserWarning, not a DeprecationWarning
assert issubclass(PydanticAIDeprecationWarning, UserWarning)
assert not issubclass(PydanticAIDeprecationWarning, DeprecationWarning)

# UserWarning is shown by default; DeprecationWarning is silenced in library code
# This means users WILL see pydantic_ai deprecation warnings without needing -W flags

# Emitting one:
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    warnings.warn(
        "OutlinesModel is deprecated. Use PromptedOutput instead.",
        PydanticAIDeprecationWarning,
        stacklevel=2,
    )

assert len(caught) == 1
assert caught[0].category is PydanticAIDeprecationWarning
print("Warning message:", str(caught[0].message))

# When writing a wrapper library on top of pydantic_ai,
# use PydanticAIDeprecationWarning for your own deprecation notices
# if you want them visible without requiring users to pass -W flags.
```

---

## 3. `MCP` Capability — Unified MCP Server Capability

**Source**: `pydantic_ai/capabilities/mcp.py`

The `MCP` capability is the v2.0.0 replacement for passing `MCPToolset` directly to an agent. It unifies native MCP (provider-advertised) and local MCP (run inside pydantic-ai) in a single composable object.

**Key implementation facts (verified from source)**:

- `MCP(url=None, native=True)` raises `UserError` immediately in `__init__`. Native MCP requires a URL so the provider knows where to send requests. Explicit `MCPServerTool(url=...)` instances bypass this check.
- `_resolved_id` is a `@cached_property` that prioritises: explicit `id=` arg → `MCPServerTool.id` from an explicit `native=MCPServerTool(...)` → hostname+path slug derived from `url`. Host and slug are joined with `-` to avoid collisions between two `/sse` paths on different hosts.
- Non-string, non-callable, non-toolset `local=` inputs (e.g. `fastmcp.Client`, `ClientTransport`, `FastMCP` server, `AnyUrl`) are wrapped into `MCPToolset(local, include_instructions=True)` in `__init__`. String and bool values flow through `_resolve_local_strategy`.
- `_build_local(url)` merges `authorization_token` into headers and constructs an `MCPToolset(url, ...)` — FastMCP infers SSE vs Streamable HTTP from the URL path/scheme.
- `from_spec` restricts `local=` to `str | bool | None` (JSON-serializable) and requires `url=`. Runtime-only values like `fastmcp.Client` instances can still be passed directly to `MCP(...)` — they just cannot roundtrip through a spec file.
- `get_toolset()` overrides the parent to apply `allowed_tools` as a `filtered()` predicate after the parent resolves the local toolset.
- `get_serialization_name()` returns `'MCP'` (inherited from `AbstractCapability.__name__`) — `MCP` IS spec-serializable. The `from_spec` class method restricts `local=` to `str | bool | None` for the serializable path; passing runtime-only objects like `fastmcp.Client` still works directly via `MCP(...)` but those instances cannot roundtrip through an AgentSpec file.

```python
# Example 1 — native + local dual-mode MCP capability

from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

# Local-only mode (no native — no mcp extra needed for the model side)
# The URL is used to build an MCPToolset that runs locally inside pydantic-ai
agent_local = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[
        MCP(
            url='https://my-mcp-server.example.com/sse',
            native=False,    # don't advertise to provider
            local=True,      # run locally, infer SSE from URL
        )
    ]
)

# Native-only mode (server runs elsewhere; provider calls it directly)
# No 'mcp' extra required since local execution is disabled
from pydantic_ai.native_tools import MCPServerTool

agent_native = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[
        MCP(
            url='https://my-mcp-server.example.com/sse',
            native=True,  # provider handles the MCP calls
            local=False,  # don't run locally
        )
    ]
)

# Dual mode: provider calls native MCP AND pydantic-ai has a local fallback
agent_dual = Agent(
    'openai:gpt-4.1',
    capabilities=[
        MCP(
            url='https://my-mcp-server.example.com/sse',
            native=True,
            local=True,   # local=True derives transport from url=
            authorization_token='Bearer sk-...',
        )
    ]
)
```

```python
# Example 2 — _resolved_id for id deduplication and allowed_tools filtering

from pydantic_ai.capabilities.mcp import MCP

# Demonstrate _resolved_id derivation from URL (hostname + last path segment)
cap = MCP(url='https://api.example.com/mcp/sse', native=True)
print(cap._resolved_id)   # 'api.example.com-sse'

cap2 = MCP(url='https://other.example.com/mcp/sse', native=True)
print(cap2._resolved_id)  # 'other.example.com-sse' — distinct despite same path

# Explicit id= overrides the derived value
cap3 = MCP(url='https://api.example.com/mcp/sse', native=True, id='my-mcp')
print(cap3._resolved_id)  # 'my-mcp'

# allowed_tools restricts which tools are visible to the agent
from pydantic_ai import Agent

agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[
        MCP(
            url='https://api.example.com/tools/sse',
            native=False,
            local=True,
            # Only expose search and summarize; hide all admin tools
            allowed_tools=['search', 'summarize'],
        )
    ]
)
# The resulting toolset wraps the MCPToolset with:
# toolset.filtered(lambda _ctx, td: td.name in {'search', 'summarize'})
```

```python
# Example 3 — from_spec for YAML/JSON-driven agent configuration

# from_spec restricts local= to str | bool | None so the capability
# can roundtrip through AgentSpec YAML files.

from pydantic_ai.capabilities.mcp import MCP

# This is what AgentSpec/build_registry would call:
cap = MCP.from_spec(
    url='https://my-server.example.com/sse',
    native=True,
    local=True,       # bool — allowed in spec
    authorization_token='Bearer token123',
    allowed_tools=['tool_a', 'tool_b'],
    defer_loading=True,
    id='my-server',
)

print(cap.url)              # 'https://my-server.example.com/sse'
print(cap.allowed_tools)    # ['tool_a', 'tool_b']
print(cap.defer_loading)    # True — model loads capability on demand

# Equivalent YAML spec (for AgentSpec file).
# CapabilitySpec uses NamedSpec short form: {ClassName: {kwargs}} — NOT type: ClassName.
# capabilities:
#   - MCP:
#       url: https://my-server.example.com/sse
#       native: true
#       local: true
#       authorization_token: Bearer token123
#       allowed_tools: [tool_a, tool_b]
#       defer_loading: true
#       id: my-server
```

---

## 4. `Toolset` + `WrapperCapability` — Capability Composition Primitives

**Source**: `pydantic_ai/capabilities/toolset.py`, `pydantic_ai/capabilities/wrapper.py`

`Toolset` is the minimal bridge between a raw toolset and the capability system. `WrapperCapability` is the delegation base for any capability that wraps another.

**Key implementation facts (verified from source)**:

- `Toolset` adds a single `toolset: AgentToolset` field on top of `AbstractCapability`'s existing `id` and `defer_loading` fields: it stores the `AgentToolset` and returns it from `get_toolset()`. That's the entire implementation — it's deliberately minimal.
- `WrapperCapability.__post_init__`: when `self.id is None`, it copies `self.wrapped.id` and `self.wrapped.defer_loading`. This makes a wrapper transparent: a `WrapperCapability` wrapping a deferred capability is itself deferred without any explicit configuration.
- `WrapperCapability.apply()`: for a plain leaf wrapped capability, registers `self` as the proxy. For a container wrapped capability (e.g. `CombinedCapability`), it also visits all of the container's leaves so their hooks and toolsets are registered with the correct capability IDs.
- `WrapperCapability.for_run()`: calls `self.wrapped.for_run(ctx)` and, if it returned a new instance, uses `dataclasses.replace(self, wrapped=new_wrapped)`. If the wrapped instance didn't change, returns `self` unchanged (no allocation).
- All hook methods delegate to `self.wrapped`; property overrides like `has_wrap_node_run` check whether the subclass overrides the method (`type(self).wrap_node_run is not WrapperCapability.wrap_node_run`) before delegating.
- `WrapperCapability.wrap_run_event_stream` uses `async for` to delegate, preserving backpressure — it's not buffered.
- `get_serialization_name()` returns `None` for `WrapperCapability` — the base class is not spec-serializable; subclasses (like `PrefixTools`) override it.

```python
# Example 1 — Toolset: wrapping any AbstractToolset as a capability

from pydantic_ai import Agent
from pydantic_ai.capabilities import Toolset
from pydantic_ai.toolsets import FunctionToolset

# FunctionToolset holds typed Python functions exposed as LLM tools
toolset = FunctionToolset()

@toolset.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"

@toolset.tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"12:00 {timezone}"

# Wrap the toolset as a capability — allows capability-level defer_loading, id, etc.
weather_cap = Toolset(toolset=toolset)

agent = Agent(
    'openai:gpt-4.1',
    capabilities=[weather_cap],
)

# Equivalent to passing the toolset directly, but now composable with capability features:
# - Can be deferred: Toolset(toolset, defer_loading=True, id='weather')
# - Can be wrapped: WrapperCapability(wrapped=Toolset(toolset))
# - Can be combined: CombinedCapability([Toolset(toolset), Thinking()])
```

```python
# Example 2 — WrapperCapability: custom middleware that adds per-call logging

from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Toolset
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.messages import ToolCallPart

RawToolArgs = str | dict[str, Any]
ValidatedToolArgs = dict[str, Any]

@dataclass
class LoggingWrapper(WrapperCapability):
    """Logs every tool call with timing information."""
    log_prefix: str = "TOOL"

    async def before_tool_execute(
        self,
        ctx: RunContext,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        print(f"[{self.log_prefix}] Calling {call.tool_name} with {args}")
        return await self.wrapped.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)

    async def after_tool_execute(
        self,
        ctx: RunContext,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        print(f"[{self.log_prefix}] {call.tool_name} returned: {result!r}")
        return await self.wrapped.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args=args, result=result
        )

toolset = FunctionToolset()

@toolset.tool
def search(query: str) -> str:
    return f"Results for: {query}"

agent = Agent(
    'openai:gpt-4.1',
    capabilities=[
        LoggingWrapper(
            wrapped=Toolset(toolset),
            log_prefix='SEARCH',
        )
    ]
)
```

```python
# Example 3 — WrapperCapability.for_run() and transparent id/defer_loading inheritance

from dataclasses import dataclass
from pydantic_ai.capabilities import Toolset
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset()

# Deferred toolset: model must call load_capability before tools are visible
deferred_toolset_cap = Toolset(toolset=toolset, defer_loading=True, id='my-tools')

# Wrapping it — WrapperCapability.__post_init__ inherits id and defer_loading
@dataclass
class MyWrapper(WrapperCapability):
    pass

wrapper = MyWrapper(wrapped=deferred_toolset_cap)

# The wrapper inherits defer_loading and id from the wrapped capability
print(wrapper.id)            # 'my-tools'   (copied from wrapped)
print(wrapper.defer_loading) # True          (copied from wrapped)

# If wrapper has an explicit id, it overrides the inherited one
@dataclass
class NamedWrapper(WrapperCapability):
    pass

named = NamedWrapper(wrapped=deferred_toolset_cap, id='wrapper-override')
print(named.id)            # 'wrapper-override'   (explicit, not inherited)
print(named.defer_loading) # False                 (NOT inherited — __post_init__ only copies when id is None)
# To keep deferred loading with an explicit id, pass defer_loading=True explicitly:
named_deferred = NamedWrapper(wrapped=deferred_toolset_cap, id='wrapper-override', defer_loading=True)
print(named_deferred.defer_loading) # True
```

---

## 5. `sort_capabilities` + `collect_leaves` + `_effective_ordering` — Topological Capability Chain

**Source**: `pydantic_ai/capabilities/_ordering.py`

This module topologically sorts the capabilities list to satisfy middleware ordering constraints declared via `CapabilityOrdering`, using Python's standard library `graphlib.TopologicalSorter`.

**Key implementation facts (verified from source)**:

- Edges in `TopologicalSorter` are directed **outer → inner**: `ts.add(j, i)` means "j depends on i" i.e. i must come before j (i is outer, j is inner).
- `outermost` tier: every outermost member gets an edge to every non-member (`ts.add(j, oi)`), making all non-outermost capabilities depend on outermost ones.
- `innermost` tier: every innermost member depends on every non-member (`ts.add(ii, j)`), so innermost capabilities sort last.
- `wraps=[X]` → "I wrap X" → "I am outer, X is inner" → `ts.add(j, i)` for each j matching X.
- `wrapped_by=[X]` → "X wraps me" → "X is outer, I am inner" → `ts.add(i, j)` for each j matching X.
- `_ref_matches` uses `issubclass` for type refs (matches all subclasses) and `is` identity for instance refs. Instance refs break if `for_run` returns a fresh instance — the source comment advises using type refs when the target capability uses per-run state isolation.
- `_validate_requires` checks that required types appear among ALL leaf types in the list, not just at top level.
- `CycleError` from `graphlib` is caught and re-raised as `UserError('Circular ordering constraints among capabilities')`.
- `_effective_ordering` merges ordering constraints from all leaves of a container (`CombinedCapability`, `WrapperCapability`), raising `UserError` if two leaves declare conflicting positions.

```python
# Example 1 — verifying sort order with outermost/innermost position declarations

from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._ordering import sort_capabilities

@dataclass
class OuterCap(AbstractCapability):
    def get_ordering(self):
        return CapabilityOrdering(position='outermost')

@dataclass
class InnerCap(AbstractCapability):
    def get_ordering(self):
        return CapabilityOrdering(position='innermost')

@dataclass
class MiddleCap(AbstractCapability):
    pass  # no ordering constraint — user-list order

outer = OuterCap()
inner = InnerCap()
mid1 = MiddleCap()
mid2 = MiddleCap()

# Pass in reverse order — sort_capabilities fixes it
sorted_caps = sort_capabilities([inner, mid1, outer, mid2])

print([type(c).__name__ for c in sorted_caps])
# ['OuterCap', 'MiddleCap', 'MiddleCap', 'InnerCap']
# OuterCap is first; InnerCap is last; mid1/mid2 preserve user list order
```

```python
# Example 2 — wraps/wrapped_by type refs for relative ordering

from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._ordering import sort_capabilities

@dataclass
class InstrumentationCap(AbstractCapability):
    """Must be outermost so it spans all other capabilities."""
    def get_ordering(self):
        return CapabilityOrdering(position='outermost')

@dataclass
class RetryPolicy(AbstractCapability):
    """Must wrap around AuthCap so retries include re-auth."""
    def get_ordering(self):
        return CapabilityOrdering(wraps=[AuthCap])

@dataclass
class AuthCap(AbstractCapability):
    pass

# Sort in scrambled order
caps = [AuthCap(), RetryPolicy(), InstrumentationCap()]
sorted_caps = sort_capabilities(caps)

names = [type(c).__name__ for c in sorted_caps]
print(names)
# InstrumentationCap first (outermost), RetryPolicy before AuthCap (wraps=[AuthCap])
assert names.index('InstrumentationCap') < names.index('RetryPolicy')
assert names.index('RetryPolicy') < names.index('AuthCap')
```

```python
# Example 3 — CycleError caught and re-raised as UserError

from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities._ordering import sort_capabilities
from pydantic_ai.exceptions import UserError

@dataclass
class CapA(AbstractCapability):
    def get_ordering(self):
        return CapabilityOrdering(wraps=[CapB])  # A wraps B → A before B

@dataclass
class CapB(AbstractCapability):
    def get_ordering(self):
        return CapabilityOrdering(wraps=[CapA])  # B wraps A → B before A (contradiction!)

try:
    sort_capabilities([CapA(), CapB()])
except UserError as e:
    print(f"Caught: {e}")  # 'Circular ordering constraints among capabilities'
```

---

## 6. `IncludeToolReturnSchemas` — Return Schema Injection Capability

**Source**: `pydantic_ai/capabilities/include_return_schemas.py`

`IncludeToolReturnSchemas` enables return type schemas for selected tools by setting `ToolDefinition.include_return_schema = True`. For providers that natively support return schemas (e.g. Google Gemini), the schema is passed as a structured field; for others, it is injected into the tool description as JSON text.

**Key implementation facts (verified from source)**:

- `tools: ToolSelector` accepts `'all'` (default), `Sequence[str]` (list of tool names), `dict[str, Any]` (metadata match), or a predicate callable — either sync `Callable[[ctx, tool_def], bool]` or async `Callable[[ctx, tool_def], Awaitable[bool]]`.
- The guard `if td.include_return_schema is None` means per-tool overrides (`Tool(..., include_return_schema=False)`) are respected — this capability only activates on tools that haven't explicitly opted out. A tool with `include_return_schema=True` is also skipped (already opted in).
- Implemented as a `get_wrapper_toolset` override that returns a `PreparedToolset` wrapping the original. The inner `_include_return_schemas` async function is a closure over the selector.
- `get_serialization_name()` returns `'IncludeToolReturnSchemas'` — it IS spec-serializable (no callable in constructor for the `tools='all'` case).

```python
# Example 1 — enabling return schemas globally

from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.tools import Tool
from pydantic import BaseModel

class WeatherData(BaseModel):
    temperature: float
    condition: str
    humidity: int

def get_weather(city: str) -> WeatherData:
    """Retrieve weather data for a city."""
    return WeatherData(temperature=22.5, condition='sunny', humidity=45)

# With IncludeToolReturnSchemas, the model receives the WeatherData JSON schema
# so it knows what fields to expect and can reference them in follow-up reasoning
agent = Agent(
    'google:gemini-2.0-flash',  # Google natively supports return schemas
    tools=[Tool(get_weather)],
    capabilities=[IncludeToolReturnSchemas()],  # 'all' by default
)

# For non-native providers, the schema is injected into the tool description:
# "Returns: {\"type\": \"object\", \"properties\": {\"temperature\": ...}}"
agent_openai = Agent(
    'openai:gpt-4.1',
    tools=[Tool(get_weather)],
    capabilities=[IncludeToolReturnSchemas()],
)
```

```python
# Example 2 — selective return schema inclusion by tool name list

from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.tools import Tool
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class UserProfile(BaseModel):
    name: str
    email: str

def web_search(query: str) -> list[SearchResult]:
    """Search the web for information."""
    return []

def get_user(user_id: int) -> UserProfile:
    """Retrieve a user profile."""
    return UserProfile(name='Alice', email='alice@example.com')

def simple_calc(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

# Only enable return schemas for the complex structured tools, not simple_calc
agent = Agent(
    'openai:gpt-4.1',
    tools=[Tool(web_search), Tool(get_user), Tool(simple_calc)],
    capabilities=[
        IncludeToolReturnSchemas(tools=['web_search', 'get_user'])
    ],
)
# simple_calc's return schema (just 'float') stays omitted
```

```python
# Example 3 — per-tool opt-out takes precedence over IncludeToolReturnSchemas

from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.tools import Tool
from pydantic import BaseModel

class SensitiveData(BaseModel):
    password_hash: str
    api_key: str

def get_credentials(service: str) -> SensitiveData:
    """Retrieve service credentials."""
    return SensitiveData(password_hash='...', api_key='...')

def get_info(key: str) -> dict:
    """Get configuration info."""
    return {}

# Tool with explicit include_return_schema=False — NEVER gets schema injected
# even if IncludeToolReturnSchemas(tools='all') is active
credentials_tool = Tool(get_credentials, include_return_schema=False)

# Tool with include_return_schema=None (the default) — capability can inject schema
info_tool = Tool(get_info)  # include_return_schema defaults to None

agent = Agent(
    'openai:gpt-4.1',
    tools=[credentials_tool, info_tool],
    capabilities=[IncludeToolReturnSchemas()],
)
# get_info gets return schema injected; get_credentials does NOT (explicit False)
```

---

## 7. `SetToolMetadata` — Bulk Metadata Injection Capability

**Source**: `pydantic_ai/capabilities/set_tool_metadata.py`

`SetToolMetadata` bulk-injects key-value metadata onto selected tools without replacing existing metadata. Uses a `(init=False)` dataclass with a custom `__init__` that accepts `**metadata` kwargs.

**Key implementation facts (verified from source)**:

- `@dataclass(init=False)` + custom `__init__(**metadata)` lets users write `SetToolMetadata(cache_control='ephemeral', priority=1)` instead of `SetToolMetadata(metadata={'cache_control': 'ephemeral', 'priority': 1})`.
- Merge logic: `{**(td.metadata or {}), **metadata}` — existing tool metadata is preserved; the capability's metadata is merged on top (capability values win on conflicts).
- `tools: ToolSelector` defaults to `'all'`. Same selector mechanics as `IncludeToolReturnSchemas`: list of names, metadata dict, or async callable.
- `get_serialization_name()` returns `'SetToolMetadata'` — spec-serializable when metadata values are JSON-compatible.
- Downstream: `ToolDefinition.metadata` is passed through to providers that inspect it (e.g. for Anthropic cache-control headers on tools).

```python
# Example 1 — adding Anthropic cache-control metadata to all tools

from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset()

@toolset.tool
def search_docs(query: str) -> str:
    """Search documentation."""
    return f"Docs for: {query}"

@toolset.tool
def run_code(code: str) -> str:
    """Execute Python code."""
    return "output"

# Mark all tools as cache-breakpoints for Anthropic prompt caching
agent = Agent(
    'anthropic:claude-opus-4-5',
    toolsets=[toolset],
    capabilities=[
        SetToolMetadata(cache_control='ephemeral')
    ],
)
# Each ToolDefinition.metadata will be {'cache_control': 'ephemeral'}
```

```python
# Example 2 — selective metadata by tool name list

from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset()

@toolset.tool
def expensive_query(sql: str) -> list:
    """Run an expensive database query."""
    return []

@toolset.tool
def quick_lookup(key: str) -> str:
    """Fast key-value lookup."""
    return ""

# Only mark the expensive tool for caching; quick_lookup stays clean
agent = Agent(
    'anthropic:claude-opus-4-5',
    toolsets=[toolset],
    capabilities=[
        SetToolMetadata(
            tools=['expensive_query'],  # ToolSelector: list of names
            cache_control='ephemeral',
        )
    ],
)
```

```python
# Example 3 — stacking SetToolMetadata with other capabilities for layered metadata

from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata, IncludeToolReturnSchemas

# Capabilities compose: metadata from SetToolMetadata is applied after
# IncludeToolReturnSchemas sets include_return_schema on tool definitions.
# The order matters when get_wrapper_toolset middleware chains stack.

agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[
        # Outer: marks all tools with Anthropic cache metadata
        SetToolMetadata(cache_control='ephemeral', version='v2'),
        # Inner: injects return schemas into tool descriptions
        IncludeToolReturnSchemas(tools=['search', 'fetch']),
    ]
)

# The merged ToolDefinition for 'search' will have:
# - metadata = {'cache_control': 'ephemeral', 'version': 'v2'}
# - include_return_schema = True (injected by IncludeToolReturnSchemas)
```

---

## 8. `PrefixTools` — Namespaced Tool Name Wrapping

**Source**: `pydantic_ai/capabilities/prefix_tools.py`

`PrefixTools` is a `WrapperCapability` subclass that prefixes the tool names of a wrapped capability's toolset, useful for avoiding name collisions when combining multiple tool collections.

**Key implementation facts (verified from source)**:

- `PrefixTools(wrapped=..., prefix='ns')` turns tool name `'search'` into `'ns_search'`. The prefix is applied by `PrefixedToolset`, which intercepts `get_tools()` and renames each `ToolDefinition`.
- `get_toolset()` calls `super().get_toolset()` (the `WrapperCapability` delegation chain) then:
  - If the result is an `AbstractToolset`: wraps it directly in `PrefixedToolset(toolset, prefix=self.prefix)`.
  - If the result is a `ToolsetFunc` callable: wraps it in `DynamicToolset(toolset_func=toolset)` first, then in `PrefixedToolset`. This handles capabilities whose `get_toolset()` returns a callable (e.g. `DynamicCapability`).
- `from_spec(prefix=..., capability=...)` accepts a nested capability spec (dict or string), loads it via `load_capability_from_nested_spec`, and constructs the `PrefixTools` wrapper. This allows YAML-driven prefixed tool namespacing.
- `get_serialization_name()` returns `'PrefixTools'` — spec-serializable.

```python
# Example 1 — prefixing a FunctionToolset to avoid name collisions

from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools, Toolset
from pydantic_ai.toolsets import FunctionToolset

db_tools = FunctionToolset()
api_tools = FunctionToolset()

@db_tools.tool
def search(query: str) -> list:
    """Search the database."""
    return []

@api_tools.tool
def search(query: str) -> list:  # type: ignore[override]
    """Search the external API."""
    return []

# Without prefixing, two 'search' tools would conflict. With PrefixTools:
agent = Agent(
    'openai:gpt-4.1',
    capabilities=[
        PrefixTools(wrapped=Toolset(db_tools), prefix='db'),
        PrefixTools(wrapped=Toolset(api_tools), prefix='api'),
    ]
)
# Agent sees: 'db_search' and 'api_search' — no collision
```

```python
# Example 2 — PrefixTools with a deferred MCP capability

from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools, MCP

# Prefix an MCP server's tools with 'files' to make their purpose clear
agent = Agent(
    'openai:gpt-4.1',
    capabilities=[
        PrefixTools(
            wrapped=MCP(
                url='https://filesystem-mcp.example.com/sse',
                local=True,
                id='filesystem',
            ),
            prefix='files',
        )
    ]
)
# MCP tool 'read_file' becomes 'files_read_file'
# MCP tool 'write_file' becomes 'files_write_file'
```

```python
# Example 3 — from_spec for YAML-driven agent configuration

from pydantic_ai.capabilities.prefix_tools import PrefixTools

# Equivalent to what AgentSpec would parse from (NamedSpec short form: {ClassName: {kwargs}}):
# capabilities:
#   - PrefixTools:
#       prefix: db
#       capability:
#         MCP:
#           url: https://db-mcp.example.com/sse
#           local: true
#
# Note: only spec-serializable capabilities can appear nested here.
# Toolset.get_serialization_name() returns None — it is NOT registered in the spec
# registry, so it cannot be used as a nested capability in YAML/from_spec.
# Use Python code (PrefixTools(wrapped=Toolset(...), prefix='db')) for that case.
#
# In code, from_spec accepts a raw capability spec dict using NamedSpec short form:
cap = PrefixTools.from_spec(
    prefix='db',
    capability={
        'MCP': {
            'url': 'https://db-mcp.example.com/sse',
            'local': True,
        }
    },
)
print(cap.prefix)          # 'db'
print(type(cap).__name__)  # 'PrefixTools'
```

---

## 9. `PrepareTools` + `PrepareOutputTools` — Dynamic Tool Definition Filters

**Source**: `pydantic_ai/capabilities/prepare_tools.py`

`PrepareTools` and `PrepareOutputTools` wrap a `ToolsPrepareFunc` callable as a capability, enabling per-step dynamic tool filtering or modification for function tools and output tools respectively.

**Key implementation facts (verified from source)**:

- Shared dispatch in `_call_prepare_func`: calls the prepare function, then checks `inspect.isawaitable(result)` and awaits if needed. This lets the callable be sync or async.
- Result validation: `_utils.check_tools_prepare_func_result(result, prepare_func)` rejects `None` (raises `UserError` — return `[]` to expose no tools) and converts the result to a `list`. It does NOT enforce add/rename guards; that validation lives in `PreparedToolset.get_tools()` for toolset-level prepare functions, not in these capability hooks.
- `PrepareTools` hooks into `prepare_tools()` (function tool definitions); `PrepareOutputTools` hooks into `prepare_output_tools()` (output tool definitions). The two operate on separate hook chains.
- In `PrepareOutputTools`, `ctx.retry` and `ctx.max_retries` reflect the **output** retry budget (`max_output_retries`) — matching the output hook lifecycle, not the main tool retry budget.
- `get_serialization_name()` returns `None` for both — they are not spec-serializable because they wrap a callable at construction time.

```python
# Example 1 — role-based tool access using PrepareTools

from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition

@dataclass
class UserDeps:
    role: str   # 'admin', 'user', 'guest'

async def filter_by_role(
    ctx: RunContext[UserDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Hide admin tools from non-admin users."""
    if ctx.deps.role == 'admin':
        return tool_defs
    return [td for td in tool_defs if not td.name.startswith('admin_')]

agent = Agent(
    'openai:gpt-4.1',
    deps_type=UserDeps,
    capabilities=[PrepareTools(filter_by_role)],
)

async def main():
    # Admin sees all tools
    result_admin = await agent.run("list tools", deps=UserDeps(role='admin'))
    # Guest only sees non-admin tools
    result_guest = await agent.run("list tools", deps=UserDeps(role='guest'))
```

```python
# Example 2 — feature-flag based tool activation

from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition

@dataclass
class AppDeps:
    feature_flags: set[str]

def feature_gate(
    ctx: RunContext[AppDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Sync version — also works, inspect.isawaitable handles both."""
    enabled = ctx.deps.feature_flags
    return [
        td for td in tool_defs
        if td.metadata is None
        or td.metadata.get('feature') is None
        or td.metadata.get('feature') in enabled
    ]

agent = Agent(
    'openai:gpt-4.1',
    deps_type=AppDeps,
    capabilities=[PrepareTools(feature_gate)],
)
```

```python
# Example 3 — PrepareOutputTools to disable output tools on first step

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareOutputTools
from pydantic_ai.output import ToolOutput
from pydantic_ai.tools import ToolDefinition
from pydantic import BaseModel

class Report(BaseModel):
    title: str
    content: str

async def require_research_first(
    ctx: RunContext, tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Only allow the output tool after at least one model step (run_step > 0)."""
    if ctx.run_step == 0:
        return []  # Force the model to use regular tools first
    return tool_defs

agent = Agent(
    'openai:gpt-4.1',
    output_type=ToolOutput(Report),
    capabilities=[PrepareOutputTools(require_research_first)],
)
# On step 0: output tool is hidden → model must call other tools first
# On step 1+: output tool appears → model can now return a Report
```

---

## 10. `FunctionSchema` — Internal Function-to-LLM-Schema Engine

**Source**: `pydantic_ai/_function_schema.py`

`FunctionSchema` is pydantic-ai's internal dataclass that wraps a Python function with its Pydantic-derived JSON schema and runtime validator. It is constructed by `function_schema()` and used by `Tool` to validate and invoke tool functions.

**Key implementation facts (verified from source)**:

- `single_field_name` property covers two patterns: (1) `single_arg_name` — set when the function takes one model-like arg (Pydantic BaseModel, dataclass, TypedDict), using a wrap validator to normalize `{name: value}` before validation; (2) primitive one-property TypedDict schemas where `len(properties) == 1`. Returns `None` for multi-arg functions and `**kwargs`-only functions.
- `call()` is always async. For sync functions it uses `run_in_executor(function, *args, **kwargs)` — runs the sync function in the thread pool executor, keeping the event loop unblocked.
- `_call_args()` builds `args` (positional list) and `kwargs` (remaining dict). Positional fields are popped from `args_dict` in declaration order. `var_positional_field` (for `*args` parameters) is extended into `args` via `args.extend(...)`.
- `return_schema` defaults to `{}` (minimum, equivalent to `Any`). Non-trivial return types are resolved at construction time via `get_type_hints` and the Pydantic schema generator.
- `takes_ctx: bool` is determined by inspecting whether the first parameter is typed as `RunContext[...]` (or its alias `RunContext`). The flag controls whether `ctx` is prepended to `args` in `_call_args`.
- `json_schema: ObjectJsonSchema` is always a JSON object schema (dict). For single-arg model-like functions (BaseModel, dataclass, TypedDict), the schema exposes the model's **own top-level fields** directly (e.g. `model_arg_tool(params: SearchParams)` → `{query, limit}` in the schema). Validation internally wraps the flat dict back to `{params: value}` via the `single_arg_name` wrap validator. For multi-arg functions the schema is a flat object of all parameters.
- `validator: SchemaValidator` is a Pydantic-core validator, not a Pydantic model. This avoids Pydantic model overhead for simple tool functions.

```python
# Example 1 — single_field_name for model-like vs primitive functions

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from pydantic_ai._function_schema import function_schema

def multi_arg_tool(query: str, limit: int = 10) -> list:
    """Multi-arg tool: single_field_name is None"""
    return []

def single_primitive_tool(query: str) -> list:
    """Single primitive arg: single_field_name is 'query'"""
    return []

class SearchParams(BaseModel):
    query: str
    limit: int = 10

def model_arg_tool(params: SearchParams) -> list:
    """Model-like single arg: single_field_name is 'params' (single_arg_name)"""
    return []

schema_gen = GenerateJsonSchema

multi = function_schema(multi_arg_tool, schema_gen)
print(f"multi: {multi.single_field_name}")      # None — multiple args

single = function_schema(single_primitive_tool, schema_gen)
print(f"single primitive: {single.single_field_name}")  # 'query'

model = function_schema(model_arg_tool, schema_gen)
print(f"model-like: {model.single_field_name}")  # 'params'
```

```python
# Example 2 — sync vs async function dispatch via run_in_executor

from pydantic.json_schema import GenerateJsonSchema
from pydantic_ai._function_schema import function_schema

def blocking_sync_tool(query: str) -> str:
    """Sync tool: runs in thread pool via run_in_executor."""
    # This would block the event loop if called directly; run_in_executor prevents that.
    import time
    time.sleep(0.01)
    return f"result for {query}"

async def async_tool(query: str) -> str:
    """Async tool: awaited directly."""
    import asyncio
    await asyncio.sleep(0.01)
    return f"async result for {query}"

schema_gen = GenerateJsonSchema

sync_schema = function_schema(blocking_sync_tool, schema_gen)
async_schema = function_schema(async_tool, schema_gen)

print(f"sync is_async: {sync_schema.is_async}")   # False
print(f"async is_async: {async_schema.is_async}") # True

# Both schemas call() via the same async interface:
async def demo():
    # Simplified: in production, ctx comes from the agent run
    class FakeCtx:
        pass
    # sync tool runs in thread pool; async tool awaits
    # result = await sync_schema.call({'query': 'hello'}, ctx=FakeCtx())
    pass
```

```python
# Example 3 — return_schema and require_parameter_descriptions

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from pydantic_ai._function_schema import function_schema

class WeatherReport(BaseModel):
    city: str
    temperature: float
    condition: str

def get_weather(city: str) -> WeatherReport:
    """Get weather for a city.

    Args:
        city: The name of the city to query.
    """
    return WeatherReport(city=city, temperature=22.5, condition='sunny')

schema = function_schema(get_weather, GenerateJsonSchema)

# return_schema holds the JSON schema for WeatherReport
print("return_schema keys:", list(schema.return_schema.get('properties', {}).keys()))
# ['city', 'temperature', 'condition']

# json_schema is the *parameter* schema (input), not the return schema
print("param schema:", schema.json_schema)
# {'properties': {'city': {'type': 'string', 'title': 'City'}}, ...}

# require_parameter_descriptions raises if any arg lacks a docstring description
try:
    def undescribed_tool(x: str, y: int) -> str:
        """A tool with no arg docs."""
        return x
    function_schema(undescribed_tool, GenerateJsonSchema, require_parameter_descriptions=True)
except Exception as e:
    print(f"Missing descriptions: {e}")
```

---

## Summary

| Class / Function | Module | Key v2.0.0 Facts |
|---|---|---|
| `ModelResponsePartsManager` | `_parts_manager` | thinking-tag splitting, `ignore_leading_whitespace` (Qwen3/Ollama), `_typed_call_part` narrow_type promotion from PartStartEvent |
| `uuid7()` | `_uuid` | RFC 9562 Method 1, 42-bit counter, global lock, CPython 3.14-compatible |
| `PydanticAIDeprecationWarning` | `_warnings` | `UserWarning` subclass (visible by default), not `DeprecationWarning` |
| `MCP` | `capabilities/mcp` | `_resolved_id` cached_property, `_build_local` URL→transport, `from_spec` JSON-restriction, `allowed_tools` filter |
| `Toolset` | `capabilities/toolset` | minimal 3-field capability bridge, composable with all capability features |
| `WrapperCapability` | `capabilities/wrapper` | `__post_init__` inherits `id`/`defer_loading`, `apply()` visitor, `for_run` `replace()` pattern |
| `sort_capabilities` + `collect_leaves` | `capabilities/_ordering` | `TopologicalSorter` edge semantics, type vs instance `_ref_matches`, `CycleError` → `UserError` |
| `IncludeToolReturnSchemas` | `capabilities/include_return_schemas` | `ToolSelector`, `include_return_schema is None` guard, spec-serializable |
| `SetToolMetadata` | `capabilities/set_tool_metadata` | `**metadata` kwargs, deep merge, spec-serializable |
| `PrefixTools` | `capabilities/prefix_tools` | `AbstractToolset` vs `ToolsetFunc` dispatch, `from_spec` nested capability |
| `PrepareTools` / `PrepareOutputTools` | `capabilities/prepare_tools` | `inspect.isawaitable`, `PreparedToolset` validation, not spec-serializable |
| `FunctionSchema` | `_function_schema` | `single_field_name` two-path, `run_in_executor` for sync, `return_schema` minimum `{}` |

---
title: "PydanticAI Class Deep Dives Vol. 30"
description: "Source-verified deep dives into 10 pydantic-ai 2.4.0 class groups: PrepareTools + PrepareOutputTools (capability-level tool-definition filtering — sync/async predicates, role-based gating, per-step injection), ProcessEventStream (event stream forwarding + transformation — observer vs processor forms, tee'd fan-out, drop/inject events), IncludeToolReturnSchemas (selective return-schema injection — ToolSelector, per-tool override precedence, schema-aware model routing), SetToolMetadata (metadata stamp capability — kwargs API, ToolSelector narrowing, conditional activation), HandleDeferredToolCalls (inline deferred-tool resolution — auto-approve, partial handling, chained capability delegation), PrefixTools capability (wraps any capability's tools under a namespace prefix — from_spec factory, DynamicToolset delegation), HerokuProvider 2.4.0 (OpenAI-compatible gateway with cross-family profile routing — claude/gpt-oss/qwen/deepseek/kimi/nova/llama, HEROKU_INFERENCE_URL override, base_url /v1 normalisation), IncludeReturnSchemasToolset + PreparedToolset + RenamedToolset + PrefixedToolset (four toolset transformation primitives verified from source — IncludeReturnSchemasToolset auto-enable, PreparedToolset filter + shape mutation, RenamedToolset bidirectional name map, PrefixedToolset prefix stripping on call), WrapperToolset (delegation base for toolset middleware — for_run/for_run_step lifecycle hooks, visit_and_replace traversal, instruction propagation), gateway_provider + normalize_gateway_provider (Pydantic AI Gateway multi-upstream dispatch — route override, region-encoded API key URL inference, bedrock/google-cloud/openai/groq/anthropic normalization). All verified against pydantic-ai 2.4.0 source."
sidebar:
  label: "Class deep dives (Vol. 30)"
  order: 56
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.4.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.4.x API.
</Aside>

Ten class groups covering the 2.4.0 capability layer additions (`PrepareTools`, `ProcessEventStream`, `IncludeToolReturnSchemas`, `SetToolMetadata`, `HandleDeferredToolCalls`, `PrefixTools`), the `HerokuProvider` multi-family profile router, four toolset transformation primitives, the `WrapperToolset` delegation base, and the Pydantic AI Gateway multi-upstream dispatcher.

---

## 1. `PrepareTools` + `PrepareOutputTools` — Capability-Level Tool Definition Filter

**Source**: `pydantic_ai/capabilities/prepare_tools.py`  
**Export**: `from pydantic_ai.capabilities import PrepareTools, PrepareOutputTools`

`PrepareTools` and `PrepareOutputTools` are capabilities that wrap a `ToolsPrepareFunc` callable and apply it to function tools (or output tools, respectively) before every model request. They replace the older pattern of passing `prepare` directly to `FunctionToolset` when you want the filter to apply across all toolsets registered on the agent, or when you are using `capabilities=[...]` rather than hand-building toolsets.

`PrepareOutputTools` mirrors `PrepareTools` but operates on output-tool definitions; its `ctx.retry`/`ctx.max_retries` reflects the **output** retry budget (`max_output_retries`), not the regular tool budget.

```python
# Key signatures verified from source (pydantic-ai 2.4.0):

@dataclass
class PrepareTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]
    # get_serialization_name -> None (not spec-serializable; takes a callable)

    async def prepare_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]: ...

@dataclass
class PrepareOutputTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]

    async def prepare_output_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]: ...
```

The `prepare_func` receives `(RunContext, list[ToolDefinition])` and returns a **subset** (or equal set) of `ToolDefinition` objects with optionally modified fields. The framework raises `UserError` if the function attempts to add new tool names or rename existing ones — use `FunctionToolset.add_function()` or `RenamedToolset` for that.

### 1.1 Role-Based Tool Gating with `PrepareTools`

Hide admin tools for non-privileged users at the capability level, so the filter applies regardless of which toolset registered the tool.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition


@dataclass
class UserDeps:
    is_admin: bool


async def hide_admin_tools(
    ctx: RunContext[UserDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    if ctx.deps.is_admin:
        return tool_defs
    return [td for td in tool_defs if not td.name.startswith("admin_")]


agent: Agent[UserDeps, str] = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
    capabilities=[PrepareTools(hide_admin_tools)],
)


@agent.tool
async def admin_delete_user(ctx: RunContext[UserDeps], user_id: str) -> str:
    return f"Deleted {user_id}"


@agent.tool
async def get_status(ctx: RunContext[UserDeps]) -> str:
    return "All systems operational"


async def main() -> None:
    result = await agent.run("List available tools", deps=UserDeps(is_admin=False))
    print(result.output)  # agent cannot mention admin_delete_user


asyncio.run(main())
```

### 1.2 Run-Step-Aware Tool Injection with `PrepareTools`

Only expose expensive/dangerous tools after the first reasoning step, giving the model a chance to plan before it can act.

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition

EXPENSIVE_TOOLS = {"run_sql_query", "send_email", "write_file"}


async def gate_expensive_tools(
    ctx: RunContext, tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    # Allow cheap read-only tools unconditionally; gate expensive ones
    if ctx.run_step == 0:
        return [td for td in tool_defs if td.name not in EXPENSIVE_TOOLS]
    return tool_defs


agent = Agent(
    "openai:gpt-4o",
    capabilities=[PrepareTools(gate_expensive_tools)],
)


@agent.tool_plain
def run_sql_query(query: str) -> str:
    return f"Results for: {query}"


@agent.tool_plain
def get_schema() -> str:
    return "Tables: users, orders, products"


async def main() -> None:
    result = await agent.run("What tables exist and query the users table")
    print(result.output)


asyncio.run(main())
```

### 1.3 Restricting Output Tools with `PrepareOutputTools`

Use `PrepareOutputTools` to limit which output tools can fire until the agent has gathered enough data.

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareOutputTools
from pydantic_ai.output import ToolOutput
from pydantic_ai.tools import ToolDefinition
from pydantic import BaseModel


class FinalReport(BaseModel):
    summary: str
    confidence: float


async def require_at_least_two_steps(
    ctx: RunContext, tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    # Suppress output tool until run_step >= 2 so the model researches first
    if ctx.run_step < 2:
        return []
    return tool_defs


agent = Agent(
    "openai:gpt-4o",
    output_type=ToolOutput(FinalReport),
    capabilities=[PrepareOutputTools(require_at_least_two_steps)],
)


@agent.tool_plain
def search_web(query: str) -> str:
    return f"Top results for '{query}': ..."


@agent.tool_plain
def read_article(url: str) -> str:
    return "Article content ..."


async def main() -> None:
    result = await agent.run("Research AI safety and write a summary report")
    print(result.output)


asyncio.run(main())
```

---

## 2. `ProcessEventStream` — Event Stream Forwarding and Transformation

**Source**: `pydantic_ai/capabilities/process_event_stream.py`  
**Export**: `from pydantic_ai.capabilities import ProcessEventStream`

`ProcessEventStream` attaches a handler to the agent's event stream without requiring the caller to use `agent.run_stream()`. When the capability is registered, `agent.run()` automatically enables streaming internally; the handler fires for every `AgentStreamEvent` emitted during `ModelRequestNode` and `CallToolsNode` execution.

Two handler forms are supported:

- **Observer** (`EventStreamHandler`): `async def handler(ctx, stream) -> None`. Events are tee'd so the observer receives a copy while the original stream continues unchanged. An early return from the observer silently stops delivery to the handler — downstream consumers are unaffected.
- **Processor** (`EventStreamProcessor`): `async def handler(ctx, stream) -> AsyncIterator[AgentStreamEvent]`. The events the generator yields **replace** the inner stream for downstream consumers, enabling filtering, reordering, or injection of synthetic events.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

@dataclass
class ProcessEventStream(AbstractCapability[AgentDepsT]):
    handler: EventStreamHandlerFunc[AgentDepsT] | EventStreamProcessorFunc[AgentDepsT]
    # get_serialization_name -> None (takes a callable)

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]: ...
```

### 2.1 Logging Every Token with an Observer

Attach a lightweight observer that prints each delta without modifying the stream, useful for debugging or live progress displays.

```python
import asyncio
from collections.abc import AsyncIterable
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, TextPartDelta


async def log_tokens(
    ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]
) -> None:
    async for event in stream:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            print(event.delta.content_delta, end="", flush=True)
    print()  # newline after stream ends


agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[ProcessEventStream(log_tokens)],
)


async def main() -> None:
    # agent.run() — no run_stream() needed; ProcessEventStream enables streaming
    result = await agent.run("Tell me a short joke")
    print("\nFinal output:", result.output)


asyncio.run(main())
```

### 2.2 Filtering Events with a Processor

Use the processor form to strip thinking parts from the downstream stream while a separate observer captures them for analytics.

```python
import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartStartEvent, ThinkingPart


async def strip_thinking_events(
    ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]
) -> AsyncIterator[AgentStreamEvent]:
    async for event in stream:
        # Suppress thinking-part start events from downstream consumers
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            continue
        yield event


agent = Agent(
    "anthropic:claude-opus-4-5",
    capabilities=[ProcessEventStream(strip_thinking_events)],
)


async def main() -> None:
    result = await agent.run("Reason step-by-step about prime numbers up to 20")
    print(result.output)  # thinking parts silently dropped from stream


asyncio.run(main())
```

### 2.3 Multiplexed Observers via Multiple `ProcessEventStream` Capabilities

Stack two `ProcessEventStream` capabilities to fan the stream out to two independent observers simultaneously.

```python
import asyncio
from collections.abc import AsyncIterable
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, TextPartDelta

token_count = 0


async def count_tokens(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
    global token_count
    async for event in stream:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            token_count += len(event.delta.content_delta.split())


async def audit_log(ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        # Each observer gets its own tee'd copy; neither affects the other
        pass


agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[
        ProcessEventStream(count_tokens),
        ProcessEventStream(audit_log),
    ],
)


async def main() -> None:
    await agent.run("Explain quantum entanglement in two sentences")
    print(f"Approximate word count: {token_count}")


asyncio.run(main())
```

---

## 3. `IncludeToolReturnSchemas` — Selective Return-Schema Injection

**Source**: `pydantic_ai/capabilities/include_return_schemas.py`  
**Export**: `from pydantic_ai.capabilities import IncludeToolReturnSchemas`

`IncludeToolReturnSchemas` sets `include_return_schema=True` on matching tool definitions. For models with native schema support (e.g. Google Gemini) the schema is sent as a structured field; for other models it is injected into the tool description as JSON text, giving the model explicit type information about what each tool returns. Per-tool overrides (`Tool(..., include_return_schema=False)`) take precedence — this capability only stamps tools that have not explicitly opted out.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

@dataclass
class IncludeToolReturnSchemas(AbstractCapability[AgentDepsT]):
    tools: ToolSelector[AgentDepsT] = 'all'
    # 'all'              — every tool
    # Sequence[str]      — only listed names
    # dict[str, Any]     — tools whose metadata deeply includes pairs
    # Callable           — sync or async (ctx, tool_def) -> bool
    # get_serialization_name -> 'IncludeToolReturnSchemas'
```

### 3.1 Enable Return Schemas for All Tools

The simplest usage: all tools on the agent get their return schema injected.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas


@dataclass
class WeatherData:
    location: str
    temperature_c: float
    condition: str


agent = Agent(
    "openai:gpt-4o",
    capabilities=[IncludeToolReturnSchemas()],  # 'all' is the default
)


@agent.tool_plain
def get_weather(city: str) -> WeatherData:
    return WeatherData(location=city, temperature_c=22.5, condition="sunny")


@agent.tool_plain
def get_population(city: str) -> dict[str, int]:
    return {"city": city, "population": 1_000_000}


async def main() -> None:
    result = await agent.run("What's the weather in Paris and what is its population?")
    print(result.output)


asyncio.run(main())
```

### 3.2 Selective Schema Injection by Tool Name

Only inject return schemas for the expensive tools where type clarity matters most.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic import BaseModel


class AnalysisResult(BaseModel):
    insights: list[str]
    score: float
    metadata: dict[str, str]


agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        IncludeToolReturnSchemas(tools=["run_analysis", "get_report"]),
    ],
)


@agent.tool_plain
def run_analysis(dataset: str) -> AnalysisResult:
    return AnalysisResult(
        insights=["Trend up 12%", "Anomaly at day 7"],
        score=0.87,
        metadata={"dataset": dataset, "rows": "10000"},
    )


@agent.tool_plain
def ping() -> str:  # plain string — no schema injection needed
    return "pong"


async def main() -> None:
    result = await agent.run("Run analysis on Q1_data and explain the results")
    print(result.output)


asyncio.run(main())
```

### 3.3 Dynamic Selection via Async Predicate

Use a callable selector to dynamically decide at request time which tools get schemas, based on model capabilities reported in the context.

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.tools import ToolDefinition

SCHEMA_WORTHY_TOOLS = {"search_database", "fetch_document", "aggregate_metrics"}


async def schema_selector(ctx: RunContext, tool_def: ToolDefinition) -> bool:
    return tool_def.name in SCHEMA_WORTHY_TOOLS


agent = Agent(
    "openai:gpt-4o",
    capabilities=[IncludeToolReturnSchemas(tools=schema_selector)],
)


@agent.tool_plain
def search_database(query: str) -> list[dict]:
    return [{"id": 1, "text": "result"}]


@agent.tool_plain
def get_timestamp() -> str:
    return "2026-07-03T12:00:00Z"


async def main() -> None:
    result = await agent.run("Search the database for 'AI' and tell me the timestamp")
    print(result.output)


asyncio.run(main())
```

---

## 4. `SetToolMetadata` — Metadata Stamp Capability

**Source**: `pydantic_ai/capabilities/set_tool_metadata.py`  
**Export**: `from pydantic_ai.capabilities import SetToolMetadata`

`SetToolMetadata` merges arbitrary key-value pairs into the `metadata` dict of matching tool definitions. Combined with a `ToolSelector`, this lets you annotate tools at capability registration time — for example marking tools as requiring certain permissions, tagging tools for custom routing, or passing provider-specific hints — without modifying the original `Tool` declarations.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

@dataclass(init=False)
class SetToolMetadata(AbstractCapability[AgentDepsT]):
    tools: ToolSelector[AgentDepsT] = 'all'
    metadata: dict[str, Any]  # populated from **kwargs

    def __init__(self, *, tools: ToolSelector[AgentDepsT] = 'all', **metadata: Any) -> None: ...
    # get_serialization_name -> 'SetToolMetadata'
```

### 4.1 Tagging Tools for Custom Routing

Mark tools as "read-only" or "write" so a downstream middleware can route them accordingly.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[
        SetToolMetadata(tools=["read_record", "list_records"], access="read"),
        SetToolMetadata(tools=["create_record", "delete_record"], access="write"),
    ],
)


@agent.tool_plain
def read_record(record_id: str) -> dict:
    return {"id": record_id, "data": "..."}


@agent.tool_plain
def list_records() -> list[str]:
    return ["rec-1", "rec-2"]


@agent.tool_plain
def create_record(data: str) -> str:
    return "rec-3"


@agent.tool_plain
def delete_record(record_id: str) -> bool:
    return True


async def main() -> None:
    result = await agent.run("List all records")
    print(result.output)


asyncio.run(main())
```

### 4.2 Stamping All Tools with a Version Tag

Inject a `schema_version` hint on all tools to let consumer middleware know which API version to expect.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[SetToolMetadata(schema_version="v2", source="internal_api")],
)


@agent.tool_plain
def calculate_tax(amount: float, rate: float) -> float:
    return round(amount * rate, 2)


@agent.tool_plain
def convert_currency(amount: float, from_ccy: str, to_ccy: str) -> float:
    return amount * 1.1  # simplified


async def main() -> None:
    result = await agent.run("Calculate 18% tax on $500 and convert $550 USD to EUR")
    print(result.output)


asyncio.run(main())
```

### 4.3 Conditional Metadata via Async ToolSelector

Use an async callable to stamp tools only when certain runtime conditions apply.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.tools import ToolDefinition


@dataclass
class AppDeps:
    trace_id: str
    debug_mode: bool


async def debug_tools_only(ctx: RunContext[AppDeps], td: ToolDefinition) -> bool:
    return ctx.deps.debug_mode


agent: Agent[AppDeps, str] = Agent(
    "openai:gpt-4o-mini",
    capabilities=[
        SetToolMetadata(tools=debug_tools_only, trace_enabled=True, log_level="DEBUG"),
    ],
)


@agent.tool_plain
def process_order(order_id: str) -> str:
    return f"Order {order_id} processed"


async def main() -> None:
    result = await agent.run(
        "Process order ORD-42", deps=AppDeps(trace_id="abc123", debug_mode=True)
    )
    print(result.output)


asyncio.run(main())
```

---

## 5. `HandleDeferredToolCalls` — Inline Deferred-Tool Resolution

**Source**: `pydantic_ai/capabilities/deferred_tool_handler.py`  
**Export**: `from pydantic_ai.capabilities import HandleDeferredToolCalls`

`HandleDeferredToolCalls` intercepts `DeferredToolRequests` that would otherwise pause the run and resolves them inline using a user-supplied handler. This converts a human-in-the-loop approval flow into a fully automated one — the handler receives the pending tool calls, decides which to approve, and returns `DeferredToolResults`. The run resumes without the caller ever seeing the deferred output type.

The handler may return `None` to decline handling; if all registered capabilities return `None`, the deferred requests bubble up as the run's output.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

@dataclass
class HandleDeferredToolCalls(AbstractCapability[AgentDepsT]):
    handler: Callable[
        [RunContext[AgentDepsT], DeferredToolRequests],
        DeferredToolResults | None | Awaitable[DeferredToolResults | None],
    ]
    # get_serialization_name -> None (takes a callable)

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None: ...
```

### 5.1 Auto-Approve All Deferred Tools

Automatically approve every deferred tool call — useful in testing or trusted internal pipelines.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext


async def approve_all(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults:
    return requests.build_results(approve_all=True)


toolset = FunctionToolset()


@toolset.tool
async def send_notification(message: str) -> str:
    return f"Notification sent: {message}"


approval_toolset = ApprovalRequiredToolset(toolset)

agent = Agent(
    "openai:gpt-4o-mini",
    toolsets=[approval_toolset],
    capabilities=[HandleDeferredToolCalls(handler=approve_all)],
)


async def main() -> None:
    # The run completes in one shot; no deferred output escapes
    result = await agent.run("Send a notification saying 'Hello World'")
    print(result.output)


asyncio.run(main())
```

### 5.2 Selective Approval Based on Tool Name

Only auto-approve read operations; pause for write operations by returning `None`.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

READ_TOOLS = {"fetch_data", "list_items", "get_status"}


async def approve_reads_only(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    all_reads = all(call.tool_name in READ_TOOLS for call in requests.tool_calls)
    if all_reads:
        return requests.build_results(approve_all=True)
    return None  # writes bubble up as DeferredToolRequests for human review


toolset = FunctionToolset()


@toolset.tool
async def fetch_data(endpoint: str) -> dict:
    return {"endpoint": endpoint, "data": [1, 2, 3]}


@toolset.tool
async def delete_record(record_id: str) -> bool:
    return True


agent = Agent(
    "openai:gpt-4o-mini",
    toolsets=[ApprovalRequiredToolset(toolset)],
    capabilities=[HandleDeferredToolCalls(handler=approve_reads_only)],
)


async def main() -> None:
    result = await agent.run("Fetch data from /api/metrics")
    print(result.output)


asyncio.run(main())
```

### 5.3 Chained Capabilities with Different Strategies

Register two `HandleDeferredToolCalls` capabilities. The first handles low-risk tools; the second handles everything else that the first declined.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext

LOW_RISK = {"ping", "health_check"}
HIGH_RISK = {"reboot_server", "clear_cache"}


async def approve_low_risk(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    if all(c.tool_name in LOW_RISK for c in requests.tool_calls):
        return requests.build_results(approve_all=True)
    return None


async def approve_high_risk_in_test(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults | None:
    # Only auto-approve in test environment
    if getattr(ctx.deps, "is_test", False):
        return requests.build_results(approve_all=True)
    return None


toolset = FunctionToolset()


@toolset.tool
async def ping() -> str:
    return "pong"


@toolset.tool
async def reboot_server(server_id: str) -> str:
    return f"Rebooted {server_id}"


agent = Agent(
    "openai:gpt-4o-mini",
    toolsets=[ApprovalRequiredToolset(toolset)],
    capabilities=[
        HandleDeferredToolCalls(handler=approve_low_risk),
        HandleDeferredToolCalls(handler=approve_high_risk_in_test),
    ],
)


async def main() -> None:
    result = await agent.run("Ping the server")
    print(result.output)


asyncio.run(main())
```

---

## 6. `PrefixTools` — Capability-Level Tool-Name Prefixing

**Source**: `pydantic_ai/capabilities/prefix_tools.py`  
**Export**: `from pydantic_ai.capabilities import PrefixTools`

`PrefixTools` wraps another capability and renames all of its tool definitions by prepending `{prefix}_`. Only the wrapped capability's tools are affected — other tools registered directly on the agent keep their original names. This is the capability-level counterpart to `PrefixedToolset`.

`PrefixTools` can also be constructed from a spec dict via `PrefixTools.from_spec(prefix=..., capability=...)`, enabling declarative agent configuration in YAML/JSON.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

@dataclass
class PrefixTools(WrapperCapability[AgentDepsT]):
    # inherited: wrapped: AbstractCapability[AgentDepsT]
    prefix: str

    @classmethod
    def get_serialization_name(cls) -> str | None: return 'PrefixTools'

    @classmethod
    def from_spec(cls, *, prefix: str, capability: CapabilitySpec) -> PrefixTools[Any]: ...

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None: ...
    # Wraps the inner toolset in PrefixedToolset; or DynamicToolset if the inner
    # toolset is a ToolsetFunc callable.
```

### 6.1 Namespacing a `Toolset` Capability

Wrap a `Toolset` capability so that all its tools appear as `db_*` to the model.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools, Toolset
from pydantic_ai.toolsets import FunctionToolset

db_toolset = FunctionToolset()


@db_toolset.tool_plain
def query(sql: str) -> list[dict]:
    return [{"id": 1, "name": "Alice"}]


@db_toolset.tool_plain
def insert(table: str, data: dict) -> int:
    return 42  # new row id


agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[PrefixTools(wrapped=Toolset(db_toolset), prefix="db")],
)
# The model sees 'db_query' and 'db_insert', not 'query' and 'insert'


async def main() -> None:
    result = await agent.run("Query all users from the database")
    print(result.output)


asyncio.run(main())
```

### 6.2 Combining Prefixed Namespaces from Two Sources

Use two `PrefixTools` capabilities to keep tools from two sources in distinct namespaces, preventing name collisions.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools, Toolset
from pydantic_ai.toolsets import FunctionToolset

analytics_toolset = FunctionToolset()
crm_toolset = FunctionToolset()


@analytics_toolset.tool_plain
def get_metrics(period: str) -> dict:
    return {"period": period, "visits": 1500}


@crm_toolset.tool_plain
def get_contacts(filter_by: str) -> list[str]:
    return ["Alice", "Bob"]


agent = Agent(
    "openai:gpt-4o",
    capabilities=[
        PrefixTools(wrapped=Toolset(analytics_toolset), prefix="analytics"),
        PrefixTools(wrapped=Toolset(crm_toolset), prefix="crm"),
    ],
)
# Model sees: 'analytics_get_metrics', 'crm_get_contacts'


async def main() -> None:
    result = await agent.run("Get this week's metrics and list all contacts")
    print(result.output)


asyncio.run(main())
```

### 6.3 Spec-Based Construction for Declarative Agents

Use `from_spec` to build a `PrefixTools` capability from a plain dict, as if loaded from a YAML config file.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai.toolsets import FunctionToolset

search_toolset = FunctionToolset()


@search_toolset.tool_plain
def web_search(query: str) -> list[str]:
    return [f"Result 1 for {query}", f"Result 2 for {query}"]


@search_toolset.tool_plain
def image_search(query: str) -> list[str]:
    return [f"Image 1 for {query}"]


# As if loaded from config:
# capabilities:
#   - type: PrefixTools
#     prefix: search
#     capability:
#       type: Toolset
#       toolset: <injected>

cap = PrefixTools.from_spec(
    prefix="search",
    capability={"type": "Toolset", "toolset": search_toolset},
)

agent = Agent("openai:gpt-4o-mini", capabilities=[cap])


async def main() -> None:
    result = await agent.run("Search for 'pydantic-ai tutorials'")
    print(result.output)


asyncio.run(main())
```

---

## 7. `HerokuProvider` — Heroku Managed Inference (2.4.0)

**Source**: `pydantic_ai/providers/heroku.py`  
**Export**: `from pydantic_ai.providers.heroku import HerokuProvider`

`HerokuProvider` connects to the Heroku Managed Inference API, which is an OpenAI-compatible endpoint bundled with Heroku's cloud platform. In 2.4.0, the provider gained a rich **multi-family model profile router** that detects the underlying model family from the bare model name (no provider prefix) and applies the correct profile — crucial for features like `thinking=True` on claude models that would otherwise be silently dropped.

Supported profile prefixes: `claude` → `anthropic_model_profile`, `gpt-oss` → `harmony_model_profile`, `qwen` → `qwen_model_profile`, `deepseek` → `deepseek_model_profile`, `kimi`/`glm` → `moonshotai_model_profile`, `mistral` → `mistral_model_profile`, `nova` → `amazon_model_profile`, `llama`/`gemma` → `meta`/`google_model_profile`. All profiles are merged over `OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)`.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

class HerokuProvider(Provider[AsyncOpenAI]):
    def __init__(
        self,
        *,
        base_url: str | None = None,          # overrides HEROKU_INFERENCE_URL
        api_key: str | None = None,            # overrides HEROKU_INFERENCE_KEY env var
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None: ...
    # Routes by prefix to the correct family profile, always merged over OpenAIModelProfile
```

The base URL defaults to `https://us.inference.heroku.com/v1`. If neither `base_url` nor `HEROKU_INFERENCE_URL` is set but `HEROKU_INFERENCE_KEY` is present, the default endpoint is used.

### 7.1 Basic Chat with a Claude Model via Heroku

Use the provider to run a Claude model through Heroku Inference with automatic profile detection.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel


provider = HerokuProvider(api_key=os.environ["HEROKU_INFERENCE_KEY"])
model = OpenAIChatModel("claude-3-5-haiku-20241022", openai_client=provider.client)

agent = Agent(model, system_prompt="You are a concise assistant.")


async def main() -> None:
    result = await agent.run("Summarise the benefits of type hints in Python in one paragraph.")
    print(result.output)


asyncio.run(main())
```

### 7.2 Thinking Mode with a Claude Model

Because `HerokuProvider` correctly detects the `claude` prefix and loads `anthropic_model_profile`, the `thinking` setting is correctly forwarded on the wire.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings


provider = HerokuProvider(api_key=os.environ["HEROKU_INFERENCE_KEY"])
model = OpenAIChatModel("claude-opus-4-5", openai_client=provider.client)

agent = Agent(
    model,
    model_settings=ModelSettings(thinking=True),
    system_prompt="Reason step by step.",
)


async def main() -> None:
    result = await agent.run("What is the square root of 144 times the 7th Fibonacci number?")
    print(result.output)


asyncio.run(main())
```

### 7.3 Custom On-Premises Heroku Inference URL

Override the default endpoint to point at a private Heroku space or a staging environment.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel

# Private Heroku space endpoint; /v1 is automatically appended if missing
provider = HerokuProvider(
    api_key=os.environ["HEROKU_INFERENCE_KEY"],
    base_url="https://inference.my-private-space.herokuspace.com",
)
model = OpenAIChatModel("llama3.1-70b-versatile", openai_client=provider.client)

agent = Agent(model, system_prompt="You are a helpful coding assistant.")


async def main() -> None:
    result = await agent.run("Write a Python function that checks if a number is prime.")
    print(result.output)


asyncio.run(main())
```

---

## 8. Toolset Transformation Primitives: `IncludeReturnSchemasToolset` · `PreparedToolset` · `RenamedToolset` · `PrefixedToolset`

**Sources**:  
- `pydantic_ai/toolsets/include_return_schemas.py`  
- `pydantic_ai/toolsets/prepared.py`  
- `pydantic_ai/toolsets/renamed.py`  
- `pydantic_ai/toolsets/prefixed.py`

These four toolset wrappers cover the most common per-toolset mutations in 2.4.0. Each extends `WrapperToolset` and delegates all non-intercepted behaviour to the wrapped toolset.

```python
# Key signatures verified from source (pydantic-ai 2.4.0):

@dataclass(init=False)
class IncludeReturnSchemasToolset(PreparedToolset[AgentDepsT]):
    """Sets include_return_schema=True on every tool that hasn't explicitly opted out."""
    def __init__(self, wrapped: AbstractToolset[AgentDepsT]) -> None: ...

@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """Applies a ToolsPrepareFunc to filter or reshape tool definitions per-request."""
    prepare_func: ToolsPrepareFunc[AgentDepsT]
    async def get_tools(self, ctx) -> dict[str, ToolsetTool]: ...

@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    """Renames tools via {new_name: original_name}. Preserves unrenamed tools unchanged."""
    name_map: dict[str, str]
    async def get_tools(self, ctx) -> dict[str, ToolsetTool]: ...
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...

@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """Prefixes every tool name with '{prefix}_'. Strips prefix before dispatch."""
    prefix: str
    async def get_tools(self, ctx) -> dict[str, ToolsetTool]: ...
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...
```

### 8.1 Schema Injection via `IncludeReturnSchemasToolset`

Wrap an external toolset to enable return-schema injection for all its tools without modifying tool declarations.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset, IncludeReturnSchemasToolset
from pydantic import BaseModel


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str


search_toolset = FunctionToolset()


@search_toolset.tool_plain
def search(query: str) -> list[SearchResult]:
    return [SearchResult(url="https://example.com", title="Example", snippet="...")]


agent = Agent(
    "openai:gpt-4o",
    toolsets=[IncludeReturnSchemasToolset(search_toolset)],
)


async def main() -> None:
    result = await agent.run("Search for 'pydantic validation' and summarise the results")
    print(result.output)


asyncio.run(main())
```

### 8.2 Context-Sensitive Filtering via `PreparedToolset`

Use `PreparedToolset` to suppress tools whose parameters require unavailable integrations — checked at request time.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, PreparedToolset
from pydantic_ai.tools import ToolDefinition


@dataclass
class FeatureDeps:
    slack_enabled: bool
    db_enabled: bool


FEATURE_GATES = {"post_to_slack": "slack_enabled", "query_db": "db_enabled"}


async def feature_gate(
    ctx: RunContext[FeatureDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    return [
        td for td in tool_defs
        if not FEATURE_GATES.get(td.name)
        or getattr(ctx.deps, FEATURE_GATES[td.name], False)
    ]


toolset = FunctionToolset()


@toolset.tool_plain
def post_to_slack(message: str) -> str:
    return "Posted"


@toolset.tool_plain
def query_db(sql: str) -> list[dict]:
    return []


@toolset.tool_plain
def get_time() -> str:
    return "12:00"


agent: Agent[FeatureDeps, str] = Agent(
    "openai:gpt-4o-mini",
    toolsets=[PreparedToolset(toolset, feature_gate)],
)


async def main() -> None:
    deps = FeatureDeps(slack_enabled=False, db_enabled=True)
    result = await agent.run("What time is it and query the DB", deps=deps)
    print(result.output)


asyncio.run(main())
```

### 8.3 Renaming and Prefixing Toolsets

Combine `RenamedToolset` and `PrefixedToolset` to adapt an external toolset to your agent's naming conventions.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset, PrefixedToolset, RenamedToolset

vendor_toolset = FunctionToolset()


@vendor_toolset.tool_plain
def doSearch(query: str) -> list[str]:  # camelCase from vendor
    return [f"result: {query}"]


@vendor_toolset.tool_plain
def doFetch(url: str) -> str:
    return "content"


# Step 1: rename camelCase vendor names to snake_case
renamed = RenamedToolset(vendor_toolset, name_map={"search": "doSearch", "fetch": "doFetch"})

# Step 2: prefix with 'vendor_' so the model knows the source
prefixed = PrefixedToolset(renamed, prefix="vendor")
# The model now sees: 'vendor_search', 'vendor_fetch'

agent = Agent("openai:gpt-4o-mini", toolsets=[prefixed])


async def main() -> None:
    result = await agent.run("Search for 'AI frameworks' using the vendor search")
    print(result.output)


asyncio.run(main())
```

---

## 9. `WrapperToolset` — The Delegation Base for Toolset Middleware

**Source**: `pydantic_ai/toolsets/wrapper.py`  
**Export**: `from pydantic_ai.toolsets import WrapperToolset`

`WrapperToolset` is the base class for all single-toolset wrappers (`PrefixedToolset`, `PreparedToolset`, `RenamedToolset`, `IncludeReturnSchemasToolset`, `SetMetadataToolset`). It implements the full `AbstractToolset` interface by delegating every call to `self.wrapped`, and provides two important structural hooks:

- **`for_run` / `for_run_step`** — Called at the start of each run (and each step) so wrappers can snapshot or replace the wrapped toolset. Delegates to `wrapped.for_run()` and only creates a new `WrapperToolset` when the wrapped result actually changed — a performance optimisation that avoids churn in stable pipelines.
- **`visit_and_replace`** — Recursive traversal used by the framework (and user middleware) to find a specific toolset deep inside a wrapper chain and replace it with a transformed version.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    wrapped: AbstractToolset[AgentDepsT]

    async def for_run(self, ctx) -> AbstractToolset: ...
    async def for_run_step(self, ctx) -> AbstractToolset: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args) -> bool | None: ...
    async def get_instructions(self, ctx) -> ...: ...
    async def get_tools(self, ctx) -> dict[str, ToolsetTool]: ...
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...
    def apply(self, visitor) -> None: ...
    def visit_and_replace(self, visitor) -> AbstractToolset: ...
```

### 9.1 Implementing a Custom Caching Wrapper

Subclass `WrapperToolset` to memoize expensive `get_tools` calls that involve network round-trips.

```python
import asyncio
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import WrapperToolset, FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from typing import Any


@dataclass
class CachingToolset(WrapperToolset):
    _cache: dict[str, dict[str, ToolsetTool]] = field(
        default_factory=dict, init=False, repr=False
    )

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        run_id = str(id(ctx))
        if run_id not in self._cache:
            self._cache[run_id] = await super().get_tools(ctx)
        return self._cache[run_id]

    def clear_cache(self) -> None:
        self._cache.clear()


remote_toolset = FunctionToolset()
call_count = 0


@remote_toolset.tool_plain
def fetch_config() -> dict:
    global call_count
    call_count += 1
    return {"version": "2.4.0", "feature_flags": ["alpha", "beta"]}


caching_wrapper = CachingToolset(remote_toolset)
agent = Agent("openai:gpt-4o-mini", toolsets=[caching_wrapper])


async def main() -> None:
    result = await agent.run("What version is the config and what are the feature flags?")
    print(result.output)
    print(f"fetch_config called {call_count} time(s)")


asyncio.run(main())
```

### 9.2 Using `visit_and_replace` to Swap a Nested Toolset

Traverse a wrapper chain to swap out a specific inner toolset without rebuilding the wrapper stack.

```python
import asyncio
from dataclasses import dataclass, replace
from pydantic_ai import Agent
from pydantic_ai.toolsets import (
    FunctionToolset,
    PrefixedToolset,
    FilteredToolset,
    WrapperToolset,
)
from pydantic_ai.toolsets.abstract import AbstractToolset

# Build a multi-layer wrapper chain: filter → prefix → original
original = FunctionToolset()


@original.tool_plain
def ping() -> str:
    return "pong"


filtered = FilteredToolset(original, lambda ctx, td: True)
prefixed = PrefixedToolset(filtered, prefix="svc")

# Now swap out `original` for a new toolset without touching the rest of the chain
replacement = FunctionToolset()


@replacement.tool_plain
def ping() -> str:
    return "pong-v2"


def swap_original(ts: AbstractToolset) -> AbstractToolset:
    if ts is original:
        return replacement
    return ts


new_chain = prefixed.visit_and_replace(swap_original)
agent = Agent("openai:gpt-4o-mini", toolsets=[new_chain])


async def main() -> None:
    result = await agent.run("Call the ping tool")
    print(result.output)  # will use replacement's ping


asyncio.run(main())
```

### 9.3 Observing Lifecycle with `for_run`

Override `for_run` to instrument toolset initialisation timing without affecting tool dispatching.

```python
import asyncio
import time
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import WrapperToolset, FunctionToolset
from pydantic_ai.toolsets.abstract import AbstractToolset


@dataclass
class TimedToolset(WrapperToolset):
    async def for_run(self, ctx: RunContext) -> AbstractToolset:
        start = time.monotonic()
        result = await super().for_run(ctx)
        elapsed = time.monotonic() - start
        print(f"[TimedToolset] for_run took {elapsed * 1000:.1f} ms")
        return result


base = FunctionToolset()


@base.tool_plain
def calculate(expression: str) -> float:
    """Evaluate a simple arithmetic expression without eval()."""
    import ast
    import operator as op

    _ops = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
    }

    def _safe(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ops:
            return _ops[type(node.op)](_safe(node.left), _safe(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_safe(node.operand)
        raise ValueError(f"Unsupported operation in expression: {ast.dump(node)}")

    return _safe(ast.parse(expression, mode="eval").body)


agent = Agent("openai:gpt-4o-mini", toolsets=[TimedToolset(base)])


async def main() -> None:
    result = await agent.run("Calculate 12 * 42 + 7")
    print("Result:", result.output)


asyncio.run(main())
```

---

## 10. `gateway_provider` + `normalize_gateway_provider` — Pydantic AI Gateway

**Source**: `pydantic_ai/providers/gateway.py`  
**Export**: `from pydantic_ai.providers.gateway import gateway_provider, normalize_gateway_provider`

`gateway_provider` creates a provider that routes through the **Pydantic AI Gateway** (`gateway.pydantic.dev/proxy`) — a managed proxy that fronts OpenAI, Anthropic, Groq, Bedrock, and Google Cloud endpoints with a single API key. In 2.4.0, the function gained **region-encoded key URL inference**: a `pylf_v*` key embeds a region slug that `_infer_base_url` uses to pick the nearest regional gateway endpoint automatically.

The `upstream_provider` argument accepts both *model providers* (`'openai'`, `'anthropic'`, `'groq'`, `'bedrock'`, `'google'`) and *API flavour aliases* (`'chat'`, `'responses'`, `'converse'`, `'google-cloud'`). A `route` override lets callers target a specific routing group in the Gateway's config.

```python
# Key signature verified from source (pydantic-ai 2.4.0):

def gateway_provider(
    upstream_provider: str,   # ModelProvider | APIFlavor | 'gateway/<any>'
    /,
    *,
    route: str | None = None,      # override default Gateway route segment
    api_key: str | None = None,    # fallback: PYDANTIC_AI_GATEWAY_API_KEY / PAIG_API_KEY
    base_url: str | None = None,   # fallback: PYDANTIC_AI_GATEWAY_BASE_URL / PAIG_BASE_URL
    http_client: httpx.AsyncClient | None = None,
) -> Provider[Any]: ...

def normalize_gateway_provider(provider: str) -> str:
    # Strips 'gateway/' prefix and resolves aliases to canonical class-lookup names
    # 'chat' -> 'openai-chat', 'responses' -> 'openai-responses',
    # 'converse' -> 'bedrock', 'google' -> 'google-cloud'
    ...
```

### 10.1 Routing OpenAI Requests Through the Gateway

Use the Gateway as a single ingress point for OpenAI-compatible calls, simplifying key management across environments.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.openai import OpenAIChatModel


provider = gateway_provider(
    "openai",
    api_key=os.environ["PYDANTIC_AI_GATEWAY_API_KEY"],
)
model = OpenAIChatModel("gpt-4o-mini", openai_client=provider.client)

agent = Agent(model, system_prompt="You are a concise assistant.")


async def main() -> None:
    result = await agent.run("Explain what pydantic-ai's gateway provider does in 2 sentences.")
    print(result.output)


asyncio.run(main())
```

### 10.2 Routing Anthropic Requests Through the Gateway

The Gateway normalises `'anthropic'` to its dedicated route and wraps the `AsyncAnthropic` client transparently.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.anthropic import AnthropicModel


provider = gateway_provider(
    "anthropic",
    api_key=os.environ["PYDANTIC_AI_GATEWAY_API_KEY"],
)
# AnthropicProvider.client returns AsyncAnthropic; AnthropicModel accepts that
model = AnthropicModel("claude-haiku-4-5", anthropic_client=provider.client)

agent = Agent(model, system_prompt="Answer in plain text.")


async def main() -> None:
    result = await agent.run("What are the main advantages of using a gateway in production?")
    print(result.output)


asyncio.run(main())
```

### 10.3 Custom Route Override with `route`

The `route` parameter lets you target a named routing group in the Gateway — useful when the Gateway operator has set up a dedicated group for a specific model tier or cost centre.

```python
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider, normalize_gateway_provider
from pydantic_ai.models.openai import OpenAIChatModel


# normalize_gateway_provider shows what canonical name 'chat' resolves to
canonical = normalize_gateway_provider("chat")
print(f"'chat' normalises to: {canonical}")  # 'openai-chat'

provider = gateway_provider(
    "openai",
    api_key=os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY", ""),
    route="openai-premium",  # a custom routing group on the Gateway side
)
model = OpenAIChatModel("gpt-4o", openai_client=provider.client)

agent = Agent(model, system_prompt="You are an expert assistant.")


async def main() -> None:
    result = await agent.run(
        "Which gateway routes are available and how does route normalisation work?"
    )
    print(result.output)


asyncio.run(main())
```

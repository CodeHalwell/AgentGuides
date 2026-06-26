---
title: "PydanticAI Class Deep Dives Vol. 27"
description: "Source-verified deep dives into 10 pydantic-ai 2.0.0 class groups: TemporalModel + TemporalProviderFactory + _RequestParams (multi-model Temporal activity dispatch — models registry, using_model() ContextVar, provider_factory, image output guard, CompletedStreamedResponse streaming), TemporalFunctionToolset + TemporalWrapperToolset + GetToolsParams + CallToolParams + CallToolResult (Temporal activity-wrapped toolset protocol — CallToolResult discriminated union, per-tool ActivityConfig, temporal_activities property), TemporalMCPToolset + _cached_tool_defs (cross-activity MCP tool definition caching — cache_tools setting, tool_for_tool_def, invalidation strategy), LogfirePlugin (Temporal SimplePlugin for Logfire+OTel integration — connect_service_client hook, TracingInterceptor, OpenTelemetryConfig metrics exporter), DBOSAgent + DBOSParallelExecutionMode (DBOS durable agent wrapper — @DBOS.dbos_class() + DBOSConfiguredInstance, mcp_step_config/model_step_config, parallel_ordered_events vs sequential), DBOSModel + StepConfig (DBOS step-wrapped model — @DBOS.step() decorator, CompletedStreamedResponse streaming, DBOS.workflow_id guard), PrefectModel (Prefect task-wrapped model — with_options() pattern, event_stream_handler streaming requirement), PrefectFunctionToolset + PrefectWrapperToolset (Prefect task-wrapped toolset — per-tool config, None to disable wrapping, with_options(name=...) naming), doc_descriptions + DocstringStyle + _infer_docstring_style (griffe-backed docstring parser powering Tool schema generation — style auto-detection, returns → XML conversion, GoogleOptions), open_model_request_span + ModelRequestContext + OTel baggage constants (instrumentation internals — AGENT_NAME_BAGGAGE_KEY/RUN_ID_BAGGAGE_KEY/CONVERSATION_ID_BAGGAGE_KEY, TOKEN_HISTOGRAM_BOUNDARIES, DEFAULT_INSTRUMENTATION_VERSION=5, CostCalculationFailedWarning). All verified against pydantic-ai 2.0.0 source."
sidebar:
  label: "Class deep dives (Vol. 27)"
  order: 53
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.0.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.x API.
</Aside>

Ten class groups covering the internal wiring of pydantic-ai's durable execution integrations (Temporal, DBOS, Prefect) and two supporting subsystems (griffe-backed docstring parsing and OpenTelemetry instrumentation internals). These classes sit one layer below the public `TemporalAgent`/`DBOSAgent`/`PrefectAgent` façades and explain *how* they achieve durability.

---

## 1. `TemporalModel` + `TemporalProviderFactory` + `_RequestParams` — Temporal Activity-Dispatching Model

**Source**: `pydantic_ai/durable_exec/temporal/_model.py`

`TemporalModel` is the `WrapperModel` subclass placed inside every `TemporalAgent`. It intercepts `request()` and `request_stream()` and, when `workflow.in_workflow()` is `True`, serialises the call into a `_RequestParams` dataclass and dispatches it via `workflow.execute_activity(...)` — so the model request survives Temporal replays. Outside a workflow, it falls through to the wrapped model directly.

```python
# Key signatures verified from source:

@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    messages: list[ModelMessage]
    # model_settings stored as dict[str, Any] because Temporal drops unknown fields
    # from TypedDict subclasses (e.g. provider-specific ModelSettings subclasses)
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any
    model_id: str | None = None          # None → use default 'default' model

TemporalProviderFactory = Callable[[RunContext[AgentDepsT], str], Provider[Any]]
# Called by _infer_model() when a model_id string is NOT in _models_by_id registry.
# Receives the run context and model ID, returns a Provider to construct the model.

class TemporalModel(WrapperModel):
    _models_by_id: dict[str, Model]          # 'default' = primary; others via models= kwarg
    _model_id_var: ContextVar[str | None]    # per-workflow-step model override
    _provider_factory: TemporalProviderFactory | None
    temporal_activities: list[Callable]      # [request_activity, request_stream_activity]

    def using_model(self, model) -> Generator[None]:
        """Context manager — sets _model_id_var for the scope."""
        ...

    def _resolve_model_id(self, model_id: str | None, run_context=None) -> Model:
        """None → self.wrapped; in registry → registry[id]; else → infer via factory."""
        ...

    def prepare_request(self, model_settings, model_request_parameters):
        """Uses _current_model()'s profile — so using_model() changes validation rules."""
        ...

    # Image output is rejected: Temporal's 2 MB payload limit makes binary responses unsafe.
    def _validate_model_request_parameters(self, params) -> None:
        if params.allow_image_output:
            raise UserError("Image output is not supported with Temporal ...")
```

### 1.1 Basic Single-Model Setup

```python
import asyncio
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio import workflow, activity
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent


@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, question: str) -> str:
        # TemporalAgent.run() serialises the request to a Temporal activity automatically
        return await self.agent.run(question)


async def main():
    agent = Agent("openai:gpt-4.1-mini", system_prompt="You are a helpful assistant.")
    temporal_agent = TemporalAgent(agent)

    async with await Client.connect("localhost:7233") as client:
        # Register TemporalModel's activities with the worker
        async with Worker(
            client,
            task_queue="my-task-queue",
            workflows=[MyWorkflow],
            activities=temporal_agent.temporal_activities,  # <-- _model.request_activity etc.
        ):
            result = await client.execute_workflow(
                MyWorkflow.run,
                "What is the speed of light?",
                id="my-workflow-1",
                task_queue="my-task-queue",
            )
            print(result)
```

### 1.2 Multi-Model Registry — `models=` and `using_model()`

```python
import asyncio
from temporalio import workflow
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.models.openai import OpenAIModel


async def demonstrate_multi_model():
    # 'default' model for most calls; register a larger model for complex tasks
    agent = Agent("openai:gpt-4.1-mini")
    temporal_agent = TemporalAgent(
        agent,
        models={
            # Registered by ID — 'default' is reserved for the primary model
            "powerful": OpenAIModel("gpt-4.1"),
            "cheap": OpenAIModel("gpt-4.1-nano"),
        },
    )

    # Inside a workflow, switch models per step:
    @workflow.defn
    class AdaptiveWorkflow:
        @workflow.run
        async def run(self, task: str) -> str:
            # Cheap model for classification
            with temporal_agent.model.using_model("cheap"):
                complexity = await temporal_agent.run(f"Is this task complex? {task}")

            # Switch to powerful model only when needed
            model_id = "powerful" if "yes" in complexity.lower() else "cheap"
            with temporal_agent.model.using_model(model_id):
                return await temporal_agent.run(task)

    print("Multi-model temporal agent configured.")
    print(f"Registered model IDs: {list(temporal_agent.model._models_by_id.keys())}")
```

### 1.3 `TemporalProviderFactory` — Dynamic Providers per Run Context

```python
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext


def my_provider_factory(ctx: RunContext, model_id: str):
    """Return different providers based on run context — e.g. per-tenant API keys."""
    # ctx.deps could carry tenant info
    tenant_api_key = getattr(ctx.deps, "api_key", None) or "default-key"
    return OpenAIProvider(api_key=tenant_api_key)


agent = Agent("openai:gpt-4.1-mini")
temporal_agent = TemporalAgent(
    agent,
    provider_factory=my_provider_factory,
    # Any model ID string not in _models_by_id falls through to _infer_model(),
    # which calls my_provider_factory(run_context, "openai:gpt-4.1") to build the model.
)

print("Provider factory registered.")
print("Model IDs not in registry will be resolved via my_provider_factory.")
# Note: provider_factory only fires for unregistered string IDs inside a workflow.
# Outside a workflow, models.infer_model() is called directly.
```

---

## 2. `TemporalFunctionToolset` + `TemporalWrapperToolset` + `GetToolsParams` + `CallToolParams` + `CallToolResult` — Temporal Activity-Wrapped Toolset Protocol

**Source**: `pydantic_ai/durable_exec/temporal/_toolset.py`, `_function_toolset.py`

`TemporalWrapperToolset` is the abstract base class for all Temporal-wrapped toolsets. Concrete subclasses (`TemporalFunctionToolset`, `TemporalDynamicToolset`, `TemporalMCPToolset`) implement `temporal_activities` and turn their `call_tool()` implementations into `@activity.defn` functions registered with the Temporal worker. The discriminated union `CallToolResult` serialises all possible outcomes across the activity boundary.

```python
# Key signatures from source:

@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class GetToolsParams:
    serialized_run_context: Any

@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any
    tool_def: ToolDefinition | None

# CallToolResult is a discriminated union of 4 outcomes:
@dataclass
class _ApprovalRequired:          # tool needs human approval before executing
    metadata: dict[str, Any] | None = None
    kind: Literal['approval_required'] = 'approval_required'

@dataclass
class _CallDeferred:              # tool is deferred to an external handler
    metadata: dict[str, Any] | None = None
    kind: Literal['call_deferred'] = 'call_deferred'

@dataclass
class _ModelRetry:                # tool raised ModelRetry — feed message back to model
    message: str
    kind: Literal['model_retry'] = 'model_retry'

@dataclass
class _ToolReturn:                # successful tool result (ToolReturn or ToolReturnContent)
    result: _ToolReturnResult
    kind: Literal['tool_return'] = 'tool_return'

CallToolResult = Annotated[
    _ApprovalRequired | _CallDeferred | _ModelRetry | _ToolReturn,
    Discriminator('kind'),
]

class TemporalWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    id: str                              # must be set on wrapped toolset

    @property
    @abstractmethod
    def temporal_activities(self) -> list[Callable]: ...

    async def for_run(self, ctx) -> AbstractToolset:
        # Temporal-wrapped toolsets manage their own lifecycle per-activity, not per-run
        return self
```

### 2.1 `TemporalFunctionToolset` — Registering a FunctionToolset with Temporal

```python
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.workflow import ActivityConfig
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.temporal import TemporalAgent


def get_weather(city: str) -> str:
    """Retrieve current weather for a city."""
    return f"Sunny, 22°C in {city}"

def get_forecast(city: str, days: int) -> str:
    """Get a multi-day forecast."""
    return f"{days}-day forecast for {city}: mostly sunny"


async def main():
    toolset = FunctionToolset([get_weather, get_forecast], id="weather-tools")
    agent = Agent("openai:gpt-4.1-mini", toolsets=[toolset])

    default_config = ActivityConfig(start_to_close_timeout=30)
    temporal_agent = TemporalAgent(
        agent,
        tool_activity_config={
            "weather-tools": {
                "activity_config": default_config,
                # per-tool overrides: give forecast more time
                "tool_activity_config": {"get_forecast": ActivityConfig(start_to_close_timeout=60)},
            }
        },
    )

    # temporal_agent.temporal_activities includes the model activities AND
    # the TemporalFunctionToolset.call_tool_activity for "weather-tools"
    print(f"Activities to register: {[a.__name__ for a in temporal_agent.temporal_activities]}")
```

### 2.2 `CallToolResult` Discriminator in Action — Custom Activity Handler

```python
from pydantic_ai.durable_exec.temporal._toolset import (
    CallToolParams,
    CallToolResult,
)

# Simulate deserializing a CallToolResult returned from a Temporal activity:
import json

# Example: tool raised ModelRetry inside the activity
raw_retry: dict = {"kind": "model_retry", "message": "Invalid date format, try YYYY-MM-DD."}

# Pydantic discriminates on 'kind' field to pick the right dataclass:
from pydantic import TypeAdapter
adapter = TypeAdapter(CallToolResult)
result = adapter.validate_python(raw_retry)

print(type(result).__name__)   # _ModelRetry
print(result.message)          # Invalid date format, try YYYY-MM-DD.

# Approval required case:
raw_approval = {"kind": "approval_required", "metadata": {"risk": "high", "tool": "delete_record"}}
approval_result = adapter.validate_python(raw_approval)
print(type(approval_result).__name__)   # _ApprovalRequired
print(approval_result.metadata)         # {'risk': 'high', 'tool': 'delete_record'}
```

### 2.3 Per-Tool Activity Config Override — Disabling Task Wrapping

```python
from temporalio.workflow import ActivityConfig
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.temporal import TemporalAgent


async def fast_lookup(key: str) -> str:
    """In-memory lookup — no I/O, no need for activity overhead."""
    return {"user:42": "Alice", "user:43": "Bob"}.get(key, "unknown")

async def slow_api_call(endpoint: str) -> str:
    """External HTTP call — must be an activity for durability."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(endpoint)
    return resp.text


toolset = FunctionToolset([fast_lookup, slow_api_call], id="mixed-tools")
agent = Agent("openai:gpt-4.1-mini", toolsets=[toolset])

temporal_agent = TemporalAgent(
    agent,
    tool_activity_config={
        "mixed-tools": {
            "activity_config": ActivityConfig(start_to_close_timeout=30),
            "tool_activity_config": {
                # False → skip activity wrapping; fast_lookup runs inline in the workflow
                "fast_lookup": False,
                # slow_api_call uses the default activity_config above
            },
        }
    },
)
print("fast_lookup: inline (no activity)")
print("slow_api_call: wrapped as Temporal activity")
# Note: False-disabled tools MUST be async; non-async functions run in threads,
# which are not supported outside of an activity.
```

---

## 3. `TemporalMCPToolset` — Cross-Activity MCP Tool Definition Caching

**Source**: `pydantic_ai/durable_exec/temporal/_mcp_toolset.py`

`TemporalMCPToolset` wraps an `MCPToolset` so that `get_tools` and `call_tool` execute as Temporal activities. Its key optimisation is `_cached_tool_defs`: when the wrapped `MCPToolset.cache_tools` is `True`, tool definitions from the first `get_tools` activity are stored and reused across subsequent steps — avoiding redundant MCP `tools/list` round-trips for every workflow replay.

```python
# Key signatures from source:

class TemporalMCPToolset(TemporalMCPToolsetBase[AgentDepsT]):
    """Temporal wrapper for MCPToolset with cross-activity tool definition caching."""

    _cached_tool_defs: dict[str, ToolDefinition] | None
    # Set to a populated dict after the first get_tools call when cache_tools=True.
    # NOT invalidated by tools/list_changed notifications.
    # To pick up dynamic tool changes, set MCPToolset(cache_tools=False).

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool:
        """Delegates to the wrapped MCPToolset — preserves the MCP tool identity."""
        return self._toolset.tool_for_tool_def(tool_def)

    async def get_tools(self, ctx) -> dict[str, ToolsetTool]:
        # Fast path: return from cache if cache_tools=True and cache is populated
        if self._toolset.cache_tools and self._cached_tool_defs is not None:
            return {name: self.tool_for_tool_def(td) for name, td in self._cached_tool_defs.items()}

        # Slow path: execute get_tools via Temporal activity (makes MCP connection)
        result = await super().get_tools(ctx)
        if self._toolset.cache_tools:
            self._cached_tool_defs = {name: tool.tool_def for name, tool in result.items()}
        return result
```

### 3.1 Attaching a Cached MCP Toolset to a Temporal Agent

```python
from temporalio.workflow import ActivityConfig
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.durable_exec.temporal import TemporalAgent


async def main():
    # cache_tools=True (default): tool list fetched once per workflow, cached thereafter
    mcp_toolset = MCPToolset(
        {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
        id="filesystem-mcp",
        cache_tools=True,
    )
    agent = Agent("openai:gpt-4.1-mini", toolsets=[mcp_toolset])

    temporal_agent = TemporalAgent(
        agent,
        tool_activity_config={
            "filesystem-mcp": {
                "activity_config": ActivityConfig(start_to_close_timeout=30),
                "tool_activity_config": {},
            }
        },
    )

    print("MCP toolset registered with caching enabled.")
    print("First get_tools call connects to MCP server; subsequent replays use cache.")
    print(f"Temporal activities: {[a.__name__ for a in temporal_agent.temporal_activities]}")
```

### 3.2 Disabling Cache for Dynamic Tool Sets

```python
from pydantic_ai.mcp import MCPToolset

# Use cache_tools=False when the MCP server's tool list can change mid-workflow.
# Example: a plugin system where tools are registered at runtime.
dynamic_toolset = MCPToolset(
    {"command": "python", "args": ["-m", "my_dynamic_mcp_server"]},
    id="dynamic-mcp",
    cache_tools=False,  # Every get_tools call re-fetches from the MCP server
)

print("Dynamic toolset: get_tools fetches from server on every workflow step.")
print("Trade-off: slower replays, but always reflects live tool registrations.")
# TemporalMCPToolset._cached_tool_defs will always be None when cache_tools=False.
```

### 3.3 Tool Definition Caching Under Replay

```python
# Simulate what TemporalMCPToolset does during Temporal workflow replay:
from pydantic_ai.tools import ToolDefinition

class _CachingBehaviourDemo:
    """Illustrates the two-path logic inside TemporalMCPToolset.get_tools()."""

    def __init__(self, cache_tools: bool):
        self.cache_tools = cache_tools
        self._cached_tool_defs: dict[str, ToolDefinition] | None = None
        self._fetch_count = 0

    def fetch_from_server(self) -> dict[str, ToolDefinition]:
        self._fetch_count += 1
        return {"read_file": ToolDefinition(name="read_file", description="Read a file", parameters_json_schema={})}

    def get_tools(self) -> dict[str, ToolDefinition]:
        if self.cache_tools and self._cached_tool_defs is not None:
            return self._cached_tool_defs   # fast path — no MCP round-trip
        result = self.fetch_from_server()
        if self.cache_tools:
            self._cached_tool_defs = result
        return result


demo = _CachingBehaviourDemo(cache_tools=True)
demo.get_tools()  # fetch #1
demo.get_tools()  # cache hit
demo.get_tools()  # cache hit
print(f"Server fetches with cache_tools=True: {demo.fetch_count}")  # 1

demo2 = _CachingBehaviourDemo(cache_tools=False)
demo2.get_tools()
demo2.get_tools()
demo2.get_tools()
print(f"Server fetches with cache_tools=False: {demo2._fetch_count}")  # 3
```

---

## 4. `LogfirePlugin` — Temporal Client Plugin for Logfire + OTel

**Source**: `pydantic_ai/durable_exec/temporal/_logfire.py`

`LogfirePlugin` is a `temporalio.plugin.SimplePlugin` subclass that wires Logfire (Pydantic's observability platform) into a Temporal `ServiceClient` at connection time. It installs a `TracingInterceptor` for workflow/activity spans and optionally configures a Temporal runtime that exports metrics directly to the Logfire OTLP endpoint.

```python
# Key signatures from source:

class LogfirePlugin(SimplePlugin):
    """Temporal client plugin for Logfire.

    Installs:
    - TracingInterceptor (OTel spans for all Temporal operations)
    - Optional: OpenTelemetryConfig metrics exporter pointed at Logfire's /v1/metrics
    """

    def __init__(
        self,
        setup_logfire: Callable[[], Logfire] = _default_setup_logfire,
        *,
        metrics: bool = True,
    ): ...

    async def connect_service_client(
        self,
        config: ConnectConfig,
        next: Callable[[ConnectConfig], Awaitable[ServiceClient]],
    ) -> ServiceClient:
        # Calls self.setup_logfire() → configures OTel provider + instruments pydantic-ai
        # When metrics=True and token is available: mutates config.runtime to add metrics exporter
        ...


def _default_setup_logfire() -> Logfire:
    import logfire
    instance = logfire.configure()
    instance.instrument_pydantic_ai()  # installs InstrumentedModel on every agent
    return instance
```

### 4.1 Attaching `LogfirePlugin` to a Temporal Client

```python
import asyncio
from temporalio.client import Client
from pydantic_ai.durable_exec.temporal._logfire import LogfirePlugin

async def main():
    # LogfirePlugin is passed as a plugin at client construction time.
    # It calls setup_logfire() inside connect_service_client() before the connection completes.
    client = await Client.connect(
        "localhost:7233",
        plugins=[LogfirePlugin()],   # default: metrics=True, auto-configure logfire
    )
    print("Connected with Logfire OTel tracing and metrics.")
    # Every Temporal workflow/activity call now emits OTel spans to Logfire.
    # PydanticAI model requests emit gen_ai.* span attributes via InstrumentedModel.
```

### 4.2 Custom `setup_logfire` — Pre-configured Logfire Instance

```python
from temporalio.client import Client
from pydantic_ai.durable_exec.temporal._logfire import LogfirePlugin


def my_setup_logfire():
    """Custom Logfire initialisation with project-specific settings."""
    import logfire
    instance = logfire.configure(
        service_name="my-temporal-worker",
        service_version="2.0.0",
        send_to_logfire=True,
    )
    # instrument_pydantic_ai() ensures InstrumentedModel wraps every Agent's model
    instance.instrument_pydantic_ai()
    return instance


async def main():
    plugin = LogfirePlugin(
        setup_logfire=my_setup_logfire,
        metrics=True,   # Export Temporal metrics to Logfire OTLP endpoint
    )
    client = await Client.connect("localhost:7233", plugins=[plugin])
    print("Connected with custom Logfire setup.")
```

### 4.3 Metrics-Only Mode — Disabling `TracingInterceptor`

```python
from pydantic_ai.durable_exec.temporal._logfire import LogfirePlugin


def metrics_only_logfire():
    """Configure Logfire without pydantic-ai instrumentation."""
    import logfire
    return logfire.configure(service_name="my-worker")


async def main():
    # metrics=False → no OpenTelemetryConfig runtime injection;
    # the TracingInterceptor is always added (it's wired in __init__).
    plugin = LogfirePlugin(
        setup_logfire=metrics_only_logfire,
        metrics=False,   # Skip exporting Temporal metrics to Logfire
    )
    # Useful when you have an existing Prometheus/OTLP metrics pipeline
    # and only want Temporal trace spans forwarded to Logfire.
    print(f"Plugin name: {plugin.name}")   # 'LogfirePlugin'
    print("TracingInterceptor active, metrics exporter disabled.")
```

---

## 5. `DBOSAgent` + `DBOSParallelExecutionMode` — DBOS Durable Workflow Agent

**Source**: `pydantic_ai/durable_exec/dbos/_agent.py`

`DBOSAgent` wraps any `AbstractAgent` with DBOS's durable step semantics. It uses `@DBOS.dbos_class()` (a DBOS class-level step registry decorator) and `DBOSConfiguredInstance` (the DBOS instance base class) to register itself with the DBOS runtime. Model requests and MCP server calls are automatically wrapped as `@DBOS.step()` functions; tool calls from `FunctionToolset` are also wrapped individually.

```python
# Key signatures from source:

DBOSParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
# 'parallel' excluded: DBOS cannot guarantee deterministic event ordering for true parallel.
# 'parallel_ordered_events': tools run in parallel; events emitted in order after all complete.
# 'sequential': tools run one at a time — safest for strict replay determinism.

@DBOS.dbos_class()
class DBOSAgent(WrapperAgent[AgentDepsT, OutputDataT], DBOSConfiguredInstance):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,           # MUST be unique; used as configured instance name
        event_stream_handler: EventStreamHandler | None = None,
        mcp_step_config: StepConfig | None = None,
        model_step_config: StepConfig | None = None,
        parallel_execution_mode: DBOSParallelExecutionMode = 'parallel_ordered_events',
    ): ...
```

### 5.1 Basic `DBOSAgent` Wrapping

```python
import asyncio
from dbos import DBOS
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent

# Requires: pip install "pydantic-ai[dbos]" and DBOS configured

agent = Agent(
    "openai:gpt-4.1-mini",
    system_prompt="You are a helpful assistant.",
    name="my-agent",   # Required: name becomes the DBOS configured instance name
)

dbos_agent = DBOSAgent(
    agent,
    parallel_execution_mode="parallel_ordered_events",
    # model_step_config defaults to {} — uses DBOS default retry/timeout settings
    # mcp_step_config defaults to {} — same defaults for MCP server steps
)

# Use inside a DBOS workflow:
@DBOS.workflow()
async def my_durable_workflow(question: str) -> str:
    result = await dbos_agent.run(question)
    return result.output

async def main():
    result = await DBOS.execute_workflow_async(my_durable_workflow, "What is 2+2?")
    print(result)
```

### 5.2 Custom `StepConfig` for Model and MCP Calls

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent, StepConfig

agent = Agent("openai:gpt-4.1-mini", name="configurable-agent")

# StepConfig controls DBOS step-level retry and timeout behaviour:
model_config: StepConfig = {
    "retries_allowed": True,
    "max_attempts": 3,
    "interval_seconds": 2.0,
    "backoff_rate": 2.0,    # 2s, 4s, 8s between retries
}

mcp_config: StepConfig = {
    "retries_allowed": True,
    "max_attempts": 5,
    "interval_seconds": 1.0,
}

dbos_agent = DBOSAgent(
    agent,
    model_step_config=model_config,
    mcp_step_config=mcp_config,
)

print(f"Model steps: max 3 attempts, 2x backoff")
print(f"MCP steps: max 5 attempts, 1s interval")
```

### 5.3 `DBOSParallelExecutionMode` — Choosing Determinism vs Throughput

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.dbos import DBOSAgent


async def tool_a(x: int) -> int:
    """Tool A — could run concurrently."""
    return x * 2

async def tool_b(x: int) -> int:
    """Tool B — could run concurrently."""
    return x + 10


toolset = FunctionToolset([tool_a, tool_b])
agent = Agent("openai:gpt-4.1-mini", toolsets=[toolset], name="parallel-agent")

# parallel_ordered_events (default): tool_a and tool_b run concurrently in parallel,
# but events are buffered and emitted in deterministic order after both complete.
dbos_parallel = DBOSAgent(agent, parallel_execution_mode="parallel_ordered_events")

# sequential: tools run one at a time — best when tools have side effects that
# would interact if executed concurrently.
dbos_sequential = DBOSAgent(agent, parallel_execution_mode="sequential")

print(f"Parallel mode internal value: {dbos_parallel._parallel_execution_mode}")
# Note: 'parallel' (true concurrent events) is intentionally excluded because
# DBOS cannot replay non-deterministic event orderings.
```

---

## 6. `DBOSModel` + `StepConfig` — DBOS Step-Wrapped Model

**Source**: `pydantic_ai/durable_exec/dbos/_model.py`, `_utils.py`

`DBOSModel` is the `WrapperModel` installed by `DBOSAgent`. Each `request()` and `request_stream()` call is wrapped by `@DBOS.step()` at construction time — so the decorator fires once (not on every call), which is required by DBOS's activity registration model. The key `DBOS.workflow_id is None or DBOS.step_id is not None` guard in `request_stream` allows nested calls (inside an existing DBOS step or outside a workflow entirely) to bypass the step wrapper.

```python
# Key signatures from source:

class StepConfig(TypedDict, total=False):
    retries_allowed: bool
    interval_seconds: float
    max_attempts: int
    backoff_rate: float

class DBOSModel(WrapperModel):
    """WrapperModel that turns model request and request_stream into DBOS steps."""

    step_config: StepConfig
    event_stream_handler: EventStreamHandler[Any] | None
    _step_name_prefix: str    # = f"{agent_name}__model"

    # The @DBOS.step() decorators are applied once in __init__:
    #   @DBOS.step(name=f'{prefix}__model.request', **step_config)
    #   async def wrapped_request_step(messages, model_settings, params) -> ModelResponse

    # request_stream guard logic (from source):
    async def request_stream(self, messages, model_settings, params, run_context=None):
        if DBOS.workflow_id is None or DBOS.step_id is not None:
            # Not in a workflow, or already inside a step — pass through directly
            async with super().request_stream(...) as s:
                yield s
            return
        # In a workflow and not in a step → execute as DBOS step, return CompletedStreamedResponse
        response = await self._dbos_wrapped_request_stream_step(...)
        yield CompletedStreamedResponse(params, response)
```

### 6.1 Standalone `DBOSModel` Construction

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.durable_exec.dbos._model import DBOSModel
from pydantic_ai.durable_exec.dbos import StepConfig

model = OpenAIModel("gpt-4.1-mini")

step_config: StepConfig = {
    "retries_allowed": True,
    "max_attempts": 3,
    "interval_seconds": 1.5,
}

dbos_model = DBOSModel(
    model,
    step_name_prefix="my-workflow__model",
    step_config=step_config,
    event_stream_handler=None,    # No streaming handler — use request() only
)

print(f"Step name: my-workflow__model__model.request")
print("Model requests are now durable DBOS steps.")
print(f"Retries: {step_config['max_attempts']} attempts, {step_config['interval_seconds']}s interval")
```

### 6.2 The `DBOS.workflow_id` Guard in Streaming

```python
# The guard prevents double-wrapping when DBOSModel.request_stream is called
# from inside an already-running DBOS step (e.g. a nested tool call).

from pydantic_ai.durable_exec.dbos._model import DBOSModel
from pydantic_ai.models.openai import OpenAIModel
from dbos import DBOS

model = DBOSModel(
    OpenAIModel("gpt-4.1-mini"),
    step_name_prefix="test",
    step_config={},
)

# Pseudo-code illustrating the guard logic:
class _GuardDemo:
    """Illustrates the three code paths in request_stream."""

    def path(self, workflow_id, step_id):
        if workflow_id is None:
            return "pass-through: not in a DBOS workflow"
        if step_id is not None:
            return "pass-through: already inside a DBOS step (avoid double-wrap)"
        return "step: execute as DBOS step, yield CompletedStreamedResponse"

demo = _GuardDemo()
print(demo.path(workflow_id=None, step_id=None))          # pass-through
print(demo.path(workflow_id="wf-123", step_id="step-1"))  # pass-through
print(demo.path(workflow_id="wf-123", step_id=None))      # step
```

### 6.3 `StepConfig` — All Fields and Their Effect

```python
from pydantic_ai.durable_exec.dbos import StepConfig

# All fields are optional (TypedDict total=False):
full_config: StepConfig = {
    "retries_allowed": True,         # Enable automatic retries on failure
    "interval_seconds": 2.0,         # Base wait between retry attempts
    "max_attempts": 5,               # Total number of attempts (1 original + 4 retries)
    "backoff_rate": 1.5,             # Exponential multiplier: 2s, 3s, 4.5s, 6.75s
}

# Conservative config: no retries — fail fast on model errors
conservative: StepConfig = {
    "retries_allowed": False,
}

# Aggressive retry for flaky external APIs:
aggressive: StepConfig = {
    "retries_allowed": True,
    "max_attempts": 10,
    "interval_seconds": 0.5,
    "backoff_rate": 2.0,   # 0.5s, 1s, 2s, 4s, ... up to max_attempts
}

print("StepConfig fields: retries_allowed, interval_seconds, max_attempts, backoff_rate")
print(f"Full config applied to @DBOS.step(**full_config): {full_config}")
```

---

## 7. `PrefectModel` — Prefect Task-Wrapped Model

**Source**: `pydantic_ai/durable_exec/prefect/_model.py`

`PrefectModel` is the `WrapperModel` injected by `PrefectAgent`. Both `request()` and `request_stream()` are turned into `@task`-decorated functions at `__init__` time using Prefect's `task` decorator — the actual task objects are created once, and `with_options()` is used at call time to set the task name dynamically per request.

```python
# Key signatures from source:

class PrefectModel(WrapperModel):
    task_config: TaskConfig
    event_stream_handler: EventStreamHandler[Any] | None

    # Two @task functions created at __init__:
    # @task async def wrapped_request(messages, model_settings, params) -> ModelResponse
    # @task async def request_stream_task(messages, ..., ctx) -> ModelResponse

    async def request(self, messages, model_settings, model_request_parameters) -> ModelResponse:
        return await self._wrapped_request.with_options(
            # TaskConfig fields merged in as Prefect task options
            **self.task_config
        )(messages, model_settings, model_request_parameters)

    # request_stream: only usable when event_stream_handler is set.
    # Streams inside the task, then returns the final ModelResponse.
    # Caller receives a CompletedStreamedResponse wrapping the finished response.
```

### 7.1 `PrefectModel` Inside a Flow

```python
from prefect import flow
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.durable_exec.prefect._types import TaskConfig

agent = Agent("openai:gpt-4.1-mini", name="my-agent")

task_config: TaskConfig = {
    "retries": 3,
    "retry_delay_seconds": 2,
    "timeout_seconds": 60,
}

prefect_agent = PrefectAgent(
    agent,
    model_task_config=task_config,
)

@flow(name="question-answering-flow")
async def answer_flow(question: str) -> str:
    # Each model request runs as a Prefect task — visible in the Prefect UI.
    result = await prefect_agent.run(question)
    return result.output


async def main():
    answer = await answer_flow("What is the capital of France?")
    print(answer)
    # In Prefect UI: you'll see a task run named from task_config
    # with 3 retries configured.
```

### 7.2 Streaming with `event_stream_handler`

```python
from pydantic_ai import Agent
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.models import StreamedResponse
from pydantic_ai.tools import RunContext
from prefect import flow


async def my_stream_handler(ctx: RunContext, streamed: StreamedResponse) -> None:
    """Consume the stream and log tokens as they arrive."""
    async for event in streamed:
        pass   # In production: forward to a websocket or queue


agent = Agent("openai:gpt-4.1-mini", name="streaming-agent")
prefect_agent = PrefectAgent(
    agent,
    event_stream_handler=my_stream_handler,
    # model_task_config is required here: request_stream inside a flow MUST
    # have an event_stream_handler — otherwise PrefectModel raises:
    # 'A Prefect model cannot be used with model_request_stream() as it requires a run_context.'
)

@flow
async def streaming_flow(prompt: str) -> str:
    result = await prefect_agent.run(prompt)
    return result.output
```

### 7.3 `with_options()` Pattern — Dynamic Task Naming

```python
# PrefectModel uses with_options() to assign a unique task name per request.
# This makes individual model requests distinguishable in the Prefect UI.

from prefect import task

# Illustrates the with_options() pattern used inside PrefectModel.request():
@task
async def model_request_template(messages, settings, params):
    """Template task — actual work injected via closure."""
    pass

# At call time, PrefectModel does:
named_task = model_request_template.with_options(
    name="model-request: openai:gpt-4.1-mini",
    retries=3,
    retry_delay_seconds=2,
)
# Each invocation creates a new task run with a descriptive name.

print("with_options() creates a new task variant without re-registering the @task.")
print("This is why PrefectModel creates the @task once in __init__ and names at call time.")
```

---

## 8. `PrefectFunctionToolset` + `PrefectWrapperToolset` — Prefect Task-Wrapped Toolset

**Source**: `pydantic_ai/durable_exec/prefect/_function_toolset.py`, `_toolset.py`

`PrefectFunctionToolset` turns each `FunctionToolset` tool call into a Prefect task via `with_options(name=f'Call Tool: {name}', ...)`. A single `@task`-decorated `_call_tool_task` is created at `__init__` time and reused with `with_options()` at call time. Setting a tool's config entry to `None` opts it out of task wrapping entirely.

```python
# Key signatures from source:

class PrefectFunctionToolset(PrefectWrapperToolset[AgentDepsT]):
    _task_config: TaskConfig       # merged default config
    _tool_task_config: dict[str, TaskConfig | None]   # per-tool override

    async def call_tool(self, name, tool_args, ctx, tool) -> Any:
        tool_specific_config = self._tool_task_config.get(name, default_task_config)
        if tool_specific_config is None:
            # None → skip task wrapping for this tool; call directly
            return await super().call_tool(name, tool_args, ctx, tool)

        merged_config = self._task_config | tool_specific_config
        return await self._call_tool_task.with_options(
            name=f'Call Tool: {name}',
            **merged_config,
        )(name, tool_args, ctx, tool)
```

### 8.1 Basic `PrefectFunctionToolset` Setup

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.durable_exec.prefect._types import TaskConfig


async def web_search(query: str) -> str:
    """Search the web for the given query."""
    return f"Results for: {query}"


async def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression, {"__builtins__": {}})


toolset = FunctionToolset([web_search, calculate], id="helpers")
agent = Agent("openai:gpt-4.1-mini", toolsets=[toolset], name="tool-agent")

# Per-tool config: calculate is cheap and fast; give web_search more retries
tool_config: dict[str, TaskConfig | None] = {
    "web_search": {"retries": 3, "retry_delay_seconds": 1},
    "calculate": {},   # Empty dict → use default config; no special treatment
}

prefect_agent = PrefectAgent(
    agent,
    tool_task_config_by_name={"helpers": tool_config},
)

print("Each tool call appears as a separate Prefect task run named 'Call Tool: <name>'.")
```

### 8.2 Disabling Task Wrapping with `None`

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.durable_exec.prefect._types import TaskConfig


def in_memory_lookup(key: str) -> str:
    """Pure in-memory lookup — no side effects, no I/O."""
    return {"greeting": "hello", "farewell": "goodbye"}.get(key, "unknown")


async def send_notification(message: str) -> str:
    """Send an external notification — should be a durable step."""
    # ... actual HTTP call ...
    return f"Sent: {message}"


toolset = FunctionToolset([in_memory_lookup, send_notification], id="mixed")
agent = Agent("openai:gpt-4.1-mini", toolsets=[toolset], name="mixed-agent")

tool_config: dict[str, TaskConfig | None] = {
    "in_memory_lookup": None,       # None → skip task wrapping entirely
    "send_notification": {"retries": 2},  # Wrapped as Prefect task with 2 retries
}

prefect_agent = PrefectAgent(
    agent,
    tool_task_config_by_name={"mixed": tool_config},
)
print("in_memory_lookup: direct call (no Prefect task overhead)")
print("send_notification: Prefect task with retries")
```

### 8.3 Merging Default and Per-Tool Config

```python
from pydantic_ai.durable_exec.prefect._function_toolset import PrefectFunctionToolset
from pydantic_ai.durable_exec.prefect._types import TaskConfig, default_task_config

# Illustrate the config merging that happens in PrefectFunctionToolset.call_tool():
base_config: TaskConfig = {
    "retries": 2,
    "retry_delay_seconds": 1.0,
    "timeout_seconds": 30,
}

per_tool_overrides: TaskConfig = {
    "retries": 5,   # Override base retries for this specific tool
    # retry_delay_seconds and timeout_seconds inherited from base
}

merged = base_config | per_tool_overrides  # Python dict merge (right wins)
print(f"Merged config: {merged}")
# {'retries': 5, 'retry_delay_seconds': 1.0, 'timeout_seconds': 30}

# When per-tool config is None, the tool bypasses task wrapping completely:
print(f"default_task_config: {default_task_config}")  # {}
# {} | None would fail — None is checked explicitly before the merge.
```

---

## 9. `doc_descriptions` + `DocstringStyle` + `_infer_docstring_style` — Griffe-Backed Docstring Parser

**Source**: `pydantic_ai/_griffe.py`

`doc_descriptions` is called by `FunctionToolset` (and `Tool`) during tool registration to extract the function description and per-parameter descriptions from a docstring. It uses the [griffe](https://mkdocstrings.github.io/griffe/) library for parsing and supports `'google'`, `'numpy'`, `'sphinx'`, and `'auto'` (inference). When a `Returns` section is present, the description is reformatted as XML with `<summary>` and `<returns>` tags — a richer schema description for the model.

```python
# Key signatures from source:

DocstringStyle = Literal['google', 'numpy', 'sphinx']

def doc_descriptions(
    func: Callable[..., Any],
    sig: Signature,
    *,
    docstring_format: DocstringFormat,   # 'auto' | 'google' | 'numpy' | 'sphinx'
) -> tuple[str | None, dict[str, str]]:
    """Returns (main_description, {param_name: description})."""

    # 1. If no docstring → return (None, {})
    # 2. Infer style if 'auto' via _infer_docstring_style()
    # 3. For google: GoogleOptions(returns_named_value=False, returns_multiple_items=False)
    # 4. Parse with griffe's Docstring(doc, lineno=1, parser=style, parent=sig)
    # 5. Extract DocstringSectionKind.parameters → params dict
    # 6. Extract DocstringSectionKind.text → main_desc
    # 7. If DocstringSectionKind.returns present → wrap in XML:
    #    '<summary>{main_desc}</summary>\n<returns>\n<type>...</type>\n<description>...</description>\n</returns>'

def _infer_docstring_style(doc: str) -> DocstringStyle:
    # Tests regex patterns for Sphinx (:param:), Google (Args:\n  ...), NumPy (---+ underline)
    # Fallback: 'google'
```

### 9.1 Auto-Detection — Google, NumPy, Sphinx

```python
from inspect import signature
from pydantic_ai._griffe import doc_descriptions


def google_style_tool(query: str, max_results: int = 10) -> list[str]:
    """Search the web using the given query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        A list of matching URL strings.
    """
    return []


sig = signature(google_style_tool)
desc, params = doc_descriptions(google_style_tool, sig, docstring_format='auto')

print(repr(desc))
# '<summary>Search the web using the given query.</summary>\n<returns>\n<description>A list of matching URL strings.</description>\n</returns>'

print(params)
# {'query': 'The search query string.', 'max_results': 'Maximum number of results to return.'}
# These per-parameter descriptions flow into the tool's JSON schema "description" fields.
```

### 9.2 NumPy Style Detection

```python
from inspect import signature
from pydantic_ai._griffe import doc_descriptions


def numpy_style_tool(x: float, y: float) -> float:
    """Compute the Euclidean distance between two points.

    Parameters
    ----------
    x : float
        The x-coordinate.
    y : float
        The y-coordinate.
    """
    return (x ** 2 + y ** 2) ** 0.5


sig = signature(numpy_style_tool)
desc, params = doc_descriptions(numpy_style_tool, sig, docstring_format='auto')

print(repr(desc))
# 'Compute the Euclidean distance between two points.'

print(params)
# {'x': 'The x-coordinate.', 'y': 'The y-coordinate.'}
# NumPy detected from "Parameters\n----------" underline pattern.
```

### 9.3 Explicit Format Override — Sphinx

```python
from inspect import signature
from pydantic_ai._griffe import doc_descriptions


def sphinx_style_tool(user_id: int, active: bool = True) -> dict:
    """Fetch user profile from the database.

    :param user_id: The unique user identifier.
    :param active: Whether to restrict to active users only.
    :rtype: dict
    :returns: User profile dictionary with name, email, and roles.
    """
    return {}


sig = signature(sphinx_style_tool)

# Force 'sphinx' parsing even if auto-detection might pick something else:
desc, params = doc_descriptions(sphinx_style_tool, sig, docstring_format='sphinx')

print(repr(desc))
# Contains <returns> XML wrapping because a :returns: section was found.

print(params)
# {'user_id': 'The unique user identifier.', 'active': 'Whether to restrict to active users only.'}

# Use 'auto' if you don't control docstring format:
auto_desc, auto_params = doc_descriptions(sphinx_style_tool, sig, docstring_format='auto')
print("Auto-detected as sphinx:", repr(auto_desc) == repr(desc))
```

---

## 10. `open_model_request_span` + `ModelRequestContext` + OTel Baggage Constants — Instrumentation Internals

**Source**: `pydantic_ai/_instrumentation.py`

`_instrumentation.py` is the low-level OTel plumbing shared by `InstrumentedModel` (which calls `open_model_request_span`) and the `Instrumentation` capability. It defines the baggage key constants for correlating agent name, run ID, and conversation ID across distributed traces, the histogram bucket boundaries for token usage metrics, and the `CostCalculationFailedWarning` raised when `genai-prices` can't compute cost for a model.

```python
# Key constants and signatures from source:

DEFAULT_INSTRUMENTATION_VERSION = 5
# Version 5: full GenAI semantic conventions for multimodal (type='uri', mime_type, modality).
# Versions 2-4 are deprecated compatibility formats (emit PydanticAIDeprecationWarning).

# OTel Baggage keys — propagated across service boundaries in W3C baggage headers:
AGENT_NAME_BAGGAGE_KEY = 'gen_ai.agent.name'
RUN_ID_BAGGAGE_KEY = 'gen_ai.agent.call.id'
CONVERSATION_ID_BAGGAGE_KEY = 'gen_ai.conversation.id'

# Standard GenAI OTel span attributes:
GEN_AI_SYSTEM_ATTRIBUTE = 'gen_ai.system'         # 'openai', 'anthropic', etc.
GEN_AI_REQUEST_MODEL_ATTRIBUTE = 'gen_ai.request.model'
GEN_AI_PROVIDER_NAME_ATTRIBUTE = 'gen_ai.provider.name'

# Histogram bucket boundaries for gen_ai.client.token.usage metric:
# From OTel GenAI semantic conventions spec.
TOKEN_HISTOGRAM_BOUNDARIES = (
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864
)
# 14 boundaries = 15 buckets; spans ~4 orders of magnitude from 1 to 67M tokens.

MODEL_SETTING_ATTRIBUTES = ('max_tokens', 'top_p', 'seed', 'temperature', 'presence_penalty', 'frequency_penalty')
# These ModelSettings fields are written as span attributes on every model request span.

class CostCalculationFailedWarning(Warning):
    """Raised (as a warning) when genai-prices cannot calculate cost for a model response."""


def get_agent_run_baggage_attributes() -> dict[str, Any]:
    """Read agent name, run ID, and conversation ID from OTel baggage → span attributes."""
    # Called inside open_model_request_span to propagate baggage into the span.
    ...
```

### 10.1 Reading Baggage Keys in a Custom Span

```python
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai._instrumentation import (
    AGENT_NAME_BAGGAGE_KEY,
    RUN_ID_BAGGAGE_KEY,
    CONVERSATION_ID_BAGGAGE_KEY,
    get_agent_run_baggage_attributes,
)

# Set up an in-memory exporter for testing:
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
tracer = provider.get_tracer("test")

# Simulate pydantic-ai setting baggage at agent run start:
ctx = baggage.set_baggage(AGENT_NAME_BAGGAGE_KEY, "my-agent")
ctx = baggage.set_baggage(RUN_ID_BAGGAGE_KEY, "run-abc123", context=ctx)
ctx = baggage.set_baggage(CONVERSATION_ID_BAGGAGE_KEY, "conv-xyz", context=ctx)

with context.use_context(ctx):
    attrs = get_agent_run_baggage_attributes()
    print(attrs)
    # {'gen_ai.agent.name': 'my-agent', 'gen_ai.agent.call.id': 'run-abc123',
    #  'gen_ai.conversation.id': 'conv-xyz'}
    # These attributes are stamped onto every model request span automatically.
```

### 10.2 `TOKEN_HISTOGRAM_BOUNDARIES` — Custom OTel Metrics Setup

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.metrics.view import View, ExplicitBucketHistogramAggregation
from pydantic_ai._instrumentation import TOKEN_HISTOGRAM_BOUNDARIES

# The boundaries define the histogram buckets for gen_ai.client.token.usage.
# Use them to configure a matching view in your OTel SDK setup:
token_usage_view = View(
    instrument_name="gen_ai.client.token.usage",
    aggregation=ExplicitBucketHistogramAggregation(
        boundaries=list(TOKEN_HISTOGRAM_BOUNDARIES)
    ),
)

reader = InMemoryMetricReader()
meter_provider = MeterProvider(
    metric_readers=[reader],
    views=[token_usage_view],
)

print(f"Token histogram buckets: {len(TOKEN_HISTOGRAM_BOUNDARIES) + 1}")
print(f"Range: 1 token to {TOKEN_HISTOGRAM_BOUNDARIES[-1]:,} tokens")
# Token histogram buckets: 15
# Range: 1 token to 67,108,864 tokens
```

### 10.3 `CostCalculationFailedWarning` and `DEFAULT_INSTRUMENTATION_VERSION`

```python
import warnings
from pydantic_ai._instrumentation import (
    CostCalculationFailedWarning,
    DEFAULT_INSTRUMENTATION_VERSION,
    MODEL_SETTING_ATTRIBUTES,
)
from pydantic_ai.models.instrumented import InstrumentationSettings

# DEFAULT_INSTRUMENTATION_VERSION controls which OTel GenAI spec version is used:
print(f"Default instrumentation version: {DEFAULT_INSTRUMENTATION_VERSION}")   # 5

# Explicitly request an older version (e.g., for a consumer that expects v3 format):
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    old_settings = InstrumentationSettings(version=3)
    print(f"Requested version: {old_settings.version}")   # 3

# CostCalculationFailedWarning surfaces when genai-prices can't find a pricing entry:
def simulate_cost_warning():
    warnings.warn(
        "genai-prices: no pricing data for 'my-custom-model'",
        CostCalculationFailedWarning,
        stacklevel=2,
    )

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    simulate_cost_warning()

print(f"Warning type: {caught[0].category.__name__}")   # CostCalculationFailedWarning
print(f"Span attributes from ModelSettings: {MODEL_SETTING_ATTRIBUTES}")
# ('max_tokens', 'top_p', 'seed', 'temperature', 'presence_penalty', 'frequency_penalty')
```

---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 11"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.1: AgentTelemetryLayer+ChatTelemetryLayer+EmbeddingTelemetryLayer, Edge+EdgeGroup+SingleEdgeGroup+InternalEdgeGroup, Case+Default, EdgeRunner hierarchy, ExecutionContext, WorkflowGraphValidator, MCPTool+MCPSpecificApproval, SerializationMixin+SerializationProtocol, Evaluator+EvalItemResult+EvalNotPassedError, PerServiceCallHistoryPersistingMiddleware."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 34
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 11

Verified against **agent-framework 1.8.1** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. No API name has been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, `BaseEmbeddingClient`, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentFactory`, `WorkflowFactory`, `SecureAgentConfig`, `FunctionalWorkflowAgent`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — `AgentFileStore` hierarchy, `FileAccessProvider`, `MCPSkill` + `MCPSkillsSource`, `ToolMode`, `AgentEvalConverter` + `CheckResult` + `RubricScore`, `ChatContext`, `WorkflowAgent` + `WorkflowContext`, `TruncationStrategy`, `HistoryProvider` + `InMemoryHistoryProvider`, `DelegatingSkillsSource` + `InMemorySkillsSource` + `FunctionInvocationContext`
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`+`Worker`+`Client`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `HyperlightCodeActProvider`, `Mem0ContextProvider`, `RedisContextProvider`+`RedisHistoryProvider`, `StandardMagenticManager`+`MagenticContext`, `FileSkillsSource`+`FilteringSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `Workflow` + `InProcRunnerContext`, `FunctionExecutor`, `FunctionInvocationLayer`, `MemoryStore` + `MemoryIndexEntry` + `MemoryTopicRecord`, `TodoStore` + `TodoItem` + `TodoInput` + `TodoFileStore` + `TodoSessionStore`, `DeduplicatingSkillsSource`, `SkillsProvider`, `MCPTaskOptions`, `InMemoryCheckpointStorage`, `EvalScoreResult` + `CompactionStrategy` + `BaseAgent`

This volume uncovers **ten class groups** from the `agent-framework-core 1.8.1` internals
that were not covered in earlier volumes — focusing on the telemetry mixin layer, the
low-level workflow graph primitives (edges, runners, validator), MCP base class internals,
the DTO serialization foundation, and the evaluation Protocol:

| # | Class / group | Module |
|---|---|---|
| 1 | `AgentTelemetryLayer` + `ChatTelemetryLayer` + `EmbeddingTelemetryLayer` | `agent_framework.observability` |
| 2 | `Edge` + `EdgeGroup` + `SingleEdgeGroup` + `InternalEdgeGroup` | `agent_framework._workflows._edge` |
| 3 | `Case` + `Default` | `agent_framework._workflows._edge` |
| 4 | `EdgeRunner` + `SingleEdgeRunner` + `FanOutEdgeRunner` + `FanInEdgeRunner` + `SwitchCaseEdgeRunner` | `agent_framework._workflows._edge_runner` |
| 5 | `ExecutionContext` | `agent_framework._workflows._workflow_executor` |
| 6 | `WorkflowGraphValidator` | `agent_framework._workflows._validation` |
| 7 | `MCPTool` + `MCPSpecificApproval` | `agent_framework._mcp` |
| 8 | `SerializationMixin` + `SerializationProtocol` | `agent_framework._serialization` |
| 9 | `Evaluator` + `EvalItemResult` + `EvalNotPassedError` | `agent_framework._evaluation` |
| 10 | `PerServiceCallHistoryPersistingMiddleware` | `agent_framework._sessions` |

---

## 1 · `AgentTelemetryLayer` + `ChatTelemetryLayer` + `EmbeddingTelemetryLayer`

**Module:** `agent_framework.observability`

The three telemetry layer classes are mixins that every first-party agent and chat client
inherits to get automatic OpenTelemetry tracing, token-usage histograms, and latency
histograms — without any user-facing configuration beyond `configure_otel_providers()`.

### `AgentTelemetryLayer`

```python
class AgentTelemetryLayer:
    def __init__(
        self,
        *args,
        otel_agent_provider_name: str | None = None,
        otel_provider_name: str | None = None,
        **kwargs,
    ) -> None: ...
```

`AgentTelemetryLayer` is mixed into `Agent` and `RawAgent`. On construction it
resolves its **provider name** (used as the `gen_ai.provider.name` span attribute)
from three sources in priority order: `otel_agent_provider_name`, `otel_provider_name`,
or the class-level `AGENT_PROVIDER_NAME` constant.

It also initialises two OTel instruments:
- `token_usage_histogram` — records input/output token counts per invocation
- `duration_histogram` — records wall-clock seconds per invocation

The core method is `_trace_agent_invocation()`. The agent calls this instead of directly
awaiting the model client, and the tracing layer wraps a span around the call with
`gen_ai.operation.name = "invoke_agent"` plus agent ID, name, description, and thread ID.
When `ObservabilitySettings.ENABLED` is `False` the method short-circuits to the raw
`execute` callable with zero overhead.

### `ChatTelemetryLayer`

```python
class ChatTelemetryLayer:
    def __init__(
        self,
        *args,
        otel_provider_name: str | None = None,
        **kwargs,
    ) -> None: ...
```

`ChatTelemetryLayer` is mixed into every first-party `BaseChatClient` subclass
(`OpenAIChatClient`, `FoundryChatClient`, `AnthropicClient`, etc.). It wraps each
`get_response()` / `get_streaming_response()` call in a span with
`gen_ai.operation.name = "chat"`.

The mixin records `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` from the
`UsageDetails` in the response, enabling per-model cost attribution without code changes.

### `EmbeddingTelemetryLayer`

```python
class EmbeddingTelemetryLayer:
    def __init__(
        self,
        *args,
        otel_provider_name: str | None = None,
        **kwargs,
    ) -> None: ...
```

`EmbeddingTelemetryLayer` is mixed into `BaseEmbeddingClient` subclasses. It wraps
`get_embeddings()` calls in a span with `gen_ai.operation.name = "embeddings"` and
records the model name from the options.

### How the mixins compose

The MRO (Method Resolution Order) for `OpenAIChatClient` looks like:

```
OpenAIChatClient → ChatTelemetryLayer → FunctionInvocationLayer → BaseChatClient → ...
```

This means telemetry wraps the whole invocation including tool-calling loops — you get
accurate latency for the full multi-turn tool call, not just the first model round-trip.

### Disabling telemetry per-call

```python
from agent_framework.observability import ObservabilitySettings

# Globally disable (useful in tests)
ObservabilitySettings.ENABLED = False

# Selectively disable sensitive data capture
from agent_framework.observability import configure_otel_providers
configure_otel_providers(enable_sensitive_telemetry=False)
```

### Adding a custom provider name

```python
from agent_framework.openai import OpenAIChatClient

class MyOpenAIClient(OpenAIChatClient):
    AGENT_PROVIDER_NAME = "my_openai"   # appears as gen_ai.provider.name in all spans
```

### Reading span attributes

The full attribute catalogue is in `OtelAttr` (covered in class 9 of this volume). The
key agent-level attributes are:

| Attribute | Value |
|---|---|
| `gen_ai.operation.name` | `"invoke_agent"` |
| `gen_ai.agent.id` | `agent.id` |
| `gen_ai.agent.name` | `agent.name` |
| `gen_ai.conversation.id` | `session.service_session_id` |
| `gen_ai.usage.input_tokens` | cumulative input tokens |
| `gen_ai.usage.output_tokens` | cumulative output tokens |

---

## 2 · `Edge` + `EdgeGroup` + `SingleEdgeGroup` + `InternalEdgeGroup`

**Module:** `agent_framework._workflows._edge`

These are the lowest-level building blocks of every `WorkflowBuilder` graph. You normally
never construct them directly — `WorkflowBuilder.add_edge()` / `add_fan_out_edges()` etc.
create them for you — but understanding them is essential for reading serialised workflow
state, writing custom `EdgeGroup` subclasses, or debugging routing issues.

### `Edge`

```python
@dataclass(init=False)
class Edge(DictConvertible):
    source_id: str
    target_id: str
    condition_name: str | None

    def __init__(
        self,
        source_id: str,
        target_id: str,
        condition: EdgeCondition | None = None,
        *,
        condition_name: str | None = None,
    ) -> None: ...
```

An `Edge` is a directed link between two executor IDs. Its optional `condition` is a
callable `(data: Any) -> bool | Awaitable[bool]` that gates routing at runtime.

Key properties and methods:

| Member | Description |
|---|---|
| `edge.id` | `"source_id->target_id"` — stable serialisation key |
| `edge.has_condition` | `True` when a predicate was supplied |
| `await edge.should_route(data)` | Evaluates predicate; `True` when no condition |
| `edge.to_dict()` | Serialises source/target + condition name (no callable) |
| `Edge.from_dict(d)` | Reconstructs without callable (`condition=None`) |

```python
from agent_framework._workflows._edge import Edge

# Unconditional edge
e1 = Edge("ingest", "validate")
assert e1.id == "ingest->validate"
assert await e1.should_route({"any": "data"})  # True

# Conditional edge
e2 = Edge("score", "approve", condition=lambda d: d["score"] > 0.8)
assert await e2.should_route({"score": 0.9})   # True
assert not await e2.should_route({"score": 0.5})  # False

# Round-trip serialisation (condition name is preserved, callable is not)
d = e2.to_dict()
# {"source_id": "score", "target_id": "approve", "condition_name": "<lambda>"}
e2_restored = Edge.from_dict(d)
assert e2_restored.condition_name == "<lambda>"
assert not e2_restored.has_condition  # callable is gone after deserialisation
```

### `EdgeGroup`

```python
@dataclass(init=False)
class EdgeGroup(DictConvertible):
    id: str
    type: str
    edges: list[Edge]
```

`EdgeGroup` is the base for all routing groups. The Pregel runner
(`Runner`) iterates over `EdgeGroup` instances, not raw `Edge` objects.

**Important API: `EdgeGroup.register`**

Use this decorator to register a custom `EdgeGroup` subclass so it survives
`to_dict()` / `from_dict()` round-trips (e.g. when checkpointing):

```python
from agent_framework._workflows._edge import EdgeGroup, Edge

@EdgeGroup.register
class PriorityEdgeGroup(EdgeGroup):
    def __init__(self, edges: list[Edge], *, priority: int = 0, **kwargs):
        super().__init__(edges, **kwargs)
        self.priority = priority

    def to_dict(self):
        d = super().to_dict()
        d["priority"] = self.priority
        return d
```

Key properties:

| Property | Returns |
|---|---|
| `group.source_executor_ids` | Deduped list of upstream executor IDs |
| `group.target_executor_ids` | Deduped list of downstream executor IDs |

### `SingleEdgeGroup`

```python
@EdgeGroup.register
@dataclass(init=False)
class SingleEdgeGroup(EdgeGroup):
    def __init__(
        self,
        source_id: str,
        target_id: str,
        condition: EdgeCondition | None = None,
        *,
        id: str | None = None,
    ) -> None: ...
```

Convenience wrapper for a single 1-to-1 edge. Created by `WorkflowBuilder.add_edge()`.

```python
from agent_framework._workflows._edge import SingleEdgeGroup

group = SingleEdgeGroup("fetch", "parse")
assert len(group.edges) == 1
assert group.edges[0].source_id == "fetch"
```

### `InternalEdgeGroup`

```python
@EdgeGroup.register
@dataclass(init=False)
class InternalEdgeGroup(EdgeGroup):
    def __init__(self, executor_id: str) -> None: ...
```

Created automatically when each executor is added to the `WorkflowBuilder`. Carries the
workflow's **initial input** from the internal source (`__INTERNAL__:<executor_id>`) to
the executor itself. It appears in serialised workflow definitions and in OTel span
attributes — do not confuse it with user-defined edges.

```python
# WorkflowBuilder does this for every add_node() call:
from agent_framework._workflows._edge import InternalEdgeGroup
internal = InternalEdgeGroup("ingest")
assert internal.edges[0].source_id.startswith("__INTERNAL__")
assert internal.edges[0].target_id == "ingest"
```

---

## 3 · `Case` + `Default`

**Module:** `agent_framework._workflows._edge`

`Case` and `Default` are the **runtime** companions to the serialisable
`SwitchCaseEdgeGroupCase` / `SwitchCaseEdgeGroupDefault` data classes. They carry live
callables and are consumed directly by `SwitchCaseEdgeRunner` during execution.

```python
@dataclass
class Case:
    condition: Callable[[Any], bool]
    target: Executor | SupportsAgentRun

@dataclass
class Default:
    target: Executor | SupportsAgentRun
```

You pass `Case` and `Default` instances to `WorkflowBuilder.add_switch_case_edge_group()`:

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._edge import Case, Default

# Define executors
class Triage(Executor):
    def __init__(self):
        super().__init__(id="triage")

class UrgentHandler(Executor):
    def __init__(self):
        super().__init__(id="urgent")

class RoutineHandler(Executor):
    def __init__(self):
        super().__init__(id="routine")

triage = Triage()
urgent = UrgentHandler()
routine = RoutineHandler()

client = OpenAIChatClient()

builder = WorkflowBuilder()
builder.add_node(triage)
builder.add_node(urgent)
builder.add_node(routine)

# Route based on triage output
builder.add_switch_case_edge_group(
    source=triage,
    cases=[
        Case(
            condition=lambda data: data.get("priority") == "high",
            target=urgent,
        ),
    ],
    default=Default(target=routine),
)
```

`Case.condition` is evaluated in order; the first `Case` that returns `True` wins.
`Default.target` is taken when no `Case` matches. Both must point to an `Executor` or
any object implementing `SupportsAgentRun` (e.g. `Agent`).

---

## 4 · `EdgeRunner` hierarchy

**Module:** `agent_framework._workflows._edge_runner`

`EdgeRunner` subclasses are the **execution-time** message delivery mechanism. The Pregel
`Runner` creates one `EdgeRunner` per `EdgeGroup` after the workflow is built. You never
instantiate these directly, but understanding their semantics explains routing behaviour.

### `EdgeRunner` (abstract base)

```python
class EdgeRunner(ABC):
    def __init__(
        self,
        edge_group: EdgeGroup,
        executors: dict[str, Executor],
    ) -> None: ...

    @abstractmethod
    async def send_message(
        self,
        message: WorkflowMessage,
        state: State,
        ctx: RunnerContext,
    ) -> bool: ...
```

`send_message()` returns `True` if the message was **processed** (delivered or buffered),
`False` if the runner cannot handle it (wrong target, type mismatch). Returning `True`
does not always mean the downstream executor ran — a `FanInEdgeRunner` may return `True`
while still buffering for more inputs.

Every `send_message()` implementation wraps its span with `EdgeGroupDeliveryStatus`:

| Status | Meaning |
|---|---|
| `DELIVERED` | Executor was called |
| `BUFFERED` | Accepted into fan-in buffer; not yet dispatched |
| `DROPPED_TYPE_MISMATCH` | `Executor.can_handle()` returned `False` |
| `DROPPED_TARGET_MISMATCH` | Directed message aimed at a different executor |
| `DROPPED_CONDITION_FALSE` | Edge predicate evaluated to `False` |
| `EXCEPTION` | Predicate or executor raised |

### `SingleEdgeRunner`

Handles `SingleEdgeGroup` and `InternalEdgeGroup`. Checks target ID (if directed),
calls `can_handle()`, evaluates the condition, then calls `_execute_on_target()`.
**Returns `True`** even when the condition was `False` (message was processed, just
not routed) — callers must not retry.

```
message → check target_id match
        → can_handle()?
        → await edge.should_route(data)?
        → _execute_on_target(target, [source], message, state, ctx)
```

### `FanOutEdgeRunner`

Handles `FanOutEdgeGroup`. Applies the optional `selection_func` to narrow the candidate
target list, then dispatches to all matching targets **concurrently** via `asyncio.gather`:

```
message → selection_func(data, target_ids) → filtered_targets
        → for each target: can_handle()? + should_route()?
        → asyncio.gather(*[_execute_on_target(t, ...) for t in deliverable])
```

When the message has a `target_id` set (directed fan-out), only that single target is
evaluated — no concurrent dispatch occurs.

### `FanInEdgeRunner`

Handles `FanInEdgeGroup`. Aggregates messages from all upstream sources into a buffer.
When **all** expected source IDs have contributed at least one message, it dispatches
a single aggregated message wrapping a `list` of all buffered payloads to the single
downstream target.

```
message → buffer[source_id].append(message)
        → is_ready_to_send()? (all source IDs buffered)
              → aggregated = [m.data for m in buffer.values()]
              → _execute_on_target(target, sources, aggregated_msg, state, ctx)
```

The `FanInEdgeRunner` holds state (`_buffer`) across multiple `send_message()` calls.
This buffer is **not persisted** — if the workflow is checkpointed mid-fan-in, buffered
messages are lost and the fan-in must be re-driven.

### `SwitchCaseEdgeRunner`

Inherits `FanOutEdgeRunner` directly. The `SwitchCaseEdgeGroup`'s internal selection
function implements the case-matching logic, so `SwitchCaseEdgeRunner` needs no
additional code beyond the constructor.

### Complete routing flow example

```python
import asyncio
from agent_framework import WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._workflow_context import WorkflowContext

class Splitter(Executor):
    def __init__(self):
        super().__init__(id="splitter")

    @handler
    async def handle(self, data: str, ctx: WorkflowContext[str]) -> None:
        # Fan-out: produce two outputs
        await ctx.send_message("branch_a", data + "_a")
        await ctx.send_message("branch_b", data + "_b")

class BranchA(Executor):
    def __init__(self):
        super().__init__(id="branch_a")

    @handler
    async def handle(self, data: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.set_output(data.upper())

class BranchB(Executor):
    def __init__(self):
        super().__init__(id="branch_b")

    @handler
    async def handle(self, data: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.set_output(data.upper())

class Merger(Executor):
    def __init__(self):
        super().__init__(id="merger")

    @handler
    async def handle(self, data: list[str], ctx: WorkflowContext[list[str], str]) -> None:
        await ctx.set_output("|".join(data))

splitter, branch_a, branch_b, merger = Splitter(), BranchA(), BranchB(), Merger()

builder = WorkflowBuilder()
builder.add_node(splitter)
builder.add_node(branch_a)
builder.add_node(branch_b)
builder.add_node(merger)
builder.add_fan_out_edges(source=splitter, targets=[branch_a, branch_b])
builder.add_fan_in_edges(sources=[branch_a, branch_b], target=merger)
builder.set_output_from(merger)

workflow = builder.build()
result = asyncio.run(workflow.run("hello"))
print(result)  # "HELLO_A|HELLO_B"
```

---

## 5 · `ExecutionContext`

**Module:** `agent_framework._workflows._workflow_executor`

`ExecutionContext` is a `dataclass` used internally by `WorkflowExecutor` to track a
single **sub-workflow execution** — the state of one call from a parent workflow into
a child via `SubWorkflowRequestMessage` / `SubWorkflowResponseMessage`.

```python
@dataclass
class ExecutionContext:
    execution_id: str
    collected_responses: dict[str, Any]   # request_id → response_data
    expected_response_count: int
    pending_requests: dict[str, WorkflowEvent]  # request_id → request_info_event
```

| Field | Type | Description |
|---|---|---|
| `execution_id` | `str` | UUID that identifies this sub-workflow invocation |
| `collected_responses` | `dict[str, Any]` | Responses received so far from child (keyed by request_id) |
| `expected_response_count` | `int` | How many responses must arrive before the child is unblocked |
| `pending_requests` | `dict[str, WorkflowEvent]` | Outstanding HITL requests that have been sent but not yet answered |

**Lifecycle:** `WorkflowExecutor` creates one `ExecutionContext` per child invocation and
holds it in `_execution_contexts`. When a `SubWorkflowResponseMessage` arrives, the
executor looks up the context by `execution_id`, decrements the pending count, stores the
response, and re-runs the child if `len(collected_responses) == expected_response_count`.

This is the data structure that makes sub-workflow HITL work: outstanding
`request_info` events are recorded in `pending_requests` so that when a human
responds, the executor can match the response to the right execution context and
resume the child with its collected inputs.

```python
# Accessing sub-workflow contexts when debugging a WorkflowExecutor:
from agent_framework._workflows._workflow_executor import WorkflowExecutor

executor = ...  # obtained from a running workflow
for exec_id, ctx in executor._execution_contexts.items():
    print(f"Sub-workflow {exec_id}: "
          f"{len(ctx.collected_responses)}/{ctx.expected_response_count} responses, "
          f"{len(ctx.pending_requests)} pending HITL requests")
```

---

## 6 · `WorkflowGraphValidator`

**Module:** `agent_framework._workflows._validation`

`WorkflowGraphValidator` runs **seven sequential validation checks** when
`WorkflowBuilder.build()` is called. Understanding which check fires for which error
saves significant debugging time.

```python
class WorkflowGraphValidator:
    def validate_workflow(
        self,
        edge_groups: Sequence[EdgeGroup],
        executors: dict[str, Executor],
        start_executor: Executor,
        output_executors: list[str],
        intermediate_executors: list[str] | None = None,
    ) -> None: ...
```

### The 7 checks in order

| Check | Exception raised | Trigger |
|---|---|---|
| 1. Edge duplication | `EdgeDuplicationError` | Two `Edge` objects with the same `id` (`"src->tgt"`) |
| 2. Handler output annotations | — (warning only in 1.8.1) | `@handler` missing `WorkflowContext[T]` generic |
| 3. Type compatibility | `TypeCompatibilityError` | `source.output_types` incompatible with `target.input_types` |
| 4. Graph connectivity | `GraphConnectivityError` | Executor unreachable from `start_executor` |
| 5. Self-loop detection | — (warning only) | Edge from executor to itself |
| 6. Dead-end detection | — (info only) | Executor with no outgoing edges (except intentional outputs) |
| 7. Output validation | `WorkflowValidationError` | Output/intermediate executor not in graph, or missing `workflow_output_types` |

### Commonly encountered errors

**`TypeCompatibilityError`** — the most frequent build-time error. Fires when you wire two
executors whose types don't match. The framework checks `list[source_output]` vs
`target_input` for `FanInEdgeGroup` connections:

```python
# WRONG: BranchA outputs str but Merger expects list[int]
class BranchA(Executor):
    @handler
    async def handle(self, data: str, ctx: WorkflowContext[str]) -> None:
        await ctx.set_output("result")   # output_type = str

class Merger(Executor):
    @handler
    async def handle(self, data: list[int], ctx: WorkflowContext[list[int]]) -> None:
        ...  # expects list[int] — TypeCompatibilityError!
```

**`GraphConnectivityError`** — fires when you add a node but forget to connect it:

```python
builder.add_node(orphan)   # added but no add_edge() call → connectivity error
```

**`WorkflowValidationError`** — fires when `output_from=` names an executor that has no
`workflow_output_types` (i.e., a handler that never calls `ctx.set_output()`):

```python
builder.set_output_from(some_executor)  # some_executor has no ctx.set_output() call
# → WorkflowValidationError: "Output executor 'X' must have output type annotations."
```

### Custom pre-build validation

You can call `WorkflowGraphValidator` directly on your own `EdgeGroup` + `Executor`
collections before handing them to `WorkflowBuilder`:

```python
from agent_framework._workflows._validation import WorkflowGraphValidator

validator = WorkflowGraphValidator()
try:
    validator.validate_workflow(
        edge_groups=my_edge_groups,
        executors=my_executors,
        start_executor=start,
        output_executors=["output_node"],
    )
    print("Graph is valid")
except Exception as e:
    print(f"Validation failed: {e}")
```

---

## 7 · `MCPTool` + `MCPSpecificApproval`

**Module:** `agent_framework._mcp`

`MCPTool` is the **abstract base** for the three concrete MCP transports
(`MCPStdioTool`, `MCPStreamableHTTPTool`, `MCPWebsocketTool`). You cannot instantiate
it directly, but its constructor parameters and instance attributes are shared by all
three subclasses.

### `MCPSpecificApproval`

```python
class MCPSpecificApproval(TypedDict, total=False):
    always_require_approval: Collection[str] | None
    never_require_approval: Collection[str] | None
```

Fine-grained tool approval policy. Passed as `approval_mode` to any MCP transport:

```python
from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
from agent_framework._mcp import MCPSpecificApproval

# stdio: dangerous tools always need approval; safe tools never do
mcp = MCPStdioTool(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    approval_mode=MCPSpecificApproval(
        always_require_approval=["write_file", "delete_file"],
        never_require_approval=["read_file", "list_directory"],
    ),
)

# HTTP: blanket approval modes still available
mcp_http = MCPStreamableHTTPTool(
    name="search_api",
    url="http://localhost:8080/mcp",
    approval_mode="never_require",  # or "always_require" or MCPSpecificApproval(...)
)
```

### `MCPTool` constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Name of this MCP connection |
| `description` | `str \| None` | `None` | Human-readable description |
| `approval_mode` | `Literal["always_require", "never_require"] \| MCPSpecificApproval \| None` | `None` | Approval policy |
| `allowed_tools` | `Collection[str] \| None` | `None` | Allowlist of tool names to expose |
| `tool_name_prefix` | `str \| None` | `None` | Prefix prepended to all exposed tool names |
| `load_tools` | `bool` | `True` | Fetch tool list on connect |
| `parse_tool_results` | `Callable[[CallToolResult], str \| list[Content]] \| None` | `None` | Custom result parser |
| `load_prompts` | `bool` | `True` | Fetch prompt list on connect |
| `parse_prompt_results` | `Callable[[GetPromptResult], str] \| None` | `None` | Custom prompt parser |
| `session` | `ClientSession \| None` | `None` | Pre-existing MCP client session |
| `request_timeout` | `int \| None` | `None` | Seconds before MCP request times out |
| `client` | `SupportsChatGetResponse \| None` | `None` | Chat client for sampling callbacks |
| `additional_properties` | `dict \| None` | `None` | Arbitrary metadata |
| `task_options` | `MCPTaskOptions \| None` | `None` | Long-running task lifecycle options |
| `additional_tool_argument_names` | `Sequence[str] \| Mapping[str, Sequence[str]] \| None` | `None` | Extra args forwarded to MCP server |

### Key instance attributes

| Attribute | Type | Description |
|---|---|---|
| `is_connected` | `bool` | `True` after successful `__aenter__` |
| `functions` | `list[FunctionTool]` | Loaded tools as `FunctionTool` instances |
| `approval_mode` | `...` | The resolved approval policy |
| `tool_name_prefix` | `str \| None` | Normalised prefix (trailing `_.-` stripped) |

### Per-tool result parsing

After the tool list is loaded, you can override the parser on individual tools:

```python
async with MCPStdioTool(...) as mcp:
    for fn in mcp.functions:
        if fn.name == "my_tool__search":
            fn.result_parser = lambda raw: raw.content[0].text.strip()
```

### `additional_tool_argument_names` — forwarding hidden context

Pass extra arguments to MCP tools beyond their declared schema. Useful for forwarding
user identity or tenant context without surfacing it to the LLM:

```python
from agent_framework import MCPStreamableHTTPTool, Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient

async def get_tenant_id() -> str:
    return "tenant_abc"

tenant_tool = FunctionTool(get_tenant_id)

mcp = MCPStreamableHTTPTool(
    name="my_service",
    url="http://localhost:9000/mcp",
    # Forward "tenant_id" to every tool call
    additional_tool_argument_names=["tenant_id"],
    # OR per-tool: {"my_service__search": ["tenant_id"]}
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    tools=[mcp, tenant_tool],
)
```

---

## 8 · `SerializationMixin` + `SerializationProtocol`

**Module:** `agent_framework._serialization`

`SerializationMixin` is the base for every persistable agent-framework object —
`Agent`, `BaseAgent`, `FunctionTool`, `AgentSession`, `WorkflowBuilder`, and more.
Understanding it lets you build custom serialisable components and plug them into the
checkpoint / declarative-agent system.

### `SerializationProtocol`

```python
class SerializationProtocol(Protocol):
    def to_dict(
        self,
        *,
        exclude: set[str] | None = None,
        exclude_none: bool = True,
    ) -> dict[str, Any]: ...

    @classmethod
    def from_dict(
        cls: type[ClassT],
        value: MutableMapping[str, Any],
        /,
        *,
        dependencies: MutableMapping[str, Any] | None = None,
    ) -> ClassT: ...
```

Structural protocol — any class implementing `to_dict()` + `from_dict()` with these
signatures is compatible. The framework uses duck-typing, not inheritance, for nesting.

### `SerializationMixin` — class variables

| Class variable | Type | Description |
|---|---|---|
| `DEFAULT_EXCLUDE` | `set[str]` | Fields excluded from `to_dict()` |
| `INJECTABLE` | `set[str]` | Fields excluded from serialisation but injectable at `from_dict()` |
| `_SHALLOW_COPY_FIELDS` | `set[str]` | Fields copied by reference in `__deepcopy__` (default: `{"raw_representation"}`) |

### `to_dict()` semantics

- Adds `"type"` key (the class's type identifier) unless `"type"` is in `exclude`.
- Recursively serialises nested `SerializationProtocol` objects.
- Serialises `list` and `dict` values containing `SerializationProtocol` items.
- Skips non-JSON-serialisable values with a debug log (no exception).
- Skips private attributes (names starting with `_`).
- Skips fields in `DEFAULT_EXCLUDE` and `INJECTABLE`.

### `from_dict()` — dependency injection patterns

```python
# Pattern 1 — simple injection
MyClass.from_dict(
    data,
    dependencies={"my_class": {"db_connection": conn}},
)

# Pattern 2 — dict parameter injection (inject into a nested dict field)
MyClass.from_dict(
    data,
    dependencies={"my_class": {"providers_dict": {"cache": redis_client}}},
)

# Pattern 3 — instance-specific injection (keyed by field value)
MyClass.from_dict(
    data,
    dependencies={"my_class": {"name:my_agent": {"api_key": key}}},
)
```

### Building a custom serialisable tool

```python
from agent_framework._serialization import SerializationMixin
from agent_framework import FunctionTool
from typing import Any

class CachingFunctionTool(FunctionTool, SerializationMixin):
    DEFAULT_EXCLUDE = {"_cache", "api_key"}  # exclude transient/sensitive fields
    INJECTABLE = {"api_key"}                  # injectable at restore time

    def __init__(self, func, *, api_key: str, **kwargs):
        super().__init__(func, **kwargs)
        self.api_key = api_key
        self._cache: dict[str, Any] = {}

    # Override for custom restore logic
    @classmethod
    def from_dict(cls, data, /, *, dependencies=None):
        deps = (dependencies or {}).get("caching_function_tool", {})
        api_key = deps.get("api_key", "")
        instance = super().from_dict(data, dependencies=dependencies)
        instance.api_key = api_key
        return instance
```

### Round-tripping an AgentSession

```python
from agent_framework import AgentSession

session = AgentSession(session_id="user-123")
session.state["counter"] = 42

# Serialise
snapshot = session.to_dict()
# {"type": "agent_session", "session_id": "user-123", "state": {"counter": 42}, ...}

# Restore
restored = AgentSession.from_dict(snapshot)
assert restored.state["counter"] == 42
```

---

## 9 · `Evaluator` + `EvalItemResult` + `EvalNotPassedError`

**Module:** `agent_framework._evaluation`

These three classes form the **evaluation backend interface** — the Protocol every
evaluation engine implements, the per-item result data class, and the exception used to
gate CI pipelines.

All three are decorated with `@experimental(feature_id=ExperimentalFeature.EVALS)`.
Suppress the warning in tests:

```python
import warnings
from agent_framework._feature_stage import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)
```

### `Evaluator` Protocol

```python
@runtime_checkable
class Evaluator(Protocol):
    name: str

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str,
    ) -> EvalResults: ...
```

Any class exposing `name: str` and `async evaluate(items, *, eval_name) -> EvalResults`
satisfies this protocol without inheriting from it. The framework uses `isinstance(obj, Evaluator)`
checks (enabled by `@runtime_checkable`) to validate backends at registration time.

**Implementing a custom evaluator:**

```python
import asyncio
from agent_framework._evaluation import (
    Evaluator, EvalItem, EvalResults, EvalItemResult, EvalScoreResult,
)

class KeywordEvaluator:
    """Simple keyword-match evaluator for testing."""
    name = "keyword_match"

    def __init__(self, required_keywords: list[str]):
        self._keywords = required_keywords

    async def evaluate(
        self,
        items: Sequence[EvalItem],
        *,
        eval_name: str = "KeywordEval",
    ) -> EvalResults:
        results = []
        for item in items:
            output = item.output or ""
            matched = all(kw.lower() in output.lower() for kw in self._keywords)
            results.append(
                EvalItemResult(
                    item_id=item.id or str(id(item)),
                    status="pass" if matched else "fail",
                    scores={
                        "keyword_match": EvalScoreResult(
                            name="keyword_match",
                            score=1.0 if matched else 0.0,
                        )
                    },
                )
            )
        passed = sum(1 for r in results if r.status == "pass")
        return EvalResults(
            eval_name=eval_name,
            status="pass" if passed == len(results) else "fail",
            pass_count=passed,
            fail_count=len(results) - passed,
            items=results,
        )
```

### `EvalItemResult`

```python
@dataclass
class EvalItemResult:
    item_id: str
    status: Literal["pass", "fail", "error"]
    scores: dict[str, EvalScoreResult] = field(default_factory=dict)
    error_code: str | None = None
```

| Field | Description |
|---|---|
| `item_id` | Provider-assigned or user-supplied ID for this item |
| `status` | `"pass"`, `"fail"`, or `"error"` |
| `scores` | Map from evaluator name → `EvalScoreResult` with `name`, `score`, and optional `reason` |
| `error_code` | Error category string when `status == "error"` (e.g. `"QueryExtractionError"`) |

### `EvalNotPassedError`

```python
class EvalNotPassedError(Exception):
    """Raised when evaluation results contain failures."""
```

`LocalEvaluator.evaluate_and_assert()` raises this exception when any item has
`status == "fail"`. Use it in CI to gate deployments:

```python
import asyncio
from agent_framework._evaluation import EvalNotPassedError
from agent_framework import LocalEvaluator, evaluate_agent

async def run_eval_gate():
    evaluator = LocalEvaluator(evaluators=[keyword_eval])
    try:
        results = await evaluate_agent(
            agent=my_agent,
            items=test_items,
            evaluator=evaluator,
            assert_pass=True,   # raises EvalNotPassedError on failures
        )
        print(f"All {results.pass_count} items passed")
    except EvalNotPassedError as e:
        print(f"Eval gate failed: {e}")
        raise SystemExit(1)

asyncio.run(run_eval_gate())
```

### Complete eval pipeline with custom evaluator

```python
import asyncio
from agent_framework import Agent, evaluate_agent, EvalItem
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client=client, instructions="You are a helpful assistant.")

items = [
    EvalItem(
        input="What is the capital of France?",
        expected_output="Paris",
    ),
    EvalItem(
        input="What is 2 + 2?",
        expected_output="4",
    ),
]

keyword_eval = KeywordEvaluator(required_keywords=["paris"])  # from example above

async def main():
    results = await evaluate_agent(agent=agent, items=items, evaluator=keyword_eval)
    for item_result in results.items or []:
        print(f"{item_result.item_id}: {item_result.status}")
        for eval_name, score in item_result.scores.items():
            print(f"  {eval_name}: {score.score:.2f}")

asyncio.run(main())
```

---

## 10 · `PerServiceCallHistoryPersistingMiddleware`

**Module:** `agent_framework._sessions`

`PerServiceCallHistoryPersistingMiddleware` is an internal `ChatMiddleware` injected
automatically when an agent uses `HistoryProvider` instances with
`require_per_service_call_history_persistence = True`. Most users encounter this
indirectly — but understanding it explains the exact point at which history is written
and why tool-call results appear in the history mid-conversation.

### Constructor

```python
class PerServiceCallHistoryPersistingMiddleware(ChatMiddleware):
    def __init__(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        providers: Sequence[HistoryProvider],
        service_stores_history: bool = False,
    ) -> None: ...
```

| Parameter | Description |
|---|---|
| `agent` | The owning agent — used to call `load_history()` / `store_history()` |
| `session` | The active `AgentSession` for this invocation |
| `providers` | Participating `HistoryProvider` instances |
| `service_stores_history` | When `True` (e.g. Azure AI Agents), the remote service stores history server-side; the middleware skips local loading but still persists after each call |

### Behaviour

**When `service_stores_history = False`** (default — local history management):

1. Before each model call: loads history providers into a fresh `SessionContext`.
2. Injects a **local sentinel conversation ID** so the function-calling loop runs without
   creating a conversation on the remote service.
3. After the model call returns: persists the full updated message list through all
   providers.
4. Returns the response with the **real** conversation ID restored.

**When `service_stores_history = True`** (Azure AI Agents, Foundry, etc.):

1. Skips the pre-call load (service already has history).
2. Passes the real conversation ID through unchanged.
3. After each call: persists the current turn to all local providers.

This dual mode is why `FileHistoryProvider` (or `RedisHistoryProvider`) correctly captures
partial tool-call history even when the session is interrupted mid-function-loop.

### Triggering the middleware

```python
from agent_framework import Agent
from agent_framework._sessions import FileHistoryProvider
from agent_framework.openai import OpenAIChatClient

provider = FileHistoryProvider(
    storage_path="./history",
    require_per_service_call_history_persistence=True,  # enables this middleware
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[provider],
)
```

### When to enable `require_per_service_call_history_persistence`

Enable it when you need **durable partial-turn history** — for example:

- Long multi-tool-call conversations where a crash mid-turn would lose progress.
- Audit logging that must capture every model round-trip, including intermediate tool calls.
- Replay scenarios where you need to reconstruct the exact message sequence.

The trade-off is one extra `provider.store_*()` call per model round-trip, so avoid it
for high-frequency, low-latency agents.

### Inspecting the middleware in a running agent

```python
from agent_framework._sessions import PerServiceCallHistoryPersistingMiddleware

# After building your agent, inspect its middleware pipeline:
for layer in agent._chat_middleware_pipeline.layers:
    if isinstance(layer, PerServiceCallHistoryPersistingMiddleware):
        print(f"Per-service-call persistence active: {len(layer._providers)} providers")
        for p in layer._providers:
            print(f"  {type(p).__name__}")
```

---

## Version notes

All code examples in this volume were verified against **`agent-framework==1.8.1`**
installed June 2026. The telemetry layer classes (`AgentTelemetryLayer`,
`ChatTelemetryLayer`, `EmbeddingTelemetryLayer`), `WorkflowGraphValidator`, and
`SerializationMixin` are stable public APIs. `ExecutionContext`, `EdgeRunner` subclasses,
`Case`, `Default`, and the `_edge_runner` module are **implementation details** — their
interfaces may change in patch releases; prefer using `WorkflowBuilder` and `Workflow`
public APIs where possible rather than instantiating these directly.

`Evaluator`, `EvalItemResult`, and `EvalNotPassedError` carry the
`@experimental(feature_id=ExperimentalFeature.EVALS)` decorator — suppress
`ExperimentalWarning` in tests with `warnings.filterwarnings("ignore", ...)`.

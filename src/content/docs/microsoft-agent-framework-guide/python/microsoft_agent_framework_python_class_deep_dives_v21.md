---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 21"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.9.0: WorkflowContext[OutT,W_OutT] (per-executor execution context — send_message, yield_output, request_info, state R/W, source_executor_ids, is_streaming), FanInEdgeGroup+FanOutEdgeGroup (converging/broadcasting edge groups — min-source/target validation, selection_func dynamic routing, to_dict serialization), SwitchCaseEdgeGroup+SwitchCaseEdgeGroupCase+SwitchCaseEdgeGroupDefault (switch/case routing — sequential predicate evaluation, exactly-one-default invariant, missing-condition placeholder), TokenBudgetComposedStrategy (multi-strategy composition — early_stop, annotate_message_groups pipeline, strict-budget double-fallback), SelectiveToolCallCompactionStrategy+ToolResultCompactionStrategy (tool history reduction — exclude vs inline summary replacement, keep_last_tool_call_groups), SlidingWindowStrategy+SummarizationStrategy (sliding window + LLM-backed summarization — preserve_system, target_count+threshold, bidirectional summary trace metadata), StepWrapper+FunctionalWorkflow+RunContext (functional workflow internals — caching by call_index, executor_bypassed on cache hit, checkpoint per step), MCPWebsocketTool+MCPStreamableHTTPTool (MCP network transports — wss:// WebSocket, streamable HTTP/SSE with header_provider ContextVar injection), MCPTaskOptions (long-running MCP task lifecycle — SEP-2663 tasks/call→tasks/get→tasks/result, default_ttl, max_task_wait, cancel_remote_task_on_local_cancellation), AgentResponseUpdate+ChatResponseUpdate+ContinuationToken (streaming chunk types — .text property, continuation_token resumption, author_name in multi-agent, finish_reason)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 44
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 21

Verified against **agent-framework 1.9.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework._workflows._workflow_context`,
`agent_framework._workflows._edge`, `agent_framework._compaction`,
`agent_framework._workflows._functional`, `agent_framework._mcp`,
`agent_framework._types`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, middleware ABCs, compaction, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — harness providers, compaction strategies, `WorkflowViz`, MCP transports
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — message/chat types, `ResponseStream`, `AgentContext`, functional workflows, `SkillsSource`, eval model, tokenizer, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exceptions
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — feature staging, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, embedding clients, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, orchestration builders, `AgentFactory`, `SecureAgentConfig`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — file store hierarchy, `FileAccessProvider`, `MCPSkill`, `ToolMode`, eval helpers, `ChatContext`, `WorkflowAgent`, compaction, history providers, skills composition
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `Mem0ContextProvider`, Redis providers, Magentic internals, `FileSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `Workflow`, `InProcRunnerContext`, `FunctionExecutor`, `FunctionInvocationLayer`, memory harness, todo harness, `DeduplicatingSkillsSource`, `SkillsProvider`, `MCPTaskOptions`, `InMemoryCheckpointStorage`, `BaseAgent`
- [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) — telemetry layers, `Edge`+`EdgeGroup` primitives, `Case`+`Default`, `EdgeRunner` hierarchy, `ExecutionContext`, `WorkflowGraphValidator`, `MCPTool`, serialization mixin, `Evaluator`, `PerServiceCallHistoryPersistingMiddleware`
- [Vol. 12](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v12/) — Skills ABCs, `FileSkill`, `InlineSkillResource`+`InlineSkillScript`, `FileSkillScript`+`SkillScriptRunner`, `SupportsAgentRun`, `RunnerContext`, edge-routing descriptors, `WorkflowValidationError` hierarchy, `A2AAgent`+`A2AExecutor`, exception leaf classes
- [Vol. 13](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v13/) — OpenAI Responses/Completions/Embedding clients, Anthropic + Claude agent clients, multi-cloud Claude variants, group-chat + handoff + Magentic orchestration internals, declarative HTTP/MCP/approval handlers
- [Vol. 14](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v14/) — `State` (superstep cache), `OutputDesignation`, `MessageType`+`WorkflowMessage` internals, `DictConvertible` mixin, middleware pipeline hierarchy, `MiddlewareDict`, `FunctionRequestResult`, `OtelAttr`, security policy classes
- [Vol. 15](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v15/) — AG-UI client layer, AG-UI protocol wrappers, ChatKit, DevServer, GAIA benchmark, CopilotStudioAgent, AzureAISearchContextProvider, CosmosHistoryProvider, Durable external layer, AgentFunctionApp
- [Vol. 16](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v16/) — FoundryAgent+FoundryAgentOptions, FoundryLocalClient, FoundryMemoryProvider, FoundryEvals, BedrockChatClient, BedrockEmbeddingClient, MagenticManagerBase, BaseGroupChatOrchestrator, AgentRequestInfoResponse+CacheProvider, Purview exception hierarchy
- [Vol. 17](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v17/) — ToolApprovalMiddleware+ToolApprovalRule+ToolApprovalState, AgentLoopMiddleware+JudgeVerdict, SamplingApprovalCallback+MCP sampling security, to_prompt_agent, FoundryEmbeddingClient, ContentUnderstandingContextProvider, FileSearchConfig, AgentFrameworkTracer, TaskRunner (Tau2), FoundryChatClient hosted tool factories
- [Vol. 18](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v18/) — Skill+SkillFrontmatter+SkillScriptRunner, InlineSkill, skills source pipeline, AgentFileStore+InMemoryAgentFileStore, FileAccessProvider, BackgroundAgentsProvider, MemoryStore, WorkflowGraphValidator, MagenticBuilder+MagenticManagerBase+MagenticProgressLedger, LocalEvaluator+EvalItem+ConversationSplit
- [Vol. 19](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v19/) — ConcurrentBuilder, SequentialBuilder, HandoffBuilder+HandoffConfiguration+HandoffSentEvent, HandoffAgentUserRequest, OrchestrationState, AgentModeProvider+get_agent_mode+set_agent_mode, TodoItem+TodoInput+TodoCompleteInput, TodoStore+TodoSessionStore+TodoFileStore, TodoProvider, MagenticResetSignal+StandardMagenticManager
- [Vol. 20](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v20/) — SupportsCodeInterpreterTool and 5 other hosted-tool Protocols, SupportsGetEmbeddings, ReleaseCandidateFeature+FeatureStageWarning+ExperimentalWarning, EmbeddingGenerationOptions+Embedding+GeneratedEmbeddings, WorkflowEventSource, SubWorkflowRequestMessage+SubWorkflowResponseMessage, RequestInfoMixin+response_handler, WorkflowAgent.RequestInfoFunctionArgs, EdgeGroupDeliveryStatus, IntegrityLabel+LabelTrackingFunctionMiddleware, MiddlewareTermination+WorkflowConvergenceException

This volume covers **ten class groups** focussed on the per-executor execution context,
the fan-in/fan-out and switch/case edge group primitives, the full compaction strategy
hierarchy, functional workflow internals, both MCP network transports and long-running task
options, and the streaming update and continuation-token types:

| # | Class / group | Sub-package |
|---|---|---|
| 1 | `WorkflowContext[OutT, W_OutT]` | `agent_framework._workflows._workflow_context` |
| 2 | `FanInEdgeGroup` · `FanOutEdgeGroup` | `agent_framework._workflows._edge` |
| 3 | `SwitchCaseEdgeGroup` · `SwitchCaseEdgeGroupCase` · `SwitchCaseEdgeGroupDefault` | `agent_framework._workflows._edge` |
| 4 | `TokenBudgetComposedStrategy` | `agent_framework._compaction` |
| 5 | `SelectiveToolCallCompactionStrategy` · `ToolResultCompactionStrategy` | `agent_framework._compaction` |
| 6 | `SlidingWindowStrategy` · `SummarizationStrategy` | `agent_framework._compaction` |
| 7 | `StepWrapper` · `FunctionalWorkflow` · `RunContext` | `agent_framework._workflows._functional` |
| 8 | `MCPWebsocketTool` · `MCPStreamableHTTPTool` | `agent_framework._mcp` |
| 9 | `MCPTaskOptions` | `agent_framework._mcp` |
| 10 | `AgentResponseUpdate` · `ChatResponseUpdate` · `ContinuationToken` | `agent_framework._types` |

---

## 1 · `WorkflowContext[OutT, W_OutT]`

**Sub-package:** `agent_framework._workflows._workflow_context`  
**Install:** `pip install agent-framework-core`

`WorkflowContext` is the primary interface injected into every executor handler at
runtime. Its two type parameters correspond to what an executor can *send* to downstream
executors (`OutT`) and what it can *yield* as workflow-level output (`W_OutT`). The
framework creates and injects the context automatically — you never construct one directly
in production code, but understanding its constructor constraints helps with testing and
subclassing.

### Class signature (1.9.0)

```python
from typing import Generic, TypeVar, Any
from collections.abc import Awaitable

OutT = TypeVar("OutT")
W_OutT = TypeVar("W_OutT")

class WorkflowContext(Generic[OutT, W_OutT]):
    def __init__(
        self,
        executor,
        source_executor_ids: list[str],
        state,
        runner_context,
        trace_contexts: list | None = None,
        source_span_ids: list[str] | None = None,
        request_id: str | None = None,
    ) -> None:
        # Raises ValueError if source_executor_ids is empty
        ...

    # Message routing
    async def send_message(
        self,
        message: OutT,
        target_id: str | None = None,
    ) -> None: ...

    # Workflow output
    async def yield_output(self, output: W_OutT) -> None: ...

    # Custom events (protected event types are blocked)
    async def add_event(self, event: "WorkflowEvent") -> None: ...

    # Human-in-the-loop
    async def request_info(
        self,
        request_data: Any,
        response_type: type,
        *,
        request_id: str | None = None,
    ) -> None: ...

    # State access
    def get_state(self, key: str, default: Any = None) -> Any: ...
    def set_state(self, key: str, value: Any) -> None: ...

    # Source introspection
    def get_source_executor_id(self) -> str: ...  # raises RuntimeError if multiple sources
    @property
    def source_executor_ids(self) -> list[str]: ...
    @property
    def request_id(self) -> str | None: ...

    # Streaming guard
    def is_streaming(self) -> bool: ...

    # Introspection (defensive copies)
    def get_sent_messages(self) -> list[OutT]: ...
    def get_yielded_outputs(self) -> list[W_OutT]: ...
```

### Key facts

| Member | Behaviour | Gotcha |
|---|---|---|
| `source_executor_ids` | List of IDs that sent the message triggering this invocation | Empty list raises `ValueError` at construction time |
| `get_source_executor_id()` | Convenience single-source accessor | Raises `RuntimeError` if `len(source_executor_ids) > 1` — always check in fan-in executors |
| `send_message(msg, target_id)` | Delivers `msg` to all targets, or to `target_id` only | OTel trace context injected automatically when tracing is enabled |
| `yield_output(output)` | Classification (`output`, `intermediate`, hidden) fixed at workflow-build time | Cannot vary per call — the designation is set on the `OutputDesignation` for the executor |
| `add_event(event)` | Allows custom events | Raises `ValueError` for `output`, `intermediate`, and lifecycle event types |
| `request_info(...)` | Fires `WorkflowEvent.request_info`; framework logs a warning if no `@response_handler` is registered | `request_id` kwarg overrides the auto-generated one |
| `is_streaming()` | Delegates to `runner_context.is_streaming()` | Returns `False` when run outside a streaming session |

### Example 1 — basic typed contexts

```python
import asyncio
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


# Side-effects only: no typed send/yield
class LoggerExecutor(Executor):
    @handler
    async def on_message(
        self,
        message: str,
        ctx: WorkflowContext,  # untyped — only side-effects needed
    ) -> None:
        print(f"[log] {message}")


# Sends integers downstream
class CountingExecutor(Executor):
    @handler
    async def on_text(
        self,
        message: str,
        ctx: WorkflowContext[int],  # OutT=int
    ) -> None:
        await ctx.send_message(len(message))


# Sends strings and yields strings as workflow output
class SummaryExecutor(Executor):
    @handler
    async def on_count(
        self,
        count: int,
        ctx: WorkflowContext[str, str],  # OutT=str, W_OutT=str
    ) -> None:
        summary = f"Processed {count} characters."
        await ctx.send_message(summary)
        await ctx.yield_output(summary)
```

### Example 2 — `send_message` with `target_id` for selective fan-out

```python
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class RouterExecutor(Executor):
    """Routes incoming documents to the correct downstream executor by type."""

    @handler
    async def on_document(
        self,
        doc: dict,
        ctx: WorkflowContext[dict],
    ) -> None:
        doc_type = doc.get("type", "unknown")

        if doc_type == "invoice":
            # target_id matches the executor ID registered in the WorkflowBuilder
            await ctx.send_message(doc, target_id="invoice_processor")
        elif doc_type == "contract":
            await ctx.send_message(doc, target_id="contract_processor")
        else:
            # No target_id — broadcast to all connected executors
            await ctx.send_message(doc)
```

### Example 3 — `request_info` and reading `request_id` in a response handler

```python
from dataclasses import dataclass
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._request_info_mixin import response_handler
from agent_framework._workflows._workflow_context import WorkflowContext


@dataclass
class ApprovalRequest:
    action: str
    payload: dict


class ApprovalGateExecutor(Executor):
    @handler
    async def on_action(
        self,
        message: dict,
        ctx: WorkflowContext[dict],
    ) -> None:
        # Pause and ask a human (or orchestrator) for approval
        await ctx.request_info(
            request_data=ApprovalRequest(
                action=message["action"],
                payload=message,
            ),
            response_type=bool,
        )

    @response_handler
    async def on_approval(
        self,
        original_request: ApprovalRequest,
        response: bool,
        ctx: WorkflowContext[dict],
    ) -> None:
        # ctx.request_id is set to the ID of the request_info that triggered this handler
        print(f"Handling approval for request_id={ctx.request_id}")
        if response:
            await ctx.send_message(
                {"status": "approved", "action": original_request.action}
            )
        else:
            await ctx.send_message(
                {"status": "rejected", "action": original_request.action}
            )
```

### Example 4 — state read/write and `is_streaming` guard

```python
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class StatefulSummaryExecutor(Executor):
    """Accumulates a running total and streams partial results when in streaming mode."""

    @handler
    async def on_value(
        self,
        value: int,
        ctx: WorkflowContext[str, str],
    ) -> None:
        # Read + update running total from shared state
        total: int = ctx.get_state("running_total", default=0)
        total += value
        ctx.set_state("running_total", total)

        # Only yield intermediate progress updates to streaming clients
        if ctx.is_streaming():
            await ctx.yield_output(f"[streaming] running total: {total}")
        else:
            # In batch mode, only yield on milestone boundaries
            if total % 100 == 0:
                await ctx.yield_output(f"milestone: {total}")
```

---

## 2 · `FanInEdgeGroup` · `FanOutEdgeGroup`

**Sub-package:** `agent_framework._workflows._edge`  
**Install:** `pip install agent-framework-core`

Both classes extend `EdgeGroup` and are registered via `@EdgeGroup.register`. They
complement each other: `FanOutEdgeGroup` broadcasts a single source to multiple targets
(optionally with dynamic routing), while `FanInEdgeGroup` converges multiple upstream
sources onto a single downstream processor. Together they form the backbone of parallel
pipeline patterns.

### Class signatures (1.9.0)

```python
from collections.abc import Callable
from typing import Any

class FanInEdgeGroup(EdgeGroup):
    def __init__(
        self,
        source_ids: list[str],
        target_id: str,
        *,
        id: str | None = None,
    ) -> None:
        # Raises ValueError if len(source_ids) <= 1
        ...


class FanOutEdgeGroup(EdgeGroup):
    def __init__(
        self,
        source_id: str,
        target_ids: list[str],
        selection_func: Callable[[Any, list[str]], list[str]] | None = None,
        *,
        selection_func_name: str | None = None,
        id: str | None = None,
    ) -> None:
        # Raises ValueError if len(target_ids) <= 1
        ...

    @property
    def target_ids(self) -> list[str]: ...  # defensive copy

    def to_dict(self) -> dict[str, Any]: ...
    # to_dict includes selection_func_name in its payload;
    # the callable itself is not persisted
```

### Key facts

| Class | Minimum cardinality | `selection_func` | Serialization |
|---|---|---|---|
| `FanInEdgeGroup` | ≥ 2 sources, 1 target | N/A — all sources converge | `source_ids` + `target_id` round-trip cleanly |
| `FanOutEdgeGroup` | 1 source, ≥ 2 targets | Optional — receives `(message, target_ids)`, returns subset | `selection_func_name` serialized; callable must be re-registered after restore |

- `FanInEdgeGroup` stores arriving messages in an internal buffer until **all** registered
  sources have contributed for a given superstep, then delivers the batch to the target.
- `FanOutEdgeGroup` without a `selection_func` delivers to **every** target ID in
  `target_ids` — true broadcast.
- `selection_func_name` is auto-extracted from `func.__name__` when not provided
  explicitly. Supply it explicitly for lambdas or if you plan to restore from a checkpoint.

### Example 1 — simple `FanInEdgeGroup` (parser + enricher → writer)

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._edge import FanInEdgeGroup
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class ParserExecutor(Executor):
    @handler
    async def on_raw(self, raw: str, ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message({"parsed": raw.strip()})


class EnricherExecutor(Executor):
    @handler
    async def on_raw(self, raw: str, ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message({"enriched": raw.upper()})


class WriterExecutor(Executor):
    @handler
    async def on_combined(self, doc: dict, ctx: WorkflowContext) -> None:
        print(f"Writing: {doc}")


def build_fan_in_workflow():
    parser = ParserExecutor(id="parser")
    enricher = EnricherExecutor(id="enricher")
    writer = WriterExecutor(id="writer")

    # Both parser and enricher must contribute before writer is invoked
    fan_in = FanInEdgeGroup(
        source_ids=["parser", "enricher"],
        target_id="writer",
    )

    return (
        WorkflowBuilder()
        .add_executor(parser)
        .add_executor(enricher)
        .add_executor(writer)
        .add_edge_group(fan_in)
        .build()
    )
```

### Example 2 — `FanOutEdgeGroup` without `selection_func` (broadcast all)

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._edge import FanOutEdgeGroup
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class IngestExecutor(Executor):
    @handler
    async def on_event(self, event: dict, ctx: WorkflowContext[dict]) -> None:
        # Broadcast the same event to every downstream executor
        await ctx.send_message(event)


class AuditExecutor(Executor):
    @handler
    async def on_event(self, event: dict, ctx: WorkflowContext) -> None:
        print(f"[audit] {event}")


class MetricsExecutor(Executor):
    @handler
    async def on_event(self, event: dict, ctx: WorkflowContext) -> None:
        print(f"[metrics] {event.get('type')}")


def build_broadcast_workflow():
    ingest = IngestExecutor(id="ingest")
    audit = AuditExecutor(id="audit")
    metrics = MetricsExecutor(id="metrics")

    fan_out = FanOutEdgeGroup(
        source_id="ingest",
        target_ids=["audit", "metrics"],
        # no selection_func → full broadcast
    )

    return (
        WorkflowBuilder()
        .add_executor(ingest)
        .add_executor(audit)
        .add_executor(metrics)
        .add_edge_group(fan_out)
        .build()
    )
```

### Example 3 — `FanOutEdgeGroup` with `selection_func` for conditional routing

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._edge import FanOutEdgeGroup
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


def route_by_priority(message: dict, targets: list[str]) -> list[str]:
    """Route high-priority messages to the fast lane, all others to standard."""
    if message.get("priority") == "high":
        return [t for t in targets if t == "fast_processor"]
    return [t for t in targets if t == "standard_processor"]


class DispatchExecutor(Executor):
    @handler
    async def on_task(self, task: dict, ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message(task)


class FastProcessorExecutor(Executor):
    @handler
    async def on_task(self, task: dict, ctx: WorkflowContext) -> None:
        print(f"[fast] processing {task['id']}")


class StandardProcessorExecutor(Executor):
    @handler
    async def on_task(self, task: dict, ctx: WorkflowContext) -> None:
        print(f"[standard] processing {task['id']}")


def build_routing_workflow():
    dispatch = DispatchExecutor(id="dispatch")
    fast = FastProcessorExecutor(id="fast_processor")
    standard = StandardProcessorExecutor(id="standard_processor")

    fan_out = FanOutEdgeGroup(
        source_id="dispatch",
        target_ids=["fast_processor", "standard_processor"],
        selection_func=route_by_priority,
        selection_func_name="route_by_priority",  # explicit name for checkpoint restore
    )

    return (
        WorkflowBuilder()
        .add_executor(dispatch)
        .add_executor(fast)
        .add_executor(standard)
        .add_edge_group(fan_out)
        .build()
    )
```

### Example 4 — combining fan-out and fan-in in a single workflow

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._edge import FanInEdgeGroup, FanOutEdgeGroup
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class SplitterExecutor(Executor):
    @handler
    async def on_document(self, doc: dict, ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message(doc)


class SentimentAnalyserExecutor(Executor):
    @handler
    async def on_document(self, doc: dict, ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message({"sentiment": "positive", "doc_id": doc["id"]})


class EntityExtractorExecutor(Executor):
    @handler
    async def on_document(self, doc: dict, ctx: WorkflowContext[dict]) -> None:
        await ctx.send_message({"entities": ["ACME Corp"], "doc_id": doc["id"]})


class AggregatorExecutor(Executor):
    @handler
    async def on_result(self, result: dict, ctx: WorkflowContext[dict, dict]) -> None:
        await ctx.yield_output(result)


def build_parallel_enrichment_workflow():
    splitter = SplitterExecutor(id="splitter")
    sentiment = SentimentAnalyserExecutor(id="sentiment")
    entities = EntityExtractorExecutor(id="entities")
    aggregator = AggregatorExecutor(id="aggregator")

    fan_out = FanOutEdgeGroup(
        source_id="splitter",
        target_ids=["sentiment", "entities"],
    )
    fan_in = FanInEdgeGroup(
        source_ids=["sentiment", "entities"],
        target_id="aggregator",
    )

    return (
        WorkflowBuilder()
        .add_executor(splitter)
        .add_executor(sentiment)
        .add_executor(entities)
        .add_executor(aggregator)
        .add_edge_group(fan_out)
        .add_edge_group(fan_in)
        .build()
    )
```

---

## 3 · `SwitchCaseEdgeGroup` · `SwitchCaseEdgeGroupCase` · `SwitchCaseEdgeGroupDefault`

**Sub-package:** `agent_framework._workflows._edge`  
**Install:** `pip install agent-framework-core`

`SwitchCaseEdgeGroup` extends `FanOutEdgeGroup` to provide a structured switch/case
routing pattern. Cases are evaluated **sequentially**; the first case whose predicate
returns `True` wins and no further cases are evaluated. The default case always fires when
no case matches — it is not optional. The framework enforces a single-default invariant at
construction time and warns when the default is not the last case.

### Class signatures (1.9.0)

```python
from collections.abc import Callable
from typing import Any

class SwitchCaseEdgeGroupCase:
    def __init__(
        self,
        condition: Callable[[Any], bool] | None,
        target_id: str,
        *,
        condition_name: str | None = None,
    ) -> None:
        # If condition is None, a placeholder that raises RuntimeError on call is installed.
        # This supports deserialization where the callable is absent but the name is known.
        ...

    def to_dict(self) -> dict[str, Any]:
        # Returns {"type": "Case", "target_id": ..., "condition_name": ...}
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SwitchCaseEdgeGroupCase":
        # Installs placeholder condition when condition_name is present but no callable
        ...


class SwitchCaseEdgeGroupDefault:
    def __init__(self, target_id: str) -> None: ...

    def to_dict(self) -> dict[str, Any]:
        # Returns {"type": "Default", "target_id": ...}
        ...


class SwitchCaseEdgeGroup(FanOutEdgeGroup):
    def __init__(
        self,
        source_id: str,
        cases: list[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault],
        *,
        id: str | None = None,
    ) -> None:
        # Requires >= 2 cases
        # Requires exactly one SwitchCaseEdgeGroupDefault
        # Warns (not raises) if default is not last
        ...
```

### Key facts

- The internal `selection_func` iterates `cases` in order, calls each `condition`, and
  returns the `[target_id]` of the **first match**. Exceptions raised by a predicate are
  **caught and logged** rather than propagated — so a buggy predicate silently skips its
  case.
- `SwitchCaseEdgeGroupDefault` has no condition; it always matches when no earlier case
  fired. Its `to_dict()` uses `"type": "Default"` as discriminator.
- After `from_dict()` deserialization, conditions are placeholder callables. You must
  re-register the real callables before running the workflow (see Example 4).
- Provide `condition_name` explicitly on `SwitchCaseEdgeGroupCase` so that checkpoint
  restore knows which callable to look up. If you omit it the name is auto-set from
  `condition.__name__`, which fails for lambdas.

### Example 1 — basic switch/case with three cases and a default

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._edge import (
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class ClassifierExecutor(Executor):
    @handler
    async def on_text(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)


class UrgentExecutor(Executor):
    @handler
    async def on_text(self, text: str, ctx: WorkflowContext) -> None:
        print(f"[URGENT] {text}")


class NormalExecutor(Executor):
    @handler
    async def on_text(self, text: str, ctx: WorkflowContext) -> None:
        print(f"[normal] {text}")


class SpamExecutor(Executor):
    @handler
    async def on_text(self, text: str, ctx: WorkflowContext) -> None:
        print(f"[spam filtered] {text}")


class FallbackExecutor(Executor):
    @handler
    async def on_text(self, text: str, ctx: WorkflowContext) -> None:
        print(f"[default] {text}")


def build_switch_workflow():
    classifier = ClassifierExecutor(id="classifier")
    urgent = UrgentExecutor(id="urgent")
    normal = NormalExecutor(id="normal")
    spam = SpamExecutor(id="spam")
    fallback = FallbackExecutor(id="fallback")

    switch = SwitchCaseEdgeGroup(
        source_id="classifier",
        cases=[
            SwitchCaseEdgeGroupCase(
                condition=lambda msg: "URGENT" in msg,
                target_id="urgent",
                condition_name="is_urgent",
            ),
            SwitchCaseEdgeGroupCase(
                condition=lambda msg: "unsubscribe" in msg.lower(),
                target_id="spam",
                condition_name="is_spam",
            ),
            SwitchCaseEdgeGroupCase(
                condition=lambda msg: len(msg) < 200,
                target_id="normal",
                condition_name="is_short",
            ),
            SwitchCaseEdgeGroupDefault(target_id="fallback"),  # must be last
        ],
    )

    return (
        WorkflowBuilder()
        .add_executor(classifier)
        .add_executor(urgent)
        .add_executor(normal)
        .add_executor(spam)
        .add_executor(fallback)
        .add_edge_group(switch)
        .build()
    )
```

### Example 2 — serialization round-trip with `to_dict()` / `from_dict()`

```python
import json
from agent_framework._workflows._edge import (
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)


def is_vip(msg: dict) -> bool:
    return msg.get("tier") == "vip"


def is_trial(msg: dict) -> bool:
    return msg.get("tier") == "trial"


switch = SwitchCaseEdgeGroup(
    source_id="router",
    cases=[
        SwitchCaseEdgeGroupCase(is_vip, "vip_handler", condition_name="is_vip"),
        SwitchCaseEdgeGroupCase(is_trial, "trial_handler", condition_name="is_trial"),
        SwitchCaseEdgeGroupDefault(target_id="default_handler"),
    ],
)

serialized = switch.to_dict()
print(json.dumps(serialized, indent=2))
# {
#   "type": "SwitchCase",
#   "source_id": "router",
#   "cases": [
#     {"type": "Case", "target_id": "vip_handler", "condition_name": "is_vip"},
#     {"type": "Case", "target_id": "trial_handler", "condition_name": "is_trial"},
#     {"type": "Default", "target_id": "default_handler"}
#   ]
# }

restored = SwitchCaseEdgeGroup.from_dict(serialized)
# Conditions are placeholder callables after restore — see Example 4
```

### Example 3 — warning when default is not last

```python
import warnings
from agent_framework._workflows._edge import (
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)

# The framework emits a UserWarning when default is not the final case
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    switch = SwitchCaseEdgeGroup(
        source_id="src",
        cases=[
            SwitchCaseEdgeGroupDefault(target_id="fallback"),   # default FIRST — warns
            SwitchCaseEdgeGroupCase(
                condition=lambda m: m == "ping",
                target_id="pong",
                condition_name="is_ping",
            ),
        ],
    )

if caught:
    print(f"Warning emitted: {caught[0].message}")
    # Warning emitted: SwitchCaseEdgeGroupDefault should be the last case
```

### Example 4 — re-registering conditions after deserialization

```python
from agent_framework._workflows._edge import (
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
)


# Condition registry — keyed by condition_name
CONDITION_REGISTRY: dict[str, callable] = {
    "is_vip": lambda msg: msg.get("tier") == "vip",
    "is_trial": lambda msg: msg.get("tier") == "trial",
}


def restore_switch_group(serialized: dict) -> SwitchCaseEdgeGroup:
    """Restore a SwitchCaseEdgeGroup from a checkpoint and re-inject conditions."""
    group = SwitchCaseEdgeGroup.from_dict(serialized)

    for case in group.cases:
        if isinstance(case, SwitchCaseEdgeGroupCase):
            name = case.condition_name
            real_condition = CONDITION_REGISTRY.get(name)
            if real_condition is None:
                raise RuntimeError(
                    f"Cannot restore condition '{name}': not found in registry. "
                    "Register it in CONDITION_REGISTRY before restoring."
                )
            # Re-install the real callable, preserving condition_name
            case.condition = real_condition

    return group
```

---

## 4 · `TokenBudgetComposedStrategy`

**Sub-package:** `agent_framework._compaction`  
**Install:** `pip install agent-framework-core`

`TokenBudgetComposedStrategy` is a **meta-strategy** that runs an ordered sequence of
compaction strategies until the token budget is satisfied (or, with `early_stop=False`,
until all strategies have been applied). When no strategy brings the conversation within
budget, a two-pass strict fallback excludes non-system groups first and, if that still
exceeds the budget, excludes system groups too.

### Class signature (1.9.0)

```python
from collections.abc import Sequence
from agent_framework._compaction import CompactionStrategy, TokenizerProtocol

class TokenBudgetComposedStrategy:
    def __init__(
        self,
        *,
        token_budget: int,
        tokenizer: TokenizerProtocol,
        strategies: Sequence[CompactionStrategy],
        early_stop: bool = True,
    ) -> None: ...

    def __call__(self, message_groups: list) -> bool:
        # Returns True if any compaction was applied, False if already within budget
        ...
```

### Key facts

- After each strategy runs, `annotate_message_groups` + `annotate_token_counts` are
  re-run to refresh exclusion flags and token counts before the next strategy is attempted.
- `early_stop=True` (default) — stops immediately after the first strategy that satisfies
  the budget. Use this in most production pipelines.
- `early_stop=False` — always runs every strategy in sequence, even if budget is already
  satisfied. Useful for aggressive pre-compaction pipelines where you want all clean-up
  applied regardless.
- Strict fallback path: `reason="token_budget_fallback_strict"` is set on groups excluded
  by the fallback so you can identify them in post-run inspection.
- Returns `False` (no-op) when the current token count is already within `token_budget`
  before any strategy runs.

### Example 1 — `TruncationStrategy` + `SelectiveToolCallCompactionStrategy` under a token budget

```python
import asyncio
from agent_framework._compaction import (
    TokenBudgetComposedStrategy,
    TruncationStrategy,
    SelectiveToolCallCompactionStrategy,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent


def build_budget_agent(model: str = "gpt-4o", token_budget: int = 8192) -> Agent:
    from agent_framework._compaction import CharacterCountTokenizer

    tokenizer = CharacterCountTokenizer(chars_per_token=4)

    compaction = TokenBudgetComposedStrategy(
        token_budget=token_budget,
        tokenizer=tokenizer,
        strategies=[
            # First: remove older tool-call groups
            SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2),
            # Second: if still over budget, truncate oldest non-system groups
            TruncationStrategy(keep_last_groups=10),
        ],
        early_stop=True,  # stop as soon as budget is met
    )

    return Agent(
        client=OpenAIChatClient(model=model),
        name="budget-agent",
        history_compaction=compaction,
    )
```

### Example 2 — using `early_stop=False` to always run all strategies

```python
from agent_framework._compaction import (
    TokenBudgetComposedStrategy,
    SlidingWindowStrategy,
    SelectiveToolCallCompactionStrategy,
    CharacterCountTokenizer,
)

# Use early_stop=False when you want deterministic compaction regardless of current size,
# e.g., in batch pipelines where you want a clean, minimal history every turn.
aggressive_compaction = TokenBudgetComposedStrategy(
    token_budget=4096,
    tokenizer=CharacterCountTokenizer(chars_per_token=4),
    strategies=[
        SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=1),
        SlidingWindowStrategy(keep_last_groups=6, preserve_system=True),
    ],
    early_stop=False,  # always run both strategies
)
```

### Example 3 — custom `TokenizerProtocol` (character estimator)

```python
from agent_framework._compaction import TokenBudgetComposedStrategy, TruncationStrategy


class WordCountTokenizer:
    """Rough tokenizer: 1 token ≈ 0.75 words (GPT-style estimate)."""

    def count_tokens(self, text: str) -> int:
        words = len(text.split())
        return max(1, int(words / 0.75))


compaction = TokenBudgetComposedStrategy(
    token_budget=2048,
    tokenizer=WordCountTokenizer(),
    strategies=[TruncationStrategy(keep_last_groups=8)],
)
```

### Example 4 — inspecting which message groups were excluded after compaction

```python
from agent_framework._compaction import (
    TokenBudgetComposedStrategy,
    TruncationStrategy,
    CharacterCountTokenizer,
    annotate_message_groups,
)


def inspect_compaction_results(message_groups: list) -> None:
    """Run compaction and report which groups were excluded and why."""
    tokenizer = CharacterCountTokenizer(chars_per_token=4)
    strategy = TokenBudgetComposedStrategy(
        token_budget=1000,
        tokenizer=tokenizer,
        strategies=[TruncationStrategy(keep_last_groups=4)],
    )

    applied = strategy(message_groups)
    print(f"Compaction applied: {applied}")

    for i, group in enumerate(message_groups):
        if group.excluded:
            print(
                f"  Group {i} excluded — reason: {group.exclude_reason}, "
                f"role: {group.role}"
            )
        else:
            print(f"  Group {i} included — ~{group.token_count} tokens")
```

---

## 5 · `SelectiveToolCallCompactionStrategy` · `ToolResultCompactionStrategy`

**Sub-package:** `agent_framework._compaction`  
**Install:** `pip install agent-framework-core`

Both strategies target the same problem — tool-call history growing unboundedly — but
solve it differently. `SelectiveToolCallCompactionStrategy` **removes** older tool-call
groups entirely. `ToolResultCompactionStrategy` **replaces** them with a human-readable
inline summary message, preserving a readable audit trail at far lower token cost.

### Class signatures (1.9.0)

```python
class SelectiveToolCallCompactionStrategy:
    def __init__(self, keep_last_tool_call_groups: int = 1) -> None:
        # Raises ValueError if keep_last_tool_call_groups < 0
        ...

    def __call__(self, message_groups: list) -> bool: ...


class ToolResultCompactionStrategy:
    def __init__(self, keep_last_tool_call_groups: int = 1) -> None:
        # Raises ValueError if keep_last_tool_call_groups < 0
        ...

    def __call__(self, message_groups: list) -> bool: ...
```

### Key facts

**`SelectiveToolCallCompactionStrategy`:**
- Marks older tool-call groups `excluded=True, reason="tool_call_compaction"`.
- With `keep_last_tool_call_groups=0` every tool-call group is excluded.
- No summary is inserted — the history is simply shorter after this strategy runs.

**`ToolResultCompactionStrategy`:**
- Also marks older tool-call groups excluded, but **inserts a summary `Message`** at the
  original position: `Message(role="assistant", message_id=f"tool_summary_{group_id}", ...)`.
- Builds a `call_id → function_name` map from `function_call` contents so the summary
  can include human-readable function names alongside truncated results.
- Sets `SUMMARY_OF_MESSAGE_IDS_KEY` and `SUMMARY_OF_GROUP_IDS_KEY` on the summary
  (forward links) and `_set_group_summarized_by_summary_id` on each original group
  (back link) — both sides of the trace are preserved for introspection.
- `keep_last_tool_call_groups=0` drops all tool groups and inserts one combined summary.

### Example 1 — `SelectiveToolCallCompactionStrategy` keeping the last two groups

```python
from agent_framework._compaction import SelectiveToolCallCompactionStrategy
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent, tool


@tool
async def search_database(query: str) -> str:
    return f"[db results for: {query}]"


@tool
async def call_external_api(endpoint: str, payload: dict) -> dict:
    return {"status": "ok", "endpoint": endpoint}


# Keep the 2 most recent tool-call exchanges; discard all older ones completely
compaction = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="db-agent",
    tools=[search_database, call_external_api],
    history_compaction=compaction,
)
```

### Example 2 — `ToolResultCompactionStrategy` replacing old tool groups with inline summaries

```python
from agent_framework._compaction import ToolResultCompactionStrategy
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent, tool


@tool
async def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


@tool
async def write_file(path: str, content: str) -> str:
    return f"Wrote {len(content)} bytes to {path}"


# Keep the last tool-call group verbatim; replace older ones with
# "[Tool results: read_file: <result>; write_file: <result>]" summaries
compaction = ToolResultCompactionStrategy(keep_last_tool_call_groups=1)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="file-agent",
    tools=[read_file, write_file],
    history_compaction=compaction,
)
```

### Example 3 — choosing between the two strategies

```python
from agent_framework._compaction import (
    SelectiveToolCallCompactionStrategy,
    ToolResultCompactionStrategy,
    TokenBudgetComposedStrategy,
    CharacterCountTokenizer,
)

# Use SelectiveToolCallCompactionStrategy when:
#   - You care only about token budget and don't need a history of tool results
#   - The tool calls are deterministic / retryable and you don't need an audit trail
selective = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=1)

# Use ToolResultCompactionStrategy when:
#   - You want the model (and humans) to have a readable summary of past tool interactions
#   - You need to trace which tool calls happened (e.g., for debugging or compliance)
#   - You are fine with a slightly higher token overhead for the summary messages
summarized = ToolResultCompactionStrategy(keep_last_tool_call_groups=1)

# In a composed pipeline, run summarized compaction first to preserve history,
# then fall back to full exclusion if still over budget
budget_strategy = TokenBudgetComposedStrategy(
    token_budget=6000,
    tokenizer=CharacterCountTokenizer(chars_per_token=4),
    strategies=[summarized, selective],
    early_stop=True,
)
```

---

## 6 · `SlidingWindowStrategy` · `SummarizationStrategy`

**Sub-package:** `agent_framework._compaction`  
**Install:** `pip install agent-framework-core`

`SlidingWindowStrategy` is the simplest stateless compaction primitive — it keeps a fixed
window of the most recent non-system message groups and marks older ones excluded.
`SummarizationStrategy` is its LLM-backed counterpart: when the window exceeds a threshold
it calls a chat client to generate a prose summary, inserts that summary into the history,
and marks the summarized groups excluded.

### Class signatures (1.9.0)

```python
from agent_framework._clients import BaseChatClient

class SlidingWindowStrategy:
    def __init__(
        self,
        *,
        keep_last_groups: int,
        preserve_system: bool = True,
    ) -> None:
        # Raises ValueError if keep_last_groups <= 0
        ...

    def __call__(self, message_groups: list) -> bool: ...


class SummarizationStrategy:
    def __init__(
        self,
        *,
        client: BaseChatClient,
        target_count: int = 4,
        threshold: int | None = None,
        prompt: str | None = None,
    ) -> None:
        # target_count must be >= 1
        # threshold must be >= 0 (None treated as 0)
        ...

    def __call__(self, message_groups: list) -> bool:
        # Returns False (and logs warning) if summarizer returns empty text or raises
        ...
```

### Key facts

**`SlidingWindowStrategy`:**
- Marks groups outside the window `excluded=True, reason="sliding_window"`.
- `preserve_system=True` (default) — system message groups are never marked excluded,
  regardless of their position.
- Returns `False` if all non-system groups already fit within `keep_last_groups`.

**`SummarizationStrategy`:**
- Triggers when `included_non_system_message_count > target_count + threshold`.
- Uses `client.get_response()` to generate the summary text; the call is synchronous from
  the caller's perspective (the strategy is a synchronous callable, not async).
- **Fails silently on empty response or exception**: logs a warning and returns `False`
  without raising. Always instrument the compaction pipeline with logging if you rely on
  summarization.
- Bidirectional trace metadata: `SUMMARY_OF_MESSAGE_IDS_KEY` and
  `SUMMARY_OF_GROUP_IDS_KEY` are stored on the summary message, and
  `_set_group_summarized_by_summary_id` is called on the originals.
- Custom `prompt` overrides the default summarization instruction — useful for
  domain-specific summarization (e.g., "Summarize the following medical consultation...").

### Example 1 — `SlidingWindowStrategy` with `preserve_system=True` (default)

```python
from agent_framework._compaction import SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent


# Keep the 8 most recent non-system message groups; always preserve system messages
compaction = SlidingWindowStrategy(keep_last_groups=8, preserve_system=True)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="windowed-agent",
    history_compaction=compaction,
)
```

### Example 2 — `SlidingWindowStrategy` combined with an agent session

```python
import asyncio
from agent_framework._compaction import SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent, AgentSession


async def run_long_conversation():
    compaction = SlidingWindowStrategy(keep_last_groups=6)
    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="chat-agent",
        instructions="You are a helpful assistant.",
        history_compaction=compaction,
    )

    session = AgentSession(agent)
    turns = [
        "Tell me about the history of Rome.",
        "What was the significance of Julius Caesar?",
        "How did the Roman Empire fall?",
        "What replaced it?",
        "Compare it to the Byzantine Empire.",
        "What are the lessons for modern governance?",
    ]
    for turn in turns:
        response = await session.run(turn)
        print(f"User: {turn}")
        print(f"Agent: {response.messages[-1].text[:100]}...")
        print()
```

### Example 3 — `SummarizationStrategy` with a custom prompt

```python
import asyncio
from agent_framework._compaction import SummarizationStrategy, SlidingWindowStrategy
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent

LEGAL_SUMMARY_PROMPT = (
    "You are a legal summarization assistant. "
    "Summarize the following conversation excerpt, preserving all cited statutes, "
    "case references, and decisions. Be concise but legally precise."
)


async def build_legal_agent() -> Agent:
    # A separate client dedicated to summarization (can be a cheaper model)
    summarizer_client = OpenAIChatClient(model="gpt-4o-mini")
    main_client = OpenAIChatClient(model="gpt-4o")

    compaction = SummarizationStrategy(
        client=summarizer_client,
        target_count=6,   # summarize when more than 6 + threshold non-system groups
        threshold=2,      # allows up to 8 non-system groups before triggering
        prompt=LEGAL_SUMMARY_PROMPT,
    )

    return Agent(
        client=main_client,
        name="legal-agent",
        instructions="You are an expert legal research assistant.",
        history_compaction=compaction,
    )
```

### Example 4 — `SummarizationStrategy` inside `TokenBudgetComposedStrategy`

```python
from agent_framework._compaction import (
    SummarizationStrategy,
    SlidingWindowStrategy,
    TokenBudgetComposedStrategy,
    CharacterCountTokenizer,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent


def build_smart_compaction_agent() -> Agent:
    summarizer_client = OpenAIChatClient(model="gpt-4o-mini")
    main_client = OpenAIChatClient(model="gpt-4o")

    tokenizer = CharacterCountTokenizer(chars_per_token=4)

    composed = TokenBudgetComposedStrategy(
        token_budget=12000,
        tokenizer=tokenizer,
        strategies=[
            # Attempt LLM summarization first — preserves the most context
            SummarizationStrategy(client=summarizer_client, target_count=8),
            # Fall back to sliding window if summarization fails or budget still exceeded
            SlidingWindowStrategy(keep_last_groups=6),
        ],
        early_stop=True,
    )

    return Agent(
        client=main_client,
        name="smart-agent",
        instructions="You are a research assistant with long-running context.",
        history_compaction=composed,
    )
```

---

## 7 · `StepWrapper` · `FunctionalWorkflow` · `RunContext`

**Sub-package:** `agent_framework._workflows._functional`  
**Install:** `pip install agent-framework-core`

The `@experimental` functional workflow API lets you build graph-free, step-by-step
pipelines using plain Python async functions. `StepWrapper` is the decorator machinery
behind `@step`; `RunContext` is the context object injected into step functions;
`FunctionalWorkflow` is the assembled workflow. All three carry the `@experimental` marker.

### Class signatures (1.9.0)

```python
from agent_framework._workflows._functional import StepWrapper, RunContext, FunctionalWorkflow
from agent_framework._feature_stage import experimental, ExperimentalFeature

@experimental
class StepWrapper:
    def __init__(
        self,
        func,
        *,
        name: str | None = None,
    ) -> None:
        # Raises TypeError if func is not async
        # name defaults to func.__name__; uses functools.update_wrapper
        ...

    async def __call__(self, *args, **kwargs): ...
    # Caches by (step_name, call_index)
    # On cache hit: emits executor_bypassed event instead of executor_invoked/executor_completed
    # Injects RunContext if parameter annotated as RunContext or named "ctx"
    # Saves checkpoint after each live execution
    # Outside a workflow: transparent delegation to original function


@experimental
class RunContext:
    def get_state(self, key: str, default=None) -> Any: ...
    def set_state(self, key: str, value: Any) -> None: ...
    async def yield_output(self, value: Any) -> None: ...
    async def request_info(self, request_data: Any, response_type: type) -> None: ...


@experimental
class FunctionalWorkflow:
    """Assembled workflow from @workflow + @step decorated functions."""
    ...
```

### Key facts

- `StepWrapper` cache key is `(step_name, call_index)`: if the same step function is
  called twice with the same `call_index` in a replay scenario, the second call is served
  from cache and the `executor_bypassed` event fires instead of `executor_invoked`.
- **Outside a workflow context** (e.g., in unit tests), `StepWrapper.__call__` delegates
  transparently to the original function — no caching, no events, no injection. This means
  you can test step functions in isolation without any framework infrastructure.
- `RunContext` is injected when a parameter is annotated as `RunContext` OR when the
  parameter is named `ctx` regardless of annotation. Use the annotation form in production
  code for clarity.
- Checkpoint storage must be configured on the `FunctionalWorkflowAgent` (or the
  surrounding runner) for `StepWrapper` to save checkpoints between steps.

### Example 1 — basic `@step` + `@workflow` pipeline

```python
import asyncio
from agent_framework._workflows._functional import step, workflow, FunctionalWorkflowAgent
from agent_framework.openai import OpenAIChatClient


@step
async def fetch_data(source_url: str) -> dict:
    # In production this would be an HTTP call
    return {"url": source_url, "data": "sample content"}


@step
async def transform_data(raw: dict) -> dict:
    return {**raw, "transformed": raw["data"].upper()}


@step
async def store_result(processed: dict) -> str:
    key = processed["url"].replace("https://", "").replace("/", "_")
    return f"stored:{key}"


@workflow
async def etl_pipeline(source_url: str) -> str:
    raw = await fetch_data(source_url)
    processed = await transform_data(raw)
    return await store_result(processed)


async def main():
    agent = FunctionalWorkflowAgent(
        workflow=etl_pipeline,
        client=OpenAIChatClient(model="gpt-4o"),
        name="etl-agent",
    )
    result = await agent.run("https://example.com/feed")
    print(result.messages[-1].text)


asyncio.run(main())
```

### Example 2 — step caching on replay (second call bypassed)

```python
import asyncio
from agent_framework._workflows._functional import step, workflow, FunctionalWorkflowAgent
from agent_framework._compaction import InMemoryCheckpointStorage
from agent_framework.openai import OpenAIChatClient


call_count = 0


@step
async def expensive_llm_call(prompt: str) -> str:
    global call_count
    call_count += 1
    # Simulates a costly model call
    return f"[result {call_count}]"


@workflow
async def idempotent_pipeline(prompt: str) -> str:
    result = await expensive_llm_call(prompt)
    return result


async def demonstrate_caching():
    checkpoint_storage = InMemoryCheckpointStorage()
    agent = FunctionalWorkflowAgent(
        workflow=idempotent_pipeline,
        client=OpenAIChatClient(model="gpt-4o"),
        name="cached-agent",
        checkpoint_storage=checkpoint_storage,
    )

    # First run: step executes and checkpoint is saved
    result1 = await agent.run("Hello")
    print(f"Run 1: {result1.messages[-1].text}, call_count={call_count}")

    # Second run from the same checkpoint: step is bypassed (executor_bypassed event fires)
    result2 = await agent.run("Hello")
    print(f"Run 2: {result2.messages[-1].text}, call_count={call_count}")
    # call_count remains 1 — the step was served from cache on the second run
```

### Example 3 — `RunContext` inside a step for state and HITL

```python
import asyncio
from agent_framework._workflows._functional import (
    step,
    workflow,
    RunContext,
    FunctionalWorkflowAgent,
)
from agent_framework.openai import OpenAIChatClient


@step
async def analyse_sentiment(text: str, ctx: RunContext) -> str:
    sentiment = "positive" if "good" in text.lower() else "negative"
    ctx.set_state("last_sentiment", sentiment)
    await ctx.yield_output(f"[intermediate] sentiment: {sentiment}")
    return sentiment


@step
async def request_human_review(
    sentiment: str,
    ctx: RunContext,
) -> str:
    if sentiment == "negative":
        # Ask a human for guidance before proceeding
        await ctx.request_info(
            request_data={"sentiment": sentiment, "guidance_needed": True},
            response_type=str,
        )
    return f"Processed with sentiment: {sentiment}"


@workflow
async def sentiment_pipeline(text: str) -> str:
    sentiment = await analyse_sentiment(text)
    return await request_human_review(sentiment)
```

### Example 4 — using `StepWrapper` directly in unit tests

```python
import asyncio
import pytest
from agent_framework._workflows._functional import StepWrapper


# The function under test
async def _parse_invoice(raw: str) -> dict:
    lines = raw.strip().split("\n")
    return {"line_count": len(lines), "preview": lines[0] if lines else ""}


# Wrap it for production use
parse_invoice = StepWrapper(_parse_invoice, name="parse_invoice")


# In tests: call the wrapper directly — it delegates to the original function
# with no caching, no context injection, and no framework overhead
@pytest.mark.asyncio
async def test_parse_invoice_empty():
    result = await parse_invoice("")
    assert result["line_count"] == 1  # one empty line from split


@pytest.mark.asyncio
async def test_parse_invoice_multiline():
    raw = "Item A: $10\nItem B: $20\nTotal: $30"
    result = await parse_invoice(raw)
    assert result["line_count"] == 3
    assert result["preview"] == "Item A: $10"
```

---

## 8 · `MCPWebsocketTool` · `MCPStreamableHTTPTool`

**Sub-package:** `agent_framework._mcp`  
**Install:** `pip install "agent-framework[mcp]"` · For WebSocket support add `mcp[ws]`

Both classes extend `MCPTool` and override `get_mcp_client()` and
`_mcp_base_span_attributes()`. They expose all standard `MCPTool` configuration options
(tool filtering, approval modes, sampling, task options) while handling the transport
details of WebSocket and streamable HTTP/SSE connections respectively.

### Class signatures (1.9.0)

```python
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

class MCPWebsocketTool(MCPTool):
    def __init__(
        self,
        name: str,
        url: str,
        *,
        tool_name_prefix: str | None = None,
        load_tools: bool = True,
        load_prompts: bool = False,
        allowed_tools: list[str] | None = None,
        approval_mode: ApprovalMode | None = None,
        request_timeout: float | None = None,
        session: Any | None = None,
        sampling_approval_callback: Any | None = None,
        sampling_max_tokens: int | None = None,
        sampling_max_requests: int | None = None,
        task_options: MCPTaskOptions | None = None,
        additional_tool_argument_names: list[str] | None = None,
    ) -> None:
        # Raises ModuleNotFoundError if mcp[ws] is not installed
        # Sets OTel attributes:
        #   network.transport = "tcp"
        #   network.protocol.name = "websocket"
        # Parses URL for address/port: wss:// defaults to 443, ws:// to 80
        ...


class MCPStreamableHTTPTool(MCPTool):
    def __init__(
        self,
        name: str,
        url: str,
        *,
        terminate_on_close: bool | None = None,
        http_client: Any | None = None,
        header_provider: Callable[[dict[str, Any]], dict[str, str]] | None = None,
        # + all MCPTool common kwargs (same as MCPWebsocketTool above)
        **kwargs,
    ) -> None:
        # header_provider: receives runtime kwargs from FunctionInvocationContext
        # Uses contextvars.ContextVar (_mcp_call_headers) for per-call header injection
        # without global state — safe for concurrent async calls
        ...
```

### Key facts

| Feature | `MCPWebsocketTool` | `MCPStreamableHTTPTool` |
|---|---|---|
| Transport | WebSocket (`ws://`, `wss://`) | HTTP/SSE (`http://`, `https://`) |
| Install extra | `mcp[ws]` | none (included in `mcp`) |
| Default port | 443 for `wss://`, 80 for `ws://` | n/a (from URL) |
| OTel `network.transport` | `"tcp"` | HTTP transport |
| `header_provider` | N/A | `Callable[[dict], dict[str, str]]` — per-call auth injection |
| Long-running tasks | via `task_options` | via `task_options` |

- `header_provider` receives the **runtime kwargs** from the tool invocation context
  (e.g., `request_id`, `session_id`, middleware-injected values) and returns a
  `dict[str, str]` of HTTP headers to include in that specific call. Headers are injected
  only for the target origin via an httpx event hook, preventing header leakage to other
  hosts.

### Example 1 — `MCPWebsocketTool` connecting to a real-time service

```python
import asyncio
from agent_framework._mcp import MCPWebsocketTool
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

# Requires: pip install "agent-framework[mcp]" mcp[ws]

realtime_tool = MCPWebsocketTool(
    name="realtime_analytics",
    url="wss://analytics.internal.example.com:8443/mcp",
    tool_name_prefix="analytics_",
    load_tools=True,
    load_prompts=False,
    request_timeout=30.0,
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="analytics-agent",
    tools=[realtime_tool],
    instructions="You have access to real-time analytics tools.",
)


async def main():
    response = await agent.run("What are the top 5 events in the last hour?")
    print(response.messages[-1].text)


asyncio.run(main())
```

### Example 2 — `MCPStreamableHTTPTool` with a static auth token in the URL

```python
from agent_framework._mcp import MCPStreamableHTTPTool
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

# Simple case: embed a long-lived token directly in the URL
code_analysis_tool = MCPStreamableHTTPTool(
    name="code_analysis",
    url="https://mcp.codetools.example.com/v1/mcp?api_key=sk-prod-abc123",
    tool_name_prefix="code_",
    load_tools=True,
    allowed_tools=["lint", "format", "analyse_complexity"],
    request_timeout=60.0,
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="code-agent",
    tools=[code_analysis_tool],
)
```

### Example 3 — `MCPStreamableHTTPTool` with `header_provider` for per-call auth

```python
import asyncio
from typing import Any
from agent_framework._mcp import MCPStreamableHTTPTool
from agent_framework import Agent, FunctionMiddleware, FunctionInvocationContext
from agent_framework.openai import OpenAIChatClient
from collections.abc import Callable, Awaitable


class TokenInjectionMiddleware(FunctionMiddleware):
    """Injects a per-request JWT into the invocation context kwargs."""

    def __init__(self, token_factory: Callable[[], str]) -> None:
        self._token_factory = token_factory

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        context.additional_kwargs["auth_token"] = self._token_factory()
        await call_next()


def header_provider(runtime_kwargs: dict[str, Any]) -> dict[str, str]:
    """Extracts auth token from runtime context and returns it as an HTTP header."""
    token = runtime_kwargs.get("auth_token", "")
    return {"Authorization": f"Bearer {token}"}


def get_rotating_token() -> str:
    # In production this would call a token service or read from a ContextVar
    return "eyJhbGciOiJSUzI1NiJ9.example-token"


secure_tool = MCPStreamableHTTPTool(
    name="secure_data",
    url="https://secure-mcp.example.com/mcp",
    header_provider=header_provider,
    load_tools=True,
    request_timeout=45.0,
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="secure-agent",
    tools=[secure_tool],
    middleware=[TokenInjectionMiddleware(get_rotating_token)],
)
```

### Example 4 — multi-server agent combining WebSocket + HTTP MCP tools

```python
import asyncio
from agent_framework._mcp import MCPWebsocketTool, MCPStreamableHTTPTool
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

# Real-time stream tool via WebSocket
stream_tool = MCPWebsocketTool(
    name="live_stream",
    url="wss://stream.internal.example.com/mcp",
    tool_name_prefix="stream_",
    load_tools=True,
    request_timeout=15.0,
)

# Document processing via HTTP/SSE
doc_tool = MCPStreamableHTTPTool(
    name="doc_processor",
    url="https://docs.internal.example.com/mcp",
    tool_name_prefix="doc_",
    load_tools=True,
    allowed_tools=["parse_pdf", "extract_tables", "summarise_document"],
    request_timeout=120.0,
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="multi-mcp-agent",
    tools=[stream_tool, doc_tool],
    instructions=(
        "You can access both a real-time event stream and a document processing service. "
        "Use stream_ tools for live data and doc_ tools for document analysis."
    ),
)


async def main():
    response = await agent.run(
        "Summarise today's top events from the stream and cross-reference with "
        "the quarterly report PDF."
    )
    print(response.messages[-1].text)


asyncio.run(main())
```

---

## 9 · `MCPTaskOptions`

**Sub-package:** `agent_framework._mcp`  
**Install:** `pip install "agent-framework[mcp]"`

`MCPTaskOptions` is a frozen dataclass (decorated with `@experimental`) that configures
long-running MCP task behaviour per the SEP-2663 specification. When a server advertises
`execution.taskSupport == "required"`, the framework drives a `tools/call` →
`tasks/get` (poll) → `tasks/result` lifecycle instead of a synchronous call-and-response.
`MCPTaskOptions` governs the client-side timeout, TTL hint to the server, and cancellation
semantics.

### Class signature (1.9.0)

```python
from dataclasses import dataclass
from datetime import timedelta
from agent_framework._feature_stage import experimental, ExperimentalFeature

@experimental(feature_id=ExperimentalFeature.MCP_LONG_RUNNING_TASKS)
@dataclass(frozen=True)
class MCPTaskOptions:
    default_ttl: timedelta | None = None
    cancel_remote_task_on_local_cancellation: bool = True
    max_task_wait: timedelta | None = None

    def __post_init__(self) -> None:
        # Raises ValueError if default_ttl is set and not positive
        # Raises ValueError if max_task_wait is set and not positive
        ...
```

### Key facts

| Attribute | Type | Default | Meaning |
|---|---|---|---|
| `default_ttl` | `timedelta \| None` | `None` | Sent to server as `params.task.ttl` (ms). Controls how long the server retains task records **after** terminal status. |
| `cancel_remote_task_on_local_cancellation` | `bool` | `True` | Sends `tasks/cancel` to the server on `asyncio.CancelledError`. |
| `max_task_wait` | `timedelta \| None` | `None` | Client-side deadline. Exceeding it raises `ToolExecutionException` and fires `tasks/cancel`. |

- `cancel_remote_task_on_local_cancellation=False` only controls the `CancelledError` path.
  Abandonment paths (e.g., agent shutdown) still send `tasks/cancel` regardless — the flag
  does not bypass all remote cancellation.
- `MCPTaskOptions` is **frozen** — create a new instance rather than mutating an existing one.
- Pass `task_options=MCPTaskOptions(...)` to any `MCPTool` constructor. It applies to all
  tool calls made through that tool instance.

### Example 1 — basic `MCPTaskOptions` with TTL and max wait

```python
from datetime import timedelta
from agent_framework._mcp import MCPTaskOptions, MCPStreamableHTTPTool
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

task_options = MCPTaskOptions(
    default_ttl=timedelta(minutes=10),    # server keeps result for 10 min after completion
    max_task_wait=timedelta(minutes=5),   # client gives up after 5 min
    cancel_remote_task_on_local_cancellation=True,
)

long_running_tool = MCPStreamableHTTPTool(
    name="code_analysis",
    url="https://analysis.example.com/mcp",
    task_options=task_options,
    load_tools=True,
)

agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="analysis-agent",
    tools=[long_running_tool],
)
```

### Example 2 — disabling remote cancellation on local cancellation

```python
from datetime import timedelta
from agent_framework._mcp import MCPTaskOptions, MCPStreamableHTTPTool

# Use cancel_remote_task_on_local_cancellation=False when:
#   - The server task is idempotent and can be polled again if the client reconnects
#   - You want to abandon the wait without interrupting server-side work
#     (e.g., the task is a batch job that should complete even if the client disconnects)
fire_and_forget_options = MCPTaskOptions(
    default_ttl=timedelta(hours=2),
    max_task_wait=timedelta(seconds=30),  # wait at most 30s before abandoning the poll
    cancel_remote_task_on_local_cancellation=False,
)

batch_tool = MCPStreamableHTTPTool(
    name="batch_processor",
    url="https://batch.example.com/mcp",
    task_options=fire_and_forget_options,
    load_tools=True,
)
```

### Example 3 — `MCPTaskOptions` with `MCPStreamableHTTPTool` for a long-running code analysis pipeline

```python
import asyncio
from datetime import timedelta
from agent_framework._mcp import MCPTaskOptions, MCPStreamableHTTPTool
from agent_framework.exceptions import ToolExecutionException
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def run_code_analysis(repo_url: str) -> str:
    task_options = MCPTaskOptions(
        default_ttl=timedelta(minutes=30),
        max_task_wait=timedelta(minutes=15),
        cancel_remote_task_on_local_cancellation=True,
    )

    analysis_tool = MCPStreamableHTTPTool(
        name="repo_analyser",
        url="https://static-analysis.example.com/mcp",
        tool_name_prefix="analysis_",
        load_tools=True,
        allowed_tools=["analyse_repo", "check_security", "measure_coverage"],
        task_options=task_options,
        request_timeout=10.0,  # connection timeout (separate from task wait)
    )

    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="repo-analysis-agent",
        tools=[analysis_tool],
        instructions="You are a code quality expert. Analyse the repository thoroughly.",
    )

    try:
        response = await agent.run(
            f"Run a full security and coverage analysis on the repository at {repo_url}."
        )
        return response.messages[-1].text
    except ToolExecutionException as exc:
        # max_task_wait exceeded or server reported a terminal error
        return f"Analysis failed or timed out: {exc}"


asyncio.run(run_code_analysis("https://github.com/example/my-service"))
```

---

## 10 · `AgentResponseUpdate` · `ChatResponseUpdate` · `ContinuationToken`

**Sub-package:** `agent_framework._types`  
**Install:** `pip install agent-framework-core`

These three types form the streaming output layer of the framework. `AgentResponseUpdate`
is the chunk type from `Agent.run(stream=True)`. `ChatResponseUpdate` is the chunk type
from `BaseChatClient.get_response(stream=True)`. `ContinuationToken` is the opaque
resumption token that allows long-running or paused agents to be continued across process
boundaries.

### Class signatures (1.9.0)

```python
from typing import Any, TypedDict
from agent_framework._types import SerializationMixin

class AgentResponseUpdate(SerializationMixin):
    def __init__(
        self,
        *,
        contents: list[Any] | None = None,
        role: str | None = None,
        author_name: str | None = None,
        agent_id: str | None = None,
        response_id: str | None = None,
        message_id: str | None = None,
        created_at: Any | None = None,
        finish_reason: str | None = None,
        continuation_token: "ContinuationToken | None" = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
    ) -> None: ...

    @property
    def text(self) -> str:
        # Concatenates content.text for all contents where content.type == "text"
        ...

    @property
    def user_input_requests(self) -> list[Any]:
        # Filters contents by content.user_input_request
        ...

    DEFAULT_EXCLUDE: ClassVar[set[str]] = {"raw_representation"}
    # raw_representation is excluded from to_dict() / to_json()


class ChatResponseUpdate(SerializationMixin):
    # Same fields as AgentResponseUpdate except:
    #   adds: conversation_id, model
    #   omits: agent_id, user_input_requests
    @property
    def text(self) -> str: ...


class ContinuationToken(TypedDict):
    # TypedDict — opaque by design
    # Each provider subclasses with its own fields
    # JSON-serializable: safe for json.dumps / json.loads round-trip
    ...
```

### Key facts

| Field | `AgentResponseUpdate` | `ChatResponseUpdate` | Notes |
|---|---|---|---|
| `.text` | ✓ | ✓ | Concatenated text from all `type=="text"` content items |
| `author_name` | ✓ | — | Identifies agent in multi-agent scenarios |
| `agent_id` | ✓ | — | ID of the agent that produced this chunk |
| `conversation_id` | — | ✓ | Conversation ID from the underlying client |
| `model` | — | ✓ | Model name from the underlying client |
| `finish_reason` | ✓ | ✓ | `"stop"`, `"length"`, `"tool_calls"`, `None` (mid-stream) |
| `continuation_token` | ✓ | ✓ | `None` while streaming; non-`None` only when operation can be resumed |
| `user_input_requests` | ✓ | — | HITL requests embedded in the stream |

- `continuation_token` being non-`None` **does not** mean the stream is complete — it
  signals that the operation can be resumed from this point. A `None` token typically
  means the stream is still in progress or the operation completed normally.
- `ContinuationToken` is opaque: do not inspect its fields; treat it as a black box and
  pass it back unchanged to the same agent's `run()` method.
- `author_name` is propagated to the final `Message` when a stream of updates is
  assembled into an `AgentResponse`, making it available for post-run attribution.

### Example 1 — consuming `AgentResponseUpdate` stream with `finish_reason` check

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def stream_agent_response(prompt: str) -> str:
    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="streaming-agent",
    )

    full_text = []
    async for update in await agent.run(prompt, stream=True):
        if update.text:
            full_text.append(update.text)
            print(update.text, end="", flush=True)

        if update.finish_reason == "stop":
            print()  # newline after stream ends
            break
        elif update.finish_reason == "length":
            print("\n[truncated: token limit reached]")
            break
        elif update.finish_reason == "tool_calls":
            # Model is invoking a tool; framework will continue the loop
            pass

    return "".join(full_text)


asyncio.run(stream_agent_response("Explain the difference between TCP and UDP."))
```

### Example 2 — multi-agent scenario tracking `author_name`

```python
import asyncio
from agent_framework import Agent
from agent_framework_orchestrations import HandoffBuilder
from agent_framework.openai import OpenAIChatClient


async def run_multi_agent_stream(user_query: str) -> None:
    researcher = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="researcher",
        instructions="You research topics and gather facts.",
    )
    writer = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="writer",
        instructions="You write polished prose from research notes.",
    )

    orchestration = (
        HandoffBuilder(agents=[researcher, writer])
        .build()
    )

    per_author: dict[str, list[str]] = {}

    async for update in await orchestration.run(user_query, stream=True):
        author = update.author_name or "unknown"
        per_author.setdefault(author, []).append(update.text)
        if update.text:
            print(f"[{author}] {update.text}", end="", flush=True)

    print("\n\n--- attribution summary ---")
    for author, chunks in per_author.items():
        print(f"{author}: {sum(len(c) for c in chunks)} chars")
```

### Example 3 — `ChatResponseUpdate` stream from a raw client

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._types import Message


async def stream_raw_client(prompt: str) -> str:
    client = OpenAIChatClient(model="gpt-4o")
    messages = [Message(role="user", content=prompt)]

    full_text = []
    conversation_id = None
    model_used = None

    async for update in await client.get_response(messages, stream=True):
        if update.text:
            full_text.append(update.text)
        if update.conversation_id and conversation_id is None:
            conversation_id = update.conversation_id
        if update.model:
            model_used = update.model

    print(f"Model: {model_used}, conversation_id: {conversation_id}")
    return "".join(full_text)


asyncio.run(stream_raw_client("What is the boiling point of water?"))
```

### Example 4 — resuming a long-running agent with `ContinuationToken`

```python
import asyncio
import json
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework._types import ContinuationToken


async def start_and_pause(prompt: str) -> tuple[str, str | None]:
    """Start a long-running task; capture the continuation token for later resumption."""
    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        name="long-task-agent",
        instructions="You perform multi-step analysis tasks.",
    )

    token_json: str | None = None
    text_so_far = []

    async for update in await agent.run(prompt, stream=True):
        if update.text:
            text_so_far.append(update.text)

        if update.continuation_token is not None:
            # Serialize the token so it can be persisted (e.g., to a database)
            token_json = json.dumps(update.continuation_token)
            # In a real scenario you might break here and resume in a new process
            break

    return "".join(text_so_far), token_json


async def resume_from_token(agent: Agent, token_json: str) -> str:
    """Resume the agent from a previously captured continuation token."""
    token: ContinuationToken = json.loads(token_json)

    text_chunks = []
    async for update in await agent.run(
        "",  # prompt is ignored when resuming from a token
        stream=True,
        continuation_token=token,
    ):
        if update.text:
            text_chunks.append(update.text)

    return "".join(text_chunks)


async def main():
    partial_text, token_json = await start_and_pause(
        "Write a 2000-word technical overview of transformer architectures."
    )
    print(f"Partial output ({len(partial_text)} chars) captured.")

    if token_json:
        agent = Agent(
            client=OpenAIChatClient(model="gpt-4o"),
            name="long-task-agent",
            instructions="You perform multi-step analysis tasks.",
        )
        remainder = await resume_from_token(agent, token_json)
        print(f"Resumed and received {len(remainder)} additional chars.")


asyncio.run(main())
```

---

## Quick-reference summary

| Class / item | Module | Stable? | Key use |
|---|---|---|---|
| `WorkflowContext[OutT, W_OutT]` | `_workflows._workflow_context` | ✓ | Per-executor send/yield/HITL/state interface |
| `FanInEdgeGroup` | `_workflows._edge` | ✓ | Converge ≥2 upstream sources onto one target |
| `FanOutEdgeGroup` | `_workflows._edge` | ✓ | Broadcast one source to ≥2 targets with optional `selection_func` |
| `SwitchCaseEdgeGroup` | `_workflows._edge` | ✓ | Sequential predicate routing; exactly one default required |
| `SwitchCaseEdgeGroupCase` | `_workflows._edge` | ✓ | Single case with `condition` callable; placeholder on deserialization |
| `SwitchCaseEdgeGroupDefault` | `_workflows._edge` | ✓ | Unconditional fallback; `"type": "Default"` in `to_dict()` |
| `TokenBudgetComposedStrategy` | `_compaction` | ✓ | Compose strategies under a token budget; two-pass strict fallback |
| `SelectiveToolCallCompactionStrategy` | `_compaction` | ✓ | Exclude older tool-call groups entirely |
| `ToolResultCompactionStrategy` | `_compaction` | ✓ | Replace older tool-call groups with inline summary messages |
| `SlidingWindowStrategy` | `_compaction` | ✓ | Keep last N non-system groups; simple stateless window |
| `SummarizationStrategy` | `_compaction` | ✓ | LLM-backed summarization; silent failure on empty/error |
| `StepWrapper` | `_workflows._functional` | ✗ experimental | `@step` decorator machinery; caches by `(name, call_index)` |
| `FunctionalWorkflow` | `_workflows._functional` | ✗ experimental | Assembled `@workflow` + `@step` pipeline |
| `RunContext` | `_workflows._functional` | ✗ experimental | State/HITL/yield inside a `@step` function |
| `MCPWebsocketTool` | `_mcp` | ✓ | WebSocket MCP transport; requires `mcp[ws]` |
| `MCPStreamableHTTPTool` | `_mcp` | ✓ | HTTP/SSE MCP transport; `header_provider` for per-call auth |
| `MCPTaskOptions` | `_mcp` | ✗ experimental | SEP-2663 long-running task lifecycle config |
| `AgentResponseUpdate` | `_types` | ✓ | Streaming chunk from `Agent.run(stream=True)` |
| `ChatResponseUpdate` | `_types` | ✓ | Streaming chunk from `BaseChatClient.get_response(stream=True)` |
| `ContinuationToken` | `_types` | ✓ | Opaque resumption token; JSON-serializable |

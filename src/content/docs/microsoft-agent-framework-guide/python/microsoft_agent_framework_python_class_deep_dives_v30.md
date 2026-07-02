---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 30"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.10.0: WorkflowViz (DOT/Mermaid/SVG export — to_digraph include_internal_executors, to_mermaid, export format dispatch, save_svg/save_png/save_pdf + IPython.display for Jupyter rendering); FunctionalWorkflow+FunctionalWorkflowAgent (functional workflow execution — @workflow decorator, streaming run() overloads, checkpoint_storage= per-run override, HITL resume via responses=+checkpoint_id=, FunctionalWorkflowAgent as agent-compatible adapter); StepWrapper+RunContext (step caching/replay — (step_name, call_index) cache key, executor_bypassed vs executed events, RunContext.request_info() HITL suspension, get_state/set_state per-run KV, add_event custom events); CompactionProvider+ContextWindowCompactionStrategy (two-phase compaction pipeline — before_strategy/after_strategy phases, DEFAULT_TOOL_EVICTION_THRESHOLD=0.5/DEFAULT_TRUNCATION_THRESHOLD=0.8, tokenizer override, keep_last_tool_call_groups); SlidingWindowStrategy+SelectiveToolCallCompactionStrategy+CharacterEstimatorTokenizer (targeted compaction — keep_last_groups+preserve_system, keep_last_tool_call_groups=0 to remove all, 4-char/token heuristic); AGUIChatClient+AGUIEventConverter (AG-UI chat integration — multi-layer MRO FunctionInvocationLayer+ChatMiddlewareLayer+ChatTelemetryLayer, thread_id continuity via additional_properties, convert_event 10+ event-type dispatch); ClassSkill+AggregatingSkillsSource+FilteringSkillsSource+DelegatingSkillsSource (skills composition — @ClassSkill.resource/@ClassSkill.script decorators, resources+scripts auto-discovery, decorator stack AggregatingSkillsSource→FilteringSkillsSource→DelegatingSkillsSource); BackgroundAgentsProvider+BackgroundTaskInfo+BackgroundTaskStatus (background task delegation — 6 tools exposed to LLM, source_id session key, wait_for_first_completion, continue_task resume, BackgroundTaskStatus RUNNING/COMPLETED/FAILED/LOST); TodoItem+TodoFileStore+TodoInput+TodoCompleteInput (structured todo tracking — SessionContext source_id, TodoFileStore persistent store, TodoItem status lifecycle, TodoProvider toolset); AgentEvalConverter+ConversationSplitter+EvalResults+EvalItem+CheckResult (evaluation pipeline — convert_message typed content, ConversationSplitter callable protocol + ConversationSplit.LAST_TURN built-in strategy, EvalResults sub_results per-agent breakdown, EvalItem query/response properties, CheckResult pass/reason/check_name) — source-verified at agent-framework 1.10.0 / 30 volumes / 300+ classes."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 53
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 30

Verified against **agent-framework 1.10.0** / **agent-framework-ag-ui 1.0.0rc7** (installed July 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`. Sub-packages introspected:
`agent_framework._workflows._viz`,
`agent_framework._workflows._functional`,
`agent_framework._compaction`,
`agent_framework.ag_ui`,
`agent_framework._skills`,
`agent_framework._harness._background_agents`,
`agent_framework._harness._todo`,
`agent_framework._evaluation`.

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
- [Vol. 20](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v20/) — capability Protocols, feature staging, embedding DTOs, WorkflowEventSource, SubWorkflowRequestMessage, RequestInfoMixin, WorkflowAgent.RequestInfoFunctionArgs, EdgeGroupDeliveryStatus, IntegrityLabel+LabelTrackingFunctionMiddleware, MiddlewareTermination+WorkflowConvergenceException
- [Vol. 21](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v21/) — WorkflowContext, FanInEdgeGroup+FanOutEdgeGroup, SwitchCaseEdgeGroup, compaction strategy hierarchy, StepWrapper+FunctionalWorkflow+RunContext, MCPWebsocketTool+MCPStreamableHTTPTool, MCPTaskOptions, AgentResponseUpdate+ContinuationToken
- [Vol. 22](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v22/) — declarative workflow internals
- [Vol. 23](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/) — `DeclarativeActionExecutor`, `DeclarativeWorkflowState`, `DeclarativeEnvConfig`, condition+foreach+break/continue executors, basic variable executors, `AgentManifest`+`PromptAgent`, `Property`+`PropertySchema`, `Connection` hierarchy, `McpTool`+approval modes, `Model`+`ModelOptions`+`Template`, `InvokeAzureAgentExecutor`
- [Vol. 24](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v24/) — `_workflows._typing_utils`, `_workflows._checkpoint_encoding`, `_workflows._runner`, `_harness._loop`, `_harness._tool_approval`, orchestrations protocol utils, Magentic observability, Foundry/OpenAI raw clients, `_workflows` message utilities
- [Vol. 25](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v25/) — `FunctionTool`+`OpenApiTool`+`WebSearchTool`+`FileSearchTool`+`CodeInterpreterTool`+`Binding`, `AgentFactory`+`DeclarativeLoaderError`+`ProviderLookupError`, `WorkflowFactory`+`DeclarativeWorkflowBuilder`, `QuestionExecutor`+`RequestExternalInputExecutor`, `HttpRequestActionExecutor`, `InvokeMcpToolActionExecutor`, `BaseToolExecutor`+`InvokeFunctionToolExecutor`, `JoinExecutor`+termination nodes, `ActionComplete`+`ActionTrigger`+`DeclarativeStateData`, `ClearAllVariablesExecutor`+`EditTableExecutor`+`ResetVariableExecutor`+`SetTextVariableExecutor`
- [Vol. 26](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v26/) — `DurableAIAgentWorker`+`DurableAIAgentClient`, `DurableAIAgent`+`DurableAgentExecutor`, `DurableAIAgentOrchestrationContext`, `AgentEntity`+`AgentEntityStateProviderMixin`, `AgentCallbackContext`+`AgentResponseCallbackProtocol`, `RunRequest`, `AgentSessionId`+`DurableAgentSession`, `DurableAgentState`+`DurableAgentStateData`, entry hierarchy+`DurableAgentStateUsage`, content hierarchy+`DurableStateFields`+`ContentTypes`
- [Vol. 27](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v27/) — `ContentLabel`+`combine_labels`+`check_confidentiality_allowed`, `store_untrusted_content`+`get_security_tools`+security tool constructors, `enable/disable_instrumentation`+`enable_sensitive_telemetry`, `create_resource`+`create_metric_views`+histogram boundary constants, `get_tracer`+`get_meter`+`INNER_ACCUMULATED_USAGE`, `create_mcp_client_span`+`set_mcp_span_error`, `group_messages`+`annotate_message_groups`, `apply_compaction`+`project_included_messages`, `normalize_messages`+`merge_chat_options`, `normalize_tools`+`validate_chat_options`+`add_usage_details`
- [Vol. 28](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v28/) — `RawClaudeAgent`+`ClaudeAgent`+`ClaudeAgentOptions`+`ClaudeAgentSettings`, `AgentFrameworkAgent`+`AgentConfig`, `AGUIThreadSnapshot`+`AGUIThreadSnapshotStore`+`InMemoryAGUIThreadSnapshotStore`, `PredictiveStateHandler`+`PredictStateConfig`, `AGUIRequest`+`AGUIChatOptions`+`AgentState`+`RunMetadata`, `A2AAgentSession`+`A2AContinuationToken`, `ThreadItemConverter`, `AgentApprovalExecutor`+`AgentRequestInfoExecutor`+`AgentRequestInfoResponse`, `AgentFrameworkWorkflow`, `FlowState`+`AGUIHttpService`+`run_workflow_stream`
- [Vol. 29](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v29/) — `SecureMCPToolProxy`+`apply_mcp_security_labels`, `BedrockGuardrailConfig`+`BedrockChatOptions.guardrailConfig`, `GroupChatRequestMessage`+`GroupChatRequestSentEvent`+`GroupChatResponseReceivedEvent`, `MagenticOrchestratorEvent`+`MagenticOrchestratorEventType`+`MagenticProgressLedgerItem`, `ReleaseCandidateFeature`+`ExperimentalWarning`+`FeatureStageWarning`, `ToolExecutionException`+`_MCPTaskAbandoned`+`_MCPDeadlineExpired`, `AgentMiddlewareLayer`+`ChatMiddlewareLayer`, `DiscoveryResponse`+`EntityInfo`+`AgentFrameworkRequest`+`OpenAIError`, `GroupChatState`+`AgentOrchestrationOutput`+`create_completion_message`+`clean_conversation_for_handoff`, `RawGitHubCopilotAgent`+`GitHubCopilotSettings`+`GitHubCopilotOptions`

This volume covers **ten class groups** across workflow visualization, functional workflow execution, context-window compaction, AG-UI chat integration, the skills composition system, background agent delegation, the todo harness, and the evaluation pipeline. All examples verified against `agent-framework==1.10.0`.

| # | Class / group | Module |
|---|---|---|
| 1 | `WorkflowViz` | `agent_framework._workflows._viz` |
| 2 | `FunctionalWorkflow` · `FunctionalWorkflowAgent` | `agent_framework._workflows._functional` |
| 3 | `StepWrapper` · `RunContext` | `agent_framework._workflows._functional` |
| 4 | `CompactionProvider` · `ContextWindowCompactionStrategy` | `agent_framework._compaction` |
| 5 | `SlidingWindowStrategy` · `SelectiveToolCallCompactionStrategy` · `CharacterEstimatorTokenizer` | `agent_framework._compaction` |
| 6 | `AGUIChatClient` · `AGUIEventConverter` | `agent_framework.ag_ui` |
| 7 | `ClassSkill` · `AggregatingSkillsSource` · `FilteringSkillsSource` · `DelegatingSkillsSource` | `agent_framework._skills` |
| 8 | `BackgroundAgentsProvider` · `BackgroundTaskInfo` · `BackgroundTaskStatus` | `agent_framework._harness._background_agents` |
| 9 | `TodoItem` · `TodoFileStore` · `TodoInput` · `TodoCompleteInput` | `agent_framework._harness._todo` |
| 10 | `AgentEvalConverter` · `ConversationSplitter` · `EvalResults` · `EvalItem` · `CheckResult` | `agent_framework._evaluation` |

---

## 1 · `WorkflowViz`

**Module:** `agent_framework._workflows._viz`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework import WorkflowViz`

`WorkflowViz` is the built-in workflow visualization helper. It wraps a compiled `Workflow` object and exposes three output formats — Graphviz DOT, Mermaid diagram syntax, and rendered image files (SVG/PNG/PDF). The constructor stores a reference to the workflow; no graph traversal happens until you call an output method.

### Constructor

```python
WorkflowViz(workflow: Workflow)
```

- **`workflow`** — the compiled `Workflow` instance to visualize.

### `to_digraph(include_internal_executors=False) → str`

Returns the workflow as a DOT-language string. Set `include_internal_executors=True` to expose framework-internal nodes (fan-out routing, merge nodes) that are normally hidden.

```python
from agent_framework import Workflow, WorkflowBuilder
from agent_framework import WorkflowViz

builder = WorkflowBuilder()
builder.add_executor("fetch", fetch_agent)
builder.add_executor("summarise", summary_agent)
builder.add_edge("fetch", "summarise")
wf = builder.build()

viz = WorkflowViz(wf)
dot_src = viz.to_digraph()
print(dot_src)
# digraph Workflow {
#   rankdir=TD;
#   node [shape=box, style=filled, fillcolor=lightblue];
#   ...
# }
```

### `to_mermaid(include_internal_executors=False) → str`

Returns Mermaid flowchart syntax (`flowchart TD`). Paste directly into any Mermaid renderer (GitHub Markdown, Notion, Mermaid Live Editor).

```python
mermaid_src = viz.to_mermaid()
print(mermaid_src)
# flowchart TD
#   fetch["fetch"] --> summarise["summarise"]
```

### `export(format="svg", filename=None, include_internal_executors=False) → str`

Renders the workflow to a file using the system `graphviz` binary. Returns the output file path. Supported formats: `"svg"`, `"png"`, `"pdf"`, `"dot"`.

- When `format="dot"`, no graphviz binary is needed — the DOT string is written directly.
- When `filename=None`, a temporary file is created and its path returned.
- Raises `ImportError` if the `graphviz` Python package is not installed (install with `pip install graphviz>=0.20.0`).

```python
path = viz.export(format="svg", filename="pipeline.svg")
print(f"Saved to: {path}")

# Render all internal routing nodes for debugging
debug_path = viz.export(
    format="svg",
    filename="pipeline_debug.svg",
    include_internal_executors=True,
)
```

### `save_svg` / `save_png` / `save_pdf` — convenience savers

Each is a thin wrapper around `export(format=…)` that returns the saved file path. Use `save_svg` + `IPython.display.SVG` to render inline in Jupyter — `WorkflowViz` has no `display()` method.

```python
# In a Jupyter notebook cell:
from agent_framework import WorkflowViz
from IPython.display import SVG, display

viz = WorkflowViz(wf)
svg_path = viz.save_svg("pipeline.svg")               # returns path string
display(SVG(svg_path))                                 # render inline

# Include internal routing executors:
svg_path2 = viz.save_svg("pipeline_full.svg", include_internal_executors=True)
display(SVG(svg_path2))
```

---

## 2 · `FunctionalWorkflow` + `FunctionalWorkflowAgent`

**Module:** `agent_framework._workflows._functional`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._workflows._functional import FunctionalWorkflow, FunctionalWorkflowAgent`
**Decorator:** `@experimental(feature_id=ExperimentalFeature.FUNCTIONAL_WORKFLOWS)`

`FunctionalWorkflow` is the class produced by the `@workflow` decorator. Unlike graph-based `Workflow` objects, it runs the decorated `async` function directly — no edge wiring, no graph compilation. Branching, parallelism, and loops use native Python control flow.

`FunctionalWorkflowAgent` wraps a `FunctionalWorkflow` so it can be used anywhere an agent-compatible object (with `.run()`) is expected, including inside multi-agent orchestrations.

### `FunctionalWorkflow` constructor

```python
FunctionalWorkflow(
    func,                                    # async function body
    *,
    name=None,                               # defaults to func.__name__
    description=None,
    checkpoint_storage=None,                 # default per-workflow storage
)
```

The `@workflow` decorator constructs this for you. At decoration time it:
1. Validates that the function has **at most one** non-`RunContext` parameter (raises `ValueError` if more are present).
2. Discovers step names referenced inside the body for stable `graph_signature_hash` computation.

### Running a functional workflow

```python
from agent_framework._workflows._functional import workflow, step

@step
async def extract(text: str) -> list[str]:
    return text.split()

@step
async def score(tokens: list[str]) -> int:
    return len(tokens)

@workflow
async def count_words(text: str) -> int:
    tokens = await extract(text)
    return await score(tokens)

# Non-streaming run
result = await count_words.run("hello world foo")
print(result.get_outputs())   # [3]

# Streaming run
stream = count_words.run("hello world foo", stream=True)
async for update in stream:
    print(update)
final = await stream
```

### Checkpoint-enabled run with per-run storage override

```python
from agent_framework import InMemoryCheckpointStorage

storage = InMemoryCheckpointStorage()

# First run – steps are executed and cached
r1 = await count_words.run("hello world foo", checkpoint_storage=storage)
checkpoint_id = r1.checkpoint_id

# Resume run – @step calls return cached results (bypass event emitted)
r2 = await count_words.run(
    "hello world foo",
    checkpoint_id=checkpoint_id,
    checkpoint_storage=storage,
)
print(r2.get_outputs())    # [3]  (from cache, not re-executed)
```

### `FunctionalWorkflowAgent` — agent-compatible adapter

```python
from agent_framework._workflows._functional import FunctionalWorkflowAgent

agent = FunctionalWorkflowAgent(
    workflow=count_words,
    name="word-counter",
    description="Counts words in a document.",
)

# Drop-in replacement for Agent.run():
response = await agent.run("hello world foo")
print(response.text)

# Streaming
stream = agent.run("hello world foo", stream=True)
async for update in stream:
    print(update.text)
final = await stream
```

HITL: when a workflow step suspends, the framework catches `WorkflowInterrupted` internally and `run()` returns an interrupted result — `agent.pending_requests` is then populated with the suspended `WorkflowEvent` objects. Resume by passing `responses` keyed by `request_id`:

```python
r = await agent.run("draft document text")
if agent.pending_requests:
    req_id = next(iter(agent.pending_requests))
    r2 = await agent.run(
        "draft document text",
        responses={req_id: "Approved — ship it."},
        checkpoint_id=r.checkpoint_id,
    )
    print(r2.text)
```

---

## 3 · `StepWrapper` + `RunContext`

**Module:** `agent_framework._workflows._functional`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._workflows._functional import StepWrapper, RunContext`
**Decorator:** `@experimental(feature_id=ExperimentalFeature.FUNCTIONAL_WORKFLOWS)`

`StepWrapper` is the object produced by the `@step` decorator. When called *inside* a running `@workflow` function it provides caching, event emission, and `RunContext` auto-injection. When called *outside* a workflow (e.g. in unit tests) it delegates directly to the original function.

`RunContext` is the opt-in handle injected into `@workflow` and `@step` functions that need HITL, custom events, or per-run state. Declare it by parameter name (`ctx`) or annotation (`ctx: RunContext`).

### `StepWrapper` caching contract

Cache key is `(step_name, call_index)` where `call_index` counts how many times the same step name has been called in this run. On a cache hit, the wrapper:
- Skips the original function call.
- Emits an `executor_bypassed` event instead of the normal `executor_invoked` / `executor_completed` pair.
- Returns the cached result immediately.

On a cache miss (first run or expired checkpoint), it:
1. Emits `executor_invoked`.
2. Awaits the original function (injecting `RunContext` if declared).
3. Emits `executor_completed` (or `executor_failed` on exception).
4. Persists a checkpoint if `checkpoint_storage` is configured.

```python
from agent_framework._workflows._functional import step, workflow, RunContext

@step
async def validate(doc: str) -> bool:
    return len(doc) > 10

@step(name="review")              # explicit name overrides func.__name__
async def review_doc(doc: str, ctx: RunContext) -> str:
    feedback = await ctx.request_info({"draft": doc}, response_type=str)
    return feedback

@workflow
async def approval_pipeline(doc: str) -> str:
    ok = await validate(doc)
    if not ok:
        return "too short"
    return await review_doc(doc)   # ctx auto-injected at call time
```

### `StepWrapper` outside a workflow (unit testing)

```python
# Called outside a @workflow — behaves like the original function
result = await validate("this is a valid document")
assert result is True
```

### `RunContext` — HITL with `request_info`

`request_info` suspends the workflow on first call. The workflow is paused and returns a `WorkflowRunResult` containing the pending `WorkflowEvent`. On resume (via `responses=` dict keyed by `request_id`), the cached response is returned and execution continues.

```python
@workflow
async def human_review(draft: str, ctx: RunContext) -> str:
    # First call — workflow suspends here
    approval = await ctx.request_info(
        request_data={"draft": draft},
        response_type=str,
        request_id="review-1",   # explicit ID; auto-assigned when None
    )
    return f"Approved text: {approval}"

# Run #1 — suspends
r = await human_review.run("my draft")
# r.checkpoint_id is set; pending requests available

# Run #2 — resume with response
r2 = await human_review.run(
    "my draft",
    responses={"review-1": "LGTM"},
    checkpoint_id=r.checkpoint_id,
)
print(r2.get_outputs())   # ["Approved text: LGTM"]
```

### `RunContext` — per-run state and custom events

```python
@step
async def fetch_data(url: str, ctx: RunContext) -> dict:
    ctx.set_state("source_url", url)          # persist across steps
    ctx.add_event({"type": "fetching", "url": url})   # custom event
    return {"content": "..."}

@workflow
async def enrich_pipeline(url: str, ctx: RunContext) -> dict:
    data = await fetch_data(url)
    source = ctx.get_state("source_url")   # read back in workflow body
    return {**data, "source": source}
```

---

## 4 · `CompactionProvider` + `ContextWindowCompactionStrategy`

**Module:** `agent_framework._compaction`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework import CompactionProvider` / `from agent_framework._compaction import ContextWindowCompactionStrategy`

`CompactionProvider` is a `ContextProvider` that fires compaction strategies at two points in an agent turn:
- **`before_strategy`** — runs in `before_run`, compacting messages already loaded into context (before the model sees them).
- **`after_strategy`** — runs in `after_run`, compacting the persisted history so the *next* turn starts smaller.

`ContextWindowCompactionStrategy` is the two-phase pipeline strategy that models after a real LLM's context window. It computes an input budget from `max_context_window_tokens - max_output_tokens`, then fires two sub-strategies in sequence:

1. **Tool eviction** — triggers at `tool_eviction_threshold` (default 50%) of the input budget.
2. **Truncation** — triggers at `truncation_threshold` (default 80%) of the input budget.

### `ContextWindowCompactionStrategy` constructor

```python
ContextWindowCompactionStrategy(
    *,
    max_context_window_tokens: int,           # e.g. 128_000
    max_output_tokens: int,                   # e.g. 16_384
    tokenizer=None,                           # defaults to CharacterEstimatorTokenizer
    tool_eviction_threshold: float = 0.5,     # fraction of input budget
    truncation_threshold: float = 0.8,        # must be >= tool_eviction_threshold
    keep_last_tool_call_groups: int = 4,      # tool groups preserved during eviction
)
```

### `CompactionProvider` constructor

```python
CompactionProvider(
    *,
    before_strategy=None,          # applied before each run
    after_strategy=None,           # applied to persisted history after each run
    tokenizer=None,                # passed to token-aware strategies
    source_id="compaction",
    history_source_id="in_memory", # source_id of the history provider to compact
)
```

### Wiring both phases

```python
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider
from agent_framework._compaction import ContextWindowCompactionStrategy

history = InMemoryHistoryProvider()

strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=128_000,
    max_output_tokens=16_384,
    tool_eviction_threshold=0.5,
    truncation_threshold=0.8,
    keep_last_tool_call_groups=4,
)

compaction = CompactionProvider(
    before_strategy=strategy,       # compact loaded context before model call
    after_strategy=strategy,        # compact stored history after model call
    history_source_id=history.source_id,
)

agent = Agent(
    client=client,
    name="assistant",
    context_providers=[history, compaction],
)
```

### Using only one phase

```python
from agent_framework._compaction import SlidingWindowStrategy

# Only compact what's stored — never trim the in-context messages
compaction = CompactionProvider(
    after_strategy=SlidingWindowStrategy(keep_last_groups=30),
    history_source_id=history.source_id,
)
```

### Custom tokenizer for precise token counting

```python
import tiktoken
from agent_framework._compaction import ContextWindowCompactionStrategy

class TiktokenWrapper:
    def __init__(self):
        self._enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

strategy = ContextWindowCompactionStrategy(
    max_context_window_tokens=128_000,
    max_output_tokens=4_096,
    tokenizer=TiktokenWrapper(),
)
```

---

## 5 · `SlidingWindowStrategy` + `SelectiveToolCallCompactionStrategy` + `CharacterEstimatorTokenizer`

**Module:** `agent_framework._compaction`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._compaction import SlidingWindowStrategy, SelectiveToolCallCompactionStrategy, CharacterEstimatorTokenizer`

These are the targeted compaction strategies for more surgical context management.

### `SlidingWindowStrategy`

Keeps the `keep_last_groups` most recent non-system message groups, optionally preserving all system groups as permanent anchors. Groups are annotation-identified; each group spans one logical turn (user+assistant+tool results).

```python
SlidingWindowStrategy(*, keep_last_groups: int, preserve_system: bool = True)
```

- **`keep_last_groups`** — must be > 0; raises `ValueError` otherwise.
- **`preserve_system`** — when `True` (default), system groups are never excluded.

```python
from agent_framework._compaction import SlidingWindowStrategy

# Keep only the last 20 turns; system instructions always survive
strategy = SlidingWindowStrategy(keep_last_groups=20)

# Keep last 5 turns AND allow system messages to be dropped too
aggressive = SlidingWindowStrategy(keep_last_groups=5, preserve_system=False)
```

```python
from agent_framework import Agent, CompactionProvider, InMemoryHistoryProvider

history = InMemoryHistoryProvider()
compaction = CompactionProvider(
    after_strategy=SlidingWindowStrategy(keep_last_groups=20),
    history_source_id=history.source_id,
)
agent = Agent(client=client, name="bot", context_providers=[history, compaction])
```

### `SelectiveToolCallCompactionStrategy`

Only targets `tool_call` annotated groups. Non-tool groups are untouched, making it safe to combine with `SlidingWindowStrategy` as a two-pass pipeline.

```python
SelectiveToolCallCompactionStrategy(*, keep_last_tool_call_groups: int = 1)
```

- **`keep_last_tool_call_groups`** — number of newest tool-call groups to retain. Pass `0` to remove **all** tool-call groups. Raises `ValueError` if negative.

```python
from agent_framework._compaction import (
    SelectiveToolCallCompactionStrategy,
    SlidingWindowStrategy,
)
from agent_framework import CompactionProvider

# Stage 1: drop all but the last 2 tool-call groups
# Stage 2: keep only the last 40 non-system groups
# Combine as a TokenBudgetComposedStrategy or run sequentially via nested providers:
tool_compact = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2)
window = SlidingWindowStrategy(keep_last_groups=40)

# Remove ALL tool history (aggressive; useful when tool output is just scaffolding)
no_tools = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=0)
```

```python
# Pipeline: tool eviction first, then sliding window
from agent_framework import CompactionProvider, InMemoryHistoryProvider

history = InMemoryHistoryProvider()
# Use ContextWindowCompactionStrategy which already sequences these internally,
# or wire them as separate before/after strategies:
pre_compact = CompactionProvider(
    before_strategy=SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=3),
    history_source_id=history.source_id,
)
```

### `CharacterEstimatorTokenizer`

The default tokenizer used when none is provided. Estimates tokens as `max(1, len(text) // 4)` — the classic "4 characters per token" heuristic. Sufficient for budget-based decisions where exact counts are not critical.

```python
from agent_framework._compaction import CharacterEstimatorTokenizer

tok = CharacterEstimatorTokenizer()
print(tok.count_tokens("Hello, world!"))   # 3 (13 chars // 4 = 3)
print(tok.count_tokens("a"))              # 1 (minimum is always 1)
```

---

## 6 · `AGUIChatClient` + `AGUIEventConverter`

**Module:** `agent_framework.ag_ui`
**Install:** `pip install agent-framework agent-framework-ag-ui`
**Import:** `from agent_framework.ag_ui import AGUIChatClient, AGUIEventConverter`

`AGUIChatClient` is a multi-layer chat client for AG-UI compliant servers. Its MRO stacks four layers:

```
AGUIChatClient
  └─ FunctionInvocationLayer   (client-side tool execution)
  └─ ChatMiddlewareLayer        (middleware pipeline)
  └─ ChatTelemetryLayer         (OTel spans/metrics)
  └─ BaseChatClient             (core get_response/stream)
```

The client maintains **thread continuity** via `thread_id` in `additional_properties` — the server is responsible for history; the client sends only the messages it receives per call.

### Constructor

```python
AGUIChatClient(
    endpoint: str,               # AG-UI server URL, e.g. "http://localhost:8888/"
    *,
    http_client=None,            # optional httpx.AsyncClient
    timeout: float = 60.0,
    # inherits all BaseChatClient / middleware kwargs
)
```

### Direct usage — server manages thread history

```python
from agent_framework.ag_ui import AGUIChatClient

client = AGUIChatClient(endpoint="http://localhost:8888/")

# First turn — thread ID auto-generated
r1 = await client.get_response("What is 2+2?")
thread_id = r1.additional_properties.get("thread_id")
print(r1.text)

# Subsequent turns — pass thread_id so server retrieves history
r2 = await client.get_response(
    "Multiply that by 3.",
    metadata={"thread_id": thread_id},
)
print(r2.text)
```

### Recommended usage — wrap with `Agent` for client-side history

```python
from agent_framework import Agent
from agent_framework.ag_ui import AGUIChatClient

client = AGUIChatClient(endpoint="http://localhost:8888/")
agent = Agent(client=client, name="ui-agent")
session = agent.create_session()

r = await agent.run("Hello", session=session)
print(r.text)
r2 = await agent.run("Follow up question", session=session)
print(r2.text)
```

### `AGUIEventConverter` — event-by-event conversion

`AGUIEventConverter` maps raw AG-UI SSE event dicts to `ChatResponseUpdate` objects. Instantiate one per stream run; it carries internal state (`current_message_id`, `accumulated_tool_args`, `thread_id`, `run_id`).

Supported event types (dispatched in `convert_event`):
- `RUN_STARTED` → stores `thread_id` / `run_id`, emits `additional_properties` update
- `TEXT_MESSAGE_START` → begins a new text message
- `TEXT_MESSAGE_CONTENT` → emits delta text
- `TEXT_MESSAGE_END` → closes the current text message
- `TOOL_CALL_START` → begins a function call
- `TOOL_CALL_ARGS` → accumulates JSON argument fragments
- `TOOL_CALL_END` → emits completed `function_call` content
- `TOOL_CALL_RESULT` → emits `function_result` content
- `RUN_FINISHED` → signals stream completion
- Unknown types → returns `None` (no update emitted)

```python
from agent_framework.ag_ui import AGUIEventConverter

converter = AGUIEventConverter()

events = [
    {"type": "RUN_STARTED", "threadId": "t-42", "runId": "r-99"},
    {"type": "TEXT_MESSAGE_START", "messageId": "m-1"},
    {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m-1", "delta": "Hello "},
    {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m-1", "delta": "world"},
    {"type": "TEXT_MESSAGE_END", "messageId": "m-1"},
]

for event in events:
    update = converter.convert_event(event)
    if update:
        print(update.contents)
```

```python
# Lower-level: stream raw SSE events from AGUIHttpService
from agent_framework.ag_ui import AGUIHttpService, AGUIEventConverter

service = AGUIHttpService("http://localhost:8888/")
converter = AGUIEventConverter()

async with service as svc:
    async for event in svc.post_run(
        thread_id="t-42",
        run_id="r-1",
        messages=[{"role": "user", "content": "Hello"}],
    ):
        update = converter.convert_event(event)
        if update and update.contents:
            for c in update.contents:
                if c.type == "text":
                    print(c.text, end="", flush=True)
```

---

## 7 · `ClassSkill` + `AggregatingSkillsSource` + `FilteringSkillsSource` + `DelegatingSkillsSource`

**Module:** `agent_framework._skills`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._skills import ClassSkill, AggregatingSkillsSource, FilteringSkillsSource, DelegatingSkillsSource`
**Decorator:** `@experimental(feature_id=ExperimentalFeature.SKILLS)`

`ClassSkill` lets you define a reusable skill as a Python class rather than as a filesystem directory. Resources and scripts can be declared with `@ClassSkill.resource` / `@ClassSkill.script` decorators (auto-discovery) or by overriding the `resources` / `scripts` properties directly.

The **source composition** classes — `AggregatingSkillsSource`, `FilteringSkillsSource`, `DelegatingSkillsSource` — follow the decorator pattern: each wraps another `SkillsSource` and modifies what `get_skills()` returns.

### Defining a `ClassSkill` with decorators

```python
import json
from agent_framework._skills import ClassSkill
from agent_framework import SkillFrontmatter

class UnitConverterSkill(ClassSkill):
    def __init__(self):
        super().__init__(
            frontmatter=SkillFrontmatter(
                name="unit-converter",
                description="Convert between common units of measurement.",
            ),
        )

    @property
    def instructions(self) -> str:
        return "Use the conversion table resource and the convert script to answer unit questions."

    @ClassSkill.resource(name="table")
    def conversion_table(self) -> str:
        return "| From | To | Factor |\n|------|-------|-------|\n| km | miles | 0.621 |"

    @ClassSkill.script(name="convert")
    def convert(self, value: float, factor: float) -> str:
        return json.dumps({"result": round(value * factor, 4)})
```

### Using `ClassSkill` with an agent

```python
from agent_framework import Agent
from agent_framework import SkillsProvider, InMemorySkillsSource

skill = UnitConverterSkill()
skills_source = InMemorySkillsSource([skill])
provider = SkillsProvider(source=skills_source)

agent = Agent(
    client=client,
    name="converter-bot",
    context_providers=[provider],
)
response = await agent.run("Convert 10 km to miles.")
print(response.text)
```

### `AggregatingSkillsSource` — merge multiple sources

```python
from agent_framework._skills import AggregatingSkillsSource
from agent_framework import InMemorySkillsSource

source_a = InMemorySkillsSource([UnitConverterSkill()])
source_b = InMemorySkillsSource([DateFormatterSkill()])

combined = AggregatingSkillsSource([source_a, source_b])
skills = await combined.get_skills()
print([s.frontmatter.name for s in skills])
# ["unit-converter", "date-formatter"]
```

### `FilteringSkillsSource` — exclude skills by predicate

```python
from agent_framework._skills import FilteringSkillsSource

# Hide internal-use skills from the agent
production_skills = FilteringSkillsSource(
    inner_source=combined,
    predicate=lambda s: not s.frontmatter.name.startswith("internal"),
)
skills = await production_skills.get_skills()
```

### `DelegatingSkillsSource` — base for custom decorators

```python
from agent_framework._skills import DelegatingSkillsSource

class CachingSkillsSource(DelegatingSkillsSource):
    """Cache get_skills() results for 60 seconds."""
    def __init__(self, inner_source):
        super().__init__(inner_source)
        self._cache = None
        self._expires = 0

    async def get_skills(self):
        import time
        now = time.monotonic()
        if self._cache is None or now > self._expires:
            self._cache = await self._inner_source.get_skills()
            self._expires = now + 60
        return self._cache
```

---

## 8 · `BackgroundAgentsProvider` + `BackgroundTaskInfo` + `BackgroundTaskStatus`

**Module:** `agent_framework._harness._background_agents`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._harness._background_agents import BackgroundAgentsProvider, BackgroundTaskInfo, BackgroundTaskStatus`
**Decorator:** `@experimental(feature_id=ExperimentalFeature.HARNESS)`

`BackgroundAgentsProvider` is a `ContextProvider` that exposes **six LLM-callable tools** to a parent agent, enabling it to delegate work to named sub-agents running concurrently in separate sessions:

| Tool | Purpose |
|------|---------|
| `background_agents_start_task` | Start a background task on a named agent |
| `background_agents_wait_for_first_completion` | Block until the first listed task finishes |
| `background_agents_get_task_results` | Retrieve a completed task's text output |
| `background_agents_get_all_tasks` | List all tasks with IDs, statuses, descriptions |
| `background_agents_continue_task` | Send follow-up input to a completed task's session |
| `background_agents_clear_completed_task` | Remove a completed task and release its session |

### Constructor

```python
BackgroundAgentsProvider(
    agents: Sequence[SupportsAgentRun],
    *,
    source_id: str = DEFAULT_BACKGROUND_AGENTS_SOURCE_ID,
    instructions: str | None = None,   # may include {background_agents} placeholder
)
```

- **`agents`** — must be non-empty, all names non-empty, all names unique (case-insensitive). Raises `ValueError` otherwise.
- **`instructions`** — optional override; include `{background_agents}` to get the agent listing injected automatically.

### Wiring to a parent agent

```python
from agent_framework import Agent
from agent_framework._harness._background_agents import BackgroundAgentsProvider

research_agent = Agent(client=client, name="researcher")
writer_agent = Agent(client=client, name="writer")

provider = BackgroundAgentsProvider(
    agents=[research_agent, writer_agent],
)

coordinator = Agent(
    client=client,
    name="coordinator",
    context_providers=[provider],
)

# The coordinator can now delegate to "researcher" and "writer" via tool calls
session = coordinator.create_session()
response = await coordinator.run(
    "Research Python async patterns and write a 200-word summary.",
    session=session,
)
print(response.text)
```

### `BackgroundTaskInfo` — task metadata

`BackgroundTaskInfo` is a `SerializationMixin` slot-class persisted in session state. Fields: `id` (int), `agent_name`, `description`, `status` (`BackgroundTaskStatus`), `result_text`, `error_text`.

```python
from agent_framework._harness._background_agents import BackgroundTaskInfo, BackgroundTaskStatus

info = BackgroundTaskInfo(
    id=1,
    agent_name="researcher",
    description="Find Python async patterns",
    status=BackgroundTaskStatus.RUNNING,
)
print(info.status)      # BackgroundTaskStatus.RUNNING

# Serialization
d = info.to_dict()
restored = BackgroundTaskInfo.from_dict(d)
assert restored.agent_name == "researcher"
```

### `BackgroundTaskStatus` enum

```python
from agent_framework._harness._background_agents import BackgroundTaskStatus

for status in BackgroundTaskStatus:
    print(status.value)
# running
# completed
# failed
# lost        ← set when the session is lost before completion
```

---

## 9 · `TodoItem` + `TodoFileStore` + `TodoInput` + `TodoCompleteInput`

**Module:** `agent_framework._harness._todo`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._harness._todo import TodoItem, TodoFileStore, TodoInput, TodoCompleteInput`
**Decorator:** `@experimental(feature_id=ExperimentalFeature.HARNESS)`

The todo harness gives an agent a structured task-tracking system with file-persisted state. `TodoProvider` exposes five tools (`todos_add`, `todos_complete`, `todos_remove`, `todos_get_remaining`, `todos_get_all`) and manages session-scoped state via a `WeakKeyDictionary`-backed per-session `asyncio.Lock`.

### `TodoItem`

A `SerializationMixin` class with fields: `id` (int), `title` (str), `description` (str | None, optional), `is_complete` (bool, default `False`).

```python
from agent_framework._harness._todo import TodoItem

item = TodoItem(id=1, title="Draft the introduction section")
print(item.is_complete)   # False
print(item.description)   # None

d = item.to_dict()
# {"id": 1, "title": "Draft the introduction section", "is_complete": False}
```

### `TodoInput` + `TodoCompleteInput`

`TodoInput` carries the `title` (+ optional `description`) for `todos_add`. `TodoCompleteInput` carries the `id` and a mandatory `reason` string for `todos_complete`.

```python
from agent_framework._harness._todo import TodoInput, TodoCompleteInput

add_input = TodoInput(title="Write conclusion", description="Cover key findings")
complete_input = TodoCompleteInput(id=3, reason="Finished drafting all sections")
```

### `TodoFileStore` — persistent todo storage

`TodoFileStore` persists todo state to disk as JSON, keyed by session + source ID. Constructor takes `base_path` (a root directory). The abstract `TodoStore` protocol defines `load_state(session, *, source_id)` → `(list[TodoItem], next_id)`, `save_state(session, items, *, next_id, source_id)`, and the convenience `load_items(session, *, source_id)`.

```python
from agent_framework._harness._todo import TodoFileStore

store = TodoFileStore(base_path="/tmp/agent_todos")
# Files are written to: /tmp/agent_todos/<owner>/<kind>/todos.json
# Access items via the TodoProvider tools — TodoFileStore is a backing store,
# not a direct list API.
```

### Wiring `TodoProvider`

```python
from agent_framework import Agent
from agent_framework import TodoProvider    # public re-export
from agent_framework._harness._todo import TodoFileStore

store = TodoFileStore(base_path="/workspace/todos")
provider = TodoProvider(store=store)

agent = Agent(
    client=client,
    name="task-manager",
    context_providers=[provider],
)

session = agent.create_session()
response = await agent.run(
    "I need to write a blog post. Track the steps: outline, draft, edit, publish.",
    session=session,
)
print(response.text)
```

---

## 10 · `AgentEvalConverter` + `ConversationSplitter` + `EvalResults` + `EvalItem` + `CheckResult`

**Module:** `agent_framework._evaluation`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._evaluation import AgentEvalConverter, EvalResults, EvalItem, CheckResult`
**Decorator:** `@experimental(feature_id=ExperimentalFeature.EVALS)`

The evaluation subsystem converts agent-framework conversations into provider-agnostic evaluation items, runs them through evaluation services, and returns structured results.

### `AgentEvalConverter` — type bridge

All methods are static. `convert_message` maps a single `Message` to a list of Foundry-format dicts. A `Message` with multiple `function_result` contents produces **multiple** output dicts (one per tool result).

```python
from agent_framework import Message, Content
from agent_framework._evaluation import AgentEvalConverter

msg = Message("assistant", [
    Content(type="text", text="Here is the result:"),
    Content(type="function_call", name="search", arguments='{"q": "AI"}'),
])

converted = AgentEvalConverter.convert_message(msg)
# [
#   {"role": "assistant", "content": [
#       {"type": "text", "text": "Here is the result:"},
#       {"type": "tool_call", "tool_call_id": "...", "function": {"name": "search", ...}}
#   ]}
# ]
```

### `ConversationSplitter` — split strategy protocol

Any callable `(list[Message]) -> tuple[list[Message], list[Message]]` satisfies the protocol. Built-in options are the `ConversationSplit` enum members (`LAST_TURN`, etc.).

```python
from agent_framework._evaluation import ConversationSplitter
from agent_framework import Message, Content

# Custom splitter: split just before a memory retrieval call
def split_before_memory(
    conversation: list[Message],
) -> tuple[list[Message], list[Message]]:
    for i, msg in enumerate(conversation):
        for c in msg.contents or []:
            if c.type == "function_call" and c.name == "retrieve_memory":
                return conversation[:i], conversation[i:]
    # Fallback: last user/assistant turn
    from agent_framework import ConversationSplit
    return ConversationSplit.LAST_TURN(conversation)
```

### `EvalItem` — provider-agnostic eval record

```python
from agent_framework._evaluation import EvalItem
from agent_framework import Message, Content, ConversationSplit

conversation = [
    Message("user", [Content(type="text", text="What is the capital of France?")]),
    Message("assistant", [Content(type="text", text="Paris.")]),
]

item = EvalItem(
    conversation=conversation,
    expected_output="Paris",
    split_strategy=ConversationSplit.LAST_TURN,
)

print(item.query)      # "What is the capital of France?"
print(item.response)   # "Paris."

# Custom split
item2 = EvalItem(conversation=conversation, split_strategy=split_before_memory)
```

### `EvalResults` — structured provider results

```python
from agent_framework._evaluation import EvalResults, EvalItemResult

results = EvalResults(
    provider="foundry",
    eval_id="eval-001",
    run_id="run-abc",
    status="completed",
    result_counts={"passed": 8, "failed": 2},
    report_url="https://ai.azure.com/evals/run-abc",
    per_evaluator={
        "groundedness": {"passed": 9, "failed": 1},
        "coherence":    {"passed": 8, "failed": 2},
    },
)

print(f"Pass rate: {results.passed}/{results.total}")
# Pass rate: 8/10

# Per-evaluator breakdown
for name, counts in results.per_evaluator.items():
    print(f"  {name}: {counts['passed']}/{sum(counts.values())}")

# Workflow eval — per-agent sub_results
for agent_name, sub in (results.sub_results or {}).items():
    print(f"  {agent_name}: {sub.passed}/{sub.total}")
```

### `CheckResult` — local check outcome

```python
from agent_framework._evaluation import CheckResult

result = CheckResult(
    passed=True,
    reason="Response contained expected keyword 'Paris'",
    check_name="keyword-match",
)
print(result.passed)       # True
print(result.check_name)   # "keyword-match"
```

### End-to-end local evaluation with `LocalEvaluator`

```python
from agent_framework import LocalEvaluator, Agent
from agent_framework._evaluation import EvalItem, CheckResult
from agent_framework import ConversationSplit

def keyword_check(item: EvalItem) -> CheckResult:
    found = item.expected_output and item.expected_output.lower() in item.response.lower()
    return CheckResult(
        passed=bool(found),
        reason=f"Expected '{item.expected_output}' in response",
        check_name="keyword-match",
    )

evaluator = LocalEvaluator(checks=[keyword_check])

agent = Agent(client=client, name="qa-bot")
item = EvalItem(
    conversation=[
        Message("user", [Content(type="text", text="Capital of France?")]),
    ],
    expected_output="Paris",
)

results = await evaluator.evaluate(agent=agent, items=[item])
for r in results:
    print(r.passed, r.reason)
```

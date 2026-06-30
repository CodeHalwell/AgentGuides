---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 28"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.9.0: RawClaudeAgent+ClaudeAgent+ClaudeAgentOptions+ClaudeAgentSettings (Claude Code CLI integration — AGENT_PROVIDER_NAME='anthropic.claude', ClaudeSDKClient lifecycle, PermissionMode literals, SandboxSettings, ClaudeAgentSettings env-prefix resolution); AgentFrameworkAgent+AgentConfig (AG-UI protocol wrapper for any SupportsAgentRun — state_schema Pydantic dispatch, predict_state_config tool-argument interception, require_confirmation gate, snapshot_store wiring); AGUIThreadSnapshot+AGUIThreadSnapshotStore+InMemoryAGUIThreadSnapshotStore (replayable thread state — messages+state+interrupt triad, Protocol async save/get/delete, max_snapshots LRU eviction, scope+thread_id composite key); PredictiveStateHandler+PredictStateConfig (streaming tool-call state prediction — predict_state_config dict[state_key→{tool,tool_argument}], streaming_tool_args accumulator, state_delta_count throttle, pending_state_updates flush); AGUIRequest+AGUIChatOptions+AgentState+RunMetadata (AG-UI protocol types — AliasChoices camelCase aliases, client-side tools forwarding, thread_id continuity, predict_state list[PredictStateConfig]); A2AAgentSession+A2AContinuationToken (A2A session management — _CONTEXT_ID_KEY/task_id/task_state slot keys, service_session_id=context_id, TaskState continuation detection); ThreadItemConverter (ChatKit→agent-framework bridge — attachment_data_fetcher async callback, UserMessageItem→Message dispatch, to_agent_input() normalisation); AgentApprovalExecutor+AgentRequestInfoExecutor+AgentRequestInfoResponse (orchestration approval gate — allow_direct_output terminal mode, @handler/@response_handler pair, AgentRequestInfoResponse.approve() shortcut); AgentFrameworkWorkflow (workflow-to-AG-UI bridge — workflow vs workflow_factory mutex, (scope,thread_id) cache key, snapshot_store activation guard); FlowState+AGUIHttpService+run_workflow_stream (run-level streaming state — 14 dataclass fields, AGUIHttpService SSE parsing, run_workflow_stream thread/run-id resolution)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 51
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 28

Verified against **agent-framework 1.9.0** / **agent-framework-claude 1.0.0b260609** / **agent-framework-ag-ui 1.0.0rc5** / **agent-framework-a2a 1.0.0b260604** / **agent-framework-chatkit 1.0.0b260528** / **agent-framework-orchestrations 1.0.0** (installed June 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`. Sub-packages introspected:
`agent_framework_claude._agent`,
`agent_framework_ag_ui._agent`,
`agent_framework_ag_ui._snapshots`,
`agent_framework_ag_ui._orchestration._predictive_state`,
`agent_framework_ag_ui._types`,
`agent_framework_ag_ui._workflow`,
`agent_framework_ag_ui._run_common`,
`agent_framework_ag_ui._http_service`,
`agent_framework_ag_ui._workflow_run`,
`agent_framework_a2a._agent`,
`agent_framework_chatkit._converter`,
`agent_framework_orchestrations._orchestration_request_info`.

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
- [Vol. 22](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v22/) — declarative workflow internals v22
- [Vol. 23](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/) — `DeclarativeActionExecutor`, `DeclarativeWorkflowState`, `DeclarativeEnvConfig`, condition+foreach+break/continue executors, basic variable executors, `AgentManifest`+`PromptAgent`, `Property`+`PropertySchema`, `Connection` hierarchy, `McpTool`+approval modes, `Model`+`ModelOptions`+`Template`, `InvokeAzureAgentExecutor`
- [Vol. 24](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v24/) — `_workflows._typing_utils`, `_workflows._checkpoint_encoding`, `_workflows._runner`, `_harness._loop`, `_harness._tool_approval`, orchestrations protocol utils, Magentic observability, Foundry/OpenAI raw clients, `_workflows` message utilities
- [Vol. 25](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v25/) — `FunctionTool`+`OpenApiTool`+`WebSearchTool`+`FileSearchTool`+`CodeInterpreterTool`+`Binding`, `AgentFactory`+`DeclarativeLoaderError`+`ProviderLookupError`, `WorkflowFactory`+`DeclarativeWorkflowBuilder`, `QuestionExecutor`+`RequestExternalInputExecutor`, `HttpRequestActionExecutor`, `InvokeMcpToolActionExecutor`, `BaseToolExecutor`+`InvokeFunctionToolExecutor`, `JoinExecutor`+termination nodes, `ActionComplete`+`ActionTrigger`+`DeclarativeStateData`, `ClearAllVariablesExecutor`+`EditTableExecutor`+`ResetVariableExecutor`+`SetTextVariableExecutor`
- [Vol. 26](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v26/) — `DurableAIAgentWorker`+`DurableAIAgentClient`, `DurableAIAgent`+`DurableAgentExecutor`, `DurableAIAgentOrchestrationContext`, `AgentEntity`+`AgentEntityStateProviderMixin`, `AgentCallbackContext`+`AgentResponseCallbackProtocol`, `RunRequest`, `AgentSessionId`+`DurableAgentSession`, `DurableAgentState`+`DurableAgentStateData`, entry hierarchy+`DurableAgentStateUsage`, content hierarchy+`DurableStateFields`+`ContentTypes`
- [Vol. 27](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v27/) — `ContentLabel`+`combine_labels`+`check_confidentiality_allowed`, `store_untrusted_content`+`get_security_tools`+security tool constructors, `enable/disable_instrumentation`+`enable_sensitive_telemetry`, `create_resource`+`create_metric_views`+histogram boundary constants, `get_tracer`+`get_meter`+`INNER_ACCUMULATED_USAGE`, `create_mcp_client_span`+`set_mcp_span_error`, `group_messages`+`annotate_message_groups`, `apply_compaction`+`project_included_messages`, `normalize_messages`+`merge_chat_options`, `normalize_tools`+`validate_chat_options`+`add_usage_details`

This volume covers **ten class groups** across five provider/integration sub-packages:
`agent-framework-claude` (Claude Code CLI integration), `agent-framework-ag-ui` (AG-UI streaming
protocol — wrappers, snapshots, predictive state, protocol types, workflow bridge, run state),
`agent-framework-a2a` (A2A session management), `agent-framework-chatkit` (ChatKit thread
converter), and `agent-framework-orchestrations` (approval gate executor). Some of these
classes appeared in Vol. 15 and Vol. 16 with brief coverage; this volume goes deeper —
covering `AgentConfig` TypedDict wiring, `predict_state_config` internals, `snapshot_store`
activation guards, and the full `AgentApprovalExecutor`/`AgentRequestInfoExecutor` executor
pair that was not documented before.

| # | Class / group | Sub-package |
|---|---|---|
| 1 | `RawClaudeAgent` · `ClaudeAgent` · `ClaudeAgentOptions` · `ClaudeAgentSettings` | `agent_framework_claude._agent` |
| 2 | `AgentFrameworkAgent` · `AgentConfig` | `agent_framework_ag_ui._agent` |
| 3 | `AGUIThreadSnapshot` · `AGUIThreadSnapshotStore` · `InMemoryAGUIThreadSnapshotStore` | `agent_framework_ag_ui._snapshots` |
| 4 | `PredictiveStateHandler` · `PredictStateConfig` | `._orchestration._predictive_state` / `._types` |
| 5 | `AGUIRequest` · `AGUIChatOptions` · `AgentState` · `RunMetadata` | `agent_framework_ag_ui._types` |
| 6 | `A2AAgentSession` · `A2AContinuationToken` | `agent_framework_a2a._agent` |
| 7 | `ThreadItemConverter` | `agent_framework_chatkit._converter` |
| 8 | `AgentApprovalExecutor` · `AgentRequestInfoExecutor` · `AgentRequestInfoResponse` | `agent_framework_orchestrations._orchestration_request_info` |
| 9 | `AgentFrameworkWorkflow` | `agent_framework_ag_ui._workflow` |
| 10 | `FlowState` · `AGUIHttpService` · `run_workflow_stream` | `._run_common` / `._http_service` / `._workflow_run` |

---

## 1 · `RawClaudeAgent` + `ClaudeAgent` + `ClaudeAgentOptions` + `ClaudeAgentSettings`

**Sub-package:** `agent_framework_claude._agent`
**Install:** `pip install agent-framework-claude`
**Import:** `from agent_framework_claude import ClaudeAgent, RawClaudeAgent`

`agent-framework-claude` integrates the **Claude Code CLI** (not the Anthropic REST API) as a
first-class agent-framework provider. The agent shells out to the `claude` binary and communicates
over its JSON-RPC protocol. This is distinct from `agent_framework_anthropic`, which uses the
Anthropic HTTP API directly.

### Inheritance chain

```
BaseAgent
  └─ RawClaudeAgent[OptionsT]         # core — no telemetry
       └─ ClaudeAgent[OptionsT]       # adds AgentTelemetryLayer (OTel spans)
```

`AGENT_PROVIDER_NAME = "anthropic.claude"` is set on `RawClaudeAgent` and is used as the OTel
span provider attribute.

### `RawClaudeAgent.__init__`

```python
RawClaudeAgent(
    instructions: str | None = None,
    *,
    client: ClaudeSDKClient | None = None,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: Sequence[AgentMiddlewareTypes] | None = None,
    tools: ToolTypes | Callable[..., Any] | str | Sequence[...] | None = None,
    default_options: OptionsT | MutableMapping[str, Any] | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
)
```

- **`client`** — pre-configured `ClaudeSDKClient`. When `None` (default), a new client is
  created from the remaining kwargs. The agent sets `self._owns_client = client is None` and
  only calls `client.close()` on `__aexit__` when it owns the client.
- **`tools`** — accepts strings naming built-in Claude tools (`"Read"`, `"Write"`, `"Bash"`,
  `"Glob"`) as well as callables and `ToolTypes` objects.
- **`env_file_path`** — path to a `.env` file whose vars override `CLAUDE_AGENT_*` environment
  variables during settings resolution.

### `ClaudeAgentOptions` (TypedDict)

All keys are optional (`total=False`):

| Key | Type | Notes |
|-----|------|-------|
| `system_prompt` | `str` | Overrides `instructions` for this call |
| `cli_path` | `str \| Path` | Path to `claude` binary; auto-detected otherwise |
| `cwd` | `str \| Path` | Working directory for the CLI subprocess |
| `env` | `dict[str, str]` | Extra environment variables forwarded to the subprocess |
| `settings` | `str` | Path to Claude settings JSON file |
| `model` | `str` | `"sonnet"`, `"opus"`, or `"haiku"` (default: `"sonnet"`) |
| `fallback_model` | `str` | Secondary model when the primary fails |
| `allowed_tools` | `list[str]` | Allowlist — Claude may **only** use these tools |
| `disallowed_tools` | `list[str]` | Blocklist — Claude must never use these tools |
| `mcp_servers` | `dict[str, McpServerConfig]` | MCP server configs for external tools |
| `permission_mode` | `PermissionMode` | `"default"`, `"acceptEdits"`, `"plan"`, `"bypassPermissions"` |
| `can_use_tool` | `CanUseTool` | Per-call-site tool permission callback |
| `max_turns` | `int` | Maximum conversation turns |
| `max_budget_usd` | `float` | Hard budget cap in USD |
| `hooks` | `dict[str, list[HookMatcher]]` | Pre/post tool hook configuration |
| `add_dirs` | `list[str \| Path]` | Extra directories added to Claude's context |
| `sandbox` | `SandboxSettings` | Bash isolation sandbox configuration |

### `ClaudeAgentSettings` (TypedDict)

Settings are resolved in priority order: **explicit kwargs → `.env` file →
`CLAUDE_AGENT_*` environment variables**.

```python
class ClaudeAgentSettings(TypedDict, total=False):
    cli_path: str | None
    model: str | None
    cwd: str | None
    permission_mode: str | None
    max_turns: int | None
    max_budget_usd: float | None
```

### Key facts

- `ClaudeAgent` is a `Generic[OptionsT]` where `OptionsT` defaults to `ClaudeAgentOptions`. Pass
  a custom `TypedDict` subclass to get typed `default_options` and per-call `options=` kwargs.
- `ClaudeAgent` overloads `run()` with `stream: Literal[False]` and `stream: Literal[True]`
  signatures exactly as `Agent` and `RawAgent` do, so it integrates transparently into any
  orchestration that accepts `SupportsAgentRun`.
- The agent is an **async context manager** — always use `async with ClaudeAgent(...) as agent:`.
  Entering the context creates (or borrows) the `ClaudeSDKClient`; exiting closes the subprocess.

### Code example

```python
import asyncio
from agent_framework_claude import ClaudeAgent

async def main():
    async with ClaudeAgent(
        instructions="You are a Python coding assistant.",
        default_options={
            "model": "sonnet",
            "allowed_tools": ["Read", "Write", "Bash"],
            "max_turns": 10,
            "permission_mode": "acceptEdits",
        },
    ) as agent:
        # Simple one-shot query
        response = await agent.run("Explain how Python's GIL works.")
        print(response.text)

        # Streaming
        async with agent.run("Refactor this file: main.py", stream=True) as stream:
            async for update in stream:
                print(update.text, end="", flush=True)

asyncio.run(main())
```

```python
# Use tools restriction for a sandboxed code-review agent
from agent_framework_claude import ClaudeAgent

async def code_review(repo_path: str) -> str:
    async with ClaudeAgent(
        instructions="Review the Python code for bugs and security issues.",
        default_options={
            "model": "opus",
            "allowed_tools": ["Read", "Glob"],   # read-only — no writes
            "cwd": repo_path,
            "max_budget_usd": 0.50,
        },
    ) as agent:
        response = await agent.run("Review all *.py files in the repository.")
        return response.text
```

---

## 2 · `AgentFrameworkAgent` + `AgentConfig`

**Sub-package:** `agent_framework_ag_ui._agent`
**Install:** `pip install agent-framework-ag-ui`
**Import:** `from agent_framework_ag_ui import AgentFrameworkAgent, AgentConfig`

`AgentFrameworkAgent` wraps **any** `SupportsAgentRun` object (including `Agent`,
`WorkflowAgent`, `ClaudeAgent`, etc.) and translates its streaming responses into
the AG-UI event protocol: `RUN_STARTED → TEXT_MESSAGE_CONTENT → TOOL_CALL_* →
RUN_FINISHED`.

### `AgentConfig`

```python
AgentConfig(
    state_schema: Any | None = None,
    predict_state_config: dict[str, dict[str, str]] | None = None,
    use_service_session: bool = False,
    require_confirmation: bool = True,
    snapshot_store: AGUIThreadSnapshotStore | None = None,
)
```

| Parameter | Default | Behaviour |
|-----------|---------|-----------|
| `state_schema` | `None` | Pydantic model or dict; used to deserialise the incoming AG-UI `state` field |
| `predict_state_config` | `None` | Tool-argument→state-key map for predictive state emission (see §4) |
| `use_service_session` | `False` | When `True`, the `thread_id` is passed as `service_session_id` to enable server-side session continuity |
| `require_confirmation` | `True` | Waits for AG-UI `INTERRUPT_RESOLVED` before forwarding a tool call to the agent |
| `snapshot_store` | `None` | Persists `AGUIThreadSnapshot` at the end of each run |

### `AgentFrameworkAgent.__init__`

```python
AgentFrameworkAgent(
    agent: SupportsAgentRun,
    name: str | None = None,
    description: str | None = None,
    state_schema: Any | None = None,
    predict_state_config: dict[str, dict[str, str]] | None = None,
    require_confirmation: bool = True,
    use_service_session: bool = False,
    snapshot_store: AGUIThreadSnapshotStore | None = None,
)
```

The `AgentConfig` TypedDict is the structured form of these same fields and can be
passed to `AgentFrameworkAgent(**config, agent=my_agent)`.

### Key facts

- The wrapper follows a **linear event flow**: `RUN_STARTED` → one or more
  `TEXT_MESSAGE_CONTENT`/`TOOL_CALL_*` events → `RUN_FINISHED`. It does not support
  branching or sub-workflows — for that, use `AgentFrameworkWorkflow` (§9).
- When `require_confirmation=True` (default), each tool call emits a
  `TOOL_CALL_START` event and then **pauses** until the UI sends back an
  `INTERRUPT_RESOLVED` event with `approved: true`. Set `require_confirmation=False`
  to disable this gate.
- `state_schema` dispatches on the type: if it is a Pydantic `BaseModel` subclass,
  incoming state is parsed with `state_schema.model_validate(state_dict)`; if it
  is a plain `dict` or `type(None)`, the raw dict is used as-is.

### Code example

```python
from agent_framework.openai import OpenAIResponsesClient
from agent_framework_ag_ui import AgentFrameworkAgent
from agent_framework_ag_ui._snapshots import InMemoryAGUIThreadSnapshotStore

# Build the underlying agent
client = OpenAIResponsesClient(model="gpt-4o")
agent = client.as_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
)

# Wrap it for AG-UI
snapshot_store = InMemoryAGUIThreadSnapshotStore(max_snapshots=500)
ag_ui_agent = AgentFrameworkAgent(
    agent=agent,
    name="Assistant",
    description="A helpful AI assistant",
    require_confirmation=False,   # auto-approve tool calls
    use_service_session=True,     # resume conversations by thread_id
    snapshot_store=snapshot_store,
)

# Stream AG-UI events (e.g., from a FastAPI endpoint)
async def stream_events(request_data: dict):
    async for event in ag_ui_agent.run(request_data):
        yield f"data: {event.model_dump_json()}\n\n"
```

```python
# With state schema for structured generative UI
from pydantic import BaseModel

class DashboardState(BaseModel):
    chart_type: str = "bar"
    data_range: str = "7d"
    filters: list[str] = []

ag_ui_agent = AgentFrameworkAgent(
    agent=agent,
    state_schema=DashboardState,       # incoming state parsed as DashboardState
    predict_state_config={
        "chart_type": {                # when set_chart tool fires…
            "tool": "set_chart",
            "tool_argument": "type",   # …stream chart_type from its `type` arg
        }
    },
)
```

---

## 3 · `AGUIThreadSnapshot` + `AGUIThreadSnapshotStore` + `InMemoryAGUIThreadSnapshotStore`

**Sub-package:** `agent_framework_ag_ui._snapshots`
**Install:** `pip install agent-framework-ag-ui`
**Import:** `from agent_framework_ag_ui._snapshots import AGUIThreadSnapshot, AGUIThreadSnapshotStore, InMemoryAGUIThreadSnapshotStore`

Thread snapshots capture the **replayable state** of an AG-UI conversation so that a UI can
reconstruct the view on reconnect without replaying raw events or storing provider responses.

### `AGUIThreadSnapshot` (dataclass)

```python
@dataclass(slots=True)
class AGUIThreadSnapshot:
    messages: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] | None = None
    interrupt: list[dict[str, Any]] | None = None
```

- **`messages`** — serialised AG-UI message snapshots (not raw streaming events).
- **`state`** — the AG-UI Shared State at the end of the run (generative UI data).
- **`interrupt`** — pending interrupt payloads from `RUN_FINISHED.interrupt` (if the run
  paused waiting for human confirmation).

Intentional omissions: raw events, request metadata, auth claims, OTel traces, and provider
responses are **never** stored in a snapshot.

### `AGUIThreadSnapshotStore` (Protocol)

```python
@runtime_checkable
class AGUIThreadSnapshotStore(Protocol):
    async def save(self, *, scope: SnapshotScope, thread_id: AGUIThreadID,
                   snapshot: AGUIThreadSnapshot) -> None: ...
    async def get(self, *, scope: SnapshotScope, thread_id: AGUIThreadID,
                  ) -> AGUIThreadSnapshot | None: ...
    async def delete(self, *, scope: SnapshotScope, thread_id: AGUIThreadID,
                     ) -> bool: ...
```

`scope` is an application-defined **authorization boundary** — typically a user ID or
tenant ID. Two threads with the same `thread_id` but different `scope` values are
completely isolated. `scope` is part of the composite storage key.

### `InMemoryAGUIThreadSnapshotStore`

```python
InMemoryAGUIThreadSnapshotStore(*, max_snapshots: int = 1000)
```

- Bounded LRU map: keyed by `(scope, thread_id)`, capacity `max_snapshots`.
- `save()` evicts the oldest entry when full. It deep-copies the snapshot on write so
  that in-place mutations to the original object are not reflected in storage.
- `delete()` returns `True` if the key existed, `False` otherwise.
- Raises `ValueError` if `max_snapshots < 1`.
- **Not durable** — process-local only, intended for local development and tests.

### Key facts

- `snapshot_store` is activated only when **both** the store instance and a
  **Snapshot Scope resolver** are provided in the endpoint setup. Passing a store
  to `AgentFrameworkAgent` without a scope resolver is valid but results in no
  snapshots being written.
- The store is `@runtime_checkable`, so you can do `isinstance(obj, AGUIThreadSnapshotStore)`
  without importing the concrete class.

### Code example

```python
import asyncio
from agent_framework_ag_ui._snapshots import (
    InMemoryAGUIThreadSnapshotStore,
    AGUIThreadSnapshot,
)

store = InMemoryAGUIThreadSnapshotStore(max_snapshots=200)

async def demo():
    snapshot = AGUIThreadSnapshot(
        messages=[{"role": "user", "content": "Hello"}],
        state={"chart_type": "bar"},
        interrupt=None,
    )

    # Persist after a run
    await store.save(scope="user-42", thread_id="thread-abc", snapshot=snapshot)

    # Retrieve on reconnect
    recovered = await store.get(scope="user-42", thread_id="thread-abc")
    assert recovered is not None
    print(recovered.messages)   # [{"role": "user", "content": "Hello"}]

    # Cleanup
    deleted = await store.delete(scope="user-42", thread_id="thread-abc")
    assert deleted is True

asyncio.run(demo())
```

```python
# Custom Redis-backed store (satisfies AGUIThreadSnapshotStore Protocol)
import json
import redis.asyncio as aioredis
from agent_framework_ag_ui._snapshots import AGUIThreadSnapshot

class RedisSnapshotStore:
    def __init__(self, redis_url: str, ttl_seconds: int = 86400):
        self._redis = aioredis.from_url(redis_url)
        self._ttl = ttl_seconds

    def _key(self, scope: str, thread_id: str) -> str:
        return f"agui:snap:{scope}:{thread_id}"

    async def save(self, *, scope, thread_id, snapshot: AGUIThreadSnapshot):
        payload = {
            "messages": snapshot.messages,
            "state": snapshot.state,
            "interrupt": snapshot.interrupt,
        }
        await self._redis.setex(self._key(scope, thread_id), self._ttl, json.dumps(payload))

    async def get(self, *, scope, thread_id) -> AGUIThreadSnapshot | None:
        raw = await self._redis.get(self._key(scope, thread_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return AGUIThreadSnapshot(**data)

    async def delete(self, *, scope, thread_id) -> bool:
        count = await self._redis.delete(self._key(scope, thread_id))
        return count > 0
```

---

## 4 · `PredictiveStateHandler` + `PredictStateConfig`

**Sub-package:** `agent_framework_ag_ui._orchestration._predictive_state` / `._types`
**Install:** `pip install agent-framework-ag-ui`

`PredictiveStateHandler` intercepts **streaming tool-call argument chunks** and emits partial
Shared State updates to the UI before the tool call completes. This creates a low-latency
"predictive" UX where the UI updates visually as the model generates arguments.

### `PredictStateConfig` (TypedDict)

```python
class PredictStateConfig(TypedDict):
    state_key: str        # the Shared State key to update
    tool: str             # the tool name to watch for
    tool_argument: str | None  # the argument name to read from the tool call
```

### `PredictiveStateHandler.__init__`

```python
PredictiveStateHandler(
    predict_state_config: dict[str, dict[str, str]] | None = None,
    current_state: dict[str, Any] | None = None,
)
```

The `predict_state_config` dict maps **state keys** to `{"tool": ..., "tool_argument": ...}` dicts.

Internal state fields (reset on each tool call via `reset_streaming()`):

| Field | Type | Purpose |
|-------|------|---------|
| `streaming_tool_args` | `str` | Accumulated raw JSON string from streaming chunks |
| `last_emitted_state` | `dict` | Last state dict that was sent to the UI (dedup guard) |
| `state_delta_count` | `int` | Number of deltas emitted for the current call (throttle counter) |
| `pending_state_updates` | `dict` | Buffered updates not yet flushed |

### Key methods

```
reset_streaming() -> None
    # Called at start of each new tool call

extract_state_value(tool_name: str, args: dict | str | None) -> tuple[str, Any] | None
    # Tries to parse accumulated args as JSON and extract the watched argument
    # Returns (state_key, state_value) or None if no match

handle_streaming_chunk(tool_name: str, delta: str) -> dict[str, Any] | None
    # Appends delta to streaming_tool_args and calls extract_state_value
    # Returns partial state update when value changes

handle_tool_complete(tool_name: str, args: dict | str | None) -> dict[str, Any] | None
    # Called when streaming ends; flushes any pending state update
```

### Key facts

- State changes are only emitted when the extracted value **differs** from `last_emitted_state`
  (deduplication on `state_delta_count`). The framework does not emit a state update for every
  chunk — only when the parsed value actually changes.
- `predict_state_config` is a flat dict where the keys are **state keys** (not tool names).
  Multiple state keys can watch the same tool (emit different state fields as different args
  stream in).

### Code example

```python
from agent_framework_ag_ui._orchestration._predictive_state import PredictiveStateHandler

# Agent has a `generate_report` tool with arguments:
#   { "title": str, "format": str, "data": list }
# We want to stream `report_title` and `report_format` to the UI in real time.

config = {
    "report_title": {"tool": "generate_report", "tool_argument": "title"},
    "report_format": {"tool": "generate_report", "tool_argument": "format"},
}

handler = PredictiveStateHandler(predict_state_config=config)

# During streaming
handler.reset_streaming()  # called once per tool invocation

chunks = ['{"title": "Q', '1 Sales"', ', "format": "pdf"}']
for chunk in chunks:
    update = handler.handle_streaming_chunk("generate_report", chunk)
    if update:
        # Emit STATE_DELTA event to the UI with partial values
        print("Partial state update:", update)

# After tool completes
final_update = handler.handle_tool_complete("generate_report", None)
if final_update:
    print("Final state update:", final_update)
```

```python
# Wire into AgentFrameworkAgent via predict_state_config
from agent_framework_ag_ui import AgentFrameworkAgent

predict_config = {
    "chart_type": {"tool": "update_chart", "tool_argument": "type"},
    "chart_title": {"tool": "update_chart", "tool_argument": "title"},
}

agent = AgentFrameworkAgent(
    agent=my_agent,
    predict_state_config=predict_config,
    # AgentFrameworkAgent creates PredictiveStateHandler internally
)
```

---

## 5 · `AGUIRequest` + `AGUIChatOptions` + `AgentState` + `RunMetadata`

**Sub-package:** `agent_framework_ag_ui._types`
**Install:** `pip install agent-framework-ag-ui`

These types form the AG-UI protocol boundary — they are the structured
request/response types for AG-UI HTTP endpoints.

### `AGUIRequest` (Pydantic BaseModel)

```python
class AGUIRequest(BaseModel):
    messages: list[dict[str, Any]]            # required — AG-UI format messages
    run_id: str | None                        # AliasChoices("run_id", "runId")
    thread_id: str | None                     # AliasChoices("thread_id", "threadId")
    state: dict[str, Any] | None              # Shared State for generative UI
    tools: list[dict[str, Any]] | None        # client-side tool schemas
    context: list[dict[str, Any]] | None      # context objects for the agent
    forwarded_props: dict[str, Any] | None    # AliasChoices("forwarded_props", "forwardedProps")
    parent_run_id: str | None                 # AliasChoices("parent_run_id", "parentRunId")
    available_interrupts: list[dict] | None   # AliasChoices("availableInterrupts", ...)
```

`AliasChoices` means **both** snake_case and camelCase are accepted. A frontend sending
`{ "threadId": "abc", "runId": "xyz" }` is equivalent to `{ "thread_id": "abc", "run_id": "xyz" }`.

### `AGUIChatOptions` (TypedDict)

Extends `ChatOptions` with AG-UI-specific keys:

- Inherits: `model`, `temperature`, `top_p`, `max_tokens`, `stop`, `tool_choice`.
- **`tools`** — sent to the remote server so the LLM sees client-side tool schemas; client tools
  execute locally via the framework's `FunctionInvocationLayer`.
- **`metadata`** — dict containing `thread_id` for conversation continuity across multiple HTTP
  calls.

### `AgentState` (TypedDict)

```python
class AgentState(TypedDict):
    messages: list[Any] | None
```

Minimal base for AG-UI agent state. Subclass this for typed state schemas.

### `RunMetadata` (TypedDict)

```python
class RunMetadata(TypedDict):
    run_id: str
    thread_id: str
    predict_state: list[PredictStateConfig] | None
```

Carries per-run metadata forwarded from the AG-UI client to the server. `predict_state` is
the typed list form of `predict_state_config` — the server merges it with any static config
set on the agent wrapper.

### Code example

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agent_framework_ag_ui._types import AGUIRequest

app = FastAPI()
ag_ui_agent = ...  # Replace with your actual AgentFrameworkAgent instance

@app.post("/agent/run")
async def run_agent(request: AGUIRequest):
    """AG-UI endpoint — accepts both snake_case and camelCase."""
    print(f"Thread: {request.thread_id}, Run: {request.run_id}")
    print(f"State: {request.state}")
    print(f"Messages count: {len(request.messages)}")

    async def event_stream():
        async for event in ag_ui_agent.run(request.model_dump(by_alias=False)):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

```python
# Using AGUIChatOptions with an AG-UI client
from agent_framework.openai import OpenAIResponsesClient
from agent_framework_ag_ui._types import AGUIChatOptions

client = OpenAIResponsesClient(model="gpt-4o")
agent = client.as_agent(instructions="You are a helpful assistant.")

# Chat options for a thread-aware request
options: AGUIChatOptions = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "metadata": {"thread_id": "thread-abc"},
}
```

---

## 6 · `A2AAgentSession` + `A2AContinuationToken`

**Sub-package:** `agent_framework_a2a._agent`
**Install:** `pip install agent-framework-a2a`
**Import:** `from agent_framework.a2a import A2AAgentSession, A2AContinuationToken`

`A2AAgent` (documented in [Vol. 12](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v12/))
uses these two types to maintain conversation continuity across multiple HTTP round-trips with
a remote A2A server. This volume covers the session and continuation token classes that were
not documented there.

### `A2AAgentSession`

```python
class A2AAgentSession(AgentSession):
    _CONTEXT_ID_KEY = "a2a_context_id"
    _TASK_ID_KEY    = "a2a_task_id"
    _TASK_STATE_KEY = "a2a_task_state"

    def __init__(
        self,
        *,
        context_id: str | None = None,
        task_id: str | None = None,
        task_state: TaskState | None = None,
    )
```

- **`context_id`** — the A2A conversation context identifier; passed as
  `super().__init__(service_session_id=context_id)` so the base `AgentSession` uses it
  for session key storage.
- **`task_id`** — the most recent A2A task ID returned by the remote agent.
- **`task_state`** — the last known `TaskState` (`"completed"`, `"input-required"`, etc.).
  The framework checks this to detect whether a follow-up message should be sent as a
  **task continuation** (when state is `"input-required"`) vs. a **new task**.
- Serialises to/from `dict` via `to_dict()` / `AgentSession.from_dict()`.

### `A2AContinuationToken`

```python
class A2AContinuationToken(ContinuationToken):
    task_id: str
    context_id: str
```

Extends `ContinuationToken` (see [Vol. 21](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v21/)) with A2A-specific fields. When a long-running A2A task is still in progress, `A2AAgent.run()` returns an `AgentResponseUpdate` whose `continuation_token` is an `A2AContinuationToken`. The caller can resume the task by passing the token back:

```python
response = await agent.run("Process this large dataset", session=session)
while response.continuation_token:
    token = response.continuation_token  # A2AContinuationToken
    response = await agent.run(None, session=session, continuation_token=token)
print(response.text)
```

### Key facts

- `A2AAgentSession` stores `task_id` and `task_state` across turns in its slot keys, so the
  session object must be **passed back** on every subsequent call to the same conversation.
- When `task_state == "input-required"`, the next `agent.run()` call sends a **task
  continuation** (not a new task). When `task_state == "completed"` or `None`, a new task
  is created.
- `context_id` maps to A2A's `contextId` field in the JSON-RPC request envelope, enabling
  multi-turn conversation tracking on the remote server.

### Code example

```python
import asyncio
from agent_framework.a2a import A2AAgent, A2AAgentSession

async def multi_turn_conversation():
    agent = A2AAgent(
        url="http://remote-agent:9999/",
        name="RemoteAgent",
        description="A remote A2A-compatible agent",
        timeout=30.0,
    )

    session = A2AAgentSession()  # starts with no context_id

    # First turn
    response = await agent.run("What is the capital of France?", session=session)
    print(response.text)  # "Paris"
    print(f"Context: {session.context_id}")  # populated after first run

    # Second turn — same session carries context_id to the remote
    response = await agent.run("And what is its population?", session=session)
    print(response.text)  # "Approximately 2.2 million..."

asyncio.run(multi_turn_conversation())
```

```python
# Serialise session across requests (e.g., store in a database)
session_data = session.to_dict()
# {"service_session_id": "ctx-xyz", "a2a_context_id": "ctx-xyz",
#  "a2a_task_id": "task-abc", "a2a_task_state": "completed"}

# Restore on next request
restored_session = A2AAgentSession.from_dict(session_data)
```

---

## 7 · `ThreadItemConverter`

**Sub-package:** `agent_framework_chatkit._converter`
**Install:** `pip install agent-framework-chatkit`
**Import:** `from agent_framework_chatkit import ThreadItemConverter`

`ThreadItemConverter` bridges **OpenAI ChatKit thread items** (the message format used
by the `openai-chatkit` library) and agent-framework `Message` objects. Subclass it to
handle attachments, `@`-mentions, hidden context items, and custom thread item types.

### Constructor

```python
ThreadItemConverter(
    attachment_data_fetcher: Callable[[str], Awaitable[bytes]] | None = None,
)
```

- **`attachment_data_fetcher`** — async callback `(attachment_id: str) -> bytes`. When
  provided, attachments are fetched as binary data and converted to `BinaryContent` items.
  When `None`, attachments fall back to `UriContent` using available download URLs.

### Key methods

| Method | Signature | Notes |
|--------|-----------|-------|
| `to_agent_input` | `async (items: list[ThreadItem]) -> list[Message]` | Main entry point — normalises all item types and returns ordered messages |
| `user_message_to_input` | `async (item: UserMessageItem, is_last_message: bool) -> Message \| list[Message] \| None` | Override to customise user message conversion |
| `agent_message_to_input` | `async (item: AgentMessageItem) -> Message \| None` | Override to include (or exclude) previous assistant turns |
| `hidden_item_to_input` | `async (item: HiddenItem) -> Message \| None` | Override to inject hidden context into the agent's input |
| `mention_to_input` | `async (item: MentionItem) -> Message \| None` | Override to handle `@`-mentioned agents/users |

### Key facts

- `to_agent_input()` is the **only method you should call directly**. The individual
  `*_to_input()` methods are internal hooks for subclasses.
- **Attachment handling:** when `attachment_data_fetcher` is set and the attachment has a
  known MIME type, the binary data is wrapped in a `BinaryContent` item with the detected
  content type. When the fetcher is `None` or the attachment has no binary representation,
  a `UriContent` is created instead.
- **Quoted messages:** The converter detects `quoted_text` in `UserMessageItem` and
  injects it as a separate quoted-context message only on the **last user message**
  (`is_last_message=True`). Prior messages with quoted text are converted without the quote.

### Code example

```python
import asyncio
from agent_framework_chatkit import ThreadItemConverter
from agent_framework.openai import OpenAIResponsesClient

# Basic usage without attachment fetching
converter = ThreadItemConverter()

async def chat(thread_items):
    messages = await converter.to_agent_input(thread_items)
    client = OpenAIResponsesClient(model="gpt-4o")
    async with client.as_agent(instructions="You are a helpful assistant.") as agent:
        response = await agent.run(messages)
        return response.text
```

```python
# With binary attachment fetching
import httpx

async def fetch_attachment(attachment_id: str) -> bytes:
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"https://files.example.com/{attachment_id}")
        return resp.content

converter = ThreadItemConverter(attachment_data_fetcher=fetch_attachment)

# Custom subclass to filter @-mentions
class FilteringConverter(ThreadItemConverter):
    async def mention_to_input(self, item):
        """Ignore @mentions in the agent's input."""
        return None

    async def hidden_item_to_input(self, item):
        """Inject hidden context as a system-role message."""
        from agent_framework._types import Message
        return Message(role="system", contents=[item.text])

filtering_converter = FilteringConverter(attachment_data_fetcher=fetch_attachment)
```

---

## 8 · `AgentApprovalExecutor` + `AgentRequestInfoExecutor` + `AgentRequestInfoResponse`

**Sub-package:** `agent_framework_orchestrations._orchestration_request_info`
**Install:** `pip install agent-framework-orchestrations`
**Import:** `from agent_framework.orchestrations import AgentApprovalExecutor`

`AgentApprovalExecutor` implements the **agent approval gate pattern**: an agent runs, a human
reviews its output, and either approves (sending the response downstream) or provides correction
messages (causing the agent to run again). It wraps an internal sub-workflow containing an
`AgentExecutor` and an `AgentRequestInfoExecutor`.

### `AgentApprovalExecutor.__init__`

```python
AgentApprovalExecutor(
    agent: SupportsAgentRun,
    context_mode: Literal["full", "last_agent", "custom"] | None = None,
    *,
    allow_direct_output: bool = False,
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `agent` | — | Any `SupportsAgentRun` — `Agent`, `ClaudeAgent`, `FoundryAgent`, etc. |
| `context_mode` | `None` | How prior conversation history is provided to the agent: `"full"` (all history), `"last_agent"` (only the last agent turn), `"custom"` (caller-managed) |
| `allow_direct_output` | `False` | When `True`, the agent's approved response surfaces as the **workflow's output** event. Set this when `AgentApprovalExecutor` is the terminal node of the workflow. |

### `AgentRequestInfoExecutor`

The inner executor that presents the agent's response to the human and routes the reply:

```python
class AgentRequestInfoExecutor(Executor):
    @handler
    async def request_info(self, agent_response: AgentExecutorResponse, ctx: WorkflowContext) -> None:
        await ctx.request_info(agent_response, AgentRequestInfoResponse)

    @response_handler
    async def handle_request_info_response(
        self,
        original_request: AgentExecutorResponse,
        response: AgentRequestInfoResponse,
        ctx: WorkflowContext[AgentExecutorRequest, AgentExecutorResponse],
    ) -> None:
        if response.messages:
            # User provided follow-up — re-run the agent
            await ctx.send_message(AgentExecutorRequest(
                messages=response.messages, should_respond=True
            ))
        else:
            # No follow-up messages — approve as-is and pass downstream
            await ctx.yield_output(original_request)
```

### `AgentRequestInfoResponse`

```python
@dataclass
class AgentRequestInfoResponse:
    messages: list[Message]

    @staticmethod
    def from_messages(messages: list[Message]) -> "AgentRequestInfoResponse": ...
    @staticmethod
    def from_strings(texts: list[str]) -> "AgentRequestInfoResponse": ...
    @staticmethod
    def approve() -> "AgentRequestInfoResponse":
        return AgentRequestInfoResponse(messages=[])  # empty = approve
```

- `approve()` creates a response with `messages=[]`, which the executor interprets as an
  approval of the agent's original output.
- `from_strings(["Please be more concise."])` wraps each string in a `Message(role="user")`.

### Key facts

- The approval pattern works by having `AgentRequestInfoExecutor` emit a
  `request_info` event (pausing workflow execution) and then a `response_handler`
  that routes the human's reply. This is the same HITL mechanism used by
  `WorkflowAgent` (see [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/)).
- `allow_direct_output=True` must be set when the approval executor is the **last node**
  in the workflow; otherwise the approved `AgentExecutorResponse` is forwarded as a
  message to a downstream participant (which may not exist).
- A `_TerminalAgentRequestInfoExecutor` (private) is used internally when
  `allow_direct_output=True` to ensure the workflow's `output` event carries an
  `AgentResponse` rather than an `AgentExecutorResponse`.

### Code example

```python
import asyncio
from agent_framework import Workflow
from agent_framework.openai import OpenAIResponsesClient
from agent_framework.orchestrations import AgentApprovalExecutor

async def human_review_loop():
    client = OpenAIResponsesClient(model="gpt-4o")
    agent = client.as_agent(
        name="writer",
        instructions="You are a professional copywriter. Write compelling content.",
    )

    # Build a workflow where human approves every agent response
    workflow = (
        Workflow()
        .add_executor(
            AgentApprovalExecutor(
                agent=agent,
                context_mode="full",
                allow_direct_output=True,  # terminal node
            )
        )
    )

    # The workflow pauses at AgentRequestInfoExecutor waiting for a response.
    # Pass AgentRequestInfoResponse.approve() to approve, or from_strings(...) to correct.
    from agent_framework._types import Message
    from agent_framework.orchestrations import AgentRequestInfoResponse

    result = await workflow.run(
        "Write a tagline for a new coffee brand called 'Aurora Roast'.",
        responses={
            # request_id is discovered from the paused workflow's pending requests
            "some-request-id": AgentRequestInfoResponse.from_strings(
                ["Make it more poetic and mention the sunrise."]
            ),
        },
    )
    print(result.text)

asyncio.run(human_review_loop())
```

```python
# Non-terminal: approval executor feeds into a downstream formatter agent
from agent_framework.orchestrations import SequentialBuilder

formatter = client.as_agent(
    name="formatter",
    instructions="Format the approved content as HTML.",
)

workflow = (
    SequentialBuilder()
    .add(AgentApprovalExecutor(agent=writer_agent, allow_direct_output=False))
    .add(formatter)
    .build()
)
```

---

## 9 · `AgentFrameworkWorkflow`

**Sub-package:** `agent_framework_ag_ui._workflow`
**Install:** `pip install agent-framework-ag-ui`
**Import:** `from agent_framework_ag_ui import AgentFrameworkWorkflow`

`AgentFrameworkWorkflow` is the AG-UI equivalent of `AgentFrameworkAgent` (§2) but for
**`Workflow`** objects rather than single agents. It translates workflow execution (including
HITL interrupts, fan-out, fan-in, and checkpointing) into the AG-UI streaming event protocol.

### Constructor

```python
AgentFrameworkWorkflow(
    workflow: Workflow | None = None,
    *,
    workflow_factory: WorkflowFactory | None = None,
    name: str | None = None,
    description: str | None = None,
    snapshot_store: AGUIThreadSnapshotStore | None = None,
)
```

| Parameter | Notes |
|-----------|-------|
| `workflow` | A single shared `Workflow` instance. Raises `ValueError` if also passing `workflow_factory`. |
| `workflow_factory` | Per-thread workflow factory. Used when each conversation thread needs its own workflow instance (e.g., per-thread checkpointing). Mutually exclusive with `workflow`. |
| `name` / `description` | Defaults to `workflow.name` / `workflow.description` if not provided. |
| `snapshot_store` | Persists `AGUIThreadSnapshot` after each run (same protocol as §3). Requires a Snapshot Scope resolver in the endpoint setup to activate. |

### Internal caching

The `_workflow_by_thread` dict caches workflow instances by `(snapshot_scope, thread_id)`:

```python
self._workflow_by_thread: dict[tuple[str | None, str], Workflow] = {}
```

The scope is part of the composite cache key because two users with the same `thread_id`
must not share a workflow instance (and its in-memory checkpoint). Passing
`snapshot_scope=None` creates a separate entry for scope-less usage.

### Key facts

- **`workflow` vs `workflow_factory`** — pass `workflow=` for a stateless workflow that
  resets on every run (no per-thread memory). Pass `workflow_factory=` when each thread
  needs a fresh `Workflow` with its own `FileCheckpointStorage` or `InMemoryCheckpointStorage`.
- When `workflow_factory` is set, the factory is called once per `(scope, thread_id)` pair
  and the result is cached. The same workflow instance handles all future runs on that thread.
- Subclass `AgentFrameworkWorkflow` and override `run()` to inject custom logic (e.g.,
  pre/post-processing of AG-UI events) before delegating to `run_workflow_stream()`.

### Code example

```python
from agent_framework.openai import OpenAIResponsesClient
from agent_framework.orchestrations import SequentialBuilder
from agent_framework_ag_ui import AgentFrameworkWorkflow
from agent_framework_ag_ui._snapshots import InMemoryAGUIThreadSnapshotStore

client = OpenAIResponsesClient(model="gpt-4o")

# Build a two-stage sequential workflow
researcher = client.as_agent(name="researcher", instructions="Research the topic.")
writer = client.as_agent(name="writer", instructions="Write a report based on the research.")

workflow = SequentialBuilder().add(researcher).add(writer).build()

# Expose via AG-UI
ag_ui_workflow = AgentFrameworkWorkflow(
    workflow=workflow,
    name="research-pipeline",
    description="Research and write a report",
    snapshot_store=InMemoryAGUIThreadSnapshotStore(),
)

# In a FastAPI endpoint:
async def stream_events(request_data: dict):
    async for event in ag_ui_workflow.run(request_data):
        yield f"data: {event.model_dump_json()}\n\n"
```

```python
# Per-thread workflow with checkpointing
from agent_framework import Workflow
from agent_framework.checkpointing import InMemoryCheckpointStorage

def make_workflow() -> Workflow:
    storage = InMemoryCheckpointStorage()
    return SequentialBuilder().add(researcher).add(writer).build(
        checkpoint_storage=storage
    )

ag_ui_workflow = AgentFrameworkWorkflow(
    workflow_factory=make_workflow,  # one Workflow + one checkpoint store per thread
    name="stateful-pipeline",
)
```

---

## 10 · `FlowState` + `AGUIHttpService` + `run_workflow_stream`

**Sub-packages:** `agent_framework_ag_ui._run_common` / `._http_service` / `._workflow_run`
**Install:** `pip install agent-framework-ag-ui`

### `FlowState` (dataclass)

`FlowState` is the **mutable run-level state** maintained by `run_workflow_stream` (and the
`AgentFrameworkAgent` run loop) during a single AG-UI streaming run. It is not part of the
public API but understanding its fields clarifies the event emission logic.

```python
@dataclass
class FlowState:
    message_id: str | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    waiting_for_approval: bool = False
    current_state: dict[str, Any] = field(default_factory=dict)
    accumulated_text: str = ""
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_ended: set[str] = field(default_factory=set)
    interrupts: list[dict[str, Any]] = field(default_factory=list)
    reasoning_messages: list[dict[str, Any]] = field(default_factory=list)
    accumulated_reasoning: dict[str, str] = field(default_factory=dict)
    reasoning_message_id: str | None = None
```

| Field | Purpose |
|-------|---------|
| `message_id` | The current `TEXT_MESSAGE_START` message ID being streamed |
| `tool_call_id` / `tool_call_name` | The in-flight tool call (one at a time) |
| `waiting_for_approval` | `True` after a `TOOL_CALL_START` is emitted and before approval |
| `current_state` | Accumulated Shared State updates (merged on each `STATE_DELTA`) |
| `accumulated_text` | Full text accumulated across streaming `TEXT_MESSAGE_CONTENT` chunks |
| `pending_tool_calls` | Tool calls queued but not yet sent to the agent |
| `tool_calls_by_id` | Map from call ID to its full tool-call dict (for name lookup) |
| `tool_results` | Results gathered during the run (for snapshot) |
| `tool_calls_ended` | Set of call IDs whose `TOOL_CALL_END` has been emitted |
| `interrupts` | Pending interrupt payloads for `RUN_FINISHED.interrupt` |
| `reasoning_messages` | Extended-thinking reasoning message chunks |
| `accumulated_reasoning` | Accumulated reasoning text keyed by `reasoning_message_id` |
| `reasoning_message_id` | The current reasoning message ID being accumulated |

### `AGUIHttpService`

```python
AGUIHttpService(
    endpoint: str,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = 60.0,
)
```

HTTP client for **consuming** a remote AG-UI server endpoint (the client side of the
AG-UI protocol). It posts a run request and parses the Server-Sent Events (SSE) stream.

```python
async def post_run(
    self,
    thread_id: str,
    run_id: str,
    messages: list[dict[str, Any]],
    state: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    available_interrupts: list[dict[str, Any]] | None = None,
    resume: dict[str, Any] | None = None,
) -> AsyncIterable[dict[str, Any]]:
```

- Posts to `self.endpoint + "/"` with a JSON body.
- Parses the SSE `data:` lines as JSON and yields each event dict.
- Supports `resume` for sending interrupt responses back to the server.
- Is an async context manager: `async with AGUIHttpService(...) as svc:` manages the
  underlying `httpx.AsyncClient` lifecycle.

### `run_workflow_stream`

```python
async def run_workflow_stream(
    input_data: dict[str, Any],
    workflow: Workflow,
) -> AsyncGenerator[BaseEvent]:
```

The core function that drives workflow execution and emits AG-UI events. It:

1. Resolves `thread_id` from `input_data["thread_id"]` (or `"threadId"`) — generates a UUID if absent.
2. Resolves `run_id` similarly.
3. Normalises incoming `messages` via `normalize_agui_input_messages()`.
4. Emits `RUN_STARTED`.
5. Runs the workflow, translating each `WorkflowEvent` to the appropriate AG-UI event.
6. Emits `RUN_FINISHED` (with optional `interrupt` payloads if the workflow paused).

### Code example

```python
import asyncio
from agent_framework_ag_ui._http_service import AGUIHttpService

async def call_remote_agent():
    """Consume a remote AG-UI server as a client."""
    async with AGUIHttpService("http://localhost:8888", timeout=120.0) as svc:
        async for event in svc.post_run(
            thread_id="thread-abc",
            run_id="run-001",
            messages=[{"role": "user", "content": "Hello, what can you help me with?"}],
            state={"user_preferences": {"language": "en"}},
        ):
            event_type = event.get("type")
            if event_type == "TEXT_MESSAGE_CONTENT":
                print(event.get("delta", ""), end="", flush=True)
            elif event_type == "RUN_FINISHED":
                print("\n[Run finished]")
                break

asyncio.run(call_remote_agent())
```

```python
# Resume an interrupted workflow
async with AGUIHttpService("http://localhost:8888") as svc:
    # First call — workflow pauses awaiting approval
    events = []
    async for event in svc.post_run(
        thread_id="thread-xyz",
        run_id="run-001",
        messages=[{"role": "user", "content": "Send email to all users."}],
    ):
        events.append(event)
        if event.get("type") == "RUN_FINISHED":
            interrupt_payload = event.get("interrupt", [])
            break

    # Resume with approval
    async for event in svc.post_run(
        thread_id="thread-xyz",
        run_id="run-002",
        messages=[],  # no new user message
        resume={
            "type": "interrupt",
            "interrupts": [
                {"id": interrupt_payload[0]["id"], "value": {"approved": True}}
            ],
        },
    ):
        if event.get("type") == "TEXT_MESSAGE_CONTENT":
            print(event.get("delta", ""), end="")
```

```python
# Using run_workflow_stream directly (server-side)
from agent_framework_ag_ui._workflow_run import run_workflow_stream
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
my_workflow = ...  # Replace with your actual AgentFrameworkWorkflow or Workflow instance

@app.post("/run")
async def run(request: dict):
    async def stream():
        async for event in run_workflow_stream(request, my_workflow):
            yield f"data: {json.dumps(event.model_dump())}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
```

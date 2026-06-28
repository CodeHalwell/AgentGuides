---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 26"
description: "Source-verified deep dives into 10 class groups from agent-framework-durabletask 1.0.0b260521: DurableAIAgentWorker+DurableAIAgentClient (worker registration and external client), DurableAIAgent+DurableAgentExecutor (proxy shim and abstract executor), DurableAIAgentOrchestrationContext (orchestration-context wrapper), AgentEntity+AgentEntityStateProviderMixin (platform-agnostic entity execution and state caching mixin), AgentCallbackContext+AgentResponseCallbackProtocol (streaming and final-response callback protocol), RunRequest (request data model — correlation IDs, role coercion, fire-and-forget mode), AgentSessionId+DurableAgentSession (entity naming, @name@key parse format, session serialization), DurableAgentState+DurableAgentStateData (root state container — schema 1.1.0, from_dict/from_json, try_get_agent_response), DurableAgentStateEntry+Request+Response+Usage (entry hierarchy — $type discriminator, is_error flag, usage token tracking), DurableAgentStateContent hierarchy+DurableStateFields+ContentTypes (9 content type subclasses, from_ai_content factory, camelCase field constants)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 49
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 26

Verified against **agent-framework-durabletask 1.0.0b260521** / **agent-framework 1.9.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source using `inspect.getsource()`. Sub-packages introspected:
`agent_framework_durabletask._worker`,
`agent_framework_durabletask._client`,
`agent_framework_durabletask._shim`,
`agent_framework_durabletask._executors`,
`agent_framework_durabletask._orchestration_context`,
`agent_framework_durabletask._entities`,
`agent_framework_durabletask._callbacks`,
`agent_framework_durabletask._models`,
`agent_framework_durabletask._durable_agent_state`,
`agent_framework_durabletask._constants`.

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

This volume covers **ten class groups** drawn entirely from the **`agent-framework-durabletask`** sub-package — the integration layer that turns any `agent-framework` agent into a long-running Azure Durable Entity. `DurableAIAgent`, `DurableAIAgentWorker`, and `DurableAIAgentClient` were briefly introduced in [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/); this volume provides the complete source-verified treatment covering the entire sub-package for the first time.

| # | Class / group | Module |
|---|---|---|
| 1 | `DurableAIAgentWorker` · `DurableAIAgentClient` | `agent_framework_durabletask._worker` / `._client` |
| 2 | `DurableAIAgent` · `DurableAgentExecutor` | `._shim` / `._executors` |
| 3 | `DurableAIAgentOrchestrationContext` | `._orchestration_context` |
| 4 | `AgentEntity` · `AgentEntityStateProviderMixin` | `._entities` |
| 5 | `AgentCallbackContext` · `AgentResponseCallbackProtocol` | `._callbacks` |
| 6 | `RunRequest` | `._models` |
| 7 | `AgentSessionId` · `DurableAgentSession` | `._models` |
| 8 | `DurableAgentState` · `DurableAgentStateData` | `._durable_agent_state` |
| 9 | `DurableAgentStateEntry` · `DurableAgentStateRequest` · `DurableAgentStateResponse` · `DurableAgentStateUsage` | `._durable_agent_state` |
| 10 | `DurableAgentStateContent` · content subclasses · `DurableStateFields` · `ContentTypes` | `._durable_agent_state` / `._constants` |

---

## 1 · `DurableAIAgentWorker` + `DurableAIAgentClient`

**Sub-packages:** `agent_framework_durabletask._worker` · `agent_framework_durabletask._client`  
**Install:** `pip install agent-framework-durabletask`

`DurableAIAgentWorker` wraps a `TaskHubGrpcWorker` and registers `agent-framework` agents as Azure Durable Entities named `dafx-{agent_name}`. `DurableAIAgentClient` wraps a `TaskHubGrpcClient` and returns `DurableAIAgent` proxy objects for external callers to drive those entities.

### Key source facts

- `add_agent()` validates that the agent has a non-empty `name` and is not already registered; it then calls an inner factory that creates a `ConfiguredAgentEntity` subclass and registers it with `worker.add_entity()`.
- The dynamically generated entity class name is set to `dafx-{agent_name}` via `__name__` and `__qualname__` so that durabletask uses the prefixed name as the entity key.
- `DurableAIAgentClient.__init__` clamps `max_poll_retries` to `max(1, value)` and resets `poll_interval_seconds` to its default if ≤ 0.
- `get_agent()` on both worker and client does **not** validate that the entity exists — validation is deferred to the first `run()` call.

### Signatures

```python
class DurableAIAgentWorker:
    def __init__(
        self,
        worker: TaskHubGrpcWorker,
        callback: AgentResponseCallbackProtocol | None = None,
    ) -> None: ...

    def add_agent(
        self,
        agent: SupportsAgentRun,
        callback: AgentResponseCallbackProtocol | None = None,
    ) -> None: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    registered_agent_names: list[str]  # property

class DurableAIAgentClient:
    def __init__(
        self,
        client: TaskHubGrpcClient,
        max_poll_retries: int = DEFAULT_MAX_POLL_RETRIES,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None: ...

    def get_agent(self, agent_name: str) -> DurableAIAgent[AgentResponse]: ...
```

### Example 1 — Registering two agents and starting the worker

```python
from durabletask.worker import TaskHubGrpcWorker
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_durabletask import DurableAIAgentWorker

worker = TaskHubGrpcWorker(host_address="localhost:4001")
agent_worker = DurableAIAgentWorker(worker)

client = OpenAIChatCompletionClient(model="gpt-4o")
agent_worker.add_agent(Agent(client=client, name="assistant"))
agent_worker.add_agent(Agent(client=client, name="researcher"))

print(agent_worker.registered_agent_names)  # ['assistant', 'researcher']
# Entities registered as 'dafx-assistant' and 'dafx-researcher'
worker.start()
```

### Example 2 — Calling a durable agent from an external client

```python
from durabletask import TaskHubGrpcClient
from agent_framework_durabletask import DurableAIAgentClient

# Wrap the gRPC client
grpc_client = TaskHubGrpcClient(host_address="localhost:4001")
agent_client = DurableAIAgentClient(
    grpc_client,
    max_poll_retries=30,       # poll up to 30 times
    poll_interval_seconds=2.0, # 2 s between polls
)

agent = agent_client.get_agent("assistant")
response = agent.run("Summarise the latest earnings report.")
print(response.text)
```

### Example 3 — Per-agent callback at registration time

```python
from durabletask.worker import TaskHubGrpcWorker
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_durabletask import DurableAIAgentWorker
from agent_framework_durabletask import AgentCallbackContext, AgentResponseCallbackProtocol
from agent_framework_core._types import AgentResponse, AgentResponseUpdate

class AuditCallback(AgentResponseCallbackProtocol):
    async def on_streaming_response_update(
        self, update: AgentResponseUpdate, context: AgentCallbackContext
    ) -> None:
        print(f"[{context.agent_name}] streaming chunk received")

    async def on_agent_response(
        self, response: AgentResponse, context: AgentCallbackContext
    ) -> None:
        print(f"[{context.agent_name}] final: {response.text[:80]}")

worker = TaskHubGrpcWorker(host_address="localhost:4001")
agent_worker = DurableAIAgentWorker(worker)

client = OpenAIChatCompletionClient(model="gpt-4o")
# Override callback for only the 'audited' agent
agent_worker.add_agent(
    Agent(client=client, name="audited"),
    callback=AuditCallback(),
)
worker.start()
```

---

## 2 · `DurableAIAgent` + `DurableAgentExecutor`

**Sub-packages:** `agent_framework_durabletask._shim` · `agent_framework_durabletask._executors`  
**Install:** `pip install agent-framework-durabletask`

`DurableAIAgent` is a generic proxy that implements `SupportsAgentRun` but **returns `TaskT` synchronously** rather than a coroutine — it delegates to an injected `DurableAgentExecutor`. `DurableAgentExecutor` is the abstract base that handles entity routing, session ID creation, and `RunRequest` construction.

### Key source facts

- `DurableAIAgent.run()` raises `ValueError` if `stream=True` is passed — durable agents do not support streaming mode because the entity call must complete in a single request/response cycle.
- `DurableAgentExecutor.get_run_request()` pops the keys `response_format`, `enable_tool_calls`, and `wait_for_response` from the `options` dict before setting `options` on `RunRequest` — so callers can pass these in the options dict and they reach the correct field without leaking into the arbitrary options payload.
- `DurableAgentExecutor._create_session_id()` reuses an existing `DurableAgentSession.durable_session_id` when one is provided; otherwise it generates a new `uuid4().hex` key.
- `DurableAIAgent.id` defaults to `name` when `agent_id` is omitted.

### Signatures

```python
class DurableAIAgent(SupportsAgentRun, Generic[TaskT]):
    id: str
    name: str
    display_name: str
    description: str | None

    def __init__(
        self,
        executor: DurableAgentExecutor[TaskT],
        name: str,
        *,
        agent_id: str | None = None,
    ) -> None: ...

    def run(  # type: ignore[override]
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = False,
        session: AgentSession | None = None,
        options: dict[str, Any] | None = None,
    ) -> TaskT: ...

class DurableAgentExecutor(ABC, Generic[TaskT]):
    @abstractmethod
    def run_durable_agent(
        self,
        agent_name: str,
        run_request: RunRequest,
        session: AgentSession | None = None,
    ) -> TaskT: ...

    def get_new_session(self, agent_name: str, ...) -> DurableAgentSession: ...
    def get_run_request(self, message: str, *, options: dict | None = None) -> RunRequest: ...
```

### Example 1 — Using the shim from an orchestration (yield pattern)

```python
from durabletask import OrchestrationContext
from agent_framework_durabletask import DurableAIAgentOrchestrationContext

def research_orchestration(ctx: OrchestrationContext):
    agent_ctx = DurableAIAgentOrchestrationContext(ctx)
    researcher = agent_ctx.get_agent("researcher")

    # run() returns a DurableAgentTask — must yield, not await
    result = yield researcher.run("Find the top AI papers from last month.")
    return result.text
```

### Example 2 — Passing structured options through the shim

```python
from pydantic import BaseModel
from durabletask import TaskHubGrpcClient
from agent_framework_durabletask import DurableAIAgentClient

class Summary(BaseModel):
    title: str
    points: list[str]

grpc_client = TaskHubGrpcClient(host_address="localhost:4001")
agent_client = DurableAIAgentClient(grpc_client)
agent = agent_client.get_agent("assistant")

response = agent.run(
    "Summarise this quarter's results.",
    options={
        "response_format": Summary,   # extracted by get_run_request()
        "enable_tool_calls": False,   # extracted by get_run_request()
        "temperature": 0.2,           # forwarded as-is in RunRequest.options
    },
)
print(response.text)
```

### Example 3 — Reusing a session across calls

```python
from durabletask import TaskHubGrpcClient
from agent_framework_durabletask import DurableAIAgentClient
from agent_framework_durabletask._models import AgentSessionId, DurableAgentSession

grpc_client = TaskHubGrpcClient(host_address="localhost:4001")
agent_client = DurableAIAgentClient(grpc_client)
agent = agent_client.get_agent("assistant")

# Build a stable session using the public model types (see §7 for full API)
sid = AgentSessionId.with_random_key("assistant")
session = DurableAgentSession.from_session_id(sid)

r1 = agent.run("Hello! My name is Alice.", session=session)
print(r1.text)

# Second call — same session keeps conversation context
r2 = agent.run("What is my name?", session=session)
print(r2.text)  # "Your name is Alice."
```

---

## 3 · `DurableAIAgentOrchestrationContext`

**Sub-package:** `agent_framework_durabletask._orchestration_context`  
**Install:** `pip install agent-framework-durabletask`

Wraps a `durabletask` `OrchestrationContext` so that orchestration functions can call durable agents using the same `get_agent()` / `agent.run()` pattern as external callers — but via entity calls that return a `DurableAgentTask` (yield-compatible) rather than an `AgentResponse`.

### Key source facts

- Accepts any `OrchestrationContext` from `durabletask` — the wrapper is thin: it creates an `OrchestrationAgentExecutor` internally and does nothing else at construction time.
- `get_agent()` returns `DurableAIAgent[DurableAgentTask]`; the caller must **yield** the result to allow durabletask to suspend and resume the orchestration.
- The entity is addressed as `dafx-{agent_name}` — the `ENTITY_NAME_PREFIX` constant from `AgentSessionId`.

### Signature

```python
class DurableAIAgentOrchestrationContext(DurableAgentProvider[DurableAgentTask]):
    def __init__(self, context: OrchestrationContext) -> None: ...
    def get_agent(self, agent_name: str) -> DurableAIAgent[DurableAgentTask]: ...
```

### Example 1 — Single-agent orchestration

```python
from durabletask import OrchestrationContext
from agent_framework_durabletask import DurableAIAgentOrchestrationContext

def qa_orchestration(ctx: OrchestrationContext):
    agent_ctx = DurableAIAgentOrchestrationContext(ctx)
    agent = agent_ctx.get_agent("qa-agent")
    answer = yield agent.run("What is the capital of France?")
    return answer.text
```

### Example 2 — Sequential multi-agent pipeline

```python
from durabletask import OrchestrationContext
from agent_framework_durabletask import DurableAIAgentOrchestrationContext

def pipeline_orchestration(ctx: OrchestrationContext):
    ac = DurableAIAgentOrchestrationContext(ctx)

    fetcher = ac.get_agent("fetcher")
    summariser = ac.get_agent("summariser")

    raw = yield fetcher.run("Retrieve latest earnings data for ACME Corp.")
    summary = yield summariser.run(f"Summarise: {raw.text}")

    return summary.text
```

### Example 3 — Fan-out to multiple agents in parallel

```python
from durabletask import OrchestrationContext, when_all
from agent_framework_durabletask import DurableAIAgentOrchestrationContext

def parallel_analysis(ctx: OrchestrationContext):
    ac = DurableAIAgentOrchestrationContext(ctx)

    analysts = [ac.get_agent(f"analyst-{i}") for i in range(3)]
    tasks = [a.run("Analyse Q1 revenue data from your perspective.") for a in analysts]

    results = yield when_all(tasks)
    combined = "\n".join(r.text for r in results)
    return combined
```

---

## 4 · `AgentEntity` + `AgentEntityStateProviderMixin`

**Sub-package:** `agent_framework_durabletask._entities`  
**Install:** `pip install agent-framework-durabletask`

`AgentEntity` is the platform-agnostic execution kernel: it holds the agent reference, drives the run loop (streaming first, non-streaming fallback), appends entries to conversation history, persists state, and notifies callbacks. `AgentEntityStateProviderMixin` is the abstract state-caching layer that concrete entity classes must implement.

### Key source facts

- `AgentEntity.run()` always tries `agent.run(stream=True, ...)` first; it falls back to non-streaming only if the result is not an `AsyncIterator` or if streaming raises an exception — the `TypeError` check inspects the error message for `__aiter__` or `stream`.
- Error responses are persisted with `DurableAgentStateResponse.is_error = True`, and the next `run()` skips them when reconstructing `chat_messages` via `_is_error_response()`.
- `_to_replayable_message()` strips `type == "reasoning"` content items before replaying history into chat clients (prevents thinking tokens being sent back as user context).
- `AgentEntityStateProviderMixin._state_cache` is a class-level `None` sentinel; the first `state` property access deserialises from `_get_state_dict()` and caches it for the entity's lifetime.
- `AgentEntityStateProviderMixin.reset()` creates a fresh `DurableAgentState()` and immediately calls `persist_state()`.

### Signatures

```python
class AgentEntity:
    def __init__(
        self,
        agent: SupportsAgentRun,
        callback: AgentResponseCallbackProtocol | None = None,
        *,
        state_provider: AgentEntityStateProviderMixin,
    ) -> None: ...

    async def run(
        self, request: RunRequest | dict[str, Any] | str
    ) -> AgentResponse: ...

    def reset(self) -> None: ...

class AgentEntityStateProviderMixin:
    def _get_state_dict(self) -> dict[str, Any]: ...       # must implement
    def _set_state_dict(self, state: dict[str, Any]) -> None: ...  # must implement
    def _get_thread_id_from_entity(self) -> str: ...       # must implement
    def persist_state(self) -> None: ...
    def reset(self) -> None: ...
    thread_id: str          # property → _get_thread_id_from_entity()
    state: DurableAgentState  # property with lazy load + setter
```

### Example 1 — Minimal in-memory state provider for testing

```python
import asyncio
from agent_framework_durabletask._entities import AgentEntity, AgentEntityStateProviderMixin
from agent_framework_durabletask._durable_agent_state import DurableAgentState
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

class InMemoryStateProvider(AgentEntityStateProviderMixin):
    def __init__(self, thread_id: str) -> None:
        self._thread_id = thread_id
        self._state_dict: dict = {}

    def _get_state_dict(self) -> dict:
        return self._state_dict

    def _set_state_dict(self, state: dict) -> None:
        self._state_dict = state

    def _get_thread_id_from_entity(self) -> str:
        return self._thread_id

provider = InMemoryStateProvider(thread_id="session-abc")
client = OpenAIChatCompletionClient(model="gpt-4o")
entity = AgentEntity(
    agent=Agent(client=client, name="test-agent"),
    state_provider=provider,
)

async def main():
    response = await entity.run("Hello!")
    print(response.text)

asyncio.run(main())
```

### Example 2 — Inspecting skipped error entries in replay

```python
from agent_framework_durabletask._durable_agent_state import (
    DurableAgentState, DurableAgentStateResponse, DurableAgentStateMessage,
    DurableAgentStateTextContent,
)
from datetime import datetime, timezone

# Simulate a state with one error response already in history
state = DurableAgentState()
err_response = DurableAgentStateResponse(
    correlation_id="corr-1",
    created_at=datetime.now(tz=timezone.utc),
    messages=[DurableAgentStateMessage(
        role="assistant",
        contents=[DurableAgentStateTextContent(text="Tool failed.")],
    )],
    is_error=True,  # will be skipped during replay
)
state.data.conversation_history.append(err_response)

# AgentEntity._is_error_response filters these out
entry = state.data.conversation_history[0]
print(isinstance(entry, DurableAgentStateResponse) and entry.is_error)  # True
```

### Example 3 — Resetting conversation history

```python
import asyncio
from agent_framework_durabletask._entities import AgentEntity
# Assumes InMemoryStateProvider from Example 1 is available

async def demo():
    provider = InMemoryStateProvider(thread_id="session-xyz")
    from agent_framework import Agent
    from agent_framework.openai import OpenAIChatCompletionClient

    entity = AgentEntity(
        agent=Agent(client=OpenAIChatCompletionClient(model="gpt-4o"), name="demo"),
        state_provider=provider,
    )

    await entity.run("Remember: the secret code is 42.")
    print(entity.state.message_count)  # 2 (request + response)

    entity.reset()
    print(entity.state.message_count)  # 0 — history cleared and persisted
    print(provider._state_dict["schemaVersion"])  # "1.1.0"

asyncio.run(demo())
```

---

## 5 · `AgentCallbackContext` + `AgentResponseCallbackProtocol`

**Sub-package:** `agent_framework_durabletask._callbacks`  
**Install:** `pip install agent-framework-durabletask`

`AgentCallbackContext` is a frozen dataclass carrying per-invocation metadata (agent name, correlation ID, thread ID, original request). `AgentResponseCallbackProtocol` defines the two async hook points — streaming updates and the final response.

### Key source facts

- `AgentCallbackContext` is `frozen=True` — implementations cannot modify it during a callback.
- `AgentEntity._notify_stream_update()` and `_notify_final_response()` both catch all exceptions from callbacks and log them at `WARNING` level — a misbehaving callback can never crash the entity run.
- Both callback methods are probed with `inspect.isawaitable()` so a synchronous implementation also works at runtime (the Protocol declares them `async`, but the entity handles sync callbacks silently).

### Signatures

```python
@dataclass(frozen=True)
class AgentCallbackContext:
    agent_name: str
    correlation_id: str
    thread_id: str | None = None
    request_message: str | None = None

class AgentResponseCallbackProtocol(Protocol):
    async def on_streaming_response_update(
        self,
        update: AgentResponseUpdate,
        context: AgentCallbackContext,
    ) -> None: ...

    async def on_agent_response(
        self,
        response: AgentResponse,
        context: AgentCallbackContext,
    ) -> None: ...
```

### Example 1 — Logging callback with full context

```python
import logging
from agent_framework_durabletask import AgentCallbackContext, AgentResponseCallbackProtocol
from agent_framework_core._types import AgentResponse, AgentResponseUpdate

logger = logging.getLogger(__name__)

class LoggingCallback(AgentResponseCallbackProtocol):
    async def on_streaming_response_update(
        self, update: AgentResponseUpdate, ctx: AgentCallbackContext
    ) -> None:
        logger.debug(
            "agent=%s corr=%s thread=%s chunk_len=%d",
            ctx.agent_name, ctx.correlation_id, ctx.thread_id,
            len(update.text or ""),
        )

    async def on_agent_response(
        self, response: AgentResponse, ctx: AgentCallbackContext
    ) -> None:
        logger.info(
            "agent=%s corr=%s completed, tokens=%s",
            ctx.agent_name, ctx.correlation_id,
            response.usage,
        )
```

### Example 2 — Publishing final responses to a message queue

```python
import json
import asyncio
from agent_framework_durabletask import AgentCallbackContext, AgentResponseCallbackProtocol
from agent_framework_core._types import AgentResponse, AgentResponseUpdate

class QueuePublishCallback(AgentResponseCallbackProtocol):
    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    async def on_streaming_response_update(
        self, update: AgentResponseUpdate, ctx: AgentCallbackContext
    ) -> None:
        pass  # streaming chunks not needed in the queue

    async def on_agent_response(
        self, response: AgentResponse, ctx: AgentCallbackContext
    ) -> None:
        payload = {
            "correlation_id": ctx.correlation_id,
            "agent_name": ctx.agent_name,
            "text": response.text,
        }
        await self._queue.put(json.dumps(payload))

async def main() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    callback = QueuePublishCallback(queue)
    # pass callback to DurableAIAgentWorker(worker, callback=callback)

asyncio.run(main())
```

### Example 3 — Inspecting a frozen context object

```python
from agent_framework_durabletask._callbacks import AgentCallbackContext

ctx = AgentCallbackContext(
    agent_name="assistant",
    correlation_id="abc-123",
    thread_id="thread-xyz",
    request_message="Hello!",
)
print(ctx.agent_name)       # "assistant"
print(ctx.correlation_id)   # "abc-123"

try:
    ctx.agent_name = "other"  # raises FrozenInstanceError — dataclass is frozen
except Exception as e:
    print(type(e).__name__)   # FrozenInstanceError
```

---

## 6 · `RunRequest`

**Sub-package:** `agent_framework_durabletask._models`  
**Install:** `pip install agent-framework-durabletask`

The canonical request message passed from a client or orchestrator into a durable entity. Supports JSON round-trip via `to_dict()` / `from_dict()` / `from_json()`, fire-and-forget mode, structured response formats, and per-request role coercion.

### Key source facts

- `correlation_id` is **required** in `from_dict()` — if the key `correlationId` is absent or falsy, a `ValueError` is raised immediately. This is the ID used to match the response in `DurableAgentState.try_get_agent_response()`.
- `coerce_role()` strips and lowercases the role string; `None`, empty string, or whitespace-only values are all normalised to `"user"`.
- `wait_for_response=False` enables fire-and-forget mode: the entity is signalled but the client returns immediately without polling for a response.
- `enable_tool_calls=False` sets `options.setdefault("tools", None)` inside `AgentEntity.run()` — it doesn't modify the `RunRequest` itself but causes downstream tool suppression.
- Serialisation maps `correlation_id` → `correlationId` and `orchestration_id` → `orchestrationId`; the camelCase keys match the HTTP API contract.

### Signature

```python
@dataclass
class RunRequest:
    message: str
    correlation_id: str
    request_response_format: str = "text"  # REQUEST_RESPONSE_FORMAT_TEXT
    role: str = "user"
    response_format: type[BaseModel] | None = None
    enable_tool_calls: bool = True
    wait_for_response: bool = True
    created_at: datetime | None = None
    orchestration_id: str | None = None
    options: dict[str, Any] = field(default_factory=lambda: {})

    def __init__(
        self,
        message: str,
        correlation_id: str,
        request_response_format: str = "text",
        role: str | None = "user",
        response_format: type[BaseModel] | None = None,
        enable_tool_calls: bool = True,
        wait_for_response: bool = True,
        created_at: datetime | None = None,
        orchestration_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_json(cls, data: str) -> RunRequest: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRequest: ...

    @staticmethod
    def coerce_role(value: str | None) -> str: ...
```

### Example 1 — Basic round-trip serialisation

```python
import uuid
from agent_framework_durabletask._models import RunRequest

req = RunRequest(
    message="What is the weather in London?",
    correlation_id=uuid.uuid4().hex,
)
d = req.to_dict()
print(d["message"])       # "What is the weather in London?"
print("correlationId" in d)  # True  (camelCase in wire format)

restored = RunRequest.from_dict(d)
print(restored.correlation_id == req.correlation_id)  # True
```

### Example 2 — Fire-and-forget with structured response format

```python
import uuid
from pydantic import BaseModel
from agent_framework_durabletask._models import RunRequest

class Report(BaseModel):
    title: str
    body: str

req = RunRequest(
    message="Generate a quarterly report.",
    correlation_id=uuid.uuid4().hex,
    response_format=Report,
    wait_for_response=False,   # fire-and-forget
    enable_tool_calls=False,
)
d = req.to_dict()
# response_format is serialised as JSON Schema in "response_format" key
print(req.wait_for_response)  # False
```

### Example 3 — Role coercion edge cases

```python
from agent_framework_durabletask._models import RunRequest

print(RunRequest.coerce_role("User"))    # "user"
print(RunRequest.coerce_role("SYSTEM"))  # "system"
print(RunRequest.coerce_role(None))      # "user"
print(RunRequest.coerce_role("  "))      # "user"
print(RunRequest.coerce_role("assistant"))  # "assistant"
```

---

## 7 · `AgentSessionId` + `DurableAgentSession`

**Sub-package:** `agent_framework_durabletask._models`  
**Install:** `pip install agent-framework-durabletask`

`AgentSessionId` encodes the durable entity address as `@name@key` and maps to the `dafx-{name}` entity prefix. `DurableAgentSession` extends `AgentSession` with an optional `durable_session_id` that is serialised and deserialised transparently alongside base session state.

### Key source facts

- `AgentSessionId.__str__()` produces `@{name}@{key}` — this exact format is parsed back by `AgentSessionId.parse()`.
- `parse()` accepts either `@name@key` (full format) or a plain key string when `agent_name` is provided separately; it raises `ValueError` for plain strings without an `agent_name` argument.
- `ENTITY_NAME_PREFIX = "dafx-"` is a class constant; `entity_name` property concatenates it with `name` to produce the key used to address the durabletask entity.
- `DurableAgentSession.from_dict()` pops `durable_session_id` from a defensive copy of the dict before calling `super().from_dict()`, so it never leaks into the base `AgentSession` state dict.

### Signatures

```python
@dataclass
class AgentSessionId:
    name: str
    key: str
    ENTITY_NAME_PREFIX: str = "dafx-"

    @staticmethod
    def to_entity_name(name: str) -> str: ...          # "dafx-{name}"
    @staticmethod
    def with_random_key(name: str) -> AgentSessionId: ...
    entity_name: str                                   # property
    def __str__(self) -> str: ...                      # "@{name}@{key}"

    @staticmethod
    def parse(
        session_id_string: str,
        agent_name: str | None = None,
    ) -> AgentSessionId: ...

class DurableAgentSession(AgentSession):
    durable_session_id: AgentSessionId | None

    def __init__(
        self,
        *,
        durable_session_id: AgentSessionId | None = None,
        session_id: str | None = None,
        service_session_id: str | None = None,
    ) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_session_id(cls, durable_session_id: AgentSessionId, ...) -> DurableAgentSession: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentSession: ...
```

### Example 1 — Creating and parsing session IDs

```python
from agent_framework_durabletask._models import AgentSessionId

sid = AgentSessionId.with_random_key("assistant")
print(str(sid))           # "@assistant@<hex>"
print(sid.entity_name)    # "dafx-assistant"

# Round-trip parse
restored = AgentSessionId.parse(str(sid))
print(restored.name == sid.name)  # True
print(restored.key == sid.key)    # True

# Parse with explicit agent_name override
sid2 = AgentSessionId.parse("plain-key", agent_name="researcher")
print(sid2.name)  # "researcher"
print(sid2.key)   # "plain-key"
```

### Example 2 — Serialising a DurableAgentSession

```python
from agent_framework_durabletask._models import AgentSessionId, DurableAgentSession

sid = AgentSessionId(name="assistant", key="abc123")
session = DurableAgentSession.from_session_id(
    sid, session_id="local-session-1"
)

d = session.to_dict()
print(d["durable_session_id"])  # "@assistant@abc123"

restored = DurableAgentSession.from_dict(d)
print(restored.durable_session_id.entity_name)  # "dafx-assistant"
```

### Example 3 — Passing a session to maintain conversation context

```python
from durabletask import TaskHubGrpcClient
from agent_framework_durabletask import DurableAIAgentClient
from agent_framework_durabletask._models import AgentSessionId, DurableAgentSession

grpc_client = TaskHubGrpcClient(host_address="localhost:4001")
agent_client = DurableAIAgentClient(grpc_client)
agent = agent_client.get_agent("assistant")

# Build a stable session that maps to a fixed entity key
sid = AgentSessionId(name="assistant", key="user-alice-thread-1")
session = DurableAgentSession.from_session_id(sid)

r1 = agent.run("My favourite colour is blue.", session=session)
r2 = agent.run("What is my favourite colour?", session=session)
print(r2.text)  # "Your favourite colour is blue."
```

---

## 8 · `DurableAgentState` + `DurableAgentStateData`

**Sub-package:** `agent_framework_durabletask._durable_agent_state`  
**Install:** `pip install agent-framework-durabletask`

`DurableAgentState` is the root JSON document persisted inside the Azure Durable Entity, versioned at schema `1.1.0`. `DurableAgentStateData` holds the `conversation_history` list and optional `extension_data`.

### Key source facts

- `SCHEMA_VERSION = "1.1.0"` is a class constant on `DurableAgentState`; `from_dict()` logs a `WARNING` and returns a fresh empty state if `schemaVersion` is absent — all conversation history is silently discarded on incompatible schema upgrades.
- `try_get_agent_response()` searches `conversation_history` for the **first** `DurableAgentStateResponse` with a matching `correlation_id` and returns an `AgentResponse`; it returns `None` if not found.
- `message_count` counts all entries (requests + responses together), not just messages — useful for quick history-size checks.
- `DurableAgentStateData.from_dict()` calls `_parse_history_entries()` which dispatches on the `$type` field: `"request"` → `DurableAgentStateRequest.from_dict()`, `"response"` → `DurableAgentStateResponse.from_dict()`.

### Signatures

```python
class DurableAgentState:
    SCHEMA_VERSION: str = "1.1.0"

    data: DurableAgentStateData
    schema_version: str

    def __init__(self, schema_version: str = SCHEMA_VERSION) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_json(self) -> str: ...
    message_count: int   # property

    def try_get_agent_response(self, correlation_id: str) -> AgentResponse | None: ...

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> DurableAgentState: ...
    @classmethod
    def from_json(cls, json_str: str) -> DurableAgentState: ...

class DurableAgentStateData:
    conversation_history: list[DurableAgentStateEntry]
    extension_data: dict[str, Any] | None

    def __init__(
        self,
        conversation_history: list[DurableAgentStateEntry] | None = None,
        extension_data: dict[str, Any] | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> DurableAgentStateData: ...
```

### Example 1 — JSON round-trip with version guard

```python
import json
from agent_framework_durabletask._durable_agent_state import DurableAgentState

state = DurableAgentState()
print(state.schema_version)  # "1.1.0"

j = state.to_json()
obj = json.loads(j)
print(obj["schemaVersion"])          # "1.1.0"
print(obj["data"]["conversationHistory"])  # []

restored = DurableAgentState.from_json(j)
print(restored.message_count)  # 0

# Missing schemaVersion → returns fresh state (no crash)
stale = DurableAgentState.from_dict({})
print(stale.message_count)  # 0
```

### Example 2 — Looking up a response by correlation ID

```python
from datetime import datetime, timezone
from agent_framework_durabletask._durable_agent_state import (
    DurableAgentState, DurableAgentStateResponse, DurableAgentStateMessage,
    DurableAgentStateTextContent,
)

state = DurableAgentState()
resp = DurableAgentStateResponse(
    correlation_id="corr-42",
    created_at=datetime.now(tz=timezone.utc),
    messages=[
        DurableAgentStateMessage(
            role="assistant",
            contents=[DurableAgentStateTextContent(text="The answer is 42.")],
        )
    ],
)
state.data.conversation_history.append(resp)

agent_response = state.try_get_agent_response("corr-42")
print(agent_response is not None)          # True
print(agent_response.text)                 # "The answer is 42."
print(state.try_get_agent_response("corr-99"))  # None
```

### Example 3 — Using extension_data for custom metadata

```python
from agent_framework_durabletask._durable_agent_state import (
    DurableAgentState, DurableAgentStateData
)
import json

state = DurableAgentState()
state.data.extension_data = {"tenant": "acme", "deployment": "prod"}

d = state.to_dict()
print(d["data"]["extensionData"])  # {"tenant": "acme", "deployment": "prod"}

restored = DurableAgentState.from_dict(d)
print(restored.data.extension_data["tenant"])  # "acme"
```

---

## 9 · `DurableAgentStateEntry` + `DurableAgentStateRequest` + `DurableAgentStateResponse` + `DurableAgentStateUsage`

**Sub-package:** `agent_framework_durabletask._durable_agent_state`  
**Install:** `pip install agent-framework-durabletask`

The three-level entry hierarchy that fills `conversation_history`. `DurableAgentStateEntry` is the base; `Request` and `Response` are discriminated by `$type`. `DurableAgentStateUsage` tracks token counts inside response entries.

### Key source facts

- `DurableAgentStateEntryJsonType` enum has two members: `REQUEST = "request"` and `RESPONSE = "response"` — the value written to the `$type` field in JSON.
- `DurableAgentStateResponse.is_error` is **not** serialised to the state dict — it is an in-memory flag set by `AgentEntity.run()` when execution fails, and is reconstructed to `False` on `from_dict()`.
- `DurableAgentStateRequest.from_run_request()` is the factory bridge between `RunRequest` and the persisted representation; it copies `correlation_id`, `created_at`, `response_type`, `response_schema`, and `orchestration_id`.
- `DurableAgentStateUsage.from_usage()` accepts `UsageDetails | MutableMapping | None`; non-standard keys (anything not in `{input_token_count, output_token_count, total_token_count}`) are collected into `extensionData` to preserve provider-specific usage fields.
- `to_usage_details()` reconstructs the SDK `UsageDetails` object by calling `.update(extensionData)` — extension fields round-trip correctly.

### Signatures

```python
class DurableAgentStateEntry:
    json_type: DurableAgentStateEntryJsonType
    correlation_id: str | None
    created_at: datetime
    messages: list[DurableAgentStateMessage]
    extension_data: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentStateEntry: ...

class DurableAgentStateRequest(DurableAgentStateEntry):
    response_type: str | None
    response_schema: dict[str, Any] | None
    orchestration_id: str | None

    @staticmethod
    def from_run_request(request: RunRequest) -> DurableAgentStateRequest: ...

class DurableAgentStateResponse(DurableAgentStateEntry):
    usage: DurableAgentStateUsage | None
    is_error: bool  # not persisted

class DurableAgentStateUsage:
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    extensionData: dict[str, Any] | None  # camelCase as written in source; mirrors the JSON wire key

    def to_dict(self) -> dict[str, Any]: ...
    def to_usage_details(self) -> UsageDetails: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DurableAgentStateUsage: ...
    @staticmethod
    def from_usage(usage: UsageDetails | MutableMapping | None) -> DurableAgentStateUsage | None: ...
```

### Example 1 — Building a request entry from a RunRequest

```python
import uuid
from agent_framework_durabletask._models import RunRequest
from agent_framework_durabletask._durable_agent_state import DurableAgentStateRequest

run_req = RunRequest(
    message="What is the square root of 144?",
    correlation_id=uuid.uuid4().hex,
    request_response_format="text",
)
entry = DurableAgentStateRequest.from_run_request(run_req)
print(entry.json_type.value)          # "request"
print(entry.correlation_id)           # matches run_req.correlation_id
print(entry.messages[0].text)         # "What is the square root of 144?"
```

### Example 2 — Inspecting token usage in a response entry

```python
from datetime import datetime, timezone
from agent_framework_durabletask._durable_agent_state import (
    DurableAgentStateResponse, DurableAgentStateMessage,
    DurableAgentStateTextContent, DurableAgentStateUsage,
)

usage = DurableAgentStateUsage(
    input_token_count=120,
    output_token_count=45,
    total_token_count=165,
)
resp = DurableAgentStateResponse(
    correlation_id="corr-9",
    created_at=datetime.now(tz=timezone.utc),
    messages=[DurableAgentStateMessage(
        role="assistant",
        contents=[DurableAgentStateTextContent(text="The answer is 12.")],
    )],
    usage=usage,
)
d = resp.to_dict()
print(d["usage"]["inputTokenCount"])   # 120
print(d["usage"]["outputTokenCount"])  # 45
print(resp.is_error)                   # False
```

### Example 3 — Round-tripping usage with extension fields

```python
from agent_framework_durabletask._durable_agent_state import DurableAgentStateUsage

raw_usage = {
    "input_token_count": 200,
    "output_token_count": 80,
    "total_token_count": 280,
    "cache_read_token_count": 50,  # provider-specific extension field
}
usage = DurableAgentStateUsage.from_usage(raw_usage)
print(usage.total_token_count)              # 280
print(usage.extensionData)                  # {"cache_read_token_count": 50}

sdk_usage = usage.to_usage_details()
print(sdk_usage.get("cache_read_token_count"))  # 50 — round-tripped via extensionData
```

---

## 10 · `DurableAgentStateContent` + content subclasses + `DurableStateFields` + `ContentTypes`

**Sub-package:** `agent_framework_durabletask._durable_agent_state` · `._constants`  
**Install:** `pip install agent-framework-durabletask`

The full content type system: one abstract base, nine concrete subclasses, and the two constant classes that define every camelCase field name and `$type` discriminator string used in the JSON schema.

### Key source facts

- `DurableAgentStateContent.from_ai_content()` is a `match content.type` factory; unknown content types are wrapped in `DurableAgentStateUnknownContent` rather than raising — this makes the state forward-compatible with new content kinds.
- `DurableAgentStateMessage.to_dict()` translates the internal `type` field to the `$type` discriminator: it reads `c.to_dict().get("type", ContentTypes.TEXT)` and emits it as `DurableStateFields.TYPE_DISCRIMINATOR = "$type"`.
- `DurableAgentStateFunctionCallContent.from_function_call_content()` parses `arguments` from a JSON string if it is not already a dict — handles both OpenAI-style (JSON string) and already-parsed dict arguments.
- `DurableAgentStateTextReasoningContent` stores `type = "reasoning"` and its content is **stripped** by `AgentEntity._to_replayable_message()` before replaying history — thinking tokens never leak back into the model context.
- `DurableStateFields.TYPE_DISCRIMINATOR = "$type"` and `DurableStateFields.TYPE_INTERNAL = "type"` are kept separate: `TYPE_INTERNAL` is the Python-side key inside each `to_dict()` result; `TYPE_DISCRIMINATOR` is the JSON-wire key written by the message serialiser.
- `ContentTypes` defines 11 string constants: `text`, `data`, `error`, `functionCall`, `functionResult`, `hostedFile`, `hostedVectorStore`, `reasoning`, `uri`, `usage`, `unknown`.

### Content type summary

| Subclass | `type` constant | Maps from SDK `Content` type |
|---|---|---|
| `DurableAgentStateTextContent` | `"text"` | `content.type == "text"` |
| `DurableAgentStateDataContent` | `"data"` | `"data"` |
| `DurableAgentStateErrorContent` | `"error"` | `"error"` |
| `DurableAgentStateFunctionCallContent` | `"functionCall"` | `"function_call"` |
| `DurableAgentStateFunctionResultContent` | `"functionResult"` | `"function_result"` |
| `DurableAgentStateHostedFileContent` | `"hostedFile"` | `"hosted_file"` |
| `DurableAgentStateHostedVectorStoreContent` | `"hostedVectorStore"` | `"hosted_vector_store"` |
| `DurableAgentStateTextReasoningContent` | `"reasoning"` | `"reasoning"` (stripped on replay) |
| `DurableAgentStateUriContent` | `"uri"` | `"uri"` |
| `DurableAgentStateUsageContent` | `"usage"` | `"usage"` |
| `DurableAgentStateUnknownContent` | `"unknown"` | anything else |

### Example 1 — Inspecting the $type field constants

```python
from agent_framework_durabletask._constants import ContentTypes, DurableStateFields

print(ContentTypes.TEXT)           # "text"
print(ContentTypes.FUNCTION_CALL)  # "functionCall"  (camelCase in wire format)
print(ContentTypes.REASONING)      # "reasoning"
print(ContentTypes.UNKNOWN)        # "unknown"

print(DurableStateFields.TYPE_DISCRIMINATOR)  # "$type"
print(DurableStateFields.TYPE_INTERNAL)       # "type"
print(DurableStateFields.CALL_ID)             # "callId"
print(DurableStateFields.ARGUMENTS)           # "arguments"
print(DurableStateFields.SCHEMA_VERSION)      # "schemaVersion"
```

### Example 2 — Using from_ai_content factory

```python
from agent_framework_core._types import Content
from agent_framework_durabletask._durable_agent_state import (
    DurableAgentStateContent,
    DurableAgentStateTextContent,
    DurableAgentStateFunctionCallContent,
)

# Text content
text_sdk = Content.from_text("Hello, world!")
durable_text = DurableAgentStateContent.from_ai_content(text_sdk)
assert isinstance(durable_text, DurableAgentStateTextContent)
print(durable_text.text)  # "Hello, world!"

# Function call content
call_sdk = Content.from_function_call(
    call_id="call-1",
    name="search",
    arguments='{"query": "Python docs"}',
)
durable_call = DurableAgentStateContent.from_ai_content(call_sdk)
assert isinstance(durable_call, DurableAgentStateFunctionCallContent)
print(durable_call.name)       # "search"
print(durable_call.arguments)  # {"query": "Python docs"} (parsed to dict)
```

### Example 3 — Round-tripping a message with mixed content types

```python
from agent_framework_durabletask._durable_agent_state import (
    DurableAgentStateMessage,
    DurableAgentStateTextContent,
    DurableAgentStateFunctionCallContent,
)

msg = DurableAgentStateMessage(
    role="assistant",
    contents=[
        DurableAgentStateTextContent(text="I'll look that up."),
        DurableAgentStateFunctionCallContent(
            call_id="call-99",
            name="web_search",
            arguments={"query": "latest news"},
        ),
    ],
    author_name="Assistant",
)

d = msg.to_dict()
# $type discriminators in wire format
print(d["contents"][0]["$type"])  # "text"
print(d["contents"][1]["$type"])  # "functionCall"
print(d["contents"][1]["callId"])  # "call-99"

# Convert back to agent framework Message
chat_msg = msg.to_chat_message()
print(chat_msg.role)         # "assistant"
print(chat_msg.author_name)  # "Assistant"
print(len(chat_msg.contents))  # 2
```

---

## Summary

| # | Class group | Package | Key facts |
|---|---|---|---|
| 1 | `DurableAIAgentWorker` + `DurableAIAgentClient` | `_worker` / `_client` | Worker wraps `TaskHubGrpcWorker`; entity named `dafx-{name}`; client clamps poll retries to ≥ 1 |
| 2 | `DurableAIAgent` + `DurableAgentExecutor` | `_shim` / `_executors` | Shim returns `TaskT` not coroutine; `stream=True` raises `ValueError`; options dict keys extracted in `get_run_request()` |
| 3 | `DurableAIAgentOrchestrationContext` | `_orchestration_context` | Thin wrapper for `OrchestrationContext`; `get_agent()` returns yield-compatible `DurableAIAgent[DurableAgentTask]` |
| 4 | `AgentEntity` + `AgentEntityStateProviderMixin` | `_entities` | Streaming-first with non-streaming fallback; error responses marked `is_error` and skipped on replay; reasoning stripped from replayable messages |
| 5 | `AgentCallbackContext` + `AgentResponseCallbackProtocol` | `_callbacks` | Frozen context dataclass; callbacks catch all exceptions; sync callbacks work at runtime despite Protocol declaring `async` |
| 6 | `RunRequest` | `_models` | `correlationId` required in `from_dict()`; role coerced to lowercase; fire-and-forget via `wait_for_response=False` |
| 7 | `AgentSessionId` + `DurableAgentSession` | `_models` | `@name@key` wire format; `entity_name` = `"dafx-{name}"`; `from_dict()` defensive copy prevents key leak |
| 8 | `DurableAgentState` + `DurableAgentStateData` | `_durable_agent_state` | Schema v1.1.0; missing `schemaVersion` resets state silently; `try_get_agent_response()` searches by `correlation_id` |
| 9 | Entry hierarchy + `DurableAgentStateUsage` | `_durable_agent_state` | `$type` discriminator; `is_error` not serialised; non-standard usage keys collected into `extensionData` |
| 10 | Content hierarchy + `DurableStateFields` + `ContentTypes` | `_durable_agent_state` / `_constants` | 11 content types; unknown types → `DurableAgentStateUnknownContent`; `reasoning` stripped on replay; `$type` ≠ `type` (wire vs Python) |

**Install:** `pip install agent-framework-durabletask`  
**Also see:** [Vol. 9 — `DurableAIAgent` first mention](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) · [Vol. 15 — Durable external layer overview](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v15/) · [Vol. 24 — `_harness._loop` and durable integration context](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v24/)

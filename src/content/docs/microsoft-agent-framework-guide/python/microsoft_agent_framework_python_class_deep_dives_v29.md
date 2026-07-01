---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 29"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.10.0: SecureMCPToolProxy+apply_mcp_security_labels (FIDES experimental MCP security proxy — wrap-or-URL init, auto-label on connect, annotation_overrides, mark_write_tools_as_sinks, refresh_labels(), static-header AsyncClient injection); BedrockGuardrailConfig+BedrockChatOptions.guardrailConfig (Amazon Bedrock content safety — guardrailIdentifier+guardrailVersion+trace+streamProcessingMode TypedDict, per-request or per-session wiring); GroupChatRequestMessage+GroupChatRequestSentEvent+GroupChatResponseReceivedEvent (observable group-chat events — additional_instruction envelope, round_index+participant_name for observability hooks); MagenticOrchestratorEvent+MagenticOrchestratorEventType+MagenticProgressLedgerItem (Magentic event taxonomy — PLAN_CREATED/REPLANNED/PROGRESS_LEDGER_UPDATED enum, Message|MagenticProgressLedger content union, reason+answer str|bool ledger item); ReleaseCandidateFeature+ExperimentalWarning+FeatureStageWarning (feature staging tiers — RC enum currently empty, ExperimentalWarning→FeatureStageWarning→FutureWarning hierarchy, @experimental/@release_candidate decorators); ToolExecutionException+_MCPTaskAbandoned+_MCPDeadlineExpired (MCP error taxonomy — ToolException→ToolExecutionException→_MCPTaskAbandoned subclass chain, _MCPDeadlineExpired internal sentinel, max_task_wait cancellation contract); AgentMiddlewareLayer+ChatMiddlewareLayer (MRO mixin layers — AgentMiddlewareLayer.run() categorize+pipeline+ResponseStream.from_awaitable streaming bridge, ChatMiddlewareLayer.get_response() middleware pop from client_kwargs); DiscoveryResponse+EntityInfo+AgentFrameworkRequest+OpenAIError (DevUI protocol types — 18-field EntityInfo Pydantic model, AgentFrameworkRequest model-as-entity_id + conversation param, OpenAIError.create() factory); GroupChatState+AgentOrchestrationOutput+create_completion_message+clean_conversation_for_handoff (orchestration utilities — frozen GroupChatState dataclass, strict AgentOrchestrationOutput required/forbid config, text-only handoff cleaning); RawGitHubCopilotAgent+GitHubCopilotSettings+GitHubCopilotOptions (GitHub Copilot CLI agent — AGENT_PROVIDER_NAME='github.copilot', cli_path/model/timeout/base_directory env-prefix resolution, on_pre_tool_use/on_permission_request hook pair, BYOK provider config)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 52
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 29

Verified against **agent-framework 1.10.0** / **agent-framework-ag-ui 1.0.0rc7** / **agent-framework-orchestrations 1.0.0** / **agent-framework-claude 1.0.0b260609** (installed July 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`. Sub-packages introspected:
`agent_framework.security`,
`agent_framework.amazon`,
`agent_framework.orchestrations`,
`agent_framework.devui`,
`agent_framework._feature_stage`,
`agent_framework._mcp`,
`agent_framework._middleware`,
`agent_framework.github`.

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
- [Vol. 28](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v28/) — `RawClaudeAgent`+`ClaudeAgent`+`ClaudeAgentOptions`+`ClaudeAgentSettings`, `AgentFrameworkAgent`+`AgentConfig`, `AGUIThreadSnapshot`+`AGUIThreadSnapshotStore`+`InMemoryAGUIThreadSnapshotStore`, `PredictiveStateHandler`+`PredictStateConfig`, `AGUIRequest`+`AGUIChatOptions`+`AgentState`+`RunMetadata`, `A2AAgentSession`+`A2AContinuationToken`, `ThreadItemConverter`, `AgentApprovalExecutor`+`AgentRequestInfoExecutor`+`AgentRequestInfoResponse`, `AgentFrameworkWorkflow`, `FlowState`+`AGUIHttpService`+`run_workflow_stream`

This volume covers **ten class groups** across the core security module, Amazon Bedrock, orchestration events, DevUI, feature staging, MCP error taxonomy, middleware MRO layers, and the GitHub Copilot integration. All examples verified against `agent-framework==1.10.0`.

| # | Class / group | Module |
|---|---|---|
| 1 | `SecureMCPToolProxy` · `apply_mcp_security_labels` | `agent_framework.security` |
| 2 | `BedrockGuardrailConfig` · `BedrockChatOptions.guardrailConfig` | `agent_framework.amazon` |
| 3 | `GroupChatRequestMessage` · `GroupChatRequestSentEvent` · `GroupChatResponseReceivedEvent` | `agent_framework.orchestrations` |
| 4 | `MagenticOrchestratorEvent` · `MagenticOrchestratorEventType` · `MagenticProgressLedgerItem` | `agent_framework.orchestrations` |
| 5 | `ReleaseCandidateFeature` · `ExperimentalWarning` · `FeatureStageWarning` | `agent_framework._feature_stage` |
| 6 | `ToolExecutionException` · `_MCPTaskAbandoned` · `_MCPDeadlineExpired` | `agent_framework._mcp` / `._tools` |
| 7 | `AgentMiddlewareLayer` · `ChatMiddlewareLayer` | `agent_framework._middleware` |
| 8 | `DiscoveryResponse` · `EntityInfo` · `AgentFrameworkRequest` · `OpenAIError` | `agent_framework.devui` |
| 9 | `GroupChatState` · `AgentOrchestrationOutput` · `create_completion_message` · `clean_conversation_for_handoff` | `agent_framework.orchestrations` |
| 10 | `RawGitHubCopilotAgent` · `GitHubCopilotSettings` · `GitHubCopilotOptions` | `agent_framework.github` |

---

## 1 · `SecureMCPToolProxy` + `apply_mcp_security_labels`

**Module:** `agent_framework.security`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework.security import SecureMCPToolProxy, apply_mcp_security_labels`
**Decorator:** `@experimental(feature_id=ExperimentalFeature.FIDES)`

`SecureMCPToolProxy` is the FIDES (Federated Information Designation and Enforcement System) convenience wrapper that auto-labels every tool advertised by an MCP server when the connection is established. It integrates with `LabelTrackingFunctionMiddleware` and `PolicyEnforcementFunctionMiddleware` without requiring any middleware changes — labels are stamped on each `FunctionTool.additional_properties` so the existing pipeline picks them up automatically.

### Initialization modes

The class takes **exactly one** of two mutually exclusive init paths:

```
mcp_tool (MCPTool) — wrap a pre-constructed MCPStdioTool / MCPWebsocketTool / MCPStreamableHTTPTool
url (str)          — auto-create an MCPStreamableHTTPTool internally; headers injected via AsyncClient
```

When `url` is supplied with `headers`, the proxy creates an `AsyncClient` that includes the static headers on **every** request — including `session.initialize()`. This avoids 401 errors that arise when header providers are populated only during `call_tool()`.

### Constructor

```python
SecureMCPToolProxy(
    mcp_tool: MCPTool | None = None,
    *,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    name: str | None = None,
    description: str | None = None,
    default_integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED,
    annotation_overrides: dict[str, tuple[IntegrityLabel, ConfidentialityLabel | None]] | None = None,
    mark_write_tools_as_sinks: bool = True,
)
```

| Parameter | Notes |
|-----------|-------|
| `default_integrity` | Label applied when the server provides no `ToolAnnotations`. Defaults to `UNTRUSTED` (conservative). |
| `annotation_overrides` | Per-remote-tool-name override dict: `{"write_file": (IntegrityLabel.TRUSTED, ConfidentialityLabel.PRIVATE)}`. Takes priority over server annotations. |
| `mark_write_tools_as_sinks` | When `True` (default), non-read-only tools get `max_allowed_confidentiality=PUBLIC`, blocking data exfiltration through tool arguments. |

### Key facts

- `proxy.tools` (alias for `proxy.functions`) returns the labeled `list[FunctionTool]` — pass it directly to `Agent(tools=proxy.tools)`.
- The async context manager calls `connect()` then `_apply_labels()` on `__aenter__`. Labels are re-applied each time you call `refresh_labels()` for servers that add tools during long-lived connections.
- Per-tool wrappers installed by `_apply_labels()` consume `_meta.ifc` payloads from MCP responses so that **server-supplied** labels win over the static label; the static label is the fallback when `_meta` is absent.
- `apply_mcp_security_labels()` is the underlying async function. Call it directly after connecting if you manage the `MCPTool` lifecycle yourself.

### Code examples

```python
# Example 1 — Wrap an existing MCPStdioTool (e.g. the GitHub MCP server)
import asyncio
from agent_framework import Agent
from agent_framework.security import SecureMCPToolProxy, SecureAgentConfig
from agent_framework._mcp import MCPStdioTool
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()

    async with SecureMCPToolProxy(
        MCPStdioTool(name="github", command="gh-mcp", args=["stdio"])
    ) as proxy:
        agent = Agent(
            client=client,
            tools=proxy.tools,
            context_providers=[SecureAgentConfig(chat_client=client)],
            instructions="You are a GitHub assistant.",
        )
        response = await agent.run("List my open pull requests.")
        print(response.text)

asyncio.run(main())
```

```python
# Example 2 — URL mode with bearer-token auth (headers on every request)
import asyncio
from agent_framework import Agent
from agent_framework.security import SecureMCPToolProxy
from agent_framework.security import IntegrityLabel, ConfidentialityLabel
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()

    async with SecureMCPToolProxy(
        url="https://mcp.example.com/",
        headers={"Authorization": "Bearer my-token"},
        name="company-mcp",
        # Trust read-only tools; keep write tools conservative
        annotation_overrides={
            "send_email": (IntegrityLabel.TRUSTED, ConfidentialityLabel.PRIVATE),
        },
        mark_write_tools_as_sinks=True,
    ) as proxy:
        agent = Agent(client=client, tools=proxy.tools)
        response = await agent.run("Draft and send a status update.")
        print(response.text)

asyncio.run(main())
```

```python
# Example 3 — apply_mcp_security_labels() called manually on an already-connected tool
import asyncio
from agent_framework._mcp import MCPStreamableHTTPTool
from agent_framework.security import apply_mcp_security_labels, IntegrityLabel
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()
    mcp = MCPStreamableHTTPTool(name="docs", url="https://docs-mcp.example.com/")

    async with mcp:
        # Stamp labels without using the proxy wrapper
        await apply_mcp_security_labels(
            mcp,
            default_integrity=IntegrityLabel.TRUSTED,
            mark_write_tools_as_sinks=False,
        )
        agent = Agent(client=client, tools=mcp.functions)
        response = await agent.run("Search for authentication docs.")
        print(response.text)

asyncio.run(main())
```

---

## 2 · `BedrockGuardrailConfig` + `BedrockChatOptions.guardrailConfig`

**Module:** `agent_framework.amazon`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework.amazon import BedrockGuardrailConfig, BedrockChatOptions, BedrockChatClient`

`BedrockGuardrailConfig` is a `TypedDict` (`total=False`) that maps to Amazon Bedrock's `guardrailConfig` parameter in the Converse API. It lets you apply Bedrock Guardrails — content filtering, PII redaction, grounding checks — to every model invocation without changing agent code.

### `BedrockGuardrailConfig` fields

| Field | Type | Notes |
|-------|------|-------|
| `guardrailIdentifier` | `str` | ARN or short ID of the guardrail resource |
| `guardrailVersion` | `str` | Version string e.g. `"1"` or `"DRAFT"` |
| `trace` | `"enabled"` \| `"disabled"` | Include guardrail trace in the API response |
| `streamProcessingMode` | `"sync"` \| `"async"` | `"sync"` blocks until guardrail evaluation completes; `"async"` does not |

### Wiring into `BedrockChatOptions`

Pass `guardrailConfig` as a key in the `options` dict at the **client** level (via `default_options`) or per `agent.run()` call:

```python
options: BedrockChatOptions = {
    "guardrailConfig": {
        "guardrailIdentifier": "arn:aws:bedrock:us-east-1::guardrail/abc123",
        "guardrailVersion": "1",
        "trace": "enabled",
        "streamProcessingMode": "sync",
    }
}
```

### Key facts

- `BedrockChatOptions` is a `total=False` TypedDict extending `ChatOptions`. All keys are optional. `guardrailConfig` is a Bedrock-specific key — it does **not** exist on `ChatOptions` or other provider option dicts.
- `streamProcessingMode="sync"` makes the Converse call block until the guardrail evaluates the output, which is required when you need the trace in the response body. Use `"async"` when guardrail latency would affect perceived streaming performance.
- `trace="enabled"` returns a `guardrailTrace` field in the Bedrock API response, which agent-framework surfaces in `ChatResponse.additional_properties["guardrailTrace"]`.
- Guardrails apply at the model level: the policy — topic denial, content filters, PII anonymization, grounding checks — is configured in the AWS console or via `bedrock:CreateGuardrail`, not in agent code.

### Code examples

```python
# Example 1 — Apply a guardrail to every call via default_options
import asyncio
from agent_framework import Agent
from agent_framework.amazon import BedrockChatClient, BedrockGuardrailConfig

async def main():
    client = BedrockChatClient(
        default_options={
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "guardrailConfig": BedrockGuardrailConfig(
                guardrailIdentifier="arn:aws:bedrock:us-east-1::guardrail/abc123",
                guardrailVersion="1",
                trace="enabled",
                streamProcessingMode="sync",
            ),
        }
    )
    agent = Agent(client=client, instructions="You are a customer support agent.")
    response = await agent.run("How do I reset my password?")
    # Check if guardrail intervened
    trace = getattr(response, "additional_properties", {}).get("guardrailTrace")
    print(response.text)
    if trace:
        print("Guardrail trace:", trace)

asyncio.run(main())
```

```python
# Example 2 — Override guardrail settings per run call
import asyncio
from agent_framework import Agent
from agent_framework.amazon import BedrockChatClient

async def main():
    client = BedrockChatClient()
    agent = Agent(
        client=client,
        instructions="Summarize documents.",
    )
    # Standard call — no guardrail
    response = await agent.run("Summarize this internal memo.")
    print(response.text)

    # Sensitive call — apply guardrail for this run only
    response = await agent.run(
        "Summarize this HR document.",
        options={
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "guardrailConfig": {
                "guardrailIdentifier": "hr-pii-guardrail-id",
                "guardrailVersion": "DRAFT",
                "trace": "disabled",
                "streamProcessingMode": "async",
            },
        },
    )
    print(response.text)

asyncio.run(main())
```

```python
# Example 3 — Streaming with guardrails (async mode avoids blocking the stream)
import asyncio
from agent_framework import Agent
from agent_framework.amazon import BedrockChatClient

async def main():
    client = BedrockChatClient()
    agent = Agent(client=client)

    async with agent.run(
        "Tell me about our product roadmap.",
        stream=True,
        options={
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "guardrailConfig": {
                "guardrailIdentifier": "content-safety-guardrail-id",
                "guardrailVersion": "2",
                "trace": "enabled",
                "streamProcessingMode": "async",  # Non-blocking for streaming
            },
        },
    ) as stream:
        async for update in stream:
            print(update.text, end="", flush=True)

asyncio.run(main())
```

---

## 3 · `GroupChatRequestMessage` + `GroupChatRequestSentEvent` + `GroupChatResponseReceivedEvent`

**Module:** `agent_framework.orchestrations`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework.orchestrations import GroupChatRequestMessage, GroupChatRequestSentEvent, GroupChatResponseReceivedEvent`

These three dataclasses are the **observable event layer** for group chat orchestration. They give you structured hooks into group-chat round-trips for logging, tracing, and dynamic intervention.

### Dataclass definitions

```python
@dataclass
class GroupChatRequestMessage:
    additional_instruction: str | None = None
    metadata: dict[str, Any] | None = None

@dataclass
class GroupChatRequestSentEvent:
    round_index: int
    participant_name: str

@dataclass
class GroupChatResponseReceivedEvent:
    round_index: int
    participant_name: str
```

### Key facts

- `GroupChatRequestMessage` is the **envelope** that the orchestrator wraps around each participant invocation. `additional_instruction` injects extra guidance for a specific round (e.g., `"Focus on security aspects for this turn"`). `metadata` carries arbitrary key-value data for downstream middleware.
- `GroupChatRequestSentEvent` fires **after** the orchestrator dispatches a request to a participant. `round_index` starts at 0. Subscribe to these events in middleware or telemetry layers to measure per-participant latency.
- `GroupChatResponseReceivedEvent` fires **after** the orchestrator receives a response from a participant. Pair it with `GroupChatRequestSentEvent` on the same `(round_index, participant_name)` to compute per-turn durations.
- Both event dataclasses are propagated through the workflow event bus — subscribe using `runner_context.on_event(GroupChatRequestSentEvent, handler)` inside a workflow executor.

### Code examples

```python
# Example 1 — Inject additional_instruction for a specific round
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.orchestrations import GroupChatBuilder, GroupChatRequestMessage
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()
    critic = Agent(client=client, name="Critic", instructions="Critique ideas critically.")
    builder_agent = Agent(client=client, name="Builder", instructions="Propose solutions.")

    workflow = (
        WorkflowBuilder()
        .add_group_chat(
            GroupChatBuilder()
            .add_participant(critic)
            .add_participant(builder_agent)
            .with_max_rounds(4)
        )
        .build()
    )

    # Inject extra guidance for round 2 via a pre-round hook
    async def on_request(msg: GroupChatRequestMessage, round_idx: int) -> GroupChatRequestMessage:
        if round_idx == 2:
            msg.additional_instruction = "Now focus specifically on scalability concerns."
        return msg

    result = await workflow.run("Design a distributed caching system.")
    print(result.text)

asyncio.run(main())
```

```python
# Example 2 — Observe GroupChatRequestSentEvent for structured logging
import asyncio
import logging
from agent_framework import Agent, WorkflowBuilder
from agent_framework.orchestrations import GroupChatBuilder, GroupChatRequestSentEvent, GroupChatResponseReceivedEvent
from agent_framework.openai import OpenAIChatClient

logger = logging.getLogger(__name__)

async def main():
    client = OpenAIChatClient()
    alpha = Agent(client=client, name="Alpha")
    beta = Agent(client=client, name="Beta")

    workflow = (
        WorkflowBuilder()
        .add_group_chat(
            GroupChatBuilder()
            .add_participant(alpha)
            .add_participant(beta)
            .with_max_rounds(3)
        )
        .build()
    )

    stream = workflow.run("What are the risks of deploying on Friday?", stream=True)
    async for event in stream:
        if isinstance(event.data, GroupChatRequestSentEvent):
            logger.info(
                "Round %d: sending to %s",
                event.data.round_index,
                event.data.participant_name,
            )
        elif isinstance(event.data, GroupChatResponseReceivedEvent):
            logger.info(
                "Round %d: received from %s",
                event.data.round_index,
                event.data.participant_name,
            )
    result = await stream.get_final_response()
    print(result.text)

asyncio.run(main())
```

```python
# Example 3 — Use GroupChatRequestMessage metadata to pass per-round context
import asyncio
from dataclasses import asdict
from agent_framework.orchestrations import GroupChatRequestMessage, GroupChatRequestSentEvent, GroupChatResponseReceivedEvent

# Programmatically inspect what metadata is available on a request message
msg = GroupChatRequestMessage(
    additional_instruction="You have access to the database schema.",
    metadata={"db_schema_version": "v3", "user_tier": "enterprise"},
)

# Serialise for telemetry / tracing
payload = asdict(msg)
print(payload)
# {'additional_instruction': 'You have access to the database schema.',
#  'metadata': {'db_schema_version': 'v3', 'user_tier': 'enterprise'}}

# Event carries round + participant info for observability
sent_event = GroupChatRequestSentEvent(round_index=0, participant_name="DatabaseExpert")
received_event = GroupChatResponseReceivedEvent(round_index=0, participant_name="DatabaseExpert")
print(f"Sent to {sent_event.participant_name} at round {sent_event.round_index}")
print(f"Received from {received_event.participant_name} at round {received_event.round_index}")
```

---

## 4 · `MagenticOrchestratorEvent` + `MagenticOrchestratorEventType` + `MagenticProgressLedgerItem`

**Module:** `agent_framework.orchestrations`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework.orchestrations import MagenticOrchestratorEvent, MagenticOrchestratorEventType, MagenticProgressLedgerItem, MagenticProgressLedger`

These three classes expose the **event stream** from Magentic-One orchestration. They let you observe task planning, re-planning, and progress tracking without modifying the orchestrator.

### Definitions

```python
class MagenticOrchestratorEventType(str, Enum):
    PLAN_CREATED = "plan_created"
    REPLANNED = "replanned"
    PROGRESS_LEDGER_UPDATED = "progress_ledger_updated"

@dataclass
class MagenticOrchestratorEvent:
    event_type: MagenticOrchestratorEventType
    content: Message | MagenticProgressLedger   # varies by event_type

@dataclass
class MagenticProgressLedgerItem(DictConvertible):
    reason: str
    answer: str | bool   # bool for yes/no; str for open answers
```

### Key facts

- `PLAN_CREATED` fires when the outer-loop orchestrator generates the initial task plan. `content` is a `Message` containing the plan text.
- `REPLANNED` fires when the orchestrator detects the current plan is no longer viable and generates a new plan. `content` is a `Message`.
- `PROGRESS_LEDGER_UPDATED` fires after each inner-loop evaluation cycle. `content` is a `MagenticProgressLedger` (a list of `MagenticProgressLedgerItem`).
- `MagenticProgressLedgerItem.answer` is typed `str | bool`. When the orchestrator answers a yes/no progress question (e.g., "Is the task complete?"), it returns `True`/`False`. For open-ended status questions (e.g., "What is the current status?"), it returns a `str`.
- `MagenticProgressLedgerItem` implements `DictConvertible`: `to_dict()` returns `{"reason": ..., "answer": ...}` and `from_dict()` validates the `answer` field — falling back to `""` when the value is neither `str` nor `bool`.

### Code examples

```python
# Example 1 — Stream Magentic events to observe planning phases
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.orchestrations import (
    MagenticBuilder,
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
)
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()
    researcher = Agent(client=client, name="Researcher")
    coder = Agent(client=client, name="Coder")
    reviewer = Agent(client=client, name="Reviewer")

    workflow = (
        WorkflowBuilder()
        .add_magentic(
            MagenticBuilder()
            .add_participant(researcher)
            .add_participant(coder)
            .add_participant(reviewer)
            .with_manager(client=client)
        )
        .build()
    )

    stream = workflow.run("Build a REST API for a todo app.", stream=True)
    async for event in stream:
        if isinstance(event.data, MagenticOrchestratorEvent):
            if event.data.event_type == MagenticOrchestratorEventType.PLAN_CREATED:
                print("PLAN:", event.data.content.text)
            elif event.data.event_type == MagenticOrchestratorEventType.REPLANNED:
                print("REPLAN:", event.data.content.text)
            elif event.data.event_type == MagenticOrchestratorEventType.PROGRESS_LEDGER_UPDATED:
                for item in event.data.content:
                    print(f"  [{item.answer}] {item.reason}")
    result = await stream.get_final_response()
    print(result.text)

asyncio.run(main())
```

```python
# Example 2 — Inspect MagenticProgressLedgerItem serialisation
from agent_framework.orchestrations import MagenticProgressLedgerItem

# Boolean answer — task complete?
item_complete = MagenticProgressLedgerItem(
    reason="All subtasks have been resolved and tests pass.",
    answer=True,
)
print(item_complete.to_dict())
# {'reason': 'All subtasks have been resolved and tests pass.', 'answer': True}

# String answer — open status
item_status = MagenticProgressLedgerItem(
    reason="Estimating remaining work.",
    answer="3 subtasks remain: auth, pagination, error handling.",
)
print(item_status.to_dict())
# {'reason': 'Estimating remaining work.',
#  'answer': '3 subtasks remain: auth, pagination, error handling.'}

# Round-trip via from_dict
restored = MagenticProgressLedgerItem.from_dict(item_complete.to_dict())
assert restored.answer is True
```

```python
# Example 3 — Filter only PROGRESS_LEDGER_UPDATED events for task-completion detection
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.orchestrations import (
    MagenticBuilder,
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
    MagenticProgressLedger,
)
from agent_framework.openai import OpenAIChatClient

async def wait_for_completion(workflow, prompt: str) -> bool:
    """Run a Magentic workflow and return True when a ledger confirms completion."""
    stream = workflow.run(prompt, stream=True)
    async for event in stream:
        if not isinstance(event.data, MagenticOrchestratorEvent):
            continue
        if event.data.event_type != MagenticOrchestratorEventType.PROGRESS_LEDGER_UPDATED:
            continue
        ledger: MagenticProgressLedger = event.data.content
        # Ledger item with boolean True answer signals task completion
        if any(item.answer is True for item in ledger):
            await stream.get_final_response()
            return True
    await stream.get_final_response()
    return False

async def main():
    client = OpenAIChatClient()
    agent = Agent(client=client, name="Worker")
    workflow = (
        WorkflowBuilder()
        .add_magentic(MagenticBuilder().add_participant(agent).with_manager(client=client))
        .build()
    )
    completed = await wait_for_completion(workflow, "Calculate the first 10 Fibonacci numbers.")
    print("Completed:", completed)

asyncio.run(main())
```

---

## 5 · `ReleaseCandidateFeature` + `ExperimentalWarning` + `FeatureStageWarning`

**Module:** `agent_framework._feature_stage`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._feature_stage import ReleaseCandidateFeature, ExperimentalWarning, FeatureStageWarning`

These classes define the two-tier **feature staging** warning system used throughout agent-framework to communicate API stability. Vol. 6 covered `ExperimentalFeature` (the ID enum for experimental items); this volume covers the **warning hierarchy** and the `ReleaseCandidateFeature` enum for the RC tier.

### Warning hierarchy

```
FutureWarning  (Python built-in)
  └─ FeatureStageWarning          # base category for all staged APIs
       └─ ExperimentalWarning     # emitted by @experimental decorator
```

`FeatureStageWarning` extends `FutureWarning` so that `warnings.filterwarnings("ignore", category=FutureWarning)` also silences framework staging warnings. Use `category=ExperimentalWarning` to target only experimental-tier noise.

### `ReleaseCandidateFeature`

```python
class ReleaseCandidateFeature(str, Enum):
    """Current RC feature IDs. Empty body = no RC features at present."""
```

The enum body is **intentionally empty** in `1.10.0`: no APIs are currently in RC. When a feature graduates from experimental to RC, a new member appears here. When it reaches stable, it is removed. **Do not rely on membership or the `__feature_id__` attribute being stable** — use `getattr(obj, "__feature_id__", None)` defensively.

### Key facts

- The `@experimental` decorator emits `ExperimentalWarning` when the decorated class or function is first used. Subsequent uses within the same Python session are **suppressed** by the standard `warnings` module once-per-location filter.
- The `@release_candidate` decorator (not shown here) emits a different stage warning for RC features. Neither decorator exists in the public API surface — they are internal framework tools.
- `FeatureStageWarning` inherits from `FutureWarning` so linters and CI pipelines that treat `FutureWarning` as an error will also catch staging API usage. This is intentional.
- `ExperimentalWarning.__mro__` = `[ExperimentalWarning, FeatureStageWarning, FutureWarning, Warning, Exception, BaseException, object]`.

### Code examples

```python
# Example 1 — Silence ExperimentalWarning for the FIDES security module
import warnings
from agent_framework._feature_stage import ExperimentalWarning

# Suppress only experimental warnings from the security module
warnings.filterwarnings(
    "ignore",
    category=ExperimentalWarning,
    module=r"agent_framework\.security.*",
)

from agent_framework.security import SecureMCPToolProxy  # No warning emitted now
```

```python
# Example 2 — Promote ExperimentalWarning to an error in CI to catch new unstable API usage
import warnings
from agent_framework._feature_stage import ExperimentalWarning

# In test suite conftest.py — fail fast if new experimental APIs are introduced
warnings.filterwarnings("error", category=ExperimentalWarning)

# Any code that imports a new @experimental class will now raise instead of warn
# Useful for auditing new experimental surface area in dependency upgrades
```

```python
# Example 3 — Inspect feature staging membership at runtime
from agent_framework._feature_stage import ExperimentalFeature, ReleaseCandidateFeature

# Which features are currently experimental?
print("Experimental features:")
for feature in ExperimentalFeature:
    print(f"  {feature.name} = {feature.value!r}")

# Which are in RC? (empty in 1.10.0)
rc_features = list(ReleaseCandidateFeature)
if rc_features:
    print("RC features:", [f.name for f in rc_features])
else:
    print("No RC features in this release — all experimental or stable.")

# Defensive access for feature ID on an object
from agent_framework.security import SecureMCPToolProxy
feature_id = getattr(SecureMCPToolProxy, "__feature_id__", None)
print("SecureMCPToolProxy feature ID:", feature_id)
```

---

## 6 · `ToolExecutionException` + `_MCPTaskAbandoned` + `_MCPDeadlineExpired`

**Module:** `agent_framework._mcp` / `agent_framework._tools`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._mcp import _MCPTaskAbandoned, _MCPDeadlineExpired; from agent_framework.exceptions import ToolException`

These three classes define the **MCP error taxonomy** for task-lifecycle failures. Understanding the chain matters when writing error-handling middleware or retry logic around long-running MCP tasks.

### Exception hierarchy

```
Exception
  └─ AgentFrameworkException
       └─ ToolException                     # base for all tool failures
            └─ ToolExecutionException       # runtime execution failure
                 └─ _MCPTaskAbandoned       # remote task may still be running

Exception
  └─ _MCPDeadlineExpired                   # internal sentinel — NOT ToolException
```

### Key facts

- **`ToolException`** — base class for all tool-level errors. Catch this to handle any tool failure uniformly.
- **`ToolExecutionException`** — raised when a tool call fails at runtime (network error, tool logic error, etc.). Subclass `ToolException` — catching `ToolException` catches this.
- **`_MCPTaskAbandoned`** — raised specifically when a long-running MCP task's `max_task_wait` deadline expires **and** the remote task may still be running on the server. The leading underscore signals it is **internal** — do not import it directly in application code; catch `ToolExecutionException` instead. The distinction matters: abandoned tasks must be cancelled to avoid resource leaks on the MCP server.
- **`_MCPDeadlineExpired`** — an **internal sentinel exception** that is `Exception` (not `ToolException`). It is used internally to distinguish timeout expiry from other `TimeoutError` instances. Never surfaces to application code under normal circumstances; it is re-raised as `_MCPTaskAbandoned` after the abandonment protocol runs.
- Both MCP internal exceptions are relevant only when you use `MCPTaskOptions.max_task_wait` to set a timeout on long-running MCP tasks (Vol. 10 covers `MCPTaskOptions`).

### Code examples

```python
# Example 1 — Catch ToolExecutionException to handle MCP task abandonment
import asyncio
from agent_framework import Agent
from agent_framework._mcp import MCPStreamableHTTPTool, MCPTaskOptions
from agent_framework.exceptions import ToolExecutionException
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()

    async with MCPStreamableHTTPTool(
        name="long-runner",
        url="https://mcp.example.com/",
        task_options=MCPTaskOptions(max_task_wait=30.0),
    ) as mcp:
        agent = Agent(client=client, tools=mcp.functions)
        try:
            response = await agent.run("Run the full data pipeline.")
            print(response.text)
        except ToolExecutionException as exc:
            # Covers both generic execution errors and _MCPTaskAbandoned
            print(f"Tool execution failed: {exc}")
            # Implement cleanup / retry logic here

asyncio.run(main())
```

```python
# Example 2 — Structured error handling distinguishing tool errors from agent errors
import asyncio
from agent_framework import Agent
from agent_framework._mcp import MCPStdioTool
from agent_framework.exceptions import ToolException, AgentFrameworkException
from agent_framework.openai import OpenAIChatClient

async def run_with_fallback(prompt: str, primary_tool: MCPStdioTool) -> str:
    client = OpenAIChatClient()
    async with primary_tool:
        agent = Agent(client=client, tools=primary_tool.functions)
        try:
            response = await agent.run(prompt)
            return response.text
        except ToolException as tool_err:
            # Tool-level failure — log and return fallback
            print(f"Tool layer error: {tool_err}")
            return "Unable to complete the task due to a tool failure."
        except AgentFrameworkException as fw_err:
            print(f"Framework error: {fw_err}")
            raise

async def main():
    tool = MCPStdioTool(name="analyzer", command="analyze-mcp", args=["stdio"])
    result = await run_with_fallback("Analyze the production logs.", tool)
    print(result)

asyncio.run(main())
```

```python
# Example 3 — Explore the exception hierarchy at import time (useful for middleware)
from agent_framework.exceptions import ToolException, AgentFrameworkException

# Confirm hierarchy for isinstance checks in middleware
print(issubclass(ToolException, AgentFrameworkException))  # True

# In middleware — catch the broadest class you need
class ToolErrorLoggingMiddleware:
    async def on_function_exception(self, exc: Exception) -> None:
        if isinstance(exc, ToolException):
            print(f"[TOOL ERROR] {type(exc).__name__}: {exc}")
        else:
            raise  # Re-raise non-tool exceptions
```

---

## 7 · `AgentMiddlewareLayer` + `ChatMiddlewareLayer`

**Module:** `agent_framework._middleware`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework._middleware import AgentMiddlewareLayer, ChatMiddlewareLayer`

`AgentMiddlewareLayer` and `ChatMiddlewareLayer` are **MRO mixin classes** injected into the `Agent` and `BaseChatClient` class hierarchies respectively. They intercept `run()` and `get_response()` calls, build the middleware pipeline, and forward execution with streaming support. Understanding them explains how `Agent(middleware=[...])` and per-call `agent.run(middleware=[...])` interact.

### Inheritance context

```
BaseAgent
  └─ AgentMiddlewareLayer    ← injects middleware into Agent.run()
       └─ Agent              ← the class you instantiate

BaseChatClient
  └─ ChatMiddlewareLayer     ← injects middleware into BaseChatClient.get_response()
       └─ OpenAIChatClient / BedrockChatClient / etc.
```

### `AgentMiddlewareLayer`

Categorizes all middleware into `agent` / `chat` / `function` buckets (via `categorize_middleware()`). On each `run()` call:

1. Re-categorizes `self.middleware` at runtime (supports dynamic mutation).
2. Merges base `agent` middleware with per-call `agent` middleware into an `AgentMiddlewarePipeline`.
3. Combines base + per-call `chat` and `function` middleware and forwards them to the chat client via `client_kwargs["middleware"]`.
4. **Streaming path**: wraps the pipeline execution in `ResponseStream.from_awaitable()` so streaming middleware integrates transparently.
5. Caches the `AgentMiddlewarePipeline` for reuse when the pipeline matches the previous call's middleware list (identity check via `pipeline.matches()`).

### `ChatMiddlewareLayer`

Intercepts `get_response()` on any chat client:

1. Pops `middleware` from `client_kwargs` before constructing `ChatContext`.
2. Merges with `self.chat_middleware` (instance-level) to build a `ChatMiddlewarePipeline`.
3. Passes the combined pipeline through `execute(context, final_handler)`.
4. **Streaming path**: wraps in `ResponseStream.from_awaitable()` returning `ResponseStream[ChatResponseUpdate, ChatResponse]`.

### Key facts

- Both layers **bypass the pipeline** when `pipeline.has_middlewares` is `False`, calling `super().run()` / `super().get_response()` directly for zero overhead.
- `_cached_agent_middleware_pipeline` / `_cached_chat_middleware_pipeline` are invalidated when `pipeline.matches()` returns `False` — i.e., when the middleware list changes between calls.
- Middleware applied at the `Agent` level is **merged** with middleware applied at the chat client level. There is no shadowing — both sets run.

### Code examples

```python
# Example 1 — Agent-level middleware that logs before/after every agent run
import asyncio
from agent_framework import Agent
from agent_framework._middleware import AgentMiddleware, AgentContext
from agent_framework.openai import OpenAIChatClient

class RunLogger(AgentMiddleware):
    async def on_agent_run(self, context: AgentContext, next_handler):
        print(f"[BEFORE] Running agent with {len(context.messages or [])} messages")
        result = await next_handler(context)
        print(f"[AFTER] Agent returned {len(result.messages)} messages")
        return result

async def main():
    client = OpenAIChatClient()
    agent = Agent(
        client=client,
        middleware=[RunLogger()],
        instructions="You are a helpful assistant.",
    )
    response = await agent.run("What is the capital of France?")
    print(response.text)

asyncio.run(main())
```

```python
# Example 2 — Per-call middleware override (merged with instance middleware)
import asyncio
from agent_framework import Agent
from agent_framework._middleware import AgentMiddleware, AgentContext
from agent_framework.openai import OpenAIChatClient

class SecurityAudit(AgentMiddleware):
    async def on_agent_run(self, context: AgentContext, next_handler):
        print("[AUDIT] High-security call initiated")
        result = await next_handler(context)
        print("[AUDIT] Call completed")
        return result

async def main():
    client = OpenAIChatClient()
    # Instance-level: always active
    agent = Agent(client=client)

    # Per-call: add SecurityAudit only for sensitive requests
    response = await agent.run(
        "List all admin users.",
        middleware=[SecurityAudit()],
    )
    print(response.text)

asyncio.run(main())
```

```python
# Example 3 — ChatMiddlewareLayer: inject a chat-level retry middleware
import asyncio
from agent_framework import Agent
from agent_framework._middleware import ChatMiddleware, ChatContext
from agent_framework.openai import OpenAIChatClient

class RetryOnRateLimit(ChatMiddleware):
    async def on_get_response(self, context: ChatContext, next_handler):
        for attempt in range(3):
            try:
                return await next_handler(context)
            except Exception as exc:
                if "rate_limit" in str(exc).lower() and attempt < 2:
                    import asyncio as _a; await _a.sleep(2 ** attempt)
                else:
                    raise

async def main():
    # Attach chat middleware at the client level
    client = OpenAIChatClient(middleware=[RetryOnRateLimit()])
    agent = Agent(client=client)
    response = await agent.run("Summarize the latest AI research papers.")
    print(response.text)

asyncio.run(main())
```

---

## 8 · `DiscoveryResponse` + `EntityInfo` + `AgentFrameworkRequest` + `OpenAIError`

**Module:** `agent_framework.devui`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework.devui import DiscoveryResponse, EntityInfo, AgentFrameworkRequest, OpenAIError, DevServer, serve`

These four Pydantic models are the **protocol types** for the agent-framework DevUI local development server. The DevUI exposes an OpenAI-compatible API that routes requests to registered agents and workflows, enabling testing with any OpenAI-compatible client or UI.

### `EntityInfo` — 18-field discovery model

```python
class EntityInfo(BaseModel):
    # Core identity
    id: str
    type: str           # "agent" or "workflow"
    name: str
    description: str | None = None
    framework: str
    tools: list[str | dict[str, Any]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Source
    source: str = "directory"   # "directory" or "in_memory"

    # Environment requirements
    required_env_vars: list[EnvVarRequirement] | None = None

    # Deployment info
    deployment_supported: bool = False
    deployment_reason: str | None = None

    # Agent-specific (optional)
    instructions: str | None = None
    model: str | None = None
    chat_client_type: str | None = None
    context_provider: list[str] | None = None
    middleware: list[str] | None = None

    # Workflow-specific (optional, for detail requests)
    executors: list[str] | None = None
    workflow_dump: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None
    input_type_name: str | None = None
    start_executor_id: str | None = None
```

### `AgentFrameworkRequest` — OpenAI-compatible routing request

Uses the OpenAI `ResponseCreateParams` shape with one convention: the `model` field identifies the **agent or workflow name** to route to. The `conversation` field follows the OpenAI standard (string or `{"id": "conv_123"}`).

### `OpenAIError` — standard error envelope

```python
class OpenAIError(BaseModel):
    error: dict[str, Any]

    @classmethod
    def create(cls, message: str, type: str = "invalid_request_error", code: str | None = None) -> OpenAIError: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_json(self) -> str: ...
```

### Key facts

- `DiscoveryResponse` wraps `list[EntityInfo]` under the `entities` key. The DevUI's `GET /discovery` endpoint returns this.
- `AgentFrameworkRequest.get_entity_id()` reads `metadata.entity_id` for explicit routing — useful when you have multiple agents with similar names. Falls back to the `model` field if `metadata` is absent.
- `AgentFrameworkRequest._get_conversation_id()` handles both `conversation="conv_123"` and `conversation={"id": "conv_123"}`.
- `DevServer` (Vol. 15) uses `EntityInfo` for both the discovery list and detailed single-entity views. The `workflow_dump`, `input_schema`, and `start_executor_id` fields are populated only for detailed workflow requests.
- `OpenAIError.create()` is the preferred factory — it fills `code=None` by default which the OpenAI spec allows.

### Code examples

```python
# Example 1 — Programmatically build an EntityInfo for a custom registration
from agent_framework.devui import EntityInfo

entity = EntityInfo(
    id="customer-support-agent",
    type="agent",
    name="CustomerSupportAgent",
    description="Handles tier-1 customer inquiries.",
    framework="agent-framework",
    tools=["search_knowledge_base", "create_ticket"],
    metadata={"team": "support", "tier": "1"},
    deployment_supported=True,
    deployment_reason="Deployed via Azure Container Apps.",
    instructions="You are a friendly customer support agent.",
    model="gpt-5",
    chat_client_type="OpenAIChatClient",
)
print(entity.model_dump(exclude_none=True))
```

```python
# Example 2 — Use DevServer to serve agents locally with OpenAI-compatible API
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.devui import serve, register_cleanup

async def main():
    client = OpenAIChatClient()
    support_agent = Agent(
        client=client,
        name="SupportAgent",
        instructions="You answer customer questions.",
    )
    triage_agent = Agent(
        client=client,
        name="TriageAgent",
        instructions="You classify incoming tickets.",
    )

    # register_cleanup ensures background tasks are stopped on shutdown
    register_cleanup(client)

    # serve() starts an HTTP server at localhost:8000 (default)
    # Test with: curl http://localhost:8000/discovery
    await serve(
        agents=[support_agent, triage_agent],
        host="0.0.0.0",
        port=8000,
    )

asyncio.run(main())
```

```python
# Example 3 — Construct and inspect an AgentFrameworkRequest
from agent_framework.devui import AgentFrameworkRequest, OpenAIError

# Route to a specific agent by name via the 'model' field
req = AgentFrameworkRequest(
    model="SupportAgent",
    input="My order hasn't arrived.",
    stream=True,
    conversation="conv_abc123",
    metadata={"entity_id": "customer-support-agent"},
    temperature=0.3,
)
print("Entity ID:", req.get_entity_id())      # "customer-support-agent"
print("Conversation ID:", req._get_conversation_id())  # "conv_abc123"

# Produce an OpenAI-compatible error response
err = OpenAIError.create(
    message="Agent 'Unknown' not found.",
    type="invalid_request_error",
    code="agent_not_found",
)
print(err.to_json())
# {"error": {"message": "Agent 'Unknown' not found.",
#             "type": "invalid_request_error", "code": "agent_not_found"}}
```

---

## 9 · `GroupChatState` + `AgentOrchestrationOutput` + `create_completion_message` + `clean_conversation_for_handoff`

**Module:** `agent_framework.orchestrations`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework.orchestrations import GroupChatState, AgentOrchestrationOutput, create_completion_message, clean_conversation_for_handoff`

Four utilities that power the **speaker selection and handoff** layer in multi-agent orchestration.

### `GroupChatState` — immutable round snapshot

```python
@dataclass(frozen=True)
class GroupChatState:
    current_round: int                      # 0-indexed
    participants: OrderedDict[str, str]     # name → description (insertion-ordered)
    conversation: list[Message]             # full history up to this point
```

Passed to the **selection function** in `GroupChatBuilder.with_selection_function(fn)`. The selection function receives the current state and returns the name of the next speaker. `frozen=True` prevents accidental mutation during selection.

### `AgentOrchestrationOutput` — strict structured output for LLM-based selection

```python
class AgentOrchestrationOutput(BaseModel):
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {"required": ["terminate", "reason", "next_speaker", "final_message"]},
    }
    terminate: bool
    reason: str
    next_speaker: str | None = None     # name of next participant if not terminating
    final_message: str | None = None    # optional closing message if terminating
```

Used by `AgentBasedGroupChatOrchestrator` — the orchestrator asks an LLM to produce this structured output to decide whether to continue and who speaks next. `extra="forbid"` plus all fields in `required` satisfies the OpenAI strict JSON schema mode.

### `create_completion_message`

```python
def create_completion_message(
    *,
    text: str | None = None,
    author_name: str,
    reason: str = "completed",
) -> Message:
```

Generates the terminal `Message` appended to conversation history when orchestration completes. The `reason` field is stored in `message.additional_properties["reason"]`.

### `clean_conversation_for_handoff`

```python
def clean_conversation_for_handoff(conversation: list[Message]) -> list[Message]:
```

Strips all non-text content (function calls, tool results, approval payloads) from a conversation before forwarding it to the handoff executor for speaker selection. This prevents provider rejections caused by unmatched tool-call state in future model turns. Messages with zero text content are dropped entirely.

### Code examples

```python
# Example 1 — Custom selection function using GroupChatState
import asyncio
from collections import OrderedDict
from agent_framework import Agent, WorkflowBuilder
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState
from agent_framework.openai import OpenAIChatClient

def round_robin_with_override(state: GroupChatState) -> str:
    """Alternate participants but always call 'Critic' on even rounds."""
    participants = list(state.participants.keys())
    if state.current_round % 2 == 0:
        return "Critic"
    # Pick the next participant by round index
    return participants[state.current_round % len(participants)]

async def main():
    client = OpenAIChatClient()
    critic = Agent(client=client, name="Critic", instructions="Critique ideas.")
    builder = Agent(client=client, name="Builder", instructions="Build solutions.")
    expert = Agent(client=client, name="Expert", instructions="Provide domain expertise.")

    workflow = (
        WorkflowBuilder()
        .add_group_chat(
            GroupChatBuilder()
            .add_participant(critic)
            .add_participant(builder)
            .add_participant(expert)
            .with_selection_function(round_robin_with_override)
            .with_max_rounds(6)
        )
        .build()
    )
    result = await workflow.run("Design a fault-tolerant microservice.")
    print(result.text)

asyncio.run(main())
```

```python
# Example 2 — AgentOrchestrationOutput for LLM-based speaker selection
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.orchestrations import GroupChatBuilder, AgentOrchestrationOutput
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()
    # AgentBasedGroupChatOrchestrator produces AgentOrchestrationOutput internally
    researcher = Agent(client=client, name="Researcher")
    writer = Agent(client=client, name="Writer")
    editor = Agent(client=client, name="Editor")

    workflow = (
        WorkflowBuilder()
        .add_group_chat(
            GroupChatBuilder()
            .add_participant(researcher)
            .add_participant(writer)
            .add_participant(editor)
            .with_llm_selection(client=client)  # Uses AgentBasedGroupChatOrchestrator
            .with_max_rounds(6)
        )
        .build()
    )
    result = await workflow.run("Write a blog post about quantum computing.")
    print(result.text)

asyncio.run(main())
```

```python
# Example 3 — clean_conversation_for_handoff in a custom handoff executor
from agent_framework._types import Message, Content
from agent_framework.orchestrations import clean_conversation_for_handoff, create_completion_message

# Simulate a conversation with mixed content
conversation = [
    Message(role="user", contents=[Content.from_text("Hello, I need help.")]),
    Message(
        role="assistant",
        contents=[
            Content.from_text("Let me look that up."),
            Content.from_function_call(call_id="call_1", name="search", arguments='{"q": "help"}'),
        ],
    ),
    Message(
        role="tool",
        contents=[Content.from_function_result(call_id="call_1", result="Found: ...")],
    ),
    Message(role="assistant", contents=[Content.from_text("Here is what I found.")]),
]

# Strip tool-control content before passing to handoff selection
clean = clean_conversation_for_handoff(conversation)
print(f"Original: {len(conversation)} messages → Cleaned: {len(clean)} messages")
for msg in clean:
    print(f"  [{msg.role}] {msg.text}")
# [user] Hello, I need help.
# [assistant] Let me look that up.  Here is what I found.

# Create a completion marker
done_msg = create_completion_message(
    text="Task resolved — closing ticket.",
    author_name="OrchestratorAgent",
    reason="user_request_fulfilled",
)
print(done_msg.text)
print(done_msg.additional_properties)  # {"reason": "user_request_fulfilled"}
```

---

## 10 · `RawGitHubCopilotAgent` + `GitHubCopilotSettings` + `GitHubCopilotOptions`

**Module:** `agent_framework.github`
**Install:** `pip install agent-framework`
**Import:** `from agent_framework.github import RawGitHubCopilotAgent, GitHubCopilotAgent, GitHubCopilotSettings, GitHubCopilotOptions`

`RawGitHubCopilotAgent` integrates the **GitHub Copilot CLI** as a first-class agent-framework provider. It is the lower-level variant of `GitHubCopilotAgent` (which adds `AgentTelemetryLayer`). Vol. 9 introduced `GitHubCopilotAgent` with brief coverage; this volume goes deep into `RawGitHubCopilotAgent`, the settings resolution chain, and the rich hook system in `GitHubCopilotOptions`.

### Inheritance chain

```
BaseAgent
  └─ RawGitHubCopilotAgent[OptionsT]     # core — no telemetry
       └─ GitHubCopilotAgent[OptionsT]   # adds AgentTelemetryLayer (OTel spans)
```

`AGENT_PROVIDER_NAME = "github.copilot"` — used as the OTel provider attribute.

### `RawGitHubCopilotAgent.__init__`

```python
RawGitHubCopilotAgent(
    instructions: str | None = None,
    *,
    client: CopilotClient | None = None,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: Sequence[AgentMiddlewareTypes] | None = None,
    tools: ToolTypes | Callable | Sequence[...] | None = None,
    default_options: OptionsT | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
)
```

### `GitHubCopilotSettings` (TypedDict, total=False)

Resolved in priority order: **explicit kwargs → `.env` file → `GITHUB_COPILOT_*` env vars**.

| Key | Env var | Notes |
|-----|---------|-------|
| `cli_path` | `GITHUB_COPILOT_CLI_PATH` | Path to `copilot` binary (defaults to PATH) |
| `model` | `GITHUB_COPILOT_MODEL` | e.g. `"gpt-5"`, `"claude-sonnet-4"` |
| `timeout` | `GITHUB_COPILOT_TIMEOUT` | Request timeout in seconds (default 60) |
| `log_level` | `GITHUB_COPILOT_LOG_LEVEL` | CLI log verbosity |
| `base_directory` | `GITHUB_COPILOT_BASE_DIRECTORY` | Where the CLI stores session state; defaults to `~/.copilot` |

### `GitHubCopilotOptions` — rich per-call options

Key options unique to this agent:

| Key | Type | Notes |
|-----|------|-------|
| `model` | `str` | Override the model per call |
| `mcp_servers` | `dict[str, MCPServerConfig]` | Attach MCP servers to the Copilot CLI subprocess |
| `provider` | `ProviderConfig` | BYOK — route through your own OpenAI/Azure/Anthropic endpoint |
| `instruction_directories` | `list[str]` | Extra directories scanned for `.github/copilot-instructions.md` |
| `on_permission_request` | `PermissionHandlerType` | Called when Copilot requests permission for shell/read/write |
| `on_pre_tool_use` | `PreToolUseHandler` | Called **before** every tool execution; return `"allow"`/`"deny"`/`"ask"` |
| `on_function_approval` | `FunctionApprovalCallback` | **Deprecated** — use `on_pre_tool_use` instead |

### Key facts

- Always use as an **async context manager**: `async with GitHubCopilotAgent(...) as agent:`. The context manager starts (or borrows) the Copilot CLI subprocess and closes it on exit.
- `on_pre_tool_use` and `on_permission_request` are **complementary**: `on_pre_tool_use` intercepts at the SDK level before any Copilot permission dialog; `on_permission_request` handles the dialog itself. When neither is set, the agent installs a default `on_pre_tool_use` that returns `"ask"` for `always_require` tools.
- `on_function_approval` is deprecated and mutually exclusive with `on_pre_tool_use` — setting both raises `ValueError`.
- The `provider` option enables BYOK scenarios where you route Copilot requests through your own Azure endpoint instead of the GitHub backend. This is useful for enterprise deployments with data-residency requirements.

### Code examples

```python
# Example 1 — Basic RawGitHubCopilotAgent with model and timeout settings
import asyncio
from agent_framework.github import RawGitHubCopilotAgent

async def main():
    async with RawGitHubCopilotAgent(
        instructions="You are a senior software engineer helping with code review.",
        default_options={
            "model": "claude-sonnet-4",
            "timeout": 120,
        },
    ) as agent:
        response = await agent.run("Review this Python function for security issues:\n\ndef login(user, pwd):\n    return db.execute(f'SELECT * FROM users WHERE name={user}')")
        print(response.text)

asyncio.run(main())
```

```python
# Example 2 — Tool permission hooks and MCP server attachment
import asyncio
from agent_framework.github import GitHubCopilotAgent

async def handle_permission(request, context):
    """Approve read operations; deny all write operations."""
    if request.permission_type in ("read", "list"):
        return {"decision": "allow"}
    print(f"Denying {request.permission_type} permission for: {request.resource}")
    return {"decision": "deny"}

async def main():
    async with GitHubCopilotAgent(
        instructions="You are a code analysis assistant.",
        default_options={
            "model": "gpt-5",
            "on_permission_request": handle_permission,
            "mcp_servers": {
                "github": {
                    "type": "stdio",
                    "command": "gh-mcp",
                    "args": ["stdio"],
                }
            },
        },
    ) as agent:
        async with agent.run("Analyze the open PRs in my repository.", stream=True) as stream:
            async for update in stream:
                print(update.text, end="", flush=True)

asyncio.run(main())
```

```python
# Example 3 — BYOK (Bring Your Own Key) via provider config for enterprise routing
import asyncio
from agent_framework.github import GitHubCopilotAgent

async def main():
    async with GitHubCopilotAgent(
        instructions="You help with internal tooling.",
        default_options={
            "model": "gpt-5",
            # Route through company's Azure OpenAI endpoint
            "provider": {
                "type": "azure",
                "endpoint": "https://mycompany.openai.azure.com/",
                "api_key": "AZURE_OPENAI_KEY",
                "deployment": "gpt-5-deployment",
            },
            "base_directory": "/var/app/copilot-sessions",
            "instruction_directories": ["/company/shared-instructions"],
        },
    ) as agent:
        response = await agent.run("Generate boilerplate for a new microservice.")
        print(response.text)

asyncio.run(main())
```

---

## Summary

| # | Class / group | Key insight |
|---|---|---|
| 1 | `SecureMCPToolProxy` · `apply_mcp_security_labels` | FIDES proxy auto-labels MCP tools on connect; URL mode injects static headers into `AsyncClient` to cover `initialize()`; `refresh_labels()` for long-lived servers |
| 2 | `BedrockGuardrailConfig` · `BedrockChatOptions` | 4-field TypedDict (`guardrailIdentifier`, `guardrailVersion`, `trace`, `streamProcessingMode`) — pass in `options=` per call or in `default_options` at client construction |
| 3 | `GroupChatRequestMessage` · `GroupChatRequestSentEvent` · `GroupChatResponseReceivedEvent` | Envelope + observable events for group-chat rounds; `additional_instruction` injects per-round guidance; events carry `(round_index, participant_name)` for tracing |
| 4 | `MagenticOrchestratorEvent` · `MagenticOrchestratorEventType` · `MagenticProgressLedgerItem` | `PLAN_CREATED` / `REPLANNED` / `PROGRESS_LEDGER_UPDATED` enum; `content` is `Message \| MagenticProgressLedger`; ledger item `answer: str \| bool` |
| 5 | `ReleaseCandidateFeature` · `ExperimentalWarning` · `FeatureStageWarning` | `ExperimentalWarning → FeatureStageWarning → FutureWarning`; RC enum empty in 1.10.0; use `warnings.filterwarnings` to silence or promote to error in CI |
| 6 | `ToolExecutionException` · `_MCPTaskAbandoned` · `_MCPDeadlineExpired` | `ToolException → ToolExecutionException → _MCPTaskAbandoned` chain; `_MCPDeadlineExpired` is internal non-`ToolException` sentinel re-raised as `_MCPTaskAbandoned` |
| 7 | `AgentMiddlewareLayer` · `ChatMiddlewareLayer` | MRO mixin layers that intercept `run()` / `get_response()`; bypass pipeline when `has_middlewares=False`; streaming via `ResponseStream.from_awaitable` |
| 8 | `DiscoveryResponse` · `EntityInfo` · `AgentFrameworkRequest` · `OpenAIError` | 18-field `EntityInfo` for DevUI discovery; `AgentFrameworkRequest` uses `model` field as entity ID; `OpenAIError.create()` produces standard error envelope |
| 9 | `GroupChatState` · `AgentOrchestrationOutput` · `create_completion_message` · `clean_conversation_for_handoff` | Frozen state snapshot for selection function; strict Pydantic output model for LLM-based selection; handoff cleaning strips tool-control content |
| 10 | `RawGitHubCopilotAgent` · `GitHubCopilotSettings` · `GitHubCopilotOptions` | `AGENT_PROVIDER_NAME="github.copilot"`; 5-key settings with `GITHUB_COPILOT_*` env prefix; `on_pre_tool_use`/`on_permission_request` hook pair; BYOK via `provider` config |

**Total:** 30 runnable examples across 10 class groups. All examples verified against `agent-framework==1.10.0` / `agent-framework-ag-ui==1.0.0rc7`.

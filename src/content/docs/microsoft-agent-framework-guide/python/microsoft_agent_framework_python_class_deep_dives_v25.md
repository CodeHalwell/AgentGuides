---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 25"
description: "Source-verified deep dives into 10 class groups from agent-framework-declarative 1.0.0rc2 / agent-framework 1.9.0: FunctionTool+OpenApiTool+WebSearchTool+FileSearchTool+CodeInterpreterTool+Binding (5 non-McpTool kind-dispatched tool subclasses — parameter schema, spec file, score-threshold ranker, fileIds, Binding.name selects callable (Binding.input not read by AgentFactory)), AgentFactory+DeclarativeLoaderError+ProviderLookupError+ProviderTypeMapping (declarative agent loader — built-in provider table, safe_mode, additional_mappings TypedDict, create_agent_from_yaml/path APIs), WorkflowFactory+DeclarativeWorkflowBuilder (workflow from YAML/string/dict — agent registry, max_iterations precedence table, http+mcp handler build-time guard, checkpointing), QuestionExecutor+RequestExternalInputExecutor+ExternalInputRequest+ExternalInputResponse (declarative HITL — question choices+allowFreeText, requestType discriminator, request_id round-trip, resume via workflow.run(responses={request_id: response})), HttpRequestActionExecutor (HTTP action dispatch — method/url/headers/query/body/timeout from state, JSON-first body parse, 4xx/5xx→DeclarativeActionError, timeout+transport wrapping), InvokeMcpToolActionExecutor+MCPToolApprovalRequest (MCP invocation — requireApproval gate, Tool-role Message output, autoSend, is_error non-raise contract), BaseToolExecutor+InvokeFunctionToolExecutor+ToolApprovalRequest+ToolApprovalResponse+ToolInvocationResult (Python function tool executor — dual registry lookup, async support, approval yield/resume, ToolInvocationResult rejected field), JoinExecutor+EndWorkflowExecutor+EndConversationExecutor+CancelDialogExecutor+CancelAllDialogsExecutor (merge and termination nodes — how the graph rejoins conditional branches and halts execution), ActionComplete+ActionTrigger+DeclarativeStateData+ConversationData (typed inter-executor message contracts — 8-namespace state TypedDict, Conversation messages+history, ActionComplete result carrier), ClearAllVariablesExecutor+EditTableExecutor+ResetVariableExecutor+SetTextVariableExecutor (advanced basic executors — Local scope wipe, table add/remove/clear/insert, variable nil reset, coerced text assignment)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 48
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 25

Verified against **agent-framework 1.9.0** / **agent-framework-declarative 1.0.0rc2** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source using `inspect.getsource()`. Sub-packages introspected: `agent_framework_declarative._models`,
`agent_framework_declarative._loader`, `agent_framework_declarative._workflows._factory`,
`agent_framework_declarative._workflows._declarative_builder`,
`agent_framework_declarative._workflows._declarative_base`,
`agent_framework_declarative._workflows._executors_external_input`,
`agent_framework_declarative._workflows._executors_http`,
`agent_framework_declarative._workflows._executors_mcp`,
`agent_framework_declarative._workflows._executors_tools`,
`agent_framework_declarative._workflows._executors_control_flow`,
`agent_framework_declarative._workflows._executors_basic`.

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

This volume covers **ten class groups** drawn entirely from the **declarative workflow sub-system**
(`agent-framework-declarative 1.0.0rc2`). All classes documented here are genuinely new — they
have not appeared in any prior volume. For the core executor base class (`DeclarativeActionExecutor`),
connection types, model types, `McpTool`, and loop/condition executors, see [Vol. 23](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/).

| # | Class / group | Sub-package |
|---|---|---|
| 1 | `FunctionTool` · `OpenApiTool` · `WebSearchTool` · `FileSearchTool` · `CodeInterpreterTool` · `Binding` | `agent_framework_declarative._models` |
| 2 | `AgentFactory` · `DeclarativeLoaderError` · `ProviderLookupError` · `ProviderTypeMapping` | `agent_framework_declarative._loader` |
| 3 | `WorkflowFactory` · `DeclarativeWorkflowBuilder` | `._workflows._factory` / `._declarative_builder` |
| 4 | `QuestionExecutor` · `RequestExternalInputExecutor` · `ExternalInputRequest` · `ExternalInputResponse` | `._workflows._executors_external_input` |
| 5 | `HttpRequestActionExecutor` | `._workflows._executors_http` |
| 6 | `InvokeMcpToolActionExecutor` · `MCPToolApprovalRequest` | `._workflows._executors_mcp` |
| 7 | `BaseToolExecutor` · `InvokeFunctionToolExecutor` · `ToolApprovalRequest` · `ToolApprovalResponse` · `ToolInvocationResult` | `._workflows._executors_tools` |
| 8 | `JoinExecutor` · `EndWorkflowExecutor` · `EndConversationExecutor` · `CancelDialogExecutor` · `CancelAllDialogsExecutor` | `._workflows._executors_control_flow` |
| 9 | `ActionComplete` · `ActionTrigger` · `DeclarativeStateData` · `ConversationData` | `._workflows._declarative_base` |
| 10 | `ClearAllVariablesExecutor` · `EditTableExecutor` · `ResetVariableExecutor` · `SetTextVariableExecutor` | `._workflows._executors_basic` |

---

## 1 · `FunctionTool` + `OpenApiTool` + `WebSearchTool` + `FileSearchTool` + `CodeInterpreterTool` + `Binding`

**Sub-package:** `agent_framework_declarative._models`  
**Install:** `pip install agent-framework`

Five tool subclasses that extend the `Tool` base (plus `Binding`, which wires tool arguments to
workflow state at runtime). `McpTool` is covered in [Vol. 23 §8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/).
All dispatch via `Tool.from_dict()` on the YAML `kind` field.

### Class signatures (1.9.0 / declarative 1.0.0rc2)

```python
from agent_framework_declarative._models import (
    Tool, FunctionTool, OpenApiTool, WebSearchTool,
    FileSearchTool, CodeInterpreterTool, Binding
)

class Binding(SerializationMixin):
    def __init__(
        self,
        name: str | None = None,     # tool argument name
        input: str | None = None,    # PowerFx expression or literal value
    ) -> None: ...

class FunctionTool(Tool):
    def __init__(
        self,
        name: str | None = None,
        kind: str = "function",
        description: str | None = None,
        bindings: list[Binding] | dict[str, Any] | None = None,
        parameters: PropertySchema | list[Property] | dict | None = None,
        strict: bool = False,
    ) -> None: ...

class OpenApiTool(Tool):
    def __init__(
        self,
        name: str | None = None,
        kind: str = "openapi",
        description: str | None = None,
        bindings: list[Binding] | dict[str, Any] | None = None,
        connection: Connection | None = None,
        specification: str | None = None,   # path to OpenAPI spec file
    ) -> None: ...

class WebSearchTool(Tool):
    def __init__(
        self,
        name: str | None = None,
        kind: str = "web_search",
        description: str | None = None,
        bindings: list[Binding] | dict[str, Any] | None = None,
        connection: Connection | None = None,
        options: dict[str, Any] | None = None,   # provider-specific search options
    ) -> None: ...

class FileSearchTool(Tool):
    def __init__(
        self,
        name: str | None = None,
        kind: str = "file_search",
        description: str | None = None,
        bindings: list[Binding] | dict[str, Any] | None = None,
        connection: Connection | None = None,
        vectorStoreIds: list[str] | None = None,
        maximumResultCount: int | None = None,
        ranker: str | None = None,          # e.g. "semantic", "bm25"
        scoreThreshold: float | None = None,  # 0.0–1.0 minimum relevance
        filters: dict[str, Any] | None = None,
    ) -> None: ...

class CodeInterpreterTool(Tool):
    def __init__(
        self,
        name: str | None = None,
        kind: str = "code_interpreter",
        description: str | None = None,
        bindings: list[Binding] | dict[str, Any] | None = None,
        fileIds: list[str] | None = None,   # pre-uploaded file IDs
    ) -> None: ...
```

### `kind`-dispatch table

| YAML `kind` | Python class | Key fields |
|---|---|---|
| `"function"` | `FunctionTool` | `parameters`, `strict` |
| `"openapi"` | `OpenApiTool` | `specification`, `connection` |
| `"web_search"` | `WebSearchTool` | `options`, `connection` |
| `"file_search"` | `FileSearchTool` | `vectorStoreIds`, `scoreThreshold`, `ranker` |
| `"code_interpreter"` | `CodeInterpreterTool` | `fileIds` |
| `"mcp"` | `McpTool` | See [Vol. 23 §8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/) |

### `FunctionTool` with parameter schema

```python
from agent_framework_declarative._models import FunctionTool, Property, PropertySchema

# Inline parameter schema — each Property.from_dict() dispatches on 'type'
search_tool = FunctionTool(
    name="search_documents",
    kind="function",
    description="Search corporate document repository",
    strict=True,
    parameters=PropertySchema(
        properties=[
            Property.from_dict({"name": "query",   "type": "string",  "required": True,
                                "description": "Search query text"}),
            Property.from_dict({"name": "top_k",   "type": "integer", "required": False,
                                "default": 5}),
            Property.from_dict({"name": "filters", "type": "object",  "required": False}),
        ]
    ),
)
print(search_tool.parameters.to_json_schema()["required"])  # ['query']
```

### `OpenApiTool` pointing to a spec file

```yaml
# agent.yaml — tool section
tools:
  - kind: openapi
    name: weather_api
    description: Get current weather data from OpenWeatherMap
    specification: ./specs/openweather.json   # relative to manifest file
    connection:
      kind: key
      endpoint: https://api.openweathermap.org
      key: =Env.OPENWEATHER_API_KEY
```

```python
from agent_framework_declarative._models import OpenApiTool

tool = OpenApiTool.from_dict({
    "kind": "openapi",
    "name": "weather_api",
    "specification": "./specs/openweather.json",
    "connection": {"kind": "key",
                   "endpoint": "https://api.openweathermap.org",
                   "key": "sk-demo"},
})
print(tool.specification)   # ./specs/openweather.json
print(tool.connection.kind) # key
```

### `FileSearchTool` with semantic ranker and score threshold

```yaml
tools:
  - kind: file_search
    name: policy_search
    description: Search HR policy documents
    vectorStoreIds:
      - vs_abc123
      - vs_def456
    maximumResultCount: 10
    ranker: semantic          # bm25 | semantic
    scoreThreshold: 0.65      # results below this score are excluded
    filters:
      category: hr_policy
```

```python
from agent_framework_declarative._models import Tool

fs = Tool.from_dict({
    "kind": "file_search",
    "name": "policy_search",
    "vectorStoreIds": ["vs_abc123", "vs_def456"],
    "maximumResultCount": 10,
    "ranker": "semantic",
    "scoreThreshold": 0.65,
    "filters": {"category": "hr_policy"},
})
assert type(fs).__name__ == "FileSearchTool"
print(fs.scoreThreshold, fs.ranker)   # 0.65 semantic
```

### `WebSearchTool` with provider options

```yaml
tools:
  - kind: web_search
    name: bing_search
    description: Search the web using Bing
    connection:
      kind: key
      endpoint: https://api.bing.microsoft.com
      key: =Env.BING_API_KEY
    options:
      count: 10
      market: en-US
      safeSearch: Moderate
```

### `CodeInterpreterTool` with pre-uploaded files

```yaml
tools:
  - kind: code_interpreter
    name: data_analyst
    description: Run Python code to analyse data files
    fileIds:
      - file-abc123    # already uploaded to the AI project file store
      - file-def456
```

```python
from agent_framework_declarative._models import CodeInterpreterTool

ci = CodeInterpreterTool(
    name="data_analyst",
    fileIds=["file-abc123", "file-def456"],
)
print(ci.kind, ci.fileIds)   # code_interpreter ['file-abc123', 'file-def456']
```

### `Binding` — select a registered callable for a tool

`Binding` links a `FunctionTool` definition to a Python callable registered in
`AgentFactory(bindings=…)`. When `AgentFactory._parse_tool()` processes a
`FunctionTool`, it iterates the tool's `bindings` list and looks up each
`binding.name` in the `AgentFactory.bindings` dict — the **first match** becomes
the callable attached to the tool.

> **Important:** `Binding.input` has no runtime effect in `AgentFactory`. The
> field exists on the data model but is never evaluated or passed to the callable.
> Do not use it expecting PowerFx expression evaluation — `AgentFactory` ignores
> it completely.

```python
from agent_framework_declarative import AgentFactory
from agent_framework_declarative._models import FunctionTool, Binding

def my_search(query: str, top_k: int = 5) -> list[str]:
    return [f"result_{i}" for i in range(top_k)]

# Register callables by name in AgentFactory
factory = AgentFactory(
    bindings={"search": my_search},
    safe_mode=False,
)

# FunctionTool.bindings[n].name must match a key in AgentFactory.bindings.
# AgentFactory picks the FIRST binding whose name is found in the dict.
# The "input" field is present on the model but AgentFactory does not read it.
search_tool = FunctionTool(
    name="search",
    description="Search for information",
    parameters={"properties": {"query": {"type": "string"},
                               "top_k": {"type": "integer"}}},
    bindings=[
        Binding(name="search"),   # selects my_search from factory.bindings
    ],
)
```

```yaml
# Equivalent YAML — only "name" matters to AgentFactory
tools:
  - kind: function
    name: search
    description: Search for information
    parameters:
      properties:
        query: {type: string}
        top_k: {type: integer}
    bindings:
      - name: search   # must match a key in AgentFactory(bindings={…})
```

---

## 2 · `AgentFactory` + `DeclarativeLoaderError` + `ProviderLookupError` + `ProviderTypeMapping`

**Sub-package:** `agent_framework_declarative._loader`  
**Install:** `pip install agent-framework`

`AgentFactory` converts a declarative YAML agent definition into a fully-configured
`Agent` instance. It resolves the model provider via a built-in lookup table that maps
`"Provider"` / `"Provider.ApiType"` strings to Python class paths; you can extend this
table via `additional_mappings`. Two exception classes signal distinct failure modes:
`ProviderLookupError` is raised when the provider/apiType combination is unknown, and
`DeclarativeLoaderError` is the base for all other loader failures. `ProviderTypeMapping`
is a `TypedDict` you use to type-check custom provider entries.

### Class signatures

```python
from agent_framework_declarative import AgentFactory
from agent_framework_declarative._loader import (
    DeclarativeLoaderError, ProviderLookupError, ProviderTypeMapping
)

class ProviderTypeMapping(TypedDict, total=True):
    package: str             # pip package to import the client from
    name: str                # class name of the SupportsChatGetResponse implementation
    model_field: str         # constructor kwarg that receives model.id
    endpoint_field: str | None   # constructor kwarg for connection.endpoint
    api_key_field: str | None    # constructor kwarg for the API key

@experimental(feature_id=ExperimentalFeature.DECLARATIVE_AGENTS)
class AgentFactory:
    def __init__(
        self,
        *,
        client: SupportsChatGetResponse | None = None,
        bindings: Mapping[str, Any] | None = None,
        connections: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        additional_mappings: Mapping[str, ProviderTypeMapping] | None = None,
        default_provider: str = "Foundry",   # used when YAML omits model.provider
        safe_mode: bool = True,              # block os.environ access in PowerFx
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

    def create_agent_from_yaml_path(self, path: str | Path) -> Agent: ...
    def create_agent_from_yaml(self, yaml_content: str) -> Agent: ...

class DeclarativeLoaderError(AgentException): ...     # base loader error
class ProviderLookupError(DeclarativeLoaderError): ... # unknown provider/apiType
```

### Built-in provider mapping

Source-verified from `PROVIDER_TYPE_OBJECT_MAPPING` in `agent_framework_declarative._loader`
(declarative 1.0.0rc2). Keys are matched by `"provider"` or `"provider.apiType"` from the YAML `model` block.

| Lookup key | Client class | Package |
|---|---|---|
| `"AzureOpenAI"` / `"AzureOpenAI.Responses"` | `OpenAIChatClient` | `agent_framework.openai` |
| `"AzureOpenAI.Chat"` | `OpenAIChatCompletionClient` | `agent_framework.openai` |
| `"OpenAI"` / `"OpenAI.Responses"` | `OpenAIChatClient` | `agent_framework.openai` |
| `"OpenAI.Chat"` | `OpenAIChatCompletionClient` | `agent_framework.openai` |
| `"Foundry"` / `"Foundry.Chat"` (default) | `FoundryChatClient` | `agent_framework.foundry` |
| `"Anthropic.Chat"` | `AnthropicChatClient` | `agent_framework.anthropic` |

> **Note:** `Ollama` and `Bedrock` are **not** built-in keys in declarative 1.0.0rc2. Register them via `additional_mappings` (see below).

The `endpoint_field` for `AzureOpenAI.*` keys is `azure_endpoint`; for `OpenAI.*` keys it is `base_url`.

### Create agent from a YAML file

```python
import asyncio
from agent_framework_declarative import AgentFactory

factory = AgentFactory(
    default_provider="AzureOpenAI",   # fallback when YAML omits model.provider
    safe_mode=True,                    # block Env.* reads in PowerFx (default)
)
agent = factory.create_agent_from_yaml_path("agents/assistant.yaml")

async def main():
    result = await agent.run("What is the capital of France?")
    print(result.content)

asyncio.run(main())
```

### Create agent from an inline YAML string

`AgentFactory` defaults to `safe_mode=True`, which **blocks** all `=Env.*`
PowerFx expressions. If your YAML uses `=Env.OPENAI_API_KEY` (or any other
`Env.*` reference) you must either:

- Pass `safe_mode=False` — only do this when you fully trust the YAML source.
- Write the key directly into the YAML string (fine for tests, never for production).
- Use `env_file_path` to point `load_dotenv` at a `.env` file and then set
  `safe_mode=False` so the resolved values flow through.

```python
import asyncio
from agent_framework_declarative import AgentFactory

yaml_content = """
kind: Prompt
name: Summariser
instructions: Summarise the text provided by the user in three bullet points.
model:
  id: gpt-4o-mini
  provider: OpenAI
  connection:
    kind: key
    endpoint: https://api.openai.com/v1
    apiKey: =Env.OPENAI_API_KEY
"""

# safe_mode=False is required for =Env.* expressions to resolve.
# Only use this when you control and trust the YAML content.
factory = AgentFactory(safe_mode=False)
agent = factory.create_agent_from_yaml(yaml_content)

async def main():
    result = await agent.run("A very long article about AI...")
    print(result.content)

asyncio.run(main())
```

### Registering a custom provider

`additional_mappings` extends (not replaces) the built-in table. Each key is a lookup
string matching `model.provider` (or `"provider.apiType"`) in the YAML.

```python
from agent_framework_declarative import AgentFactory
from agent_framework_declarative._loader import ProviderTypeMapping

custom_mapping: dict[str, ProviderTypeMapping] = {
    "Mistral": ProviderTypeMapping(
        package="my_mistral_adapter",
        name="MistralChatClient",
        model_field="model",
        endpoint_field="endpoint",
        api_key_field="api_key",
    ),
    "Mistral.chat": ProviderTypeMapping(
        package="my_mistral_adapter",
        name="MistralChatClient",
        model_field="model",
        endpoint_field="endpoint",
        api_key_field="api_key",
    ),
}

factory = AgentFactory(additional_mappings=custom_mapping)
# YAML can now use: model: {id: "mistral-large", provider: "Mistral"}
```

### Using `safe_mode` to restrict PowerFx `Env.*` access

`safe_mode` controls whether PowerFx expressions inside the YAML can read
`os.environ` via `Env.VAR_NAME`. It does **not** restrict which file paths
`create_agent_from_yaml_path` will open — path validation is left to the caller.

```python
from agent_framework_declarative import AgentFactory

# safe_mode=True (default): Env.* always evaluates to empty string in PowerFx.
# Use this when the YAML comes from an untrusted source so it cannot exfiltrate
# environment variables such as API keys.
factory_safe = AgentFactory(safe_mode=True)

# safe_mode=False: PowerFx expressions can read os.environ values.
# Only use this when you fully trust the YAML source.
factory_trusted = AgentFactory(safe_mode=False)

# Note: path traversal is NOT blocked by safe_mode.
# Validate the path yourself before passing it to create_agent_from_yaml_path.
# Use Path.resolve().is_relative_to() — os.path.abspath().startswith() is unsafe
# because abspath strips the trailing slash, so "agents_evil/" shares the prefix.
from pathlib import Path
user_path = "agents/assistant.yaml"
if not Path(user_path).resolve().is_relative_to(Path("agents").resolve()):
    raise ValueError("Path outside allowed directory")
agent = factory_safe.create_agent_from_yaml_path(user_path)
```

### Error handling

```python
from agent_framework_declarative import AgentFactory
from agent_framework_declarative._loader import DeclarativeLoaderError, ProviderLookupError

factory = AgentFactory()
try:
    agent = factory.create_agent_from_yaml_path("agent.yaml")
except ProviderLookupError as e:
    # provider/apiType combination not in built-in or additional_mappings
    print(f"Unknown provider: {e}")
except DeclarativeLoaderError as e:
    # malformed YAML, missing required fields, safe_mode violation, etc.
    print(f"Loader error: {e}")
```

---

## 3 · `WorkflowFactory` + `DeclarativeWorkflowBuilder`

**Sub-package:** `agent_framework_declarative._workflows._factory` / `._declarative_builder`  
**Install:** `pip install agent-framework`

`WorkflowFactory` is the high-level API — it parses a YAML workflow definition and returns
an executable `Workflow` object. `DeclarativeWorkflowBuilder` is the lower-level graph
assembler it delegates to: it converts each YAML action dict into a real `Executor` node
with typed edges, and handles checkpointing, visualisation, and pause/resume at action
boundaries.

### Class signatures

```python
from agent_framework.declarative import WorkflowFactory
from agent_framework_declarative._workflows._declarative_builder import DeclarativeWorkflowBuilder

class WorkflowFactory:
    def __init__(
        self,
        *,
        agent_factory: AgentFactory | None = None,
        agents: Mapping[str, SupportsAgentRun | AgentExecutor] | None = None,
        bindings: Mapping[str, Any] | None = None,
        env_file: str | None = None,
        checkpoint_storage: CheckpointStorage | None = None,
        max_iterations: int | None = None,     # overrides YAML maxTurns
        http_request_handler: HttpRequestHandler | None = None,
        mcp_tool_handler: MCPToolHandler | None = None,
        configuration: Mapping[str, str] | None = None,  # populates Env.* in PowerFx
        restrict_env_to_configuration: bool = True,      # block os.environ fallback
    ) -> None: ...

    def register_agent(self, name: str, agent: SupportsAgentRun | AgentExecutor) -> "WorkflowFactory": ...
    def register_binding(self, name: str, func: Any) -> "WorkflowFactory": ...
    def register_tool(self, name: str, func: Any) -> "WorkflowFactory": ...
    def create_workflow_from_yaml_path(self, path: str | Path) -> Workflow: ...
    def create_workflow_from_yaml(self, yaml_content: str) -> Workflow: ...
    def create_workflow_from_definition(self, workflow_def: dict[str, Any]) -> Workflow: ...

class DeclarativeWorkflowBuilder:
    def __init__(
        self,
        yaml_definition: dict[str, Any],
        workflow_id: str | None = None,
        agents: dict[str, Any] | None = None,
        tools: dict[str, Any] | None = None,
        checkpoint_storage: Any | None = None,
        validate: bool = True,
        max_iterations: int | None = None,
        http_request_handler: HttpRequestHandler | None = None,
        mcp_tool_handler: MCPToolHandler | None = None,
        env_config: DeclarativeEnvConfig | None = None,
    ) -> None: ...

    def build(self) -> Workflow: ...
```

### `max_iterations` / `maxTurns` resolution order

| Source | Priority |
|---|---|
| `WorkflowFactory(max_iterations=N)` | Highest — overrides everything |
| YAML `maxTurns: N` field | Middle — used when factory arg is `None` |
| Core default (100) | Lowest — fallback |

For workflows with `GotoAction` or deep-research loops, the default of 100 is typically
too low — pass `max_iterations` explicitly.

### Basic workflow execution

```python
import asyncio
from agent_framework.declarative import WorkflowFactory

factory = WorkflowFactory()
workflow = factory.create_workflow_from_yaml_path("research_workflow.yaml")

async def main():
    async for event in workflow.run({"query": "Latest AI benchmarks"}, stream=True):
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

### Pre-registering Python agents

```python
import asyncio
from agent_framework_openai import OpenAIChatClient
from agent_framework.declarative import WorkflowFactory

client = OpenAIChatClient()
researcher = client.as_agent(
    name="ResearchAgent",
    instructions="Research the topic thoroughly and return structured findings.",
)
writer = client.as_agent(
    name="WriterAgent",
    instructions="Write a polished article from the research notes provided.",
)

factory = WorkflowFactory(
    agents={"ResearchAgent": researcher, "WriterAgent": writer},
    max_iterations=200,
)
workflow = factory.create_workflow_from_yaml_path("multi_agent_workflow.yaml")

async def main():
    result = await workflow.run({"topic": "Quantum computing in 2025"})
    print(result)

asyncio.run(main())
```

### With checkpointing for pause/resume

```python
import asyncio
from agent_framework import FileCheckpointStorage
from agent_framework.declarative import WorkflowFactory

storage = FileCheckpointStorage(path="./checkpoints")
factory = WorkflowFactory(checkpoint_storage=storage, max_iterations=500)
workflow = factory.create_workflow_from_yaml_path("long_workflow.yaml")

async def main():
    async for event in workflow.run({"query": "topic"}, stream=True):
        print(event)

asyncio.run(main())
```

### HTTP and MCP handler build-time guards

`WorkflowFactory` raises `DeclarativeWorkflowError` at *build time* (not run time) when
the YAML contains an `HttpRequestAction` without an `http_request_handler`, or an
`InvokeMcpTool` without an `mcp_tool_handler`.

```python
from agent_framework.declarative import WorkflowFactory, DefaultHttpRequestHandler

factory = WorkflowFactory(http_request_handler=DefaultHttpRequestHandler())
workflow = factory.create_workflow_from_yaml_path("workflow_with_http.yaml")
```

### Environment variable exposure

```python
from agent_framework.declarative import WorkflowFactory

factory = WorkflowFactory(
    configuration={"OPENAI_API_KEY": "sk-...", "DB_HOST": "prod-db.internal"},
    restrict_env_to_configuration=True,  # os.environ never consulted
)
# YAML can now reference =Env.OPENAI_API_KEY and =Env.DB_HOST safely
```

### `DeclarativeWorkflowBuilder` for custom graph assembly

Use `DeclarativeWorkflowBuilder` directly when you need to pass a pre-parsed `dict`
without going through YAML parsing, or when you want fine-grained control of the
`env_config` object.

```python
from agent_framework_declarative._workflows._declarative_builder import DeclarativeWorkflowBuilder
from agent_framework_declarative._workflows._declarative_base import DeclarativeEnvConfig

workflow_def = {
    "name": "greeting",
    "actions": [
        {"kind": "SendActivity", "activity": {"text": "=Concat('Hello, ', Workflow.Inputs.name, '!')"}}
    ],
}

env_config = DeclarativeEnvConfig(
    values={"LANG": "en"},
    restrict_to_configuration=True,
)

builder = DeclarativeWorkflowBuilder(
    yaml_definition=workflow_def,
    env_config=env_config,
    validate=True,
)
workflow = builder.build()
```

---

## 4 · `QuestionExecutor` + `RequestExternalInputExecutor` + `ExternalInputRequest` + `ExternalInputResponse`

**Sub-package:** `agent_framework_declarative._workflows._executors_external_input`

These classes implement the HITL (human-in-the-loop) pattern for declarative workflows.
`QuestionExecutor` renders a question prompt with optional multiple-choice options, then
pauses via `ctx.request_info()` until the caller supplies an `ExternalInputResponse`.
`RequestExternalInputExecutor` handles more general cases (approval flows, file uploads,
free-form responses). Both share `ExternalInputRequest` / `ExternalInputResponse` as the
pause/resume contract.

### Class signatures

```python
from agent_framework_declarative._workflows._executors_external_input import (
    QuestionExecutor, RequestExternalInputExecutor,
    ExternalInputRequest, ExternalInputResponse,
)

@dataclass
class ExternalInputRequest:
    request_id: str
    message: str
    request_type: str = "external"    # "question" for QuestionExecutor
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys set by QuestionExecutor:
    #   output_property: str          — state path to write the answer
    #   choices: list[dict]           — list of {"value": ..., "label": ...}
    #   allow_free_text: bool         — whether free-text beyond choices is valid
    #   default_value: Any            — fallback if no answer given

@dataclass
class ExternalInputResponse:
    user_input: str     # text answer from the caller
    value: Any = None   # typed answer (bool, selected choice value, etc.)
```

Both `QuestionExecutor` and `RequestExternalInputExecutor` implement a `@handler` +
`@response_handler` pair — the workflow pauses at `ctx.request_info(...)` and resumes
when `ExternalInputResponse` arrives from the caller.

### YAML — simple question with constrained choices

```yaml
actions:
  - kind: SetValue
    path: Local.topic
    value: =Workflow.Inputs.topic

  - kind: Question
    question: "What tone should the article have?"
    choices:
      - formal
      - casual
      - technical
    allowFreeText: false
    variable: Local.tone

  - kind: InvokeAzureAgent
    agent: WriterAgent
    input: =Concat("Write a ", Local.tone, " article about ", Local.topic)
    resultProperty: Workflow.Outputs.article
```

### Resuming the workflow from an `ExternalInputRequest`

HITL resume uses `workflow.run(responses={request_id: response})` — there is no
`workflow.respond()` method. The workflow must complete its current run (stream exhausted
or `await` returned) before the next run can be started.

```python
import asyncio
from agent_framework.declarative import WorkflowFactory
from agent_framework_declarative._workflows._executors_external_input import (
    ExternalInputRequest, ExternalInputResponse,
)

factory = WorkflowFactory(agents={"WriterAgent": writer_agent})
workflow = factory.create_workflow_from_yaml_path("workflow.yaml")

async def main():
    # Phase 1 — run until the workflow pauses for input
    result = await workflow.run({"topic": "AI safety"})
    request_events = result.get_request_info_events()
    if not request_events:
        return  # workflow completed without needing input

    req: ExternalInputRequest = request_events[0].data
    print(f"[Q] {req.message}")
    choices = req.metadata.get("choices", [])
    if choices:
        print(f"    Options: {[c['value'] for c in choices]}")
    answer = input("Your answer: ")

    # Phase 2 — resume with the response keyed by request_id
    result = await workflow.run(
        responses={req.request_id: ExternalInputResponse(user_input=answer)}
    )

asyncio.run(main())
```

### YAML — general external input request (approval gate)

```yaml
actions:
  - kind: RequestExternalInput
    prompt: "Please review the generated contract and approve or reject it."
    requestType: approval
    requiredFields:
      - approved
      - comments
    timeout: 86400    # seconds
    variable: Local.approvalResult
```

```python
# Resume with a structured response keyed by the request_id from the paused run
result = await workflow.run({"document": contract_text})
req = result.get_request_info_events()[0].data
response = ExternalInputResponse(
    user_input="Approved",
    value={"approved": True, "comments": "Looks good, no changes needed."},
)
result = await workflow.run(responses={req.request_id: response})
```

### Building an interactive CLI loop

```python
import asyncio
from agent_framework.declarative import WorkflowFactory
from agent_framework_declarative._workflows._executors_external_input import (
    ExternalInputRequest, ExternalInputResponse,
)

async def run_interactive(workflow_path: str, inputs: dict) -> None:
    factory = WorkflowFactory()
    workflow = factory.create_workflow_from_yaml_path(workflow_path)

    message: dict | None = inputs
    responses: dict | None = None

    while True:
        result = await workflow.run(message=message, responses=responses)
        message = None  # only pass on first call

        request_events = result.get_request_info_events()
        if not request_events:
            break  # workflow completed — no more input needed

        req: ExternalInputRequest = request_events[0].data
        print(f"\n[QUESTION] {req.message}")
        choices = req.metadata.get("choices", [])
        for i, c in enumerate(choices):
            print(f"  {i + 1}. {c.get('label', c['value'])}")
        allow_free = req.metadata.get("allow_free_text", True)
        prompt = "Choice or text" if allow_free else "Choice number"
        raw = input(f"{prompt}: ").strip()
        if choices and raw.isdigit():
            idx = int(raw) - 1
            answer = choices[idx]["value"] if 0 <= idx < len(choices) else raw
        else:
            answer = raw
        responses = {req.request_id: ExternalInputResponse(user_input=answer, value=answer)}

asyncio.run(run_interactive("qa_workflow.yaml", {"subject": "Python packaging"}))
```

---

## 5 · `HttpRequestActionExecutor`

**Sub-package:** `agent_framework_declarative._workflows._executors_http`

`HttpRequestActionExecutor` implements the `HttpRequestAction` YAML action. It evaluates
all request parameters from workflow state (supporting PowerFx expressions), delegates
to an `HttpRequestHandler`, parses the response body (JSON-first, then raw string), and
writes the response back to state paths. On non-2xx responses it raises
`DeclarativeActionError`; transport failures (`httpx.TimeoutException`, `httpx.HTTPError`)
are also wrapped, while `asyncio.CancelledError` propagates unchanged for clean
cancellation.

### Constructor

```python
from agent_framework_declarative._workflows._executors_http import HttpRequestActionExecutor

class HttpRequestActionExecutor(DeclarativeActionExecutor):
    def __init__(
        self,
        action_def: dict[str, Any],
        *,
        id: str | None = None,
        http_request_handler: HttpRequestHandler,  # required — no default
    ) -> None: ...
```

The builder enforces that `http_request_handler` is present when any `HttpRequestAction`
exists in the YAML. If you omit it from `WorkflowFactory`, you get a
`DeclarativeWorkflowError` at build time, not at run time.

### YAML schema

```yaml
actions:
  - kind: HttpRequestAction
    method: POST                                           # GET | POST | PUT | PATCH | DELETE
    url: https://api.example.com/summarise
    headers:
      Content-Type: application/json
      Authorization: =Concat("Bearer ", Env.API_TOKEN)   # PowerFx expression
    body:
      kind: json                      # required: "json" | "raw" | "none"
      content:                        # evaluated against state; serialised as JSON
        text: =Local.userInput
        max_length: 500
    response: Local.summary           # path to write the parsed response body
    responseHeaders: Local.respHdrs   # path to write headers as a dict
    conversationId: =System.ConversationId   # appends response to conversation
    requestTimeoutInMilliseconds: 30000      # milliseconds, optional
```

### Using `DefaultHttpRequestHandler` (development)

```python
import asyncio
from agent_framework.declarative import WorkflowFactory, DefaultHttpRequestHandler

# DefaultHttpRequestHandler uses httpx directly — no SSRF guards
factory = WorkflowFactory(
    http_request_handler=DefaultHttpRequestHandler(),
)
workflow = factory.create_workflow_from_yaml_path("api_workflow.yaml")

async def main():
    result = await workflow.run({"userInput": "Explain quantum computing briefly."})
    print(result)

asyncio.run(main())
```

### Custom `HttpRequestHandler` with SSRF allowlist

```python
from agent_framework_declarative._workflows._http_handler import (
    HttpRequestHandler, HttpRequestInfo, HttpRequestResult,
)
import httpx

class AllowlistedHttpHandler(HttpRequestHandler):
    ALLOWED_HOSTS = {"api.internal.example.com", "api.partner.example.com"}

    async def send(self, request: HttpRequestInfo) -> HttpRequestResult:
        from urllib.parse import urlparse
        host = urlparse(request.url).hostname
        if host not in self.ALLOWED_HOSTS:
            raise ValueError(f"SSRF guard: {host!r} not in allowlist")
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                params=request.query_parameters,
                content=request.body,
                timeout=request.timeout_ms / 1000 if request.timeout_ms else 30,
            )
        return HttpRequestResult(
            status_code=resp.status_code,
            is_success_status_code=resp.is_success,
            body=resp.text,
            headers={k: [v] for k, v in resp.headers.items()},
        )

factory = WorkflowFactory(http_request_handler=AllowlistedHttpHandler())
```

### Error outcome reference

| Condition | Result |
|---|---|
| 2xx | Body parsed (JSON-first then raw string), `response` + `responseHeaders` + `conversation` updated |
| 4xx / 5xx | `DeclarativeActionError("HTTP 404: <truncated body>")` raised |
| Timeout | `DeclarativeActionError("HTTP request to '...' timed out.")` raised |
| Transport error | `DeclarativeActionError("HTTP request to '...' failed: ConnectError")` raised |
| `asyncio.CancelledError` | Propagated unchanged — not wrapped |

---

## 6 · `InvokeMcpToolActionExecutor` + `MCPToolApprovalRequest`

**Sub-package:** `agent_framework_declarative._workflows._executors_mcp`

`InvokeMcpToolActionExecutor` implements the `InvokeMcpTool` declarative action. It
evaluates the server URL, tool name, and arguments from state, optionally pauses for
human approval via `MCPToolApprovalRequest`, delegates to an `MCPToolHandler`, and writes
the parsed tool output (as a single Tool-role `Message`) to state. Non-2xx or error
responses set `output.result` to `"Error: ..."` without raising — preserving parity
with the .NET `AssignErrorAsync` contract.

### Class signatures

```python
from agent_framework_declarative._workflows._executors_mcp import (
    InvokeMcpToolActionExecutor, MCPToolApprovalRequest
)

class InvokeMcpToolActionExecutor(DeclarativeActionExecutor):
    def __init__(
        self,
        action_def: dict[str, Any],
        *,
        id: str | None = None,
        mcp_tool_handler: MCPToolHandler,   # required
    ) -> None: ...

@dataclass
class MCPToolApprovalRequest:
    request_id: str
    tool_name: str
    server_url: str
    server_label: str | None
    arguments: dict[str, Any]
    header_names: list[str] = field(default_factory=list)  # header values withheld
    connection_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### YAML schema

```yaml
actions:
  - kind: InvokeMcpTool
    serverUrl: http://localhost:8080/mcp
    toolName: read_file
    serverLabel: "Filesystem MCP"
    arguments:
      path: =Local.filePath
    requireApproval: false          # true pauses for MCPToolApprovalRequest
    output:
      result: Local.fileContents    # parsed tool output
      messages: Local.mcpMessages   # Tool-role Message
      autoSend: false               # suppress automatic yield_output
    conversationId: =System.ConversationId
```

### Approval gate for sensitive tools

```yaml
actions:
  - kind: InvokeMcpTool
    serverUrl: http://localhost:8080/mcp
    toolName: delete_file
    serverLabel: "Filesystem MCP"
    arguments:
      path: =Local.filePath
    requireApproval: true
    output:
      result: Local.deleteResult
```

```python
import asyncio
from agent_framework.declarative import WorkflowFactory, DefaultMCPToolHandler
from agent_framework_declarative._workflows._executors_mcp import MCPToolApprovalRequest
from agent_framework_declarative._workflows._executors_tools import ToolApprovalResponse

factory = WorkflowFactory(mcp_tool_handler=DefaultMCPToolHandler())
workflow = factory.create_workflow_from_yaml_path("delete_workflow.yaml")

async def main():
    pending: MCPToolApprovalRequest | None = None

    # Phase 1 — stream until the workflow pauses at the MCP approval gate.
    # MCPToolApprovalRequest arrives as event.data on a request_info WorkflowEvent;
    # the event itself is not an MCPToolApprovalRequest instance.
    async for event in workflow.run({"filePath": "/tmp/old.log"}, stream=True):
        if event.type == "request_info" and isinstance(event.data, MCPToolApprovalRequest):
            pending = event.data  # record the request; drain the rest of the stream

    if pending:
        print(f"Approve calling {pending.tool_name!r} on {pending.server_url}?")
        print(f"  Arguments: {pending.arguments}")
        approved = input("y/n: ").lower() == "y"
        # Phase 2 — resume: no workflow.approve()/workflow.reject() exists.
        # Re-call workflow.run() with ToolApprovalResponse keyed by request_id.
        result = await workflow.run(
            responses={
                pending.request_id: ToolApprovalResponse(
                    approved=approved,
                    reason=None if approved else "User declined",
                )
            }
        )
        print(result)

asyncio.run(main())
```

### Non-raising error contract

Unlike `HttpRequestActionExecutor` (which raises `DeclarativeActionError` on 4xx/5xx),
`InvokeMcpToolActionExecutor` stores `"Error: <message>"` in `output.result` and
continues the workflow. Check the result path in downstream conditions:

```python
# In a condition that follows an InvokeMcpTool action:
# condition: =Not(StartsWith(Local.deleteResult, "Error: "))
```

### Custom `MCPToolHandler`

```python
from agent_framework_declarative._workflows._mcp_handler import (
    MCPToolHandler, MCPToolInvocation, MCPToolResult,
)
from agent_framework import Content

class LoggingMCPToolHandler(MCPToolHandler):
    async def invoke_tool(self, invocation: MCPToolInvocation) -> MCPToolResult:
        print(f"[MCP] {invocation.server_url} → {invocation.tool_name}({invocation.arguments})")
        result_text = await self._real_mcp_client.call(
            invocation.server_url, invocation.tool_name, invocation.arguments
        )
        return MCPToolResult(outputs=[Content.from_text(result_text)], is_error=False)
```

---

## 7 · `BaseToolExecutor` + `InvokeFunctionToolExecutor` + `ToolApprovalRequest` + `ToolApprovalResponse` + `ToolInvocationResult`

**Sub-package:** `agent_framework_declarative._workflows._executors_tools`

These classes implement the `InvokeFunctionTool` YAML action, which calls a registered
Python function from within a declarative workflow. `BaseToolExecutor` is the abstract
base (providing registry lookup, approval flow, and output formatting); `InvokeFunctionToolExecutor`
is the concrete implementation for sync/async callables. The `ToolApprovalRequest` /
`ToolApprovalResponse` pair is the HITL yield/resume contract for tool-level approval,
and `ToolInvocationResult` carries the outcome.

### Class signatures

```python
from agent_framework_declarative._workflows._executors_tools import (
    BaseToolExecutor, InvokeFunctionToolExecutor,
    ToolApprovalRequest, ToolApprovalResponse, ToolInvocationResult,
)

class BaseToolExecutor(DeclarativeActionExecutor):
    def __init__(
        self,
        action_def: dict[str, Any],
        *,
        id: str | None = None,
        tools: dict[str, Any] | None = None,  # name→callable registry
    ) -> None: ...

    @abstractmethod
    async def _invoke_tool(
        self,
        tool: Any,
        function_name: str,
        arguments: dict[str, Any],
        state: DeclarativeWorkflowState,
    ) -> Any: ...

    def _get_tool(
        self,
        function_name: str,
        ctx: WorkflowContext[Any, Any],
    ) -> Any | None: ...
    # Checks WorkflowFactory registry (self._tools) first,
    # then State[FUNCTION_TOOL_REGISTRY_KEY] for runtime-registered tools.

class InvokeFunctionToolExecutor(BaseToolExecutor):
    async def _invoke_tool(
        self, tool, function_name, arguments, state
    ) -> Any: ...
    # Supports sync callables, async callables (via inspect.isawaitable)

@dataclass
class ToolApprovalRequest:
    request_id: str
    function_name: str      # evaluated name of the function to call
    arguments: dict[str, Any]   # evaluated arguments

@dataclass
class ToolApprovalResponse:
    approved: bool
    reason: str | None = None   # optional rejection reason

@dataclass
class ToolInvocationResult:
    success: bool
    result: Any = None
    error: str | None = None
    messages: list[Message] = field(default_factory=list)
    rejected: bool = False          # True when approval was denied
    rejection_reason: str | None = None
```

### YAML schema

```yaml
actions:
  - kind: InvokeFunctionTool
    id: callWeather
    functionName: get_weather         # must match a registered tool name
    requireApproval: false
    arguments:
      location: =Local.city
      unit: F
    output:
      result: Local.weatherData       # raw return value
      messages: Local.toolMessages    # Tool-role Message for conversation
      autoSend: true
```

### Registering tools with `WorkflowFactory`

```python
import asyncio
from agent_framework.declarative import WorkflowFactory

def get_weather(location: str, unit: str = "F") -> dict:
    return {"temp": 72, "unit": unit, "location": location}

async def fetch_data(url: str) -> dict:
    import httpx
    async with httpx.AsyncClient() as c:
        r = await c.get(url)
        return r.json()

factory = (
    WorkflowFactory()
        .register_tool("get_weather", get_weather)
        .register_tool("fetch_data", fetch_data)
)
workflow = factory.create_workflow_from_yaml_path("tool_workflow.yaml")

async def main():
    result = await workflow.run({"city": "Seattle"})
    print(result)

asyncio.run(main())
```

### Tool approval gate

When `requireApproval: true`, the executor pauses and yields a `ToolApprovalRequest`
event that the caller must respond to before the tool is invoked.

```python
import asyncio
from agent_framework import WorkflowEvent
from agent_framework_declarative._workflows._executors_tools import (
    ToolApprovalRequest, ToolApprovalResponse,
)

async def main():
    pending: ToolApprovalRequest | None = None

    # First run: stream until the workflow pauses at the approval gate.
    # The ToolApprovalRequest arrives as a request_info WorkflowEvent
    # (event.type == "request_info", event.data == ToolApprovalRequest).
    async for event in workflow.run({"city": "Seattle"}, stream=True):
        if event.type == "request_info" and isinstance(event.data, ToolApprovalRequest):
            pending = event.data  # workflow has yielded; record the request

    if pending:
        print(f"Approve calling {pending.function_name!r}?")
        print(f"  Args: {pending.arguments}")
        approved = input("y/n: ").lower() == "y"
        # Resume: there is no workflow.respond() — re-call workflow.run()
        # with responses keyed by the request_id from the event.
        result = await workflow.run(
            responses={
                pending.request_id: ToolApprovalResponse(
                    approved=approved,
                    reason=None if approved else "User declined",
                )
            }
        )
        print(result)

asyncio.run(main())
```

### Handling `ToolInvocationResult`

`ToolInvocationResult` is not sent as a workflow event — it is returned internally by the
executor and its fields are written to the YAML `output` paths. To inspect the outcome in
a downstream action, read from the state paths defined in `output`:

```yaml
actions:
  - kind: InvokeFunctionTool
    functionName: get_weather
    arguments:
      location: =Local.city
    output:
      result: Local.weather
      autoSend: false

  - kind: If
    condition: =IsBlank(Local.weather)
    actions:
      - kind: SendActivity
        activity:
          text: "Could not retrieve weather."
    else:
      - kind: SendActivity
        activity:
          text: =Concat("Weather: ", Text(Local.weather))
```

### Dual tool registry lookup

`BaseToolExecutor._get_tool()` checks two sources in order:

1. **Factory-registered tools** (`self._tools`) — passed at `WorkflowFactory` or
   `DeclarativeWorkflowBuilder` construction time. Fastest lookup, available globally.
2. **State-registered tools** (`State[FUNCTION_TOOL_REGISTRY_KEY]`) — a dict in workflow
   state allowing tools to be registered at runtime (useful for dynamic tool sets).

```python
from agent_framework_declarative._workflows._executors_tools import FUNCTION_TOOL_REGISTRY_KEY
# FUNCTION_TOOL_REGISTRY_KEY == "_tool_registry"

async def my_executor_pre_action(ctx):
    # Tools registered here are accessible to any InvokeFunctionToolExecutor
    ctx.state[FUNCTION_TOOL_REGISTRY_KEY] = {
        "dynamic_tool": lambda x: f"processed: {x}",
    }
```

---

## 8 · `JoinExecutor` + `EndWorkflowExecutor` + `EndConversationExecutor` + `CancelDialogExecutor` + `CancelAllDialogsExecutor`

**Sub-package:** `agent_framework_declarative._workflows._executors_control_flow`

These five executors implement the **merge** and **termination** nodes in the declarative
workflow graph. Together with the condition and loop executors from
[Vol. 23 §2–3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/),
they complete the control-flow vocabulary.

### Class signatures and docstrings

```python
class JoinExecutor(DeclarativeActionExecutor):
    """Joins multiple branches back into a single execution path.

    The DeclarativeWorkflowBuilder inserts a JoinExecutor as a merge point after
    every If/ConditionGroup block — one for the 'then' path and one for the 'else'
    path both converge here. Also used as a passthrough for empty else/default
    branches.
    """
    @handler
    async def handle_action(self, trigger, ctx: WorkflowContext[ActionComplete]) -> None:
        await self._ensure_state_initialized(ctx, trigger)
        await ctx.send_message(ActionComplete())

class EndWorkflowExecutor(DeclarativeActionExecutor):
    """Terminates the workflow (maps to EndWorkflow / EndDialog YAML kind).

    Does NOT send ActionComplete — withholding the message causes the graph
    executor to reach a dead end and the workflow finishes cleanly.
    """
    @handler
    async def handle_action(self, trigger, ctx: WorkflowContext[ActionComplete]) -> None:
        pass  # deliberate no-op — no ActionComplete sent

class EndConversationExecutor(DeclarativeActionExecutor):
    """Signals end-of-conversation (maps to EndConversation YAML kind)."""
    @handler
    async def handle_action(self, trigger, ctx: WorkflowContext[ActionComplete]) -> None:
        pass  # same dead-end pattern as EndWorkflowExecutor

class CancelDialogExecutor(DeclarativeActionExecutor):
    """Cancels the current dialog/workflow (maps to CancelDialog kind).

    Semantically distinct from EndWorkflow: termination is due to cancellation,
    not natural completion.
    """
    @handler
    async def handle_action(self, trigger, ctx: WorkflowContext[ActionComplete]) -> None:
        pass

class CancelAllDialogsExecutor(DeclarativeActionExecutor):
    """Cancels all dialogs in the execution stack (maps to CancelAllDialogs kind)."""
    @handler
    async def handle_action(self, trigger, ctx: WorkflowContext[ActionComplete]) -> None:
        pass
```

### The "dead-end" termination pattern

`EndWorkflowExecutor`, `EndConversationExecutor`, `CancelDialogExecutor`, and
`CancelAllDialogsExecutor` all stop execution by *not* sending a message. In the
workflow graph, no outgoing edges fire, the executor pool drains, and the `Workflow`
object returns its final state.

`JoinExecutor` is the opposite: it always sends `ActionComplete()` so execution
continues to the downstream node. The builder creates a join node after every `If` block.

### YAML — conditional with explicit termination

```yaml
actions:
  - kind: If
    condition: =Local.userConfirmed
    actions:
      - kind: InvokeAzureAgent
        agent: ExecutorAgent
        input: =Local.plan
        resultProperty: Workflow.Outputs.result
      - kind: EndWorkflow          # natural completion
    else:
      - kind: SendActivity
        activity:
          text: "Execution cancelled."
      - kind: CancelDialog         # cancellation path
```

### YAML — multi-branch condition with join

```yaml
actions:
  - kind: ConditionGroup
    conditions:
      - condition: =Local.score >= 90
        actions:
          - kind: SetValue
            path: Local.grade
            value: A
          # JoinExecutor inserted automatically by the builder after this branch
      - condition: =Local.score >= 60
        actions:
          - kind: SetValue
            path: Local.grade
            value: B
      - condition: true             # else branch
        actions:
          - kind: SetValue
            path: Local.grade
            value: F
          - kind: EndConversation   # terminate on fail grade

  # Execution continues here only when score >= 60
  - kind: SendActivity
    activity:
      text: =Concat("Your grade: ", Local.grade)
```

### Programmatic construction

```python
from agent_framework_declarative._workflows._executors_control_flow import (
    JoinExecutor, EndWorkflowExecutor, CancelDialogExecutor,
)

join = JoinExecutor({"kind": "Join", "id": "merge_point"}, id="merge_point")
end  = EndWorkflowExecutor({"kind": "EndWorkflow", "id": "done"}, id="done")
cancel = CancelDialogExecutor({"kind": "CancelDialog", "id": "cancel"}, id="cancel")

# In practice DeclarativeWorkflowBuilder handles construction and graph-wiring.
# Direct instantiation is useful in unit tests for executor nodes in isolation.
print(join.id, end.id, cancel.id)  # merge_point  done  cancel
```

---

## 9 · `ActionComplete` + `ActionTrigger` + `DeclarativeStateData` + `ConversationData`

**Sub-package:** `agent_framework_declarative._workflows._declarative_base`

These four types are the typed **inter-executor message contracts** and the **state schema**
that underpin the entire declarative execution layer. They are rarely used directly by
application code but are essential for writing custom executors or understanding how the
graph passes control.

### Class signatures

```python
from agent_framework_declarative._workflows._declarative_base import (
    ActionComplete, ActionTrigger, DeclarativeStateData, ConversationData,
)

class ActionComplete:
    """Sent by an executor when it finishes, allowing downstream nodes to fire."""
    def __init__(self, result: Any = None) -> None:
        self.result = result   # optional return value for the next executor

class ActionTrigger:
    """Carries data into an executor from the previous node in the graph."""
    def __init__(self, data: Any = None) -> None:
        self.data = data

class ConversationData(TypedDict):
    messages: list[Any]   # active conversation messages for the current interaction
    history: list[Any]    # deprecated — use messages; kept for backwards compat

class DeclarativeStateData(TypedDict, total=False):
    Inputs: dict[str, Any]        # read-only workflow inputs
    Outputs: dict[str, Any]       # values returned from the workflow
    Local: dict[str, Any]         # variables for the current turn
    System: dict[str, Any]        # ConversationId, LastMessage, conversations
    Agent: dict[str, Any]         # last agent invocation result
    Conversation: ConversationData  # message history
    Custom: dict[str, Any]        # user-defined variables
    _declarative_loop_state: dict[str, Any]  # internal foreach bookkeeping
```

### `ActionComplete` — the inter-executor handshake

Every executor that wants the workflow to continue sends `ActionComplete()` via
`ctx.send_message(ActionComplete())`. Executors that *terminate* the workflow simply
do not send it (see `EndWorkflowExecutor` in §8). The `result` field can carry data
to the next node but is rarely used since state is the canonical data store.

```python
from agent_framework_declarative._workflows._declarative_base import (
    DeclarativeActionExecutor, ActionComplete
)

class MyCustomExecutor(DeclarativeActionExecutor):
    @handler
    async def handle_action(self, trigger, ctx):
        state = await self._ensure_state_initialized(ctx, trigger)
        # Do work…
        state.set("Local.processed", True)
        # Signal completion to allow downstream executors to run
        await ctx.send_message(ActionComplete(result="done"))
```

### `ActionTrigger` — passing data into executors

`ActionTrigger` is typically used in tests or specialised sub-workflow patterns to pass
data directly into an executor, bypassing the normal state-based communication.

```python
from agent_framework_declarative._workflows._declarative_base import ActionTrigger

trigger = ActionTrigger(data={"filePath": "/tmp/input.txt"})
# In a unit test scenario: invoke executor.handle_action(trigger, mock_ctx)
print(trigger.data)  # {"filePath": "/tmp/input.txt"}
```

### `DeclarativeStateData` — the 8-namespace schema

`DeclarativeStateData` is the `TypedDict` that defines what lives inside the workflow's
`State` object. `DeclarativeWorkflowState.get_state_data()` returns it; `set_state_data()`
writes it back.

```python
from agent_framework._workflows._state import State
from agent_framework_declarative._workflows._declarative_base import (
    DeclarativeEnvConfig, DeclarativeStateData, DeclarativeWorkflowState
)

state = State()
wf_state = DeclarativeWorkflowState(state, env_config=DeclarativeEnvConfig())
wf_state.initialize(inputs={"topic": "AI"})

# Read the raw TypedDict
data: DeclarativeStateData = wf_state.get_state_data()
print(data.get("Inputs"))   # {"topic": "AI"}
print(data.get("Local"))    # {}

# Mutate via the TypedDict directly (then write back)
data["Local"]["counter"] = 0
wf_state.set_state_data(data)

# Or use the dot-path API (preferred)
wf_state.set("Local.counter", 1)
print(wf_state.get("Local.counter"))  # 1
```

### `ConversationData` — messages vs history

`ConversationData` has two fields. `messages` is the canonical list used by
`InvokeAzureAgentExecutor` and `CreateConversationExecutor`. `history` is deprecated —
it was previously a separate buffer but both fields are now kept in sync. New code
should read and write only `messages`.

```python
from agent_framework_declarative._workflows._declarative_base import ConversationData

# Inspect conversation state after an InvokeAzureAgent action
conv: ConversationData = wf_state.get("Conversation")
if conv:
    print(f"Turn count: {len(conv['messages'])}")
    # history is kept for backwards compatibility — ignore in new code
    assert conv["messages"] == conv["history"] or conv["history"] == []
```

### Namespace cheat sheet

| Namespace | Set by | Read by | Notes |
|---|---|---|---|
| `Inputs` | `DeclarativeWorkflowState.initialize()` | Any executor | Read-only at runtime |
| `Outputs` | SetValue / executor | Workflow caller | Returned when workflow ends |
| `Local` | Any executor | Any executor | `ClearAllVariablesExecutor` wipes this |
| `System` | Framework | Any executor | `ConversationId`, `LastMessage`, `conversations` |
| `Agent` | `InvokeAzureAgentExecutor` | Downstream | `Agent.response`, `Agent.messages` |
| `Conversation` | `CreateConversation` / agents | Any | `ConversationData` — use `messages` field |
| `Custom` | Any executor | Any executor | Arbitrary user-defined scope |
| `_declarative_loop_state` | `ForeachInitExecutor` | `ForeachNextExecutor` | Internal; do not modify directly |

---

## 10 · `ClearAllVariablesExecutor` + `EditTableExecutor` + `ResetVariableExecutor` + `SetTextVariableExecutor`

**Sub-package:** `agent_framework_declarative._workflows._executors_basic`

Four additional basic executors not covered in [Vol. 23 §4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/)
(which covered `SetValueExecutor`, `SetVariableExecutor`, `CreateConversationExecutor`,
`SetMultipleVariablesExecutor`, `SendActivityExecutor`, and `ParseValueExecutor`). These
four handle the remaining YAML variable-manipulation actions: bulk-clearing the `Local`
scope, table operations, null-reset, and coerced string assignment.

### Class signatures and behaviour

```python
class ClearAllVariablesExecutor(DeclarativeActionExecutor):
    """kind: ClearAllVariables — wipes the entire Local scope.

    Reads DeclarativeStateData from state, sets Local = {}, and writes back.
    All other namespaces (Inputs, Outputs, System, Agent, Conversation) are unaffected.
    """

class EditTableExecutor(DeclarativeActionExecutor):
    """kind: EditTable — add / insert / remove / clear items in a list variable.

    Fields:
        table:      state path to the list (supports _get_variable_path() resolution)
        operation:  "add" | "insert" | "remove" | "clear"
        value:      item to add/remove (PowerFx-evaluated if starts with "=")
        index:      position for insert (PowerFx-evaluated; appends if absent)

    If the variable at `table` is None, a new empty list is created.
    If it is a non-list scalar, it is wrapped in a list before the operation.
    """

class ResetVariableExecutor(DeclarativeActionExecutor):
    """kind: ResetVariable — sets the variable at `variable` or `path` to None."""

class SetTextVariableExecutor(DeclarativeActionExecutor):
    """kind: SetTextVariable — evaluates `text`, coerces to str, and stores at `variable`.

    Equivalent to SetValueExecutor but always casts the result to str(). Useful when
    you need to guarantee the stored value is a string, e.g. for Concat expressions.
    """
```

### `ClearAllVariablesExecutor` — wipe the Local scope

```yaml
actions:
  # Reset all Local variables at the start of each loop iteration
  - kind: Foreach
    source: =Workflow.Inputs.batches
    itemName: batch
    actions:
      - kind: ClearAllVariables   # wipes Local.* before processing each batch
      - kind: SetValue
        path: Local.batchData
        value: =Local.batch
      - kind: InvokeAzureAgent
        agent: ProcessorAgent
        input: =Local.batchData
        resultProperty: Workflow.Outputs.lastResult
```

```python
from agent_framework_declarative._workflows._executors_basic import ClearAllVariablesExecutor

executor = ClearAllVariablesExecutor(
    action_def={"kind": "ClearAllVariables", "id": "resetLocals"},
    id="resetLocals",
)
# At runtime: state_data["Local"] = {}; other namespaces unchanged.
print(executor.id)  # resetLocals
```

### `EditTableExecutor` — list CRUD operations

```yaml
actions:
  # Initialise empty list
  - kind: SetValue
    path: Local.results
    value: []

  # Append an item
  - kind: EditTable
    table: Local.results
    operation: add
    value: =Local.agentOutput

  # Insert at position 0
  - kind: EditTable
    table: Local.results
    operation: insert
    value: =Local.priorityItem
    index: 0

  # Remove a specific item
  - kind: EditTable
    table: Local.results
    operation: remove
    value: =Local.itemToDelete

  # Clear the entire list
  - kind: EditTable
    table: Local.results
    operation: clear
```

```python
from agent_framework_declarative._workflows._executors_basic import EditTableExecutor

# Append
append_exec = EditTableExecutor(
    action_def={
        "kind": "EditTable",
        "id": "appendResult",
        "table": "Local.results",
        "operation": "add",
        "value": "=Local.agentOutput",
    },
    id="appendResult",
)

# Insert at index
insert_exec = EditTableExecutor(
    action_def={
        "kind": "EditTable",
        "id": "insertFirst",
        "table": "Local.results",
        "operation": "insert",
        "value": "=Local.priorityItem",
        "index": 0,
    },
    id="insertFirst",
)
```

### `ResetVariableExecutor` — null-reset a variable

```yaml
actions:
  - kind: ResetVariable
    variable: Local.tempBuffer   # sets Local.tempBuffer to None
```

```python
from agent_framework_declarative._workflows._executors_basic import ResetVariableExecutor

executor = ResetVariableExecutor(
    action_def={"kind": "ResetVariable", "id": "clearBuf",
                "variable": "Local.tempBuffer"},
    id="clearBuf",
)
# At runtime: state.set("Local.tempBuffer", None)
```

### `SetTextVariableExecutor` — coerced string assignment

```yaml
actions:
  - kind: SetTextVariable
    variable: Local.displayCount
    text: =Text(Local.count, "[$-en-US]0")   # always stored as str
```

```python
from agent_framework_declarative._workflows._executors_basic import SetTextVariableExecutor

executor = SetTextVariableExecutor(
    action_def={
        "kind": "SetTextVariable",
        "id": "fmtCount",
        "variable": "Local.displayCount",
        "text": "=Text(Local.count)",
    },
    id="fmtCount",
)
# At runtime: state.set(path, str(evaluated_text))
# Even if Local.count is an int, Local.displayCount is always a str.
```

### End-to-end: accumulating results with `EditTable`

```python
import asyncio
from agent_framework_openai import OpenAIChatClient
from agent_framework.declarative import WorkflowFactory

yaml_content = """
name: BatchSummariser
actions:
  - kind: SetValue
    path: Workflow.Outputs.summaries
    value: []

  - kind: Foreach
    source: =Workflow.Inputs.documents
    itemName: doc
    indexName: i
    actions:
      # Foreach binds Local.doc and Local.i before entering the body.
      # Do NOT use ClearAllVariables here — it would wipe those loop variables.
      # Use ResetVariable to clear only specific intermediates from prior iterations.
      - kind: ResetVariable
        path: Local.summary

      - kind: InvokeAzureAgent
        agent: SummaryAgent
        input: =Concat("Summarise in one sentence: ", Local.doc)
        resultProperty: Local.summary

      - kind: EditTable
        table: Workflow.Outputs.summaries
        operation: add
        value: =Local.summary

  - kind: SendActivity
    activity:
      text: =Concat("Processed ", Text(Local.i + 1), " documents.")
"""

client = OpenAIChatClient()
agent = client.as_agent(
    name="SummaryAgent",
    instructions="Summarise the text provided in exactly one sentence.",
)

factory = WorkflowFactory(agents={"SummaryAgent": agent})
workflow = factory.create_workflow_from_yaml(yaml_content)

async def main():
    documents = [
        "The Eiffel Tower was completed in 1889.",
        "Python was created by Guido van Rossum.",
        "The speed of light is approximately 299,792 km/s.",
    ]
    result = await workflow.run({"documents": documents})
    print(result)

asyncio.run(main())
```

---

## Summary

This volume filled **ten previously uncovered class groups** from the declarative
workflow sub-system, complementing the foundation laid in [Vol. 23](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v23/):

- **Tool variety beyond MCP** — `FunctionTool`, `OpenApiTool`, `WebSearchTool`, `FileSearchTool`,
  and `CodeInterpreterTool` each serve distinct integration patterns; `Binding.name`
  selects a registered Python callable from `AgentFactory(bindings=…)` (the `input`
  field is a data-model artifact that `AgentFactory` does not evaluate at call time).
- **Factory layer** — `AgentFactory` and `WorkflowFactory` are the entry points that
  convert YAML into live `Agent` and `Workflow` instances; `ProviderTypeMapping` lets you
  extend the built-in provider table with custom client adapters.
- **Human-in-the-loop** — `QuestionExecutor` / `RequestExternalInputExecutor` pause the
  graph and yield `ExternalInputRequest` as a `request_info` `WorkflowEvent`; callers
  resume by calling `workflow.run(responses={request_id: ExternalInputResponse(…)})` —
  there is no `workflow.respond()` method. The same yield/resume pattern applies to
  `InvokeMcpToolActionExecutor` (via `MCPToolApprovalRequest`) and `BaseToolExecutor`
  (via `ToolApprovalRequest`).
- **Termination nodes** — `EndWorkflowExecutor`, `EndConversationExecutor`,
  `CancelDialogExecutor`, and `CancelAllDialogsExecutor` stop execution by deliberately
  not sending `ActionComplete`; `JoinExecutor` does the opposite — it always continues.
- **Typed contracts** — `ActionComplete`, `ActionTrigger`, `DeclarativeStateData`, and
  `ConversationData` are the inter-executor message types and state schema; understanding
  them is essential for writing custom `DeclarativeActionExecutor` subclasses.
- **Table and scope management** — `EditTableExecutor` provides list CRUD operations
  (`add` / `insert` / `remove` / `clear`) on any state path; `ClearAllVariablesExecutor`
  wipes the `Local` scope in one step; `ResetVariableExecutor` null-resets a single
  variable; `SetTextVariableExecutor` stores coerced strings.

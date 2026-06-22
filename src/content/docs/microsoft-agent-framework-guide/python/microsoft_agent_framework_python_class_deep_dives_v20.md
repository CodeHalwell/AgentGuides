---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 20"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.9.0: capability-check Protocols (SupportsCodeInterpreterTool / SupportsWebSearchTool / SupportsImageGenerationTool / SupportsMCPTool / SupportsFileSearchTool / SupportsShellTool / SupportsGetEmbeddings), feature-staging system (ReleaseCandidateFeature / FeatureStageWarning / ExperimentalWarning + formatter install), embedding DTOs (EmbeddingGenerationOptions / Embedding / GeneratedEmbeddings), WorkflowEventSource enum, SubWorkflowRequestMessage + SubWorkflowResponseMessage (hierarchical workflow HITL bridge), RequestInfoMixin + response_handler decorator (executor-level request/response wiring), WorkflowAgent.RequestInfoFunctionArgs (request-info serialization dataclass), EdgeGroupDeliveryStatus (OTel observability enum), IntegrityLabel + LabelTrackingFunctionMiddleware (IFC security — label tracking, variable indirection, auto-hide), MiddlewareTermination + WorkflowConvergenceException (control-flow + convergence exceptions)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 43
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 20

Verified against **agent-framework 1.9.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework._clients`,
`agent_framework._feature_stage`, `agent_framework._types`,
`agent_framework._workflows._events`, `agent_framework._workflows._workflow_executor`,
`agent_framework._workflows._request_info_mixin`, `agent_framework._workflows._agent`,
`agent_framework.observability`, `agent_framework.security`,
`agent_framework._middleware`, `agent_framework.exceptions`.

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

This volume covers **ten class groups** focussed on runtime capability-check protocols,
the embedding DTO layer, the sub-workflow HITL bridge, the request/response handler wiring
mixin, observability and security enums, the label-tracking security middleware, and the
control-flow + convergence exception pair:

| # | Class / group | Sub-package |
|---|---|---|
| 1 | `SupportsCodeInterpreterTool` · `SupportsWebSearchTool` · `SupportsImageGenerationTool` · `SupportsMCPTool` · `SupportsFileSearchTool` · `SupportsShellTool` | `agent_framework._clients` |
| 2 | `SupportsGetEmbeddings` | `agent_framework._clients` |
| 3 | `ReleaseCandidateFeature` · `FeatureStageWarning` · `ExperimentalWarning` | `agent_framework._feature_stage` |
| 4 | `EmbeddingGenerationOptions` · `Embedding` · `GeneratedEmbeddings` | `agent_framework._types` |
| 5 | `WorkflowEventSource` | `agent_framework._workflows._events` |
| 6 | `SubWorkflowRequestMessage` · `SubWorkflowResponseMessage` | `agent_framework._workflows._workflow_executor` |
| 7 | `RequestInfoMixin` · `response_handler` decorator | `agent_framework._workflows._request_info_mixin` |
| 8 | `WorkflowAgent.RequestInfoFunctionArgs` | `agent_framework._workflows._agent` |
| 9 | `EdgeGroupDeliveryStatus` | `agent_framework.observability` |
| 10 | `IntegrityLabel` · `LabelTrackingFunctionMiddleware` | `agent_framework.security` |
| 11 | `MiddlewareTermination` · `WorkflowConvergenceException` | `agent_framework._middleware` · `agent_framework.exceptions` |

---

## 1 · Hosted-tool capability Protocols

**Sub-package:** `agent_framework._clients`  
**Import:** `from agent_framework import SupportsCodeInterpreterTool` *(and the other six)*

These six `@runtime_checkable Protocol` classes let you guard tool-factory calls at
runtime without importing provider-specific modules.  All six share the same shape: one
`@staticmethod` that forwards `**kwargs` to the underlying provider and returns a ready-to-use
tool configuration.

### Class signatures (1.9.0)

```python
from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class SupportsCodeInterpreterTool(Protocol):
    @staticmethod
    def get_code_interpreter_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsWebSearchTool(Protocol):
    @staticmethod
    def get_web_search_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsImageGenerationTool(Protocol):
    @staticmethod
    def get_image_generation_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsMCPTool(Protocol):
    @staticmethod
    def get_mcp_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsFileSearchTool(Protocol):
    @staticmethod
    def get_file_search_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsShellTool(Protocol):
    @staticmethod
    def get_shell_tool(**kwargs: Any) -> Any: ...
```

### Key facts

| Protocol | `isinstance` guard | Factory method | Notes |
|---|---|---|---|
| `SupportsCodeInterpreterTool` | ✓ | `get_code_interpreter_tool(**kwargs)` | Foundry/OpenAI hosted sandbox |
| `SupportsWebSearchTool` | ✓ | `get_web_search_tool(**kwargs)` | Bing grounding, OpenAI search |
| `SupportsImageGenerationTool` | ✓ | `get_image_generation_tool(**kwargs)` | DALL-E, Foundry image gen |
| `SupportsMCPTool` | ✓ | `get_mcp_tool(**kwargs)` | Hosted MCP; pass `name=` and `url=` |
| `SupportsFileSearchTool` | ✓ | `get_file_search_tool(**kwargs)` | Requires `vector_store_ids=[...]` |
| `SupportsShellTool` | ✓ | `get_shell_tool(**kwargs)` | Pass `func=shell.as_function()` |

### Example 1 — provider-agnostic tool attachment

```python
from agent_framework import Agent
from agent_framework import (
    SupportsCodeInterpreterTool,
    SupportsWebSearchTool,
    SupportsFileSearchTool,
)


def build_agent(client, vector_store_ids: list[str]) -> Agent:
    """Attach every tool the client supports at runtime."""
    tools = []

    if isinstance(client, SupportsCodeInterpreterTool):
        tools.append(client.get_code_interpreter_tool())

    if isinstance(client, SupportsWebSearchTool):
        tools.append(client.get_web_search_tool())

    if isinstance(client, SupportsFileSearchTool):
        tools.append(client.get_file_search_tool(vector_store_ids=vector_store_ids))

    return Agent(client=client, name="multi-tool-agent", tools=tools)
```

### Example 2 — MCP hosted tool with name and URL

```python
from agent_framework import Agent, SupportsMCPTool
from agent_framework.foundry import FoundryChatClient

client = FoundryChatClient(model="gpt-4o")

if isinstance(client, SupportsMCPTool):
    mcp_tool = client.get_mcp_tool(
        name="my_server",
        url="https://my-mcp-server.example.com/mcp",
    )
    agent = Agent(client=client, name="mcp-agent", tools=[mcp_tool])
```

### Example 3 — image generation guard

```python
from agent_framework import Agent, SupportsImageGenerationTool
from agent_framework.openai import OpenAIChatClient


async def run_creative_agent(prompt: str) -> str:
    client = OpenAIChatClient(model="gpt-4o")
    tools = []
    if isinstance(client, SupportsImageGenerationTool):
        tools.append(client.get_image_generation_tool())

    agent = Agent(client=client, name="creative", tools=tools)
    response = await agent.run(prompt)
    return response.messages[-1].text
```

### Example 4 — writing a custom client that satisfies multiple protocols

```python
from agent_framework._clients import BaseChatClient
from agent_framework import (
    SupportsWebSearchTool,
    SupportsShellTool,
    FunctionTool,
    tool,
)


@tool
async def web_search(query: str) -> str:
    return f"[mock results for: {query}]"


@tool
async def run_shell(command: str) -> str:
    return f"[mock output of: {command}]"


class MyCustomClient(BaseChatClient, SupportsWebSearchTool, SupportsShellTool):
    @staticmethod
    def get_web_search_tool(**kwargs) -> FunctionTool:
        return web_search

    @staticmethod
    def get_shell_tool(**kwargs) -> FunctionTool:
        return run_shell

    async def _inner_get_response(self, messages, options, *, stream):
        ...  # your model call


assert isinstance(MyCustomClient(), SupportsWebSearchTool)
assert isinstance(MyCustomClient(), SupportsShellTool)
```

---

## 2 · `SupportsGetEmbeddings`

**Sub-package:** `agent_framework._clients`  
**Import:** `from agent_framework import SupportsGetEmbeddings`

`SupportsGetEmbeddings` is the duck-typing Protocol for embedding generation. Unlike the
hosted-tool protocols it carries a real async method with typed parameters, making it
suitable for dependency injection and type-checked callers.

### Class signature (1.9.0)

```python
from collections.abc import Sequence, Awaitable
from typing import Protocol, runtime_checkable, Any
from typing_extensions import TypeVar
from agent_framework._types import GeneratedEmbeddings

EmbeddingInputContraT = TypeVar("EmbeddingInputContraT", contravariant=True, default="str")
EmbeddingT = TypeVar("EmbeddingT", default="list[float]")
EmbeddingProtocolOptionsT = TypeVar("EmbeddingProtocolOptionsT", default="EmbeddingGenerationOptions")

@runtime_checkable
class SupportsGetEmbeddings(Protocol[EmbeddingInputContraT, EmbeddingT, EmbeddingProtocolOptionsT]):
    additional_properties: dict[str, Any]

    def get_embeddings(
        self,
        values: Sequence[EmbeddingInputContraT],
        *,
        options: EmbeddingProtocolOptionsT | None = None,
    ) -> Awaitable[GeneratedEmbeddings[EmbeddingT, EmbeddingProtocolOptionsT]]: ...
```

### Key facts

- `@runtime_checkable` — safe to use with `isinstance()`.
- All `BaseEmbeddingClient` subclasses (`OpenAIEmbeddingClient`, `BedrockEmbeddingClient`,
  `FoundryEmbeddingClient`, etc.) satisfy this protocol automatically.
- `additional_properties` is part of the protocol surface (required attribute) — it holds
  provider-specific metadata from `SerializationMixin`.
- Generic parameters are optional; `SupportsGetEmbeddings` (no brackets) means
  `SupportsGetEmbeddings[str, list[float], EmbeddingGenerationOptions]`.

### Example 1 — inject any embedding client

```python
from agent_framework import SupportsGetEmbeddings


async def embed_chunks(
    client: SupportsGetEmbeddings,
    chunks: list[str],
) -> list[list[float]]:
    result = await client.get_embeddings(chunks)
    return [emb.vector for emb in result]
```

### Example 2 — runtime guard before embedding

```python
from agent_framework import SupportsGetEmbeddings
from agent_framework.openai import OpenAIChatClient


async def maybe_embed(client, text: str) -> list[float] | None:
    if not isinstance(client, SupportsGetEmbeddings):
        return None
    result = await client.get_embeddings([text])
    return result[0].vector
```

### Example 3 — custom embedding client satisfying the protocol

```python
from agent_framework._clients import BaseEmbeddingClient, EmbeddingGenerationOptions
from agent_framework._types import Embedding, GeneratedEmbeddings


class MockEmbeddingClient(BaseEmbeddingClient):
    async def get_embeddings(
        self,
        values: list[str],
        *,
        options: EmbeddingGenerationOptions | None = None,
    ) -> GeneratedEmbeddings:
        dim = (options or {}).get("dimensions", 4)
        embeddings = [Embedding(vector=[0.1] * dim, model="mock") for _ in values]
        return GeneratedEmbeddings(embeddings, usage={"prompt_tokens": len(values)})


from agent_framework import SupportsGetEmbeddings
assert isinstance(MockEmbeddingClient(), SupportsGetEmbeddings)
```

---

## 3 · Feature-staging system — `ReleaseCandidateFeature` · `FeatureStageWarning` · `ExperimentalWarning`

**Sub-package:** `agent_framework._feature_stage`  
**Import:** `from agent_framework._feature_stage import ReleaseCandidateFeature, FeatureStageWarning, ExperimentalWarning`

Vol. 6 covered `ExperimentalFeature` and the `@experimental` decorator. This volume
documents the complementary staging primitives: the `ReleaseCandidateFeature` enum (for
near-GA APIs), the warning base class `FeatureStageWarning`, and `ExperimentalWarning`.

### Class signatures (1.9.0)

```python
from enum import Enum

class ReleaseCandidateFeature(str, Enum):
    """Current release-candidate feature IDs (inventory; members may move as features GA)."""
    # (empty in 1.9.0 — all RC features have graduated to stable)

class FeatureStageWarning(FutureWarning):
    """Base warning category for staged APIs."""

class ExperimentalWarning(FeatureStageWarning):
    """Warning emitted when an experimental API is first used in a session."""
```

### Key facts

- `ExperimentalWarning` is a subclass of `FeatureStageWarning`, which is a subclass of
  `FutureWarning`.  Filtering `FutureWarning` will also suppress both.
- The framework installs a **single-line formatter** at import time
  (`_install_feature_stage_formatter`) that prints
  `filename:lineno: ExperimentalWarning: message` instead of the verbose two-line stdlib
  default.  This applies only to `FeatureStageWarning` subclasses.
- Warnings are **deduplicated**: the set `_WARNED_FEATURES` tracks `(category, feature_id)`
  pairs so each combination fires at most once per interpreter session.
- `ReleaseCandidateFeature` is empty in 1.9.0 because all RC features have graduated to
  stable.  Adding a member to it and decorating with `@release_candidate(feature_id=...)` 
  is the intended extension path.

### Example 1 — filtering experimental warnings in tests

```python
import warnings
from agent_framework._feature_stage import ExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ExperimentalWarning)
    from agent_framework._harness._memory import MemoryContextProvider  # experimental
    ctx = MemoryContextProvider()
```

### Example 2 — turning ExperimentalWarning into errors in CI

```python
# conftest.py
import warnings
from agent_framework._feature_stage import ExperimentalWarning

def pytest_configure(config):
    warnings.filterwarnings("error", category=ExperimentalWarning)
```

### Example 3 — authoring a release-candidate API using the staging decorator

```python
from agent_framework._feature_stage import ReleaseCandidateFeature, release_candidate
from enum import auto


# Step 1: add the feature to the enum (hypothetical future feature)
# class ReleaseCandidateFeature(str, Enum):
#     MY_NEW_FEATURE = "MY_NEW_FEATURE"


# Step 2: decorate your class (shown with ExperimentalFeature for a real example)
from agent_framework._feature_stage import ExperimentalFeature, experimental

@experimental(feature_id=ExperimentalFeature.HARNESS)
class MyExperimentalProvider:
    """Experimental provider — subject to change."""

    def __init__(self, config: dict) -> None:
        self.config = config
```

### Example 4 — detecting the feature stage of an object at runtime

```python
from agent_framework._feature_stage import _FEATURE_STAGE_ATTR, _FEATURE_ID_ATTR
from agent_framework._harness._memory import MemoryContextProvider


stage = getattr(MemoryContextProvider, _FEATURE_STAGE_ATTR, None)
feature_id = getattr(MemoryContextProvider, _FEATURE_ID_ATTR, None)
print(stage, feature_id)  # "experimental"  "HARNESS"
```

---

## 4 · Embedding DTOs — `EmbeddingGenerationOptions` · `Embedding` · `GeneratedEmbeddings`

**Sub-package:** `agent_framework._types`  
**Import:** `from agent_framework import EmbeddingGenerationOptions, Embedding, GeneratedEmbeddings`

These three types form the embedding data-transfer layer used by every embedding client in
the framework.

### Class signatures (1.9.0)

```python
from typing import TypedDict, Generic, Any
from typing_extensions import TypeVar
from datetime import datetime
from collections.abc import Iterable

EmbeddingT = TypeVar("EmbeddingT", default="list[float]")
EmbeddingOptionsT = TypeVar("EmbeddingOptionsT", default="EmbeddingGenerationOptions")

class EmbeddingGenerationOptions(TypedDict, total=False):
    """Common request settings for embedding generation (all fields optional)."""
    model: str       # e.g. "text-embedding-3-small"
    dimensions: int  # explicit dimension count requested from the provider

class Embedding(Generic[EmbeddingT]):
    def __init__(
        self,
        vector: EmbeddingT,
        *,
        model: str | None = None,
        dimensions: int | None = None,       # computed from len(vector) if omitted
        created_at: datetime | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None: ...
    vector: EmbeddingT
    model: str | None
    dimensions: int
    created_at: datetime | None
    additional_properties: dict[str, Any] | None

class GeneratedEmbeddings(list[Embedding[EmbeddingT]], Generic[EmbeddingT, EmbeddingOptionsT]):
    def __init__(
        self,
        embeddings: Iterable[Embedding[EmbeddingT]] | None = None,
        *,
        options: EmbeddingOptionsT | None = None,
        usage: UsageDetails | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None: ...
    options: EmbeddingOptionsT | None
    usage: UsageDetails | None
    additional_properties: dict[str, Any] | None
```

### Key facts

| Type | Extends | Key fields | Notes |
|---|---|---|---|
| `EmbeddingGenerationOptions` | `TypedDict` | `model`, `dimensions` | All fields optional; provider TypedDicts extend this |
| `Embedding[T]` | `Generic[T]` | `vector`, `model`, `dimensions`, `created_at` | `dimensions` auto-computed from `len(vector)` |
| `GeneratedEmbeddings[T, O]` | `list[Embedding[T]]` | `options`, `usage`, `additional_properties` | Direct iteration and indexing over embeddings |

### Example 1 — basic embedding pipeline

```python
from agent_framework import EmbeddingGenerationOptions
from agent_framework.openai import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient()

opts: EmbeddingGenerationOptions = {
    "model": "text-embedding-3-small",
    "dimensions": 256,
}

async def embed(texts: list[str]):
    result = await client.get_embeddings(texts, options=opts)
    print(f"Got {len(result)} embeddings, dims={result[0].dimensions}")
    print(f"Tokens used: {result.usage.get('prompt_tokens', 'n/a')}")
    return [emb.vector for emb in result]
```

### Example 2 — cosine similarity using GeneratedEmbeddings

```python
import math
from agent_framework import Embedding


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    dot = sum(x * y for x, y in zip(a.vector, b.vector))
    norm_a = math.sqrt(sum(x ** 2 for x in a.vector))
    norm_b = math.sqrt(sum(x ** 2 for x in b.vector))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def most_similar(query: str, corpus: list[str], client) -> str:
    from agent_framework import GeneratedEmbeddings
    combined = await client.get_embeddings([query] + corpus)
    q_emb = combined[0]
    scores = [cosine_similarity(q_emb, combined[i + 1]) for i in range(len(corpus))]
    return corpus[scores.index(max(scores))]
```

### Example 3 — accessing usage and options from a result

```python
from agent_framework.openai import OpenAIEmbeddingClient

async def embed_with_audit(texts: list[str]) -> None:
    client = OpenAIEmbeddingClient()
    result = await client.get_embeddings(texts)

    print("Model used:", result[0].model)
    print("Prompt tokens:", result.usage.get("prompt_tokens") if result.usage else "n/a")
    print("Dimensions:", result[0].dimensions)
    print("First 5 values:", result[0].vector[:5])
```

### Example 4 — typed int8 embeddings (binary quantization)

```python
from agent_framework import Embedding, GeneratedEmbeddings

# Binary/int8 embeddings returned by some providers
int8_emb: Embedding[list[int]] = Embedding(vector=[127, -42, 0, 99], model="matryoshka")
result: GeneratedEmbeddings[list[int]] = GeneratedEmbeddings(
    [int8_emb],
    usage={"prompt_tokens": 1, "total_tokens": 1},
)
assert result[0].dimensions == 4
```

---

## 5 · `WorkflowEventSource`

**Sub-package:** `agent_framework._workflows._events`  
**Import:** `from agent_framework._workflows._events import WorkflowEventSource`

`WorkflowEventSource` is a two-member `str` enum that tags every `WorkflowEvent` with its
origin. It travels inside `WorkflowEvent.origin` and is surfaced to OTel spans and
`EdgeGroupDeliveryStatus` log records.

### Class signature (1.9.0)

```python
from enum import Enum

class WorkflowEventSource(str, Enum):
    EXECUTOR  = "EXECUTOR"   # event raised by an Executor (AgentExecutor, FunctionExecutor, WorkflowExecutor, …)
    FRAMEWORK = "FRAMEWORK"  # event raised by the framework itself (routing, fan-in, checkpoint, …)
```

### Key facts

- All `WorkflowEvent` objects carry an `.origin` field typed as `WorkflowEventSource`.
- `EXECUTOR` events originate from user-authored executors; `FRAMEWORK` events come from
  the runner, edge delivery, or the orchestration layer.
- Because it extends `str` you can compare with plain string literals:
  `event.origin == "EXECUTOR"` works without importing the enum.

### Example 1 — filtering events by source in an observer

```python
from agent_framework._workflows._events import WorkflowEvent, WorkflowEventSource


def log_executor_events(event: WorkflowEvent) -> None:
    if event.origin == WorkflowEventSource.EXECUTOR:
        print(f"Executor event: {event.type} from {event.executor_id}")
```

### Example 2 — branching on event source in an EdgeRunner callback

```python
from agent_framework._workflows._events import WorkflowEvent, WorkflowEventSource


async def handle_event(event: WorkflowEvent) -> None:
    match event.origin:
        case WorkflowEventSource.EXECUTOR:
            await route_to_agent(event)
        case WorkflowEventSource.FRAMEWORK:
            await record_system_transition(event)
```

### Example 3 — OTel span attribute from source

```python
from opentelemetry import trace
from agent_framework._workflows._events import WorkflowEventSource

tracer = trace.get_tracer("my_app")


def trace_event(event) -> None:
    with tracer.start_as_current_span("workflow.event") as span:
        span.set_attribute("event.origin", str(event.origin.value))
        span.set_attribute("event.type", event.type)
```

---

## 6 · `SubWorkflowRequestMessage` · `SubWorkflowResponseMessage`

**Sub-package:** `agent_framework._workflows._workflow_executor`  
**Import:** `from agent_framework._workflows._workflow_executor import SubWorkflowRequestMessage, SubWorkflowResponseMessage`

These two dataclasses are the **HITL bridge for hierarchical workflows**. When a
`WorkflowExecutor` (a sub-workflow) needs information from its parent, it emits a
`SubWorkflowRequestMessage`; the parent responds with a `SubWorkflowResponseMessage`
created via `SubWorkflowRequestMessage.create_response()`.

### Class signatures (1.9.0)

```python
from dataclasses import dataclass
from typing import Any
from agent_framework._workflows._events import WorkflowEvent


@dataclass
class SubWorkflowResponseMessage:
    """Message sent from parent → sub-workflow to provide requested information."""
    data: Any             # the response payload
    source_event: WorkflowEvent  # the original WorkflowEvent that triggered the request


@dataclass
class SubWorkflowRequestMessage:
    """Message sent from sub-workflow → parent executor to request information."""
    source_event: WorkflowEvent  # original event raised inside the sub-workflow
    executor_id: str             # ID of the WorkflowExecutor in the parent workflow

    def create_response(self, data: Any) -> SubWorkflowResponseMessage:
        """Wrap response data after type-checking against source_event.response_type."""
        ...
```

### Key facts

- `create_response(data)` validates `isinstance(data, source_event.response_type)` and
  raises `TypeError` if the types don't match — preventing silent protocol errors.
- `executor_id` lets a parent workflow that contains **multiple** `WorkflowExecutor` nodes
  route the response to the correct sub-workflow instance.
- The `WorkflowExecutor` automatically appends `SubWorkflowRequestMessage` to its declared
  output types so the parent sees it on the graph edge.

### Example 1 — parent workflow handling a sub-workflow HITL request

```python
from agent_framework._workflows._workflow_executor import SubWorkflowRequestMessage
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext


class ParentExecutor(Executor):
    @handler
    async def handle_subworkflow_request(
        self,
        request: SubWorkflowRequestMessage,
        ctx: WorkflowContext[None],
    ) -> None:
        # The sub-workflow sent us a request — reply with the right data
        print(f"Sub-workflow {request.executor_id} needs: {request.source_event.type}")
        reply = request.create_response(data={"answer": "Paris"})
        await ctx.send_message(reply)
```

### Example 2 — validating response type before sending

```python
from agent_framework._workflows._workflow_executor import SubWorkflowRequestMessage


def build_reply(request: SubWorkflowRequestMessage, user_answer: str):
    try:
        return request.create_response(data=user_answer)
    except TypeError as e:
        # Response type mismatch — e.g. sub-workflow expected int, got str
        raise ValueError(f"Invalid response type for sub-workflow request: {e}") from e
```

### Example 3 — routing to multiple sub-workflows by executor_id

```python
from agent_framework._workflows._workflow_executor import SubWorkflowRequestMessage


async def dispatch_subworkflow_requests(
    requests: list[SubWorkflowRequestMessage],
    answers: dict[str, str],
) -> list:
    responses = []
    for req in requests:
        answer = answers.get(req.executor_id, "unknown")
        responses.append(req.create_response(data=answer))
    return responses
```

---

## 7 · `RequestInfoMixin` · `response_handler` decorator

**Sub-package:** `agent_framework._workflows._request_info_mixin`  
**Import:** `from agent_framework._workflows._request_info_mixin import RequestInfoMixin, response_handler`

`RequestInfoMixin` provides the request/response infrastructure that `Executor` and
`WorkflowAgent` inherit. `response_handler` is the decorator that registers handler methods
onto any class that uses the mixin.

### Class signature (1.9.0)

```python
class RequestInfoMixin:
    is_request_response_capable: bool  # set to True after _discover_response_handlers()

    def is_request_supported(
        self,
        request_type: type,
        response_type: type,
    ) -> bool: ...
    """Return True if a @response_handler covers this (request_type, response_type) pair."""

    def _find_response_handler(
        self, request: Any, response: Any
    ) -> Callable[..., Awaitable[None]] | None: ...

    def _discover_response_handlers(self) -> None: ...
    """Scan the class at init time and register all @response_handler methods."""
```

#### `response_handler` decorator

```python
from agent_framework._workflows._request_info_mixin import response_handler

# Mode 1 — type introspection (annotations on parameters)
@response_handler
async def handle(self, original_request: MyRequest, response: MyResponse, ctx: WorkflowContext[Out]) -> None:
    ...

# Mode 2 — explicit types (all annotations disabled on the function)
@response_handler(request=MyRequest, response=MyResponse, output=int)
async def handle(self, original_request, response, ctx):
    ...
```

### Key facts

- Handlers are discovered lazily via `_discover_response_handlers()` which is called during
  `Executor.__init__`.
- A duplicate `(request_type, response_type)` pair raises `ValueError` immediately at init.
- `is_request_response_capable` is `True` only if at least one handler is registered.
- In explicit mode (`request=...`) type annotations on the function parameters are ignored
  entirely — mixing explicit params with annotations is an error.
- String forward references work in explicit mode: `@response_handler(request="MyRequest", response="MyResponse")`.

### Example 1 — HITL approval handler with type introspection

```python
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._request_info_mixin import response_handler
from agent_framework._workflows._workflow_context import WorkflowContext
from dataclasses import dataclass


@dataclass
class ApprovalRequest:
    action: str
    payload: dict


class ApprovalExecutor(Executor):
    @response_handler
    async def handle_approval(
        self,
        original_request: ApprovalRequest,
        response: bool,          # True = approved, False = rejected
        ctx: WorkflowContext[str],
    ) -> None:
        if response:
            await ctx.yield_output(f"Approved: {original_request.action}")
        else:
            await ctx.yield_output(f"Rejected: {original_request.action}")
```

### Example 2 — explicit type mode for forward references

```python
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._request_info_mixin import response_handler
from agent_framework._workflows._workflow_context import WorkflowContext


class DataPipelineExecutor(Executor):
    @response_handler(request="DataRequest", response="DataResponse", output=str)
    async def handle_data(self, original_request, response, ctx: WorkflowContext[str]):
        await ctx.send_message(response.processed_value)
```

### Example 3 — checking request support before sending

```python
from agent_framework._workflows._executor import Executor


def can_handle_approval(executor: Executor) -> bool:
    from dataclasses import dataclass

    @dataclass
    class ApprovalRequest:
        action: str

    return executor.is_request_supported(ApprovalRequest, bool)
```

### Example 4 — multiple handlers on one executor

```python
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._request_info_mixin import response_handler
from agent_framework._workflows._workflow_context import WorkflowContext
from dataclasses import dataclass


@dataclass
class TextRequest:
    prompt: str


@dataclass
class ImageRequest:
    description: str


class MultiModalExecutor(Executor):
    @response_handler
    async def handle_text(
        self,
        original_request: TextRequest,
        response: str,
        ctx: WorkflowContext[str],
    ) -> None:
        await ctx.yield_output(f"Text result: {response}")

    @response_handler
    async def handle_image(
        self,
        original_request: ImageRequest,
        response: bytes,
        ctx: WorkflowContext[str],
    ) -> None:
        await ctx.yield_output(f"Image: {len(response)} bytes")
```

---

## 8 · `WorkflowAgent.RequestInfoFunctionArgs`

**Sub-package:** `agent_framework._workflows._agent`  
**Import:** `from agent_framework._workflows._agent import WorkflowAgent` (inner class)

`RequestInfoFunctionArgs` is a **nested dataclass** on `WorkflowAgent` that serialises the
metadata for the `request_info` tool call that a wrapped workflow sends when it needs
information from an orchestrator.

### Class signature (1.9.0)

```python
from dataclasses import dataclass
from agent_framework._workflows._events import WorkflowEvent

class WorkflowAgent:
    REQUEST_INFO_FUNCTION_NAME: ClassVar[str] = "request_info"

    @dataclass
    class RequestInfoFunctionArgs:
        request_id: str
        request_event: WorkflowEvent

        def to_dict(self) -> dict[str, Any]:
            return {
                "request_id": self.request_id,
                "request_event": self.request_event.to_dict(),
            }

        @classmethod
        def from_dict(
            cls,
            payload: dict[str, Any],
        ) -> WorkflowAgent.RequestInfoFunctionArgs:
            """Reconstruct from dict; raises ValueError if request_id/request_event missing."""
            ...
```

### Key facts

- `REQUEST_INFO_FUNCTION_NAME = "request_info"` is the name of the synthetic tool injected
  into the wrapped workflow's agent loop so the orchestrator can identify HITL tool calls.
- `to_dict` / `from_dict` provide a round-trip through JSON-serialisable dicts so the
  payload survives checkpoint serialization and cross-process transport.
- `from_dict` raises `ValueError` if `request_id` or `request_event` is absent **or if
  `request_id` is an empty string**.

### Example 1 — deserializing a HITL tool call payload

```python
from agent_framework._workflows._agent import WorkflowAgent

raw_tool_call_args = {
    "request_id": "req-abc123",
    "request_event": {
        "type": "request_info",
        "data": {"question": "What is the capital of France?"},
        "request_id": "req-abc123",
        "source_executor_id": "data_executor",
        "request_type": "builtins.dict",
        "response_type": "builtins.str",
    },
}

args = WorkflowAgent.RequestInfoFunctionArgs.from_dict(raw_tool_call_args)
print(args.request_id)         # "req-abc123"
print(args.request_event.type) # "request_info"
```

### Example 2 — serializing for checkpoint storage

```python
from agent_framework._workflows._agent import WorkflowAgent
from agent_framework._workflows._events import WorkflowEvent, WorkflowEventSource


event = WorkflowEvent.request_info(
    request_id="req-xyz789",
    source_executor_id="approval_executor",
    request_data={"action": "deploy_to_production"},
    response_type=bool,
)

args = WorkflowAgent.RequestInfoFunctionArgs(
    request_id="req-xyz789",
    request_event=event,
)

serialized = args.to_dict()
restored = WorkflowAgent.RequestInfoFunctionArgs.from_dict(serialized)
assert restored.request_id == args.request_id
```

### Example 3 — identifying request_info tool calls in a middleware

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext
from agent_framework._workflows._agent import WorkflowAgent
from collections.abc import Awaitable, Callable


class RequestInfoInterceptorMiddleware(FunctionMiddleware):
    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        if context.function.name == WorkflowAgent.REQUEST_INFO_FUNCTION_NAME:
            args = WorkflowAgent.RequestInfoFunctionArgs.from_dict(context.arguments)
            print(f"HITL request intercepted: id={args.request_id}")
        await call_next()
```

---

## 9 · `EdgeGroupDeliveryStatus`

**Sub-package:** `agent_framework.observability`  
**Import:** `from agent_framework.observability import EdgeGroupDeliveryStatus`

`EdgeGroupDeliveryStatus` is a six-member `Enum` that captures why a message was delivered,
buffered, or dropped when traversing an edge group in the workflow graph. It is emitted as
an OTel span attribute by the telemetry layers.

### Class signature (1.9.0)

```python
from enum import Enum

class EdgeGroupDeliveryStatus(Enum):
    DELIVERED               = "delivered"
    BUFFERED                = "buffered"
    DROPPED_TYPE_MISMATCH   = "dropped type mismatch"
    DROPPED_CONDITION_FALSE = "dropped condition evaluated to false"
    DROPPED_TARGET_MISMATCH = "dropped target mismatch"
    EXCEPTION               = "exception"
```

### Member semantics

| Member | Meaning |
|---|---|
| `DELIVERED` | Message reached the target executor successfully |
| `BUFFERED` | Message stored in a `FanInEdgeGroup` waiting for all fan-out branches to complete |
| `DROPPED_TYPE_MISMATCH` | Edge type filter excluded the message (wrong Python type for this edge) |
| `DROPPED_CONDITION_FALSE` | `SwitchCaseEdgeGroup` condition evaluated to `False` |
| `DROPPED_TARGET_MISMATCH` | Message carries a `target_id` that does not match the edge's target executor |
| `EXCEPTION` | An exception was raised during delivery; captured in the span |

### Example 1 — interpreting delivery status in a custom OTel exporter

```python
from agent_framework.observability import EdgeGroupDeliveryStatus
from opentelemetry.sdk.trace import ReadableSpan


def summarise_delivery(span: ReadableSpan) -> str:
    raw = span.attributes.get("agent_framework.edge.delivery_status")
    if raw is None:
        return "no delivery info"
    status = EdgeGroupDeliveryStatus(raw)
    match status:
        case EdgeGroupDeliveryStatus.DELIVERED:
            return "OK"
        case EdgeGroupDeliveryStatus.BUFFERED:
            return "waiting for fan-in"
        case EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH:
            return "type mismatch — check edge type annotations"
        case EdgeGroupDeliveryStatus.DROPPED_CONDITION_FALSE:
            return "switch-case branch not taken"
        case EdgeGroupDeliveryStatus.DROPPED_TARGET_MISMATCH:
            return "target executor ID mismatch"
        case EdgeGroupDeliveryStatus.EXCEPTION:
            return "delivery exception — see span events"
```

### Example 2 — aggregating drop counts from a span processor

```python
from collections import Counter
from agent_framework.observability import EdgeGroupDeliveryStatus
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan


class EdgeDropCounter(SpanProcessor):
    def __init__(self):
        self.counts: Counter = Counter()

    def on_end(self, span: ReadableSpan) -> None:
        raw = span.attributes.get("agent_framework.edge.delivery_status")
        if raw:
            self.counts[raw] += 1

    def report(self) -> None:
        drops = [
            EdgeGroupDeliveryStatus.DROPPED_TYPE_MISMATCH,
            EdgeGroupDeliveryStatus.DROPPED_CONDITION_FALSE,
            EdgeGroupDeliveryStatus.DROPPED_TARGET_MISMATCH,
        ]
        for d in drops:
            if self.counts[d.value]:
                print(f"{d.name}: {self.counts[d.value]} drops")
```

### Example 3 — alert on exception delivery in production

```python
from agent_framework.observability import EdgeGroupDeliveryStatus


def alert_on_exception_delivery(span) -> None:
    status_raw = span.attributes.get("agent_framework.edge.delivery_status")
    if status_raw == EdgeGroupDeliveryStatus.EXCEPTION.value:
        executor_id = span.attributes.get("agent_framework.executor.id", "unknown")
        print(f"ALERT: Exception during edge delivery in executor {executor_id}")
```

---

## 10 · `IntegrityLabel` · `LabelTrackingFunctionMiddleware`

**Sub-package:** `agent_framework.security`  
**Import:** `from agent_framework.security import IntegrityLabel, LabelTrackingFunctionMiddleware`

`IntegrityLabel` is the companion enum to `ConfidentialityLabel` (covered in Vol. 14). It
tags content as `TRUSTED` or `UNTRUSTED` to drive information-flow control (IFC). 
`LabelTrackingFunctionMiddleware` propagates these labels through tool calls using a strict
three-tier priority.

### Class signatures (1.9.0)

```python
from enum import Enum
from agent_framework import FunctionMiddleware

class IntegrityLabel(str, Enum):
    """Integrity level of content."""
    TRUSTED   = "trusted"    # originated from user input or system messages
    UNTRUSTED = "untrusted"  # AI-generated or from external APIs

class LabelTrackingFunctionMiddleware(FunctionMiddleware):
    def __init__(
        self,
        default_integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED,
        default_confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC,
        auto_hide_untrusted: bool = True,
        hide_threshold: IntegrityLabel = IntegrityLabel.UNTRUSTED,
    ) -> None: ...

    # Public API
    def get_context_label(self) -> ContentLabel: ...
    def reset_context_label(self) -> None: ...
    def get_variable_store(self) -> ContentVariableStore: ...
    def get_security_instructions(self) -> str: ...
    def get_security_tools(self) -> list[FunctionTool]: ...
    def list_variables(self) -> list[str]: ...
    def get_variable_metadata(self, var_id: str) -> dict[str, Any] | None: ...

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None: ...
```

### Three-tier label propagation

| Tier | Source | Priority |
|---|---|---|
| 1 | Per-item embedded label in the result (`additional_properties.security_label`) | Highest — always wins |
| 2 | Tool's `source_integrity` declaration in `additional_properties` | Wins over tier 3 |
| 3 | Join of input argument labels | Fallback; defaults to UNTRUSTED if no inputs |

### `IntegrityLabel` facts

- `TRUSTED` — content came from a human operator (user turn, system prompt).
- `UNTRUSTED` — content came from an AI model or external tool result.
- Because it extends `str`: `IntegrityLabel.TRUSTED == "trusted"` is `True`.

### `LabelTrackingFunctionMiddleware` facts

- `auto_hide_untrusted=True` (default) automatically stores untrusted results as variable
  references in `ContentVariableStore`, replacing them with a `{var_id}` placeholder in
  the conversation.  This prevents prompt injection from leaking into the model's context.
- `hide_threshold` controls which integrity level triggers hiding (default `UNTRUSTED`).
- `get_security_tools()` returns a list of `FunctionTool` objects that allow the model to
  inspect stored variables by ID (e.g., `inspect_variable`).
- `reset_context_label()` clears accumulated labels, typically called between turns.

### Example 1 — basic IFC setup with automatic untrusted hiding

```python
from agent_framework import Agent, tool
from agent_framework.security import (
    IntegrityLabel,
    LabelTrackingFunctionMiddleware,
    SecureAgentConfig,
)
from agent_framework.openai import OpenAIChatClient


@tool(additional_properties={"source_integrity": "untrusted"})
async def fetch_web_page(url: str) -> str:
    return f"<html>... content from {url} ...</html>"


@tool(additional_properties={"source_integrity": "trusted"})
async def get_system_config(key: str) -> str:
    return f"system.{key}=value"


tracker = LabelTrackingFunctionMiddleware(auto_hide_untrusted=True)
agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="secure-agent",
    tools=[fetch_web_page, get_system_config, *tracker.get_security_tools()],
    middleware=[tracker],
    instructions=tracker.get_security_instructions(),
)
```

### Example 2 — using SecureAgentConfig as a one-liner

```python
from agent_framework import Agent, tool
from agent_framework.security import SecureAgentConfig
from agent_framework.openai import OpenAIChatClient


@tool(additional_properties={"source_integrity": "untrusted"})
async def search_web(query: str) -> str:
    return f"[web results for: {query}]"


config = SecureAgentConfig(auto_hide_untrusted=True, block_on_violation=True)
agent = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="safe-agent",
    tools=[search_web],
    context_providers=[config],
)
```

### Example 3 — reading and resetting the context label between turns

```python
from agent_framework.security import LabelTrackingFunctionMiddleware, IntegrityLabel


async def multi_turn_loop(agent, turns: list[str]) -> list[str]:
    tracker = agent.middleware[0]  # assume LabelTrackingFunctionMiddleware is first
    responses = []
    for turn in turns:
        response = await agent.run(turn)
        label = tracker.get_context_label()
        print(f"Turn integrity: {label.integrity}")
        tracker.reset_context_label()   # reset for next turn
        responses.append(response.messages[-1].text)
    return responses
```

### Example 4 — introspecting stored variables after a run

```python
from agent_framework.security import LabelTrackingFunctionMiddleware


async def run_and_inspect(agent, user_message: str) -> None:
    tracker: LabelTrackingFunctionMiddleware = agent.middleware[0]
    await agent.run(messages=[{"role": "user", "content": user_message}])

    variables = tracker.list_variables()
    print(f"Stored untrusted variables: {variables}")
    for var_id in variables:
        meta = tracker.get_variable_metadata(var_id)
        print(f"  {var_id}: integrity={meta.get('integrity') if meta else 'n/a'}")
```

### Example 5 — per-item embedded label in a tool result

```python
from agent_framework import tool
from agent_framework.security import ContentLabel, IntegrityLabel, ConfidentialityLabel


@tool
async def mixed_integrity_search(query: str) -> list[dict]:
    return [
        {
            "text": "Internal report summary",
            "additional_properties": {
                "security_label": ContentLabel(
                    integrity=IntegrityLabel.TRUSTED,
                    confidentiality=ConfidentialityLabel.PUBLIC,
                ).to_dict() if hasattr(ContentLabel, "to_dict") else {
                    "integrity": "trusted",
                    "confidentiality": "public",
                },
            },
        },
        {
            "text": "Content scraped from external site",
            # no security_label → falls back to tier 2/3
        },
    ]
```

---

## 11 · `MiddlewareTermination` · `WorkflowConvergenceException`

**Sub-packages:** `agent_framework._middleware` · `agent_framework.exceptions`  
**Import:** `from agent_framework import MiddlewareTermination` · `from agent_framework.exceptions import WorkflowConvergenceException`

These two exceptions serve distinct roles: `MiddlewareTermination` is a **control-flow
signal** (raise-to-stop-chain) while `WorkflowConvergenceException` indicates the
**runner's iteration limit was exceeded**.

### Class signatures (1.9.0)

```python
from agent_framework.exceptions import MiddlewareException, WorkflowRunnerException
from typing import Any

class MiddlewareTermination(MiddlewareException):
    """Control-flow exception to terminate middleware execution early."""
    def __init__(
        self,
        message: str = "Middleware terminated execution.",
        *,
        result: Any = None,
    ) -> None: ...
    result: Any   # optional early-return value

class WorkflowConvergenceException(WorkflowRunnerException):
    """Raised when a workflow runner fails to converge within the maximum iterations."""
    def __init__(
        self,
        message: str,
        inner_exception: Exception | None = None,
        log_level: int | None = 10,   # logging.DEBUG by default
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
```

### Key facts — `MiddlewareTermination`

- Raise it inside any middleware's `process()` to **short-circuit** the pipeline; no
  subsequent middlewares or the final handler are called.
- The `result` field carries an optional early-return value that the pipeline runner
  extracts and returns to the caller.
- Inherits from `MiddlewareException` → `AgentFrameworkException` → `Exception`.
- Already used internally by `PolicyEnforcementFunctionMiddleware` (when
  `block_on_violation=True`) and `AgentLoopMiddleware`.

### Key facts — `WorkflowConvergenceException`

- Raised by the `Runner` when an agent loop exceeds its configured maximum iteration count.
- `log_level=10` (DEBUG) by default so it doesn't clutter production logs unless you've
  explicitly set DEBUG-level logging.
- `inner_exception` stores the last agent-loop exception if one triggered the convergence
  failure.
- Inherits from `WorkflowRunnerException` → `WorkflowException` → `AgentFrameworkException`.

### Example 1 — short-circuit middleware for rate limiting

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext
from agent_framework._middleware import MiddlewareTermination
from collections import defaultdict
from collections.abc import Awaitable, Callable
import time


class RateLimitMiddleware(FunctionMiddleware):
    def __init__(self, max_calls_per_minute: int = 10) -> None:
        self._calls: dict[str, list[float]] = defaultdict(list)
        self._limit = max_calls_per_minute

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        now = time.monotonic()
        calls = self._calls[context.function.name]
        # drop timestamps older than 60 s
        self._calls[context.function.name] = [t for t in calls if now - t < 60]
        if len(self._calls[context.function.name]) >= self._limit:
            raise MiddlewareTermination(
                f"Rate limit exceeded for {context.function.name}",
                result={"error": "rate_limit_exceeded"},
            )
        self._calls[context.function.name].append(now)
        await call_next()
```

### Example 2 — catching MiddlewareTermination and extracting early result

```python
from agent_framework._middleware import MiddlewareTermination


async def safe_invoke(pipeline, context) -> dict:
    try:
        await pipeline.execute(context)
        return {"status": "ok"}
    except MiddlewareTermination as exc:
        return {
            "status": "terminated",
            "reason": str(exc),
            "result": exc.result,
        }
```

### Example 3 — handling WorkflowConvergenceException in production

```python
from agent_framework.exceptions import WorkflowConvergenceException
from agent_framework import Agent
import logging

logger = logging.getLogger(__name__)


async def run_with_convergence_guard(agent: Agent, prompt: str) -> str | None:
    try:
        response = await agent.run(prompt)
        return response.messages[-1].text
    except WorkflowConvergenceException as exc:
        logger.warning(
            "Agent did not converge: %s — inner: %s",
            exc,
            exc.inner_exception,
        )
        return None
```

### Example 4 — triggering MiddlewareTermination for blocked content

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext
from agent_framework._middleware import MiddlewareTermination
from collections.abc import Awaitable, Callable

BLOCKED_TOOLS = {"delete_all_records", "drop_database"}


class BlocklistMiddleware(FunctionMiddleware):
    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        if context.function.name in BLOCKED_TOOLS:
            raise MiddlewareTermination(
                f"Tool '{context.function.name}' is blocked by security policy.",
                result={"error": "tool_blocked", "tool": context.function.name},
            )
        await call_next()
```

---

## Quick-reference summary

| Class / item | Module | Stable? | Key use |
|---|---|---|---|
| `SupportsCodeInterpreterTool` | `_clients` | ✓ | `isinstance` guard before `get_code_interpreter_tool()` |
| `SupportsWebSearchTool` | `_clients` | ✓ | Guard for web-search hosted tools |
| `SupportsImageGenerationTool` | `_clients` | ✓ | Guard for image-generation hosted tools |
| `SupportsMCPTool` | `_clients` | ✓ | Guard for MCP hosted tools |
| `SupportsFileSearchTool` | `_clients` | ✓ | Guard for file-search / vector-store tools |
| `SupportsShellTool` | `_clients` | ✓ | Guard for shell-execution hosted tools |
| `SupportsGetEmbeddings` | `_clients` | ✓ | Duck-typing for any embedding client |
| `ReleaseCandidateFeature` | `_feature_stage` | ✓ | Enum of near-GA API IDs (empty in 1.9.0) |
| `FeatureStageWarning` | `_feature_stage` | ✓ | Base warning; catch to silence all staged-API warnings |
| `ExperimentalWarning` | `_feature_stage` | ✓ | Subclass; deduplicated per `(category, feature_id)` |
| `EmbeddingGenerationOptions` | `_types` | ✓ | TypedDict with `model` and `dimensions` |
| `Embedding[T]` | `_types` | ✓ | Single embedding vector with metadata |
| `GeneratedEmbeddings[T, O]` | `_types` | ✓ | List of embeddings + usage |
| `WorkflowEventSource` | `_workflows._events` | ✓ | `EXECUTOR` vs `FRAMEWORK` origin tag |
| `SubWorkflowRequestMessage` | `_workflows._workflow_executor` | ✓ | Sub-workflow→parent HITL request |
| `SubWorkflowResponseMessage` | `_workflows._workflow_executor` | ✓ | Parent→sub-workflow HITL response |
| `RequestInfoMixin` | `_workflows._request_info_mixin` | ✓ | Handler discovery and request-support checking |
| `response_handler` | `_workflows._request_info_mixin` | ✓ | Decorator to register HITL response handlers |
| `WorkflowAgent.RequestInfoFunctionArgs` | `_workflows._agent` | ✓ | Serializable HITL tool-call payload |
| `EdgeGroupDeliveryStatus` | `observability` | ✓ | OTel enum: delivered/buffered/dropped/exception |
| `IntegrityLabel` | `security` | ✗ experimental | `TRUSTED`/`UNTRUSTED` integrity tag |
| `LabelTrackingFunctionMiddleware` | `security` | ✗ experimental | IFC label propagation + variable indirection |
| `MiddlewareTermination` | `_middleware` | ✓ | Raise to short-circuit middleware chain |
| `WorkflowConvergenceException` | `exceptions` | ✓ | Runner exceeded max iteration count |

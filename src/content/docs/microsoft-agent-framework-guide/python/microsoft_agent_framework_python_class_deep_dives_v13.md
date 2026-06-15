---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 13"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.1: OpenAIChatClient+OpenAIChatOptions+OpenAISettings+RawOpenAIChatClient, OpenAIChatCompletionClient+OpenAIChatCompletionOptions, OpenAIEmbeddingClient+OpenAIEmbeddingOptions+ContentFilterResultSeverity+OpenAIContentFilterException, AnthropicClient+AnthropicChatOptions+RawAnthropicClient, ClaudeAgent+ClaudeAgentOptions+RawClaudeAgent, AnthropicFoundryClient+AnthropicBedrockClient+AnthropicVertexClient, GroupChatOrchestrator+AgentBasedGroupChatOrchestrator+GroupChatState+AgentOrchestrationOutput, HandoffAgentExecutor+HandoffAgentUserRequest+HandoffSentEvent+OrchestrationState, MagenticOrchestrator+MagenticAgentExecutor+MagenticOrchestratorEvent+MagenticPlanReviewRequest+MagenticPlanReviewResponse, HttpRequestHandler+DefaultHttpRequestHandler+MCPToolHandler+DefaultMCPToolHandler+ExternalInputRequest+ToolApprovalRequest."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 36
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 13

Verified against **agent-framework 1.8.1** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework.openai`, `agent_framework.anthropic`,
`agent_framework.orchestrations`, `agent_framework.declarative`.

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

This volume covers **ten class groups** from the provider sub-packages and orchestration
internals that were not documented in earlier volumes — focusing on the OpenAI Responses API
and Chat Completions API clients in depth, the Anthropic and Claude agent clients, multi-cloud
Anthropic variants, group-chat and handoff orchestration internals, Magentic planning
primitives, and the declarative workflow HTTP/MCP/approval handler protocols:

| # | Class / group | Sub-package |
|---|---|---|
| 1 | `OpenAIChatClient` + `OpenAIChatOptions` + `OpenAISettings` + `RawOpenAIChatClient` | `agent_framework.openai` |
| 2 | `OpenAIChatCompletionClient` + `OpenAIChatCompletionOptions` + `RawOpenAIChatCompletionClient` | `agent_framework.openai` |
| 3 | `OpenAIEmbeddingClient` + `OpenAIEmbeddingOptions` + `OpenAIContinuationToken` + `ContentFilterResultSeverity` + `OpenAIContentFilterException` | `agent_framework.openai` |
| 4 | `AnthropicClient` + `AnthropicChatOptions` + `RawAnthropicClient` | `agent_framework.anthropic` |
| 5 | `ClaudeAgent` + `ClaudeAgentOptions` + `RawClaudeAgent` | `agent_framework.anthropic` |
| 6 | `AnthropicFoundryClient` + `AnthropicBedrockClient` + `AnthropicVertexClient` | `agent_framework.anthropic` |
| 7 | `GroupChatOrchestrator` + `AgentBasedGroupChatOrchestrator` + `GroupChatState` + `AgentOrchestrationOutput` | `agent_framework.orchestrations` |
| 8 | `HandoffAgentExecutor` + `HandoffAgentUserRequest` + `HandoffSentEvent` + `OrchestrationState` | `agent_framework.orchestrations` |
| 9 | `MagenticOrchestrator` + `MagenticAgentExecutor` + `MagenticOrchestratorEvent` + `MagenticPlanReviewRequest` + `MagenticPlanReviewResponse` | `agent_framework.orchestrations` |
| 10 | `HttpRequestHandler` + `DefaultHttpRequestHandler` + `HttpRequestInfo` + `HttpRequestResult` + `MCPToolHandler` + `DefaultMCPToolHandler` + `MCPToolInvocation` + `MCPToolResult` + `ToolApprovalRequest` + `ToolApprovalResponse` + `ExternalInputRequest` + `ExternalInputResponse` | `agent_framework.declarative` |

---

## 1 · `OpenAIChatClient` + `OpenAIChatOptions` + `OpenAISettings` + `RawOpenAIChatClient`

**Sub-package:** `agent_framework.openai`

`OpenAIChatClient` is the primary entry-point for the **OpenAI Responses API** (the modern
multi-turn, stateful API that supersedes the Chat Completions endpoint for most scenarios).
It stacks four MRO layers in a fixed order: `FunctionInvocationLayer` → `ChatMiddlewareLayer`
→ `ChatTelemetryLayer` → `RawOpenAIChatClient`. `RawOpenAIChatClient` is the bare transport
with no cross-cutting concerns; the other three layers add the tool loop, middleware pipeline,
and OpenTelemetry spans respectively.

### Class signatures

```python
from agent_framework.openai import (
    OpenAIChatClient,
    OpenAIChatOptions,
    OpenAISettings,
    RawOpenAIChatClient,
)

class OpenAIChatClient(
    FunctionInvocationLayer[OpenAIChatOptionsT],
    ChatMiddlewareLayer[OpenAIChatOptionsT],
    ChatTelemetryLayer[OpenAIChatOptionsT],
    RawOpenAIChatClient[OpenAIChatOptionsT],
):
    OTEL_PROVIDER_NAME: ClassVar[str] = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        additional_properties: dict[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

class OpenAIChatOptions(ChatOptions[ResponseFormatT], total=False):
    # Responses API specific
    include: list[str]           # extra output data (logprobs, sources, encrypted reasoning…)
    max_tool_calls: int          # cap on built-in tool invocations per response
    prompt: dict[str, Any]       # reusable prompt template reference
    prompt_cache_key: str        # cache key for identical requests
    prompt_cache_retention: Literal["24h"]
    reasoning: ReasoningOptions  # o-series / gpt-5 extended reasoning config
    verbosity: Literal["low", "medium", "high"]   # GPT-5 response length hint
    safety_identifier: str       # stable per-user identifier for policy monitoring
    service_tier: Literal["auto", "default", "flex", "priority", "scale"]
    store: bool                  # persist conversation in OpenAI dashboard

class OpenAISettings(TypedDict, total=False):
    api_key: str
    base_url: str
    org_id: str
    model: str
    embedding_model: str
    chat_model: str               # preferred model for OpenAIChatClient
    chat_completion_model: str    # preferred model for OpenAIChatCompletionClient

class RawOpenAIChatClient(BaseChatClient[OpenAIChatOptionsT]):
    """Bare Responses API transport — no middleware, telemetry, or function loop."""
    INJECTABLE: ClassVar[set[str]] = {"client"}
    STORES_BY_DEFAULT: ClassVar[bool] = True
    SUPPORTS_RICH_FUNCTION_OUTPUT: ClassVar[bool] = True
    SERVED_MODEL_HEADER: ClassVar[str] = "x-ms-served-model"
    FILE_SEARCH_MAX_RESULTS: int = 50
```

### Key parameter details

| Parameter | Type | Resolved from env when `None` |
|---|---|---|
| `model` | `str \| None` | `OPENAI_CHAT_MODEL` → `OPENAI_MODEL` |
| `api_key` | `str \| Callable \| None` | `OPENAI_API_KEY` |
| `org_id` | `str \| None` | `OPENAI_ORG_ID` |
| `base_url` | `str \| None` | `OPENAI_BASE_URL` |
| `instruction_role` | `str \| None` | — `"system"` by default |
| `compaction_strategy` | `CompactionStrategy \| None` | — client-level context compaction |
| `middleware` | `Sequence[...] \| None` | — chat and function middleware |
| `function_invocation_configuration` | `FunctionInvocationConfiguration \| None` | — per-client tool-loop config |
| `additional_properties` | `dict \| None` | — arbitrary metadata attached to the client |

`STORES_BY_DEFAULT = True` means the Responses API client automatically persists the
conversation server-side; you only need to forward `previous_response_id` (via
`OpenAIContinuationToken`) for stateful multi-turn. `RawOpenAIChatClient` also
reads `x-ms-served-model` from Azure OpenAI responses so that
`ChatResponse.model` reflects the actual snapshot served (e.g.
`gpt-5-nano-2025-08-07`) not just the deployment alias.

### Example 1 — Minimal Responses API agent

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient("gpt-4o")          # reads OPENAI_API_KEY automatically
    agent = Agent(
        client=client,
        instructions="You are a concise assistant.",
    )
    response = await agent.run("Summarise the Responses API in one sentence.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — Passing `OpenAIChatOptions` per request

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions

async def main():
    agent = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You reason step-by-step.",
    )
    # Override per run — does not mutate the client
    options: OpenAIChatOptions = {
        "temperature": 0.2,
        "max_tokens": 512,
        "service_tier": "flex",
    }
    response = await agent.run("What is 17 × 23?", options=options)
    print(response.text)

asyncio.run(main())
```

### Example 3 — Reasoning model with extended thinking

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from openai.types.responses import ReasoningOptions

async def main():
    agent = Agent(
        client=OpenAIChatClient("o3-mini"),
        instructions="Solve programming problems rigorously.",
    )
    options: OpenAIChatOptions = {
        "reasoning": ReasoningOptions(effort="high"),
        "include": ["reasoning.encrypted_content"],
    }
    response = await agent.run("Implement a balanced BST insert in Python.", options=options)
    print(response.text)

asyncio.run(main())
```

### Example 4 — Using `OpenAISettings` + `load_settings` for config files

```python
import asyncio
from agent_framework import Agent, load_settings
from agent_framework.openai import OpenAIChatClient, OpenAISettings

async def main():
    settings = load_settings(OpenAISettings, env_prefix="OPENAI_", env_file_path=".env")
    client = OpenAIChatClient(
        settings.get("chat_model") or settings.get("model"),
        api_key=settings.get("api_key"),
        org_id=settings.get("org_id"),
    )
    agent = Agent(client=client, instructions="You are a helpful assistant.")
    response = await agent.run("Hello!")
    print(response.text)

asyncio.run(main())
```

### Example 5 — `RawOpenAIChatClient` with explicit layer composition

```python
import asyncio
from agent_framework._clients import BaseChatClient
from agent_framework.openai import RawOpenAIChatClient
from agent_framework._telemetry import ChatTelemetryLayer
from agent_framework._middleware import ChatMiddlewareLayer

# Use only telemetry, skip function-invocation loop
class MinimalTracedClient(
    ChatTelemetryLayer,
    RawOpenAIChatClient,
): ...

async def main():
    client = MinimalTracedClient("gpt-4o")
    response = await client.get_response(
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

---

## 2 · `OpenAIChatCompletionClient` + `OpenAIChatCompletionOptions` + `RawOpenAIChatCompletionClient`

**Sub-package:** `agent_framework.openai`

`OpenAIChatCompletionClient` targets the **Chat Completions API** — the stateless
`/v1/chat/completions` endpoint. It has the same four-layer MRO as `OpenAIChatClient` but
routes through `RawOpenAIChatCompletionClient` instead. The two clients are not interchangeable:
the Responses API (`OpenAIChatClient`) manages conversation state server-side, while Chat
Completions requires the caller to send the full message list every turn.

### Class signatures

```python
from agent_framework.openai import (
    OpenAIChatCompletionClient,
    OpenAIChatCompletionOptions,
    RawOpenAIChatCompletionClient,
)

class OpenAIChatCompletionClient(
    FunctionInvocationLayer[OpenAIChatCompletionOptionsT],
    ChatMiddlewareLayer[OpenAIChatCompletionOptionsT],
    ChatTelemetryLayer[OpenAIChatCompletionOptionsT],
    RawOpenAIChatCompletionClient[OpenAIChatCompletionOptionsT],
): ...

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        base_url: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
    ) -> None: ...

class OpenAIChatCompletionOptions(ChatOptions[ResponseModelT], total=False):
    # Chat Completions specific
    logit_bias: dict[str | int, float]     # token-level probability adjustments
    logprobs: bool                          # return log-probabilities of output tokens
    top_logprobs: int                       # top N logprob tokens per position (0–20)
    prediction: Prediction                  # predicted output tokens for latency reduction
    verbosity: Literal["low", "medium", "high"]   # GPT-5 brevity hint
    seed: int                               # deterministic sampling
    frequency_penalty: float               # -2.0 to 2.0
    presence_penalty: float                # -2.0 to 2.0
    store: bool                             # persist in OpenAI dashboard
```

### Responses API vs Chat Completions — quick comparison

| Aspect | `OpenAIChatClient` (Responses) | `OpenAIChatCompletionClient` (Completions) |
|---|---|---|
| Endpoint | `/v1/responses` | `/v1/chat/completions` |
| State management | Server-side (`previous_response_id`) | Caller must send full history |
| `STORES_BY_DEFAULT` | `True` | `False` |
| Structured output | Both | Both |
| Reasoning models | `reasoning:` option field | Limited |
| Logprobs | `include: ["message.output_text.logprobs"]` | `logprobs: True` |
| Azure served-model header | Reads `x-ms-served-model` | Uses `response.model` |

### Example 1 — Stateless Chat Completions agent

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient

async def main():
    agent = Agent(
        client=OpenAIChatCompletionClient("gpt-4o"),
        instructions="You are a concise code reviewer.",
    )
    response = await agent.run("Review this: `def add(a, b): return a + b`")
    print(response.text)

asyncio.run(main())
```

### Example 2 — Streaming with `OpenAIChatCompletionOptions`

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatCompletionOptions

async def main():
    agent = Agent(
        client=OpenAIChatCompletionClient("gpt-4o"),
        instructions="Stream output token by token.",
    )
    options: OpenAIChatCompletionOptions = {
        "temperature": 0.7,
        "max_tokens": 256,
        "logprobs": True,
        "top_logprobs": 3,
    }
    async for update in await agent.run("Tell me a short story.", stream=True, options=options):
        print(update.text, end="", flush=True)
    print()

asyncio.run(main())
```

### Example 3 — Structured output with Chat Completions

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatCompletionOptions

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str

async def main():
    agent = Agent(
        client=OpenAIChatCompletionClient("gpt-4o"),
        instructions="Classify sentiment. Return structured JSON.",
    )
    options: OpenAIChatCompletionOptions = {"response_format": SentimentResult}
    response = await agent.run("I love this product!", options=options)
    result = response.output_as(SentimentResult)
    print(f"Sentiment: {result.sentiment} ({result.confidence:.0%})")

asyncio.run(main())
```

### Example 4 — Predictive caching with `prediction`

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient, OpenAIChatCompletionOptions
from openai.types.chat.chat_completion_predicted_output import ChatCompletionPredictedOutput

KNOWN_CODE = "def fibonacci(n):\n    ..."

async def main():
    agent = Agent(
        client=OpenAIChatCompletionClient("gpt-4o"),
        instructions="Complete the function body.",
    )
    options: OpenAIChatCompletionOptions = {
        "prediction": ChatCompletionPredictedOutput(
            type="content",
            content=KNOWN_CODE,
        ),
    }
    response = await agent.run(f"Complete:\n{KNOWN_CODE}", options=options)
    print(response.text)

asyncio.run(main())
```

---

## 3 · `OpenAIEmbeddingClient` + `OpenAIEmbeddingOptions` + `OpenAIContinuationToken` + `ContentFilterResultSeverity` + `OpenAIContentFilterException`

**Sub-package:** `agent_framework.openai`

`OpenAIEmbeddingClient` is the agent framework's wrapper for OpenAI's embeddings endpoint. It
stacks two layers: `EmbeddingTelemetryLayer` (OTel spans for embedding calls) →
`RawOpenAIEmbeddingClient` (bare transport). `OpenAIContinuationToken` extends
`ContinuationToken` to carry the Responses API `response_id` for background-operation polling.
`OpenAIContentFilterException` is raised when Azure OpenAI's content filter blocks a
generation, carrying the filter verdict details.

### Class signatures

```python
from agent_framework.openai import (
    OpenAIEmbeddingClient,
    OpenAIEmbeddingOptions,
    OpenAIContinuationToken,
    ContentFilterResultSeverity,
    OpenAIContentFilterException,
)

class OpenAIEmbeddingClient(
    EmbeddingTelemetryLayer[str, list[float], OpenAIEmbeddingOptionsT],
    RawOpenAIEmbeddingClient[OpenAIEmbeddingOptionsT],
):
    OTEL_PROVIDER_NAME: ClassVar[str] = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,     # OPENAI_EMBEDDING_MODEL → OPENAI_MODEL
        api_key: str | Callable[[], str | Awaitable[str]] | None = None,
        org_id: str | None = None,
        default_headers: Mapping[str, str] | None = None,
        async_client: AsyncOpenAI | None = None,
        base_url: str | None = None,
        otel_provider_name: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

class OpenAIEmbeddingOptions(EmbeddingGenerationOptions, total=False):
    encoding_format: Literal["float", "base64"]
    user: str                   # end-user identifier for abuse monitoring

class OpenAIContinuationToken(ContinuationToken):
    response_id: str            # Responses API ID for background polling

class ContentFilterResultSeverity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    SAFE = "safe"
    LOW = "low"

@dataclass
class OpenAIContentFilterException(ChatClientContentFilterException):
    param: str | None
    content_filter_code: ContentFilterCodes
    content_filter_result: dict[str, ContentFilterResult]
```

### Example 1 — Generate embeddings for semantic search

```python
import asyncio
from agent_framework.openai import OpenAIEmbeddingClient, OpenAIEmbeddingOptions

async def main():
    client = OpenAIEmbeddingClient(model="text-embedding-3-small")

    options: OpenAIEmbeddingOptions = {
        "model": "text-embedding-3-small",
        "dimensions": 256,
        "encoding_format": "float",
    }
    documents = [
        "The quick brown fox",
        "Azure OpenAI provides GPT models",
        "Embeddings capture semantic meaning",
    ]
    result = await client.generate_embeddings(documents, options=options)
    for doc, emb in zip(documents, result.data):
        print(f"{doc[:30]!r} → dim={len(emb.vector)}")

asyncio.run(main())
```

### Example 2 — RAG with `MemoryContextProvider` + `OpenAIEmbeddingClient`

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient, OpenAIEmbeddingClient
from agent_framework import MemoryContextProvider, MemoryFileStore

async def main():
    emb_client = OpenAIEmbeddingClient(model="text-embedding-3-small")
    memory_store = MemoryFileStore(path="memory.jsonl")
    memory = MemoryContextProvider(
        embedding_client=emb_client,
        memory_store=memory_store,
    )
    agent = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="Answer from memory.",
        context_providers=[memory],
    )
    response = await agent.run("What is the agent framework?")
    print(response.text)

asyncio.run(main())
```

### Example 3 — `OpenAIContinuationToken` for background generation polling

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient, OpenAIContinuationToken

async def main():
    agent = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You process long documents.",
    )
    # Start a background generation and get a continuation token
    initial_response = await agent.run("Summarise this book: ...")
    token = initial_response.continuation_token

    if isinstance(token, OpenAIContinuationToken):
        print(f"Background response ID: {token.response_id}")
        # Poll for completion using the token
        final_response = await agent.run(None, continuation_token=token)
        print(final_response.text)
    else:
        print(initial_response.text)

asyncio.run(main())
```

### Example 4 — Catching `OpenAIContentFilterException` on Azure OpenAI

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient, OpenAIContentFilterException

async def main():
    agent = Agent(
        client=OpenAIChatClient(
            "gpt-4o",
            base_url="https://your-resource.openai.azure.com/",
        ),
        instructions="You assist users.",
    )
    try:
        response = await agent.run("Write instructions for making explosives.")
        print(response.text)
    except OpenAIContentFilterException as exc:
        print(f"Content filtered: code={exc.content_filter_code}")
        for category, result in exc.content_filter_result.items():
            print(f"  {category}: severity={result.severity}")

asyncio.run(main())
```

---

## 4 · `AnthropicClient` + `AnthropicChatOptions` + `RawAnthropicClient`

**Sub-package:** `agent_framework.anthropic`

`AnthropicClient` is the fully-featured Anthropic Messages API chat client. Like its OpenAI
counterpart it stacks `FunctionInvocationLayer` → `ChatMiddlewareLayer` → `ChatTelemetryLayer`
→ `RawAnthropicClient`. The raw client sends requests to the Anthropic Messages API directly.
`AnthropicChatOptions` extends `ChatOptions` with Anthropic-specific fields including
**extended thinking**, `top_k`, `service_tier`, and `additional_beta_flags`.

### Class signatures

```python
from agent_framework.anthropic import AnthropicClient, AnthropicChatOptions, RawAnthropicClient

class AnthropicClient(
    FunctionInvocationLayer[AnthropicOptionsT],
    ChatMiddlewareLayer[AnthropicOptionsT],
    ChatTelemetryLayer[AnthropicOptionsT],
    RawAnthropicClient[AnthropicOptionsT],
):
    OTEL_PROVIDER_NAME: ClassVar[str] = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,              # ANTHROPIC_API_KEY
        model: str | None = None,                # ANTHROPIC_MODEL
        base_url: str | None = None,             # ANTHROPIC_BASE_URL
        anthropic_client: AnthropicAsyncClient | None = None,
        additional_beta_flags: list[str] | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

class AnthropicChatOptions(ChatOptions[ResponseModelT], total=False):
    top_k: int                              # top-K sampling
    service_tier: Literal["auto", "standard_only"]
    thinking: ThinkingConfig                # extended thinking config
    container: dict[str, Any]              # skills container config
    additional_beta_flags: list[str]       # extra beta headers per request
    # Unsupported base fields typed as None:
    logit_bias: None
    seed: None
    frequency_penalty: None
    presence_penalty: None
    store: None
    conversation_id: None

class RawAnthropicClient(BaseChatClient[AnthropicOptionsT]):
    """Bare Messages API transport — no middleware, telemetry, or function loop."""
    OTEL_PROVIDER_NAME: ClassVar[str] = "anthropic"
```

> **Note:** `additional_beta_flags` defaults to `["mcp-client-2025-04-04", "code-execution-2025-08-25"]` at the client level. You can append more flags without replacing the defaults using the per-request `additional_beta_flags` in `AnthropicChatOptions`.

### Example 1 — Minimal Anthropic agent

```python
import asyncio
from agent_framework import Agent
from agent_framework.anthropic import AnthropicClient

async def main():
    agent = Agent(
        client=AnthropicClient(model="claude-opus-4-5"),
        instructions="You are a helpful assistant.",
    )
    response = await agent.run("What are the key features of extended thinking?")
    print(response.text)

asyncio.run(main())
```

### Example 2 — Extended thinking with `AnthropicChatOptions`

```python
import asyncio
from agent_framework import Agent
from agent_framework.anthropic import AnthropicClient, AnthropicChatOptions
from anthropic.types import ThinkingConfigEnabled

async def main():
    agent = Agent(
        client=AnthropicClient(model="claude-opus-4-8"),
        instructions="Solve complex reasoning problems step by step.",
    )
    options: AnthropicChatOptions = {
        "thinking": ThinkingConfigEnabled(
            type="enabled",
            budget_tokens=8192,
        ),
        "max_tokens": 16384,   # must exceed budget_tokens
    }
    response = await agent.run(
        "A train leaves at 9am at 60mph. Another at 11am at 90mph. When do they meet?",
        options=options,
    )
    print(response.text)

asyncio.run(main())
```

### Example 3 — Structured output from Anthropic

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework.anthropic import AnthropicClient, AnthropicChatOptions

class CodeReview(BaseModel):
    issues: list[str]
    severity: str
    recommendation: str

async def main():
    agent = Agent(
        client=AnthropicClient(model="claude-sonnet-4-6"),
        instructions="Review Python code and return structured feedback.",
    )
    options: AnthropicChatOptions = {
        "response_format": CodeReview,
        "max_tokens": 1024,
    }
    code = "def divide(a, b): return a / b"
    response = await agent.run(f"Review: {code}", options=options)
    review = response.output_as(CodeReview)
    print(f"Severity: {review.severity}")
    for issue in review.issues:
        print(f"  - {issue}")

asyncio.run(main())
```

### Example 4 — Tool use with the Anthropic client

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.anthropic import AnthropicClient

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, partly cloudy"

async def main():
    agent = Agent(
        client=AnthropicClient(model="claude-sonnet-4-6"),
        instructions="You are a weather assistant.",
        tools=[get_weather],
    )
    response = await agent.run("What's the weather in London?")
    print(response.text)

asyncio.run(main())
```

### Example 5 — Passing beta flags per request

```python
import asyncio
from agent_framework import Agent
from agent_framework.anthropic import AnthropicClient, AnthropicChatOptions

async def main():
    agent = Agent(
        client=AnthropicClient(model="claude-sonnet-4-6"),
        instructions="You are a code assistant.",
    )
    # Enable an additional beta feature for just this request
    options: AnthropicChatOptions = {
        "additional_beta_flags": ["interleaved-thinking-2025-05-14"],
        "max_tokens": 2048,
    }
    response = await agent.run("Implement bubble sort.", options=options)
    print(response.text)

asyncio.run(main())
```

---

## 5 · `ClaudeAgent` + `ClaudeAgentOptions` + `RawClaudeAgent`

**Sub-package:** `agent_framework.anthropic`

`ClaudeAgent` wraps the **Claude Code CLI** (via `claude-agent-sdk`) as an agent-framework
`BaseAgent`. Instead of routing through an HTTP chat client, it spawns a Claude Code process
and drives agentic tasks including file editing, code execution, and MCP tool use. It is the
framework's drop-in integration for Claude's native agent loop. `RawClaudeAgent` is the same
without the `AgentTelemetryLayer` OTel instrumentation.

### Class signatures

```python
from agent_framework.anthropic import ClaudeAgent, ClaudeAgentOptions, RawClaudeAgent

class ClaudeAgent(AgentTelemetryLayer, RawClaudeAgent[OptionsT]):
    """ClaudeAgent with OTel instrumentation (recommended for production)."""
    ...

class RawClaudeAgent(BaseAgent, Generic[OptionsT]):
    AGENT_PROVIDER_NAME: ClassVar[str] = "anthropic.claude"

    def __init__(
        self,
        instructions: str | None = None,
        *,
        client: ClaudeSDKClient | None = None,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: Sequence[AgentMiddlewareTypes] | None = None,
        tools: ToolTypes | Callable | str | Sequence[...] | None = None,
        default_options: OptionsT | MutableMapping[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

class ClaudeAgentOptions(TypedDict, total=False):
    system_prompt: str
    cli_path: str | Path              # auto-detected by default
    cwd: str | Path                   # working directory for CLI
    env: dict[str, str]               # environment variables to forward
    settings: str                     # path to Claude settings file
    model: str                        # "sonnet" | "opus" | "haiku"
    fallback_model: str
    allowed_tools: list[str]          # tool allowlist (empty = all)
    disallowed_tools: list[str]       # tool blocklist
    mcp_servers: dict[str, McpServerConfig]
    permission_mode: PermissionMode   # "default" | "acceptEdits" | "plan" | "bypassPermissions"
    can_use_tool: CanUseTool          # permission callback
    max_turns: int
    max_budget_usd: float
    hooks: dict[str, list[HookMatcher]]
    add_dirs: list[str | Path]
    sandbox: SandboxSettings
    agents: dict[str, AgentDefinition]
    output_format: dict[str, Any]     # JSON schema for structured output
    enable_file_checkpointing: bool
    betas: list[SdkBeta]
    plugins: list[SdkPluginConfig]
    setting_sources: list[SettingSource]
```

### Example 1 — Basic Claude agent via CLI

```python
import asyncio
from agent_framework.anthropic import ClaudeAgent

async def main():
    async with ClaudeAgent(
        instructions="You are a helpful coding assistant.",
    ) as agent:
        response = await agent.run("Write a Python function to reverse a string.")
        print(response.text)

asyncio.run(main())
```

### Example 2 — Claude agent with tool restrictions

```python
import asyncio
from agent_framework.anthropic import ClaudeAgent, ClaudeAgentOptions

async def main():
    options: ClaudeAgentOptions = {
        "allowed_tools": ["read_file", "list_files"],   # only read access
        "model": "sonnet",
        "max_turns": 5,
        "permission_mode": "default",
    }
    async with ClaudeAgent(
        instructions="Analyse the project structure.",
        default_options=options,
    ) as agent:
        response = await agent.run("List all Python files and their sizes.")
        print(response.text)

asyncio.run(main())
```

### Example 3 — Claude agent with MCP servers

```python
import asyncio
from agent_framework.anthropic import ClaudeAgent, ClaudeAgentOptions

async def main():
    options: ClaudeAgentOptions = {
        "mcp_servers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"],
            }
        },
        "model": "sonnet",
    }
    async with ClaudeAgent(
        instructions="Use the filesystem MCP to manage files.",
        default_options=options,
    ) as agent:
        response = await agent.run("Create a file called hello.txt with 'Hello, World!'")
        print(response.text)

asyncio.run(main())
```

### Example 4 — Claude agent as participant in multi-agent workflow

```python
import asyncio
from agent_framework import WorkflowBuilder
from agent_framework.anthropic import ClaudeAgent
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent

async def main():
    # Claude handles coding tasks, OpenAI handles review
    coder = ClaudeAgent(
        instructions="You write Python code.",
        name="coder",
    )
    reviewer = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You review Python code for correctness and style.",
        name="reviewer",
    )
    builder = WorkflowBuilder()
    builder.add_agent(coder, id="coder")
    builder.add_agent(reviewer, id="reviewer")
    builder.add_edge(source="coder", target="reviewer")
    builder.set_output_from("reviewer")

    workflow = builder.build()
    async with workflow.run(messages=["Write a function to compute factorial."]) as run:
        async for event in run:
            print(event)

asyncio.run(main())
```

---

## 6 · `AnthropicFoundryClient` + `AnthropicBedrockClient` + `AnthropicVertexClient`

**Sub-package:** `agent_framework.anthropic`

These three clients extend the Anthropic layer stack to reach Claude through **Azure AI Foundry**,
**AWS Bedrock**, and **Google Vertex AI** respectively. They share the same
`FunctionInvocationLayer` → `ChatMiddlewareLayer` → `ChatTelemetryLayer` → `Raw*` stack but
each `Raw*` client handles the cloud-specific authentication and endpoint routing.

### Class signatures

```python
from agent_framework.anthropic import (
    AnthropicFoundryClient,
    AnthropicBedrockClient,
    AnthropicVertexClient,
)

class AnthropicFoundryClient(...):
    def __init__(
        self,
        *,
        model: str | None = None,
        resource: str | None = None,             # Foundry resource name
        api_key: str | None = None,
        azure_ad_token_provider: AnthropicFoundryAzureADTokenProvider | None = None,
        base_url: str | None = None,
        anthropic_client: AsyncAnthropicFoundry | None = None,
        additional_beta_flags: list[str] | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

class AnthropicBedrockClient(...):
    def __init__(
        self,
        *,
        model: str | None = None,
        aws_secret_key: str | None = None,
        aws_access_key: str | None = None,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        aws_session_token: str | None = None,
        base_url: str | None = None,
        anthropic_client: AsyncAnthropicBedrock | None = None,
        additional_beta_flags: list[str] | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

class AnthropicVertexClient(...):
    def __init__(
        self,
        *,
        model: str | None = None,
        region: str | None = None,               # CLOUD_ML_REGION
        project_id: str | None = None,           # ANTHROPIC_VERTEX_PROJECT_ID
        access_token: str | None = None,
        credentials: GoogleCredentials | None = None,
        base_url: str | None = None,
        anthropic_client: AsyncAnthropicVertex | None = None,
        additional_beta_flags: list[str] | None = None,
        additional_properties: dict[str, Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...
```

### Environment variable resolution

| Client | Auth env vars |
|---|---|
| `AnthropicFoundryClient` | `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_FOUNDRY_BASE_URL` |
| `AnthropicBedrockClient` | `AWS_SECRET_ACCESS_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_DEFAULT_REGION`, `AWS_PROFILE`, `AWS_SESSION_TOKEN` |
| `AnthropicVertexClient` | `CLOUD_ML_REGION`, `ANTHROPIC_VERTEX_PROJECT_ID`, Application Default Credentials |

### Example 1 — Claude on Azure AI Foundry

```python
import asyncio
from azure.identity.aio import DefaultAzureCredential
from agent_framework import Agent
from agent_framework.anthropic import AnthropicFoundryClient

async def main():
    credential = DefaultAzureCredential()
    client = AnthropicFoundryClient(
        model="claude-opus-4-8",
        resource="my-foundry-resource",
        azure_ad_token_provider=credential.get_token,
    )
    agent = Agent(client=client, instructions="You are an Azure AI assistant.")
    response = await agent.run("Explain Azure AI Foundry.")
    print(response.text)
    await credential.close()

asyncio.run(main())
```

### Example 2 — Claude on AWS Bedrock with IAM role

```python
import asyncio
from agent_framework import Agent
from agent_framework.anthropic import AnthropicBedrockClient

async def main():
    client = AnthropicBedrockClient(
        model="us.anthropic.claude-opus-4-5-20251101-v1:0",
        aws_region="us-east-1",
        aws_profile="bedrock-prod",          # uses named AWS profile
    )
    agent = Agent(client=client, instructions="You assist with AWS infrastructure.")
    response = await agent.run("List best practices for S3 bucket security.")
    print(response.text)

asyncio.run(main())
```

### Example 3 — Claude on Google Vertex AI

```python
import asyncio
from agent_framework import Agent
from agent_framework.anthropic import AnthropicVertexClient

async def main():
    client = AnthropicVertexClient(
        model="claude-opus-4-5@20251101",
        region="us-central1",
        project_id="my-gcp-project",
        # credentials resolved via Application Default Credentials if not provided
    )
    agent = Agent(client=client, instructions="You assist with GCP architecture.")
    response = await agent.run("How do I set up Vertex AI Pipelines?")
    print(response.text)

asyncio.run(main())
```

### Example 4 — Multi-cloud failover pattern

```python
import asyncio
from agent_framework import Agent
from agent_framework.anthropic import (
    AnthropicClient,
    AnthropicFoundryClient,
    AnthropicBedrockClient,
)

def build_agent_with_fallback() -> Agent:
    """Try direct Anthropic, fall back to Foundry, then Bedrock."""
    import os
    if os.getenv("ANTHROPIC_API_KEY"):
        client = AnthropicClient(model="claude-sonnet-4-6")
    elif os.getenv("ANTHROPIC_FOUNDRY_RESOURCE"):
        client = AnthropicFoundryClient(model="claude-sonnet-4-6")
    else:
        client = AnthropicBedrockClient(
            model="us.anthropic.claude-sonnet-4-6-20251005-v1:0",
            aws_region="us-east-1",
        )
    return Agent(client=client, instructions="You are a resilient assistant.")

async def main():
    agent = build_agent_with_fallback()
    response = await agent.run("Hello!")
    print(response.text)

asyncio.run(main())
```

---

## 7 · `GroupChatOrchestrator` + `AgentBasedGroupChatOrchestrator` + `GroupChatState` + `AgentOrchestrationOutput`

**Sub-package:** `agent_framework.orchestrations`

These classes are the **internal implementation layer** behind `GroupChatBuilder`. When you call
`GroupChatBuilder.build()`, it produces a `Workflow` whose central executor is a
`GroupChatOrchestrator` (function-driven) or `AgentBasedGroupChatOrchestrator` (LLM-driven).
Direct use lets you embed orchestrators in custom workflows or inspect their internals.
`GroupChatState` is the read-only snapshot passed to selection functions.
`AgentOrchestrationOutput` is the Pydantic model the LLM must produce for agent-based selection.

### Class signatures

```python
from agent_framework.orchestrations import (
    GroupChatOrchestrator,
    AgentBasedGroupChatOrchestrator,
    GroupChatState,
    AgentOrchestrationOutput,
    GroupChatRequestMessage,
    GroupChatRequestSentEvent,
    GroupChatResponseReceivedEvent,
)

class GroupChatOrchestrator(BaseGroupChatOrchestrator):
    def __init__(
        self,
        id: str,
        participant_registry: ParticipantRegistry,
        selection_func: GroupChatSelectionFunction,
        *,
        name: str | None = None,
        max_rounds: int | None = None,
        termination_condition: TerminationCondition | None = None,
    ) -> None: ...

class AgentBasedGroupChatOrchestrator(BaseGroupChatOrchestrator):
    def __init__(
        self,
        agent: Agent,
        participant_registry: ParticipantRegistry,
        *,
        max_rounds: int | None = None,
        termination_condition: TerminationCondition | None = None,
        retry_attempts: int | None = None,
        session: AgentSession | None = None,
    ) -> None: ...

@dataclass(frozen=True)
class GroupChatState:
    current_round: int
    participants: OrderedDict[str, str]    # name → description
    conversation: list[Message]

class AgentOrchestrationOutput(BaseModel):
    terminate: bool
    reason: str
    next_speaker: str | None = None
    final_message: str | None = None

@dataclass
class GroupChatRequestMessage:
    additional_instruction: str | None = None
    metadata: dict[str, Any] | None = None
```

### Example 1 — Round-robin selection with `GroupChatOrchestrator`

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import GroupChatState

async def main():
    writers = [
        Agent(client=OpenAIChatClient("gpt-4o"), instructions="You write action.", name="writer_a"),
        Agent(client=OpenAIChatClient("gpt-4o"), instructions="You write dialogue.", name="writer_b"),
    ]
    names = [w.name for w in writers]
    idx = [0]

    def round_robin(state: GroupChatState) -> str:
        speaker = names[idx[0] % len(names)]
        idx[0] += 1
        return speaker

    builder = WorkflowBuilder()
    for w in writers:
        builder.add_agent(w, id=w.name)
    grp = builder.add_group_chat(
        selection_func=round_robin,
        participants=writers,
        max_rounds=4,
    )
    workflow = builder.build()
    async with workflow.run(messages=["Continue this story: 'The door creaked open...'"]) as run:
        async for event in run:
            if hasattr(event, "text"):
                print(event.text)

asyncio.run(main())
```

### Example 2 — LLM-driven selection with `AgentBasedGroupChatOrchestrator`

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

async def main():
    selector = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions=(
            "You select the best next speaker for a technical discussion. "
            "Return AgentOrchestrationOutput JSON."
        ),
        name="selector",
    )
    expert_a = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You are a backend expert.",
        name="backend_expert",
    )
    expert_b = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You are a frontend expert.",
        name="frontend_expert",
    )
    builder = WorkflowBuilder()
    builder.add_agent(expert_a, id="backend_expert")
    builder.add_agent(expert_b, id="frontend_expert")
    builder.add_agent_based_group_chat(
        selector_agent=selector,
        participants=[expert_a, expert_b],
        max_rounds=6,
    )
    workflow = builder.build()
    async with workflow.run(messages=["Design a real-time collaborative editor architecture."]) as run:
        async for event in run:
            if hasattr(event, "text"):
                print(event.text)

asyncio.run(main())
```

### Example 3 — Inspecting `GroupChatState` in selection function

```python
from agent_framework.orchestrations import GroupChatState

def smart_selector(state: GroupChatState) -> str:
    """Select based on conversation content."""
    if state.current_round == 0:
        return "planner"   # always start with planner
    # Check last message for handoff cues
    last_msg = state.conversation[-1] if state.conversation else None
    if last_msg and "code" in last_msg.text.lower():
        return "coder"
    if last_msg and "review" in last_msg.text.lower():
        return "reviewer"
    # Default: cycle through participants
    names = list(state.participants.keys())
    return names[state.current_round % len(names)]
```

---

## 8 · `HandoffAgentExecutor` + `HandoffAgentUserRequest` + `HandoffSentEvent` + `OrchestrationState`

**Sub-package:** `agent_framework.orchestrations`

`HandoffAgentExecutor` is the per-agent executor that powers **handoff workflows**. It intercepts
tool calls that match registered handoff targets and routes control to the target agent.
`HandoffAgentUserRequest` is the HITL request emitted when no handoff fires and user input is
needed. `HandoffSentEvent` carries the handoff source and target for observability.
`OrchestrationState` is the unified checkpoint dataclass all three orchestrator types serialise
to when checkpointing a workflow.

### Class signatures

```python
from agent_framework.orchestrations import (
    HandoffAgentExecutor,
    HandoffAgentUserRequest,
    HandoffSentEvent,
    OrchestrationState,
    AgentRequestInfoResponse,
)

class HandoffAgentExecutor(AgentExecutor):
    def __init__(
        self,
        agent: Agent,
        handoffs: Sequence[HandoffConfiguration],
        *,
        agent_session: AgentSession | None = None,
        is_start_agent: bool = False,
        termination_condition: TerminationCondition | None = None,
        autonomous_mode: bool = False,
        autonomous_mode_prompt: str | None = None,
        autonomous_mode_turn_limit: int | None = None,
    ) -> None: ...

@dataclass
class HandoffAgentUserRequest:
    agent_response: AgentResponse

    @staticmethod
    def create_response(
        response: str | list[str] | Message | list[Message],
    ) -> list[Message]: ...

    @staticmethod
    def terminate() -> list[Message]: ...

@dataclass
class HandoffSentEvent:
    source: str
    target: str

@dataclass
class OrchestrationState:
    conversation: list[Message]
    round_index: int = 0
    orchestrator_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    task: Message | None = None

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestrationState": ...
```

### Example 1 — HITL handoff with `HandoffAgentUserRequest`

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffAgentUserRequest

async def main():
    triage = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="Triage support tickets. Hand off to billing or technical team.",
        name="triage",
    )
    billing = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="Handle billing inquiries.",
        name="billing",
    )
    technical = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="Handle technical support.",
        name="technical",
    )

    builder = WorkflowBuilder()
    builder.add_handoff(
        agents=[triage, billing, technical],
        start_agent=triage,
    )
    workflow = builder.build()

    async def handle_request(request: HandoffAgentUserRequest) -> list:
        print(f"Agent says: {request.agent_response.text}")
        user_reply = input("Your response (or 'done' to end): ")
        if user_reply.lower() == "done":
            return HandoffAgentUserRequest.terminate()
        return HandoffAgentUserRequest.create_response(user_reply)

    async with workflow.run(
        messages=["I was charged twice for my subscription."],
        request_handler=handle_request,
    ) as run:
        async for event in run:
            pass

asyncio.run(main())
```

### Example 2 — Observing handoff events via `HandoffSentEvent`

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffSentEvent
from agent_framework import WorkflowEventType

async def main():
    triage = Agent(client=OpenAIChatClient("gpt-4o"), instructions="Triage.", name="triage")
    specialist = Agent(client=OpenAIChatClient("gpt-4o"), instructions="Specialist.", name="specialist")

    builder = WorkflowBuilder()
    builder.add_handoff(agents=[triage, specialist], start_agent=triage)
    workflow = builder.build()

    async with workflow.run(messages=["I need technical help."]) as run:
        async for event in run:
            if (
                event.type == WorkflowEventType.CUSTOM
                and isinstance(event.data, HandoffSentEvent)
            ):
                print(f"Handoff: {event.data.source} → {event.data.target}")

asyncio.run(main())
```

### Example 3 — Checkpointing orchestration with `OrchestrationState`

```python
import asyncio
import json
from agent_framework import Agent, Message
from agent_framework.orchestrations import OrchestrationState

async def save_checkpoint(state: OrchestrationState, path: str):
    """Persist orchestration state to disk."""
    with open(path, "w") as f:
        json.dump(state.to_dict(), f, default=str)

async def load_checkpoint(path: str) -> OrchestrationState:
    """Restore orchestration state from disk."""
    with open(path) as f:
        data = json.load(f)
    return OrchestrationState.from_dict(data)

async def main():
    state = OrchestrationState(
        conversation=[Message(role="user", contents=["Hello"])],
        round_index=3,
        orchestrator_name="triage_orchestrator",
        metadata={"last_agent": "billing"},
    )
    await save_checkpoint(state, "/tmp/orch_checkpoint.json")
    restored = await load_checkpoint("/tmp/orch_checkpoint.json")
    print(f"Round: {restored.round_index}, Orchestrator: {restored.orchestrator_name}")

asyncio.run(main())
```

---

## 9 · `MagenticOrchestrator` + `MagenticAgentExecutor` + `MagenticOrchestratorEvent` + `MagenticPlanReviewRequest` + `MagenticPlanReviewResponse`

**Sub-package:** `agent_framework.orchestrations`

`MagenticOrchestrator` is the **Magentic-One workflow engine** — the inner/outer loop
manager that drives planning, progress tracking, stall detection, and replanning.
`MagenticAgentExecutor` wraps participant agents and adds reset support for replanning cycles.
`MagenticOrchestratorEvent` carries plan-created, replanned, and progress-ledger-updated events.
`MagenticPlanReviewRequest` + `MagenticPlanReviewResponse` enable HITL plan signoff before
execution begins or after a stall triggers replanning.

### Class signatures

```python
from agent_framework.orchestrations import (
    MagenticOrchestrator,
    MagenticAgentExecutor,
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
    MagenticProgressLedgerItem,
    MagenticPlanReviewRequest,
    MagenticPlanReviewResponse,
    MagenticResetSignal,
)

class MagenticOrchestrator(BaseGroupChatOrchestrator):
    def __init__(
        self,
        manager: MagenticManagerBase,
        participant_registry: ParticipantRegistry,
        *,
        require_plan_signoff: bool = False,
    ) -> None: ...

class MagenticAgentExecutor(AgentExecutor):
    def __init__(self, agent: SupportsAgentRun) -> None: ...

    @handler
    async def handle_magentic_reset(
        self, signal: MagenticResetSignal, ctx: WorkflowContext
    ) -> None: ...

class MagenticOrchestratorEventType(str, Enum):
    PLAN_CREATED = "plan_created"
    REPLANNED = "replanned"
    PROGRESS_LEDGER_UPDATED = "progress_ledger_updated"

@dataclass
class MagenticOrchestratorEvent:
    event_type: MagenticOrchestratorEventType
    content: Message | MagenticProgressLedger

@dataclass
class MagenticProgressLedgerItem(DictConvertible):
    reason: str
    answer: str | bool

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MagenticProgressLedgerItem": ...

@dataclass
class MagenticPlanReviewRequest:
    plan: Message
    current_progress: MagenticProgressLedger | None
    is_stalled: bool

    def approve(self) -> "MagenticPlanReviewResponse": ...
    def revise(self, feedback: str | list[str] | Message | list[Message]) -> "MagenticPlanReviewResponse": ...

@dataclass
class MagenticPlanReviewResponse:
    review: list[Message]

    @staticmethod
    def approve() -> "MagenticPlanReviewResponse": ...
    @staticmethod
    def revise(feedback: str | list[str] | Message | list[Message]) -> "MagenticPlanReviewResponse": ...
```

### Example 1 — Magentic-One workflow with `MagenticBuilder`

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

async def main():
    manager_agent = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You are a Magentic-One manager. Plan, assign, and track tasks.",
        name="manager",
    )
    coder = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You write Python code.",
        name="coder",
    )
    reviewer = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You review code for bugs and style.",
        name="reviewer",
    )

    builder = WorkflowBuilder()
    builder.add_magentic(
        manager=manager_agent,
        participants=[coder, reviewer],
    )
    workflow = builder.build()

    async with workflow.run(
        messages=["Build a web scraper for news headlines."]
    ) as run:
        async for event in run:
            if hasattr(event, "text") and event.text:
                print(event.text[:120])

asyncio.run(main())
```

### Example 2 — Human plan signoff with `MagenticPlanReviewRequest`

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import MagenticPlanReviewRequest, MagenticPlanReviewResponse

async def plan_review_handler(request: MagenticPlanReviewRequest) -> MagenticPlanReviewResponse:
    print("=== PROPOSED PLAN ===")
    print(request.plan.text)
    if request.is_stalled and request.current_progress:
        print("[Plan triggered by stall — previous progress shown above]")
    decision = input("Approve? (y/n/feedback): ").strip()
    if decision.lower() == "y":
        return request.approve()
    elif decision.lower() == "n":
        return request.revise("Please try a different approach.")
    else:
        return request.revise(decision)

async def main():
    manager = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You plan and coordinate complex research tasks.",
        name="manager",
    )
    researcher = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="You research topics thoroughly.",
        name="researcher",
    )

    builder = WorkflowBuilder()
    builder.add_magentic(
        manager=manager,
        participants=[researcher],
        require_plan_signoff=True,
    )
    workflow = builder.build()

    async with workflow.run(
        messages=["Research the latest advances in quantum computing."],
        request_handler=plan_review_handler,
    ) as run:
        async for event in run:
            pass

asyncio.run(main())
```

### Example 3 — Monitoring Magentic events

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import (
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
)
from agent_framework import WorkflowEventType

async def main():
    manager = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="Coordinate agents.",
        name="manager",
    )
    worker = Agent(
        client=OpenAIChatClient("gpt-4o"),
        instructions="Complete assigned tasks.",
        name="worker",
    )

    builder = WorkflowBuilder()
    builder.add_magentic(manager=manager, participants=[worker])
    workflow = builder.build()

    async with workflow.run(messages=["Summarise the pros and cons of microservices."]) as run:
        async for event in run:
            if (
                event.type == WorkflowEventType.CUSTOM
                and isinstance(event.data, MagenticOrchestratorEvent)
            ):
                etype = event.data.event_type
                if etype == MagenticOrchestratorEventType.PLAN_CREATED:
                    print("Plan created.")
                elif etype == MagenticOrchestratorEventType.REPLANNED:
                    print("Replanning triggered (stall detected).")
                elif etype == MagenticOrchestratorEventType.PROGRESS_LEDGER_UPDATED:
                    ledger = event.data.content
                    print(f"Progress ledger updated: {ledger}")

asyncio.run(main())
```

### Example 4 — `MagenticProgressLedgerItem` round-trip serialisation

```python
from agent_framework.orchestrations import MagenticProgressLedgerItem

# Serialise for checkpointing
item = MagenticProgressLedgerItem(
    reason="The coder has produced a working implementation.",
    answer=True,
)
data = item.to_dict()
print(data)  # {'reason': '...', 'answer': True}

# Restore from checkpoint
restored = MagenticProgressLedgerItem.from_dict(data)
assert restored.answer is True
```

---

## 10 · Declarative workflow handlers — `HttpRequestHandler`, `DefaultHttpRequestHandler`, `HttpRequestInfo`, `HttpRequestResult`, `MCPToolHandler`, `DefaultMCPToolHandler`, `MCPToolInvocation`, `MCPToolResult`, `ToolApprovalRequest`, `ToolApprovalResponse`, `ExternalInputRequest`, `ExternalInputResponse`

**Sub-package:** `agent_framework.declarative`

Declarative workflows (loaded from YAML via `WorkflowFactory`) can make HTTP calls, invoke
MCP tools, and pause for human approval. Each capability is exposed via a Protocol + default
implementation pair that you can replace with a custom handler at workflow creation time.
These classes are the **extension points** for adding URL filtering, SSRF protection,
authentication, and custom approval UI to declarative workflows.

### Class signatures

```python
from agent_framework.declarative import (
    # HTTP
    HttpRequestHandler,
    DefaultHttpRequestHandler,
    HttpRequestInfo,
    HttpRequestResult,
    # MCP
    MCPToolHandler,
    DefaultMCPToolHandler,
    MCPToolInvocation,
    MCPToolResult,
    MCPToolApprovalRequest,
    # Tool approval
    ToolApprovalRequest,
    ToolApprovalResponse,
    # External input / HITL
    ExternalInputRequest,
    ExternalInputResponse,
    AgentExternalInputRequest,
    AgentExternalInputResponse,
)

@runtime_checkable
class HttpRequestHandler(Protocol):
    async def send(self, info: HttpRequestInfo) -> HttpRequestResult: ...

class DefaultHttpRequestHandler:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        client_provider: ClientProvider | None = None,
    ) -> None: ...
    async def send(self, info: HttpRequestInfo) -> HttpRequestResult: ...
    async def aclose(self) -> None: ...

@dataclass
class HttpRequestInfo:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    query_parameters: dict[str, str] = field(default_factory=dict)
    body: str | None = None
    body_content_type: str | None = None
    timeout_ms: int | None = None
    connection_name: str | None = None

@dataclass
class HttpRequestResult:
    status_code: int
    is_success_status_code: bool
    body: str
    headers: dict[str, list[str]] = field(default_factory=dict)

@runtime_checkable
class MCPToolHandler(Protocol):
    async def invoke_tool(self, invocation: MCPToolInvocation) -> MCPToolResult: ...

class DefaultMCPToolHandler:
    LIST_TOOLS_TOOL_NAME: ClassVar[str] = "tools/list"

    def __init__(
        self,
        *,
        client_provider: Callable[[MCPToolInvocation], httpx.AsyncClient | None] | None = None,
        cache_max_size: int = 32,
    ) -> None: ...
    async def invoke_tool(self, invocation: MCPToolInvocation) -> MCPToolResult: ...
    async def aclose(self) -> None: ...

@dataclass
class MCPToolInvocation:
    server_url: str
    tool_name: str
    server_label: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    connection_name: str | None = None

@dataclass
class MCPToolResult:
    outputs: list[Content] = field(default_factory=list)
    is_error: bool = False
    error_message: str | None = None

@dataclass
class ToolApprovalRequest:
    request_id: str
    function_name: str
    arguments: dict[str, Any]

@dataclass
class ToolApprovalResponse:
    approved: bool
    reason: str | None = None

@dataclass
class ExternalInputRequest:
    request_id: str
    message: str
    request_type: str = "external"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ExternalInputResponse:
    user_input: str
    value: Any = None

@dataclass
class AgentExternalInputRequest:
    request_id: str
    agent_name: str
    agent_response: str
    iteration: int = 0
    messages: list[Message] = field(default_factory=list)
    function_calls: list[Content] = field(default_factory=list)

@dataclass
class AgentExternalInputResponse:
    user_input: str
    messages: list[Message] = field(default_factory=list)
    function_results: dict[str, Content] = field(default_factory=dict)
```

### Example 1 — Running a declarative workflow with `DefaultHttpRequestHandler`

```yaml
# workflow.yaml
name: weather-workflow
actions:
  - type: HttpRequest
    method: GET
    url: "https://api.open-meteo.com/v1/forecast?latitude=51.5&longitude=-0.1&current_weather=true"
    outputVariable: Local.weather
  - type: SetVariable
    variable: Workflow.Outputs.result
    value: "{{Local.weather.current_weather.temperature}}°C"
```

```python
import asyncio
from agent_framework.declarative import WorkflowFactory, DefaultHttpRequestHandler

async def main():
    handler = DefaultHttpRequestHandler()
    factory = WorkflowFactory(http_request_handler=handler)
    workflow = factory.create_workflow_from_yaml_path("workflow.yaml")

    result = await workflow.run(inputs={})
    print(result.outputs.get("result"))
    await handler.aclose()

asyncio.run(main())
```

### Example 2 — Custom HTTP handler with SSRF protection

```python
import asyncio
import ipaddress
from urllib.parse import urlparse
from agent_framework.declarative import (
    HttpRequestHandler,
    HttpRequestInfo,
    HttpRequestResult,
    DefaultHttpRequestHandler,
    WorkflowFactory,
)

class SafeHttpRequestHandler:
    """Wraps DefaultHttpRequestHandler with SSRF protection."""

    BLOCKED_RANGES = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
    ]

    def __init__(self) -> None:
        self._inner = DefaultHttpRequestHandler()

    def _is_blocked(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        try:
            addr = ipaddress.ip_address(host)
            return any(addr in net for net in self.BLOCKED_RANGES)
        except ValueError:
            return False

    async def send(self, info: HttpRequestInfo) -> HttpRequestResult:
        if self._is_blocked(info.url):
            return HttpRequestResult(
                status_code=403,
                is_success_status_code=False,
                body="SSRF blocked",
            )
        return await self._inner.send(info)

    async def aclose(self) -> None:
        await self._inner.aclose()

async def main():
    factory = WorkflowFactory(http_request_handler=SafeHttpRequestHandler())
    workflow = factory.create_workflow_from_yaml_path("workflow.yaml")
    result = await workflow.run(inputs={"url": "https://example.com/data"})
    print(result)

asyncio.run(main())
```

### Example 3 — Tool approval gate with `ToolApprovalRequest`

```yaml
# approval_workflow.yaml
actions:
  - type: InvokeFunction
    function: delete_records
    arguments:
      table: "{{Workflow.Inputs.table}}"
    requireApproval: true
    outputVariable: Local.deleteResult
```

```python
import asyncio
from agent_framework.declarative import (
    WorkflowFactory,
    ToolApprovalRequest,
    ToolApprovalResponse,
)

async def approval_handler(request: ToolApprovalRequest) -> ToolApprovalResponse:
    print(f"[APPROVAL REQUIRED] {request.function_name}")
    print(f"  Arguments: {request.arguments}")
    decision = input("Approve? (y/n): ").strip().lower()
    return ToolApprovalResponse(approved=(decision == "y"))

async def main():
    factory = WorkflowFactory(tool_approval_handler=approval_handler)
    workflow = factory.create_workflow_from_yaml_path("approval_workflow.yaml")
    result = await workflow.run(inputs={"table": "users"})
    print(result)

asyncio.run(main())
```

### Example 4 — MCP tool invocation with `DefaultMCPToolHandler`

```yaml
# mcp_workflow.yaml
actions:
  - type: InvokeMcpTool
    serverUrl: "http://localhost:3000/mcp"
    toolName: search_documents
    arguments:
      query: "{{Workflow.Inputs.query}}"
    outputVariable: Local.searchResults
```

```python
import asyncio
from agent_framework.declarative import WorkflowFactory, DefaultMCPToolHandler

async def main():
    mcp_handler = DefaultMCPToolHandler(cache_max_size=16)
    factory = WorkflowFactory(mcp_tool_handler=mcp_handler)
    workflow = factory.create_workflow_from_yaml_path("mcp_workflow.yaml")
    result = await workflow.run(inputs={"query": "quarterly report"})
    print(result.outputs.get("searchResults"))
    await mcp_handler.aclose()

asyncio.run(main())
```

### Example 5 — HITL with `ExternalInputRequest` and `AgentExternalInputRequest`

```python
import asyncio
from agent_framework.declarative import (
    WorkflowFactory,
    ExternalInputRequest,
    ExternalInputResponse,
    AgentExternalInputRequest,
    AgentExternalInputResponse,
)

async def external_input_handler(
    request: ExternalInputRequest | AgentExternalInputRequest,
) -> ExternalInputResponse | AgentExternalInputResponse:
    if isinstance(request, AgentExternalInputRequest):
        print(f"Agent '{request.agent_name}' says: {request.agent_response}")
        user_input = input("Your reply: ")
        return AgentExternalInputResponse(user_input=user_input)
    else:
        print(f"Workflow asks: {request.message}")
        user_input = input("Your reply: ")
        return ExternalInputResponse(user_input=user_input)

async def main():
    factory = WorkflowFactory(request_handler=external_input_handler)
    workflow = factory.create_workflow_from_yaml_path("hitl_workflow.yaml")
    result = await workflow.run(inputs={"task": "Book a flight to Tokyo"})
    print(result)

asyncio.run(main())
```

### Example 6 — Discovering available MCP tools via `LIST_TOOLS_TOOL_NAME`

```python
import asyncio
from agent_framework.declarative import (
    DefaultMCPToolHandler,
    MCPToolInvocation,
)

async def list_mcp_tools(server_url: str) -> None:
    handler = DefaultMCPToolHandler()
    # Use the reserved constant to invoke tools/list
    invocation = MCPToolInvocation(
        server_url=server_url,
        tool_name=DefaultMCPToolHandler.LIST_TOOLS_TOOL_NAME,
    )
    result = await handler.invoke_tool(invocation)
    if result.is_error:
        print(f"Error: {result.error_message}")
    else:
        for output in result.outputs:
            print(output.text)
    await handler.aclose()

asyncio.run(list_mcp_tools("http://localhost:3000/mcp"))
```

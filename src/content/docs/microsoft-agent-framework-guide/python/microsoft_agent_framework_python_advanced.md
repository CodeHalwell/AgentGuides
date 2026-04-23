---
title: "Microsoft Agent Framework (Python) — Advanced Patterns"
description: "Custom BaseChatClient / BaseEmbeddingClient implementations, custom ContextProvider subclasses, capability protocols (SupportsMCPTool, SupportsFileSearchTool), and the feature_stage gating system. Verified against agent-framework-core 1.1.0."
framework: microsoft-agent-framework
language: python
---

# Advanced Patterns — Python

This page covers the extensibility hooks in `agent-framework-core` that the feature-level pages gloss over: writing your own chat client, embedding client, or context provider, feature-detecting clients at runtime via the `Supports*` protocols, and opting into experimental/RC APIs.

Verified against `agent-framework-core==1.1.0`. The abstract base classes live in `agent_framework._clients` and `agent_framework._sessions` but are re-exported from the top-level `agent_framework` package.

## Custom chat client — `BaseChatClient`

`BaseChatClient` is the abstract base every first-party client inherits from. Subclass it to wrap any request/response HTTP API, an in-memory mock for tests, or a caching/rate-limiting façade around another client. The contract is a single abstract method — `_inner_get_response` — that handles both streaming and non-streaming via a `stream` flag.

```python
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from typing import Any, ClassVar
from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Message,
    ResponseStream,
)


class EchoChatClient(BaseChatClient):
    """Test double that echoes the last user message back as the assistant response."""

    OTEL_PROVIDER_NAME: ClassVar[str] = "echo"

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        text = (last_user.text if last_user else "") or "<no input>"

        if stream:
            async def _iter() -> AsyncIterable[ChatResponseUpdate]:
                for token in text.split():
                    yield ChatResponseUpdate(
                        role="assistant",
                        contents=[token + " "],
                    )
            return self._build_response_stream(_iter())

        async def _single() -> ChatResponse:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[text])],
                response_id="echo-1",
            )

        return _single()


agent = Agent(client=EchoChatClient(), instructions="Echo only.")
response = await agent.run("Hello")
assert response.text == "Hello"
```

What you get for free by subclassing:

- **Middleware, telemetry, and tool loop** all wrap your implementation automatically — you never implement the function-calling protocol.
- `_build_response_stream(...)` returns the right `ResponseStream` shape that the upstream layers expect.
- `_prepare_messages_for_model_call` handles compaction when a `compaction_strategy` is attached.
- `to_dict()` / serialization falls out of `SerializationMixin`.

Override `OTEL_PROVIDER_NAME` so the generated OpenTelemetry spans attribute correctly. Override the class attribute `STORES_BY_DEFAULT = True` when your backend manages conversation history server-side (like OpenAI Responses API with `store=True`) — the agent will skip auto-injecting an `InMemoryHistoryProvider`.

### Caching wrapper

A thin subclass can memoise chat calls by key:

```python
import hashlib
import json

class CachingChatClient(BaseChatClient):
    OTEL_PROVIDER_NAME: ClassVar[str] = "caching"

    def __init__(self, inner: BaseChatClient) -> None:
        super().__init__()
        self._inner = inner
        self._cache: dict[str, ChatResponse] = {}

    def _key(self, messages: Sequence[Message], options: Mapping[str, Any]) -> str:
        blob = json.dumps(
            {"m": [m.text for m in messages], "o": dict(options)},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        if stream:
            return self._inner._inner_get_response(
                messages=messages, stream=True, options=options, **kwargs
            )
        key = self._key(messages, options)
        cached = self._cache.get(key)
        if cached is not None:
            async def _hit() -> ChatResponse:
                return cached
            return _hit()

        async def _miss() -> ChatResponse:
            response = await self._inner._inner_get_response(
                messages=messages, stream=False, options=options, **kwargs
            )
            self._cache[key] = response
            return response
        return _miss()
```

Wrap any real client:

```python
from agent_framework.openai import OpenAIChatClient

cached = CachingChatClient(OpenAIChatClient(model="gpt-5"))
agent = Agent(client=cached, instructions="…")
```

## Custom embedding client — `BaseEmbeddingClient`

The embeddings surface follows the same shape as chat — one abstract method, generic over the input type (`str` by default), output type (typically `list[float]`), and options.

```python
from collections.abc import Sequence
from agent_framework import BaseEmbeddingClient, Embedding, GeneratedEmbeddings


class HashEmbeddingClient(BaseEmbeddingClient):
    """Toy deterministic embedding — useful for tests that need repeatable vectors."""

    OTEL_PROVIDER_NAME = "hash-toy"

    async def get_embeddings(self, values, *, options=None):
        vectors = [
            Embedding(
                vector=[(hash(v) >> i) & 0xFF for i in range(8)],
                model="hash-toy-v1",
            )
            for v in values
        ]
        return GeneratedEmbeddings(vectors, options=options)


client = HashEmbeddingClient()
result = await client.get_embeddings(["hello", "world"])
assert len(result) == 2
assert result[0].dimensions == 8
```

`GeneratedEmbeddings` subclasses `list` — iterate it directly, or access `.usage` and `.options` metadata. The `Embedding.dimensions` property is lazy — it uses `len(vector)` when you don't pass an explicit `dimensions=`.

### Batching wrapper

```python
class BatchedEmbeddingClient(BaseEmbeddingClient):
    """Flushes to the inner client in batches of N — useful to rate-limit providers."""

    def __init__(self, inner: BaseEmbeddingClient, batch_size: int = 16) -> None:
        super().__init__()
        self._inner = inner
        self._batch_size = batch_size

    async def get_embeddings(self, values, *, options=None):
        values = list(values)
        all_embeddings: list[Embedding] = []
        for i in range(0, len(values), self._batch_size):
            chunk = values[i : i + self._batch_size]
            batch = await self._inner.get_embeddings(chunk, options=options)
            all_embeddings.extend(batch)
        return GeneratedEmbeddings(all_embeddings, options=options)
```

## Custom context provider — `ContextProvider`

`ContextProvider` is the base class for anything that mutates the `SessionContext` before a run (injecting messages, tools, instructions, or middleware) or observes the response afterwards. The `SkillsProvider` from the [Skills page](./microsoft_agent_framework_python_skills/) is itself a `ContextProvider`. Roll your own when you have domain-specific context to attach.

```python
from typing import Any
from agent_framework import Agent, ContextProvider, Message
from agent_framework.openai import OpenAIChatClient


class TenantContextProvider(ContextProvider):
    """Load a per-tenant system prompt prefix before every run."""

    DEFAULT_SOURCE_ID = "tenant_context"

    def __init__(self, tenant_prompts: dict[str, str]) -> None:
        super().__init__(self.DEFAULT_SOURCE_ID)
        self._tenant_prompts = tenant_prompts

    async def before_run(self, *, agent, session, context, state) -> None:
        tenant_id = context.kwargs.get("tenant_id")
        prefix = self._tenant_prompts.get(tenant_id)
        if prefix:
            context.extend_instructions(self.source_id, prefix)

    async def after_run(self, *, agent, session, context, state) -> None:
        # Log the answered message count for this tenant — stored in provider-scoped state.
        state["turns"] = state.get("turns", 0) + 1


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    context_providers=[
        TenantContextProvider({
            "acme": "The user works at ACME Corp. Use metric units.",
            "globex": "The user works at Globex. Use imperial units.",
        }),
    ],
)

await agent.run("What's the local weather?", tenant_id="acme")
```

Key points:

- `source_id` tags everything the provider adds (messages, tools, instructions) so other providers / the `HistoryProvider` can filter by source. Keep it unique per instance.
- `state` is a **provider-scoped** dict (isolated from other providers). Full cross-provider state is still reachable via `session.state`.
- `context.extend_messages(...)`, `context.extend_tools(...)`, `context.extend_middleware(...)`, and `context.extend_instructions(...)` are the mutation APIs — the framework merges everything before the model call.
- `after_run` sees the final response at `context.response` — this is where RAG-style stores update their indexes.

### Loading reference docs on demand

```python
class DocumentContextProvider(ContextProvider):
    """Inject a relevant snippet into the prompt based on the last user message."""

    def __init__(self, retriever) -> None:
        super().__init__("doc_retriever")
        self._retriever = retriever

    async def before_run(self, *, agent, session, context, state) -> None:
        user_msgs = [m for m in session.messages if m.role == "user"]
        if not user_msgs:
            return
        snippets = await self._retriever.search(user_msgs[-1].text, top_k=3)
        context.extend_messages(
            self,
            [Message(role="system", contents=[f"<context>\n{s}\n</context>"]) for s in snippets],
        )
```

Compare with `HistoryProvider` (in the same module): use `HistoryProvider` for persistent conversation storage, use plain `ContextProvider` for one-shot context injection per run.

## Capability protocols — `Supports*`

Several first-party client classes publish optional capabilities through `runtime_checkable` protocols. Use `isinstance(client, Supports*)` to feature-detect at runtime:

```python
from agent_framework import (
    Agent,
    SupportsMCPTool,
    SupportsFileSearchTool,
    SupportsWebSearchTool,
    SupportsCodeInterpreterTool,
    SupportsImageGenerationTool,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework.anthropic import AnthropicClient


def build_tools(client) -> list:
    tools: list = []
    if isinstance(client, SupportsWebSearchTool):
        tools.append(client.get_web_search_tool())
    if isinstance(client, SupportsFileSearchTool):
        tools.append(client.get_file_search_tool(vector_store_ids=["vs_123"]))
    if isinstance(client, SupportsCodeInterpreterTool):
        tools.append(client.get_code_interpreter_tool())
    if isinstance(client, SupportsMCPTool):
        tools.append(client.get_mcp_tool(name="learn", url="https://learn.microsoft.com/api/mcp"))
    return tools


# OpenAI → web search + file search + code interpreter available.
# Anthropic → only MCP tool supported.
for client in [OpenAIChatClient(), AnthropicClient()]:
    agent = Agent(client=client, tools=build_tools(client))
```

Why protocols instead of inheritance? Each provider implements a disjoint set of hosted tools — OpenAI exposes file search and code interpreter via the Responses API, Anthropic exposes MCP natively, Bedrock exposes guardrails. A single inheritance hierarchy would force everyone to stub everything; protocols give precise, runtime-checkable capability negotiation.

The full list of capability protocols:

| Protocol | What it guarantees |
|---|---|
| `SupportsChatGetResponse` | Non-streaming chat — the universal protocol every chat client satisfies |
| `SupportsAgentRun` | The agent-level run protocol (implemented by `Agent`, `ChatAgent`, `WorkflowAgent`, `FoundryAgent`, `CopilotStudioAgent`) |
| `SupportsGetEmbeddings` | The embedding protocol every `*EmbeddingClient` satisfies |
| `SupportsMCPTool` | Client can produce an MCP tool descriptor |
| `SupportsFileSearchTool` | Client supports file search against vector stores |
| `SupportsWebSearchTool` | Client supports hosted web search |
| `SupportsCodeInterpreterTool` | Client supports hosted code interpreter |
| `SupportsImageGenerationTool` | Client supports hosted image generation |

## Feature stages — experimental and release-candidate APIs

Parts of agent-framework are gated behind stages. Using a staged API emits a `FeatureStageWarning` (subclass `ExperimentalWarning` for experimental, a sibling category for release-candidate) **once per feature ID per process** — the framework dedupes so you get one warning at startup, not one per call. The `__feature_stage__` and `__feature_id__` attributes on the decorated class or function carry the metadata.

```python
import warnings
from agent_framework import ExperimentalFeature, ReleaseCandidateFeature
from agent_framework._feature_stage import (
    ExperimentalWarning,
    FeatureStageWarning,  # parent class of all staged-API warnings
)

# 1. Silence every staged-API warning (experimental + RC) for the whole process.
warnings.filterwarnings("ignore", category=FeatureStageWarning)

# 2. Silence only experimental ones — still surface RC warnings.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# 3. Turn staged-API warnings into test failures (pytest-style).
warnings.filterwarnings("error", category=FeatureStageWarning)
```

The currently gated features:

| Stage | Enum member | Covers |
|---|---|---|
| Experimental | `ExperimentalFeature.EXPERIMENTAL` (generic) | Anything without a more specific ID |
| Experimental | `ExperimentalFeature.SKILLS` | `Skill`, `SkillResource`, `SkillScript`, `SkillsProvider` |
| Experimental | `ExperimentalFeature.EVALS` | `LocalEvaluator`, `evaluate_agent`, `evaluate_workflow`, `@evaluator` |
| Experimental | `ExperimentalFeature.COMPACTION` | `SlidingWindowStrategy`, `SummarizationStrategy`, `TokenBudgetComposedStrategy`, `SelectiveToolCallCompactionStrategy`, `ToolResultCompactionStrategy` |
| Release candidate | `ReleaseCandidateFeature.WORKFLOW_VIZ` | `WorkflowViz` diagram rendering |

Inspect any class or callable at runtime to see what stage it belongs to:

```python
from agent_framework import LocalEvaluator

print(LocalEvaluator.__feature_stage__)  # "experimental"
print(LocalEvaluator.__feature_id__)     # "EVALS"
```

Use this in CI to fail the build if anyone imports an experimental API without an explicit opt-in — e.g. assert that every class you import in production code has `__feature_stage__` unset or equal to `"stable"`.

## Resilience patterns

Three composable layers:

- **Per-tool circuit breaker** — `FunctionTool(max_invocation_exceptions=5)` stops calling a flapping tool.
- **Per-request retry** — `FunctionMiddleware` with exponential backoff (see [Middleware → Retrying](./microsoft_agent_framework_python_middleware/#retrying-a-failed-tool-call)).
- **Per-run budget** — `AgentMiddleware` raising `MiddlewareTermination` when a usage cap is hit.

Example combining all three:

```python
from agent_framework import (
    Agent,
    AgentMiddleware,
    FunctionMiddleware,
    MiddlewareTermination,
    tool,
)
import asyncio

@tool(max_invocation_exceptions=5)
def call_internal_api(endpoint: str) -> str:
    ...


class RetryMiddleware(FunctionMiddleware):
    async def process(self, context, call_next):
        for attempt in range(3):
            try:
                await call_next()
                return
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * 2**attempt)


class TokenBudget(AgentMiddleware):
    def __init__(self, ceiling: int) -> None:
        self.used = 0
        self.ceiling = ceiling

    async def process(self, context, call_next):
        await call_next()
        usage = getattr(context.result, "usage_details", None) or {}
        self.used += int(usage.get("total_token_count", 0) or 0)
        if self.used > self.ceiling:
            raise MiddlewareTermination(
                f"token budget {self.ceiling} exceeded (used {self.used})"
            )


agent = Agent(
    client=OpenAIChatClient(),
    tools=[call_internal_api],
    middleware=[TokenBudget(ceiling=100_000), RetryMiddleware()],
)
```

## When to reach for each pattern

- Building an **integration with a new provider** that isn't in the first-party list → subclass `BaseChatClient` + optionally `BaseEmbeddingClient`. You inherit the tool loop, middleware, telemetry, and serialization for free.
- Adding **cross-cutting behaviour** to an existing client (caching, request coalescing, shadow traffic) → wrap the client with a thin `BaseChatClient` subclass that delegates to `_inner_get_response`.
- Injecting **domain context per run** (retrieved docs, tenant prefixes, entitlements) → subclass `ContextProvider`.
- Switching **implementations at runtime** based on capability → `isinstance(client, Supports*)` guards.
- Gating **in-development APIs** in CI → use `enable_experimental_feature` in a `conftest.py` and fail the build if a warning escapes.

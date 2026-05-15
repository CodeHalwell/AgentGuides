---
title: "Microsoft Agent Framework (Python) — Advanced Patterns"
description: "Custom BaseChatClient / BaseEmbeddingClient implementations, custom ContextProvider subclasses, capability protocols (SupportsMCPTool, SupportsFileSearchTool), and the feature_stage gating system. Verified against agent-framework-core 1.4.0."
framework: microsoft-agent-framework
language: python
---

# Advanced Patterns — Python

This page covers the extensibility hooks in `agent-framework-core` that the feature-level pages gloss over: writing your own chat client, embedding client, or context provider, feature-detecting clients at runtime via the `Supports*` protocols, and opting into experimental/RC APIs.

Verified against `agent-framework-core==1.4.0`. The abstract base classes live in `agent_framework._clients` and `agent_framework._sessions` but are re-exported from the top-level `agent_framework` package.

## Custom agent — `BaseAgent` and `RawAgent`

`Agent` (the everyday class) is `RawAgent` wrapped with middleware, telemetry, and deduping layers. The hierarchy:

```
BaseAgent              ← ABC: id, name, description, context_providers, middleware, sessions
└── RawAgent           ← Chat-loop implementation (no telemetry/middleware wrapping)
    └── Agent          ← What you import — adds telemetry + middleware layer wrapping
```

Reach for `BaseAgent` directly when you need a non-chat agent — for example, an agent that delegates to a workflow, an external HTTP service, or a deterministic rules engine. Reach for `RawAgent` when you want chat semantics but **without** any of the wrapping layers — handy for benchmarking the raw client cost or for embedding inside another agent that already provides telemetry.

### `BaseAgent` subclass — non-chat agent over a workflow

A `BaseAgent` subclass only has to implement `run(...)`. The base class gives you `id`, `create_session()`, `get_session()`, and the `as_tool()` helper for free.

```python
from collections.abc import AsyncIterator
from typing import Any
from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    BaseAgent,
    Message,
    ResponseStream,
)


class WorkflowBackedAgent(BaseAgent):
    """An agent whose ``run()`` delegates to a precomputed workflow.

    Useful when you have deterministic logic (a state machine, a graph of
    deterministic transforms) but you want it to plug into the same agent
    surface that consumers already use — including ``as_tool()``.
    """

    def __init__(self, workflow, *, name: str, description: str) -> None:
        super().__init__(name=name, description=description)
        self._workflow = workflow

    async def run(
        self,
        messages: str | list[Message] | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> AgentResponse | ResponseStream[AgentResponseUpdate, AgentResponse]:
        text = messages if isinstance(messages, str) else (messages[-1].text if messages else "")

        if stream:
            async def _stream() -> AsyncIterator[AgentResponseUpdate]:
                async for event in self._workflow.run(text, stream=True):
                    yield AgentResponseUpdate(
                        contents=[str(event)],
                        role="assistant",
                    )

            # Wrap the iterator in a ResponseStream so callers can `await stream.get_final_response()`.
            return ResponseStream(_stream(), finalizer=lambda updates: AgentResponse(
                messages=[Message(role="assistant", contents=[u.text or "" for u in updates])],
            ))

        result = await self._workflow.run(text)
        return AgentResponse(
            messages=[Message(role="assistant", contents=[str(result.get_outputs()[-1])])],
        )
```

Once defined, the agent works with `agent.as_tool()`, multi-agent orchestrations, and any code path that accepts `SupportsAgentRun`.

### Wrapping an agent as a tool — `as_tool()`

Both `BaseAgent` and its concrete subclasses expose `as_tool()`. It returns a `FunctionTool` that other agents can call as if your agent were a single function — a clean way to expose a specialised sub-agent to a generalist supervisor.

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


client = OpenAIChatClient()

billing_specialist = Agent(
    client=client,
    name="billing_specialist",
    description="Answers detailed billing and invoicing questions.",
    instructions="You are a billing expert. Quote exact dollar amounts only when present in context.",
)

# Promote the specialist to a tool — supervisor can now invoke it inline.
specialist_tool = billing_specialist.as_tool(
    name="ask_billing",
    description="Forward a billing-related question to the billing specialist.",
    arg_name="question",
    arg_description="The billing question, in plain English.",
    approval_mode="never_require",     # set to "always_require" to gate every delegation
    propagate_session=False,           # parent session is *not* shared with the specialist
)

supervisor = Agent(
    client=client,
    name="supervisor",
    instructions=(
        "You triage user questions. If the question is about billing, call ask_billing — "
        "do not answer it yourself. Otherwise reply directly."
    ),
    tools=[specialist_tool],
)

response = await supervisor.run("What was the late-payment fee on my March invoice?")
```

Two flags worth knowing:

- **`approval_mode="always_require"`** turns the delegation into a HITL pause point. Every supervisor-to-specialist call emits a `function_approval_request` you must resolve before the specialist runs — useful when the sub-agent has access to expensive tools.
- **`propagate_session=True`** forwards the supervisor's `AgentSession` into the specialist's `run(...)`. Both agents then share the same conversation history. Default is `False` because most supervisor/specialist designs want the specialist to start fresh on every delegation.

If you supply `stream_callback=`, the supervisor invokes the specialist with `stream=True` and forwards each `AgentResponseUpdate` through your callback — useful when you want to display sub-agent progress in real time without exposing the streaming surface to the model.

### `RawAgent` vs `Agent` — when to skip the wrapper

Reach for `RawAgent` when you want chat semantics but no middleware/telemetry overhead — typically a benchmark harness or a load-test client. Otherwise stick with `Agent`.

```python
from agent_framework import RawAgent
from agent_framework.openai import OpenAIChatClient

# Identical surface to Agent (run, run with stream=True, sessions, tools).
# Skips the AgentMiddlewareLayer, ChatMiddlewareLayer, and OTel span emission.
raw = RawAgent(
    client=OpenAIChatClient(),
    instructions="You are a benchmark target.",
    tools=[],
)

response = await raw.run("hello")
```

When the docs mention `Agent`, every example works unchanged with `RawAgent` — only the wrapping layers change.

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
    Content,
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
                        contents=[Content.from_text(token + " ")],
                    )
            return self._build_response_stream(_iter())

        async def _single() -> ChatResponse:
            return ChatResponse(
                messages=[Message(role="assistant", contents=[Content.from_text(text)])],
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

### Building a streaming chunk — `ChatResponseUpdate`

Every streaming chat client yields `ChatResponseUpdate` instances. The class is a thin dataclass-style container — pass `Content` items, optionally a role, and any of the optional metadata fields the framework propagates downstream (`response_id`, `message_id`, `conversation_id`, `model`, `created_at`, `finish_reason`, `continuation_token`, `additional_properties`).

```python
from agent_framework import ChatResponseUpdate, Content, FinishReason

# Plain text chunk — what most streaming clients emit per token batch.
chunk = ChatResponseUpdate(
    contents=[Content.from_text(" Hello")],
    role="assistant",
    response_id="resp_1",
    message_id="msg_1",
    model="gpt-4o-mini",
)
print(chunk.text)        # " Hello" — concatenates every text Content
print(str(chunk))        # same as chunk.text — __str__ is text

# Final chunk — signal completion with finish_reason.
final = ChatResponseUpdate(
    contents=[Content.from_text("")],
    role="assistant",
    response_id="resp_1",
    message_id="msg_1",
    finish_reason=FinishReason.STOP,
)
```

Two practical things every custom-streaming-client implementation hits:

- **`text` ignores non-text content.** The property concatenates `content.text` only for `content.type == "text"` items. Function calls and image content remain in `update.contents` but don't pollute `update.text`.
- **Long-running operations resume via `continuation_token`.** When a hosted backend pauses (background job, deferred completion), set `continuation_token=ContinuationToken(...)` on the update. The agent loop sees the token, stores it, and the consumer can resume by passing it back into the next `run(...)` call.

Round-tripping is symmetric — useful for testing, replay, and persisted streams:

```python
import json
from agent_framework import ChatResponseUpdate, Content

original = ChatResponseUpdate(
    contents=[Content.from_text("partial")],
    role="assistant",
    message_id="msg_42",
    additional_properties={"trace_id": "abc-123"},
)

# Dict round-trip — handy when you persist updates to a database.
restored_dict = ChatResponseUpdate.from_dict(original.to_dict())
assert restored_dict.text == "partial"
assert restored_dict.additional_properties["trace_id"] == "abc-123"

# JSON round-trip — DEFAULT_EXCLUDE drops `raw_representation`,
# so providers' SDK objects don't leak into your serialised stream.
encoded = original.to_json()
restored_json = ChatResponseUpdate.from_json(encoded)
assert restored_json.message_id == "msg_42"
assert "raw_representation" not in json.loads(encoded)
```

`AgentResponseUpdate` follows the same shape but adds `agent_id` and `author_name` (so multi-agent orchestrations can attribute each chunk to the right participant) — use it when emitting updates from a `BaseAgent` subclass like the `WorkflowBackedAgent` above.

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
import hashlib
from collections.abc import Sequence
from agent_framework import BaseEmbeddingClient, Embedding, GeneratedEmbeddings


class HashEmbeddingClient(BaseEmbeddingClient):
    """Toy deterministic embedding — useful for tests that need repeatable vectors.

    Uses SHA-256 so the vectors are stable across processes; Python's built-in
    ``hash()`` is salted per process for strings (PYTHONHASHSEED) and would
    produce different output on every run.
    """

    OTEL_PROVIDER_NAME = "hash-toy"

    async def get_embeddings(self, values, *, options=None):
        vectors = [
            Embedding(
                vector=list(hashlib.sha256(v.encode("utf-8")).digest()[:8]),
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
            self.source_id,
            [Message(role="system", contents=[f"<context>\n{s}\n</context>"]) for s in snippets],
        )
```

Compare with `HistoryProvider` (in the same module): use `HistoryProvider` for persistent conversation storage, use plain `ContextProvider` for one-shot context injection per run.

### Using provider-scoped `state` for cross-turn memory

`state` is the second-class citizen most provider implementations underuse. Each provider sees an isolated dict keyed by `source_id` — perfect for caching expensive lookups across turns of the same session, or for tracking cumulative spend without polluting the messages list.

```python
import time
from agent_framework import Agent, ContextProvider, Message
from agent_framework.openai import OpenAIChatClient


class UserProfileProvider(ContextProvider):
    """Fetch the user's profile once per session and reuse it on every turn.

    Stashing the profile in provider state avoids hitting the profile API on
    every agent.run() call. ``state`` is automatically scoped to this provider,
    so multiple providers can each maintain their own caches without colliding.
    """

    DEFAULT_SOURCE_ID = "user_profile"

    def __init__(self, profile_client, *, ttl: float = 3600) -> None:
        super().__init__(self.DEFAULT_SOURCE_ID)
        self._client = profile_client
        self._ttl = ttl

    async def before_run(self, *, agent, session, context, state) -> None:
        user_id = context.kwargs.get("user_id")
        if not user_id:
            return

        cache: dict = state.setdefault("cache", {})
        entry = cache.get(user_id)
        now = time.monotonic()

        if entry is None or now - entry["fetched_at"] > self._ttl:
            profile = await self._client.fetch(user_id)
            entry = {"profile": profile, "fetched_at": now}
            cache[user_id] = entry

        context.extend_instructions(
            self.source_id,
            f"User profile: name={entry['profile'].name}, plan={entry['profile'].plan}.",
        )

    async def after_run(self, *, agent, session, context, state) -> None:
        # Tally turn count against this user's profile entry.
        user_id = context.kwargs.get("user_id")
        if user_id:
            cache = state.setdefault("cache", {})
            if user_id in cache:
                cache[user_id]["turns"] = cache[user_id].get("turns", 0) + 1


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a personalised assistant.",
    context_providers=[UserProfileProvider(profile_client)],
)

session = agent.create_session(session_id="user-7")
await agent.run("What's my plan?", session=session, user_id="user-7")
await agent.run("Upgrade me.", session=session, user_id="user-7")  # cache hit
```

Note that the comments above emphasize:

- `state` is a **provider-scoped** dict — distinct from `session.state` (cross-provider). Mutate it freely without coordinating with other providers.
- `state` survives across `before_run` / `after_run` of the same session. The framework persists it through `session.to_dict()` if the values implement `SerializationProtocol`.
- For ephemeral per-run scratch space, use `context.metadata` instead — that dict is rebuilt every call.

### Combining `before_run` and `after_run` for a citation tracker

When you want the model to cite sources, the cleanest pattern is a single provider that injects retrieved snippets *before* the call and harvests citation IDs *after* it:

```python
import re
from agent_framework import ContextProvider, Message


class CitationProvider(ContextProvider):
    DEFAULT_SOURCE_ID = "citations"
    # Doc IDs may contain hyphens (UUIDs), dots, or colons depending on the
    # retriever — `\w+` alone would silently drop everything after the first
    # `-`, so accept the common URL-safe id alphabet here.
    CITE_PATTERN = re.compile(r"\[\[doc:([\w.:-]+)\]\]")

    def __init__(self, retriever) -> None:
        super().__init__(self.DEFAULT_SOURCE_ID)
        self._retriever = retriever

    async def before_run(self, *, agent, session, context, state) -> None:
        last_user = next(
            (m for m in reversed(context.input_messages) if m.role == "user"),
            None,
        )
        if not last_user:
            return
        docs = await self._retriever.search(last_user.text, top_k=5)
        # Stash the docs so after_run can correlate citations back to URLs.
        state.setdefault("docs_by_id", {}).update({d.id: d for d in docs})
        context.extend_messages(
            self.source_id,
            [
                Message(
                    role="system",
                    contents=[f"[[doc:{d.id}]] {d.title}\n{d.excerpt}"],
                )
                for d in docs
            ],
        )
        context.extend_instructions(
            self.source_id,
            "Cite sources using [[doc:ID]] markers.",
        )

    async def after_run(self, *, agent, session, context, state) -> None:
        if not context.response:
            return
        cited_ids = set()
        for msg in context.response.messages:
            for content in msg.contents:
                text = getattr(content, "text", None) or ""
                cited_ids.update(self.CITE_PATTERN.findall(text))
        # Surface a structured citations payload through session.state for the caller.
        session.state["last_citations"] = [
            {"id": cid, "url": state["docs_by_id"][cid].url}
            for cid in cited_ids
            if cid in state.get("docs_by_id", {})
        ]
```

After every run the caller can read `session.state["last_citations"]` to render footnotes alongside the agent's reply — no parsing of the model output needed in user code.

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

The currently gated features (verified against `agent-framework-core==1.4.0`):

| Stage | Enum member | Covers |
|---|---|---|
| Experimental | `ExperimentalFeature.SKILLS` | `Skill`, `SkillResource`, `SkillScript`, `SkillsProvider` |
| Experimental | `ExperimentalFeature.EVALS` | `LocalEvaluator`, `evaluate_agent`, `evaluate_workflow`, `@evaluator` |
| Experimental | `ExperimentalFeature.FUNCTIONAL_WORKFLOWS` | `@workflow`, `@step`, `RunContext`, `FunctionalWorkflow`, `FunctionalWorkflowAgent` |
| Experimental | `ExperimentalFeature.FILE_HISTORY` | `FileHistoryProvider` |
| Experimental | `ExperimentalFeature.TOOLBOXES` | Toolbox APIs (preview) |
| Release candidate | *(empty in 1.4.0)* | The enum exists but has no members in this release — features that previously sat at RC have either been promoted to stable or rolled back to experimental. |

Both enums are stage-scoped inventories, not stable introspection surfaces. Members move or disappear as features advance — the docstring on `ReleaseCandidateFeature` is explicit about this. Always enumerate at runtime when you need an authoritative list:

```python
from agent_framework import ExperimentalFeature, ReleaseCandidateFeature

print([f.value for f in ExperimentalFeature])
# ['EVALS', 'FILE_HISTORY', 'FUNCTIONAL_WORKFLOWS', 'SKILLS', 'TOOLBOXES']

print([f.value for f in ReleaseCandidateFeature])
# []  ← empty in 1.4.0; non-empty in releases that have RC features queued
```

Inspect any class or callable at runtime to see what stage it belongs to:

```python
from agent_framework import LocalEvaluator

print(LocalEvaluator.__feature_stage__)  # "experimental"
print(LocalEvaluator.__feature_id__)     # "EVALS"
```

Use this in CI to fail the build if anyone imports an experimental API without an explicit opt-in:

```python
# tests/test_no_unstable_imports.py
import importlib
import pytest


PROD_MODULES = ["my_app.agents", "my_app.workflows", "my_app.tools"]
STABLE_STAGES = {None, "stable"}


@pytest.mark.parametrize("modname", PROD_MODULES)
def test_module_uses_only_stable_apis(modname: str) -> None:
    mod = importlib.import_module(modname)
    for name in dir(mod):
        obj = getattr(mod, name)
        # Use getattr with a default — the attribute disappears when a
        # feature is promoted to stable, so absence means "stable".
        stage = getattr(obj, "__feature_stage__", None)
        assert stage in STABLE_STAGES, (
            f"{modname}.{name} is at stage={stage!r} — wrap usage in an "
            "explicit opt-in so the team knows it can change without notice."
        )
```

When you genuinely want to depend on a staged API, silence its specific warning rather than the whole `FeatureStageWarning` category — that way you still hear about *new* staged APIs you accidentally pull in:

```python
import warnings
from agent_framework._feature_stage import ExperimentalWarning

# Allow `LocalEvaluator` (EVALS) but keep every other staged warning loud.
warnings.filterwarnings(
    "ignore",
    message=r"\[EVALS\] .*",
    category=ExperimentalWarning,
)
```

## Functional workflows — `@workflow` and `@step`

Graph-based `WorkflowBuilder` is great for fan-out/fan-in topologies that need explicit edges, but for the *vast majority* of pipelines you really just want to write Python. The `@workflow` decorator (experimental, gated behind `ExperimentalFeature.FUNCTIONAL_WORKFLOWS`) lets you do exactly that — native `if`/`else`, `for`, `asyncio.gather`, whatever — while keeping the framework's HITL, checkpointing, and event stream on tap.

### The minimum viable workflow

```python
import asyncio
from agent_framework import workflow


@workflow
async def my_pipeline(data: str) -> str:
    return data.upper()


result = asyncio.run(my_pipeline.run("hello"))
print(result.get_outputs())  # ['HELLO']
```

`my_pipeline` is now a `FunctionalWorkflow` instance. It exposes the same `run(message, *, stream=False, responses=..., checkpoint_id=...)` surface as graph workflows and returns a `WorkflowRunResult`.

A workflow function takes **at most one** non-`RunContext` parameter. That single parameter is the message you pass to `.run()`. Bundle multiple inputs into a dict or Pydantic model.

### Adding `@step` for caching, checkpointing, and events

Plain async functions work inside `@workflow`, but wrapping them in `@step` gives you per-call caching keyed on `(step_name, call_index)`. On HITL replay or checkpoint restore, completed steps are skipped instead of re-run.

```python
import json
from agent_framework import workflow, step


@step
async def fetch(url: str) -> dict:
    # Real HTTP call — would re-run on every replay without @step.
    return await http_get(url)


@step(name="transform")  # parameterised form: explicit display name
async def transform(raw: dict) -> str:
    return json.dumps(raw)


@workflow
async def pipeline(url: str) -> str:
    raw = await fetch(url)
    return await transform(raw)
```

The framework only re-executes `fetch` and `transform` once per logical call site. Loops keep distinct cache keys via the call counter, so `for url in urls: await fetch(url)` works as expected.

#### What `@step` actually wraps — `StepWrapper`

The `@step` decorator returns a `StepWrapper`, which is transparent outside a workflow context (call it from a unit test and it just runs the function) and switched-on inside one (caching + lifecycle events + `RunContext` injection + per-step checkpointing). A few practical consequences:

```python
from agent_framework import workflow, step

@step
async def expensive(x: int) -> int:
    print(f"running expensive({x})")
    return x * x


# Outside a workflow: behaves like a plain async function. No caching.
import asyncio
asyncio.run(expensive(5))         # prints "running expensive(5)" → returns 25
asyncio.run(expensive(5))         # prints "running expensive(5)" again — no cache


@workflow
async def pipeline(x: int) -> int:
    a = await expensive(x)        # call_index=0 → cache miss, runs
    b = await expensive(x + 1)    # call_index=1 → distinct key, runs
    return a + b


# Inside a workflow: each call site is cached on (step_name, call_index).
# A second `pipeline.run(...)` with the same input rebuilds the cache from
# scratch — caching is per run, not global.
```

The cache is keyed on `(step_name, call_index)`, **not on arguments**. That means:

- A second call to the same step inside the same run is treated as a fresh call site (it gets `call_index=1`) and will execute again, even if the arguments are identical.
- Arguments are recorded in the cache value so HITL replay and checkpoint restore can deliver the recorded result without re-running the function. After a `request_info()` suspension, the framework replays the workflow function from the top and serves cached results until it reaches the call site that suspended.
- `executor_bypassed` events are emitted on cache hits so you can spot replay noise in observability dashboards. `executor_invoked` / `executor_completed` events fire only on live calls.

If a step needs to *always* run (e.g., it depends on the wall clock or external state), don't decorate it with `@step` — use a plain async function inside the `@workflow` body. Or use a name that varies per call:

```python
import time
from agent_framework import step

@step  # explicit name keeps the cache deterministic
async def measure_now(label: str) -> float: ...

# Want every call uncached? Skip the decorator:
async def now_uncached() -> float:
    return time.time()
```

### `RunContext` — HITL, custom events, and per-run state

When a workflow needs human-in-the-loop input, custom events, or key/value state, declare a `RunContext` parameter (by type annotation, or by parameter name `ctx`):

```python
from agent_framework import workflow, RunContext


@workflow
async def review_workflow(draft: str, ctx: RunContext) -> str:
    feedback = await ctx.request_info(
        request_data={"draft": draft},
        response_type=str,
    )
    ctx.set_state("last_feedback", feedback)
    return f"{draft}\n\nReviewer says: {feedback}"
```

On the first call to `request_info` the workflow **suspends** by raising an internal interruption signal (caught by the framework). The caller receives a `WorkflowRunResult` whose `get_request_info_events()` lists the pending requests. To resume, pass `responses={request_id: value}`:

```python
async def main():
    first = await review_workflow.run("draft v1")
    pending = first.get_request_info_events()
    request_id = pending[0].request_id

    final = await review_workflow.run(
        responses={request_id: "Tighten the third paragraph."},
    )
    print(final.get_outputs()[-1])
```

The same `request_id` round-trip works under streaming too — pass `stream=True` and consume `WorkflowEvent`s from the returned `ResponseStream`.

`RunContext` also exposes:

- `add_event(event)` — emit application-specific events alongside framework lifecycle events. Useful for progress reporting and trace correlation.
- `get_state(key, default=None)` / `set_state(key, value)` — workflow-scoped key/value state. Persisted across HITL pauses and saved into checkpoints when checkpoint storage is configured. Keys starting with `_` are reserved.
- `is_streaming()` — whether the current run was started with `stream=True`.
- `get_run_context()` (module-level helper) — fetch the active `RunContext` from any function called transitively from a workflow, even when you didn't thread `ctx` through every signature.

### Hooking up checkpointing

Pass `checkpoint_storage=` to the decorator or to `.run()`. The framework checkpoints after each completed step:

```python
from agent_framework import workflow, step, FileCheckpointStorage

storage = FileCheckpointStorage("/var/lib/checkpoints")


@workflow(name="research", checkpoint_storage=storage)
async def research(topic: str) -> str:
    notes = await gather_notes(topic)
    summary = await summarise(notes)
    return summary


# Resume from the latest checkpoint of this workflow definition.
latest = await storage.get_latest(workflow_name="research")
if latest is not None:
    result = await research.run(checkpoint_id=latest.checkpoint_id)
```

Checkpoints are keyed on the workflow definition (name + `graph_signature_hash`), not on a specific instance — any process running the same `@workflow` function can resume them.

### Adapting a functional workflow as an agent

`FunctionalWorkflow.as_agent()` returns a `FunctionalWorkflowAgent` that satisfies `SupportsAgentRun`. Pending HITL requests surface as `FunctionApprovalRequestContent` items, exactly like the graph `WorkflowAgent`:

```python
agent = review_workflow.as_agent(name="reviewer")
response = await agent.run("Please review this draft.")
```

The adapter is the right entry point when you want a functional pipeline to plug into a multi-agent supervisor or be used wherever `Agent`-shaped objects are expected.

### When to choose functional vs graph workflows

| Choose `@workflow` when | Choose `WorkflowBuilder` when |
|---|---|
| Logic is mostly sequential or uses standard Python control flow | You need explicit fan-out / fan-in / switch-case topology |
| Branching depends on runtime values that are awkward to express as edges | You want the topology validated up-front (type compatibility, connectivity) |
| You want minimal ceremony — one decorator, one async function | You're handing the workflow to a non-Python visualisation or runtime |
| You need ad-hoc parallelism (`asyncio.gather`) inside the workflow | You want declarative graph export (e.g., for `WorkflowViz`) |

Functional workflows are experimental — silence the `ExperimentalWarning` with the patterns from the previous section, or pin them in CI to prove no surprise breakage.

## Building messages with `Content` factory methods

`Content` is the **single unified container** for every variant the framework knows about — text, images, function calls, tool results, errors, hosted-file references, MCP tool calls, shell commands, OAuth consent prompts. You almost never construct it via the `__init__`; use the factory class methods so you don't have to remember which `__init__` keyword goes with which content type.

```python
from agent_framework import Content

# Plain text
text = Content.from_text("Hello world.")

# Reasoning trace (for models that emit chain-of-thought tokens separately)
reasoning = Content.from_text_reasoning("Let me think step by step…")

# Inline binary data (image, audio, file). Encoded into a data: URI internally.
with open("chart.png", "rb") as f:
    image = Content.from_data(f.read(), media_type="image/png")

# Linked external resource
remote_image = Content.from_uri(
    "https://example.com/chart.png", media_type="image/png"
)

# Hosted file already uploaded to the provider
hosted = Content.from_hosted_file(file_id="file_abc123", media_type="application/pdf")
hosted_vector_store = Content.from_hosted_vector_store(vector_store_id="vs_xyz")

# Function/tool calls — the framework emits these for you, but you can construct
# them when replaying transcripts or building eval fixtures.
call = Content.from_function_call("call_42", "get_weather", arguments={"city": "Seattle"})
result = Content.from_function_result("call_42", result={"temp_c": 18, "conditions": "sunny"})

# Error from a failed tool, model call, or upstream service
err = Content.from_error(message="Rate limit exceeded", error_code="429")

# Token usage record (attached to a Message for observability pipelines)
from agent_framework import UsageDetails
usage = Content.from_usage(UsageDetails(input_token_count=120, output_token_count=80))
```

The `media_type` on `from_data` / `from_uri` is what providers and middleware key on to pick the right transport (vision input vs. attachment vs. inline blob). Always set it.

For tool integrations the framework ships with, there are dedicated factories that capture provider-specific payload shapes:

| Factory | Returns content type |
|---|---|
| `Content.from_search_tool_call` / `from_search_tool_result` | Hosted web/file search calls |
| `Content.from_code_interpreter_tool_call` / `from_code_interpreter_tool_result` | Hosted code interpreter calls |
| `Content.from_image_generation_tool_call` / `from_image_generation_tool_result` | Hosted image generation calls |
| `Content.from_mcp_server_tool_call` / `from_mcp_server_tool_result` | MCP tool invocations |
| `Content.from_shell_tool_call` / `from_shell_tool_result` / `from_shell_command_output` | Shell tool invocations and their stdout/stderr/exit_code records |
| `Content.from_function_approval_request` / `from_function_approval_response` | HITL approval gates around tool calls |
| `Content.from_oauth_consent_request` | OAuth consent prompts forwarded to the user |

Round-trip serialisation works on every variant via `Content.to_dict()` / `Content.from_dict(...)`, so you can persist a transcript and hydrate it later without losing structure. Add provider-specific extras through `additional_properties=` and keep the original SDK payload on `raw_representation=` if you need to debug provider quirks.

## Exception hierarchy

Every framework exception inherits from `AgentFrameworkException`, which logs the message at debug level on construction (turn it off per-call with `log_level=None`). The hierarchy mirrors the layers of the framework, so you can catch as broadly or as narrowly as makes sense for your call site.

```
AgentFrameworkException                    # base — every framework error
├── AgentException                         # raised inside an Agent.run loop
│   ├── AgentInvalidAuthException
│   ├── AgentInvalidRequestException
│   ├── AgentInvalidResponseException
│   └── AgentContentFilterException
├── ChatClientException                    # raised by chat clients
│   ├── ChatClientInvalidAuthException
│   ├── ChatClientInvalidRequestException
│   ├── ChatClientInvalidResponseException
│   └── ChatClientContentFilterException
├── IntegrationException                   # external service / dependency
│   ├── IntegrationInitializationError
│   ├── IntegrationInvalidAuthException
│   ├── IntegrationInvalidRequestException
│   ├── IntegrationInvalidResponseException
│   └── IntegrationContentFilterException
├── ContentError
│   └── AdditionItemMismatch              # mismatched types when merging streamed chunks
├── ToolException
│   ├── ToolExecutionException
│   └── UserInputRequiredException        # tool needs HITL input to proceed
├── MiddlewareException
│   └── MiddlewareTermination             # control-flow exit from a middleware pipeline
├── SettingNotFoundError                   # config not resolvable from env / kwargs
└── WorkflowException
    ├── WorkflowValidationError
    │   ├── EdgeDuplicationError
    │   ├── TypeCompatibilityError
    │   └── GraphConnectivityError
    └── WorkflowRunnerException
        ├── WorkflowConvergenceException
        └── WorkflowCheckpointException
```

`AgentFrameworkException.__init__` accepts an `inner_exception=` and a `log_level=` (default `logging.DEBUG = 10`, set to `None` to skip logging). Most framework code wraps third-party exceptions with these so you keep both the high-level reason and the underlying cause:

```python
import logging
from agent_framework import (
    AgentFrameworkException,
    MiddlewareException,
    MiddlewareTermination,
)

log = logging.getLogger(__name__)

try:
    response = await agent.run("Hello")
except MiddlewareTermination as exc:
    # Short-circuit from a middleware: exc.result holds whatever was passed in.
    fallback_text = getattr(exc, "result", None) or "Request blocked."
except MiddlewareException as exc:
    # Anything else gone wrong inside the middleware pipeline.
    log.exception("middleware failure")
except AgentFrameworkException as exc:
    # Last-resort catch — keeps you out of provider-specific exception classes.
    log.exception("framework failure: %s", exc)
```

`UserInputRequiredException` is special: it's raised by tool middleware when a wrapped sub-agent needs HITL input, and carries the request `Content` items on `exc.contents` so the outer agent can surface them instead of swallowing them as a generic tool error.

## Workflow validation errors

`WorkflowBuilder.build()` runs a graph validator before returning a `Workflow`. Failures raise `WorkflowValidationError` subclasses with a `validation_type: ValidationTypeEnum` field, so you can branch on the kind of validation problem programmatically:

```python
from agent_framework import (
    EdgeDuplicationError,
    GraphConnectivityError,
    TypeCompatibilityError,
    ValidationTypeEnum,
    WorkflowBuilder,
    WorkflowValidationError,
)

try:
    workflow = builder.build()
except EdgeDuplicationError as exc:
    print(f"duplicate edge: {exc.edge_id}")
except TypeCompatibilityError as exc:
    print(
        f"{exc.source_executor_id} → {exc.target_executor_id}: "
        f"source emits {exc.source_types}, target accepts {exc.target_types}"
    )
except GraphConnectivityError as exc:
    print(f"connectivity: {exc.message}")
except WorkflowValidationError as exc:
    # Generic catch-all — exc.validation_type tells you which check fired.
    if exc.validation_type is ValidationTypeEnum.HANDLER_OUTPUT_ANNOTATION:
        ...
```

The validation types currently emitted:

| `ValidationTypeEnum` member | Trigger |
|---|---|
| `EDGE_DUPLICATION` | Two identical `(source, target)` edges added to the builder |
| `EXECUTOR_DUPLICATION` | Two executors registered under the same id |
| `TYPE_COMPATIBILITY` | Source executor's output types and target's handler types are disjoint |
| `GRAPH_CONNECTIVITY` | Unreachable executors, missing start, or no path to an output node |
| `HANDLER_OUTPUT_ANNOTATION` | Executor handler missing the output-type annotation the validator needs |
| `OUTPUT_VALIDATION` | Declared output executors don't actually emit the workflow's declared output type |

`str(exc)` on any of these is formatted as `[VALIDATION_TYPE] human-readable message`, which is what surfaces in test failures and CI logs.

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

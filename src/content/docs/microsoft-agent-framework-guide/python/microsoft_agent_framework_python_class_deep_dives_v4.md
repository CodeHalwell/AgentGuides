---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 4"
description: "Source-verified deep dives into 10 classes from agent-framework 1.7.0: Message + Content, ChatOptions + ChatResponse + ChatResponseUpdate + UsageDetails, ResponseStream, AgentContext, FunctionalWorkflow + StepWrapper, WorkflowEvent taxonomy, SkillsSource composition layer, EvalItem + EvalResults + EvalScoreResult, TokenizerProtocol + CharacterEstimatorTokenizer, ConversationSplit + ConversationSplitter."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 23
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 4

Verified against **agent-framework-core 1.7.0** (installed June 2026). Every constructor signature,
parameter description, and code example was derived from the installed package source at
`/tmp/agent_fw/agent_framework/`. No API name has been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`

This volume fills gaps across four areas: the **core message/content model**, the **raw chat layer**,
the **streaming and middleware contracts**, and the **evaluation data model**.

---

## Table of Contents

1. [`Message` + `Content`](#1-message--content)
2. [`ChatOptions` + `ChatResponse` + `ChatResponseUpdate` + `UsageDetails`](#2-chatoptions--chatresponse--chatresponseupdate--usagedetails)
3. [`ResponseStream`](#3-responsestream)
4. [`AgentContext`](#4-agentcontext)
5. [`FunctionalWorkflow` + `StepWrapper`](#5-functionalworkflow--stepwrapper)
6. [`WorkflowEvent` + `WorkflowEventType` + `WorkflowEventSource`](#6-workflowevent--workfloweventtype--workfloweventsource)
7. [`SkillsSource` + `AggregatingSkillsSource` + `FilteringSkillsSource` + `DeduplicatingSkillsSource`](#7-skillssource--aggregatingskillssource--filteringskillssource--deduplicatingskillssource)
8. [`EvalItem` + `EvalItemResult` + `EvalResults` + `EvalScoreResult`](#8-evalitem--evalitemresult--evalresults--evalscoreresult)
9. [`TokenizerProtocol` + `CharacterEstimatorTokenizer`](#9-tokenizerprotocol--characterestimatortokenizer)
10. [`ConversationSplit` + `ConversationSplitter`](#10-conversationsplit--conversationsplitter)

---

## 1. `Message` + `Content`

**Source:** `agent_framework/_types.py`

`Message` and `Content` are the two lowest-level building blocks of the framework.
Every prompt you send, every response you receive, and every tool call the agent makes is
represented as a `Message` containing one or more `Content` items.

### `Message`

```python
Message(
    role: RoleLiteral | str,
    contents: Sequence[Content | str | Mapping[str, Any]] | None = None,
    *,
    author_name: str | None = None,
    message_id: str | None = None,
    additional_properties: MutableMapping[str, Any] | None = None,
    raw_representation: Any | None = None,
)
```

| Parameter | Description |
|---|---|
| `role` | `"user"`, `"assistant"`, `"system"`, or `"tool"`. Arbitrary strings are accepted. |
| `contents` | Strings are auto-coerced to `Content.from_text()`; dicts go through `Content.from_dict()`. |
| `author_name` | Optional display name of the author — propagated by multi-agent orchestration. |
| `message_id` | Optional stable identifier (used for compaction bookkeeping). |
| `additional_properties` | Internal metadata dict; **not** forwarded to the model provider. |

Key properties:

| Property | Type | Description |
|---|---|---|
| `text` | `str` | Concatenated text of all `TextContent` items in `contents`. |
| `role` | `str` | Role string. |
| `contents` | `list[Content]` | All content items after coercion. |
| `author_name` | `str \| None` | Author name if set. |

```python
from agent_framework import Message, Content

# Simplest form — string auto-coerced to TextContent
user_msg = Message("user", ["Tell me about asyncio."])
print(user_msg.text)  # "Tell me about asyncio."

# Multi-content message: text + image
multimodal = Message(
    "user",
    [
        "What's in this image?",
        Content.from_image_uri("https://example.com/chart.png", media_type="image/png"),
    ],
)
print(len(multimodal.contents))  # 2

# System message
system = Message("system", ["You are a concise assistant."])

# Serialization round-trip
msg_dict = user_msg.to_dict()
# {'type': 'chat_message', 'role': 'user', 'contents': [...], 'additional_properties': {}}
restored = Message.from_dict(msg_dict)
assert restored.text == user_msg.text

msg_json = user_msg.to_json()
restored_json = Message.from_json(msg_json)
assert restored_json.role == "user"
```

### `Content`

`Content` is a unified container for every content variant. Prefer the factory classmethods over
calling `__init__` directly — they set the correct `type` discriminator and populate only the
relevant fields.

**Factory methods:**

| Method | ContentType | Key field(s) |
|---|---|---|
| `Content.from_text(text)` | `"text"` | `text` |
| `Content.from_data(data, media_type)` | `"data"` | `uri` (base64), `media_type` |
| `Content.from_uri(uri, media_type)` | `"uri"` | `uri`, `media_type` |
| `Content.from_image_uri(uri, media_type)` | `"uri"` | shortcut for images |
| `Content.from_function_call(call_id, name, arguments)` | `"function_call"` | `call_id`, `name`, `arguments` |
| `Content.from_function_result(call_id, name, result)` | `"function_result"` | `call_id`, `name`, `result` |
| `Content.from_error(message, error_code, error_details)` | `"error"` | `message`, `error_code` |
| `Content.from_usage(usage_details)` | `"usage"` | `usage_details` |

```python
from agent_framework import Content

# Text
text_c = Content.from_text("Hello world")
print(text_c.type)   # "text"
print(text_c.text)   # "Hello world"

# Image from URL
img = Content.from_image_uri("https://example.com/photo.jpg", media_type="image/jpeg")
print(img.type)   # "uri"
print(img.uri)    # "https://example.com/photo.jpg"

# Inline binary data (e.g. screenshot bytes) — from_data handles base64 encoding internally
raw = b"\x89PNG..."
img_data = Content.from_data(data=raw, media_type="image/png")
print(img_data.type)       # "data"
print(img_data.media_type) # "image/png"

# Tool call / result round-trip
call = Content.from_function_call(call_id="c1", name="get_weather", arguments='{"city":"London"}')
result = Content.from_function_result(call_id="c1", name="get_weather", result="Rainy, 12°C")
print(call.type)    # "function_call"
print(result.type)  # "function_result"

# Serialization
d = text_c.to_dict()
# {'type': 'text', 'text': 'Hello world'}
round_tripped = Content.from_dict(d)
assert round_tripped.text == "Hello world"
```

### Building a conversation manually

```python
import asyncio
from agent_framework import Agent, Message, Content
from agent_framework.openai import OpenAIChatClient

async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
    )

    history: list[Message] = [
        Message("user", ["What is Python?"]),
        Message("assistant", ["Python is a high-level programming language."]),
        Message("user", ["What are its main uses?"]),
    ]

    response = await agent.run(history)
    print(response.text)

asyncio.run(main())
```

---

## 2. `ChatOptions` + `ChatResponse` + `ChatResponseUpdate` + `UsageDetails`

**Source:** `agent_framework/_types.py`

These four types form the **raw chat layer** that sits directly below `Agent.run()`.
You interact with them when writing chat middleware, custom clients, or calling a
`BaseChatClient` directly.

### `ChatOptions`

`ChatOptions` is an open `TypedDict` (all keys `total=False`) covering the common denominator
of options supported by every provider:

| Key | Type | Description |
|---|---|---|
| `model` | `str` | Override the default model for this request. |
| `temperature` | `float` | Sampling temperature. |
| `top_p` | `float` | Nucleus sampling threshold. |
| `max_tokens` | `int` | Maximum tokens to generate. |
| `stop` | `str \| Sequence[str]` | Stop sequences. |
| `seed` | `int` | Deterministic seed (provider-specific support). |
| `logit_bias` | `dict[str \| int, float]` | Logit bias map. |
| `frequency_penalty` | `float` | Frequency penalty. |
| `presence_penalty` | `float` | Presence penalty. |
| `tools` | `ToolTypes \| ...` | Per-request tool override. |
| `tool_choice` | `ToolMode \| "auto" \| "required" \| "none"` | Tool selection mode. |
| `allow_multiple_tool_calls` | `bool` | Allow multiple tool calls per turn. |
| `response_format` | `type[BaseModel] \| Mapping[str, Any] \| None` | Structured output schema. |
| `metadata` | `dict[str, Any]` | Request metadata. |
| `user` | `str` | End-user identifier (for provider abuse monitoring). |
| `store` | `bool` | Store the conversation on the provider side. |
| `conversation_id` | `str` | Conversation identifier (multi-turn continuity). |
| `instructions` | `str` | Per-request system prompt override. |

Provider-specific TypedDicts extend `ChatOptions` with additional keys.

```python
from agent_framework import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient

agent = Agent(client=OpenAIChatClient(), instructions="Be concise.")

# Pass options at run-time
options: ChatOptions = {"temperature": 0.2, "max_tokens": 200, "model": "gpt-4o-mini"}

async def main():
    response = await agent.run("Summarize asyncio.", options=options)
    print(response.text)
```

### `ChatResponse`

`ChatResponse` is the final aggregated response from a `BaseChatClient`.

```python
ChatResponse(
    messages: list[Message] | None = None,
    *,
    response_id: str | None = None,
    conversation_id: str | None = None,
    model: str | None = None,
    created_at: datetime | str | None = None,
    finish_reason: FinishReasonLiteral | FinishReason | None = None,
    usage_details: UsageDetails | None = None,
    structured_output: ResponseModelT | None = None,
    additional_properties: MutableMapping[str, Any] | None = None,
    raw_representation: Any | None = None,
)
```

| Property/Method | Description |
|---|---|
| `text` | Concatenated text of all assistant messages in the response. |
| `messages` | Full list of `Message` objects in the response. |
| `usage_details` | Token counts — see `UsageDetails`. |
| `finish_reason` | `"stop"`, `"length"`, `"tool_calls"`, `"content_filter"`, etc. |
| `model` | Model string returned by the provider. |
| `structured_output` | Populated when `response_format` was a Pydantic model. |
| `ChatResponse.from_updates(updates)` | Class method — reassemble from a list of `ChatResponseUpdate` chunks. |

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework import Message

async def raw_chat_example():
    client = OpenAIChatClient()
    messages = [Message("user", ["What is 2 + 2?"])]

    response = await client.get_chat_message_content(messages)
    print(response.text)                        # "4"
    print(response.finish_reason)               # "stop"
    print(response.usage_details)              # {'input_token_count': ..., 'output_token_count': ...}
    print(response.model)                       # "gpt-4o"

asyncio.run(raw_chat_example())
```

### `ChatResponseUpdate`

`ChatResponseUpdate` is a single streaming chunk from `get_streaming_chat_message_content`.

```python
ChatResponseUpdate(
    *,
    contents: Sequence[Content] | None = None,
    role: RoleLiteral | Role | None = None,
    author_name: str | None = None,
    response_id: str | None = None,
    message_id: str | None = None,
    conversation_id: str | None = None,
    model: str | None = None,
    created_at: ...,
    finish_reason: FinishReasonLiteral | FinishReason | None = None,
    ...
)
```

| Property | Description |
|---|---|
| `text` | Text portion of this chunk's `contents`. |
| `contents` | Content items in this chunk. |
| `finish_reason` | Non-None on the final chunk (e.g. `"stop"`). |
| `usage_details` | Token usage — typically populated on the final chunk only. |

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework import Message, ChatResponse

async def stream_raw():
    client = OpenAIChatClient()
    messages = [Message("user", ["Count to five."])]

    stream = await client.get_streaming_chat_message_content(messages)
    updates = []
    async for update in stream:
        print(update.text, end="", flush=True)
        updates.append(update)

    # Reassemble into a ChatResponse
    final = ChatResponse.from_updates(updates)
    print()
    print(f"finish_reason: {final.finish_reason}")
    print(f"tokens: {final.usage_details}")

asyncio.run(stream_raw())
```

### `UsageDetails`

`UsageDetails` is an open `TypedDict` (`total=False`) carrying token-count metadata:

```python
UsageDetails(
    input_token_count: int | None,
    output_token_count: int | None,
    total_token_count: int | None,
    # ... provider-specific extra keys
)
```

Because it extends `TypedDict` with `total=False` and `extra_items=int`, providers can add
arbitrary integer fields (e.g. `"cached_tokens"`, `"reasoning_tokens"`) without breaking
type-checking.

```python
from agent_framework import UsageDetails

# Reading usage from an agent response
async def log_usage(agent, query: str) -> None:
    response = await agent.run(query)
    usage: UsageDetails = response.usage_details or {}
    print(f"Input tokens:  {usage.get('input_token_count')}")
    print(f"Output tokens: {usage.get('output_token_count')}")
    print(f"Total tokens:  {usage.get('total_token_count')}")
    # Provider-specific extras (if present)
    if "cached_tokens" in usage:
        print(f"Cached tokens: {usage['cached_tokens']}")
```

---

## 3. `ResponseStream`

**Source:** `agent_framework/_types.py` — `ResponseStream(AsyncIterable[UpdateT], Generic[UpdateT, FinalT])`

`ResponseStream` is the async streaming abstraction used throughout the framework for both
the chat layer (`ChatResponseUpdate → ChatResponse`) and the agent layer
(`AgentResponseUpdate → AgentResponse`). It is a single-consume async iterable with optional
**transform hooks**, **cleanup hooks**, and **result hooks** that let middleware inject
side-effects without subclassing.

### Constructor

```python
ResponseStream(
    stream: AsyncIterable[UpdateT] | Awaitable[AsyncIterable[UpdateT]],
    *,
    finalizer: Callable[[Sequence[UpdateT]], FinalT | Awaitable[FinalT]] | None = None,
    transform_hooks: list[Callable[[UpdateT], UpdateT | Awaitable[UpdateT | None] | None]] | None = None,
    cleanup_hooks: list[Callable[[], Awaitable[None] | None]] | None = None,
    result_hooks: list[Callable[[FinalT], FinalT | Awaitable[FinalT | None] | None]] | None = None,
)
```

| Parameter | Description |
|---|---|
| `stream` | Source of `UpdateT` chunks. May be an awaitable that resolves to the iterable. |
| `finalizer` | Called with all collected updates after iteration; produces the `FinalT` result. |
| `transform_hooks` | Per-chunk transforms applied in order as each `UpdateT` is yielded. |
| `cleanup_hooks` | Run after all chunks are yielded and before the finalizer. Use for releasing resources. |
| `result_hooks` | Post-finalizer transforms applied to the `FinalT` result in order. |

### Key methods

| Method | Description |
|---|---|
| `async for update in stream:` | Iterate over `UpdateT` chunks. |
| `await stream.get_final_response()` | Consume remaining chunks and call the finalizer; returns `FinalT`. |
| `stream.map(transform, finalizer)` | Chain a new stream that transforms each update and uses a new finalizer. |
| `stream.with_finalizer(finalizer)` | Swap in a different finalizer without changing the update type. |
| `ResponseStream.from_awaitable(awaitable)` | Wrap an `Awaitable[ResponseStream]` for deferred construction. |

The `transform_hooks`, `cleanup_hooks`, and `result_hooks` are how agent middleware participates
in the streaming pipeline without wrapping the entire stream — they are composed into the stream
at middleware registration time.

### Usage patterns

**Pattern 1 — iterate and then get the final result:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def streaming_example():
    agent = Agent(client=OpenAIChatClient(), instructions="Be concise.")

    stream = await agent.run("List three planets.", stream=True)
    async for update in stream:
        print(update.text, end="", flush=True)

    final = await stream.get_final_response()
    print(f"\nFinish reason: {final.finish_reason}")
    print(f"Tokens: {final.usage_details}")

asyncio.run(streaming_example())
```

**Pattern 2 — transform hook for token logging:**

```python
from agent_framework import AgentMiddleware, AgentContext, ResponseStream, AgentResponseUpdate

class TokenLogMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        await call_next()

        if context.stream and isinstance(context.result, ResponseStream):
            token_counts: list[int] = []

            def count_tokens(update: AgentResponseUpdate) -> AgentResponseUpdate:
                if update.usage_details:
                    count = update.usage_details.get("output_token_count") or 0
                    token_counts.append(count)
                return update

            context.result._transform_hooks.append(count_tokens)
```

**Pattern 3 — map to a different type:**

```python
from agent_framework import ResponseStream, ChatResponseUpdate, ChatResponse

async def map_example(chat_stream: ResponseStream):
    # Transform raw chat updates into a simpler string stream
    text_stream = chat_stream.map(
        transform=lambda u: u.text,
        finalizer=lambda texts: "".join(texts),
    )
    async for text_chunk in text_stream:
        print(text_chunk, end="")
    full_text = await text_stream.get_final_response()
    print(f"\nComplete: {full_text!r}")
```

**Pattern 4 — cleanup hook for resource release:**

```python
import asyncio
from agent_framework import ResponseStream

async def stream_with_cleanup_hook(raw_stream):
    connection = await acquire_db_connection()

    async def release_connection():
        await connection.close()

    raw_stream._cleanup_hooks.append(release_connection)

    async for update in raw_stream:
        yield update  # connection is released after the last update
```

---

## 4. `AgentContext`

**Source:** `agent_framework/_middleware.py`

`AgentContext` is the dataclass passed through the `AgentMiddleware` pipeline. It carries the
full invocation request, is mutated by each `call_next()`, and exposes the result after execution.
Writing effective agent middleware requires understanding every field.

### Constructor

```python
AgentContext(
    *,
    agent: SupportsAgentRun,
    messages: list[Message],
    session: AgentSession | None = None,
    tools: ToolTypes | Callable | Sequence[...] | None = None,
    options: Mapping[str, Any] | None = None,
    stream: bool = False,
    compaction_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    metadata: Mapping[str, Any] | None = None,
    result: AgentResponse | ResponseStream[AgentResponseUpdate, AgentResponse] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    client_kwargs: Mapping[str, Any] | None = None,
    function_invocation_kwargs: Mapping[str, Any] | None = None,
    stream_transform_hooks: Sequence[...] | None = None,
    stream_result_hooks: Sequence[...] | None = None,
    stream_cleanup_hooks: Sequence[...] | None = None,
)
```

| Field | Mutable? | Description |
|---|---|---|
| `agent` | No | The `Agent` or `RawAgent` being invoked. |
| `messages` | Yes | The input message list for this turn. Middleware can prepend/append. |
| `session` | No | Active `AgentSession` (if any). |
| `tools` | Yes | Per-run tool override. Set to add/remove tools for this turn. |
| `options` | Yes | `ChatOptions`-compatible dict. Merge to change model, temperature, etc. |
| `stream` | No | Whether `agent.run(..., stream=True)` was called. |
| `compaction_strategy` | Yes | Per-run compaction override. |
| `tokenizer` | Yes | Per-run tokenizer override. |
| `metadata` | Yes | Shared scratchpad for passing data between middleware layers. |
| `result` | Yes | Populated after `call_next()` — `AgentResponse` or `ResponseStream`. |
| `client_kwargs` | Yes | Passed verbatim to the underlying chat client. |
| `function_invocation_kwargs` | Yes | Forwarded to tool invocation. |
| `stream_transform_hooks` | Yes | Transform hooks injected into the `ResponseStream`. |
| `stream_result_hooks` | Yes | Result hooks injected into the `ResponseStream`. |
| `stream_cleanup_hooks` | Yes | Cleanup hooks injected into the `ResponseStream`. |

### Common middleware patterns using `AgentContext`

**Logging middleware — inspect request and response:**

```python
import time
from agent_framework import AgentMiddleware, AgentContext, AgentResponse

class TimingMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        start = time.monotonic()
        context.metadata["start_time"] = start

        print(f"[{context.agent.name}] turn start — {len(context.messages)} messages")
        await call_next()

        elapsed = time.monotonic() - start
        if isinstance(context.result, AgentResponse):
            print(
                f"[{context.agent.name}] turn done in {elapsed:.2f}s — "
                f"{context.result.usage_details}"
            )
```

**Short-circuit middleware — return early without calling the agent:**

```python
from agent_framework import AgentMiddleware, AgentContext, AgentResponse, Message

BLOCKED_PATTERNS = ["hack", "exploit"]

class SafetyMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        last_text = context.messages[-1].text.lower() if context.messages else ""
        if any(p in last_text for p in BLOCKED_PATTERNS):
            context.result = AgentResponse(
                messages=[Message("assistant", ["I can't help with that."])],
                finish_reason="content_filter",
            )
            return  # skip call_next entirely

        await call_next()
```

**Per-turn model override:**

```python
from agent_framework import AgentMiddleware, AgentContext

class CheapModelForShortQueriesMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        last_text = context.messages[-1].text if context.messages else ""
        if len(last_text) < 100:
            # Shallow-copy options to avoid mutating upstream mapping
            context.options = dict(context.options or {})
            context.options["model"] = "gpt-4o-mini"
        await call_next()
```

**Injecting streaming hooks via `AgentContext`:**

```python
from agent_framework import AgentMiddleware, AgentContext, AgentResponseUpdate, ResponseStream

class StreamAuditMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        chunks: list[str] = []

        def capture(update: AgentResponseUpdate) -> AgentResponseUpdate:
            chunks.append(update.text or "")
            return update

        hooks = list(context.stream_transform_hooks or [])
        hooks.append(capture)
        context.stream_transform_hooks = hooks

        await call_next()

        if isinstance(context.result, ResponseStream):
            print(f"Streaming started (captured {len(chunks)} chunks so far)")
```

---

## 5. `FunctionalWorkflow` + `StepWrapper`

**Source:** `agent_framework/_workflows/_functional.py`

`FunctionalWorkflow` and `StepWrapper` are the implementation classes behind the
`@workflow` and `@step` decorators. Understanding their internals enables:
- Custom HITL resume patterns
- Per-step checkpointing
- Step result caching for replay
- Running steps outside a workflow context (for unit tests)

> **Experimental** — both classes emit `ExperimentalWarning` on import.

### `FunctionalWorkflow`

```python
@workflow
async def my_pipeline(data: str) -> str:
    # FunctionalWorkflow wraps this function
    ...
```

`@workflow` calls `FunctionalWorkflow(func, name=..., description=..., checkpoint_storage=...)`.

**Key attributes set at decoration time:**

| Attribute | Description |
|---|---|
| `name` | Display name; defaults to `func.__name__`. |
| `description` | Optional description string. |
| `graph_signature_hash` | Stable hash of step names for checkpoint compatibility checks. |
| `_non_ctx_param_names` | Names of non-`RunContext` parameters (max 1). |
| `_step_names` | Set of `StepWrapper.name` values discovered in the function body at decoration time. |

The `_classify_signature` validator runs at decoration time and raises `TypeError` if the function
declares more than one non-`RunContext` parameter, preventing silent data-loss bugs.

**`FunctionalWorkflow.run()` — resume via checkpoint:**

```python
# Initial run
result = await my_pipeline.run("hello")

# Resume after HITL pause — provide checkpoint_id and responses
result = await my_pipeline.run(
    "hello",
    checkpoint_id="<id from the HITL request_info event>",
    responses={"confirm_step": "yes"},
)
outputs = result.get_outputs()
```

**`FunctionalWorkflow.as_agent()` — expose as an `Agent`:**

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

@workflow
async def research_pipeline(query: str) -> str:
    summary = await summarize(query)
    return summary

# Wrap as an agent — any agent.run() call becomes a workflow run
agent = research_pipeline.as_agent(
    client=OpenAIChatClient(),
    name="ResearchAgent",
)
response = await agent.run("What is quantum computing?")
print(response.text)
```

### `StepWrapper`

`@step` wraps an async function in a `StepWrapper` that provides:
1. **Result caching** by `(step_name, call_index)` — replays skip already-completed work on HITL resume.
2. **Event emission** — `executor_invoked`, `executor_completed`, `executor_failed`, `executor_bypassed`.
3. **`RunContext` auto-injection** — if the function has a `ctx: RunContext` parameter, the active context is injected automatically.
4. **Per-step checkpointing** — a checkpoint is written after each live step execution.

```python
from agent_framework import step, RunContext

@step
async def validate_input(data: str) -> str:
    if not data.strip():
        raise ValueError("Empty input")
    return data.upper()

@step
async def enrich_data(data: str, ctx: RunContext) -> str:
    # ctx is injected automatically inside a @workflow
    await ctx.yield_output(f"Enriching: {data}")
    return f"{data} [enriched]"
```

**Testing a step in isolation** — outside a workflow, `StepWrapper` is transparent:

```python
import asyncio

async def test_validate_input():
    # No workflow context — calls the underlying function directly
    result = await validate_input("  hello  ")
    assert result == "HELLO"

asyncio.run(test_validate_input())
```

**Full functional workflow with HITL, checkpointing, and parallel steps:**

```python
import asyncio
from agent_framework import workflow, step, RunContext, InMemoryCheckpointStorage
from agent_framework.openai import OpenAIChatClient

storage = InMemoryCheckpointStorage()

@step
async def fetch_data(query: str) -> dict:
    return {"query": query, "raw": "some data"}

@step
async def validate_and_confirm(data: dict, ctx: RunContext) -> dict:
    # Pause for human review
    confirmation = await ctx.request_info(
        "Please confirm the data is correct",
        {"data_preview": str(data)},
        request_id="confirm_data",
    )
    if confirmation.get("approved") != "yes":
        raise ValueError("Data rejected by user")
    return data

@step
async def process_data(data: dict) -> str:
    return f"Processed: {data['query']}"

@workflow(checkpoint_storage=storage)
async def data_pipeline(query: str, ctx: RunContext) -> str:
    raw = await fetch_data(query)
    validated = await validate_and_confirm(raw)
    result = await process_data(validated)
    return result

async def main():
    import asyncio

    # Start the run — will pause at validate_and_confirm
    try:
        result = await data_pipeline.run("sales Q4")
    except Exception as e:
        print(f"Paused for HITL: {e}")

    # Resume with approval (checkpoint_id comes from the WorkflowEvent.request_info event)
    result = await data_pipeline.run(
        "sales Q4",
        checkpoint_id="<checkpoint_id>",
        responses={"confirm_data": {"approved": "yes"}},
    )
    print(result.get_outputs())

asyncio.run(main())
```

---

## 6. `WorkflowEvent` + `WorkflowEventType` + `WorkflowEventSource`

**Source:** `agent_framework/_workflows/_events.py`

`WorkflowEvent` is the unified event emitted by all workflow executions. When you call
`workflow.run(..., stream=True)` you receive a `ResponseStream[WorkflowEvent, WorkflowRunResult]`.
Understanding the event taxonomy is essential for building observability tooling,
progress UIs, or custom HITL dispatch loops.

### `WorkflowEventType` (Literal)

The `type` discriminator on `WorkflowEvent` is one of these string literals:

| Event type | When emitted | `data` field |
|---|---|---|
| `"started"` | Workflow run begins. | `None` |
| `"status"` | Workflow state changes (e.g. `IN_PROGRESS`). | `WorkflowRunState` |
| `"failed"` | Workflow terminated with error. | `WorkflowErrorDetails` |
| `"warning"` | User code emitted a warning. | `str` |
| `"error"` | User code raised an exception (non-fatal). | `Exception` |
| `"request_info"` | An executor is pausing for HITL input. | request payload |
| `"superstep_started"` | A graph superstep begins. | `None` |
| `"superstep_completed"` | A graph superstep finishes. | `None` |
| `"executor_invoked"` | An executor (agent/step) is about to run. | `None` |
| `"executor_completed"` | An executor finished successfully. | `None` |
| `"executor_failed"` | An executor raised an exception. | `WorkflowErrorDetails` |
| `"executor_bypassed"` | An executor was skipped (cache hit). | `None` |
| `"output"` | An executor produced an output. | output value |
| `"intermediate"` | An executor produced an intermediate output. | intermediate value |
| `"data"` | Raw `AgentResponse` or `AgentResponseUpdate` from an agent executor. | `AgentResponse` or update |

### `WorkflowEventSource`

| Value | Description |
|---|---|
| `"workflow"` | Emitted by the workflow runner (lifecycle events). |
| `"executor"` | Emitted by an executor (agent or functional step). |

### `WorkflowEvent` fields

| Field | Type | Description |
|---|---|---|
| `type` | `WorkflowEventType` | Event discriminator (see table above). |
| `data` | `DataT \| None` | Payload — type depends on `type`. |
| `origin` | `WorkflowEventSource \| None` | `"workflow"` or `"executor"`. |
| `state` | `WorkflowRunState \| None` | Present on `"status"` events. |
| `details` | `WorkflowErrorDetails \| None` | Present on `"failed"` and `"executor_failed"` events. |
| `executor_id` | `str \| None` | Name of the executor that produced this event. |
| `request_id` | `str \| None` | HITL request ID on `"request_info"` events. |
| `iteration` | `int \| None` | Superstep index on `"superstep_*"` events. |

### Factory methods

```python
from agent_framework._workflows._events import WorkflowEvent, WorkflowEventSource

# Lifecycle (framework creates these — shown here for testing/mocking)
started = WorkflowEvent.started()
status  = WorkflowEvent.status(state)
failed  = WorkflowEvent.failed(error_details)

# Executor bookkeeping
invoked   = WorkflowEvent.executor_invoked("my_step")
completed = WorkflowEvent.executor_completed("my_step")
bypassed  = WorkflowEvent.executor_bypassed("my_step")
failed_ex = WorkflowEvent.executor_failed("my_step", error_details)

# Superstep
sup_start = WorkflowEvent.superstep_started(iteration=1)
sup_done  = WorkflowEvent.superstep_completed(iteration=1)

# HITL
req = WorkflowEvent.request_info(request_id="confirm", executor_id="validate_step", ...)
```

### Consuming the workflow event stream

```python
import asyncio
from agent_framework import WorkflowBuilder, Agent, WorkflowEvent
from agent_framework.openai import OpenAIChatClient

agent_a = Agent(client=OpenAIChatClient(), name="ResearchAgent",
                instructions="Research the query.")
agent_b = Agent(client=OpenAIChatClient(), name="SummaryAgent",
                instructions="Summarize the research.")

wf = (
    WorkflowBuilder()
    .add_agent("research", agent_a, output_from=["research"])
    .add_agent("summary", agent_b, output_from=["summary"])
    .add_edge("research", "summary")
    .build(name="ResearchPipeline")
)

async def run_with_events():
    stream = await wf.run("What is GraphRAG?", stream=True)

    async for event in stream:
        if event.type == "executor_invoked":
            print(f"▶ {event.executor_id} started")
        elif event.type == "executor_completed":
            print(f"✓ {event.executor_id} completed")
        elif event.type == "output":
            print(f"📤 Output from {event.executor_id}: {event.data}")
        elif event.type == "failed":
            print(f"✗ Workflow failed: {event.details}")
        elif event.type == "data":
            # Streaming agent tokens
            if hasattr(event.data, "text"):
                print(event.data.text, end="", flush=True)

    result = await stream.get_final_response()
    print(f"\nFinal outputs: {result.get_outputs()}")

asyncio.run(run_with_events())
```

**HITL dispatch loop via event stream:**

```python
async def hitl_loop(wf, query: str):
    checkpoint_id = None
    responses = {}

    while True:
        stream = await wf.run(
            query,
            checkpoint_id=checkpoint_id,
            responses=responses,
        )
        async for event in stream:
            if event.type == "request_info":
                # Pause and collect user input
                user_input = input(f"[HITL] {event.request_id}: ")
                responses[event.request_id] = user_input
                checkpoint_id = event.data.get("checkpoint_id")
                break
        else:
            # Stream completed without HITL — done
            result = await stream.get_final_response()
            return result
```

---

## 7. `SkillsSource` + `AggregatingSkillsSource` + `FilteringSkillsSource` + `DeduplicatingSkillsSource`

**Source:** `agent_framework/_skills.py`

> **Experimental** — all skill classes emit `ExperimentalWarning` on import.

`SkillsSource` is the abstract base class for skill discovery backends. The three concrete
decorators compose sources into a pipeline: aggregate → filter → deduplicate.

### `SkillsSource` (ABC)

```python
class SkillsSource(ABC):
    @abstractmethod
    async def get_skills(self) -> list[Skill]: ...
```

Subclass to create custom skill discovery (e.g. from a database, API, or ZIP archive):

```python
import warnings
from agent_framework import SkillsSource, Skill, SkillFrontmatter

class DatabaseSkillsSource(SkillsSource):
    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    async def get_skills(self) -> list[Skill]:
        # Fetch skill definitions from your database
        rows = await fetch_skills_from_db(self._db_url)
        skills = []
        for row in rows:
            skill = Skill(
                frontmatter=SkillFrontmatter(name=row["name"], description=row["description"]),
                content=row["content"],
            )
            skills.append(skill)
        return skills
```

### `AggregatingSkillsSource`

Combines multiple `SkillsSource` instances into a single source — skills from all sources
are concatenated in order:

```python
AggregatingSkillsSource(sources: Sequence[SkillsSource])
```

```python
import warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)

from agent_framework import (
    AggregatingSkillsSource, FilteringSkillsSource, DeduplicatingSkillsSource,
    FileSkillsSource, InMemorySkillsSource, SkillsProvider, Agent,
)
from agent_framework.openai import OpenAIChatClient

# Three separate skill sources
team_skills   = FileSkillsSource("./skills/team/")
shared_skills = FileSkillsSource("./skills/shared/")
dynamic       = InMemorySkillsSource([])  # populated at runtime

# Merge into one
merged = AggregatingSkillsSource([team_skills, shared_skills, dynamic])
```

### `FilteringSkillsSource`

Wraps an inner source and applies a predicate — only skills for which `predicate` returns
`True` are returned:

```python
FilteringSkillsSource(
    inner_source: SkillsSource,
    predicate: Callable[[Skill], bool],
)
```

```python
# Exclude internal/private skills
public_only = FilteringSkillsSource(
    inner_source=merged,
    predicate=lambda s: not s.frontmatter.name.startswith("_"),
)

# Keep only skills tagged for the current user's role
def role_filter(skill: Skill) -> bool:
    tags = getattr(skill.frontmatter, "tags", []) or []
    return "admin" not in tags or current_user.is_admin

role_scoped = FilteringSkillsSource(inner_source=merged, predicate=role_filter)
```

### `DeduplicatingSkillsSource`

Wraps an inner source and removes skills with duplicate names (case-insensitive, first-wins):

```python
DeduplicatingSkillsSource(inner_source: SkillsSource)
```

```python
# Guarantee no duplicate names when merging from multiple sources
unique = DeduplicatingSkillsSource(inner_source=merged)
```

### Full composition pipeline

```python
from agent_framework import (
    AggregatingSkillsSource, FilteringSkillsSource, DeduplicatingSkillsSource,
    FileSkillsSource, SkillsProvider, Agent,
)
from agent_framework.openai import OpenAIChatClient

# Build layered skill pipeline
pipeline = DeduplicatingSkillsSource(
    inner_source=FilteringSkillsSource(
        inner_source=AggregatingSkillsSource([
            FileSkillsSource("./skills/core/"),
            FileSkillsSource("./skills/plugins/"),
            DatabaseSkillsSource("postgresql://..."),
        ]),
        predicate=lambda s: s.frontmatter.name != "debug_only",
    )
)

provider = SkillsProvider(pipeline)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="Use available skills to help users.",
    context_providers=[provider],
)

async def main():
    response = await agent.run("What skills do you have?")
    print(response.text)
```

### Custom source with caching

```python
import asyncio
from agent_framework import SkillsSource, Skill

class CachedApiSkillsSource(SkillsSource):
    _cache: list[Skill] | None = None
    _lock = asyncio.Lock()
    TTL = 300  # seconds

    def __init__(self, api_url: str) -> None:
        self._api_url = api_url
        self._fetched_at: float = 0

    async def get_skills(self) -> list[Skill]:
        import time
        now = time.monotonic()
        async with self._lock:
            if self._cache is None or (now - self._fetched_at) > self.TTL:
                self._cache = await self._fetch_from_api()
                self._fetched_at = now
        return list(self._cache)

    async def _fetch_from_api(self) -> list[Skill]:
        # ... HTTP call ...
        return []
```

---

## 8. `EvalItem` + `EvalItemResult` + `EvalResults` + `EvalScoreResult`

**Source:** `agent_framework/_evaluation.py`

> **Experimental** — all evaluation classes emit `ExperimentalWarning` on import.

These four types form the data model for `evaluate_agent()` and `evaluate_workflow()`.
[Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/)
covered `LocalEvaluator`; this volume covers the _input_ and _output_ data model.

### `EvalItem`

Represents one query/response interaction for evaluation:

```python
EvalItem(
    conversation: list[Message],
    tools: list[FunctionTool] | None = None,
    context: str | None = None,
    expected_output: str | None = None,
    expected_tool_calls: list[ExpectedToolCall] | None = None,
    split_strategy: ConversationSplitter | None = None,
)
```

| Field | Description |
|---|---|
| `conversation` | Full conversation as `Message` objects — single source of truth. |
| `tools` | Tool objects available to the evaluator for tool-correctness evaluation. |
| `context` | Optional grounding context document (RAG use-cases). |
| `expected_output` | Expected text output for ground-truth comparison evaluators. |
| `expected_tool_calls` | List of `ExpectedToolCall` for tool-correctness evaluators. |
| `split_strategy` | How to split `conversation` into query vs. response — defaults to `ConversationSplit.LAST_TURN`. |

**Derived properties** (computed from `conversation` + `split_strategy`):

| Property | Description |
|---|---|
| `query` | Concatenated user text from the query side of the split. |
| `response` | Concatenated text from the response side of the split. |

```python
from agent_framework import Message, EvalItem
from agent_framework._evaluation import ConversationSplit

# Simple single-turn item
item = EvalItem(
    conversation=[
        Message("user", ["What is the capital of France?"]),
        Message("assistant", ["The capital of France is Paris."]),
    ],
    expected_output="Paris",
)
print(item.query)     # "What is the capital of France?"
print(item.response)  # "The capital of France is Paris."

# Multi-turn item — FULL split evaluates the entire trajectory
multi_turn = EvalItem(
    conversation=[
        Message("user", ["Book a table for two at 7pm."]),
        Message("assistant", ["I'd be happy to help. Which restaurant?"]),
        Message("user", ["Le Gavroche please."]),
        Message("assistant", ["Table booked at Le Gavroche for two at 7pm."]),
    ],
    split_strategy=ConversationSplit.FULL,
    expected_output="Le Gavroche",
)
print(multi_turn.query)     # "Book a table for two at 7pm."
print(multi_turn.response)  # full multi-turn response
```

### `EvalScoreResult`

A single evaluator score on a single item:

```python
@dataclass
class EvalScoreResult:
    name: str               # Evaluator name, e.g. "relevance"
    score: float            # Numeric score
    passed: bool | None     # Whether the item passed the threshold
    sample: dict | None     # Raw evaluator output / rationale
```

### `EvalItemResult`

Per-item results (populated when the provider supports per-item retrieval):

| Field | Type | Description |
|---|---|---|
| `item_id` | `str` | Item identifier. |
| `status` | `str` | `"pass"`, `"fail"`, or `"error"`. |
| `scores` | `list[EvalScoreResult]` | Per-evaluator scores. |
| `error_code` | `str \| None` | Error code if `is_error`. |
| `error_message` | `str \| None` | Human-readable error message. |
| `is_error` | `bool` | Whether this item errored during evaluation. |
| `token_usage` | `UsageDetails \| None` | Token consumption for this item's evaluation. |

### `EvalResults`

The top-level result object from `evaluate_agent()`:

```python
EvalResults(
    provider: str,
    eval_id: str = "",
    run_id: str = "",
    status: str = "unknown",
    result_counts: CheckResult | None = None,
    report_url: str | None = None,
    error: str | None = None,
    per_evaluator: dict[str, CheckResult] | None = None,
    items: list[EvalItemResult] | None = None,
    sub_results: dict[str, "EvalResults"] | None = None,
)
```

| Field | Description |
|---|---|
| `provider` | `"local"`, `"azure_ai_evaluation"`, etc. |
| `status` | `"completed"`, `"failed"`, `"canceled"`, `"timeout"`. |
| `result_counts` | Pass/fail summary (`CheckResult.passed`, `.failed`, `.total`). |
| `per_evaluator` | Per-evaluator pass/fail counts, keyed by evaluator name. |
| `items` | Per-item results with scores and error details. |
| `sub_results` | Per-agent breakdown for multi-agent workflow evaluations. |

### Complete evaluation example

```python
import asyncio
import warnings
from agent_framework import Agent, FunctionTool, evaluate_agent, LocalEvaluator
from agent_framework import Message, EvalItem
from agent_framework._evaluation import keyword_check, tool_called_check
from agent_framework.openai import OpenAIChatClient

warnings.filterwarnings("ignore", category=ExperimentalWarning)

async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a weather assistant.",
    tools=[FunctionTool(get_weather)],
)

# Build evaluation items
items = [
    EvalItem(
        conversation=[
            Message("user", ["What's the weather in Paris?"]),
        ],
        expected_output="Paris",
        expected_tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}}],
    ),
    EvalItem(
        conversation=[
            Message("user", ["Is it raining in Tokyo?"]),
        ],
        expected_output="Tokyo",
    ),
]

evaluators = [
    LocalEvaluator("keyword", keyword_check(field="response", keywords=["°C"])),
    LocalEvaluator("tool_used", tool_called_check(tool_name="get_weather")),
]

async def main():
    results_list = await evaluate_agent(
        agent=agent,
        queries=items,
        evaluators=evaluators,
    )

    for results in results_list:
        print(f"Provider: {results.provider}")
        print(f"Status: {results.status}")
        print(f"Pass: {results.result_counts.passed}, Fail: {results.result_counts.failed}")

        for item_result in (results.items or []):
            icon = "✓" if item_result.status == "pass" else "✗"
            print(f"  {icon} {item_result.item_id}")
            for score in item_result.scores:
                print(f"    {score.name}: {score.score:.2f} ({'pass' if score.passed else 'fail'})")

asyncio.run(main())
```

---

## 9. `TokenizerProtocol` + `CharacterEstimatorTokenizer`

**Source:** `agent_framework/_compaction.py`

Token counting is the critical dependency for all context-window-aware compaction strategies
(`SlidingWindowStrategy`, `TokenBudgetComposedStrategy`, `ContextWindowCompactionStrategy`).
`TokenizerProtocol` defines the interface; `CharacterEstimatorTokenizer` provides a
zero-dependency fast heuristic.

### `TokenizerProtocol`

```python
@runtime_checkable
class TokenizerProtocol(Protocol):
    def count_tokens(self, text: str) -> int:
        """Count tokens for a serialized message payload."""
        ...
```

Because it is `@runtime_checkable`, you can use `isinstance(obj, TokenizerProtocol)` to
validate an arbitrary object at runtime.

Any object with a `count_tokens(text: str) -> int` method satisfies the protocol.

### `CharacterEstimatorTokenizer`

```python
class CharacterEstimatorTokenizer:
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
```

The 4-characters-per-token heuristic is a reasonable approximation for English text with
modern models. It is the default when no tokenizer is provided.

### Providing a real tokenizer

For precise token counting (e.g. when you need to stay just under a context-window limit),
plug in `tiktoken` for OpenAI models or `tokenizers` for open-source models:

```python
import tiktoken
from agent_framework import TokenizerProtocol

class TiktokenTokenizer:
    def __init__(self, model: str = "gpt-4o") -> None:
        self._enc = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

# Verify it satisfies the protocol at runtime
assert isinstance(TiktokenTokenizer(), TokenizerProtocol)
```

### Wiring a custom tokenizer

Pass a tokenizer to `Agent`, `SlidingWindowStrategy`, or `TokenBudgetComposedStrategy`:

```python
import asyncio
from agent_framework import (
    Agent, SlidingWindowStrategy, TokenBudgetComposedStrategy,
    CompactionProvider, SummarizationStrategy,
)
from agent_framework.openai import OpenAIChatClient

tokenizer = TiktokenTokenizer("gpt-4o")

# Per-agent window strategy using real token counts
sliding = SlidingWindowStrategy(max_tokens=6000, tokenizer=tokenizer)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful assistant.",
    tokenizer=tokenizer,                     # used by built-in strategies
    compaction_strategy=CompactionProvider(sliding),
)
```

**Budget-aware compaction with a real tokenizer:**

```python
from agent_framework import TokenBudgetComposedStrategy, SummarizationStrategy

compactor = TokenBudgetComposedStrategy(
    max_tokens=8000,
    tokenizer=tokenizer,
    strategies=[
        SummarizationStrategy(
            client=OpenAIChatClient(),
            summary_max_tokens=800,
        ),
        SlidingWindowStrategy(max_tokens=6000, tokenizer=tokenizer),
    ],
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a long-context assistant.",
    compaction_strategy=CompactionProvider(compactor),
    tokenizer=tokenizer,
)
```

**Custom tokenizer for `HuggingFace` / open-source models:**

```python
from tokenizers import Tokenizer as HFTokenizer

class HuggingFaceTokenizer:
    def __init__(self, tokenizer_path: str) -> None:
        self._tok = HFTokenizer.from_pretrained(tokenizer_path)

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text).ids)

hf_tokenizer = HuggingFaceTokenizer("mistralai/Mistral-7B-v0.1")
agent = Agent(
    client=my_mistral_client,
    instructions="...",
    tokenizer=hf_tokenizer,
    compaction_strategy=CompactionProvider(
        SlidingWindowStrategy(max_tokens=4096, tokenizer=hf_tokenizer)
    ),
)
```

---

## 10. `ConversationSplit` + `ConversationSplitter`

**Source:** `agent_framework/_evaluation.py`

> **Experimental** — emits `ExperimentalWarning` on import.

`ConversationSplitter` is the protocol; `ConversationSplit` is an enum of built-in strategies.
Together they determine how `EvalItem.conversation` is partitioned into a *query* half and a
*response* half for evaluation purposes.

### `ConversationSplitter` (Protocol)

```python
class ConversationSplitter(Protocol):
    def __call__(
        self,
        conversation: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        ...
```

The return value is `(query_messages, response_messages)`.

### `ConversationSplit` (Enum)

```python
class ConversationSplit(str, Enum):
    LAST_TURN = "last_turn"
    FULL      = "full"
```

| Strategy | Split point | Best for |
|---|---|---|
| `LAST_TURN` | At the last user message — everything up to and including that message is the query; everything after is the response. | Evaluating whether the agent correctly answered the *latest* question. |
| `FULL` | The first user message (and preceding system messages) is the query; the whole remainder is the response. | Evaluating whether the *entire conversation trajectory* served the original request. |

Both enum members are directly callable:

```python
from agent_framework import Message
from agent_framework._evaluation import ConversationSplit

conversation = [
    Message("system",    ["You are a booking assistant."]),
    Message("user",      ["Book a table for two."]),
    Message("assistant", ["Which restaurant?"]),
    Message("user",      ["Le Gavroche please."]),
    Message("assistant", ["Done. Le Gavroche, two, 7pm."]),
]

query_msgs, response_msgs = ConversationSplit.LAST_TURN(conversation)
print([m.text for m in query_msgs])
# [..., 'Le Gavroche please.']   <- up to and including last user message
print([m.text for m in response_msgs])
# ['Done. Le Gavroche, two, 7pm.']

query_msgs_full, response_msgs_full = ConversationSplit.FULL(conversation)
print([m.text for m in query_msgs_full])
# ['You are a booking assistant.', 'Book a table for two.']
print(len(response_msgs_full))  # 3
```

### Custom split strategy

```python
from agent_framework import Message
from agent_framework._evaluation import ConversationSplitter

class FirstAssistantSplitter:
    """Split at the first assistant message — query is everything before it."""

    def __call__(
        self,
        conversation: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        for i, msg in enumerate(conversation):
            if msg.role == "assistant":
                return conversation[:i], conversation[i:]
        return conversation, []

first_split = FirstAssistantSplitter()
assert isinstance(first_split, ConversationSplitter)  # runtime check
```

### Using a custom splitter with `EvalItem`

```python
from agent_framework import EvalItem, Message

item = EvalItem(
    conversation=[
        Message("user",      ["Translate 'hello' to French."]),
        Message("assistant", ["Bonjour."]),
        Message("user",      ["And in Spanish?"]),
        Message("assistant", ["Hola."]),
    ],
    expected_output="Hola",
    split_strategy=ConversationSplit.LAST_TURN,
)
print(item.query)     # "And in Spanish?"
print(item.response)  # "Hola."
```

### Combining with a multi-step evaluation pipeline

```python
import asyncio
from agent_framework import (
    Agent, FunctionTool, evaluate_agent, LocalEvaluator, EvalItem, Message,
)
from agent_framework._evaluation import ConversationSplit, keyword_check
from agent_framework.openai import OpenAIChatClient

agent = Agent(client=OpenAIChatClient(), instructions="You are a translation assistant.")

items = [
    EvalItem(
        conversation=[
            Message("user", ["Translate 'good morning' to French."]),
            Message("assistant", ["Bonjour."]),
            Message("user", ["And to Spanish?"]),
            Message("assistant", ["Buenos días."]),
        ],
        split_strategy=ConversationSplit.LAST_TURN,
        expected_output="Buenos días",
    ),
    EvalItem(
        conversation=[
            Message("user", ["Translate 'thank you' to Italian."]),
            Message("assistant", ["Grazie."]),
        ],
        split_strategy=ConversationSplit.FULL,
        expected_output="Grazie",
    ),
]

evaluators = [
    LocalEvaluator("keyword", keyword_check(field="response", keywords=["expected_output"])),
]

async def main():
    results_list = await evaluate_agent(agent=agent, queries=items, evaluators=evaluators)
    for r in results_list:
        print(f"Pass: {r.result_counts.passed} / {r.result_counts.total}")

asyncio.run(main())
```

---

*This document was introspected from **agent-framework-core 1.7.0** source on 2026-06-01.*
*Forward reference: See [Class Deep Dives Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) for `Executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, and the full exception hierarchy.*
*See also [azure-ai-agents integration Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives_v6/) for the Azure AI Agents add-on class reference.*

---
title: "Microsoft Agent Framework (Python) — 10-API Deep Dives Vol. 5 (1.19.0)"
description: "Source-verified deep dives for WorkflowEvent, AgentContext, MiddlewareBundle, ConversationSplit/ConversationSplitter, VectorStoreHistoryProvider, MemoryStore/MemoryFileStore, MemoryTopicRecord, WorkflowRunResult, FunctionInvocationContext, and ChatOptions — all verified against agent-framework 1.19.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-API Deep Dives Vol. 5

**Verified against:** `agent-framework==1.19.0`
**Python requirement:** 3.10+

This volume covers 10 additional public APIs spanning the workflow event bus, agent and function middleware contexts, indivisible middleware bundles, evaluation conversation splitting, vector-store-backed history, topic-based memory stores, workflow run results, progressive tool exposure, and the cross-provider `ChatOptions` TypedDict. Each section includes the full constructor or signature, every meaningful method or factory, and self-contained runnable examples verified against the 1.19.0 source.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

See [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) for `FanInEdgeGroup`, `FanOutEdgeGroup`, `FunctionalWorkflow`, `FunctionalWorkflowAgent`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `MCPStdioTool`, `MCPStreamableHTTPTool`, `SelectiveToolCallCompactionStrategy`, and `TodoProvider`.

See [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) for `WorkflowBuilder`, `SlidingWindowStrategy`, `TruncationStrategy`, `ContextWindowCompactionStrategy`, `LocalEvaluator`, `InlineSkill`, `FileAccessProvider`, `MemoryContextProvider`, `FileHistoryProvider`, and `MCPWebsocketTool`.

See [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) for `VectorStoreField`, `VectorStoreCollectionDefinition`, `InMemoryCollection`, `InMemoryStore`, `Filter`, `FilterGroup`, `SecretString`, `load_settings`, `create_agent_hooks_middleware`, and `GroupChatBuilder`.

---

## 1. `WorkflowEvent[DataT]`

**Module:** `agent_framework._workflows._events` (re-exported via `agent_framework`)

`WorkflowEvent` is the single generic class for all events emitted during a workflow run. Every emission carries a `type` discriminator string (lifecycle, diagnostic, data, or bookkeeping) and an optional typed `data` payload. The framework emits these automatically; application code consumes them from `WorkflowRunResult` or from a streaming `async for` loop.

> **Note on `WorkflowEvent.emit()`:** The `emit()` factory is deprecated since 1.14.0. Use `ctx.yield_output()` from an intermediate-designated executor instead.

### Constructor

```python
WorkflowEvent(
    type: WorkflowEventType,     # discriminator string — see table below
    data: DataT | None = None,
    *,
    origin: WorkflowEventSource | None = None,
    state: WorkflowRunState | None = None,           # STATUS events
    details: WorkflowErrorDetails | None = None,     # FAILED events
    executor_id: str | None = None,                  # OUTPUT / DATA / executor events
    request_id: str | None = None,                   # REQUEST_INFO events
    source_executor_id: str | None = None,           # REQUEST_INFO events
    request_type: type[Any] | None = None,           # REQUEST_INFO events
    response_type: type[Any] | None = None,          # REQUEST_INFO events
    iteration: int | None = None,                    # SUPERSTEP events
)
```

> Prefer the factory methods over the constructor directly.

### Event type table

| `type` string | Factory method | `data` type | Key extra fields |
|---|---|---|---|
| `"started"` | `WorkflowEvent.started()` | `None` / DataT | — |
| `"status"` | `WorkflowEvent.status(state)` | `None` / DataT | `state` |
| `"failed"` | `WorkflowEvent.failed(details)` | `None` / DataT | `details` |
| `"warning"` | `WorkflowEvent.warning(msg)` | `str` | — |
| `"error"` | `WorkflowEvent.error(exc)` | `Exception` | — |
| `"output"` | emitted by `ctx.yield_output()` | DataT | `executor_id` |
| `"intermediate"` | emitted by `ctx.yield_output()` (intermediate) | DataT | `executor_id` |
| `"request_info"` | `WorkflowEvent.request_info(...)` | DataT | `request_id`, `source_executor_id` |
| `"superstep_started"` | `WorkflowEvent.superstep_started(n)` | `None` / DataT | `iteration` |
| `"superstep_completed"` | `WorkflowEvent.superstep_completed(n)` | `None` / DataT | `iteration` |
| `"executor_invoked"` | `WorkflowEvent.executor_invoked(id)` | `None` / DataT | `executor_id` |
| `"executor_completed"` | `WorkflowEvent.executor_completed(id)` | `None` / DataT | `executor_id` |
| `"executor_failed"` | `WorkflowEvent.executor_failed(id, details)` | `WorkflowErrorDetails` | `executor_id`, `details` |
| `"executor_bypassed"` | `WorkflowEvent.executor_bypassed(id)` | `None` / DataT | `executor_id` — cache-hit replay |

### Factory methods (classmethod)

| Method | Signature | Notes |
|---|---|---|
| `started` | `(data=None) → WorkflowEvent[DataT]` | First event of every run |
| `status` | `(state: WorkflowRunState, data=None) → WorkflowEvent[DataT]` | State transitions |
| `failed` | `(details: WorkflowErrorDetails, data=None) → WorkflowEvent[DataT]` | Run termination |
| `warning` | `(message: str) → WorkflowEvent[str]` | User-emitted diagnostic |
| `error` | `(exception: Exception) → WorkflowEvent[Exception]` | User-emitted diagnostic |
| `request_info` | `(request_id, source_executor_id, request_data, response_type) → WorkflowEvent[DataT]` | Human-in-the-loop pause |
| `superstep_started` | `(iteration: int, data=None) → WorkflowEvent[DataT]` | Pregel superstep begin |
| `superstep_completed` | `(iteration: int, data=None) → WorkflowEvent[DataT]` | Pregel superstep end |
| `executor_invoked` | `(executor_id: str, data=None) → WorkflowEvent[DataT]` | Bookkeeping |
| `executor_completed` | `(executor_id: str, data=None) → WorkflowEvent[DataT]` | Bookkeeping |
| `executor_failed` | `(executor_id: str, details: WorkflowErrorDetails) → WorkflowEvent[WorkflowErrorDetails]` | Bookkeeping |
| `executor_bypassed` | `(executor_id: str, data=None) → WorkflowEvent[DataT]` | Cache replay |

### Type-safe property accessors

`request_id`, `source_executor_id`, `request_type`, `response_type` — each raises `RuntimeError` when accessed on an event whose `type` is not `"request_info"`.

### Serialization

```python
event.to_dict() → dict[str, Any]        # only for "request_info" events
WorkflowEvent.from_dict(data, allowed_types=None) → WorkflowEvent[Any]
```

### Example — consuming events from a non-streaming run

```python
import asyncio
from agent_framework import (
    Agent, WorkflowEvent, WorkflowBuilder, WorkflowRunState
)
from agent_framework.openai import OpenAIChatClient

async def main():
    client = OpenAIChatClient()
    agent = Agent(client=client, name="summarizer",
                  instructions="Summarize the user's text in one sentence.")

    workflow = WorkflowBuilder(start_executor=agent).build()
    # include_status_events=True adds status/failed events to the iterable list;
    # without it they are only accessible via result.status_timeline().
    result = await workflow.run(
        "The quick brown fox jumps over the lazy dog.",
        include_status_events=True,
    )

    for event in result:
        if event.type == "output":
            print(f"Output from {event.executor_id}: {event.data}")
        elif event.type == "status":
            print(f"State → {event.state.value}")
        elif event.type == "failed":
            print(f"FAILED: {event.details}")

    final = result.get_final_state()
    assert final == WorkflowRunState.IDLE

asyncio.run(main())
```

### Example — streaming events

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

async def stream():
    agent = Agent(client=OpenAIChatClient(), name="poet",
                  instructions="Write a haiku.")
    workflow = WorkflowBuilder(start_executor=agent).build()

    async for event in workflow.run("Cherry blossoms fall", stream=True):
        match event.type:
            case "started":
                print("Workflow started")
            case "superstep_started":
                print(f"  Superstep {event.iteration} begin")
            case "executor_invoked":
                print(f"  Executor '{event.executor_id}' invoked")
            case "output":
                print(f"  Output: {event.data}")
            case "superstep_completed":
                print(f"  Superstep {event.iteration} done")
            case "status":
                print(f"  State: {event.state.value}")

asyncio.run(stream())
```

### Example — human-in-the-loop via `request_info`

```python
import asyncio
from agent_framework import WorkflowEvent

# Pause-and-resume pattern: the executor emits a request_info event,
# the host reads it, supplies the answer, then resumes the workflow via run().
async def handle_pending_request(run_result, workflow, checkpoint_id: str):
    pending = run_result.get_request_info_events()
    if not pending:
        return

    req: WorkflowEvent = pending[0]
    # req.request_id, req.source_executor_id, req.data, req.response_type
    print(f"Workflow is asking: {req.data}")
    user_answer = await asyncio.to_thread(input, "Your answer: ")

    resumed = await workflow.run(
        responses={req.request_id: user_answer},
        checkpoint_id=checkpoint_id,
    )
    outputs = resumed.get_outputs()
    print(f"Final output: {outputs}")
```

---

## 2. `AgentContext`

**Module:** `agent_framework._middleware` (re-exported via `agent_framework`)

`AgentContext` is the mutable context object passed through the **agent middleware** pipeline on every `agent.run()` call. Middleware reads it before calling `call_next()` (to inspect or mutate the incoming request) and reads it again after (to inspect or replace the result).

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
    result: AgentResponse | ResponseStream | None = None,
    kwargs: Mapping[str, Any] | None = None,
    client_kwargs: Mapping[str, Any] | None = None,
    function_invocation_kwargs: Mapping[str, Any] | None = None,
    stream_transform_hooks: Sequence[Callable] | None = None,
    stream_result_hooks: Sequence[Callable] | None = None,
    stream_cleanup_hooks: Sequence[Callable] | None = None,
)
```

> The framework constructs `AgentContext` for you. You receive it as the first argument of `AgentMiddleware.process()`.

### Attributes

| Attribute | Type | Notes |
|---|---|---|
| `agent` | `SupportsAgentRun` | The agent being invoked. Read-only in practice. |
| `messages` | `list[Message]` | Messages sent to the agent. Mutate to inject/remove messages before the call. |
| `session` | `AgentSession \| None` | The current session, or `None` for stateless runs. |
| `tools` | tool types | Run-level tool overrides. `None` → agent's declared tools apply. |
| `options` | `dict[str, Any]` | Merged run options (model, temperature, etc.). |
| `stream` | `bool` | `True` for streaming invocations. |
| `compaction_strategy` | `CompactionStrategy \| None` | Per-run compaction override. |
| `tokenizer` | `TokenizerProtocol \| None` | Per-run tokenizer override. |
| `metadata` | `dict[str, Any]` | Shared scratchpad for passing data between middleware layers. |
| `result` | `AgentResponse \| ResponseStream \| None` | Set after `call_next()`. Replace to override the agent's response. |
| `kwargs` | `dict[str, Any]` | Legacy runtime keyword arguments. |
| `client_kwargs` | `dict[str, Any]` | Client-specific kwargs forwarded to the underlying chat client. |
| `function_invocation_kwargs` | `dict[str, Any]` | Kwargs forwarded into every tool invocation on this run. |
| `stream_transform_hooks` | `list[Callable]` | Per-update streaming transformers. |
| `stream_result_hooks` | `list[Callable]` | Transformers applied to the final streaming result. |
| `stream_cleanup_hooks` | `list[Callable]` | Cleanup callbacks run after streaming completes. |

### Example — timing middleware

```python
import asyncio
import time
from agent_framework import Agent, AgentMiddleware, AgentContext
from agent_framework.openai import OpenAIChatClient


class TimingMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        start = time.perf_counter()
        # Inspect what is being sent
        print(f"Running agent: {context.agent.name!r}")
        print(f"Messages:      {len(context.messages)}")
        print(f"Streaming:     {context.stream}")

        context.metadata["start_time"] = start
        await call_next()

        elapsed = time.perf_counter() - context.metadata["start_time"]
        if not context.stream:
            # Non-streaming: elapsed covers the full model round-trip.
            print(f"Elapsed:       {elapsed:.3f}s")
            print(f"Tokens used:   {context.result.usage_details}")
        else:
            # Streaming: call_next() returns a ResponseStream quickly; elapsed
            # measures stream setup only. Measure inside the consumer loop for
            # accurate generation latency.
            print(f"Stream ready in {elapsed:.3f}s (measure generation in the consumer)")


async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        name="demo",
        instructions="Answer concisely.",
        middleware=[TimingMiddleware()],
    )
    result = await agent.run("What is 2 + 2?")
    print(result.text)

asyncio.run(main())
```

### Example — injecting a system message

```python
from agent_framework import Agent, AgentMiddleware, AgentContext, Message


class DateInjectorMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        import datetime
        today = datetime.date.today().isoformat()
        context.messages = [
            Message.from_system(f"Today's date is {today}."),
            *context.messages,
        ]
        await call_next()
```

### Example — result override (mock testing)

```python
from agent_framework import Agent, AgentMiddleware, AgentContext, AgentResponse, Message


class MockMiddleware(AgentMiddleware):
    """Short-circuit the model call and return a canned response."""

    def __init__(self, canned_text: str):
        self._text = canned_text

    async def process(self, context: AgentContext, call_next):
        # Skip call_next entirely — return canned response
        context.result = AgentResponse(
            messages=[Message.from_assistant(self._text)],
        )
```

---

## 3. `MiddlewareBundle`

**Module:** `agent_framework._middleware` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.AGENT_HOOKS` to be acknowledged.

A `MiddlewareBundle` groups several middleware objects into one opaque, indivisible unit. Features like `create_agent_hooks_middleware()` return a bundle because their internal middleware objects only uphold their contract when installed together — a bundle prevents accidental partial installation.

Unlike a plain list, a `MiddlewareBundle` cannot be unpacked or sliced. Passing it to `Agent(middleware=[bundle, ...])` or `agent.run(middleware=[bundle, ...])` causes the framework to split its members into their agent/function/chat categories while preserving the ordering guarantee.

### Constructor

```python
MiddlewareBundle(
    middleware: Sequence[
        AgentMiddleware | FunctionMiddleware | ChatMiddleware
        | AgentMiddlewareCallable | FunctionMiddlewareCallable | ChatMiddlewareCallable
    ]
)
```

**Raises:**
- `MiddlewareException` if a nested `MiddlewareBundle` is included (bundles cannot nest).
- `MiddlewareException` if any member's middleware category cannot be determined.

### Example — creating and using a bundle

```python
from agent_framework import (
    Agent, MiddlewareBundle,
    AgentMiddleware, AgentContext,
    FunctionMiddleware, FunctionInvocationContext,
)
from agent_framework.openai import OpenAIChatClient


class IngressMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        print("[Ingress] agent call started")
        await call_next()


class EgressMiddleware(AgentMiddleware):
    async def process(self, context: AgentContext, call_next):
        await call_next()
        print("[Egress] agent call completed")


class ToolAuditMiddleware(FunctionMiddleware):
    async def process(self, context: FunctionInvocationContext, call_next):
        print(f"[Audit] tool call: {context.function.name}")
        await call_next()


# All three travel together as one indivisible bundle
enforcement_bundle = MiddlewareBundle([
    IngressMiddleware(),
    EgressMiddleware(),
    ToolAuditMiddleware(),
])

agent = Agent(
    client=OpenAIChatClient(),
    name="guarded",
    middleware=[enforcement_bundle],
)
```

### Example — bundle returned by a factory (agent-hooks pattern)

```python
from agent_framework import Agent, acknowledge_experimental_feature, ExperimentalFeature
from agent_framework import create_agent_hooks_middleware
from agent_framework.openai import OpenAIChatClient

# agent_hooks Interceptor objects come from the agent-hooks-sdk package.
# Install it: pip install --pre agent-hooks-sdk
from agent_hooks import Interceptor, InterceptionContext, Verdict

acknowledge_experimental_feature(ExperimentalFeature.AGENT_HOOKS)


class LoggingInterceptor(Interceptor):
    """Log each interception point and allow all through."""

    def intercept(self, context: InterceptionContext) -> Verdict:
        # InterceptionContext is a mapping; access data via .get()
        print(f"[Hook] point={context.get('interception_point')!r} agent={context.get('agent_id')!r}")
        return Verdict.allow()


bundle = create_agent_hooks_middleware(
    interceptors=[LoggingInterceptor()],
)

agent = Agent(
    client=OpenAIChatClient(),
    name="governed",
    middleware=[bundle],  # bundle, not a flat list
)
```

---

## 4. `ConversationSplit` and `ConversationSplitter`

**Module:** `agent_framework._evaluation` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.EVALS` to be acknowledged.

These two types work together in the evaluation harness. `ConversationSplitter` is a **structural protocol** — any callable with the signature `(list[Message]) → tuple[list[Message], list[Message]]` satisfies it. `ConversationSplit` is an **enum** of built-in splitters that also satisfy the protocol.

### `ConversationSplit` enum

| Member | Value | Behaviour |
|---|---|---|
| `ConversationSplit.LAST_TURN` | `"last_turn"` | Query = everything up to and including the last user message; response = all messages after. Evaluates whether the agent answered the *latest* question well. |
| `ConversationSplit.FULL` | `"full"` | Query = the first user message (plus any preceding system messages); response = the whole remainder. Evaluates the *complete conversation trajectory*. |

Both members are callable: `query_msgs, response_msgs = ConversationSplit.LAST_TURN(conversation)`.

### `ConversationSplitter` protocol

```python
# Any callable with this signature satisfies ConversationSplitter:
def my_splitter(
    conversation: list[Message],
) -> tuple[list[Message], list[Message]]:
    ...
```

### Example — built-in split with `LocalEvaluator`

```python
import asyncio
from agent_framework import (
    Agent, EvalItem, EvalCheck, CheckResult, LocalEvaluator, ConversationSplit,
    Message, acknowledge_experimental_feature, ExperimentalFeature,
)
from agent_framework.openai import OpenAIChatClient

acknowledge_experimental_feature(ExperimentalFeature.EVALS)


# An EvalCheck is a callable: (EvalItem) -> CheckResult
# item.response is already a str (the joined assistant text from the response split).
async def factual_check(item: EvalItem) -> CheckResult:
    """Pass if the response contains 'Paris'."""
    passed = "paris" in item.response.lower()
    return CheckResult(
        check_name="factual_check",
        passed=passed,
        reason="Response mentions Paris" if passed else "Missing 'Paris'",
    )


async def main():
    # EvalItem.conversation expects list[Message], not raw dicts
    # split_strategy belongs on EvalItem, not on evaluate()
    item = EvalItem(
        conversation=[
            Message("user", ["What is the capital of France?"]),
            Message("assistant", ["The capital of France is Paris."]),
        ],
        split_strategy=ConversationSplit.LAST_TURN,
    )

    evaluator = LocalEvaluator(factual_check)

    results = await evaluator.evaluate(items=[item])
    for r in results.items:
        # r.is_passed: bool; r.scores: list[EvalScoreResult] with per-check detail
        print(r.status, r.is_passed)
        for score in r.scores:
            print(f"  check={score.name!r} passed={score.passed}")

asyncio.run(main())
```

### Example — custom `ConversationSplitter`

```python
from agent_framework import Message


def split_before_tool_call(
    conversation: list[Message],
) -> tuple[list[Message], list[Message]]:
    """Split just before the first tool call, to evaluate what led the agent to call the tool."""
    for i, msg in enumerate(conversation):
        for content in (msg.contents or []):
            if content.type == "function_call":
                return conversation[:i], conversation[i:]
    # Fallback: last-turn split
    from agent_framework import ConversationSplit
    return ConversationSplit.LAST_TURN(conversation)


# Pass as split_strategy= on EvalItem, not as an argument to evaluate():
# item = EvalItem(conversation=..., split_strategy=split_before_tool_call)
# results = await evaluator.evaluate(items=[item])
```

### Example — `FULL` split for trajectory evaluation

```python
from agent_framework import ConversationSplit, Message

conversation = [
    Message.from_user("Plan a weekend trip to London."),
    Message.from_assistant("Sure! Day 1: Arrive and check into your hotel..."),
    Message.from_user("What about museums?"),
    Message.from_assistant("London has the British Museum, the Tate Modern, and the Natural History Museum..."),
]

query, response = ConversationSplit.FULL(conversation)
print("Query messages:", [m.role for m in query])      # ['user']
print("Response messages:", [m.role for m in response]) # ['assistant', 'user', 'assistant']
```

---

## 5. `VectorStoreHistoryProvider`

**Module:** `agent_framework._vectors` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.VECTOR_STORES` to be acknowledged.

`VectorStoreHistoryProvider` stores full conversation history in a provider-owned vector collection. Unlike `VectorCollectionContextProvider` (which exposes a caller-owned data model), this provider owns the collection schema and translates `Message` objects into a fixed history schema with optional embedding support.

History is scoped by `application_id` + optional `tenant_id` + optional `agent_id` + `source_id` + session ID. This prevents overlap between different agents and tenants but is **not** an authorization boundary — use appropriately scoped store credentials.

### Constructor

```python
VectorStoreHistoryProvider(
    vector_store: BaseVectorStore,
    source_id: str = "vector_store_history",
    *,
    application_id: str,                        # required
    tenant_id: str | None = None,
    agent_id: str | None = None,
    collection_name: str | None = None,         # required when embedding_generator is set
    contents_format: Literal["json", "msgpack"] = "json",
    embedding_generator: EmbeddingClient | None = None,
    embedding_options: Mapping[str, Any] | None = None,  # must include "dimensions" when embedding_generator is set
    compaction_strategy: CompactionStrategy | None = None,
    compaction_tokenizer: TokenizerProtocol | None = None,
    include_search_tool: bool = False,          # requires embedding_generator
    search_approval_mode: Literal["always_require", "never_require"] = "never_require",
    load_messages: bool = True,
    store_inputs: bool = True,
    store_context_messages: bool = False,
    store_context_from: set[str] | None = None,
    store_outputs: bool = True,
)
```

| Parameter | Notes |
|---|---|
| `application_id` | Required. Isolates this application's history from all others. |
| `collection_name` | Required when `embedding_generator` is supplied; otherwise derived automatically. |
| `embedding_options` | Must include `"dimensions": int` when `embedding_generator` is supplied. |
| `include_search_tool` | Adds a scoped `search_history` tool to the agent; requires embedding. |
| `contents_format` | `"json"` (text) or `"msgpack"` (base64-encoded binary). |
| `store_context_messages` | Whether to also persist context injected by other providers. |
| `store_context_from` | Restrict context persistence to specific source IDs. |

### Constants

| Constant | Value |
|---|---|
| `DEFAULT_SOURCE_ID` | `"vector_store_history"` |
| `SEARCH_TOOL_NAME` | `"search_history"` |
| `SEARCH_TOOL_DESCRIPTION` | `"Search the full conversation history..."` |

### Methods

| Method | Signature | Notes |
|---|---|---|
| `get_messages` | `async (session_id, *, state=None, **kwargs) → list[Message]` | Returns the full scoped transcript, sorted by creation time. |
| `save_messages` | `async (session_id, messages, *, state=None, **kwargs) → None` | Upserts new messages; assigns IDs to messages that lack one. |
| `clear` | `async (session_id) → None` | Deletes all records for the scoped history. |
| `before_run` | `async (*, agent, session, context, state) → None` | Loads history, runs optional compaction, adds search tool. |

### Example — basic vector-backed history

```python
import asyncio
from agent_framework import (
    Agent, acknowledge_experimental_feature, ExperimentalFeature,
)
from agent_framework._vectors import VectorStoreHistoryProvider
from agent_framework.openai import OpenAIChatClient

# Use any supported vector store, e.g. InMemoryStore (already deep-dived in Vol. 4)
from agent_framework import InMemoryStore

acknowledge_experimental_feature(ExperimentalFeature.VECTOR_STORES)


async def main():
    store = InMemoryStore()
    history_provider = VectorStoreHistoryProvider(
        store,
        application_id="my-app",
        agent_id="support-bot",
    )

    agent = Agent(
        client=OpenAIChatClient(),
        name="support-bot",
        instructions="You are a helpful customer-support agent.",
        context_providers=[history_provider],
    )

    session = agent.create_session()
    await agent.run("Hi, I need help with my order.", session=session)
    await agent.run("It's order #12345.", session=session)

    # Retrieve persisted history directly
    msgs = await history_provider.get_messages(session.session_id)
    print(f"Stored {len(msgs)} messages.")

asyncio.run(main())
```

### Example — history with semantic search tool

```python
import asyncio
from agent_framework import (
    Agent, InMemoryStore, acknowledge_experimental_feature, ExperimentalFeature,
)
from agent_framework._vectors import VectorStoreHistoryProvider
from agent_framework.openai import OpenAIChatClient, OpenAIEmbeddingClient

acknowledge_experimental_feature(ExperimentalFeature.VECTOR_STORES)


async def main():
    store = InMemoryStore()
    history_provider = VectorStoreHistoryProvider(
        store,
        application_id="semantic-app",
        collection_name="chat-history-v1",
        embedding_generator=OpenAIEmbeddingClient(model="text-embedding-3-small"),
        embedding_options={"dimensions": 1536},
        include_search_tool=True,     # adds search_history tool to the agent
        search_approval_mode="never_require",
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Use search_history to recall past conversations.",
        context_providers=[history_provider],
    )

    session = agent.create_session()
    await agent.run("My favourite colour is blue.", session=session)
    # Later in the same session the agent can search for "favourite colour"
    result = await agent.run("What did I say about colours?", session=session)
    print(result.text)

asyncio.run(main())
```

### Example — clear session history

```python
async def reset_user_history(history_provider, session_id: str):
    await history_provider.clear(session_id)
    print(f"Cleared history for session {session_id!r}.")
```

---

## 6. `MemoryStore` (ABC) and `MemoryFileStore`

**Module:** `agent_framework._harness._memory` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.HARNESS` to be acknowledged.

`MemoryStore` is the **abstract base class** for all memory backing stores used by `MemoryContextProvider`. It manages topic-based long-term memory organised as a set of per-topic markdown files plus a `MEMORY.md` index and a transcript archive.

`MemoryFileStore` is the concrete filesystem implementation provided by the framework.

### `MemoryStore` abstract interface

| Method | Signature | Notes |
|---|---|---|
| `get_owner_id` | `(session) → str \| None` | Logical owner for isolation. Default returns `None`. |
| `export_provider_state` | `(session) → dict[str, Any]` | Routing metadata needed to reopen storage across sessions. |
| `import_provider_state` | `(session, *, state) → None` | Restore routing metadata onto a temporary session. |
| `list_topics` | `(session, *, source_id) → list[MemoryTopicRecord]` | **Abstract.** All topic files for the current owner. |
| `get_topic` | `(session, *, source_id, topic) → MemoryTopicRecord` | **Abstract.** One topic by name or slug. |
| `write_topic` | `(session, record, *, source_id) → None` | **Abstract.** Persist a topic file. |
| `delete_topic` | `(session, *, source_id, topic) → None` | **Abstract.** Remove a topic file. |
| `rebuild_index` | `(session, *, source_id, line_limit, line_length) → list[MemoryIndexEntry]` | **Abstract.** Rebuild `MEMORY.md` from current topic files. |
| `get_index_text` | `(session, *, source_id, line_limit, line_length, index_entries=None) → str` | **Abstract.** Return current `MEMORY.md` text. |
| `read_state` | `(session, *, source_id) → dict[str, Any]` | **Abstract.** Read maintenance state JSON. |
| `write_state` | `(session, state, *, source_id) → None` | **Abstract.** Write maintenance state JSON. |
| `get_transcripts_directory` | `(session, *, source_id) → Path` | **Abstract.** Owner-level transcript archive directory. |
| `search_transcripts` | `(session, *, source_id, query, session_id=None, limit=20) → list[dict]` | **Abstract.** Full-text search over the JSONL transcript archive. |

### `MemoryFileStore` constructor

```python
MemoryFileStore(
    base_path: str | Path,
    *,
    kind: str = "memory",
    owner_prefix: str = "",
    owner_state_key: str,       # session state key holding the logical owner ID
    index_file_name: str = "MEMORY.md",
    topics_directory_name: str = "topics",
    transcripts_directory_name: str = "transcripts",
    state_file_name: str = "state.json",
    dumps: JsonDumps | None = None,
    loads: JsonLoads | None = None,
)
```

| Parameter | Notes |
|---|---|
| `base_path` | Root directory for all memory data. |
| `owner_state_key` | Session state key that resolves to the logical owner ID (e.g. user ID). Required. |
| `kind` | Subdirectory bucket name within each owner root. Useful to separate memory types. |
| `owner_prefix` | String prepended to the resolved owner ID for namespacing. |
| `dumps` / `loads` | Custom JSON serialization hooks (defaults to `json.dumps` / `json.loads`). |

**Path resolution** follows `base_path / source_component / owner_component / kind`. Path traversal in owner IDs (`..`, absolute paths) raises `ValueError`.

### Example — file-backed memory with `MemoryContextProvider`

```python
import asyncio
from agent_framework import (
    Agent, MemoryContextProvider, acknowledge_experimental_feature, ExperimentalFeature,
)
from agent_framework._harness._memory import MemoryFileStore
from agent_framework.openai import OpenAIChatClient

acknowledge_experimental_feature(ExperimentalFeature.HARNESS)


async def main():
    store = MemoryFileStore(
        base_path="/tmp/agent-memory",
        owner_state_key="user_id",
    )

    # MemoryContextProvider uses consolidation_client for the model that writes memories;
    # it does not accept a separate memory_agent argument.
    provider = MemoryContextProvider(
        store=store,
        consolidation_client=OpenAIChatClient(),
    )

    agent = Agent(
        client=OpenAIChatClient(),
        name="assistant",
        instructions="You remember details about the user from past sessions.",
        context_providers=[provider],
    )

    session = agent.create_session()
    session.state["user_id"] = "user-42"

    result = await agent.run("My cat's name is Mochi.", session=session)
    print(result.text)

    result2 = await agent.run("What's my cat's name?", session=session)
    print(result2.text)   # Should recall "Mochi"

asyncio.run(main())
```

### Example — custom `MemoryStore` implementation

```python
import re
from pathlib import Path
from agent_framework import AgentSession
from agent_framework._harness._memory import MemoryStore, MemoryTopicRecord, MemoryIndexEntry


def _safe(s: str) -> str:
    """Sanitize a string for use as a path component."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', s)[:64] or 'default'


class InMemoryMemoryStore(MemoryStore):
    """In-memory MemoryStore for unit testing."""

    def __init__(self):
        # Outer key: (source_id, owner) — isolates each provider instance per user
        self._topics: dict[tuple[str, str], dict[str, MemoryTopicRecord]] = {}  # topic → record
        self._slug_idx: dict[tuple[str, str], dict[str, str]] = {}              # slug → topic
        self._states: dict[tuple[str, str], dict] = {}
        self._tmp = Path("/tmp/in-memory-store-transcripts")

    def _key(self, session: AgentSession, source_id: str) -> tuple[str, str]:
        return (source_id, str(session.state.get("user_id", "default")))

    def get_owner_id(self, session: AgentSession) -> str:
        return str(session.state.get("user_id", "default"))

    def list_topics(self, session, *, source_id):
        return sorted(self._topics.get(self._key(session, source_id), {}).values(),
                      key=lambda r: r.topic)

    def get_topic(self, session, *, source_id, topic):
        key = self._key(session, source_id)
        # resolve slug to canonical topic name if needed
        resolved = self._slug_idx.get(key, {}).get(topic, topic)
        record = self._topics.get(key, {}).get(resolved)
        if record is None:
            raise FileNotFoundError(topic)
        return record

    def write_topic(self, session, record, *, source_id):
        key = self._key(session, source_id)
        self._topics.setdefault(key, {})[record.topic] = record
        self._slug_idx.setdefault(key, {})[record.slug] = record.topic

    def delete_topic(self, session, *, source_id, topic):
        key = self._key(session, source_id)
        resolved = self._slug_idx.get(key, {}).get(topic, topic)
        rec = self._topics.get(key, {}).pop(resolved, None)
        if rec is not None:
            self._slug_idx.get(key, {}).pop(rec.slug, None)

    def rebuild_index(self, session, *, source_id, line_limit, line_length):
        return [MemoryIndexEntry.from_topic_record(t) for t in self.list_topics(session, source_id=source_id)]

    def get_index_text(self, session, *, source_id, line_limit, line_length, index_entries=None):
        entries = index_entries or self.rebuild_index(session, source_id=source_id,
                                                     line_limit=line_limit, line_length=line_length)
        return "\n".join(e.to_pointer_line(max_length=line_length) for e in entries)

    def read_state(self, session, *, source_id):
        return dict(self._states.get(self._key(session, source_id), {}))

    def write_state(self, session, state, *, source_id):
        self._states[self._key(session, source_id)] = dict(state)

    def get_transcripts_directory(self, session, *, source_id):
        owner = self._key(session, source_id)[1]
        # Sanitize both components to prevent path traversal
        scoped = self._tmp / _safe(source_id) / _safe(owner)
        scoped.mkdir(parents=True, exist_ok=True)
        return scoped

    def search_transcripts(self, session, *, source_id, query, session_id=None, limit=20):
        return []
```

---

## 7. `MemoryTopicRecord`

**Module:** `agent_framework._harness._memory` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.HARNESS`.

`MemoryTopicRecord` represents one **topic memory file** — the unit of long-term memory storage. Each record has a human-readable topic, a stable `slug` (filesystem name), a short `summary`, a deduplicated list of `memories` (bullet points), a timestamp, and the session IDs that contributed to this topic.

### Constructor

```python
MemoryTopicRecord(
    *,
    topic: str,
    slug: str | None = None,       # derived from topic if omitted
    summary: str,
    memories: Sequence[str],
    updated_at: str,               # ISO 8601 timestamp string
    session_ids: Sequence[str] | None = None,
)
```

| Parameter | Notes |
|---|---|
| `topic` | Human-readable name. Normalised (whitespace collapsed, stripped). |
| `slug` | Filesystem stem for the `.md` file. Derived automatically from `topic` when omitted. |
| `memories` | Deduplicated list of durable bullet-point memories. |
| `summary` | Short topic summary. Required for meaningful index rendering. |
| `updated_at` | Last-updated ISO timestamp. |
| `session_ids` | Sessions that contributed to this topic. Deduplicated. |

### Attributes (all from `__slots__`)

`topic`, `slug`, `summary`, `memories`, `updated_at`, `session_ids`.

### Methods

| Method | Notes |
|---|---|
| `to_dict() → dict[str, Any]` | JSON-compatible serialization. |
| `from_dict(raw_record) → MemoryTopicRecord` | Deserialize from a dict. Validates required fields. |
| `to_markdown() → str` | Render the canonical on-disk markdown format. |
| `from_markdown(markdown, *, fallback_topic=None) → MemoryTopicRecord` | Parse from the canonical markdown format. |
| `__eq__` | Value equality via `to_dict()`. |

### Markdown format

```markdown
# Travel Plans

Updated: 2025-10-01T10:00:00
Sessions: session-001, session-002

## Summary
User's upcoming travel plans and preferences.

## Memories
- User is flying to Tokyo in November 2025.
- User prefers window seats on long-haul flights.
- User wants to visit Shibuya and Shinjuku.
```

### Example — creating and serializing a topic record

```python
from agent_framework._harness._memory import MemoryTopicRecord

record = MemoryTopicRecord(
    topic="Travel Plans",
    summary="User's upcoming travel plans and preferences.",
    memories=[
        "User is flying to Tokyo in November 2025.",
        "User prefers window seats on long-haul flights.",
        "User wants to visit Shibuya and Shinjuku.",
    ],
    updated_at="2025-10-01T10:00:00",
    session_ids=["session-001"],
)

print(record.slug)          # "travel-plans"
print(record.to_markdown())
print(record.to_dict())
```

### Example — round-tripping through markdown

```python
from agent_framework._harness._memory import MemoryTopicRecord

original = MemoryTopicRecord(
    topic="Dietary Preferences",
    summary="User's food preferences and restrictions.",
    memories=["User is lactose intolerant.", "User loves sushi."],
    updated_at="2025-09-15T08:30:00",
)

md = original.to_markdown()
restored = MemoryTopicRecord.from_markdown(md)
assert restored == original
```

### Example — searching memory topics

```python
from agent_framework._harness._memory import MemoryFileStore, MemoryTopicRecord
from agent_framework import AgentSession


def find_topics_matching(store: MemoryFileStore, session: AgentSession, keyword: str):
    """Return all topics whose memories contain the given keyword."""
    return [
        record
        for record in store.list_topics(session, source_id="memory")
        if any(keyword.lower() in m.lower() for m in record.memories)
    ]
```

---

## 8. `WorkflowRunResult`

**Module:** `agent_framework._workflows._workflow` (re-exported via `agent_framework`)

`WorkflowRunResult` is a `list[WorkflowEvent]` subclass returned by `await workflow.run(...)`. It holds the **data-plane** events (executor invocations, completions, outputs, and `request_info` pauses) in the list itself, and the **control-plane** status events in a separate private list accessible via `status_timeline()`.

### Constructor

```python
WorkflowRunResult(
    events: list[WorkflowEvent[Any]],
    status_events: list[WorkflowEvent[Any]] | None = None,
)
```

> The framework constructs `WorkflowRunResult` for you. You receive it from `await workflow.run(...)`.

### Methods

| Method | Returns | Notes |
|---|---|---|
| `get_outputs()` | `list[Any]` | Data from every `"output"` event. The typical way to extract final results. |
| `get_intermediate_outputs()` | `list[Any]` | Data from every `"intermediate"` event. |
| `get_request_info_events()` | `list[WorkflowEvent[Any]]` | Pause events requesting external input. Non-empty when final state is `IDLE_WITH_PENDING_REQUESTS`. |
| `get_final_state()` | `WorkflowRunState` | Last status event's state. Raises `RuntimeError` if no status events were emitted. |
| `status_timeline()` | `list[WorkflowEvent[Any]]` | Ordered list of all status-transition events (control-plane copy). |

Because `WorkflowRunResult` subclasses `list`, you can iterate over it directly to process all data-plane events.

### `WorkflowRunState` enum

| Value | Meaning |
|---|---|
| `STARTED` | Run has begun. |
| `IN_PROGRESS` | Executors are running. |
| `IN_PROGRESS_PENDING_REQUESTS` | Running but paused waiting for external input. |
| `IDLE` | Run completed cleanly. |
| `IDLE_WITH_PENDING_REQUESTS` | Run paused; `get_request_info_events()` has pending items. |
| `FAILED` | Run terminated with an error. |
| `CANCELLED` | Run was cancelled. |

### Example — basic result inspection

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, WorkflowRunState
from agent_framework.openai import OpenAIChatClient


async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        name="haiku-writer",
        instructions="Write a haiku about the given subject.",
    )
    workflow = WorkflowBuilder(start_executor=agent).build()
    result = await workflow.run("spring rain")

    # Primary output
    outputs = result.get_outputs()
    print(f"Outputs: {outputs}")

    # Final state
    state = result.get_final_state()
    print(f"Final state: {state.value}")
    assert state == WorkflowRunState.IDLE

    # Status timeline
    for ev in result.status_timeline():
        print(f"  {ev.state.value}")

asyncio.run(main())
```

### Example — multi-agent pipeline output extraction

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient


async def main():
    client = OpenAIChatClient()
    researcher = Agent(client=client, name="researcher",
                       instructions="Research the given topic and write 3 key facts.")
    writer = Agent(client=client, id="writer", name="writer",
                   instructions="Turn the researcher's facts into a polished paragraph.")

    workflow = (
        WorkflowBuilder(start_executor=researcher)
        .add_edge(researcher, writer)
        .build()
    )
    result = await workflow.run("quantum computing")

    # All outputs in order
    for i, output in enumerate(result.get_outputs()):
        print(f"Output {i + 1}: {output}")

    # Only the writer's output
    writer_outputs = [
        ev.data for ev in result
        if ev.type == "output" and ev.executor_id == "writer"
    ]
    print(f"Writer: {writer_outputs}")

asyncio.run(main())
```

### Example — detecting pending requests

```python
import asyncio
from agent_framework import WorkflowRunState


async def run_and_handle(workflow, initial_prompt: str, checkpoint_id: str):
    result = await workflow.run(initial_prompt)

    # Collect ALL pending answers before resuming; each workflow.run(responses=...)
    # call restarts from the stored checkpoint, so all answers must go in one map.
    while result.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS:
        pending = result.get_request_info_events()
        responses = {}
        for req_event in pending:
            print(f"Workflow is asking ({req_event.source_executor_id}): {req_event.data}")
            responses[req_event.request_id] = await asyncio.to_thread(input, "Your answer: ")
        result = await workflow.run(
            responses=responses,
            checkpoint_id=checkpoint_id,
        )
        # Advance to the new checkpoint written by this round
        checkpoint_id = result.checkpoint_id

    return result.get_outputs()
```

---

## 9. `FunctionInvocationContext`

**Module:** `agent_framework._middleware` (re-exported via `agent_framework`)

`FunctionInvocationContext` is the mutable context object passed through the **function middleware** pipeline on every tool invocation. It mirrors `AgentContext` but scopes to a single tool call. A key capability added in 1.16.0+ is **progressive tool exposure**: tools can add or remove other tools from the live run via `add_tools()` / `remove_tools()`.

### Constructor

```python
FunctionInvocationContext(
    function: FunctionTool,
    arguments: BaseModel | Mapping[str, Any],
    session: AgentSession | None = None,
    metadata: Mapping[str, Any] | None = None,
    result: Any = None,
    kwargs: Mapping[str, Any] | None = None,
    tools: list[ToolTypes] | None = None,   # live tool list for progressive exposure
)
```

> The framework constructs `FunctionInvocationContext` for you.

### Attributes

| Attribute | Type | Notes |
|---|---|---|
| `function` | `FunctionTool` | The tool being called. |
| `arguments` | `BaseModel \| Mapping[str, Any]` | Parsed tool arguments. May be raw JSON-parsed mapping if provisional validation failed. |
| `session` | `AgentSession \| None` | Current session, or `None` outside a session run. |
| `metadata` | `dict[str, Any]` | Scratchpad shared between function middleware layers on this invocation. |
| `result` | `Any` | Tool result set after `call_next()`. Replace to override. `list[Content]` or `str` passed through intact; other types are stringified. |
| `kwargs` | `dict[str, Any]` | Extra kwargs forwarded to the tool. |
| `tools` | `list[ToolTypes] \| None` | **Live** mutable tool list for the current agent run. `None` outside a function-calling loop. |

### Methods (experimental: `ExperimentalFeature.PROGRESSIVE_TOOLS`)

```python
context.add_tools(
    tools: ToolTypes | Callable | Sequence[...]
) → None
```
Add tools to the live run. Duplicate names raise `ValueError` if the duplicate is a different object. Takes effect on the **next** model iteration.

```python
context.remove_tools(
    tools: ToolTypes | Callable | Sequence[...] | str | Sequence[str]
) → None
```
Remove tools by object, callable, or name string. Unknown names are silently ignored. Takes effect on the next model iteration.

### Example — argument logging middleware

```python
import json
from agent_framework import FunctionMiddleware, FunctionInvocationContext


class ToolAuditMiddleware(FunctionMiddleware):
    async def process(self, context: FunctionInvocationContext, call_next):
        args = context.arguments
        args_str = json.dumps(dict(args), default=str) if hasattr(args, "items") else repr(args)
        print(f"[Audit] {context.function.name}({args_str})")

        await call_next()

        print(f"[Audit] {context.function.name} → {context.result!r}")
```

### Example — argument pattern-matching middleware

> **Security note:** A substring blocklist is **not** a reliable SQL injection guard — it covers only a small subset of payloads, ignores nested structures and encoding variants, and should never be the primary defence. Real SQL injection protection requires parameterized queries (e.g. SQLAlchemy bound parameters). The example below shows how to use function middleware to inspect and reject tool arguments by pattern, which is useful for logging, rate-limiting, or format validation — not as a security boundary.

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext, MiddlewareTermination


class ToolInputPatternGuard(FunctionMiddleware):
    """Illustrative example: block tool calls containing specific substrings."""
    BLOCKED = {"'; drop table", "union select", "--"}

    async def process(self, context: FunctionInvocationContext, call_next):
        args = context.arguments
        args_dict = (
            args.model_dump() if hasattr(args, "model_dump") else dict(args or {})
        )
        for value in args_dict.values():
            if isinstance(value, str):
                lower = value.lower()
                if any(bad in lower for bad in self.BLOCKED):
                    raise MiddlewareTermination("Input blocked by pattern guard.")
        await call_next()
```

### Example — progressive tool exposure

```python
import asyncio
from agent_framework import (
    Agent, FunctionInvocationContext, tool,
    acknowledge_experimental_feature, ExperimentalFeature,
)
from agent_framework.openai import OpenAIChatClient

acknowledge_experimental_feature(ExperimentalFeature.PROGRESSIVE_TOOLS)


@tool
def factorial(n: int) -> int:
    """Compute n!"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@tool
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@tool
def load_math_tools(ctx: FunctionInvocationContext) -> str:
    """Load advanced math tools into this conversation."""
    ctx.add_tools([factorial, fibonacci])
    return "Math tools loaded: factorial, fibonacci."


async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        name="math-bot",
        instructions="You have access to load_math_tools. Call it before doing factorial or fibonacci calculations.",
        tools=[load_math_tools],  # only this tool initially
    )
    result = await agent.run("What is 7! and the 10th Fibonacci number?")
    print(result.text)

asyncio.run(main())
```

### Example — injecting per-call kwargs via agent middleware

```python
from agent_framework import AgentMiddleware, AgentContext


class TraceIdMiddleware(AgentMiddleware):
    """Inject a trace ID into every tool invocation."""

    def __init__(self, trace_id: str):
        self._trace_id = trace_id

    async def process(self, context: AgentContext, call_next):
        context.function_invocation_kwargs["trace_id"] = self._trace_id
        await call_next()
```

---

## 10. `ChatOptions`

**Module:** `agent_framework._types` (re-exported via `agent_framework`)

`ChatOptions` is a `TypedDict` (total=False) that describes the common request parameters accepted by all `agent_framework` chat clients. All fields are **optional**, allowing partial specification. Individual providers may raise errors for unsupported options.

### Fields

| Field | Type | Notes |
|---|---|---|
| `model` | `str` | Model identifier (e.g. `"gpt-4o"`, `"claude-sonnet-5"`). |
| `temperature` | `float` | Sampling temperature (0–2 for most providers). |
| `top_p` | `float` | Nucleus sampling probability mass. |
| `max_tokens` | `int` | Maximum tokens in the response. |
| `stop` | `str \| Sequence[str]` | Stop sequence(s). |
| `seed` | `int` | Seed for deterministic sampling (provider-dependent). |
| `logit_bias` | `dict[str \| int, float]` | Token-level probability adjustments. |
| `frequency_penalty` | `float` | Penalise token frequency. |
| `presence_penalty` | `float` | Penalise token presence. |
| `tools` | tool types | Tool set for this call. |
| `tool_choice` | `ToolMode \| "auto" \| "required" \| "none"` | How the model selects tools. |
| `allow_multiple_tool_calls` | `bool` | Whether to allow more than one tool call per response. |
| `response_format` | `type[BaseModel] \| Mapping[str, Any] \| None` | Structured output schema. |
| `metadata` | `dict[str, Any]` | Provider-specific metadata. |
| `user` | `str` | End-user identifier (e.g. for OpenAI abuse monitoring). |
| `store` | `bool` | Whether to persist the conversation server-side (provider-specific; e.g. OpenAI Responses API, Foundry). |
| `conversation_id` | `str` | Conversation identifier (provider-specific). |
| `instructions` | `str` | System-level instructions override. |

### Usage patterns

`ChatOptions` is used as:

1. **`Agent` constructor `default_options`** — applied on every run unless overridden.
2. **Per-run override** via `agent.run(..., options={...})`.
3. **`Unpack[ChatOptions]` function signatures** for type-safe option forwarding.

### Example — setting default options on an agent

```python
import asyncio
from agent_framework import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient


async def main():
    options: ChatOptions = {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 512,
    }

    agent = Agent(
        client=OpenAIChatClient(),
        name="concise-bot",
        instructions="Be brief.",
        default_options=options,
    )

    result = await agent.run("Explain black holes.")
    print(result.text)

asyncio.run(main())
```

### Example — per-run override

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a creative writer.",
        default_options={"temperature": 0.7},
    )

    # Override temperature for this specific call via options=
    creative_result = await agent.run("Write a haiku.", options={"temperature": 1.2})
    print(creative_result.text)

    # Use a different model for a specific call
    fast_result = await agent.run("Summarize AI.", options={"model": "gpt-4o-mini", "max_tokens": 50})
    print(fast_result.text)

asyncio.run(main())
```

### Example — structured output with `response_format`

```python
import asyncio
from pydantic import BaseModel
from agent_framework import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient


class Haiku(BaseModel):
    line1: str
    line2: str
    line3: str


async def main():
    options: ChatOptions = {
        "response_format": Haiku,
        "temperature": 0.9,
    }

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Write a haiku about the given subject. Reply as JSON.",
        default_options=options,
    )

    result = await agent.run("autumn leaves")
    # result.value holds the validated Pydantic model when response_format is set
    haiku: Haiku = result.value
    print(f"{haiku.line1} / {haiku.line2} / {haiku.line3}")

asyncio.run(main())
```

### Example — type-safe option forwarding

```python
# typing.Unpack requires Python 3.11+; use typing_extensions on Python 3.10
from typing_extensions import Unpack
from agent_framework import Agent, ChatOptions
from agent_framework.openai import OpenAIChatClient


class ConfigurableAgent:
    def __init__(self, **default_options: Unpack[ChatOptions]):
        self._agent = Agent(
            client=OpenAIChatClient(),
            instructions="You are a helpful assistant.",
            default_options=dict(default_options),
        )

    async def ask(self, prompt: str, **override: Unpack[ChatOptions]) -> str:
        result = await self._agent.run(prompt, options=dict(override))
        return result.text


# Usage:
# bot = ConfigurableAgent(model="gpt-4o", temperature=0.5)
# answer = await bot.ask("What is ML?", max_tokens=100)
```

---

## What's new in 1.19.0

The 1.19.0 release refines several of the APIs deep-dived in this and prior volumes. Key areas:

| Area | Change |
|---|---|
| **Progressive tools** | `FunctionInvocationContext.add_tools()` / `remove_tools()` stabilised under `ExperimentalFeature.PROGRESSIVE_TOOLS`. All-or-nothing batch semantics: a duplicate name raises before the live list is mutated. |
| **Vector history** | `VectorStoreHistoryProvider` adds `store_context_from` for fine-grained control over which source IDs have their context messages persisted. |
| **Memory harness** | `MemoryFileStore.search_transcripts` now resolves the target transcript file stem via `_transcript_file_stem()` — supporting even very long session IDs stored under an irreversible digest. |
| **WorkflowEvent** | `WorkflowEvent.executor_bypassed` documents the cache-hit replay path more precisely. The `emit()` factory deprecation warning is now emitted with `stacklevel=2` for correct source attribution. |
| **ChatOptions** | `conversation_id` field added for providers that support conversation-level threading. |

---

## See also

- [Python Comprehensive Guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/) — framework overview, verified against 1.19.0
- [Class Deep Dives Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — workflow visualization, file memory, background agents, tool approval
- [Class Deep Dives Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — fan-in/out edges, functional workflows, checkpointing, MCP tools
- [Class Deep Dives Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — workflow builder, compaction strategies, evaluation, inline skills, file access
- [Class Deep Dives Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — vector store fields, in-memory collections, filters, settings, group chat

---
title: "Microsoft Agent Framework (Python) — 10-API Deep Dives Vol. 9 (1.22.0)"
description: "Source-verified deep dives for ResponseStream, AgentResponseUpdate, SessionContext, FileSessionStore, AgentLoopMiddleware, SingleEdgeGroup, WorkflowInvocationKwargs, StepWrapper, DelegatingSkillsSource, and SubWorkflowRequestMessage/SubWorkflowResponseMessage — all verified against agent-framework 1.22.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-API Deep Dives Vol. 9

**Verified against:** `agent-framework==1.22.0`
**Python requirement:** 3.10+

This volume covers the streaming infrastructure layer (`ResponseStream`, `AgentResponseUpdate`), the context-provider pipeline state (`SessionContext`), file-backed session persistence (`FileSessionStore`), the judge-driven loop middleware (`AgentLoopMiddleware`), low-level workflow graph primitives (`SingleEdgeGroup`, `WorkflowInvocationKwargs`), the functional-workflow step decorator wrapper (`StepWrapper`), the skills-source decorator base (`DelegatingSkillsSource`), and the sub-workflow message-passing pair (`SubWorkflowRequestMessage` / `SubWorkflowResponseMessage`). Each section includes the full verified constructor signature, every meaningful method or factory, and self-contained runnable examples.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

See [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) for `FanInEdgeGroup`, `FanOutEdgeGroup`, `FunctionalWorkflow`, `FunctionalWorkflowAgent`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `MCPStdioTool`, `MCPStreamableHTTPTool`, `SelectiveToolCallCompactionStrategy`, and `TodoProvider`.

See [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) for `WorkflowBuilder`, `SlidingWindowStrategy`, `TruncationStrategy`, `ContextWindowCompactionStrategy`, `LocalEvaluator`, `InlineSkill`, `FileAccessProvider`, `MemoryContextProvider`, `FileHistoryProvider`, and `MCPWebsocketTool`.

See [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) for `VectorStoreField`, `VectorStoreCollectionDefinition`, `InMemoryCollection`, `InMemoryStore`, `Filter`, `FilterGroup`, `SecretString`, `load_settings`, `create_agent_hooks_middleware`, and `GroupChatBuilder`.

See [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) for `WorkflowEvent`, `AgentContext`, `MiddlewareBundle`, `ConversationSplit`/`ConversationSplitter`, `VectorStoreHistoryProvider`, `MemoryStore`/`MemoryFileStore`, `MemoryTopicRecord`, `WorkflowRunResult`, `FunctionInvocationContext`, and `ChatOptions`.

See [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) for `Agent`, `RawAgent`, `AgentSession`, `AgentResponse`, `FunctionTool`, `WorkflowAgent`, `WorkflowContext`, `CompactionProvider`, `VectorCollectionContextProvider`, and `TodoItem`/`TodoStore`/`TodoFileStore`.

See [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) for `AgentExecutorRequest`, `AgentExecutorResponse`, `AgentExecutorCheckpointState`, `AgentExecutor`, `WorkflowExecutor`, `WorkflowMessage`, `WorkflowCheckpoint`, `TodoSessionStore`, `ClassSkill`, and `AggregatingSkillsSource`/`CachingSkillsSource`.

See [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) for `InProcRunnerContext`, `FunctionExecutor`, `EvalItem`, `EvalResults`/`EvalItemResult`, `ChatResponse`/`ChatResponseUpdate`, `UsageDetails`, `FileSkillsSource`, `MCPSkillsSource`, `DeduplicatingSkillsSource`/`FilteringSkillsSource`, and `AgentFileStore`/`InMemoryAgentFileStore`/`FileSystemAgentFileStore`.

---

## 1. `ResponseStream[UpdateT, FinalT]`

**Module:** `agent_framework._types` (re-exported via `agent_framework`)

`ResponseStream` is the generic async-iterable wrapper that backs every streaming call in the framework. `Agent.run(stream=True)` returns `ResponseStream[AgentResponseUpdate, AgentResponse]`; `Workflow.run(stream=True)` returns `ResponseStream[WorkflowEvent, WorkflowRunResult]`; a chat client's `get_streaming_response()` returns `ResponseStream[ChatResponseUpdate, ChatResponse]`. Consuming the stream yields individual update chunks; after full consumption call `await stream.get_final_response()` to obtain the aggregated final value.

> **When you need it directly:** most code simply `async for`-iterates the stream or wraps it with `async with`. Reach for the factory methods (`map`, `flat_map`, `with_finalizer`, `buffered_and_gated`) when you need to transform or gate a stream without losing its type safety.

### Constructor

```python
ResponseStream(
    stream: AsyncIterable[UpdateT] | Awaitable[AsyncIterable[UpdateT]],
    *,
    finalizer: Callable[[Sequence[UpdateT]], FinalT | Awaitable[FinalT]] | None = None,
    transform_hooks: Sequence[Callable[[UpdateT], UpdateT | Awaitable[UpdateT | None] | None]] | None = None,
    cleanup_hooks: Sequence[Callable[[], Awaitable[None] | None]] | None = None,
    result_hooks: Sequence[Callable[[FinalT], FinalT | Awaitable[FinalT | None] | None]] | None = None,
    update_transforms: Sequence[...] | None = None,   # preferred alias for transform_hooks
    result_transforms: Sequence[...] | None = None,   # preferred alias for result_hooks
    stream_updates: bool = True,
    result_to_updates: Callable[[FinalT], Sequence[UpdateT]] | None = None,
)
```

| Parameter | Notes |
|---|---|
| `stream` | The raw async-iterable source, or an awaitable that resolves to one. |
| `finalizer` | Combines all collected updates into the single `FinalT`. When `None` the result is `None`. |
| `update_transforms` | Called in order for each yielded update; return `None` to suppress the update. Alias for the older `transform_hooks`. |
| `result_transforms` | Called in order on the finalized result before it is published to `.result`. Alias for `result_hooks`. |
| `cleanup_hooks` | Run after the stream is exhausted (before the finalizer). |
| `stream_updates` | When `False`, updates are buffered and only released after the finalizer and all gates pass. |
| `result_to_updates` | Required when `stream_updates=False` and a `result_transforms` replaces the final value. |

### Key methods and properties

| Member | Description |
|---|---|
| `async for update in stream` | Iterate updates as they arrive. |
| `await stream.get_final_response()` | `FinalT` — obtain the aggregated final result after fully consuming the stream. |
| `map(fn, finalizer)` | Return a new `ResponseStream` that applies `fn` to every update; `finalizer` builds the final value from the transformed updates. |
| `flat_map(fn, finalizer)` | Return a new stream where each update is replaced by zero-or-more updates from `fn`; `finalizer` is required as the second positional argument. |
| `with_finalizer(fn)` | Return a new stream with a different finalizer, preserving all hooks. |
| `from_awaitable(aw)` | Class method — wrap an awaitable-of-stream; inherits the inner stream's finalizer (no `finalizer` keyword). |
| `buffered_and_gated(consume, gate, rederive)` | Class method — create a buffered stream that only releases after all gates pass. |

### Example 1 — iterate streaming agent response

```python
import asyncio
from agent_framework import Agent, Content

async def main():
    agent = Agent(client=my_client)

    stream = agent.run("Summarise this article in three bullet points.", stream=True)

    # Consume updates as they arrive
    async for update in stream:
        # update is AgentResponseUpdate
        print(update.text, end="", flush=True)

    # Full aggregated response available after iteration
    response = await stream.get_final_response()
    tokens = (response.usage_details or {}).get("total_token_count")
    print(f"\nTotal tokens: {tokens}")

asyncio.run(main())
```

### Example 2 — async context manager (auto-close on exit)

```python
async def stream_with_timeout(agent: Agent, message: str) -> str:
    stream = agent.run(message, stream=True)
    async for update in stream:
        print(update.text, end="")
    return (await stream.get_final_response()).text
```

### Example 3 — transform every update with `map`

```python
from agent_framework import ResponseStream, AgentResponseUpdate

def uppercase_update(u: AgentResponseUpdate) -> AgentResponseUpdate:
    from agent_framework import Content
    return AgentResponseUpdate(
        contents=[Content.from_text(u.text.upper())],
        role=u.role,
        response_id=u.response_id,
    )

# Wrap a stream produced elsewhere — map requires both an update transformer and a finalizer
from agent_framework import AgentResponse
transformed = original_stream.map(
    uppercase_update,
    lambda updates: AgentResponse.from_updates(updates),
)
async for update in transformed:
    print(update.text)
```

### Example 4 — manually buffer and gate a stream

```python
from agent_framework import AgentResponseUpdate, AgentResponse

def safety_gate(update: AgentResponseUpdate) -> None:
    # Raise to block the entire stream
    if "BLOCKED" in update.text:
        raise ValueError("Blocked content detected")

async def buffered_and_gated(raw_stream, gate_fn) -> AgentResponse:
    """Collect all updates, apply the gate function, then return the full response."""
    updates: list[AgentResponseUpdate] = []
    async for update in raw_stream:
        updates.append(update)
    for update in updates:
        gate_fn(update)  # raises ValueError if blocked
    return AgentResponse.from_updates(updates)

response = await buffered_and_gated(raw_stream, safety_gate)
print(response.text)
```

---

## 2. `AgentResponseUpdate`

**Module:** `agent_framework._types` (re-exported via `agent_framework`)

`AgentResponseUpdate` is a single streaming response chunk yielded by `Agent.run(..., stream=True)`. Many chunks combine to form the final `AgentResponse`. Each chunk carries an optional role, author identification (useful in multi-agent scenarios), content items, and merge mode semantics.

> **Relation to `ChatResponseUpdate`:** `ChatResponseUpdate` (Vol. 8) is the lower-level chat-client chunk. `AgentResponseUpdate` is the higher-level agent wrapper — it adds `author_name`, `agent_id`, and `merge_mode` and is what application code normally sees.

### Constructor

```python
AgentResponseUpdate(
    *,
    contents: Sequence[Content] | None = None,
    role: RoleLiteral | str | None = None,
    author_name: str | None = None,
    agent_id: str | None = None,
    response_id: str | None = None,
    message_id: str | None = None,
    merge_mode: Literal["append", "replace"] | None = None,
    created_at: datetime | str | None = None,
    additional_properties: dict[str, Any] | None = None,
    raw_representation: Any = None,
)
```

| Parameter | Notes |
|---|---|
| `contents` | Content items in this chunk. |
| `role` | `"assistant"`, `"user"`, etc. |
| `author_name` | Which agent produced the chunk in a multi-agent scenario. |
| `agent_id` | Stable identifier of the producing agent. |
| `response_id` | Groups chunks that belong to the same logical response. |
| `merge_mode` | `"append"` (default) accumulates text; `"replace"` signals a full replacement. |

### Properties and methods

| Member | Description |
|---|---|
| `update.text` | `str` — concatenated text of all text content items. |
| `update.user_input_requests` | `list[Content]` — any user-input-request content items in this chunk. |
| `update.to_dict(*, exclude_none=True)` | Serialize to a JSON-safe dict. |
| `AgentResponseUpdate.from_dict(d)` | Deserialize from a dict. |
| `update.to_json()` | Serialize to a JSON string. |
| `AgentResponseUpdate.from_json(s)` | Deserialize from a JSON string. |

### Example 1 — stream and track which agent produced each chunk

```python
import asyncio
from agent_framework import Agent

async def multi_agent_stream(orchestrator: Agent, question: str) -> None:
    stream = orchestrator.run(question, stream=True)
    current_author = None

    async for update in stream:
        if update.author_name != current_author:
            current_author = update.author_name
            print(f"\n[{current_author or 'unknown'}]: ", end="")
        print(update.text, end="", flush=True)

    final = await stream.get_final_response()
    print(f"\n\nTotal messages: {len(final.messages)}")

asyncio.run(multi_agent_stream(agent, "Compare the Q3 and Q4 revenue numbers."))
```

### Example 2 — collect all updates and replay them

```python
import asyncio
from agent_framework import Agent, AgentResponseUpdate

async def collect_and_replay(agent: Agent, question: str) -> list[AgentResponseUpdate]:
    updates: list[AgentResponseUpdate] = []
    async for update in agent.run(question, stream=True):
        updates.append(update)
    return updates

# Replaying stored updates (e.g., for testing or audit logging)
async def replay(updates: list[AgentResponseUpdate]) -> str:
    return "".join(u.text for u in updates)
```

### Example 3 — roundtrip serialisation

```python
from agent_framework import AgentResponseUpdate, Content

update = AgentResponseUpdate(
    contents=[Content.from_text("Hello from the agent!")],
    role="assistant",
    author_name="research-agent",
    response_id="resp_abc123",
)

payload = update.to_dict()
# {'type': 'agent_response_update', 'contents': [...], 'role': 'assistant', ...}

restored = AgentResponseUpdate.from_dict(payload)
assert restored.text == "Hello from the agent!"
assert restored.author_name == "research-agent"
```

---

## 3. `SessionContext`

**Module:** `agent_framework._sessions` (re-exported via `agent_framework`)

`SessionContext` is the mutable state bag that flows through the context-provider pipeline on every `Agent.run()` call. Providers registered with `Agent(context_providers=[...])` receive a freshly created `SessionContext` and may add messages, instructions, tools, and middleware before the underlying chat call is made. After the call, `context.response` holds the complete `AgentResponse`.

> **When you need it:** implement `ContextProvider.before_run(*, agent, session, context, state)` to mutate the context before each call. Reading `context.response` in an `after_run` hook gives you the final response for post-processing (logging, caching, guardrails).

### Constructor

```python
SessionContext(
    *,
    session_id: str | None = None,
    service_session_id: str | None = None,
    input_messages: list[Message],
    context_messages: dict[str, list[Message]] | None = None,
    instructions: list[str] | None = None,
    tools: list[Any] | None = None,
    middleware: dict[str, list[MiddlewareTypes]] | None = None,
    options: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
)
```

### Key attributes and methods

| Member | Description |
|---|---|
| `ctx.session_id` | The session ID, or `None` for stateless calls. |
| `ctx.input_messages` | The messages the caller is sending. |
| `ctx.context_messages` | Dict `source_id → [Message]` added by providers. |
| `ctx.instructions` | Extra instruction strings injected by providers. |
| `ctx.tools` | Extra tool objects injected by providers. |
| `ctx.middleware` | Dict `source_id → [Middleware]` injected by providers. |
| `ctx.response` | `AgentResponse \| None` — set after invocation. |
| `ctx.metadata` | Shared dict for cross-provider communication. |
| `ctx.extend_messages(source_id, messages)` | Append messages under a given source key. |
| `ctx.extend_instructions(source_id, instructions)` | Append instruction strings. |
| `ctx.extend_tools(source_id, tools)` | Append extra tools. |
| `ctx.extend_middleware(source_id, middleware)` | Append middleware instances. |
| `ctx.get_messages()` | Returns all context messages in provider-insertion order. |
| `ctx.get_middleware()` | Returns all middleware in provider-insertion order. |

### Example 1 — write a context provider that injects a retrieval preamble

```python
import asyncio
from agent_framework import Agent, SessionContext, Content
from agent_framework._sessions import ContextProvider  # protocol

class RAGContextProvider(ContextProvider):
    def __init__(self, retriever):
        super().__init__("rag-provider")
        self._retriever = retriever

    async def before_run(self, *, agent, session, context: SessionContext, state) -> None:
        # Pull relevant chunks from the last user message
        query = " ".join(
            c.text
            for msg in context.input_messages
            if msg.role == "user"
            for c in msg.contents
            if hasattr(c, "text")
        )
        chunks = await self._retriever.search(query, top_k=4)
        preamble = "\n\n".join(chunks)
        from agent_framework import Message
        context.extend_messages(
            "rag",
            [Message("system", [f"Relevant context:\n{preamble}"])],
        )

agent = Agent(client=my_client, context_providers=[RAGContextProvider(retriever)])
response = await agent.run("What is the refund policy?")
```

### Example 2 — post-invocation hook using `ctx.response`

```python
import json
from pathlib import Path
from agent_framework import SessionContext

class AuditLogProvider(ContextProvider):
    def __init__(self, log_path: Path):
        super().__init__("audit-log")
        self._log_path = log_path

    async def after_run(self, *, agent, session, context: SessionContext, state) -> None:
        if context.response is None:
            return
        record = {
            "session_id": context.session_id,
            "tokens": (context.response.usage_details or {}).get("total_token_count"),
            "messages": len(context.response.messages),
        }
        with self._log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
```

### Example 3 — sharing state between two providers via `metadata`

```python
class TimerStartProvider(ContextProvider):
    def __init__(self):
        super().__init__("timer-start")

    async def before_run(self, *, agent, session, context: SessionContext, state) -> None:
        import time
        context.metadata["start_time"] = time.monotonic()

class TimerEndProvider(ContextProvider):
    def __init__(self):
        super().__init__("timer-end")

    async def after_run(self, *, agent, session, context: SessionContext, state) -> None:
        import time
        elapsed = time.monotonic() - context.metadata.get("start_time", 0)
        print(f"Agent call took {elapsed:.3f}s")

agent = Agent(
    client=my_client,
    context_providers=[TimerStartProvider(), TimerEndProvider()],
)
```

---

## 4. `FileSessionStore`

**Module:** `agent_framework._sessions` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.SESSION_STORE` opt-in.

`FileSessionStore` persists `AgentSession` snapshots to the local filesystem — one file per session. JSON is the default format; pass `serialization_format="msgpack"` for a more compact binary representation. Writes are atomic (temp-file + `os.replace`) so readers never see a partial write. The store is process-local; concurrent writers across processes use last-writer-wins semantics.

### Constructor

```python
FileSessionStore(
    storage_path: str | Path,
    *,
    serialization_format: Literal["json", "msgpack"] = "json",
)
```

| Parameter | Notes |
|---|---|
| `storage_path` | Directory where session files are stored. Created automatically if it does not exist. |
| `serialization_format` | `"json"` (readable, default) or `"msgpack"` (compact binary). |

### Methods

| Method | Signature | Description |
|---|---|---|
| `get` | `async (session_id: str) → AgentSession \| None` | Load a session snapshot. Returns `None` if not found. |
| `set` | `async (session_id: str, session: AgentSession) → None` | Atomically write the session snapshot. |
| `delete` | `async (session_id: str) → None` | Remove the session file. No-op if absent. |
| `validate_session_id` | `(session_id: str) → None` | Raises `ValueError` for empty IDs or IDs longer than 128 characters. Path-traversal safety is enforced separately at the file-path level by the store. |

### Example 1 — persist a session across process restarts

```python
import asyncio
from pathlib import Path
from agent_framework import Agent, AgentSession
from agent_framework._sessions import FileSessionStore
from agent_framework.experimental import enable_experimental, ExperimentalFeature

enable_experimental(ExperimentalFeature.SESSION_STORE)

STORE = FileSessionStore(Path("/var/myapp/sessions"))

async def chat(user_id: str, message: str) -> str:
    session = await STORE.get(user_id) or AgentSession()
    agent = Agent(client=my_client)
    response = await agent.run(message, session=session)
    await STORE.set(user_id, session)
    return response.text

asyncio.run(chat("user-42", "What were we talking about last time?"))
```

### Example 2 — choose MessagePack for high-throughput storage

```python
from pathlib import Path
from agent_framework._sessions import FileSessionStore
from agent_framework.experimental import enable_experimental, ExperimentalFeature

enable_experimental(ExperimentalFeature.SESSION_STORE)

# Binary format is faster to read/write and about 30% smaller than JSON
store = FileSessionStore(
    Path("/var/myapp/sessions"),
    serialization_format="msgpack",
)
```

### Example 3 — validate session IDs before storing

```python
from agent_framework._sessions import FileSessionStore

store = FileSessionStore("/var/myapp/sessions")

def safe_session_id(raw_id: str) -> str:
    try:
        FileSessionStore.validate_session_id(raw_id)
        return raw_id
    except ValueError as e:
        raise ValueError(f"Invalid session ID '{raw_id}': {e}") from e
```

---

## 5. `AgentLoopMiddleware`

**Module:** `agent_framework._harness._loop` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.HARNESS` opt-in.

`AgentLoopMiddleware` is an `AgentMiddleware` that re-runs an agent in a loop until a `should_continue` predicate returns `False` or `max_iterations` is reached. Use `AgentLoopMiddleware.with_judge(judge_client)` to drive the loop with a second LLM that judges whether the original request was satisfactorily answered.

The loop kwargs available to `should_continue` and `next_message` callables are:

| Kwarg | Type | Description |
|---|---|---|
| `iteration` | `int` | Completed run count (1 after first run). |
| `last_result` | `AgentResponse` | Result of the iteration that just finished. |
| `messages` | `list[Message]` | Messages used for the last iteration. |
| `original_messages` | `list[Message]` | Input used for the first iteration. |
| `session` | `AgentSession \| None` | The active session. |
| `agent` | `Agent` | The agent being looped. |
| `progress` | `list[str]` | Feedback log accumulated across iterations. |
| `feedback` | `str \| None` | Feedback string from the current `should_continue` call. |

### Constructor

```python
AgentLoopMiddleware(
    should_continue: ShouldContinueCallable,
    *,
    max_iterations: int | None = 10,
    next_message: NextMessageCallable | None = None,
    return_final_only: bool = False,
    inject_progress: bool = True,
    fresh_context: bool = False,
    record_feedback: Callable[..., str | None] | None = None,
    additional_instructions: str | None = None,
)
```

| Parameter | Notes |
|---|---|
| `should_continue` | `async (*, iteration, last_result, ..., **kwargs) -> bool \| tuple[bool, str \| None]`. Return `(False, reason)` to stop and attach a rationale. |
| `max_iterations` | Hard cap. `None` for unbounded — ensure `should_continue` will eventually stop. |
| `next_message` | Builds the user message for the next iteration. When `None` (default), uses the framework continuation nudge: `"Continue working on the task. If it is complete, say so."` |
| `return_final_only` | When `True`, only the last iteration's `AgentResponse` is returned instead of the aggregate. |
| `inject_progress` | When `True` (default), the accumulated progress log (from `record_feedback`) is injected into each iteration's context. |
| `fresh_context` | When `True`, each iteration restarts from the original request messages (plus the progress log and nudge), and any attached session is restored to its pre-loop snapshot, discarding in-loop history. |
| `record_feedback` | Called after each iteration to append a string to the `progress` log. |
| `additional_instructions` | Optional `system` message injected on every iteration; preserved across `fresh_context` resets. |

### Class method — `with_judge`

```python
@classmethod
AgentLoopMiddleware.with_judge(
    judge_client: SupportsChatGetResponse,
    *,
    criteria: Sequence[str] | None = None,
    instructions: str | None = None,
    response_format: type[BaseModel] | Mapping[str, Any] | None = JudgeVerdict,
    verdict_parser: Callable[[ChatResponse], JudgeVerdict | Awaitable[JudgeVerdict]] | None = None,
    max_iterations: int | None = 5,
    next_message: NextMessageCallable | None = None,
    fresh_context: bool = False,
) -> AgentLoopMiddleware
```

Creates a loop whose stop condition is determined by a second LLM (`judge_client`) that evaluates whether the original request was answered. The judge's reasoning is fed back as the next-iteration input so the agent knows why its previous answer was incomplete.

> **Security:** the judge client receives the original request and the agent's latest response on every iteration — use a client that points to a trusted endpoint.

### Example 1 — loop until the agent says "DONE"

```python
import asyncio
from agent_framework import Agent, AgentResponse
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.experimental import enable_experimental, ExperimentalFeature

enable_experimental(ExperimentalFeature.HARNESS)

async def should_continue(*, iteration: int, last_result: AgentResponse, **kwargs) -> bool:
    return iteration < 5 and "DONE" not in last_result.text

agent = Agent(
    client=my_client,
    middleware=[AgentLoopMiddleware(should_continue, max_iterations=5)],
)

response = await agent.run(
    "Research renewable energy trends and iterate until you have covered at least 3 regions. "
    "When complete, end your response with the word DONE."
)
print(response.text)
```

### Example 2 — judge-driven loop

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.experimental import enable_experimental, ExperimentalFeature

enable_experimental(ExperimentalFeature.HARNESS)

agent = Agent(
    client=my_client,
    middleware=[
        AgentLoopMiddleware.with_judge(
            judge_client=judge_chat_client,
            criteria=[
                "The response must include specific numerical data.",
                "The response must cover at least three distinct geographic regions.",
            ],
            max_iterations=4,
        )
    ],
)

response = await agent.run("Summarise global solar capacity growth in 2024.")
print(response.text)
```

### Example 3 — track progress and inject custom next message

```python
from agent_framework import Agent, AgentResponse, Message
from agent_framework._harness._loop import AgentLoopMiddleware
from agent_framework.experimental import enable_experimental, ExperimentalFeature

enable_experimental(ExperimentalFeature.HARNESS)

async def should_continue(
    *, iteration: int, last_result: AgentResponse, **kwargs
) -> tuple[bool, str | None]:
    if "FINAL ANSWER" in last_result.text:
        return False, "Request fully answered."
    if iteration >= 3:
        return False, "Max guided attempts reached."
    return True, f"Step {iteration} incomplete — continue."

async def build_next_message(*, feedback: str | None, **kwargs) -> Message:
    return Message("user", [feedback or "Please continue."])

agent = Agent(
    client=my_client,
    middleware=[
        AgentLoopMiddleware(
            should_continue,
            next_message=build_next_message,
            return_final_only=True,
        )
    ],
)
```

---

## 6. `SingleEdgeGroup`

**Module:** `agent_framework._workflows._edge` (re-exported via `agent_framework`)

`SingleEdgeGroup` is the internal representation for a single directed edge between two executors. `WorkflowBuilder.add_edge(source_executor, target_executor, condition=None)` accepts executor objects directly and creates a `SingleEdgeGroup` internally. The `SingleEdgeGroup` class is useful for inspection, serialization, and for lower-level workflow graph APIs that accept `EdgeGroup` objects explicitly.

### Constructor

```python
SingleEdgeGroup(
    source_id: str,
    target_id: str,
    condition: EdgeCondition | None = None,
    *,
    id: str | None = None,
)
```

| Parameter | Notes |
|---|---|
| `source_id` | ID of the source executor. |
| `target_id` | ID of the target executor. |
| `condition` | Optional `(data) -> bool | Awaitable[bool]` predicate. Edge is skipped when `False`. |
| `id` | Optional explicit group ID. Auto-generated when omitted. |

### Example 1 — unconditional edge in a `WorkflowBuilder`

```python
from agent_framework import WorkflowBuilder, Agent, SingleEdgeGroup

ingest = Agent(client=my_client, name="ingest")
validate = Agent(client=my_client, name="validate")
report = Agent(client=my_client, name="report")

workflow = (
    WorkflowBuilder(start_executor=ingest)
    .add_edge(ingest, validate)
    .add_edge(validate, report)
    .build()
)
```

### Example 2 — conditional edge (skip validate when ingest output is short)

```python
from agent_framework import SingleEdgeGroup, WorkflowMessage

from agent_framework._workflows._agent_executor import AgentExecutorResponse

# Note: the condition receives the INGEST agent's output (AgentExecutorResponse),
# not the original user input.  These predicates route on output length.
def is_long_ingest_output(data: AgentExecutorResponse) -> bool:
    return len(data.agent_response.text or "") > 200

def is_short_ingest_output(data: AgentExecutorResponse) -> bool:
    return len(data.agent_response.text or "") <= 200

# Mutually exclusive routing: long ingest outputs go through validate → report;
# short ingest outputs skip validation and route directly to report.
workflow = (
    WorkflowBuilder(start_executor=ingest)
    .add_edge(ingest, validate, condition=is_long_ingest_output)
    .add_edge(ingest, report, condition=is_short_ingest_output)
    .add_edge(validate, report)
    .build()
)
```

---

## 7. `WorkflowInvocationKwargs`

**Module:** `agent_framework._workflows._workflow` (re-exported via `agent_framework`)

`WorkflowInvocationKwargs` is a frozen dataclass for passing keyword arguments to a workflow run with optional per-executor overrides. Pass it as the `function_invocation_kwargs` argument to `Workflow.run()` when you need to provide shared defaults AND override them for specific executors in the same call.

> **Plain dict vs. `WorkflowInvocationKwargs`:** passing a plain `dict` distributes the same kwargs to every executor. `WorkflowInvocationKwargs` lets you set global defaults and selectively override them per executor without duplicating keys.

### Constructor

```python
WorkflowInvocationKwargs(
    global_kwargs: Mapping[str, Any] = {},
    executor_kwargs: Mapping[str, Mapping[str, Any]] = {},
)
```

| Parameter | Notes |
|---|---|
| `global_kwargs` | Kwargs passed to every executor in the workflow. |
| `executor_kwargs` | Dict `executor_id → {kwarg: value}` — overrides `global_kwargs` for that executor. |

### Example 1 — per-executor client kwargs overrides

```python
import asyncio
from agent_framework import Workflow, WorkflowInvocationKwargs

async def run_invoice_workflow(workflow: Workflow) -> str:
    # Chat parameters such as temperature belong in client_kwargs, NOT
    # function_invocation_kwargs (which forwards kwargs to tool/function calls).
    # executor_kwargs keys are executor IDs; values are per-executor overrides.
    kwargs = WorkflowInvocationKwargs(
        executor_kwargs={
            "invoice-parser": {"temperature": 0.1},
            "summariser": {"temperature": 0.7},
        },
    )
    result = await workflow.run("Process the attached invoice.", client_kwargs=kwargs)
    outputs = result.get_outputs()
    return outputs[0].text if outputs else ""

asyncio.run(run_invoice_workflow(my_workflow))
```

### Example 2 — different language hints per executor

```python
from agent_framework import WorkflowInvocationKwargs

multi_lingual_kwargs = WorkflowInvocationKwargs(
    global_kwargs={"language_hint": "en"},
    executor_kwargs={
        "french-summariser": {"language_hint": "fr"},
        "spanish-extractor": {"language_hint": "es"},
    },
)

result = await workflow.run(input_text, function_invocation_kwargs=multi_lingual_kwargs)
```

---

## 8. `StepWrapper[R]`

**Module:** `agent_framework._workflows._functional` (re-exported via `agent_framework`)

> **Experimental:** requires `ExperimentalFeature.FUNCTIONAL_WORKFLOWS` opt-in.

`StepWrapper` is the object returned by the `@step` decorator in functional workflows. You never construct it directly — you obtain it by decorating an `async` function with `@step`. Inside a `@workflow` function, calls to a `StepWrapper` are intercepted to provide result caching, event emission, `RunContext` injection, and per-step checkpointing. Outside a workflow (e.g., in unit tests), the wrapper is transparent and calls the original function directly.

### Produced by `@step`

```python
from agent_framework._workflows._functional import step

@step
async def parse_invoice(raw: str) -> dict:
    ...  # your async logic

# parse_invoice is now a StepWrapper[dict]
```

The `@step` decorator also accepts keyword arguments:

```python
@step(name="invoice-parser", replay_key=lambda raw: hashlib.md5(raw.encode()).hexdigest())
async def parse_invoice(raw: str) -> dict:
    ...
```

| `@step` kwarg | Notes |
|---|---|
| `name` | Display name for events. Defaults to `func.__name__`. |
| `replay_key` | Callable `(*args, **kwargs) -> str` — provides a stable cache key for opaque or concurrent calls. |

### Constructor (internal — prefer `@step`)

```python
StepWrapper(
    func: Callable[..., Awaitable[R]],
    *,
    name: str | None = None,
    replay_key: Callable[..., str] | None = None,
)
```

### What `StepWrapper` provides inside a `@workflow`

| Feature | Description |
|---|---|
| **Caching** | Stable cache key derived from canonical args; replays emit `executor_bypassed` instead of re-running. |
| **Event emission** | `executor_invoked`, `executor_completed`, and `executor_failed` events for observability. |
| **`RunContext` injection** | If the step declares a parameter typed `RunContext` (or named `ctx`), the active context is auto-injected. |
| **Per-step checkpointing** | A checkpoint is saved after each live execution when checkpoint storage is configured. |

### Example 1 — basic functional workflow with two steps

```python
import asyncio
from agent_framework._workflows._functional import step, workflow, RunContext
from agent_framework.experimental import enable_experimental, ExperimentalFeature

enable_experimental(ExperimentalFeature.FUNCTIONAL_WORKFLOWS)

@step
async def fetch_data(url: str) -> str:
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
        return r.text

@step
async def summarise(text: str, *, ctx: RunContext | None = None) -> str:
    # RunContext is injected automatically; it exposes get_state/set_state/is_streaming.
    # The agent is captured from the enclosing scope — RunContext has no get_agent().
    response = await my_agent.run(f"Summarise:\n{text}")
    return response.text

@workflow
async def research_pipeline(url: str) -> str:
    raw = await fetch_data(url)
    return await summarise(raw)

async def main():
    result = await research_pipeline.build().run("https://example.com/article")
    # run() returns WorkflowRunResult; extract the pipeline's return value via get_outputs()
    print(result.get_outputs()[0])

asyncio.run(main())
```

### Example 2 — replay key for concurrent calls with identical arguments

```python
import asyncio
import hashlib
from agent_framework._workflows._functional import step, workflow
from agent_framework.experimental import enable_experimental, ExperimentalFeature

enable_experimental(ExperimentalFeature.FUNCTIONAL_WORKFLOWS)

@step(replay_key=lambda url, **_: hashlib.sha256(url.encode()).hexdigest())
async def fetch_page(url: str) -> str:
    import httpx
    async with httpx.AsyncClient() as client:
        return (await client.get(url)).text

@workflow
async def compare_pages(urls: tuple[str, str]) -> tuple[str, str]:
    # Both calls are concurrent; distinct replay keys prevent cache collision
    url_a, url_b = urls
    page_a, page_b = await asyncio.gather(fetch_page(url_a), fetch_page(url_b))
    return page_a, page_b

# FunctionalWorkflow.run() accepts one positional workflow input and returns WorkflowRunResult
async def main():
    result = await compare_pages.build().run(
        ("https://example.com/a", "https://example.com/b")
    )
    a, b = result.get_outputs()[0]  # get_outputs()[0] is the workflow's tuple return value

asyncio.run(main())
```

### Example 3 — unit-testing a step in isolation

```python
import asyncio
from agent_framework._workflows._functional import step

@step
async def double(n: int) -> int:
    return n * 2

# Outside a workflow, StepWrapper is transparent
async def test_double():
    result = await double(21)
    assert result == 42, f"Expected 42, got {result}"

asyncio.run(test_double())
```

---

## 9. `DelegatingSkillsSource`

**Module:** `agent_framework._skills` (re-exported via `agent_framework`)

`DelegatingSkillsSource` is an abstract base class for building composable skill-source decorators. It wraps an inner `SkillsSource` and delegates `get_skills()` to it. Subclass it to add cross-cutting concerns — custom filtering, logging, rate-limiting, or any transformation — without modifying the inner source.

> **Relation to built-in decorators:** `DeduplicatingSkillsSource`, `FilteringSkillsSource`, `AggregatingSkillsSource`, and `CachingSkillsSource` (Vols. 7–8) all extend `DelegatingSkillsSource`. Use this base when none of the built-ins fit your requirement.

### Constructor

```python
DelegatingSkillsSource(inner_source: SkillsSource)
```

### Properties and methods

| Member | Description |
|---|---|
| `source.inner_source` | The wrapped `SkillsSource`. |
| `async source.get_skills(context)` | Delegates to `inner_source.get_skills(context)`. Override to intercept. |

### Example 1 — logging decorator

```python
import logging
from agent_framework import DelegatingSkillsSource, SkillsSource, Skill, SkillsSourceContext

logger = logging.getLogger(__name__)

class LoggingSkillsSource(DelegatingSkillsSource):
    async def get_skills(self, context: SkillsSourceContext) -> list[Skill]:
        skills = await super().get_skills(context)
        session_id = context.session.session_id if context.session else None
        logger.info(
            "Loaded %d skills for session %s",
            len(skills),
            session_id,
        )
        return skills
```

### Example 2 — rate-limited decorator

```python
import asyncio
import time
from agent_framework import DelegatingSkillsSource, SkillsSource, Skill, SkillsSourceContext

class RateLimitedSkillsSource(DelegatingSkillsSource):
    def __init__(self, inner_source: SkillsSource, *, calls_per_minute: int = 10):
        super().__init__(inner_source)
        self._min_interval = 60 / calls_per_minute
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def get_skills(self, context: SkillsSourceContext) -> list[Skill]:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
        return await super().get_skills(context)
```

### Example 3 — chaining multiple decorators

```python
from agent_framework import (
    FileSkillsSource,
    DeduplicatingSkillsSource,
    FilteringSkillsSource,
    SkillsProvider,
)

base = FileSkillsSource("./skills")

# Compose: deduplicate → filter to "approved" category → log
source = LoggingSkillsSource(
    FilteringSkillsSource(
        DeduplicatingSkillsSource(base),
        predicate=lambda skill, _ctx: (skill.frontmatter.metadata or {}).get("category") == "approved",
    )
)

# Wrap in SkillsProvider and pass via context_providers
agent = Agent(client=my_client, context_providers=[SkillsProvider(source)])
```

---

## 10. `SubWorkflowRequestMessage` and `SubWorkflowResponseMessage`

**Module:** `agent_framework._workflows._workflow_executor` (re-exported via `agent_framework`)

These two dataclasses form the message-passing protocol between a `WorkflowExecutor` (sub-workflow node) and its parent workflow when the sub-workflow needs to pause and request information — the human-in-the-loop (HITL) scenario at the sub-workflow boundary.

When an executor inside a sub-workflow emits a `REQUEST_INFO` `WorkflowEvent`, the `WorkflowExecutor` node wraps it in a `SubWorkflowRequestMessage` and yields it upward. The parent workflow handler inspects the request, gathers the response, and hands back a `SubWorkflowResponseMessage` (via `request.create_response(data)`) to resume the sub-workflow.

### `SubWorkflowRequestMessage`

```python
@dataclass
class SubWorkflowRequestMessage:
    source_event: WorkflowEvent   # the REQUEST_INFO event from the sub-workflow
    executor_id: str              # ID of the WorkflowExecutor in the parent graph
```

#### Method

```python
request.create_response(data: Any) -> SubWorkflowResponseMessage
```

Validates that `data` matches the expected type declared in `source_event.response_type`, then returns the corresponding `SubWorkflowResponseMessage`. Raises `TypeError` on type mismatch.

### `SubWorkflowResponseMessage`

```python
@dataclass
class SubWorkflowResponseMessage:
    data: Any              # the response value, type-checked at creation
    source_event: WorkflowEvent  # the original REQUEST_INFO event
```

### Example 1 — HITL approval gate in a sub-workflow

```python
import asyncio
from agent_framework import (
    WorkflowEvent, SubWorkflowRequestMessage, SubWorkflowResponseMessage,
)

# --- HITL message-passing protocol demonstration ---
# In a real workflow, a WorkflowExecutor(inner_workflow, id="sub") wraps an inner
# Workflow whose Executor calls ctx.request_info(request_data, bool) from a
# @handler method. The outer workflow receives the resulting SubWorkflowRequestMessage
# and resumes by supplying a SubWorkflowResponseMessage via create_response().

def handle_approval_request(request: SubWorkflowRequestMessage) -> SubWorkflowResponseMessage:
    """Simulate routing a HITL request to a user and returning their decision."""
    # In production: send to UI and await user input
    user_approved = True  # simulated approval
    return request.create_response(user_approved)

async def demonstrate_hitl_protocol():
    # Build a mock request_info event (response_type=bool means caller expects a bool)
    event = WorkflowEvent(
        type="request_info",
        response_type=bool,
        request_id="approval-001",
    )
    request = SubWorkflowRequestMessage(
        source_event=event,
        executor_id="approval-subworkflow",
    )

    response = handle_approval_request(request)
    assert response.data is True
    assert isinstance(response, SubWorkflowResponseMessage)
    print("HITL protocol: approval granted, sub-workflow can continue")

asyncio.run(demonstrate_hitl_protocol())
```

### Example 2 — type-safe `create_response` validation

```python
from agent_framework import WorkflowEvent, SubWorkflowRequestMessage

# Simulated REQUEST_INFO event expecting a str response
mock_event = WorkflowEvent(
    type="request_info",
    response_type=str,
    request_id="req-001",
)
request = SubWorkflowRequestMessage(
    source_event=mock_event,
    executor_id="my-executor",
)

# Correct type — succeeds
response = request.create_response("Approved by ops team")
assert response.data == "Approved by ops team"

# Wrong type — raises TypeError
try:
    bad_response = request.create_response(42)  # expected str, got int
except TypeError as e:
    print(f"Caught: {e}")
    # Caught: Response data type <class 'int'> does not match expected type <class 'str'>
```

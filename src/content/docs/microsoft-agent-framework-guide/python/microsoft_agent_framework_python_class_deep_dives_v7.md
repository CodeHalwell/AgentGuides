---
title: "Microsoft Agent Framework (Python) — 10-API Deep Dives Vol. 7 (1.20.0)"
description: "Source-verified deep dives for AgentExecutorRequest, AgentExecutorResponse, AgentExecutorCheckpointState, AgentExecutor, WorkflowExecutor, WorkflowMessage, WorkflowCheckpoint, TodoSessionStore, ClassSkill, and AggregatingSkillsSource/CachingSkillsSource — all verified against agent-framework 1.20.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-API Deep Dives Vol. 7

**Verified against:** `agent-framework==1.20.0`
**Python requirement:** 3.10+

This volume covers the executor/checkpoint layer and composable skills pipeline: the three data classes that flow into and out of `AgentExecutor`, the executor itself, `WorkflowExecutor` for embedding a sub-workflow inside a parent, the `WorkflowMessage` envelope, the `WorkflowCheckpoint` snapshot, `TodoSessionStore` for session-resident todo state, `ClassSkill` for class-based skill packages, and the two source combinators `AggregatingSkillsSource` and `CachingSkillsSource`.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

See [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) for `FanInEdgeGroup`, `FanOutEdgeGroup`, `FunctionalWorkflow`, `FunctionalWorkflowAgent`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `MCPStdioTool`, `MCPStreamableHTTPTool`, `SelectiveToolCallCompactionStrategy`, and `TodoProvider`.

See [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) for `WorkflowBuilder`, `SlidingWindowStrategy`, `TruncationStrategy`, `ContextWindowCompactionStrategy`, `LocalEvaluator`, `InlineSkill`, `FileAccessProvider`, `MemoryContextProvider`, `FileHistoryProvider`, and `MCPWebsocketTool`.

See [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) for `VectorStoreField`, `VectorStoreCollectionDefinition`, `InMemoryCollection`, `InMemoryStore`, `Filter`, `FilterGroup`, `SecretString`, `load_settings`, `create_agent_hooks_middleware`, and `GroupChatBuilder`.

See [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) for `WorkflowEvent`, `AgentContext`, `MiddlewareBundle`, `ConversationSplit`/`ConversationSplitter`, `VectorStoreHistoryProvider`, `MemoryStore`/`MemoryFileStore`, `MemoryTopicRecord`, `WorkflowRunResult`, `FunctionInvocationContext`, and `ChatOptions`.

See [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) for `Agent`, `RawAgent`, `AgentSession`, `AgentResponse`, `FunctionTool`, `WorkflowAgent`, `WorkflowContext`, `CompactionProvider`, `VectorCollectionContextProvider`, and `TodoItem`/`TodoStore`/`TodoFileStore`.

---

## 1. `AgentExecutorRequest`

**Module:** `agent_framework._workflows._agent_executor` (re-exported via `agent_framework`)

`AgentExecutorRequest` is a lightweight dataclass that packages the canonical input to an `AgentExecutor`. It wraps a list of messages together with a `should_respond` flag so callers can pre-fill the executor cache with context without triggering an agent run. Passing `should_respond=False` is the standard way to prime an agent with system-level context before the first real user turn.

### Constructor

```python
@dataclass
class AgentExecutorRequest:
    messages: list[Message]
    should_respond: bool = True
```

### Fields

| Field | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | Messages appended to the executor's cache |
| `should_respond` | `bool` | `True` (default) — run the agent immediately; `False` — cache only |

### Example 1 — Prime context, then trigger a response

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutorRequest, WorkflowBuilder,
    AgentExecutor, WorkflowRunResult, Message,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

support_agent = Agent(client, instructions="You are a helpful support agent.")

async def main() -> None:
    support_exec = AgentExecutor(support_agent, id="support")

    wb = WorkflowBuilder(name="support-wf", start_executor=support_exec, output_from=[support_exec])
    workflow = wb.build()

    # Prime the executor with account context — agent does not reply yet
    prime = AgentExecutorRequest(
        messages=[Message("user", ["Account: acct-123"])],
        should_respond=False,
    )
    # Now the user's real question triggers the run
    question = AgentExecutorRequest(
        messages=[Message("user", ["What is my account balance?"])],
        should_respond=True,
    )

    result: WorkflowRunResult = await workflow.run(prime)
    result = await workflow.run(question)
    print(result.get_outputs())

asyncio.run(main())
```

### Example 2 — Batch-load multi-turn history

```python
from agent_framework import AgentExecutorRequest, Message

history: list[Message] = [
    Message("user", ["What is the capital of France?"]),
    Message("assistant", ["The capital of France is Paris."]),
]
# Replay history without re-running the agent
replay_request = AgentExecutorRequest(messages=history, should_respond=False)
# Then ask the follow-up
followup = AgentExecutorRequest(
    messages=[Message("user", ["Tell me more about its history."])],
    should_respond=True,
)
```

---

## 2. `AgentExecutorResponse`

**Module:** `agent_framework._workflows._agent_executor` (re-exported via `agent_framework`)

`AgentExecutorResponse` carries the output of an `AgentExecutor` run. It bundles the underlying `AgentResponse`, the executor's identifier, and the **full conversation** (all prior messages plus the new assistant messages). Downstream `AgentExecutor` instances receive this type and use the full conversation to maintain multi-turn context without losing earlier user prompts.

### Constructor

```python
@dataclass
class AgentExecutorResponse:
    executor_id: str
    agent_response: AgentResponse
    full_conversation: list[Message]
```

### Methods

| Method | Signature | Notes |
|---|---|---|
| `with_text` | `(text: str) -> AgentExecutorResponse` | Replace the assistant text while preserving `full_conversation` for downstream chains |

### `with_text()` detail

When a custom executor sits between two `AgentExecutor` nodes and transforms the text (e.g. translates it), emitting a bare `str` loses the full conversation because downstream sees only that string. `with_text()` returns a new `AgentExecutorResponse` whose `agent_response` contains the replacement text but whose `full_conversation` is the prior conversation plus the replacement message.

```python
AgentExecutorResponse.with_text(text: str) -> AgentExecutorResponse
```

### Example 1 — Inspect executor output in a custom handler

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutorResponse, WorkflowContext,
    WorkflowBuilder, AgentExecutor, executor, handler,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()
summariser = Agent(client, instructions="Summarise the following text in one sentence.")

@executor(id="log_response", input=AgentExecutorResponse, output=AgentExecutorResponse)
async def log_and_pass(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse, str],
) -> None:
    print(f"[{response.executor_id}] {response.agent_response.text[:80]!r}")
    # Re-emit unchanged so the next executor receives AgentExecutorResponse
    await ctx.send_message(response)

summariser_exec = AgentExecutor(summariser, id="summariser")
wb = WorkflowBuilder(name="log-wf", start_executor=summariser_exec, output_from=[log_and_pass])
wb.add_edge(summariser_exec, log_and_pass)
workflow = wb.build()
```

### Example 2 — Preserve context via `with_text()`

```python
from agent_framework import AgentExecutorResponse, WorkflowContext, executor

@executor(id="upper", input=AgentExecutorResponse, output=AgentExecutorResponse, workflow_output=str)
async def upper_case(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse, str],
) -> None:
    # CORRECT: with_text preserves full_conversation for downstream AgentExecutors
    await ctx.send_message(response.with_text(response.agent_response.text.upper()))
    await ctx.yield_output(response.agent_response.text.upper())
```

---

## 3. `AgentExecutorCheckpointState`

**Module:** `agent_framework._workflows._agent_executor` (re-exported via `agent_framework`)

`AgentExecutorCheckpointState` is a `TypedDict` (all keys `total=False`) that describes the schema `AgentExecutor` persists in `WorkflowCheckpoint.state["_executor_state"][executor_id]`. Understanding this layout is essential when writing custom executors that inherit from `AgentExecutor` or when inspecting restored checkpoints.

### Schema

```python
class AgentExecutorCheckpointState(TypedDict, total=False):
    cache: list[Message]
    full_conversation: list[Message]
    agent_session: AgentSessionDict
    pending_agent_requests: dict[str, Content]
    pending_responses_to_agent: list[Content]
    pending_request_order: list[str]
```

### Field reference

| Key | Type | Description |
|---|---|---|
| `cache` | `list[Message]` | Messages buffered since the last agent run |
| `full_conversation` | `list[Message]` | All prior inputs + all assistant/tool outputs |
| `agent_session` | `AgentSessionDict` | Serialised session payload (service session id, history) |
| `pending_agent_requests` | `dict[str, Content]` | In-flight user-input requests keyed by request id |
| `pending_responses_to_agent` | `list[Content]` | Queued responses not yet delivered to the agent |
| `pending_request_order` | `list[str]` | Original request IDs for the batch, preserving order |

> All keys are optional (`total=False`). Missing keys reset to empty defaults during restore, enabling backward-compatible checkpoint additions.

### Example 1 — Inspect an executor checkpoint after pause

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder,
    InMemoryCheckpointStorage, WorkflowRunResult,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()
agent = Agent(client, instructions="You are helpful.")

storage = InMemoryCheckpointStorage()

helper_exec = AgentExecutor(agent, id="helper")
wb = WorkflowBuilder(
    name="ckpt-demo",
    start_executor=helper_exec,
    output_from=[helper_exec],
    checkpoint_storage=storage,
)
workflow = wb.build()

async def main() -> None:
    result: WorkflowRunResult = await workflow.run("Hello")
    checkpoint = await storage.get_latest(workflow_name=workflow.name)
    if checkpoint:
        executor_states = checkpoint.state.get("_executor_state", {})
        helper_state = executor_states.get("helper", {})
        print("Cache length:", len(helper_state.get("cache", [])))
        print("Full convo:", [m.role for m in helper_state.get("full_conversation", [])])

asyncio.run(main())
```

### Example 2 — Custom executor extending AgentExecutor with extra state

```python
from typing import Any
from agent_framework import AgentExecutorCheckpointState, AgentSession
from agent_framework._workflows._agent_executor import AgentExecutor  # type: ignore[reportPrivateUsage]

class TaggedAgentExecutor(AgentExecutor):
    """AgentExecutor that also checkpoints a custom tag."""

    def __init__(self, agent, *, tag: str, **kwargs):
        super().__init__(agent, **kwargs)
        self._tag = tag

    async def on_checkpoint_save(self) -> dict[str, Any]:
        state = await super().on_checkpoint_save()
        state["tag"] = self._tag
        return state

    async def on_checkpoint_restore(self, state: dict) -> None:
        await super().on_checkpoint_restore(state)
        self._tag = state.get("tag", self._tag)
```

---

## 4. `AgentExecutor`

**Module:** `agent_framework._workflows._agent_executor` (re-exported via `agent_framework`)

`AgentExecutor` is the built-in `Executor` subclass that wraps any `SupportsAgentRun` (e.g. `Agent`, `RawAgent`) and routes incoming `WorkflowMessage` data to the appropriate input handler via `@handler` dispatch. It manages an internal message cache between runs, tracks pending user-input requests (approvals, computer calls), and handles three conversation-context modes.

### Constructor

```python
AgentExecutor(
    agent: SupportsAgentRun,
    *,
    session: AgentSession | None = None,
    id: str | None = None,
    context_mode: Literal["full", "last_agent", "custom"] | None = None,
    context_filter: Callable[[list[Message]], list[Message]] | None = None,
)
```

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent` | `SupportsAgentRun` | — | The agent to wrap (typically `Agent`) |
| `session` | `AgentSession \| None` | `None` | Explicit session; otherwise a fresh one is created |
| `id` | `str \| None` | `None` | Unique executor ID; falls back to `agent.name` |
| `context_mode` | `"full"` \| `"last_agent"` \| `"custom"` | `"full"` | How prior context is assembled when chaining `AgentExecutorResponse` inputs |
| `context_filter` | `Callable` \| `None` | `None` | Required when `context_mode="custom"` |

### `context_mode` options

| Mode | Behaviour |
|---|---|
| `"full"` | Appends `prior.full_conversation` (all history + last response) to cache |
| `"last_agent"` | Appends only `prior.agent_response.messages` (last turn only) |
| `"custom"` | Applies `context_filter(full_conversation)` — you decide what to pass |

### Input handlers (`@handler`)

| Handler | Accepted type | Notes |
|---|---|---|
| `run` | `AgentExecutorRequest` | Canonical path; respects `should_respond` flag |
| `from_response` | `AgentExecutorResponse` | Chains from upstream `AgentExecutor`; respects `context_mode` |
| `from_str` | `str` | Plain prompt string; logs warning if cache is empty |
| `from_message` | `Message` | Single `Message` instance |
| `from_messages` | `list[str \| Message]` | Multi-message batch |
| `handle_user_input_response` | `(Content, Content)` | User-input approval / computer-call result |

### Properties

| Property | Type | Description |
|---|---|---|
| `agent` | `SupportsAgentRun` | The wrapped agent |
| `description` | `str \| None` | Delegates to `agent.description` |

### Checkpoint hooks

| Hook | Description |
|---|---|
| `on_checkpoint_save()` | Returns `dict[str, Any]` containing `AgentExecutorCheckpointState` fields; override via `state = await super().on_checkpoint_save(); state["key"] = val; return state` |
| `on_checkpoint_restore(state)` | Reads and validates `AgentExecutorCheckpointState` from `state` |

### Example 1 — Basic two-agent chain

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder, WorkflowRunResult,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

researcher = Agent(client, name="researcher", instructions="Research the topic thoroughly.")
writer = Agent(client, name="writer", instructions="Write a concise report.")

async def main() -> None:
    researcher_exec = AgentExecutor(researcher)
    writer_exec = AgentExecutor(writer, context_mode="last_agent")

    wb = WorkflowBuilder(name="research-write", start_executor=researcher_exec, output_from=[writer_exec])
    wb.add_edge(researcher_exec, writer_exec)
    workflow = wb.build()

    result: WorkflowRunResult = await workflow.run("Write a report on quantum computing.")
    print(result.get_outputs()[-1])

asyncio.run(main())
```

### Example 2 — Custom context filter

```python
from agent_framework import Agent, AgentExecutor, Message

def last_two_turns(conversation: list[Message]) -> list[Message]:
    """Keep only the last two messages to control token cost."""
    return conversation[-2:] if len(conversation) > 2 else conversation

agent = Agent(client, name="concise")
executor = AgentExecutor(
    agent,
    context_mode="custom",
    context_filter=last_two_turns,
)
```

### Example 3 — Streaming output from `AgentExecutor`

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, AgentResponseUpdate, WorkflowBuilder,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()
agent = Agent(client, name="streamer")

streamer_exec = AgentExecutor(agent)
wb = WorkflowBuilder(name="stream-wf", start_executor=streamer_exec, output_from=[streamer_exec])
workflow = wb.build()

async def main() -> None:
    stream = await workflow.run("Tell me a joke.", stream=True)
    async for event in stream:
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            print(event.data.text, end="", flush=True)
    print()

asyncio.run(main())
```

### Example 4 — Human-in-the-loop tool approval

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder, WorkflowRunResult,
    ToolApprovalMiddleware, WorkflowRunState,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

# ToolApprovalMiddleware with no auto-approval rules requires manual approval for all tools
approval_mw = ToolApprovalMiddleware(source_id="db")
db_agent = Agent(client, name="db", tools=[...], middleware=[approval_mw])

db_exec = AgentExecutor(db_agent)
wb = WorkflowBuilder(name="db-wf", start_executor=db_exec, output_from=[db_exec])
workflow = wb.build()

async def main() -> None:
    result: WorkflowRunResult = await workflow.run("Show me all orders from last week.")

    # Approve pending tool calls
    while result.get_final_state() == WorkflowRunState.IDLE_WITH_PENDING_REQUESTS:
        events = result.get_request_info_events()
        responses = {e.request_id: "APPROVED" for e in events}
        result = await workflow.run(responses=responses)

    print(result.get_outputs())

asyncio.run(main())
```

---

## 5. `WorkflowExecutor`

**Module:** `agent_framework._workflows._workflow_executor` (re-exported via `agent_framework`)

`WorkflowExecutor` wraps an entire `Workflow` as a single `Executor` node within a parent workflow. This enables **hierarchical workflow composition**: the sub-workflow runs to completion (or pauses for external input), its outputs are forwarded into the parent as messages, and any sub-workflow request/response events are proxied through the parent's executor graph.

### Constructor

```python
WorkflowExecutor(
    workflow: Workflow,
    id: str,
    allow_direct_output: bool = False,
    propagate_request: bool = False,
    **kwargs: Any,
)
```

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `workflow` | `Workflow` | — | The sub-workflow to embed. Must be a unique instance — do not share across `WorkflowExecutor` instances |
| `id` | `str` | — | Unique identifier for this executor in the parent |
| `allow_direct_output` | `bool` | `False` | `True` — sub-workflow outputs are yielded directly to the parent's event stream; `False` — outputs become `send_message` calls to downstream executors |
| `propagate_request` | `bool` | `False` | `True` — sub-workflow `request_info` events surface unchanged in the parent; `False` — wrapped in `SubWorkflowRequestMessage` |

### Input / output type inheritance

- **Input types:** the wrapped workflow's `input_types` plus `SubWorkflowResponseMessage`
- **Output types:** the wrapped workflow's `output_types` plus `SubWorkflowRequestMessage` (when any sub-workflow executor is request-response capable)

### Properties

| Property | Type | Description |
|---|---|---|
| `workflow` | `Workflow` | The embedded sub-workflow instance |
| `input_types` | `list[type]` | Derived from sub-workflow |
| `output_types` | `list[type]` | Derived from sub-workflow + `SubWorkflowRequestMessage` |

### Example 1 — Embed a summariser sub-workflow inside a pipeline

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder, WorkflowExecutor, WorkflowRunResult,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

# ---- sub-workflow: summarise ----
summariser = Agent(client, name="summariser", instructions="Summarise the input text.")
summariser_exec = AgentExecutor(summariser)
sub_wb = WorkflowBuilder(name="summarise-wf", start_executor=summariser_exec, output_from=[summariser_exec])
sub_workflow = sub_wb.build()

# ---- parent workflow: fetch → summarise → format ----
formatter = Agent(client, name="formatter", instructions="Format the summary as bullet points.")

sub_exec = WorkflowExecutor(sub_workflow, id="summarise", allow_direct_output=False)
formatter_exec = AgentExecutor(formatter)
parent_wb = WorkflowBuilder(name="pipeline-wf", start_executor=sub_exec, output_from=[formatter_exec])
parent_wb.add_edge(sub_exec, formatter_exec)
parent_workflow = parent_wb.build()

async def main() -> None:
    long_text = "The quick brown fox..." * 20
    result: WorkflowRunResult = await parent_workflow.run(long_text)
    print(result.get_outputs()[-1])

asyncio.run(main())
```

### Example 2 — Direct output passthrough

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder, WorkflowExecutor,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

inner_agent = Agent(client, name="inner")
inner_exec = AgentExecutor(inner_agent)
inner_wb = WorkflowBuilder(name="inner-wf", start_executor=inner_exec, output_from=[inner_exec])
inner_wf = inner_wb.build()

# allow_direct_output=True: sub-workflow outputs become parent workflow outputs
wf_exec = WorkflowExecutor(inner_wf, id="inner_exec", allow_direct_output=True)
outer_wb = WorkflowBuilder(name="outer-wf", start_executor=wf_exec, output_from=[wf_exec])
outer_wf = outer_wb.build()

async def main() -> None:
    result = await outer_wf.run("What is 2 + 2?")
    print(result.get_outputs())

asyncio.run(main())
```

---

## 6. `WorkflowMessage`

**Module:** `agent_framework._workflows._runner_context` (re-exported via `agent_framework`)

`WorkflowMessage` is the **internal envelope** that carries data between executors inside a running workflow. Application code rarely constructs `WorkflowMessage` directly — the framework creates and routes them — but understanding the type is important when writing custom executors, debugging graph state, or interpreting checkpoint payloads.

### Constructor

```python
@dataclass
class WorkflowMessage:
    data: Any
    source_id: str
    target_id: str | None = None
    type: MessageType = MessageType.STANDARD
    trace_contexts: list[dict[str, str]] | None = None
    source_span_ids: list[str] | None = None
    original_request_info_event: WorkflowEvent[Any] | None = None
```

### Fields

| Field | Type | Description |
|---|---|---|
| `data` | `Any` | The payload dispatched to an executor's `@handler` |
| `source_id` | `str` | Executor ID that emitted this message |
| `target_id` | `str \| None` | Explicit routing target; `None` means broadcast-to-capable |
| `type` | `MessageType` | `STANDARD` or `RESPONSE` |
| `trace_contexts` | `list[dict]` | W3C Trace Context headers from all contributing sources (fan-in) |
| `source_span_ids` | `list[str]` | OpenTelemetry span IDs for linking |
| `original_request_info_event` | `WorkflowEvent \| None` | Non-`None` when this is a response to a `request_info` event |

### Backward-compatible properties

| Property | Returns | Description |
|---|---|---|
| `trace_context` | `dict[str, str] \| None` | First entry of `trace_contexts` |
| `source_span_id` | `str \| None` | First entry of `source_span_ids` |

### Methods

| Method | Returns | Description |
|---|---|---|
| `to_dict()` | `dict[str, Any]` | Shallow serialization |
| `WorkflowMessage.from_dict(data)` | `WorkflowMessage` | Deserialize from dict; raises `KeyError` on missing `data`/`source_id` |

### Example 1 — Inspect incoming messages in a debug executor

```python
from agent_framework import WorkflowContext, WorkflowMessage, executor, handler

@executor(id="debug_sink", input=object, workflow_output=str)
async def debug_sink(
    message: object,
    ctx: WorkflowContext[str, str],
) -> None:
    # The framework passes the unwrapped `data`; reconstruct for inspection
    print(f"Received type={type(message).__name__!r}")
    if isinstance(message, str):
        await ctx.yield_output(message)
```

### Example 2 — Build a WorkflowMessage manually for testing

```python
from agent_framework import WorkflowMessage

msg = WorkflowMessage(
    data="Hello from test",
    source_id="mock_executor",
    target_id="consumer_executor",
)
print(msg.to_dict())
# {'data': 'Hello from test', 'source_id': 'mock_executor',
#  'target_id': 'consumer_executor', 'type': 'standard', ...}

restored = WorkflowMessage.from_dict(msg.to_dict())
assert restored.source_id == "mock_executor"
```

### Example 3 — Fan-in trace context

```python
from agent_framework import WorkflowMessage

# Fan-in executor receives a single WorkflowMessage aggregating multiple sources
fan_in_msg = WorkflowMessage(
    data=["result-a", "result-b"],
    source_id="fan_in",
    trace_contexts=[
        {"traceparent": "00-trace-a-00"},
        {"traceparent": "00-trace-b-00"},
    ],
    source_span_ids=["span-a", "span-b"],
)
# Backward-compatible single-source access:
print(fan_in_msg.trace_context)   # {'traceparent': '00-trace-a-00'}
print(fan_in_msg.source_span_id)  # 'span-a'
```

---

## 7. `WorkflowCheckpoint`

**Module:** `agent_framework._workflows._checkpoint` (re-exported via `agent_framework`)

`WorkflowCheckpoint` is a mutable dataclass snapshot of a complete workflow execution state at a superstep boundary. It is the unit of persistence stored by `CheckpointStorage` implementations and the structure exchanged between workflow runs to support pause/resume and fault-tolerance patterns.

### Constructor (dataclass)

```python
@dataclass
class WorkflowCheckpoint:
    workflow_name: str
    graph_signature_hash: str
    checkpoint_id: CheckpointID = field(default_factory=lambda: str(uuid.uuid4()))
    previous_checkpoint_id: CheckpointID | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    messages: dict[str, list[WorkflowMessage]] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    pending_request_info_events: dict[str, WorkflowEvent[Any]] = field(default_factory=dict)
    iteration_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
```

### Fields

| Field | Type | Description |
|---|---|---|
| `workflow_name` | `str` | Logical grouping name; compatible checkpoints share a name |
| `graph_signature_hash` | `str` | Hash of the graph topology; restore validates this to detect incompatible changes |
| `checkpoint_id` | `CheckpointID` (str) | UUID generated automatically |
| `previous_checkpoint_id` | `CheckpointID \| None` | Forms the lineage chain |
| `timestamp` | `str` | ISO 8601 UTC creation time |
| `messages` | `dict[str, list[WorkflowMessage]]` | Per-executor buffered messages |
| `state` | `dict[str, Any]` | Committed state including `_executor_state` and `_edge_state` |
| `pending_request_info_events` | `dict[str, WorkflowEvent]` | Unresolved `request_info` events |
| `iteration_count` | `int` | Superstep boundary number (not unique — see note below) |
| `metadata` | `dict[str, Any]` | Free-form metadata |
| `version` | `str` | Checkpoint format version (`"1.0"`) |

> **`iteration_count` is not a unique identifier.** Two checkpoints at the same superstep boundary can share the same `iteration_count` (e.g. a "pause" checkpoint and a "response-entry" checkpoint before the next superstep). Use `checkpoint_id` / `previous_checkpoint_id` for ordering.

### Methods

| Method | Signature | Description |
|---|---|---|
| `to_dict()` | `() -> dict[str, Any]` | Shallow serialization of all fields |
| `from_dict(data)` | `(dict) -> WorkflowCheckpoint` | Deserialize; raises on missing required fields |

### State dict reserved keys

| Key | Owner | Description |
|---|---|---|
| `_executor_state` | Framework | `AgentExecutorCheckpointState` for each executor, keyed by `executor_id` |
| `_edge_state` | Framework | Partial fan-in buffer state managed by edge runners |

### Example 1 — Walk a checkpoint lineage chain

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder,
    InMemoryCheckpointStorage, WorkflowRunResult,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()
agent = Agent(client, name="a")
storage = InMemoryCheckpointStorage()

a_exec = AgentExecutor(agent)
wb = WorkflowBuilder(
    name="ckpt-chain",
    start_executor=a_exec,
    output_from=[a_exec],
    checkpoint_storage=storage,
)
workflow = wb.build()

async def main() -> None:
    await workflow.run("Step 1")
    await workflow.run("Step 2")

    latest = await storage.get_latest(workflow_name=workflow.name)
    assert latest is not None
    print(f"Latest: {latest.checkpoint_id}  iteration={latest.iteration_count}")
    print(f"Parent: {latest.previous_checkpoint_id}")
    print(f"State keys: {list(latest.state.keys())}")

asyncio.run(main())
```

### Example 2 — Verify graph compatibility before restore

```python
import asyncio
from agent_framework import WorkflowCheckpoint, InMemoryCheckpointStorage

async def safe_restore(
    storage: InMemoryCheckpointStorage,
    workflow_name: str,
    expected_hash: str,
) -> WorkflowCheckpoint | None:
    checkpoint = await storage.get_latest(workflow_name=workflow_name)
    if checkpoint is None:
        return None
    if checkpoint.graph_signature_hash != expected_hash:
        raise ValueError(
            f"Graph changed: stored hash {checkpoint.graph_signature_hash!r} "
            f"!= current {expected_hash!r}"
        )
    return checkpoint
```

### Example 3 — Read executor state from a checkpoint

```python
from agent_framework import WorkflowCheckpoint

def inspect_executor_state(checkpoint: WorkflowCheckpoint, executor_id: str) -> None:
    executor_state = checkpoint.state.get("_executor_state", {}).get(executor_id, {})
    cache = executor_state.get("cache", [])
    full_conv = executor_state.get("full_conversation", [])
    print(f"Executor '{executor_id}': cache={len(cache)}, history={len(full_conv)}")
    pending = executor_state.get("pending_agent_requests", {})
    if pending:
        print(f"  Pending requests: {list(pending.keys())}")
```

---

## 8. `TodoSessionStore`

**Module:** `agent_framework._harness._todo` (re-exported via `agent_framework`)

`TodoSessionStore` is the session-resident implementation of `TodoStore`. It persists todo items inside `AgentSession.state` under the source-id key instead of writing to the filesystem, making it the right choice for ephemeral or in-memory workflows that do not need todo items to survive process restarts.

Contrast with `TodoFileStore` (covered in Vol. 6): `TodoSessionStore` survives as long as the `AgentSession` lives; `TodoFileStore` survives across sessions using a per-session JSON file on disk.

### Constructor

`TodoSessionStore` has no constructor parameters — instantiate it directly:

```python
store = TodoSessionStore()
```

### Methods (all async)

| Method | Signature | Returns | Description |
|---|---|---|---|
| `load_state` | `(session, *, source_id) -> tuple[list[TodoItem], int]` | Items + next ID | Reads from `session.state[source_id]`, creating the key if absent |
| `save_state` | `(session, items, *, next_id, source_id) -> None` | — | Writes back to `session.state[source_id]` |
| `load_items` | `(session, *, source_id) -> list[TodoItem]` | Items only | Convenience wrapper around `load_state` |

### Use with `TodoProvider`

`TodoSessionStore` is passed to `TodoProvider` via the `store=` keyword argument (added in 1.19.0+). When omitted, `TodoProvider` defaults to `TodoSessionStore`.

```python
TodoProvider(store=TodoSessionStore())
```

### Example 1 — Ephemeral todos inside a session-scoped agent

```python
import asyncio
from agent_framework import Agent, AgentSession, TodoProvider, TodoSessionStore
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

store = TodoSessionStore()

agent = Agent(
    client,
    instructions="You are a task manager.",
    context_providers=[
        TodoProvider(store=store),
    ],
)

async def main() -> None:
    session = agent.create_session()
    resp = await agent.run(
        "Plan my morning: exercise, breakfast, emails.",
        session=session,
    )
    print(resp.text)
    # Items live in session.state — no files written
    items = await store.load_items(session, source_id="todo")
    print([i.title for i in items])

asyncio.run(main())
```

### Example 2 — Manually read and modify session todos

```python
import asyncio
from agent_framework import AgentSession, TodoItem, TodoSessionStore

store = TodoSessionStore()

async def add_todo(session: AgentSession, title: str) -> TodoItem:
    items, next_id = await store.load_state(session, source_id="todo")
    new_item = TodoItem(id=next_id, title=title)
    items.append(new_item)
    await store.save_state(session, items, next_id=next_id + 1, source_id="todo")
    return new_item

async def main() -> None:
    from agent_framework import AgentSession
    session = AgentSession(session_id="test-session")
    item = await add_todo(session, "Write unit tests")
    print(item)  # TodoItem(id=1, title='Write unit tests', ...)

asyncio.run(main())
```

### Example 3 — Session-scoped todos in a workflow context

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder,
    TodoProvider, TodoSessionStore, AgentSession,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()
store = TodoSessionStore()

task_agent = Agent(
    client,
    name="task_agent",
    context_providers=[TodoProvider(store=store)],
)

task_exec = AgentExecutor(task_agent, session=AgentSession(session_id="wf-session"))
wb = WorkflowBuilder(name="todo-wf", start_executor=task_exec, output_from=[task_exec])
workflow = wb.build()

async def main() -> None:
    result = await workflow.run("Add buy groceries to my todo list.")
    print(result.get_outputs()[-1])

asyncio.run(main())
```

---

## 9. `ClassSkill`

**Module:** `agent_framework._skills` (re-exported via `agent_framework`)

`ClassSkill` is the abstract base class for **class-based reusable skills**. Subclass it to create self-contained skill packages — complete with instructions, supplementary resources, and executable scripts — that can be distributed via PyPI or shared across projects. Class-based skills are discoverable by `SkillsProvider` and honour the same progressive-disclosure protocol as `InlineSkill` and `FileSkill`.

### Constructor

```python
ClassSkill.__init__(
    self,
    *,
    frontmatter: SkillFrontmatter,
    argument_parser: SkillScriptArgumentParser | None = None,
)
```

| Parameter | Type | Description |
|---|---|---|
| `frontmatter` | `SkillFrontmatter` | Name, description, and optional metadata |
| `argument_parser` | `SkillScriptArgumentParser \| None` | Normalizes raw LLM tool arguments before scripts run; useful for backends that encode args as a JSON string |

### Abstract property you must implement

```python
@property
@abstractmethod
def instructions(self) -> str: ...
```

### Decorator-based API (recommended)

| Decorator | Purpose |
|---|---|
| `@ClassSkill.resource(name="…")` | Marks a method as a readable resource; method returns `str` |
| `@ClassSkill.script(name="…")` | Marks a method as an executable script; method returns `str` |

### Explicit override API (alternative)

Override the `resources` and `scripts` properties to return lists of `InlineSkillResource` / `InlineSkillScript` instances.

### Inherited discovery properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | From `frontmatter.name` |
| `description` | `str \| None` | From `frontmatter.description` |
| `resources` | `list[SkillResource]` | Auto-collected from `@resource` decorators |
| `scripts` | `list[SkillScript]` | Auto-collected from `@script` decorators |

### Example 1 — A unit-converter skill with a resource and a script

```python
import json
from agent_framework import Agent, ClassSkill, SkillFrontmatter, SkillsProvider
from agent_framework_openai import AzureOpenAIChatClient

CONVERSION_TABLE = {
    "km_to_miles": 0.621371,
    "kg_to_lbs": 2.20462,
    "c_to_f_offset": 32.0,
}


class UnitConverterSkill(ClassSkill):
    def __init__(self) -> None:
        super().__init__(
            frontmatter=SkillFrontmatter(
                name="unit-converter",
                description="Convert between common measurement units.",
            ),
        )

    @property
    def instructions(self) -> str:
        return (
            "# Unit Converter\n\n"
            "Use the `read_skill_resource` tool with resource `table` to see available conversions.\n"
            "Use the `unit-converter/convert` script to convert a value."
        )

    @ClassSkill.resource(name="table")
    def conversion_table(self) -> str:
        rows = "\n".join(f"| {k} | {v} |" for k, v in CONVERSION_TABLE.items())
        return f"| Conversion | Factor |\n|---|---|\n{rows}"

    @ClassSkill.script(name="convert")
    def convert(self, value: float, conversion: str) -> str:
        factor = CONVERSION_TABLE.get(conversion)
        if factor is None:
            return json.dumps({"error": f"Unknown conversion: {conversion!r}"})
        return json.dumps({"result": round(value * factor, 4)})


client = AzureOpenAIChatClient.from_env()
agent = Agent(
    client,
    context_providers=[
        SkillsProvider(source=[UnitConverterSkill()]),
    ],
)
```

### Example 2 — A reusable date-formatting skill package

```python
from datetime import datetime
from agent_framework import ClassSkill, SkillFrontmatter
import json


class DateFormatterSkill(ClassSkill):
    """Distributable skill that formats dates for the agent."""

    def __init__(self, default_format: str = "%Y-%m-%d") -> None:
        super().__init__(
            frontmatter=SkillFrontmatter(
                name="date-formatter",
                description="Format and parse dates.",
            ),
        )
        self._default_format = default_format

    @property
    def instructions(self) -> str:
        return (
            "# Date Formatter\n\n"
            f"Default output format: `{self._default_format}`.\n"
            "Use the `format_date` script to reformat a date string."
        )

    @ClassSkill.script(name="format_date")
    def format_date(self, date_str: str, fmt: str | None = None) -> str:
        try:
            dt = datetime.fromisoformat(date_str)
            out_fmt = fmt or self._default_format
            return json.dumps({"formatted": dt.strftime(out_fmt)})
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
```

### Example 3 — Use explicit `resources` / `scripts` overrides

```python
from agent_framework import ClassSkill, InlineSkillResource, InlineSkillScript, SkillFrontmatter
import json


def _hello_script(name: str = "World") -> str:
    return json.dumps({"greeting": f"Hello, {name}!"})


class HelloSkill(ClassSkill):
    def __init__(self) -> None:
        super().__init__(
            frontmatter=SkillFrontmatter(name="hello", description="Say hello."),
        )

    @property
    def instructions(self) -> str:
        return "# Hello\nUse the `greet` script to say hello."

    @property
    def resources(self) -> list:
        return [
            InlineSkillResource(name="readme", content="This skill says hello."),
        ]

    @property
    def scripts(self) -> list:
        return [
            InlineSkillScript(name="greet", function=_hello_script),
        ]
```

---

## 10. `AggregatingSkillsSource` and `CachingSkillsSource`

**Module:** `agent_framework._skills` (both re-exported via `agent_framework`)

These two classes form the **composable skill-source pipeline**: `AggregatingSkillsSource` merges multiple skill sources into one; `CachingSkillsSource` wraps any source and caches the skill list it returns, optionally per-agent and with a configurable refresh interval.

---

### 10a. `AggregatingSkillsSource`

Merges skill lists from multiple `SkillsSource` implementations into a single list. Sources are queried sequentially and their results concatenated in order.

#### Constructor

```python
AggregatingSkillsSource(sources: Sequence[SkillsSource])
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `get_skills` | `async (context: SkillsSourceContext) -> list[Skill]` | Calls each inner source and concatenates the results |

#### Example — Combine file-based and inline skills

```python
import asyncio
from agent_framework import (
    Agent, AggregatingSkillsSource, InlineSkill,
    SkillFrontmatter, SkillsProvider,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

# Inline skills
code_skill = InlineSkill(
    frontmatter=SkillFrontmatter(name="python-expert", description="Expert Python tips."),
    instructions="# Python Expert\nProvide idiomatic Python advice.",
)
docs_skill = InlineSkill(
    frontmatter=SkillFrontmatter(name="docs-writer", description="Technical writing guidance."),
    instructions="# Docs Writer\nWrite clear, concise documentation.",
)

# Aggregate multiple sources — could include FileSkillsSource, MCPSkillsSource, etc.
from agent_framework._skills import InMemorySkillsSource  # type: ignore[reportPrivateUsage]

source_a = InMemorySkillsSource([code_skill])
source_b = InMemorySkillsSource([docs_skill])
combined_source = AggregatingSkillsSource([source_a, source_b])

agent = Agent(
    client,
    context_providers=[SkillsProvider(source=combined_source)],
)
```

---

### 10b. `CachingSkillsSource`

Wraps any `SkillsSource` and caches its `get_skills()` result so expensive sources (filesystem scans, MCP network calls) are queried at most once — or once per `refresh_interval`.

#### Constructor

```python
CachingSkillsSource(
    inner_source: SkillsSource,
    *,
    cache_isolation_key_selector: Callable[[SkillsSourceContext], str | None] | None = None,
    refresh_interval: timedelta | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `inner_source` | `SkillsSource` | — | The wrapped source to cache |
| `cache_isolation_key_selector` | `Callable \| None` | `None` | Returns a per-agent/per-tenant key; `None` uses a shared bucket |
| `refresh_interval` | `timedelta \| None` | `None` | `None` — never expire; `timedelta(0)` — always expire; positive — TTL per key |

#### Concurrency behaviour

Concurrent callers share a single in-flight fetch per cache key — the inner source is queried **at most once** even under concurrent access. A failed fetch leaves the cache empty so the next call retries. A failed refresh keeps the previous cached list.

#### Methods

| Method | Signature | Description |
|---|---|---|
| `get_skills` | `async (context) -> list[Skill]` | Returns cached list or queries inner source |
| `inner_source` | property | The wrapped source |

#### Example 1 — Cache a slow MCP source forever

```python
from datetime import timedelta
from agent_framework import CachingSkillsSource

# Assume mcp_source is an MCPSkillsSource over a slow network
cached_mcp = CachingSkillsSource(mcp_source)
# After the first call, further calls are instant
```

#### Example 2 — Per-agent cache with 5-minute TTL

```python
from datetime import timedelta
from agent_framework import CachingSkillsSource, AggregatingSkillsSource, SkillsProvider, Agent

cached = CachingSkillsSource(
    expensive_file_source,
    cache_isolation_key_selector=lambda ctx: ctx.agent.name if ctx.agent else None,
    refresh_interval=timedelta(minutes=5),
)

agent_a = Agent(client, name="agent_a", context_providers=[SkillsProvider(source=cached)])
agent_b = Agent(client, name="agent_b", context_providers=[SkillsProvider(source=cached)])
# agent_a and agent_b get separate cache buckets ("agent_a" vs "agent_b")
# Each bucket expires independently after 5 minutes
```

#### Example 3 — Full pipeline: aggregate → cache → agent

```python
from datetime import timedelta
from agent_framework import (
    Agent, AggregatingSkillsSource, CachingSkillsSource,
    InlineSkill, SkillFrontmatter, SkillsProvider,
)
from agent_framework_openai import AzureOpenAIChatClient

client = AzureOpenAIChatClient.from_env()

skill_a = InlineSkill(
    frontmatter=SkillFrontmatter(name="skill-a", description="Skill A."),
    instructions="# Skill A",
)
skill_b = InlineSkill(
    frontmatter=SkillFrontmatter(name="skill-b", description="Skill B."),
    instructions="# Skill B",
)

from agent_framework._skills import InMemorySkillsSource  # type: ignore[reportPrivateUsage]

combined = AggregatingSkillsSource([
    InMemorySkillsSource([skill_a]),
    InMemorySkillsSource([skill_b]),
])

# Cache the combined source, per agent, refreshing every 10 minutes
cached_combined = CachingSkillsSource(
    combined,
    cache_isolation_key_selector=lambda ctx: ctx.agent.name if ctx.agent else None,
    refresh_interval=timedelta(minutes=10),
)

agent = Agent(
    client,
    context_providers=[SkillsProvider(source=cached_combined)],
)
```

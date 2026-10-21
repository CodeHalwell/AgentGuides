---
title: "Microsoft Agent Framework (Python) — 10-API Deep Dives Vol. 8 (1.21.0)"
description: "Source-verified deep dives for InProcRunnerContext, FunctionExecutor, EvalItem, EvalResults/EvalItemResult, ChatResponse/ChatResponseUpdate, UsageDetails, FileSkillsSource, MCPSkillsSource, DeduplicatingSkillsSource/FilteringSkillsSource, and AgentFileStore/InMemoryAgentFileStore/FileSystemAgentFileStore — all verified against agent-framework 1.21.0 source."
framework: microsoft-agent-framework
language: python
---

# agent-framework (Python) — 10-API Deep Dives Vol. 8

**Verified against:** `agent-framework==1.21.0`
**Python requirement:** 3.10+

This volume covers the in-process workflow execution context, function-based executors, the evaluation data model, the low-level chat-response layer, cross-provider token usage, filesystem and MCP skill discovery, skill pipeline decorators, and the agent file store abstraction. Each section includes the full verified constructor signature, every meaningful method or property, and self-contained runnable examples.

See [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) for `WorkflowViz`, `FileMemoryProvider`, `AgentModeProvider`, `BackgroundAgentsProvider`, `ToolApprovalMiddleware`, `SwitchCaseEdgeGroup`, `MessageInjectionMiddleware`, `ToolResultCompactionStrategy`, `SummarizationStrategy`, and `TokenBudgetComposedStrategy`.

See [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) for `FanInEdgeGroup`, `FanOutEdgeGroup`, `FunctionalWorkflow`, `FunctionalWorkflowAgent`, `FileCheckpointStorage`, `InMemoryCheckpointStorage`, `MCPStdioTool`, `MCPStreamableHTTPTool`, `SelectiveToolCallCompactionStrategy`, and `TodoProvider`.

See [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) for `WorkflowBuilder`, `SlidingWindowStrategy`, `TruncationStrategy`, `ContextWindowCompactionStrategy`, `LocalEvaluator`, `InlineSkill`, `FileAccessProvider`, `MemoryContextProvider`, `FileHistoryProvider`, and `MCPWebsocketTool`.

See [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) for `VectorStoreField`, `VectorStoreCollectionDefinition`, `InMemoryCollection`, `InMemoryStore`, `Filter`, `FilterGroup`, `SecretString`, `load_settings`, `create_agent_hooks_middleware`, and `GroupChatBuilder`.

See [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) for `WorkflowEvent`, `AgentContext`, `MiddlewareBundle`, `ConversationSplit`/`ConversationSplitter`, `VectorStoreHistoryProvider`, `MemoryStore`/`MemoryFileStore`, `MemoryTopicRecord`, `WorkflowRunResult`, `FunctionInvocationContext`, and `ChatOptions`.

See [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) for `Agent`, `RawAgent`, `AgentSession`, `AgentResponse`, `FunctionTool`, `WorkflowAgent`, `WorkflowContext`, `CompactionProvider`, `VectorCollectionContextProvider`, and `TodoItem`/`TodoStore`/`TodoFileStore`.

See [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) for `AgentExecutorRequest`, `AgentExecutorResponse`, `AgentExecutorCheckpointState`, `AgentExecutor`, `WorkflowExecutor`, `WorkflowMessage`, `WorkflowCheckpoint`, `TodoSessionStore`, `ClassSkill`, and `AggregatingSkillsSource`/`CachingSkillsSource`.

> **Vol. 9 available:** See [10-API Deep Dives Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) for `ResponseStream`, `AgentResponseUpdate`, `SessionContext`, `FileSessionStore`, `AgentLoopMiddleware`, `SingleEdgeGroup`, `WorkflowInvocationKwargs`, `StepWrapper`, `DelegatingSkillsSource`, and `SubWorkflowRequestMessage`/`SubWorkflowResponseMessage` — verified against `agent-framework==1.22.0`.

---

## 1. `InProcRunnerContext`

**Module:** `agent_framework._workflows._runner_context` (re-exported via `agent_framework`)

`InProcRunnerContext` is the concrete in-process implementation of the `RunnerContext` protocol — the single object that routes messages between executors, queues workflow events for streaming, and optionally persists checkpoints during a workflow run. Every `Workflow` built with `WorkflowBuilder` creates one internally; you interact with it explicitly only when you need to inject custom checkpoint storage at runtime, drive a request-info response programmatically, or unit-test workflow internals.

> **Protocol vs. concrete class:** `RunnerContext` is a `@runtime_checkable Protocol` that any object can satisfy. `InProcRunnerContext` is the standard concrete implementation. Prefer accessing it through the `Workflow` API; reach for the class directly only for testing or advanced customisation.

### Constructor

```python
InProcRunnerContext(
    checkpoint_storage: CheckpointStorage | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `checkpoint_storage` | `CheckpointStorage \| None` | Build-time checkpoint backend. Can be overridden per-run with `set_runtime_checkpoint_storage()`. |

### Key methods

```python
# Messaging
await ctx.send_message(message: WorkflowMessage) -> None
await ctx.drain_messages() -> dict[str, list[WorkflowMessage]]
await ctx.has_messages() -> bool

# Event streaming
await ctx.add_event(event: WorkflowEvent) -> None
await ctx.drain_events() -> list[WorkflowEvent]
await ctx.has_events() -> bool
await ctx.next_event() -> WorkflowEvent      # waits until next event

# Streaming mode
ctx.set_streaming(True)
ctx.is_streaming() -> bool

# Runtime checkpoint override
ctx.set_runtime_checkpoint_storage(storage: CheckpointStorage)
ctx.clear_runtime_checkpoint_storage()
ctx.has_checkpointing() -> bool

# Human-in-the-loop (request_info)
await ctx.add_request_info_event(event: WorkflowEvent)
await ctx.send_request_info_response(request_id: str, response: Any)
await ctx.get_pending_request_info_events() -> dict[str, WorkflowEvent]
await ctx.cancel_request_info_events(request_ids: set[str]) -> dict[str, WorkflowEvent]

# Checkpointing
await ctx.build_checkpoint(...) -> WorkflowCheckpoint     # in-memory, no storage required
await ctx.create_checkpoint(...) -> CheckpointID          # persists to storage
await ctx.load_checkpoint(checkpoint_id) -> WorkflowCheckpoint | None
await ctx.apply_checkpoint(checkpoint: WorkflowCheckpoint) -> None

# Output classification (output vs. intermediate)
ctx.set_yield_output_classifier(classifier: Callable[[str], "output" | "intermediate" | None])
ctx.classify_yielded_output(executor_id: str) -> "output" | "intermediate" | None

# Runtime tools (request-scoped)
ctx.set_runtime_tools(tools: list[ToolTypes] | None)
ctx.get_runtime_tools() -> list[ToolTypes] | None
ctx.clear_runtime_tools()
```

### Event queue lifecycle note

The internal `asyncio.Queue` is created **lazily** the first time an event is added inside a running event loop. This means a single `InProcRunnerContext` instance can safely be reused across multiple `asyncio.run()` calls — each run gets a fresh queue bound to the correct loop, avoiding the `"bound to a different event loop"` error that would occur if the queue were created in the constructor.

### Example 1 — Inject runtime checkpoint storage per run

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder, InProcRunnerContext,
    InMemoryCheckpointStorage,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client, instructions="You are a helpful assistant.")

exec_node = AgentExecutor(agent, id="main")
workflow = WorkflowBuilder(
    name="chat-wf",
    start_executor=exec_node,
    output_from=[exec_node],
).build()

async def run_with_checkpointing() -> None:
    storage = InMemoryCheckpointStorage()

    # Override checkpoint storage at runtime (without rebuilding the workflow)
    result = await workflow.run(
        "Tell me a fact about penguins.",
        checkpoint_storage=storage,  # Workflow passes this to InProcRunnerContext
    )
    outputs = result.get_outputs()
    print(outputs[0].text)

    # Verify a checkpoint was created
    checkpoint_ids = await storage.list_checkpoint_ids(workflow_name="chat-wf")
    print(f"Checkpoints stored: {len(checkpoint_ids)}")

asyncio.run(run_with_checkpointing())
```

### Example 2 — Resolve a pending request_info event outside the workflow

```python
import asyncio
from agent_framework import (
    Executor, WorkflowBuilder, WorkflowContext, handler, response_handler,
)
from typing_extensions import Never

# Use an Executor subclass so @response_handler can be a registered method.
# (The @executor decorator creates a standalone FunctionExecutor and cannot
# carry @response_handler methods; use a class for any HITL step.)
class ApprovalExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="approval-step")

    @handler
    async def run(self, msg: str, ctx: WorkflowContext[Never, str]) -> None:
        # request_info(request_data, response_type) — returns None; workflow pauses
        await ctx.request_info(
            "Proceed with this action? (yes/no)",  # request_data
            str,                                    # response_type
        )

    @response_handler
    async def handle_approval(
        self,
        original_request: str,
        response: str,
        ctx: WorkflowContext[Never, str],
    ) -> None:
        if response.lower() == "yes":
            await ctx.yield_output("Action approved and completed.")
        else:
            await ctx.yield_output("Action cancelled by user.")

approval_executor = ApprovalExecutor()

wf = WorkflowBuilder(
    name="approval-wf",
    start_executor=approval_executor,
    output_from=[approval_executor],
).build()

async def main() -> None:
    # Phase 1: run until the workflow emits a request_info event.
    # Do NOT break early — drain all events so the workflow reaches idle status
    # before the next run() call, otherwise it raises RuntimeError.
    stream = wf.run("Proceed with the deployment.", stream=True)
    pending_id: str | None = None
    async for event in stream:
        if event.type == "request_info":
            print(f"Workflow paused — question: {event.data}")
            pending_id = event.request_id

    # Phase 2: resume by supplying the operator's answer
    if pending_id:
        result = await wf.run(responses={pending_id: "yes"})
        for out in result.get_outputs():
            print(f"Result: {out}")

asyncio.run(main())
```

---

## 2. `FunctionExecutor`

**Module:** `agent_framework._workflows._function_executor` (re-exported via `agent_framework`)

`FunctionExecutor` wraps any plain Python function (sync or async) into a workflow `Executor`. The `@executor` decorator is the normal way to create one; `FunctionExecutor` is the class the decorator produces. Sync functions are run in a thread pool via `asyncio.to_thread()` so they never block the event loop. The class introspects type annotations to validate message routing at build time, or you can override with explicit `input`, `output`, and `workflow_output` arguments.

> **When to use:** Reach for `FunctionExecutor` (via `@executor`) for pure-logic steps that transform data without calling an LLM. Use `AgentExecutor` when a step requires an `Agent`.

### Constructor

```python
FunctionExecutor(
    func: Callable[..., Any],
    id: str | None = None,
    *,
    input: type | types.UnionType | str | None = None,
    output: type | types.UnionType | str | None = None,
    workflow_output: type | types.UnionType | str | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `func` | `Callable` | Any module-level function, sync or async. `staticmethod`/`classmethod` are rejected; use `@handler` on class instances. |
| `id` | `str \| None` | Executor ID. Defaults to `func.__name__`. |
| `input` | `type \| UnionType \| str \| None` | Explicit input type — overrides introspection of the function's message parameter annotation. Accepts `"MyType \| int"` string forward references. |
| `output` | `type \| UnionType \| str \| None` | Explicit output type for `ctx.send_message()`. Overrides introspection of `WorkflowContext`'s first generic parameter. |
| `workflow_output` | `type \| UnionType \| str \| None` | Explicit type for `ctx.yield_output()`. Overrides the second generic parameter of `WorkflowContext`. |

### `@executor` decorator — the preferred API

```python
from agent_framework import executor, WorkflowContext

@executor(id="my-step")
async def my_step(message: str, ctx: WorkflowContext[str]) -> None:
    transformed = message.upper()
    await ctx.send_message(transformed)
```

The decorator creates a `FunctionExecutor` instance from the decorated function. The instance is then passed to `WorkflowBuilder` like any `AgentExecutor`.

### Example 1 — Simple data-transform pipeline

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, AgentExecutorResponse, FunctionExecutor,
    WorkflowBuilder, WorkflowContext, executor,
)
from agent_framework_openai import OpenAIChatClient
from typing_extensions import Never

client = OpenAIChatClient()
summariser = Agent(client, instructions="Summarise the user's text in one sentence.")

# Step 1: pre-process input (function executor)
@executor(id="normalise")
async def normalise(text: str, ctx: WorkflowContext[str]) -> None:
    await ctx.send_message(text.strip().lower())

# Step 2: LLM summarisation (agent executor)
summarise_exec = AgentExecutor(summariser, id="summarise")

# Step 3: post-process output (function executor — yields final workflow output)
@executor(id="format-output")
async def format_output(summary: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    await ctx.yield_output(f"Summary: {summary.agent_response.text}")

workflow = (
    WorkflowBuilder(name="summarise-pipeline", start_executor=normalise,
                    output_from=[format_output])
    .add_edge(normalise, summarise_exec)
    .add_edge(summarise_exec, format_output)
    .build()
)

async def main() -> None:
    result = await workflow.run("  Alice went to the market and bought apples.  ")
    print(result.get_outputs()[0])  # "Summary: ..."

asyncio.run(main())
```

### Example 2 — Async function executor with explicit types

```python
import asyncio
from agent_framework import executor, WorkflowBuilder, WorkflowContext
from typing_extensions import Never
import json

@executor(id="json-parse", input=str, workflow_output=dict)
async def parse_json(raw: str, ctx: WorkflowContext[Never, dict]) -> None:
    data = json.loads(raw)
    await ctx.yield_output(data)
```

> **Note:** Use `async def` executors so you can call `await ctx.yield_output()` directly. Sync functions run in a thread pool via `asyncio.to_thread()`, where the event loop is inaccessible, so they cannot call async context methods.

### Example 3 — Fan-out with multiple typed outputs

```python
import asyncio
from agent_framework import executor, WorkflowBuilder, WorkflowContext
from agent_framework import Agent, AgentExecutor
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
english_agent = Agent(client, instructions="Answer only in English.")
french_agent = Agent(client, instructions="Answer only in French.")

@executor(id="broadcast")
async def broadcast(query: str, ctx: WorkflowContext[str]) -> None:
    # One send_message is broadcast to all downstream agents in the fan-out group
    await ctx.send_message(query)

eng_exec = AgentExecutor(english_agent, id="english")
fr_exec = AgentExecutor(french_agent, id="french")

workflow = (
    WorkflowBuilder(name="multilingual", start_executor=broadcast,
                    output_from=[eng_exec, fr_exec])
    .add_fan_out_edges(broadcast, [eng_exec, fr_exec])
    .build()
)

async def main() -> None:
    result = await workflow.run("What is the capital of France?")
    for out in result.get_outputs():
        print(out.text)

asyncio.run(main())
```

---

## 3. `EvalItem`

**Module:** `agent_framework._evaluation` (re-exported via `agent_framework`)

`EvalItem` is a single evaluation test case. It holds a full conversation transcript as `Message` objects and exposes `query` / `response` derived via a configurable split strategy. Evaluators (Foundry, `LocalEvaluator`, or custom) receive a list of `EvalItem` instances and return `EvalResults`. The class is marked `@experimental(ExperimentalFeature.EVALS)`.

### Constructor

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

| Parameter | Type | Notes |
|---|---|---|
| `conversation` | `list[Message]` | Full conversation. The single source of truth — `query` and `response` are derived from it. |
| `tools` | `list[FunctionTool] \| None` | Typed tool objects for evaluator logic. |
| `context` | `str \| None` | Optional grounding document for RAG / faithfulness evaluation. |
| `expected_output` | `str \| None` | Ground-truth expected answer for correctness comparisons. |
| `expected_tool_calls` | `list[ExpectedToolCall] \| None` | Expected tool calls; evaluator decides matching semantics. |
| `split_strategy` | `ConversationSplitter \| None` | Controls how `query`/`response` are derived. Default: `ConversationSplit.LAST_TURN`. |

### Properties and methods

```python
item.query      # str — user query text from the query half of the split
item.response   # str — assistant response text from the response half

# Explicit split — returns (query_messages, response_messages)
query_msgs, response_msgs = item.split_messages()
query_msgs, response_msgs = item.split_messages(split=ConversationSplit.FULL)
query_msgs, response_msgs = item.split_messages(split=my_custom_splitter)
```

### `ConversationSplit` built-in strategies

| Strategy | Description |
|---|---|
| `ConversationSplit.LAST_TURN` (default) | Everything up to and including the last user message is the query; everything after is the response. Evaluates whether the agent answered the *latest* question. |
| `ConversationSplit.FULL` | First user message (+ preceding system messages) is the query; the entire remainder is the response. Evaluates whether the *whole conversation* served the original request. |

### Example 1 — Run LocalEvaluator via evaluate_agent

```python
import asyncio
from agent_framework import (
    Agent, EvalResults, LocalEvaluator, evaluate_agent,
    keyword_check, tool_called_check, ExpectedToolCall,
)
from agent_framework_openai import OpenAIChatClient
from agent_framework import tool
from typing import Annotated

client = OpenAIChatClient()

@tool
def get_weather(city: Annotated[str, "City name"]) -> str:
    """Return weather for a city."""
    return f"Sunny, 22°C in {city}"

weather_agent = Agent(
    client,
    instructions="You are a weather assistant. Always call get_weather.",
    tools=[get_weather],
)

local_eval = LocalEvaluator(
    keyword_check("Sunny"),
    tool_called_check("get_weather"),
)

async def main() -> None:
    # evaluate_agent runs the agent against each query, then scores with each evaluator.
    # Returns list[EvalResults] — one EvalResults per evaluator.
    results_list: list[EvalResults] = await evaluate_agent(
        agent=weather_agent,
        queries=["What is the weather in London?", "What is the weather in Tokyo?"],
        expected_tool_calls=[
            [ExpectedToolCall(name="get_weather", arguments={"city": "London"})],
            [ExpectedToolCall(name="get_weather")],
        ],
        evaluators=local_eval,
    )
    for r in results_list:
        pass_rate = r.passed / r.total if r.total else 0.0
        print(f"{r.passed}/{r.total} passed ({pass_rate:.0%})")

asyncio.run(main())
```

### Example 2 — Custom split strategy

```python
from agent_framework import EvalItem, Message, Content, ConversationSplit

def split_before_tool_call(conversation: list[Message]) -> tuple[list[Message], list[Message]]:
    """Split just before the first tool-call message."""
    for i, msg in enumerate(conversation):
        for content in msg.contents or []:
            if content.type == "function_call":
                return conversation[:i], conversation[i:]
    return ConversationSplit.LAST_TURN(conversation)

item = EvalItem(
    conversation=[
        Message("user", ["Find the price of AAPL."]),
        Message("assistant", [
            Content.from_function_call(
                call_id="fc_1",
                name="get_price",
                arguments={"ticker": "AAPL"},
            ),
        ]),
    ],
    split_strategy=split_before_tool_call,
)
query_msgs, response_msgs = item.split_messages()
print(f"Query messages: {len(query_msgs)}, Response messages: {len(response_msgs)}")
```

---

## 4. `EvalResults` and `EvalItemResult`

**Module:** `agent_framework._evaluation` (re-exported via `agent_framework`)

`EvalResults` is the aggregate outcome of one evaluation provider's run. `EvalItemResult` holds per-item pass/fail status, per-evaluator scores, and optional token usage. Both classes are `@experimental(ExperimentalFeature.EVALS)`.

### `EvalResults` constructor

```python
EvalResults(
    *,
    provider: str,
    eval_id: str = "",
    run_id: str = "",
    status: str = "completed",
    result_counts: dict[str, int] | None = None,
    report_url: str | None = None,
    error: str | None = None,
    per_evaluator: dict[str, dict[str, int]] | None = None,
    items: list[EvalItemResult] | None = None,
    sub_results: dict[str, EvalResults] | None = None,
)
```

### `EvalResults` properties and methods

```python
r.passed          # int — count of passing items
r.failed          # int — count of failing items
r.total           # int — passed + failed
r.all_passed      # bool — True only when status=="completed", failed==0,
                  #         errored==0, total>0, and all sub-results also pass

r.raise_for_status()   # raises EvalNotPassedError when all_passed is False
                       # (non-completed run, any failures, any errors, or empty batch)
```

### `EvalItemResult` dataclass

```python
@dataclass
class EvalItemResult:
    item_id: str
    status: str                                  # "pass" | "fail" | "error"
    scores: list[EvalScoreResult]                # per-evaluator verdicts
    error_code: str | None = None
    error_message: str | None = None
    response_id: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    token_usage: dict[str, int] | None = None    # prompt_tokens, completion_tokens, total_tokens
    metadata: dict[str, Any] | None = None

    # Convenience properties
    item.is_error   # status in ("error", "errored")
    item.is_passed  # status == "pass"
    item.is_failed  # status == "fail"
```

### Example 1 — Inspect results and per-item scores

```python
import asyncio
from agent_framework import (
    Agent, evaluate_agent, LocalEvaluator, keyword_check, EvalResults,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client, instructions="You are a knowledgeable assistant.")

local = LocalEvaluator(keyword_check("Paris"))

async def main() -> None:
    results_list: list[EvalResults] = await evaluate_agent(
        agent=agent,
        queries=["What is the capital of France?"],
        evaluators=local,
    )

    for r in results_list:
        print(f"Provider : {r.provider}")
        pass_rate = r.passed / r.total if r.total else 0.0
        print(f"Pass rate: {pass_rate:.0%}  ({r.passed}/{r.total})")
        print(f"Report   : {r.report_url or 'N/A'}")

        for item in r.items:
            tag = "✓" if item.is_passed else ("✗" if item.is_failed else "!")
            print(f"  [{tag}] {item.item_id}: {item.status}")
            for score in item.scores:
                print(f"       {score.name}: {score.score}")

asyncio.run(main())
```

### Example 2 — Fail fast with `raise_for_status`

```python
import asyncio
from agent_framework import (
    Agent, evaluate_agent, LocalEvaluator, keyword_check, EvalNotPassedError,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client, instructions="Answer only with a single word.")

# Strict evaluator — expects "Paris" in the response
strict_eval = LocalEvaluator(keyword_check("Paris"))

async def main() -> None:
    results_list = await evaluate_agent(
        agent=agent,
        queries=["Capital of France?"],
        evaluators=strict_eval,
    )
    try:
        for r in results_list:
            r.raise_for_status()
        print("All evaluations passed!")
    except EvalNotPassedError as e:
        print(f"Evaluation failed: {e}")

asyncio.run(main())
```

### Example 3 — Workflow evaluation with per-agent sub-results

```python
import asyncio
from agent_framework import (
    Agent, AgentExecutor, WorkflowBuilder, evaluate_workflow,
    LocalEvaluator, keyword_check,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
research = Agent(client, instructions="Research the topic.")
writer = Agent(client, instructions="Write a summary.")

research_exec = AgentExecutor(research, id="research")
writer_exec = AgentExecutor(writer, id="writer")

wf = (
    WorkflowBuilder(name="research-wf", start_executor=research_exec,
                    output_from=[writer_exec])
    .add_edge(research_exec, writer_exec)
    .build()
)

local = LocalEvaluator(keyword_check("summary"))

async def main() -> None:
    results_list = await evaluate_workflow(
        workflow=wf,
        queries=["Summarise the history of AI."],
        evaluators=local,
    )
    for r in results_list:
        pass_rate = r.passed / r.total if r.total else 0.0
        print(f"Pass rate: {pass_rate:.0%}  ({r.passed}/{r.total})")
        for agent_name, sub in r.sub_results.items():
            print(f"  {agent_name}: {sub.passed}/{sub.total}")

asyncio.run(main())
```

---

## 5. `ChatResponse` and `ChatResponseUpdate`

**Module:** `agent_framework._types` (re-exported via `agent_framework`)

`ChatResponse` is the raw response produced by a `BaseChatClient` (`OpenAIChatClient`, `FoundryChatClient`, etc.). It sits one layer *below* `AgentResponse` — the agent middleware pipeline translates `ChatResponse` into `AgentResponse`. You encounter `ChatResponse` when implementing a custom chat client, a custom middleware, or when you call `BaseChatClient` directly.

`ChatResponseUpdate` is a single streaming chunk from a chat client. `ChatResponse.from_updates()` combines a sequence of updates into a complete response.

### `ChatResponse` constructor

```python
ChatResponse(
    *,
    messages: Message | Sequence[Message] | None = None,
    response_id: str | None = None,
    conversation_id: str | None = None,
    model: str | None = None,
    created_at: datetime | str | int | float | None = None,
    finish_reason: FinishReasonLiteral | FinishReason | None = None,
    usage_details: UsageDetails | None = None,
    value: ResponseModelT | None = None,       # structured output
    response_format: StructuredResponseFormat = None,
    continuation_token: ContinuationToken | None = None,
    additional_properties: dict[str, Any] | None = None,
    raw_representation: Any | None = None,
)
```

### `ChatResponse` properties and methods

```python
r.text                      # str — concatenated text from all assistant messages
r.messages                  # list[Message] — individual messages
r.finish_reason             # "stop" | "length" | "tool_calls" | "content_filter" | ...
r.usage_details             # UsageDetails TypedDict or None
r.model                     # str | None
r.response_id               # str | None
r.value                     # structured output value (ResponseModelT | None)

# Build from streaming chunks
response = ChatResponse.from_updates(updates)
response = ChatResponse.from_updates(updates, output_format_type=MyModel)

# Serialization
d = response.to_dict()
response = ChatResponse.from_dict(d)
json_str = response.to_json()
response = ChatResponse.from_json(json_str)
```

### `ChatResponseUpdate` constructor

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
    created_at: datetime | str | int | float | None = None,
    finish_reason: FinishReasonLiteral | FinishReason | None = None,
    continuation_token: ContinuationToken | None = None,
    additional_properties: dict[str, Any] | None = None,
    raw_representation: Any | None = None,
)
```

### Example 1 — Call a chat client directly and inspect the response

```python
import asyncio
from agent_framework import ChatResponse, Message, UsageDetails
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    messages = [Message("user", ["What is 2 + 2?"])]
    response: ChatResponse = await client.get_response(messages)

    print(f"Text        : {response.text}")
    print(f"Model       : {response.model}")
    print(f"Finish      : {response.finish_reason}")

    usage: UsageDetails | None = response.usage_details
    if usage:
        print(f"Input tokens : {usage.get('input_token_count')}")
        print(f"Output tokens: {usage.get('output_token_count')}")
        print(f"Total tokens : {usage.get('total_token_count')}")

asyncio.run(main())
```

### Example 2 — Aggregate streaming updates

```python
import asyncio
from agent_framework import ChatResponseUpdate, ChatResponse, Message, Content
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    messages = [Message("user", ["Tell me a haiku about the ocean."])]

    updates: list[ChatResponseUpdate] = []
    async for update in client.get_response(messages, stream=True):
        updates.append(update)
        # Print incremental text
        if update.text:
            print(update.text, end="", flush=True)
    print()

    # Combine into a complete response
    complete: ChatResponse = ChatResponse.from_updates(updates)
    print(f"\nFull response: {complete.text}")
    print(f"Response ID  : {complete.response_id}")

asyncio.run(main())
```

### Example 3 — Serialise and restore a response

```python
from agent_framework import ChatResponse, Message, UsageDetails

response = ChatResponse(
    messages=[Message("assistant", ["The answer is 42."])],
    model="gpt-4o",
    finish_reason="stop",
    usage_details={
        "input_token_count": 12,
        "output_token_count": 8,
        "total_token_count": 20,
    },
)

# Round-trip via JSON
json_str = response.to_json()
restored = ChatResponse.from_json(json_str)
assert restored.text == "The answer is 42."
assert restored.model == "gpt-4o"
print("Round-trip OK:", restored.usage_details)
```

---

## 6. `UsageDetails`

**Module:** `agent_framework._types` (re-exported via `agent_framework`)

`UsageDetails` is a `TypedDict` that standardises token-usage reporting across all LLM providers. It is returned on `ChatResponse.usage_details` and surfaced through `AgentResponse.usage_details`. All keys are optional (`total=False`) so providers populate only the fields they support.

### Type definition

```python
class UsageDetails(TypedDict, total=False, extra_items=int):
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    cache_creation_input_token_count: int | None    # provider-managed cache write tokens
    cache_read_input_token_count: int | None        # provider-managed cache read tokens
    reasoning_output_token_count: int | None        # reasoning/thinking tokens
    # Any provider-specific int field is also accepted (extra_items=int)
```

The `extra_items=int` annotation means a provider can add non-standard int fields (e.g. `"audio_tokens"`) without breaking type checkers.

### Example 1 — Aggregate token usage across multiple agent calls

```python
import asyncio
from agent_framework import Agent, AgentResponse, UsageDetails
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client, instructions="You are a helpful assistant.")

async def main() -> None:
    queries = [
        "What is the speed of light?",
        "Explain quantum entanglement briefly.",
        "What is the Fibonacci sequence?",
    ]

    total_input = 0
    total_output = 0
    total_cache_read = 0

    for query in queries:
        response: AgentResponse = await agent.run(query)
        usage: UsageDetails | None = response.usage_details
        if usage:
            total_input += usage.get("input_token_count") or 0
            total_output += usage.get("output_token_count") or 0
            total_cache_read += usage.get("cache_read_input_token_count") or 0

    print(f"Total input tokens  : {total_input}")
    print(f"Total output tokens : {total_output}")
    print(f"Cache reads         : {total_cache_read}")

asyncio.run(main())
```

### Example 2 — Budget guard middleware using UsageDetails

```python
import asyncio
from agent_framework import (
    Agent, ChatContext, ChatMiddleware, UsageDetails,
)
from agent_framework_openai import OpenAIChatClient

class TokenBudgetGuard(ChatMiddleware):
    """Raises after any single call that exceeds the per-call output budget."""

    def __init__(self, max_output_tokens: int) -> None:
        self._budget = max_output_tokens

    async def process(self, context: ChatContext, call_next) -> None:
        await call_next()
        usage: UsageDetails | None = (
            context.result.usage_details if context.result else None
        )
        if usage:
            output_count = usage.get("output_token_count") or 0
            if output_count > self._budget:
                raise RuntimeError(
                    f"Output tokens {output_count} exceeded budget {self._budget}."
                )

client = OpenAIChatClient()
agent = Agent(
    client,
    instructions="You are a concise assistant.",
    middleware=[TokenBudgetGuard(max_output_tokens=200)],
)

async def main() -> None:
    try:
        response = await agent.run("Write a 10-page essay on philosophy.")
        print(response.text[:200])
    except RuntimeError as e:
        print(f"Budget exceeded: {e}")

asyncio.run(main())
```

---

## 7. `FileSkillsSource`

**Module:** `agent_framework._skills` (re-exported via `agent_framework`)

`FileSkillsSource` discovers skills by recursively scanning directories for `SKILL.md` files (up to 2 levels deep by default). Each `SKILL.md` is parsed for YAML frontmatter and optionally paired with resource files (`.md`, `.json`, `.yaml`, `.csv`, etc.) and runnable script files. The resulting `FileSkill` instances are served through `SkillsProvider` on each agent invocation.

### Constructor

```python
FileSkillsSource(
    skill_paths: str | Path | Sequence[str | Path],
    *,
    script_runner: SkillScriptRunner | None = None,
    resource_extensions: tuple[str, ...] | None = (".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".txt"),
    script_extensions: tuple[str, ...] | None = DEFAULT_SCRIPT_EXTENSIONS,
    search_depth: int = 2,
    script_filter: Callable[[str, str], bool] | None = None,
    resource_filter: Callable[[str, str], bool] | None = None,
)
```

| Parameter | Type | Notes |
|---|---|---|
| `skill_paths` | `str \| Path \| Sequence` | One or more root directories to scan for `SKILL.md` files. |
| `script_runner` | `SkillScriptRunner \| None` | Strategy for executing scripts discovered inside skills. Required only if skills expose runnable scripts. |
| `resource_extensions` | `tuple[str, ...] \| None` | Extensions that qualify as resources. Pass an empty tuple to skip resource discovery. |
| `script_extensions` | `tuple[str, ...] \| None` | Extensions that qualify as executable scripts. |
| `search_depth` | `int` | Depth limit for resource and script traversal **within each discovered skill directory** (not for finding `SKILL.md` files). Default `2`. |
| `script_filter` | `Callable[[name, path], bool] \| None` | Return `False` to exclude a script from discovery. |
| `resource_filter` | `Callable[[name, path], bool] \| None` | Return `False` to exclude a resource from discovery. |

### Example 1 — Basic filesystem skill loading

```python
# skills/
#   code-review/
#     SKILL.md         ← frontmatter: name, description
#     guidelines.md    ← resource file

import asyncio
from pathlib import Path
from agent_framework import Agent, FileSkillsSource, SkillsProvider
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

source = FileSkillsSource(skill_paths=Path("./skills"))

# disable_load_skill_approval=True lets the agent call load_skill without
# an approval handshake. In production, omit this and wire ToolApprovalMiddleware.
agent = Agent(
    client,
    instructions="You are a code review assistant.",
    context_providers=[SkillsProvider(source, disable_load_skill_approval=True)],
)

async def main() -> None:
    response = await agent.run("Review this Python function for best practices.")
    print(response.text)

asyncio.run(main())
```

### Example 2 — Multiple skill directories with filters

```python
from pathlib import Path
from agent_framework import FileSkillsSource, SkillsProvider, Agent
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

import os

# Callbacks receive (skill_name, relative_file_path); inspect path's basename
# to filter by file name rather than the containing skill's name.
def not_test_script(name: str, path: str) -> bool:
    return not os.path.basename(path).startswith("test_")

def not_hidden_resource(name: str, path: str) -> bool:
    return not os.path.basename(path).startswith(".")

source = FileSkillsSource(
    skill_paths=[Path("./skills/production"), Path("./skills/shared")],
    search_depth=3,
    script_filter=not_test_script,
    resource_filter=not_hidden_resource,
)

agent = Agent(
    client,
    instructions="Use available skills to answer user queries.",
    context_providers=[SkillsProvider(source, disable_load_skill_approval=True)],
)
```

### Example 3 — Inspect discovered skills at startup

```python
import asyncio
from agent_framework import FileSkillsSource, SkillsSourceContext, Agent
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client, instructions="Assistant")

source = FileSkillsSource("./skills")

async def list_skills() -> None:
    context = SkillsSourceContext(agent=agent)
    skills = await source.get_skills(context)
    for skill in skills:
        print(f"- {skill.frontmatter.name}: {skill.frontmatter.description}")
        content = await skill.get_content()
        print(f"  Content length: {len(content)} chars")

asyncio.run(list_skills())
```

---

## 8. `MCPSkillsSource`

**Module:** `agent_framework._skills` (re-exported via `agent_framework`)

`MCPSkillsSource` discovers agent skills from an MCP server that implements the SEP-2640 agent-skills convention. It reads the well-known `skill://index.json` resource and builds one `MCPSkill` per entry. Entries of type `skill-md` reference a lazy-fetched `SKILL.md`; entries of type `archive` contain a ZIP file that is unpacked in memory. This class is `@experimental(ExperimentalFeature.MCP_SKILLS)`.

### Constructor

```python
MCPSkillsSource(
    client: ClientSession | None = None,
    *,
    session_provider: Callable[[], ClientSession] | None = None,
    # archive options
    archive_resource_extensions: tuple[str, ...] | None = None,
    archive_resource_search_depth: int = 2,
    archive_max_file_count: int = 20,
    archive_max_size_bytes: int = 1024 * 1024,           # 1 MB default
    archive_max_uncompressed_size_bytes: int = 1024 * 1024,  # 1 MB default
)
```

Provide **exactly one** of `client` or `session_provider`. Use `session_provider` when the underlying MCP connection may be replaced (e.g., a reconnecting `MCPStdioTool`).

### Example 1 — Discover skills from an MCP server

```python
import asyncio
from agent_framework import (
    Agent, MCPStdioTool, MCPSkillsSource, SkillsProvider,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

async def main() -> None:
    async with MCPStdioTool(name="skill-server", command="python3", args=["my_skill_server.py"]) as mcp:
        # Use the live MCP session for skill discovery
        source = MCPSkillsSource(session_provider=lambda: mcp.session)

        agent = Agent(
            client,
            instructions="Use available skills to help users.",
            tools=[mcp],
            context_providers=[SkillsProvider(source, disable_load_skill_approval=True)],
        )

        response = await agent.run("What skills do you have available?")
        print(response.text)

asyncio.run(main())
```

### Example 2 — Archive-based skills with custom limits

```python
import asyncio
from agent_framework import (
    Agent, MCPStdioTool, MCPSkillsSource, SkillsSourceContext, SkillsProvider,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client, instructions="Assistant")

async def inspect_mcp_skills() -> None:
    async with MCPStdioTool(name="skill-server", command="python3", args=["skill_server.py"]) as mcp:
        # Configure archive limits for large skill bundles
        source = MCPSkillsSource(
            session_provider=lambda: mcp.session,
            archive_max_file_count=50,
            archive_max_size_bytes=5 * 1024 * 1024,       # 5 MB
            archive_max_uncompressed_size_bytes=10 * 1024 * 1024,  # 10 MB
        )

        ctx = SkillsSourceContext(agent=agent)
        skills = await source.get_skills(ctx)
        for skill in skills:
            print(f"Skill: {skill.frontmatter.name}")
            # get_content() fetches SKILL.md via resources/read on demand
            content = await skill.get_content()
            print(f"  Length: {len(content)} chars")
            # get_resource() fetches supporting files on demand
            resource = await skill.get_resource("guidelines.md")
            if resource:
                print(f"  Resource: {(await resource.read())[:80]}...")

asyncio.run(inspect_mcp_skills())
```

---

## 9. `DeduplicatingSkillsSource` and `FilteringSkillsSource`

**Module:** `agent_framework._skills` (re-exported via `agent_framework`)

These two decorator classes wrap any `SkillsSource` to post-process the list it returns. Both extend `DelegatingSkillsSource` and can be composed freely with `AggregatingSkillsSource`, `CachingSkillsSource`, and each other to build a multi-layer skills pipeline.

### `DeduplicatingSkillsSource`

Removes skills that share the same name (case-insensitive) as an earlier skill — first-one-wins. Useful when composing multiple sources that might serve overlapping skill sets.

```python
DeduplicatingSkillsSource(inner_source: SkillsSource)
```

### `FilteringSkillsSource`

Keeps only skills for which a predicate returns `True`. The predicate receives both the `Skill` and the live `SkillsSourceContext`, enabling per-session or per-agent filtering.

```python
FilteringSkillsSource(
    inner_source: SkillsSource,
    predicate: Callable[[Skill, SkillsSourceContext], bool],
)
```

### Example 1 — Compose a full skills pipeline

```python
import asyncio
from agent_framework import (
    Agent, FileSkillsSource, InMemorySkillsSource, InlineSkill, SkillFrontmatter,
    AggregatingSkillsSource, DeduplicatingSkillsSource,
    FilteringSkillsSource, CachingSkillsSource, SkillsProvider,
    Skill, SkillsSourceContext,
)
from agent_framework_openai import OpenAIChatClient
from datetime import timedelta

client = OpenAIChatClient()

# Three independent sources
file_source = FileSkillsSource("./skills/production")
shared_source = FileSkillsSource("./skills/shared")

emergency_skill = InlineSkill(
    frontmatter=SkillFrontmatter(
        name="emergency-protocol",
        description="Emergency response procedures",
    ),
    instructions="Follow these steps for emergency situations: ...",
)
inline_source = InMemorySkillsSource([emergency_skill])

# 1. Combine all sources
combined = AggregatingSkillsSource([file_source, shared_source, inline_source])

# 2. Deduplicate (production skills take priority over shared when names clash)
deduped = DeduplicatingSkillsSource(combined)

# 3. Filter out skills marked as internal
def is_public(skill: Skill, ctx: SkillsSourceContext) -> bool:
    return "internal" not in skill.frontmatter.name.lower()

filtered = FilteringSkillsSource(deduped, predicate=is_public)

# 4. Cache the result for 5 minutes to avoid repeated filesystem scans
cached = CachingSkillsSource(filtered, refresh_interval=timedelta(minutes=5))

agent = Agent(
    client,
    instructions="Use the available skills to help users.",
    context_providers=[SkillsProvider(cached, disable_load_skill_approval=True)],
)
```

### Example 2 — Context-aware filtering (per-agent skill selection)

```python
from agent_framework import (
    FilteringSkillsSource, Skill, SkillsSourceContext, FileSkillsSource,
    SkillsProvider, Agent,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

# Only expose skills that match the agent's own name prefix
def agent_scoped_filter(skill: Skill, ctx: SkillsSourceContext) -> bool:
    agent_name = (ctx.agent.name or "").lower()
    skill_name = skill.frontmatter.name.lower()
    return skill_name.startswith(agent_name)

base_source = FileSkillsSource("./skills")
scoped = FilteringSkillsSource(base_source, predicate=agent_scoped_filter)

finance_agent = Agent(
    client,
    name="finance",
    instructions="You are a financial assistant.",
    context_providers=[SkillsProvider(scoped, disable_load_skill_approval=True)],
    # Only skills starting with "finance" will be injected for this agent
)
```

### Example 3 — Standalone use without SkillsProvider (for testing)

```python
import asyncio
from agent_framework import (
    InlineSkill, SkillFrontmatter, InMemorySkillsSource, DeduplicatingSkillsSource,
    FilteringSkillsSource, SkillsSourceContext, Agent,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()
agent = Agent(client, instructions="Assistant")

skill_a = InlineSkill(frontmatter=SkillFrontmatter(name="weather", description="Weather skill"), instructions="...")
skill_b = InlineSkill(frontmatter=SkillFrontmatter(name="weather", description="Duplicate weather skill"), instructions="...")
skill_c = InlineSkill(frontmatter=SkillFrontmatter(name="internal-admin", description="Internal only"), instructions="...")

source = InMemorySkillsSource([skill_a, skill_b, skill_c])
deduped = DeduplicatingSkillsSource(source)
filtered = FilteringSkillsSource(
    deduped,
    predicate=lambda s, _: not s.frontmatter.name.startswith("internal"),
)

async def main() -> None:
    ctx = SkillsSourceContext(agent=agent)
    skills = await filtered.get_skills(ctx)
    # Result: only skill_a ("weather") — skill_b deduplicated, skill_c filtered
    for s in skills:
        print(s.frontmatter.name)

asyncio.run(main())
```

---

## 10. `AgentFileStore`, `InMemoryAgentFileStore`, and `FileSystemAgentFileStore`

**Module:** `agent_framework._harness._file_access` (re-exported via `agent_framework`)

`AgentFileStore` is the abstract base for persistent file storage exposed to agents via `FileAccessProvider`. Two concrete implementations ship with the framework: `InMemoryAgentFileStore` (dict-backed, suitable for tests) and `FileSystemAgentFileStore` (disk-backed, with traversal and symlink protections). All three are `@experimental(ExperimentalFeature.HARNESS)`.

> **Relationship to `FileAccessProvider`:** `FileAccessProvider` registers CRUD and search tools (`file_access_write`, `file_access_read`, `file_access_read_lines`, `file_access_delete`, `file_access_ls`, `file_access_grep`, `file_access_replace`, `file_access_replace_lines`) on the session context. It delegates every operation to its configured `AgentFileStore`. Unlike `MemoryContextProvider` (session-scoped), the file store is shared and persistent across sessions.

### `AgentFileStore` abstract interface

```python
class AgentFileStore(ABC):
    # All paths are relative to the implementation's root.
    # Implementations reject path traversal (e.g. ".." segments).

    async def write(self, path: str, content: str, *, overwrite: bool = True) -> None: ...
    async def read(self, path: str) -> str | None: ...
    async def delete(self, path: str) -> bool: ...
    async def list_children(self, directory: str = "") -> list[FileStoreEntry]: ...
    async def file_exists(self, path: str) -> bool: ...
    async def create_directory(self, path: str) -> None: ...
    async def search(
        self,
        directory: str,
        regex_pattern: str,
        glob_pattern: str | None = None,
        *,
        recursive: bool = False,
    ) -> list[FileSearchResult]: ...
```

### `InMemoryAgentFileStore` constructor

```python
InMemoryAgentFileStore()     # no parameters — empty store
```

Thread-safe via `asyncio.Lock`. Key lookup is case-insensitive so the store behaves consistently on both case-sensitive and case-insensitive filesystems.

### `FileSystemAgentFileStore` constructor

```python
FileSystemAgentFileStore(root_directory: str | os.PathLike[str])
```

The root directory is **not** created at construction — it is created lazily on the first write. Symbolic links and reparse points anywhere along a resolved path are rejected.

### Example 1 — Test an agent with InMemoryAgentFileStore

> **Approval note:** `FileAccessProvider` tools require user approval by default. In production use `ToolApprovalMiddleware` to handle the approval handshake. For unattended tests, pass `disable_readonly_tool_approval=True` (read-only tools) or `disable_write_tool_approval=True` (write tools) to `FileAccessProvider`.

```python
import asyncio
from agent_framework import (
    Agent, FileAccessProvider, InMemoryAgentFileStore,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

store = InMemoryAgentFileStore()
# Disable read-only approval for this automated test
file_provider = FileAccessProvider(store=store, disable_readonly_tool_approval=True)

agent = Agent(
    client,
    instructions=(
        "You have access to a file store. "
        "Use it to save and retrieve information as instructed."
    ),
    context_providers=[file_provider],
)

async def main() -> None:
    # Seed the store with a document before the agent runs
    await store.write("project/notes.md", "# Project Notes\n\n- Deadline: 2026-12-01")

    response = await agent.run("Read the project notes and summarise the deadline.")
    print(response.text)

    # Inspect the store after the run
    children = await store.list_children("project")
    for entry in children:
        print(f"  {entry.name} ({entry.type})")

asyncio.run(main())
```

### Example 2 — Production setup with FileSystemAgentFileStore

> **Approval note:** In production, wire `ToolApprovalMiddleware` or use `create_harness_agent()` — both handle the approval handshake that `FileAccessProvider` requires before tools execute.

```python
import asyncio
from pathlib import Path
from agent_framework import (
    Agent, FileAccessProvider, FileSystemAgentFileStore,
)
from agent_framework_openai import OpenAIChatClient

client = OpenAIChatClient()

# Root directory is created lazily on first write
store = FileSystemAgentFileStore(root_directory=Path("/var/app/agent-files"))
# disable_write_tool_approval=True lets writes run unattended in this demo.
# In production, remove these flags and wire ToolApprovalMiddleware or
# use create_harness_agent() to handle the approval handshake instead.
file_provider = FileAccessProvider(
    store=store,
    disable_readonly_tool_approval=True,
    disable_write_tool_approval=True,
)

agent = Agent(
    client,
    instructions="You have access to persistent file storage. Save and retrieve files as needed.",
    context_providers=[file_provider],
)

async def main() -> None:
    # Agent can use file_access_write, file_access_read, file_access_ls, etc.
    response = await agent.run(
        "Write a file called 'report.md' with today's summary, then read it back."
    )
    print(response.text)

    # Verify the file was written
    content = await store.read("report.md")
    print(f"\nStored content (first 200 chars): {(content or '')[:200]}")

asyncio.run(main())
```

### Example 3 — Direct CRUD and search operations

```python
import asyncio
from agent_framework import InMemoryAgentFileStore, FileSearchResult

async def main() -> None:
    store = InMemoryAgentFileStore()

    # Write files
    await store.write("docs/intro.md", "# Introduction\nWelcome to the project.")
    await store.write("docs/api.md", "# API Reference\nEndpoints and schemas.")
    await store.write("config/settings.json", '{"debug": true, "version": "1.0.0"}')

    # Read
    content = await store.read("docs/intro.md")
    print(f"Read: {content}")

    # List directory
    entries = await store.list_children("docs")
    print(f"Files in docs/: {[e.name for e in entries]}")

    # Search with regex across all files (recursive)
    results: list[FileSearchResult] = await store.search(
        directory="",
        regex_pattern=r"version",
        recursive=True,
    )
    for result in results:
        for match in result.matching_lines:
            print(f"Match in {result.file_name}: line {match.line_number} — {match.line.strip()}")

    # Overwrite guard
    try:
        await store.write("docs/intro.md", "Overwrite attempt", overwrite=False)
    except FileExistsError as e:
        print(f"Rejected: {e}")

    # Delete
    deleted = await store.delete("config/settings.json")
    print(f"Deleted settings.json: {deleted}")

    # Existence check
    print(f"docs/intro.md exists: {await store.file_exists('docs/intro.md')}")
    print(f"config/settings.json exists: {await store.file_exists('config/settings.json')}")

asyncio.run(main())
```

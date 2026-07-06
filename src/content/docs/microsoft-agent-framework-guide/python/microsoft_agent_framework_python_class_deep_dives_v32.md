---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 32"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.10.0: evaluate_agent/evaluate_workflow, EvalResults/EvalItem, keyword_check/tool_called_check/evaluator decorator, WorkflowEvent factory catalog, WorkflowErrorDetails/WorkflowRunState, compaction annotation constants, background_tasks_running_message/todos_remaining_message, ToolApprovalRuleCallback/create_always_approve helpers, FileStoreEntry/DEFAULT_FILE_ACCESS constants, and BackgroundAgentsProvider wait/continue deep patterns."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 55
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 32

All examples are source-verified against **agent-framework 1.10.0**. This volume covers
ten class groups that were previously undocumented or only lightly covered.

---

## 1. `evaluate_agent` + `evaluate_workflow` + `LocalEvaluator` + `EvalItem`

**Module:** `agent_framework._evaluation`  
**Imports:** `from agent_framework import evaluate_agent, evaluate_workflow, LocalEvaluator, EvalItem`

The evaluation subsystem is marked `@experimental` (feature ID `ExperimentalFeature.EVALS`). It provides
provider-agnostic evaluation of individual agents and multi-agent workflows, using a common `Evaluator` protocol.

`EvalItem` is the unit of evaluation: it wraps a `conversation: list[Message]` and exposes `query`
and `response` computed properties derived by splitting the conversation via a `ConversationSplitter`.
The default split is `ConversationSplit.LAST_TURN` — everything up to and including the last user message
is the query; everything after is the response.

```python
# EvalItem construction
from agent_framework import EvalItem, Message
item = EvalItem(
    conversation=[Message("user", ["What is 2+2?"]), Message("assistant", ["4"])],
    context="Math facts",
    expected_output="4",
)
print(item.query)     # "What is 2+2?"
print(item.response)  # "4"
```

`LocalEvaluator` implements the `Evaluator` protocol and runs checks locally (no API calls). It accepts
`*checks: EvalCheck` — any callable `(item: EvalItem) -> CheckResult | Awaitable[CheckResult]`.

`evaluate_agent` runs an agent against test queries, builds `EvalItem`s, and submits them to one or more
evaluators. `evaluate_workflow` does the same for multi-agent workflows, producing per-agent breakdowns in
`EvalResults.sub_results`.

### Example 1 — Single-agent local evaluation

```python
import asyncio
from agent_framework import (
    Agent,
    FunctionTool,
    LocalEvaluator,
    evaluate_agent,
    keyword_check,
    tool_called_check,
)
from agent_framework.openai import OpenAIChatClient


async def get_weather(location: str) -> str:
    """Return current weather for a location."""
    return f"The weather in {location} is sunny and 22°C."


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a weather assistant. Use get_weather to answer weather queries.",
        name="weather-agent",
        tools=[FunctionTool(get_weather)],
    )

    local = LocalEvaluator(
        keyword_check("weather"),
        tool_called_check("get_weather"),
    )

    results = await evaluate_agent(
        agent=agent,
        queries=["What's the weather like in London today?"],
        evaluators=local,
    )

    for r in results:
        print(f"{r.provider}: {r.passed}/{r.total} passed")
        for item in r.items:
            print(f"  item {item.item_id}: {item.status}")
            for score in item.scores:
                print(f"    {score.name}: {'pass' if score.passed else 'fail'}")


asyncio.run(main())
```

### Example 2 — Workflow evaluation with per-agent breakdown

```python
import asyncio
from agent_framework import evaluate_workflow, LocalEvaluator, keyword_check
from agent_framework.openai import OpenAIChatClient
from agent_framework import WorkflowBuilder, Agent


async def main() -> None:
    planner = Agent(client=OpenAIChatClient(), name="planner", instructions="Plan travel itineraries.")
    booker = Agent(client=OpenAIChatClient(), name="booker", instructions="Book travel arrangements.")

    wf = (
        WorkflowBuilder(start_executor=planner, name="travel")
        .add_chain([planner, booker])
        .build()
    )
    local = LocalEvaluator(keyword_check("Paris"))

    results = await evaluate_workflow(
        workflow=wf,
        queries=["Plan a 3-day trip to Paris"],
        evaluators=local,
        include_per_agent=True,
        include_overall=True,
    )

    for r in results:
        print(f"Overall: {r.passed}/{r.total}")
        for name, sub in r.sub_results.items():
            print(f"  {name}: {sub.passed}/{sub.total}")


asyncio.run(main())
```

### Example 3 — Evaluate pre-existing responses without re-running the agent

```python
import asyncio
from agent_framework import evaluate_agent, LocalEvaluator, keyword_check, Message, Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="You are helpful.")
    local = LocalEvaluator(keyword_check("helpful"))

    # Run once and cache the response
    response = await agent.run([Message("user", ["Are you helpful?"])])

    # Evaluate without a second model call
    results = await evaluate_agent(
        agent=agent,
        queries="Are you helpful?",
        responses=response,
        evaluators=local,
    )
    results[0].raise_for_status()
    print("Evaluation passed")


asyncio.run(main())
```

---

## 2. `EvalResults` + `EvalItemResult` + `EvalScoreResult` + `ConversationSplit` + `EvalNotPassedError`

**Module:** `agent_framework._evaluation`  
**Imports:** `from agent_framework import EvalResults, EvalItemResult, EvalScoreResult, ConversationSplit, EvalNotPassedError`

`EvalResults` wraps the output of one provider's evaluation run. Its key properties are:

| Property | Description |
|---|---|
| `passed` | Count of items that passed all checks |
| `failed` | Count of items that failed at least one check |
| `total` | `passed + failed` |
| `all_passed` | True only if `failed == 0 and errored == 0 and total > 0` |
| `items` | `list[EvalItemResult]` — per-item detail |
| `sub_results` | `dict[str, EvalResults]` — per-executor breakdown for workflow evals |
| `report_url` | Portal link for cloud providers (Foundry etc.) |

Three assertion methods make `EvalResults` suitable for CI gates:

```python
results.raise_for_status()                       # fail if any item failed
results.assert_score_at_least(0.80)              # fail if any score < 0.80
results.assert_dimension_score_at_least("accuracy", 0.75)  # rubric-based
```

`ConversationSplit` is a `str` Enum with two built-in splitters callable as functions:

```python
from agent_framework import ConversationSplit, Message

conversation = [
    Message("user", ["hello"]),
    Message("assistant", ["hi"]),
    Message("user", ["follow up"]),
    Message("assistant", ["response"]),
]

query_msgs, response_msgs = ConversationSplit.LAST_TURN(conversation)
# query_msgs: first 3 messages (up to and including last user message)
# response_msgs: final assistant message

full_query, full_response = ConversationSplit.FULL(conversation)
# full_query: first user message only
# full_response: all remaining messages
```

### Example 1 — CI gate with `raise_for_status`

```python
import asyncio
from agent_framework import evaluate_agent, LocalEvaluator, keyword_check, Agent
from agent_framework.openai import OpenAIChatClient


async def test_agent_ci() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Answer math questions.")
    local = LocalEvaluator(keyword_check("4"))

    results = await evaluate_agent(
        agent=agent,
        queries=["What is 2 + 2?"],
        evaluators=local,
    )

    # Raises EvalNotPassedError in CI if any item failed
    results[0].raise_for_status()
    print("CI gate passed")


asyncio.run(test_agent_ci())
```

### Example 2 — Inspecting per-item scores

```python
import asyncio
from agent_framework import evaluate_agent, LocalEvaluator, keyword_check, Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="You are helpful.")
    local = LocalEvaluator(keyword_check("helpful", "assist"))

    results = await evaluate_agent(
        agent=agent,
        queries=["What can you help with?"],
        evaluators=local,
    )

    for r in results:
        for item in r.items:
            print(f"Item {item.item_id}: {item.status}")
            print(f"  query:    {item.input_text}")
            print(f"  response: {item.output_text}")
            for score in item.scores:
                detail = score.sample.get("reason", "") if score.sample else ""
                print(f"  {score.name}: {'pass' if score.passed else 'fail'} — {detail}")


asyncio.run(main())
```

### Example 3 — Custom split strategy

Pass any callable with signature `(list[Message]) -> tuple[list[Message], list[Message]]` as `conversation_split`.

```python
import asyncio
from agent_framework import evaluate_agent, LocalEvaluator, keyword_check, Message, Agent, ConversationSplit
from agent_framework.openai import OpenAIChatClient


def split_before_tool(conversation: list[Message]) -> tuple[list[Message], list[Message]]:
    """Split just before the first tool call to evaluate the final answer only."""
    for i, msg in enumerate(conversation):
        for c in msg.contents or []:
            if c.type == "function_call":
                return conversation[:i], conversation[i:]
    return ConversationSplit.LAST_TURN(conversation)  # fallback


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Answer questions using tools.")
    local = LocalEvaluator(keyword_check("result"))

    results = await evaluate_agent(
        agent=agent,
        queries=["Compute 42 * 17"],
        evaluators=local,
        conversation_split=split_before_tool,
    )
    results[0].raise_for_status()


asyncio.run(main())
```

---

## 3. `keyword_check` + `tool_called_check` + `tool_calls_present` + `tool_call_args_match` + `@evaluator`

**Module:** `agent_framework._evaluation`  
**Imports:** `from agent_framework import keyword_check, tool_called_check, tool_calls_present, tool_call_args_match, evaluator`

These are the built-in `EvalCheck` factory functions. Each returns an `EvalCheck` callable suitable for
`LocalEvaluator`.

| Function | What it checks |
|---|---|
| `keyword_check(*words, case_sensitive=False)` | Response contains all listed keywords |
| `tool_called_check(*names, mode="all")` | Specified tools were called (`mode="any"` for at-least-one) |
| `tool_calls_present(item)` | All `item.expected_tool_calls` names appear in conversation |
| `tool_call_args_match(item)` | All expected tool calls match on name AND arguments (subset match) |
| `@evaluator` | Wrap any plain function as an `EvalCheck` |

`tool_calls_present` and `tool_call_args_match` are standalone `EvalCheck` functions (not factories) that
read `item.expected_tool_calls` — set those via `evaluate_agent(expected_tool_calls=...)`.

### Example 1 — Keyword and tool-call checks

```python
import asyncio
from agent_framework import (
    Agent,
    FunctionTool,
    LocalEvaluator,
    evaluate_agent,
    keyword_check,
    tool_called_check,
)
from agent_framework.openai import OpenAIChatClient


async def get_weather(location: str) -> str:
    """Return current weather for a location."""
    return f"The weather in {location} is 18°C and partly cloudy."


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Use get_weather to answer weather questions.",
        tools=[FunctionTool(get_weather)],
    )
    local = LocalEvaluator(
        keyword_check("London", "degrees"),      # both must appear in response
        tool_called_check("get_weather"),         # tool must have been called
    )
    results = await evaluate_agent(
        agent=agent,
        queries=["What is the weather in London right now?"],
        evaluators=local,
    )
    results[0].raise_for_status()


asyncio.run(main())
```

### Example 2 — Argument-level tool-call assertion with `tool_call_args_match`

```python
import asyncio
from agent_framework import (
    Agent,
    ExpectedToolCall,
    FunctionTool,
    LocalEvaluator,
    evaluate_agent,
    tool_call_args_match,
)
from agent_framework.openai import OpenAIChatClient


async def get_weather(location: str) -> str:
    """Return current weather for a location."""
    return f"The weather in {location} is 18°C and partly cloudy."


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Use get_weather for weather queries.",
        tools=[FunctionTool(get_weather)],
    )
    local = LocalEvaluator(tool_call_args_match)  # pass the function directly

    results = await evaluate_agent(
        agent=agent,
        queries=["What's the weather in New York City?"],
        expected_tool_calls=[ExpectedToolCall("get_weather", {"location": "New York City"})],
        evaluators=local,
    )
    results[0].raise_for_status()


asyncio.run(main())
```

### Example 3 — Custom check with the `@evaluator` decorator

`@evaluator` wraps any plain function that accepts named parameters matching `EvalItem` fields
(`query`, `response`, `expected_output`, `conversation`, `tools`, `context`). It handles sync and async
functions; return `bool`, `float`, `dict`, or `CheckResult`.

```python
import asyncio
from agent_framework import Agent, LocalEvaluator, evaluate_agent, evaluator
from agent_framework.openai import OpenAIChatClient


@evaluator
def not_too_long(response: str) -> bool:
    """Response should be concise."""
    return len(response) < 2000


@evaluator(name="contains_expected")
def ground_truth_match(response: str, expected_output: str) -> float:
    """Soft match: score 1.0 if expected text found anywhere in response."""
    return 1.0 if expected_output.lower() in response.lower() else 0.0


@evaluator
async def async_quality_check(query: str, response: str) -> dict:
    """Async check that can call external services."""
    passed = len(response) > 10  # placeholder logic
    return {"passed": passed, "reason": f"response length {len(response)}"}


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Be concise and accurate.")
    local = LocalEvaluator(not_too_long, ground_truth_match, async_quality_check)

    results = await evaluate_agent(
        agent=agent,
        queries=["What is the capital of France?"],
        expected_output="Paris",
        evaluators=local,
    )
    results[0].raise_for_status()


asyncio.run(main())
```

---

## 4. `WorkflowEvent` — Complete Factory Catalog

**Module:** `agent_framework._workflows._events`  
**Import:** `from agent_framework import WorkflowEvent`

`WorkflowEvent[DataT]` is the single generic event class for all workflow emissions. The `type` field acts
as a discriminator; `DataT` is the payload type. Use the factory class methods rather than the constructor
directly.

**All 12 factory methods:**

| Factory | Event type | Key fields |
|---|---|---|
| `WorkflowEvent.started(data=None)` | `"started"` | — |
| `WorkflowEvent.status(state, data=None)` | `"status"` | `.state: WorkflowRunState` |
| `WorkflowEvent.failed(details, data=None)` | `"failed"` | `.details: WorkflowErrorDetails` |
| `WorkflowEvent.warning(message)` | `"warning"` | `.data: str` |
| `WorkflowEvent.error(exception)` | `"error"` | `.data: Exception` |
| `WorkflowEvent.request_info(request_id, source_executor_id, request_data, response_type)` | `"request_info"` | `.request_id`, `.source_executor_id`, `.request_type`, `.response_type` |
| `WorkflowEvent.superstep_started(iteration, data=None)` | `"superstep_started"` | `.iteration: int` |
| `WorkflowEvent.superstep_completed(iteration, data=None)` | `"superstep_completed"` | `.iteration: int` |
| `WorkflowEvent.executor_invoked(executor_id, data=None)` | `"executor_invoked"` | `.executor_id: str` |
| `WorkflowEvent.executor_completed(executor_id, data=None)` | `"executor_completed"` | `.executor_id: str` |
| `WorkflowEvent.executor_failed(executor_id, details)` | `"executor_failed"` | `.executor_id`, `.details` |
| `WorkflowEvent.executor_bypassed(executor_id, data=None)` | `"executor_bypassed"` | `.executor_id: str` |

The deprecated `WorkflowEvent.emit(executor_id, data)` creates a `"data"` event (alias for `"intermediate"`).
Avoid it in new code; use `ctx.yield_output()` instead.

**`AGENT_FORWARDED_EVENT_TYPES`** is a `frozenset[str]` containing `{"output", "intermediate", "data", "request_info"}`.
When a workflow is used as an agent via `workflow.as_agent()`, only these event types cross the workflow boundary
to the outer caller; all lifecycle and orchestration events stay internal.

**`WorkflowEventSource`** is a `str` Enum: `FRAMEWORK` or `EXECUTOR`. Every event records its `origin`
automatically from a `ContextVar`; framework-owned code temporarily sets `FRAMEWORK` via the
`_framework_event_origin()` context manager.

### Example 1 — Consuming events from `workflow.run_stream()`

```python
import asyncio
from agent_framework import WorkflowEvent, WorkflowRunState, Workflow, AgentExecutor, Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), name="summarizer", instructions="Summarise text.")
    workflow = Workflow(executors=[AgentExecutor(agent, "summarizer")])

    async for event in workflow.run_stream("Summarise the history of Python."):
        if event.type == "started":
            print("Workflow started")
        elif event.type == "status":
            print(f"State → {event.state.value}")
        elif event.type == "output":
            print(f"Final output from {event.executor_id}: {event.data}")
        elif event.type == "executor_invoked":
            print(f"  [{event.executor_id}] invoked")
        elif event.type == "executor_completed":
            print(f"  [{event.executor_id}] completed")
        elif event.type == "superstep_started":
            print(f"  Superstep {event.iteration} started")
        elif event.type == "failed":
            print(f"Workflow FAILED: {event.details.message}")


asyncio.run(main())
```

### Example 2 — Inspecting `request_info` events (HITL)

```python
import asyncio
from agent_framework import WorkflowEvent
from my_workflow import build_hitl_workflow  # hypothetical


async def handle_requests(workflow, query: str) -> None:
    responses: dict[str, str] = {}

    async for event in workflow.run_stream(query):
        if event.type == "request_info":
            # Type-safe properties only available on request_info events
            rid = event.request_id
            executor = event.source_executor_id
            payload = event.data
            resp_type = event.response_type

            print(f"Request from {executor}: {payload!r} (expects {resp_type.__name__})")
            # Collect human response
            answer = input(f"Answer for request {rid}: ")
            responses[rid] = answer

        elif event.type == "output":
            print(f"Final answer: {event.data}")


asyncio.run(handle_requests(build_hitl_workflow(), "What is my account balance?"))
```

### Example 3 — Forwarding events across the `as_agent()` boundary

```python
import asyncio
from agent_framework import Workflow, AgentExecutor, Agent, WorkflowEvent, AGENT_FORWARDED_EVENT_TYPES
from agent_framework.openai import OpenAIChatClient


async def demonstrate_forwarded_events() -> None:
    inner_agent = Agent(client=OpenAIChatClient(), name="inner", instructions="Do inner work.")
    workflow = Workflow(executors=[AgentExecutor(inner_agent, "inner")])

    # as_agent() wraps the workflow as an agent; only forwarded types cross the boundary
    outer_agent = workflow.as_agent()

    print("Forwarded event types:", AGENT_FORWARDED_EVENT_TYPES)
    # {"output", "intermediate", "data", "request_info"}

    response = await outer_agent.run("Process this input")
    print(response.text)


asyncio.run(demonstrate_forwarded_events())
```

---

## 5. `WorkflowErrorDetails` + `WorkflowRunState` + `WorkflowEventSource`

**Module:** `agent_framework._workflows._events`  
**Imports:** `from agent_framework import WorkflowErrorDetails, WorkflowRunState, WorkflowEventSource`

`WorkflowErrorDetails` is a `dataclass` that captures structured error information for `"failed"` and
`"executor_failed"` events.

```python
@dataclass
class WorkflowErrorDetails:
    error_type: str        # exception class name
    message: str           # str(exc)
    traceback: str | None  # formatted traceback
    executor_id: str | None
    extra: dict[str, Any] | None
```

Create from a live exception with `WorkflowErrorDetails.from_exception(exc, executor_id=..., extra=...)`.

`WorkflowRunState` is the lifecycle state machine for a workflow run:

| State | Meaning |
|---|---|
| `STARTED` | Run begun |
| `IN_PROGRESS` | Executors running |
| `IN_PROGRESS_PENDING_REQUESTS` | Running with outstanding `request_info` |
| `IDLE` | No executors active, waiting for external input |
| `IDLE_WITH_PENDING_REQUESTS` | Idle with unanswered `request_info` |
| `FAILED` | Terminal error |
| `CANCELLED` | Cancelled |

`WorkflowEventSource` has two values: `FRAMEWORK` (internal orchestration) and `EXECUTOR` (user code). Every
event exposes `.origin` so you can distinguish framework noise from user-emitted events.

### Example 1 — Creating and inspecting `WorkflowErrorDetails`

```python
from agent_framework import WorkflowErrorDetails


def risky_operation() -> None:
    raise ValueError("Something went wrong in executor logic")


try:
    risky_operation()
except Exception as exc:
    details = WorkflowErrorDetails.from_exception(
        exc,
        executor_id="my_executor",
        extra={"context": "post-processing step"},
    )
    print(f"type:       {details.error_type}")
    print(f"message:    {details.message}")
    print(f"executor:   {details.executor_id}")
    print(f"traceback:  {details.traceback[:80] if details.traceback else None}")
    print(f"extra:      {details.extra}")
```

### Example 2 — Filtering events by origin

```python
import asyncio
from agent_framework import WorkflowEventSource, Workflow, AgentExecutor, Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), name="worker", instructions="Process tasks.")
    workflow = Workflow(executors=[AgentExecutor(agent, "worker")])

    async for event in workflow.run_stream("Do some work"):
        if event.origin == WorkflowEventSource.EXECUTOR:
            # Only surface events from user-supplied executor code
            print(f"Executor event: {event.type} from {event.executor_id}")
        # Silently skip framework-internal bookkeeping events


asyncio.run(main())
```

### Example 3 — Monitoring workflow state transitions

```python
import asyncio
from agent_framework import WorkflowRunState, Workflow, AgentExecutor, Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), name="worker", instructions="Process tasks.")
    workflow = Workflow(executors=[AgentExecutor(agent, "worker")])

    states_seen = []

    async for event in workflow.run_stream("Execute this task"):
        if event.type == "status":
            state = event.state
            states_seen.append(state.value)
            if state == WorkflowRunState.FAILED:
                print(f"Workflow failed: {event.details}")
                break
            elif state in (WorkflowRunState.IDLE, WorkflowRunState.IDLE_WITH_PENDING_REQUESTS):
                print("Workflow paused — external input may be required")

    print("States seen:", states_seen)


asyncio.run(main())
```

---

## 6. Compaction Annotation Constants + `included_messages` + `included_token_count`

**Module:** `agent_framework._compaction`  
**Imports:** `from agent_framework import (GROUP_ANNOTATION_KEY, EXCLUDED_KEY, EXCLUDE_REASON_KEY, GROUP_ID_KEY, GROUP_KIND_KEY, GROUP_INDEX_KEY, GROUP_TOKEN_COUNT_KEY, SUMMARIZED_BY_SUMMARY_ID_KEY, SUMMARY_OF_GROUP_IDS_KEY, SUMMARY_OF_MESSAGE_IDS_KEY, COMPACTION_STATE_KEY, included_messages, included_token_count)`

Compaction strategies annotate messages in-place via `message.additional_properties`. These constants are
the canonical keys for reading and writing those annotations. They are used by built-in strategies
(`ContextWindowCompactionStrategy`, `LastNMessagesCompactionStrategy`) and are essential when writing
custom compaction strategies.

| Constant | Value | Purpose |
|---|---|---|
| `GROUP_ANNOTATION_KEY` | `"_group"` | Dict holding group metadata on each message |
| `GROUP_ID_KEY` | `"id"` | UUID string identifying the message's conversation group |
| `GROUP_KIND_KEY` | `"kind"` | One of `"system"`, `"user"`, `"assistant_text"`, `"tool_call"` |
| `GROUP_INDEX_KEY` | `"index"` | Position of this group in the sequence |
| `GROUP_TOKEN_COUNT_KEY` | `"token_count"` | Estimated token count for the group |
| `EXCLUDED_KEY` | `"_excluded"` | `True` when message is excluded by compaction |
| `EXCLUDE_REASON_KEY` | `"_exclude_reason"` | Human-readable reason for exclusion |
| `SUMMARY_OF_MESSAGE_IDS_KEY` | `"_summary_of_message_ids"` | IDs of messages this summary replaces |
| `SUMMARY_OF_GROUP_IDS_KEY` | `"_summary_of_group_ids"` | Group IDs this summary replaces |
| `SUMMARIZED_BY_SUMMARY_ID_KEY` | `"_summarized_by_summary_id"` | ID of the summary that replaced this message |
| `COMPACTION_STATE_KEY` | `"_compaction_messages"` | Session state key for persisted compaction state |

`included_messages(messages)` returns only messages where `_excluded` is falsy.  
`included_token_count(messages)` sums the `token_count` annotation of all included messages.

### Example 1 — Reading compaction annotations from a session

```python
import asyncio
from agent_framework import (
    Agent,
    EXCLUDED_KEY,
    EXCLUDE_REASON_KEY,
    GROUP_ANNOTATION_KEY,
    GROUP_KIND_KEY,
    GROUP_TOKEN_COUNT_KEY,
    included_messages,
    included_token_count,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework import CompactionProvider, ContextWindowCompactionStrategy


async def main() -> None:
    strategy = ContextWindowCompactionStrategy(max_context_window_tokens=4096, max_output_tokens=512)
    compaction = CompactionProvider(before_strategy=strategy)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=[compaction],
    )

    session = agent.create_session()
    for _ in range(5):
        await agent.run("Tell me something interesting", session=session)

    # Access the compaction messages from session state
    from agent_framework import COMPACTION_STATE_KEY
    compaction_msgs = session.state.get(COMPACTION_STATE_KEY, [])

    for msg in compaction_msgs:
        group = msg.additional_properties.get(GROUP_ANNOTATION_KEY, {})
        excluded = msg.additional_properties.get(EXCLUDED_KEY, False)
        reason = msg.additional_properties.get(EXCLUDE_REASON_KEY)
        kind = group.get(GROUP_KIND_KEY, "unknown")
        tokens = group.get(GROUP_TOKEN_COUNT_KEY, 0)
        status = f"EXCLUDED ({reason})" if excluded else "included"
        print(f"  [{kind}] tokens={tokens} status={status}")

    print(f"Included count: {len(included_messages(compaction_msgs))}")
    print(f"Included tokens: {included_token_count(compaction_msgs)}")


asyncio.run(main())
```

### Example 2 — Custom compaction strategy using annotation constants

```python
import asyncio
from agent_framework import (
    EXCLUDED_KEY,
    EXCLUDE_REASON_KEY,
    GROUP_ANNOTATION_KEY,
    GROUP_KIND_KEY,
    included_messages,
)
from agent_framework._types import Message


async def system_only_compaction(messages: list[Message]) -> bool:
    """Keep only system messages and the last 3 user/assistant messages."""
    non_system = [
        m for m in messages
        if not m.additional_properties.get(EXCLUDED_KEY, False)
        and m.additional_properties.get(GROUP_ANNOTATION_KEY, {}).get(GROUP_KIND_KEY) != "system"
    ]

    to_exclude = non_system[:-3]  # drop everything except the last 3 turns
    changed = False
    for msg in to_exclude:
        if not msg.additional_properties.get(EXCLUDED_KEY, False):
            msg.additional_properties[EXCLUDED_KEY] = True
            msg.additional_properties[EXCLUDE_REASON_KEY] = "custom_truncation"
            changed = True

    return changed


# Use it with CompactionProvider
from agent_framework import CompactionProvider, Agent
from agent_framework.openai import OpenAIChatClient

compaction = CompactionProvider(before_strategy=system_only_compaction)
agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are helpful.",
    context_providers=[compaction],
)
```

### Example 3 — Checking whether compaction ran during a session

```python
import asyncio
from agent_framework import (
    Agent,
    CompactionProvider,
    ContextWindowCompactionStrategy,
    COMPACTION_STATE_KEY,
    EXCLUDED_KEY,
    included_messages,
    included_token_count,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    strategy = ContextWindowCompactionStrategy(max_context_window_tokens=2048, max_output_tokens=512)
    compaction = CompactionProvider(before_strategy=strategy)
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=[compaction],
    )

    session = agent.create_session()
    for prompt in ["Explain quantum computing", "What are qubits?", "How does entanglement work?"]:
        await agent.run(prompt, session=session)

    msgs = session.state.get(COMPACTION_STATE_KEY, [])
    total = len(msgs)
    included = len(included_messages(msgs))
    excluded = total - included

    print(f"Total messages tracked: {total}")
    print(f"Currently included:     {included}")
    print(f"Compacted away:         {excluded}")
    print(f"Estimated token usage:  {included_token_count(msgs)}")


asyncio.run(main())
```

---

## 7. `background_tasks_running_message` + `todos_remaining_message`

**Module:** `agent_framework._harness._loop`  
**Imports:** `from agent_framework import background_tasks_running_message, todos_remaining_message`

These are `next_message` callables designed to pair with the loop predicates `background_tasks_running()`
and `todos_remaining()` respectively (covered in Vol. 31). They return a `str | None`:

- Return a formatted reminder message when there is outstanding work, prompting the agent to finish before stopping.
- Return `None` when no work remains (or the agent/session is unavailable).

The signature for both follows the `NextMessageCallable` pattern:
```python
def background_tasks_running_message(*, session=None, agent=None, **kwargs) -> str | None: ...
async def todos_remaining_message(*, session=None, agent=None, **kwargs) -> str | None: ...
```

Both work by resolving their respective provider (`BackgroundAgentsProvider` / `TodoProvider`) from
`agent.context_providers` at call time — no constructor injection required.

### Example 1 — Wiring `todos_remaining_message` with `create_harness_agent`

```python
import asyncio
from agent_framework import (
    TodoProvider,
    TodoSessionStore,
    create_harness_agent,
    todos_remaining,
    todos_remaining_message,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    harness_agent = create_harness_agent(
        OpenAIChatClient(),
        agent_instructions="Complete all assigned tasks.",
        todo_provider=TodoProvider(store=TodoSessionStore()),
        loop_should_continue=todos_remaining(),
        loop_next_message=todos_remaining_message,  # function reference, not called
        loop_max_iterations=10,
    )

    session = harness_agent.create_session()
    response = await harness_agent.run(
        "Please add a todo for 'Write unit tests' and complete it.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 2 — Wiring `background_tasks_running_message` with `AgentLoopMiddleware`

```python
import asyncio
from agent_framework import (
    Agent,
    BackgroundAgentsProvider,
    AgentLoopMiddleware,
    background_tasks_running,
    background_tasks_running_message,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    worker = Agent(
        client=OpenAIChatClient(),
        name="worker",
        instructions="Perform delegated analysis tasks.",
    )
    main_agent = Agent(
        client=OpenAIChatClient(),
        instructions="Delegate analysis to worker agents and wait for results.",
        context_providers=[BackgroundAgentsProvider(agents=[worker])],
        middleware=[
            AgentLoopMiddleware(
                should_continue=background_tasks_running(),
                next_message=background_tasks_running_message,
                max_iterations=20,
            )
        ],
    )

    session = main_agent.create_session()
    response = await main_agent.run(
        "Start a background task to analyse the Q3 sales data and report back.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 3 — Combining both messages with a custom loop predicate

```python
import asyncio
from agent_framework import (
    Agent,
    BackgroundAgentsProvider,
    TodoProvider,
    TodoSessionStore,
    AgentLoopMiddleware,
    background_tasks_running,
    todos_remaining,
    background_tasks_running_message,
    todos_remaining_message,
)
from agent_framework.openai import OpenAIChatClient


async def combined_next_message(*, session=None, agent=None, **kwargs) -> str | None:
    """Return whichever message is most relevant, preferring tasks over todos."""
    bg_msg = background_tasks_running_message(session=session, agent=agent)
    if bg_msg:
        return bg_msg
    return await todos_remaining_message(session=session, agent=agent)


async def combined_should_continue(*, session=None, agent=None, **kwargs) -> bool:
    bg_running = background_tasks_running()(session=session, agent=agent)
    todos = await todos_remaining()(session=session, agent=agent)
    return bg_running or todos


async def main() -> None:
    worker = Agent(client=OpenAIChatClient(), name="researcher", instructions="Research topics.")
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Research topics and track todos.",
        context_providers=[
            BackgroundAgentsProvider(agents=[worker]),
            TodoProvider(store=TodoSessionStore()),
        ],
        middleware=[
            AgentLoopMiddleware(
                should_continue=combined_should_continue,
                next_message=combined_next_message,
                max_iterations=15,
            )
        ],
    )
    session = agent.create_session()
    response = await agent.run("Research Python and asyncio, tracking your progress with todos.", session=session)
    print(response.text)


asyncio.run(main())
```

---

## 8. `ToolApprovalRuleCallback` + `create_always_approve_tool_response` + `create_always_approve_tool_with_arguments_response`

**Module:** `agent_framework._harness._tool_approval`  
**Imports:** `from agent_framework import ToolApprovalRuleCallback, create_always_approve_tool_response, create_always_approve_tool_with_arguments_response`

`ToolApprovalRuleCallback` is a type alias:

```python
ToolApprovalRuleCallback = Callable[[Content], bool | Awaitable[bool]]
```

It is passed to `ToolApprovalMiddleware` (or `create_harness_agent` via `auto_approval_rules`) as a
programmatic approval predicate. The callback receives the `function_call` `Content` and returns `True` to
auto-approve without user prompting.

The two factory functions create persistent standing-rule approval responses that `ToolApprovalMiddleware`
persists in session state so the approval applies to all future matching calls:

- `create_always_approve_tool_response(request)` — approves the whole tool (by name, regardless of arguments).
- `create_always_approve_tool_with_arguments_response(request)` — approves the tool with this exact argument combination.

Both accept an optional `reason` keyword for auditing.

### Example 1 — Auto-approval callback based on tool name

```python
import asyncio
from agent_framework import (
    Agent,
    ToolApprovalMiddleware,
)
from agent_framework._types import Content
from agent_framework.openai import OpenAIChatClient


def safe_tools_auto_approve(function_call: Content) -> bool:
    """Auto-approve read-only tools; require approval for mutations."""
    safe = {"search_web", "get_weather", "read_file", "list_files"}
    return (function_call.name or "") in safe


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Research topics using available tools.",
        middleware=[
            ToolApprovalMiddleware(
                auto_approval_rules=[safe_tools_auto_approve],
            )
        ],
    )

    session = agent.create_session()
    response = await agent.run("Search for information about Python 3.13 features.", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 2 — Creating standing approval rules from `request_info` events

When `ToolApprovalMiddleware` requires approval it emits a `"request_info"` event. The caller creates
a response `Content` using one of the factory functions to record a standing rule.

```python
import asyncio
from agent_framework import (
    Agent,
    ToolApprovalMiddleware,
    create_always_approve_tool_response,
    create_always_approve_tool_with_arguments_response,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Use tools to complete work.",
        middleware=[ToolApprovalMiddleware()],
    )

    session = agent.create_session()

    async for event in agent.run_stream("Analyse and write the monthly report.", session=session):
        if event.type == "request_info":
            request_content = event.data  # the function_approval_request Content

            tool_name = request_content.function_call.name if request_content.function_call else "unknown"
            print(f"Approval needed for tool: {tool_name}")

            # For low-risk read operations: approve whole tool permanently
            if tool_name in ("read_file", "list_files"):
                response = create_always_approve_tool_response(
                    request_content,
                    reason="read-only tool — always safe",
                )
            else:
                # For specific argument combinations: approve this call and identical future calls
                response = create_always_approve_tool_with_arguments_response(
                    request_content,
                    reason="user approved this exact invocation",
                )

            await agent.provide_response(event.request_id, response, session=session)


asyncio.run(main())
```

### Example 3 — Async approval callback using `ToolApprovalRuleCallback`

```python
import asyncio
from agent_framework import Agent, ToolApprovalMiddleware
from agent_framework._types import Content
from agent_framework.openai import OpenAIChatClient


async def policy_check_auto_approve(function_call: Content) -> bool:
    """Async callback: check a policy service before auto-approving."""
    tool_name = function_call.name or ""
    # Simulate async policy lookup
    await asyncio.sleep(0)  # placeholder for real async call
    allowed_tools = {"get_data", "compute_stats", "format_report"}
    return tool_name in allowed_tools


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Process data using approved tools.",
        middleware=[
            ToolApprovalMiddleware(
                auto_approval_rules=[policy_check_auto_approve],  # async callback supported
            )
        ],
    )

    session = agent.create_session()
    response = await agent.run("Compute statistics for the uploaded dataset.", session=session)
    print(response.text)


asyncio.run(main())
```

---

## 9. `FileStoreEntry` + `DEFAULT_FILE_ACCESS_INSTRUCTIONS` + `DEFAULT_FILE_ACCESS_SOURCE_ID`

**Module:** `agent_framework._harness._file_access`  
**Imports:** `from agent_framework import FileStoreEntry, DEFAULT_FILE_ACCESS_INSTRUCTIONS, DEFAULT_FILE_ACCESS_SOURCE_ID`

`FileStoreEntry` represents one entry in a directory listing from any `AgentFileStore` implementation.
It is returned by `AgentFileStore.list_children()` and by the `file_access_ls` tool.

```python
class FileStoreEntry:
    FILE: ClassVar[str] = "file"
    DIRECTORY: ClassVar[str] = "directory"

    name: str   # entry name (not full path), relative to the listed directory
    type: str   # "file" or "directory"
```

`DEFAULT_FILE_ACCESS_SOURCE_ID = "file_access"` is the default `source_id` for `FileAccessProvider`.  
`DEFAULT_FILE_ACCESS_INSTRUCTIONS` is the multi-line string injected as instructions when
`FileAccessProvider` is attached. It describes the `file_access_*` tool set and usage guidelines.

### Example 1 — Iterating directory listings with `FileStoreEntry`

```python
import asyncio
from agent_framework import (
    FileAccessProvider,
    FileSystemAgentFileStore,
    FileStoreEntry,
)


async def main() -> None:
    store = FileSystemAgentFileStore("./data")

    # List root
    entries: list[FileStoreEntry] = await store.list_children()
    for entry in entries:
        if entry.type == FileStoreEntry.DIRECTORY:
            print(f"[DIR]  {entry.name}")
        elif entry.type == FileStoreEntry.FILE:
            print(f"[FILE] {entry.name}")

    # Recurse into a subdirectory
    sub_entries = await store.list_children("reports")
    file_names = [e.name for e in sub_entries if e.type == FileStoreEntry.FILE]
    print("Report files:", file_names)


asyncio.run(main())
```

### Example 2 — Customising `FileAccessProvider` instructions

```python
import asyncio
from agent_framework import (
    Agent,
    FileAccessProvider,
    FileSystemAgentFileStore,
    DEFAULT_FILE_ACCESS_INSTRUCTIONS,
    DEFAULT_FILE_ACCESS_SOURCE_ID,
)
from agent_framework.openai import OpenAIChatClient

CUSTOM_INSTRUCTIONS = DEFAULT_FILE_ACCESS_INSTRUCTIONS + (
    "\n\n## Project conventions\n"
    "- Store all output reports under the `reports/` subdirectory.\n"
    "- Never modify files in `archive/`.\n"
    "- Prefix draft files with `draft_`."
)

async def main() -> None:
    store = FileSystemAgentFileStore("./workspace")
    file_access = FileAccessProvider(
        store=store,
        source_id=DEFAULT_FILE_ACCESS_SOURCE_ID,
        instructions=CUSTOM_INSTRUCTIONS,
    )
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Manage project files according to conventions.",
        context_providers=[file_access],
    )

    session = agent.create_session()
    response = await agent.run("List all files in the reports directory.", session=session)
    print(response.text)


asyncio.run(main())
```

### Example 3 — Filtering directory entries by type

```python
import asyncio
from agent_framework import FileSystemAgentFileStore, FileStoreEntry


async def walk_files_only(store: FileSystemAgentFileStore, directory: str = "") -> list[str]:
    """Recursively collect all file paths under a directory."""
    file_paths: list[str] = []
    entries: list[FileStoreEntry] = await store.list_children(directory)

    for entry in entries:
        path = f"{directory}/{entry.name}".lstrip("/")
        if entry.type == FileStoreEntry.FILE:
            file_paths.append(path)
        elif entry.type == FileStoreEntry.DIRECTORY:
            sub = await walk_files_only(store, path)
            file_paths.extend(sub)

    return file_paths


async def main() -> None:
    store = FileSystemAgentFileStore("./project")
    all_files = await walk_files_only(store)
    print(f"Found {len(all_files)} file(s):")
    for f in all_files:
        print(f"  {f}")


asyncio.run(main())
```

---

## 10. `BackgroundAgentsProvider` + `BackgroundTaskInfo` + `BackgroundTaskStatus` + `_RuntimeState`

**Module:** `agent_framework._harness._background_agents`  
**Imports:** `from agent_framework import BackgroundAgentsProvider, BackgroundTaskInfo, BackgroundTaskStatus`

`BackgroundAgentsProvider` enables a parent agent to delegate work to child agents running concurrently in
independent sessions. `BackgroundTaskStatus` is a `str` Enum with four values:

| Status | Meaning |
|---|---|
| `RUNNING` | Task is executing |
| `COMPLETED` | Task finished successfully (`result_text` available) |
| `FAILED` | Task raised an exception (`error_text` available) |
| `LOST` | Provider instance was replaced (in-flight asyncio task no longer reachable) |

`BackgroundTaskInfo` holds all serializable metadata for a single task. It uses `SerializationMixin`
and is round-tripped via `to_dict()` / `from_dict()` into session state so task status survives across
turns.

`_RuntimeState` is the non-serializable per-session complement: it holds the live `asyncio.Task` objects
(`in_flight_tasks`) and `AgentSession` handles (`background_sessions`) keyed by task ID. When the
provider instance is lost (e.g. process restart), `_RuntimeState` is gone and tasks are marked `LOST`.

The six registered tools and their semantics:

| Tool | Description |
|---|---|
| `background_agents_start_task` | Start task (sync; schedules `asyncio.create_task`) |
| `background_agents_wait_for_first_completion` | Await `asyncio.wait(FIRST_COMPLETED)` |
| `background_agents_get_task_results` | Return `result_text` or `error_text` |
| `background_agents_get_all_tasks` | List all tasks with status |
| `background_agents_continue_task` | Resume on the same session with new input |
| `background_agents_clear_completed_task` | Remove task and release session |

### Example 1 — Fan-out to multiple background agents

```python
import asyncio
from agent_framework import (
    Agent,
    BackgroundAgentsProvider,
    create_harness_agent,
    background_tasks_running,
    background_tasks_running_message,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    # Three specialist sub-agents
    researchers = [
        Agent(client=OpenAIChatClient(), name="science-agent", instructions="Research scientific topics."),
        Agent(client=OpenAIChatClient(), name="history-agent", instructions="Research historical topics."),
        Agent(client=OpenAIChatClient(), name="tech-agent", instructions="Research technology topics."),
    ]

    orchestrator = create_harness_agent(
        agent=Agent(
            client=OpenAIChatClient(),
            instructions="Delegate research across specialist agents, then synthesise results.",
        ),
        context_providers=[BackgroundAgentsProvider(agents=researchers)],
        loop_should_continue=background_tasks_running(),
        loop_next_message=background_tasks_running_message,
        loop_max_iterations=15,
    )

    session = orchestrator.create_session()
    response = await orchestrator.run(
        "Research quantum computing from science, history, and tech perspectives in parallel.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 2 — Continue a completed task for follow-up work

```python
import asyncio
from agent_framework import (
    Agent,
    BackgroundAgentsProvider,
    BackgroundTaskStatus,
    AgentLoopMiddleware,
    background_tasks_running,
    background_tasks_running_message,
)
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    analyst = Agent(
        client=OpenAIChatClient(),
        name="analyst",
        description="Data analysis specialist",
        instructions="Analyse data and answer follow-up questions.",
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions=(
            "Start a background analysis task, retrieve the result, then ask a follow-up "
            "question using background_agents_continue_task on the same session."
        ),
        context_providers=[BackgroundAgentsProvider(agents=[analyst])],
        middleware=[
            AgentLoopMiddleware(
                should_continue=background_tasks_running(),
                next_message=background_tasks_running_message,
                max_iterations=10,
            )
        ],
    )

    session = agent.create_session()
    response = await agent.run(
        "Analyse the Q3 sales dataset. Once done, ask the analyst for trend predictions.",
        session=session,
    )
    print(response.text)


asyncio.run(main())
```

### Example 3 — Inspecting `BackgroundTaskInfo` state directly

```python
import asyncio
from agent_framework import (
    Agent,
    BackgroundAgentsProvider,
    BackgroundTaskInfo,
    BackgroundTaskStatus,
)
from agent_framework.openai import OpenAIChatClient

DEFAULT_BACKGROUND_AGENTS_SOURCE_ID = "background_agents"


async def print_task_summary(session, source_id: str = DEFAULT_BACKGROUND_AGENTS_SOURCE_ID) -> None:
    """Print a summary of all background tasks from session state."""
    state = session.state.get(source_id, {})
    raw_tasks = state.get("tasks", [])
    tasks = [BackgroundTaskInfo.from_dict(t) for t in raw_tasks]

    if not tasks:
        print("No background tasks recorded.")
        return

    for task in tasks:
        print(f"Task {task.id} [{task.status.value}] — {task.agent_name}: {task.description}")
        if task.status == BackgroundTaskStatus.COMPLETED:
            preview = (task.result_text or "")[:100]
            print(f"  Result: {preview}...")
        elif task.status == BackgroundTaskStatus.FAILED:
            print(f"  Error: {task.error_text}")
        elif task.status == BackgroundTaskStatus.LOST:
            print("  (session reference lost — restart provider)")


async def main() -> None:
    worker = Agent(client=OpenAIChatClient(), name="worker", instructions="Process requests.")
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Start a background task and report when done.",
        context_providers=[BackgroundAgentsProvider(agents=[worker])],
    )

    session = agent.create_session()
    await agent.run("Start a background task to summarise the Python docs.", session=session)

    # Inspect persisted task metadata
    await print_task_summary(session)


asyncio.run(main())
```

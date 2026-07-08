---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 34"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.10.0: OpenAIChatCompletionOptions+Prediction+PredictionTextContent+RawOpenAIChatCompletionClient+OpenAIChatCompletionClient (Chat Completions API); ReasoningOptions+StreamOptions+OpenAIContinuationToken (reasoning controls for Responses API); OpenAIContentFilterException+ContentFilterResult+ContentFilterResultSeverity+ContentFilterCodes (Azure content filter hierarchy); WorkflowOrchestrationContext (host-agnostic durable protocol); DurableWorkflowClient (start/stream/HITL durable workflows); WorkflowRegistrationPlan+plan_workflow_registration (entity vs activity classification); TaskType+TaskMetadata+PendingHITLRequest+ExecutorResult (orchestrator data structures); DurableTaskWorkflowContext (standalone durabletask adapter); ConversationStore+InMemoryConversationStore+CheckpointConversationManager (DevUI conversation store); AgentTask+PreCompletedTask+AzureFunctionsAgentExecutor (Azure Functions durable orchestration)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 57
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 34

Source-verified against `agent-framework==1.10.0` · Python 3.10 – 3.13 · July 2026

Ten class groups with three runnable examples each (30 total). Every constructor
signature, constant value, and method name was read directly from the installed
`agent-framework==1.10.0` packages (`agent_framework`, `agent_framework_openai`,
`agent_framework_durabletask`, `agent_framework_devui`, `agent_framework_azurefunctions`).

---

## 1 · `OpenAIChatCompletionOptions` + `Prediction` + `PredictionTextContent` + `RawOpenAIChatCompletionClient` + `OpenAIChatCompletionClient`

**Module:** `agent_framework_openai._chat_completion_client`  
**Import:** `from agent_framework.openai import OpenAIChatCompletionClient`

The **Chat Completions API** client complements `OpenAIChatClient` (Responses API, Vol. 13).
Use it for `gpt-4o`, `gpt-4.1`, and legacy models that do not support the stateful Responses
API surface. The four-layer MRO is identical:
`FunctionInvocationLayer → ChatMiddlewareLayer → ChatTelemetryLayer → RawOpenAIChatCompletionClient`.

`OpenAIChatCompletionOptions` extends the shared `ChatOptions` TypedDict with Completions-only
fields:

| Key | Type | Completions-specific? |
|-----|------|-----------------------|
| `logprobs` | `bool` | ✓ |
| `top_logprobs` | `int` (0–20) | ✓ |
| `prediction` | `Prediction` | ✓ |
| `store` | `bool` | ✓ |
| `web_search_options` | `WebSearchOptions` | ✓ |
| `stream_options` | `StreamOptions` | ✓ |
| `reasoning_effort` | `str` | shared |

`Prediction` enables OpenAI's speculative-decoding feature: supply known output text so the
API can reuse cached tokens when regenerating similar content.

**Example 1 — Basic Chat Completions client with logprobs**

```python
import asyncio
from agent_framework import Message
from agent_framework.openai import OpenAIChatCompletionClient

async def main() -> None:
    client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key="sk-...",           # or set OPENAI_API_KEY env var
    )
    response = await client.get_response(
        messages=[Message("user", ["Name the planets in order."])],
        options={
            "logprobs": True,
            "top_logprobs": 3,
            "max_tokens": 200,
        },
    )
    print(response.text)
    # logprobs are available on response.raw_representation if needed

asyncio.run(main())
```

**Example 2 — Prediction (speculative decoding) to accelerate refactoring**

```python
import asyncio
from agent_framework import Message
from agent_framework.openai import OpenAIChatCompletionClient

ORIGINAL = "def greet(name):\n    print('Hello, ' + name)"

async def main() -> None:
    client = OpenAIChatCompletionClient(model="gpt-4o")
    # Predict the output will look like the original but type-annotated.
    prediction: dict = {
        "type": "content",
        "content": "def greet(name: str) -> None:\n    print('Hello, ' + name)",
    }
    response = await client.get_response(
        messages=[Message("user", [f"Add a type annotation to:\n{ORIGINAL}"])],
        options={"prediction": prediction},
    )
    print(response.text)

asyncio.run(main())
```

**Example 3 — Azure OpenAI Chat Completions with web search**

```python
import asyncio
from agent_framework import Message
from agent_framework.openai import OpenAIChatCompletionClient

async def main() -> None:
    client = OpenAIChatCompletionClient(
        model="gpt-4o",
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_key="my-azure-key",
        api_version="2024-12-01-preview",
    )
    response = await client.get_response(
        messages=[Message("user", ["What happened in AI news today?"])],
        options={"web_search_options": {"search_context_size": "medium"}},
    )
    print(response.text)
    if response.usage_details:
        # Non-streaming responses include usage directly — no stream_options needed
        print(f"Tokens: {response.usage_details.get('total_token_count')}")

asyncio.run(main())
```

---

## 2 · `ReasoningOptions` + `StreamOptions` + `OpenAIContinuationToken`

**Module:** `agent_framework_openai._chat_client`  
**Import:** `from agent_framework.openai import OpenAIChatClient`

These three TypedDicts / classes extend the Responses API surface documented in Vol. 13.

`ReasoningOptions` controls extended thinking for `o-series` and `gpt-5` models:

```python
class ReasoningOptions(TypedDict, total=False):
    effort: Literal["none", "low", "medium", "high", "xhigh"]
    summary: Literal["auto", "concise", "detailed"]
```

> **Note:** Reasoning token limits are controlled by the top-level `max_tokens` option on the request (not a field of `ReasoningOptions`). The client translates `max_tokens` → `max_output_tokens` for the Responses API. Pass it via `options={"max_tokens": N}` alongside the `"reasoning"` key.

`StreamOptions` gates usage-statistics inclusion in streaming events:

```python
class StreamOptions(TypedDict, total=False):
    include_usage: bool
```

`OpenAIContinuationToken` extends the base `ContinuationToken` with a `response_id` field
used for **background polling** — the Responses API can create a response asynchronously and
clients poll for its completion using the stored response ID.

**Example 1 — Reasoning effort control on an o-series model**

```python
import asyncio
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    client = OpenAIChatClient(model="o3")
    response = await client.get_response(
        messages=[Message("user", ["Prove the Pythagorean theorem."])],
        options={
            "reasoning": {"effort": "high", "summary": "detailed"},
        },
    )
    print(response.text)

asyncio.run(main())
```

**Example 2 — Streaming with usage statistics included**

```python
import asyncio
from agent_framework import Message, ResponseStream
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4.1")
    stream: ResponseStream = client.get_response(
        messages=[Message("user", ["Write a haiku about the ocean."])],
        options={"stream_options": {"include_usage": True}},
        stream=True,
    )
    async for update in stream:
        print(update.text or "", end="", flush=True)
    response = await stream.get_final_response()
    if response.usage_details:
        print(f"\nTotal tokens: {response.usage_details.get('total_token_count')}")

asyncio.run(main())
```

**Example 3 — Background response polling with `OpenAIContinuationToken`**

```python
import asyncio
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient, OpenAIContinuationToken

async def main() -> None:
    client = OpenAIChatClient(model="gpt-4.1")
    # First call: start the response; "store" persists it server-side for later polling
    response = await client.get_response(
        messages=[Message("user", ["Summarize the history of computing."])],
        options={"background": True, "store": True},
    )
    # OpenAIContinuationToken is a TypedDict — at runtime it is a plain dict.
    # isinstance() checks raise TypeError, so use key presence to guard access.
    token: OpenAIContinuationToken | None = response.continuation_token
    if token is not None and "response_id" in token:
        print(f"Response ID for polling: {token['response_id']}")
        # Poll later using the same client + token — pass the full token dict
        # under the "continuation_token" key; the client checks that key and
        # retrieves token["response_id"] to resume the background response.
        final = await client.get_response(
            messages=[],
            options={"continuation_token": token},
        )
        print(final.text)

asyncio.run(main())
```

---

## 3 · `OpenAIContentFilterException` + `ContentFilterResult` + `ContentFilterResultSeverity` + `ContentFilterCodes`

**Module:** `agent_framework_openai._exceptions`  
**Import:** `from agent_framework.openai import OpenAIContentFilterException`

When Azure OpenAI's content filter blocks a request, the framework raises
`OpenAIContentFilterException` (a subclass of `ChatClientContentFilterException`).
It carries the Azure error code (`ContentFilterCodes.RESPONSIBLE_AI_POLICY_VIOLATION`) and
a per-category dict of `ContentFilterResult` objects — each with a severity enum value.

```python
class ContentFilterResultSeverity(Enum):
    HIGH = "high"; MEDIUM = "medium"; LOW = "low"; SAFE = "safe"

@dataclass
class ContentFilterResult:
    filtered: bool
    severity: ContentFilterResultSeverity

class ContentFilterCodes(Enum):
    RESPONSIBLE_AI_POLICY_VIOLATION = "ResponsibleAIPolicyViolation"

class OpenAIContentFilterException(ChatClientContentFilterException):
    param: str | None
    content_filter_code: ContentFilterCodes
    content_filter_result: dict[str, ContentFilterResult]
```

**Example 1 — Catching and inspecting a content filter exception**

```python
import asyncio
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient, OpenAIContentFilterException, ContentFilterResultSeverity

async def main() -> None:
    client = OpenAIChatClient(
        model="gpt-4o",
        azure_endpoint="https://my-resource.openai.azure.com/",
        api_key="my-azure-key",
    )
    try:
        await client.get_response(
            messages=[Message("user", ["Generate detailed instructions for a dangerous activity."])],
        )
    except OpenAIContentFilterException as exc:
        print(f"Content filter triggered: {exc.content_filter_code.value}")
        for category, result in exc.content_filter_result.items():
            if result.filtered:
                print(f"  {category}: severity={result.severity.value}")

asyncio.run(main())
```

**Example 2 — Wrapping a chat client with a safe fallback on filter exceptions**

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient, OpenAIContentFilterException

SAFE_FALLBACK = "I'm sorry, I can't help with that."

async def safe_run(agent: Agent, prompt: str) -> str:
    try:
        response = await agent.run(prompt)
        return response.text
    except OpenAIContentFilterException as exc:
        blocked = [c for c, r in exc.content_filter_result.items() if r.filtered]
        return f"{SAFE_FALLBACK} (blocked categories: {', '.join(blocked)})"

async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(model="gpt-4o"),
        instructions="You are a helpful assistant.",
    )
    reply = await safe_run(agent, "Describe how to perform an activity that violates safety guidelines.")
    print(reply)

asyncio.run(main())
```

**Example 3 — Logging severity levels for all filter categories**

```python
import asyncio, logging
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient, OpenAIContentFilterException, ContentFilterResultSeverity

logger = logging.getLogger("content_filter_audit")

async def run_with_audit(prompt: str) -> str | None:
    client = OpenAIChatClient(model="gpt-4o")
    try:
        r = await client.get_response(
            messages=[Message("user", [prompt])],
        )
        return r.text
    except OpenAIContentFilterException as exc:
        for cat, result in exc.content_filter_result.items():
            if result.severity != ContentFilterResultSeverity.SAFE:
                logger.warning(
                    "filter=%s severity=%s filtered=%s",
                    cat, result.severity.value, result.filtered,
                )
        return None

result = asyncio.run(run_with_audit("Explain SQL injection."))
if result:
    print(result)
```

---

## 4 · `WorkflowOrchestrationContext`

**Module:** `agent_framework_durabletask._workflows.context`  
**Import:** `from agent_framework_durabletask._workflows.context import WorkflowOrchestrationContext`

`WorkflowOrchestrationContext` is a `@runtime_checkable` `Protocol` that abstracts the
differences between **Azure Functions** `DurableOrchestrationContext` and the **standalone**
`durabletask.task.OrchestrationContext`. The shared workflow orchestrator
(`run_workflow_orchestrator`) programmes only against this protocol, so the same graph
execution logic runs on any durable host.

Key properties and methods:

| Member | Purpose |
|--------|---------|
| `instance_id: str` | Unique orchestration run ID |
| `is_replaying: bool` | Skip side-effects during replay |
| `supports_event_streaming: bool` | Whether host can publish accumulating event log to custom status |
| `current_utc_datetime: datetime` | Replay-safe UTC clock |
| `prepare_agent_task(executor_id, message, orchestration_instance_id)` | Creates a yieldable task dispatching to a durable entity |
| `prepare_activity_task(activity_name, input_json)` | Creates a yieldable task dispatching to an activity function |

**Example 1 — Checking replay safety before logging**

```python
import json
# Inside a durable orchestration function registered with DurableAIAgentWorker
from agent_framework_durabletask._workflows.context import WorkflowOrchestrationContext

def my_orchestrator(ctx: WorkflowOrchestrationContext):
    # Only emit live-observable side effects (streaming status, external calls)
    # when NOT replaying, to avoid duplicate emissions.
    if not ctx.is_replaying:
        print(f"[{ctx.instance_id}] Starting at {ctx.current_utc_datetime.isoformat()}")

    task = ctx.prepare_agent_task(
        executor_id="summarizer",
        message="Summarize the Q4 report.",
        orchestration_instance_id=ctx.instance_id,
    )
    result = yield task
    yield ctx.prepare_activity_task("save_result", json.dumps({"text": str(result)}))
```

**Example 2 — Conditional event streaming based on host capability**

```python
from agent_framework_durabletask._workflows.context import WorkflowOrchestrationContext

def workflow_with_conditional_streaming(ctx: WorkflowOrchestrationContext):
    """The orchestrator publishes events only when the host can carry them."""
    events: list[dict] = []

    task = ctx.prepare_agent_task("planner", "Plan next sprint.", ctx.instance_id)
    result = yield task

    if ctx.supports_event_streaming:
        # DurableTaskWorkflowContext returns True; AzureFunctionsWorkflowContext False
        events.append({"type": "agent_completed", "executor": "planner"})
        # publish events to custom status ...

    yield ctx.prepare_activity_task("notify_team", '{"message": "Planning done."}')
```

**Example 3 — Custom host adapter implementing the protocol**

```python
from datetime import datetime, timezone
from typing import Any
from agent_framework_durabletask._workflows.context import WorkflowOrchestrationContext

class InMemoryOrchestrationContext:
    """Minimal in-process adapter for unit-testing workflow orchestrators."""

    def __init__(self, instance_id: str = "test-run") -> None:
        self._instance_id = instance_id
        self._tasks: list[tuple[str, str, Any]] = []

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def is_replaying(self) -> bool:
        return False  # always live in tests

    @property
    def supports_event_streaming(self) -> bool:
        return True

    @property
    def current_utc_datetime(self) -> datetime:
        return datetime.now(timezone.utc)

    def prepare_agent_task(self, executor_id: str, message: str, orchestration_instance_id: str) -> Any:
        self._tasks.append(("agent", executor_id, message))
        return f"result_from_{executor_id}"  # pre-completed stub

    def prepare_activity_task(self, activity_name: str, input_json: str) -> Any:
        self._tasks.append(("activity", activity_name, input_json))
        return "{}"

# NOTE: A complete adapter must also implement the remaining protocol members:
# task_all, task_any, wait_for_external_event, create_timer, set_custom_status,
# new_uuid, cancel_task, get_task_result.
# This snippet illustrates only the dispatch methods; isinstance() checks against
# the full protocol will fail until all members are present.
```

---

## 5 · `DurableWorkflowClient`

**Module:** `agent_framework_durabletask._workflows.client`  
**Import:** `from agent_framework.azure import DurableWorkflowClient`

`DurableWorkflowClient` wraps a `TaskHubGrpcClient` and provides a clean interface for
**external clients** to start, await, stream, and drive HITL pauses in durable workflows
registered via `DurableAIAgentWorker.configure_workflow`. It is the workflow counterpart
to `DurableAIAgentClient` (which drives individual durable *agents*).

| Method | Returns | Description |
|--------|---------|-------------|
| `start_workflow(input, *, instance_id?)` | `str` | Schedule the workflow; returns instance ID |
| `await_workflow_output(instance_id, *, timeout_seconds?)` | `Any` | Block until terminal; returns deserialized output |
| `get_runtime_status(instance_id)` | `str \| None` | Non-blocking status poll — returns `"RUNNING"`, `"COMPLETED"`, `"FAILED"`, `"TERMINATED"`, etc., or `None` if unknown |
| `get_pending_hitl_requests(instance_id)` | `list[dict]` | Fetch pending HITL request payloads |
| `send_hitl_response(instance_id, request_id, response)` | `None` | Unblock a HITL pause |
| `stream_workflow(instance_id, *, poll_interval_seconds?, timeout_seconds?)` | `AsyncIterator[WorkflowEvent]` | Stream typed `WorkflowEvent` objects as they arrive |

**Example 1 — Start a workflow and await its output**

```python
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from agent_framework.azure import DurableWorkflowClient

def main() -> None:
    base_client = DurableTaskSchedulerClient(
        host_address="my-scheduler.eastus.durabletask.io:443",
        taskhub="production",
    )
    wf_client = DurableWorkflowClient(base_client)

    instance_id = wf_client.start_workflow(
        input="Analyse the attached quarterly report and produce an executive summary.",
    )
    print(f"Started: {instance_id}")

    output = wf_client.await_workflow_output(instance_id, timeout_seconds=300)
    print("Final output:", output)

main()
```

**Example 2 — Streaming workflow events in real time**

```python
import asyncio
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from agent_framework.azure import DurableWorkflowClient
from agent_framework import WorkflowEvent

async def main() -> None:
    base_client = DurableTaskSchedulerClient(
        host_address="my-scheduler.eastus.durabletask.io:443",
        taskhub="production",
    )
    wf_client = DurableWorkflowClient(base_client)
    instance_id = wf_client.start_workflow(input="Draft a product launch plan.")

    async for event in wf_client.stream_workflow(
        instance_id,
        poll_interval_seconds=2.0,
        timeout_seconds=600,
    ):
        print(f"[{event.type}] executor={event.executor_id} data={event.data}")

asyncio.run(main())
```

**Example 3 — HITL: detecting and responding to approval pauses**

```python
import time
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from agent_framework.azure import DurableWorkflowClient

def main() -> None:
    base_client = DurableTaskSchedulerClient(
        host_address="my-scheduler.eastus.durabletask.io:443",
        taskhub="production",
    )
    wf_client = DurableWorkflowClient(base_client)
    instance_id = wf_client.start_workflow(input="Process payment of $12,500.")

    # Poll until a HITL pause appears — get_pending_hitl_requests/send_hitl_response/
    # await_workflow_output are synchronous/blocking; only stream_workflow is async
    for _ in range(30):
        time.sleep(5)
        pending = wf_client.get_pending_hitl_requests(instance_id)
        if pending:
            req = pending[0]
            request_id = req["request_id"]
            print(f"Pending approval: {req.get('description', 'No description')}")
            # Human approves — send the response
            wf_client.send_hitl_response(
                instance_id,
                request_id,
                response={"approved": True, "approver": "finance_team"},
            )
            break

    output = wf_client.await_workflow_output(instance_id, timeout_seconds=120)
    print("Workflow output:", output)

main()
```

---

## 6 · `WorkflowRegistrationPlan` + `plan_workflow_registration`

**Module:** `agent_framework_durabletask._workflows.registration`  
**Import:** `from agent_framework_durabletask._workflows.registration import WorkflowRegistrationPlan, plan_workflow_registration`

`plan_workflow_registration` introspects a `Workflow` and classifies each node:

- **`AgentExecutor` nodes** → durable **entities** (stateful, addressable by `executor.id`)
- **other `Executor` nodes** → durable **activities** (stateless functions)
- The orchestrator itself is always named `WORKFLOW_ORCHESTRATOR_NAME = "workflow_orchestrator"`

The result is a `WorkflowRegistrationPlan` dataclass that each host reads and maps to its
own registration API — identical decision, different mechanism.

```python
@dataclass
class WorkflowRegistrationPlan:
    agent_executors: list[AgentExecutor]
    activity_executors: list[Executor]
    orchestrator_name: str   # always "workflow_orchestrator"
```

**Example 1 — Inspecting what a workflow registers**

```python
from agent_framework import Agent, AgentExecutor, FunctionExecutor, Workflow
from agent_framework.openai import OpenAIChatClient
from agent_framework_durabletask._workflows.registration import plan_workflow_registration

researcher = Agent(client=OpenAIChatClient(model="gpt-4o"), name="researcher")
writer = Agent(client=OpenAIChatClient(model="gpt-4o"), name="writer")

def format_report(inputs: dict) -> str:
    return f"## Report\n{inputs.get('draft', '')}"

workflow = (
    Workflow()
    .add_executor(AgentExecutor(agent=researcher))
    .add_executor(AgentExecutor(agent=writer))
    .add_executor(FunctionExecutor(func=format_report, id="formatter"))
    .add_edge("researcher", "writer")
    .add_edge("writer", "formatter")
)

plan = plan_workflow_registration(workflow)
print("Entities (agents):", [e.id for e in plan.agent_executors])
print("Activities (functions):", [e.id for e in plan.activity_executors])
print("Orchestrator name:", plan.orchestrator_name)
# Entities (agents): ['researcher', 'writer']
# Activities (functions): ['formatter']
# Orchestrator name: workflow_orchestrator
```

**Example 2 — Using the plan for custom host registration**

```python
from agent_framework import AgentExecutor, FunctionExecutor, Workflow
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent
from agent_framework_durabletask._workflows.registration import plan_workflow_registration

def extract_data(inputs: dict) -> dict:
    return {"rows": 42}

workflow = (
    Workflow()
    .add_executor(AgentExecutor(agent=Agent(client=OpenAIChatClient(model="gpt-4o"), name="analyst")))
    .add_executor(FunctionExecutor(func=extract_data, id="extractor"))
    .add_edge("extractor", "analyst")
)

plan = plan_workflow_registration(workflow)

# A custom host can now register exactly the right primitives:
for ae in plan.agent_executors:
    print(f"Register entity: {ae.id}")         # → Register entity: analyst
for fe in plan.activity_executors:
    print(f"Register activity: {fe.id}")        # → Register activity: extractor
```

**Example 3 — Plan is deterministic across calls**

```python
from agent_framework import Agent, AgentExecutor, Workflow
from agent_framework.openai import OpenAIChatClient
from agent_framework_durabletask._workflows.registration import (
    WorkflowRegistrationPlan,
    plan_workflow_registration,
)

def build_workflow() -> Workflow:
    agent = Agent(client=OpenAIChatClient(model="gpt-4o"), name="agent")
    return Workflow().add_executor(AgentExecutor(agent=agent))

plan1: WorkflowRegistrationPlan = plan_workflow_registration(build_workflow())
plan2: WorkflowRegistrationPlan = plan_workflow_registration(build_workflow())

assert plan1.orchestrator_name == plan2.orchestrator_name == "workflow_orchestrator"
assert len(plan1.agent_executors) == len(plan2.agent_executors) == 1
assert plan1.activity_executors == plan2.activity_executors == []
print("Plans are deterministic ✓")
```

---

## 7 · `TaskType` + `TaskMetadata` + `PendingHITLRequest` + `ExecutorResult`

**Module:** `agent_framework_durabletask._workflows.orchestrator`  
**Import:** `from agent_framework_durabletask._workflows.orchestrator import TaskType, TaskMetadata, PendingHITLRequest, ExecutorResult`

These four dataclasses form the internal data model of `run_workflow_orchestrator` — the
generator-based engine that turns a MAF `Workflow` into a sequence of durable tasks:

```python
class TaskType(Enum):
    AGENT = "agent"
    ACTIVITY = "activity"

@dataclass
class TaskMetadata:
    executor_id: str
    message: Any
    source_executor_id: str
    task_type: TaskType
    remaining_messages: list[tuple[str, Any, str]] | None = None

@dataclass
class PendingHITLRequest:
    request_id: str
    source_executor_id: str
    request_data: Any
    request_type: str | None
    response_type: str | None

@dataclass
class ExecutorResult:
    executor_id: str
    output_message: AgentExecutorResponse | None
    activity_result: dict[str, Any] | None
    task_type: TaskType
```

**Example 1 — Identifying task types in a custom orchestration monitor**

```python
from agent_framework_durabletask._workflows.orchestrator import TaskType, TaskMetadata

def log_pending_tasks(pending: list[TaskMetadata]) -> None:
    for task in pending:
        kind = "Agent entity call" if task.task_type == TaskType.AGENT else "Activity call"
        print(f"[{kind}] executor={task.executor_id} source={task.source_executor_id}")

# Simulated pending task list (in production these come from the orchestrator):
pending = [
    TaskMetadata(
        executor_id="summarizer",
        message="Summarize this document.",
        source_executor_id="__workflow_start__",
        task_type=TaskType.AGENT,
    ),
    TaskMetadata(
        executor_id="send_email",
        message='{"to": "user@example.com"}',
        source_executor_id="summarizer",
        task_type=TaskType.ACTIVITY,
    ),
]
log_pending_tasks(pending)
```

**Example 2 — Tracking HITL pause metadata**

```python
from agent_framework_durabletask._workflows.orchestrator import PendingHITLRequest

def display_pending_approvals(requests: list[PendingHITLRequest]) -> None:
    if not requests:
        print("No pending approvals.")
        return
    for req in requests:
        print(
            f"Request ID : {req.request_id}\n"
            f"Executor   : {req.source_executor_id}\n"
            f"Request    : {req.request_data}\n"
            f"Expected   : {req.response_type or 'any'}\n"
        )

# Simulated pending request:
display_pending_approvals([
    PendingHITLRequest(
        request_id="req-abc123",
        source_executor_id="approver_agent",
        request_data={"action": "deploy", "environment": "production"},
        request_type="approval",
        response_type="bool",
    )
])
```

**Example 3 — Processing an `ExecutorResult` after task completion**

```python
from agent_framework import AgentExecutorResponse, AgentResponse, Message
from agent_framework_durabletask._workflows.orchestrator import ExecutorResult, TaskType

def process_result(result: ExecutorResult) -> str:
    """Extract the text output from an executor result."""
    if result.task_type == TaskType.AGENT:
        if result.output_message and result.output_message.agent_response:
            response: AgentResponse = result.output_message.agent_response
            return response.text
        return "(no agent output)"
    else:
        # Activity result is a raw dict
        if result.activity_result:
            return str(result.activity_result)
        return "(no activity output)"

# Example usage:
mock_response = AgentResponse(
    messages=[Message("assistant", ["Done."])],
)
mock_executor_response = AgentExecutorResponse(
    executor_id="summarizer",
    agent_response=mock_response,
    full_conversation=[],
)
result = ExecutorResult(
    executor_id="summarizer",
    output_message=mock_executor_response,
    activity_result=None,
    task_type=TaskType.AGENT,
)
print(process_result(result))  # Done.
```

---

## 8 · `DurableTaskWorkflowContext`

**Module:** `agent_framework_durabletask._workflows.dt_context`  
**Import:** `from agent_framework_durabletask._workflows.dt_context import DurableTaskWorkflowContext`

`DurableTaskWorkflowContext` is the **standalone durabletask** adapter implementing
`WorkflowOrchestrationContext`. It wraps `durabletask.task.OrchestrationContext` and maps
its API to the protocol.

Key difference from the Azure Functions adapter: `supports_event_streaming` returns `True`
because the standalone Durable Task Scheduler backend has no 16 KB custom-status cap, so the
full accumulated `WorkflowEvent` log can be published and streamed by `DurableWorkflowClient`.

```
DurableTaskWorkflowContext.supports_event_streaming → True
AzureFunctionsWorkflowContext.supports_event_streaming → False
```

The adapter also composes with `OrchestrationAgentExecutor` (which dispatches entity calls
through the standalone SDK) and provides `task_all` / `task_any` via the native
`when_all` / `when_any` from `durabletask.task`.

**Example 1 — Adapter construction in a registered orchestrator**

```python
from durabletask.task import OrchestrationContext
from agent_framework_durabletask._workflows.dt_context import DurableTaskWorkflowContext
from agent_framework_durabletask._workflows.orchestrator import run_workflow_orchestrator
from agent_framework import Workflow

# This is registered with DurableAIAgentWorker.configure_workflow internally.
# Shown here to illustrate the adapter's usage:
def my_workflow_orchestrator(ctx: OrchestrationContext):
    workflow: Workflow = ...   # retrieved from worker configuration
    adapted = DurableTaskWorkflowContext(ctx)
    yield from run_workflow_orchestrator(adapted, workflow, initial_message=ctx.get_input())
```

**Example 2 — Replay-safe logging via `is_replaying`**

```python
import logging
from durabletask.task import OrchestrationContext
from agent_framework_durabletask._workflows.dt_context import DurableTaskWorkflowContext

logger = logging.getLogger("workflow")

def traced_orchestrator(ctx: OrchestrationContext):
    adapted = DurableTaskWorkflowContext(ctx)

    # Avoid logging (or calling external APIs) on replay passes
    if not adapted.is_replaying:
        logger.info("Workflow %s started at %s", adapted.instance_id, adapted.current_utc_datetime)

    task = adapted.prepare_agent_task("agent1", "Do the work.", adapted.instance_id)
    result = yield task

    if not adapted.is_replaying:
        logger.info("Agent returned: %s", result)
```

**Example 3 — `task_all` for parallel fan-out**

```python
from durabletask.task import OrchestrationContext
from agent_framework_durabletask._workflows.dt_context import DurableTaskWorkflowContext

def parallel_orchestrator(ctx: OrchestrationContext):
    adapted = DurableTaskWorkflowContext(ctx)

    # Dispatch two agents in parallel
    t1 = adapted.prepare_agent_task("researcher_a", "Research topic A.", adapted.instance_id)
    t2 = adapted.prepare_agent_task("researcher_b", "Research topic B.", adapted.instance_id)

    results = yield adapted.task_all([t1, t2])
    # results[0] = response from researcher_a, results[1] from researcher_b

    synthesis_input = f"A: {results[0]}\nB: {results[1]}"
    yield adapted.prepare_agent_task("synthesizer", synthesis_input, adapted.instance_id)
```

---

## 9 · `ConversationStore` + `InMemoryConversationStore` + `CheckpointConversationManager`

**Module:** `agent_framework_devui._conversations`  
**Import:** `from agent_framework_devui._conversations import ConversationStore, InMemoryConversationStore, CheckpointConversationManager`

The **DevUI** package exposes an OpenAI-compatible `/v1/conversations` REST surface. These
three classes form its storage layer:

- `ConversationStore` — ABC with 11 abstract methods forming the full contract:
  `create_conversation`, `get_conversation`, `update_conversation(id, metadata)`,
  `delete_conversation`, `add_items`, `list_items`, `get_item`,
  `get_session(id) → AgentSession | None`,
  `list_conversations_by_metadata(filter) → list[Conversation]`,
  `add_trace(id, event)`, `get_traces(id) → list[dict]`.
- `InMemoryConversationStore` — thread-safe in-memory implementation.
  Each conversation maps to `messages: list[Message]`, `session: AgentSession`, and
  a `_item_index` dict for O(1) item lookup.
- `CheckpointConversationManager` — wraps `InMemoryConversationStore` to expose
  per-conversation `InMemoryCheckpointStorage` instances, enabling `FunctionalWorkflow`
  checkpoint resumption across multiple HTTP turns within the same DevUI session.

**Example 1 — Create and retrieve a conversation**

```python
from agent_framework_devui._conversations import InMemoryConversationStore

store = InMemoryConversationStore()

# Create a conversation with metadata
conv = store.create_conversation(
    metadata={"agent_id": "travel_agent", "user_id": "user42"},
)
print(f"Conversation ID: {conv.id}")

# Retrieve it later
retrieved = store.get_conversation(conv.id)
assert retrieved is not None
assert retrieved.id == conv.id
print("Retrieved successfully ✓")
```

**Example 2 — Adding and listing conversation items**

```python
import asyncio
from agent_framework_devui._conversations import InMemoryConversationStore

async def main() -> None:
    store = InMemoryConversationStore()
    conv = store.create_conversation()

    # add_items takes a list of OpenAI-style message dicts and is async
    added = await store.add_items(
        conv.id,
        [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}],
    )
    print(f"Added {len(added)} item(s); first id: {added[0].id}")

    # list_items returns (items, has_more)
    items, has_more = await store.list_items(conv.id)
    print(f"Items in conversation: {len(items)}, has_more={has_more}")

    # Retrieve a specific item by id
    item = await store.get_item(conv.id, added[0].id)
    print(f"Retrieved item role: {item.role if item else 'not found'}")

asyncio.run(main())
```

**Example 3 — Checkpoint-backed conversation for workflow resumption**

```python
from agent_framework_devui._conversations import (
    CheckpointConversationManager,
    InMemoryConversationStore,
)

store = InMemoryConversationStore()
manager = CheckpointConversationManager(store)

# Create two independent conversations, each with its own checkpoint storage
conv_a = store.create_conversation(metadata={"workflow": "order_processing"})
conv_b = store.create_conversation(metadata={"workflow": "support_ticket"})

# Each conversation gets an isolated InMemoryCheckpointStorage
checkpoint_a = manager.get_checkpoint_storage(conv_a.id)
checkpoint_b = manager.get_checkpoint_storage(conv_b.id)

# Checkpoints are isolated — saving to A doesn't affect B
assert checkpoint_a is not checkpoint_b
print("Checkpoint storage is per-conversation isolated ✓")

# The storage can be passed to FunctionalWorkflow for HITL resumption:
# workflow = FunctionalWorkflow(my_func, checkpoint_storage=checkpoint_a)
# response = await agent.run(prompt, session=session, checkpoint_storage=checkpoint_a)
```

---

## 10 · `AgentTask` + `PreCompletedTask` + `AzureFunctionsAgentExecutor`

**Module:** `agent_framework_azurefunctions._orchestration`  
**Import:** `from agent_framework_azurefunctions._orchestration import AgentTask, PreCompletedTask, AzureFunctionsAgentExecutor`

These three classes are the Azure Functions-specific durable orchestration primitives:

- `PreCompletedTask` — a `TaskBase` subclass that is **already completed** at construction
  time. Used in fire-and-forget mode (when `wait_for_response=False`) to return immediately
  with an acceptance response without waiting for entity processing.
- `AgentTask` — a `CompoundTask` subclass that wraps an entity call and intercepts its
  completion to convert the raw entity result into a typed `AgentResponse`.  
  Constructor: `AgentTask(entity_task, response_format, correlation_id)`.
- `AzureFunctionsAgentExecutor` — extends `DurableAgentExecutor[AgentTask]` with the
  Azure Functions `DurableOrchestrationContext`. Overrides `generate_unique_id()` (uses
  `context.new_uuid()` for deterministic replay safety) and `get_run_request()`.

**Example 1 — Fire-and-forget using `PreCompletedTask`**

```python
from agent_framework_azurefunctions._orchestration import PreCompletedTask

# Inside an Azure Durable Functions orchestrator:
def fire_and_forget_orchestrator(context):
    # Immediately return an accepted response instead of waiting for entity:
    task = PreCompletedTask(result={"status": "accepted", "queued": True})
    # The task is already done — yielding returns immediately
    result = yield task
    print("Accepted:", result)  # {'status': 'accepted', 'queued': True}
```

**Example 2 — Typed `AgentTask` wrapping an entity call**

```python
from pydantic import BaseModel
from agent_framework_azurefunctions._orchestration import AgentTask, RunRequest
import azure.durable_functions as df

class SummaryResult(BaseModel):
    summary: str
    word_count: int

def typed_orchestrator(context: df.DurableOrchestrationContext):
    entity_id = df.EntityId(name="summarizer", key=context.instance_id)
    correlation_id = context.new_uuid()  # replay-safe unique ID

    # The entity's "run" operation expects a RunRequest dict, not a bare string
    run_request = RunRequest(
        message="Summarize the quarterly earnings report.",
        correlation_id=correlation_id,
        orchestration_id=context.instance_id,
    )
    raw_task = context.call_entity(entity_id, "run", run_request.to_dict())

    # Wrap it so the result is parsed as SummaryResult
    typed_task = AgentTask(
        entity_task=raw_task,
        response_format=SummaryResult,
        correlation_id=correlation_id,
    )
    result = yield typed_task
    # result is AgentResponse; result.value is SummaryResult if parsing succeeded
    print(f"Summary: {result.text}")
```

**Example 3 — `AzureFunctionsAgentExecutor` for replay-safe IDs**

```python
import azure.durable_functions as df
from agent_framework_azurefunctions._orchestration import AzureFunctionsAgentExecutor

def orchestrator_using_executor(context: df.DurableOrchestrationContext):
    executor = AzureFunctionsAgentExecutor(context)

    # generate_unique_id() calls context.new_uuid() — deterministic on replay
    uid = executor.generate_unique_id()
    print(f"Correlation ID: {uid}")  # same value on every replay pass

    run_req = executor.get_run_request(
        message="Draft a product announcement for the new API.",
        options={"response_format": None, "wait_for_response": True},
    )
    # run_req is a RunRequest that the executor dispatches to the durable entity
```

---

## Summary

| # | Class group | Module | Key facts |
|---|-------------|--------|-----------|
| 1 | `OpenAIChatCompletionOptions` + `Prediction` + `PredictionTextContent` + `RawOpenAIChatCompletionClient` + `OpenAIChatCompletionClient` | `agent_framework_openai._chat_completion_client` | Chat Completions API (not Responses); `logprobs`/`top_logprobs`, `prediction` speculative decoding, `store`, `web_search_options`, `stream_options`; same 4-layer MRO as `OpenAIChatClient` |
| 2 | `ReasoningOptions` + `StreamOptions` + `OpenAIContinuationToken` | `agent_framework_openai._chat_client` | `ReasoningOptions(effort, summary)` for o-series/gpt-5; `StreamOptions(include_usage)` for usage stats in stream; `OpenAIContinuationToken.response_id` for background response polling |
| 3 | `OpenAIContentFilterException` + `ContentFilterResult` + `ContentFilterResultSeverity` + `ContentFilterCodes` | `agent_framework_openai._exceptions` | Azure content filter exception with per-category `ContentFilterResult(filtered, severity)` dict; `ContentFilterResultSeverity`: HIGH/MEDIUM/LOW/SAFE; `ContentFilterCodes.RESPONSIBLE_AI_POLICY_VIOLATION` |
| 4 | `WorkflowOrchestrationContext` | `agent_framework_durabletask._workflows.context` | `@runtime_checkable` Protocol; `is_replaying`, `supports_event_streaming`, `current_utc_datetime`; `prepare_agent_task` / `prepare_activity_task`; host-agnostic engine interface |
| 5 | `DurableWorkflowClient` | `agent_framework_durabletask._workflows.client` | `start_workflow(input)` → instance ID; `await_workflow_output(id, timeout?)` blocking poll; `stream_workflow(id)` → `AsyncIterator[WorkflowEvent]`; `get_pending_hitl_requests` + `send_hitl_response` for HITL |
| 6 | `WorkflowRegistrationPlan` + `plan_workflow_registration` | `agent_framework_durabletask._workflows.registration` | Classifies `AgentExecutor` → entities, other `Executor` → activities; deterministic; `orchestrator_name = "workflow_orchestrator"` |
| 7 | `TaskType` + `TaskMetadata` + `PendingHITLRequest` + `ExecutorResult` | `agent_framework_durabletask._workflows.orchestrator` | `TaskType.AGENT`/`.ACTIVITY` enum; `TaskMetadata` routing state; `PendingHITLRequest` HITL tracking; `ExecutorResult` post-completion result container |
| 8 | `DurableTaskWorkflowContext` | `agent_framework_durabletask._workflows.dt_context` | Standalone durabletask adapter; `supports_event_streaming=True` (no 16 KB cap); `task_all` / `task_any` via native `when_all` / `when_any`; composes with `OrchestrationAgentExecutor` |
| 9 | `ConversationStore` + `InMemoryConversationStore` + `CheckpointConversationManager` | `agent_framework_devui._conversations` | OpenAI Conversations API surface for DevUI; O(1) item lookup via `_item_index`; `CheckpointConversationManager` isolates per-conversation `InMemoryCheckpointStorage` for workflow resumption |
| 10 | `AgentTask` + `PreCompletedTask` + `AzureFunctionsAgentExecutor` | `agent_framework_azurefunctions._orchestration` | `PreCompletedTask` for fire-and-forget; `AgentTask` typed entity-call wrapper with `response_format` Pydantic parsing; `AzureFunctionsAgentExecutor.generate_unique_id()` uses `context.new_uuid()` for replay safety |

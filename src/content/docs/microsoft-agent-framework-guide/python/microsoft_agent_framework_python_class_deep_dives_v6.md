---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 6"
description: "Source-verified deep dives into 10 class groups from agent-framework-core 1.7.0: ExperimentalFeature + feature-staging decorators, WorkflowRunState + WorkflowErrorDetails, WorkflowExecutor + SubWorkflow message pair, AgentResponse + AgentResponseUpdate + ContinuationToken, BaseEmbeddingClient + embedding type family, FunctionInvocationConfiguration, ClassSkill + SkillFrontmatter + FileSkillsSource + SkillsProvider, Annotation + TextSpanRegion, provider capability protocols (SupportsCodeInterpreterTool + SupportsWebSearchTool + SupportsImageGenerationTool + SupportsMCPTool + SupportsFileSearchTool), and MiddlewareType + AgentMiddlewareLayer + ChatMiddlewareLayer."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 25
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 6

Verified against **agent-framework-core 1.7.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source at `/usr/local/lib/python3.11/dist-packages/agent_framework/`. No API name has
been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse` + `ChatResponseUpdate`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor` + `AgentExecutorRequest` + `AgentExecutorResponse`, edge groups, `Runner` + `WorkflowMessage`, `SessionContext`, `AgentSession`, `BaseChatClient` + `SupportsChatGetResponse`, `SecretString` + `load_settings`, `WorkflowCheckpoint` + `CheckpointStorage`, exception hierarchy

This volume fills gaps across six areas: the **feature-staging system**, **workflow
monitoring and error internals**, **hierarchical workflow composition**, **agent-level
response types**, the **embedding client family**, and several **infrastructure** types
that show up repeatedly in real projects.

---

## Table of Contents

1. [`ExperimentalFeature` + `ReleaseCandidateFeature` + `@experimental` / `@release_candidate`](#1-experimentalfeature--releasecandidatefeature--experimental--release_candidate)
2. [`WorkflowRunState` + `WorkflowErrorDetails`](#2-workflowrunstate--workflowerrordetails)
3. [`WorkflowExecutor` + `SubWorkflowRequestMessage` + `SubWorkflowResponseMessage`](#3-workflowexecutor--subworkflowrequestmessage--subworkflowresponsemessage)
4. [`AgentResponse` + `AgentResponseUpdate` + `ContinuationToken`](#4-agentresponse--agentresponseupdate--continuationtoken)
5. [`BaseEmbeddingClient` + `SupportsGetEmbeddings` + `Embedding` + `EmbeddingGenerationOptions` + `GeneratedEmbeddings`](#5-baseembeddingclient--supportsgetsembeddings--embedding--embeddinggenerationoptions--generatedembeddings)
6. [`FunctionInvocationConfiguration`](#6-functioninvocationconfiguration)
7. [`ClassSkill` + `SkillFrontmatter` + `FileSkillsSource` + `SkillsProvider`](#7-classskill--skillfrontmatter--fileskillssource--skillsprovider)
8. [`Annotation` + `TextSpanRegion`](#8-annotation--textspanregion)
9. [Provider capability protocols — `SupportsCodeInterpreterTool`, `SupportsWebSearchTool`, `SupportsImageGenerationTool`, `SupportsMCPTool`, `SupportsFileSearchTool`](#9-provider-capability-protocols)
10. [`MiddlewareType` + `AgentMiddlewareLayer` + `ChatMiddlewareLayer`](#10-middlewaretype--agentmiddlewarelayer--chatmiddlewarelayer)

---

## 1. `ExperimentalFeature` + `ReleaseCandidateFeature` + `@experimental` / `@release_candidate`

**Source:** `agent_framework/_feature_stage.py`

The framework gates unstable APIs behind a two-tier staging system so you know at a
glance what is safe to ship. Every class or function that is not yet stable is
decorated with either `@experimental` or `@release_candidate`, which:

- emits a one-time `ExperimentalWarning` (a `FutureWarning` subclass) on first use, and
- injects a `.. warning:: Experimental` / `.. note:: Release candidate` block into the
  object's docstring.

### Enums

```python
class ExperimentalFeature(str, Enum):
    """Inventory of currently experimental feature IDs."""
    EVALS               = "EVALS"
    FILE_HISTORY        = "FILE_HISTORY"
    FIDES               = "FIDES"
    FOUNDRY_TOOLS       = "FOUNDRY_TOOLS"
    FOUNDRY_PREVIEW_TOOLS = "FOUNDRY_PREVIEW_TOOLS"
    FUNCTIONAL_WORKFLOWS = "FUNCTIONAL_WORKFLOWS"
    HARNESS             = "HARNESS"
    SKILLS              = "SKILLS"
    TO_PROMPT_AGENT     = "TO_PROMPT_AGENT"

class ReleaseCandidateFeature(str, Enum):
    """Inventory of release-candidate feature IDs (currently empty at 1.7.0)."""
```

### Decorators

```python
def experimental(*, feature_id: ExperimentalFeature) -> Callable[[T], T]: ...
def release_candidate(*, feature_id: ReleaseCandidateFeature) -> Callable[[T], T]: ...
```

Both decorators accept classes, plain functions, `async` functions, `staticmethod`, and
`classmethod` descriptors. For Protocol classes the runtime warning is suppressed (they
are structural types that are not instantiated directly).

The warning fires once per `(category, feature_id)` pair per interpreter session — repeat
imports do not produce duplicate warnings.

### Practical impact

| Feature ID | Affected classes (1.7.0) |
|---|---|
| `SKILLS` | `SkillFrontmatter`, `InlineSkill`, `ClassSkill`, `SkillResource`, `FileSkill`, `SkillsProvider`, `SkillScriptRunner`, `MemoryStore` |
| `HARNESS` | `MemoryContextProvider`, `BackgroundAgentsProvider`, `TodoProvider`, `AgentModeProvider` |
| `EVALS` | `EvalNotPassedError`, `AgentEvalConverter` |
| `FILE_HISTORY` | `FileHistoryProvider` |
| `FUNCTIONAL_WORKFLOWS` | `FunctionalWorkflow`, `@workflow`, `RunContext` |

### Suppressing warnings in test suites

```python
import warnings
from agent_framework._feature_stage import ExperimentalWarning

# Silence all experimental warnings during tests
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ExperimentalWarning)
    from agent_framework import MemoryContextProvider  # no warning
```

### Detecting whether an object is experimental

```python
import agent_framework
from agent_framework._feature_stage import _FEATURE_STAGE_ATTR, _FEATURE_ID_ATTR

obj = agent_framework.MemoryContextProvider
stage = getattr(obj, _FEATURE_STAGE_ATTR, None)    # "experimental" or None
fid   = getattr(obj, _FEATURE_ID_ATTR, None)        # "HARNESS" or None
print(stage, fid)   # experimental HARNESS
```

### Key points

- `ExperimentalWarning` is a `FutureWarning` — it is shown by default in Python's
  `-Wall` mode and in many CI configurations.
- The `ReleaseCandidateFeature` enum is currently empty at 1.7.0, which means all
  staged APIs are either experimental or fully stable.
- Members of these enums are an _inventory_, not a stability contract. Members may
  disappear when a feature graduates — check with `getattr(obj, _FEATURE_STAGE_ATTR, None)`
  rather than enum membership.

---

## 2. `WorkflowRunState` + `WorkflowErrorDetails`

**Source:** `agent_framework/_workflows/_events.py`

`WorkflowRunState` and `WorkflowErrorDetails` appear in `WorkflowEvent` payloads and in
`WorkflowRunResult`. They are the two primary observability hooks for monitoring live
workflow execution.

### `WorkflowRunState`

```python
class WorkflowRunState(str, Enum):
    STARTED                      = "STARTED"
    IN_PROGRESS                  = "IN_PROGRESS"
    IN_PROGRESS_PENDING_REQUESTS = "IN_PROGRESS_PENDING_REQUESTS"
    IDLE                         = "IDLE"
    IDLE_WITH_PENDING_REQUESTS   = "IDLE_WITH_PENDING_REQUESTS"
    FAILED                       = "FAILED"
    CANCELLED                    = "CANCELLED"
```

**State machine transitions:**

```
STARTED → IN_PROGRESS → IDLE (converged, no pending requests)
                      → IN_PROGRESS_PENDING_REQUESTS (waiting for request_info responses)
                      → IDLE_WITH_PENDING_REQUESTS (superstep done, outstanding requests remain)
                      → FAILED (executor raised an unhandled exception)
                      → CANCELLED (runner told to stop)
```

`IDLE` is the terminal happy-path state after `run()` returns. `FAILED` carries a
`WorkflowErrorDetails` payload.

### `WorkflowErrorDetails`

```python
@dataclass
class WorkflowErrorDetails:
    error_type:  str               # Exception class name
    message:     str               # str(exc)
    traceback:   str | None = None # Full traceback text, or None if unavailable
    executor_id: str | None = None # Which executor raised the exception
    extra:       dict[str, Any] | None = None  # Caller-supplied metadata

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        executor_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> WorkflowErrorDetails: ...
```

`from_exception` builds the instance from a live exception, capturing the full traceback
via `traceback.format_exception`. It is called automatically by the framework when an
executor raises. You can also build one manually for test assertions.

### Reading `WorkflowRunState` from the event stream

```python
import asyncio
from agent_framework import WorkflowBuilder, Agent
from agent_framework._workflows._events import WorkflowRunState, WorkflowErrorDetails

async def monitor_workflow(builder: WorkflowBuilder, message: str) -> None:
    workflow = builder.build()
    async for event in workflow.run_stream(message):
        if event.type == "status":
            print(f"State → {event.state.value}")
            if event.state == WorkflowRunState.IDLE:
                print("Workflow converged.")
            elif event.state == WorkflowRunState.IN_PROGRESS_PENDING_REQUESTS:
                print("Waiting for human input …")
        elif event.type == "failed":
            details: WorkflowErrorDetails = event.details
            print(f"FAILED in executor '{details.executor_id}': {details.message}")
            if details.traceback:
                print(details.traceback)
        elif event.type == "output":
            print(f"Output from {event.executor_id}: {event.data}")

asyncio.run(monitor_workflow(my_builder, "Start the analysis."))
```

### Structured error logging with `WorkflowErrorDetails`

```python
import json
from agent_framework._workflows._events import WorkflowErrorDetails

def log_workflow_failure(details: WorkflowErrorDetails) -> None:
    record = {
        "error_type": details.error_type,
        "message": details.message,
        "executor_id": details.executor_id,
        "has_traceback": details.traceback is not None,
    }
    print(json.dumps(record))

# Build from a live exception for unit tests
try:
    raise ValueError("Simulated executor failure")
except ValueError as exc:
    details = WorkflowErrorDetails.from_exception(exc, executor_id="summarise")
    log_workflow_failure(details)
# {"error_type": "ValueError", "message": "Simulated executor failure",
#  "executor_id": "summarise", "has_traceback": true}
```

---

## 3. `WorkflowExecutor` + `SubWorkflowRequestMessage` + `SubWorkflowResponseMessage`

**Source:** `agent_framework/_workflows/_workflow_executor.py`

`WorkflowExecutor` wraps any `Workflow` object so it behaves as a single `Executor`
inside a parent workflow. This is the mechanism for **hierarchical workflow composition**
— building complex orchestrations from reusable sub-workflows.

### `SubWorkflowResponseMessage` and `SubWorkflowRequestMessage`

```python
@dataclass
class SubWorkflowResponseMessage:
    """Response sent from a parent executor back into a sub-workflow."""
    data:         Any           # The response payload
    source_event: WorkflowEvent # The original request_info event from the sub-workflow

@dataclass
class SubWorkflowRequestMessage:
    """Request emitted by a sub-workflow executor, routed to the parent workflow."""
    source_event: WorkflowEvent # The request_info event from the sub-workflow
    executor_id:  str           # ID of the WorkflowExecutor in the parent workflow

    def create_response(self, data: Any) -> SubWorkflowResponseMessage:
        """Wrap response data; validates type against source_event.response_type."""
        ...
```

### `WorkflowExecutor`

```python
class WorkflowExecutor(Executor):
    def __init__(
        self,
        workflow: Workflow,
        *,
        id: str | None = None,
        allow_direct_output: bool = False,
    ) -> None: ...
```

| Parameter | Description |
|---|---|
| `workflow` | The `Workflow` instance to wrap as a sub-workflow |
| `id` | Executor ID in the parent workflow's graph (defaults to `workflow.id`) |
| `allow_direct_output` | When `True`, sub-workflow outputs are yielded directly to the parent's event stream rather than forwarded as messages |

**Execution model:**
1. The parent runner invokes `WorkflowExecutor` with a message.
2. `WorkflowExecutor` starts the sub-workflow and runs it to completion.
3. Sub-workflow outputs are forwarded to the parent as messages (or directly, if `allow_direct_output=True`).
4. If the sub-workflow emits `request_info` events, they are wrapped in `SubWorkflowRequestMessage` and forwarded to a parent executor.
5. The parent executor responds via `SubWorkflowResponseMessage`, which resumes the sub-workflow.

### Building a nested workflow

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder
from agent_framework._workflows._workflow_executor import WorkflowExecutor

# ── Inner workflow: data extraction ──────────────────────────────────────────
inner_builder = WorkflowBuilder()

@inner_builder.executor("extract")
async def extract(ctx):
    msg = await ctx.receive(str)
    await ctx.yield_output(f"Extracted: {msg}")

inner_workflow = inner_builder.build()

# ── Outer workflow: orchestration ─────────────────────────────────────────────
outer_builder = WorkflowBuilder()

# Wrap the inner workflow as a single executor
inner_executor = WorkflowExecutor(inner_workflow, id="extractor")
outer_builder.add_executor(inner_executor)

@outer_builder.executor("summarise", input_types=[str])
async def summarise(ctx):
    msg = await ctx.receive(str)
    # Forward to inner workflow via WorkflowExecutor
    await ctx.send_message("extractor", msg)
    result = await ctx.receive(str)
    await ctx.yield_output(f"Summary complete: {result}")

outer_builder.add_edge("summarise", "extractor")
outer_workflow = outer_builder.build(start_executors=["summarise"])

async def main():
    result = await outer_workflow.run("Process this text.")
    print(result.output)  # "Summary complete: Extracted: Process this text."

asyncio.run(main())
```

### HITL requests across workflow boundaries

When a sub-workflow executor calls `ctx.request_info(...)`, the event is surfaced to the
parent workflow as a `SubWorkflowRequestMessage`. A parent executor can intercept it and
respond:

```python
from agent_framework._workflows._workflow_executor import (
    SubWorkflowRequestMessage,
    SubWorkflowResponseMessage,
)
from agent_framework._workflows._executor import handler

@outer_builder.executor("human-relay")
async def human_relay(ctx):
    # Receive a HITL request forwarded from the inner sub-workflow
    req: SubWorkflowRequestMessage = await ctx.receive(SubWorkflowRequestMessage)

    # Collect human input (could call an API, wait for a web form, etc.)
    approval = input(f"Sub-workflow asks: {req.source_event.data}\nApprove? (y/n): ")

    # Send typed response back to WorkflowExecutor to resume the sub-workflow
    response = req.create_response("approved" if approval == "y" else "denied")
    await ctx.send_message(req.executor_id, response)
```

---

## 4. `AgentResponse` + `AgentResponseUpdate` + `ContinuationToken`

**Source:** `agent_framework/_types.py`

`AgentResponse` is the **agent-level** response object returned by `Agent.run()`. It is
distinct from `ChatResponse` (which is the raw LLM output) — `AgentResponse` aggregates
the full conversation turn including any function calls and the final assistant message.

### `AgentResponse`

```python
class AgentResponse(SerializationMixin, Generic[ResponseModelT]):
    def __init__(
        self,
        *,
        messages: Message | Sequence[Message] | None = None,
        response_id: str | None = None,
        agent_id: str | None = None,
        created_at: datetime | None = None,
        finish_reason: FinishReasonLiteral | None = None,
        usage_details: UsageDetails | None = None,
        value: ResponseModelT | None = None,
        response_format: StructuredResponseFormat = None,
        continuation_token: ContinuationToken | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None: ...
```

| Attribute | Type | Description |
|---|---|---|
| `messages` | `list[Message]` | All messages in the response turn (may include tool-call messages) |
| `text` | `str` (property) | Concatenated text of all assistant messages |
| `response_id` | `str \| None` | ID of the underlying chat response |
| `agent_id` | `str \| None` | Which agent produced this response |
| `finish_reason` | `str \| None` | `"stop"`, `"length"`, `"tool_calls"`, etc. |
| `usage_details` | `UsageDetails \| None` | Token counts for the whole turn |
| `value` | `ResponseModelT \| None` | Parsed structured output (when `response_format` used) |
| `continuation_token` | `ContinuationToken \| None` | Present when the operation is still in progress |
| `user_input_requests` | `list[Content]` (property) | Any HITL request Content items in the response |

#### Serialisation round-trip

```python
from agent_framework import AgentResponse, Message

msg  = Message("assistant", ["Analysis complete — 3 anomalies found."])
resp = AgentResponse(messages=[msg], response_id="run_abc123")

# To / from dict
d    = resp.to_dict()
# {'type': 'agent_response', 'messages': [...], 'response_id': 'run_abc123', ...}
back = AgentResponse.from_dict(d)
assert back.text == resp.text

# To / from JSON
json_str = resp.to_json()
back2    = AgentResponse.from_json(json_str)
assert back2.response_id == "run_abc123"
```

#### Structured output response

```python
from pydantic import BaseModel
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

class Report(BaseModel):
    summary: str
    anomaly_count: int

async def run_structured():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Analyse the data and return a structured report.",
    )
    resp: AgentResponse[Report] = await agent.run(
        "Analyse these numbers: 1, 2, 99, 3, 4",
        options={"response_format": Report},
    )
    report: Report = resp.value   # type: Report, not None
    print(report.anomaly_count)
```

### `AgentResponseUpdate`

`AgentResponseUpdate` is the streaming variant — one update per token or tool call.

```python
class AgentResponseUpdate(SerializationMixin):
    def __init__(
        self,
        *,
        role: str | None = None,
        contents: Sequence[Content | dict] | None = None,
        response_id: str | None = None,
        agent_id: str | None = None,
        ...
    ) -> None: ...

    @property
    def text(self) -> str:
        """Concatenated text across all Content items."""
        ...

    @classmethod
    def from_updates(
        cls,
        updates: Sequence[AgentResponseUpdate],
    ) -> AgentResponse:
        """Collapse a list of streaming updates into a final AgentResponse."""
        ...
```

```python
import asyncio
from agent_framework import Agent, AgentResponseUpdate, AgentResponse
from agent_framework.openai import OpenAIChatClient

async def stream_agent():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Write a haiku about Python.",
    )
    stream = agent.run_stream("Go ahead.")

    # Process individual deltas
    async for update in stream:                          # AgentResponseUpdate
        print(update.text, end="", flush=True)

    # Collect the final, complete response
    final: AgentResponse = await stream.get_response()
    print(f"\nFinish reason: {final.finish_reason}")
    print(f"Tokens used: {final.usage_details}")

asyncio.run(stream_agent())
```

### `ContinuationToken`

```python
class ContinuationToken(TypedDict):
    """Opaque JSON-serializable dict for resuming long-running background operations.

    When present on an AgentResponse, the operation is still in progress.
    None means the operation has completed.
    Each provider subclasses this with its own fields; treat it as opaque.
    """
```

`ContinuationToken` is a bare `TypedDict` — it declares no fields. Provider-specific
extensions carry the actual fields. The contract is:

- `continuation_token is not None` → operation still in progress; poll with it.
- `continuation_token is None` → operation complete.

```python
import json
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient

async def background_poll(agent: Agent, query: str) -> str:
    resp = await agent.run(query)

    # Keep polling while the operation is incomplete
    while resp.continuation_token is not None:
        token_json = json.dumps(resp.continuation_token)
        # (persist token_json, wait, then resume)
        resp = await agent.run(
            query,
            options={"continuation_token": json.loads(token_json)},
        )

    return resp.text
```

---

## 5. `BaseEmbeddingClient` + `SupportsGetEmbeddings` + `Embedding` + `EmbeddingGenerationOptions` + `GeneratedEmbeddings`

**Source:** `agent_framework/_clients.py` (protocols and `BaseEmbeddingClient`); `agent_framework/_types.py` (data classes)

The embedding family enables vector search and semantic similarity without depending on a
specific provider. The design mirrors the chat client family: a `Protocol` for duck-typing,
an abstract base for implementation, and typed data classes for the results.

### `EmbeddingGenerationOptions`

```python
class EmbeddingGenerationOptions(TypedDict, total=False):
    """Common request settings for all embedding providers."""
    model:      str   # e.g. "text-embedding-3-small"
    dimensions: int   # e.g. 1536
```

All fields are optional. Providers extend this TypedDict with their own fields.

### `Embedding`

```python
class Embedding(Generic[EmbeddingT]):
    def __init__(
        self,
        vector: EmbeddingT,
        *,
        model:                 str | None = None,
        dimensions:            int | None = None,
        created_at:            datetime | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None: ...

    @property
    def dimensions(self) -> int | None:
        """Explicit count if set, else len(vector), else None."""
        ...
```

`EmbeddingT` defaults to `list[float]` but can be `list[int]`, `bytes`, or any other
numeric sequence, depending on the provider.

### `GeneratedEmbeddings`

```python
class GeneratedEmbeddings(
    list[Embedding[EmbeddingT]],
    Generic[EmbeddingT, EmbeddingOptionsT],
):
    def __init__(
        self,
        embeddings:            Iterable[Embedding[EmbeddingT]] | None = None,
        *,
        options:               EmbeddingOptionsT | None = None,
        usage:                 dict[str, Any] | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None: ...
```

`GeneratedEmbeddings` extends `list` so you can iterate and index directly.
`usage` carries provider-reported token counts.

### `SupportsGetEmbeddings` protocol

```python
@runtime_checkable
class SupportsGetEmbeddings(Protocol[...]):
    additional_properties: dict[str, Any]

    def get_embeddings(
        self,
        values: Sequence[EmbeddingInputContraT],
        *,
        options: EmbeddingProtocolOptionsT | None = None,
    ) -> Awaitable[GeneratedEmbeddings]: ...
```

Because it is `@runtime_checkable`, you can use `isinstance(client, SupportsGetEmbeddings)`.

### `BaseEmbeddingClient`

```python
class BaseEmbeddingClient(SerializationMixin, ABC, Generic[EmbeddingInputT, EmbeddingT, EmbeddingOptionsT]):
    OTEL_PROVIDER_NAME: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        additional_properties: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_embeddings(
        self,
        values: Sequence[EmbeddingInputT],
        *,
        options: EmbeddingOptionsT | None = None,
    ) -> GeneratedEmbeddings[EmbeddingT, EmbeddingOptionsT]: ...
```

Subclass `BaseEmbeddingClient` to add a custom embedding provider. Set `OTEL_PROVIDER_NAME`
for telemetry attribution.

### Custom embedding client example

```python
import asyncio
import numpy as np
from agent_framework import (
    BaseEmbeddingClient,
    Embedding,
    EmbeddingGenerationOptions,
    GeneratedEmbeddings,
)

class LocalEmbeddingClient(BaseEmbeddingClient[str, list[float], EmbeddingGenerationOptions]):
    """Simple random-projection client for offline testing."""

    OTEL_PROVIDER_NAME = "local-random"

    def __init__(self, dimensions: int = 128) -> None:
        super().__init__()
        self._dims = dimensions

    async def get_embeddings(
        self,
        values: list[str],
        *,
        options: EmbeddingGenerationOptions | None = None,
    ) -> GeneratedEmbeddings[list[float], EmbeddingGenerationOptions]:
        embeddings = [
            Embedding(
                vector=np.random.randn(self._dims).tolist(),
                model="local-random",
                dimensions=self._dims,
            )
            for _ in values
        ]
        return GeneratedEmbeddings(
            embeddings,
            usage={"prompt_tokens": sum(len(v.split()) for v in values)},
        )

async def semantic_search():
    client = LocalEmbeddingClient(dimensions=64)
    result = await client.get_embeddings(["Python async", "JavaScript promises"])
    print(f"Got {len(result)} embeddings, dims={result[0].dimensions}")
    print(f"Token usage: {result.usage}")

asyncio.run(semantic_search())
```

### Duck-typing check before use

```python
from agent_framework import SupportsGetEmbeddings
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
if isinstance(client, SupportsGetEmbeddings):
    result = await client.get_embeddings(["query text"])
    print(result[0].vector[:5])
else:
    print(f"{type(client).__name__} does not support embeddings.")
```

---

## 6. `FunctionInvocationConfiguration`

**Source:** `agent_framework/_tools.py`

`FunctionInvocationConfiguration` is a `TypedDict` that controls the **LLM ↔ tool loop**:
how many times the model can call tools, when errors abort the loop, and whether hidden
tools are available.

### Class signature

```python
class FunctionInvocationConfiguration(TypedDict, total=False):
    enabled:                         bool
    max_iterations:                  int
    max_function_calls:              int | None
    max_consecutive_errors_per_request: int
    terminate_on_unknown_calls:      bool
    additional_tools:                Sequence[FunctionTool]
    include_detailed_errors:         bool
```

### Default values (applied by `normalize_function_invocation_configuration`)

| Key | Default | Notes |
|---|---|---|
| `enabled` | `True` | Set `False` to disable function calling entirely |
| `max_iterations` | `40` | Max LLM round-trips per `agent.run()` call |
| `max_function_calls` | `None` | Max total tool invocations; `None` = unlimited |
| `max_consecutive_errors_per_request` | `3` | Consecutive errors before aborting the loop |
| `terminate_on_unknown_calls` | `False` | Raise on unknown tool names instead of ignoring |
| `additional_tools` | `[]` | Hidden tools: available for execution, not in the model's tool list |
| `include_detailed_errors` | `False` | Include exception details in tool result sent to model |

### `max_iterations` vs `max_function_calls`

```
max_iterations: caps the number of LLM round-trips (supersteps)
                regardless of how many tools are called in each round-trip.

max_function_calls: caps the total number of individual tool executions
                    across all iterations. Best-effort — enforced after
                    each batch of parallel tool calls completes.

Example:
  max_iterations=5, max_function_calls=10
  → Up to 5 round-trips; stop after 10 total individual tool calls (whichever hits first)
```

### Setting per-client defaults

```python
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()

# Cap each agent.run() call to 5 round-trips and 20 tool calls
client.function_invocation_configuration["max_iterations"] = 5
client.function_invocation_configuration["max_function_calls"] = 20

# Make error details visible to the model for self-correction
client.function_invocation_configuration["include_detailed_errors"] = True
```

### Per-request overrides via `function_invocation_kwargs`

```python
import asyncio
from agent_framework import Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient

async def run_with_budget(agent: Agent, question: str) -> str:
    # Pass per-request configuration via function_invocation_kwargs
    resp = await agent.run(
        question,
        function_invocation_kwargs={
            "config": {
                "max_iterations": 3,
                "max_function_calls": 5,
                "terminate_on_unknown_calls": True,
            }
        },
    )
    return resp.text
```

### `additional_tools` pattern — hidden execution tools

```python
from agent_framework import Agent, FunctionTool, tool

@tool
def internal_audit(payload: str) -> str:
    """Log tool calls for compliance — not advertised to the model."""
    print(f"[AUDIT] {payload}")
    return "logged"

client = OpenAIChatClient()
# Model never sees internal_audit in its tool list, but the framework
# can invoke it as a middleware side-effect or from other tools.
client.function_invocation_configuration["additional_tools"] = [internal_audit]

agent = Agent(client=client, instructions="Help the user.")
```

### Disabling function calling for a single request

```python
async def factual_only(agent: Agent, question: str) -> str:
    resp = await agent.run(
        question,
        function_invocation_kwargs={
            "config": {"enabled": False}  # pure text generation, no tool calls
        },
    )
    return resp.text
```

---

## 7. `ClassSkill` + `SkillFrontmatter` + `FileSkillsSource` + `SkillsProvider`

**Source:** `agent_framework/_skills.py`

The agent framework implements the [Agent Skills specification](https://agentskills.io/)
via a three-phase progressive-disclosure pattern:

1. **Advertise** — skill names and descriptions are injected into the system prompt.
2. **Load** — the full `SKILL.md` body is returned when the model calls `load_skill`.
3. **Read resources** — supplementary content is fetched on demand via `read_skill_resource`.

> **Experimental:** All skills APIs emit `ExperimentalWarning` on first use. Guard with
> `warnings.filterwarnings("ignore", ..., ExperimentalWarning)` in production if needed.

### `SkillFrontmatter`

```python
@experimental(feature_id=ExperimentalFeature.SKILLS)
class SkillFrontmatter:
    def __init__(
        self,
        *,
        name:          str,               # [a-z0-9-]{1,64}; no leading/trailing hyphens
        description:   str,               # ≤1024 characters
        license:       str | None = None,
        compatibility: str | None = None, # ≤500 characters
        allowed_tools: str | None = None, # space-delimited pre-approved tool names
        metadata:      dict[str, str] | None = None,
    ) -> None: ...
```

`SkillFrontmatter` validates `name`, `description`, and `compatibility` at construction
time and raises `ValueError` on violations. Post-construction assignments are not
re-validated.

```python
from agent_framework import SkillFrontmatter

fm = SkillFrontmatter(
    name="sql-helper",
    description="Generates and validates SQL queries.",
    compatibility="Works with PostgreSQL 14+ and SQLite 3.40+.",
    allowed_tools="execute_query validate_schema",
)
```

### `ClassSkill`

```python
@experimental(feature_id=ExperimentalFeature.SKILLS)
class ClassSkill(Skill, ABC):
    def __init__(self, *, frontmatter: SkillFrontmatter) -> None: ...

    @property
    @abstractmethod
    def instructions(self) -> str: ...    # The SKILL.md body

    @property
    def resources(self) -> list[SkillResource]: ...  # Auto-discovered via @ClassSkill.resource

    @property
    def scripts(self) -> list[SkillScript]: ...      # Auto-discovered via @ClassSkill.script

    @staticmethod
    def resource(func=None, *, name=None, description=None) -> Any: ...

    @staticmethod
    def script(func=None, *, name=None, description=None) -> Any: ...
```

`@ClassSkill.resource` marks methods whose return values are served as supplementary
resources. `@ClassSkill.script` marks in-process callable scripts. Both decorators are
applied _before_ `@property` if used together.

```python
import json
import warnings
from agent_framework import ClassSkill, SkillFrontmatter
from agent_framework._feature_stage import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

class SQLHelperSkill(ClassSkill):
    def __init__(self) -> None:
        super().__init__(
            frontmatter=SkillFrontmatter(
                name="sql-helper",
                description="Generates and validates SQL queries.",
            )
        )

    @property
    def instructions(self) -> str:
        return (
            "Use this skill to generate and validate SQL queries.\n"
            "Always use parameterised queries to prevent injection.\n"
            "Use the `schema` resource for table definitions.\n"
            "Use the `validate` script to check query syntax before execution."
        )

    @ClassSkill.resource(name="schema", description="Database schema reference")
    def get_schema(self) -> str:
        return "users(id INT PK, email TEXT UNIQUE, created_at TIMESTAMP)"

    @ClassSkill.script(name="validate", description="Validate SQL syntax")
    def validate_sql(self, sql: str) -> str:
        # In production: call a SQL parser
        if "DROP" in sql.upper():
            return json.dumps({"valid": False, "error": "DROP statements not permitted"})
        return json.dumps({"valid": True})

# Usage
skill = SQLHelperSkill()
print(skill.frontmatter.name)           # "sql-helper"
print(len(skill.resources))             # 1
print(skill.resources[0].name)          # "schema"
print(len(skill.scripts))               # 1
print(skill.scripts[0].name)            # "validate"
```

### `FileSkillsSource`

`FileSkillsSource` scans filesystem directories for `SKILL.md` files following the
[Agent Skills file format](https://agentskills.io/specification#file-format). Each
`SKILL.md` becomes a `FileSkill` with frontmatter parsed from the YAML header.

```python
from agent_framework._skills import FileSkillsSource

# Scan ./skills/ recursively for SKILL.md files
source = FileSkillsSource(directories=["./skills"])
skills = await source.list_skills()        # list[FileSkill]
for s in skills:
    print(s.frontmatter.name, s.frontmatter.description)
```

Expected directory structure:

```
skills/
├── sql-helper/
│   └── SKILL.md
└── data-viz/
    ├── SKILL.md
    └── chart-examples.md
```

`SKILL.md` format (minimal):
```markdown
---
name: sql-helper
description: Generates SQL queries for PostgreSQL and SQLite.
---

Use this skill whenever the user asks for a SQL query...
```

### `SkillsProvider`

`SkillsProvider` is a `ContextProvider` that wires the three-phase disclosure pattern
into an `Agent`. Pass it via `context_providers` on `Agent`.

```python
import warnings
from agent_framework import Agent, SkillsProvider
from agent_framework._feature_stage import ExperimentalWarning
from agent_framework._skills import FileSkillsSource, InMemorySkillsSource
from agent_framework.openai import OpenAIChatClient

warnings.filterwarnings("ignore", category=ExperimentalWarning)

# Mix code-defined and file-based sources
code_source  = InMemorySkillsSource(skills=[SQLHelperSkill()])
file_source  = FileSkillsSource(directories=["./skills"])

provider = SkillsProvider(sources=[code_source, file_source])

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a helpful SQL assistant.",
    context_providers=[provider],   # Injects load_skill + read_skill_resource tools
)
```

When the model calls `load_skill(name="sql-helper")`, `SkillsProvider` returns the full
`SKILL.md` body. When the model calls `read_skill_resource(skill_name="sql-helper", resource_name="schema")`, it returns the resource content.

---

## 8. `Annotation` + `TextSpanRegion`

**Source:** `agent_framework/_types.py`

`Annotation` and `TextSpanRegion` represent citation metadata attached to `Content`
items. They surface when models use Bing grounding, file search, or other retrieval
tools that attribute responses to source documents.

### `TextSpanRegion`

```python
class TextSpanRegion(TypedDict, total=False):
    """A character-range annotation over message text."""
    type:        Literal["text_span"]
    start_index: int  # Inclusive start character offset
    end_index:   int  # Exclusive end character offset
```

`TextSpanRegion` is always embedded inside an `Annotation.annotated_regions` list. It
marks which part of the text body the citation applies to.

### `Annotation`

```python
class Annotation(TypedDict, total=False):
    """Citation / grounding annotation attached to a Content item."""
    type:               Literal["citation"]
    title:              str              # Source document title
    url:                str              # Source URL (if web-grounded)
    file_id:            str              # Azure file ID (if file-search-grounded)
    tool_name:          str              # Which tool produced this annotation
    snippet:            str              # Cited text excerpt
    annotated_regions:  Sequence[TextSpanRegion]
    additional_properties: dict[str, Any]
    raw_representation: Any              # Provider-native annotation object
```

All fields are optional (`total=False`). In practice, web-grounded responses populate
`url` + `title`, file-search responses populate `file_id` + `title`, and both may
include `annotated_regions`.

### Reading annotations from a response

```python
from agent_framework import Agent, AgentResponse

async def grounded_query(agent: Agent, query: str) -> None:
    resp: AgentResponse = await agent.run(query)

    for msg in resp.messages:
        for content_item in msg.contents:
            # Annotations are stored in Content.annotations
            for ann in getattr(content_item, "annotations", []):
                print(f"Source: {ann.get('title', '(no title)')}")
                if url := ann.get("url"):
                    print(f"  URL: {url}")
                for region in ann.get("annotated_regions", []):
                    start = region.get("start_index", 0)
                    end   = region.get("end_index", 0)
                    text  = msg.text[start:end]
                    print(f"  Cited: '{text}' (chars {start}–{end})")
```

### Building annotations manually (test doubles)

```python
from agent_framework._types import Annotation, TextSpanRegion

annotation: Annotation = {
    "type": "citation",
    "title": "Python 3.12 Release Notes",
    "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
    "snippet": "Python 3.12 introduces the new type parameter syntax.",
    "annotated_regions": [
        TextSpanRegion(type="text_span", start_index=0, end_index=42),
    ],
}
```

### Annotation filtering by tool

```python
def filter_annotations(annotations: list[Annotation], tool_name: str) -> list[Annotation]:
    """Return only annotations produced by a specific tool."""
    return [ann for ann in annotations if ann.get("tool_name") == tool_name]

bing_annotations = filter_annotations(all_annotations, "bing_grounding")
file_annotations = filter_annotations(all_annotations, "file_search")
```

---

## 9. Provider capability protocols

**Source:** `agent_framework/_clients.py`

Five `@runtime_checkable` Protocol classes let you test whether a concrete chat client
supports optional provider-managed tools. This enables write-once orchestration code
that adapts to whatever features the underlying provider exposes.

### The five protocols

```python
@runtime_checkable
class SupportsCodeInterpreterTool(Protocol):
    @staticmethod
    def get_code_interpreter_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsWebSearchTool(Protocol):
    @staticmethod
    def get_web_search_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsImageGenerationTool(Protocol):
    @staticmethod
    def get_image_generation_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsMCPTool(Protocol):
    @staticmethod
    def get_mcp_tool(**kwargs: Any) -> Any: ...

@runtime_checkable
class SupportsFileSearchTool(Protocol):
    @staticmethod
    def get_file_search_tool(**kwargs: Any) -> Any: ...
```

Each protocol has exactly one static factory method. Because they are `@runtime_checkable`,
you can use `isinstance` at runtime.

### Provider support matrix (1.7.0)

| Protocol | FoundryChatClient | OpenAIChatClient | AnthropicClient |
|---|:---:|:---:|:---:|
| `SupportsCodeInterpreterTool` | ✓ | ✓ | — |
| `SupportsWebSearchTool` | ✓ | ✓ | — |
| `SupportsImageGenerationTool` | ✓ | — | — |
| `SupportsMCPTool` | ✓ | ✓ | — |
| `SupportsFileSearchTool` | ✓ | ✓ | — |

### Adaptive tool builder

```python
from agent_framework import Agent, SupportsCodeInterpreterTool, SupportsWebSearchTool
from agent_framework import SupportsFileSearchTool, SupportsMCPTool

def build_tools(client, *, vector_store_ids: list[str] | None = None) -> list:
    tools = []

    if isinstance(client, SupportsCodeInterpreterTool):
        tools.append(client.get_code_interpreter_tool())

    if isinstance(client, SupportsWebSearchTool):
        tools.append(client.get_web_search_tool())

    if vector_store_ids and isinstance(client, SupportsFileSearchTool):
        tools.append(client.get_file_search_tool(vector_store_ids=vector_store_ids))

    return tools

# Works identically with FoundryChatClient, OpenAIChatClient, or any future provider
from agent_framework.openai import OpenAIChatClient
client = OpenAIChatClient()
tools  = build_tools(client, vector_store_ids=["vs_abc123"])
agent  = Agent(client=client, instructions="Help the user.", tools=tools)
```

### Capability probe utility

```python
from agent_framework import (
    SupportsCodeInterpreterTool,
    SupportsFileSearchTool,
    SupportsGetEmbeddings,
    SupportsImageGenerationTool,
    SupportsMCPTool,
    SupportsWebSearchTool,
)

def client_capabilities(client) -> dict[str, bool]:
    return {
        "code_interpreter":    isinstance(client, SupportsCodeInterpreterTool),
        "web_search":          isinstance(client, SupportsWebSearchTool),
        "image_generation":    isinstance(client, SupportsImageGenerationTool),
        "mcp":                 isinstance(client, SupportsMCPTool),
        "file_search":         isinstance(client, SupportsFileSearchTool),
        "embeddings":          isinstance(client, SupportsGetEmbeddings),
    }
```

### Custom client that satisfies multiple protocols

```python
from agent_framework import BaseChatClient, SupportsWebSearchTool
from agent_framework._types import ChatResponse

class MockSearchClient(BaseChatClient, SupportsWebSearchTool):
    """Test double that satisfies the web search protocol."""

    @staticmethod
    def get_web_search_tool(**kwargs) -> dict:
        return {"type": "web_search_preview"}

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        return ChatResponse(messages=[], response_id="mock")

assert isinstance(MockSearchClient(), SupportsWebSearchTool)
```

---

## 10. `MiddlewareType` + `AgentMiddlewareLayer` + `ChatMiddlewareLayer`

**Source:** `agent_framework/_middleware.py`

While the public-facing `AgentMiddleware`, `ChatMiddleware`, and `FunctionMiddleware`
base classes are covered in Vol. 2, this section documents the three _layer_ classes that
wrap those middleware implementations into the actual invocation pipeline. Understanding
the layers is useful when building custom clients or inspecting middleware execution.

### `MiddlewareType`

```python
class MiddlewareType(str, Enum):
    """Identifies the middleware category used for telemetry and introspection."""
    AGENT    = "agent"
    FUNCTION = "function"
    CHAT     = "chat"
```

`MiddlewareType` is used internally to route telemetry events and can be read via
`layer.middleware_type` on any layer object.

### `AgentMiddlewareLayer`

`AgentMiddlewareLayer` wraps the list of `AgentMiddleware` instances on a `BaseChatClient`
into a composable pipeline with telemetry.

```python
class AgentMiddlewareLayer:
    """Composable pipeline over a list of AgentMiddleware."""

    def __init__(
        self,
        *,
        middleware: Sequence[AgentMiddleware],
        client: SupportsChatGetResponse,
    ) -> None: ...

    async def get_response(
        self,
        messages: Sequence[Message],
        *,
        agent: SupportsAgentRun,
        session: AgentSession | None = None,
        tools: ToolTypes | None = None,
        options: Mapping[str, Any] | None = None,
        stream: bool = False,
        ...
    ) -> AgentResponse | ResponseStream[AgentResponseUpdate, AgentResponse]: ...
```

The pipeline executes middleware in order, passing `AgentContext` through the chain.
Calling `call_next()` inside a middleware passes control to the next layer. Setting
`context.result` before or without calling `call_next()` short-circuits the rest of the
chain.

### `ChatMiddlewareLayer`

`ChatMiddlewareLayer` wraps the list of `ChatMiddleware` instances for the raw LLM call.

```python
class ChatMiddlewareLayer(Generic[OptionsCoT]):
    """Composable pipeline over a list of ChatMiddleware."""

    def __init__(
        self,
        *,
        middleware: Sequence[ChatMiddleware],
        client: SupportsChatGetResponse[OptionsCoT],
    ) -> None: ...

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: OptionsCoT | ChatOptions[Any] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]: ...
```

### Execution order

```
agent.run("…")
  │
  ▼
AgentMiddlewareLayer.get_response()
  ├── AgentMiddleware[0].process(ctx, call_next)  ← outermost, runs first
  ├── AgentMiddleware[1].process(ctx, call_next)
  │     │
  │     ▼
  │   FunctionInvocationLayer.get_response()   ← tool loop (max_iterations rounds)
  │     │
  │     ▼
  │   ChatMiddlewareLayer.get_response()
  │     ├── ChatMiddleware[0].process(ctx, call_next)
  │     ├── ChatMiddleware[1].process(ctx, call_next)
  │     │     │
  │     │     ▼
  │     │   BaseChatClient._inner_get_response()   ← actual LLM call
```

### Observing the pipeline (read-only)

```python
from agent_framework.openai import OpenAIChatClient
from agent_framework._middleware import MiddlewareType

client = OpenAIChatClient()

# Inspect attached middleware via the client's internal layers
agent_layer = getattr(client, "_agent_middleware_layer", None)
if agent_layer:
    for mw in agent_layer._middleware:
        print(type(mw).__name__)

# MiddlewareType as an enum value
print(MiddlewareType.AGENT)     # "agent"
print(MiddlewareType.CHAT)      # "chat"
print(MiddlewareType.FUNCTION)  # "function"
```

### Short-circuiting the pipeline with `MiddlewareTermination`

```python
from agent_framework import AgentMiddleware, AgentContext
from agent_framework._middleware import MiddlewareTermination

class RateLimiterMiddleware(AgentMiddleware):
    """Deny requests when a rate limit is exceeded."""

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        self._max = max_requests_per_minute
        self._count = 0

    async def process(self, context: AgentContext, call_next) -> None:
        self._count += 1
        if self._count > self._max:
            # Terminate the entire pipeline immediately — no LLM call is made.
            # The agent.run() caller receives the provided result.
            from agent_framework import AgentResponse, Message
            raise MiddlewareTermination(
                "Rate limit exceeded.",
                result=AgentResponse(
                    messages=[Message("assistant", ["Rate limit exceeded. Please wait."])],
                ),
            )
        await call_next()
```

---

## Summary table

| # | Class group | Module | Key use case |
|---|---|---|---|
| 1 | `ExperimentalFeature` + `@experimental` | `_feature_stage` | Know which APIs are unstable; suppress warnings in tests |
| 2 | `WorkflowRunState` + `WorkflowErrorDetails` | `_workflows/_events` | Monitor live workflow state; structured error capture |
| 3 | `WorkflowExecutor` + sub-workflow messages | `_workflows/_workflow_executor` | Hierarchical workflow composition + HITL across boundaries |
| 4 | `AgentResponse` + `AgentResponseUpdate` + `ContinuationToken` | `_types` | Consume agent outputs; stream tokens; resume background ops |
| 5 | `BaseEmbeddingClient` + embedding family | `_clients` / `_types` | Custom embedding providers + RAG pipelines |
| 6 | `FunctionInvocationConfiguration` | `_tools` | Cap LLM round-trips and total tool calls per request |
| 7 | `ClassSkill` + `SkillFrontmatter` + `FileSkillsSource` + `SkillsProvider` | `_skills` | Production-grade progressive-disclosure knowledge injection |
| 8 | `Annotation` + `TextSpanRegion` | `_types` | Read grounding citations from Bing / file-search responses |
| 9 | Capability protocols | `_clients` | Adaptive tool configuration across providers |
| 10 | `MiddlewareType` + layer classes | `_middleware` | Understand pipeline ordering; build custom clients |

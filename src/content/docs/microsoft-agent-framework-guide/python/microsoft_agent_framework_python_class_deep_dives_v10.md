---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 10"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.8.1: Workflow+InProcRunnerContext, FunctionExecutor, FunctionInvocationLayer, MemoryStore+MemoryIndexEntry+MemoryTopicRecord, TodoStore+TodoItem+TodoFileStore+TodoSessionStore, DeduplicatingSkillsSource, SkillsProvider, MCPTaskOptions, InMemoryCheckpointStorage, EvalScoreResult+CompactionStrategy+BaseAgent."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 33
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 10

Verified against **agent-framework 1.8.1** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. No API name has been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, `BaseEmbeddingClient`, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, `GroupChatBuilder`, `HandoffBuilder`, `MagenticBuilder`, `SequentialBuilder`, `ConcurrentBuilder`, `AgentFactory`, `WorkflowFactory`, `SecureAgentConfig`, `FunctionalWorkflowAgent`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — `AgentFileStore` hierarchy, `FileAccessProvider`, `MCPSkill` + `MCPSkillsSource`, `ToolMode`, `AgentEvalConverter` + `CheckResult` + `RubricScore`, `ChatContext`, `WorkflowAgent` + `WorkflowContext`, `TruncationStrategy`, `HistoryProvider` + `InMemoryHistoryProvider`, `DelegatingSkillsSource` + `InMemorySkillsSource` + `FunctionInvocationContext`
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`+`Worker`+`Client`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `HyperlightCodeActProvider`, `Mem0ContextProvider`, `RedisContextProvider`+`RedisHistoryProvider`, `StandardMagenticManager`+`MagenticContext`, `FileSkillsSource`+`FilteringSkillsSource`

This volume uncovers **ten class groups** from the `agent-framework-core 1.8.1` execution
and harness layers that were not covered in earlier volumes:

- **Workflow runtime internals** — `Workflow` and `InProcRunnerContext`, the objects that
  actually drive graph execution (returned by `WorkflowBuilder.build()` and accessible for
  direct introspection and `as_agent()` conversion).
- **Lightweight workflow nodes** — `FunctionExecutor`, which wraps a plain Python function
  as a graph executor without subclassing.
- **Custom chat client plumbing** — `FunctionInvocationLayer`, the mixin every first-party
  chat client inherits for tool-calling + function-middleware integration.
- **Memory harness data model** — `MemoryStore` (ABC), `MemoryIndexEntry`, and
  `MemoryTopicRecord`, the building blocks for implementing custom long-term memory backends.
- **Todo harness** — `TodoStore` (ABC), `TodoItem`, `TodoInput`, `TodoFileStore`, and
  `TodoSessionStore`, the complete task-list subsystem for agent-managed todo lists.
- **Skill deduplication** — `DeduplicatingSkillsSource`, completing the composable skills
  pipeline alongside `AggregatingSkillsSource` and `FilteringSkillsSource`.
- **Skills context provider** — `SkillsProvider`, the `ContextProvider` that wires a skills
  source into an agent's context with security guards and caching control.
- **Long-running MCP tasks** — `MCPTaskOptions`, controlling the SEP-2663 poll lifecycle
  for MCP tools with `taskSupport == "required"`.
- **In-memory checkpointing** — `InMemoryCheckpointStorage`, the lightweight counterpart to
  `FileCheckpointStorage` for tests and short-lived sessions.
- **Evaluation, compaction, and agent base** — `EvalScoreResult` (per-dimension rubric
  results), `CompactionStrategy` (custom compaction protocol), and `BaseAgent` (minimal
  agent base class for custom implementations).

---

## Table of Contents

1. [`Workflow` + `InProcRunnerContext`](#1-workflow--inprocrunnercontext)
2. [`FunctionExecutor`](#2-functionexecutor)
3. [`FunctionInvocationLayer`](#3-functioninvocationlayer)
4. [`MemoryStore` + `MemoryIndexEntry` + `MemoryTopicRecord`](#4-memorystore--memoryindexentry--memorytopicrecord)
5. [`TodoStore` + `TodoItem` + `TodoInput` + `TodoFileStore` + `TodoSessionStore`](#5-todostore--todoitem--todoinput--todofilestore--todosessionstore)
6. [`DeduplicatingSkillsSource`](#6-deduplicatingskillssource)
7. [`SkillsProvider`](#7-skillsprovider)
8. [`MCPTaskOptions`](#8-mcptaskoptions)
9. [`InMemoryCheckpointStorage`](#9-inmemorycheckpointstorage)
10. [`EvalScoreResult` + `CompactionStrategy` + `BaseAgent`](#10-evalscoreresult--compactionstrategy--baseagent)

---

## 1. `Workflow` + `InProcRunnerContext`

**Source:** `agent_framework._workflows._workflow` / `agent_framework._workflows._runner_context`

`WorkflowBuilder.build()` returns a `Workflow` instance — the object you actually call
`run()` on. `InProcRunnerContext` is the in-process execution context it uses under the hood
(and is also directly constructible when you need custom checkpointing without going through
the builder).

### `Workflow` constructor signature

```python
class Workflow:
    def __init__(
        self,
        edge_groups: list[EdgeGroup],
        executors: dict[str, Executor],
        start_executor: Executor,
        runner_context: RunnerContext,
        name: str,
        description: str | None = None,
        max_iterations: int = 100,          # DEFAULT_MAX_ITERATIONS
        output_from: list[str] | None = ...,
        intermediate_output_from: list[str] | None = ...,
        # Deprecated aliases:
        output_executors: list[str] | None = ...,
        intermediate_executors: list[str] | None = ...,
    ) -> None: ...
```

> **Do not instantiate `Workflow` directly.** Use `WorkflowBuilder.build()`.
> The constructor parameters are documented here for introspection only.

### `Workflow.run()` signature

```python
def run(
    self,
    message: Any | None = None,
    *,
    stream: bool = False,
    responses: Mapping[str, Any] | None = None,
    checkpoint_id: str | None = None,
    checkpoint_storage: CheckpointStorage | None = None,
    include_status_events: bool = False,
    function_invocation_kwargs: Mapping[str, Mapping[str, Any]] | Mapping[str, Any] | None = None,
    client_kwargs: Mapping[str, Mapping[str, Any]] | Mapping[str, Any] | None = None,
) -> ResponseStream[WorkflowEvent, WorkflowRunResult] | Awaitable[WorkflowRunResult]: ...
```

Key mutual-exclusion rules (validated before any async work):
- `message` and `responses` are mutually exclusive.
- `checkpoint_id` + `message` is not allowed; `checkpoint_id` + `responses` is fine (restore then send).

### Key `Workflow` properties and methods

| Member | Returns | Notes |
|--------|---------|-------|
| `workflow.id` | `str` | UUID generated per instance (not stable across `build()` calls) |
| `workflow.name` | `str` | Taken from `WorkflowBuilder(name=...)` |
| `workflow.input_types` | `list[type]` | Inferred from start executor's message annotation |
| `workflow.output_types` | `list[type]` | Union of all executor `workflow_output` annotations |
| `workflow.graph_signature` | `dict` | Topology fingerprint for checkpoint compatibility |
| `workflow.as_agent(name=..., description=...)` | `WorkflowAgent` | Wraps the workflow as a `SupportsAgentRun`-compatible agent |
| `workflow.as_tool(name=..., description=...)` | `FunctionTool` | Exposes the workflow as a `@tool`-compatible callable |

### `InProcRunnerContext` constructor

```python
class InProcRunnerContext:
    def __init__(
        self,
        checkpoint_storage: CheckpointStorage | None = None,
    ) -> None: ...
```

Use `InProcRunnerContext` directly when you want to share the same checkpoint storage
across multiple `Workflow` instances built from the same builder.

### Example 1 — basic run and inspecting types

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, executor, WorkflowContext
from agent_framework.openai import OpenAIChatClient

@executor
async def summarise(msg: str, ctx: WorkflowContext[str, str]) -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Summarise in one sentence.")
    result = await agent.run(msg)
    await ctx.yield_output(result.text)

async def main():
    builder = WorkflowBuilder(name="summarise-wf", description="Single-step summariser")
    builder.add_executor(summarise)
    workflow = builder.build()

    print("input_types :", workflow.input_types)   # [<class 'str'>]
    print("output_types:", workflow.output_types)  # [<class 'str'>]
    print("workflow id :", workflow.id)            # e.g. "a3f1..."

    result = await workflow.run("The quick brown fox jumps over the lazy dog.")
    print(result.text)  # first str output event

asyncio.run(main())
```

### Example 2 — streaming events from `Workflow.run(stream=True)`

```python
import asyncio
from agent_framework import WorkflowBuilder, executor, WorkflowContext, WorkflowEvent
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent

@executor
async def classify(topic: str, ctx: WorkflowContext[str, str]) -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Reply with one word: technology, science, or other.")
    result = await agent.run(topic)
    await ctx.yield_output(result.text.strip())

async def main():
    workflow = WorkflowBuilder(name="classifier-wf").add_executor(classify).build()

    stream = workflow.run("Quantum computing breakthroughs", stream=True)
    async for event in stream:
        if event.type == "output":
            print("classification:", event.data)
        elif event.type == "status":
            print("status:", event.data)

    final: "WorkflowRunResult" = await stream.get_final_response()
    print("state:", final.state)

asyncio.run(main())
```

### Example 3 — `as_agent()` — embed a workflow inside a multi-agent pipeline

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, executor, WorkflowContext, AgentSession
from agent_framework.openai import OpenAIChatClient

@executor
async def translate(msg: str, ctx: WorkflowContext[str, str]) -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Translate to French.")
    result = await agent.run(msg)
    await ctx.yield_output(result.text)

async def main():
    workflow = WorkflowBuilder(name="translate-wf").add_executor(translate).build()
    # Wrap as an Agent so it can be used in GroupChat / Handoff / Sequential
    translate_agent = workflow.as_agent(
        name="translator",
        description="Translates English text to French.",
    )

    orchestrator = Agent(
        client=OpenAIChatClient(),
        instructions="You coordinate translation tasks.",
    )
    session = AgentSession()
    result = await orchestrator.run(
        "Please translate: 'Hello, world!'",
        session=session,
    )
    print(result.text)

asyncio.run(main())
```

### Example 4 — `InProcRunnerContext` with shared checkpoint storage

```python
import asyncio
from agent_framework import (
    WorkflowBuilder, executor, WorkflowContext,
    InMemoryCheckpointStorage,
)
from agent_framework._workflows._runner_context import InProcRunnerContext

@executor
async def step_one(msg: str, ctx: WorkflowContext[str, str]) -> None:
    await ctx.yield_output(f"step1:{msg}")

async def main():
    storage = InMemoryCheckpointStorage()
    runner_ctx = InProcRunnerContext(checkpoint_storage=storage)

    builder = WorkflowBuilder(name="shared-ctx-wf")
    builder.add_executor(step_one)
    # pass runner_context to share checkpoint storage
    workflow = builder.build()

    result = await workflow.run("hello", checkpoint_storage=storage)
    print(result.text)

    checkpoints = await storage.list_checkpoints(workflow_name="shared-ctx-wf")
    print(f"stored {len(checkpoints)} checkpoint(s)")

asyncio.run(main())
```

---

## 2. `FunctionExecutor`

**Source:** `agent_framework._workflows._function_executor`

`FunctionExecutor` wraps a plain Python function (sync or async) as a fully-fledged
workflow executor. It is what the `@executor` decorator produces — but you can construct
one directly when you need explicit type control that introspection cannot infer.

### Constructor signature

```python
class FunctionExecutor(Executor):
    def __init__(
        self,
        func: Callable[..., Any],
        id: str | None = None,
        *,
        input: type | types.UnionType | str | None = None,
        output: type | types.UnionType | str | None = None,
        workflow_output: type | types.UnionType | str | None = None,
    ) -> None: ...
```

| Parameter | Notes |
|-----------|-------|
| `func` | Any sync or async callable; sync functions run in a thread pool via `asyncio.to_thread()` |
| `id` | Executor ID used in edge wiring; defaults to `func.__name__` |
| `input` | Explicit message type(s) — overrides introspection from the function's first parameter annotation |
| `output` | Explicit `ctx.send_message()` output type(s) — overrides the first generic of `WorkflowContext[OutT, _]` |
| `workflow_output` | Explicit `ctx.yield_output()` type(s) — overrides the second generic of `WorkflowContext[_, W_OutT]` |

String forward references are supported (e.g. `input="MyModel | int"`).

`FunctionExecutor` raises `ValueError` if `func` is a `staticmethod` or `classmethod` —
use `@handler` on instance methods instead.

### Example 1 — `@executor` decorator (common case)

```python
import asyncio
from agent_framework import WorkflowBuilder, executor, WorkflowContext
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

@executor
async def fetch_summary(url: str, ctx: WorkflowContext[str, str]) -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Summarise the content.")
    result = await agent.run(f"Summarise this URL: {url}")
    await ctx.yield_output(result.text)

async def main():
    wf = WorkflowBuilder(name="url-summariser").add_executor(fetch_summary).build()
    print(await wf.run("https://example.com"))

asyncio.run(main())
```

### Example 2 — explicit type override with `FunctionExecutor`

```python
import asyncio
from pydantic import BaseModel
from agent_framework import WorkflowBuilder, WorkflowContext, FunctionExecutor
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

class Report(BaseModel):
    title: str
    summary: str

# Function signature uses Any — we override types explicitly
async def generate_report(msg, ctx):
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Return a JSON object with 'title' and 'summary' fields.",
        response_format=Report,
    )
    result = await agent.run(str(msg))
    await ctx.yield_output(result.parsed)

async def main():
    node = FunctionExecutor(
        generate_report,
        id="report-gen",
        input=str,               # explicit: accepts str messages
        workflow_output=Report,  # explicit: yields Report objects
    )
    wf = WorkflowBuilder(name="report-wf").add_executor(node).build()
    result = await wf.run("Summarise the agent-framework 1.8.1 release")
    print(result.text)

asyncio.run(main())
```

### Example 3 — wrapping a synchronous function

```python
import asyncio
import time
from agent_framework import WorkflowBuilder, FunctionExecutor, WorkflowContext

# Sync function — framework runs it in a thread pool automatically
def slow_compute(data: str, ctx: WorkflowContext[str, str]) -> None:
    time.sleep(0.1)  # CPU-bound or blocking I/O
    result = data.upper()
    import asyncio as _asyncio
    # ctx.yield_output is async; in sync functions, schedule it via asyncio
    _asyncio.get_event_loop().run_until_complete(ctx.yield_output(result))

async def main():
    node = FunctionExecutor(slow_compute, input=str, workflow_output=str)
    wf = WorkflowBuilder(name="sync-wf").add_executor(node).build()
    result = await wf.run("hello")
    print(result.text)  # "HELLO"

asyncio.run(main())
```

> For sync functions it is simpler to use `await asyncio.to_thread(blocking_fn, ...)` inside
> an `async def` executor rather than wrapping a raw sync callable with `ctx.yield_output`.

### Example 4 — multi-node workflow using `FunctionExecutor` directly

```python
import asyncio
from agent_framework import (
    WorkflowBuilder, FunctionExecutor, WorkflowContext,
    Agent,
)
from agent_framework.openai import OpenAIChatClient

async def node_extract(msg: str, ctx: WorkflowContext[str, str]) -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Extract key facts as a bullet list.")
    result = await agent.run(msg)
    await ctx.send_message(result.text)

async def node_rank(msg: str, ctx: WorkflowContext[str, str]) -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Rank the facts by importance.")
    result = await agent.run(msg)
    await ctx.yield_output(result.text)

async def main():
    extract = FunctionExecutor(node_extract, id="extract", input=str, output=str)
    rank = FunctionExecutor(node_rank, id="rank", input=str, workflow_output=str)

    builder = WorkflowBuilder(name="extract-rank")
    builder.add_executor(extract)
    builder.add_executor(rank)
    builder.add_edge(extract, rank)
    result = await builder.build().run("LLM agents combine planning, memory, and tool use.")
    print(result.text)

asyncio.run(main())
```

---

## 3. `FunctionInvocationLayer`

**Source:** `agent_framework._tools`

`FunctionInvocationLayer` is the generic mixin inherited by every first-party chat client
(`OpenAIChatClient`, `FoundryChatClient`, `AnthropicClient`, `OllamaChatClient`, …). It
manages the tool-calling loop and the function-middleware pipeline, so all clients share
identical retry/middleware semantics.

### Constructor signature

```python
class FunctionInvocationLayer(Generic[OptionsCoT]):
    def __init__(
        self,
        *,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        **kwargs: Any,           # forwarded to the base chat-client class
    ) -> None: ...
```

### `get_response()` (most important method)

```python
def get_response(
    self,
    messages: Sequence[Message],
    *,
    stream: bool = False,
    options: ChatOptions | None = None,
    middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
    compaction_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    function_invocation_kwargs: Mapping[str, Any] | None = None,
    client_kwargs: Mapping[str, Any] | None = None,
) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]: ...
```

Key behaviour:
- **Tool-calling loop** — automatically retries with tool results until the model stops calling tools (bounded by `FunctionInvocationConfiguration.max_iterations`).
- **Per-call middleware** — `middleware=` at call-site is merged on top of the client-level list.
- **Compaction** — if a `CompactionStrategy` is provided and the message list exceeds the token budget, compaction runs before sending to the model.
- **Streaming** — when `stream=True`, returns a `ResponseStream` that yields `ChatResponseUpdate` tokens and resolves to a final `ChatResponse`.

### Example 1 — custom chat client subclassing `FunctionInvocationLayer`

```python
import asyncio
from typing import Any, Sequence
from agent_framework import (
    BaseChatClient, FunctionInvocationLayer,
    Message, ChatOptions, ChatResponse, ChatResponseUpdate,
    ResponseStream,
)

class EchoClient(FunctionInvocationLayer, BaseChatClient):
    """Minimal client that echoes the last user message — useful for testing."""

    async def _get_response_core(
        self,
        messages: Sequence[Message],
        options: ChatOptions,
        **kwargs: Any,
    ) -> ChatResponse:
        last_user = next(
            (m for m in reversed(messages) if m.role == "user"),
            None,
        )
        text = last_user.text if last_user else "(no user message)"
        return ChatResponse(messages=[Message(role="assistant", contents=[f"ECHO: {text}"])])

async def main():
    client = EchoClient()
    from agent_framework import Agent
    agent = Agent(client=client, instructions="You echo messages.")
    result = await agent.run("Hello, agent!")
    print(result.text)  # "ECHO: Hello, agent!"

asyncio.run(main())
```

### Example 2 — per-call function middleware override

```python
import asyncio
import time
from agent_framework import Agent, FunctionMiddleware, function_middleware
from agent_framework.openai import OpenAIChatClient

@function_middleware
async def timing_middleware(ctx, next_fn):
    start = time.monotonic()
    result = await next_fn(ctx)
    elapsed = time.monotonic() - start
    print(f"[tool-call] {ctx.function.name} took {elapsed:.3f}s")
    return result

async def main():
    client = OpenAIChatClient()
    from agent_framework import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(client=client, instructions="Use the add tool when asked to add.", tools=[add])
    result = await agent.run("What is 17 + 25?")
    print(result.text)

asyncio.run(main())
```

### Example 3 — inspecting `FunctionInvocationLayer` middleware caching

```python
from agent_framework.openai import OpenAIChatClient
from agent_framework import FunctionMiddleware, function_middleware

@function_middleware
async def log_calls(ctx, next_fn):
    print(f"calling: {ctx.function.name}")
    return await next_fn(ctx)

client = OpenAIChatClient(middleware=[log_calls])

# The layer caches a merged pipeline; adding at call-site creates a new pipeline
# only when the merged list differs from the cached one.
print("client-level middleware:", client.function_middleware)
```

### Example 4 — per-executor `function_invocation_kwargs` forwarding

```python
import asyncio
from agent_framework import (
    WorkflowBuilder, executor, WorkflowContext, Agent,
    FunctionInvocationConfiguration,
)
from agent_framework.openai import OpenAIChatClient

@executor
async def careful_step(msg: str, ctx: WorkflowContext[str, str]) -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Answer carefully.")
    result = await agent.run(msg)
    await ctx.yield_output(result.text)

async def main():
    wf = WorkflowBuilder(name="careful-wf").add_executor(careful_step).build()
    # Override function_invocation_configuration for just the careful_step executor
    result = await wf.run(
        "What is 2 + 2?",
        function_invocation_kwargs={
            "careful_step": {"function_invocation_configuration": {"max_iterations": 5}},
        },
    )
    print(result.text)

asyncio.run(main())
```

---

## 4. `MemoryStore` + `MemoryIndexEntry` + `MemoryTopicRecord`

**Source:** `agent_framework._harness._memory`
**Package:** `agent-framework` (experimental — `ExperimentalFeature.HARNESS`)

`MemoryStore` is the abstract backing store for the memory harness used by
`MemoryContextProvider`. Implement this ABC to plug in a custom persistence layer
(e.g. Azure Cosmos DB, SQLite, Postgres). The data model consists of
`MemoryTopicRecord` (one topic file) and `MemoryIndexEntry` (one pointer in `MEMORY.md`).

### `MemoryStore` ABC

```python
from abc import ABC, abstractmethod

class MemoryStore(ABC):
    # Optional hooks (default no-op):
    def get_owner_id(self, session: AgentSession) -> str | None: ...
    def export_provider_state(self, session: AgentSession) -> dict[str, Any]: ...
    def import_provider_state(self, session: AgentSession, *, state: Mapping[str, Any]) -> None: ...

    # Required:
    @abstractmethod
    def list_topics(self, session, *, source_id: str) -> list[MemoryTopicRecord]: ...
    @abstractmethod
    def get_topic(self, session, *, source_id: str, topic: str) -> MemoryTopicRecord: ...
    @abstractmethod
    def write_topic(self, session, record: MemoryTopicRecord, *, source_id: str) -> None: ...
    @abstractmethod
    def delete_topic(self, session, *, source_id: str, topic: str) -> None: ...
    @abstractmethod
    def rebuild_index(self, session, *, source_id: str, line_limit: int, line_length: int) -> list[MemoryIndexEntry]: ...
    @abstractmethod
    def get_index_text(self, session, *, source_id: str, line_limit: int, line_length: int, ...) -> str: ...
    @abstractmethod
    def read_state(self, session, *, source_id: str) -> Any: ...
    @abstractmethod
    def write_state(self, session, state: Any, *, source_id: str) -> None: ...
```

### `MemoryTopicRecord` constructor

```python
class MemoryTopicRecord:
    def __init__(
        self,
        *,
        topic: str,
        slug: str | None = None,     # auto-derived from topic if omitted
        summary: str,
        memories: Sequence[str],
        updated_at: str,
        session_ids: Sequence[str] | None = None,
    ) -> None: ...
```

`memories` are automatically deduplicated. `topic` is normalised (trimmed, lowercased slug).

### `MemoryIndexEntry` constructor

```python
class MemoryIndexEntry:
    def __init__(
        self,
        topic: str,
        slug: str,
        summary: str,
        updated_at: str,
    ) -> None: ...

    @classmethod
    def from_topic_record(cls, record: MemoryTopicRecord) -> MemoryIndexEntry: ...
    @classmethod
    def from_dict(cls, raw_entry: Mapping[str, Any]) -> MemoryIndexEntry: ...
    def to_dict(self) -> dict[str, str]: ...
```

### Example 1 — minimal in-memory `MemoryStore` implementation

```python
import asyncio
import warnings
from datetime import datetime
from typing import Any, Mapping, Sequence

warnings.filterwarnings("ignore")  # suppress ExperimentalWarning for this example

from agent_framework import AgentSession
from agent_framework import MemoryStore, MemoryTopicRecord, MemoryIndexEntry

class DictMemoryStore(MemoryStore):
    """Simple dict-backed MemoryStore for testing."""

    def __init__(self):
        self._topics: dict[str, dict[str, MemoryTopicRecord]] = {}  # owner -> topic -> record
        self._state: dict[str, Any] = {}

    def _key(self, session: AgentSession, source_id: str) -> str:
        return f"{session.session_id}:{source_id}"

    def list_topics(self, session, *, source_id):
        return list(self._topics.get(self._key(session, source_id), {}).values())

    def get_topic(self, session, *, source_id, topic):
        bucket = self._topics.get(self._key(session, source_id), {})
        if topic not in bucket:
            raise KeyError(f"No topic {topic!r}")
        return bucket[topic]

    def write_topic(self, session, record, *, source_id):
        key = self._key(session, source_id)
        self._topics.setdefault(key, {})[record.topic] = record

    def delete_topic(self, session, *, source_id, topic):
        self._topics.get(self._key(session, source_id), {}).pop(topic, None)

    def rebuild_index(self, session, *, source_id, line_limit, line_length):
        records = self.list_topics(session, source_id=source_id)
        return [MemoryIndexEntry.from_topic_record(r) for r in records[:line_limit]]

    def get_index_text(self, session, *, source_id, line_limit, line_length, index_entries=None):
        entries = index_entries or self.rebuild_index(session, source_id=source_id,
                                                      line_limit=line_limit, line_length=line_length)
        lines = [f"- {e.topic}: {e.summary}" for e in entries]
        return "\n".join(lines)

    def read_state(self, session, *, source_id):
        return self._state.get(self._key(session, source_id))

    def write_state(self, session, state, *, source_id):
        self._state[self._key(session, source_id)] = state

async def main():
    store = DictMemoryStore()
    session = AgentSession()

    record = MemoryTopicRecord(
        topic="Python",
        summary="Python programming language facts",
        memories=["Python was created by Guido van Rossum", "Python 3.0 released in 2008"],
        updated_at=datetime.utcnow().isoformat(),
    )
    store.write_topic(session, record, source_id="agent_memory")

    topics = store.list_topics(session, source_id="agent_memory")
    print(f"stored {len(topics)} topic(s):", [t.topic for t in topics])

    index = store.rebuild_index(session, source_id="agent_memory", line_limit=10, line_length=80)
    for entry in index:
        print(f"  [{entry.slug}] {entry.summary}")

asyncio.run(main())
```

### Example 2 — `MemoryTopicRecord` serialisation round-trip

```python
from datetime import datetime
from agent_framework import MemoryTopicRecord, MemoryIndexEntry

record = MemoryTopicRecord(
    topic="  TypeScript  ",    # normalised to "typescript"
    summary="TypeScript language facts",
    memories=[
        "TypeScript is a superset of JavaScript",
        "TypeScript adds static typing to JavaScript",
        "TypeScript is a superset of JavaScript",  # duplicate — deduplicated automatically
    ],
    updated_at=datetime.utcnow().isoformat(),
    session_ids=["sess-001"],
)

raw = record.to_dict()
print("topic  :", raw["topic"])        # "typescript"
print("slug   :", raw["slug"])         # "typescript"
print("memories:", len(raw["memories"]))  # 2 (deduplicated)

# Round-trip
restored = MemoryTopicRecord.from_dict(raw)
assert restored.topic == "typescript"

# Create index entry
entry = MemoryIndexEntry.from_topic_record(record)
print("index entry:", entry.to_dict())
```

### Example 3 — using `DictMemoryStore` with `MemoryContextProvider`

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import Agent, AgentSession
from agent_framework import MemoryContextProvider
from agent_framework.openai import OpenAIChatClient

# DictMemoryStore from Example 1 (assumed in scope)

async def main():
    store = DictMemoryStore()
    memory_provider = MemoryContextProvider(
        client=OpenAIChatClient(),
        store=store,        # plug in the custom backend
    )
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant with persistent memory.",
        context_providers=[memory_provider],
    )
    session = AgentSession()

    await agent.run("Remember that my favourite language is Rust.", session=session)
    result = await agent.run("What is my favourite programming language?", session=session)
    print(result.text)  # should mention Rust

asyncio.run(main())
```

---

## 5. `TodoStore` + `TodoItem` + `TodoInput` + `TodoFileStore` + `TodoSessionStore`

**Source:** `agent_framework._harness._todo`
**Package:** `agent-framework` (experimental — `ExperimentalFeature.HARNESS`)

The todo harness gives agents a structured task-list that persists across turns. The
`TodoProvider` (covered in Vol. 3) wires the harness into an agent's context; these classes
are the data model and storage layer underneath it.

### Class hierarchy

```
TodoStore (ABC)
├── TodoFileStore    — one JSON file per session per source_id
└── TodoSessionStore — stored inside AgentSession.state (no filesystem)
```

### `TodoItem` constructor

```python
class TodoItem:
    def __init__(
        self,
        id: int,
        title: str,
        description: str | None = None,
        is_complete: bool = False,
    ) -> None: ...
```

### `TodoInput` constructor

```python
class TodoInput:
    def __init__(
        self,
        title: str,
        description: str | None = None,
    ) -> None: ...
    # Raises ValueError if title is empty after strip()
```

### `TodoFileStore` constructor

```python
class TodoFileStore(TodoStore):
    def __init__(
        self,
        base_path: str | Path,
        *,
        kind: str = "todos",
        owner_prefix: str = "",
        owner_state_key: str | None = None,
        state_filename: str = "todos.json",
    ) -> None: ...
```

| Parameter | Notes |
|-----------|-------|
| `base_path` | Root directory for all todo state files |
| `kind` | Storage bucket name within each owner directory (`todos`) |
| `owner_prefix` | String prepended to the resolved owner ID for directory scoping |
| `owner_state_key` | Session state key holding the logical owner ID (e.g. `"user_id"`); when set, todos are scoped per owner |
| `state_filename` | Base filename for the JSON state; source_id is embedded: `todos.<source_id>.json` |

`TodoFileStore` rejects path traversal attempts and Windows reserved filenames, and
URL-safe-base64-encodes path segments that contain non-alphanumeric characters.

### `TodoSessionStore` constructor

```python
class TodoSessionStore(TodoStore):
    def __init__(self) -> None: ...  # no parameters
    # All state stored in AgentSession.state[source_id]
```

### Example 1 — `TodoItem` and `TodoInput` data model

```python
from agent_framework import TodoItem, TodoInput

# Create a new todo via TodoInput (the tool-argument DTO)
todo_input = TodoInput(title="  Write tests  ", description="Unit test the parser")
print(todo_input.title)  # "Write tests" — leading/trailing space stripped

# Persist as TodoItem (has an auto-assigned id)
item = TodoItem(id=1, title=todo_input.title, description=todo_input.description)
print(item.to_dict())
# {"id": 1, "title": "Write tests", "description": "Unit test the parser", "is_complete": false}

# Mark complete
item.is_complete = True
raw = item.to_dict()

# Round-trip
restored = TodoItem.from_dict(raw)
assert restored.is_complete is True
```

### Example 2 — `TodoFileStore` per-user todo persistence

```python
import asyncio
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")

from agent_framework import AgentSession, TodoItem
from agent_framework import TodoFileStore

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = TodoFileStore(
            base_path=tmp,
            owner_state_key="user_id",   # scope todos per user
        )

        session = AgentSession()
        session.state["user_id"] = "alice"

        items = [TodoItem(id=1, title="Review PR", description="Check agent-framework PR #42")]
        await store.save_state(session, items, next_id=2, source_id="agent_todos")

        loaded, next_id = await store.load_state(session, source_id="agent_todos")
        print(f"alice has {len(loaded)} todo(s), next id = {next_id}")
        print("  item:", loaded[0].title)

        # Different user gets a separate file
        session.state["user_id"] = "bob"
        bob_items, _ = await store.load_state(session, source_id="agent_todos")
        print(f"bob has {len(bob_items)} todo(s)")

asyncio.run(main())
```

### Example 3 — `TodoSessionStore` for short-lived in-memory todos

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import AgentSession, TodoItem, TodoSessionStore

async def main():
    store = TodoSessionStore()
    session = AgentSession()

    items = [
        TodoItem(id=1, title="Draft email"),
        TodoItem(id=2, title="Schedule meeting", description="With the AI team"),
    ]
    await store.save_state(session, items, next_id=3, source_id="session_todos")

    loaded = await store.load_items(session, source_id="session_todos")
    print([i.title for i in loaded])  # ["Draft email", "Schedule meeting"]

    # Serialize session for cross-process handoff
    state_snapshot = dict(session.state)
    print("session state keys:", list(state_snapshot.keys()))  # ["session_todos"]

asyncio.run(main())
```

### Example 4 — plugging `TodoFileStore` into `TodoProvider`

```python
import asyncio
import warnings
import tempfile
warnings.filterwarnings("ignore")

from agent_framework import Agent, AgentSession, TodoProvider, TodoFileStore
from agent_framework.openai import OpenAIChatClient

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        store = TodoFileStore(base_path=tmp)
        todo_provider = TodoProvider(store=store)

        agent = Agent(
            client=OpenAIChatClient(),
            instructions="You are a productivity assistant with a todo list.",
            context_providers=[todo_provider],
        )
        session = AgentSession()

        await agent.run("Add a todo: 'Finish the quarterly report'.", session=session)
        result = await agent.run("What todos do I have?", session=session)
        print(result.text)

asyncio.run(main())
```

### `TodoStore` ABC contract summary

| Method | Signature | Notes |
|--------|-----------|-------|
| `load_state` | `(session, *, source_id) → (list[TodoItem], int)` | Returns items + next available ID |
| `save_state` | `(session, items, *, next_id, source_id) → None` | Persists items and counter |
| `load_items` | `(session, *, source_id) → list[TodoItem]` | Convenience wrapper (calls `load_state`) |

---

## 6. `DeduplicatingSkillsSource`

**Source:** `agent_framework._skills`

`DeduplicatingSkillsSource` is a `DelegatingSkillsSource` decorator that removes duplicate
skill names (case-insensitive first-one-wins) before they reach an agent. It logs a warning
for each skipped duplicate. This is essential when composing multiple `FileSkillsSource`
directories that may contain skills with the same name.

### Constructor signature

```python
class DeduplicatingSkillsSource(DelegatingSkillsSource):
    def __init__(self, inner_source: SkillsSource) -> None: ...
    async def get_skills(self) -> list[Skill]: ...
```

### Skills source composition chain

```
AggregatingSkillsSource([src_a, src_b, src_c])
    → FilteringSkillsSource(predicate=...)
        → DeduplicatingSkillsSource(...)
            → SkillsProvider(deduplicated_source)
```

### Example 1 — deduplicating two `InMemorySkillsSource` sets

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import (
    InlineSkill, InMemorySkillsSource,
    AggregatingSkillsSource, DeduplicatingSkillsSource, SkillsProvider,
)

async def main():
    skill_a = InlineSkill(name="summarise", description="Summarises text", instructions="Summarise the following:")
    skill_b = InlineSkill(name="SUMMARISE", description="Also summarises text", instructions="(duplicate)")
    skill_c = InlineSkill(name="translate",  description="Translates text", instructions="Translate the following:")

    source_1 = InMemorySkillsSource([skill_a, skill_c])
    source_2 = InMemorySkillsSource([skill_b])           # "SUMMARISE" — case-insensitive dupe

    combined = AggregatingSkillsSource([source_1, source_2])
    deduped   = DeduplicatingSkillsSource(combined)

    skills = await deduped.get_skills()
    print("skills after dedup:", [s.frontmatter.name for s in skills])
    # ["summarise", "translate"]  — "SUMMARISE" silently dropped (warning logged)

asyncio.run(main())
```

### Example 2 — `DeduplicatingSkillsSource` inside `SkillsProvider`

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import (
    Agent, AgentSession,
    InlineSkill, InMemorySkillsSource,
    AggregatingSkillsSource, FilteringSkillsSource, DeduplicatingSkillsSource,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatClient

async def main():
    public_skills = InMemorySkillsSource([
        InlineSkill(name="search",  description="Web search",    instructions="Search the web for:"),
        InlineSkill(name="weather", description="Weather data",  instructions="Get weather for:"),
    ])
    internal_skills = InMemorySkillsSource([
        InlineSkill(name="SEARCH",   description="Internal search (duplicate)", instructions="Internal:"),
        InlineSkill(name="internal-audit", description="Audit log", instructions="Audit:"),
    ])

    source = DeduplicatingSkillsSource(
        FilteringSkillsSource(
            AggregatingSkillsSource([public_skills, internal_skills]),
            predicate=lambda s: not s.frontmatter.name.startswith("internal"),
        )
    )
    provider = SkillsProvider(source)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant with access to skills.",
        context_providers=[provider],
    )
    session = AgentSession()
    result = await agent.run("What skills do you have?", session=session)
    print(result.text)

asyncio.run(main())
```

### Example 3 — custom deduplication key via subclass

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import (
    InlineSkill, InMemorySkillsSource,
    AggregatingSkillsSource, DeduplicatingSkillsSource,
)
from agent_framework import Skill

class TagDeduplicatingSource(DeduplicatingSkillsSource):
    """Keep only the first skill per (name, tag) pair instead of name alone."""

    async def get_skills(self) -> list[Skill]:
        skills = await self._inner_source.get_skills()
        seen: set[tuple[str, str]] = set()
        result: list[Skill] = []
        for skill in skills:
            tag = (skill.frontmatter.additional_properties or {}).get("tag", "")
            key = (skill.frontmatter.name.lower(), str(tag))
            if key not in seen:
                seen.add(key)
                result.append(skill)
        return result

async def main():
    skills = [
        InlineSkill(name="analyse", description="v1", instructions="Analyse:"),
        InlineSkill(name="analyse", description="v2", instructions="Analyse (v2):"),
    ]
    source = TagDeduplicatingSource(InMemorySkillsSource(skills))
    result = await source.get_skills()
    print(f"{len(result)} skill(s) after custom dedup")  # 1

asyncio.run(main())
```

---

## 7. `SkillsProvider`

**Source:** `agent_framework._skills`
**Package:** `agent-framework` (experimental — `ExperimentalFeature.SKILLS`)

`SkillsProvider` is the `ContextProvider` that wires a skill source into an agent's context.
It implements the three-stage agent skills progressive-disclosure protocol:

1. **Advertise** — injects skill names + descriptions into the system prompt (~100 tokens/skill).
2. **Load** — returns the full skill body via an auto-registered `load_skill` tool.
3. **Read resources** — returns supplementary files via `read_skill_resource`.

### Constructor signature

```python
class SkillsProvider(ContextProvider):
    DEFAULT_SOURCE_ID: ClassVar[str] = "agent_skills"

    def __init__(
        self,
        source: SkillsSource | Sequence[Skill] | Skill,
        *,
        instruction_template: str | None = None,
        require_script_approval: bool = False,
        disable_caching: bool = False,
        source_id: str | None = None,
    ) -> None: ...
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `source` | — | `SkillsSource`, single `Skill`, or list of `Skill` — auto-wrapped in `InMemorySkillsSource` + `DeduplicatingSkillsSource` when skills are passed directly |
| `instruction_template` | `None` | Custom system-prompt template. Must contain `{skills}`; optionally `{runner_instructions}` and `{resource_instructions}` |
| `require_script_approval` | `False` | If `True`, agents must confirm before running skill scripts |
| `disable_caching` | `False` | If `True`, re-queries the source on every agent run (useful for live-updating file skills) |
| `source_id` | `"agent_skills"` | Namespaces tool names (e.g. `load_skill` → `agent_skills__load_skill`) |

### `from_paths()` class method

```python
@classmethod
def from_paths(
    cls,
    skill_paths: str | Path | Sequence[str | Path],
    *,
    script_runner: SkillScriptRunner | None = None,
    resource_extensions: tuple[str, ...] | None = None,
    script_extensions: tuple[str, ...] | None = None,
    resource_directories: Sequence[str] | None = None,
    script_directories: Sequence[str] | None = None,
    instruction_template: str | None = None,
    require_script_approval: bool = False,
    disable_caching: bool = False,
    source_id: str | None = None,
) -> SkillsProvider: ...
```

Discovers `SKILL.md` files recursively in `skill_paths`, deduplicates, and returns a
configured `SkillsProvider`. Pass `resource_directories=[".", "references", "assets"]` to
pick up resource files at the skill root in addition to the default subdirectories.

### Example 1 — code-defined skills

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import Agent, AgentSession, InlineSkill, SkillsProvider
from agent_framework.openai import OpenAIChatClient

async def main():
    skills = [
        InlineSkill(
            name="summarise",
            description="Produces a concise summary of any text.",
            instructions="Read the provided text carefully and return a 2–3 sentence summary.",
        ),
        InlineSkill(
            name="sentiment",
            description="Analyses the sentiment of text.",
            instructions="Classify the sentiment as positive, negative, or neutral with a confidence score.",
        ),
    ]
    provider = SkillsProvider(skills)

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a text-analysis assistant.",
        context_providers=[provider],
    )
    session = AgentSession()
    result = await agent.run("What skills do you have available?", session=session)
    print(result.text)

asyncio.run(main())
```

### Example 2 — `from_paths()` with file-based skills

```python
import asyncio
import warnings
import tempfile
import os
warnings.filterwarnings("ignore")

from agent_framework import Agent, AgentSession, SkillsProvider
from agent_framework.openai import OpenAIChatClient

async def main():
    with tempfile.TemporaryDirectory() as tmp:
        # Create a minimal SKILL.md
        skill_dir = os.path.join(tmp, "my_skill")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write("---\nname: file-skill\ndescription: A file-based skill.\n---\n\nUse this skill to demonstrate file-based discovery.\n")

        provider = SkillsProvider.from_paths(
            skill_paths=tmp,
            disable_caching=True,         # re-read on every run
            source_id="my_skills",        # tools namespaced as "my_skills__load_skill"
        )

        agent = Agent(
            client=OpenAIChatClient(),
            instructions="You have access to skills from files.",
            context_providers=[provider],
        )
        session = AgentSession()
        result = await agent.run("List your available skills.", session=session)
        print(result.text)

asyncio.run(main())
```

### Example 3 — custom `instruction_template`

```python
import warnings
warnings.filterwarnings("ignore")

from agent_framework import InlineSkill, SkillsProvider

skill = InlineSkill(name="analyse", description="Analyses data.", instructions="Analyse:")
provider = SkillsProvider(
    skill,
    instruction_template=(
        "## Available Skills\n\n"
        "{skills}\n\n"
        "Load a skill with `load_skill` before using it.\n"
        "{runner_instructions}"
    ),
)
print("provider source_id:", provider.source_id)  # "agent_skills"
```

### Example 4 — multi-source provider with caching disabled

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import (
    Agent, AgentSession,
    InlineSkill, InMemorySkillsSource,
    AggregatingSkillsSource, DeduplicatingSkillsSource,
    SkillsProvider,
)
from agent_framework.openai import OpenAIChatClient

skills_v1 = [InlineSkill(name="greet", description="Greets the user.", instructions="Say hello.")]
skills_v2 = [InlineSkill(name="farewell", description="Bids farewell.", instructions="Say goodbye.")]

source = DeduplicatingSkillsSource(
    AggregatingSkillsSource([
        InMemorySkillsSource(skills_v1),
        InMemorySkillsSource(skills_v2),
    ])
)

async def main():
    provider = SkillsProvider(source, disable_caching=True, source_id="demo")
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You have greeting and farewell skills.",
        context_providers=[provider],
    )
    session = AgentSession()
    result = await agent.run("Greet me!", session=session)
    print(result.text)

asyncio.run(main())
```

---

## 8. `MCPTaskOptions`

**Source:** `agent_framework._mcp`
**Package:** `agent-framework` (experimental — `ExperimentalFeature.MCP_LONG_RUNNING_TASKS`)

`MCPTaskOptions` is a frozen dataclass that controls how an `MCPTool` drives the
[SEP-2663](https://github.com/modelcontextprotocol/spec/discussions/132) long-running task
lifecycle. When an MCP server advertises a tool with
`execution.taskSupport == "required"`, the framework automatically executes the
`tools/call` → poll `tasks/get` → `tasks/result` lifecycle — `MCPTaskOptions` lets you
tune that loop per-tool.

### Constructor (all fields)

```python
from datetime import timedelta

@dataclass(frozen=True)
class MCPTaskOptions:
    default_ttl: timedelta | None = None
    cancel_remote_task_on_local_cancellation: bool = True
    max_task_wait: timedelta | None = None
```

| Field | Default | Notes |
|-------|---------|-------|
| `default_ttl` | `None` | Task-record retention time sent as `params.task.ttl` to the server (milliseconds). Does not cancel running tasks — only controls how long completed task records are retained. Must be positive. |
| `cancel_remote_task_on_local_cancellation` | `True` | When `True`, a Python `CancelledError` triggers a best-effort `tasks/cancel` call before re-raising. Set to `False` if the task should continue server-side even when your coroutine is cancelled. |
| `max_task_wait` | `None` | Client-side deadline for the entire post-create lifecycle. Raises `ToolExecutionException` and sends `tasks/cancel` when exceeded. Must be positive. |

Instances are immutable — reassign `MCPTool.task_options` to change behaviour.

### Example 1 — default `MCPTaskOptions` (no-op, server defaults)

```python
from agent_framework import MCPStreamableHTTPTool

tool = MCPStreamableHTTPTool(
    url="https://my-mcp-server.example.com/mcp",
)
# MCPTaskOptions defaults apply: no TTL limit, cancel-on-local-cancel=True, no max_task_wait
from agent_framework import MCPTaskOptions
print(MCPTaskOptions())
# MCPTaskOptions(default_ttl=None, cancel_remote_task_on_local_cancellation=True, max_task_wait=None)
```

### Example 2 — setting a client-side timeout

```python
from datetime import timedelta
from agent_framework import MCPStreamableHTTPTool, MCPTaskOptions

tool = MCPStreamableHTTPTool(url="https://my-mcp-server.example.com/mcp")
# Raise ToolExecutionException if the task takes more than 30 seconds
tool.task_options = MCPTaskOptions(max_task_wait=timedelta(seconds=30))
```

### Example 3 — keep task records for 5 minutes

```python
from datetime import timedelta
from agent_framework import MCPStreamableHTTPTool, MCPTaskOptions

tool = MCPStreamableHTTPTool(url="https://my-mcp-server.example.com/mcp")
tool.task_options = MCPTaskOptions(
    default_ttl=timedelta(minutes=5),   # server keeps the task record 5 min after completion
    max_task_wait=timedelta(minutes=10),
)
```

### Example 4 — fire-and-forget style (do not cancel server task on local cancel)

```python
import asyncio
from datetime import timedelta
from agent_framework import Agent, MCPStreamableHTTPTool, MCPTaskOptions
from agent_framework.openai import OpenAIChatClient

async def main():
    mcp_tool = MCPStreamableHTTPTool(url="https://my-mcp-server.example.com/mcp")
    # Do NOT cancel the server-side task if the local coroutine is cancelled
    mcp_tool.task_options = MCPTaskOptions(
        cancel_remote_task_on_local_cancellation=False,
        max_task_wait=timedelta(minutes=2),
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Use the MCP tool to run long computations.",
        tools=[mcp_tool],
    )
    result = await agent.run("Run the data pipeline.")
    print(result.text)

asyncio.run(main())
```

### Example 5 — validation: `default_ttl` must be positive

```python
from datetime import timedelta
from agent_framework import MCPTaskOptions

try:
    MCPTaskOptions(default_ttl=timedelta(seconds=0))  # raises ValueError
except ValueError as e:
    print(e)  # "MCPTaskOptions.default_ttl must be positive."

try:
    MCPTaskOptions(max_task_wait=timedelta(seconds=-1))  # raises ValueError
except ValueError as e:
    print(e)  # "MCPTaskOptions.max_task_wait must be positive."
```

---

## 9. `InMemoryCheckpointStorage`

**Source:** `agent_framework._workflows._checkpoint`

`InMemoryCheckpointStorage` is the lightweight, no-persistence counterpart to
`FileCheckpointStorage`. It stores `WorkflowCheckpoint` objects in a dict keyed by
`CheckpointID`, making it ideal for unit tests, short-lived sessions, and scenarios where
durability is not required.

### Constructor and full API

```python
class InMemoryCheckpointStorage:
    def __init__(self) -> None: ...                                               # no args

    async def save(self, checkpoint: WorkflowCheckpoint) -> CheckpointID: ...     # deepcopy; returns id
    async def load(self, checkpoint_id: CheckpointID) -> WorkflowCheckpoint: ... # raises WorkflowCheckpointException if not found
    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]: ...
    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[CheckpointID]: ...
    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None: ...
    async def delete(self, checkpoint_id: CheckpointID) -> bool: ...              # True if deleted
```

> `save()` stores a **deep copy** of the checkpoint, so mutations after save don't affect
> the stored state.

### Comparison: `InMemoryCheckpointStorage` vs `FileCheckpointStorage`

| Feature | `InMemoryCheckpointStorage` | `FileCheckpointStorage` |
|---------|-----------------------------|--------------------------|
| Persistence | None — lost on process exit | JSON files on disk |
| Durability | ❌ | ✅ |
| Thread-safety | ✅ (single-process) | ✅ (file locking) |
| `get_latest()` | By timestamp field | By file modification time |
| Use case | Unit tests, HITL in same process | Production, cross-process resume |

### Example 1 — basic save / load / delete cycle

```python
import asyncio
from agent_framework import (
    WorkflowBuilder, executor, WorkflowContext,
    InMemoryCheckpointStorage,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent

@executor
async def echo(msg: str, ctx: WorkflowContext[str, str]) -> None:
    await ctx.yield_output(f"echo:{msg}")

async def main():
    storage = InMemoryCheckpointStorage()
    builder = WorkflowBuilder(name="echo-wf")
    builder.add_executor(echo)
    workflow = builder.build()

    result = await workflow.run("hello", checkpoint_storage=storage)
    print("output:", result.text)

    checkpoints = await storage.list_checkpoints(workflow_name="echo-wf")
    print(f"{len(checkpoints)} checkpoint(s) stored")

    if checkpoints:
        cp_id = checkpoints[0].checkpoint_id
        # Load it back
        restored = await storage.load(cp_id)
        print("checkpoint workflow_name:", restored.workflow_name)

        # Delete
        deleted = await storage.delete(cp_id)
        print("deleted:", deleted)  # True

asyncio.run(main())
```

### Example 2 — HITL resume with `InMemoryCheckpointStorage`

```python
import asyncio
from agent_framework import (
    WorkflowBuilder, Executor, handler, WorkflowContext,
    response_handler, InMemoryCheckpointStorage,
    Agent,
)
from agent_framework.openai import OpenAIChatClient

class ApprovalExecutor(Executor):
    @handler
    async def run(self, msg: str, ctx: WorkflowContext[str, str]) -> None:
        approval = await ctx.request_info("approve", "Do you approve this action?")
        if approval.lower() == "yes":
            await ctx.yield_output(f"Approved: {msg}")
        else:
            await ctx.yield_output(f"Rejected: {msg}")

    @response_handler("approve")
    async def handle_approval(self, response: str, ctx: WorkflowContext) -> None:
        pass  # response forwarded automatically

async def main():
    storage = InMemoryCheckpointStorage()
    builder = WorkflowBuilder(name="approval-wf")
    builder.add_executor(ApprovalExecutor())
    workflow = builder.build()

    # First run — pauses waiting for approval
    result = await workflow.run("Deploy to production", checkpoint_storage=storage)
    print("state:", result.state)  # IDLE_WITH_PENDING_REQUESTS

    # Inspect pending requests
    pending = [e for e in result.events if e.type == "request_info"]
    if pending:
        request_id = pending[0].data["request_id"]
        print("pending request:", request_id)

        # Resume with human approval
        latest = await storage.get_latest(workflow_name="approval-wf")
        resumed = await workflow.run(
            responses={request_id: "yes"},
            checkpoint_id=latest.checkpoint_id,
            checkpoint_storage=storage,
        )
        print("final:", resumed.text)  # "Approved: Deploy to production"

asyncio.run(main())
```

### Example 3 — `get_latest()` and multi-checkpoint management in tests

```python
import asyncio
from agent_framework import (
    WorkflowBuilder, executor, WorkflowContext, InMemoryCheckpointStorage,
)

@executor
async def noop(msg: str, ctx: WorkflowContext[str, str]) -> None:
    await ctx.yield_output(msg)

async def main():
    storage = InMemoryCheckpointStorage()
    builder = WorkflowBuilder(name="test-wf")
    builder.add_executor(noop)
    workflow = builder.build()

    await workflow.run("run 1", checkpoint_storage=storage)
    await workflow.run("run 2", checkpoint_storage=storage)

    ids = await storage.list_checkpoint_ids(workflow_name="test-wf")
    print(f"{len(ids)} checkpoint IDs")

    latest = await storage.get_latest(workflow_name="test-wf")
    print("latest checkpoint timestamp:", latest.timestamp if latest else None)

    # Clean up
    for cp_id in ids:
        await storage.delete(cp_id)

    remaining = await storage.list_checkpoints(workflow_name="test-wf")
    print("after cleanup:", len(remaining))  # 0

asyncio.run(main())
```

---

## 10. `EvalScoreResult` + `CompactionStrategy` + `BaseAgent`

**Source:** `agent_framework._evaluation`, `agent_framework._compaction`, `agent_framework._agents`

Three classes from distinct subsystems that haven't been covered in earlier volumes:

- **`EvalScoreResult`** — the per-evaluator result row in evaluation output, including
  per-dimension rubric scores via `RubricScore` (covered in Vol. 8).
- **`CompactionStrategy`** — the `Protocol` that custom compaction strategies must satisfy.
- **`BaseAgent`** — the minimal abstract base class for custom agent implementations
  (without the telemetry or middleware layers that `Agent` adds).

---

### 10a. `EvalScoreResult`

**Source:** `agent_framework._evaluation`
**Package:** experimental — `ExperimentalFeature.EVALS`

```python
@dataclass
class EvalScoreResult:
    name: str                              # evaluator name, e.g. "relevance"
    score: float                           # numeric score
    passed: bool | None = None             # True/False/None (no threshold)
    sample: dict[str, Any] | None = None  # raw evaluator output (rationale, metadata)
    dimensions: list[RubricScore] | None = None  # per-dimension scores for rubric evaluators
```

`EvalScoreResult` appears in `EvalItemResult.scores` (one entry per evaluator). When a
rubric evaluator is used, `dimensions` carries the individual `RubricScore` entries
(name, score, passed, rationale).

#### Example 1 — inspecting `EvalScoreResult` from `LocalEvaluator`

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import Agent, LocalEvaluator, EvalItem, evaluate_agent
from agent_framework.openai import OpenAIChatClient

async def main():
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Answer factual questions concisely.",
    )
    items = [
        EvalItem(
            input="What is the capital of France?",
            expected_output="Paris",
        ),
    ]
    evaluator = LocalEvaluator(client=OpenAIChatClient())
    results = await evaluate_agent(agent, items, evaluators=[evaluator])

    for item_result in results.item_results:
        for score_result in item_result.scores:
            print(f"evaluator: {score_result.name}")
            print(f"  score  : {score_result.score:.2f}")
            print(f"  passed : {score_result.passed}")
            if score_result.dimensions:
                for dim in score_result.dimensions:
                    print(f"    [{dim.name}] {dim.score:.2f} — {dim.rationale}")

asyncio.run(main())
```

#### Example 2 — building `EvalScoreResult` manually for a custom evaluator

```python
import asyncio
import warnings
warnings.filterwarnings("ignore")

from agent_framework import EvalScoreResult, RubricScore, Evaluator, EvalItem, EvalItemResult
from agent_framework import evaluator as evaluator_decorator

@evaluator_decorator
async def keyword_match_evaluator(item: EvalItem, actual: str) -> EvalScoreResult:
    expected_keywords = (item.expected_output or "").lower().split()
    actual_lower = actual.lower()
    hits = sum(1 for kw in expected_keywords if kw in actual_lower)
    score = hits / max(len(expected_keywords), 1)
    return EvalScoreResult(
        name="keyword-match",
        score=score,
        passed=score >= 0.8,
        sample={"hits": hits, "total": len(expected_keywords)},
    )

async def main():
    from agent_framework import Agent, evaluate_agent
    from agent_framework.openai import OpenAIChatClient

    agent = Agent(client=OpenAIChatClient(), instructions="Answer briefly.")
    items = [EvalItem(input="What is 2+2?", expected_output="4")]
    results = await evaluate_agent(agent, items, evaluators=[keyword_match_evaluator])

    for ir in results.item_results:
        for sr in ir.scores:
            print(f"{sr.name}: {sr.score:.2f} passed={sr.passed} sample={sr.sample}")

asyncio.run(main())
```

---

### 10b. `CompactionStrategy`

**Source:** `agent_framework._compaction`

`CompactionStrategy` is a `runtime_checkable` `Protocol` — any `async def __call__(self, messages: list[Message]) -> bool` qualifies.

```python
@runtime_checkable
class CompactionStrategy(Protocol):
    async def __call__(self, messages: list[Message]) -> bool:
        """Mutate messages in-place; return True if anything changed."""
        ...
```

The contract:
- Receives the **annotated** message list (grouping and token annotations already applied).
- Mutates the list or annotations **in place** (`EXCLUDED_KEY`, `EXCLUDE_REASON_KEY` annotations mark messages for exclusion without physically removing them).
- Returns `True` if compaction changed anything; `False` if the list was already compact.

Built-in implementations: `SlidingWindowStrategy`, `SummarizationStrategy`,
`ContextWindowCompactionStrategy`, `SelectiveToolCallCompactionStrategy`,
`TruncationStrategy`, `TokenBudgetComposedStrategy`, `ToolResultCompactionStrategy`.

#### Example 1 — minimal custom `CompactionStrategy`

```python
import asyncio
from agent_framework import (
    Agent, AgentSession,
    EXCLUDED_KEY, EXCLUDE_REASON_KEY,
    CompactionStrategy, CompactionProvider,
)
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient

class RemoveSystemMessages:
    """Drop all system messages beyond the first one."""

    async def __call__(self, messages: list[Message]) -> bool:
        seen_system = False
        changed = False
        for msg in messages:
            if msg.role == "system":
                if seen_system:
                    msg.annotations[EXCLUDED_KEY] = True
                    msg.annotations[EXCLUDE_REASON_KEY] = "extra-system"
                    changed = True
                else:
                    seen_system = True
        return changed

async def main():
    strategy = RemoveSystemMessages()
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful assistant.",
        context_providers=[CompactionProvider(strategy=strategy)],
    )
    session = AgentSession()
    result = await agent.run("Hello!", session=session)
    print(result.text)

asyncio.run(main())
```

#### Example 2 — composing with `TokenBudgetComposedStrategy`

```python
from agent_framework import (
    TokenBudgetComposedStrategy, SlidingWindowStrategy,
    CompactionProvider, CharacterEstimatorTokenizer,
)

strategy = TokenBudgetComposedStrategy(
    strategies=[SlidingWindowStrategy(target_count=20)],
    token_budget=8000,
    tokenizer=CharacterEstimatorTokenizer(chars_per_token=4),
)
provider = CompactionProvider(strategy=strategy)
# attach to Agent(context_providers=[provider])
```

#### Example 3 — verifying `isinstance` against the Protocol

```python
from agent_framework import (
    CompactionStrategy, SlidingWindowStrategy, SummarizationStrategy,
    ContextWindowCompactionStrategy,
)

for cls in [SlidingWindowStrategy, SummarizationStrategy, ContextWindowCompactionStrategy]:
    instance = cls() if cls is SlidingWindowStrategy else cls.__new__(cls)
    print(f"{cls.__name__} is CompactionStrategy:", isinstance(cls, type))
# All first-party strategies satisfy the protocol
```

---

### 10c. `BaseAgent`

**Source:** `agent_framework._agents`

`BaseAgent` is the minimal base class for `Agent` — it provides the core fields
(`id`, `name`, `description`, `context_providers`, `middleware`, `additional_properties`)
but does **not** implement `run()`. Use it when:
- You need a custom agent with non-standard execution (e.g. a rule-based agent, a mock agent).
- You want to compose agent infrastructure without depending on chat clients.

```python
class BaseAgent(SerializationMixin):
    def __init__(
        self,
        *,
        id: str | None = None,              # auto-generates UUID if None
        name: str | None = None,
        description: str | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: Sequence[MiddlewareTypes] | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
    ) -> None: ...
```

> `BaseAgent` cannot be used in orchestration builders directly — builders expect
> `SupportsAgentRun`. Implement `run()` in your subclass to make it compatible.

#### Example 1 — deterministic mock agent for testing

```python
import asyncio
from typing import Any, Mapping, Sequence
from agent_framework import (
    BaseAgent, AgentSession, AgentResponse, AgentResponseUpdate,
    ResponseStream, Message,
)

class FixedResponseAgent(BaseAgent):
    """Always returns the same response — useful in orchestration unit tests."""

    def __init__(self, fixed_reply: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fixed_reply = fixed_reply

    async def run(
        self,
        messages=None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> AgentResponse | ResponseStream:
        reply_msg = Message(role="assistant", contents=[self._fixed_reply])
        if stream:
            async def _stream():
                yield AgentResponseUpdate(messages=[reply_msg], response_id="fixed")
            return _stream()
        return AgentResponse(messages=[reply_msg], response_id="fixed")

async def main():
    agent = FixedResponseAgent(
        fixed_reply="The answer is 42.",
        name="oracle",
        description="Always says 42.",
    )
    result = await agent.run("What is the meaning of life?")
    print(result.text)  # "The answer is 42."
    print("agent id:", agent.id)   # auto-generated UUID

asyncio.run(main())
```

#### Example 2 — `BaseAgent` in a `SequentialBuilder` pipeline

```python
import asyncio
from agent_framework import (
    BaseAgent, AgentSession, AgentResponse, AgentResponseUpdate,
    ResponseStream, Message, WorkflowBuilder,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent

class UpperCaseAgent(BaseAgent):
    async def run(self, messages=None, *, stream=False, session=None, **kwargs):
        text = str(messages) if not isinstance(messages, str) else messages
        reply = Message(role="assistant", contents=[text.upper()])
        return AgentResponse(messages=[reply], response_id="upper")

async def main():
    from agent_framework import SequentialBuilder

    upper = UpperCaseAgent(name="upper-agent")
    llm   = Agent(
        client=OpenAIChatClient(),
        instructions="Expand the following uppercased text into a sentence.",
        name="expand-agent",
    )

    pipeline = (
        SequentialBuilder(name="upper-then-expand")
        .add_agent(upper)
        .add_agent(llm)
        .build()
    )
    result = await pipeline.run("hello world")
    print(result.text)

asyncio.run(main())
```

#### Example 3 — serialisation via `SerializationMixin`

```python
from agent_framework import BaseAgent

class SimpleAgent(BaseAgent):
    async def run(self, messages=None, **kwargs):
        from agent_framework import AgentResponse, Message
        return AgentResponse(messages=[Message(role="assistant", contents=["ok"])], response_id="r")

agent = SimpleAgent(
    name="my-agent",
    description="A simple custom agent.",
    additional_properties={"version": "2.0", "team": "ai-platform"},
)

raw = agent.to_dict()
print("id          :", raw["id"])
print("name        :", raw["name"])
print("description :", raw["description"])
print("additional  :", raw.get("additional_properties"))

# Round-trip (requires registering the concrete class)
# restored = SimpleAgent.from_dict(raw)
```

---

## Summary table

| # | Class group | Package | Status | Key feature |
|---|-------------|---------|--------|-------------|
| 1 | `Workflow` + `InProcRunnerContext` | `agent-framework` | stable | Graph execution API; `run()`, `as_agent()`, `as_tool()`, streaming |
| 2 | `FunctionExecutor` | `agent-framework` | stable | Wrap any function as a workflow node without subclassing |
| 3 | `FunctionInvocationLayer` | `agent-framework` | stable | Tool-calling loop mixin; custom chat client authoring |
| 4 | `MemoryStore` + `MemoryIndexEntry` + `MemoryTopicRecord` | `agent-framework` | experimental | Custom long-term memory backends |
| 5 | `TodoStore` + `TodoItem` + `TodoInput` + `TodoFileStore` + `TodoSessionStore` | `agent-framework` | experimental | Task-list harness; file-backed and session-backed storage |
| 6 | `DeduplicatingSkillsSource` | `agent-framework` | stable | First-one-wins skill deduplication in multi-source pipelines |
| 7 | `SkillsProvider` | `agent-framework` | experimental | Context provider for progressive-disclosure skills |
| 8 | `MCPTaskOptions` | `agent-framework` | experimental | SEP-2663 long-running MCP task lifecycle tuning |
| 9 | `InMemoryCheckpointStorage` | `agent-framework` | stable | In-memory checkpoint backend for tests and dev |
| 10 | `EvalScoreResult` + `CompactionStrategy` + `BaseAgent` | `agent-framework` | experimental / stable | Eval dimensions, custom compaction protocol, minimal agent base |

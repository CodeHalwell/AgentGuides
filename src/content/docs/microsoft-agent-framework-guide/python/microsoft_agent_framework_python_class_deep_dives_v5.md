---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 5"
description: "Source-verified deep dives into 10 class groups from agent-framework-core 1.7.0: Executor + @handler + @executor, AgentExecutor + AgentExecutorRequest + AgentExecutorResponse, FanOutEdgeGroup + FanInEdgeGroup + SwitchCaseEdgeGroup, Runner + WorkflowMessage, SessionContext, AgentSession + register_state_type, BaseChatClient + SupportsChatGetResponse, SecretString + load_settings, WorkflowCheckpoint + CheckpointStorage, Exception hierarchy."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 24
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 5

Verified against **agent-framework-core 1.7.0** (installed June 2026). Every constructor signature,
parameter description, and code example was derived from the installed package source at
`/usr/local/lib/python3.11/dist-packages/agent_framework/`. No API name has been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse` + `ChatResponseUpdate`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` data model, `TokenizerProtocol`, `ConversationSplit`

This volume fills gaps across five areas: the **workflow execution engine**, the **agent executor pattern**, **edge routing primitives**, **session state and context pipeline**, and the **exception hierarchy**.

---

## Table of Contents

1. [`Executor` + `@handler` + `@executor`](#1-executor--handler--executor)
2. [`AgentExecutor` + `AgentExecutorRequest` + `AgentExecutorResponse`](#2-agentexecutor--agentexecutorrequest--agentexecutorresponse)
3. [`FanOutEdgeGroup` + `FanInEdgeGroup` + `SwitchCaseEdgeGroup`](#3-fanoutedgegroup--fainedgegroup--switchcaseedgegroup)
4. [`Runner` + `WorkflowMessage`](#4-runner--workflowmessage)
5. [`SessionContext`](#5-sessioncontext)
6. [`AgentSession` + `register_state_type`](#6-agentsession--register_state_type)
7. [`BaseChatClient` + `SupportsChatGetResponse`](#7-basechatclient--supportschatgetresponse)
8. [`SecretString` + `load_settings`](#8-secretstring--load_settings)
9. [`WorkflowCheckpoint` + `CheckpointStorage` + `InMemoryCheckpointStorage`](#9-workflowcheckpoint--checkpointstorage--inmemorycheckpointstorage)
10. [Exception Hierarchy](#10-exception-hierarchy)

---

## 1. `Executor` + `@handler` + `@executor`

**Source:** `agent_framework/_workflows/_executor.py`

`Executor` is the abstract base class for every processing unit in a workflow graph. An executor receives typed messages from upstream edges, processes them in a `@handler` method, and optionally sends new messages downstream or yields workflow-level outputs. The `@executor` decorator wraps a plain async or sync function as a lightweight `Executor` without requiring a class.

### Constructor

```python
Executor(
    id: str,
    *,
    type: str | None = None,         # defaults to class name
    type_: str | None = None,         # alias for type
    defer_discovery: bool = False,    # skip handler auto-discovery at init time
)
```

`id` must be a non-empty string — passing `""` raises `ValueError` immediately. The `type` field (alias `type_`) is used for serialization; if omitted it defaults to the class name.

When `defer_discovery=False` (the default) the constructor inspects the class for `@handler`-decorated methods and builds the dispatch table. If no handlers are found, `ValueError` is raised. Set `defer_discovery=True` only when you need to register handlers programmatically after construction (rare — mostly used internally).

### Key properties

| Property | Type | Description |
|---|---|---|
| `id` | `str` | Unique executor identifier. |
| `type` | `str` | Logical type name (used in serialization). |
| `input_types` | `list[type \| UnionType]` | Message types this executor can handle, one entry per `@handler`. |
| `output_types` | `list[type \| UnionType]` | Types sendable via `ctx.send_message()`, from `WorkflowContext[OutT]` annotations. |
| `workflow_output_types` | `list[type \| UnionType]` | Types yieldable via `ctx.yield_output()`, from `WorkflowContext[OutT, WOutT]` second param. |

### The `@handler` decorator

`@handler` marks a method as the dispatcher for a specific message type. The framework discovers all `@handler` methods during `__init__` and registers them in a type-dispatch table.

**Introspection mode** (default) — types are inferred from the method's first positional parameter annotation and the `WorkflowContext[...]` annotation:

```python
from agent_framework import Executor, WorkflowContext
from agent_framework._workflows._executor import handler

class UpperCaseExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="upper_case")

    @handler
    async def process(self, text: str, ctx: WorkflowContext[str]) -> None:
        # Receives str, sends str downstream
        await ctx.send_message(text.upper())
```

**Explicit mode** — used when types cannot be inferred from annotations (e.g., forward references that cannot be resolved at class-definition time, or dynamic type unions):

```python
from agent_framework._workflows._executor import handler

class FlexibleExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="flex")

    @handler(input=str, output=int, workflow_output=str)
    async def process(self, text, ctx) -> None:
        await ctx.send_message(len(text))      # int downstream
        await ctx.yield_output(text[:10])       # str workflow output
```

**Critical rule:** If you pass ANY of `input=`, `output=`, or `workflow_output=` to `@handler(...)`, ALL type information must come from explicit params. Mixing explicit and annotation-derived types is not allowed — the decorator raises `ValueError` if both are present.

### `WorkflowContext` variants

The `WorkflowContext` annotation on the second parameter of a `@handler` method controls what operations are available inside the handler:

| Annotation | `send_message` | `yield_output` |
|---|---|---|
| `WorkflowContext` (no params) | Not available | Not available |
| `WorkflowContext[OutT]` | Yes — sends `OutT` messages | Not available |
| `WorkflowContext[OutT, WOutT]` | Yes — sends `OutT` messages | Yes — yields `WOutT` workflow outputs |

```python
from agent_framework import Executor, WorkflowContext
from agent_framework._workflows._executor import handler

class SideEffectOnly(Executor):
    """An executor that logs but never sends messages."""

    def __init__(self) -> None:
        super().__init__(id="logger")

    @handler
    async def log(self, msg: str, ctx: WorkflowContext) -> None:
        print(f"[LOG] {msg}")
        # ctx.send_message would raise AttributeError here — not exposed


class DualOutputExecutor(Executor):
    """Sends downstream messages AND emits workflow-level outputs."""

    def __init__(self) -> None:
        super().__init__(id="dual")

    @handler
    async def process(self, msg: str, ctx: WorkflowContext[int, str]) -> None:
        await ctx.send_message(len(msg))   # int goes to next executor
        await ctx.yield_output(msg[:20])    # str appears in WorkflowRunResult.outputs
```

### Multiple `@handler` methods — polymorphic dispatch

An executor can handle more than one input type by declaring multiple `@handler` methods. At runtime, the framework picks the first handler whose type matches `isinstance(message, handler_input_type)`:

```python
class PolymorphicExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="poly")

    @handler
    async def handle_str(self, msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(f"string: {msg}")

    @handler
    async def handle_int(self, msg: int, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(f"int: {msg}")
```

Duplicate handlers for the same type raise `ValueError` at construction time.

### `@executor` — function-based executor

For simple processing steps that do not need class-level state, the `@executor` decorator converts a plain function into an `Executor` instance:

```python
from agent_framework import WorkflowContext
from agent_framework._workflows._executor import executor

# Simplest form — ID defaults to function name
@executor
async def normalize(text: str, ctx: WorkflowContext[str]) -> None:
    await ctx.send_message(text.strip().lower())


# With explicit overrides
@executor(id="word_counter", input=str, output=int)
async def count_words(text: str, ctx: WorkflowContext[int]) -> None:
    await ctx.send_message(len(text.split()))
```

Sync functions are automatically wrapped in `asyncio.get_event_loop().run_in_executor()` so they run in a thread pool without blocking the event loop:

```python
import time

@executor(id="cpu_bound")
def heavy_computation(data: str, ctx: WorkflowContext[str]) -> None:
    time.sleep(1)  # runs in thread pool — does not block async event loop
    ctx.send_message(data.upper())
```

### Checkpoint hooks

Override `on_checkpoint_save` and `on_checkpoint_restore` to persist any custom state your executor accumulates across supersteps:

```python
from typing import Any
from agent_framework import Executor, WorkflowContext
from agent_framework._workflows._executor import handler

class CountingExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="counter")
        self._count = 0

    @handler
    async def count(self, msg: str, ctx: WorkflowContext[int]) -> None:
        self._count += 1
        await ctx.send_message(self._count)

    async def on_checkpoint_save(self) -> dict[str, Any]:
        return {"count": self._count}

    async def on_checkpoint_restore(self, state: dict[str, Any]) -> None:
        self._count = state.get("count", 0)
```

When checkpointing is enabled on the workflow (see `WorkflowBuilder`), the framework calls `on_checkpoint_save()` after each superstep and `on_checkpoint_restore()` when resuming from a checkpoint. If you do not override them, the base class returns `{}` / does nothing, so stateless executors require no extra work.

### Advanced: union types in explicit mode

Explicit mode accepts `str | int` union syntax as well as pipe-unions for Python 3.10+:

```python
@handler(input=str | int, output=bytes)
async def handle_either(self, msg, ctx: WorkflowContext[bytes]) -> None:
    data = msg.encode() if isinstance(msg, str) else msg.to_bytes(4, "big")
    await ctx.send_message(data)
```

String forward references also work in explicit mode:

```python
@handler(input="MyDataClass", output="MyResultClass")
async def process(self, msg, ctx) -> None:
    ...
```

---

## 2. `AgentExecutor` + `AgentExecutorRequest` + `AgentExecutorResponse`

**Source:** `agent_framework/_workflows/_agent_executor.py`

`AgentExecutor` bridges an `Agent` (or any `SupportsAgentRun`) into the workflow graph. It maintains an internal per-executor conversation cache across supersteps and exposes five built-in `@handler` methods to accept different input shapes from upstream executors.

### Constructor

```python
AgentExecutor(
    agent: SupportsAgentRun,
    *,
    session: AgentSession | None = None,           # auto-created if None
    id: str | None = None,                          # defaults to agent.name
    context_mode: Literal["full", "last_agent", "custom"] | None = None,   # default "full"
    context_filter: Callable[[list[Message]], list[Message]] | None = None, # required for "custom"
)
```

The `id` falls back to the agent's `name` attribute when not provided. If neither is present, `ValueError` is raised at construction time.

### Context modes

Context mode controls which prior messages are added to the internal cache when `AgentExecutor` receives an `AgentExecutorResponse` from an upstream executor:

| Mode | What goes into the cache |
|---|---|
| `"full"` (default) | `prior.full_conversation` — the complete prior exchange including all user messages and agent turns |
| `"last_agent"` | `prior.agent_response.messages` — only the most recent agent response turn |
| `"custom"` | `context_filter(prior.full_conversation)` — caller decides what to include |

```python
from agent_framework import Agent, AgentExecutor
from agent_framework.openai import OpenAIChatClient

orchestrator = Agent(client=OpenAIChatClient(), name="orchestrator", instructions="Route requests.")
reviewer = Agent(client=OpenAIChatClient(), name="reviewer", instructions="Review the orchestrator's answer.")

orchestrator_exec = AgentExecutor(orchestrator)

# reviewer only sees the orchestrator's most recent answer, not prior user messages
reviewer_exec = AgentExecutor(reviewer, context_mode="last_agent")

# reviewer uses a custom filter — keep only user messages for re-evaluation
reviewer_exec_custom = AgentExecutor(
    reviewer,
    context_mode="custom",
    context_filter=lambda msgs: [m for m in msgs if m.role == "user"],
)
```

### Built-in handlers

`AgentExecutor` auto-registers five `@handler` methods. The framework dispatches to whichever one matches the incoming message type:

| Handler | Input type | Behavior |
|---|---|---|
| `run` | `AgentExecutorRequest` | Extends cache; runs agent only if `should_respond=True` |
| `from_response` | `AgentExecutorResponse` | Extends cache from prior response; runs agent |
| `from_str` | `str` | Adds plain string to cache as user message; runs agent |
| `from_message` | `Message` | Adds single `Message` to cache; runs agent |
| `from_messages` | `list[str \| Message]` | Adds multiple messages to cache; runs agent |

### `AgentExecutorRequest`

```python
from dataclasses import dataclass
from agent_framework import Message

@dataclass
class AgentExecutorRequest:
    messages: list[Message]
    should_respond: bool = True
```

Use `should_respond=False` to "prime" an executor with context messages without triggering a model call. This is useful for injecting background facts before the workflow reaches a prompt-expecting step:

```python
from agent_framework import AgentExecutorRequest, Message
from agent_framework._workflows._agent_executor import AgentExecutorRequest

# Prime with background without triggering response
background_request = AgentExecutorRequest(
    messages=[Message("user", ["Background: the user's name is Alice."])],
    should_respond=False,
)

# Later, actual prompt triggers response
prompt_request = AgentExecutorRequest(
    messages=[Message("user", ["What is my name?"])],
    should_respond=True,
)
```

### `AgentExecutorResponse`

```python
@dataclass
class AgentExecutorResponse:
    executor_id: str
    agent_response: AgentResponse
    full_conversation: list[Message]
```

`full_conversation` contains **all** messages in the session: the input messages from the cache, plus the assistant's response messages. Downstream `AgentExecutor` instances use this field (in `"full"` context mode) to reconstruct the entire conversational context.

The `.with_text(text: str) -> AgentExecutorResponse` method is the critical pattern for **custom transform executors** that sit between two `AgentExecutor` nodes:

```python
from agent_framework import WorkflowContext
from agent_framework._workflows._agent_executor import AgentExecutorResponse
from agent_framework._workflows._executor import executor

@executor(
    id="summarizer",
    input=AgentExecutorResponse,
    output=AgentExecutorResponse,
)
async def summarize_response(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse],
) -> None:
    # Transform the text — but preserve full_conversation for downstream agents
    short_text = response.agent_response.text[:200] + "..."
    await ctx.send_message(response.with_text(short_text))
    # NOT: await ctx.send_message(short_text)
    # If you send a plain str, the downstream AgentExecutor.from_str handler fires
    # and loses all prior conversation context — only short_text will be in the cache.
```

**Critical gotcha:** Never emit a plain `str` from a transform executor that sits between two `AgentExecutor` nodes. The downstream `from_str` handler will lose all prior messages and produce a WARNING log. Use `.with_text()` to keep the response type as `AgentExecutorResponse` so `from_response` is dispatched instead.

### Building a two-agent pipeline

```python
import asyncio
from agent_framework import Agent, AgentExecutor, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    writer = Agent(
        client=OpenAIChatClient(),
        name="writer",
        instructions="Write a short essay on the given topic.",
    )
    critic = Agent(
        client=OpenAIChatClient(),
        name="critic",
        instructions="Critique the essay and suggest improvements.",
    )

    writer_exec = AgentExecutor(writer)
    critic_exec = AgentExecutor(critic, context_mode="full")

    workflow = (
        WorkflowBuilder(name="write_and_critique")
        .add_executor(writer_exec)
        .add_executor(critic_exec)
        .connect(writer_exec, critic_exec)
        .output_from(writer_exec)
        .output_from(critic_exec)
        .build()
    )

    result = await workflow.run("Write about the benefits of testing.")
    print("Writer:", result.outputs[0])
    print("Critic:", result.outputs[1])

asyncio.run(main())
```

### Checkpoint integration

`AgentExecutor` automatically implements `on_checkpoint_save` and `on_checkpoint_restore` to persist its internal cache, `full_conversation`, session state, and any pending HITL (human-in-the-loop) requests. You do not need to add anything extra to get durable multi-step agent pipelines — just configure a `CheckpointStorage` on the workflow.

---

## 3. `FanOutEdgeGroup` + `FanInEdgeGroup` + `SwitchCaseEdgeGroup`

**Source:** `agent_framework/_workflows/_edge.py`

These are the three higher-order edge group types that express broadcast, merge, and conditional routing. `WorkflowBuilder` creates them under the hood when you call `.fan_out()`, `.fan_in()`, and `.switch()`, but they can also be instantiated directly when you need fine-grained control.

### `SingleEdgeGroup` — the baseline

The simplest group: exactly one source → one target. `WorkflowBuilder.connect()` creates this by default:

```python
from agent_framework._workflows._edge import SingleEdgeGroup

edge = SingleEdgeGroup("parser", "writer")
# Optional condition:
edge_with_cond = SingleEdgeGroup(
    "scorer",
    "high_priority_queue",
    condition=lambda msg: msg["score"] > 0.9,
)
```

`condition` is an `EdgeCondition = Callable[[Any], bool | Awaitable[bool]]`. Async conditions are fully supported.

### `FanOutEdgeGroup`

Broadcasts one source's output to multiple targets simultaneously. All targets receive the same message in the same superstep.

```python
FanOutEdgeGroup(
    source_id: str,
    target_ids: Sequence[str],      # >= 2 required
    selection_func: Callable[[Any, list[str]], list[str]] | None = None,
    *,
    selection_func_name: str | None = None,
    id: str | None = None,
)
```

Without `selection_func` every target receives the message. With `selection_func(message, available_targets) -> list[str]` you can dynamically narrow the fan-out:

```python
from agent_framework._workflows._edge import FanOutEdgeGroup

def route_by_language(msg: dict, targets: list[str]) -> list[str]:
    lang = msg.get("language", "en")
    preferred = f"agent_{lang}"
    return [preferred] if preferred in targets else ["agent_en"]

fan_out = FanOutEdgeGroup(
    source_id="classifier",
    target_ids=["agent_en", "agent_fr", "agent_de"],
    selection_func=route_by_language,
)
```

The `selection_func_name` parameter pins a human-readable name for the callable in serialized form — useful when checkpoints need to be inspected:

```python
fan_out = FanOutEdgeGroup(
    "source",
    ["a", "b", "c"],
    selection_func=lambda msg, targets: [t for t in targets if msg.get(t, False)],
    selection_func_name="select_flagged_targets",  # recorded in checkpoint JSON
)
```

### `FanInEdgeGroup`

Merges messages from multiple upstream sources into a single downstream executor. Each message from any of the sources is forwarded independently — there is no barrier / all-gather semantics. If you need to wait for all sources to finish, you need to implement that logic inside the target executor.

```python
FanInEdgeGroup(
    source_ids: Sequence[str],   # >= 2 required
    target_id: str,
    *,
    id: str | None = None,
)
```

```python
from agent_framework._workflows._edge import FanInEdgeGroup

fan_in = FanInEdgeGroup(
    source_ids=["search_agent", "database_agent", "cache_agent"],
    target_id="aggregator",
)
```

When used with `WorkflowBuilder` for a multi-agent gather:

```python
from agent_framework import WorkflowBuilder, AgentExecutor

search = AgentExecutor(search_agent)
db = AgentExecutor(db_agent)
aggregator = AgentExecutor(aggregator_agent)

workflow = (
    WorkflowBuilder(name="parallel_search")
    .add_executor(search)
    .add_executor(db)
    .add_executor(aggregator)
    .fan_in([search, db], aggregator)   # shorthand that creates FanInEdgeGroup internally
    .build()
)
```

### `SwitchCaseEdgeGroup`

Switch/case routing — exactly one branch is taken per message. Inherits `FanOutEdgeGroup` but replaces `selection_func` with an ordered list of case descriptors.

```python
SwitchCaseEdgeGroup(
    source_id: str,
    cases: Sequence[SwitchCaseEdgeGroupCase | SwitchCaseEdgeGroupDefault],
    *,
    id: str | None = None,
)
```

Supporting types:

```python
SwitchCaseEdgeGroupCase(
    condition: Callable[[Any], bool] | None,  # None installs a placeholder that raises at runtime
    target_id: str,
    *,
    condition_name: str | None = None,
)

SwitchCaseEdgeGroupDefault(target_id: str)
```

Construction constraints (enforced in `__init__`):
- Must have **at least 2 cases** (including the default).
- Must have **exactly one `SwitchCaseEdgeGroupDefault`**.
- Default should be **last** — if it is not, a WARNING is logged (the runtime will still pick it as a fallback after all conditions fail).

```python
from agent_framework._workflows._edge import (
    SwitchCaseEdgeGroup,
    SwitchCaseEdgeGroupCase,
    SwitchCaseEdgeGroupDefault,
)

switch = SwitchCaseEdgeGroup(
    source_id="router",
    cases=[
        SwitchCaseEdgeGroupCase(
            condition=lambda msg: msg["priority"] == "high",
            target_id="fast_track_agent",
        ),
        SwitchCaseEdgeGroupCase(
            condition=lambda msg: msg["priority"] == "medium",
            target_id="standard_agent",
        ),
        SwitchCaseEdgeGroupDefault(target_id="batch_agent"),
    ],
)
```

The runtime evaluates cases **in order**. The first matching condition wins; if none match, the default branch fires. A `RuntimeError` is raised only if somehow both conditions fail AND there is no default (which is blocked at construction time).

### Serialization and checkpoint compatibility

All edge group types implement `to_dict()` / `from_dict()` for checkpoint serialization. **Callable predicates are not serialized** — only their names (via `_extract_function_name`) are stored. When a checkpoint is restored:

- For `FanOutEdgeGroup`: `_selection_func` becomes `_missing_callable(name)` — a placeholder that raises `RuntimeError` on invocation if the workflow definition is not rebuilt.
- For `SwitchCaseEdgeGroup`: each `SwitchCaseEdgeGroupCase._condition` is similarly replaced.

This means you must **rebuild the workflow from Python code** before restoring a checkpoint — do not attempt to create a workflow purely from a serialized checkpoint dict.

### Introspecting edge groups

```python
edge_group = FanOutEdgeGroup("sensor", ["db", "cache", "audit"])
print(edge_group.source_executor_ids)  # ["sensor"]
print(edge_group.target_executor_ids)  # ["db", "cache", "audit"]
print(edge_group.id)                   # auto-generated: "FanOutEdgeGroup/<uuid>"
print(edge_group.type)                 # "FanOutEdgeGroup"
snapshot = edge_group.to_dict()
```

---

## 4. `Runner` + `WorkflowMessage`

**Source:** `agent_framework/_workflows/_runner.py` and `agent_framework/_workflows/_runner_context.py`

`Runner` is the Pregel-style superstep engine that drives workflow execution. Users do not normally instantiate it directly — `WorkflowBuilder.build().run()` creates it internally. Understanding `Runner` is essential for debugging convergence problems, writing custom event monitors, and reasoning about checkpointing.

### `Runner` constructor

```python
Runner(
    edge_groups: Sequence[EdgeGroup],
    executors: dict[str, Executor],
    state: State,
    ctx: RunnerContext,
    workflow_name: str,
    graph_signature_hash: str,
    max_iterations: int = 100,
)
```

| Parameter | Description |
|---|---|
| `edge_groups` | The complete edge topology of the workflow. |
| `executors` | Map of executor `id` → `Executor` instance. |
| `state` | Shared workflow state object (executor states live under `_executor_state` key). |
| `ctx` | `RunnerContext` — the low-level message/event/checkpoint bus. |
| `workflow_name` | Used to label checkpoints. |
| `graph_signature_hash` | Topology hash — must match stored checkpoints on restore. |
| `max_iterations` | Maximum number of supersteps before `WorkflowConvergenceException` is raised. |

### Superstep execution model

`run_until_convergence()` is an **async generator** that yields `WorkflowEvent` objects as it progresses:

```
Superstep 1:
  yield WorkflowEvent.superstep_started(iteration=1)
  [all pending messages delivered concurrently through edge runners]
  state.commit()
  [checkpoint saved if storage is configured]
  yield WorkflowEvent.superstep_completed(iteration=1)
Superstep 2:
  ...
```

Within each superstep, messages from different sources are delivered **concurrently** via `asyncio.create_task`. Messages from the same source to the same target are delivered **in order**. Events emitted by executor handlers (e.g., `executor_invoked`, `executor_completed`, `executor_failed`) are streamed via a 50 ms poll loop so they arrive interleaved with execution.

```python
# Consuming the event stream directly (advanced use)
import asyncio

async def monitor_workflow(runner: Runner) -> None:
    async for event in runner.run_until_convergence():
        print(f"[{event.type}] {event.source_id} → {event.data!r}")

asyncio.run(monitor_workflow(runner))
```

### Convergence and termination

The loop continues until `ctx.has_messages()` returns `False` — meaning no executor sent any message in the last superstep. If this condition is not reached within `max_iterations`, `WorkflowConvergenceException` is raised.

Common causes of non-convergence:
- A cycle in the graph where executors ping-pong messages indefinitely.
- An executor that always sends a message without a terminal condition.
- A mis-wired fan-out where outputs loop back to the start.

```python
# Increase the cap for intentionally long workflows
workflow = (
    WorkflowBuilder(name="long_pipeline")
    # ... add executors and edges ...
    .max_iterations(500)  # WorkflowBuilder exposes this parameter
    .build()
)
```

### `WorkflowMessage`

`WorkflowMessage` is the internal envelope that carries every message through the routing layer:

```python
@dataclass
class WorkflowMessage:
    data: Any
    source_id: str
    target_id: str | None = None
    type: MessageType = MessageType.STANDARD
    trace_contexts: list[dict[str, str]] | None = None   # W3C Trace Context (plural for fan-in)
    source_span_ids: list[str] | None = None              # OTel span IDs for linking
    original_request_info_event: WorkflowEvent | None = None  # for HITL RESPONSE messages
```

`MessageType` is an enum with two values:
- `MessageType.STANDARD` — normal executor-to-executor message.
- `MessageType.RESPONSE` — a HITL response to a pending `request_info` event. The `original_request_info_event` field links back to the original request.

Backward-compatible properties exist for code that accesses single trace/span values:

```python
msg = WorkflowMessage(data="hello", source_id="step_a")
msg.trace_context   # first element of trace_contexts, or None
msg.source_span_id  # first element of source_span_ids, or None
```

### Serialization for checkpointing

`WorkflowMessage` supports `to_dict()` and `from_dict()` for checkpoint storage. The `data` payload must itself be serializable by the checkpoint encoding layer:

```python
msg = WorkflowMessage(
    data={"result": 42},
    source_id="compute",
    target_id="output",
    type=MessageType.STANDARD,
)
serialized = msg.to_dict()
restored = WorkflowMessage.from_dict(serialized)
assert restored.source_id == "compute"
```

### Inspecting in-flight messages during debugging

Because `Runner` holds all pending messages in `RunnerContext`, you can peek at workflow state without interrupting execution by subscribing to the event stream:

```python
from agent_framework._workflows._events import WorkflowEventType

async def debug_run(workflow, prompt: str) -> None:
    async for event in workflow.run_stream(prompt):
        if event.type == WorkflowEventType.EXECUTOR_INVOKED:
            print(f"  >> {event.source_id} received: {event.data!r}")
        elif event.type == WorkflowEventType.EXECUTOR_COMPLETED:
            print(f"  << {event.source_id} sent: {event.data!r}")
        elif event.type == WorkflowEventType.EXECUTOR_FAILED:
            print(f"  !! {event.source_id} FAILED: {event.data!r}")
```

---

## 5. `SessionContext`

**Source:** `agent_framework/_sessions.py`

`SessionContext` is the **per-invocation pipeline state** object created fresh at the start of every `agent.run()` call. Context providers (`MemoryContextProvider`, `BackgroundAgentsProvider`, etc.) read from and write to it via its mutation methods. It is passed to every `before_run` and `after_run` hook in provider execution order.

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

The framework constructs `SessionContext` automatically for each `agent.run()` call. You only construct it directly when testing context providers or writing custom provider pipelines.

### Field reference

| Field | Type | Description |
|---|---|---|
| `session_id` | `str \| None` | Session identifier, if known. |
| `service_session_id` | `str \| None` | Service-managed ID (e.g., when the model backend owns session storage). |
| `input_messages` | `list[Message]` | The new messages being sent in this invocation (read-only intent). |
| `context_messages` | `dict[str, list[Message]]` | Messages per provider `source_id`; populated by `extend_messages`. |
| `instructions` | `list[str]` | Accumulated instructions from all providers. |
| `tools` | `list[Any]` | Accumulated tools from all providers. |
| `middleware` | `dict[str, list[MiddlewareTypes]]` | Per-provider middleware; keyed by `source_id`. |
| `options` | `dict[str, Any]` | Options from `agent.run()` — providers should treat this as read-only. |
| `metadata` | `dict[str, Any]` | Shared scratch space for cross-provider communication. |
| `response` | `AgentResponse \| None` | Set by framework after the model call; populated in `after_run` hooks. |

### `extend_messages` and attribution

`extend_messages` is the primary method providers use to inject conversation history. It **copies** each message before storing it, then stamps the copy with an `_attribution` dict in `additional_properties`:

```python
context.extend_messages(self, [msg1, msg2])
# Each stored copy has:
# msg.additional_properties["_attribution"] = {
#     "source_id": "my_provider",
#     "source_type": "MyProviderClass",
# }
```

When `source` is a plain string rather than a provider object, `source_type` is omitted:

```python
context.extend_messages("custom_source", [msg])
# msg.additional_properties["_attribution"] = {"source_id": "custom_source"}
```

The `_attribution` key allows downstream providers to filter messages by their origin:

```python
context.get_messages(sources={"history_provider"})        # only history messages
context.get_messages(exclude_sources={"noisy_provider"})  # everything except noisy
```

### `get_messages` filtering options

```python
context.get_messages(
    sources={"source_a", "source_b"} | None,    # include only these sources
    exclude_sources={"source_c"} | None,         # exclude these sources
    include_input=True,                           # append input_messages at end
    include_response=True,                        # append response.messages at end
)
```

### Writing a custom context provider

```python
from agent_framework._sessions import ContextProvider, SessionContext, AgentSession
from agent_framework._agents import SupportsAgentRun
from agent_framework import Message
import asyncio

class CurrentTimeProvider(ContextProvider):
    """Injects a system message with the current UTC time before each run."""

    def __init__(self) -> None:
        super().__init__(source_id="current_time")

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict,
    ) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        context.extend_messages(
            self,
            [Message("system", [f"Current time: {now}"])],
        )
```

### Cross-provider communication via `metadata`

The `metadata` dict is an untyped scratch space where one provider can leave information for another downstream provider in the same pipeline run:

```python
class LanguageDetectionProvider(ContextProvider):
    def __init__(self) -> None:
        super().__init__(source_id="lang_detect")

    async def before_run(self, *, context: SessionContext, **kwargs) -> None:
        detected_lang = self._detect(context.input_messages)
        context.metadata["detected_language"] = detected_lang


class LocalizationProvider(ContextProvider):
    def __init__(self) -> None:
        super().__init__(source_id="localization")

    async def before_run(self, *, context: SessionContext, **kwargs) -> None:
        lang = context.metadata.get("detected_language", "en")
        context.extend_instructions(self.source_id, f"Respond in {lang}.")
```

### Middleware injection from providers

Providers can also add `ChatMiddleware` or `FunctionMiddleware` per invocation. This is the mechanism used by compaction providers, audit loggers, and rate limiters:

```python
from agent_framework._sessions import ContextProvider
from agent_framework import ChatMiddleware

class AuditMiddlewareProvider(ContextProvider):
    def __init__(self) -> None:
        super().__init__(source_id="audit")
        self._log = []

    async def before_run(self, *, context: SessionContext, **kwargs) -> None:
        context.extend_middleware(self.source_id, AuditChatMiddleware(self._log))
```

Note: Providers may only add **chat or function middleware** — adding `AgentMiddleware` from a context provider raises `MiddlewareException`.

---

## 6. `AgentSession` + `register_state_type`

**Source:** `agent_framework/_sessions.py`

`AgentSession` is the lightweight, **cross-call** state container that persists across multiple `agent.run()` calls. Unlike `SessionContext` (which is created fresh each time), `AgentSession` lives as long as you pass the same object between calls.

### Constructor

```python
AgentSession(
    *,
    session_id: str | None = None,           # auto-generates UUID if None
    service_session_id: str | None = None,   # service-managed ID
)
```

```python
from agent_framework._sessions import AgentSession

# Create once, reuse across multiple agent.run() calls
session = AgentSession()
print(session.session_id)  # stable UUID

response1 = await agent.run("Hello!", session=session)
response2 = await agent.run("What did I just say?", session=session)
# response2 sees history from response1 because session carries state
```

### The `state` dict

`state` is a plain `dict[str, Any]` that all providers share. Provider implementations typically scope their data under their own `source_id` key to avoid collisions:

```python
# Inside a provider's before_run:
my_state = session.state.setdefault(self.source_id, {})
my_state["last_query"] = "..."
```

### Serialization: `to_dict` / `from_dict`

`AgentSession` can be round-tripped through a dictionary for external storage (databases, Redis, etc.):

```python
# Serialize for storage
session_dict = session.to_dict()
# {
#   "type": "session",
#   "session_id": "...",
#   "service_session_id": None,
#   "state": { ... }
# }

# Restore
restored_session = AgentSession.from_dict(session_dict)
assert restored_session.session_id == session.session_id
```

The `state` dict is deep-serialized. The framework handles these types automatically:

| Value type | Serialized form | Restored via |
|---|---|---|
| `str`, `int`, `float`, `bool`, `None` | Kept as-is | Identity |
| `list`, `dict` | Recursed | Recursed |
| Object with `to_dict()` / `from_dict()` | `to_dict()` result | `from_dict(data)` |
| Pydantic `BaseModel` subclass | `model_dump()` + `"type"` discriminator | `model_validate(data)` |

### `register_state_type`

Pydantic models stored in `state` are **auto-registered** on first serialization. However, if you restore a session from storage on a cold start before the model has ever been serialized in the current process, the registry will be empty and deserialization will silently return the raw dict instead of a typed object.

Call `register_state_type` at module import time to guarantee cold-start correctness:

```python
from agent_framework._sessions import register_state_type
from pydantic import BaseModel

class UserPreferences(BaseModel):
    language: str = "en"
    timezone: str = "UTC"
    notifications_enabled: bool = True

register_state_type(UserPreferences)   # call once at module level
```

The type identifier defaults to `cls.__name__.lower()` (i.e., `"userpreferences"`). Override with a classmethod if you need a different key:

```python
class UserPreferences(BaseModel):
    language: str = "en"

    @classmethod
    def _get_type_identifier(cls) -> str:
        return "prefs"   # stored as "type": "prefs" in the dict

register_state_type(UserPreferences)
```

### Storing and restoring a typed object

```python
import asyncio
import json
from agent_framework._sessions import AgentSession, register_state_type
from pydantic import BaseModel

class ConversationMeta(BaseModel):
    turn_count: int = 0
    topic: str = "unknown"

register_state_type(ConversationMeta)

async def main() -> None:
    session = AgentSession()
    meta = ConversationMeta(turn_count=5, topic="Python async")
    session.state["meta"] = meta

    # Serialize and save
    raw = json.dumps(session.to_dict())

    # Restore (simulates cold start after process restart)
    restored_session = AgentSession.from_dict(json.loads(raw))
    restored_meta = restored_session.state["meta"]
    assert isinstance(restored_meta, ConversationMeta)
    assert restored_meta.turn_count == 5

asyncio.run(main())
```

### Non-Pydantic objects with `to_dict` / `from_dict`

Any object implementing `to_dict()` and `from_dict()` round-trips cleanly. The `from_dict` classmethod must accept the same dict shape that `to_dict()` returns, and the dict must contain a `"type"` key matching the registered identifier:

```python
from agent_framework._sessions import register_state_type

class SearchHistory:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    def to_dict(self) -> dict:
        return {"type": "search_history", "queries": self.queries}

    @classmethod
    def from_dict(cls, data: dict) -> "SearchHistory":
        return cls(queries=data["queries"])

register_state_type(SearchHistory)
```

---

## 7. `BaseChatClient` + `SupportsChatGetResponse`

**Source:** `agent_framework/_clients.py`

These two types form the extensibility boundary of the model layer. `SupportsChatGetResponse` is the minimal structural protocol — any class that provides the right methods qualifies. `BaseChatClient` is the recommended abstract base class for implementing a fully integrated custom client.

### `SupportsChatGetResponse` protocol

```python
@runtime_checkable
class SupportsChatGetResponse(Protocol[OptionsContraT]):
    additional_properties: dict[str, Any]

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: ... | None = None,
        compaction_strategy: ... | None = None,
        tokenizer: ... | None = None,
        function_invocation_kwargs: ... | None = None,
        client_kwargs: ... | None = None,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        ...
```

The protocol is `@runtime_checkable`, so you can use `isinstance` for duck-type verification:

```python
from agent_framework import SupportsChatGetResponse
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
assert isinstance(client, SupportsChatGetResponse)

# Any custom class that implements the required interface also qualifies:
class MockClient:
    additional_properties: dict = {}

    def get_response(self, messages, *, stream=False, options=None, **kwargs):
        import asyncio
        from agent_framework import ChatResponse, Message
        async def _resp() -> ChatResponse:
            return ChatResponse(messages=[Message("assistant", ["mock response"])])
        return _resp()

assert isinstance(MockClient(), SupportsChatGetResponse)
```

### `BaseChatClient` — the implementation base

`BaseChatClient` is `ABC + Generic[OptionsCoT]`. The generic `OptionsCoT` param is a `TypedDict` describing the provider-specific chat options, enabling IDE autocomplete for things like `temperature`, `model`, and provider-specific keys.

**Required abstract method:**

```python
@abstractmethod
def _inner_get_response(
    self,
    *,
    messages: Sequence[Message],
    stream: bool,
    options: Mapping[str, Any],
    **kwargs: Any,
) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
    ...
```

When `stream=False` return an awaitable `ChatResponse`. When `stream=True` return a `ResponseStream`.

### Minimal custom client implementation

```python
import asyncio
from collections.abc import Mapping, Sequence, AsyncIterable
from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Message,
    ResponseStream,
)


class EchoClient(BaseChatClient):
    """Chat client that echoes the last user message."""

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ChatResponse | ResponseStream[ChatResponseUpdate, ChatResponse]:
        last_user = next(
            (m.text for m in reversed(list(messages)) if m.role == "user"),
            "(no user message)",
        )
        reply_text = f"Echo: {last_user}"

        if stream:
            async def _chunks() -> AsyncIterable[ChatResponseUpdate]:
                for word in reply_text.split():
                    yield ChatResponseUpdate(
                        role="assistant",
                        contents=[{"type": "text", "text": word + " "}],
                    )
            return self._build_response_stream(_chunks())

        return ChatResponse(
            messages=[Message("assistant", [reply_text])],
        )
```

### Adding provider-specific options via TypedDict

```python
from typing import TypedDict
from agent_framework import BaseChatClient, ChatResponse, ResponseStream, Message

class MyProviderOptions(TypedDict, total=False):
    temperature: float
    sampling_method: str   # provider-specific

class MyProviderClient(BaseChatClient["MyProviderOptions"]):
    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        temp = options.get("temperature", 0.7)
        method = options.get("sampling_method", "greedy")
        # ... use temp and method ...
        return ChatResponse(messages=[Message("assistant", ["response"])])


# IDE gives type-checked autocomplete for MyProviderOptions:
client = MyProviderClient()
agent = client.as_agent(
    name="my_agent",
    default_options={"temperature": 0.3, "sampling_method": "beam"},
)
```

### `as_agent()` convenience method

Every `BaseChatClient` exposes `.as_agent(...)` which constructs an `Agent` wrapping the client. This is the recommended way to configure agents from client instances:

```python
from agent_framework import FunctionTool

def get_weather(city: str) -> str:
    return f"Sunny in {city}"

agent = EchoClient().as_agent(
    name="weather_bot",
    instructions="Help with weather queries.",
    tools=[FunctionTool(get_weather)],
)
```

### Class-level constants

| Constant | Type | Description |
|---|---|---|
| `OTEL_PROVIDER_NAME` | `ClassVar[str]` | Provider name emitted in OTel spans. Defaults to `"unknown"`. Override in subclasses. |
| `STORES_BY_DEFAULT` | `ClassVar[bool]` | If `True`, agent skips auto-injecting `InMemoryHistoryProvider`. Set to `True` for clients that manage history server-side. |
| `DEFAULT_EXCLUDE` | `ClassVar[set[str]]` | Fields excluded from `to_dict()` serialization by default. |

---

## 8. `SecretString` + `load_settings`

**Source:** `agent_framework/_settings.py`

### `SecretString`

`SecretString` is a `str` subclass that overrides `__repr__` to mask the value. All string operations work normally — concatenation, slicing, `len()`, formatting — but `repr()` shows `SecretString('**********')` instead of the actual value.

```python
from agent_framework._settings import SecretString

api_key = SecretString("sk-real-key-goes-here")

# Normal str operations
assert api_key == "sk-real-key-goes-here"
assert api_key.startswith("sk-")
assert len(api_key) == 20
assert f"Key={api_key}"  == "Key=sk-real-key-goes-here"  # f-strings show plaintext

# Masking in repr (what gets logged)
print(repr(api_key))   # SecretString('**********')

# Backward compat with pydantic.SecretStr
assert api_key.get_secret_value() == "sk-real-key-goes-here"
```

**When to use `SecretString`:** Wrap any string value that could expose credentials if accidentally included in a log, traceback, or `str()` call on a parent object. The framework auto-coerces plain `str` overrides to `SecretString` in `load_settings` when the annotation expects it.

```python
import os
from agent_framework import SecretString
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(
    api_key=SecretString(os.environ["OPENAI_API_KEY"]),
    # api_key never appears in repr() output, tracebacks, or debug logs
)
```

### `load_settings`

`load_settings` is a generic settings loader for `TypedDict`-defined configuration. It resolves values in priority order: explicit kwargs → `.env` file → environment variables → defaults.

```python
load_settings(
    settings_type: type[SettingsT],   # must be a TypedDict subclass
    *,
    env_prefix: str = "",
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
    required_fields: Sequence[str | tuple[str, ...]] | None = None,
    **overrides: Any,
) -> SettingsT
```

**Resolution order (highest priority first):**
1. Keyword `overrides` (skips `None` values — they fall through to lower-priority sources)
2. `.env` file (only when `env_file_path` is explicitly provided)
3. Environment variables (`<env_prefix><FIELD_NAME>`)
4. TypedDict class-level defaults, or `None` for optional fields

```python
from typing import TypedDict
from agent_framework._settings import SecretString, load_settings

class DatabaseSettings(TypedDict, total=False):
    host: str | None
    port: int | None
    api_key: SecretString | None

# Will raise SettingNotFoundError if DB_HOST is not set
settings = load_settings(
    DatabaseSettings,
    env_prefix="DB_",
    required_fields=["host"],      # "host" must be non-None
)
print(settings["host"])            # from DB_HOST env var
print(repr(settings["api_key"]))   # SecretString('**********') if set, or None
```

**Mutually exclusive fields:** Use a tuple in `required_fields` to require exactly one of a group:

```python
class AzureSettings(TypedDict, total=False):
    connection_string: str | None
    account_name: str | None
    account_key: SecretString | None

settings = load_settings(
    AzureSettings,
    env_prefix="AZURE_",
    required_fields=[
        ("connection_string", "account_name"),  # exactly one must be set
    ],
)
```

**Loading from a `.env` file:**

```python
settings = load_settings(
    DatabaseSettings,
    env_prefix="DB_",
    env_file_path=".env.production",
    env_file_encoding="utf-8",
)
```

If `env_file_path` is provided but the file does not exist, `FileNotFoundError` is raised immediately. Unlike `python-dotenv`'s default behavior, the file is not silently skipped.

**Type coercion:** String values from env vars are automatically coerced:

| Annotation | Input string | Result |
|---|---|---|
| `int \| None` | `"8080"` | `8080` |
| `float \| None` | `"0.7"` | `0.7` |
| `bool \| None` | `"true"` / `"1"` / `"yes"` | `True` |
| `SecretString \| None` | `"sk-..."` | `SecretString("sk-...")` |
| `str \| None` | any string | passed through unchanged |

### Building a custom client with `load_settings`

```python
from typing import TypedDict
from agent_framework._settings import SecretString, load_settings
from agent_framework import BaseChatClient, ChatResponse, Message

class GroqOptions(TypedDict, total=False):
    model: str | None
    temperature: float | None

class GroqSettings(TypedDict, total=False):
    api_key: SecretString | None
    base_url: str | None

class GroqChatClient(BaseChatClient[GroqOptions]):
    OTEL_PROVIDER_NAME = "groq"

    def __init__(
        self,
        *,
        api_key: SecretString | str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        settings = load_settings(
            GroqSettings,
            env_prefix="GROQ_",
            api_key=api_key,
            base_url=base_url,
        )
        self._api_key = settings["api_key"]  # SecretString — masked in logs
        self._base_url = settings["base_url"] or "https://api.groq.com"

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        # ... call Groq API using self._api_key.get_secret_value() ...
        return ChatResponse(messages=[Message("assistant", ["response"])])
```

---

## 9. `WorkflowCheckpoint` + `CheckpointStorage` + `InMemoryCheckpointStorage`

**Source:** `agent_framework/_workflows/_checkpoint.py`

Checkpointing allows workflows to be paused after any superstep and resumed later — across process restarts, HITL pauses, or scheduled batch execution. Every checkpoint captures the complete workflow state: pending messages, committed executor states, and any unresolved HITL request events.

### `WorkflowCheckpoint`

```python
@dataclass(slots=True)
class WorkflowCheckpoint:
    workflow_name: str
    graph_signature_hash: str
    checkpoint_id: CheckpointID         # auto-generated UUID
    previous_checkpoint_id: CheckpointID | None
    timestamp: str                       # ISO 8601 UTC, auto-generated
    messages: dict[str, list[WorkflowMessage]]
    state: dict[str, Any]
    pending_request_info_events: dict[str, WorkflowEvent]
    iteration_count: int
    metadata: dict[str, Any]
    version: str = "1.0"
```

| Field | Description |
|---|---|
| `workflow_name` | Logical grouping for checkpoints. All runs of the same workflow definition share a name. |
| `graph_signature_hash` | Hash of the executor+edge topology. Changes when the graph changes. |
| `checkpoint_id` | Auto-generated UUID for this checkpoint. |
| `previous_checkpoint_id` | Links to prior checkpoint, forming a history chain. |
| `timestamp` | UTC ISO 8601 creation time. |
| `messages` | In-flight messages keyed by source executor ID — what was pending when the checkpoint was taken. |
| `state` | Committed workflow state; executor states are under `state["_executor_state"]`. |
| `pending_request_info_events` | Unresolved HITL requests keyed by request ID. |
| `iteration_count` | Superstep number when the checkpoint was taken. |

**`graph_signature_hash` compatibility:** If you change your workflow definition — add/remove an executor, change an edge, or rename an executor ID — the hash changes. Restoring an old checkpoint with a different graph raises `WorkflowCheckpointException`. This prevents subtle state corruption from mismatched topology.

```python
# Inspecting a checkpoint
from agent_framework._workflows._checkpoint import WorkflowCheckpoint

checkpoint = await storage.get_latest(workflow_name="my_pipeline")
if checkpoint:
    print(f"Checkpoint from: {checkpoint.timestamp}")
    print(f"Iteration: {checkpoint.iteration_count}")
    print(f"Pending messages from: {list(checkpoint.messages.keys())}")
    print(f"Graph hash: {checkpoint.graph_signature_hash}")
```

### `CheckpointStorage` protocol

```python
class CheckpointStorage(Protocol):
    async def save(self, checkpoint: WorkflowCheckpoint) -> CheckpointID: ...
    async def load(self, checkpoint_id: CheckpointID) -> WorkflowCheckpoint: ...
    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]: ...
    async def delete(self, checkpoint_id: CheckpointID) -> bool: ...
    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None: ...
    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[CheckpointID]: ...
```

`CheckpointStorage` is a structural Protocol — implement any class with these five async methods and it qualifies without subclassing.

### `InMemoryCheckpointStorage`

The reference in-memory implementation. Does **not** persist across process restarts. Use it for testing, development, and HITL workflows within a single process:

```python
from agent_framework._workflows._checkpoint import InMemoryCheckpointStorage

storage = InMemoryCheckpointStorage()
```

For production use, use `FileCheckpointStorage` (see Vol. 2) or implement a custom storage backend backed by a database.

### Enabling checkpointing on a workflow

```python
from agent_framework import WorkflowBuilder
from agent_framework._workflows._checkpoint import InMemoryCheckpointStorage

storage = InMemoryCheckpointStorage()

workflow = (
    WorkflowBuilder(name="resumable_pipeline")
    .add_executor(step1_exec)
    .add_executor(step2_exec)
    .connect(step1_exec, step2_exec)
    .checkpoint_storage(storage)   # enables checkpointing
    .build()
)

result = await workflow.run("initial input")
```

### Resuming from a checkpoint

```python
# On restart / after HITL pause
latest = await storage.get_latest(workflow_name="resumable_pipeline")
if latest:
    result = await workflow.run_from_checkpoint(latest.checkpoint_id)
```

### Custom checkpoint storage backend

```python
import json
from agent_framework._workflows._checkpoint import (
    CheckpointStorage,
    CheckpointID,
    WorkflowCheckpoint,
)


class RedisCheckpointStorage:
    """Checkpoint storage backed by Redis (sketch)."""

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    async def save(self, checkpoint: WorkflowCheckpoint) -> CheckpointID:
        key = f"checkpoint:{checkpoint.checkpoint_id}"
        # Note: WorkflowCheckpoint.to_dict() returns a shallow dict;
        # WorkflowMessage objects inside messages/pending_request_info_events
        # require additional encoding via the checkpoint_encoding module.
        raw = json.dumps(checkpoint.to_dict(), default=str)
        await self._redis.set(key, raw)
        await self._redis.rpush(
            f"workflow:{checkpoint.workflow_name}",
            checkpoint.checkpoint_id,
        )
        return checkpoint.checkpoint_id

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        ids = await self._redis.lrange(f"workflow:{workflow_name}", -1, -1)
        if not ids:
            return None
        return await self.load(ids[0].decode())

    async def load(self, checkpoint_id: CheckpointID) -> WorkflowCheckpoint:
        raw = await self._redis.get(f"checkpoint:{checkpoint_id}")
        if raw is None:
            from agent_framework.exceptions import WorkflowCheckpointException
            raise WorkflowCheckpointException(f"No checkpoint with id: {checkpoint_id}")
        return WorkflowCheckpoint.from_dict(json.loads(raw))

    async def delete(self, checkpoint_id: CheckpointID) -> bool:
        result = await self._redis.delete(f"checkpoint:{checkpoint_id}")
        return result > 0

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        ids = await self.list_checkpoint_ids(workflow_name=workflow_name)
        return [await self.load(cid) for cid in ids]

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[CheckpointID]:
        raw_ids = await self._redis.lrange(f"workflow:{workflow_name}", 0, -1)
        return [cid.decode() for cid in raw_ids]
```

### Checkpoint chains

Each `WorkflowCheckpoint` records its `previous_checkpoint_id`, forming a linked history. You can walk the chain to reconstruct execution history or roll back to a specific point:

```python
async def get_checkpoint_history(
    storage: CheckpointStorage,
    workflow_name: str,
) -> list[WorkflowCheckpoint]:
    all_checkpoints = await storage.list_checkpoints(workflow_name=workflow_name)
    # Sort by timestamp for chronological order
    return sorted(all_checkpoints, key=lambda c: c.timestamp)
```

---

## 10. Exception Hierarchy

**Source:** `agent_framework/exceptions.py`, `agent_framework/_middleware.py`, `agent_framework/_workflows/_validation.py`, `agent_framework/_evaluation.py`

The framework uses a unified exception hierarchy rooted at `AgentFrameworkException`. Every exception in the hierarchy auto-logs at DEBUG level by default — this means exceptions caught and re-raised internally never silently disappear from your debug logs.

### Full tree

```
AgentFrameworkException(Exception)
│   log_level=10 (DEBUG) by default; pass log_level=None to suppress
│
├── AgentException
│   ├── AgentInvalidAuthException         — invalid credentials passed to agent
│   ├── AgentInvalidRequestException      — malformed or rejected agent request
│   ├── AgentInvalidResponseException     — agent returned unexpected/unparseable response
│   └── AgentContentFilterException       — content filter triggered during agent run
│
├── ChatClientException
│   ├── ChatClientInvalidAuthException
│   ├── ChatClientInvalidRequestException
│   ├── ChatClientInvalidResponseException
│   └── ChatClientContentFilterException
│
├── IntegrationException                  — external service / dependency failures
│   ├── IntegrationInitializationError    — setup / lifecycle failure
│   ├── IntegrationInvalidAuthException
│   ├── IntegrationInvalidRequestException
│   ├── IntegrationInvalidResponseException
│   └── IntegrationContentFilterException
│
├── ContentError                          — content item processing failure
│   └── AdditionItemMismatch              — type mismatch when merging content
│
├── ToolException
│   ├── ToolExecutionException            — tool call failed at runtime
│   └── UserInputRequiredException        — tool/sub-agent requires human input
│
├── MiddlewareException                   — middleware pipeline failure
│   └── MiddlewareTermination             — graceful early exit (not an error)
│
├── SettingNotFoundError                  — required setting missing from all sources
│
├── WorkflowException
│   ├── WorkflowRunnerException
│   │   ├── WorkflowConvergenceException  — max_iterations exceeded with pending messages
│   │   └── WorkflowCheckpointException  — save/load checkpoint failure
│   └── WorkflowValidationError          — graph/type validation error at build time
│       ├── EdgeDuplicationError          — duplicate edges in graph
│       ├── TypeCompatibilityError        — incompatible message types on connected edge
│       └── GraphConnectivityError        — disconnected nodes or unreachable executors
│
└── EvalNotPassedError                    — evaluation assertion failed
```

### `AgentFrameworkException` — base constructor

```python
AgentFrameworkException(
    message: str,
    inner_exception: Exception | None = None,
    log_level: 0 | 10 | 20 | 30 | 40 | 50 | None = 10,
)
```

The `log_level` controls automatic logging:

| Value | Level | Effect |
|---|---|---|
| `10` (default) | DEBUG | Always logged at debug — appears in `logging.DEBUG` output |
| `20` | INFO | Logged at info |
| `30` | WARNING | Logged at warning |
| `None` | Suppressed | No automatic log — useful for control-flow exceptions |

```python
from agent_framework.exceptions import AgentFrameworkException

# Log at WARNING and chain the original cause:
raise AgentFrameworkException(
    "Custom integration failed.",
    inner_exception=original_exc,
    log_level=30,
)
```

### `MiddlewareTermination` — graceful short-circuit

`MiddlewareTermination` is raised inside a middleware `process()` method to **stop the middleware pipeline** without propagating an error to the caller. It carries an optional `result` payload that the pipeline can return instead of calling the model:

```python
from agent_framework import MiddlewareTermination, ChatMiddleware, ChatContext
from collections.abc import Callable, Awaitable

class CachedResponseMiddleware(ChatMiddleware):
    def __init__(self, cache: dict) -> None:
        self._cache = cache

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        key = str(context.messages)
        if key in self._cache:
            context.result = self._cache[key]
            raise MiddlewareTermination("Cache hit — skipping model call.")
        await call_next()
        self._cache[key] = context.result
```

`MiddlewareTermination` is a subclass of `MiddlewareException` (which is itself `AgentFrameworkException`). It uses `log_level=None` so it does not emit debug logs — it is a control-flow signal, not an error.

### `WorkflowConvergenceException` — infinite loop detection

Raised by `Runner.run_until_convergence()` when `max_iterations` is exceeded:

```python
from agent_framework.exceptions import WorkflowConvergenceException

try:
    result = await workflow.run("prompt")
except WorkflowConvergenceException as exc:
    print(f"Workflow did not converge: {exc}")
    # Inspect the workflow graph for cycles or missing terminal conditions
```

When you see this exception, check:
1. Does any executor always send at least one message regardless of input?
2. Is there a cycle in the graph that lacks a termination condition?
3. Is `max_iterations` too low for a legitimate long-running workflow? Increase it with `.max_iterations(N)` on `WorkflowBuilder`.

### `WorkflowValidationError` family — build-time graph checks

These are raised by `WorkflowBuilder.build()` when the graph topology is invalid, before any messages are sent:

```python
from agent_framework.exceptions import (
    WorkflowValidationError,
    EdgeDuplicationError,
    TypeCompatibilityError,
    GraphConnectivityError,
)

try:
    workflow = (
        WorkflowBuilder(name="test")
        .add_executor(exec_a)
        .add_executor(exec_b)
        .connect(exec_a, exec_b)
        .connect(exec_a, exec_b)   # duplicate edge
        .build()
    )
except EdgeDuplicationError as exc:
    print(f"Duplicate edge: {exc.edge_id}")
except TypeCompatibilityError as exc:
    print(f"Type mismatch: {exc.source_executor_id} -> {exc.target_executor_id}")
    print(f"  Source outputs: {exc.source_types}")
    print(f"  Target accepts: {exc.target_types}")
except GraphConnectivityError as exc:
    print(f"Connectivity problem: {exc.message}")
```

`WorkflowValidationError` carries a `validation_type: ValidationTypeEnum` field that discriminates between `EDGE_DUPLICATION`, `TYPE_COMPATIBILITY`, and `GRAPH_CONNECTIVITY`.

### `UserInputRequiredException` — HITL escalation

Raised when an agent run requires human input that was not provided (typically from a HITL tool call). In `AgentExecutor`, this is caught and converted into a `request_info` HITL event rather than propagating to the workflow:

```python
from agent_framework.exceptions import UserInputRequiredException

try:
    response = await agent.run("Approve this transaction?")
except UserInputRequiredException as exc:
    # exc.contents contains the list of user-input-request Content items
    for item in exc.contents:
        print(f"User input needed: {item.type}")
```

### `WorkflowCheckpointException` — save/restore failure

Raised when checkpoint serialization or deserialization fails, or when a checkpoint's `graph_signature_hash` does not match the current workflow:

```python
from agent_framework.exceptions import WorkflowCheckpointException

try:
    result = await workflow.run_from_checkpoint(old_checkpoint_id)
except WorkflowCheckpointException as exc:
    # Most likely cause: workflow graph was changed since the checkpoint was taken
    print(f"Checkpoint incompatible: {exc}")
    # Create a fresh run instead
    result = await workflow.run("prompt")
```

### `EvalNotPassedError` — evaluation gate failure

Raised by `LocalEvaluator.run_and_check()` when an evaluation score falls below the configured threshold. Carries the full `EvalResults` so you can inspect individual item scores:

```python
from agent_framework._evaluation import EvalNotPassedError

try:
    eval_results = await evaluator.run_and_check(eval_items)
except EvalNotPassedError as exc:
    print(f"Evaluation failed: {exc}")
```

### Exception handling best practices

```python
from agent_framework.exceptions import (
    AgentFrameworkException,
    AgentContentFilterException,
    ChatClientInvalidAuthException,
    WorkflowConvergenceException,
    WorkflowCheckpointException,
)

async def robust_run(workflow, prompt: str, storage):
    # Attempt to resume from checkpoint first
    checkpoint = await storage.get_latest(workflow_name=workflow.name)
    try:
        if checkpoint:
            result = await workflow.run_from_checkpoint(checkpoint.checkpoint_id)
        else:
            result = await workflow.run(prompt)
        return result

    except ChatClientInvalidAuthException:
        # Credentials expired — refresh and retry once
        refresh_credentials()
        return await workflow.run(prompt)

    except AgentContentFilterException:
        # Content was blocked — return safe fallback
        return "I'm sorry, I cannot respond to that request."

    except WorkflowConvergenceException:
        # Workflow looped — abort and alert
        send_alert("Workflow did not converge")
        raise

    except WorkflowCheckpointException:
        # Stale checkpoint — start fresh
        return await workflow.run(prompt)

    except AgentFrameworkException as exc:
        # Catch-all for framework errors
        log_error(exc)
        raise
```

---

## Summary Table

| Class / Group | Module | Typical use | Reach for this when... |
|---|---|---|---|
| `Executor` + `@handler` + `@executor` | `_workflows/_executor.py` | Defining custom workflow processing units | You need a reusable, typed, stateful processing step in a workflow graph |
| `AgentExecutor` + `AgentExecutorRequest` + `AgentExecutorResponse` | `_workflows/_agent_executor.py` | Wrapping `Agent` as a workflow node | You want an agent to participate in a multi-step workflow with context continuity |
| `FanOutEdgeGroup` + `FanInEdgeGroup` + `SwitchCaseEdgeGroup` | `_workflows/_edge.py` | Advanced graph routing | You need broadcast, merge, or conditional routing between workflow executors |
| `Runner` + `WorkflowMessage` | `_workflows/_runner.py` + `_runner_context.py` | Debugging execution flow, advanced event handling | You need to understand superstep scheduling, debug convergence, or inspect in-flight messages |
| `SessionContext` | `_sessions.py` | Writing custom context providers | You are building a provider that injects messages, tools, instructions, or middleware per invocation |
| `AgentSession` + `register_state_type` | `_sessions.py` | Cross-call state persistence | You need conversation history, typed custom state, or session serialization for storage |
| `BaseChatClient` + `SupportsChatGetResponse` | `_clients.py` | Integrating a new model provider | You need to connect a model API that has no built-in framework client |
| `SecretString` + `load_settings` | `_settings.py` | Safe credential handling and settings loading | You need to load API keys from env vars / `.env` files without risking log exposure |
| `WorkflowCheckpoint` + `CheckpointStorage` + `InMemoryCheckpointStorage` | `_workflows/_checkpoint.py` | Durable workflow execution and HITL pauses | You need workflows that survive process restarts or wait for human input |
| Exception hierarchy | `exceptions.py` + others | Structured error handling | You want to catch specific failure categories (auth, content filter, convergence, validation) |

---

## Revision History

| Version | Date | Notes |
|---|---|---|
| 1.0 | 2026-06-02 | Initial publication, verified against agent-framework-core 1.7.0 |

---

*This document was introspected from **agent-framework-core 1.7.0** source on 2026-06-02.*

---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 7"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.7.0: ContextProvider custom authoring, BackgroundTaskInfo + BackgroundTaskStatus, GroupChatBuilder + TerminationCondition, HandoffBuilder + HandoffConfiguration, MagenticBuilder + StandardMagenticManager + MagenticProgressLedger, SequentialBuilder + ConcurrentBuilder, AgentFactory + WorkflowFactory (declarative), SecureAgentConfig + ContentLabel + IntegrityLabel (security), FunctionalWorkflowAgent, ObservabilitySettings + configure_otel_providers."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 30
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 7

Verified against **agent-framework 1.7.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source at `/usr/local/lib/python3.11/dist-packages/agent_framework/`. No API name has
been guessed or inferred from documentation alone.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool` + `MCPWebsocketTool`
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — `Message` + `Content`, `ChatOptions` + `ChatResponse` + `ChatResponseUpdate`, `ResponseStream`, `AgentContext`, `FunctionalWorkflow` + `StepWrapper`, `WorkflowEvent` taxonomy, `SkillsSource` composition, `EvalItem` + `EvalResults`, `TokenizerProtocol`, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor` + `@handler` + `@executor`, `AgentExecutor` + `AgentExecutorRequest` + `AgentExecutorResponse`, edge groups, `Runner` + `WorkflowMessage`, `SessionContext`, `AgentSession`, `BaseChatClient` + `SupportsChatGetResponse`, `SecretString` + `load_settings`, `WorkflowCheckpoint` + `CheckpointStorage`, exception hierarchy
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — `ExperimentalFeature`, `WorkflowRunState`, `WorkflowExecutor` + SubWorkflow pair, `AgentResponse` + `AgentResponseUpdate` + `ContinuationToken`, `BaseEmbeddingClient` family, `FunctionInvocationConfiguration`, `ClassSkill` + `FileSkillsSource` + `SkillsProvider`, `Annotation` + `TextSpanRegion`, provider capability protocols, middleware layers

This volume covers three subpackages that are shipped alongside `agent-framework-core`
but often under-documented: the **orchestrations** package (GroupChat, Handoff, Magentic,
Sequential, Concurrent builders), the **declarative** package (YAML-first agent and
workflow factories), and the **security** package (prompt injection defence via
information-flow control). It also documents the `ContextProvider` base class that
underpins every harness provider, the `FunctionalWorkflowAgent` adapter, and the
`ObservabilitySettings` / `configure_otel_providers` telemetry bootstrap API.

---

## Table of Contents

1. [`ContextProvider`](#1-contextprovider)
2. [`BackgroundTaskInfo` + `BackgroundTaskStatus`](#2-backgroundtaskinfo--backgroundtaskstatus)
3. [`GroupChatBuilder` + `TerminationCondition` + `GroupChatSelectionFunction`](#3-groupchatbuilder--terminationcondition--groupchatselectionfunction)
4. [`HandoffBuilder` + `HandoffConfiguration`](#4-handoffbuilder--handoffconfiguration)
5. [`MagenticBuilder` + `StandardMagenticManager` + `MagenticProgressLedger`](#5-magenticbuilder--standardmagenticmanager--magenticprogressledger)
6. [`SequentialBuilder` + `ConcurrentBuilder`](#6-sequentialbuilder--concurrentbuilder)
7. [`AgentFactory` + `WorkflowFactory` (declarative)](#7-agentfactory--workflowfactory)
8. [Security — `SecureAgentConfig` + `ContentLabel` + `IntegrityLabel` + `LabelTrackingFunctionMiddleware`](#8-security--secureagentconfig--contentlabel--integritylabel--labeltrackingfunctionmiddleware)
9. [`FunctionalWorkflowAgent`](#9-functionalworkflowagent)
10. [`ObservabilitySettings` + `configure_otel_providers`](#10-observabilitysettings--configure_otel_providers)

---

## 1. `ContextProvider`

**Source:** `agent_framework._sessions`

`ContextProvider` is the base class every first-party harness provider inherits from
(`BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`,
`SecureAgentConfig`, …). Subclass it to inject context, tools, or instructions before
each model call and to process the response after.

### Class signature

```python
class ContextProvider:
    source_id: str  # unique per provider instance; used for attribution

    def __init__(self, source_id: str) -> None: ...

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,   # mutate this to add messages / tools / middleware
        state: dict[str, Any],     # provider-scoped mutable state dict
    ) -> None: ...

    async def after_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,   # context.response is populated here
        state: dict[str, Any],
    ) -> None: ...
```

Both lifecycle hooks are optional — override only the ones you need.

### Custom provider — injecting a system message

```python
import asyncio
from agent_framework import Agent, ContextProvider, AgentSession, SessionContext
from agent_framework._types import Message, Content
from agent_framework.openai import OpenAIChatClient

class WeatherContextProvider(ContextProvider):
    """Prepends today's weather to every invocation."""

    def __init__(self, city: str) -> None:
        super().__init__(source_id="weather_context")
        self._city = city

    async def before_run(
        self, *, agent, session: AgentSession, context: SessionContext, state
    ) -> None:
        weather = await self._fetch_weather(self._city)
        context.add_message(
            Message(
                role="system",
                content=[Content.from_text(f"Current weather in {self._city}: {weather}")],
                source=self.source_id,
            )
        )

    async def _fetch_weather(self, city: str) -> str:
        # stub — replace with a real API call
        return "22 °C, partly cloudy"


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a helpful travel assistant.",
        context_providers=[WeatherContextProvider("London")],
    )
    response = await agent.run("Should I bring an umbrella?")
    print(response.text)

asyncio.run(main())
```

### Custom provider — recording response latency

```python
import time
from agent_framework import ContextProvider

class LatencyRecorderProvider(ContextProvider):
    def __init__(self) -> None:
        super().__init__(source_id="latency_recorder")
        self.last_latency_ms: float | None = None

    async def before_run(self, *, agent, session, context, state) -> None:
        state["t0"] = time.perf_counter()

    async def after_run(self, *, agent, session, context, state) -> None:
        elapsed = (time.perf_counter() - state["t0"]) * 1000
        self.last_latency_ms = elapsed
        print(f"[latency] {elapsed:.1f} ms")
```

### Key design rules

- `state` is provider-scoped — it is reset per invocation by default and is **not** shared
  between providers. Cross-provider read access is via `session.state`.
- `context.response` is `None` during `before_run` and populated during `after_run`.
- Multiple providers in `context_providers=[...]` run in **list order** for `before_run`
  and in **reverse list order** for `after_run`.

---

## 2. `BackgroundTaskInfo` + `BackgroundTaskStatus`

**Source:** `agent_framework._harness._background_agents`

These two classes are the data model that `BackgroundAgentsProvider` uses internally and
also exposes to the LLM as serialised state. Knowing their fields lets you inspect
background-task state programmatically from host code.

### `BackgroundTaskStatus` enum

```python
class BackgroundTaskStatus(str, Enum):
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    LOST      = "lost"   # task runner crashed; result will never arrive
```

### `BackgroundTaskInfo` dataclass

```python
class BackgroundTaskInfo:          # SerializationMixin — round-trips to/from dict
    id:           int              # auto-assigned integer handle
    agent_name:   str              # name of the child agent that ran the task
    description:  str              # user-supplied task description
    status:       BackgroundTaskStatus
    result_text:  str | None       # populated on COMPLETED
    error_text:   str | None       # populated on FAILED

    def to_dict(
        self,
        *,
        exclude: set[str] | None = None,
        exclude_none: bool = True,
    ) -> dict[str, Any]: ...
```

### Reading task state from host code

`BackgroundAgentsProvider` persists task state in `session.state` under
`source_id` (default `"background_agents"`). You can read it between `agent.run` calls:

```python
import asyncio
from agent_framework import Agent, AgentSession, BackgroundTaskInfo, BackgroundTaskStatus
from agent_framework._harness._background_agents import BackgroundAgentsProvider
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    researcher = Agent(
        client=OpenAIChatClient(),
        name="researcher",
        instructions="You are a researcher. Complete the given research task.",
    )

    provider = BackgroundAgentsProvider(agents=[researcher])

    coordinator = Agent(
        client=OpenAIChatClient(),
        name="coordinator",
        instructions="Delegate research tasks to background agents and summarise results.",
        context_providers=[provider],
    )

    session = AgentSession()
    await coordinator.run(
        "Research the latest AI agent frameworks and delegate to the researcher.",
        session=session,
    )

    # Inspect task state
    raw_state = session.state.get("background_agents", {})
    for task_dict in raw_state.get("tasks", {}).values():
        task = BackgroundTaskInfo(**{
            "id": task_dict["id"],
            "agent_name": task_dict["agent_name"],
            "description": task_dict["description"],
            "status": BackgroundTaskStatus(task_dict["status"]),
            "result_text": task_dict.get("result_text"),
            "error_text": task_dict.get("error_text"),
        })
        print(f"Task {task.id} [{task.status.value}]: {task.description[:60]}")
        if task.status == BackgroundTaskStatus.COMPLETED and task.result_text:
            print(f"  → {task.result_text[:100]}")

asyncio.run(main())
```

### Handling `LOST` status

A task enters `LOST` if the session storing it is garbage-collected before the runner
reports back. Treat it as a transient failure — re-issue the same task if needed:

```python
async def retry_lost_tasks(
    agent: Agent, session: AgentSession, user_input: str
) -> None:
    state = session.state.get("background_agents", {})
    for task_dict in state.get("tasks", {}).values():
        if task_dict.get("status") == BackgroundTaskStatus.LOST.value:
            await agent.run(
                f"Task {task_dict['id']} was lost. Please re-start it: {task_dict['description']}",
                session=session,
            )
```

---

## 3. `GroupChatBuilder` + `TerminationCondition` + `GroupChatSelectionFunction`

**Source:** `agent_framework.orchestrations`  
**Package:** `agent-framework-orchestrations`

`GroupChatBuilder` wires multiple agents into a star-topology multi-agent conversation
coordinated by an orchestrator. The orchestrator decides which participant speaks next
at every round until a termination condition fires.

### Constructor summary

```python
class GroupChatBuilder:
    def __init__(
        self,
        *,
        # Participants (at least one of these required)
        participants: Sequence[SupportsAgentRun | Executor] | None = None,
        participant_factories: Sequence[Callable[[], SupportsAgentRun | Executor]] | None = None,
        # Orchestrator — exactly one required
        orchestrator_agent: Agent | Callable[[], Agent] | None = None,
        orchestrator: BaseGroupChatOrchestrator | Callable[[], BaseGroupChatOrchestrator] | None = None,
        selection_func: GroupChatSelectionFunction | None = None,
        orchestrator_name: str | None = None,
        # Termination
        termination_condition: TerminationCondition | None = None,
        max_rounds: int | None = None,
        # Persistence
        checkpoint_storage: CheckpointStorage | None = None,
        # Output routing
        output_from: Sequence[str | SupportsAgentRun] | Literal["all"] | None = ...,
        intermediate_output_from: ... = None,
    ) -> None: ...
```

### `TerminationCondition` type alias

```python
# Sync or async callable: receives the full conversation history → returns True to stop
TerminationCondition = Callable[[list[Message]], bool | Awaitable[bool]]
```

### `GroupChatSelectionFunction` type alias

```python
# Receives current GroupChatState → returns the next participant's name (str)
GroupChatSelectionFunction = Callable[[GroupChatState], str | Awaitable[str]]
```

### Pattern 1 — LLM orchestrator selects next speaker

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import GroupChatBuilder

async def main() -> None:
    client = OpenAIChatClient()

    planner = Agent(client=client, name="planner",
                    instructions="You break complex tasks into clear sub-tasks.")
    writer  = Agent(client=client, name="writer",
                    instructions="You write well-structured content.")
    critic  = Agent(client=client, name="critic",
                    instructions="You critique content and suggest improvements.")
    orchestrator = Agent(
        client=client,
        name="orchestrator",
        instructions=(
            "You coordinate a group of specialists. Decide who should speak next "
            "based on the conversation. Reply with ONLY the agent name."
        ),
    )

    workflow = (
        GroupChatBuilder(
            participants=[planner, writer, critic],
            orchestrator_agent=orchestrator,
            max_rounds=10,
        )
        .build()
    )

    async for event in workflow.run("Write a blog post about AI agents.", stream=True):
        if hasattr(event, "message"):
            print(f"[{event.message.source}] {event.message.text[:80]}")

asyncio.run(main())
```

### Pattern 2 — Round-robin selection with custom termination

```python
from agent_framework import Message
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState

def round_robin(state: GroupChatState) -> str:
    participants = state.participant_names           # ordered list
    last_speaker = state.last_message.source if state.last_message else None
    idx = (participants.index(last_speaker) + 1) % len(participants) if last_speaker in participants else 0
    return participants[idx]

def terminate_on_done(messages: list[Message]) -> bool:
    return any("DONE" in (m.text or "") for m in messages[-3:])

workflow = GroupChatBuilder(
    participants=[agent_a, agent_b, agent_c],
    selection_func=round_robin,
    termination_condition=terminate_on_done,
    max_rounds=20,
).build()
```

### Pattern 3 — Human-in-the-loop pause via `request_info`

```python
from agent_framework.orchestrations import GroupChatBuilder
from agent_framework import FileCheckpointStorage

storage = FileCheckpointStorage(path="./checkpoints")
workflow = GroupChatBuilder(
    participants=[agent_a, agent_b],
    orchestrator_agent=orchestrator,
    checkpoint_storage=storage,
).build()

# First pass — will yield a WorkflowEvent with request_info if agents ask
events = []
async for event in workflow.run("Analyse this dataset.", stream=True):
    events.append(event)
    if hasattr(event, "request_info"):
        print("Agent needs input:", event.request_info.question)
        # Resume later:
        # async for e in workflow.run(..., responses={event.request_info.id: "Yes"}, checkpoint_id=..., stream=True):
        break
```

---

## 4. `HandoffBuilder` + `HandoffConfiguration`

**Source:** `agent_framework.orchestrations`

`HandoffBuilder` creates workflows where agents route control to each other by calling
a generated `transfer_to_<agent_name>` tool. Routing is explicit and deterministic —
unlike GroupChat there is no external orchestrator.

### Constructor summary

```python
class HandoffBuilder:
    def __init__(
        self,
        *,
        name: str | None = None,
        participants: Sequence[Agent] | None = None,
        description: str | None = None,
        checkpoint_storage: CheckpointStorage | None = None,
        termination_condition: TerminationCondition | None = None,
        output_from: Sequence[str | Agent] | Literal["all"] | None = ...,
        intermediate_output_from: ... = None,
    ) -> None: ...

    # Fluent API methods
    def participants(self, agents: Sequence[Agent]) -> "HandoffBuilder": ...
    def with_handoff(
        self,
        source: str | Agent,
        target: str | Agent,
        *,
        description: str | None = None,
    ) -> "HandoffBuilder": ...
    def start_with(self, agent: str | Agent) -> "HandoffBuilder": ...
    def with_autonomous_mode(
        self,
        agent: str | Agent,
        *,
        prompt: str | None = None,
        turn_limit: int = 5,
    ) -> "HandoffBuilder": ...
    def build(self) -> Workflow: ...
```

### `HandoffConfiguration` dataclass

```python
@dataclass
class HandoffConfiguration:
    target_id: str             # resolved agent identifier
    description: str | None   # shown to the LLM when choosing handoffs

    def __init__(self, *, target: str | SupportsAgentRun, description: str | None = None) -> None: ...
```

### Complete handoff workflow example

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import HandoffBuilder

async def main() -> None:
    client = OpenAIChatClient()

    triage = Agent(
        client=client,
        name="triage",
        instructions=(
            "You assess incoming requests. Route billing questions to billing_agent, "
            "technical issues to support_agent, and general questions to general_agent."
        ),
    )
    billing = Agent(
        client=client,
        name="billing_agent",
        instructions="You handle billing, invoices, and subscription questions.",
    )
    support = Agent(
        client=client,
        name="support_agent",
        instructions="You handle technical support issues and bug reports.",
    )
    general = Agent(
        client=client,
        name="general_agent",
        instructions="You handle general enquiries.",
    )

    workflow = (
        HandoffBuilder(participants=[triage, billing, support, general])
        .start_with(triage)
        .with_handoff(triage, billing, description="Billing or invoice questions")
        .with_handoff(triage, support, description="Technical issues or bug reports")
        .with_handoff(triage, general, description="All other questions")
        .with_handoff(billing, triage, description="Route back to triage")
        .with_handoff(support, triage, description="Route back to triage")
        .build()
    )

    async for event in workflow.run("My invoice looks wrong — can you help?", stream=True):
        if hasattr(event, "message") and event.message:
            print(f"[{event.message.source}] {event.message.text}")

asyncio.run(main())
```

### Autonomous mode — agent loops without human input

When you call `.with_autonomous_mode(agent, turn_limit=N)`, the named agent can take
multiple internal turns (tool calls, reasoning) before yielding control back to the
workflow. This is useful for research sub-agents that need to iterate:

```python
workflow = (
    HandoffBuilder(participants=[coordinator, researcher])
    .start_with(coordinator)
    .with_handoff(coordinator, researcher, description="Delegate research tasks")
    .with_handoff(researcher, coordinator, description="Return results to coordinator")
    .with_autonomous_mode(researcher, prompt="Iterate until you have a complete answer.", turn_limit=8)
    .build()
)
```

---

## 5. `MagenticBuilder` + `StandardMagenticManager` + `MagenticProgressLedger`

**Source:** `agent_framework.orchestrations`

The Magentic builder implements Magentic-One: an LLM manager that creates a **task
ledger** (facts + plan), coordinates participants via a **progress ledger**, detects
stalls, replans when stuck, and synthesises a final answer.

### `MagenticBuilder` constructor summary

```python
class MagenticBuilder:
    def __init__(
        self,
        *,
        participants: Sequence[SupportsAgentRun | Executor],
        # Manager — exactly one required
        manager: MagenticManagerBase | None = None,
        manager_factory: Callable[[], MagenticManagerBase] | None = None,
        manager_agent: SupportsAgentRun | None = None,
        manager_agent_factory: Callable[[], SupportsAgentRun] | None = None,
        # StandardMagenticManager prompt overrides (only with manager_agent/factory)
        task_ledger_facts_prompt: str | None = None,
        task_ledger_plan_prompt: str | None = None,
        task_ledger_full_prompt: str | None = None,
        task_ledger_facts_update_prompt: str | None = None,
        task_ledger_plan_update_prompt: str | None = None,
        progress_ledger_prompt: str | None = None,
        final_answer_prompt: str | None = None,
        # Stall / reset thresholds
        max_stall_count: int = 3,
        max_reset_count: int | None = None,
        max_round_count: int | None = None,
        # HITL
        enable_plan_review: bool = False,
        # Persistence
        checkpoint_storage: CheckpointStorage | None = None,
        # Output routing
        output_from: Sequence[str | SupportsAgentRun] | Literal["all"] | None = ...,
        intermediate_output_from: ... = None,
    ) -> None: ...
```

### `MagenticProgressLedger` (internal data model)

The manager emits a structured JSON progress ledger at every round. Knowing its shape
helps when you override `progress_ledger_prompt` or inspect debug logs:

```python
@dataclass
class MagenticProgressLedger:
    is_request_satisfied:    MagenticProgressLedgerItem  # answer: bool
    is_in_loop:              MagenticProgressLedgerItem  # answer: bool
    is_progress_being_made:  MagenticProgressLedgerItem  # answer: bool
    next_speaker:            MagenticProgressLedgerItem  # answer: str (participant name)
    instruction_or_question: MagenticProgressLedgerItem  # answer: str

@dataclass
class MagenticProgressLedgerItem:
    reason: str         # LLM's chain-of-thought
    answer: str | bool  # the decision
```

### `StandardMagenticManager` constructor

```python
class StandardMagenticManager(MagenticManagerBase):
    def __init__(
        self,
        agent: SupportsAgentRun,   # the LLM-backed manager agent
        task_ledger: _MagenticTaskLedger | None = None,
        *,
        # Prompt overrides — all default to built-in Magentic-One prompts
        task_ledger_facts_prompt: str | None = None,
        task_ledger_plan_prompt: str | None = None,
        task_ledger_full_prompt: str | None = None,
        task_ledger_facts_update_prompt: str | None = None,
        task_ledger_plan_update_prompt: str | None = None,
        progress_ledger_prompt: str | None = None,
        final_answer_prompt: str | None = None,
        max_stall_count: int = 3,
        max_reset_count: int | None = None,
        max_round_count: int | None = None,
        progress_ledger_retry_count: int | None = None,
    ) -> None: ...
```

### Complete Magentic-One example

```python
import asyncio
from agent_framework import Agent, FileCheckpointStorage
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import MagenticBuilder

async def main() -> None:
    client = OpenAIChatClient()

    manager_agent = Agent(
        client=client,
        name="manager",
        instructions="You are an orchestrator. Coordinate agents to complete the task.",
    )
    web_surfer = Agent(
        client=client,
        name="web_surfer",
        instructions="You retrieve web pages and extract relevant information.",
    )
    coder = Agent(
        client=client,
        name="coder",
        instructions="You write and execute Python code to analyse data.",
    )
    file_handler = Agent(
        client=client,
        name="file_handler",
        instructions="You read and write files.",
    )

    storage = FileCheckpointStorage(path="./magentic_checkpoints")

    workflow = MagenticBuilder(
        participants=[web_surfer, coder, file_handler],
        manager_agent=manager_agent,
        max_stall_count=3,
        max_reset_count=2,
        max_round_count=30,
        enable_plan_review=True,      # pause before execution for human review
        checkpoint_storage=storage,
    ).build()

    async for event in workflow.run(
        "Analyse the latest GitHub trending repos and produce a markdown summary.",
        stream=True,
    ):
        # MagenticPlanReviewRequest fires if enable_plan_review=True
        if hasattr(event, "plan_review_request"):
            print("Plan:", event.plan_review_request.plan)
            # In a real app, suspend here and resume with approval:
            # async for e in workflow.run(..., responses={event.plan_review_request.id: "APPROVE"}, ...):
            break
        if hasattr(event, "message") and event.message:
            print(f"[{event.message.source}] {event.message.text[:100]}")

asyncio.run(main())
```

### Overriding the progress ledger prompt

Use `progress_ledger_prompt` to change how the manager evaluates round progress.
The template must produce a JSON object matching the `MagenticProgressLedger` schema:

```python
CUSTOM_PROGRESS_PROMPT = """
Review the conversation and output JSON with keys:
is_request_satisfied, is_in_loop, is_progress_being_made,
next_speaker, instruction_or_question.
Each key maps to {"reason": "<reasoning>", "answer": <value>}.
Be concise. next_speaker.answer must be one of: {participant_names}.
"""

workflow = MagenticBuilder(
    participants=[agent_a, agent_b],
    manager_agent=manager_agent,
    progress_ledger_prompt=CUSTOM_PROGRESS_PROMPT,
).build()
```

---

## 6. `SequentialBuilder` + `ConcurrentBuilder`

**Source:** `agent_framework.orchestrations`

These two builders are the simplest orchestration primitives — they do not involve a
separate orchestrator agent and require no routing logic.

### `SequentialBuilder` — run agents one after another

```python
class SequentialBuilder:
    def __init__(
        self,
        *,
        participants: Sequence[SupportsAgentRun | Executor],
        checkpoint_storage: CheckpointStorage | None = None,
        chain_only_agent_responses: bool = False,
        output_from: Sequence[str | SupportsAgentRun] | Literal["all"] | None = ...,
        intermediate_output_from: ... = None,
    ) -> None: ...

    def build(self) -> Workflow: ...
```

`chain_only_agent_responses=True` strips tool-call messages from the context before
passing to the next agent — useful when intermediate tool traffic is noisy.

### `ConcurrentBuilder` — run agents in parallel, then collect

```python
from agent_framework.orchestrations import ConcurrentBuilder

class ConcurrentBuilder:
    def __init__(
        self,
        *,
        participants: Sequence[SupportsAgentRun | Executor],
        checkpoint_storage: CheckpointStorage | None = None,
        output_from: Sequence[str | SupportsAgentRun] | Literal["all"] | None = ...,
        intermediate_output_from: ... = None,
    ) -> None: ...

    def build(self) -> Workflow: ...
```

### Sequential pipeline example

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder

async def main() -> None:
    client = OpenAIChatClient()

    extractor = Agent(
        client=client,
        name="extractor",
        instructions="Extract key facts from the provided text as a bullet list.",
    )
    summariser = Agent(
        client=client,
        name="summariser",
        instructions="Summarise the bullet list you receive into one paragraph.",
    )
    translator = Agent(
        client=client,
        name="translator",
        instructions="Translate the text you receive into French.",
    )

    workflow = SequentialBuilder(
        participants=[extractor, summariser, translator],
        chain_only_agent_responses=True,  # pass only each agent's final reply downstream
    ).build()

    result = await workflow.run(long_article_text)
    print(result.text)

asyncio.run(main())
```

### Concurrent fan-out + synthesis example

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import ConcurrentBuilder, SequentialBuilder

async def main() -> None:
    client = OpenAIChatClient()

    # Three agents run in parallel, all receive the same input
    analyst_1 = Agent(client=client, name="analyst_market",
                      instructions="Analyse the market opportunity.")
    analyst_2 = Agent(client=client, name="analyst_tech",
                      instructions="Analyse the technical feasibility.")
    analyst_3 = Agent(client=client, name="analyst_risk",
                      instructions="Assess key risks.")
    synthesiser = Agent(
        client=client,
        name="synthesiser",
        instructions="Synthesise the three analyses into one executive summary.",
    )

    # Fan-out → collect all three in parallel, then synthesise sequentially
    fan_out = ConcurrentBuilder(participants=[analyst_1, analyst_2, analyst_3]).build()
    pipeline = SequentialBuilder(participants=[fan_out, synthesiser]).build()

    result = await pipeline.run("Evaluate launching a new B2B SaaS product in 2026.")
    print(result.text)

asyncio.run(main())
```

---

## 7. `AgentFactory` + `WorkflowFactory`

**Source:** `agent_framework.declarative`  
**Package:** `agent-framework-declarative`

The declarative package lets you define agents and workflows in YAML files. At runtime
the factories parse the YAML and construct fully wired `Agent` / `Workflow` objects.

### `AgentFactory` constructor

```python
class AgentFactory:
    def __init__(
        self,
        *,
        client: SupportsChatGetResponse | None = None,
        bindings: Mapping[str, Any] | None = None,
        connections: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        additional_mappings: Mapping[str, ProviderTypeMapping] | None = None,
        default_provider: str = "Foundry",
        safe_mode: bool = True,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

    def create_agent_from_yaml(self, yaml_content: str) -> Agent: ...
    def create_agent_from_yaml_path(self, path: str) -> Agent: ...
```

`safe_mode=True` (default) disables PowerFx expression evaluation in tool arguments,
preventing injection via external YAML.

### `WorkflowFactory` constructor

```python
class WorkflowFactory:
    def __init__(
        self,
        *,
        client: SupportsChatGetResponse | None = None,
        checkpoint_storage: CheckpointStorage | None = None,
        bindings: Mapping[str, Any] | None = None,
        connections: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        additional_mappings: Mapping[str, ProviderTypeMapping] | None = None,
        default_provider: str = "Foundry",
        safe_mode: bool = True,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None: ...

    def create_workflow_from_yaml(self, yaml_content: str) -> Workflow: ...
    def create_workflow_from_yaml_path(self, path: str) -> Workflow: ...
```

### YAML agent definition (`agent.yaml`)

```yaml
kind: Prompt
name: ResearchAgent
instructions: |
  You are a research assistant. Answer questions with citations
  from reliable sources.
model:
  id: gpt-4o
  provider: AzureOpenAI
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_key: ${AZURE_OPENAI_API_KEY}
response_format: text
tools:
  - name: web_search
    description: Search the web for current information
    parameters:
      type: object
      properties:
        query:
          type: string
          description: Search query
      required: [query]
```

### Loading an agent from YAML

```python
import asyncio
from agent_framework.declarative import AgentFactory

async def main() -> None:
    factory = AgentFactory(safe_mode=True)
    agent = factory.create_agent_from_yaml_path("agent.yaml")

    response = await agent.run("What are the top AI papers from 2026?")
    print(response.text)

asyncio.run(main())
```

### YAML workflow definition (`workflow.yaml`)

```yaml
kind: Workflow
name: ResearchPipeline
actions:
  - kind: InvokeAgent
    name: research
    agent: ResearchAgent
    input: "{{Workflow.Inputs.query}}"
    output: Local.research_result

  - kind: InvokeAgent
    name: summarise
    agent: SummariserAgent
    input: "{{Local.research_result}}"
    output: Workflow.Outputs.summary
```

### Loading a workflow from YAML with checkpointing

```python
import asyncio
from agent_framework import FileCheckpointStorage
from agent_framework.declarative import WorkflowFactory

async def main() -> None:
    storage = FileCheckpointStorage(path="./workflow_checkpoints")
    factory = WorkflowFactory(checkpoint_storage=storage)
    workflow = factory.create_workflow_from_yaml_path("workflow.yaml")

    async for event in workflow.run({"query": "Latest advances in quantum computing"}, stream=True):
        if hasattr(event, "output"):
            print("Summary:", event.output)

asyncio.run(main())
```

### `ProviderTypeMapping` TypedDict

Register custom chat clients as YAML provider names:

```python
from agent_framework.declarative import AgentFactory, ProviderTypeMapping

my_provider: ProviderTypeMapping = {
    "package": "my_pkg.clients",
    "name": "MyChatClient",
    "model_field": "model",
    "endpoint_field": "endpoint",
    "api_key_field": "api_key",
}

factory = AgentFactory(additional_mappings={"MyProvider": my_provider})
# Now YAML can use: provider: MyProvider
```

---

## 8. Security — `SecureAgentConfig` + `ContentLabel` + `IntegrityLabel` + `LabelTrackingFunctionMiddleware`

**Source:** `agent_framework.security`  
**Feature gate:** `ExperimentalFeature.FIDES`  
**Package:** `agent-framework-core` (security.py)

The security module provides **information-flow control** (IFC) to defend against
prompt injection. Every piece of content flowing through the agent is tagged with an
`IntegrityLabel` (trusted / untrusted) and a `ConfidentialityLabel` (public / private /
user_identity). Middleware tracks label propagation and can block or quarantine
untrusted content before it influences tool calls.

### Core security primitives

```python
from agent_framework.security import (
    IntegrityLabel,          # trusted | untrusted
    ConfidentialityLabel,    # public | private | user_identity
    ContentLabel,            # combines both labels + optional metadata
    ContentVariableStore,    # maps var IDs → (content, label) without exposing raw content to LLM
    LabeledMessage,          # Message subclass carrying a ContentLabel
)

class IntegrityLabel(str, Enum):
    TRUSTED   = "trusted"
    UNTRUSTED = "untrusted"

class ConfidentialityLabel(str, Enum):
    PUBLIC        = "public"
    PRIVATE       = "private"
    USER_IDENTITY = "user_identity"

class ContentLabel:
    integrity:      IntegrityLabel
    confidentiality: ConfidentialityLabel
    metadata:       dict[str, Any] | None

    def __init__(
        self,
        integrity: IntegrityLabel = IntegrityLabel.TRUSTED,
        confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
```

### `SecureAgentConfig` constructor

```python
class SecureAgentConfig(ContextProvider):  # also a ContextProvider!
    DEFAULT_SOURCE_ID = "secure_agent"

    def __init__(
        self,
        auto_hide_untrusted: bool = True,
        default_integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED,
        default_confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC,
        allow_untrusted_tools: set[str] | None = None,
        block_on_violation: bool = True,
        approval_on_violation: bool = False,
        enable_audit_log: bool = True,
        enable_policy_enforcement: bool = True,
        quarantine_chat_client: SupportsChatGetResponse | None = None,
        source_id: str | None = None,
    ) -> None: ...
```

| Parameter | Meaning |
|---|---|
| `auto_hide_untrusted` | Replace untrusted content bodies with variable references before sending to the LLM |
| `default_integrity` | Label applied to all incoming tool results unless overridden |
| `allow_untrusted_tools` | Tool names that may run even in an untrusted context |
| `block_on_violation` | Raise `MiddlewareTermination` when policy is violated |
| `approval_on_violation` | Instead of blocking, emit a `request_info` for human approval |
| `quarantine_chat_client` | Chat client used to process untrusted content in isolation |

### `LabelTrackingFunctionMiddleware` — 3-tier label propagation

Labels on tool results follow a strict priority:

| Tier | Source | Wins when |
|---|---|---|
| 1 (highest) | `additional_properties.security_label` on each result item | Always |
| 2 | Tool's `source_integrity` declaration in `additional_properties` | No embedded labels |
| 3 (lowest) | Join of input argument labels | No tier 1 or 2 |

Tools that fetch external data should declare `source_integrity="untrusted"`:

```python
from agent_framework import tool

@tool(additional_properties={"source_integrity": "untrusted"})
async def fetch_external_data(url: str) -> str:
    """Fetch content from an external URL."""
    import httpx
    resp = httpx.get(url, timeout=10)
    return resp.text
```

### Minimal secure agent

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import SecureAgentConfig, IntegrityLabel, ConfidentialityLabel

async def main() -> None:
    security = SecureAgentConfig(
        auto_hide_untrusted=True,
        default_integrity=IntegrityLabel.UNTRUSTED,    # assume tool results are untrusted by default
        allow_untrusted_tools={"search_internal_kb"},  # this tool is always trusted
        block_on_violation=True,
        enable_audit_log=True,
    )

    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a customer support agent. Never leak confidential data.",
        tools=[fetch_external_data, search_internal_kb],
        context_providers=[security],
    )

    response = await agent.run("Summarise the content at https://example.com")
    print(response.text)

asyncio.run(main())
```

### `ContentVariableStore` — preventing raw untrusted content from reaching the LLM

```python
from agent_framework.security import ContentVariableStore, ContentLabel, IntegrityLabel

store = ContentVariableStore()

# Store untrusted external content — returns an opaque variable reference
untrusted_label = ContentLabel(integrity=IntegrityLabel.UNTRUSTED)
var_id = store.store("potentially malicious instructions here", untrusted_label)

# The LLM only sees the var_id (e.g. "var_a3f2c1d8"), not the raw content
print(var_id)    # "var_a3f2c1d8..."

# Retrieve when you need the actual value
content, label = store.retrieve(var_id)
```

### Approval on violation pattern

```python
security = SecureAgentConfig(
    block_on_violation=False,    # do not hard-block
    approval_on_violation=True,  # surface a request_info instead
)

agent = Agent(client=OpenAIChatClient(), context_providers=[security])

async for event in agent.run("Process this external link...", stream=True):
    if hasattr(event, "request_info") and "security" in event.request_info.source_id:
        decision = await ask_human(event.request_info.question)
        # Resume with: await agent.run(..., responses={event.request_info.id: decision})
        break
```

---

## 9. `FunctionalWorkflowAgent`

**Source:** `agent_framework._workflows._functional`  
**Feature gate:** `ExperimentalFeature.FUNCTIONAL_WORKFLOWS`

`FunctionalWorkflowAgent` wraps a `FunctionalWorkflow` (written with `@workflow` / `@step`)
in an `Agent`-compatible interface, letting you use a functional workflow anywhere an
agent is expected — as a participant in orchestration builders, inside a `WorkflowBuilder`
graph, or as a top-level callable.

### Class signature

```python
class FunctionalWorkflowAgent:
    REQUEST_INFO_FUNCTION_NAME: str = "request_info"

    def __init__(
        self,
        workflow: FunctionalWorkflow,
        *,
        name: str | None = None,              # defaults to workflow.name
        description: str | None = None,       # defaults to workflow.description
        context_providers: Sequence[Any] | None = None,
        **kwargs: Any,                         # ignored; accepted for API parity
    ) -> None: ...

    @property
    def pending_requests(self) -> dict[str, WorkflowEvent[Any]]:
        """request_info events emitted during the last run."""
        ...

    # run() overloads mirror Agent.run() exactly:
    async def run(
        self,
        messages: Any | None = None,
        *,
        stream: bool = False,
        responses: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
        checkpoint_storage: CheckpointStorage | None = None,
        **kwargs: Any,
    ) -> AgentResponse | ResponseStream[AgentResponseUpdate, AgentResponse]: ...
```

### Creating a functional workflow and wrapping as an agent

```python
import asyncio
from agent_framework import Agent, RunContext, FunctionalWorkflow, FunctionalWorkflowAgent
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._functional import workflow, step

client = OpenAIChatClient()

@step
async def gather_requirements(ctx: RunContext, user_input: str) -> str:
    """Step 1: clarify requirements with the user."""
    agent = Agent(client=client, instructions="Clarify the user's coding task.")
    r = await agent.run(user_input, session=ctx.session)
    return r.text

@step
async def write_code(ctx: RunContext, requirements: str) -> str:
    """Step 2: write the code."""
    agent = Agent(client=client, instructions="Write Python code for the requirements.")
    r = await agent.run(requirements, session=ctx.session)
    return r.text

@step
async def review_code(ctx: RunContext, code: str) -> str:
    """Step 3: review and return feedback."""
    agent = Agent(client=client, instructions="Review the Python code for bugs.")
    r = await agent.run(code, session=ctx.session)
    return r.text

@workflow(name="coding_pipeline")
async def coding_pipeline(ctx: RunContext, user_request: str) -> str:
    reqs  = await gather_requirements(ctx, user_request)
    code  = await write_code(ctx, reqs)
    review = await review_code(ctx, code)
    return f"Code:\n{code}\n\nReview:\n{review}"

fw = FunctionalWorkflow(steps=[gather_requirements, write_code, review_code], workflow_fn=coding_pipeline)
fw_agent = FunctionalWorkflowAgent(fw, name="coding_pipeline", description="End-to-end coding assistant")
```

### Accessing `pending_requests` for HITL

```python
async def run_with_hitl(fw_agent: FunctionalWorkflowAgent, task: str) -> None:
    async for event in fw_agent.run(task, stream=True):
        if hasattr(event, "request_info"):
            question = event.request_info.question
            answer = input(f"Agent asks: {question}\nYour answer: ")
            async for e in fw_agent.run(
                task,
                responses={event.request_info.id: answer},
                checkpoint_id=event.checkpoint_id,
                stream=True,
            ):
                if hasattr(e, "message"):
                    print(e.message.text)
            return
        if hasattr(event, "message"):
            print(event.message.text)
```

### Using `FunctionalWorkflowAgent` as an orchestration participant

Because `FunctionalWorkflowAgent` implements the same `run()` interface as `Agent`, it
drops straight into any builder that accepts `SupportsAgentRun`:

```python
from agent_framework.orchestrations import HandoffBuilder

dispatcher = Agent(
    client=OpenAIChatClient(),
    name="dispatcher",
    instructions="Route coding tasks to the coding pipeline.",
)

workflow = (
    HandoffBuilder(participants=[dispatcher, fw_agent])
    .start_with(dispatcher)
    .with_handoff(dispatcher, fw_agent, description="Coding tasks")
    .with_handoff(fw_agent, dispatcher, description="Return result")
    .build()
)
```

---

## 10. `ObservabilitySettings` + `configure_otel_providers`

**Source:** `agent_framework.observability`

Two APIs govern telemetry bootstrap. `ObservabilitySettings` reads configuration from
environment variables or `.env` files. `configure_otel_providers` wires OpenTelemetry
providers (traces, metrics, logs) and enables framework instrumentation in one call.

### `ObservabilitySettings`

```python
class ObservabilitySettings:
    """Reads observability settings from env vars or .env file."""

    def __init__(
        self,
        *,
        enable_instrumentation: bool = True,   # ENABLE_INSTRUMENTATION env var
        enable_sensitive_data: bool = False,    # ENABLE_SENSITIVE_DATA env var
        enable_console_exporters: bool = False, # ENABLE_CONSOLE_EXPORTERS env var
        vs_code_extension_port: int | None = None,  # VS_CODE_EXTENSION_PORT env var
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        **kwargs: Any,
    ) -> None: ...
```

Reading from `.env`:

```env
ENABLE_INSTRUMENTATION=true
ENABLE_SENSITIVE_DATA=false
ENABLE_CONSOLE_EXPORTERS=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
```

```python
from agent_framework import ObservabilitySettings

settings = ObservabilitySettings(env_file_path=".env")
print(settings.enable_instrumentation)   # True
print(settings.enable_console_exporters) # True
```

### `configure_otel_providers`

```python
def configure_otel_providers(
    *,
    enable_sensitive_data: bool | None = None,
    enable_console_exporters: bool | None = None,
    exporters: list[LogRecordExporter | SpanExporter | MetricExporter] | None = None,
    views: list[View] | None = None,
    vs_code_extension_port: int | None = None,
    env_file_path: str | None = None,
    env_file_encoding: str | None = None,
) -> None: ...
```

Reads standard OTLP env vars automatically:
- `OTEL_EXPORTER_OTLP_ENDPOINT` — base endpoint for all signals
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `_METRICS_ENDPOINT` / `_LOGS_ENDPOINT`
- `OTEL_EXPORTER_OTLP_PROTOCOL` — `grpc` or `http`
- `OTEL_EXPORTER_OTLP_HEADERS`

Call this **once at application startup** before any agent runs. Calling it multiple times
leads to duplicate exporters.

### Minimal OTLP setup (Azure Monitor)

```python
from azure.monitor.opentelemetry import configure_azure_monitor
from agent_framework.observability import enable_instrumentation, enable_sensitive_telemetry

# Use Azure Monitor's own provider setup
configure_azure_monitor(connection_string="InstrumentationKey=...")

# Then opt into framework-level instrumentation
enable_instrumentation()

# In dev only — captures message content, tool arguments, model responses
# enable_sensitive_telemetry()
```

### Minimal OTLP setup (Jaeger / any OTLP collector)

```python
import os
from agent_framework.observability import configure_otel_providers

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"

configure_otel_providers(
    enable_sensitive_data=False,       # never in production
    enable_console_exporters=True,     # useful during local dev
)
```

### Custom exporter + metric view filter

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.metrics import view as metric_view
from agent_framework.observability import configure_otel_providers

# Only collect agent_framework.* metrics; drop gen_ai.* to reduce cardinality
drop_gen_ai = metric_view.View(
    instrument_name="gen_ai.*",
    aggregation=metric_view.DropAggregation(),
)

configure_otel_providers(
    exporters=[ConsoleSpanExporter()],
    views=[drop_gen_ai],
    enable_sensitive_data=False,
)
```

### Enabling / disabling instrumentation at runtime

```python
from agent_framework.observability import (
    enable_instrumentation,
    disable_instrumentation,
    enable_sensitive_telemetry,
)

enable_instrumentation()      # start capturing spans + metrics
# ... run agents ...
disable_instrumentation()     # stop capturing (e.g. for test isolation)

# Sensitive data (message bodies, tool args) — DEV/TEST only
enable_sensitive_telemetry()
```

---

## Summary

| # | Class / Group | Package | Key takeaway |
|---|---|---|---|
| 1 | `ContextProvider` | `agent-framework-core` | Subclass with `before_run`/`after_run` to inject context, tools, and middleware; `state` is provider-scoped and resets per invocation |
| 2 | `BackgroundTaskInfo` + `BackgroundTaskStatus` | `agent-framework-core` | Data model inside `BackgroundAgentsProvider`; read via `session.state["background_agents"]["tasks"]`; handle `LOST` status with a re-issue loop |
| 3 | `GroupChatBuilder` | `agent-framework-orchestrations` | Star-topology group chat; orchestrator can be an LLM agent, a `BaseGroupChatOrchestrator`, or a `GroupChatSelectionFunction`; `TerminationCondition` is any `(list[Message]) → bool` callable |
| 4 | `HandoffBuilder` + `HandoffConfiguration` | `agent-framework-orchestrations` | Tool-based routing; agents call `transfer_to_<name>` tools; `.with_autonomous_mode()` allows N internal turns before yielding |
| 5 | `MagenticBuilder` + `StandardMagenticManager` | `agent-framework-orchestrations` | Magentic-One: task ledger + progress ledger; stall detection + reset; `enable_plan_review=True` for HITL approval before execution |
| 6 | `SequentialBuilder` + `ConcurrentBuilder` | `agent-framework-orchestrations` | Simplest orchestration primitives; compose them (ConcurrentBuilder inside SequentialBuilder) for fan-out / synthesise pipelines |
| 7 | `AgentFactory` + `WorkflowFactory` | `agent-framework-declarative` | YAML-first agent and workflow creation; `ProviderTypeMapping` to register custom clients; `safe_mode=True` disables PowerFx injection |
| 8 | `SecureAgentConfig` + `ContentLabel` | `agent-framework-core` (security.py) | IFC-based prompt injection defence; 3-tier label propagation in `LabelTrackingFunctionMiddleware`; `ContentVariableStore` keeps raw untrusted content off the LLM context |
| 9 | `FunctionalWorkflowAgent` | `agent-framework-core` | Wraps `@workflow`/`@step` functions as a drop-in `Agent`; `pending_requests` for HITL; participates in all orchestration builders |
| 10 | `ObservabilitySettings` + `configure_otel_providers` | `agent-framework-core` | One-shot OTLP bootstrap; reads env vars; custom exporters and metric `View` filters; `enable_sensitive_telemetry()` for dev environments only |

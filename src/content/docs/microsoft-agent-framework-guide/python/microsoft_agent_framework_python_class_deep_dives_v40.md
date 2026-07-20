---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 40"
description: "Source-verified deep dives into 11 class groups from agent-framework 1.11.0: MagenticOrchestrator (outer/inner loop, stall detection, require_plan_signoff HITL gate); MagenticContext+MagenticResetSignal (orchestrator state dataclass, task/round/stall/reset counts, reset trigger); MagenticPlanReviewRequest+MagenticPlanReviewResponse (HITL plan approval — approve()/revise() helpers, feedback message injection, replanning loop); StandardMagenticManager+MagenticProgressLedger (LLM-driven planning — 8 prompt overrides, max_stall_count/max_reset_count/max_round_count, 5-field progress ledger with is_request_satisfied/is_in_loop/is_progress_being_made/next_speaker/instruction); GroupChatOrchestrator (selection-function-driven round-robin — selection_func protocol, max_rounds, termination_condition); ContentVariableStore+VariableReferenceContent (prompt-injection guard — store/retrieve/exists/list_variables, variable ID indirection, ContentLabel); LabelTrackingFunctionMiddleware (3-tier integrity label propagation — embedded > source_integrity > join, auto_hide_untrusted, context_label accumulation); PolicyEnforcementFunctionMiddleware (tool execution policy — allow_untrusted_tools whitelist, block_on_violation, audit_log, approval_on_violation); FileMemoryProvider+FileStoreEntry (session-scoped file memory — 7 tools: write/read/delete/ls/grep/replace/replace_lines, scope= user grouping); AgentModeProvider (plan/execute mode switching — mode_descriptions, default_mode, mode_set/mode_get tools, get_agent_mode/set_agent_mode helpers); GAIA+GAIATelemetryConfig+Task+TaskResult+Evaluator (GAIA benchmark harness — hf_token, telemetry tracing, custom Evaluator protocol, per-task TaskResult with runtime/error) — source-verified at agent-framework 1.11.0."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 63
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 40

Verified against **agent-framework 1.11.0** (installed July 2026). Every constructor signature, parameter description, and code example was derived from the installed package source using `inspect.getsource()`.

Sub-packages introspected:
`agent_framework.orchestrations`,
`agent_framework.security`,
`agent_framework._harness._file_memory`,
`agent_framework._harness._mode`,
`agent_framework.lab.gaia`.

**Previous volumes:** [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) through [Vol. 39](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v39/) — 390+ classes covered.

This volume covers **eleven class groups** across the Magentic orchestration system, the FIDES security subsystem, session-scoped file memory, agent mode switching, and the GAIA benchmark integration.

| # | Class / group | Module |
|---|---|---|
| 1 | `MagenticOrchestrator` | `agent_framework.orchestrations` |
| 2 | `MagenticContext` · `MagenticResetSignal` | `agent_framework.orchestrations` |
| 3 | `MagenticPlanReviewRequest` · `MagenticPlanReviewResponse` | `agent_framework.orchestrations` |
| 4 | `StandardMagenticManager` · `MagenticProgressLedger` | `agent_framework.orchestrations` |
| 5 | `GroupChatOrchestrator` | `agent_framework.orchestrations` |
| 6 | `ContentVariableStore` · `VariableReferenceContent` | `agent_framework.security` |
| 7 | `LabelTrackingFunctionMiddleware` | `agent_framework.security` |
| 8 | `PolicyEnforcementFunctionMiddleware` | `agent_framework.security` |
| 9 | `FileMemoryProvider` · `FileStoreEntry` | `agent_framework._harness._file_memory` |
| 10 | `AgentModeProvider` | `agent_framework._harness._mode` |
| 11 | `GAIA` · `GAIATelemetryConfig` · `Task` · `TaskResult` · `Evaluator` | `agent_framework.lab.gaia` |

---

## 1 · `MagenticOrchestrator`

**Module:** `agent_framework.orchestrations`

`MagenticOrchestrator` drives the Magentic One-style two-level loop: an outer replanning loop around an inner coordination loop. It sits inside a `MagenticBuilder`-constructed workflow and manages stall detection, plan sign-off, progress ledger creation, and final answer synthesis.

### Constructor

```python
class MagenticOrchestrator(BaseGroupChatOrchestrator):
    def __init__(
        self,
        manager: MagenticManagerBase,
        participant_registry: ParticipantRegistry,
        *,
        require_plan_signoff: bool = False,
    ) -> None: ...
```

| Parameter | Type | Description |
|---|---|---|
| `manager` | `MagenticManagerBase` | Handles LLM calls for plan creation, progress evaluation, and replanning. Typically a `StandardMagenticManager`. |
| `participant_registry` | `ParticipantRegistry` | Populated automatically by `MagenticBuilder`; maps agent names to executor types. |
| `require_plan_signoff` | `bool` | When `True`, the orchestrator emits a `MagenticPlanReviewRequest` before executing any step. The workflow suspends until a `MagenticPlanReviewResponse` is returned (see §3). |

### The Two-Level Loop

```
outer loop (replanning)
└── inner loop (coordination)
    ├── create_progress_ledger()      ← calls manager LLM
    ├── is_request_satisfied?         ← done
    ├── stall_count > max_stall_count → reset_and_replan()
    └── send_request_to_participant(next_speaker, instruction)
```

Each inner-loop tick emits a `MagenticOrchestratorEvent` of type `PROGRESS_LEDGER_UPDATED`, `PLAN_CREATED`, or `REPLANNED` that you can consume via `add_event` observers.

### Wiring Magentic with the builder

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import (
    MagenticBuilder,
    MagenticOrchestrator,
    StandardMagenticManager,
)

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")

    # Manager agent handles all orchestration LLM calls
    manager_agent = Agent(client=client, name="manager")

    researcher = Agent(client=client, name="researcher",
                       instructions="You are a research expert. Find facts.")
    coder = Agent(client=client, name="coder",
                  instructions="You are a Python expert. Write and run code.")

    builder = (
        MagenticBuilder(StandardMagenticManager(manager_agent))
        .add_participant(researcher)
        .add_participant(coder)
        # Optional: require a human to approve the plan before execution
        # .require_plan_signoff()
    )

    workflow = builder.build()
    result = await workflow.run("Write a Python script that computes the 100th Fibonacci number.")
    print(result.text)

asyncio.run(main())
```

### Observing orchestrator events

```python
from agent_framework.orchestrations import (
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
)
from agent_framework._workflows._events import WorkflowEvent

async def event_callback(event: WorkflowEvent) -> None:
    data = event.data
    if isinstance(data, MagenticOrchestratorEvent):
        match data.event_type:
            case MagenticOrchestratorEventType.PLAN_CREATED:
                print(f"Plan: {data.content.text}")
            case MagenticOrchestratorEventType.PROGRESS_LEDGER_UPDATED:
                ledger = data.content
                print(
                    f"Round {ledger.is_request_satisfied.answer} | "
                    f"Next: {ledger.next_speaker.answer}"
                )
            case MagenticOrchestratorEventType.REPLANNED:
                print("Replanning triggered after stall detection")

# Attach via workflow.add_event_observer(event_callback) before run()
```

---

## 2 · `MagenticContext` · `MagenticResetSignal`

**Module:** `agent_framework.orchestrations`

### `MagenticContext`

`MagenticContext` is the mutable state object that `MagenticOrchestrator` maintains across all inner-loop iterations. It is a `@dataclass` that extends `DictConvertible` for serialisation.

```python
@dataclass
class MagenticContext(DictConvertible):
    task: str
    chat_history: list[Message]         = field(default_factory=list)
    participant_descriptions: dict[str, str] = field(default_factory=dict)
    round_count: int  = 0
    stall_count: int  = 0
    reset_count: int  = 0
```

| Field | Description |
|---|---|
| `task` | The original task string passed to the workflow. Never mutated after construction. |
| `chat_history` | Running message log shared with all participants and the manager's prompts. Grows with every participant response. |
| `participant_descriptions` | Name → description mapping injected into manager prompts so the LLM knows what each participant can do. |
| `round_count` | Incremented on every inner-loop tick. Used for `max_round_count` enforcement. |
| `stall_count` | Incremented when `is_progress_being_made=False` or `is_in_loop=True`; decremented (min 0) on good progress. Triggers replanning when it exceeds `max_stall_count`. |
| `reset_count` | Incremented on every outer-loop reset. Triggers hard termination when it exceeds `max_reset_count`. |

`MagenticContext` is always cloned (`context.clone(deep=True)`) before being passed to manager calls so that manager mutations cannot corrupt the orchestrator's live state.

```python
# Serialise / deserialise context (e.g. for custom checkpoint storage)
from agent_framework.orchestrations import MagenticContext

ctx = MagenticContext(task="Analyse sales data", participant_descriptions={"analyst": "Data analysis expert"})
payload = ctx.to_dict()          # JSON-serialisable dict
restored = MagenticContext.from_dict(payload)
assert restored.task == ctx.task
```

### `MagenticResetSignal`

`MagenticResetSignal` is a sentinel class with no fields. Raising it (or returning it from a custom subclass override) tells the orchestrator to exit the inner loop and trigger the outer-loop reset path: clear stale chat history, increment `reset_count`, and call `manager.replan()`.

```python
class MagenticResetSignal:
    """Signal that the Magentic workflow should reset."""
    pass
```

You will only encounter it if you subclass `BaseGroupChatOrchestrator` and override `_run_inner_loop`. End users interact with reset behaviour exclusively through `max_stall_count`.

---

## 3 · `MagenticPlanReviewRequest` · `MagenticPlanReviewResponse`

**Module:** `agent_framework.orchestrations`

When `require_plan_signoff=True` is set on `MagenticOrchestrator`, the workflow suspends before any agent is called and issues a `MagenticPlanReviewRequest` via `ctx.request_info()`. Your application must supply a `MagenticPlanReviewResponse` to resume execution.

### `MagenticPlanReviewRequest`

```python
@dataclass
class MagenticPlanReviewRequest:
    plan: Message                          # The proposed task ledger
    current_progress: MagenticProgressLedger | None  # None on first review
    is_stalled: bool                       # True when triggered by stall detection

    def approve(self) -> MagenticPlanReviewResponse: ...
    def revise(self, feedback) -> MagenticPlanReviewResponse: ...
```

The `approve()` and `revise()` convenience methods build the correct response without constructing it by hand.

### `MagenticPlanReviewResponse`

```python
@dataclass
class MagenticPlanReviewResponse:
    review: list[Message]   # Empty list = approved; non-empty = revision requested

    @staticmethod
    def approve() -> "MagenticPlanReviewResponse": ...

    @staticmethod
    def revise(feedback: str | list[str] | Message | list[Message]) -> "MagenticPlanReviewResponse": ...
```

When `review` is non-empty the orchestrator appends the messages to `chat_history`, calls `manager.replan()`, emits a `REPLANNED` event, and re-issues the review request — looping until an empty-review response is received.

### End-to-end plan sign-off

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import (
    MagenticBuilder,
    MagenticPlanReviewRequest,
    MagenticPlanReviewResponse,
    StandardMagenticManager,
)

async def human_plan_review(request: MagenticPlanReviewRequest) -> MagenticPlanReviewResponse:
    """Simulate a human reviewing the plan."""
    print("=== Proposed Plan ===")
    print(request.plan.text)
    if request.is_stalled:
        print("[Workflow was stalled — this is a replan]")

    # Mimic human decision (in production: call input() or a web API)
    approved = True
    if approved:
        return request.approve()
    else:
        return request.revise("Please include a data-validation step before analysis.")

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    manager_agent = Agent(client=client, name="manager")
    worker = Agent(client=client, name="analyst", instructions="Analyse data.")

    builder = (
        MagenticBuilder(StandardMagenticManager(manager_agent))
        .add_participant(worker)
        .require_plan_signoff()    # Equivalent to require_plan_signoff=True
    )
    workflow = builder.build()

    # Provide HITL handler before run() so it can be called during execution
    workflow.add_request_info_handler(MagenticPlanReviewRequest, human_plan_review)

    result = await workflow.run("Summarise Q1 revenue trends.")
    print(result.text)

asyncio.run(main())
```

---

## 4 · `StandardMagenticManager` · `MagenticProgressLedger`

**Module:** `agent_framework.orchestrations`

### `StandardMagenticManager`

`StandardMagenticManager` is the default `MagenticManagerBase` implementation. It wraps an `Agent` and issues structured LLM calls for every orchestration decision: initial planning, progress evaluation, replanning, and final answer synthesis.

```python
class StandardMagenticManager(MagenticManagerBase):
    def __init__(
        self,
        agent: SupportsAgentRun,
        task_ledger: _MagenticTaskLedger | None = None,
        *,
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

| Parameter | Default | Description |
|---|---|---|
| `agent` | — | Any `SupportsAgentRun` (typically `Agent`). Its chat options, temperature, and system instructions apply to all manager LLM calls. |
| `max_stall_count` | `3` | Consecutive rounds with no progress before replanning is triggered. |
| `max_reset_count` | `None` | Hard cap on outer-loop resets before the workflow terminates with failure. `None` = unlimited. |
| `max_round_count` | `None` | Hard cap on inner-loop ticks before forced termination. `None` = unlimited. |
| `progress_ledger_retry_count` | `None` | How many times to retry a failed progress-ledger LLM call before triggering a reset. |
| `*_prompt` args | `None` | Override any of the 6 built-in prompt templates. `None` = use the default Magentic One prompts. |

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import StandardMagenticManager

# Manager with custom limits and a custom progress ledger prompt
client = OpenAIChatClient("gpt-4o-mini")   # cheaper model is fine for management calls
manager_agent = Agent(client=client, name="manager")

manager = StandardMagenticManager(
    manager_agent,
    max_stall_count=2,           # replan after 2 consecutive no-progress rounds
    max_reset_count=3,           # give up after 3 resets
    max_round_count=20,          # hard cap at 20 inner-loop rounds
    progress_ledger_prompt=(
        "You are a task supervisor. Given the conversation history, "
        "answer these JSON fields: is_request_satisfied, is_in_loop, "
        "is_progress_being_made, next_speaker, instruction_or_question."
    ),
)
```

### `MagenticProgressLedger`

The progress ledger is the structured output the manager LLM returns on every inner-loop tick. It contains five `MagenticProgressLedgerItem` fields, each with an `answer` property and an optional `reason` string.

```python
@dataclass
class MagenticProgressLedger(DictConvertible):
    is_request_satisfied:    MagenticProgressLedgerItem   # bool answer
    is_in_loop:              MagenticProgressLedgerItem   # bool answer
    is_progress_being_made:  MagenticProgressLedgerItem   # bool answer
    next_speaker:            MagenticProgressLedgerItem   # str  answer (agent name)
    instruction_or_question: MagenticProgressLedgerItem   # str  answer
```

| Field | Type of `answer` | Orchestrator action when `True` / non-empty |
|---|---|---|
| `is_request_satisfied` | `bool` | Exit inner loop, call `_prepare_final_answer()` |
| `is_in_loop` | `bool` | Increment `stall_count` |
| `is_progress_being_made` | `bool` | Decrement `stall_count` if `True`; increment if `False` |
| `next_speaker` | `str` | Name of the agent to call next |
| `instruction_or_question` | `str` | Injected into the agent's context as an assistant message before the request |

```python
# Access the ledger inside a custom event observer
from agent_framework.orchestrations import (
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
    MagenticProgressLedger,
)

async def on_event(event):
    if isinstance(event.data, MagenticOrchestratorEvent):
        if event.data.event_type == MagenticOrchestratorEventType.PROGRESS_LEDGER_UPDATED:
            ledger: MagenticProgressLedger = event.data.content
            print(f"Satisfied: {ledger.is_request_satisfied.answer} "
                  f"({ledger.is_request_satisfied.reason})")
            print(f"Next: {ledger.next_speaker.answer} — "
                  f"{ledger.instruction_or_question.answer}")
```

---

## 5 · `GroupChatOrchestrator`

**Module:** `agent_framework.orchestrations`

`GroupChatOrchestrator` is the lightweight, custom-selection-function alternative to `MagenticOrchestrator`. Instead of an LLM manager, you supply a plain Python function that decides which participant speaks next.

```python
class GroupChatOrchestrator(BaseGroupChatOrchestrator):
    def __init__(
        self,
        id: str,
        participant_registry: ParticipantRegistry,
        selection_func: GroupChatSelectionFunction,
        *,
        name: str | None = None,
        max_rounds: int | None = None,
        termination_condition: TerminationCondition | None = None,
    ) -> None: ...
```

| Parameter | Description |
|---|---|
| `id` | Unique executor ID, must be unique within the workflow. |
| `selection_func` | `async def f(state: GroupChatState) -> str` — returns the name of the next participant. |
| `max_rounds` | Stop after this many full participant turns. `None` = no limit. |
| `termination_condition` | Callable `async def f(state: GroupChatState) -> bool`; when it returns `True` the chat ends. |

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import (
    GroupChatBuilder,
    GroupChatOrchestrator,
    GroupChatState,
)

async def round_robin_selector(state: GroupChatState) -> str:
    """Rotate through participants in order."""
    participants = list(state.participant_names)
    idx = state.round_count % len(participants)
    return participants[idx]

async def stop_after_keyword(state: GroupChatState) -> bool:
    """Stop when any participant says DONE."""
    if state.messages:
        return "DONE" in (state.messages[-1].text or "")
    return False

async def main() -> None:
    client = OpenAIChatClient("gpt-4o-mini")

    writer = Agent(client=client, name="writer",
                   instructions="Write a haiku about the given topic.")
    critic = Agent(client=client, name="critic",
                   instructions="Critique the haiku. If it's perfect, say DONE.")

    workflow = (
        GroupChatBuilder(
            selection_func=round_robin_selector,
            max_rounds=10,
            termination_condition=stop_after_keyword,
        )
        .add_participant(writer)
        .add_participant(critic)
        .build()
    )

    result = await workflow.run("Write a haiku about autumn leaves.")
    print(result.text)

asyncio.run(main())
```

### Last-message selector (common pattern)

```python
async def last_message_mentions_selector(state: GroupChatState) -> str:
    """Select whoever was mentioned by name in the last message."""
    last = state.messages[-1].text if state.messages else ""
    for name in state.participant_names:
        if name.lower() in last.lower():
            return name
    # Fallback: first participant
    return next(iter(state.participant_names))
```

---

## 6 · `ContentVariableStore` · `VariableReferenceContent`

**Module:** `agent_framework.security`

> **Experimental (FIDES subsystem).** Import from `agent_framework.security`. APIs may change without notice.

The FIDES security subsystem prevents prompt-injection attacks by keeping untrusted content out of the LLM context window. `ContentVariableStore` stores the raw content; `VariableReferenceContent` puts an opaque reference placeholder in the prompt instead.

### `ContentVariableStore`

```python
class ContentVariableStore:
    def store(self, content: Any, label: ContentLabel) -> str: ...
    def retrieve(self, var_id: str) -> tuple[Any, ContentLabel]: ...
    def exists(self, var_id: str) -> bool: ...
    def list_variables(self) -> list[str]: ...
    def clear(self) -> None: ...
```

`store()` generates a unique `var_XXXX` ID, saves `(content, label)`, and returns the ID. The LLM never sees the actual content — only the ID in a `VariableReferenceContent` message.

### `VariableReferenceContent`

```python
class VariableReferenceContent:
    variable_id: str
    label: ContentLabel
    description: str | None
    type: str = "variable_reference"   # discriminator for serialisation
```

### Preventing prompt injection from an external API

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import (
    ContentLabel,
    ContentVariableStore,
    ConfidentialityLabel,
    IntegrityLabel,
    VariableReferenceContent,
)
from agent_framework._types import Message

# Global store shared between the tool and the result handler
_store = ContentVariableStore()

@tool(description="Fetch news headlines from an external API")
async def fetch_news(topic: str) -> str:
    """Simulate fetching untrusted external content."""
    raw_content = f"BREAKING: {topic} news — <script>steal_cookies()</script>"

    # Store with UNTRUSTED label — LLM sees only the var ID
    label = ContentLabel(
        integrity=IntegrityLabel.UNTRUSTED,
        confidentiality=ConfidentialityLabel.PUBLIC,
    )
    var_id = _store.store(raw_content, label)

    # Return a variable reference, not the raw content
    ref = VariableReferenceContent(
        variable_id=var_id,
        label=label,
        description=f"News headlines about '{topic}'",
    )
    return f"[variable:{var_id}] {ref.description}"

async def process_result(var_id: str) -> str:
    """Retrieve and sanitise the content after the LLM decides what to do."""
    content, label = _store.retrieve(var_id)
    print(f"Label: {label.integrity} / {label.confidentiality}")
    # Sanitise here before returning to the user
    return content.replace("<script>", "").replace("</script>", "")

async def main() -> None:
    client = OpenAIChatClient("gpt-4o-mini")
    agent = Agent(
        client=client,
        tools=[fetch_news],
        instructions="Summarise the fetched news. Refer to content by its variable reference.",
    )
    response = await agent.run("Get me news about climate change.")
    print(response.text)

    # After the run, retrieve any stored variables for safe processing
    for var_id in _store.list_variables():
        safe = await process_result(var_id)
        print(f"{var_id}: {safe}")

asyncio.run(main())
```

---

## 7 · `LabelTrackingFunctionMiddleware`

**Module:** `agent_framework.security`

> **Experimental (FIDES subsystem).**

`LabelTrackingFunctionMiddleware` is a `FunctionMiddleware` that automatically propagates security integrity and confidentiality labels through every tool call using a strict three-tier priority system.

```python
class LabelTrackingFunctionMiddleware(FunctionMiddleware):
    def __init__(
        self,
        default_integrity: IntegrityLabel = IntegrityLabel.UNTRUSTED,
        default_confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC,
        auto_hide_untrusted: bool = True,
        hide_threshold: IntegrityLabel = IntegrityLabel.UNTRUSTED,
    ) -> None: ...
```

### Three-tier label propagation

| Priority | Source | When used |
|---|---|---|
| **Tier 1** | Per-item embedded `additional_properties.security_label` in the result | Always wins when present |
| **Tier 2** | Tool's `source_integrity` declared in `@tool(additional_properties=...)` | No embedded label in result |
| **Tier 3** | `combine_labels()` join of all input argument labels | No embedded label AND no `source_integrity` |

### Declaring tool integrity with `@tool`

```python
from agent_framework import tool

@tool(additional_properties={"source_integrity": "trusted"})
async def compute_sum(a: int, b: int) -> int:
    """Pure computation — result is always trusted."""
    return a + b

@tool(additional_properties={"source_integrity": "untrusted"})
async def web_search(query: str) -> str:
    """External data — always untrusted."""
    return f"Search results for '{query}': ..."

@tool  # No source_integrity — tier 3: label inherits from inputs
async def format_data(raw: str) -> str:
    return raw.strip().upper()
```

### Full wiring example

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import (
    LabelTrackingFunctionMiddleware,
    ConfidentialityLabel,
    IntegrityLabel,
)

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")

    tracker = LabelTrackingFunctionMiddleware(
        default_integrity=IntegrityLabel.UNTRUSTED,   # tools default to untrusted
        default_confidentiality=ConfidentialityLabel.PUBLIC,
        auto_hide_untrusted=True,     # untrusted results stored in ContentVariableStore
        hide_threshold=IntegrityLabel.UNTRUSTED,
    )

    agent = Agent(
        client=client,
        tools=[compute_sum, web_search, format_data],
        middleware=[tracker],
    )

    response = await agent.run("Search for Python news and compute 40 + 2.")
    print(response.text)

    # Inspect the accumulated context label after the run
    context_label = tracker.get_context_label()
    print(f"Final context integrity: {context_label.integrity}")
    print(f"Stored variables: {tracker.get_variable_store().list_variables()}")

asyncio.run(main())
```

---

## 8 · `PolicyEnforcementFunctionMiddleware`

**Module:** `agent_framework.security`

> **Experimental (FIDES subsystem).**

`PolicyEnforcementFunctionMiddleware` is a companion `FunctionMiddleware` that checks security labels *before* tool execution, blocking or flagging calls that violate declared policies. It is typically stacked after `LabelTrackingFunctionMiddleware`.

```python
class PolicyEnforcementFunctionMiddleware(FunctionMiddleware):
    def __init__(
        self,
        allow_untrusted_tools: set[str] | None = None,
        block_on_violation: bool = True,
        enable_audit_log: bool = True,
        approval_on_violation: bool = False,
    ) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `allow_untrusted_tools` | `None` | Set of tool names explicitly permitted to run even when the execution context is untrusted (e.g. because a previous tool returned untrusted data). `None` = empty whitelist. |
| `block_on_violation` | `True` | When `True`, a policy violation raises `MiddlewareTermination` and the tool is not called. When `False`, violations are logged but execution proceeds. |
| `enable_audit_log` | `True` | Append each violation to `self.audit_log` for post-run inspection. |
| `approval_on_violation` | `False` | When `True`, violations emit a HITL `ToolApprovalState` request instead of blocking. Requires `ToolApprovalMiddleware` to be present. |

### Stacking the two security middlewares

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.security import (
    LabelTrackingFunctionMiddleware,
    PolicyEnforcementFunctionMiddleware,
    IntegrityLabel,
    ConfidentialityLabel,
)

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")

    tracker = LabelTrackingFunctionMiddleware(auto_hide_untrusted=True)
    policy = PolicyEnforcementFunctionMiddleware(
        allow_untrusted_tools={"web_search"},   # web_search is allowed to run even in untrusted context
        block_on_violation=True,
        enable_audit_log=True,
    )

    agent = Agent(
        client=client,
        tools=[compute_sum, web_search, format_data],
        middleware=[tracker, policy],    # order matters: tracker runs first
    )

    response = await agent.run("Search for news then compute 10 + 5.")
    print(response.text)

    # Inspect audit log after the run
    for entry in policy.audit_log:
        print(f"Policy violation: {entry}")

asyncio.run(main())
```

### Checking violations without blocking

```python
# Audit-only mode: record violations but never block tool execution
audit_policy = PolicyEnforcementFunctionMiddleware(
    allow_untrusted_tools=set(),
    block_on_violation=False,    # log only
    enable_audit_log=True,
)
# After the run:
if audit_policy.audit_log:
    print(f"{len(audit_policy.audit_log)} policy violations detected:")
    for v in audit_policy.audit_log:
        print(f"  - {v}")
```

---

## 9 · `FileMemoryProvider` · `FileStoreEntry`

**Module:** `agent_framework._harness._file_memory`

> **Experimental (HARNESS subsystem).** Import from `agent_framework`.

`FileMemoryProvider` is a `ContextProvider` that gives each agent session a private, file-based scratchpad. The agent accesses it through seven auto-injected tools without any additional code in the agent loop.

```python
class FileMemoryProvider(ContextProvider):
    def __init__(
        self,
        store: AgentFileStore,
        *,
        source_id: str = "file_memory",
        scope: str | None = None,
        instructions: str | None = None,
    ) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `store` | — | Storage backend. `InMemoryAgentFileStore` for testing; `AgentFileStore` for a real filesystem. |
| `source_id` | `"file_memory"` | Unique ID for the provider within the agent harness. |
| `scope` | `None` | Namespace for memory isolation. `None` = use session ID (per-session). Pass a user ID to share memory across sessions for the same user. |
| `instructions` | `None` | Custom instructions appended to the agent's system prompt. `None` = use the built-in file-memory instructions. |

### Injected tools

| Tool | Description |
|---|---|
| `file_memory_write` | Write/overwrite a named file with a description |
| `file_memory_read` | Read a file's content by name |
| `file_memory_delete` | Delete a file |
| `file_memory_ls` | List all memory files and their descriptions |
| `file_memory_grep` | Search file contents with a regex |
| `file_memory_replace` | Replace a substring within a file |
| `file_memory_replace_lines` | Replace specific lines within a file |

### `FileStoreEntry`

`FileStoreEntry` is the data model returned by `file_memory_ls`. It has two class-level constants (`FILE = "file"`, `DIRECTORY = "directory"`) and two fields.

```python
class FileStoreEntry(SerializationMixin):
    FILE: ClassVar[str] = "file"
    DIRECTORY: ClassVar[str] = "directory"
    name: str    # entry name (not a full path)
    type: str    # FileStoreEntry.FILE or FileStoreEntry.DIRECTORY
```

### Wiring up file memory

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._file_access import InMemoryAgentFileStore
from agent_framework._harness._file_memory import FileMemoryProvider

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    store = InMemoryAgentFileStore()
    memory = FileMemoryProvider(store=store, scope="user-42")

    agent = Agent(
        client=client,
        instructions=(
            "You are a personal assistant. Use your file memory to persist "
            "important facts the user tells you. Check memory at the start of "
            "each conversation."
        ),
        context_providers=[memory],
    )

    session = agent.create_session()
    await agent.run("My birthday is March 15th. Remember this.", session=session)

    # New session for the same user — scope="user-42" shares memory
    session2 = agent.create_session()
    result = await agent.run("When is my birthday?", session=session2)
    print(result.text)   # → "Your birthday is March 15th."

asyncio.run(main())
```

### Persistent memory with a real file store

```python
from agent_framework._harness._file_access import FileSystemAgentFileStore
from agent_framework._harness._file_memory import FileMemoryProvider

# Persist to disk under /tmp/agent-memory/ (created lazily on first write)
file_store = FileSystemAgentFileStore(root_directory="/tmp/agent-memory")
memory = FileMemoryProvider(store=file_store, scope="user-42")
# Attach to agent as above — memories survive process restarts
```

---

## 10 · `AgentModeProvider`

**Module:** `agent_framework._harness._mode`

> **Experimental (HARNESS subsystem).** Import from `agent_framework`.

`AgentModeProvider` lets an agent switch between named operating modes (e.g. `plan` vs `execute`) that are persisted in session state and exposed to the LLM through its instructions on every invocation.

```python
class AgentModeProvider(ContextProvider):
    def __init__(
        self,
        source_id: str = "agent_mode",
        *,
        default_mode: str | None = None,
        mode_descriptions: Mapping[str, str] | None = None,
        instructions: str | None = None,
    ) -> None: ...
```

| Parameter | Default | Description |
|---|---|---|
| `source_id` | `"agent_mode"` | Unique ID for the provider. |
| `default_mode` | `None` | Initial mode. When `None`, the first entry of `mode_descriptions` is used. |
| `mode_descriptions` | `None` | `{mode_name: description}` mapping. `None` = built-in `{"plan": "...", "execute": "..."}`. |
| `instructions` | `None` | Custom instruction template. Supports `{available_modes}` and `{current_mode}` placeholders. |

### Injected tools

| Tool | Description |
|---|---|
| `mode_set` | Switch the agent's current mode (persisted in `AgentSession` state) |
| `mode_get` | Retrieve the current mode name |

### External mode access helpers

```python
from agent_framework._harness._mode import get_agent_mode, set_agent_mode

# Read the current mode for a session (returns the mode string or the default if not set)
current = get_agent_mode(session)

# Programmatically switch mode from outside the agent loop — both helpers are synchronous
set_agent_mode(session, "execute")
```

### Task-completion pattern: plan → execute

```python
import asyncio
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._mode import AgentModeProvider

MODES = {
    "plan": (
        "You are in PLANNING mode. Ask clarifying questions, outline steps, "
        "and switch to 'execute' mode only when you have a complete plan."
    ),
    "execute": (
        "You are in EXECUTION mode. Follow the plan step-by-step. "
        "Do not ask questions — take action."
    ),
}

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    mode_provider = AgentModeProvider(
        default_mode="plan",
        mode_descriptions=MODES,
    )

    agent = Agent(
        client=client,
        instructions=(
            "You are a project assistant. Use mode_set to switch between plan and execute modes."
        ),
        context_providers=[mode_provider],
    )

    session = agent.create_session()

    # Turn 1: agent is in plan mode, will ask questions
    r1 = await agent.run("I need to build a REST API for my todo app.", session=session)
    print("Plan mode response:", r1.text)

    # Turn 2: user confirms — agent should switch to execute mode
    r2 = await agent.run("Sounds good. Let's proceed with that plan.", session=session)
    print("Execute mode response:", r2.text)

asyncio.run(main())
```

### Custom modes for a research agent

```python
research_modes = {
    "search": "Find and collect raw sources. Prioritise breadth over depth.",
    "analyse": "Critically evaluate the collected sources for relevance and accuracy.",
    "write": "Synthesise findings into a clear, well-structured written report.",
}

mode_provider = AgentModeProvider(
    default_mode="search",
    mode_descriptions=research_modes,
    instructions=(
        "You are a research assistant. Current mode: {current_mode}.\n"
        "Available modes: {available_modes}\n"
        "Switch modes as appropriate using the mode_set tool."
    ),
)
```

---

## 11 · `GAIA` · `GAIATelemetryConfig` · `Task` · `TaskResult` · `Evaluator`

**Module:** `agent_framework.lab.gaia`

The GAIA benchmark integration provides a complete harness for running agents against the [GAIA benchmark dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) — the standard for evaluating general-purpose AI assistants across real-world multi-step tasks.

### `Task` and `TaskResult`

```python
@dataclass
class Task:
    task_id: str
    question: str
    answer: str | None = None     # ground-truth, absent in test split
    level: int | None = None      # GAIA difficulty level (1, 2, or 3)
    file_name: str | None = None  # attached file path if the task has one
    metadata: dict | None = None

@dataclass
class TaskResult:
    task_id: str
    task: Task
    prediction: Prediction        # agent's answer + raw response
    evaluation: Evaluation        # is_correct + score (0.0–1.0)
    runtime_seconds: float | None = None
    error: str | None = None      # populated if the agent threw an exception
```

### `Evaluator` protocol

```python
@runtime_checkable
class Evaluator(Protocol):
    async def __call__(self, task: Task, prediction: Prediction) -> Evaluation: ...
```

Implement this protocol to swap in a custom scoring function. The default evaluator uses the official GAIA scorer (exact-match normalisation).

### `GAIATelemetryConfig`

```python
class GAIATelemetryConfig:
    def __init__(
        self,
        enable_tracing: bool = False,
        otlp_endpoint: str | None = None,
        trace_to_file: bool = False,
        file_path: str | None = None,
    ) -> None: ...
```

### `GAIA`

```python
class GAIA:
    def __init__(
        self,
        evaluator: Evaluator | None = None,
        data_dir: str | None = None,
        hf_token: str | None = None,
        telemetry_config: GAIATelemetryConfig | None = None,
    ) -> None: ...

    async def run(
        self,
        task_runner: TaskRunner,
        level: int | list[int] = 1,
        max_n: int | None = None,
        parallel: int = 1,
        timeout: int | None = None,
        out: str | None = None,
    ) -> list[TaskResult]: ...
```

`TaskRunner` is a protocol — any async callable `(task: Task) -> Prediction` satisfies it. The `Agent` class does **not** satisfy it directly; wrap your agent in a thin adapter that calls the agent and packages the response as a `Prediction`.

### Running a GAIA evaluation

```python
import asyncio
import os
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework.lab.gaia import GAIA, GAIATelemetryConfig, Task, Prediction

async def main() -> None:
    client = OpenAIChatClient("gpt-4o")
    agent = Agent(
        client=client,
        instructions=(
            "You are a highly capable general assistant. "
            "Think carefully, use all available tools, and give precise final answers."
        ),
    )

    # TaskRunner adapter — wraps Agent.run into the (Task) -> Prediction protocol
    async def run_task(task: Task) -> Prediction:
        session = agent.create_session()
        result = await agent.run(task.question, session=session)
        return Prediction(prediction=result.text)

    telemetry = GAIATelemetryConfig(
        enable_tracing=True,
        trace_to_file=True,
        file_path="gaia_traces.json",
    )

    gaia = GAIA(
        hf_token=os.environ["HF_TOKEN"],
        telemetry_config=telemetry,
    )

    # Run level-1 tasks (up to 20), 4 in parallel
    results = await gaia.run(
        run_task,
        level=1,
        max_n=20,
        parallel=4,
    )

    correct = sum(1 for r in results if r.evaluation.is_correct)
    print(f"Score: {correct}/{len(results)} ({100 * correct / len(results):.1f}%)")

    for r in results:
        status = "✓" if r.evaluation.is_correct else "✗"
        print(f"  {status} [{r.task.level}] {r.task.task_id}: {r.prediction.prediction!r}")
        if r.error:
            print(f"       Error: {r.error}")

asyncio.run(main())
```

### Custom evaluator

```python
import asyncio
from agent_framework.lab.gaia import GAIA, Task, Evaluation, Prediction, Evaluator

class FuzzyMatchEvaluator:
    """Accept an answer if it contains the ground truth (case-insensitive)."""

    async def __call__(self, task: Task, prediction: Prediction) -> Evaluation:
        if task.answer is None:
            return Evaluation(is_correct=False, score=0.0)
        match = task.answer.lower() in (prediction.prediction or "").lower()
        return Evaluation(is_correct=match, score=1.0 if match else 0.0)

gaia = GAIA(evaluator=FuzzyMatchEvaluator())
```

### Running a single task

```python
from agent_framework.lab.gaia import GAIA, Task

gaia = GAIA()
task = Task(
    task_id="custom-001",
    question="What is the capital city of France?",
    answer="Paris",
    level=1,
)
result = await gaia.run_task(agent, task)
print(f"Correct: {result.evaluation.is_correct}, "
      f"Answer: {result.prediction.prediction!r}, "
      f"Runtime: {result.runtime_seconds:.1f}s")
```

---

## Summary

| Class | Key insight |
|---|---|
| `MagenticOrchestrator` | Two-level loop: outer (replan) wraps inner (coordinate). Stall detection via `stall_count > max_stall_count`. |
| `MagenticContext` | Mutable orchestrator state. Always cloned before manager calls to prevent mutation. |
| `MagenticResetSignal` | Sentinel class — raise to force an outer-loop reset from a custom subclass. |
| `MagenticPlanReviewRequest` | HITL gate: `approve()` or `revise(feedback)` resumes/replans. |
| `MagenticPlanReviewResponse` | Empty `review=[]` = approved; non-empty = feedback injected into chat and plan redone. |
| `StandardMagenticManager` | Wraps an `Agent` for all management LLM calls; 8 prompt overrides, 3 budget knobs. |
| `MagenticProgressLedger` | 5-field structured output from manager on each inner tick. |
| `GroupChatOrchestrator` | Custom-selection-function group chat; no LLM manager needed. |
| `ContentVariableStore` | Keeps untrusted content off the LLM context; returns opaque var IDs. |
| `VariableReferenceContent` | Placeholder that travels through the LLM prompt instead of raw content. |
| `LabelTrackingFunctionMiddleware` | 3-tier label propagation; `auto_hide_untrusted` stores results in `ContentVariableStore`. |
| `PolicyEnforcementFunctionMiddleware` | Pre-execution policy check; `allow_untrusted_tools` whitelist, audit log, optional blocking. |
| `FileMemoryProvider` | 7-tool file scratchpad per session; `scope` for cross-session sharing. |
| `FileStoreEntry` | Typed directory-listing entry (`FILE` / `DIRECTORY`). |
| `AgentModeProvider` | Plan/execute (or any custom) mode switching persisted in session state. |
| `GAIA` | End-to-end GAIA benchmark runner; `run_benchmark()` or `run_task()`. |
| `GAIATelemetryConfig` | OTLP or file-based tracing for benchmark runs. |
| `Task` / `TaskResult` / `Evaluator` | Data model + protocol for custom GAIA scoring. |

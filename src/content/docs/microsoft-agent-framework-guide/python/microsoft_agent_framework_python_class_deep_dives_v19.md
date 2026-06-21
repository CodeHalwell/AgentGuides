---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 19"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.9.0: ConcurrentBuilder (fan-out/fan-in + custom aggregators + output routing), SequentialBuilder (shared-conversation chains + chain_only_agent_responses), HandoffBuilder + HandoffConfiguration + HandoffSentEvent (decentralised routing + autonomous mode + 1.9.0 output_from), HandoffAgentUserRequest (HITL factories for handoff workflows), OrchestrationState (unified checkpoint dataclass for GroupChat/Handoff/Magentic), AgentModeProvider + get_agent_mode + set_agent_mode (plan/execute mode harness — mode_descriptions Mapping, template placeholders, external-change notification), TodoItem + TodoInput + TodoCompleteInput (todo DTO primitives — SerializationMixin round-trip, validation), TodoStore + TodoSessionStore + TodoFileStore (abstract backing store + session state + file-backed persistence with path-traversal guard), TodoProvider (5-tool harness — per-session asyncio.Lock via WeakKeyDictionary, custom store injection), MagenticResetSignal + StandardMagenticManager (stall recovery signal + fully-customisable LLM manager)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 42
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 19

Verified against **agent-framework 1.9.0** (installed June 2026). Every constructor
signature, parameter description, and code example was derived from the installed package
source. Sub-packages introspected: `agent_framework_orchestrations._concurrent`,
`agent_framework_orchestrations._sequential`, `agent_framework_orchestrations._handoff`,
`agent_framework_orchestrations._orchestration_state`, `agent_framework._harness._mode`,
`agent_framework._harness._todo`, `agent_framework_orchestrations._magentic`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, middleware ABCs, compaction, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — harness providers, compaction strategies, `WorkflowViz`, MCP transports
- [Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — message/chat types, `ResponseStream`, `AgentContext`, functional workflows, `SkillsSource`, eval model, tokenizer, `ConversationSplit`
- [Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5/) — `Executor`, `AgentExecutor`, edge groups, `Runner`, `SessionContext`, `AgentSession`, `BaseChatClient`, `SecretString`, `WorkflowCheckpoint`, exceptions
- [Vol. 6](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6/) — feature staging, `WorkflowRunState`, `WorkflowExecutor`, `AgentResponse`, embedding clients, `FunctionInvocationConfiguration`, `ClassSkill`, `Annotation`, capability protocols, middleware layers
- [Vol. 7](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7/) — `ContextProvider`, `BackgroundTaskInfo`, orchestration builders, `AgentFactory`, `SecureAgentConfig`, `ObservabilitySettings`
- [Vol. 8](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8/) — file store hierarchy, `FileAccessProvider`, `MCPSkill`, `ToolMode`, eval helpers, `ChatContext`, `WorkflowAgent`, compaction, history providers, skills composition
- [Vol. 9](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9/) — `OllamaChatClient`, `PurviewPolicyMiddleware`, `DurableAIAgent`, `GitHubCopilotAgent`, `HyperlightExecuteCodeTool`, `Mem0ContextProvider`, Redis providers, Magentic internals, `FileSkillsSource`
- [Vol. 10](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10/) — `Workflow`, `InProcRunnerContext`, `FunctionExecutor`, `FunctionInvocationLayer`, memory harness, todo harness, `DeduplicatingSkillsSource`, `SkillsProvider`, `MCPTaskOptions`, `InMemoryCheckpointStorage`, `BaseAgent`
- [Vol. 11](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v11/) — telemetry layers, `Edge`+`EdgeGroup` primitives, `Case`+`Default`, `EdgeRunner` hierarchy, `ExecutionContext`, `WorkflowGraphValidator`, `MCPTool`, serialization mixin, `Evaluator`, `PerServiceCallHistoryPersistingMiddleware`
- [Vol. 12](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v12/) — Skills ABCs, `FileSkill`, `InlineSkillResource`+`InlineSkillScript`, `FileSkillScript`+`SkillScriptRunner`, `SupportsAgentRun`, `RunnerContext`, edge-routing descriptors, `WorkflowValidationError` hierarchy, `A2AAgent`+`A2AExecutor`, exception leaf classes
- [Vol. 13](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v13/) — OpenAI Responses/Completions/Embedding clients, Anthropic + Claude agent clients, multi-cloud Claude variants, group-chat + handoff + Magentic orchestration internals, declarative HTTP/MCP/approval handlers
- [Vol. 14](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v14/) — `State` (superstep cache), `OutputDesignation`, `MessageType`+`WorkflowMessage` internals, `DictConvertible` mixin, middleware pipeline hierarchy, `MiddlewareDict`, `FunctionRequestResult`, `OtelAttr`, security policy classes
- [Vol. 15](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v15/) — AG-UI client layer, AG-UI protocol wrappers, ChatKit, DevServer, GAIA benchmark, CopilotStudioAgent, AzureAISearchContextProvider, CosmosHistoryProvider, Durable external layer, AgentFunctionApp
- [Vol. 16](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v16/) — FoundryAgent+FoundryAgentOptions, FoundryLocalClient, FoundryMemoryProvider, FoundryEvals, BedrockChatClient, BedrockEmbeddingClient, MagenticManagerBase, BaseGroupChatOrchestrator, AgentRequestInfoResponse+CacheProvider, Purview exception hierarchy
- [Vol. 17](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v17/) — ToolApprovalMiddleware+ToolApprovalRule+ToolApprovalState, AgentLoopMiddleware+JudgeVerdict, SamplingApprovalCallback+MCP sampling security, to_prompt_agent, FoundryEmbeddingClient, ContentUnderstandingContextProvider, FileSearchConfig, AgentFrameworkTracer, TaskRunner (Tau2), FoundryChatClient hosted tool factories
- [Vol. 18](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v18/) — Skill+SkillFrontmatter+SkillScriptRunner, InlineSkill, skills source pipeline, AgentFileStore+InMemoryAgentFileStore, FileAccessProvider, BackgroundAgentsProvider, MemoryStore, WorkflowGraphValidator, MagenticBuilder+MagenticManagerBase+MagenticProgressLedger, LocalEvaluator+EvalItem+ConversationSplit

This volume covers **ten class groups** focussed on the high-level orchestration builders at their 1.9.0 API (including newly-added output-routing and HITL parameters), the brand-new `OrchestrationState` checkpoint dataclass, and a full deep dive into the todo and mode harness providers including their 1.9.0 file-backed storage backends:

| # | Class / group | Sub-package |
|---|---|---|
| 1 | `ConcurrentBuilder` | `agent_framework_orchestrations._concurrent` |
| 2 | `SequentialBuilder` | `agent_framework_orchestrations._sequential` |
| 3 | `HandoffBuilder` · `HandoffConfiguration` · `HandoffSentEvent` | `agent_framework_orchestrations._handoff` |
| 4 | `HandoffAgentUserRequest` | `agent_framework_orchestrations._handoff` |
| 5 | `OrchestrationState` | `agent_framework_orchestrations._orchestration_state` |
| 6 | `AgentModeProvider` · `get_agent_mode` · `set_agent_mode` | `agent_framework._harness._mode` |
| 7 | `TodoItem` · `TodoInput` · `TodoCompleteInput` | `agent_framework._harness._todo` |
| 8 | `TodoStore` · `TodoSessionStore` · `TodoFileStore` | `agent_framework._harness._todo` |
| 9 | `TodoProvider` | `agent_framework._harness._todo` |
| 10 | `MagenticResetSignal` · `StandardMagenticManager` | `agent_framework_orchestrations._magentic` |

---

## 1 · `ConcurrentBuilder`

**Sub-package:** `agent_framework_orchestrations._concurrent`  
**Install:** `pip install agent-framework[orchestrations]`

`ConcurrentBuilder` assembles a **fan-out / fan-in** workflow where every participant
receives the same input simultaneously and results are merged by an aggregator.  The
default aggregator yields one `AgentResponse` whose `messages` list contains one
assistant turn per participant.

### Class signature (1.9.0)

```python
from collections.abc import Callable, Sequence
from typing import Any, Literal
from agent_framework import SupportsAgentRun
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._checkpoint import CheckpointStorage
from agent_framework._workflows._agent_executor import AgentExecutorResponse
from agent_framework._workflows._workflow import Workflow
from agent_framework_orchestrations import ConcurrentBuilder

class ConcurrentBuilder:
    def __init__(
        self,
        *,
        participants: Sequence[SupportsAgentRun | Executor],
        checkpoint_storage: CheckpointStorage | None = None,
        output_from: Sequence[str | SupportsAgentRun] | Literal["all"] | None = ...,
        intermediate_output_from: Sequence[str | SupportsAgentRun] | Literal["all_other"] | None = None,
    ) -> None: ...

    def with_aggregator(
        self,
        aggregator: Executor | Callable[[list[AgentExecutorResponse]], Any],
    ) -> "ConcurrentBuilder": ...

    def with_request_info(
        self,
        agents: list[str | SupportsAgentRun] | None = None,
    ) -> "ConcurrentBuilder": ...

    def build(self) -> Workflow: ...
```

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `participants` | `Sequence[SupportsAgentRun \| Executor]` | required | Agents/executors to run in parallel. No duplicates allowed. |
| `checkpoint_storage` | `CheckpointStorage \| None` | `None` | Optional persistence backend for resuming interrupted runs. |
| `output_from` | `Sequence[...] \| "all" \| None` | framework default | Which participants' `yield_output` calls surface as workflow `output` events. |
| `intermediate_output_from` | `Sequence[...] \| "all_other" \| None` | `None` | Which participants surface as `intermediate` events (e.g., for streaming preview). |

### `with_aggregator(aggregator)`

Override the default fan-in aggregator.  The aggregator can be:
- An **`Executor`** subclass — receives `list[AgentExecutorResponse]` via a `@handler`.
- A **sync or async callback** — `(results: list[AgentExecutorResponse]) -> Any`.  A non-`None`
  return becomes the workflow output.

### `with_request_info(agents=None)`

Pause after the aggregator and request human feedback before completing.  Pass
`agents=[...]` to pause only before specific participants.

### Code examples

**Example 1 — Minimal parallel research**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import ConcurrentBuilder

client = OpenAIChatClient("gpt-4o")
web_agent   = client.as_agent(name="web",   instructions="Search the web for recent facts.")
code_agent  = client.as_agent(name="code",  instructions="Write Python code examples.")
docs_agent  = client.as_agent(name="docs",  instructions="Summarise official documentation.")

workflow = ConcurrentBuilder(
    participants=[web_agent, code_agent, docs_agent],
).build()

async def main() -> None:
    result = await workflow.run("Explain asyncio task cancellation in Python 3.12")
    # get_outputs()[0] is the aggregated AgentResponse — one message per participant
    for msg in result.get_outputs()[0].messages:
        print(msg.contents[0])

asyncio.run(main())
```

**Example 2 — Custom aggregator callback**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._agent_executor import AgentExecutorResponse
from agent_framework_orchestrations import ConcurrentBuilder

client = OpenAIChatClient("gpt-4o-mini")
analyst   = client.as_agent(name="analyst",   instructions="Analyse the market trend.")
strategist = client.as_agent(name="strategist", instructions="Suggest a go-to-market strategy.")

def merge_results(results: list[AgentExecutorResponse]) -> str:
    """Combine each agent's last assistant turn into a single report."""
    parts = []
    for r in results:
        agent_name = r.agent_response.agent_id or "agent"
        last_turn  = r.agent_response.messages[-1].contents[0]
        parts.append(f"### {agent_name}\n{last_turn}")
    return "\n\n".join(parts)

workflow = (
    ConcurrentBuilder(participants=[analyst, strategist])
    .with_aggregator(merge_results)
    .build()
)

async def main() -> None:
    result = await workflow.run("Electric vehicle market in 2025")
    print(result)   # the merged markdown report

asyncio.run(main())
```

**Example 3 — Per-participant intermediate streaming + checkpoint**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._checkpoint import FileCheckpointStorage
from agent_framework_orchestrations import ConcurrentBuilder

storage = FileCheckpointStorage("./checkpoints")
client  = OpenAIChatClient("gpt-4o")
a1 = client.as_agent(name="researcher", instructions="Research the topic.")
a2 = client.as_agent(name="writer",    instructions="Draft prose from given facts.")
a3 = client.as_agent(name="critic",    instructions="Critique the draft and list improvements.")

workflow = ConcurrentBuilder(
    participants=[a1, a2, a3],
    checkpoint_storage=storage,
    # Surface a1 and a2 as primary output; a3 as intermediate
    output_from=[a1, a2],
    intermediate_output_from=[a3],
).build()

async def main() -> None:
    async for event in workflow.stream("Future of quantum computing"):
        if event.type == "output":
            print(f"[OUTPUT from {event.source}]", event.data)
        elif event.type == "intermediate":
            print(f"[INTERMEDIATE from {event.source}]", event.data)

asyncio.run(main())
```

---

## 2 · `SequentialBuilder`

**Sub-package:** `agent_framework_orchestrations._sequential`

`SequentialBuilder` wires agents into a **shared-conversation chain**: each participant
receives the full conversation so far (all prior turns) and appends its response before
passing the updated list on to the next participant.  Custom `Executor` participants can
transform the conversation (summarise, translate, redact) and return a revised
`list[Message]`.

### Class signature (1.9.0)

```python
from agent_framework_orchestrations import SequentialBuilder

class SequentialBuilder:
    def __init__(
        self,
        *,
        participants: Sequence[SupportsAgentRun | Executor],
        checkpoint_storage: CheckpointStorage | None = None,
        chain_only_agent_responses: bool = False,
        output_from: Sequence[str | SupportsAgentRun] | Literal["all"] | None = ...,
        intermediate_output_from: Sequence[str | SupportsAgentRun] | Literal["all_other"] | None = None,
    ) -> None: ...

    def with_request_info(
        self,
        agents: list[str | SupportsAgentRun] | None = None,
    ) -> "SequentialBuilder": ...

    def build(self) -> Workflow: ...
```

### Key parameter: `chain_only_agent_responses`

When `False` (default) the **full conversation** (all `Message` objects, including tool
calls) is passed to each participant.  When `True`, only the assistant `AgentResponse`
messages are forwarded — useful when intermediate participants produce verbose tool traces
that the final participant should not see.

### Code examples

**Example 1 — Research → Draft → Review pipeline**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import SequentialBuilder

client   = OpenAIChatClient("gpt-4o")
research = client.as_agent(name="researcher", instructions="Research the question thoroughly.")
draft    = client.as_agent(name="drafter",    instructions="Write a structured report from the research.")
review   = client.as_agent(name="reviewer",   instructions="Review the draft, fix errors and tighten the prose.")

workflow = SequentialBuilder(participants=[research, draft, review]).build()

async def main() -> None:
    result = await workflow.run("What caused the 2008 financial crisis?")
    # result is the reviewer's AgentResponse — the final refined report
    print(result.get_outputs()[-1].messages[-1].contents[0])

asyncio.run(main())
```

**Example 2 — Chain only assistant responses (hide tool noise)**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import SequentialBuilder

client  = OpenAIChatClient("gpt-4o")
coder   = client.as_agent(name="coder",   instructions="Write Python code. Use code-execution tools.")
explainer = client.as_agent(name="explainer", instructions="Explain what the provided code does in plain English.")

# explainer sees only the assistant text from coder, not tool call messages
workflow = SequentialBuilder(
    participants=[coder, explainer],
    chain_only_agent_responses=True,
).build()

async def main() -> None:
    result = await workflow.run("Write a function to compute Fibonacci numbers iteratively.")
    print(result.get_outputs()[-1].messages[-1].contents[0])

asyncio.run(main())
```

**Example 3 — With HITL checkpoint and per-step streaming**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._checkpoint import FileCheckpointStorage
from agent_framework_orchestrations import SequentialBuilder

storage  = FileCheckpointStorage("./seq_checkpoints")
client   = OpenAIChatClient("gpt-4o")
a1 = client.as_agent(name="outline",    instructions="Create a document outline.")
a2 = client.as_agent(name="writer",     instructions="Write the document from the outline.")
a3 = client.as_agent(name="translator", instructions="Translate the document to French.")

workflow = (
    SequentialBuilder(
        participants=[a1, a2, a3],
        checkpoint_storage=storage,
        # Show a2's output as intermediate; a3 is the primary output
        output_from=[a3],
        intermediate_output_from=[a2],
    )
    .with_request_info(agents=[a2])  # pause for human feedback before a2 runs
    .build()
)

async def main() -> None:
    async for event in workflow.stream("Write a product launch announcement"):
        print(event.type, getattr(event, "data", ""))

asyncio.run(main())
```

---

## 3 · `HandoffBuilder` · `HandoffConfiguration` · `HandoffSentEvent`

**Sub-package:** `agent_framework_orchestrations._handoff`

`HandoffBuilder` implements **decentralised routing**: agents signal handoffs by calling
a tool whose name encodes the target agent (`handoff_to_{target_id}`).  This contrasts
with `GroupChatBuilder` where a central orchestrator decides who speaks next.

`HandoffConfiguration` is the immutable routing descriptor stored per source agent.  
`HandoffSentEvent` is the observable event payload emitted on each handoff.

### Class signatures

```python
from agent_framework import Agent
from agent_framework_orchestrations import HandoffBuilder
from agent_framework_orchestrations._handoff import HandoffConfiguration, HandoffSentEvent
from dataclasses import dataclass

@dataclass
class HandoffSentEvent:
    source: str    # agent ID that handed off
    target: str    # agent ID that received the handoff

class HandoffConfiguration:
    target_id:   str
    description: str | None

    def __init__(
        self,
        *,
        target: str | SupportsAgentRun,
        description: str | None = None,
    ) -> None: ...

class HandoffBuilder:
    def __init__(
        self,
        *,
        name: str | None = None,
        participants: Sequence[Agent] | None = None,
        description: str | None = None,
        checkpoint_storage: CheckpointStorage | None = None,
        termination_condition: Callable[[list[Message]], bool | Awaitable[bool]] | None = None,
        output_from: Sequence[str | Agent] | Literal["all"] | None = ...,
        intermediate_output_from: Sequence[str | Agent] | Literal["all_other"] | None = None,
    ) -> None: ...

    # Fluent methods
    def participants(self, participants: Sequence[Agent]) -> "HandoffBuilder": ...
    def add_handoff(self, source: Agent, targets: Sequence[Agent], *, description: str | None = None) -> "HandoffBuilder": ...
    def with_start_agent(self, agent: Agent) -> "HandoffBuilder": ...
    def with_autonomous_mode(
        self,
        *,
        agents: Sequence[Agent | str] | None = None,
        prompts: dict[str, str] | None = None,
        turn_limits: dict[str, int] | None = None,
    ) -> "HandoffBuilder": ...
    def build(self) -> Workflow: ...
```

### Code examples

**Example 1 — Customer-service mesh topology (default: all-to-all)**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import HandoffBuilder

client   = OpenAIChatClient("gpt-4o")
triage   = client.as_agent(name="triage",   instructions="Route the customer to the right department.", description="General routing")
billing  = client.as_agent(name="billing",  instructions="Handle billing and payment queries.", description="Billing specialist")
support  = client.as_agent(name="support",  instructions="Handle technical support queries.", description="Tech support")

# No add_handoff() calls → all agents can hand off to all others (mesh)
workflow = (
    HandoffBuilder(participants=[triage, billing, support])
    .with_start_agent(triage)
    .build()
)

async def main() -> None:
    async for event in workflow.stream("I was double-charged last month"):
        print(event.type, getattr(event, "data", ""))

asyncio.run(main())
```

**Example 2 — Explicit routing graph with autonomous agents**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import HandoffBuilder

client      = OpenAIChatClient("gpt-4o-mini")
orchestrator = client.as_agent(name="orchestrator",  description="Initial triage")
researcher  = client.as_agent(name="researcher",    description="Deep research specialist")
writer      = client.as_agent(name="writer",        description="Report writer")

workflow = (
    HandoffBuilder(participants=[orchestrator, researcher, writer])
    # orchestrator can hand off to both researcher and writer
    .add_handoff(orchestrator, [researcher, writer])
    # researcher can only hand off to writer
    .add_handoff(researcher, [writer])
    .with_start_agent(orchestrator)
    .with_autonomous_mode(
        agents=[researcher, writer],         # only these two run autonomously
        turn_limits={"researcher": 5, "writer": 3},
    )
    .build()
)

async def main() -> None:
    result = await workflow.run("Research and write a report on battery technology trends.")
    print(result.get_outputs()[-1].messages[-1].contents[0])

asyncio.run(main())
```

**Example 3 — Observing `HandoffSentEvent` and checkpointing**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._checkpoint import FileCheckpointStorage
from agent_framework_orchestrations import HandoffBuilder

storage = FileCheckpointStorage("./handoff_cp")
client  = OpenAIChatClient("gpt-4o")
a = client.as_agent(name="agent_a", description="Specialist A")
b = client.as_agent(name="agent_b", description="Specialist B")

workflow = (
    HandoffBuilder(
        participants=[a, b],
        checkpoint_storage=storage,
    )
    .with_start_agent(a)
    .build()
)

async def main() -> None:
    async for event in workflow.stream("Begin task"):
        if event.type == "handoff_sent":
            # event.data is a HandoffSentEvent dataclass
            print(f"Handoff: {event.data.source} → {event.data.target}")
        elif event.type == "output":
            print("Output:", event.data)

asyncio.run(main())
```

---

## 4 · `HandoffAgentUserRequest`

**Sub-package:** `agent_framework_orchestrations._handoff`

When a handoff workflow runs in interactive (non-autonomous) mode the framework pauses
after each agent turn that does **not** trigger a handoff and issues a
`HandoffAgentUserRequest` to the caller.  The caller inspects the agent's response and
returns either follow-up `Message` objects or an empty list to terminate the workflow.

### Class signature

```python
from agent_framework import AgentResponse, Message
from dataclasses import dataclass

@dataclass
class HandoffAgentUserRequest:
    """Request issued to the user after an agent response in a non-autonomous handoff run."""
    agent_response: AgentResponse

    @staticmethod
    def create_response(response: str | list[str] | Message | list[Message]) -> list[Message]: ...
    @staticmethod
    def terminate() -> list[Message]: ...
```

### Factory methods

| Method | Returns | Purpose |
|---|---|---|
| `create_response(response)` | `list[Message]` | Convert str / `Message` / mixed list to a proper `list[Message]` for the next turn |
| `terminate()` | `list[Message]` (empty) | Signal that the workflow should stop cleanly |

### Code examples

**Example 1 — Interactive handoff with human feedback**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import HandoffBuilder
from agent_framework_orchestrations._handoff import HandoffAgentUserRequest

client  = OpenAIChatClient("gpt-4o")
analyst = client.as_agent(name="analyst",  description="Data analyst")
advisor = client.as_agent(name="advisor",  description="Investment advisor")

workflow = (
    HandoffBuilder(participants=[analyst, advisor])
    .with_start_agent(analyst)
    .build()
)

async def interactive_loop(prompt: str) -> None:
    """Drive the workflow interactively, resuming with responses for each request_info pause."""
    result = await workflow.run(prompt)

    while True:
        # Collect any pending HITL requests from this run
        pending = result.get_request_info_events()
        if not pending:
            # Workflow completed — print all outputs
            for output in result.get_outputs():
                print("Final output:", output)
            break

        # Ask the human for each paused request, then resume in one call
        responses: dict[str, list] = {}
        for event in pending:
            request: HandoffAgentUserRequest = event.data
            print("Agent said:", request.agent_response.messages[-1].contents[0])
            user_input = input("Your reply (or blank to terminate): ").strip()
            responses[event.request_id] = (
                HandoffAgentUserRequest.terminate()
                if not user_input
                else HandoffAgentUserRequest.create_response(user_input)
            )

        result = await workflow.run(responses=responses)

asyncio.run(interactive_loop("Analyse the S&P 500 trend and advise on portfolio allocation."))
```

**Example 2 — Automated driver (test / CI use)**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import HandoffBuilder
from agent_framework_orchestrations._handoff import HandoffAgentUserRequest

client = OpenAIChatClient("gpt-4o-mini")
a = client.as_agent(name="a", description="Specialist A")
b = client.as_agent(name="b", description="Specialist B")

workflow = (
    HandoffBuilder(participants=[a, b])
    .with_start_agent(a)
    .build()
)

SCRIPTED_RESPONSES = [
    "Please dig deeper into the cost implications.",
    "That's enough, please wrap up.",
]

async def automated() -> None:
    response_iter = iter(SCRIPTED_RESPONSES)
    result = await workflow.run("Analyse pricing strategies")

    while True:
        pending = result.get_request_info_events()
        if not pending:
            for output in result.get_outputs():
                print("Done:", output)
            break

        responses: dict[str, list] = {}
        for event in pending:
            try:
                reply = next(response_iter)
                responses[event.request_id] = HandoffAgentUserRequest.create_response(reply)
            except StopIteration:
                responses[event.request_id] = HandoffAgentUserRequest.terminate()

        result = await workflow.run(responses=responses)

asyncio.run(automated())
```

---

## 5 · `OrchestrationState`

**Sub-package:** `agent_framework_orchestrations._orchestration_state`

`OrchestrationState` is a **unified checkpoint dataclass** added in 1.9.0 to standardise
how conversation history, round counts, and pattern-specific metadata are persisted
across `GroupChatOrchestrator`, `HandoffBuilder`, and `MagenticOrchestrator` workflows.
Previously each orchestrator had its own ad-hoc serialisation format; this class
replaces all of them.

### Class signature

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from agent_framework._types import Message

@dataclass
class OrchestrationState:
    """Unified checkpoint container for all three orchestration patterns."""
    conversation:       list[Message]        # Full conversation history
    round_index:        int                  # Coordination rounds completed
    orchestrator_name:  str                  # Name tag for logging/debugging
    metadata:           dict[str, Any]       # Pattern-specific extension bag
    task:               Message | None = None  # The primary task/question

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestrationState": ...
```

### Field reference

| Field | Purpose | Pattern-specific notes |
|---|---|---|
| `conversation` | Full message history for all turns | All three patterns |
| `round_index` | Number of completed coordination rounds | GroupChat and Magentic use this; Handoff sets 0 |
| `orchestrator_name` | Identifier string for logging | Defaults to the builder's `name=` parameter |
| `metadata` | Extensible `dict` for pattern-specific data | Handoff stores `current_agent_id`; Magentic stores ledger state |
| `task` | Optional primary task message | Magentic sets this from the initial user prompt |

### Code examples

**Example 1 — Inspect checkpoint state after a run**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._workflows._checkpoint import FileCheckpointStorage
from agent_framework_orchestrations import HandoffBuilder
from agent_framework_orchestrations._orchestration_state import OrchestrationState

storage = FileCheckpointStorage("./orch_cp")
client  = OpenAIChatClient("gpt-4o")
a = client.as_agent(name="agent_a", description="Specialist A")
b = client.as_agent(name="agent_b", description="Specialist B")

workflow = (
    HandoffBuilder(
        participants=[a, b],
        checkpoint_storage=storage,
    )
    .with_start_agent(a)
    .build()
)

async def main() -> None:
    await workflow.run("Discuss distributed systems trade-offs")

    # List saved checkpoints — list_checkpoints takes a keyword-only workflow_name
    checkpoints = await storage.list_checkpoints(workflow_name=workflow.name)
    for cp in checkpoints:
        # cp is a WorkflowCheckpoint dataclass — state is a dict[str, Any]
        print(f"Checkpoint: {cp.checkpoint_id} @ {cp.timestamp}")
        print(f"Message threads: {list(cp.messages.keys())}")
        print(f"State keys: {list(cp.state.keys())}")

asyncio.run(main())
```

**Example 2 — Build a custom checkpoint for resumption**

```python
from agent_framework_orchestrations._orchestration_state import OrchestrationState

# Reconstruct an orchestration state from stored data
raw = {
    "conversation": [
        {"role": "user",      "contents": ["What is event sourcing?"]},
        {"role": "assistant", "contents": ["Event sourcing persists state changes as …"]},
    ],
    "round_index": 2,
    "orchestrator_name": "my_group_chat",
    "metadata": {"next_speaker": "expert_agent"},
    "task": {"role": "user", "contents": ["What is event sourcing?"]},
}
state = OrchestrationState.from_dict(raw)
assert state.round_index == 2
assert state.task is not None

# Serialise back to dict for persistence
serialised = state.to_dict()
assert "conversation" in serialised
assert "metadata" in serialised
```

**Example 3 — Extend metadata for a custom orchestrator**

```python
from typing import Any
from agent_framework._types import Message
from agent_framework_orchestrations._orchestration_state import OrchestrationState

def build_initial_state(task_text: str, orchestrator_name: str) -> OrchestrationState:
    """Build the initial orchestration state for a custom orchestrator."""
    task_message = Message(role="user", contents=[task_text])
    return OrchestrationState(
        conversation=[task_message],
        round_index=0,
        orchestrator_name=orchestrator_name,
        metadata={
            "phase": "planning",
            "assigned_agent": None,
            "stall_count": 0,
        },
        task=task_message,
    )

def advance_state(state: OrchestrationState, new_message: Message, agent_id: str) -> OrchestrationState:
    """Return an updated state after one round."""
    state.conversation.append(new_message)
    state.round_index += 1
    state.metadata["assigned_agent"] = agent_id
    return state

initial = build_initial_state("Summarise recent AI research", "my_orchestrator")
print(initial.round_index)      # 0
print(initial.metadata["phase"])  # "planning"
```

---

## 6 · `AgentModeProvider` · `get_agent_mode` · `set_agent_mode`

**Sub-package:** `agent_framework._harness._mode`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

`AgentModeProvider` injects **mode-switching tools** (`mode_get`, `mode_set`) and
matching instructions into each agent turn.  It enables agents to operate in distinct
behavioural modes — by default `"plan"` (interactive, asks clarifying questions) and
`"execute"` (autonomous, runs without asking).  The current mode is persisted in
`AgentSession.state`.

The public helpers `get_agent_mode` and `set_agent_mode` let external code read and
change the mode between turns (e.g., a UI layer switching an agent from planning to
execution).

### Class signature

```python
from collections.abc import Mapping
from agent_framework._harness._mode import AgentModeProvider, get_agent_mode, set_agent_mode

class AgentModeProvider:  # ContextProvider subclass
    def __init__(
        self,
        source_id: str = "agent_mode",
        *,
        default_mode:      str | None = None,
        mode_descriptions: Mapping[str, str] | None = None,
        instructions:      str | None = None,
    ) -> None: ...

def get_agent_mode(
    session: AgentSession,
    *,
    source_id:     str = "agent_mode",
    default_mode:  str | None = None,
    available_modes: tuple[str, ...] | None = None,
) -> str: ...

def set_agent_mode(
    session: AgentSession,
    mode:    str,
    *,
    source_id:      str = "agent_mode",
    available_modes: tuple[str, ...] | None = None,
) -> None: ...
```

### `mode_descriptions` Mapping

Keys are mode names (case-insensitive after normalisation); values are multi-line
descriptions injected into the `{available_modes}` placeholder of the instructions
template.  The **default** map contains two entries:

| Mode | Default behaviour |
|---|---|
| `"plan"` | Interactive — analyse, break down, ask clarifying questions, create todos, get user approval |
| `"execute"` | Autonomous — carry out the approved plan without asking for confirmation |

### Instructions template placeholders

The `instructions` string can contain two placeholders:

| Placeholder | Replaced with |
|---|---|
| `{current_mode}` | The active mode name for this turn |
| `{available_modes}` | All mode descriptions rendered as `#### ModeName\n\ndescription\n\n` |

### Code examples

**Example 1 — Default plan / execute modes**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._mode import AgentModeProvider

client = OpenAIChatClient("gpt-4o")
agent  = client.as_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[AgentModeProvider()],
)

async def main() -> None:
    session = agent.create_session()
    # First turn: agent is in "plan" mode → will ask clarifying questions
    response = await agent.run("Help me refactor my codebase", session=session)
    print(response.messages[-1].contents[0])

asyncio.run(main())
```

**Example 2 — Custom modes with custom instructions**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._mode import AgentModeProvider

MODES = {
    "research": (
        "Gather information from multiple sources. "
        "Do not write final answers yet — only collect raw data."
    ),
    "synthesise": (
        "Combine the research into a coherent, well-structured report. "
        "Do not ask further questions."
    ),
    "review": (
        "Read the synthesised report, identify errors or gaps, "
        "and output a list of suggested improvements."
    ),
}

mode_provider = AgentModeProvider(
    default_mode="research",
    mode_descriptions=MODES,
    instructions=(
        "You are a research assistant operating in {current_mode} mode.\n\n"
        "{available_modes}"
        "Strictly follow the instructions for your current mode."
    ),
)

client = OpenAIChatClient("gpt-4o")
agent  = client.as_agent(
    name="researcher",
    context_providers=[mode_provider],
)

async def main() -> None:
    session = agent.create_session()
    # Research phase
    await agent.run("Topic: large language model evaluation benchmarks", session=session)

    # External switch to synthesise
    from agent_framework._harness._mode import set_agent_mode
    set_agent_mode(session, "synthesise", source_id="agent_mode")

    # Next turn: agent is now in "synthesise" mode
    report = await agent.run("Now write the final report.", session=session)
    print(report.messages[-1].contents[0])

asyncio.run(main())
```

**Example 3 — Read mode from a UI layer and drive workflow**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._mode import AgentModeProvider, get_agent_mode, set_agent_mode

client = OpenAIChatClient("gpt-4o")
agent  = client.as_agent(
    name="workflow_agent",
    context_providers=[AgentModeProvider(default_mode="plan")],
)

async def ui_driver() -> None:
    session = agent.create_session()

    # Simulate a user starting in plan mode
    await agent.run("Design a CI/CD pipeline for a Python monorepo", session=session)
    current = get_agent_mode(session)
    print(f"After planning, mode is: {current}")  # "plan"

    # UI layer approves — switch to execute
    set_agent_mode(session, "execute")
    print(f"Switched to: {get_agent_mode(session)}")  # "execute"

    # Execute phase: agent receives a notification message that mode changed
    # agent.run() returns AgentResponse directly (not WorkflowRunResult)
    result = await agent.run("Proceed with implementation.", session=session)
    print(result.messages[-1].contents[0])

asyncio.run(ui_driver())
```

---

## 7 · `TodoItem` · `TodoInput` · `TodoCompleteInput`

**Sub-package:** `agent_framework._harness._todo`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

These three classes are the **DTO layer** of the todo harness.  They all implement
`SerializationMixin` so they can round-trip through JSON / session state / file storage
cleanly.

### Class signatures

```python
from agent_framework._harness._todo import TodoItem, TodoInput, TodoCompleteInput

class TodoItem:          # SerializationMixin
    id:          int
    title:       str
    description: str | None
    is_complete: bool

    def __init__(self, id: int, title: str, description: str | None = None, is_complete: bool = False) -> None: ...
    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, raw_item: MutableMapping[str, Any], /, *, dependencies: MutableMapping[str, Any] | None = None) -> "TodoItem": ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class TodoInput:          # SerializationMixin
    title:       str       # stripped; non-empty enforced
    description: str | None

    def __init__(self, title: str, description: str | None = None) -> None: ...
    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, raw_todo: MutableMapping[str, Any], /, *, dependencies: MutableMapping[str, Any] | None = None) -> "TodoInput": ...

class TodoCompleteInput:   # SerializationMixin
    id:     int
    reason: str    # stripped; non-empty enforced

    def __init__(self, id: int, reason: str) -> None: ...
    def to_dict(self, *, exclude: set[str] | None = None, exclude_none: bool = True) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, raw_item: MutableMapping[str, Any], /, *, dependencies: MutableMapping[str, Any] | None = None) -> "TodoCompleteInput": ...
```

### Validation rules

| Class | Validation enforced at `__init__` |
|---|---|
| `TodoItem` | `id` must be `int`; `title` must be non-empty `str`; `is_complete` must be `bool` |
| `TodoInput` | `title` is stripped; empty string raises `ValueError` |
| `TodoCompleteInput` | `id` must be `int`; `reason` is stripped; empty string raises `ValueError` |

### Code examples

**Example 1 — Basic DTO round-trip**

```python
from agent_framework._harness._todo import TodoItem, TodoInput, TodoCompleteInput

# Create a todo item directly (normally done by TodoProvider)
item = TodoItem(id=1, title="Write unit tests", description="Cover edge cases", is_complete=False)
print(repr(item))
# TodoItem(id=1, title='Write unit tests', description='Cover edge cases', is_complete=False)

# Serialise
payload = item.to_dict(exclude_none=False)
assert payload == {"id": 1, "title": "Write unit tests", "description": "Cover edge cases", "is_complete": False}

# Round-trip from persisted data
restored = TodoItem.from_dict({"id": 1, "title": "Write unit tests", "description": "Cover edge cases", "is_complete": False})
assert restored == item
```

**Example 2 — `TodoInput` for batch creation**

```python
from agent_framework._harness._todo import TodoInput

inputs = [
    TodoInput(title="Set up project scaffold"),
    TodoInput(title="Write README", description="Include quick-start section"),
    TodoInput(title="Configure CI"),
]
for inp in inputs:
    serialised = inp.to_dict()  # exclude_none=True → description omitted if None
    print(serialised)

# Validation: empty title raises ValueError
try:
    TodoInput(title="   ")  # stripped → empty
except ValueError as exc:
    print(exc)  # "Todo input title must be a non-empty string."
```

**Example 3 — `TodoCompleteInput` for marking done**

```python
from agent_framework._harness._todo import TodoCompleteInput

complete_ops = [
    TodoCompleteInput(id=1, reason="Scaffold created with cookiecutter template"),
    TodoCompleteInput(id=3, reason="GitHub Actions workflow added and passing"),
]
for op in complete_ops:
    print(op.to_dict())
# {'id': 1, 'reason': 'Scaffold created with cookiecutter template'}
# {'id': 3, 'reason': 'GitHub Actions workflow added and passing'}

# Round-trip from raw tool argument payload
raw = {"id": 2, "reason": "README written with all required sections"}
op = TodoCompleteInput.from_dict(raw)
assert op.id == 2
assert op.reason == "README written with all required sections"
```

---

## 8 · `TodoStore` · `TodoSessionStore` · `TodoFileStore`

**Sub-package:** `agent_framework._harness._todo`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

`TodoStore` is the abstract backing-store ABC.  Two concrete implementations are
provided: `TodoSessionStore` (state lives inside `AgentSession.state`) and
`TodoFileStore` (state lives in one JSON file per session on disk).

### Class signatures

```python
from abc import ABC, abstractmethod
from pathlib import Path
from agent_framework._sessions import AgentSession
from agent_framework._harness._todo import TodoItem, TodoStore, TodoSessionStore, TodoFileStore

class TodoStore(ABC):
    @abstractmethod
    async def load_state(self, session: AgentSession, *, source_id: str) -> tuple[list[TodoItem], int]: ...
    @abstractmethod
    async def save_state(self, session: AgentSession, items: list[TodoItem], *, next_id: int, source_id: str) -> None: ...
    async def load_items(self, session: AgentSession, *, source_id: str) -> list[TodoItem]: ...

class TodoSessionStore(TodoStore):
    """Store todo state inside AgentSession.state — zero-config, ephemeral."""
    # No __init__ parameters

class TodoFileStore(TodoStore):
    """Store todo state in one JSON file per session — durable across process restarts."""
    def __init__(
        self,
        base_path: str | Path,
        *,
        kind:             str = "todos",
        owner_prefix:     str = "",
        owner_state_key:  str | None = None,
        state_filename:   str = "todos.json",
    ) -> None: ...
```

### `TodoFileStore` parameters

| Parameter | Default | Purpose |
|---|---|---|
| `base_path` | required | Root directory for all todo state files |
| `kind` | `"todos"` | Sub-directory bucket name within each owner directory |
| `owner_prefix` | `""` | String prepended to the resolved owner ID in the path |
| `owner_state_key` | `None` | Session-state key that holds the logical owner ID (e.g., user ID for multi-tenant apps) |
| `state_filename` | `"todos.json"` | File name for the persisted state JSON |

### Path-traversal security

`TodoFileStore._get_state_path` always resolves the final path with `.resolve()` and
asserts `state_path.is_relative_to(self._base_root)` before returning.  Any
`session_id` or owner value containing `..` or path separators raises `ValueError`,
preventing directory traversal attacks.  Windows reserved file stems (CON, NUL, COM1…
LPT9) are detected and encoded with a `~todo-` prefix.

### Code examples

**Example 1 — Session store (in-memory, ephemeral)**

```python
import asyncio
from agent_framework._sessions import AgentSession
from agent_framework._harness._todo import TodoItem, TodoSessionStore

store   = TodoSessionStore()
session = AgentSession(session_id="sess-001")

async def main() -> None:
    # Initially empty
    items, next_id = await store.load_state(session, source_id="todo")
    assert items == [] and next_id == 1

    # Add an item
    item = TodoItem(id=next_id, title="Draft proposal")
    await store.save_state(session, [item], next_id=next_id + 1, source_id="todo")

    # Read back
    loaded, _ = await store.load_state(session, source_id="todo")
    assert loaded[0].title == "Draft proposal"

asyncio.run(main())
```

**Example 2 — File store with multi-tenant owner isolation**

```python
import asyncio
from agent_framework._sessions import AgentSession
from agent_framework._harness._todo import TodoItem, TodoFileStore

store = TodoFileStore(
    base_path="./todo_storage",
    kind="tasks",
    owner_state_key="user_id",   # reads owner from session.state["user_id"]
    owner_prefix="u_",
)

async def main() -> None:
    session = AgentSession(session_id="session-42")
    session.state["user_id"] = "alice"

    item = TodoItem(id=1, title="Review pull request")
    await store.save_state(session, [item], next_id=2, source_id="tasks")
    # Saved to:  ./todo_storage/u_alice/tasks/session-42/todos.json

    loaded = await store.load_items(session, source_id="tasks")
    print(loaded[0].title)  # "Review pull request"

asyncio.run(main())
```

**Example 3 — Custom `TodoStore` implementation**

```python
import asyncio
from agent_framework._sessions import AgentSession
from agent_framework._harness._todo import TodoItem, TodoStore

class InMemoryTodoStore(TodoStore):
    """Simple dict-backed store for testing."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[list[TodoItem], int]] = {}

    async def load_state(self, session: AgentSession, *, source_id: str) -> tuple[list[TodoItem], int]:
        key = f"{session.session_id}:{source_id}"
        return self._data.get(key, ([], 1))

    async def save_state(self, session: AgentSession, items: list[TodoItem], *, next_id: int, source_id: str) -> None:
        key = f"{session.session_id}:{source_id}"
        self._data[key] = (list(items), next_id)

async def demo() -> None:
    store   = InMemoryTodoStore()
    session = AgentSession(session_id="test")

    items, nid = await store.load_state(session, source_id="todo")
    assert items == []

    await store.save_state(
        session,
        [TodoItem(id=1, title="Build custom store"), TodoItem(id=2, title="Write tests")],
        next_id=3,
        source_id="todo",
    )
    loaded, next_id = await store.load_state(session, source_id="todo")
    assert len(loaded) == 2 and next_id == 3

asyncio.run(demo())
```

---

## 9 · `TodoProvider`

**Sub-package:** `agent_framework._harness._todo`  
**Status:** `@experimental(ExperimentalFeature.HARNESS)`

`TodoProvider` is a `ContextProvider` that injects **five todo-management tools** and
matching instructions into each agent turn.  It wires a `TodoStore` backend (default:
`TodoSessionStore`) and serialises concurrent writes safely using a per-session
`asyncio.Lock` held in a `WeakKeyDictionary` (so locks are garbage-collected when the
session is GC'd).

### Class signature

```python
from agent_framework._harness._todo import TodoProvider, TodoStore

class TodoProvider:   # ContextProvider subclass
    def __init__(
        self,
        source_id:    str = "todo",
        *,
        instructions: str | None = None,
        store:        TodoStore | None = None,
    ) -> None: ...
```

### Injected tools

| Tool name | Approval mode | Description |
|---|---|---|
| `todos_add` | `"never_require"` | Add one or more todo items |
| `todos_complete` | `"never_require"` | Mark one or more items complete with reasons |
| `todos_remove` | `"never_require"` | Remove items by IDs |
| `todos_get_remaining` | `"never_require"` | Return only incomplete items |
| `todos_get_all` | `"never_require"` | Return all items including completed ones |

### Concurrency model

Every mutating tool (`todos_add`, `todos_complete`, `todos_remove`) acquires
`self._mutation_lock(session)` — a per-session `asyncio.Lock` — to prevent race
conditions when multiple concurrent tool calls operate on the same session.  The locks
are stored in a `weakref.WeakKeyDictionary` so they do not pin sessions in memory.

### Code examples

**Example 1 — Session-backed todos (default)**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._todo import TodoProvider

client = OpenAIChatClient("gpt-4o")
agent  = client.as_agent(
    name="planner",
    instructions="You are a project planning assistant. Use the todo tools to track tasks.",
    context_providers=[TodoProvider()],
)

async def main() -> None:
    session = agent.create_session()
    response = await agent.run(
        "Plan a 3-sprint roadmap for a mobile app MVP. Create todos for each sprint.",
        session=session,
    )
    print(response.messages[-1].contents[0])

asyncio.run(main())
```

**Example 2 — File-backed todos for persistence across restarts**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._todo import TodoProvider, TodoFileStore

store  = TodoFileStore(base_path="./project_todos")
client = OpenAIChatClient("gpt-4o")
agent  = client.as_agent(
    name="task_tracker",
    instructions="Track project tasks in the todo list. Mark items complete as work finishes.",
    context_providers=[TodoProvider(store=store)],
)

async def main() -> None:
    session = agent.create_session()

    # Turn 1: create initial tasks
    await agent.run("We need to build a REST API with auth, CRUD endpoints, and docs.", session=session)

    # Turn 2 (same session, survives process restart with file store)
    response = await agent.run("Auth is done. Update the task list.", session=session)
    print(response.messages[-1].contents[0])

asyncio.run(main())
```

**Example 3 — Combined with `AgentModeProvider` for plan/execute workflow**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework._harness._mode import AgentModeProvider, set_agent_mode
from agent_framework._harness._todo import TodoProvider, TodoFileStore

store = TodoFileStore(base_path="./workflow_todos")
client = OpenAIChatClient("gpt-4o")
agent = client.as_agent(
    name="workflow_agent",
    instructions="Use the todo list to track tasks and the mode system to manage your workflow.",
    context_providers=[
        AgentModeProvider(default_mode="plan"),
        TodoProvider(store=store),
    ],
)

async def main() -> None:
    session = agent.create_session()

    # Planning phase: agent asks questions, creates todos
    await agent.run(
        "Migrate our PostgreSQL database to Azure Cosmos DB for NoSQL.",
        session=session,
    )

    # Approve plan and switch to execute
    set_agent_mode(session, "execute")

    # Execution phase: agent works autonomously, marks todos done
    # agent.run() returns AgentResponse directly (not WorkflowRunResult)
    result = await agent.run("Begin the migration.", session=session)
    print(result.messages[-1].contents[0])

asyncio.run(main())
```

---

## 10 · `MagenticResetSignal` · `StandardMagenticManager`

**Sub-package:** `agent_framework_orchestrations._magentic`

`MagenticResetSignal` is the sentinel exception that signals the Magentic inner loop to
**reset context** (clear chat history, reset stall counts) and start the task over with
updated facts.  It is raised inside the orchestrator when the manager determines no
progress is being made.

`StandardMagenticManager` is the **production-ready `MagenticManagerBase` implementation**
that drives LLM calls for planning, progress assessment, and final-answer synthesis.
Every prompt it uses is individually overridable.

### Class signatures

```python
from agent_framework_orchestrations._magentic import (
    MagenticResetSignal,
    StandardMagenticManager,
)

class MagenticResetSignal:
    """Raised inside the Magentic inner loop to trigger a full context reset."""
    pass

class StandardMagenticManager:   # MagenticManagerBase subclass
    MANAGER_NAME: ClassVar[str] = "StandardMagenticManager"

    def __init__(
        self,
        agent: SupportsAgentRun,
        task_ledger: Any | None = None,
        *,
        # Prompt overrides — all default to built-in Microsoft prompts
        task_ledger_facts_prompt:        str | None = None,
        task_ledger_plan_prompt:         str | None = None,
        task_ledger_full_prompt:         str | None = None,
        task_ledger_facts_update_prompt: str | None = None,
        task_ledger_plan_update_prompt:  str | None = None,
        progress_ledger_prompt:          str | None = None,
        final_answer_prompt:             str | None = None,
        # Flow control
        max_stall_count:          int      = 3,
        max_reset_count:          int | None = None,
        max_round_count:          int | None = None,
        progress_ledger_retry_count: int | None = None,  # default 3
    ) -> None: ...
```

### Flow-control parameters

| Parameter | Default | Effect |
|---|---|---|
| `max_stall_count` | `3` | Number of consecutive rounds with no progress before a reset is triggered |
| `max_reset_count` | `None` (unlimited) | Maximum number of context resets before the task is abandoned |
| `max_round_count` | `None` (unlimited) | Hard cap on total orchestration rounds |
| `progress_ledger_retry_count` | `3` | How many times to retry LLM calls for the progress ledger when the response is malformed |

### Prompt override fields

After construction each prompt is stored as an instance attribute and can be mutated:

```python
mgr = StandardMagenticManager(agent)
mgr.task_ledger_full_prompt        # the initial facts+plan prompt
mgr.task_ledger_facts_update_prompt  # facts re-extraction after stall/reset
mgr.progress_ledger_prompt         # the structured JSON progress assessment prompt
mgr.final_answer_prompt            # the synthesis prompt for the closing answer
```

### Code examples

**Example 1 — Basic `MagenticBuilder` with `StandardMagenticManager`**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import MagenticBuilder
from agent_framework_orchestrations._magentic import StandardMagenticManager

client    = OpenAIChatClient("gpt-4o")
orchestrator_agent = client.as_agent(
    name="orchestrator",
    instructions="You are the Magentic orchestrator. Create plans and assess progress.",
)
worker_a  = client.as_agent(name="searcher",  description="Web search specialist")
worker_b  = client.as_agent(name="coder",     description="Python coding specialist")

manager = StandardMagenticManager(
    agent=orchestrator_agent,
    max_stall_count=2,
    max_round_count=20,
)

workflow = MagenticBuilder(participants=[worker_a, worker_b], manager=manager).build()

async def main() -> None:
    result = await workflow.run("Build and test a Python script that downloads the top 10 Hacker News stories")
    print(result.get_outputs()[-1].messages[-1].contents[0])

asyncio.run(main())
```

**Example 2 — Custom prompt override for domain-specific planning**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import MagenticBuilder
from agent_framework_orchestrations._magentic import StandardMagenticManager

MEDICAL_PLAN_PROMPT = """
You are coordinating a medical literature review task.
Task: {task}
Participants: {participants}

Create a structured research plan that:
1. Identifies key search terms and databases
2. Assigns each database to the most appropriate specialist
3. Sets acceptance criteria for sufficient evidence
4. Plans synthesis and citation format
"""

MEDICAL_PROGRESS_PROMPT = """
Assess the current progress of the medical literature review.
Return ONLY valid JSON:
{
  "is_complete": false,
  "next_speaker": "name_of_next_participant",
  "instruction": "specific instruction for next participant",
  "is_stalled": false
}
"""

client  = OpenAIChatClient("gpt-4o")
manager_agent = client.as_agent(name="medical_coordinator")
specialist_a  = client.as_agent(name="pubmed_searcher",  description="PubMed search specialist")
specialist_b  = client.as_agent(name="clinicaltrials",   description="ClinicalTrials.gov specialist")

manager = StandardMagenticManager(
    agent=manager_agent,
    task_ledger_full_prompt=MEDICAL_PLAN_PROMPT,
    progress_ledger_prompt=MEDICAL_PROGRESS_PROMPT,
    max_stall_count=3,
    max_reset_count=2,
    progress_ledger_retry_count=5,  # medical prompts may need more retries
)

workflow = MagenticBuilder(participants=[specialist_a, specialist_b], manager=manager).build()

async def main() -> None:
    result = await workflow.run("Review RCT evidence for metformin in type 2 diabetes prevention")
    print(result.get_outputs()[-1].messages[-1].contents[0])

asyncio.run(main())
```

**Example 3 — Observing `MagenticResetSignal` via workflow events**

```python
import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework_orchestrations import MagenticBuilder
from agent_framework_orchestrations._magentic import StandardMagenticManager

client = OpenAIChatClient("gpt-4o")
mgr_agent = client.as_agent(name="mgr", instructions="Orchestrate the task.")
worker    = client.as_agent(name="worker", description="General purpose worker")

manager  = StandardMagenticManager(agent=mgr_agent, max_stall_count=1)
workflow = MagenticBuilder(participants=[worker], manager=manager).build()

async def main() -> None:
    reset_count = 0
    async for event in workflow.stream("Complete a complex multi-step research task"):
        if event.type == "magentic_orchestrator":
            from agent_framework_orchestrations._magentic import MagenticOrchestratorEventType
            if event.data.event_type == MagenticOrchestratorEventType.REPLANNED:
                reset_count += 1
                print(f"Reset #{reset_count}: context cleared, replanning from scratch")
        elif event.type == "output":
            print("Final output:", event.data)

    print(f"Total resets triggered: {reset_count}")

asyncio.run(main())
```

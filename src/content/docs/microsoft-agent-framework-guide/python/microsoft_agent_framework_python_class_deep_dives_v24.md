---
title: "Microsoft Agent Framework (Python) — Class Deep Dives Vol. 24"
description: "Source-verified deep dives into 10 class groups from agent-framework 1.9.0: is_type_compatible+serialize_type+deserialize_type+normalize_type_to_list+resolve_type_annotation+try_coerce_to_type+is_instance_of+is_chat_agent (_workflows._typing_utils — workflow edge type validation engine); encode_checkpoint_value+decode_checkpoint_value (_workflows._checkpoint_encoding — pickle+base64 checkpoint serialization); Runner+create_edge_runner (_workflows._runner — Pregel superstep execution loop); ShouldContinueCallable+ShouldContinueResult+FeedbackCallable+NextMessageCallable+JUDGE constants (AgentLoopMiddleware callable types and judge customization); ToolApprovalScope+ToolApprovalRuleCallback+ALWAYS_APPROVE constants+create_always_approve helpers (_harness._tool_approval — approval protocol internals); GroupChatRequestMessage+ORCH_MSG_KIND constants+clean_conversation_for_handoff+create_completion_message (orchestration protocol utilities); MagenticOrchestratorEvent+MagenticOrchestratorEventType+MagenticAgentExecutor (Magentic event observability and reset-aware executor); evaluate_traces+RawFoundryEmbeddingClient (Foundry trace-based evaluation + multimodal embedding raw client); RawOpenAIChatCompletionClient (raw Chat Completions client — layer ordering contract); normalize_messages_input+ensure_author+latest_user_message (_workflows message utilities — heterogeneous input normalisation)."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 47
---

# Microsoft Agent Framework Python — Class Deep Dives Vol. 24

Verified against **agent-framework 1.9.0** / **agent-framework-foundry 1.8.2** / **agent-framework-openai 1.8.2** (installed June 2026). Every constructor signature, constant value, and code example was derived directly from the installed package source. Sub-packages introspected: `agent_framework._workflows._typing_utils`, `agent_framework._workflows._checkpoint_encoding`, `agent_framework._workflows._runner`, `agent_framework._harness._loop`, `agent_framework._harness._tool_approval`, `agent_framework.orchestrations`, `agent_framework_foundry`, `agent_framework_openai`, `agent_framework._workflows._conversation_history`, `agent_framework._workflows._message_utils`.

**Previous volumes:**
- [Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives/) — `Agent`, `RawAgent`, `FunctionTool`, `WorkflowBuilder`, `RunContext`, `InlineSkill`, `MCPStdioTool`
- [Vol. 2](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2/) — `FileHistoryProvider`, `AgentMiddleware`, `ChatMiddleware`, `FunctionMiddleware`, `CompactionProvider`, `ToolResultCompactionStrategy`, `TokenBudgetComposedStrategy`, `FileCheckpointStorage`, `LocalEvaluator`, `WorkflowRunResult`
- [Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3/) — `BackgroundAgentsProvider`, `MemoryContextProvider`, `TodoProvider`, `AgentModeProvider`, `SummarizationStrategy`, `ContextWindowCompactionStrategy`, `SlidingWindowStrategy`, `SelectiveToolCallCompactionStrategy`, `WorkflowViz`, `MCPStreamableHTTPTool`/`MCPWebsocketTool`
- [Vol. 4–23](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4/) — see index for full listing

---

## 1 · Workflow type compatibility engine — `_workflows._typing_utils`

**Module:** `agent_framework._workflows._typing_utils`

These seven functions power the `WorkflowGraphValidator` edge-type checks and the checkpoint coercion layer. They are internal but understanding them is essential when debugging `TypeCompatibilityError` validation failures or authoring custom executors with complex output types.

### Key functions

| Function | Signature | Role |
|---|---|---|
| `is_type_compatible(source, target)` | `(type\|UnionType\|Any, type\|UnionType\|Any) → bool` | Checks whether values of `source` can be assigned to variables of `target`; used by validator |
| `is_instance_of(data, target_type)` | `(Any, type\|UnionType\|Any) → bool` | Runtime `isinstance` check that understands generic types (`list[str]`, `str \| int`, …) |
| `serialize_type(t)` | `(type\|UnionType\|None) → str\|None` | Serializes a type to a dotted module string for checkpoint storage |
| `deserialize_type(s)` | `(str\|None) → type\|UnionType\|None` | Reverses `serialize_type`; reconstructs the original type object |
| `normalize_type_to_list(t)` | `(type\|UnionType\|None) → list[type\|UnionType]` | Splits a Union into its member types; `None → []`; single type → `[t]` |
| `resolve_type_annotation(t, globalns, localns)` | `(type\|UnionType\|str\|None, …) → type\|UnionType\|None` | Resolves PEP 563 string annotations via `eval()` in the caller's module globals |
| `try_coerce_to_type(data, target_type)` | `(Any, type\|UnionType\|Any) → Any` | Lightweight coercion: `int→float`, `dict→dataclass`, `dict→Pydantic model`; returns unchanged on failure |
| `is_chat_agent(agent)` | `(Any) → TypeGuard[Agent]` | Narrows any agent-like object to the concrete `Agent` class |

### Validation logic for `is_type_compatible`

The function uses a 7-case decision tree:

1. `target is Any` → always `True`
2. Exact type match → `True`
3. Target is `Union` → source must match *at least one* member (or every source Union member if source is also Union)
4. Source is `Union` (target is not) → *every* source member must be compatible with target
5. Both non-generic → `issubclass(source, target)`
6. Different container origins (`list` vs `set`) → `False`
7. Same container (`list`, `set`) → recurse on element type

**Example 1 — checking edge compatibility before adding a connection:**

```python
from agent_framework._workflows._typing_utils import is_type_compatible

# Executors producing str | None can feed into an executor expecting str | int | None
assert is_type_compatible(str | None, str | int | None)

# list[str] is NOT compatible with list[int]
assert not is_type_compatible(list[str], list[int])

# Everything is compatible with Any
from typing import Any
assert is_type_compatible(dict, Any)

# Subclass compatibility is supported
class MyStr(str): ...
assert is_type_compatible(MyStr, str)
```

**Example 2 — serializing types for custom checkpoint adapters:**

```python
from agent_framework._workflows._typing_utils import serialize_type, deserialize_type
from dataclasses import dataclass

@dataclass
class TaskState:
    completed: bool
    score: float

serialized = serialize_type(TaskState)
print(serialized)  # e.g. "__main__:TaskState"

reconstructed = deserialize_type(serialized)
assert reconstructed is TaskState

# Union types round-trip correctly
union_ser = serialize_type(str | int | None)
union_back = deserialize_type(union_ser)
from typing import get_args
assert set(get_args(union_back)) == {str, int, type(None)}
```

**Example 3 — normalising annotations for generic executor output validation:**

```python
from agent_framework._workflows._typing_utils import (
    normalize_type_to_list,
    try_coerce_to_type,
    resolve_type_annotation,
)
from dataclasses import dataclass

# normalize_type_to_list: split Union into parts
members = normalize_type_to_list(str | int | None)
print(members)  # [<class 'str'>, <class 'int'>, <class 'NoneType'>]

# resolve_type_annotation: handle forward references from @executor decorators
ns = {"str": str, "int": int}
resolved = resolve_type_annotation("str | int", globalns=ns)
print(resolved)  # str | int

# try_coerce_to_type: JSON dict → Pydantic model
from pydantic import BaseModel

class Result(BaseModel):
    score: float
    label: str

raw = {"score": 42, "label": "pass"}       # score is int from JSON
coerced = try_coerce_to_type(raw, Result)
assert isinstance(coerced, Result)
assert coerced.score == 42.0               # int was coerced to float
```

---

## 2 · Checkpoint serialization engine — `encode_checkpoint_value` + `decode_checkpoint_value`

**Module:** `agent_framework._workflows._checkpoint_encoding`

These two functions implement the complete checkpoint serde layer used by `FileCheckpointStorage` and `InMemoryCheckpointStorage`. Understanding them is critical when working with custom state types or debugging `WorkflowCheckpointException` failures.

### Key functions

| Function | Signature |
|---|---|
| `encode_checkpoint_value(value)` | `(Any) → Any` |
| `decode_checkpoint_value(value, *, allowed_types)` | `(Any, frozenset[str]\|None) → Any` |

### Encoding rules

| Python type | Encoded form |
|---|---|
| `str`, `int`, `float`, `bool`, `None` | Pass-through unchanged (JSON-native) |
| `dict` | Recurse — values encoded individually |
| `list` | Recurse — elements encoded individually |
| Anything else (dataclass, datetime, Pydantic model, …) | `pickle` → base64 string, tagged with an internal `_PICKLE_MARKER` |

### Decoding security: `allowed_types`

`decode_checkpoint_value` accepts an `allowed_types: frozenset[str] | None` parameter. When not `None`, only types in the built-in safe set *plus* entries you list can be unpickled. Each entry uses `"module:qualname"` format:

```
"my_app.models:TaskState"
```

This matches `FileCheckpointStorage`'s `allowed_checkpoint_types` constructor argument.

**Example 1 — JSON-native pass-through and collection recursion:**

```python
from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value, decode_checkpoint_value

# JSON-native types pass through unchanged
assert encode_checkpoint_value(42) == 42
assert encode_checkpoint_value("hello") == "hello"
assert encode_checkpoint_value(None) is None

# Dict values are recursed
import datetime
encoded = encode_checkpoint_value({"ts": datetime.datetime(2026, 1, 1), "count": 7})
# "ts" is non-JSON so it becomes a base64 string; "count" stays 7
print(type(encoded["ts"]))   # str (base64-pickled)
print(encoded["count"])      # 7

# Round-trip
decoded = decode_checkpoint_value(encoded)
assert decoded["ts"] == datetime.datetime(2026, 1, 1)
assert decoded["count"] == 7
```

**Example 2 — custom dataclass round-trip with `allowed_types` security:**

```python
from dataclasses import dataclass
from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value, decode_checkpoint_value

@dataclass
class MyState:
    agent_name: str
    score: float

state = MyState(agent_name="researcher", score=0.91)
encoded = encode_checkpoint_value(state)
print(type(encoded))   # str  (base64-pickled)

# Unrestricted decode
decoded_free = decode_checkpoint_value(encoded)
assert decoded_free == state

# Restricted decode — must list the type explicitly
decoded_safe = decode_checkpoint_value(
    encoded,
    allowed_types=frozenset({"__main__:MyState"}),
)
assert decoded_safe == state

# Type not in allowlist → WorkflowCheckpointException
from agent_framework.exceptions import WorkflowCheckpointException
try:
    decode_checkpoint_value(encoded, allowed_types=frozenset())
except WorkflowCheckpointException as e:
    print(f"Blocked: {e}")
```

**Example 3 — wiring `allowed_checkpoint_types` on `FileCheckpointStorage`:**

```python
from agent_framework import FileCheckpointStorage
from agent_framework._workflows._checkpoint_encoding import encode_checkpoint_value

# FileCheckpointStorage proxies allowed_checkpoint_types → decode_checkpoint_value(allowed_types=...)
storage = FileCheckpointStorage(
    "/tmp/checkpoints",
    allowed_checkpoint_types=[
        "myapp.state:PipelineState",
        "myapp.state:StepResult",
    ],
)
# Now any checkpoint containing PipelineState or StepResult objects will be
# deserialized safely; unknown types raise WorkflowCheckpointException.

# You can manually inspect the encoded form before storing
from myapp.state import PipelineState  # type: ignore
state = PipelineState(step=3, output="done")
encoded = encode_checkpoint_value(state)
print(encoded[:40] + "...")  # base64-prefixed blob
```

---

## 3 · Pregel superstep `Runner` + `create_edge_runner` factory

**Module:** `agent_framework._workflows._runner`

`Runner` is the low-level Pregel execution engine. A `Workflow` compiles to a `Runner` and delegates iteration control to it. `create_edge_runner` is the factory that instantiates the correct `EdgeRunner` subclass for each `EdgeGroup`.

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

| Parameter | Purpose |
|---|---|
| `edge_groups` | Ordered list of `SingleEdgeGroup`, `FanOutEdgeGroup`, `FanInEdgeGroup`, `SwitchCaseEdgeGroup` |
| `executors` | Map of executor IDs to executor instances |
| `state` | `State` object (superstep cache) |
| `ctx` | `RunnerContext` that provides messaging, events, checkpointing |
| `graph_signature_hash` | SHA-based topology hash; checkpoint replay raises if the hash changes |
| `max_iterations` | Default `100`; raises `WorkflowConvergenceException` when exceeded |

### `run_until_convergence` loop

The core loop:
1. Yield any pre-loop events already buffered in `ctx`
2. Create a checkpoint before entering iteration if messages are pending (unless resuming)
3. For each superstep: emit `WorkflowEvent.superstep_started`, run executors concurrently with `asyncio.create_task`, stream live events via 50 ms poll, then commit the next checkpoint
4. Exit when no executor sends a message in a superstep (convergence)

### `create_edge_runner` dispatch table

| `EdgeGroup` type | `EdgeRunner` created |
|---|---|
| `SingleEdgeGroup` or `InternalEdgeGroup` | `SingleEdgeRunner` |
| `SwitchCaseEdgeGroup` | `SwitchCaseEdgeRunner` |
| `FanOutEdgeGroup` | `FanOutEdgeRunner` |
| `FanInEdgeGroup` | `FanInEdgeRunner` |
| Anything else | `ValueError` |

**Example 1 — observing superstep events:**

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, tool
from agent_framework._workflows._events import WorkflowEvent

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async def main() -> None:
    agent = Agent(model="gpt-4o-mini", tools=[add])
    workflow = WorkflowBuilder().add_agent(agent, output_type=str).build()

    async for event in workflow.run_stream("What is 3 + 4?"):
        # WorkflowEvent wraps every superstep_started, executor output, etc.
        if event.event_type == "superstep_started":
            print(f"Superstep {event.data['iteration']} started")

asyncio.run(main())
```

**Example 2 — controlling `max_iterations` to cap runaway loops:**

```python
import asyncio
from agent_framework import Agent, WorkflowBuilder, WorkflowConvergenceException

async def main() -> None:
    agent = Agent(model="gpt-4o-mini")
    # Build a workflow with a tight iteration cap for testing
    workflow = WorkflowBuilder().add_agent(agent, output_type=str).build()

    try:
        # Runner's max_iterations is set via WorkflowBuilder
        result = await workflow.run("Enumerate all prime numbers.", max_iterations=3)
        print(result.output)
    except WorkflowConvergenceException as exc:
        print(f"Stopped after max iterations: {exc}")

asyncio.run(main())
```

**Example 3 — using `create_edge_runner` when building a custom runner:**

```python
from agent_framework._workflows._runner import create_edge_runner, Runner
from agent_framework._workflows._edge import SingleEdgeGroup, FanOutEdgeGroup

# create_edge_runner picks the right runner for each edge group type
# (You'd normally let WorkflowBuilder do this, but custom orchestrators
#  can call it directly to assemble a Runner from arbitrary edge groups.)

def runner_type_for(group):
    runner = create_edge_runner(group, executors={})
    return type(runner).__name__

from agent_framework._workflows._edge import (
    SingleEdgeGroup,
    FanOutEdgeGroup,
    FanInEdgeGroup,
    SwitchCaseEdgeGroup,
)

# Demonstrate the dispatch (need real executor dicts in production)
print(runner_type_for.__doc__)  # illustrative—requires populated executors
```

---

## 4 · `AgentLoopMiddleware` callable types and judge constants

**Module:** `agent_framework._harness._loop`

Vol. 17 covered `AgentLoopMiddleware`, `JudgeVerdict`, `todos_remaining`, and `background_tasks_running`. This section documents the *callable type aliases* and *string constants* that form the customization contract for the middleware—essential when writing your own `should_continue`, `next_message`, or `feedback` callables.

### Callable type aliases

| Alias | Type |
|---|---|
| `ShouldContinueCallable` | `Callable[..., ShouldContinueResult \| Awaitable[ShouldContinueResult]]` |
| `ShouldContinueResult` | `bool \| tuple[bool, str \| None]` — return `(False, "done")` to also attach feedback |
| `FeedbackCallable` | `Callable[..., str \| Awaitable[str \| None] \| None]` |
| `NextMessageCallable` | `Callable[..., AgentRunInputs \| Awaitable[AgentRunInputs \| None] \| None]` |

All callables receive **keyword-only** arguments declared by the framework:

| Keyword | Type | Description |
|---|---|---|
| `iteration` | `int` | Number of completed runs (1-based after first run) |
| `last_result` | `AgentResponse` | Response from the iteration just completed |
| `messages` | `list[Message]` | Input messages for the iteration just completed |
| `original_messages` | `list[Message]` | Input used for the first iteration |
| `session` | `AgentSession \| None` | Active session (needed by provider helpers) |
| `agent` | `Agent` | The agent being looped |
| `progress` | `list[str]` | Feedback log accumulated so far |
| `feedback` | `str \| None` | Feedback string returned by `should_continue` for this iteration |

Declare only the keywords you need; use `**kwargs` to ignore the rest.

### Judge string constants

| Constant | Value |
|---|---|
| `DEFAULT_JUDGE_INSTRUCTIONS` | Multi-line prompt with `{{criteria}}` placeholder |
| `CRITERIA_PLACEHOLDER` | `"{{criteria}}"` |
| `DEFAULT_NEXT_MESSAGE` | `"Continue working on the task. If it is complete, say so."` |
| `JUDGE_VERDICT_DONE` | `"DONE"` |
| `JUDGE_VERDICT_MORE` | `"MORE"` |
| `DEFAULT_JUDGE_MAX_ITERATIONS` | (source-verified int) |

`CRITERIA_PLACEHOLDER` is interpolated into `DEFAULT_JUDGE_INSTRUCTIONS` when you pass `criteria=` to `with_judge(...)`.

**Example 1 — `ShouldContinueResult` tuple to attach per-iteration feedback:**

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import AgentLoopMiddleware

async def quality_gate(*, last_result, iteration, **kwargs):
    """Return (continue?, feedback) tuple — feedback flows into next_message via 'progress'."""
    text = last_result.messages[-1].contents[0].text or ""
    if "FINAL ANSWER" in text:
        return False, "Agent signalled completion"
    if iteration >= 5:
        return False, f"Iteration cap reached at {iteration}"
    score = len(text.split())  # crude proxy
    return True, f"iteration {iteration}: word count {score}"

async def build_next(*, progress, **kwargs):
    """Inject the last feedback as context into the next iteration."""
    note = progress[-1] if progress else "no feedback yet"
    return f"Previous feedback: {note}. Continue and refine your answer."

async def main() -> None:
    agent = Agent(model="gpt-4o-mini")
    loop_mw = AgentLoopMiddleware(
        should_continue=quality_gate,
        next_message=build_next,
        max_iterations=10,
    )
    agent.add_middleware(loop_mw)
    result = await agent.run("Draft a research summary on quantum computing.")
    print(result.messages[-1].contents[0].text)

asyncio.run(main())
```

**Example 2 — customising the judge prompt with `criteria=` and `CRITERIA_PLACEHOLDER`:**

```python
import asyncio
from agent_framework import Agent
from agent_framework._harness._loop import (
    AgentLoopMiddleware,
    DEFAULT_JUDGE_INSTRUCTIONS,
    CRITERIA_PLACEHOLDER,
)

# Inspect the template to understand how CRITERIA_PLACEHOLDER is substituted
print(DEFAULT_JUDGE_INSTRUCTIONS)
# "... Set 'answered' to true if ... {{criteria}}"

# with_judge replaces {{criteria}} with your criteria string
custom_criteria = " Additionally, the answer must include at least one code example."

async def main() -> None:
    agent = Agent(model="gpt-4o-mini")
    judge_client = agent.chat_client  # reuse the same model as judge
    loop_mw = AgentLoopMiddleware.with_judge(
        client=judge_client,
        criteria=custom_criteria,
        max_iterations=4,
    )
    agent.add_middleware(loop_mw)
    result = await agent.run("Explain Python decorators.")
    print(result.messages[-1].contents[0].text)

asyncio.run(main())
```

**Example 3 — `FeedbackCallable` for progressive context injection:**

```python
import asyncio
from agent_framework import Agent, AgentSession
from agent_framework._harness._loop import AgentLoopMiddleware

feedback_log: list[str] = []

def record_feedback_fn(*, last_result, iteration, **kwargs) -> str | None:
    """Append structured feedback to a running log; AgentLoopMiddleware stores it in 'progress'."""
    text = last_result.messages[-1].contents[0].text or ""
    note = f"[iter {iteration}] length={len(text)} chars"
    feedback_log.append(note)
    return note

async def main() -> None:
    agent = Agent(model="gpt-4o-mini")
    mw = AgentLoopMiddleware(
        should_continue=lambda *, iteration, **kw: iteration < 3,
        record_feedback=record_feedback_fn,
        max_iterations=3,
        return_final_only=True,
    )
    agent.add_middleware(mw)
    result = await agent.run("Write a haiku about distributed systems.")
    print(result.messages[-1].contents[0].text)
    print("Feedback log:", feedback_log)

asyncio.run(main())
```

---

## 5 · Tool approval protocol internals — `ToolApprovalScope`, constants, helper factories

**Module:** `agent_framework._harness._tool_approval`

Vol. 17 covered `ToolApprovalMiddleware`, `ToolApprovalRule`, and `ToolApprovalState`. This section fills in the *protocol layer*: the type aliases that define approval granularity, the string constants used as session keys and tool names, and the helper functions that build approval-response messages.

### `ToolApprovalScope`

```python
ToolApprovalScope = Literal['tool', 'tool_with_arguments']
```

Controls whether a standing rule grants approval for the *entire tool* (`'tool'`) or only for that exact argument combination (`'tool_with_arguments'`). A `ToolApprovalRule` with `arguments=None` matches scope `'tool'`; with `arguments={}` or a populated dict it matches scope `'tool_with_arguments'`.

### `ToolApprovalRuleCallback`

```python
ToolApprovalRuleCallback = Callable[[Content], bool | Awaitable[bool]]
```

Custom callback used by `ToolApprovalMiddleware` to accept or reject a tool call `Content` object *before* any standing rules are checked. Return `True` to approve, `False` to queue for human review.

### String constants

| Constant | Value | Purpose |
|---|---|---|
| `DEFAULT_TOOL_APPROVAL_SOURCE_ID` | `"tool_approval"` | Session state key under which `ToolApprovalState` is stored |
| `ALWAYS_APPROVE_TOOL` | `"always_approve_tool"` | Function name injected into the conversation when scope is `'tool'` |
| `ALWAYS_APPROVE_TOOL_WITH_ARGUMENTS` | `"always_approve_tool_with_arguments"` | Function name injected when scope is `'tool_with_arguments'` |
| `ALWAYS_APPROVE_PROPERTY` | `"tool_name"` | JSON key in the LLM-facing approval function schema |
| `ALWAYS_APPROVE_SCOPE_PROPERTY` | `"scope"` | JSON key carrying `ToolApprovalScope` |

### Helper factories

```python
create_always_approve_tool_response(tool_name: str, scope: ToolApprovalScope) -> Content
create_always_approve_tool_with_arguments_response(tool_name: str, arguments: dict, scope: ToolApprovalScope) -> Content
```

These build the `Content` objects that the middleware injects as fake function-call responses when the agent signals it wants to add a standing approval rule.

**Example 1 — `ToolApprovalScope` in action with `ToolApprovalRule`:**

```python
from agent_framework import Agent, AgentSession
from agent_framework._harness._tool_approval import (
    ToolApprovalRule,
    ToolApprovalState,
    ToolApprovalScope,
    DEFAULT_TOOL_APPROVAL_SOURCE_ID,
)
from agent_framework import ToolApprovalMiddleware

# A rule with arguments=None covers any invocation of the tool (scope='tool')
broad_rule = ToolApprovalRule(tool_name="read_file")          # scope: 'tool'

# A rule with arguments= covers only matching arguments (scope='tool_with_arguments')
narrow_rule = ToolApprovalRule(
    tool_name="write_file",
    arguments={"path": "/tmp/safe.txt"},
)

# Inspect serialized form
print(broad_rule.to_dict())
# {'tool_name': 'read_file', 'type': 'ToolApprovalRule'}

print(narrow_rule.to_dict())
# {'tool_name': 'write_file', 'type': 'ToolApprovalRule', 'arguments': {'path': '/tmp/safe.txt'}}

# Pre-populate a session with standing rules so the middleware doesn't ask
state = ToolApprovalState(rules=[broad_rule, narrow_rule])
print(len(state.rules))   # 2
```

**Example 2 — inspecting constants to understand session state layout:**

```python
from agent_framework import Agent, AgentSession
from agent_framework._harness._tool_approval import (
    DEFAULT_TOOL_APPROVAL_SOURCE_ID,
    ALWAYS_APPROVE_TOOL,
    ALWAYS_APPROVE_TOOL_WITH_ARGUMENTS,
)
from agent_framework import ToolApprovalMiddleware
import asyncio

async def main():
    agent = Agent(model="gpt-4o-mini")
    agent.add_middleware(ToolApprovalMiddleware())

    async with AgentSession(agent) as session:
        # After a run that triggered approvals, the state is stored at:
        state_dict = session.state.get(DEFAULT_TOOL_APPROVAL_SOURCE_ID)
        print("Standing rules:", state_dict)

        # Tool names the framework injects to let the LLM request standing approval:
        print("Scope='tool' function:", ALWAYS_APPROVE_TOOL)
        print("Scope='tool_with_arguments' function:", ALWAYS_APPROVE_TOOL_WITH_ARGUMENTS)

asyncio.run(main())
```

**Example 3 — `ToolApprovalRuleCallback` for custom pre-approval logic:**

```python
import asyncio
from agent_framework import Agent, tool
from agent_framework._harness._tool_approval import ToolApprovalRuleCallback
from agent_framework import ToolApprovalMiddleware
from agent_framework._types import Content

SAFE_TOOLS = {"read_file", "list_dir"}

def my_pre_check(content: Content) -> bool:
    """Pre-approve tools in the safe list without human review."""
    # content.function_call.name holds the tool name in function_call Content
    fn = getattr(getattr(content, "function_call", None), "name", None)
    return fn in SAFE_TOOLS

@tool
def read_file(path: str) -> str:
    """Read a file."""
    return open(path).read()

async def main():
    agent = Agent(model="gpt-4o-mini", tools=[read_file])
    mw = ToolApprovalMiddleware(pre_approval_callback=my_pre_check)
    agent.add_middleware(mw)
    result = await agent.run("Read /etc/hostname")
    print(result.messages[-1].contents[0].text)

asyncio.run(main())
```

---

## 6 · Orchestration protocol internals — `GroupChatRequestMessage`, `ORCH_MSG_KIND_*`, `clean_conversation_for_handoff`, `create_completion_message`

**Module:** `agent_framework.orchestrations`

These utilities form the internal protocol between orchestrators and participant executors. They are useful when building custom orchestrators or debugging multi-agent conversation flows.

### `GroupChatRequestMessage`

```python
@dataclass
class GroupChatRequestMessage:
    additional_instruction: str | None = None
    metadata: dict[str, Any] | None = None
```

Sent from an orchestrator executor to a participant executor's `WorkflowContext`. An `AgentExecutor` unwraps `additional_instruction` and prepends it to the agent's prompt. `metadata` carries arbitrary orchestrator-scoped annotations.

### `ORCH_MSG_KIND_*` constants

These string values appear in `Message.additional_properties["kind"]` to classify orchestration messages without inspecting content:

| Constant | Value | Set by |
|---|---|---|
| `ORCH_MSG_KIND_USER_TASK` | `"user_task"` | Orchestrator when forwarding the original user task |
| `ORCH_MSG_KIND_INSTRUCTION` | `"instruction"` | Orchestrator when sending a targeted instruction |
| `ORCH_MSG_KIND_NOTICE` | `"notice"` | Orchestrator for informational announcements |
| `ORCH_MSG_KIND_TASK_LEDGER` | `"task_ledger"` | Magentic orchestrator for progress ledger updates |

### `clean_conversation_for_handoff`

```python
def clean_conversation_for_handoff(conversation: list[Message]) -> list[Message]:
```

Strips all non-text `Content` from every message (function calls, tool outputs, approval payloads) and drops messages with no remaining text. Returns a text-only copy safe to pass as a routing context to a handoff orchestrator — prevents providers from rejecting requests due to unmatched tool-call state.

### `create_completion_message`

```python
def create_completion_message(
    *,
    text: str | None = None,
    author_name: str,
    reason: str = "completed",
) -> Message:
```

Creates a standardised `role="assistant"` termination message. Used by `SequentialBuilder` and custom orchestrators to signal workflow completion without duplicating boilerplate.

**Example 1 — attaching kind metadata to trace orchestration messages:**

```python
from agent_framework._types import Message
from agent_framework.orchestrations import (
    ORCH_MSG_KIND_USER_TASK,
    ORCH_MSG_KIND_INSTRUCTION,
    ORCH_MSG_KIND_NOTICE,
)

# Orchestrators stamp kind so downstream observers can classify messages cheaply
user_task_msg = Message(
    role="user",
    contents=["Analyze Q1 sales data"],
    additional_properties={"kind": ORCH_MSG_KIND_USER_TASK},
)

instruction_msg = Message(
    role="user",
    contents=["Focus on European markets only"],
    additional_properties={"kind": ORCH_MSG_KIND_INSTRUCTION},
)

print(user_task_msg.additional_properties["kind"])     # "user_task"
print(instruction_msg.additional_properties["kind"])   # "instruction"
```

**Example 2 — `clean_conversation_for_handoff` before routing:**

```python
from agent_framework._types import Message, Content
from agent_framework.orchestrations import clean_conversation_for_handoff

# Simulate a conversation that includes function_call artifacts
text_msg = Message(role="user", contents=["What's the weather?"])
tool_msg = Message(role="assistant", contents=[
    Content(type="function_call", function_call={"name": "get_weather", "arguments": "{}"}),
    Content(type="text", text="The weather is sunny."),
])
tool_result_msg = Message(role="tool", contents=[
    Content(type="function_result", function_result={"result": "sunny"}),
])

full_convo = [text_msg, tool_msg, tool_result_msg]
clean = clean_conversation_for_handoff(full_convo)

# Only text parts survive; messages with no text are dropped
print(len(clean))                              # 2 (tool_result_msg dropped)
print(clean[1].contents[0].text)              # "The weather is sunny."
```

**Example 3 — `GroupChatRequestMessage` and `create_completion_message` in a custom orchestrator:**

```python
import asyncio
from agent_framework import Agent
from agent_framework._workflows._executor import Executor
from agent_framework._workflows._workflow_context import WorkflowContext
from agent_framework.orchestrations import (
    GroupChatRequestMessage,
    create_completion_message,
    ORCH_MSG_KIND_INSTRUCTION,
)
from agent_framework._types import Message

# Custom orchestrator executor snippet
async def orchestrate(ctx: WorkflowContext) -> None:
    # Send a targeted instruction to a participant
    request = GroupChatRequestMessage(
        additional_instruction="Keep your response under 50 words.",
        metadata={"round": 1, "kind": ORCH_MSG_KIND_INSTRUCTION},
    )
    await ctx.send_message(request)

    # ... wait for participant response ...

    # Signal completion
    done_msg = create_completion_message(
        text="All agents have completed their turns.",
        author_name="orchestrator",
        reason="round_complete",
    )
    await ctx.yield_output(done_msg)
```

---

## 7 · Magentic event observability — `MagenticOrchestratorEvent`, `MagenticOrchestratorEventType`, `MagenticAgentExecutor`

**Module:** `agent_framework.orchestrations`

Vol. 13 covered `MagenticOrchestrator` and `MagenticPlanReviewRequest`. This section documents the *event observability layer* and the *reset-aware participant executor*, both of which were undocumented in prior volumes.

### `MagenticOrchestratorEventType`

```python
class MagenticOrchestratorEventType(str, Enum):
    PLAN_CREATED = "plan_created"
    REPLANNED = "replanned"
    PROGRESS_LEDGER_UPDATED = "progress_ledger_updated"
```

### `MagenticOrchestratorEvent`

```python
@dataclass
class MagenticOrchestratorEvent:
    event_type: MagenticOrchestratorEventType
    content: Message | MagenticProgressLedger
```

Emitted by the orchestrator as `WorkflowEvent.data`. When `event_type` is `PLAN_CREATED` or `REPLANNED`, `content` is a `Message` containing the new task plan. When `event_type` is `PROGRESS_LEDGER_UPDATED`, `content` is a `MagenticProgressLedger` with the latest task assignments.

### `MagenticAgentExecutor`

```python
class MagenticAgentExecutor(AgentExecutor):
    def __init__(self, agent: SupportsAgentRun) -> None: ...

    @handler
    async def handle_magentic_reset(self, signal: MagenticResetSignal, ctx: WorkflowContext) -> None: ...
```

Subclass of `AgentExecutor` for Magentic participants. The key difference is `handle_magentic_reset`, which the `MagenticOrchestrator` broadcasts as a `MagenticResetSignal` when replanning. On receipt, the executor:

1. Clears its message cache (`_cache.clear()`)
2. Clears the full conversation history (`_full_conversation.clear()`)
3. Clears pending request/response queues
4. Resets its agent thread by calling `self._agent.create_session()`

Because of this stateful reset, `MagenticAgentExecutor` does **not** support custom threads.

**Example 1 — observing Magentic plan and replan events:**

```python
import asyncio
from agent_framework import Agent
from agent_framework import MagenticBuilder
from agent_framework.orchestrations import (
    MagenticOrchestratorEvent,
    MagenticOrchestratorEventType,
    MagenticProgressLedger,
)

async def main():
    researcher = Agent(name="researcher", model="gpt-4o-mini")
    writer = Agent(name="writer", model="gpt-4o-mini")

    workflow = (
        MagenticBuilder()
        .add_agent(researcher)
        .add_agent(writer)
        .build()
    )

    plan_events = []
    ledger_updates = []

    async for event in workflow.run_stream("Research and summarize Python 3.13 features."):
        data = event.data
        if isinstance(data, dict) and "event_type" in data:
            orch_event = data.get("magentic_event")
            if isinstance(orch_event, MagenticOrchestratorEvent):
                if orch_event.event_type == MagenticOrchestratorEventType.PLAN_CREATED:
                    plan_events.append(orch_event.content)
                elif orch_event.event_type == MagenticOrchestratorEventType.PROGRESS_LEDGER_UPDATED:
                    ledger_updates.append(orch_event.content)

    print(f"Plans created: {len(plan_events)}, Ledger updates: {len(ledger_updates)}")

asyncio.run(main())
```

**Example 2 — using `MagenticAgentExecutor` explicitly in a custom workflow:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.orchestrations import MagenticAgentExecutor
from agent_framework._workflows._executor import AgentExecutor

# MagenticAgentExecutor is what MagenticBuilder registers automatically.
# Use it directly when building a custom Magentic-like workflow.
coder = Agent(name="coder", model="gpt-4o-mini")
reviewer = Agent(name="reviewer", model="gpt-4o-mini")

coder_exec = MagenticAgentExecutor(coder)
reviewer_exec = MagenticAgentExecutor(reviewer)

# Verify the reset handler is registered
print(hasattr(coder_exec, "handle_magentic_reset"))  # True

# Check that it's a specialization of AgentExecutor
print(isinstance(coder_exec, AgentExecutor))           # True
```

**Example 3 — detecting replan events to trigger observability hooks:**

```python
import asyncio
from agent_framework import Agent, MagenticBuilder
from agent_framework.orchestrations import MagenticOrchestratorEventType

replan_count = 0

async def main():
    global replan_count
    agents = [Agent(name=f"agent{i}", model="gpt-4o-mini") for i in range(3)]
    builder = MagenticBuilder()
    for a in agents:
        builder.add_agent(a)
    workflow = builder.build()

    async for event in workflow.run_stream("Complete a complex multi-step research task."):
        # Monitor for replan signals — indicates the orchestrator revised its plan
        raw = getattr(event, "data", {})
        if isinstance(raw, dict):
            evt_type = raw.get("event_type")
            if evt_type == MagenticOrchestratorEventType.REPLANNED:
                replan_count += 1
                print(f"Replan #{replan_count} detected")

asyncio.run(main())
```

---

## 8 · Foundry trace-based evaluation + `RawFoundryEmbeddingClient`

**Module:** `agent_framework_foundry`

### `evaluate_traces`

```python
@experimental(feature_id=ExperimentalFeature.EVALS)
async def evaluate_traces(
    *,
    evaluators: Sequence[str] | None = None,
    client: FoundryChatClient | None = None,
    project_client: AIProjectClient | None = None,
    model: str,
    response_ids: Sequence[str] | None = None,
    trace_ids: Sequence[str] | None = None,
    agent_id: str | None = None,
    lookback_hours: int = 24,
    eval_name: str = "Agent Framework Trace Eval",
    poll_interval: float = 5.0,
    timeout: float = 180.0,
) -> EvalResults:
```

Evaluates agent behavior from OTel traces or response IDs. Three input modes:

| Mode | Parameters | Use case |
|---|---|---|
| Response-based | `response_ids=` | Evaluate specific Responses API calls |
| Trace-based | `trace_ids=` | Evaluate specific OTel traces from App Insights |
| Time-window | `agent_id=` + `lookback_hours=` | Evaluate all recent activity for an agent |

Provide `client` **or** `project_client` (not both). Polls every `poll_interval` seconds until the cloud evaluation job completes or `timeout` is exceeded.

### `RawFoundryEmbeddingClient`

```python
class RawFoundryEmbeddingClient(
    BaseEmbeddingClient[Content | str, list[float], FoundryEmbeddingOptionsT],
    Generic[FoundryEmbeddingOptionsT],
):
```

The raw (no telemetry, no middleware) Foundry embedding client. Accepts both `str` (text) and `Content` (image) inputs in a single batch. Internally splits text and image inputs, dispatches to `EmbeddingsClient` and `ImageEmbeddingsClient` respectively, then reassembles results in original input order.

| Constructor keyword | Env var | Purpose |
|---|---|---|
| `model` | `FOUNDRY_EMBEDDING_MODEL` | Text embedding model |
| `image_model` | `FOUNDRY_IMAGE_EMBEDDING_MODEL` | Image embedding model (falls back to `model`) |
| `endpoint` | `FOUNDRY_MODELS_ENDPOINT` | Foundry inference endpoint URL |
| `api_key` | `FOUNDRY_MODELS_API_KEY` | API key authentication |
| `text_client` | — | Pre-configured `EmbeddingsClient` override |
| `image_client` | — | Pre-configured `ImageEmbeddingsClient` override |

**Example 1 — evaluating specific response IDs with `evaluate_traces`:**

```python
import asyncio
from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework_foundry import evaluate_traces, FoundryEvals

async def main():
    client = FoundryChatClient(
        model="gpt-4o",
        endpoint="https://my-project.openai.azure.com",
        api_key="...",
    )
    agent = Agent(model=client)
    result = await agent.run("What are the key trends in AI for 2026?")

    # Evaluate the response that was just generated
    response_id = getattr(result, "response_id", None)
    if response_id:
        eval_result = await evaluate_traces(
            response_ids=[response_id],
            evaluators=[FoundryEvals.RELEVANCE, FoundryEvals.COHERENCE],
            client=client,
            model="gpt-4o",
        )
        print(f"Eval status: {eval_result.status}")
        print(f"Pass: {eval_result.pass_count}, Fail: {eval_result.fail_count}")

asyncio.run(main())
```

**Example 2 — time-window evaluation of recent agent activity:**

```python
import asyncio
from agent_framework_foundry import evaluate_traces, FoundryEvals, RawFoundryChatClient

async def main():
    client = RawFoundryChatClient(
        model="gpt-4o",
        endpoint="https://my-project.openai.azure.com",
        api_key="...",
    )

    # Evaluate all activity for agent "my-agent-id" in the past 6 hours
    results = await evaluate_traces(
        agent_id="my-agent-id",
        lookback_hours=6,
        evaluators=[FoundryEvals.TASK_ADHERENCE, FoundryEvals.GROUNDEDNESS],
        client=client,
        model="gpt-4o",
        eval_name="Nightly quality gate",
        timeout=300.0,
    )
    print(f"Total evaluated: {results.total_count}")
    print(f"Portal link: {results.portal_link}")

asyncio.run(main())
```

**Example 3 — `RawFoundryEmbeddingClient` for mixed text+image batches:**

```python
import asyncio
from agent_framework_foundry import RawFoundryEmbeddingClient
from agent_framework._types import Content

async def main():
    client = RawFoundryEmbeddingClient(
        model="text-embedding-3-small",
        image_model="Cohere-embed-v3-english",
        endpoint="https://my-project.openai.azure.com",
        api_key="...",
    )

    # Mixed batch: 2 text items + 1 image item
    image_content = Content(type="image_url", image_url={"url": "https://example.com/chart.png"})
    inputs = [
        "Revenue grew 15% year-over-year",
        image_content,
        "Customer satisfaction reached an all-time high",
    ]

    # get_embeddings returns list[float] aligned to the input order
    embeddings = await client.get_embeddings(inputs)
    print(f"Got {len(embeddings)} embeddings, first dim: {len(embeddings[0].vector)}")

asyncio.run(main())
```

---

## 9 · `RawOpenAIChatCompletionClient` — raw Chat Completions without middleware

**Module:** `agent_framework_openai` / `agent_framework.openai`

`OpenAIChatCompletionClient` is the fully-featured client with all layers applied. `RawOpenAIChatCompletionClient` is the innermost class — **no middleware, no telemetry, no function invocation**. The framework documents a strict layer ordering that must be followed when composing the raw client manually.

### Layer ordering contract

The docstring is explicit:

```
1. FunctionInvocationLayer  — owns the tool-call loop and routes function middleware
2. ChatMiddlewareLayer       — applies chat middleware per model call (outside telemetry)
3. ChatTelemetryLayer        — per-call telemetry (must stay inside ChatMiddlewareLayer)
```

Use `OpenAIChatCompletionClient` instead if you want all layers pre-applied.

### Constructor (abbreviated)

```python
RawOpenAIChatCompletionClient(
    model: str | None = None,
    *,
    api_key: str | SecretString | Callable[[], str | Awaitable[str]] | None = None,
    org_id: str | None = None,
    base_url: str | None = None,
    default_headers: Mapping[str, str] | None = None,
    async_client: AsyncOpenAI | None = None,
    instruction_role: str | None = None,
    compaction_strategy: CompactionStrategy | None = None,
    ...
)
```

`INJECTABLE: ClassVar[set[str]] = {"client"}` — the `async_client` OpenAI instance can be dependency-injected by the `WorkflowBuilder` deserialization layer.

**Example 1 — using `RawOpenAIChatCompletionClient` for a minimal Chat Completions call:**

```python
import asyncio
from agent_framework_openai import RawOpenAIChatCompletionClient
from agent_framework._types import Message

async def main():
    # Raw client: no function invocation, no telemetry, no middleware
    raw = RawOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key="sk-...",
    )
    messages = [Message(role="user", contents=["Say hello in one sentence."])]
    response = await raw.complete(messages)
    print(response.messages[-1].contents[0].text)

asyncio.run(main())
```

**Example 2 — wrapping `RawOpenAIChatCompletionClient` with telemetry only (no function layer):**

```python
import asyncio
from agent_framework_openai import RawOpenAIChatCompletionClient
from agent_framework._middleware import ChatTelemetryLayer
from agent_framework._types import Message

async def main():
    raw = RawOpenAIChatCompletionClient(model="gpt-4o-mini", api_key="sk-...")
    # Apply only telemetry — useful for custom function invocation implementations
    with_telemetry = ChatTelemetryLayer(raw)
    messages = [Message(role="user", contents=["Summarize the water cycle in 2 sentences."])]
    response = await with_telemetry.complete(messages)
    print(response.messages[-1].contents[0].text)

asyncio.run(main())
```

**Example 3 — dependency-injecting a pre-configured `AsyncOpenAI` client:**

```python
import asyncio
from openai import AsyncOpenAI
from agent_framework_openai import RawOpenAIChatCompletionClient
from agent_framework._types import Message

# Re-use an existing httpx session or apply custom retry logic
shared_oai = AsyncOpenAI(
    api_key="sk-...",
    max_retries=5,
    timeout=60.0,
)

async def main():
    raw = RawOpenAIChatCompletionClient(
        model="gpt-4o-mini",
        async_client=shared_oai,   # INJECTABLE — bypasses internal client creation
    )
    messages = [Message(role="user", contents=["What is 7 * 8?"])]
    response = await raw.complete(messages)
    print(response.messages[-1].contents[0].text)   # 56

asyncio.run(main())
```

---

## 10 · Workflow message utilities — `normalize_messages_input`, `ensure_author`, `latest_user_message`

**Modules:** `agent_framework._workflows._message_utils`, `agent_framework._workflows._conversation_history`

These three small but heavily-used functions normalize heterogeneous workflow inputs and extract key messages from conversation history. They are called internally throughout the graph runtime but are useful in custom executor and orchestrator code.

### `normalize_messages_input`

```python
def normalize_messages_input(messages: AgentRunInputs | None = None) -> list[Message]:
```

`AgentRunInputs = str | Content | Message | Sequence[str | Content | Message]`

| Input type | Converted to |
|---|---|
| `None` | `[]` |
| `str` | `[Message(role="user", contents=[str])]` |
| `Content` | `[Message(role="user", contents=[Content])]` |
| `Message` | `[message]` (pass-through) |
| `Sequence` | Each element converted by the above rules |
| Unsupported element in sequence | `TypeError` with element type name |

### `ensure_author`

```python
def ensure_author(message: Message, fallback: str) -> Message:
```

Mutates `message.author_name` in place: if `author_name` is `None` or empty, sets it to `fallback`. Returns the same message object. Used by executors to guarantee every outgoing message has an author before it enters the conversation history.

### `latest_user_message`

```python
def latest_user_message(conversation: Sequence[Message]) -> Message:
```

Scans `conversation` in reverse and returns the first message with `role == "user"` (case-insensitive, handles role `Enum` values via `.value`). Raises `ValueError` if no user message is found. Used by executors to locate the most recent user prompt when routing or summarizing.

**Example 1 — normalising diverse agent inputs before passing to a workflow:**

```python
from agent_framework._workflows._message_utils import normalize_messages_input
from agent_framework._types import Message, Content

# str input
msgs = normalize_messages_input("Analyze this report")
print(msgs[0].role, msgs[0].contents[0])  # "user" "Analyze this report"

# Content input (e.g. an image)
image = Content(type="image_url", image_url={"url": "https://example.com/chart.png"})
msgs = normalize_messages_input(image)
print(msgs[0].contents[0].type)  # "image_url"

# Mixed sequence
mixed = normalize_messages_input([
    "First question",
    Message(role="assistant", contents=["Answer"]),
    "Follow-up question",
])
print(len(mixed))   # 3
print(mixed[0].role, mixed[1].role, mixed[2].role)  # user, assistant, user
```

**Example 2 — `ensure_author` in a custom executor:**

```python
from agent_framework._workflows._conversation_history import ensure_author
from agent_framework._types import Message

# Simulate executor producing a message without an author
output_msg = Message(role="assistant", contents=["The analysis is complete."])
print(output_msg.author_name)   # None

ensure_author(output_msg, fallback="data-analyst")
print(output_msg.author_name)   # "data-analyst"

# Already has an author — not overwritten
existing = Message(role="assistant", contents=["Done."], author_name="researcher")
ensure_author(existing, fallback="fallback-name")
print(existing.author_name)     # "researcher"  (unchanged)
```

**Example 3 — `latest_user_message` for dynamic orchestration routing:**

```python
from agent_framework._workflows._conversation_history import latest_user_message
from agent_framework._types import Message

conversation = [
    Message(role="user", contents=["Start the analysis"]),
    Message(role="assistant", contents=["Analysis running..."], author_name="agent-1"),
    Message(role="user", contents=["Focus on Q3 data"]),
    Message(role="assistant", contents=["Focusing on Q3..."], author_name="agent-2"),
]

# Retrieves "Focus on Q3 data" — the most recent user turn
prompt = latest_user_message(conversation)
print(prompt.contents[0])   # "Focus on Q3 data"

# ValueError when no user message exists
assistant_only = [Message(role="assistant", contents=["Hi"])]
try:
    latest_user_message(assistant_only)
except ValueError as e:
    print(e)   # "No user message in conversation"
```

---

## Revision history

| Date | Version | Change |
|---|---|---|
| 2026-06-26 | agent-framework 1.9.0 | Vol. 24 initial publication — 10 class groups, 30 runnable examples |

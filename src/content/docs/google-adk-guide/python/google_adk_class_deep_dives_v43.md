---
title: "Class deep dives — volume 43 (run_llm_agent_as_node, prepare_llm_agent_context/input/output, chat-mode task dispatch loop, task-mode FinishTask handshake, _should_retry_node/_get_retry_delay, RetryConfig, graph validation suite, build_node, HITL workflow utilities, resolve_and_derive_transfer_context)"
description: "10 source-verified deep dives for google-adk 2.4.0: run_llm_agent_as_node async generator (single_turn/chat/task modes), prepare_llm_agent_context+input+output helpers, chat-mode outer dispatch loop with task-delegation FC chaining, task-mode FinishTask FC/FR handshake, _should_retry_node+_get_retry_delay backoff internals, RetryConfig Pydantic model, graph validation suite (cycle detection, connectivity, duplicate checks, DEFAULT_ROUTE, schema mismatch, chat-wiring), build_node NodeLike→BaseNode converter, HITL utilities (create_request_input_event, process_auth_resume), resolve_and_derive_transfer_context context routing."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 43"
  order: 112
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.4.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `run_llm_agent_as_node` — running `LlmAgent` as a workflow node

**Source:** `google/adk/workflow/_llm_agent_wrapper.py`

`run_llm_agent_as_node` is the async generator that the workflow engine calls
whenever an `LlmAgent` sits inside a `Workflow` graph. It handles three
distinct execution modes (`single_turn`, `chat`, `task`) and enforces the
invariants each mode requires.

### Signature

```python
async def run_llm_agent_as_node(
    agent: Any,
    *,
    ctx: Context,
    node_input: Any,
) -> AsyncGenerator[Any, None]:
```

### Mode selection and defaults

When an `LlmAgent` is added to a `Workflow` graph via `build_node`, its `mode`
is set to `'single_turn'` for standalone agents or `'chat'` for sub-agents
attached to a parent. `run_llm_agent_as_node` then applies two automatic
defaults before delegating:

```python
# mode defaults to 'single_turn' when the agent hasn't set it
if agent.mode is None:
    agent.mode = 'single_turn'

# single_turn nodes automatically ignore session history unless the
# agent explicitly set include_contents
include_contents_explicit = 'include_contents' in agent.model_fields_set
if agent.mode == 'single_turn' and not include_contents_explicit:
    agent.include_contents = 'none'
```

The `include_contents='none'` default is crucial: it prevents the agent from
seeing the entire session history, keeping workflow nodes stateless by default.

### Mode guard

Only `'single_turn'`, `'chat'`, and `'task'` are accepted. Any other value
raises immediately:

```python
if agent.mode not in ('task', 'single_turn', 'chat'):
    raise ValueError(
        f"LlmAgent as node only supports task, single_turn, and chat mode,"
        f" but agent '{agent.name}' has mode='{agent.mode}'."
    )
```

### `single_turn` path

The simplest path — one `run_async` invocation, output extracted on every
event, then return:

```python
if agent.mode == 'single_turn':
    async with aclosing(agent.run_async(ic)) as run_iter:
        async for event in run_iter:
            process_llm_agent_output(agent, ctx, event)
            yield event
    return
```

`aclosing` ensures the generator is properly finalised even if the caller
breaks out of the outer loop early.

### Full working example — `single_turn` node

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the text in the node_input in one sentence.",
    # mode is not set here; build_node will default it to 'single_turn'
    output_key="summary",
)

# Nodes are declared through the edges list passed to the Workflow constructor.
# There is no END sentinel — nodes with no outgoing edges are terminal automatically.
workflow = Workflow(
    name="summarise_wf",
    edges=[(START, summariser)],
)

async def main():
    session_svc = InMemorySessionService()
    session = await session_svc.create_session(app_name="demo", user_id="u1")
    runner = Runner(agent=workflow, app_name="demo", session_service=session_svc)
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Long document text goes here…")],
        ),
    ):
        if event.output:
            print("Summary:", event.output)

asyncio.run(main())
```

---

## 2 · `prepare_llm_agent_context` / `prepare_llm_agent_input` / `process_llm_agent_output`

**Source:** `google/adk/workflow/_llm_agent_wrapper.py`

These three helpers encapsulate the before/after bookkeeping for each
`run_llm_agent_as_node` invocation.

### `prepare_llm_agent_context`

For `single_turn` mode, creates a shallow-copied `InvocationContext` so that
events emitted by the agent are scoped to its `isolation_scope` without
polluting the parent context. For `chat` and `task` modes the original context
is returned unchanged.

```python
def prepare_llm_agent_context(agent: Any, ctx: Context) -> Context:
    if agent.mode != 'single_turn':
        return ctx

    ic = ctx._invocation_context.model_copy()
    ic._event_queue = ctx._invocation_context._event_queue
    ic.isolation_scope = ctx.isolation_scope
    agent_ctx = Context(
        invocation_context=ic,
        node_path=ctx.node_path,
        run_id=ctx.run_id,
        resume_inputs=ctx.resume_inputs,
    )
    agent_ctx.isolation_scope = ctx.isolation_scope
    ic.session = ic.session.model_copy(deep=False)
    return agent_ctx
```

Key detail: `ic.session.model_copy(deep=False)` gives the agent its own
session reference without performing a deep copy of all events — this keeps
memory usage manageable while still allowing isolation.

### `prepare_llm_agent_input`

Converts `node_input` to a `Content` object and appends it as a user event:

```python
def prepare_llm_agent_input(agent: Any, ctx: Context, node_input: Any) -> None:
    if node_input is None or agent.mode != 'single_turn':
        return
    agent_input = _node_input_to_content(node_input)
    user_event = Event(author='user', message=agent_input)
    if user_event.content is not None:
        user_event.content.role = 'user'
    # Stamp isolation_scope and branch if set
    iso = getattr(ctx, 'isolation_scope', None)
    if iso:
        user_event.isolation_scope = iso
    branch = ctx._invocation_context.branch
    if branch:
        user_event.branch = branch
    ctx.session.events.append(user_event)
```

For `task` mode, the input is NOT appended here; instead
`ic.user_content` is set as a fallback for the content builder.

### `_node_input_to_content` — type coercion

`node_input` can be nearly anything; the helper normalises it to
`types.Content`:

```python
def _node_input_to_content(node_input: Any) -> types.Content:
    if isinstance(node_input, types.Content):
        return types.Content(role='user', parts=node_input.parts)
    if isinstance(node_input, str):
        text = node_input
    elif isinstance(node_input, BaseModel):
        text = node_input.model_dump_json()
    elif isinstance(node_input, (dict, list)):
        text = json.dumps(node_input)
    else:
        text = str(node_input)
    return types.Content(role='user', parts=[types.Part(text=text)])
```

### `process_llm_agent_output`

Called on every streamed event; extracts the final text, validates it against
`output_schema` if set, writes to `output_key` in state, and marks the event:

```python
def process_llm_agent_output(agent: Any, ctx: Context, event: Event) -> None:
    # Skip partial / FC events — only process complete model-role messages
    if (
        event.get_function_calls()
        or event.partial
        or not event.content
        or event.content.role != 'model'
    ):
        return

    text = (
        ''.join(p.text for p in event.content.parts if p.text and not p.thought)
        if event.content.parts
        else ''
    )
    if agent.output_schema:
        output = validate_schema(agent.output_schema, text) if text.strip() else None
    else:
        output = text

    if agent.output_key and output is not None:
        ctx.actions.state_delta[agent.output_key] = output

    event.output = output
    event.node_info.message_as_output = True
```

`p.thought` parts (chain-of-thought) are excluded from the concatenated text,
ensuring only the agent's final response is promoted as output.

### Using `output_schema` with a workflow node

```python
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START

class SentimentResult(BaseModel):
    sentiment: str        # "positive" | "neutral" | "negative"
    confidence: float

analyser = LlmAgent(
    name="sentiment",
    model="gemini-2.5-flash",
    instruction=(
        "Classify the sentiment of the text in node_input. "
        "Return JSON with 'sentiment' and 'confidence' (0.0-1.0)."
    ),
    output_schema=SentimentResult,
    output_key="sentiment_result",
)

wf = Workflow(
    name="sentiment_wf",
    edges=[(START, analyser)],
)
# process_llm_agent_output will call validate_schema and write the
# parsed SentimentResult into ctx.state["sentiment_result"]
```

---

## 3 · Chat-mode task-delegation dispatch loop

**Source:** `google/adk/workflow/_llm_agent_wrapper.py`

When `agent.mode == 'chat'`, `run_llm_agent_as_node` runs an outer `while
True` dispatch loop instead of a single `run_async` call. This enables a
coordinator agent to sequentially delegate tasks to sub-agents across multiple
LLM rounds within the same workflow invocation.

### The three phases

```
Phase 1 (resume scan)
  └─ _find_unresolved_task_delegations()
       Walk session events; find FCs from owner without matching FRs.
       Dispatch each via _dispatch_task_fc → yield synthesized FR event.

Phase 2 (live dispatch loop)
  └─ while True:
       agent.run_async(ic)
         for each event:
           if event has task-delegation FC:
             _dispatch_task_fc() → yield synthesized FR
             break (close this run_async, re-enter loop)
           if event has transfer_to_agent:
             set_agent_state(end_of_agent=True)
             break
       if no task FC was found → return (LLM finished)
```

### `_find_unresolved_task_delegations`

Walks all session events attributed to `owner` and the `'user'` author,
collecting FCs from `_TaskAgentTool` instances. Any FC without a matching FR
by ID is considered unresolved:

```python
def _find_unresolved_task_delegations(
    session, owner: str, tools_dict: dict
) -> list[types.FunctionCall]:
    fc_by_id: dict[str, types.FunctionCall] = {}
    fr_ids: set[str] = set()
    for event in session.events:
        if event.author != owner and event.author != 'user':
            continue
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            fc = part.function_call
            if fc and fc.id and fc.name in tools_dict and isinstance(tools_dict[fc.name], _TaskAgentTool):
                fc_by_id[fc.id] = fc
            fr = part.function_response
            if fr and fr.id:
                fr_ids.add(fr.id)
    return [fc for fc_id, fc in fc_by_id.items() if fc_id not in fr_ids]
```

Note: `isolation_scope` is deliberately NOT used as a filter. A chat
coordinator's conversation persists across user turns; each turn gets its own
`wf:<user_event_id>` scope, so filtering by scope would hide the
coordinator's own FC from a prior turn.

### `_dispatch_task_fc`

Finds the target agent by name and runs it as a workflow node. `run_id=fc.id`
makes the child run idempotent across resumes:

```python
async def _dispatch_task_fc(parent_agent, fc, ctx) -> Any:
    target_agent = parent_agent.root_agent.find_agent(fc.name)
    if target_agent is None:
        raise ValueError(f'Task target agent {fc.name!r} not found.')
    wrapped_target = build_node(target_agent)
    wrapped_target.parent_agent = target_agent.parent_agent
    return await ctx.run_node(
        wrapped_target,
        node_input=fc.args,
        run_id=fc.id,
        override_isolation_scope=fc.id,
        raise_on_wait=True,
    )
```

### `_synthesize_task_fr_event`

Builds the synthesized `FunctionResponse` event that feeds the result back to
the coordinator's next LLM round:

```python
def _synthesize_task_fr_event(fc, output) -> Event:
    if isinstance(output, dict):
        response = output
    else:
        response = {'output': output}
    fr_part = types.Part(
        function_response=types.FunctionResponse(
            id=fc.id,
            name=fc.name,
            response=response,
        )
    )
    return Event(author='user', content=types.Content(role='user', parts=[fr_part]))
```

### Full example — chat coordinator with sequential task delegation

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

# Two specialist sub-agents
researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the topic provided and return key facts.",
    output_key="research_notes",
)
writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Write a short article based on the research_notes in state.",
    output_key="article",
)

# Coordinator delegates to researcher first, then writer
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-pro",
    mode="chat",   # enables the outer dispatch loop
    instruction=(
        "You coordinate research and writing. First call 'researcher' with "
        "the user's topic, then call 'writer' to produce the final article."
    ),
    tools=[researcher, writer],  # _TaskAgentTool wrappers are created automatically
)

wf = Workflow(
    name="research_wf",
    edges=[(START, coordinator)],
)

async def main():
    svc = InMemorySessionService()
    session = await svc.create_session(app_name="demo", user_id="u1")
    runner = Runner(agent=wf, app_name="demo", session_service=svc)
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="quantum computing")]),
    ):
        if hasattr(event, 'output') and event.output:
            print("Output:", event.output)

asyncio.run(main())
```

---

## 4 · Task-mode `FinishTask` FC/FR handshake

**Source:** `google/adk/workflow/_llm_agent_wrapper.py`

In `task` mode the LLM calls a special `finish_task` function when its
structured output is ready. The wrapper waits for `FinishTaskTool`'s
`FunctionResponse` before promoting the result — this allows the tool to
validate the output and give the LLM another chance on schema failure.

### The four helpers

```python
# Returns the finish_task FunctionCall from an event, or None
def _extract_finish_task_fc(event: Event) -> Optional[types.FunctionCall]:
    for fc in event.get_function_calls():
        if fc.name == _FINISH_TASK_FC_NAME:
            return fc
    return None

# True only when FinishTaskTool signals success (not a validation error)
def _is_finish_task_success_fr(event: Event) -> bool:
    for fr in event.get_function_responses():
        if fr.name == _FINISH_TASK_FC_NAME:
            response = fr.response or {}
            return response.get('result') == FINISH_TASK_SUCCESS_RESULT
    return False

# Finds the FinishTaskTool instance on the agent (to read _wrapper_key)
def _find_finish_task_tool(agent: Any) -> Any:
    for tool in getattr(agent, 'tools', []) or []:
        if getattr(tool, 'name', None) == _FINISH_TASK_FC_NAME:
            return tool
    return None
```

### Task-mode event loop

```python
finish_tool = _find_finish_task_tool(agent)
pending_fc_args: Optional[dict] = None

async with aclosing(run_method) as run_iter:
    async for event in run_iter:
        finish_fc = _extract_finish_task_fc(event)
        if finish_fc is not None:
            # Cache FC args; wait for success FR before emitting output
            pending_fc_args = dict(finish_fc.args or {})
            yield event
            continue

        if pending_fc_args is not None and _is_finish_task_success_fr(event):
            # Extract value: unwrap primitive wrapper_key if present
            wrapper_key = getattr(finish_tool, '_wrapper_key', None)
            if wrapper_key and wrapper_key in pending_fc_args:
                event.output = pending_fc_args[wrapper_key]
            else:
                event.output = pending_fc_args
            yield event
            return   # task complete

        yield event
```

The `_wrapper_key` mechanism handles primitive output types. When
`output_schema` is `str`, `int`, `float`, or `bool`, `FinishTaskTool` wraps
the value under a sentinel key (e.g. `"__value__"`). The loop unwraps it so
`event.output` is the bare primitive, not a dict.

### Example — task-mode agent with structured output

> **Important:** `mode='task'` agents cannot be placed directly as static
> workflow graph nodes. `Workflow.__init__` raises `ValueError` if it detects
> one. Task-mode agents must be used either (a) as sub-agents delegated by a
> `mode='chat'` coordinator via `AgentTool`, or (b) dispatched dynamically
> from a `FunctionNode` via `ctx.run_node()`. Option (a) is the canonical
> pattern:

```python
from pydantic import BaseModel
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, START

class TravelPlan(BaseModel):
    destination: str
    days: int
    highlights: list[str]
    estimated_budget_usd: float

# Task-mode agent: uses the finish_task FC/FR handshake for schema validation.
# It CANNOT be a static workflow graph node — it must run as a sub-agent.
planner = LlmAgent(
    name="planner",
    model="gemini-2.5-pro",
    mode="task",            # enables the finish_task FC/FR handshake
    instruction=(
        "Create a detailed travel plan for the destination provided. "
        "Return a valid TravelPlan object."
    ),
    output_schema=TravelPlan,
    output_key="travel_plan",
)

# Chat coordinator dispatches the planner as a sub-agent via AgentTool.
coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    mode="chat",
    instruction=(
        "Ask the travel planner to create a plan for the user's destination, "
        "then summarise the result."
    ),
    tools=[planner],  # planner is wrapped as a _TaskAgentTool automatically
)

wf = Workflow(
    name="travel_wf",
    edges=[(START, coordinator)],
)
# After the workflow runs, ctx.state["travel_plan"] holds a TravelPlan dict.
# The finish_task tool validates the schema; if invalid, the LLM retries
# automatically before the coordinator receives the result.
```

---

## 5 · `_should_retry_node` + `_get_retry_delay` — retry decision and backoff

**Source:** `google/adk/workflow/utils/_retry_utils.py`

These two functions implement the core retry logic for workflow nodes. They are
called by the node runner after every node failure.

### `_should_retry_node`

```python
def _should_retry_node(
    exception: BaseException,
    retry_config: RetryConfig | None,
    node_state: NodeState,
) -> bool:
    if not retry_config:
        return False

    attempt_count = node_state.attempt_count
    max_attempts = retry_config.max_attempts or 5  # default is 5

    # attempt_count is 1-based; >= max_attempts means limit reached
    if attempt_count >= max_attempts:
        return False

    # If exceptions list is set, only retry for listed exception types
    if retry_config.exceptions is not None:
        ex_name = type(exception).__name__
        if ex_name not in retry_config.exceptions:
            return False

    return True
```

Two key details:
- `attempt_count` starts at **1** for the original attempt (not 0).
- `retry_config.exceptions` stores **string names**, not class objects. This
  is enforced by `RetryConfig._normalize_exceptions`.

### `_get_retry_delay` — exponential backoff formula

```python
def _get_retry_delay(
    retry_config: RetryConfig | None,
    node_state: NodeState,
) -> float:
    if not retry_config:
        return 1.0

    initial_delay = retry_config.initial_delay if retry_config.initial_delay is not None else 1.0
    max_delay     = retry_config.max_delay     if retry_config.max_delay is not None     else 60.0
    backoff_factor= retry_config.backoff_factor if retry_config.backoff_factor is not None else 2.0
    jitter        = retry_config.jitter         if retry_config.jitter is not None         else 1.0

    attempt_count = node_state.attempt_count or 1
    # exponent is 0 for the first failure (attempt_count==1)
    attempt_for_calc = max(0, attempt_count - 1)

    delay = initial_delay * (backoff_factor ** attempt_for_calc)
    delay = min(delay, max_delay)

    if jitter > 0.0:
        random_offset = random.uniform(-jitter * delay, jitter * delay)
        delay = max(0.0, delay + random_offset)

    return delay
```

### Delay schedule with default settings

| Attempt | `attempt_for_calc` | `delay` before jitter |
|---------|--------------------|-----------------------|
| 1 (original) | — (no retry yet) | — |
| 2 (1st retry) | 0 | 1.0 s |
| 3 (2nd retry) | 1 | 2.0 s |
| 4 (3rd retry) | 2 | 4.0 s |
| 5 (4th retry) | 3 | 8.0 s |

With `jitter=1.0` (the default) the actual delay is sampled uniformly from
`[0, 2×delay]`, adding randomness to prevent thundering-herd retries.

### Practical retry configuration

```python
from google.adk.workflow import RetryConfig

# Retry quickly on transient API errors, give up fast on logic errors
api_retry = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    backoff_factor=3.0,
    jitter=0.5,                    # ±50 % randomness
    exceptions=["TimeoutError", "ConnectionError"],
)

# No jitter for deterministic testing
deterministic_retry = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    jitter=0.0,    # disables all randomness
)

# Verify the schedule manually
from google.adk.workflow._node_state import NodeState
from google.adk.workflow.utils._retry_utils import _get_retry_delay

class FakeState:
    attempt_count = 2   # simulates the first retry

delay = _get_retry_delay(deterministic_retry, FakeState())
print(f"1st retry delay: {delay}s")   # 1.0 (exponent=0, no jitter)
```

---

## 6 · `RetryConfig` — retry configuration model

**Source:** `google/adk/workflow/_retry_config.py`

`RetryConfig` is the Pydantic model you attach to any workflow node, either
directly on the node constructor (`LlmAgent(..., retry_config=...)`) or via
`build_node(node_like, retry_config=...)` before passing the node into a
`Workflow`'s `edges` list.

### Full model

```python
class RetryConfig(BaseModel):
    max_attempts: int | None = None
    """Max attempts including the original. 0 or 1 = no retries. Default: 5."""

    initial_delay: float | None = None
    """Seconds before first retry. Default: 1.0."""

    max_delay: float | None = None
    """Cap on delay between retries in seconds. Default: 60.0."""

    backoff_factor: float | None = None
    """Multiplier applied after each attempt. Default: 2.0."""

    jitter: float | None = None
    """Randomness factor. Default: 1.0 (±100 %). Set to 0.0 to disable."""

    exceptions: list[str | type[BaseException]] | None = None
    """Exception class names or classes to retry on. None = retry all."""
```

### `_normalize_exceptions` field validator

`exceptions` accepts both string names and live exception classes; the
validator converts classes to names for uniform runtime checking:

```python
@field_validator('exceptions', mode='before')
@classmethod
def _normalize_exceptions(cls, v):
    if v is None:
        return None
    normalized = []
    for item in v:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, type) and issubclass(item, BaseException):
            normalized.append(item.__name__)
        else:
            raise ValueError(
                'exceptions must contain exception class names (str) or'
                f' exception classes, got {type(item).__name__}: {item!r}'
            )
    return normalized
```

### Attaching `RetryConfig` to nodes

The simplest approach is to pass `retry_config` directly to the node
constructor — all `BaseNode` subclasses (including `LlmAgent`) accept it:

```python
from google.adk.workflow import Workflow, RetryConfig, START
from google.adk.agents import LlmAgent

flaky_agent = LlmAgent(
    name="flaky",
    model="gemini-2.5-flash",
    instruction="Call an unreliable external API and return the result.",
    retry_config=RetryConfig(
        max_attempts=4,
        initial_delay=2.0,
        backoff_factor=2.0,
        max_delay=20.0,
        jitter=0.3,
        exceptions=[TimeoutError, "ConnectionRefusedError"],
    ),
)

wf = Workflow(
    name="resilient_wf",
    edges=[(START, flaky_agent)],
)
```

### Using class objects and strings together

```python
import httpx

retry = RetryConfig(
    exceptions=[
        httpx.TimeoutException,       # class → normalised to "TimeoutException"
        "ReadTimeout",                # string stays as-is
        ValueError,                   # class → "ValueError"
    ]
)
print(retry.exceptions)
# ['TimeoutException', 'ReadTimeout', 'ValueError']
```

---

## 7 · Graph validation suite

**Source:** `google/adk/workflow/utils/_graph_validation.py`

`validate_graph` is called automatically at `Workflow` construction time (in
`model_post_init` → `_build_graph`) to catch structural errors early. It
delegates to seven specialised validators, each raising `ValueError` with a
descriptive message.

### `validate_graph` — entry point

```python
def validate_graph(nodes: list[BaseNode], edges: list[Edge]) -> set[str]:
    node_names = _validate_duplicate_node_names(nodes)
    _validate_start_node(node_names)
    _validate_connectivity(edges, node_names)
    _validate_duplicate_edges(edges)
    _validate_default_routes(edges)
    _detect_unconditional_cycles(edges, node_names)
    _validate_static_schemas(edges)
    _validate_chat_agent_wiring(edges)
    return _compute_terminal_nodes(nodes, edges)
```

Returns the set of terminal node names (nodes with no outgoing edges).

### `_validate_duplicate_node_names`

Uses `collections.Counter` to find names appearing more than once:

```python
def _validate_duplicate_node_names(nodes: list[BaseNode]) -> set[str]:
    names = [node.name for node in nodes]
    duplicates = sorted(
        name for name, count in Counter(names).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Graph validation failed. Duplicate node names found: "
            f"{duplicates}. ..."
        )
    return set(names)
```

### `_detect_unconditional_cycles` — DFS cycle detection

Builds an adjacency list of **unconditional** edges (those without a `route`)
and runs recursive DFS. An edge with a `route` is safe because the router can
break the cycle:

```python
def _detect_unconditional_cycles(edges, node_names):
    unconditional_adj = {name: [] for name in node_names}
    for edge in edges:
        if edge.route is None:
            unconditional_adj[edge.from_node.name].append(edge.to_node.name)

    in_stack, done = set(), set()

    def _dfs(node, path):
        in_stack.add(node)
        path.append(node)
        for neighbor in unconditional_adj[node]:
            if neighbor in in_stack:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                raise ValueError(
                    "Graph validation failed. Unconditional cycle detected: "
                    f"{' -> '.join(cycle)}. Cycles must include at least "
                    "one conditional (routed) edge to avoid infinite loops."
                )
            if neighbor not in done:
                _dfs(neighbor, path)
        path.pop()
        in_stack.remove(node)
        done.add(node)

    for name in node_names:
        if name not in done:
            _dfs(name, [])
```

### `_validate_connectivity`

BFS/DFS from `START`; any node not reachable raises an error. Also ensures
`START` has no incoming edges:

```python
def _validate_connectivity(edges, node_names):
    adj = {name: set() for name in node_names}
    to_nodes = set()
    for edge in edges:
        adj[edge.from_node.name].add(edge.to_node.name)
        to_nodes.add(edge.to_node.name)

    reachable, stack = set(), [START.name]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(adj[node] - reachable)

    unreachable = node_names - reachable
    if unreachable:
        raise ValueError(
            f"Graph validation failed. The following nodes are unreachable "
            f"from START: {sorted(unreachable)}"
        )
    if START.name in to_nodes:
        raise ValueError("Graph validation failed. START node must not have incoming edges.")
```

### `_validate_default_routes`

`DEFAULT_ROUTE` has two constraints: it cannot appear in a list with other
routes, and at most one `DEFAULT_ROUTE` edge may exist per source node:

```python
def _validate_default_routes(edges):
    default_route_edges: dict[str, str] = {}
    for edge in edges:
        if isinstance(edge.route, list) and DEFAULT_ROUTE in edge.route:
            raise ValueError(
                "Graph validation failed. DEFAULT_ROUTE cannot be combined "
                "with other routes in a list ..."
            )
        if edge.route == DEFAULT_ROUTE:
            from_name = edge.from_node.name
            if from_name in default_route_edges:
                raise ValueError(
                    f"Graph validation failed. Multiple DEFAULT_ROUTE edges "
                    f"found from node {from_name} ..."
                )
            default_route_edges[from_name] = edge.to_node.name
```

### `_validate_chat_agent_wiring`

Chat-mode agents cannot receive direct node inputs from non-START nodes, because
they rely on session history rather than structured `node_input`:

```python
def _validate_chat_agent_wiring(edges):
    for edge in edges:
        to_node = edge.to_node
        if isinstance(to_node, LlmAgent) and getattr(to_node, "mode", None) == "chat":
            if edge.from_node.name != START.name:
                raise ValueError(
                    f"The agent '{to_node.name}' has been added to the workflow "
                    f"with mode='chat' following node '{edge.from_node.name}'. "
                    "This is not supported because chat-mode agents rely on "
                    "conversational history ..."
                )
```

### Triggering validation errors intentionally (for testing)

Validation runs at `Workflow(...)` construction time, so `pytest.raises` wraps
the constructor call:

```python
from google.adk.workflow import Workflow, START, Edge
from google.adk.agents import LlmAgent
import pytest

def test_unconditional_cycle_detected():
    a = LlmAgent(name="a", model="gemini-2.5-flash", instruction="step a")
    b = LlmAgent(name="b", model="gemini-2.5-flash", instruction="step b")
    with pytest.raises(ValueError, match="Unconditional cycle detected"):
        Workflow(
            name="bad_wf",
            edges=[
                (START, a, b),              # START → a → b
                Edge(from_node=b, to_node=a),  # b → a: unconditional back-edge
            ],
        )

def test_unreachable_nodes_detected():
    a = LlmAgent(name="a", model="gemini-2.5-flash", instruction="step a")
    b = LlmAgent(name="b", model="gemini-2.5-flash", instruction="unreachable")
    c = LlmAgent(name="c", model="gemini-2.5-flash", instruction="also unreachable")
    # b and c form a subgraph not connected to START
    with pytest.raises(ValueError, match="unreachable from START"):
        Workflow(
            name="bad_wf2",
            edges=[
                (START, a),                    # only a is reachable
                Edge(from_node=b, to_node=c),  # b→c subgraph disconnected from START
            ],
        )
```

---

## 8 · `build_node` — universal `NodeLike` → `BaseNode` converter

**Source:** `google/adk/workflow/utils/_workflow_graph_utils.py`

`build_node` is the factory called internally by the graph edge parser (and by
`_dispatch_task_fc`) to turn any valid node-like object into a concrete
`BaseNode`. It also applies mode defaults for `LlmAgent` instances.

### Signature

```python
def build_node(
    node_like: NodeLike,
    *,
    name: str | None = None,
    rerun_on_resume: bool | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
    auth_config: Any = None,
    parameter_binding: Literal['state', 'node_input'] = 'state',
) -> BaseNode:
```

### Dispatch logic

```
node_like == 'START'    → returns the singleton START node

isinstance(node_like, LlmAgent):
    - clone the agent applying name/rerun_on_resume/retry_config/timeout
    - set rerun_on_resume=True by default (LlmAgents must rerun on resume)
    - if mode is None:
        parent_agent is set  → mode = 'chat'   (sub-agent, transfer-capable)
        parent_agent is None → mode = 'single_turn'  (standalone node)
    - if mode in ('task', 'chat'): set wait_for_output = True
    - if parallel_worker: wrap in _ParallelWorker

isinstance(node_like, BaseNode) (but not LlmAgent):
    - model_copy(update=kwargs) if any overrides, else return as-is

isinstance(node_like, BaseTool):
    - wrap in _ToolNode(tool, name, retry_config, timeout)

callable(node_like):
    - wrap in FunctionNode(func, name, rerun_on_resume, retry_config,
                           timeout, auth_config, parameter_binding)
```

### `parameter_binding` — `'state'` vs `'node_input'`

When `parameter_binding='node_input'`, `FunctionNode` binds parameters from
the incoming `node_input` dict rather than from session state, and also infers
`input_schema`/`output_schema` from the function's type annotations:

```python
# Default: parameters come from ctx.state
async def enrich(ctx, session_id: str) -> dict:
    ...

# node_input mode: parameters come from the dict passed as node_input
async def transform(text: str, max_words: int = 50) -> str:
    ...

from google.adk.workflow import Workflow, START, FunctionNode

# Build a FunctionNode explicitly to set parameter_binding
transform_node = FunctionNode(func=transform, parameter_binding='node_input')

wf = Workflow(
    name="wf",
    edges=[(START, transform_node)],
)
```

### LlmAgent mode defaults in practice

```python
from google.adk.agents import LlmAgent
from google.adk.workflow.utils._workflow_graph_utils import build_node

standalone = LlmAgent(name="s", model="gemini-2.5-flash", instruction="...")
sub = LlmAgent(name="sub", model="gemini-2.5-flash", instruction="...")
sub.parent_agent = standalone  # simulate being a sub-agent

node_s = build_node(standalone)
node_sub = build_node(sub)

print(node_s.mode)    # 'single_turn'
print(node_sub.mode)  # 'chat'
print(node_s.rerun_on_resume)   # True (always set for LlmAgent)
print(node_sub.wait_for_output) # True (chat mode requires wait_for_output)
```

---

## 9 · HITL workflow utilities

**Source:** `google/adk/workflow/utils/_workflow_hitl_utils.py`

These utilities power Human-in-the-Loop (HITL) interrupts in workflows — the
mechanism by which a node can pause execution and request data from a human
operator or external system.

### Constants

```python
REQUEST_INPUT_FUNCTION_CALL_NAME    = 'adk_request_input'
REQUEST_CREDENTIAL_FUNCTION_CALL_NAME = 'adk_request_credential'
```

### `create_request_input_event`

Wraps a `RequestInput` object into a model-role event containing an
`adk_request_input` function call. The workflow engine yields this event to the
caller, which then resumes with the human's response.

```python
def create_request_input_event(request_input: RequestInput) -> Event:
    args = request_input.model_dump(exclude={'response_schema'}, by_alias=True)
    args['response_schema'] = (
        schema_to_json_schema(request_input.response_schema)
        if request_input.response_schema is not None
        else None
    )
    return Event(
        content=types.Content(
            role='model',
            parts=[types.Part(
                function_call=types.FunctionCall(
                    name=REQUEST_INPUT_FUNCTION_CALL_NAME,
                    args=args,
                    id=request_input.interrupt_id,
                )
            )],
        ),
        long_running_tool_ids=[request_input.interrupt_id],
    )
```

`long_running_tool_ids` tells the runtime that this FC will be resolved
asynchronously — the session stays interrupted until a matching FR arrives.

### `create_request_input_response`

Builds the resume payload — a `FunctionResponse` part that the human's reply
is wrapped in:

```python
def create_request_input_response(
    interrupt_id: str,
    response: Mapping[str, Any],
) -> types.Part:
    return types.Part(
        function_response=types.FunctionResponse(
            id=interrupt_id,
            name=REQUEST_INPUT_FUNCTION_CALL_NAME,
            response=response,
        )
    )
```

### `process_auth_resume`

Stores credentials returned from an auth interrupt into session state. Tries
three formats in order: full `AuthConfig` dict, `AuthCredential` dict, or a
plain value (API key string):

```python
async def process_auth_resume(response_data, auth_config, state) -> None:
    try:
        response_config = AuthConfig.model_validate(response_data)
    except (ValidationError, TypeError):
        response_config = auth_config.model_copy(deep=True)
        response_config.exchanged_auth_credential = _build_credential_from_value(
            auth_config, response_data
        )
    response_config.credential_key = auth_config.credential_key
    await AuthHandler(auth_config=response_config).parse_and_store_auth_response(state=state)
```

### End-to-end HITL example — human approval gate

```python
import asyncio
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_event,
    create_request_input_response,
    has_request_input_function_call,
    get_request_input_interrupt_ids,
)
from google.adk.workflow import Workflow, START
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# A node that emits a HITL interrupt.
#
# IMPORTANT: the NodeRunner does NOT stop consuming the async generator when
# it sees long_running_tool_ids.  Code placed *after* a yield inside the
# generator runs immediately in the same invocation with empty resume_inputs.
#
# The correct pattern:
#   1. Check ctx.resume_inputs at the TOP of the function (populated on resume).
#   2. If present → process the answer and return.
#   3. If absent  → yield the interrupt event and let the generator end.
# On resume the Workflow re-invokes the node from scratch, ctx.resume_inputs
# is pre-populated, and the early-return branch handles the answer.
async def approval_gate(ctx):
    # Step 1: on resume, ctx.resume_inputs is populated — handle the answer.
    response = ctx.resume_inputs.get("approval-001")
    if response is not None:
        approval = response.get("result")
        if approval != "approve":
            raise ValueError("Action rejected by human operator.")
        return  # completed; downstream node is triggered by the graph edge

    # Step 2: first run — request user input and end the generator.
    request = RequestInput(
        interrupt_id="approval-001",
        message="Please approve or reject this action (approve/reject):",
        # Use an object schema — create_request_input_response takes a
        # Mapping[str, Any], so the resume payload must be a dict.
        # Wrap the user's choice under the conventional "result" key.
        response_schema={
            "type": "object",
            "properties": {"result": {"type": "string", "enum": ["approve", "reject"]}},
            "required": ["result"],
        },
    )
    yield create_request_input_event(request)
    # Generator ends here. NodeRunner registers the interrupt_id from
    # long_running_tool_ids and the Workflow marks this node WAITING.
    # When the user sends a matching FunctionResponse, the node re-runs
    # from the top with ctx.resume_inputs["approval-001"] populated.

downstream = LlmAgent(
    name="action",
    model="gemini-2.5-flash",
    instruction="Perform the approved action.",
)

wf = Workflow(
    name="hitl_wf",
    edges=[(START, approval_gate, downstream)],
)

async def main():
    svc = InMemorySessionService()
    session = await svc.create_session(app_name="demo", user_id="u1")
    runner = Runner(agent=wf, app_name="demo", session_service=svc)
    first_message = types.Content(role="user", parts=[types.Part(text="start")])

    interrupt_id = None
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=first_message):
        if has_request_input_function_call(event):
            ids = get_request_input_interrupt_ids(event)
            interrupt_id = ids[0] if ids else None
            print("Workflow paused — waiting for human approval")

    # Resume with human's response
    if interrupt_id:
        resume_part = create_request_input_response(interrupt_id, {"result": "approve"})
        resume_message = types.Content(role="user", parts=[resume_part])
        async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=resume_message):
            if event.output:
                print("Final output:", event.output)

asyncio.run(main())
```

---

## 10 · `resolve_and_derive_transfer_context` — agent transfer routing

**Source:** `google/adk/workflow/utils/_transfer_utils.py`

When an `LlmAgent` calls `transfer_to_agent`, the runtime needs to resolve the
target agent and determine which parent `Context` the target should run under.
`resolve_and_derive_transfer_context` does both in a single pass.

### Signature

```python
def resolve_and_derive_transfer_context(
    target_name: str,
    current_agent: BaseAgent,
    root_agent: BaseAgent,
    curr_ctx: Context,
    curr_parent_ctx: Context | None,
) -> tuple[BaseAgent, Context | None] | tuple[None, None]:
```

Returns `(target_agent, next_parent_context)` or `(None, None)` if the target
is not found. If found but the relationship is unrelated, returns
`(target_agent, None)`.

Raises `ValueError` if the target is the same agent as the caller.

### Four routing cases

```
Case 1: SELF
    target_agent.name == current_agent.name
    → raise ValueError("Agent cannot transfer to itself")

Case 2: Direct CHILD
    target_agent.parent_agent.name == current_agent.name
    → return (target_agent, curr_ctx)     # nested under current context

Case 3: SIBLING
    target_agent.parent_agent.name == current_agent.parent_agent.name
    → return (target_agent, curr_parent_ctx)  # same level

Case 4: Direct PARENT
    current_agent.parent_agent.name == target_agent.name
    → walk up context chain to find target's context, return its parent
    → fallback to outermost root context if not found in chain
```

### Implementation

```python
def resolve_and_derive_transfer_context(
    target_name, current_agent, root_agent, curr_ctx, curr_parent_ctx
):
    target_agent = root_agent.find_agent(target_name)
    if not target_agent:
        return None, None

    if target_agent.name == current_agent.name:
        raise ValueError(f"Agent '{target_name}' cannot transfer to itself.")

    # Child transfer
    if (target_agent.parent_agent and
            target_agent.parent_agent.name == current_agent.name):
        return target_agent, curr_ctx

    # Sibling transfer
    if (target_agent.parent_agent and current_agent.parent_agent and
            target_agent.parent_agent.name == current_agent.parent_agent.name):
        return target_agent, curr_parent_ctx

    # Parent transfer — walk up the context chain
    if (current_agent.parent_agent and
            current_agent.parent_agent.name == target_agent.name):
        curr = curr_ctx
        while curr is not None and curr.node is not None:
            if curr.node.name == target_name:
                return target_agent, curr.parent_ctx
            curr = curr.parent_ctx
        # Root coordinator fallback
        root_ctx = curr_ctx
        while root_ctx.parent_ctx is not None and root_ctx.node is not None:
            root_ctx = root_ctx.parent_ctx
        return target_agent, root_ctx

    # Unrelated — target found but no direct routing relationship
    return target_agent, None
```

### Agent hierarchy and transfer patterns

```python
from google.adk.agents import LlmAgent

# Three-level hierarchy: coordinator → specialist → helper
helper = LlmAgent(name="helper", model="gemini-2.5-flash",
                  instruction="Answer simple questions.")
specialist = LlmAgent(name="specialist", model="gemini-2.5-flash",
                      instruction=(
                          "Handle complex queries. Transfer to 'helper' for "
                          "simple ones, or back to 'coordinator' when done."
                      ),
                      sub_agents=[helper])
coordinator = LlmAgent(name="coordinator", model="gemini-2.5-pro",
                       instruction=(
                           "Route tasks. Transfer to 'specialist' for complex work."
                       ),
                       sub_agents=[specialist])

# Transfer scenarios the function handles:
#
# coordinator → specialist:  CHILD transfer
#   target.parent_agent.name == "coordinator" == current_agent.name
#   next_parent_ctx = coordinator's current Context
#
# specialist → helper:       CHILD transfer
#   target.parent_agent.name == "specialist" == current_agent.name
#   next_parent_ctx = specialist's current Context
#
# specialist → coordinator:  PARENT transfer
#   current_agent.parent_agent.name == "coordinator" == target_agent.name
#   walks up context chain to find coordinator's context, returns its parent
#
# helper → specialist:       PARENT transfer
#   current_agent.parent_agent.name == "specialist" == target_agent.name
```

### What "unrelated" means

If `specialist` tried to transfer to a `helper2` that belongs to a completely
different branch of the agent tree, `resolve_and_derive_transfer_context`
returns `(helper2, None)`. The caller is then responsible for deciding whether
to proceed (with a root-level context) or raise an error.

---

*All content verified against source at `/usr/local/lib/python3.11/dist-packages/google/adk` on google-adk 2.4.0.*

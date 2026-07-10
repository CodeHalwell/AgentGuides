---
title: "Workflows (graph orchestration)"
description: "Compose agents and functions into a DAG with conditional routing, retries, timeouts, HITL, and state schemas."
framework: google-adk
language: python
sidebar:
  order: 25
---

Verified against google-adk==2.4.0 (`google/adk/workflow/`).

`Workflow` is the graph-based orchestrator that replaces `SequentialAgent`, `ParallelAgent`, and `LoopAgent` in ADK 2.x. It is a `BaseNode` (not a `BaseAgent`) — wire it to a `Runner` via `App(root_agent=workflow)`.

## Public surface

`google.adk.workflow` exports (`workflow/__init__.py`):

| Name | Purpose |
|---|---|
| `Workflow` | Graph orchestrator. Takes `edges=[...]` and runs the DAG. |
| `BaseNode` | Pydantic base for every node. |
| `Node` | Subclass-friendly base — implement `run_node_impl`. |
| `node` | Decorator / function to wrap a callable, agent, or tool as a `BaseNode`. |
| `FunctionNode` | What `@node` produces for a function. |
| `JoinNode` | Fan-in — waits for all predecessors before emitting. |
| `Edge` | Explicit `from_node → to_node`, optional `route`. |
| `RetryConfig` | Per-node retry policy. |
| `NodeTimeoutError` | Raised when a node exceeds its `timeout`. |
| `START` | Sentinel for the graph entry point. |
| `DEFAULT_ROUTE` | Matches routes with no explicit mapping. |

## Minimal example

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START

drafter = LlmAgent(
    name="drafter",
    model="gemini-2.5-flash",
    instruction="Write a tight 3-sentence summary of the input.",
    mode="single_turn",
)
polisher = LlmAgent(
    name="polisher",
    model="gemini-2.5-flash",
    instruction="Shorten and sharpen the input. Return only the final text.",
    mode="single_turn",
)

@node
def trim(node_input: str) -> str:
    return node_input.strip()

pipeline = Workflow(
    name="summarize_pipeline",
    edges=[(START, trim, drafter, polisher)],
)

async def main():
    app = App(name="demo", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Electric cars sold well in Q1.", user_id="u1", session_id="s1"
    )

asyncio.run(main())
```

A tuple in `edges=` is a **chain** — each adjacent pair becomes an `Edge`. `START` is the sentinel that receives `new_message` on invocation.

## Edge syntax

Four ways to declare edges, all verified in `workflow/_workflow_graph.py`:

### 1. Chain (tuple)

```python
edges = [(START, a, b, c)]   # START→a, a→b, b→c
```

### 2. Fan-out (nested tuple)

```python
edges = [(START, (a, b, c), join)]   # START fans out to a/b/c, all feed join
```

### 3. Routing map (dict)

```python
edges = [
    (START, classifier, {"billing": billing_agent, "support": support_agent}),
]
```
Edges carry a `route` value. The source node picks one by setting `ctx.route = "billing"` (in a `FunctionNode`) or by the router's output matching a key. Use `DEFAULT_ROUTE` for the fallback.

### 4. Explicit `Edge`

```python
from google.adk.workflow import Edge
edges = [Edge(from_node=a, to_node=b, route="yes")]
```

Mix them freely. `BaseAgent`, `BaseTool`, and plain callables are auto-wrapped via `build_node()` when they appear in an edge.

## `@node` decorator

```python
from google.adk.workflow import node, RetryConfig

@node
async def fetch(node_input: str, ctx) -> dict:
    ctx.state["last_query"] = node_input
    return {"query": node_input, "results": [...]}

@node(
    name="safe_fetch",
    retry_config=RetryConfig(max_attempts=3),
    timeout=10.0,
    rerun_on_resume=True,
)
async def safe_fetch(node_input: str, ctx): ...
```

### `RetryConfig` fields

`RetryConfig` lives in `google.adk.workflow` and implements exponential backoff with jitter. All fields are optional — omitting them uses the stated defaults:

| Field | Default | Description |
|---|---|---|
| `max_attempts` | `5` | Total attempts including the first. Set to `0` or `1` for no retries. |
| `initial_delay` | `1.0` s | Delay before the first retry. |
| `max_delay` | `60.0` s | Cap on inter-retry delay. |
| `backoff_factor` | `2.0` | Multiplier applied after each failure. |
| `jitter` | `1.0` | Randomness injected into delay (`0.0` = no jitter). |
| `exceptions` | `None` | List of exception class names or classes to retry on. `None` = retry on any exception. |

```python
from google.adk.workflow import RetryConfig, node, Workflow, START

# Retry only on network-related errors, up to 4 attempts with exponential backoff
@node(
    retry_config=RetryConfig(
        max_attempts=4,
        initial_delay=0.5,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=0.5,
        exceptions=["httpx.TimeoutException", "httpx.ConnectError", "ConnectionError"],
    ),
    timeout=15.0,         # NodeTimeoutError if the node takes > 15s
    rerun_on_resume=True,
)
async def resilient_fetch(url: str, ctx) -> dict:
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

pipeline = Workflow(
    name="fetch_pipeline",
    edges=[(START, resilient_fetch)],
)
```

`NodeTimeoutError` is raised when a node exceeds its `timeout`. It IS retried (only when `exceptions` is `None` or includes `NodeTimeoutError`), so set a timeout shorter than `max_delay * max_attempts` if you want the workflow to eventually fail fast.

Signatures recognised by `FunctionNode`:
- `node_input` — the incoming value (the predecessor's output, or the user's `new_message` for START successors).
- `ctx` — the `Context` (state, `run_node`, `route`, `interrupt`, artifact helpers).
- Any other parameter must be declared in the enclosing `Workflow.state_schema` — the framework injects its current value.

Return types are honoured: returning a value sets `ctx.output`; yielding values from an async generator lets you stream partials.

## `Node` subclass

Use when you need class-level state or `parallel_worker=True`:

```python
from google.adk.workflow import Node
from collections.abc import AsyncGenerator
from typing import Any

class DedupeNode(Node):
    name: str = "dedupe"
    seen: set[str] = set()

    async def run_node_impl(self, *, ctx, node_input: list[str]) -> AsyncGenerator[Any, None]:
        fresh = [x for x in node_input if x not in self.seen]
        self.seen.update(fresh)
        yield fresh
```

Setting `parallel_worker=True` lets the node be invoked concurrently per trigger without sharing state with other invocations.

## `Workflow` fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Must be a Python identifier |
| `edges` | `list[EdgeItem]` | `[]` | Tuples or `Edge` objects |
| `max_concurrency` | `int \| None` | `None` | Caps concurrent graph-scheduled nodes |
| `state_schema` | `type[BaseModel]` | `None` | Validates `ctx.state` mutations |
| `rerun_on_resume` | `bool` | `True` | Workflow-level resume behaviour |
| `input_schema` / `output_schema` | `SchemaType` | `None` | Validates workflow-level in/out |

All fields from `BaseNode` (`retry_config`, `timeout`, `wait_for_output`, ...) apply too.

## Routing and conditions

A node can steer the graph by setting `ctx.route`:

```python
@node
async def classify(node_input: str, ctx):
    intent = "billing" if "invoice" in node_input.lower() else "support"
    ctx.route = intent
    return node_input

workflow = Workflow(
    name="triage",
    edges=[
        (START, classify, {
            "billing": billing_agent,
            "support": support_agent,
            DEFAULT_ROUTE: fallback_agent,
        }),
    ],
)
```

## Loops

Replace `LoopAgent` with a routing map that either loops back to the same node or flows to a terminal node (any node with no outgoing edges). There is **no** `END_NODE` sentinel — terminality is structural.

```python
from google.adk.workflow import Workflow, node, START

@node(rerun_on_resume=True)
async def critic(draft: str, ctx) -> str:
    if len(draft) < 500:
        ctx.route = "done"
        return draft
    ctx.route = "continue"
    return draft[:500]  # trimmed draft fed back in

@node
def publish(draft: str) -> str:
    return draft  # terminal: no outgoing edge

loop = Workflow(
    name="refine",
    edges=[
        (START, critic, {"continue": critic, "done": publish}),
    ],
)
```

`publish` is the terminal node — the workflow finishes when routing lands there. Persist iteration count in `ctx.state` if you need `max_iterations` semantics, and set `max_concurrency=1` to keep the loop single-threaded.

## JoinNode

Fan-in that waits for **all** declared predecessors. Useful after a `(START, (a, b, c), join)` fan-out.

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import JoinNode, Workflow, node, START

a = LlmAgent(name="a", model="gemini-2.5-flash", instruction="Reply 'A'.", mode="single_turn")
b = LlmAgent(name="b", model="gemini-2.5-flash", instruction="Reply 'B'.", mode="single_turn")

join = JoinNode(name="merge")

@node
def finalize(node_input: dict) -> str:
    # node_input is a dict keyed by predecessor name
    return f"A={node_input['a']} B={node_input['b']}"

wf = Workflow(
    name="fanin",
    edges=[(START, (a, b), join, finalize)],
)
```

`JoinNode` receives a dict of `{predecessor_name: output}` as its `node_input`.

## Dynamic nodes (`ctx.run_node`)

Call another node from inside a node — the result is awaited in-place. **The caller must have `rerun_on_resume=True`** or `run_node` raises `ValueError` (`agents/context.py:399-405`).

```python
@node(rerun_on_resume=True)
async def supervisor(q: str, ctx):
    research = await ctx.run_node(research_agent, q)
    answer = await ctx.run_node(writer_agent, research)
    return answer
```

## Human-in-the-loop

Yield a `RequestInput` from a node to pause the workflow until `Runner.run_async` is called again with a `new_message` carrying a matching function-response. Pair with `auth_config=` on the `@node` to gate on an OAuth flow.

```python
from google.adk.events.request_input import RequestInput

@node(rerun_on_resume=True)
async def approve(draft: str, ctx):
    decision = yield RequestInput(
        interrupt_id="approval_001",
        message="Approve the draft? yes/no",
    )
    if decision == "yes":
        return draft
    ctx.route = "rewrite"
```

Combine with `ResumabilityConfig(is_resumable=True)` on the `App` so state is persisted across the pause.

## Retries and timeouts

```python
from google.adk.workflow import node, RetryConfig

@node(retry_config=RetryConfig(max_attempts=3, backoff_factor=2.0), timeout=30.0)
async def flaky(q: str, ctx): ...
```

A node that exceeds `timeout` is cancelled and raises `NodeTimeoutError`. If retries are configured, the node is restarted.

## Patterns

### 1 — Linear pipeline
`edges=[(START, step1, step2, step3)]` — direct replacement for `SequentialAgent`.

### 2 — Map-reduce
`edges=[(START, split, (worker1, worker2, worker3), join, summarize)]` — fan-out with `JoinNode`.

### 3 — Router → specialist fleet
`classify` node sets `ctx.route` to a string; routing map fans to the matching agent; optional `DEFAULT_ROUTE` catches unknowns.

### 4 — Retry-aware scraper
Wrap the scraper with `retry_config=RetryConfig(max_attempts=5)` and `timeout=20`. Log retries via `LoggingPlugin` on the `App`.

### 5 — HITL review gate
Insert a `@node(rerun_on_resume=True, auth_config=...)` that yields `RequestInput` between producer and publisher. The workflow pauses, the event is persisted, `Runner.run_async` resumes on the next user turn.

### 6 — `state_schema` — validated shared state

Pass a Pydantic `BaseModel` class as `Workflow.state_schema`. The framework validates every `ctx.state` write against it and raises `StateSchemaError` on unknown keys. Schema fields can be injected directly as `@node` function parameters.

```python
from pydantic import BaseModel
from google.adk.workflow import Workflow, node, START

class PipelineState(BaseModel):
    iteration: int = 0
    best_score: float = 0.0
    status: str = "pending"

@node
def tracker(node_input: str, ctx, iteration: int = 0) -> str:
    # `iteration` is injected from ctx.state["iteration"]; default 0 for first run
    # (state_schema validates writes but does not pre-populate state with defaults)
    ctx.state["iteration"] = iteration + 1
    ctx.state["status"] = "running"
    return f"pass {iteration + 1}: {node_input}"

@node
def scorer(node_input: str, ctx) -> float:
    score = len(node_input) / 100.0
    if score > ctx.state.get("best_score", 0.0):
        ctx.state["best_score"] = score
    return score

pipeline = Workflow(
    name="scored_pipeline",
    state_schema=PipelineState,
    edges=[(START, tracker, scorer)],
)
```

Writing an unknown key (`ctx.state["typo_key"] = 1`) raises `StateSchemaError` at runtime — catching schema drift early.

## Full conditional routing example

A classifier node routes between specialist agents based on detected intent.  `DEFAULT_ROUTE` catches any unrecognised intent so the workflow never deadlocks on missing edges.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START, DEFAULT_ROUTE

billing_agent = LlmAgent(
    name="billing",
    model="gemini-2.5-flash",
    instruction="Handle billing questions. Be concise.",
    mode="single_turn",
)
support_agent = LlmAgent(
    name="support",
    model="gemini-2.5-flash",
    instruction="Handle technical support questions. Be concise.",
    mode="single_turn",
)
general_agent = LlmAgent(
    name="general",
    model="gemini-2.5-flash",
    instruction="Handle general queries. Be concise.",
    mode="single_turn",
)

@node
def classify(node_input: str, ctx) -> str:
    lower = node_input.lower()
    if any(w in lower for w in ("invoice", "payment", "charge", "refund")):
        ctx.route = "billing"
    elif any(w in lower for w in ("error", "bug", "crash", "broken", "help")):
        ctx.route = "support"
    else:
        ctx.route = DEFAULT_ROUTE
    return node_input   # pass through to the selected agent

triage_wf = Workflow(
    name="triage",
    edges=[
        (START, classify, {
            "billing": billing_agent,
            "support": support_agent,
            DEFAULT_ROUTE: general_agent,
        }),
    ],
)

async def main():
    app = App(name="triage_app", root_agent=triage_wf)
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name="triage_app", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "I need a refund for invoice #1234", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

## Fan-out / join (map-reduce)

Spawn multiple specialist agents in parallel, then aggregate with a `JoinNode` and a summary node.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.workflow import JoinNode, Workflow, node, START

security_reviewer = LlmAgent(
    name="security",
    model="gemini-2.5-flash",
    instruction="Review the code for security issues only. List findings as bullet points.",
    mode="single_turn",
)
perf_reviewer = LlmAgent(
    name="performance",
    model="gemini-2.5-flash",
    instruction="Review the code for performance issues only. List findings as bullet points.",
    mode="single_turn",
)
style_reviewer = LlmAgent(
    name="style",
    model="gemini-2.5-flash",
    instruction="Review the code for style/readability issues only. List findings as bullet points.",
    mode="single_turn",
)

join = JoinNode(name="join_reviews")

@node
def summarize(node_input: dict) -> str:
    # node_input is {predecessor_name: output} — one key per fan-out branch
    parts = []
    for reviewer, findings in node_input.items():
        parts.append(f"### {reviewer.capitalize()}\n{findings}")
    return "\n\n".join(parts)

review_wf = Workflow(
    name="code_review",
    edges=[
        # Fan-out to all three reviewers in parallel, then fan-in via join
        (START, (security_reviewer, perf_reviewer, style_reviewer), join, summarize),
    ],
)

async def main():
    # Intentionally vulnerable SQL — this is the code the review agents will catch
    code = "def get_user(id): return db.query(f'SELECT * FROM users WHERE id={id}')"
    app = App(name="review_app", root_agent=review_wf)
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name="review_app", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(code, user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

`JoinNode` waits for **all** predecessor branches before forwarding a `{name: output}` dict to the next node.

## HITL with persisted sessions

For real pause/resume you need a `DatabaseSessionService` so state survives the process restart between user turns.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow, node, START
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.genai import types

drafter = LlmAgent(
    name="drafter",
    model="gemini-2.5-flash",
    instruction="Write a short marketing email for the product described in the input.",
    mode="single_turn",
)

@node(rerun_on_resume=True)
async def human_review(node_input: str, ctx):
    # Pause the workflow and wait for the user to approve or reject.
    decision = yield RequestInput(
        interrupt_id="email_review",
        message=f"Draft:\n\n{node_input}\n\nApprove? (yes/no)",
    )
    # decision is the FunctionResponse payload dict, e.g. {"result": "yes"}
    result = decision.get("result", "") if isinstance(decision, dict) else str(decision)
    if result.strip().lower() == "yes":
        ctx.state["approved_email"] = node_input
        ctx.output = node_input
        return
    ctx.route = "revise"
    ctx.output = node_input
    return   # send back for another draft pass

approval_wf = Workflow(
    name="email_approval",
    edges=[
        # drafter → human_review on every pass (chain handles re-review automatically)
        # human_review routing: "revise" loops back to drafter; no route = approved = done
        (START, drafter, human_review, {
            "revise": drafter,   # rejection: loop back for another draft
            # No DEFAULT_ROUTE — when human_review sets no route, the workflow ends
        }),
    ],
)

async def main():
    session_svc = DatabaseSessionService("sqlite+aiosqlite:///./sessions.db")
    await session_svc.prepare_tables()   # public method (source: database_session_service.py:403); also called lazily on first use

    app = App(
        name="email_app",
        root_agent=approval_wf,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = Runner(app=app, session_service=session_svc)

    await session_svc.create_session(
        app_name="email_app", user_id="u1", session_id="s1"
    )

    # First run — triggers the HITL pause
    # run_async is keyword-only; wrap the user text in types.Content
    user_msg = types.Content(
        role="user",
        parts=[types.Part(text="noise-cancelling headphones for office workers")],
    )
    async for event in runner.run_async(
        new_message=user_msg,
        user_id="u1", session_id="s1",
    ):
        # Use has_request_input_function_call to detect a RequestInput pause
        # (event.actions.requested_auth_configs is for OAuth, not HITL interrupts)
        if has_request_input_function_call(event):
            print("Workflow paused. Waiting for human approval.")
        if hasattr(event, "content") and event.content:
            print(event.content.parts[0].text if event.content.parts else "")

asyncio.run(main())
```

To resume, send a `FunctionResponse` that matches the `interrupt_id` emitted by the `RequestInput` — plain text does not work here. Use `get_request_input_interrupt_ids(event)` to capture the ID during the first run, then build the response:

```python
from google.adk.workflow.utils._workflow_hitl_utils import (
    get_request_input_interrupt_ids,
    create_request_input_response,
)

# Collect the interrupt_id emitted during the first run
interrupt_id = None
async for event in runner.run_async(new_message=user_msg, user_id="u1", session_id="s1"):
    ids = get_request_input_interrupt_ids(event)
    if ids:
        interrupt_id = ids[0]

# Resume with a matching FunctionResponse
approval_response = types.Content(
    role="user",
    parts=[create_request_input_response(interrupt_id, {"result": "yes"})],
)
async for event in runner.run_async(
    new_message=approval_response,
    user_id="u1", session_id="s1",
):
    if event.content:
        print(event.content.parts[0].text if event.content.parts else "")
```

## Gotchas

- Nodes are **Pydantic models** — if you subclass `Node`, annotate fields or they won't serialise.
- `Workflow` is not a `BaseAgent`. `Runner(agent=workflow)` fails. Use `App(root_agent=workflow)` and `Runner(app=app, session_service=...)`.
- `ctx.run_node(callable)` requires `rerun_on_resume=True` on the calling node.
- `wait_for_output=True` means a node *must* yield output/route before it's marked complete. Forget that and the workflow deadlocks.
- When a tuple contains only one element (e.g. `(START, single_node)`), you still get a single edge — not sugar for fan-out. Fan-out needs a **nested** tuple: `(START, (a, b))`.
- Setting `nodes=` explicitly on `Workflow` raises — nodes are inferred from `edges`.
- `state_schema` validates `ctx.state` writes — fields on the schema class can be injected directly as node parameters; this is the recommended way to thread typed state through a pipeline.

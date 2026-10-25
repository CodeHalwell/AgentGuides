---
title: "Workflows (graph orchestration)"
description: "Compose agents and functions into a DAG with conditional routing, retries, timeouts, HITL, and state schemas."
framework: "google-adk"
language: "python"
sidebar:
  label: "Workflows"
  order: 30
---

Verified against `google-adk` %%version=google-adk%% (`google/adk/workflow/`), checked %%versions-checked%%. The examples were run against a stub model that stands in for Gemini. The human-in-the-loop helpers changed after 2.10 (see that section).

`Workflow` is the graph-based orchestrator that replaces `SequentialAgent`, `ParallelAgent`, and `LoopAgent` in ADK 2.x. It is a `BaseNode` (not a `BaseAgent`). Wire it to a `Runner` with `App(root_agent=workflow)`, which is also where plugins and resumability go; `Runner(agent=workflow, ...)` and `Runner(node=workflow, ...)` work too.

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
    model="%%model=gemini%%",
    instruction="Write a tight 3-sentence summary of the input.",
    mode="single_turn",
)
polisher = LlmAgent(
    name="polisher",
    model="%%model=gemini%%",
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

Four ways to declare edges:

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

## `build_node()` — automatic node conversion

`build_node` is an internal helper that converts any `NodeLike` value (callable, `LlmAgent`, `BaseTool`, or `Workflow`) into a concrete `BaseNode`. The edge parser calls it automatically whenever you place an agent, callable, or tool directly in an edge — you do not import or call it yourself. Understanding its mode-defaulting rules helps you predict how the edge parser will configure each node.

```python
from google.adk.workflow import Workflow, START
from google.adk.agents import LlmAgent

writer = LlmAgent(
    name="writer",
    model="%%model=gemini%%",
    instruction="Summarise the input in one sentence.",
)

# Place the agent directly in an edge — the edge parser calls build_node()
# internally, defaulting mode to 'single_turn' (no parent_agent provided).
wf = Workflow(name="demo", edges=[(START, writer)])
```

**Mode defaulting rules applied by `build_node` (for reference):**

| Input | `mode` result |
|---|---|
| `LlmAgent` with `mode=None`, no `parent_agent` | `'single_turn'` |
| `LlmAgent` with `mode=None`, `parent_agent` provided | `'chat'` |
| `LlmAgent` with `mode='task'` or `mode='chat'` | forces `wait_for_output=True` |
| `LlmAgent` in `single_turn` mode, `include_contents` not set | sets `include_contents='none'`: the node sees only its input, not the conversation |
| Any `LlmAgent` | defaults `rerun_on_resume=True` |

To control the mode explicitly, set it on the `LlmAgent` before placing it in the edge:

```python
from google.adk.workflow import Workflow, START
from google.adk.agents import LlmAgent

# Explicitly set mode so build_node() preserves it rather than defaulting
writer = LlmAgent(name="writer", model="%%model=gemini%%",
                  instruction="Summarise the input.", mode="single_turn")

wf = Workflow(name="demo", edges=[(START, writer)])
```

**Nested workflows.** A `Workflow` is itself a `BaseNode` and can appear inside another `Workflow`'s edge list. This lets you compose large pipelines from reusable sub-pipelines:

```python
from google.adk.workflow import Workflow, node, START

@node
def validate(node_input: str) -> str:
    if not node_input.strip():
        raise ValueError("empty input")
    return node_input

@node
def publish(node_input: str) -> dict:
    return {"published": True, "content": node_input}

# Inner pipeline — reusable unit
inner = Workflow(
    name="inner_pipeline",
    edges=[(START, validate)],
)

# Outer pipeline — nests inner as a node
outer = Workflow(
    name="outer_pipeline",
    edges=[(START, inner, publish)],   # Workflow is already BaseNode; edge parser accepts it directly
)
```

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
async def resilient_fetch(node_input: str, ctx) -> dict:
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(node_input, timeout=10.0)   # node_input is the URL
        resp.raise_for_status()
        return resp.json()

pipeline = Workflow(
    name="fetch_pipeline",
    edges=[(START, resilient_fetch)],
)
```

`NodeTimeoutError` is raised when a node exceeds its `timeout`, and it counts as a failure for retries: with `exceptions=None` a timed-out node is retried, and with an `exceptions` list it is retried only if `"NodeTimeoutError"` is in the list. The worst-case time for a node is therefore roughly `max_attempts × timeout` plus the delays between attempts.

Signatures recognised by `FunctionNode`:
- `node_input` — the incoming value (the predecessor's output, or the user's `new_message` for START successors).
- `ctx` — the `Context` (state, `run_node`, `route`, `interrupt`, artifact helpers).
- Any other parameter is read from session state by name. A missing key without a default fails the node with `WorkflowDataError: Missing value for parameter ...`, and if the workflow has a `state_schema` the name must be one of its fields (checked when the `Workflow` is built). A common mistake is naming the input after its meaning (`draft`, `url`) instead of `node_input`: that parameter is then looked up in state.

That is the default `parameter_binding="state"`. With `@node(parameter_binding="node_input")` the parameters are filled from the fields of a `dict`/Pydantic `node_input` instead, and `input_schema`/`output_schema` are inferred from the signature (this is the mode used when a node acts as an agent's tool):

```python
@node(parameter_binding="node_input")
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"      # called with node_input={"name": "Alice"}
```

Values are coerced to the annotations with a Pydantic `TypeAdapter`: a `dict` becomes the annotated `BaseModel`, a `list[dict]` becomes `list[Model]`, and a `types.Content` (such as the user's message reaching a START successor) becomes `str` when the parameter expects one.

Return types are honoured: returning a value sets `ctx.output`; yielding values from an async generator lets you stream partials.

## `Node` subclass

Use when you need a node with its own fields and logic:

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

`parallel_worker=True` (on `Node` or `@node`) makes the node a map step: given a list as input, it runs once per item in parallel and outputs the list of results. `max_parallel_workers` caps how many run at once.

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

### `Workflow` — `max_concurrency` and `state_schema` in action

Source-verified against `google.adk.workflow._workflow.Workflow` and its `_validate_state_schema` method (first verified in 2.7.1; re-checked against %%version=google-adk%%). Two fields deserve dedicated coverage because they change failure modes at graph-build time (`state_schema`) and at runtime (`max_concurrency`).

```python
from pydantic import BaseModel, Field
from google.adk.workflow import Workflow, node, START

# ── State schema is a Pydantic model. Every FunctionNode parameter that is
#    not `ctx` / `node_input` / `self` must appear as a field here, or the
#    Workflow raises StateSchemaError on construction — *before* it runs. ──
class ResearchState(BaseModel):
    topic: str = Field(default="")
    findings: list[str] = Field(default_factory=list)
    iteration: int = 0

@node
async def search(topic: str, ctx) -> list[str]:
    # `topic` matches a state field → auto-injected from ctx.state at run time.
    results = [f"result about {topic} #{i}" for i in range(3)]
    # A node's return value becomes event.output — it does NOT auto-populate
    # ctx.state. Write to state explicitly so the next node's state-bound
    # `findings` parameter reads the value we just produced instead of the
    # schema default (empty list).
    ctx.state["findings"] = results
    return results

@node
async def critique(findings: list[str], ctx) -> str:
    # `findings` matches a state field → auto-injected from ctx.state.
    return "\n".join(findings)

wf = Workflow(
    name="research",
    state_schema=ResearchState,
    max_concurrency=4,                  # cap graph-scheduled parallel nodes at 4
    edges=[(START, search, critique)],
)
# StateSchemaError would fire here if search() or critique() had a parameter
# name (say, `authors`) that is not declared on ResearchState.
```

**Why the check happens at construction, not runtime.** `_validate_state_schema` runs from `model_post_init`, right after `_build_graph` compiles the edge list. That means a mis-named parameter (typo, refactor drift) surfaces the moment you import the module — *not* on the ninetieth production request. The exact error message:

> `FunctionNode 'search' parameter 'authors' is not declared in state_schema 'ResearchState'. Declared fields: ['findings', 'iteration', 'topic']`

**What `max_concurrency` does — and does not.** It caps parallelism for **graph-scheduled** nodes only. `ctx.run_node(...)` dynamic nodes are excluded from the cap because throttling them would deadlock (their parent awaits them inline). So:

- A `(START, (a, b, c, d, e, f), join)` fan-out with `max_concurrency=2` runs `a, b` in parallel, then `c, d`, then `e, f`, then `join`.
- A single node that internally does `await asyncio.gather(*[ctx.run_node(...) for _ in range(20)])` still fans out to 20 dynamic tasks regardless of `max_concurrency`.

**`rerun_on_resume=True` (workflow level, default).** When a paused workflow resumes, it rebuilds its progress from the session's events and runs only the nodes that had not finished. Nodes default to `rerun_on_resume=False`: an interrupted node is completed with the resume response as its output rather than run again. Set `rerun_on_resume=True` on a node that needs to see the response (`ctx.resume_inputs`) and carry on, as in [Human-in-the-loop](#human-in-the-loop).

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

How the next nodes are chosen after a node finishes (`Graph.get_next_pending_nodes`):

1. Edges with no route are always followed (an unconditional fan-out).
2. Edges whose route matches the emitted `ctx.route` are followed; a list route matches if any value overlaps.
3. `DEFAULT_ROUTE` edges are followed only when no specific route matched.
4. If a node has routing edges and none matched, the branch ends with a logged warning, not an error.

Route values can be `bool`, `int` or `str`.

### Routing on a classifier (`ClassifierNode`, experimental)

Since 2.13.0, `ClassifierNode` (in `google.adk.labs.classifiers`) routes on a classifier's answer instead of code you write. The classifier gets the node's input and one question, either `Choice` (route is the chosen label) or `Binary` (route is `True` or `False`, with a `threshold` that defaults to 0.5). The input passes through unchanged, so the next node sees what the classifier saw. Graph validation checks that every label has an edge, unless a `DEFAULT_ROUTE` or unrouted edge catches the rest. Set `answer_key` to store the full answer, including each label's probability, in session state.

`LlmClassifier(model=...)` answers the question with any ADK model, as structured output. A `BaseClassifier` subclass can wrap a dedicated decision model instead; ADK includes `TypesafeClassifier` for TypeSafe's Jev models. The example below uses a stub model that stands in for Gemini; pass a real model name or `BaseLlm` in its place.

```python
import asyncio
import json

from google.adk.labs.classifiers import Choice, ClassifierNode, LlmClassifier
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, Workflow, node
from google.genai import types


class StubModel(BaseLlm):
    """Stands in for Gemini: always answers 'billing'."""
    model: str = "stub-classifier"

    async def generate_content_async(self, llm_request, stream=False):
        answer = {"route": {"billing": 0.9, "support": 0.1}}
        yield LlmResponse(content=types.Content(
            role="model", parts=[types.Part(text=json.dumps(answer))]))


@node
def billing(node_input: str):
    return f"billing handled: {node_input}"


@node
def support(node_input: str):
    return f"support handled: {node_input}"


router = ClassifierNode(
    name="router",
    classifier=LlmClassifier(model=StubModel()),
    question=Choice(
        instructions="Which team should handle this message?",
        criteria={"billing": "Invoices and payments", "support": "Everything else"},
    ),
    answer_key="route_answer",
)
workflow = Workflow(name="triage", edges=[(START, router, {"billing": billing, "support": support})])


async def main():
    runner = InMemoryRunner(agent=workflow, app_name="triage_app")
    session = await runner.session_service.create_session(app_name="triage_app", user_id="u1")
    message = types.Content(role="user", parts=[types.Part(text="Where is my invoice?")])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=message):
        if event.output and isinstance(event.output, str):
            print(event.output)   # billing handled: Where is my invoice?


asyncio.run(main())
```

## Loops

Replace `LoopAgent` with a routing map that either loops back to the same node or flows to a terminal node (any node with no outgoing edges). There is **no** `END_NODE` sentinel — terminality is structural.

```python
from google.adk.workflow import Workflow, node, START

@node
async def critic(node_input: str, ctx) -> str:
    if len(node_input) <= 500:
        ctx.route = "done"
        return node_input
    ctx.route = "continue"
    return node_input[:500]  # trimmed draft fed back in

@node
def publish(node_input: str) -> str:
    return node_input  # terminal: no outgoing edge

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

a = LlmAgent(name="a", model="%%model=gemini%%", instruction="Reply 'A'.", mode="single_turn")
b = LlmAgent(name="b", model="%%model=gemini%%", instruction="Reply 'B'.", mode="single_turn")

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

Call another node from inside a node — the result is awaited in place. **The caller must have `rerun_on_resume=True`**, or `run_node` raises `ValueError: A node must have rerun_on_resume=True`: if the child pauses, the workflow resumes by running the parent again.

```python
@node(rerun_on_resume=True)
async def supervisor(node_input: str, ctx):
    research = await ctx.run_node(research_agent, node_input)
    answer = await ctx.run_node(writer_agent, research)
    return answer
```

## Human-in-the-loop

Yield a `RequestInput` from a node to pause the workflow. The run ends with an `adk_request_input` function call whose ID is the `interrupt_id`; the workflow resumes when `Runner.run_async` is called again on the same session with a `FunctionResponse` of that ID and name. With `rerun_on_resume=True` the node runs again and finds the response in `ctx.resume_inputs[interrupt_id]`; without it, the response itself becomes the node's output. `yield` does not return the answer: the node is re-run instead.

```python
import asyncio

from google.adk.apps import App
from google.adk.events import RequestInput
from google.adk.runners import InMemoryRunner
from google.adk.workflow import START, Workflow, node
from google.genai import types


@node
def draft(node_input: str) -> str:
    return f"Draft post about {node_input}"


@node(rerun_on_resume=True)
def approve(node_input: str, ctx):
    decision = ctx.resume_inputs.get("approval")
    if decision is None:                       # first run: ask and pause
        yield RequestInput(interrupt_id="approval", message=f"Publish this? {node_input}")
        return
    ctx.route = "publish" if decision.get("approved") else "rewrite"
    yield node_input


@node
def publish(node_input: str) -> str:
    return f"published: {node_input}"


@node
def rewrite(node_input: str) -> str:
    return "sent back for rewriting"


review = Workflow(
    name="review",
    edges=[(START, draft, approve, {"publish": publish, "rewrite": rewrite})],
)


async def main():
    runner = InMemoryRunner(app=App(name="review_app", root_agent=review))
    session = await runner.session_service.create_session(app_name="review_app", user_id="u1")

    pending = None
    async for event in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="pricing")]),
    ):
        for call in event.get_function_calls():
            if call.name == "adk_request_input":
                pending = call.id                      # == the interrupt_id
                print("paused:", call.args["message"])

    answer = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=pending, name="adk_request_input", response={"approved": True},
    ))])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=answer):
        if event.output is not None:
            print(event.output)


asyncio.run(main())
```

Pair with `auth_config=` on the `@node` to gate a node on an OAuth flow instead of a free-form answer. For pauses that can outlast the process, use a persistent session service (next section).

## Retries and timeouts

```python
from google.adk.workflow import node, RetryConfig

@node(retry_config=RetryConfig(max_attempts=3, backoff_factor=2.0), timeout=30.0)
async def flaky(node_input: str, ctx): ...
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
Insert a `@node(rerun_on_resume=True)` that yields `RequestInput` between producer and publisher, as in [Human-in-the-loop](#human-in-the-loop). The workflow pauses, the event is persisted, and `Runner.run_async` resumes it when the answer arrives. Since 2.12.0, `@node(require_confirmation=True)` does the same for a plain yes/no approval before a node runs.

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
    model="%%model=gemini%%",
    instruction="Handle billing questions. Be concise.",
    mode="single_turn",
)
support_agent = LlmAgent(
    name="support",
    model="%%model=gemini%%",
    instruction="Handle technical support questions. Be concise.",
    mode="single_turn",
)
general_agent = LlmAgent(
    name="general",
    model="%%model=gemini%%",
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
    model="%%model=gemini%%",
    instruction="Review the code for security issues only. List findings as bullet points.",
    mode="single_turn",
)
perf_reviewer = LlmAgent(
    name="performance",
    model="%%model=gemini%%",
    instruction="Review the code for performance issues only. List findings as bullet points.",
    mode="single_turn",
)
style_reviewer = LlmAgent(
    name="style",
    model="%%model=gemini%%",
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
    # The last event comes from the `summarize` function node: its result is in
    # `event.output`, not `event.content`.
    print(events[-1].output)

asyncio.run(main())
```

`JoinNode` waits for **all** predecessor branches before forwarding a `{name: output}` dict to the next node.

## HITL with persisted sessions

For a pause that can outlast the process (a person approves tomorrow), keep sessions in a database. The first request runs until the workflow pauses and returns the interrupt ID; a later request, possibly in another process, answers it:

```python
import asyncio

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.events import RequestInput
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types

drafter = LlmAgent(
    name="drafter",
    model="%%model=gemini%%",
    instruction="Write a short marketing email for the product described in the input.",
    mode="single_turn",
)


@node(rerun_on_resume=True)
def human_review(node_input: str, ctx):
    decision = ctx.resume_inputs.get("email_review")
    if decision is None:
        yield RequestInput(
            interrupt_id="email_review",
            message=f"Draft:\n\n{node_input}\n\nApprove? (yes/no)",
        )
        return
    # A response of exactly {"result": value} arrives unwrapped, as value.
    if str(decision).strip().lower() == "yes":
        ctx.route = "approved"
        yield node_input
        return
    ctx.route = "revise"                    # loop back for another draft
    yield node_input


@node
def send(node_input: str, ctx) -> str:
    ctx.state["approved_email"] = node_input
    return "email queued"


approval_wf = Workflow(
    name="email_approval",
    edges=[(START, drafter, human_review, {"approved": send, "revise": drafter})],
)

app = App(
    name="email_app",
    root_agent=approval_wf,
    resumability_config=ResumabilityConfig(is_resumable=True),
)


def request_input_ids(event) -> list[str]:
    """Interrupt ids of any RequestInput pauses carried by this event."""
    return [fc.id for fc in event.get_function_calls() if fc.name == "adk_request_input"]


async def start(runner: Runner) -> str | None:
    """First request: runs until the workflow pauses; returns the interrupt id."""
    await runner.session_service.create_session(app_name="email_app", user_id="u1", session_id="s1")
    message = types.Content(role="user", parts=[types.Part(text="noise-cancelling headphones for office workers")])
    interrupt_id = None
    async for event in runner.run_async(user_id="u1", session_id="s1", new_message=message):
        for call_id in request_input_ids(event):
            interrupt_id = call_id
            print("Workflow paused, waiting for approval.")
    return interrupt_id


async def resume(runner: Runner, interrupt_id: str, answer: str) -> None:
    """Later request (possibly another process): answers the pause."""
    response = types.Content(role="user", parts=[create_request_input_response(interrupt_id, {"result": answer})])
    async for event in runner.run_async(user_id="u1", session_id="s1", new_message=response):
        if event.output is not None:
            print("output:", event.output)


async def main():
    sessions = DatabaseSessionService("sqlite+aiosqlite:///./sessions.db")
    runner = Runner(app=app, session_service=sessions)
    interrupt_id = await start(runner)
    # ... store interrupt_id with your task and show the draft to a person ...
    if interrupt_id:
        await resume(Runner(app=app, session_service=sessions), interrupt_id, "yes")


asyncio.run(main())
```

- Send a `FunctionResponse` that matches the `interrupt_id`: plain text does not resume a paused node. A response of exactly `{"result": value}` reaches the node unwrapped, as `value`; any other dict arrives as is.
- `create_request_input_response` builds that part. It lives in a private module, so pin your ADK version if you rely on it, or build the `types.Part` yourself as in the previous section. The `has_request_input_function_call` and `get_request_input_interrupt_ids` helpers that module shipped up to 2.10 have been removed; read the ID from the event's `adk_request_input` function call, as `request_input_ids` does.
- `DatabaseSessionService` creates its tables on first use (`await sessions.prepare_tables()` does it eagerly). With `ResumabilityConfig(is_resumable=True)` on the `App`, an invocation interrupted by a crash can also be resumed from the session's events.
- Give every node that can be on a routing branch somewhere to go: if a node has routing edges and sets no matching route, the branch ends with a logged warning (`... none were matched by the emitted route(s)`), which is why approval routes to `send` above.

## Gotchas

- Nodes are **Pydantic models** — if you subclass `Node`, annotate fields or they won't serialise.
- `Workflow` is not a `BaseAgent`, and it cannot be an `LlmAgent` sub-agent. As a root it works with `App(root_agent=workflow)`, `Runner(node=workflow, ...)` or `Runner(agent=workflow, ...)`; use `App` when you need plugins or resumability.
- `ctx.run_node(callable)` requires `rerun_on_resume=True` on the calling node.
- `wait_for_output=True` means a node is not complete until it yields an output. A node with it set that finishes without one leaves its successors waiting.
- When a tuple contains only one element (e.g. `(START, single_node)`), you still get a single edge — not sugar for fan-out. Fan-out needs a **nested** tuple: `(START, (a, b))`.
- Nodes are inferred from `edges`; there is no `nodes=` field (an unknown keyword is ignored, not rejected).
- `state_schema` validates `ctx.state` writes — fields on the schema class can be injected directly as node parameters; this is the recommended way to thread typed state through a pipeline.

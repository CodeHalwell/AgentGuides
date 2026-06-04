---
title: "Command, Send & control flow — API reference"
description: "The Command and Send primitives let a node update state, jump to another node, and fan out to many parallel node instances in a single return value — plus Command.PARENT for cross-subgraph routing."
framework: langgraph
language: python
sidebar:
  label: "Ref · Command / Send"
  order: 34
---

# Command, Send & control flow — API reference

Verified against **`langgraph==1.2.4`** (module: `langgraph.types`).

LangGraph's control flow primitives live in `langgraph.types`:

| Symbol | Purpose |
|---|---|
| `Command(update, goto, resume, graph)` | Update state **and/or** jump to another node **and/or** resume an interrupt — all in one return value from a node. |
| `Send(node, arg)` | Dispatch a node with custom state; used from conditional edges for fan-out and from `Command.goto` for dynamic routing. |
| `interrupt(value)` | Pause the current task and surface `value` to the client (resume with `Command(resume=...)`). |
| `Overwrite(value)` | Write directly to a reducing channel, bypassing the reducer. |
| `Interrupt(value, id)` | The dataclass surfaced inside `StateSnapshot.interrupts` (v1.1: `value` and `id` only; older attributes `ns`, `when`, `resumable` were removed in v0.6). |

## Minimal runnable example

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class State(TypedDict):
    messages: list[str]
    next: str


def planner(state: State) -> Command[Literal["writer", "critic", "__end__"]]:
    if len(state["messages"]) >= 3:
        return Command(goto=END)
    if state["messages"] and state["messages"][-1].startswith("draft"):
        return Command(update={"next": "critique"}, goto="critic")
    return Command(update={"next": "write"}, goto="writer")


def writer(state: State) -> Command[Literal["planner"]]:
    return Command(update={"messages": state["messages"] + ["draft v1"]}, goto="planner")


def critic(state: State) -> Command[Literal["planner"]]:
    return Command(update={"messages": state["messages"] + ["critique"]}, goto="planner")


builder = StateGraph(State)
builder.add_node("planner", planner)
builder.add_node("writer", writer)
builder.add_node("critic", critic)
builder.add_edge(START, "planner")

graph = builder.compile()
print(graph.invoke({"messages": [], "next": ""}))
```

Notes:

- No `add_edge` from `planner` to `writer` / `critic` / `END` — the node returns `Command(goto=...)`. Declare `destinations={"writer", "critic"}` on `add_node` only for diagram purposes.
- Type-hinting the return as `Command[Literal["writer", "critic", "__end__"]]` keeps the Mermaid visualization accurate.

## `Command` in full

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class Command(Generic[N], ToolOutputMixin):
    graph:  str | None = None               # target graph ("__parent__" for Command.PARENT)
    update: Any | None = None               # state update (dict, dataclass, Pydantic, tuple list, scalar)
    resume: dict[str, Any] | Any | None = None
    goto:   Send | Sequence[Send | N] | N = ()
    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
```

Any subset of `update`, `resume`, `goto`, `graph` can be set. When a node returns `Command(update={...})` without `goto`, it behaves like returning a dict — the graph's edges decide where to go next.

### `update`

`update` accepts the same shapes as a normal node return:

- `dict` — keys are channel names.
- A list of `(channel, value)` tuples.
- A Pydantic model / dataclass matching the state schema.
- A scalar — written to the `__root__` channel when the state has a root channel.

Reducers apply as usual; wrap a value in `Overwrite(...)` to bypass them.

### `goto`

```python
Command(goto="next_node")                        # single
Command(goto=["fan_out_a", "fan_out_b"])         # multiple (unrelated to Send fan-out)
Command(goto=Send("worker", {"item": x}))        # dispatch a node with custom input
Command(goto=[Send("w", {"i": i}) for i in xs])  # fan-out with Sends
```

Special values:

- `END` → terminate this execution path.
- A node name not in the graph raises `ValueError` at runtime.
- Mixing `Send` and plain names in the same list is allowed.

### `resume`

Used to resume from an `interrupt()`. Two shapes:

```python
Command(resume="a single value")                                # next interrupt gets this value
Command(resume={"interrupt-id-1": "v1", "interrupt-id-2": "v2"})# address by interrupt id
```

See the `interrupt()` section below.

### `graph` / `Command.PARENT`

```python
Command(graph=Command.PARENT, goto="retry", update={"reason": "timeout"})
```

From inside a subgraph node, this routes the command to the **parent** graph — useful for bubbling an error or a handoff signal up to a supervisor.

## `Send`

```python
class Send:
    node: str
    arg:  Any
    timeout: TimeoutPolicy | None   # added in 1.2.x

    def __init__(
        self,
        /,
        node: str,
        arg: Any,
        *,
        timeout: float | timedelta | TimeoutPolicy | None = None,
    ) -> None:
        self.node = node
        self.arg = arg
        self.timeout = TimeoutPolicy.coerce(timeout)  # normalised to TimeoutPolicy | None
```

`Send` packages a node name and a custom state payload. Two places accept it:

1. **Conditional edges**: return one or more `Send`s from the `path` callable.
2. **`Command.goto`**: return `Command(goto=Send("worker", {...}))` from a node.

The receiving node runs with the provided `arg` as its state snapshot for this task. The node is a concrete named node; the sent state can be any subset of the node's input schema.

Equality is structural (`node` + `arg` + `timeout`), and `Send` is hashable.

### Per-task timeout on `Send`

The `timeout` parameter overrides the target node's default timeout for this specific dispatched task. Pass a `float` (seconds), `timedelta`, or `TimeoutPolicy` directly:

```python
import operator
from datetime import timedelta
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, TimeoutPolicy


class BatchState(TypedDict):
    jobs: list[dict]
    results: Annotated[list[str], operator.add]


class JobState(TypedDict):
    id: str
    payload: str
    priority: str


def dispatch(state: BatchState) -> list[Send]:
    """Fan-out jobs; high-priority jobs get a 10-second hard cap, normal gets 60."""
    sends = []
    for job in state["jobs"]:
        timeout = (
            10.0                                     # high-priority: 10 s hard cap
            if job["priority"] == "high"
            else timedelta(seconds=60)               # normal: 60 s hard cap
        )
        sends.append(Send("run_job", job, timeout=timeout))
    return sends


def run_job(state: JobState) -> dict:
    return {"results": [f"done:{state['id']}"]}


builder = StateGraph(BatchState)
builder.add_node("run_job", run_job)
builder.add_conditional_edges(START, dispatch)
builder.add_edge("run_job", END)
graph = builder.compile()
```

For fine-grained control (both run and idle caps), pass a `TimeoutPolicy`:

```python
sends = [
    Send(
        "expensive_node",
        {"item": item},
        timeout=TimeoutPolicy(run_timeout=60.0, idle_timeout=10.0),
    )
    for item in items
]
```

## `interrupt()`

```python
from langgraph.types import interrupt, Command

def ask(state: State) -> dict:
    answer = interrupt({"question": "How old are you?"})
    return {"age": int(answer)}
```

Semantics:

- First execution inside a node raises a `GraphInterrupt` containing an `Interrupt(value, id)`. The graph pauses; the `Interrupt` shows up in `StateSnapshot.interrupts` and in the `__interrupt__` key emitted on `stream_mode="updates"`.
- The client resumes with `graph.invoke(Command(resume="42"), cfg)`. The node **re-runs from the top**, this time `interrupt(...)` returns `"42"`.
- Multiple `interrupt()` calls in one node are matched by order in the current task. Resume values scope to the task, not the graph.
- A checkpointer is **required**. Without one, `interrupt()` raises with no way to resume.

Resume by id when a node has several interrupts:

```python
from langgraph.types import Command
cfg = {"configurable": {"thread_id": "t"}}
# From the streaming output, you saw:
# __interrupt__ = (Interrupt(value=..., id='abc'), Interrupt(value=..., id='def'))
graph.invoke(Command(resume={"abc": "yes", "def": "no"}), cfg)
```

### `Interrupt` dataclass

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any
    id: str
```

Only `value` and `id` are supported. The deprecated `interrupt_id` property still exists but warns. `ns`, `when`, and `resumable` were removed in v0.6 — use `StateSnapshot.interrupts` for structural info.

## `Overwrite`

Writes a value to a reducing channel without applying the reducer:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.types import Overwrite

class S(TypedDict):
    items: Annotated[list[str], operator.add]

def reset(state: S) -> dict:
    return {"items": Overwrite(["start-over"])}
```

Two `Overwrite`s for the same channel in one super-step raise `InvalidUpdateError`.

## Patterns

### 1. Map-reduce with `Send`

```python
from langgraph.types import Send

def dispatch(state: dict) -> list[Send]:
    return [Send("score", {"item": x}) for x in state["items"]]

builder.add_node("score", score_fn)
builder.add_conditional_edges("dispatch", dispatch)
builder.add_edge("score", "aggregate")
builder.add_edge(["dispatch", "score"], "aggregate")   # barrier wait
```

`score` runs once per item with its own state snapshot. Use a reducer on the downstream channel (e.g., `Annotated[list, operator.add]`) so results concatenate.

### 2. Supervisor routing without edges

```python
from typing import Literal
from langgraph.types import Command

def supervisor(state: dict) -> Command[Literal["researcher", "writer", "__end__"]]:
    if not state.get("notes"):
        return Command(goto="researcher")
    if not state.get("draft"):
        return Command(goto="writer", update={"phase": "drafting"})
    return Command(goto=END)

builder.add_node("supervisor", supervisor, destinations=("researcher", "writer", END))
```

`destinations=` feeds the diagram only; the supervisor's typed return drives execution.

### 3. Subgraph bubbling to parent

```python
def worker(state: dict) -> Command:
    if state["escalate"]:
        return Command(
            graph=Command.PARENT,
            goto="human_review",
            update={"reason": state["reason"]},
        )
    return Command(update={"done": True})
```

Inside a compiled subgraph `worker` can hand control back to the parent graph's `human_review` node while carrying state.

### 4. Tool-authored commands

Any `@tool` that returns a `Command` is treated as control flow by `ToolNode`. Example:

```python
from langchain_core.tools import tool
from langgraph.types import Command

@tool
def transfer_to_refunds(reason: str) -> Command:
    """Hand this conversation to the refunds agent."""
    return Command(goto="refunds_agent", update={"transfer_reason": reason})
```

`ToolNode` unpacks the `Command` into a state update plus goto.

### 5. Interrupt + resume + update

```python
from langgraph.types import interrupt, Command

def approve(state):
    decision = interrupt({"approve?": state["proposal"]})
    if decision == "yes":
        return Command(goto="execute", update={"approved_by": "human"})
    return Command(goto="cancel")

# Client:
graph.stream(initial, cfg)                              # emits __interrupt__
graph.invoke(Command(resume="yes"), cfg)                # continues into "execute"
```

### 6. `Send` with a per-task timeout

Pass a `timeout=` to `Send` to cap how long an individual parallel task may run. A plain `float` is a hard wall-clock limit (`run_timeout`); pass a `TimeoutPolicy` for idle-based cancellation.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, TimeoutPolicy, RetryPolicy

class Scrape(TypedDict):
    urls: list[str]
    results: Annotated[list[str], operator.add]


def scrape_page(state: dict) -> dict:
    """Scrape one URL — runs per-Send with its own timeout."""
    url = state["url"]
    # ... real HTTP fetch here ...
    return {"results": [f"content:{url}"]}


def dispatch(state: Scrape) -> list[Send]:
    return [
        Send(
            "scrape_page",
            {"url": url, "results": []},
            # Each individual task gets 10 s wall-clock; retry up to 2 extra times
            timeout=10.0,
        )
        for url in state["urls"]
    ]


builder = StateGraph(Scrape)
builder.add_node(
    "scrape_page",
    scrape_page,
    retry_policy=RetryPolicy(max_attempts=3, retry_on=TimeoutError),
)
builder.add_conditional_edges(START, dispatch)
builder.add_edge("scrape_page", END)

graph = builder.compile()
result = graph.invoke({"urls": ["https://a.com", "https://b.com"], "results": []})
print(result["results"])
```

### 7. Complete multi-agent handoff with `Command.PARENT`

A supervisor graph runs two specialised subgraphs. Each subgraph can escalate back to the supervisor using `Command(graph=Command.PARENT, goto="supervisor")`.

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


# ── Shared state used by both the parent and subgraphs ──────────────────────

class SharedState(TypedDict):
    task: str
    output: str
    escalated: bool


# ── Subgraph A: researcher ──────────────────────────────────────────────────

def researcher_node(state: SharedState) -> Command[Literal["__end__"]]:
    # Simulate research; escalate if topic is too complex
    if "complex" in state["task"]:
        return Command(
            graph=Command.PARENT,   # send to the parent supervisor
            goto="supervisor",
            update={"escalated": True, "output": "Research: escalated — topic too complex"},
        )
    return Command(
        update={"output": f"Research done: {state['task']}"},
        goto=END,
    )


researcher = (
    StateGraph(SharedState)
    .add_node("researcher_node", researcher_node, destinations=["__end__"])
    .add_edge(START, "researcher_node")
    .compile()
)


# ── Subgraph B: writer ──────────────────────────────────────────────────────

def writer_node(state: SharedState) -> dict:
    return {"output": f"Draft written for: {state['task']}"}


writer = (
    StateGraph(SharedState)
    .add_node("writer_node", writer_node)
    .add_edge(START, "writer_node")
    .add_edge("writer_node", END)
    .compile()
)


# ── Parent supervisor ───────────────────────────────────────────────────────

class SupervisorState(SharedState):
    phase: str


def supervisor(state: SupervisorState) -> Command[Literal["researcher", "writer", "__end__"]]:
    if state.get("escalated"):
        # Researcher escalated — handle manually and finish
        return Command(
            update={"output": "Supervisor resolved escalation.", "phase": "done"},
            goto=END,
        )
    if state["phase"] == "start":
        return Command(update={"phase": "research"}, goto="researcher")
    if state["phase"] == "research":
        return Command(update={"phase": "write"}, goto="writer")
    return Command(goto=END)


parent = StateGraph(SupervisorState)
parent.add_node("supervisor", supervisor, destinations=["researcher", "writer", "__end__"])
parent.add_node("researcher", researcher)
parent.add_node("writer", writer)
parent.add_edge(START, "supervisor")

graph = parent.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "multi-1"}}

# Normal task
result = graph.invoke({"task": "Write about LangGraph", "output": "", "escalated": False, "phase": "start"}, cfg)
print(result["output"])   # "Draft written for: Write about LangGraph"

# Complex task triggers escalation from researcher → parent supervisor
cfg2 = {"configurable": {"thread_id": "multi-2"}}
result2 = graph.invoke({"task": "complex quantum theory", "output": "", "escalated": False, "phase": "start"}, cfg2)
print(result2["output"])   # "Supervisor resolved escalation."
```

### 8. Fan-out then fan-in with `Send` and a barrier edge

`Send` from a conditional edge fans out to N parallel tasks. A barrier edge (`add_edge(["source1", "source2"], "target")`) waits for all of them before running the aggregation node.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class Pipeline(TypedDict):
    documents: list[str]
    scores: Annotated[list[dict], operator.add]   # accumulates results from all workers


def score_document(state: dict) -> dict:
    """Score one document — runs once per Send."""
    doc = state["document"]
    score = len(doc) / 100.0   # replace with a real scoring call
    return {"scores": [{"doc": doc[:30], "score": score}]}


def aggregate(state: Pipeline) -> dict:
    avg = sum(s["score"] for s in state["scores"]) / len(state["scores"])
    best = max(state["scores"], key=lambda s: s["score"])
    print(f"Scored {len(state['scores'])} documents. avg={avg:.2f}, best={best['doc']!r}")
    return {}


builder = StateGraph(Pipeline)
builder.add_node("score_document", score_document)
builder.add_node("aggregate", aggregate)

# Fan out: one Send per document, all run in parallel
builder.add_conditional_edges(
    START,
    lambda s: [Send("score_document", {"document": d, "scores": []}) for d in s["documents"]],
)
# Fan in: barrier waits for all score_document tasks
builder.add_edge("score_document", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()
graph.invoke({
    "documents": ["Short doc", "A much longer document with more content", "Medium length document here"],
    "scores": [],
})
```

## Gotchas

- **`Command(goto="name")` bypasses explicit edges.** A node that returns a Command will follow the command's goto even if you called `add_edge("node", "next")`. Pick one style per node.
- **`Command.goto` does not accept `str` for subgraph namespaces.** Always use a plain node name at the current graph level; cross-graph jumps use `graph=Command.PARENT`.
- **The type parameter on `Command[Literal[...]]` is for the visualizer.** It doesn't narrow to runtime errors.
- **`Send(node, arg)` ignores the main state.** `arg` *is* the snapshot for the target node's run. If you need context, stuff it into `arg`.
- **Equality compares `arg` too.** Two `Send("x", {...})` with unhashable dicts are hashable at the `Send` level but raise if you stick them in a set without care — dict compares structurally, hash uses tuple of `(node, arg)`.
- **A node that returns `Command(graph=Command.PARENT)` outside a subgraph raises.** Only valid when the node runs inside a compiled subgraph used by a parent.
- **`update=` in a `Command` still goes through reducers.** Use `Overwrite(...)` in the `update` values if you need to replace a reducing channel.
- **Resuming an interrupt re-runs the node from the top.** Make side effects idempotent or put them in `@task`s.

## Breaking changes

| Version | Change |
|---|---|
| 1.0 | `Command` is the canonical way for a node/tool to return control-flow intent. Returning a dict still works for pure state updates. |
| 0.6 | `Interrupt.ns`, `Interrupt.when`, `Interrupt.resumable` removed. `Interrupt.interrupt_id` deprecated in favor of `Interrupt.id`. |
| 0.4 | `Interrupt.id` introduced as a property, supporting resume-by-id via `Command(resume={id: value})`. |
| 0.2.24 | `RetryPolicy`, `CachePolicy`, `Interrupt` first exported from `langgraph.types`. |

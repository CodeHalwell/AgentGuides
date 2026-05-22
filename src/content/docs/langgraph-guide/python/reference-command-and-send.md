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

Verified against **`langgraph==1.2.1`** (module: `langgraph.types`).

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
    def __init__(self, /, node: str, arg: Any) -> None: ...
```

`Send` packages a node name and a custom state payload. Two places accept it:

1. **Conditional edges**: return one or more `Send`s from the `path` callable.
2. **`Command.goto`**: return `Command(goto=Send("worker", {...}))` from a node.

The receiving node runs with the provided `arg` as its state snapshot for this task. The node is a concrete named node; the sent state can be any subset of the node's input schema.

Equality is structural (`node` + `arg`), and `Send` is hashable.

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

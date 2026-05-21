---
title: "IsLastStep, RemainingSteps — Managed values API reference"
description: "Built-in managed state values that LangGraph injects automatically: IsLastStep detects when a graph is on its final step, RemainingSteps counts how many steps are left."
framework: langgraph
language: python
sidebar:
  label: "Ref · Managed values"
  order: 38
---

# Managed values — `IsLastStep` & `RemainingSteps`

Verified against **`langgraph==1.2.0`** (module: `langgraph.managed.is_last_step`).

**Managed values** are special state-field annotations that LangGraph fills in automatically from the Pregel executor's scratchpad. They are declared in the state schema like any other field, but the graph — not node code — writes them at each step.

Two managed values ship with LangGraph out of the box:

| Type alias | Module | Type | Value |
|---|---|---|---|
| `IsLastStep` | `langgraph.managed.is_last_step` | `bool` | `True` when the current step is `step == (recursion_limit - 1)` |
| `RemainingSteps` | `langgraph.managed.is_last_step` | `int` | `recursion_limit - current_step` |

Both are declared as `Annotated[T, ManagedValueManager]` type aliases. The `Annotated` wrapper is the annotation; you use the alias directly in your `TypedDict` / dataclass / Pydantic schema.

## Imports at a glance

```python
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps
```

## Minimal runnable example

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps


class State(TypedDict):
    count: int
    is_last: IsLastStep        # bool — injected by the graph
    remaining: RemainingSteps  # int  — injected by the graph


def worker(state: State) -> dict:
    print(f"step count={state['count']} last={state['is_last']} left={state['remaining']}")
    if state["is_last"]:
        return {"count": state["count"]}   # graceful stop on recursion limit
    return {"count": state["count"] + 1}


def router(state: State) -> str:
    return END if state["is_last"] or state["count"] >= 5 else "worker"


builder = StateGraph(State)
builder.add_node("worker", worker)
builder.add_edge(START, "worker")
builder.add_conditional_edges("worker", router)

graph = builder.compile()
graph.invoke({"count": 0})
# Prints something like:
#   step count=0 last=False left=24
#   step count=1 last=False left=23
#   ...
#   step count=5 last=False left=19
```

> **Do not write to `IsLastStep` or `RemainingSteps`.** They are read-only managed values. Any node return that includes these keys is silently ignored — the graph writes the correct value.

## `IsLastStep`

```python
IsLastStep = Annotated[bool, IsLastStepManager]
```

`IsLastStep` is `True` exactly when `current_step == recursion_limit - 1`. Use it to detect that the graph is about to hit its recursion limit so you can return a graceful partial result instead of raising `GraphRecursionError`.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep


class AgentState(TypedDict):
    messages: list[str]
    is_last: IsLastStep


def agent(state: AgentState) -> dict:
    if state["is_last"]:
        # We're about to exhaust the recursion limit — return what we have
        return {"messages": state["messages"] + ["[truncated: recursion limit reached]"]}
    # Normal processing
    new_message = call_llm(state["messages"])
    return {"messages": state["messages"] + [new_message]}


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1] if state["messages"] else ""
    if state["is_last"] or last_msg.startswith("FINAL"):
        return END
    return "agent"


builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)

graph = builder.compile()
result = graph.invoke({"messages": ["user: hello"]})
```

### How the step count works

The Pregel executor tracks `step` (starting at 0) and `stop` (the recursion limit, default 25). `IsLastStep` returns `step == stop - 1`.

- Default recursion limit: **25** steps.
- Override per call: `graph.invoke(input, {"recursion_limit": 50})`.
- `IsLastStep` becomes `True` at step **24** with the default limit, or step **49** with `recursion_limit=50`.

## `RemainingSteps`

```python
RemainingSteps = Annotated[int, RemainingStepsManager]
```

`RemainingSteps` returns `stop - step` — how many steps are left before the recursion limit fires. It decrements by 1 each step.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import RemainingSteps


class PipelineState(TypedDict):
    items: list[str]
    processed: list[str]
    steps_left: RemainingSteps


def process_one(state: PipelineState) -> dict:
    remaining = state["steps_left"]
    if remaining <= 2:
        # Not enough steps to process everything — flush remaining items
        return {"processed": state["processed"] + [f"[skipped:{len(state['items'])} items]"]}
    first, *rest = state["items"]
    return {
        "items": rest,
        "processed": state["processed"] + [first.upper()],
    }


def router(state: PipelineState) -> str:
    if not state["items"] or state["steps_left"] <= 1:
        return END
    return "process_one"


builder = StateGraph(PipelineState)
builder.add_node("process_one", process_one)
builder.add_edge(START, "process_one")
builder.add_conditional_edges("process_one", router)

graph = builder.compile()
result = graph.invoke({
    "items": ["a", "b", "c", "d"],
    "processed": [],
})
print(result["processed"])   # ['A', 'B', 'C', 'D'] (if steps available)
```

## Patterns

### 1. ReAct agent with graceful truncation

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import IsLastStep


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_last: IsLastStep


def call_agent(state: AgentState) -> dict:
    if state["is_last"]:
        return {"messages": [AIMessage(content="I've reached my step limit. Here's what I found so far: ...")]}
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def router(state: AgentState) -> str:
    last = state["messages"][-1]
    if state["is_last"]:
        return END
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", call_agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", router)
builder.add_edge("tools", "agent")

graph = builder.compile()
```

### 2. Multi-phase pipeline that skips phases when steps are short

```python
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


PHASES = ["plan", "research", "draft", "review", "finalize"]


class WriterState(TypedDict):
    topic: str
    phase: int
    output: str
    steps: RemainingSteps


def run_phase(state: WriterState) -> dict:
    phase_name = PHASES[state["phase"]]
    if state["steps"] <= (len(PHASES) - state["phase"]):
        # Not enough steps — skip to finalize
        return {"phase": len(PHASES) - 1, "output": state["output"] + f"\n[{phase_name} skipped]"}
    result = run_phase_logic(phase_name, state["topic"])
    return {"output": state["output"] + f"\n{phase_name}: {result}", "phase": state["phase"] + 1}


def router(state: WriterState) -> str:
    if state["phase"] >= len(PHASES):
        return END
    return "phase"


builder = StateGraph(WriterState)
builder.add_node("phase", run_phase)
builder.add_edge(START, "phase")
builder.add_conditional_edges("phase", router)

graph = builder.compile()
```

### 3. `IsLastStep` in a Pydantic state schema

```python
from pydantic import BaseModel
from langgraph.managed.is_last_step import IsLastStep


class State(BaseModel):
    data: str = ""
    is_last: IsLastStep = False   # default False; graph overwrites each step


def node(state: State) -> dict:
    if state.is_last:
        return {"data": state.data + " [FINAL]"}
    return {"data": state.data + " more"}
```

Using Pydantic with managed values works the same as `TypedDict`: declare the field with the type alias and provide a default value. The graph overwrites it at each step regardless of the default.

## How managed values work (internals)

Each managed value is a subclass of `ManagedValue[T]` with a single `get(scratchpad)` static method. The Pregel executor calls `get` before every step and injects the return value into the state the node sees — but does **not** persist it to a channel (so it never appears in checkpoints or reducer chains).

```python
# Simplified internals (don't import these directly):
from langgraph._internal._scratchpad import PregelScratchpad

class IsLastStepManager(ManagedValue[bool]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> bool:
        return scratchpad.step == scratchpad.stop - 1

class RemainingStepsManager(ManagedValue[int]):
    @staticmethod
    def get(scratchpad: PregelScratchpad) -> int:
        return scratchpad.stop - scratchpad.step
```

`PregelScratchpad.stop` is the recursion limit; `PregelScratchpad.step` is the 0-indexed current step.

## Gotchas

- **Managed values are read-only.** Writing to `is_last` or `steps_left` from a node return has no effect — the graph overwrites them before the next node runs.
- **Managed values do not appear in checkpoints.** They are reconstructed from the scratchpad at runtime. You cannot read `is_last` from a `StateSnapshot` or `get_state_history` result.
- **Always provide a default in the schema.** TypedDict requires all fields to be provided in the initial `invoke` input unless they have defaults. Since managed values are never in the initial input, declare them with a default that matches their type:
  ```python
  class State(TypedDict, total=False):
      is_last: IsLastStep       # total=False makes it optional in invoke input
  # or use a dataclass/Pydantic with default:
  class State(BaseModel):
      is_last: IsLastStep = False
  ```
- **`recursion_limit` is per-invoke, not per-graph.** Different calls to `graph.invoke` can use different limits. `IsLastStep` tracks the limit that was active when the run started.
- **Step counter resets on each `invoke` call.** Checkpointers save the channel values but not the step counter. A new `invoke` on an existing thread starts the step counter at 0 again.

## Breaking changes

| Version | Change |
|---|---|
| 1.2 | `RemainingSteps` added alongside the existing `IsLastStep`. Both re-exported from `langgraph.managed.is_last_step`. |
| 1.0 | `IsLastStep` and `RemainingSteps` moved from `langgraph.managed` to `langgraph.managed.is_last_step`; old import path still re-exported. |
| 0.3 | `IsLastStep` introduced as a managed value for recursion-limit detection. |

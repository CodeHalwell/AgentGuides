---
title: "Class Deep Dives Vol. 3 — 10 API Classes (v1.2.16)"
description: "Source-verified deep dives for BinaryOperatorAggregate, Topic, EphemeralValue, LastValue/LastValueAfterFinish, GraphCallbackHandler/events, ToolNode, MessagesState/add_messages, UIMessage/RemoveUIMessage, Runtime/ExecutionInfo/RunControl, and HumanResponse — each with multiple runnable code examples. Verified against langgraph==1.2.16."
framework: langgraph
language: python
sidebar:
  label: "Ref · Class Deep Dives Vol. 3"
  order: 45
---

# Class Deep Dives Vol. 3 — 10 API Classes (v1.2.16)

> **Source-verified against `langgraph==1.2.16`** — October 2026.

This third volume dives into ten classes spanning channels, observability, the prebuilt
tool layer, message state, UI streaming, and the unified run-context API. Each section
shows the complete constructor or type signature straight from source, explains every
field, and gives **multiple runnable examples** covering the real-world usage patterns
most developers encounter.

**Classes covered:**

| # | Class / Symbol | Module |
|---|---|---|
| 1 | `BinaryOperatorAggregate` | `langgraph.channels.binop` |
| 2 | `Topic` | `langgraph.channels.topic` |
| 3 | `EphemeralValue` | `langgraph.channels.ephemeral_value` |
| 4 | `LastValue` + `LastValueAfterFinish` | `langgraph.channels.last_value` |
| 5 | `GraphCallbackHandler` + `GraphInterruptEvent` + `GraphResumeEvent` | `langgraph.callbacks` |
| 6 | `ToolNode` | `langgraph.prebuilt.tool_node` |
| 7 | `MessagesState` + `add_messages` | `langgraph.graph.message` |
| 8 | `UIMessage` + `RemoveUIMessage` + `push_ui_message` | `langgraph.graph.ui` |
| 9 | `Runtime` + `ExecutionInfo` + `RunControl` | `langgraph.runtime` |
| 10 | `HumanResponse` | `langgraph.prebuilt.interrupt` |

---

## 1. `BinaryOperatorAggregate` — Reduce-based State Channel

**Module:** `langgraph.channels.binop`  
**Import:** `from langgraph.channels.binop import BinaryOperatorAggregate`  
**Status:** Stable — used internally by `Annotated[T, operator.add]` and similar patterns.

```python
class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    def __init__(
        self,
        typ: type[Value],
        operator: Callable[[Value, Value], Value],
    ): ...
```

### What it does

`BinaryOperatorAggregate` accumulates state by applying a binary `operator` to the
current value and each incoming write. It is the channel that backs the very common
`Annotated[list[str], operator.add]` pattern: whenever multiple nodes write to the
same key in the same superstep, LangGraph folds them together rather than raising a
conflict error.

**`Overwrite` escape hatch** — any single write may be wrapped in `Overwrite(value=…)` to
completely replace the accumulated value instead of reducing it. At most one `Overwrite`
is allowed per superstep; a second raises `InvalidUpdateError`.

**Type normalisation** — abstract collection types (`Sequence`, `MutableMapping`, etc.) are
replaced with their concrete equivalents (`list`, `dict`, `set`) when constructing the
initial empty value.

### Example 1a — Classic append-only list with `operator.add`

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    steps: Annotated[list[str], operator.add]

def step_a(state: State) -> dict:
    return {"steps": ["step_a"]}

def step_b(state: State) -> dict:
    return {"steps": ["step_b"]}

builder = StateGraph(State)
builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", END)
builder.add_edge("b", END)
graph = builder.compile()

result = graph.invoke({"steps": []})
print(result["steps"])
# ["step_a", "step_b"]  or  ["step_b", "step_a"]  (parallel — order not guaranteed)
```

### Example 1b — Numeric accumulator across parallel branches

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

class State(TypedDict):
    numbers: list[int]
    total: Annotated[int, operator.add]

class WorkerState(TypedDict):
    number: int

def add_number(state: WorkerState) -> dict:
    return {"total": state["number"]}

def fan_out(state: State) -> list[Send]:
    return [Send("worker", {"number": n}) for n in state["numbers"]]

builder = StateGraph(State)
builder.add_node("worker", add_number)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("worker", END)
graph = builder.compile()

result = graph.invoke({"numbers": [10, 20, 30], "total": 0})
print(result["total"])  # 60
```

### Example 1c — Using `Overwrite` to reset accumulated state

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Overwrite

class State(TypedDict):
    log: Annotated[list[str], operator.add]

def append_node(state: State) -> dict:
    return {"log": ["item"]}

def reset_node(state: State) -> dict:
    # Overwrite replaces the entire accumulated value
    return {"log": Overwrite(value=["fresh_start"])}

builder = StateGraph(State)
builder.add_node("append", append_node)
builder.add_node("reset", reset_node)
builder.add_edge(START, "append")
builder.add_edge("append", "reset")
builder.add_edge("reset", END)
graph = builder.compile()

result = graph.invoke({"log": ["existing"]})
print(result["log"])  # ["fresh_start"]
```

### Example 1d — Custom set-union reducer

```python
from typing import Annotated, FrozenSet
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph

def union(a: frozenset, b: frozenset) -> frozenset:
    return a | b

class State(TypedDict):
    seen_ids: Annotated[frozenset, union]

def tag_a(state: State) -> dict:
    return {"seen_ids": frozenset({"a", "b"})}

def tag_b(state: State) -> dict:
    return {"seen_ids": frozenset({"b", "c"})}

builder = StateGraph(State)
builder.add_node("a", tag_a)
builder.add_node("b", tag_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", END)
builder.add_edge("b", END)
graph = builder.compile()

result = graph.invoke({"seen_ids": frozenset()})
print(result["seen_ids"])  # frozenset({'a', 'b', 'c'})
```

---

## 2. `Topic` — PubSub Fan-In Channel

**Module:** `langgraph.channels.topic`  
**Import:** `from langgraph.channels.topic import Topic`  
**Status:** Stable.

```python
class Topic(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value], list[Value]]):
    def __init__(
        self,
        typ: type[Value],
        accumulate: bool = False,
    ): ...
```

### What it does

`Topic` is a multi-producer, single-consumer channel. Multiple nodes can write values
(or lists of values) to a `Topic` key in the same superstep; readers always receive a
**`Sequence[Value]`** — the concatenation of all writes.

The `accumulate` flag controls retention across supersteps:

- `accumulate=False` *(default)* — values are cleared after each superstep (ephemeral
  fan-in: see which nodes ran in this step).
- `accumulate=True` — values persist across supersteps (growing log of all events
  delivered through this channel).

### Example 2a — Ephemeral multi-node event bus

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    # Each superstep collects all published events then clears
    events: Annotated[Sequence[str], Topic(str)]

def producer_a(state: State) -> dict:
    return {"events": "event_from_a"}

def producer_b(state: State) -> dict:
    return {"events": ["event_b1", "event_b2"]}

def consumer(state: State) -> dict:
    print("Received:", list(state["events"]))
    return {}

builder = StateGraph(State)
builder.add_node("a", producer_a)
builder.add_node("b", producer_b)
builder.add_node("consumer", consumer)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", "consumer")
builder.add_edge("b", "consumer")
builder.add_edge("consumer", END)
graph = builder.compile()
graph.invoke({"events": []})
# Received: ['event_from_a', 'event_b1', 'event_b2']
```

### Example 2b — Accumulating audit log (`accumulate=True`)

```python
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.channels.topic import Topic
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    # Grows across runs — never cleared
    audit_log: Annotated[Sequence[str], Topic(str, accumulate=True)]
    step: int

def run_step(state: State) -> dict:
    entry = f"step_{state['step']}_completed"
    return {"audit_log": entry, "step": state["step"] + 1}

builder = StateGraph(State)
builder.add_node("run", run_step)
builder.add_edge(START, "run")
builder.add_edge("run", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "audit-demo"}}
graph.invoke({"audit_log": [], "step": 1}, config)
graph.invoke({"step": 2}, config)
graph.invoke({"step": 3}, config)

snap = graph.get_state(config)
print(list(snap.values["audit_log"]))
# ['step_1_completed', 'step_2_completed', 'step_3_completed']
```

---

## 3. `EphemeralValue` — Single-Step Transient Channel

**Module:** `langgraph.channels.ephemeral_value`  
**Import:** `from langgraph.channels.ephemeral_value import EphemeralValue`  
**Status:** Stable — used internally for `__interrupt__` and similar transient signals.

```python
class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    def __init__(
        self,
        typ: Any,
        guard: bool = True,   # raise if more than one value written per step
    ): ...
```

### What it does

`EphemeralValue` stores the last value written to it in a superstep, then **clears it
automatically** at the start of the next superstep. It is the right channel when you
need to pass a signal between two adjacent nodes that should not persist into the
checkpoint as permanent state.

- `guard=True` *(default)*: raises `InvalidUpdateError` if more than one write arrives
  in the same superstep. Use this for strict single-writer invariants.
- `guard=False`: silently keeps the last write when multiple arrive. Useful for
  advisory signals where any write is fine.

### Example 3a — Transient trigger signal

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    data: str
    # trigger is only visible in the superstep it was written; gone from checkpoint
    trigger: Annotated[str | None, EphemeralValue(str)]

def emitter(state: State) -> dict:
    return {"trigger": "process_now"}

def handler(state: State) -> dict:
    if state.get("trigger"):
        print(f"Handling trigger: {state['trigger']}")
        return {"data": f"processed:{state['data']}"}
    return {}

builder = StateGraph(State)
builder.add_node("emitter", emitter)
builder.add_node("handler", handler)
builder.add_edge(START, "emitter")
builder.add_edge("emitter", "handler")
builder.add_edge("handler", END)
graph = builder.compile()

result = graph.invoke({"data": "raw", "trigger": None})
print(result["data"])            # processed:raw
print(result.get("trigger"))    # None  — channel absent from result when cleared
```

### Example 3b — Passing per-step context without polluting state

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    messages: list[str]
    # Per-invocation metadata — not accumulated across steps
    request_id: Annotated[str | None, EphemeralValue(str, guard=False)]

def start_node(state: State) -> dict:
    import uuid
    return {"request_id": str(uuid.uuid4())}

def process_node(state: State) -> dict:
    rid = state.get("request_id", "none")
    return {"messages": [f"[{rid}] processed"]}

builder = StateGraph(State)
builder.add_node("start", start_node)
builder.add_node("process", process_node)
builder.add_edge(START, "start")
builder.add_edge("start", "process")
builder.add_edge("process", END)
graph = builder.compile()
result = graph.invoke({"messages": [], "request_id": None})
print(result["messages"])  # ['[<uuid>] processed']
```

---

## 4. `LastValue` + `LastValueAfterFinish`

**Module:** `langgraph.channels.last_value`  
**Imports:**
```python
from langgraph.channels.last_value import LastValue, LastValueAfterFinish
```

### 4a. `LastValue` — The Default Channel Type

```python
class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    def __init__(self, typ: Any, key: str = ""): ...
```

`LastValue` is the channel LangGraph uses for every plain (non-`Annotated`) key in a
`TypedDict` state schema. It stores exactly one value and **raises `InvalidUpdateError`
if more than one write arrives in the same superstep**.

This is a deliberate safeguard: if two parallel nodes both try to set `state["result"]`
in the same step, LangGraph surfaces the conflict rather than silently picking one.
The fix is to annotate the key with a reducer (e.g. `Annotated[list[str], operator.add]`).

### Example 4a — Observing the single-write constraint

```python
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph

class State(TypedDict):
    result: str  # backed by LastValue — at most one writer per superstep

def writer_a(state: State) -> dict:
    return {"result": "from_a"}

def writer_b(state: State) -> dict:
    return {"result": "from_b"}

builder = StateGraph(State)
builder.add_node("a", writer_a)
builder.add_node("b", writer_b)
# Running a and b in the SAME superstep would raise InvalidUpdateError.
# Chain them sequentially to avoid the conflict:
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
graph = builder.compile()

result = graph.invoke({"result": ""})
print(result["result"])  # "from_b"  — b's write wins (last in chain)
```

### 4b. `LastValueAfterFinish` — Barrier Channel

```python
class LastValueAfterFinish(Generic[Value], BaseChannel[Value, Value, tuple[Value, bool]]):
    def __init__(self, typ: Any, key: str = ""): ...
```

`LastValueAfterFinish` acts as a barrier: it accepts writes during a superstep but
**does not expose the value to readers until `finish()` is called** on the channel
at the end of the step. Once read via `get()`, it clears itself (one-shot).

This is the channel behind `NamedBarrierValue` and similar synchronisation primitives
where a value should only flow to downstream nodes after a specific checkpoint event.

### Example 4c — Inspecting `LastValue` from a checkpoint snapshot

```python
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    counter: int
    message: str

def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1, "message": f"step_{state['counter']+1}"}

builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "lv-demo"}}
graph.invoke({"counter": 0, "message": ""}, config)
graph.invoke({"counter": 1, "message": ""}, config)

snap = graph.get_state(config)
print(snap.values["counter"])  # 2
print(snap.values["message"])  # "step_2"
```

---

## 5. `GraphCallbackHandler` + `GraphInterruptEvent` + `GraphResumeEvent`

**Module:** `langgraph.callbacks`  
**Imports:**
```python
from langgraph.callbacks import (
    GraphCallbackHandler,
    GraphInterruptEvent,
    GraphResumeEvent,
    GraphLifecycleStatus,
)
```

### What it does

`GraphCallbackHandler` is a subclass of LangChain's `BaseCallbackHandler` that receives
**LangGraph-specific lifecycle events** — specifically when a graph pauses at an interrupt
or resumes from a checkpoint. Only handlers that inherit from `GraphCallbackHandler`
receive these events; generic `BaseCallbackHandler` subclasses do not.

Pass instances through `config["callbacks"]` when invoking a graph.

```python
class GraphInterruptEvent:      # frozen dataclass
    run_id:        UUID | None
    status:        GraphLifecycleStatus  # "interrupt_before" | "interrupt_after" | ...
    checkpoint_id: str
    checkpoint_ns: tuple[str, ...]
    interrupts:    tuple[Interrupt, ...]  # the interrupt payloads

class GraphResumeEvent:         # frozen dataclass
    run_id:        UUID | None
    status:        GraphLifecycleStatus
    checkpoint_id: str
    checkpoint_ns: tuple[str, ...]

# GraphLifecycleStatus values:
# "input" | "pending" | "done" | "interrupt_before" | "interrupt_after" | "out_of_steps"
```

### Example 5a — Logging every interrupt and resume

```python
from typing import Any
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent, GraphResumeEvent
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class AuditHandler(GraphCallbackHandler):
    def __init__(self):
        super().__init__()
        self.events: list[str] = []

    def on_interrupt(self, event: GraphInterruptEvent) -> Any:
        ids = [str(i.value) for i in event.interrupts]
        self.events.append(f"INTERRUPT checkpoint={event.checkpoint_id[:8]} values={ids}")

    def on_resume(self, event: GraphResumeEvent) -> Any:
        self.events.append(f"RESUME    checkpoint={event.checkpoint_id[:8]} status={event.status}")

class State(dict): pass

def human_step(state: dict) -> dict:
    answer = interrupt("Please approve this action")
    return {"approved": answer}

builder = StateGraph(dict)
builder.add_node("ask", human_step)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

handler = AuditHandler()
config = {"configurable": {"thread_id": "cb-demo"}, "callbacks": [handler]}

# First run — will pause at the interrupt
try:
    graph.invoke({"approved": None}, config)
except Exception:
    pass  # GraphInterrupt pauses execution

# Resume with an answer
graph.invoke(Command(resume="yes"), config)

for event in handler.events:
    print(event)
# INTERRUPT checkpoint=<id>     values=["Please approve this action"]
# RESUME    checkpoint=<id>     status=pending
```

### Example 5b — Subgraph-aware namespace tracking

```python
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent

class NamespaceLogger(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        depth = len(event.checkpoint_ns)
        prefix = "  " * depth
        path = " > ".join(event.checkpoint_ns) if event.checkpoint_ns else "(root)"
        print(f"{prefix}Interrupted in: {path}")
```

### Example 5c — Conditional alerting on `out_of_steps`

```python
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent

class StepLimitAlerter(GraphCallbackHandler):
    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        if event.status == "out_of_steps":
            print(f"[ALERT] Graph hit recursion limit! checkpoint={event.checkpoint_id}")
```

---

## 6. `ToolNode` — Prebuilt Tool Executor

**Module:** `langgraph.prebuilt.tool_node`  
**Import:** `from langgraph.prebuilt import ToolNode`  
**Status:** Stable.

```python
class ToolNode(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[BaseTool | Callable],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool | str | Callable[..., str]
            | type[Exception] | tuple[type[Exception], ...] = <default>,
        messages_key: str = "messages",
        wrap_tool_call: ToolCallWrapper | None = None,
        awrap_tool_call: AsyncToolCallWrapper | None = None,
    ): ...
```

### What it does

`ToolNode` is the standard prebuilt node for executing LLM tool calls inside a
`StateGraph`. Given a list of messages where the last `AIMessage` contains `tool_calls`,
it runs each tool (in parallel using a thread-pool executor), and returns `ToolMessage`
objects that carry the results back to the messages list.

**Input formats** — `ToolNode` accepts three different shapes:

1. **Dict with `messages` key** — the standard `MessagesState` format
2. **List of messages** — the last `AIMessage` must have `tool_calls`
3. **List of raw tool call dicts** — `[{"name": "…", "args": {…}, "id": "…", "type": "tool_call"}]`

**Error handling** — the `handle_tool_errors` parameter is very flexible:

| Value | Effect |
|---|---|
| `True` | Catch everything, return default error `ToolMessage` |
| `False` | Re-raise all exceptions |
| `"custom error"` | Catch everything, return this fixed string |
| `ValueError` | Only catch `ValueError` |
| `(ValueError, KeyError)` | Only catch these exception types |
| `lambda e: str(e)` | Catch all; call the lambda to format the error string |

### Example 6a — Minimal ReAct-style tool executor

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Sunny and 22°C in {city}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression."""
    return str(eval(expression, {"__builtins__": {}}))

tools = [get_weather, calculator]
tool_node = ToolNode(tools)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

builder = StateGraph(State)
builder.add_node("tools", tool_node)
# add_node("llm", llm_node) — wire your LLM here
builder.add_edge(START, "tools")
builder.add_edge("tools", END)
graph = builder.compile()
```

### Example 6b — Custom error handler per exception type

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

class DatabaseError(Exception):
    pass

@tool
def fetch_record(record_id: str) -> str:
    """Fetch a record from the database."""
    if record_id == "missing":
        raise DatabaseError("Record not found")
    return f"Record {record_id}: data"

def format_db_error(e: DatabaseError) -> str:
    return f"Database unavailable: {e}. Try again with a different ID."

# Only DatabaseError is caught; other exceptions propagate
tool_node = ToolNode(
    [fetch_record],
    handle_tool_errors=(DatabaseError,),
)

# Or use a formatter callable:
tool_node_verbose = ToolNode(
    [fetch_record],
    handle_tool_errors=format_db_error,
)
```

### Example 6c — State injection via `InjectedState`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, InjectedState

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str

@tool
def get_user_profile(
    field: str,
    # InjectedState passes the full graph state into the tool invisibly
    state: Annotated[State, InjectedState],
) -> str:
    """Look up a field from the current user's profile."""
    return f"user {state['user_id']} field '{field}': example_value"

tool_node = ToolNode([get_user_profile])
```

### Example 6d — Using `wrap_tool_call` for cross-cutting concerns

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

@tool
def slow_api(query: str) -> str:
    """Call a slow external API."""
    import time; time.sleep(0.1)
    return f"result:{query}"

call_log: list[str] = []

def audit_wrapper(request: ToolCallRequest, execute):
    call_log.append(f"calling {request.tool_call['name']}")
    result = execute(request)
    call_log.append(f"done {request.tool_call['name']}")
    return result

tool_node = ToolNode([slow_api], wrap_tool_call=audit_wrapper)
```

### Example 6e — Custom `messages_key`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class MyState(TypedDict):
    chat_history: Annotated[list[AnyMessage], add_messages]
    metadata: dict

# Tell ToolNode to look for tool calls in "chat_history" instead of "messages"
tool_node = ToolNode([...], messages_key="chat_history")
```

---

## 7. `MessagesState` + `add_messages`

**Module:** `langgraph.graph.message`  
**Imports:**
```python
from langgraph.graph.message import MessagesState, add_messages
```
**Status:** `MessagesState` is stable. `MessageGraph` is deprecated since v1.0; use `StateGraph(MessagesState)` instead.

### What `MessagesState` is

```python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

`MessagesState` is the canonical starting-point state schema for chat and agent
workflows. Passing it to `StateGraph(MessagesState)` gives you a single `messages`
field backed by the `add_messages` reducer — the most important convenience in
LangGraph's message layer.

### What `add_messages` does

`add_messages(left, right)` merges two message lists by **message ID**:

- A message in `right` that does not exist in `left` is appended.
- A message in `right` whose `id` matches one already in `left` **replaces** it.
- A `RemoveMessage(id=…)` in `right` removes the matching message from `left`.
- A `RemoveMessage(id=REMOVE_ALL_MESSAGES)` drops all messages before it in `right`.

The `format="langchain-openai"` keyword converts the merged list to OpenAI-compatible
blocks (text + image_url, separate ToolMessage objects).

### Example 7a — Building a minimal chat graph with `MessagesState`

```python
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState

def chatbot(state: MessagesState) -> dict:
    # In production: pass state["messages"] to an LLM and return its response.
    last = state["messages"][-1]
    return {"messages": [AIMessage(content=f"Echo: {last.content}")]}

builder = StateGraph(MessagesState)
builder.add_node("bot", chatbot)
builder.add_edge(START, "bot")
builder.add_edge("bot", END)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage(content="Hello")]})
for m in result["messages"]:
    print(f"{m.type}: {m.content}")
# human: Hello
# ai: Echo: Hello
```

### Example 7b — In-place message editing (update by ID)

```python
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState

def correct_message(state: MessagesState) -> dict:
    # Replace the AI's first response by reusing its id
    original_ai = state["messages"][-1]
    corrected = AIMessage(
        id=original_ai.id,           # same ID → replaces in-place
        content="Corrected response",
    )
    return {"messages": [corrected]}

builder = StateGraph(MessagesState)
builder.add_node("correct", correct_message)
builder.add_edge(START, "correct")
builder.add_edge("correct", END)
graph = builder.compile()

ai_msg = AIMessage(id="msg-001", content="Original response")
result = graph.invoke({"messages": [ai_msg]})
print(result["messages"][-1].content)  # "Corrected response"
```

### Example 7c — Trimming with `RemoveMessage`

```python
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState

def keep_last_two(state: MessagesState) -> dict:
    msgs = state["messages"]
    if len(msgs) > 2:
        # Remove everything except the last 2 messages
        return {"messages": [RemoveMessage(id=m.id) for m in msgs[:-2]]}
    return {}

builder = StateGraph(MessagesState)
builder.add_node("trim", keep_last_two)
builder.add_edge(START, "trim")
builder.add_edge("trim", END)
graph = builder.compile()

msgs = [
    HumanMessage(id="1", content="Hi"),
    AIMessage(id="2", content="Hello"),
    HumanMessage(id="3", content="What's 2+2?"),
    AIMessage(id="4", content="4"),
]
result = graph.invoke({"messages": msgs})
print([m.content for m in result["messages"]])
# ["What's 2+2?", "4"]
```

### Example 7d — OpenAI format conversion

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class OpenAIState(TypedDict):
    # All messages are normalised to OpenAI format on every merge
    messages: Annotated[list[AnyMessage], add_messages(format="langchain-openai")]
```

---

## 8. `UIMessage` + `RemoveUIMessage` + `push_ui_message`

**Module:** `langgraph.graph.ui`  
**Imports:**
```python
from langgraph.graph.ui import UIMessage, RemoveUIMessage, push_ui_message, delete_ui_message, ui_message_reducer
```
**Status:** Stable — part of the LangGraph streaming UI protocol.

### What it does

`UIMessage` carries a **named UI component update** from a graph node to a frontend
renderer. The frontend maps the `name` field to a React (or other framework) component
and passes `props` to it. This enables streaming interactive UI elements — progress
bars, structured results, approval buttons — directly from within a graph node.

```python
class UIMessage(TypedDict):
    type: Literal["ui"]
    id:   str             # unique message ID (UUID if not set)
    name: str             # React component name / UI widget identifier
    props: dict[str, Any] # properties forwarded to the component
    metadata: dict[str, Any]  # run_id, tags, merge flag, etc.

class RemoveUIMessage(TypedDict):
    type: Literal["remove-ui"]
    id:   str             # ID of the UIMessage to remove
```

The `push_ui_message` helper writes a `UIMessage` to both the custom stream and the
graph state. `delete_ui_message` emits a `RemoveUIMessage` that splices it out.

The `ui_message_reducer` (used with `Annotated`) supports the same merge/remove logic
as `add_messages`: new messages are appended, existing IDs are replaced (or merged if
`metadata.merge=True`), and `RemoveUIMessage` removes by ID.

### Example 8a — Streaming a progress indicator

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.ui import UIMessage, push_ui_message, ui_message_reducer

class State(TypedDict):
    query: str
    ui: Annotated[list[UIMessage], ui_message_reducer]

def search_node(state: State) -> dict:
    # Push a "searching" spinner to the UI stream
    spinner = push_ui_message(
        name="ProgressSpinner",
        props={"label": f"Searching for: {state['query']}", "active": True},
    )
    # ... do actual search work ...
    result_card = push_ui_message(
        name="ResultCard",
        props={"title": "Search Results", "items": ["result_a", "result_b"]},
    )
    # Update the spinner to show completion
    push_ui_message(
        name="ProgressSpinner",
        id=spinner["id"],      # same ID → replaces the existing spinner
        props={"label": "Done", "active": False},
    )
    return {}

builder = StateGraph(State)
builder.add_node("search", search_node)
builder.add_edge(START, "search")
builder.add_edge("search", END)
graph = builder.compile()
```

### Example 8b — Merging props incrementally (`merge=True`)

```python
from langgraph.graph.ui import push_ui_message

def streaming_node(state: dict) -> dict:
    # Create a streaming text component
    msg = push_ui_message(
        name="StreamingText",
        props={"content": "Hello"},
    )
    # Append to props without replacing the whole message
    push_ui_message(
        name="StreamingText",
        id=msg["id"],
        props={"content": " world"},
        merge=True,  # props are merged: {"content": " world"} merged into existing
    )
    return {}
```

### Example 8c — Removing a UI component when it's no longer needed

```python
from langgraph.graph.ui import push_ui_message, delete_ui_message

def approval_node(state: dict) -> dict:
    # Show an approval button
    btn = push_ui_message(
        name="ApprovalButton",
        props={"label": "Approve action?"},
    )
    # ... wait for approval via interrupt ...
    # Once resolved, remove the button from the UI
    delete_ui_message(btn["id"])
    return {}
```

### Example 8d — Associating a UI message with an LLM message

```python
from langchain_core.messages import AIMessage
from langgraph.graph.ui import push_ui_message

def structured_output_node(state: dict) -> dict:
    ai_msg = AIMessage(id="ai-001", content="Here are your results:")
    chart = push_ui_message(
        name="BarChart",
        props={"data": [10, 20, 30], "labels": ["A", "B", "C"]},
        message=ai_msg,   # links the chart to the AI message in the stream
    )
    return {"messages": [ai_msg]}
```

---

## 9. `Runtime` + `ExecutionInfo` + `RunControl`

**Module:** `langgraph.runtime`  
**Imports:**
```python
from langgraph.runtime import Runtime, ExecutionInfo, RunControl, ServerInfo, get_runtime
```
**Status:** Stable (`Runtime` added in v0.6.0).

### What it does

`Runtime` is a **frozen dataclass** injected automatically into every node function that
declares a `runtime: Runtime[ContextT]` parameter. It bundles all run-scoped utilities
into a single object:

```python
@dataclass(frozen=True)
class Runtime(Generic[ContextT]):
    context:        ContextT            # typed run context (user_id, db_conn, etc.)
    store:          BaseStore | None    # persistent cross-thread KV/vector store
    stream_writer:  StreamWriter        # write to the custom stream
    heartbeat:      Callable[[], None]  # extend idle timeout
    previous:       Any                 # previous return value (functional API)
    execution_info: ExecutionInfo | None
    server_info:    ServerInfo | None   # populated by LangGraph Server
    control:        RunControl | None   # cooperative draining
```

### `ExecutionInfo` — per-node metadata

```python
@dataclass(frozen=True)
class ExecutionInfo:
    checkpoint_id:          str
    checkpoint_ns:          str
    task_id:                str
    thread_id:              str | None   # None without a checkpointer
    run_id:                 str | None
    node_attempt:           int          # retry attempt (1-indexed)
    node_first_attempt_time: float | None  # Unix timestamp of first attempt
```

### `RunControl` — cooperative draining

```python
class RunControl:
    def request_drain(self, reason: str = "shutdown") -> None: ...
    @property
    def drain_requested(self) -> bool: ...
    @property
    def drain_reason(self) -> str | None: ...
```

### Example 9a — Accessing typed context via `Runtime`

```python
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

@dataclass
class AppContext:
    user_id: str
    locale: str = "en"

class State(TypedDict):
    greeting: str

store = InMemoryStore()
store.put(("profiles",), "user_42", {"name": "Alice", "plan": "pro"})

def greet(state: State, runtime: Runtime[AppContext]) -> dict:
    profile = runtime.store.get(("profiles",), runtime.context.user_id)
    name = profile.value["name"] if profile else "guest"
    locale = runtime.context.locale
    return {"greeting": f"[{locale}] Hello, {name}!"}

graph = (
    StateGraph(state_schema=State, context_schema=AppContext)
    .add_node("greet", greet)
    .set_entry_point("greet")
    .set_finish_point("greet")
    .compile(store=store)
)

result = graph.invoke({}, context=AppContext(user_id="user_42", locale="fr"))
print(result["greeting"])  # "[fr] Hello, Alice!"
```

### Example 9b — Reading `ExecutionInfo` for retry-aware logic

```python
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
import time

@dataclass
class Ctx:
    pass

class State(TypedDict):
    result: str

def flaky_node(state: State, runtime: Runtime[Ctx]) -> dict:
    info = runtime.execution_info
    attempt = info.node_attempt if info else 1
    elapsed = (
        time.time() - info.node_first_attempt_time
        if info and info.node_first_attempt_time
        else 0
    )
    print(f"Attempt #{attempt}, {elapsed:.2f}s since first try")
    if attempt < 3:
        raise ConnectionError("Simulated transient failure")  # ValueError is excluded by default retry_on
    return {"result": "success after retries"}

graph = (
    StateGraph(state_schema=State, context_schema=Ctx)
    .add_node(
        "flaky",
        flaky_node,
        retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.05),
    )
    .set_entry_point("flaky")
    .set_finish_point("flaky")
    .compile()
)

result = graph.invoke({}, context=Ctx())
print(result["result"])  # "success after retries"
```

### Example 9c — Cooperative draining with `RunControl`

```python
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime

@dataclass
class Ctx:
    shutdown_requested: bool = False

class State(TypedDict):
    items_processed: int

def worker(state: State, runtime: Runtime[Ctx]) -> dict:
    # Check both the runtime control plane and application-level context
    if runtime.drain_requested:
        print(f"Draining: {runtime.drain_reason}")
        return {"items_processed": state["items_processed"]}

    if runtime.context.shutdown_requested:
        if runtime.control:
            runtime.control.request_drain("app_shutdown")
        return {"items_processed": state["items_processed"]}

    # Normal work
    return {"items_processed": state["items_processed"] + 1}
```

### Example 9d — Streaming custom events with `Runtime.stream_writer`

```python
from dataclasses import dataclass
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime

@dataclass
class Ctx:
    pass

class State(TypedDict):
    result: str

def streaming_node(state: State, runtime: Runtime[Ctx]) -> dict:
    for i in range(3):
        runtime.stream_writer({"progress": i, "total": 3})
    return {"result": "done"}

graph = (
    StateGraph(state_schema=State, context_schema=Ctx)
    .add_node("stream", streaming_node)
    .set_entry_point("stream")
    .set_finish_point("stream")
    .compile()
)

for chunk in graph.stream({}, context=Ctx(), stream_mode="custom"):
    print(chunk)
# {"progress": 0, "total": 3}
# {"progress": 1, "total": 3}
# {"progress": 2, "total": 3}
```

---

## 10. `HumanResponse`

**Module:** `langgraph.prebuilt.interrupt`  
**Import:** `from langgraph.prebuilt.interrupt import HumanResponse`  
**Status:** Stable.

> **Note on related deprecated types** — `HumanInterruptConfig`, `ActionRequest`, and
> `HumanInterrupt` have all been **moved to `langchain.agents.interrupt`** as of
> `langgraph==1.0.0`. Import them from there. `HumanResponse` remains in
> `langgraph.prebuilt.interrupt` and is not deprecated.

### What it is

`HumanResponse` is the structured reply a frontend sends back when the graph resumes
from a human-in-the-loop interrupt. The `type` field drives different code paths on
resume; `args` carries the actual payload.

```python
class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
```

| `type` | Meaning | `args` |
|---|---|---|
| `"accept"` | Human approves — proceed as-is | `None` |
| `"ignore"` | Human skips this step | `None` |
| `"response"` | Human provides text feedback | `str` |
| `"edit"` | Human modifies the content | `ActionRequest` with updated `args` |

### Example 10a — Four-way response handling in a review node

```python
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.interrupt import HumanResponse

class State(TypedDict):
    draft: str
    final: str
    skipped: bool

def review_node(state: State) -> Any:
    response: HumanResponse = interrupt({
        "type": "review_request",
        "content": state["draft"],
    })

    if response["type"] == "accept":
        return {"final": state["draft"]}
    elif response["type"] == "ignore":
        return {"skipped": True, "final": ""}
    elif response["type"] == "response":
        # Human provided textual feedback — store it in draft (wire a loop back to "review" to re-process)
        return {"draft": f"{state['draft']} [feedback: {response['args']}]", "final": ""}
    elif response["type"] == "edit":
        # Human directly edited the content
        return {"final": response["args"]["args"].get("content", state["draft"])}
    return {}

builder = StateGraph(State)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)
graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "review-demo"}}

# Invoke — pauses at interrupt
graph.invoke({"draft": "My first draft", "final": "", "skipped": False}, config)

# Resume with an "accept" response
result = graph.invoke(
    Command(resume=HumanResponse(type="accept", args=None)),
    config,
)
print(result["final"])  # "My first draft"
```

### Example 10b — "Edit" response updating structured args

```python
from langgraph.types import Command
from langgraph.prebuilt.interrupt import HumanResponse

# Simulate a human editing a command before it executes
edit_response = HumanResponse(
    type="edit",
    args={
        "action": "run_command",
        "args": {"command": "ls", "args": ["-la", "/tmp"]},  # edited args
    },
)

# Pass back via Command.resume when re-invoking:
# graph.invoke(Command(resume=edit_response), config)
```

### Example 10c — Batch review with mixed response types

```python
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanResponse

def batch_review_node(state: dict) -> dict:
    items = state["pending_items"]
    approved = []
    rejected = []

    for item in items:
        response: HumanResponse = interrupt({"item": item, "action": "approve_or_reject"})
        if response["type"] == "accept":
            approved.append(item)
        elif response["type"] in ("ignore", "response"):
            rejected.append(item)

    return {"approved": approved, "rejected": rejected}
```

---

## Version Reference

| Symbol | Module | Added/Changed in |
|---|---|---|
| `BinaryOperatorAggregate` | `langgraph.channels.binop` | ≤ 0.1 |
| `Topic` | `langgraph.channels.topic` | ≤ 0.1 |
| `EphemeralValue` | `langgraph.channels.ephemeral_value` | ≤ 0.1 |
| `LastValue` / `LastValueAfterFinish` | `langgraph.channels.last_value` | ≤ 0.1 |
| `GraphCallbackHandler` / events | `langgraph.callbacks` | 1.2.x |
| `ToolNode` | `langgraph.prebuilt.tool_node` | ≤ 0.2 |
| `MessagesState` / `add_messages` | `langgraph.graph.message` | 0.2 |
| `UIMessage` / `RemoveUIMessage` | `langgraph.graph.ui` | 1.1.x |
| `Runtime` / `ExecutionInfo` / `RunControl` | `langgraph.runtime` | 0.6.0 |
| `HumanResponse` | `langgraph.prebuilt.interrupt` | 0.2 |
| `HumanInterruptConfig` / `ActionRequest` / `HumanInterrupt` | `langchain.agents.interrupt` *(moved)* | deprecated in 1.0 |

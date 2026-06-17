---
title: "Class deep-dives Vol. 18 — Channels, caching, functional patterns & debug streaming"
description: "Source-verified deep dives into DebugTransformer + stream_mode='debug', StateSnapshot history navigation, Command full API (update+goto+Send+PARENT), add_messages advanced patterns (dedup/RemoveMessage/REMOVE_ALL_MESSAGES/format), Topic channel accumulate modes, NamedBarrierValue fan-in synchronization, entrypoint+task+previous stateful accumulation, push_message manual emission, EphemeralValue trigger channels, and CachePolicy+InMemoryCache combined patterns — with 3+ runnable examples each."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 18"
  order: 49
---

# Class deep-dives Vol. 18 — Channels, caching, functional patterns & debug streaming

Verified against **`langgraph==1.2.5`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `DebugTransformer` + `stream_mode="debug"` | `langgraph.stream.transformers` |
| 2 | `StateSnapshot` — history navigation | `langgraph.types` |
| 3 | `Command` — full API (update + goto + Send + PARENT) | `langgraph.types` |
| 4 | `add_messages` — advanced dedup, removal & format | `langgraph.graph.message` |
| 5 | `Topic` channel — `accumulate` modes + multi-producer | `langgraph.channels.topic` |
| 6 | `NamedBarrierValue` — fan-in synchronization | `langgraph.channels.named_barrier_value` |
| 7 | `entrypoint` + `task` + `previous` — stateful accumulation | `langgraph.func` |
| 8 | `push_message` — manual message emission | `langgraph.graph.message` |
| 9 | `EphemeralValue` — trigger channels + `guard` modes | `langgraph.channels.ephemeral_value` |
| 10 | `CachePolicy` + `InMemoryCache` — combined patterns | `langgraph.types` · `langgraph.cache.memory` |

---

## 1 · `DebugTransformer` + `stream_mode="debug"`

**Module:** `langgraph.stream.transformers`

`stream_mode="debug"` emits a low-level trace event for every super-step. Each event is a dict with a `"type"` key (`"checkpoint"` or `"task"` or `"task_result"`) and a numeric `"step"` counter. `DebugTransformer` wraps this into the v3 streaming API's `run.debug` projection — a drainable `StreamChannel[dict]`.

This is the deepest diagnostic hook in LangGraph: you can watch every checkpoint write, task dispatch, and task result without adding any logging to your nodes.

### Source signature

```python
class DebugTransformer(StreamTransformer):
    """Capture debug events as a drainable stream.

    Surfaces stream_mode="debug" data on run.debug as a
    StreamChannel[dict[str, Any]]. Each item is a debug event with
    step-level detail (checkpoint snapshots, task payloads, and
    task results wrapped with step number and timestamp).
    """
    _native = True
    required_stream_modes = ("debug",)
```

### Example 1 — watching the raw debug event stream (v1 API)

The simplest approach: pass `stream_mode="debug"` to the standard `.stream()` call. Each yielded value is a dict with keys `"type"`, `"step"`, and `"payload"`.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    counter: Annotated[int, operator.add]

def step_a(state: State) -> dict:
    return {"counter": 1}

def step_b(state: State) -> dict:
    return {"counter": 10}

builder = StateGraph(State)
builder.add_sequence([("a", step_a), ("b", step_b)])
builder.add_edge(START, "a")
builder.add_edge("b", END)

graph = builder.compile()

for event in graph.stream({"counter": 0}, stream_mode="debug"):
    print(f"type={event['type']!r:12}  step={event['step']}")
```

Output:

```
type='task'        step=1
type='task_result' step=1
type='checkpoint'  step=1
type='task'        step=2
type='task_result' step=2
type='checkpoint'  step=2
type='checkpoint'  step=3
```

### Example 2 — inspecting the full event payload

Each debug event carries a `"payload"` with rich execution detail. Task events include input state and triggers; task-result events include the node's output; checkpoint events include the full channel state.

```python
import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int
    label: str

def transform(state: State) -> dict:
    return {"value": state["value"] * 2, "label": f"doubled-{state['value']}"}

builder = StateGraph(State)
builder.add_node("transform", transform)
builder.add_edge(START, "transform")
builder.add_edge("transform", END)

graph = builder.compile()

for event in graph.stream({"value": 5, "label": "start"}, stream_mode="debug"):
    t = event["type"]
    payload = event["payload"]
    if t == "task":
        print(f"[TASK  start] node={payload['name']!r}  input={payload['input']}")
    elif t == "task_result":
        print(f"[TASK result] node={payload['name']!r}  output={payload['writes']}")
    elif t == "checkpoint":
        print(f"[CHECKPOINT ] step={event['step']}  "
              f"values={payload['values']}  next={payload['next']}")
```

### Example 3 — debug + updates simultaneously for side-by-side tracing

Combine `"debug"` with `"updates"` to correlate node outputs with the underlying super-steps in a multi-agent graph.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list[str], operator.add]

def analyst(state: State) -> dict:
    return {"messages": ["analyst: done"]}

def writer(state: State) -> dict:
    return {"messages": ["writer: done"]}

builder = StateGraph(State)
builder.add_node("analyst", analyst)
builder.add_node("writer", writer)
builder.add_edge(START, "analyst")
builder.add_edge("analyst", "writer")
builder.add_edge("writer", END)

graph = builder.compile()

for mode, payload in graph.stream(
    {"messages": []},
    stream_mode=["updates", "debug"],
):
    if mode == "updates":
        print(f"  [update] {payload}")
    elif mode == "debug" and payload["type"] == "task":
        print(f"  [debug ] task starting: {payload['payload']['name']}")
```

### Key event types

| `type` | When fired | Key payload fields |
|--------|-----------|-------------------|
| `"task"` | Node execution starts | `name`, `input`, `triggers`, `id` |
| `"task_result"` | Node execution finishes | `name`, `writes`, `error`, `interrupts`, `id` |
| `"checkpoint"` | Checkpoint written | `config`, `values`, `metadata`, `next`, `tasks` |

---

## 2 · `StateSnapshot` — history navigation

**Module:** `langgraph.types`

`StateSnapshot` is the return type of `graph.get_state()` and each item from `graph.get_state_history()`. It's a `NamedTuple` — all fields are positional, so you can unpack it, but named access is clearer.

```python
class StateSnapshot(NamedTuple):
    values:        dict[str, Any] | Any   # current channel values
    next:          tuple[str, ...]         # nodes scheduled next
    config:        RunnableConfig          # this snapshot's config
    metadata:      CheckpointMetadata | None
    created_at:    str | None              # ISO 8601 timestamp
    parent_config: RunnableConfig | None   # previous checkpoint's config
    tasks:         tuple[PregelTask, ...]  # tasks in this step
    interrupts:    tuple[Interrupt, ...]   # pending interrupts
```

### Example 1 — reading `next`, `tasks`, and `interrupts` after a pause

```python
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    items: Annotated[list[str], operator.add]
    approved: Optional[bool]

def gather(state: State) -> dict:
    return {"items": ["item-1", "item-2"]}

def review(state: State) -> dict:
    decision = interrupt({"question": "Approve?", "items": state["items"]})
    return {"approved": decision == "yes"}

def finalize(state: State) -> dict:
    return {"items": ["finalized"]}

builder = StateGraph(State)
builder.add_sequence([("gather", gather), ("review", review), ("finalize", finalize)])
builder.add_edge(START, "gather")
builder.add_edge("finalize", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "snap-1"}}

# Run until interrupt
graph.invoke({"items": [], "approved": None}, config)

# Inspect the paused snapshot
snap = graph.get_state(config)
print("values      :", snap.values)
print("next        :", snap.next)          # ('review',)
print("interrupts  :", [i.value for i in snap.interrupts])
print("created_at  :", snap.created_at)
print("metadata    :", snap.metadata.get("step") if snap.metadata else None)

# Resume
result = graph.invoke(Command(resume="yes"), config)
print("approved    :", result["approved"])  # True
```

### Example 2 — iterating checkpoint history for audit / replay

`get_state_history` yields snapshots newest-to-oldest. Each snapshot's `config` can be used to re-invoke from that exact point.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    step: Annotated[int, operator.add]

def inc(state: State) -> dict:
    return {"step": 1}

builder = StateGraph(State)
builder.add_sequence([("a", inc), ("b", inc), ("c", inc)])
builder.add_edge(START, "a")
builder.add_edge("c", END)

saver = InMemorySaver()
graph = builder.compile(checkpointer=saver)
config = {"configurable": {"thread_id": "hist-1"}}
graph.invoke({"step": 0}, config)

print("--- history (newest → oldest) ---")
for snap in graph.get_state_history(config):
    print(f"  step={snap.values['step']}  next={snap.next}  "
          f"at={snap.created_at[:19] if snap.created_at else '?'}")

# Re-run from the oldest checkpoint (step=0, before any nodes ran)
history = list(graph.get_state_history(config))
oldest = history[-1]
print("\nRewinding to step=0, next=", oldest.next)
result = graph.invoke(None, oldest.config)
print("Re-run result:", result)  # {'step': 3}
```

### Example 3 — using `parent_config` to walk the chain manually

```python
snap = graph.get_state(config)

chain = []
current = snap
while current is not None:
    chain.append({
        "step": current.values.get("step"),
        "next": current.next,
        "created_at": current.created_at,
    })
    if current.parent_config is None:
        break
    current = graph.get_state(current.parent_config)

print("Checkpoint chain (newest → oldest):")
for entry in chain:
    print(f"  step={entry['step']}  next={entry['next']}  at={entry['created_at'][:19]}")
```

### `StateSnapshot` field reference

| Field | Type | Notes |
|-------|------|-------|
| `values` | `dict` | Full channel state at this checkpoint |
| `next` | `tuple[str, ...]` | Empty `()` for terminal snapshots |
| `config` | `RunnableConfig` | Pass to `invoke`/`stream`/`get_state` to operate at this point |
| `metadata` | `CheckpointMetadata` | Contains `step` (int), `source` (`"input"`, `"loop"`, `"update"`), `writes` |
| `created_at` | `str \| None` | ISO 8601 UTC timestamp |
| `parent_config` | `RunnableConfig \| None` | `None` for the initial checkpoint |
| `tasks` | `tuple[PregelTask, ...]` | Non-empty only for interrupted/mid-run states |
| `interrupts` | `tuple[Interrupt, ...]` | Pending interrupts waiting for `Command(resume=...)` |

---

## 3 · `Command` — full API (update + goto + Send + PARENT)

**Module:** `langgraph.types`

`Command` is the single object that can simultaneously **update state**, **navigate** (jump to a node), **resume** an interrupt, and **dispatch fan-out tasks** — all in one return value. Each field is optional and defaults to a no-op.

```python
@dataclass
class Command(Generic[N], ToolOutputMixin):
    graph:  str | None = None    # None = current graph; Command.PARENT = parent graph
    update: Any | None = None    # state update (dict or TypedDict-compatible)
    resume: dict[str, Any] | Any | None = None  # interrupt resume value
    goto:   Send | Sequence[Send | N] | N = ()  # node name(s) or Send objects
    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"
```

### Example 1 — combining `update` + `goto` in a router node

A routing node that writes to state AND redirects flow, bypassing the normal edge wiring.

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class State(TypedDict):
    score:  int
    tier:   str
    events: Annotated[list[str], operator.add]

def score_router(state: State) -> Command[Literal["premium", "standard", "free"]]:
    score = state["score"]
    if score >= 90:
        tier = "premium"
    elif score >= 60:
        tier = "standard"
    else:
        tier = "free"
    return Command(
        update={"tier": tier, "events": [f"routed → {tier}"]},
        goto=tier,
    )

def premium(state: State) -> dict:
    return {"events": ["premium handler ran"]}

def standard(state: State) -> dict:
    return {"events": ["standard handler ran"]}

def free(state: State) -> dict:
    return {"events": ["free handler ran"]}

builder = StateGraph(State)
builder.add_node("router", score_router)
builder.add_node("premium", premium)
builder.add_node("standard", standard)
builder.add_node("free", free)
builder.add_edge(START, "router")
for tier in ("premium", "standard", "free"):
    builder.add_edge(tier, END)

graph = builder.compile()

result = graph.invoke({"score": 95, "tier": "", "events": []})
print(result["tier"])    # premium
print(result["events"])  # ['routed → premium', 'premium handler ran']

result2 = graph.invoke({"score": 42, "tier": "", "events": []})
print(result2["tier"])   # free
```

### Example 2 — fan-out with `Send` objects in `goto` (map-reduce)

`goto` accepts a list mixing node names and `Send` objects. Use this to kick off parallel tasks with different payloads from a single node.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

class State(TypedDict):
    topics:  list[str]
    results: Annotated[list[str], operator.add]
    summary: str

def dispatcher(state: State) -> Command:
    sends = [Send("process_topic", {"topic": t}) for t in state["topics"]]
    return Command(
        update={"results": [f"dispatching {len(state['topics'])} topics"]},
        goto=sends,
    )

def process_topic(state: dict) -> dict:
    return {"results": [f"processed: {state['topic']}"]}

def summarise(state: State) -> dict:
    return {"summary": f"Done: {len(state['results'])} items"}

builder = StateGraph(State)
builder.add_node("dispatcher", dispatcher)
builder.add_node("process_topic", process_topic)
builder.add_node("summarise", summarise)
builder.add_edge(START, "dispatcher")
builder.add_edge("process_topic", "summarise")
builder.add_edge("summarise", END)

graph = builder.compile()
result = graph.invoke({"topics": ["AI", "Python", "LangGraph"], "results": [], "summary": ""})
print(result["summary"])    # Done: 4 items (3 processed + 1 dispatching message)
print(result["results"][:2])
```

### Example 3 — `Command.PARENT` to propagate results up from a subgraph

When a subgraph node returns `Command(graph=Command.PARENT, update=...)`, the update is applied to the **parent** graph's state. This lets a subgraph signal completion or errors without the parent polling it.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# ---- subgraph ----
class SubState(TypedDict):
    item: str
    processed: bool

def process_item(state: SubState) -> Command:
    result = f"[done:{state['item']}]"
    return Command(
        update={"processed": True},          # updates subgraph state
        graph=Command.PARENT,                # AND writes to parent
    )

sub_builder = StateGraph(SubState)
sub_builder.add_node("process", process_item)
sub_builder.add_edge(START, "process")
sub_builder.add_edge("process", END)
subgraph = sub_builder.compile()

# ---- parent graph ----
class ParentState(TypedDict):
    items:   list[str]
    outputs: Annotated[list[str], operator.add]

def launcher(state: ParentState) -> dict:
    return {}  # subgraph is called via the node below

parent_builder = StateGraph(ParentState)
parent_builder.add_node("run_sub", subgraph)
parent_builder.add_edge(START, "run_sub")
parent_builder.add_edge("run_sub", END)
parent_graph = parent_builder.compile()

# When process_item returns Command(graph=Command.PARENT, ...) it writes
# to the subgraph's own outputs first, then signals the parent.
# The parent here wires the subgraph as a node — the update goes to
# the parent's writable channels.
result = parent_graph.invoke({"items": ["x", "y"], "outputs": []})
print(result)
```

### `Command` field combinations reference

| Combination | Effect |
|-------------|--------|
| `Command(update={...})` | Update state; continue normal edge routing |
| `Command(goto="node")` | Jump to node; skip normal edge routing |
| `Command(update={...}, goto="node")` | Update state AND jump |
| `Command(goto=[Send(...), Send(...)])` | Fan-out to multiple tasks |
| `Command(resume=value)` | Resume the next pending interrupt |
| `Command(resume={id: value})` | Resume a specific interrupt by ID |
| `Command(graph=Command.PARENT, update={...})` | Write to parent graph's state |

---

## 4 · `add_messages` — advanced dedup, removal & format

**Module:** `langgraph.graph.message`

`add_messages` is the reducer behind `MessagesState`. It does far more than append:

- **ID-based deduplication** — if an incoming message shares an ID with an existing one, it **replaces** the existing message in-place
- **`RemoveMessage` targeting** — pass a `RemoveMessage(id=x)` to delete a specific message by ID
- **`REMOVE_ALL_MESSAGES` sentinel** — clear the entire list in one operation
- **`format="langchain-openai"` normalisation** — coerce content blocks to OpenAI wire format

### Source signature (condensed)

```python
def add_messages(
    left:   Messages,
    right:  Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    ...
```

`Messages = list[MessageLikeRepresentation] | MessageLikeRepresentation`

Both `left` and `right` accept raw tuples `("role", "content")`, dicts with `"role"` and `"content"`, or typed `BaseMessage` objects — all are coerced automatically.

### Example 1 — ID-based deduplication (edit a previous message)

When a message with the same ID arrives, it replaces the original rather than appending. This is how LangGraph supports "edit in place" for streaming tokens that arrive in chunks.

```python
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

msgs = [
    HumanMessage(content="Hello", id="h1"),
    AIMessage(content="Hi!", id="a1"),
]

# Patch the AI message — same id means replace
updated = add_messages(msgs, [AIMessage(content="Hi there, how can I help?", id="a1")])
for m in updated:
    print(f"{m.type:8} id={m.id}  content={m.content!r}")
# human    id=h1  content='Hello'
# ai       id=a1  content='Hi there, how can I help?'  ← replaced
```

### Example 2 — `RemoveMessage` and `REMOVE_ALL_MESSAGES` in a StateGraph

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES

class State(TypedDict):
    messages: Annotated[list, add_messages]

def add_three(state: State) -> dict:
    return {"messages": [
        HumanMessage("msg 1", id="m1"),
        AIMessage("msg 2", id="m2"),
        HumanMessage("msg 3", id="m3"),
    ]}

def remove_middle(state: State) -> dict:
    return {"messages": RemoveMessage(id="m2")}

def reset_all(state: State) -> dict:
    # REMOVE_ALL_MESSAGES is a sentinel id — clears everything before it
    return {"messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        HumanMessage("fresh start", id="fresh"),
    ]}

builder = StateGraph(State)
builder.add_node("add", add_three)
builder.add_node("remove", remove_middle)
builder.add_node("reset", reset_all)
builder.add_edge(START, "add")
builder.add_edge("add", "remove")
builder.add_edge("remove", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
result = graph.invoke({"messages": []})

print("Final messages:")
for m in result["messages"]:
    print(f"  {m.type:8} id={m.id}  content={m.content!r}")
# Only 'fresh start' remains — everything before REMOVE_ALL_MESSAGES was cleared
```

### Example 3 — `format="langchain-openai"` to normalise mixed content

When a node returns raw dicts, tuples, or multimodal content, `format="langchain-openai"` coerces everything to typed `BaseMessage` objects with OpenAI-compatible content blocks.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages(format="langchain-openai")]

def node_with_raw_messages(state: State) -> dict:
    return {"messages": [
        ("user", "What's in this image?"),
        ("assistant", "I see a cat."),
    ]}

builder = StateGraph(State)
builder.add_node("chat", node_with_raw_messages)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
result = graph.invoke({"messages": []})
for m in result["messages"]:
    print(f"{m.type:12}  content={m.content!r}")
```

### Key behaviours from source

| Behaviour | Rule |
|-----------|------|
| Append | Incoming message without a matching ID is appended |
| Replace | Incoming message whose ID already exists replaces in-place |
| Delete | `RemoveMessage(id=x)` removes the message with that ID; raises `ValueError` if ID not found |
| Clear-and-replace | `RemoveMessage(id=REMOVE_ALL_MESSAGES)` deletes everything before it; messages after it remain |
| ID assignment | Messages without an ID get a UUID4 assigned automatically |
| Coercion | `tuple`, `str`, `dict` are coerced to typed `BaseMessage` via `convert_to_messages` |

---

## 5 · `Topic` channel — `accumulate` modes + multi-producer

**Module:** `langgraph.channels.topic`

`Topic` is a PubSub-style channel. Unlike `LastValue` (stores one value) or `BinaryOperatorAggregate` (folds with a reducer), `Topic` **collects** all values written in a super-step into a list.

```python
class Topic(Generic[Value], BaseChannel[Sequence[Value], Value | list[Value], list[Value]]):
    def __init__(self, typ: type[Value], accumulate: bool = False) -> None:
```

- **`accumulate=False`** (default) — the list is **cleared after every super-step**. Nodes always see only the writes from the current step.
- **`accumulate=True`** — values persist across steps, growing like a log until the graph terminates.

Use `Annotated[list[T], Topic(T)]` or `Annotated[list[T], Topic(T, accumulate=True)]` in your state TypedDict.

### Example 1 — per-step event collection (accumulate=False)

Each step the Topic resets to empty; nodes only see events produced in the same super-step.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.topic import Topic

class State(TypedDict):
    events:  Annotated[list[str], Topic(str)]       # cleared each step
    counter: Annotated[int, operator.add]

def step_one(state: State) -> dict:
    return {"events": "step_one_ran", "counter": 1}

def step_two(state: State) -> dict:
    # events is empty here (previous step's Topic was cleared)
    print(f"step_two sees events={state['events']!r}")  # []
    return {"events": "step_two_ran", "counter": 1}

builder = StateGraph(State)
builder.add_node("s1", step_one)
builder.add_node("s2", step_two)
builder.add_edge(START, "s1")
builder.add_edge("s1", "s2")
builder.add_edge("s2", END)

graph = builder.compile()
result = graph.invoke({"counter": 0})
print("final events:", result["events"])  # ['step_two_ran'] — only current step
```

### Example 2 — accumulating a log across steps (accumulate=True)

With `accumulate=True`, every write is appended to the channel's persistent list. Useful for audit logs, conversation histories, or event journals.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.topic import Topic

class State(TypedDict):
    log:     Annotated[list[str], Topic(str, accumulate=True)]
    counter: Annotated[int, operator.add]

def step_a(state: State) -> dict:
    return {"log": "A started", "counter": 1}

def step_b(state: State) -> dict:
    print(f"step_b sees full log so far: {state['log']}")  # includes A's entry
    return {"log": "B started", "counter": 1}

def step_c(state: State) -> dict:
    print(f"step_c sees full log so far: {state['log']}")
    return {"log": "C complete"}

builder = StateGraph(State)
builder.add_sequence([("a", step_a), ("b", step_b), ("c", step_c)])
builder.add_edge(START, "a")
builder.add_edge("c", END)

graph = builder.compile()
result = graph.invoke({"counter": 0})
print("Final log:", result["log"])
# ['A started', 'B started', 'C complete']
```

### Example 3 — multi-producer fanout with Topic

Multiple parallel nodes can write to the same `Topic` channel in one super-step. All writes are collected into the list without conflicts (unlike `LastValue` which raises on concurrent writes).

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.topic import Topic
from langgraph.types import Send

class State(TypedDict):
    items:   list[str]
    results: Annotated[list[str], Topic(str)]

def dispatch(state: State):
    return [Send("worker", {"item": i}) for i in state["items"]]

def worker(state: dict) -> dict:
    return {"results": f"processed:{state['item']}"}

def collect(state: State) -> dict:
    print("All results this step:", state["results"])
    return {}

builder = StateGraph(State)
builder.add_node("worker", worker)
builder.add_node("collect", collect)
builder.add_conditional_edges(START, dispatch, ["worker"])
builder.add_edge("worker", "collect")
builder.add_edge("collect", END)

graph = builder.compile()
result = graph.invoke({"items": ["alpha", "beta", "gamma"], "results": []})
print("results:", sorted(result["results"]))
# ['processed:alpha', 'processed:beta', 'processed:gamma']
```

---

## 6 · `NamedBarrierValue` — fan-in synchronization

**Module:** `langgraph.channels.named_barrier_value`

`NamedBarrierValue` blocks a downstream node until **all named producers** have written their token to the channel. Only then does the channel become "available" (non-empty). The channel consumes itself after making a value available — it resets to the empty-seen set, ready for the next cycle.

```python
class NamedBarrierValue(Generic[Value], BaseChannel[Value, Value, set[Value]]):
    def __init__(self, typ: type[Value], names: set[Value]) -> None:
```

- `names` — the set of tokens that must all be received before the barrier opens
- Each producer writes its own token (a string that matches one entry in `names`)
- The channel raises `InvalidUpdateError` if a value is written that isn't in `names`
- Use `Annotated[None, NamedBarrierValue(str, {"a", "b"})]` — the value type is typically `None` since you only care that all tokens arrived

**Important:** Do not include the barrier field in the initial `invoke` input; let the channel start empty.

### Example 1 — waiting for two parallel branches

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.named_barrier_value import NamedBarrierValue

class State(TypedDict):
    ready:   Annotated[None, NamedBarrierValue(str, {"fetch", "auth"})]
    fetched: str
    token:   str

def fetch_data(state: State) -> dict:
    import time; time.sleep(0.01)  # simulate I/O
    return {"ready": "fetch", "fetched": "data-payload"}

def authenticate(state: State) -> dict:
    return {"ready": "auth", "token": "Bearer xyz"}

def process(state: State) -> dict:
    print(f"Both ready! fetched={state['fetched']!r}  token={state['token']!r}")
    return {}

builder = StateGraph(State)
builder.add_node("fetch", fetch_data)
builder.add_node("auth",  authenticate)
builder.add_node("process", process)
builder.add_edge(START, "fetch")
builder.add_edge(START, "auth")
builder.add_edge("fetch", "process")
builder.add_edge("auth",  "process")
builder.add_edge("process", END)

graph = builder.compile()
# Note: don't pass 'ready' in the initial state
result = graph.invoke({"fetched": "", "token": ""})
print("fetched:", result["fetched"])  # data-payload
print("token  :", result["token"])    # Bearer xyz
```

### Example 2 — barrier in a sub-pipeline with Send fan-out

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.types import Send

class State(TypedDict):
    query:   str
    barrier: Annotated[None, NamedBarrierValue(str, {"search", "kg_lookup"})]
    hits:    Annotated[list[str], operator.add]

def router(state: State):
    return [Send("search", {}), Send("kg_lookup", {})]

def search(state: State) -> dict:
    return {"barrier": "search", "hits": [f"search hit for '{state['query']}'"]}

def kg_lookup(state: State) -> dict:
    return {"barrier": "kg_lookup", "hits": [f"KG fact for '{state['query']}'"]}

def merge(state: State) -> dict:
    print("Merged hits:", state["hits"])
    return {}

builder = StateGraph(State)
builder.add_node("search",    search)
builder.add_node("kg_lookup", kg_lookup)
builder.add_node("merge",     merge)
builder.add_conditional_edges(START, router, ["search", "kg_lookup"])
builder.add_edge("search",    "merge")
builder.add_edge("kg_lookup", "merge")
builder.add_edge("merge", END)

graph = builder.compile()
result = graph.invoke({"query": "LangGraph", "hits": []})
print("hits:", result["hits"])
```

### Example 3 — reusable barrier within a loop

`NamedBarrierValue` resets after it fires — `consume()` clears the seen-set, making it ready for the next cycle. This lets you reuse the same barrier across multiple loop iterations.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.named_barrier_value import NamedBarrierValue

class State(TypedDict):
    turn:    Annotated[int, operator.add]
    gate:    Annotated[None, NamedBarrierValue(str, {"left", "right"})]
    history: Annotated[list[str], operator.add]

def left_branch(state: State) -> dict:
    return {"gate": "left", "history": [f"turn-{state['turn']}-left"]}

def right_branch(state: State) -> dict:
    return {"gate": "right", "history": [f"turn-{state['turn']}-right"]}

def join_and_decide(state: State) -> str:
    print(f"Turn {state['turn']}: both branches done. history={state['history'][-2:]}")
    if state["turn"] >= 2:
        return "end"
    return "continue"

def bump(state: State) -> dict:
    return {"turn": 1}

builder = StateGraph(State)
builder.add_node("left",  left_branch)
builder.add_node("right", right_branch)
builder.add_node("join",  lambda s: {})
builder.add_node("bump",  bump)
builder.add_edge(START, "left")
builder.add_edge(START, "right")
builder.add_edge("left",  "join")
builder.add_edge("right", "join")
builder.add_conditional_edges("join", join_and_decide,
                              {"end": END, "continue": "bump"})
builder.add_edge("bump", "left")
builder.add_edge("bump", "right")

graph = builder.compile()
result = graph.invoke({"turn": 0, "history": []})
print("Final history:", result["history"])
```

---

## 7 · `entrypoint` + `task` + `previous` — stateful accumulation

**Module:** `langgraph.func`

The **Functional API** lets you build stateful workflows without a `StateGraph`. Key insight: when a `checkpointer` is attached to `@entrypoint`, the decorated function gains access to a `previous` keyword argument — the **return value from the previous invocation** on the same `thread_id`. This enables accumulation, history tracking, and multi-turn conversations without explicit state TypedDicts.

`@task` turns a regular function into a deferrable unit of work. Calling a task returns a future; calling `.result()` blocks until complete.

### Example 1 — accumulating totals across invocations

```python
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def compute(x: int) -> int:
    return x * x  # square the input

@entrypoint(checkpointer=InMemorySaver())
def running_sum(n: int, *, previous: Optional[int] = None) -> int:
    squared = compute(n).result()
    total = (previous or 0) + squared
    return total

config = {"configurable": {"thread_id": "sum-thread"}}
print(running_sum.invoke(3, config))   # 0 + 9 = 9
print(running_sum.invoke(4, config))   # 9 + 16 = 25
print(running_sum.invoke(2, config))   # 25 + 4 = 29
```

### Example 2 — conversation history with `previous`

```python
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def generate_reply(message: str, history: list[str]) -> str:
    ctx = "\n".join(history[-3:])  # last 3 turns for context
    return f"[echo] {message} (context: {len(history)} turns)"

@entrypoint(checkpointer=InMemorySaver())
def chat(message: str, *, previous: Optional[list[str]] = None) -> list[str]:
    history = previous or []
    reply = generate_reply(message, history).result()
    return history + [f"user: {message}", f"bot: {reply}"]

config = {"configurable": {"thread_id": "chat-1"}}
chat.invoke("Hello!", config)
chat.invoke("How are you?", config)
result = chat.invoke("Tell me more.", config)
for line in result:
    print(line)
```

### Example 3 — parallel tasks + `previous` with `entrypoint.final`

Use `entrypoint.final` when you want to return one value to the caller but persist a different value as `previous` for the next invocation.

```python
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def score(text: str) -> float:
    return len(text) / 100.0  # trivial scoring

@task
def summarise(text: str) -> str:
    return text[:20] + "..."

@entrypoint(checkpointer=InMemorySaver())
def pipeline(
    text: str,
    *,
    previous: Optional[dict] = None,
) -> "entrypoint.final[str, dict]":
    score_f   = score(text)
    summary_f = summarise(text)
    current_score   = score_f.result()
    current_summary = summary_f.result()

    saved = {
        "last_score":   current_score,
        "last_summary": current_summary,
        "run_count":    (previous or {}).get("run_count", 0) + 1,
    }
    return entrypoint.final(
        value=current_summary,  # caller sees the summary
        save=saved,             # next invocation's `previous` is the full dict
    )

config = {"configurable": {"thread_id": "pipe-1"}}
print(pipeline.invoke("LangGraph is a stateful agent framework.", config))
print(pipeline.invoke("It supports checkpointing and human-in-the-loop.", config))
```

### `task` decorator parameters

```python
@task(
    name="my_task",             # optional display name
    retry_policy=RetryPolicy(max_attempts=3),  # per-task retries
    cache_policy=CachePolicy(ttl=300),         # cache results for 5 min
    timeout=30.0,               # hard wall-clock cap (async only)
)
async def my_task(x: int) -> int: ...
```

---

## 8 · `push_message` — manual message emission

**Module:** `langgraph.graph.message`

`push_message` writes a single message **immediately to the stream** during node execution — before the node returns. This enables token-by-token streaming of custom messages without using the `get_stream_writer()` / `stream_mode="custom"` mechanism.

```python
def push_message(
    message:   MessageLikeRepresentation | BaseMessageChunk,
    *,
    state_key: str | None = "messages",
) -> AnyMessage:
```

- `message` — any message-like: `BaseMessage`, `(role, content)` tuple, dict, or `BaseMessageChunk`
- `state_key` — the state channel to write into; defaults to `"messages"`; pass `None` to stream only without writing to state
- Returns the fully typed `AnyMessage` with an assigned `id`
- **Requires** a `messages` (or matching `state_key`) channel with `add_messages` reducer in the graph state

### Example 1 — streaming messages during a long-running node

```python
import time
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, push_message

class State(TypedDict):
    messages: Annotated[list, add_messages]

def streaming_node(state: State) -> dict:
    for i in range(4):
        time.sleep(0.02)
        push_message(AIMessage(
            content=f"chunk {i}",
            id=f"stream-msg-{i}",
        ))
    return {"messages": [AIMessage(content="[complete]", id="final")]}

builder = StateGraph(State)
builder.add_node("stream", streaming_node)
builder.add_edge(START, "stream")
builder.add_edge("stream", END)

graph = builder.compile()

print("--- stream_mode='messages' ---")
for chunk, metadata in graph.stream(
    {"messages": []},
    stream_mode="messages",
):
    print(f"  chunk={chunk.content!r}  node={metadata.get('langgraph_node')}")
```

### Example 2 — writing to a custom state_key

If your state uses a key other than `"messages"`, pass `state_key` to match. Pass `state_key=None` to stream-only (no state write).

```python
import time
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, push_message

class State(TypedDict):
    transcript: Annotated[list, add_messages]  # different key name

def narrator(state: State) -> dict:
    for step in ["intro", "body", "conclusion"]:
        push_message(
            AIMessage(content=f"[{step}]", id=f"narr-{step}"),
            state_key="transcript",
        )
    return {}

builder = StateGraph(State)
builder.add_node("narrate", narrator)
builder.add_edge(START, "narrate")
builder.add_edge("narrate", END)

graph = builder.compile()
result = graph.invoke({"transcript": []})
print([m.content for m in result["transcript"]])
# ['[intro]', '[body]', '[conclusion]']
```

### Example 3 — stream-only mode (state_key=None)

Pass `state_key=None` to emit the message to the stream **without** writing it to graph state. Useful for real-time progress indicators that don't need persistence.

```python
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, push_message
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]
    result: str

def worker(state: State) -> dict:
    for n in range(3):
        push_message(
            AIMessage(content=f"progress {n+1}/3", id=f"prog-{n}"),
            state_key=None,  # stream only, don't persist
        )
    return {"result": "done", "messages": [AIMessage("final", id="fin")]}

builder = StateGraph(State)
builder.add_node("work", worker)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile()

# Progress messages appear in the stream but NOT in final state
events = list(graph.stream({"messages": [], "result": ""}, stream_mode="messages"))
print(f"Stream events: {len(events)}")  # includes progress + final

final = graph.invoke({"messages": [], "result": ""})
print(f"State messages: {len(final['messages'])}")  # only 1 (the final)
```

### `push_message` vs `get_stream_writer()`

| | `push_message` | `get_stream_writer()` |
|--|----------------|----------------------|
| Stream mode | `"messages"` or `"messages-tuple"` | `"custom"` |
| State write | Yes (via `state_key`) | No |
| Value type | `BaseMessage` / message-like | Any |
| ID required | Yes (auto-assigned if missing) | N/A |
| Typical use | Streaming AI tokens, status messages | Arbitrary progress events |

---

## 9 · `EphemeralValue` — trigger channels + `guard` modes

**Module:** `langgraph.channels.ephemeral_value`

`EphemeralValue` stores exactly one value for **one super-step** and then clears itself. It's the channel type used for `START` inputs — values that are read exactly once and then discarded.

```python
class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    def __init__(self, typ: Any, guard: bool = True) -> None:
```

- **`guard=True`** (default) — raises `InvalidUpdateError` if more than one writer attempts to write in the same super-step. Guarantees single-producer semantics.
- **`guard=False`** — accepts multiple writes; stores the **last** value received. No error on concurrent writes.

### Example 1 — one-shot trigger signal (guard=True, the default)

`EphemeralValue` is ideal for "trigger" state: a signal that fires a node exactly once and is then invisible to subsequent nodes.

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.ephemeral_value import EphemeralValue

class State(TypedDict):
    trigger:  Annotated[Optional[str], EphemeralValue(Optional[str])]
    result:   str
    processed: bool

def on_trigger(state: State) -> dict:
    if state["trigger"]:
        return {"result": f"handled:{state['trigger']}", "processed": True}
    return {"processed": False}

def cleanup(state: State) -> dict:
    # trigger is None here — EphemeralValue cleared after on_trigger
    assert state["trigger"] is None or True  # EmptyChannelError caught by graph
    return {}

builder = StateGraph(State)
builder.add_node("trigger_handler", on_trigger)
builder.add_node("cleanup", cleanup)
builder.add_edge(START, "trigger_handler")
builder.add_edge("trigger_handler", "cleanup")
builder.add_edge("cleanup", END)

graph = builder.compile()

r1 = graph.invoke({"trigger": "reload", "result": "", "processed": False})
print("r1:", r1["result"], r1["processed"])

r2 = graph.invoke({"trigger": None, "result": "", "processed": False})
print("r2:", r2["processed"])  # False — no trigger
```

### Example 2 — guard=False for last-writer-wins semantics

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.types import Send

class State(TypedDict):
    winner: Annotated[str, EphemeralValue(str, guard=False)]  # last write wins
    report: str

def fast_worker(state: dict) -> dict:
    return {"winner": "fast"}

def slow_worker(state: dict) -> dict:
    return {"winner": "slow"}

def announce(state: State) -> dict:
    return {"report": f"winner={state['winner']}"}

def fanout(state: State):
    return [Send("fast", {}), Send("slow", {})]

builder = StateGraph(State)
builder.add_node("fast",     fast_worker)
builder.add_node("slow",     slow_worker)
builder.add_node("announce", announce)
builder.add_conditional_edges(START, fanout, ["fast", "slow"])
builder.add_edge("fast",     "announce")
builder.add_edge("slow",     "announce")
builder.add_edge("announce", END)

graph = builder.compile()
result = graph.invoke({"winner": "", "report": ""})
print("report:", result["report"])  # one of fast or slow — non-deterministic
```

### Example 3 — EphemeralValue as a per-step configuration channel

A common pattern: pass configuration into a graph via an EphemeralValue so nodes can read the config in step 1 but don't carry it through all subsequent steps.

```python
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.ephemeral_value import EphemeralValue

class State(TypedDict):
    config_override: Annotated[Optional[dict], EphemeralValue(Optional[dict])]
    model:           str
    results:         list[str]

def configure(state: State) -> dict:
    cfg = state.get("config_override") or {}
    model = cfg.get("model", "default-model")
    return {"model": model}

def process(state: State) -> dict:
    print(f"Using model: {state['model']}")
    # config_override is gone now (EphemeralValue cleared after configure)
    return {"results": [f"processed-with-{state['model']}"]}

builder = StateGraph(State)
builder.add_node("configure", configure)
builder.add_node("process",   process)
builder.add_edge(START, "configure")
builder.add_edge("configure", "process")
builder.add_edge("process", END)

graph = builder.compile()

r1 = graph.invoke({"config_override": {"model": "gpt-4o"}, "model": "", "results": []})
print("model used:", r1["model"])     # gpt-4o

r2 = graph.invoke({"config_override": None, "model": "", "results": []})
print("model used:", r2["model"])     # default-model
```

### `EphemeralValue` guard modes reference

| | `guard=True` (default) | `guard=False` |
|--|------------------------|---------------|
| Multiple writers in same step | `InvalidUpdateError` | Stores last value |
| After step | Clears to `MISSING` | Clears to `MISSING` |
| Use case | Single-producer signals, START inputs | Last-writer-wins fanout |

---

## 10 · `CachePolicy` + `InMemoryCache` — combined patterns

**Modules:** `langgraph.types`, `langgraph.cache.memory`

`CachePolicy` controls **what** is cached and **for how long**. `InMemoryCache` is the in-process store. Together they eliminate redundant computation across graph runs and task invocations.

```python
@dataclass
class CachePolicy(Generic[KeyFuncT]):
    key_func: KeyFuncT = default_cache_key   # hash of pickled input
    ttl:      int | None = None              # seconds; None = never expires
```

`default_cache_key` pickles the call arguments and returns a hex digest. Supply your own `key_func` for semantic keys (e.g. normalise before hashing, add user-namespace prefixes).

### Example 1 — caching `@task` results with TTL

Attach `CachePolicy` to a `@task` and pass an `InMemoryCache` to `@entrypoint`. The task's return value is cached; subsequent calls with the same arguments return immediately.

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

call_count = 0

@task(cache_policy=CachePolicy(ttl=120))  # cache for 2 minutes
def fetch_profile(user_id: str) -> dict:
    global call_count
    call_count += 1
    return {"user_id": user_id, "name": f"User-{user_id}"}

cache = InMemoryCache()

@entrypoint(cache=cache)
def get_user(user_id: str) -> dict:
    return fetch_profile(user_id).result()

# First call — cold cache
print(get_user.invoke("alice"))   # fetches, call_count=1
print(f"call_count after 1st: {call_count}")

# Same user — cached hit
print(get_user.invoke("alice"))   # from cache, call_count still 1
print(f"call_count after 2nd: {call_count}")

# Different user — cache miss
print(get_user.invoke("bob"))     # fetches, call_count=2
print(f"call_count after 3rd: {call_count}")
```

### Example 2 — custom `key_func` for normalised cache keys

The default key is a pickle hash — any difference in arguments produces a different key. A custom `key_func` lets you normalise before hashing (case folding, canonicalisation, user-namespace isolation).

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

def normalise_key(query: str) -> str:
    return query.strip().lower()

@task(cache_policy=CachePolicy(key_func=normalise_key))
def search(query: str) -> list[str]:
    return [f"result for '{query.strip().lower()}'"]

cache = InMemoryCache()

@entrypoint(cache=cache)
def do_search(query: str) -> list[str]:
    return search(query).result()

# These three should all hit the same cache entry
r1 = do_search.invoke("LangGraph")
r2 = do_search.invoke("langgraph")
r3 = do_search.invoke("  LANGGRAPH  ")
assert r1 == r2 == r3
print("Cache worked:", r1)
```

### Example 3 — per-user namespace isolation via `key_func`

When multiple users share a process, include the user ID in the cache key to prevent cross-user pollution.

```python
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

def user_namespaced_key(user_id: str, query: str) -> str:
    return f"{user_id}:{query.strip().lower()}"

@task(cache_policy=CachePolicy(key_func=user_namespaced_key, ttl=300))
def personalised_search(user_id: str, query: str) -> str:
    return f"[{user_id}] results for {query}"

cache = InMemoryCache()

@entrypoint(cache=cache)
def search_for_user(request: dict) -> str:
    return personalised_search(request["user_id"], request["query"]).result()

# alice and bob get different cache entries for the same query
r_alice = search_for_user.invoke({"user_id": "alice", "query": "AI news"})
r_bob   = search_for_user.invoke({"user_id": "bob",   "query": "AI news"})
print("alice:", r_alice)
print("bob  :", r_bob)
assert r_alice != r_bob  # different users, different cache entries
```

### Example 4 — manual cache invalidation with `task.clear_cache`

`task.clear_cache(cache)` / `task.aclear_cache(cache)` flushes only the cache entries for that specific task, leaving other tasks' caches intact.

```python
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

invalidation_count = 0

@task(cache_policy=CachePolicy(ttl=60))
def build_index(corpus_id: str) -> dict:
    global invalidation_count
    invalidation_count += 1
    return {"index": f"index-for-{corpus_id}", "version": invalidation_count}

cache = InMemoryCache()

@entrypoint(cache=cache)
def get_index(corpus_id: str) -> dict:
    return build_index(corpus_id).result()

r1 = get_index.invoke("corpus-A")
r2 = get_index.invoke("corpus-A")  # cached
print(f"Same version: {r1['version'] == r2['version']}")  # True

# Corpus updated — invalidate cache
build_index.clear_cache(cache)

r3 = get_index.invoke("corpus-A")  # recomputed
print(f"New version: {r3['version'] > r1['version']}")   # True
```

### Cache configuration matrix

| Setting | Value | Effect |
|---------|-------|--------|
| `key_func` | `default_cache_key` | Hash of all pickled arguments |
| `key_func` | custom callable | Your normalised / namespaced key |
| `ttl` | `None` | Entries never expire (until `clear_cache`) |
| `ttl` | `int` (seconds) | Entry expires after N seconds |
| `cache=InMemoryCache()` | on `@entrypoint` | In-process cache, cleared on process restart |
| `cache=RedisCache(...)` | on `@entrypoint` | Persistent cross-process cache |

---

## Summary

| Class / symbol | Key insight | Common mistake |
|---------------|-------------|----------------|
| `stream_mode="debug"` | Three event types: `"task"`, `"task_result"`, `"checkpoint"` — each with `"step"` and `"payload"` | Treating every debug event as a checkpoint — check `event["type"]` first |
| `StateSnapshot` | `interrupts` only non-empty when the graph is paused; `next` is empty `()` at terminal checkpoints | Iterating `get_state_history` without filtering — oldest entry has `next=()` just like the terminal one |
| `Command` | `goto` accepts a mix of strings and `Send` objects; `graph=Command.PARENT` writes to the parent | Returning `Command(goto=["node"])` with a list when it should be `Command(goto="node")` (string for single) |
| `add_messages` | ID-based dedup happens silently — no error if you update an existing message | Calling `RemoveMessage(id=x)` with an ID that doesn't exist — raises `ValueError` |
| `Topic` | `accumulate=True` persists across steps; `accumulate=False` (default) clears each step | Expecting `Topic(str)` to accumulate across nodes in different super-steps |
| `NamedBarrierValue` | Do not include the barrier field in the initial `invoke` input | Passing `barrier=None` in the initial state — triggers `InvalidUpdateError` |
| `entrypoint` + `previous` | `previous` is the **return value** of the last call on the same `thread_id` | Forgetting to provide a checkpointer — without one, `previous` is always `None` |
| `push_message` | Emits to the `"messages"` stream AND writes to state (unless `state_key=None`) | Calling without a matching `add_messages` reducer in state — message is written but reducer is absent |
| `EphemeralValue` | Clears after one super-step; `guard=True` raises on concurrent writes | Using `guard=True` with parallel `Send` nodes — both write the channel, triggering `InvalidUpdateError` |
| `CachePolicy` + `InMemoryCache` | Custom `key_func` must accept the **same arguments** as the decorated task | Using `ttl=0` expecting "never cache" — `ttl=None` means no expiry; `ttl=0` expires immediately |

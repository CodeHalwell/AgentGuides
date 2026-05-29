---
title: "Channels — API reference"
description: "The six channel types underlying LangGraph state: LastValue, BinaryOperatorAggregate, Topic, EphemeralValue, NamedBarrierValue, and AnyValue — how each is declared with Annotated[], what happens on concurrent writes, and when to pick each."
framework: langgraph
language: python
sidebar:
  label: "Ref · Channels"
  order: 37
---

# Channels — API reference

Verified against **`langgraph==1.2.2`** (module: `langgraph.channels`).

Every key in a `StateGraph` state schema is backed by a **channel**. Channels define how values are stored and how concurrent writes within the same super-step are resolved. Most users interact with channels only through `Annotated[type, reducer]` syntax; this page documents what those annotations actually create, their semantics under parallel execution, and when to choose each one.

## Minimal runnable example

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels import Topic


class State(TypedDict):
    # BinaryOperatorAggregate: accumulate a running sum across parallel writers
    total: Annotated[int, operator.add]
    # Topic: collect all writes in a step into a list
    events: Annotated[list[str], Topic(str)]


def node_a(state: State) -> dict:
    return {"total": 10, "events": "node_a ran"}


def node_b(state: State) -> dict:
    return {"total": 5, "events": "node_b ran"}


# Run a and b in parallel
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge(["a", "b"], END)

graph = builder.compile()
result = graph.invoke({"total": 0, "events": []})
print(result["total"])   # 15  (10 + 5, applied in order)
print(result["events"])  # ['node_a ran', 'node_b ran']
```

> `LastValue` (the default channel for plain, unannotated keys) rejects concurrent writes with `InvalidUpdateError`. If both `node_a` and `node_b` wrote to an unannotated `status: str` key, the graph would crash. Use `BinaryOperatorAggregate` or `Topic` for channels that parallel nodes all write to.

## Imports at a glance

| Channel | Import path | `Annotated` shorthand |
|---|---|---|
| `LastValue` | `langgraph.channels.last_value` | `T` (no annotation) |
| `BinaryOperatorAggregate` | `langgraph.channels.binop` | `Annotated[T, operator_fn]` |
| `Topic` | `langgraph.channels.topic` | `Annotated[list[T], Topic(T)]` |
| `EphemeralValue` | `langgraph.channels.ephemeral_value` | `Annotated[T, EphemeralValue(T)]` |
| `NamedBarrierValue` | `langgraph.channels.named_barrier_value` | `Annotated[None, NamedBarrierValue(str, names={...})]` |
| `AnyValue` | `langgraph.channels.any_value` | `Annotated[T, AnyValue(T)]` |
| `UntrackedValue` | `langgraph.channels.untracked_value` | `Annotated[T, UntrackedValue(T)]` |

All seven are also accessible via `langgraph.channels` (top-level re-export).

## Channel comparison

| Channel | Concurrent writes | Cleared after step | Checkpointed | Use case |
|---|---|---|---|---|
| `LastValue` | Error — at most one write per super-step | No | ✅ Yes | Normal scalar/message state |
| `BinaryOperatorAggregate` | Allowed — applied in arrival order | No | ✅ Yes | Running counters, message lists |
| `Topic` | Allowed — all collected into a list | Yes (`accumulate=False`) | ✅ Yes | Fan-in event buffers |
| `EphemeralValue` | Error (`guard=True`) or last wins (`guard=False`) | Yes — cleared if not written to | ✅ Yes | One-step trigger signals |
| `NamedBarrierValue` | Required — must see every named write | After consumed | ✅ Yes | N-source fan-in barriers |
| `AnyValue` | Allowed — takes the last value | No | ✅ Yes | Parallel-safe shared flags |
| `UntrackedValue` | Error (`guard=True`) or last wins (`guard=False`) | No | ❌ No | Computed values, secrets, large blobs |

---

## `LastValue`

```python
from langgraph.channels.last_value import LastValue
```

The default channel for every state key that has no `Annotated` wrapper. Stores exactly one value and raises `InvalidUpdateError` if two nodes write to the same key in the same super-step.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    value: str   # LastValue — one write per step only


def step(state: S) -> dict:
    return {"value": "hello"}


graph = StateGraph(S).add_node("step", step).add_edge(START, "step").add_edge("step", END).compile()
print(graph.invoke({"value": ""}))  # {'value': 'hello'}
```

**When two parallel nodes both write to a `LastValue` channel in one super-step, the graph raises immediately.** This is intentional — it forces the author to pick an explicit merge strategy (a reducer or a barrier) rather than silently losing writes.

```python
# This raises InvalidUpdateError at runtime:
class Bad(TypedDict):
    x: int  # LastValue, no reducer

def node_a(state): return {"x": 1}
def node_b(state): return {"x": 2}

builder = StateGraph(Bad)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge(["a", "b"], END)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
# graph.invoke(...) → InvalidUpdateError: two concurrent writes to "x"
```

---

## `BinaryOperatorAggregate`

```python
from langgraph.channels.binop import BinaryOperatorAggregate
```

Created whenever you write `Annotated[T, fn]` where `fn` is any callable `(current: T, update: T) -> T`. The standard library `operator` module provides the common cases.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Integer counter — each write adds to the running total
    hits: Annotated[int, operator.add]

    # String log — each write concatenates
    log: Annotated[str, lambda a, b: a + "\n" + b if a else b]

    # Chat messages — merge by id (add_messages is a BinaryOperatorAggregate internally)
    messages: Annotated[list[AnyMessage], add_messages]
```

Multiple concurrent writes in the same super-step all apply in order:

```python
def worker_a(state): return {"hits": 3}
def worker_b(state): return {"hits": 5}
# After both run in parallel: hits = 0 + 3 + 5 = 8
```

The initial value for `BinaryOperatorAggregate` is the zero value of the declared type (`0` for `int`, `""` for `str`, `[]` for `list`, etc.). For types whose zero value is not constructable, the channel starts as `MISSING` and the first write sets it directly.

### `add_messages` reducer

`add_messages` from `langgraph.graph.message` is the canonical message-list reducer. It merges by message `id`: messages with the same `id` overwrite the older version; new-id messages append.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langchain_core.messages import RemoveMessage


class Chat(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Remove a specific message by id
def prune(state: Chat) -> dict:
    return {"messages": [RemoveMessage(id=state["messages"][0].id)]}

# Wipe the entire history at once
def reset(state: Chat) -> dict:
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

---

## `Topic`

```python
from langgraph.channels.topic import Topic
```

A fan-in channel that collects **all** values written to it in one super-step into a list, rather than erroring on concurrent writes. After the step completes, the list is cleared (unless `accumulate=True`).

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import Topic


class State(TypedDict):
    # Collect every event written by any node this step
    events: Annotated[list[str], Topic(str)]
    # Accumulate events across ALL steps (unbounded growth — use carefully)
    all_events: Annotated[list[str], Topic(str, accumulate=True)]
```

Each node can write a single value or a list of values to a `Topic` channel:

```python
def node_a(state): return {"events": "a_finished"}           # single value
def node_b(state): return {"events": ["b_result", "b_warn"]} # list of values
# After both run: state["events"] == ["a_finished", "b_result", "b_warn"]
```

The `accumulate` parameter:

- `accumulate=False` (default) — the list is reset to `[]` at the start of each super-step before new writes are applied. Use this for per-step event buffers.
- `accumulate=True` — values are appended across all steps. The list grows indefinitely unless you explicitly reset it with `Overwrite([])`.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import Topic
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class Pipeline(TypedDict):
    items: list[str]
    results: Annotated[list[str], Topic(str)]   # cleared each step; collects all writes


def process(state: dict) -> dict:
    # Each parallel worker writes one result; Topic collects them all
    return {"results": f"processed:{state['item']}"}


builder = StateGraph(Pipeline)
builder.add_node("process", process)
# Fan out directly from START: one Send per item, all run in parallel
builder.add_conditional_edges(
    START,
    lambda s: [Send("process", {"item": item, "results": []}) for item in s["items"]],
)
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({"items": ["a", "b", "c"], "results": []})
print(result["results"])  # ['processed:a', 'processed:b', 'processed:c']
```

---

## `EphemeralValue`

```python
from langgraph.channels.ephemeral_value import EphemeralValue
```

Stores the value written to it in the **previous** step, then clears itself at the start of the next step if no new write arrives. Use it for one-shot trigger signals that should only be visible for a single step.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import EphemeralValue


class State(TypedDict):
    data: str
    # trigger is set by one node and visible to the next node only
    trigger: Annotated[str | None, EphemeralValue(str)]
```

Full constructor signature:

```python
EphemeralValue(typ: Any, guard: bool = True)
```

- `guard=True` (default) — raises `InvalidUpdateError` if two nodes write to the channel in the same super-step.
- `guard=False` — silently takes the last write when multiple nodes write concurrently.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.channels import EphemeralValue
from typing import Annotated
from typing_extensions import TypedDict


class S(TypedDict):
    msg: str
    flag: Annotated[bool | None, EphemeralValue(bool)]


def setter(state: S) -> dict:
    # Sets flag; it will be visible to downstream nodes this step
    return {"flag": True, "msg": "set"}


def reader(state: S) -> dict:
    # flag is True here (set by the previous node in the same run)
    print("flag:", state["flag"])
    return {}


def clearer_check(state: S) -> dict:
    # flag is None here — cleared because no node wrote to it this step
    print("flag after clear:", state["flag"])
    return {}


builder = StateGraph(S)
builder.add_node("setter", setter)
builder.add_node("reader", reader)
builder.add_node("check", clearer_check)
builder.add_edge(START, "setter")
builder.add_edge("setter", "reader")
builder.add_edge("reader", "check")
builder.add_edge("check", END)

graph = builder.compile()
graph.invoke({"msg": "", "flag": None})
# Prints:
#   flag: True      (setter → reader, same run)
#   flag after clear: None  (second run, flag expired)
```

---

## `NamedBarrierValue`

```python
from langgraph.channels.named_barrier_value import NamedBarrierValue
```

A synchronization channel that becomes **available** only after every string in a predefined `names` set has been written to it at least once. Until all names are seen, the channel raises `EmptyChannelError` and downstream nodes that depend on it will not run.

After the channel is consumed (its value read by a dependent step), the `seen` set resets — making it a reusable one-shot barrier per step.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import NamedBarrierValue


class Pipeline(TypedDict):
    # This channel becomes available only once BOTH "fetch" and "validate" have written to it
    ready: Annotated[None, NamedBarrierValue(str, names={"fetch", "validate"})]
    data: str
    valid: bool
```

Writing to a `NamedBarrierValue` channel:

```python
def fetch(state): return {"ready": "fetch", "data": "raw_data"}
def validate(state): return {"ready": "validate", "valid": True}

# Only after BOTH nodes run (writing "fetch" and "validate") does the barrier open.
# A node that reads `ready` will not be scheduled until the barrier is satisfied.
```

Any write that is not in `names` raises `InvalidUpdateError` immediately.

Full constructor:

```python
NamedBarrierValue(typ: type[str], names: set[str])
```

- `typ` — the element type. In practice always `str` (the token strings each writer sends).
- `names` — the complete set of expected writers. Every element in this set must be written before `get()` returns.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.channels import NamedBarrierValue
from typing import Annotated
from typing_extensions import TypedDict


class S(TypedDict):
    result_a: str
    result_b: str
    # Barrier: wait for both workers before the combiner runs
    done: Annotated[None, NamedBarrierValue(str, names={"worker_a", "worker_b"})]


def worker_a(state: S) -> dict:
    return {"result_a": "from_a", "done": "worker_a"}


def worker_b(state: S) -> dict:
    return {"result_b": "from_b", "done": "worker_b"}


def combiner(state: S) -> dict:
    # Runs only after both workers have written their "done" token
    print("Both done:", state["result_a"], state["result_b"])
    return {}


builder = StateGraph(S)
builder.add_node("worker_a", worker_a)
builder.add_node("worker_b", worker_b)
builder.add_node("combiner", combiner)
builder.add_edge(START, "worker_a")
builder.add_edge(START, "worker_b")
# combiner depends on "done" — it is gated by the barrier
builder.add_edge(["worker_a", "worker_b"], "combiner")
builder.add_edge("combiner", END)

graph = builder.compile()
graph.invoke({"result_a": "", "result_b": "", "done": None})
```

---

## `AnyValue`

```python
from langgraph.channels.any_value import AnyValue
```

Like `LastValue`, but accepts multiple concurrent writes without raising. When two or more nodes write in the same super-step, the last write wins. The channel assumes all concurrent writers produce the same value — if they differ, the result is non-deterministic (last write depending on execution order).

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import AnyValue


class State(TypedDict):
    # All parallel workers write the same config flag; AnyValue avoids the concurrent-write error
    debug_mode: Annotated[bool, AnyValue(bool)]
    result: str
```

Usage:

```python
def node_a(state): return {"debug_mode": True, "result": "from_a"}
def node_b(state): return {"debug_mode": True, "result": "from_b"}
# Both write True to debug_mode — no error; result uses LastValue semantics and would error
```

`AnyValue` is appropriate for:
- Global flags that all nodes in a parallel fan-out write identically (e.g., a run-level `debug` or `dry_run` boolean).
- Computed properties derived from input that any node could reconstruct.

It is **not** appropriate when parallel nodes may write different values — use a reducer (`BinaryOperatorAggregate`) or a barrier (`NamedBarrierValue`) instead.

---

## Patterns

### 1. `operator.add` accumulator

Accumulate a list of results from parallel workers:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class State(TypedDict):
    items: list[str]
    scores: Annotated[list[float], operator.add]  # BinaryOperatorAggregate


def dispatch(state: State) -> list[Send]:
    return [Send("score", {"item": item}) for item in state["items"]]


def score(state: dict) -> dict:
    # Each parallel invocation appends its score list via operator.add
    return {"scores": [len(state["item"]) / 10.0]}


builder = StateGraph(State)
builder.add_node("dispatch", lambda s: {})   # no-op; conditional edge does the fan-out
builder.add_node("score", score)
builder.add_conditional_edges("dispatch", dispatch)
builder.add_edge(START, "dispatch")
builder.add_edge("score", END)

graph = builder.compile()
result = graph.invoke({"items": ["hello", "world", "!"], "scores": []})
print(result["scores"])   # [0.5, 0.5, 0.1]
```

### 2. `Topic` as a fan-in event buffer

Collect structured events from parallel workers into one list for downstream processing:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import Topic
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class State(TypedDict):
    urls: list[str]
    events: Annotated[list[dict], Topic(dict)]


def crawl(state: dict) -> dict:
    url = state["url"]
    # Emit a structured event; Topic collects all of them
    return {"events": {"url": url, "status": "ok", "length": len(url)}}


def summarize(state: State) -> dict:
    print(f"Crawled {len(state['events'])} pages")
    return {}


builder = StateGraph(State)
builder.add_node("crawl", crawl)
builder.add_node("summarize", summarize)
# then="summarize" ensures summarize runs once after ALL Send-spawned crawl tasks finish,
# not after each individual one completes.
builder.add_conditional_edges(
    START,
    lambda s: [Send("crawl", {"url": u, "events": []}) for u in s["urls"]],
    then="summarize",
)
builder.add_edge("summarize", END)

graph = builder.compile()
graph.invoke({"urls": ["http://a.com", "http://b.com"], "events": []})
```

### 3. `EphemeralValue` as a one-step trigger

Use an ephemeral channel to pass a signal from one node to the next without polluting permanent state:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import EphemeralValue
from langgraph.graph import StateGraph, START, END


class S(TypedDict):
    doc: str
    # Signal: set by "fetch", consumed by "process", gone by "save"
    doc_ready: Annotated[bool | None, EphemeralValue(bool)]


def fetch(state: S) -> dict:
    return {"doc": "raw content", "doc_ready": True}


def process(state: S) -> dict:
    # doc_ready is True here — set by fetch
    if state["doc_ready"]:
        return {"doc": state["doc"].upper()}
    return {}


def save(state: S) -> dict:
    # doc_ready is None here — ephemeral, cleared after process ran
    assert state["doc_ready"] is None
    return {}


builder = StateGraph(S)
builder.add_node("fetch", fetch)
builder.add_node("process", process)
builder.add_node("save", save)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", "process")
builder.add_edge("process", "save")
builder.add_edge("save", END)

graph = builder.compile()
graph.invoke({"doc": "", "doc_ready": None})
```

### 4. `NamedBarrierValue` — explicit N-of-N fan-in

Wait for results from exactly N named nodes before proceeding:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import NamedBarrierValue
from langgraph.graph import StateGraph, START, END

WORKERS = {"alpha", "beta", "gamma"}


class S(TypedDict):
    inputs: list[str]
    # Use a merge reducer so parallel workers can each add their own key without collision
    outputs: Annotated[dict, lambda a, b: {**a, **b}]
    # Barrier: all three workers must report in
    barrier: Annotated[None, NamedBarrierValue(str, names=WORKERS)]


def make_worker(name: str):
    def worker(state: S) -> dict:
        return {
            "outputs": {name: f"result_from_{name}"},  # merged by reducer
            "barrier": name,   # write our name to the barrier channel
        }
    worker.__name__ = name
    return worker


def combiner(state: S) -> dict:
    print("All results:", state["outputs"])
    return {}


builder = StateGraph(S)
for w in WORKERS:
    builder.add_node(w, make_worker(w))
    builder.add_edge(START, w)
    builder.add_edge(w, "combiner")

builder.add_node("combiner", combiner)
builder.add_edge("combiner", END)

graph = builder.compile()
graph.invoke({"inputs": [], "outputs": {}, "barrier": None})
```

### 5. `AnyValue` — parallel-safe shared configuration flag

When multiple parallel nodes all need to write the same read-only flag, use `AnyValue` to avoid the `InvalidUpdateError`:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import AnyValue
from langgraph.graph import StateGraph, START, END


class Config(TypedDict):
    # All parallel nodes inherit and re-emit this flag
    dry_run: Annotated[bool, AnyValue(bool)]
    result_a: str
    result_b: str


def node_a(state: Config) -> dict:
    return {"dry_run": state["dry_run"], "result_a": "done" if not state["dry_run"] else "skip"}


def node_b(state: Config) -> dict:
    return {"dry_run": state["dry_run"], "result_b": "done" if not state["dry_run"] else "skip"}


builder = StateGraph(Config)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge(["a", "b"], END)

graph = builder.compile()
print(graph.invoke({"dry_run": True, "result_a": "", "result_b": ""}))
```

---

## `UntrackedValue`

```python
from langgraph.channels.untracked_value import UntrackedValue
# also re-exported from:
from langgraph.channels import UntrackedValue
```

Like `LastValue`, but **never written to checkpoints**. Use it for large, derived, or ephemeral values that you want visible to nodes during a run but don't need to persist between invocations.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import UntrackedValue


class State(TypedDict):
    query: str
    # embedding is computed each run; no need to persist it in checkpoints
    embedding: Annotated[list[float] | None, UntrackedValue(list)]
```

Full constructor signature:

```python
UntrackedValue(typ: Any, guard: bool = True)
```

- `guard=True` (default) — raises `InvalidUpdateError` if two nodes write to the channel in the same super-step. Identical to `LastValue` semantics.
- `guard=False` — silently takes the last write; no error on concurrent writes.

**Behaviour at a glance:**

| Property | `LastValue` | `UntrackedValue` |
|---|---|---|
| Concurrent writes | Error | Error (`guard=True`) or last wins (`guard=False`) |
| Stored in checkpoint | ✅ Yes | ❌ No |
| Survives across `invoke` calls | ✅ Yes | ❌ No — cleared on resume |
| Good for | Persistent scalar state | Large computed / sensitive values |

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels import UntrackedValue
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class S(TypedDict):
    text: str
    # embedding is computed at runtime but never checkpointed
    embedding: Annotated[list[float] | None, UntrackedValue(list)]
    label: str


def embed(state: S) -> dict:
    """Compute a toy embedding from the text — not persisted."""
    vec = [len(state["text"]) / 100.0, hash(state["text"]) % 1000 / 1000.0]
    return {"embedding": vec}


def classify(state: S) -> dict:
    """Classify based on the embedding — embedding is still available this run."""
    vec = state["embedding"]
    label = "long" if vec and vec[0] > 0.5 else "short"
    return {"label": label}


builder = StateGraph(S)
builder.add_node("embed", embed)
builder.add_node("classify", classify)
builder.add_edge(START, "embed")
builder.add_edge("embed", "classify")
builder.add_edge("classify", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}

result = graph.invoke({"text": "A fairly long sentence that exceeds fifty characters", "embedding": None, "label": ""}, cfg)
print(result["label"])       # "long"
print(result["embedding"])   # [0.52, 0.xxx]  — present in the final return value

# The embedding is NOT in the checkpointed state:
snap = graph.get_state(cfg)
print(snap.values.get("embedding"))   # None — not persisted
```

**When to use `UntrackedValue`:**

- Embeddings, token counts, or other computed properties derived purely from other state.
- Sensitive values (API keys, credentials) that you want available to nodes during a run without persisting them to disk.
- Large binary blobs that would bloat the checkpoint without adding time-travel value.

---

## Gotchas

- **`LastValue` raises on concurrent writes.** Two parallel nodes writing to the same `LastValue` key in the same super-step will crash the graph. Add a reducer or use `AnyValue` / `Topic` instead.
- **`Topic(accumulate=True)` grows unbounded.** Once enabled, the list never clears automatically. Wrap it in an `Overwrite` reset if you need to cap it.
- **`Topic` clears between steps, not between graph invocations.** With a checkpointer, each new `invoke` call can re-accumulate the list unless you reset it in your first node.
- **`NamedBarrierValue` resets after being consumed.** It acts as a one-shot barrier per super-step. If the same set of nodes runs again in a later step, the barrier will collect their writes again.
- **All writes to `NamedBarrierValue` must be from within `names`.** Any write whose string value is not in the `names` set raises `InvalidUpdateError` immediately.
- **`EphemeralValue(guard=True)` still errors on concurrent writes.** Set `guard=False` if multiple parallel nodes may write the trigger in the same step.
- **Channel types are internal implementation details.** You should not store `BaseChannel` instances in state values — they are graph-level constructs, not user-visible state. Your state dict holds the channel's value, not the channel object.
- **`AnyValue` is non-deterministic when writers differ.** If two nodes concurrently write different values, the result depends on task execution order. Use it only when you can guarantee all writers produce the same value.
- **`BinaryOperatorAggregate` initial value is the zero of the type.** For `int` that's `0`, for `list` that's `[]`, for `str` that's `""`. There is no way to set a non-zero default in the channel itself — set the initial value in your `invoke` call instead.
- **`UntrackedValue` fields will be `None` (or their zero value) after resuming from a checkpoint.** Because the channel is never persisted, any resume or replay will start with the field unset. Always handle `None` defensively.
- **`UntrackedValue` still appears in the `invoke`/`stream` return value** — only checkpointing is skipped. The value is computed and available within the run, just not stored for the next one.

---

## Breaking changes

| Version | Change |
|---|---|
| 1.2 | `UntrackedValue` added — same semantics as `LastValue` but never written to checkpoints. Use for computed or sensitive fields you don't want persisted. |
| 1.0 | `Topic`, `EphemeralValue`, `NamedBarrierValue`, `AnyValue` moved from `langgraph.channels` to their own submodules but remain re-exported at `langgraph.channels`. Existing imports unaffected. |
| 0.6 | `DeltaChannel` added (beta) — a write-efficient channel that stores only deltas and reconstructs state by replaying ancestor writes. Not covered here; see the beta warning in source. |
| 0.2 | `BinaryOperatorAggregate` introduced; `add_messages` became the canonical reducer for message lists. |

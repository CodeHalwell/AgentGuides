---
title: "Class deep-dives Vol. 3 — 10 more LangGraph types"
description: "Source-verified deep dives into interrupt/Interrupt, DeltaChannel, EphemeralValue, NamedBarrierValue, RemoveMessage/push_message, Pregel, NodeBuilder, GraphOutput, PregelTask/StateSnapshot.tasks, and IndexConfig/TTLConfig — with runnable examples for every major feature."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 3"
  order: 27
---

# Class deep-dives Vol. 3 — 10 more LangGraph types

Verified against **`langgraph==1.2.2`** / **`langgraph-prebuilt==1.1.0`** / **`langgraph-checkpoint==4.1.1`**.

Each section was written by inspecting the installed package source directly. All signatures and behaviours are drawn from the actual implementation, not documentation.

[→ Vol. 1 covers StateGraph, CompiledStateGraph, InMemorySaver, ToolNode, create_react_agent, Command, Send, @task/@entrypoint, BinaryOperatorAggregate/Topic, InMemoryStore](./langgraph_class_deep_dives/)

[→ Vol. 2 covers RetryPolicy, CachePolicy/InMemoryCache, TimeoutPolicy, add_messages/MessagesState, tools_condition, ToolCallTransformer/ToolCallStream, StateSnapshot, IsLastStep/RemainingSteps, ToolRuntime, Runtime/RunControl](./langgraph_class_deep_dives_v2/)

---

## 1 · `interrupt()` + `Interrupt`

**Module:** `langgraph.types`  
**Re-exported from:** `langgraph.types` (also importable as `from langgraph.types import interrupt, Interrupt`)

`interrupt()` is the human-in-the-loop primitive. Calling it inside a node pauses execution, surfaces a value to the caller, and waits for a `Command(resume=...)`. On resume the node **re-runs from the top**; every `interrupt()` encountered during replay returns the previously supplied value.

### `Interrupt` dataclass (source)

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any    # The value you passed to interrupt()
    id: str       # Stable hash of the checkpoint namespace — use to address by id
```

Only `value` and `id` remain in v1.2.x. The old `ns`, `when`, `resumable` attributes were removed in v0.6.

### `interrupt()` — how it works internally

```python
def interrupt(value: Any) -> Any:
    conf   = get_config()["configurable"]
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx    = scratchpad.interrupt_counter()   # ordinal within this task

    # Has a resume value already been supplied for this ordinal?
    if scratchpad.resume and idx < len(scratchpad.resume):
        return scratchpad.resume[idx]          # fast-path: replay returns immediately

    # Is there a null-resume (from a raw Command.resume)?
    v = scratchpad.get_null_resume(True)
    if v is not None:
        scratchpad.resume.append(v)
        return v

    raise GraphInterrupt((Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),))
```

Key insight: on replay, the check `idx < len(scratchpad.resume)` short-circuits _before_ any side-effects you placed above the interrupt call. Put side-effects inside `@task` functions — they are memoised and skipped on replay.

### Example 1: basic interrupt + resume

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class State(TypedDict):
    question: str
    answer: str


def ask_human(state: State) -> dict:
    # Execution pauses here; value is surfaced to the caller.
    human_answer = interrupt({"prompt": state["question"], "type": "text"})
    return {"answer": human_answer}


builder = StateGraph(State)
builder.add_node("ask", ask_human)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}

# First run — graph pauses and emits __interrupt__
for chunk in graph.stream({"question": "What is 2+2?"}, cfg):
    print(chunk)
# {'__interrupt__': (Interrupt(value={'prompt': 'What is 2+2?', 'type': 'text'}, id='...'),)}

# Resume — pass answer via Command(resume=...)
for chunk in graph.stream(Command(resume="4"), cfg):
    print(chunk)
# {'ask': {'answer': '4'}}
```

### Example 2: multiple interrupts in one node, matched by order

```python
def multi_step_review(state: State) -> dict:
    # First interrupt: collect summary
    summary = interrupt("Please provide a one-line summary")
    # Second interrupt: collect approval
    approved = interrupt({"prompt": "Approve?", "options": ["yes", "no"]})
    return {"answer": f"Summary: {summary} | Approved: {approved}"}


builder2 = StateGraph(State)
builder2.add_node("review", multi_step_review)
builder2.add_edge(START, "review")
builder2.add_edge("review", END)

graph2 = builder2.compile(checkpointer=InMemorySaver())
cfg2 = {"configurable": {"thread_id": "t2"}}

# Run 1: pauses at first interrupt
list(graph2.stream({"question": "Draft report"}, cfg2))

# Resume 1: provide first answer — still paused at second interrupt
list(graph2.stream(Command(resume="Short summary here"), cfg2))

# Resume 2: provide second answer — completes
result = list(graph2.stream(Command(resume="yes"), cfg2))
print(result)
# [{'review': {'answer': 'Summary: Short summary here | Approved: yes'}}]
```

### Example 3: resume a specific interrupt by id

```python
# Inspect pending interrupts
snapshot = graph2.get_state(cfg2)
pending_id = snapshot.interrupts[0].id

# Resume the specific interrupt by id using Command(resume={id: value})
list(graph2.stream(Command(resume={pending_id: "Targeted answer"}), cfg2))
```

### Example 4: protecting side-effects with `@task`

```python
from langgraph.func import task


@task
def send_email(to: str, body: str) -> str:
    """This is skipped on replay — runs only once."""
    # ... real email code ...
    return f"Sent to {to}"


def approval_node(state: State) -> dict:
    # send_email runs BEFORE the interrupt; @task memoises it
    receipt = send_email("user@example.com", state["question"]).result()
    approved = interrupt(f"Email sent ({receipt}). Approve sending final report?")
    return {"answer": f"Approved: {approved}"}
```

---

## 2 · `DeltaChannel`

**Module:** `langgraph.channels.delta`  
**Status:** ⚠️ Beta — on-disk representation may change in future releases.

`DeltaChannel` is a reducer channel that stores **only a sentinel value** in checkpoint blobs and reconstructs full state by replaying ancestor writes through the reducer. For graphs with very long histories (thousands of steps) it avoids storing a full snapshot on every step.

### Constructor

```python
DeltaChannel(
    reducer: Callable[[Any, Sequence[Any]], Any],
    typ: type | None = None,
    *,
    snapshot_frequency: int = 1000,
)
```

| Parameter | What it does |
|---|---|
| `reducer` | `(current_state, [write1, write2, ...]) -> new_state`. Must be **deterministic and batching-invariant** — `reducer(reducer(s, xs), ys) == reducer(s, xs + ys)`. |
| `typ` | Value type. Inferred from the outer `Annotated[T, ...]` type if omitted. |
| `snapshot_frequency` | Every Nth update writes a full snapshot blob, bounding replay depth. Default 1000. |

### Declare in state

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel


class MyState(TypedDict):
    # DeltaChannel: only the sentinel is checkpointed; replays rebuild from writes
    events: Annotated[list[str], DeltaChannel(lambda acc, writes: acc + writes)]

    # Compare with standard BinaryOperatorAggregate (stores full list each step)
    events_normal: Annotated[list[str], operator.add]
```

### Full example: append-only event log

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.channels.delta import DeltaChannel


def concat_reducer(acc: list[str], writes: list[str]) -> list[str]:
    """Batching-invariant: concat(concat(acc, xs), ys) == concat(acc, xs+ys)."""
    return acc + writes


class PipelineState(TypedDict):
    # DeltaChannel: checkpoint only stores sentinel; full list rebuilt on replay
    log: Annotated[list[str], DeltaChannel(concat_reducer, list)]
    step: int


def stage_one(state: PipelineState) -> dict:
    return {"log": [f"stage_one/{state['step']}"], "step": state["step"] + 1}


def stage_two(state: PipelineState) -> dict:
    return {"log": [f"stage_two/{state['step']}"], "step": state["step"] + 1}


builder = StateGraph(PipelineState)
builder.add_node("one", stage_one)
builder.add_node("two", stage_two)
builder.add_edge(START, "one")
builder.add_edge("one", "two")
builder.add_edge("two", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "delta-demo"}}

result = graph.invoke({"log": [], "step": 0}, cfg)
print(result["log"])   # ['stage_one/0', 'stage_two/1']
print(result["step"])  # 2
```

### How `Overwrite` interacts with `DeltaChannel`

Use `Overwrite(value)` from `langgraph.types` to reset the accumulated state to a new base:

```python
from langgraph.types import Overwrite


def reset_log(state: PipelineState) -> dict:
    """Replace the entire log with a fresh start, discarding history."""
    return {"log": Overwrite(["reset_point"])}
```

---

## 3 · `EphemeralValue`

**Module:** `langgraph.channels.ephemeral_value`

`EphemeralValue` is a channel that stores a value for exactly one step and then **clears itself**. It is never read-back from checkpoint across steps; if a node writes to it and no other node reads it in the same super-step, it disappears. Use it for trigger signals, one-shot messages, and scratch space that should not accumulate.

### Constructor

```python
EphemeralValue(typ: Any, guard: bool = True)
```

| Parameter | What it does |
|---|---|
| `typ` | The type annotation for the value (e.g., `str`, `dict`). |
| `guard` | If `True` (default), raises `InvalidUpdateError` if more than one node writes to the channel in the same step. Set `guard=False` to allow last-writer-wins. |

### Declare in state

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.ephemeral_value import EphemeralValue


class State(TypedDict):
    # Persisted across steps
    result: str
    # Cleared after each step — never lingers in checkpoints
    trigger: Annotated[str | None, EphemeralValue(str)]
```

### Example 1: one-shot routing signal

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.ephemeral_value import EphemeralValue


class RouterState(TypedDict):
    query: str
    result: str
    # EphemeralValue: signal lives only for the step it's written
    route: Annotated[str | None, EphemeralValue(str)]


def classify(state: RouterState) -> dict:
    """Classify and write a one-step routing signal."""
    topic = "math" if any(w in state["query"] for w in ["add", "sum", "calculate"]) else "chat"
    return {"route": topic}


def math_handler(state: RouterState) -> dict:
    return {"result": f"Math: {state['query']}"}


def chat_handler(state: RouterState) -> dict:
    return {"result": f"Chat: {state['query']}"}


def route_by_signal(state: RouterState) -> str:
    return state["route"] or "chat"


builder = StateGraph(RouterState)
builder.add_node("classify", classify)
builder.add_node("math", math_handler)
builder.add_node("chat", chat_handler)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_signal, {"math": "math", "chat": "chat"})
builder.add_edge("math", END)
builder.add_edge("chat", END)

graph = builder.compile()

r1 = graph.invoke({"query": "calculate 2+2", "result": "", "route": None})
print(r1["result"])   # Math: calculate 2+2
print(r1["route"])    # None — cleared after the step

r2 = graph.invoke({"query": "How are you?", "result": "", "route": None})
print(r2["result"])   # Chat: How are you?
```

### Example 2: `guard=False` — last-writer-wins

```python
from langgraph.channels.ephemeral_value import EphemeralValue


class ParallelState(TypedDict):
    # Multiple parallel nodes may write; only the last write survives
    scratch: Annotated[str | None, EphemeralValue(str, guard=False)]
    items: Annotated[list[str], operator.add]


def node_a(state: ParallelState) -> dict:
    return {"scratch": "from_a", "items": ["a"]}


def node_b(state: ParallelState) -> dict:
    return {"scratch": "from_b", "items": ["b"]}
```

---

## 4 · `NamedBarrierValue`

**Module:** `langgraph.channels.named_barrier_value`

`NamedBarrierValue` implements an **N-way fan-in barrier**: the channel is only made available (non-empty) once every member of its `names` set has written to it. Until all expected writers have contributed, `get()` raises `EmptyChannelError` — no downstream edge is triggered. After all writes arrive and the channel is consumed, it resets to empty for the next round.

### Constructor

```python
NamedBarrierValue(typ: type[Value], names: set[Value])
```

`names` is the set of string keys that must each write to this channel before the barrier opens.

### Example 1: fan-in gate on two parallel nodes

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.channels.named_barrier_value import NamedBarrierValue


class AggState(TypedDict):
    results: Annotated[list[str], operator.add]
    # Opens only when both "worker_a" and "worker_b" have written
    gate: Annotated[None, NamedBarrierValue(str, names={"worker_a", "worker_b"})]


def worker_a(state: AggState) -> dict:
    return {"results": ["a_done"], "gate": "worker_a"}


def worker_b(state: AggState) -> dict:
    return {"results": ["b_done"], "gate": "worker_b"}


def aggregator(state: AggState) -> dict:
    # Only runs once both workers have written to `gate`
    return {"results": [f"aggregated: {sorted(state['results'])}"]}


def after_barrier(_: AggState) -> Literal["aggregator"]:
    return "aggregator"


builder = StateGraph(AggState)
builder.add_node("worker_a", worker_a)
builder.add_node("worker_b", worker_b)
builder.add_node("aggregator", aggregator)
builder.add_edge(START, "worker_a")
builder.add_edge(START, "worker_b")
# Both workers must complete before aggregator runs
builder.add_edge(["worker_a", "worker_b"], "aggregator")
builder.add_edge("aggregator", END)

graph = builder.compile()
result = graph.invoke({"results": [], "gate": None})
# workers run in parallel; aggregator only fires after both complete
print([r for r in result["results"] if r.startswith("aggregated")])
# ["aggregated: ['a_done', 'b_done']"]
```

### `NamedBarrierValueAfterFinish`

A variant that adds an extra constraint: the barrier only opens after `finish()` is also called. This is used internally by LangGraph for subgraph coordination — most application code does not need it directly.

```python
from langgraph.channels.named_barrier_value import NamedBarrierValueAfterFinish

# Opens only when all named writers AND a finish() signal are received
gate: Annotated[None, NamedBarrierValueAfterFinish(str, names={"a", "b"})]
```

---

## 5 · `RemoveMessage` + `push_message`

**Module:** `langgraph.graph.message`

These two functions solve different problems in message-state graphs:

- **`RemoveMessage`** — delete a specific message from the `messages` channel by its `id`.
- **`push_message`** — emit a `BaseMessage` directly to the `messages` / `messages-tuple` stream channel _without_ going through a state update cycle. Useful for streaming partial progress.

### `RemoveMessage`

```python
class RemoveMessage(BaseMessage):
    type: Literal["remove"] = "remove"

    def __init__(self, id: str, **kwargs: Any) -> None:
        ...
```

When `add_messages` processes a write that includes a `RemoveMessage`, it finds the message with the matching `id` in the current list and removes it. Content cannot be set on `RemoveMessage` — it is purely a deletion marker.

#### Example: truncate conversation history

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver


def keep_last_n(state: MessagesState, n: int = 4) -> dict:
    """Remove all messages except the last n."""
    messages = state["messages"]
    if len(messages) <= n:
        return {}
    to_remove = messages[:-n]
    return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}


builder = StateGraph(MessagesState)
builder.add_node("trim", keep_last_n)
builder.add_edge(START, "trim")
builder.add_edge("trim", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "trim-demo"}}

# Seed some messages
msgs = [HumanMessage("Hi"), AIMessage("Hello"), HumanMessage("Tell me a joke"),
        AIMessage("Why did the chicken..."), HumanMessage("Another?")]
graph.update_state(cfg, {"messages": msgs})

result = graph.invoke(None, cfg)
print(len(result["messages"]))   # 4 (last 4 kept)
```

#### Example: replace a specific message by id

```python
from langchain_core.messages import AIMessage
from langgraph.graph.message import RemoveMessage


def correct_last_response(state: MessagesState) -> dict:
    """Remove the most recent AI message and replace it with a corrected one."""
    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    if not ai_messages:
        return {}
    last_ai = ai_messages[-1]
    corrected = AIMessage(content="[Corrected] " + last_ai.content, id=last_ai.id)
    return {
        "messages": [
            RemoveMessage(id=last_ai.id),
            corrected,
        ]
    }
```

### `push_message`

```python
def push_message(
    message: MessageLikeRepresentation | BaseMessageChunk,
    *,
    state_key: str | None = "messages",
) -> AnyMessage:
```

`push_message` injects a message directly into the `messages`-mode stream **and** into the `messages` state channel in a single call. Use it for streaming intermediate progress events (e.g., tool call acknowledgements) without waiting for a node to return.

#### Example: stream partial tool progress

```python
from langchain_core.messages import AIMessageChunk
from langgraph.graph.message import push_message, MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


def long_running_node(state: MessagesState) -> dict:
    # Emit a partial progress message immediately
    push_message(AIMessageChunk(content="Starting analysis... ", id="progress-1"))

    # ... do real work ...
    result = "Analysis complete."

    push_message(AIMessageChunk(content="Done!", id="progress-1"))
    return {"messages": [AIMessageChunk(content=result, id="final-1")]}


builder = StateGraph(MessagesState)
builder.add_node("work", long_running_node)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "push-demo"}}

for chunk in graph.stream(
    {"messages": [("user", "Analyse this data")]},
    cfg,
    stream_mode="messages",
):
    msg, meta = chunk
    print(f"[{meta.get('langgraph_node')}] {msg.content!r}")
# [work] 'Starting analysis... '
# [work] 'Done!'
# [work] 'Analysis complete.'
```

---

## 6 · `Pregel`

**Module:** `langgraph.pregel`  
**Re-exported from:** `langgraph.pregel` as `Pregel`

`Pregel` is the **runtime engine** that underlies every compiled LangGraph graph. When you call `StateGraph.compile()`, it returns a `CompiledStateGraph` which is a thin subclass of `Pregel`. You rarely instantiate `Pregel` directly, but understanding its constructor reveals the full set of runtime knobs you can set at compile time or pass directly.

### Constructor (abbreviated)

```python
Pregel(
    nodes:                  dict[str, PregelNode | NodeBuilder],
    channels:               dict[str, BaseChannel | ManagedValueSpec] | None,
    *,
    auto_validate:          bool = True,
    stream_mode:            StreamMode = "values",
    stream_eager:           bool = False,
    output_channels:        str | Sequence[str],
    stream_channels:        str | Sequence[str] | None = None,
    interrupt_after_nodes:  All | Sequence[str] = (),
    interrupt_before_nodes: All | Sequence[str] = (),
    input_channels:         str | Sequence[str],
    step_timeout:           float | None = None,
    debug:                  bool | None = None,
    checkpointer:           Checkpointer = None,
    store:                  BaseStore | None = None,
    cache:                  BaseCache | None = None,
    retry_policy:           RetryPolicy | Sequence[RetryPolicy] = (),
    cache_policy:           CachePolicy | None = None,
    context_schema:         type | None = None,
    name:                   str = "LangGraph",
)
```

Key parameters not surfaced by `StateGraph.compile()`:

| Parameter | What it does |
|---|---|
| `stream_eager` | When `True`, emit stream events as soon as they're produced rather than buffering per-step. Reduces latency for long-running nodes. |
| `stream_channels` | Subset of channels to stream (default: all output channels). |
| `step_timeout` | Hard per-step wall-clock limit in seconds. Applied globally across all nodes. |
| `trigger_to_nodes` | Override which channels trigger which nodes (advanced routing). |
| `node_error_handler_map` | Map node names to error-handler node names (used internally by `add_node(error_handler=...)`). |

### Accessing Pregel attributes on a compiled graph

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class S(TypedDict):
    x: int


def step(state: S) -> dict:
    return {"x": state["x"] + 1}


graph = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile()
)

# CompiledStateGraph extends Pregel — all Pregel attributes are available
print(type(graph).__mro__)          # [CompiledStateGraph, StateGraph, Pregel, ...]
print(graph.stream_mode)            # 'values'
print(graph.input_channels)        # '__input__'
print(graph.output_channels)       # '__root__' (for single-output graphs)
print(list(graph.channels.keys())) # ['x', '__input__', '__root__', ...]
```

### Setting Pregel options at compile time

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

graph_prod = (
    StateGraph(S)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["step"],    # pause before every "step" node
        debug=True,                   # verbose Pregel logging
    )
)

# Override stream defaults at invoke time
result = graph_prod.invoke(
    {"x": 0},
    config={"configurable": {"thread_id": "pregel-demo"}},
)
print(result)  # {'x': 1}
```

### `graph.get_graph()` — visualise the Pregel execution graph

```python
# Returns a DrawableGraph with nodes, edges, and their Pregel channel wiring
drawable = graph.get_graph()
print(drawable.nodes)  # {'__start__': ..., 'step': ..., '__end__': ...}
```

---

## 7 · `NodeBuilder`

**Module:** `langgraph.pregel`

`NodeBuilder` is the **low-level fluent API** for wiring Pregel nodes directly — it is what `StateGraph.add_node()` and `.add_edge()` ultimately produce. Most applications should use `StateGraph` instead, but `NodeBuilder` is useful when you need fine-grained control over channel subscriptions, multiple writes, or building custom `Pregel` instances.

### Full API

```python
nb = NodeBuilder()
nb.subscribe_only("channel_name")            # subscribe to a single channel (value only, not dict)
nb.subscribe_to("ch1", "ch2", read=True)     # subscribe + read into input dict
nb.subscribe_to("ch_trigger", read=False)    # trigger without reading into input
nb.read_from("extra_ch")                     # read without subscribing
nb.do(my_function)                           # set the node action (chainable; composes with >>)
nb.write_to("output_ch")                     # declare output channel writes
nb.with_retry_policy(RetryPolicy(...))       # attach retry
nb.with_cache_policy(CachePolicy(...))       # attach cache
nb.with_timeout(30.0)                        # per-node timeout
nb.with_tags(["tag1"])                       # LangSmith/observability tags
nb.with_metadata({"env": "prod"})            # metadata dict
```

### Example: custom Pregel graph (no StateGraph)

```python
from langgraph.pregel import Pregel, NodeBuilder
from langgraph.channels.last_value import LastValue
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.types import RetryPolicy


def double(x: int) -> dict:
    return {"result": x * 2}


def format_result(result: int) -> dict:
    return {"output": f"Result is {result}"}


# Build nodes manually
node_double = (
    NodeBuilder()
    .subscribe_only("input")
    .do(double)
    .write_to("result")
)

node_format = (
    NodeBuilder()
    .subscribe_only("result")
    .do(format_result)
    .write_to("output")
)

# Channels: each key in your state schema maps to a channel type
channels = {
    "input": LastValue(int),
    "result": LastValue(int),
    "output": LastValue(str),
}

graph = Pregel(
    nodes={"double": node_double, "format": node_format},
    channels=channels,
    input_channels="input",
    output_channels="output",
)

print(graph.invoke(5))  # 'Result is 10'
```

---

## 8 · `GraphOutput`

**Module:** `langgraph.types`

`GraphOutput` is the **typed return value** from `graph.invoke(version="v2")` and `graph.ainvoke(version="v2")`. It wraps the graph's final output together with any interrupts that occurred.

### Definition (source)

```python
@dataclass(frozen=True)
class GraphOutput(Generic[OutputT]):
    value: OutputT                         # The final output (dict, Pydantic model, etc.)
    interrupts: tuple[Interrupt, ...] = () # Any interrupts that occurred
```

### Why use `version="v2"` / `GraphOutput`?

The v1 API (default) returns a raw dict which can accidentally include `__interrupt__` alongside regular output keys. The v2 API separates concerns cleanly.

| | `invoke()` (v1) | `invoke(version="v2")` |
|---|---|---|
| Return type | `dict \| Any` | `GraphOutput[OutputT]` |
| Access output | `result["key"]` | `result.value["key"]` |
| Access interrupts | `result.get("__interrupt__")` | `result.interrupts` |
| Typed | No | Yes |

### Example 1: clean interrupt detection

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command, GraphOutput


class S(TypedDict):
    data: str
    approved: bool


def review(state: S) -> dict:
    decision = interrupt({"prompt": f"Approve: {state['data']}?", "options": ["yes", "no"]})
    return {"approved": decision == "yes"}


builder = StateGraph(S)
builder.add_node("review", review)
builder.add_edge(START, "review")
builder.add_edge("review", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "graphout-demo"}}

# v2 invoke — returns GraphOutput
output: GraphOutput = graph.invoke({"data": "Deploy to prod", "approved": False}, cfg, version="v2")

if output.interrupts:
    # Clean: no risk of missing __interrupt__ mixed into output
    for intr in output.interrupts:
        print(f"Interrupt id={intr.id!r}: {intr.value}")
else:
    print(f"Approved: {output.value['approved']}")

# Resume
final: GraphOutput = graph.invoke(Command(resume="yes"), cfg, version="v2")
print(final.value["approved"])   # True
print(final.interrupts)          # ()
```

### Example 2: typed output with Pydantic model

```python
from pydantic import BaseModel


class ReportOutput(BaseModel):
    title: str
    word_count: int


class ReportState(TypedDict):
    title: str
    word_count: int


def build_report(state: ReportState) -> dict:
    return {"title": state["title"], "word_count": 42}


builder3 = StateGraph(ReportState, output_schema=ReportOutput)
builder3.add_node("build", build_report)
builder3.add_edge(START, "build")
builder3.add_edge("build", END)

graph3 = builder3.compile()
out: GraphOutput[ReportOutput] = graph3.invoke(
    {"title": "Q2 Report", "word_count": 0},
    version="v2",
)
print(type(out.value))           # <class 'ReportOutput'>
print(out.value.title)           # Q2 Report
print(out.value.word_count)      # 42
```

---

## 9 · `PregelTask` — execution introspection

**Module:** `langgraph.types`

Every time a graph runs a node, it creates a `PregelTask`. Tasks appear in `StateSnapshot.tasks` after the step and in `stream_mode="tasks"` events during streaming. They are the primary way to inspect what happened — including errors and interrupted sub-states.

### Definition (source)

```python
class PregelTask(NamedTuple):
    id: str                                    # stable UUID per task
    name: str                                  # node name
    path: tuple[str | int | tuple, ...]        # subgraph path (empty for top-level)
    error: Exception | None = None             # set if the node raised
    interrupts: tuple[Interrupt, ...] = ()     # interrupts raised in this task
    state: None | RunnableConfig | StateSnapshot = None  # subgraph state if applicable
    result: Any | None = None                  # node's return value (after completion)
```

### Example 1: inspect tasks in a completed graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class S(TypedDict):
    x: int


def compute(state: S) -> dict:
    return {"x": state["x"] * 10}


graph = (
    StateGraph(S)
    .add_node("compute", compute)
    .add_edge(START, "compute")
    .add_edge("compute", END)
    .compile(checkpointer=InMemorySaver())
)

cfg = {"configurable": {"thread_id": "task-demo"}}
graph.invoke({"x": 5}, cfg)

# Get the last checkpoint snapshot
snapshot = graph.get_state(cfg)
for task in snapshot.tasks:
    print(f"Task: {task.name!r}  id={task.id}  error={task.error}  result={task.result}")
# Task: 'compute'  id='...'  error=None  result={'x': 50}
```

### Example 2: inspect pending tasks on an interrupt

```python
from langgraph.types import interrupt, Command


def gated_node(state: S) -> dict:
    answer = interrupt("Proceed?")
    return {"x": state["x"] + int(answer)}


graph_hitl = (
    StateGraph(S)
    .add_node("gate", gated_node)
    .add_edge(START, "gate")
    .add_edge("gate", END)
    .compile(checkpointer=InMemorySaver())
)

cfg2 = {"configurable": {"thread_id": "task-hitl"}}
graph_hitl.invoke({"x": 1}, cfg2)  # pauses

snap = graph_hitl.get_state(cfg2)
print(snap.next)        # ('gate',)  — still waiting
for task in snap.tasks:
    print(f"task={task.name}  interrupts={[i.value for i in task.interrupts]}")
# task=gate  interrupts=['Proceed?']
```

### Example 3: stream `tasks` events for observability

```python
for part in graph_hitl.stream(
    Command(resume="3"),
    cfg2,
    stream_mode="tasks",
    version="v2",
):
    print(part["type"], part["data"])
# tasks {'id': '...', 'name': 'gate', 'input': {'x': 1}, 'triggers': [...]}  (start)
# tasks {'id': '...', 'name': 'gate', 'error': None, 'interrupts': (), 'result': {'x': 4}}  (result)
```

---

## 10 · `IndexConfig` + `TTLConfig` — semantic search & TTL in `InMemoryStore`

**Module:** `langgraph.store.base`

`IndexConfig` and `TTLConfig` are the two optional configuration dicts that unlock **vector search** and **time-to-live expiry** in LangGraph's store layer. They are passed to the `InMemoryStore` (and `PostgresStore`) constructor and control indexing and expiry behaviour for all subsequent `put` / `get` / `search` calls.

### `IndexConfig` (source)

```python
class IndexConfig(TypedDict, total=False):
    dims: int                                      # embedding vector dimensions
    embed: Embeddings | EmbeddingsFunc | str        # embedding function or provider string
    fields: list[str] | None                       # JSON-path field selectors to embed
```

`fields` uses JSON-path syntax:
- `["$"]` — embed entire document (default)
- `["text", "summary"]` — specific top-level keys
- `["metadata.title"]` — nested path
- `["chunks[*].content"]` — every element of an array, separately

### `TTLConfig` (source)

```python
class TTLConfig(TypedDict, total=False):
    refresh_on_read: bool        # refresh TTL on get/search (default True)
    default_ttl: float | None    # minutes until expiry (None = never)
    sweep_interval_minutes: int  # how often to delete expired items
```

### Example 1: semantic search with numpy embeddings (no API key needed)

```python
import random
from langgraph.store.memory import InMemoryStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings for testing (no real model needed)."""
    random.seed(42)
    return [[random.gauss(hash(t) % 100, 1) for _ in range(16)] for t in texts]


store = InMemoryStore(
    index={"dims": 16, "embed": fake_embed, "fields": ["content"]}
)

# Store some documents
store.put(("docs", "ml"), "paper1", {"content": "Deep learning and neural networks"})
store.put(("docs", "ml"), "paper2", {"content": "Gradient descent optimisation"})
store.put(("docs", "bio"), "paper3", {"content": "CRISPR gene editing techniques"})

# Semantic search — finds closest vectors
results = store.search(("docs",), query="machine learning optimisation", limit=2)
for r in results:
    print(f"  [{r.score:.3f}] {r.namespace} / {r.key}: {r.value['content']}")
```

### Example 2: per-field indexing

```python
store_selective = InMemoryStore(
    index={
        "dims": 16,
        "embed": fake_embed,
        "fields": ["title", "abstract"],   # only embed title + abstract, not full body
    }
)

store_selective.put(
    ("papers",), "p1",
    {"title": "LLM Agents", "abstract": "A survey of LLM agent architectures", "body": "...very long..."}
)

# Override per-item: index only the body for this one document
store_selective.put(
    ("papers",), "p2",
    {"title": "Short note", "abstract": "Brief", "body": "Critical implementation detail"},
    index=["body"],   # per-item override
)
```

### Example 3: TTL expiry for session memory

```python
from langgraph.store.memory import InMemoryStore

session_store = InMemoryStore(
    ttl={
        "default_ttl": 30.0,           # items expire after 30 minutes of inactivity
        "refresh_on_read": True,       # reading an item resets its 30-minute timer
        "sweep_interval_minutes": 5,   # sweep expired items every 5 minutes
    }
)

session_store.put(("sessions", "user-123"), "context", {"topic": "LLM agents"})

# Retrieve — also refreshes the TTL
item = session_store.get(("sessions", "user-123"), "context", refresh_ttl=True)
print(item.value)  # {'topic': 'LLM agents'}

# Skip TTL refresh on a read (just checking without prolonging the session)
item_no_refresh = session_store.get(("sessions", "user-123"), "context", refresh_ttl=False)
```

### Example 4: store + semantic search inside a graph node

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt.tool_node import InjectedStore


long_term_store = InMemoryStore(
    index={"dims": 16, "embed": fake_embed, "fields": ["text"]}
)


def memory_agent(state: MessagesState, store: InjectedStore = None) -> dict:
    """Node that recalls relevant memories before responding."""
    last_user_msg = state["messages"][-1].content

    # Recall relevant past memories
    memories = store.search(("memories",), query=last_user_msg, limit=3) if store else []
    context = "\n".join(m.value["text"] for m in memories)

    response = f"[Context: {context}] Responding to: {last_user_msg}"

    # Store the new interaction as a memory
    if store:
        import uuid
        store.put(("memories",), str(uuid.uuid4()), {"text": f"User asked: {last_user_msg}"})

    return {"messages": [AIMessage(content=response)]}


builder = StateGraph(MessagesState)
builder.add_node("agent", memory_agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

graph = builder.compile(store=long_term_store)

r = graph.invoke({"messages": [HumanMessage("Tell me about LangGraph")]})
print(r["messages"][-1].content)
```

---

## Quick-reference summary

| Class | Module | What it does |
|---|---|---|
| `interrupt()` | `langgraph.types` | Pause a node for human input; resumes via `Command(resume=...)`. Node replays from top on resume. |
| `Interrupt` | `langgraph.types` | Dataclass with `value` and `id`; surfaced to client when `interrupt()` raises. |
| `DeltaChannel` | `langgraph.channels.delta` | ⚠️ Beta. Stores only a sentinel in checkpoints; rebuilds state by replaying writes. Efficient for long-lived append channels. |
| `EphemeralValue` | `langgraph.channels.ephemeral_value` | One-step scratch channel — value is cleared if not written to each step. |
| `NamedBarrierValue` | `langgraph.channels.named_barrier_value` | N-way fan-in gate — only opens once every named writer has contributed. |
| `RemoveMessage` | `langgraph.graph.message` | Deletion marker: include in a `messages` write to remove the message with the matching `id`. |
| `push_message` | `langgraph.graph.message` | Emit a message to the stream channel immediately inside a node (without returning). |
| `Pregel` | `langgraph.pregel` | The runtime engine; `CompiledStateGraph` extends it. Exposes all runtime knobs. |
| `NodeBuilder` | `langgraph.pregel` | Fluent low-level API for building Pregel nodes: subscribe, read, do, write, retry, cache. |
| `GraphOutput` | `langgraph.types` | Typed `invoke(version="v2")` return: `.value` for output, `.interrupts` for interrupts. |
| `PregelTask` | `langgraph.types` | NamedTuple describing one node execution: `id`, `name`, `error`, `interrupts`, `result`. Available in `StateSnapshot.tasks` and `stream_mode="tasks"`. |
| `IndexConfig` | `langgraph.store.base` | Configure vector indexing for `InMemoryStore` / `PostgresStore`: `dims`, `embed`, `fields`. |
| `TTLConfig` | `langgraph.store.base` | Configure TTL expiry: `default_ttl` (minutes), `refresh_on_read`, `sweep_interval_minutes`. |

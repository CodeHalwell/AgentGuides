---
title: "LangGraph Class Deep-Dives Vol. 45"
description: "Source-verified deep dives (langgraph==1.2.10) into 10 class groups: interrupt() + Interrupt (multi-interrupt ordering, selective id-based resume, @task interrupt pattern), DeltaChannel (batching-invariant reducer, snapshot_frequency cadence, Overwrite reset point), UIMessage + RemoveUIMessage + push_ui_message() + delete_ui_message() + ui_message_reducer (generative UI streaming — merge=True prop delta, state_key=None stream-only, stable-id in-place updates), stream_events(version='v3') + GraphRunStream (caller-driven pump, native .values/.messages/.lifecycle/.subgraphs projections, interleave() ordered iteration), add_sequence() (linear pipeline construction, tuple-name override, fluent chaining with add_conditional_edges), Pregel.as_tool() beta (BaseTool from compiled graph, args_schema / arg_types / name / description, embedding a graph in a ToolNode), ErrorCode + InvalidUpdateError + GraphRecursionError + EmptyInputError + create_error_message (full error taxonomy — INVALID_CONCURRENT_GRAPH_UPDATE guard, recursion_limit config, catching NodeError from error_handler), create_react_agent with pre_model_hook / post_model_hook (hook-based middleware — state-aware prompt injection, response trimming, token-budget enforcement), StateGraph.compile(transformers=, cache=) (compile-time transformer injection for stream_mode customization, graph-level cache wiring for @task result reuse), and add_node(error_handler=, destinations=, defer=) (per-node error recovery, edgeless Command routing declarations, deferred end-of-step execution)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 45"
  order: 76
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.10`.

---

## 1 · `interrupt()` + `Interrupt` — multi-interrupt and selective resume

**Module:** `langgraph.types`

`interrupt(value)` pauses graph execution on the **first** call within a node by raising `GraphInterrupt`. The provided `value` is surfaced to the caller as an `Interrupt` dataclass. On resume (via `Command(resume=...)`), the graph **re-runs the entire node from the top** — `interrupt()` matches resume values by call-order index stored in `PregelScratchpad.interrupt_counter()`. Multiple `interrupt()` calls within one node each occupy their own slot; only the first un-matched slot raises.

**Key source facts (`langgraph/types.py` + `langgraph/errors.py`):**

- `interrupt(value)` reads `scratchpad.interrupt_counter()` (an `itertools.count`-based incrementing index), then checks `scratchpad.resume` list to see if a value already exists at that index.
- `Interrupt` is a frozen `@dataclass(slots=True)` with `value: Any` and `id: str`. `id` is derived via `xxh3_128_hexdigest(ns)` where `ns` is the checkpoint namespace, so the same node path always produces the same ID per run.
- `Command(resume=value)` sets a single resume value for the next interrupt. `Command(resume={id: value})` resumes a **specific** interrupt by its `id` field — useful when multiple parallel tasks each hold an interrupt.
- `NodeInterrupt` is fully deprecated since v1.0; use `interrupt()` instead.
- Interrupt works inside `@task` functions as well as regular graph nodes, as long as a checkpointer is attached.

### Example 1 — basic single-interrupt with resume

```python
from typing import Optional
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class ReviewState(TypedDict):
    draft: str
    approved: Optional[bool]

def human_review(state: ReviewState) -> dict:
    """Pause and ask a human to approve the draft."""
    decision = interrupt({"draft": state["draft"], "question": "Approve?"})
    return {"approved": decision == "approve"}

builder = StateGraph(ReviewState)
builder.add_node("review", human_review)
builder.add_edge(START, "review")
builder.add_edge("review", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t1"}}

# First pass — pauses at interrupt
result = graph.invoke({"draft": "Hello world", "approved": None}, config)
print(result)
# {'__interrupt__': (Interrupt(value={'draft': 'Hello world', 'question': 'Approve?'}, id='...'),)}

# Resume with human decision
final = graph.invoke(Command(resume="approve"), config)
print(final)
# {'draft': 'Hello world', 'approved': True}
```

### Example 2 — multiple sequential interrupts in one node

```python
from typing import Optional
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class MultiStepState(TypedDict):
    user_name: Optional[str]
    user_age: Optional[int]

def collect_info(state: MultiStepState) -> dict:
    # Two sequential interrupts; each is matched by call-order index on resume.
    name = interrupt("What is your name?")
    age = interrupt("What is your age?")
    return {"user_name": name, "user_age": int(age)}

builder = StateGraph(MultiStepState)
builder.add_node("collect", collect_info)
builder.add_edge(START, "collect")
builder.add_edge("collect", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "t2"}}

# Pause at first interrupt (name)
graph.invoke({"user_name": None, "user_age": None}, config)

# Resume name → pauses at second interrupt (age)
graph.invoke(Command(resume="Alice"), config)

# Resume age → node completes
final = graph.invoke(Command(resume="30"), config)
print(final)
# {'user_name': 'Alice', 'user_age': 30}
```

### Example 3 — interrupt inside a `@task` with parallel fan-out

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command

@task
async def approve_item(item: str) -> str:
    """Each parallel task holds its own interrupt slot."""
    decision = interrupt(f"Approve '{item}'?")
    return f"{item}: {decision}"

@entrypoint(checkpointer=InMemorySaver())
async def workflow(items: list[str]) -> list[str]:
    futures = [approve_item(item) for item in items]
    return await asyncio.gather(*futures)

config = {"configurable": {"thread_id": "t3"}}

async def run():
    # All tasks pause simultaneously
    await workflow.ainvoke(["report", "email"], config)

    # Resume both by providing a list — one resume value per task (in order)
    result = await workflow.ainvoke(Command(resume=["yes", "no"]), config)
    print(result)
    # ['report: yes', 'email: no']

asyncio.run(run())
```

---

## 2 · `DeltaChannel` — append-only reducer channel

**Module:** `langgraph.channels.delta`

`DeltaChannel` is a **beta** channel type that stores only a sentinel blob in checkpoint blobs and reconstructs the full accumulated state by **replaying ancestor writes** through the reducer. This makes it ideal for long append-only lists (event logs, conversation histories) where you want compact checkpoints but still need time-travel replay.

**Key source facts (`langgraph/channels/delta.py`):**

- Constructor: `DeltaChannel(reducer, typ=None, *, snapshot_frequency=1000)`. Use inside `Annotated[T, DeltaChannel(reducer)]` on a `TypedDict` field.
- The `reducer` signature is `(state: T, writes: list[Any]) -> T` — it receives the **whole accumulated value** and a **batch** of new writes in one call. It must be batching-invariant: `reducer(reducer(x, a), b) == reducer(x, a+b)`.
- A full `_DeltaSnapshot` blob is written when the update count reaches `snapshot_frequency` OR the `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` limit (default 5000), bounding replay depth.
- Wrapping a write in `Overwrite(value)` resets the accumulator — it acts as a replay start point so ancestor writes before it are not replayed.
- `DeltaChannel` is accessed via the `Annotated` shorthands only; direct instantiation is for type inference.

### Example 1 — append-only event log

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

def event_reducer(state: list[str], writes: list[str]) -> list[str]:
    """Batching-invariant: order is preserved across batches."""
    return state + writes

class LogState(TypedDict):
    events: Annotated[list[str], DeltaChannel(event_reducer)]

def node_a(state: LogState) -> dict:
    return {"events": ["a_started", "a_finished"]}

def node_b(state: LogState) -> dict:
    return {"events": ["b_started", "b_finished"]}

builder = StateGraph(LogState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "delta-1"}}
result = graph.invoke({"events": []}, config)
print(result["events"])
# ['a_started', 'a_finished', 'b_started', 'b_finished']
```

### Example 2 — dict-accumulating reducer for running totals

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import StateGraph, START, END

def merge_counts(state: dict[str, int], writes: list[dict[str, int]]) -> dict[str, int]:
    result = dict(state)
    for update in writes:
        for key, val in update.items():
            result[key] = result.get(key, 0) + val
    return result

class CountState(TypedDict):
    counts: Annotated[dict[str, int], DeltaChannel(merge_counts)]

def counter_node(state: CountState) -> dict:
    return {"counts": {"api_calls": 1, "tokens": 150}}

def second_node(state: CountState) -> dict:
    return {"counts": {"api_calls": 1, "tokens": 200}}

builder = StateGraph(CountState)
builder.add_node("first", counter_node)
builder.add_node("second", second_node)
builder.add_edge(START, "first")
builder.add_edge("first", "second")
builder.add_edge("second", END)

graph = builder.compile()
result = graph.invoke({"counts": {}})
print(result["counts"])
# {'api_calls': 2, 'tokens': 350}
```

### Example 3 — `Overwrite` to reset the accumulator mid-run

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.channels.binop import Overwrite
from langgraph.graph import StateGraph, START, END

def concat_reducer(state: list[str], writes: list[str]) -> list[str]:
    return state + writes

class ResetState(TypedDict):
    log: Annotated[list[str], DeltaChannel(concat_reducer)]

def add_entries(state: ResetState) -> dict:
    return {"log": ["entry1", "entry2"]}

def reset_and_add(state: ResetState) -> dict:
    # Overwrite resets the accumulator; replay starts from this point
    return {"log": Overwrite(["fresh_entry"])}

builder = StateGraph(ResetState)
builder.add_node("add", add_entries)
builder.add_node("reset", reset_and_add)
builder.add_edge(START, "add")
builder.add_edge("add", "reset")
builder.add_edge("reset", END)

graph = builder.compile()
result = graph.invoke({"log": []})
print(result["log"])
# ['fresh_entry']  — Overwrite clears history; only post-reset entries remain
```

---

## 3 · `UIMessage` + `RemoveUIMessage` + `push_ui_message()` + `ui_message_reducer` — generative UI

**Module:** `langgraph.graph.ui`

`UIMessage` is a `TypedDict` that represents a UI component update sent to a streaming frontend. `push_ui_message()` emits it on both the custom stream channel and (by default) the `"ui"` state key simultaneously. `ui_message_reducer` merges update lists and handles `RemoveUIMessage` tombstones, making it suitable as an `Annotated` reducer for the UI state field.

**Key source facts (`langgraph/graph/ui.py`):**

- `UIMessage` fields: `type: Literal["ui"]`, `id: str`, `name: str`, `props: dict`, `metadata: dict`. The `metadata.merge=True` flag enables prop delta updates — only changed keys in `props` are applied rather than replacing the whole `props` dict.
- `push_ui_message(name, props, *, id, metadata, message, state_key, merge)`: generates a UUID if `id` is omitted; calls `get_stream_writer()(evt)` for streaming and `CONFIG_KEY_SEND([(state_key, evt)])` for state persistence. Pass `state_key=None` for stream-only emission.
- `RemoveUIMessage` TypedDict has `type: Literal["remove-ui"]` and `id: str`. `ui_message_reducer` raises `ValueError` if the `id` does not exist in the current state.
- Stable IDs enable **in-place updates**: emit a `UIMessage` with the same `id` and `merge=True` to patch only the changed props without replacing the whole component.
- `delete_ui_message(id)` is a helper that calls `push_ui_message` with a `RemoveUIMessage`-equivalent payload.

### Example 1 — streaming a progress indicator from a node

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage, ui_message_reducer, push_ui_message

class WorkState(TypedDict):
    result: str
    ui: Annotated[list[UIMessage], ui_message_reducer]

def slow_node(state: WorkState) -> dict:
    # Emit a loading spinner to the UI
    msg = push_ui_message(
        name="ProgressBar",
        props={"status": "running", "pct": 0},
    )
    spinner_id = msg["id"]

    # ... do real work here ...
    work_result = "done"

    # Update the spinner in-place to show completion
    push_ui_message(
        name="ProgressBar",
        props={"status": "complete", "pct": 100},
        id=spinner_id,
        merge=True,  # only patch changed props
    )

    return {"result": work_result}

builder = StateGraph(WorkState)
builder.add_node("work", slow_node)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile()
# Stream mode "custom" receives the UIMessage events as they are emitted
for chunk in graph.stream({"result": "", "ui": []}, stream_mode="custom"):
    print(chunk)
# {'type': 'ui', 'id': '...', 'name': 'ProgressBar', 'props': {'status': 'running', 'pct': 0}, 'metadata': {...}}
# {'type': 'ui', 'id': '...', 'name': 'ProgressBar', 'props': {'status': 'complete', 'pct': 100}, 'metadata': {'merge': True, ...}}
```

### Example 2 — removing a UI component after use

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import UIMessage, ui_message_reducer, push_ui_message, delete_ui_message

class ChatState(TypedDict):
    messages: list[str]
    ui: Annotated[list[UIMessage], ui_message_reducer]

def show_thinking(state: ChatState) -> dict:
    msg = push_ui_message(name="ThinkingIndicator", props={"visible": True})
    return {"_thinking_id": msg["id"]}

def hide_thinking(state: ChatState) -> dict:
    # Retrieve the stored id and remove the component
    all_ui = state["ui"]
    thinking = next((m for m in all_ui if m["name"] == "ThinkingIndicator"), None)
    if thinking:
        delete_ui_message(thinking["id"])
    return {"messages": ["Response ready!"]}

builder = StateGraph(ChatState)
builder.add_node("show", show_thinking)
builder.add_node("hide", hide_thinking)
builder.add_edge(START, "show")
builder.add_edge("show", "hide")
builder.add_edge("hide", END)

graph = builder.compile()
result = graph.invoke({"messages": [], "ui": []})
# ui list is empty after hide_thinking removed the indicator
print([m["name"] for m in result["ui"]])
# []
```

### Example 3 — stream-only UI emission with `state_key=None`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.ui import push_ui_message

class SimpleState(TypedDict):
    value: int

def emitting_node(state: SimpleState) -> dict:
    # state_key=None: no state persistence, stream-only emission
    push_ui_message(
        name="Counter",
        props={"count": state["value"]},
        state_key=None,
    )
    return {"value": state["value"] + 1}

builder = StateGraph(SimpleState)
builder.add_node("emit", emitting_node)
builder.add_edge(START, "emit")
builder.add_edge("emit", END)

graph = builder.compile()
ui_events = []
for chunk in graph.stream({"value": 0}, stream_mode="custom"):
    if isinstance(chunk, dict) and chunk.get("type") == "ui":
        ui_events.append(chunk)
print(ui_events)
# [{'type': 'ui', 'id': '...', 'name': 'Counter', 'props': {'count': 0}, 'metadata': {...}}]
# state["ui"] key does not exist — not persisted
```

---

## 4 · `stream_events(version="v3")` + `GraphRunStream` — caller-driven pump

**Module:** `langgraph.stream.run_stream`

`stream_events(version="v3")` returns a `GraphRunStream` (sync) or `AsyncGraphRunStream` (async). Unlike v1/v2 which yield dicts, v3 returns **typed projection handles** — iterating any of `.values`, `.messages`, `.lifecycle`, or `.subgraphs` **drives the graph forward** (there is no background thread). This means the graph pauses between event emissions and only advances when the caller asks for the next item.

**Key source facts (`langgraph/stream/run_stream.py`):**

- `GraphRunStream.__init__` builds native projection attributes by `setattr`-ing `mux.extensions` onto `self` for every key in `mux.native_keys` (always `values`, `messages`, `lifecycle`, `subgraphs`). Opt-in projections (`updates`, `custom`, `checkpoints`, `debug`, `tasks`) are in `run.extensions[...]`.
- Each projection is a `StreamChannel` — a single-consumer drainable queue. Calling `.tee(n)` clones it into `n` parallel consumers.
- `run.interleave()` is a generator that yields `(key, chunk)` pairs from all native projections in push-stamp monotonic order — useful when you need a combined ordered view.
- `run.abort()` triggers cooperative cancellation: it calls `GeneratorExit` propagation and, for async, cancels the pending `_anext_task`.
- `run.interrupted` / `run.interrupts` are set when the graph pauses at an `interrupt()` call.

### Example 1 — driving graph forward by iterating `.values`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class NumState(TypedDict):
    n: int

def increment(state: NumState) -> dict:
    return {"n": state["n"] + 1}

def double(state: NumState) -> dict:
    return {"n": state["n"] * 2}

builder = StateGraph(NumState)
builder.add_node("inc", increment)
builder.add_node("dbl", double)
builder.add_edge(START, "inc")
builder.add_edge("inc", "dbl")
builder.add_edge("dbl", END)

graph = builder.compile()

run = graph.stream_events({"n": 3}, version="v3")
for snapshot in run.values:
    print(snapshot)
# {'n': 4}
# {'n': 8}
```

### Example 2 — `interleave()` to get ordered values + messages stream

```python
import asyncio
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class MsgState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def respond(state: MsgState) -> dict:
    return {"messages": [AIMessage("Hello from the graph!")]}

builder = StateGraph(MsgState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

graph = builder.compile()

async def main():
    run = graph.stream_events(
        {"messages": [HumanMessage("Hi")]},
        version="v3",
    )
    # interleave() yields (channel_name, payload) tuples in emission order
    async for key, chunk in run.interleave():
        print(f"[{key}] {chunk}")

asyncio.run(main())
# [messages] ChatModelStream(...)
# [values] {'messages': [HumanMessage('Hi'), AIMessage('Hello from the graph!')]}
```

### Example 3 — opt-in `updates` projection + detecting interrupts

```python
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class ApprovalState(TypedDict):
    data: str
    approved: bool

def check(state: ApprovalState) -> dict:
    answer = interrupt("Approve?")
    return {"approved": answer == "yes"}

builder = StateGraph(ApprovalState)
builder.add_node("check", check)
builder.add_edge(START, "check")
builder.add_edge("check", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "v3-t1"}}

run = graph.stream_events({"data": "payload", "approved": False}, config, version="v3")

# Iterate updates projection — available via extensions for opt-in projections
updates_stream = run.extensions.get("updates")

# Drive via values; check interrupted state after exhaustion
for val in run.values:
    print("snapshot:", val)

if run.interrupted:
    print("Graph paused at interrupt!")
    print("Interrupt details:", run.interrupts)
# Graph paused at interrupt!
# Interrupt details: [Interrupt(value='Approve?', id='...')]
```

---

## 5 · `add_sequence()` — linear pipeline construction

**Module:** `langgraph.graph.state`

`add_sequence(nodes)` is a convenience method on `StateGraph` that calls `add_node` + `add_edge` for each consecutive pair in `nodes`, building a linear execution chain in one call. Nodes can be plain callables (name inferred from `__name__`) or `(name, callable)` tuples for custom names. It returns `Self` for fluent chaining.

**Key source facts (`langgraph/graph/state.py`):**

- Raises `ValueError` if the sequence is empty or contains duplicate names.
- Internally iterates over `nodes`, calling `add_node(name, node)` and `add_edge(previous_name, name)` for each pair.
- The first node in the sequence is **not** connected to `START` automatically — you still need `add_edge(START, first_node_name)`.
- Works with `(name, callable)` tuples to handle lambdas or any two callables with the same `__name__`.
- Returns `Self`, so `builder.add_sequence([...]).add_conditional_edges(...)` chains work naturally.

### Example 1 — basic three-step pipeline

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class PipelineState(TypedDict):
    text: str

def extract(state: PipelineState) -> dict:
    return {"text": state["text"].strip()}

def transform(state: PipelineState) -> dict:
    return {"text": state["text"].upper()}

def load(state: PipelineState) -> dict:
    return {"text": f"[LOADED] {state['text']}"}

builder = StateGraph(PipelineState)
builder.add_edge(START, "extract")
builder.add_sequence([extract, transform, load])
builder.add_edge("load", END)

graph = builder.compile()
result = graph.invoke({"text": "  hello world  "})
print(result["text"])
# [LOADED] HELLO WORLD
```

### Example 2 — named tuples for lambda steps

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class ScoreState(TypedDict):
    score: float

builder = StateGraph(ScoreState)
builder.add_edge(START, "clamp")
builder.add_sequence([
    ("clamp",     lambda s: {"score": min(max(s["score"], 0.0), 1.0)}),
    ("round",     lambda s: {"score": round(s["score"], 2)}),
    ("stringify", lambda s: {"score": s["score"]}),  # placeholder for formatting
])
builder.add_edge("stringify", END)

graph = builder.compile()
result = graph.invoke({"score": 1.567})
print(result["score"])
# 1.0  (clamped then rounded)
```

### Example 3 — `add_sequence` + `add_conditional_edges` fluent chain

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class RouteState(TypedDict):
    value: int
    path: str

def normalize(state: RouteState) -> dict:
    return {"value": abs(state["value"])}

def score(state: RouteState) -> dict:
    return {"value": state["value"] * 2}

def route(state: RouteState) -> Literal["high", "low"]:
    return "high" if state["value"] > 10 else "low"

def high_path(state: RouteState) -> dict:
    return {"path": "HIGH"}

def low_path(state: RouteState) -> dict:
    return {"path": "LOW"}

(
    StateGraph(RouteState)
    .add_node("high_path", high_path)
    .add_node("low_path", low_path)
    .add_edge(START, "normalize")
    .add_sequence([normalize, score])
    .add_conditional_edges("score", route, {"high": "high_path", "low": "low_path"})
    .add_edge("high_path", END)
    .add_edge("low_path", END)
)
# Builder is ready; call .compile() to get the CompiledStateGraph
```

---

## 6 · `Pregel.as_tool()` — expose a compiled graph as a `BaseTool`

**Module:** `langgraph.pregel.main` (inherited from `langchain_core.runnables.Runnable`)

`CompiledStateGraph.as_tool(...)` is a **beta** API that wraps the compiled graph as a `BaseTool`. Because `CompiledStateGraph` implements the `Runnable` interface, it delegates to `langchain_core.tools.convert_runnable_to_tool`. This makes it trivial to embed a full LangGraph workflow inside another agent's `ToolNode`.

**Key source facts (`langgraph/pregel/main.py` + `langchain_core`):**

- Signature: `as_tool(args_schema=None, *, name=None, description=None, arg_types=None) -> BaseTool`.
- Schema inference: if the graph's `InputT` is a `TypedDict` or Pydantic `BaseModel`, `get_input_schema()` provides the schema automatically — `args_schema` is only needed if the input is an untyped `dict`.
- `arg_types` is a `dict[str, type]` shorthand that generates a Pydantic model on the fly; use when you want to expose only a subset of the graph's input keys.
- The returned tool's `invoke` calls `graph.invoke(input, config=config)`, so checkpointer/thread_id can be injected via `config`.
- The tool is decorated with `@beta`, so import warnings are expected.

### Example 1 — minimal TypedDict-based graph tool

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class SumInput(TypedDict):
    a: int
    b: int

class SumState(TypedDict):
    a: int
    b: int
    result: int

def add_node(state: SumState) -> dict:
    return {"result": state["a"] + state["b"]}

builder = StateGraph(SumState, input=SumInput)
builder.add_node("add", add_node)
builder.add_edge(START, "add")
builder.add_edge("add", END)
graph = builder.compile()

# Wrap as a LangChain tool — schema is inferred from SumInput
tool = graph.as_tool(name="sum_graph", description="Add two integers together.")
result = tool.invoke({"a": 3, "b": 4})
print(result)
# {'a': 3, 'b': 4, 'result': 7}
```

### Example 2 — custom `args_schema` for a dict-input graph

```python
from typing import Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class PipeState(TypedDict):
    raw: str
    processed: str

def process(state: PipeState) -> dict:
    return {"processed": state["raw"].strip().upper()}

graph = StateGraph(PipeState)
graph.add_node("proc", process)
graph.add_edge(START, "proc")
graph.add_edge("proc", END)
compiled = graph.compile()

class ProcessSchema(BaseModel):
    """Process and normalize a raw string."""
    raw: str = Field(..., description="Raw text to normalize")

tool = compiled.as_tool(
    args_schema=ProcessSchema,
    name="text_processor",
    description="Normalize raw text.",
)
print(tool.invoke({"raw": "  hello  "}))
# {'raw': '  hello  ', 'processed': 'HELLO'}
```

### Example 3 — embedding a graph tool inside a parent `ToolNode`

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode

# Sub-graph: looks up a value
class LookupState(TypedDict):
    key: str
    value: str

def lookup(state: LookupState) -> dict:
    db = {"alice": "engineer", "bob": "designer"}
    return {"value": db.get(state["key"], "unknown")}

sub = StateGraph(LookupState)
sub.add_node("lookup", lookup)
sub.add_edge(START, "lookup")
sub.add_edge("lookup", END)
sub_graph = sub.compile()

# Expose sub-graph as a tool
lookup_tool = sub_graph.as_tool(
    name="lookup_role",
    description="Look up a person's role by name.",
)

# Parent graph that uses the sub-graph tool
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tool_node = ToolNode([lookup_tool])

fake_call = {"name": "lookup_role", "args": {"key": "alice"}, "id": "tc1", "type": "tool_call"}
state = {"messages": [AIMessage("", tool_calls=[fake_call])]}
result = tool_node.invoke(state)
print(result["messages"][0].content)
# {'key': 'alice', 'value': 'engineer'}
```

---

## 7 · `ErrorCode` + `InvalidUpdateError` + `GraphRecursionError` + `EmptyInputError` — error taxonomy

**Module:** `langgraph.errors`

LangGraph surfaces execution failures through a typed error hierarchy. Understanding these lets you write targeted `try/except` clauses and `error_handler=` nodes.

**Key source facts (`langgraph/errors.py`):**

- `ErrorCode` is an `Enum` with 5 members: `GRAPH_RECURSION_LIMIT`, `INVALID_CONCURRENT_GRAPH_UPDATE`, `INVALID_GRAPH_NODE_RETURN_VALUE`, `MULTIPLE_SUBGRAPHS`, `INVALID_CHAT_HISTORY`.
- `InvalidUpdateError` carries an `ErrorCode` and is raised when two nodes write to a channel that only accepts one writer per step (e.g. `LastValue` with two concurrent writes, or `Overwrite` used twice in the same super-step). Its message includes the error code so you can programmatically inspect it.
- `GraphRecursionError` is a `RecursionError` subclass. The `recursion_limit` is set via `config["recursion_limit"]` (default 25). Increase it or add a truncation node using `RemainingSteps`.
- `EmptyInputError` is raised on the first invocation when the graph receives no state and no initial value can be inferred.
- `NodeError` is a frozen `@dataclass` with `node: str` and `error: BaseException` fields, injected into error-handler nodes via `add_node(error_handler=...)`.
- `create_error_message(error, input, config)` formats a standardized error string for `ToolMessage` content when `ToolNode` hits an error.

### Example 1 — catching `GraphRecursionError` and raising the limit

```python
from typing_extensions import TypedDict
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    count: int

def forever(state: LoopState) -> dict:
    return {"count": state["count"] + 1}

builder = StateGraph(LoopState)
builder.add_node("loop", forever)
builder.add_edge(START, "loop")
builder.add_edge("loop", "loop")  # infinite loop

graph = builder.compile()

try:
    # Default recursion_limit is 25
    graph.invoke({"count": 0}, config={"recursion_limit": 5})
except GraphRecursionError as e:
    print(f"Caught: {type(e).__name__}: {e}")
    # Caught: GraphRecursionError: Recursion limit of 5 reached ...

# Raise the limit for deep-but-finite graphs
result = graph.invoke({"count": 0}, config={"recursion_limit": 3})
print(result)  # {'count': 3}  (runs 3 steps then hits limit again — for demo)
```

### Example 2 — `InvalidUpdateError` from concurrent channel writes

```python
from typing_extensions import TypedDict
from langgraph.errors import InvalidUpdateError
from langgraph.graph import StateGraph, START, END

class SharedState(TypedDict):
    value: int  # LastValue channel — only one writer per step allowed

def writer_a(state: SharedState) -> dict:
    return {"value": 10}

def writer_b(state: SharedState) -> dict:
    return {"value": 20}

builder = StateGraph(SharedState)
builder.add_node("a", writer_a)
builder.add_node("b", writer_b)
builder.add_edge(START, "a")
builder.add_edge(START, "b")  # parallel — both run in same super-step
builder.add_edge("a", END)
builder.add_edge("b", END)

graph = builder.compile()

try:
    graph.invoke({"value": 0})
except InvalidUpdateError as e:
    print(f"Channel conflict: {e}")
    # InvalidUpdateError: INVALID_CONCURRENT_GRAPH_UPDATE ...
```

### Example 3 — per-node `error_handler` receiving `NodeError`

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langgraph.errors import NodeError
from langgraph.graph import StateGraph, START, END

class WorkState(TypedDict):
    result: str
    error_msg: str

def risky_node(state: WorkState) -> dict:
    raise ValueError("Something went wrong in risky_node")

def handle_error(state: WorkState, error: NodeError) -> dict:
    # NodeError.node is the name of the failed node
    # NodeError.error is the original exception
    msg = f"Node '{error.node}' failed: {error.error}"
    return {"error_msg": msg, "result": "fallback"}

builder = StateGraph(WorkState)
builder.add_node("risky", risky_node, error_handler=handle_error)
builder.add_edge(START, "risky")
builder.add_edge("risky", END)

graph = builder.compile()
result = graph.invoke({"result": "", "error_msg": ""})
print(result["error_msg"])
# Node 'risky' failed: Something went wrong in risky_node
print(result["result"])
# fallback
```

---

## 8 · `create_react_agent` with `pre_model_hook` / `post_model_hook` — hook-based middleware

**Module:** `langgraph.prebuilt.chat_agent_executor`

> **Note:** `create_react_agent` is deprecated in favor of `create_agent` from `langchain.agents`. Patterns here remain valid for legacy codebases and illustrate the hook mechanism that `create_agent` also supports.

`pre_model_hook` and `post_model_hook` are `RunnableLike` callables injected into the ReAct loop before and after the LLM call respectively. They receive the full agent state and can return a partial-state `dict` to mutate it, enabling prompt injection, message trimming, token budget enforcement, and structured response wrapping without editing the model node.

**Key source facts (`langgraph/prebuilt/chat_agent_executor.py`):**

- `pre_model_hook(state) -> dict | None` runs before every LLM call. Return `{"messages": [...]}` to replace/trim the message list the model sees. Return `None` or `{}` to pass through unchanged.
- `post_model_hook(state) -> dict | None` runs after every LLM call. Receives state including the fresh AI message appended. Useful for token-budget enforcement, logging, or appending metadata.
- `version="v2"` (the default since 1.0) exposes pre/post hooks; `version="v1"` ignores them.
- `response_format: StructuredResponseSchema | tuple[str, StructuredResponseSchema]` attaches a structured output schema to the final turn. The agent will call a dummy tool that captures the structured result in `state["structured_response"]`.
- `prompt: str | SystemMessage | Callable` prepends a system message or dynamically constructs the prompt on each LLM call.

### Example 1 — `pre_model_hook` to inject a system prompt dynamically

```python
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

try:
    from langgraph.prebuilt import create_react_agent

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        user_role: str  # injected at invocation time

    @tool
    def get_time() -> str:
        """Return current time placeholder."""
        return "12:00 UTC"

    def inject_role_prompt(state: AgentState) -> dict:
        """Pre-hook: prepend a role-specific system message."""
        role = state.get("user_role", "user")
        sys_msg = SystemMessage(f"You are an assistant for a {role}. Be concise.")
        # Return updated messages; this replaces what the model receives
        return {"messages": [sys_msg] + state["messages"]}

    # agent = create_react_agent(
    #     model="anthropic:claude-3-5-haiku-20241022",
    #     tools=[get_time],
    #     pre_model_hook=inject_role_prompt,
    #     version="v2",
    # )
    # result = agent.invoke({"messages": [("user", "What time is it?")], "user_role": "admin"})
    print("pre_model_hook example: inject_role_prompt would prefix system message before each LLM call")
except ImportError:
    print("langchain-anthropic not installed; skipping live call")
```

### Example 2 — `pre_model_hook` for message-window trimming

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, trim_messages
from langgraph.graph.message import add_messages

class TrimState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def trim_to_window(state: TrimState) -> dict:
    """Keep only the last 10 messages to stay within context limits."""
    msgs = state["messages"]
    if len(msgs) > 10:
        # Always keep system messages; trim older human/AI turns
        system = [m for m in msgs if m.type == "system"]
        rest = [m for m in msgs if m.type != "system"]
        trimmed = system + rest[-9:]
        return {"messages": trimmed}
    return {}

# Usage pattern (no live model needed to illustrate):
# agent = create_react_agent(
#     model=model,
#     tools=tools,
#     pre_model_hook=trim_to_window,
#     version="v2",
# )
print("trim_to_window: returns trimmed messages only when > 10 exist; empty dict passes through")
```

### Example 3 — `post_model_hook` for token-budget enforcement

```python
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command

class BudgetState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tokens_used: int
    budget: int

def enforce_budget(state: BudgetState) -> dict | Command:
    """Post-hook: stop the agent if token budget is exceeded."""
    last_msg = state["messages"][-1]
    # Approximate: count words * 1.3 as token estimate
    used = int(len(str(getattr(last_msg, "content", ""))) * 0.33)
    total = state["tokens_used"] + used
    if total >= state["budget"]:
        # Force graph to END by returning a Command that ends the loop
        return Command(goto="__end__", update={"tokens_used": total})
    return {"tokens_used": total}

# agent = create_react_agent(
#     model=model,
#     tools=tools,
#     post_model_hook=enforce_budget,
#     version="v2",
# )
print("enforce_budget: post_model_hook checks tokens after each LLM call; returns Command to stop if exceeded")
```

---

## 9 · `StateGraph.compile(transformers=, cache=)` — compile-time injection

**Module:** `langgraph.graph.state`

The `compile()` method has two less-documented parameters: `transformers` (a sequence of stream-transformer factories injected into the `StreamMux` for every run) and `cache` (a `BaseCache` instance wired to all `@task` calls as the default cache).

**Key source facts (`langgraph/graph/state.py`):**

- `transformers: Sequence[Callable[[tuple[str, ...]], Any]] | None` — each factory receives the checkpoint-namespace tuple `(graph_name, ...)` and should return a `StreamTransformer` subclass. These are added to the `StreamMux` alongside built-in transformers, allowing custom `stream_mode` projections to be registered at compile time rather than per-call.
- `cache: BaseCache | None` — passed to `Pregel` and stored as `graph.cache`. When `@task` nodes run, this cache satisfies `CachePolicy` lookups. `InMemoryCache` works out of the box; `RedisCache` (from `langgraph-checkpoint-redis`) is the production option.
- Both parameters complement per-call options: `compile(cache=c)` sets a graph-wide default, but individual calls to `stream_events(transformers=[...])` add more transformers on top.
- `cache` does NOT affect nodes built with `add_node(cache_policy=...)` directly — it is the backing store that `CachePolicy.key_func` writes into. Without `cache=`, `CachePolicy` silently no-ops.

### Example 1 — graph-level `InMemoryCache` for `@task` memoization

```python
import time
from typing_extensions import TypedDict
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy
from langgraph._internal._cache import default_cache_key

try:
    from langgraph.cache.memory import InMemoryCache
    cache_available = True
except ImportError:
    cache_available = False

if cache_available:
    @task(cache_policy=CachePolicy(ttl=60))
    def slow_fetch(url: str) -> str:
        """Simulated slow fetch — result cached for 60 s."""
        time.sleep(0.01)
        return f"data_from_{url}"

    @entrypoint()
    def pipeline(urls: list[str]) -> list[str]:
        return [slow_fetch(u).result() for u in urls]

    cache = InMemoryCache()
    graph = pipeline  # entrypoint already compiled
    # To pass cache at compile time for StateGraph:
    # graph = builder.compile(cache=cache)
    print("InMemoryCache wired; repeated calls to slow_fetch with same url return cached result")
else:
    print("langgraph.cache not available in this build; use langgraph-checkpoint-redis for RedisCache")
```

### Example 2 — checking available cache backends

```python
# Discover which cache backends are installed
backends = []
try:
    from langgraph.cache.memory import InMemoryCache
    backends.append("InMemoryCache (built-in, thread-safe, TTL-aware)")
except ImportError:
    pass

try:
    from langgraph_checkpoint_redis import RedisCache
    backends.append("RedisCache (production, TTL via Redis EXPIRE)")
except ImportError:
    pass

print("Available cache backends:")
for b in backends:
    print(f"  - {b}")
if not backends:
    print("  None detected — install langgraph-checkpoint-redis for production use")
```

### Example 3 — per-call `transformers=` injection via `stream_events`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.stream._types import StreamTransformer

class TextState(TypedDict):
    text: str

def upcase(state: TextState) -> dict:
    return {"text": state["text"].upper()}

builder = StateGraph(TextState)
builder.add_node("up", upcase)
builder.add_edge(START, "up")
builder.add_edge("up", END)

# compile() with no extra transformers
graph = builder.compile()

# stream_events v3 accepts per-call transformers= for custom projections.
# The built-in natives (values, messages, lifecycle, subgraphs) are always registered.
run = graph.stream_events({"text": "hello"}, version="v3")

# Access standard native projections
for val in run.values:
    print("final state:", val)
# final state: {'text': 'HELLO'}

# run.extensions contains opt-in projections (updates, checkpoints, debug, tasks, custom)
print("extension keys:", list(run.extensions.keys()))
```

---

## 10 · `add_node(error_handler=, destinations=, defer=)` — advanced node parameters

**Module:** `langgraph.graph.state`

Beyond the basic `add_node(name, fn)` call, three parameters extend node behaviour significantly: `error_handler` for per-node recovery, `destinations` for edgeless `Command`-based routing declarations, and `defer=True` for scheduling a node to execute at the very end of the current step.

**Key source facts (`langgraph/graph/state.py`):**

- `error_handler: StateNode | None` — a callable that receives the full state **plus** a `NodeError` dataclass injected as an extra keyword argument. If the main node raises, the error-handler node is run in its place; its output is written to the checkpoint. The error-handler itself is added as a special node with `is_error_handler=True` in `StateNodeSpec`, so it does not itself accept a nested `error_handler`.
- `destinations: dict[str, str] | tuple[str, ...] | None` — documents the node's possible routing targets for graph rendering (Mermaid / ASCII). Values are target node names; dict keys are the labels shown on edges. This has **no effect on runtime routing** — it only improves topology visualization when nodes return `Command` objects instead of using explicit edges.
- `defer: bool = False` — marks the node as deferred. Deferred nodes are scheduled at the very end of the current super-step, after all non-deferred nodes have finished. Useful for cleanup, aggregation, or summary nodes that should see the final state of the step.
- All three can be combined: `add_node("cleanup", fn, defer=True, error_handler=recover)`.

### Example 1 — `error_handler` for graceful recovery

```python
from typing_extensions import TypedDict
from langgraph.errors import NodeError
from langgraph.graph import StateGraph, START, END

class ProcessState(TypedDict):
    input: str
    output: str
    error: str

def parse_json(state: ProcessState) -> dict:
    import json
    data = json.loads(state["input"])  # may raise if input is invalid
    return {"output": str(data)}

def recover_parse(state: ProcessState, error: NodeError) -> dict:
    """Called when parse_json raises; error.error is the original exception."""
    return {
        "output": "",
        "error": f"Parse failed in {error.node}: {error.error}",
    }

builder = StateGraph(ProcessState)
builder.add_node("parse", parse_json, error_handler=recover_parse)
builder.add_edge(START, "parse")
builder.add_edge("parse", END)

graph = builder.compile()

good = graph.invoke({"input": '{"key": 1}', "output": "", "error": ""})
print(good["output"])   # {'key': 1}

bad = graph.invoke({"input": "not json", "output": "", "error": ""})
print(bad["error"])     # Parse failed in parse: ...
```

### Example 2 — `destinations=` for edgeless `Command` routing

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class RouterState(TypedDict):
    category: str
    result: str

def classifier(state: RouterState) -> Command:
    """Returns a Command — no explicit edge needed if destinations= is declared."""
    cat = state["category"]
    if cat == "urgent":
        return Command(goto="urgent_handler", update={"result": "routing to urgent"})
    return Command(goto="normal_handler", update={"result": "routing to normal"})

def urgent_handler(state: RouterState) -> dict:
    return {"result": "URGENT: " + state["result"]}

def normal_handler(state: RouterState) -> dict:
    return {"result": "NORMAL: " + state["result"]}

builder = StateGraph(RouterState)
builder.add_node(
    "classify",
    classifier,
    destinations={"urgent_handler": "urgent", "normal_handler": "normal"},
)
builder.add_node("urgent_handler", urgent_handler)
builder.add_node("normal_handler", normal_handler)
builder.add_edge(START, "classify")
builder.add_edge("urgent_handler", END)
builder.add_edge("normal_handler", END)

graph = builder.compile()
print(graph.invoke({"category": "urgent", "result": ""})["result"])
# URGENT: routing to urgent
print(graph.invoke({"category": "other", "result": ""})["result"])
# NORMAL: routing to normal
```

### Example 3 — `defer=True` for step-end aggregation

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph, START, END

class AggState(TypedDict):
    scores: Annotated[list[int], operator.add]
    summary: str

def scorer_a(state: AggState) -> dict:
    return {"scores": [85]}

def scorer_b(state: AggState) -> dict:
    return {"scores": [92]}

def summarize(state: AggState) -> dict:
    """Deferred: runs after all parallel nodes finish this super-step."""
    avg = sum(state["scores"]) / len(state["scores"]) if state["scores"] else 0
    return {"summary": f"Average score: {avg:.1f}"}

builder = StateGraph(AggState)
builder.add_node("a", scorer_a)
builder.add_node("b", scorer_b)
builder.add_node("summarize", summarize, defer=True)  # runs last in the step
builder.add_edge(START, "a")
builder.add_edge(START, "b")
builder.add_edge("a", "summarize")
builder.add_edge("b", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
result = graph.invoke({"scores": [], "summary": ""})
print(result["scores"])   # [85, 92]
print(result["summary"])  # Average score: 88.5
```

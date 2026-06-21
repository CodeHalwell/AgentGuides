---
title: "Class deep-dives Vol. 20 — execution engine internals (1.2.6)"
description: "Source-verified deep dives into 10 previously undocumented internals of LangGraph 1.2.6: StreamToolCallHandler+ToolCallWriter (stream_mode='tools' internals), ReplayState (time-travel subgraph coordination), PregelScratchpad (execution context), LangGraphDeprecationWarning hierarchy, create_checkpoint+delta_channels_to_snapshot (checkpoint construction), chain_future+run_coroutine_threadsafe (cross-loop bridge), validate_graph+validate_keys (graph validation rules), read_channel+map_input+map_command (I/O layer), should_interrupt+apply_writes+prepare_next_tasks (core Pregel algorithm), and LazyAtomicCounter+task_path_str (task identity utilities). All signatures and behaviours verified from installed package source."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 20"
  order: 51
---

# Class deep-dives Vol. 20 — execution engine internals (1.2.6)

Verified against **`langgraph==1.2.6`** / **`langgraph-checkpoint==4.1.1`** / **`langgraph-prebuilt==1.1.0`**.

Every section was written by inspecting the installed package source directly at `/usr/local/lib/python3.11/dist-packages/langgraph/`. All signatures, field names, constants, and behaviours are drawn from the actual implementation, not from documentation.

---

## Classes covered

| # | Class / symbol | Module |
|---|---------------|--------|
| 1 | `StreamToolCallHandler` + `ToolCallWriter` + `_tool_call_writer` | `langgraph.pregel._tools` |
| 2 | `ReplayState` | `langgraph._internal._replay` |
| 3 | `PregelScratchpad` | `langgraph._internal._scratchpad` |
| 4 | `LangGraphDeprecationWarning` + subclasses | `langgraph.warnings` |
| 5 | `create_checkpoint` + `delta_channels_to_snapshot` + `empty_checkpoint` | `langgraph.pregel._checkpoint` |
| 6 | `chain_future` + `run_coroutine_threadsafe` + `_ensure_future` | `langgraph._internal._future` |
| 7 | `validate_graph` + `validate_keys` | `langgraph.pregel._validate` |
| 8 | `read_channel` + `read_channels` + `map_input` + `map_command` | `langgraph.pregel._io` |
| 9 | `should_interrupt` + `apply_writes` + `prepare_next_tasks` | `langgraph.pregel._algo` |
| 10 | `LazyAtomicCounter` + `task_path_str` + `_uuid5_str` / `_xxhash_str` | `langgraph.pregel._algo` |

---

## 1 · `StreamToolCallHandler` + `ToolCallWriter`

**Module:** `langgraph.pregel._tools`

`StreamToolCallHandler` is the callback handler that powers `stream_mode="tools"`. When `"tools"` is in `stream_modes`, Pregel attaches one instance to the LangChain callback chain. It fires on `on_tool_start` / `on_tool_end` / `on_tool_error` and emits structured protocol events on the stream.

### Key behaviours

| Attribute | Value | Effect |
|-----------|-------|--------|
| `run_inline` | `True` | Runs in the main thread — events are ordered relative to other stream chunks |
| `_tool_call_writer` | `ContextVar[ToolCallWriter \| None]` | Bound writer closure; read by `ToolRuntime.emit_output_delta()` |
| Namespace derivation | strips last `NS_SEP`-segment from `langgraph_checkpoint_ns` | emits at the subgraph's own ns, not the node's |
| `TAG_NOSTREAM` in tags | `None` returned by `_ns_for_emit` | suppresses all events for that tool call |
| `subgraphs=False` + nested ns | only emits if `ns == self.parent_ns` | prevents double-emission when parent already streams |

### Event shapes emitted

```python
# tool-started
{"event": "tool-started", "tool_call_id": "...", "tool_name": "...", "input": {...}}

# tool-output-delta (emitted by ToolRuntime.emit_output_delta())
{"event": "tool-output-delta", "tool_call_id": "...", "delta": <any>}

# tool-finished
{"event": "tool-finished", "tool_call_id": "...", "output": <any>}

# tool-error
{"event": "tool-error", "tool_call_id": "...", "message": "..."}
```

### Example 1 — consume `stream_mode="tools"` events

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}, 22°C"

class State(TypedDict):
    messages: Annotated[list, operator.add]

def call_tool(state: State) -> dict:
    # Simulate an LLM calling the tool
    from langchain_core.messages import AIMessage, ToolCall
    tool_call = ToolCall(name="get_weather", args={"city": "Paris"}, id="tc_001")
    return {"messages": [AIMessage(content="", tool_calls=[tool_call])]}

builder = StateGraph(State)
builder.add_node("call", call_tool)
builder.add_node("tools", ToolNode([get_weather]))
builder.add_edge(START, "call")
builder.add_edge("call", "tools")
builder.add_edge("tools", END)
graph = builder.compile()

for ns, mode, payload in graph.stream(
    {"messages": []},
    stream_mode=["tools"],
    subgraphs=True,
):
    print(f"[{mode}] {payload['event']}: {payload.get('tool_name', payload.get('tool_call_id'))}")
    if payload.get("output"):
        print(f"  → {payload['output']}")
```

### Example 2 — streaming partial output with `ToolRuntime.emit_output_delta()`

```python
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
import operator
from typing import Annotated
from typing_extensions import TypedDict

@tool
def stream_analysis(query: str, runtime: ToolRuntime) -> str:
    """Analyse query and stream partial results."""
    steps = ["Parsing query", "Fetching context", "Running model", "Formatting"]
    for step in steps:
        runtime.emit_output_delta({"step": step, "status": "in_progress"})
    return f"Analysis complete for: {query}"

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

agent = create_react_agent("anthropic:claude-sonnet-4-6", [stream_analysis])

# Consume deltas in real-time
for ns, mode, payload in agent.stream(
    {"messages": [{"role": "user", "content": "Analyse AI trends"}]},
    stream_mode=["tools"],
    subgraphs=True,
):
    if payload["event"] == "tool-output-delta":
        print(f"Delta: {payload['delta']}")
    elif payload["event"] == "tool-finished":
        print(f"Done: {payload['output'][:50]}")
```

### Example 3 — suppress tool streaming with `TAG_NOSTREAM`

```python
from langchain_core.tools import tool
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import operator
from typing import Annotated
from typing_extensions import TypedDict

@tool
def visible_tool(x: int) -> int:
    """This tool appears in stream_mode='tools'."""
    return x * 2

@tool
def hidden_tool(x: int) -> int:
    """This tool is silently suppressed from stream_mode='tools'."""
    return x * 3

hidden_tool.tags = [TAG_NOSTREAM]  # @tool() does not accept tags=; set after creation

class State(TypedDict):
    results: Annotated[list, operator.add]

def router(state: State) -> dict:
    from langchain_core.messages import AIMessage, ToolCall
    calls = [
        ToolCall(name="visible_tool", args={"x": 5}, id="tc_v"),
        ToolCall(name="hidden_tool", args={"x": 5}, id="tc_h"),  # suppressed
    ]
    return {"results": [AIMessage(content="", tool_calls=calls)]}

builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("tools", ToolNode([visible_tool, hidden_tool], messages_key="results"))
builder.add_edge(START, "router")
builder.add_edge("router", "tools")
builder.add_edge("tools", END)
graph = builder.compile()

events = [p["event"] for _, _, p in graph.stream({"results": []}, stream_mode=["tools"], subgraphs=True)]
# Only visible_tool events appear; hidden_tool is suppressed
assert "tool-started" in events  # from visible_tool
print("Events seen:", events)
```

---

## 2 · `ReplayState`

**Module:** `langgraph._internal._replay`

`ReplayState` coordinates which subgraph checkpoint to load during a time-travel replay. A single instance is created when `graph.invoke(config={"configurable": {"checkpoint_id": "..."}})` replays a previous run and is shared (by reference) across all derived configs within that execution.

### Source signature

```python
class ReplayState:
    __slots__ = ("checkpoint_id", "_visited_ns")

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        self._visited_ns: set[str] = set()

    def _is_first_visit(self, checkpoint_ns: str) -> bool:
        # strips ":task_id" suffix so loops recognise the same subgraph
        stable_ns = checkpoint_ns.rsplit(":", 1)[0] if ":" in checkpoint_ns else checkpoint_ns
        if stable_ns in self._visited_ns:
            return False
        self._visited_ns.add(stable_ns)
        return True
```

### Key behaviours

| Method | First call for a namespace | Subsequent calls |
|--------|--------------------------|-----------------|
| `get_checkpoint()` | `checkpointer.list(..., before={"configurable": {"checkpoint_id": self.checkpoint_id}}, limit=1)` | `checkpointer.get_tuple(config)` |
| `aget_checkpoint()` | async version of above | async `aget_tuple` |
| Namespace matching | strips `NS_END` (`:`) suffix to get stable subgraph name | same stable name matches loop iterations |

### Example 1 — understand what `ReplayState` does during time-travel

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int

def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}

def should_continue(state: State) -> str:
    return "increment" if state["counter"] < 3 else END

checkpointer = MemorySaver()
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_conditional_edges("increment", should_continue)
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "replay-demo"}}

# Single invoke loops 3 times → 4+ checkpoints (one per superstep)
result = graph.invoke({"counter": 0}, config)
print(f"Final: counter={result['counter']}")  # 3

# List all checkpoints
history = list(graph.get_state_history(config))
print(f"Checkpoints saved: {len(history)}")  # 4 (start + 3 increments)

# Time-travel: replay from the very first checkpoint (counter=0)
first_checkpoint_id = history[-1].config["configurable"]["checkpoint_id"]
replay_config = {**config, "configurable": {**config["configurable"], "checkpoint_id": first_checkpoint_id}}
replayed = graph.invoke(None, replay_config)
# ReplayState ensures subgraphs restore their pre-replay checkpoints first
print(f"Replayed from step 1: counter={replayed['counter']}")  # 3 (re-runs from counter=0)
```

### Example 2 — `ReplayState` with nested subgraphs

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    value: int

def double_node(state: State) -> dict:
    return {"value": state["value"] * 2}

# Inner graph compiled WITHOUT its own checkpointer — the parent's checkpointer
# manages all state, so replay config and ReplayState propagate correctly.
inner_builder = StateGraph(State)
inner_builder.add_node("double", double_node)
inner_builder.add_edge(START, "double")
inner_builder.add_edge("double", END)
inner_graph = inner_builder.compile()

# Embed the compiled inner graph as a node — this is a true subgraph invocation.
# Pregel will propagate the parent's replay config into the inner graph's
# execution context, and ReplayState._is_first_visit() will strip the
# task-id suffix from the subgraph's checkpoint namespace on replay.
outer_builder = StateGraph(State)
outer_builder.add_node("subgraph", inner_graph)
outer_builder.add_edge(START, "subgraph")
outer_builder.add_edge("subgraph", END)
outer_graph = outer_builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "nested-demo"}}
result = outer_graph.invoke({"value": 3}, config)
print(f"Result: {result['value']}")  # 6 (3 × 2)

history = list(outer_graph.get_state_history(config))
print(f"Checkpoints: {len(history)}")
```

### Example 3 — inspecting `ReplayState` via config

```python
from langgraph._internal._replay import ReplayState

# Demonstrate namespace stripping manually
rs = ReplayState(checkpoint_id="cp_abc123")

# First call for "sub_node:task001" — strips to "sub_node"
assert rs._is_first_visit("sub_node:task001") is True
# Second call with different task-id — same stable namespace "sub_node"
assert rs._is_first_visit("sub_node:task002") is False  # already visited
# A completely different subgraph
assert rs._is_first_visit("other_sub:task003") is True
print("Visited namespaces:", rs._visited_ns)  # {'sub_node', 'other_sub'}

# Empty namespace (root graph)
rs2 = ReplayState(checkpoint_id="cp_xyz")
assert rs2._is_first_visit("") is True
assert rs2._is_first_visit("") is False  # second call
print("Root ns visited:", "" in rs2._visited_ns)  # True
```

---

## 3 · `PregelScratchpad`

**Module:** `langgraph._internal._scratchpad`

`PregelScratchpad` is the per-step execution context passed through `CONFIG_KEY_SCRATCHPAD`. It carries all the mutable counters and resume state that nodes need during a single Pregel superstep.

### Source signature

```python
@dataclasses.dataclass(**_DC_KWARGS)
class PregelScratchpad:
    step: int                              # current superstep index (0-based)
    stop: int                              # max step (recursion_limit)
    call_counter: Callable[[], int]        # LazyAtomicCounter for @task calls
    interrupt_counter: Callable[[], int]   # counter for interrupt() IDs
    get_null_resume: Callable[[bool], Any] # pops from resume list
    resume: list[Any]                      # values from Command(resume=...)
    subgraph_counter: Callable[[], int]    # counter for subgraph task IDs
```

### Key behaviours

| Field | Type | Usage |
|-------|------|-------|
| `step` / `stop` | `int` | `IsLastStepManager.get()` returns `step == stop - 1`; `RemainingStepsManager.get()` returns `stop - step` |
| `call_counter` | `Callable[[], int]` | Returns next int each call; used to create deterministic `@task` IDs |
| `interrupt_counter` | `Callable[[], int]` | Incremented per `interrupt()` call; forms part of the resume key |
| `get_null_resume(is_resuming)` | `Callable[[bool], Any]` | Pops the next value from `resume` list; returns `None` when list exhausted |
| `resume` | `list[Any]` | Populated from `Command(resume=...)` at the start of a resumed step |
| `subgraph_counter` | `Callable[[], int]` | Used to generate unique namespaces for nested `@task` subgraph invocations |

### Example 1 — read `IsLastStep` / `RemainingSteps` (which read the scratchpad)

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed.is_last_step import IsLastStep, RemainingSteps

class State(TypedDict):
    messages: list[str]
    is_last: IsLastStep        # managed: scratchpad.step == scratchpad.stop - 1
    remaining: RemainingSteps  # managed: scratchpad.stop - scratchpad.step

def check_limits(state: State) -> dict:
    if state["is_last"]:
        return {"messages": state["messages"] + ["LAST STEP (0 remaining)"]}
    return {"messages": state["messages"] + [f"{state['remaining']} steps remain"]}

def loop_control(state: State) -> str:
    return "check" if len(state["messages"]) < 3 else END

builder = StateGraph(State)
builder.add_node("check", check_limits)
builder.add_edge(START, "check")
builder.add_conditional_edges("check", loop_control)
graph = builder.compile()

result = graph.invoke({"messages": []}, {"recursion_limit": 4})
for msg in result["messages"]:
    print(msg)
```

### Example 2 — observe scratchpad via config in a node

```python
from langgraph.config import get_config
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD, CONF
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    step_info: list[str]

def inspect_scratchpad(state: State) -> dict:
    config = get_config()
    scratchpad = config[CONF].get(CONFIG_KEY_SCRATCHPAD)
    if scratchpad:
        info = (
            f"step={scratchpad.step} "
            f"stop={scratchpad.stop} "
            f"remaining={scratchpad.stop - scratchpad.step}"
        )
    else:
        info = "no scratchpad (outside Pregel context)"
    return {"step_info": state["step_info"] + [info]}

builder = StateGraph(State)
builder.add_node("inspect", inspect_scratchpad)
builder.add_edge(START, "inspect")
builder.add_edge("inspect", END)
graph = builder.compile()

result = graph.invoke({"step_info": []}, {"recursion_limit": 10})
print(result["step_info"][0])  # e.g. "step=0 stop=10 remaining=10"
```

### Example 3 — how `interrupt()` uses `interrupt_counter` under the hood

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class ApprovalState(TypedDict):
    action: str
    approved: bool

def request_approval(state: ApprovalState) -> dict:
    # Each call to interrupt() increments scratchpad.interrupt_counter
    # The counter forms part of the resume key, enabling selective resume
    response = interrupt({"action": state["action"], "reason": "needs human approval"})
    return {"approved": response}

checkpointer = MemorySaver()
builder = StateGraph(ApprovalState)
builder.add_node("approve", request_approval)
builder.add_edge(START, "approve")
builder.add_edge("approve", END)
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-1"}}

# First invocation — pauses at interrupt
try:
    graph.invoke({"action": "deploy_to_prod", "approved": False}, config)
except Exception:
    pass

# The scratchpad.interrupt_counter was 0 when interrupt() was called.
# Resuming provides the response for counter=0.
from langgraph.types import Command
result = graph.invoke(Command(resume=True), config)
print(f"Approved: {result['approved']}")  # True
```

---

## 4 · `LangGraphDeprecationWarning` + subclasses

**Module:** `langgraph.warnings`

LangGraph uses a structured deprecation warning hierarchy that records `since` and `expected_removal` versions explicitly. This lets users filter for specific version ranges and write tests that assert warnings are (or are not) emitted.

### Source signatures

```python
class LangGraphDeprecationWarning(DeprecationWarning):
    message: str
    since: tuple[int, int]
    expected_removal: tuple[int, int]

    def __init__(self, message, *args, since, expected_removal=None):
        # expected_removal defaults to (since[0] + 1, 0) when not given
        self.expected_removal = expected_removal if expected_removal is not None else (since[0] + 1, 0)

class LangGraphDeprecatedSinceV05(LangGraphDeprecationWarning):
    # since=(0, 5), expected_removal=(2, 0) — next major
    ...

class LangGraphDeprecatedSinceV10(LangGraphDeprecationWarning):
    # since=(1, 0), expected_removal=(2, 0) — next major
    ...

class LangGraphDeprecatedSinceV11(LangGraphDeprecationWarning):
    # since=(1, 1), expected_removal=(3, 0) — two majors out
    ...
```

### What triggers these warnings

| API | Warning class | Since | Remove in |
|-----|--------------|-------|-----------|
| `MessageGraph` | `LangGraphDeprecatedSinceV10` | v1.0 | v2.0 |
| `ValidationNode` (standalone) | `LangGraphDeprecatedSinceV10` | v1.0 | v2.0 |
| `langgraph.prebuilt.interrupt.HumanInterrupt` re-export | `LangGraphDeprecatedSinceV10` | v1.0 | v2.0 |
| `GraphOutput` dict/key access (`output["key"]`, `"key" in output`) | `LangGraphDeprecatedSinceV11` | v1.1 | v3.0 |

### Example 1 — catch and inspect deprecation warnings

```python
import warnings
from langgraph.warnings import LangGraphDeprecationWarning, LangGraphDeprecatedSinceV10

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # MessageGraph triggers LangGraphDeprecatedSinceV10
    from langgraph.graph.message import MessageGraph
    _ = MessageGraph()

    lg_warnings = [x for x in w if issubclass(x.category, LangGraphDeprecationWarning)]
    for warning in lg_warnings:
        print(f"Warning: {warning.message.args[0]}")
        # since/expected_removal are instance attributes set in __init__, not class attrs
        print(f"  since=v{warning.message.since}")
```

### Example 2 — filter by version range in tests

```python
import warnings
import pytest
from langgraph.warnings import LangGraphDeprecatedSinceV10, LangGraphDeprecatedSinceV11

def test_no_v10_deprecations_in_my_graph():
    """Assert that my graph code doesn't use any v1.0-deprecated APIs."""
    from langgraph.graph import StateGraph, START, END
    from typing_extensions import TypedDict

    class State(TypedDict):
        value: int

    with warnings.catch_warnings():
        # Turn v1.0 deprecations into errors — test will fail if we use them
        warnings.filterwarnings("error", category=LangGraphDeprecatedSinceV10)

        builder = StateGraph(State)
        builder.add_node("step", lambda s: {"value": s["value"] + 1})
        builder.add_edge(START, "step")
        builder.add_edge("step", END)
        graph = builder.compile()
        result = graph.invoke({"value": 0})
        assert result["value"] == 1  # No deprecated APIs used

def test_message_graph_emits_deprecation():
    """Assert that MessageGraph emits the expected v1.0 deprecation."""
    with pytest.warns(LangGraphDeprecatedSinceV10):
        from langgraph.graph.message import MessageGraph
        MessageGraph()
```

### Example 3 — build a custom deprecation warning

```python
from langgraph.warnings import LangGraphDeprecationWarning
import warnings

# Create a custom warning for your own langgraph-based library
class MyLibDeprecationWarning(LangGraphDeprecationWarning):
    """Deprecation warnings from my-langgraph-lib."""
    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args, since=(0, 3), expected_removal=(1, 0))

def old_create_agent(model_name: str):
    """Deprecated factory — use create_agent() instead."""
    warnings.warn(
        "old_create_agent() is deprecated. Use create_agent() instead.",
        MyLibDeprecationWarning,
        stacklevel=2,
    )
    return {"model": model_name}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    old_create_agent("gpt-4o")
    for warning in w:
        if issubclass(warning.category, LangGraphDeprecationWarning):
            inst = warning.category.__new__(warning.category)
            LangGraphDeprecationWarning.__init__(
                inst, str(warning.message), since=(0, 3), expected_removal=(1, 0)
            )
            print(f"str(warning): {inst}")
            # Output: "...Deprecated in LangGraph V0.3 to be removed in V1.0."
```

---

## 5 · `create_checkpoint` + `delta_channels_to_snapshot` + `empty_checkpoint`

**Module:** `langgraph.pregel._checkpoint`

These three functions form the checkpoint construction layer. They are called by the Pregel loop at the end of each superstep.

### Key constants and types

```python
LATEST_VERSION = 4  # current checkpoint schema version

# Checkpoint v4 structure (from langgraph.checkpoint.base)
Checkpoint = TypedDict("Checkpoint", {
    "v": int,                              # always 4 for new checkpoints
    "id": str,                             # UUID6 (time-ordered)
    "ts": str,                             # ISO-8601 UTC timestamp
    "channel_values": dict[str, Any],      # serialised channel state
    "channel_versions": dict[str, Any],    # monotonic version per channel
    "versions_seen": dict[str, dict[str, Any]],  # last version each node saw
    "updated_channels": list[str] | None, # channels written in this step; None if unknown
})
```

### `empty_checkpoint()` internals

```python
def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,              # v4
        id=str(uuid6(clock_seq=-2)),   # clock_seq=-2 sorts before all real checkpoints
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
        updated_channels=None,         # no channels written yet
    )
```

`uuid6(clock_seq=-2)` guarantees the empty checkpoint ID sorts lexicographically before any real checkpoint ID produced by `uuid6()` with the default `clock_seq`.

### `delta_channels_to_snapshot()` snapshot decision

A `DeltaChannel` only writes a `_DeltaSnapshot` blob when EITHER of two conditions fires:
- `updates >= channel.snapshot_frequency` (default: 1000)
- `supersteps >= DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` (default: 5000)

This bounds the ancestor-write replay depth.

### Example 1 — inspect checkpoint structure after a graph run

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int
    messages: list[str]

checkpointer = MemorySaver()
builder = StateGraph(State)
builder.add_node("step", lambda s: {"counter": s["counter"] + 1, "messages": s["messages"] + ["tick"]})
builder.add_edge(START, "step")
builder.add_edge("step", END)
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "cp-inspect"}}
graph.invoke({"counter": 0, "messages": []}, config)

# Retrieve the latest checkpoint tuple
cp_tuple = checkpointer.get_tuple(config)
cp = cp_tuple.checkpoint
print(f"Checkpoint version: v{cp['v']}")          # v4
print(f"Checkpoint ID: {cp['id']}")
print(f"Channel versions: {cp['channel_versions']}")
print(f"Values keys: {list(cp['channel_values'].keys())}")
```

### Example 2 — `empty_checkpoint()` sorts before real checkpoints

```python
from langgraph.pregel._checkpoint import empty_checkpoint

ec = empty_checkpoint()
print(f"Empty checkpoint v={ec['v']}")  # v=4
print(f"Empty ID: {ec['id']}")
print(f"Values empty: {ec['channel_values'] == {}}")  # True

# uuid6(clock_seq=-2) makes the empty checkpoint ID sort before all real checkpoint
# IDs — it represents "no checkpoint yet / before any run".
# Note: do NOT use it as the before= arg to get_state_history(); since before=
# is an exclusive upper bound, the "oldest" ID would filter out every real
# checkpoint and return an empty history.
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class S(TypedDict):
    n: int

ckptr = MemorySaver()
builder = StateGraph(S)
builder.add_node("n", lambda s: {"n": s["n"] + 1})
builder.add_edge(START, "n")
builder.add_edge("n", END)
g = builder.compile(checkpointer=ckptr)
cfg = {"configurable": {"thread_id": "sort-demo"}}
g.invoke({"n": 0}, cfg)
g.invoke(None, cfg)

history = list(g.get_state_history(cfg))
print(f"Total checkpoints: {len(history)}")  # at least 2
for h in history:
    print(f"  {h.config['configurable']['checkpoint_id'][:8]}... n={h.values.get('n')}")
```

### Example 3 — `DeltaChannel` snapshot frequency

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.channels.delta import DeltaChannel
from langgraph.graph import StateGraph, START, END

def list_reducer(current: list, updates: list) -> list:
    # updates = batch of raw write values for this superstep; each is a scalar
    return current + updates

class State(TypedDict):
    seed: str  # required input — a DeltaChannel-only graph rejects empty initial input
    # DeltaChannel with snapshot every 5 updates (for demo; default is 1000)
    log: Annotated[list, DeltaChannel(list_reducer, snapshot_frequency=5)]

def append_entry(state: State) -> dict:
    # log may be absent on first call (DeltaChannel starts MISSING);
    # use .get() so index counts from 0 on the very first invocation
    current = state.get("log", [])
    return {"log": f"entry-{len(current)}"}

builder = StateGraph(State)
builder.add_node("append", append_entry)
builder.add_edge(START, "append")
builder.add_edge("append", END)
graph = builder.compile()

# The DeltaChannel stores MISSING in checkpoint blobs (not the actual value)
# and reconstructs by replaying ancestor writes — until snapshot_frequency
# is reached, at which point a _DeltaSnapshot(value) blob is saved.
result = graph.invoke({"seed": "start"})
print(f"Log length: {len(result['log'])}")  # 1

# After 5 updates, delta_channels_to_snapshot() includes this channel
# and create_checkpoint() writes _DeltaSnapshot(value) into channel_values
```

---

## 6 · `chain_future` + `run_coroutine_threadsafe` + `_ensure_future`

**Module:** `langgraph._internal._future`

These utilities form the cross-thread, cross-event-loop coroutine bridge used by `BackgroundExecutor`, `AsyncBackgroundExecutor`, and the main `Pregel` loop when bridging sync and async contexts.

### Key constants

```python
CONTEXT_NOT_SUPPORTED = sys.version_info < (3, 11)  # contextvars.copy_context()
EAGER_NOT_SUPPORTED   = sys.version_info < (3, 12)  # asyncio.eager_task_factory
```

### `_ensure_future()` behaviour by Python version

| Python | `lazy=True` | `lazy=False` |
|--------|------------|--------------|
| < 3.11 | `loop.create_task(coro)` | same (no context) |
| 3.11 | `loop.create_task(coro, context=ctx)` | same |
| ≥ 3.12 | `loop.create_task(coro, context=ctx)` | `asyncio.eager_task_factory(loop, coro, ctx)` |

`eager_task_factory` starts the coroutine immediately (on the calling thread) rather than scheduling it — dramatically reducing latency for fast-completing tasks like cache lookups.

### `chain_future()` — cross-loop state propagation

```python
def chain_future(source: AnyFuture, destination: AnyFuture) -> AnyFuture:
    # Registers callbacks on both futures so:
    # - when source completes, destination gets the same result/exception
    # - if destination is cancelled, source is cancelled too
    # - handles asyncio.Future ↔ concurrent.futures.Future bridging
    ...
```

### Example 1 — submit an async task from a sync thread

```python
import asyncio
from langgraph._internal._future import run_coroutine_threadsafe

async def slow_task(n: int) -> int:
    await asyncio.sleep(0.01)
    return n * n

async def main():
    loop = asyncio.get_running_loop()

    # Submit from async context (same loop) — returns asyncio.Future
    fut = run_coroutine_threadsafe(slow_task(5), loop, lazy=True)
    result = await fut
    print(f"5² = {result}")  # 25

    # With lazy=False on Python 3.12+, the coroutine starts immediately
    fut_eager = run_coroutine_threadsafe(slow_task(7), loop, lazy=False)
    result_eager = await fut_eager
    print(f"7² = {result_eager}")  # 49

asyncio.run(main())
```

### Example 2 — chain futures across sync/async boundary

```python
import asyncio
import concurrent.futures
from langgraph._internal._future import chain_future

async def async_producer() -> str:
    await asyncio.sleep(0.01)
    return "produced"

def sync_consumer():
    loop = asyncio.new_event_loop()
    sync_fut: concurrent.futures.Future[str] = concurrent.futures.Future()

    async def bridge():
        async_fut = asyncio.ensure_future(async_producer())
        # chain: when async_fut completes, sync_fut gets the result
        chain_future(async_fut, sync_fut)
        await async_fut

    loop.run_until_complete(bridge())
    loop.close()
    return sync_fut.result()

result = sync_consumer()
print(f"Sync consumer got: {result}")  # "produced"
```

### Example 3 — understand lazy vs eager task scheduling

```python
import asyncio
import time
from langgraph._internal._future import _ensure_future, EAGER_NOT_SUPPORTED

async def benchmark_future_creation(n: int) -> list[int]:
    loop = asyncio.get_running_loop()
    results = []

    async def quick_task(i: int) -> int:
        return i * 2

    start = time.perf_counter()
    tasks = []
    for i in range(n):
        # lazy=False uses eager_task_factory on Python 3.12+
        t = _ensure_future(quick_task(i), loop=loop, lazy=False)
        tasks.append(t)

    collected = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    eager_label = "lazy" if EAGER_NOT_SUPPORTED else "eager"
    print(f"{n} tasks ({eager_label}): {elapsed*1000:.1f}ms, sum={sum(collected)}")
    return list(collected)

asyncio.run(benchmark_future_creation(100))
```

---

## 7 · `validate_graph` + `validate_keys`

**Module:** `langgraph.pregel._validate`

`validate_graph` is called during `StateGraph.compile()` to catch configuration errors before the graph runs. Understanding its checks helps write better graphs and produce clearer custom error messages.

### Validation rules applied by `validate_graph`

| Check | Error raised | Condition |
|-------|-------------|-----------|
| Channel name in `RESERVED` | `ValueError` | channel name is a LangGraph internal constant |
| Managed value name in `RESERVED` | `ValueError` | same for managed values |
| Node name in `RESERVED` | `ValueError` | same for node names |
| Node reads unknown channel | `ValueError` | channel not in `channels` or `managed` |
| Subscribed channel missing | `ValueError` | a node subscribes to a channel that doesn't exist |
| Input channel missing | `ValueError` | not in `channels` |
| Input channel not subscribed | `ValueError` | no node reads it |
| Output channel missing | `ValueError` | not in `channels` |
| `interrupt_after` node missing | `ValueError` | node name not in graph |
| `interrupt_before` node missing | `ValueError` | node name not in graph |

### Example 1 — see validation errors from a misconfigured graph

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    counter: int

builder = StateGraph(State)

# Add a valid node
builder.add_node("step", lambda s: {"counter": s["counter"] + 1})
builder.add_edge(START, "step")
builder.add_edge("step", END)

# This compiles fine
graph = builder.compile()

# Now attempt to interrupt on a non-existent node
try:
    bad_graph = builder.compile(interrupt_before=["nonexistent_node"])
except ValueError as e:
    print(f"Caught: {e}")  # "Node nonexistent_node not in nodes"

# Reserved name check
class BadState(TypedDict):
    __interrupt__: str  # "__interrupt__" is RESERVED

bad_builder = StateGraph(BadState)
bad_builder.add_node("step", lambda s: s)
bad_builder.add_edge(START, "step")
bad_builder.add_edge("step", END)
try:
    bad_builder.compile()
except ValueError as e:
    print(f"Reserved name error: {e}")
```

### Example 2 — `validate_keys` for stream/output channel checks

```python
from langgraph.pregel._validate import validate_keys
from langgraph.channels.last_value import LastValue

# Simulate the channels dict a compiled graph would have
channels = {
    "messages": LastValue(list),
    "counter": LastValue(int),
}

# Valid — key exists
try:
    validate_keys("messages", channels)
    print("'messages' key: valid")
except ValueError as e:
    print(f"Error: {e}")

# Invalid — key does not exist
try:
    validate_keys("nonexistent", channels)
    print("'nonexistent' key: valid")
except ValueError as e:
    print(f"Caught: {e}")  # "Key nonexistent not in channels"

# Multiple keys
try:
    validate_keys(["messages", "counter"], channels)
    print("Both keys: valid")
except ValueError:
    print("One key missing")
```

### Example 3 — build a pre-compile graph linter

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    value: int
    result: str

def lint_graph(builder: StateGraph) -> list[str]:
    """Run validate_graph early to surface errors as strings instead of exceptions."""
    issues = []

    # Compile first to get the internal graph structure
    try:
        graph = builder.compile()
        # Access internal compiled graph attributes for validation
        pregel = graph
        issues.append("Graph compiled OK")

        # Additional custom checks
        state_fields = list(builder.state_schema.__annotations__.keys())
        if len(state_fields) == 0:
            issues.append("WARNING: State has no fields")
        else:
            issues.append(f"State fields: {state_fields}")
    except ValueError as e:
        issues.append(f"COMPILE ERROR: {e}")
    except Exception as e:
        issues.append(f"UNEXPECTED ERROR: {e}")

    return issues

builder = StateGraph(State)
builder.add_node("process", lambda s: {"value": s["value"] * 2, "result": str(s["value"] * 2)})
builder.add_edge(START, "process")
builder.add_edge("process", END)

for issue in lint_graph(builder):
    print(issue)
```

---

## 8 · `read_channel` + `read_channels` + `map_input` + `map_command`

**Module:** `langgraph.pregel._io`

These four functions form the I/O layer between the Pregel loop and the channel state. Understanding them explains how graph input is mapped to channels and how `Command` objects are converted to pending writes.

### Function signatures and semantics

```python
def read_channel(channels, chan, *, catch=True) -> Any:
    # Returns channels[chan].get() or None if EmptyChannelError and catch=True
    # Raises EmptyChannelError if catch=False

def read_channels(channels, select, *, skip_empty=True) -> dict | Any:
    # If select is a str: returns read_channel(channels, select)
    # If select is a list: returns {k: value} dict, skipping empty channels

def map_input(input_channels, chunk) -> Iterator[tuple[str, Any]]:
    # Maps invoke()/stream() input to (channel_name, value) pairs
    # If input_channels is str: single yield (input_channels, chunk)
    # If list: iterates chunk.items() and yields matching channel names

def map_command(cmd: Command) -> Iterator[tuple[str, str, Any]]:
    # Maps a Command to (task_id, channel, value) triples
    # cmd.goto → (NULL_TASK_ID, "branch:to:node", START) or TASKS/Send
    # cmd.resume → (NULL_TASK_ID, RESUME, value)
    # cmd.update → (NULL_TASK_ID, field_name, field_value)
```

### Example 1 — `read_channels` for node input reading

```python
from langgraph.channels.last_value import LastValue
from langgraph.pregel._io import read_channel, read_channels

# Build a minimal channels dict
messages_ch = LastValue(list)
counter_ch = LastValue(int)
empty_ch = LastValue(str)  # never updated → EmptyChannelError on .get()

# Populate channels manually for demonstration
messages_ch.update([[1, 2, 3]])
counter_ch.update([42])
# empty_ch is intentionally left unset

channels = {"messages": messages_ch, "counter": counter_ch, "empty": empty_ch}

# read_channel: single channel
print(read_channel(channels, "messages"))  # [1, 2, 3]
print(read_channel(channels, "counter"))   # 42

# catch=True converts EmptyChannelError (existing but unset channel) to None
# Note: a truly missing key raises KeyError regardless of catch=True
print(read_channel(channels, "empty", catch=True))  # None

# read_channels: multiple
result = read_channels(channels, ["messages", "counter"])
print(result)  # {"messages": [1, 2, 3], "counter": 42}
```

### Example 2 — `map_input` routing

```python
from langgraph.pregel._io import map_input

# Single-channel graph (input_channels is a string)
pairs = list(map_input("messages", [{"role": "user", "content": "hello"}]))
print(pairs)  # [("messages", [{"role": "user", "content": "hello"}])]

# Multi-channel graph (input_channels is a list)
pairs_multi = list(map_input(
    ["messages", "context"],
    {"messages": ["msg1"], "context": "some context", "extra": "ignored"},
))
# "extra" is not in input_channels, so it's dropped (with a logger.warning)
print(pairs_multi)
# [("messages", ["msg1"]), ("context", "some context")]

# None input yields nothing (used for resumed runs)
pairs_none = list(map_input("messages", None))
print(pairs_none)  # []
```

### Example 3 — `map_command` trace

```python
from langgraph.pregel._io import map_command
from langgraph.types import Command, Send
from langgraph._internal._constants import RESUME, TASKS, NULL_TASK_ID
from langgraph.constants import START

# Resume command
cmd = Command(resume={"approved": True})
for task_id, channel, value in map_command(cmd):
    print(f"  ({task_id!r}, {channel!r}, {value!r})")
# (NULL_TASK_ID, "RESUME", {"approved": True})

# Goto a node by name
cmd2 = Command(goto="review_node")
for task_id, channel, value in map_command(cmd2):
    print(f"  ({task_id!r}, {channel!r}, {value!r})")
# (NULL_TASK_ID, "branch:to:review_node", <START sentinel>)

# Goto with a Send (includes payload)
cmd3 = Command(goto=[Send("process", {"input": "data"})])
for task_id, channel, value in map_command(cmd3):
    print(f"  ({task_id!r}, {channel!r}, {value!r})")
# (NULL_TASK_ID, TASKS, Send(node="process", arg={"input": "data"}))

# Update + goto together
cmd4 = Command(update={"counter": 10}, goto="next_step")
writes = list(map_command(cmd4))
print(f"Total writes from Command: {len(writes)}")
# 2 writes: one for goto branch, one for counter update
```

---

## 9 · `should_interrupt` + `apply_writes` + `prepare_next_tasks`

**Module:** `langgraph.pregel._algo`

These three functions implement the core Pregel superstep loop. Each is called once per superstep; together they decide whether to pause, how to commit writes, and which nodes to run next.

### `should_interrupt()` — the interrupt decision

```python
def should_interrupt(checkpoint, interrupt_nodes, tasks) -> list[PregelExecutableTask]:
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})
    any_updates = any(
        version > seen.get(chan, null_version)
        for chan, version in checkpoint["channel_versions"].items()
    )
    if not any_updates:
        return []
    # Filter to tasks whose node name is in interrupt_nodes (or "*")
    return [t for t in tasks if (
        not t.config or TAG_HIDDEN not in t.config.get("tags", [])
        if interrupt_nodes == "*"
        else t.name in interrupt_nodes
    )]
```

Key insight: `should_interrupt` fires **only when at least one channel version changed since the last interrupt**. If nothing changed (a no-op step), it returns `[]` even if all node names match `interrupt_nodes`.

### `apply_writes()` — committing a superstep

`apply_writes` sorts tasks by `task_path_str(path[:3])` before applying writes. This deterministic ordering ensures that two concurrent tasks writing to the same `BinaryOperatorAggregate` channel produce the same result regardless of thread scheduling.

### `prepare_next_tasks()` — which nodes run next

`prepare_next_tasks` iterates all `PregelNode` objects, checks their `triggers` (subscribed channels), and schedules nodes whose trigger channels have updated versions. It builds `PregelExecutableTask` instances (for actual execution) or `PregelTask` instances (for state inspection).

### Example 1 — observe `should_interrupt` firing

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict

class State(TypedDict):
    step: int
    data: str

def process(state: State) -> dict:
    return {"step": state["step"] + 1, "data": f"processed-{state['step']}"}

def review(state: State) -> dict:
    # interrupt() causes should_interrupt to fire for interrupt_before=["review"]
    # only if channel versions changed since last interrupt checkpoint
    response = interrupt({"data": state["data"]})
    return {"data": f"approved:{response}"}

checkpointer = MemorySaver()
builder = StateGraph(State)
builder.add_node("process", process)
builder.add_node("review", review)
builder.add_edge(START, "process")
builder.add_edge("process", "review")
builder.add_edge("review", END)

# interrupt_before fires should_interrupt() before "review" runs
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["review"])
config = {"configurable": {"thread_id": "interrupt-demo"}}

state = graph.invoke({"step": 0, "data": ""}, config)
print(f"Paused at step={state['step']}, data={state['data']!r}")

# Resume — should_interrupt won't fire again (same checkpoint state)
from langgraph.types import Command
final = graph.invoke(Command(resume="human-approved"), config)
print(f"Final: step={final['step']}, data={final['data']!r}")
```

### Example 2 — trace `apply_writes` ordering with concurrent nodes

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    results: Annotated[list[str], operator.add]

def fan_out(state: State) -> dict:
    # Regular node: signals that fan-out should proceed
    return {}

def route_workers(state: State) -> list[Send]:
    # Conditional edge function: returns Send objects to dispatch parallel workers
    return [Send("worker", {"results": [], "id": i}) for i in range(3)]

def worker(state: dict) -> dict:
    # apply_writes sorts by path[:3] before applying, so results
    # accumulate in a deterministic order even though workers run concurrently
    return {"results": [f"worker-{state['id']}"]}

builder = StateGraph(State)
builder.add_node("fan_out", fan_out)
builder.add_node("worker", worker)
builder.add_edge(START, "fan_out")
builder.add_conditional_edges("fan_out", route_workers)
builder.add_edge("worker", END)
graph = builder.compile()

result = graph.invoke({"results": []})
# apply_writes deterministic sorting ensures consistent order
print(f"Results: {sorted(result['results'])}")  # always sorted
```

### Example 3 — understand `prepare_next_tasks` via `get_state()`

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict

class State(TypedDict):
    step: int

def node_a(state: State) -> dict:
    return {"step": state["step"] + 1}

def node_b(state: State) -> dict:
    interrupt("waiting for approval")
    return {"step": state["step"] + 10}

checkpointer = MemorySaver()
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile(checkpointer=checkpointer, interrupt_before=["b"])
config = {"configurable": {"thread_id": "next-tasks-demo"}}

graph.invoke({"step": 0}, config)

# prepare_next_tasks output is surfaced as state.next
current_state = graph.get_state(config)
print(f"Next nodes to run: {current_state.next}")  # ('b',)
print(f"Current step: {current_state.values['step']}")  # 1
print(f"Pending tasks: {len(current_state.tasks)}")  # 1
```

---

## 10 · `LazyAtomicCounter` + `task_path_str` + `_uuid5_str` / `_xxhash_str`

**Module:** `langgraph.pregel._algo`

These utilities form the task identity layer. Every `PregelExecutableTask` gets a stable, deterministic ID derived from its path, and counters are used to make those paths unique across concurrent invocations.

### Source signatures

```python
LAZY_ATOMIC_COUNTER_LOCK = threading.Lock()

class LazyAtomicCounter:
    """Thread-safe counter that initialises only on first call.
    Used by PregelScratchpad for call_counter, interrupt_counter, subgraph_counter.
    """
    __slots__ = ("_counter",)

    def __init__(self) -> None:
        self._counter = None

    def __call__(self) -> int:
        if self._counter is None:
            with LAZY_ATOMIC_COUNTER_LOCK:
                if self._counter is None:
                    self._counter = itertools.count(0).__next__
        return self._counter()  # 0, 1, 2, 3, ...


def task_path_str(tup: str | int | tuple) -> str:
    """Convert a task path element to a sortable string.
    Ints are zero-padded to 10 digits for lexicographic sort stability.
    Nested tuples get a '~'-prefixed comma-separated representation.
    """
    if isinstance(tup, (tuple, list)):
        return f"~{', '.join(task_path_str(x) for x in tup)}"
    elif isinstance(tup, int):
        return f"{tup:010d}"     # "0000000003" for int 3
    else:
        return str(tup)          # node name as-is


def _uuid5_str(namespace: bytes, *parts: str | bytes) -> str:
    """SHA-1 based UUID (deterministic, slower). Used for trusted inputs."""
    ...

def _xxhash_str(namespace: bytes, *parts: str | bytes) -> str:
    """XXH3-128 based UUID (non-cryptographic, fast). Default for task IDs."""
    ...
```

### Why `LazyAtomicCounter` is lazy

The counter is created on first call, not at `__init__` time. This avoids creating an `itertools.count` iterator for every node in a graph even if that node is never actually called. The double-checked locking pattern (check outside lock, then check again inside) prevents multiple threads from each creating their own counter.

### Example 1 — `LazyAtomicCounter` in isolation

```python
from langgraph.pregel._algo import LazyAtomicCounter
import threading

counter = LazyAtomicCounter()

# First call initialises the underlying itertools.count
assert counter() == 0
assert counter() == 1
assert counter() == 2

# Thread-safe: multiple threads see consecutive values
results = []
lock = threading.Lock()

def increment():
    val = counter()
    with lock:
        results.append(val)

threads = [threading.Thread(target=increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Counter values: {sorted(results)}")  # [3, 4, 5, 6, 7]
# Always consecutive integers regardless of thread ordering
```

### Example 2 — `task_path_str` for deterministic sorting

```python
from langgraph.pregel._algo import task_path_str

# String paths — node names
print(task_path_str("node_a"))    # "node_a"
print(task_path_str("node_b"))    # "node_b"

# Integer paths — zero-padded for sort stability
print(task_path_str(3))           # "0000000003"
print(task_path_str(10))          # "0000000010"
# "0000000003" < "0000000010" lexicographically — correct numeric order

# Nested tuple paths
print(task_path_str(("node_a", 2)))     # "~node_a, 0000000002"
print(task_path_str(("node_b", 1, 0)))  # "~node_b, 0000000001, 0000000000"

# apply_writes sorts by task_path_str(task.path[:3])
# so deterministic ordering is guaranteed even with concurrent workers
paths = [("worker", 2), ("worker", 0), ("worker", 1)]
sorted_paths = sorted(paths, key=lambda p: task_path_str(p))
print(f"Sorted: {sorted_paths}")  # [('worker', 0), ('worker', 1), ('worker', 2)]
```

### Example 3 — `_uuid5_str` vs `_xxhash_str` for task ID generation

```python
from langgraph.pregel._algo import _uuid5_str, _xxhash_str
import time

# Same namespace + parts → same UUID (deterministic)
NS = b"langgraph-task"

uid5_a = _uuid5_str(NS, "node_a", "run_1", "0")
uid5_b = _uuid5_str(NS, "node_a", "run_1", "0")
print(f"UUID5 deterministic: {uid5_a == uid5_b}")  # True

uxx_a = _xxhash_str(NS, "node_a", "run_1", "0")
uxx_b = _xxhash_str(NS, "node_a", "run_1", "0")
print(f"XXHash deterministic: {uxx_a == uxx_b}")   # True

# Different inputs → different UUIDs
uid5_diff = _uuid5_str(NS, "node_a", "run_2", "0")
print(f"Different run: {uid5_a != uid5_diff}")  # True

# Performance: xxhash is much faster for non-security use
N = 10_000
start = time.perf_counter()
for i in range(N):
    _xxhash_str(NS, "node", str(i))
xx_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
for i in range(N):
    _uuid5_str(NS, "node", str(i))
sha_ms = (time.perf_counter() - start) * 1000

print(f"XXHash: {xx_ms:.0f}ms for {N} IDs")
print(f"SHA-1:  {sha_ms:.0f}ms for {N} IDs")
print(f"XXHash speedup: {sha_ms/xx_ms:.1f}x")
```

---

## Summary: execution engine internals covered

All code examples in this volume are verified against **`langgraph==1.2.6`**. The symbols documented here existed in prior releases; 1.2.6 itself is a patch release focused on bug fixes (nested subgraph checkpoint namespace handling and v3 stream abort). This volume provides the first detailed source-verified documentation of these previously undocumented internals:

- `pregel._tools` — `StreamToolCallHandler` lifecycle, `_tool_call_writer` ContextVar, `TAG_NOSTREAM` suppression
- `_internal._replay` — `ReplayState._is_first_visit()` and NS_END namespace stripping
- `_internal._scratchpad` — `PregelScratchpad` (all 7 fields) and `IsLastStep`/`RemainingSteps` managed values
- `warnings` — `LangGraphDeprecationWarning` versioned hierarchy and `expected_removal` contract
- `pregel._checkpoint` — `empty_checkpoint()`, `create_checkpoint()`, `delta_channels_to_snapshot()` pipeline
- `_internal._future` — `chain_future`, `run_coroutine_threadsafe`, `_ensure_future` cross-loop bridge
- `pregel._validate` — all compile-time graph validation rules
- `pregel._io` — `read_channel`, `map_input`, `map_command` I/O layer
- `pregel._algo` — `should_interrupt`, `apply_writes`, `prepare_next_tasks` core Pregel algorithm
- `pregel._algo` — `LazyAtomicCounter`, `task_path_str`, `_uuid5_str`/`_xxhash_str` task identity utilities

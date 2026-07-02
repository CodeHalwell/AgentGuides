---
title: "LangGraph Class Deep-Dives Vol. 30"
description: "Source-verified deep dives into 10 practical patterns — abatch/abatch_as_completed, Pydantic/dataclass state schemas, ToolNode handle_tool_errors callable, @task timeouts, BinaryOperatorAggregate dict/set reducers, get_stream_writer, Durability modes, ToolCallRequest.override + wrap_tool_call, get_state_history + StateSnapshot time-travel, and @entrypoint previous + runtime.store — verified against langgraph==1.2.7."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 30"
  order: 61
---

Source-verified deep dives into **10 practical patterns**, each with **3 runnable examples**, verified against `langgraph==1.2.7` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `Pregel.abatch()` + `abatch_as_completed()`

`abatch` runs `ainvoke` concurrently on a list of inputs and returns a gathered list. `abatch_as_completed` is an async iterator that yields `(index, result)` tuples in completion order — the fastest graph wins, not the first input.

**Key signatures** (inherited from `langchain_core.runnables.base.Runnable`, overridden by Pregel):

```python
async def abatch(
    inputs: list[Input],
    config: RunnableConfig | list[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs,
) -> list[Output]: ...

async def abatch_as_completed(
    inputs: Sequence[Input],
    config: RunnableConfig | Sequence[RunnableConfig] | None = None,
    *,
    return_exceptions: bool = False,
    **kwargs,
) -> AsyncIterator[tuple[int, Output | Exception]]: ...
```

`max_concurrency` in the config dict controls how many `ainvoke` calls run simultaneously. Without it, all inputs fire at once.

**Example 1 — fire-and-gather batch**

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    question: str
    answer: str

async def respond(state: State) -> State:
    # Simulate variable latency
    await asyncio.sleep(0.1)
    return {"answer": f"Echo: {state['question']}"}

graph = (
    StateGraph(State)
    .add_node("respond", respond)
    .add_edge(START, "respond")
    .add_edge("respond", END)
    .compile()
)

async def main():
    inputs = [{"question": f"Q{i}"} for i in range(5)]
    results = await graph.abatch(inputs)
    for r in results:
        print(r["answer"])
    # Echo: Q0 … Echo: Q4  (ordered by input position)

asyncio.run(main())
```

**Example 2 — stream results as they complete**

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    n: int
    result: int

async def compute(state: State) -> State:
    await asyncio.sleep(state["n"] * 0.05)  # fast inputs finish first
    return {"result": state["n"] ** 2}

graph = (
    StateGraph(State)
    .add_node("compute", compute)
    .add_edge(START, "compute")
    .add_edge("compute", END)
    .compile()
)

async def main():
    inputs = [{"n": i} for i in [4, 1, 3, 2, 0]]
    async for idx, result in graph.abatch_as_completed(inputs):
        print(f"input[{idx}] (n={inputs[idx]['n']}) → {result['result']}")
    # input[4] (n=0) → 0  ← finishes first
    # input[1] (n=1) → 1
    # ...

asyncio.run(main())
```

**Example 3 — `return_exceptions=True` prevents one failure from cancelling others**

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int
    output: int

async def risky(state: State) -> State:
    if state["value"] == 2:
        raise ValueError("bad input 2")
    return {"output": state["value"] * 10}

graph = (
    StateGraph(State)
    .add_node("risky", risky)
    .add_edge(START, "risky")
    .add_edge("risky", END)
    .compile()
)

async def main():
    inputs = [{"value": i} for i in range(5)]
    results = await graph.abatch(inputs, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[{i}] ERROR: {r}")
        else:
            print(f"[{i}] OK: {r['output']}")
    # [2] ERROR: bad input 2  — others succeed

asyncio.run(main())
```

---

## 2 · Pydantic `BaseModel` / `@dataclass` as state schema

LangGraph supports three state schema styles: `TypedDict`, Pydantic `BaseModel`, and `@dataclass`. When a Pydantic `BaseModel` is the `state_schema`, LangGraph coerces dict input via the constructor (`schema(**input_dict)`), which triggers Pydantic validation including `@field_validator`. It tracks which fields were explicitly set via `model_fields_set` and only writes those back to channels on each step. Dataclasses work similarly — only explicitly returned keys update state.

**How it works internally (`get_update_as_tuples`):** When a Pydantic node return is merged back into state, LangGraph reads `model.model_fields_set` to discover which fields the node explicitly set. Only those fields are written to channels — fields absent from `model_fields_set` are skipped, so a node that returns `MyState(field_a="x")` without touching `field_b` leaves `field_b` unchanged.

**Example 1 — Pydantic state with field coercion**

```python
from typing import Annotated
from pydantic import BaseModel, field_validator
from langgraph.graph import StateGraph, START, END

class AgentState(BaseModel):
    messages: list[str] = []
    count: int = 0

    @field_validator("count")
    @classmethod
    def count_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("count must be >= 0")
        return v

def increment(state: AgentState) -> AgentState:
    return AgentState(count=state.count + 1)

graph = (
    StateGraph(AgentState)
    .add_node("increment", increment)
    .add_edge(START, "increment")
    .add_edge("increment", END)
    .compile()
)

result = graph.invoke({"count": 3})
print(result)  # {'messages': [], 'count': 4}

# Pydantic coerces on input too:
result2 = graph.invoke({"count": "5"})  # str "5" coerced to int
print(result2)  # {'messages': [], 'count': 6}
```

**Example 2 — `@dataclass` state with partial updates**

```python
from dataclasses import dataclass, field
from typing import Annotated
from langgraph.graph import StateGraph, START, END
import operator

@dataclass
class PipelineState:
    items: Annotated[list[str], operator.add] = field(default_factory=list)
    step: int = 0

def step_one(state: PipelineState) -> dict:
    return {"items": ["a", "b"], "step": 1}

def step_two(state: PipelineState) -> dict:
    return {"items": ["c"], "step": 2}

graph = (
    StateGraph(PipelineState)
    .add_node("step_one", step_one)
    .add_node("step_two", step_two)
    .add_edge(START, "step_one")
    .add_edge("step_one", "step_two")
    .add_edge("step_two", END)
    .compile()
)

result = graph.invoke(PipelineState())
print(result)
# {'items': ['a', 'b', 'c'], 'step': 2}
# 'items' accumulated via operator.add reducer
```

**Example 3 — Pydantic state: `model_fields_set` means only touched fields update**

```python
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

class AnalysisState(BaseModel):
    text: str = ""
    word_count: int = 0
    sentiment: str = "neutral"

def count_words(state: AnalysisState) -> AnalysisState:
    # Only sets word_count — sentiment stays "neutral"
    return AnalysisState(word_count=len(state.text.split()))

def analyze_text(state: AnalysisState) -> AnalysisState:
    # Only sets sentiment — word_count unchanged
    return AnalysisState(sentiment="positive" if "good" in state.text else "negative")

graph = (
    StateGraph(AnalysisState)
    .add_node("count", count_words)
    .add_node("analyze", analyze_text)
    .add_edge(START, "count")
    .add_edge("count", "analyze")
    .add_edge("analyze", END)
    .compile()
)

result = graph.invoke({"text": "This is a good day"})
print(result)
# {'text': 'This is a good day', 'word_count': 5, 'sentiment': 'positive'}
```

---

## 3 · `ToolNode` `handle_tool_errors` callable

By default `handle_tool_errors=True` catches any exception during tool execution and returns a `ToolMessage` with `status="error"` and the exception string as content. Passing a callable lets you format the error message yourself — useful for filtering sensitive stack traces or adding structured error context.

**Callable signature:** `(exception: Exception) -> str`

The returned string becomes the `ToolMessage.content`. If `handle_tool_errors=False`, the exception propagates and the graph run fails (or hits the `error_handler` if one is registered).

**Example 1 — default `True` vs custom callable**

```python
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import operator

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b

def format_error(exc: Exception) -> str:
    return f"Tool failed ({type(exc).__name__}): division by zero is not allowed"

tool_node = ToolNode(
    tools=[divide],
    handle_tool_errors=format_error,  # callable, not bool
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# Simulate a tool call that will fail
from langchain_core.messages import AIMessage
bad_call = AIMessage(
    content="",
    tool_calls=[{"name": "divide", "args": {"a": 1.0, "b": 0.0}, "id": "tc1", "type": "tool_call"}],
)
result = tool_node.invoke({"messages": [bad_call]})
error_msg: ToolMessage = result["messages"][0]
print(error_msg.status)   # error
print(error_msg.content)  # Tool failed (ZeroDivisionError): division by zero is not allowed
```

**Example 2 — `handle_tool_errors=False` lets errors propagate**

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import operator

@tool
def must_succeed(x: int) -> int:
    """Returns x doubled, fails on negative."""
    if x < 0:
        raise ValueError("negative not allowed")
    return x * 2

tool_node = ToolNode(tools=[must_succeed], handle_tool_errors=False)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

bad_call = AIMessage(
    content="",
    tool_calls=[{"name": "must_succeed", "args": {"x": -1}, "id": "tc2", "type": "tool_call"}],
)

try:
    tool_node.invoke({"messages": [bad_call]})
except Exception as e:
    print(f"Got uncaught error: {e}")  # Got uncaught error: negative not allowed
```

**Example 3 — `messages_key` for non-standard state keys**

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import operator

@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"ECHO: {text}"

# State uses "chat_history" instead of "messages"
class State(TypedDict):
    chat_history: Annotated[list[BaseMessage], operator.add]

tool_node = ToolNode(
    tools=[echo],
    messages_key="chat_history",  # non-default key
    handle_tool_errors=lambda e: f"[ERROR] {e}",
)

call = AIMessage(
    content="",
    tool_calls=[{"name": "echo", "args": {"text": "hello"}, "id": "tc3", "type": "tool_call"}],
)
result = tool_node.invoke({"chat_history": [call]})
msg: ToolMessage = result["chat_history"][0]
print(msg.content)  # ECHO: hello
```

---

## 4 · `@task(timeout=TimeoutPolicy(...))` + `NodeTimeoutError`

`@task` wraps an async function into a LangGraph functional-API task. The optional `timeout` parameter accepts a `TimeoutPolicy` (or bare float for `run_timeout`). **Async only** — sync tasks raise `ValueError` if `timeout` is set.

Each **retry** of a task gets a **fresh timeout clock** — the timeout is per-attempt, not total. `NodeTimeoutError` is injected into the `error_handler` with `kind="run"` (hard cap exceeded) or `kind="idle"` (no-progress timeout).

```python
from langgraph.types import TimeoutPolicy, RetryPolicy
from langgraph.errors import NodeTimeoutError
```

**Example 1 — hard run timeout**

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.types import TimeoutPolicy

@task(timeout=TimeoutPolicy(run_timeout=1.0))  # hard 1-second cap
async def slow_fetch(url: str) -> str:
    await asyncio.sleep(5)   # will be cancelled
    return "done"

@entrypoint()
async def pipeline(input: dict) -> dict:
    try:
        result = await slow_fetch(input["url"])
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

result = asyncio.run(pipeline.ainvoke({"url": "https://example.com"}))
print(result)  # {'error': '...NodeTimeoutError...'} or similar timeout message
```

**Example 2 — idle timeout with heartbeat reset**

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.types import TimeoutPolicy
from langgraph.runtime import Runtime

@task(timeout=TimeoutPolicy(idle_timeout=2.0, refresh_on="heartbeat"))
async def chunked_processor(items: list[str], runtime: Runtime) -> list[str]:
    results = []
    for item in items:
        await asyncio.sleep(0.5)
        runtime.heartbeat()  # reset idle clock
        results.append(item.upper())
    return results

@entrypoint()
async def pipeline(input: dict, runtime: Runtime) -> dict:
    processed = await chunked_processor(input["items"], runtime)
    return {"processed": processed}

result = asyncio.run(
    pipeline.ainvoke({"items": ["a", "b", "c"]})
)
print(result)  # {'processed': ['A', 'B', 'C']}
```

**Example 3 — retry with fresh timeout per attempt**

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.types import TimeoutPolicy, RetryPolicy

attempt_count = 0

@task(
    timeout=TimeoutPolicy(run_timeout=1.5),
    retry=RetryPolicy(max_attempts=3),
)
async def flaky_task(n: int) -> str:
    global attempt_count
    attempt_count += 1
    if attempt_count < 3:
        await asyncio.sleep(2.0)  # times out; each retry gets a fresh 1.5s
    return f"success on attempt {attempt_count}"

@entrypoint()
async def pipeline(input: dict) -> dict:
    result = await flaky_task(input["n"])
    return {"result": result}

result = asyncio.run(pipeline.ainvoke({"n": 1}))
print(result)  # {'result': 'success on attempt 3'}
print(f"Total attempts: {attempt_count}")  # 3
```

---

## 5 · `BinaryOperatorAggregate` with dict-merge / set-union reducers + `Overwrite` bypass

`BinaryOperatorAggregate` is the channel backing `Annotated[T, reducer_fn]` fields. The reducer is called `reducer(current_value, new_value)` at each step. Two practical patterns: **dict-merge** (overlay keys from new onto existing) and **set-union** (accumulate members across steps). The `Overwrite` sentinel bypasses the reducer entirely — `Overwrite(value)` writes `value` directly as the new channel state.

**`MISSING` bootstrap:** On the very first step, `current_value` is `MISSING` (a sentinel object). Robust reducers must handle this. LangGraph calls `reducer(MISSING, first_value)` to initialize. For dict merge, check `if a is MISSING`.

**Example 1 — dict-merge reducer (overlay new keys)**

```python
from typing import Annotated, TypedDict
from langgraph._internal._typing import MISSING
from langgraph.graph import StateGraph, START, END

def merge_dicts(a: dict, b: dict) -> dict:
    if a is MISSING:
        return b
    return {**a, **b}  # b keys override a

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]

def add_source(state: State) -> dict:
    return {"metadata": {"source": "web", "lang": "en"}}

def add_model(state: State) -> dict:
    return {"metadata": {"model": "claude", "lang": "fr"}}  # overrides "lang"

graph = (
    StateGraph(State)
    .add_node("source", add_source)
    .add_node("model", add_model)
    .add_edge(START, "source")
    .add_edge("source", "model")
    .add_edge("model", END)
    .compile()
)

result = graph.invoke({"metadata": {}})
print(result["metadata"])
# {'source': 'web', 'lang': 'fr', 'model': 'claude'}
```

**Example 2 — set-union reducer (accumulate unique members)**

```python
from typing import Annotated, TypedDict
from langgraph._internal._typing import MISSING
from langgraph.graph import StateGraph, START, END

def union_sets(a: set, b: set) -> set:
    if a is MISSING:
        return set(b)
    return a | b

class State(TypedDict):
    visited: Annotated[set, union_sets]
    next_page: str

def crawl_a(state: State) -> dict:
    return {"visited": {"page_a", "page_b"}, "next_page": "page_c"}

def crawl_b(state: State) -> dict:
    return {"visited": {"page_b", "page_c"}}  # page_b deduped

graph = (
    StateGraph(State)
    .add_node("crawl_a", crawl_a)
    .add_node("crawl_b", crawl_b)
    .add_edge(START, "crawl_a")
    .add_edge("crawl_a", "crawl_b")
    .add_edge("crawl_b", END)
    .compile()
)

result = graph.invoke({"visited": set(), "next_page": ""})
print(result["visited"])   # {'page_a', 'page_b', 'page_c'}
print(len(result["visited"]))  # 3 — page_b not duplicated
```

**Example 3 — `Overwrite` sentinel bypasses the reducer**

```python
from typing import Annotated, TypedDict
from langgraph._internal._typing import MISSING
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite
import operator

def accumulate(a: list, b: list) -> list:
    if a is MISSING:
        return list(b)
    return a + b

class State(TypedDict):
    items: Annotated[list[str], accumulate]

def add_items(state: State) -> dict:
    return {"items": ["x", "y"]}

def reset_items(state: State) -> dict:
    # Overwrite bypasses the accumulate reducer entirely
    return {"items": Overwrite(["fresh"])}

graph = (
    StateGraph(State)
    .add_node("add", add_items)
    .add_node("reset", reset_items)
    .add_edge(START, "add")
    .add_edge("add", "reset")
    .add_edge("reset", END)
    .compile()
)

result = graph.invoke({"items": ["initial"]})
print(result["items"])   # ['fresh']  — not ['initial', 'x', 'y', 'fresh']
```

---

## 6 · `get_stream_writer()` + custom event emission

`get_stream_writer()` returns the `StreamWriter` callable bound to the current node's execution context. Calling it emits events on the `"custom"` stream mode — use this to push structured progress updates, intermediate results, or UI events from inside any node or task without threading it through state.

```python
from langgraph.config import get_stream_writer
```

The returned callable accepts a single serializable value. Outside an active graph run it returns a no-op writer. In `stream_mode="custom"` (or combined modes like `["values", "custom"]`), these events surface as `StreamPart(type="custom", data=...)`.

**Example 1 — emit progress events from a node**

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    items: list[str]
    processed: list[str]

async def process_items(state: State) -> dict:
    writer = get_stream_writer()
    results = []
    for i, item in enumerate(state["items"]):
        await asyncio.sleep(0.01)
        results.append(item.upper())
        writer({"type": "progress", "done": i + 1, "total": len(state["items"])})
    return {"processed": results}

graph = (
    StateGraph(State)
    .add_node("process", process_items)
    .add_edge(START, "process")
    .add_edge("process", END)
    .compile()
)

async def main():
    async for chunk in graph.astream(
        {"items": ["a", "b", "c"], "processed": []},
        stream_mode="custom",
    ):
        print(chunk)  # {'type': 'progress', 'done': 1, 'total': 3} ...

asyncio.run(main())
```

**Example 2 — combined `values` + `custom` stream modes**

```python
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    query: str
    answer: str

async def answer_node(state: State) -> dict:
    writer = get_stream_writer()
    writer({"status": "thinking"})
    await asyncio.sleep(0.05)
    writer({"status": "done"})
    return {"answer": f"Answer to: {state['query']}"}

graph = (
    StateGraph(State)
    .add_node("answer", answer_node)
    .add_edge(START, "answer")
    .add_edge("answer", END)
    .compile()
)

async def main():
    async for mode, data in graph.astream(
        {"query": "What is LangGraph?", "answer": ""},
        stream_mode=["values", "custom"],
    ):
        if mode == "custom":
            print(f"CUSTOM: {data}")
        else:
            print(f"VALUES: {data}")

asyncio.run(main())
```

**Example 3 — `get_stream_writer` in a functional API `@task`**

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.config import get_stream_writer

@task
async def search(query: str) -> list[str]:
    writer = get_stream_writer()
    writer({"event": "search_started", "query": query})
    await asyncio.sleep(0.05)
    results = [f"result_{i}" for i in range(3)]
    writer({"event": "search_done", "count": len(results)})
    return results

@entrypoint()
async def agent(input: dict) -> dict:
    results = await search(input["query"])
    return {"results": results}

async def main():
    events = []
    async for mode, data in agent.astream(
        {"query": "langgraph"},
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            events.append(data)
    print(events)
    # [{'event': 'search_started', ...}, {'event': 'search_done', ...}]

asyncio.run(main())
```

---

## 7 · `StateGraph.compile(durability=)` + `Durability` literal

`Durability` is a `Literal["sync", "async", "exit"]` that controls when checkpoints are written:

- **`"sync"`** (default) — checkpoint written synchronously after each step completes, before the next step starts. Safest: you can resume from any step.
- **`"async"`** — checkpoint written in a background thread while the next step starts immediately. Faster throughput; a crash mid-step may lose that step's checkpoint.
- **`"exit"`** — checkpoint written only when the graph run completes (or on interrupt). Fastest; no recovery possible for in-progress runs.

```python
from langgraph.types import Durability  # Literal["sync", "async", "exit"]
graph.compile(checkpointer=saver, durability="exit")
```

**Example 1 — `"sync"` durability (default, safest)**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    step: int
    log: list[str]

saver = InMemorySaver()

def node_a(state: State) -> dict:
    return {"step": state["step"] + 1, "log": ["node_a ran"]}

def node_b(state: State) -> dict:
    return {"step": state["step"] + 1, "log": ["node_b ran"]}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("b", END)
    .compile(checkpointer=saver, durability="sync")  # explicit
)

config = {"configurable": {"thread_id": "t1"}}
result = graph.invoke({"step": 0, "log": []}, config=config)
print(result)

# Can resume from any checkpoint
history = list(graph.get_state_history(config))
print(f"{len(history)} checkpoints saved")  # one per step + initial
```

**Example 2 — `"exit"` durability for high-throughput pipelines**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import time

class State(TypedDict):
    value: int

saver = InMemorySaver()

def step(state: State) -> dict:
    return {"value": state["value"] + 1}

# "exit" only checkpoints at completion — fastest for batch pipelines
graph = (
    StateGraph(State)
    .add_node("step", step)
    .add_edge(START, "step")
    .add_edge("step", END)
    .compile(checkpointer=saver, durability="exit")
)

config = {"configurable": {"thread_id": "batch-1"}}
result = graph.invoke({"value": 0}, config=config)
print(result)  # {'value': 1}

# Only the final checkpoint exists
history = list(graph.get_state_history(config))
print(f"checkpoints: {len(history)}")  # 1 (only completion checkpoint)
```

**Example 3 — choosing durability based on use case**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: list[str]

saver = InMemorySaver()

def chat_node(state: State) -> dict:
    return {"messages": state["messages"] + ["response"]}

def build_graph(durability: str):
    return (
        StateGraph(State)
        .add_node("chat", chat_node)
        .add_edge(START, "chat")
        .add_edge("chat", END)
        .compile(checkpointer=saver, durability=durability)
    )

# HITL / long-running: use "sync" for step-by-step recoverability
hitl_graph = build_graph("sync")

# Background batch processing: use "exit" for throughput
batch_graph = build_graph("exit")

# Streaming with parallel work: use "async" for latency reduction
streaming_graph = build_graph("async")

cfg = {"configurable": {"thread_id": "demo"}}
print(hitl_graph.invoke({"messages": ["hi"]}, config=cfg))
```

---

## 8 · `ToolCallRequest.override()` + `wrap_tool_call` patterns

`ToolNode` accepts a `wrap_tool_call` parameter. It is **middleware-style**: the wrapper receives the request *and* a `call_next` callable, then decides whether to call `call_next(req)` (possibly with a modified `req`) or short-circuit entirely.

```python
# Type aliases (from langgraph.prebuilt source):
# ToolCallWrapper      = Callable[[ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]], ToolMessage | Command]
# AsyncToolCallWrapper = Callable[[ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]], Awaitable[ToolMessage | Command]]
```

`ToolCallRequest` is a frozen dataclass:
```python
@dataclass(frozen=True)
class ToolCallRequest:
    tool_call: dict        # {"name": ..., "args": ..., "id": ...}
    tool: BaseTool         # the bound tool object
    state: dict            # current graph state snapshot

    def override(self, **kwargs) -> ToolCallRequest: ...  # returns new instance
```

`.override()` accepts keys `tool_call`, `tool`, `state`. Returns a new immutable instance; the original is unchanged.

**Example 1 — logging middleware**

```python
import json
import time
from typing import Annotated, TypedDict, Callable
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, ToolCallRequest
import operator

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

def logging_middleware(
    req: ToolCallRequest,
    call_next: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] → {req.tool_call['name']}({json.dumps(req.tool_call['args'])})")
    result = call_next(req)  # actually execute the tool
    print(f"[{ts}] ← {result.content!r}")
    return result

tool_node = ToolNode(
    tools=[get_weather],
    wrap_tool_call=logging_middleware,
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

call = AIMessage(
    content="",
    tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "tc1", "type": "tool_call"}],
)
result = tool_node.invoke({"messages": [call]})
# Logs:  → get_weather({"city": "Paris"})
#        ← 'Sunny in Paris'
print(result["messages"][0].content)  # Sunny in Paris
```

**Example 2 — argument injection via `.override()`**

```python
from typing import Annotated, TypedDict, Callable
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, ToolCallRequest
import operator

@tool
def search_database(query: str, tenant_id: str) -> str:
    """Search the database for a tenant."""
    return f"Results for '{query}' in tenant '{tenant_id}'"

def inject_tenant(
    req: ToolCallRequest,
    call_next: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    # Inject tenant_id from state before the model's args reach the tool
    tenant_id = req.state.get("tenant_id", "default")
    new_args = {**req.tool_call["args"], "tenant_id": tenant_id}
    new_tool_call = {**req.tool_call, "args": new_args}
    return call_next(req.override(tool_call=new_tool_call))

tool_node = ToolNode(
    tools=[search_database],
    wrap_tool_call=inject_tenant,
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tenant_id: str

call = AIMessage(
    content="",
    tool_calls=[{"name": "search_database", "args": {"query": "orders"}, "id": "tc2", "type": "tool_call"}],
)
result = tool_node.invoke({"messages": [call], "tenant_id": "acme-corp"})
print(result["messages"][0].content)  # Results for 'orders' in tenant 'acme-corp'
```

**Example 3 — approval gate (short-circuit without calling `call_next`)**

```python
from typing import Annotated, TypedDict, Callable
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, ToolCallRequest
import operator

BLOCKED_TOOLS = {"delete_records", "drop_table"}

@tool
def delete_records(table: str) -> str:
    """Delete all records from a table."""
    return f"Deleted from {table}"

@tool
def list_records(table: str) -> str:
    """List records from a table."""
    return f"Records in {table}: ..."

def approval_gate(
    req: ToolCallRequest,
    call_next: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    if req.tool_call["name"] in BLOCKED_TOOLS:
        raise PermissionError(
            f"Tool '{req.tool_call['name']}' is blocked by policy"
        )
    return call_next(req)  # allow — execute the tool

tool_node = ToolNode(
    tools=[delete_records, list_records],
    wrap_tool_call=approval_gate,
    handle_tool_errors=lambda e: f"BLOCKED: {e}",
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

blocked_call = AIMessage(
    content="",
    tool_calls=[{"name": "delete_records", "args": {"table": "users"}, "id": "tc3", "type": "tool_call"}],
)
result = tool_node.invoke({"messages": [blocked_call]})
msg = result["messages"][0]
print(msg.status)   # error
print(msg.content)  # BLOCKED: Tool 'delete_records' is blocked by policy
```

---

## 9 · `Pregel.get_state()` + `get_state_history()` + `StateSnapshot` time-travel

`get_state(config)` returns a `StateSnapshot` of the latest checkpoint for a thread. `get_state_history(config)` returns an iterator over **all** checkpoints, **newest first**. Use `before=`, `filter=`, and `limit=` to narrow the result. Replaying from an old checkpoint (time-travel) is as simple as passing its `checkpoint_id` back to `invoke`.

```python
@dataclass
class StateSnapshot:
    values: dict                      # channel values at this point
    next: tuple[str, ...]             # nodes that would run next
    config: RunnableConfig            # config including checkpoint_id
    metadata: CheckpointMetadata      # step number, source, parents
    created_at: str                   # ISO-8601 timestamp
    parent_config: RunnableConfig | None
    tasks: tuple[PregelTask, ...]     # pending tasks (if interrupted)
```

**Example 1 — inspect current state and next nodes**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

class State(TypedDict):
    counter: int
    approved: bool

saver = InMemorySaver()

def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}

def approve(state: State) -> dict:
    interrupt("Approve this?")
    return {"approved": True}

graph = (
    StateGraph(State)
    .add_node("increment", increment)
    .add_node("approve", approve)
    .add_edge(START, "increment")
    .add_edge("increment", "approve")
    .add_edge("approve", END)
    .compile(checkpointer=saver, interrupt_before=["approve"])
)

config = {"configurable": {"thread_id": "t1"}}
graph.invoke({"counter": 0, "approved": False}, config=config)

snap = graph.get_state(config)
print(snap.values)           # {'counter': 1, 'approved': False}
print(snap.next)             # ('approve',) — waiting at approve node
```

**Example 2 — iterate history and replay from a past snapshot**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    value: int

saver = InMemorySaver()

def add_one(state: State) -> dict:
    return {"value": state["value"] + 1}

graph = (
    StateGraph(State)
    .add_node("add", add_one)
    .add_edge(START, "add")
    .add_edge("add", END)
    .compile(checkpointer=saver)
)

thread_config = {"configurable": {"thread_id": "replay-demo"}}

# Run 3 times to build history
for _ in range(3):
    graph.invoke(graph.get_state(thread_config).values if graph.get_state(thread_config) else {"value": 0}, config=thread_config)

# List all checkpoints (newest first)
history = list(graph.get_state_history(thread_config))
print(f"Total checkpoints: {len(history)}")

# Time-travel: replay from the very first checkpoint
oldest = history[-1]
print(f"Oldest value: {oldest.values}")

replay_config = oldest.config  # contains checkpoint_id
replay_result = graph.invoke(None, config=replay_config)
print(f"Replayed result: {replay_result}")
```

**Example 3 — `filter=` and `limit=` to scope history**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    step: int

saver = InMemorySaver()

def tick(state: State) -> dict:
    return {"step": state["step"] + 1}

graph = (
    StateGraph(State)
    .add_node("tick", tick)
    .add_edge(START, "tick")
    .add_edge("tick", END)
    .compile(checkpointer=saver)
)

config = {"configurable": {"thread_id": "filter-demo"}}
state = {"step": 0}
for _ in range(5):
    state = graph.invoke(state, config=config)

# Limit to last 2 checkpoints
recent = list(graph.get_state_history(config, limit=2))
print(f"Recent snapshots: {[s.values for s in recent]}")
# [{'step': 5}, {'step': 4}]  — newest first

# Get snapshot right before a specific checkpoint
third = list(graph.get_state_history(config))[2]
before_third = list(graph.get_state_history(config, before=third.config, limit=1))
print(f"Before 3rd: {[s.values for s in before_third]}")
```

---

## 10 · `@entrypoint(checkpointer=, store=)` + `previous` + `runtime.store`

The functional API's `@entrypoint` supports the same persistence primitives as `StateGraph`. Pass a `checkpointer` to enable thread-level memory — the entrypoint's return value is automatically saved and surfaced as `previous` on the next call with the same `thread_id`. Pass a `store` to get a `BaseStore` available via `runtime.store` for cross-thread memory.

`entrypoint.final(value=..., save=...)` lets you return one value to the caller while saving a different value to the checkpoint — useful when the checkpoint representation differs from the API response.

**Example 1 — `previous` for stateful accumulation**

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()

@task
async def process(item: str, history: list[str]) -> str:
    # Summarize with history context
    return f"{item} (after {len(history)} prior items)"

@entrypoint(checkpointer=saver)
async def accumulator(new_item: str, previous: list[str] | None = None) -> list[str]:
    history = previous or []
    result = await process(new_item, history)
    updated = history + [result]
    return updated

async def main():
    config = {"configurable": {"thread_id": "acc-1"}}
    r1 = await accumulator.ainvoke("apple", config=config)
    r2 = await accumulator.ainvoke("banana", config=config)
    r3 = await accumulator.ainvoke("cherry", config=config)
    print(r3)
    # ['apple (after 0 prior items)',
    #  'banana (after 1 prior items)',
    #  'cherry (after 2 prior items)']

asyncio.run(main())
```

**Example 2 — `runtime.store` for cross-thread memory**

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

saver = InMemorySaver()
store = InMemoryStore()

# Pre-populate store
store.put(("profiles",), "user_42", {"name": "Alice", "pref": "concise"})

@entrypoint(checkpointer=saver, store=store)
async def personalized_agent(query: str, runtime: Runtime) -> str:
    user_id = "user_42"
    profile = None
    if runtime.store:
        item = runtime.store.get(("profiles",), user_id)
        if item:
            profile = item.value

    if profile and profile.get("pref") == "concise":
        return f"Short answer: {query}"
    return f"Detailed answer to: {query}"

async def main():
    config = {"configurable": {"thread_id": "personalized-1"}}
    result = await personalized_agent.ainvoke("What is LangGraph?", config=config)
    print(result)  # Short answer: What is LangGraph?

asyncio.run(main())
```

**Example 3 — `entrypoint.final(value=, save=)` — return ≠ save**

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

saver = InMemorySaver()

@task
async def summarize(history: list[dict]) -> str:
    # Build a compact summary from full history
    return f"Summary of {len(history)} exchanges"

@entrypoint(checkpointer=saver)
async def chat(message: str, previous: list[dict] | None = None) -> str:
    history = previous or []
    response = f"Reply to: {message}"

    full_history = history + [{"user": message, "bot": response}]

    # Return the human-readable response to the caller, but
    # save the full history dict for the next invocation's `previous`
    return entrypoint.final(value=response, save=full_history)

async def main():
    config = {"configurable": {"thread_id": "chat-final"}}
    r1 = await chat.ainvoke("Hello", config=config)
    r2 = await chat.ainvoke("How are you?", config=config)
    print(r1)  # Reply to: Hello       (string, not list)
    print(r2)  # Reply to: How are you?

    # previous on next call will be the full_history list, not the string
    r3 = await chat.ainvoke("Bye", config=config)
    print(r3)  # Reply to: Bye

asyncio.run(main())
```

---

## Quick-reference table

| Class / API | Module | Key behaviour |
|---|---|---|
| `Pregel.abatch()` | `langgraph.pregel` | Concurrent `ainvoke`, gathered results |
| `Pregel.abatch_as_completed()` | `langgraph.pregel` | Async iterator, `(index, result)` in completion order |
| `BaseModel` / `@dataclass` state | `langgraph.graph` | Pydantic coercion; `model_fields_set` filters updates |
| `ToolNode(handle_tool_errors=fn)` | `langgraph.prebuilt` | Callable receives `Exception`, returns error `str` |
| `@task(timeout=TimeoutPolicy)` | `langgraph.func` | Async only; per-attempt timeout; `NodeTimeoutError.kind` |
| `BinaryOperatorAggregate` | `langgraph.channels` | dict-merge / set-union via `Annotated`; `Overwrite` bypass |
| `get_stream_writer()` | `langgraph.config` | `StreamWriter` callable; emits `"custom"` mode events |
| `Durability` | `langgraph.types` | `"sync"` / `"async"` / `"exit"` checkpoint timing |
| `ToolCallRequest.override()` | `langgraph.prebuilt` | Immutable override before tool execution |
| `get_state_history()` | `langgraph.pregel` | `StateSnapshot` iterator, newest-first; `filter` / `before` / `limit` |
| `@entrypoint(checkpointer, store)` | `langgraph.func` | `previous` + `runtime.store`; `entrypoint.final(value, save)` |

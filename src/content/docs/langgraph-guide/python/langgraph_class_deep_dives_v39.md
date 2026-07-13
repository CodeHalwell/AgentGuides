---
title: "LangGraph Class Deep-Dives Vol. 39"
description: "Source-verified deep dives (langgraph==1.2.9) into 10 class groups: Command/Send (core control-flow routing primitives — graph/update/resume/goto fields, PARENT sentinel, Send per-task timeout), StateSnapshot/CheckpointMetadata (time-travel NamedTuple carrying values/next/tasks/interrupts; TypedDict with source/step/parents counters_since_delta_snapshot), PregelExecutableTask (runtime task introspection — name/input/proc/writes/config/triggers/retry_policy/cache_key/id/path/writers/subgraphs/timeout dataclass), create_react_agent/tools_condition (prebuilt ReAct loop factory — model/tools/prompt/response_format/pre_model_hook/post_model_hook; tools_condition list/dict/BaseModel state dispatch returning 'tools' or '__end__'), RemoteGraph/get_client/get_sync_client (remote deployment adapter — assistant_id/url/api_key/headers/client/sync_client; ASGI in-process loopback vs HTTP transport; api_key NOT_PROVIDED sentinel), ToolCallTransformer/ToolCallStream (per-tool-call streaming handles — tool-started/tool-output-delta/tool-finished/tool-error lifecycle; output_deltas StreamChannel; sync and async iteration), DebugTransformer/TasksTransformer (native stream projections for debug and tasks modes — scope-filtered StreamChannel push, run.debug and run.tasks attributes), SubgraphTransformer (direct-child subgraph discovery via lifecycle events — _make_child mini-mux cloning, SubgraphRunStream/AsyncSubgraphRunStream handle pushing, recursive grandchild tracking), LifecyclePayload/GraphDrained (lifecycle signal typed payload — event/namespace/graph_name/trigger_call_id/cause/error; GraphDrained cooperative drain with reason field), and RunnableCallable (lightweight callable node wrapper — func/afunc sync+async pair; KWARGS_CONFIG_KEYS auto-injection of config/store/stream_writer; explode_args dict unpacking; trace/recurse flags)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 39"
  order: 70
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.9` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `Command` · `Send`

**Module:** `langgraph.types`

`Command` and `Send` are the two core control-flow primitives that let nodes steer graph execution without hard-wired edges.

**Key source facts** (from `langgraph/types.py`):

- `Command` is a `@dataclass` with four fields: `graph: str | None` (defaults to current graph; `Command.PARENT = "__parent__"` targets the nearest parent), `update: Any | None` (state patch applied before routing), `resume: dict | Any | None` (value delivered to the next `interrupt()` call — either a `{interrupt_id: value}` mapping or a bare value that resolves the next interrupt in order), and `goto: Send | Sequence[Send | N] | N = ()` (destination node(s) or `Send` objects).
- `Command._update_as_tuples()` handles three update shapes: plain `dict`, a list/tuple of `(key, value)` pairs, and a Pydantic/dataclass model where `get_cached_annotated_keys` extracts annotated keys. Non-dict scalars become `[("__root__", value)]` for root-type states.
- `Send` uses `__slots__ = ("node", "arg", "timeout")` for zero-overhead iteration. The `timeout` argument accepts a bare `float` (seconds), `timedelta`, or a `TimeoutPolicy` — all normalised via `TimeoutPolicy.coerce(timeout)`. `Send` is hashable and set/dict-safe **only when `arg` is hashable** (e.g. a frozen dataclass or a string); the common case of passing a state `dict` as `arg` makes `Send` unhashable.
- A `Command` returned from a node is treated identically to one returned from a conditional edge. Returning `Command(goto="other_node", update={"key": val})` atomically updates state **and** routes.
- `Command.PARENT` works only when the current subgraph was invoked by a parent Pregel; using it at the root raises a `KeyError` from the Pregel config lookup.

### Example 1 — `Command` for dynamic routing with state update

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class State(TypedDict):
    counter: int
    status: str


def increment(state: State) -> Command:
    new_val = state["counter"] + 1
    if new_val >= 3:
        return Command(update={"counter": new_val, "status": "done"}, goto=END)
    return Command(update={"counter": new_val, "status": "running"}, goto="increment")


builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")

graph = builder.compile()
result = graph.invoke({"counter": 0, "status": "running"})
print(result)  # {'counter': 3, 'status': 'done'}
```

### Example 2 — `Send` for parallel map-reduce fan-out

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class State(TypedDict):
    topics: list[str]
    summaries: Annotated[list[str], operator.add]


def fan_out(state: State) -> list[Send]:
    return [Send("summarise", {"topic": t}) for t in state["topics"]]


def summarise(state: dict) -> dict:
    return {"summaries": [f"Summary of: {state['topic']}"]}


builder = StateGraph(State)
builder.add_node("summarise", summarise)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("summarise", END)

graph = builder.compile()
result = graph.invoke({"topics": ["AI", "Climate", "Finance"], "summaries": []})
print(result["summaries"])
# ['Summary of: AI', 'Summary of: Climate', 'Summary of: Finance']
```

### Example 3 — `Command(resume=...)` to provide a value to `interrupt()`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    question: str
    answer: str


def ask_human(state: State) -> dict:
    human_answer = interrupt({"prompt": state["question"]})
    return {"answer": human_answer}


builder = StateGraph(State)
builder.add_node("ask", ask_human)
builder.add_edge(START, "ask")
builder.add_edge("ask", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "q1"}}

# First invoke — pauses at interrupt()
graph.invoke({"question": "What is 2+2?", "answer": ""}, config=cfg)

# Resume with Command(resume=...) — delivers "4" to the interrupt() call
result = graph.invoke(Command(resume="4"), config=cfg)
print(result["answer"])  # 4
```

---

## 2 · `StateSnapshot` · `CheckpointMetadata`

**Module:** `langgraph.types`

`StateSnapshot` and `CheckpointMetadata` are the two types you work with when doing time-travel: inspecting saved states, rewinding to a prior step, or forking a thread.

**Key source facts** (from `langgraph/types.py`):

- `StateSnapshot` is a `NamedTuple` with eight fields: `values` (the channel dict at that step), `next` (tuple of node names to execute *next* — empty at terminal steps), `config` (the `RunnableConfig` that fetches this snapshot — pass it back to `update_state` to fork here), `metadata: CheckpointMetadata | None`, `created_at: str | None` (ISO-8601 UTC string), `parent_config: RunnableConfig | None` (the prior snapshot's config — walk `parent_config` to traverse history without `get_state_history`), `tasks: tuple[PregelTask, ...]` (tasks that *were* scheduled at this step; if the step failed mid-execution these carry the error), and `interrupts: tuple[Interrupt, ...]` (pending `interrupt()` calls not yet resolved).
- `CheckpointMetadata` is `TypedDict(total=False)` with `source: Literal["input","loop","update","fork"]` (how the checkpoint was created), `step: int` (-1 for the `"input"` checkpoint, 0 for the first loop checkpoint, then incrementing), `parents: dict[str, str]` (mapping from checkpoint namespace to parent checkpoint ID — used by subgraph hydration), `run_id: str`, and the beta field `counters_since_delta_snapshot: dict[str, tuple[int, int]]` (per-channel `(updates, supersteps)` counters since the last delta-snapshot blob was written — absent on threads that don't use `DeltaChannel`).
- The `config` field of a snapshot contains `configurable.checkpoint_id` and `configurable.thread_id`; passing this config to `graph.invoke(None, config=snapshot.config)` replays from that exact checkpoint.
- `next` being an empty tuple signals the thread has reached `END` — the terminal snapshot. `tasks` at a terminal snapshot will be empty or contain the last successful task results.

### Example 1 — inspect state history and find the last interrupted step

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    value: int
    approved: bool


def process(state: State) -> dict:
    approved = interrupt({"value": state["value"], "prompt": "Approve?"})
    return {"approved": approved}


builder = StateGraph(State)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "demo"}}
graph.invoke({"value": 42, "approved": False}, config=cfg)

# Walk history
for snap in graph.get_state_history(cfg):
    meta = snap.metadata or {}
    print(
        f"step={meta.get('step')} "
        f"next={snap.next} "
        f"interrupts={len(snap.interrupts)} "
        f"source={meta.get('source')}"
    )
```

### Example 2 — fork a thread by rewinding to a prior snapshot

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    messages: list[str]


def append_a(state: State) -> dict:
    return {"messages": state["messages"] + ["A"]}


def append_b(state: State) -> dict:
    return {"messages": state["messages"] + ["B"]}


builder = StateGraph(State)
builder.add_node("a", append_a)
builder.add_node("b", append_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "fork-demo"}}
graph.invoke({"messages": []}, config=cfg)

# Get the snapshot just before node "b" ran
snapshots = list(graph.get_state_history(cfg))
before_b = next(s for s in snapshots if "b" in s.next)
meta = before_b.metadata or {}
print(f"Forking from step {meta.get('step')}, next={before_b.next}")

# Fork: update_state() returns the config for the new checkpoint
fork_config = graph.update_state(before_b.config, {"messages": ["FORKED"]})
forked_result = graph.invoke(None, config=fork_config)
print(forked_result["messages"])  # ['FORKED', 'B']
```

### Example 3 — read `CheckpointMetadata` to audit checkpoint sources

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    n: int


def step(state: State) -> dict:
    return {"n": state["n"] + 1}


builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "meta-demo"}}
graph.invoke({"n": 0}, config=cfg)

# Manually update state — creates a "update" source checkpoint
graph.update_state(cfg, {"n": 99})

for snap in graph.get_state_history(cfg):
    meta = snap.metadata or {}
    print(
        f"source={meta.get('source','?')!r:8s}  "
        f"step={meta.get('step','?')}  "
        f"values={snap.values}"
    )
# source='update'   step=2  values={'n': 99}
# source='loop'     step=1  values={'n': 1}
# source='input'    step=-1 values={'n': 0}
```

---

## 3 · `PregelExecutableTask`

**Module:** `langgraph.types`

`PregelExecutableTask` is the runtime representation of a single scheduled node invocation. Understanding it unlocks precise inspection of what the graph *is about to run* at any superstep.

**Key source facts** (from `langgraph/types.py`):

- `PregelExecutableTask` is a `@dataclass` with fields: `name: str` (node name), `input: Any` (the value passed to the node's `proc`), `proc: Runnable` (the actual runnable being invoked), `writes: deque[tuple[str, Any]]` (output channel writes accumulated during the task — read after the task completes to see what was written), `config: RunnableConfig` (the task's own config, including `configurable.checkpoint_ns`, `configurable.langgraph_task_id`, etc.), `triggers: Sequence[str]` (channel names that activated this task), `retry_policy: Sequence[RetryPolicy]`, `cache_key: CacheKey | None`, `id: str` (stable UUID for this task execution — matches `PregelTask.id` in `StateSnapshot.tasks`), `path: tuple[str | int | tuple, ...]` (hierarchical path including subgraph namespace), `writers: Sequence[Runnable]` (post-node output processors), `subgraphs: Sequence[PregelProtocol]` (embedded subgraph instances reachable from this task), and `timeout: TimeoutPolicy | None`.
- `PregelExecutableTask` appears in `StateSnapshot.tasks` as lighter `PregelTask` objects after execution. During execution it lives in the `PregelRunner`'s task list.
- The `writes` deque starts empty; each channel write appended by the node (via `ChannelWrite`) lands here before being applied to the channel graph. Inspect it in a custom `ChannelWrite` wrapper to audit output before it persists.
- `triggers` identifies which channels fired this task. For a PULL task the trigger is the subscribed channel; for a PUSH (`Send`) task it is the `TASKS` channel.

### Example 1 — inspect tasks from a `StateSnapshot`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    x: int


def node_a(state: State) -> dict:
    return {"x": state["x"] * 2}


def node_b(state: State) -> dict:
    return {"x": state["x"] + 10}


builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "task-demo"}}
graph.invoke({"x": 5}, config=cfg)

# Inspect tasks in each historical snapshot
for snap in graph.get_state_history(cfg):
    if snap.tasks:
        for t in snap.tasks:
            print(f"task={t.name!r}  id={t.id[:8]}…  error={t.error}")
```

### Example 2 — use task `id` to correlate snapshot tasks with debug events

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    data: str
    result: str


def validate(state: State) -> dict:
    if not state["data"]:
        raise ValueError("data must not be empty")
    return {"result": f"processed: {state['data']}"}


builder = StateGraph(State)
builder.add_node("validate", validate)
builder.add_edge(START, "validate")
builder.add_edge("validate", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "err-demo"}}

try:
    graph.invoke({"data": "", "result": ""}, config=cfg)
except Exception:
    pass

# The failing snapshot carries the error on its task
for snap in graph.get_state_history(cfg):
    for t in snap.tasks:
        if t.error:
            print(f"Failed task: {t.name!r}, error: {t.error!r}")
            print(f"Task id: {t.id}")
```

### Example 3 — read task `path` to distinguish subgraph tasks from root tasks

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver


class RootState(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]


class ItemState(TypedDict):
    item: str


def fan_out(state: RootState) -> list[Send]:
    return [Send("process", {"item": i}) for i in state["items"]]


def process(state: ItemState) -> dict:
    return {"results": [state["item"].upper()]}


builder = StateGraph(RootState)
builder.add_node("process", process)
builder.add_conditional_edges(START, fan_out)
builder.add_edge("process", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

cfg = {"configurable": {"thread_id": "path-demo"}}
result = graph.invoke({"items": ["a", "b", "c"], "results": []}, config=cfg)
print(result["results"])  # ['A', 'B', 'C']

# Snapshot tasks show individual Send-task paths
for snap in graph.get_state_history(cfg):
    for t in snap.tasks:
        print(f"task={t.name!r}  path={t.path}")
```

---

## 4 · `create_react_agent` · `tools_condition`

**Module:** `langgraph.prebuilt.chat_agent_executor`, `langgraph.prebuilt.tool_node`

`create_react_agent` is the fastest way to build a tool-calling ReAct loop; `tools_condition` is the conditional edge function at its heart.

> **Deprecation note:** `create_react_agent` is deprecated in `langgraph>=1.0.0` in favour of [`create_agent`](https://python.langchain.com/docs/concepts/agents) from `langchain`. The function remains fully functional in 1.2.9 and the examples below work as-is, but new projects should prefer `langchain.agents.create_agent`.

**Key source facts** (from `langgraph/prebuilt/chat_agent_executor.py` and `tool_node.py`):

- `create_react_agent(model, tools, *, prompt=None, response_format=None, pre_model_hook=None, post_model_hook=None, state_schema=None, context_schema=None, checkpointer=None, store=None, interrupt_before=None, interrupt_after=None, debug=False, version="v2", name=None)` — compiles a `StateGraph` with `"agent"` and `"tools"` nodes connected by `tools_condition`. The `prompt` kwarg accepts a plain string (becomes `SystemMessage`), a `SystemMessage`, a callable `(state) -> PromptValue`, or a `Runnable`.
- The `response_format` kwarg activates a structured-output node; when set, the compiled graph adds `"generate_structured_response"` after the agent decides to stop, and the output lands in `state["structured_response"]`.
- `pre_model_hook` and `post_model_hook` are `Runnable | Callable` objects inserted before/after the LLM call respectively; they receive and return the full state dict.
- `tools_condition(state, messages_key="messages")` inspects the last message: if it has non-empty `.tool_calls`, returns `"tools"`; otherwise returns `"__end__"`. Handles three state shapes: a plain `list` (takes `state[-1]`), a `dict` (reads `state[messages_key][-1]`), or a Pydantic `BaseModel` (reads `getattr(state, messages_key)[-1]`).
- The compiled graph has `.name` set to the `name` kwarg (defaulting to `"LangGraph"`), which appears in LangSmith traces. Pass `name="my-agent"` to distinguish multiple agents in the same project.

### Example 1 — minimal ReAct agent with a custom tool

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# A simple pure-Python tool — no external API needed
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# Swap in any chat model that supports tool calling
from langchain_openai import ChatOpenAI  # requires OPENAI_API_KEY
model = ChatOpenAI(model="gpt-4o-mini")

agent = create_react_agent(model, tools=[add, multiply])

result = agent.invoke({"messages": [HumanMessage("What is (3 + 4) * 5?")]})
print(result["messages"][-1].content)
```

### Example 2 — `tools_condition` on a custom state schema

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


@tool
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny in {city}"


model = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_weather])


def call_model(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}  # add_messages reducer appends, not overwrites


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([get_weather]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()
result = graph.invoke({"messages": [HumanMessage("Weather in Paris?")]})
print(result["messages"][-1].content)
```

### Example 3 — `create_react_agent` with `response_format` for structured output

```python
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


class MathResult(BaseModel):
    expression: str
    answer: float
    steps: list[str]


@tool
def calculator(expression: str) -> float:
    """Evaluate a simple arithmetic expression (+ - * / and parentheses only)."""
    import ast as _ast
    _ALLOWED = frozenset({
        _ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Constant,
        _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.UAdd, _ast.USub,
    })
    try:
        tree = _ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc
    for node in _ast.walk(tree):
        if type(node) not in _ALLOWED:
            raise ValueError(f"Disallowed operation: {type(node).__name__}")
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}))


model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(
    model,
    tools=[calculator],
    response_format=MathResult,
)

result = agent.invoke({"messages": [HumanMessage("Compute (12 + 8) * 3.5")]})
structured = result.get("structured_response")
if structured:
    print(f"Answer: {structured.answer}")
    print(f"Steps: {structured.steps}")
```

---

## 5 · `RemoteGraph` · `get_client` · `get_sync_client`

**Module:** `langgraph.pregel.remote`

`RemoteGraph` is a drop-in replacement for a local compiled graph that instead calls a deployed LangGraph Server endpoint — or another graph running in the same process via ASGI loopback.

**Key source facts** (from `langgraph/pregel/remote.py`):

- `RemoteGraph(assistant_id, /, *, url=None, api_key=None, headers=None, client=None, sync_client=None, config=None, name=None, distributed_tracing=False)` — `assistant_id` is the graph ID or assistant UUID on the server. Pass `url` to auto-create both async (`LangGraphClient`) and sync (`SyncLangGraphClient`) HTTP clients; pass `client`/`sync_client` to inject pre-built clients (e.g. for testing with a mock).
- `get_client(*, url=None, api_key=NOT_PROVIDED, headers=None, timeout=None)` — when `url=None` it tries an ASGI in-process loopback (`langgraph_api.server.app`) so you can call graphs hosted in the same process without network overhead. The `api_key=NOT_PROVIDED` sentinel triggers auto-loading from `LANGGRAPH_API_KEY` → `LANGSMITH_API_KEY` → `LANGCHAIN_API_KEY`; pass `api_key=None` explicitly to skip env-var loading. Returns a `LangGraphClient` with sub-clients `.assistants`, `.threads`, `.runs`, `.crons`.
- `get_sync_client(*, url=None, ...)` builds an `httpx.Client` (not `AsyncClient`) — use in scripts or sync contexts where `asyncio` is unavailable.
- `RemoteGraph.invoke()` delegates to `sync_client.runs.wait()`; `ainvoke()` delegates to `client.runs.wait()`. Streaming calls (`stream()`, `astream()`) use `runs.stream()` with the requested `stream_mode`.
- `RemoteGraph` satisfies `PregelProtocol`, so it can be embedded as a node in a parent graph: `parent_builder.add_node("remote_worker", remote_graph)`. The parent handles routing and checkpointing; the remote graph handles its own internal execution.
- `distributed_tracing=True` injects LangSmith `x-parent-*` headers into requests, linking the parent and remote traces in the LangSmith UI.

### Example 1 — connect to a running LangGraph Server and invoke a graph

```python
from langgraph.pregel.remote import RemoteGraph

# Requires a LangGraph Server running locally (e.g. via `langgraph dev`)
# and an assistant/graph with this ID deployed on it.
remote = RemoteGraph(
    "my_agent",           # assistant_id / graph name
    url="http://localhost:8123",
    api_key=None,         # skip env-var lookup for local dev
)

result = remote.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    config={"configurable": {"thread_id": "t1"}},
)
print(result["messages"][-1]["content"])
```

### Example 2 — use `RemoteGraph` as a node inside a parent graph

```python
from langgraph.pregel.remote import RemoteGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict


class State(TypedDict):
    input: str
    output: str


# The remote graph handles a sub-task; the parent orchestrates
remote_worker = RemoteGraph(
    "summariser",
    url="http://localhost:8123",
    name="summariser",  # label shown in parent's get_graph()
)

def prepare(state: State) -> dict:
    return {"input": state["input"].strip()}

builder = StateGraph(State)
builder.add_node("prepare", prepare)
builder.add_node("summariser", remote_worker)  # remote graph as node
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "summariser")
builder.add_edge("summariser", END)

checkpointer = InMemorySaver()
parent = builder.compile(checkpointer=checkpointer)
# parent.invoke({"input": "...", "output": ""}, config={"configurable": {"thread_id": "p1"}})
print("Parent graph compiled with remote node:", parent.get_graph().nodes)
```

### Example 3 — async streaming from a `RemoteGraph`

```python
import asyncio
from langgraph.pregel.remote import RemoteGraph


async def stream_remote() -> None:
    remote = RemoteGraph(
        "chat_agent",
        url="http://localhost:8123",
    )

    async for chunk in remote.astream(
        {"messages": [{"role": "user", "content": "Tell me a joke"}]},
        config={"configurable": {"thread_id": "stream-demo"}},
        stream_mode="messages",
    ):
        # Each chunk is a (message, metadata) pair in messages mode
        msg, meta = chunk
        if hasattr(msg, "content") and msg.content:
            print(msg.content, end="", flush=True)
    print()


# asyncio.run(stream_remote())
```

---

## 6 · `ToolCallTransformer` · `ToolCallStream`

**Modules:** `langgraph.prebuilt._tool_call_transformer` (`ToolCallTransformer`) · `langgraph.prebuilt._tool_call_stream` (`ToolCallStream`)

`ToolCallTransformer` is an opt-in native stream transformer that projects `tools`-mode events into per-tool-call `ToolCallStream` handles, enabling real-time per-tool output streaming.

**Key source facts** (from `langgraph/prebuilt/_tool_call_transformer.py` and `langgraph/prebuilt/_tool_call_stream.py`):

- `ToolCallTransformer` (from `langgraph.prebuilt._tool_call_transformer`) extends `StreamTransformer` with `_native = True` and `required_stream_modes = ("tools",)`. It is **not** a default built-in — you must pass it at compile time: `builder.compile(transformers=[ToolCallTransformer])`.
- The transformer maintains `_active: dict[str, ToolCallStream]` keyed by `tool_call_id`. It dispatches four event types: `"tool-started"` → creates a new `ToolCallStream` and pushes it onto `run.tool_calls`; `"tool-output-delta"` → calls `stream._push_delta(data["delta"])`; `"tool-finished"` → calls `stream._finish(data["output"])`; `"tool-error"` → calls `stream._fail(data["message"])`.
- `ToolCallStream` (from `langgraph.prebuilt._tool_call_stream`) fields: `tool_call_id: str`, `tool_name: str`, `input: dict | None`, `output_deltas: StreamChannel[Any]` (a drainable channel of streamed delta chunks), `output: Any` (set on `tool-finished`), `error: str | None` (set on `tool-error`), `completed: bool`. Iterate via `for delta in stream` (sync) or `async for delta in stream` (async).
- The `_bind_pump` / `_bind_apump` methods are called by `StreamMux.bind_pump` to wire the pump callback so `output_deltas` can be iterated on demand. The active mode (sync or async) is locked at pump-bind time.
- Events on the `tools` channel still pass through (`process` returns `True`) — wire consumers subscribing to the raw `tools` channel still see every event. `ToolCallTransformer` is purely additive.
- `finalize()` closes any still-active streams when the run ends normally; `fail(err)` fails them when the run errors.

### Example 1 — watch tool call lifecycle via `astream_events()`

```python
import asyncio
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import tools_condition
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


@tool
def get_stock_price(ticker: str) -> float:
    """Get the current stock price for a ticker symbol."""
    prices = {"AAPL": 189.5, "GOOG": 172.3, "MSFT": 415.0}
    return prices.get(ticker.upper(), 0.0)


model = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    messages: list


m = model.bind_tools([get_stock_price])


def call_model(state: State) -> dict:
    return {"messages": state["messages"] + [m.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([get_stock_price]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


async def watch_tool_calls() -> None:
    async for event in graph.astream_events(
        {"messages": [HumanMessage("Price of AAPL and MSFT?")]},
        version="v2",
    ):
        if event["event"] == "on_tool_start":
            print(f"Tool started: {event['name']} with {event['data'].get('input')}")
        elif event["event"] == "on_tool_end":
            print(f"Tool finished: {event['name']} → {event['data'].get('output')}")


asyncio.run(watch_tool_calls())
```

### Example 2 — consume `ToolCallStream` deltas via `run.tool_calls`

```python
import asyncio
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langgraph.graph import StateGraph, START, END


@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Top results for '{query}': LangGraph docs, tutorials, examples."


llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([search])


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([search]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# Register transformer: tools events are projected onto run.tool_calls
graph = builder.compile(transformers=[ToolCallTransformer])


async def stream_tool_output() -> None:
    async with graph.astream(
        {"messages": [HumanMessage("Search for LangGraph streaming docs")]},
        stream_mode="tools",
    ) as run:
        async for tc_stream in run.tool_calls:
            print(f"Tool: {tc_stream.tool_name}  id={tc_stream.tool_call_id}")
            async for delta in tc_stream.output_deltas:
                print(f"  delta: {delta!r}")
            print(f"  output: {tc_stream.output!r}")
            print(f"  completed: {tc_stream.completed}")


asyncio.run(stream_tool_output())
```

### Example 3 — handle `ToolCallStream.error` for failed tools

```python
import asyncio
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt._tool_call_transformer import ToolCallTransformer
from langgraph.graph import StateGraph, START, END


@tool
def risky_divide(x: int, y: int) -> float:
    """Divide x by y."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y


llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([risky_divide])


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode([risky_divide], handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(transformers=[ToolCallTransformer])


async def handle_tool_errors() -> None:
    async with graph.astream(
        {"messages": [HumanMessage("Compute 10 divided by 0")]},
        stream_mode="tools",
    ) as run:
        async for tc_stream in run.tool_calls:
            async for _ in tc_stream.output_deltas:  # drain any partial deltas
                pass
            if tc_stream.error:
                print(f"Tool '{tc_stream.tool_name}' failed: {tc_stream.error!r}")
            else:
                print(f"Result: {tc_stream.output}")


asyncio.run(handle_tool_errors())
```

---

## 7 · `DebugTransformer` · `TasksTransformer`

**Module:** `langgraph.stream.transformers`

`DebugTransformer` and `TasksTransformer` are native stream transformers that project `stream_mode="debug"` and `stream_mode="tasks"` events onto typed `StreamChannel` attributes (`run.debug` / `run.tasks`) on the v3 run handle when registered via `compile(transformers=[DebugTransformer])`. The examples below demonstrate the equivalent **raw `stream_mode` API** — which works without registering the transformers and is the most common usage pattern; the transformer attributes are available on the run handle for advanced v3 pump-based consumers.

**Key source facts** (from `langgraph/stream/transformers.py`):

- Both classes extend `StreamTransformer` with `_native = True` — meaning the `init()` return dict keys become **direct attributes** on `GraphRunStream` (e.g. `run.debug`, `run.tasks`) rather than going through the extension protocol.
- `DebugTransformer(scope=())` captures events where `event["method"] == "debug"` and `params["namespace"] == list(scope)`. Each matching event's `params["data"]` — a dict with `type` (`"checkpoint"`, `"task"`, or `"task_result"`) and `step`/`timestamp`/`payload` fields — is pushed onto `self._log: StreamChannel[dict[str, Any]]`, exposed as `run.debug`.
- `TasksTransformer(scope=())` works identically but filters on `event["method"] == "tasks"`. Each task payload (a dict with `id`/`name`/`result`/`interrupts` etc.) is pushed onto `run.tasks`.
- Both transformers are **scope-filtered**: only events at the run's own namespace reach the log. Events from deeper subgraphs appear on the respective subgraph handle's `.debug` / `.tasks` projection (populated by `SubgraphTransformer`'s child mini-mux).
- `required_stream_modes = ("debug",)` / `("tasks",)` means including either transformer in `compile(transformers=[...])` automatically activates the corresponding stream mode — you don't need to pass `stream_mode=["debug"]` separately.

### Example 1 — stream debug events to inspect checkpoint snapshots

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    count: int


def tick(state: State) -> dict:
    return {"count": state["count"] + 1}


builder = StateGraph(State)
builder.add_node("tick", tick)
builder.add_edge(START, "tick")
builder.add_edge("tick", END)

checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
)


async def watch_debug() -> None:
    async for data in graph.astream(
        {"count": 0},
        config={"configurable": {"thread_id": "debug-demo"}},
        stream_mode="debug",
    ):
        print(f"debug event: type={data.get('type')!r}  step={data.get('step')}")


asyncio.run(watch_debug())
```

### Example 2 — stream task payloads directly with `stream_mode='tasks'`

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    value: int


def double(state: State) -> dict:
    return {"value": state["value"] * 2}


def add_one(state: State) -> dict:
    return {"value": state["value"] + 1}


builder = StateGraph(State)
builder.add_node("double", double)
builder.add_node("add_one", add_one)
builder.add_edge(START, "double")
builder.add_edge("double", "add_one")
builder.add_edge("add_one", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


async def watch_tasks() -> None:
    async for data in graph.astream(
        {"value": 3},
        config={"configurable": {"thread_id": "tasks-demo"}},
        stream_mode="tasks",
    ):
        task_id = data.get("id", "?")[:8]
        name = data.get("name", "?")
        if "result" in data:
            print(f"  RESULT  task={name!r}  id={task_id}…  writes={data['result']}")
        else:
            print(f"  STARTED task={name!r}  id={task_id}…")


asyncio.run(watch_tasks())
```

### Example 3 — combine `debug` and `tasks` stream modes

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    x: int


def square(state: State) -> dict:
    return {"x": state["x"] ** 2}


builder = StateGraph(State)
builder.add_node("square", square)
builder.add_edge(START, "square")
builder.add_edge("square", END)

graph = builder.compile(checkpointer=InMemorySaver())


async def combined_stream() -> None:
    async for mode, data in graph.astream(
        {"x": 4},
        config={"configurable": {"thread_id": "combo"}},
        stream_mode=["debug", "tasks", "values"],
    ):
        if mode == "debug":
            print(f"[debug] type={data.get('type')}")
        elif mode == "tasks":
            print(f"[tasks] name={data.get('name')!r} has_result={'result' in data}")
        elif mode == "values":
            print(f"[values] x={data.get('x')}")


asyncio.run(combined_stream())
```

---

## 8 · `SubgraphTransformer`

**Module:** `langgraph.stream.transformers`

`SubgraphTransformer` is the native transformer that discovers direct-child subgraph invocations and creates in-process navigation handles (`SubgraphRunStream` / `AsyncSubgraphRunStream`) for drilling into their projections.

**Key source facts** (from `langgraph/stream/transformers.py`):

- `SubgraphTransformer` extends `_TasksLifecycleBase` (a template-method base tracking which subgraph namespaces are `"open"`). `_native = True` exposes the result on `run.subgraphs` as a `StreamChannel[SubgraphRunStream | AsyncSubgraphRunStream]`.
- On each `"tasks"` event the transformer calls `_handle_task_start()` to detect the namespace boundary where a subgraph begins. When a subgraph starts (`lifecycle.event == "started"`), `_on_started(ns, graph_name, trigger_call_id)` calls `self._mux._make_child(ns)` to clone a scoped mini-mux, then constructs a `SubgraphRunStream` (or `AsyncSubgraphRunStream`) wrapping it and pushes the handle onto `self._log`.
- **Direct children only**: `_should_track(ns)` checks `len(ns) == len(self.scope) + 1`. Grandchildren are tracked by the child mini-mux's own `SubgraphTransformer`, so they appear on `child_handle.subgraphs` — not on the root's.
- After construction, every subsequent event that starts with the child namespace is forwarded to `handle._mux.push(event)` (sync) or `handle._mux.apush(event)` (async), keeping the child's projections populated in real time.
- Terminal events (`status == "completed" | "failed" | "interrupted" | "drained"`) call `_close_or_fail_handle` which closes or fails the child mini-mux, draining any iterators waiting on it.
- `finalize()` / `afinalize()` close any open handles that never received a terminal event (e.g. if the run was cut short). This ensures iterators on child handles don't block forever.

### Example 1 — trace subgraph namespace events via `subgraphs=True`

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    value: int


def multiply(state: State) -> dict:
    return {"value": state["value"] * 3}


inner_builder = StateGraph(State)
inner_builder.add_node("multiply", multiply)
inner_builder.add_edge(START, "multiply")
inner_builder.add_edge("multiply", END)
inner_graph = inner_builder.compile()

outer_builder = StateGraph(State)
outer_builder.add_node("inner", inner_graph)  # wire compiled graph directly as a node
outer_builder.add_edge(START, "inner")
outer_builder.add_edge("inner", END)

graph = outer_builder.compile(checkpointer=InMemorySaver())

for ns, chunk in graph.stream(
    {"value": 7},
    config={"configurable": {"thread_id": "sub-demo"}},
    stream_mode="values",
    subgraphs=True,
):
    print(f"namespace={ns}  data={chunk}")
```

### Example 2 — async subgraph streaming with `subgraphs=True`

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    counter: int


def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}


sub_builder = StateGraph(State)
sub_builder.add_node("inc", increment)
sub_builder.add_edge(START, "inc")
sub_builder.add_edge("inc", END)
sub_graph = sub_builder.compile()


main_builder = StateGraph(State)
main_builder.add_node("sub", sub_graph)  # wire compiled graph directly as a node
main_builder.add_edge(START, "sub")
main_builder.add_edge("sub", END)

graph = main_builder.compile(checkpointer=InMemorySaver())


async def async_subgraph_stream() -> None:
    async for ns, chunk in graph.astream(
        {"counter": 0},
        config={"configurable": {"thread_id": "async-sub"}},
        stream_mode="values",
        subgraphs=True,
    ):
        depth = len(ns)
        indent = "  " * depth
        print(f"{indent}[ns={ns}] counter={chunk.get('counter')}")


asyncio.run(async_subgraph_stream())
```

### Example 3 — use `subgraphs=True` with `stream_mode="updates"` to trace writes per namespace

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    n: int
    total: int


def add_ten(state: State) -> dict:
    return {"n": state["n"] + 10}


inner_builder = StateGraph(State)
inner_builder.add_node("add", add_ten)
inner_builder.add_edge(START, "add")
inner_builder.add_edge("add", END)
inner_graph = inner_builder.compile()


def accumulate(state: State) -> dict:
    return {"total": state["total"] + state["n"]}


outer_builder = StateGraph(State)
outer_builder.add_node("inner", inner_graph)  # compiled subgraph wired as a node
outer_builder.add_node("acc", accumulate)
outer_builder.add_edge(START, "inner")
outer_builder.add_edge("inner", "acc")
outer_builder.add_edge("acc", END)

graph = outer_builder.compile(checkpointer=InMemorySaver())


async def trace_updates() -> None:
    async for ns, update in graph.astream(
        {"n": 5, "total": 0},
        config={"configurable": {"thread_id": "trace"}},
        stream_mode="updates",
        subgraphs=True,
    ):
        print(f"ns={ns!r}  update={update}")


asyncio.run(trace_updates())
```

---

## 9 · `LifecyclePayload` · `GraphDrained`

**Modules:** `langgraph.stream.transformers` (`LifecyclePayload`) · `langgraph.errors` (`GraphDrained`)

`LifecyclePayload` is the typed payload pushed onto the `lifecycle` channel by `LifecycleTransformer`; `GraphDrained` is the exception raised when a cooperative drain request is honoured.

**Key source facts** (from `langgraph/stream/transformers.py` and `langgraph/errors.py`):

- `LifecyclePayload` is `TypedDict(total=False)` with fields: `event: SubgraphStatus` (`"started" | "completed" | "failed" | "interrupted" | "drained"`), `namespace: list[str]` (the subgraph's namespace path), `graph_name: NotRequired[str]`, `trigger_call_id: NotRequired[str]` (the `@task` call ID that spawned this subgraph, if any), `cause: NotRequired[LifecycleCause]` (`"node_step"` or `"@task"`), and `error: NotRequired[str]` (error message on `"failed"` events). All fields are optional (`total=False`) so the dict may be sparse.
- `LifecycleTransformer` pushes a `LifecyclePayload` onto `run.lifecycle` only for **child** namespaces — namespaces strictly nested below the transformer's own scope (`len(ns) > depth`). The root namespace `[]` is not tracked. Consumers filter by `payload["namespace"]` to identify which subgraph emitted an event.
- `LifecycleTransformer` is **built into `stream_events(version="v3")`** and its async counterpart by default. Consume lifecycle events with `with graph.stream_events(..., version="v3") as run: for payload in run.lifecycle:` (sync) or `async with await graph.astream_events(..., version="v3") as run: async for payload in run.lifecycle:` (async). You do **not** need to pass `transformers=[LifecycleTransformer]` to `compile()` — it is already wired into the v3 mux. `"lifecycle"` is not a recognised `stream_mode` string for `astream()` / `stream()`.
- Lifecycle events are emitted for nested compiled graph invocations (either wired as a direct node or invoked via `inner_graph.invoke()` inside a node function). Plain `Send` fan-out to regular function nodes does **not** produce lifecycle payloads — those tasks execute in a parallel branch but are not nested graph namespaces.
- `GraphDrained(reason="shutdown")` is a `GraphBubbleUp` subclass — a sentinel exception in the same family as `GraphInterrupt` that Pregel catches and converts into a graceful exit. The graph saves a checkpoint at the last superstep boundary and raises `GraphDrained` to the caller; callers should catch it and may resume the thread later with `graph.invoke(None, config=cfg)`.
- `RunControl.request_drain(reason="...")` is the trigger for `GraphDrained`. Call it from a `SIGTERM` handler or any background thread — it is thread-safe (no locks needed — it only sets a `bool` attribute on the `RunControl` dataclass, which Pregel polls at each superstep boundary).
- `LifecyclePayload` events flow through the same `lifecycle` channel regardless of whether the subgraph is local or a `RemoteGraph`; remote SDK clients receive identical payloads via the wire protocol.

### Example 1 — watch lifecycle events for a graph with a subgraph

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    n: int


def step(state: State) -> dict:
    return {"n": state["n"] + 1}


# Inner subgraph wired directly as a node so it gets its own namespace
inner = StateGraph(State)
inner.add_node("step", step)
inner.add_edge(START, "step")
inner.add_edge("step", END)
inner_graph = inner.compile()

outer = StateGraph(State)
outer.add_node("inner", inner_graph)  # compiled graph as node → creates child namespace
outer.add_edge(START, "inner")
outer.add_edge("inner", END)

# LifecycleTransformer is hardcoded in the v3 mux — no compile(transformers=[...]) needed
graph = outer.compile(checkpointer=InMemorySaver())


async def watch_lifecycle() -> None:
    # astream_events(version="v3") returns AsyncGraphRunStream; run.lifecycle yields LifecyclePayload
    async with await graph.astream_events(
        {"n": 0},
        config={"configurable": {"thread_id": "lc-demo"}},
        version="v3",
    ) as run:
        async for payload in run.lifecycle:
            event = payload.get("event", "?")
            ns = payload.get("namespace", [])
            graph_name = payload.get("graph_name", "root")
            print(f"  [{event:12s}] ns={ns}  graph={graph_name!r}")
        _ = await run.output


asyncio.run(watch_lifecycle())
```

### Example 2 — observe drain state inside a long-running node

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import get_runtime
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    step_count: int


def long_step(state: State) -> dict:
    runtime = get_runtime()
    if runtime is not None and runtime.drain_requested:
        print(f"  Drain requested ({runtime.drain_reason}), finishing current step")
    return {"step_count": state["step_count"] + 1}


builder = StateGraph(State)
builder.add_node("work", long_step)
builder.add_edge(START, "work")
builder.add_edge("work", END)

graph = builder.compile(checkpointer=InMemorySaver())

result = graph.invoke(
    {"step_count": 0},
    config={"configurable": {"thread_id": "drain-demo"}},
)
print(f"Completed with step_count={result['step_count']}")
```

### Example 3 — filter `LifecyclePayload` by event type

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class S(TypedDict):
    value: int


def double(state: S) -> dict:
    return {"value": state["value"] * 2}


# Compiled inner graph — invoking it inside a node creates a child lifecycle namespace
inner_builder = StateGraph(S)
inner_builder.add_node("double", double)
inner_builder.add_edge(START, "double")
inner_builder.add_edge("double", END)
inner_graph = inner_builder.compile()


def run_inner(state: S) -> dict:
    return inner_graph.invoke({"value": state["value"]})


outer_builder = StateGraph(S)
outer_builder.add_node("sub", run_inner)
outer_builder.add_edge(START, "sub")
outer_builder.add_edge("sub", END)

graph = outer_builder.compile(checkpointer=InMemorySaver())


async def count_events() -> None:
    started = 0
    completed = 0

    async with await graph.astream_events(
        {"value": 3},
        config={"configurable": {"thread_id": "count"}},
        version="v3",
    ) as run:
        async for payload in run.lifecycle:
            event = payload.get("event")
            if event == "started":
                started += 1
            elif event == "completed":
                completed += 1
        _ = await run.output

    print(f"Lifecycle events: started={started}  completed={completed}")
    # started=1  completed=1


asyncio.run(count_events())
```

---

## 10 · `RunnableCallable`

**Module:** `langgraph.utils.runnable`

`RunnableCallable` is LangGraph's lightweight alternative to LangChain's `RunnableLambda`. It wraps a plain Python function (or a sync + async pair) into a `Runnable` while adding automatic injection of LangGraph-specific kwargs.

**Key source facts** (from `langgraph/utils/runnable.py`):

- `RunnableCallable(func, afunc=None, *, name=None, tags=None, trace=True, recurse=True, explode_args=False, **kwargs)` — at least one of `func` / `afunc` must be provided. `name` defaults to `func.__name__` (skipped for lambdas). Extra `**kwargs` are stored and merged into the invocation call.
- **Auto-injection**: the constructor inspects the function signature against `KWARGS_CONFIG_KEYS` — a list of `(kwarg_name, accepted_types, runtime_key, default)` tuples. Matching parameters are stored in `self.func_accepts` and injected at call time from `config[runtime_key]`. This covers: `config` (injects `RunnableConfig`), `store` (injects the graph's `BaseStore`), `writer` (injects the `get_stream_writer()` callable), and `previous` (injects the previous checkpoint value for `@entrypoint` graphs).
- `explode_args=True` means the function receives the `input` dict unpacked as keyword arguments: `func(**input)` instead of `func(input)`. This is the pattern used internally by `ToolNode` for tool functions.
- `trace=True` (default) wraps the call in a LangSmith trace span; set `trace=False` for internal utility functions you don't want in traces.
- `recurse=True` (default) means if the function returns another `Runnable`, it is invoked recursively. Set `recurse=False` to return the runnable itself rather than chasing it.
- `invoke()` calls `func(input, **injected_kwargs)`; `ainvoke()` calls `afunc(input, **injected_kwargs)` if present, otherwise falls back to calling `invoke()` synchronously — this blocks the event loop, so always supply an explicit `afunc` for non-blocking async behaviour.

### Example 1 — wrap a sync/async function pair for a node

```python
import asyncio
from typing_extensions import TypedDict
from langgraph.utils.runnable import RunnableCallable
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    value: int


def sync_double(state: State) -> dict:
    return {"value": state["value"] * 2}


async def async_double(state: State) -> dict:
    await asyncio.sleep(0)  # simulate async I/O
    return {"value": state["value"] * 2}


# Wrap into a RunnableCallable with both sync and async implementations
node = RunnableCallable(sync_double, async_double, name="double")

builder = StateGraph(State)
builder.add_node("double", node)
builder.add_edge(START, "double")
builder.add_edge("double", END)

graph = builder.compile()

# Sync invoke uses sync_double
print(graph.invoke({"value": 5}))  # {'value': 10}

# Async invoke uses async_double
print(asyncio.run(graph.ainvoke({"value": 7})))  # {'value': 14}
```

### Example 2 — `explode_args=True` for dict-unpacking node functions

```python
from typing_extensions import TypedDict
from langgraph.utils.runnable import RunnableCallable
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    x: int
    y: int
    result: int


def add(x: int, y: int, **_: object) -> dict:
    """Receives state fields unpacked as kwargs; extra state keys ignored via **_."""
    return {"result": x + y}


# explode_args=True calls func(**state_dict); function must accept all state keys
node = RunnableCallable(add, name="add", explode_args=True)

builder = StateGraph(State)
builder.add_node("add", node)
builder.add_edge(START, "add")
builder.add_edge("add", END)

graph = builder.compile()
print(graph.invoke({"x": 3, "y": 7, "result": 0}))  # {'x': 3, 'y': 7, 'result': 10}
```

### Example 3 — `trace=False` for internal utility nodes that should not appear in LangSmith

```python
from typing_extensions import TypedDict
from langgraph.utils.runnable import RunnableCallable
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    raw: str
    cleaned: str
    processed: str


def _clean(state: State) -> dict:
    """Internal preprocessing — strip whitespace, lower-case."""
    return {"cleaned": state["raw"].strip().lower()}


def process(state: State) -> dict:
    return {"processed": f"result: {state['cleaned']}"}


# trace=False keeps _clean out of LangSmith spans
clean_node = RunnableCallable(_clean, name="clean", trace=False)
process_node = RunnableCallable(process, name="process")

builder = StateGraph(State)
builder.add_node("clean", clean_node)
builder.add_node("process", process_node)
builder.add_edge(START, "clean")
builder.add_edge("clean", "process")
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({"raw": "  Hello World  ", "cleaned": "", "processed": ""})
print(result["processed"])  # result: hello world
```

---

## Quick-reference summary

| Class / function | Module | Key use |
|---|---|---|
| `Command` | `langgraph.types` | Atomically update state + route to node(s) / resume interrupt |
| `Send` | `langgraph.types` | Push a task to a named node with custom input and optional timeout |
| `StateSnapshot` | `langgraph.types` | NamedTuple returned by `get_state()` — values, next, tasks, interrupts |
| `CheckpointMetadata` | `langgraph.types` | TypedDict on each snapshot — source, step, parents, run_id |
| `PregelExecutableTask` | `langgraph.types` | Runtime task dataclass — name, input, writes, triggers, path, timeout |
| `create_react_agent` | `langgraph.prebuilt` | Build a ReAct tool-calling loop (deprecated; prefer `langchain.agents`) |
| `tools_condition` | `langgraph.prebuilt.tool_node` | Conditional edge — `"tools"` if AIMessage has tool calls, else `"__end__"` |
| `RemoteGraph` | `langgraph.pregel.remote` | Drop-in `PregelProtocol` that calls a LangGraph Server deployment |
| `get_client` | `langgraph.pregel.remote` | Create async `LangGraphClient` (supports ASGI loopback) |
| `get_sync_client` | `langgraph.pregel.remote` | Create sync `SyncLangGraphClient` for script contexts |
| `ToolCallTransformer` | `langgraph.prebuilt._tool_call_transformer` | Opt-in transformer projecting `tools` events onto `run.tool_calls` |
| `ToolCallStream` | `langgraph.prebuilt._tool_call_stream` | Per-tool handle with `output_deltas`, `output`, `error`, `completed` |
| `DebugTransformer` | `langgraph.stream.transformers` | Native transformer surfacing `stream_mode="debug"` on `run.debug` |
| `TasksTransformer` | `langgraph.stream.transformers` | Native transformer surfacing `stream_mode="tasks"` on `run.tasks` |
| `SubgraphTransformer` | `langgraph.stream.transformers` | Discovers direct-child subgraphs and pushes navigation handles onto `run.subgraphs` |
| `LifecyclePayload` | `langgraph.stream.transformers` | TypedDict payload on `run.lifecycle` — event/namespace/graph_name/cause/error |
| `GraphDrained` | `langgraph.errors` | Raised on cooperative drain; graph checkpoints and exits gracefully |
| `RunnableCallable` | `langgraph.utils.runnable` | Lightweight callable node — sync/async pair, auto-inject config/store/writer |
